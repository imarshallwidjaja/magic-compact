import type {
  Agent,
  PermissionRuleset,
  Session,
  TextPart,
} from "@opencode-ai/sdk/v2";
import { randomBytes } from "node:crypto";
import { unwrap, type V2Client } from "../api";
import { countGptTokens } from "../stats/tokenize";
import {
  attemptWithinDeadline,
  CONTROL_OPERATION_TIMEOUT_MS,
  isRecord,
} from "../util";
import {
  MAGIC_COMPACT_SUMMARIZER_AGENT,
  boundaryPartID,
  summaryEnvelopePartID,
  summaryEnvelopeTextPart,
  turnHasPersistedSummary,
} from "./constants";
import {
  computeNativeBatchBudget,
  estimateNativeTurnCost,
  NATIVE_BATCH_MAX_CONCURRENCY,
  NATIVE_BATCH_PLANNING_NONCE,
  planNativeGenerationBatches,
  toolAttachmentsModelVisible,
  type NativeBatchProtocol,
  type NativeBatchBudget,
  type NativeGenerationBatch,
} from "./batch";
import { createMediaManifest, type MediaManifest } from "./media";
import {
  assertForkMessageShape,
  buildTurns,
  type CompactionPlan,
  type MessageWithParts,
  type Turn,
} from "./plan";

const GENERATION_PERMISSION: PermissionRuleset = [
  { permission: "*", pattern: "*", action: "deny" },
];
const TARGET_ANCHOR_CHARACTER_LIMIT = 300;
const GLOBAL_ANCHOR_CHARACTER_LIMIT = 160;
const GENERATION_NONCE_ATTEMPTS = 8;
const GENERATION_NONCE_PATTERN = /^[A-Za-z0-9_-]{22}$/u;

type CompactionSettings = {
  model: { providerID: string; modelID: string };
  variant?: string;
  descriptor: {
    limit: { context: number; input?: number; output?: number };
    capabilities?: { input?: { image?: boolean } };
  };
};

type GenerationCandidate = {
  planIndex: number;
  sourceTurnIndex: number;
  ordinal: number;
  anchor: string;
  marker: string;
  expectedInlineImages: number;
  inputTokens: number;
  outputTokens: number;
  mediaManifest?: MediaManifest;
};

type GenerationBatchPlan = NativeGenerationBatch<GenerationCandidate> & {
  retainedMessageIndexes: number[];
  targetUserMessageIndexes: number[];
  followingHalo?: { turn: Turn; userMessageIndex: number };
};

type GenerationBatch = GenerationBatchPlan & {
  generationNonce: string;
  stopMarker: string;
  prompt: string;
};

type ForkMessageIdentity = {
  id: string;
  sessionID: string | undefined;
  role: "user" | "assistant";
  parentID: string | undefined;
  parts: Array<{
    id: string;
    sessionID: string;
    messageID: string;
    type: string;
    tailStartID: string | undefined;
  }>;
};

type GenerationTurnResult = {
  summary: string;
  visualDescriptions: Array<{ ordinal: number; description: string }>;
};

type SummaryGenerationOptions = {
  controlTimeoutMs?: number;
  nonceFactory?: () => string;
};

type GenerationRoute = {
  directory: string;
  workspace?: string;
};

type GenerationStage =
  | "fork"
  | "permission"
  | "history"
  | "isolation"
  | "marker"
  | "provider"
  | "validation"
  | "abort"
  | "delete";

export type GeneratedSummary = {
  text: string;
  mediaManifest?: MediaManifest;
} | null;

export async function generateCompactionSummaries(
  v2: V2Client,
  session: Session,
  plan: CompactionPlan,
  options: SummaryGenerationOptions = {},
): Promise<GeneratedSummary[]> {
  const controlTimeoutMs =
    options.controlTimeoutMs ?? CONTROL_OPERATION_TIMEOUT_MS;
  if (!Number.isFinite(controlTimeoutMs) || controlTimeoutMs <= 0) {
    throw new Error("Compaction generation control deadline must be positive.");
  }

  assertGeneratedPartIDsAvailable(plan);
  const globalTurnIndexPrompt = buildGlobalTurnIndexPrompt(plan);
  const generated: GeneratedSummary[] = plan.summarizedTurns.map(() => null);
  const candidates = plan.summarizedTurns.flatMap((turn, planIndex) => {
    if (turnHasPersistedSummary(turn)) return [];
    const mediaManifest = createMediaManifest(turn);
    const cost = estimateNativeTurnCost(turn);
    return [
      {
        planIndex,
        sourceTurnIndex: plan.sourceTurnStartIndex + planIndex,
        ordinal: 0,
        anchor: firstOrdinaryUserLine(turn, TARGET_ANCHOR_CHARACTER_LIMIT),
        marker: "",
        expectedInlineImages: mediaManifest?.images.length ?? 0,
        ...cost,
        ...(mediaManifest ? { mediaManifest } : {}),
      },
    ];
  });
  candidates.forEach((candidate, index) => {
    candidate.ordinal = index + 1;
    candidate.marker = generationTurnMarker(
      NATIVE_BATCH_PLANNING_NONCE,
      candidate.ordinal,
    );
  });
  if (candidates.length === 0) return generated;

  const settings = await resolveCompactionSettings(
    v2,
    session,
    historyMayExposeImages(plan.sourceSnapshot),
  );
  const budget = computeNativeBatchBudget(settings.descriptor.limit);
  const protocol: NativeBatchProtocol<GenerationCandidate> = {
    explicitRequestTokens: batchCandidates =>
      estimateGenerationExplicitRequestTokens(
        batchCandidates,
        undefined,
        "",
        globalTurnIndexPrompt,
      ),
    responseProtocolTokens: estimateGenerationResponseProtocolTokens,
  };
  const batchPlans = planNativeGenerationBatches(
    candidates,
    budget,
    protocol,
  ).map(batch =>
    createGenerationBatch(batch, plan, budget, globalTurnIndexPrompt),
  );
  const generationNonces = new Set<string>();
  const batches = batchPlans.map(batch =>
    finalizeGenerationBatch(
      batch,
      plan,
      budget,
      globalTurnIndexPrompt,
      options.nonceFactory ?? createGenerationNonce,
      generationNonces,
    ),
  );
  const retainedMessageIndexes = new Set(
    batches.flatMap(batch => batch.retainedMessageIndexes),
  );
  if (
    historyMayExposeImages(
      plan.sourceSnapshot.filter((_, index) =>
        retainedMessageIndexes.has(index),
      ),
    )
  ) {
    assertConfiguredModelSupportsImages(settings);
  }
  const results = await requestCompactionGenerationBatches(
    v2,
    session,
    settings,
    batches,
    plan,
    controlTimeoutMs,
  );
  for (const [index, candidate] of candidates.entries()) {
    const result = results[index]!;
    generated[candidate.planIndex] = {
      text: frameSummary(result),
      ...(candidate.mediaManifest
        ? { mediaManifest: candidate.mediaManifest }
        : {}),
    };
  }
  return generated;
}

export async function injectSummaries(
  v2: V2Client,
  sessionID: string,
  compactionTurns: Turn[],
  summaries: GeneratedSummary[],
): Promise<TextPart[]> {
  const inserted: TextPart[] = [];
  for (const [index, turn] of compactionTurns.entries()) {
    const summary = summaries[index];
    if (summary === undefined || summary === null) {
      throw new Error("Missing summary for assistant turn.");
    }
    const firstAssistant = turn.assistants[0];
    if (!firstAssistant) throw new Error("Turn missing assistant message.");
    const part = summaryEnvelopeTextPart(
      firstAssistant.info.id,
      sessionID,
      summary.text,
      summary.mediaManifest,
    );
    unwrap(
      await v2.part.update({
        sessionID,
        messageID: firstAssistant.info.id,
        partID: part.id,
        part,
      }),
    );
    inserted.push(part);
  }
  return inserted;
}

function createGenerationBatch(
  batch: NativeGenerationBatch<GenerationCandidate>,
  plan: CompactionPlan,
  budget: NativeBatchBudget,
  globalTurnIndexPrompt: string,
): GenerationBatchPlan {
  const sourceMessages = plan.sourceSnapshot as MessageWithParts[];
  const sourceTurns = buildTurns(sourceMessages);
  const retainedTurnIndexes = new Set(
    batch.candidates.map(candidate => candidate.sourceTurnIndex),
  );
  const firstTurnIndex = batch.candidates[0]!.sourceTurnIndex;
  const lastTurnIndex = batch.candidates.at(-1)!.sourceTurnIndex;
  if (
    [...retainedTurnIndexes].some(turnIndex =>
      turnHasNativeCheckpointArtifact(sourceTurns[turnIndex]),
    )
  ) {
    throw new Error(
      "Compaction generation cannot safely isolate a native checkpoint target.",
    );
  }
  let retainedInput = batch.estimatedInputTokens;
  let estimatedExplicitRequestTokens = batch.estimatedExplicitRequestTokens;
  let inputLimit = batch.inputLimit;
  let followingHalo: GenerationBatchPlan["followingHalo"];
  const followingTurn = sourceTurns[lastTurnIndex + 1];
  if (
    followingTurn
    && !retainedTurnIndexes.has(lastTurnIndex + 1)
    && !turnHasNativeCheckpointArtifact(followingTurn)
  ) {
    const followingCost = estimateNativeTurnCost(followingTurn).inputTokens;
    const stopMarker = generationStopMarker(
      NATIVE_BATCH_PLANNING_NONCE,
      batch.candidates.at(-1)!.ordinal + 1,
    );
    const followingExplicitRequestTokens =
      estimateGenerationExplicitRequestTokens(
        batch.candidates,
        firstOrdinaryUserLine(followingTurn, TARGET_ANCHOR_CHARACTER_LIMIT),
        stopMarker,
        globalTurnIndexPrompt,
      );
    const followingInput =
      retainedInput
      - estimatedExplicitRequestTokens
      + followingExplicitRequestTokens
      + followingCost;
    if (followingInput <= budget.haloInputLimit) {
      const userMessageIndex = sourceMessages.indexOf(followingTurn.user[0]!);
      if (userMessageIndex < 0) {
        throw new Error(
          "Compaction generation halo is missing from source order.",
        );
      }
      retainedInput = followingInput;
      estimatedExplicitRequestTokens = followingExplicitRequestTokens;
      inputLimit = budget.haloInputLimit;
      retainedTurnIndexes.add(lastTurnIndex + 1);
      followingHalo = { turn: followingTurn, userMessageIndex };
    }
  }
  const precedingTurn = sourceTurns[firstTurnIndex - 1];
  if (
    precedingTurn
    && !retainedTurnIndexes.has(firstTurnIndex - 1)
    && !turnHasNativeCheckpointArtifact(precedingTurn)
  ) {
    const precedingCost = estimateNativeTurnCost(precedingTurn).inputTokens;
    if (retainedInput + precedingCost <= budget.haloInputLimit) {
      retainedInput += precedingCost;
      inputLimit = budget.haloInputLimit;
      retainedTurnIndexes.add(firstTurnIndex - 1);
    }
  }

  const retainedMessages = new Set(
    [...retainedTurnIndexes].flatMap(turnIndex => {
      const turn = sourceTurns[turnIndex];
      if (!turn) {
        throw new Error(
          "Compaction generation target turn is missing from source order.",
        );
      }
      return [...turn.user, ...turn.assistants];
    }),
  );
  const retainedMessageIndexes = sourceMessages.flatMap((message, index) =>
    retainedMessages.has(message) ? [index] : [],
  );
  const targetUserMessageIndexes = batch.candidates.map(candidate => {
    const message = sourceTurns[candidate.sourceTurnIndex]?.user[0];
    const index = message ? sourceMessages.indexOf(message) : -1;
    if (index < 0) {
      throw new Error(
        "Compaction generation target message is missing from source order.",
      );
    }
    return index;
  });
  return {
    ...batch,
    estimatedInputTokens: retainedInput,
    estimatedExplicitRequestTokens,
    inputLimit,
    retainedMessageIndexes,
    targetUserMessageIndexes,
    ...(followingHalo ? { followingHalo } : {}),
  };
}

function finalizeGenerationBatch(
  batch: GenerationBatchPlan,
  plan: CompactionPlan,
  budget: NativeBatchBudget,
  globalTurnIndexPrompt: string,
  nonceFactory: () => string,
  generationNonces: Set<string>,
): GenerationBatch {
  const generationNonce = selectGenerationNonce(
    [plan.sourceSnapshot],
    nonceFactory,
    generationNonces,
  );
  const candidates = batch.candidates.map(candidate => ({
    ...candidate,
    marker: generationTurnMarker(generationNonce, candidate.ordinal),
  }));
  const stopMarker = batch.followingHalo
    ? generationStopMarker(generationNonce, candidates.at(-1)!.ordinal + 1)
    : "";
  const nextTurnAnchor = batch.followingHalo
    ? firstOrdinaryUserLine(
        batch.followingHalo.turn,
        TARGET_ANCHOR_CHARACTER_LIMIT,
      )
    : undefined;
  const prompt = buildGenerationPrompt(
    candidates,
    nextTurnAnchor,
    stopMarker,
    globalTurnIndexPrompt,
  );
  const exactExplicitRequestTokens = estimateGenerationExplicitRequestTokens(
    candidates,
    nextTurnAnchor,
    stopMarker,
    globalTurnIndexPrompt,
  );
  const exactInputTokens =
    batch.estimatedInputTokens
    - batch.estimatedExplicitRequestTokens
    + exactExplicitRequestTokens;
  const exactOutputTokens =
    candidates.reduce(
      (total, candidate) => total + Math.ceil(candidate.outputTokens),
      0,
    ) + estimateGenerationResponseProtocolTokens(candidates);
  if (exactInputTokens > batch.estimatedInputTokens) {
    throw new Error(
      "Magic Compact exact generation request exceeds its conservative pre-fork token estimate.",
    );
  }
  if (
    exactInputTokens > batch.inputLimit
    || exactInputTokens > budget.usableInput
  ) {
    throw new Error(
      "Magic Compact exact generation request cannot fit the configured model input. Run /magic-trim first and retry.",
    );
  }
  if (
    exactOutputTokens > batch.estimatedOutputTokens
    || exactOutputTokens > budget.outputTarget
  ) {
    throw new Error(
      batch.candidates.length === 1
        ? `A single compaction turn cannot fit the configured ${budget.outputTarget}-token output target. Configure a compaction model with more output capacity.`
        : `A compaction generation batch cannot fit the configured ${budget.outputTarget}-token output target. Keep more recent turns unchanged and retry.`,
    );
  }
  return {
    ...batch,
    candidates,
    generationNonce,
    stopMarker,
    prompt,
  };
}

function turnHasNativeCheckpointArtifact(turn: Turn | undefined): boolean {
  if (!turn) return false;
  return [...turn.user, ...turn.assistants].some(
    message =>
      (message.info.role === "assistant" && message.info.summary === true)
      || message.parts.some(part => part.type === "compaction"),
  );
}

async function requestCompactionGenerationBatches(
  v2: V2Client,
  sourceSession: Session,
  settings: CompactionSettings,
  batches: GenerationBatch[],
  plan: CompactionPlan,
  controlTimeoutMs: number,
): Promise<GenerationTurnResult[]> {
  const results: Array<GenerationTurnResult[] | undefined> = Array.from({
    length: batches.length,
  });
  const failures: Array<Error | undefined> = Array.from({
    length: batches.length,
  });
  let nextBatchIndex = 0;
  let stopAdmission = false;
  const worker = async () => {
    while (!stopAdmission) {
      const batchIndex = nextBatchIndex;
      nextBatchIndex += 1;
      const batch = batches[batchIndex];
      if (!batch) return;
      try {
        const batchResults = await requestCompactionGeneration(
          v2,
          sourceSession,
          settings,
          batch,
          plan,
          controlTimeoutMs,
          () => {
            stopAdmission = true;
          },
        );
        if (batchResults.length !== batch.candidates.length) {
          throw validationFailure();
        }
        results[batchIndex] = batchResults;
      } catch (error) {
        failures[batchIndex] = sanitizeGenerationFailure(
          sourceSession.id,
          error,
        );
        stopAdmission = true;
      }
    }
  };
  await Promise.all(
    Array.from(
      { length: Math.min(NATIVE_BATCH_MAX_CONCURRENCY, batches.length) },
      worker,
    ),
  );
  const orderedFailures = failures.filter(
    (failure): failure is Error => failure !== undefined,
  );
  if (orderedFailures.length === 1) throw orderedFailures[0];
  if (orderedFailures.length > 1) {
    throw new AggregateError(
      orderedFailures,
      "Multiple compaction generation batches failed.",
      { cause: orderedFailures[0] },
    );
  }

  const candidates = batches.flatMap(batch => batch.candidates);
  const orderedResults = results.flatMap(batchResults => batchResults ?? []);
  if (
    orderedResults.length !== candidates.length
    || candidates.some(
      (candidate, index) =>
        candidate.ordinal !== index + 1 || !orderedResults[index],
    )
  ) {
    throw validationFailure();
  }
  return orderedResults;
}

async function requestCompactionGeneration(
  v2: V2Client,
  sourceSession: Session,
  settings: CompactionSettings,
  batch: GenerationBatch,
  plan: CompactionPlan,
  controlTimeoutMs: number,
  onFailureObserved: () => void,
): Promise<GenerationTurnResult[]> {
  const candidates = batch.candidates;
  const route: GenerationRoute = {
    directory: sourceSession.directory,
    ...(sourceSession.workspaceID
      ? { workspace: sourceSession.workspaceID }
      : {}),
  };
  let fork: Session;
  try {
    fork = unwrapLifecycleResponse(
      await v2.session.fork({ sessionID: sourceSession.id, ...route }),
    );
  } catch (error) {
    onFailureObserved();
    throw lifecycleFailure(sourceSession.id, "fork", "rejection", error);
  }

  let operationError: Error | undefined;
  const recordOperationError = (error: Error): void => {
    operationError = error;
    onFailureObserved();
  };
  let results: GenerationTurnResult[] | undefined;
  const generationNonce = batch.generationNonce;
  const stopMarker = batch.stopMarker;
  let retainedIdentity: ForkMessageIdentity[] | undefined;
  const permissionOutcome = await attemptWithinDeadline(
    signal =>
      v2.session
        .update(
          {
            sessionID: fork.id,
            ...route,
            permission: GENERATION_PERMISSION,
          },
          { signal },
        )
        .then(unwrapLifecycleResponse),
    controlTimeoutMs,
  );
  if (permissionOutcome.kind !== "success") {
    recordOperationError(
      lifecycleFailure(
        sourceSession.id,
        "permission",
        permissionOutcome.kind === "timed-out" ? "timeout" : "rejection",
        permissionOutcome.kind === "failed"
          ? permissionOutcome.error
          : undefined,
      ),
    );
  }

  if (!operationError) {
    const messagesOutcome = await attemptWithinDeadline(
      signal =>
        v2.session
          .messages({ sessionID: fork.id, ...route }, { signal })
          .then(unwrapLifecycleResponse),
      controlTimeoutMs,
    );
    if (messagesOutcome.kind !== "success") {
      recordOperationError(
        lifecycleFailure(
          sourceSession.id,
          "history",
          messagesOutcome.kind === "timed-out" ? "timeout" : "rejection",
          messagesOutcome.kind === "failed" ? messagesOutcome.error : undefined,
        ),
      );
    } else {
      try {
        assertForkMessageShape(messagesOutcome.value, plan);
        assertForkRecordIdentity(messagesOutcome.value, fork.id);
        retainedIdentity = batch.retainedMessageIndexes.map(index =>
          forkMessageIdentity(messagesOutcome.value[index]!),
        );
      } catch (error) {
        recordOperationError(
          lifecycleFailure(sourceSession.id, "history", "rejection", error),
        );
      }

      if (!operationError) {
        const retained = new Set(batch.retainedMessageIndexes);
        for (const [index, message] of messagesOutcome.value.entries()) {
          if (retained.has(index)) continue;
          const isolationOutcome = await attemptWithinDeadline(
            signal =>
              v2.session
                .deleteMessage(
                  {
                    sessionID: fork.id,
                    messageID: message.info.id,
                    ...route,
                  },
                  { signal },
                )
                .then(unwrapLifecycleResponse),
            controlTimeoutMs,
          );
          if (isolationOutcome.kind === "success") continue;
          recordOperationError(
            lifecycleFailure(
              sourceSession.id,
              "isolation",
              isolationOutcome.kind === "timed-out" ? "timeout" : "rejection",
              isolationOutcome.kind === "failed"
                ? isolationOutcome.error
                : undefined,
            ),
          );
          break;
        }
      }

      if (!operationError) {
        const retainedOutcome = await attemptWithinDeadline(
          signal =>
            v2.session
              .messages({ sessionID: fork.id, ...route }, { signal })
              .then(unwrapLifecycleResponse),
          controlTimeoutMs,
        );
        if (retainedOutcome.kind !== "success") {
          recordOperationError(
            lifecycleFailure(
              sourceSession.id,
              "history",
              retainedOutcome.kind === "timed-out" ? "timeout" : "rejection",
              retainedOutcome.kind === "failed"
                ? retainedOutcome.error
                : undefined,
            ),
          );
        } else {
          try {
            if (!retainedIdentity) {
              throw new Error(
                "Compaction generation fork history shape mismatch.",
              );
            }
            assertRetainedForkMessageShape(
              retainedOutcome.value,
              plan,
              batch.retainedMessageIndexes,
            );
            assertForkRecordIdentity(retainedOutcome.value, fork.id);
            assertRetainedForkIdentity(retainedOutcome.value, retainedIdentity);
            if (containsString(retainedOutcome.value, generationNonce)) {
              throw new GenerationMarkerNonceCollisionError();
            }
            const markerTargets = generationMarkerTargets(
              retainedOutcome.value,
              fork.id,
              batch,
              stopMarker,
            );
            for (const target of markerTargets) {
              const markerOutcome = await attemptWithinDeadline(
                signal =>
                  v2.part
                    .update(
                      {
                        sessionID: fork.id,
                        messageID: target.message.info.id,
                        partID: target.part.id,
                        ...route,
                        part: target.part,
                      },
                      { signal },
                    )
                    .then(unwrapLifecycleResponse),
                controlTimeoutMs,
              );
              if (markerOutcome.kind === "success") continue;
              recordOperationError(
                lifecycleFailure(
                  sourceSession.id,
                  "marker",
                  markerOutcome.kind === "timed-out" ? "timeout" : "rejection",
                  markerOutcome.kind === "failed"
                    ? markerOutcome.error
                    : undefined,
                ),
              );
              break;
            }
          } catch (error) {
            recordOperationError(
              lifecycleFailure(sourceSession.id, "history", "rejection", error),
            );
          }
        }
      }
    }
  }

  if (!operationError) {
    let stage: Extract<GenerationStage, "provider" | "validation"> = "provider";
    try {
      const body = unwrapLifecycleResponse(
        await v2.session.prompt({
          sessionID: fork.id,
          ...route,
          agent: MAGIC_COMPACT_SUMMARIZER_AGENT,
          model: settings.model,
          ...(settings.variant ? { variant: settings.variant } : {}),
          parts: [
            {
              type: "text",
              text: batch.prompt,
            },
          ],
        }),
      );
      if (isRecord(body) && body["info"] && isRecord(body["info"])) {
        const responseInfo = body["info"];
        const responseError = responseInfo["error"];
        if (responseError) throw responseError;
        if (isOutputTruncation(responseInfo["finish"])) {
          throw { name: "MessageOutputLengthError" };
        }
      }
      stage = "validation";
      results = parseGenerationResponse(
        isRecord(body) ? body["parts"] : undefined,
        candidates,
        generationNonce,
      );
    } catch (error) {
      recordOperationError(
        lifecycleFailure(
          sourceSession.id,
          stage,
          "rejection",
          error,
          candidates.length === 1,
        ),
      );
      try {
        unwrapLifecycleResponse(
          await v2.session.abort({ sessionID: fork.id, ...route }),
        );
      } catch (abortError) {
        operationError = combineLifecycleErrors(
          operationError!,
          lifecycleFailure(sourceSession.id, "abort", "rejection", abortError),
        );
      }
    }
  }

  let deleteError: Error | undefined;
  try {
    unwrapLifecycleResponse(
      await v2.session.delete({ sessionID: fork.id, ...route }),
    );
  } catch (error) {
    deleteError = lifecycleFailure(
      sourceSession.id,
      "delete",
      "rejection",
      error,
    );
    onFailureObserved();
  }
  if (operationError && deleteError) {
    throw combineLifecycleErrors(operationError, deleteError);
  }
  if (operationError) throw operationError;
  if (deleteError) throw deleteError;
  return results!;
}

function assertRetainedForkMessageShape(
  messages: MessageWithParts[],
  plan: CompactionPlan,
  retainedMessageIndexes: number[],
): void {
  if (messages.length !== retainedMessageIndexes.length) {
    throw new Error("Compaction generation fork history shape mismatch.");
  }
  for (const [index, message] of messages.entries()) {
    const source = plan.sourceSnapshot[retainedMessageIndexes[index]!];
    if (
      !source
      || !isRecord(source.info)
      || source.info["role"] !== message.info.role
      || source.parts.length !== message.parts.length
      || source.parts.some(
        (part, partIndex) =>
          !isRecord(part) || part["type"] !== message.parts[partIndex]?.type,
      )
    ) {
      throw new Error("Compaction generation fork history shape mismatch.");
    }
  }
}

function forkMessageIdentity(message: MessageWithParts): ForkMessageIdentity {
  return {
    id: message.info.id,
    sessionID: "sessionID" in message.info ? message.info.sessionID : undefined,
    role: message.info.role,
    parentID:
      "parentID" in message.info && typeof message.info.parentID === "string"
        ? message.info.parentID
        : undefined,
    parts: message.parts.map(part => ({
      id: part.id,
      sessionID: part.sessionID,
      messageID: part.messageID,
      type: part.type,
      tailStartID:
        part.type === "compaction" && typeof part.tail_start_id === "string"
          ? part.tail_start_id
          : undefined,
    })),
  };
}

function assertRetainedForkIdentity(
  messages: MessageWithParts[],
  expected: ForkMessageIdentity[],
): void {
  if (messages.length !== expected.length) {
    throw new Error("Compaction generation fork history shape mismatch.");
  }
  for (const [index, message] of messages.entries()) {
    const actual = forkMessageIdentity(message);
    const identity = expected[index];
    if (
      !identity
      || actual.id !== identity.id
      || actual.sessionID !== identity.sessionID
      || actual.role !== identity.role
      || actual.parentID !== identity.parentID
      || actual.parts.length !== identity.parts.length
      || actual.parts.some((part, partIndex) => {
        const expectedPart = identity.parts[partIndex];
        return (
          !expectedPart
          || part.id !== expectedPart.id
          || part.sessionID !== expectedPart.sessionID
          || part.messageID !== expectedPart.messageID
          || part.type !== expectedPart.type
          || part.tailStartID !== expectedPart.tailStartID
        );
      })
    ) {
      throw new Error("Compaction generation fork history shape mismatch.");
    }
  }
}

function assertForkRecordIdentity(
  messages: MessageWithParts[],
  forkSessionID: string,
): void {
  const messageIDs = new Set<string>();
  const partIDs = new Set<string>();
  for (const message of messages) {
    if (
      !message.info.id
      || message.info.sessionID !== forkSessionID
      || messageIDs.has(message.info.id)
    ) {
      throw new Error("Compaction generation fork history shape mismatch.");
    }
    messageIDs.add(message.info.id);
    for (const part of message.parts) {
      if (
        !part.id
        || part.sessionID !== forkSessionID
        || part.messageID !== message.info.id
        || partIDs.has(part.id)
      ) {
        throw new Error("Compaction generation fork history shape mismatch.");
      }
      partIDs.add(part.id);
    }
  }
}

function generationMarkerTargets(
  messages: MessageWithParts[],
  forkSessionID: string,
  batch: GenerationBatch,
  stopMarker: string,
): Array<{ message: MessageWithParts; part: TextPart }> {
  const targets = batch.candidates.map((candidate, index) => {
    const message = retainedForkMessage(
      messages,
      batch,
      batch.targetUserMessageIndexes[index]!,
    );
    return {
      message,
      part: generationMarkerPart(
        forkSessionID,
        message.info.id,
        candidate.marker,
        candidate.ordinal,
        false,
      ),
    };
  });
  if (batch.followingHalo) {
    const message = retainedForkMessage(
      messages,
      batch,
      batch.followingHalo.userMessageIndex,
    );
    targets.push({
      message,
      part: generationMarkerPart(
        forkSessionID,
        message.info.id,
        stopMarker,
        batch.candidates.at(-1)!.ordinal + 1,
        true,
      ),
    });
  }
  for (const target of targets) {
    if (target.message.parts.some(part => part.id === target.part.id)) {
      throw new Error("Compaction generation fork marker collision.");
    }
  }
  return targets;
}

function retainedForkMessage(
  messages: MessageWithParts[],
  batch: GenerationBatch,
  sourceMessageIndex: number,
): MessageWithParts {
  const retainedIndex =
    batch.retainedMessageIndexes.indexOf(sourceMessageIndex);
  const message = messages[retainedIndex];
  if (retainedIndex < 0 || !message || message.info.role !== "user") {
    throw new Error("Compaction generation fork history shape mismatch.");
  }
  return message;
}

function generationMarkerPart(
  sessionID: string,
  messageID: string,
  text: string,
  ordinal: number,
  stop: boolean,
): TextPart {
  return {
    id: `prt_-magic_generation_${stop ? "stop" : "turn"}_${ordinal}_${messageID}`,
    sessionID,
    messageID,
    type: "text",
    text,
    synthetic: true,
    metadata: {
      magicCompact: {
        generationMarker: true,
        ordinal,
        ...(stop ? { stop: true } : {}),
      },
    },
  };
}

function generationTurnMarker(nonce: string, ordinal: number): string {
  return `<magic-compact-turn nonce="${nonce}" ordinal="${ordinal}" />`;
}

function generationStopMarker(nonce: string, ordinal: number): string {
  return `<magic-compact-stop nonce="${nonce}" ordinal="${ordinal}" />`;
}

function createGenerationNonce(): string {
  return randomBytes(16).toString("base64url");
}

function selectGenerationNonce(
  histories: unknown[],
  nonceFactory: () => string,
  generationNonces: Set<string>,
): string {
  for (let attempt = 0; attempt < GENERATION_NONCE_ATTEMPTS; attempt += 1) {
    const nonce = nonceFactory();
    if (!GENERATION_NONCE_PATTERN.test(nonce)) {
      throw new Error("Compaction generation marker nonce is invalid.");
    }
    if (
      !generationNonces.has(nonce)
      && !histories.some(history => containsString(history, nonce))
    ) {
      generationNonces.add(nonce);
      return nonce;
    }
  }
  throw new GenerationMarkerNonceCollisionError();
}

function containsString(value: unknown, needle: string): boolean {
  const pending = [value];
  const seen = new Set<object>();
  while (pending.length > 0) {
    const current = pending.pop();
    if (typeof current === "string") {
      if (current.includes(needle)) return true;
      continue;
    }
    if (
      (typeof current !== "object" && typeof current !== "function")
      || current === null
      || seen.has(current)
    ) {
      continue;
    }
    seen.add(current);
    for (const key of Object.keys(current)) {
      pending.push(ownDataProperty(current, key));
    }
  }
  return false;
}

async function resolveCompactionSettings(
  v2: V2Client,
  sourceSession: Session,
  requiresImages: boolean,
): Promise<CompactionSettings> {
  let agents: Agent[];
  try {
    agents = unwrap(
      await v2.app.agents({
        directory: sourceSession.directory,
        ...(sourceSession.workspaceID
          ? { workspace: sourceSession.workspaceID }
          : {}),
      }),
    );
  } catch {
    throw new Error(
      "Magic Compact could not resolve the native compaction agent.",
    );
  }
  const compactionAgent = agents.find(
    (agent: Agent) =>
      agent.name === "compaction"
      && agent.native === true
      && agent.hidden === true,
  );
  if (!compactionAgent) {
    throw new Error(
      "Magic Compact requires the native hidden compaction agent, but it is missing or disabled.",
    );
  }
  if (
    !compactionAgent.model
    || !compactionAgent.model.providerID.trim()
    || !compactionAgent.model.modelID.trim()
  ) {
    throw new Error(
      "Magic Compact requires an explicit model on the native compaction agent.",
    );
  }
  let catalog: ReturnType<typeof unwrap>;
  try {
    catalog = unwrap(
      await v2.provider.list({
        directory: sourceSession.directory,
        ...(sourceSession.workspaceID
          ? { workspace: sourceSession.workspaceID }
          : {}),
      }),
    );
  } catch {
    throw new Error(
      requiresImages
        ? "Magic Compact could not verify an image-capable configured compaction model."
        : "Magic Compact could not resolve the configured compaction model descriptor.",
    );
  }
  const provider =
    isRecord(catalog) && Array.isArray(catalog["all"])
      ? catalog["all"].find(
          value =>
            isRecord(value)
            && value["id"] === compactionAgent.model!.providerID,
        )
      : undefined;
  const models = isRecord(provider) ? provider["models"] : undefined;
  const descriptor = isRecord(models)
    ? models[compactionAgent.model.modelID]
    : undefined;
  const limit = isRecord(descriptor) ? descriptor["limit"] : undefined;
  if (
    !isRecord(descriptor)
    || !isRecord(limit)
    || typeof limit["context"] !== "number"
  ) {
    throw new Error(
      requiresImages
        ? "Magic Compact requires an image-capable configured compaction model for retained fork history containing images."
        : "Magic Compact could not resolve the configured compaction model descriptor.",
    );
  }
  const capabilities = descriptor["capabilities"];
  const inputCapabilities = isRecord(capabilities)
    ? capabilities["input"]
    : undefined;
  return {
    model: compactionAgent.model,
    ...(compactionAgent.variant ? { variant: compactionAgent.variant } : {}),
    descriptor: {
      limit: {
        context: limit["context"],
        ...(typeof limit["input"] === "number"
          ? { input: limit["input"] }
          : {}),
        ...(typeof limit["output"] === "number"
          ? { output: limit["output"] }
          : {}),
      },
      ...(isRecord(inputCapabilities)
        ? {
            capabilities: {
              input: { image: inputCapabilities["image"] === true },
            },
          }
        : {}),
    },
  };
}

function assertConfiguredModelSupportsImages(
  settings: CompactionSettings,
): void {
  if (settings.descriptor.capabilities?.input?.image !== true) {
    throw new Error(
      "Magic Compact requires an image-capable configured compaction model for retained fork history containing images.",
    );
  }
}

function historyMayExposeImages(sourceSnapshot: unknown): boolean {
  if (!Array.isArray(sourceSnapshot)) return false;
  return sourceSnapshot.some(message => {
    if (!isRecord(message) || !Array.isArray(message["parts"])) return false;
    const role = isRecord(message["info"])
      ? message["info"]["role"]
      : undefined;
    return message["parts"].some(part => {
      if (!isRecord(part)) return false;
      if (
        (role === "user" || role === "assistant")
        && part["type"] === "file"
        && typeof part["mime"] === "string"
        && part["mime"].startsWith("image/")
      ) {
        return true;
      }
      if (part["type"] !== "tool" || !isRecord(part["state"])) return false;
      if (!toolAttachmentsModelVisible(part)) return false;
      const attachments = part["state"]["attachments"];
      return (
        Array.isArray(attachments)
        && attachments.some(
          attachment =>
            isRecord(attachment)
            && typeof attachment["mime"] === "string"
            && attachment["mime"].startsWith("image/"),
        )
      );
    });
  });
}

function buildGenerationPrompt(
  candidates: GenerationCandidate[],
  nextTurnAnchor: string | undefined,
  stopMarker: string,
  globalTurnIndexPrompt: string,
): string {
  return [
    "Generate one continuation summary for each indexed turn using the retained native fork history.",
    "Return only the JSON object required by your system prompt.",
    globalTurnIndexPrompt,
    "Use the ephemeral markers only as ordering guides. Echo each exact marker token, target ordinal, and target anchor; marker echo is allowed. Never include the active generation nonce or marker token in summaries or visual descriptions; useful source identifiers may remain when you include them.",
    JSON.stringify({
      turns: candidates.map(candidate => ({
        ordinal: candidate.ordinal,
        marker: candidate.marker,
        anchor: candidate.anchor,
        expectedInlineImages: candidate.expectedInlineImages,
      })),
      ...(nextTurnAnchor !== undefined
        ? {
            nextTurn: {
              marker: stopMarker,
              anchor: nextTurnAnchor,
            },
          }
        : {}),
    }),
  ].join("\n");
}

function estimateGenerationExplicitRequestTokens(
  candidates: GenerationCandidate[],
  nextTurnAnchor: string | undefined,
  stopMarker: string,
  globalTurnIndexPrompt: string,
): number {
  return (
    countGptTokens(
      buildGenerationPrompt(
        candidates,
        nextTurnAnchor,
        stopMarker,
        globalTurnIndexPrompt,
      ),
    )
    + candidates.reduce(
      (total, candidate) => total + countGptTokens(candidate.marker),
      0,
    )
    + (nextTurnAnchor === undefined ? 0 : countGptTokens(stopMarker))
  );
}

function estimateGenerationResponseProtocolTokens(
  candidates: GenerationCandidate[],
): number {
  return countGptTokens(
    JSON.stringify({
      turns: candidates.map(candidate => ({
        ordinal: candidate.ordinal,
        marker: candidate.marker,
        anchor: candidate.anchor,
        summary: "",
        visualDescriptions: Array.from(
          { length: candidate.expectedInlineImages },
          (_, index) => ({ ordinal: index + 1, description: "" }),
        ),
      })),
    }),
  );
}

function buildGlobalTurnIndexPrompt(plan: CompactionPlan): string {
  const sourceTurns = buildTurns(
    plan.sourceSnapshot as unknown as MessageWithParts[],
  );
  const globalTurnIndex = sourceTurns
    .slice(plan.effectiveSourceTurnStartIndex, plan.effectiveSourceTurnEndIndex)
    .map((turn, index) => ({
      sourceOrdinal: plan.effectiveSourceTurnStartIndex + index + 1,
      anchor: firstOrdinaryUserLine(turn, GLOBAL_ANCHOR_CHARACTER_LIMIT),
    }));
  return [
    "The globalTurnIndex is an orientation-only user-anchor map. Its sourceOrdinal values are immutable source-history positions, not target ordinals, and it does not supply distant assistant or tool detail. Each entry contains only its sourceOrdinal and bounded user anchor:",
    JSON.stringify({ globalTurnIndex }),
  ].join("\n");
}

function parseGenerationResponse(
  parts: unknown,
  candidates: GenerationCandidate[],
  nonce: string,
): GenerationTurnResult[] {
  if (!Array.isArray(parts)) throw validationFailure();
  const textParts = parts.flatMap(part => {
    if (!isRecord(part) || part["type"] !== "text") return [];
    const text = part["text"];
    return typeof text === "string" && text.trim() ? [text] : [];
  });
  if (textParts.length !== 1) throw validationFailure();

  let parsed: unknown;
  try {
    parsed = JSON.parse(textParts[0]!);
  } catch {
    throw validationFailure();
  }
  if (
    !isPlainRecord(parsed)
    || !hasExactKeys(parsed, ["turns"])
    || !Array.isArray(parsed["turns"])
    || parsed["turns"].length !== candidates.length
  ) {
    throw validationFailure();
  }

  return parsed["turns"].map((value, index) => {
    const candidate = candidates[index]!;
    if (
      !isPlainRecord(value)
      || !hasExactKeys(value, [
        "anchor",
        "marker",
        "ordinal",
        "summary",
        "visualDescriptions",
      ])
      || value["ordinal"] !== candidate.ordinal
      || value["marker"] !== candidate.marker
      || value["anchor"] !== candidate.anchor
      || typeof value["summary"] !== "string"
      || !value["summary"].trim()
      || !Array.isArray(value["visualDescriptions"])
    ) {
      throw validationFailure();
    }
    const expectedImages = candidate.mediaManifest?.images.length ?? 0;
    if (value["visualDescriptions"].length !== expectedImages) {
      throw validationFailure();
    }
    const summary = value["summary"].trim();
    if (summary.includes(nonce)) {
      throw validationFailure();
    }
    const visualDescriptions = value["visualDescriptions"].map(
      (description, imageIndex) => {
        if (
          !isPlainRecord(description)
          || !hasExactKeys(description, ["description", "ordinal"])
          || description["ordinal"] !== imageIndex + 1
          || typeof description["description"] !== "string"
          || !description["description"].trim()
        ) {
          throw validationFailure();
        }
        const text = description["description"].trim();
        if (text.includes(nonce)) {
          throw validationFailure();
        }
        return {
          ordinal: imageIndex + 1,
          description: text,
        };
      },
    );
    return {
      summary,
      visualDescriptions,
    };
  });
}

function frameSummary(result: GenerationTurnResult): string {
  if (result.visualDescriptions.length === 0) return result.summary;
  return [
    result.summary,
    "",
    "Visual summary:",
    ...result.visualDescriptions.map(
      item => `${item.ordinal}. ${item.description}`,
    ),
  ].join("\n");
}

function firstOrdinaryUserLine(turn: Turn, characterLimit: number): string {
  const text = turn.user
    .flatMap(message => message.parts)
    .find(
      (part): part is TextPart =>
        part.type === "text"
        && part.synthetic !== true
        && part.ignored !== true,
    )?.text;
  const firstLine = (text ?? "").split(/\r\n|\r|\n/u, 1)[0] ?? "";
  return Array.from(firstLine).slice(0, characterLimit).join("");
}

function assertGeneratedPartIDsAvailable(plan: CompactionPlan): void {
  const partIDs = new Set(
    [...plan.summarizedTurns, ...(plan.nextTurn ? [plan.nextTurn] : [])]
      .flatMap(turn => [...turn.user, ...turn.assistants])
      .flatMap(message => message.parts.map(part => part.id)),
  );
  for (const turn of plan.summarizedTurns) {
    if (turnHasPersistedSummary(turn)) continue;
    const assistant = turn.assistants[0];
    if (!assistant) throw new Error("Turn missing assistant message.");
    if (partIDs.has(summaryEnvelopePartID(assistant.info.id))) {
      throw new Error(
        "Compaction plan collides with a generated summary part.",
      );
    }
  }
  const nextUser = plan.nextTurn?.user[0];
  if (nextUser && partIDs.has(boundaryPartID(nextUser.info.id))) {
    throw new Error(
      "Compaction plan collides with the generated boundary part.",
    );
  }
}

function validationFailure(): Error {
  return new Error("Compaction generation response validation failed.");
}

function hasExactKeys(
  value: Record<string, unknown>,
  expected: string[],
): boolean {
  const keys = Object.keys(value).sort();
  const sortedExpected = [...expected].sort();
  return (
    keys.length === sortedExpected.length
    && keys.every((key, index) => key === sortedExpected[index])
  );
}

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  if (!isRecord(value) || Array.isArray(value)) return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

type LifecycleFailureKind = "timeout" | "rejection";

class GenerationMarkerNonceCollisionError extends Error {
  constructor() {
    super("Compaction generation marker nonce collision.");
    this.name = "GenerationMarkerNonceCollisionError";
  }
}

class GenerationLifecycleError extends Error {
  readonly status: number | undefined;

  constructor(
    readonly sourceSessionID: string,
    readonly stage: GenerationStage,
    readonly kind: LifecycleFailureKind,
    status: number | undefined,
    message: string,
  ) {
    super(message);
    this.name = "GenerationLifecycleError";
    this.status = status;
  }
}

function lifecycleFailure(
  sourceSessionID: string,
  stage: GenerationStage,
  kind: LifecycleFailureKind,
  error?: unknown,
  singleTurn = false,
): GenerationLifecycleError {
  const status = safeStatus(error);
  const overflow = stage === "provider" && isContextOverflow(error, status);
  const truncated = stage === "provider" && isOutputTruncation(error);
  const message =
    error instanceof GenerationMarkerNonceCollisionError
      ? error.message
      : overflow
        ? "The complete forked session exceeded the configured compaction model context. Run /magic-trim first, then retry /magic-compact."
        : truncated
          ? singleTurn
            ? "The compaction model reached its output limit while summarizing a single turn. Configure a compaction model with more output capacity."
            : "The compaction model reached its output limit. Retry /magic-compact with a larger N so fewer turns are selected."
          : lifecycleMessage(stage, kind);
  return new GenerationLifecycleError(
    sourceSessionID,
    stage,
    kind,
    status,
    message,
  );
}

function sanitizeGenerationFailure(
  sourceSessionID: string,
  error: unknown,
): Error {
  if (error instanceof GenerationLifecycleError) return error;
  if (error instanceof AggregateError) {
    const errors = error.errors.map(item =>
      sanitizeGenerationFailure(sourceSessionID, item),
    );
    if (errors.length > 0) {
      return new AggregateError(
        errors,
        "Compaction generation batch and fork cleanup failed.",
        { cause: errors[0] },
      );
    }
  }
  return lifecycleFailure(sourceSessionID, "validation", "rejection");
}

function lifecycleMessage(
  stage: GenerationStage,
  kind: LifecycleFailureKind,
): string {
  if (stage === "fork") {
    return "Compaction generation fork creation failed.";
  }
  if (stage === "permission") {
    return kind === "timeout"
      ? "Compaction generation fork permission update timed out."
      : "Compaction generation fork permission update failed.";
  }
  if (stage === "history") {
    return kind === "timeout"
      ? "Compaction generation fork history verification timed out."
      : "Compaction generation fork history shape verification failed.";
  }
  if (stage === "isolation") {
    return kind === "timeout"
      ? "Compaction generation fork message isolation timed out."
      : "Compaction generation fork message isolation failed.";
  }
  if (stage === "marker") {
    return kind === "timeout"
      ? "Compaction generation fork marker injection timed out."
      : "Compaction generation fork marker injection failed.";
  }
  if (stage === "provider")
    return "Compaction generation model request failed.";
  if (stage === "validation") {
    return "Compaction generation response validation failed.";
  }
  if (stage === "abort") {
    return "Compaction generation fork abort failed.";
  }
  return "Compaction generation fork deletion failed.";
}

function safeStatus(error: unknown): number | undefined {
  for (const item of diagnosticChain(error)) {
    const data = ownDataProperty(item, "data");
    const value =
      ownDataProperty(item, "status")
      ?? ownDataProperty(item, "statusCode")
      ?? ownDataProperty(data, "statusCode");
    if (
      typeof value === "number"
      && Number.isInteger(value)
      && value >= 100
      && value <= 599
    ) {
      return value;
    }
  }
  return undefined;
}

function isContextOverflow(
  error: unknown,
  status: number | undefined,
): boolean {
  return (
    status === 413
    || diagnosticChain(error).some(item => {
      const categories = diagnosticCategories(item);
      return categories.some(category =>
        [
          "contextoverflowerror",
          "context_length_exceeded",
          "context_window_exceeded",
          "prompt_too_long",
          "input_too_long",
        ].includes(category),
      );
    })
  );
}

function isOutputTruncation(value: unknown): boolean {
  const outputCategories = [
    "length",
    "max_token",
    "max_tokens",
    "max_output_tokens",
    "output_length",
    "output_length_exceeded",
    "messageoutputlengtherror",
  ];
  if (typeof value === "string" && outputCategories.includes(category(value))) {
    return true;
  }
  return diagnosticChain(value).some(item => {
    return diagnosticCategories(item).some(itemCategory =>
      outputCategories.includes(itemCategory),
    );
  });
}

function diagnosticChain(value: unknown): unknown[] {
  const chain: unknown[] = [];
  const pending = [value];
  const seen = new Set<object>();
  while (pending.length > 0 && chain.length < 32) {
    const current = pending.shift();
    if (current === undefined) continue;
    if (
      (typeof current === "object" || typeof current === "function")
      && current !== null
    ) {
      if (seen.has(current)) continue;
      seen.add(current);
    }
    chain.push(current);
    pending.push(
      ownDataProperty(current, "cause"),
      ownDataProperty(current, "error"),
      ownDataProperty(current, "data"),
    );
  }
  return chain;
}

function diagnosticCategories(value: unknown): string[] {
  return ["name", "code", "type", "reason"].flatMap(key => {
    const item = ownDataProperty(value, key);
    return typeof item === "string" ? [category(item)] : [];
  });
}

function category(value: string): string {
  return value.trim().toLowerCase().replaceAll("-", "_");
}

function ownDataProperty(value: unknown, key: string): unknown {
  if ((typeof value !== "object" && typeof value !== "function") || !value) {
    return undefined;
  }
  try {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    return descriptor && "value" in descriptor ? descriptor.value : undefined;
  } catch {
    return undefined;
  }
}

function unwrapLifecycleResponse<T, E>(response: { data?: T; error?: E }): T {
  if ("error" in response && response.error) throw response.error;
  return response.data as T;
}

function combineLifecycleErrors(primary: Error, secondary: Error): Error {
  return new AggregateError(
    [primary, secondary],
    "Compaction generation and fork cleanup both failed.",
    { cause: primary },
  );
}

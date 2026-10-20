import type {
  Agent,
  PermissionRuleset,
  Session,
  TextPart,
} from "@opencode-ai/sdk/v2";
import { isDeepStrictEqual } from "node:util";
import { unwrap, type V2Client } from "../api";
import {
  attemptWithinDeadline,
  CONTROL_OPERATION_TIMEOUT_MS,
  isRecord,
} from "../util";
import {
  MAGIC_COMPACT_SUMMARIZER_AGENT,
  boundaryPartID,
  summaryEnvelopePartID,
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
  buildGenerationPrompt,
  capCodePoints,
  containsString,
  createGenerationNonce,
  diagnosticCategories,
  diagnosticChain,
  estimateGenerationExplicitRequestTokens,
  estimateGenerationResponseProtocolTokens,
  frameSummary,
  GENERATION_RESPONSE_SCHEMA,
  GenerationMarkerNonceCollisionError,
  GenerationResponseValidationError,
  generationTurnMarker,
  type GenerationTurnResult,
  GLOBAL_ANCHOR_CHARACTER_LIMIT,
  globalTurnIndexPrompt,
  isContextOverflow,
  isOutputTruncation,
  parseGenerationResponse,
  safeStatus,
  selectGenerationNonce,
  validationCategoryText,
  validationFailure,
} from "../policy/generation";
import {
  distinctReasons,
  markPluginMessage,
  PluginMessageError,
  undeletedSessionIDs,
} from "../policy/feedback";
import { ForkCleanupError, ForkHistoryError, forkSession } from "./fork";
import {
  assertForkMessageShape,
  buildTurns,
  snapshotSourceMessages,
  type CompactionPlan,
  type MessageWithParts,
  type Turn,
} from "./plan";

const GENERATION_PERMISSION: PermissionRuleset = [
  { permission: "*", pattern: "*", action: "deny" },
  { permission: "StructuredOutput", pattern: "*", action: "allow" },
];
// OpenCode transmits the response schema as the model-visible
// StructuredOutput tool schema without enforcing it.
const GENERATION_FORMAT = {
  type: "json_schema" as const,
  schema: GENERATION_RESPONSE_SCHEMA,
};
export type CompactionSettings = {
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
  marker: string;
  expectedInlineImages: number;
  inputTokens: number;
  outputTokens: number;
  mediaManifest?: MediaManifest;
};

type GenerationBatchPlan = NativeGenerationBatch<GenerationCandidate> & {
  retainedMessageIndexes: number[];
  targetUserMessageIndexes: number[];
};

type GenerationBatch = GenerationBatchPlan & {
  generationNonce: string;
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

// Generation forks still owned by a running command. Native compaction of one
// would send an extra model request and replace its retained history, so the
// plugin's compacting hook ends the fork's prompt loop as a context overflow,
// which reports trim-first guidance.
const ownedGenerationForks = new Set<string>();

export function rejectNativeCompactionOfGenerationFork(
  sessionID: string,
): void {
  if (!ownedGenerationForks.has(sessionID)) return;
  const error = new Error(
    "Magic Compact owns this generation session; native compaction is blocked.",
  );
  error.name = "ContextOverflowError";
  throw error;
}

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
        globalTurnIndexPrompt,
      ),
    responseProtocolTokens: estimateGenerationResponseProtocolTokens,
  };
  const batchPlans = planNativeGenerationBatches(
    candidates,
    budget,
    protocol,
  ).map(batch => createGenerationBatch(batch, plan, budget));
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
  const sourceOutcome = await attemptWithinDeadline(
    signal =>
      v2.session
        .messages(
          {
            sessionID: session.id,
            directory: session.directory,
            ...(session.workspaceID ? { workspace: session.workspaceID } : {}),
          },
          { signal },
        )
        .then(unwrapLifecycleResponse),
    controlTimeoutMs,
  );
  if (sourceOutcome.kind !== "success") {
    throw lifecycleFailure(
      session.id,
      "history",
      sourceOutcome.kind === "timed-out" ? "timeout" : "rejection",
      sourceOutcome.kind === "failed" ? sourceOutcome.error : undefined,
    );
  }
  if (
    !isDeepStrictEqual(
      snapshotSourceMessages(sourceOutcome.value),
      plan.sourceSnapshot,
    )
  ) {
    throw lifecycleFailure(session.id, "history", "rejection");
  }
  const results = await requestCompactionGenerationBatches(
    v2,
    session,
    settings,
    batches,
    plan,
    sourceOutcome.value,
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

function createGenerationBatch(
  batch: NativeGenerationBatch<GenerationCandidate>,
  plan: CompactionPlan,
  budget: NativeBatchBudget,
): GenerationBatchPlan {
  const sourceMessages = plan.sourceSnapshot as MessageWithParts[];
  const sourceTurns = buildTurns(sourceMessages);
  const retainedTurnIndexes = new Set(
    batch.candidates.map(candidate => candidate.sourceTurnIndex),
  );
  const firstTurnIndex = batch.candidates[0]!.sourceTurnIndex;
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
  let inputLimit = batch.inputLimit;
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
    inputLimit,
    retainedMessageIndexes,
    targetUserMessageIndexes,
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
  const prompt = buildGenerationPrompt(candidates, globalTurnIndexPrompt);
  const exactExplicitRequestTokens = estimateGenerationExplicitRequestTokens(
    candidates,
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
    throw new PluginMessageError(
      "Magic Compact exact generation request cannot fit the configured model input. Run /magic-trim first and retry.",
    );
  }
  if (
    exactOutputTokens > batch.estimatedOutputTokens
    || exactOutputTokens > budget.outputTarget
  ) {
    throw new PluginMessageError(
      `A single compaction turn cannot fit the configured ${budget.outputTarget}-token output target. Configure a compaction model with more output capacity.`,
    );
  }
  return {
    ...batch,
    candidates,
    generationNonce,
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
  sourceMessages: MessageWithParts[],
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
          sourceMessages,
          controlTimeoutMs,
          () => {
            stopAdmission = true;
          },
        );
        if (batchResults.length !== batch.candidates.length) {
          throw validationFailure("turn count mismatch", {
            expected: batch.candidates.length,
            received: batchResults.length,
          });
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
    throw ownedAggregate(
      orderedFailures,
      "Multiple compaction generation batches failed.",
    );
  }

  const candidates = batches.flatMap(batch => batch.candidates);
  const orderedResults = results.flatMap(batchResults => batchResults ?? []);
  if (orderedResults.length !== candidates.length) {
    throw lifecycleFailure(
      sourceSession.id,
      "validation",
      "rejection",
      validationFailure("turn count mismatch", {
        expected: candidates.length,
        received: orderedResults.length,
      }),
    );
  }
  if (
    candidates.some(
      (candidate, index) =>
        candidate.ordinal !== index + 1 || !orderedResults[index],
    )
  ) {
    throw lifecycleFailure(
      sourceSession.id,
      "validation",
      "rejection",
      validationFailure("invalid envelope"),
    );
  }
  return orderedResults;
}

async function requestCompactionGeneration(
  v2: V2Client,
  sourceSession: Session,
  settings: CompactionSettings,
  batch: GenerationBatch,
  plan: CompactionPlan,
  sourceMessages: MessageWithParts[],
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
    fork = await forkSession(
      v2,
      sourceSession,
      sourceMessages,
      controlTimeoutMs,
    );
  } catch (error) {
    onFailureObserved();
    if (error instanceof ForkCleanupError) throw error;
    throw lifecycleFailure(
      sourceSession.id,
      error instanceof ForkHistoryError ? "history" : "fork",
      error instanceof ForkHistoryError && error.reason === "read-timeout"
        ? "timeout"
        : "rejection",
      error,
    );
  }

  ownedGenerationForks.add(fork.id);
  let operationError: Error | undefined;
  const recordOperationError = (error: Error): void => {
    operationError = error;
    onFailureObserved();
  };
  let results: GenerationTurnResult[] | undefined;
  const generationNonce = batch.generationNonce;
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
          format: GENERATION_FORMAT,
          parts: [
            {
              type: "text",
              text: batch.prompt,
            },
          ],
        }),
      );
      const responseInfo =
        isRecord(body) && isRecord(body["info"]) ? body["info"] : undefined;
      const structuredOutput = responseInfo?.["structured"];
      if (responseInfo) {
        const responseError = responseInfo["error"];
        const outputTruncated = isOutputTruncation(responseInfo["finish"]);
        const structuredOutputFailure = isStructuredOutputError(responseError);
        if (responseError && !(outputTruncated && structuredOutputFailure)) {
          throw responseError;
        }
        if (outputTruncated && structuredOutput === undefined) {
          throw { name: "MessageOutputLengthError" };
        }
      }
      stage = "validation";
      results = parseGenerationResponse(
        structuredOutput,
        candidates,
        generationNonce,
      );
    } catch (error) {
      recordOperationError(
        lifecycleFailure(sourceSession.id, stage, "rejection", error),
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
    ownedGenerationForks.delete(fork.id);
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

export function assertForkRecordIdentity(
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
      ),
    };
  });
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
): TextPart {
  return {
    id: `prt_-magic_generation_turn_${ordinal}_${messageID}`,
    sessionID,
    messageID,
    type: "text",
    text,
    synthetic: true,
    metadata: {
      magicCompact: {
        generationMarker: true,
        ordinal,
      },
    },
  };
}

export const SUMMARIZER_MODEL_REQUIRED = `Magic Compact requires an explicit model on its ${MAGIC_COMPACT_SUMMARIZER_AGENT} agent. Set agent["${MAGIC_COMPACT_SUMMARIZER_AGENT}"].model (and optionally variant) in your OpenCode config.`;

export async function resolveCompactionSettings(
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
    throw new PluginMessageError(
      `Magic Compact could not resolve the ${MAGIC_COMPACT_SUMMARIZER_AGENT} agent.`,
    );
  }
  const summarizer = agents.find(
    (agent: Agent) => agent.name === MAGIC_COMPACT_SUMMARIZER_AGENT,
  );
  if (!summarizer) {
    throw new PluginMessageError(
      `Magic Compact requires its ${MAGIC_COMPACT_SUMMARIZER_AGENT} agent, but it is missing or disabled.`,
    );
  }
  // No fallback: the source session's model and OpenCode's native compaction
  // agent are never used in its place.
  if (
    !summarizer.model
    || !summarizer.model.providerID.trim()
    || !summarizer.model.modelID.trim()
  ) {
    throw new PluginMessageError(SUMMARIZER_MODEL_REQUIRED);
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
    throw new PluginMessageError(
      requiresImages
        ? "Magic Compact could not verify an image-capable configured compaction model."
        : "Magic Compact could not resolve the configured compaction model descriptor.",
    );
  }
  const provider =
    isRecord(catalog) && Array.isArray(catalog["all"])
      ? catalog["all"].find(
          value =>
            isRecord(value) && value["id"] === summarizer.model!.providerID,
        )
      : undefined;
  const models = isRecord(provider) ? provider["models"] : undefined;
  const descriptor = isRecord(models)
    ? models[summarizer.model.modelID]
    : undefined;
  const limit = isRecord(descriptor) ? descriptor["limit"] : undefined;
  if (
    !isRecord(descriptor)
    || !isRecord(limit)
    || typeof limit["context"] !== "number"
  ) {
    throw new PluginMessageError(
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
    model: summarizer.model,
    ...(summarizer.variant ? { variant: summarizer.variant } : {}),
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

export function assertConfiguredModelSupportsImages(
  settings: CompactionSettings,
): void {
  if (settings.descriptor.capabilities?.input?.image !== true) {
    throw new PluginMessageError(
      "Magic Compact requires an image-capable configured compaction model for retained fork history containing images.",
    );
  }
}

export function historyMayExposeImages(sourceSnapshot: unknown): boolean {
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

function buildGlobalTurnIndexPrompt(plan: CompactionPlan): string {
  const sourceTurns = buildTurns(
    plan.sourceSnapshot as unknown as MessageWithParts[],
  );
  const globalTurnIndex = sourceTurns
    .slice(plan.effectiveSourceTurnStartIndex, plan.effectiveSourceTurnEndIndex)
    .map((turn, index) => ({
      sourceOrdinal: plan.effectiveSourceTurnStartIndex + index + 1,
      anchor: capCodePoints(
        firstOrdinaryUserLine(turn),
        GLOBAL_ANCHOR_CHARACTER_LIMIT,
      ),
    }));
  return globalTurnIndexPrompt(globalTurnIndex);
}

function firstOrdinaryUserLine(turn: Turn): string {
  const text = turn.user
    .flatMap(message => message.parts)
    .find(
      (part): part is TextPart =>
        part.type === "text"
        && part.synthetic !== true
        && part.ignored !== true,
    )?.text;
  return (text ?? "").split(/\r\n|\r|\n/u, 1)[0] ?? "";
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

type LifecycleFailureKind = "timeout" | "rejection";

class GenerationLifecycleError extends PluginMessageError {
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
): GenerationLifecycleError {
  const status = safeStatus(error);
  if (error instanceof ForkHistoryError) {
    return new GenerationLifecycleError(
      sourceSessionID,
      stage,
      kind,
      status,
      `Compaction generation fork history verification ${kind === "timeout" ? "timed out" : "failed"} (${error.detail}).`,
    );
  }
  const overflow = stage === "provider" && isContextOverflow(error, status);
  const truncated = stage === "provider" && isOutputTruncation(error);
  const missingStructuredOutput =
    stage === "provider" && isStructuredOutputError(error);
  const message =
    error instanceof GenerationMarkerNonceCollisionError
      ? error.message
      : stage === "validation"
          && error instanceof GenerationResponseValidationError
        ? `Compaction generation response validation failed: ${validationCategoryText(error)}.`
        : overflow
          ? "The complete forked session exceeded the configured compaction model context. Run /magic-trim first, then retry /magic-compact."
          : truncated
            ? "The compaction model reached its output limit while summarizing a single turn. Configure a compaction model with more output capacity."
            : missingStructuredOutput
              ? "The compaction model finished without submitting its StructuredOutput response."
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
  if (
    error instanceof GenerationLifecycleError
    || error instanceof ForkCleanupError
  )
    return error;
  if (error instanceof AggregateError) {
    const errors = error.errors.map(item =>
      sanitizeGenerationFailure(sourceSessionID, item),
    );
    if (errors.length > 0) {
      return ownedAggregate(
        errors,
        "Compaction generation batch and fork cleanup failed.",
      );
    }
  }
  return lifecycleFailure(sourceSessionID, "validation", "rejection", error);
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

function isStructuredOutputError(value: unknown): boolean {
  return diagnosticChain(value).some(item =>
    diagnosticCategories(item).includes("structuredoutputerror"),
  );
}

function unwrapLifecycleResponse<T, E>(response: { data?: T; error?: E }): T {
  if ("error" in response && response.error) throw response.error;
  return response.data as T;
}

function combineLifecycleErrors(primary: Error, secondary: Error): Error {
  return ownedAggregate(
    [primary, secondary],
    "Compaction generation and fork cleanup both failed.",
  );
}

// An aggregate of this plugin's already sanitized generation failures. Its
// message adds their distinct reasons, so a person sees why the batches
// failed (for example, to run /magic-trim first) and not only that several
// did, and then every partial fork that could not be deleted, which the
// reason cap never drops. The failures stay in `errors`, unchanged and in
// batch order.
function ownedAggregate(errors: Error[], summary: string): AggregateError {
  const forks = undeletedSessionIDs(errors);
  const message = [
    summary,
    ...distinctReasons(errors),
    ...(forks.length === 0
      ? []
      : [
          `Delete partial fork${forks.length === 1 ? "" : "s"} ${forks.join(", ")} manually.`,
        ]),
  ].join(" ");
  return markPluginMessage(
    new AggregateError(errors, message, { cause: errors[0] }),
  );
}

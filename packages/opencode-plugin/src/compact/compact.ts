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
import { MAGIC_COMPACT_SUMMARIZER_AGENT } from "../policy/protocol";
import { boundaryPartID, summaryEnvelopePartID } from "./capsule";
import { estimateNativeTurnCost, toolAttachmentsModelVisible } from "./batch";
import {
  computeNativeBatchBudget,
  fits,
  NATIVE_BATCH_PLANNING_NONCE,
  outputCapacityMessage,
  planNativeGenerationRequest,
  refuse,
  requireFit,
  type Fit,
  type NativeBatchBudget,
  type NativeGenerationRequest,
  type NativeRequestProtocol,
} from "../policy/budget";
import { createMediaManifest, type MediaManifest } from "./media";
import {
  buildGenerationPrompt,
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
  globalTurnIndexPrompt,
  isContextOverflow,
  isOutputTruncation,
  parseGenerationResponse,
  safeStatus,
  selectGenerationNonce,
  validationCategoryText,
  type GenerationTarget,
  type GenerationTurnResult,
} from "../policy/generation";
import {
  distinctReasons,
  markPluginMessage,
  PluginMessageError,
  unremovedSessions,
  unremovedSessionsText,
  type LeavesSessions,
  type UnremovedSession,
} from "../policy/feedback";
import { runGenerationBatches } from "../policy/generation-pool";
import {
  createForkCut,
  ForkHistoryError,
  sessionRoute,
  type ForkCut,
  type HistoryRoute,
} from "./fork";
import { registerOwnedFork, releaseOwnedFork, type OwnedFork } from "./owned";
import {
  chooseStage,
  matchCandidate,
  reasoningContextMode,
  settleMatch,
  stageReport as sharedStageReport,
  verifiedReasoning,
  type MatchCandidate,
  type ModelIdentity,
  type Qualifies,
  type ReasoningRoute,
  type StageChoice,
  type StageReport,
} from "../policy/reasoning";
import {
  indexReplay,
  reasoningUnits,
  type ReplayContext,
  type ReplayIndex,
} from "./reasoning";
import {
  assertForkMessageShape,
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
    // The provider package, wire model ID and earlier-reasoning field the
    // host lists, which decide the verified reasoning routes, and the
    // variants it offers.
    npm?: string;
    wireModelID?: string;
    reasoningField?: string;
    variants?: readonly string[];
  };
};

// One target as planning estimates it, under the conservative planning
// marker; the request that runs names it under its real marker.
type GenerationCandidate = GenerationTarget & {
  // Position in the plan's targets.
  index: number;
  sourceTurnIndex: number;
  inputTokens: number;
  outputTokens: number;
  mediaManifest?: MediaManifest;
};

type RetainedRequest = NativeGenerationRequest<GenerationCandidate> & {
  retainedMessageIndexes: number[];
  targetUserMessageIndex: number;
};

type GenerationBatch = {
  target: GenerationTarget;
  generationNonce: string;
  prompt: string;
  // Source message positions the request keeps: the target turn and at most
  // the one complete preceding turn.
  retainedMessageIndexes: number[];
  targetUserMessageIndex: number;
  // The summary model this request runs on: the configured summarizer, or
  // under `--match-model` the model that wrote the target's reasoning.
  settings: CompactionSettings;
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

export type GenerationOptions = {
  // The command's summarizer catalog, shared with its retention summary.
  catalog: LoadSummarizerCatalog;
  matchModel?: boolean;
  controlTimeoutMs?: number;
  nonceFactory?: () => string;
};

export type GeneratedSummary = {
  text: string;
  mediaManifest?: MediaManifest;
};

export type GenerationResult = {
  // One framed summary per plan target, in target order.
  summaries: GeneratedSummary[];
  // Each turn request's reasoning report, in target order.
  reports: StageReport[];
  // The model the successor continues with and its reasoning context.
  replay: ReplayContext | undefined;
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

// A summary request carries the protected reasoning of earlier turns only
// when OpenAI Responses runs it with every turn's reasoning context. The
// plugin's `chat.params` hook asks for that on this plugin's own summary and
// generation forks on a verified model, and changes nothing else. Whether a
// later hook changes it again is not observed; budgets already assume the
// costlier `all_turns`.
export function requestAllTurnsReasoning(
  owned: boolean,
  model: {
    providerID: string;
    id: string;
    api?: { id?: string; npm?: string };
  },
  options: Record<string, unknown>,
): void {
  if (!owned) return;
  if (
    !verifiedReasoning(
      v1ReasoningRoute(
        { providerID: model.providerID, modelID: model.id },
        { npm: model.api?.npm, wireModelID: model.api?.id },
      ),
    )
  ) {
    return;
  }
  options["reasoningContext"] = "all_turns";
}

export async function generateCompactionSummaries(
  v2: V2Client,
  session: Session,
  plan: CompactionPlan,
  options: GenerationOptions,
): Promise<GenerationResult> {
  const controlTimeoutMs =
    options.controlTimeoutMs ?? CONTROL_OPERATION_TIMEOUT_MS;
  if (!Number.isFinite(controlTimeoutMs) || controlTimeoutMs <= 0) {
    throw new Error("Compaction generation control deadline must be positive.");
  }
  // Nothing to summarize needs no catalog.
  if (plan.targets.length === 0) {
    return { summaries: [], reports: [], replay: undefined };
  }

  assertGeneratedPartIDsAvailable(plan);
  const globalTurnIndexPrompt = buildGlobalTurnIndexPrompt(plan);
  const candidates = plan.targets.map(
    ({ turn, sourceTurnIndex }, index): GenerationCandidate => {
      const mediaManifest = createMediaManifest(turn);
      return {
        index,
        sourceTurnIndex,
        ordinal: index + 1,
        marker: generationTurnMarker(NATIVE_BATCH_PLANNING_NONCE, index + 1),
        expectedInlineImages: mediaManifest?.images.length ?? 0,
        ...estimateNativeTurnCost(turn),
        ...(mediaManifest ? { mediaManifest } : {}),
      };
    },
  );

  const catalog = await options.catalog(
    historyMayExposeImages(plan.sourceMessages),
  );
  const configured = catalog.configured;
  const protocol: NativeRequestProtocol<GenerationCandidate> = {
    explicitRequestTokens: candidate =>
      estimateGenerationExplicitRequestTokens(candidate, globalTurnIndexPrompt),
    responseProtocolTokens: estimateGenerationResponseProtocolTokens,
  };
  // One attribution of the source for every request's replay estimate.
  const replayIndex = indexReplay(plan.sourceMessages);
  // A request's input with each attributable protected unit its model is sent
  // counted by that unit's replay estimate instead of its visible text.
  const planBatch = (
    candidate: GenerationCandidate,
    settings: CompactionSettings,
    nonce: string,
  ): Fit<GenerationBatch> => {
    const budget = computeNativeBatchBudget(settings.descriptor.limit);
    if (budget.kind === "capacity") return budget;
    const replayDelta = (turn: Turn) => {
      const estimate = replayIndex.estimate(
        catalog.requestContext(settings.model),
        new Set(turnMessages(turn).map(message => message.info.id)),
      );
      return estimate.tokens - estimate.visibleTokens;
    };
    const target = plan.targets[candidate.index]!.turn;
    const request = planNativeGenerationRequest(
      {
        ...candidate,
        inputTokens: candidate.inputTokens + replayDelta(target),
      },
      budget.value,
      protocol,
    );
    if (request.kind === "capacity") return request;
    const batch = finalizeGenerationBatch(
      retainPrecedingTurn(request.value, plan, budget.value, replayDelta),
      budget.value,
      globalTurnIndexPrompt,
      nonce,
    );
    if (batch.kind === "capacity") return batch;
    return fits({ ...batch.value, settings });
  };
  const generationNonces = new Set<string>();
  const reports: StageReport[] = [];
  const batches = candidates.map(candidate => {
    const turn = plan.targets[candidate.index]!.turn;
    // One nonce per target, shared by the matched model's fit check and the
    // request that runs, so a plan that is not used claims none.
    const nonce = selectGenerationNonce(
      [plan.sourceMessages],
      options.nonceFactory ?? createGenerationNonce,
      generationNonces,
    );
    // The target's own reasoning decides the match, never the halo's. The
    // image check covers the turn that may join it as context.
    const preceding = plan.sourceTurns[candidate.sourceTurnIndex - 1];
    const exposesImages = historyMayExposeImages([
      ...(preceding ? turnMessages(preceding) : []),
      ...turnMessages(turn),
    ]);
    const match = matchCandidate(
      reasoningUnits(turn.assistants),
      catalog.qualifies,
    );
    const { choice, matched } = settleMatch(
      chooseStage(match, configured, model => {
        const descriptor = catalog.describe(model);
        return {
          variants: new Set(descriptor?.variants ?? []),
          optionsConflict: catalog.summarizerOptions,
          imagesBlocked:
            exposesImages && descriptor?.capabilities?.input?.image !== true,
        };
      }),
      options.matchModel === true,
      (model, variant) =>
        planBatch(
          candidate,
          {
            model,
            ...(variant === undefined ? {} : { variant }),
            descriptor: catalog.describe(model)!,
          },
          nonce,
        ),
    );
    const batch =
      matched ?? requireFit(planBatch(candidate, configured, nonce));
    const retained = batch.retainedMessageIndexes.map(
      index => plan.sourceMessages[index]!,
    );
    if (historyMayExposeImages(retained)) {
      assertConfiguredModelSupportsImages(batch.settings);
    }
    reports.push(
      stageReport({
        stage: "turn",
        candidate: match,
        choice,
        settings: batch.settings,
        history: retained,
        catalog,
        replay: replayIndex,
      }),
    );
    return batch;
  });
  const sourceMessages = await control(controlTimeoutMs, "history", signal =>
    v2.session.messages(
      { sessionID: session.id, ...sessionRoute(session) },
      { signal },
    ),
  );
  if (
    !isDeepStrictEqual(
      snapshotSourceMessages(sourceMessages),
      plan.sourceSnapshot,
    )
  ) {
    throw lifecycleFailure("history", "rejection");
  }
  const outcome = await runGenerationBatches(batches, (batch, stop) =>
    requestCompactionGeneration(
      v2,
      session,
      batch,
      plan,
      sourceMessages,
      controlTimeoutMs,
      stop,
    ),
  );
  if (outcome.kind === "failed") {
    const [first, ...others] = outcome.failures;
    if (others.length === 0) throw first;
    throw ownedAggregate(
      outcome.failures,
      "Multiple compaction generation batches failed.",
    );
  }
  return {
    summaries: candidates.map((candidate, index) => ({
      text: frameSummary(outcome.results[index]!),
      ...(candidate.mediaManifest
        ? { mediaManifest: candidate.mediaManifest }
        : {}),
    })),
    reports,
    replay: catalog.replayContext(),
  };
}

// Keeps the complete preceding non-checkpoint turn as context while the
// request stays within the halo limit. `replayDelta` is what a turn's
// protected reasoning adds to the request model's input beyond its visible
// text, which the halo choice counts too.
function retainPrecedingTurn(
  request: NativeGenerationRequest<GenerationCandidate>,
  plan: CompactionPlan,
  budget: NativeBatchBudget,
  replayDelta: (turn: Turn) => number,
): RetainedRequest {
  const { sourceTurns, sourceMessages } = plan;
  const turnIndex = request.candidate.sourceTurnIndex;
  const targetTurn = sourceTurns[turnIndex];
  if (!targetTurn) {
    throw new Error(
      "Compaction generation target turn is missing from source order.",
    );
  }
  if (turnHasNativeCheckpointArtifact(targetTurn)) {
    throw new Error(
      "Compaction generation cannot safely isolate a native checkpoint target.",
    );
  }
  const retainedTurns = [targetTurn];
  let estimatedInputTokens = request.estimatedInputTokens;
  let inputLimit = request.inputLimit;
  const precedingTurn = sourceTurns[turnIndex - 1];
  if (precedingTurn && !turnHasNativeCheckpointArtifact(precedingTurn)) {
    const precedingCost =
      estimateNativeTurnCost(precedingTurn).inputTokens
      + replayDelta(precedingTurn);
    if (estimatedInputTokens + precedingCost <= budget.haloInputLimit) {
      estimatedInputTokens += precedingCost;
      inputLimit = budget.haloInputLimit;
      retainedTurns.push(precedingTurn);
    }
  }

  const retainedMessages = new Set(retainedTurns.flatMap(turnMessages));
  const targetUserMessageIndex = targetTurn.user[0]
    ? sourceMessages.indexOf(targetTurn.user[0])
    : -1;
  if (targetUserMessageIndex < 0) {
    throw new Error(
      "Compaction generation target message is missing from source order.",
    );
  }
  return {
    ...request,
    estimatedInputTokens,
    inputLimit,
    retainedMessageIndexes: sourceMessages.flatMap((message, index) =>
      retainedMessages.has(message) ? [index] : [],
    ),
    targetUserMessageIndex,
  };
}

// Renders the request under its real nonce and checks it against the
// conservative estimate and the model's limits. A real request larger than
// its estimate is a protocol defect, not a capacity outcome.
function finalizeGenerationBatch(
  request: RetainedRequest,
  budget: NativeBatchBudget,
  globalTurnIndexPrompt: string,
  generationNonce: string,
): Fit<Omit<GenerationBatch, "settings">> {
  const { candidate } = request;
  const target: GenerationTarget = {
    ordinal: candidate.ordinal,
    marker: generationTurnMarker(generationNonce, candidate.ordinal),
    expectedInlineImages: candidate.expectedInlineImages,
  };
  const exactInputTokens =
    request.estimatedInputTokens
    - request.estimatedExplicitRequestTokens
    + estimateGenerationExplicitRequestTokens(target, globalTurnIndexPrompt);
  const exactOutputTokens =
    Math.ceil(candidate.outputTokens)
    + estimateGenerationResponseProtocolTokens(target);
  if (exactInputTokens > request.estimatedInputTokens) {
    throw new Error(
      "Magic Compact exact generation request exceeds its conservative pre-fork token estimate.",
    );
  }
  if (exactOutputTokens > request.estimatedOutputTokens) {
    throw new Error(
      "Magic Compact exact generation response exceeds its conservative pre-fork token estimate.",
    );
  }
  if (
    exactInputTokens > request.inputLimit
    || exactInputTokens > budget.usableInput
  ) {
    return refuse(
      "Magic Compact exact generation request cannot fit the configured model input. Run /magic-trim first and retry.",
    );
  }
  if (exactOutputTokens > budget.outputTarget) {
    return refuse(outputCapacityMessage(budget));
  }
  return fits({
    target,
    generationNonce,
    prompt: buildGenerationPrompt(target, globalTurnIndexPrompt),
    retainedMessageIndexes: request.retainedMessageIndexes,
    targetUserMessageIndex: request.targetUserMessageIndex,
  });
}

function turnMessages(turn: Turn): MessageWithParts[] {
  return [...turn.user, ...turn.assistants];
}

// One summary request's reasoning report over the history it sends, with the
// protected units whose replay cost its input estimate leaves out.
export function stageReport(request: {
  stage: StageReport["stage"];
  candidate: MatchCandidate;
  choice: StageChoice;
  settings: CompactionSettings;
  history: readonly MessageWithParts[];
  catalog: Pick<SummarizerCatalog, "qualifies" | "requestContext">;
  replay: ReplayIndex;
}): StageReport {
  const model = request.settings.model;
  return sharedStageReport({
    stage: request.stage,
    candidate: request.candidate,
    choice: request.choice,
    model,
    units: reasoningUnits(request.history),
    qualifies: request.catalog.qualifies,
    replayUnknown: request.replay.estimate(
      request.catalog.requestContext(model),
      new Set(request.history.map(message => message.info.id)),
    ).unknown,
  });
}

function turnHasNativeCheckpointArtifact(turn: Turn | undefined): boolean {
  if (!turn) return false;
  return [...turn.user, ...turn.assistants].some(
    message =>
      (message.info.role === "assistant" && message.info.summary === true)
      || message.parts.some(part => part.type === "compaction"),
  );
}

// One generation batch: its fork is created and registered at once, then
// populated, isolated to the request's retained turns, prompted once and
// deleted. `stop` is called as soon as a failure is observed, before the fork
// is aborted or deleted, so no further batch starts while this one cleans up.
// A fork whose deletion failed stays registered, so its guards stay in place,
// and its failure names it.
async function requestCompactionGeneration(
  v2: V2Client,
  sourceSession: Session,
  batch: GenerationBatch,
  plan: CompactionPlan,
  sourceMessages: MessageWithParts[],
  controlTimeoutMs: number,
  stop: () => void,
): Promise<GenerationTurnResult> {
  const route = sessionRoute(sourceSession);
  let fork: ForkCut;
  try {
    fork = await createForkCut(v2, sourceSession, sourceMessages);
  } catch (error) {
    stop();
    throw forkFailure(error);
  }

  const owned = registerOwnedFork(fork.id);
  const failures: unknown[] = [];
  let prompted = false;
  let result: GenerationTurnResult | undefined;
  try {
    await fork.populate(controlTimeoutMs).catch((error: unknown) => {
      throw forkFailure(error);
    });
    await isolateGenerationFork(
      v2,
      fork.id,
      route,
      batch,
      plan,
      controlTimeoutMs,
    );
    prompted = true;
    result = await promptGenerationFork(v2, owned, route, batch);
  } catch (error) {
    failures.push(error);
    stop();
  }
  // Only a prompt can leave the fork running.
  if (prompted && failures.length > 0) {
    try {
      unwrapLifecycleResponse(
        await v2.session.abort({ sessionID: fork.id, ...route }),
      );
    } catch (error) {
      failures.push(lifecycleFailure("abort", "rejection", error));
    }
  }
  try {
    unwrapLifecycleResponse(
      await v2.session.delete({ sessionID: fork.id, ...route }),
    );
    releaseOwnedFork(owned);
  } catch (error) {
    failures.push(
      new GenerationLifecycleError(
        "delete",
        safeStatus(error),
        lifecycleMessage("delete", "rejection"),
        [{ id: fork.id, removal: "failed" }],
      ),
    );
    stop();
  }
  const [first, ...others] = failures;
  if (others.length > 0) {
    throw ownedAggregate(
      failures,
      "Compaction generation and fork cleanup both failed.",
    );
  }
  if (first !== undefined) throw first;
  return result!;
}

// Leaves the fork holding exactly the request's retained turns, checked
// record for record against the plan, with the target marker inserted.
async function isolateGenerationFork(
  v2: V2Client,
  forkID: string,
  route: HistoryRoute,
  batch: GenerationBatch,
  plan: CompactionPlan,
  controlTimeoutMs: number,
): Promise<void> {
  await control(controlTimeoutMs, "permission", signal =>
    v2.session.update(
      { sessionID: forkID, ...route, permission: GENERATION_PERMISSION },
      { signal },
    ),
  );
  const forked = await control(controlTimeoutMs, "history", signal =>
    v2.session.messages({ sessionID: forkID, ...route }, { signal }),
  );
  const retainedIdentity = checkedHistory(() => {
    assertForkMessageShape(forked, plan);
    assertForkRecordIdentity(forked, forkID);
    return batch.retainedMessageIndexes.map(index =>
      forkMessageIdentity(forked[index]!),
    );
  });
  const retainedIndexes = new Set(batch.retainedMessageIndexes);
  for (const [index, message] of forked.entries()) {
    if (retainedIndexes.has(index)) continue;
    await control(controlTimeoutMs, "isolation", signal =>
      v2.session.deleteMessage(
        { sessionID: forkID, messageID: message.info.id, ...route },
        { signal },
      ),
    );
  }
  const retained = await control(controlTimeoutMs, "history", signal =>
    v2.session.messages({ sessionID: forkID, ...route }, { signal }),
  );
  const marker = checkedHistory(() => {
    assertRetainedForkMessageShape(
      retained,
      plan,
      batch.retainedMessageIndexes,
    );
    assertForkRecordIdentity(retained, forkID);
    assertRetainedForkIdentity(retained, retainedIdentity);
    if (containsString(retained, batch.generationNonce)) {
      throw new GenerationMarkerNonceCollisionError();
    }
    return generationMarkerTarget(retained, forkID, batch);
  });
  await control(controlTimeoutMs, "marker", signal =>
    v2.part.update(
      {
        sessionID: forkID,
        messageID: marker.message.info.id,
        partID: marker.part.id,
        ...route,
        part: marker.part,
      },
      { signal },
    ),
  );
}

// The one model request, awaited without a local deadline, and its untrusted
// structured response. A native compaction the hook blocked means the request
// outgrew the model, whatever error the host then returns for the prompt.
async function promptGenerationFork(
  v2: V2Client,
  fork: OwnedFork,
  route: HistoryRoute,
  batch: GenerationBatch,
): Promise<GenerationTurnResult> {
  let body: unknown;
  try {
    body = unwrapLifecycleResponse(
      await v2.session.prompt({
        sessionID: fork.id,
        ...route,
        agent: MAGIC_COMPACT_SUMMARIZER_AGENT,
        model: batch.settings.model,
        ...(batch.settings.variant ? { variant: batch.settings.variant } : {}),
        format: GENERATION_FORMAT,
        parts: [{ type: "text", text: batch.prompt }],
      }),
    );
  } catch (error) {
    if (fork.nativeCompactionBlocked) throw overflowFailure();
    throw lifecycleFailure("provider", "rejection", error);
  }
  if (fork.nativeCompactionBlocked) throw overflowFailure();
  const responseInfo =
    isRecord(body) && isRecord(body["info"]) ? body["info"] : undefined;
  const structuredOutput = responseInfo?.["structured"];
  if (responseInfo) {
    const responseError = responseInfo["error"];
    const outputTruncated = isOutputTruncation(responseInfo["finish"]);
    // A response cut at its output limit still counts when its structured
    // output was captured.
    if (
      responseError
      && !(outputTruncated && isStructuredOutputError(responseError))
    ) {
      throw lifecycleFailure("provider", "rejection", responseError);
    }
    if (outputTruncated && structuredOutput === undefined) {
      throw new GenerationLifecycleError(
        "provider",
        undefined,
        GENERATION_OUTPUT_LIMIT,
      );
    }
  }
  try {
    return parseGenerationResponse(
      structuredOutput,
      batch.target,
      batch.generationNonce,
    );
  } catch (error) {
    throw lifecycleFailure("validation", "rejection", error);
  }
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

function generationMarkerTarget(
  messages: MessageWithParts[],
  forkSessionID: string,
  batch: GenerationBatch,
): { message: MessageWithParts; part: TextPart } {
  const retainedIndex = batch.retainedMessageIndexes.indexOf(
    batch.targetUserMessageIndex,
  );
  const message = messages[retainedIndex];
  if (retainedIndex < 0 || !message || message.info.role !== "user") {
    throw new Error("Compaction generation fork history shape mismatch.");
  }
  const part = generationMarkerPart(
    forkSessionID,
    message.info.id,
    batch.target.marker,
    batch.target.ordinal,
  );
  if (message.parts.some(existing => existing.id === part.id)) {
    throw new Error("Compaction generation fork marker collision.");
  }
  return { message, part };
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

// The configured summarizer plus the live model list. `describe` gives another
// listed model's descriptor for `--match-model`, `qualifies` says whether a
// model's protected reasoning behavior is verified, `requestContext` is the
// reasoning context a summary request on a model uses, and `replayContext`
// names the model the successor continues with and the reasoning context its
// requests use.
export type SummarizerCatalog = {
  configured: CompactionSettings;
  // The summarizer agent carries operator options written for its own model.
  summarizerOptions: boolean;
  describe: (model: ModelIdentity) => ModelDescriptor | undefined;
  qualifies: Qualifies;
  requestContext: (model: ModelIdentity) => ReplayContext;
  replayContext: () => ReplayContext | undefined;
};

// One command's summarizer catalog. The agent list and the model list are
// each read at most once, when a summary request first needs them, so the
// retention summary and every turn request share one resolution and a
// command that makes no summary request reads neither. `requiresImages` only
// chooses the failure text for its caller.
export type LoadSummarizerCatalog = (
  requiresImages: boolean,
) => Promise<SummarizerCatalog>;

export function summarizerCatalogLoader(
  v2: V2Client,
  sourceSession: Session,
): LoadSummarizerCatalog {
  let agents: Promise<SummarizerAgents> | undefined;
  let providers: Promise<unknown[] | undefined> | undefined;
  return async requiresImages => {
    const read = await (agents ??= readSummarizerAgents(v2, sourceSession));
    const listed = await (providers ??= readProviders(v2, sourceSession));
    if (listed === undefined) {
      throw new PluginMessageError(
        requiresImages
          ? "Magic Compact could not verify an image-capable configured compaction model."
          : "Magic Compact could not resolve the configured compaction model descriptor.",
      );
    }
    return summarizerCatalog(sourceSession, read, listed, requiresImages);
  };
}

type SummarizerAgents = {
  agents: Agent[];
  summarizer: Agent & { model: ModelIdentity };
};

async function readSummarizerAgents(
  v2: V2Client,
  sourceSession: Session,
): Promise<SummarizerAgents> {
  let agents: Agent[];
  try {
    agents = unwrap(await v2.app.agents(sessionRoute(sourceSession)));
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
  const model = summarizer.model;
  if (!model || !model.providerID.trim() || !model.modelID.trim()) {
    throw new PluginMessageError(SUMMARIZER_MODEL_REQUIRED);
  }
  return { agents, summarizer: { ...summarizer, model } };
}

// The listed providers, or undefined when the list could not be read.
async function readProviders(
  v2: V2Client,
  sourceSession: Session,
): Promise<unknown[] | undefined> {
  let catalog: ReturnType<typeof unwrap>;
  try {
    catalog = unwrap(await v2.provider.list(sessionRoute(sourceSession)));
  } catch {
    return undefined;
  }
  return isRecord(catalog) && Array.isArray(catalog["all"])
    ? catalog["all"]
    : [];
}

function summarizerCatalog(
  sourceSession: Session,
  { agents, summarizer }: SummarizerAgents,
  providers: unknown[],
  requiresImages: boolean,
): SummarizerCatalog {
  const providerRecord = (providerID: string) =>
    providers.find(value => isRecord(value) && value["id"] === providerID) as
      Record<string, unknown> | undefined;
  const modelRecord = (model: ModelIdentity) => {
    const models = providerRecord(model.providerID)?.["models"];
    const record = isRecord(models) ? models[model.modelID] : undefined;
    return isRecord(record) ? record : undefined;
  };
  const describe = (model: ModelIdentity) =>
    modelDescriptor(modelRecord(model));
  const descriptor = describe(summarizer.model);
  if (!descriptor) {
    throw new PluginMessageError(
      requiresImages
        ? "Magic Compact requires an image-capable configured compaction model for retained fork history containing images."
        : "Magic Compact could not resolve the configured compaction model descriptor.",
    );
  }
  const options = (value: unknown) =>
    isRecord(value) && isRecord(value["options"]) ? value["options"] : {};
  const verified = (model: ModelIdentity) =>
    verifiedReasoning(v1ReasoningRoute(model, describe(model)));
  return {
    configured: {
      model: summarizer.model,
      ...(summarizer.variant ? { variant: summarizer.variant } : {}),
      descriptor,
    },
    summarizerOptions: Object.keys(options(summarizer)).length > 0,
    describe,
    qualifies: model => verified(model) !== undefined,
    // The `chat.params` hook asks for `all_turns` on owned summary requests on
    // a verified model, the costlier context.
    requestContext: model => ({
      primary: model,
      mode: verified(model) ? "all_turns" : "unknown",
    }),
    replayContext: () => {
      const primary = sourceSession.model;
      if (!primary?.providerID || !primary.id) return undefined;
      const model = { providerID: primary.providerID, modelID: primary.id };
      const record = modelRecord(model);
      const facts = verified(model);
      if (!record || !facts) return { primary: model, mode: "unknown" };
      const variants = record["variants"];
      const variant =
        primary.variant !== undefined && isRecord(variants)
          ? variants[primary.variant]
          : undefined;
      const agent = agents.find(item => item.name === sourceSession.agent);
      return {
        primary: model,
        mode: reasoningContextMode(
          [
            options(providerRecord(model.providerID)),
            options(record),
            isRecord(variant) ? variant : {},
            options(agent),
          ].map(value => value["reasoningContext"]),
          facts,
        ),
      };
    },
  };
}

export type ModelDescriptor = CompactionSettings["descriptor"];

function modelDescriptor(
  record: Record<string, unknown> | undefined,
): ModelDescriptor | undefined {
  const limit = record?.["limit"];
  if (!record || !isRecord(limit) || typeof limit["context"] !== "number") {
    return undefined;
  }
  const capabilities = record["capabilities"];
  const inputCapabilities = isRecord(capabilities)
    ? capabilities["input"]
    : undefined;
  const interleaved = isRecord(capabilities)
    ? capabilities["interleaved"]
    : undefined;
  const api = record["api"];
  const variants = record["variants"];
  return {
    limit: {
      context: limit["context"],
      ...(typeof limit["input"] === "number" ? { input: limit["input"] } : {}),
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
    ...(isRecord(api) && typeof api["npm"] === "string"
      ? { npm: api["npm"] }
      : {}),
    ...(isRecord(api) && typeof api["id"] === "string"
      ? { wireModelID: api["id"] }
      : {}),
    ...(isRecord(interleaved) && typeof interleaved["field"] === "string"
      ? { reasoningField: interleaved["field"] }
      : {}),
    variants: isRecord(variants) ? Object.keys(variants) : [],
  };
}

// The route facts of an OpenCode 1 model descriptor: OpenCode 1 serializes
// `@ai-sdk/openai` models with OpenAI Responses and sends `api.id` as the
// wire model.
function v1ReasoningRoute(
  model: ModelIdentity,
  descriptor: Pick<ModelDescriptor, "npm" | "wireModelID"> | undefined,
): ReasoningRoute | undefined {
  if (!descriptor) return undefined;
  return {
    model,
    wireModelID: descriptor.wireModelID,
    responses: descriptor.npm === "@ai-sdk/openai",
  };
}

// The DeepSeek route whose earlier plain reasoning OpenCode 1 sends back as
// `reasoning_content`, which DeepSeek reads only when the request has tools.
export function isQualifiedDeepSeekModel(
  model: ModelIdentity,
  descriptor: ModelDescriptor | undefined,
): boolean {
  return (
    model.providerID === "deepseek"
    && descriptor?.reasoningField === "reasoning_content"
  );
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

// A turn is anchored by its first ordinary (non-synthetic, non-ignored) text
// part across its user messages, even when that text is empty.
function buildGlobalTurnIndexPrompt(plan: CompactionPlan): string {
  return globalTurnIndexPrompt(
    plan.sourceTurns
      .slice(
        plan.effectiveSourceTurnStartIndex,
        plan.effectiveSourceTurnEndIndex,
      )
      .map((turn, index) => ({
        sourceOrdinal: plan.effectiveSourceTurnStartIndex + index + 1,
        userText:
          turn.user
            .flatMap(message => message.parts)
            .find(
              (part): part is TextPart =>
                part.type === "text"
                && part.synthetic !== true
                && part.ignored !== true,
            )?.text ?? "",
      })),
  );
}

// Successor edits apply to the whole source, so a generated summary or
// boundary part ID must not already be taken anywhere in it.
function assertGeneratedPartIDsAvailable(plan: CompactionPlan): void {
  const partIDs = new Set(
    plan.sourceMessages.flatMap(message => message.parts.map(part => part.id)),
  );
  for (const { turn } of plan.targets) {
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

const GENERATION_OVERFLOW =
  "The complete forked session exceeded the configured compaction model context. Run /magic-trim first, then retry /magic-compact.";
const GENERATION_OUTPUT_LIMIT =
  "The compaction model reached its output limit while summarizing a single turn. Configure a compaction model with more output capacity.";

// A generation failure with fixed text, its stage and the classified HTTP
// status; never a host or provider body. A failed deletion also names the
// fork it left.
class GenerationLifecycleError
  extends PluginMessageError
  implements LeavesSessions
{
  constructor(
    readonly stage: GenerationStage,
    readonly status: number | undefined,
    readonly reason: string,
    readonly unremovedSessions: readonly UnremovedSession[] = [],
  ) {
    super(
      unremovedSessions.length === 0
        ? reason
        : `${reason} ${unremovedSessionsText(unremovedSessions, "Fork")}`,
    );
    this.name = "GenerationLifecycleError";
  }
}

// A fork that could not be created, or populated as planned.
function forkFailure(error: unknown): GenerationLifecycleError {
  return error instanceof ForkHistoryError
    ? lifecycleFailure(
        "history",
        error.reason === "read-timeout" ? "timeout" : "rejection",
        error,
      )
    : lifecycleFailure("fork", "rejection", error);
}

function overflowFailure(): GenerationLifecycleError {
  return new GenerationLifecycleError(
    "provider",
    undefined,
    GENERATION_OVERFLOW,
  );
}

// One control request under the deadline. A timeout or rejection is this
// stage's failure.
async function control<T>(
  controlTimeoutMs: number,
  stage: Extract<
    GenerationStage,
    "permission" | "history" | "isolation" | "marker"
  >,
  request: (signal: AbortSignal) => Promise<{ data?: T; error?: unknown }>,
): Promise<T> {
  const outcome = await attemptWithinDeadline(
    signal => request(signal).then(unwrapLifecycleResponse),
    controlTimeoutMs,
  );
  if (outcome.kind === "success") return outcome.value;
  throw lifecycleFailure(
    stage,
    outcome.kind === "timed-out" ? "timeout" : "rejection",
    outcome.kind === "failed" ? outcome.error : undefined,
  );
}

// A check of fork history the host returned; any failure is a history
// failure.
function checkedHistory<T>(check: () => T): T {
  try {
    return check();
  } catch (error) {
    throw lifecycleFailure("history", "rejection", error);
  }
}

function lifecycleFailure(
  stage: GenerationStage,
  kind: LifecycleFailureKind,
  error?: unknown,
): GenerationLifecycleError {
  const status = safeStatus(error);
  if (error instanceof ForkHistoryError) {
    return new GenerationLifecycleError(
      stage,
      status,
      `Compaction generation fork history verification ${kind === "timeout" ? "timed out" : "failed"} (${error.detail}).`,
    );
  }
  const provider = stage === "provider";
  const message =
    error instanceof GenerationMarkerNonceCollisionError
      ? error.message
      : stage === "validation"
          && error instanceof GenerationResponseValidationError
        ? `Compaction generation response validation failed: ${validationCategoryText(error)}.`
        : provider && isContextOverflow(error, status)
          ? GENERATION_OVERFLOW
          : provider && isOutputTruncation(error)
            ? GENERATION_OUTPUT_LIMIT
            : provider && isStructuredOutputError(error)
              ? "The compaction model finished without submitting its StructuredOutput response."
              : lifecycleMessage(stage, kind);
  return new GenerationLifecycleError(stage, status, message);
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

// An aggregate of this plugin's already sanitized generation failures. Its
// message adds their distinct reasons, so a person sees why the batches
// failed (for example, to run /magic-trim first) and not only that several
// did, and then every fork that could not be deleted, which the reason cap
// never drops. The failures stay in `errors`, unchanged and in batch order.
function ownedAggregate(
  errors: readonly unknown[],
  summary: string,
): AggregateError {
  const forks = unremovedSessions(errors);
  const message = [
    summary,
    ...distinctReasons(errors),
    ...(forks.length === 0 ? [] : [unremovedSessionsText(forks, "Fork")]),
  ].join(" ");
  return markPluginMessage(
    new AggregateError(errors, message, { cause: errors[0] }),
  );
}

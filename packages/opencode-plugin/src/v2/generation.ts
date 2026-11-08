import { isDeepStrictEqual } from "node:util";
import {
  computeNativeBatchBudget,
  estimateTurnOutputTokens,
  fits,
  NATIVE_BATCH_PLANNING_NONCE,
  planNativeGenerationRequest,
  refuse,
  requireFit,
  type Fit,
} from "../policy/budget";
import { MAGIC_COMPACT_SUMMARIZER_AGENT } from "../policy/protocol";
import {
  buildGenerationPrompt,
  createGenerationNonce,
  estimateGenerationExplicitRequestTokens,
  estimateGenerationResponseProtocolTokens,
  frameSummary,
  generationTurnMarker,
  GenerationResponseValidationError,
  parseGenerationResponse,
  selectGenerationNonce,
  validationCategoryText,
  validationFailure,
  type GenerationTarget,
} from "../policy/generation";
import {
  distinctReasons,
  PluginMessageError,
  unremovedSessions,
  unremovedSessionsText,
  type LeavesSessions,
  type UnremovedSession,
} from "../policy/feedback";
import { runGenerationBatches } from "../policy/generation-pool";
import { attemptWithinDeadline, jsonForm } from "../util";
import type { V2Host } from "./host";
import {
  v2HistoryShowsImages,
  v2MediaManifest,
  v2UnprojectedImages,
  type V2MediaManifest,
} from "./media";
import { removeOwnedSession, type OwnedSessions } from "./owned";
import {
  readableReasoningHeaderTokens,
  v2ReasoningUnits,
  v2ReplayEstimate,
} from "./reasoning";
import {
  chooseStage,
  matchCandidate,
  settleMatch,
  stageReport,
  type StageReport,
} from "../policy/reasoning";
import {
  mintV2ID,
  readV2History,
  withFreshMessageIDs,
  type V2Message,
  type V2Source,
} from "./session";
import {
  estimateV2HistoryTokens,
  matchedV2Model,
  V2_IMAGE_MODEL_REQUIRED,
  v2Configured,
  v2MatchPreflight,
  v2Qualifies,
  v2SummarizerRoute,
  type V2SummarizerModel,
} from "./summarizer";

// The response tool every generation request may call. Its name matches the
// OpenCode 1 host tool, so both hosts send the same protocol text.
export const V2_RESPONSE_TOOL = "StructuredOutput";

const GENERATION_PERMISSIONS = [
  { action: "*", resource: "*", effect: "deny" as const },
  { action: V2_RESPONSE_TOOL, resource: "*", effect: "allow" as const },
];

// One target turn and the complete preceding turn that may join it as
// context. `span` holds every record from the turn's first user message up to
// the next turn. Later turns never enter a generation request.
export type V2GenerationTurn = {
  span: V2Message[];
  previous?: V2Message[];
};

export type V2GeneratedSummary = {
  text: string;
  mediaManifest?: V2MediaManifest;
};

type Candidate = GenerationTarget & {
  inputTokens: number;
  outputTokens: number;
};

type Batch = {
  mediaManifest?: V2MediaManifest;
  history: V2Message[];
  nonce: string;
  target: GenerationTarget;
  prompt: string;
  // The model this request runs on.
  summarizer: V2SummarizerModel;
};

// `reason` says what failed without naming a temporary session left behind,
// so an aggregate can explain failures once and still name every session.
export class V2GenerationError
  extends PluginMessageError
  implements LeavesSessions
{
  constructor(
    readonly reason: string,
    readonly unremovedSessions: readonly UnremovedSession[] = [],
  ) {
    super(
      unremovedSessions.length === 0
        ? reason
        : `${reason} ${unremovedSessionsText(unremovedSessions, "Temporary session")}`,
    );
    this.name = "V2GenerationError";
  }
}

export const V2_IMAGE_LIMIT_EXCEEDED =
  "The images in a compaction request exceed the 25 MiB OpenCode sends to a model, so the summarizer would not see every image the summary must describe. Run /magic-trim or start a new session for this history.";

// What OpenCode 2.0.28 does to request media after the plugin's hooks
// (`session/model-request.ts`): an image the model cannot take, or one xAI
// does not accept, becomes an error text, and over 25 MiB of image data
// drops the oldest images. A summary has to have seen every image its
// manifest binds, so such a request fails before any session is created.
const HOST_IMAGE_BYTES_LIMIT = 25 * 1024 * 1024;
const XAI_IMAGE_MIMES = new Set(["image/png", "image/jpeg", "image/webp"]);
type RequestImage = ReturnType<typeof v2UnprojectedImages>[number];

// Whether the model takes every image the request still shows. A matched
// model's preflight already required image input over a history that holds
// this request's, and only OpenAI routes are matched, so only the configured
// summarizer can fail here.
function assertImagesReachModel(
  images: readonly RequestImage[],
  model: V2SummarizerModel,
): void {
  if (images.length === 0) return;
  const xai = model.info.providerID === "xai" || model.info.canonical === "xai";
  if (
    !model.info.capabilities.input.some(input => input.startsWith("image"))
    || (xai
      && images.some(image => !XAI_IMAGE_MIMES.has(image.mime.toLowerCase())))
  ) {
    throw new V2GenerationError(V2_IMAGE_MODEL_REQUIRED);
  }
}

// Whether the request's images stay within what OpenCode sends: past 25 MiB
// the host drops the oldest, so the summarizer would miss an image the
// target's manifest binds. The preceding turn joins only within a model's
// halo limit, so this depends on the model's plan.
function imagesWithinHostLimit(images: readonly RequestImage[]): boolean {
  return (
    images.reduce((total, image) => total + image.bytes, 0)
    <= HOST_IMAGE_BYTES_LIMIT
  );
}

const OVERFLOW =
  "The compaction request exceeded the configured compaction model context. Run /magic-trim first, then retry /magic-compact.";

// Plans every target before any session is created: the summary model, its
// budgets, the single target per request, the optional preceding turn within
// the halo limit, and the real nonce and prompt, checked against the
// planning estimate. Under `--match-model` a target whose own protected
// reasoning comes from one verified model runs on that model, unless the
// preflight or that model's capacity sends it back to the configured one.
function planBatches(
  turns: V2GenerationTurn[],
  model: V2SummarizerModel,
  globalIndexPrompt: string,
  nonceFactory: () => string,
  matchModel: boolean,
): { batches: Batch[]; reports: StageReport[] } {
  const nonces = new Set<string>();
  const qualifies = v2Qualifies(model);
  const planOne = (
    turn: V2GenerationTurn,
    index: number,
    summarizer: V2SummarizerModel,
    nonce: string,
  ): Fit<Batch> => {
    const budget = computeNativeBatchBudget(summarizer.info.limit);
    if (budget.kind === "capacity") return budget;
    const destination = v2SummarizerRoute(summarizer);
    const estimate = estimateV2HistoryTokens(turn.span);
    const mediaManifest = v2MediaManifest(turn.span);
    const ordinal = index + 1;
    const candidate: Candidate = {
      ordinal,
      marker: generationTurnMarker(NATIVE_BATCH_PLANNING_NONCE, ordinal),
      expectedInlineImages: mediaManifest?.images.length ?? 0,
      inputTokens:
        estimate.tokens + readableReasoningHeaderTokens(turn.span, destination),
      outputTokens: estimateTurnOutputTokens(
        estimate.textTokens,
        estimate.mediaCount,
      ),
    };
    const request = planNativeGenerationRequest(candidate, budget.value, {
      explicitRequestTokens: target =>
        estimateGenerationExplicitRequestTokens(target, globalIndexPrompt),
      responseProtocolTokens: estimateGenerationResponseProtocolTokens,
    });
    if (request.kind === "capacity") return request;
    const previous =
      turn.previous
      && request.value.estimatedInputTokens
        + estimateV2HistoryTokens(turn.previous).tokens
        + readableReasoningHeaderTokens(turn.previous, destination)
        <= budget.value.haloInputLimit
        ? turn.previous
        : undefined;
    const target: GenerationTarget = {
      ordinal: candidate.ordinal,
      marker: generationTurnMarker(nonce, candidate.ordinal),
      expectedInlineImages: candidate.expectedInlineImages,
    };
    if (
      estimateGenerationExplicitRequestTokens(target, globalIndexPrompt)
        > request.value.estimatedExplicitRequestTokens
      || estimateGenerationResponseProtocolTokens(target)
        > estimateGenerationResponseProtocolTokens(candidate)
    ) {
      throw new V2GenerationError(
        "Compaction generation request exceeded its planned size.",
      );
    }
    const firstUser = turn.span.findIndex(message => message.type === "user");
    const user = turn.span[firstUser]!;
    const marker = {
      id: user.id,
      type: "synthetic",
      text: target.marker,
      time: { created: user.time.created },
    } as V2Message;
    const history = [
      ...(previous ?? []),
      ...turn.span.slice(0, firstUser),
      marker,
      ...turn.span.slice(firstUser),
    ];
    const images = v2UnprojectedImages(history);
    assertImagesReachModel(images, summarizer);
    if (mediaManifest && !imagesWithinHostLimit(images)) {
      return refuse(V2_IMAGE_LIMIT_EXCEEDED);
    }
    return fits({
      ...(mediaManifest ? { mediaManifest } : {}),
      history,
      nonce,
      target,
      prompt: buildGenerationPrompt(target, globalIndexPrompt),
      summarizer,
    });
  };
  const reports: StageReport[] = [];
  const batches = turns.map((turn, index) => {
    // One nonce per target, absent from every history its request may send,
    // shared by the matched model's fit check and the request that runs.
    const nonce = selectGenerationNonce(
      [[...(turn.previous ?? []), ...turn.span]],
      nonceFactory,
      nonces,
    );
    // The target's own reasoning decides the match, never the halo's; the
    // image check covers the turn that may join it.
    const match = matchCandidate(v2ReasoningUnits(turn.span), qualifies);
    const { choice, matched } = settleMatch(
      chooseStage(match, v2Configured(model), candidate =>
        v2MatchPreflight(
          model,
          candidate,
          v2HistoryShowsImages([...(turn.previous ?? []), ...turn.span]),
        ),
      ),
      matchModel,
      (candidate, variant) =>
        planOne(turn, index, matchedV2Model(model, candidate, variant), nonce),
    );
    const batch = matched ?? requireFit(planOne(turn, index, model, nonce));
    // OpenCode 2 replay cost is never estimated, so the request's own
    // protected units are reported as left out of its input estimate.
    const requestModel = {
      providerID: batch.summarizer.ref.providerID,
      modelID: batch.summarizer.ref.id,
    };
    reports.push(
      stageReport({
        stage: "turn",
        candidate: match,
        choice,
        model: requestModel,
        units: v2ReasoningUnits(batch.history),
        qualifies,
        replayUnknown: v2ReplayEstimate(batch.history, requestModel).unknown,
      }),
    );
    return batch;
  });
  return { batches, reports };
}

// Generates one validated summary per target turn, in order. Every request
// runs in its own owned, deny-all temporary session that may call only the
// response tool and is removed afterwards. The first failure stops new
// requests; requests already running finish their own cleanup, and no result
// is returned unless every target passed.
export async function generateV2Summaries(
  host: V2Host,
  owned: OwnedSessions,
  source: V2Source,
  model: V2SummarizerModel,
  turns: V2GenerationTurn[],
  globalIndexPrompt: string,
  options: {
    controlTimeoutMs: number;
    nonceFactory?: () => string;
    matchModel?: boolean;
  },
): Promise<{ summaries: V2GeneratedSummary[]; reports: StageReport[] }> {
  if (turns.length === 0) return { summaries: [], reports: [] };
  const { batches, reports } = planBatches(
    turns,
    model,
    globalIndexPrompt,
    options.nonceFactory ?? createGenerationNonce,
    options.matchModel === true,
  );
  const outcome = await runGenerationBatches(batches, (batch, stop) =>
    runBatch(host, owned, source, batch, options.controlTimeoutMs, stop),
  );
  if (outcome.kind === "succeeded") {
    return { summaries: outcome.results, reports };
  }
  // `runBatch` reports every failure, its cleanup's included, as one
  // V2GenerationError.
  const errors = outcome.failures.map(error => {
    if (!(error instanceof V2GenerationError)) throw error;
    return error;
  });
  if (errors.length === 1) throw errors[0];
  throw new V2GenerationError(
    `Compaction generation failed for ${errors.length} turns: ${distinctReasons(errors).join(" ")}`,
    unremovedSessions(errors),
  );
}

// One owned session: imported, prompted once, checked and removed. `stop` is
// called as soon as a failure is observed, before the removal, so no further
// batch starts while this one cleans up. A failed removal fails the batch and
// names the session; a session the host no longer finds is named as not found
// only beside another failure.
async function runBatch(
  host: V2Host,
  owned: OwnedSessions,
  source: V2Source,
  batch: Batch,
  controlTimeoutMs: number,
  stop: () => void,
): Promise<V2GeneratedSummary> {
  const control = async <T>(
    failure: string,
    request: (signal: AbortSignal) => Promise<T>,
  ): Promise<T> => {
    const outcome = await attemptWithinDeadline(request, controlTimeoutMs);
    if (outcome.kind !== "success") throw new V2GenerationError(failure);
    return outcome.value;
  };
  const now = Date.now();
  const sessionID = mintV2ID("ses_", now, 1, true);
  const history = withFreshMessageIDs(batch.history, now);
  const promptID = mintV2ID("msg_", now, history.length + 1, false);
  const info = {
    ...structuredClone(source.session),
    id: sessionID,
    title: "Magic Compact generation",
    agent: MAGIC_COMPACT_SUMMARIZER_AGENT,
    model: batch.summarizer.ref,
    permissions: GENERATION_PERMISSIONS,
    metadata: {},
    time: { created: now, updated: now },
  } as Record<string, unknown>;
  delete info["parentID"];
  delete info["fork"];
  delete info["revert"];

  owned.register(
    sessionID,
    "generation",
    v2SummarizerRoute(batch.summarizer).responses,
  );
  let result: V2GeneratedSummary | undefined;
  let failure: V2GenerationError | undefined;
  try {
    // Directly awaited, like the removal: a timed-out import could still
    // create the session after cleanup.
    try {
      await host.session.import({
        info: info as never,
        messages: history as never,
        location: source.session.location,
      });
    } catch {
      throw new V2GenerationError(
        "Compaction generation session creation failed.",
      );
    }
    const read = <T>(request: (signal: AbortSignal) => Promise<T>) =>
      control("Compaction generation history verification failed.", request);
    if (
      !isDeepStrictEqual(
        jsonForm(await readV2History(host, sessionID, read)),
        jsonForm(history),
      )
    ) {
      throw new V2GenerationError(
        "Compaction generation history verification failed.",
      );
    }
    // No local deadline on the model request; the owned controls bound it to
    // one provider request.
    try {
      await host.session.prompt({
        sessionID,
        id: promptID,
        text: batch.prompt,
      });
      await host.session.wait({ sessionID });
    } catch {
      throw new V2GenerationError(
        "Compaction generation model request failed.",
      );
    }
    const after = await readV2History(host, sessionID, read);
    result = acceptResponse(owned, batch, history, promptID, after);
  } catch (error) {
    failure =
      error instanceof V2GenerationError
        ? error
        : new V2GenerationError("Compaction generation model request failed.");
    stop();
  }
  const unremoved = await removeOwnedSession(host, owned, sessionID);
  if (
    unremoved?.removal === "failed"
    || (unremoved !== undefined && failure !== undefined)
  ) {
    stop();
    failure = new V2GenerationError(
      failure?.reason ?? "Compaction generation session deletion failed.",
      [unremoved],
    );
  }
  if (failure) throw failure;
  return result!;
}

// The copied history, the prompt, one summarizer reply whose only tool call
// is a completed response-tool call, then the host's record of this plugin's
// own stop. The persisted call input keeps the model's raw arguments, so the
// shared strict validation sees exactly what the model sent.
function acceptResponse(
  owned: OwnedSessions,
  batch: Batch,
  history: V2Message[],
  promptID: string,
  after: V2Message[],
): V2GeneratedSummary {
  if (
    after.some(
      message =>
        (message.type === "idle" || message.type === "assistant")
        && message.error?.message.includes(owned.compactionBlocked),
    )
  ) {
    throw new V2GenerationError(OVERFLOW);
  }
  if (
    !isDeepStrictEqual(
      jsonForm(after.slice(0, history.length)),
      jsonForm(history),
    )
  ) {
    throw new V2GenerationError(
      "Compaction generation history verification failed.",
    );
  }
  const [prompt, reply, ...rest] = after.slice(history.length);
  if (
    prompt?.type !== "user"
    || prompt.id !== promptID
    || prompt.text !== batch.prompt
  ) {
    throw new V2GenerationError(
      "Compaction generation history verification failed.",
    );
  }
  if (reply?.type !== "assistant" || reply.error !== undefined) {
    const type = reply?.type === "assistant" ? reply.error?.type : undefined;
    throw new V2GenerationError(
      type && /overflow|too.long|context/iu.test(type)
        ? OVERFLOW
        : "Compaction generation model request failed.",
    );
  }
  try {
    const calls = reply.content.filter(item => item.type === "tool");
    if (
      reply.agent !== MAGIC_COMPACT_SUMMARIZER_AGENT
      || calls.length === 0
      || reply.content.some(
        item => item.type === "tool" && item.name !== V2_RESPONSE_TOOL,
      )
    ) {
      throw validationFailure("missing structured output");
    }
    if (calls.length !== 1) {
      throw validationFailure("turn count mismatch", {
        expected: 1,
        received: calls.length,
      });
    }
    const call = calls[0]!;
    if (
      call.state.status !== "completed"
      || rest.length !== 1
      || rest[0]!.type !== "idle"
      || rest[0]!.error?.message !== owned.stop
    ) {
      throw validationFailure("missing structured output");
    }
    const parsed = parseGenerationResponse(
      call.state.input,
      batch.target,
      batch.nonce,
    );
    return {
      text: frameSummary(parsed),
      ...(batch.mediaManifest ? { mediaManifest: batch.mediaManifest } : {}),
    };
  } catch (error) {
    if (error instanceof GenerationResponseValidationError) {
      throw new V2GenerationError(
        `Compaction generation response validation failed: ${validationCategoryText(error)}.`,
      );
    }
    throw error;
  }
}

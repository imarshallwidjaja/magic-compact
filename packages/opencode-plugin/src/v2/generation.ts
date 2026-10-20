import type { ModelInfo, ModelRef } from "@opencode/client";
import { isDeepStrictEqual } from "node:util";
import {
  computeNativeBatchBudget,
  estimateTurnOutputTokens,
  NATIVE_BATCH_MAX_CONCURRENCY,
  NATIVE_BATCH_MEDIA_TOKEN_CHARGE,
  NATIVE_BATCH_PLANNING_NONCE,
  planNativeGenerationBatches,
  type NativeBatchBudget,
} from "../compact/batch";
import { MAGIC_COMPACT_SUMMARIZER_AGENT } from "../compact/constants";
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
  MAX_REASONS,
  PluginMessageError,
  undeletedSessionIDs,
  type LeavesSessions,
} from "../policy/feedback";
import { countGptTokens } from "../stats/tokenize";
import { attemptWithinDeadline } from "../util";
import type { V2Host } from "./host";
import {
  planV2MediaProjection,
  v2MediaManifest,
  v2UnprojectedImages,
  type V2MediaManifest,
} from "./media";
import { removeOwnedSession, type OwnedSessions } from "./owned";
import {
  mintV2ID,
  readV2History,
  withFreshMessageIDs,
  type V2Message,
  type V2Source,
} from "./session";

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

export type V2SummarizerModel = {
  ref: ModelRef;
  info: Pick<ModelInfo, "limit" | "capabilities" | "providerID" | "canonical">;
};

type Candidate = GenerationTarget & {
  inputTokens: number;
  outputTokens: number;
  turn: V2GenerationTurn;
  mediaManifest?: V2MediaManifest;
};

type Batch = {
  candidate: Candidate;
  history: V2Message[];
  nonce: string;
  target: GenerationTarget;
  prompt: string;
};

export class V2GenerationError
  extends PluginMessageError
  implements LeavesSessions
{
  // `reason` says what failed without naming a temporary session left behind,
  // so an aggregate can explain failures once and still name every session.
  constructor(
    message: string,
    readonly reason: string = message,
    readonly undeletedSessionIDs: readonly string[] = [],
  ) {
    super(message);
    this.name = "V2GenerationError";
  }
}

export const V2_IMAGE_MODEL_REQUIRED =
  "Magic Compact requires an image-capable configured compaction model for history containing images.";

export const V2_IMAGE_LIMIT_EXCEEDED =
  "The images in a compaction request exceed the 25 MiB OpenCode sends to a model, so the summarizer would not see every image the summary must describe. Run /magic-trim or start a new session for this history.";

// What OpenCode 2.0.28 does to request media after the plugin's hooks
// (`session/model-request.ts`): an image the model cannot take, or one xAI
// does not accept, becomes an error text, and over 25 MiB of image data
// drops the oldest images. A summary has to have seen every image its
// manifest binds, so such a request fails before any session is created.
const HOST_IMAGE_BYTES_LIMIT = 25 * 1024 * 1024;
const XAI_IMAGE_MIMES = new Set(["image/png", "image/jpeg", "image/webp"]);

function assertImagesReachModel(batch: Batch, model: V2SummarizerModel): void {
  const images = v2UnprojectedImages(batch.history);
  if (images.length === 0) return;
  const xai = model.info.providerID === "xai" || model.info.canonical === "xai";
  if (
    !model.info.capabilities.input.some(input => input.startsWith("image"))
    || (xai
      && images.some(image => !XAI_IMAGE_MIMES.has(image.mime.toLowerCase())))
  ) {
    throw new V2GenerationError(V2_IMAGE_MODEL_REQUIRED);
  }
  if (
    batch.candidate.mediaManifest
    && images.reduce((total, image) => total + image.bytes, 0)
      > HOST_IMAGE_BYTES_LIMIT
  ) {
    throw new V2GenerationError(V2_IMAGE_LIMIT_EXCEEDED);
  }
}

const OVERFLOW =
  "The compaction request exceeded the configured compaction model context. Run /magic-trim first, then retry /magic-compact.";

// Model-visible input estimate of OpenCode 2 records: text through the local
// tokenizer, inline image and PDF files at the fixed media charge unless a
// manifest projects the image to its placeholder, and other files by their
// encoded length.
export function estimateV2HistoryTokens(messages: readonly V2Message[]): {
  tokens: number;
  textTokens: number;
  mediaCount: number;
} {
  let tokens = 0;
  let textTokens = 0;
  let mediaCount = 0;
  const text = (value: string | undefined) => {
    if (!value) return;
    const count = countGptTokens(value);
    tokens += count;
    textTokens += count;
  };
  const projection = planV2MediaProjection(messages);
  for (const message of messages) {
    switch (message.type) {
      case "user": {
        text(message.text);
        const covered = projection.get(message.id) ?? [];
        for (const [index, file] of (message.files ?? []).entries()) {
          const placeholder = covered.find(
            image => image.index === index,
          )?.placeholder;
          if (placeholder !== undefined) {
            text(placeholder);
          } else if (
            file.mime.startsWith("image/")
            || file.mime === "application/pdf"
          ) {
            tokens += NATIVE_BATCH_MEDIA_TOKEN_CHARGE;
            mediaCount += 1;
          } else {
            tokens += file.data.length;
          }
        }
        break;
      }
      case "synthetic":
      case "system":
      case "skill":
        text(message.text);
        break;
      case "shell":
        text(message.command);
        text(message.output?.output);
        break;
      case "compaction":
        if (message.status === "completed") {
          text(message.summary);
          text(message.recent);
        }
        break;
      case "assistant":
        for (const item of message.content) {
          if (item.type !== "tool") {
            text(item.text);
            continue;
          }
          const state = item.state;
          text(
            typeof state.input === "string"
              ? state.input
              : JSON.stringify(state.input),
          );
          if (state.status === "completed" || state.status === "error") {
            for (const content of state.content ?? []) {
              if (content.type === "text") text(content.text);
              else tokens += NATIVE_BATCH_MEDIA_TOKEN_CHARGE;
            }
          }
          if (state.status === "error") text(state.error.message);
        }
        break;
      default:
        break;
    }
  }
  return { tokens, textTokens, mediaCount };
}

// Plans every target before any session is created: budgets, the single
// target per request, the optional preceding turn within the halo limit, and
// the real nonce and prompt, checked against the planning estimate.
function planBatches(
  turns: V2GenerationTurn[],
  model: V2SummarizerModel,
  globalIndexPrompt: string,
  nonceFactory: () => string,
): { budget: NativeBatchBudget; batches: Batch[] } {
  const budget = computeNativeBatchBudget(model.info.limit);
  const candidates: Candidate[] = turns.map((turn, index) => {
    const estimate = estimateV2HistoryTokens(turn.span);
    const mediaManifest = v2MediaManifest(turn.span);
    const ordinal = index + 1;
    return {
      ordinal,
      marker: generationTurnMarker(NATIVE_BATCH_PLANNING_NONCE, ordinal),
      expectedInlineImages: mediaManifest?.images.length ?? 0,
      inputTokens: estimate.tokens,
      outputTokens: estimateTurnOutputTokens(
        estimate.textTokens,
        estimate.mediaCount,
      ),
      turn,
      ...(mediaManifest ? { mediaManifest } : {}),
    };
  });
  const planned = planNativeGenerationBatches(candidates, budget, {
    explicitRequestTokens: targets =>
      estimateGenerationExplicitRequestTokens(targets, globalIndexPrompt),
    responseProtocolTokens: estimateGenerationResponseProtocolTokens,
  });
  const nonces = new Set<string>();
  const batches = planned.map(batch => {
    const candidate = batch.candidates[0]!;
    const previous =
      candidate.turn.previous
      && batch.estimatedInputTokens
        + estimateV2HistoryTokens(candidate.turn.previous).tokens
        <= budget.haloInputLimit
        ? candidate.turn.previous
        : undefined;
    const nonce = selectGenerationNonce(
      [[...(previous ?? []), ...candidate.turn.span]],
      nonceFactory,
      nonces,
    );
    const target: GenerationTarget = {
      ordinal: candidate.ordinal,
      marker: generationTurnMarker(nonce, candidate.ordinal),
      expectedInlineImages: candidate.expectedInlineImages,
    };
    if (
      estimateGenerationExplicitRequestTokens([target], globalIndexPrompt)
        > batch.estimatedExplicitRequestTokens
      || estimateGenerationResponseProtocolTokens([target])
        > estimateGenerationResponseProtocolTokens([candidate])
    ) {
      throw new V2GenerationError(
        "Compaction generation request exceeded its planned size.",
      );
    }
    const firstUser = candidate.turn.span.findIndex(
      message => message.type === "user",
    );
    const user = candidate.turn.span[firstUser]!;
    const marker = {
      id: user.id,
      type: "synthetic",
      text: target.marker,
      time: { created: user.time.created },
    } as V2Message;
    const history = [
      ...(previous ?? []),
      ...candidate.turn.span.slice(0, firstUser),
      marker,
      ...candidate.turn.span.slice(firstUser),
    ];
    return {
      candidate,
      history,
      nonce,
      target,
      prompt: buildGenerationPrompt([target], globalIndexPrompt),
    };
  });
  for (const batch of batches) assertImagesReachModel(batch, model);
  return { budget, batches };
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
  options: { controlTimeoutMs: number; nonceFactory?: () => string },
): Promise<V2GeneratedSummary[]> {
  if (turns.length === 0) return [];
  const { batches } = planBatches(
    turns,
    model,
    globalIndexPrompt,
    options.nonceFactory ?? createGenerationNonce,
  );
  const results: V2GeneratedSummary[] = [];
  const failures: Array<{ index: number; error: V2GenerationError }> = [];
  let nextIndex = 0;
  const worker = async () => {
    while (failures.length === 0 && nextIndex < batches.length) {
      const index = nextIndex++;
      try {
        results[index] = await runBatch(
          host,
          owned,
          source,
          model.ref,
          batches[index]!,
          options.controlTimeoutMs,
        );
      } catch (error) {
        failures.push({
          index,
          error:
            error instanceof V2GenerationError
              ? error
              : new V2GenerationError(
                  "Compaction generation model request failed.",
                ),
        });
      }
    }
  };
  await Promise.all(
    Array.from(
      { length: Math.min(NATIVE_BATCH_MAX_CONCURRENCY, batches.length) },
      worker,
    ),
  );
  if (failures.length > 0) {
    failures.sort((a, b) => a.index - b.index);
    if (failures.length === 1) throw failures[0]!.error;
    const errors = failures.map(failure => failure.error);
    const reasons = [...new Set(errors.map(error => error.reason))].slice(
      0,
      MAX_REASONS,
    );
    const sessions = undeletedSessionIDs(errors);
    throw new V2GenerationError(
      `Compaction generation failed for ${failures.length} turns: ${reasons.join(" ")}${
        sessions.length === 0
          ? ""
          : ` Temporary session${sessions.length === 1 ? "" : "s"} ${sessions.join(", ")} could not be deleted.`
      }`,
      undefined,
      sessions,
    );
  }
  return results;
}

async function runBatch(
  host: V2Host,
  owned: OwnedSessions,
  source: V2Source,
  model: ModelRef,
  batch: Batch,
  controlTimeoutMs: number,
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
    model,
    permissions: GENERATION_PERMISSIONS,
    metadata: {},
    time: { created: now, updated: now },
  } as Record<string, unknown>;
  delete info["parentID"];
  delete info["fork"];
  delete info["revert"];

  owned.register(sessionID, "generation");
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
        json(await readV2History(host, sessionID, read)),
        json(history),
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
  }
  if ((await removeOwnedSession(host, owned, sessionID)) === "unknown") {
    failure = new V2GenerationError(
      failure
        ? `${failure.message} Its temporary session ${sessionID} could not be deleted.`
        : `Compaction generation session deletion failed; temporary session ${sessionID} could not be deleted.`,
      failure?.reason ?? "Compaction generation session deletion failed.",
      [sessionID],
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
  if (!isDeepStrictEqual(json(after.slice(0, history.length)), json(history))) {
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
    const [parsed] = parseGenerationResponse(
      call.state.input,
      [batch.target],
      batch.nonce,
    );
    return {
      text: frameSummary(parsed!),
      ...(batch.candidate.mediaManifest
        ? { mediaManifest: batch.candidate.mediaManifest }
        : {}),
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

function json(value: unknown): unknown {
  return JSON.parse(JSON.stringify(value));
}

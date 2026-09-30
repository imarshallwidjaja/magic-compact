import type { FilePart, Part } from "@opencode-ai/sdk/v2";
import { countGptTokens } from "../stats/tokenize";
import { isRecord } from "../util";
import type { MessageWithParts, Turn } from "./plan";

export const NATIVE_BATCH_MAX_CONCURRENCY = 5;
export const NATIVE_BATCH_NORMAL_INPUT_TARGET = 400_000;
export const NATIVE_BATCH_FIXED_RESERVE_TOKENS = 8_000;
export const NATIVE_BATCH_MEDIA_TOKEN_CHARGE = 8_000;
export const NATIVE_BATCH_MAX_OUTPUT_TARGET = 32_000;
export const NATIVE_BATCH_MAX_HALO_INPUT = 500_000;
// The pinned tokenizer emits one token per byte here, the maximum for a valid 22-byte nonce body.
export const NATIVE_BATCH_PLANNING_NONCE = "uB_4fLx2Xv5Tz7E1M-8vQn";

const NATIVE_BATCH_FALLBACK_OUTPUT_LIMIT = 32_000;
const NATIVE_BATCH_MIN_TURN_OUTPUT = 512;
const NATIVE_BATCH_MAX_TURN_TEXT_OUTPUT = 8_000;
const NATIVE_BATCH_MEDIA_DESCRIPTION_OUTPUT = 512;

export type NativeBatchBudget = {
  outputTarget: number;
  usableInput: number;
  normalInputTarget: number;
  hardSingleTurnInput: number;
  haloInputLimit: number;
};

export type NativeTurnCost = {
  inputTokens: number;
  outputTokens: number;
};

export type NativeBatchProtocol<T extends NativeTurnCost> = {
  explicitRequestTokens: (candidates: T[]) => number;
  responseProtocolTokens: (candidates: T[]) => number;
};

export type NativeGenerationBatch<T extends NativeTurnCost> = {
  candidates: T[];
  estimatedInputTokens: number;
  estimatedExplicitRequestTokens: number;
  estimatedOutputTokens: number;
  inputLimit: number;
};

export function computeNativeBatchBudget(limit: {
  context: number;
  input?: number;
  output?: number;
}): NativeBatchBudget {
  if (!Number.isFinite(limit.context) || limit.context <= 0) {
    throw new Error(
      "Magic Compact requires a positive configured compaction model context limit.",
    );
  }
  const context = Math.floor(limit.context);
  const outputLimit =
    Number.isFinite(limit.output) && (limit.output ?? 0) > 0
      ? Math.floor(limit.output!)
      : NATIVE_BATCH_FALLBACK_OUTPUT_LIMIT;
  const outputTarget = Math.min(
    NATIVE_BATCH_MAX_OUTPUT_TARGET,
    Math.floor(outputLimit * 0.5),
  );
  const configuredInput =
    Number.isFinite(limit.input) && (limit.input ?? 0) > 0
      ? Math.floor(limit.input!)
      : Number.POSITIVE_INFINITY;
  const usableInput = Math.floor(
    Math.min(configuredInput, context - outputTarget),
  );
  if (usableInput <= NATIVE_BATCH_FIXED_RESERVE_TOKENS) {
    throw new Error(
      "Magic Compact configured compaction model has no usable input budget.",
    );
  }
  const hardSingleTurnInput = Math.floor(usableInput * 0.9);
  if (hardSingleTurnInput <= NATIVE_BATCH_FIXED_RESERVE_TOKENS) {
    throw new Error(
      "Magic Compact configured compaction model input budget is too small to fit the fixed host reserve.",
    );
  }
  return {
    outputTarget,
    usableInput,
    normalInputTarget: Math.min(
      NATIVE_BATCH_NORMAL_INPUT_TARGET,
      Math.floor(usableInput * 0.8),
    ),
    hardSingleTurnInput,
    haloInputLimit: Math.min(NATIVE_BATCH_MAX_HALO_INPUT, hardSingleTurnInput),
  };
}

export function estimateNativeTurnCost(turn: Turn): NativeTurnCost {
  const estimate = [...turn.user, ...turn.assistants].reduce(
    (total, message) => {
      for (const part of message.parts) {
        const partEstimate = estimateNativePart(part, message.info.role);
        total.inputTokens += partEstimate.tokens;
        total.textTokens += partEstimate.textTokens;
        total.mediaCount += partEstimate.mediaCount;
      }
      return total;
    },
    { inputTokens: 0, textTokens: 0, mediaCount: 0 },
  );
  return {
    inputTokens: estimate.inputTokens,
    outputTokens:
      Math.max(
        NATIVE_BATCH_MIN_TURN_OUTPUT,
        Math.min(
          NATIVE_BATCH_MAX_TURN_TEXT_OUTPUT,
          Math.ceil(estimate.textTokens * 0.05),
        ),
      )
      + estimate.mediaCount * NATIVE_BATCH_MEDIA_DESCRIPTION_OUTPUT,
  };
}

// Estimates native model input for a whole message history with the same
// per-part rules as turn estimates.
export function estimateNativeHistoryTokens(
  messages: MessageWithParts[],
): number {
  let tokens = 0;
  for (const message of messages) {
    for (const part of message.parts) {
      tokens += estimateNativePart(part, message.info.role).tokens;
    }
  }
  return tokens;
}

// Every generation request owns exactly one target turn. The host does not
// enforce the response schema, and multi-target requests returned the wrong
// number of results live, so the response must never have to map several
// targets back to their ordinals.
export function planNativeGenerationBatches<T extends NativeTurnCost>(
  candidates: T[],
  budget: NativeBatchBudget,
  protocol: NativeBatchProtocol<T>,
): NativeGenerationBatch<T>[] {
  const emptyRequestTokens = Math.ceil(protocol.explicitRequestTokens([]));
  if (
    NATIVE_BATCH_FIXED_RESERVE_TOKENS + emptyRequestTokens
    > budget.usableInput
  ) {
    throw new Error(
      "Magic Compact explicit request and global turn index cannot fit the configured model input budget. Run /magic-trim first or configure a compaction model with more input capacity.",
    );
  }
  return candidates.map(candidate => {
    const estimatedExplicitRequestTokens = Math.ceil(
      protocol.explicitRequestTokens([candidate]),
    );
    const estimatedInputTokens =
      NATIVE_BATCH_FIXED_RESERVE_TOKENS
      + estimatedExplicitRequestTokens
      + Math.ceil(candidate.inputTokens);
    const estimatedOutputTokens =
      Math.ceil(protocol.responseProtocolTokens([candidate]))
      + Math.ceil(candidate.outputTokens);
    if (estimatedInputTokens > budget.hardSingleTurnInput) {
      throw new Error(
        "A single compaction turn cannot fit the configured model input. Run /magic-trim first and retry.",
      );
    }
    if (estimatedOutputTokens > budget.outputTarget) {
      throw new Error(
        `A single compaction turn cannot fit the configured ${budget.outputTarget}-token output target. Configure a compaction model with more output capacity.`,
      );
    }
    return {
      candidates: [candidate],
      estimatedInputTokens,
      estimatedExplicitRequestTokens,
      estimatedOutputTokens,
      inputLimit:
        estimatedInputTokens > budget.normalInputTarget
          ? budget.hardSingleTurnInput
          : budget.normalInputTarget,
    };
  });
}

type ContentEstimate = {
  tokens: number;
  textTokens: number;
  mediaCount: number;
};

export function toolAttachmentsModelVisible(part: unknown): boolean {
  if (!isRecord(part) || part["type"] !== "tool") return false;
  const state = part["state"];
  if (!isRecord(state) || state["status"] !== "completed") return false;
  const time = state["time"];
  return !(isRecord(time) && Boolean(time["compacted"]));
}

function isMediaMime(mime: string): boolean {
  return mime.startsWith("image/") || mime === "application/pdf";
}

function filePartEstimate(part: FilePart): ContentEstimate {
  if (part.mime === "text/plain" || part.mime === "application/x-directory") {
    return emptyEstimate();
  }
  if (isMediaMime(part.mime)) return mediaEstimate();
  return part.url.startsWith("data:")
    ? encodedTextEstimate(part.url)
    : textEstimate(part.url);
}

function attachmentEstimate(attachment: FilePart): ContentEstimate {
  if (isMediaMime(attachment.mime)) return mediaEstimate();
  return attachment.url.startsWith("data:")
    ? encodedTextEstimate(attachment.url)
    : textEstimate(attachment.url);
}

function estimateNativePart(
  part: Part,
  role: "user" | "assistant",
): ContentEstimate {
  if (part.type === "file") return filePartEstimate(part);
  if (part.type === "text") {
    return part.ignored === true && role === "user"
      ? emptyEstimate()
      : textEstimate(part.text);
  }
  if (part.type === "reasoning") return textEstimate(part.text);
  if (part.type === "subtask") {
    return combineEstimates([
      textEstimate(part.prompt),
      textEstimate(part.description),
    ]);
  }
  if (part.type === "snapshot") return textEstimate(part.snapshot);
  if (part.type === "patch") {
    return combineEstimates(part.files.map(textEstimate));
  }
  if (part.type !== "tool") return emptyEstimate();

  const values: unknown[] = [part.state.input];
  let attachments: FilePart[] = [];
  if (part.state.status === "completed") {
    if (toolAttachmentsModelVisible(part)) {
      values.push(part.state.output);
      attachments = part.state.attachments ?? [];
    } else {
      values.push("[Old tool result content cleared]");
    }
  }
  if (part.state.status === "error") {
    const interruptedOutput =
      part.state.metadata?.interrupted === true
      && typeof part.state.metadata.output === "string"
        ? part.state.metadata.output
        : undefined;
    values.push(interruptedOutput ?? part.state.error);
  }
  if (part.state.status === "pending" || part.state.status === "running") {
    values.push("[Tool execution was interrupted]");
  }
  return combineEstimates([
    ...values.map(value => estimateModelValue(value, new Set())),
    ...attachments.map(attachmentEstimate),
  ]);
}

function estimateModelValue(
  value: unknown,
  ancestry: Set<object>,
): ContentEstimate {
  if (typeof value === "string") {
    return value.startsWith("data:")
      ? encodedTextEstimate(value)
      : textEstimate(value);
  }
  if (
    value === null
    || typeof value === "number"
    || typeof value === "boolean"
    || typeof value === "bigint"
  ) {
    return textEstimate(String(value));
  }
  if (value === undefined) return emptyEstimate();
  if (typeof value !== "object") return textEstimate(String(value));
  if (ancestry.has(value)) {
    throw new Error(
      "Compaction generation token estimate encountered circular tool content.",
    );
  }

  ancestry.add(value);
  const estimate = Array.isArray(value)
    ? combineEstimates(
        value.map(item => estimateModelValue(item, ancestry)),
        value.length + 2,
      )
    : combineEstimates(
        Object.entries(value).flatMap(([key, item]) => [
          key.startsWith("data:")
            ? encodedTextEstimate(key)
            : textEstimate(key),
          estimateModelValue(item, ancestry),
        ]),
        Object.keys(value).length + 2,
      );
  ancestry.delete(value);
  return estimate;
}

function textEstimate(text: string): ContentEstimate {
  const tokens = countGptTokens(text);
  return { tokens, textTokens: tokens, mediaCount: 0 };
}

function encodedTextEstimate(text: string): ContentEstimate {
  const tokens = Buffer.byteLength(JSON.stringify(text));
  return { tokens, textTokens: tokens, mediaCount: 0 };
}

function mediaEstimate(): ContentEstimate {
  return {
    tokens: NATIVE_BATCH_MEDIA_TOKEN_CHARGE,
    textTokens: 0,
    mediaCount: 1,
  };
}

function emptyEstimate(): ContentEstimate {
  return { tokens: 0, textTokens: 0, mediaCount: 0 };
}

function combineEstimates(
  estimates: ContentEstimate[],
  structuralTokens = 0,
): ContentEstimate {
  return estimates.reduce(
    (total, estimate) => ({
      tokens: total.tokens + estimate.tokens,
      textTokens: total.textTokens + estimate.textTokens,
      mediaCount: total.mediaCount + estimate.mediaCount,
    }),
    {
      tokens: structuralTokens,
      textTokens: structuralTokens,
      mediaCount: 0,
    },
  );
}

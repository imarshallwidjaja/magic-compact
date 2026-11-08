import type { FilePart, Part } from "@opencode-ai/sdk/v2";
import {
  estimateTurnOutputTokens,
  NATIVE_BATCH_MEDIA_TOKEN_CHARGE,
  type NativeTurnCost,
} from "../policy/budget";
import { countGptTokens } from "../stats/tokenize";
import { isRecord } from "../util";
import type { MessageWithParts, Turn } from "./plan";

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
    outputTokens: estimateTurnOutputTokens(
      estimate.textTokens,
      estimate.mediaCount,
    ),
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

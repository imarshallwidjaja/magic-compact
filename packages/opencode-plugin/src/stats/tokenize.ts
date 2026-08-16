import { createRequire } from "node:module";
import type { Part } from "@opencode-ai/sdk/v2";
import { countTokens } from "gpt-tokenizer";
import { unwrap, type V2Client } from "../api";
import { projectManifestCoveredMedia } from "../compact/media";
import type { MessageWithParts } from "../compact/plan";

const PINNED_GPT_TOKENIZER_VERSION = "3.4.0";
const PINNED_O200K_MAX_TOKEN_BYTES = 128;
const require = createRequire(import.meta.url);
const installedTokenizerVersion = (
  require("gpt-tokenizer/package.json") as { version: string }
).version;
const MAX_EXACT_TOKEN_COUNT_ENTRIES = 1;
const MAX_EXACT_TOKEN_COUNT_KEY_BYTES = 4_096;
const exactTokenCounts = new Map<string, number>();

function exactTokenCount(text: string): number {
  if (Buffer.byteLength(text) > MAX_EXACT_TOKEN_COUNT_KEY_BYTES) {
    return countTokens(text);
  }
  const cached = exactTokenCounts.get(text);
  if (cached !== undefined) return cached;
  const count = countTokens(text);
  if (exactTokenCounts.size >= MAX_EXACT_TOKEN_COUNT_ENTRIES) {
    const oldest = exactTokenCounts.keys().next().value;
    if (oldest !== undefined) exactTokenCounts.delete(oldest);
  }
  exactTokenCounts.set(text, count);
  return count;
}

function isClearlyOverTokenLimit(text: string, limit: number): boolean {
  return (
    installedTokenizerVersion === PINNED_GPT_TOKENIZER_VERSION
    && Buffer.byteLength(text) > limit * PINNED_O200K_MAX_TOKEN_BYTES
  );
}

export function countGptTokens(text: string): number {
  return exactTokenCount(text);
}

export function countGptTokensWithinLimit(
  text: string,
  limit: number,
): number | null {
  if (limit < 0 || isClearlyOverTokenLimit(text, limit)) return null;
  const count = exactTokenCount(text);
  return count <= limit ? count : null;
}

function stringifyToolContent(value: unknown): string {
  return typeof value === "string" ? value : JSON.stringify(value);
}

function extractToolContent(part: Part): string[] {
  if (part.type !== "tool") {
    return [];
  }

  const contents: string[] = [];

  if (part.state.input !== undefined) {
    contents.push(stringifyToolContent(part.state.input));
  }

  if (part.state.status === "completed" && part.state.output !== undefined) {
    contents.push(stringifyToolContent(part.state.output));
  } else if (part.state.status === "error" && part.state.error) {
    contents.push(stringifyToolContent(part.state.error));
  }

  return contents;
}

function extractPartTexts(
  part: Part,
  role: MessageWithParts["info"]["role"],
): string[] {
  switch (part.type) {
    case "text":
      return part.ignored === true && role === "user" ? [] : [part.text];
    case "reasoning":
      return [part.text];
    case "tool":
      return extractToolContent(part);
    case "subtask":
      return [part.prompt, part.description];
    case "file":
      return [part.url];
    case "snapshot":
      return [part.snapshot];
    case "patch":
      return part.files;
    default:
      return [];
  }
}

export async function countSessionTokens(
  v2: V2Client,
  sessionID: string,
): Promise<number> {
  return (await countSessionTokenProjection(v2, sessionID)).tokens;
}

export async function countSessionTokenProjection(
  v2: V2Client,
  sessionID: string,
): Promise<ModelFacingTokenCount> {
  const messages = unwrap(
    await v2.session.messages({ sessionID }),
  ) as MessageWithParts[];
  return countModelFacingContentTokens(messages);
}

export type ModelFacingTokenCount = {
  tokens: number;
  projectedMediaTokens: number;
};

export function countModelFacingContentTokens(
  messages: MessageWithParts[],
): ModelFacingTokenCount {
  const projectedMessages = projectManifestCoveredMedia(messages);
  const persistedTokens = countCanonicalContentTokens(messages);
  const tokens = countCanonicalContentTokens(projectedMessages);
  return {
    tokens,
    projectedMediaTokens: persistedTokens - tokens,
  };
}

export function countCanonicalContentTokens(
  messages: MessageWithParts[],
): number {
  return messages.reduce(
    (sum, message) => sum + countMessageGptTokens(message),
    0,
  );
}

function countMessageGptTokens(message: MessageWithParts): number {
  const count = message.parts.map(part =>
    countGptTokens(extractPartTexts(part, message.info.role).join("\n\n")),
  );
  return count.reduce((sum, count) => sum + count, 0);
}

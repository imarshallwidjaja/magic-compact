import { createHash } from "node:crypto";
import { createRequire } from "node:module";
import { performance } from "node:perf_hooks";
import type { Part } from "@opencode-ai/sdk/v2";
import { countTokens, encodeGenerator, setMergeCacheSize } from "gpt-tokenizer";
import { unwrap, type V2Client } from "../api";
import { projectManifestCoveredMedia } from "../compact/media";
import type { MessageWithParts } from "../compact/plan";

const PINNED_GPT_TOKENIZER_VERSION = "3.4.0";
const PINNED_O200K_MAX_TOKEN_BYTES = 128;
const require = createRequire(import.meta.url);
const installedTokenizerVersion = (
  require("gpt-tokenizer/package.json") as { version: string }
).version;
// The tokenizer's merge cache is a 100,000-entry LRU Map. Under Bun, once
// one input brings more distinct pieces than that (base64 file data does),
// its evictions grow superlinearly: one count of 4 MiB of random base64 took
// 31 s with the cache and 1.7 s without, and three counts of the same 1 MiB
// took 0.9, 3.6 and 8.3 s. Counts are the same either way.
setMergeCacheSize(0);
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

// A part text this long (a file URL with inline data, or a large tool value)
// is counted exactly, but in slices that give the event loop back, and is
// memoized by its SHA-256, because accounting counts the same unchanged
// text several times per command and the host serves every client from this
// event loop. One tokenizer piece is never split, so a single long uniform run
// still blocks while it is counted.
const LARGE_PART_TEXT_CHARS = 65_536;
const LARGE_PART_TEXT_COUNT_ENTRIES = 1_024;
const COUNT_SLICE_MS = 10;
const largePartTextCounts = new Map<string, number>();

async function countLargeTextTokens(text: string): Promise<number> {
  const key = createHash("sha256").update(text).digest("hex");
  const cached = largePartTextCounts.get(key);
  if (cached !== undefined) return cached;
  let count = 0;
  let pieces = 0;
  let sliceStart = performance.now();
  // The same pieces, special-token check and count as `countTokens`.
  for (const tokens of encodeGenerator(text)) {
    count += tokens.length;
    if (
      ++pieces % 256 === 0
      && performance.now() - sliceStart >= COUNT_SLICE_MS
    ) {
      await new Promise(resolve => setImmediate(resolve));
      sliceStart = performance.now();
    }
  }
  if (largePartTextCounts.size >= LARGE_PART_TEXT_COUNT_ENTRIES) {
    largePartTextCounts.delete(largePartTextCounts.keys().next().value!);
  }
  largePartTextCounts.set(key, count);
  return count;
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

export async function countModelFacingContentTokens(
  messages: MessageWithParts[],
): Promise<ModelFacingTokenCount> {
  const projectedMessages = projectManifestCoveredMedia(messages);
  const persistedTokens = await countCanonicalContentTokens(messages);
  const tokens = await countCanonicalContentTokens(projectedMessages);
  return {
    tokens,
    projectedMediaTokens: persistedTokens - tokens,
  };
}

export async function countCanonicalContentTokens(
  messages: MessageWithParts[],
): Promise<number> {
  let total = 0;
  for (const message of messages) {
    for (const part of message.parts) {
      const text = extractPartTexts(part, message.info.role).join("\n\n");
      total +=
        text.length > LARGE_PART_TEXT_CHARS
          ? await countLargeTextTokens(text)
          : countGptTokens(text);
    }
  }
  return total;
}

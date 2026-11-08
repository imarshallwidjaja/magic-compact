// The one tokenizer every estimate on both hosts uses. Every count goes
// through this module, which pins the tokenizer's behavior and settings.
import { createHash } from "node:crypto";
import { createRequire } from "node:module";
import { performance } from "node:perf_hooks";
import { countTokens, encodeGenerator, setMergeCacheSize } from "gpt-tokenizer";

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
// Session text is counted as the text it is. Without this option the
// tokenizer throws on a special-token spelling such as `<|endoftext|>`,
// which any tool output or user message can contain.
const LITERAL_TEXT = { disallowedSpecial: new Set<string>() };
const MAX_EXACT_TOKEN_COUNT_ENTRIES = 1;
const MAX_EXACT_TOKEN_COUNT_KEY_BYTES = 4_096;
const exactTokenCounts = new Map<string, number>();

function exactTokenCount(text: string): number {
  if (Buffer.byteLength(text) > MAX_EXACT_TOKEN_COUNT_KEY_BYTES) {
    return countTokens(text, LITERAL_TEXT);
  }
  const cached = exactTokenCounts.get(text);
  if (cached !== undefined) return cached;
  const count = countTokens(text, LITERAL_TEXT);
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

// Counts a long text exactly, but in slices that give the event loop back,
// and memoizes it by its SHA-256, because accounting counts the same
// unchanged text several times per command and the host serves every client
// from this event loop. One tokenizer piece is never split, so a single long
// uniform run still blocks while it is counted.
const YIELDING_COUNT_ENTRIES = 1_024;
const COUNT_SLICE_MS = 10;
const yieldingCounts = new Map<string, number>();

export async function countGptTokensYielding(text: string): Promise<number> {
  const key = createHash("sha256").update(text).digest("hex");
  const cached = yieldingCounts.get(key);
  if (cached !== undefined) return cached;
  let count = 0;
  let pieces = 0;
  let sliceStart = performance.now();
  // The same pieces and count as `countTokens`.
  for (const tokens of encodeGenerator(text, LITERAL_TEXT)) {
    count += tokens.length;
    if (
      ++pieces % 256 === 0
      && performance.now() - sliceStart >= COUNT_SLICE_MS
    ) {
      await new Promise(resolve => setImmediate(resolve));
      sliceStart = performance.now();
    }
  }
  if (yieldingCounts.size >= YIELDING_COUNT_ENTRIES) {
    yieldingCounts.delete(yieldingCounts.keys().next().value!);
  }
  yieldingCounts.set(key, count);
  return count;
}

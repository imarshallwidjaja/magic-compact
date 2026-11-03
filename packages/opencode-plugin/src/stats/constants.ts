import { getCachedReadPrice } from "./pricing";
import type { ConversationStats } from "../storage/stats";

export const STATS_METADATA = {
  magicCompact: {
    stats: true,
  },
};

function formatTokenCount(tokens: number): string {
  if (tokens >= 1000) {
    return `${(tokens / 1000).toFixed(1)}K`.replace(".0K", "K");
  }
  return tokens.toString();
}

const PROVIDER_METER_NOTE =
  "Counts are canonical active-content estimates of stored history; this ignored reporting notice is excluded. They can include inline file data counted as text, and they are not provider usage or media billing. OpenCode's context meter is a provider usage snapshot; in a successor session it can show the copied last response's usage until a later assistant response refreshes it.";

export function statsSummaryMessage(
  compactionCount: number,
  stats: ConversationStats,
  modelID: string | null,
): string {
  const moneySaved = moneySavedMessage(stats, modelID);

  return `
Magic Compact      | ${compactionCount} compactions performed
Conversation Stats | Total tokens pruned: ~${formatTokenCount(stats.totalTokensPruned)} (Tokens physically pruned, net canonical active content) | Inline image tokens projected: ~${formatTokenCount(stats.mediaTokensProjected ?? 0)} per model request
Cache Savings      | Physical cache reads saved: ~${formatTokenCount(stats.cachedTokensSaved)} tokens | Projected-media cache reads saved: ~${formatTokenCount(stats.cachedMediaTokensSaved ?? 0)} tokens
Cost Savings       | ${moneySaved}
Note               | ${PROVIDER_METER_NOTE}
`.trim();
}

// A compaction that removes estimated reasoning replay cost can still grow the
// stored history, which is then reported as growth rather than as a removal.
function contextReductionLine(
  beforeTokens: number,
  afterTokens: number,
  unit: string,
): string {
  const savedTokens = beforeTokens - afterTokens;
  const percent = (tokens: number) =>
    beforeTokens > 0 ? Math.round((tokens / beforeTokens) * 100) : 0;
  return `${formatTokenCount(beforeTokens)} → ${formatTokenCount(afterTokens)} ${unit} | ${
    savedTokens < 0
      ? `${percent(-savedTokens)}% grown (~${formatTokenCount(-savedTokens)} added)`
      : `${percent(savedTokens)}% trimmed (~${formatTokenCount(savedTokens)} removed)`
  }`;
}

export type PreservedTurns = { requested: number; preserved: number };

export function preservedTurnsFeedback({
  requested,
  preserved,
}: PreservedTurns): string {
  if (preserved === 0) return `N=${requested} preserved no request turns.`;
  return `N=${requested} preserved ${preserved} recent request turn(s) whole. A request turn is one user request plus every assistant and tool step until the next user request; /magic-compact 0 also summarizes the most recent turn.`;
}

// Counts cover the whole stored history, including any prefix a native
// checkpoint hides from the model, so they are not the model's request size.
const CONTEXT_LABEL = "Stored history (estimate)";

// Inline files a stored history holds, whether or not a request projects them.
export type InlineFileInventory = { count: number; encodedBytes: number };

// One side of a compact or trim notice. `tokens` is the canonical
// stored-history count that gates, profitability, stats and the successor
// check use; `inlineFileTokens` is the part of it that inline file data still
// counted as text contributes.
export type StoredHistoryCount = {
  tokens: number;
  inlineFileTokens: number;
  inlineFiles: InlineFileInventory;
};

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
}

function changeOf(before: string, after: string): string {
  return before === after ? before : `${before} → ${after}`;
}

// Base64 file data tokenized as text can dwarf the conversation and says
// nothing about what a provider charges for the file, so once either side
// holds an inline file the headline counts stored text only and the files are
// reported by size. Without inline files the lines are unchanged.
function storedHistoryLines(
  before: StoredHistoryCount,
  after: StoredHistoryCount,
): string[] {
  if (before.inlineFiles.count === 0 && after.inlineFiles.count === 0) {
    return [
      `${CONTEXT_LABEL} | ${contextReductionLine(before.tokens, after.tokens, "tokens")}`,
    ];
  }
  const same =
    before.inlineFiles.count === after.inlineFiles.count
    && before.inlineFiles.encodedBytes === after.inlineFiles.encodedBytes;
  const beforeText = before.tokens - before.inlineFileTokens;
  const afterText = after.tokens - after.inlineFileTokens;
  const unit = "text tokens, excluding inline file data";
  const textChange =
    beforeText === 0 && afterText > 0
      ? `0 → ${formatTokenCount(afterText)} ${unit} | ~${formatTokenCount(afterText)} added`
      : contextReductionLine(beforeText, afterText, unit);
  return [
    `${CONTEXT_LABEL} | ${textChange}`,
    `Inline files | ${changeOf(
      `${before.inlineFiles.count}`,
      `${after.inlineFiles.count}`,
    )} file(s), ${changeOf(
      formatBytes(before.inlineFiles.encodedBytes),
      formatBytes(after.inlineFiles.encodedBytes),
    )} encoded data in stored history${same ? " before and after" : ""}. Provider token cost for these files is not estimated; the text tokens above count stored text, not the model request.`,
  ];
}

export function statsMessage(
  compactionCount: number,
  before: StoredHistoryCount,
  after: StoredHistoryCount,
  summarizedTurnCount: number,
  preserved: PreservedTurns,
  retentionFeedback: string,
  // Further `Label | text` lines, such as the reasoning lines.
  extraLines: readonly string[] = [],
): string {
  return [
    `Magic Compaction #${compactionCount}`,
    ...storedHistoryLines(before, after),
    `Changes | Request turns summarized: ${summarizedTurnCount}`,
    `Preserved | ${preservedTurnsFeedback(preserved)}`,
    `Retention | ${retentionFeedback}`,
    ...extraLines,
  ].join("\n");
}

export function trimStatsMessage(
  before: StoredHistoryCount,
  after: StoredHistoryCount,
  trimmedToolCallCount: number,
  retentionFeedback: string,
  extraLines: readonly string[] = [],
): string {
  return [
    "Magic Trim",
    ...storedHistoryLines(before, after),
    `Changes | Tool calls trimmed: ${trimmedToolCallCount}`,
    `Retention | ${retentionFeedback}`,
    ...extraLines,
  ].join("\n");
}

function moneySavedMessage(
  stats: ConversationStats,
  modelID: string | null,
): string {
  const dollarsPerMillion =
    modelID !== null ? getCachedReadPrice(modelID) : null;
  if (dollarsPerMillion === null) {
    return `Cost savings unavailable: Model not supported: ${modelID}`;
  }

  const cachedTokens =
    stats.cachedTokensSaved + (stats.cachedMediaTokensSaved ?? 0);
  return `Est. cost saved (${modelID}): $${((cachedTokens / 1_000_000) * dollarsPerMillion).toFixed(4)}`;
}

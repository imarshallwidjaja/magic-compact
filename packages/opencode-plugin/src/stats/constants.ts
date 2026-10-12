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
  "Counts are canonical active-content estimates; this ignored reporting notice is excluded. They are not provider usage. OpenCode's context meter is a provider usage snapshot and can stay stale after in-place compaction until a later assistant response refreshes it.";

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

function contextReductionLine(
  beforeTokens: number,
  afterTokens: number,
): string {
  const savedTokens = beforeTokens - afterTokens;
  const percent =
    beforeTokens > 0 ? Math.round((savedTokens / beforeTokens) * 100) : 0;
  return `${formatTokenCount(beforeTokens)} → ${formatTokenCount(afterTokens)} tokens | ${percent}% trimmed (~${formatTokenCount(savedTokens)} removed)`;
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

export function statsMessage(
  compactionCount: number,
  beforeTokens: number,
  afterTokens: number,
  summarizedTurnCount: number,
  preserved: PreservedTurns,
): string {
  return [
    `Magic Compaction #${compactionCount}`,
    `${CONTEXT_LABEL} | ${contextReductionLine(beforeTokens, afterTokens)}`,
    `Changes | Request turns summarized: ${summarizedTurnCount}`,
    `Preserved | ${preservedTurnsFeedback(preserved)}`,
  ].join("\n");
}

export function trimStatsMessage(
  beforeTokens: number,
  afterTokens: number,
  trimmedToolCallCount: number,
  retentionFeedback: string,
): string {
  return [
    "Magic Trim",
    `${CONTEXT_LABEL} | ${contextReductionLine(beforeTokens, afterTokens)}`,
    `Changes | Tool calls trimmed: ${trimmedToolCallCount}`,
    `Retention | ${retentionFeedback}`,
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

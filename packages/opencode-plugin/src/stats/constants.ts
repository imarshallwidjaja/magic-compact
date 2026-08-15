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
Conversation Stats | Tokens pruned (canonical active content): ~${formatTokenCount(stats.totalTokensPruned)} | Total cache reads saved: ~${formatTokenCount(stats.cachedTokensSaved)} tokens
Cost Savings       | ${moneySaved}
Note               | ${PROVIDER_METER_NOTE}
`.trim();
}

export function statsMessage(
  compactionCount: number,
  beforeTokens: number,
  afterTokens: number,
  stats: ConversationStats,
  modelID: string | null,
): string {
  const savedTokens = beforeTokens - afterTokens;
  const percent =
    beforeTokens > 0 ? Math.round((savedTokens / beforeTokens) * 100) : 0;
  const moneySaved = moneySavedMessage(stats, modelID);

  return `
Magic Compaction #${compactionCount}
Active Content (canonical estimate; ignored notice excluded) | ${formatTokenCount(beforeTokens)} → ${formatTokenCount(afterTokens)} tokens (${percent}% reduced) | ~${formatTokenCount(savedTokens)} tokens pruned
Conversation Stats | Total tokens pruned (canonical active content): ~${formatTokenCount(stats.totalTokensPruned)} | Total cache reads saved: ~${formatTokenCount(stats.cachedTokensSaved)} tokens
Cost Savings       | ${moneySaved}
Note               | ${PROVIDER_METER_NOTE}
`.trim();
}

export function trimStatsMessage(
  beforeTokens: number,
  afterTokens: number,
  stats: ConversationStats,
  modelID: string | null,
): string {
  const savedTokens = beforeTokens - afterTokens;
  const percent =
    beforeTokens > 0 ? Math.round((savedTokens / beforeTokens) * 100) : 0;
  const moneySaved = moneySavedMessage(stats, modelID);

  return `
Magic Trim
Active Content (canonical estimate; ignored notice excluded) | ${formatTokenCount(beforeTokens)} → ${formatTokenCount(afterTokens)} tokens (${percent}% reduced) | ~${formatTokenCount(savedTokens)} tokens pruned
Conversation Stats | Total tokens pruned (canonical active content): ~${formatTokenCount(stats.totalTokensPruned)} | Total cache reads saved: ~${formatTokenCount(stats.cachedTokensSaved)} tokens
Cost Savings       | ${moneySaved}
Note               | ${PROVIDER_METER_NOTE}
`.trim();
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

  return `Est. cost saved (${modelID}): $${((stats.cachedTokensSaved / 1_000_000) * dollarsPerMillion).toFixed(4)}`;
}

import type { AssistantMessage, Event } from "@opencode-ai/sdk";
import { readStats, writeStats } from "../storage/stats";
import type { ConversationStats } from "../storage/stats";

export async function handleStatsEvent(event: Event): Promise<void> {
  if (event.type !== "message.updated") {
    return;
  }

  const info = event.properties.info;
  if (info.role !== "assistant") {
    return;
  }

  const stats = await readStats(info.sessionID);
  if (!stats) {
    return;
  }

  const nextStats = applyAssistantMessageStats(stats, info);
  if (!nextStats) {
    return;
  }

  await writeStats(info.sessionID, nextStats);
}

function applyAssistantMessageStats(
  stats: ConversationStats,
  info: AssistantMessage,
): ConversationStats | null {
  if (!info.time.completed) {
    return null;
  }

  if (stats.processedAssistantMessageIds.includes(info.id)) {
    return null;
  }

  return {
    ...stats,
    cachedTokensSaved: stats.cachedTokensSaved + stats.totalTokensPruned,
    cachedMediaTokensSaved:
      (stats.cachedMediaTokensSaved ?? 0) + (stats.mediaTokensProjected ?? 0),
    processedAssistantMessageIds: [
      ...stats.processedAssistantMessageIds,
      info.id,
    ],
  };
}

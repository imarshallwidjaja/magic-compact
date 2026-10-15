import type { Event } from "@opencode-ai/sdk";
import { readStats, writeStats } from "../storage/stats";

export async function handleStatsEvent(event: Event): Promise<void> {
  if (event.type !== "message.updated") {
    return;
  }

  const info = event.properties.info;
  if (info.role !== "assistant" || !info.time.completed) {
    return;
  }

  await recordAssistantCompletion(info.sessionID, info.id);
}

// Adds the conversation's current savings once per completed assistant
// message. `statsKey` names the session's stats file.
export async function recordAssistantCompletion(
  statsKey: string,
  messageID: string,
): Promise<void> {
  const stats = await readStats(statsKey);
  if (!stats || stats.processedAssistantMessageIds.includes(messageID)) {
    return;
  }

  await writeStats(statsKey, {
    ...stats,
    cachedTokensSaved: stats.cachedTokensSaved + stats.totalTokensPruned,
    cachedMediaTokensSaved:
      (stats.cachedMediaTokensSaved ?? 0) + (stats.mediaTokensProjected ?? 0),
    processedAssistantMessageIds: [
      ...stats.processedAssistantMessageIds,
      messageID,
    ],
  });
}

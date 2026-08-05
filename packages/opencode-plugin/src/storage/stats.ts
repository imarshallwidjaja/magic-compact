import { rm } from "node:fs/promises";
import { z } from "zod";
import { pluginStorageDirectory, readJSONFile, writeJSONFile } from "./store";

const ConversationStatsSchema = z.object({
  version: z.literal(1),
  sourceSessionId: z.string().nullable(),
  rootSessionId: z.string(),
  totalTokensPruned: z.number(),
  cachedTokensSaved: z.number(),
  processedAssistantMessageIds: z.array(z.string()),
});

export type ConversationStats = z.infer<typeof ConversationStatsSchema>;

const STATS_VERSION = 1;

function createEmptyStats(input: {
  sourceSessionID: string | null;
  rootSessionID: string;
}): ConversationStats {
  return {
    version: STATS_VERSION,
    sourceSessionId: input.sourceSessionID,
    rootSessionId: input.rootSessionID,
    totalTokensPruned: 0,
    cachedTokensSaved: 0,
    processedAssistantMessageIds: [],
  };
}

export async function readStats(
  sessionID: string,
): Promise<ConversationStats | null> {
  return readJSONFile(statsPath(sessionID), ConversationStatsSchema);
}

export async function writeStats(
  sessionID: string,
  stats: ConversationStats,
): Promise<void> {
  await writeJSONFile(statsPath(sessionID), stats);
}

export async function copyStats(
  sourceSessionID: string,
  targetSessionID: string,
): Promise<ConversationStats> {
  const source = await readStats(sourceSessionID);
  const stats: ConversationStats = source
    ? {
        ...source,
        sourceSessionId: sourceSessionID,
      }
    : createEmptyStats({
        sourceSessionID: sourceSessionID,
        rootSessionID: sourceSessionID,
      });

  await writeStats(targetSessionID, stats);
  return stats;
}

export async function deleteStats(sessionID: string): Promise<void> {
  await rm(statsPath(sessionID), { force: true });
}

function statsPath(sessionID: string): string {
  return `${pluginStorageDirectory()}/stats/${sessionID}.json`;
}

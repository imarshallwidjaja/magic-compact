import { z } from "zod";
import { pluginStorageDirectory, readJSONFile, writeJSONFile } from "./store";

export type OmissionTarget = "input" | "output";

const OmissionEntrySchema = z.object({
  content: z.string(),
});

export type OmissionEntry = z.infer<typeof OmissionEntrySchema>;

const OmissionCacheSchema = z.object({
  version: z.literal(1),
  nextId: z.number(),
  entries: z.record(z.string(), OmissionEntrySchema),
});

export type OmissionCache = z.infer<typeof OmissionCacheSchema>;

const CACHE_VERSION = 1;
const temporaryOmissionSources = new Map<string, string>();

export async function allocateOmission(
  sessionID: string,
  entry: OmissionEntry,
): Promise<string> {
  const cache = (await readCache(sessionID)) ?? createEmptyCache();
  const contentID = formatContentID(cache.nextId);
  const nextCache: OmissionCache = {
    ...cache,
    nextId: cache.nextId + 1,
    entries: {
      ...cache.entries,
      [contentID]: entry,
    },
  };

  await writeCache(sessionID, nextCache);
  return contentID;
}

export async function copyCache(
  sourceSessionID: string,
  targetSessionID: string,
): Promise<void> {
  const cache = await readCache(sourceSessionID);
  if (!cache) {
    return;
  }

  await writeCache(targetSessionID, cache);
}

export async function readOmittedContent(
  sessionID: string,
  contentID: string,
): Promise<string | null> {
  const sourceSessionID = temporaryOmissionSources.get(sessionID) ?? sessionID;
  const cache = await readCache(sourceSessionID);
  if (!cache) {
    return null;
  }

  return cache.entries[contentID]?.content ?? null;
}

export function installTemporaryOmissionSource(
  temporarySessionID: string,
  sourceSessionID: string,
): () => void {
  temporaryOmissionSources.set(temporarySessionID, sourceSessionID);

  return () => {
    temporaryOmissionSources.delete(temporarySessionID);
  };
}

export async function readCache(
  sessionID: string,
): Promise<OmissionCache | null> {
  return readJSONFile(cachePath(sessionID), OmissionCacheSchema);
}

export async function writeCache(
  sessionID: string,
  cache: OmissionCache,
): Promise<void> {
  await writeJSONFile(cachePath(sessionID), cache);
}

export function cachePath(sessionID: string): string {
  return `${pluginStorageDirectory()}/${sessionID}.json`;
}

function createEmptyCache(): OmissionCache {
  return {
    version: CACHE_VERSION,
    nextId: 1,
    entries: {},
  };
}

function formatContentID(id: number): string {
  return `omitted-${id.toString().padStart(3, "0")}`;
}

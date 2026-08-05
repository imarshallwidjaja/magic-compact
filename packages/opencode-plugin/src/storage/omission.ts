import { createHash, randomBytes, randomUUID } from "node:crypto";
import { mkdir, rename, rm, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { z } from "zod";
import { pluginStorageDirectory } from "./store";

const LegacyOmissionEntrySchema = z.object({ content: z.string() }).strict();
const OmissionEntrySchema = z
  .object({
    content: z.string(),
    sha256: z.string().regex(/^[a-f0-9]{64}$/),
  })
  .strict();
const QualifiedContentIDSchema = z
  .string()
  .regex(/^[A-Za-z0-9_-]{1,12}:omitted-[A-Za-z0-9_-]{22}$/);
const LegacyContentIDSchema = z.string().regex(/^omitted-\d+$/);

const LegacyOmissionCacheSchema = z
  .object({
    version: z.literal(1),
    nextId: z.number().int().positive(),
    entries: z.record(LegacyContentIDSchema, LegacyOmissionEntrySchema),
  })
  .strict();

const OmissionCacheSchema = z
  .object({
    version: z.literal(2),
    entries: z.record(QualifiedContentIDSchema, OmissionEntrySchema),
    legacy: z
      .object({
        entries: z.record(LegacyContentIDSchema, LegacyOmissionEntrySchema),
      })
      .strict(),
  })
  .strict();

export type OmissionEntry = { content: string };
export type OmissionCache = z.infer<typeof OmissionCacheSchema>;

export class OmissionIntegrityError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "OmissionIntegrityError";
  }
}

export function omissionReadDiagnostic(
  contentID: string,
  error: unknown,
): string {
  if (error instanceof OmissionIntegrityError) {
    return `Omitted content unavailable for Content ID ${contentID}: integrity verification failed. ${error.message}`;
  }
  return `Omitted content unavailable for Content ID ${contentID}: the omission cache could not be read safely.`;
}

const temporaryOmissionSources = new Map<string, string>();
const mutationQueues = new Map<string, Promise<void>>();

export async function allocateOmission(
  sessionID: string,
  entry: OmissionEntry,
  tokenFactory: () => string = createToken,
): Promise<string> {
  return withSessionMutation(sessionID, async () => {
    const cache = (await readCacheFile(sessionID)) ?? createEmptyCache();
    for (let attempt = 0; attempt < 128; attempt++) {
      const contentID = formatContentID(sessionID, tokenFactory());
      if (cache.entries[contentID]) continue;
      cache.entries[contentID] = createEntry(contentID, entry.content);
      await writeCacheFile(sessionID, cache);
      return contentID;
    }
    throw new Error("Unable to allocate a unique omission Content ID.");
  });
}

export async function copyCache(
  sourceSessionID: string,
  targetSessionID: string,
): Promise<void> {
  const cache = await withSessionMutation(
    sourceSessionID,
    async () => (await readCacheFile(sourceSessionID)) ?? createEmptyCache(),
  );
  await withSessionMutation(targetSessionID, async () => {
    await readCacheFile(targetSessionID);
    await writeCacheFile(targetSessionID, cache);
  });
}

export async function deleteCache(sessionID: string): Promise<void> {
  await withSessionMutation(sessionID, async () => {
    await rm(cachePath(sessionID), { force: true });
  });
}

export async function readOmittedContent(
  sessionID: string,
  contentID: string,
): Promise<string | null> {
  if (LegacyContentIDSchema.safeParse(contentID).success) {
    throw new OmissionIntegrityError(
      `Unverifiable legacy omission ${contentID} is unavailable because v1 bare entries have no cryptographic binding.`,
    );
  }

  const sourceSessionID = temporaryOmissionSources.get(sessionID) ?? sessionID;
  const cache = await readCacheFile(sourceSessionID);
  if (!cache) return null;

  if (QualifiedContentIDSchema.safeParse(contentID).success) {
    const entry = cache.entries[contentID];
    if (!entry) return null;
    return entry.content;
  }
  return null;
}

export function installTemporaryOmissionSource(
  temporarySessionID: string,
  sourceSessionID: string,
): () => void {
  temporaryOmissionSources.set(temporarySessionID, sourceSessionID);
  return () => {
    if (temporaryOmissionSources.get(temporarySessionID) === sourceSessionID) {
      temporaryOmissionSources.delete(temporarySessionID);
    }
  };
}

export async function readCache(
  sessionID: string,
): Promise<OmissionCache | null> {
  return readCacheFile(sessionID);
}

export async function writeCache(
  sessionID: string,
  cache: OmissionCache,
): Promise<void> {
  const parsed = OmissionCacheSchema.safeParse(cache);
  if (!parsed.success) {
    throw new Error(`Invalid omission cache for session ${sessionID}.`, {
      cause: parsed.error,
    });
  }
  await withSessionMutation(sessionID, async () => {
    await readCacheFile(sessionID);
    await writeCacheFile(sessionID, parsed.data);
  });
}

export function cachePath(sessionID: string): string {
  return `${pluginStorageDirectory()}/${sessionID}.json`;
}

function createEmptyCache(): OmissionCache {
  return { version: 2, entries: {}, legacy: { entries: {} } };
}

async function readCacheFile(sessionID: string): Promise<OmissionCache | null> {
  const file = Bun.file(cachePath(sessionID));
  if (!(await file.exists())) return null;

  let value: unknown;
  try {
    value = JSON.parse(await file.text());
  } catch (error) {
    throw new Error(`Invalid omission cache for session ${sessionID}.`, {
      cause: error,
    });
  }

  const current = OmissionCacheSchema.safeParse(value);
  if (current.success) {
    assertCacheIntegrity(current.data);
    return current.data;
  }
  const legacy = LegacyOmissionCacheSchema.safeParse(value);
  if (legacy.success) {
    return {
      version: 2,
      entries: {},
      legacy: { entries: legacy.data.entries },
    };
  }
  throw new Error(`Invalid omission cache for session ${sessionID}.`, {
    cause: current.error,
  });
}

async function writeCacheFile(
  sessionID: string,
  cache: OmissionCache,
): Promise<void> {
  assertCacheIntegrity(cache);
  const filePath = cachePath(sessionID);
  await mkdir(dirname(filePath), { recursive: true });
  const temporaryPath = `${filePath}.${process.pid}.${randomUUID()}.tmp`;
  try {
    await writeFile(temporaryPath, `${JSON.stringify(cache, null, 2)}\n`);
    await rename(temporaryPath, filePath);
  } catch (error) {
    await Bun.file(temporaryPath)
      .delete()
      .catch(() => undefined);
    throw error;
  }
}

async function withSessionMutation<T>(
  sessionID: string,
  operation: () => Promise<T>,
): Promise<T> {
  const previous = mutationQueues.get(sessionID) ?? Promise.resolve();
  let release!: () => void;
  const gate = new Promise<void>(resolve => {
    release = resolve;
  });
  const queued = previous.catch(() => undefined).then(() => gate);
  mutationQueues.set(sessionID, queued);
  await previous.catch(() => undefined);
  try {
    return await operation();
  } finally {
    release();
    if (mutationQueues.get(sessionID) === queued)
      mutationQueues.delete(sessionID);
  }
}

function createToken(): string {
  return randomBytes(16).toString("base64url");
}

function formatContentID(sessionID: string, token: string): string {
  if (!/^[A-Za-z0-9_-]{22}$/.test(token)) {
    throw new Error("Omission token must be 128-bit base64url text.");
  }
  const prefix = sessionID.slice(-12);
  if (!/^[A-Za-z0-9_-]{1,12}$/.test(prefix)) {
    throw new Error(
      "Session ID cannot be represented in an omission Content ID.",
    );
  }
  return `${prefix}:omitted-${token}`;
}

function createEntry(
  contentID: string,
  content: string,
): OmissionCache["entries"][string] {
  return {
    content,
    sha256: contentSha256(contentID, content),
  };
}

function assertEntryIntegrity(
  contentID: string,
  entry: OmissionCache["entries"][string],
): void {
  const sha256 = contentSha256(contentID, entry.content);
  if (entry.sha256 !== sha256) {
    throw new OmissionIntegrityError(
      `Omission ${contentID} is unavailable: cached content failed its SHA-256 integrity check.`,
    );
  }
}

function assertCacheIntegrity(cache: OmissionCache): void {
  for (const [contentID, entry] of Object.entries(cache.entries)) {
    assertEntryIntegrity(contentID, entry);
  }
}

function contentSha256(contentID: string, content: string): string {
  const id = Buffer.from(contentID);
  const idLength = Buffer.allocUnsafe(4);
  idLength.writeUInt32BE(id.length);
  return createHash("sha256")
    .update("magic-compact:omission-entry:v2\0")
    .update(idLength)
    .update(id)
    .update(Buffer.from(content, "utf16le"))
    .digest("hex");
}

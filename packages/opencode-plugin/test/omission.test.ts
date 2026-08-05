import { beforeAll, afterAll, describe, expect, test } from "bun:test";
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import {
  OmissionIntegrityError,
  allocateOmission,
  cachePath,
  copyCache,
  installTemporaryOmissionSource,
  omissionReadDiagnostic,
  readCache,
  readOmittedContent,
  writeCache,
} from "../src/storage/omission";

let storageDirectory: string;
const originalDataHome = process.env.XDG_DATA_HOME;

beforeAll(async () => {
  storageDirectory = await mkdtemp(join(tmpdir(), "magic-compact-omission-"));
  process.env.XDG_DATA_HOME = storageDirectory;
});

afterAll(async () => {
  if (originalDataHome === undefined) delete process.env.XDG_DATA_HOME;
  else process.env.XDG_DATA_HOME = originalDataHome;
  await rm(storageDirectory, { recursive: true, force: true });
});

describe("OpenCode omission storage", () => {
  test("allocates session-qualified random IDs with SHA-256 integrity metadata", async () => {
    const sessionID = "ses_1234567890abcdefghijkl";
    const content = "three bytes: \u20ac";

    const contentID = await allocateOmission(sessionID, { content });
    const cache = await readCache(sessionID);

    expect(contentID).toMatch(/^abcdefghijkl:omitted-[A-Za-z0-9_-]{22}$/);
    expect(cache).toEqual({
      version: 2,
      entries: {
        [contentID]: {
          content,
          sha256: contentSha256(contentID, content),
        },
      },
      legacy: { entries: {} },
    });
  });

  test("retries a generated collision without overwriting existing bytes", async () => {
    const sessionID = "ses_collision_retry";
    const firstToken = "A".repeat(22);
    const secondToken = "B".repeat(22);
    await allocateOmission(sessionID, { content: "first" }, () => firstToken);
    const tokens = [firstToken, secondToken];

    const secondID = await allocateOmission(
      sessionID,
      { content: "second" },
      () => tokens.shift()!,
    );

    expect(secondID).toEndWith(secondToken);
    expect(
      await readOmittedContent(
        sessionID,
        `${sessionID.slice(-12)}:omitted-${firstToken}`,
      ),
    ).toBe("first");
    expect(await readOmittedContent(sessionID, secondID)).toBe("second");
  });

  test("binds distinct lone-surrogate JavaScript strings without normalization", async () => {
    const sessionID = "ses_lone_surrogate_write";
    const d800 = "\ud800";
    const d801 = "\ud801";
    const d800ID = await allocateOmission(sessionID, { content: d800 }, () =>
      "A".repeat(22),
    );
    const d801ID = await allocateOmission(sessionID, { content: d801 }, () =>
      "B".repeat(22),
    );
    const cache = (await readCache(sessionID))!;

    expect(cache.entries[d800ID]!.sha256).toBe(contentSha256(d800ID, d800));
    expect(cache.entries[d801ID]!.sha256).toBe(contentSha256(d801ID, d801));
    expect(cache.entries[d800ID]!.sha256).not.toBe(
      cache.entries[d801ID]!.sha256,
    );
    expect(await readOmittedContent(sessionID, d800ID)).toBe(d800);
    expect(await readOmittedContent(sessionID, d801ID)).toBe(d801);
  });

  test("rejects U+D800 to U+D801 content substitution on read", async () => {
    const sessionID = "ses_lone_surrogate_read_corruption";
    const contentID = await allocateOmission(
      sessionID,
      { content: "\ud800" },
      () => "R".repeat(22),
    );
    const cache = (await readCache(sessionID))!;
    cache.entries[contentID]!.content = "\ud801";
    await writeRawCache(sessionID, `${JSON.stringify(cache, null, 2)}\n`);

    await expect(
      readOmittedContent(sessionID, contentID),
    ).rejects.toBeInstanceOf(OmissionIntegrityError);
  });

  test("rejects U+D800 to U+D801 content substitution before copy", async () => {
    const source = "ses_lone_surrogate_copy_source";
    const target = "ses_lone_surrogate_copy_target";
    const contentID = await allocateOmission(
      source,
      { content: "\ud800" },
      () => "C".repeat(22),
    );
    const cache = (await readCache(source))!;
    cache.entries[contentID]!.content = "\ud801";
    await writeRawCache(source, `${JSON.stringify(cache, null, 2)}\n`);

    await expect(copyCache(source, target)).rejects.toBeInstanceOf(
      OmissionIntegrityError,
    );
    expect(await Bun.file(cachePath(target)).exists()).toBeFalse();
  });

  test("binds complete entry records to their exact Content IDs across read, copy, and write", async () => {
    const source = "ses_entry_swap_source";
    const copied = "ses_entry_swap_valid_copy";
    const copyTarget = "ses_entry_swap_copy_target";
    const writeTarget = "ses_entry_swap_write_target";
    const firstID = await allocateOmission(
      source,
      { content: "first content" },
      () => "A".repeat(22),
    );
    const secondID = await allocateOmission(
      source,
      { content: "second content" },
      () => "B".repeat(22),
    );
    const valid = (await readCache(source))!;

    await writeCache(copied, structuredClone(valid));
    expect(await readOmittedContent(copied, firstID)).toBe("first content");
    expect(await readOmittedContent(copied, secondID)).toBe("second content");

    const swapped = structuredClone(valid);
    [swapped.entries[firstID], swapped.entries[secondID]] = [
      swapped.entries[secondID]!,
      swapped.entries[firstID]!,
    ];
    await writeRawCache(source, `${JSON.stringify(swapped, null, 2)}\n`);

    await expect(readOmittedContent(source, firstID)).rejects.toBeInstanceOf(
      OmissionIntegrityError,
    );
    await expect(copyCache(source, copyTarget)).rejects.toBeInstanceOf(
      OmissionIntegrityError,
    );
    expect(await Bun.file(cachePath(copyTarget)).exists()).toBeFalse();

    await allocateOmission(writeTarget, { content: "preserve target" }, () =>
      "T".repeat(22),
    );
    const originalTarget = await Bun.file(cachePath(writeTarget)).text();
    await expect(writeCache(writeTarget, swapped)).rejects.toBeInstanceOf(
      OmissionIntegrityError,
    );
    expect(await Bun.file(cachePath(writeTarget)).text()).toBe(originalTarget);
  });

  test("serializes same-session allocations while leaving distinct sessions independent", async () => {
    const token = (value: number) => value.toString().padStart(22, "0");
    let index = 0;
    const ids = await Promise.all(
      Array.from({ length: 20 }, (_, value) =>
        allocateOmission("ses_concurrent", { content: String(value) }, () =>
          token(index++),
        ),
      ),
    );

    expect(new Set(ids).size).toBe(20);
    expect(
      Object.keys((await readCache("ses_concurrent"))!.entries),
    ).toHaveLength(20);
  });

  test("does not resolve an old ID to replacement bytes after cache loss", async () => {
    const sessionID = "ses_cache_loss";
    const oldID = await allocateOmission(sessionID, { content: "old" }, () =>
      "A".repeat(22),
    );
    await rm(cachePath(sessionID));
    await allocateOmission(sessionID, { content: "replacement" }, () =>
      "B".repeat(22),
    );

    expect(await readOmittedContent(sessionID, oldID)).toBeNull();
  });

  test("fails loudly on malformed cache data without overwriting it", async () => {
    const sessionID = "ses_malformed";
    const malformed = '{"version":2,"entries":"wrong"}\n';
    await writeRawCache(sessionID, malformed);

    await expect(
      allocateOmission(sessionID, { content: "must not replace" }),
    ).rejects.toThrow("Invalid omission cache");
    expect(await Bun.file(cachePath(sessionID)).text()).toBe(malformed);
  });

  test("validates every v2 entry before allocating and leaves a corrupt cache unchanged", async () => {
    const sessionID = "ses_invalid_allocate";
    const contentID = qualifiedContentID(sessionID, "A");
    const corrupt = rawV2Cache({
      [contentID]: v2Entry(contentID, "existing", {
        sha256: "0".repeat(64),
      }),
    });
    await writeRawCache(sessionID, corrupt);

    await expect(
      allocateOmission(sessionID, { content: "must not append" }),
    ).rejects.toBeInstanceOf(OmissionIntegrityError);
    expect(await Bun.file(cachePath(sessionID)).text()).toBe(corrupt);
  });

  test("copies an absent source as an explicit empty v2 backup cache", async () => {
    await copyCache("ses_absent", "ses_empty_backup");

    expect(await readCache("ses_empty_backup")).toEqual({
      version: 2,
      entries: {},
      legacy: { entries: {} },
    });
  });

  test("refuses to overwrite a malformed backup cache during copy", async () => {
    const target = "ses_malformed_copy_target";
    const malformed = "not-json\n";
    await writeRawCache(target, malformed);

    await expect(copyCache("ses_absent_copy_source", target)).rejects.toThrow(
      "Invalid omission cache",
    );
    expect(await Bun.file(cachePath(target)).text()).toBe(malformed);
  });

  test("validates every source entry before copying and preserves the target bytes", async () => {
    const source = "ses_invalid_copy_source";
    const target = "ses_invalid_copy_target";
    const sourceID = qualifiedContentID(source, "C");
    const corruptSource = rawV2Cache({
      [sourceID]: v2Entry(sourceID, "corrupt source", {
        sha256: "0".repeat(64),
      }),
    });
    await writeRawCache(source, corruptSource);
    await allocateOmission(target, { content: "original target" }, () =>
      "T".repeat(22),
    );
    const originalTarget = await Bun.file(cachePath(target)).text();

    await expect(copyCache(source, target)).rejects.toBeInstanceOf(
      OmissionIntegrityError,
    );
    expect(await Bun.file(cachePath(source)).text()).toBe(corruptSource);
    expect(await Bun.file(cachePath(target)).text()).toBe(originalTarget);
  });

  test("authorizes only the current cache unless a scoped temporary mapping is installed", async () => {
    const source = "ses_mapping_source";
    const other = "ses_mapping_other";
    const temporary = "ses_mapping_temporary";
    const contentID = await allocateOmission(
      source,
      { content: "source bytes" },
      () => "S".repeat(22),
    );

    expect(await readOmittedContent(other, contentID)).toBeNull();
    const clear = installTemporaryOmissionSource(temporary, source);
    expect(await readOmittedContent(temporary, contentID)).toBe("source bytes");
    clear();
    expect(await readOmittedContent(temporary, contentID)).toBeNull();
  });

  test("preserves valid v1 bytes when migrating and copying but never reads them", async () => {
    const source = "ses_legacy_migration_source";
    const target = "ses_legacy_migration_target";
    const legacyEntries = {
      "omitted-001": { content: "legacy \ud800 bytes" },
    };
    await writeLegacyCache(source, legacyEntries);

    await copyCache(source, target);
    const newID = await allocateOmission(source, { content: "new" }, () =>
      "N".repeat(22),
    );

    expect((await readCache(source))!.legacy.entries).toEqual(legacyEntries);
    expect((await readCache(target))!.legacy.entries).toEqual(legacyEntries);
    expect(newID).toMatch(/:omitted-[A-Za-z0-9_-]{22}$/);
    expect(newID).not.toBe("omitted-236");
    for (const sessionID of [source, target]) {
      await expect(
        readOmittedContent(sessionID, "omitted-001"),
      ).rejects.toThrow(
        "Unverifiable legacy omission omitted-001 is unavailable",
      );
    }
  });

  test("never returns collision-prone or missing bare legacy IDs", async () => {
    const sessionID = "ses_legacy_collision";
    await writeLegacyCache(sessionID, {
      "omitted-001": { content: "historical first" },
      "omitted-002": { content: "historical second" },
    });

    for (const contentID of ["omitted-001", "omitted-002", "omitted-235"]) {
      await expect(readOmittedContent(sessionID, contentID)).rejects.toThrow(
        `Unverifiable legacy omission ${contentID} is unavailable because v1 bare entries have no cryptographic binding.`,
      );
    }
  });

  test("rejects a read when any v2 entry fails integrity, not only the requested entry", async () => {
    const sessionID = "ses_v2_corrupt";
    const requestedID = qualifiedContentID(sessionID, "R");
    const corruptID = qualifiedContentID(sessionID, "C");
    await writeRawCache(
      sessionID,
      rawV2Cache({
        [requestedID]: v2Entry(requestedID, "requested"),
        [corruptID]: v2Entry(corruptID, "corrupt", {
          sha256: "0".repeat(64),
        }),
      }),
    );

    await expect(
      readOmittedContent(sessionID, requestedID),
    ).rejects.toBeInstanceOf(OmissionIntegrityError);
    await expect(
      readOmittedContent(sessionID, "not-an-id"),
    ).rejects.toBeInstanceOf(OmissionIntegrityError);
  });

  test("rejects an integrity-invalid supplied write and preserves the original file byte-for-byte", async () => {
    const sessionID = "ses_invalid_write";
    await allocateOmission(sessionID, { content: "original" }, () =>
      "O".repeat(22),
    );
    const original = await Bun.file(cachePath(sessionID)).text();
    const replacementID = qualifiedContentID(sessionID, "W");

    await expect(
      writeCache(sessionID, {
        version: 2,
        entries: {
          [replacementID]: v2Entry(replacementID, "replacement", {
            sha256: "0".repeat(64),
          }),
        },
        legacy: { entries: {} },
      }),
    ).rejects.toBeInstanceOf(OmissionIntegrityError);

    expect(await Bun.file(cachePath(sessionID)).text()).toBe(original);
    expect(await readOmittedContent(sessionID, replacementID)).toBeNull();
  });

  test("rejects the obsolete v2 byteLength field instead of migrating an unshipped schema", async () => {
    const sessionID = "ses_obsolete_byte_length";
    const contentID = qualifiedContentID(sessionID, "B");
    const entry = v2Entry(contentID, "existing");
    await writeRawCache(
      sessionID,
      JSON.stringify({
        version: 2,
        entries: {
          [contentID]: { ...entry, byteLength: 8 },
        },
        legacy: { entries: {} },
      }),
    );

    await expect(readCache(sessionID)).rejects.toThrow(
      "Invalid omission cache",
    );
  });

  test("rejects non-qualified v2 entry IDs before reading or writing bytes", async () => {
    const sessionID = "ses_invalid_entry_id";
    const invalid = rawV2Cache({
      "omitted-001": v2Entry("omitted-001", "must stay unavailable"),
    });
    await writeRawCache(sessionID, invalid);

    await expect(readCache(sessionID)).rejects.toThrow(
      "Invalid omission cache",
    );
    expect(await Bun.file(cachePath(sessionID)).text()).toBe(invalid);
  });

  test("formats explicit integrity and cache-unavailable tool diagnostics", () => {
    expect(
      omissionReadDiagnostic(
        "omitted-001",
        new OmissionIntegrityError(
          "legacy content has no cryptographic binding",
        ),
      ),
    ).toContain("integrity verification failed");
    const unsafeDetail = `/home/ivan/.local/share/session-secret.json`;
    const diagnostic = omissionReadDiagnostic(
      "qualified:omitted-id",
      new Error(`EACCES: ${unsafeDetail}`),
    );
    expect(diagnostic).toBe(
      "Omitted content unavailable for Content ID qualified:omitted-id: the omission cache could not be read safely.",
    );
    expect(diagnostic).not.toContain(unsafeDetail);
    expect(diagnostic).not.toContain("EACCES");
  });

  test("uses atomic replacement without leaving temporary cache files", async () => {
    const sessionID = "ses_atomic";
    await allocateOmission(sessionID, { content: "atomic" });
    const names = await readdir(dirname(cachePath(sessionID)));

    expect(names.filter(name => name.includes(`${sessionID}.json.`))).toEqual(
      [],
    );
  });

  test("preserves exact IDs and bytes through repeated backup copies and later allocations", async () => {
    const sourceID = await allocateOmission(
      "ses_repeat_source",
      { content: "original" },
      () => "R".repeat(22),
    );
    await copyCache("ses_repeat_source", "ses_repeat_backup_1");
    await copyCache("ses_repeat_backup_1", "ses_repeat_backup_2");
    await allocateOmission("ses_repeat_source", { content: "later" }, () =>
      "L".repeat(22),
    );

    expect(await readOmittedContent("ses_repeat_backup_2", sourceID)).toBe(
      "original",
    );
  });
});

async function writeRawCache(
  sessionID: string,
  content: string,
): Promise<void> {
  await mkdir(dirname(cachePath(sessionID)), { recursive: true });
  await Bun.write(cachePath(sessionID), content);
}

async function writeLegacyCache(
  sessionID: string,
  entries: Record<string, { content: string }>,
): Promise<void> {
  await writeRawCache(
    sessionID,
    JSON.stringify({ version: 1, nextId: 236, entries }),
  );
}

function qualifiedContentID(sessionID: string, token: string): string {
  return `${sessionID.slice(-12)}:omitted-${token.repeat(22)}`;
}

function v2Entry(
  contentID: string,
  content: string,
  overrides: Partial<{
    sha256: string;
  }> = {},
) {
  return {
    content,
    sha256: contentSha256(contentID, content),
    ...overrides,
  };
}

function rawV2Cache(
  entries: Record<string, ReturnType<typeof v2Entry>>,
): string {
  return `${JSON.stringify({ version: 2, entries, legacy: { entries: {} } }, null, 2)}\n`;
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

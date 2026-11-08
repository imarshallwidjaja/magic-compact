import { beforeAll, afterAll, describe, expect, spyOn, test } from "bun:test";
import { createHash } from "node:crypto";
import * as fs from "node:fs/promises";
import { chmod, mkdir, mkdtemp, readdir, rm, stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import {
  OmissionIntegrityError,
  cachePath,
  commitPlannedOmission,
  copyCache,
  omissionReadDiagnostic,
  planOmission,
  readOmittedContent,
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
  test("plans session-qualified random IDs and commits SHA-256 integrity metadata", async () => {
    const sessionID = "ses_1234567890abcdefghijkl";
    const content = "three bytes: \u20ac";

    const planned = planOmission(sessionID, { content });
    await commitPlannedOmission(sessionID, planned);
    const { contentID } = planned;

    expect(contentID).toMatch(/^abcdefghijkl:omitted-[A-Za-z0-9_-]{22}$/);
    expect(await readRawCache(sessionID)).toEqual({
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

  test("refuses a duplicate planned Content ID without overwriting the committed entry", async () => {
    const sessionID = "ses_collision_refusal";
    const contentID = await commitContent(sessionID, "first", "A");
    const original = await Bun.file(cachePath(sessionID)).text();

    await expect(
      commitPlannedOmission(sessionID, { contentID, content: "second" }),
    ).rejects.toThrow("Planned omission Content ID collided before pruning.");

    expect(await Bun.file(cachePath(sessionID)).text()).toBe(original);
    expect(await readOmittedContent(sessionID, contentID)).toBe("first");
  });

  test("binds distinct lone-surrogate JavaScript strings without normalization", async () => {
    const sessionID = "ses_lone_surrogate_write";
    const d800 = "\ud800";
    const d801 = "\ud801";
    const d800ID = await commitContent(sessionID, d800, "A");
    const d801ID = await commitContent(sessionID, d801, "B");
    const cache = await readRawCache(sessionID);

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
    const contentID = await commitContent(sessionID, "\ud800", "R");
    const cache = await readRawCache(sessionID);
    cache.entries[contentID]!.content = "\ud801";
    await writeRawCache(sessionID, `${JSON.stringify(cache, null, 2)}\n`);

    await expect(
      readOmittedContent(sessionID, contentID),
    ).rejects.toBeInstanceOf(OmissionIntegrityError);
  });

  test("rejects U+D800 to U+D801 content substitution before copy", async () => {
    const source = "ses_lone_surrogate_copy_source";
    const target = "ses_lone_surrogate_copy_target";
    const contentID = await commitContent(source, "\ud800", "C");
    const cache = await readRawCache(source);
    cache.entries[contentID]!.content = "\ud801";
    await writeRawCache(source, `${JSON.stringify(cache, null, 2)}\n`);

    await expect(copyCache(source, target)).rejects.toBeInstanceOf(
      OmissionIntegrityError,
    );
    expect(await Bun.file(cachePath(target)).exists()).toBeFalse();
  });

  test("binds complete entry records to their exact Content IDs across read, copy, and commit", async () => {
    const source = "ses_entry_swap_source";
    const copied = "ses_entry_swap_valid_copy";
    const copyTarget = "ses_entry_swap_copy_target";
    const firstID = await commitContent(source, "first content", "A");
    const secondID = await commitContent(source, "second content", "B");
    const valid = await readRawCache(source);

    await copyCache(source, copied);
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

    const swappedBytes = await Bun.file(cachePath(source)).text();
    await expect(
      commitContent(source, "must not append", "T"),
    ).rejects.toBeInstanceOf(OmissionIntegrityError);
    expect(await Bun.file(cachePath(source)).text()).toBe(swappedBytes);
  });

  test("serializes concurrent same-session commits without losing an entry", async () => {
    const sessionID = "ses_concurrent";
    const planned = Array.from({ length: 20 }, (_, value) =>
      planOmission(sessionID, { content: String(value) }),
    );

    await Promise.all(
      planned.map(omission => commitPlannedOmission(sessionID, omission)),
    );

    const entries = (await readRawCache(sessionID)).entries;
    expect(Object.keys(entries).sort()).toEqual(
      planned.map(omission => omission.contentID).sort(),
    );
    for (const omission of planned) {
      expect(await readOmittedContent(sessionID, omission.contentID)).toBe(
        omission.content,
      );
    }
  });

  test("does not resolve an old ID to replacement bytes after cache loss", async () => {
    const sessionID = "ses_cache_loss";
    const oldID = await commitContent(sessionID, "old", "A");
    await rm(cachePath(sessionID));
    await commitContent(sessionID, "replacement", "B");

    expect(await readOmittedContent(sessionID, oldID)).toBeNull();
  });

  test("fails loudly on malformed cache data without overwriting it", async () => {
    const sessionID = "ses_malformed";
    const malformed = '{"version":2,"entries":"wrong"}\n';
    await writeRawCache(sessionID, malformed);

    await expect(
      commitContent(sessionID, "must not replace", "M"),
    ).rejects.toThrow(`Invalid omission cache for session ${sessionID}.`);
    expect(await Bun.file(cachePath(sessionID)).text()).toBe(malformed);
  });

  test("validates every v2 entry before committing and leaves a corrupt cache unchanged", async () => {
    const sessionID = "ses_invalid_allocate";
    const contentID = qualifiedContentID(sessionID, "A");
    const corrupt = rawV2Cache({
      [contentID]: v2Entry(contentID, "existing", {
        sha256: "0".repeat(64),
      }),
    });
    await writeRawCache(sessionID, corrupt);

    await expect(
      commitContent(sessionID, "must not append", "N"),
    ).rejects.toBeInstanceOf(OmissionIntegrityError);
    expect(await Bun.file(cachePath(sessionID)).text()).toBe(corrupt);
  });

  test("copies an absent source as an explicit empty v2 backup cache", async () => {
    await copyCache("ses_absent", "ses_empty_backup");

    expect(await readRawCache("ses_empty_backup")).toEqual({
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
      `Invalid omission cache for session ${target}.`,
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
    await commitContent(target, "original target", "T");
    const originalTarget = await Bun.file(cachePath(target)).text();

    await expect(copyCache(source, target)).rejects.toBeInstanceOf(
      OmissionIntegrityError,
    );
    expect(await Bun.file(cachePath(source)).text()).toBe(corruptSource);
    expect(await Bun.file(cachePath(target)).text()).toBe(originalTarget);
  });

  test("authorizes omission reads only through the current session cache", async () => {
    const source = "ses_mapping_source";
    const other = "ses_mapping_other";
    const temporary = "ses_mapping_temporary";
    const contentID = await commitContent(source, "source bytes", "S");

    expect(await readOmittedContent(source, contentID)).toBe("source bytes");
    expect(await readOmittedContent(other, contentID)).toBeNull();
    expect(await readOmittedContent(temporary, contentID)).toBeNull();
  });

  test("preserves valid v1 bytes when migrating on commit and copy", async () => {
    const source = "ses_legacy_migration_source";
    const target = "ses_legacy_migration_target";
    const legacyEntries = {
      "omitted-001": { content: "legacy \ud800 bytes" },
    };
    await writeLegacyCache(source, legacyEntries);

    await copyCache(source, target);
    const newID = await commitContent(source, "new", "N");

    expect((await readRawCache(source)).legacy.entries).toEqual(legacyEntries);
    expect((await readRawCache(target)).legacy.entries).toEqual(legacyEntries);
    expect(await readOmittedContent(source, newID)).toBe("new");
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

    await expect(readOmittedContent(sessionID, contentID)).rejects.toThrow(
      `Invalid omission cache for session ${sessionID}.`,
    );
  });

  test("rejects non-qualified v2 entry IDs before reading or writing bytes", async () => {
    const sessionID = "ses_invalid_entry_id";
    const invalid = rawV2Cache({
      "omitted-001": v2Entry("omitted-001", "must stay unavailable"),
    });
    await writeRawCache(sessionID, invalid);

    await expect(
      readOmittedContent(sessionID, qualifiedContentID(sessionID, "Q")),
    ).rejects.toThrow(`Invalid omission cache for session ${sessionID}.`);
    await expect(commitContent(sessionID, "new", "Q")).rejects.toThrow(
      `Invalid omission cache for session ${sessionID}.`,
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
    ).toBe(
      "Omitted content unavailable for Content ID omitted-001: integrity verification failed. legacy content has no cryptographic binding",
    );
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

  test("never opens cache permissions and a failed replacement keeps the old file", async () => {
    const sessionID = "ses_cache_mode";
    const path = cachePath(sessionID);
    const mode = async () => (await stat(path)).mode & 0o777;
    await mkdir(dirname(path), { recursive: true });
    // A umask that strips owner read must not leave the first write unreadable.
    const umask = process.umask(0o400);
    try {
      const firstID = await commitContent(sessionID, "first", "F");
      expect(await mode()).toBe(0o600);
      expect(await readOmittedContent(sessionID, firstID)).toBe("first");
      await chmod(path, 0o640);
      await commitContent(sessionID, "second", "S");
      expect(await mode()).toBe(0o640);

      const before = await Bun.file(path).text();
      const renames = spyOn(fs, "rename").mockRejectedValue(
        Object.assign(new Error("EIO: rename failed"), { code: "EIO" }),
      );
      try {
        await expect(commitContent(sessionID, "third", "T")).rejects.toThrow(
          "EIO",
        );
      } finally {
        renames.mockRestore();
      }
      expect(await Bun.file(path).text()).toBe(before);
      expect(await mode()).toBe(0o640);
      expect(await readOmittedContent(sessionID, firstID)).toBe("first");
    } finally {
      process.umask(umask);
    }
    const names = await readdir(dirname(path));
    expect(names.filter(name => name.includes(`${sessionID}.json.`))).toEqual(
      [],
    );
  });

  test("preserves exact IDs and bytes through repeated copies and later commits", async () => {
    const sourceID = await commitContent("ses_repeat_source", "original", "R");
    await copyCache("ses_repeat_source", "ses_repeat_backup_1");
    await copyCache("ses_repeat_backup_1", "ses_repeat_backup_2");
    await commitContent("ses_repeat_source", "later", "L");

    expect(await readOmittedContent("ses_repeat_backup_2", sourceID)).toBe(
      "original",
    );
  });
});

// Commits through the production write path with a fixed token, so a test
// can name the Content ID before it exists.
async function commitContent(
  sessionID: string,
  content: string,
  token: string,
): Promise<string> {
  const contentID = qualifiedContentID(sessionID, token);
  await commitPlannedOmission(sessionID, { contentID, content });
  return contentID;
}

async function readRawCache(sessionID: string) {
  return JSON.parse(await Bun.file(cachePath(sessionID)).text()) as {
    entries: Record<string, { content: string; sha256: string }>;
    legacy: { entries: Record<string, { content: string }> };
  };
}

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

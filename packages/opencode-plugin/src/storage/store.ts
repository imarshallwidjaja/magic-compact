import { randomUUID } from "node:crypto";
import { mkdir, open, rename, rm, stat } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname } from "node:path";

export function pluginStorageDirectory(): string {
  const base = Bun.env.XDG_DATA_HOME ?? `${homedir()}/.local/share`;
  return `${base}/opencode/storage/magic-compact`;
}

// Replaces `filePath` with `value` as two-space-indented JSON and a final
// newline. The text goes to a new temporary file in the same directory that is
// renamed over the target, so a reader sees the old file or the new one and
// never a partial write. The temporary file is created exclusively and its
// permission bits are set explicitly before the rename, whatever the umask: a
// first write is owner read/write (0600) and a replacement keeps the
// permissions it replaces. On failure the target is unchanged and only the
// temporary file this call created is removed.
export async function replaceJSONFile(
  filePath: string,
  value: unknown,
): Promise<void> {
  await mkdir(dirname(filePath), { recursive: true });
  const temporaryPath = `${filePath}.${process.pid}.${randomUUID()}.tmp`;
  const file = await open(temporaryPath, "wx", 0o600);
  try {
    try {
      await file.writeFile(`${JSON.stringify(value, null, 2)}\n`);
      await file.chmod((await permissionBits(filePath)) ?? 0o600);
    } finally {
      await file.close();
    }
    await rename(temporaryPath, filePath);
  } catch (error) {
    await rm(temporaryPath, { force: true }).catch(() => undefined);
    throw error;
  }
}

// Undefined when no file exists yet.
async function permissionBits(filePath: string): Promise<number | undefined> {
  try {
    return (await stat(filePath)).mode & 0o777;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return undefined;
    throw error;
  }
}

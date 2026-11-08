// Checks the staged content of added, copied, modified and type-changed
// regular files with Prettier. The files, Prettier configuration and ignore
// files all come from one snapshot of the index (the one in GIT_INDEX_FILE
// when `git commit` sets it), written to a temporary directory that is removed
// afterwards. The index and the working tree are only read.
import { execFileSync } from "node:child_process";
import {
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import * as prettier from "prettier";

const SYMLINK = "120000";
const isRegularFile = (mode: string) => mode.startsWith("100");

const git = (args: string[], input?: string) =>
  execFileSync("git", args, { input, maxBuffer: 1024 ** 3 });

// `ls-files -s -z` yields "<mode> <object> <stage>\t<path>" per entry.
const entries = git(["ls-files", "--stage", "-z"])
  .toString("utf8")
  .split("\0")
  .slice(0, -1)
  .map(entry => {
    const tab = entry.indexOf("\t");
    const [mode, object, stage] = entry.slice(0, tab).split(" ");
    return { mode: mode!, object: object!, stage, path: entry.slice(tab + 1) };
  })
  .filter(
    entry =>
      entry.stage === "0"
      && (isRegularFile(entry.mode) || entry.mode === SYMLINK),
  );

// `cat-file --batch` yields "<object> <type> <size>\n<content>\n" per object.
const batch = git(
  ["cat-file", "--batch"],
  entries.map(entry => `${entry.object}\n`).join(""),
);
const contents: Buffer[] = [];
for (let offset = 0; contents.length < entries.length;) {
  const headerEnd = batch.indexOf(0x0a, offset);
  const [object, type, size] = batch
    .toString("utf8", offset, headerEnd)
    .split(" ");
  if (type !== "blob") throw new Error(`Index object ${object} is not a blob.`);
  const end = headerEnd + 1 + Number(size);
  contents.push(batch.subarray(headerEnd + 1, end));
  offset = end + 1;
}

// `--raw -z` yields ":<old mode> <new mode> <old sha> <new sha> <status>",
// then the path.
const fields = git([
  "diff",
  "--cached",
  "--raw",
  "-z",
  "--no-abbrev",
  "--no-renames",
  "--diff-filter=ACMT",
])
  .toString("utf8")
  .split("\0");
const staged: string[] = [];
for (let i = 0; i < fields.length - 1; i += 2) {
  const newMode = fields[i]!.slice(1).split(" ")[1]!;
  if (isRegularFile(newMode)) staged.push(fields[i + 1]!);
}

const snapshot = mkdtempSync(join(tmpdir(), "staged-format-"));
const unformatted: string[] = [];
let failed = false;
try {
  // Regular files first, so that no file is written through a symlink.
  const order = entries.map((_, index) => index);
  order.sort(
    (a, b) =>
      Number(entries[a]!.mode === SYMLINK)
      - Number(entries[b]!.mode === SYMLINK),
  );
  for (const index of order) {
    const { mode, path } = entries[index]!;
    const target = join(snapshot, path);
    mkdirSync(dirname(target), { recursive: true });
    if (mode === SYMLINK) symlinkSync(contents[index]!.toString(), target);
    else writeFileSync(target, contents[index]!);
  }
  // Prettier stops its .editorconfig search at the directory holding `.git`,
  // which is the repository root in a checkout.
  writeFileSync(join(snapshot, ".git"), "");

  for (const path of staged) {
    const file = join(snapshot, path);
    const info = await prettier.getFileInfo(file, {
      ignorePath: [
        join(snapshot, ".gitignore"),
        join(snapshot, ".prettierignore"),
      ],
    });
    if (info.ignored || info.inferredParser === null) continue;
    const options = await prettier.resolveConfig(file, { editorconfig: true });
    const source = readFileSync(file, "utf8");
    try {
      if (!(await prettier.check(source, { ...options, filepath: file })))
        unformatted.push(path);
    } catch (error) {
      failed = true;
      console.error(`[error] ${path}: ${(error as Error).message}`);
    }
  }
} finally {
  rmSync(snapshot, { recursive: true, force: true });
}

for (const path of unformatted) console.error(`[warn] ${path}`);
if (unformatted.length > 0) {
  console.error(
    "Staged content is not formatted. Run `bun run format` (or `npx prettier --write` on these files), then stage the result.",
  );
}
if (failed || unformatted.length > 0) process.exit(1);

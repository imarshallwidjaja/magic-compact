import { afterAll, expect, test } from "bun:test";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import {
  chmodSync,
  copyFileSync,
  lstatSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  readlinkSync,
  rmSync,
  symlinkSync,
  unlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

// Each case stages content in a disposable repository that carries this
// repository's Prettier configuration and ignore files, then runs
// scripts/check-staged-format.ts there with this checkout's Prettier. Its
// verdict must match the oracle, the Prettier CLI run on a checkout of the
// index alone, and must leave the index, working tree and HEAD unchanged.

const REPOSITORY = join(import.meta.dir, "../../..");
const CHECKER = join(REPOSITORY, "scripts/check-staged-format.ts");
const PRETTIER_CLI = join(REPOSITORY, "node_modules/prettier/bin/prettier.cjs");
const GIT = Bun.which("git")!;
const NODE = Bun.which("node")!;
const scratch = mkdtempSync(join(tmpdir(), "staged-format-test-"));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));

const GOOD = "export const answer = (value: number) => value + 1;\n";
const BAD = "export const answer   =   ( value:number )=>{return value+1}\n";
const NO_SEMICOLON = "export const answer = (value: number) => value + 1\n";
const INDENTED_2 = "export function f() {\n  return 1;\n}\n";
const INDENTED_4 = "export function f() {\n    return 1;\n}\n";
const EDITORCONFIG_4 = "[*]\nindent_style = space\nindent_size = 4\n";

type Fixture = {
  root: string;
  write: (path: string, text: string) => void;
  remove: (path: string) => void;
  link: (path: string, target: string) => void;
  git: (...args: string[]) => string;
  // Runs `git commit` with a pre-commit hook that runs the checker.
  commitWithHook: (...args: string[]) => {
    status: number | null;
    stderr: string;
  };
  check: () => { status: number | null; stderr: string; temporary: string[] };
  oracle: () => number | null;
  fingerprint: () => string;
};

let fixtureCount = 0;

function createFixture(): Fixture {
  const root = join(scratch, String(fixtureCount++));
  const work = join(root, "work");
  const temporary = join(root, "tmp");
  mkdirSync(join(work, "src"), { recursive: true });
  mkdirSync(temporary);
  const env = {
    PATH: process.env.PATH,
    HOME: root,
    TMPDIR: temporary,
    GIT_CONFIG_GLOBAL: "/dev/null",
    GIT_CONFIG_NOSYSTEM: "1",
    GIT_AUTHOR_NAME: "Fixture",
    GIT_AUTHOR_EMAIL: "fixture@example.invalid",
    GIT_COMMITTER_NAME: "Fixture",
    GIT_COMMITTER_EMAIL: "fixture@example.invalid",
  };
  const exec = (cwd: string, command: string, args: string[]) =>
    spawnSync(command, args, { cwd, env, encoding: "utf8" });
  const git = (...args: string[]) => {
    const result = exec(work, GIT, args);
    if (result.status !== 0)
      throw new Error(`git ${args.join(" ")}: ${result.stderr}`);
    return result.stdout;
  };
  const write = (path: string, text: string) =>
    writeFileSync(join(work, path), text);

  git("init", "-q", "-b", "main");
  for (const name of [".prettierrc", ".prettierignore", ".gitignore"])
    copyFileSync(join(REPOSITORY, name), join(work, name));
  write("src/base.ts", GOOD);
  git("add", "-A");
  git("commit", "-q", "-m", "initial");

  const hook = join(work, ".git/hooks/pre-commit");
  writeFileSync(hook, `#!/bin/sh\nexec "${process.execPath}" "${CHECKER}"\n`);
  chmodSync(hook, 0o755);

  let oracleCount = 0;
  return {
    root,
    write,
    remove: path => unlinkSync(join(work, path)),
    link: (path, target) => symlinkSync(target, join(work, path)),
    git,
    commitWithHook: (...args) => {
      const result = exec(work, GIT, ["commit", "-q", ...args]);
      return { status: result.status, stderr: result.stderr };
    },
    check: () => {
      const result = exec(work, process.execPath, [CHECKER]);
      return {
        status: result.status,
        stderr: result.stderr,
        temporary: readdirSync(temporary),
      };
    },
    oracle: () => {
      const checkout = join(root, `oracle-${oracleCount++}`);
      git("checkout-index", "-a", `--prefix=${checkout}/`);
      // A checkout has .git at its root, where Prettier stops looking for
      // .editorconfig.
      writeFileSync(join(checkout, ".git"), "");
      const paths = git(
        "diff",
        "--cached",
        "--name-only",
        "-z",
        "--no-renames",
        "--diff-filter=ACMT",
      )
        .split("\0")
        .filter(
          path => path !== "" && lstatSync(join(checkout, path)).isFile(),
        );
      if (paths.length === 0) return 0;
      return exec(checkout, NODE, [
        PRETTIER_CLI,
        "--check",
        "--ignore-unknown",
        ...paths,
      ]).status;
    },
    fingerprint: () => {
      const hash = createHash("sha256");
      hash.update(readFileSync(join(work, ".git/index")));
      hash.update(git("rev-parse", "HEAD"));
      const visit = (dir: string) => {
        for (const name of readdirSync(dir).sort()) {
          const path = join(dir, name);
          if (path === join(work, ".git")) continue;
          const stat = lstatSync(path);
          hash.update(`${path}\0${stat.mode}\0`);
          if (stat.isDirectory()) visit(path);
          else if (stat.isSymbolicLink()) hash.update(readlinkSync(path));
          else hash.update(readFileSync(path));
        }
      };
      visit(work);
      return hash.digest("hex");
    },
  };
}

const cases: [string, (f: Fixture) => void, string[] | null][] = [
  [
    "a formatted file",
    f => (f.write("src/a.ts", GOOD), f.git("add", "src/a.ts")),
    null,
  ],
  [
    "an unformatted file",
    f => (f.write("src/a.ts", BAD), f.git("add", "src/a.ts")),
    ["src/a.ts"],
  ],
  [
    "unformatted content staged under a formatted working copy",
    f => {
      f.write("src/a.ts", BAD);
      f.git("add", "src/a.ts");
      f.write("src/a.ts", GOOD);
    },
    ["src/a.ts"],
  ],
  [
    "formatted content staged under an unformatted working copy",
    f => {
      f.write("src/a.ts", GOOD);
      f.git("add", "src/a.ts");
      f.write("src/a.ts", BAD);
    },
    null,
  ],
  [
    "an unformatted file deleted from the working tree",
    f => {
      f.write("src/a.ts", BAD);
      f.git("add", "src/a.ts");
      f.remove("src/a.ts");
    },
    ["src/a.ts"],
  ],
  [
    "an unformatted file that only an unstaged .prettierignore entry names",
    f => {
      f.write("src/a.ts", BAD);
      f.git("add", "src/a.ts");
      f.write(".prettierignore", "src/a.ts\n");
    },
    ["src/a.ts"],
  ],
  [
    "a formatted file under an unstaged .prettierrc change",
    f => {
      f.write("src/a.ts", GOOD);
      f.git("add", "src/a.ts");
      f.write(".prettierrc", '{ "semi": false }\n');
    },
    null,
  ],
  [
    "an unformatted file named by a staged .prettierignore entry that the working tree drops",
    f => {
      f.write("src/a.ts", BAD);
      const ignore = f.git("show", "HEAD:.prettierignore");
      f.write(".prettierignore", `${ignore}src/a.ts\n`);
      f.git("add", "src/a.ts", ".prettierignore");
      f.write(".prettierignore", ignore);
    },
    null,
  ],
  [
    "a file formatted for a staged .prettierrc change that the working tree reverts",
    f => {
      const config = f.git("show", "HEAD:.prettierrc");
      f.write(".prettierrc", '{ "semi": false }\n');
      f.write("src/a.ts", NO_SEMICOLON);
      f.git("add", "src/a.ts", ".prettierrc");
      f.write(".prettierrc", config);
    },
    null,
  ],
  [
    "a file formatted for a staged .editorconfig that the working tree removes",
    f => {
      f.write(".editorconfig", `root = true\n${EDITORCONFIG_4}`);
      f.write("src/a.ts", INDENTED_4);
      f.git("add", "src/a.ts", ".editorconfig");
      f.remove(".editorconfig");
    },
    null,
  ],
  [
    "a formatted file below an .editorconfig outside the repository",
    f => {
      writeFileSync(join(f.root, ".editorconfig"), EDITORCONFIG_4);
      f.write("src/a.ts", INDENTED_2);
      f.git("add", "src/a.ts");
    },
    null,
  ],
  [
    "an unformatted node script whose working copy has no shebang",
    f => {
      f.write("tool", "#!/usr/bin/env node\nconst   x=1\n");
      f.git("add", "tool");
      f.write("tool", "plain text\n");
    },
    ["tool"],
  ],
  [
    "a symlink replaced by an unformatted file",
    f => {
      f.link("src/a.ts", "base.ts");
      f.git("add", "src/a.ts");
      f.git("commit", "-q", "--no-verify", "-m", "link");
      f.remove("src/a.ts");
      f.write("src/a.ts", BAD);
      f.git("add", "src/a.ts");
    },
    ["src/a.ts"],
  ],
  [
    "a file replaced by a symlink",
    f => {
      f.write("src/a.ts", GOOD);
      f.git("add", "src/a.ts");
      f.git("commit", "-q", "--no-verify", "-m", "file");
      f.remove("src/a.ts");
      f.link("src/a.ts", "base.ts");
      f.git("add", "src/a.ts");
    },
    null,
  ],
  [
    "a renamed unformatted file",
    f => {
      f.write("src/a.ts", BAD);
      f.git("add", "src/a.ts");
      f.git("commit", "-q", "--no-verify", "-m", "unformatted");
      f.git("mv", "src/a.ts", "src/renamed.ts");
    },
    ["src/renamed.ts"],
  ],
  [
    "unformatted files with a space and a line break in their names",
    f => {
      f.write("src/with space.ts", BAD);
      f.write("src/line\nbreak.ts", BAD);
      f.git("add", "src");
    },
    ["src/line\nbreak.ts", "src/with space.ts"],
  ],
  [
    "a force-added file that .gitignore names",
    f => {
      mkdirSync(join(f.root, "work/out"));
      f.write("out/a.ts", BAD);
      f.git("add", "-f", "out/a.ts");
    },
    null,
  ],
  [
    "a file of a type Prettier does not know",
    f => (f.write("src/a.unknown", "  x  \n"), f.git("add", "src/a.unknown")),
    null,
  ],
];

test.each(cases)("checks staged content: %s", (_name, arrange, unformatted) => {
  const fixture = createFixture();
  arrange(fixture);
  const before = fixture.fingerprint();

  const result = fixture.check();

  expect(fixture.fingerprint()).toBe(before);
  expect(result.temporary).toEqual([]);
  expect(result.status).toBe(unformatted ? 1 : 0);
  expect(result.status === 0).toBe(fixture.oracle() === 0);
  if (unformatted)
    expect(result.stderr).toStartWith(
      unformatted.map(path => `[warn] ${path}\n`).join(""),
    );
});

test("reports a syntax error in staged content", () => {
  const fixture = createFixture();
  fixture.write("src/a.ts", GOOD);
  fixture.git("add", "src/a.ts");
  fixture.write("src/a.ts", "export const = ;\n");
  fixture.git("add", "src/a.ts");
  fixture.write("src/a.ts", GOOD);
  const before = fixture.fingerprint();

  const result = fixture.check();

  expect(fixture.fingerprint()).toBe(before);
  expect(result.temporary).toEqual([]);
  expect(result.status).toBe(1);
  expect(result.stderr).toStartWith("[error] src/a.ts: ");
  expect(fixture.oracle()).not.toBe(0);
});

// `git commit --only` and `git commit -a` hand the hook a temporary index in
// GIT_INDEX_FILE; the checker must read that index and its configuration.
test.each([
  [
    "fails a commit of an unformatted file whose ignore entry is unstaged",
    (f: Fixture) => {
      f.write("src/bad.ts", BAD);
      f.git("add", "-N", "src/bad.ts");
      f.write(".prettierignore", "src/bad.ts\n");
      return ["--only", "-m", "bad", "--", "src/bad.ts"];
    },
    false,
  ],
  [
    "passes a commit of a formatted file while an unformatted file stays staged",
    (f: Fixture) => {
      f.write("src/bad.ts", BAD);
      f.git("add", "src/bad.ts");
      f.write("src/good.ts", GOOD);
      f.git("add", "-N", "src/good.ts");
      return ["--only", "-m", "good", "--", "src/good.ts"];
    },
    true,
  ],
])("%s", (_name, arrange, commits) => {
  const fixture = createFixture();
  const args = arrange(fixture);
  const head = fixture.git("rev-parse", "HEAD");

  const result = fixture.commitWithHook(...args);

  expect(result.status === 0).toBe(commits);
  expect(fixture.git("rev-parse", "HEAD") === head).toBe(!commits);
  if (!commits) expect(result.stderr).toStartWith("[warn] src/bad.ts\n");
});

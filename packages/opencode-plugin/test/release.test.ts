import { afterAll, expect, test } from "bun:test";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import {
  chmodSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  utimesSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

// Each case runs scripts/release-opencode.sh in a disposable repository whose
// origin is a local bare repository, with stand-ins for npm and bun and, where
// a case says so, a git wrapper that fails one call or stages a file, or a
// repository hook that changes the configuration, HEAD or a ref.

const RELEASE_SCRIPT = join(
  import.meta.dir,
  "../../../scripts/release-opencode.sh",
);
const MANIFEST = "packages/opencode-plugin/package.json";
const REAL_GIT = Bun.which("git")!;
const scratch = mkdtempSync(join(tmpdir(), "release-helper-"));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));

const FAKE_NPM = `#!/bin/sh
set -eu
printf 'npm %s\\n' "$*" >> "$CALL_LOG"
version=\${FAKE_NPM_VERSION:-$2}
node -e '
const fs = require("fs");
const manifest = JSON.parse(fs.readFileSync(process.argv[1], "utf8"));
manifest.version = process.argv[2];
fs.writeFileSync(process.argv[1], JSON.stringify(manifest, null, 2) + "\\n");
' packages/opencode-plugin/package.json "$version"
`;

const FAKE_BUN = `#!/bin/sh
set -eu
printf 'bun %s\\n' "$*" >> "$CALL_LOG"
version=$(node -p "require('./packages/opencode-plugin/package.json').version")
printf '{ "lock": "%s" }\\n' "$version" > bun.lock
# Stands in for a lifecycle script that bun install runs.
eval "\${FAKE_BUN_SCRIPT:-}"
`;

// FAKE_GIT_FAIL=<subcommand>:<n> fails the nth call of that subcommand.
// FAKE_GIT_STAGE_BEFORE_COMMIT stages a README edit just before a commit runs.
// FAKE_GIT_COMMIT_BEFORE_PUSH commits unreviewed.txt on the current branch
// just before a push runs.
const FAKE_GIT = `#!/bin/sh
set -eu
sub=
skip=
for arg in "$@"; do
  if [ -n "$skip" ]; then skip=; continue; fi
  case $arg in
    -c | -C) skip=1 ;;
    -*) ;;
    *) sub=$arg; break ;;
  esac
done
if [ -n "\${FAKE_GIT_FAIL:-}" ] && [ -n "$sub" ]; then
  count_file=$FAKE_GIT_COUNTS/$sub
  count=$(( $(cat "$count_file" 2>/dev/null || echo 0) + 1 ))
  printf '%s\\n' "$count" > "$count_file"
  if [ "$FAKE_GIT_FAIL" = "$sub:$count" ]; then
    echo "fatal: injected $sub failure" >&2
    exit 128
  fi
fi
if [ "$sub" = commit ] && [ -n "\${FAKE_GIT_STAGE_BEFORE_COMMIT:-}" ]; then
  printf 'staged before commit\\n' >> README.md
  "$REAL_GIT" add README.md
fi
if [ "$sub" = push ] && [ -n "\${FAKE_GIT_COMMIT_BEFORE_PUSH:-}" ]; then
  printf 'unreviewed\\n' > unreviewed.txt
  "$REAL_GIT" add unreviewed.txt
  "$REAL_GIT" commit -q -m "unreviewed late commit"
fi
exec "$REAL_GIT" "$@"
`;

type Fixture = {
  root: string;
  work: string;
  bare: string;
  run: (...args: string[]) => string;
  origin: (...args: string[]) => string;
  release: (
    env?: Record<string, string>,
    version?: string,
  ) => {
    status: number | null;
    stderr: string;
    calls: string;
  };
  // Environment that puts the git wrapper first on PATH.
  wrappedGit: (env: Record<string, string>) => Record<string, string>;
  // A bare clone of origin, for a push URL that differs from the fetch URL.
  cloneOrigin: (name: string) => string;
  // A bare clone of origin at work/upstream, ignored, for the relative push
  // URL "upstream".
  relativeOrigin: () => string;
  write: (path: string, text: string) => void;
  setVersion: (version: string) => void;
  // Installs a server-side hook in the bare origin.
  originHook: (name: string, script: string) => void;
  // Installs a hook in the work repository, which the helper's own Git
  // commands run.
  hook: (name: string, script: string) => void;
  // `git status --porcelain=v1 --untracked-files=all`, untrimmed.
  status: () => string;
  // Reads nothing that refreshes the index; the raw index bytes come first.
  state: () => string;
};

let fixtureCount = 0;

function refs(bare: string) {
  const result = spawnSync(
    REAL_GIT,
    [`--git-dir=${bare}`, "for-each-ref", "--format=%(refname) %(objectname)"],
    { encoding: "utf8" },
  );
  if (result.status !== 0) throw new Error(result.stderr);
  return result.stdout.trim();
}

function createFixture(): Fixture {
  const root = join(scratch, `release fixture-${fixtureCount++}`);
  const work = join(root, "work");
  const bare = join(root, "origin.git");
  const bin = join(root, "bin");
  const gitBin = join(root, "git-bin");
  const gitCounts = join(root, "git-counts");
  const callLog = join(root, "calls.log");
  mkdirSync(join(work, "packages/opencode-plugin"), { recursive: true });
  mkdirSync(bin);
  mkdirSync(gitBin);
  mkdirSync(gitCounts);
  for (const [dir, name, script] of [
    [bin, "npm", FAKE_NPM],
    [bin, "bun", FAKE_BUN],
    [gitBin, "git", FAKE_GIT],
  ] as const) {
    writeFileSync(join(dir, name), script);
    chmodSync(join(dir, name), 0o755);
  }

  const env = {
    PATH: `${bin}:${process.env.PATH}`,
    HOME: root,
    GIT_CONFIG_GLOBAL: "/dev/null",
    GIT_CONFIG_NOSYSTEM: "1",
    GIT_AUTHOR_NAME: "Fixture",
    GIT_AUTHOR_EMAIL: "fixture@example.invalid",
    GIT_COMMITTER_NAME: "Fixture",
    GIT_COMMITTER_EMAIL: "fixture@example.invalid",
    CALL_LOG: callLog,
  };
  const exec = (cwd: string, command: string, args: string[], extra = {}) =>
    spawnSync(command, args, {
      cwd,
      env: { ...env, ...extra },
      encoding: "utf8",
    });
  const git = (cwd: string, args: string[]) => {
    const result = exec(cwd, REAL_GIT, args);
    if (result.status !== 0)
      throw new Error(`git ${args.join(" ")}: ${result.stderr}`);
    return result.stdout.trim();
  };
  const write = (path: string, text: string) =>
    writeFileSync(join(work, path), text);
  const setVersion = (version: string) =>
    write(
      MANIFEST,
      `${JSON.stringify({ name: "arkive-compact", version }, null, 2)}\n`,
    );

  git(root, ["init", "-q", "--bare", "-b", "main", bare]);
  git(root, ["init", "-q", "-b", "main", work]);
  setVersion("1.0.0");
  write("bun.lock", '{ "lock": "1.0.0" }\n');
  write("README.md", "readme\n");
  git(work, ["add", "-A"]);
  git(work, ["commit", "-q", "-m", "initial"]);
  git(work, ["remote", "add", "origin", bare]);
  git(work, ["push", "-q", "origin", "main"]);
  writeFileSync(callLog, "");

  const run = (...args: string[]) => git(work, args);
  const rawStatus = () =>
    exec(work, REAL_GIT, [
      "--no-optional-locks",
      "status",
      "--porcelain=v1",
      "--untracked-files=all",
    ]);
  const origin = (...args: string[]) => git(bare, args);
  const installHook = (dir: string, name: string, script: string) => {
    writeFileSync(join(dir, name), script);
    chmodSync(join(dir, name), 0o755);
  };
  return {
    root,
    work,
    bare,
    run,
    origin,
    write,
    setVersion,
    originHook: (name, script) =>
      installHook(join(bare, "hooks"), name, script),
    hook: (name, script) => installHook(join(work, ".git/hooks"), name, script),
    relativeOrigin: () => {
      const path = join(work, "upstream");
      git(root, ["clone", "-q", "--bare", bare, path]);
      writeFileSync(join(work, ".git/info/exclude"), "/upstream/\n");
      return path;
    },
    wrappedGit: extra => ({
      PATH: `${gitBin}:${env.PATH}`,
      REAL_GIT,
      FAKE_GIT_COUNTS: gitCounts,
      ...extra,
    }),
    cloneOrigin: name => {
      const path = join(root, `${name}.git`);
      git(root, ["clone", "-q", "--bare", bare, path]);
      return path;
    },
    release: (extra = {}, version = "1.1.0") => {
      const result = exec(work, "sh", [RELEASE_SCRIPT, version], extra);
      return {
        status: result.status,
        stderr: result.stderr,
        calls: readFileSync(callLog, "utf8"),
      };
    },
    status: () => {
      const result = rawStatus();
      if (result.status !== 0) throw new Error(result.stderr);
      return result.stdout;
    },
    state: () => {
      const index = createHash("sha256")
        .update(readFileSync(join(work, ".git/index")))
        .digest("hex");
      // A case may break `git status` itself, so record its result raw.
      const status = rawStatus();
      return [
        index,
        run("for-each-ref", "--format=%(refname) %(objectname)"),
        run("rev-parse", "HEAD"),
        run("rev-parse", "--symbolic-full-name", "HEAD"),
        `${status.status} ${status.stdout}`,
        run("ls-files", "-s"),
        // `git diff` would refresh the index; diff-files compares without it.
        run("diff-files", "-p", "--ignore-submodules"),
        run("config", "--list", "--local"),
        refs(bare),
      ].join("\n--\n");
    },
  };
}

function committedVersion(fixture: Fixture, rev: string) {
  return JSON.parse(fixture.run("show", `${rev}:${MANIFEST}`))
    .version as string;
}

function releasedRefs(commit: string) {
  return `refs/heads/main ${commit}\nrefs/tags/v1.1.0 ${commit}`;
}

const refusals: [
  string,
  (fixture: Fixture) => Record<string, string> | void,
  string,
][] = [
  [
    "another branch",
    f => {
      f.run("switch", "-q", "-c", "feature");
      f.write("feature.txt", "x\n");
      f.run("add", "feature.txt");
      f.run("commit", "-q", "-m", "feature");
    },
    "Release must run on branch main, not feature.",
  ],
  [
    "a detached HEAD",
    f => void f.run("switch", "-q", "--detach"),
    "HEAD is detached",
  ],
  [
    "a staged unrelated file",
    f => {
      f.write("unrelated.txt", "x\n");
      f.run("add", "unrelated.txt");
    },
    "clean index and working tree",
  ],
  [
    "an untracked file",
    f => f.write("scratch.txt", "x\n"),
    "clean index and working tree",
  ],
  [
    "an uncommitted manifest bump from an interrupted run",
    f => f.setVersion("1.1.0"),
    "clean index and working tree",
  ],
  [
    "an existing local tag on another commit",
    f => {
      f.write("README.md", "later\n");
      f.run("commit", "-q", "-am", "later");
      f.run("tag", "v1.1.0", "HEAD~1");
    },
    'Tag v1.1.0 already exists locally. If origin does not have it, an earlier release stopped after tagging: follow "Interrupted release helper" in docs/Development.md.',
  ],
  [
    "an existing tag on origin",
    f => {
      f.run("tag", "v1.1.0");
      f.run("push", "-q", "origin", "v1.1.0");
      f.run("tag", "-d", "v1.1.0");
    },
    "Tag v1.1.0 already exists on origin.",
  ],
  [
    "origin main ahead of local main",
    f => {
      f.run("switch", "-q", "-c", "elsewhere");
      f.write("remote.txt", "x\n");
      f.run("add", "remote.txt");
      f.run("commit", "-q", "-m", "remote");
      f.run("push", "-q", "origin", "elsewhere:main");
      f.run("switch", "-q", "main");
      f.run("branch", "-q", "-D", "elsewhere");
    },
    "is not contained in local main",
  ],
  // A read that fails must stop the helper; its empty output is not data.
  [
    "an index that cannot be read, beside an untracked file",
    f => {
      f.write("scratch.txt", "x\n");
      f.write("corrupt.index", "not an index\n");
      return { GIT_INDEX_FILE: join(f.root, "work", "corrupt.index") };
    },
    "Could not read the state of the index and working tree.",
  ],
  [
    "a submodule whose repository is missing, beside an untracked file",
    f => {
      const commit = f.run("rev-parse", "HEAD");
      f.run("update-index", "--add", "--cacheinfo", `160000,${commit},sub`);
      f.write(
        ".gitmodules",
        '[submodule "sub"]\n\tpath = sub\n\turl = ./nowhere\n',
      );
      f.run("add", ".gitmodules");
      f.run("commit", "-q", "-m", "submodule");
      f.run("push", "-q", "origin", "main");
      mkdirSync(join(f.root, "work", "sub"));
      f.write("sub/.git", "gitdir: /nonexistent/modules/sub\n");
      f.write("scratch.txt", "x\n");
    },
    "Could not read the state of the index and working tree.",
  ],
  ...(
    [
      ["symbolic-ref:1", "Could not read the current branch."],
      ["rev-parse:1", "Could not read local main."],
      ["status:1", "Could not read the state of the index and working tree."],
      ["for-each-ref:1", "Could not read local tags."],
      ["remote:1", "Could not read the push URL of origin."],
      ["remote:2", "Could not check how Git reads origin's push URL."],
      ["remote:3", "Could not resolve the push URL of origin."],
      ["ls-remote:1", "Could not read refs from origin's push URL."],
      ["merge-base:1", "Could not check that origin's main"],
    ] as const
  ).map(
    ([call, message]) =>
      [
        `a failed git ${call}`,
        (f: Fixture) => f.wrappedGit({ FAKE_GIT_FAIL: call }),
        message,
      ] as const,
  ),
  [
    "a failed read of the committed manifest, when HEAD already has the version",
    f => {
      f.setVersion("1.1.0");
      f.run("commit", "-q", "-am", "Release v1.1.0");
      return f.wrappedGit({ FAKE_GIT_FAIL: "show:1" });
    },
    `Could not read the version of ${MANIFEST} at HEAD.`,
  ],
  [
    "a committed manifest that is not JSON",
    f => {
      f.write(MANIFEST, '{ "version": "secret-content\n');
      f.run("commit", "-q", "-am", "broken manifest");
      f.run("push", "-q", "origin", "main");
    },
    `Could not read the version of ${MANIFEST} at HEAD.`,
  ],
  [
    "more than one push URL",
    f => {
      for (const name of ["push-a", "push-b"])
        f.run("config", "--add", "remote.origin.pushurl", f.cloneOrigin(name));
    },
    "Origin has more than one push URL.",
  ],
  [
    "a push URL that Git would rewrite again",
    f => void f.run("config", `url.${f.bare}-mirror.pushInsteadOf`, f.bare),
    "Git rewrites origin's push URL when the URL is used directly",
  ],
  // The helper writes the relative push URL as ./upstream, and a remote may
  // carry even that name.
  [
    "a relative push URL that Git still reads as a remote name",
    f => {
      f.relativeOrigin();
      f.run("config", "remote.origin.pushurl", "upstream");
      f.run("config", "remote../upstream.url", f.cloneOrigin("push"));
    },
    "Git reads origin's push URL as the name of a remote",
  ],
  [
    "the tag on the push URL when the fetch URL lacks it",
    f => {
      const push = f.cloneOrigin("push");
      f.run("tag", "v1.1.0");
      f.run("push", "-q", push, "v1.1.0");
      f.run("tag", "-d", "v1.1.0");
      f.run("config", "remote.origin.pushurl", push);
    },
    "Tag v1.1.0 already exists on origin.",
  ],
];

test.each([
  ["no version", "", "Usage: bun run release-opencode -- <version>"],
  ["a v prefix", "v1.1.0", "Version must be in X.Y.Z format."],
  ["two parts", "1.1", "Version must be in X.Y.Z format."],
  ["a prerelease", "1.1.0-beta.1", "Version must be in X.Y.Z format."],
  ["a trailing line break", "1.1.0\n", "Version must be in X.Y.Z format."],
])("refuses %s before reading the repository", (_name, version, message) => {
  const fixture = createFixture();
  const before = fixture.state();

  const result = fixture.release({}, version);

  expect(result.status).toBe(1);
  expect(result.stderr).toBe(`${message}\n`);
  expect(result.calls).toBe("");
  expect(fixture.state()).toBe(before);
});

test.each(refusals)(
  "refuses %s before writing anything",
  (_name, arrange, message) => {
    const fixture = createFixture();
    const env = arrange(fixture) ?? {};
    // A tracked file whose time no longer matches the index makes any index
    // refresh visible in the raw index bytes.
    const stale = new Date("2030-01-01T00:00:00Z");
    utimesSync(join(fixture.work, "README.md"), stale, stale);
    const before = fixture.state();
    const pushRepos = [
      ...["push", "push-a", "push-b"].map(name =>
        join(fixture.root, `${name}.git`),
      ),
      join(fixture.work, "upstream"),
    ];
    const pushBefore = pushRepos.map(path => {
      try {
        return refs(path);
      } catch {
        return null;
      }
    });

    const result = fixture.release(env);

    expect(result.status).toBe(1);
    expect(result.stderr).toContain(message);
    expect(result.stderr).not.toContain("secret-content");
    expect(result.stderr).not.toContain("SyntaxError");
    expect(result.calls).toBe("");
    expect(fixture.state()).toBe(before);
    expect(
      pushRepos.map(path => {
        try {
          return refs(path);
        } catch {
          return null;
        }
      }),
    ).toEqual(pushBefore);
  },
);

test("bumps, commits only the manifest and lockfile, and tags that commit", () => {
  const fixture = createFixture();
  const base = fixture.run("rev-parse", "HEAD");

  const result = fixture.release();

  expect(result.status).toBe(0);
  expect(result.calls).toBe(
    "npm version 1.1.0 --no-git-tag-version --workspaces=false --prefix packages/opencode-plugin\nbun install\n",
  );
  const head = fixture.run("rev-parse", "HEAD");
  expect(fixture.run("rev-parse", "HEAD~1")).toBe(base);
  expect(fixture.run("log", "-1", "--format=%s")).toBe("Release v1.1.0");
  expect(
    fixture.run("show", "--name-only", "--format=", "HEAD").split("\n"),
  ).toEqual(["bun.lock", MANIFEST]);
  expect(committedVersion(fixture, "v1.1.0")).toBe("1.1.0");
  expect(fixture.run("rev-parse", "v1.1.0^{commit}")).toBe(head);
  expect(refs(fixture.bare)).toBe(releasedRefs(head));
});

test("tags an already committed release version without bumping", () => {
  const fixture = createFixture();
  fixture.setVersion("1.1.0");
  fixture.run("commit", "-q", "-am", "Release v1.1.0");
  const head = fixture.run("rev-parse", "HEAD");

  const result = fixture.release();

  expect(result.status).toBe(0);
  expect(result.calls).toBe("");
  expect(fixture.run("rev-parse", "HEAD")).toBe(head);
  expect(refs(fixture.bare)).toBe(releasedRefs(head));
});

// Push mappings and push.followTags in the repository's configuration must
// not change which refs reach origin.
test("pushes exactly main and the release tag whatever the push configuration", () => {
  const fixture = createFixture();
  fixture.write("feature.txt", "x\n");
  fixture.run("add", "feature.txt");
  fixture.run("commit", "-q", "-m", "unpushed feature");
  fixture.run("tag", "-a", "v0.9.0", "-m", "never pushed");
  fixture.run(
    "config",
    "--add",
    "remote.origin.push",
    "refs/heads/*:refs/heads/staging/*",
  );
  fixture.run(
    "config",
    "--add",
    "remote.origin.push",
    "refs/tags/*:refs/tags/staging/*",
  );
  fixture.run("config", "push.followTags", "true");

  const result = fixture.release();

  expect(result.status).toBe(0);
  expect(refs(fixture.bare)).toBe(
    releasedRefs(fixture.run("rev-parse", "HEAD")),
  );
});

// The helper's last check cannot stop a commit made after it, so the push must
// name the checked commit, not local main.
test("publishes the checked release commit when local main gains a commit just before the push", () => {
  const fixture = createFixture();

  const result = fixture.release(
    fixture.wrappedGit({ FAKE_GIT_COMMIT_BEFORE_PUSH: "1" }),
  );

  expect(result.status).toBe(0);
  const release = fixture.run("rev-parse", "v1.1.0^{commit}");
  expect(committedVersion(fixture, release)).toBe("1.1.0");
  expect(refs(fixture.bare)).toBe(releasedRefs(release));
  expect(
    fixture.origin("ls-tree", "-r", "--name-only", "main").split("\n"),
  ).not.toContain("unreviewed.txt");
  expect(fixture.run("rev-parse", "main~1")).toBe(release);
  expect(
    fixture.run("ls-tree", "-r", "--name-only", "main").split("\n"),
  ).toContain("unreviewed.txt");
});

// Each arrangement returns origin's push repository, then repositories that
// must stay unchanged besides the fetch repository.
const relativeUpstream = (f: Fixture) => {
  const push = f.relativeOrigin();
  f.run("config", "remote.origin.pushurl", "upstream");
  return push;
};

test.each([
  [
    "a push URL beside the fetch URL",
    (f: Fixture) => {
      const push = f.cloneOrigin("push");
      f.run("config", "remote.origin.pushurl", push);
      return [push];
    },
  ],
  [
    "a pushInsteadOf rewrite",
    (f: Fixture) => {
      const push = f.cloneOrigin("push");
      f.run("config", `url.${push}.pushInsteadOf`, f.bare);
      return [push];
    },
  ],
  ["a relative push URL", (f: Fixture) => [relativeUpstream(f)]],
  // Git reads a bare word as the name of a remote when one exists, including
  // a legacy .git/remotes or .git/branches file; origin's push URL is a path.
  [
    "a relative push URL that a configured remote shares",
    (f: Fixture) => {
      const push = relativeUpstream(f);
      const decoy = f.cloneOrigin("decoy");
      const decoyPush = f.cloneOrigin("decoy-push");
      f.run("remote", "add", "upstream", decoy);
      f.run("config", "remote.upstream.pushurl", decoyPush);
      return [push, decoy, decoyPush];
    },
  ],
  [
    "a relative push URL that a legacy .git/remotes file shares",
    (f: Fixture) => {
      const push = relativeUpstream(f);
      const decoy = f.cloneOrigin("decoy");
      mkdirSync(join(f.work, ".git/remotes"));
      writeFileSync(join(f.work, ".git/remotes/upstream"), `URL: ${decoy}\n`);
      return [push, decoy];
    },
  ],
  [
    "a relative push URL that a legacy .git/branches file shares",
    (f: Fixture) => {
      const push = relativeUpstream(f);
      const decoy = f.cloneOrigin("decoy");
      mkdirSync(join(f.work, ".git/branches"), { recursive: true });
      writeFileSync(join(f.work, ".git/branches/upstream"), `${decoy}\n`);
      return [push, decoy];
    },
  ],
  // Git rewrites origin's push URL "u" to the path "upstream" once, and that
  // path is where `git push origin` writes, although "upstream" given as a
  // URL would be rewritten again.
  [
    "a push URL that Git rewrites to a relative path",
    (f: Fixture) => {
      const push = f.relativeOrigin();
      f.run("config", "remote.origin.pushurl", "u");
      f.run("config", "url.upstream.insteadOf", "u");
      return [push];
    },
  ],
])("releases to origin's one push repository with %s", (_name, arrange) => {
  const fixture = createFixture();
  const [push, ...others] = arrange(fixture);
  const unchanged = [fixture.bare, ...others];
  const unchangedBefore = unchanged.map(refs);

  const result = fixture.release();

  expect(result.status).toBe(0);
  expect(refs(push)).toBe(releasedRefs(fixture.run("rev-parse", "HEAD")));
  expect(unchanged.map(refs)).toEqual(unchangedBefore);
});

test("pushes to the push URL it checked when origin's push URL changes during the bump", () => {
  const fixture = createFixture();
  const checked = fixture.cloneOrigin("push");
  const other = fixture.cloneOrigin("other");
  fixture.run("config", "remote.origin.pushurl", checked);
  const otherBefore = refs(other);

  const result = fixture.release({
    FAKE_BUN_SCRIPT: 'git config remote.origin.pushurl "$OTHER"',
    OTHER: other,
  });

  expect(result.status).toBe(0);
  expect(fixture.run("config", "--get-all", "remote.origin.pushurl")).toBe(
    other,
  );
  expect(refs(checked)).toBe(releasedRefs(fixture.run("rev-parse", "HEAD")));
  expect(refs(other)).toBe(otherBefore);
});

// A hook that the release itself runs (a lifecycle script under bun install,
// or the repository's post-commit or reference-transaction hook) may change
// how Git reads the checked push URL, or move HEAD, main or the tag. The
// release checks again after each stage and stops, leaving what it created.
const addUnreviewedSideBranch = (f: Fixture) => {
  f.run("switch", "-q", "-c", "side");
  f.write("unreviewed.txt", "x\n");
  f.run("add", "unreviewed.txt");
  f.run("commit", "-q", "-m", "unreviewed side commit");
  f.run("switch", "-q", "main");
};

test.each([
  [
    "bump",
    "adds a pushInsteadOf rewrite of the push URL",
    'git config "url.$REDIRECT.pushInsteadOf" "$CHECKED"',
    "Git rewrites origin's push URL",
  ],
  [
    "bump",
    "adds an insteadOf rewrite of the push URL",
    'git config "url.$REDIRECT.insteadOf" "$CHECKED"',
    "Git rewrites origin's push URL",
  ],
  [
    "bump",
    "switches to a branch with an unreviewed commit",
    "git switch -q side",
    "HEAD left branch main, or main moved",
  ],
  [
    "commit",
    "adds a pushInsteadOf rewrite of the push URL",
    'git config "url.$REDIRECT.pushInsteadOf" "$CHECKED"',
    "Git rewrites origin's push URL",
  ],
  [
    "commit",
    "commits again on main",
    "git commit -q --allow-empty -m unreviewed",
    "HEAD left branch main, or main moved",
  ],
  [
    "tag",
    "adds a pushInsteadOf rewrite of the push URL",
    'git config "url.$REDIRECT.pushInsteadOf" "$CHECKED"',
    "Git rewrites origin's push URL",
  ],
  [
    "tag",
    "switches to a branch with an unreviewed commit",
    "git switch -q side",
    "HEAD left branch main, or main moved",
  ],
  [
    "tag",
    "moves the tag to the previous commit",
    "git tag -f v1.1.0 HEAD~1",
    "Tag v1.1.0 no longer names the release commit.",
  ],
] as const)(
  "stops when a hook run at the %s %s, and pushes nothing",
  (stage, _name, action, message) => {
    const fixture = createFixture();
    addUnreviewedSideBranch(fixture);
    const base = fixture.run("rev-parse", "main");
    const redirect = fixture.cloneOrigin("redirect");
    const repos = [fixture.bare, redirect];
    const reposBefore = repos.map(refs);
    // Each hook acts once; the inner Git commands it runs fire hooks again.
    const script = `[ -z "\${IN_HOOK:-}" ] || exit 0
export IN_HOOK=1
${action}`;
    const env: Record<string, string> = {
      REDIRECT: redirect,
      CHECKED: fixture.bare,
    };
    if (stage === "bump") env.FAKE_BUN_SCRIPT = script;
    if (stage === "commit")
      fixture.hook("post-commit", `#!/bin/sh\n${script}\n`);
    if (stage === "tag")
      fixture.hook(
        "reference-transaction",
        `#!/bin/sh
input=$(cat)
[ "$1" = committed ] || exit 0
case $input in *" refs/tags/v1.1.0"*) ;; *) exit 0 ;; esac
${script}
`,
      );

    const result = fixture.release(env);

    expect(result.status).toBe(1);
    expect(result.stderr).toContain(message);
    expect(repos.map(refs)).toEqual(reposBefore);
    const subjects = fixture.run("log", "--format=%s", "refs/heads/main");
    if (stage === "bump") {
      expect(result.stderr).toContain("No release commit or tag was created");
      expect(fixture.run("rev-parse", "refs/heads/main")).toBe(base);
      expect(
        JSON.parse(readFileSync(join(fixture.work, MANIFEST), "utf8")).version,
      ).toBe("1.1.0");
      expect(fixture.run("tag", "--list")).toBe("");
    } else if (stage === "commit") {
      expect(result.stderr).toContain("no tag was created");
      expect(subjects).toContain("Release v1.1.0");
      expect(fixture.run("tag", "--list")).toBe("");
    } else {
      expect(result.stderr).toContain(
        'Tag v1.1.0 exists locally and nothing was pushed; follow "Interrupted release helper"',
      );
      expect(subjects.split("\n")[0]).toBe("Release v1.1.0");
      expect(fixture.run("tag", "--list")).toBe("v1.1.0");
    }
  },
);

test.each([
  ["changed", "printf 'changed\\n' >> README.md", " M README.md"],
  [
    "staged",
    "printf 'staged\\n' >> README.md; git add README.md",
    "M  README.md",
  ],
])(
  "creates no commit when the bump %s another file, and leaves that change",
  (_name, script, readmeStatus) => {
    const fixture = createFixture();
    const base = fixture.run("rev-parse", "HEAD");
    const originBefore = refs(fixture.bare);

    const result = fixture.release({ FAKE_BUN_SCRIPT: script });

    expect(result.status).toBe(1);
    expect(result.stderr).toContain(
      `The version bump changed files other than ${MANIFEST} and bun.lock. No release commit or tag was created`,
    );
    expect(fixture.run("rev-parse", "HEAD")).toBe(base);
    expect(fixture.status()).toBe(
      `${readmeStatus}\n M bun.lock\n M ${MANIFEST}\n`,
    );
    expect(fixture.run("tag", "--list")).toBe("");
    expect(refs(fixture.bare)).toBe(originBefore);
  },
);

test("commits only the manifest and lockfile when another file is staged just before the commit", () => {
  const fixture = createFixture();
  const base = fixture.run("rev-parse", "HEAD");
  const originBefore = refs(fixture.bare);

  const result = fixture.release(
    fixture.wrappedGit({ FAKE_GIT_STAGE_BEFORE_COMMIT: "1" }),
  );

  expect(result.status).toBe(1);
  expect(result.stderr).toContain("does not match HEAD. No tag was created");
  expect(fixture.run("rev-parse", "HEAD~1")).toBe(base);
  expect(
    fixture.run("show", "--name-only", "--format=", "HEAD").split("\n"),
  ).toEqual(["bun.lock", MANIFEST]);
  expect(fixture.status()).toBe("M  README.md\n");
  expect(fixture.run("tag", "--list")).toBe("");
  expect(refs(fixture.bare)).toBe(originBefore);
});

// `bun install` runs the root package's real `prepare` script, which points
// core.hooksPath at .githooks; a hooks path given in the environment still
// overrides it. Its post-commit hook amends the release commit once, keeping
// the parent, the version and a clean working tree, so only the commit's tree
// shows the change. The manifest case stays within the two committed paths.
const ROOT_PREPARE = JSON.parse(
  readFileSync(join(import.meta.dir, "../../../package.json"), "utf8"),
).scripts.prepare as string;

test.each([
  ["README.md", "printf 'amended\\n' >> README.md"],
  [
    MANIFEST,
    `printf '{\\n  "name": "arkive-compact",\\n  "version": "1.1.0",\\n  "extra": "unreviewed"\\n}\\n' > ${MANIFEST}`,
  ],
])(
  "creates no tag when a post-commit hook amends %s into the release commit",
  (path, edit) => {
    const fixture = createFixture();
    const base = fixture.run("rev-parse", "HEAD");
    const originBefore = refs(fixture.bare);
    const hooks = join(fixture.root, "external-hooks");
    // The space checks that the marker path reaches the hook as one word;
    // a split path would skip the one-shot guard and amend without end.
    const marker = join(fixture.root, "amended marker");
    mkdirSync(hooks);
    writeFileSync(
      join(hooks, "post-commit"),
      `#!/bin/sh
set -eu
[ -e "$AMEND_MARKER" ] && exit 0
: > "$AMEND_MARKER"
${edit}
git add -- ${path}
git commit -q --amend --no-edit
`,
    );
    chmodSync(join(hooks, "post-commit"), 0o755);

    const result = fixture.release({
      FAKE_BUN_SCRIPT: ROOT_PREPARE,
      AMEND_MARKER: marker,
      GIT_CONFIG_COUNT: "1",
      GIT_CONFIG_KEY_0: "core.hooksPath",
      GIT_CONFIG_VALUE_0: hooks,
    });

    expect(fixture.run("config", "--local", "core.hooksPath")).toBe(
      ".githooks",
    );
    expect(result.status).toBe(1);
    expect(result.stderr).toContain(
      "The release commit's content differs from the checked version bump. The release commit exists and no tag was created",
    );
    expect(fixture.run("tag", "--list")).toBe("");
    expect(refs(fixture.bare)).toBe(originBefore);
    expect(fixture.run("rev-parse", "main~1")).toBe(base);
    expect(committedVersion(fixture, "main")).toBe("1.1.0");
    expect(fixture.run("show", "main:README.md")).toBe(
      path === "README.md" ? "readme\namended" : "readme",
    );
    expect(JSON.parse(fixture.run("show", `main:${MANIFEST}`)).extra).toBe(
      path === MANIFEST ? "unreviewed" : undefined,
    );
    expect(fixture.status()).toBe("");
  },
);

test.each([
  [
    "the committed manifest has another version",
    { FAKE_NPM_VERSION: "1.1.1" },
    "has version 1.1.1, not 1.1.0",
  ],
  [
    "the state after the bump cannot be read",
    { FAKE_GIT_FAIL: "status:2" },
    "Could not read the state of the index and working tree after the version bump.",
  ],
  [
    "the committed manifest cannot be read after the commit",
    { FAKE_GIT_FAIL: "show:2" },
    `Could not read the version of ${MANIFEST} in the release commit. No tag was created.`,
  ],
  [
    "the state after staging the bump cannot be read",
    { FAKE_GIT_FAIL: "status:3" },
    "Could not read the state of the index and working tree after staging the version bump. No release commit or tag was created",
  ],
  [
    "the state after the commit cannot be read",
    { FAKE_GIT_FAIL: "status:4" },
    "Could not read the state of the index and working tree. No tag was created.",
  ],
])("creates no tag when %s", (_name, env, message) => {
  const fixture = createFixture();
  const originBefore = refs(fixture.bare);

  const result = fixture.release(fixture.wrappedGit(env));

  expect(result.status).toBe(1);
  expect(result.stderr).toContain(message);
  expect(fixture.run("tag", "--list")).toBe("");
  expect(refs(fixture.bare)).toBe(originBefore);
});

// Origin refuses the tag alone; the atomic push must then leave its main
// where it was.
test("leaves origin main unchanged when origin refuses the release tag", () => {
  const fixture = createFixture();
  fixture.originHook(
    "update",
    `#!/bin/sh
case "$1" in refs/tags/*) echo "tags are refused" >&2; exit 1 ;; esac
`,
  );
  const originBefore = refs(fixture.bare);

  const result = fixture.release();

  expect(result.status).not.toBe(0);
  expect(result.stderr).toContain("tags are refused");
  expect(refs(fixture.bare)).toBe(originBefore);
});

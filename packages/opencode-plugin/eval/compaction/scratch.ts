// The run's private scratch tree: one root under the system temp directory
// and the project, judge and inspect locations inside it. Every directory is
// created with a marker naming the run, its output directory and the root, and
// nothing is disposed or deleted until the root and the location both prove
// that ownership again. The ledger only says where to look; it is never proof.
import { execFileSync } from "node:child_process";
import {
  lstatSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  realpathSync,
  rmdirSync,
  rmSync,
  unlinkSync,
  writeFileSync,
} from "node:fs";
import { homedir, tmpdir } from "node:os";
import { basename, dirname, isAbsolute, join, sep } from "node:path";
import { isDeepStrictEqual } from "node:util";

export const MARKER = ".arkive-compaction-eval";
const ROOT_NAME = /^arkive-compaction-eval-(?:judge-)?[A-Za-z0-9]{6}$/u;
export const LOCATION_ROLES = ["inspect", "project", "judge"] as const;
export type LocationRole = (typeof LOCATION_ROLES)[number];

export type Owner = { runNonce: string; out: string };

function writeMarker(dir: string, value: Record<string, unknown>) {
  writeFileSync(join(dir, MARKER), `${JSON.stringify(value)}\n`, {
    flag: "wx",
    mode: 0o600,
  });
}

// A fresh root for one run (or one later grading pass) of `owner.out`.
export function createRoot(owner: Owner, kind: "run" | "judge"): string {
  const root = realpathSync(
    mkdtempSync(
      join(
        realpathSync(tmpdir()),
        kind === "run"
          ? "arkive-compaction-eval-"
          : "arkive-compaction-eval-judge-",
      ),
    ),
  );
  writeMarker(root, { ...owner, root, role: "root" });
  return root;
}

// A location inside a root: its marker, its own git root and the given
// config.
export function createLocation(
  root: string,
  role: LocationRole,
  owner: Owner,
  config?: { file: string; content: unknown },
): string {
  const directory = join(root, role);
  mkdirSync(directory, { mode: 0o700 });
  writeMarker(directory, { ...owner, root, role });
  if (config)
    writeFileSync(
      join(directory, config.file),
      `${JSON.stringify(config.content, null, 2)}\n`,
    );
  const git = (...args: string[]) =>
    execFileSync(
      "git",
      [
        "-c",
        "user.name=arkive-eval",
        "-c",
        "user.email=arkive-eval@invalid",
        ...args,
      ],
      { cwd: directory, stdio: "ignore" },
    );
  git("init", "-q");
  git("add", "-A");
  git("commit", "-q", "-m", "arkive compaction eval location");
  return directory;
}

// A real directory at its canonical path (no symlink anywhere on the way)
// holding exactly the expected marker.
function markedProblem(dir: string, marker: unknown): string | null {
  if (!isAbsolute(dir)) return `${dir} is not an absolute path`;
  try {
    if (!lstatSync(dir).isDirectory()) return `${dir} is not a directory`;
    if (realpathSync(dir) !== dir) return `${dir} is not a canonical path`;
    const file = join(dir, MARKER);
    if (!lstatSync(file).isFile()) return `${dir} has no marker file`;
    if (!isDeepStrictEqual(JSON.parse(readFileSync(file, "utf8")), marker))
      return `${dir} belongs to another run`;
    return null;
  } catch {
    return `${dir} has no readable marker`;
  }
}

const within = (parent: string, path: string) =>
  path === parent || path.startsWith(`${parent}${sep}`);

export function rootProblem(root: string, owner: Owner): string | null {
  if (!ROOT_NAME.test(basename(root)))
    return `${root} is not an eval scratch root`;
  for (const kept of [owner.out, homedir(), process.cwd()])
    if (within(root, kept)) return `${root} contains ${kept}`;
  return markedProblem(root, { ...owner, root, role: "root" });
}

export function locationProblem(
  root: string,
  location: string,
  owner: Owner,
): string | null {
  const role = basename(location);
  if (
    dirname(location) !== root
    || !(LOCATION_ROLES as readonly string[]).includes(role)
  )
    return `${location} is not a location of ${root}`;
  return markedProblem(location, { ...owner, root, role });
}

// Deletes validated locations and then the root itself, which must then be
// empty apart from its marker. Anything else in the root keeps it.
export function removeOwnedTree(
  root: string,
  locations: string[],
  owner: Owner,
): string | null {
  const problem =
    rootProblem(root, owner)
    ?? locations.map(l => locationProblem(root, l, owner)).find(Boolean)
    ?? null;
  if (problem) return problem;
  for (const location of locations)
    rmSync(location, { recursive: true, force: true });
  unlinkSync(join(root, MARKER));
  try {
    rmdirSync(root);
  } catch {
    writeMarker(root, { ...owner, root, role: "root" });
    return `${root} holds files the run did not create`;
  }
  return null;
}

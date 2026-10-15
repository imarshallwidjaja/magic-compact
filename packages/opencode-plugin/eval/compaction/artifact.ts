// The packed subject and its eval wrappers, kept in a content-addressed cache.
// An entry's key binds the subject tarball and the wrapper template, never a
// run, so repeated runs of the same artifact load the same installed modules.
// Run identity reaches the wrapper only through plugin options.
import { execFileSync } from "node:child_process";
import { randomBytes } from "node:crypto";
import {
  copyFileSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  renameSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { homedir, tmpdir } from "node:os";
import { join, relative } from "node:path";
import { sha256 } from "./corpus";

export const WRAPPER_PROTOCOL = "arkive-compaction-eval-wrapper/1";
const WRAPPER_DIR = join(import.meta.dir, "wrapper");
const WRAPPER_FILES = ["core.ts", "v1.ts", "v2.ts"] as const;
export const PACKAGE_DIR = join(import.meta.dir, "..", "..");

export type Artifact = {
  key: string;
  dir: string;
  reused: boolean;
  subjectTgzSha256: string;
  subjectManifestSha256: string;
  subjectFiles: number;
  subjectName: string;
  subjectVersion: string;
  wrapperTemplateSha256: string;
  v1Tgz: string;
  v1TgzSha256: string;
  v2Dir: string;
  // SHA-256 over every file of the entry, node_modules included.
  entryTreeSha256: string;
  entryFiles: number;
};

export function defaultCacheRoot(
  env: Record<string, string | undefined> = process.env,
): string {
  return join(
    env["XDG_CACHE_HOME"] || join(homedir(), ".cache"),
    "arkive-compact-eval",
  );
}

export function wrapperTemplateSha256(): string {
  return sha256(
    JSON.stringify([
      WRAPPER_PROTOCOL,
      ...WRAPPER_FILES.map(name => [
        name,
        sha256(readFileSync(join(WRAPPER_DIR, name))),
      ]),
    ]),
  );
}

// Packs the working-tree package exactly as npm would publish it.
export function packSubject(packageDir: string, destination: string): string {
  const name = execFileSync(
    "npm",
    ["pack", "--ignore-scripts", "--silent", "--pack-destination", destination],
    { cwd: packageDir, encoding: "utf8" },
  )
    .trim()
    .split("\n")
    .at(-1)!;
  return join(destination, name);
}

// Every file under `root` with its SHA-256, by sorted relative path.
export function treeHashes(root: string): Record<string, string> {
  const out: Record<string, string> = {};
  const walk = (dir: string) => {
    for (const name of readdirSync(dir).sort()) {
      const path = join(dir, name);
      const stat = statSync(path, { throwIfNoEntry: false });
      if (!stat) continue;
      if (stat.isDirectory()) walk(path);
      else out[relative(root, path)] = sha256(readFileSync(path));
    }
  };
  walk(root);
  return out;
}

function unpackManifest(tgz: string): {
  manifest: Record<string, string>;
  name: string;
  version: string;
} {
  const dir = mkdtempSync(join(tmpdir(), "arkive-compaction-eval-unpack-"));
  try {
    execFileSync("tar", ["-xzf", tgz, "-C", dir]);
    const root = join(dir, "package");
    const pkg = JSON.parse(readFileSync(join(root, "package.json"), "utf8"));
    return { manifest: treeHashes(root), name: pkg.name, version: pkg.version };
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

export class ArtifactCacheError extends Error {}

// Returns the verified cache entry for this subject and wrapper template,
// building it in a private temporary directory and renaming it into place
// when absent. An existing entry whose bytes differ from its own record is
// never reused or overwritten.
export function prepareArtifact(
  subjectTgz: string,
  cacheRoot: string,
  npmEnv: Record<string, string | undefined> = process.env,
): Artifact {
  const subjectTgzSha256 = sha256(readFileSync(subjectTgz));
  const subject = unpackManifest(subjectTgz);
  const subjectManifestSha256 = sha256(JSON.stringify(subject.manifest));
  const templateSha256 = wrapperTemplateSha256();
  const key = sha256(
    JSON.stringify({
      protocol: WRAPPER_PROTOCOL,
      subjectTgzSha256,
      templateSha256,
    }),
  );
  const dir = join(cacheRoot, key);
  mkdirSync(cacheRoot, { recursive: true });
  let reused = existsSync(dir);
  if (!reused) {
    const staging = join(
      cacheRoot,
      `.staging-${randomBytes(8).toString("hex")}`,
    );
    try {
      buildEntry(
        staging,
        dir,
        key,
        subjectTgz,
        {
          subjectTgzSha256,
          subjectManifestSha256,
          templateSha256,
          manifest: subject.manifest,
        },
        npmEnv,
      );
      try {
        renameSync(staging, dir);
      } catch (error) {
        // Another run published the same key first; verify that one.
        if (!existsSync(dir)) throw error;
        reused = true;
      }
    } finally {
      rmSync(staging, { recursive: true, force: true });
    }
  }
  return verifyEntry(dir, key, reused, {
    subjectTgzSha256,
    subjectManifestSha256,
    templateSha256,
    subject,
  });
}

function buildEntry(
  staging: string,
  finalDir: string,
  key: string,
  subjectTgz: string,
  hashes: {
    subjectTgzSha256: string;
    subjectManifestSha256: string;
    templateSha256: string;
    manifest: Record<string, string>;
  },
  npmEnv: Record<string, string | undefined>,
) {
  mkdirSync(staging);
  copyFileSync(subjectTgz, join(staging, "subject.tgz"));
  const witness = `${JSON.stringify(
    {
      protocol: WRAPPER_PROTOCOL,
      artifactKey: key,
      subjectTgzSha256: hashes.subjectTgzSha256,
      subjectManifestSha256: hashes.subjectManifestSha256,
      subjectManifest: hashes.manifest,
    },
    null,
    2,
  )}\n`;
  const version = `0.0.0-eval.${key.slice(0, 16)}`;
  const writePackage = (
    target: string,
    name: string,
    entry: string,
    subjectSpec: string,
  ) => {
    mkdirSync(target, { recursive: true });
    copyFileSync(join(WRAPPER_DIR, "core.ts"), join(target, "core.ts"));
    copyFileSync(join(WRAPPER_DIR, entry), join(target, "index.ts"));
    writeFileSync(join(target, "witness.json"), witness);
    writeFileSync(
      join(target, "package.json"),
      `${JSON.stringify(
        {
          name,
          version,
          private: true,
          type: "module",
          main: "index.ts",
          exports: { ".": "./index.ts", "./server": "./index.ts" },
          dependencies: { "arkive-compact-eval-subject": subjectSpec },
        },
        null,
        2,
      )}\n`,
    );
  };
  const env = { ...npmEnv, npm_config_update_notifier: "false" };
  // OpenCode 1 installs this tarball itself; its subject dependency names the
  // entry's final path.
  const stage = join(staging, "v1-package");
  writePackage(
    stage,
    "arkive-compact",
    "v1.ts",
    `file:${join(finalDir, "subject.tgz")}`,
  );
  mkdirSync(join(staging, "v1"));
  const packed = execFileSync(
    "npm",
    [
      "pack",
      "--ignore-scripts",
      "--silent",
      "--pack-destination",
      join(staging, "v1"),
    ],
    { cwd: stage, encoding: "utf8", env },
  )
    .trim()
    .split("\n")
    .at(-1)!;
  renameSync(join(staging, "v1", packed), join(staging, "v1", "wrapper.tgz"));
  rmSync(stage, { recursive: true });
  // OpenCode 2 loads a local directory and installs nothing, so its
  // dependencies are installed here.
  const v2 = join(staging, "v2");
  writePackage(v2, "arkive-compact-eval", "v2.ts", "file:../subject.tgz");
  execFileSync(
    "npm",
    ["install", "--ignore-scripts", "--no-audit", "--no-fund", "--omit=dev"],
    { cwd: v2, env, stdio: ["ignore", "ignore", "pipe"] },
  );
  const files = treeHashes(staging);
  writeFileSync(
    join(staging, "entry.json"),
    `${JSON.stringify({ protocol: WRAPPER_PROTOCOL, key, files }, null, 2)}\n`,
  );
}

function verifyEntry(
  dir: string,
  key: string,
  reused: boolean,
  expected: {
    subjectTgzSha256: string;
    subjectManifestSha256: string;
    templateSha256: string;
    subject: {
      manifest: Record<string, string>;
      name: string;
      version: string;
    };
  },
): Artifact {
  const fail = (why: string): never => {
    throw new ArtifactCacheError(
      `Artifact cache entry ${dir} failed verification (${why}). It was not used or changed; remove it to rebuild.`,
    );
  };
  let record: {
    protocol?: unknown;
    key?: unknown;
    files?: Record<string, string>;
  };
  try {
    record = JSON.parse(readFileSync(join(dir, "entry.json"), "utf8"));
  } catch {
    return fail("missing entry record");
  }
  if (
    record.protocol !== WRAPPER_PROTOCOL
    || record.key !== key
    || !record.files
  )
    return fail("entry record does not match");
  const actual = treeHashes(dir);
  delete actual["entry.json"];
  if (JSON.stringify(actual) !== JSON.stringify(record.files))
    fail("files differ from the entry record");
  if (actual["subject.tgz"] !== expected.subjectTgzSha256)
    fail("subject tarball differs");
  const installed = join("v2", "node_modules", "arkive-compact-eval-subject");
  for (const [rel, hash] of Object.entries(expected.subject.manifest)) {
    if (actual[join(installed, rel)] !== hash)
      fail(`installed subject file ${rel} differs`);
  }
  return {
    key,
    dir,
    reused,
    subjectTgzSha256: expected.subjectTgzSha256,
    subjectManifestSha256: expected.subjectManifestSha256,
    subjectFiles: Object.keys(expected.subject.manifest).length,
    subjectName: expected.subject.name,
    subjectVersion: expected.subject.version,
    wrapperTemplateSha256: expected.templateSha256,
    v1Tgz: join(dir, "v1", "wrapper.tgz"),
    v1TgzSha256: actual[join("v1", "wrapper.tgz")]!,
    v2Dir: join(dir, "v2"),
    entryTreeSha256: sha256(JSON.stringify(actual)),
    entryFiles: Object.keys(actual).length,
  };
}

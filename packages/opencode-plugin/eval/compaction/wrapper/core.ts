// Eval-only instrumentation around an unmodified packed arkive-compact. It is
// not a production API and never proxies credentials. It delegates every
// product hook unchanged and adds, for one eval run named by its options:
// a content-free load witness, the exact outcome text of each Magic Compact
// command, content-free usage numbers, and removal of a successor's cache and
// stats through the subject's own storage functions once the host reports
// that successor deleted. Everything it reports goes to the run's channel
// directory as JSON lines.
import { createHash } from "node:crypto";
import { appendFileSync, readFileSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { dirname, isAbsolute, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

export const WITNESS = "arkive-eval-witness";
export const CONTROL = "arkive-eval-control";
export const COMMANDS = ["magic-compact", "magic-trim", "magic-stats"];
export const SUMMARIZER = "magic-compact-summarizer";

const here = dirname(fileURLToPath(import.meta.url));
let initCount = 0;

export type EvalOptions = {
  runNonce: string;
  channel: string;
  // The summarizer variant the run requested; null for the model's default.
  summarizerVariant: string | null;
};

// The run options this instance was configured with. Without them the wrapper
// refuses to load rather than run the subject unobserved.
export function evalOptions(raw: unknown): EvalOptions {
  const value = (raw ?? {}) as Record<string, unknown>;
  const runNonce = value["evalRunNonce"];
  const channel = value["evalChannel"];
  const variant = value["evalSummarizerVariant"];
  if (
    typeof runNonce !== "string"
    || !/^[0-9a-f]{32}$/u.test(runNonce)
    || typeof channel !== "string"
    || !isAbsolute(channel)
    || !(variant === null || (typeof variant === "string" && variant !== ""))
  ) {
    throw new Error(`${WITNESS}: missing or invalid eval run options`);
  }
  return { runNonce, channel, summarizerVariant: variant };
}

// Sets the summarizer's variant to exactly the run's request in this
// instance's in-memory agent config, after the subject applied its own
// config. OpenCode 1 merges agent config field by field, so a variant from a
// global summarizer profile would otherwise reach a model the run selected
// without one. No config file changes.
export function applySummarizerVariant(
  agents: Record<string, Record<string, unknown>>,
  options: EvalOptions,
): void {
  const summarizer = (agents[SUMMARIZER] ??= {});
  if (options.summarizerVariant === null) delete summarizer["variant"];
  else summarizer["variant"] = options.summarizerVariant;
}

export function record(options: EvalOptions, line: Record<string, unknown>) {
  appendFileSync(
    join(options.channel, `host-${process.pid}.jsonl`),
    `${JSON.stringify({ at: Date.now(), pid: process.pid, runNonce: options.runNonce, ...line })}\n`,
  );
}

function subjectRoot(): string {
  // <root>/src/index.ts
  return dirname(
    dirname(fileURLToPath(import.meta.resolve("arkive-compact-eval-subject"))),
  );
}

// Called after the subject initialized. Checks the installed subject files
// against the artifact manifest and reports the result.
export async function witness(
  options: EvalOptions,
  host: "v1" | "v2",
  pluginId: string,
  directory: string,
): Promise<string> {
  const expected = JSON.parse(readFileSync(join(here, "witness.json"), "utf8"));
  initCount += 1;
  const root = subjectRoot();
  let mismatches = 0;
  for (const [rel, sha] of Object.entries(
    expected.subjectManifest as Record<string, string>,
  )) {
    const actual = await readFile(join(root, rel)).then(
      bytes => createHash("sha256").update(bytes).digest("hex"),
      () => "missing",
    );
    if (actual !== sha) mismatches += 1;
  }
  const installed = JSON.parse(
    await readFile(join(root, "package.json"), "utf8"),
  );
  const line = {
    kind: "init",
    host,
    pluginId,
    directory,
    protocol: expected.protocol,
    artifactKey: expected.artifactKey,
    subjectTgzSha256: expected.subjectTgzSha256,
    subjectManifestSha256: expected.subjectManifestSha256,
    subjectFiles: Object.keys(expected.subjectManifest).length,
    subjectMismatches: mismatches,
    subjectName: installed.name,
    subjectVersion: installed.version,
    initCount,
  };
  record(options, line);
  return `${WITNESS} ${JSON.stringify({ ...line, runNonce: options.runNonce, pid: process.pid })}`;
}

// Wraps one product command handler: the subject runs unchanged and its
// result, which both hosts deliver by throwing, is recorded before it is
// rethrown. OpenCode 1 masks that message over HTTP.
export async function observeCommand(
  options: EvalOptions,
  sessionID: string,
  command: string,
  run: () => Promise<unknown>,
): Promise<unknown> {
  const started = Date.now();
  record(options, { kind: "command-started", sessionID, command });
  try {
    const value = await run();
    record(options, {
      kind: "command-outcome",
      sessionID,
      command,
      threw: false,
      ms: Date.now() - started,
    });
    return value;
  } catch (error) {
    record(options, {
      kind: "command-outcome",
      sessionID,
      command,
      threw: true,
      message: String((error as Error)?.message ?? error).slice(0, 8000),
      ms: Date.now() - started,
    });
    throw error;
  }
}

// The subject's storage for one session, located by the subject itself.
export type StorageOps = {
  key(sessionID: string): string;
  paths(key: string): { cache: string; stats: string };
  deleteCache(key: string): Promise<void>;
  deleteStats(key: string): Promise<void>;
};

// Loads the storage functions of the installed subject, or of another
// package root that has the same layout.
export async function subjectStorage(
  host: "v1" | "v2",
  root: string = subjectRoot(),
): Promise<StorageOps> {
  const load = (rel: string) => import(pathToFileURL(join(root, rel)).href);
  const omission = await load("src/storage/omission.ts");
  const stats = await load("src/storage/stats.ts");
  const store = await load("src/storage/store.ts");
  const session = host === "v2" ? await load("src/v2/session.ts") : null;
  return {
    key: id => (session ? session.v2StorageKey(id) : id),
    paths: key => ({
      cache: omission.cachePath(key),
      stats: `${store.pluginStorageDirectory()}/stats/${key}.json`,
    }),
    deleteCache: key => omission.deleteCache(key),
    deleteStats: key => stats.deleteStats(key),
  };
}

// What the control needs to know about a session, read with the host's own
// client.
export type SessionFacts = {
  directory: string | undefined;
  title: string | undefined;
  metadata: Record<string, unknown> | undefined;
};
export type SessionRead = (id: string) => Promise<SessionFacts | undefined>;

type Receipt = {
  requestId: string;
  key: string;
  paths: { cache: string; stats: string };
};

// Removes a successor's storage in two steps. `prepare` runs as a command
// inside the successor and proves it belongs to this run: the request names
// this run, the successor and its source live in this wrapper's own private
// location, the source's title carries the run nonce, and the successor's
// lineage names that source. It writes nothing but an in-memory receipt.
// The driver then checks the successor's full state itself and deletes the
// session through the public API. Only when the host reports that deletion
// does `deleted` remove the receipted storage, so a session the driver kept
// keeps its storage too.
export function createControl(
  options: EvalOptions,
  directory: string,
  read: SessionRead,
  storage: () => Promise<StorageOps>,
) {
  const receipts = new Map<string, Receipt>();
  return {
    async prepare(sessionID: string, argumentsText: string): Promise<string> {
      let request: Record<string, unknown>;
      try {
        request = JSON.parse(argumentsText);
      } catch {
        request = {};
      }
      const requestId =
        typeof request["requestId"] === "string"
        && /^[0-9a-f]{16,64}$/u.test(request["requestId"])
          ? request["requestId"]
          : null;
      const sourceID = request["sourceID"];
      const refuse = (reason: string) => {
        record(options, {
          kind: "control",
          phase: "refused",
          requestId,
          sessionID,
          reason,
        });
        return `${CONTROL}: refused (${reason})`;
      };
      if (request["runNonce"] !== options.runNonce || !requestId)
        return refuse("the request does not name this run");
      if (typeof sourceID !== "string" || sourceID === sessionID)
        return refuse("the request names no source");
      const [session, source] = await Promise.all([
        read(sessionID).catch(() => undefined),
        read(sourceID).catch(() => undefined),
      ]);
      if (!session || !source) return refuse("a session could not be read");
      if (session.directory !== directory || source.directory !== directory)
        return refuse("a session is outside this run's location");
      if (!source.title?.includes(options.runNonce))
        return refuse("the source does not carry this run's nonce");
      const lineage = session.metadata?.["magicCompact"] as
        { successorOf?: unknown; ready?: unknown } | undefined;
      if (lineage?.successorOf !== sourceID || lineage.ready !== true)
        return refuse("the session is not a ready successor of the source");
      const ops = await storage();
      const key = ops.key(sessionID);
      const receipt = { requestId, key, paths: ops.paths(key) };
      receipts.set(sessionID, receipt);
      record(options, {
        kind: "control",
        phase: "prepared",
        sessionID,
        ...receipt,
      });
      return `${CONTROL}: ${requestId} prepared`;
    },

    // Called for every session deletion the host reports.
    async deleted(sessionID: string): Promise<void> {
      const receipt = receipts.get(sessionID);
      if (!receipt) return;
      receipts.delete(sessionID);
      const ops = await storage();
      const failed: string[] = [];
      await ops.deleteCache(receipt.key).catch(() => failed.push("cache"));
      await ops.deleteStats(receipt.key).catch(() => failed.push("stats"));
      record(options, {
        kind: "control",
        phase: "storage",
        sessionID,
        requestId: receipt.requestId,
        removed: failed.length === 0,
        failed,
        paths: receipt.paths,
      });
    },
  };
}

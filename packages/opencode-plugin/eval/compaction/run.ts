// One eval run against an existing OpenCode server: a private temporary
// project that loads the packed subject through the eval wrapper, fresh
// synthetic source sessions, one /magic-compact attempt each, mechanical
// checks, optional judge grading and ownership-checked cleanup. Everything the
// run creates is recorded in its ledger before and as it happens.
import { execFileSync } from "node:child_process";
import { randomBytes } from "node:crypto";
import {
  closeSync,
  existsSync,
  fsyncSync,
  lstatSync,
  mkdirSync,
  mkdtempSync,
  openSync,
  readdirSync,
  readFileSync,
  realpathSync,
  rmSync,
  writeFileSync,
  writeSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { basename, dirname, join, resolve } from "node:path";
import { isDeepStrictEqual } from "node:util";
import { z } from "zod";
import { COMPACT_COMMAND, COMMAND_DESCRIPTIONS } from "../../src/commands";
import {
  defaultCacheRoot,
  PACKAGE_DIR,
  packSubject,
  prepareArtifact,
  type Artifact,
} from "./artifact";
import {
  checkCompactionPlans,
  loadCorpus,
  type Corpus,
  type CorpusCase,
} from "./corpus";
import {
  JUDGE_SYSTEM,
  GRADES_FORMAT,
  judgePrompt,
  judgmentProblem,
  outputSha256,
  parseJudgeAnswer,
  type GradesFile,
} from "./grade";
import {
  EvalError,
  identifyServer,
  readChannel,
  type ChannelLine,
  type EffectiveSetup,
  type EvalHost,
  type ExpectedState,
  type FetchLike,
  type ModelDescriptor,
  type ModelSelector,
  type ServerAuth,
  type ServerIdentity,
  type SessionLedger,
  type SessionState,
  type Witness,
} from "./host";
import { v1Host } from "./host-v1";
import { v2Host } from "./host-v2";
import { classifyOutcome, type CommandOutcome } from "./inspect";
import {
  createLocation,
  createRoot,
  locationProblem,
  removeOwnedTree,
  rootProblem,
  type LocationRole,
  type Owner,
} from "./scratch";

export const RUN_FORMAT = "arkive-compaction-eval-run/1";
const CONTROL_COMMAND = "arkive-eval-control";
const SUMMARIZER = "magic-compact-summarizer";
const JUDGE_AGENT = "arkive-eval-judge";
const CHANNEL_WAIT_MS = 15_000;

// Bun's fetch may time out a long command request; the eval waits for the
// command's own outcome instead.
export const untimedFetch: FetchLike = (input, init) =>
  fetch(input, { ...init, timeout: false } as RequestInit);

export type LedgerEntry = {
  seq: number;
  at: string;
  kind: string;
  [key: string]: unknown;
};

// Append-only and flushed per entry, so a crash leaves every intent and
// returned ID on disk.
export class Ledger {
  private seq: number;
  constructor(readonly path: string) {
    this.seq = existsSync(path) ? Ledger.read(path).length : 0;
  }
  static read(path: string): LedgerEntry[] {
    return readFileSync(path, "utf8")
      .split("\n")
      .filter(Boolean)
      .map(line => JSON.parse(line) as LedgerEntry);
  }
  append(kind: string, fields: Record<string, unknown> = {}): number {
    const entry = {
      seq: ++this.seq,
      at: new Date().toISOString(),
      kind,
      ...fields,
    };
    const fd = openSync(this.path, "a");
    try {
      writeSync(fd, `${JSON.stringify(entry)}\n`);
      fsyncSync(fd);
    } finally {
      closeSync(fd);
    }
    return entry.seq;
  }
  // Each creation names the intent it fulfils, so a later creation never
  // settles an earlier intent whose request failed.
  sessions(attemptId: string | null): SessionLedger {
    const intents = new Map<string, number>();
    return {
      intent: (role, plannedID, title) =>
        intents.set(
          role,
          this.append("session-intent", { attemptId, role, plannedID, title }),
        ),
      created: (role, id) =>
        this.append("session-created", {
          attemptId,
          role,
          id,
          intentSeq: intents.get(role) ?? null,
        }),
      removed: id => this.append("session-removed", { attemptId, id }),
    };
  }
}

// What the ledger leaves open: sessions created and not removed or found
// absent; session intents with no creation or resolution recorded, whether
// or not their ID was chosen in advance; and attempts whose command started
// without a settled outcome, including after a crash that wrote nothing more.
export function outstanding(entries: LedgerEntry[]) {
  const gone = new Set(
    entries
      .filter(e => e.kind === "session-removed" || e.kind === "session-absent")
      .map(e => String(e["id"])),
  );
  const sessions = entries.filter(
    e => e.kind === "session-created" && !gone.has(String(e["id"])),
  );
  const resolved = new Set(
    entries
      .filter(
        e =>
          (e.kind === "session-created" && e["intentSeq"] != null)
          || e.kind === "session-intent-resolved",
      )
      .map(e => Number(e["intentSeq"])),
  );
  const intents = entries.filter(
    e => e.kind === "session-intent" && !resolved.has(e.seq),
  );
  const settled = new Set(
    entries
      .filter(e => e.kind === "attempt-settled")
      .map(e => String(e["attemptId"])),
  );
  const attempts = entries.filter(
    e => e.kind === "command-start" && !settled.has(String(e["attemptId"])),
  );
  return { sessions, intents, attempts };
}

export type RunOptions = {
  server: string;
  auth: ServerAuth;
  model: ModelSelector;
  out: string;
  repeat: number;
  cases: string[] | null;
  judge: ModelSelector | null;
  dryRun: boolean;
  cacheRoot: string | null;
  subject: string | null;
  assertHostVersion: string | null;
  fetch?: FetchLike;
  // How long to wait for the outcome of a command whose HTTP response was
  // lost before recording it as unknown.
  lostOutcomeWaitMs?: number;
  log?: (line: string) => void;
};

export const ATTEMPT_STATUSES = [
  "setup-failed",
  "preflight-skipped",
  "outcome-unknown",
  "failed",
  "noop",
  "delivered",
  "delivered-invalid",
] as const;

export type CleanupResult = {
  id: string;
  action: "removed" | "retained" | "absent";
  reason?: string;
  // For a removed successor: whether the wrapper removed its storage after
  // the host reported the deletion, and where it was when it did not.
  storage?: "removed" | { orphaned: { cache: string; stats: string } | null };
};

export type AttemptRecord = {
  // The run and requested model that wrote this record; a saved run refuses
  // a record written by any other run.
  runNonce: string;
  model: string;
  attemptId: string;
  caseId: string;
  repeat: number;
  status: (typeof ATTEMPT_STATUSES)[number];
  skipReason: string | null;
  sourceID: string | null;
  successorID: string | null;
  sourceChecks: Array<{ name: string; ok: boolean }>;
  successorChecks: Array<{ name: string; ok: boolean }>;
  sourceUnchanged: boolean | null;
  outcome: CommandOutcome | null;
  http: { status: number | null; lost: boolean } | null;
  commandMs: number | null;
  capsule: string | null;
  outputSha256: string | null;
  storedHistory: string | null;
  usage: UsageTotals | null;
  cleanup: CleanupResult[];
  error: string | null;
};

type UsageTotals = {
  requests: number;
  input: number;
  output: number;
  reasoning: number;
  cacheRead: number;
  cacheWrite: number;
  cost: number;
  finishes: string[];
};

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));
const message = (error: unknown) =>
  String((error as Error)?.message ?? error).slice(0, 500);

function writeNew(path: string, value: unknown) {
  writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`, { flag: "wx" });
}

function makeHost(
  identity: ServerIdentity,
  options: { server: string; auth: ServerAuth; fetch?: FetchLike },
) {
  const fetchImpl = options.fetch ?? untimedFetch;
  return identity.kind === "v1"
    ? v1Host(options.server, options.auth, fetchImpl)
    : v2Host(options.server, options.auth, fetchImpl);
}

function gitState() {
  try {
    const git = (...args: string[]) =>
      execFileSync("git", args, {
        cwd: PACKAGE_DIR,
        encoding: "utf8",
        stdio: ["ignore", "pipe", "ignore"],
      }).trim();
    const commit = git("rev-parse", "HEAD");
    const dirty = git("status", "--porcelain", "--", ".");
    return { commit, packageDirty: dirty.length > 0 };
  } catch {
    return { commit: null, packageDirty: null };
  }
}

function usageBetween(
  lines: ChannelLine[],
  from: number,
  to: number,
  kind: "v1" | "v2",
): UsageTotals {
  const byMessage = new Map<string, ChannelLine>();
  for (const line of lines)
    if (
      line.kind === "usage"
      && line.at >= from
      && line.at <= to
      && (kind === "v2" || line["agent"] === SUMMARIZER)
    )
      byMessage.set(String(line["messageID"]), line);
  const totals: UsageTotals = {
    requests: 0,
    input: 0,
    output: 0,
    reasoning: 0,
    cacheRead: 0,
    cacheWrite: 0,
    cost: 0,
    finishes: [],
  };
  for (const line of byMessage.values()) {
    const tokens = (line["tokens"] ?? {}) as Record<string, unknown> & {
      cache?: Record<string, unknown>;
    };
    const num = (value: unknown) => (typeof value === "number" ? value : 0);
    totals.requests += 1;
    totals.input += num(tokens["input"]);
    totals.output += num(tokens["output"]);
    totals.reasoning += num(tokens["reasoning"]);
    totals.cacheRead += num(tokens.cache?.["read"]);
    totals.cacheWrite += num(tokens.cache?.["write"]);
    totals.cost += num(line["cost"]);
    totals.finishes.push(String(line["finish"]));
  }
  return totals;
}

async function waitFor<T>(
  probe: () => T | undefined,
  ms: number,
  every = 500,
): Promise<T | undefined> {
  const end = Date.now() + ms;
  for (;;) {
    const value = probe();
    if (value !== undefined || Date.now() >= end) return value;
    await sleep(every);
  }
}

export type RunContext = {
  corpus: Corpus;
  host: EvalHost;
  identity: ServerIdentity;
  ledger: Ledger;
  runNonce: string;
  out: string;
  channel: string;
  project: string;
  log: (line: string) => void;
  lostOutcomeWaitMs: number;
  // The process ID of the server that created this context's locations, as
  // recorded once preflight identified it; null when none was recorded.
  serverPid: number | null;
};

// Why the server answering now cannot be shown to be the process that
// created the run's locations: another server on the same URL, or the same
// one restarted, could hold another store. Only a matching live PID counts;
// a channel line from earlier is not proof.
async function serverProblem(ctx: RunContext): Promise<string | null> {
  if (ctx.serverPid === null)
    return "the server process that created this location was not recorded";
  let pid: number | null;
  try {
    pid = await ctx.host.serverPid(ctx.project, ctx.runNonce);
  } catch (error) {
    return `the server process could not be identified: ${message(error)}`;
  }
  return pid === ctx.serverPid
    ? null
    : `the server process changed (recorded PID ${ctx.serverPid}, now ${pid ?? "unknown"})`;
}

const expectedOf = (state: ExpectedState): ExpectedState => ({
  headerSha256: state.headerSha256,
  historySha256: state.historySha256,
});

// Removes one session the run created, only when a fresh read shows it at
// its own location with exactly the recorded header and history and no
// pending work. This is a fresh guard, not a transaction: another writer
// acting between the read and the delete is not excluded. Anything else is
// retained and reported.
export async function removeOwned(
  ctx: RunContext,
  directory: string,
  id: string,
  expected: ExpectedState | null,
  attemptId: string | null,
): Promise<CleanupResult> {
  try {
    const pending = await ctx.host.pending(directory, id);
    if (!pending.exists) {
      // Another server answering on the same URL would not have it either.
      const problem = await serverProblem(ctx);
      if (problem)
        return retain(ctx, id, attemptId, `absence not proven: ${problem}`);
      ctx.ledger.append("session-absent", { attemptId, id });
      return { id, action: "absent" };
    }
    if (pending.reasons.length)
      return retain(ctx, id, attemptId, pending.reasons.join("; "));
    if (!expected)
      return retain(ctx, id, attemptId, "no recorded state to compare");
    const now = await ctx.host.read(directory, id);
    if (now.directory !== directory)
      return retain(ctx, id, attemptId, "session is outside its eval location");
    if (!isDeepStrictEqual(expectedOf(now), expectedOf(expected)))
      return retain(
        ctx,
        id,
        attemptId,
        "session changed since the run recorded it",
      );
    await ctx.host.remove(directory, id);
    ctx.ledger.append("session-removed", { attemptId, id });
    return { id, action: "removed" };
  } catch (error) {
    return retain(
      ctx,
      id,
      attemptId,
      `ownership check failed: ${message(error)}`,
    );
  }
}

function retain(
  ctx: RunContext,
  id: string,
  attemptId: string | null,
  reason: string,
): CleanupResult {
  ctx.ledger.append("session-retained", { attemptId, id, reason });
  return { id, action: "retained", reason };
}

const controlLine = (ctx: RunContext, requestId: string, phases: string[]) =>
  readChannel(ctx.channel, ctx.runNonce).find(
    l =>
      l.kind === "control"
      && l["requestId"] === requestId
      && phases.includes(String(l["phase"])),
  );

// A successor's cache and stats live in the server's storage. The wrapper
// first confirms the successor is this run's (a receipt, no write), then the
// session is removed under the same checks as any other, and the wrapper
// removes the storage only once the host reports the deletion. A successor
// the run keeps keeps its storage; a deletion whose storage removal is not
// confirmed reports the storage paths left behind.
export async function removeSuccessor(
  ctx: RunContext,
  id: string,
  sourceID: string,
  expected: ExpectedState | null,
  attemptId: string | null,
): Promise<CleanupResult> {
  if (!expected)
    return retain(ctx, id, attemptId, "no recorded state to compare");
  const requestId = randomBytes(12).toString("hex");
  ctx.ledger.append("storage-receipt-request", {
    attemptId,
    id,
    requestId,
  });
  await ctx.host.runCommand(
    ctx.project,
    id,
    CONTROL_COMMAND,
    JSON.stringify({ runNonce: ctx.runNonce, requestId, sourceID }),
  );
  const receipt = await waitFor(
    () => controlLine(ctx, requestId, ["prepared", "refused"]),
    CHANNEL_WAIT_MS,
  );
  if (receipt?.["phase"] !== "prepared")
    return retain(
      ctx,
      id,
      attemptId,
      `the wrapper did not confirm the successor is this run's${receipt ? ` (${String(receipt["reason"])})` : ""}`,
    );
  const result = await removeOwned(ctx, ctx.project, id, expected, attemptId);
  if (result.action !== "removed") return result;
  const storage = await waitFor(
    () => controlLine(ctx, requestId, ["storage"]),
    CHANNEL_WAIT_MS,
  );
  if (storage?.["removed"] === true) {
    ctx.ledger.append("storage-removed", { attemptId, id, requestId });
    return { ...result, storage: "removed" };
  }
  const paths = (receipt["paths"] ?? null) as {
    cache: string;
    stats: string;
  } | null;
  ctx.ledger.append("storage-orphaned", { attemptId, id, requestId, paths });
  return { ...result, storage: { orphaned: paths } };
}

// A successor's state can authorize its removal only when the same read
// passed every mechanical check, there was at least one, and the source was
// unchanged: then that state is exactly what the product wrote.
const successorVerified = (
  checks: Array<{ ok: boolean }>,
  sourceUnchanged: boolean | null,
) => checks.length > 0 && checks.every(c => c.ok) && sourceUnchanged === true;

// The only ledger record a successor's removal accepts. It names the source
// state the checks ran against and the summary they read.
function recordVerifiedSuccessor(
  ctx: RunContext,
  attemptId: string | null,
  id: string,
  state: SessionState,
  facts: {
    sourceState: ExpectedState;
    outputSha256: string | null;
    reconciled?: true;
  },
) {
  ctx.ledger.append("successor-verified", {
    attemptId,
    id,
    ...expectedOf(state),
    source: expectedOf(facts.sourceState),
    outputSha256: facts.outputSha256,
    ...(facts.reconciled ? { reconciled: true } : {}),
  });
}

// A kept successor can later be verified, and its storage removed, only
// against its source, so the source stays with it.
const SOURCE_OF_KEPT_SUCCESSOR =
  "its successor is kept, and only this source can verify it later";

const sourceTitle = (runNonce: string, attemptId: string) =>
  `arkive compaction eval ${runNonce} ${attemptId}`;

export async function runAttempt(
  ctx: RunContext,
  corpusCase: CorpusCase,
  repeat: number,
  model: ModelSelector,
  descriptor: ModelDescriptor,
): Promise<AttemptRecord> {
  const attemptId = `r${repeat}-${corpusCase.id}`;
  const record: AttemptRecord = {
    runNonce: ctx.runNonce,
    model: model.selector,
    attemptId,
    caseId: corpusCase.id,
    repeat,
    status: "setup-failed",
    skipReason: null,
    sourceID: null,
    successorID: null,
    sourceChecks: [],
    successorChecks: [],
    sourceUnchanged: null,
    outcome: null,
    http: null,
    commandMs: null,
    capsule: null,
    outputSha256: null,
    storedHistory: null,
    usage: null,
    cleanup: [],
    error: null,
  };
  if (corpusCase.image && !descriptor.imageInput) {
    record.status = "preflight-skipped";
    record.skipReason = "the selected model does not advertise image input";
    ctx.ledger.append("attempt-skipped", {
      attemptId,
      reason: record.skipReason,
    });
    return record;
  }
  ctx.ledger.append("attempt-start", { attemptId });
  const ledger = ctx.ledger.sessions(attemptId);
  let before: SessionState | null = null;
  try {
    const source = await ctx.host.createSource({
      directory: ctx.project,
      title: sourceTitle(ctx.runNonce, attemptId),
      corpus: ctx.corpus,
      corpusCase,
      model,
      ledger,
    });
    record.sourceID = source.sessionID;
    record.sourceChecks = source.checks;
    if (source.template)
      record.cleanup.push(
        await removeOwned(
          ctx,
          ctx.project,
          source.template.id,
          source.template.expected,
          attemptId,
        ),
      );
    // The source's recorded state is the read that matched the corpus, never
    // a later or unvalidated one: a source someone else changed first has no
    // recorded state and is kept.
    if (source.checks.length > 0 && source.checks.every(check => check.ok)) {
      before = source.state;
      ctx.ledger.append("source-snapshot", {
        attemptId,
        id: source.sessionID,
        ...expectedOf(before),
      });
    }
  } catch (error) {
    record.error = `source setup failed: ${message(error)}`;
  }
  if (!record.sourceID || !before) {
    if (record.sourceID)
      record.cleanup.push(
        retain(
          ctx,
          record.sourceID,
          attemptId,
          "the source did not match the corpus when it was read; nothing was recorded for it",
        ),
      );
    record.error ??= "source history differs from the corpus";
    return record;
  }

  const sourceID = record.sourceID;
  const t0 = Date.now();
  ctx.ledger.append("command-start", { attemptId, sourceID, startedAt: t0 });
  const command = await ctx.host.runCommand(
    ctx.project,
    sourceID,
    COMPACT_COMMAND,
    ctx.corpus.command.arguments,
  );
  record.commandMs = command.ms;
  record.http =
    command.kind === "returned"
      ? { status: command.status, lost: false }
      : { status: null, lost: true };
  const line = await waitFor(
    () => commandOutcome(ctx, sourceID, t0),
    command.kind === "returned" ? CHANNEL_WAIT_MS : ctx.lostOutcomeWaitMs,
    1000,
  );
  const text =
    typeof line?.["message"] === "string"
      ? line["message"]
      : command.kind === "returned"
        ? command.message
        : null;
  ctx.ledger.append("command-end", {
    attemptId,
    sourceID,
    http: record.http,
    observed: !!line,
  });
  record.usage = usageBetween(
    readChannel(ctx.channel, ctx.runNonce),
    t0,
    Date.now(),
    ctx.host.kind,
  );
  if (text === null) {
    record.status = "outcome-unknown";
    record.error =
      "the command's outcome was not observed; its sessions are retained for `cleanup`";
    ctx.ledger.append("attempt-unknown", { attemptId, sourceID });
    return record;
  }
  record.outcome = classifyOutcome(text);
  record.status = record.outcome.kind;
  // The successor is on the ledger before the attempt counts as settled.
  if (record.outcome.kind === "delivered") {
    record.successorID = record.outcome.successorID;
    ctx.ledger.append("session-created", {
      attemptId,
      role: "successor",
      id: record.successorID,
      intentSeq: null,
    });
  }
  ctx.ledger.append("attempt-settled", { attemptId, status: record.status });
  try {
    const after = await ctx.host.read(ctx.project, sourceID);
    record.sourceUnchanged = isDeepStrictEqual(
      expectedOf(after),
      expectedOf(before),
    );
  } catch (error) {
    record.error = `source reread failed: ${message(error)}`;
  }

  let successorKept = false;
  if (record.successorID) {
    let verified: SessionState | null = null;
    try {
      const inspection = await ctx.host.inspectSuccessor({
        directory: ctx.project,
        source: before,
        successorID: record.successorID,
        corpus: ctx.corpus,
        corpusCase,
      });
      record.successorChecks = inspection.checks;
      record.capsule = inspection.capsule;
      record.outputSha256 =
        inspection.capsule === null ? null : outputSha256(inspection.capsule);
      record.storedHistory = inspection.storedHistory;
      if (successorVerified(inspection.checks, record.sourceUnchanged))
        verified = inspection.state;
    } catch (error) {
      record.error = `successor inspection failed: ${message(error)}`;
    }
    if (verified) {
      recordVerifiedSuccessor(ctx, attemptId, record.successorID, verified, {
        sourceState: before,
        outputSha256: record.outputSha256,
      });
      const removal = await removeSuccessor(
        ctx,
        record.successorID,
        sourceID,
        verified,
        attemptId,
      );
      record.cleanup.push(removal);
      successorKept = removal.action === "retained";
    } else {
      // What was read is not known to be only the product's output, so it
      // never becomes a removal baseline: the successor, its storage and
      // the location stay.
      record.status = "delivered-invalid";
      successorKept = true;
      record.cleanup.push(
        retain(
          ctx,
          record.successorID,
          attemptId,
          "the successor did not pass every mechanical check against its unchanged source; nothing was recorded for it",
        ),
      );
    }
  }
  record.cleanup.push(
    successorKept
      ? retain(ctx, sourceID, attemptId, SOURCE_OF_KEPT_SUCCESSOR)
      : await removeOwned(ctx, ctx.project, sourceID, before, attemptId),
  );
  return record;
}

const commandOutcome = (ctx: RunContext, sourceID: string, from: number) =>
  readChannel(ctx.channel, ctx.runNonce).find(
    line =>
      line.kind === "command-outcome"
      && line["sessionID"] === sourceID
      && line.at >= from,
  );

// Everything that must hold before the run creates a session: the eval
// wrapper, and only it, loaded once from this run's artifact; every product
// command is present; the summarizer resolves to the requested model and
// variant, both as configured and as the plugin itself reads it; the server
// lists that model and variant; and every inherited MCP server is disabled.
export function preflightProblems(input: {
  runNonce: string;
  artifactKey: string;
  setup: EffectiveSetup;
  witness: Witness | null;
  inits: number;
  model: ModelSelector;
  descriptor: ModelDescriptor | null;
  mcp: string[];
}): string[] {
  const { runNonce, setup, witness, model, descriptor, mcp } = input;
  const problems: string[] = [];
  if (!witness || witness.runNonce !== runNonce)
    problems.push("the eval wrapper did not load in the eval project");
  else {
    if (witness.artifactKey !== input.artifactKey)
      problems.push("the loaded wrapper belongs to another artifact");
    if (witness.subjectMismatches !== 0)
      problems.push("the installed subject differs from the packed artifact");
  }
  if (input.inits !== 1)
    problems.push(
      `expected one wrapper initialization in the eval project, saw ${input.inits}`,
    );
  const arkivePlugins = setup.plugins.filter(p => /arkive-compact/u.test(p));
  if (arkivePlugins.length !== 1)
    problems.push(
      `expected exactly one Arkive plugin in the eval project, found ${arkivePlugins.length}: ${arkivePlugins.join(", ")}`,
    );
  for (const name of [...Object.keys(COMMAND_DESCRIPTIONS), CONTROL_COMMAND])
    if (!setup.commands.includes(name))
      problems.push(`command ${name} is missing`);
  const wanted = `${model.providerID}/${model.modelID}`;
  const variant = model.variant ?? null;
  if (setup.summarizer.model !== wanted || setup.summarizer.variant !== variant)
    problems.push(
      `the summarizer resolved to ${setup.summarizer.model ?? "nothing"}#${setup.summarizer.variant ?? ""}, not ${model.selector}`,
    );
  else if (
    setup.subjectSummarizer.model !== wanted
    || setup.subjectSummarizer.variant !== variant
  )
    problems.push(
      `the plugin's own summarizer lookup returned ${setup.subjectSummarizer.model ?? "nothing"}#${setup.subjectSummarizer.variant ?? ""}, not ${model.selector}`,
    );
  if (!descriptor) problems.push(`the server's catalog has no model ${wanted}`);
  else if (model.variant && !descriptor.variants.includes(model.variant))
    problems.push(
      `model ${wanted} has no variant ${model.variant} (variants: ${descriptor.variants.join(", ") || "none"})`,
    );
  const notDisabled = mcp.filter(name => !setup.mcpDisabled.includes(name));
  if (notDisabled.length)
    problems.push(
      `MCP servers not disabled in the eval project: ${notDisabled.join(", ")}`,
    );
  return problems;
}

// The judge agent must resolve to exactly the requested model and variant
// at its own location, and the server must list them.
async function judgeProblems(
  host: EvalHost,
  location: string,
  judge: ModelSelector,
): Promise<{ problems: string[]; descriptor: ModelDescriptor | null }> {
  const problems: string[] = [];
  const descriptor = await host.describeModel(location, judge);
  const wanted = `${judge.providerID}/${judge.modelID}`;
  if (!descriptor)
    problems.push(`the server's catalog has no judge model ${wanted}`);
  else if (judge.variant && !descriptor.variants.includes(judge.variant))
    problems.push(`judge model has no variant ${judge.variant}`);
  const resolved = await host.resolvedAgent(location, JUDGE_AGENT);
  if (resolved.model !== wanted || resolved.variant !== (judge.variant ?? null))
    problems.push(
      `the judge agent resolved to ${resolved.model ?? "nothing"}#${resolved.variant ?? ""}, not ${judge.selector}`,
    );
  return { problems, descriptor };
}

export type RunResult = {
  out: string;
  attempts: AttemptRecord[];
  status: string;
};

export async function runEval(options: RunOptions): Promise<RunResult> {
  const log = options.log ?? (line => console.error(line));
  const requestedOut = resolve(options.out);
  if (existsSync(requestedOut))
    throw new EvalError(
      `--out ${requestedOut} already exists; every run needs a new directory.`,
    );
  if (!existsSync(dirname(requestedOut)))
    throw new EvalError(`The parent of --out ${requestedOut} does not exist.`);
  const corpus = loadCorpus();
  const plans = await checkCompactionPlans(corpus);
  if (
    plans.some(
      p =>
        p.v1.targets !== 1
        || p.v2.targets !== 1
        || p.v1.candidates + p.v2.candidates > 0,
    )
  )
    throw new EvalError(
      "The corpus no longer plans one retention-free target per case; run `check`.",
    );
  const selected = options.cases
    ? options.cases.map(id => {
        const found = corpus.cases.find(c => c.id === id);
        if (!found)
          throw new EvalError(
            `Unknown case ${id}; the corpus has ${corpus.cases.map(c => c.id).join(", ")}.`,
          );
        return found;
      })
    : corpus.cases;
  const fetchImpl = options.fetch ?? untimedFetch;
  const identity = await identifyServer(
    options.server,
    options.auth,
    fetchImpl,
    options.assertHostVersion ?? undefined,
  );
  const host = makeHost(identity, { ...options, fetch: fetchImpl });

  mkdirSync(requestedOut);
  const out = realpathSync(requestedOut);
  const channel = join(out, "channel");
  mkdirSync(channel, { mode: 0o700 });
  mkdirSync(join(out, "attempts"));
  const runNonce = randomBytes(16).toString("hex");
  const owner: Owner = { runNonce, out };
  const ledger = new Ledger(join(out, "ledger.jsonl"));
  ledger.append("run-start", {
    runNonce,
    server: options.server,
    host: identity,
  });
  const header: Record<string, unknown> = {
    format: RUN_FORMAT,
    runNonce,
    startedAt: new Date().toISOString(),
    server: identity,
    corpus: {
      id: corpus.corpusId,
      sha256: corpus.corpusSha256,
      rubricId: corpus.rubricId,
      rubricSha256: corpus.rubricSha256,
    },
    requested: {
      model: options.model.selector,
      judge: options.judge?.selector ?? null,
      repeat: options.repeat,
      cases: selected.map(c => c.id),
      dryRun: options.dryRun,
    },
    source: gitState(),
  };
  writeNew(join(out, "run.json"), header);
  const preflight: Record<string, unknown> = {};
  const finish = (status: string, extra: Record<string, unknown> = {}) => {
    writeNew(join(out, "preflight.json"), preflight);
    writeNew(join(out, "results.json"), {
      status,
      finishedAt: new Date().toISOString(),
      ...extra,
    });
  };

  const scratch: string[] = [];
  let artifact: Artifact;
  try {
    let subjectTgz = options.subject ? resolve(options.subject) : null;
    if (!subjectTgz) {
      const packDir = mkdtempSync(
        join(tmpdir(), "arkive-compaction-eval-pack-"),
      );
      scratch.push(packDir);
      subjectTgz = packSubject(PACKAGE_DIR, packDir);
    }
    artifact = prepareArtifact(
      subjectTgz,
      options.cacheRoot ?? defaultCacheRoot(),
    );
  } catch (error) {
    finish("preflight-failed", { error: `artifact: ${message(error)}` });
    throw error;
  } finally {
    for (const dir of scratch) rmSync(dir, { recursive: true, force: true });
  }
  preflight["artifact"] = artifact;
  log(
    `artifact ${artifact.key.slice(0, 16)} (${artifact.reused ? "reused" : "built"}), subject ${artifact.subjectTgzSha256.slice(0, 16)}`,
  );

  const root = createRoot(owner, "run");
  ledger.append("scratch-root", { directory: root });
  const locations: Array<{ directory: string; role: LocationRole }> = [];
  const addLocation = (
    role: LocationRole,
    config?: { file: string; content: unknown },
  ) => {
    const directory = createLocation(root, role, owner, config);
    locations.push({ directory, role });
    ledger.append("location", { role, directory });
    return directory;
  };
  const ctx: RunContext = {
    corpus,
    host,
    identity,
    ledger,
    runNonce,
    out,
    channel,
    project: "",
    log,
    lostOutcomeWaitMs: options.lostOutcomeWaitMs ?? 15 * 60_000,
    serverPid: null,
  };
  let stopping = false;
  const onSignal = () => {
    if (stopping) process.exit(130);
    stopping = true;
    log(
      "Interrupted: no new attempts start; the current one finishes and cleans up. Interrupt again to exit now (then use `cleanup`).",
    );
  };
  process.on("SIGINT", onSignal);
  const attempts: AttemptRecord[] = [];
  try {
    // Only for reading global config; OpenCode 2 evicts it right away and
    // OpenCode 1 never instantiates it.
    const mcp = await host.globalMcpNames(addLocation("inspect"));
    const pluginEntry = host.pluginEntry(artifact, {
      evalRunNonce: runNonce,
      evalChannel: channel,
      evalSummarizerVariant: options.model.variant ?? null,
    });
    const project = addLocation(
      "project",
      host.projectConfig({ pluginEntry, summarizer: options.model, mcp }),
    );
    ctx.project = project;
    const { setup, witness } = await host.setup(project);
    const descriptor = await host.describeModel(project, options.model);
    const inits = readChannel(channel, runNonce).filter(
      l => l.kind === "init" && l["directory"] === project,
    );
    const problems = preflightProblems({
      runNonce,
      artifactKey: artifact.key,
      setup,
      witness,
      inits: inits.length,
      model: options.model,
      descriptor,
      mcp,
    });
    // The live server process the run's sessions will belong to: the
    // wrapper's witness on OpenCode 1, /api/info on OpenCode 2.
    const serverPid =
      problems.length === 0 ? await host.serverPid(project, runNonce) : null;
    if (
      problems.length === 0
      && (serverPid === null
        || (identity.pid !== null && serverPid !== identity.pid))
    )
      problems.push(
        `the server process could not be identified (reported PID ${serverPid ?? "none"})`,
      );
    let judgeDescriptor: ModelDescriptor | null = null;
    let judgeLocation: string | null = null;
    if (options.judge && problems.length === 0) {
      judgeLocation = addLocation(
        "judge",
        host.judgeConfig({ judge: options.judge, mcp, system: JUDGE_SYSTEM }),
      );
      const judged = await judgeProblems(host, judgeLocation, options.judge);
      judgeDescriptor = judged.descriptor;
      problems.push(...judged.problems);
    }
    Object.assign(preflight, {
      setup,
      witness,
      wrapperInits: inits.length,
      mcpNames: mcp,
      problems,
      model: descriptor,
      judgeModel: judgeDescriptor,
      locations: { root, project, judge: judgeLocation },
      serverPid,
    });
    if (problems.length) {
      ledger.append("preflight-failed", { problems });
      throw new EvalError(
        `Preflight refused before creating any session:\n- ${problems.join("\n- ")}`,
      );
    }
    ledger.append("server-witness", { directory: root, pid: serverPid });
    ctx.serverPid = serverPid;
    log(
      `preflight passed on OpenCode ${identity.version} (${identity.kind}); model ${options.model.selector}`,
    );
    if (!options.dryRun) {
      for (let repeat = 1; repeat <= options.repeat && !stopping; repeat++) {
        for (const corpusCase of selected) {
          if (stopping) break;
          log(`attempt r${repeat}-${corpusCase.id} ...`);
          const record = await runAttempt(
            ctx,
            corpusCase,
            repeat,
            options.model,
            descriptor!,
          );
          writeNew(join(out, "attempts", `${record.attemptId}.json`), record);
          attempts.push(record);
          log(
            `attempt ${record.attemptId}: ${record.status}${record.outcome?.kind === "failed" ? ` (${record.outcome.category})` : ""}`,
          );
        }
      }
      if (options.judge && judgeLocation && !stopping) {
        await gradeWithJudge(ctx, judgeLocation, options.judge, attempts);
      }
    }
  } catch (error) {
    await disposeLocations(ctx, owner, root, locations);
    process.off("SIGINT", onSignal);
    finish(
      error instanceof EvalError && attempts.length === 0
        ? "preflight-failed"
        : "failed",
      {
        error: message(error),
        attempts: attempts.map(a => a.attemptId),
      },
    );
    throw error;
  }
  const disposal = await disposeLocations(ctx, owner, root, locations);
  process.off("SIGINT", onSignal);
  const status = options.dryRun
    ? "dry-run"
    : stopping
      ? "interrupted"
      : "completed";
  finish(status, { attempts: attempts.map(a => a.attemptId), disposal });
  return { out, attempts, status };
}

// Disposes each owned location and deletes the scratch tree, only when the
// ledger leaves nothing open, the root and every location prove ownership by
// their markers, and the host lists no session at any of them. Otherwise
// nothing is disposed or deleted, so retained or unknown sessions keep their
// location, config and wrapper for a later `cleanup`.
async function disposeLocations(
  ctx: RunContext,
  owner: Owner,
  root: string,
  locations: Array<{ directory: string; role: LocationRole }>,
) {
  const keep = (reason: string, extra: Record<string, unknown> = {}) => {
    ctx.ledger.append("scratch-kept", { directory: root, reason, ...extra });
    return { scratchRemoved: false, reason, locations: [] };
  };
  const open = outstanding(Ledger.read(ctx.ledger.path));
  if (open.sessions.length + open.intents.length + open.attempts.length > 0)
    return keep(
      "the run still owns sessions, unresolved session intents or attempts with an unknown outcome",
    );
  const problem =
    rootProblem(root, owner)
    ?? locations
      .map(l => locationProblem(root, l.directory, owner))
      .find(Boolean)
    ?? null;
  if (problem) return keep(`ownership not proven: ${problem}`);
  // An empty session list from another server proves nothing about this
  // one. Without a recorded process (preflight stopped before identifying
  // it, so no session exists) the endpoint itself is the binding.
  const server = ctx.serverPid === null ? null : await serverProblem(ctx);
  if (server) return keep(server);
  // The inspect location is never instantiated, so it is neither listed nor
  // disposed; it is only deleted.
  const instantiated = locations.filter(l => l.role !== "inspect");
  for (const { directory } of instantiated) {
    try {
      const listed = await ctx.host.inventory(directory);
      if (listed.length)
        return keep("the host still lists sessions at an eval location", {
          location: directory,
          sessions: listed,
        });
    } catch (error) {
      return keep(`could not list sessions at ${directory}: ${message(error)}`);
    }
  }
  const results: Array<{
    directory: string;
    disposed: boolean;
    error?: string;
  }> = [];
  for (const { directory } of instantiated) {
    try {
      await ctx.host.dispose(directory);
      ctx.ledger.append("location-disposed", { directory });
      results.push({ directory, disposed: true });
    } catch (error) {
      ctx.ledger.append("location-dispose-failed", {
        directory,
        error: message(error),
      });
      results.push({ directory, disposed: false, error: message(error) });
    }
  }
  if (!results.every(r => r.disposed))
    return { ...keep("a location was not disposed"), locations: results };
  const removal = removeOwnedTree(
    root,
    locations.map(l => l.directory),
    owner,
  );
  if (removal) return { ...keep(removal), locations: results };
  ctx.ledger.append("scratch-removed", { directory: root });
  return { scratchRemoved: true, reason: null, locations: results };
}

async function gradeWithJudge(
  ctx: RunContext,
  judgeLocation: string,
  judge: ModelSelector,
  attempts: AttemptRecord[],
) {
  const items: GradesFile["items"] = [];
  const results: Array<Record<string, unknown>> = [];
  for (const attempt of attempts) {
    if (
      !attempt.capsule
      || !attempt.outputSha256
      || attempt.status !== "delivered"
    )
      continue;
    const corpusCase = ctx.corpus.cases.find(c => c.id === attempt.caseId)!;
    const blindID = randomBytes(8).toString("hex");
    ctx.log(`judge ${blindID} ...`);
    const answer = await ctx.host
      .judge({
        directory: judgeLocation,
        judge,
        title: `arkive compaction eval judge ${ctx.runNonce} ${blindID}`,
        prompt: judgePrompt(corpusCase, ctx.corpus.followup, attempt.capsule),
        ledger: ctx.ledger.sessions(attempt.attemptId),
      })
      .catch(error => ({
        sessionID: null,
        rawText: null,
        ms: 0,
        usage: null,
        error: `judge setup failed: ${message(error)}`,
        expected: null,
        ownership: null,
      }));
    // On OpenCode 1 the judge's context is its session history; anything but
    // the run's own prompt in it would have reached the judge.
    let problem =
      answer.error
      ?? (answer.ownership && ctx.host.kind === "v1"
        ? `judge context was not the blind packet: ${answer.ownership}`
        : null)
      ?? (answer.rawText === null ? "judge returned no answer" : null);
    if (!problem) {
      const parsed = parseJudgeAnswer(answer.rawText!);
      problem =
        "error" in parsed
          ? parsed.error
          : judgmentProblem(parsed.judgment, corpusCase, attempt.capsule);
      if (!problem && "judgment" in parsed)
        items.push({
          attemptId: attempt.attemptId,
          caseId: attempt.caseId,
          outputSha256: attempt.outputSha256,
          ...parsed.judgment,
        });
    }
    let cleanup: CleanupResult | null = null;
    if (answer.sessionID) {
      if (answer.expected) {
        ctx.ledger.append("judge-snapshot", {
          attemptId: attempt.attemptId,
          id: answer.sessionID,
          ...answer.expected,
        });
        cleanup = await removeOwned(
          ctx,
          judgeLocation,
          answer.sessionID,
          answer.expected,
          attempt.attemptId,
        );
      } else
        cleanup = retain(
          ctx,
          answer.sessionID,
          attempt.attemptId,
          answer.ownership ?? "the judge session's state is unknown",
        );
    }
    results.push({
      attemptId: attempt.attemptId,
      blindID,
      ms: answer.ms,
      usage: answer.usage,
      judgeError: problem,
      cleanup,
    });
  }
  const grades: GradesFile = {
    format: GRADES_FORMAT,
    runNonce: ctx.runNonce,
    corpusSha256: ctx.corpus.corpusSha256,
    rubricSha256: ctx.corpus.rubricSha256,
    grader: {
      kind: "judge",
      label: `judge ${judge.selector}`,
      model: judge.selector,
    },
    items,
  };
  mkdirSync(join(ctx.out, "grades"), { recursive: true });
  const stamp = new Date().toISOString().replace(/[:.]/gu, "-");
  writeNew(join(ctx.out, "grades", `judge-${stamp}.json`), grades);
  writeNew(join(ctx.out, "grades", `judge-${stamp}.calls.json`), results);
}

// --- saved runs ------------------------------------------------------------

const Hex = z.string().regex(/^[0-9a-f]{64}$/u);
const RunHeaderSchema = z.looseObject({
  format: z.literal(RUN_FORMAT),
  runNonce: z.string().regex(/^[0-9a-f]{32}$/u),
  server: z.looseObject({
    url: z.string(),
    kind: z.enum(["v1", "v2"]),
    version: z.string(),
    qualification: z.enum(["qualified", "operator-asserted"]),
    pid: z.number().int().nullable(),
  }),
  corpus: z.strictObject({
    id: z.string(),
    sha256: Hex,
    rubricId: z.string(),
    rubricSha256: Hex,
  }),
  requested: z.looseObject({
    model: z.string(),
    judge: z.string().nullable(),
    repeat: z.number().int().min(1).max(20),
    cases: z.array(z.string()).min(1),
  }),
});
export type RunHeader = z.infer<typeof RunHeaderSchema>;

const Checks = z.array(z.strictObject({ name: z.string(), ok: z.boolean() }));
const AttemptSchema = z.looseObject({
  runNonce: z.string(),
  model: z.string(),
  sourceID: z.string().nullable(),
  successorID: z.string().nullable(),
  attemptId: z.string(),
  caseId: z.string(),
  repeat: z.number().int(),
  status: z.enum(ATTEMPT_STATUSES),
  successorChecks: Checks,
  sourceUnchanged: z.boolean().nullable(),
  capsule: z.string().nullable(),
  outputSha256: Hex.nullable(),
  cleanup: z.array(z.looseObject({ id: z.string() })),
});

export function readRunHeader(dir: string): RunHeader {
  const path = join(dir, "run.json");
  let raw: unknown;
  try {
    raw = JSON.parse(readFileSync(path, "utf8"));
  } catch {
    throw new EvalError(`${dir} is not a ${RUN_FORMAT} run directory.`);
  }
  const parsed = RunHeaderSchema.safeParse(raw);
  if (!parsed.success)
    throw new EvalError(
      `${dir} is not a valid ${RUN_FORMAT} run (${parsed.error.issues[0]?.path.join(".")}: ${parsed.error.issues[0]?.message}).`,
    );
  return parsed.data;
}

// Reads a saved run and refuses it as a whole unless it matches the corpus
// it names and every attempt record is one that run could have written: a
// requested case and repeat, a known status, and an output digest
// recomputed from the saved capsule itself. Grading and reporting only ever
// see a run that passed.
export function readRun(dir: string, corpus: Corpus = loadCorpus()) {
  const run = readRunHeader(dir);
  const fail = (why: string): never => {
    throw new EvalError(`Refusing run ${dir}: ${why}.`);
  };
  if (
    run.corpus.id !== corpus.corpusId
    || run.corpus.sha256 !== corpus.corpusSha256
    || run.corpus.rubricId !== corpus.rubricId
    || run.corpus.rubricSha256 !== corpus.rubricSha256
  )
    fail("this checkout's corpus or rubric differs from the run's");
  for (const id of run.requested.cases)
    if (!corpus.cases.some(c => c.id === id))
      fail(`it requested unknown case ${id}`);
  const read = (name: string) =>
    existsSync(join(dir, name))
      ? (JSON.parse(readFileSync(join(dir, name), "utf8")) as Record<
          string,
          unknown
        >)
      : null;
  const attemptsDir = join(dir, "attempts");
  const attemptFiles = existsSync(attemptsDir)
    ? readdirSync(attemptsDir).sort()
    : [];
  const ledgerPath = join(dir, "ledger.jsonl");
  const ledger =
    attemptFiles.length && existsSync(ledgerPath)
      ? Ledger.read(ledgerPath)
      : [];
  const attempts: AttemptRecord[] = [];
  for (const name of attemptFiles) {
    const parsed = AttemptSchema.safeParse(
      JSON.parse(readFileSync(join(attemptsDir, name), "utf8")),
    );
    if (!parsed.success)
      fail(
        `attempts/${name} is not an attempt record (${parsed.error.issues[0]?.message})`,
      );
    const a = parsed.data!;
    if (
      name !== `${a.attemptId}.json`
      || a.attemptId !== `r${a.repeat}-${a.caseId}`
    )
      fail(`attempts/${name} names another attempt`);
    if (a.runNonce !== run.runNonce || a.model !== run.requested.model)
      fail(`attempts/${name} was written by another run`);
    // Every session an attempt names was created for it in this run's
    // ledger. Sessions it does not name are not required.
    for (const [role, id] of [
      ["source", a.sourceID],
      ["successor", a.successorID],
    ] as const)
      if (
        id !== null
        && !ledger.some(
          e =>
            e.kind === "session-created"
            && e["id"] === id
            && e["role"] === role
            && e["attemptId"] === a.attemptId,
        )
      )
        fail(`attempts/${name} names a ${role} this run did not create for it`);
    if (!run.requested.cases.includes(a.caseId))
      fail(`attempts/${name} is for a case the run did not request`);
    if (a.repeat < 1 || a.repeat > run.requested.repeat)
      fail(`attempts/${name} is for a repeat the run did not request`);
    if (
      (a.capsule === null) !== (a.outputSha256 === null)
      || (a.capsule !== null && outputSha256(a.capsule) !== a.outputSha256)
    )
      fail(`attempts/${name}: the saved summary does not match its digest`);
    if (
      a.status === "delivered"
      && (a.capsule === null
        || a.sourceUnchanged !== true
        || a.successorChecks.length === 0
        || a.successorChecks.some(c => !c.ok))
    )
      fail(`attempts/${name} is delivered without a valid successor record`);
    attempts.push(a as unknown as AttemptRecord);
  }
  return {
    run,
    preflight: read("preflight.json"),
    results: read("results.json"),
    attempts,
  };
}

// Grading and cleanup talk only to the server the run used, named by the
// exact canonical URL it recorded. Another URL, even another name for the
// same address, may reach another server with another store.
async function sameServer(
  run: RunHeader,
  options: {
    server: string;
    auth: ServerAuth;
    assertHostVersion: string | null;
  },
  fetchImpl: FetchLike,
): Promise<ServerIdentity> {
  if (options.server !== run.server.url)
    throw new EvalError(
      `This run used the server at ${run.server.url}; pass exactly --server ${run.server.url}. Nothing was changed.`,
    );
  const identity = await identifyServer(
    options.server,
    options.auth,
    fetchImpl,
    options.assertHostVersion ?? undefined,
  );
  if (identity.kind !== run.server.kind)
    throw new EvalError(
      "This server is not the kind of host the run used. Nothing was changed.",
    );
  return identity;
}

// Grading roots are owned by the run directory they grade.
function ownerOf(dir: string, run: RunHeader): Owner {
  return { runNonce: run.runNonce, out: realpathSync(dir) };
}

export async function gradeRunWithJudge(options: {
  dir: string;
  server: string;
  auth: ServerAuth;
  judge: ModelSelector;
  assertHostVersion: string | null;
  fetch?: FetchLike;
  log?: (line: string) => void;
}) {
  const log = options.log ?? (line => console.error(line));
  const corpus = loadCorpus();
  const { run, attempts } = readRun(options.dir, corpus);
  const fetchImpl = options.fetch ?? untimedFetch;
  const identity = await sameServer(run, options, fetchImpl);
  const host = makeHost(identity, { ...options, fetch: fetchImpl });
  const ledger = new Ledger(join(options.dir, "ledger.jsonl"));
  const owner = ownerOf(options.dir, run);
  const root = createRoot(owner, "judge");
  ledger.append("scratch-root", { directory: root });
  const ctx: RunContext = {
    corpus,
    host,
    identity,
    ledger,
    runNonce: run.runNonce,
    out: owner.out,
    channel: join(options.dir, "channel"),
    project: "",
    log,
    lostOutcomeWaitMs: 0,
    serverPid: null,
  };
  const locations: Array<{ directory: string; role: LocationRole }> = [];
  const addLocation = (
    role: LocationRole,
    config?: { file: string; content: unknown },
  ) => {
    const directory = createLocation(root, role, owner, config);
    locations.push({ directory, role });
    ledger.append("location", { role, directory });
    return directory;
  };
  try {
    const mcp = await host.globalMcpNames(addLocation("inspect"));
    const judgeLocation = addLocation(
      "judge",
      host.judgeConfig({ judge: options.judge, mcp, system: JUDGE_SYSTEM }),
    );
    const { problems } = await judgeProblems(
      host,
      judgeLocation,
      options.judge,
    );
    if (problems.length)
      throw new EvalError(
        `The server cannot run judge ${options.judge.selector}; nothing was sent:\n- ${problems.join("\n- ")}`,
      );
    // OpenCode 2 names its process. The OpenCode 1 judge location loads no
    // eval wrapper, so its root stays unbound and a later `cleanup` keeps
    // whatever it still holds.
    ctx.serverPid = identity.pid;
    if (identity.pid !== null)
      ledger.append("server-witness", { directory: root, pid: identity.pid });
    await gradeWithJudge(ctx, judgeLocation, options.judge, attempts);
  } finally {
    await disposeLocations(ctx, owner, root, locations);
  }
}

// --- cleanup of an interrupted run -------------------------------------------

// Any entry at the path counts, a dangling symlink included; only a missing
// path is absent.
function entryExists(path: string): boolean {
  try {
    lstatSync(path);
    return true;
  } catch (error) {
    return (error as NodeJS.ErrnoException).code !== "ENOENT";
  }
}

// Reconciles a run from its ledger, then removes only sessions the run
// created whose recorded state is unchanged, inactive and childless, and
// finally disposes the run's locations when nothing it owns remains.
//
// Reconciliation adds facts, it never assumes them: an attempt whose command
// started settles only on the command's own outcome line in the channel (a
// delivered successor is registered first); an intent with a planned ID
// settles on a read of that ID; an intent without one settles only when the
// location's session list shows exactly one session with its title, or
// none. A successor registered this way is removable only when it passes
// every mechanical check against its unchanged source, which pins down its
// whole content.
export async function cleanupRun(options: {
  dir: string;
  server: string;
  auth: ServerAuth;
  assertHostVersion: string | null;
  fetch?: FetchLike;
  log?: (line: string) => void;
  host?: EvalHost;
}) {
  const log = options.log ?? (line => console.error(line));
  const run = readRunHeader(options.dir);
  const fetchImpl = options.fetch ?? untimedFetch;
  const identity = await sameServer(run, options, fetchImpl);
  const host =
    options.host ?? makeHost(identity, { ...options, fetch: fetchImpl });
  const owner = ownerOf(options.dir, run);
  const ledgerPath = join(options.dir, "ledger.jsonl");
  const ledger = new Ledger(ledgerPath);
  const channel = join(options.dir, "channel");
  const corpus = loadCorpus();
  const first = Ledger.read(ledgerPath);
  // Locations count only when they prove ownership on disk.
  const roots = [
    ...new Set(
      first
        .filter(e => e.kind === "scratch-root")
        .map(e => String(e["directory"])),
    ),
  ];
  const owned = (directory: string) =>
    roots.some(
      root =>
        dirname(directory) === root
        && !rootProblem(root, owner)
        && !locationProblem(root, directory, owner),
    );
  const projectEntry = first.find(
    e => e.kind === "location" && e["role"] === "project",
  );
  const project =
    projectEntry && owned(String(projectEntry["directory"]))
      ? String(projectEntry["directory"])
      : "";
  const ctx: RunContext = {
    corpus,
    host,
    identity,
    ledger,
    runNonce: run.runNonce,
    out: owner.out,
    channel,
    project,
    log,
    lostOutcomeWaitMs: 0,
    serverPid: null,
  };
  // Each root is bound to the server process that created its locations.
  // Nothing in a root is read for absence, removed or disposed unless the
  // live process is that one.
  const rootPid = new Map(
    first
      .filter(e => e.kind === "server-witness")
      .map(e => [String(e["directory"]), Number(e["pid"])]),
  );
  let livePid: number | null = null;
  try {
    livePid = await host.serverPid(project, run.runNonce);
  } catch (error) {
    log(`the server process could not be identified: ${message(error)}`);
  }
  const unproven = (root: string) =>
    !rootPid.has(root)
      ? "the server process that created it was not recorded"
      : rootPid.get(root) !== livePid
        ? `the server process changed (recorded PID ${rootPid.get(root)}, now ${livePid ?? "unknown"}); keep it for manual review`
        : null;
  const rootCtx = (root: string): RunContext => ({
    ...ctx,
    serverPid: rootPid.get(root) ?? null,
  });
  for (const root of roots)
    if (existsSync(root) && rootPid.has(root) && unproven(root))
      log(
        `Not cleaning ${root}: ${unproven(root)}. The server at ${run.server.url} is not the process that created it, so everything there is kept.`,
      );
  // A session belongs to the location recorded for its role before it.
  const locationOf = (entries: LedgerEntry[], entry: LedgerEntry) => {
    const role = entry["role"] === "judge" ? "judge" : "project";
    const directory = String(
      entries
        .filter(
          e => e.seq < entry.seq && e.kind === "location" && e["role"] === role,
        )
        .at(-1)?.["directory"] ?? "",
    );
    return owned(directory) ? directory : null;
  };

  // Commands that settled after the CLI stopped waiting.
  for (const entry of outstanding(first).attempts) {
    const attemptId = String(entry["attemptId"]);
    const from =
      typeof entry["startedAt"] === "number"
        ? entry["startedAt"]
        : Date.parse(entry.at);
    const lines = readChannel(channel, run.runNonce).filter(
      l => l["sessionID"] === entry["sourceID"] && l.at >= from,
    );
    const done = lines.find(l => l.kind === "command-outcome");
    if (!done) {
      ledger.append("attempt-reconciled", {
        attemptId,
        outcome: lines.some(l => l.kind === "command-started")
          ? "still running"
          : "still unknown",
      });
      continue;
    }
    const outcome =
      typeof done["message"] === "string"
        ? classifyOutcome(done["message"])
        : null;
    if (
      outcome?.kind === "delivered"
      && !Ledger.read(ledgerPath).some(
        e => e.kind === "session-created" && e["id"] === outcome.successorID,
      )
    )
      ledger.append("session-created", {
        attemptId,
        role: "successor",
        id: outcome.successorID,
        intentSeq: null,
        reconciled: true,
      });
    ledger.append("attempt-settled", {
      attemptId,
      status: outcome?.kind ?? "failed",
      reconciled: true,
    });
  }

  // Creations whose returned ID never reached the ledger.
  const known = new Set(
    Ledger.read(ledgerPath)
      .filter(e => e.kind === "session-created")
      .map(e => String(e["id"])),
  );
  for (const intent of outstanding(Ledger.read(ledgerPath)).intents) {
    const directory = locationOf(Ledger.read(ledgerPath), intent);
    if (!directory || unproven(dirname(directory))) continue;
    const settle = (id: string) => {
      known.add(id);
      ledger.append("session-created", {
        attemptId: intent["attemptId"],
        role: intent["role"],
        id,
        intentSeq: intent.seq,
        reconciled: true,
      });
    };
    const absent = () =>
      ledger.append("session-intent-resolved", {
        intentSeq: intent.seq,
        outcome: "absent",
      });
    try {
      if (typeof intent["plannedID"] === "string") {
        const id = intent["plannedID"];
        if (!(await host.pending(directory, id)).exists) absent();
        else {
          const state = await host.read(directory, id);
          if (
            state.directory === directory
            && state.session["title"] === intent["title"]
          )
            settle(id);
        }
        continue;
      }
      const matches: string[] = [];
      for (const id of await host.inventory(directory)) {
        if (known.has(id)) continue;
        const state = await host.read(directory, id);
        if (state.session["title"] === intent["title"]) matches.push(id);
      }
      if (matches.length === 0) absent();
      else if (matches.length === 1) settle(matches[0]!);
    } catch (error) {
      log(`intent ${intent.seq} not reconciled: ${message(error)}`);
    }
  }

  let all = Ledger.read(ledgerPath);
  // A successor's removal baseline is only a state recorded as verified;
  // any other session's is the state recorded when the run validated it.
  const snapshotOf = (
    id: string,
    kinds: string[] = ["source-snapshot", "judge-snapshot"],
  ): ExpectedState | null => {
    const entry = [...all]
      .reverse()
      .find(e => kinds.includes(e.kind) && e["id"] === id);
    return entry
      ? {
          headerSha256: String(entry["headerSha256"]),
          historySha256: String(entry["historySha256"]),
        }
      : null;
  };
  const unsettled = new Set(
    outstanding(all).attempts.map(e => String(e["attemptId"])),
  );
  const sourceOf = (attemptId: string) =>
    all.find(e => e.kind === "command-start" && e["attemptId"] === attemptId);

  const results: CleanupResult[] = [];
  // Attempts whose successor stays; their source stays with it.
  const keptSuccessor = new Set<string>();
  // Successors first: their storage removal runs through the wrapper, which
  // the project location loads, and their removal needs the source.
  const created = outstanding(all).sessions.sort(
    (a, b) =>
      Number(b["role"] === "successor") - Number(a["role"] === "successor"),
  );
  for (const entry of created) {
    const id = String(entry["id"]);
    const attemptId =
      entry["attemptId"] == null ? null : String(entry["attemptId"]);
    const directory = locationOf(all, entry);
    if (!directory) {
      results.push(
        retain(
          ctx,
          id,
          attemptId,
          "its eval location does not prove ownership",
        ),
      );
      continue;
    }
    const notProven = unproven(dirname(directory));
    if (notProven) {
      results.push(
        retain(ctx, id, attemptId, `server identity not proven: ${notProven}`),
      );
      continue;
    }
    if (attemptId && unsettled.has(attemptId)) {
      results.push(
        retain(ctx, id, attemptId, "its attempt's outcome is unknown"),
      );
      continue;
    }
    const bound = rootCtx(dirname(directory));
    if (
      entry["role"] === "source"
      && attemptId
      && keptSuccessor.has(attemptId)
    ) {
      results.push(retain(ctx, id, attemptId, SOURCE_OF_KEPT_SUCCESSOR));
      continue;
    }
    if (entry["role"] !== "successor") {
      results.push(
        await removeOwned(bound, directory, id, snapshotOf(id), attemptId),
      );
      continue;
    }
    const sourceID = String(sourceOf(attemptId ?? "")?.["sourceID"] ?? "");
    let expected = snapshotOf(id, ["successor-verified"]);
    if (!expected)
      expected = await reconciledSuccessor(bound, directory, id, sourceID, {
        attemptId,
        sourceSnapshot: snapshotOf(sourceID, ["source-snapshot"]),
      });
    const removal = await removeSuccessor(
      bound,
      id,
      sourceID,
      expected,
      attemptId,
    );
    results.push(removal);
    if (removal.action === "retained" && attemptId)
      keptSuccessor.add(attemptId);
    all = Ledger.read(ledgerPath);
  }

  const after = Ledger.read(ledgerPath);
  const disposal = [];
  for (const root of roots) {
    const problem = rootProblem(root, owner);
    if (problem) {
      // A root that is gone was cleaned. Anything else at its path stays,
      // untouched, and fails the cleanup.
      if (entryExists(root)) {
        const reason = `ownership not proven: ${problem}`;
        ctx.ledger.append("scratch-kept", { directory: root, reason });
        disposal.push({
          directory: root,
          scratchRemoved: false,
          reason,
          locations: [],
        });
      }
      continue;
    }
    const locations = after
      .filter(
        e =>
          e.kind === "location"
          && dirname(String(e["directory"])) === root
          && existsSync(String(e["directory"])),
      )
      .map(e => ({
        directory: String(e["directory"]),
        role: basename(String(e["directory"])) as LocationRole,
      }));
    // An unbound root is disposable only when the run never created a
    // session in it: preflight stopped before identifying the server.
    const notProven = unproven(root);
    const created = after.some(
      e =>
        (e.kind === "session-created" || e.kind === "session-intent")
        && dirname(locationOf(after, e) ?? "") === root,
    );
    if (notProven && (rootPid.has(root) || created)) {
      const reason = `server identity not proven: ${notProven}`;
      ctx.ledger.append("scratch-kept", { directory: root, reason });
      disposal.push({
        directory: root,
        scratchRemoved: false,
        reason,
        locations: [],
      });
      continue;
    }
    disposal.push({
      directory: root,
      ...(await disposeLocations(rootCtx(root), owner, root, locations)),
    });
  }
  const open = outstanding(Ledger.read(ledgerPath));
  const summary = {
    sessions: results,
    disposal,
    stillOwned: open.sessions.map(e => e["id"]),
    unknownOutcomes: open.attempts.map(e => e["attemptId"]),
    unresolvedIntents: open.intents.map(e => ({
      attemptId: e["attemptId"],
      role: e["role"],
      plannedID: e["plannedID"],
    })),
  };
  log(JSON.stringify(summary, null, 2));
  return summary;
}

// A successor known only from a late command outcome has no recorded state.
// It counts as the run's own exactly when its source is unchanged since the
// run recorded it and the successor passes every mechanical check against
// that source; its state is then recorded as it was read.
async function reconciledSuccessor(
  ctx: RunContext,
  directory: string,
  id: string,
  sourceID: string,
  facts: { attemptId: string | null; sourceSnapshot: ExpectedState | null },
): Promise<ExpectedState | null> {
  const caseId = facts.attemptId?.replace(/^r\d+-/u, "");
  const corpusCase = ctx.corpus.cases.find(c => c.id === caseId);
  if (!facts.sourceSnapshot || !corpusCase) return null;
  try {
    const source = await ctx.host.read(directory, sourceID);
    if (!isDeepStrictEqual(expectedOf(source), facts.sourceSnapshot))
      return null;
    const inspection = await ctx.host.inspectSuccessor({
      directory,
      source,
      successorID: id,
      corpus: ctx.corpus,
      corpusCase,
    });
    if (!successorVerified(inspection.checks, true)) return null;
    recordVerifiedSuccessor(ctx, facts.attemptId, id, inspection.state, {
      sourceState: facts.sourceSnapshot,
      outputSha256:
        inspection.capsule === null ? null : outputSha256(inspection.capsule),
      reconciled: true,
    });
    return expectedOf(inspection.state);
  } catch {
    return null;
  }
}

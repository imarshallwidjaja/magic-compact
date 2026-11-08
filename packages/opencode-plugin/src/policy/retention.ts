// Task-aware retention for /magic-compact and /magic-trim, shared by both
// hosts. The existing tool rules still decide which calls would change; this
// step only vetoes changes to calls the selected decision backend judges
// likely to help the current task. Decisions are made per invocation and
// never persisted, so kept calls are reassessed next time.
import {
  DEFAULT_RETENTION_BACKEND,
  isRetentionBackendID,
  RETENTION_BACKEND_ENV,
  RETENTION_BACKENDS,
  type RetentionBackendID,
} from "../retention/backends";
import {
  classifyRetention,
  formatKeepThresholdPercent,
  resolveKeepThreshold,
  retentionFailureGuidance,
  RETENTION_KEEP_THRESHOLD,
  RETENTION_KEEP_THRESHOLD_PERCENT_ENV,
  RetentionClassificationError,
  type RetentionCandidate,
  type RetentionFetch,
  type RetentionInvocation,
} from "../retention/classifier";
import {
  hasPluginMessage,
  PluginMessageError,
  unremovedSessionsText,
  type LeavesSessions,
  type UnremovedSession,
} from "./feedback";
import type { StageReport } from "./reasoning";
import { toolTrimChangesContent, type ToolCallView } from "./tool-rules";

export type RetentionPolicy =
  | { mode: "heuristic" }
  | {
      mode: "task-aware";
      // Defaults to DEFAULT_RETENTION_BACKEND.
      backend?: RetentionBackendID;
      // Supplied by the plugin runtime from the selected backend's key
      // variable, and from no other.
      apiKey: string | undefined;
      // Probability threshold from 0 to 1; defaults to RETENTION_KEEP_THRESHOLD.
      keepThreshold?: number;
      // Set when a command ran: its arguments, for `--truncate` and for the
      // command a failure suggests. Absent for API and evaluation callers.
      invocation?: RetentionInvocation;
      fetch?: RetentionFetch;
    };

// The arguments /magic-compact and /magic-trim share, after parsing.
// `backend` and `keepThreshold` are the explicit `--backend` and
// `--threshold` values, absent when they were not given. `matchModel` is
// `--match-model`, which chooses summary models rather than retention.
export type RetentionArguments =
  | { keepTurns: number; heuristic: true; matchModel: boolean }
  | {
      keepTurns: number;
      heuristic: false;
      backend?: RetentionBackendID;
      keepThreshold?: number;
      truncate: boolean;
      matchModel: boolean;
    };

const PERCENT_TEXT = /^(\d+)(?:\.(\d+))?$/;

// Percent text, such as "60" or "70.5", to a 0..1 threshold, or undefined.
// The `e-2` suffix moves the decimal point in the text, so "60.1" becomes the
// double nearest 0.601 and not 60.1 / 100. The range is checked on the digits,
// before any rounding, so "100.00000000000000001" is above 100 and not 1; a
// nonzero percent too small for a double is undefined and never reads as 0.
// Signs, "%", exponents, hex, whitespace and other forms are undefined.
// Shared by RETENTION_KEEP_THRESHOLD_PERCENT_ENV and `--threshold`.
export function percentThreshold(text: string): number | undefined {
  const match = PERCENT_TEXT.exec(text);
  if (!match) return undefined;
  const [, whole, fraction = ""] = match;
  const wholePercent = Number(whole);
  if (wholePercent > 100 || (wholePercent === 100 && /[1-9]/.test(fraction))) {
    return undefined;
  }
  const threshold = Number(`${text}e-2`);
  return threshold === 0 && /[1-9]/.test(text) ? undefined : threshold;
}

// The server environment's percent, with surrounding whitespace ignored.
export function parseKeepThresholdPercent(text: string): number {
  const threshold = percentThreshold(text.trim());
  if (threshold === undefined) {
    throw new RetentionClassificationError("invalid-threshold-percent");
  }
  return threshold;
}

// The server environment's backend name, with surrounding whitespace
// ignored. Only the exact lowercase names are accepted.
export function parseRetentionBackend(text: string): RetentionBackendID {
  const name = text.trim();
  if (!isRetentionBackendID(name)) {
    throw new RetentionClassificationError("invalid-backend");
  }
  return name;
}

// The task-aware policy for one command invocation. The backend is the
// explicit `--backend` when given, else the environment's backend, else
// jev; the threshold is the explicit `--threshold`, else the environment's
// percent, else the default. An explicit value means its environment
// variable is never read, and an invalid or empty environment value fails
// instead of falling back. Backend and threshold are resolved before the
// key, and only the selected backend's key is read: a key for the other
// backend never selects it.
export function taskAwareRetentionPolicy(
  env: Readonly<Record<string, string | undefined>>,
  invocation?: RetentionInvocation,
): Extract<RetentionPolicy, { mode: "task-aware" }> {
  const chosen =
    invocation?.backend
    ?? withInvocation(invocation, () => environmentBackend(env));
  const backend = chosen ?? DEFAULT_RETENTION_BACKEND;
  // A retry suggestion names a backend the person or the server chose.
  const named =
    invocation && chosen !== undefined
      ? { ...invocation, backend: chosen }
      : invocation;
  const keepThreshold =
    invocation?.keepThreshold
    ?? withInvocation(named, () => environmentKeepThreshold(env));
  return {
    mode: "task-aware",
    backend,
    apiKey: env[RETENTION_BACKENDS[backend].keyEnv] || undefined,
    keepThreshold,
    ...(named ? { invocation: named } : {}),
  };
}

// The policy of /magic-compact or /magic-trim from its parsed arguments,
// shared by both commands on both hosts. `--heuristic` reads nothing from the
// environment.
export function commandRetentionPolicy(
  command: RetentionInvocation["command"],
  args: RetentionArguments,
  env: Readonly<Record<string, string | undefined>>,
): RetentionPolicy {
  if (args.heuristic) return { mode: "heuristic" };
  const { keepTurns, backend, keepThreshold, truncate, matchModel } = args;
  return taskAwareRetentionPolicy(env, {
    command,
    keepTurns,
    ...(backend === undefined ? {} : { backend }),
    ...(keepThreshold === undefined ? {} : { keepThreshold }),
    truncate,
    ...(matchModel ? { matchModel } : {}),
  });
}

function environmentBackend(
  env: Readonly<Record<string, string | undefined>>,
): RetentionBackendID | undefined {
  const name = env[RETENTION_BACKEND_ENV];
  return name === undefined ? undefined : parseRetentionBackend(name);
}

function environmentKeepThreshold(
  env: Readonly<Record<string, string | undefined>>,
): number {
  const percent = env[RETENTION_KEEP_THRESHOLD_PERCENT_ENV];
  return percent === undefined
    ? RETENTION_KEEP_THRESHOLD
    : parseKeepThresholdPercent(percent);
}

// Runs `read`; its failure suggests the person's own command.
function withInvocation<T>(
  invocation: RetentionInvocation | undefined,
  read: () => T,
): T {
  try {
    return read();
  } catch (error) {
    if (invocation && error instanceof RetentionClassificationError) {
      throw new RetentionClassificationError(
        error.kind,
        error.status,
        error.diagnostics,
        invocation,
      );
    }
    throw error;
  }
}

// What a command reports when it evaluated nothing: heuristic mode, or the
// task-aware mode with its effective threshold and no candidate to judge.
// Makes no request and reads no key.
export function retentionWithoutCandidates(
  policy: RetentionPolicy,
): RetentionPlan {
  return policy.mode === "heuristic"
    ? HEURISTIC_RETENTION
    : {
        mode: "task-aware",
        ...selectedBackend(policy),
        evaluated: 0,
        keep: new Set(),
        keepThreshold: resolveKeepThreshold(policy.keepThreshold),
        truncated: 0,
      };
}

function selectedBackend(
  policy: Extract<RetentionPolicy, { mode: "task-aware" }>,
): { backend: RetentionBackendID; model: string } {
  const backend = policy.backend ?? DEFAULT_RETENTION_BACKEND;
  return { backend, model: RETENTION_BACKENDS[backend].model };
}

export type RetentionPlan =
  | { mode: "heuristic"; evaluated: 0; keep: ReadonlySet<string> }
  | {
      mode: "task-aware";
      // The backend that judged the candidates, with its pinned model.
      backend: RetentionBackendID;
      model: string;
      evaluated: number;
      // Host tool-call references kept whole for this invocation.
      keep: ReadonlySet<string>;
      // The effective probability threshold.
      keepThreshold: number;
      // Decisions made from the --truncate summary excerpt.
      truncated: number;
    };

export const HEURISTIC_RETENTION: RetentionPlan = {
  mode: "heuristic",
  evaluated: 0,
  keep: new Set(),
};

// One completed tool call. `ref` is the host's unique reference for it.
export type RetentionCall = { ref: string; tool: string; view: ToolCallView };

// Every request turn of the source in order, with its ordinary user text.
export type RetentionTurn = { request: string; calls: RetentionCall[] };

export type RetentionCandidates = {
  latestUserRequest: string;
  candidates: RetentionCandidate[];
};

// Completed calls in `eligible` whose content the existing rules would
// change. Calls whose trim would only stamp metadata are not candidates.
// A request is one turn with ordinary user text plus the following turns
// whose user messages carry no ordinary text, matching the retention
// evaluation dataset.
export async function collectRetentionCandidates(
  turns: readonly RetentionTurn[],
  eligible: ReadonlySet<string>,
): Promise<RetentionCandidates> {
  let index = -1;
  let text = "";
  const located: Array<{ call: RetentionCall; text: string; index: number }> =
    [];
  for (const turn of turns) {
    if (turn.request) {
      index += 1;
      text = turn.request;
    }
    for (const call of turn.calls) located.push({ call, text, index });
  }
  const latest = index;
  const seen = new Set<string>();
  const candidates: RetentionCandidate[] = [];
  for (const { call, text: request, index: at } of located) {
    if (!eligible.has(call.ref)) continue;
    seen.add(call.ref);
    if (!(await toolTrimChangesContent(call.view))) continue;
    candidates.push({
      id: call.ref,
      tool: call.tool,
      input: call.view.input as Record<string, unknown>,
      output: call.view.output,
      turnRequest: request,
      requestsBeforeLatest: Math.max(0, latest - at),
    });
  }
  if (seen.size !== eligible.size) {
    throw new Error("Retention candidate is missing from source history.");
  }
  return { latestUserRequest: text, candidates };
}

// The whole-session summary retention judges against, and the reasoning
// report of the request that made it.
export type SessionSummary = { text: string; report: StageReport };

// What retention decided, with the reasoning report of each summary request
// it made: none without candidates, otherwise the one session summary.
export type RetentionDecision = {
  plan: RetentionPlan;
  reports: StageReport[];
};

// Summarizes the session and classifies every candidate. Any failure throws
// before the caller creates a successor, and no partial decision set is ever
// returned. Without candidates nothing is summarized and no key is read; a
// missing selected key fails before the summary is requested.
export async function decideRetentionWith(
  found: RetentionCandidates,
  policy: Extract<RetentionPolicy, { mode: "task-aware" }>,
  summarize: () => Promise<SessionSummary>,
): Promise<RetentionDecision> {
  if (found.candidates.length === 0) {
    return { plan: retentionWithoutCandidates(policy), reports: [] };
  }
  const keepThreshold = resolveKeepThreshold(policy.keepThreshold);
  const { invocation } = policy;
  const selected = selectedBackend(policy);
  try {
    if (!policy.apiKey) {
      throw new RetentionClassificationError("missing-api-key", undefined, {
        backend: selected.backend,
        model: selected.model,
      });
    }
    let summary: SessionSummary;
    try {
      summary = await summarize();
    } catch (error) {
      throw sessionSummaryFailure(error, invocation);
    }
    const { decisions, truncated } = await classifyRetention(
      {
        latestUserRequest: found.latestUserRequest,
        sessionSummary: summary.text,
      },
      found.candidates,
      {
        backend: selected.backend,
        apiKey: policy.apiKey,
        keepThreshold,
        ...(invocation?.truncate ? { truncateSummary: true } : {}),
        ...(policy.fetch ? { fetch: policy.fetch } : {}),
      },
    );
    const keep = new Set<string>();
    for (const candidate of found.candidates) {
      const decision = decisions.get(candidate.id);
      if (!decision) throw new RetentionClassificationError("invalid-response");
      if (decision.keep) keep.add(candidate.id);
    }
    return {
      plan: {
        mode: "task-aware",
        ...selected,
        evaluated: found.candidates.length,
        keep,
        keepThreshold,
        truncated,
      },
      reports: [summary.report],
    };
  } catch (error) {
    // A command's failure suggests repeating the person's own arguments.
    if (invocation && error instanceof RetentionClassificationError) {
      throw new RetentionClassificationError(
        error.kind,
        error.status,
        error.diagnostics,
        invocation,
      );
    }
    throw error;
  }
}

// `calls` names every tool call of the source by reference and tool name.
export function formatRetentionFeedback(
  plan: RetentionPlan,
  calls: Iterable<{ ref: string; tool: string }>,
): string {
  if (plan.mode === "heuristic") {
    return "Heuristic mode: tool calls not evaluated.";
  }
  const retention = `Task-aware retention with ${plan.backend} (${plan.model}) at keep threshold ${formatKeepThresholdPercent(plan.keepThreshold)}%`;
  if (plan.evaluated === 0) {
    return `${retention}: no tool calls needed evaluation.`;
  }
  const counts = new Map<string, number>();
  for (const call of calls) {
    if (plan.keep.has(call.ref)) {
      counts.set(call.tool, (counts.get(call.tool) ?? 0) + 1);
    }
  }
  const shown = [...counts]
    .sort(([a, x], [b, y]) => y - x || (a < b ? -1 : a > b ? 1 : 0))
    .slice(0, 3);
  const breakdown = shown.map(([tool, count]) => {
    // Tool names are display labels, never arguments or content. Strip control
    // and formatting characters and cap each label to keep feedback one line.
    const name = tool.replace(/[\p{C}\s]+/gu, " ").trim() || "unnamed";
    const codePoints = Array.from(name);
    return `${codePoints.length > 24 ? `${codePoints.slice(0, 21).join("")}...` : name} ${count}`;
  });
  const other =
    plan.keep.size - shown.reduce((sum, [, count]) => sum + count, 0);
  if (shown.length > 0 && other > 0) breakdown.push(`+${other} other calls`);
  const truncated =
    plan.truncated === 0
      ? ""
      : ` ${plan.truncated} decision(s) used a shortened session summary after a token-limit refusal (--truncate) and could have missed useful context.`;
  return `${retention} kept ${plan.keep.size} of ${plan.evaluated} evaluated tool call(s) whole${breakdown.length > 0 ? ` (${breakdown.join(", ")})` : ""}.${truncated}`;
}

// The summary a structured whole-session response submitted: exactly one
// nonempty `summary` string, or undefined for any other shape.
export function structuredSessionSummary(value: unknown): string | undefined {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return undefined;
  }
  const keys = Object.keys(value);
  const summary = (value as Record<string, unknown>)["summary"];
  if (keys.length !== 1 || typeof summary !== "string") return undefined;
  return summary.trim() || undefined;
}

export type SessionSummaryStage =
  | "budget"
  | "fork"
  | "permission"
  | "history"
  | "provider"
  | "validation"
  | "abort"
  | "delete";

// The fork (OpenCode 1) or temporary session (OpenCode 2) a summary request
// creates.
export type SummaryTemporary = "fork" | "session";

// One stage that failed. `overflow` marks a request that outgrew the model,
// and `detail` is a fixed fork-history reason. A `delete` stage names the
// fork or temporary session whose removal failed, or the OpenCode 2 session
// the host no longer found after an earlier stage failed.
export type SessionSummaryStageFailure =
  | {
      kind: "stage";
      stage: Exclude<SessionSummaryStage, "delete">;
      status?: number;
      overflow?: boolean;
      detail?: string;
    }
  | {
      kind: "stage";
      stage: "delete";
      status?: number;
      temporary: SummaryTemporary;
      session: UnremovedSession;
    };

// What went wrong while making the retention session summary, without the
// advice that closes its message.
export type SessionSummaryFailure =
  | SessionSummaryStageFailure
  // Several stages failed, in order: the request's own, if any, then the
  // cleanup of the fork or temporary session it created.
  | {
      kind: "cleanup";
      temporary: SummaryTemporary;
      failures: readonly SessionSummaryStageFailure[];
    }
  // A message the plugin wrote itself before the summary asked the host for
  // any session, such as a missing summarizer model.
  | { kind: "refusal"; message: string };

export const SESSION_SUMMARY_TOO_LARGE =
  "The session is too large for the configured compaction model to summarize for task-aware retention.";

export function sessionSummaryStageFailure(
  stage: Exclude<SessionSummaryStage, "delete">,
  facts: { status?: number; overflow?: boolean; detail?: string } = {},
): SessionSummaryError {
  return new SessionSummaryError({ kind: "stage", stage, ...facts });
}

// The stage a summary request reports for an error it caught: its own stage
// failure, or otherwise a validation failure.
export function caughtSummaryStage(error: unknown): SessionSummaryStageFailure {
  return error instanceof SessionSummaryError && error.failure.kind === "stage"
    ? error.failure
    : { kind: "stage", stage: "validation" };
}

// The one error for the stages a summary request and its cleanup failed, or
// undefined when none did.
export function failedSummaryStages(
  failures: readonly SessionSummaryStageFailure[],
  temporary: SummaryTemporary,
): SessionSummaryError | undefined {
  if (failures.length === 0) return undefined;
  return new SessionSummaryError(
    failures.length === 1
      ? failures[0]!
      : { kind: "cleanup", temporary, failures },
  );
}

// Every retention session summary failure. The message is fixed text with
// each stage's optional HTTP status and fork-history reason, then the fork
// or temporary session whose removal it could not confirm, then the advice
// for `invocation`: the person's own command, or /magic-trim for a caller
// without one. It never carries session content, model output or provider
// bodies.
export class SessionSummaryError
  extends PluginMessageError
  implements LeavesSessions
{
  readonly reason: string;
  readonly unremovedSessions: readonly UnremovedSession[];

  constructor(
    readonly failure: SessionSummaryFailure,
    invocation?: RetentionInvocation,
  ) {
    const reason = sessionSummaryFailureText(failure);
    const deletions = failedDeletions(failure);
    super(
      [
        reason,
        ...deletions.map(({ temporary, session }) =>
          unremovedSessionsText(
            [session],
            temporary === "fork" ? "Fork" : "Temporary session",
          ),
        ),
        retentionFailureGuidance(invocation),
      ].join(" "),
    );
    this.name = "SessionSummaryError";
    this.reason = reason;
    this.unremovedSessions = deletions.map(({ session }) => session);
  }
}

function failedDeletions(failure: SessionSummaryFailure) {
  const stages =
    failure.kind === "stage"
      ? [failure]
      : failure.kind === "cleanup"
        ? failure.failures
        : [];
  return stages.flatMap(stage => (stage.stage === "delete" ? [stage] : []));
}

// A request that outgrew the model is explained as such, even when its
// cleanup failed; the stage list then follows the explanation.
function sessionSummaryFailureText(failure: SessionSummaryFailure): string {
  if (failure.kind === "refusal") return failure.message;
  if (failure.kind === "stage") {
    return outgrewModel(failure)
      ? SESSION_SUMMARY_TOO_LARGE
      : `Task-aware retention session summary failed (${stageLabel(failure)}).`;
  }
  const stages = `Task-aware retention session summary and ${failure.temporary === "fork" ? "fork " : ""}cleanup failed (${failure.failures.map(stageLabel).join(", ")}).`;
  return failure.failures.some(outgrewModel)
    ? `${SESSION_SUMMARY_TOO_LARGE} ${stages}`
    : stages;
}

function outgrewModel(failure: SessionSummaryStageFailure): boolean {
  return (
    failure.stage === "budget"
    || (failure.stage !== "delete" && failure.overflow === true)
  );
}

function stageLabel(failure: SessionSummaryStageFailure): string {
  const status = failure.status === undefined ? "" : ` ${failure.status}`;
  const detail =
    failure.stage === "delete" || failure.detail === undefined
      ? ""
      : `: ${failure.detail}`;
  return `${failure.stage}${status}${detail}`;
}

// A failure while summarizing, as this invocation reports it: a new error
// whose message ends in this invocation's retention advice for every refusal
// the plugin wrote. Any other error is passed on unchanged and is reported as
// unexpected.
function sessionSummaryFailure(
  error: unknown,
  invocation: RetentionInvocation | undefined,
): unknown {
  if (error instanceof SessionSummaryError) {
    return new SessionSummaryError(error.failure, invocation);
  }
  if (!hasPluginMessage(error)) return error;
  return new SessionSummaryError(
    { kind: "refusal", message: error.message },
    invocation,
  );
}

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
  RETENTION_FAILURE_GUIDANCE,
  retentionFailureGuidance,
  RETENTION_KEEP_THRESHOLD,
  RETENTION_KEEP_THRESHOLD_PERCENT_ENV,
  RetentionClassificationError,
  type RetentionCandidate,
  type RetentionFetch,
  type RetentionInvocation,
} from "../retention/classifier";
import { hasPluginMessage, PluginMessageError } from "./feedback";
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

// Summarizes the session and classifies every candidate. Any failure throws
// before the caller creates a successor, and no partial decision set is ever
// returned.
export async function decideRetentionWith(
  found: RetentionCandidates,
  policy: Extract<RetentionPolicy, { mode: "task-aware" }>,
  summarize: () => Promise<string>,
): Promise<RetentionPlan> {
  if (found.candidates.length === 0) return retentionWithoutCandidates(policy);
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
    const sessionSummary = await summarize();
    const { decisions, truncated } = await classifyRetention(
      { latestUserRequest: found.latestUserRequest, sessionSummary },
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
      mode: "task-aware",
      ...selected,
      evaluated: found.candidates.length,
      keep,
      keepThreshold,
      truncated,
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
    if (invocation) addOwnCommandGuidance(error, invocation);
    throw error;
  }
}

// A session-summary failure ends in the guidance for a caller with no command.
// The error keeps its type, stage and any sessions it names; only the closing
// advice becomes the person's own command, as for a classification failure.
function addOwnCommandGuidance(
  error: unknown,
  invocation: RetentionInvocation,
): void {
  if (
    hasPluginMessage(error)
    && error.message.endsWith(RETENTION_FAILURE_GUIDANCE)
  ) {
    error.message = `${error.message.slice(0, -RETENTION_FAILURE_GUIDANCE.length)}${retentionFailureGuidance(invocation)}`;
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

// Messages are fixed text plus an optional HTTP status; they never carry
// session content, model output, or provider bodies.
export class SessionSummaryError extends PluginMessageError {
  constructor(
    readonly stage: SessionSummaryStage,
    readonly status?: number,
    overflow = false,
    detail?: string,
  ) {
    super(
      `${
        stage === "budget" || overflow
          ? "The session is too large for the configured compaction model to summarize for task-aware retention."
          : `Task-aware retention session summary failed (${stage}${status === undefined ? "" : ` ${status}`}${detail === undefined ? "" : `: ${detail}`}).`
      } ${RETENTION_FAILURE_GUIDANCE}`,
    );
    this.name = "SessionSummaryError";
  }
}

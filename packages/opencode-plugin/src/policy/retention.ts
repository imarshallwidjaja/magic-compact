// Task-aware retention for /magic-trim and /magic-compact, shared by both
// hosts. The existing tool rules still decide which calls would change; this
// step only vetoes changes to calls Jev judges likely to help the current
// task. Decisions are made per invocation and never persisted, so kept calls
// are reassessed next time.
import {
  classifyRetention,
  formatKeepThresholdPercent,
  resolveKeepThreshold,
  RETENTION_FAILURE_GUIDANCE,
  RETENTION_KEEP_THRESHOLD,
  RETENTION_KEEP_THRESHOLD_PERCENT_ENV,
  RetentionClassificationError,
  type RetentionCandidate,
  type RetentionFetch,
} from "../retention/classifier";
import type { MagicTrimArguments } from "../commands";
import { PluginMessageError } from "./feedback";
import { toolTrimChangesContent, type ToolCallView } from "./tool-rules";

export type RetentionPolicy =
  | { mode: "heuristic" }
  | {
      mode: "task-aware";
      // Supplied by the plugin runtime from TYPESAFE_API_KEY.
      apiKey: string | undefined;
      // Probability threshold from 0 to 1; defaults to RETENTION_KEEP_THRESHOLD.
      keepThreshold?: number;
      // Set only for /magic-trim: its arguments, for `--truncate` and for the
      // command a failure suggests.
      trim?: TaskAwareTrimArguments;
      fetch?: RetentionFetch;
    };

// The task-aware /magic-trim arguments. `keepThreshold` is the explicit
// `--threshold` probability, absent when it was not given.
export type TaskAwareTrimArguments = {
  keepTurns: number;
  keepThreshold?: number;
  truncate: boolean;
};

const PERCENT_TEXT = /^(\d+)(?:\.(\d+))?$/;

// Percent text, such as "60" or "70.5", to a 0..1 threshold, or undefined.
// The `e-2` suffix moves the decimal point in the text, so "60.1" becomes the
// double nearest 0.601 and not 60.1 / 100. The range is checked on the digits,
// before any rounding, so "100.00000000000000001" is above 100 and not 1; a
// nonzero percent too small for a double is undefined and never reads as 0.
// Signs, "%", exponents, hex, whitespace and other forms are undefined.
// Shared by RETENTION_KEEP_THRESHOLD_PERCENT_ENV and /magic-trim --threshold.
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

// The task-aware policy for one command invocation. The threshold is the
// explicit /magic-trim --threshold when given, and the environment's percent
// or the default is then never read; otherwise an invalid or empty percent
// fails instead of falling back to the default. The threshold is resolved
// before the key is read.
export function taskAwareRetentionPolicy(
  env: Readonly<Record<string, string | undefined>>,
  trim?: TaskAwareTrimArguments,
): Extract<RetentionPolicy, { mode: "task-aware" }> {
  const keepThreshold = trim?.keepThreshold ?? environmentKeepThreshold(env);
  return {
    mode: "task-aware",
    apiKey: env["TYPESAFE_API_KEY"] || undefined,
    keepThreshold,
    ...(trim ? { trim } : {}),
  };
}

// The /magic-trim policy from its parsed arguments, shared by both hosts.
// `--heuristic` reads nothing from the environment.
export function trimRetentionPolicy(
  args: MagicTrimArguments,
  env: Readonly<Record<string, string | undefined>>,
): RetentionPolicy {
  if (args.heuristic) return { mode: "heuristic" };
  const { keepTurns, keepThreshold, truncate } = args;
  return taskAwareRetentionPolicy(env, {
    keepTurns,
    ...(keepThreshold === undefined ? {} : { keepThreshold }),
    truncate,
  });
}

function environmentKeepThreshold(
  env: Readonly<Record<string, string | undefined>>,
): number {
  const percent = env[RETENTION_KEEP_THRESHOLD_PERCENT_ENV];
  return percent === undefined
    ? RETENTION_KEEP_THRESHOLD
    : parseKeepThresholdPercent(percent);
}

export type RetentionPlan =
  | { mode: "heuristic"; evaluated: 0; keep: ReadonlySet<string> }
  | {
      mode: "task-aware";
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
  const keepThreshold = resolveKeepThreshold(policy.keepThreshold);
  if (found.candidates.length === 0) {
    return {
      mode: "task-aware",
      evaluated: 0,
      keep: new Set(),
      keepThreshold,
      truncated: 0,
    };
  }
  const { trim } = policy;
  try {
    if (!policy.apiKey) {
      throw new RetentionClassificationError("missing-api-key");
    }
    const sessionSummary = await summarize();
    const { decisions, truncated } = await classifyRetention(
      { latestUserRequest: found.latestUserRequest, sessionSummary },
      found.candidates,
      {
        apiKey: policy.apiKey,
        keepThreshold,
        ...(trim?.truncate ? { truncateSummary: true } : {}),
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
      evaluated: found.candidates.length,
      keep,
      keepThreshold,
      truncated,
    };
  } catch (error) {
    // A /magic-trim failure suggests repeating the person's own arguments.
    if (trim && error instanceof RetentionClassificationError) {
      throw new RetentionClassificationError(
        error.kind,
        error.status,
        error.diagnostics,
        trim,
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
  const retention = `Task-aware retention at keep threshold ${formatKeepThresholdPercent(plan.keepThreshold)}%`;
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
      : ` ${plan.truncated} decision(s) used a shortened session summary after a Jev token-limit refusal (--truncate) and could have missed useful context.`;
  return `${retention} kept ${plan.keep.size} of ${plan.evaluated} evaluated tool call(s) whole${breakdown.length > 0 ? ` (${breakdown.join(", ")})` : ""}.${truncated}`;
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

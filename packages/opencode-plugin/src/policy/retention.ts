// Task-aware retention for /magic-trim and /magic-compact, shared by both
// hosts. The existing tool rules still decide which calls would change; this
// step only vetoes changes to calls Jev judges likely to help the current
// task. Decisions are made per invocation and never persisted, so kept calls
// are reassessed next time.
import {
  classifyRetention,
  resolveKeepThreshold,
  RETENTION_FAILURE_GUIDANCE,
  RETENTION_KEEP_THRESHOLD,
  RETENTION_KEEP_THRESHOLD_PERCENT_ENV,
  RetentionClassificationError,
  type RetentionCandidate,
  type RetentionFetch,
} from "../retention/classifier";
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
      fetch?: RetentionFetch;
    };

const PERCENT_TEXT = /^\d+(?:\.\d+)?$/;

// Percent text, such as "60" or "70.5", to a 0..1 threshold. The `e-2` suffix
// moves the decimal point in the text, so "60.1" becomes the double nearest
// 0.601 and not 60.1 / 100. Signs, "%", exponents, hex and other forms fail.
export function parseKeepThresholdPercent(text: string): number {
  const trimmed = text.trim();
  const threshold = PERCENT_TEXT.test(trimmed) ? Number(`${trimmed}e-2`) : NaN;
  if (!(threshold <= 1)) {
    throw new RetentionClassificationError("invalid-threshold-percent");
  }
  return threshold;
}

// The task-aware policy for one command invocation, from the server
// environment. The percent is validated before the key is read, and an
// invalid or empty value fails instead of falling back to the default.
export function taskAwareRetentionPolicy(
  env: Readonly<Record<string, string | undefined>>,
): Extract<RetentionPolicy, { mode: "task-aware" }> {
  const percent = env[RETENTION_KEEP_THRESHOLD_PERCENT_ENV];
  const keepThreshold =
    percent === undefined
      ? RETENTION_KEEP_THRESHOLD
      : parseKeepThresholdPercent(percent);
  return {
    mode: "task-aware",
    apiKey: env["TYPESAFE_API_KEY"] || undefined,
    keepThreshold,
  };
}

export type RetentionPlan = {
  mode: RetentionPolicy["mode"];
  evaluated: number;
  // Host tool-call references kept whole for this invocation.
  keep: ReadonlySet<string>;
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
    return { mode: "task-aware", evaluated: 0, keep: new Set() };
  }
  if (!policy.apiKey) {
    throw new RetentionClassificationError("missing-api-key");
  }
  const sessionSummary = await summarize();
  const { decisions } = await classifyRetention(
    { latestUserRequest: found.latestUserRequest, sessionSummary },
    found.candidates,
    {
      apiKey: policy.apiKey,
      keepThreshold,
      ...(policy.fetch ? { fetch: policy.fetch } : {}),
    },
  );
  const keep = new Set<string>();
  for (const candidate of found.candidates) {
    const decision = decisions.get(candidate.id);
    if (!decision) throw new RetentionClassificationError("invalid-response");
    if (decision.keep) keep.add(candidate.id);
  }
  return { mode: "task-aware", evaluated: found.candidates.length, keep };
}

// `calls` names every tool call of the source by reference and tool name.
export function formatRetentionFeedback(
  plan: RetentionPlan,
  calls: Iterable<{ ref: string; tool: string }>,
): string {
  if (plan.mode === "heuristic") {
    return "Heuristic mode: tool calls not evaluated.";
  }
  if (plan.evaluated === 0) {
    return "Task-aware retention: no tool calls needed evaluation.";
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
  return `Task-aware retention kept ${plan.keep.size} of ${plan.evaluated} evaluated tool call(s) whole${breakdown.length > 0 ? ` (${breakdown.join(", ")})` : ""}.`;
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

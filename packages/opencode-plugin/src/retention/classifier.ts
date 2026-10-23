// Task-aware tool retention classifier. Trimming rules pick the candidates;
// this module only asks the selected decision backend (TypeSafe Jev by
// default, or OpenAI Decisions) whether each candidate's content is useful for
// the current task, and a candidate is kept whole only when the probability
// clears the keep threshold. Evidence, question, scheduling and thresholds are
// shared; `backends.ts` owns each backend's wire format, limits and exact
// error recognition. See docs/RetentionEvaluation.md.
import { PluginMessageError } from "../policy/feedback";
import {
  DEFAULT_RETENTION_BACKEND,
  RETENTION_BACKEND_ENV,
  RETENTION_BACKENDS,
  RETENTION_MODEL,
  RETENTION_REQUEST_BODY_BYTE_LIMIT,
  type RetentionBackend,
  type RetentionBackendID,
} from "./backends";

export {
  RETENTION_ENDPOINT,
  RETENTION_MODEL,
  RETENTION_REQUEST_BODY_BYTE_LIMIT,
} from "./backends";

// The default keep threshold, selected by the operator after a development
// study of Jev. It is not a held-out or native-summary calibration, and it is
// not calibrated for OpenAI Decisions; the study and its limits are in
// docs/RetentionEvaluation.md. The server environment overrides it per
// command through RETENTION_KEEP_THRESHOLD_PERCENT_ENV, for either backend.
export const RETENTION_KEEP_THRESHOLD = 0.6;
export const RETENTION_KEEP_THRESHOLD_PERCENT_ENV =
  "MAGIC_COMPACT_KEEP_THRESHOLD_PERCENT";
export const RETENTION_MAX_CONCURRENCY = 4;
export const RETENTION_REQUEST_TIMEOUT_MS = 30_000;

// Only a 400 body this small is read, to recognize a backend's token-limit
// refusal.
const ERROR_BODY_BYTE_LIMIT = 4_096;
const REQUEST_ID = /^[A-Za-z0-9_-]{1,128}$/;

// Fixed per-field allocation for candidate content and the latest request.
// Content within its allocation is sent whole; larger content keeps its head
// and tail. Short input fields, such as paths, commands and patterns, always
// stay whole. The session summary has no allocation: it is sent whole.
// Only an opted-in retry after a token-limit refusal (`truncateSummary`)
// sends its head and tail within RETENTION_SUMMARY_EXCERPT_BYTES instead.
export const RETENTION_SUMMARY_EXCERPT_BYTES = 8_000;
const LATEST_REQUEST_BYTES = 5_000;
const TURN_REQUEST_BYTES = 600;
const INPUT_BYTES = 3_000;
const INPUT_FIELD_WHOLE_BYTES = 200;
const OUTPUT_BYTES = 9_000;

export type RetentionCandidate = {
  // Caller-owned identity, such as the tool part ID. Never sent to Jev.
  id: string;
  tool: string;
  input: Record<string, unknown>;
  output: string;
  // First user request line of the turn that made the call.
  turnRequest: string;
  // 0 when the call was made while handling the latest user request.
  requestsBeforeLatest: number;
};

export type RetentionContext = {
  latestUserRequest: string;
  // One fresh summary of the whole session, newest work included.
  sessionSummary: string;
};

export type RetentionFetch = (
  url: string,
  init: RequestInit,
) => Promise<Response>;

export type RetentionOptions = {
  // Defaults to DEFAULT_RETENTION_BACKEND. Every request and retry of one
  // invocation goes to this backend; nothing fails over to the other.
  backend?: RetentionBackendID;
  // The selected backend's own key.
  apiKey: string;
  // Evaluation-only, and Jev-only: compares batched formats against the
  // shipped one.
  batchSize?: number;
  timeoutMs?: number;
  fetch?: RetentionFetch;
  // A finite probability from 0 to 1; a candidate is kept at or above it.
  // Defaults to RETENTION_KEEP_THRESHOLD.
  keepThreshold?: number;
  // `--truncate` on /magic-compact or /magic-trim: a request the backend
  // refuses as over its token limit is sent once more, identical except that
  // the session summary is cut to its head and tail within
  // RETENTION_SUMMARY_EXCERPT_BYTES.
  // Only singleton requests (batchSize 1) support it.
  truncateSummary?: boolean;
};

export type RetentionDecision = {
  useful: number;
  keep: boolean;
};

// The command a person ran, as /magic-compact or /magic-trim, with the
// arguments that shape what a failure suggests running again: the turn count
// always, the keep threshold only when `--threshold` gave it explicitly, and
// the backend when `--backend` or the server environment chose it, so a
// suggested retry never silently selects the other provider.
export type RetentionInvocation = {
  command: "compact" | "trim";
  keepTurns: number;
  backend?: RetentionBackendID;
  keepThreshold?: number;
  truncate: boolean;
};

export type RetentionRequest = {
  body: {
    model: string;
    state: RetentionState;
    questions: Record<string, NoulQuestion>;
  };
  // Question key to candidate ID, in candidate order.
  candidates: Array<[string, string]>;
};

export type RetentionClassification = {
  decisions: Map<string, RetentionDecision>;
  // Logical requests, each answered once; one per candidate in production.
  requests: number;
  // HTTP attempts sent: every request once, plus each --truncate retry.
  attempts: number;
  // Requests answered by their --truncate retry, with the summary excerpt.
  truncated: number;
  // Sum of the input tokens the backend reported, over
  // `reportedUsageRequests` of the `attempts`. A response without a usable
  // count, including every refused attempt, adds nothing here and stays
  // unknown, not zero.
  inputTokens: number;
  reportedUsageRequests: number;
};

// Logical requests: planned = succeeded + failed + cancelled + unstarted,
// and started = planned - unstarted. `cancelled` requests were refused as
// over the token limit but their --truncate retry was never sent because
// the invocation had already failed. `attempts` counts HTTP requests sent,
// retries included.
export type RetentionRequestCounts = {
  planned: number;
  started: number;
  succeeded: number;
  failed: number;
  cancelled: number;
  unstarted: number;
  attempts: number;
};

// Whether a shorter summary exists for a --truncate retry, independent of
// the option: "unavailable" when the whole summary already fits the excerpt.
export type SummaryTruncation = "unavailable" | "available";

// Content-free facts about a failed invocation, for callers and tests. They
// never include request or response text, headers other than the validated
// request ID, the key, or an underlying error.
export type RetentionFailureDiagnostics = {
  // The backend the failed invocation used, with its pinned model.
  backend?: RetentionBackendID;
  model?: string;
  // The attempt whose failure is reported, by zero-based planned position
  // of its request and the summary it carried.
  request?: {
    ordinal: number;
    summary: "whole" | "excerpt";
    bodyBytes: number;
    status?: number;
    requestId?: string;
  };
  truncation?: SummaryTruncation;
  counts?: RetentionRequestCounts;
};

type RetentionState = {
  task: {
    latestUserRequest: string;
    sessionSummary: string;
  };
  candidates: Array<{
    tool: string;
    calledWhile: string;
    turnRequest: string;
    input: Record<string, string>;
    output: string;
  }>;
};

type NoulQuestion = {
  type: "noul";
  instructions: string;
  criteria: { true: string; false: string };
};

export type RetentionFailureKind =
  | "configuration"
  | "missing-api-key"
  | "invalid-threshold-percent"
  | "invalid-backend"
  // A Jev request body above RETENTION_REQUEST_BODY_BYTE_LIMIT, refused
  // locally.
  | "budget"
  // The backend refused a request as longer than the model's token limits.
  | "token-limit"
  | "timeout"
  | "network"
  | "http"
  | "invalid-response"
  // The answer named a model other than the pinned one.
  | "model-mismatch"
  // The backend declined to answer the question (OpenAI Decisions `refusal`).
  | "refusal";

// The closing advice of a failure before or around classification. Without an
// invocation it names /magic-trim without the caller's N or an explicit
// threshold.
export function retentionFailureGuidance(
  invocation?: RetentionInvocation,
): string {
  const compact = invocation?.command === "compact";
  const command = invocationCommand(invocation);
  return compact
    ? `No tool content was changed. Run ${command} --heuristic to compact without classification (turn summaries are still generated).`
    : `No tool content was changed. Run ${command} --heuristic to trim without classification.`;
}

export const RETENTION_FAILURE_GUIDANCE = retentionFailureGuidance();

// The command a person ran, with its turn count when not 0.
function invocationCommand(invocation: RetentionInvocation | undefined) {
  const turns =
    invocation && invocation.keepTurns > 0 ? ` ${invocation.keepTurns}` : "";
  return `/magic-${invocation?.command === "compact" ? "compact" : "trim"}${turns}`;
}

const UNCHANGED =
  "The original session is unchanged and no retention decisions were applied.";

// Public messages are fixed text plus an optional HTTP status and the
// validated numbers of `invocation`. They never carry request state, response
// bodies, credentials, or a rejected value. `invocation` is given only when a
// command ran, so its suggestions repeat the person's own command and
// arguments; without one they name /magic-trim without the caller's N or an
// explicit threshold.
export class RetentionClassificationError extends PluginMessageError {
  constructor(
    readonly kind: RetentionFailureKind,
    readonly status?: number,
    readonly diagnostics?: RetentionFailureDiagnostics,
    invocation?: RetentionInvocation,
  ) {
    super(retentionFailureText(kind, status, diagnostics, invocation));
    this.name = "RetentionClassificationError";
  }
}

// A keep threshold probability as the plain percent `percentThreshold` reads
// back to the same probability: 60.1 for 0.601, 0.000000000000000000001 for
// 1e-23. `String` gives the shortest digits that identify the probability, and
// the decimal point moves two places in that text, so no floating-point
// arithmetic can round the percent. The result has no exponent, so its length
// is bounded by the Number range (about 325 characters).
export function formatKeepThresholdPercent(threshold: number): string {
  const match = /^(\d+)(?:\.(\d+))?(?:e([+-]\d+))?$/.exec(String(threshold));
  if (!match) throw new Error("A keep threshold must be a finite probability.");
  const [, whole, fraction = "", exponent = "0"] = match;
  const digits = whole! + fraction;
  const point = whole!.length + Number(exponent) + 2;
  const percent =
    point <= 0
      ? `0.${"0".repeat(-point)}${digits}`
      : point >= digits.length
        ? digits + "0".repeat(point - digits.length)
        : `${digits.slice(0, point)}.${digits.slice(point)}`;
  return percent.replace(/^0+(?=\d)/, "");
}

// The failure toast shows at most 700 characters (`src/util.ts`).
const MESSAGE_CHARACTER_LIMIT = 700;

// `--heuristic` skips only the retention step. /magic-trim then makes no
// request at all, while /magic-compact still summarizes its turns with the
// model, so its suggestion says so.
function retentionFailureText(
  kind: RetentionFailureKind,
  status: number | undefined,
  diagnostics: RetentionFailureDiagnostics | undefined,
  invocation: RetentionInvocation | undefined,
): string {
  const compact = invocation?.command === "compact";
  const command = invocationCommand(invocation);
  const backend =
    diagnostics?.backend === undefined
      ? undefined
      : RETENTION_BACKENDS[diagnostics.backend];
  const heuristic = compact
    ? `Run ${command} --heuristic to compact without task-aware keep protection (turn summaries are still generated).`
    : `Run ${command} --heuristic to trim without task-aware keep protection.`;
  if (kind === "token-limit") {
    const name = (backend ?? RETENTION_BACKENDS[DEFAULT_RETENTION_BACKEND])
      .name;
    const refused = `${name} refused a task-aware retention request because it exceeds the model's token limit.`;
    if (diagnostics?.request?.summary === "excerpt") {
      return `${name} refused a task-aware retention request again after --truncate retried it with a shorter session summary, so the retry limit was exceeded. ${UNCHANGED} ${heuristic}`;
    }
    if (diagnostics?.truncation === "unavailable") {
      return `${refused} The session summary already fits the 8,000-byte excerpt that --truncate would use, so --truncate cannot shorten the request. ${UNCHANGED} ${heuristic}`;
    }
    // The retry repeats the person's turn count, backend and threshold, so
    // it asks the same backend at the same threshold.
    const retry = `${command}${invocation?.backend === undefined ? "" : ` --backend ${invocation.backend}`}${
      invocation?.keepThreshold === undefined
        ? ""
        : ` --threshold ${formatKeepThresholdPercent(invocation.keepThreshold)}`
    } --truncate`;
    const full = `${refused} ${UNCHANGED} Run ${retry} to retry each refused call once with a shorter session summary, which can omit useful context, or ${command} --heuristic to ${compact ? "compact" : "trim"} without task-aware keep protection${compact ? " (turn summaries are still generated)" : ""}. --threshold does not reduce the request size.`;
    // A threshold of hundreds of digits would push the --heuristic command
    // out of the toast, so the same guidance is then worded more briefly.
    return full.length < MESSAGE_CHARACTER_LIMIT
      ? full
      : `${name} refused a retention request as over its token limit; nothing was changed. Run ${retry} (a shorter summary can omit context) or ${command} --heuristic${compact ? " (turns are still summarized)" : ""}. --threshold does not shrink it.`;
  }
  if (kind === "budget") {
    return `A task-aware retention request is larger than TypeSafe's 32 MiB request body limit and was not sent; --truncate does not apply to this limit. ${UNCHANGED} ${heuristic}`;
  }
  return `${retentionFailureDetail(kind, status, backend)} ${retentionFailureGuidance(invocation)}`;
}

function retentionFailureDetail(
  kind: RetentionFailureKind,
  status: number | undefined,
  backend: RetentionBackend | undefined,
): string {
  if (kind === "missing-api-key") {
    const selected = backend ?? RETENTION_BACKENDS[DEFAULT_RETENTION_BACKEND];
    return `Task-aware retention with the ${selected.id} backend requires ${selected.keyEnv} in the OpenCode server environment.`;
  }
  if (kind === "invalid-threshold-percent") {
    return `Task-aware retention requires ${RETENTION_KEEP_THRESHOLD_PERCENT_ENV} in the OpenCode server environment to be a plain decimal percent from 0 to 100, such as 60 or 70.5, or to be unset.`;
  }
  if (kind === "invalid-backend") {
    return `Task-aware retention requires ${RETENTION_BACKEND_ENV} in the OpenCode server environment to be jev or openai-decisions, or to be unset; --backend jev or --backend openai-decisions on the command overrides it.`;
  }
  if (backend !== undefined) {
    if (kind === "http" && (status === 401 || status === 403)) {
      return `${backend.provider} refused the ${backend.keyEnv} credentials (http ${status}). Check the key in the OpenCode server environment.`;
    }
    if (
      kind === "http"
      && status !== undefined
      && status >= 300
      && status < 400
    ) {
      return `${backend.name} answered with a redirect (http ${status}), which is never followed, so the request went nowhere else.`;
    }
    if (kind === "model-mismatch") {
      return `${backend.name} answered with a model other than the pinned ${backend.model}, so its decisions were not used.`;
    }
    if (kind === "refusal") {
      return `${backend.name} declined to judge a tool call, so its decisions were not used.`;
    }
  }
  return `Task-aware retention classification${backend === undefined ? "" : ` with ${backend.id}`} failed (${kind}${status === undefined ? "" : ` ${status}`}).`;
}

// Resolves an explicit probability threshold before any request is built or
// sent. An invalid value is never clamped or replaced by the default.
export function resolveKeepThreshold(threshold: number | undefined): number {
  if (threshold === undefined) return RETENTION_KEEP_THRESHOLD;
  if (!Number.isFinite(threshold) || threshold < 0 || threshold > 1) {
    throw new RetentionClassificationError("configuration");
  }
  return threshold;
}

const QUESTION_CRITERIA = {
  true: "It holds concrete content the assistant would likely use again for the current task: results, findings, current file or artifact contents, specifications, instructions or procedures the remaining work still follows, identifiers, links, commands, errors, or decisions. A file that was partly edited later still counts when its unchanged content is useful.",
  false:
    "Its content is empty, an omission or status notice, or tool bookkeeping; or its content is superseded by later work, belongs to a finished or different task, or is only related to the current task by topic.",
};

function question(index: number): NoulQuestion {
  return {
    type: "noul",
    instructions: `Would the assistant likely use concrete content from the input or output of the tool call \`candidates[${index}]\` again while continuing the current task described by \`task.latestUserRequest\` and \`task.sessionSummary\`? Answer yes even when the same content also appears elsewhere or could be fetched again.`,
    criteria: QUESTION_CRITERIA,
  };
}

// Canonical requests. `bodyByteLimit` caps the canonical body, which is the
// Jev wire body; it defaults to TypeSafe's limit and is null for a
// backend that documents none, so that backend's request is never refused
// here for a limit that is not its own.
export function buildRetentionRequests(
  context: RetentionContext,
  candidates: RetentionCandidate[],
  batchSize = 1,
  bodyByteLimit: number | null = RETENTION_REQUEST_BODY_BYTE_LIMIT,
): RetentionRequest[] {
  if (!Number.isInteger(batchSize) || batchSize < 1) {
    throw new RetentionClassificationError("configuration");
  }
  const ids = new Set(candidates.map(candidate => candidate.id));
  if (ids.size !== candidates.length) {
    throw new RetentionClassificationError("configuration");
  }
  // Every request built here carries the same task, with the summary byte for
  // byte; only a --truncate retry swaps it for the excerpt.
  const task = {
    latestUserRequest: excerpt(context.latestUserRequest, LATEST_REQUEST_BYTES),
    sessionSummary: context.sessionSummary,
  };
  const requests: RetentionRequest[] = [];
  let pending: RetentionCandidate[] = [];
  const flush = () => {
    if (pending.length > 0) requests.push(buildRequest(task, pending));
    pending = [];
  };
  for (const candidate of candidates) {
    if (
      pending.length === batchSize
      || (pending.length > 0
        && !withinBodyLimit(
          buildRequest(task, [...pending, candidate]),
          bodyByteLimit,
        ))
    ) {
      flush();
    }
    pending.push(candidate);
    if (!withinBodyLimit(buildRequest(task, pending), bodyByteLimit)) {
      throw new RetentionClassificationError("budget");
    }
  }
  flush();
  return requests;
}

function buildRequest(
  task: RetentionState["task"],
  batch: RetentionCandidate[],
): RetentionRequest {
  const questions: Record<string, NoulQuestion> = {};
  const keys: Array<[string, string]> = [];
  batch.forEach((candidate, index) => {
    const key = `candidate_${index}`;
    questions[key] = question(index);
    keys.push([key, candidate.id]);
  });
  return {
    body: {
      model: RETENTION_MODEL,
      state: {
        task,
        candidates: batch.map(candidate => ({
          tool: candidate.tool,
          calledWhile: calledWhile(candidate.requestsBeforeLatest),
          turnRequest: excerpt(candidate.turnRequest, TURN_REQUEST_BYTES),
          input: excerptInput(candidate.input),
          output: excerpt(candidate.output, OUTPUT_BYTES),
        })),
      },
      questions,
    },
    candidates: keys,
  };
}

// The serialized text is measured and dropped; the send serializes again.
function withinBodyLimit(
  request: RetentionRequest,
  limit: number | null,
): boolean {
  return limit === null || jsonBytes(request.body) <= limit;
}

function calledWhile(requestsBeforeLatest: number): string {
  if (!Number.isInteger(requestsBeforeLatest) || requestsBeforeLatest < 0) {
    throw new RetentionClassificationError("configuration");
  }
  if (requestsBeforeLatest === 0) return "handling the latest user request";
  if (requestsBeforeLatest === 1) return "handling the previous user request";
  return `handling an earlier user request, ${requestsBeforeLatest} requests before the latest`;
}

function excerptInput(input: Record<string, unknown>): Record<string, string> {
  const fields = Object.entries(input).map(
    ([key, value]) =>
      [
        key,
        typeof value === "string" ? value : (JSON.stringify(value) ?? ""),
      ] as const,
  );
  const short = fields.filter(
    ([, value]) => jsonBytes(value) <= INPUT_FIELD_WHOLE_BYTES,
  );
  const long = fields.filter(
    ([, value]) => jsonBytes(value) > INPUT_FIELD_WHOLE_BYTES,
  );
  const shortBytes = short.reduce(
    (sum, [, value]) => sum + jsonBytes(value),
    0,
  );
  const perLong = Math.max(
    INPUT_FIELD_WHOLE_BYTES,
    Math.floor((INPUT_BYTES - shortBytes) / Math.max(1, long.length)),
  );
  const excerpted = new Map(
    long.map(([key, value]) => [key, excerpt(value, perLong)] as const),
  );
  const result = Object.fromEntries(
    fields.map(([key, value]) => [key, excerpted.get(key) ?? value]),
  );
  // Many short fields can still overflow; show them as one excerpted blob.
  return jsonBytes(result) <= INPUT_BYTES + INPUT_FIELD_WHOLE_BYTES
    ? result
    : { arguments: excerpt(JSON.stringify(input) ?? "", INPUT_BYTES) };
}

// Returns head and tail text whose JSON serialization fits `maxBytes`,
// joined by a marker naming how many characters were left out.
export function excerpt(text: string, maxBytes: number): string {
  if (jsonBytes(text) <= maxBytes) return text;
  const render = (keep: number) => {
    const head = Math.ceil((keep * 2) / 3);
    const tail = keep - head;
    const omitted = text.length - keep;
    return `${sliceChars(text, 0, head)}\n[... ${omitted} characters omitted ...]\n${tail > 0 ? sliceChars(text, text.length - tail, text.length) : ""}`;
  };
  let low = 0;
  let high = text.length;
  while (low < high) {
    const mid = Math.ceil((low + high) / 2);
    if (jsonBytes(render(mid)) <= maxBytes) low = mid;
    else high = mid - 1;
  }
  return render(low);
}

// Avoids splitting a surrogate pair at either edge of a slice.
function sliceChars(text: string, start: number, end: number): string {
  let from = start;
  let to = end;
  if (from > 0 && isLowSurrogate(text.charCodeAt(from))) from += 1;
  if (to < text.length && isLowSurrogate(text.charCodeAt(to))) to -= 1;
  return text.slice(from, Math.max(from, to));
}

function isLowSurrogate(code: number): boolean {
  return code >= 0xdc00 && code <= 0xdfff;
}

function jsonBytes(value: unknown): number {
  return Buffer.byteLength(JSON.stringify(value), "utf8");
}

export async function classifyRetention(
  context: RetentionContext,
  candidates: RetentionCandidate[],
  options: RetentionOptions,
): Promise<RetentionClassification> {
  const keepThreshold = resolveKeepThreshold(options.keepThreshold);
  const timeoutMs = options.timeoutMs ?? RETENTION_REQUEST_TIMEOUT_MS;
  const batchSize = options.batchSize ?? 1;
  const backend =
    RETENTION_BACKENDS[options.backend ?? DEFAULT_RETENTION_BACKEND];
  if (
    !options.apiKey
    || !(timeoutMs > 0)
    || (batchSize !== 1
      && (options.truncateSummary === true || backend.id !== "jev"))
  ) {
    throw new RetentionClassificationError("configuration");
  }
  const requests = buildRetentionRequests(
    context,
    candidates,
    batchSize,
    backend.bodyByteLimit,
  );
  // A summary that already fits the excerpt has no shorter retry. That fact
  // shapes the failure guidance with or without --truncate; the excerpt
  // itself is built once, and only when the retry is asked for and possible.
  const truncation: SummaryTruncation =
    jsonBytes(context.sessionSummary) <= RETENTION_SUMMARY_EXCERPT_BYTES
      ? "unavailable"
      : "available";
  const summaryExcerpt =
    options.truncateSummary && truncation === "available"
      ? excerpt(context.sessionSummary, RETENTION_SUMMARY_EXCERPT_BYTES)
      : undefined;
  const fetchImpl = options.fetch ?? fetch;
  const run = await runAttempts(
    requests.length,
    RETENTION_MAX_CONCURRENCY,
    summaryExcerpt !== undefined,
    (ordinal, summary, rejected) =>
      sendRequest(
        summary === "whole"
          ? requests[ordinal]!
          : withSummary(requests[ordinal]!, summaryExcerpt!),
        { ordinal, summary },
        backend,
        options.apiKey,
        timeoutMs,
        fetchImpl,
        rejected,
      ),
  );
  if ("failure" in run) {
    const { failure, counts } = run;
    if (!(failure instanceof RetentionClassificationError)) throw failure;
    throw new RetentionClassificationError(failure.kind, failure.status, {
      ...failure.diagnostics,
      truncation,
      counts,
    });
  }
  const decisions = new Map<string, RetentionDecision>();
  let inputTokens = 0;
  let reportedUsageRequests = 0;
  let truncated = 0;
  run.results.forEach(({ answers, summary }, index) => {
    if (answers.inputTokens !== undefined) {
      inputTokens += answers.inputTokens;
      reportedUsageRequests += 1;
    }
    const keys = requests[index]!.candidates;
    if (summary === "excerpt") truncated += keys.length;
    for (const [key, id] of keys) {
      const useful = answers.probabilities.get(key)!;
      decisions.set(id, { useful, keep: useful >= keepThreshold });
    }
  });
  return {
    decisions,
    requests: requests.length,
    attempts: run.attempts,
    truncated,
    inputTokens,
    reportedUsageRequests,
  };
}

// The same request with only `state.task.sessionSummary` replaced.
function withSummary(
  request: RetentionRequest,
  sessionSummary: string,
): RetentionRequest {
  const { body } = request;
  return {
    ...request,
    body: {
      ...body,
      state: { ...body.state, task: { ...body.state.task, sessionSummary } },
    },
  };
}

type SummaryVariant = "whole" | "excerpt";

type AttemptRun =
  | {
      results: Array<{ answers: RetentionAnswers; summary: SummaryVariant }>;
      attempts: number;
    }
  | { failure: unknown; counts: RetentionRequestCounts };

// Runs `count` requests with at most `limit` HTTP attempts in flight. An
// attempt calls `rejected` as soon as a non-OK status arrives, before any
// error body is read; its thrown error settles it.
//
// The first terminal failure stops admission for good. Attempts in flight
// drain, and then that failure is returned in place of every result, so no
// partial result is returned and a later failure never replaces it.
//
// With `retry`, a whole-summary attempt rejected with HTTP 400 is not yet a
// failure: it pauses admission while its body is classified. A confirmed
// token-limit refusal queues one excerpt attempt for the same request,
// ahead of unstarted requests; any other outcome is terminal. Admission
// resumes once no attempt holds a pause and is checked for every attempt
// immediately before it is sent, so a retry still queued when a terminal
// failure stops admission is never sent and counts as cancelled.
function runAttempts(
  count: number,
  limit: number,
  retry: boolean,
  send: (
    ordinal: number,
    summary: SummaryVariant,
    rejected: (status: number) => void,
  ) => Promise<RetentionAnswers>,
): Promise<AttemptRun> {
  const queue = Array.from({ length: count }, (_, ordinal) => ({
    ordinal,
    summary: "whole" as SummaryVariant,
  }));
  const results: Array<{ answers: RetentionAnswers; summary: SummaryVariant }> =
    new Array(count);
  const outcomes = new Array<
    "pending" | "succeeded" | "failed" | "cancelled" | undefined
  >(count);
  const errors = new Map<number, unknown>();
  let first: number | undefined;
  let active = 0;
  let paused = 0;
  let attempts = 0;
  return new Promise(resolve => {
    const finish = () => {
      if (first === undefined) {
        resolve({ results, attempts });
        return;
      }
      for (const item of queue) {
        if (item.summary === "excerpt") outcomes[item.ordinal] = "cancelled";
      }
      const tally = (outcome: (typeof outcomes)[number]) =>
        outcomes.filter(value => value === outcome).length;
      const unstarted = count - outcomes.filter(Boolean).length;
      resolve({
        failure: errors.has(first)
          ? errors.get(first)
          : new Error("A retention request halted admission without failing."),
        counts: {
          planned: count,
          started: count - unstarted,
          succeeded: tally("succeeded"),
          failed: tally("failed"),
          cancelled: tally("cancelled"),
          unstarted,
          attempts,
        },
      });
    };
    const pump = () => {
      while (
        first === undefined
        && paused === 0
        && active < limit
        && queue.length > 0
      ) {
        start(queue.shift()!);
      }
      if (active === 0 && (first !== undefined || queue.length === 0)) {
        finish();
      }
    };
    const start = ({
      ordinal,
      summary,
    }: {
      ordinal: number;
      summary: SummaryVariant;
    }) => {
      const id = attempts++;
      active += 1;
      outcomes[ordinal] = "pending";
      let pausing = false;
      const rejected = (status: number) => {
        if (retry && summary === "whole" && status === 400) {
          pausing = true;
          paused += 1;
        } else {
          first ??= id;
        }
      };
      send(ordinal, summary, rejected)
        .then(
          answers => {
            results[ordinal] = { answers, summary };
            outcomes[ordinal] = "succeeded";
          },
          (error: unknown) => {
            const recoverable =
              pausing
              && error instanceof RetentionClassificationError
              && error.kind === "token-limit";
            if (recoverable) {
              // Admission is checked again before it is sent.
              queue.unshift({ ordinal, summary: "excerpt" });
            } else {
              errors.set(id, error);
              first ??= id;
              outcomes[ordinal] = "failed";
            }
          },
        )
        .finally(() => {
          if (pausing) paused -= 1;
          active -= 1;
          pump();
        });
    };
    pump();
  });
}

type RetentionAnswers = {
  // Keyed by question key.
  probabilities: Map<string, number>;
  inputTokens: number | undefined;
};

async function sendRequest(
  request: RetentionRequest,
  attempt: { ordinal: number; summary: SummaryVariant },
  backend: RetentionBackend,
  apiKey: string,
  timeoutMs: number,
  fetchImpl: RetentionFetch,
  rejected: (status: number) => void,
): Promise<RetentionAnswers> {
  // Serialized once; the bytes measured are the bytes sent.
  const body = backend.serialize(request);
  const bodyBytes = Buffer.byteLength(body, "utf8");
  const facts: NonNullable<RetentionFailureDiagnostics["request"]> = {
    ...attempt,
    bodyBytes,
  };
  const fail = (kind: RetentionFailureKind, status?: number) =>
    new RetentionClassificationError(kind, status, {
      backend: backend.id,
      model: backend.model,
      request: facts,
    });
  if (backend.bodyByteLimit !== null && bodyBytes > backend.bodyByteLimit) {
    throw fail("budget");
  }
  const signal = AbortSignal.timeout(timeoutMs);
  let payload: unknown;
  try {
    const response = await fetchImpl(backend.endpoint, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body,
      // A redirect keeps the method, key and body, so following one would
      // send them to an address nobody chose. A 3xx is an HTTP failure.
      redirect: "manual",
      signal,
    });
    const status = httpStatus(response.status);
    if (status !== undefined) facts.status = status;
    const requestId = response.headers.get(backend.requestIdHeader);
    if (requestId !== null && REQUEST_ID.test(requestId)) {
      facts.requestId = requestId;
    }
    if (!response.ok) {
      rejected(response.status);
      if (response.status !== 400) {
        discardBody(response);
        throw fail("http", status);
      }
      const text = await readErrorBody(response, signal);
      throw fail(
        text !== undefined && backend.isTokenLimitBody(text)
          ? "token-limit"
          : "http",
        status,
      );
    }
    payload = await response.json();
  } catch (error) {
    if (error instanceof RetentionClassificationError) throw error;
    if (error instanceof DOMException && error.name === "TimeoutError") {
      throw fail("timeout");
    }
    if (error instanceof SyntaxError) throw fail("invalid-response");
    throw fail("network");
  }
  const answers = backend.parse(payload, request);
  if (!answers.ok) throw fail(answers.reason);
  // Usage is informational: a missing or unusable count stays unknown.
  const usage = isObject(payload) ? payload["usage"] : undefined;
  const reported = isObject(usage) ? usage["input_tokens"] : undefined;
  return {
    probabilities: answers.probabilities,
    inputTokens:
      Number.isSafeInteger(reported) && (reported as number) >= 0
        ? (reported as number)
        : undefined,
  };
}

// Cancels an unread body without waiting for the cancellation.
function discardBody(response: Response): void {
  try {
    response.body?.cancel().catch(() => {});
  } catch {
    // An already locked or released body has nothing more to discard.
  }
}

// Reads a 400 body of at most ERROR_BODY_BYTE_LIMIT bytes of UTF-8 before
// the request deadline. Any other outcome, including a body that is larger,
// malformed, failing or still open at the deadline, reads as undefined; an
// unfinished body is cancelled without waiting for the cancellation.
async function readErrorBody(
  response: Response,
  signal: AbortSignal,
): Promise<string | undefined> {
  const reader = (() => {
    try {
      return response.body?.getReader();
    } catch {
      return undefined;
    }
  })();
  if (!reader) return undefined;
  let onAbort = () => {};
  const deadline = new Promise<undefined>(resolve => {
    onAbort = () => resolve(undefined);
  });
  if (signal.aborted) onAbort();
  else signal.addEventListener("abort", onAbort, { once: true });
  const chunks: Uint8Array[] = [];
  let bytes = 0;
  let complete = false;
  try {
    for (;;) {
      const next = await Promise.race([reader.read(), deadline]);
      if (next === undefined) return undefined;
      if (next.done) break;
      if (!(next.value instanceof Uint8Array)) return undefined;
      bytes += next.value.byteLength;
      if (bytes > ERROR_BODY_BYTE_LIMIT) return undefined;
      chunks.push(next.value);
    }
    complete = true;
  } catch {
    return undefined;
  } finally {
    signal.removeEventListener("abort", onAbort);
    if (!complete) {
      try {
        reader.cancel().catch(() => {});
      } catch {
        // A released reader has nothing more to cancel.
      }
    }
  }
  try {
    return new TextDecoder("utf-8", { fatal: true }).decode(
      Buffer.concat(chunks),
    );
  } catch {
    return undefined;
  }
}

function httpStatus(status: number): number | undefined {
  return Number.isInteger(status) && status >= 100 && status <= 599
    ? status
    : undefined;
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

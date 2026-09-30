// Task-aware tool retention classifier. Trimming rules pick the candidates;
// this module only asks TypeSafe Jev whether each candidate's content is
// useful for the current task, and a candidate is kept whole only when the
// probability clears the keep threshold. See docs/RetentionEvaluation.md.

// The threshold was chosen against this exact model version, so the version
// is pinned rather than using the moving `jev-latest` alias.
export const RETENTION_MODEL = "jev-1.13.0";
export const RETENTION_ENDPOINT = "https://api.typesafe.ai/v1/systemone";
// The lowest of 0.80, 0.85 and 0.90 that met 80% precision on development
// sessions with native session summaries. Held-out results are recorded in
// docs/RetentionEvaluation.md.
export const RETENTION_KEEP_THRESHOLD = 0.8;
export const RETENTION_MAX_CONCURRENCY = 4;
export const RETENTION_REQUEST_TIMEOUT_MS = 30_000;

// Jev accepts 32K tokens for state plus the longest question and 64K for
// state plus all questions. Its tokenizer is not public, so budgets are in
// UTF-8 bytes of the serialized JSON. Bytes are not a documented token bound;
// measured usage is well under one token per byte, which is the headroom
// these limits rely on.
export const RETENTION_STATE_AND_QUESTION_BYTE_LIMIT = 28_000;
export const RETENTION_TOTAL_BYTE_LIMIT = 56_000;

// Fixed per-field allocation. Content within its allocation is sent whole;
// larger content keeps its head and tail. Short input fields, such as paths,
// commands and patterns, always stay whole. The largest possible request is
// about 27 KB, so every allocation fits at once.
const LATEST_REQUEST_BYTES = 5_000;
const SESSION_SUMMARY_BYTES = 8_000;
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
  apiKey: string;
  // Evaluation-only: compares batched formats against the shipped one.
  batchSize?: number;
  timeoutMs?: number;
  fetch?: RetentionFetch;
};

export type RetentionDecision = {
  useful: number;
  keep: boolean;
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
  requests: number;
  inputTokens: number;
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
  | "budget"
  | "timeout"
  | "network"
  | "http"
  | "invalid-response";

export const RETENTION_FAILURE_GUIDANCE =
  "No tool content was changed. Run /magic-trim --heuristic to trim without classification.";

// Public messages are fixed text plus an optional HTTP status. They never
// carry request state, response bodies, or credentials.
export class RetentionClassificationError extends Error {
  constructor(
    readonly kind: RetentionFailureKind,
    readonly status?: number,
  ) {
    super(
      `${kind === "missing-api-key" ? "Task-aware retention requires TYPESAFE_API_KEY in the OpenCode server environment." : `Task-aware retention classification failed (${kind}${status === undefined ? "" : ` ${status}`}).`} ${RETENTION_FAILURE_GUIDANCE}`,
    );
    this.name = "RetentionClassificationError";
  }
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

export function buildRetentionRequests(
  context: RetentionContext,
  candidates: RetentionCandidate[],
  batchSize = 1,
): RetentionRequest[] {
  if (!Number.isInteger(batchSize) || batchSize < 1) {
    throw new RetentionClassificationError("configuration");
  }
  const ids = new Set(candidates.map(candidate => candidate.id));
  if (ids.size !== candidates.length) {
    throw new RetentionClassificationError("configuration");
  }
  const task = {
    latestUserRequest: excerpt(context.latestUserRequest, LATEST_REQUEST_BYTES),
    sessionSummary: excerpt(context.sessionSummary, SESSION_SUMMARY_BYTES),
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
        && !fits(buildRequest(task, [...pending, candidate])))
    ) {
      flush();
    }
    pending.push(candidate);
    if (!fits(buildRequest(task, pending))) {
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

function fits(request: RetentionRequest): boolean {
  const state = jsonBytes(request.body.state);
  const questions = Object.values(request.body.questions).map(jsonBytes);
  const longest = Math.max(0, ...questions);
  const total = questions.reduce((sum, bytes) => sum + bytes, state);
  return (
    state + longest <= RETENTION_STATE_AND_QUESTION_BYTE_LIMIT
    && total <= RETENTION_TOTAL_BYTE_LIMIT
  );
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
  const timeoutMs = options.timeoutMs ?? RETENTION_REQUEST_TIMEOUT_MS;
  if (!options.apiKey || !(timeoutMs > 0)) {
    throw new RetentionClassificationError("configuration");
  }
  const requests = buildRetentionRequests(
    context,
    candidates,
    options.batchSize ?? 1,
  );
  const answers = await runBounded(
    requests,
    RETENTION_MAX_CONCURRENCY,
    request =>
      sendRequest(request, options.apiKey, timeoutMs, options.fetch ?? fetch),
  );
  const decisions = new Map<string, RetentionDecision>();
  let inputTokens = 0;
  answers.forEach((answer, index) => {
    inputTokens += answer.inputTokens;
    for (const [key, id] of requests[index]!.candidates) {
      const useful = answer.nouls.get(key)!;
      decisions.set(id, { useful, keep: useful >= RETENTION_KEEP_THRESHOLD });
    }
  });
  return { decisions, requests: requests.length, inputTokens };
}

// Runs every request with at most `limit` in flight. The first failure stops
// new admissions; in-flight requests drain before the first error is thrown,
// so no partial result is ever returned.
async function runBounded<T, R>(
  items: T[],
  limit: number,
  run: (item: T) => Promise<R>,
): Promise<R[]> {
  const results: R[] = new Array(items.length);
  let next = 0;
  let failure: { error: unknown } | undefined;
  const worker = async () => {
    while (!failure && next < items.length) {
      const index = next;
      next += 1;
      try {
        results[index] = await run(items[index]!);
      } catch (error) {
        failure ??= { error };
      }
    }
  };
  await Promise.all(
    Array.from({ length: Math.min(limit, items.length) }, worker),
  );
  if (failure) throw failure.error;
  return results;
}

async function sendRequest(
  request: RetentionRequest,
  apiKey: string,
  timeoutMs: number,
  fetchImpl: RetentionFetch,
): Promise<{ nouls: Map<string, number>; inputTokens: number }> {
  let response: Response;
  let payload: unknown;
  try {
    response = await fetchImpl(RETENTION_ENDPOINT, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request.body),
      signal: AbortSignal.timeout(timeoutMs),
    });
    if (!response.ok) {
      throw new RetentionClassificationError(
        "http",
        httpStatus(response.status),
      );
    }
    payload = await response.json();
  } catch (error) {
    if (error instanceof RetentionClassificationError) throw error;
    if (error instanceof DOMException && error.name === "TimeoutError") {
      throw new RetentionClassificationError("timeout");
    }
    if (error instanceof SyntaxError) {
      throw new RetentionClassificationError("invalid-response");
    }
    throw new RetentionClassificationError("network");
  }
  return parseAnswers(payload, request);
}

function httpStatus(status: number): number | undefined {
  return Number.isInteger(status) && status >= 100 && status <= 599
    ? status
    : undefined;
}

function parseAnswers(
  payload: unknown,
  request: RetentionRequest,
): { nouls: Map<string, number>; inputTokens: number } {
  const invalid = () => new RetentionClassificationError("invalid-response");
  if (!isObject(payload) || payload["model"] !== RETENTION_MODEL) {
    throw invalid();
  }
  const answers = payload["answers"];
  if (!isObject(answers)) throw invalid();
  const expected = request.candidates.map(([key]) => key);
  const received = Object.keys(answers);
  if (
    received.length !== expected.length
    || !expected.every(key => Object.hasOwn(answers, key))
  ) {
    throw invalid();
  }
  const nouls = new Map<string, number>();
  for (const key of expected) {
    const answer = answers[key];
    if (!isObject(answer) || answer["type"] !== "noul") throw invalid();
    const noul = answer["noul"];
    if (typeof noul !== "number" || !(noul >= 0 && noul <= 1)) {
      throw invalid();
    }
    nouls.set(key, noul);
  }
  const usage = payload["usage"];
  const inputTokens =
    isObject(usage)
    && typeof usage["input_tokens"] === "number"
    && Number.isFinite(usage["input_tokens"])
      ? usage["input_tokens"]
      : 0;
  return { nouls, inputTokens };
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

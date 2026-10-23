// The two task-aware retention backends, as a closed map. Both answer the same
// canonical request: the shared candidate evidence, question and criteria that
// `classifier.ts` builds. Each adapter owns only what differs between the
// providers: its endpoint, key variable, pinned model, request ID header and
// any documented body limit, how the canonical request goes on the wire, how
// an answer is read back strictly, and the exact token-limit refusal it
// recognizes. There is no registry, plugin loading or provider fallback.
//
// The evaluation harness (`eval/decisions/providers.ts`) imports the wire
// serializers and answer readers from here; nothing here imports evaluation
// code.
import type { RetentionRequest } from "./classifier";

export const RETENTION_BACKEND_IDS = ["jev", "openai-decisions"] as const;
export type RetentionBackendID = (typeof RETENTION_BACKEND_IDS)[number];
export const DEFAULT_RETENTION_BACKEND: RetentionBackendID = "jev";
// Read only when a command gives no `--backend`.
export const RETENTION_BACKEND_ENV = "MAGIC_COMPACT_RETENTION_BACKEND";

export function isRetentionBackendID(
  value: string,
): value is RetentionBackendID {
  return (RETENTION_BACKEND_IDS as readonly string[]).includes(value);
}

// The threshold was chosen against this exact model version, so the version
// is pinned rather than using the moving `jev-latest` alias.
export const RETENTION_MODEL = "jev-1.13.0";
export const RETENTION_ENDPOINT = "https://api.typesafe.ai/v1/systemone";
// TypeSafe refuses a request body above 32 MiB, counting every byte of the
// JSON (model, state and questions). It is the only local size check, and it
// applies to Jev alone: Jev's token limits (32K tokens for state plus one
// question, 64K for state plus all questions) are enforced by the service,
// because its tokenizer is not public and bytes are not a token bound.
export const RETENTION_REQUEST_BODY_BYTE_LIMIT = 32 * 1024 * 1024;

export const OPENAI_DECISIONS_ENDPOINT = "https://api.openai.com/v1/decisions";
// OpenAI documents gpt-6-luna as the only Decisions model.
export const OPENAI_DECISIONS_MODEL = "gpt-6-luna";
// OpenAI's refusal of a request over the model's input token limit, as
// observed from gpt-6-luna on 2026-10-23: HTTP 400 with this 119-byte body.
export const OPENAI_TOKEN_LIMIT_MESSAGE =
  "Decision input exceeds the token limit.";

// What a backend's answer reader concluded; the probabilities are keyed by
// the asked name.
export type BackendAnswers =
  | { ok: true; probabilities: Map<string, number> }
  | { ok: false; reason: "invalid-response" | "model-mismatch" | "refusal" };

export type RetentionBackend = {
  id: RetentionBackendID;
  // Display name in feedback and failure messages.
  name: string;
  // Who issues the key, for credential failures.
  provider: string;
  endpoint: string;
  // The only environment variable read for this backend.
  keyEnv: string;
  model: string;
  requestIdHeader: string;
  // A cap on the exact wire body, only where the provider documents one.
  bodyByteLimit: number | null;
  serialize(request: RetentionRequest): string;
  parse(payload: unknown, request: RetentionRequest): BackendAnswers;
  isTokenLimitBody(text: string): boolean;
};

const questionKeys = (request: RetentionRequest) =>
  request.candidates.map(([key]) => key);

export const RETENTION_BACKENDS: Readonly<
  Record<RetentionBackendID, RetentionBackend>
> = {
  jev: {
    id: "jev",
    name: "TypeSafe Jev",
    provider: "TypeSafe",
    endpoint: RETENTION_ENDPOINT,
    keyEnv: "TYPESAFE_API_KEY",
    model: RETENTION_MODEL,
    requestIdHeader: "x-typesafe-request-id",
    bodyByteLimit: RETENTION_REQUEST_BODY_BYTE_LIMIT,
    // The canonical body is Jev's own format.
    serialize: request => JSON.stringify(request.body),
    parse: (payload, request) =>
      parseJevAnswers(payload, questionKeys(request), RETENTION_MODEL),
    isTokenLimitBody: isJevTokenLimitBody,
  },
  "openai-decisions": {
    id: "openai-decisions",
    name: "OpenAI Decisions",
    provider: "OpenAI",
    endpoint: OPENAI_DECISIONS_ENDPOINT,
    keyEnv: "MAGIC_COMPACT_OPENAI_DECISIONS_API_KEY",
    model: OPENAI_DECISIONS_MODEL,
    requestIdHeader: "x-request-id",
    // OpenAI documents no request body cap; the service enforces its limits.
    bodyByteLimit: null,
    // The predicate is named by its opaque question key, never by a host ID.
    serialize: request =>
      openAIDecisionsBody(
        request,
        OPENAI_DECISIONS_MODEL,
        request.candidates[0]![0],
      ),
    parse: (payload, request) =>
      parseOpenAIDecisionAnswers(
        payload,
        request.candidates[0]![0],
        OPENAI_DECISIONS_MODEL,
      ),
    isTokenLimitBody: isOpenAITokenLimitBody,
  },
};

type Question = RetentionRequest["body"]["questions"][string];

// A predicate has no separate criteria field, so the canonical question and
// both of its criteria go into the instructions in a fixed order.
export function renderPredicateInstructions(question: Question): string {
  return `${question.instructions}\n\nYes, probability near 1: ${question.criteria.true}\nNo, probability near 0: ${question.criteria.false}`;
}

// The canonical singleton request as an OpenAI Decisions body: the state as
// the JSON string `input` and its question as one predicate called `name`.
// Production names it by the question key; the evaluation by its case ID.
export function openAIDecisionsBody(
  request: RetentionRequest,
  model: string,
  name: string,
): string {
  if (request.candidates.length !== 1) {
    throw new Error(
      "An OpenAI Decisions request carries exactly one question.",
    );
  }
  const question = request.body.questions[request.candidates[0]![0]]!;
  return JSON.stringify({
    model,
    input: JSON.stringify(request.body.state),
    questions: [
      {
        type: "predicate",
        name,
        instructions: renderPredicateInstructions(question),
      },
    ],
  });
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function validProbability(value: unknown): value is number {
  return typeof value === "number" && value >= 0 && value <= 1;
}

type AnswerFailure = "invalid-response" | "model-mismatch";

// The response model must equal the requested model exactly: no dated
// snapshot label or variant is normalized. The result is either the response
// object or a failure string, which a response object can never be, so a
// field the provider put in the body (`ok`, `reason`) is never read as an
// outcome. The readers take only `answers` from it.
function answeredBy(
  payload: unknown,
  model: string,
): Record<string, unknown> | AnswerFailure {
  if (!isObject(payload) || typeof payload["model"] !== "string") {
    return "invalid-response";
  }
  return payload["model"] === model ? payload : "model-mismatch";
}

const invalid: BackendAnswers = { ok: false, reason: "invalid-response" };

// `{"model", "answers": {<name>: {"type": "noul", "noul": p}}}` with exactly
// the asked names.
export function parseJevAnswers(
  payload: unknown,
  names: readonly string[],
  model: string,
): BackendAnswers {
  const body = answeredBy(payload, model);
  if (typeof body === "string") return { ok: false, reason: body };
  const answers = body["answers"];
  if (
    !isObject(answers)
    || Object.keys(answers).length !== names.length
    || !names.every(name => Object.hasOwn(answers, name))
  ) {
    return invalid;
  }
  const probabilities = new Map<string, number>();
  for (const name of names) {
    const answer = answers[name];
    if (
      !isObject(answer)
      || answer["type"] !== "noul"
      || !validProbability(answer["noul"])
    ) {
      return invalid;
    }
    probabilities.set(name, answer["noul"]);
  }
  return { ok: true, probabilities };
}

// `{"model", "answers": [{"type": "predicate", "name", "probability"}]}`: one
// answer, for the one predicate asked. A `refusal` answer is its own outcome,
// never a probability.
export function parseOpenAIDecisionAnswers(
  payload: unknown,
  name: string,
  model: string,
): BackendAnswers {
  const body = answeredBy(payload, model);
  if (typeof body === "string") return { ok: false, reason: body };
  const answers = body["answers"];
  if (!Array.isArray(answers) || answers.length !== 1) return invalid;
  const answer: unknown = answers[0];
  if (!isObject(answer) || answer["name"] !== name) return invalid;
  if (answer["type"] === "refusal") return { ok: false, reason: "refusal" };
  if (
    answer["type"] !== "predicate"
    || !validProbability(answer["probability"])
  ) {
    return invalid;
  }
  return { ok: true, probabilities: new Map([[name, answer["probability"]]]) };
}

function parseObject(text: string): Record<string, unknown> | undefined {
  try {
    const parsed: unknown = JSON.parse(text);
    return isObject(parsed) ? parsed : undefined;
  } catch {
    return undefined;
  }
}

// Exactly `{"detail": {"error_type": "max_tokens_exceeded", ...}}`. Other
// keys are ignored and never shown; arrays and matching text elsewhere do
// not count.
function isJevTokenLimitBody(text: string): boolean {
  const detail = parseObject(text)?.["detail"];
  return isObject(detail) && detail["error_type"] === "max_tokens_exceeded";
}

// Exactly the root `error` object OpenAI returned for an over-limit input:
// this message and type, with `param` and `code` present and null. Other
// keys are ignored and never shown; no substring, other code or other
// endpoint's error shape counts.
function isOpenAITokenLimitBody(text: string): boolean {
  const error = parseObject(text)?.["error"];
  return (
    isObject(error)
    && error["message"] === OPENAI_TOKEN_LIMIT_MESSAGE
    && error["type"] === "invalid_request_error"
    && Object.hasOwn(error, "param")
    && error["param"] === null
    && Object.hasOwn(error, "code")
    && error["code"] === null
  );
}

// Decision providers. Each adapter turns the production singleton request into
// its own wire body and strictly reads the answer back. Adding a provider is
// one more adapter in PROVIDERS; there is no registry or plugin loading.
//
// Failures are reduced to a fixed category and optional HTTP status. Request
// text, response bodies, error messages and keys never reach a result.
import {
  RETENTION_ENDPOINT,
  RETENTION_REQUEST_TIMEOUT_MS,
  type RetentionFetch,
  type RetentionRequest,
} from "../../src/retention/classifier";
import {
  canonicalJson,
  DecisionEvalError,
  semanticQuestion,
  sha256,
} from "./dataset";

export const PROVIDER_IDS = ["jev", "openai-decisions"] as const;
export type ProviderID = (typeof PROVIDER_IDS)[number];

export const DEFAULT_TIMEOUT_MS = RETENTION_REQUEST_TIMEOUT_MS;
export const OPENAI_DECISIONS_ENDPOINT = "https://api.openai.com/v1/decisions";
// OpenAI documents gpt-6-luna as the only Decisions model. Selecting another
// is refused up front rather than sent to be rejected one case at a time.
export const OPENAI_DECISIONS_MODELS = ["gpt-6-luna"] as const;
export const OPENAI_TRANSLATION_VERSION = "openai-predicate-1";

export type Usage = { inputTokens?: number; outputTokens?: number };

export type FailureCategory =
  | "auth"
  | "rate-limit"
  | "http"
  | "timeout"
  | "network"
  | "invalid-response"
  | "model-mismatch";

// A rejected key or an exhausted rate limit affects every remaining request,
// so the runner stops admitting new ones after either.
export const FATAL_CATEGORIES: ReadonlySet<FailureCategory> = new Set([
  "auth",
  "rate-limit",
]);

export type Transport =
  | {
      status: "scored";
      probability: number;
      requestedModel: string;
      returnedModel: string;
      usage?: Usage;
      latencyMs: number;
    }
  | {
      status: "refused";
      reason: "provider-refusal";
      requestedModel: string;
      returnedModel: string;
      usage?: Usage;
      latencyMs: number;
    }
  | {
      status: "failed";
      category: FailureCategory;
      httpStatus?: number;
      latencyMs: number;
    };

type Interpreted =
  | {
      kind: "scored";
      probability: number;
      returnedModel: string;
      usage?: Usage;
    }
  | { kind: "refused"; returnedModel: string; usage?: Usage };

export type ProviderAdapter = {
  id: ProviderID;
  endpoint: string;
  // The only environment variable the CLI reads for this provider.
  keyEnv: string;
  // Null when the wire body is the production body with only `model` replaced.
  translation(): { version: string; sha256: string } | null;
  checkModel(model: string): void;
  translate(request: RetentionRequest, model: string): string;
  interpret(
    payload: unknown,
    request: RetentionRequest,
    model: string,
  ): Interpreted;
};

class ResponseError extends Error {
  constructor(readonly category: FailureCategory) {
    super(category);
  }
}

const invalid = () => new ResponseError("invalid-response");

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function validProbability(value: unknown): value is number {
  return typeof value === "number" && value >= 0 && value <= 1;
}

// A count outside the safe integer range cannot be stored or summed exactly,
// so that dimension counts as not reported.
function reportedCount(value: unknown): number | undefined {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0
    ? value
    : undefined;
}

// Each provider's documented answer model is the ID that was requested. A
// dated snapshot label or another model is a mismatch until a first-party
// contract says how to normalize it. Result loading applies the same rule.
export function returnedModelMatches(
  requested: string,
  returned: string,
): boolean {
  return returned === requested;
}

function readUsage(payload: Record<string, unknown>): Usage | undefined {
  const raw = payload["usage"];
  if (!isObject(raw)) return undefined;
  const inputTokens = reportedCount(raw["input_tokens"]);
  const outputTokens = reportedCount(raw["output_tokens"]);
  if (inputTokens === undefined && outputTokens === undefined) {
    return undefined;
  }
  return {
    ...(inputTokens === undefined ? {} : { inputTokens }),
    ...(outputTokens === undefined ? {} : { outputTokens }),
  };
}

function readModel(payload: unknown): Record<string, unknown> & {
  model: string;
} {
  if (!isObject(payload) || typeof payload["model"] !== "string") {
    throw invalid();
  }
  return payload as Record<string, unknown> & { model: string };
}

type Question = RetentionRequest["body"]["questions"][string];

// One predicate has no separate criteria field, so the production question and
// both of its criteria go into the instructions in a fixed order.
export function renderPredicateInstructions(question: Question): string {
  return `${question.instructions}\n\nYes, probability near 1: ${question.criteria.true}\nNo, probability near 0: ${question.criteria.false}`;
}

const JEV_VERSION = /^jev-\d+\.\d+\.\d+$/;

const jev: ProviderAdapter = {
  id: "jev",
  endpoint: RETENTION_ENDPOINT,
  keyEnv: "TYPESAFE_API_KEY",
  translation: () => null,
  checkModel(model) {
    if (!JEV_VERSION.test(model)) {
      throw new DecisionEvalError(
        "Jev models must be versioned IDs such as jev-1.13.0. Aliases such as jev-latest move between releases, so results would not be repeatable.",
      );
    }
  },
  // Replacing the value keeps the key's position, so the bytes equal the
  // production body whenever the model is the production model.
  translate: (request, model) => JSON.stringify({ ...request.body, model }),
  interpret(payload, request, model) {
    const body = readModel(payload);
    if (!returnedModelMatches(model, body.model)) {
      throw new ResponseError("model-mismatch");
    }
    const answers = body["answers"];
    const key = request.candidates[0]![0];
    if (
      !isObject(answers)
      || Object.keys(answers).length !== 1
      || !Object.hasOwn(answers, key)
    ) {
      throw invalid();
    }
    const answer = answers[key];
    if (
      !isObject(answer)
      || answer["type"] !== "noul"
      || !validProbability(answer["noul"])
    ) {
      throw invalid();
    }
    const usage = readUsage(body);
    return {
      kind: "scored",
      probability: answer["noul"],
      returnedModel: body.model,
      ...(usage ? { usage } : {}),
    };
  },
};

const openaiDecisions: ProviderAdapter = {
  id: "openai-decisions",
  endpoint: OPENAI_DECISIONS_ENDPOINT,
  keyEnv: "OPENAI_API_KEY",
  translation: () => ({
    version: OPENAI_TRANSLATION_VERSION,
    sha256: sha256(
      canonicalJson({
        version: OPENAI_TRANSLATION_VERSION,
        instructions: renderPredicateInstructions(semanticQuestion()),
      }),
    ),
  }),
  checkModel(model) {
    if (!(OPENAI_DECISIONS_MODELS as readonly string[]).includes(model)) {
      throw new DecisionEvalError(
        `OpenAI Decisions currently supports only ${OPENAI_DECISIONS_MODELS.join(", ")}; ${model} is not available.`,
      );
    }
  },
  translate(request, model) {
    const question = request.body.questions[request.candidates[0]![0]]!;
    return JSON.stringify({
      model,
      input: JSON.stringify(request.body.state),
      questions: [
        {
          type: "predicate",
          name: request.candidates[0]![1],
          instructions: renderPredicateInstructions(question),
        },
      ],
    });
  },
  interpret(payload, request, model) {
    const body = readModel(payload);
    if (!returnedModelMatches(model, body.model)) {
      throw new ResponseError("model-mismatch");
    }
    const answers = body["answers"];
    if (!Array.isArray(answers) || answers.length !== 1) throw invalid();
    const answer: unknown = answers[0];
    if (!isObject(answer) || answer["name"] !== request.candidates[0]![1]) {
      throw invalid();
    }
    const usage = readUsage(body);
    const extra = {
      returnedModel: body.model,
      ...(usage ? { usage } : {}),
    };
    if (answer["type"] === "refusal") return { kind: "refused", ...extra };
    if (
      answer["type"] !== "predicate"
      || !validProbability(answer["probability"])
    ) {
      throw invalid();
    }
    return { kind: "scored", probability: answer["probability"], ...extra };
  },
};

export const PROVIDERS: Readonly<Record<ProviderID, ProviderAdapter>> = {
  jev,
  "openai-decisions": openaiDecisions,
};

export function isProviderID(value: string): value is ProviderID {
  return (PROVIDER_IDS as readonly string[]).includes(value);
}

export type SendOptions = {
  model: string;
  apiKey: string;
  endpoint: string;
  timeoutMs: number;
  fetch: RetentionFetch;
};

function httpCategory(status: number): FailureCategory {
  if (status === 401 || status === 403) return "auth";
  if (status === 429) return "rate-limit";
  return "http";
}

// One request, no retry. The caller owns admission and bookkeeping.
export async function sendDecision(
  adapter: ProviderAdapter,
  body: string,
  request: RetentionRequest,
  options: SendOptions,
): Promise<Transport> {
  const started = performance.now();
  const latencyMs = () => Math.round(performance.now() - started);
  const failed = (
    category: FailureCategory,
    httpStatus?: number,
  ): Transport => ({
    status: "failed",
    category,
    ...(httpStatus === undefined ? {} : { httpStatus }),
    latencyMs: latencyMs(),
  });
  let payload: unknown;
  try {
    const response = await options.fetch(options.endpoint, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${options.apiKey}`,
        "Content-Type": "application/json",
      },
      body,
      // A redirect keeps the method and body, so following one would send the
      // case text to an origin nobody chose. It is recorded as an HTTP failure.
      redirect: "manual",
      signal: AbortSignal.timeout(options.timeoutMs),
    });
    if (!response.ok) {
      const status =
        Number.isInteger(response.status)
        && response.status >= 100
        && response.status <= 599
          ? response.status
          : undefined;
      return failed(httpCategory(response.status), status);
    }
    payload = await response.json();
  } catch (error) {
    if ((error as { name?: unknown } | null)?.name === "TimeoutError") {
      return failed("timeout");
    }
    return failed(
      error instanceof SyntaxError ? "invalid-response" : "network",
    );
  }
  try {
    const result = adapter.interpret(payload, request, options.model);
    const common = {
      requestedModel: options.model,
      returnedModel: result.returnedModel,
      ...(result.usage ? { usage: result.usage } : {}),
      latencyMs: latencyMs(),
    };
    return result.kind === "scored"
      ? { status: "scored", probability: result.probability, ...common }
      : { status: "refused", reason: "provider-refusal", ...common };
  } catch (error) {
    if (error instanceof ResponseError) return failed(error.category);
    throw error;
  }
}

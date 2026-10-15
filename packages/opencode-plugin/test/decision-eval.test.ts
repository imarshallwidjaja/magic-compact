import { afterAll, describe, expect, test } from "bun:test";
import {
  copyFileSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  buildCaseRequest,
  canonicalJson,
  datasetCases,
  datasetSha256,
  DecisionEvalError,
  isProductionTrimCandidate,
  loadDataset,
  parseDataset,
  requestSetSha256,
  canonicalRequestSha256,
  semanticQuestion,
  semanticQuestionSha256,
  sha256,
  type DecisionDataset,
} from "../eval/decisions/dataset";
import {
  OPENAI_DECISIONS_MODELS,
  PROVIDERS,
  renderPredicateInstructions,
  sendDecision,
  type ProviderAdapter,
  type Transport,
} from "../eval/decisions/providers";
import { computeMetrics, renderReport } from "../eval/decisions/report";
import { main, type CliDeps } from "../eval/decisions/run";
import {
  executeRun,
  loadRun,
  planRun,
  type LoadedRun,
  type Outcome,
  type RunOptions,
} from "../eval/decisions/runner";
import {
  classifyRetention,
  RETENTION_KEEP_THRESHOLD,
  RETENTION_MAX_CONCURRENCY,
  RETENTION_MODEL,
  RETENTION_REQUEST_TIMEOUT_MS,
  RETENTION_STATE_AND_QUESTION_BYTE_LIMIT,
  RETENTION_TOTAL_BYTE_LIMIT,
  type RetentionFetch,
} from "../src/retention/classifier";

// Frozen identity of the bundled pack and of the production request it maps
// to. A change to either is deliberate: bump the dataset version, or follow
// docs/RetentionEvaluation.md before touching the production question.
const FROZEN = {
  datasetSha256:
    "859d9cb4e15b68b2fa52af9c3650541a2c54c096f05bcbaa0bdadb886adc9973",
  requestSetSha256:
    "44a626f4e44a95246e79c99b5c2a6ae845d50d67b6cbd17fb44c632204bb5239",
  questionSha256:
    "1ae66735b0866acd3602c4de2905b9b2d491eedb62d9c8d401102579d2ccae00",
  openaiTranslationSha256:
    "7e9c32d00be33020214af32ca04ff2a89544425e7d5d9b10688992c3138aac8f",
};

const KEY = "sk-canary-7f3a9c1e-never-persist";
const dataset = loadDataset();
const cases = datasetCases(dataset);
const scratch = mkdtempSync(join(tmpdir(), "decision-eval-"));
let counter = 0;
const freshDir = () => join(scratch, `run-${(counter += 1)}`);
afterAll(() => rmSync(scratch, { recursive: true, force: true }));

function options(
  adapter: ProviderAdapter,
  overrides: Partial<RunOptions> = {},
): RunOptions {
  return {
    dataset,
    adapter,
    model: adapter.id === "jev" ? RETENTION_MODEL : "gpt-6-luna",
    threshold: RETENTION_KEEP_THRESHOLD,
    repeat: 1,
    concurrency: RETENTION_MAX_CONCURRENCY,
    timeoutMs: 5_000,
    maxRequests: 200,
    endpoint: adapter.endpoint,
    ...overrides,
  };
}

// What a fake provider sends for one case.
type Reply =
  | number
  | "refusal"
  | { status: number }
  | { payload: unknown }
  | "not-json"
  | "throw";

function reply(
  adapter: ProviderAdapter,
  caseID: string,
  value: Reply,
  model = adapter.id === "jev" ? RETENTION_MODEL : "gpt-6-luna",
): Response {
  if (value === "throw") throw new Error(`connect failed with ${KEY}`);
  if (value === "not-json") return new Response(`<html>${KEY}</html>`);
  if (typeof value === "object" && "status" in value) {
    return new Response(`{"error":"${KEY}"}`, { status: value.status });
  }
  if (typeof value === "object") return Response.json(value.payload);
  if (adapter.id === "jev") {
    return Response.json({
      model,
      answers: { candidate_0: { type: "noul", noul: value } },
      usage: { input_tokens: 120, output_tokens: 3 },
    });
  }
  return Response.json({
    model,
    answers: [
      value === "refusal"
        ? { type: "refusal", name: caseID }
        : { type: "predicate", name: caseID, probability: value },
    ],
    usage: { input_tokens: 120, output_tokens: 0, total_tokens: 120 },
  });
}

// Fake endpoint: recognizes each request by its exact wire body.
function fakeProvider(
  adapter: ProviderAdapter,
  behavior: (caseID: string, call: number) => Reply | Promise<Reply>,
  model?: string,
) {
  const byBody = new Map(
    cases.map(item => [
      adapter.translate(
        buildCaseRequest(item),
        model ?? options(adapter).model,
      ),
      item.caseID,
    ]),
  );
  const calls: Array<{ caseID: string; body: string; headers: Headers }> = [];
  const fetchImpl: RetentionFetch = async (_url, init) => {
    const body = String(init.body);
    const caseID = byBody.get(body);
    if (!caseID) throw new Error("unexpected request body");
    calls.push({ caseID, body, headers: new Headers(init.headers) });
    return reply(
      adapter,
      caseID,
      await behavior(caseID, calls.length - 1),
      model ?? modelOf(adapter),
    );
  };
  return { fetch: fetchImpl, calls };
}

const modelOf = (adapter: ProviderAdapter) =>
  adapter.id === "jev" ? RETENTION_MODEL : "gpt-6-luna";

const keepProbability = (caseID: string) =>
  cases.find(item => item.caseID === caseID)!.keep ? 0.93 : 0.08;

async function runFake(
  adapter: ProviderAdapter,
  behavior: Parameters<typeof fakeProvider>[1],
  overrides: Partial<RunOptions> = {},
) {
  const provider = fakeProvider(adapter, behavior, overrides.model);
  const out = freshDir();
  const plan = planRun(options(adapter, overrides));
  const outcomes = await executeRun(plan, {
    apiKey: KEY,
    fetch: provider.fetch,
    out,
  });
  return { ...provider, out, plan, outcomes };
}

function allFileText(dir: string): string {
  return readdirSync(dir)
    .map(name => readFileSync(join(dir, name), "utf8"))
    .join("\n");
}

// Loosely typed view of the dataset JSON for deliberate corruption.
type Draft = {
  extra?: unknown;
  schemaVersion: unknown;
  split: unknown;
  version: unknown;
  snapshots: Array<{
    cases: Array<{
      caseID: unknown;
      candidate: Record<string, unknown>;
      label: Record<string, unknown>;
    }>;
  }>;
};

describe("dataset", () => {
  test("the bundled pack is a strict, frozen synthetic pack of production candidates", async () => {
    expect(dataset.split).toBe("smoke");
    expect(cases.length).toBeGreaterThanOrEqual(12);
    expect(cases.length).toBeLessThanOrEqual(20);
    expect(new Set(cases.map(item => item.caseID)).size).toBe(cases.length);
    expect(cases.some(item => item.keep)).toBe(true);
    expect(cases.some(item => !item.keep)).toBe(true);
    expect(datasetSha256(dataset)).toBe(FROZEN.datasetSha256);
    expect(semanticQuestionSha256()).toBe(FROZEN.questionSha256);
    expect(
      requestSetSha256(
        cases.map(item => ({
          caseID: item.caseID,
          sha256: canonicalRequestSha256(buildCaseRequest(item)),
        })),
      ),
    ).toBe(FROZEN.requestSetSha256);
    // Fixtures describe decisions the shipped command makes: the production
    // tool rules change every candidate, so none is a marker-only call.
    for (const item of cases) {
      expect(await isProductionTrimCandidate(item.candidate)).toBe(true);
    }
    expect(
      await isProductionTrimCandidate({
        id: "small",
        tool: "bash",
        input: { command: "ls" },
        output: "ok",
        turnRequest: "",
        requestsBeforeLatest: 0,
      }),
    ).toBe(false);
  });

  test("production constants the pack relies on are unchanged", () => {
    expect(RETENTION_MODEL).toBe("jev-1.13.0");
    expect(RETENTION_KEEP_THRESHOLD).toBe(0.8);
    expect(RETENTION_MAX_CONCURRENCY).toBe(4);
    expect(RETENTION_REQUEST_TIMEOUT_MS).toBe(30_000);
    expect(RETENTION_STATE_AND_QUESTION_BYTE_LIMIT).toBe(28_000);
    expect(RETENTION_TOTAL_BYTE_LIMIT).toBe(56_000);
  });

  test("parsing is strict and identity ignores formatting", () => {
    const text = readFileSync(
      fileURLToPath(
        new URL("../eval/decisions/fixtures.json", import.meta.url),
      ),
      "utf8",
    );
    const raw = JSON.parse(text) as DecisionDataset;
    const mutate = (change: (copy: Draft) => void) => {
      const copy = structuredClone(raw);
      change(copy);
      return JSON.stringify(copy);
    };
    expect(() => parseDataset("{")).toThrow(DecisionEvalError);
    expect(() => parseDataset(mutate(c => (c.extra = 1)))).toThrow(/extra/);
    expect(() =>
      parseDataset(
        mutate(c => (c.snapshots[0]!.cases[0]!.candidate.extra = 1)),
      ),
    ).toThrow(DecisionEvalError);
    expect(() => parseDataset(mutate(c => (c.schemaVersion = 2)))).toThrow(
      DecisionEvalError,
    );
    expect(() => parseDataset(mutate(c => (c.split = "validation")))).toThrow(
      DecisionEvalError,
    );
    expect(() =>
      parseDataset(
        mutate(
          c =>
            (c.snapshots[1]!.cases[0]!.caseID =
              c.snapshots[0]!.cases[0]!.caseID),
        ),
      ),
    ).toThrow(/Duplicate caseID/);
    expect(() =>
      parseDataset(
        mutate(c => (c.snapshots[0]!.cases[0]!.caseID = "Has Space")),
      ),
    ).toThrow(DecisionEvalError);
    expect(() =>
      parseDataset(
        mutate(c => (c.snapshots[0]!.cases[0]!.label.rationale = "")),
      ),
    ).toThrow(DecisionEvalError);
    expect(() =>
      parseDataset(mutate(c => (c.snapshots[0]!.cases[0]!.label.keep = "yes"))),
    ).toThrow(DecisionEvalError);

    const reordered = JSON.stringify(raw, Object.keys(raw).reverse());
    expect(reordered).not.toBe(JSON.stringify(raw));
    expect(datasetSha256(parseDataset(JSON.stringify(raw, null, 4)))).toBe(
      datasetSha256(dataset),
    );
    expect(canonicalJson({ b: 1, a: [2, { d: 1, c: 2 }] })).toBe(
      '{"a":[2,{"c":2,"d":1}],"b":1}',
    );
    expect(
      datasetSha256(parseDataset(mutate(c => (c.version = "1.0.1")))),
    ).not.toBe(datasetSha256(dataset));
  });
});

describe("request mapping", () => {
  test("Jev sends the exact bytes of the production classifier for the production model", async () => {
    for (const item of cases) {
      let sent = "";
      await classifyRetention(item.context, [item.candidate], {
        apiKey: KEY,
        fetch: async (_url, init) => {
          sent = String(init.body);
          return reply(PROVIDERS.jev, item.caseID, 0.5);
        },
      });
      expect(
        PROVIDERS.jev.translate(buildCaseRequest(item), RETENTION_MODEL),
      ).toBe(sent);
    }
  });

  test("an explicit Jev version changes only the model field", () => {
    const request = buildCaseRequest(cases[0]!);
    const production = JSON.parse(
      PROVIDERS.jev.translate(request, RETENTION_MODEL),
    );
    const other = JSON.parse(PROVIDERS.jev.translate(request, "jev-1.14.0"));
    expect(other.model).toBe("jev-1.14.0");
    expect({ ...other, model: production.model }).toEqual(production);
    expect(() => PROVIDERS.jev.checkModel("jev-1.14.0")).not.toThrow();
  });

  test("OpenAI receives the state as input, the production question as one predicate named by the case", () => {
    const item = cases[0]!;
    const request = buildCaseRequest(item);
    const body = PROVIDERS["openai-decisions"].translate(request, "gpt-6-luna");
    const parsed = JSON.parse(body);
    expect(Object.keys(parsed)).toEqual(["model", "input", "questions"]);
    expect(parsed.model).toBe("gpt-6-luna");
    expect(parsed.input).toBe(JSON.stringify(request.body.state));
    expect(parsed.questions).toHaveLength(1);
    const [question] = parsed.questions;
    expect(Object.keys(question)).toEqual(["type", "name", "instructions"]);
    expect(question.type).toBe("predicate");
    expect(question.name).toBe(item.caseID);
    const production = request.body.questions["candidate_0"]!;
    expect(question.instructions).toBe(renderPredicateInstructions(production));
    expect(question.instructions).toContain(production.instructions);
    expect(question.instructions).toContain(production.criteria.true);
    expect(question.instructions).toContain(production.criteria.false);
    expect(PROVIDERS["openai-decisions"].translate(request, "gpt-6-luna")).toBe(
      body,
    );
    expect(PROVIDERS["openai-decisions"].translation()).toEqual({
      version: "openai-predicate-1",
      sha256: FROZEN.openaiTranslationSha256,
    });
    expect(PROVIDERS.jev.translation()).toBeNull();
    expect(semanticQuestion().instructions).toContain("candidates[0]");
  });

  test("canonical and provider request hashes are recorded separately", async () => {
    const { outcomes, calls } = await runFake(
      PROVIDERS["openai-decisions"],
      id => keepProbability(id),
    );
    for (const outcome of outcomes) {
      const call = calls.find(c => c.caseID === outcome.caseID)!;
      if (outcome.status === "not-attempted") throw new Error("unexpected");
      expect(outcome.providerRequestSha256).toBe(sha256(call.body));
      expect(outcome.canonicalRequestSha256).toBe(
        canonicalRequestSha256(
          buildCaseRequest(cases.find(c => c.caseID === outcome.caseID)!),
        ),
      );
      expect(outcome.providerRequestSha256).not.toBe(
        outcome.canonicalRequestSha256,
      );
    }
  });

  test("model selection refuses aliases and unsupported OpenAI models before any request", () => {
    for (const alias of ["jev-latest", "jev-preview", "jev", "jev-1.13"]) {
      expect(() => planRun(options(PROVIDERS.jev, { model: alias }))).toThrow(
        /versioned IDs/,
      );
    }
    expect(() =>
      planRun(options(PROVIDERS.jev, { model: "jev-1.14.0" })),
    ).not.toThrow();
    expect(OPENAI_DECISIONS_MODELS).toEqual(["gpt-6-luna"]);
    for (const model of ["gpt-6", "gpt-5.5", "gpt-6-luna-fast"]) {
      expect(() =>
        planRun(options(PROVIDERS["openai-decisions"], { model })),
      ).toThrow(/only gpt-6-luna/);
    }
  });
});

async function sendOne(
  adapter: ProviderAdapter,
  behavior: RetentionFetch,
  timeoutMs = 5_000,
  model = options(adapter).model,
): Promise<Transport> {
  const request = buildCaseRequest(cases[0]!);
  return sendDecision(adapter, adapter.translate(request, model), request, {
    model,
    apiKey: KEY,
    endpoint: adapter.endpoint,
    timeoutMs,
    fetch: behavior,
  });
}

const respondWith =
  (payload: unknown): RetentionFetch =>
  async () =>
    Response.json(payload);

describe("response interpretation", () => {
  const jev = PROVIDERS.jev;
  const openai = PROVIDERS["openai-decisions"];
  const name = cases[0]!.caseID;
  const jevPayload = (answer: unknown, model = RETENTION_MODEL) => ({
    model,
    answers: { candidate_0: answer },
  });
  const openaiPayload = (answers: unknown, model = "gpt-6-luna") => ({
    model,
    answers,
  });

  test("Jev accepts one noul and reports model, usage and latency", async () => {
    const result = await sendOne(jev, async () => reply(jev, name, 0.8));
    expect(result).toMatchObject({
      status: "scored",
      probability: 0.8,
      requestedModel: RETENTION_MODEL,
      returnedModel: RETENTION_MODEL,
      usage: { inputTokens: 120, outputTokens: 3 },
    });
    expect((result as { latencyMs: number }).latencyMs).toBeGreaterThanOrEqual(
      0,
    );
    const bare = await sendOne(
      jev,
      respondWith(jevPayload({ type: "noul", noul: 0 })),
    );
    expect(bare).toMatchObject({ status: "scored", probability: 0 });
    expect(bare).not.toHaveProperty("usage");
  });

  test("Jev rejects a returned model that differs from the request, aliases included", async () => {
    const result = await sendOne(
      jev,
      respondWith(jevPayload({ type: "noul", noul: 0.9 }, "jev-1.14.0")),
    );
    expect(result).toMatchObject({
      status: "failed",
      category: "model-mismatch",
    });
    const alias = await sendOne(
      jev,
      respondWith(jevPayload({ type: "noul", noul: 0.9 }, "jev-latest")),
    );
    expect(alias).toMatchObject({
      status: "failed",
      category: "model-mismatch",
    });
    const missing = await sendOne(
      jev,
      respondWith({ answers: { candidate_0: { type: "noul", noul: 0.9 } } }),
    );
    expect(missing).toMatchObject({
      status: "failed",
      category: "invalid-response",
    });
  });

  test("Jev rejects invalid probabilities and answer shapes", async () => {
    const invalid = [
      jevPayload({ type: "noul", noul: "0.5" }),
      jevPayload({ type: "noul", noul: 1.0001 }),
      jevPayload({ type: "noul", noul: -0.0001 }),
      jevPayload({ type: "noul", noul: null }),
      jevPayload({ type: "noul" }),
      jevPayload({ type: "choice", choice: "a" }),
      jevPayload("noul"),
      { model: RETENTION_MODEL, answers: {} },
      {
        model: RETENTION_MODEL,
        answers: { other: { type: "noul", noul: 0.5 } },
      },
      {
        model: RETENTION_MODEL,
        answers: {
          candidate_0: { type: "noul", noul: 0.5 },
          candidate_1: { type: "noul", noul: 0.5 },
        },
      },
      { model: RETENTION_MODEL, answers: [] },
      [],
      null,
    ];
    for (const payload of invalid) {
      expect(await sendOne(jev, respondWith(payload))).toMatchObject({
        status: "failed",
        category: "invalid-response",
      });
    }
  });

  test("OpenAI accepts one predicate and treats a refusal as neither keep nor trim", async () => {
    const scored = await sendOne(openai, async () => reply(openai, name, 0.65));
    expect(scored).toMatchObject({
      status: "scored",
      probability: 0.65,
      requestedModel: "gpt-6-luna",
      usage: { inputTokens: 120, outputTokens: 0 },
    });
    const refused = await sendOne(openai, async () =>
      reply(openai, name, "refusal"),
    );
    expect(refused).toMatchObject({
      status: "refused",
      reason: "provider-refusal",
    });
    expect(refused).not.toHaveProperty("probability");
  });

  test("OpenAI must report exactly the requested model for a predicate and for a refusal", async () => {
    // Only the exact ID is documented. A dated label or another model is a
    // mismatch until a first-party contract says how to normalize it.
    for (const returned of [
      "gpt-6-luna-2026-10-01",
      "gpt-6-luna-fast",
      "gpt-6-sol",
      "GPT-6-LUNA",
    ]) {
      for (const answer of [
        { type: "predicate", name, probability: 0.5 },
        { type: "refusal", name },
      ]) {
        expect(
          await sendOne(openai, respondWith(openaiPayload([answer], returned))),
        ).toMatchObject({ status: "failed", category: "model-mismatch" });
      }
    }
  });

  test("usage keeps each reported count on its own and omits one that cannot be stored", async () => {
    const usageOf = async (usage: unknown) => {
      const result = await sendOne(
        jev,
        respondWith({ ...jevPayload({ type: "noul", noul: 0.5 }), usage }),
      );
      expect(result).toMatchObject({ status: "scored" });
      return (result as { usage?: unknown }).usage;
    };
    expect(await usageOf({ input_tokens: 7 })).toEqual({ inputTokens: 7 });
    expect(await usageOf({ output_tokens: 0 })).toEqual({ outputTokens: 0 });
    expect(await usageOf({ input_tokens: 0, output_tokens: 0 })).toEqual({
      inputTokens: 0,
      outputTokens: 0,
    });
    expect(await usageOf({})).toBeUndefined();
    expect(await usageOf({ input_tokens: "5" })).toBeUndefined();
    expect(
      await usageOf({ input_tokens: -1, output_tokens: 1.5 }),
    ).toBeUndefined();
    // A count above the safe integer range is not a number the harness can
    // store or sum exactly, so that dimension is treated as not reported.
    expect(
      await usageOf({
        input_tokens: Number.MAX_SAFE_INTEGER,
        output_tokens: Number.MAX_SAFE_INTEGER + 1,
      }),
    ).toEqual({ inputTokens: Number.MAX_SAFE_INTEGER });
    expect(await usageOf({ input_tokens: 1e300 })).toBeUndefined();
  });

  test("a redirect to another origin is not followed and is recorded as an HTTP failure", async () => {
    // Real fetch against two loopback origins: a 307 keeps method and body, so
    // following it would send the case text to a server nobody chose.
    const reached: string[] = [];
    const target = Bun.serve({
      port: 0,
      hostname: "127.0.0.1",
      fetch(request) {
        reached.push(request.url);
        return Response.json({});
      },
    });
    const redirector = Bun.serve({
      port: 0,
      hostname: "127.0.0.1",
      fetch: () =>
        new Response(null, {
          status: 307,
          headers: { location: `http://127.0.0.1:${target.port}/elsewhere` },
        }),
    });
    try {
      const request = buildCaseRequest(cases[0]!);
      const result = await sendDecision(
        jev,
        jev.translate(request, RETENTION_MODEL),
        request,
        {
          model: RETENTION_MODEL,
          apiKey: KEY,
          endpoint: `http://127.0.0.1:${redirector.port}/v1/systemone`,
          timeoutMs: 5_000,
          fetch: (url, init) => fetch(url, init),
        },
      );
      expect(result).toMatchObject({
        status: "failed",
        category: "http",
        httpStatus: 307,
      });
      expect(reached).toEqual([]);
    } finally {
      await redirector.stop(true);
      await target.stop(true);
    }
  });

  test("OpenAI answers must be exactly one answer for the requested name and type", async () => {
    const predicate = { type: "predicate", name, probability: 0.5 };
    const invalid = [
      openaiPayload([]),
      openaiPayload([predicate, predicate]),
      openaiPayload([predicate, { type: "refusal", name }]),
      openaiPayload([{ ...predicate, name: "other-case" }]),
      openaiPayload([{ ...predicate, name: null }]),
      openaiPayload([{ type: "predicate", probability: 0.5 }]),
      openaiPayload([{ type: "refusal", name: "other-case" }]),
      openaiPayload([{ type: "refusal" }]),
      openaiPayload([{ type: "choice", name, choice: "a", probabilities: [] }]),
      openaiPayload([{ ...predicate, probability: "0.5" }]),
      openaiPayload([{ ...predicate, probability: 1.2 }]),
      openaiPayload([{ ...predicate, probability: -0.2 }]),
      openaiPayload([{ type: "predicate", name }]),
      openaiPayload({ [name]: predicate }),
      openaiPayload(null),
      { answers: [predicate] },
    ];
    for (const payload of invalid) {
      expect(await sendOne(openai, respondWith(payload))).toMatchObject({
        status: "failed",
        category: "invalid-response",
      });
    }
  });

  test("failures reduce to a fixed category and never carry request, response or key text", async () => {
    const statuses: Array<[number, string]> = [
      [401, "auth"],
      [403, "auth"],
      [429, "rate-limit"],
      [400, "http"],
      [404, "http"],
      [500, "http"],
      [529, "http"],
    ];
    for (const adapter of [PROVIDERS.jev, PROVIDERS["openai-decisions"]]) {
      for (const [status, category] of statuses) {
        const result = await sendOne(adapter, async () =>
          reply(adapter, name, { status }),
        );
        expect(result).toMatchObject({
          status: "failed",
          category,
          httpStatus: status,
        });
        expect(JSON.stringify(result)).not.toContain(KEY);
      }
      const malformed = await sendOne(adapter, async () =>
        reply(adapter, name, "not-json"),
      );
      expect(malformed).toMatchObject({
        status: "failed",
        category: "invalid-response",
      });
      const network = await sendOne(adapter, async () =>
        reply(adapter, name, "throw"),
      );
      expect(network).toMatchObject({ status: "failed", category: "network" });
      for (const result of [malformed, network]) {
        expect(JSON.stringify(result)).not.toContain(KEY);
        expect(result).not.toHaveProperty("httpStatus");
      }
    }
  });

  test("a request that outlives its timeout fails as a timeout", async () => {
    const result = await sendOne(
      jev,
      (_url, init) =>
        new Promise<Response>((_resolve, reject) => {
          init.signal!.addEventListener("abort", () =>
            reject(init.signal!.reason),
          );
        }),
      20,
    );
    expect(result).toMatchObject({ status: "failed", category: "timeout" });
  });

  test("requests carry the key only in the Authorization header", async () => {
    const { calls } = await runFake(PROVIDERS["openai-decisions"], id =>
      keepProbability(id),
    );
    for (const call of calls) {
      expect(call.headers.get("authorization")).toBe(`Bearer ${KEY}`);
      expect(call.body).not.toContain(KEY);
    }
  });
});

describe("run execution", () => {
  const openai = PROVIDERS["openai-decisions"];

  test("every outcome is saved, nothing is retried and failures are not scored as zero", async () => {
    const { out, calls, outcomes } = await runFake(openai, (caseID, call) =>
      call === 0
        ? { status: 500 }
        : call === 1
          ? "refusal"
          : keepProbability(caseID),
    );
    expect(calls).toHaveLength(cases.length);
    expect(new Set(calls.map(c => c.caseID)).size).toBe(cases.length);
    expect(outcomes.map(o => o.status).sort()).toEqual(
      ["failed", "refused", ...Array(cases.length - 2).fill("scored")].sort(),
    );
    const lines = readFileSync(join(out, "outcomes.jsonl"), "utf8")
      .trim()
      .split("\n");
    expect(lines).toHaveLength(cases.length);
    const failed = outcomes.find(o => o.status === "failed")!;
    expect(failed).not.toHaveProperty("probability");
    expect(failed).toMatchObject({ category: "http", httpStatus: 500 });
    expect(allFileText(out)).not.toContain(KEY);
    const run = loadRun(out, dataset);
    expect(run.planned).toBe(cases.length);
    const metrics = computeMetrics(
      run,
      new Map(cases.map(item => [item.caseID, item.keep])),
      0.8,
    );
    expect(metrics).toMatchObject({
      scored: cases.length - 2,
      refused: 1,
      failed: 1,
      notAttempted: 0,
    });
    expect(metrics.failures).toEqual({ http: 1 });
    // The failure and the refusal did not stop the other cases, and every
    // planned request has a record.
    expect(metrics.unknown).toBe(0);
  });

  test("concurrency stays within the requested bound", async () => {
    let inFlight = 0;
    let peak = 0;
    await runFake(
      openai,
      async caseID => {
        inFlight += 1;
        peak = Math.max(peak, inFlight);
        await Bun.sleep(5);
        inFlight -= 1;
        return keepProbability(caseID);
      },
      { concurrency: 3 },
    );
    expect(peak).toBe(3);
  });

  test("repeats send each case again and the plan is capped before any request", async () => {
    const { calls, plan } = await runFake(openai, id => keepProbability(id), {
      repeat: 2,
    });
    expect(plan.units).toHaveLength(cases.length * 2);
    expect(calls).toHaveLength(cases.length * 2);
    expect(plan.units.slice(0, cases.length).every(u => u.repeat === 1)).toBe(
      true,
    );
    const cap = () =>
      planRun(
        options(openai, { repeat: 3, maxRequests: cases.length * 3 - 1 }),
      );
    expect(cap).toThrow(/above --max-requests/);
    expect(() =>
      planRun(options(openai, { repeat: 3, maxRequests: cases.length * 3 })),
    ).not.toThrow();
    for (const bad of [0, 11, 1.5, NaN]) {
      expect(() => planRun(options(openai, { repeat: bad }))).toThrow(
        /--repeat/,
      );
    }
    for (const bad of [0, 17, 2.5]) {
      expect(() => planRun(options(openai, { concurrency: bad }))).toThrow(
        /--concurrency/,
      );
    }
  });

  test("a rejected key stops admission, lets in-flight requests finish and records the rest as not attempted", async () => {
    const serial = await runFake(openai, () => ({ status: 401 }), {
      concurrency: 1,
    });
    expect(serial.calls).toHaveLength(1);
    expect(
      serial.outcomes.filter(o => o.status === "not-attempted"),
    ).toHaveLength(cases.length - 1);
    expect(
      serial.outcomes.find(o => o.status === "not-attempted"),
    ).toMatchObject({
      reason: "stopped-after-auth",
    });
    const run = loadRun(serial.out, dataset);
    expect(run.outcomes).toHaveLength(cases.length);
    const metrics = computeMetrics(run, new Map(), 0.8);
    // A run that closed after a rejected key says exactly what it did not
    // send; nothing about it is unknown.
    expect(metrics).toMatchObject({
      scored: 0,
      failed: 1,
      notAttempted: cases.length - 1,
      unknown: 0,
    });

    const parallel = await runFake(openai, () => ({ status: 429 }), {
      concurrency: 3,
    });
    expect(parallel.calls).toHaveLength(3);
    expect(parallel.outcomes.filter(o => o.status === "failed")).toHaveLength(
      3,
    );
    expect(
      parallel.outcomes.find(o => o.status === "not-attempted"),
    ).toMatchObject({
      reason: "stopped-after-rate-limit",
    });
  });

  test("an existing output path is never reused and no request is made", async () => {
    const provider = fakeProvider(openai, id => keepProbability(id));
    const out = freshDir();
    mkdirSync(out);
    writeFileSync(join(out, "keep.txt"), "mine");
    await expect(
      executeRun(planRun(options(openai)), {
        apiKey: KEY,
        fetch: provider.fetch,
        out,
      }),
    ).rejects.toThrow(/already exists/);
    expect(provider.calls).toHaveLength(0);
    expect(readdirSync(out)).toEqual(["keep.txt"]);
  });
});

function syntheticRun(outcomes: Outcome[], repeat = 1): LoadedRun {
  const digest = "0".repeat(64);
  return {
    name: "synthetic",
    planned: cases.length * repeat,
    outcomes,
    manifest: {
      formatVersion: 1,
      startedAt: "",
      provider: "jev",
      model: RETENTION_MODEL,
      endpoint: "",
      dataset: {
        datasetID: dataset.datasetID,
        version: dataset.version,
        split: dataset.split,
        sha256: digest,
        caseCount: cases.length,
      },
      prompt: {
        semanticQuestionSha256: digest,
        requestSetSha256: digest,
        translation: null,
      },
      production: { jevModel: RETENTION_MODEL, keepThreshold: 0.8 },
      threshold: 0.8,
      repeat,
      concurrency: 4,
      timeoutMs: 1,
      maxRequests: 1,
      cases: [],
    },
  };
}

function scoredOutcome(
  caseID: string,
  probability: number,
  repeat = 1,
  usage?: { inputTokens?: number; outputTokens?: number },
): Outcome {
  return {
    status: "scored",
    caseID,
    repeat,
    canonicalRequestSha256: "0".repeat(64),
    providerRequestSha256: "1".repeat(64),
    probability,
    requestedModel: RETENTION_MODEL,
    returnedModel: RETENTION_MODEL,
    latencyMs: 10,
    ...(usage ? { usage } : {}),
  };
}

describe("metrics", () => {
  const labels = new Map(cases.map(item => [item.caseID, item.keep]));

  test("confusion, precision, recall, specificity, F1, Brier and AUC for known probabilities", () => {
    const [a, b, c, d] = [
      cases.find(item => item.keep)!,
      cases.filter(item => item.keep)[1]!,
      cases.find(item => !item.keep)!,
      cases.filter(item => !item.keep)[1]!,
    ];
    const metrics = computeMetrics(
      syntheticRun([
        scoredOutcome(a.caseID, 0.9), // true keep
        scoredOutcome(b.caseID, 0.5), // false prune
        scoredOutcome(c.caseID, 0.9), // false keep
        scoredOutcome(d.caseID, 0.1), // true trim
      ]),
      labels,
      0.8,
    );
    expect(metrics).toMatchObject({ tp: 1, fp: 1, fn: 1, tn: 1, scored: 4 });
    expect(metrics.precision).toBe(0.5);
    expect(metrics.recall).toBe(0.5);
    expect(metrics.specificity).toBe(0.5);
    expect(metrics.f1).toBe(0.5);
    expect(metrics.brier).toBeCloseTo((0.01 + 0.25 + 0.81 + 0.01) / 4, 10);
    // pairs (useful, not useful): 0.9>0.1, 0.9=0.9, 0.5<0.9, 0.5>0.1
    expect(metrics.auc).toBeCloseTo((1 + 0.5 + 0 + 1) / 4, 10);
    expect(metrics.falsePrunes.map(e => e.caseID)).toEqual([b.caseID]);
    expect(metrics.falseKeeps.map(e => e.caseID)).toEqual([c.caseID]);
  });

  test("a scored run with no true keep has F1 0, not undefined", () => {
    // One useful call trimmed and one wasteful call kept: every decision is
    // wrong, so the scores are zero, not missing.
    const keep = cases.find(item => item.keep)!;
    const trim = cases.find(item => !item.keep)!;
    const inverted = computeMetrics(
      syntheticRun([
        scoredOutcome(keep.caseID, 0.1),
        scoredOutcome(trim.caseID, 0.95),
      ]),
      labels,
      0.8,
    );
    expect(inverted).toMatchObject({ tp: 0, fp: 1, fn: 1, tn: 0 });
    expect(inverted.precision).toBe(0);
    expect(inverted.recall).toBe(0);
    expect(inverted.f1).toBe(0);
    expect(inverted.accuracy).toBe(0);

    // F1 is undefined only when it has no true keeps, false keeps or false
    // trims to count; accuracy then still reports the correct trim.
    const onlyTrim = computeMetrics(
      syntheticRun([scoredOutcome(trim.caseID, 0.1)]),
      labels,
      0.8,
    );
    expect(onlyTrim).toMatchObject({ tp: 0, fp: 0, fn: 0, tn: 1 });
    expect(onlyTrim.f1).toBeNull();
    expect(onlyTrim.accuracy).toBe(1);
    expect(computeMetrics(syntheticRun([]), labels, 0.8).accuracy).toBeNull();
  });

  test("accuracy counts correct decisions among scored requests only", () => {
    const [k1, k2] = cases.filter(item => item.keep);
    const [t1, t2] = cases.filter(item => !item.keep);
    const metrics = computeMetrics(
      syntheticRun([
        scoredOutcome(k1!.caseID, 0.9),
        scoredOutcome(k2!.caseID, 0.2),
        scoredOutcome(t1!.caseID, 0.1),
        scoredOutcome(t2!.caseID, 0.1),
      ]),
      labels,
      0.8,
    );
    // 3 of 4 scored decisions are right; the other planned requests have no
    // answer and are not counted as right or wrong.
    expect(metrics.accuracy).toBe(0.75);
    expect(metrics.f1).toBeCloseTo(2 / 3, 10);
    expect(metrics.scored).toBe(4);
    expect(metrics.planned).toBe(cases.length);
  });

  test("usage totals count each reported dimension separately and say how many responses reported it", () => {
    const ids = cases.map(item => item.caseID);
    const run = (usages: Array<{ i?: number; o?: number } | undefined>) =>
      computeMetrics(
        syntheticRun(
          usages.map((usage, index) =>
            scoredOutcome(
              ids[index]!,
              0.5,
              1,
              usage && {
                ...(usage.i === undefined ? {} : { inputTokens: usage.i }),
                ...(usage.o === undefined ? {} : { outputTokens: usage.o }),
              },
            ),
          ),
        ),
        labels,
        0.8,
      ).usage;

    expect(
      run([
        { i: 10, o: 1 },
        { i: 20, o: 2 },
      ]),
    ).toEqual({
      eligibleResponses: 2,
      inputTokens: 30n,
      inputReporting: 2,
      outputTokens: 3n,
      outputReporting: 2,
    });
    // Input only, output only and neither are different from a reported 0.
    expect(run([{ i: 10 }, { i: 5 }])).toMatchObject({
      inputTokens: 15n,
      inputReporting: 2,
      outputTokens: null,
      outputReporting: 0,
    });
    expect(run([{ o: 4 }, { o: 6 }])).toMatchObject({
      inputTokens: null,
      inputReporting: 0,
      outputTokens: 10n,
      outputReporting: 2,
    });
    expect(run([undefined, undefined])).toEqual({
      eligibleResponses: 2,
      inputTokens: null,
      inputReporting: 0,
      outputTokens: null,
      outputReporting: 0,
    });
    // A reported zero is a report, and a missing count stays missing.
    expect(run([{ i: 0, o: 0 }, undefined])).toMatchObject({
      inputTokens: 0n,
      inputReporting: 1,
      outputTokens: 0n,
      outputReporting: 1,
      eligibleResponses: 2,
    });
    // Mixed coverage: a subtotal over the responses that reported it.
    expect(run([{ i: 10, o: 1 }, { i: 20 }, { o: 5 }, undefined])).toEqual({
      eligibleResponses: 4,
      inputTokens: 30n,
      inputReporting: 2,
      outputTokens: 6n,
      outputReporting: 2,
    });
    // Sums do not lose precision.
    expect(
      run([{ i: Number.MAX_SAFE_INTEGER }, { i: Number.MAX_SAFE_INTEGER }])
        .inputTokens,
    ).toBe(2n * BigInt(Number.MAX_SAFE_INTEGER));
  });

  test("failed and unattempted requests are not usage-eligible", () => {
    const [keep, trim] = [cases.find(i => i.keep)!, cases.find(i => !i.keep)!];
    const usage = computeMetrics(
      syntheticRun([
        scoredOutcome(keep.caseID, 0.9, 1, { inputTokens: 100 }),
        {
          status: "refused",
          reason: "provider-refusal",
          caseID: trim.caseID,
          repeat: 1,
          canonicalRequestSha256: "0".repeat(64),
          providerRequestSha256: "1".repeat(64),
          requestedModel: "gpt-6-luna",
          returnedModel: "gpt-6-luna",
          latencyMs: 5,
          usage: { inputTokens: 40, outputTokens: 0 },
        },
        {
          status: "failed",
          category: "timeout",
          caseID: cases[2]!.caseID,
          repeat: 1,
          canonicalRequestSha256: "0".repeat(64),
          providerRequestSha256: "1".repeat(64),
          latencyMs: 30_000,
        },
      ]),
      labels,
      0.8,
    ).usage;
    expect(usage).toEqual({
      eligibleResponses: 2,
      inputTokens: 140n,
      inputReporting: 2,
      outputTokens: 0n,
      outputReporting: 1,
    });
  });

  test("a probability equal to the threshold is a keep", () => {
    const keep = cases.find(item => item.keep)!;
    const trim = cases.find(item => !item.keep)!;
    const at = (p: number, item: typeof keep) =>
      computeMetrics(
        syntheticRun([scoredOutcome(item.caseID, p)]),
        labels,
        0.8,
      );
    expect(at(0.8, keep)).toMatchObject({ tp: 1, fn: 0 });
    expect(at(0.7999, keep)).toMatchObject({ tp: 0, fn: 1 });
    expect(at(0.8, trim)).toMatchObject({ fp: 1, tn: 0 });
    expect(at(0.79, trim)).toMatchObject({ fp: 0, tn: 1 });
  });

  test("refusals, failures and unattempted requests stay out of the confusion denominators", () => {
    const [keep, trim] = [cases.find(i => i.keep)!, cases.find(i => !i.keep)!];
    const metrics = computeMetrics(
      syntheticRun([
        scoredOutcome(keep.caseID, 0.95),
        {
          status: "refused",
          reason: "provider-refusal",
          caseID: trim.caseID,
          repeat: 1,
          canonicalRequestSha256: "0".repeat(64),
          providerRequestSha256: "1".repeat(64),
          requestedModel: "gpt-6-luna",
          returnedModel: "gpt-6-luna",
          latencyMs: 5,
        },
        {
          status: "failed",
          category: "timeout",
          caseID: cases[2]!.caseID,
          repeat: 1,
          canonicalRequestSha256: "0".repeat(64),
          providerRequestSha256: "1".repeat(64),
          latencyMs: 30_000,
        },
      ]),
      labels,
      0.8,
    );
    expect(metrics).toMatchObject({
      planned: cases.length,
      scored: 1,
      refused: 1,
      failed: 1,
      notAttempted: 0,
      unknown: cases.length - 3,
      tp: 1,
      fp: 0,
      fn: 0,
      tn: 0,
    });
    expect(metrics.precision).toBe(1);
    expect(metrics.specificity).toBeNull();
    expect(metrics.latencyP50Ms).toBe(5);
  });

  test("no valid answers leaves every ratio undefined instead of zero or NaN", () => {
    const metrics = computeMetrics(syntheticRun([]), labels, 0.8);
    expect(metrics).toMatchObject({
      scored: 0,
      notAttempted: 0,
      unknown: cases.length,
    });
    for (const key of [
      "precision",
      "recall",
      "specificity",
      "f1",
      "accuracy",
      "brier",
      "auc",
    ] as const) {
      expect(metrics[key]).toBeNull();
    }
    expect(metrics.latencyP50Ms).toBeNull();
    const single = computeMetrics(
      syntheticRun([scoredOutcome(cases.find(i => i.keep)!.caseID, 0.9)]),
      labels,
      0.8,
    );
    expect(single.auc).toBeNull();
    expect(single.specificity).toBeNull();
  });

  test("repeats report probability spread and decision flips", () => {
    const keep = cases.find(i => i.keep)!;
    const stable = cases.filter(i => i.keep)[1]!;
    const metrics = computeMetrics(
      syntheticRun(
        [
          scoredOutcome(keep.caseID, 0.78),
          scoredOutcome(keep.caseID, 0.82, 2),
          scoredOutcome(stable.caseID, 0.9),
          scoredOutcome(stable.caseID, 0.91, 2),
        ],
        2,
      ),
      labels,
      0.8,
    );
    expect(metrics.stability?.cases).toBe(2);
    expect(metrics.stability?.flips).toBe(1);
    expect(metrics.stability?.maxSpread).toBeCloseTo(0.04, 10);
  });
});

describe("report and result integrity", () => {
  const openai = PROVIDERS["openai-decisions"];
  const perfect = (id: string) => keepProbability(id);

  test("reports separate runs with different thresholds and carry the reference caveat", async () => {
    const a = await runFake(openai, perfect);
    const b = await runFake(openai, perfect);
    const c = await runFake(openai, perfect, { threshold: 0.5 });
    const same = renderReport(
      [loadRun(a.out, dataset), loadRun(b.out, dataset)],
      dataset,
    );
    expect(same).toContain("Group 1");
    expect(same).not.toContain("Group 2");
    expect(same).toContain("not calibrated");
    expect(same).toContain("1.000"); // perfect oracle precision
    const mixed = renderReport(
      [loadRun(a.out, dataset), loadRun(c.out, dataset)],
      dataset,
    );
    expect(mixed).toContain("Group 2");
    expect(mixed).toContain("must not be compared across groups");
    expect(mixed).toContain("threshold 0.5");
    const override = renderReport([loadRun(a.out, dataset)], dataset, 0.95);
    expect(override).toContain("threshold 0.95");
  });

  test("requests that left no record are reported as unknown, never as not attempted", async () => {
    const { out } = await runFake(openai, perfect);
    const file = join(out, "outcomes.jsonl");
    const kept = readFileSync(file, "utf8").split("\n").slice(0, 5).join("\n");
    writeFileSync(file, `${kept}\n`);
    const run = loadRun(out, dataset);
    expect(
      computeMetrics(run, new Map(cases.map(i => [i.caseID, i.keep])), 0.8),
    ).toMatchObject({
      scored: 5,
      notAttempted: 0,
      unknown: cases.length - 5,
    });
    const text = renderReport([run], dataset);
    expect(text).toContain(`5/${cases.length}`);
    expect(text).toContain("incomplete coverage");
    expect(text).toContain(
      `${cases.length - 5} requests have no outcome record`,
    );
    expect(text).toContain("may have been sent");
  });

  test("a process killed after the provider received a request leaves its outcome unknown", async () => {
    // The fake endpoint has the request, then the evaluator dies before it
    // can append anything. Only the manifest exists on disk.
    let receivedFirst!: () => void;
    const received = new Promise<void>(resolve => (receivedFirst = resolve));
    const server = Bun.serve({
      port: 0,
      hostname: "127.0.0.1",
      async fetch(request) {
        await request.text();
        receivedFirst();
        return new Promise<Response>(() => {});
      },
    });
    const out = freshDir();
    try {
      const proc = Bun.spawn(
        [
          process.execPath,
          fileURLToPath(new URL("../eval/decisions/run.ts", import.meta.url)),
          "run",
          "--provider",
          "jev",
          "--endpoint",
          `http://127.0.0.1:${server.port}/v1/systemone`,
          "--out",
          out,
        ],
        {
          env: { PATH: process.env["PATH"] ?? "", TYPESAFE_API_KEY: KEY },
          stdout: "ignore",
          stderr: "ignore",
        },
      );
      await received;
      proc.kill("SIGKILL");
      await proc.exited;
    } finally {
      await server.stop(true);
    }
    expect(readdirSync(out)).toEqual(["manifest.json"]);

    const run = loadRun(out, dataset);
    const metrics = computeMetrics(
      run,
      new Map(cases.map(i => [i.caseID, i.keep])),
      0.8,
    );
    expect(metrics).toMatchObject({
      scored: 0,
      refused: 0,
      failed: 0,
      notAttempted: 0,
      unknown: cases.length,
    });
    expect(metrics.f1).toBeNull();
    const printed: string[] = [];
    const cli: CliDeps = {
      env: {},
      fetch: async () => {
        throw new Error("network is not available");
      },
      stdout: text => printed.push(text),
      stderr: () => {},
    };
    expect(await main(["report", out], cli)).toBe(1);
    const text = printed.join("\n");
    expect(text).toContain(`${cases.length} requests have no outcome record`);
    expect(text).not.toMatch(/not attempted \d/);
  });

  test("a final record cut off mid-write is reported as corrupt data, not as a request that was never sent", async () => {
    const { out } = await runFake(openai, perfect);
    const file = join(out, "outcomes.jsonl");
    const lines = readFileSync(file, "utf8").trim().split("\n");
    const cut = lines[4]!.slice(0, Math.floor(lines[4]!.length / 2));
    writeFileSync(file, `${lines.slice(0, 4).join("\n")}\n${cut}`);
    expect(() => loadRun(out, dataset)).toThrow(
      /unreadable outcome record at line 5/,
    );
  });

  test("a held-out dataset is scored and reported only at the frozen threshold", async () => {
    const heldout = parseDataset(
      JSON.stringify({
        ...JSON.parse(
          readFileSync(
            fileURLToPath(
              new URL("../eval/decisions/fixtures.json", import.meta.url),
            ),
            "utf8",
          ),
        ),
        split: "heldout",
      }),
    );
    expect(() =>
      planRun({ ...options(openai), dataset: heldout, threshold: 0.7 }),
    ).toThrow(/frozen threshold 0.8/);
    expect(() =>
      planRun({ ...options(openai), dataset: heldout, threshold: 0.8 }),
    ).not.toThrow();
    const provider = fakeProvider(openai, perfect);
    const out = freshDir();
    await executeRun(planRun({ ...options(openai), dataset: heldout }), {
      apiKey: KEY,
      fetch: provider.fetch,
      out,
    });
    const run = loadRun(out, heldout);
    expect(renderReport([run], heldout)).toContain("(heldout)");
    expect(() => renderReport([run], heldout, 0.5)).toThrow(/frozen threshold/);
    // A manifest edited to a different threshold is not a valid held-out result.
    const manifestPath = join(out, "manifest.json");
    const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
    writeFileSync(
      manifestPath,
      JSON.stringify({ ...manifest, threshold: 0.5 }),
    );
    expect(() => loadRun(out, heldout)).toThrow(/frozen/);
    // Results never mix across splits: the smoke dataset rejects held-out runs.
    expect(() => loadRun(out, dataset)).toThrow(/different dataset/);
  });

  test("results that do not belong to the dataset or plan are rejected", async () => {
    const { out } = await runFake(openai, perfect);
    expect(() => loadRun(join(scratch, "missing"), dataset)).toThrow(
      /not a decision evaluation result/,
    );
    const empty = freshDir();
    mkdirSync(empty);
    writeFileSync(join(empty, "manifest.json"), "{}");
    expect(() => loadRun(empty, dataset)).toThrow(
      /not a decision evaluation result/,
    );

    const edited = parseDataset(
      JSON.stringify({
        ...JSON.parse(
          readFileSync(
            fileURLToPath(
              new URL("../eval/decisions/fixtures.json", import.meta.url),
            ),
            "utf8",
          ),
        ),
        version: "1.0.1",
      }),
    );
    expect(() => loadRun(out, edited)).toThrow(/different dataset/);

    const file = join(out, "outcomes.jsonl");
    const original = readFileSync(file, "utf8");
    const first = original.split("\n")[0]!;
    const tamper = (text: string, pattern: RegExp) => {
      writeFileSync(file, text);
      expect(() => loadRun(out, dataset)).toThrow(pattern);
    };
    tamper(`${original}${first}\n`, /does not belong to its plan/);
    tamper(
      original.replace(/"caseID":"[^"]+"/, '"caseID":"unknown-case"'),
      /does not belong/,
    );
    tamper(original.replace(/"repeat":1/, '"repeat":2'), /does not belong/);
    tamper(
      original.replace(/"probability":[0-9.]+/, '"probability":1.5'),
      /invalid outcome record/,
    );
    tamper(`${original}not json\n`, /unreadable outcome record/);
    tamper(
      original.replace(
        /"canonicalRequestSha256":"[0-9a-f]{64}"/,
        `"canonicalRequestSha256":"${"a".repeat(64)}"`,
      ),
      /does not belong/,
    );
    writeFileSync(file, original);
    const manifestPath = join(out, "manifest.json");
    const manifestText = readFileSync(manifestPath, "utf8");
    const manifest = JSON.parse(manifestText);
    manifest.prompt.requestSetSha256 = "b".repeat(64);
    writeFileSync(manifestPath, JSON.stringify(manifest));
    expect(() => loadRun(out, dataset)).toThrow(/request set/);
    writeFileSync(manifestPath, manifestText);
    expect(() => loadRun(out, dataset)).not.toThrow();
  });
});

describe("result provenance", () => {
  const jev = PROVIDERS.jev;
  const openai = PROVIDERS["openai-decisions"];
  const perfect = (id: string) => keepProbability(id);
  type Json = Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any
  const readManifest = (out: string): Json =>
    JSON.parse(readFileSync(join(out, "manifest.json"), "utf8"));
  const writeManifest = (out: string, manifest: Json) =>
    writeFileSync(join(out, "manifest.json"), JSON.stringify(manifest));
  const rewriteOutcomes = (out: string, change: (record: Json) => void) => {
    const file = join(out, "outcomes.jsonl");
    const records: Json[] = readFileSync(file, "utf8")
      .trim()
      .split("\n")
      .map(line => JSON.parse(line));
    records.forEach(change);
    writeFileSync(file, `${records.map(r => JSON.stringify(r)).join("\n")}\n`);
  };
  const swapOutcomes = (from: string, into: string) =>
    copyFileSync(join(from, "outcomes.jsonl"), join(into, "outcomes.jsonl"));

  test("outcomes recorded for another Jev version are not accepted into a directory that planned this one", async () => {
    const v13 = await runFake(jev, perfect);
    const v14 = await runFake(jev, perfect, { model: "jev-1.14.0" });
    expect(() => loadRun(v13.out, dataset)).not.toThrow();
    expect(() => loadRun(v14.out, dataset)).not.toThrow();
    swapOutcomes(v13.out, v14.out);
    expect(() => loadRun(v14.out, dataset)).toThrow(
      /does not belong to its plan \(requested model\)/,
    );
    // Failures name no model; the request they answered still ties them to it.
    const failed13 = await runFake(jev, () => ({ status: 500 }));
    const failed14 = await runFake(jev, () => ({ status: 500 }), {
      model: "jev-1.14.0",
    });
    swapOutcomes(failed13.out, failed14.out);
    expect(() => loadRun(failed14.out, dataset)).toThrow(
      /does not belong to its plan \(provider request\)/,
    );
  });

  test("outcomes recorded for another provider are not accepted", async () => {
    const jevRun = await runFake(jev, perfect);
    const openaiRun = await runFake(openai, perfect);
    swapOutcomes(openaiRun.out, jevRun.out);
    expect(() => loadRun(jevRun.out, dataset)).toThrow(
      /does not belong to its plan \(requested model\)/,
    );
    const jevFailed = await runFake(jev, () => ({ status: 500 }));
    const openaiFailed = await runFake(openai, () => ({ status: 500 }));
    swapOutcomes(openaiFailed.out, jevFailed.out);
    expect(() => loadRun(jevFailed.out, dataset)).toThrow(
      /does not belong to its plan \(provider request\)/,
    );
  });

  test("a manifest must match the requests this checkout builds for its dataset, provider, model and translation", async () => {
    const fresh = async (adapter: ProviderAdapter) =>
      (await runFake(adapter, perfect)).out;
    const mutations: Array<
      [string, ProviderAdapter, (manifest: Json, out: string) => void, RegExp]
    > = [
      [
        "a provider request hash",
        openai,
        m => (m.cases[0].providerRequestSha256 = "c".repeat(64)),
        /plans requests/,
      ],
      [
        "a Jev version that was not the one requested",
        jev,
        m => (m.model = "jev-1.15.0"),
        /plans requests/,
      ],
      ["a Jev alias", jev, m => (m.model = "jev-latest"), /does not accept/],
      [
        "an OpenAI model that is not offered",
        openai,
        m => (m.model = "gpt-6-luna-fast"),
        /does not accept/,
      ],
      [
        "an older translation version",
        openai,
        m => (m.prompt.translation.version = "openai-predicate-0"),
        /translation/,
      ],
      [
        "a different translation hash",
        openai,
        m => (m.prompt.translation.sha256 = "d".repeat(64)),
        /translation/,
      ],
      [
        "a translation for a provider that sends the production body",
        jev,
        m =>
          (m.prompt.translation = {
            version: "openai-predicate-1",
            sha256: "d".repeat(64),
          }),
        /translation/,
      ],
      [
        "no translation for a provider that needs one",
        openai,
        m => (m.prompt.translation = null),
        /translation/,
      ],
      [
        "a different production question",
        jev,
        m => (m.prompt.semanticQuestionSha256 = "e".repeat(64)),
        /question/,
      ],
      [
        "a canonical hash edited together with its request set and outcomes",
        jev,
        (m, out) => {
          const forged = "f".repeat(64);
          const caseID = m.cases[0].caseID;
          m.cases[0].canonicalRequestSha256 = forged;
          m.prompt.requestSetSha256 = requestSetSha256(
            m.cases.map((entry: Json) => ({
              caseID: entry.caseID,
              sha256: entry.canonicalRequestSha256,
            })),
          );
          rewriteOutcomes(out, record => {
            if (record["caseID"] === caseID) {
              record["canonicalRequestSha256"] = forged;
            }
          });
        },
        /plans requests/,
      ],
    ];
    for (const [label, adapter, mutate, pattern] of mutations) {
      const out = await fresh(adapter);
      expect(() => loadRun(out, dataset)).not.toThrow();
      const manifest = readManifest(out);
      mutate(manifest, out);
      writeManifest(out, manifest);
      expect(() => loadRun(out, dataset), label).toThrow(pattern);
    }
  });

  test("every outcome must carry the planned model, the provider's exact answer model and the planned request", async () => {
    const scored = await runFake(openai, perfect);
    const refused = await runFake(openai, () => "refusal");
    const tampers: Array<[string, string, string, string, RegExp]> = [
      [
        "scored requested model",
        scored.out,
        "requestedModel",
        "gpt-6-luna-fast",
        /\(requested model\)/,
      ],
      [
        "scored returned model",
        scored.out,
        "returnedModel",
        "gpt-6-luna-2026-10-01",
        /\(returned model\)/,
      ],
      [
        "scored provider request",
        scored.out,
        "providerRequestSha256",
        "a".repeat(64),
        /\(provider request\)/,
      ],
      [
        "scored canonical request",
        scored.out,
        "canonicalRequestSha256",
        "a".repeat(64),
        /\(canonical request\)/,
      ],
      [
        "refusal requested model",
        refused.out,
        "requestedModel",
        "gpt-6-luna-fast",
        /\(requested model\)/,
      ],
      [
        "refusal returned model",
        refused.out,
        "returnedModel",
        "gpt-6",
        /\(returned model\)/,
      ],
    ];
    for (const [label, out, field, value, pattern] of tampers) {
      const file = join(out, "outcomes.jsonl");
      const original = readFileSync(file, "utf8");
      rewriteOutcomes(out, record => {
        record[field] = value;
      });
      expect(() => loadRun(out, dataset), label).toThrow(pattern);
      writeFileSync(file, original);
      expect(() => loadRun(out, dataset), label).not.toThrow();
    }
  });

  test("runs of known providers on the same canonical question compare, whichever translation they use", async () => {
    // Jev sends the production body and OpenAI a predicate rendering of the
    // same question. Both translations are implemented and verified when a run
    // loads, so their runs share one group; a translation this checkout does
    // not implement is refused above instead of being grouped.
    const a = await runFake(jev, perfect);
    const b = await runFake(openai, perfect);
    const report = renderReport(
      [loadRun(a.out, dataset), loadRun(b.out, dataset)],
      dataset,
    );
    expect(report).toContain("Group 1");
    expect(report).not.toContain("Group 2");
    expect(readManifest(a.out).prompt.translation).toBeNull();
    expect(readManifest(b.out).prompt.translation).toMatchObject({
      version: "openai-predicate-1",
    });
  });

  test("a reported token count at the largest safe integer is stored and loads; one above is not stored", async () => {
    const { out } = await runFake(openai, id => ({
      payload: {
        model: "gpt-6-luna",
        answers: [
          { type: "predicate", name: id, probability: keepProbability(id) },
        ],
        usage: {
          input_tokens: Number.MAX_SAFE_INTEGER,
          output_tokens: Number.MAX_SAFE_INTEGER + 1,
        },
      },
    }));
    const text = readFileSync(join(out, "outcomes.jsonl"), "utf8");
    expect(text).toContain(`"inputTokens":${Number.MAX_SAFE_INTEGER}`);
    expect(text).not.toContain("outputTokens");
    const run = loadRun(out, dataset);
    const { usage } = computeMetrics(
      run,
      new Map(cases.map(i => [i.caseID, i.keep])),
      0.8,
    );
    expect(usage).toEqual({
      eligibleResponses: cases.length,
      inputTokens: BigInt(Number.MAX_SAFE_INTEGER) * BigInt(cases.length),
      inputReporting: cases.length,
      outputTokens: null,
      outputReporting: 0,
    });
  });

  test("the report labels a usage subtotal with how many responses reported it", () => {
    const ids = cases.map(item => item.caseID);
    const run = (usages: Array<{ inputTokens?: number } | undefined>) =>
      renderReport(
        [
          syntheticRun(
            usages.map((usage, index) =>
              scoredOutcome(ids[index]!, 0.5, 1, usage),
            ),
          ),
        ],
        dataset,
      );
    const all = run([{ inputTokens: 10 }, { inputTokens: 20 }]);
    expect(all).toContain("30 (2/2)");
    expect(all).not.toContain("subtotal");
    expect(all).toContain("n/a (0/2)"); // output tokens: none reported
    const some = run([{ inputTokens: 10 }, { inputTokens: 20 }, undefined]);
    expect(some).toContain("subtotal 30 (2/3)");
    expect(run([undefined, undefined])).toContain("n/a (0/2)");
  });
});

describe("command line", () => {
  function harness(
    env: Record<string, string> = {},
    behavior?: Parameters<typeof fakeProvider>[1],
    adapter = PROVIDERS["openai-decisions"],
  ) {
    const provider = behavior ? fakeProvider(adapter, behavior) : undefined;
    const stdout: string[] = [];
    const stderr: string[] = [];
    let fetched = 0;
    const deps: CliDeps = {
      env,
      fetch: async (url, init) => {
        fetched += 1;
        if (!provider) throw new Error("network is not available");
        return provider.fetch(url, init);
      },
      stdout: text => stdout.push(text),
      stderr: text => stderr.push(text),
    };
    return { deps, stdout, stderr, fetched: () => fetched };
  }

  test("help describes commands, keys and exit codes; bad use fails with a pointer to help", async () => {
    const help = harness();
    expect(await main(["--help"], help.deps)).toBe(0);
    const text = help.stdout.join("\n");
    for (const needle of [
      "check",
      "run",
      "report",
      "TYPESAFE_API_KEY",
      "OPENAI_API_KEY",
      "gpt-6-luna",
      "--threshold",
      "Exit status",
    ]) {
      expect(text).toContain(needle);
    }
    expect(await main(["run", "--help"], harness().deps)).toBe(0);

    const bad: string[][] = [
      [],
      ["unknown"],
      ["run", "--provider", "jev", "--bogus"],
      ["run", "--provider", "nobody"],
      ["run"],
      ["run", "--provider"],
      ["run", "--provider", "jev", "extra"],
      ["check", "--provider", "jev"],
      ["report"],
      ["report", "--threshold", "0.5"],
      ["run", "--provider", "jev", "--dry-run", "--repeat", "abc"],
      ["run", "--provider", "jev", "--dry-run", "--threshold", "1.5"],
      [
        "run",
        "--provider",
        "jev",
        "--dry-run",
        "--endpoint",
        "http://example.com/x",
      ],
      [
        "run",
        "--provider",
        "jev",
        "--dry-run",
        "--endpoint",
        "https://user:pass@example.com/x",
      ],
      ["run", "--provider", "jev", "--dry-run", "--endpoint", "nope"],
    ];
    for (const argv of bad) {
      const h = harness({ TYPESAFE_API_KEY: KEY });
      expect(await main(argv, h.deps)).toBe(2);
      expect(h.stderr.join("\n")).toMatch(/Error:|Keep\/trim/);
      expect(h.fetched()).toBe(0);
    }
    const unknown = harness();
    await main(["run", "--provider", "jev", "--bogus"], unknown.deps);
    expect(unknown.stderr.join("\n")).toContain("--bogus");
    expect(unknown.stderr.join("\n")).toContain("--help");
  });

  test("check validates the dataset and identity offline", async () => {
    const h = harness();
    expect(await main(["check"], h.deps)).toBe(0);
    const text = h.stdout.join("\n");
    expect(text).toContain(FROZEN.datasetSha256);
    expect(text).toContain(FROZEN.requestSetSha256);
    expect(text).toContain(`all ${cases.length} candidates`);
    expect(h.fetched()).toBe(0);
    const bad = freshDir();
    mkdirSync(bad);
    writeFileSync(join(bad, "d.json"), '{"schemaVersion":1}');
    const failing = harness();
    expect(
      await main(["check", "--dataset", join(bad, "d.json")], failing.deps),
    ).toBe(2);
    expect(failing.stderr.join("\n")).toContain("Invalid dataset");
    expect(
      await main(
        ["check", "--dataset", join(bad, "absent.json")],
        harness().deps,
      ),
    ).toBe(2);
  });

  test("a missing or foreign key fails before any request and creates nothing", async () => {
    const envs: Array<Record<string, string>> = [
      {},
      { TYPESAFE_API_KEY: KEY },
      { OPENAI_API_KEY: "" },
    ];
    for (const env of envs) {
      const out = freshDir();
      const h = harness(env, id => keepProbability(id));
      expect(
        await main(
          ["run", "--provider", "openai-decisions", "--out", out],
          h.deps,
        ),
      ).toBe(2);
      expect(h.stderr.join("\n")).toContain("OPENAI_API_KEY is not set");
      expect(h.fetched()).toBe(0);
      expect(existsSync(out)).toBe(false);
    }
    const h = harness({ OPENAI_API_KEY: KEY }, id => keepProbability(id));
    expect(
      await main(["run", "--provider", "jev", "--out", freshDir()], h.deps),
    ).toBe(2);
    expect(h.stderr.join("\n")).toContain("TYPESAFE_API_KEY is not set");
    expect(h.fetched()).toBe(0);
  });

  test("dry run, output collisions and oversized plans make no request and need no key", async () => {
    const dry = harness();
    expect(
      await main(["run", "--provider", "jev", "--dry-run"], dry.deps),
    ).toBe(0);
    expect(dry.stdout.join("\n")).toContain(`${cases.length} requests`);
    expect(dry.stdout.join("\n")).toContain("Dry run");
    expect(dry.fetched()).toBe(0);

    const out = freshDir();
    mkdirSync(out);
    const collide = harness(
      { TYPESAFE_API_KEY: KEY },
      id => keepProbability(id),
      PROVIDERS.jev,
    );
    expect(
      await main(["run", "--provider", "jev", "--out", out], collide.deps),
    ).toBe(2);
    expect(collide.stderr.join("\n")).toContain("already exists");
    expect(collide.fetched()).toBe(0);
    expect(readdirSync(out)).toEqual([]);

    const big = harness(
      { TYPESAFE_API_KEY: KEY },
      id => keepProbability(id),
      PROVIDERS.jev,
    );
    expect(
      await main(
        [
          "run",
          "--provider",
          "jev",
          "--out",
          freshDir(),
          "--repeat",
          "2",
          "--max-requests",
          String(cases.length),
        ],
        big.deps,
      ),
    ).toBe(2);
    expect(big.stderr.join("\n")).toContain("--max-requests");
    expect(big.fetched()).toBe(0);
  });

  test("run writes records, prints the report, and report recomputes it offline", async () => {
    const out = freshDir();
    const h = harness({ OPENAI_API_KEY: KEY }, id => keepProbability(id));
    expect(
      await main(
        [
          "run",
          "--provider",
          "openai-decisions",
          "--out",
          out,
          "--repeat",
          "2",
        ],
        h.deps,
      ),
    ).toBe(0);
    const printed = h.stdout.join("\n");
    expect(printed).toContain("scored 0.93");
    expect(printed).toContain(`${cases.length * 2}/${cases.length * 2}`);
    expect(printed).not.toContain(KEY);
    expect(allFileText(out)).not.toContain(KEY);
    expect(h.fetched()).toBe(cases.length * 2);

    const again = harness();
    expect(await main(["report", out], again.deps)).toBe(0);
    expect(again.stdout.join("\n")).toContain("Group 1");
    expect(again.stdout.join("\n")).toContain("Repeats measure stability");
    expect(again.fetched()).toBe(0);
    expect(
      await main(["report", join(scratch, "nowhere")], harness().deps),
    ).toBe(2);
  });

  test("a rejected key ends the run with exit status 1 and a report of what was not attempted", async () => {
    const out = freshDir();
    const h = harness({ OPENAI_API_KEY: KEY }, () => ({ status: 401 }));
    expect(
      await main(
        [
          "run",
          "--provider",
          "openai-decisions",
          "--out",
          out,
          "--concurrency",
          "1",
        ],
        h.deps,
      ),
    ).toBe(1);
    expect(h.fetched()).toBe(1);
    expect(h.stdout.join("\n")).toContain("not attempted (stopped-after-auth)");
    expect(h.stdout.join("\n")).toContain("incomplete coverage");
  });

  test("a real process against a local server records answers without persisting the key", async () => {
    const bodies = new Map(
      cases.map(item => [
        PROVIDERS.jev.translate(buildCaseRequest(item), "jev-1.14.0"),
        item,
      ]),
    );
    const seen: Array<string | null> = [];
    let reject = false;
    const server = Bun.serve({
      port: 0,
      hostname: "127.0.0.1",
      async fetch(request) {
        seen.push(request.headers.get("authorization"));
        if (reject) return new Response(`{"error":"${KEY}"}`, { status: 401 });
        const item = bodies.get(await request.text());
        if (!item) return new Response("unknown", { status: 400 });
        return Response.json({
          model: "jev-1.14.0",
          answers: {
            candidate_0: { type: "noul", noul: item.keep ? 0.9 : 0.2 },
          },
          usage: { input_tokens: 100, output_tokens: 2 },
        });
      },
    });
    try {
      const script = fileURLToPath(
        new URL("../eval/decisions/run.ts", import.meta.url),
      );
      const spawn = async (out: string) => {
        const proc = Bun.spawn(
          [
            process.execPath,
            script,
            "run",
            "--provider",
            "jev",
            "--model",
            "jev-1.14.0",
            "--endpoint",
            `http://127.0.0.1:${server.port}/v1/systemone?token=url-secret`,
            "--out",
            out,
          ],
          {
            env: { PATH: process.env["PATH"] ?? "", TYPESAFE_API_KEY: KEY },
            stdout: "pipe",
            stderr: "pipe",
          },
        );
        const [code, stdout] = await Promise.all([
          proc.exited,
          new Response(proc.stdout).text(),
        ]);
        return { code, stdout };
      };
      const ok = freshDir();
      const first = await spawn(ok);
      expect(first.code).toBe(0);
      expect(seen.every(header => header === `Bearer ${KEY}`)).toBe(true);
      expect(seen).toHaveLength(cases.length);
      expect(first.stdout).toContain("jev-1.14.0");
      expect(first.stdout).toContain("1.000");
      expect(allFileText(ok)).not.toContain(KEY);
      expect(allFileText(ok)).not.toContain("url-secret");
      expect(readFileSync(join(ok, "manifest.json"), "utf8")).toContain(
        `"endpoint": "http://127.0.0.1:${server.port}/v1/systemone"`,
      );
      expect(first.stdout).not.toContain(KEY);

      reject = true;
      const denied = freshDir();
      const second = await spawn(denied);
      expect(second.code).toBe(1);
      expect(second.stdout).toContain("stopped-after-auth");
      expect(allFileText(denied)).not.toContain(KEY);
      expect(second.stdout).not.toContain(KEY);
    } finally {
      await server.stop(true);
    }
  });
});

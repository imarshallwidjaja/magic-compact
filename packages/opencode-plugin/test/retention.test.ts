import { describe, expect, test } from "bun:test";
import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";
import { checkNative } from "../eval/retention/run";
import type { PluginInput } from "@opencode-ai/plugin";
import type { Message, Part, ToolPart } from "@opencode-ai/sdk/v2";
import type { V2Client } from "../src/api";
import {
  buildTurns,
  nativeModelVisibleMessages,
  type MessageWithParts,
} from "../src/compact/plan";
import { trimWouldChangeContent } from "../src/compact/prune";
import server from "../src/host-v1";
import {
  decideRetentionWith,
  taskAwareRetentionPolicy,
} from "../src/policy/retention";
import {
  decideRetention,
  findRetentionCandidates,
  retentionFeedback,
} from "../src/retention/retention";
import {
  generateSessionSummary,
  isOwnedSessionSummaryFork,
  SESSION_SUMMARY_PROMPT,
  SessionSummaryError,
} from "../src/retention/summary";
import {
  summarizerAgent,
  persistedForkHistory,
  providerCatalog,
  SESSION_SUMMARY_TEXT,
  summaryReply,
  TOKEN_LIMIT_BODY,
  TOKEN_LIMIT_REQUEST_ID,
  tokenLimitResponse,
} from "./retention-host";
import {
  buildRetentionRequests,
  classifyRetention,
  excerpt,
  RETENTION_KEEP_THRESHOLD,
  RETENTION_KEEP_THRESHOLD_PERCENT_ENV,
  RETENTION_MAX_CONCURRENCY,
  RETENTION_MODEL,
  RETENTION_REQUEST_BODY_BYTE_LIMIT,
  RetentionClassificationError,
  type RetentionCandidate,
  type RetentionFetch,
} from "../src/retention/classifier";

const context = {
  latestUserRequest: "Fix the failing parser test.",
  sessionSummary: "## Objective\n- Add a parser and keep its tests passing.",
};

function candidate(id: string, output = "ok"): RetentionCandidate {
  return {
    id,
    tool: "read",
    input: { filePath: "src/parser.ts" },
    output,
    turnRequest: "Fix the failing parser test.",
    requestsBeforeLatest: 0,
  };
}

type Body = {
  model: string;
  questions: Record<string, unknown>;
};

// Answers every question with the probability `score(output)` of the single
// candidate in the request.
function jevFetch(
  score: (
    body: Body & { state: { candidates: Array<{ output: string }> } },
  ) => number | Record<string, unknown>,
  options: { model?: string; onCall?: () => void } = {},
): RetentionFetch {
  return async (_url, init) => {
    options.onCall?.();
    const body = JSON.parse(String(init.body));
    const value = score(body);
    const answers =
      typeof value === "number"
        ? Object.fromEntries(
            Object.keys(body.questions).map(key => [
              key,
              { type: "noul", noul: value },
            ]),
          )
        : value;
    return Response.json({
      model: options.model ?? RETENTION_MODEL,
      answers,
      usage: { input_tokens: 10, output_tokens: 1 },
    });
  };
}

async function failure(promise: Promise<unknown>) {
  const error = await promise.then(
    () => undefined,
    (caught: unknown) => caught,
  );
  expect(error).toBeInstanceOf(RetentionClassificationError);
  return error as RetentionClassificationError;
}

describe("frozen retention cohort parity", () => {
  // `frozenSummary` is the summary the frozen hashes were built with.
  function fixture(
    duplicateBodies = false,
    sessionSummary = context.sessionSummary,
    frozenSummary = sessionSummary,
  ) {
    const snapshots = ["first", "second"].map(snapshotID => ({
      snapshotID,
      split: "development",
      context: { ...context, sessionSummary },
      candidates: [
        candidate(`${snapshotID}#01`, "parser contents"),
        candidate(
          `${snapshotID}#02`,
          duplicateBodies ? "parser contents" : "failing test output",
        ),
      ],
    }));
    const frozen = {
      snapshots: snapshots.map(snapshot => ({
        snapshotID: snapshot.snapshotID,
        split: snapshot.split,
        cases: buildRetentionRequests(
          { ...context, sessionSummary: frozenSummary },
          snapshot.candidates,
        ).map(request => ({
          caseID: request.candidates[0]![1],
          bodySha256: createHash("sha256")
            .update(JSON.stringify(request.body))
            .digest("hex"),
          labels: { a: true, b: true, primary: true },
          scores: {},
        })),
      })),
    };
    return { input: { snapshots }, frozen };
  }

  test("rejects swapped IDs even when the snapshot body set is unchanged", async () => {
    const { input, frozen } = fixture();
    const [a, b] = input.snapshots[0]!.candidates;
    [a!.id, b!.id] = [b!.id, a!.id];
    await expect(checkNative(input, frozen)).rejects.toThrow(
      "Request body differs from frozen evidence",
    );
  });

  test("rejects bodies frozen with the 8,000-byte summary excerpt once the summary is sent whole", async () => {
    // The recorded native-summary cohort was built when the builder clipped
    // the summary to this excerpt, so a longer summary no longer matches it.
    const summary = `START${"s".repeat(10_000)}END`;
    const clipped = fixture(false, summary, excerpt(summary, 8_000));
    await expect(checkNative(clipped.input, clipped.frozen)).rejects.toThrow(
      "Request body differs from frozen evidence",
    );
    const whole = fixture(false, summary);
    await checkNative(whole.input, whole.frozen);
  });

  test("accepts whole-candidate and snapshot reordering", async () => {
    const { input, frozen } = fixture();
    input.snapshots.reverse();
    for (const snapshot of input.snapshots) snapshot.candidates.reverse();
    await checkNative(input, frozen);
  });

  test("accepts identical bodies bound to distinct case IDs", async () => {
    const { input, frozen } = fixture(true);
    expect(frozen.snapshots[0]!.cases[0]!.bodySha256).toBe(
      frozen.snapshots[0]!.cases[1]!.bodySha256,
    );
    await checkNative(input, frozen);
  });

  test.each(["missing", "foreign", "moved", "repeated", "split"])(
    "rejects %s cohort membership",
    async mutation => {
      const { input, frozen } = fixture();
      const first = input.snapshots[0]!;
      const second = input.snapshots[1]!;
      if (mutation === "missing") first.candidates.pop();
      else if (mutation === "foreign") first.candidates[0]!.id = "foreign";
      else if (mutation === "moved") {
        const moved = first.candidates.pop()!;
        second.candidates.push(moved);
      } else if (mutation === "repeated") {
        first.candidates.push(first.candidates[0]!);
      } else first.split = "heldout";
      await expect(checkNative(input, frozen)).rejects.toThrow();
    },
  );
});

describe("retention request format", () => {
  test("asks one pinned Noul per candidate without sending caller IDs", () => {
    const requests = buildRetentionRequests(context, [
      candidate("prt_secret_a"),
      candidate("prt_secret_b"),
    ]);

    expect(requests).toHaveLength(2);
    for (const request of requests) {
      expect(request.body.model).toBe(RETENTION_MODEL);
      expect(Object.keys(request.body.questions)).toEqual(["candidate_0"]);
      expect(request.body.questions["candidate_0"]!.type).toBe("noul");
      expect(request.body.questions["candidate_0"]!.instructions).toContain(
        "`candidates[0]`",
      );
      expect(request.body.state.task.latestUserRequest).toBe(
        context.latestUserRequest,
      );
      expect(JSON.stringify(request.body)).not.toContain("prt_secret");
    }
    expect(requests.map(request => request.candidates)).toEqual([
      [["candidate_0", "prt_secret_a"]],
      [["candidate_0", "prt_secret_b"]],
    ]);
  });

  test("keeps head and tail of oversized candidate content and the latest request", () => {
    const output = `HEAD${"x".repeat(200_000)}TAIL`;
    const latest = "é".repeat(50_000);
    const prompt = "p".repeat(50_000);
    const turnRequest = "t".repeat(50_000);
    const [request] = buildRetentionRequests(
      { latestUserRequest: latest, sessionSummary: "s".repeat(50_000) },
      [
        {
          ...candidate("a", output),
          input: { prompt, description: "short" },
          turnRequest,
          requestsBeforeLatest: 12,
        },
      ],
    );
    const shown = request!.body.state.candidates[0]!;

    expect(shown.output.startsWith("HEAD")).toBe(true);
    expect(shown.output.endsWith("TAIL")).toBe(true);
    expect(shown.output).toContain("characters omitted");
    expect(shown.input["description"]).toBe("short");
    // The fixed allocations are unchanged by sending the summary whole.
    expect(shown.output).toBe(excerpt(output, 9_000));
    expect(shown.turnRequest).toBe(excerpt(turnRequest, 600));
    expect(shown.input["prompt"]).toBe(excerpt(prompt, 3_000 - 7));
    expect(request!.body.state.task.latestUserRequest).toBe(
      excerpt(latest, 5_000),
    );
    expect(request!.body.state.task.sessionSummary).toBe("s".repeat(50_000));
  });

  test("never splits a surrogate pair when excerpting", () => {
    const text = "😀".repeat(5_000);
    const result = excerpt(text, 1_000);

    expect(result).not.toMatch(/[\uD800-\uDBFF](?![\uDC00-\uDFFF])/);
    expect(result).not.toMatch(/(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/);
    expect(Buffer.byteLength(JSON.stringify(result))).toBeLessThanOrEqual(
      1_000,
    );
  });

  test("sends the session summary whole, middle included, beside the latest request", () => {
    // Past the old 8,000-byte excerpt, with escapes and non-ASCII text in the
    // part that excerpt dropped.
    const middle =
      '## Current\n- Active: "quoted" C:\\path\\file\ttab é 中文 😀 \u2028 \u0000 end';
    const summary = `START${"s".repeat(6_000)}${middle}${"t".repeat(6_000)}END`;
    expect(Buffer.byteLength(JSON.stringify(summary))).toBeGreaterThan(8_000);
    const requests = buildRetentionRequests(
      { ...context, sessionSummary: summary },
      [candidate("a"), candidate("b"), candidate("c", "x".repeat(20_000))],
    );

    expect(requests).toHaveLength(3);
    for (const request of requests) {
      const task = request.body.state.task;
      expect(Object.keys(task)).toEqual([
        "latestUserRequest",
        "sessionSummary",
      ]);
      expect(task.sessionSummary).toBe(summary);
      expect(JSON.stringify(request.body)).toContain(JSON.stringify(summary));
      expect(request.body.questions["candidate_0"]!.instructions).toContain(
        "`task.sessionSummary`",
      );
    }
    const [short] = buildRetentionRequests(context, [candidate("a")]);
    expect(short!.body.state.task.sessionSummary).toBe(context.sessionSummary);
  });

  test("sends a 50,000-character summary whole, past the removed byte guards", async () => {
    const summary = `HEAD${"m".repeat(50_000)}TAIL`;
    const bodies: string[] = [];
    await classifyRetention(
      { ...context, sessionSummary: summary },
      [candidate("a"), candidate("b")],
      {
        apiKey: "key",
        fetch: async (url, init) => {
          bodies.push(String(init.body));
          return jevFetch(() => 0.5)(url, init);
        },
      },
    );

    expect(bodies).toHaveLength(2);
    for (const body of bodies) {
      // Above the removed 28,000-byte state-plus-question guard.
      expect(Buffer.byteLength(body)).toBeGreaterThan(50_000);
      expect(JSON.parse(body).state.task.sessionSummary).toBe(summary);
    }
  });

  test("sends typical content whole", () => {
    const output = `START${"o".repeat(8_000)}END`;
    const summary = `START${"s".repeat(7_000)}END`;
    const [request] = buildRetentionRequests(
      { ...context, sessionSummary: summary },
      [
        {
          ...candidate("a", output),
          input: { prompt: "p".repeat(2_500), filePath: "src/parser.ts" },
        },
      ],
    );
    const shown = request!.body.state.candidates[0]!;

    expect(shown.output).toBe(output);
    expect(shown.input).toEqual({
      prompt: "p".repeat(2_500),
      filePath: "src/parser.ts",
    });
    expect(request!.body.state.task.sessionSummary).toBe(summary);
  });

  test("rejects duplicate candidate IDs", async () => {
    const error = await failure(
      classifyRetention(context, [candidate("a"), candidate("a")], {
        apiKey: "key",
        fetch: jevFetch(() => 0.5),
      }),
    );
    expect(error.kind).toBe("configuration");
  });
});

describe("retention classification", () => {
  test("without a threshold keeps only candidates at or above the 0.60 default", async () => {
    const scores: Record<string, number> = { keep: 0.6, trim: 0.599 };
    const result = await classifyRetention(
      context,
      [candidate("a", "keep"), candidate("b", "trim")],
      {
        apiKey: "key",
        fetch: jevFetch(body => scores[body.state.candidates[0]!.output]!),
      },
    );

    expect(result.decisions.get("a")).toEqual({ useful: 0.6, keep: true });
    expect(result.decisions.get("b")).toEqual({ useful: 0.599, keep: false });
    expect(result.requests).toBe(2);
    expect(result.inputTokens).toBe(20);
  });

  test("bounds concurrency and stops admitting requests after a failure", async () => {
    let inFlight = 0;
    let maxInFlight = 0;
    let calls = 0;
    const fetchImpl: RetentionFetch = async (url, init) => {
      calls += 1;
      inFlight += 1;
      maxInFlight = Math.max(maxInFlight, inFlight);
      await Bun.sleep(5);
      inFlight -= 1;
      if (calls === RETENTION_MAX_CONCURRENCY + 1) {
        return new Response("private body", { status: 500 });
      }
      return jevFetch(() => 0.5)(url, init);
    };
    const candidates = Array.from({ length: 20 }, (_, i) => candidate(`c${i}`));

    const error = await failure(
      classifyRetention(context, candidates, {
        apiKey: "key",
        fetch: fetchImpl,
      }),
    );

    expect(error.kind).toBe("http");
    expect(error.status).toBe(500);
    expect(maxInFlight).toBe(RETENTION_MAX_CONCURRENCY);
    expect(calls).toBeLessThan(candidates.length);
    expect(inFlight).toBe(0);
  });

  const invalidResponses: Array<[string, RetentionFetch]> = [
    ["another model", jevFetch(() => 0.9, { model: "jev-1.14.0" })],
    ["a missing answer", jevFetch(() => ({}))],
    [
      "an extra answer",
      jevFetch(() => ({
        candidate_0: { type: "noul", noul: 0.9 },
        candidate_1: { type: "noul", noul: 0.9 },
      })),
    ],
    [
      "a non-Noul answer",
      jevFetch(() => ({ candidate_0: { type: "score", score: 1 } })),
    ],
    ["a probability above one", jevFetch(() => 1.5)],
    ["a NaN probability", jevFetch(() => Number.NaN)],
    ["a non-JSON body", async () => new Response("not json")],
  ];
  for (const [name, fetchImpl] of invalidResponses) {
    test(`rejects the whole invocation on ${name}`, async () => {
      const error = await failure(
        classifyRetention(context, [candidate("a"), candidate("b")], {
          apiKey: "key",
          fetch: fetchImpl,
        }),
      );
      expect(error.kind).toBe("invalid-response");
    });
  }

  test("reports timeouts and network failures without request content", async () => {
    const timeout = await failure(
      classifyRetention(context, [candidate("a", "secret output")], {
        apiKey: "sk-secret",
        timeoutMs: 5,
        fetch: (_url, init) =>
          new Promise((_, reject) =>
            init.signal!.addEventListener("abort", () =>
              reject(init.signal!.reason),
            ),
          ),
      }),
    );
    expect(timeout.kind).toBe("timeout");

    const network = await failure(
      classifyRetention(context, [candidate("a", "secret output")], {
        apiKey: "sk-secret",
        fetch: async () => {
          throw new TypeError("connect failed: secret output sk-secret");
        },
      }),
    );
    expect(network.kind).toBe("network");
    for (const error of [timeout, network]) {
      expect(error.message).not.toContain("secret");
      expect(error.cause).toBeUndefined();
      expect(error.message).toContain("/magic-trim --heuristic");
    }
  });

  test("requires an API key before any request", async () => {
    let calls = 0;
    const error = await failure(
      classifyRetention(context, [candidate("a")], {
        apiKey: "",
        fetch: jevFetch(() => 0.9, { onCall: () => (calls += 1) }),
      }),
    );
    expect(error.kind).toBe("configuration");
    expect(calls).toBe(0);
  });
});

describe("retention request body limit", () => {
  const LIMIT = RETENTION_REQUEST_BODY_BYTE_LIMIT;
  const bodyBytes = (sessionSummary: string, item: RetentionCandidate) =>
    Buffer.byteLength(
      JSON.stringify(
        buildRetentionRequests({ ...context, sessionSummary }, [item])[0]!.body,
      ),
    );
  // An ASCII summary that makes `item`'s request body exactly `target` bytes.
  const summaryFor = (item: RetentionCandidate, target: number) =>
    "s".repeat(target - bodyBytes("", item));

  test("sends a body of exactly 32 MiB, counting model, state and questions", async () => {
    const item = candidate("a");
    const sessionSummary = summaryFor(item, LIMIT);
    const sent: number[] = [];
    const result = await classifyRetention(
      { ...context, sessionSummary },
      [item],
      {
        apiKey: "key",
        fetch: async (url, init) => {
          sent.push(Buffer.byteLength(String(init.body)));
          return jevFetch(() => 0.7)(url, init);
        },
      },
    );

    expect(sent).toEqual([LIMIT]);
    expect(result.decisions.get("a")).toEqual({ useful: 0.7, keep: true });
  });

  test("refuses a body one byte over before any request, even for the last candidate", async () => {
    const late = candidate("late", "o".repeat(100));
    const sessionSummary = summaryFor(late, LIMIT + 1);
    const early = [candidate("a"), candidate("b")];
    for (const item of early) {
      expect(bodyBytes(sessionSummary, item)).toBeLessThanOrEqual(LIMIT);
    }
    let calls = 0;
    const error = await failure(
      classifyRetention({ ...context, sessionSummary }, [...early, late], {
        apiKey: "key",
        fetch: jevFetch(() => 0.5, { onCall: () => (calls += 1) }),
      }),
    );

    expect(calls).toBe(0);
    expect(error.kind).toBe("budget");
    expect(error.message).toBe(
      "A task-aware retention request is larger than TypeSafe's 32 MiB request body limit and was not sent. The original session is unchanged and no retention decisions were applied. Run /magic-trim --heuristic to trim without task-aware keep protection.",
    );
    expect(error.message).not.toContain("sss");
  });

  test("batched requests flush at the body limit and keep the task and candidate mapping", async () => {
    const items = ["c0", "c1", "c2"].map(id =>
      candidate(id, `${id} ${"o".repeat(5_000)}`),
    );
    const near = summaryFor(items[0]!, LIMIT - 1_000);
    const flushed = buildRetentionRequests(
      { ...context, sessionSummary: near },
      items,
      3,
    );
    expect(flushed.map(request => request.candidates)).toEqual([
      [["candidate_0", "c0"]],
      [["candidate_0", "c1"]],
      [["candidate_0", "c2"]],
    ]);

    const grouped = buildRetentionRequests(context, items, 3);
    expect(grouped).toHaveLength(1);
    expect(grouped[0]!.candidates).toEqual([
      ["candidate_0", "c0"],
      ["candidate_1", "c1"],
      ["candidate_2", "c2"],
    ]);
    for (const request of flushed) {
      expect(request.body.state.task).toEqual({
        ...context,
        sessionSummary: near,
      });
    }
    expect(grouped[0]!.body.state.task).toEqual(context);
    const result = await classifyRetention(context, items, {
      apiKey: "key",
      batchSize: 3,
      fetch: jevFetch(body =>
        Object.fromEntries(
          body.state.candidates.map((shown, index) => [
            `candidate_${index}`,
            { type: "noul", noul: Number(shown.output[1]) / 10 },
          ]),
        ),
      ),
    });
    expect([...result.decisions]).toEqual([
      ["c0", { useful: 0, keep: false }],
      ["c1", { useful: 0.1, keep: false }],
      ["c2", { useful: 0.2, keep: false }],
    ]);
  });
});

describe("retention service refusals", () => {
  const CANARY = "canary-request-echo-sk-secret";
  // A body stream that records whether it was read or cancelled.
  function trackedBody(
    chunks: Array<string | Uint8Array>,
    options: { hang?: boolean; fail?: boolean; cancelHangs?: boolean } = {},
  ) {
    const seen = { pulls: 0, cancelled: false };
    const encoder = new TextEncoder();
    let index = 0;
    const stream = new ReadableStream<Uint8Array>(
      {
        pull(controller) {
          seen.pulls += 1;
          if (index < chunks.length) {
            const chunk = chunks[index++]!;
            controller.enqueue(
              typeof chunk === "string" ? encoder.encode(chunk) : chunk,
            );
            return;
          }
          if (options.fail) controller.error(new Error(CANARY));
          else if (options.hang) return new Promise<void>(() => {});
          else controller.close();
        },
        cancel() {
          seen.cancelled = true;
          if (options.cancelHangs) return new Promise<void>(() => {});
        },
      },
      { highWaterMark: 0 },
    );
    return { stream, seen };
  }

  async function refusal(
    respond: () => Response,
    options: { timeoutMs?: number } = {},
  ) {
    let calls = 0;
    const error = await failure(
      classifyRetention(context, [candidate("a", CANARY)], {
        apiKey: "sk-secret",
        ...options,
        fetch: async () => {
          calls += 1;
          return respond();
        },
      }),
    );
    expect(calls).toBe(1);
    expect(error.cause).toBeUndefined();
    for (const text of [error.message, JSON.stringify(error.diagnostics)]) {
      expect(text).not.toContain("canary");
      expect(text).not.toContain("sk-secret");
    }
    return error;
  }

  test("reports Jev's recorded token-limit refusal once, with the manual fallback", async () => {
    let bodyBytes = 0;
    let calls = 0;
    const error = await failure(
      classifyRetention(context, [candidate("a")], {
        apiKey: "key",
        fetch: async (_url, init) => {
          calls += 1;
          bodyBytes = Buffer.byteLength(String(init.body));
          return tokenLimitResponse();
        },
      }),
    );

    expect(calls).toBe(1);
    expect(error.kind).toBe("token-limit");
    expect(error.status).toBe(400);
    expect(error.message).toBe(
      "TypeSafe Jev refused a task-aware retention request because it exceeds the model's token limit. The original session is unchanged and no retention decisions were applied. Run /magic-trim --heuristic to trim without task-aware keep protection.",
    );
    expect(error.diagnostics).toEqual({
      request: {
        ordinal: 0,
        bodyBytes,
        status: 400,
        requestId: TOKEN_LIMIT_REQUEST_ID,
      },
      counts: { planned: 1, started: 1, succeeded: 0, failed: 1, unstarted: 0 },
    });
  });

  test("recognizes the token-limit refusal across chunks and with extra keys it never shows", async () => {
    const body = JSON.stringify({
      detail: { error_type: "max_tokens_exceeded", message: `é ${CANARY}` },
      trace: CANARY,
    });
    const split = Buffer.from(body).indexOf(Buffer.from("é")) + 1;
    const bytes = Buffer.from(body);
    const error = await refusal(
      () =>
        new Response(
          trackedBody([bytes.subarray(0, split), bytes.subarray(split)]).stream,
          {
            status: 400,
            headers: { "x-typesafe-request-id": `req ${CANARY}` },
          },
        ),
    );
    expect(error.kind).toBe("token-limit");
    // An invalid request ID header is dropped, not shown.
    expect(error.diagnostics?.request).toEqual({
      ordinal: 0,
      bodyBytes: expect.any(Number),
      status: 400,
    });
  });

  const otherBadRequests: Array<[string, string | Uint8Array]> = [
    [
      "another error type",
      JSON.stringify({ detail: { error_type: "invalid_utf8", input: CANARY } }),
    ],
    [
      "a detail array",
      JSON.stringify({ detail: [{ error_type: "max_tokens_exceeded" }] }),
    ],
    [
      "a root array",
      JSON.stringify([{ detail: { error_type: "max_tokens_exceeded" } }]),
    ],
    ["a detail string", JSON.stringify({ detail: "max_tokens_exceeded" })],
    [
      "a root error type",
      JSON.stringify({ error_type: "max_tokens_exceeded" }),
    ],
    ["matching text that is not JSON", `max_tokens_exceeded ${CANARY}`],
    ["an empty body", ""],
    [
      "a body over 4,096 bytes",
      JSON.stringify({
        detail: { error_type: "max_tokens_exceeded" },
        pad: "x".repeat(5_000),
      }),
    ],
    [
      "invalid UTF-8",
      new Uint8Array([
        ...Buffer.from('{"detail":{"error_type":"max_tokens_exceeded","x":"'),
        0xff,
        ...Buffer.from('"}}'),
      ]),
    ],
  ];
  for (const [name, body] of otherBadRequests) {
    test(`keeps HTTP 400 with ${name} a plain failure`, async () => {
      const error = await refusal(
        () => new Response(trackedBody([body]).stream, { status: 400 }),
      );
      expect(error.kind).toBe("http");
      expect(error.status).toBe(400);
      expect(error.message).toBe(
        "Task-aware retention classification failed (http 400). No tool content was changed. Run /magic-trim --heuristic to trim without classification.",
      );
    });
  }

  test.each([401, 413, 422, 429, 500, 529])(
    "cancels an HTTP %p body without reading it",
    async status => {
      // A 422 body details the offending field, which can echo the request.
      const { stream, seen } = trackedBody([
        JSON.stringify({ detail: [{ input: CANARY }] }),
      ]);
      const error = await refusal(() => new Response(stream, { status }));
      expect(error.kind).toBe("http");
      expect(error.status).toBe(status);
      expect(seen).toEqual({ pulls: 0, cancelled: true });
    },
  );

  test.each([
    ["never finishes", { hang: true }],
    [
      "never finishes and never settles its cancellation",
      {
        hang: true,
        cancelHangs: true,
      },
    ],
    ["fails while it is read", { fail: true }],
  ] as const)(
    "keeps HTTP 400 a plain failure within the deadline when its body %s",
    async (_name, options) => {
      const { stream, seen } = trackedBody(['{"detail":'], options);
      const started = performance.now();
      const error = await refusal(() => new Response(stream, { status: 400 }), {
        timeoutMs: 50,
      });
      expect(performance.now() - started).toBeLessThan(2_000);
      expect(error.kind).toBe("http");
      expect(error.status).toBe(400);
      if (!("fail" in options)) expect(seen.cancelled).toBe(true);
    },
  );
});

describe("retention invocation outcome", () => {
  test("stops admission at failed headers while the body is still arriving and discards drained successes", async () => {
    let calls = 0;
    let callsAtBody = -1;
    const fetchImpl: RetentionFetch = async (url, init) => {
      const call = calls++;
      if (call === 0) {
        const encoder = new TextEncoder();
        const stream = new ReadableStream<Uint8Array>(
          {
            async pull(controller) {
              await Bun.sleep(60);
              callsAtBody = calls;
              controller.enqueue(encoder.encode(TOKEN_LIMIT_BODY));
              controller.close();
            },
          },
          { highWaterMark: 0 },
        );
        return new Response(stream, { status: 400 });
      }
      await Bun.sleep(5);
      return jevFetch(() => 0.9)(url, init);
    };
    const candidates = Array.from({ length: 20 }, (_, i) => candidate(`c${i}`));

    const error = await failure(
      classifyRetention(context, candidates, {
        apiKey: "key",
        fetch: fetchImpl,
      }),
    );

    expect(error.kind).toBe("token-limit");
    expect(calls).toBe(RETENTION_MAX_CONCURRENCY);
    expect(callsAtBody).toBe(RETENTION_MAX_CONCURRENCY);
    expect(error.diagnostics?.request?.ordinal).toBe(0);
    expect(error.diagnostics?.counts).toEqual({
      planned: 20,
      started: RETENTION_MAX_CONCURRENCY,
      succeeded: RETENTION_MAX_CONCURRENCY - 1,
      failed: 1,
      unstarted: 20 - RETENTION_MAX_CONCURRENCY,
    });
  });

  test("keeps the first failure when later requests fail during and after its body read", async () => {
    let calls = 0;
    const fetchImpl: RetentionFetch = async () => {
      const call = calls++;
      if (call === 0) {
        const stream = new ReadableStream<Uint8Array>(
          {
            async pull(controller) {
              await Bun.sleep(30);
              controller.enqueue(new TextEncoder().encode(TOKEN_LIMIT_BODY));
              controller.close();
            },
          },
          { highWaterMark: 0 },
        );
        return new Response(stream, { status: 400 });
      }
      // One failure lands while the body is read, one after it is read.
      if (call === 1) await Bun.sleep(60);
      throw new TypeError("connect failed");
    };

    const error = await failure(
      classifyRetention(
        context,
        Array.from({ length: 8 }, (_, i) => candidate(`c${i}`)),
        { apiKey: "key", fetch: fetchImpl },
      ),
    );

    expect(error.kind).toBe("token-limit");
    expect(error.diagnostics?.request?.ordinal).toBe(0);
    expect(error.diagnostics?.counts).toMatchObject({
      started: RETENTION_MAX_CONCURRENCY,
      succeeded: 0,
      failed: RETENTION_MAX_CONCURRENCY,
    });
  });

  test("counts only usable reported usage and leaves missing usage unknown", async () => {
    const usages: unknown[] = [
      { input_tokens: 10, output_tokens: 1 },
      undefined,
      { input_tokens: -1 },
      { input_tokens: 1.5 },
      { input_tokens: "10" },
      { output_tokens: 3 },
      { input_tokens: 0 },
    ];
    let call = 0;
    const result = await classifyRetention(
      context,
      usages.map((_, i) => candidate(`c${i}`)),
      {
        apiKey: "key",
        fetch: async () => {
          const usage = usages[call++];
          return Response.json({
            model: RETENTION_MODEL,
            answers: { candidate_0: { type: "noul", noul: 0.5 } },
            ...(usage === undefined ? {} : { usage }),
          });
        },
      },
    );
    expect(result.requests).toBe(7);
    expect(result.inputTokens).toBe(10);
    expect(result.reportedUsageRequests).toBe(2);
  });
});

describe("retention keep threshold", () => {
  const percentEnv = (percent?: string) => ({
    TYPESAFE_API_KEY: "test-key",
    ...(percent === undefined
      ? {}
      : { [RETENTION_KEEP_THRESHOLD_PERCENT_ENV]: percent }),
  });
  const thrown = (run: () => unknown) => {
    try {
      run();
    } catch (error) {
      return error;
    }
    throw new Error("Expected a throw.");
  };
  // Scores each single-candidate request by the number in its output.
  const scoredByOutput = jevFetch(body =>
    Number(body.state.candidates[0]!.output.slice(1)),
  );

  test("defaults to the operator-selected 0.60 and reads the key", () => {
    expect(RETENTION_KEEP_THRESHOLD).toBe(0.6);
    expect(RETENTION_KEEP_THRESHOLD_PERCENT_ENV).toBe(
      "MAGIC_COMPACT_KEEP_THRESHOLD_PERCENT",
    );
    expect(taskAwareRetentionPolicy(percentEnv())).toEqual({
      mode: "task-aware",
      apiKey: "test-key",
      keepThreshold: 0.6,
    });
    expect(taskAwareRetentionPolicy({ TYPESAFE_API_KEY: "" }).apiKey).toBe(
      undefined,
    );
  });

  test.each([
    ["60", 0.6],
    ["70", 0.7],
    ["80", 0.8],
    ["60.5", 0.605],
    ["60.1", 0.601],
    ["0", 0],
    ["100", 1],
    ["100.0", 1],
    ["060", 0.6],
    [" 70\n", 0.7],
  ])("reads %p percent as probability %p", (percent, expected) => {
    expect(taskAwareRetentionPolicy(percentEnv(percent)).keepThreshold).toBe(
      expected,
    );
  });

  test.each([
    "",
    "  ",
    "abc",
    "NaN",
    "Infinity",
    "-1",
    "-0",
    "+60",
    "101",
    "100.5",
    "60%",
    "0x3c",
    "6e1",
    ".5",
    "5.",
    "60 70",
    "6,5",
    "\u0666\u0660",
    "9".repeat(400),
  ])("rejects percent %p without falling back to the default", percent => {
    const error = thrown(() => taskAwareRetentionPolicy(percentEnv(percent)));
    expect(error).toBeInstanceOf(RetentionClassificationError);
    expect((error as RetentionClassificationError).kind).toBe(
      "invalid-threshold-percent",
    );
    const message = (error as Error).message;
    expect(message).toContain(RETENTION_KEEP_THRESHOLD_PERCENT_ENV);
    expect(message).toContain("0 to 100");
    expect(message).toContain("/magic-trim --heuristic");
  });

  test("the rejection never echoes the value and the key is not read first", () => {
    let keyReads = 0;
    const error = thrown(() =>
      taskAwareRetentionPolicy({
        get TYPESAFE_API_KEY() {
          keyReads += 1;
          return "sk-secret";
        },
        [RETENTION_KEEP_THRESHOLD_PERCENT_ENV]: "canary-77%",
      }),
    );
    expect((error as Error).message).not.toMatch(/canary|77|sk-secret/);
    expect(keyReads).toBe(0);
  });

  test.each([
    [
      undefined,
      [
        [0.599, false],
        [0.6, true],
        [0.601, true],
      ],
    ],
    [
      "80",
      [
        [0.799, false],
        [0.8, true],
      ],
    ],
    [
      "0",
      [
        [0, true],
        [0.3, true],
      ],
    ],
    [
      "100",
      [
        [0.999, false],
        [1, true],
      ],
    ],
  ] as const)(
    "keeps at or above percent %p and leaves the probabilities unchanged",
    async (percent, scores) => {
      const { keepThreshold } = taskAwareRetentionPolicy(percentEnv(percent));
      const result = await classifyRetention(
        context,
        scores.map(([score], index) => candidate(`c${index}`, `s${score}`)),
        { apiKey: "key", keepThreshold, fetch: scoredByOutput },
      );
      scores.forEach(([score, keep], index) => {
        expect(result.decisions.get(`c${index}`)).toEqual({
          useful: score,
          keep,
        });
      });
    },
  );

  test("the threshold is never part of the Jev request", async () => {
    const sent = async (keepThreshold?: number) => {
      const bodies: string[] = [];
      await classifyRetention(context, [candidate("a"), candidate("b")], {
        apiKey: "key",
        ...(keepThreshold === undefined ? {} : { keepThreshold }),
        fetch: async (url, init) => {
          bodies.push(String(init.body));
          return jevFetch(() => 0.7)(url, init);
        },
      });
      return bodies;
    };
    expect(await sent(0.8)).toEqual(await sent());
    expect(await sent(0)).toEqual(await sent());
  });

  test.each([Number.NaN, -0.1, 1.01, Infinity, -Infinity])(
    "rejects explicit threshold %p before any request",
    async keepThreshold => {
      let calls = 0;
      const error = await failure(
        classifyRetention(context, [candidate("a")], {
          apiKey: "key",
          keepThreshold,
          fetch: jevFetch(() => 0.9, { onCall: () => (calls += 1) }),
        }),
      );
      expect(error.kind).toBe("configuration");
      expect(calls).toBe(0);
    },
  );

  test("the policy rejects an invalid threshold before the session summary", async () => {
    let summaries = 0;
    const error = await failure(
      decideRetentionWith(
        { latestUserRequest: "", candidates: [candidate("a")] },
        { mode: "task-aware", apiKey: "key", keepThreshold: Number.NaN },
        async () => {
          summaries += 1;
          return "summary";
        },
      ),
    );
    expect(error.kind).toBe("configuration");
    expect(summaries).toBe(0);
  });

  test.each([
    [undefined, ["a"]],
    ["60", ["a"]],
    ["80", []],
  ])(
    "a 0.7 probability follows the configured percent %p",
    async (percent, kept) => {
      const plan = await decideRetentionWith(
        { latestUserRequest: "Ship it.", candidates: [candidate("a")] },
        {
          ...taskAwareRetentionPolicy(percentEnv(percent)),
          fetch: jevFetch(() => 0.7),
        },
        async () => "summary",
      );
      expect(plan.evaluated).toBe(1);
      expect([...plan.keep]).toEqual(kept);
    },
  );
});

describe("retention keep threshold in the OpenCode 1 server environment", () => {
  const command = (name: string, args = "") => ({
    command: name,
    arguments: args,
    sessionID: "session",
  });
  const withPercent = async <T>(value: string, run: () => Promise<T>) => {
    const previous = process.env[RETENTION_KEEP_THRESHOLD_PERCENT_ENV];
    process.env[RETENTION_KEEP_THRESHOLD_PERCENT_ENV] = value;
    try {
      return await run();
    } finally {
      if (previous === undefined) {
        delete process.env[RETENTION_KEEP_THRESHOLD_PERCENT_ENV];
      } else process.env[RETENTION_KEEP_THRESHOLD_PERCENT_ENV] = previous;
    }
  };

  test.each(["magic-compact", "magic-trim"])(
    "%s rejects an invalid percent before it contacts the server",
    async name => {
      const toasts: Array<{ title: string; message: string }> = [];
      const client = {
        tui: {
          showToast: async (options: {
            body: { title: string; message: string };
          }) => {
            toasts.push(options.body);
            return { data: true };
          },
        },
      };
      const hooks = await server({ client } as unknown as PluginInput);
      const error = await withPercent("60%", () =>
        promiseError(
          hooks["command.execute.before"]!(command(name), { parts: [] }),
        ),
      );
      expect(error).toBeInstanceOf(RetentionClassificationError);
      expect(String(error)).toContain(RETENTION_KEEP_THRESHOLD_PERCENT_ENV);
      // The operator sees the same fixed message in the TUI.
      expect(toasts).toMatchObject([
        {
          title:
            name === "magic-compact"
              ? "Magic Compact Failed"
              : "Magic Trim Failed",
          message: (error as Error).message,
          variant: "error",
        },
      ]);
    },
  );

  test.each([
    ["magic-trim", "--heuristic"],
    ["magic-stats", ""],
  ])("%s %s neither reads nor validates the percent", async (name, args) => {
    const hooks = await server({} as PluginInput);
    const error = await withPercent("60%", () =>
      promiseError(
        hooks["command.execute.before"]!(command(name, args), { parts: [] }),
      ),
    );
    // The command proceeds to the listener, which this input cannot supply.
    expect(error).not.toBeInstanceOf(RetentionClassificationError);
    expect(String(error)).not.toContain(RETENTION_KEEP_THRESHOLD_PERCENT_ENV);
  });
});

describe("native-summary evaluation metrics", () => {
  test("keep the recorded 0.80 threshold, not the runtime default", () => {
    const run = Bun.spawnSync([
      process.execPath,
      fileURLToPath(new URL("../eval/retention/run.ts", import.meta.url)),
      "metrics",
    ]);
    expect(run.exitCode).toBe(0);
    const [development, heldout] = JSON.parse(
      `[${run.stdout.toString().replace(/\}\s*\n\{/u, "},{")}]`,
    ) as Array<{ split: string; majority: Record<string, number> }>;
    expect(development).toMatchObject({
      split: "development",
      majority: { TP: 24, FP: 0, FN: 13, TN: 17 },
    });
    expect(heldout).toMatchObject({
      split: "heldout",
      majority: { TP: 9, FP: 1, FN: 19, TN: 37 },
    });
  });
});

describe("retention feedback", () => {
  test("aggregates kept tool names, orders ties lexically and caps the breakdown at three", () => {
    const names = [
      "task",
      "read",
      "hive_status",
      "read",
      "task",
      "bash",
      "skill",
      "read",
    ];
    const parts = names.map((name, index) =>
      tool(
        name,
        { secret: "hidden-input" },
        "hidden-output",
        false,
        `kept-${index}`,
      ),
    );
    const messages = [
      assistantMessage("assistant", [
        ...parts,
        tool("excluded", {}, "", false, "not-kept"),
      ]),
    ];
    const plan = {
      mode: "task-aware" as const,
      evaluated: 12,
      keep: new Set(parts.map(part => part.id)),
    };
    const feedback = retentionFeedback(plan, messages);
    expect(feedback).toBe(
      "Task-aware retention kept 8 of 12 evaluated tool call(s) whole (read 3, task 2, bash 1, +2 other calls).",
    );
    expect(
      retentionFeedback(plan, [
        assistantMessage("assistant", [...messages[0]!.parts].reverse()),
      ]),
    ).toBe(feedback);
    expect(feedback).not.toMatch(/hidden|excluded|kept-/);
  });

  test("bounds long labels and removes whitespace, control and formatting characters", () => {
    const names = [
      "custom".repeat(100),
      "\n\tread\u001b\u202e tool\r",
      "\u0000\n",
    ];
    const parts = names.map((name, index) =>
      tool(name, {}, "", false, `kept-${index}`),
    );
    const feedback = retentionFeedback(
      {
        mode: "task-aware",
        evaluated: 3,
        keep: new Set(parts.map(part => part.id)),
      },
      [assistantMessage("assistant", parts)],
    );
    expect(feedback).toContain("customcustomcustomcus... 1");
    expect(feedback).toContain("read tool 1");
    expect(feedback).toContain("unnamed 1");
    expect(feedback).not.toMatch(/[\p{C}]/u);
    expect(feedback.length).toBeLessThan(180);

    const astralPart = tool(
      "a".repeat(20) + "😀" + "bbbb",
      {},
      "",
      false,
      "kept-astral",
    );
    const astralFeedback = retentionFeedback(
      {
        mode: "task-aware",
        evaluated: 1,
        keep: new Set([astralPart.id]),
      },
      [assistantMessage("assistant", [astralPart])],
    );
    const label = astralFeedback.match(/whole \((.*) 1\)\.$/u)![1]!;
    expect(label).toBe("a".repeat(20) + "😀...");
    expect(label.isWellFormed()).toBeTrue();
    expect(Array.from(label)).toHaveLength(24);
  });
});

describe("retention candidates", () => {
  const long = "word ".repeat(300);
  const cases: Array<[string, ToolPart, boolean]> = [
    ["a read output", tool("read", {}, "file"), true],
    ["a short bash call", tool("bash", { command: "ls" }, "ok"), false],
    [
      "a long bash command",
      tool("bash", { command: "x".repeat(1_100) }, "ok"),
      true,
    ],
    ["a large write", tool("write", { content: long }, "ok"), true],
    [
      "a large apply_patch",
      tool("apply_patch", { patchText: long }, "ok"),
      true,
    ],
    [
      "a large edit",
      tool("edit", { oldString: long, newString: "b" }, "ok"),
      true,
    ],
    ["a large task prompt", tool("task", { prompt: long }, "done"), true],
    ["a short task", tool("task", { prompt: "go" }, "done"), false],
    [
      "a marked historical task placeholder",
      tool(
        "task",
        {
          prompt:
            "[Historical task prompt omitted; Content ID: 2c4f6a8b0d1e:omitted-yH8kQ2pL5vN7sR4tU6wXzA. Retrieve the original with read_omitted_content before reuse.]",
        },
        long,
        true,
      ),
      false,
    ],
    [
      "marked compact task prompt and report references",
      tool(
        "task",
        {
          prompt:
            "[Historical task prompt omitted; Content ID: 2c4f6a8b0d1e:omitted-yH8kQ2pL5vN7sR4tU6wXzA. Retrieve before reuse.]",
          note: "[Omitted; Content ID: 2c4f6a8b0d1e:omitted-yH8kQ2pL5vN7sR4tU6wXzA]",
        },
        "[Task output omitted (4900 bytes); child session: ses_2f3a9bC1; Content ID: 2c4f6a8b0d1e:omitted-yH8kQ2pL5vN7sR4tU6wXzA]",
        true,
      ),
      false,
    ],
    [
      "marked compact edit references",
      tool(
        "edit",
        {
          oldString:
            "[Omitted with newString; Content ID: 2c4f6a8b0d1e:omitted-yH8kQ2pL5vN7sR4tU6wXzA]",
          newString:
            "[Omitted with oldString; Content ID: 2c4f6a8b0d1e:omitted-yH8kQ2pL5vN7sR4tU6wXzA]",
        },
        "ok",
        true,
      ),
      false,
    ],
    [
      "a marked bare task placeholder",
      tool("task", { prompt: "[Omitted]" }, long, true),
      false,
    ],
    ["a question", tool("question", {}, long), false],
    [
      "a todo list already at its replacement",
      tool("todowrite", {}, "Successfully updated todos."),
      false,
    ],
    ["a todo list", tool("todowrite", {}, "- [ ] item"), true],
    ["a marked read", tool("read", {}, "file", true), false],
    [
      "a marked task with a large prompt",
      tool("task", { prompt: long }, "done", true),
      true,
    ],
    [
      "a running call",
      {
        ...tool("read", {}, "x"),
        state: { status: "running", input: {}, time: { start: 1 } },
      } as ToolPart,
      false,
    ],
  ];
  for (const [name, part, expected] of cases) {
    test(`${expected ? "includes" : "excludes"} ${name} without changing it`, async () => {
      const before = structuredClone(part);
      expect(await trimWouldChangeContent(part)).toBe(expected);
      expect(part).toEqual(before);
    });
  }

  test("maps request context from ordinary user text and synthetic continuations", async () => {
    const messages = [
      userMessage("u1", "Add the parser."),
      assistantMessage("a1", [tool("read", {}, "first", false, "t1")]),
      userMessage("u2", "task finished", true),
      assistantMessage("a2", [tool("read", {}, "second", false, "t2")]),
      userMessage("u3", "Fix the tests."),
      assistantMessage("a3", [tool("read", {}, "third", false, "t3")]),
      userMessage("u4", "Ship it."),
    ];
    const turns = buildTurns(messages).slice(0, 3);

    const found = await findRetentionCandidates(messages, turns);

    expect(found.latestUserRequest).toBe("Ship it.");
    expect(
      found.candidates.map(item => [
        item.id,
        item.turnRequest,
        item.requestsBeforeLatest,
      ]),
    ).toEqual([
      ["t1", "Add the parser.", 2],
      ["t2", "Add the parser.", 2],
      ["t3", "Fix the tests.", 1],
    ]);
  });

  test("makes no host or classifier request without candidates", async () => {
    let calls = 0;
    const v2 = new Proxy({}, { get: () => (calls += 1) }) as V2Client;
    const plan = await decideRetention(
      v2,
      session(),
      [],
      { latestUserRequest: "", candidates: [] },
      {
        mode: "task-aware",
        apiKey: undefined,
        fetch: async () => {
          calls += 1;
          return new Response();
        },
      },
      { controlTimeoutMs: 1_000 },
    );
    expect(plan).toEqual({ mode: "task-aware", evaluated: 0, keep: new Set() });
    expect(calls).toBe(0);
  });

  test("requires the API key before any summary request", async () => {
    let calls = 0;
    const v2 = new Proxy({}, { get: () => (calls += 1) }) as V2Client;
    const error = await failure(
      decideRetention(
        v2,
        session(),
        [],
        { latestUserRequest: "", candidates: [candidate("a")] },
        { mode: "task-aware", apiKey: undefined },
        { controlTimeoutMs: 1_000 },
      ),
    );
    expect(error.kind).toBe("missing-api-key");
    expect(error.message).toContain("TYPESAFE_API_KEY");
    expect(calls).toBe(0);
  });
});

describe("retention session summary", () => {
  test("preserves persisted source chronology when JSON creation times disagree", async () => {
    const source = sourceHistory();
    source[0]!.info.time = { created: 7 };
    source[1]!.info.time = { created: 1 };
    source[2]!.info.time = { created: 8 };
    const host = summaryHost(source);
    const before = structuredClone(source);
    try {
      await expect(
        generateSessionSummary(host.v2, session(), source),
      ).resolves.toBe(SESSION_SUMMARY_TEXT);
      expect(host.promptHistories[0]!.map(item => item.info.role)).toEqual([
        "user",
        "assistant",
        "user",
      ]);
      expect(source).toEqual(before);
    } finally {
      host.history.close();
    }
  });

  test("retains sanitized partial-fork cleanup guidance when chronology replay fails", async () => {
    const source = sourceHistory();
    source.forEach((item, index) => {
      item.info.time = { created: index === 0 ? 7 : index };
    });
    const host = summaryHost(source);
    host.v2.sync.replay = async () => {
      throw new Error("private replay body");
    };
    host.deleteError = new Error("private delete body");
    try {
      const error = await promiseError(
        generateSessionSummary(host.v2, session(), source),
      );
      expect(String(error)).toContain("(history replay failed)");
      expect(String(error)).toContain("delete partial fork fork-1 manually");
      expect(String(error)).toContain("/magic-trim --heuristic");
      expect(String(error)).not.toContain("private");
      expect(host.prompts).toEqual([]);
      expect(host.calls).toEqual(["fork:source", "delete:fork-1"]);
    } finally {
      host.history.close();
    }
  });
  test("recovers one failed fork history read and names a persistent failure", async () => {
    const stall = (
      host: ReturnType<typeof summaryHost>,
      failures: number,
    ): void => {
      const read = host.v2.session.messages.bind(host.v2.session);
      let seen = 0;
      host.v2.session.messages = (async (request: { sessionID: string }) => {
        if (request.sessionID !== "source" && (seen += 1) <= failures)
          throw new Error("private read body");
        return read(request as never);
      }) as never;
    };
    const recovered = summaryHost(sourceHistory());
    stall(recovered, 1);
    await expect(
      generateSessionSummary(recovered.v2, session(), recovered.source),
    ).resolves.toBe(SESSION_SUMMARY_TEXT);
    expect(recovered.calls.filter(call => call.startsWith("fork:"))).toEqual([
      "fork:source",
    ]);

    const failed = summaryHost(sourceHistory());
    stall(failed, 2);
    const error = await promiseError(
      generateSessionSummary(failed.v2, session(), failed.source),
    );
    expect((error as SessionSummaryError).stage).toBe("history");
    expect(String(error)).toContain("(history: history read failed)");
    expect(String(error)).not.toContain("private");
    expect(failed.prompts).toEqual([]);
    expect(failed.calls).toEqual(["fork:source", "delete:fork-1"]);
  });

  test("prompts one deny-all fork of the whole session through the summarizer agent", async () => {
    const host = summaryHost(sourceHistory());
    let ownedDuringPrompt = false;
    host.onPrompt = async forkID => {
      ownedDuringPrompt = isOwnedSessionSummaryFork(forkID);
    };

    const summary = await generateSessionSummary(
      host.v2,
      session(),
      host.source,
    );

    expect(summary).toBe(SESSION_SUMMARY_TEXT);
    expect(host.permissions).toEqual([
      [{ permission: "*", pattern: "*", action: "deny" }],
    ]);
    const prompt = host.prompts[0]!;
    expect(prompt).toMatchObject({
      sessionID: "fork-1",
      agent: "magic-compact-summarizer",
      model: summarizerAgent().model,
      variant: "low",
      parts: [{ type: "text", text: SESSION_SUMMARY_PROMPT }],
    });
    expect(String(prompt["messageID"])).toMatch(
      /^msg_[0-9a-f]{12}[0-9A-Za-z]{14}$/u,
    );
    expect(prompt).not.toHaveProperty("format");
    expect(SESSION_SUMMARY_PROMPT).toStartWith(
      "Create a new anchored summary from the entire conversation history above, including the most recent messages, so another coding agent can continue the work.\n\nOutput exactly the Markdown structure",
    );
    expect(ownedDuringPrompt).toBe(true);
    expect(isOwnedSessionSummaryFork("fork-1")).toBe(false);
    expect(host.calls).toEqual(["fork:source", "delete:fork-1"]);
    expect(host.source).toEqual(sourceHistory());
  });

  // OpenCode runs this hook before any native compaction request, replay,
  // or auto-continue; the thrown error ends the fork's prompt loop, and the
  // host then fails or returns the prompt call.
  test.each(["fails", "returns"] as const)(
    "blocks native compaction of an owned fork when the prompt then %s",
    async outcome => {
      const host = summaryHost(sourceHistory());
      const hooks = await server({} as PluginInput);
      const compacting = (sessionID: string) =>
        hooks["experimental.session.compacting"]!(
          { sessionID },
          { context: [], prompt: undefined },
        );
      let hookError: unknown;
      host.onPrompt = async forkID => {
        hookError = await promiseError(compacting(forkID));
        if (outcome === "fails") throw new Error("host defect with private");
      };

      const error = await promiseError(
        generateSessionSummary(host.v2, session(), host.source),
      );

      expect(hookError).toBeInstanceOf(SessionSummaryError);
      expect(error).toBeInstanceOf(SessionSummaryError);
      expect((error as SessionSummaryError).stage).toBe("budget");
      expect(String(error)).toContain("too large");
      expect(String(error)).not.toContain("private");
      expect(host.calls).toEqual([
        "fork:source",
        "abort:fork-1",
        "delete:fork-1",
      ]);
      expect(isOwnedSessionSummaryFork("fork-1")).toBe(false);
      await expect(compacting("fork-1")).resolves.toBeUndefined();
      await expect(compacting("source")).resolves.toBeUndefined();
    },
  );

  test.each([
    [
      "a context overflow",
      { name: "ContextOverflowError", data: { message: "private body" } },
      "provider",
      "too large",
    ],
    [
      "an API error",
      {
        name: "APIError",
        data: { message: "private body", statusCode: 429 },
      },
      "provider 429",
      "/magic-trim --heuristic",
    ],
  ])(
    "reports %s on a 200 response as a provider failure",
    async (_name, responseError, detail, text) => {
      const host = summaryHost(sourceHistory());
      host.afterPrompt = fork => {
        (fork.at(-1)!.info as { error?: unknown }).error = responseError;
      };

      const error = await promiseError(
        generateSessionSummary(host.v2, session(), host.source),
      );

      expect((error as SessionSummaryError).stage).toBe("provider");
      expect(String(error)).toContain(text);
      if (detail !== "provider") expect(String(error)).toContain(detail);
      expect(String(error)).not.toContain("private");
      expect(host.calls).toEqual([
        "fork:source",
        "abort:fork-1",
        "delete:fork-1",
      ]);
    },
  );

  test("sees history the way OpenCode sends it after a native checkpoint", () => {
    const history = checkpointHistory({ tail: true });
    expect(
      nativeModelVisibleMessages(history).map(message => message.info.id),
    ).toEqual(["msg_c1", "msg_s1", "msg_u2", "msg_a2", "msg_u3"]);
    expect(
      nativeModelVisibleMessages(checkpointHistory({ tail: false })).map(
        message => message.info.id,
      ),
    ).toEqual(["msg_c1", "msg_s1", "msg_u3"]);
    expect(nativeModelVisibleMessages(sourceHistory())).toEqual(
      sourceHistory(),
    );
  });

  test("budgets and gates images on the history after a native checkpoint", async () => {
    const limit = { context: 60_000, output: 8_000 };
    const hidden = summaryHost(
      checkpointHistory({ tail: true, hiddenImage: true }),
      { limit, image: false },
    );
    await expect(
      generateSessionSummary(hidden.v2, session(), hidden.source),
    ).resolves.toBe(SESSION_SUMMARY_TEXT);
    expect(hidden.calls).toEqual(["fork:source", "delete:fork-1"]);

    const visible = summaryHost(
      checkpointHistory({ tail: true, tailImage: true }),
      { limit, image: false },
    );
    const error = await promiseError(
      generateSessionSummary(visible.v2, session(), visible.source),
    );
    expect(String(error)).toContain("image-capable");
    expect(visible.calls).toEqual([]);
  });

  // Expected membership/order from filterCompacted at 830d5eb535. Its final
  // reorder selects the last tail marker independently of summary completion.
  test.each([
    [
      "failed no-tail only",
      false,
      "error",
      undefined,
      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    ],
    [
      "successful then failed no-tail",
      true,
      "error",
      undefined,
      [3, 4, 5, 6, 7, 8, 9, 10],
    ],
    ["failed tail only", false, "error", "m5", [7, 8, 5, 6, 9, 10]],
    ["successful then failed tail", true, "error", "m5", [7, 8, 5, 6, 9, 10]],
    [
      "successful then incomplete tail",
      true,
      "incomplete",
      "m5",
      [7, 8, 5, 6, 9, 10],
    ],
    [
      "two completed crossing-tail",
      true,
      "completed",
      "m1",
      [7, 8, 1, 2, 3, 4, 5, 6, 9, 10],
    ],
  ] as const)(
    "mirrors native history and permits summary generation: %s",
    async (_name, firstCompleted, secondStatus, tail, expected) => {
      const history = Array.from({ length: 10 }, (_, index) => {
        const id = `m${index + 1}`;
        return index % 2 === 0
          ? userMessage(id, "Continue parser work.")
          : assistantMessage(id, []);
      });
      for (const [markerIndex, status, tailStartID] of [
        [2, firstCompleted ? "completed" : "ordinary", undefined],
        [6, secondStatus, tail],
      ] as const) {
        if (status === "ordinary") continue;
        const marker = history[markerIndex]!;
        marker.parts = [
          {
            id: `${marker.info.id}_compaction`,
            sessionID: "source",
            messageID: marker.info.id,
            type: "compaction",
            auto: true,
            ...(tailStartID ? { tail_start_id: tailStartID } : {}),
          } as Part,
        ];
        const summary = history[markerIndex + 1]!;
        summary.info = {
          ...summary.info,
          parentID: marker.info.id,
          summary: true,
          ...(status === "incomplete" ? {} : { finish: "stop" }),
          ...(status === "error"
            ? {
                error: {
                  name: "MessageAbortedError",
                  data: { message: "Aborted" },
                },
              }
            : {}),
        } as Message;
      }
      const before = structuredClone(history);
      const view = nativeModelVisibleMessages(history);
      expect(view.map(message => message.info.id)).toEqual(
        expected.map(index => `m${index}`),
      );
      for (const message of view) {
        expect(message).toBe(
          history.find(item => item.info.id === message.info.id)!,
        );
      }
      expect(history).toEqual(before);
      const host = summaryHost(history);
      await expect(
        generateSessionSummary(host.v2, session(), history),
      ).resolves.toBe(SESSION_SUMMARY_TEXT);
      expect(host.calls).toEqual(["fork:source", "delete:fork-1"]);
      expect(history).toEqual(before);
    },
  );

  test("budgets and gates images after an errored tail drops the prior checkpoint", async () => {
    const history = checkpointHistory({ tail: true, tailImage: true });
    const priorTailTool = history[3]!.parts[0] as ToolPart;
    if (priorTailTool.state.status === "completed") {
      priorTailTool.state.output = "x ".repeat(80_000);
    }
    const marker = structuredClone(history[4]!);
    marker.info.id = "msg_c2";
    marker.parts = [
      {
        ...marker.parts[0]!,
        id: "prt_c2",
        messageID: "msg_c2",
        tail_start_id: "msg_u3",
      } as Part,
    ];
    const summary = structuredClone(history[5]!);
    summary.info = {
      ...summary.info,
      id: "msg_s2",
      parentID: "msg_c2",
      error: { name: "MessageAbortedError", data: { message: "Aborted" } },
    } as Message;
    summary.parts = [];
    history.push(marker, summary, userMessage("msg_u4", "Continue."));
    expect(
      nativeModelVisibleMessages(history).map(message => message.info.id),
    ).toEqual(["msg_c2", "msg_s2", "msg_u3", "msg_u4"]);
    const catalog = { limit: { context: 60_000, output: 8_000 }, image: false };
    const hidden = summaryHost(history, catalog);
    await expect(
      generateSessionSummary(hidden.v2, session(), history),
    ).resolves.toBe(SESSION_SUMMARY_TEXT);
    expect(hidden.calls).toEqual(["fork:source", "delete:fork-1"]);

    history[6]!.parts.push({
      ...history[2]!.parts.find(part => part.type === "file")!,
      id: "prt_visible_image",
      messageID: "msg_u3",
    } as Part);
    const visible = summaryHost(history, catalog);
    const error = await promiseError(
      generateSessionSummary(visible.v2, session(), history),
    );
    expect(String(error)).toContain("image-capable");
    expect(visible.calls).toEqual([]);
  });

  test.each(["completed", "error", "incomplete"])(
    "fails before any fork on a dangling selected %s tail",
    async status => {
      const history = checkpointHistory({ tail: true });
      const marker = history.find(message => message.info.id === "msg_c1")!;
      (marker.parts[0] as { tail_start_id?: string }).tail_start_id =
        "msg_gone";
      const summary = history[5]!.info as { finish?: string; error?: unknown };
      if (status === "error") summary.error = { name: "MessageAbortedError" };
      if (status === "incomplete") delete summary.finish;
      const host = summaryHost(history);

      const error = await promiseError(
        generateSessionSummary(host.v2, session(), host.source),
      );

      expect((error as SessionSummaryError).stage).toBe("history");
      expect(host.calls).toEqual([]);
    },
  );

  // Native compaction artifacts mean the request did not fit, even when the
  // compacting hook did not run.
  const nativeArtifacts: Array<
    [string, (fork: MessageWithParts[], id: string) => void]
  > = [
    [
      "a native compaction checkpoint",
      (fork, id) => {
        fork.splice(fork.length - 1, 0, {
          info: { id: `${id}-marker`, sessionID: id, role: "user" } as Message,
          parts: [
            {
              id: `${id}-compaction`,
              sessionID: id,
              messageID: `${id}-marker`,
              type: "compaction",
              auto: true,
            } as Part,
          ],
        });
      },
    ],
    [
      "a native summary reply",
      fork => {
        (fork.at(-1)!.info as { summary?: boolean }).summary = true;
      },
    ],
  ];
  for (const [name, mutate] of nativeArtifacts) {
    test(`reports ${name} as too large, aborts, and deletes the fork`, async () => {
      const host = summaryHost(sourceHistory());
      host.afterPrompt = mutate;

      const error = await promiseError(
        generateSessionSummary(host.v2, session(), host.source),
      );

      expect((error as SessionSummaryError).stage).toBe("budget");
      expect(String(error)).toContain("too large");
      expect(host.calls).toEqual([
        "fork:source",
        "abort:fork-1",
        "delete:fork-1",
      ]);
      expect(isOwnedSessionSummaryFork("fork-1")).toBe(false);
    });
  }

  const rejected: Array<
    [string, (fork: MessageWithParts[], id: string) => void]
  > = [
    [
      "a replayed prompt",
      (fork, id) => {
        fork.at(-2)!.info.id = `${id}-replayed`;
        for (const part of fork.at(-2)!.parts)
          part.messageID = `${id}-replayed`;
      },
    ],
    [
      "an unfinished reply",
      fork => {
        (fork.at(-1)!.info as { finish?: string }).finish = "length";
      },
    ],
    [
      "an empty reply",
      fork => {
        fork.at(-1)!.parts = fork
          .at(-1)!
          .parts.filter(part => part.type !== "text");
      },
    ],
    [
      "a tool call",
      (fork, id) => {
        fork.at(-1)!.parts.push({
          id: `${id}-tool`,
          sessionID: id,
          messageID: fork.at(-1)!.info.id,
          type: "tool",
        } as Part);
      },
    ],
    [
      "an auto-continue follow-up",
      (fork, id) => {
        fork.push({
          info: {
            id: `${id}-continue`,
            sessionID: id,
            role: "user",
          } as Message,
          parts: [],
        });
      },
    ],
    [
      "changed history",
      fork => {
        fork.splice(0, 1);
      },
    ],
  ];
  for (const [name, mutate] of rejected) {
    test(`rejects ${name}, aborts, and deletes the fork`, async () => {
      const host = summaryHost(sourceHistory());
      host.afterPrompt = mutate;

      const error = await promiseError(
        generateSessionSummary(host.v2, session(), host.source),
      );

      expect(error).toBeInstanceOf(SessionSummaryError);
      expect((error as SessionSummaryError).stage).toBe("validation");
      expect(String(error)).toContain("/magic-trim --heuristic");
      expect(host.calls).toEqual([
        "fork:source",
        "abort:fork-1",
        "delete:fork-1",
      ]);
      expect(isOwnedSessionSummaryFork("fork-1")).toBe(false);
    });
  }

  test("fails before any fork when the history cannot fit the compaction model", async () => {
    const host = summaryHost(sourceHistory("x ".repeat(80_000)), {
      limit: { context: 60_000, output: 8_000 },
    });

    const error = await promiseError(
      generateSessionSummary(host.v2, session(), host.source),
    );

    expect((error as SessionSummaryError).stage).toBe("budget");
    expect(String(error)).toContain("too large");
    expect(host.calls).toEqual([]);
  });

  test("reports sanitized provider and cleanup failures together", async () => {
    const host = summaryHost(sourceHistory());
    host.promptError = Object.assign(
      new Error("raw provider body with secret"),
      {
        status: 503,
      },
    );
    host.deleteError = new Error("raw delete body");

    const error = await promiseError(
      generateSessionSummary(host.v2, session(), host.source),
    );

    expect(error).toBeInstanceOf(AggregateError);
    const errors = (error as AggregateError).errors as SessionSummaryError[];
    expect(errors.map(item => [item.stage, item.status])).toEqual([
      ["provider", 503],
      ["delete", undefined],
    ]);
    expect(JSON.stringify(errors.map(String))).not.toContain("raw");
    expect(String(error)).toContain("/magic-trim --heuristic");
    expect(host.calls).toEqual([
      "fork:source",
      "abort:fork-1",
      "delete:fork-1",
    ]);
  });
});

function tool(
  name: string,
  input: Record<string, unknown>,
  output: string,
  marked = false,
  id = `tool_${name}`,
): ToolPart {
  return {
    id,
    sessionID: "source",
    messageID: "assistant",
    type: "tool",
    callID: `call_${id}`,
    tool: name,
    state: {
      status: "completed",
      input,
      output,
      title: name,
      metadata: marked ? { magicCompact: { trimmed: true } } : {},
      time: { start: 1, end: 2 },
    },
  };
}

// The host stamps every stored message with a creation time.
const created = () => ({ created: 1 });

function userMessage(
  id: string,
  text: string,
  synthetic = false,
): MessageWithParts {
  return {
    info: { id, sessionID: "source", role: "user", time: created() } as Message,
    parts: [
      {
        id: `${id}_text`,
        sessionID: "source",
        messageID: id,
        type: "text",
        text,
        ...(synthetic ? { synthetic: true } : {}),
      },
    ],
  };
}

function assistantMessage(id: string, parts: Part[]): MessageWithParts {
  return {
    info: {
      id,
      sessionID: "source",
      role: "assistant",
      time: created(),
    } as Message,
    parts: parts.map(part => ({ ...part, messageID: id })),
  };
}

function sourceHistory(output = "listing"): MessageWithParts[] {
  return [
    userMessage("msg_u1", "Add the parser."),
    assistantMessage("msg_a1", [tool("read", {}, output, false, "prt_t1")]),
    userMessage("msg_u2", "Fix the tests."),
  ];
}

// A completed native checkpoint whose summary replaces a large prefix and
// whose tail starts at msg_u2.
function checkpointHistory(options: {
  tail: boolean;
  hiddenImage?: boolean;
  tailImage?: boolean;
}): MessageWithParts[] {
  const image = (messageID: string): Part => ({
    id: `${messageID}_image`,
    sessionID: "source",
    messageID,
    type: "file",
    mime: "image/png",
    url: "data:image/png;base64,iVBORw0KGgo=",
  });
  const withImage = (message: MessageWithParts, add?: boolean) =>
    add
      ? { ...message, parts: [...message.parts, image(message.info.id)] }
      : message;
  return [
    withImage(userMessage("msg_u1", "Start the parser."), options.hiddenImage),
    assistantMessage("msg_a1", [
      tool("read", {}, "x ".repeat(80_000), false, "prt_hidden"),
    ]),
    withImage(userMessage("msg_u2", "Add tests."), options.tailImage),
    assistantMessage("msg_a2", [tool("read", {}, "tail", false, "prt_tail")]),
    {
      info: {
        id: "msg_c1",
        sessionID: "source",
        role: "user",
        time: created(),
      } as Message,
      parts: [
        {
          id: "prt_c1",
          sessionID: "source",
          messageID: "msg_c1",
          type: "compaction",
          auto: true,
          ...(options.tail ? { tail_start_id: "msg_u2" } : {}),
        } as Part,
      ],
    },
    {
      info: {
        id: "msg_s1",
        sessionID: "source",
        role: "assistant",
        time: created(),
        parentID: "msg_c1",
        summary: true,
        finish: "stop",
      } as Message,
      parts: [
        {
          id: "prt_s1",
          sessionID: "source",
          messageID: "msg_s1",
          type: "text",
          text: "## Objective\n- Parser work.",
        },
      ],
    },
    userMessage("msg_u3", "Fix the tests."),
  ];
}

function session() {
  return { id: "source", directory: "/workspace" } as never;
}

async function promiseError(promise: Promise<unknown>): Promise<unknown> {
  return promise.then(
    () => {
      throw new Error("Expected a rejection.");
    },
    (error: unknown) => error,
  );
}

function summaryHost(
  source: MessageWithParts[],
  catalog: {
    limit?: { context: number; input?: number; output?: number };
    image?: boolean;
  } = {},
) {
  let forkCount = 0;
  const history = persistedForkHistory(source);
  const host = {
    history,
    promptHistories: [] as MessageWithParts[][],
    source,
    calls: [] as string[],
    prompts: [] as Record<string, unknown>[],
    permissions: [] as unknown[],
    onPrompt: undefined as ((forkID: string) => Promise<void>) | undefined,
    afterPrompt: undefined as
      ((fork: MessageWithParts[], forkID: string) => void) | undefined,
    promptError: undefined as unknown,
    deleteError: undefined as unknown,
    v2: undefined as unknown as V2Client,
  };
  host.v2 = {
    app: { agents: async () => ({ data: [summarizerAgent()] }) },
    provider: {
      list: async () => ({
        data: providerCatalog(catalog.limit, catalog.image),
      }),
    },
    session: {
      fork: async (request: { sessionID: string }) => {
        host.calls.push(`fork:${request.sessionID}`);
        const id = `fork-${++forkCount}`;
        history.fork(
          request.sessionID,
          id,
          (request as { messageID?: string }).messageID,
        );
        return { data: { id, directory: "/workspace" } };
      },
      update: async (request: { permission?: unknown }) => {
        host.permissions.push(request.permission);
        return { data: {} };
      },
      messages: async (request: { sessionID: string }) => ({
        data: history.read(request.sessionID),
      }),
      prompt: async (request: Record<string, unknown>) => {
        host.prompts.push(request);
        const forkID = String(request["sessionID"]);
        await host.onPrompt?.(forkID);
        if (host.promptError) throw host.promptError;
        const fork = history.read(forkID);
        host.promptHistories.push(structuredClone(fork));
        const reply = summaryReply(fork, request as never, forkID);
        host.afterPrompt?.(fork, forkID);
        // The host persists the reply and any tampering made after it.
        history.delete(forkID);
        fork.forEach((item, index) => history.write(item, index));
        return reply;
      },
      abort: async (request: { sessionID: string }) => {
        host.calls.push(`abort:${request.sessionID}`);
        return { data: true };
      },
      delete: async (request: { sessionID: string }) => {
        host.calls.push(`delete:${request.sessionID}`);
        if (host.deleteError) throw host.deleteError;
        history.delete(request.sessionID);
        return { data: true };
      },
    },
    sync: {
      replay: async (request: {
        events: Parameters<typeof history.replay>[0];
      }) => {
        history.replay(request.events);
        return { data: { sessionID: request.events[0]!.aggregateID } };
      },
    },
  } as unknown as V2Client;
  return host;
}

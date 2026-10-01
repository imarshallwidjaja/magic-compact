import { describe, expect, test } from "bun:test";
import { createHash } from "node:crypto";
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
import server from "../src/index";
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
  compactionAgent,
  providerCatalog,
  rewriteFork,
  SESSION_SUMMARY_TEXT,
  summaryReply,
} from "./retention-host";
import {
  buildRetentionRequests,
  classifyRetention,
  excerpt,
  RETENTION_KEEP_THRESHOLD,
  RETENTION_MAX_CONCURRENCY,
  RETENTION_MODEL,
  RETENTION_STATE_AND_QUESTION_BYTE_LIMIT,
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
  function fixture(duplicateBodies = false) {
    const snapshots = ["first", "second"].map(snapshotID => ({
      snapshotID,
      split: "development",
      context,
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
        cases: buildRetentionRequests(context, snapshot.candidates).map(
          request => ({
            caseID: request.candidates[0]![1],
            bodySha256: createHash("sha256")
              .update(JSON.stringify(request.body))
              .digest("hex"),
            labels: { a: true, b: true, primary: true },
            scores: {},
          }),
        ),
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

  test("keeps head and tail of oversized content within the byte budget", () => {
    const output = `HEAD${"x".repeat(200_000)}TAIL`;
    const [request] = buildRetentionRequests(
      {
        latestUserRequest: "é".repeat(50_000),
        sessionSummary: "s".repeat(50_000),
      },
      [
        {
          ...candidate("a", output),
          input: { prompt: "p".repeat(50_000), description: "short" },
          turnRequest: "t".repeat(50_000),
          requestsBeforeLatest: 12,
        },
      ],
    );
    const shown = request!.body.state.candidates[0]!;

    expect(shown.output.startsWith("HEAD")).toBe(true);
    expect(shown.output.endsWith("TAIL")).toBe(true);
    expect(shown.output).toContain("characters omitted");
    expect(shown.input["description"]).toBe("short");
    const bytes =
      Buffer.byteLength(JSON.stringify(request!.body.state))
      + Buffer.byteLength(JSON.stringify(request!.body.questions));
    expect(bytes).toBeLessThanOrEqual(RETENTION_STATE_AND_QUESTION_BYTE_LIMIT);
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

  test("sends one bounded session summary beside the latest request", () => {
    const [request] = buildRetentionRequests(
      { ...context, sessionSummary: `START${"s".repeat(20_000)}END` },
      [candidate("a")],
    );
    const task = request!.body.state.task;

    expect(Object.keys(task)).toEqual(["latestUserRequest", "sessionSummary"]);
    expect(task.sessionSummary.startsWith("START")).toBe(true);
    expect(task.sessionSummary.endsWith("END")).toBe(true);
    expect(Buffer.byteLength(JSON.stringify(task.sessionSummary))).toBe(8_000);
    expect(request!.body.questions["candidate_0"]!.instructions).toContain(
      "`task.sessionSummary`",
    );
    const [short] = buildRetentionRequests(context, [candidate("a")]);
    expect(short!.body.state.task.sessionSummary).toBe(context.sessionSummary);
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
  test("keeps only candidates at or above the calibrated threshold", async () => {
    const scores: Record<string, number> = {
      keep: RETENTION_KEEP_THRESHOLD,
      trim: RETENTION_KEEP_THRESHOLD - 0.001,
    };
    const result = await classifyRetention(
      context,
      [candidate("a", "keep"), candidate("b", "trim")],
      {
        apiKey: "key",
        fetch: jevFetch(body => scores[body.state.candidates[0]!.output]!),
      },
    );

    expect(result.decisions.get("a")).toEqual({
      useful: RETENTION_KEEP_THRESHOLD,
      keep: true,
    });
    expect(result.decisions.get("b")!.keep).toBe(false);
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
  test("prompts one deny-all fork of the whole session through the native compaction agent", async () => {
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
      agent: "compaction",
      model: compactionAgent().model,
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

function userMessage(
  id: string,
  text: string,
  synthetic = false,
): MessageWithParts {
  return {
    info: { id, sessionID: "source", role: "user" } as Message,
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
    info: { id, sessionID: "source", role: "assistant" } as Message,
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
      info: { id: "msg_c1", sessionID: "source", role: "user" } as Message,
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
  const forks = new Map<string, MessageWithParts[]>();
  const host = {
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
    app: { agents: async () => ({ data: [compactionAgent()] }) },
    provider: {
      list: async () => ({
        data: providerCatalog(catalog.limit, catalog.image),
      }),
    },
    session: {
      fork: async (request: { sessionID: string }) => {
        host.calls.push(`fork:${request.sessionID}`);
        const id = `fork-${forks.size + 1}`;
        forks.set(id, rewriteFork(source, id));
        return { data: { id, directory: "/workspace" } };
      },
      update: async (request: { permission?: unknown }) => {
        host.permissions.push(request.permission);
        return { data: {} };
      },
      messages: async (request: { sessionID: string }) => ({
        data: structuredClone(forks.get(request.sessionID)),
      }),
      prompt: async (request: Record<string, unknown>) => {
        host.prompts.push(request);
        const forkID = String(request["sessionID"]);
        await host.onPrompt?.(forkID);
        if (host.promptError) throw host.promptError;
        const fork = forks.get(forkID)!;
        const reply = summaryReply(fork, request as never, forkID);
        host.afterPrompt?.(fork, forkID);
        return reply;
      },
      abort: async (request: { sessionID: string }) => {
        host.calls.push(`abort:${request.sessionID}`);
        return { data: true };
      },
      delete: async (request: { sessionID: string }) => {
        host.calls.push(`delete:${request.sessionID}`);
        if (host.deleteError) throw host.deleteError;
        forks.delete(request.sessionID);
        return { data: true };
      },
    },
  } as unknown as V2Client;
  return host;
}

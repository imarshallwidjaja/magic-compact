import { afterAll, beforeAll, describe, expect, jest, test } from "bun:test";
import { mkdir, mkdtemp, rm } from "node:fs/promises";
import { createHash } from "node:crypto";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import type { Config, PluginInput, ToolContext } from "@opencode-ai/plugin";
import type {
  Agent,
  Message,
  Part,
  Session,
  TextPart,
  ToolPart,
} from "@opencode-ai/sdk/v2";
import type { V2Client } from "../src/api";
import { encode } from "gpt-tokenizer";
import {
  generateCompactionSummaries,
  injectSummaries,
  MAX_SUMMARY_INPUT_TOKENS,
  SUMMARY_REQUEST_TIMEOUT_MS,
} from "../src/compact/compact";
import { MAX_SUMMARY_OUTPUT_TOKENS } from "../src/compact/constants";
import {
  DETERMINISTIC_SUMMARY_MAX_TOKENS,
  encodeOrderedLiteralRuns,
  renderDeterministicSummary,
  renderStandardTurnEvidence,
} from "../src/compact/high-risk";
import {
  assertCompactionSourceUnchanged,
  createCompactionPlan,
  NativeCheckpointChangedError,
  type MessageWithParts,
  type Turn,
} from "../src/compact/plan";
import { pruneSummarizedTurns } from "../src/compact/prune";
import { buildCompactionPrompt } from "../src/compact/template";
import { applyBackup, createBackup } from "../src/compact/session";
import { executeMagicCompact } from "../src/magic-compact";
import server from "../src/index";
import {
  cachePath,
  readOmittedContent,
  writeCache,
} from "../src/storage/omission";
import {
  deleteStats,
  readStats,
  type ConversationStats,
  writeStats,
} from "../src/storage/stats";
import {
  statsMessage,
  statsSummaryMessage,
  trimStatsMessage,
} from "../src/stats/constants";
import {
  countCanonicalContentTokens,
  countSessionTokens,
} from "../src/stats/tokenize";
import {
  INCIDENT_ASSISTANT_REPEAT_COUNT,
  INCIDENT_ASSISTANT_REPEAT_LITERAL,
  INCIDENT_DETERMINISTIC_TURN_COUNT,
  INCIDENT_ERROR_TOOL_COUNT,
  INCIDENT_RUNNING_TOOL_COUNT,
  INCIDENT_STANDARD_TURN_COUNT,
  INCIDENT_TOOL_COUNT,
  INCIDENT_TURN_COUNT,
  incidentSessionFixture,
} from "./fixtures/incident-session";

let storageDirectory: string;
const originalDataHome = process.env.XDG_DATA_HOME;

beforeAll(async () => {
  storageDirectory = await mkdtemp(join(tmpdir(), "magic-compact-test-"));
  process.env.XDG_DATA_HOME = storageDirectory;
});

afterAll(async () => {
  if (originalDataHome === undefined) {
    delete process.env.XDG_DATA_HOME;
  } else {
    process.env.XDG_DATA_HOME = originalDataHome;
  }
  await rm(storageDirectory, { recursive: true, force: true });
});

describe("magic compact", () => {
  test("registers the hidden Magic Compact summarizer with its exact permission policy", async () => {
    const hooks = await server({} as PluginInput);
    const config = {} as Config;

    await hooks.config?.(config);

    expect(config.agent?.["magic-compact-summarizer"]).toMatchObject({
      description: "Faithful per-turn plain-JSON summarizer for Magic Compact",
      mode: "subagent",
      hidden: true,
    });
    expect(config.agent?.["magic-compact-summarizer"]?.permission).toEqual({
      "*": "deny",
    });
    expect(config.agent?.["magic-compact-summarizer"]?.prompt).toContain(
      "bounded exact evidence map in each request is authoritative",
    );
    expect(config.agent?.["magic-compact-summarizer"]?.prompt).toContain(
      "continue unfinished work",
    );
    expect(config.agent?.["magic-compact-summarizer"]?.prompt).toContain(
      "Reply only with the JSON object required by each Magic Compact request",
    );
  });

  test("clamps only existing positive finite summarizer limits at the provider boundary", async () => {
    const hooks = await server({} as PluginInput);
    const applyParams = hooks["chat.params"];
    expect(applyParams).toBeDefined();

    const absentLimit = chatParams(undefined);
    await applyParams!(
      chatParamsInput("scratch-1", "magic-compact-summarizer"),
      absentLimit,
    );
    expect(absentLimit.maxOutputTokens).toBeUndefined();

    const higherLimit = chatParams(8_192);
    await applyParams!(
      chatParamsInput("scratch-1", "magic-compact-summarizer"),
      higherLimit,
    );
    expect(higherLimit.maxOutputTokens).toBe(MAX_SUMMARY_OUTPUT_TOKENS);

    const lowerLimit = chatParams(1_500);
    await applyParams!(
      chatParamsInput("scratch-2", "magic-compact-summarizer"),
      lowerLimit,
    );
    expect(lowerLimit.maxOutputTokens).toBe(1_500);

    for (const current of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
      const invalidLimit = chatParams(current);
      await applyParams!(
        chatParamsInput("scratch-3", "magic-compact-summarizer"),
        invalidLimit,
      );
      expect(invalidLimit.maxOutputTokens).toBe(current);
    }

    for (const [sessionID, agent] of [
      ["source", "build"],
      ["scratch-1", "compaction"],
    ] as const) {
      const ordinary = chatParams(8_192);
      await applyParams!(chatParamsInput(sessionID, agent), ordinary);
      expect(ordinary.maxOutputTokens).toBe(8_192);
    }

    const options = { custom: { seed: 7 } };
    const withOptions = chatParams(8_192, options);
    await applyParams!(
      chatParamsInput("scratch-4", "magic-compact-summarizer"),
      withOptions,
    );
    expect(withOptions.maxOutputTokens).toBe(MAX_SUMMARY_OUTPUT_TOKENS);
    expect(withOptions.options).toBe(options);
  });

  test("uses the native compaction model and variant with the custom agent", async () => {
    const requests = await runCompaction(
      {
        ...session("source"),
        agent: "build",
        model: {
          providerID: "provider",
          id: "model",
          variant: "fast",
        },
        permission: [{ permission: "read", pattern: "*", action: "allow" }],
      },
      {
        agents: [
          compactionAgent({
            model: {
              providerID: "compaction-provider",
              modelID: "compaction-model",
            },
            variant: "precise",
          }),
        ],
      },
    );

    expect(requests.agentRequests).toEqual([
      { directory: "/workspace", workspace: "workspace" },
    ]);
    expect(requests.updates).toEqual([]);
    expect(requests.creates).toEqual([
      {
        directory: "/workspace",
        workspace: "workspace",
        workspaceID: "workspace",
        title: "[TEMP] Test session",
        permission: [{ permission: "*", pattern: "*", action: "deny" }],
      },
    ]);
    expect(requests.prompts[0]).toMatchObject({
      sessionID: "scratch-1",
      agent: "magic-compact-summarizer",
      model: {
        providerID: "compaction-provider",
        modelID: "compaction-model",
      },
      variant: "precise",
    });
    expect(requests.prompts[0]).not.toHaveProperty("format");
  });

  test("falls back to the captured source model when compaction has no model", async () => {
    const requests = await runCompaction({
      ...session("source"),
      model: {
        providerID: "source-provider",
        id: "source-model",
        variant: "fast",
      },
    });

    expect(requests.prompts[0]).toMatchObject({
      agent: "magic-compact-summarizer",
      model: {
        providerID: "source-provider",
        modelID: "source-model",
      },
      variant: "fast",
    });
  });

  test("fails clearly and deletes the temporary session when compaction is unavailable", async () => {
    const lifecycle: string[] = [];

    await expect(
      runCompaction(session("source"), { agents: [], lifecycle }),
    ).rejects.toThrow("native hidden compaction agent");

    expect(lifecycle).toEqual(["messages", "create", "agents", "delete"]);
  });

  test("rejects invalid explicit compaction model configuration", async () => {
    await expect(
      runCompaction(session("source"), {
        agents: [
          compactionAgent({
            model: { providerID: "", modelID: "compaction-model" },
          }),
        ],
      }),
    ).rejects.toThrow("invalid model configuration");
  });

  test("does not route omission reads from transcript-free scratch sessions", async () => {
    const contentID = "source:omitted-AAAAAAAAAAAAAAAAAAAAAA";
    await writeCache("source", v2Cache(contentID, "historical output"));
    let contentDuringPrompt: string | null = null;

    await runCompaction(session("source"), {
      onPrompt: async () => {
        contentDuringPrompt = await readOmittedContent("scratch-1", contentID);
      },
    });

    expect(contentDuringPrompt).toBeNull();
    expect(await readOmittedContent("scratch-1", contentID)).toBeNull();
  });

  test("keeps omission reads unavailable when scratch summarization fails", async () => {
    const contentID = "source:omitted-BBBBBBBBBBBBBBBBBBBBBB";
    await writeCache("source", v2Cache(contentID, "historical output"));
    let contentDuringPrompt: string | null = null;

    await expect(
      runCompaction(session("source"), {
        onPrompt: async () => {
          contentDuringPrompt = await readOmittedContent(
            "scratch-1",
            contentID,
          );
        },
        promptError: new Error("summary failed"),
      }),
    ).rejects.toThrow("summary failed");

    expect(contentDuringPrompt).toBeNull();
    expect(await readOmittedContent("scratch-1", contentID)).toBeNull();
  });

  test("requests strict plain JSON without a format property while preserving summarizer constraints", async () => {
    const requests = await runCompaction(session("source"));
    const prompt = requests.prompts[0]?.parts as Array<{ text: string }>;
    const promptText = prompt[0]?.text ?? "";

    expect(requests.prompts[0]).not.toHaveProperty("format");
    expect(promptText).toContain('"assistantTurnID": "assistant"');
    expect(promptText).toContain('"userPreview": "Request\\n..."');
    expect(promptText).toContain('"stopAnchor": null');
    expect(promptText).toContain('Required output keys: ["assistant"]');
    expect(promptText).toContain("Return only one JSON object");
    expect(promptText).toContain(
      "Do not return Markdown fences, prose, XML, or a nested wrapper",
    );
    expect(promptText).not.toContain("<summary>");
    expect(promptText).toContain(
      "No tools or external inspection are available",
    );
    expect(promptText).toContain('"exactEvidence":');
    expect(promptText).toContain(
      "Preserve failed, not-run, running, and unresolved state exactly",
    );
    expect(promptText).toContain("human-readable continuation capsule");
    expect(promptText).toContain(
      "Correctness and continuation fidelity come before brevity",
    );
    expect(promptText).toContain("exact next action");
    expect(promptText).toContain(
      "without restating tool structure or evidence the transcript already keeps",
    );
    expect(requests.partUpdates[0]).toMatchObject({
      part: {
        text: modelCapsule("analysis-only", "Completed the request."),
        metadata: { magicCompact: { summary: true } },
      },
    });
  });

  test("maps summaries by stable turn ID and accepts response reordering", async () => {
    const messages = [
      ...completedTurn("first", "One"),
      ...completedTurn("second", "Two"),
    ];
    const requests = await runCompaction(session("source"), {
      messages,
      textResponses: [
        JSON.stringify({
          "second-assistant": modelCapsule("analysis-only", "Second result."),
          "first-assistant": modelCapsule("analysis-only", "First result."),
        }),
      ],
    });

    expect(
      requests.partUpdates.map(request => (request.part as TextPart).text),
    ).toEqual([
      modelCapsule("analysis-only", "First result."),
      modelCapsule("analysis-only", "Second result."),
    ]);
  });

  test("reconstructs split plain-JSON responses by exact part concatenation", async () => {
    const technical =
      'Compared <left> & <right>.\nRan `bun test "packages/unit suite.test.ts"`.\nUpdated C:\\work\\app and /workspace/app/[slug]/page.tsx, then returned \ud83d\ude80 and \u4f60\u597d.';
    const capsule = modelCapsule("analysis-only", technical);
    const encoded = JSON.stringify({ assistant: capsule });
    const splits = [
      encoded.indexOf("\\n") + 1,
      encoded.indexOf('\\"') + 1,
      encoded.indexOf("\\\\") + 1,
      encoded.indexOf("\u4f60"),
      encoded.length - 3,
    ];
    for (const split of splits) {
      const requests = await runCompaction(session("source"), {
        textResponses: [[encoded.slice(0, split), encoded.slice(split)]],
      });
      expect((requests.partUpdates[0]!.part as TextPart).text).toBe(capsule);
    }
  });

  test("rejects invalid plain-JSON summary contracts before source writes", async () => {
    const invalid: Array<{
      output: string | undefined;
      category: string;
    }> = [
      { output: undefined, category: "missing-text-output" },
      { output: "", category: "missing-text-output" },
      { output: "not private summary content JSON", category: "invalid-json" },
      {
        output: '```json\n{"assistant":"Private summary content."}\n```',
        category: "invalid-json",
      },
      {
        output: '<summary>{"assistant":"Private summary content."}</summary>',
        category: "invalid-json",
      },
      {
        output: 'Here is the summary: {"assistant":"Private summary content."}',
        category: "invalid-json",
      },
      { output: "null", category: "invalid-output-shape" },
      { output: "[]", category: "invalid-output-shape" },
      {
        output: JSON.stringify({ assistant: 42 }),
        category: "invalid-summary-type",
      },
      { output: "{}", category: "missing-id" },
      {
        output: JSON.stringify({
          unknown: "Unexpected private summary content.",
        }),
        category: "unknown-id",
      },
      {
        output: JSON.stringify({
          assistant: "Completed the requested work.",
          unknown: "Unexpected private summary content.",
        }),
        category: "unknown-id",
      },
      {
        output: JSON.stringify({ assistant: " " }),
        category: "empty-summary",
      },
      {
        output: JSON.stringify({
          assistant: "oversized ".repeat(MAX_SUMMARY_OUTPUT_TOKENS + 1),
        }),
        category: "response-too-large",
      },
      ...[
        "[Replace with summary]",
        "Replace with your summary...",
        "[TODO]",
        "[TBD]",
        "[TODO: add evidence]",
        "[placeholder for outcome]",
        "Details [Replace with summary] remain.",
        "TODO",
        "TBD",
        "unknown",
        "N/A",
        "yes",
        "none",
        "Done.",
        "Implemented",
      ].map(output => ({
        output: JSON.stringify({ assistant: output }),
        category: "placeholder-summary",
      })),
    ];

    for (const { output, category } of invalid) {
      const partUpdates: Record<string, unknown>[] = [];
      let error: unknown;
      try {
        await runCompaction(session("source"), {
          messages: compactableMessages(),
          textResponses: [output, output],
          partUpdates,
        });
      } catch (caught) {
        error = caught;
      }
      expect(error).toMatchObject({
        name: "SummaryContractError",
        category,
        batch: 1,
        batchCount: 1,
        attempt: 2,
      });
      expect(String(error)).toContain(
        `category=${category}, batch=1/1, attempt=2/2`,
      );
      expect(String(error)).not.toContain("private summary content");
      expect(String(error)).not.toContain("Private summary content");
      expect(partUpdates).toEqual([]);
    }
  });

  test.each([
    ["Completed the request.", "invalid-capsule-shape"],
    [
      "Outcome: completed\nWork: Claimed completion without source evidence.",
      "outcome-mismatch",
    ],
    [
      '{"kind":"assistant-text","sourceBytes":42,"windows":["raw"]}',
      "machine-evidence-summary",
    ],
    [
      'Outcome: analysis-only\nWork: {"sourceBytes":42,"windows":["raw"]}',
      "machine-evidence-summary",
    ],
    [
      'Outcome: analysis-only\nWork: sourceBytes=42; windows=["raw"]',
      "machine-evidence-summary",
    ],
    [
      'Outcome: analysis-only\nWork: "kind": "tool"',
      "machine-evidence-summary",
    ],
    [
      "Outcome: analysis-only\nWork: sourceBytes = 128",
      "machine-evidence-summary",
    ],
    [
      "Outcome: analysis-only\nWork: Reviewed the request.\n- omitted-record-count=2",
      "machine-evidence-summary",
    ],
    [
      "Outcome: analysis-only\nWork: Reviewed the request.\nevidence-records\nomitted-record-count=2",
      "machine-evidence-summary",
    ],
    [
      "Outcome: analysis-only\nWork: Reviewed the work.\nVerification:",
      "invalid-capsule-shape",
    ],
  ])(
    "rejects a visible model summary outside the readable capsule contract: %s",
    async (summary, category) => {
      const partUpdates: Record<string, unknown>[] = [];
      let error: unknown;
      try {
        await runCompaction(session("source"), {
          textResponses: [
            JSON.stringify({ assistant: summary }),
            JSON.stringify({ assistant: summary }),
          ],
          partUpdates,
        });
      } catch (caught) {
        error = caught;
      }

      expect(error).toMatchObject({
        name: "SummaryContractError",
        category,
        attempt: 2,
      });
      expect(String(error)).not.toContain(summary);
      expect(partUpdates).toEqual([]);
    },
  );

  test("rejects a model Outcome that contradicts completed source evidence", async () => {
    let error: unknown;
    try {
      await runCompaction(session("source"), {
        messages: completedCommandTurn("bun test", "10 passed, 0 failed"),
        textResponses: [
          JSON.stringify({
            "metric-assistant-0":
              "Outcome: analysis-only\nWork: Reviewed the historical request.",
          }),
          JSON.stringify({
            "metric-assistant-0":
              "Outcome: analysis-only\nWork: Reviewed the historical request.",
          }),
        ],
      });
    } catch (caught) {
      error = caught;
    }

    expect(error).toMatchObject({
      name: "SummaryContractError",
      category: "outcome-mismatch",
      attempt: 2,
    });
  });

  test("does not retry a contract failure when scratch cleanup fails", async () => {
    const lifecycle: string[] = [];
    let error: unknown;

    try {
      await runCompaction(session("source"), {
        textResponses: [JSON.stringify({ assistant: "[TODO]" })],
        deleteError: new Error("scratch cleanup failed"),
        deleteErrorAt: 1,
        lifecycle,
      });
    } catch (caught) {
      error = caught;
    }

    expect(error).toBeInstanceOf(AggregateError);
    expect(nestedErrorMessages(error)).toEqual(
      expect.arrayContaining([
        expect.stringContaining("category=placeholder-summary"),
        "scratch cleanup failed",
      ]),
    );
    expect(lifecycle.filter(event => event === "prompt")).toHaveLength(1);
    expect(lifecycle.filter(event => event === "create")).toHaveLength(1);
    expect(lifecycle.filter(event => event === "delete")).toHaveLength(1);
  });

  test("does not retry provider, body-info, or SDK failures", async () => {
    for (const name of ["ProviderAuthError", "StructuredOutputError"]) {
      const providerLifecycle: string[] = [];
      await expect(
        runCompaction(session("source"), {
          promptInfoErrors: [
            {
              name,
              data: { providerID: "provider", message: "provider failed" },
            },
          ],
          lifecycle: providerLifecycle,
        }),
      ).rejects.toThrow(name);
      expect(
        providerLifecycle.filter(event => event === "prompt"),
      ).toHaveLength(1);
    }

    const sdkLifecycle: string[] = [];
    await expect(
      runCompaction(session("source"), {
        promptError: new Error("prompt transport failed"),
        lifecycle: sdkLifecycle,
      }),
    ).rejects.toThrow("prompt transport failed");
    expect(sdkLifecycle.filter(event => event === "prompt")).toHaveLength(1);

    for (const failure of [
      new Error("summary timed out"),
      new DOMException("summary cancelled", "AbortError"),
    ]) {
      const lifecycle: string[] = [];
      await expect(
        runCompaction(session("source"), {
          promptError: failure,
          lifecycle,
        }),
      ).rejects.toThrow(failure.message);
      expect(lifecycle.filter(event => event === "prompt")).toHaveLength(1);
      expect(lifecycle.filter(event => event === "delete")).toHaveLength(1);
    }
  });

  test("aborts the server-side scratch runner before delete when the client request times out", async () => {
    jest.useFakeTimers();
    const lifecycle: string[] = [];
    const abortRequests: Record<string, unknown>[] = [];
    const deleteRequests: Record<string, unknown>[] = [];
    let promptSignal: AbortSignal | undefined;
    try {
      const compaction = runCompaction(session("source"), {
        lifecycle,
        abortRequests,
        deleteRequests,
        onPrompt: async signal => {
          promptSignal = signal;
          await new Promise<never>((_resolve, reject) => {
            signal?.addEventListener("abort", () => reject(signal.reason), {
              once: true,
            });
          });
        },
      });
      for (
        let index = 0;
        index < 20 && !lifecycle.includes("prompt");
        index += 1
      ) {
        await Promise.resolve();
      }
      expect(lifecycle).toContain("prompt");

      jest.advanceTimersByTime(SUMMARY_REQUEST_TIMEOUT_MS);

      await expect(compaction).rejects.toThrow("Summary request timed out.");
      expect(promptSignal).toBeInstanceOf(AbortSignal);
      expect(promptSignal?.aborted).toBeTrue();
      expect(abortRequests).toEqual([
        {
          sessionID: "scratch-1",
          directory: "/workspace",
          workspace: "workspace",
        },
      ]);
      expect(deleteRequests).toEqual(abortRequests);
      expect(lifecycle).toEqual([
        "messages",
        "create",
        "agents",
        "prompt",
        "abort",
        "delete",
      ]);
    } finally {
      jest.useRealTimers();
    }
  });

  test("retains prompt, server abort, and delete failures after client cancellation", async () => {
    const lifecycle: string[] = [];
    const abortRequests: Record<string, unknown>[] = [];
    const deleteRequests: Record<string, unknown>[] = [];
    const promptFailure = new DOMException("summary cancelled", "AbortError");
    let error: unknown;

    try {
      await runCompaction(session("source"), {
        promptError: promptFailure,
        abortError: new Error("scratch abort failed"),
        deleteError: new Error("scratch cleanup failed"),
        deleteErrorAt: 1,
        lifecycle,
        abortRequests,
        deleteRequests,
      });
    } catch (caught) {
      error = caught;
    }

    expect(error).toBeInstanceOf(AggregateError);
    expect(nestedErrorMessages(error)).toEqual(
      expect.arrayContaining([
        "summary cancelled",
        "scratch abort failed",
        "scratch cleanup failed",
      ]),
    );
    expect(lifecycle).toEqual([
      "messages",
      "create",
      "agents",
      "prompt",
      "abort",
      "delete",
    ]);
    expect(abortRequests).toEqual(deleteRequests);
  });

  test("prefers body-info errors over valid or malformed response text without retrying", async () => {
    for (const text of [
      '{"assistant":"Valid summary JSON."}',
      "not valid JSON",
    ]) {
      const lifecycle: string[] = [];
      await expect(
        runCompaction(session("source"), {
          promptInfoErrors: [
            {
              name: "ProviderBodyError",
              data: { providerID: "provider", message: "provider failed" },
              parts: [{ type: "text", text }],
            },
          ],
          lifecycle,
        }),
      ).rejects.toThrow("Summary provider failed (ProviderBodyError).");
      expect(lifecycle.filter(event => event === "prompt")).toHaveLength(1);
      expect(lifecycle.filter(event => event === "create")).toHaveLength(1);
      expect(lifecycle.filter(event => event === "delete")).toHaveLength(1);
    }
  });

  test("keeps the post-prompt checkpoint re-read healthy by omitting format", async () => {
    const lifecycle: string[] = [];
    const requests = await runCompaction(session("source"), {
      lifecycle,
      textResponses: [
        JSON.stringify({
          assistant: modelCapsule("analysis-only", "Completed the request."),
        }),
        JSON.stringify({
          assistant: modelCapsule("analysis-only", "Probe summary."),
        }),
      ],
    });

    expect(requests.prompts[0]).not.toHaveProperty("format");
    expect((requests.partUpdates[0]!.part as TextPart).text).toBe(
      modelCapsule("analysis-only", "Completed the request."),
    );
    expect(lifecycle.filter(event => event === "messages")).toHaveLength(1);

    await requests.v2.session.prompt({
      sessionID: "probe-format",
      agent: "magic-compact-summarizer",
      format: { type: "json_schema" },
      parts: [{ type: "text", text: "probe" }],
    });
    await expect(
      requests.v2.session.messages({ sessionID: "probe-format" }),
    ).resolves.toMatchObject({
      error: expect.stringContaining("format-poisoned"),
    });
  });

  test.each([
    ["missing text", undefined],
    ["invalid JSON", "not valid JSON"],
  ])("retries %s once in a fresh fork", async (_name, failedOutput) => {
    const lifecycle: string[] = [];
    const requests = await runCompaction(session("source"), {
      textResponses: [
        failedOutput,
        JSON.stringify({
          assistant: modelCapsule("analysis-only", "Recovered summary."),
        }),
      ],
      lifecycle,
    });

    expect(requests.creates).toHaveLength(2);
    expect(requests.deletes).toEqual(["scratch-1", "scratch-2"]);
    expect(lifecycle.filter(event => event === "prompt")).toHaveLength(2);
    expect(lifecycle.filter(event => event === "delete")).toHaveLength(2);
    expect((requests.partUpdates[0]!.part as TextPart).text).toBe(
      modelCapsule("analysis-only", "Recovered summary."),
    );
  });

  test.each([
    "Indexed arr[0] after parse.",
    "See [docs](https://example.com/path) for setup.",
    'Returned ["ok","pending"] from the parser.',
    "Updated app/[slug]/page.tsx route handler.",
    "Implemented the parser [add details].",
    "Implemented the parser. TODO: add evidence.",
    "TBD after validation.",
    "[waiting for details]",
  ])("accepts substantive standard summary text: %s", async text => {
    const summary = modelCapsule("analysis-only", text);
    const requests = await runCompaction(session("source"), {
      messages: compactableMessages(),
      textResponses: [JSON.stringify({ assistant: summary })],
    });

    expect((requests.partUpdates[0]!.part as TextPart).text).toBe(summary);
  });

  test.each([
    "Documented sourceBytes, windows, evidence-records, and omitted-record-count as ordinary field names.",
    'Documented the JSON field "kind": "tool".',
    "Set sourceBytes = 128.",
  ])(
    "accepts machine-field syntax inside ordinary capsule prose: %s",
    async text => {
      const summary = `Outcome: analysis-only\nWork: ${text}`;
      const requests = await runCompaction(session("source"), {
        textResponses: [JSON.stringify({ assistant: summary })],
      });

      expect((requests.partUpdates[0]!.part as TextPart).text).toBe(summary);
    },
  );

  test("serializes user previews as JSON", () => {
    const turn = {
      user: [
        message("user", "user", [
          textPart("text", "source", "user", '<request & "proof">'),
        ]),
      ],
      assistants: [message("assistant", "assistant", [])],
    };

    const prompt = buildCompactionPrompt(
      [turn],
      null,
      new Map([[turn, "bounded evidence"]]),
    );

    expect(prompt).toContain(JSON.stringify('<request & "proof">\n...'));
    expect(prompt).not.toContain("&lt;request");
  });

  test("truncates user previews by Unicode code point and removes provider-invalid input", () => {
    const emojiBoundary = `${"a".repeat(299)}\ud83d\ude80ignored`;
    const invalidXml = "before\u0000\u000b\ud800middle\ud801after";
    const turns = [
      {
        user: [
          message("emoji-user", "user", [
            textPart("emoji-text", "source", "emoji-user", emojiBoundary),
          ]),
        ],
        assistants: [message("emoji-assistant", "assistant", [])],
      },
      {
        user: [
          message("invalid-user", "user", [
            textPart("invalid-text", "source", "invalid-user", invalidXml),
          ]),
        ],
        assistants: [message("invalid-assistant", "assistant", [])],
      },
    ];
    const prompt = buildCompactionPrompt(
      turns,
      null,
      new Map(turns.map(turn => [turn, "bounded evidence"])),
    );

    expect(prompt).toContain(`${"a".repeat(299)}\ud83d\ude80...`);
    expect(prompt).not.toContain("ignored");
    expect(prompt).toContain("beforemiddleafter\\n...");
    expect(prompt).not.toContain("\u0000");
    expect(prompt).not.toContain("\u000b");
    expect(hasUnpairedSurrogate(prompt)).toBeFalse();
  });

  test("routes large completed turns to bounded scratch summarization", async () => {
    const turn = metricTurn(170_000, 21, 32);
    const finalText = turn.assistants
      .at(-1)!
      .parts.find((part): part is TextPart => part.type === "text")!;
    finalText.text = `Historical exact delivery marker. ${finalText.text}`;

    const requests = await runCompaction(session("source"), {
      messages: [...turn.user, ...turn.assistants],
    });

    expect(requests.prompts).toHaveLength(1);
    expect(requests.creates).toHaveLength(1);
    expect(requests.creates[0]).toMatchObject({
      directory: "/workspace",
      workspace: "workspace",
      permission: [{ permission: "*", pattern: "*", action: "deny" }],
    });
    expect(requests.creates[0]).not.toHaveProperty("parentID");
    expect(requests.forks).toEqual([]);
    expect(requests.updates).toEqual([]);
    expect(requests.prompts[0]).toMatchObject({
      sessionID: "scratch-1",
      agent: "magic-compact-summarizer",
    });
    expect(requests.prompts[0]).not.toHaveProperty("tools");
    expect(promptText(requests.prompts[0]!)).toContain(
      "Historical exact delivery marker",
    );
    expect(encode(promptText(requests.prompts[0]!)).length).toBeLessThanOrEqual(
      24_000,
    );
    expect(requests.deletes).toEqual(["scratch-1"]);
  }, 30_000);

  test("renders deterministic summaries as human-readable continuation capsules without machine scaffolding", async () => {
    const requests = await runDeterministicRenderer(session("source"), {
      messages: outlierTurn(),
    });
    const stored = storedSummary(requests, "outlier-assistant-0");

    expect(requests.prompts).toEqual([]);
    expect(requests.creates).toEqual([]);
    expect(requests.agentRequests).toEqual([]);
    expect(stored).toStartWith("Outcome: partial");
    expect(stored).toContain(
      "Work: Delivered implementation in src/critical.ts.",
    );
    expect(stored).toContain("VCS:");
    expect(stored).toContain("Unresolved:");
    expect(stored).toContain(
      "[error] bun run integration → integration failed",
    );
    expect(stored).toContain("[running] bun run long-check");
    expect(stored).toContain("[failed] bun test: Process exited with code 1");
    expect(stored).toContain("[not-run] bun run smoke: smoke not run");
    expect(stored).toContain("bun test reported 88 pass");
    expect(stored).toContain("commit abc123 was pushed");
    expect(stored).toContain("final assistant evidence");
    for (const scaffolding of [
      '"kind"',
      '"polarity"',
      '"sourceBytes"',
      '"windows"',
      "omitted-record-count",
      "required-record-counts",
      "required-record-overflow",
      "required-record-total",
      "assistant-text-records",
      "evidence-records",
      "Historical assistant text (authoritative)",
      "Historical terminal evidence (authoritative)",
    ]) {
      expect(stored).not.toContain(scaffolding);
    }
    expect(encode(stored).length).toBeLessThanOrEqual(
      DETERMINISTIC_SUMMARY_MAX_TOKENS,
    );
    expect(hasUnpairedSurrogate(stored)).toBeFalse();
  }, 300_000);

  test("routes unresolved semantic evidence deterministically with no scratch prompts", async () => {
    const errorState = metricTurn(1, 1, 1, "error");
    const runningState = metricTurn(1, 1, 0);
    runningState.assistants[0]!.parts.push(
      genericToolPart("running", "metric-assistant-0", "bash", {
        status: "running",
        input: { command: "synthetic-watch" },
        metadata: {},
        time: { start: 1 },
      }),
    );
    const unknownState = metricTurn(1, 1, 0);
    unknownState.assistants[0]!.parts.push(
      genericToolPart("unknown", "metric-assistant-0", "custom", {
        status: "unknown",
        input: {},
        metadata: {},
      }),
    );
    const failedResult = metricTurn(1, 1, 0);
    failedResult.assistants[0]!.parts.push(
      commandToolPart(
        "failed",
        "metric-assistant-0",
        "bun test",
        "Process exited with code 1",
      ),
    );
    const notRunResult = metricTurn(1, 1, 0);
    notRunResult.assistants[0]!.parts.push(
      commandToolPart(
        "not-run",
        "metric-assistant-0",
        "bun test",
        "Tests were not run",
      ),
    );
    const incompleteProse = metricTurn(1, 1, 0);
    incompleteProse.assistants[0]!.parts.push(
      textPart(
        "incomplete",
        "source",
        "metric-assistant-0",
        "Work remains: run the integration checks.",
      ),
    );

    for (const turn of [
      errorState,
      runningState,
      unknownState,
      failedResult,
      notRunResult,
      incompleteProse,
    ]) {
      const requests = await runCompaction(session("source"), {
        messages: [...turn.user, ...turn.assistants],
      });
      expect(requests.prompts).toEqual([]);
      expect(requests.creates).toEqual([]);
      expect(requests.forks).toEqual([]);
      expect(requests.agentRequests).toEqual([]);
    }
  });

  test.each([
    "Blocked: provider credentials are missing.",
    "Blocked: waiting for operator approval.",
    "| Status | Blocked: waiting for operator approval. |",
    "| Release | Blocked: awaiting operator approval. |",
    "Could not finish the migration because the signing key is unavailable.",
    "I couldn't complete the request without deployment credentials.",
    "Unable to continue: API credentials are missing.",
    "The task remains unfinished because the required token is unavailable.",
  ])(
    "routes explicit unresolved assistant prose deterministically: %s",
    async text => {
      const messages = completedTurn("explicit-unresolved", "Finish the work");
      messages[1]!.parts.push(
        textPart(
          "explicit-unresolved-text",
          "source",
          messages[1]!.info.id,
          text,
        ),
      );

      const requests = await runCompaction(session("source"), { messages });
      const stored = storedSummary(requests, messages[1]!.info.id);

      expect(requests.prompts).toEqual([]);
      expect(requests.creates).toEqual([]);
      expect(stored).toStartWith("Outcome: partial");
      expect(stored).toContain(text);
      expect(stored).not.toStartWith("Outcome: completed");
    },
  );

  test("routes a final assistant metadata error deterministically with readable evidence", async () => {
    const messages = completedTurn("assistant-error", "Authenticate provider");
    const assistant = messages[1]!;
    Object.assign(assistant.info, {
      error: {
        name: "ProviderAuthError",
        data: {
          providerID: "provider-x",
          message: "Authentication token expired.",
        },
      },
    });

    const requests = await runCompaction(session("source"), { messages });
    const stored = storedSummary(requests, assistant.info.id);

    expect(requests.prompts).toEqual([]);
    expect(requests.creates).toEqual([]);
    expect(requests.agentRequests).toEqual([]);
    expect(stored).toStartWith("Outcome: partial");
    expect(stored).toContain("Unresolved:");
    expect(stored).toContain("[error] ProviderAuthError");
    expect(stored).toContain("Authentication token expired.");
    expect(stored).toContain("provider-x");
    expect(stored).not.toContain('"kind"');
    expect(stored).not.toContain('"status":"error"');
  });

  test("keeps an earlier assistant metadata error model-eligible after a successful recovery", async () => {
    const messages = completedTurn("assistant-recovery", "Retry provider");
    const failed = messages[1]!;
    Object.assign(failed.info, {
      error: {
        name: "ProviderAuthError",
        data: {
          providerID: "provider-x",
          message: "Authentication token expired.",
        },
      },
    });
    const recovered = message("assistant-recovery-success", "assistant", [
      textPart(
        "assistant-recovery-success-text",
        "source",
        "assistant-recovery-success",
        "Recovered after provider credentials were refreshed.",
      ),
    ]);
    Object.assign(recovered.info, { finish: "stop" });
    messages.push(recovered);

    const requests = await runCompaction(session("source"), { messages });
    const evidence = summaryRequestEvidence(requests.prompts[0]!);

    expect(requests.prompts).toHaveLength(1);
    expect(requests.creates).toHaveLength(1);
    expect(requests.agentRequests).toHaveLength(1);
    expect(evidence).toContain('"kind":"assistant-message-status"');
    expect(evidence).toContain('"status":"recovered"');
    expect(evidence).toContain('"errorType":"ProviderAuthError"');
    expect(evidence).toContain('"message":"Authentication token expired."');
    expect(evidence).toContain(
      "Recovered after provider credentials were refreshed.",
    );
  });

  test("keeps the final substantive assistant response in a saturated standard scratch prompt", async () => {
    const marker = "FINAL RESPONSE DELIVERED";
    const denseBody = `${marker}\n${Array.from(
      { length: 120 },
      (_, index) => `"key${index}": "value \\"escaped\\" ${"z".repeat(30)}"`,
    ).join("\n")}`;
    const messages = completedTurn("saturated-final", "Deliver the change");
    messages[1]!.parts.push(
      textPart(
        "saturated-final-text",
        "source",
        messages[1]!.info.id,
        denseBody,
      ),
    );

    const requests = await runCompaction(session("source"), { messages });
    const evidence = summaryRequestEvidence(requests.prompts[0]!);

    expect(requests.prompts).toHaveLength(1);
    expect(requests.creates).toHaveLength(1);
    expect(evidence).toContain(marker);
    expect(
      labeledSummarySection(
        evidence,
        "Historical assistant text (authoritative)",
        "Verification",
      ),
    ).not.toContain("omitted-record-count=1");
    expect(encode(evidence).length).toBeLessThanOrEqual(
      MAX_SUMMARY_INPUT_TOKENS,
    );
  });

  test("keeps recovered assistant error status in a saturated standard scratch prompt", async () => {
    const failed = message("recovered-dense-error", "assistant", []);
    Object.assign(failed.info, {
      error: {
        name: "ProviderAuthError",
        data: {
          providerID: "provider-x",
          message: "Authentication token expired.",
        },
      },
    });
    failed.parts.push(
      ...Array.from({ length: 12 }, (_, index) =>
        commandToolPart(
          `recovered-dense-tool-${index}`,
          failed.info.id,
          "bun test",
          `10 passed, 0 failed ${"y".repeat(3_000)}`,
        ),
      ),
    );
    const recovered = message("recovered-dense-success", "assistant", [
      textPart(
        "recovered-dense-success-text",
        "source",
        "recovered-dense-success",
        "Recovered after provider credentials were refreshed.",
      ),
    ]);
    Object.assign(recovered.info, { finish: "stop" });
    const user = message("recovered-dense-user", "user", [
      textPart(
        "recovered-dense-user-text",
        "source",
        "recovered-dense-user",
        "Retry the provider",
      ),
    ]);

    const requests = await runCompaction(session("source"), {
      messages: [user, failed, recovered],
    });
    const evidence = summaryRequestEvidence(requests.prompts[0]!);

    expect(requests.prompts).toHaveLength(1);
    expect(evidence).toContain('"kind":"assistant-message-status"');
    expect(evidence).toContain('"status":"recovered"');
    expect(evidence).toContain('"errorType":"ProviderAuthError"');
    expect(encode(evidence).length).toBeLessThanOrEqual(
      DETERMINISTIC_SUMMARY_MAX_TOKENS,
    );
  });

  test("keeps the final substantive assistant response as exact evidence beside recovered statuses", async () => {
    const marker = "FINAL SUBSTANTIVE RESPONSE";
    const denseBody = `${marker}\n${Array.from(
      { length: 120 },
      (_, index) => `"key${index}": "value \\"escaped\\" ${"z".repeat(30)}"`,
    ).join("\n")}`;
    const user = message("overflow-capsule-user", "user", [
      textPart(
        "overflow-capsule-user-text",
        "source",
        "overflow-capsule-user",
        "Recover and finish",
      ),
    ]);
    const failed = Array.from({ length: 40 }, (_, index) => {
      const item = message(`overflow-capsule-error-${index}`, "assistant", []);
      Object.assign(item.info, {
        error: {
          name: `ProviderAuthError-${index}-${"x".repeat(180)}`,
          data: {
            providerID: "provider-x",
            message: `Authentication token expired for attempt ${index}.`,
          },
        },
      });
      return item;
    });
    const recovered = message("overflow-capsule-success", "assistant", [
      textPart(
        "overflow-capsule-success-text",
        "source",
        "overflow-capsule-success",
        "Recovered after provider credentials were refreshed.",
      ),
    ]);
    Object.assign(recovered.info, { finish: "stop" });
    recovered.parts.push(
      textPart(
        "overflow-capsule-final-text",
        "source",
        "overflow-capsule-success",
        denseBody,
      ),
    );

    const requests = await runCompaction(session("source"), {
      messages: [user, ...failed, recovered],
    });
    const evidence = summaryRequestEvidence(requests.prompts[0]!);

    expect(requests.prompts).toHaveLength(1);
    expect(requests.creates).toHaveLength(1);
    expect(requests.agentRequests).toHaveLength(1);
    expect(evidence).toContain(marker);
    expect(evidence).toContain('"partID":"overflow-capsule-final-text"');
    expect(evidence).toContain('"status":"recovered"');
    expect(evidence).toContain('"errorType":"ProviderAuthError-0-');
    expect(evidence).not.toContain(
      "Required deterministic evidence overflow capsule:",
    );
    expect(
      labeledSummarySection(
        evidence,
        "Historical assistant text (authoritative)",
        "Verification",
      ),
    ).not.toMatch(/omitted-record-count=[1-9]/);
    expect(encode(evidence).length).toBeLessThanOrEqual(
      MAX_SUMMARY_INPUT_TOKENS,
    );
  });

  test.each([
    ["assistant", "No tests were run."],
    ["assistant", "No tests have been executed."],
    ["assistant", "Tests have not yet run."],
    ["assistant", "Tests were never executed."],
    ["assistant", "Unable to run tests."],
    ["assistant", "Tests have yet to run."],
    ["tool", "No tests were run."],
    ["tool", "No tests have been executed."],
    ["tool", "Tests have not yet run."],
    ["tool", "never executed"],
    ["tool", "unable to run"],
    ["tool", "have yet to run"],
  ] as const)(
    "routes %s not-run evidence deterministically without a scratch prompt: %s",
    async (source, text) => {
      const messages = completedTurn("not-run-form", "Verify the change");
      const assistant = messages[1]!;
      assistant.parts.push(
        source === "assistant"
          ? textPart(
              "not-run-assistant-text",
              "source",
              assistant.info.id,
              text,
            )
          : commandToolPart(
              "not-run-tool",
              assistant.info.id,
              "bun test",
              text,
            ),
      );

      const requests = await runCompaction(session("source"), { messages });
      const stored = storedSummary(requests, assistant.info.id);

      expect(requests.prompts).toEqual([]);
      expect(requests.creates).toEqual([]);
      expect(requests.agentRequests).toEqual([]);
      expect(stored).toStartWith("Outcome: partial");
      expect(stored).toContain("Unresolved:");
      expect(stored).toContain(text);
      expect(stored).toContain("[not-run]");
      expect(stored).not.toContain('"polarity"');
    },
  );

  test.each(["assistant", "passive tool"] as const)(
    "keeps generic %s inability prose model-eligible",
    async source => {
      const messages = completedTurn("generic-inability", "Inspect the parser");
      const assistant = messages[1]!;
      assistant.parts.push(
        source === "assistant"
          ? textPart(
              "generic-inability-text",
              "source",
              assistant.info.id,
              "Unable to run parser normalization in this code path.",
            )
          : genericToolPart(
              "generic-inability-tool",
              assistant.info.id,
              "read",
              {
                status: "completed",
                input: { filePath: "/workspace/notes.md" },
                output: "unable to run",
                title: "Read notes",
                metadata: {},
                time: { start: 1, end: 2 },
              },
            ),
      );

      const requests = await runCompaction(session("source"), { messages });

      expect(requests.prompts).toHaveLength(1);
      expect(requests.creates).toHaveLength(1);
      expect(requests.agentRequests).toHaveLength(1);
    },
  );

  test("keeps a packed standard request within the 24K input cap", async () => {
    const messages = Array.from({ length: 24 }, (_, index) =>
      completedTurn(`packed-${index}`, `Request ${index}`),
    ).flat();
    const expectedIDs = Array.from(
      { length: 24 },
      (_, index) => `packed-${index}-assistant`,
    );

    const requests = await runCompaction(session("source"), { messages });

    expect(requests.prompts.length).toBeGreaterThanOrEqual(1);
    expect(requests.prompts.flatMap(summaryRequestIDs)).toEqual(expectedIDs);
    for (const prompt of requests.prompts) {
      expect(encode(promptText(prompt)).length).toBeLessThanOrEqual(24_000);
    }
    expect(requests.creates).toHaveLength(requests.prompts.length);
    expect(requests.forks).toEqual([]);
  });

  test("rejects a single standard request that exceeds the 24K input cap", async () => {
    const oversizedID = `oversized-${"\u03ff".repeat(7_000)}`;
    const messages = [
      ...completedTurn("small", "Small request"),
      ...completedTurn(oversizedID, "Oversized exact ID"),
    ];

    await expect(
      runCompaction(session("source"), { messages }),
    ).rejects.toThrow("One bounded turn evidence map exceeds");
  });

  test("runs no more than two scratch summary requests concurrently", async () => {
    const messages = Array.from({ length: 240 }, (_, index) =>
      completedTurn(`concurrent-${index}`, `Request ${index}`),
    ).flat();

    const requests = await runCompaction(session("source"), {
      messages,
      onPrompt: async () => {
        await Bun.sleep(5);
      },
    });

    expect(requests.prompts.length).toBeGreaterThan(1);
    expect(requests.maxPromptConcurrency).toBe(2);
    expect(requests.forks).toEqual([]);
    expect(requests.deletes).toHaveLength(requests.creates.length);
  });

  test("rejects weak projected savings before backup, source writes, or stats", async () => {
    const messages = completedTurn("weak", "Small request");
    messages[1]!.parts.push(
      textPart(
        "weak-result",
        "source",
        "weak-assistant",
        "A short final answer.",
      ),
    );
    addProviderTokens(messages);
    const command = createCommandHarness(messages, {
      summaryTextOutputs: [
        JSON.stringify({
          "weak-assistant": modelCapsule(
            "analysis-only",
            "A replacement that cannot provide useful net context reduction.",
          ),
        }),
      ],
    });

    await expect(executeMagicCompact(command.v2, "source", 0)).rejects.toThrow(
      "useful canonical active-content savings",
    );

    expect(command.backupForks).toEqual([]);
    expect(command.sourcePrompts).toEqual([]);
    expect(command.partUpdates).toEqual([]);
    expect(command.partDeletes).toEqual([]);
    expect(command.messageDeletes).toEqual([]);
    expect(
      command.sessionUpdates.filter(request => request.sessionID === "source"),
    ).toEqual([]);
    expect(await readStats("source")).toBeNull();
    expect(command.toasts.at(-1)).toMatchObject({
      title: "Magic Compact Failed",
      message: expect.stringContaining(
        "useful canonical active-content savings",
      ),
      variant: "error",
    });
  });

  test("labels every stats string as a canonical active-content estimate with a stale provider snapshot note", () => {
    const stats: ConversationStats = {
      version: 1,
      sourceSessionId: "source",
      rootSessionId: "source",
      totalTokensPruned: 133_194,
      cachedTokensSaved: 0,
      processedAssistantMessageIds: [],
    };

    const compact = statsMessage(1, 274_692, 141_498, stats, null);
    const trim = trimStatsMessage(90_000, 50_000, stats, null);
    const summary = statsSummaryMessage(1, stats, null);

    expect(compact).toContain("Magic Compaction #1");
    expect(trim).toContain("Magic Trim");
    expect(summary).toContain("Magic Compact");
    for (const text of [compact, trim, summary]) {
      expect(text).toContain("canonical active-content");
      expect(text).toContain("ignored reporting notice is excluded");
      expect(text).toContain("provider usage snapshot");
      expect(text).toContain("stale");
      expect(text).toContain("later assistant response");
    }
    expect(compact).toContain("274.7K");
    expect(compact).toContain("141.5K");
    expect(
      trimStatsMessage(100, 120, { ...stats, totalTokensPruned: -20 }, null),
    ).toContain("~-20 tokens pruned");
  });

  test("reports compact stats on a cycle-free active basis across the 999-to-1K boundary", async () => {
    const messages = completedTurn("stats-threshold", "Compact the history");
    messages[1]!.parts.push({
      id: "stats-threshold-reasoning",
      sessionID: "source",
      messageID: messages[1]!.info.id,
      type: "reasoning",
      text: "historical removable detail ".repeat(100),
      time: { start: 1, end: 2 },
    });
    const beforeTokens = countCanonicalContentTokens(messages);
    const command = createCommandHarness(messages);

    try {
      await writeStats("source", {
        version: 1,
        sourceSessionId: "source",
        rootSessionId: "source",
        totalTokensPruned: 999,
        cachedTokensSaved: 0,
        processedAssistantMessageIds: [],
      });

      expect(await executeMagicCompact(command.v2, "source", 0)).toBeTrue();

      const activeMessages = messages.map(item => ({
        ...item,
        parts: item.parts.filter(
          part => part.type !== "text" || part.ignored !== true,
        ),
      }));
      const afterTokens = countCanonicalContentTokens(activeMessages);
      const stats = await readStats("source");
      const notice = messages
        .flatMap(item => item.parts)
        .find(
          (part): part is TextPart =>
            part.type === "text"
            && part.ignored === true
            && part.metadata?.magicCompact?.stats === true,
        )!;

      expect(await countSessionTokens(command.v2, "source")).toBe(afterTokens);
      expect(stats?.totalTokensPruned).toBe(999 + beforeTokens - afterTokens);
      expect(notice.text).toBe(
        statsMessage(1, beforeTokens, afterTokens, stats!, null),
      );
      expect(notice.text).toContain("ignored reporting notice is excluded");
      expect(notice.text).toMatch(/Total tokens pruned[^\n]*~1(?:\.\d)?K/);
    } finally {
      await Promise.all([deleteStats("source"), deleteStats("backup")]);
    }
  });

  test("never mutates provider usage token metadata while compacting", async () => {
    const messages = completedTurn("provider-usage", "Compact me");
    const assistant = messages[1]!;
    assistant.parts.push(
      genericToolPart("provider-usage-read", assistant.info.id, "read", {
        status: "completed",
        input: { filePath: "/workspace/bulky.txt" },
        output: "bulky provider usage output\n".repeat(400),
        title: "Read bulky artifact",
        metadata: {},
        time: { start: 1, end: 2 },
      }),
    );
    addProviderTokens(messages);
    const providerAssistant = messages.find(
      item => item.info.id === "provider-usage-assistant",
    )!;
    const providerTokens = structuredClone(providerAssistant.info.tokens);
    const command = createCommandHarness(messages, {
      summaryTextOutputs: [
        JSON.stringify({
          "provider-usage-assistant": modelCapsule(
            "completed",
            "Compacted the resolved read turn.",
          ),
        }),
      ],
    });

    expect(await executeMagicCompact(command.v2, "source", 0)).toBe(true);
    expect(providerAssistant.info.tokens).toEqual(providerTokens);
    expect(
      messages.some(message =>
        message.parts.some(
          part =>
            part.type === "text"
            && part.text.includes("canonical active-content"),
        ),
      ),
    ).toBeTrue();
  });

  test("keeps a net-negative replacement turn message-identical while compacting profitable turns", async () => {
    const messages = [
      ...completedTurn("profitable", "Trim the bulky tool output"),
      ...completedTurn("net-negative", "Small request"),
    ];
    const profitableAssistant = messages[1]!;
    profitableAssistant.parts.push(
      genericToolPart("profitable-read", profitableAssistant.info.id, "read", {
        status: "completed",
        input: { filePath: "/workspace/artifact.txt" },
        output: "exact bulky historical read output\n".repeat(400),
        title: "Read artifact",
        metadata: {},
        time: { start: 1, end: 2 },
      }),
    );
    const negativeAssistant = messages[3]!;
    negativeAssistant.parts.push(
      textPart(
        "net-negative-answer",
        "source",
        negativeAssistant.info.id,
        "done.",
      ),
    );
    addProviderTokens(messages);
    const negativeSnapshot = structuredClone(
      messages.filter(message =>
        ["net-negative-user", "net-negative-assistant"].includes(
          message.info.id,
        ),
      ),
    );
    const verboseReplacement =
      "A verbose replacement summary that cannot fit the terse source turn. ".repeat(
        12,
      );
    const command = createCommandHarness(messages, {
      summaryTextOutputs: [
        JSON.stringify({
          "profitable-assistant": modelCapsule(
            "completed",
            "Trimmed the bulky tool output.",
          ),
          "net-negative-assistant": modelCapsule(
            "analysis-only",
            verboseReplacement,
          ),
        }),
      ],
    });

    expect(await executeMagicCompact(command.v2, "source", 0)).toBe(true);

    expect(summaryText(messages, "profitable-assistant")).toBe(
      modelCapsule("completed", "Trimmed the bulky tool output."),
    );
    expect(
      messages.filter(message =>
        ["net-negative-user", "net-negative-assistant"].includes(
          message.info.id,
        ),
      ),
    ).toEqual(negativeSnapshot);
    expect(
      command.partUpdates.filter(
        update =>
          update.messageID === "net-negative-assistant"
          || update.messageID === "net-negative-user",
      ),
    ).toEqual([]);
    expect(
      command.partDeletes.filter(
        update =>
          update.messageID === "net-negative-assistant"
          || update.messageID === "net-negative-user",
      ),
    ).toEqual([]);
    expect(command.messageDeletes).toEqual([]);
    expect(boundaryParts(messages)).toHaveLength(1);
    const pruned = messages
      .flatMap(item => item.parts)
      .find((part): part is ToolPart => part.id === "profitable-read")!;
    expect(omissionContentIDs(pruned)).toHaveLength(1);
    expect((await readStats("source"))?.totalTokensPruned).toBeGreaterThan(0);
  });

  test("rejects a net-negative already-trimmed deterministic candidate without mutating the turn", async () => {
    const messages = [
      ...completedTurn("profitable", "Trim the bulky tool output"),
      ...completedTurn("deterministic", "Retry the failing test"),
    ];
    const profitableAssistant = messages[1]!;
    profitableAssistant.parts.push(
      genericToolPart("profitable-read", profitableAssistant.info.id, "read", {
        status: "completed",
        input: { filePath: "/workspace/artifact.txt" },
        output: "exact bulky historical read output\n".repeat(400),
        title: "Read artifact",
        metadata: {},
        time: { start: 1, end: 2 },
      }),
    );
    const deterministicAssistant = messages[3]!;
    deterministicAssistant.parts.push(
      genericToolPart(
        "deterministic-failure",
        deterministicAssistant.info.id,
        "bash",
        {
          status: "error",
          input: { command: "bun test" },
          error: "Process exited with code 1",
          metadata: {},
          time: { start: 1, end: 2 },
        },
      ),
      genericToolPart(
        "deterministic-trimmed",
        deterministicAssistant.info.id,
        "todowrite",
        {
          status: "completed",
          input: { todos: [] },
          output: "Successfully updated todos.",
          title: "Update todos",
          metadata: { magicCompact: { trimmed: true } },
          time: { start: 1, end: 2 },
        },
      ),
    );
    addProviderTokens(messages);
    const deterministicSnapshot = structuredClone(
      messages.filter(message =>
        ["deterministic-user", "deterministic-assistant"].includes(
          message.info.id,
        ),
      ),
    );
    const command = createCommandHarness(messages, {
      summaryTextOutputs: [
        JSON.stringify({
          "profitable-assistant": modelCapsule(
            "completed",
            "Trimmed the bulky tool output.",
          ),
        }),
      ],
    });

    expect(await executeMagicCompact(command.v2, "source", 0)).toBe(true);

    expect(command.summaryRequests).toHaveLength(1);
    expect(summaryRequestIDs(command.summaryRequests[0]!)).toEqual([
      "profitable-assistant",
    ]);
    expect(summaryText(messages, "profitable-assistant")).toBe(
      modelCapsule("completed", "Trimmed the bulky tool output."),
    );
    expect(
      messages.filter(message =>
        ["deterministic-user", "deterministic-assistant"].includes(
          message.info.id,
        ),
      ),
    ).toEqual(deterministicSnapshot);
    expect(
      command.partUpdates.filter(
        update =>
          update.messageID === "deterministic-assistant"
          || update.messageID === "deterministic-user",
      ),
    ).toEqual([]);
    expect(
      command.partDeletes.filter(
        update =>
          update.messageID === "deterministic-assistant"
          || update.messageID === "deterministic-user",
      ),
    ).toEqual([]);
    expect(command.messageDeletes).toEqual([]);
    expect(boundaryParts(messages)).toHaveLength(1);
    expect((await readStats("source"))?.totalTokensPruned).toBeGreaterThan(0);
  });

  test("projects each candidate turn exactly once so near-boundary selection stays stable", async () => {
    const messages: MessageWithParts[] = [];
    for (let index = 0; index < 240; index += 1) {
      messages.push(...nearBoundaryReadTurn(`stable-${index}`));
    }
    for (const id of ["bulky-a", "bulky-b"]) {
      messages.push(
        ...nearBoundaryReadTurn(
          id,
          "exact bulky historical read output\n".repeat(800),
        ),
      );
    }
    addProviderTokens(messages);
    const boundarySnapshot = structuredClone(
      messages.filter(message => message.info.id.startsWith("stable-")),
    );
    const command = createCommandHarness(messages);

    expect(await executeMagicCompact(command.v2, "source", 0)).toBe(true);

    const compactedMessageIDs = new Set(
      messages
        .filter(message =>
          message.parts.some(
            part =>
              part.type === "text"
              && (
                part.metadata?.magicCompact as
                  | Record<string, unknown>
                  | undefined
              )?.["summary"] === true,
          ),
        )
        .map(message => message.info.id),
    );
    const boundaryByID = new Map(
      boundarySnapshot.map(message => [message.info.id, message]),
    );
    const liveByID = new Map(
      messages.map(message => [message.info.id, message]),
    );
    for (const [messageID, snapshot] of boundaryByID) {
      if (compactedMessageIDs.has(messageID)) {
        expect(liveByID.get(messageID)).toBeDefined();
      } else {
        expect(liveByID.get(messageID)).toEqual(snapshot);
      }
    }
    expect(summaryText(messages, "bulky-a-assistant")).toBe(
      modelCapsule("completed", "Preserved."),
    );
  });

  test("retrieves every planned omission notice through the authorized plugin tool", async () => {
    const inputContent = "exact planned input\n".repeat(180);
    const outputContent = "exact planned output\n".repeat(180);
    const messages = completedTurn("planned-omissions", "Write the artifact");
    const assistant = messages[1]!;
    assistant.parts.push(
      genericToolPart("planned-write", assistant.info.id, "write", {
        status: "completed",
        input: { filePath: "/workspace/artifact.txt", content: inputContent },
        output: outputContent,
        title: "Write artifact",
        metadata: {},
        time: { start: 1, end: 2 },
      }),
    );
    addProviderTokens(messages);
    await rm(statsPath("source"), { force: true });
    await rm(cachePath("source"), { force: true });
    const command = createCommandHarness(messages);

    expect(await executeMagicCompact(command.v2, "source", 0)).toBe(true);
    const compacted = messages
      .flatMap(item => item.parts)
      .find((part): part is ToolPart => part.id === "planned-write")!;
    const contentIDs = omissionContentIDs(compacted);
    const omissionTool = await canonicalOmissionTool();

    expect(contentIDs).toHaveLength(2);
    const retrieved = await Promise.all(
      contentIDs.map(contentID =>
        omissionTool.execute(
          { contentId: contentID },
          omissionToolContext("source"),
        ),
      ),
    );
    expect(retrieved).toEqual([inputContent, outputContent]);
    for (const [index, original] of [inputContent, outputContent].entries()) {
      expect(Buffer.from(String(retrieved[index]), "utf16le")).toEqual(
        Buffer.from(original, "utf16le"),
      );
    }
    for (const contentID of contentIDs) {
      expect(
        await omissionTool.execute(
          { contentId: contentID },
          omissionToolContext("other-session"),
        ),
      ).toContain("no authorized current-session cache entry exists");
    }
  });

  test("compacts a second boundary while preserving summaries, cache lineage, stats, users, and native checkpoint", async () => {
    for (const sessionID of ["source", "backup", "ephemeral"]) {
      await rm(cachePath(sessionID), { force: true });
      await rm(statsPath(sessionID), { force: true });
    }
    const firstOutput = "first exact historical read\n".repeat(180);
    const secondOutput = "second exact historical read\n".repeat(180);
    const messages = nativeCheckpointFixture({ tailStartID: "tail-assistant" });
    const firstAssistant = messages.find(
      item => item.info.id === "suffix-1-assistant",
    )!;
    firstAssistant.parts.push(
      genericToolPart("first-lineage-read", firstAssistant.info.id, "read", {
        status: "completed",
        input: { filePath: "/workspace/first.txt" },
        output: firstOutput,
        title: "Read first artifact",
        metadata: {},
        time: { start: 1, end: 2 },
      }),
    );
    addProviderTokens(messages);
    const nativeSummaryIndex = messages.findIndex(
      item => item.info.id === "native-summary",
    );
    const nativePrefix = structuredClone(
      messages.slice(0, nativeSummaryIndex + 1),
    );
    const exactUserText = userTextByPartID(messages);
    const command = createCommandHarness(messages);

    expect(await executeMagicCompact(command.v2, "source", 1)).toBe(true);
    const firstSummary = summaryText(messages, "suffix-1-assistant");
    const firstID = omissionContentIDs(
      messages
        .flatMap(item => item.parts)
        .find((part): part is ToolPart => part.id === "first-lineage-read")!,
    )[0]!;
    const firstStats = await readStats("source");
    expect(firstStats?.totalTokensPruned).toBeGreaterThan(0);
    expect(boundaryParts(messages)).toEqual([
      expect.objectContaining({ messageID: "suffix-2-user" }),
    ]);

    const secondAssistant = messages.find(
      item => item.info.id === "suffix-2-assistant",
    )!;
    secondAssistant.parts.push(
      {
        id: "suffix-2-removable-detail",
        sessionID: "source",
        messageID: secondAssistant.info.id,
        type: "reasoning",
        text: "second removable historical detail ".repeat(1_200),
        time: { start: 1, end: 2 },
      } as Part,
      genericToolPart("second-lineage-read", secondAssistant.info.id, "read", {
        status: "completed",
        input: { filePath: "/workspace/second.txt" },
        output: secondOutput,
        title: "Read second artifact",
        metadata: {},
        time: { start: 1, end: 2 },
      }),
    );
    messages.push(
      ...completedTurn(
        "suffix-3",
        "Current exact user request after compaction",
      ),
    );
    addProviderTokens(messages);
    const beforeSecond = countCanonicalContentTokens(messages);
    const secondPlan = await planFor(messages, 1);
    const secondEligible = countCanonicalContentTokens(
      secondPlan.summarizedTurns.flatMap(turn => turn.assistants),
    );

    expect(await executeMagicCompact(command.v2, "source", 1)).toBe(true);
    const afterSecond = countCanonicalContentTokens(messages);
    const secondSaved = beforeSecond - afterSecond;
    const secondID = omissionContentIDs(
      messages
        .flatMap(item => item.parts)
        .find((part): part is ToolPart => part.id === "second-lineage-read")!,
    )[0]!;
    const secondStats = await readStats("source");
    const omissionTool = await canonicalOmissionTool();

    expect(summaryText(messages, "suffix-1-assistant")).toBe(firstSummary);
    expect(boundaryParts(messages)).toHaveLength(1);
    expect(boundaryParts(messages)[0]!.messageID).not.toBe("suffix-2-user");
    expect(secondSaved).toBeGreaterThanOrEqual(
      Math.max(32, Math.ceil(secondEligible * 0.2)),
    );
    expect(secondStats?.totalTokensPruned).toBe(
      firstStats!.totalTokensPruned + secondSaved,
    );
    expect(command.currentSourceSession.metadata?.magicCompact).toMatchObject({
      compactionCount: 2,
    });
    expect(
      messages.some(message =>
        message.parts.some(
          part =>
            part.type === "text" && part.text.includes("Magic Compaction #2"),
        ),
      ),
    ).toBeTrue();
    expect(messages.slice(0, nativeSummaryIndex + 1)).toEqual(nativePrefix);
    expect(userTextByPartID(messages)).toMatchObject(exactUserText);
    expect(userTextByPartID(messages)["suffix-3-text"]).toBe(
      "Current exact user request after compaction",
    );
    expect(
      await omissionTool.execute(
        { contentId: firstID },
        omissionToolContext("source"),
      ),
    ).toBe(firstOutput);
    expect(
      await omissionTool.execute(
        { contentId: secondID },
        omissionToolContext("source"),
      ),
    ).toBe(secondOutput);
    expect(
      await omissionTool.execute(
        { contentId: firstID },
        omissionToolContext("ephemeral"),
      ),
    ).toBe(firstOutput);
    expect(
      await omissionTool.execute(
        { contentId: secondID },
        omissionToolContext("ephemeral"),
      ),
    ).toContain("no authorized current-session cache entry exists");
  });

  test("meets the incident-shaped structural and token acceptance gates", async () => {
    const singleCopy = await runCompaction(session("source"), {
      messages: incidentSessionFixture(1),
    });
    const singleCopyInputTokens = singleCopy.prompts.reduce(
      (sum, request) => sum + encode(promptText(request)).length,
      0,
    );
    const messages = incidentSessionFixture();
    const beforeTokens = countCanonicalContentTokens(messages);
    const toolParts = messages.flatMap(message =>
      message.parts.filter(part => part.type === "tool"),
    );
    const errorTools = toolParts.filter(
      part => part.type === "tool" && part.state.status === "error",
    );
    const runningTools = toolParts.filter(
      part => part.type === "tool" && part.state.status === "running",
    );

    await rm(statsPath("source"), { force: true });
    await rm(cachePath("source"), { force: true });
    const command = createCommandHarness(messages);
    expect(await executeMagicCompact(command.v2, "source", 0)).toBe(true);
    const afterTokens = await countSessionTokens(command.v2, "source");
    const stats = await readStats("source");
    const summaryUpdates = command.partUpdates.filter(update =>
      (update.part as TextPart).metadata?.magicCompact
        ? (
            (update.part as TextPart).metadata!.magicCompact as Record<
              string,
              unknown
            >
          ).summary === true
        : false,
    );
    const summaries = summaryUpdates.map(
      update => (update.part as TextPart).text,
    );
    const standardIDs = new Set(
      command.summaryRequests.flatMap(summaryRequestIDs),
    );
    const deterministic = summaryUpdates
      .filter(update => !standardIDs.has(String(update.messageID)))
      .map(update => (update.part as TextPart).text);
    const replacements = summaries.reduce(
      (sum, summary) => sum + encode(summary).length,
      0,
    );
    const deterministicTokens = deterministic.reduce(
      (sum, summary) => sum + encode(summary).length,
      0,
    );
    const promptTokens = command.summaryRequests.map(
      request => encode(promptText(request)).length,
    );
    const result = {
      turns: messages.filter(message => message.info.role === "assistant")
        .length,
      tools: toolParts.length,
      errorTools: errorTools.length,
      runningTools: runningTools.length,
      beforeTokens,
      afterTokens,
      reduction: (beforeTokens - afterTokens) / beforeTokens,
      replacementTokens: replacements,
      deterministicTokens,
      deterministicTurns: deterministic.length,
      standardTurns: command.summaryRequests.flatMap(summaryRequestIDs).length,
      modelInputTokens: promptTokens.reduce((sum, count) => sum + count, 0),
      maxRequestTokens: Math.max(...promptTokens),
      requests: command.summaryRequests.length,
      maxConcurrency: command.maxSummaryConcurrency,
      backupForks: command.backupForks.length,
      scratchCreates: command.scratchCreates.length,
      singleCopyInputTokens,
      singleCopyRequests: singleCopy.prompts.length,
    };
    console.info(`INCIDENT_ACCEPTANCE ${JSON.stringify(result)}`);

    expect(
      messages.filter(message => message.info.role === "assistant"),
    ).toHaveLength(INCIDENT_TURN_COUNT);
    expect(toolParts).toHaveLength(INCIDENT_TOOL_COUNT);
    expect(errorTools).toHaveLength(INCIDENT_ERROR_TOOL_COUNT);
    expect(runningTools).toHaveLength(INCIDENT_RUNNING_TOOL_COUNT);
    expect(deterministic).toHaveLength(INCIDENT_DETERMINISTIC_TURN_COUNT);
    expect(command.summaryRequests.flatMap(summaryRequestIDs)).toHaveLength(
      INCIDENT_STANDARD_TURN_COUNT,
    );
    expect(beforeTokens).toBeGreaterThanOrEqual(300_000);
    expect(beforeTokens).toBeLessThanOrEqual(360_000);
    expect(afterTokens).toBeLessThanOrEqual(195_000);
    expect(stats?.totalTokensPruned).toBe(beforeTokens - afterTokens);
    expect(result.reduction).toBeGreaterThanOrEqual(0.4);
    const cardinalityOverhead =
      INCIDENT_STANDARD_TURN_COUNT
      * (encode(String(INCIDENT_ASSISTANT_REPEAT_COUNT)).length + 32);
    expect(result.modelInputTokens).toBeLessThanOrEqual(
      singleCopyInputTokens + cardinalityOverhead,
    );
    expect(result.requests).toBe(singleCopy.prompts.length);
    expect(
      encode(
        INCIDENT_ASSISTANT_REPEAT_LITERAL.repeat(
          INCIDENT_ASSISTANT_REPEAT_COUNT - 1,
        ),
      ).length * INCIDENT_STANDARD_TURN_COUNT,
    ).toBeGreaterThan(result.modelInputTokens);
    expect(result.maxRequestTokens).toBeLessThanOrEqual(
      MAX_SUMMARY_INPUT_TOKENS,
    );
    expect(command.summaryRequests.length).toBeLessThanOrEqual(
      INCIDENT_STANDARD_TURN_COUNT,
    );
    expect(command.maxSummaryConcurrency).toBeGreaterThanOrEqual(1);
    expect(command.maxSummaryConcurrency).toBeLessThanOrEqual(2);
    for (const request of command.summaryRequests) {
      expect(request).not.toHaveProperty("tools");
      for (const turn of summaryPromptSelection(request).turns) {
        const records = assistantTextRecords(turn.exactEvidence);
        expect(records).toHaveLength(1);
        expect(records[0]!.textEncoding).toBe("ordered-literal-runs-v1");
        expect(
          records[0]!.textSegments?.some(
            segment =>
              segment.literal === INCIDENT_ASSISTANT_REPEAT_LITERAL
              && segment.repetitions === INCIDENT_ASSISTANT_REPEAT_COUNT,
          ),
        ).toBeTrue();
      }
    }
    expect(command.backupForks).toHaveLength(1);
    for (const create of command.scratchCreates) {
      expect(create).not.toHaveProperty("parentID");
      expect(create).toHaveProperty("permission", [
        { permission: "*", pattern: "*", action: "deny" },
      ]);
    }
    for (const summary of summaries) {
      expect(encode(summary).length).toBeLessThanOrEqual(
        MAX_SUMMARY_OUTPUT_TOKENS,
      );
    }
    for (const summary of deterministic) {
      expect(encode(summary).length).toBeLessThanOrEqual(
        DETERMINISTIC_SUMMARY_MAX_TOKENS,
      );
      expect(summary).toStartWith("Outcome: partial");
      expect(summary).toContain("Unresolved:");
      expect(summary).toContain("[error]");
      for (const scaffolding of [
        '"kind"',
        '"sourceBytes"',
        '"windows"',
        "omitted-record-count",
        "required-record-",
      ]) {
        expect(summary).not.toContain(scaffolding);
      }
    }
  }, 30_000);

  test("renders a measured outlier deterministically between standard batches", async () => {
    const lifecycle: string[] = [];
    const isolated = await runCompaction(session("source"), {
      messages: outlierTurn(),
      lifecycle,
    });
    expect(isolated.prompts).toHaveLength(0);
    expect(isolated.agentRequests).toHaveLength(0);
    expect(isolated.updates).toHaveLength(0);
    expect(lifecycle).not.toContain("fork");
    expect(lifecycle).not.toContain("delete");

    const messages = [
      ...completedTurn("ordinary-before", "Before"),
      ...outlierTurn(),
      ...completedTurn("ordinary-after", "After"),
    ];
    const requests = await runCompaction(session("source"), {
      messages,
      textResponses: [
        JSON.stringify({
          "ordinary-before-assistant": modelCapsule(
            "analysis-only",
            "Before summary.",
          ),
          "ordinary-after-assistant": modelCapsule(
            "analysis-only",
            "After summary.",
          ),
        }),
      ],
    });
    const prompts = requests.prompts.map(promptText);

    expect(prompts).toHaveLength(1);
    expect(requests.updates).toHaveLength(0);
    expect(requests.agentRequests).toHaveLength(1);
    expect(requests.prompts.map(summaryRequestIDs)).toEqual([
      ["ordinary-before-assistant", "ordinary-after-assistant"],
    ]);
    expect(prompts[0]).not.toContain(
      '"assistantTurnID": "outlier-assistant-0"',
    );
    expect(prompts.join("\n")).not.toContain("Replace with your summary");
    expect(requests.partUpdates.map(update => update.messageID)).toEqual([
      "ordinary-before-assistant",
      "outlier-assistant-0",
      "ordinary-after-assistant",
    ]);

    const stored = storedSummary(requests, "outlier-assistant-0");
    expect(stored).toStartWith("Outcome: partial");
    expect(stored).toContain(
      "Work: Delivered implementation in src/critical.ts.",
    );
    expect(stored).toContain("bun test reported 88 pass");
    expect(stored).toContain("commit abc123 was pushed");
    expect(stored).toContain("smoke not run");
    expect(stored).toContain("integration failed");
    expect(stored).toContain("final assistant evidence");
    expect(stored).toContain("Process exited with code 1");
    expect(labeledSummarySection(stored, "VCS", "Unresolved")).toContain(
      "commit abc123 was pushed",
    );
    expect(stored).toContain("[running]");
    expect(stored).not.toContain('"kind"');
    expect(stored).not.toContain("omitted-record-count=");
    expect(encode(stored).length).toBeLessThanOrEqual(
      DETERMINISTIC_SUMMARY_MAX_TOKENS,
    );
    expect(hasUnpairedSurrogate(stored)).toBeFalse();
  }, 300_000);

  test("classifies a failure from the middle of full tool output before bounding it", async () => {
    const marker = "Process exited with code 1";
    const messages = completedCommandTurn(
      'bun test "packages/middle suite.test.ts"',
      `${"a".repeat(80_000)}\n${marker}\n${"b".repeat(80_000)}`,
    );
    const requests = await runCompaction(session("source"), { messages });
    const stored = storedSummary(requests, "metric-assistant-0");

    expect(requests.prompts).toHaveLength(0);
    expect(stored).toStartWith("Outcome: partial");
    expect(stored).toContain(marker);
    expect(stored).toContain('bun test "packages/middle suite.test.ts"');
    expect(labeledSummarySection(stored, "Unresolved")).toContain(marker);
  });

  test("includes ignored model-visible assistant text while excluding synthetic, Magic summary, empty, and placeholder text", async () => {
    const turn = metricTurn(1, 20, 0);
    const assistant = turn.assistants.at(-1)!;
    assistant.parts.push(
      textPart(
        "real-final",
        "source",
        assistant.info.id,
        "Authoritative final response from a real assistant message.",
      ),
      {
        ...textPart(
          "ignored-final",
          "source",
          assistant.info.id,
          "ignored evidence",
        ),
        ignored: true,
      },
      {
        ...textPart(
          "synthetic-final",
          "source",
          assistant.info.id,
          "synthetic evidence",
        ),
        synthetic: true,
      },
      {
        ...textPart(
          "summary-final",
          "source",
          assistant.info.id,
          "old Magic summary",
        ),
        metadata: { magicCompact: { summary: true } },
      },
      textPart("placeholder-final", "source", assistant.info.id, "[TODO]"),
      textPart("empty-final", "source", assistant.info.id, "   "),
    );
    const requests = await runDeterministicRenderer(session("source"), {
      messages: [...turn.user, ...turn.assistants],
    });
    const stored = storedSummary(requests, "metric-assistant-0");

    expect(stored).toStartWith("Outcome: analysis-only");
    expect(stored).toContain("Authoritative final response");
    expect(stored).toContain("ignored evidence");
    expect(stored).not.toContain("synthetic evidence");
    expect(stored).not.toContain("old Magic summary");
    expect(stored).not.toContain("[TODO]");
  });

  test("keeps ignored assistant text counted, evidenced, and represented after pruning", async () => {
    const standardMarker = "STANDARD_IGNORED_ASSISTANT_EVIDENCE";
    const deterministicMarker = "DETERMINISTIC_IGNORED_ASSISTANT_EVIDENCE";
    const ignoredUserNotice = "IGNORED_USER_REPORTING_NOTICE";
    const standard = completedTurn(
      "ignored-standard",
      "Compact standard history",
    );
    standard[0]!.parts.push({
      ...textPart(
        "ignored-user-report",
        "source",
        standard[0]!.info.id,
        ignoredUserNotice,
      ),
      ignored: true,
    });
    standard[1]!.parts.push({
      ...textPart(
        "ignored-standard-text",
        "source",
        standard[1]!.info.id,
        `${standardMarker}${" historical context".repeat(5_000)}`,
      ),
      ignored: true,
    });
    const deterministic = completedTurn(
      "ignored-deterministic",
      "Compact unresolved history",
    );
    deterministic[1]!.parts.push(
      pendingCommandToolPart(
        "ignored-deterministic-pending",
        deterministic[1]!.info.id,
        "wait-for-operator",
      ),
      {
        ...textPart(
          "ignored-deterministic-text",
          "source",
          deterministic[1]!.info.id,
          `${deterministicMarker}${" unresolved context".repeat(5_000)}`,
        ),
        ignored: true,
      },
    );
    const messages = [...standard, ...deterministic];
    addProviderTokens(messages);
    const withoutUserNotice = structuredClone(messages);
    withoutUserNotice[0]!.parts = withoutUserNotice[0]!.parts.filter(
      part => part.id !== "ignored-user-report",
    );
    const withoutIgnoredAssistantText = structuredClone(messages);
    for (const message of withoutIgnoredAssistantText) {
      message.parts = message.parts.filter(
        part =>
          part.id !== "ignored-standard-text"
          && part.id !== "ignored-deterministic-text",
      );
    }
    const beforeTokens = countCanonicalContentTokens(messages);
    const command = createCommandHarness(messages, {
      summaryTextOutputs: [
        JSON.stringify({
          "ignored-standard-assistant": modelCapsule(
            "analysis-only",
            `Preserved ${standardMarker}.`,
          ),
        }),
      ],
    });

    try {
      await Promise.all([
        deleteStats("source"),
        deleteStats("backup"),
        rm(cachePath("source"), { force: true }),
        rm(cachePath("backup"), { force: true }),
      ]);

      expect(beforeTokens).toBe(countCanonicalContentTokens(withoutUserNotice));
      expect(beforeTokens).toBeGreaterThan(
        countCanonicalContentTokens(withoutIgnoredAssistantText),
      );
      expect(await executeMagicCompact(command.v2, "source", 0)).toBeTrue();

      const standardEvidence = command.summaryRequests
        .flatMap(request => summaryPromptSelection(request).turns)
        .find(
          turn => turn.assistantTurnID === "ignored-standard-assistant",
        )!.exactEvidence;
      expect(standardEvidence).toContain(standardMarker);
      expect(standardEvidence).not.toContain(ignoredUserNotice);
      expect(
        storedSummary(command, "ignored-deterministic-assistant"),
      ).toContain(deterministicMarker);
      expect(storedSummary(command, "ignored-standard-assistant")).toContain(
        standardMarker,
      );
      expect(command.partDeletes).toContainEqual(
        expect.objectContaining({ partID: "ignored-standard-text" }),
      );
      expect(command.partDeletes).toContainEqual(
        expect.objectContaining({ partID: "ignored-deterministic-text" }),
      );
      const activeText = messages
        .flatMap(message => message.parts)
        .filter((part): part is TextPart => part.type === "text")
        .map(part => part.text)
        .join("\n");
      expect(activeText).toContain(standardMarker);
      expect(activeText).toContain(deterministicMarker);
      expect(activeText).toContain(ignoredUserNotice);
    } finally {
      await Promise.all([
        deleteStats("source"),
        deleteStats("backup"),
        rm(cachePath("source"), { force: true }),
        rm(cachePath("backup"), { force: true }),
      ]);
    }
  });

  test("preserves the full final substantive response when it fits the capsule ceiling", async () => {
    const messages = completedTurn("full-final", "Finish the request");
    const assistant = messages[1]!;
    const finalText = [
      "Implemented the requested continuation-safe capsule.",
      "Context before the governing decision: " + "history ".repeat(90),
      "Decision: keep retrieval tool-only and preserve the exact continuation state.",
      "Context after the governing decision: " + "result ".repeat(70),
      "The implementation is ready for the remaining credential check.",
    ].join("\n");
    assistant.parts.push(
      pendingCommandToolPart(
        "full-final-pending",
        assistant.info.id,
        "check-provider-credentials",
      ),
      textPart("full-final-text", "source", assistant.info.id, finalText),
    );

    const requests = await runCompaction(session("source"), { messages });
    const stored = storedSummary(requests, assistant.info.id);
    const collapsed = finalText.replace(/\s+/g, " ").trim();

    expect(requests.prompts).toEqual([]);
    expect(stored).toContain(`Work: ${collapsed}`);
    expect(stored).toContain("Decision: keep retrieval tool-only");
  });

  test("preserves multipart final assistant text as one ordered response", async () => {
    const messages = completedTurn("multipart-final", "Finish the request");
    const assistant = messages[1]!;
    const parts = [
      "Implemented the readable capsule correction.",
      "Decision: preserve every substantive part of the final assistant message in source order.",
      "The remaining operator action is documented below.",
    ];
    assistant.parts.push(
      pendingCommandToolPart(
        "multipart-final-pending",
        assistant.info.id,
        "wait-for-operator-approval",
      ),
      ...parts.map((text, index) =>
        textPart(
          `multipart-final-text-${index}`,
          "source",
          assistant.info.id,
          text,
        ),
      ),
    );

    const requests = await runCompaction(session("source"), { messages });
    const stored = storedSummary(requests, assistant.info.id);

    expect(requests.prompts).toEqual([]);
    expect(stored).toContain(`Work: ${parts.join(" ")}`);
    expect(
      stored.match(/Decision: preserve every substantive part/g),
    ).toHaveLength(1);
  });

  test("requires a middle decision from an oversized multipart final response", async () => {
    const messages = completedTurn("multipart-cue", "Preserve the decision");
    const assistant = messages[1]!;
    const decision =
      "Decision: keep source-anchored continuation cues from every final response part.";
    assistant.parts.push(
      pendingCommandToolPart(
        "multipart-cue-pending",
        assistant.info.id,
        "wait-for-operator-approval",
      ),
      textPart(
        "multipart-cue-opening",
        "source",
        assistant.info.id,
        `Opening context ${"history ".repeat(2_500)}`,
      ),
      textPart("multipart-cue-decision", "source", assistant.info.id, decision),
      textPart(
        "multipart-cue-closing",
        "source",
        assistant.info.id,
        `Closing context ${"result ".repeat(2_500)}`,
      ),
    );

    const requests = await runCompaction(session("source"), { messages });
    const stored = storedSummary(requests, assistant.info.id);

    expect(requests.prompts).toEqual([]);
    expect(stored).toContain(decision);
    expect(encode(stored).length).toBeLessThanOrEqual(
      DETERMINISTIC_SUMMARY_MAX_TOKENS,
    );
  });

  test("preserves every classified final-response cue when the full response cannot fit", async () => {
    const messages = completedTurn("anchored-final", "Recover and continue");
    const assistant = messages[1]!;
    const filler = (label: string) => `${label} ${"context ".repeat(1_200)}`;
    const facts = [
      "Decision: preserve tool-only retrieval through read_omitted_content.",
      "Verification: bun test passed 42 tests.",
      "VCS: the changes are not committed.",
      "Blocked: provider credentials are missing.",
      "Next action: refresh PROVIDER_TOKEN and rerun bun test.",
    ];
    const finalText = [
      filler("opening"),
      facts[0],
      filler("decision-tail"),
      facts[1],
      filler("verification-tail"),
      facts[2],
      filler("vcs-tail"),
      facts[3],
      filler("blocked-tail"),
      facts[4],
      filler("closing"),
    ].join("\n");
    assistant.parts.push(
      textPart("anchored-final-text", "source", assistant.info.id, finalText),
    );

    const requests = await runCompaction(session("source"), { messages });
    const stored = storedSummary(requests, assistant.info.id);

    expect(requests.prompts).toEqual([]);
    for (const fact of facts) expect(stored).toContain(fact);
    expect(encode(stored).length).toBeLessThanOrEqual(
      DETERMINISTIC_SUMMARY_MAX_TOKENS,
    );
  });

  test("keeps distinct long next actions that collide outside the cue window", async () => {
    const messages = completedTurn(
      "action-collision",
      "Preserve both continuation actions",
    );
    const shared =
      "Next action: refresh PROVIDER_TOKEN and continue the documented operator runbook "
      + "step ".repeat(40);
    const first = `${shared} ALPHA-UNIQUE /workspace/alpha-state.json`;
    const second = `${shared} BETA-UNIQUE /workspace/beta-state.json`;
    messages[1]!.parts.push(
      pendingCommandToolPart(
        "action-collision-pending",
        messages[1]!.info.id,
        "wait-for-operator-approval",
      ),
      textPart(
        "action-collision-text",
        "source",
        messages[1]!.info.id,
        [
          `Opening context ${"history ".repeat(2_500)}`,
          first,
          second,
          `Closing context ${"result ".repeat(2_500)}`,
        ].join("\n"),
      ),
    );

    const requests = await runCompaction(session("source"), { messages });
    const stored = storedSummary(requests, messages[1]!.info.id);

    expect(requests.prompts).toEqual([]);
    expect(stored).toContain("ALPHA-UNIQUE");
    expect(stored).toContain("BETA-UNIQUE");
  });

  test("fails closed when distinct final-response decisions cannot fit the capsule ceiling", async () => {
    const messages = completedTurn("decision-overflow", "Preserve decisions");
    const assistant = messages[1]!;
    assistant.parts.push(
      pendingCommandToolPart(
        "decision-overflow-pending",
        assistant.info.id,
        "wait-for-credentials",
      ),
      textPart(
        "decision-overflow-text",
        "source",
        assistant.info.id,
        Array.from(
          { length: 180 },
          (_, index) =>
            `Decision ${index}: preserve /workspace/decision-${index}/${"segment-".repeat(12)}state.json for continuation.`,
        ).join("\n"),
      ),
    );
    const plan = await planFor(messages, 0);

    expect(() => renderDeterministicSummary(plan.summarizedTurns[0]!)).toThrow(
      "local safety cap",
    );
  });

  test("preserves every distinct unresolved repair action while it fits", async () => {
    const turn = metricTurn(1, 20, 0);
    const assistant = turn.assistants[0]!;
    assistant.parts.push(
      ...Array.from({ length: 12 }, (_, index) =>
        pendingCommandToolPart(
          `repair-${index}`,
          assistant.info.id,
          `repair-${index} --target /workspace/repair-${index}.json`,
        ),
      ),
    );

    const requests = await runCompaction(session("source"), {
      messages: [...turn.user, ...turn.assistants],
    });
    const stored = storedSummary(requests, "metric-assistant-0");

    for (let index = 0; index < 12; index += 1) {
      expect(stored).toContain(
        `repair-${index} --target /workspace/repair-${index}.json`,
      );
    }
    expect(stored).not.toContain("×12");
  });

  test("aggregates only repeated equivalent unresolved facts", async () => {
    const turn = metricTurn(1, 20, 0);
    const assistant = turn.assistants[0]!;
    assistant.parts.push(
      ...Array.from({ length: 400 }, (_, index) =>
        pendingCommandToolPart(
          `equivalent-repair-${index}`,
          assistant.info.id,
          "repair-shared-state --target /workspace/shared.json",
        ),
      ),
    );

    const requests = await runCompaction(session("source"), {
      messages: [...turn.user, ...turn.assistants],
    });
    const stored = storedSummary(requests, "metric-assistant-0");

    expect(stored).toContain("×400");
    expect(stored).toContain(
      "repair-shared-state --target /workspace/shared.json",
    );
  });

  test("does not aggregate distinct raw facts that share one rendered excerpt", async () => {
    const turn = metricTurn(1, 20, 0);
    const assistant = turn.assistants[0]!;
    const sharedPrefix = "shared failure context ".repeat(8);
    const sharedSuffix = " shared terminal suffix";
    assistant.parts.push(
      ...Array.from({ length: 300 }, (_, index) =>
        genericToolPart(`raw-fact-${index}`, assistant.info.id, "bash", {
          status: "error",
          input: { command: "repair-shared-state" },
          error: `${sharedPrefix}distinct fact ${index}${sharedSuffix}`,
          metadata: {},
          time: { start: 1, end: 2 },
        }),
      ),
    );

    await expect(
      runDeterministicRenderer(session("source"), {
        messages: [...turn.user, ...turn.assistants],
      }),
    ).rejects.toThrow("local safety cap");
  });

  test("aggregates interleaved duplicate unresolved facts at first-occurrence positions", async () => {
    const turn = metricTurn(1, 20, 0);
    const assistant = turn.assistants[0]!;
    const commands = [
      ...Array.from(
        { length: 600 },
        () => "repair-shared-state --target /workspace/group-A.json",
      ),
      ...Array.from(
        { length: 600 },
        () => "reinstall-deps --target /workspace/group-B.json",
      ),
      "single-retry --target /workspace/group-E.json",
      ...Array.from(
        { length: 600 },
        () => "bump-version --target /workspace/group-C.json",
      ),
      ...Array.from(
        { length: 600 },
        () => "rotate-secrets --target /workspace/group-D.json",
      ),
    ];
    assistant.parts.push(
      ...commands.map((command, index) =>
        pendingCommandToolPart(
          `interleaved-${index}`,
          assistant.info.id,
          command,
        ),
      ),
    );

    const requests = await runDeterministicRenderer(session("source"), {
      messages: [...turn.user, ...turn.assistants],
    });
    const stored = storedSummary(requests, "metric-assistant-0");

    const unresolved = capsuleSection(stored, "Unresolved");
    const bullets = unresolved
      .split("\n")
      .filter(line => line.startsWith("- "));
    const markers = [
      "- [pending] repair-shared-state --target /workspace/group-A.json ×600",
      "- [pending] reinstall-deps --target /workspace/group-B.json ×600",
      "- [pending] single-retry --target /workspace/group-E.json",
      "- [pending] bump-version --target /workspace/group-C.json ×600",
      "- [pending] rotate-secrets --target /workspace/group-D.json ×600",
    ];
    const positions = markers.map(marker =>
      bullets.findIndex(line => line.includes(marker)),
    );
    expect(positions.every(position => position >= 0)).toBe(true);
    expect(positions).toEqual(
      [...positions].toSorted((left, right) => left - right),
    );
    expect(bullets[0]).toBe(markers[0]);
    expect(bullets.filter(line => line.endsWith("×600")).length).toBe(4);
    expect(bullets.some(line => /×[12]$/.test(line))).toBe(false);
    expect(
      bullets.filter(
        line => line.includes("group-A.json") && line.includes("×"),
      ).length,
    ).toBe(1);
  });

  test("fails closed promptly on one hundred twenty-eight thousand distinct unresolved records", () => {
    const turn = distinctUnresolvedTurn(128_000);

    expect(() => renderDeterministicSummary(turn)).toThrow("local safety cap");
  }, 60_000);

  test("classifies interleaved assistant errors with exact recovered/error semantics and recovery IDs", () => {
    const user = [message("status-user", "user", [])];
    const errorMessage = (
      id: string,
      name: string,
      data: Record<string, unknown>,
    ) =>
      ({
        info: { id, role: "assistant", error: { name, data } } as Message,
        parts: [],
      }) satisfies MessageWithParts;
    const successMessage = (id: string) =>
      ({
        info: { id, role: "assistant", finish: "stop" } as Message,
        parts: [],
      }) satisfies MessageWithParts;
    const turn: Turn = {
      user,
      assistants: [
        errorMessage("err-1", "ProviderError", {
          message: "boom 1",
          statusCode: 500,
          providerID: "test-provider",
        }),
        successMessage("ok-1"),
        errorMessage("err-2", "RateLimitError", { message: "slow down" }),
        successMessage("ok-2"),
        errorMessage("err-3", "ProviderError", { message: "final boom" }),
      ],
    };

    const capsule = renderDeterministicSummary(turn);
    expect(capsule).toContain("Outcome: partial");
    expect(capsule).toContain(
      "Recovered from ProviderError: boom 1, status 500, test-provider",
    );
    expect(capsule).toContain("Recovered from RateLimitError: slow down");
    expect(
      capsule.indexOf(
        "Recovered from ProviderError: boom 1, status 500, test-provider",
      ),
    ).toBeLessThan(capsule.indexOf("Recovered from RateLimitError: slow down"));
    expect(capsule).toContain("[error] ProviderError: final boom");

    const statusRecords = machineEvidenceRecords(
      renderStandardTurnEvidence(turn),
    ).filter(record => record["kind"] === "assistant-message-status");
    const byMessageID = (id: string) =>
      statusRecords.find(record => record["messageID"] === id);
    expect(byMessageID("err-1")).toEqual({
      kind: "assistant-message-status",
      messageID: "err-1",
      status: "recovered",
      errorType: "ProviderError",
      categories: ["recovered"],
      polarity: "reported",
      message: "boom 1",
      statusCode: 500,
      providerID: "test-provider",
      recoveredByMessageID: "ok-1",
    });
    expect(byMessageID("err-2")).toEqual({
      kind: "assistant-message-status",
      messageID: "err-2",
      status: "recovered",
      errorType: "RateLimitError",
      categories: ["recovered"],
      polarity: "reported",
      message: "slow down",
      recoveredByMessageID: "ok-2",
    });
    expect(byMessageID("err-3")).toEqual({
      kind: "assistant-message-status",
      messageID: "err-3",
      status: "error",
      errorType: "ProviderError",
      categories: ["failure"],
      polarity: "failed",
      message: "final boom",
    });
  });

  test("fails closed promptly on two hundred fifty-six thousand assistant errors without a later success", () => {
    const turn = statusChainTurn(256_000);

    expect(() => renderDeterministicSummary(turn)).toThrow("local safety cap");
  }, 60_000);

  test("caps dense required pending evidence with readable aggregation", async () => {
    const turn = metricTurn(1, 20, 0);
    const assistant = turn.assistants[0]!;
    const latePaths = Array.from(
      { length: 16 },
      (_, index) => `/workspace/late/anchor-${index}.ts`,
    );
    assistant.parts.push(
      ...Array.from({ length: 20 }, (_, index) =>
        pendingCommandToolPart(
          `pending-${index}`,
          assistant.info.id,
          `bun run smoke --case ${index} --config /workspace/pending-${index}.json`,
        ),
      ),
      ...Array.from({ length: 60 }, (_, index) =>
        commandToolPart(
          `optional-${index}`,
          assistant.info.id,
          `printf optional-${index}`,
          `optional ${index} ${"x".repeat(4_000)}`,
        ),
      ),
      commandToolPart(
        "late-failure",
        assistant.info.id,
        'bun test "packages/late suite.test.ts"',
        `Process exited with code 1 at "/workspace/project/late failure.ts" and ./src/relative.ts ${latePaths.join(" ")}`,
      ),
    );
    turn.assistants
      .at(-1)!
      .parts.push(
        textPart(
          "late-final",
          "source",
          turn.assistants.at(-1)!.info.id,
          `${"🚀".repeat(50_000)} Final response: checks in ../shared/check.ts were not run.`,
        ),
      );
    const requests = await runCompaction(session("source"), {
      messages: [...turn.user, ...turn.assistants],
    });
    const stored = storedSummary(requests, "metric-assistant-0");

    expect(requests.prompts).toEqual([]);
    expect(stored).toStartWith("Outcome: partial");
    expect(stored).toContain("Unresolved:");
    expect(stored).toContain("[pending]");
    expect(stored).toContain(
      "bun run smoke --case 0 --config /workspace/pending-0.json",
    );
    expect(stored).toContain(
      "bun run smoke --case 19 --config /workspace/pending-19.json",
    );
    expect(stored).toContain('bun test "packages/late suite.test.ts"');
    expect(stored).toContain("Process exited with code 1");
    expect(stored).toContain("checks in ../shared/check.ts were not run");
    expect(stored).not.toContain("required-record-overflow=");
    expect(encode(stored).length).toBeLessThanOrEqual(
      DETERMINISTIC_SUMMARY_MAX_TOKENS,
    );
  }, 30_000);

  test("fails closed instead of dropping one hundred distinct not-run repairs", async () => {
    const turn = metricTurn(1, 20, 0);
    const assistant = turn.assistants[0]!;
    assistant.parts.push(
      ...Array.from({ length: 100 }, (_, index) => {
        const path = `/workspace/required/not-run-${index}.test.ts`;
        return commandToolPart(
          `required-not-run-${index}`,
          assistant.info.id,
          `bun test "${path}" --case required-${index}`,
          `Tests at ${path} were not run; required-anchor-${index}.`,
        );
      }),
    );
    const partUpdates: Record<string, unknown>[] = [];
    await expect(
      runDeterministicRenderer(session("source"), {
        messages: [...turn.user, ...turn.assistants],
        partUpdates,
      }),
    ).rejects.toThrow("local safety cap");
    expect(partUpdates).toEqual([]);
  });

  test("fails closed when required unresolved evidence cannot fit the safety cap", async () => {
    const turn = metricTurn(1, 20, 0);
    const assistant = turn.assistants[0]!;
    assistant.parts.push(
      ...Array.from({ length: 300 }, (_, index) =>
        genericToolPart(
          `distinct-error-${index}`,
          assistant.info.id,
          `runner-${index}`,
          {
            status: "error",
            input: { command: `run-${index} --case ${index}` },
            error: `distinct failure ${index}: ${"detail ".repeat(60)}`,
            metadata: {},
            time: { start: 1, end: 2 },
          },
        ),
      ),
    );

    await expect(
      runDeterministicRenderer(session("source"), {
        messages: [...turn.user, ...turn.assistants],
      }),
    ).rejects.toThrow("local safety cap");
  });

  test("compacts adjacent turns with many error tools while retaining every error tool part", async () => {
    const errorTurn = metricTurn(1, 2, 0);
    errorTurn.user[0]!.info.id = "overflow-user";
    errorTurn.assistants[0]!.info.id = "overflow-assistant";
    const errorPartIDs = Array.from(
      { length: 30 },
      (_, index) =>
        `overflow-error-${index}-${"stable-identifier-".repeat(12)}`,
    );
    errorTurn.assistants[0]!.parts = errorPartIDs.map((partID, index) =>
      genericToolPart(
        partID,
        "overflow-assistant",
        index % 2 === 0 ? "bash" : "read",
        {
          status: "error",
          input:
            index % 2 === 0
              ? { command: `bun test --case overflow-${index}` }
              : { filePath: `/workspace/missing-${index}.ts` },
          error: `Required failure ${index}: Process exited with code 1`,
          metadata: {},
          time: { start: 1, end: 2 },
        },
      ),
    );
    errorTurn.assistants[0]!.parts.push(
      {
        id: "overflow-removable-detail",
        sessionID: "source",
        messageID: "overflow-assistant",
        type: "reasoning",
        text: "removable historical detail ".repeat(4_000),
        time: { start: 1, end: 2 },
      } as Part,
      textPart(
        "overflow-final",
        "source",
        "overflow-assistant",
        "The requested checks remain failed; inspect the retained error tool records.",
      ),
    );
    const messages = [
      ...errorTurn.user,
      ...errorTurn.assistants,
      message("overflow-adjacent-user", "user", [
        textPart(
          "overflow-adjacent-user-text",
          "source",
          "overflow-adjacent-user",
          "Continue after failures",
        ),
      ]),
      message("overflow-adjacent-assistant", "assistant", [
        genericToolPart(
          "overflow-adjacent-error",
          "overflow-adjacent-assistant",
          "bash",
          {
            status: "error",
            input: { command: "bun test --case adjacent" },
            error: "Adjacent required failure: Process exited with code 1",
            metadata: {},
            time: { start: 1, end: 2 },
          },
        ),
      ]),
    ];
    addProviderTokens(messages);
    await rm(statsPath("source"), { force: true });
    await rm(cachePath("source"), { force: true });
    const command = createCommandHarness(messages);

    expect(await executeMagicCompact(command.v2, "source", 0)).toBe(true);
    const stored = storedSummary(command, "overflow-assistant");
    const activeErrors = messages
      .flatMap(message => message.parts)
      .filter(
        (part): part is ToolPart =>
          part.type === "tool" && part.state.status === "error",
      );
    console.info(
      `OVERFLOW_STRESS ${JSON.stringify({ summaryTokens: encode(stored).length, activeErrorTools: activeErrors.length, prompts: command.summaryRequests.length })}`,
    );

    expect(command.summaryRequests).toEqual([]);
    expect(stored).toStartWith("Outcome: partial");
    expect(stored).toContain("Unresolved:");
    expect(stored).toContain("[error]");
    expect(stored).toContain("bun test --case overflow-0");
    expect(stored).toContain("bun test --case overflow-28");
    expect(stored).toContain("/workspace/missing-1.ts");
    expect(stored).toContain("/workspace/missing-29.ts");
    expect(stored).toContain("Process exited with code 1");
    expect(stored).not.toContain("required-record-counts=");
    expect(stored).not.toContain("required-record-overflow=");
    expect(encode(stored).length).toBeLessThanOrEqual(
      DETERMINISTIC_SUMMARY_MAX_TOKENS,
    );
    expect(activeErrors).toHaveLength(31);
    expect(activeErrors.slice(0, 30).map(part => part.id)).toEqual(
      errorPartIDs,
    );
    expect(command.partDeletes).not.toContainEqual(
      expect.objectContaining({
        partID: expect.stringContaining("overflow-error"),
      }),
    );
    expect((await readStats("source"))?.totalTokensPruned).toBeGreaterThan(0);
  });

  test("keeps resolved verification optional when unresolved evidence makes a turn deterministic", async () => {
    const turn = metricTurn(1, 20, 0);
    const assistant = turn.assistants[0]!;
    assistant.parts.push(
      genericToolPart("running", assistant.info.id, "bash", {
        status: "running",
        input: { command: "synthetic-watch" },
        metadata: {},
        time: { start: 1 },
      }),
      ...Array.from({ length: 100 }, (_, index) =>
        commandToolPart(
          `resolved-${index}`,
          assistant.info.id,
          `bun test --case resolved-${index}`,
          "1 passed, 0 failed",
        ),
      ),
    );

    const requests = await runDeterministicRenderer(session("source"), {
      messages: [...turn.user, ...turn.assistants],
    });
    const stored = storedSummary(requests, "metric-assistant-0");

    expect(requests.prompts).toEqual([]);
    expect(stored).toContain("[running]");
    expect(stored).toContain("synthetic-watch");
    expect(stored).toContain("Unresolved:");
    expect(stored).not.toContain("omitted-record-count=");
    expect(encode(stored).length).toBeLessThanOrEqual(
      DETERMINISTIC_SUMMARY_MAX_TOKENS,
    );
  });

  test("routes completed final response parts to standard summarization regardless of count", async () => {
    const turn = metricTurn(1, 20, 0);
    const finalAssistant = turn.assistants.at(-1)!;
    finalAssistant.parts.push(
      ...Array.from({ length: 120 }, (_, index) =>
        textPart(
          `required-final-${index}`,
          "source",
          finalAssistant.info.id,
          `Required final response ${index} at /workspace/final-${index}.ts: ${"x".repeat(700)}`,
        ),
      ),
    );
    const messages = [...turn.user, ...turn.assistants];
    addProviderTokens(messages);
    const command = createCommandHarness(messages);
    expect(await executeMagicCompact(command.v2, "source", 0)).toBe(true);
    expect(command.summaryRequests).toHaveLength(1);
    expect(command.scratchCreates).toHaveLength(1);
    expect(command.events).toContain("summary-update");
    expect(command.backupForks).toHaveLength(1);
  });

  test("requires every substantive final-message text part as exact standard evidence", async () => {
    const messages = completedTurn("multipart-exact", "Deliver the change");
    const parts = Array.from(
      { length: 40 },
      (_, index) =>
        `FINAL-PART-${index} ${"body".repeat(120)} /workspace/final-${index}.ts`,
    );
    messages[1]!.parts.push(
      ...parts.map((text, index) =>
        textPart(
          `multipart-exact-${index}`,
          "source",
          messages[1]!.info.id,
          text,
        ),
      ),
    );

    const requests = await runCompaction(session("source"), { messages });
    const evidence = summaryRequestEvidence(requests.prompts[0]!);

    expect(requests.prompts).toHaveLength(1);
    expect(requests.creates).toHaveLength(1);
    expect(requests.forks).toEqual([]);
    for (const text of parts) expect(evidence).toContain(text);
    expect(
      labeledSummarySection(
        evidence,
        "Historical assistant text (authoritative)",
        "Verification",
      ),
    ).not.toMatch(/omitted-record-count=[1-9]/);
  });

  test("encodes a whole-part periodic source with literal cardinality", () => {
    const literal = " repeated historical context";
    const source = literal.repeat(137);
    const record = assistantTextRecords(
      standardEvidenceForAssistantText(source),
    )[0]!;

    expect(record.textEncoding).toBe("ordered-literal-runs-v1");
    expect(record.textSegments).toEqual([{ literal, repetitions: 137 }]);
    expect(decodeAssistantTextRecord(record)).toBe(source);
  });

  test("encodes an exact periodic suffix after a unique prefix", () => {
    const literal = " repeated historical context";
    const prefix =
      "Synthetic partial delivery with exact terminal state below.";
    const source = `${prefix}${literal.repeat(137)}`;
    const record = assistantTextRecords(
      standardEvidenceForAssistantText(source),
    )[0]!;

    expect(record.textEncoding).toBe("ordered-literal-runs-v1");
    expect(record.textSegments).toEqual([
      { literal: prefix, repetitions: 1 },
      { literal, repetitions: 137 },
    ]);
    expect(decodeAssistantTextRecord(record)).toBe(source);
  });

  test("keeps near-identical repeated runs uncompressed", () => {
    const repeated = " alpha";
    const source = `head${repeated.repeat(80)} alphx${repeated.repeat(60)}tail`;
    const record = assistantTextRecords(
      standardEvidenceForAssistantText(source),
    )[0]!;

    expect(record.text).toBe(source);
    expect(record.textEncoding).toBeUndefined();
    expect(decodeAssistantTextRecord(record)).toBe(source);
  });

  test("keeps mixed exact runs with unique separators uncompressed", () => {
    const first = " first-run";
    const second = " second-run";
    const source = `opening${first.repeat(90)}middle${second.repeat(70)}closing`;
    const record = assistantTextRecords(
      standardEvidenceForAssistantText(source),
    )[0]!;

    expect(record.text).toBe(source);
    expect(record.textEncoding).toBeUndefined();
    expect(decodeAssistantTextRecord(record)).toBe(source);
  });

  test("keeps a trailing unique segment from compressing an exact run", () => {
    const literal = " repeated historical context";
    const source = `${literal.repeat(80)}closing`;
    const record = assistantTextRecords(
      standardEvidenceForAssistantText(source),
    )[0]!;

    expect(record.text).toBe(source);
    expect(record.textEncoding).toBeUndefined();
    expect(decodeAssistantTextRecord(record)).toBe(source);
  });

  test("keeps short, low-cardinality, or ineligible periodic units uncompressed", () => {
    const sources = [
      "ab".repeat(400),
      "abcdef".repeat(100),
      " abcd".repeat(7),
      "abcdefgh".repeat(33),
    ];
    for (const source of sources) {
      const record = assistantTextRecords(
        standardEvidenceForAssistantText(source),
      )[0]!;
      expect(record.text).toBe(source);
      expect(record.textEncoding).toBeUndefined();
      expect(decodeAssistantTextRecord(record)).toBe(source);
    }
  });

  test("encodes short-period eligible units through an exact larger multiple", () => {
    const source = "ab ".repeat(150);
    const record = assistantTextRecords(
      standardEvidenceForAssistantText(source),
    )[0]!;

    expect(record.textEncoding).toBe("ordered-literal-runs-v1");
    expect(record.textSegments).toEqual([
      { literal: "ab ab ", repetitions: 75 },
    ]);
    expect(decodeAssistantTextRecord(record)).toBe(source);
  });

  test("round-trips a deterministic fuzz corpus through the run encoding", () => {
    let seed = 42_424;
    const next = () => {
      seed = (seed * 1_103_515_245 + 12_345) % 2_147_483_648;
      return seed / 2_147_483_648;
    };
    const alphabet =
      "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.\t\n\u0000\ud800\ufffe";
    const sources: string[] = [];
    for (let sample = 0; sample < 60; sample++) {
      const mode = Math.floor(next() * 4);
      const unitLength = 2 + Math.floor(next() * 30);
      let unit = "";
      while (unit.length < unitLength) {
        unit += alphabet[Math.floor(next() * alphabet.length)];
      }
      const body = unit.repeat(3 + Math.floor(next() * 30));
      if (mode === 0) sources.push(body);
      else if (mode === 1) sources.push(`prefix ${sample} ${body}`);
      else if (mode === 2) sources.push(`prefix ${body} suffix ${sample}`);
      else sources.push(`${body} tail ${sample}`);
    }
    for (const source of sources) {
      const record = assistantTextRecords(
        standardEvidenceForAssistantText(source),
      )[0]!;
      expect(codeUnits(decodeAssistantTextRecord(record))).toEqual(
        codeUnits(source),
      );
    }
  });

  test("round-trips repeated Unicode and invalid XML JavaScript code units", () => {
    const literal = " 🚀\u0000\ud800\ufffe";
    const source = literal.repeat(80);
    const evidence = standardEvidenceForAssistantText(source);
    const record = assistantTextRecords(evidence)[0]!;

    expect(record.textEncoding).toBe("ordered-literal-runs-v1");
    expect(record.textSegments).toEqual([{ literal, repetitions: 80 }]);
    expect(codeUnits(decodeAssistantTextRecord(record))).toEqual(
      codeUnits(source),
    );
    expect(evidence).not.toContain("\u0000");
    expect(hasUnpairedSurrogate(evidence)).toBeFalse();
  });

  test("detects no literal run in high-entropy text within a bounded time", () => {
    const source = deterministicHighEntropyText(100_000);
    const start = performance.now();
    const runs = encodeOrderedLiteralRuns(source);
    const elapsed = performance.now() - start;

    expect(runs).toBeNull();
    expect(elapsed).toBeLessThan(2_000);
  }, 30_000);

  test.each([
    [100_000, 5_000],
    [500_000, 12_000],
    [1_000_000, 20_000],
  ])(
    "rejects unique high-entropy evidence of %i bytes within %i ms",
    (bytes, bound) => {
      const source = deterministicHighEntropyText(bytes);
      const turn = {
        user: [message("perf-user", "user", [])],
        assistants: [
          message("perf-assistant", "assistant", [
            textPart("perf-text", "source", "perf-assistant", source),
          ]),
        ],
      };

      const start = performance.now();
      let rejected = false;
      try {
        renderStandardTurnEvidence(turn);
      } catch (error) {
        rejected = /summary input cap/.test(String(error));
      }
      const elapsed = performance.now() - start;

      expect(rejected).toBeTrue();
      expect(elapsed).toBeLessThan(bound);
    },
    120_000,
  );

  test("emits categorized standard terminal evidence records once in source order", () => {
    const messages = completedTurn("standard-categories", "Check the worktree");
    const assistant = messages[1]!;
    assistant.parts.push(
      commandToolPart(
        "standard-verification",
        assistant.info.id,
        "bun test",
        "10 passed, 0 failed",
      ),
      commandToolPart(
        "standard-vcs",
        assistant.info.id,
        "git status --short",
        "working tree clean",
      ),
    );
    const evidence = renderStandardTurnEvidence({
      user: [messages[0]!],
      assistants: [assistant],
    });
    const terminalRecords = machineEvidenceRecords(evidence).filter(
      record => record.kind === "tool-terminal",
    );

    expect(terminalRecords.map(record => record.index)).toEqual([1, 2]);
    expect(terminalRecords[0]!.categories).toContain("verification");
    expect(terminalRecords[1]!.categories).toContain("vcs");
  });

  test("fails closed before scratch when required final-message evidence cannot fit", async () => {
    const messages = completedTurn("multipart-overflow", "Deliver the change");
    messages[1]!.parts.push(
      ...Array.from({ length: 8 }, (_, index) =>
        textPart(
          `multipart-overflow-${index}`,
          "source",
          messages[1]!.info.id,
          `OVERFLOW-PART-${index} ${"x".repeat(400_000)}`,
        ),
      ),
    );
    const plan = await planFor(messages, 0);
    const creates: unknown[] = [];
    const v2 = {
      session: {
        create: async () => {
          creates.push("create");
          throw new Error("scratch should not be created");
        },
      },
    } as unknown as V2Client;

    await expect(
      generateCompactionSummaries(v2, session("source"), plan),
    ).rejects.toThrow("summary input cap");
    expect(creates).toEqual([]);
  });

  test("rejects unique oversized evidence within a bounded time without scratch work", async () => {
    const messages = completedTurn("unique-oversized", "Deliver the change");
    messages[1]!.parts.push(
      textPart(
        "unique-oversized-evidence",
        "source",
        messages[1]!.info.id,
        deterministicHighEntropyText(1_000_000),
      ),
    );
    const plan = await planFor(messages, 0);
    const calls: string[] = [];
    const v2 = {
      session: {
        create: async () => {
          calls.push("create");
          throw new Error("scratch should not be created");
        },
      },
      part: {
        update: async () => {
          calls.push("part.update");
          throw new Error("source should not be mutated");
        },
      },
    } as unknown as V2Client;

    const start = performance.now();
    await expect(
      generateCompactionSummaries(v2, session("source"), plan),
    ).rejects.toThrow("summary input cap");
    const elapsed = performance.now() - start;

    expect(calls).toEqual([]);
    expect(elapsed).toBeLessThan(20_000);
  }, 120_000);

  test("preserves quoted, spaced, absolute, and relative command paths", async () => {
    const turn = metricTurn(1, 20, 0);
    const assistant = turn.assistants[0]!;
    const commands = [
      'bun test "packages/unit suite.test.ts"',
      "cargo test --manifest-path /workspace/service/Cargo.toml",
      'git diff -- "./src/file with spaces.ts"',
      "git status --short ../shared/config.ts ~/project/readme.md",
    ];
    assistant.parts.push(
      ...commands.map((command, index) =>
        commandToolPart(
          `path-${index}`,
          assistant.info.id,
          command,
          index < 2 ? "10 passed, 0 failed" : "working tree clean",
        ),
      ),
    );
    const requests = await runDeterministicRenderer(session("source"), {
      messages: [...turn.user, ...turn.assistants],
    });
    const stored = storedSummary(requests, "metric-assistant-0");

    expect(stored).toStartWith("Outcome: completed");
    for (const command of commands.slice(0, 3)) {
      expect(stored).toContain(command);
    }
    for (const path of [
      "/workspace/service/Cargo.toml",
      "./src/file with spaces.ts",
    ]) {
      expect(stored).toContain(path);
    }
    expect(labeledSummarySection(stored, "VCS")).toContain(
      "working tree clean",
    );
    expect(stored).not.toContain("omitted-record-count=");
  });

  test("distinguishes zero failures from checks that did not run", async () => {
    const successful = await runDeterministicRenderer(session("source"), {
      messages: completedCommandTurn("bun test", "10 passed, 0 failed"),
    });
    const notRun = await runCompaction(session("source"), {
      messages: completedCommandTurn(
        "bun test",
        "No tests ran; 0 tests failed because tests were not run",
      ),
    });
    const successfulSummary = storedSummary(successful, "metric-assistant-0");
    const notRunSummary = storedSummary(notRun, "metric-assistant-0");

    expect(successfulSummary).toStartWith("Outcome: completed");
    expect(successfulSummary).toContain("10 passed, 0 failed");
    expect(successfulSummary).not.toContain("Unresolved:");
    expect(notRunSummary).toStartWith("Outcome: partial");
    expect(notRunSummary).toContain("No tests ran");
    expect(labeledSummarySection(notRunSummary, "Unresolved")).toContain(
      "tests were not run",
    );
  });

  test.each(["No tests were skipped.", "Tests are not failing."])(
    "keeps resolved tool verification negation model-eligible and preserves it as completed evidence: %s",
    async text => {
      const messages = completedCommandTurn("bun test", text);
      const routed = await runCompaction(session("source"), { messages });

      expect(routed.prompts).toHaveLength(1);
      expect(promptText(routed.prompts[0]!)).toContain(text);

      const turn = await planFor(completedCommandTurn("bun test", text), 0);
      const rendered = renderDeterministicSummary(turn.summarizedTurns[0]!);
      expect(rendered).toStartWith("Outcome: completed");
      expect(rendered).toContain(text);
      expect(rendered).not.toContain("Unresolved:");
      expect(rendered).not.toContain('"polarity"');
    },
  );

  test.each(["No tests were skipped.", "Tests are not failing."])(
    "keeps resolved assistant verification negation model-eligible and preserves it as completed evidence: %s",
    async text => {
      const messages = completedTurn("resolved-assistant", "Verify the change");
      messages[1]!.parts.push(
        textPart(
          "resolved-assistant-evidence",
          "source",
          messages[1]!.info.id,
          text,
        ),
      );
      const routed = await runCompaction(session("source"), { messages });

      expect(routed.prompts).toHaveLength(1);
      expect(promptText(routed.prompts[0]!)).toContain(text);

      const plan = await planFor(
        (() => {
          const copy = completedTurn(
            "resolved-assistant-render",
            "Verify the change",
          );
          copy[1]!.parts.push(
            textPart(
              "resolved-assistant-render-evidence",
              "source",
              copy[1]!.info.id,
              text,
            ),
          );
          return copy;
        })(),
        0,
      );
      const rendered = renderDeterministicSummary(plan.summarizedTurns[0]!);
      expect(rendered).toStartWith("Outcome: completed");
      expect(rendered).toContain(text);
      expect(rendered).not.toContain("Unresolved:");
      expect(rendered).not.toContain('"polarity"');
    },
  );

  test.each([
    "No unresolved issues",
    "Nothing remains to do",
    "no pending work",
    "implementation is not incomplete",
    "No files were modified; analysis only",
    "configuration remains unchanged",
    "parser passed value",
    "build",
    "check",
  ])(
    "does not infer execution evidence from assistant prose: %s",
    async text => {
      const turn = metricTurn(1, 20, 0);
      const finalAssistant = turn.assistants.at(-1)!;
      finalAssistant.parts.push(
        textPart("prose", "source", finalAssistant.info.id, text),
      );

      const requests = await runDeterministicRenderer(session("source"), {
        messages: [...turn.user, ...turn.assistants],
      });
      const stored = storedSummary(requests, "metric-assistant-0");

      expect(stored).toStartWith("Outcome: analysis-only");
      expect(stored).not.toContain("Verification:");
      expect(stored).not.toContain("VCS:");
      expect(stored).not.toContain("Unresolved:");
    },
  );

  test.each([
    ["tests passed", "completed", "Verification"],
    ["test not executed", "partial", "Unresolved"],
    ["work remains: run smoke test", "partial", "Next"],
    ["Tests couldn't be run", "partial", "Unresolved"],
    ["Tests could not be performed", "partial", "Unresolved"],
    ["Tests cannot be run", "partial", "Unresolved"],
    ["Tests can't be run", "partial", "Unresolved"],
    ["tests were skipped", "partial", "Unresolved"],
    ["no tests have run", "partial", "Unresolved"],
  ] as const)(
    "preserves contextual assistant execution evidence: %s",
    async (text, outcome, sectionLabel) => {
      const turn = metricTurn(1, 20, 0);
      const finalAssistant = turn.assistants.at(-1)!;
      finalAssistant.parts.push(
        textPart("prose", "source", finalAssistant.info.id, text),
      );

      const requests = await runDeterministicRenderer(session("source"), {
        messages: [...turn.user, ...turn.assistants],
      });
      const stored = storedSummary(requests, "metric-assistant-0");

      expect(stored).toStartWith(`Outcome: ${outcome}`);
      expect(capsuleSection(stored, sectionLabel)).toContain(text);
    },
  );

  test("keeps exact assistant source code units provider-safe in readable capsules", async () => {
    const sourceText =
      "exact\u0000nul\u000bvt\ud800lone and trailing\udfff units";
    const turn = metricTurn(1, 20, 0);
    const finalAssistant = turn.assistants.at(-1)!;
    finalAssistant.parts.push(
      textPart("units", "source", finalAssistant.info.id, sourceText),
    );

    const requests = await runDeterministicRenderer(session("source"), {
      messages: [...turn.user, ...turn.assistants],
    });
    const stored = storedSummary(requests, "metric-assistant-0");

    expect(stored).toStartWith("Outcome: analysis-only");
    expect(stored).toContain("\\u0000");
    expect(stored).toContain("\\ud800");
    expect(stored).toContain("\\udfff");
    expect(stored).not.toContain("\u0000");
    expect(hasUnpairedSurrogate(stored)).toBeFalse();
    expect(encode(stored).length).toBeLessThanOrEqual(
      DETERMINISTIC_SUMMARY_MAX_TOKENS,
    );
  });

  test("keeps substantive bracket arrays and TODO prose while dropping whole-body placeholders", async () => {
    const turn = metricTurn(1, 20, 0);
    const finalAssistant = turn.assistants.at(-1)!;
    finalAssistant.parts.push(
      textPart(
        "array-status",
        "source",
        finalAssistant.info.id,
        "['failed','pending']",
      ),
      textPart(
        "todo-prose",
        "source",
        finalAssistant.info.id,
        "Implemented parser. TODO: add evidence.",
      ),
      textPart("whole-todo", "source", finalAssistant.info.id, "[TODO]"),
      textPart("whole-tbd", "source", finalAssistant.info.id, "TBD"),
      textPart("whole-na", "source", finalAssistant.info.id, "N/A"),
    );

    const requests = await runDeterministicRenderer(session("source"), {
      messages: [...turn.user, ...turn.assistants],
    });
    const stored = storedSummary(requests, "metric-assistant-0");
    const next = capsuleSection(stored, "Next");

    expect(stored).toStartWith("Outcome: partial");
    expect(stored).toContain("['failed','pending']");
    expect(stored).toContain("Implemented parser. TODO: add evidence.");
    expect(stored).not.toContain('"[TODO]"');
    expect(next).toContain("TODO: add evidence");
  });

  test.each([
    "Tests couldn't be run",
    "Tests could not be performed",
    "tests were skipped",
    "no tests have run",
  ])(
    "classifies contracted tool not-run output as unresolved: %s",
    async output => {
      const requests = await runCompaction(session("source"), {
        messages: completedCommandTurn("bun test", output),
      });
      const stored = storedSummary(requests, "metric-assistant-0");

      expect(stored).toStartWith("Outcome: partial");
      expect(capsuleSection(stored, "Unresolved")).toContain(output);
    },
  );

  test("keeps tool and assistant prose provider-safe in readable capsules", async () => {
    const units = "exact\u0000nul\u000bvt\ud800lone\ud801more";
    const turn = metricTurn(1, 20, 0);
    const assistant = turn.assistants[0]!;
    assistant.parts.push(
      genericToolPart("units-io", assistant.info.id, "read", {
        status: "error",
        input: { filePath: units },
        error: units,
        title: "Read file",
        metadata: {},
        time: { start: 1, end: 2 },
      }),
    );

    const requests = await runDeterministicRenderer(session("source"), {
      messages: [...turn.user, ...turn.assistants],
    });
    const stored = storedSummary(requests, "metric-assistant-0");

    expect(stored).toStartWith("Outcome: partial");
    expect(stored).toContain("\\u0000");
    expect(stored).toContain("\\ud800");
    expect(stored).toContain("\\ud801");
    expect(stored).not.toContain("\u0000");
    expect(hasUnpairedSurrogate(stored)).toBeFalse();
  });

  test("escapes XML-invalid U+FFFE/U+FFFF in assistant prose for provider-safe capsules", async () => {
    const sourceText = "marker\ufffe\ufffftail";
    const turn = metricTurn(1, 20, 0);
    const finalAssistant = turn.assistants.at(-1)!;
    finalAssistant.parts.push(
      textPart("xml-invalid", "source", finalAssistant.info.id, sourceText),
    );

    const requests = await runDeterministicRenderer(session("source"), {
      messages: [...turn.user, ...turn.assistants],
    });
    const stored = storedSummary(requests, "metric-assistant-0");

    expect(stored).toStartWith("Outcome: analysis-only");
    expect(stored).toContain("\\ufffe");
    expect(stored).toContain("\\uffff");
    expect(stored).not.toContain("\ufffe");
    expect(stored).not.toContain("\uffff");
    expect(hasUnpairedSurrogate(stored)).toBeFalse();
  });

  test("escapes XML-invalid U+FFFE/U+FFFF in tool evidence lines", async () => {
    const units = "io\ufffe\uffffunits";
    const turn = metricTurn(1, 20, 0);
    const assistant = turn.assistants[0]!;
    assistant.parts.push(
      genericToolPart("xml-invalid-io", assistant.info.id, "read", {
        status: "error",
        input: { filePath: units },
        error: units,
        title: "Read file",
        metadata: {},
        time: { start: 1, end: 2 },
      }),
    );

    const requests = await runDeterministicRenderer(session("source"), {
      messages: [...turn.user, ...turn.assistants],
    });
    const stored = storedSummary(requests, "metric-assistant-0");

    expect(stored).toStartWith("Outcome: partial");
    expect(stored).toContain("\\ufffe");
    expect(stored).toContain("\\uffff");
    expect(stored).not.toContain("\ufffe");
    expect(stored).not.toContain("\uffff");
    expect(hasUnpairedSurrogate(stored)).toBeFalse();
  });

  test("positive independent pass overrides earlier no-run in assistant prose", async () => {
    const text = "No tests ran. 10 tests passed.";
    const turn = metricTurn(1, 20, 0);
    const finalAssistant = turn.assistants.at(-1)!;
    finalAssistant.parts.push(
      textPart("positive-override", "source", finalAssistant.info.id, text),
    );

    const requests = await runDeterministicRenderer(session("source"), {
      messages: [...turn.user, ...turn.assistants],
    });
    const stored = storedSummary(requests, "metric-assistant-0");
    const verification = capsuleSection(stored, "Verification");

    expect(stored).toStartWith("Outcome: completed");
    expect(verification).toContain("No tests ran");
    expect(verification).toContain("10 tests passed");
    expect(verification).not.toContain('"polarity"');
    expect(stored).not.toContain("Unresolved:");
  });

  test("positive independent pass overrides earlier no-run in real command tool output", async () => {
    const output = "No tests ran. 10 tests passed.";
    const requests = await runDeterministicRenderer(session("source"), {
      messages: completedCommandTurn("bun test", output),
    });
    const stored = storedSummary(requests, "metric-assistant-0");

    expect(stored).toStartWith("Outcome: completed");
    expect(capsuleSection(stored, "Verification")).toContain(output);
    expect(stored).not.toContain("Unresolved:");
  });

  test("pure no-run with zero failures remains not-run", async () => {
    const output = "No tests ran; 0 failed";
    const requests = await runDeterministicRenderer(session("source"), {
      messages: completedCommandTurn("bun test", output),
    });
    const stored = storedSummary(requests, "metric-assistant-0");

    expect(stored).toStartWith("Outcome: partial");
    expect(capsuleSection(stored, "Unresolved")).toContain("No tests ran");
  });

  test.each([
    ["Read unit.test.ts", "10 passed, 0 failed", "verification"],
    ["Read build configuration", "Process exited with code 1", "verification"],
    ["Read deployment config", "not deployed", "vcs"],
  ] as const)(
    "ignores passive read title context: %s",
    async (title, output) => {
      const turn = metricTurn(1, 20, 0);
      const assistant = turn.assistants[0]!;
      assistant.parts.push(
        genericToolPart("passive-read", assistant.info.id, "read", {
          status: "completed",
          input: { filePath: "/workspace/notes.md" },
          output,
          title,
          metadata: {},
          time: { start: 1, end: 2 },
        }),
      );

      const requests = await runDeterministicRenderer(session("source"), {
        messages: [...turn.user, ...turn.assistants],
      });
      const stored = storedSummary(requests, "metric-assistant-0");

      expect(stored).toStartWith("Outcome: completed");
      expect(stored).not.toContain("Verification:");
      expect(stored).not.toContain("VCS:");
      expect(stored).not.toContain("Unresolved:");
      expect(stored).not.toContain(output);
      expect(stored).not.toContain(title);
    },
  );

  test("treats passive read failures, pending prose, and not-deployed text as neutral", async () => {
    const turn = metricTurn(1, 20, 0);
    const assistant = turn.assistants[0]!;
    assistant.parts.push(
      genericToolPart("read-fail", assistant.info.id, "read", {
        status: "completed",
        input: { filePath: "unit.test.ts" },
        output: "failed checks listed in docs",
        title: "Read unit.test.ts",
        metadata: {},
        time: { start: 1, end: 2 },
      }),
      genericToolPart("read-pending", assistant.info.id, "read", {
        status: "completed",
        input: { filePath: "build.config.ts" },
        output: "pending color tokens remain",
        title: "Read build configuration",
        metadata: {},
        time: { start: 1, end: 2 },
      }),
      genericToolPart("read-deploy", assistant.info.id, "task", {
        status: "completed",
        input: { description: "inspect deploy notes" },
        output: "service is not deployed yet",
        title: "Read deployment config",
        metadata: {},
        time: { start: 1, end: 2 },
      }),
    );

    const requests = await runDeterministicRenderer(session("source"), {
      messages: [...turn.user, ...turn.assistants],
    });
    const stored = storedSummary(requests, "metric-assistant-0");

    expect(stored).toStartWith("Outcome: completed");
    expect(stored).not.toContain("Verification:");
    expect(stored).not.toContain("VCS:");
    expect(stored).not.toContain("Unresolved:");
    expect(stored).not.toContain("failed checks listed in docs");
    expect(stored).not.toContain("pending color tokens remain");
    expect(stored).not.toContain("service is not deployed yet");
  });

  test("accepts real bash, git, and test tool identity or command context", async () => {
    const turn = metricTurn(1, 20, 0);
    const assistant = turn.assistants[0]!;
    assistant.parts.push(
      commandToolPart(
        "bash-test",
        assistant.info.id,
        "bun test packages/opencode-plugin",
        "12 passed, 0 failed",
      ),
      commandToolPart(
        "git-status",
        assistant.info.id,
        "git status --short",
        "working tree clean",
      ),
      genericToolPart("named-pytest", assistant.info.id, "pytest", {
        status: "completed",
        input: { args: ["-q"] },
        output: "10 passed",
        title: "Run pytest",
        metadata: {},
        time: { start: 1, end: 2 },
      }),
    );

    const requests = await runDeterministicRenderer(session("source"), {
      messages: [...turn.user, ...turn.assistants],
    });
    const stored = storedSummary(requests, "metric-assistant-0");
    const verification = capsuleSection(stored, "Verification");
    const vcs = capsuleSection(stored, "VCS");

    expect(stored).toStartWith("Outcome: completed");
    expect(verification).toContain("12 passed, 0 failed");
    expect(verification).toContain("10 passed");
    expect(vcs).toContain("working tree clean");
    expect(stored).not.toContain("Unresolved:");
  });

  test.each([
    ["plugin.read-tests", "2 tests failed", "verification"],
    ["read_build", "Process exited with code 1", "verification"],
    ["list-git-status", "not deployed", "vcs"],
    ["search-deploy-notes", "deployment failed", "vcs"],
    ["view_test_results", "10 passed, 2 failed", "verification"],
    ["inspect-build-log", "Process exited with code 1", "verification"],
    ["fetch_git_status", "working tree dirty", "vcs"],
    ["get-tests", "No tests ran", "verification"],
  ] as const)(
    "keeps passive identity %s neutral despite keyword-like name",
    async (toolName, output) => {
      const turn = metricTurn(1, 20, 0);
      const assistant = turn.assistants[0]!;
      assistant.parts.push(
        genericToolPart("passive-identity", assistant.info.id, toolName, {
          status: "completed",
          input: { path: "/workspace/notes.md" },
          output,
          title: toolName,
          metadata: {},
          time: { start: 1, end: 2 },
        }),
      );

      const requests = await runDeterministicRenderer(session("source"), {
        messages: [...turn.user, ...turn.assistants],
      });
      const stored = storedSummary(requests, "metric-assistant-0");

      expect(stored).toStartWith("Outcome: completed");
      expect(stored).not.toContain("Verification:");
      expect(stored).not.toContain("VCS:");
      expect(stored).not.toContain("Unresolved:");
      expect(stored).not.toContain(output);
    },
  );

  test.each([
    ["make test", "2 tests failed"],
    ["just test", "2 tests failed"],
    ["make check", "Process exited with code 1"],
    ["just lint", "1 failed"],
    ["make build", "Process exited with code 1"],
    ["task test", "2 tests failed"],
  ] as const)(
    "treats runner verification command %s failure as required unresolved",
    async (command, output) => {
      const requests = await runCompaction(session("source"), {
        messages: completedCommandTurn(command, output),
      });
      const stored = storedSummary(requests, "metric-assistant-0");
      const unresolved = capsuleSection(stored, "Unresolved");

      expect(stored).toStartWith("Outcome: partial");
      expect(unresolved).toContain(command);
      expect(unresolved).toContain(output);
      expect(unresolved).toContain("[failed]");
      expect(stored).not.toContain('"polarity"');
    },
  );

  test.each([
    ["make test", "10 passed, 0 failed"],
    ["just test", "all tests passed"],
    ["make lint", "0 failed"],
  ] as const)(
    "treats runner verification command %s success as verification evidence",
    async (command, output) => {
      const requests = await runDeterministicRenderer(session("source"), {
        messages: completedCommandTurn(command, output),
      });
      const stored = storedSummary(requests, "metric-assistant-0");
      const verification = capsuleSection(stored, "Verification");

      expect(stored).toStartWith("Outcome: completed");
      expect(verification).toContain(command);
      expect(verification).toContain(output);
      expect(stored).not.toContain("Unresolved:");
    },
  );

  test("does not treat arbitrary prose mentioning make/test as a verification command", async () => {
    const turn = metricTurn(1, 20, 0);
    const assistant = turn.assistants[0]!;
    assistant.parts.push(
      genericToolPart("prose-bash", assistant.info.id, "bash", {
        status: "completed",
        input: {
          command: "echo please make sure the suite is ready for testing",
        },
        output: "please make sure the suite is ready for testing",
        title: "Shell command",
        metadata: {},
        time: { start: 1, end: 2 },
      }),
    );

    const requests = await runDeterministicRenderer(session("source"), {
      messages: [...turn.user, ...turn.assistants],
    });
    const stored = storedSummary(requests, "metric-assistant-0");

    expect(stored).toStartWith("Outcome: completed");
    expect(stored).not.toContain("Verification:");
    expect(stored).not.toContain("Unresolved:");
  });

  test.each([
    "No unresolved issues",
    "The value was passed to serializer",
    "configuration remains unchanged",
    "pending color",
    "fail\ud800ed",
  ])("treats completed arbitrary tool output as neutral: %s", async output => {
    const turn = metricTurn(1, 20, 0);
    const assistant = turn.assistants[0]!;
    assistant.parts.push(
      genericToolPart("neutral-read", assistant.info.id, "read", {
        status: "completed",
        input: { filePath: "/workspace/notes.md" },
        output,
        title: "Read file",
        metadata: {},
        time: { start: 1, end: 2 },
      }),
      genericToolPart("neutral-task", assistant.info.id, "task", {
        status: "completed",
        input: { description: "inspect notes" },
        output,
        title: "Custom task",
        metadata: {},
        time: { start: 1, end: 2 },
      }),
    );

    const requests = await runDeterministicRenderer(session("source"), {
      messages: [...turn.user, ...turn.assistants],
    });
    const stored = storedSummary(requests, "metric-assistant-0");

    expect(stored).toStartWith("Outcome: completed");
    expect(stored).not.toContain("Verification:");
    expect(stored).not.toContain("Unresolved:");
    expect(stored).not.toContain(JSON.stringify(output).slice(1, -1));
  });

  test("does not overflow required budget on hundreds of neutral completed tool records", async () => {
    const turn = metricTurn(1, 20, 0);
    const assistant = turn.assistants[0]!;
    const neutralOutputs = [
      "No unresolved issues",
      "The value was passed to serializer",
      "configuration remains unchanged",
      "pending color",
      "fail\ud800ed",
      "parser passed value",
      "build artifact listing",
      "check configuration sample",
    ];
    assistant.parts.push(
      ...Array.from({ length: 320 }, (_, index) =>
        genericToolPart(`neutral-${index}`, assistant.info.id, "read", {
          status: "completed",
          input: { filePath: `/workspace/neutral-${index}.md` },
          output: `${neutralOutputs[index % neutralOutputs.length]} #${index}`,
          title: "Read file",
          metadata: {},
          time: { start: 1, end: 2 },
        }),
      ),
    );

    const requests = await runDeterministicRenderer(session("source"), {
      messages: [...turn.user, ...turn.assistants],
    });
    const stored = storedSummary(requests, "metric-assistant-0");

    expect(stored).toStartWith("Outcome: completed");
    expect(stored).not.toContain("Verification:");
    expect(stored).not.toContain("VCS:");
    expect(stored).not.toContain("Unresolved:");
    expect(encode(stored).length).toBeLessThanOrEqual(
      DETERMINISTIC_SUMMARY_MAX_TOKENS,
    );
  });

  test.each([
    ["bash", "ls /secret/path", "permission denied"],
    ["shell", "cat /missing/file", "command not found"],
    ["terminal", "sleep 120", "timed out"],
    ["exec", "deploy-staging.sh", "Process exited with code 1"],
    ["plugin.bash", "rm /protected", "cancelled by signal"],
    ["mcp__shell", "long-job", "killed"],
    ["tools/terminal", "run.sh", "aborted"],
  ])(
    "requires unresolved evidence for failed command-running tool %s",
    async (tool, command, failure) => {
      const turn = metricTurn(1, 20, 0);
      const assistant = turn.assistants[0]!;
      assistant.parts.push(
        genericToolPart("cmd-fail", assistant.info.id, tool, {
          status: "completed",
          input: { command },
          output: failure,
          title: "Shell command",
          metadata: {},
          time: { start: 1, end: 2 },
        }),
      );

      const requests = await runCompaction(session("source"), {
        messages: [...turn.user, ...turn.assistants],
      });
      const stored = storedSummary(requests, "metric-assistant-0");
      const unresolved = capsuleSection(stored, "Unresolved");

      expect(stored).toStartWith("Outcome: partial");
      expect(unresolved).toContain(command);
      expect(unresolved).toContain(failure);
      expect(unresolved).toContain("[failed]");
      expect(stored).not.toContain('"polarity"');
    },
  );

  test("keeps passive completed tools neutral when output only looks like failure prose", async () => {
    const turn = metricTurn(1, 20, 0);
    const assistant = turn.assistants[0]!;
    assistant.parts.push(
      genericToolPart("passive-read", assistant.info.id, "read", {
        status: "completed",
        input: { filePath: "/workspace/notes.md" },
        output: "permission denied\nProcess exited with code 1",
        title: "Read file",
        metadata: {},
        time: { start: 1, end: 2 },
      }),
      genericToolPart("passive-task", assistant.info.id, "task", {
        status: "completed",
        input: { description: "inspect notes" },
        output: "command not found; timed out",
        title: "Custom task",
        metadata: {},
        time: { start: 1, end: 2 },
      }),
    );

    const requests = await runDeterministicRenderer(session("source"), {
      messages: [...turn.user, ...turn.assistants],
    });
    const stored = storedSummary(requests, "metric-assistant-0");

    expect(stored).toStartWith("Outcome: completed");
    expect(stored).not.toContain("Unresolved:");
    expect(stored).not.toContain("permission denied");
    expect(stored).not.toContain("command not found");
  });

  test("keeps late generic failed bash authoritative among many neutral completed tool records", async () => {
    const turn = metricTurn(1, 20, 0);
    const assistant = turn.assistants[0]!;
    const command = "ls /workspace/late-secret";
    const failure = "permission denied";
    assistant.parts.push(
      ...Array.from({ length: 1000 }, (_, index) =>
        genericToolPart(`optional-${index}`, assistant.info.id, "read", {
          status: "completed",
          input: { filePath: `/workspace/optional-${index}.md` },
          output: `optional completed body ${index} ${"x".repeat(80)}`,
          title: "Read file",
          metadata: {},
          time: { start: 1, end: 2 },
        }),
      ),
      genericToolPart("late-bash-fail", assistant.info.id, "bash", {
        status: "completed",
        input: { command },
        output: failure,
        title: "Shell command",
        metadata: {},
        time: { start: 1, end: 2 },
      }),
    );

    const requests = await runDeterministicRenderer(session("source"), {
      messages: [...turn.user, ...turn.assistants],
    });
    const stored = storedSummary(requests, "metric-assistant-0");
    const unresolved = capsuleSection(stored, "Unresolved");

    expect(stored).toStartWith("Outcome: partial");
    expect(unresolved).toContain(command);
    expect(unresolved).toContain(failure);
    expect(stored).not.toContain('"kind"');
    expect(encode(stored).length).toBeLessThanOrEqual(
      DETERMINISTIC_SUMMARY_MAX_TOKENS,
    );
  });

  test("preserves real bash bun test polarity and failed tool state", async () => {
    const turn = metricTurn(1, 20, 0);
    const assistant = turn.assistants[0]!;
    assistant.parts.push(
      commandToolPart(
        "bun-pass",
        assistant.info.id,
        "bun test",
        "10 passed, 0 failed",
      ),
      commandToolPart(
        "bun-fail",
        assistant.info.id,
        'bun test "packages/fail suite.test.ts"',
        "Process exited with code 1",
      ),
      commandToolPart(
        "bun-not-run",
        assistant.info.id,
        "bun test",
        "No tests ran; tests were not run",
      ),
      genericToolPart("failed-state", assistant.info.id, "read", {
        status: "error",
        input: { filePath: "/workspace/missing.ts" },
        error: "No unresolved issues",
        metadata: {},
        time: { start: 1, end: 2 },
      }),
    );

    const requests = await runDeterministicRenderer(session("source"), {
      messages: [...turn.user, ...turn.assistants],
    });
    const stored = storedSummary(requests, "metric-assistant-0");
    const verification = capsuleSection(stored, "Verification");
    const unresolved = capsuleSection(stored, "Unresolved");

    expect(stored).toStartWith("Outcome: partial");
    expect(verification).toContain("10 passed, 0 failed");
    expect(verification).not.toContain("Process exited with code 1");
    expect(unresolved).toContain("Process exited with code 1");
    expect(unresolved).toContain("tests were not run");
    expect(unresolved).toContain("[error]");
    expect(unresolved).toContain("/workspace/missing.ts");
    expect(unresolved).toContain("No unresolved issues");
    expect(stored).not.toContain('"status"');
  });

  test("categorizes mutation tools by identity rather than output prose", async () => {
    const turn = metricTurn(1, 20, 0);
    const assistant = turn.assistants[0]!;
    assistant.parts.push(
      genericToolPart("write-tool", assistant.info.id, "write", {
        status: "completed",
        input: { filePath: "src/a.ts", content: "export {}" },
        output: "Wrote src/a.ts",
        title: "Write",
        metadata: {},
        time: { start: 1, end: 2 },
      }),
      genericToolPart("read-wrote", assistant.info.id, "read", {
        status: "completed",
        input: { filePath: "notes.md" },
        output: "Someone wrote docs and edited examples earlier.",
        title: "Read file",
        metadata: {},
        time: { start: 1, end: 2 },
      }),
    );

    const requests = await runDeterministicRenderer(session("source"), {
      messages: [...turn.user, ...turn.assistants],
    });
    const stored = storedSummary(requests, "metric-assistant-0");

    expect(stored).toStartWith("Outcome: completed");
    expect(stored).toContain("write: src/a.ts");
    expect(stored).not.toContain("Someone wrote docs");
  });

  test.each([
    ["run_tests", "12 passed, 0 failed", "verification"],
    ["run-tests", "12 passed, 0 failed", "verification"],
    ["plugin.run-tests", "12 passed, 0 failed", "verification"],
    ["mcp:server:run_tests", "12 passed, 0 failed", "verification"],
    ["namespace/run_tests", "12 passed, 0 failed", "verification"],
    ["git_status", "working tree clean", "vcs"],
    ["git-status", "working tree clean", "vcs"],
    ["plugin.git-status", "working tree clean", "vcs"],
  ] as const)(
    "normalizes tool identity separators for context: %s",
    async (tool, output, kind) => {
      const turn = metricTurn(1, 20, 0);
      const assistant = turn.assistants[0]!;
      assistant.parts.push(
        genericToolPart("named-tool", assistant.info.id, tool, {
          status: "completed",
          input: { args: [] },
          output,
          title: "Custom tool",
          metadata: {},
          time: { start: 1, end: 2 },
        }),
      );

      const requests = await runDeterministicRenderer(session("source"), {
        messages: [...turn.user, ...turn.assistants],
      });
      const stored = storedSummary(requests, "metric-assistant-0");

      expect(stored).toStartWith("Outcome: completed");
      if (kind === "verification") {
        expect(capsuleSection(stored, "Verification")).toContain(output);
        expect(stored).not.toContain("VCS:");
      } else {
        expect(capsuleSection(stored, "VCS")).toContain(output);
        expect(stored).not.toContain("Verification:");
      }
      expect(stored).not.toContain("Unresolved:");
    },
  );

  test("keeps passive read identities neutral after tool-name normalization", async () => {
    const turn = metricTurn(1, 20, 0);
    const assistant = turn.assistants[0]!;
    assistant.parts.push(
      genericToolPart("read-snake", assistant.info.id, "read_file", {
        status: "completed",
        input: { filePath: "/workspace/unit.test.ts" },
        output: "12 passed, 0 failed",
        title: "Read unit.test.ts",
        metadata: {},
        time: { start: 1, end: 2 },
      }),
      genericToolPart("read-ns", assistant.info.id, "plugin.read-file", {
        status: "completed",
        input: { filePath: "/workspace/notes.md" },
        output: "Process exited with code 1",
        title: "Read notes",
        metadata: {},
        time: { start: 1, end: 2 },
      }),
    );

    const requests = await runDeterministicRenderer(session("source"), {
      messages: [...turn.user, ...turn.assistants],
    });
    const stored = storedSummary(requests, "metric-assistant-0");

    expect(stored).toStartWith("Outcome: completed");
    expect(stored).not.toContain("Verification:");
    expect(stored).not.toContain("VCS:");
    expect(stored).not.toContain("Unresolved:");
    expect(stored).not.toContain("12 passed, 0 failed");
    expect(stored).not.toContain("Process exited with code 1");
  });

  test.each([
    ["Tests ran. They failed.", "failed"],
    ["Tests could not run. Database unavailable.", "not-run"],
  ] as const)(
    "classifies full assistant text before segmentation: %s",
    async text => {
      const turn = metricTurn(1, 20, 0);
      const finalAssistant = turn.assistants.at(-1)!;
      finalAssistant.parts.push(
        textPart("cross-sentence", "source", finalAssistant.info.id, text),
      );

      const requests = await runCompaction(session("source"), {
        messages: [...turn.user, ...turn.assistants],
      });
      const stored = storedSummary(requests, "metric-assistant-0");
      const unresolved = capsuleSection(stored, "Unresolved");

      expect(stored).toStartWith("Outcome: partial");
      expect(unresolved).toContain(text);
      expect(stored).not.toContain('"polarity"');
    },
  );

  test("preserves positive-run precedence across segmented assistant sentences", async () => {
    const text = "No tests ran. 10 tests passed.";
    const turn = metricTurn(1, 20, 0);
    const finalAssistant = turn.assistants.at(-1)!;
    finalAssistant.parts.push(
      textPart("positive-full", "source", finalAssistant.info.id, text),
    );

    const requests = await runDeterministicRenderer(session("source"), {
      messages: [...turn.user, ...turn.assistants],
    });
    const stored = storedSummary(requests, "metric-assistant-0");

    expect(stored).toStartWith("Outcome: completed");
    expect(stored).not.toContain("Unresolved:");
    expect(capsuleSection(stored, "Verification")).toContain(text);
  });

  test.each([
    "no tasks are pending",
    "no work is pending",
    "no issues remain pending",
    "no tests are pending",
    "no checks remain pending",
  ])(
    "treats resolved pending negation as no incomplete evidence: %s",
    async text => {
      const turn = metricTurn(1, 20, 0);
      const finalAssistant = turn.assistants.at(-1)!;
      finalAssistant.parts.push(
        textPart("resolved-pending", "source", finalAssistant.info.id, text),
      );

      const requests = await runDeterministicRenderer(session("source"), {
        messages: [...turn.user, ...turn.assistants],
      });
      const stored = storedSummary(requests, "metric-assistant-0");

      expect(stored).toStartWith("Outcome: analysis-only");
      expect(stored).not.toContain("Unresolved:");
      expect(stored).not.toContain("Next:");
    },
  );

  test.each(["tasks are pending", "work remains: run smoke"])(
    "preserves real incomplete assistant evidence: %s",
    async text => {
      const turn = metricTurn(1, 20, 0);
      const finalAssistant = turn.assistants.at(-1)!;
      finalAssistant.parts.push(
        textPart("real-pending", "source", finalAssistant.info.id, text),
      );

      const requests = await runDeterministicRenderer(session("source"), {
        messages: [...turn.user, ...turn.assistants],
      });
      const stored = storedSummary(requests, "metric-assistant-0");

      expect(stored).toStartWith("Outcome: partial");
      expect(capsuleSection(stored, "Next")).toContain(text);
    },
  );

  test("keeps exact command and failure text readable in unresolved capsules", async () => {
    const command = 'bun test "/workspace/service/exact suite.test.ts"';
    const requests = await runCompaction(session("source"), {
      messages: completedCommandTurn(command, "Process exited with code 1"),
    });
    const stored = storedSummary(requests, "metric-assistant-0");
    const unresolved = capsuleSection(stored, "Unresolved");

    expect(stored).toStartWith("Outcome: partial");
    expect(unresolved).toContain(command);
    expect(unresolved).toContain("Process exited with code 1");
    expect(stored).not.toContain('"command"');
    expect(stored).not.toContain('"kind"');
  });

  test("keeps verification commands readable without promoting title labels", async () => {
    const turn = metricTurn(1, 20, 0);
    const assistant = turn.assistants[0]!;
    assistant.parts.push(
      genericToolPart("title-path", assistant.info.id, "bash", {
        status: "completed",
        input: { command: "bun test" },
        output: "10 passed, 0 failed",
        title: "Run /workspace/title-only/path.test.ts",
        metadata: {},
        time: { start: 1, end: 2 },
      }),
    );

    const requests = await runDeterministicRenderer(session("source"), {
      messages: [...turn.user, ...turn.assistants],
    });
    const stored = storedSummary(requests, "metric-assistant-0");

    expect(stored).toStartWith("Outcome: completed");
    expect(capsuleSection(stored, "Verification")).toContain("bun test");
    expect(capsuleSection(stored, "Verification")).toContain(
      "10 passed, 0 failed",
    );
    expect(stored).not.toContain("/workspace/title-only/path.test.ts");
  });

  test("high-risk path still classifies normalized identities and resolved pending at cutover", async () => {
    const turn = metricTurn(1, 20, 0);
    const assistant = turn.assistants[0]!;
    assistant.parts.push(
      genericToolPart("threshold-run-tests", assistant.info.id, "run_tests", {
        status: "completed",
        input: { args: [] },
        output: "10 passed, 0 failed",
        title: "Run tests",
        metadata: {},
        time: { start: 1, end: 2 },
      }),
      genericToolPart("threshold-git-status", assistant.info.id, "git_status", {
        status: "completed",
        input: { args: [] },
        output: "working tree clean",
        title: "Git status",
        metadata: {},
        time: { start: 1, end: 2 },
      }),
    );
    turn.assistants
      .at(-1)!
      .parts.push(
        textPart(
          "threshold-prose",
          "source",
          turn.assistants.at(-1)!.info.id,
          "no tasks are pending. Tests ran. They failed.",
        ),
      );

    const requests = await runCompaction(session("source"), {
      messages: [...turn.user, ...turn.assistants],
    });
    const stored = storedSummary(requests, "metric-assistant-0");

    expect(requests.prompts).toHaveLength(0);
    expect(stored).toStartWith("Outcome: partial");
    expect(capsuleSection(stored, "Verification")).toContain(
      "10 passed, 0 failed",
    );
    expect(capsuleSection(stored, "VCS")).toContain("working tree clean");
    expect(capsuleSection(stored, "Unresolved")).toContain(
      "Tests ran. They failed.",
    );
    expect(stored).not.toContain('"polarity"');
  });

  test("uses an empty transcript-free scratch session without checkpoint inheritance", async () => {
    const source = nativeCheckpointFixture({ tailStartID: "tail-assistant" });

    const requests = await runCompaction(session("source"), {
      messages: source,
    });

    expect(requests.creates).toHaveLength(1);
    expect(requests.creates[0]).not.toHaveProperty("parentID");
    expect(requests.forks).toEqual([]);
    expect(requests.prompts).toHaveLength(1);
    expect(requests.partUpdates).toHaveLength(2);
  });

  test("cleans the scratch session when the prompt transport fails", async () => {
    const lifecycle: string[] = [];

    await expect(
      runCompaction(session("source"), {
        lifecycle,
        promptResponseError: "prompt SDK response failed",
      }),
    ).rejects.toThrow("prompt SDK response failed");

    expect(lifecycle.filter(event => event === "create")).toHaveLength(1);
    expect(lifecycle.filter(event => event === "prompt")).toHaveLength(1);
    expect(lifecycle.filter(event => event === "delete")).toHaveLength(1);
    expect(lifecycle).not.toContain("fork");
  });
});

describe("native OpenCode checkpoint coexistence", () => {
  test("accepts a valid checkpoint without a tail anchor", async () => {
    const plan = await planFor(
      nativeCheckpointFixture({ tailStartID: null }),
      0,
    );

    expect(assistantIDs(plan.summarizedTurns)).toEqual([
      "suffix-1-assistant",
      "suffix-2-assistant",
    ]);
  });

  test("uses the latest completed checkpoint and ignores an older incomplete marker in its frozen prefix", async () => {
    const messages = nativeCheckpointFixture({ tailStartID: null });
    messages.splice(
      2,
      0,
      nativeMarker("older-incomplete", null),
      ...completedTurn("between", "Between checkpoints"),
    );
    const plan = await planFor(messages, 0);

    expect(assistantIDs(plan.summarizedTurns)).toEqual([
      "suffix-1-assistant",
      "suffix-2-assistant",
    ]);
  });

  test.each([
    [
      "an incomplete marker inside a completed interval",
      [
        nativeMarker("marker-a", null),
        nativeMarker("marker-b", null),
        nativeSummary("summary-a", "marker-a"),
      ],
    ],
    [
      "crossed completed intervals",
      [
        nativeMarker("marker-a", null),
        nativeMarker("marker-b", null),
        nativeSummary("summary-a", "marker-a"),
        nativeSummary("summary-b", "marker-b"),
      ],
    ],
    [
      "nested completed intervals",
      [
        nativeMarker("marker-a", null),
        nativeMarker("marker-b", null),
        nativeSummary("summary-b", "marker-b"),
        nativeSummary("summary-a", "marker-a"),
      ],
    ],
  ])("rejects %s", async (_name, artifacts) => {
    await expect(
      planFor([...artifacts, ...completedTurn("after", "After")], 0),
    ).rejects.toThrow("overlap");
  });

  test("accepts sequential completed checkpoint intervals", async () => {
    const plan = await planFor(
      [
        nativeMarker("marker-a", null),
        nativeSummary("summary-a", "marker-a"),
        ...completedTurn("between", "Between checkpoints"),
        nativeMarker("marker-b", null),
        nativeSummary("summary-b", "marker-b"),
        ...completedTurn("after", "After checkpoints"),
      ],
      0,
    );

    expect(assistantIDs(plan.summarizedTurns)).toEqual(["after-assistant"]);
  });

  test.each([
    [
      "completed then errored",
      nativeSummary("summary-completed", "marker"),
      nativeSummary("summary-extra", "marker", {
        error: { name: "AbortError" },
      }),
    ],
    [
      "errored then completed",
      nativeSummary("summary-extra", "marker", {
        error: { name: "AbortError" },
      }),
      nativeSummary("summary-completed", "marker"),
    ],
    [
      "completed then unfinished",
      nativeSummary("summary-completed", "marker"),
      nativeSummary("summary-extra", "marker", { finish: null }),
    ],
    [
      "unfinished then completed",
      nativeSummary("summary-extra", "marker", { finish: null }),
      nativeSummary("summary-completed", "marker"),
    ],
  ])(
    "rejects an ambiguous marker with %s summary children",
    async (_name, first, second) => {
      await expect(
        planFor(
          [
            nativeMarker("marker", null),
            first,
            second,
            ...completedTurn("after", "After checkpoint"),
          ],
          0,
        ),
      ).rejects.toThrow("ambiguous");
    },
  );

  test("allows an older ambiguous marker only when every extra child is inside a newer frozen prefix", async () => {
    const plan = await planFor(
      [
        nativeMarker("older-marker", null),
        nativeSummary("older-completed", "older-marker"),
        nativeSummary("older-unfinished", "older-marker", { finish: null }),
        ...completedTurn("between", "Between checkpoints"),
        nativeMarker("newer-marker", null),
        nativeSummary("newer-summary", "newer-marker"),
        ...completedTurn("after", "After checkpoints"),
      ],
      0,
    );

    expect(assistantIDs(plan.summarizedTurns)).toEqual(["after-assistant"]);
  });

  test("rejects an older incomplete artifact that extends past a later checkpoint marker", async () => {
    await expect(
      planFor(
        [
          nativeMarker("marker-a", null),
          nativeMarker("marker-b", null),
          nativeSummary("summary-b", "marker-b"),
          nativeSummary("summary-a", "marker-a", { finish: null }),
          ...completedTurn("after", "After checkpoints"),
        ],
        0,
      ),
    ).rejects.toThrow("incomplete");
  });

  test("rejects a malformed tail on an older completed checkpoint", async () => {
    const messages = nativeCheckpointFixture({ tailStartID: null });
    messages.splice(
      2,
      0,
      nativeMarker("older-marker", "missing-tail"),
      nativeSummary("older-summary", "older-marker"),
      ...completedTurn("between", "Between checkpoints"),
    );

    await expect(planFor(messages, 0)).rejects.toThrow("tail_start_id");
  });

  test("ignores Magic boundaries before the native checkpoint and honors boundaries after it", async () => {
    const messages = nativeCheckpointFixture({ tailStartID: null });
    messages[0]!.parts.push(magicBoundary("old-boundary", "tail-user"));
    messages.splice(
      messages.length - 2,
      0,
      ...completedTurn("boundary", "Boundary turn", true),
    );
    const plan = await planFor(messages, 0);

    expect(assistantIDs(plan.summarizedTurns)).toEqual([
      "boundary-assistant",
      "suffix-2-assistant",
    ]);
  });

  test("does not fork, prompt, or mutate when every eligible suffix turn is protected", async () => {
    const calls: string[] = [];
    const v2 = {
      session: {
        messages: async () => ({
          data: nativeCheckpointFixture({ tailStartID: null }),
        }),
        fork: async () => {
          calls.push("fork");
          return { data: session("backup") };
        },
        prompt: async () => {
          calls.push("prompt");
          return { data: {} };
        },
      },
      part: {
        update: async () => {
          calls.push("update");
          return { data: {} };
        },
      },
      tui: {
        showToast: async () => {
          calls.push("toast");
          return { data: true };
        },
      },
    } as unknown as V2Client;

    expect(await executeMagicCompact(v2, "source", 2)).toBe(false);
    expect(calls).toEqual(["toast"]);
  });

  test("surfaces an in-band no-op toast SDK error without starting compaction", async () => {
    const calls: string[] = [];
    const v2 = {
      session: {
        messages: async () => ({
          data: nativeCheckpointFixture({ tailStartID: null }),
        }),
        fork: async () => {
          calls.push("fork");
          return { data: session("backup") };
        },
      },
      tui: {
        showToast: async (request: { title: string }) => {
          calls.push(request.title);
          return request.title === "Magic Compact"
            ? { error: "no-op toast failed" }
            : { data: true };
        },
      },
    } as unknown as V2Client;

    await expect(executeMagicCompact(v2, "source", 2)).rejects.toThrow(
      "no-op toast failed",
    );
    expect(calls).toEqual(["Magic Compact", "Magic Compact Failed"]);
  });

  test("revalidates before writes and completes by pruning only the checkpoint suffix", async () => {
    const messages = nativeCheckpointFixture({ tailStartID: "tail-assistant" });
    messages
      .find(item => item.info.id === "suffix-1-assistant")!
      .parts.push(
        textPart(
          "suffix-1-old-text",
          "source",
          "suffix-1-assistant",
          "Replace this response",
        ),
      );
    addProviderTokens(messages);
    const command = createCommandHarness(messages);

    expect(await executeMagicCompact(command.v2, "source", 1)).toBe(true);
    expect(
      command.partUpdates.map(request => ({
        messageID: request.messageID,
        partID: request.partID,
      })),
    ).toEqual([
      {
        messageID: "suffix-1-assistant",
        partID: "prt_-magic_summary_suffix-1-assistant",
      },
      {
        messageID: "suffix-2-user",
        partID: "prt_-magic_boundary_suffix-2-user",
      },
    ]);
    expect(command.partDeletes).toEqual([
      {
        sessionID: "source",
        messageID: "suffix-1-assistant",
        partID: "suffix-1-removable-detail",
      },
      {
        sessionID: "source",
        messageID: "suffix-1-assistant",
        partID: "suffix-1-old-text",
      },
    ]);
    expect(command.messageDeletes).toEqual([]);
    expect(command.sessionDeletes).toEqual(["scratch-1"]);
    expect(command.selectedSessions).toEqual([]);

    const summaryPromptIndex = command.events.indexOf("summary-prompt");
    const summaryUpdateIndex = command.events.indexOf("summary-update");
    const initialRevalidationIndex = command.events.indexOf("messages", 1);
    const postSummaryRevalidationIndex = command.events.indexOf(
      "messages",
      summaryPromptIndex + 1,
    );
    expect(command.events).not.toContain("progress-prompt");
    expect(summaryPromptIndex).toBeGreaterThanOrEqual(0);
    expect(command.events[initialRevalidationIndex]).toBe("messages");
    expect(postSummaryRevalidationIndex).toBeGreaterThan(summaryPromptIndex);
    expect(postSummaryRevalidationIndex).toBeLessThan(summaryUpdateIndex);
    expect(command.toasts.at(-1)).toMatchObject({
      title: "Magic Compact",
      message: "Compacted 1 assistant turn(s).",
    });
  });

  test("surfaces an in-band success toast SDK error after durable compaction without rollback", async () => {
    const messages = nativeCheckpointFixture({ tailStartID: "tail-assistant" });
    addProviderTokens(messages);
    const command = createCommandHarness(messages, {
      successToastResponseError: "success toast failed",
    });

    await expect(executeMagicCompact(command.v2, "source", 1)).rejects.toThrow(
      "success toast failed",
    );

    expect(command.events).toContain("summary-update");
    expect(command.events).toContain("boundary-update");
    expect(command.events).not.toContain("select-backup");
    expect(command.events).not.toContain("delete-source");
    expect(command.events).not.toContain("failure-toast");
    expect(command.selectedSessions).toEqual([]);
    expect(command.sessionDeletes).toEqual(["scratch-1"]);
  });

  test("preserves the partially mutated source and backup when final projection detects late source drift", async () => {
    const messages = nativeCheckpointFixture({ tailStartID: "tail-assistant" });
    addProviderTokens(messages);
    await rm(statsPath("source"), { force: true });
    const command = createCommandHarness(messages, {
      mutateSourceBeforeFinalCount(source) {
        const target = source.find(
          message => message.info.id === "suffix-2-assistant",
        )!;
        target.parts.push(
          textPart(
            "late-concurrent-content",
            "source",
            target.info.id,
            "Concurrent content written after final source revalidation.",
          ),
        );
      },
    });
    let error: unknown;

    try {
      await executeMagicCompact(command.v2, "source", 1);
    } catch (caught) {
      error = caught;
    }

    expect(error).toMatchObject({ name: "CompactionSourceDriftError" });
    expect(String(error)).toContain("projection mismatch");
    expect(String(error)).toContain("backup");
    expect(String(error)).toContain("preserved");
    expect(command.events).not.toContain("select-backup");
    expect(command.events).not.toContain("delete-source");
    expect(command.selectedSessions).toEqual([]);
    expect(command.sessionDeletes).toEqual(["scratch-1"]);
    expect(
      command.sessionUpdates.filter(
        request =>
          request.sessionID === "backup" && request.title === "Test session",
      ),
    ).toEqual([]);
    expect(
      messages.some(message => message.info.id === "source-notice-1"),
    ).toBe(true);
    expect(
      messages.some(message =>
        message.parts.some(part => part.id === "late-concurrent-content"),
      ),
    ).toBe(true);
    expect(await readStats("source")).toBeNull();
    expect(command.toasts.at(-1)).toMatchObject({
      title: "Magic Compact Failed",
      message: expect.stringContaining("backup"),
      variant: "error",
    });
  });

  test("keeps ordinary post-backup failure rollback behavior", async () => {
    const messages = nativeCheckpointFixture({ tailStartID: "tail-assistant" });
    addProviderTokens(messages);
    await rm(statsPath("source"), { force: true });
    const primary = new Error("stats notice failed");
    const command = createCommandHarness(messages, {
      sourcePromptError: primary,
    });
    let error: unknown;

    try {
      await executeMagicCompact(command.v2, "source", 1);
    } catch (caught) {
      error = caught;
    }

    expect(error).toBe(primary);
    expect(error).not.toMatchObject({ name: "CompactionSourceDriftError" });
    expect(command.selectedSessions).toEqual(["backup"]);
    expect(command.sessionDeletes).toEqual(["scratch-1", "source"]);
    expect(
      command.sessionUpdates.filter(
        request =>
          request.sessionID === "backup" && request.title === "Test session",
      ),
    ).toEqual([{ sessionID: "backup", title: "Test session" }]);
    expect(command.events.indexOf("select-backup")).toBeLessThan(
      command.events.indexOf("delete-source"),
    );
    expect(await readStats("source")).toBeNull();
  });

  test("rolls back without partial compaction writes when a packed batch exhausts its retry", async () => {
    const messages = Array.from({ length: 9 }, (_, index) =>
      completedTurn(`exhaust-${index}`, `Request ${index}`),
    ).flat();
    addProviderTokens(messages);
    const allIDs = Array.from(
      { length: 9 },
      (_, index) => `exhaust-${index}-assistant`,
    );
    const command = createCommandHarness(messages, {
      summaryTextOutputs: [
        JSON.stringify(Object.fromEntries(allIDs.map(id => [id, "[TODO]"]))),
        JSON.stringify(Object.fromEntries(allIDs.map(id => [id, "[TODO]"]))),
      ],
    });
    let error: unknown;

    try {
      await executeMagicCompact(command.v2, "source", 0);
    } catch (caught) {
      error = caught;
    }

    expect(error).toMatchObject({
      name: "SummaryContractError",
      category: "placeholder-summary",
      batch: 1,
      batchCount: 1,
      attempt: 2,
    });
    expect(
      command.events.filter(event => event === "summary-prompt"),
    ).toHaveLength(2);
    expect(command.partUpdates).toEqual([]);
    expect(command.partDeletes).toEqual([]);
    expect(
      command.sessionUpdates.filter(request => request.sessionID === "source"),
    ).toEqual([]);
    expect(command.sourcePrompts).toEqual([]);
    expect(command.messageDeletes).toEqual([]);
    expect(command.sessionDeletes).toEqual(["scratch-1", "scratch-2"]);
    expect(command.selectedSessions).toEqual([]);
    expect(command.events).not.toContain("summary-update");
    expect(command.events).not.toContain("boundary-update");
    expect(String(command.toasts.at(-1)?.message)).not.toContain("[TODO]");
  });

  test("a future completed native checkpoint advances the immutable floor", async () => {
    const messages = nativeCheckpointFixture({ tailStartID: null });
    messages.push(
      nativeMarker("future-marker", "suffix-2-assistant"),
      nativeSummary("future-summary", "future-marker"),
      ...completedTurn("future-suffix", "After future checkpoint"),
    );
    const plan = await planFor(messages, 0);

    expect(assistantIDs(plan.summarizedTurns)).toEqual([
      "future-suffix-assistant",
    ]);
  });

  test("detects new and changed native artifacts before pruning", async () => {
    const messages = nativeCheckpointFixture({ tailStartID: null });
    const plan = await planFor(messages, 0);
    const completedRace = [
      ...messages,
      nativeMarker("raced-marker", "suffix-2-assistant"),
      nativeSummary("raced-summary", "raced-marker"),
    ];
    const changedMarkerPart = structuredClone(messages);
    Object.assign(
      changedMarkerPart.find(item => item.info.id === "native-marker")!
        .parts[0]!,
      { auto: false },
    );
    const changedSummaryInfo = structuredClone(messages);
    Object.assign(
      changedSummaryInfo.find(item => item.info.id === "native-summary")!.info,
      { finish: "length" },
    );
    const changedSummaryPart = structuredClone(messages);
    changedSummaryPart.find(item => item.info.id === "native-summary")!.parts =
      [
        textPart(
          "native-summary-text",
          "source",
          "native-summary",
          "Changed native summary",
        ),
      ];

    for (const changed of [
      completedRace,
      changedMarkerPart,
      changedSummaryInfo,
      changedSummaryPart,
    ]) {
      expect(() => assertCompactionSourceUnchanged(changed, plan)).toThrow(
        "changed",
      );
    }
  });

  test("does not report a checkpoint race for object key insertion order", async () => {
    const messages = nativeCheckpointFixture({ tailStartID: null });
    const plan = await planFor(messages, 0);
    const reordered = messages.map(item => ({
      info: Object.fromEntries(Object.entries(item.info).reverse()) as Message,
      parts: item.parts.map(
        part =>
          Object.fromEntries(Object.entries(part).reverse()) as unknown as Part,
      ),
    }));

    expect(() =>
      assertCompactionSourceUnchanged(reordered, plan),
    ).not.toThrow();
  });

  test("does not report source drift for volatile timestamp or accounting changes", async () => {
    const messages = nativeCheckpointFixture({ tailStartID: null });
    const assistant = messages.find(
      item => item.info.id === "suffix-1-assistant",
    )!;
    assistant.parts.push(
      commandToolPart(
        "stable-tool",
        assistant.info.id,
        "bun test",
        "12 tests passed",
      ),
    );
    addProviderTokens(messages);
    Object.assign(messages[0]!.info, { time: { created: 1 } });
    Object.assign(assistant.info, {
      time: { created: 1, completed: 2 },
      cost: 1,
    });
    const plan = await planFor(messages, 0);
    const changed = structuredClone(messages);
    Object.assign(changed[0]!.info, { time: { created: 10 } });
    const changedAssistant = changed.find(
      item => item.info.id === assistant.info.id,
    )!;
    Object.assign(changedAssistant.info, {
      time: { created: 10, completed: 20 },
      cost: 2,
      tokens: {
        input: 100,
        output: 50,
        reasoning: 25,
        cache: { read: 10, write: 5 },
      },
    });
    const changedReasoning = changedAssistant.parts.find(
      part => part.id === "suffix-1-removable-detail",
    ) as Part & { time: { start: number; end: number } };
    changedReasoning.time = { start: 10, end: 20 };
    const changedTool = changedAssistant.parts.find(
      (part): part is ToolPart => part.id === "stable-tool",
    )!;
    if (changedTool.state.status === "completed") {
      changedTool.state.time = { start: 10, end: 20, compacted: 30 };
    }

    expect(() => assertCompactionSourceUnchanged(changed, plan)).not.toThrow();
  });

  test("aborts before source writes and preserves both sessions when a checkpoint races the summary prompt", async () => {
    const messages = nativeCheckpointFixture({ tailStartID: null });
    addProviderTokens(messages);
    const raced = [
      ...messages,
      nativeMarker("raced-marker", "suffix-2-assistant"),
    ];
    const command = createCommandHarness(messages, { racedMessages: raced });

    await expect(
      executeMagicCompact(command.v2, "source", 1),
    ).rejects.toMatchObject({ name: "NativeCheckpointChangedError" });
    expect(command.partUpdates).toEqual([]);
    expect(command.partDeletes).toEqual([]);
    expect(command.messageDeletes).toEqual([]);
    expect(command.sessionDeletes).toEqual(["scratch-1"]);
    expect(command.selectedSessions).toEqual([]);
    expect(command.sourcePrompts).toEqual([]);
    expect(command.backupForks).toHaveLength(1);
    expect(command.toasts.at(-1)).toMatchObject({
      title: "Magic Compact Failed",
      variant: "error",
    });
  });

  test.each(["completed", "error"] as const)(
    "aborts before source writes when a pending tool becomes %s during summary generation",
    async status => {
      const messages = nativeCheckpointFixture({ tailStartID: null });
      const racedAssistant = messages.find(
        item => item.info.id === "suffix-1-assistant",
      )!;
      racedAssistant.parts.push(
        pendingCommandToolPart(
          "raced-pending-tool",
          racedAssistant.info.id,
          "bun test",
        ),
      );
      addProviderTokens(messages);
      const raced = structuredClone(messages);
      const racedTool = raced
        .find(item => item.info.id === racedAssistant.info.id)!
        .parts.find(
          (part): part is ToolPart => part.id === "raced-pending-tool",
        )!;
      racedTool.state =
        status === "completed"
          ? {
              status,
              input: { command: "bun test" },
              output: "12 tests passed",
              title: "Run tests",
              metadata: {},
              time: { start: 1, end: 2 },
            }
          : {
              status,
              input: { command: "bun test" },
              error: "Process exited with code 1",
              metadata: {},
              time: { start: 1, end: 2 },
            };
      const command = createCommandHarness(messages, { racedMessages: raced });

      await expect(
        executeMagicCompact(command.v2, "source", 0),
      ).rejects.toMatchObject({ name: "NativeCheckpointChangedError" });
      expect(command.partUpdates).toEqual([]);
      expect(command.partDeletes).toEqual([]);
      expect(command.messageDeletes).toEqual([]);
      expect(command.sourcePrompts).toEqual([]);
      expect(command.backupForks).toHaveLength(1);
      expect(command.selectedSessions).toEqual([]);
    },
  );

  test.each(["part", "message"] as const)(
    "aborts before source writes when an assistant %s is appended during summary generation",
    async addition => {
      const messages = nativeCheckpointFixture({ tailStartID: null });
      addProviderTokens(messages);
      const raced = structuredClone(messages);
      if (addition === "part") {
        const assistant = raced.find(
          item => item.info.id === "suffix-1-assistant",
        )!;
        assistant.parts.push(
          textPart(
            "raced-assistant-part",
            "source",
            assistant.info.id,
            "Late assistant state that must not be pruned under a stale summary.",
          ),
        );
      } else {
        const nextTurnIndex = raced.findIndex(
          item => item.info.id === "suffix-2-user",
        );
        raced.splice(
          nextTurnIndex,
          0,
          message("raced-assistant-message", "assistant", [
            textPart(
              "raced-assistant-message-text",
              "source",
              "raced-assistant-message",
              "Late assistant message that must remain in the live source.",
            ),
          ]),
        );
      }
      const command = createCommandHarness(messages, { racedMessages: raced });

      await expect(
        executeMagicCompact(command.v2, "source", 1),
      ).rejects.toMatchObject({ name: "NativeCheckpointChangedError" });
      expect(command.partUpdates).toEqual([]);
      expect(command.partDeletes).toEqual([]);
      expect(command.messageDeletes).toEqual([]);
      expect(command.sourcePrompts).toEqual([]);
      expect(command.backupForks).toHaveLength(1);
      expect(command.selectedSessions).toEqual([]);
    },
  );

  test("ignores stale progress-delete fault injection during a checkpoint race", async () => {
    const messages = nativeCheckpointFixture({ tailStartID: null });
    addProviderTokens(messages);
    const raced = [
      ...messages,
      nativeMarker("raced-marker", "suffix-2-assistant"),
    ];
    const cleanupError = new Error("progress cleanup failed");
    const command = createCommandHarness(messages, {
      racedMessages: raced,
      deleteProgressError: cleanupError,
    });

    let error: unknown;
    try {
      await executeMagicCompact(command.v2, "source", 1);
    } catch (caught) {
      error = caught;
    }

    expect(error).toBeInstanceOf(NativeCheckpointChangedError);
    expect(error).not.toBe(cleanupError);
    expect(nestedErrorMessages((error as Error).cause)).toContain(
      "Native compaction marker raced-marker is incomplete or errored.",
    );
    expect(command.partUpdates).toEqual([]);
    expect(command.partDeletes).toEqual([]);
    expect(command.messageDeletes).toEqual([]);
    expect(command.sessionDeletes).toEqual(["scratch-1"]);
    expect(command.selectedSessions).toEqual([]);
    expect(
      command.sessionUpdates.filter(
        request =>
          request.sessionID === "backup" && request.title === "Test session",
      ),
    ).toEqual([]);
    expect(command.toasts.at(-1)).toMatchObject({
      title: "Magic Compact Failed",
      variant: "error",
    });
  });

  test("retains scratch cleanup failure before a checkpoint race can be observed", async () => {
    const messages = nativeCheckpointFixture({ tailStartID: null });
    addProviderTokens(messages);
    const raced = [
      ...messages,
      nativeMarker("raced-marker", "suffix-2-assistant"),
    ];
    const ephemeralDeleteError = new Error("ephemeral delete failed");
    const command = createCommandHarness(messages, {
      racedMessages: raced,
      deleteEphemeralError: ephemeralDeleteError,
    });

    let error: unknown;
    try {
      await executeMagicCompact(command.v2, "source", 1);
    } catch (caught) {
      error = caught;
    }

    expect(error).toBe(ephemeralDeleteError);
    expect(command.partUpdates).toEqual([]);
    expect(command.partDeletes).toEqual([]);
    expect(command.messageDeletes).toEqual([]);
    expect(command.sessionDeletes).toEqual(["scratch-1"]);
    expect(command.selectedSessions).toEqual([]);
    expect(
      command.sessionUpdates.filter(
        request =>
          request.sessionID === "backup" && request.title === "Test session",
      ),
    ).toEqual([]);
    expect(
      command.sessionDeletes.filter(sessionID => sessionID === "source"),
    ).toEqual([]);
    expect(command.toasts.at(-1)).toMatchObject({
      title: "Magic Compact Failed",
      variant: "error",
    });
  });

  test("treats a revalidation messages-fetch SDK error after generation as a checkpoint change", async () => {
    const messages = nativeCheckpointFixture({ tailStartID: null });
    addProviderTokens(messages);
    const fetchError = new Error("revalidation messages fetch failed");
    const command = createCommandHarness(messages, {
      revalidationMessagesError: fetchError,
    });

    let error: unknown;
    try {
      await executeMagicCompact(command.v2, "source", 1);
    } catch (caught) {
      error = caught;
    }

    expect(error).toBeInstanceOf(NativeCheckpointChangedError);
    expect((error as Error).cause).toMatchObject({
      message: "revalidation messages fetch failed",
    });
    expect(command.partUpdates).toEqual([]);
    expect(command.partDeletes).toEqual([]);
    expect(command.messageDeletes).toEqual([]);
    expect(command.sessionDeletes).toEqual(["scratch-1"]);
    expect(command.selectedSessions).toEqual([]);
    expect(
      command.sessionUpdates.filter(
        request =>
          request.sessionID === "backup" && request.title === "Test session",
      ),
    ).toEqual([]);
  });

  test("does not revalidate after a generation failure", async () => {
    const messages = nativeCheckpointFixture({ tailStartID: null });
    addProviderTokens(messages);
    const generationError = new Error("summary prompt failed");
    const fetchError = new Error("revalidation messages fetch failed");
    const command = createCommandHarness(messages, {
      summaryPromptError: generationError,
      revalidationMessagesError: fetchError,
    });

    let error: unknown;
    try {
      await executeMagicCompact(command.v2, "source", 1);
    } catch (caught) {
      error = caught;
    }

    expect(error).toBe(generationError);
    expect(command.partUpdates).toEqual([]);
    expect(command.partDeletes).toEqual([]);
    expect(command.messageDeletes).toEqual([]);
    expect(command.sessionDeletes).toEqual(["scratch-1"]);
    expect(command.selectedSessions).toEqual([]);
    expect(
      command.sessionUpdates.filter(
        request =>
          request.sessionID === "backup" && request.title === "Test session",
      ),
    ).toEqual([]);
  });

  test("generation failure before backup leaves the source untouched", async () => {
    const messages = nativeCheckpointFixture({ tailStartID: null });
    addProviderTokens(messages);
    const generationError = new Error("summary prompt failed");
    const cleanupError = new Error("progress cleanup failed");
    const command = createCommandHarness(messages, {
      summaryPromptError: generationError,
      deleteProgressError: cleanupError,
    });

    let error: unknown;
    try {
      await executeMagicCompact(command.v2, "source", 1);
    } catch (caught) {
      error = caught;
    }

    expect(error).toBe(generationError);
    expect(error).not.toBeInstanceOf(NativeCheckpointChangedError);
    expect(command.partUpdates).toEqual([]);
    expect(command.partDeletes).toEqual([]);
    expect(command.messageDeletes).toEqual([]);
    expect(command.sessionDeletes).toEqual(["scratch-1"]);
    expect(command.selectedSessions).toEqual([]);
    expect(command.backupForks).toEqual([]);
  });

  test("retains scratch cleanup failure with a generation failure before backup", async () => {
    const messages = nativeCheckpointFixture({ tailStartID: null });
    addProviderTokens(messages);
    const generationError = new Error("summary prompt failed");
    const ephemeralDeleteError = new Error("ephemeral delete failed");
    const command = createCommandHarness(messages, {
      summaryPromptError: generationError,
      deleteEphemeralError: ephemeralDeleteError,
    });

    let error: unknown;
    try {
      await executeMagicCompact(command.v2, "source", 1);
    } catch (caught) {
      error = caught;
    }

    expect(error).toBeInstanceOf(AggregateError);
    expect(error).not.toBeInstanceOf(NativeCheckpointChangedError);
    expect(command.partUpdates).toEqual([]);
    expect(command.partDeletes).toEqual([]);
    expect(nestedErrorMessages(error)).toEqual(
      expect.arrayContaining([
        "summary prompt failed",
        "ephemeral delete failed",
      ]),
    );
    expect(command.messageDeletes).toEqual([]);
    expect(command.sessionDeletes).toEqual(["scratch-1"]);
    expect(command.selectedSessions).toEqual([]);
    expect(command.backupForks).toEqual([]);
  });

  test("generation failure wins before a later source race can be observed", async () => {
    const messages = nativeCheckpointFixture({ tailStartID: null });
    addProviderTokens(messages);
    const raced = [
      ...messages,
      nativeMarker("raced-marker", "suffix-2-assistant"),
    ];
    const generationError = new Error("summary prompt failed");
    const ephemeralDeleteError = new Error("ephemeral delete failed");
    const command = createCommandHarness(messages, {
      racedMessages: raced,
      summaryPromptError: generationError,
      deleteEphemeralError: ephemeralDeleteError,
    });

    let error: unknown;
    try {
      await executeMagicCompact(command.v2, "source", 1);
    } catch (caught) {
      error = caught;
    }

    expect(error).toBeInstanceOf(AggregateError);
    expect(error).not.toBe(ephemeralDeleteError);
    expect(nestedErrorMessages(error)).toEqual(
      expect.arrayContaining([
        "summary prompt failed",
        "ephemeral delete failed",
      ]),
    );
    expect(command.partUpdates).toEqual([]);
    expect(command.partDeletes).toEqual([]);
    expect(command.messageDeletes).toEqual([]);
    expect(command.sessionDeletes).toEqual(["scratch-1"]);
    expect(command.selectedSessions).toEqual([]);
    expect(
      command.sessionUpdates.filter(
        request =>
          request.sessionID === "backup" && request.title === "Test session",
      ),
    ).toEqual([]);
    expect(
      command.sessionDeletes.filter(sessionID => sessionID === "source"),
    ).toEqual([]);
    expect(command.toasts.at(-1)).toMatchObject({
      title: "Magic Compact Failed",
      variant: "error",
    });
  });

  test("stale progress cleanup injection cannot affect a successful compaction", async () => {
    const messages = nativeCheckpointFixture({ tailStartID: null });
    addProviderTokens(messages);
    const cleanupError = new Error("progress cleanup failed");
    const command = createCommandHarness(messages, {
      deleteProgressError: cleanupError,
    });

    expect(await executeMagicCompact(command.v2, "source", 1)).toBe(true);
    expect(
      command.partUpdates.map(request => ({
        messageID: request.messageID,
        partID: request.partID,
      })),
    ).toEqual([
      {
        messageID: "suffix-1-assistant",
        partID: "prt_-magic_summary_suffix-1-assistant",
      },
      {
        messageID: "suffix-2-user",
        partID: "prt_-magic_boundary_suffix-2-user",
      },
    ]);
    expect(command.messageDeletes).toEqual([]);
    expect(command.selectedSessions).toEqual([]);
  });

  test("does not attempt backup selection for a pre-backup generation failure", async () => {
    const messages = nativeCheckpointFixture({ tailStartID: null });
    addProviderTokens(messages);
    const primary = new Error("summary generation failed");
    const selection = new Error("backup selection failed");
    const command = createCommandHarness(messages, {
      summaryPromptError: primary,
      selectBackupError: selection,
    });

    let error: unknown;
    try {
      await executeMagicCompact(command.v2, "source", 1);
    } catch (caught) {
      error = caught;
    }

    expect(error).toBe(primary);
    expect(command.selectedSessions).toEqual([]);
    expect(command.sessionDeletes).toEqual(["scratch-1"]);
    expect(command.backupForks).toEqual([]);
    expect(command.events).not.toContain("delete-source");
    expect(command.events.indexOf("select-backup")).toBeLessThan(
      command.events.indexOf("failure-toast"),
    );
  });

  test("scratch ignores stale progress cleanup fault injection", async () => {
    const messages = nativeCheckpointFixture({ tailStartID: null });
    addProviderTokens(messages);
    const cleanupError = new Error("progress cleanup failed");
    const command = createCommandHarness(messages, {
      deleteProgressError: cleanupError,
    });
    let error: unknown;

    try {
      await executeMagicCompact(command.v2, "source", 1);
    } catch (caught) {
      error = caught;
    }

    expect(error).toBeUndefined();
    expect(error).not.toBeInstanceOf(NativeCheckpointChangedError);
    expect(command.sessionDeletes).toEqual(["scratch-1"]);
    expect(command.selectedSessions).toEqual([]);
  });

  test("does not attempt source deletion for a pre-backup generation failure", async () => {
    const messages = nativeCheckpointFixture({ tailStartID: null });
    addProviderTokens(messages);
    const primary = new Error("summary generation failed");
    const deletion = new Error("source deletion failed");
    const command = createCommandHarness(messages, {
      summaryPromptError: primary,
      deleteSourceError: deletion,
    });

    let error: unknown;
    try {
      await executeMagicCompact(command.v2, "source", 1);
    } catch (caught) {
      error = caught;
    }

    expect(error).toBe(primary);
    expect(command.selectedSessions).toEqual([]);
    expect(command.sessionDeletes).toEqual(["scratch-1"]);
  });

  test.each([
    [
      "throws",
      { showToastError: new Error("failure toast failed") },
      "failure toast failed",
    ],
    [
      "returns an SDK error",
      { failureToastResponseError: "failure toast SDK error" },
      "failure toast SDK error",
    ],
  ])(
    "retains primary and rollback state when the failure toast %s",
    async (_behavior, notificationOptions, expectedNotification) => {
      const messages = nativeCheckpointFixture({ tailStartID: null });
      addProviderTokens(messages);
      const command = createCommandHarness(messages, {
        summaryPromptError: new Error("summary generation failed"),
        ...notificationOptions,
      });
      let error: unknown;

      try {
        await executeMagicCompact(command.v2, "source", 1);
      } catch (caught) {
        error = caught;
      }

      expect(error).toBeInstanceOf(AggregateError);
      expect(nestedErrorMessages(error)).toEqual(
        expect.arrayContaining([
          "summary generation failed",
          expectedNotification,
        ]),
      );
      expect(command.selectedSessions).toEqual([]);
      expect(command.sessionDeletes).toEqual(["scratch-1"]);
      expect(command.backupForks).toEqual([]);
    },
  );

  test("retains the checkpoint discriminator when the failure toast also fails", async () => {
    const messages = nativeCheckpointFixture({ tailStartID: null });
    addProviderTokens(messages);
    const raced = [
      ...messages,
      nativeMarker("raced-marker", "suffix-2-assistant"),
    ];
    const notification = new Error("failure toast failed");
    const command = createCommandHarness(messages, {
      racedMessages: raced,
      showToastError: notification,
    });

    let error: unknown;
    try {
      await executeMagicCompact(command.v2, "source", 1);
    } catch (caught) {
      error = caught;
    }

    expect(error).toBeInstanceOf(NativeCheckpointChangedError);
    expect((error as Error).cause).toBeInstanceOf(AggregateError);
    expect(nestedErrorMessages((error as Error).cause)).toEqual(
      expect.arrayContaining([
        "Native compaction marker raced-marker is incomplete or errored.",
        "failure toast failed",
      ]),
    );
    expect(command.selectedSessions).toEqual([]);
    expect(command.sessionDeletes).toEqual(["scratch-1"]);
    expect(command.events).not.toContain("delete-source");
  });

  test.each([
    ["dangling tail anchor", { tailStartID: "missing" }, "tail_start_id"],
    [
      "forward tail anchor",
      { tailStartID: "suffix-1-assistant" },
      "tail_start_id",
    ],
    [
      "duplicate marker parts",
      { duplicateParts: true },
      "multiple compaction parts",
    ],
    ["unfinished summary", { finish: null }, "incomplete"],
    ["errored summary", { error: { name: "AbortError" } }, "incomplete"],
  ])("rejects %s before mutation", async (_name, fixtureOptions, error) => {
    await expect(
      planFor(nativeCheckpointFixture(fixtureOptions), 0),
    ).rejects.toThrow(error);
  });

  test("rejects a marker without a completed summary", async () => {
    await expect(
      planFor(
        [nativeMarker("marker", null), ...completedTurn("after", "After")],
        0,
      ),
    ).rejects.toThrow("incomplete");
  });

  test("rejects a compaction part attached to an assistant message", async () => {
    await expect(
      planFor(
        [
          assistantCompaction("assistant-marker"),
          ...completedTurn("after", "After marker"),
        ],
        0,
      ),
    ).rejects.toThrow("user message");
  });

  test("rejects multiple successful summaries for one marker", async () => {
    const messages = nativeCheckpointFixture({ tailStartID: null });
    messages.splice(
      messages.findIndex(item => item.info.id === "native-summary") + 1,
      0,
      nativeSummary("native-summary-duplicate", "native-marker"),
    );

    await expect(planFor(messages, 0)).rejects.toThrow(
      "multiple completed summaries",
    );
  });

  test("rejects an orphan native summary", async () => {
    await expect(
      planFor(
        [
          nativeSummary("orphan", "missing"),
          ...completedTurn("after", "After"),
        ],
        0,
      ),
    ).rejects.toThrow("orphan");
  });

  test.each([
    [
      "a user compaction part",
      {
        user: [nativeMarker("native-marker", null)],
        assistants: [message("assistant", "assistant", [])],
      },
    ],
    [
      "a native summary assistant",
      {
        user: [message("user", "user", [])],
        assistants: [nativeSummary("native-summary", "native-marker")],
      },
    ],
  ])(
    "prune preflight rejects %s before the first mutation",
    async (_name, turn) => {
      const deletes: unknown[] = [];
      const v2 = {
        part: {
          delete: async (request: unknown) => {
            deletes.push(request);
            return { data: true };
          },
        },
        session: {
          deleteMessage: async (request: unknown) => {
            deletes.push(request);
            return { data: true };
          },
        },
      } as unknown as V2Client;

      await expect(
        pruneSummarizedTurns({ v2, sessionID: "source" }, [turn], []),
      ).rejects.toThrow("native compaction artifact");
      expect(deletes).toEqual([]);
    },
  );
});

describe("backup recovery", () => {
  test("deletes a partial backup when the source omission cache is malformed", async () => {
    const sourceID = "backup_malformed_cache_source";
    const backupID = "backup_malformed_cache_target";
    await mkdir(dirname(cachePath(sourceID)), { recursive: true });
    await Bun.write(cachePath(sourceID), "not-json\n");
    const harness = createBackupHarness(backupID);

    await expect(
      createBackup(harness.v2, session(sourceID), 1),
    ).rejects.toThrow("Invalid omission cache");

    expect(harness.deletedSessions).toEqual([backupID]);
    expect(await Bun.file(cachePath(backupID)).exists()).toBeFalse();
    expect(await Bun.file(statsPath(backupID)).exists()).toBeFalse();
  });

  test("deletes a partial backup when copying source stats fails", async () => {
    const sourceID = "backup_malformed_stats_source";
    const backupID = "backup_malformed_stats_target";
    const sourceStatsPath = statsPath(sourceID);
    await mkdir(dirname(sourceStatsPath), { recursive: true });
    await Bun.write(sourceStatsPath, "not-json\n");
    const harness = createBackupHarness(backupID);

    await expect(
      createBackup(harness.v2, session(sourceID), 1),
    ).rejects.toThrow();

    expect(harness.deletedSessions).toEqual([backupID]);
    expect(await Bun.file(cachePath(backupID)).exists()).toBeFalse();
    expect(await Bun.file(statsPath(backupID)).exists()).toBeFalse();
  });

  test("deletes a partial backup when metadata and title initialization fails", async () => {
    const backupID = "backup_metadata_target";
    const harness = createBackupHarness(backupID, {
      updateResponseError: "backup metadata update failed",
    });

    await expect(
      createBackup(harness.v2, session("backup_metadata_source"), 1),
    ).rejects.toThrow("backup metadata update failed");

    expect(harness.deletedSessions).toEqual([backupID]);
    expect(await Bun.file(cachePath(backupID)).exists()).toBeFalse();
    expect(await Bun.file(statsPath(backupID)).exists()).toBeFalse();
  });

  test("retains every target-artifact and session cleanup failure", async () => {
    const backupID = "backup_delete_failure_target";
    const omissionTarget = cachePath(backupID);
    const statsTarget = statsPath(backupID);
    const harness = createBackupHarness(backupID, {
      updateResponseError: "backup metadata update failed",
      deleteResponseError: "partial backup delete failed",
      onUpdate: async () => {
        await rm(omissionTarget, { force: true });
        await mkdir(omissionTarget, { recursive: true });
        await rm(statsTarget, { force: true });
        await mkdir(statsTarget, { recursive: true });
      },
    });
    let error: unknown;

    try {
      await createBackup(
        harness.v2,
        session("backup_delete_failure_source"),
        1,
      );
    } catch (caught) {
      error = caught;
    }

    expect(error).toBeInstanceOf(AggregateError);
    expect(nestedErrorMessages(error)).toEqual([
      "backup metadata update failed",
      "Failed to delete partial backup omission cache.",
      "Failed to delete partial backup stats cache.",
      "partial backup delete failed",
    ]);
    expect(harness.deletedSessions).toEqual([backupID]);
    await rm(omissionTarget, { recursive: true, force: true });
    await rm(statsTarget, { recursive: true, force: true });
  });

  test("selects the backup before deleting the source", async () => {
    const operations: string[] = [];
    const v2 = {
      session: {
        update: async () => {
          operations.push("update");
          return { data: session("backup") };
        },
        delete: async () => {
          operations.push("delete");
          return { data: true };
        },
      },
      tui: {
        selectSession: async () => {
          operations.push("select");
          return { data: true };
        },
      },
    } as unknown as V2Client;

    await applyBackup(v2, session("source"), session("backup"));

    expect(operations).toEqual(["update", "select", "delete"]);
  });

  test("leaves the source intact when selecting the backup fails", async () => {
    const operations: string[] = [];
    const v2 = {
      session: {
        update: async () => {
          operations.push("update");
          return { data: session("backup") };
        },
        delete: async () => {
          operations.push("delete");
          return { data: true };
        },
      },
      tui: {
        selectSession: async () => {
          operations.push("select");
          return { error: "select failed" };
        },
      },
    } as unknown as V2Client;

    await expect(
      applyBackup(v2, session("source"), session("backup")),
    ).rejects.toThrow("select failed");
    expect(operations).toEqual(["update", "select"]);
  });
});

function createBackupHarness(
  backupID: string,
  options: {
    updateResponseError?: string;
    deleteResponseError?: string;
    onUpdate?: () => Promise<void>;
  } = {},
) {
  const deletedSessions: string[] = [];
  const v2 = {
    session: {
      fork: async () => ({ data: session(backupID) }),
      update: async () => {
        await options.onUpdate?.();
        return options.updateResponseError
          ? { error: options.updateResponseError }
          : { data: session(backupID) };
      },
      delete: async (request: { sessionID: string }) => {
        deletedSessions.push(request.sessionID);
        return options.deleteResponseError
          ? { error: options.deleteResponseError }
          : { data: true };
      },
    },
  } as unknown as V2Client;
  return { v2, deletedSessions };
}

function statsPath(sessionID: string): string {
  return join(dirname(cachePath(sessionID)), "stats", `${sessionID}.json`);
}

type RunOptions = {
  agents?: Agent[];
  lifecycle?: string[];
  onPrompt?: (signal?: AbortSignal) => Promise<void>;
  promptError?: Error;
  promptErrorAt?: number;
  promptResponseError?: string;
  promptInfoErrors?: Array<{
    name: string;
    data?: unknown;
    parts?: Array<{ type: "text"; text: string }>;
  }>;
  abortRequests?: Record<string, unknown>[];
  deleteRequests?: Record<string, unknown>[];
  abortError?: Error;
  deleteError?: Error;
  deleteErrorAt?: number;
  messages?: MessageWithParts[];
  textResponses?: Array<string | string[] | undefined>;
  partUpdates?: Record<string, unknown>[];
};

async function runDeterministicRenderer(
  sourceSession: Session,
  options: RunOptions = {},
) {
  return runCompaction(sourceSession, options, true);
}

async function runCompaction(
  sourceSession: Session,
  options: RunOptions = {},
  deterministicRenderer = false,
): Promise<{
  creates: Record<string, unknown>[];
  forks: Record<string, unknown>[];
  aborts: Record<string, unknown>[];
  deletes: string[];
  prompts: Record<string, unknown>[];
  updates: Record<string, unknown>[];
  agentRequests: Record<string, unknown>[];
  partUpdates: Record<string, unknown>[];
  v2: V2Client;
  maxPromptConcurrency: number;
}> {
  const creates: Record<string, unknown>[] = [];
  const forks: Record<string, unknown>[] = [];
  const aborts = options.abortRequests ?? [];
  const deleteRequests = options.deleteRequests ?? [];
  const deletes: string[] = [];
  const prompts: Record<string, unknown>[] = [];
  const updates: Record<string, unknown>[] = [];
  const agentRequests: Record<string, unknown>[] = [];
  const partUpdates = options.partUpdates ?? [];
  const lifecycle = options.lifecycle ?? [];
  const sourceMessages = options.messages ?? compactableMessages();
  let forkCount = 0;
  let scratches = 0;
  let promptCount = 0;
  let promptConcurrency = 0;
  let maxPromptConcurrency = 0;
  let deleteCount = 0;
  const promptedWithFormat = new Map<string, boolean>();
  const v2 = {
    app: {
      agents: async (request: Record<string, unknown>) => {
        lifecycle.push("agents");
        agentRequests.push(request);
        return { data: options.agents ?? [compactionAgent()] };
      },
    },
    session: {
      messages: async (request: Record<string, unknown>) => {
        lifecycle.push("messages");
        if (request.sessionID && promptedWithFormat.get(request.sessionID)) {
          return {
            error: "OpenCode 1.18.15 format-poisoned session.messages read",
          };
        }
        if (request.sessionID?.startsWith("scratch")) {
          return { data: [] };
        }
        return { data: sourceMessages };
      },
      create: async (request: Record<string, unknown>) => {
        lifecycle.push("create");
        creates.push(request);
        scratches += 1;
        return { data: session(`scratch-${scratches}`) };
      },
      fork: async (request: Record<string, unknown>) => {
        lifecycle.push("fork");
        forks.push(request);
        forkCount += 1;
        return {
          data: session(
            forkCount === 1 ? "ephemeral" : `ephemeral-${forkCount}`,
          ),
        };
      },
      update: async (request: Record<string, unknown>) => {
        lifecycle.push("update");
        updates.push(request);
        return { data: session("ephemeral") };
      },
      prompt: async (
        request: Record<string, unknown>,
        sdkOptions?: { signal?: AbortSignal },
      ) => {
        lifecycle.push("prompt");
        prompts.push(request);
        promptCount += 1;
        promptConcurrency += 1;
        maxPromptConcurrency = Math.max(
          maxPromptConcurrency,
          promptConcurrency,
        );
        try {
          await options.onPrompt?.(sdkOptions?.signal);
        } finally {
          promptConcurrency -= 1;
        }
        promptedWithFormat.set(String(request.sessionID), "format" in request);
        if (
          options.promptError
          && (options.promptErrorAt === undefined
            || options.promptErrorAt === promptCount)
        ) {
          throw options.promptError;
        }
        if (options.promptResponseError) {
          return { error: options.promptResponseError };
        }
        const infoError = options.promptInfoErrors?.[promptCount - 1];
        if (infoError !== undefined) {
          return {
            data: {
              info: { error: infoError },
              parts: infoError.parts ?? [],
            },
          };
        }
        const responseIndex = promptCount - 1;
        const textResponse =
          options.textResponses && responseIndex in options.textResponses
            ? options.textResponses[responseIndex]
            : summaryJsonForPrompt(request);
        const texts = Array.isArray(textResponse)
          ? textResponse
          : textResponse === undefined
            ? []
            : [textResponse];
        return {
          data: {
            info: {},
            parts: texts.map(text => ({ type: "text", text })),
          },
        };
      },
      abort: async (request: Record<string, unknown>) => {
        lifecycle.push("abort");
        aborts.push(request);
        if (options.abortError) throw options.abortError;
        return { data: true };
      },
      delete: async (request: { sessionID: string }) => {
        lifecycle.push("delete");
        deleteRequests.push(request);
        deletes.push(request.sessionID);
        deleteCount += 1;
        if (options.deleteErrorAt === deleteCount) {
          throw options.deleteError;
        }
        return { data: true };
      },
    },
    part: {
      update: async (request: Record<string, unknown>) => {
        partUpdates.push(request);
        const part = request.part as Part;
        const target = sourceMessages.find(
          message => message.info.id === request.messageID,
        );
        if (target) {
          const index = target.parts.findIndex(item => item.id === part.id);
          if (index === -1) {
            target.parts.push(part);
          } else {
            target.parts[index] = part;
          }
        }
        return { data: part };
      },
    },
  } as unknown as V2Client;

  const plan = await createCompactionPlan(v2, "source", 0);
  const summaries = deterministicRenderer
    ? plan.summarizedTurns.map(renderDeterministicSummary)
    : await generateCompactionSummaries(v2, sourceSession, plan);
  await injectSummaries(v2, "source", plan.summarizedTurns, summaries);
  return {
    creates,
    forks,
    aborts,
    deletes,
    prompts,
    updates,
    agentRequests,
    partUpdates,
    v2,
    maxPromptConcurrency,
  };
}

function createCommandHarness(
  messages: MessageWithParts[],
  options: {
    racedMessages?: MessageWithParts[];
    deleteProgressError?: Error;
    deleteEphemeralError?: Error;
    summaryPromptError?: Error;
    revalidationMessagesError?: Error;
    activateRaceAtSummaryPrompt?: number;
    summaryTextOutputs?: Array<string | undefined>;
    selectBackupError?: Error;
    deleteSourceError?: Error;
    showToastError?: Error;
    failureToastResponseError?: string;
    successToastResponseError?: string;
    sourcePromptError?: Error;
    mutateSourceBeforeFinalCount?: (messages: MessageWithParts[]) => void;
  } = {},
) {
  const events: string[] = [];
  const partUpdates: Record<string, unknown>[] = [];
  const partDeletes: Record<string, unknown>[] = [];
  const messageDeletes: Record<string, unknown>[] = [];
  const sessionUpdates: Record<string, unknown>[] = [];
  const sessionDeletes: string[] = [];
  const selectedSessions: string[] = [];
  const sourcePrompts: Record<string, unknown>[] = [];
  const summaryRequests: Record<string, unknown>[] = [];
  const scratchCreates: Record<string, unknown>[] = [];
  const backupForks: Record<string, unknown>[] = [];
  const toasts: Record<string, unknown>[] = [];
  let forks = 0;
  let raceActive = false;
  let summaryPrompts = 0;
  let summaryConcurrency = 0;
  let maxSummaryConcurrency = 0;
  let scratches = 0;
  let sourceNotices = 0;
  let finalCountMutationApplied = false;
  let currentSourceSession = session("source");

  const v2 = {
    app: { agents: async () => ({ data: [compactionAgent()] }) },
    session: {
      messages: async () => {
        events.push("messages");
        if (
          sourceNotices >= 1
          && !finalCountMutationApplied
          && options.mutateSourceBeforeFinalCount
        ) {
          finalCountMutationApplied = true;
          options.mutateSourceBeforeFinalCount(messages);
        }
        if (raceActive && options.revalidationMessagesError) {
          return { error: options.revalidationMessagesError.message };
        }
        return {
          data:
            raceActive && options.racedMessages
              ? options.racedMessages
              : messages,
        };
      },
      get: async () => ({ data: currentSourceSession }),
      create: async (request: Record<string, unknown>) => {
        scratchCreates.push(request);
        scratches += 1;
        return { data: session(`scratch-${scratches}`) };
      },
      abort: async () => ({ data: true }),
      fork: async (request: Record<string, unknown>) => {
        backupForks.push(request);
        forks += 1;
        return {
          data: session(
            forks === 1
              ? "backup"
              : forks === 2
                ? "ephemeral"
                : `ephemeral-${forks - 1}`,
          ),
        };
      },
      update: async (request: Record<string, unknown>) => {
        sessionUpdates.push(request);
        if (request.sessionID === "source") {
          currentSourceSession = {
            ...currentSourceSession,
            ...(request.metadata ? { metadata: request.metadata } : {}),
          } as Session;
        }
        if (
          request.sessionID === "backup"
          && request.title === "Test session"
        ) {
          events.push("backup-update");
        }
        return { data: session(String(request.sessionID)) };
      },
      prompt: async (request: Record<string, unknown>) => {
        if (
          String(request.sessionID).startsWith("ephemeral")
          || String(request.sessionID).startsWith("scratch")
        ) {
          events.push("summary-prompt");
          summaryPrompts += 1;
          summaryRequests.push(request);
          summaryConcurrency += 1;
          maxSummaryConcurrency = Math.max(
            maxSummaryConcurrency,
            summaryConcurrency,
          );
          if (summaryPrompts === (options.activateRaceAtSummaryPrompt ?? 1)) {
            raceActive = true;
          }
          if (options.summaryPromptError) {
            summaryConcurrency -= 1;
            throw options.summaryPromptError;
          }
          const responseIndex = summaryPrompts - 1;
          const text =
            options.summaryTextOutputs
            && responseIndex in options.summaryTextOutputs
              ? options.summaryTextOutputs[responseIndex]
              : summaryJsonForPrompt(request);
          const response = {
            data: {
              info: {},
              parts: text === undefined ? [] : [{ type: "text", text }],
            },
          };
          summaryConcurrency -= 1;
          return response;
        }

        if (request.noReply === true) {
          const parts = request.parts as
            | Array<{ metadata?: { magicCompact?: { progress?: boolean } } }>
            | undefined;
          if (parts?.some(part => part.metadata?.magicCompact?.progress)) {
            events.push("progress-prompt");
          }
        }

        sourcePrompts.push(request);
        sourceNotices += 1;
        if (options.sourcePromptError) throw options.sourcePromptError;
        const messageID = `source-notice-${sourceNotices}`;
        const parts = ((request.parts as Part[] | undefined) ?? []).map(
          (part, index) =>
            ({
              ...part,
              id: `${messageID}-part-${index}`,
              sessionID: "source",
              messageID,
            }) as Part,
        );
        messages.push(message(messageID, "user", parts));
        return {
          data: {
            info: { id: messageID },
            parts,
          },
        };
      },
      deleteMessage: async (request: Record<string, unknown>) => {
        messageDeletes.push(request);
        if (options.deleteProgressError) {
          throw options.deleteProgressError;
        }
        const index = messages.findIndex(
          message => message.info.id === request.messageID,
        );
        if (index !== -1) messages.splice(index, 1);
        return { data: true };
      },
      delete: async (request: { sessionID: string }) => {
        sessionDeletes.push(request.sessionID);
        if (request.sessionID === "source") {
          events.push("delete-source");
          if (options.deleteSourceError) throw options.deleteSourceError;
        }
        if (
          options.deleteEphemeralError
          && request.sessionID.startsWith("scratch")
        ) {
          throw options.deleteEphemeralError;
        }
        return { data: true };
      },
    },
    part: {
      update: async (request: Record<string, unknown>) => {
        partUpdates.push(request);
        const part = request.part as Part;
        const metadata = (part as TextPart).metadata as
          | { magicCompact?: { summary?: boolean; boundary?: boolean } }
          | undefined;
        if (metadata?.magicCompact?.summary) {
          events.push("summary-update");
        } else if (metadata?.magicCompact?.boundary) {
          events.push("boundary-update");
        }

        const target = messages.find(
          message => message.info.id === request.messageID,
        );
        if (target) {
          const index = target.parts.findIndex(item => item.id === part.id);
          if (index === -1) {
            target.parts.push(part);
          } else {
            target.parts[index] = part;
          }
        }
        return { data: part };
      },
      delete: async (request: Record<string, unknown>) => {
        partDeletes.push(request);
        const target = messages.find(
          message => message.info.id === request.messageID,
        );
        if (target) {
          target.parts = target.parts.filter(
            part => part.id !== request.partID,
          );
        }
        return { data: true };
      },
    },
    tui: {
      showToast: async (request: Record<string, unknown>) => {
        toasts.push(request);
        if (request.title === "Magic Compact Failed") {
          events.push("failure-toast");
          if (options.showToastError) throw options.showToastError;
          if (options.failureToastResponseError) {
            return { error: options.failureToastResponseError };
          }
        } else if (
          request.title === "Magic Compact"
          && options.successToastResponseError
        ) {
          return { error: options.successToastResponseError };
        }
        return { data: true };
      },
      selectSession: async (request: { sessionID: string }) => {
        selectedSessions.push(request.sessionID);
        events.push("select-backup");
        if (options.selectBackupError) throw options.selectBackupError;
        return { data: true };
      },
    },
  } as unknown as V2Client;

  return {
    v2,
    events,
    partUpdates,
    partDeletes,
    messageDeletes,
    sessionUpdates,
    sessionDeletes,
    selectedSessions,
    sourcePrompts,
    summaryRequests,
    get maxSummaryConcurrency() {
      return maxSummaryConcurrency;
    },
    get currentSourceSession() {
      return currentSourceSession;
    },
    scratchCreates,
    backupForks,
    toasts,
  };
}

function addProviderTokens(messages: MessageWithParts[]): void {
  for (const message of messages) {
    if (message.info.role === "assistant") {
      Object.assign(message.info, {
        tokens: {
          input: 0,
          output: 0,
          reasoning: 0,
          cache: { read: 0, write: 0 },
        },
      });
    }
  }
  Object.assign(messages.at(-1)!.info, {
    tokens: {
      input: 10,
      output: 5,
      reasoning: 0,
      cache: { read: 0, write: 0 },
    },
  });
}

function compactableMessages(): MessageWithParts[] {
  return [
    message("user", "user", [
      textPart("user-text", "source", "user", "Request"),
    ]),
    message("assistant", "assistant", []),
  ];
}

function message(
  id: string,
  role: "user" | "assistant",
  parts: Part[],
): MessageWithParts {
  return {
    info: { id, role } as Message,
    parts,
  };
}

function session(id: string): Session {
  return {
    id,
    title: "Test session",
    directory: "/workspace",
    workspaceID: "workspace",
  } as Session;
}

function compactionAgent(overrides: Partial<Agent> = {}): Agent {
  return {
    name: "compaction",
    mode: "primary",
    native: true,
    hidden: true,
    permission: [],
    options: {},
    ...overrides,
  };
}

function textPart(
  id: string,
  sessionID: string,
  messageID: string,
  text: string,
): TextPart {
  return {
    id,
    sessionID,
    messageID,
    type: "text",
    text,
  };
}

type NativeFixtureOptions = {
  tailStartID?: string | null;
  duplicateParts?: boolean;
  finish?: string | null;
  error?: unknown;
};

function nativeCheckpointFixture(
  options: NativeFixtureOptions = {},
): MessageWithParts[] {
  const messages = [
    ...completedTurn("tail", "Pre-checkpoint tail"),
    nativeMarker(
      "native-marker",
      options.tailStartID === undefined
        ? "tail-assistant"
        : options.tailStartID,
      options.duplicateParts,
    ),
    nativeSummary("native-summary", "native-marker", {
      finish: options.finish === undefined ? "stop" : options.finish,
      error: options.error,
    }),
    ...completedTurn("suffix-1", "First suffix request"),
    ...completedTurn("suffix-2", "Second suffix request"),
  ];
  messages
    .find(message => message.info.id === "suffix-1-assistant")!
    .parts.push({
      id: "suffix-1-removable-detail",
      sessionID: "source",
      messageID: "suffix-1-assistant",
      type: "reasoning",
      text: "historical removable detail ".repeat(2_000),
      time: { start: 1, end: 2 },
    } as Part);
  return messages;
}

function completedTurn(
  id: string,
  text: string,
  boundary = false,
): MessageWithParts[] {
  const userID = `${id}-user`;
  return [
    message(userID, "user", [
      textPart(`${id}-text`, "source", userID, text),
      ...(boundary ? [magicBoundary(`${id}-boundary`, userID)] : []),
    ]),
    message(`${id}-assistant`, "assistant", []),
  ];
}

function nativeMarker(
  id: string,
  tailStartID: string | null,
  duplicate = false,
  sessionID = "source",
): MessageWithParts {
  const part = {
    id: `${id}-part`,
    sessionID,
    messageID: id,
    type: "compaction",
    auto: true,
    ...(tailStartID ? { tail_start_id: tailStartID } : {}),
  } as unknown as Part;
  return message(
    id,
    "user",
    duplicate ? [part, { ...part, id: `${id}-part-2` }] : [part],
  );
}

function assistantCompaction(id: string): MessageWithParts {
  const marker = nativeMarker(id, null);
  marker.info = { ...marker.info, role: "assistant" } as Message;
  return marker;
}

function nativeSummary(
  id: string,
  parentID: string,
  options: { finish?: string | null; error?: unknown } = {},
): MessageWithParts {
  return {
    info: {
      id,
      role: "assistant",
      parentID,
      summary: true,
      finish: options.finish === undefined ? "stop" : options.finish,
      ...(options.error === undefined ? {} : { error: options.error }),
    } as Message,
    parts: [textPart(`${id}-text`, "source", id, "Native summary text")],
  };
}

function magicBoundary(id: string, messageID: string): TextPart {
  return {
    ...textPart(id, "source", messageID, "Magic boundary"),
    synthetic: true,
    metadata: { magicCompact: { boundary: true } },
  };
}

async function planFor(messages: MessageWithParts[], keepTurns: number) {
  const v2 = {
    session: { messages: async () => ({ data: messages }) },
  } as unknown as V2Client;
  return createCompactionPlan(v2, "source", keepTurns);
}

function assistantIDs(turns: Turn[]): string[] {
  return turns.flatMap(turn => turn.assistants.map(item => item.info.id));
}

function nestedErrorMessages(error: unknown): string[] {
  if (error instanceof AggregateError) {
    return error.errors.flatMap(nestedErrorMessages);
  }
  return [error instanceof Error ? error.message : String(error)];
}

type ChatParamsHook = NonNullable<
  Awaited<ReturnType<typeof server>>["chat.params"]
>;

function chatParamsInput(
  sessionID: string,
  agent: string,
): Parameters<ChatParamsHook>[0] {
  return {
    sessionID,
    agent,
    model: {} as Parameters<ChatParamsHook>[0]["model"],
    provider: {} as Parameters<ChatParamsHook>[0]["provider"],
    message: {} as Parameters<ChatParamsHook>[0]["message"],
  };
}

function chatParams(
  maxOutputTokens: number | undefined,
  options: Record<string, unknown> = {},
): Parameters<ChatParamsHook>[1] {
  return {
    temperature: 0,
    topP: 1,
    topK: 0,
    maxOutputTokens,
    options,
  };
}

function promptText(request: Record<string, unknown>): string {
  const parts = request.parts as Array<{ text?: string }> | undefined;
  return parts?.[0]?.text ?? "";
}

function storedSummary(
  requests: { partUpdates: Record<string, unknown>[] },
  messageID: string,
): string {
  const update = requests.partUpdates.find(
    item => item.messageID === messageID,
  );
  expect(update).toBeDefined();
  return (update!.part as TextPart).text;
}

function summaryText(messages: MessageWithParts[], messageID: string): string {
  const part = messages
    .find(message => message.info.id === messageID)
    ?.parts.find(
      (item): item is TextPart =>
        item.type === "text"
        && (
          item.metadata?.magicCompact as Record<string, unknown> | undefined
        )?.["summary"] === true,
    );
  expect(part).toBeDefined();
  return part!.text;
}

function boundaryParts(messages: MessageWithParts[]): TextPart[] {
  return messages.flatMap(message =>
    message.parts.filter(
      (part): part is TextPart =>
        part.type === "text"
        && (
          part.metadata?.magicCompact as Record<string, unknown> | undefined
        )?.["boundary"] === true,
    ),
  );
}

function userTextByPartID(
  messages: MessageWithParts[],
): Record<string, string> {
  return Object.fromEntries(
    messages.flatMap(message =>
      message.info.role === "user"
        ? message.parts.flatMap(part =>
            part.type === "text"
            && part.synthetic !== true
            && part.ignored !== true
              ? [[part.id, part.text] as const]
              : [],
          )
        : [],
    ),
  );
}

function omissionContentIDs(part: ToolPart): string[] {
  const state = part.state as unknown as Record<string, unknown>;
  const serialized = JSON.stringify([state["input"], state["output"]]);
  return [
    ...new Set(
      serialized.match(/[A-Za-z0-9_-]{1,12}:omitted-[A-Za-z0-9_-]{22}/g) ?? [],
    ),
  ];
}

async function canonicalOmissionTool() {
  const hooks = await server({} as PluginInput);
  const omissionTool = hooks.tool?.read_omitted_content;
  expect(omissionTool).toBeDefined();
  return omissionTool!;
}

function omissionToolContext(sessionID: string): ToolContext {
  return {
    sessionID,
    messageID: "test-message",
    agent: "test-agent",
    directory: "/workspace",
    worktree: "/workspace",
    abort: new AbortController().signal,
    metadata() {},
    async ask() {},
  };
}

function hasUnpairedSurrogate(value: string): boolean {
  for (let index = 0; index < value.length; index++) {
    const current = value.charCodeAt(index);
    if (current >= 0xd800 && current <= 0xdbff) {
      const next = value.charCodeAt(index + 1);
      if (next < 0xdc00 || next > 0xdfff) return true;
      index += 1;
    } else if (current >= 0xdc00 && current <= 0xdfff) {
      return true;
    }
  }
  return false;
}

function labeledSummarySection(
  value: string,
  label: string,
  nextLabel?: string,
): string {
  const start = value.indexOf(`${label}:`);
  expect(start).toBeGreaterThanOrEqual(0);
  const end = nextLabel
    ? value.indexOf(`${nextLabel}:`, start + 1)
    : value.length;
  expect(end).toBeGreaterThan(start);
  return value.slice(start, end).trim();
}

function capsuleSection(value: string, label: string): string {
  const start = value.indexOf(`${label}:\n`);
  expect(start).toBeGreaterThanOrEqual(0);
  const tail = value.slice(start);
  const next = tail
    .slice(label.length + 2)
    .match(/^(?:Work|Verification|VCS|Unresolved|Next):$/m);
  const end = next ? tail.indexOf(next[0], label.length + 2) : tail.length;
  return tail.slice(0, end).trim();
}

function summaryRequestIDs(request: Record<string, unknown>): string[] {
  return summaryPromptSelection(request).turns.map(
    turn => turn.assistantTurnID,
  );
}

function summaryRequestEvidence(request: Record<string, unknown>): string {
  const turns = summaryPromptSelection(request).turns;
  expect(turns).toHaveLength(1);
  return turns[0]!.exactEvidence;
}

function summaryPromptSelection(request: Record<string, unknown>): {
  turns: Array<{ assistantTurnID: string; exactEvidence: string }>;
} {
  const text = promptText(request);
  const startMarker = "Selected historical turns and stop anchor:\n";
  const endMarker = "\n\nEach summary";
  const start = text.indexOf(startMarker);
  const end = text.indexOf(endMarker, start + startMarker.length);
  if (start === -1 || end === -1) {
    throw new Error("Summary prompt selection is missing or malformed.");
  }
  return JSON.parse(text.slice(start + startMarker.length, end)) as {
    turns: Array<{ assistantTurnID: string; exactEvidence: string }>;
  };
}

type AssistantTextRecord = {
  kind: "assistant-text";
  text?: string;
  textEncoding?: string;
  textSegments?: Array<{ literal: string; repetitions: number }>;
};

function machineEvidenceRecords(value: string): Record<string, unknown>[] {
  return value.split("\n").flatMap(line => {
    if (!line.startsWith("{")) return [];
    return [JSON.parse(line) as Record<string, unknown>];
  });
}

function assistantTextRecords(value: string): AssistantTextRecord[] {
  return machineEvidenceRecords(value).filter(
    (record): record is AssistantTextRecord =>
      record["kind"] === "assistant-text",
  );
}

function decodeAssistantTextRecord(record: AssistantTextRecord): string {
  if (record.text !== undefined) return record.text;
  if (
    record.textEncoding !== "ordered-literal-runs-v1"
    || record.textSegments === undefined
  ) {
    throw new Error("Assistant text record has no supported exact encoding.");
  }
  return record.textSegments
    .map(segment => segment.literal.repeat(segment.repetitions))
    .join("");
}

function standardEvidenceForAssistantText(text: string): string {
  const user = message("repeat-user", "user", []);
  const assistant = message("repeat-assistant", "assistant", [
    textPart("repeat-text", "source", "repeat-assistant", text),
  ]);
  return renderStandardTurnEvidence({ user: [user], assistants: [assistant] });
}

function codeUnits(value: string): number[] {
  return Array.from({ length: value.length }, (_, index) =>
    value.charCodeAt(index),
  );
}

function deterministicHighEntropyText(bytes: number): string {
  const alphabet =
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \t\n";
  let seed = 12_345;
  const next = () => {
    seed = (seed * 1_103_515_245 + 12_345) % 2_147_483_648;
    return seed / 2_147_483_648;
  };
  const chunks: string[] = [];
  let total = 0;
  while (total < bytes) {
    const chunk = Array.from(
      { length: 4_096 },
      () => alphabet[Math.floor(next() * alphabet.length)],
    ).join("");
    chunks.push(chunk);
    total += chunk.length;
  }
  return chunks.join("");
}

function modelCapsule(
  outcome: "completed" | "partial" | "analysis-only",
  work: string,
): string {
  return `Outcome: ${outcome}\nWork: ${work}`;
}

function summaryJsonForPrompt(request: Record<string, unknown>): string {
  const selection = summaryPromptSelection(request);
  return JSON.stringify(
    Object.fromEntries(
      selection.turns.map(({ assistantTurnID, exactEvidence }) => {
        const outcome = exactEvidence.match(
          /^Outcome: (completed|partial|analysis-only)$/m,
        )?.[1] as "completed" | "partial" | "analysis-only" | undefined;
        if (outcome === undefined) {
          throw new Error("Standard evidence is missing its expected Outcome.");
        }
        return [
          assistantTurnID,
          modelCapsule(
            outcome,
            assistantTurnID === "assistant"
              ? "Completed the request."
              : "Preserved.",
          ),
        ];
      }),
    ),
  );
}

function commandToolPart(
  id: string,
  messageID: string,
  command: string,
  output: string,
): Part {
  return {
    id,
    sessionID: "source",
    messageID,
    type: "tool",
    callID: `${id}-call`,
    tool: "bash",
    state: {
      status: "completed",
      input: { command },
      output,
      title: "Shell command",
      metadata: {},
      time: { start: 1, end: 2 },
    },
  } as Part;
}

function genericToolPart(
  id: string,
  messageID: string,
  tool: string,
  state: Record<string, unknown>,
): Part {
  return {
    id,
    sessionID: "source",
    messageID,
    type: "tool",
    callID: `${id}-call`,
    tool,
    state,
  } as Part;
}

function pendingCommandToolPart(
  id: string,
  messageID: string,
  command: string,
): Part {
  return {
    id,
    sessionID: "source",
    messageID,
    type: "tool",
    callID: `${id}-call`,
    tool: "bash",
    state: {
      status: "pending",
      input: { command },
      metadata: {},
    },
  } as Part;
}

function outlierTurn(): MessageWithParts[] {
  const user = message("outlier-user", "user", [
    textPart("outlier-user-text", "source", "outlier-user", "Outlier request"),
  ]);
  const assistants = Array.from({ length: 44 }, (_, index) =>
    message(`outlier-assistant-${index}`, "assistant", []),
  );
  const completed = (input: Record<string, unknown>, output: string) => ({
    status: "completed" as const,
    input,
    output,
    title: "Shell command",
    metadata: {},
    time: { start: 1, end: 2 },
  });
  const evidence = [
    {
      tool: "write",
      state: completed(
        { filePath: "src/critical.ts", content: "hardened implementation" },
        "Wrote src/critical.ts",
      ),
    },
    {
      tool: "bash",
      state: completed({ command: "bun test" }, "Process exited with code 1"),
    },
    {
      tool: "bash",
      state: completed({ command: "git status --short" }, "working tree clean"),
    },
    {
      tool: "bash",
      state: completed(
        { command: 'git commit -m "harden summaries" && git push' },
        "[hardening abc123] harden summaries\npush succeeded",
      ),
    },
    {
      tool: "bash",
      state: completed({ command: "bun run smoke" }, "smoke not run"),
    },
    {
      tool: "bash",
      state: {
        status: "error",
        input: { command: "bun run integration" },
        error: "integration failed",
        metadata: {},
        time: { start: 1, end: 2 },
      },
    },
    {
      tool: "bash",
      state: {
        status: "running",
        input: { command: "bun run long-check" },
        title: "Shell command",
        metadata: {},
        time: { start: 1 },
      },
    },
  ];
  assistants[0]!.parts.push(
    ...Array.from({ length: 64 }, (_, index) => {
      const item = evidence[index] ?? {
        tool: "read",
        state: completed(
          { filePath: `evidence-${index}` },
          `evidence ${index}`,
        ),
      };
      return {
        id: `outlier-tool-${index}`,
        sessionID: "source",
        messageID: assistants[0]!.info.id,
        type: "tool",
        callID: `outlier-call-${index}`,
        tool: item.tool,
        state: item.state,
      } as Part;
    }),
  );
  assistants
    .at(-1)!
    .parts.push(
      textPart(
        "outlier-final",
        "source",
        assistants.at(-1)!.info.id,
        `Delivered implementation in src/critical.ts. ${"x".repeat(450_000)} bun test reported 88 pass; commit abc123 was pushed; smoke not run; integration failed.`,
      ),
      textPart(
        "outlier-final-evidence",
        "source",
        assistants.at(-1)!.info.id,
        "final assistant evidence",
      ),
    );
  return [user, ...assistants];
}

function v2Cache(contentID: string, content: string) {
  return {
    version: 2 as const,
    entries: {
      [contentID]: {
        content,
        sha256: contentSha256(contentID, content),
      },
    },
    legacy: { entries: {} },
  };
}

function contentSha256(contentID: string, content: string): string {
  const id = Buffer.from(contentID);
  const idLength = Buffer.allocUnsafe(4);
  idLength.writeUInt32BE(id.length);
  return createHash("sha256")
    .update("magic-compact:omission-entry:v2\0")
    .update(idLength)
    .update(id)
    .update(Buffer.from(content, "utf16le"))
    .digest("hex");
}

function metricTurn(
  serializedBytes: number,
  messageCount: number,
  toolCount: number,
  status: "completed" | "error" = "completed",
): Turn {
  const user = [message("metric-user", "user", [])];
  const assistants = Array.from(
    { length: Math.max(1, messageCount - 1) },
    (_, index) => message(`metric-assistant-${index}`, "assistant", []),
  );
  assistants[0]!.parts.push(
    ...Array.from(
      { length: toolCount },
      (_, index) =>
        ({
          id: `metric-tool-${index}`,
          sessionID: "source",
          messageID: assistants[0]!.info.id,
          type: "tool",
          callID: `metric-call-${index}`,
          tool: "bash",
          state:
            status === "completed"
              ? {
                  status,
                  input: {},
                  output: "ok",
                  title: "bash",
                  metadata: {},
                  time: { start: 1, end: 2 },
                }
              : {
                  status,
                  input: {},
                  error: "failed",
                  metadata: {},
                  time: { start: 1, end: 2 },
                },
        }) as Part,
    ),
  );
  const filler = textPart(
    "metric-filler",
    "source",
    assistants.at(-1)!.info.id,
    "",
  );
  assistants.at(-1)!.parts.push(filler);
  const turn = { user, assistants };
  const baseBytes = Buffer.byteLength(JSON.stringify(turn));
  filler.text = "x".repeat(Math.max(0, serializedBytes - baseBytes));
  return turn;
}

function distinctUnresolvedTurn(count: number): Turn {
  const user = [message("scale-user", "user", [])];
  const assistant = message("scale-assistant", "assistant", []);
  assistant.parts = Array.from({ length: count }, (_, index) =>
    pendingCommandToolPart(
      `scale-tool-${index}`,
      assistant.info.id,
      `repair-${index} --target /workspace/repair-${index}.json`,
    ),
  );
  return { user, assistants: [assistant] };
}

function statusChainTurn(count: number, trailingSuccess = false): Turn {
  const user = [message("scale-user", "user", [])];
  const assistants: MessageWithParts[] = Array.from(
    { length: count },
    (_, index) =>
      ({
        info: {
          id: `scale-error-${index}`,
          role: "assistant",
          error: {
            name: "ProviderError",
            data: {
              message: `boom ${index}`,
              statusCode: 500,
              providerID: "scale-provider",
            },
          },
        } as Message,
        parts: [],
      }) satisfies MessageWithParts,
  );
  if (trailingSuccess) {
    assistants.push({
      info: {
        id: "scale-success",
        role: "assistant",
        finish: "stop",
      } as Message,
      parts: [],
    });
  }
  return { user, assistants };
}

function completedCommandTurn(
  command: string,
  output: string,
): MessageWithParts[] {
  const turn = metricTurn(1, 20, 0);
  turn.assistants[0]!.parts.push({
    id: "metric-command",
    sessionID: "source",
    messageID: "metric-assistant-0",
    type: "tool",
    callID: "metric-command-call",
    tool: "bash",
    state: {
      status: "completed",
      input: { command },
      output,
      title: "Shell command",
      metadata: {},
      time: { start: 1, end: 2 },
    },
  } as Part);
  return [...turn.user, ...turn.assistants];
}

function nearBoundaryReadTurn(
  id: string,
  output = `near boundary read output ${"y".repeat(287)}`,
): MessageWithParts[] {
  return [
    message(`${id}-user`, "user", [
      textPart(`${id}-text`, "source", `${id}-user`, `Request ${id}`),
    ]),
    message(`${id}-assistant`, "assistant", [
      genericToolPart(`${id}-read`, `${id}-assistant`, "read", {
        status: "completed",
        input: { filePath: `/workspace/${id}.txt` },
        output,
        title: "Read file",
        metadata: {},
        time: { start: 1, end: 2 },
      }),
    ]),
  ];
}

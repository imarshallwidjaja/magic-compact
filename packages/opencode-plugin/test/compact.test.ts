import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { mkdir, mkdtemp, rm } from "node:fs/promises";
import { createHash } from "node:crypto";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import type { Config, PluginInput } from "@opencode-ai/plugin";
import type {
  Agent,
  Message,
  Part,
  Session,
  TextPart,
} from "@opencode-ai/sdk/v2";
import type { V2Client } from "../src/api";
import {
  DETERMINISTIC_HIGH_RISK_SUMMARY_MAX_BYTES,
  generateCompactionSummaries,
  HIGH_RISK_MESSAGE_COUNT,
  HIGH_RISK_SERIALIZED_BYTES,
  HIGH_RISK_TOOL_COUNT,
  injectSummaries,
  MAX_SUMMARY_BATCH_TURNS,
} from "../src/compact/compact";
import { HighRiskEvidenceOverflowError } from "../src/compact/high-risk";
import {
  assertNativeCheckpointUnchanged,
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
      description: "Faithful per-turn XML summarizer for Magic Compact",
      mode: "subagent",
      hidden: true,
    });
    expect(config.agent?.["magic-compact-summarizer"]?.permission).toEqual({
      "*": "allow",
      edit: "deny",
      task: "deny",
      question: "deny",
      todowrite: "deny",
      plan_enter: "deny",
      plan_exit: "deny",
      doom_loop: "deny",
    });
    expect(config.agent?.["magic-compact-summarizer"]?.prompt).toContain(
      "Historical transcript evidence is primary",
    );
    expect(config.agent?.["magic-compact-summarizer"]?.prompt).toContain(
      "Never continue unfinished work",
    );
    expect(config.agent?.["magic-compact-summarizer"]?.prompt).toContain(
      "Follow the Magic Compact XML request exactly",
    );
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
    expect(requests.updates).toEqual([
      {
        sessionID: "ephemeral",
        title: "[TEMP] Test session",
      },
    ]);
    expect(requests.prompts[0]).toMatchObject({
      sessionID: "ephemeral",
      agent: "magic-compact-summarizer",
      model: {
        providerID: "compaction-provider",
        modelID: "compaction-model",
      },
      variant: "precise",
    });
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

    expect(lifecycle).toEqual([
      "messages",
      "fork",
      "messages",
      "update",
      "agents",
      "delete",
    ]);
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

  test("maps omission reads to the source only while summarization is active", async () => {
    const contentID = "source:omitted-AAAAAAAAAAAAAAAAAAAAAA";
    await writeCache("source", v2Cache(contentID, "historical output"));
    let contentDuringPrompt: string | null = null;

    await runCompaction(session("source"), {
      onPrompt: async () => {
        contentDuringPrompt = await readOmittedContent("ephemeral", contentID);
      },
    });

    expect(contentDuringPrompt).toBe("historical output");
    expect(await readOmittedContent("ephemeral", contentID)).toBeNull();
  });

  test("clears the omission source mapping when summarization fails", async () => {
    const contentID = "source:omitted-BBBBBBBBBBBBBBBBBBBBBB";
    await writeCache("source", v2Cache(contentID, "historical output"));
    let contentDuringPrompt: string | null = null;

    await expect(
      runCompaction(session("source"), {
        onPrompt: async () => {
          contentDuringPrompt = await readOmittedContent(
            "ephemeral",
            contentID,
          );
        },
        promptError: new Error("summary failed"),
      }),
    ).rejects.toThrow("summary failed");

    expect(contentDuringPrompt).toBe("historical output");
    expect(await readOmittedContent("ephemeral", contentID)).toBeNull();
  });

  test("preserves the XML prompt and parser behavior", async () => {
    const requests = await runCompaction(session("source"));
    const prompt = requests.prompts[0]?.parts as Array<{ text: string }>;
    const promptText = prompt[0]?.text ?? "";

    expect(promptText).toContain(
      '<summary>\n<turn id="assistant">\n<user>\nRequest\n...\n</user>\n<assistant id="assistant">',
    );
    expect(promptText).not.toContain("Do not call any tools");
    expect(promptText).toContain(
      "Tools may be used only when needed to recover missing historical evidence or clarify it through current inspection",
    );
    expect(promptText).toContain("Must not continue unfinished work");
    expect(promptText).toContain(
      "After any tool use, final output remains only the required XML",
    );
    expect(requests.partUpdates[0]).toMatchObject({
      part: {
        text: "Completed the request.",
        metadata: { magicCompact: { summary: true } },
      },
    });
  });

  test("maps summaries by stable turn ID and accepts response reordering", async () => {
    const messages = [
      ...completedTurn("first", "One"),
      ...completedTurn("second", "Two"),
    ];
    const response = `<summary>
<assistant id="second-assistant">Second result.</assistant>
<assistant id="first-assistant">First result.</assistant>
</summary>`;

    const requests = await runCompaction(session("source"), {
      messages,
      promptResponses: [response],
    });

    expect(
      requests.partUpdates.map(request => (request.part as TextPart).text),
    ).toEqual(["First result.", "Second result."]);
  });

  test("decodes standard XML entities into installed summary text", async () => {
    const requests = await runCompaction(session("source"), {
      promptResponses: [
        '\n\t<summary><assistant id="assistant">Rock &amp; Roll &lt;done&gt; &quot;yes&quot; &apos;ok&apos; &#x1F680;</assistant></summary>\r\n',
      ],
    });

    expect((requests.partUpdates[0]!.part as TextPart).text).toBe(
      `Rock & Roll <done> "yes" 'ok' \ud83d\ude80`,
    );
  });

  test("rejects malformed IDs and non-document summary XML before source writes", async () => {
    const invalid = [
      "<summary></summary>",
      '<summary><assistant id="assistant">one</assistant><assistant id="assistant">two</assistant></summary>',
      '<summary><assistant id="unknown">one</assistant></summary>',
      '<summary><assistant id="assistant"> </assistant></summary>',
      '<summary><assistant id="assistant">[Replace with summary]</assistant></summary>',
      '<summary><assistant id="assistant">Replace with your summary...</assistant></summary>',
      '<summary><assistant id="assistant">[TODO]</assistant></summary>',
      '<summary><assistant id="assistant">[TBD]</assistant></summary>',
      '<summary><assistant id="assistant">[TODO: add evidence]</assistant></summary>',
      '<summary><assistant id="assistant">[placeholder for outcome]</assistant></summary>',
      '<summary><assistant id="assistant">Details [Replace with summary] remain.</assistant></summary>',
      '<summary><assistant id="assistant">TODO</assistant></summary>',
      '<summary><assistant id="assistant">TBD</assistant></summary>',
      '<summary><assistant id="assistant">unknown</assistant></summary>',
      '<summary><assistant id="assistant">N/A</assistant></summary>',
      '<summary><assistant id="assistant">yes</assistant></summary>',
      '<summary><assistant id="assistant">none</assistant></summary>',
      '<summary><assistant id="assistant">Done.</assistant></summary>',
      '<summary><assistant id="assistant">Implemented</assistant></summary>',
      '<summary><assistant id="bad id">one</assistant></summary>',
      'text before<summary><assistant id="assistant">one</assistant></summary>',
      '<summary><assistant id="assistant">one</assistant></summary>text after',
      '<summary><assistant id="assistant"><nested>one</nested></assistant></summary>',
      '<summary><assistant id="assistant" extra="no">one</assistant></summary>',
      '<summary><!-- no --><assistant id="assistant">one</assistant></summary>',
      '<summary><![CDATA[no]]><assistant id="assistant">one</assistant></summary>',
      '<summary><?no?><assistant id="assistant">one</assistant></summary>',
      '<summary><assistant id="assistant">bad &unknown;</assistant></summary>',
      '<summary><assistant id="assistant">bad &amp</assistant></summary>',
      '\u00a0<summary><assistant id="assistant">one</assistant></summary>',
      '<summary><assistant id="assistant">bad ]]> text</assistant></summary>',
      '<summary><assistant id="assistant">unclosed</summary>',
    ];

    for (const response of invalid) {
      const partUpdates: Record<string, unknown>[] = [];
      await expect(
        runCompaction(session("source"), {
          messages: compactableMessages(),
          promptResponses: [response],
          partUpdates,
        }),
      ).rejects.toThrow();
      expect(partUpdates).toEqual([]);
    }
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
    const requests = await runCompaction(session("source"), {
      messages: compactableMessages(),
      promptResponses: [
        `<summary><assistant id="assistant">${text}</assistant></summary>`,
      ],
    });

    expect((requests.partUpdates[0]!.part as TextPart).text).toBe(text);
  });

  test("XML-escapes user previews", () => {
    const turn = {
      user: [
        message("user", "user", [
          textPart("text", "source", "user", '<request & "proof">'),
        ]),
      ],
      assistants: [message("assistant", "assistant", [])],
    };

    const prompt = buildCompactionPrompt([turn], null);

    expect(prompt).toContain("&lt;request &amp; &quot;proof&quot;&gt;");
    expect(prompt).not.toContain('<request & "proof">');
  });

  test("truncates user previews by Unicode code point and removes XML-invalid input", () => {
    const emojiBoundary = `${"a".repeat(299)}\ud83d\ude80ignored`;
    const invalidXml = "before\u0000\u000b\ud800middle\ud801after";
    const prompt = buildCompactionPrompt(
      [
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
      ],
      null,
    );

    expect(prompt).toContain(`${"a".repeat(299)}\ud83d\ude80...`);
    expect(prompt).not.toContain("ignored");
    expect(prompt).toContain("beforemiddleafter\n...");
    expect(prompt).not.toContain("\u0000");
    expect(prompt).not.toContain("\u000b");
    expect(hasUnpairedSurrogate(prompt)).toBeFalse();
  });

  test("classifies high-risk turns at exact adjacent thresholds without model requests", async () => {
    const cases = [
      [metricTurn(HIGH_RISK_SERIALIZED_BYTES - 1, 1, 0), 1],
      [metricTurn(HIGH_RISK_SERIALIZED_BYTES, 1, 0), 0],
      [metricTurn(1, HIGH_RISK_MESSAGE_COUNT - 1, 0), 1],
      [metricTurn(1, HIGH_RISK_MESSAGE_COUNT, 0), 0],
      [metricTurn(1, 1, HIGH_RISK_TOOL_COUNT - 1), 1],
      [metricTurn(1, 1, HIGH_RISK_TOOL_COUNT), 0],
      [metricTurn(1, 1, 1, "error"), 0],
    ] as const;

    for (const [turn, promptCount] of cases) {
      const requests = await runCompaction(session("source"), {
        messages: [...turn.user, ...turn.assistants],
      });
      expect(requests.prompts).toHaveLength(promptCount);
      expect(requests.agentRequests).toHaveLength(promptCount > 0 ? 1 : 0);
    }
  });

  test("batches standard turns chronologically at the exported cap with source-turn anchors", async () => {
    const messages = Array.from(
      { length: MAX_SUMMARY_BATCH_TURNS + 1 },
      (_, index) => completedTurn(`batch-${index}`, `Request ${index}`),
    ).flat();

    const requests = await runCompaction(session("source"), { messages });
    const prompts = requests.prompts.map(promptText);

    expect(prompts).toHaveLength(2);
    expect(prompts[0]).toContain('<assistant id="batch-0-assistant">');
    expect(prompts[0]).toContain(
      `<assistant id="batch-${MAX_SUMMARY_BATCH_TURNS - 1}-assistant">`,
    );
    expect(prompts[0]).not.toContain(
      `<assistant id="batch-${MAX_SUMMARY_BATCH_TURNS}-assistant">`,
    );
    expect(prompts[0]).toContain(
      `<stop id="batch-${MAX_SUMMARY_BATCH_TURNS}-assistant">`,
    );
    expect(prompts[1]).toContain(
      `<assistant id="batch-${MAX_SUMMARY_BATCH_TURNS}-assistant">`,
    );
    expect(prompts[1]).not.toContain("<stop id=");
    expect(requests.agentRequests).toHaveLength(1);
    expect(requests.updates.map(update => update.sessionID)).toEqual([
      "ephemeral",
      "ephemeral-2",
    ]);
  });

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
      promptResponses: [
        summaryResponse("ordinary-before-assistant", "Before summary."),
        summaryResponse("ordinary-after-assistant", "After summary."),
      ],
    });
    const prompts = requests.prompts.map(promptText);

    expect(prompts).toHaveLength(2);
    expect(requests.updates).toHaveLength(2);
    expect(requests.agentRequests).toHaveLength(1);
    expect(prompts[0]).toContain('<assistant id="ordinary-before-assistant">');
    expect(prompts[0]).toContain('<stop id="outlier-assistant-0">');
    expect(prompts[1]).toContain('<assistant id="ordinary-after-assistant">');
    expect(prompts.join("\n")).not.toContain(
      'id="outlier-assistant-0">[Replace',
    );
    expect(requests.partUpdates.map(update => update.messageID)).toEqual([
      "ordinary-before-assistant",
      "outlier-assistant-0",
      "ordinary-after-assistant",
    ]);

    const stored = storedSummary(requests, "outlier-assistant-0");
    expect(stored).toStartWith("Outcome: partial\n");
    expect(stored).toContain("Historical assistant text (authoritative):");
    expect(stored).toContain("Delivered implementation in src/critical.ts.");
    expect(stored).toContain("bun test reported 88 pass");
    expect(stored).toContain("commit abc123 was pushed");
    expect(stored).toContain("smoke not run");
    expect(stored).toContain("integration failed");
    expect(stored).toContain("final assistant evidence");
    expect(stored).toContain('"command":"bun test"');
    expect(stored).toContain("Process exited with code 1");
    expect(stored).toContain('"command":"git status --short"');
    expect(stored).toContain("push succeeded");
    expect(stored).toContain('"status":"running"');
    expect(
      stored.split("Historical terminal evidence (authoritative):"),
    ).toHaveLength(2);
    expect(stored).toContain("omitted-record-count=");
    expect(Buffer.byteLength(stored)).toBeLessThanOrEqual(
      DETERMINISTIC_HIGH_RISK_SUMMARY_MAX_BYTES,
    );
    expect(hasUnpairedSurrogate(stored)).toBeFalse();
  });

  test("classifies a failure from the middle of full tool output before bounding it", async () => {
    const marker = "Process exited with code 1";
    const messages = completedCommandTurn(
      'bun test "packages/middle suite.test.ts"',
      `${"a".repeat(80_000)}\n${marker}\n${"b".repeat(80_000)}`,
    );
    const requests = await runCompaction(session("source"), { messages });
    const stored = storedSummary(requests, "metric-assistant-0");

    expect(requests.prompts).toHaveLength(0);
    expect(stored).toStartWith("Outcome: partial\n");
    expect(stored).toContain(marker);
    expect(stored).toContain('bun test \\"packages/middle suite.test.ts\\"');
    expect(
      labeledSummarySection(
        stored,
        "Unresolved",
        "Historical terminal evidence (authoritative)",
      ),
    ).toContain(marker);
  });

  test("excludes synthetic, ignored, Magic summary, empty, and placeholder assistant text", async () => {
    const turn = metricTurn(1, HIGH_RISK_MESSAGE_COUNT, 0);
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
    const requests = await runCompaction(session("source"), {
      messages: [...turn.user, ...turn.assistants],
    });
    const stored = storedSummary(requests, "metric-assistant-0");

    expect(stored).toStartWith("Outcome: analysis-only\n");
    expect(stored).toContain("Authoritative final response");
    expect(stored).not.toContain("ignored evidence");
    expect(stored).not.toContain("synthetic evidence");
    expect(stored).not.toContain("old Magic summary");
    expect(stored).not.toContain("[TODO]");
  });

  test("reserves twenty pending records and a late failure with exact anchors under the total UTF-8 cap", async () => {
    const turn = metricTurn(1, HIGH_RISK_MESSAGE_COUNT, 0);
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
    const terminal = labeledSummarySection(
      stored,
      "Historical terminal evidence (authoritative)",
    );
    const unresolved = labeledSummarySection(
      stored,
      "Unresolved",
      "Historical terminal evidence (authoritative)",
    );

    expect(Buffer.byteLength(stored)).toBeLessThanOrEqual(
      DETERMINISTIC_HIGH_RISK_SUMMARY_MAX_BYTES,
    );
    expect(hasUnpairedSurrogate(stored)).toBeFalse();
    expect(stored).toContain('bun test \\"packages/late suite.test.ts\\"');
    expect(stored).toContain("Process exited with code 1");
    expect(stored).toContain('\\"/workspace/project/late failure.ts\\"');
    expect(stored).toContain("./src/relative.ts");
    for (const path of latePaths) expect(unresolved).toContain(path);
    expect(stored).toContain("../shared/check.ts");
    expect(stored).toContain("were not run");
    expect(unresolved).toContain('"status":"pending"');
    expect(unresolved).toContain('"categories":["verification","incomplete"]');
    expect(unresolved).toContain('"polarity":"incomplete"');
    expect(unresolved).toContain('"polarity":"failed"');
    expect(unresolved).toContain('"state":{"input":');
    for (let index = 0; index < 20; index++) {
      expect(unresolved).toContain(`/workspace/pending-${index}.json`);
    }
    expect(omittedRecordCount(unresolved)).toBe(0);
    expect(omittedRecordCount(terminal)).toBe(
      82 - terminal.split("\n").filter(line => line.startsWith("{")).length,
    );
    expect(omittedRecordCount(terminal)).toBeGreaterThan(0);
  });

  test("fails closed rather than reporting complete inclusion for one hundred required not-run records", async () => {
    const turn = metricTurn(1, HIGH_RISK_MESSAGE_COUNT, 0);
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
    let error: unknown;

    try {
      await runCompaction(session("source"), {
        messages: [...turn.user, ...turn.assistants],
        partUpdates,
      });
    } catch (caught) {
      error = caught;
    }

    expect(error).toMatchObject({
      name: "HighRiskEvidenceOverflowError",
      message: expect.stringContaining("64 KiB"),
    });
    expect(error).toBeInstanceOf(HighRiskEvidenceOverflowError);
    expect(partUpdates).toEqual([]);
  });

  test("aborts one hundred twenty required final response parts before source summary or prune writes", async () => {
    const turn = metricTurn(1, HIGH_RISK_MESSAGE_COUNT, 0);
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
    let error: unknown;

    try {
      await executeMagicCompact(command.v2, "source", 0);
    } catch (caught) {
      error = caught;
    }

    expect(error).toMatchObject({
      name: "HighRiskEvidenceOverflowError",
      message: expect.stringContaining("64 KiB"),
    });
    expect(error).toBeInstanceOf(HighRiskEvidenceOverflowError);
    expect(command.partUpdates).toEqual([]);
    expect(command.partDeletes).toEqual([]);
    expect(command.events).not.toContain("summary-update");
    expect(command.events).not.toContain("boundary-update");
    expect(
      command.sessionUpdates.filter(request => request.sessionID === "source"),
    ).toEqual([]);
    expect(command.sourcePrompts).toEqual([]);
    expect(command.messageDeletes).toEqual([]);
    expect(command.selectedSessions).toEqual(["backup"]);
    expect(command.sessionDeletes).toEqual(["source"]);
  });

  test("preserves quoted, spaced, absolute, and relative command paths", async () => {
    const turn = metricTurn(1, HIGH_RISK_MESSAGE_COUNT, 0);
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
    const requests = await runCompaction(session("source"), {
      messages: [...turn.user, ...turn.assistants],
    });
    const stored = storedSummary(requests, "metric-assistant-0");

    expect(stored).toStartWith("Outcome: completed\n");
    for (const command of commands) {
      expect(stored).toContain(JSON.stringify(command).slice(1, -1));
    }
    for (const path of [
      "/workspace/service/Cargo.toml",
      "./src/file with spaces.ts",
      "../shared/config.ts",
      "~/project/readme.md",
    ]) {
      expect(stored).toContain(path);
    }
  });

  test("distinguishes zero failures from checks that did not run", async () => {
    const successful = await runCompaction(session("source"), {
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

    expect(successfulSummary).toStartWith("Outcome: completed\n");
    expect(successfulSummary).toContain("10 passed, 0 failed");
    expect(successfulSummary).toContain("Unresolved: None evidenced.");
    expect(notRunSummary).toStartWith("Outcome: partial\n");
    expect(notRunSummary).toContain("No tests ran");
    expect(
      labeledSummarySection(
        notRunSummary,
        "Unresolved",
        "Historical terminal evidence (authoritative)",
      ),
    ).toContain("tests were not run");
  });

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
      const turn = metricTurn(1, HIGH_RISK_MESSAGE_COUNT, 0);
      const finalAssistant = turn.assistants.at(-1)!;
      finalAssistant.parts.push(
        textPart("prose", "source", finalAssistant.info.id, text),
      );

      const requests = await runCompaction(session("source"), {
        messages: [...turn.user, ...turn.assistants],
      });
      const stored = storedSummary(requests, "metric-assistant-0");

      expect(stored).toStartWith("Outcome: analysis-only\n");
      expect(stored).toContain("Verification: None evidenced.");
      expect(stored).toContain("VCS/Deployment: None evidenced.");
      expect(stored).toContain("Unresolved: None evidenced.");
    },
  );

  test.each([
    ["tests passed", "completed", "Verification"],
    ["test not executed", "partial", "Unresolved"],
    ["work remains: run smoke test", "partial", "Unresolved"],
    ["Tests couldn't be run", "partial", "Unresolved"],
    ["Tests could not be performed", "partial", "Unresolved"],
    ["Tests cannot be run", "partial", "Unresolved"],
    ["Tests can't be run", "partial", "Unresolved"],
    ["tests were skipped", "partial", "Unresolved"],
    ["no tests have run", "partial", "Unresolved"],
  ] as const)(
    "preserves contextual assistant execution evidence: %s",
    async (text, outcome, sectionLabel) => {
      const turn = metricTurn(1, HIGH_RISK_MESSAGE_COUNT, 0);
      const finalAssistant = turn.assistants.at(-1)!;
      finalAssistant.parts.push(
        textPart("prose", "source", finalAssistant.info.id, text),
      );

      const requests = await runCompaction(session("source"), {
        messages: [...turn.user, ...turn.assistants],
      });
      const stored = storedSummary(requests, "metric-assistant-0");

      expect(stored).toStartWith(`Outcome: ${outcome}\n`);
      expect(
        labeledSummarySection(
          stored,
          sectionLabel,
          sectionLabel === "Verification"
            ? "VCS/Deployment"
            : "Historical terminal evidence (authoritative)",
        ),
      ).toContain(text);
    },
  );

  test("round-trips exact assistant source code units through JSON records", async () => {
    const sourceText =
      "exact\u0000nul\u000bvt\ud800lone and trailing\udfff units";
    const turn = metricTurn(1, HIGH_RISK_MESSAGE_COUNT, 0);
    const finalAssistant = turn.assistants.at(-1)!;
    finalAssistant.parts.push(
      textPart("units", "source", finalAssistant.info.id, sourceText),
    );

    const requests = await runCompaction(session("source"), {
      messages: [...turn.user, ...turn.assistants],
    });
    const stored = storedSummary(requests, "metric-assistant-0");
    const section = labeledSummarySection(
      stored,
      "Historical assistant text (authoritative)",
      "Verification",
    );
    const record = section
      .split("\n")
      .map(line => line.trim())
      .filter(line => line.startsWith("{"))
      .map(line => JSON.parse(line) as { text?: string; windows?: string[] })
      .find(
        entry =>
          entry.text === sourceText
          || entry.windows?.join("").includes("exact"),
      );

    expect(record).toBeDefined();
    expect(record!.text).toBe(sourceText);
    expect(stored).toContain("\\u0000");
    expect(stored).toContain("\\ud800");
    expect(Buffer.byteLength(stored)).toBeLessThanOrEqual(
      DETERMINISTIC_HIGH_RISK_SUMMARY_MAX_BYTES,
    );
  });

  test("keeps substantive bracket arrays and TODO prose while dropping whole-body placeholders", async () => {
    const turn = metricTurn(1, HIGH_RISK_MESSAGE_COUNT, 0);
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

    const requests = await runCompaction(session("source"), {
      messages: [...turn.user, ...turn.assistants],
    });
    const stored = storedSummary(requests, "metric-assistant-0");
    const assistantSection = labeledSummarySection(
      stored,
      "Historical assistant text (authoritative)",
      "Verification",
    );
    const unresolved = labeledSummarySection(
      stored,
      "Unresolved",
      "Historical terminal evidence (authoritative)",
    );

    expect(assistantSection).toContain("['failed','pending']");
    expect(assistantSection).toContain(
      "Implemented parser. TODO: add evidence.",
    );
    expect(assistantSection).not.toContain('"[TODO]"');
    expect(assistantSection).not.toMatch(/"text":"TBD"/);
    expect(assistantSection).not.toMatch(/"text":"N\/A"/);
    expect(stored).toStartWith("Outcome: partial\n");
    expect(unresolved).toContain("TODO: add evidence");
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

      expect(stored).toStartWith("Outcome: partial\n");
      expect(
        labeledSummarySection(
          stored,
          "Unresolved",
          "Historical terminal evidence (authoritative)",
        ),
      ).toContain(output);
    },
  );

  test("round-trips exact tool input/output/error code units through JSON records", async () => {
    const units = "exact\u0000nul\u000bvt\ud800lone\ud801more";
    const turn = metricTurn(1, HIGH_RISK_MESSAGE_COUNT, 0);
    const assistant = turn.assistants[0]!;
    assistant.parts.push(
      genericToolPart("units-io", assistant.info.id, "read", {
        status: "completed",
        input: units,
        output: units,
        error: units,
        title: "Read file",
        metadata: {},
        time: { start: 1, end: 2 },
      }),
    );

    const requests = await runCompaction(session("source"), {
      messages: [...turn.user, ...turn.assistants],
    });
    const stored = storedSummary(requests, "metric-assistant-0");
    const terminal = labeledSummarySection(
      stored,
      "Historical terminal evidence (authoritative)",
    );
    const record = terminal
      .split("\n")
      .map(line => line.trim())
      .filter(line => line.startsWith("{"))
      .map(
        line =>
          JSON.parse(line) as {
            state?: { input?: string; output?: string; error?: string };
          },
      )
      .find(
        entry =>
          entry.state?.input === units
          || entry.state?.output === units
          || entry.state?.error === units,
      );

    expect(record).toBeDefined();
    expect(record!.state?.input).toBe(units);
    expect(record!.state?.output).toBe(units);
    expect(record!.state?.error).toBe(units);
    expect(stored).toContain("\\u0000");
    expect(stored).toContain("\\ud800");
    expect(stored).toContain("\\ud801");
    expect(hasUnpairedSurrogate(stored)).toBeFalse();
  });

  test("escapes XML-invalid U+FFFE/U+FFFF in assistant records for provider-safe round-trip", async () => {
    const sourceText = "marker\ufffe\ufffftail";
    const turn = metricTurn(1, HIGH_RISK_MESSAGE_COUNT, 0);
    const finalAssistant = turn.assistants.at(-1)!;
    finalAssistant.parts.push(
      textPart("xml-invalid", "source", finalAssistant.info.id, sourceText),
    );

    const requests = await runCompaction(session("source"), {
      messages: [...turn.user, ...turn.assistants],
    });
    const stored = storedSummary(requests, "metric-assistant-0");
    const section = labeledSummarySection(
      stored,
      "Historical assistant text (authoritative)",
      "Verification",
    );
    const record = section
      .split("\n")
      .map(line => line.trim())
      .filter(line => line.startsWith("{"))
      .map(line => JSON.parse(line) as { text?: string })
      .find(entry => entry.text === sourceText);

    expect(record).toBeDefined();
    expect(record!.text).toBe(sourceText);
    expect(stored).toContain("\\ufffe");
    expect(stored).toContain("\\uffff");
    expect(stored).not.toContain("\ufffe");
    expect(stored).not.toContain("\uffff");
    expect(hasUnpairedSurrogate(stored)).toBeFalse();
  });

  test("escapes XML-invalid U+FFFE/U+FFFF in tool input/output/error records", async () => {
    const units = "io\ufffe\uffffunits";
    const turn = metricTurn(1, HIGH_RISK_MESSAGE_COUNT, 0);
    const assistant = turn.assistants[0]!;
    assistant.parts.push(
      genericToolPart("xml-invalid-io", assistant.info.id, "read", {
        status: "completed",
        input: units,
        output: units,
        error: units,
        title: "Read file",
        metadata: {},
        time: { start: 1, end: 2 },
      }),
    );

    const requests = await runCompaction(session("source"), {
      messages: [...turn.user, ...turn.assistants],
    });
    const stored = storedSummary(requests, "metric-assistant-0");
    const terminal = labeledSummarySection(
      stored,
      "Historical terminal evidence (authoritative)",
    );
    const record = terminal
      .split("\n")
      .map(line => line.trim())
      .filter(line => line.startsWith("{"))
      .map(
        line =>
          JSON.parse(line) as {
            state?: { input?: string; output?: string; error?: string };
          },
      )
      .find(
        entry =>
          entry.state?.input === units
          || entry.state?.output === units
          || entry.state?.error === units,
      );

    expect(record).toBeDefined();
    expect(record!.state?.input).toBe(units);
    expect(record!.state?.output).toBe(units);
    expect(record!.state?.error).toBe(units);
    expect(stored).toContain("\\ufffe");
    expect(stored).toContain("\\uffff");
    expect(stored).not.toContain("\ufffe");
    expect(stored).not.toContain("\uffff");
  });

  test("positive independent pass overrides earlier no-run in assistant prose", async () => {
    const text = "No tests ran. 10 tests passed.";
    const turn = metricTurn(1, HIGH_RISK_MESSAGE_COUNT, 0);
    const finalAssistant = turn.assistants.at(-1)!;
    finalAssistant.parts.push(
      textPart("positive-override", "source", finalAssistant.info.id, text),
    );

    const requests = await runCompaction(session("source"), {
      messages: [...turn.user, ...turn.assistants],
    });
    const stored = storedSummary(requests, "metric-assistant-0");
    const verification = labeledSummarySection(
      stored,
      "Verification",
      "VCS/Deployment",
    );

    expect(stored).toStartWith("Outcome: completed\n");
    expect(verification).toContain("No tests ran");
    expect(verification).toContain("10 tests passed");
    expect(verification).not.toContain('"polarity":"not-run"');
    expect(stored).toContain("Unresolved: None evidenced.");
  });

  test("positive independent pass overrides earlier no-run in real command tool output", async () => {
    const output = "No tests ran. 10 tests passed.";
    const requests = await runCompaction(session("source"), {
      messages: completedCommandTurn("bun test", output),
    });
    const stored = storedSummary(requests, "metric-assistant-0");

    expect(stored).toStartWith("Outcome: completed\n");
    expect(
      labeledSummarySection(stored, "Verification", "VCS/Deployment"),
    ).toContain(output);
    expect(stored).toContain("Unresolved: None evidenced.");
  });

  test("pure no-run with zero failures remains not-run", async () => {
    const output = "No tests ran; 0 failed";
    const requests = await runCompaction(session("source"), {
      messages: completedCommandTurn("bun test", output),
    });
    const stored = storedSummary(requests, "metric-assistant-0");

    expect(stored).toStartWith("Outcome: partial\n");
    expect(
      labeledSummarySection(
        stored,
        "Unresolved",
        "Historical terminal evidence (authoritative)",
      ),
    ).toContain("No tests ran");
  });

  test.each([
    ["Read unit.test.ts", "10 passed, 0 failed", "verification"],
    ["Read build configuration", "Process exited with code 1", "verification"],
    ["Read deployment config", "not deployed", "vcs"],
  ] as const)(
    "ignores passive read title context: %s",
    async (title, output, kind) => {
      const turn = metricTurn(1, HIGH_RISK_MESSAGE_COUNT, 0);
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

      const requests = await runCompaction(session("source"), {
        messages: [...turn.user, ...turn.assistants],
      });
      const stored = storedSummary(requests, "metric-assistant-0");

      expect(stored).toStartWith("Outcome: completed\n");
      expect(stored).toContain("Verification: None evidenced.");
      expect(stored).toContain("VCS/Deployment: None evidenced.");
      expect(stored).toContain("Unresolved: None evidenced.");
      expect(stored).toContain(output);
      if (kind === "verification") {
        expect(
          labeledSummarySection(stored, "Verification", "VCS/Deployment"),
        ).toBe("Verification: None evidenced.");
      } else {
        expect(
          labeledSummarySection(stored, "VCS/Deployment", "Unresolved"),
        ).toBe("VCS/Deployment: None evidenced.");
      }
    },
  );

  test("treats passive read failures, pending prose, and not-deployed text as neutral", async () => {
    const turn = metricTurn(1, HIGH_RISK_MESSAGE_COUNT, 0);
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

    const requests = await runCompaction(session("source"), {
      messages: [...turn.user, ...turn.assistants],
    });
    const stored = storedSummary(requests, "metric-assistant-0");

    expect(stored).toStartWith("Outcome: completed\n");
    expect(stored).toContain("Verification: None evidenced.");
    expect(stored).toContain("VCS/Deployment: None evidenced.");
    expect(stored).toContain("Unresolved: None evidenced.");
  });

  test("accepts real bash, git, and test tool identity or command context", async () => {
    const turn = metricTurn(1, HIGH_RISK_MESSAGE_COUNT, 0);
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

    const requests = await runCompaction(session("source"), {
      messages: [...turn.user, ...turn.assistants],
    });
    const stored = storedSummary(requests, "metric-assistant-0");
    const verification = labeledSummarySection(
      stored,
      "Verification",
      "VCS/Deployment",
    );
    const vcs = labeledSummarySection(stored, "VCS/Deployment", "Unresolved");

    expect(stored).toStartWith("Outcome: completed\n");
    expect(verification).toContain("12 passed, 0 failed");
    expect(verification).toContain("10 passed");
    expect(vcs).toContain("working tree clean");
    expect(stored).toContain("Unresolved: None evidenced.");
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
    async (toolName, output, kind) => {
      const turn = metricTurn(1, HIGH_RISK_MESSAGE_COUNT, 0);
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

      const requests = await runCompaction(session("source"), {
        messages: [...turn.user, ...turn.assistants],
      });
      const stored = storedSummary(requests, "metric-assistant-0");

      expect(stored).toStartWith("Outcome: completed\n");
      expect(stored).toContain("Verification: None evidenced.");
      expect(stored).toContain("VCS/Deployment: None evidenced.");
      expect(stored).toContain("Unresolved: None evidenced.");
      expect(stored).toContain(output);
      if (kind === "verification") {
        expect(
          labeledSummarySection(stored, "Verification", "VCS/Deployment"),
        ).toBe("Verification: None evidenced.");
      } else {
        expect(
          labeledSummarySection(stored, "VCS/Deployment", "Unresolved"),
        ).toBe("VCS/Deployment: None evidenced.");
      }
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
      const verification = labeledSummarySection(
        stored,
        "Verification",
        "VCS/Deployment",
      );
      const unresolved = labeledSummarySection(
        stored,
        "Unresolved",
        "Historical terminal evidence (authoritative)",
      );

      expect(stored).toStartWith("Outcome: partial\n");
      expect(verification).toContain(command);
      expect(verification).toContain(output);
      expect(unresolved).toContain(command);
      expect(unresolved).toContain(output);
      expect(unresolved).toContain('"polarity":"failed"');
    },
  );

  test.each([
    ["make test", "10 passed, 0 failed"],
    ["just test", "all tests passed"],
    ["make lint", "0 failed"],
  ] as const)(
    "treats runner verification command %s success as verification evidence",
    async (command, output) => {
      const requests = await runCompaction(session("source"), {
        messages: completedCommandTurn(command, output),
      });
      const stored = storedSummary(requests, "metric-assistant-0");
      const verification = labeledSummarySection(
        stored,
        "Verification",
        "VCS/Deployment",
      );

      expect(stored).toStartWith("Outcome: completed\n");
      expect(verification).toContain(command);
      expect(verification).toContain(output);
      expect(stored).toContain("Unresolved: None evidenced.");
    },
  );

  test("does not treat arbitrary prose mentioning make/test as a verification command", async () => {
    const turn = metricTurn(1, HIGH_RISK_MESSAGE_COUNT, 0);
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

    const requests = await runCompaction(session("source"), {
      messages: [...turn.user, ...turn.assistants],
    });
    const stored = storedSummary(requests, "metric-assistant-0");

    expect(stored).toStartWith("Outcome: completed\n");
    expect(stored).toContain("Verification: None evidenced.");
    expect(stored).toContain("Unresolved: None evidenced.");
  });

  test.each([
    "No unresolved issues",
    "The value was passed to serializer",
    "configuration remains unchanged",
    "pending color",
    "fail\ud800ed",
  ])("treats completed arbitrary tool output as neutral: %s", async output => {
    const turn = metricTurn(1, HIGH_RISK_MESSAGE_COUNT, 0);
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

    const requests = await runCompaction(session("source"), {
      messages: [...turn.user, ...turn.assistants],
    });
    const stored = storedSummary(requests, "metric-assistant-0");

    expect(stored).toStartWith("Outcome: completed\n");
    expect(stored).toContain("Verification: None evidenced.");
    expect(stored).toContain("Unresolved: None evidenced.");
    expect(stored).toContain(JSON.stringify(output).slice(1, -1));
  });

  test("does not overflow required budget on hundreds of neutral completed tool records", async () => {
    const turn = metricTurn(1, HIGH_RISK_MESSAGE_COUNT, 0);
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

    const requests = await runCompaction(session("source"), {
      messages: [...turn.user, ...turn.assistants],
    });
    const stored = storedSummary(requests, "metric-assistant-0");

    expect(stored).toStartWith("Outcome: completed\n");
    expect(stored).toContain("Verification: None evidenced.");
    expect(stored).toContain("VCS/Deployment: None evidenced.");
    expect(stored).toContain("Unresolved: None evidenced.");
    expect(Buffer.byteLength(stored)).toBeLessThanOrEqual(
      DETERMINISTIC_HIGH_RISK_SUMMARY_MAX_BYTES,
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
      const turn = metricTurn(1, HIGH_RISK_MESSAGE_COUNT, 0);
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
      const unresolved = labeledSummarySection(
        stored,
        "Unresolved",
        "Historical terminal evidence (authoritative)",
      );

      expect(stored).toStartWith("Outcome: partial\n");
      expect(unresolved).toContain(command);
      expect(unresolved).toContain(failure);
      expect(unresolved).toContain('"polarity":"failed"');
    },
  );

  test("keeps passive completed tools neutral when output only looks like failure prose", async () => {
    const turn = metricTurn(1, HIGH_RISK_MESSAGE_COUNT, 0);
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

    const requests = await runCompaction(session("source"), {
      messages: [...turn.user, ...turn.assistants],
    });
    const stored = storedSummary(requests, "metric-assistant-0");

    expect(stored).toStartWith("Outcome: completed\n");
    expect(stored).toContain("Unresolved: None evidenced.");
  });

  test("keeps late generic failed bash authoritative under saturated optional budget", async () => {
    const turn = metricTurn(1, HIGH_RISK_MESSAGE_COUNT, 0);
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

    const requests = await runCompaction(session("source"), {
      messages: [...turn.user, ...turn.assistants],
    });
    const stored = storedSummary(requests, "metric-assistant-0");
    const unresolved = labeledSummarySection(
      stored,
      "Unresolved",
      "Historical terminal evidence (authoritative)",
    );
    const record = unresolved
      .split("\n")
      .map(line => line.trim())
      .filter(line => line.startsWith("{"))
      .map(
        line =>
          JSON.parse(line) as {
            command?: string;
            polarity?: string;
            state?: { output?: unknown };
          },
      )
      .find(entry => entry.command === command);

    expect(stored).toStartWith("Outcome: partial\n");
    expect(record).toBeDefined();
    expect(record!.command).toBe(command);
    expect(record!.polarity).toBe("failed");
    expect(JSON.stringify(record!.state)).toContain(failure);
    expect(Buffer.byteLength(stored)).toBeLessThanOrEqual(
      DETERMINISTIC_HIGH_RISK_SUMMARY_MAX_BYTES,
    );
  });

  test("preserves real bash bun test polarity and failed tool state", async () => {
    const turn = metricTurn(1, HIGH_RISK_MESSAGE_COUNT, 0);
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

    const requests = await runCompaction(session("source"), {
      messages: [...turn.user, ...turn.assistants],
    });
    const stored = storedSummary(requests, "metric-assistant-0");
    const verification = labeledSummarySection(
      stored,
      "Verification",
      "VCS/Deployment",
    );
    const unresolved = labeledSummarySection(
      stored,
      "Unresolved",
      "Historical terminal evidence (authoritative)",
    );

    expect(stored).toStartWith("Outcome: partial\n");
    expect(verification).toContain("10 passed, 0 failed");
    expect(verification).toContain("Process exited with code 1");
    expect(unresolved).toContain("Process exited with code 1");
    expect(unresolved).toContain("tests were not run");
    expect(unresolved).toContain('"status":"error"');
    expect(unresolved).toContain("No unresolved issues");
  });

  test("categorizes mutation tools by identity rather than output prose", async () => {
    const turn = metricTurn(1, HIGH_RISK_MESSAGE_COUNT, 0);
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

    const requests = await runCompaction(session("source"), {
      messages: [...turn.user, ...turn.assistants],
    });
    const stored = storedSummary(requests, "metric-assistant-0");
    const terminal = labeledSummarySection(
      stored,
      "Historical terminal evidence (authoritative)",
    );
    const records = terminal
      .split("\n")
      .map(line => line.trim())
      .filter(line => line.startsWith("{"))
      .map(line => JSON.parse(line) as { kind?: string; tool?: string });

    expect(
      records.some(r => r.kind === "tool-mutation" && r.tool === "write"),
    ).toBe(true);
    expect(
      records.some(r => r.tool === "read" && r.kind === "tool-mutation"),
    ).toBe(false);
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
      const turn = metricTurn(1, HIGH_RISK_MESSAGE_COUNT, 0);
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

      const requests = await runCompaction(session("source"), {
        messages: [...turn.user, ...turn.assistants],
      });
      const stored = storedSummary(requests, "metric-assistant-0");

      expect(stored).toStartWith("Outcome: completed\n");
      if (kind === "verification") {
        expect(
          labeledSummarySection(stored, "Verification", "VCS/Deployment"),
        ).toContain(output);
        expect(stored).toContain("VCS/Deployment: None evidenced.");
      } else {
        expect(
          labeledSummarySection(stored, "VCS/Deployment", "Unresolved"),
        ).toContain(output);
        expect(stored).toContain("Verification: None evidenced.");
      }
      expect(stored).toContain("Unresolved: None evidenced.");
    },
  );

  test("keeps passive read identities neutral after tool-name normalization", async () => {
    const turn = metricTurn(1, HIGH_RISK_MESSAGE_COUNT, 0);
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

    const requests = await runCompaction(session("source"), {
      messages: [...turn.user, ...turn.assistants],
    });
    const stored = storedSummary(requests, "metric-assistant-0");

    expect(stored).toStartWith("Outcome: completed\n");
    expect(stored).toContain("Verification: None evidenced.");
    expect(stored).toContain("VCS/Deployment: None evidenced.");
    expect(stored).toContain("Unresolved: None evidenced.");
  });

  test.each([
    ["Tests ran. They failed.", "failed"],
    ["Tests could not run. Database unavailable.", "not-run"],
  ] as const)(
    "classifies full assistant text before segmentation: %s",
    async (text, polarity) => {
      const turn = metricTurn(1, HIGH_RISK_MESSAGE_COUNT, 0);
      const finalAssistant = turn.assistants.at(-1)!;
      finalAssistant.parts.push(
        textPart("cross-sentence", "source", finalAssistant.info.id, text),
      );

      const requests = await runCompaction(session("source"), {
        messages: [...turn.user, ...turn.assistants],
      });
      const stored = storedSummary(requests, "metric-assistant-0");
      const unresolved = labeledSummarySection(
        stored,
        "Unresolved",
        "Historical terminal evidence (authoritative)",
      );

      expect(stored).toStartWith("Outcome: partial\n");
      expect(unresolved).toContain(text);
      expect(unresolved).toContain(`"polarity":"${polarity}"`);
    },
  );

  test("preserves positive-run precedence across segmented assistant sentences", async () => {
    const text = "No tests ran. 10 tests passed.";
    const turn = metricTurn(1, HIGH_RISK_MESSAGE_COUNT, 0);
    const finalAssistant = turn.assistants.at(-1)!;
    finalAssistant.parts.push(
      textPart("positive-full", "source", finalAssistant.info.id, text),
    );

    const requests = await runCompaction(session("source"), {
      messages: [...turn.user, ...turn.assistants],
    });
    const stored = storedSummary(requests, "metric-assistant-0");

    expect(stored).toStartWith("Outcome: completed\n");
    expect(stored).toContain("Unresolved: None evidenced.");
    expect(
      labeledSummarySection(stored, "Verification", "VCS/Deployment"),
    ).toContain(text);
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
      const turn = metricTurn(1, HIGH_RISK_MESSAGE_COUNT, 0);
      const finalAssistant = turn.assistants.at(-1)!;
      finalAssistant.parts.push(
        textPart("resolved-pending", "source", finalAssistant.info.id, text),
      );

      const requests = await runCompaction(session("source"), {
        messages: [...turn.user, ...turn.assistants],
      });
      const stored = storedSummary(requests, "metric-assistant-0");

      expect(stored).toStartWith("Outcome: analysis-only\n");
      expect(stored).toContain("Unresolved: None evidenced.");
    },
  );

  test.each(["tasks are pending", "work remains: run smoke"])(
    "preserves real incomplete assistant evidence: %s",
    async text => {
      const turn = metricTurn(1, HIGH_RISK_MESSAGE_COUNT, 0);
      const finalAssistant = turn.assistants.at(-1)!;
      finalAssistant.parts.push(
        textPart("real-pending", "source", finalAssistant.info.id, text),
      );

      const requests = await runCompaction(session("source"), {
        messages: [...turn.user, ...turn.assistants],
      });
      const stored = storedSummary(requests, "metric-assistant-0");

      expect(stored).toStartWith("Outcome: partial\n");
      expect(
        labeledSummarySection(
          stored,
          "Unresolved",
          "Historical terminal evidence (authoritative)",
        ),
      ).toContain(text);
    },
  );

  test("keeps command and paths on dedicated fields without duplicating them as anchors", async () => {
    const command = 'bun test "/workspace/service/exact suite.test.ts"';
    const requests = await runCompaction(session("source"), {
      messages: completedCommandTurn(command, "Process exited with code 1"),
    });
    const stored = storedSummary(requests, "metric-assistant-0");
    const unresolved = labeledSummarySection(
      stored,
      "Unresolved",
      "Historical terminal evidence (authoritative)",
    );
    const record = unresolved
      .split("\n")
      .map(line => line.trim())
      .filter(line => line.startsWith("{"))
      .map(
        line =>
          JSON.parse(line) as {
            command?: string;
            paths?: string[];
            anchors?: string[];
          },
      )
      .find(entry => entry.command === command);

    expect(record).toBeDefined();
    expect(record!.command).toBe(command);
    expect(record!.paths).toContain('"/workspace/service/exact suite.test.ts"');
    expect(record!.anchors ?? []).not.toContain(command);
    expect(record!.anchors ?? []).not.toContain(
      '"/workspace/service/exact suite.test.ts"',
    );
    expect(record!.anchors ?? []).toContain("Process exited with code 1");
  });

  test("extracts title paths into dedicated path fields without promoting title labels to context", async () => {
    const turn = metricTurn(1, HIGH_RISK_MESSAGE_COUNT, 0);
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

    const requests = await runCompaction(session("source"), {
      messages: [...turn.user, ...turn.assistants],
    });
    const stored = storedSummary(requests, "metric-assistant-0");
    const verification = labeledSummarySection(
      stored,
      "Verification",
      "VCS/Deployment",
    );
    const record = verification
      .split("\n")
      .map(line => line.trim())
      .filter(line => line.startsWith("{"))
      .map(
        line =>
          JSON.parse(line) as {
            command?: string;
            paths?: string[];
            anchors?: string[];
          },
      )
      .find(entry => entry.command === "bun test");

    expect(record).toBeDefined();
    expect(record!.paths).toContain("/workspace/title-only/path.test.ts");
    expect(record!.anchors ?? []).not.toContain(
      "/workspace/title-only/path.test.ts",
    );
  });

  test("high-risk path still classifies normalized identities and resolved pending at cutover", async () => {
    const turn = metricTurn(1, HIGH_RISK_MESSAGE_COUNT, 0);
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
    expect(stored).toStartWith("Outcome: partial\n");
    expect(
      labeledSummarySection(stored, "Verification", "VCS/Deployment"),
    ).toContain("10 passed, 0 failed");
    expect(
      labeledSummarySection(stored, "VCS/Deployment", "Unresolved"),
    ).toContain("working tree clean");
    expect(
      labeledSummarySection(
        stored,
        "Unresolved",
        "Historical terminal evidence (authoritative)",
      ),
    ).toContain("Tests ran. They failed.");
    expect(stored).not.toMatch(
      /"polarity":"incomplete"[^}]*no tasks are pending/,
    );
  });

  test("accepts an unchanged SDK-remapped native checkpoint in an ephemeral fork", async () => {
    const source = nativeCheckpointFixture({ tailStartID: "tail-assistant" });
    const ephemeral = remapForkMessages(source, "ephemeral");

    const requests = await runCompaction(session("source"), {
      messages: source,
      ephemeralMessages: ephemeral,
    });

    expect(requests.prompts).toHaveLength(1);
    expect(requests.partUpdates).toHaveLength(2);
  });

  test("aborts when a prompt creates a native checkpoint in its ephemeral fork", async () => {
    const inherited = compactableMessages();
    const ephemeral = remapForkMessages(inherited, "ephemeral");
    const changed = [
      ...ephemeral,
      nativeMarker("ephemeral-marker", null, false, "ephemeral"),
    ];
    const lifecycle: string[] = [];

    let error: unknown;
    try {
      await runCompaction(session("source"), {
        messages: inherited,
        ephemeralMessages: ephemeral,
        ephemeralMessagesAfterPrompt: changed,
        lifecycle,
      });
    } catch (caught) {
      error = caught;
    }

    expect(error).toMatchObject({ name: "EphemeralCheckpointChangedError" });
    expect(error).not.toBeInstanceOf(NativeCheckpointChangedError);
    expect(lifecycle).toContain("prompt");
  });

  test("aborts when a prompt changes an inherited native checkpoint in its ephemeral fork", async () => {
    const source = nativeCheckpointFixture({ tailStartID: "tail-assistant" });
    const ephemeral = remapForkMessages(source, "ephemeral");
    const changed = structuredClone(ephemeral);
    Object.assign(
      changed.find(item => item.info.id.endsWith("native-marker"))!.parts[0]!,
      { auto: false },
    );
    const lifecycle: string[] = [];

    let error: unknown;
    try {
      await runCompaction(session("source"), {
        messages: source,
        ephemeralMessages: ephemeral,
        ephemeralMessagesAfterPrompt: changed,
        lifecycle,
      });
    } catch (caught) {
      error = caught;
    }

    expect(error).toMatchObject({ name: "EphemeralCheckpointChangedError" });
    expect(error).not.toBeInstanceOf(NativeCheckpointChangedError);
    expect(lifecycle).toContain("prompt");
  });

  test("retains an in-band prompt SDK error with an incomplete ephemeral checkpoint", async () => {
    const inherited = compactableMessages();
    const changed = [
      ...inherited,
      nativeMarker("ephemeral-incomplete", null, false, "ephemeral"),
    ];
    let error: unknown;

    try {
      await runCompaction(session("source"), {
        messages: inherited,
        ephemeralMessagesAfterPrompt: changed,
        promptResponseError: "prompt SDK response failed",
      });
    } catch (caught) {
      error = caught;
    }

    expect(error).toBeInstanceOf(AggregateError);
    expect(error).not.toBeInstanceOf(NativeCheckpointChangedError);
    expect((error as AggregateError).errors).toContainEqual(
      expect.objectContaining({ name: "EphemeralCheckpointChangedError" }),
    );
    expect(nestedErrorMessages(error)).toEqual(
      expect.arrayContaining([
        "Native compaction marker ephemeral-incomplete is incomplete or errored.",
        "prompt SDK response failed",
      ]),
    );
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
        partID: "suffix-1-old-text",
      },
    ]);
    expect(command.messageDeletes).toEqual([
      { sessionID: "source", messageID: "progress" },
    ]);
    expect(command.sessionDeletes).toEqual(["ephemeral"]);
    expect(command.selectedSessions).toEqual([]);

    const progressPromptIndex = command.events.indexOf("progress-prompt");
    const summaryPromptIndex = command.events.indexOf("summary-prompt");
    const summaryUpdateIndex = command.events.indexOf("summary-update");
    // plan + provider-token load + pre-progress checkpoint revalidation
    const preProgressMessageLoads = command.events
      .slice(0, progressPromptIndex)
      .filter(event => event === "messages").length;
    const initialRevalidationIndex = command.events.lastIndexOf(
      "messages",
      progressPromptIndex - 1,
    );
    const postSummaryRevalidationIndex = command.events.indexOf(
      "messages",
      summaryPromptIndex + 1,
    );
    expect(progressPromptIndex).toBeGreaterThanOrEqual(0);
    expect(summaryPromptIndex).toBeGreaterThan(progressPromptIndex);
    expect(preProgressMessageLoads).toBe(3);
    expect(initialRevalidationIndex).toBe(progressPromptIndex - 1);
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
    expect(command.sessionDeletes).toEqual(["ephemeral"]);
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
      expect(() => assertNativeCheckpointUnchanged(changed, plan)).toThrow(
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
      assertNativeCheckpointUnchanged(reordered, plan),
    ).not.toThrow();
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
    expect(command.messageDeletes).toEqual([
      { sessionID: "source", messageID: "progress" },
    ]);
    expect(command.sessionDeletes).toEqual(["ephemeral"]);
    expect(command.selectedSessions).toEqual([]);
    expect(
      command.sessionUpdates.filter(
        request =>
          request.sessionID === "backup" && request.title === "Test session",
      ),
    ).toEqual([]);
    expect(command.sourcePrompts).toHaveLength(1);
    expect(command.sourcePrompts[0]).toMatchObject({
      noReply: true,
      parts: [
        {
          metadata: { magicCompact: { progress: true } },
        },
      ],
    });
    expect(command.toasts.at(-1)).toMatchObject({
      title: "Magic Compact Failed",
      variant: "error",
    });
  });

  test("retains an incomplete-marker race and progress-delete failure while preserving both sessions", async () => {
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
    expect((error as Error).cause).toBeInstanceOf(AggregateError);
    expect(nestedErrorMessages((error as Error).cause)).toEqual(
      expect.arrayContaining([
        "Native compaction marker raced-marker is incomplete or errored.",
        "progress cleanup failed",
      ]),
    );
    expect(command.partUpdates).toEqual([]);
    expect(command.partDeletes).toEqual([]);
    expect(command.messageDeletes).toEqual([
      { sessionID: "source", messageID: "progress" },
    ]);
    expect(command.sessionDeletes).toEqual(["ephemeral"]);
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

  test("retains an incomplete-marker race and ephemeral-delete failure while preserving both sessions", async () => {
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

    expect(error).toBeInstanceOf(NativeCheckpointChangedError);
    expect(error).not.toBe(ephemeralDeleteError);
    expect((error as Error).cause).toBeInstanceOf(AggregateError);
    expect(nestedErrorMessages((error as Error).cause)).toEqual(
      expect.arrayContaining([
        "Native compaction marker raced-marker is incomplete or errored.",
        "ephemeral delete failed",
      ]),
    );
    expect(command.partUpdates).toEqual([]);
    expect(command.partDeletes).toEqual([]);
    expect(command.messageDeletes).toEqual([
      { sessionID: "source", messageID: "progress" },
    ]);
    expect(command.sessionDeletes).toEqual(["ephemeral"]);
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
    expect(command.messageDeletes).toEqual([
      { sessionID: "source", messageID: "progress" },
    ]);
    expect(command.sessionDeletes).toEqual(["ephemeral"]);
    expect(command.selectedSessions).toEqual([]);
    expect(
      command.sessionUpdates.filter(
        request =>
          request.sessionID === "backup" && request.title === "Test session",
      ),
    ).toEqual([]);
  });

  test("retains generation and revalidation messages-fetch SDK errors while preserving both sessions", async () => {
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

    expect(error).toBeInstanceOf(NativeCheckpointChangedError);
    expect((error as Error).cause).toBeInstanceOf(AggregateError);
    expect(nestedErrorMessages((error as Error).cause)).toEqual(
      expect.arrayContaining([
        "summary prompt failed",
        "revalidation messages fetch failed",
      ]),
    );
    expect(command.partUpdates).toEqual([]);
    expect(command.partDeletes).toEqual([]);
    expect(command.messageDeletes).toEqual([
      { sessionID: "source", messageID: "progress" },
    ]);
    expect(command.sessionDeletes).toEqual(["ephemeral"]);
    expect(command.selectedSessions).toEqual([]);
    expect(
      command.sessionUpdates.filter(
        request =>
          request.sessionID === "backup" && request.title === "Test session",
      ),
    ).toEqual([]);
  });

  test("retains prompt and progress-delete failures without a race and safely rolls back", async () => {
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

    expect(error).toBeInstanceOf(AggregateError);
    expect(error).not.toBeInstanceOf(NativeCheckpointChangedError);
    expect(nestedErrorMessages(error)).toEqual(
      expect.arrayContaining([
        "summary prompt failed",
        "progress cleanup failed",
      ]),
    );
    expect(command.partUpdates).toEqual([]);
    expect(command.partDeletes).toEqual([]);
    expect(command.messageDeletes).toEqual([
      { sessionID: "source", messageID: "progress" },
    ]);
    expect(command.sessionDeletes).toEqual(["ephemeral", "source"]);
    expect(command.selectedSessions).toEqual(["backup"]);
    expect(
      command.sessionUpdates.filter(
        request =>
          request.sessionID === "backup" && request.title === "Test session",
      ),
    ).toEqual([
      {
        sessionID: "backup",
        title: "Test session",
      },
    ]);
  });

  test("retains prompt and ephemeral-delete failures without a race and safely rolls back", async () => {
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
    expect(nestedErrorMessages(error)).toEqual(
      expect.arrayContaining([
        "summary prompt failed",
        "ephemeral delete failed",
      ]),
    );
    expect(command.partUpdates).toEqual([]);
    expect(command.partDeletes).toEqual([]);
    expect(command.messageDeletes).toEqual([
      { sessionID: "source", messageID: "progress" },
    ]);
    expect(command.sessionDeletes).toEqual(["ephemeral", "source"]);
    expect(command.selectedSessions).toEqual(["backup"]);
    expect(
      command.sessionUpdates.filter(
        request =>
          request.sessionID === "backup" && request.title === "Test session",
      ),
    ).toEqual([
      {
        sessionID: "backup",
        title: "Test session",
      },
    ]);
  });

  test("retains prompt and ephemeral-delete failures with a concurrent checkpoint race while preserving both sessions", async () => {
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

    expect(error).toBeInstanceOf(NativeCheckpointChangedError);
    expect(error).not.toBe(generationError);
    expect(error).not.toBe(ephemeralDeleteError);
    expect((error as Error).cause).toBeInstanceOf(AggregateError);
    expect(nestedErrorMessages((error as Error).cause)).toEqual(
      expect.arrayContaining([
        "Native compaction marker raced-marker is incomplete or errored.",
        "summary prompt failed",
        "ephemeral delete failed",
      ]),
    );
    expect(command.partUpdates).toEqual([]);
    expect(command.partDeletes).toEqual([]);
    expect(command.messageDeletes).toEqual([
      { sessionID: "source", messageID: "progress" },
    ]);
    expect(command.sessionDeletes).toEqual(["ephemeral"]);
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

  test("progress cleanup-only failure after successful summaries follows ordinary backup rollback", async () => {
    const messages = nativeCheckpointFixture({ tailStartID: null });
    addProviderTokens(messages);
    const cleanupError = new Error("progress cleanup failed");
    const command = createCommandHarness(messages, {
      deleteProgressError: cleanupError,
    });

    await expect(executeMagicCompact(command.v2, "source", 1)).rejects.toBe(
      cleanupError,
    );
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
    ]);
    expect(command.partDeletes).toEqual([]);
    expect(command.messageDeletes).toEqual([
      { sessionID: "source", messageID: "progress" },
    ]);
    expect(command.sessionDeletes).toEqual(["ephemeral", "source"]);
    expect(command.selectedSessions).toEqual(["backup"]);
    expect(
      command.sessionUpdates.filter(
        request =>
          request.sessionID === "backup" && request.title === "Test session",
      ),
    ).toEqual([
      {
        sessionID: "backup",
        title: "Test session",
      },
    ]);
    expect(command.toasts.at(-1)).toMatchObject({
      title: "Magic Compact Failed",
      variant: "error",
    });
  });

  test("retains the primary generation failure when backup selection also fails", async () => {
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

    expect(error).toBeInstanceOf(AggregateError);
    expect(nestedErrorMessages(error)).toEqual(
      expect.arrayContaining([
        "summary generation failed",
        "backup selection failed",
      ]),
    );
    expect(command.selectedSessions).toEqual(["backup"]);
    expect(command.sessionDeletes).toEqual(["ephemeral"]);
    expect(command.events.indexOf("backup-update")).toBeLessThan(
      command.events.indexOf("select-backup"),
    );
    expect(command.events).not.toContain("delete-source");
    expect(command.events.indexOf("select-backup")).toBeLessThan(
      command.events.indexOf("failure-toast"),
    );
  });

  test("ephemeral checkpoint and progress cleanup failures use ordinary safe backup rollback", async () => {
    const messages = nativeCheckpointFixture({ tailStartID: null });
    addProviderTokens(messages);
    const changedEphemeral = [
      ...messages,
      nativeMarker("ephemeral-marker", null, false, "ephemeral"),
    ];
    const cleanupError = new Error("progress cleanup failed");
    const command = createCommandHarness(messages, {
      ephemeralMessagesAfterPrompt: changedEphemeral,
      deleteProgressError: cleanupError,
    });
    let error: unknown;

    try {
      await executeMagicCompact(command.v2, "source", 1);
    } catch (caught) {
      error = caught;
    }

    expect(error).toBeInstanceOf(AggregateError);
    expect(error).not.toBeInstanceOf(NativeCheckpointChangedError);
    expect(nestedErrorMessages(error)).toEqual(
      expect.arrayContaining([
        "Native compaction marker ephemeral-marker is incomplete or errored.",
        "progress cleanup failed",
      ]),
    );
    expect(command.sessionDeletes).toEqual(["ephemeral", "source"]);
    expect(command.selectedSessions).toEqual(["backup"]);
    expect(command.events.indexOf("backup-update")).toBeLessThan(
      command.events.indexOf("select-backup"),
    );
    expect(command.events.indexOf("select-backup")).toBeLessThan(
      command.events.indexOf("delete-source"),
    );
  });

  test("retains the primary generation failure when deleting the source also fails", async () => {
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

    expect(error).toBeInstanceOf(AggregateError);
    expect(nestedErrorMessages(error)).toEqual(
      expect.arrayContaining([
        "summary generation failed",
        "source deletion failed",
      ]),
    );
    expect(command.selectedSessions).toEqual(["backup"]);
    expect(command.sessionDeletes).toEqual(["ephemeral", "source"]);
    expect(command.events.indexOf("select-backup")).toBeLessThan(
      command.events.indexOf("delete-source"),
    );
    expect(command.events.indexOf("delete-source")).toBeLessThan(
      command.events.indexOf("failure-toast"),
    );
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
      expect(command.selectedSessions).toEqual(["backup"]);
      expect(command.sessionDeletes).toEqual(["ephemeral", "source"]);
      expect(command.events.indexOf("delete-source")).toBeLessThan(
        command.events.indexOf("failure-toast"),
      );
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
    expect(command.sessionDeletes).toEqual(["ephemeral"]);
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
        pruneSummarizedTurns({ v2, sessionID: "source" }, [turn]),
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
  onPrompt?: () => Promise<void>;
  promptError?: Error;
  promptErrorAt?: number;
  promptResponseError?: string;
  deleteError?: Error;
  deleteErrorAt?: number;
  messages?: MessageWithParts[];
  promptResponses?: string[];
  partUpdates?: Record<string, unknown>[];
  ephemeralMessages?: MessageWithParts[];
  ephemeralMessagesAfterPrompt?: MessageWithParts[];
};

async function runCompaction(
  sourceSession: Session,
  options: RunOptions = {},
): Promise<{
  prompts: Record<string, unknown>[];
  updates: Record<string, unknown>[];
  agentRequests: Record<string, unknown>[];
  partUpdates: Record<string, unknown>[];
}> {
  const prompts: Record<string, unknown>[] = [];
  const updates: Record<string, unknown>[] = [];
  const agentRequests: Record<string, unknown>[] = [];
  const partUpdates = options.partUpdates ?? [];
  const lifecycle = options.lifecycle ?? [];
  const sourceMessages = options.messages ?? compactableMessages();
  let forks = 0;
  let promptCount = 0;
  let deleteCount = 0;
  const promptedSessions = new Set<string>();
  const v2 = {
    app: {
      agents: async (request: Record<string, unknown>) => {
        lifecycle.push("agents");
        agentRequests.push(request);
        return { data: options.agents ?? [compactionAgent()] };
      },
    },
    session: {
      messages: async (request: { sessionID?: string } = {}) => {
        lifecycle.push("messages");
        if (
          request.sessionID
          && promptedSessions.has(request.sessionID)
          && options.ephemeralMessagesAfterPrompt
        ) {
          return { data: options.ephemeralMessagesAfterPrompt };
        }
        if (request.sessionID?.startsWith("ephemeral")) {
          return { data: options.ephemeralMessages ?? sourceMessages };
        }
        return { data: sourceMessages };
      },
      fork: async () => {
        lifecycle.push("fork");
        forks += 1;
        return {
          data: session(forks === 1 ? "ephemeral" : `ephemeral-${forks}`),
        };
      },
      update: async (request: Record<string, unknown>) => {
        lifecycle.push("update");
        updates.push(request);
        return { data: session("ephemeral") };
      },
      prompt: async (request: Record<string, unknown>) => {
        lifecycle.push("prompt");
        prompts.push(request);
        await options.onPrompt?.();
        promptCount += 1;
        promptedSessions.add(String(request.sessionID));
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
        const response =
          options.promptResponses?.[promptCount - 1]
          ?? responseForPrompt(promptText(request));
        return {
          data: {
            parts: [textPart("response", "ephemeral", "response", response)],
          },
        };
      },
      delete: async () => {
        lifecycle.push("delete");
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
  const summaries = await generateCompactionSummaries(v2, sourceSession, plan);
  await injectSummaries(v2, "source", plan.summarizedTurns, summaries);
  return {
    prompts,
    updates,
    agentRequests,
    partUpdates,
  };
}

function createCommandHarness(
  messages: MessageWithParts[],
  options: {
    racedMessages?: MessageWithParts[];
    ephemeralMessagesAfterPrompt?: MessageWithParts[];
    deleteProgressError?: Error;
    deleteEphemeralError?: Error;
    summaryPromptError?: Error;
    revalidationMessagesError?: Error;
    activateRaceAtSummaryPrompt?: number;
    summaryResponses?: string[];
    selectBackupError?: Error;
    deleteSourceError?: Error;
    showToastError?: Error;
    failureToastResponseError?: string;
    successToastResponseError?: string;
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
  const toasts: Record<string, unknown>[] = [];
  let forks = 0;
  let raceActive = false;
  let summaryPrompts = 0;

  const v2 = {
    app: { agents: async () => ({ data: [compactionAgent()] }) },
    session: {
      messages: async (request: { sessionID?: string } = {}) => {
        events.push("messages");
        if (request.sessionID?.startsWith("ephemeral")) {
          return {
            data:
              summaryPrompts > 0 && options.ephemeralMessagesAfterPrompt
                ? options.ephemeralMessagesAfterPrompt
                : messages,
          };
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
      get: async () => ({ data: session("source") }),
      fork: async () => {
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
        if (
          request.sessionID === "backup"
          && request.title === "Test session"
        ) {
          events.push("backup-update");
        }
        return { data: session(String(request.sessionID)) };
      },
      prompt: async (request: Record<string, unknown>) => {
        if (String(request.sessionID).startsWith("ephemeral")) {
          events.push("summary-prompt");
          summaryPrompts += 1;
          if (summaryPrompts === (options.activateRaceAtSummaryPrompt ?? 1)) {
            raceActive = true;
          }
          if (options.summaryPromptError) {
            throw options.summaryPromptError;
          }
          const response =
            options.summaryResponses?.[summaryPrompts - 1]
            ?? summaryResponse("suffix-1-assistant", "Suffix summary.");
          return {
            data: {
              parts: [textPart("response", "ephemeral", "response", response)],
            },
          };
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
        return {
          data: {
            info: { id: "progress" },
            parts: [],
          },
        };
      },
      deleteMessage: async (request: Record<string, unknown>) => {
        messageDeletes.push(request);
        if (options.deleteProgressError) {
          throw options.deleteProgressError;
        }
        return { data: true };
      },
      delete: async (request: { sessionID: string }) => {
        sessionDeletes.push(request.sessionID);
        if (request.sessionID === "source") {
          events.push("delete-source");
          if (options.deleteSourceError) throw options.deleteSourceError;
        }
        if (options.deleteEphemeralError && request.sessionID === "ephemeral") {
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
  return [
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

function remapForkMessages(
  messages: MessageWithParts[],
  sessionID: string,
): MessageWithParts[] {
  const messageIDs = new Map(
    messages.map(item => [item.info.id, `${sessionID}-${item.info.id}`]),
  );
  return messages.map(item => {
    const messageID = messageIDs.get(item.info.id)!;
    const info = {
      ...item.info,
      id: messageID,
      ...(item.info.role === "assistant" && item.info.parentID
        ? { parentID: messageIDs.get(item.info.parentID)! }
        : {}),
    } as Message;
    const parts = item.parts.map(part => {
      const mapped = {
        ...part,
        id: `${sessionID}-${part.id}`,
        sessionID,
        messageID,
      } as unknown as Record<string, unknown>;
      if (
        part.type === "compaction"
        && typeof (part as unknown as Record<string, unknown>)["tail_start_id"]
          === "string"
      ) {
        mapped["tail_start_id"] = messageIDs.get(
          (part as unknown as Record<string, string>)["tail_start_id"]!,
        )!;
      }
      return mapped as unknown as Part;
    });
    return { info, parts };
  });
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

function omittedRecordCount(value: string): number {
  const match = value.match(/(?:^|\n)omitted-record-count=(\d+)(?:\n|$)/);
  expect(match).not.toBeNull();
  return Number(match![1]);
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

function summaryResponse(id: string, summary: string): string {
  return `<summary><assistant id="${id}">${summary}</assistant></summary>`;
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

function responseForPrompt(prompt: string): string {
  const ids = [
    ...new Set(
      [...prompt.matchAll(/<assistant id="([A-Za-z0-9_-]+)">/g)].map(
        match => match[1]!,
      ),
    ),
  ];
  return `<summary>${ids
    .map(
      id =>
        `<assistant id="${id}">${id === "assistant" ? "Completed the request." : `Completed ${id}.`}</assistant>`,
    )
    .join("")}</summary>`;
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

function completedCommandTurn(
  command: string,
  output: string,
): MessageWithParts[] {
  const turn = metricTurn(1, HIGH_RISK_MESSAGE_COUNT, 0);
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

import { afterEach, beforeEach, describe, expect, spyOn, test } from "bun:test";
import { createHash, randomBytes, randomUUID } from "node:crypto";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type {
  Message,
  Part,
  Session,
  TextPart,
  ToolPart,
} from "@opencode-ai/sdk/v2";
import * as gptTokenizer from "gpt-tokenizer";
import {
  planCompaction,
  planTrim,
  type MessageWithParts,
  type Turn,
} from "../src/compact/plan";
import { projectToolTrim } from "../src/compact/prune";
import server from "../src/host-v1";
import {
  COMPACT_ARGUMENTS_ERROR,
  COMPACT_BACKEND_ERROR,
  COMPACT_THRESHOLD_ERROR,
  parseMagicCompactArguments,
  parseMagicTrimArguments,
  TRIM_ARGUMENTS_ERROR,
  TRIM_BACKEND_ERROR,
  TRIM_NOOP,
  TRIM_THRESHOLD_ERROR,
} from "../src/commands";
import { executeMagicTrim } from "../src/magic-trim";
import type { PluginInput } from "@opencode-ai/plugin";
import {
  commandRetentionPolicy,
  taskAwareRetentionPolicy,
} from "../src/policy/retention";
import {
  excerpt,
  RETENTION_KEEP_THRESHOLD_PERCENT_ENV,
  RETENTION_SUMMARY_EXCERPT_BYTES,
  RetentionClassificationError,
} from "../src/retention/classifier";
import {
  scoredJevFetch,
  scoredOpenAIFetch,
  SESSION_SUMMARY_TEXT,
  tokenLimitResponse,
  type JevCall,
} from "./retention-host";
import {
  closeV1CommandHarnesses,
  completedTool,
  message,
  PRIVATE_BODY,
  userText,
  v1CommandHarness,
} from "./v1-host";
import {
  cachePath,
  commitPlannedOmission,
  planOmission,
  readOmittedContent,
} from "../src/storage/omission";
import {
  type ConversationStats,
  readStats,
  writeStats,
} from "../src/storage/stats";
import { SourceNotAtRestError } from "../src/policy/successor";
import { statsMessage, trimStatsMessage } from "../src/stats/constants";
import type { StoredHistoryCount } from "../src/stats/stored-history";
import { countModelFacingContentTokens } from "../src/compact/accounting";

const HEURISTIC_OPTIONS = { retention: { mode: "heuristic" } } as const;
const HEURISTIC_FEEDBACK = "Heuristic mode: tool calls not evaluated.";
const OMISSION_ID = /[A-Za-z0-9_-]{1,12}:omitted-[A-Za-z0-9_-]{22}/gu;

const originalDataHome = process.env.XDG_DATA_HOME;
let storageDirectory: string;
beforeEach(async () => {
  storageDirectory = await mkdtemp(join(tmpdir(), "magic-trim-test-"));
  process.env.XDG_DATA_HOME = storageDirectory;
});
afterEach(async () => {
  closeV1CommandHarnesses();
  if (originalDataHome === undefined) delete process.env.XDG_DATA_HOME;
  else process.env.XDG_DATA_HOME = originalDataHome;
  await rm(storageDirectory, { recursive: true, force: true });
});

const cacheExists = (sessionID: string) =>
  Bun.file(cachePath(sessionID)).exists();
const cacheEntryIDs = async (sessionID: string) =>
  Object.keys(
    (
      JSON.parse(await Bun.file(cachePath(sessionID)).text()) as {
        entries: Record<string, unknown>;
      }
    ).entries,
  );

// Physical pruning stats count stored records before media projection.
async function storedTokens(messages: MessageWithParts[]): Promise<number> {
  const count = await countModelFacingContentTokens(messages);
  return count.tokens + count.projectedMediaTokens;
}

function thrownMessage(run: () => unknown): string {
  try {
    run();
  } catch (error) {
    return (error as Error).message;
  }
  throw new Error("Expected a thrown error.");
}
describe("magic trim", () => {
  test("preserves the requested assistant-turn tail", async () => {
    const messages = [
      message("usr_1", "user", []),
      message("ast_1", "assistant", [readTool("tool_1", "first")]),
      message("usr_2", "user", []),
      message("ast_2", "assistant", [readTool("tool_2", "second")]),
      message("usr_3", "user", []),
      message("ast_3", "assistant", [readTool("tool_3", "third")]),
    ];
    const plan = planTrim(messages, 1);

    expect(
      plan.trimmedTurns.flatMap(turn =>
        turn.assistants.map(assistant => assistant.info.id),
      ),
    ).toEqual(["ast_1", "ast_2"]);
  });

  test("selects the complete session despite native checkpoints", async () => {
    const marker = message("usr_checkpoint", "user", [
      {
        id: "prt_checkpoint",
        sessionID: "session",
        messageID: "usr_checkpoint",
        type: "compaction",
        auto: true,
      } as unknown as Part,
    ]);
    const summary = message("ast_checkpoint", "assistant", []);
    summary.info = {
      ...summary.info,
      parentID: "usr_checkpoint",
      summary: true,
      finish: "stop",
    } as Message;
    const messages = [
      message("usr_before", "user", []),
      message("ast_before", "assistant", [readTool("tool_before", "before")]),
      marker,
      summary,
      message("usr_after", "user", []),
      message("ast_after", "assistant", [readTool("tool_after", "after")]),
    ];
    const plan = planTrim(messages, 1);

    expect(
      plan.trimmedTurns.flatMap(turn =>
        turn.assistants.map(assistant => assistant.info.id),
      ),
    ).toEqual(["ast_before", "ast_checkpoint"]);
  });

  test("marks changed tools and skips them on later trims", async () => {
    const tool = completedTool("tool_1", "todowrite", "verbose output");
    const reasoning = {
      id: "reasoning_1",
      sessionID: "session",
      messageID: "assistant",
      type: "reasoning",
      text: "unchanged",
      time: { start: 1, end: 2 },
    } satisfies Part;
    const selectedTurn = turn(tool, reasoning);

    expect(await applyToolTrim("session", [selectedTurn], new Set())).toBe(1);
    expect(tool.state.status).toBe("completed");
    if (tool.state.status !== "completed") {
      throw new Error("Expected completed tool state.");
    }
    expect(tool.state.metadata["magicCompact"]).toEqual({ trimmed: true });
    expect(tool.state.output).toBe("Successfully updated todos.");
    expect(reasoning.text).toBe("unchanged");

    expect(await applyToolTrim("session", [selectedTurn], new Set())).toBe(0);
  });

  test("writes strict session-qualified omission IDs", async () => {
    const tool = readTool("tool_strict_id", "\u20ac");
    const sessionID = `ses_trim_${randomUUID().replaceAll("-", "")}`;

    await applyToolTrim(sessionID, [turn(tool)], new Set());

    expect(tool.state.status).toBe("completed");
    if (tool.state.status !== "completed")
      throw new Error("Expected completion");
    const contentID = /Content ID: ([^\s\]]+)\]$/u.exec(tool.state.output)?.[1];
    expect(contentID).toMatch(
      new RegExp(`^${sessionID.slice(-12)}:omitted-[A-Za-z0-9_-]{22}$`, "u"),
    );
    expect(tool.state.output).toBe(
      `[Historical read result omitted (3 bytes); Content ID: ${contentID}]`,
    );
    expect(await readOmittedContent(sessionID, contentID!)).toBe("\u20ac");
  });

  const longText = "delegation context packet ".repeat(60);
  const longCommand = "echo delegation context packet; ".repeat(40);
  const inputRules: Array<{
    tool: string;
    input: Record<string, unknown>;
    original: string;
    expected: (id: string) => Record<string, unknown>;
  }> = [
    {
      tool: "write",
      input: { filePath: "/w.ts", content: longText },
      original: longText,
      expected: id => ({
        filePath: "/w.ts",
        content: `[Omitted; Content ID: ${id}]`,
      }),
    },
    {
      tool: "apply_patch",
      input: { patchText: longText },
      original: longText,
      expected: id => ({ patchText: `[Omitted; Content ID: ${id}]` }),
    },
    {
      tool: "edit",
      input: { filePath: "/e.ts", oldString: longText, newString: "small" },
      original: `${longText}\nsmall`,
      expected: id => ({
        filePath: "/e.ts",
        oldString: `[Omitted with newString; Content ID: ${id}]`,
        newString: `[Omitted with oldString; Content ID: ${id}]`,
      }),
    },
    {
      tool: "bash",
      input: { command: longCommand, description: "run checks" },
      original: longCommand,
      expected: id => ({
        command: `${longCommand.slice(0, 512)}\n[Command truncated; Content ID: ${id}]`,
        description: "run checks",
      }),
    },
    {
      tool: "task",
      input: {
        subagent_type: "forager-worker",
        description: "delegated work",
        prompt: longText,
      },
      original: longText,
      expected: id => ({
        subagent_type: "forager-worker",
        description: "delegated work",
        prompt: `[Historical task prompt omitted; Content ID: ${id}. Retrieve before reuse.]`,
      }),
    },
  ];

  test.each(
    inputRules.flatMap(rule => [
      [rule.tool, false, rule] as const,
      [rule.tool, true, rule] as const,
    ]),
  )(
    "puts one exact recovery reference in the %s input (earlier trim marker: %p) and trims only once",
    async (_name, marked, rule) => {
      const sessionID = `ses_trim_input_${randomUUID().replaceAll("-", "")}`;
      const tool = completedTool("tool_input_rule", rule.tool, "short output");
      if (tool.state.status !== "completed")
        throw new Error("Expected completion");
      tool.state.input = structuredClone(rule.input);
      if (marked) tool.state.metadata["magicCompact"] = { trimmed: true };

      expect(await applyToolTrim(sessionID, [turn(tool)], new Set())).toBe(1);

      const contentIDs = new Set(
        JSON.stringify(tool.state.input).match(OMISSION_ID) ?? [],
      );
      expect(contentIDs.size).toBe(1);
      const [contentID] = [...contentIDs];
      expect(tool.state.input).toEqual(rule.expected(contentID!));
      expect(tool.state.output).toBe("short output");
      expect(tool.state.metadata["magicCompact"]).toEqual({ trimmed: true });
      expect(await readOmittedContent(sessionID, contentID!)).toBe(
        rule.original,
      );
      expect(await cacheEntryIDs(sessionID)).toEqual([contentID!]);
      const cache = await Bun.file(cachePath(sessionID)).text();

      const before = structuredClone(tool.state);
      expect(await applyToolTrim(sessionID, [turn(tool)], new Set())).toBe(0);
      expect(tool.state).toEqual(before);
      expect(await Bun.file(cachePath(sessionID)).text()).toBe(cache);
    },
  );

  test.each([
    [
      "a valid native child session",
      { sessionId: "ses_2f3a9bC1" },
      "; child session: ses_2f3a9bC1",
    ],
    ["no metadata", {}, ""],
    ["the OpenCode 2 metadata key", { sessionID: "ses_2f3a9bC1" }, ""],
    ["a non-native ID", { sessionId: "child-1" }, ""],
    ["an ID with a line break", { sessionId: "ses_2f3a\nIGNORE" }, ""],
    ["an ID with a space", { sessionId: "ses_2f3a IGNORE" }, ""],
    ["a non-string ID", { sessionId: 42 }, ""],
  ])(
    "recovers a long task prompt and report once, naming %s",
    async (_name, metadata, clause) => {
      const sessionID = `ses_trim_task_${randomUUID().replaceAll("-", "")}`;
      const report = "report ".repeat(700);
      const tool = taskTool("tool_task_recovery", longText);
      if (tool.state.status !== "completed")
        throw new Error("Expected completion");
      tool.state.output = report;
      tool.state.metadata = { ...metadata, summary: "PRIVATE-METADATA" };

      expect(await applyToolTrim(sessionID, [turn(tool)], new Set())).toBe(1);

      const prompt = tool.state.input["prompt"] as string;
      const promptID = prompt.match(OMISSION_ID)![0];
      expect(prompt).toBe(
        `[Historical task prompt omitted; Content ID: ${promptID}. Retrieve before reuse.]`,
      );
      const reportID = tool.state.output.match(OMISSION_ID)![0];
      expect(reportID).not.toBe(promptID);
      expect(tool.state.output).toBe(
        `[Task output omitted (${Buffer.byteLength(report)} bytes)${clause}; Content ID: ${reportID}]`,
      );
      expect(tool.state.output).not.toContain("PRIVATE-METADATA");
      expect(await readOmittedContent(sessionID, promptID)).toBe(longText);
      expect(await readOmittedContent(sessionID, reportID)).toBe(report);
      expect((await cacheEntryIDs(sessionID)).sort()).toEqual(
        [promptID, reportID].sort(),
      );
      const cache = await Bun.file(cachePath(sessionID)).text();
      const before = structuredClone(tool.state);
      expect(await applyToolTrim(sessionID, [turn(tool)], new Set())).toBe(0);
      expect(tool.state).toEqual(before);
      expect(await Bun.file(cachePath(sessionID)).text()).toBe(cache);
    },
  );

  test("keeps a short task report and names no child session for other tools", async () => {
    const sessionID = `ses_trim_task_short_${randomUUID().replaceAll("-", "")}`;
    const task = taskTool("tool_task_short", "Check the failing suite.");
    task.state.metadata = { sessionId: "ses_2f3a9bC1" };
    const other = completedTool("tool_other_long", "grep", longText);
    if (other.state.status !== "completed")
      throw new Error("Expected completion");
    other.state.metadata = { sessionId: "ses_2f3a9bC1" };

    await applyToolTrim(sessionID, [turn(task, other)], new Set());

    expect(task.state.status).toBe("completed");
    if (task.state.status !== "completed")
      throw new Error("Expected completion");
    expect(task.state.input).toEqual({
      subagent_type: "forager-worker",
      description: "delegated work",
      prompt: "Check the failing suite.",
    });
    expect(task.state.output).toBe("subagent result");
    const contentID = other.state.output.match(OMISSION_ID)![0];
    expect(other.state.output).toBe(
      `[Output omitted (${Buffer.byteLength(longText)} bytes); Content ID: ${contentID}]`,
    );
    expect(await readOmittedContent(sessionID, contentID)).toBe(longText);
    expect(await cacheEntryIDs(sessionID)).toEqual([contentID]);
  });

  test("leaves question input and output untouched", async () => {
    const sessionID = `ses_trim_question_${randomUUID().replaceAll("-", "")}`;
    const tool = completedTool("tool_question", "question", longText);
    if (tool.state.status !== "completed")
      throw new Error("Expected completion");
    tool.state.input = { questions: [{ question: longText }] };
    const before = structuredClone(tool.state);

    await applyToolTrim(sessionID, [turn(tool)], new Set());

    expect(tool.state).toEqual({
      ...before,
      metadata: { magicCompact: { trimmed: true } },
    });
    expect(await cacheExists(sessionID)).toBeFalse();
  });

  test("leaves earlier notices, placeholders and bare markers unchanged", async () => {
    const sessionID = `ses_trim_earlier_${randomUUID().replaceAll("-", "")}`;
    const oldID = "2c4f6a8b0d1e:omitted-yH8kQ2pL5vN7sR4tU6wXzA";
    const earlier = [
      taskTool("tool_old_bare", "[Omitted]"),
      taskTool(
        "tool_old_placeholder",
        `[Historical task prompt omitted; Content ID: ${oldID}. Retrieve the original with read_omitted_content before reuse.]`,
      ),
      completedTool(
        "tool_old_notice",
        "write",
        `<tool-input-omission-notice>\nFile write contents omitted.\n\nOmitted Length: 3000 bytes\nContent ID: ${oldID}\n</tool-input-omission-notice>\n\nWrote file.`,
      ),
    ];
    for (const tool of earlier) {
      if (tool.state.status !== "completed")
        throw new Error("Expected completion");
      tool.state.metadata["magicCompact"] = { trimmed: true };
    }
    const write = earlier[2]!;
    if (write.state.status !== "completed")
      throw new Error("Expected completion");
    write.state.input = {
      content: "[Omitted]",
      filePath: "/w.ts",
    };
    const before = structuredClone(earlier);

    expect(await applyToolTrim(sessionID, [turn(...earlier)], new Set())).toBe(
      0,
    );
    expect(earlier).toEqual(before);
    expect(await cacheExists(sessionID)).toBeFalse();
  });

  test("keeps output rules marker-guarded for tools marked by earlier trims", async () => {
    const sessionID = `ses_trim_marked_output_${randomUUID().replaceAll("-", "")}`;
    const tool = readTool("tool_marked_output", "large output ".repeat(100));
    tool.state.metadata["magicCompact"] = { trimmed: true };
    const before = structuredClone(tool.state);

    expect(await applyToolTrim(sessionID, [turn(tool)], new Set())).toBe(0);
    expect(tool.state).toEqual(before);
    expect(await cacheExists(sessionID)).toBeFalse();
  });

  // Inline file data tokenized as text can dwarf the conversation, so with an
  // inline file on either side the headline counts stored text only and the
  // files are reported by encoded size; without one the notice is unchanged.
  test("reports stored text apart from inline file data and keeps the notice unchanged without files", () => {
    const count = (
      tokens: number,
      inlineFileTokens = 0,
      files = 0,
      encodedBytes = 0,
    ): StoredHistoryCount => ({
      tokens,
      inlineFileTokens,
      inlineFiles: { count: files, encodedBytes },
    });

    expect(trimStatsMessage(count(1_500), count(999), 3, "R")).toBe(
      [
        "Magic Trim",
        "Stored history (estimate) | 1.5K → 999 tokens | 33% trimmed (~501 removed)",
        "Changes | Tool calls trimmed: 3",
        "Retention | R",
      ].join("\n"),
    );
    for (const [after, headline] of [
      [25, "0 → 25 text tokens, excluding inline file data | ~25 added"],
      [
        0,
        "0 → 0 text tokens, excluding inline file data | 0% trimmed (~0 removed)",
      ],
    ] as const) {
      expect(
        trimStatsMessage(
          count(100, 100, 1, 400),
          count(100 + after, 100, 1, 400),
          1,
          "R",
        ),
      ).toBe(
        [
          "Magic Trim",
          `Stored history (estimate) | ${headline}`,
          "Inline files | 1 file(s), 400 B encoded data in stored history before and after. Provider token cost for these files is not estimated; the text tokens above count stored text, not the model request.",
          "Changes | Tool calls trimmed: 1",
          "Retention | R",
        ].join("\n"),
      );
    }
    expect(trimStatsMessage(count(0), count(25), 1, "R")).toBe(
      [
        "Magic Trim",
        "Stored history (estimate) | 0 → 25 tokens | 0% grown (~25 added)",
        "Changes | Tool calls trimmed: 1",
        "Retention | R",
      ].join("\n"),
    );
    expect(
      trimStatsMessage(
        count(520_000, 400_000, 2, 1_572_864),
        count(450_000, 400_000, 2, 1_572_864),
        3,
        "R",
        ["Reasoning | X"],
      ),
    ).toBe(
      [
        "Magic Trim",
        "Stored history (estimate) | 120K → 50K text tokens, excluding inline file data | 58% trimmed (~70K removed)",
        "Inline files | 2 file(s), 1.5 MiB encoded data in stored history before and after. Provider token cost for these files is not estimated; the text tokens above count stored text, not the model request.",
        "Changes | Tool calls trimmed: 3",
        "Retention | R",
        "Reasoning | X",
      ].join("\n"),
    );
    // A projected image keeps its stored file, so only a changed inventory
    // reports a before and after.
    expect(
      statsMessage(
        2,
        count(30_000, 20_000, 2, 1_048_576),
        count(4_000, 600, 1, 800),
        1,
        { requested: 0, preserved: 0 },
        "R",
      ),
    ).toBe(
      [
        "Magic Compaction #2",
        "Stored history (estimate) | 10K → 3.4K text tokens, excluding inline file data | 66% trimmed (~6.6K removed)",
        "Inline files | 2 → 1 file(s), 1.0 MiB → 800 B encoded data in stored history. Provider token cost for these files is not estimated; the text tokens above count stored text, not the model request.",
        "Changes | Request turns summarized: 1",
        "Preserved | N=0 preserved no request turns.",
        "Retention | R",
      ].join("\n"),
    );
  });

  test("reports trim stats on a cycle-free active basis across the 999-to-1K boundary", async () => {
    const sessionID = `ses_trim_accounting_${randomUUID().replaceAll("-", "")}`;
    const tool = readTool("tool_accounting", "historical output ".repeat(70));
    const messages = [
      message("usr_accounting", "user", []),
      message("ast_accounting", "assistant", [tool]),
    ];
    const harness = v1CommandHarness(sessionID, messages);

    const sourceStats = {
      version: 1 as const,
      sourceSessionId: null,
      rootSessionId: "root-session",
      totalTokensPruned: 999,
      cachedTokensSaved: 11,
      processedAssistantMessageIds: [],
    };
    await writeStats(sessionID, sourceStats);
    const before = await countModelFacingContentTokens(messages);

    await expect(
      executeMagicTrim(harness.v2, sessionID, 0, HEURISTIC_OPTIONS),
    ).resolves.not.toBeNull();

    expect(harness.sourceWrites).toEqual([]);
    expect(await readStats(sessionID)).toEqual(sourceStats);
    const successor = harness.successorMessages();
    const after = await countModelFacingContentTokens(successor);
    const stats = await readStats(harness.successorID);
    const notice = successor
      .at(-1)!
      .parts.find((part): part is TextPart => part.type === "text")!;

    expect(stats).toMatchObject({
      sourceSessionId: sessionID,
      rootSessionId: "root-session",
      totalTokensPruned: 999 + before.tokens - after.tokens,
      cachedTokensSaved: 11,
      processedAssistantMessageIds: [successor[1]!.info.id],
    });
    expect(notice.text).toBe(
      trimStatsMessage(before, after, 1, HEURISTIC_FEEDBACK),
    );
    expect(notice.ignored).toBeTrue();
    expect(notice.text).toContain(`Retention | ${HEURISTIC_FEEDBACK}`);
    expect(notice.text).toContain("Magic Trim");
    expect(notice.text).toContain("Stored history (estimate) | ");
    expect(notice.text).toContain("% trimmed (");
    expect(notice.text).toContain("Changes | Tool calls trimmed: 1");
    expect(notice.text).not.toContain("Cache Savings");
    expect(notice.text).not.toContain("Cost Savings");
    expect(notice.text).not.toContain("Inline image tokens projected");
    expect(notice.text).not.toContain("ignored reporting notice is excluded");
    expect(harness.successorInfo()).toMatchObject({
      title: "Trim source",
      metadata: {
        magicCompact: {
          successorOf: sessionID,
          operation: "trim",
          compactionCount: 0,
          ready: true,
        },
      },
    });
    expect(harness.toasts.at(-1)?.message).toBe(
      `Trimmed 1 tool call(s) into successor session ${harness.successorID}. ${HEURISTIC_FEEDBACK}`,
    );
  });

  test("reports a no-op without creating a successor or touching caches", async () => {
    const sessionID = `ses_trim_noop_${randomUUID().replaceAll("-", "")}`;
    const alreadyTrimmed = completedTool("tool_noop", "todowrite", "done");
    alreadyTrimmed.state.metadata["magicCompact"] = { trimmed: true };
    const messages = [
      message("usr_noop", "user", []),
      message("ast_noop", "assistant", [alreadyTrimmed]),
    ];
    const harness = v1CommandHarness(sessionID, messages);

    await commitPlannedOmission(
      sessionID,
      planOmission(sessionID, { content: "source cache" }),
    );
    await writeStats(sessionID, trimStats(sessionID));
    const cache = await Bun.file(cachePath(sessionID)).text();

    await expect(
      executeMagicTrim(harness.v2, sessionID, 0, HEURISTIC_OPTIONS),
    ).resolves.toBeNull();

    expect(harness.created).toEqual([]);
    expect(harness.sourceWrites).toEqual([]);
    expect(await Bun.file(cachePath(sessionID)).text()).toBe(cache);
    expect(await readStats(sessionID)).toEqual(trimStats(sessionID));
    expect(harness.toasts.at(-1)?.message).toBe(
      `${TRIM_NOOP} ${HEURISTIC_FEEDBACK}`,
    );
  });

  test("deletes an unready trim candidate when the source changes before handoff", async () => {
    const sessionID = `ses_trim_drift_${randomUUID().replaceAll("-", "")}`;
    const messages = [
      message("usr_1", "user", [userText("usr_1", "Add the parser.")]),
      message("ast_1", "assistant", [
        readTool("tool_1", "large output ".repeat(100)),
      ]),
    ];
    const harness = v1CommandHarness(sessionID, messages, {
      duringSuccessorCreate: () => {
        const late = message("usr_late", "user", [
          userText("usr_late", "concurrent request"),
        ]);
        late.info.sessionID = sessionID;
        late.info.time = { created: 50 };
        late.parts[0]!.sessionID = sessionID;
        messages.push(late);
      },
    });

    await expect(
      executeMagicTrim(harness.v2, sessionID, 0, HEURISTIC_OPTIONS),
    ).rejects.toThrow("The session changed while Magic Trim was preparing");
    expect(harness.sourceWrites).toEqual([]);
    expect(harness.deletedSessions).toEqual([harness.successorID]);
    expect(await cacheExists(harness.successorID)).toBeFalse();
    expect(await readStats(harness.successorID)).toBeNull();
    expect(harness.toasts.at(-1)?.message).toContain(
      "The session changed while Magic Trim was preparing",
    );
  });

  // The host keeps the overrides' nested objects, so an in-place edit changes
  // what every later read returns and nothing else.
  test.each([
    [
      "metadata",
      (session: Session) => {
        session.metadata!["owner"] = "changed elsewhere";
      },
    ],
    [
      "model",
      (session: Session) => {
        session.model!.variant = "changed-elsewhere";
      },
    ],
    [
      "permission rules",
      (session: Session) => {
        session.permission![0]!.action = "deny";
      },
    ],
  ])(
    "deletes an unready trim candidate when only the source's nested %s change before handoff",
    async (_name, change) => {
      const sessionID = `ses_trim_continuation_${randomUUID().replaceAll("-", "")}`;
      const messages = [
        message("usr_1", "user", [userText("usr_1", "Add the parser.")]),
        message("ast_1", "assistant", [
          readTool("tool_1", "large output ".repeat(100)),
        ]),
      ];
      const sessionOverrides: Partial<Session> = {
        metadata: { owner: "source" },
        model: { id: "model", providerID: "provider", variant: "high" },
        permission: [{ permission: "*", pattern: "*", action: "allow" }],
      };
      const harness = v1CommandHarness(sessionID, messages, {
        sessionOverrides,
        duringSuccessorCreate: () => change(sessionOverrides as Session),
      });
      const history = structuredClone(messages);

      await expect(
        executeMagicTrim(harness.v2, sessionID, 0, HEURISTIC_OPTIONS),
      ).rejects.toThrow("The session changed while Magic Trim was preparing");
      expect(messages).toEqual(history);
      expect(harness.sourceWrites).toEqual([]);
      expect(harness.deletedSessions).toEqual([harness.successorID]);
      expect(harness.successorInfo()?.metadata).toMatchObject({
        magicCompact: { ready: false },
      });
      expect(await readStats(harness.successorID)).toBeNull();
    },
  );

  test("reports a failed source read without creating a successor or leaking the body", async () => {
    const sessionID = `ses_trim_read_${randomUUID().replaceAll("-", "")}`;
    const messages = [
      message("usr_1", "user", [userText("usr_1", "Add the parser.")]),
      message("ast_1", "assistant", [
        readTool("tool_1", "large output ".repeat(100)),
      ]),
    ];
    const harness = v1CommandHarness(sessionID, messages, {
      failSourceReadAfter: 0,
    });

    let failure: unknown;
    try {
      await executeMagicTrim(harness.v2, sessionID, 0, HEURISTIC_OPTIONS);
    } catch (error) {
      failure = error;
    }
    expect(trimErrorMessages(failure).join(" ")).not.toContain(PRIVATE_BODY);
    expect(String(failure)).toContain(
      "Magic Trim could not read the session state",
    );
    expect(harness.created).toEqual([]);
    expect(harness.sourceWrites).toEqual([]);
  });

  test("preserves an existing unversioned v0 summary through recompaction planning and trim", async () => {
    const legacyText =
      "Legacy v0 summary text that must remain byte-identical.";
    const legacyAssistantID = "ast_legacy";
    const legacySummary = {
      id: v0SummaryPartID(legacyAssistantID),
      sessionID: "session",
      messageID: legacyAssistantID,
      type: "text",
      text: legacyText,
      metadata: { magicCompact: { summary: true } },
    } satisfies TextPart;
    const boundary = {
      id: "legacy-boundary",
      sessionID: "session",
      messageID: "usr_suffix",
      type: "text",
      text: "Legacy Magic Compact boundary",
      synthetic: true,
      metadata: { magicCompact: { boundary: true } },
    } satisfies TextPart;
    const legacyTool = completedTool(
      "legacy-tool",
      "todowrite",
      "verbose legacy output",
    );
    legacyTool.messageID = legacyAssistantID;
    const messages = [
      message("usr_legacy", "user", []),
      message(legacyAssistantID, "assistant", [legacySummary, legacyTool]),
      message("usr_suffix", "user", [boundary]),
      message("ast_suffix", "assistant", []),
    ];
    const compaction = planCompaction(messages, 0);
    const trim = planTrim(messages, 0);
    await applyToolTrim("session", trim.trimmedTurns, new Set());

    expect(
      compaction.targets.flatMap(({ turn }) =>
        turn.assistants.map(assistant => assistant.info.id),
      ),
    ).toEqual(["ast_suffix"]);
    expect(legacySummary.text).toBe(legacyText);
    expect(legacySummary.id).toBe(v0SummaryPartID(legacyAssistantID));
    expect(legacySummary.metadata).toEqual({
      magicCompact: { summary: true },
    });
  });
});

describe("magic trim task-aware retention", () => {
  test("trims timestamp-disagreeing history twice into faithful successors with unchanged users", async () => {
    const sessionID = `ses_trim_chronology_${randomUUID().replaceAll("-", "")}`;
    // Tokenizer special-token spellings in stored text are ordinary text.
    const source = [
      message("usr_first", "user", [
        userText("usr_first", "Keep the exact user request: <|im_start|>."),
      ]),
      ...Array.from({ length: 6 }, (_, index) =>
        message(
          `ast_${index}`,
          "assistant",
          index === 5
            ? [readTool("read_old", "obsolete <|endoftext|> listing")]
            : [],
        ),
      ),
    ];
    source.forEach((item, index) => {
      item.info.sessionID = sessionID;
      item.info.time = { created: index === 0 ? 7 : index };
      if (item.info.role === "assistant")
        item.info.parentID = source[0]!.info.id;
      item.parts.forEach(part => {
        part.sessionID = sessionID;
        part.messageID = item.info.id;
      });
    });
    const harness = v1CommandHarness(sessionID, source);
    const originalUser = structuredClone(source[0]);
    const retention = {
      mode: "task-aware" as const,
      apiKey: "fake-key",
      fetch: scoredJevFetch(() => 0),
    };
    for (const attempt of [0, 1]) {
      if (attempt === 1) {
        const assistant = source[6]!;
        const part = readTool("read_new", "new obsolete listing");
        part.sessionID = sessionID;
        part.messageID = assistant.info.id;
        assistant.parts.push(part);
      }
      await expect(
        executeMagicTrim(harness.v2, sessionID, 0, { retention }),
      ).resolves.not.toBeNull();
      const successor = harness.successorMessages();
      expect(successor.slice(0, 7).map(item => item.info.role)).toEqual(
        source.slice(0, 7).map(item => item.info.role),
      );
      expect(successor[0]!.info.time).toEqual(originalUser!.info.time);
      expect(successor[0]!.parts[0]).toMatchObject({
        type: "text",
        text: "Keep the exact user request: <|im_start|>.",
      });
      expect((successor[6]!.info as { parentID: string }).parentID).toBe(
        successor[0]!.info.id,
      );
    }
    expect(harness.modelCalls.filter(call => call === "summary")).toHaveLength(
      2,
    );
    expect(harness.sourceWrites).toEqual([]);
  });
  // Both commands parse through one shared grammar. The full table runs on
  // /magic-trim's bindings; each command's own errors reach its host handler
  // below, and the --heuristic --match-model difference has its own test.
  test.each([
    [
      "",
      {
        keepTurns: 0,
        heuristic: false,
        matchModel: false,
        truncate: false,
      },
    ],
    [
      " 3 ",
      {
        keepTurns: 3,
        heuristic: false,
        matchModel: false,
        truncate: false,
      },
    ],
    ["--heuristic", { keepTurns: 0, heuristic: true, matchModel: false }],
    ["--heuristic 2", { keepTurns: 2, heuristic: true, matchModel: false }],
    ["2  --heuristic", { keepTurns: 2, heuristic: true, matchModel: false }],
    [
      "--truncate",
      { keepTurns: 0, heuristic: false, matchModel: false, truncate: true },
    ],
    [
      "1 --threshold 70 --truncate",
      {
        keepTurns: 1,
        heuristic: false,
        matchModel: false,
        keepThreshold: 0.7,
        truncate: true,
      },
    ],
    [
      "--truncate --threshold 60.1 1",
      {
        keepTurns: 1,
        heuristic: false,
        matchModel: false,
        keepThreshold: 0.601,
        truncate: true,
      },
    ],
    [
      "--threshold 0",
      {
        keepTurns: 0,
        heuristic: false,
        matchModel: false,
        keepThreshold: 0,
        truncate: false,
      },
    ],
    [
      "--threshold 100 4",
      {
        keepTurns: 4,
        heuristic: false,
        matchModel: false,
        keepThreshold: 1,
        truncate: false,
      },
    ],
    [
      `${Number.MAX_SAFE_INTEGER}`,
      {
        keepTurns: Number.MAX_SAFE_INTEGER,
        heuristic: false,
        matchModel: false,
        truncate: false,
      },
    ],
    [
      "--backend jev",
      {
        keepTurns: 0,
        heuristic: false,
        matchModel: false,
        backend: "jev",
        truncate: false,
      },
    ],
    [
      "1 --backend openai-decisions --threshold 70 --truncate",
      {
        keepTurns: 1,
        heuristic: false,
        matchModel: false,
        backend: "openai-decisions",
        keepThreshold: 0.7,
        truncate: true,
      },
    ],
    [
      "--truncate --backend openai-decisions 2",
      {
        keepTurns: 2,
        heuristic: false,
        matchModel: false,
        backend: "openai-decisions",
        truncate: true,
      },
    ],
    [
      "--match-model",
      { keepTurns: 0, heuristic: false, truncate: false, matchModel: true },
    ],
    [
      "2 --truncate --match-model --backend jev",
      {
        keepTurns: 2,
        heuristic: false,
        backend: "jev",
        truncate: true,
        matchModel: true,
      },
    ],
  ] as const)("parses %p", (raw, expected) => {
    expect(parseMagicTrimArguments(raw)).toEqual(expected);
  });

  const trimErrors = {
    arguments: TRIM_ARGUMENTS_ERROR,
    threshold: TRIM_THRESHOLD_ERROR,
    backend: TRIM_BACKEND_ERROR,
  };
  test.each([
    ["--heuristic --heuristic", "arguments"],
    ["--truncate --truncate", "arguments"],
    ["--threshold 70 --threshold 70", "arguments"],
    ["two", "arguments"],
    ["1 2", "arguments"],
    ["-1", "arguments"],
    ["1.5", "arguments"],
    [`${Number.MAX_SAFE_INTEGER + 1}`, "arguments"],
    ["--Heuristic", "arguments"],
    ["--heuristic=true", "arguments"],
    ["heuristic", "arguments"],
    ["--threshold=70", "arguments"],
    ["--thresholds 70", "arguments"],
    ["--heuristic --truncate", "arguments"],
    ["--threshold 70 --heuristic", "arguments"],
    ["--threshold", "threshold"],
    ["--threshold --truncate", "threshold"],
    ["--threshold 101", "threshold"],
    ["--threshold 100.5", "threshold"],
    ["--threshold -1", "threshold"],
    ["--threshold +60", "threshold"],
    ["--threshold 70%", "threshold"],
    ["--threshold 6e1", "threshold"],
    ["--threshold .5", "threshold"],
    ["--threshold 5.", "threshold"],
    ["--threshold NaN", "threshold"],
    ["--threshold Infinity", "threshold"],
    ["--threshold 0x3c", "threshold"],
    ["--threshold 100.00000000000000001", "threshold"],
    [`--threshold 0.${"0".repeat(400)}1`, "threshold"],
    ["--backend jev --backend jev", "arguments"],
    ["--backend jev --backend openai-decisions", "arguments"],
    ["--backend=openai-decisions", "arguments"],
    ["--backends jev", "arguments"],
    ["--heuristic --backend jev", "arguments"],
    ["--backend openai-decisions --heuristic", "arguments"],
    ["--backend", "backend"],
    ["--backend --truncate", "backend"],
    ["--backend 70", "backend"],
    ["--backend Jev", "backend"],
    ["--backend OPENAI-DECISIONS", "backend"],
    ["--backend openai", "backend"],
    ["--backend typesafe", "backend"],
    ["--backend jev|openai-decisions", "backend"],
    ["--match-model --match-model", "arguments"],
    ["--match-model=true", "arguments"],
    ["--matchmodel", "arguments"],
  ] as const)("rejects %p with the exact %s error", (raw, kind) => {
    expect(thrownMessage(() => parseMagicTrimArguments(raw))).toBe(
      trimErrors[kind],
    );
  });

  describe.each([
    [
      "magic-compact",
      COMPACT_ARGUMENTS_ERROR,
      COMPACT_THRESHOLD_ERROR,
      COMPACT_BACKEND_ERROR,
    ],
    [
      "magic-trim",
      TRIM_ARGUMENTS_ERROR,
      TRIM_THRESHOLD_ERROR,
      TRIM_BACKEND_ERROR,
    ],
  ] as const)(
    "%s arguments",
    (command, ARGUMENTS_ERROR, THRESHOLD_ERROR, BACKEND_ERROR) => {
      // One grammar, threshold and backend error reach this command's host
      // handler, which stops before it reads the environment or looks for a
      // server.
      test.each([
        ["--heuristic --truncate", ARGUMENTS_ERROR],
        ["--threshold 101", THRESHOLD_ERROR],
        ["--backend openai", BACKEND_ERROR],
      ])("rejects %p before any work", async (raw, message) => {
        const inherited = process.env;
        const reads: string[] = [];
        process.env = new Proxy(
          {},
          {
            get: (_target, key) => {
              reads.push(String(key));
              return undefined;
            },
          },
        ) as NodeJS.ProcessEnv;
        try {
          const hooks = await server({} as PluginInput);
          const failure = await hooks["command.execute.before"]!(
            { command, arguments: raw, sessionID: "session" },
            { parts: [] },
          ).then(
            () => undefined,
            (error: unknown) => error,
          );
          expect((failure as Error).message).toBe(message);
        } finally {
          process.env = inherited;
        }
        expect(reads).toEqual([]);
      });

      test("the argument errors name this command and the flags and echo no value", () => {
        for (const message of [ARGUMENTS_ERROR, THRESHOLD_ERROR]) {
          expect(message).toStartWith(`/${command} `);
          expect(message).toContain("--threshold");
          expect(message).not.toMatch(/MAGIC_COMPACT|%|Infinity|0x/);
        }
        expect(ARGUMENTS_ERROR).toContain("--backend jev|openai-decisions");
        expect(ARGUMENTS_ERROR).toContain("--match-model");
        expect(BACKEND_ERROR).toBe(
          `/${command} --backend needs jev or openai-decisions after a space, such as --backend openai-decisions.`,
        );
      });
    },
  );

  // --match-model chooses summary models. /magic-compact --heuristic still
  // summarizes every turn, so it keeps the flag; /magic-trim --heuristic makes
  // no summary request, so the combination is an argument error.
  test("--heuristic --match-model parses for /magic-compact and is rejected by /magic-trim", () => {
    expect(parseMagicCompactArguments("--match-model 1 --heuristic")).toEqual({
      keepTurns: 1,
      heuristic: true,
      matchModel: true,
    });
    expect(() => parseMagicTrimArguments("--heuristic --match-model")).toThrow(
      TRIM_ARGUMENTS_ERROR,
    );
    expect(() => parseMagicCompactArguments("--heuristic --truncate")).toThrow(
      COMPACT_ARGUMENTS_ERROR,
    );
  });

  test("keeps judged-useful calls whole and trims the rest by the existing rules", async () => {
    const sessionID = `ses_trim_retain_${randomUUID().replaceAll("-", "")}`;
    const useful = readTool("tool_useful", "useful parser finding ".repeat(80));
    const stale = readTool("tool_stale", "stale directory listing ".repeat(80));
    const todo = completedTool(
      "tool_todo",
      "todowrite",
      "Successfully updated todos.",
    );
    const messages = [
      message("usr_1", "user", [userText("usr_1", "Add the parser.")]),
      message("ast_1", "assistant", [useful, stale, todo]),
      message("usr_2", "user", [userText("usr_2", "Now fix its tests.")]),
      message("ast_2", "assistant", []),
    ];
    const calls: JevCall[] = [];
    const harness = v1CommandHarness(sessionID, messages);
    const usefulBefore = structuredClone(useful);

    await expect(
      executeMagicTrim(harness.v2, sessionID, 0, {
        retention: {
          mode: "task-aware",
          apiKey: "test-key",
          fetch: scoredJevFetch(
            output => (output.includes("useful") ? 0.95 : 0.1),
            calls,
          ),
        },
      }),
    ).resolves.not.toBeNull();

    expect(harness.successorPart("tool_useful").state).toEqual(
      usefulBefore.state,
    );
    const trimmed = harness.successorPart("tool_stale");
    if (trimmed.state.status !== "completed") throw new Error("completed");
    expect(trimmed.state.output).toContain("Historical read result omitted");
    expect(trimmed.state.metadata["magicCompact"]).toEqual({ trimmed: true });
    expect(calls.map(call => call.candidate.output.slice(0, 6))).toEqual([
      "useful",
      "stale ",
    ]);
    expect(calls[0]!.task).toEqual({
      latestUserRequest: "Now fix its tests.",
      sessionSummary: SESSION_SUMMARY_TEXT,
    });
    expect(calls[0]!.candidate.calledWhile).toBe(
      "handling the previous user request",
    );
    expect(harness.forks).toEqual([harness.summaryForkID]);
    expect(harness.deletedSessions).toEqual([harness.summaryForkID]);
    expect(JSON.stringify(messages)).not.toContain(harness.summaryForkID);
    expect(harness.toasts.at(-1)?.message).toBe(
      `Trimmed 2 tool call(s) into successor session ${harness.successorID}. Task-aware retention with jev (jev-1.13.0) at keep threshold 60% kept 1 of 2 evaluated tool call(s) whole (read 1).`,
    );
    const notice = harness.successorMessages().at(-1)!.parts[0] as TextPart;
    expect(notice.ignored).toBeTrue();
    expect(notice.text.split("\n")).toHaveLength(4);
    expect(notice.text).toContain("Changes | Tool calls trimmed: 2");
    expect(notice.text).toContain(
      "Retention | Task-aware retention with jev (jev-1.13.0) at keep threshold 60% kept 1 of 2 evaluated tool call(s) whole (read 1).",
    );
  });

  test.each([
    [0.1, undefined, false],
    [0.95, undefined, true],
    [0.7, undefined, true],
    [0.7, "80", false],
  ])(
    "reports zero or all keeps by probability %p at percent %p without adding no-op notices (keep=%p)",
    async (score, percent, keep) => {
      const sessionID = `ses_trim_keep_${randomUUID().replaceAll("-", "")}`;
      const messages = [
        message("usr_1", "user", [userText("usr_1", "Fix tests.")]),
        message("ast_1", "assistant", [
          readTool("tool_read", "large output ".repeat(100)),
        ]),
      ];
      const harness = v1CommandHarness(sessionID, messages);
      const retention = {
        ...taskAwareRetentionPolicy({
          TYPESAFE_API_KEY: "test-key",
          ...(percent === undefined
            ? {}
            : { [RETENTION_KEEP_THRESHOLD_PERCENT_ENV]: percent }),
        }),
        fetch: scoredJevFetch(() => score, []),
      };
      expect(
        await executeMagicTrim(harness.v2, sessionID, 0, { retention }),
      ).toEqual(keep ? null : expect.anything());
      const feedback = `Task-aware retention with jev (jev-1.13.0) at keep threshold ${percent ?? 60}% kept ${keep ? 1 : 0} of 1 evaluated tool call(s) whole${keep ? " (read 1)" : ""}.`;
      expect(harness.toasts.at(-1)?.message).toBe(
        `${keep ? TRIM_NOOP : `Trimmed 1 tool call(s) into successor session ${harness.successorID}.`} ${feedback}`,
      );
      expect(harness.sourceWrites).toEqual([]);
      if (keep) {
        expect(harness.created).toEqual([]);
        expect(harness.deletedSessions).toEqual([harness.summaryForkID]);
        expect(await readStats(sessionID)).toBeNull();
      } else {
        const notice = harness.successorMessages().at(-1)!.parts[0] as TextPart;
        expect(notice.ignored).toBeTrue();
        expect(notice.text).toContain(`Retention | ${feedback}`);
        expect(notice.text).toContain("Changes | Tool calls trimmed: 1");
      }
    },
  );

  test("makes no summary or classifier request when no call would change", async () => {
    const sessionID = `ses_trim_no_candidates_${randomUUID().replaceAll("-", "")}`;
    const marked = readTool("tool_marked", "large output ".repeat(100));
    if (marked.state.status !== "completed") throw new Error("completed");
    marked.state.metadata["magicCompact"] = { trimmed: true };
    const messages = [
      message("usr_1", "user", []),
      message("ast_1", "assistant", [
        marked,
        completedTool("tool_small", "bash", "ok"),
      ]),
    ];
    const harness = v1CommandHarness(sessionID, messages);

    // The selected backend's key is not needed when nothing is evaluated.
    await expect(
      executeMagicTrim(harness.v2, sessionID, 0, {
        retention: commandRetentionPolicy(
          "trim",
          parseMagicTrimArguments("--backend openai-decisions"),
          {},
        ),
      }),
    ).resolves.not.toBeNull();

    expect(harness.modelCalls).toEqual([]);
    expect(harness.forks).toEqual([]);
    expect(harness.toasts.at(-1)?.message).toBe(
      `Trimmed 1 tool call(s) into successor session ${harness.successorID}. Task-aware retention with openai-decisions (gpt-6-luna) at keep threshold 60%: no tool calls needed evaluation.`,
    );
    expect(
      (harness.successorMessages().at(-1)!.parts[0] as TextPart).text,
    ).toContain(
      "Retention | Task-aware retention with openai-decisions (gpt-6-luna) at keep threshold 60%: no tool calls needed evaluation.",
    );
  });

  test("--backend openai-decisions judges every candidate there and names it", async () => {
    const sessionID = `ses_trim_openai_${randomUUID().replaceAll("-", "")}`;
    const harness = v1CommandHarness(sessionID, [
      message("usr_1", "user", [userText("usr_1", "Fix tests.")]),
      message("ast_1", "assistant", [
        readTool("tool_read", "large output ".repeat(100)),
      ]),
    ]);
    const calls: JevCall[] = [];
    await expect(
      executeMagicTrim(harness.v2, sessionID, 0, {
        retention: {
          ...commandRetentionPolicy(
            "trim",
            parseMagicTrimArguments(
              "--backend openai-decisions --threshold 70",
            ),
            { MAGIC_COMPACT_OPENAI_DECISIONS_API_KEY: "test-openai-key" },
          ),
          fetch: scoredOpenAIFetch(() => 0.65, calls),
        },
      }),
    ).resolves.not.toBeNull();
    expect(calls).toHaveLength(1);
    expect(calls[0]!.task.sessionSummary).toBe(SESSION_SUMMARY_TEXT);
    expect(harness.toasts.at(-1)?.message).toBe(
      `Trimmed 1 tool call(s) into successor session ${harness.successorID}. Task-aware retention with openai-decisions (gpt-6-luna) at keep threshold 70% kept 0 of 1 evaluated tool call(s) whole.`,
    );
  });

  test("heuristic mode trims candidates without a key, model, or classifier", async () => {
    const sessionID = `ses_trim_heuristic_${randomUUID().replaceAll("-", "")}`;
    const tool = readTool("tool_read", "large output ".repeat(100));
    const harness = v1CommandHarness(sessionID, [
      message("usr_1", "user", []),
      message("ast_1", "assistant", [tool]),
    ]);

    await expect(
      executeMagicTrim(harness.v2, sessionID, 0, HEURISTIC_OPTIONS),
    ).resolves.not.toBeNull();

    expect(harness.modelCalls).toEqual([]);
    expect(harness.forks).toEqual([]);
    const trimmed = harness.successorPart("tool_read");
    if (trimmed.state.status !== "completed") throw new Error("completed");
    expect(trimmed.state.output).toContain("Historical read result omitted");
  });

  // Accounting runs before and after the edit and again on the successor; an
  // unchanged large file is tokenized only on the first of those counts.
  test("tokenizes an unchanged large file once and copies it unchanged", async () => {
    const sessionID = `ses_trim_large_file_${randomUUID().replaceAll("-", "")}`;
    const url = `data:application/pdf;base64,${randomBytes(98_304).toString("base64")}`;
    const messages = [
      message("usr_pdf", "user", [
        userText("usr_pdf", "review the attached report"),
        {
          id: "prt_pdf",
          sessionID: "session",
          messageID: "usr_pdf",
          type: "file",
          mime: "application/pdf",
          filename: "report.pdf",
          url,
        } as Part,
      ]),
      message("ast_pdf", "assistant", [
        readTool("tool_read", "large output ".repeat(100)),
      ]),
    ];
    const harness = v1CommandHarness(sessionID, messages);
    const original = structuredClone(messages);
    const encodeSpy = spyOn(gptTokenizer, "encodeGenerator");

    try {
      expect(
        await executeMagicTrim(harness.v2, sessionID, 0, HEURISTIC_OPTIONS),
      ).not.toBeNull();
      expect(
        encodeSpy.mock.calls.filter(([text]) => text === url),
      ).toHaveLength(1);

      expect(harness.sourceWrites).toEqual([]);
      const successor = harness.successorMessages();
      expect(successor[0]!.parts[1]).toMatchObject({
        type: "file",
        mime: "application/pdf",
        url,
      });
      const notice = successor
        .at(-1)!
        .parts.find((part): part is TextPart => part.type === "text")!;
      const before = await countModelFacingContentTokens(original);
      const after = await countModelFacingContentTokens(successor);
      const fileTokens = gptTokenizer.countTokens(url);
      expect([before.inlineFileTokens, after.inlineFileTokens]).toEqual([
        fileTokens,
        fileTokens,
      ]);
      expect(notice.text).toBe(
        trimStatsMessage(before, after, 1, HEURISTIC_FEEDBACK),
      );
      expect(notice.text.split("\n").slice(1, 3)).toEqual([
        expect.stringMatching(
          /^Stored history \(estimate\) \| \S+ → \S+ text tokens, excluding inline file data \| \d+% trimmed/u,
        ),
        `Inline files | 1 file(s), 128.0 KiB encoded data in stored history before and after. Provider token cost for these files is not estimated; the text tokens above count stored text, not the model request.`,
      ]);
      // Stats keep the canonical totals, file data included.
      expect((await readStats(harness.successorID))?.totalTokensPruned).toBe(
        (await storedTokens(original)) - (await storedTokens(successor)),
      );
    } finally {
      encodeSpy.mockRestore();
    }
  });

  // A native checkpoint can retain a tail that starts at the assistant carrying
  // a media capsule while its images stay in the hidden prefix. The stored
  // history is valid and trim counts stored history, so it must succeed and
  // report the whole stored totals.
  test("trims a stored history whose native retained tail starts at a media capsule", async () => {
    const sessionID = `ses_trim_media_tail_${randomUUID().replaceAll("-", "")}`;
    const url = "data:image/png;base64,Zm9v";
    const messages = [
      message("usr_media", "user", [
        userText("usr_media", "review"),
        {
          id: "prt_image",
          sessionID: "session",
          messageID: "usr_media",
          type: "file",
          mime: "image/png",
          url,
        } as Part,
      ]),
      message("ast_media", "assistant", [
        {
          ...userText("ast_media", "Historical summary"),
          id: "prt_-magic_capsule_v3_ast_media",
          metadata: {
            magicCompact: {
              summary: true,
              capsule: true,
              agentic: true,
              version: 3,
              mediaManifest: {
                version: 1,
                images: [
                  {
                    ordinal: 1,
                    mime: "image/png",
                    urlSha256: createHash("sha256").update(url).digest("hex"),
                  },
                ],
              },
            },
          },
        },
        readTool("tool_read", "large output ".repeat(100)),
      ]),
      ...checkpointedSession()
        .slice(4)
        .map(item => structuredClone(item)),
    ];
    const marker = messages.find(item => item.info.id === "usr_marker")!;
    (marker.parts[0] as unknown as { tail_start_id: string }).tail_start_id =
      "ast_media";
    const harness = v1CommandHarness(sessionID, messages);
    const storedBefore = await storedTokens(messages);

    expect(
      await executeMagicTrim(harness.v2, sessionID, 0, HEURISTIC_OPTIONS),
    ).not.toBeNull();

    const successor = harness.successorMessages();
    const notice = successor
      .at(-1)!
      .parts.find((part): part is TextPart => part.type === "text")!;
    const storedAfter = await storedTokens(successor);
    expect(storedAfter).toBeLessThan(storedBefore);
    expect(notice.text).toBe(
      trimStatsMessage(
        await countModelFacingContentTokens(messages),
        await countModelFacingContentTokens(successor),
        2,
        HEURISTIC_FEEDBACK,
      ),
    );
    expect((await readStats(harness.successorID))?.totalTokensPruned).toBe(
      storedBefore - storedAfter,
    );
  });

  test.each([
    [
      "a missing API key",
      undefined,
      () => new Response("private", { status: 200 }),
    ],
    [
      "a failed classification",
      "test-key",
      () => new Response("private", { status: 500 }),
    ],
    ["a token-limit refusal", "test-key", tokenLimitResponse],
  ] as const)(
    "%s leaves the source, cache, and stats untouched without a successor",
    async (name, apiKey, respond) => {
      const sessionID = `ses_trim_retention_fail_${randomUUID().replaceAll("-", "")}`;
      const messages = [
        message("usr_1", "user", [userText("usr_1", "Add the parser.")]),
        message("ast_1", "assistant", [
          readTool("tool_read", "large output ".repeat(100)),
        ]),
      ];
      const harness = v1CommandHarness(sessionID, messages);
      let failure: unknown;

      try {
        await executeMagicTrim(harness.v2, sessionID, 0, {
          retention: {
            mode: "task-aware",
            apiKey,
            fetch: async () => respond(),
          },
        });
      } catch (error) {
        failure = error;
      }

      expect(failure).toBeInstanceOf(RetentionClassificationError);
      expect(String(failure)).toContain("/magic-trim --heuristic");
      if (name === "a token-limit refusal") {
        expect((failure as RetentionClassificationError).kind).toBe(
          "token-limit",
        );
        expect(String(failure)).toContain(
          "no retention decisions were applied",
        );
      }
      expect(String(failure)).not.toContain("private");
      expect(harness.created).toEqual([]);
      expect(harness.deletedSessions).toEqual(
        apiKey ? [harness.summaryForkID] : [],
      );
      expect(harness.modelCalls).toEqual(
        apiKey ? ["agents", "providers", "summary"] : [],
      );
      expect(harness.sourceWrites).toEqual([]);
      expect(await cacheExists(sessionID)).toBeFalse();
      expect(await readStats(sessionID)).toBeNull();
      expect(harness.toasts.at(-1)?.message).toContain(
        "/magic-trim --heuristic",
      );
    },
  );

  test("--truncate retries a token-limit refusal with the summary excerpt and reports it", async () => {
    const sessionID = `ses_trim_truncate_${randomUUID().replaceAll("-", "")}`;
    const summaryText = `## Objective\n${"- A dense detail line.\n".repeat(1_500)}## Next Move`;
    const useful = readTool("tool_useful", "useful parser finding ".repeat(80));
    const messages = [
      message("usr_1", "user", [userText("usr_1", "Add the parser.")]),
      message("ast_1", "assistant", [
        useful,
        readTool("tool_stale", "stale directory listing ".repeat(80)),
      ]),
    ];
    const usefulBefore = structuredClone(useful);
    const harness = v1CommandHarness(sessionID, messages, { summaryText });
    const summaries: string[] = [];
    const answer = scoredJevFetch(output =>
      output.includes("useful") ? 0.75 : 0.1,
    );
    const successor = await executeMagicTrim(harness.v2, sessionID, 0, {
      retention: {
        ...commandRetentionPolicy(
          "trim",
          parseMagicTrimArguments("--threshold 70 --truncate"),
          { TYPESAFE_API_KEY: "test-key" },
        ),
        fetch: async (url, init) => {
          const body = JSON.parse(String(init.body));
          summaries.push(body.state.task.sessionSummary);
          // Only the useful call's whole-summary request is refused.
          return body.state.task.sessionSummary === summaryText
            && body.state.candidates[0].output.includes("useful")
            ? tokenLimitResponse()
            : answer(url, init);
        },
      },
    });
    expect(successor).not.toBeNull();
    expect(summaries.filter(text => text === summaryText)).toHaveLength(2);
    expect(summaries.filter(text => text !== summaryText)).toEqual([
      excerpt(summaryText, RETENTION_SUMMARY_EXCERPT_BYTES),
    ]);
    const feedback =
      "Task-aware retention with jev (jev-1.13.0) at keep threshold 70% kept 1 of 2 evaluated tool call(s) whole (read 1). 1 decision(s) used a shortened session summary after a token-limit refusal (--truncate) and could have missed useful context.";
    expect(harness.toasts.at(-1)?.message).toBe(
      `Trimmed 1 tool call(s) into successor session ${harness.successorID}. ${feedback}`,
    );
    const notice = harness.successorMessages().at(-1)!.parts[0] as TextPart;
    expect(notice.text).toContain(`Retention | ${feedback}`);
    expect(harness.successorPart("tool_useful").state).toEqual(
      usefulBefore.state,
    );
  });

  test("a token-limit refusal without --truncate toasts the person's own retry once", async () => {
    const sessionID = `ses_trim_truncate_hint_${randomUUID().replaceAll("-", "")}`;
    const messages = [
      message("usr_1", "user", [userText("usr_1", "Add the parser.")]),
      message("ast_1", "assistant", [
        readTool("tool_read", "large output ".repeat(100)),
      ]),
      message("usr_2", "user", [userText("usr_2", "Continue.")]),
      message("ast_2", "assistant", []),
    ];
    const harness = v1CommandHarness(sessionID, messages, {
      summaryText: "s".repeat(20_000),
    });
    let calls = 0;
    const failure = await executeMagicTrim(harness.v2, sessionID, 1, {
      retention: {
        ...commandRetentionPolicy(
          "trim",
          parseMagicTrimArguments("--threshold 70 1"),
          { TYPESAFE_API_KEY: "test-key" },
        ),
        fetch: async () => {
          calls += 1;
          return tokenLimitResponse();
        },
      },
    }).then(
      () => undefined,
      (error: unknown) => error,
    );
    expect(failure).toBeInstanceOf(RetentionClassificationError);
    expect(calls).toBe(1);
    const errors = harness.toasts.filter(
      toast => (toast as { variant?: string }).variant === "error",
    );
    expect(errors).toHaveLength(1);
    const text = errors[0]!.message!;
    expect(text).toBe((failure as Error).message);
    expect(text.length).toBeLessThan(700);
    expect(text).toContain(
      "Run /magic-trim 1 --threshold 70 --truncate to retry each refused call once",
    );
    expect(text).toContain("or /magic-trim 1 --heuristic to trim");
    expect(text).not.toContain("sss");
    expect(harness.sourceWrites).toEqual([]);
    expect(harness.created).toEqual([]);
    expect(await readStats(sessionID)).toBeNull();
  });

  test.each([
    ["heuristic", HEURISTIC_OPTIONS],
    [
      "task-aware",
      {
        retention: {
          mode: "task-aware",
          apiKey: "test-key",
          fetch: async () => {
            throw new Error("unexpected classifier request");
          },
        },
      },
    ],
  ] as const)(
    "refuses a stored unfinished tool call in %s mode before any retention or write",
    async (_name, options) => {
      const sessionID = `ses_trim_unfinished_${randomUUID().replaceAll("-", "")}`;
      const stale = completedTool("tool_stale", "task", "never delivered");
      stale.state = { status: "running", input: {}, time: { start: 1 } };
      const messages = [
        message("usr_1", "user", [userText("usr_1", "Add the parser.")]),
        message("ast_old", "assistant", [stale]),
        message("usr_2", "user", [userText("usr_2", "Continue.")]),
        message("ast_1", "assistant", [
          readTool("tool_read", "large output ".repeat(100)),
        ]),
      ];
      const harness = v1CommandHarness(sessionID, messages);
      let failure: unknown;

      try {
        await executeMagicTrim(harness.v2, sessionID, 0, options);
      } catch (error) {
        failure = error;
      }

      expect(failure).toBeInstanceOf(SourceNotAtRestError);
      expect(String(failure)).toEndWith(
        "Record: assistant message ast_old; running task call call_tool_stale.",
      );
      expect(harness.created).toEqual([]);
      expect(harness.forks).toEqual([]);
      expect(harness.modelCalls).toEqual([]);
      expect(harness.sourceWrites).toEqual([]);
      // The toast shows the refusal itself, without the error's class name.
      expect(harness.toasts.at(-1)?.message).toBe((failure as Error).message);
      expect(await cacheExists(sessionID)).toBeFalse();
      expect(await readStats(sessionID)).toBeNull();
    },
  );

  test.each([
    [
      "source drift during classification",
      "drift",
      "The session changed while Magic Trim was preparing",
    ],
    [
      "a pending revert during classification",
      "revert",
      "it has a pending revert",
    ],
  ] as const)(
    "%s discards the candidate and leaves the source",
    async (_name, when, expected) => {
      const sessionID = `ses_trim_retention_race_${randomUUID().replaceAll("-", "")}`;
      const tool = readTool("tool_read", "large output ".repeat(100));
      const messages = [
        message("usr_1", "user", [userText("usr_1", "Add the parser.")]),
        message("ast_1", "assistant", [tool]),
      ];
      const harness = v1CommandHarness(sessionID, messages, {
        duringSummary: () => {
          if (when === "revert") {
            harness.setRevert({ messageID: "ast_1" });
            return;
          }
          const concurrent = message("usr_concurrent", "user", [
            userText("usr_concurrent", "concurrent request"),
          ]);
          concurrent.info.sessionID = sessionID;
          concurrent.info.time = { created: 50 };
          concurrent.parts[0]!.sessionID = sessionID;
          messages.push(concurrent);
        },
      });

      await expect(
        executeMagicTrim(harness.v2, sessionID, 0, {
          retention: {
            mode: "task-aware",
            apiKey: "test-key",
            fetch: scoredJevFetch(() => 0.1),
          },
        }),
      ).rejects.toThrow(expected);

      expect(harness.deletedSessions).toEqual([
        harness.summaryForkID,
        harness.successorID,
      ]);
      expect(harness.sourceWrites).toEqual([]);
      expect(await cacheExists(sessionID)).toBeFalse();
      expect(await readStats(sessionID)).toBeNull();
      expect(await readStats(harness.successorID)).toBeNull();
    },
  );
});

function trimStats(sessionID: string): ConversationStats {
  return {
    version: 1,
    sourceSessionId: null,
    rootSessionId: sessionID,
    totalTokensPruned: 0,
    cachedTokensSaved: 0,
    processedAssistantMessageIds: [],
  };
}

function trimErrorMessages(value: unknown): string[] {
  if (value instanceof AggregateError) {
    return [
      value.message,
      ...value.errors.flatMap(trimErrorMessages),
      ...trimErrorMessages(value.cause),
    ];
  }
  if (value instanceof Error) {
    return [value.message, ...trimErrorMessages(value.cause)];
  }
  return [];
}

function turn(...parts: Part[]): Turn {
  return {
    user: [message("user", "user", [])],
    assistants: [message("assistant", "assistant", parts)],
  };
}

function readTool(id: string, output: string): ToolPart {
  return completedTool(id, "read", output);
}

function taskTool(id: string, prompt: string): ToolPart {
  const tool = completedTool(id, "task", "subagent result");
  if (tool.state.status !== "completed") {
    throw new Error("Expected completed tool state.");
  }
  tool.state.input = {
    subagent_type: "forager-worker",
    description: "delegated work",
    prompt,
  };
  return tool;
}

// Hidden prefix, a retained tail and a completed native checkpoint, then a
// suffix turn.
function checkpointedSession(): MessageWithParts[] {
  const marker = message("usr_marker", "user", [
    {
      id: "prt_marker",
      sessionID: "session",
      messageID: "usr_marker",
      type: "compaction",
      auto: true,
      tail_start_id: "usr_tail",
    } as unknown as Part,
  ]);
  const summary = message("ast_summary", "assistant", [
    { ...userText("ast_summary", "checkpoint"), id: "prt_summary" },
  ]);
  summary.info = {
    ...summary.info,
    parentID: "usr_marker",
    summary: true,
    finish: "stop",
  } as Message;
  return [
    message("usr_hidden", "user", [userText("usr_hidden", "hidden request")]),
    message("ast_hidden", "assistant", [
      readTool("tool_hidden", "hidden history ".repeat(400)),
    ]),
    message("usr_tail", "user", [userText("usr_tail", "retained request")]),
    message("ast_tail", "assistant", [userText("ast_tail", "retained answer")]),
    marker,
    summary,
    message("usr_suffix", "user", [userText("usr_suffix", "latest request")]),
    message("ast_suffix", "assistant", [
      readTool("tool_suffix", "latest history ".repeat(400)),
    ]),
  ];
}

function v0SummaryPartID(messageID: string): string {
  return `prt_-magic_summary_${messageID}`;
}

// Applies the planned tool trim to the given parts in place and commits its
// planned omissions, so rule tests can inspect each part directly.
async function applyToolTrim(
  sessionID: string,
  turns: Turn[],
  keep: ReadonlySet<string>,
): Promise<number> {
  const trim = await projectToolTrim(turns, sessionID, keep);
  for (const omission of trim.omissions) {
    await commitPlannedOmission(sessionID, omission);
  }
  const parts = new Map(
    turns
      .flatMap(item => item.assistants)
      .flatMap(item => item.parts)
      .map(part => [part.id, part]),
  );
  for (const replacement of trim.replacements) {
    Object.assign(parts.get(replacement.id)!, replacement);
  }
  return trim.replacements.length;
}

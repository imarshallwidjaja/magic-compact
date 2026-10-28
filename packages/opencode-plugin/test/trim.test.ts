import { describe, expect, test } from "bun:test";
import { createHash, randomUUID } from "node:crypto";
import { rm } from "node:fs/promises";
import type { Message, Part, TextPart, ToolPart } from "@opencode-ai/sdk/v2";
import type { V2Client } from "../src/api";
import {
  createCompactionPlan,
  createTrimPlan,
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
} from "../src/magic-compact";
import {
  TRIM_ARGUMENTS_ERROR,
  TRIM_BACKEND_ERROR,
  TRIM_THRESHOLD_ERROR,
  TRIM_NOOP,
  executeMagicTrim,
  parseMagicTrimArguments,
} from "../src/magic-trim";
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
  cleanupSessions,
  completedTool,
  message,
  PRIVATE_BODY,
  userText,
  v1CommandHarness,
} from "./v1-host";
import {
  allocateOmission,
  cachePath,
  commitPlannedOmission,
  readCache,
  readOmittedContent,
} from "../src/storage/omission";
import {
  type ConversationStats,
  readStats,
  writeStats,
} from "../src/storage/stats";
import { SourceNotAtRestError } from "../src/policy/successor";
import { planToolTrim, type ToolRuleCategory } from "../src/policy/tool-rules";
import { trimStatsMessage } from "../src/stats/constants";
import {
  countCanonicalContentTokens,
  countGptTokens,
} from "../src/stats/tokenize";

const HEURISTIC_OPTIONS = { retention: { mode: "heuristic" } } as const;
const HEURISTIC_FEEDBACK = "Heuristic mode: tool calls not evaluated.";
const OMISSION_ID = /[A-Za-z0-9_-]{1,12}:omitted-[A-Za-z0-9_-]{22}/gu;

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
    const v2 = {
      session: {
        messages: async () => ({ data: messages }),
      },
    } as unknown as V2Client;

    const plan = await createTrimPlan(v2, "session", 1);

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
    const v2 = {
      session: {
        messages: async () => ({ data: messages }),
      },
    } as unknown as V2Client;

    const plan = await createTrimPlan(v2, "session", 1);

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

    try {
      await applyToolTrim(sessionID, [turn(tool)], new Set());

      expect(tool.state.status).toBe("completed");
      if (tool.state.status !== "completed")
        throw new Error("Expected completion");
      const contentID = /Content ID: ([^\s\]]+)\]$/u.exec(
        tool.state.output,
      )?.[1];
      expect(contentID).toMatch(
        new RegExp(`^${sessionID.slice(-12)}:omitted-[A-Za-z0-9_-]{22}$`, "u"),
      );
      expect(tool.state.output).toBe(
        `[Historical read result omitted (3 bytes); Content ID: ${contentID}]`,
      );
      expect(await readOmittedContent(sessionID, contentID!)).toBe("\u20ac");
    } finally {
      await rm(cachePath(sessionID), { force: true });
    }
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

      try {
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
        const cache = await readCache(sessionID);
        expect(Object.keys(cache!.entries)).toEqual([contentID!]);

        const before = structuredClone(tool.state);
        expect(await applyToolTrim(sessionID, [turn(tool)], new Set())).toBe(0);
        expect(tool.state).toEqual(before);
        expect(await readCache(sessionID)).toEqual(cache);
      } finally {
        await rm(cachePath(sessionID), { force: true });
      }
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

      try {
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
        const cache = await readCache(sessionID);
        expect(Object.keys(cache!.entries).sort()).toEqual(
          [promptID, reportID].sort(),
        );
        const before = structuredClone(tool.state);
        expect(await applyToolTrim(sessionID, [turn(tool)], new Set())).toBe(0);
        expect(tool.state).toEqual(before);
        expect(await readCache(sessionID)).toEqual(cache);
      } finally {
        await rm(cachePath(sessionID), { force: true });
      }
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

    try {
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
      expect(Object.keys((await readCache(sessionID))!.entries)).toEqual([
        contentID,
      ]);
    } finally {
      await rm(cachePath(sessionID), { force: true });
    }
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
    expect(await readCache(sessionID)).toBeNull();
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
    expect(await readCache(sessionID)).toBeNull();
  });

  test("recovery references cost fewer tokens than the notices they replaced", async () => {
    const contentID = "2c4f6a8b0d1e:omitted-yH8kQ2pL5vN7sR4tU6wXzA";
    const child = "ses_26c3e8f4bffeKXs0ovRyDPq4tw";
    const report = "report ".repeat(700);
    const listing = "line of output ".repeat(300);
    const read = "x".repeat(2_000);
    const bytes = (text: string) => Buffer.byteLength(text);
    const outputNotice = (description: string, text: string) =>
      `<tool-output-omission-notice>\n${description}\n\nOutput Length: ${bytes(text)} bytes\nContent ID: ${contentID}\n</tool-output-omission-notice>`;
    const inputNotice = (description: string, text: string) =>
      `<tool-input-omission-notice>\n${description}\n\nOmitted Length: ${bytes(text)} bytes\nContent ID: ${contentID}\n</tool-input-omission-notice>\n\n`;
    const plan = async (
      category: ToolRuleCategory,
      input: Record<string, unknown>,
      output: string,
    ) =>
      (await planToolTrim(
        { category, input, output, trimmed: false, childSessionID: child },
        async () => contentID,
      ))!;
    // Each row pairs the model-facing text the XML notices and bare
    // placeholders used for an omission with the text the references use.
    const write = await plan("write", { content: longText }, "ok");
    const edit = await plan(
      "edit",
      { oldString: longText, newString: "small" },
      "ok",
    );
    const task = await plan("subagent", { prompt: longText }, report);
    const batch: Array<[string, string]> = [
      [
        [
          "[Omitted]",
          inputNotice(
            "File write contents omitted due to a compaction operation. If necessary, reread file to see current contents.",
            longText,
          ),
        ].join(""),
        write.input["content"] as string,
      ],
      [
        [
          "[Omitted][Omitted]",
          inputNotice(
            "File edit oldString and newString omitted due to compaction operation. If necessary, reread file to see current contents.",
            `${longText}\nsmall`,
          ),
        ].join(""),
        `${edit.input["oldString"]}${edit.input["newString"]}`,
      ],
      [
        [
          `[Historical task prompt omitted; Content ID: ${contentID}. Retrieve the original with read_omitted_content before reuse.]`,
          inputNotice(
            "Task prompt omitted due to compaction operation. If necessary, reread it via read_omitted_content.",
            longText,
          ),
          outputNotice(
            "Task output omitted due to a compaction operation. If necessary, reread output via read_omitted_content tool.",
            report,
          ),
        ].join(""),
        `${task.input["prompt"]}${task.output}`,
      ],
      [
        outputNotice(
          "Historical read contents omitted due to compaction operation. Retrieve the exact captured result with read_omitted_content; reread the file for current contents.",
          read,
        ),
        (await plan("read", {}, read)).output!,
      ],
      [
        outputNotice("Output omitted due to a compaction operation.", listing),
        (await plan("other", {}, listing)).output!,
      ],
    ];

    for (const [before, after] of batch) {
      expect(countGptTokens(after)).toBeLessThan(countGptTokens(before));
    }
    const before = batch.reduce((n, [text]) => n + countGptTokens(text), 0);
    const after = batch.reduce((n, [, text]) => n + countGptTokens(text), 0);
    expect(after).toBeLessThan(before * 0.65);
  });

  test("keeps output rules marker-guarded for tools marked by earlier trims", async () => {
    const sessionID = `ses_trim_marked_output_${randomUUID().replaceAll("-", "")}`;
    const tool = readTool("tool_marked_output", "large output ".repeat(100));
    tool.state.metadata["magicCompact"] = { trimmed: true };
    const before = structuredClone(tool.state);

    expect(await applyToolTrim(sessionID, [turn(tool)], new Set())).toBe(0);
    expect(tool.state).toEqual(before);
    expect(await readCache(sessionID)).toBeNull();
  });

  test("reports trim stats on a cycle-free active basis across the 999-to-1K boundary", async () => {
    const sessionID = `ses_trim_accounting_${randomUUID().replaceAll("-", "")}`;
    const tool = readTool("tool_accounting", "historical output ".repeat(70));
    const messages = [
      message("usr_accounting", "user", []),
      message("ast_accounting", "assistant", [tool]),
    ];
    const harness = v1CommandHarness(sessionID, messages);
    const original = structuredClone(messages);

    try {
      const sourceStats = {
        version: 1 as const,
        sourceSessionId: null,
        rootSessionId: "root-session",
        totalTokensPruned: 999,
        cachedTokensSaved: 11,
        processedAssistantMessageIds: [],
      };
      await writeStats(sessionID, sourceStats);
      const beforeTokens = countCanonicalContentTokens(messages);

      await expect(
        executeMagicTrim(harness.v2, sessionID, 0, HEURISTIC_OPTIONS),
      ).resolves.not.toBeNull();

      expect(messages).toEqual(original);
      expect(harness.sourceWrites).toEqual([]);
      expect(await readStats(sessionID)).toEqual(sourceStats);
      const successor = harness.successorMessages();
      const afterTokens = countCanonicalContentTokens(successor);
      const stats = await readStats(harness.successorID);
      const notice = successor
        .at(-1)!
        .parts.find((part): part is TextPart => part.type === "text")!;

      expect(stats).toMatchObject({
        sourceSessionId: sessionID,
        rootSessionId: "root-session",
        totalTokensPruned: 999 + beforeTokens - afterTokens,
        cachedTokensSaved: 11,
        processedAssistantMessageIds: [successor[1]!.info.id],
      });
      expect(notice.text).toBe(
        trimStatsMessage(beforeTokens, afterTokens, 1, HEURISTIC_FEEDBACK),
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
    } finally {
      harness.history.close();
      await cleanupSessions(sessionID, harness.successorID);
    }
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

    try {
      await allocateOmission(sessionID, { content: "source cache" });
      await writeStats(sessionID, trimStats(sessionID));
      const cache = await readCache(sessionID);

      await expect(
        executeMagicTrim(harness.v2, sessionID, 0, HEURISTIC_OPTIONS),
      ).resolves.toBeNull();

      expect(harness.created).toEqual([]);
      expect(harness.sourceWrites).toEqual([]);
      expect(await readCache(sessionID)).toEqual(cache);
      expect(await readStats(sessionID)).toEqual(trimStats(sessionID));
      expect(harness.toasts.at(-1)?.message).toBe(
        `${TRIM_NOOP} ${HEURISTIC_FEEDBACK}`,
      );
    } finally {
      harness.history.close();
      await cleanupSessions(sessionID);
    }
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

    try {
      await expect(
        executeMagicTrim(harness.v2, sessionID, 0, HEURISTIC_OPTIONS),
      ).rejects.toThrow("The session changed while Magic Trim was preparing");
      expect(harness.sourceWrites).toEqual([]);
      expect(messages.map(item => item.info.id)).toEqual([
        "usr_1",
        "ast_1",
        "usr_late",
      ]);
      expect(harness.deletedSessions).toEqual([harness.successorID]);
      expect(await readCache(harness.successorID)).toBeNull();
      expect(await readStats(harness.successorID)).toBeNull();
      expect(harness.toasts.at(-1)?.message).toContain(
        "The session changed while Magic Trim was preparing",
      );
    } finally {
      harness.history.close();
      await cleanupSessions(sessionID, harness.successorID);
    }
  });

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

    try {
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
    } finally {
      harness.history.close();
      await cleanupSessions(sessionID);
    }
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
    const v2 = {
      session: { messages: async () => ({ data: messages }) },
      part: { update: async () => ({ data: legacyTool }) },
    } as unknown as V2Client;

    const compaction = await createCompactionPlan(v2, "session", 0);
    const trim = await createTrimPlan(v2, "session", 0);
    await applyToolTrim("session", trim.trimmedTurns, new Set());

    expect(
      compaction.summarizedTurns.flatMap(turn =>
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
    const source = [
      message("usr_first", "user", [
        userText("usr_first", "Keep the exact user request."),
      ]),
      ...Array.from({ length: 6 }, (_, index) =>
        message(
          `ast_${index}`,
          "assistant",
          index === 5 ? [readTool("read_old", "obsolete listing")] : [],
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
    try {
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
          text: "Keep the exact user request.",
        });
        expect(source[0]).toEqual(originalUser);
        expect((successor[6]!.info as { parentID: string }).parentID).toBe(
          successor[0]!.info.id,
        );
      }
      expect(
        harness.modelCalls.filter(call => call === "summary"),
      ).toHaveLength(2);
    } finally {
      harness.history.close();
      await cleanupSessions(sessionID, ...harness.created);
    }
  });
  // /magic-compact and /magic-trim share one retention grammar; each has its
  // own fixed argument errors.
  describe.each([
    [
      "magic-compact",
      parseMagicCompactArguments,
      COMPACT_ARGUMENTS_ERROR,
      COMPACT_THRESHOLD_ERROR,
      COMPACT_BACKEND_ERROR,
    ],
    [
      "magic-trim",
      parseMagicTrimArguments,
      TRIM_ARGUMENTS_ERROR,
      TRIM_THRESHOLD_ERROR,
      TRIM_BACKEND_ERROR,
    ],
  ] as const)(
    "%s arguments",
    (command, parse, ARGUMENTS_ERROR, THRESHOLD_ERROR, BACKEND_ERROR) => {
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
        [
          "2  --heuristic",
          { keepTurns: 2, heuristic: true, matchModel: false },
        ],
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
        expect(parse(raw)).toEqual(expected);
      });

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
      ] as const)("rejects %p", (raw, kind) => {
        expect(() => parse(raw)).toThrow(
          kind === "arguments"
            ? ARGUMENTS_ERROR
            : kind === "threshold"
              ? THRESHOLD_ERROR
              : BACKEND_ERROR,
        );
      });

      // The parser table above owns every form. Here one grammar error and
      // one threshold error reach this command's host handler, which stops
      // before it reads the environment or looks for a server.
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

    try {
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
    } finally {
      await cleanupSessions(sessionID, ...harness.created);
    }
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
      const before = structuredClone(messages);
      try {
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
        expect(messages).toEqual(before);
        expect(harness.sourceWrites).toEqual([]);
        if (keep) {
          expect(harness.created).toEqual([]);
          expect(harness.deletedSessions).toEqual([harness.summaryForkID]);
          expect(await readStats(sessionID)).toBeNull();
        } else {
          const notice = harness.successorMessages().at(-1)!
            .parts[0] as TextPart;
          expect(notice.ignored).toBeTrue();
          expect(notice.text).toContain(`Retention | ${feedback}`);
          expect(notice.text).toContain("Changes | Tool calls trimmed: 1");
        }
      } finally {
        await cleanupSessions(sessionID, ...harness.created);
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

    try {
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
    } finally {
      await cleanupSessions(sessionID, ...harness.created);
    }
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
    try {
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
    } finally {
      await cleanupSessions(sessionID, ...harness.created);
    }
  });

  test("heuristic mode trims candidates without a key, model, or classifier", async () => {
    const sessionID = `ses_trim_heuristic_${randomUUID().replaceAll("-", "")}`;
    const tool = readTool("tool_read", "large output ".repeat(100));
    const harness = v1CommandHarness(sessionID, [
      message("usr_1", "user", []),
      message("ast_1", "assistant", [tool]),
    ]);

    try {
      await expect(
        executeMagicTrim(harness.v2, sessionID, 0, HEURISTIC_OPTIONS),
      ).resolves.not.toBeNull();

      expect(harness.modelCalls).toEqual([]);
      expect(harness.forks).toEqual([]);
      const trimmed = harness.successorPart("tool_read");
      if (trimmed.state.status !== "completed") throw new Error("completed");
      expect(trimmed.state.output).toContain("Historical read result omitted");
    } finally {
      await cleanupSessions(sessionID, ...harness.created);
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
    const storedBefore = countCanonicalContentTokens(messages);

    try {
      expect(
        await executeMagicTrim(harness.v2, sessionID, 0, HEURISTIC_OPTIONS),
      ).not.toBeNull();

      const successor = harness.successorMessages();
      const notice = successor
        .at(-1)!
        .parts.find((part): part is TextPart => part.type === "text")!;
      const storedAfter = countCanonicalContentTokens(successor);
      expect(storedAfter).toBeLessThan(storedBefore);
      expect(notice.text).toBe(
        trimStatsMessage(storedBefore, storedAfter, 2, HEURISTIC_FEEDBACK),
      );
      expect((await readStats(harness.successorID))?.totalTokensPruned).toBe(
        storedBefore - storedAfter,
      );
    } finally {
      await cleanupSessions(sessionID, ...harness.created);
    }
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
      const before = structuredClone(messages);
      let failure: unknown;

      try {
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
        expect(messages).toEqual(before);
        expect(harness.created).toEqual([]);
        expect(harness.deletedSessions).toEqual(
          apiKey ? [harness.summaryForkID] : [],
        );
        expect(harness.modelCalls).toEqual(
          apiKey ? ["agents", "providers", "summary"] : [],
        );
        expect(harness.sourceWrites).toEqual([]);
        expect(await readCache(sessionID)).toBeNull();
        expect(await readStats(sessionID)).toBeNull();
        expect(harness.toasts.at(-1)?.message).toContain(
          "/magic-trim --heuristic",
        );
      } finally {
        await cleanupSessions(sessionID, ...harness.created);
      }
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
    try {
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
    } finally {
      await cleanupSessions(sessionID, ...harness.created);
    }
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
    const before = structuredClone(messages);
    let calls = 0;
    try {
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
      expect(messages).toEqual(before);
      expect(harness.created).toEqual([]);
      expect(await readStats(sessionID)).toBeNull();
    } finally {
      await cleanupSessions(sessionID, ...harness.created);
    }
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
      const before = structuredClone(messages);
      let failure: unknown;

      try {
        try {
          await executeMagicTrim(harness.v2, sessionID, 0, options);
        } catch (error) {
          failure = error;
        }

        expect(failure).toBeInstanceOf(SourceNotAtRestError);
        expect(String(failure)).toEndWith(
          "Record: assistant message ast_old; running task call call_tool_stale.",
        );
        expect(messages).toEqual(before);
        expect(harness.created).toEqual([]);
        expect(harness.forks).toEqual([]);
        expect(harness.modelCalls).toEqual([]);
        expect(harness.sourceWrites).toEqual([]);
        // The toast shows the refusal itself, without the error's class name.
        expect(harness.toasts.at(-1)?.message).toBe((failure as Error).message);
        expect(await readCache(sessionID)).toBeNull();
        expect(await readStats(sessionID)).toBeNull();
      } finally {
        await cleanupSessions(sessionID, ...harness.created);
      }
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
      const toolBefore = structuredClone(tool);

      try {
        await expect(
          executeMagicTrim(harness.v2, sessionID, 0, {
            retention: {
              mode: "task-aware",
              apiKey: "test-key",
              fetch: scoredJevFetch(() => 0.1),
            },
          }),
        ).rejects.toThrow(expected);

        expect(tool).toEqual(toolBefore);
        expect(harness.deletedSessions).toEqual([
          harness.summaryForkID,
          harness.successorID,
        ]);
        expect(harness.sourceWrites).toEqual([]);
        expect(await readCache(sessionID)).toBeNull();
        expect(await readStats(sessionID)).toBeNull();
        expect(await readStats(harness.successorID)).toBeNull();
      } finally {
        await cleanupSessions(sessionID, ...harness.created);
      }
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

// The stored copy of a tool part; the host never edits the caller's objects.

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

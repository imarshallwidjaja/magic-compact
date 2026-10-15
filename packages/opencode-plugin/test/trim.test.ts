import { describe, expect, test } from "bun:test";
import { createHash, randomUUID } from "node:crypto";
import { rm } from "node:fs/promises";
import type {
  Message,
  Part,
  Session,
  TextPart,
  ToolPart,
} from "@opencode-ai/sdk/v2";
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
  TRIM_ARGUMENTS_ERROR,
  TRIM_NOOP,
  executeMagicTrim,
  parseMagicTrimArguments,
} from "../src/magic-trim";
import type { PluginInput } from "@opencode-ai/plugin";
import { RetentionClassificationError } from "../src/retention/classifier";
import {
  summarizerAgent,
  persistedForkHistory,
  providerCatalog,
  scoredJevFetch,
  SESSION_SUMMARY_TEXT,
  summaryReply,
  type JevCall,
} from "./retention-host";
import {
  allocateOmission,
  cachePath,
  commitPlannedOmission,
  deleteCache,
  readCache,
  readOmittedContent,
} from "../src/storage/omission";
import {
  type ConversationStats,
  deleteStats,
  readStats,
  writeStats,
} from "../src/storage/stats";
import { trimStatsMessage } from "../src/stats/constants";
import { countCanonicalContentTokens } from "../src/stats/tokenize";

const HEURISTIC_OPTIONS = { retention: { mode: "heuristic" } } as const;
const HEURISTIC_FEEDBACK = "Heuristic mode: tool calls not evaluated.";

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
      expect(tool.state.output).toContain(
        `Content ID: ${sessionID.slice(-12)}:omitted-`,
      );
      expect(tool.state.output).toContain("Output Length: 3 bytes");
      expect(tool.state.output).toContain("Historical read contents omitted");
      expect(tool.state.output).toContain("exact captured result");
      expect(tool.state.output).toContain(
        "reread the file for current contents",
      );
      expect(tool.state.output).not.toContain("Stale");
    } finally {
      await rm(cachePath(sessionID), { force: true });
    }
  });

  test.each([
    [
      "short output",
      "delegation context packet ".repeat(60),
      "subagent result",
    ],
    [
      "large output",
      "delegation context packet ".repeat(60),
      "report ".repeat(700),
    ],
    ["word-limit prompt", "x ".repeat(129), "subagent result"],
  ])(
    "recovers task prompts with %s and trims only once",
    async (_name, prompt, output) => {
      const sessionID = `ses_trim_task_input_${randomUUID().replaceAll("-", "")}`;
      const tool = taskTool("tool_task_input", prompt);
      tool.state.output = output;

      try {
        await applyToolTrim(sessionID, [turn(tool)], new Set());

        expect(tool.state.status).toBe("completed");
        if (tool.state.status !== "completed")
          throw new Error("Expected completion");
        expect(tool.state.input["description"]).toBe("delegated work");
        expect(tool.state.input["subagent_type"]).toBe("forager-worker");
        expect(
          tool.state.output.startsWith("<tool-input-omission-notice>"),
        ).toBeTrue();
        expect(tool.state.output).toContain(
          "Task prompt omitted due to compaction operation.",
        );
        expect(tool.state.output).toContain(
          `Content ID: ${sessionID.slice(-12)}:omitted-`,
        );
        const cache = await readCache(sessionID);
        const contentIDs = tool.state.output.match(
          /[A-Za-z0-9_-]{1,12}:omitted-[A-Za-z0-9_-]{22}/gu,
        )!;
        const contentID = contentIDs[0]!;
        const placeholder = tool.state.input["prompt"] as string;
        expect(placeholder).toBe(
          `[Historical task prompt omitted; Content ID: ${contentID}. Retrieve the original with read_omitted_content before reuse.]`,
        );
        expect(Buffer.byteLength(placeholder)).toBeLessThan(200);
        expect(placeholder.length).toBeLessThan(1024);
        expect(placeholder.split(/\s+/u).length).toBeLessThan(128);
        expect(await readOmittedContent(sessionID, contentID)).toBe(prompt);
        if (output.length > 4096) {
          expect(contentIDs).toHaveLength(2);
          expect(contentIDs[1]).not.toBe(contentID);
          expect(await readOmittedContent(sessionID, contentIDs[1]!)).toBe(
            output,
          );
          expect(tool.state.output).toContain("<tool-output-omission-notice>");
        } else {
          expect(contentIDs).toHaveLength(1);
          expect(tool.state.output.endsWith(output)).toBeTrue();
        }
        expect(Object.keys(cache!.entries)).toEqual(contentIDs);
        const before = structuredClone(tool.state);
        expect(await applyToolTrim(sessionID, [turn(tool)], new Set())).toBe(0);
        expect(tool.state).toEqual(before);
        expect(await readCache(sessionID)).toEqual(cache);
      } finally {
        await rm(cachePath(sessionID), { force: true });
      }
    },
  );

  test("keeps short task prompts unchanged", async () => {
    const sessionID = `ses_trim_task_short_${randomUUID().replaceAll("-", "")}`;
    const tool = taskTool("tool_task_short", "Check the failing suite.");

    try {
      await applyToolTrim(sessionID, [turn(tool)], new Set());

      expect(tool.state.status).toBe("completed");
      if (tool.state.status !== "completed")
        throw new Error("Expected completion");
      expect(tool.state.input).toEqual({
        subagent_type: "forager-worker",
        description: "delegated work",
        prompt: "Check the failing suite.",
      });
      expect(tool.state.output).toBe("subagent result");
      expect(await readCache(sessionID)).toBeNull();
    } finally {
      await rm(cachePath(sessionID), { force: true });
    }
  });

  test("applies idempotent input rules to tools marked by earlier trims", async () => {
    const sessionID = `ses_trim_marked_input_${randomUUID().replaceAll("-", "")}`;
    const prompt = "delegation context packet ".repeat(60);
    const tool = taskTool("tool_marked_input", prompt);
    const output = "already processed task output ".repeat(200);
    tool.state.output = output;
    tool.state.metadata["magicCompact"] = { trimmed: true };

    try {
      expect(await applyToolTrim(sessionID, [turn(tool)], new Set())).toBe(1);

      expect(tool.state.status).toBe("completed");
      if (tool.state.status !== "completed")
        throw new Error("Expected completion");
      expect(tool.state.input["description"]).toBe("delegated work");
      expect(tool.state.input["subagent_type"]).toBe("forager-worker");
      expect(tool.state.metadata["magicCompact"]).toEqual({ trimmed: true });
      expect(
        tool.state.output.startsWith("<tool-input-omission-notice>"),
      ).toBeTrue();
      expect(tool.state.output).toContain(
        "Task prompt omitted due to compaction operation.",
      );
      const contentID = tool.state.output.match(
        /[A-Za-z0-9_-]{1,12}:omitted-[A-Za-z0-9_-]{22}/u,
      )?.[0];
      expect(contentID).toBeDefined();
      const placeholder = tool.state.input["prompt"];
      expect(placeholder).toBe(
        `[Historical task prompt omitted; Content ID: ${contentID}. Retrieve the original with read_omitted_content before reuse.]`,
      );
      expect(await readOmittedContent(sessionID, contentID!)).toBe(prompt);
      expect(tool.state.output.endsWith(output)).toBeTrue();
      expect(tool.state.output).not.toContain("<tool-output-omission-notice>");
      const cache = await readCache(sessionID);
      expect(Object.keys(cache!.entries)).toEqual([contentID!]);

      expect(await applyToolTrim(sessionID, [turn(tool)], new Set())).toBe(0);
      expect(tool.state.input["prompt"]).toBe(placeholder);
      expect(await readCache(sessionID)).toEqual(cache);
      expect(
        tool.state.output.match(/<tool-input-omission-notice>/gu),
      ).toHaveLength(1);
    } finally {
      await rm(cachePath(sessionID), { force: true });
    }
  });

  test("preserves earlier bare task placeholders without migrating them", async () => {
    const sessionID = `ses_trim_old_task_${randomUUID().replaceAll("-", "")}`;
    const tool = taskTool("tool_old_task", "[Omitted]");
    tool.state.metadata["magicCompact"] = { trimmed: true };
    const before = structuredClone(tool.state);

    expect(await applyToolTrim(sessionID, [turn(tool)], new Set())).toBe(0);
    expect(tool.state).toEqual(before);
    expect(await readCache(sessionID)).toBeNull();
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
    const harness = trimCommandHarness(sessionID, messages);
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
      await cleanupTrimSessions(sessionID, harness.successorID);
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
    const harness = trimCommandHarness(sessionID, messages);

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
      await cleanupTrimSessions(sessionID);
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
    const harness = trimCommandHarness(sessionID, messages, {
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
      await cleanupTrimSessions(sessionID, harness.successorID);
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
    const harness = trimCommandHarness(sessionID, messages, {
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
      await cleanupTrimSessions(sessionID);
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
    const harness = trimCommandHarness(sessionID, source);
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
      await cleanupTrimSessions(sessionID, ...harness.created);
    }
  });
  test("parses an optional turn count and --heuristic flag in either order", () => {
    expect(parseMagicTrimArguments("")).toEqual({
      keepTurns: 0,
      heuristic: false,
    });
    expect(parseMagicTrimArguments(" 3 ")).toEqual({
      keepTurns: 3,
      heuristic: false,
    });
    expect(parseMagicTrimArguments("--heuristic")).toEqual({
      keepTurns: 0,
      heuristic: true,
    });
    expect(parseMagicTrimArguments("--heuristic 2")).toEqual({
      keepTurns: 2,
      heuristic: true,
    });
    expect(parseMagicTrimArguments("2  --heuristic")).toEqual({
      keepTurns: 2,
      heuristic: true,
    });
  });

  test.each([
    "--heuristic --heuristic",
    "1 2",
    "-1",
    "1.5",
    "--Heuristic",
    "--heuristic=true",
    "heuristic",
  ])("rejects malformed arguments %p before any work", async raw => {
    expect(() => parseMagicTrimArguments(raw)).toThrow(TRIM_ARGUMENTS_ERROR);
    const hooks = await server({} as PluginInput);
    await expect(
      hooks["command.execute.before"]!(
        { command: "magic-trim", arguments: raw, sessionID: "session" },
        { parts: [] },
      ),
    ).rejects.toThrow(TRIM_ARGUMENTS_ERROR);
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
    const harness = trimCommandHarness(sessionID, messages);
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
      expect(trimmed.state.output).toContain(
        "Historical read contents omitted",
      );
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
        `Trimmed 2 tool call(s) into successor session ${harness.successorID}. Task-aware retention kept 1 of 2 evaluated tool call(s) whole (read 1).`,
      );
      const notice = harness.successorMessages().at(-1)!.parts[0] as TextPart;
      expect(notice.ignored).toBeTrue();
      expect(notice.text.split("\n")).toHaveLength(4);
      expect(notice.text).toContain("Changes | Tool calls trimmed: 2");
      expect(notice.text).toContain(
        "Retention | Task-aware retention kept 1 of 2 evaluated tool call(s) whole (read 1).",
      );
    } finally {
      await cleanupTrimSessions(sessionID, ...harness.created);
    }
  });

  test.each([false, true])(
    "reports zero or all keeps without adding no-op notices (keep=%s)",
    async keep => {
      const sessionID = `ses_trim_keep_${randomUUID().replaceAll("-", "")}`;
      const messages = [
        message("usr_1", "user", [userText("usr_1", "Fix tests.")]),
        message("ast_1", "assistant", [
          readTool("tool_read", "large output ".repeat(100)),
        ]),
      ];
      const harness = trimCommandHarness(sessionID, messages);
      const before = structuredClone(messages);
      try {
        expect(
          await executeMagicTrim(harness.v2, sessionID, 0, {
            retention: {
              mode: "task-aware",
              apiKey: "test-key",
              fetch: scoredJevFetch(() => (keep ? 0.95 : 0.1), []),
            },
          }),
        ).toEqual(keep ? null : expect.anything());
        const feedback = `Task-aware retention kept ${keep ? 1 : 0} of 1 evaluated tool call(s) whole${keep ? " (read 1)" : ""}.`;
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
        await cleanupTrimSessions(sessionID, ...harness.created);
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
    const harness = trimCommandHarness(sessionID, messages);

    try {
      await expect(
        executeMagicTrim(harness.v2, sessionID, 0, {
          retention: { mode: "task-aware", apiKey: undefined },
        }),
      ).resolves.not.toBeNull();

      expect(harness.modelCalls).toEqual([]);
      expect(harness.forks).toEqual([]);
      expect(harness.toasts.at(-1)?.message).toBe(
        `Trimmed 1 tool call(s) into successor session ${harness.successorID}. Task-aware retention: no tool calls needed evaluation.`,
      );
      expect(
        (harness.successorMessages().at(-1)!.parts[0] as TextPart).text,
      ).toContain(
        "Retention | Task-aware retention: no tool calls needed evaluation.",
      );
    } finally {
      await cleanupTrimSessions(sessionID, ...harness.created);
    }
  });

  test("heuristic mode trims candidates without a key, model, or classifier", async () => {
    const sessionID = `ses_trim_heuristic_${randomUUID().replaceAll("-", "")}`;
    const tool = readTool("tool_read", "large output ".repeat(100));
    const harness = trimCommandHarness(sessionID, [
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
      expect(trimmed.state.output).toContain(
        "Historical read contents omitted",
      );
    } finally {
      await cleanupTrimSessions(sessionID, ...harness.created);
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
    const harness = trimCommandHarness(sessionID, messages);
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
      await cleanupTrimSessions(sessionID, ...harness.created);
    }
  });

  test.each([
    ["a missing API key", undefined, 200],
    ["a failed classification", "test-key", 500],
  ] as const)(
    "%s leaves the source, cache, and stats untouched without a successor",
    async (_name, apiKey, status) => {
      const sessionID = `ses_trim_retention_fail_${randomUUID().replaceAll("-", "")}`;
      const messages = [
        message("usr_1", "user", [userText("usr_1", "Add the parser.")]),
        message("ast_1", "assistant", [
          readTool("tool_read", "large output ".repeat(100)),
        ]),
      ];
      const harness = trimCommandHarness(sessionID, messages);
      const before = structuredClone(messages);
      let failure: unknown;

      try {
        try {
          await executeMagicTrim(harness.v2, sessionID, 0, {
            retention: {
              mode: "task-aware",
              apiKey,
              fetch: async () => new Response("private", { status }),
            },
          });
        } catch (error) {
          failure = error;
        }

        expect(failure).toBeInstanceOf(RetentionClassificationError);
        expect(String(failure)).toContain("/magic-trim --heuristic");
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
        await cleanupTrimSessions(sessionID, ...harness.created);
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
      const harness = trimCommandHarness(sessionID, messages, {
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
        await cleanupTrimSessions(sessionID, ...harness.created);
      }
    },
  );
});

const PRIVATE_BODY = "PRIVATE-SESSION-BODY-7f3a";

// Every string reachable from an error: messages, stacks, causes, aggregate
// members, and any other own property.

type TrimCommandHarnessOptions = {
  toastMode?: "hang" | "fail";
  sessionRevert?: { messageID: string };
  // Runs while the retention summary request is in flight.
  duringSummary?: () => void;
  // Runs once the successor session exists, before its final source recheck.
  duringSuccessorCreate?: () => void;
  // Replaces source message reads after this many successful reads.
  failSourceReadAfter?: number;
};

// Fake OpenCode host for trim command tests. `messages` is the live source;
// the retention summary fork and successors live in the persisted host fake.
function trimCommandHarness(
  sessionID: string,
  messages: MessageWithParts[],
  options: TrimCommandHarnessOptions = {},
) {
  messages.forEach((item, index) => {
    item.info.sessionID = sessionID;
    item.info.time ??= { created: index + 1 };
    if (item.info.role === "assistant")
      (item.info.time as { completed?: number }).completed ??=
        item.info.time.created;
    for (const part of item.parts) {
      part.sessionID = sessionID;
      part.messageID = item.info.id;
    }
  });
  const history = persistedForkHistory([]);
  const successorIDs: string[] = [];
  const summaryForkID = `${sessionID}_summary`;
  const sourceSession = {
    id: sessionID,
    title: "Trim source",
    directory: "/workspace",
    time: { created: 1, updated: 1 },
  } as Session;
  const toasts: Array<{ message?: string }> = [];
  const deletedSessions: string[] = [];
  const forks: string[] = [];
  const modelCalls: string[] = [];
  const sourceWrites: string[] = [];
  const created: string[] = [];
  let sourceReads = 0;
  let revert = options.sessionRevert;
  const sessions = new Map<string, Session>();
  const v2 = {
    app: {
      agents: async () => {
        modelCalls.push("agents");
        return { data: [summarizerAgent()] };
      },
    },
    provider: {
      list: async () => {
        modelCalls.push("providers");
        return { data: providerCatalog() };
      },
    },
    permission: { list: async () => ({ data: [] }) },
    question: { list: async () => ({ data: [] }) },
    session: {
      messages: async (request: { sessionID: string }) => {
        if (request.sessionID !== sessionID)
          return { data: history.read(request.sessionID) };
        sourceReads += 1;
        if (
          options.failSourceReadAfter !== undefined
          && sourceReads > options.failSourceReadAfter
        ) {
          return { error: { data: { message: PRIVATE_BODY } } };
        }
        return { data: structuredClone(messages) };
      },
      get: async (request: { sessionID: string }) => ({
        data:
          request.sessionID === sessionID
            ? { ...sourceSession, ...(revert ? { revert } : {}) }
            : sessions.get(request.sessionID),
      }),
      status: async () => ({ data: {} }),
      children: async () => ({ data: [] }),
      fork: async (request: { messageID?: string }) => {
        forks.push(summaryForkID);
        history.delete(sessionID);
        messages.forEach((item, index) => history.write(item, index));
        history.delete(summaryForkID);
        history.fork(sessionID, summaryForkID, request.messageID);
        history.delete(sessionID);
        return { data: { ...sourceSession, id: summaryForkID } as Session };
      },
      create: async (request: Record<string, unknown>) => {
        const successorID = `${sessionID}_successor${created.length === 0 ? "" : `_${created.length + 1}`}`;
        successorIDs.push(successorID);
        created.push(successorID);
        const info = {
          ...sourceSession,
          id: successorID,
          title: String(request.title),
          metadata: request.metadata,
        } as Session;
        sessions.set(successorID, info);
        options.duringSuccessorCreate?.();
        return { data: info };
      },
      update: async (request: Record<string, unknown>) => {
        const id = String(request.sessionID);
        if (id === sessionID) sourceWrites.push("session.update");
        const info = {
          ...(sessions.get(id) ?? { ...sourceSession, id }),
          ...(request.title === undefined ? {} : { title: request.title }),
          ...(request.metadata === undefined
            ? {}
            : { metadata: request.metadata }),
        } as Session;
        sessions.set(id, info);
        return { data: info };
      },
      abort: async () => ({ data: true }),
      prompt: async (request: {
        sessionID: string;
        messageID?: string;
        agent?: string;
        parts?: Part[];
      }) => {
        if (request.sessionID !== summaryForkID) {
          sourceWrites.push("session.prompt");
          throw new Error("unexpected prompt");
        }
        modelCalls.push("summary");
        options.duringSummary?.();
        const fork = history.read(summaryForkID);
        const reply = summaryReply(fork, request, summaryForkID);
        fork
          .slice(-2)
          .forEach((item, index) => history.write(item, 100 + index));
        return reply;
      },
      delete: async (request: { sessionID: string }) => {
        if (request.sessionID === sessionID) sourceWrites.push("delete");
        deletedSessions.push(request.sessionID);
        history.delete(request.sessionID);
        return { data: true };
      },
    },
    part: {
      update: async () => {
        sourceWrites.push("part.update");
        throw new Error("unexpected part update");
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
    tui: {
      showToast: async (toast: { message?: string }) => {
        toasts.push(toast);
        if (options.toastMode === "hang") return new Promise(() => {});
        if (options.toastMode === "fail") throw new Error("toast failed");
        return { data: true };
      },
    },
  } as unknown as V2Client;

  return {
    v2,
    sessionID,
    // The most recent successor; IDs are reserved before any is created.
    get successorID() {
      return successorIDs.at(-1) ?? `${sessionID}_successor`;
    },
    summaryForkID,
    messages,
    toasts,
    deletedSessions,
    sourceWrites,
    forks,
    modelCalls,
    history,
    get created() {
      return [...created];
    },
    successorMessages() {
      return history.read(this.successorID);
    },
    successorInfo() {
      return sessions.get(this.successorID);
    },
    // Trim keeps every record in place, so a source part has the same
    // message and part position in the successor.
    successorPart(partID: string): ToolPart {
      const messageIndex = messages.findIndex(item =>
        item.parts.some(part => part.id === partID),
      );
      const partIndex = messages[messageIndex]!.parts.findIndex(
        part => part.id === partID,
      );
      return this.successorMessages()[messageIndex]!.parts[
        partIndex
      ] as ToolPart;
    },
    setRevert(value: { messageID: string }) {
      revert = value;
    },
  };
}

async function cleanupTrimSessions(...sessionIDs: string[]): Promise<void> {
  await Promise.all(
    sessionIDs.flatMap(sessionID => [
      deleteCache(sessionID),
      deleteStats(sessionID),
    ]),
  );
}

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

function userText(messageID: string, text: string): TextPart {
  return {
    id: `${messageID}_text`,
    sessionID: "session",
    messageID,
    type: "text",
    text,
  };
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

function completedTool(id: string, tool: string, output: string): ToolPart {
  return {
    id,
    sessionID: "session",
    messageID: "assistant",
    type: "tool",
    callID: `call_${id}`,
    tool,
    state: {
      status: "completed",
      input: {},
      output,
      title: tool,
      metadata: {},
      time: { start: 1, end: 2 },
    },
  };
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

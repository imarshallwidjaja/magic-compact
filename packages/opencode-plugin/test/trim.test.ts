import { describe, expect, test } from "bun:test";
import { randomUUID } from "node:crypto";
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
import { trimToolParts } from "../src/compact/prune";
import server from "../src/index";
import {
  TRIM_ARGUMENTS_ERROR,
  TRIM_NOOP,
  TRIM_PENDING_REVERT,
  TrimRevalidationError,
  executeMagicTrim,
  parseMagicTrimArguments,
} from "../src/magic-trim";
import type { PluginInput } from "@opencode-ai/plugin";
import { RetentionClassificationError } from "../src/retention/classifier";
import {
  compactionAgent,
  persistedForkHistory,
  providerCatalog,
  rewriteFork,
  scoredJevFetch,
  SESSION_SUMMARY_TEXT,
  summaryReply,
  type JevCall,
} from "./retention-host";
import {
  allocateOmission,
  cachePath,
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
    let updates = 0;
    const v2 = {
      part: {
        update: async () => {
          updates += 1;
          return { data: tool };
        },
      },
    } as unknown as V2Client;

    expect(
      await trimToolParts(
        { v2, sessionID: "session" },
        [selectedTurn],
        new Set(),
      ),
    ).toBe(1);
    expect(tool.state.status).toBe("completed");
    if (tool.state.status !== "completed") {
      throw new Error("Expected completed tool state.");
    }
    expect(tool.state.metadata["magicCompact"]).toEqual({ trimmed: true });
    expect(tool.state.output).toBe("Successfully updated todos.");
    expect(reasoning.text).toBe("unchanged");

    expect(
      await trimToolParts(
        { v2, sessionID: "session" },
        [selectedTurn],
        new Set(),
      ),
    ).toBe(0);
    expect(updates).toBe(1);
  });

  test("writes strict session-qualified omission IDs", async () => {
    const tool = readTool("tool_strict_id", "\u20ac");
    const sessionID = `ses_trim_${randomUUID().replaceAll("-", "")}`;
    const v2 = {
      part: { update: async () => ({ data: tool }) },
    } as unknown as V2Client;

    try {
      await trimToolParts({ v2, sessionID }, [turn(tool)], new Set());

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
      let updates = 0;
      const v2 = {
        part: {
          update: async () => {
            updates += 1;
            return { data: tool };
          },
        },
      } as unknown as V2Client;

      try {
        await trimToolParts({ v2, sessionID }, [turn(tool)], new Set());

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
        expect(
          await trimToolParts({ v2, sessionID }, [turn(tool)], new Set()),
        ).toBe(0);
        expect(updates).toBe(1);
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
    const v2 = {
      part: { update: async () => ({ data: tool }) },
    } as unknown as V2Client;

    try {
      await trimToolParts({ v2, sessionID }, [turn(tool)], new Set());

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
    let updates = 0;
    const v2 = {
      part: {
        update: async () => {
          updates += 1;
          return { data: tool };
        },
      },
    } as unknown as V2Client;

    try {
      expect(
        await trimToolParts({ v2, sessionID }, [turn(tool)], new Set()),
      ).toBe(1);

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

      expect(
        await trimToolParts({ v2, sessionID }, [turn(tool)], new Set()),
      ).toBe(0);
      expect(tool.state.input["prompt"]).toBe(placeholder);
      expect(updates).toBe(1);
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
    let updates = 0;
    const v2 = {
      part: {
        update: async () => {
          updates += 1;
          return { data: tool };
        },
      },
    } as unknown as V2Client;

    expect(
      await trimToolParts({ v2, sessionID }, [turn(tool)], new Set()),
    ).toBe(0);
    expect(tool.state).toEqual(before);
    expect(updates).toBe(0);
    expect(await readCache(sessionID)).toBeNull();
  });

  test("keeps output rules marker-guarded for tools marked by earlier trims", async () => {
    const sessionID = `ses_trim_marked_output_${randomUUID().replaceAll("-", "")}`;
    const tool = readTool("tool_marked_output", "large output ".repeat(100));
    tool.state.metadata["magicCompact"] = { trimmed: true };
    const before = structuredClone(tool.state);
    let updates = 0;
    const v2 = {
      part: {
        update: async () => {
          updates += 1;
          return { data: tool };
        },
      },
    } as unknown as V2Client;

    expect(
      await trimToolParts({ v2, sessionID }, [turn(tool)], new Set()),
    ).toBe(0);
    expect(tool.state).toEqual(before);
    expect(updates).toBe(0);
    expect(await readCache(sessionID)).toBeNull();
  });

  test("reports trim stats on a cycle-free active basis across the 999-to-1K boundary", async () => {
    const sessionID = `ses_trim_accounting_${randomUUID().replaceAll("-", "")}`;
    const tool = readTool("tool_accounting", "historical output ".repeat(70));
    const messages = [
      message("usr_accounting", "user", []),
      message("ast_accounting", "assistant", [tool]),
    ];
    const { v2, backupID } = trimCommandHarness(sessionID, messages);

    try {
      await writeStats(sessionID, {
        version: 1,
        sourceSessionId: null,
        rootSessionId: "root-session",
        totalTokensPruned: 999,
        cachedTokensSaved: 11,
        processedAssistantMessageIds: [],
      });
      const beforeTokens = countCanonicalContentTokens(messages);

      expect(
        await executeMagicTrim(v2, sessionID, 0, HEURISTIC_OPTIONS),
      ).toBeTrue();

      const activeMessages = messages.map(item => ({
        ...item,
        parts: item.parts.filter(
          part => part.type !== "text" || part.ignored !== true,
        ),
      }));
      const afterTokens = countCanonicalContentTokens(activeMessages);
      const stats = await readStats(sessionID);
      const notice = messages
        .at(-1)!
        .parts.find((part): part is TextPart => part.type === "text")!;

      expect(stats?.totalTokensPruned).toBe(999 + beforeTokens - afterTokens);
      expect(notice.text).toBe(
        trimStatsMessage(beforeTokens, afterTokens, 1, HEURISTIC_FEEDBACK),
      );
      expect(notice.ignored).toBeTrue();
      expect(notice.text).toContain(`Retention | ${HEURISTIC_FEEDBACK}`);
      expect(countCanonicalContentTokens(messages)).toBe(afterTokens);
      expect(notice.text).toContain("Magic Trim");
      expect(notice.text).toContain("Context | ");
      expect(notice.text).toContain("% trimmed (");
      expect(notice.text).toContain("Changes | Tool calls trimmed: 1");
      expect(notice.text).not.toContain("Cache Savings");
      expect(notice.text).not.toContain("Cost Savings");
      expect(notice.text).not.toContain("Inline image tokens projected");
      expect(notice.text).not.toContain("ignored reporting notice is excluded");
    } finally {
      await Promise.all([
        deleteCache(sessionID),
        deleteCache(backupID),
        deleteStats(sessionID),
        deleteStats(backupID),
      ]);
    }
  });

  test("deletes an initialized backup before reporting a no-op", async () => {
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

      await expect(
        executeMagicTrim(harness.v2, sessionID, 0, HEURISTIC_OPTIONS),
      ).resolves.toBeFalse();

      expect(harness.deletedSessions).toEqual([harness.backupID]);
      expect(await readCache(harness.backupID)).toBeNull();
      expect(await readStats(harness.backupID)).toBeNull();
      expect(harness.toasts.map(toast => toast.message)).toEqual([
        "Trimming session...",
        `${TRIM_NOOP} ${HEURISTIC_FEEDBACK}`,
      ]);
      expect(harness.prompts).toEqual([]);
      expect(JSON.stringify(messages)).not.toContain("Trimming session...");
    } finally {
      await cleanupTrimSessions(sessionID, harness.backupID);
    }
  });

  test("preserves a no-op source and copied caches when backup deletion fails", async () => {
    const sessionID = `ses_trim_noop_delete_${randomUUID().replaceAll("-", "")}`;
    const alreadyTrimmed = completedTool(
      "tool_noop_delete",
      "todowrite",
      "done",
    );
    alreadyTrimmed.state.metadata["magicCompact"] = { trimmed: true };
    const messages = [
      message("usr_noop_delete", "user", []),
      message("ast_noop_delete", "assistant", [alreadyTrimmed]),
    ];
    const before = structuredClone(messages);
    const harness = trimCommandHarness(sessionID, messages, {
      backupDeleteError: new Error("raw backup delete failure"),
    });
    let failure: unknown;

    try {
      await allocateOmission(sessionID, { content: "source cache" });
      await writeStats(sessionID, trimStats(sessionID));
      try {
        await executeMagicTrim(harness.v2, sessionID, 0, HEURISTIC_OPTIONS);
      } catch (error) {
        failure = error;
      }

      expect(failure).toBeInstanceOf(Error);
      expect(trimErrorMessages(failure)).toEqual(
        expect.arrayContaining([
          `Failed to delete backup session ${harness.backupID}.`,
          "raw backup delete failure",
        ]),
      );
      expect(harness.deletedSessions).toEqual([harness.backupID]);
      expect(harness.selectedSessions).toEqual([]);
      expect(harness.prompts).toEqual([]);
      expect(messages).toEqual(before);
      const sourceCache = await readCache(sessionID);
      expect(sourceCache).not.toBeNull();
      expect(await readCache(harness.backupID)).toEqual(sourceCache);
      expect(await readStats(sessionID)).toEqual(trimStats(sessionID));
      expect(await readStats(harness.backupID)).toMatchObject({
        sourceSessionId: sessionID,
        rootSessionId: sessionID,
      });
      expect(harness.toasts.at(-1)?.message).toContain(harness.backupID);
    } finally {
      await cleanupTrimSessions(sessionID, harness.backupID);
    }
  });

  test.each(["hang", "fail"] as const)(
    "toast %s preserves trim no-op, success, failure, and backup recovery outcomes",
    async toastMode => {
      const suffix = `${toastMode}_${randomUUID().replaceAll("-", "")}`;
      const noOpID = `ses_trim_toast_noop_${suffix}`;
      const noOpTool = completedTool("tool_noop", "todowrite", "done");
      noOpTool.state.metadata["magicCompact"] = { trimmed: true };
      const noOpMessages = [
        message("usr_noop", "user", []),
        message("ast_noop", "assistant", [noOpTool]),
      ];
      const noOp = trimCommandHarness(noOpID, noOpMessages, { toastMode });

      const successID = `ses_trim_toast_success_${suffix}`;
      const successMessages = [
        message("usr_success", "user", []),
        message("ast_success", "assistant", [
          readTool("tool_success", "large output ".repeat(100)),
        ]),
      ];
      const success = trimCommandHarness(successID, successMessages, {
        toastMode,
      });

      const failureID = `ses_trim_toast_failure_${suffix}`;
      const failureMessages = [
        message("usr_failure", "user", []),
        message("ast_failure", "assistant", [
          readTool("tool_failure", "large output ".repeat(100)),
        ]),
      ];
      const failed = trimCommandHarness(failureID, failureMessages, {
        toastMode,
        partUpdateError: new Error("trim update failed"),
      });

      try {
        await expect(
          settleWithin(
            executeMagicTrim(noOp.v2, noOpID, 0, {
              ...HEURISTIC_OPTIONS,
              controlTimeoutMs: 5,
            }),
            250,
          ),
        ).resolves.toBeFalse();
        expect(noOp.toasts.map(toast => toast.message)).toEqual([
          "Trimming session...",
          `${TRIM_NOOP} ${HEURISTIC_FEEDBACK}`,
        ]);

        await expect(
          settleWithin(
            executeMagicTrim(success.v2, successID, 0, {
              ...HEURISTIC_OPTIONS,
              controlTimeoutMs: 5,
            }),
            250,
          ),
        ).resolves.toBeTrue();
        expect(success.toasts.map(toast => toast.message)).toEqual([
          "Trimming session...",
          `Trimmed 1 tool call(s). ${HEURISTIC_FEEDBACK}`,
        ]);

        await expect(
          settleWithin(
            executeMagicTrim(failed.v2, failureID, 0, {
              ...HEURISTIC_OPTIONS,
              controlTimeoutMs: 5,
            }),
            250,
          ),
        ).rejects.toThrow("trim update failed");
        expect(failed.selectedSessions).toEqual([failed.backupID]);
        expect(failed.deletedSessions).toEqual([failureID]);
        expect(failed.toasts.map(toast => toast.message)).toEqual([
          "Trimming session...",
          "Error: trim update failed",
        ]);

        for (const harness of [noOp, success, failed]) {
          expect(JSON.stringify(harness.messages)).not.toContain(
            "Trimming session...",
          );
          expect(JSON.stringify(harness.prompts)).not.toContain(
            "Trimming session...",
          );
          expect(
            JSON.stringify(await readCache(harness.sessionID)),
          ).not.toContain("Trimming session...");
        }
      } finally {
        await Promise.all([
          cleanupTrimSessions(noOpID, noOp.backupID),
          cleanupTrimSessions(successID, success.backupID),
          cleanupTrimSessions(failureID, failed.backupID),
        ]);
      }
    },
  );

  test.each([false, true])(
    "preserves the root error and terminal feedback when trim recovery fails (timestamp-disagreeing history: %s)",
    async persisted => {
      const sessionID = `ses_trim_recovery_${randomUUID().replaceAll("-", "")}`;
      const messages = [
        message("usr_recovery", "user", []),
        message("ast_recovery", "assistant", [
          readTool("tool_recovery", "large output ".repeat(100)),
        ]),
      ];
      if (persisted)
        messages.forEach((item, index) => {
          item.info.sessionID = sessionID;
          item.info.time = { created: index === 0 ? 7 : 1 };
          if (item.info.role === "assistant")
            item.info.parentID = messages[0]!.info.id;
          item.parts.forEach(part => {
            part.sessionID = sessionID;
            part.messageID = item.info.id;
          });
        });
      const harness = trimCommandHarness(sessionID, messages, {
        persisted,
        partUpdateError: new Error("trim root failure"),
        selectError: new Error("raw select failure"),
      });
      let failure: unknown;

      try {
        try {
          await executeMagicTrim(harness.v2, sessionID, 0, HEURISTIC_OPTIONS);
        } catch (error) {
          failure = error;
        }

        expect(failure).toBeInstanceOf(AggregateError);
        expect(trimErrorMessages(failure)).toEqual(
          expect.arrayContaining([
            "trim root failure",
            "Magic Trim automatic recovery failed. Inspect source session "
              + `${sessionID} and backup session ${harness.backupID}, then recover manually.`,
          ]),
        );
        expect(harness.deletedSessions).toEqual([]);
        expect(await readCache(harness.backupID)).not.toBeNull();
        expect(await readStats(harness.backupID)).not.toBeNull();
        expect(harness.toasts).toHaveLength(2);
        expect(harness.toasts[0]?.message).toBe("Trimming session...");
        expect(harness.toasts[1]?.message).toContain("trim root failure");
        expect(harness.toasts[1]?.message).toContain(sessionID);
        expect(harness.toasts[1]?.message).toContain(harness.backupID);
        if (persisted) {
          const backup = harness.history!.read(harness.backupID);
          expect(backup.map(item => item.info.role)).toEqual([
            "user",
            "assistant",
          ]);
          expect((backup[1]!.info as { parentID: string }).parentID).toBe(
            backup[0]!.info.id,
          );
        }
      } finally {
        harness.history?.close();
        await cleanupTrimSessions(sessionID, harness.backupID);
      }
    },
  );

  test.each(["heuristic", "no-candidates"] as const)(
    "preserves an independently appended source before replay-backed %s trim can recover a stale backup",
    async mode => {
      const sessionID = `ses_trim_replay_drift_${randomUUID().replaceAll("-", "")}`;
      const messages = [
        message("usr_planned", "user", [
          userText("usr_planned", "Original request"),
        ]),
        message("ast_planned", "assistant", [
          completedTool(
            "tool_planned",
            mode === "heuristic" ? "read" : "question",
            "Exact result",
          ),
        ]),
      ];
      messages.forEach((item, index) => {
        item.info.sessionID = sessionID;
        item.info.time = { created: index === 0 ? 7 : 1 };
        if (item.info.role === "assistant")
          item.info.parentID = messages[0]!.info.id;
        item.parts.forEach(part => {
          part.sessionID = sessionID;
          part.messageID = item.info.id;
        });
      });
      const appended = message("usr_concurrent", "user", [
        userText("usr_concurrent", "Concurrent request must survive"),
      ]);
      appended.info.sessionID = sessionID;
      appended.info.time = { created: 20 };
      appended.parts[0]!.sessionID = sessionID;
      const harness = trimCommandHarness(sessionID, messages, {
        persisted: true,
        beforeFork: () => harness.history!.write(appended),
        partUpdateError: new Error("injected trim failure"),
      });
      const before = harness.history!.read(sessionID);
      let failure: unknown;
      try {
        try {
          await executeMagicTrim(harness.v2, sessionID, 0, {
            retention:
              mode === "heuristic"
                ? { mode: "heuristic" }
                : { mode: "task-aware", apiKey: undefined },
          });
        } catch (error) {
          failure = error;
        }
        expect(harness.history!.read(sessionID)).toEqual([...before, appended]);
        expect(harness.selectedSessions).toEqual([]);
        expect(harness.deletedSessions).toEqual([harness.backupID]);
        expect(String(failure)).toContain(
          "chronological fork initialization failed",
        );
        expect(harness.history!.read(harness.backupID)).toEqual([]);
        expect(await readCache(harness.backupID)).toBeNull();
        expect(await readStats(harness.backupID)).toBeNull();
        expect(await readStats(sessionID)).toBeNull();
        expect(harness.modelCalls).toEqual([]);
        expect(harness.prompts).toEqual([]);
      } finally {
        harness.history!.close();
        await cleanupTrimSessions(sessionID, harness.backupID);
      }
    },
  );

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
    await trimToolParts(
      { v2, sessionID: "session" },
      trim.trimmedTurns,
      new Set(),
    );

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
  test("trims timestamp-disagreeing history twice with faithful backups and unchanged users", async () => {
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
    const harness = trimCommandHarness(sessionID, source, { persisted: true });
    const originalUser = structuredClone(source[0]);
    const retention = {
      mode: "task-aware" as const,
      apiKey: "fake-key",
      fetch: scoredJevFetch(() => 0),
    };
    try {
      for (const attempt of [0, 1]) {
        if (attempt === 1) {
          const assistant = harness.history!.read(sessionID)[6]!;
          const part = readTool("read_new", "new obsolete listing");
          part.sessionID = sessionID;
          part.messageID = assistant.info.id;
          harness.history!.updatePart(part);
        }
        await expect(
          executeMagicTrim(harness.v2, sessionID, 0, { retention }),
        ).resolves.toBe(true);
        const backup = harness.history!.read(harness.backupID);
        expect(backup.slice(0, 7).map(item => item.info.role)).toEqual(
          source.slice(0, 7).map(item => item.info.role),
        );
        expect(backup[0]!.info.time).toEqual(originalUser!.info.time);
        expect(backup[0]!.parts[0]).toMatchObject({
          type: "text",
          text: "Keep the exact user request.",
        });
        expect(harness.history!.read(sessionID)[0]).toEqual(originalUser);
        expect((backup[6]!.info as { parentID: string }).parentID).toBe(
          backup[0]!.info.id,
        );
      }
      expect(
        harness.modelCalls.filter(call => call === "summary"),
      ).toHaveLength(2);
      expect(harness.selectedSessions).toEqual([]);
    } finally {
      harness.history!.close();
      await cleanupTrimSessions(sessionID, harness.backupID);
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
    const usefulBefore = structuredClone(useful);
    const calls: JevCall[] = [];
    const harness = trimCommandHarness(sessionID, messages);

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
      ).resolves.toBeTrue();

      expect(useful).toEqual(usefulBefore);
      if (stale.state.status !== "completed") throw new Error("completed");
      expect(stale.state.output).toContain("Historical read contents omitted");
      expect(stale.state.metadata["magicCompact"]).toEqual({ trimmed: true });
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
      expect(harness.forks).toEqual([harness.summaryForkID, harness.backupID]);
      expect(harness.deletedSessions).toEqual([harness.summaryForkID]);
      expect(JSON.stringify(messages)).not.toContain(harness.summaryForkID);
      expect(harness.toasts.at(-1)?.message).toBe(
        "Trimmed 2 tool call(s). Task-aware retention kept 1 of 2 evaluated tool call(s) whole (read 1).",
      );
      const notice = harness.prompts[0]!.parts![0] as TextPart;
      expect(notice.ignored).toBeTrue();
      expect(notice.text.split("\n")).toHaveLength(4);
      expect(notice.text).toContain("Changes | Tool calls trimmed: 2");
      expect(notice.text).toContain(
        "Retention | Task-aware retention kept 1 of 2 evaluated tool call(s) whole (read 1).",
      );
    } finally {
      await cleanupTrimSessions(sessionID, harness.backupID);
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
      const before = structuredClone(messages);
      const harness = trimCommandHarness(sessionID, messages);
      try {
        expect(
          await executeMagicTrim(harness.v2, sessionID, 0, {
            retention: {
              mode: "task-aware",
              apiKey: "test-key",
              fetch: scoredJevFetch(() => (keep ? 0.95 : 0.1), []),
            },
          }),
        ).toBe(!keep);
        const feedback = `Task-aware retention kept ${keep ? 1 : 0} of 1 evaluated tool call(s) whole${keep ? " (read 1)" : ""}.`;
        expect(harness.toasts.at(-1)?.message).toBe(
          `${keep ? TRIM_NOOP : "Trimmed 1 tool call(s)."} ${feedback}`,
        );
        if (keep) {
          expect(messages).toEqual(before);
          expect(harness.prompts).toEqual([]);
          expect(harness.deletedSessions).toEqual([
            harness.summaryForkID,
            harness.backupID,
          ]);
          expect(await readCache(harness.backupID)).toBeNull();
          expect(await readStats(harness.backupID)).toBeNull();
          expect(await readStats(sessionID)).toBeNull();
        } else {
          const notice = harness.prompts[0]!.parts![0] as TextPart;
          expect(notice.ignored).toBeTrue();
          expect(notice.text).toContain(`Retention | ${feedback}`);
          expect(notice.text).toContain("Changes | Tool calls trimmed: 1");
        }
      } finally {
        await cleanupTrimSessions(sessionID, harness.backupID);
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
      ).resolves.toBeTrue();

      expect(harness.modelCalls).toEqual([]);
      expect(harness.forks).toEqual([harness.backupID]);
      expect(harness.toasts.at(-1)?.message).toBe(
        "Trimmed 1 tool call(s). Task-aware retention: no tool calls needed evaluation.",
      );
      expect((harness.prompts[0]!.parts![0] as TextPart).text).toContain(
        "Retention | Task-aware retention: no tool calls needed evaluation.",
      );
    } finally {
      await cleanupTrimSessions(sessionID, harness.backupID);
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
      ).resolves.toBeTrue();

      expect(harness.modelCalls).toEqual([]);
      expect(harness.forks).toEqual([harness.backupID]);
      if (tool.state.status !== "completed") throw new Error("completed");
      expect(tool.state.output).toContain("Historical read contents omitted");
    } finally {
      await cleanupTrimSessions(sessionID, harness.backupID);
    }
  });

  test.each([
    ["a missing API key", undefined, 200],
    ["a failed classification", "test-key", 500],
  ] as const)(
    "%s leaves source, backup, cache, and stats untouched",
    async (_name, apiKey, status) => {
      const sessionID = `ses_trim_retention_fail_${randomUUID().replaceAll("-", "")}`;
      const messages = [
        message("usr_1", "user", [userText("usr_1", "Add the parser.")]),
        message("ast_1", "assistant", [
          readTool("tool_read", "large output ".repeat(100)),
        ]),
      ];
      const before = structuredClone(messages);
      const harness = trimCommandHarness(sessionID, messages);
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
        expect(harness.forks).not.toContain(harness.backupID);
        expect(harness.deletedSessions).toEqual(
          apiKey ? [harness.summaryForkID] : [],
        );
        expect(harness.modelCalls).toEqual(
          apiKey ? ["agents", "providers", "summary"] : [],
        );
        expect(harness.prompts).toEqual([]);
        expect(await readCache(sessionID)).toBeNull();
        expect(await readStats(sessionID)).toBeNull();
        expect(harness.toasts.at(-1)?.message).toContain(
          "/magic-trim --heuristic",
        );
      } finally {
        await cleanupTrimSessions(sessionID, harness.backupID);
      }
    },
  );

  test.each(["source drift", "a pending revert"] as const)(
    "%s during classification fails before writes and removes the backup",
    async change => {
      const sessionID = `ses_trim_retention_race_${randomUUID().replaceAll("-", "")}`;
      const tool = readTool("tool_read", "large output ".repeat(100));
      const messages = [
        message("usr_1", "user", [userText("usr_1", "Add the parser.")]),
        message("ast_1", "assistant", [tool]),
      ];
      const toolBefore = structuredClone(tool);
      const harness = trimCommandHarness(sessionID, messages, {
        duringSummary: () => {
          if (change === "source drift") {
            messages.push(
              message("usr_concurrent", "user", [
                userText("usr_concurrent", "concurrent request"),
              ]),
            );
          } else {
            harness.setRevert({ messageID: "ast_1" });
          }
        },
      });

      try {
        await expect(
          executeMagicTrim(harness.v2, sessionID, 0, {
            retention: {
              mode: "task-aware",
              apiKey: "test-key",
              fetch: scoredJevFetch(() => 0.1),
            },
          }),
        ).rejects.toThrow(
          change === "source drift"
            ? "OpenCode source changed during Magic Trim"
            : TRIM_PENDING_REVERT,
        );

        expect(tool).toEqual(toolBefore);
        expect(harness.deletedSessions).toEqual([
          harness.summaryForkID,
          harness.backupID,
        ]);
        expect(harness.selectedSessions).toEqual([]);
        expect(harness.prompts).toEqual([]);
        expect(await readCache(sessionID)).toBeNull();
        expect(await readStats(sessionID)).toBeNull();
      } finally {
        await cleanupTrimSessions(sessionID, harness.backupID);
      }
    },
  );

  test.each([
    ["a thrown messages reload", "messages", "throw", false],
    [
      "an error messages response and failed backup cleanup",
      "messages",
      "response",
      true,
    ],
    [
      "a thrown session reload and failed backup cleanup",
      "session",
      "throw",
      true,
    ],
    ["an error session response", "session", "response", false],
  ] as const)(
    "%s before trim writes keep only sanitized failures",
    async (_name, target, kind, cleanupFails) => {
      const sessionID = `ses_trim_reload_${randomUUID().replaceAll("-", "")}`;
      const messages = [
        message("usr_1", "user", [userText("usr_1", "Add the parser.")]),
        message("ast_1", "assistant", [
          readTool("tool_read", "large output ".repeat(100)),
        ]),
      ];
      const before = structuredClone(messages);
      const harness = trimCommandHarness(sessionID, messages, {
        sourceReload: {
          target,
          read: async () => {
            if (kind === "throw") {
              throw Object.assign(
                new Error(`fetch failed: ${PRIVATE_BODY}`, {
                  cause: new Error(PRIVATE_BODY),
                }),
                { status: 503, body: PRIVATE_BODY },
              );
            }
            return {
              error: { name: "UnknownError", data: { message: PRIVATE_BODY } },
              response: new Response(PRIVATE_BODY, { status: 502 }),
            };
          },
        },
        ...(cleanupFails
          ? {
              backupDeleteError: new Error(PRIVATE_BODY, {
                cause: PRIVATE_BODY,
              }),
            }
          : {}),
      });
      let failure: unknown;

      try {
        try {
          await executeMagicTrim(harness.v2, sessionID, 0, {
            retention: {
              mode: "task-aware",
              apiKey: "test-key",
              fetch: scoredJevFetch(() => 0.1),
            },
          });
        } catch (error) {
          failure = error;
        }

        const root = cleanupFails
          ? (failure as AggregateError).errors[0]
          : failure;
        expect(root).toBeInstanceOf(TrimRevalidationError);
        expect(root).toMatchObject({
          target,
          status: kind === "throw" ? 503 : 502,
        });
        if (cleanupFails) {
          expect(failure).toBeInstanceOf(AggregateError);
          expect(String(failure)).toContain(sessionID);
          expect(String(failure)).toContain(harness.backupID);
        }
        expect(reachableStrings(failure).join("\n")).not.toContain(
          PRIVATE_BODY,
        );
        expect(JSON.stringify(harness.toasts)).not.toContain(PRIVATE_BODY);
        expect(harness.toasts.at(-1)?.message).toContain("left unchanged");
        expect(messages).toEqual(before);
        expect(harness.deletedSessions).toEqual([
          harness.summaryForkID,
          harness.backupID,
        ]);
        expect(harness.selectedSessions).toEqual([]);
        expect(harness.prompts).toEqual([]);
        expect(await readStats(sessionID)).toBeNull();
      } finally {
        await cleanupTrimSessions(sessionID, harness.backupID);
      }
    },
  );
});

const PRIVATE_BODY = "PRIVATE-SESSION-BODY-7f3a";

// Every string reachable from an error: messages, stacks, causes, aggregate
// members, and any other own property.
function reachableStrings(value: unknown, seen = new Set<unknown>()): string[] {
  if (typeof value === "string") return [value];
  if (typeof value !== "object" || value === null || seen.has(value)) {
    return [];
  }
  seen.add(value);
  return Object.getOwnPropertyNames(value).flatMap(key =>
    reachableStrings((value as Record<string, unknown>)[key], seen),
  );
}

type TrimCommandHarnessOptions = {
  persisted?: boolean;
  toastMode?: "hang" | "fail";
  partUpdateError?: Error;
  selectError?: Error;
  backupDeleteError?: Error;
  beforeFork?: () => void;
  // Runs while the retention summary request is in flight.
  duringSummary?: () => void;
  // Replaces source reads made after the retention summary.
  sourceReload?: {
    target: "messages" | "session";
    read: () => Promise<unknown>;
  };
};

function trimCommandHarness(
  sessionID: string,
  messages: MessageWithParts[],
  options: TrimCommandHarnessOptions = {},
) {
  const history = options.persisted
    ? persistedForkHistory(messages)
    : undefined;
  const backupID = `${sessionID}_backup`;
  const sourceSession = {
    id: sessionID,
    title: "Trim source",
    directory: "/workspace",
  } as Session;
  const prompts: Record<string, unknown>[] = [];
  const toasts: Array<{ message?: string }> = [];
  const selectedSessions: string[] = [];
  const deletedSessions: string[] = [];
  const forks: string[] = [];
  const modelCalls: string[] = [];
  const summaryForkID = `${sessionID}_summary`;
  let summaryFork: MessageWithParts[] | undefined;
  let backupFork: MessageWithParts[] | undefined;
  let summaryReady = false;
  let revert: { messageID: string } | undefined;
  let noticeCount = 0;
  const v2 = {
    app: {
      agents: async () => {
        modelCalls.push("agents");
        summaryReady = true;
        return { data: [compactionAgent()] };
      },
    },
    provider: {
      list: async () => {
        modelCalls.push("providers");
        return { data: providerCatalog() };
      },
    },
    session: {
      messages: async (request: { sessionID: string }) => {
        if (
          request.sessionID === sessionID
          && options.sourceReload?.target === "messages"
          && modelCalls.includes("summary")
        ) {
          return options.sourceReload.read();
        }
        return {
          data: history
            ? history.read(request.sessionID)
            : request.sessionID === summaryForkID
              ? structuredClone(summaryFork)
              : request.sessionID === backupID
                ? structuredClone(backupFork)
                : messages,
        };
      },
      get: async () => {
        if (
          options.sourceReload?.target === "session"
          && modelCalls.includes("summary")
        ) {
          return options.sourceReload.read();
        }
        return {
          data: { ...sourceSession, ...(revert ? { revert } : {}) },
        };
      },
      // The first fork of a task-aware run is the retention summary fork.
      fork: async (request: { messageID?: string }) => {
        options.beforeFork?.();
        const summaryRun = summaryReady;
        summaryReady = false;
        const id = summaryRun ? summaryForkID : backupID;
        forks.push(id);
        if (summaryRun) summaryFork = rewriteFork(messages, id);
        else backupFork = rewriteFork(messages, id);
        history?.delete(id);
        history?.fork(sessionID, id, request.messageID);
        return { data: { ...sourceSession, id } as Session };
      },
      update: async (request: { sessionID: string }) => ({
        data: { ...sourceSession, id: request.sessionID } as Session,
      }),
      abort: async () => ({ data: true }),
      prompt: async (request: {
        sessionID: string;
        messageID?: string;
        agent?: string;
        parts?: Part[];
      }) => {
        if (request.sessionID === summaryForkID) {
          modelCalls.push("summary");
          options.duringSummary?.();
          const fork = history ? history.read(summaryForkID) : summaryFork!;
          const reply = summaryReply(fork, request, summaryForkID);
          if (history)
            fork
              .slice(-2)
              .forEach((item, index) => history.write(item, 100 + index));
          return reply;
        }
        prompts.push(request);
        noticeCount += 1;
        const messageID = `trim-notice-${noticeCount}`;
        const parts = (request.parts ?? []).map((part, index) => ({
          ...part,
          id: `${messageID}-part-${index}`,
          sessionID,
          messageID,
        })) as Part[];
        messages.push(message(messageID, "user", parts));
        if (history)
          history.write(
            {
              info: {
                ...messages.at(-1)!.info,
                sessionID,
                time: { created: 100 + noticeCount },
              },
              parts,
            },
            100 + noticeCount,
          );
        return { data: { info: { id: messageID }, parts } };
      },
      delete: async (request: { sessionID: string }) => {
        deletedSessions.push(request.sessionID);
        if (request.sessionID === summaryForkID) summaryFork = [];
        history?.delete(request.sessionID);
        if (request.sessionID === backupID && options.backupDeleteError) {
          throw options.backupDeleteError;
        }
        return { data: true };
      },
    },
    part: {
      update: async (request: { part: Part }) => {
        if (options.partUpdateError) throw options.partUpdateError;
        history?.updatePart(request.part);
        return { data: request.part };
      },
    },
    sync: {
      replay: async (request: {
        events: Parameters<NonNullable<typeof history>["replay"]>[0];
      }) => {
        history!.replay(request.events);
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
      selectSession: async (request: { sessionID: string }) => {
        selectedSessions.push(request.sessionID);
        if (options.selectError) throw options.selectError;
        return { data: true };
      },
    },
  } as unknown as V2Client;

  return {
    v2,
    sessionID,
    backupID,
    summaryForkID,
    messages,
    prompts,
    toasts,
    selectedSessions,
    deletedSessions,
    forks,
    modelCalls,
    history,
    setRevert(value: { messageID: string }) {
      revert = value;
    },
  };
}

async function settleWithin<T>(promise: Promise<T>, timeoutMs: number) {
  return Promise.race([
    promise,
    new Promise<never>((_, reject) =>
      setTimeout(
        () => reject(new Error("Test operation timed out.")),
        timeoutMs,
      ),
    ),
  ]);
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

function v0SummaryPartID(messageID: string): string {
  return `prt_-magic_summary_${messageID}`;
}

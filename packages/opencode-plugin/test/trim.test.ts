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
import { TRIM_NOOP, executeMagicTrim } from "../src/magic-trim";
import {
  allocateOmission,
  cachePath,
  deleteCache,
  readCache,
} from "../src/storage/omission";
import {
  type ConversationStats,
  deleteStats,
  readStats,
  writeStats,
} from "../src/storage/stats";
import { trimStatsMessage } from "../src/stats/constants";
import { countCanonicalContentTokens } from "../src/stats/tokenize";

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
      await trimToolParts({ v2, sessionID: "session" }, [selectedTurn]),
    ).toBe(1);
    expect(tool.state.status).toBe("completed");
    if (tool.state.status !== "completed") {
      throw new Error("Expected completed tool state.");
    }
    expect(tool.state.metadata["magicCompact"]).toEqual({ trimmed: true });
    expect(tool.state.output).toBe("Successfully updated todos.");
    expect(reasoning.text).toBe("unchanged");

    expect(
      await trimToolParts({ v2, sessionID: "session" }, [selectedTurn]),
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
      await trimToolParts({ v2, sessionID }, [turn(tool)]);

      expect(tool.state.status).toBe("completed");
      if (tool.state.status !== "completed")
        throw new Error("Expected completion");
      expect(tool.state.output).toContain(
        `Content ID: ${sessionID.slice(-12)}:omitted-`,
      );
      expect(tool.state.output).toContain("Output Length: 3 bytes");
    } finally {
      await rm(cachePath(sessionID), { force: true });
    }
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

      expect(await executeMagicTrim(v2, sessionID, 0)).toBeTrue();

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
      expect(notice.text).toBe(trimStatsMessage(beforeTokens, afterTokens, 1));
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
        executeMagicTrim(harness.v2, sessionID, 0),
      ).resolves.toBeFalse();

      expect(harness.deletedSessions).toEqual([harness.backupID]);
      expect(await readCache(harness.backupID)).toBeNull();
      expect(await readStats(harness.backupID)).toBeNull();
      expect(harness.toasts.map(toast => toast.message)).toEqual([
        "Trimming session...",
        TRIM_NOOP,
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
        await executeMagicTrim(harness.v2, sessionID, 0);
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
            executeMagicTrim(noOp.v2, noOpID, 0, { controlTimeoutMs: 5 }),
            250,
          ),
        ).resolves.toBeFalse();
        expect(noOp.toasts.map(toast => toast.message)).toEqual([
          "Trimming session...",
          TRIM_NOOP,
        ]);

        await expect(
          settleWithin(
            executeMagicTrim(success.v2, successID, 0, {
              controlTimeoutMs: 5,
            }),
            250,
          ),
        ).resolves.toBeTrue();
        expect(success.toasts.map(toast => toast.message)).toEqual([
          "Trimming session...",
          "Trimmed 1 tool call(s).",
        ]);

        await expect(
          settleWithin(
            executeMagicTrim(failed.v2, failureID, 0, {
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

  test("preserves the root error and terminal feedback when trim recovery fails", async () => {
    const sessionID = `ses_trim_recovery_${randomUUID().replaceAll("-", "")}`;
    const messages = [
      message("usr_recovery", "user", []),
      message("ast_recovery", "assistant", [
        readTool("tool_recovery", "large output ".repeat(100)),
      ]),
    ];
    const harness = trimCommandHarness(sessionID, messages, {
      partUpdateError: new Error("trim root failure"),
      selectError: new Error("raw select failure"),
    });
    let failure: unknown;

    try {
      try {
        await executeMagicTrim(harness.v2, sessionID, 0);
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
    } finally {
      await cleanupTrimSessions(sessionID, harness.backupID);
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
    await trimToolParts({ v2, sessionID: "session" }, trim.trimmedTurns);

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

type TrimCommandHarnessOptions = {
  toastMode?: "hang" | "fail";
  partUpdateError?: Error;
  selectError?: Error;
  backupDeleteError?: Error;
};

function trimCommandHarness(
  sessionID: string,
  messages: MessageWithParts[],
  options: TrimCommandHarnessOptions = {},
) {
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
  let noticeCount = 0;
  const v2 = {
    session: {
      messages: async () => ({ data: messages }),
      get: async () => ({ data: sourceSession }),
      fork: async () => ({
        data: { ...sourceSession, id: backupID } as Session,
      }),
      update: async (request: { sessionID: string }) => ({
        data: { ...sourceSession, id: request.sessionID } as Session,
      }),
      prompt: async (request: { parts?: Part[] }) => {
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
        return { data: { info: { id: messageID }, parts } };
      },
      delete: async (request: { sessionID: string }) => {
        deletedSessions.push(request.sessionID);
        if (request.sessionID === backupID && options.backupDeleteError) {
          throw options.backupDeleteError;
        }
        return { data: true };
      },
    },
    part: {
      update: async (request: { part: Part }) => {
        if (options.partUpdateError) throw options.partUpdateError;
        return { data: request.part };
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
    messages,
    prompts,
    toasts,
    selectedSessions,
    deletedSessions,
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

function v0SummaryPartID(messageID: string): string {
  return `prt_-magic_summary_${messageID}`;
}

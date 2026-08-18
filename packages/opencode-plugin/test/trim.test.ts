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
import { executeMagicTrim } from "../src/magic-trim";
import { cachePath, deleteCache } from "../src/storage/omission";
import { deleteStats, readStats, writeStats } from "../src/storage/stats";
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
    const backupID = `${sessionID}_backup`;
    const tool = readTool("tool_accounting", "historical output ".repeat(70));
    const messages = [
      message("usr_accounting", "user", []),
      message("ast_accounting", "assistant", [tool]),
    ];
    const sourceSession = {
      id: sessionID,
      title: "Trim accounting",
      directory: "/workspace",
    } as Session;
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
        delete: async () => ({ data: true }),
      },
      part: {
        update: async (request: { part: Part }) => ({ data: request.part }),
      },
      tui: {
        showToast: async () => ({ data: true }),
        selectSession: async () => ({ data: true }),
      },
    } as unknown as V2Client;

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
      expect(notice.text).toContain("Changes | 1 tool call(s) trimmed");
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

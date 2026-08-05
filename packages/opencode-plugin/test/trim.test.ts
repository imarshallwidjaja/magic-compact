import { describe, expect, test } from "bun:test";
import { randomUUID } from "node:crypto";
import { rm } from "node:fs/promises";
import type { Message, Part, ToolPart } from "@opencode-ai/sdk/v2";
import type { V2Client } from "../src/api";
import {
  createTrimPlan,
  type MessageWithParts,
  type Turn,
} from "../src/compact/plan";
import { trimToolParts } from "../src/compact/prune";
import { cachePath } from "../src/storage/omission";

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

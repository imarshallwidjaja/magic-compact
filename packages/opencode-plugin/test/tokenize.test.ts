import { describe, expect, test } from "bun:test";
import type { Message } from "@opencode-ai/sdk/v2";
import type { V2Client } from "../src/api";
import type { MessageWithParts } from "../src/compact/plan";
import { getProviderTokens } from "../src/stats/tokenize";

describe("getProviderTokens", () => {
  test("returns null when newest assistant tokens are present but all zero", async () => {
    const messages: MessageWithParts[] = [
      message("usr_1", "user"),
      message("ast_1", "assistant", {
        tokens: {
          input: 1200,
          output: 80,
          reasoning: 0,
          cache: { read: 400, write: 0 },
        },
      }),
      message("usr_2", "user"),
      message("ast_overflow", "assistant", {
        tokens: {
          input: 0,
          output: 0,
          reasoning: 0,
          cache: { read: 0, write: 0 },
        },
      }),
    ];
    const v2 = {
      session: {
        messages: async () => ({ data: messages }),
      },
    } as unknown as V2Client;

    expect(await getProviderTokens(v2, "session")).toBeNull();
  });

  test("returns the newest positive provider token total", async () => {
    const messages: MessageWithParts[] = [
      message("usr_1", "user"),
      message("ast_1", "assistant", {
        tokens: {
          input: 100,
          output: 20,
          reasoning: 5,
          cache: { read: 10, write: 2 },
        },
      }),
      message("usr_2", "user"),
      message("ast_2", "assistant", {
        tokens: {
          input: 1000,
          output: 50,
          reasoning: 25,
          cache: { read: 100, write: 5 },
        },
      }),
    ];
    const v2 = {
      session: {
        messages: async () => ({ data: messages }),
      },
    } as unknown as V2Client;

    expect(await getProviderTokens(v2, "session")).toBe(1180);
  });
});

function message(
  id: string,
  role: "user" | "assistant",
  info: Record<string, unknown> = {},
): MessageWithParts {
  return {
    info: { id, role, ...info } as Message,
    parts: [],
  };
}

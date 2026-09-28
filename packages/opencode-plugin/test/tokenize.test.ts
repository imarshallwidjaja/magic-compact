import { describe, expect, spyOn, test } from "bun:test";
import type { Message, ToolPart } from "@opencode-ai/sdk/v2";
import * as gptTokenizer from "gpt-tokenizer";
import type { V2Client } from "../src/api";
import type { MessageWithParts } from "../src/compact/plan";
import {
  countCanonicalContentTokens,
  countGptTokens,
  countGptTokensWithinLimit,
  countSessionTokens,
  countEffectiveViewTokens,
  countModelFacingContentTokens,
} from "../src/stats/tokenize";
import { bindToolSource, createToolOverlaySnapshot } from "../src/trim/plan";

describe("canonical token counting", () => {
  test("keeps stored media-only and effective tool-overlay counts distinct", () => {
    const tool: ToolPart = {
      id: "prt_tool",
      sessionID: "session",
      messageID: "ast_tool",
      type: "tool",
      callID: "call_tool",
      tool: "read",
      state: {
        status: "completed",
        input: {},
        output: "long data ".repeat(300),
        title: "read",
        metadata: {},
        time: { start: 1, end: 2 },
      },
    };
    const messages: MessageWithParts[] = [
      { ...message("usr_tool", "user"), parts: [] },
      { ...message("ast_tool", "assistant"), parts: [tool] },
    ];
    const replacement = structuredClone(tool);
    if (replacement.state.status !== "completed")
      throw new Error("Expected completed tool");
    replacement.state.output = "Short receipt";
    const snapshot = createToolOverlaySnapshot([
      { source: bindToolSource(tool), part: replacement, omissions: [] },
    ]);
    const stored = countModelFacingContentTokens(messages);
    const effective = countEffectiveViewTokens(messages, snapshot);
    expect(stored.tokens).toBeGreaterThan(effective.tokens);
    expect(stored.projectedMediaTokens).toBe(0);
    expect(effective.projectedMediaTokens).toBe(0);
    expect(countModelFacingContentTokens(messages)).toEqual(stored);
    expect(
      countEffectiveViewTokens(messages, createToolOverlaySnapshot([])),
    ).toEqual(stored);
  });
  test("counts canonical content locally regardless of provider usage", async () => {
    const messages: MessageWithParts[] = [
      {
        ...message("usr_local", "user"),
        parts: [
          {
            id: "usr_local_text",
            sessionID: "session",
            messageID: "usr_local",
            type: "text",
            text: "Canonical local content",
          },
        ],
      },
      message("ast_usage", "assistant", {
        tokens: {
          input: 999_999,
          output: 999_999,
          reasoning: 999_999,
          cache: { read: 999_999, write: 999_999 },
        },
      }),
    ];
    const v2 = {
      session: { messages: async () => ({ data: messages }) },
    } as unknown as V2Client;

    expect(await countSessionTokens(v2, "session")).toBe(
      countCanonicalContentTokens(messages),
    );
    expect(await countSessionTokens(v2, "session")).toBeLessThan(100);
  });

  test("excludes ignored reports while retaining synthetic active context", () => {
    const messages: MessageWithParts[] = [
      {
        ...message("usr_context", "user"),
        parts: [
          {
            id: "active_text",
            sessionID: "session",
            messageID: "usr_context",
            type: "text",
            text: "Active user content",
          },
          {
            id: "ignored_report",
            sessionID: "session",
            messageID: "usr_context",
            type: "text",
            text: "Ignored reporting notice",
            ignored: true,
          },
          {
            id: "synthetic_boundary",
            sessionID: "session",
            messageID: "usr_context",
            type: "text",
            text: "Synthetic model-facing boundary",
            synthetic: true,
          },
        ],
      },
    ];

    expect(countCanonicalContentTokens(messages)).toBe(
      countGptTokens("Active user content")
        + countGptTokens("Synthetic model-facing boundary"),
    );
  });

  test("includes ignored assistant text while still excluding ignored user notices", () => {
    const messages: MessageWithParts[] = [
      {
        ...message("usr_notice", "user"),
        parts: [
          {
            id: "user_notice",
            sessionID: "session",
            messageID: "usr_notice",
            type: "text",
            text: "Ignored reporting notice",
            ignored: true,
          },
        ],
      },
      {
        ...message("ast_ignored", "assistant"),
        parts: [
          {
            id: "assistant_visible",
            sessionID: "session",
            messageID: "ast_ignored",
            type: "text",
            text: "Visible assistant history",
          },
          {
            id: "assistant_ignored",
            sessionID: "session",
            messageID: "ast_ignored",
            type: "text",
            text: "Ignored assistant continuation",
            ignored: true,
          },
        ],
      },
    ];

    expect(countCanonicalContentTokens(messages)).toBe(
      countGptTokens("Visible assistant history")
        + countGptTokens("Ignored assistant continuation"),
    );
  });

  test.each([
    ["ASCII", "boundary text"],
    ["Unicode", "🚀 boundary café"],
    ["replacement", "\uFFFD"],
    ["lone high surrogate", "\uD800"],
    ["lone low surrogate", "\uDE80"],
    ["embedded lone surrogate", "a\uD800b"],
  ])("matches exact adjacent %s token limits", (_label, text) => {
    const exact = countGptTokens(text);

    expect(exact).toBe(gptTokenizer.countTokens(text));
    expect(countGptTokensWithinLimit(text, exact)).toBe(exact);
    expect(countGptTokensWithinLimit(text, exact - 1)).toBeNull();
  });

  test("evicts the previous exact token-count memo entry", () => {
    const first = `memo-first-${"a".repeat(257)}`;
    const second = `memo-second-${"b".repeat(257)}`;
    const countSpy = spyOn(gptTokenizer, "countTokens");
    try {
      countGptTokens(first);
      countGptTokens(first);
      countGptTokens(second);
      countGptTokens(first);

      expect(countSpy).toHaveBeenCalledTimes(3);
    } finally {
      countSpy.mockRestore();
    }
  });

  test("does not memoize oversized exact token-count keys", () => {
    const oversized = `oversized-memo-${"0123456789abcdef".repeat(8_192)}`;
    const countSpy = spyOn(gptTokenizer, "countTokens");
    try {
      countGptTokens(oversized);
      countGptTokens(oversized);

      expect(countSpy).toHaveBeenCalledTimes(2);
    } finally {
      countSpy.mockRestore();
    }
  });

  test("rejects the first over-budget ASCII span without tokenizing", () => {
    const accepted = "a".repeat(128);
    const rejected = "a".repeat(129);
    expect(countGptTokensWithinLimit(accepted, 16)).toBe(16);
    expect(countGptTokensWithinLimit(accepted, 15)).toBeNull();

    const encodeSpy = spyOn(gptTokenizer, "encode");
    const countSpy = spyOn(gptTokenizer, "countTokens");
    const withinSpy = spyOn(gptTokenizer, "isWithinTokenLimit");
    try {
      expect(countGptTokensWithinLimit(rejected, 1)).toBeNull();
      expect(encodeSpy).not.toHaveBeenCalled();
      expect(countSpy).not.toHaveBeenCalled();
      expect(withinSpy).not.toHaveBeenCalled();
    } finally {
      encodeSpy.mockRestore();
      countSpy.mockRestore();
      withinSpy.mockRestore();
    }
  });

  test("rejects a pathological over-budget span without entering gpt-tokenizer", () => {
    const text = `${"🚀".repeat(50_000)}!`;
    const encodeSpy = spyOn(gptTokenizer, "encode");
    const countSpy = spyOn(gptTokenizer, "countTokens");
    const withinSpy = spyOn(gptTokenizer, "isWithinTokenLimit");
    try {
      expect(countGptTokensWithinLimit(text, 1_000)).toBeNull();
      expect(encodeSpy).not.toHaveBeenCalled();
      expect(countSpy).not.toHaveBeenCalled();
      expect(withinSpy).not.toHaveBeenCalled();
    } finally {
      encodeSpy.mockRestore();
      countSpy.mockRestore();
      withinSpy.mockRestore();
    }
  });

  test("pins gpt-tokenizer to the measured 3.4.0 o200k bound", async () => {
    const pluginPackage = await import("../package.json");
    const tokenizerPackage = await import("gpt-tokenizer/package.json");

    expect(pluginPackage.default.dependencies["gpt-tokenizer"]).toBe("3.4.0");
    expect(tokenizerPackage.default.version).toBe("3.4.0");
  });

  test("delegates bounded checks without local prefix encoding", () => {
    const encodeSpy = spyOn(gptTokenizer, "encode");
    try {
      expect(countGptTokensWithinLimit("x".repeat(20_000), 1)).toBeNull();
      expect(encodeSpy).not.toHaveBeenCalled();
    } finally {
      encodeSpy.mockRestore();
    }
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

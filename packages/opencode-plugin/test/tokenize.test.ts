import { describe, expect, spyOn, test } from "bun:test";
import { randomBytes } from "node:crypto";
import type { Message, Part } from "@opencode-ai/sdk/v2";
import * as gptTokenizer from "gpt-tokenizer";
import type { V2Client } from "../src/api";
import type { MessageWithParts } from "../src/compact/plan";
import { mediaPlaceholderText, projectsSmaller } from "../src/policy/media";
import { countV2PayloadTokens } from "../src/v2/history";
import {
  countCanonicalContentTokens,
  countGptTokens,
  countGptTokensWithinLimit,
  countModelFacingContentTokens,
  countSessionTokens,
} from "../src/stats/tokenize";

describe("canonical token counting", () => {
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
      await countCanonicalContentTokens(messages),
    );
    expect(await countSessionTokens(v2, "session")).toBeLessThan(100);
  });

  test("excludes ignored reports while retaining synthetic active context", async () => {
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

    expect(await countCanonicalContentTokens(messages)).toBe(
      countGptTokens("Active user content")
        + countGptTokens("Synthetic model-facing boundary"),
    );
  });

  test("includes ignored assistant text while still excluding ignored user notices", async () => {
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

    expect(await countCanonicalContentTokens(messages)).toBe(
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

  test("decides projection of a large payload without tokenizing it", () => {
    const payload = "A".repeat(1_000_000);
    const countSpy = spyOn(gptTokenizer, "countTokens");
    try {
      expect(
        projectsSmaller(mediaPlaceholderText(1, "image/png"), payload),
      ).toBe(true);
      expect(
        countSpy.mock.calls.every(([text]) => (text as string).length < 1_000),
      ).toBe(true);
    } finally {
      countSpy.mockRestore();
    }
  });

  // A uniform run is quadratic for the tokenizer as one piece: 400 KB of one
  // letter took about 100 s when measured.
  test("counts a file payload in fixed segments, once per distinct payload", () => {
    const payload = "A".repeat(400_000);
    expect(countV2PayloadTokens(payload)).toBe(50_000);
    const countSpy = spyOn(gptTokenizer, "countTokens");
    try {
      expect(countV2PayloadTokens(payload)).toBe(50_000);
      expect(countSpy).not.toHaveBeenCalled();
      expect(countV2PayloadTokens("iVBORw0K")).toBe(countGptTokens("iVBORw0K"));
    } finally {
      countSpy.mockRestore();
    }
  });

  // A trim counted an unchanged 19 MB PDF data URL whole, four times before
  // creating its successor, without returning to the event loop.
  test("counts a large part text exactly, yields to the event loop, and reuses the count for identical text", async () => {
    const url = `data:application/pdf;base64,${randomBytes(393_216).toString("base64")}`;
    const messages = [
      { ...message("usr_pdf", "user"), parts: [filePart(url)] },
    ];
    const exact = gptTokenizer.countTokens(url);
    const encodeSpy = spyOn(gptTokenizer, "encodeGenerator");
    try {
      const order: string[] = [];
      setImmediate(() => order.push("immediate"));
      const counted = await countCanonicalContentTokens(messages).then(
        count => {
          order.push("count");
          return count;
        },
      );

      expect(counted).toBe(exact);
      expect(order).toEqual(["immediate", "count"]);
      expect(await countModelFacingContentTokens(messages)).toEqual({
        tokens: exact,
        projectedMediaTokens: 0,
      });
      expect(encodeSpy).toHaveBeenCalledTimes(1);
    } finally {
      encodeSpy.mockRestore();
    }
  });

  test("keeps the tokenizer's special-token refusal for a large part text", async () => {
    const messages = [
      {
        ...message("usr_special", "user"),
        parts: [
          filePart(`data:text/x-log;base64,${"A".repeat(70_000)}<|endoftext|>`),
        ],
      },
    ];

    await expect(countCanonicalContentTokens(messages)).rejects.toThrow(
      "Disallowed special token found: <|endoftext|>",
    );
  });

  // gpt-tokenizer 3.4.0 internals; the version is pinned below. Its merge
  // cache evicts superlinearly once one input exceeds its 100,000 entries.
  test("disables the tokenizer's merge cache", () => {
    const core = (
      gptTokenizer.default as unknown as {
        bytePairEncodingCoreProcessor: { mergeCache?: unknown };
      }
    ).bytePairEncodingCoreProcessor;
    expect(core.mergeCache).toBeUndefined();
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

function filePart(url: string): Part {
  return {
    id: "prt_file",
    sessionID: "session",
    messageID: "usr_file",
    type: "file",
    mime: "application/pdf",
    url,
  } as Part;
}

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

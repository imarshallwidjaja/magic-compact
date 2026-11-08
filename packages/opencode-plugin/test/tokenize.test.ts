import { describe, expect, spyOn, test } from "bun:test";
import { createHash, randomBytes } from "node:crypto";
import type { Message, Part } from "@opencode-ai/sdk/v2";
import * as gptTokenizer from "gpt-tokenizer";
import type { MessageWithParts } from "../src/compact/plan";
import { mediaPlaceholderText, projectsSmaller } from "../src/policy/media";
import { countV2PayloadTokens } from "../src/v2/history";
import { countModelFacingContentTokens } from "../src/compact/accounting";
import {
  countGptTokens,
  countGptTokensWithinLimit,
} from "../src/stats/tokenize";

// Every special-token spelling of o200k_base in gpt-tokenizer 3.4.0, and the
// option under which that tokenizer reads them as ordinary text.
const O200K_SPECIAL_TOKENS = [
  "<|endoftext|>",
  "<|fim_prefix|>",
  "<|fim_middle|>",
  "<|fim_suffix|>",
  "<|im_start|>",
  "<|im_end|>",
  "<|im_sep|>",
  "<|endofprompt|>",
];
const LITERAL = { disallowedSpecial: new Set<string>() };

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

    expect((await countModelFacingContentTokens(messages)).tokens).toBe(
      countGptTokens("Canonical local content"),
    );
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

    expect((await countModelFacingContentTokens(messages)).tokens).toBe(
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

    expect((await countModelFacingContentTokens(messages)).tokens).toBe(
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

  // Session text is never a tokenizer control sequence: `<|endoftext|>` in a
  // tool output is the seven ordinary pieces <, |, end, of, text, |, >.
  test("counts a special-token spelling as the ordinary text it is", () => {
    const pieces = gptTokenizer.encode("<|endoftext|>", LITERAL);
    expect(pieces.map(piece => gptTokenizer.decode([piece]))).toEqual([
      "<",
      "|",
      "end",
      "of",
      "text",
      "|",
      ">",
    ]);
    expect(countGptTokens("<|endoftext|>")).toBe(7);
  });

  test.each(O200K_SPECIAL_TOKENS)(
    "counts %s exactly wherever it appears in a text",
    special => {
      const texts = [
        `${special} leading`,
        `trailing ${special}`,
        `${special}${special}`,
        // Over the 4,096-byte memo key limit, so counted uncached.
        `${special}${"plain words ".repeat(400)}${special}`,
      ];
      for (const text of texts) {
        const exact = countGptTokens(text);

        expect(exact).toBe(gptTokenizer.countTokens(text, LITERAL));
        expect(countGptTokens(text)).toBe(exact);
        expect(countGptTokensWithinLimit(text, exact)).toBe(exact);
        expect(countGptTokensWithinLimit(text, exact - 1)).toBeNull();
      }
    },
  );

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

    const countSpy = spyOn(gptTokenizer, "countTokens");
    try {
      expect(countGptTokensWithinLimit(rejected, 1)).toBeNull();
      expect(countSpy).not.toHaveBeenCalled();
    } finally {
      countSpy.mockRestore();
    }
  });

  test("rejects a pathological over-budget span without entering gpt-tokenizer", () => {
    const text = `${"🚀".repeat(50_000)}!`;
    const countSpy = spyOn(gptTokenizer, "countTokens");
    try {
      expect(countGptTokensWithinLimit(text, 1_000)).toBeNull();
      expect(countSpy).not.toHaveBeenCalled();
    } finally {
      countSpy.mockRestore();
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

  // The canonical total still counts every file URL as text; reporting takes
  // the share of `data:` URLs out of it without changing it.
  test("partitions inline file data out of an unchanged canonical total", async () => {
    const pdf = `data:application/pdf;base64,${randomBytes(3_000).toString("base64")}`;
    const notes = `data:text/plain;base64,${Buffer.from("meeting notes").toString("base64")}`;
    const remote = "https://example.com/report.pdf";
    const chart = `data:image/png;base64,${randomBytes(600).toString("base64")}`;
    const messages = [
      {
        ...message("usr_files", "user"),
        parts: [
          textPart("usr_files", "Compare the attachments."),
          filePart(pdf),
          filePart(notes, "text/plain"),
          filePart(remote),
        ],
      },
      {
        ...message("ast_files", "assistant"),
        parts: [
          textPart("ast_files", "Chart attached."),
          filePart(chart, "image/png"),
        ],
      },
    ];
    const tokens = (texts: string[]) =>
      texts.reduce((total, text) => total + countGptTokens(text), 0);
    const inline = [pdf, notes, chart];

    const counted = await countModelFacingContentTokens(messages);

    expect(counted).toEqual({
      tokens: tokens([
        "Compare the attachments.",
        pdf,
        notes,
        remote,
        "Chart attached.",
        chart,
      ]),
      projectedMediaTokens: 0,
      inlineFileTokens: tokens(inline),
      inlineFiles: {
        count: 3,
        encodedBytes: inline.reduce((total, url) => total + url.length, 0),
      },
    });
  });

  test("keeps a projected image in the stored inventory and counts its placeholder as text", async () => {
    const url = `data:image/png;base64,${"A".repeat(4_096)}`;
    const summary = {
      ...textPart("ast_image", "Described the image."),
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
    } as Part;
    const messages = [
      {
        ...message("usr_image", "user"),
        parts: [textPart("usr_image", "Look."), filePart(url, "image/png")],
      },
      { ...message("ast_image", "assistant"), parts: [summary] },
    ];
    const placeholder = mediaPlaceholderText(1, "image/png");

    expect(await countModelFacingContentTokens(messages)).toEqual({
      tokens:
        countGptTokens("Look.")
        + countGptTokens(placeholder)
        + countGptTokens("Described the image."),
      projectedMediaTokens: countGptTokens(url) - countGptTokens(placeholder),
      inlineFileTokens: 0,
      inlineFiles: { count: 1, encodedBytes: url.length },
    });
  });

  // A trim counted an unchanged 19 MB PDF data URL whole, four times before
  // creating its successor, without returning to the event loop. Special-token
  // spellings in the payload count as ordinary text.
  test("counts a large part text exactly, yields to the event loop, and reuses the count for identical text", async () => {
    const url = `data:application/pdf;base64,${O200K_SPECIAL_TOKENS.join("")}${randomBytes(393_216).toString("base64")}${O200K_SPECIAL_TOKENS.join(" ")}`;
    const messages = [
      { ...message("usr_pdf", "user"), parts: [filePart(url)] },
    ];
    expect(url.length).toBeGreaterThan(65_536);
    const exact = gptTokenizer.countTokens(url, LITERAL);
    const encodeSpy = spyOn(gptTokenizer, "encodeGenerator");
    try {
      const order: string[] = [];
      setImmediate(() => order.push("immediate"));
      const counted = await countModelFacingContentTokens(messages).then(
        count => {
          order.push("count");
          return count;
        },
      );

      expect(order).toEqual(["immediate", "count"]);
      expect(counted).toEqual({
        tokens: exact,
        projectedMediaTokens: 0,
        inlineFileTokens: exact,
        inlineFiles: { count: 1, encodedBytes: url.length },
      });
      expect(await countModelFacingContentTokens(messages)).toEqual(counted);
      expect(encodeSpy).toHaveBeenCalledTimes(1);
    } finally {
      encodeSpy.mockRestore();
    }
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
});

function filePart(url: string, mime = "application/pdf"): Part {
  return {
    id: "prt_file",
    sessionID: "session",
    messageID: "usr_file",
    type: "file",
    mime,
    url,
  } as Part;
}

function textPart(messageID: string, text: string): Part {
  return {
    id: `${messageID}_text`,
    sessionID: "session",
    messageID,
    type: "text",
    text,
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

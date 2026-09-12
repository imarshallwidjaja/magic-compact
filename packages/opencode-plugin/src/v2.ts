import { createHash } from "node:crypto";
import type { Plugin } from "@opencode/plugin";
import type { SessionContext } from "@opencode/plugin/promise/session";
import { countTokens } from "gpt-tokenizer";
import { z } from "zod";

const settingsSchema = z
  .object({
    enabled: z.boolean(),
    keepTurns: z.number().int().min(1).max(Number.MAX_SAFE_INTEGER),
  })
  .strict();
const settingsInput = settingsSchema.partial().strict();
const statsSchema = z
  .object({
    outputs: z.number().int().min(0).max(Number.MAX_SAFE_INTEGER),
    tokensSaved: z.number().int().min(0).max(Number.MAX_SAFE_INTEGER),
  })
  .strict();
const readInput = z
  .object({ contentId: z.string().regex(/^v2-[a-f0-9]{64}$/) })
  .strict();
const defaults = { enabled: true, keepTurns: 1 };
const protectedTools = new Set([
  "question",
  "skill",
  "read_omitted_content",
  "magic_trim",
  "magic_stats",
]);

// Hash exact UTF-16 code units, including lone surrogates, and bind to the session.
function contentID(sessionID: string, text: string): string {
  return `v2-${createHash("sha256")
    .update(JSON.stringify([sessionID, text]), "utf16le")
    .digest("hex")}`;
}

export async function setup(ctx: Plugin.Context) {
  const settingsLocks = new Map<string, Promise<void>>();

  async function readSettings(key: string) {
    const result = settingsSchema.safeParse(await ctx.storage.get(key));
    return result.success ? result.data : defaults;
  }

  async function updateSettings(
    sessionID: string,
    change: z.infer<typeof settingsInput>,
  ) {
    const previous = settingsLocks.get(sessionID) ?? Promise.resolve();
    let release!: () => void;
    const lock = new Promise<void>(resolve => {
      release = resolve;
    });
    settingsLocks.set(sessionID, lock);
    await previous;

    try {
      const key = `settings/${sessionID}`;
      const settings = { ...(await readSettings(key)), ...change };
      await ctx.storage.set(key, settings);
      const stored = settingsSchema.safeParse(await ctx.storage.get(key));
      if (
        !stored.success
        || stored.data.enabled !== settings.enabled
        || stored.data.keepTurns !== settings.keepTurns
      ) {
        throw new Error(
          "Magic Compact V2 settings storage verification failed.",
        );
      }
      return settings;
    } finally {
      release();
      if (settingsLocks.get(sessionID) === lock)
        settingsLocks.delete(sessionID);
    }
  }

  async function project(event: SessionContext) {
    if (!event.sessionID)
      throw new Error("Magic Compact V2 requires a current session.");
    if (!event.tools.read_omitted_content) return;
    const settings = await readSettings(`settings/${event.sessionID}`);
    if (!settings.enabled) return;

    // A user group starts a request turn; the current turn is always protected.
    let turn = -1;
    let previousRole = "";
    const turns = event.messages.map(message => {
      if (message.role === "user" && previousRole !== "user") turn++;
      previousRole = message.role;
      return turn;
    });
    const boundary = turn - settings.keepTurns + 1;
    let outputs = 0;
    let tokensSaved = 0;
    const messages = [];
    for (const [index, message] of event.messages.entries()) {
      if (
        (message.role !== "tool" && message.role !== "assistant")
        || turns[index]! < 0
        || turns[index]! >= boundary
        || message.native !== undefined
      ) {
        messages.push(message);
        continue;
      }
      const content = [];
      for (const part of message.content) {
        if (
          part.type !== "tool-result"
          || part.providerExecuted
          || protectedTools.has(part.name)
        ) {
          content.push(part);
          continue;
        }
        const text =
          part.result.type === "text" && typeof part.result.value === "string"
            ? part.result.value
            : part.result.type === "content"
                && part.result.value.length === 1
                && part.result.value[0]?.type === "text"
              ? part.result.value[0].text
              : undefined;
        if (text === undefined || text.length <= 1024) {
          content.push(part);
          continue;
        }
        const id = contentID(event.sessionID, text);
        const notice = `[Magic Compact: historical tool output omitted. Use read_omitted_content with contentId "${id}" to retrieve the exact stale output.]`;
        const saved = countTokens(text) - countTokens(notice);
        if (saved <= 0) {
          content.push(part);
          continue;
        }
        const key = `omission/${event.sessionID}/${id}`;
        await ctx.storage.set(key, text);
        if ((await ctx.storage.get(key)) !== text) {
          throw new Error(
            "Magic Compact V2 omission storage verification failed.",
          );
        }
        content.push({
          ...part,
          result:
            part.result.type === "content"
              ? {
                  type: "content" as const,
                  value: [{ type: "text" as const, text: notice }],
                }
              : { type: "text" as const, value: notice },
        });
        outputs++;
        tokensSaved += saved;
      }
      messages.push({ ...message, content });
    }
    await ctx.storage.set(`stats/${event.sessionID}`, { outputs, tokensSaved });
    // Publish only after every original is durable and verified. Failure leaves input intact.
    event.messages = messages;
  }

  await ctx.tool.transform(editor => {
    editor.add({
      name: "magic_trim",
      options: { codemode: false },
      description:
        "Configure reversible Magic Compact V2 tool-output projection for this session. Defaults: enabled, keepTurns 1 (including the current user turn). Applies to subsequent agent-loop context requests; never edits saved history.",
      input: z.toJSONSchema(settingsInput),
      async execute(input, context) {
        const change = settingsInput.parse(input);
        const settings = await updateSettings(context.sessionID, change);
        return {
          content: `Magic Compact V2 projection settings: ${JSON.stringify(settings)}. No saved messages changed.`,
        };
      },
    });
    editor.add({
      name: "magic_stats",
      options: { codemode: false },
      description:
        "Read the last Magic Compact V2 request projection estimate. Not cumulative savings or provider usage; saved history is unchanged.",
      input: { type: "object", properties: {}, additionalProperties: false },
      async execute(_input, context) {
        const settings = await readSettings(`settings/${context.sessionID}`);
        const rawStats = await ctx.storage.get(`stats/${context.sessionID}`);
        const stats = statsSchema.safeParse(rawStats);
        return {
          content: JSON.stringify({
            settings,
            lastProjection: stats.success ? stats.data : null,
            ...(rawStats !== undefined && !stats.success
              ? {
                  diagnostic:
                    "Stored projection stats failed integrity validation.",
                }
              : {}),
            accounting:
              "Request-local tokenizer estimate, not provider usage or persisted pruning.",
          }),
        };
      },
    });
    editor.add({
      name: "read_omitted_content",
      options: { codemode: false },
      description:
        "Retrieve exact stale tool output from this V2 session's Magic Compact storage. Use only when a fresh lookup cannot answer the question.",
      input: z.toJSONSchema(readInput),
      async execute(input, context) {
        const { contentId } = readInput.parse(input);
        if (!context.sessionID)
          throw new Error("Magic Compact V2 requires a current session.");
        const text = await ctx.storage.get(
          `omission/${context.sessionID}/${contentId}`,
        );
        if (
          typeof text !== "string"
          || contentID(context.sessionID, text) !== contentId
        ) {
          throw new Error(
            "Omitted content unavailable or failed integrity validation for this session.",
          );
        }
        return { content: text };
      },
    });
  });
  await ctx.session.hook("context", project);
}

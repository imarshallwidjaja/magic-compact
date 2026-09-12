import { describe, expect, test } from "bun:test";
import type { Plugin } from "@opencode/plugin";
import type { SessionContext } from "@opencode/plugin/promise/session";
import type { Info, ToolContext } from "@opencode/plugin/promise/tool";
import plugin from "../src/index";
import tui from "../src/tui";

type Hook = (event: SessionContext) => Promise<void>;
const large =
  "Original output with enough varied words to profitably reduce context. ".repeat(
    100,
  );
const sessionID = "session-a" as SessionContext["sessionID"];

type StorageOverrides = {
  get?: (key: string, store: Map<string, unknown>) => Promise<unknown>;
  set?: (
    key: string,
    value: unknown,
    store: Map<string, unknown>,
  ) => Promise<void>;
};

async function harness(
  store = new Map<string, unknown>(),
  storage: StorageOverrides = {},
) {
  const hooks = new Map<string, Hook>();
  const tools = new Map<string, Info>();
  let transform: (editor: { add(tool: Info): void }) => void;
  let remainingWrites = Infinity;
  const context = {
    storage: {
      get: async (key: string) =>
        storage.get ? storage.get(key, store) : store.get(key),
      set: async (key: string, value: unknown) => {
        if (remainingWrites-- <= 0) throw new Error("storage offline");
        const copy = structuredClone(value);
        if (storage.set) await storage.set(key, copy, store);
        else store.set(key, copy);
      },
    },
    tool: {
      transform: async (edit: typeof transform) => {
        transform = edit;
        edit({ add: tool => tools.set(tool.name, tool) });
      },
    },
    session: {
      hook: async (name: string, hook: Hook) => hooks.set(name, hook),
    },
  };
  await plugin.setup(context as unknown as Plugin.Context);
  return {
    hooks,
    tools,
    store,
    failStorage(afterWrites = 0) {
      remainingWrites = afterWrites;
    },
    replay() {
      transform({ add: tool => tools.set(tool.name, tool) });
    },
    async call(name: string, input: unknown, session = sessionID) {
      return tools
        .get(name)!
        .execute(input, { sessionID: session } as ToolContext);
    },
  };
}

function event(): SessionContext {
  return {
    sessionID,
    agent: "agent" as SessionContext["agent"],
    model: { providerID: "test", modelID: "test" } as SessionContext["model"],
    system: [],
    options: {},
    tools: { read_omitted_content: { description: "read", input: {} } },
    messages: [
      { role: "user", content: [{ type: "text", text: "Exact user request" }] },
      {
        role: "assistant",
        content: [
          { type: "tool-call", id: "call", name: "read", input: { file: "a" } },
        ],
      },
      {
        role: "tool",
        content: [
          {
            type: "tool-result",
            id: "call",
            name: "read",
            result: { type: "text", value: large },
          },
        ],
      },
      { role: "user", content: [{ type: "text", text: "Current request" }] },
      {
        role: "tool",
        content: [
          {
            type: "tool-result",
            id: "current",
            name: "read",
            result: { type: "text", value: large },
          },
        ],
      },
    ],
  };
}

function omissionID(request: SessionContext) {
  return JSON.stringify(request.messages[2]).match(/v2-[a-f0-9]{64}/)![0];
}

describe("combined loader and V2 request projection", () => {
  test("exposes independent V1 server and genuine V2 setup registration", async () => {
    expect(plugin.id).toBe("arkive-compact");
    expect(tui.tui).toBeFunction();
    expect(tui.setup).toBeFunction();
    const v1 = await plugin.server({} as Parameters<typeof plugin.server>[0]);
    expect(v1["command.execute.before"]).toBeFunction();
    expect(v1["experimental.chat.messages.transform"]).toBeFunction();
    const h = await harness();
    expect([...h.hooks.keys()]).toEqual(["context"]);
    expect([...h.tools.keys()]).toEqual([
      "magic_trim",
      "magic_stats",
      "read_omitted_content",
    ]);
    h.replay();
    expect(h.tools.size).toBe(3);
    expect(h.tools.get("read_omitted_content")!.options?.codemode).toBe(false);
    expect(h.store.size).toBe(0);
  });

  test("projects only old output, preserves source objects and restores exact bytes after reload", async () => {
    const h = await harness();
    const request = event();
    const original = structuredClone(request);
    const source = request.messages;
    await h.hooks.get("context")!(request);
    expect(source).toEqual(original.messages);
    expect(request.messages[0]).toEqual(original.messages[0]);
    expect(request.messages[1]).toEqual(original.messages[1]);
    expect(request.messages.slice(3)).toEqual(original.messages.slice(3));
    expect(JSON.stringify(request.messages[2])).not.toContain(large);
    const id = omissionID(request);
    const reloaded = await harness(h.store);
    expect(
      (await reloaded.call("read_omitted_content", { contentId: id })).content,
    ).toBe(large);
    const stats = JSON.parse(
      (await h.call("magic_stats", {})).content as string,
    );
    expect(stats.lastProjection.outputs).toBe(1);
    expect(stats.lastProjection.tokensSaved).toBeGreaterThan(0);
    await h.hooks.get("context")!(event());
    expect(
      [...h.store.keys()].filter(key => key.startsWith("omission/")),
    ).toHaveLength(1);
  });

  test("protects all user parts, media, errors, JSON, provider execution and retrieval outputs", async () => {
    const h = await harness();
    const request = event();
    const result = request.messages[2]!.content[0]!;
    if (result.type !== "tool-result") throw new Error("fixture");
    request.messages[0]!.content = [
      result,
      { type: "media", mediaType: "image/png", data: "exact" },
    ];
    request.messages[2]!.content = [
      { ...result, name: "read_omitted_content" },
      { ...result, name: "question" },
      { ...result, providerExecuted: true },
      { ...result, result: { type: "error", value: large } },
      { ...result, result: { type: "text", value: "x".repeat(1024) } },
      { ...result, result: { type: "text", value: " ".repeat(1025) } },
      { ...result, result: { type: "json", value: { text: large } } },
      {
        ...result,
        result: {
          type: "content",
          value: [
            { type: "text", text: large },
            { type: "file", uri: "exact", mime: "image/png" },
          ],
        },
      },
    ];
    request.messages.splice(3, 0, {
      role: "tool",
      native: {},
      content: [result],
    });
    const before = structuredClone(request.messages);
    await h.hooks.get("context")!(request);
    expect(request.messages).toEqual(before);
  });

  test("supports native single-text content results and exact lone-surrogate recovery", async () => {
    const h = await harness();
    const request = event();
    const text = `${large}\ud800`;
    request.messages[2]!.content = [
      {
        type: "tool-result",
        id: "call",
        name: "read",
        result: { type: "content", value: [{ type: "text", text }] },
      },
    ];
    await h.hooks.get("context")!(request);
    const id = omissionID(request);
    expect(
      (await h.call("read_omitted_content", { contentId: id })).content,
    ).toBe(text);
    await expect(
      h.call(
        "read_omitted_content",
        { contentId: id },
        "session-b" as typeof sessionID,
      ),
    ).rejects.toThrow("unavailable");
    h.store.set(`omission/${sessionID}/${id}`, "corrupted");
    await expect(
      h.call("read_omitted_content", { contentId: id }),
    ).rejects.toThrow("integrity");
  });

  test("fails atomically on storage errors and skips projection without retrieval access", async () => {
    const h = await harness();
    const request = event();
    request.messages[2]!.content.push({
      type: "tool-result",
      id: "second",
      name: "read",
      result: { type: "text", value: `${large} second` },
    });
    const original = request.messages;
    h.failStorage(1);
    await expect(h.hooks.get("context")!(request)).rejects.toThrow(
      "storage offline",
    );
    expect(request.messages).toBe(original);
    expect(
      [...h.store.keys()].filter(key => key.startsWith("omission/")),
    ).toHaveLength(1);
    request.tools = {};
    await h.hooks.get("context")!(request);
    expect(request.messages).toBe(original);
  });

  test("settings persist, disable reverses projection, and unsafe keep counts are rejected", async () => {
    const h = await harness();
    for (const keepTurns of [0, -1, 1.5, Number.MAX_SAFE_INTEGER + 1]) {
      await expect(h.call("magic_trim", { keepTurns })).rejects.toThrow();
    }
    await h.call("magic_trim", { keepTurns: 2 });
    const request = event();
    const before = structuredClone(request.messages);
    await h.hooks.get("context")!(request);
    expect(request.messages).toEqual(before);
    await h.call("magic_trim", { enabled: false });
    const reloaded = await harness(h.store);
    await reloaded.hooks.get("context")!(request);
    expect(request.messages).toEqual(before);
  });

  test("serializes settings updates and reuses the lock after a queued failure", async () => {
    let releaseRead!: () => void;
    let markReadStarted!: () => void;
    const readStarted = new Promise<void>(resolve => {
      markReadStarted = resolve;
    });
    const readGate = new Promise<void>(resolve => {
      releaseRead = resolve;
    });
    let settingsReads = 0;
    let settingsWrites = 0;
    const key = `settings/${sessionID}`;
    const h = await harness(new Map(), {
      async get(storageKey, store) {
        if (storageKey === key) {
          settingsReads++;
          if (settingsReads === 1) markReadStarted();
          await readGate;
        }
        return store.get(storageKey);
      },
      async set(storageKey, value, store) {
        if (storageKey === key && ++settingsWrites === 2)
          throw new Error("queued settings write failed");
        store.set(storageKey, value);
      },
    });

    const keep = h.call("magic_trim", { keepTurns: 2 });
    await readStarted;
    const disable = h.call("magic_trim", { enabled: false });
    releaseRead();
    const results = await Promise.allSettled([keep, disable]);

    expect(results[0]!.status).toBe("fulfilled");
    expect(results[1]!.status).toBe("rejected");
    if (results[1]!.status === "rejected")
      expect(String(results[1]!.reason)).toContain(
        "queued settings write failed",
      );

    await h.call("magic_trim", { enabled: false });

    expect(h.store.get(key)).toEqual({ enabled: false, keepTurns: 2 });
  });

  test("does not report malformed persisted stats as a valid snapshot", async () => {
    const key = `stats/${sessionID}`;
    for (const invalid of [
      { outputs: -1, tokensSaved: 10 },
      { outputs: 1, tokensSaved: 10, future: true },
      { outputs: 1.5, tokensSaved: "invalid" },
    ]) {
      const h = await harness(new Map([[key, invalid]]));
      const stats = JSON.parse(
        (await h.call("magic_stats", {})).content as string,
      );

      expect(stats.lastProjection).toBeNull();
      expect(stats.diagnostic).toBe(
        "Stored projection stats failed integrity validation.",
      );
    }
  });

  test("verifies settings storage before reporting success", async () => {
    const key = `settings/${sessionID}`;
    const h = await harness(new Map(), {
      async set(storageKey, value, store) {
        if (storageKey !== key) store.set(storageKey, value);
      },
    });

    await expect(h.call("magic_trim", { enabled: false })).rejects.toThrow(
      "settings storage verification failed",
    );
  });

  test("uses defaults for invalid stored settings and repairs them on update", async () => {
    const key = `settings/${sessionID}`;
    for (const invalid of [
      { enabled: "invalid", keepTurns: 2 },
      { enabled: false, keepTurns: 2, future: true },
    ]) {
      const store = new Map<string, unknown>([[key, invalid]]);
      const h = await harness(store);
      const request = event();

      await h.hooks.get("context")!(request);
      expect(JSON.stringify(request.messages[2])).not.toContain(large);
      expect(
        JSON.parse((await h.call("magic_stats", {})).content as string),
      ).toMatchObject({
        settings: { enabled: true, keepTurns: 1 },
      });

      await h.call("magic_trim", { enabled: false });
      expect(store.get(key)).toEqual({ enabled: false, keepTurns: 1 });
    }
  });
});

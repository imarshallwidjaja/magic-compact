import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { createHash, randomBytes } from "node:crypto";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { GENERATION_PROTOCOL } from "../src/compact/constants";
import { setup } from "../src/host-v2";
import { taskAwareRetentionPolicy } from "../src/policy/retention";
import {
  RETENTION_KEEP_THRESHOLD_PERCENT_ENV,
  RetentionClassificationError,
} from "../src/retention/classifier";
import plugin from "../src/index";
import tuiPlugin from "../src/tui";
import { readOmittedContent } from "../src/storage/omission";
import { readStats } from "../src/storage/stats";
import {
  executeV2Compact,
  isDisposableV2HostRecord,
  planV2Compaction,
} from "../src/v2/compact";
import { generateV2Summaries, V2GenerationError } from "../src/v2/generation";
import { trimV2Tool, v2ToolCallView, v2ToolCalls } from "../src/v2/history";
import { connectV2Host, V2_ENDPOINT_REQUIRED } from "../src/v2/host";
import {
  countV2ModelFacingTokens,
  planV2MediaProjection,
  projectV2RequestMessages,
  v2VisibleImageIndexes,
} from "../src/v2/media";
import { generateV2SessionSummary, OwnedSessions } from "../src/v2/owned";
import {
  buildGenerationPrompt,
  estimateGenerationExplicitRequestTokens,
  GENERATION_RESPONSE_SCHEMA,
  generationTurnMarker,
} from "../src/policy/generation";
import {
  SourceNotAtRestError,
  SuccessorFailure,
} from "../src/policy/successor";
import { countGptTokens } from "../src/stats/tokenize";
import {
  readV2SourceAtRest,
  v2StorageKey,
  type V2Message,
} from "../src/v2/session";
import { executeV2Trim } from "../src/v2/trim";
import { scoredJevFetch, tokenLimitResponse } from "./retention-host";
import {
  assistantMessage,
  dependencyError,
  echoGeneration,
  fakeV2Host,
  fakeV2State,
  idleMessage,
  LOCATION,
  PRIVATE_SENTINEL,
  privateHostError,
  reachableStrings,
  sourceSession,
  toolItem,
  userMessage,
  type FakeReply,
  type FakeV2State,
} from "./v2-host";

const originalDataHome = process.env.XDG_DATA_HOME;
let storageDirectory: string;
beforeEach(async () => {
  storageDirectory = await mkdtemp(join(tmpdir(), "magic-compact-v2-test-"));
  process.env.XDG_DATA_HOME = storageDirectory;
});
afterEach(async () => {
  if (originalDataHome === undefined) delete process.env.XDG_DATA_HOME;
  else process.env.XDG_DATA_HOME = originalDataHome;
  await rm(storageDirectory, { recursive: true, force: true });
});

const words = (tag: string, count: number) =>
  Array.from({ length: count }, (_, index) => `${tag}-${index}`).join(" ");

function toolHistory(): V2Message[] {
  return [
    userMessage("msg_u1", "Write the file"),
    assistantMessage("msg_a1", [
      { type: "text", text: "Writing." },
      toolItem(
        "call_w",
        "write",
        { path: "/w.txt", content: words("W", 300) },
        [{ type: "text", text: "Wrote file." }],
      ),
    ]),
    idleMessage("msg_i1"),
    {
      id: "msg_sys",
      type: "system",
      text: 'The context under "a" changed',
      metadata: { notice: "instructions", instructionSources: ["api/a"] },
      time: { created: 5 },
    } as V2Message,
    userMessage("msg_u2", "Read it"),
    assistantMessage("msg_a2", [
      toolItem("call_r", "read", { path: "/w.txt" }, [
        { type: "text", text: words("R", 400) },
        { type: "file", uri: "file:///w.png", mime: "image/png" },
      ]),
    ]),
    idleMessage("msg_i2"),
  ];
}

// A subagent call that returned while its job was still running (the stored
// shape OpenCode 2 gives a background or detached call), then the job's
// notification.
function backgroundSubagent(
  input: Record<string, unknown>,
  notice: "completed" | "cancelled",
): V2Message[] {
  return [
    userMessage("msg_bg", "start"),
    assistantMessage("msg_bga", [
      toolItem(
        "call_bg",
        "subagent",
        input,
        [{ type: "text", text: "working in the background" }],
        { sessionID: "ses_kid", status: "running", truncated: false },
      ),
    ]),
    {
      id: "msg_note",
      type: "synthetic",
      text: `<subagent sessionID="ses_kid" state="${notice}">`,
      metadata: { source: "subagent", childID: "ses_kid", state: notice },
      time: { created: 99 },
    } as V2Message,
  ];
}

describe("OpenCode 2 package entry", () => {
  test("one default export serves both hosts and the TUI entry carries both shapes", () => {
    expect(plugin.id).toBe("arkive-compact");
    expect(typeof plugin.server).toBe("function");
    expect(typeof plugin.setup).toBe("function");
    expect(typeof tuiPlugin.tui).toBe("function");
    expect(typeof tuiPlugin.setup).toBe("function");
  });
});

// A ready successor opens in every plugin-equipped TUI still showing its
// source, whoever ran the command, headless clients included; other TUIs stay.
describe("successor switch in every TUI showing the source", () => {
  type Route = { sessionID: string };
  type Viewer = { route: Route; navigations: string[]; toasts: string[] };
  const lineage = (ready: boolean) => ({
    magicCompact: {
      successorOf: "ses_a",
      ready,
      outcome: "Trimmed.",
      operation: "trim",
      compactionCount: 0,
      createdAt: 1,
    },
  });

  // OpenCode 1: `session.updated` events and the `route` API.
  const v1Viewers = async (routes: string[]) => {
    const handlers: Array<(event: unknown) => void> = [];
    const viewers: Viewer[] = [];
    for (const sessionID of routes) {
      const viewer: Viewer = {
        route: { sessionID },
        navigations: [],
        toasts: [],
      };
      viewers.push(viewer);
      await tuiPlugin.tui(
        {
          event: {
            on: (type: string, handler: (event: unknown) => void) => {
              expect(type).toBe("session.updated");
              handlers.push(handler);
              return () => undefined;
            },
          },
          route: {
            get current() {
              return { name: "session", params: { ...viewer.route } };
            },
            navigate: (name: string, params: Route) => {
              expect(name).toBe("session");
              viewer.navigations.push(params.sessionID);
              viewer.route = params;
            },
          },
          lifecycle: { onDispose: () => undefined },
        } as never,
        undefined,
        {} as never,
      );
    }
    const publish = (id: string, metadata: unknown) => {
      for (const handler of handlers) {
        handler({ properties: { info: { id, metadata } } });
      }
    };
    return { viewers, publish };
  };

  // OpenCode 2: `session.metadata.updated` and the `ui.router` API.
  const v2Viewers = async (routes: string[]) => {
    const handlers: Array<(event: unknown) => void> = [];
    const viewers: Viewer[] = [];
    for (const sessionID of routes) {
      const viewer: Viewer = {
        route: { sessionID },
        navigations: [],
        toasts: [],
      };
      viewers.push(viewer);
      await tuiPlugin.setup({
        data: {
          on: (type: string, handler: (event: unknown) => void) => {
            expect(type).toBe("session.metadata.updated");
            handlers.push(handler);
            return () => undefined;
          },
        },
        ui: {
          router: {
            current: () => ({ type: "session", ...viewer.route }),
            navigate: (route: { type: string } & Route) => {
              expect(route.type).toBe("session");
              viewer.navigations.push(route.sessionID);
              viewer.route = { sessionID: route.sessionID };
            },
          },
          toast: {
            show: ({ message }: { message: string }) =>
              viewer.toasts.push(message),
          },
        },
      } as never);
    }
    const publish = (sessionID: string, metadata: unknown) => {
      for (const handler of handlers)
        handler({ data: { sessionID, metadata } });
    };
    return { viewers, publish };
  };

  for (const [host, open] of [
    ["OpenCode 1", v1Viewers],
    ["OpenCode 2", v2Viewers],
  ] as const) {
    test(`${host}: both TUIs on the source move once, and a TUI elsewhere stays`, async () => {
      const { viewers, publish } = await open(["ses_a", "ses_a", "ses_c"]);
      // A candidate that is not ready yet, or that failed, moves nobody.
      publish("ses_b", lineage(false));
      publish("ses_b", {});
      expect(viewers.map(viewer => viewer.navigations)).toEqual([[], [], []]);

      publish("ses_b", lineage(true));
      expect(viewers.map(viewer => viewer.navigations)).toEqual([
        ["ses_b"],
        ["ses_b"],
        [],
      ]);
      // A later update to the successor does not pull a TUI that went back
      // to the source away again.
      viewers[0]!.route = { sessionID: "ses_a" };
      publish("ses_b", lineage(true));
      expect(viewers.map(viewer => viewer.navigations)).toEqual([
        ["ses_b"],
        ["ses_b"],
        [],
      ]);
      if (host === "OpenCode 2") {
        expect(viewers.map(viewer => viewer.toasts)).toEqual([
          ["Trimmed."],
          ["Trimmed."],
          [],
        ]);
      }
    });
  }
});

describe("OpenCode 2 tool trim", () => {
  test("replaces only text output, keeps files in place, and keeps input references in the input", async () => {
    const tool = {
      type: "tool",
      id: "call",
      name: "read",
      state: {
        status: "completed",
        input: { path: "/x" },
        content: [
          { type: "file", uri: "file:///a.png", mime: "image/png" },
          { type: "text", text: "first" },
          { type: "file", uri: "file:///b.png", mime: "image/png" },
          { type: "text", text: "second" },
        ],
        metadata: { keep: 1 },
      },
      time: { created: 1 },
    } as never;
    const allocated: string[] = [];
    const trimmed = await trimV2Tool(tool, async entry => {
      allocated.push(entry.content);
      return "abcdefabcdef:omitted-AAAAAAAAAAAAAAAAAAAAAA";
    });
    expect(allocated).toEqual(["first\n\nsecond"]);
    const state = trimmed!.state as {
      content: Array<{ type: string; uri?: string; text?: string }>;
      metadata: unknown;
    };
    expect(state.content.map(item => item.type)).toEqual([
      "file",
      "text",
      "file",
    ]);
    expect(state.content[0]!.uri).toBe("file:///a.png");
    expect(state.content[1]!.text).toBe(
      "[Historical read result omitted (13 bytes); Content ID: abcdefabcdef:omitted-AAAAAAAAAAAAAAAAAAAAAA]",
    );
    expect(state.metadata).toEqual({
      keep: 1,
      magicCompact: { trimmed: true },
    });
    expect(await trimV2Tool(trimmed!, async () => "unused")).toBeNull();

    const write = await trimV2Tool(
      toolItem("call_w", "write", { content: words("W", 300) }, [
        { type: "text", text: "ok" },
      ]) as never,
      async () => "abcdefabcdef:omitted-BBBBBBBBBBBBBBBBBBBBBB",
    );
    const writeState = write!.state as {
      input: Record<string, unknown>;
      content: Array<{ text: string }>;
    };
    expect(writeState.input["content"]).toBe(
      "[Omitted; Content ID: abcdefabcdef:omitted-BBBBBBBBBBBBBBBBBBBBBB]",
    );
    expect(writeState.content).toEqual([{ type: "text", text: "ok" }]);
  });

  test("shortens the long input of a call an earlier trim marked and leaves its output alone", async () => {
    const marked = toolItem(
      "call_m",
      "write",
      { path: "/m.txt", content: words("M", 300) },
      [{ type: "text", text: words("O", 400) }],
      { magicCompact: { trimmed: true } },
    ) as never as Parameters<typeof trimV2Tool>[0];
    const trimmed = await trimV2Tool(
      marked,
      async () => "abcdefabcdef:omitted-CCCCCCCCCCCCCCCCCCCCCC",
    );
    const state = trimmed!.state as typeof marked.state & {
      input: Record<string, unknown>;
    };
    expect(state.input).toEqual({
      path: "/m.txt",
      content:
        "[Omitted; Content ID: abcdefabcdef:omitted-CCCCCCCCCCCCCCCCCCCCCC]",
    });
    expect((state as { content: unknown }).content).toEqual(
      (marked.state as { content: unknown }).content,
    );
    expect(state.metadata).toEqual({ magicCompact: { trimmed: true } });
    expect(await trimV2Tool(trimmed!, async () => "unused")).toBeNull();
  });

  const SUBAGENT_REPORT = words("report", 700);
  test.each([
    [
      "a native child session",
      { sessionID: "ses_0123abcXYZ" },
      "; child session: ses_0123abcXYZ",
    ],
    ["no metadata", undefined, ""],
    ["an empty metadata record", {}, ""],
    ["the OpenCode 1 metadata key", { sessionId: "ses_0123abcXYZ" }, ""],
    ["a non-native ID", { sessionID: "child-1" }, ""],
    ["an ID with a line break", { sessionID: "ses_ab\nIGNORE" }, ""],
    ["a non-string ID", { sessionID: 7 }, ""],
  ])(
    "names the child session of a long subagent report from %s",
    async (_name, metadata, clause) => {
      const call = toolItem(
        "call_s",
        "subagent",
        { prompt: "Check the suite." },
        [{ type: "text", text: SUBAGENT_REPORT }],
        { ...metadata, summary: "PRIVATE-METADATA" },
      ) as never as Parameters<typeof trimV2Tool>[0];
      const trimmed = await trimV2Tool(
        call,
        async () => "abcdefabcdef:omitted-DDDDDDDDDDDDDDDDDDDDDD",
      );
      const state = trimmed!.state as { content: Array<{ text: string }> };
      expect(state.content).toEqual([
        {
          type: "text",
          text: `[Task output omitted (${Buffer.byteLength(SUBAGENT_REPORT)} bytes)${clause}; Content ID: abcdefabcdef:omitted-DDDDDDDDDDDDDDDDDDDDDD]`,
        },
      ]);
      expect(JSON.stringify(state.content)).not.toContain("PRIVATE-METADATA");
    },
  );

  test("keeps a short subagent report and prompt unchanged and reads no child session from other tools", async () => {
    const report = [{ type: "text", text: "Short report." }];
    const subagent = toolItem(
      "call_k",
      "subagent",
      { prompt: "Check the suite." },
      report,
      { sessionID: "ses_0123abcXYZ" },
    ) as never as Parameters<typeof trimV2Tool>[0];
    const trimmed = await trimV2Tool(subagent, async () => {
      throw new Error("Nothing should be omitted.");
    });
    expect((trimmed!.state as { content: unknown }).content).toEqual(report);
    expect((trimmed!.state as { input: unknown }).input).toEqual({
      prompt: "Check the suite.",
    });
    const read = toolItem("call_k2", "read", {}, report, {
      sessionID: "ses_0123abcXYZ",
    }) as never as Parameters<typeof trimV2Tool>[0];
    expect(v2ToolCallView(read)?.childSessionID).toBeUndefined();
  });

  test("identifies calls by message, position and call ID", () => {
    const message = assistantMessage("msg_x", [
      toolItem("call_1", "read", {}, [{ type: "text", text: "a" }]),
      { type: "text", text: "between" },
      toolItem("call_1", "read", {}, [{ type: "text", text: "b" }]),
    ]);
    expect(v2ToolCalls(message).map(call => call.ref)).toEqual([
      "msg_x:0:call_1",
      "msg_x:2:call_1",
    ]);
  });
});

describe("OpenCode 2 source at rest", () => {
  const refusals: Array<[string, (state: FakeV2State) => void, RegExp]> = [
    [
      "a running session",
      state => {
        state.active["ses_src"] = { type: "running" };
      },
      /it is running/,
    ],
    [
      "a queued message",
      state => {
        state.inbox.set("ses_src", [{}]);
      },
      /queued message/,
    ],
    [
      "a pending permission",
      state => {
        state.permissions.set("ses_src", [{}]);
      },
      /permission request/,
    ],
    [
      "a pending question",
      state => {
        state.forms.set("ses_src", [{}]);
      },
      /question is waiting/,
    ],
    [
      "a pending revert",
      state => {
        state.sessions.get("ses_src")!.revert = {
          messageID: "msg_u1",
        } as never;
      },
      /pending revert/,
    ],
    [
      "a running background shell",
      state => {
        state.shells.push({
          id: "sh_1",
          status: "running",
          metadata: { sessionID: "ses_src" },
        });
      },
      /background command is still running/,
    ],
    [
      "an undelivered background result",
      state => {
        state.messages
          .get("ses_src")!
          .push(
            userMessage("msg_bg", "start"),
            assistantMessage("msg_bga", [
              toolItem(
                "call_bg",
                "shell",
                { command: "sleep", background: true },
                [{ type: "text", text: "moved" }],
                { shellID: "sh_9" },
              ),
            ]),
          );
      },
      /background command or subagent has not delivered/,
    ],
    [
      "a detached foreground call without its result",
      state => {
        state.messages
          .get("ses_src")!
          .push(
            userMessage("msg_bg", "start"),
            assistantMessage("msg_bga", [
              toolItem(
                "call_bg",
                "subagent",
                { prompt: "p", background: false },
                [{ type: "text", text: "working in the background" }],
                { sessionID: "ses_kid", status: "running" },
              ),
            ]),
          );
      },
      /background command or subagent has not delivered/,
    ],
    [
      "a cancelled background subagent that is still stopping",
      state => {
        state.messages
          .get("ses_src")!
          .push(
            ...backgroundSubagent(
              { prompt: "p", background: true },
              "cancelled",
            ),
          );
        state.active["ses_kid"] = { type: "running" };
      },
      /background command or subagent is still running/,
    ],
    [
      "a cancelled background command whose process still runs",
      state => {
        state.messages
          .get("ses_src")!
          .push(
            userMessage("msg_bg", "start"),
            assistantMessage("msg_bga", [
              toolItem(
                "call_bg",
                "shell",
                { command: "sleep", background: true },
                [{ type: "text", text: "moved" }],
                { shellID: "sh_9", status: "running" },
              ),
            ]),
            {
              id: "msg_note",
              type: "synthetic",
              text: "<shell>",
              metadata: {
                source: "shell",
                shellID: "sh_9",
                state: "cancelled",
              },
              time: { created: 99 },
            } as V2Message,
          );
        state.shells.push({ id: "sh_9", status: "running", metadata: {} });
      },
      /background command or subagent is still running/,
    ],
    [
      "a running subagent",
      state => {
        sourceSession(state, "ses_child", [], { parentID: "ses_src" } as never);
        state.active["ses_child"] = { type: "running" };
      },
      /subagent sessions is running/,
    ],
    [
      "an unfinished assistant",
      state => {
        state.messages
          .get("ses_src")!
          .push(assistantMessage("msg_open", [], { time: { created: 9 } }));
      },
      /Record: assistant message msg_open; no completion time or error\./,
    ],
  ];
  for (const [name, arrange, expected] of refusals) {
    test(`refuses ${name}`, async () => {
      const state = fakeV2State();
      sourceSession(state, "ses_src", toolHistory());
      arrange(state);
      const host = fakeV2Host(state, new OwnedSessions());
      await expect(
        readV2SourceAtRest(host, "Magic Trim", "ses_src", 1_000),
      ).rejects.toThrow(expected);
    });
  }

  test.each([
    [
      "an explicit background call",
      { prompt: "p", background: true },
      "completed",
    ],
    ["a detached foreground call", { prompt: "p" }, "completed"],
    [
      "a cancelled job that has stopped",
      { prompt: "p", background: true },
      "cancelled",
    ],
  ] as const)(
    "accepts %s once its notification is stored",
    async (_name, input, notice) => {
      const state = fakeV2State();
      sourceSession(state, "ses_src", [
        ...toolHistory(),
        ...backgroundSubagent(input, notice),
      ]);
      const source = await readV2SourceAtRest(
        fakeV2Host(state, new OwnedSessions()),
        "Magic Trim",
        "ses_src",
        1_000,
      );
      expect(source.messages).toHaveLength(10);
    },
  );

  describe("unfinished stored history", () => {
    const canary = "PRIVATE-CANARY-9b2d";
    const unfinishedTool = (id: string, name: string, status: string) => ({
      type: "tool",
      id,
      name,
      state: { status, input: { prompt: canary } },
      time: { created: 1 },
    });
    const refusal = async (
      arrange: (state: FakeV2State) => void,
      command: "Magic Compact" | "Magic Trim" = "Magic Trim",
    ) => {
      const state = fakeV2State();
      sourceSession(state, "ses_src", toolHistory());
      arrange(state);
      const error = await readV2SourceAtRest(
        fakeV2Host(state, new OwnedSessions()),
        command,
        "ses_src",
        1_000,
      ).then(
        () => undefined,
        (failure: unknown) => failure,
      );
      expect(error).toBeInstanceOf(SourceNotAtRestError);
      expect((error as Error).name).toBe("SourceNotAtRestError");
      return (error as Error).message;
    };

    test("names a terminal assistant message that still holds an unfinished tool call", async () => {
      const text = await refusal(state => {
        state.messages
          .get("ses_src")!
          .push(
            assistantMessage("msg_stale", [
              unfinishedTool("call_s", "subagent", "running"),
              unfinishedTool("call_t", "subagent", "streaming"),
            ]),
          );
      });
      expect(text).toEndWith(
        "Record: assistant message msg_stale; running subagent call call_s and 1 more call.",
      );
      expect(text).not.toContain(canary);
    });

    test("names an old blocker that later completed work did not clear", async () => {
      const text = await refusal(state => {
        const history = state.messages.get("ses_src")!;
        history.splice(
          0,
          0,
          assistantMessage("msg_old", [], { time: { created: 1 } }),
        );
      }, "Magic Compact");
      expect(text).toEndWith(
        "Record: assistant message msg_old; no completion time or error.",
      );
      expect(text).not.toContain("msg_a2");
    });

    test.each([
      ["compaction", "Record: compaction record msg_c; still marked running."],
      ["shell", "Record: shell record msg_c; still marked running."],
    ] as const)("names a running %s record", async (type, expected) => {
      const text = await refusal(state => {
        state.messages.get("ses_src")!.push({
          id: "msg_c",
          type,
          status: "running",
          time: { created: 9 },
        } as unknown as V2Message);
      });
      expect(text).toEndWith(expected);
    });

    test("names the subagent session whose history holds the record", async () => {
      const text = await refusal(state => {
        sourceSession(
          state,
          "ses_child",
          [assistantMessage("msg_kid", [], { time: { created: 9 } })],
          { parentID: "ses_src" } as never,
        );
      });
      expect(text).toEndWith(
        "Record: subagent session ses_child; assistant message msg_kid; no completion time or error.",
      );
    });

    test("reports an active session as running, not as stale history", async () => {
      const text = await refusal(state => {
        state.messages
          .get("ses_src")!
          .push(assistantMessage("msg_open", [], { time: { created: 9 } }));
        state.active["ses_src"] = { type: "running" };
      });
      expect(text).toContain("but it is running.");
      expect(text).not.toContain("unfinished record");
    });

    test("accepts an assistant message that ended with an error and no completion time", async () => {
      const state = fakeV2State();
      sourceSession(state, "ses_src", [
        ...toolHistory(),
        assistantMessage("msg_failed", [], {
          time: { created: 9 },
          error: { type: "unknown", message: "failed" },
        }),
      ]);
      await expect(
        readV2SourceAtRest(
          fakeV2Host(state, new OwnedSessions()),
          "Magic Trim",
          "ses_src",
          1_000,
        ),
      ).resolves.toMatchObject({ session: { id: "ses_src" } });
    });

    test("renders bounded, sanitized references", async () => {
      const text = await refusal(state => {
        state.messages
          .get("ses_src")!
          .push(
            assistantMessage(`msg_\n<b>${"x".repeat(200)}`, [
              unfinishedTool(
                `call\u0000 "q"${"z".repeat(200)}`,
                "t\n<i>",
                "run ning",
              ),
            ]),
          );
      });
      // The one line break separates the sentence from the record line.
      expect(text.replace("\n", " ")).not.toMatch(/[\p{C}<>"]/u);
      expect(text).toMatch(
        /message msg_\?+b\?x+\.\.\.; run\?ning t\?+i\? call call\?+q\?z+\.\.\.\.$/u,
      );
    });
  });

  test("refuses a subagent session", async () => {
    const state = fakeV2State();
    sourceSession(state, "ses_src", toolHistory(), {
      parentID: "ses_parent",
    } as never);
    await expect(
      readV2SourceAtRest(
        fakeV2Host(state, new OwnedSessions()),
        "Magic Trim",
        "ses_src",
        1_000,
      ),
    ).rejects.toThrow(/root session/);
  });
});

describe("OpenCode 2 endpoint qualification", () => {
  const session = {
    id: "ses_src",
    projectID: "p",
    time: { created: 1 },
    location: LOCATION,
  };
  const fetchFor = (pid: number, remote = session) =>
    (async (input: Request | string | URL) => {
      const url = new URL(input instanceof Request ? input.url : String(input));
      if (url.pathname === "/api/info")
        return Response.json({
          version: "2.0.28",
          pid,
          urls: [],
          paths: { tmp: "/tmp" },
        });
      return Response.json({ data: remote });
    }) as typeof fetch;

  test("accepts only an endpoint that reports this PID and the same source", async () => {
    const options = {
      serverUrl: "http://127.0.0.1:9",
      env: {},
      pid: 7,
      controlTimeoutMs: 1_000,
    };
    await expect(
      connectV2Host("ses_src", async () => session, {
        ...options,
        fetch: fetchFor(7),
      }),
    ).resolves.toBeDefined();
    await expect(
      connectV2Host("ses_src", async () => session, {
        ...options,
        fetch: fetchFor(8),
      }),
    ).rejects.toThrow(V2_ENDPOINT_REQUIRED);
    await expect(
      connectV2Host(
        "ses_src",
        async () => ({ ...session, projectID: "other" }),
        { ...options, fetch: fetchFor(7) },
      ),
    ).rejects.toThrow(V2_ENDPOINT_REQUIRED);
    await expect(
      connectV2Host("ses_src", async () => session, {
        ...options,
        serverUrl: 4101,
      }),
    ).rejects.toThrow(V2_ENDPOINT_REQUIRED);
  });

  test("never guesses an endpoint when no service is registered", async () => {
    await expect(
      connectV2Host("ses_src", async () => session, {
        discover: async () => undefined,
        controlTimeoutMs: 1_000,
      }),
    ).rejects.toThrow(V2_ENDPOINT_REQUIRED);
  });
});

describe("OpenCode 2 trim successors", () => {
  test("imports a verified successor with fresh IDs, current instructions, caches and stats", async () => {
    const state = fakeV2State();
    const source = sourceSession(state, "ses_src", toolHistory());
    state.instructions.set("ses_src", [{ key: "a", value: "ALPHA-v2" }]);
    const before = structuredClone(state.messages.get("ses_src"));
    const host = fakeV2Host(state, new OwnedSessions());

    const { successor, outcome } = await executeV2Trim(
      host,
      new OwnedSessions(),
      "ses_src",
      0,
      {
        retention: { mode: "heuristic" },
        controlTimeoutMs: 1_000,
      },
    );

    expect(successor).not.toBeNull();
    expect(outcome).toContain("Tool calls trimmed: 2");
    expect(state.messages.get("ses_src")).toEqual(before!);
    expect(state.sessions.get("ses_src")).toEqual(source);
    const id = successor!.session.id;
    const info = state.sessions.get(id)!;
    expect(info.parentID).toBeUndefined();
    expect(info.title).toBe("Source");
    expect(info.metadata).toMatchObject({
      seed: "kept",
      magicCompact: { successorOf: "ses_src", operation: "trim", ready: true },
    });
    expect(state.instructions.get(id)).toEqual([
      { key: "a", value: "ALPHA-v2" },
    ]);
    const messages = state.messages.get(id)!;
    expect(messages.map(message => message.type)).toEqual([
      "user",
      "assistant",
      "idle",
      "user",
      "assistant",
      "idle",
    ]);
    expect(
      messages.some(message => before!.some(old => old.id === message.id)),
    ).toBeFalse();
    expect(messages.filter(message => message.type === "user")).toEqual(
      before!
        .filter(message => message.type === "user")
        .map((message, index) => ({
          ...message,
          id: messages.filter(m => m.type === "user")[index]!.id,
        })),
    );
    const read = v2ToolCalls(messages[4]!)[0]!.tool;
    const notice = (read.state as { content: Array<{ text?: string }> })
      .content[0]!.text!;
    const contentID = /Content ID: ([^\s\]]+)/u.exec(notice)![1]!;
    expect(notice).toBe(
      `[Historical read result omitted (${Buffer.byteLength(words("R", 400))} bytes); Content ID: ${contentID}]`,
    );
    expect(await readOmittedContent(v2StorageKey(id), contentID)).toBe(
      words("R", 400),
    );
    expect(
      await readOmittedContent(v2StorageKey("ses_src"), contentID),
    ).toBeNull();
    const stats = await readStats(v2StorageKey(id));
    expect(stats).toMatchObject({
      sourceSessionId: "ses_src",
      rootSessionId: "ses_src",
    });
    expect(stats!.processedAssistantMessageIds).toEqual(
      messages
        .filter(message => message.type === "assistant")
        .map(message => message.id),
    );
    expect(await readStats("ses_src")).toBeNull();
  });

  test("discards an unmodified candidate when the source changes before handoff", async () => {
    const state = fakeV2State();
    sourceSession(state, "ses_src", toolHistory());
    state.afterWrite = method => {
      if (method === "instructions" || method === "import") {
        state.messages.get("ses_src")!.push(userMessage("msg_late", "late"));
        state.afterWrite = undefined;
      }
    };
    const host = fakeV2Host(state, new OwnedSessions());
    await expect(
      executeV2Trim(host, new OwnedSessions(), "ses_src", 0, {
        retention: { mode: "heuristic" },
        controlTimeoutMs: 1_000,
      }),
    ).rejects.toThrow(/session changed while Magic Trim/);
    expect([...state.sessions.keys()]).toEqual(["ses_src"]);
  });

  test("keeps and names a candidate another writer changed", async () => {
    const state = fakeV2State();
    sourceSession(state, "ses_src", toolHistory());
    state.afterWrite = (method, sessionID) => {
      if (method === "import")
        state.messages
          .get(sessionID)!
          .push(userMessage("msg_other", "other writer"));
    };
    const host = fakeV2Host(state, new OwnedSessions());
    const error = await executeV2Trim(host, new OwnedSessions(), "ses_src", 0, {
      retention: { mode: "heuristic" },
      controlTimeoutMs: 1_000,
    }).catch((caught: Error) => caught);
    const kept = [...state.sessions.keys()].find(id => id !== "ses_src")!;
    expect(String(error)).toContain(`Successor candidate ${kept} was kept`);
  });

  test("a failed import leaves no candidate and names none", async () => {
    const state = fakeV2State();
    sourceSession(state, "ses_src", toolHistory());
    state.failImport = true;
    const error = await executeV2Trim(
      fakeV2Host(state, new OwnedSessions()),
      new OwnedSessions(),
      "ses_src",
      0,
      {
        retention: { mode: "heuristic" },
        controlTimeoutMs: 1_000,
      },
    ).catch((caught: Error) => caught);
    expect(String(error)).toContain(
      "could not finish preparing its successor session. The original session was left unchanged.",
    );
    expect(String(error)).not.toContain("was kept");
    expect([...state.sessions.keys()]).toEqual(["ses_src"]);
  });

  const trim = (state: FakeV2State) =>
    executeV2Trim(
      fakeV2Host(state, new OwnedSessions()),
      new OwnedSessions(),
      "ses_src",
      0,
      { retention: { mode: "heuristic" }, controlTimeoutMs: 1_000 },
    ).catch((caught: unknown) => caught);
  const candidateID = (state: FakeV2State) =>
    [...state.sessions.keys()].find(
      id => id !== "ses_src" && state.sessions.get(id)!.parentID === undefined,
    );

  // Each change lands after the candidate's first verification, while its
  // instructions are copied.
  const foreign: Array<[string, (state: FakeV2State, id: string) => void]> = [
    [
      "title",
      (state, id) => {
        state.sessions.get(id)!.title = "Renamed elsewhere";
      },
    ],
    [
      "permissions",
      (state, id) => {
        state.sessions.get(id)!.permissions = [];
      },
    ],
    [
      "model",
      (state, id) => {
        state.sessions.get(id)!.model = { providerID: "other", id: "m" };
      },
    ],
    [
      "archive time",
      (state, id) => {
        state.sessions.get(id)!.time.archived = 7;
      },
    ],
    [
      "instructions",
      (state, id) => {
        state.instructions.get(id)!.push({ key: "foreign", value: "x" });
      },
    ],
  ];
  for (const [name, change] of foreign) {
    test(`keeps a candidate whose ${name} another writer changed after its first check`, async () => {
      const state = fakeV2State();
      sourceSession(state, "ses_src", toolHistory());
      state.instructions.set("ses_src", [{ key: "a", value: "ALPHA" }]);
      state.afterWrite = (method, sessionID) => {
        if (method === "instructions") change(state, sessionID);
      };
      const error = await trim(state);
      const kept = candidateID(state)!;
      expect(String(error)).toContain(`Successor candidate ${kept} was kept`);
      expect(state.calls).not.toContain(`update:${kept}`);
      expect(state.calls).not.toContain(`remove:${kept}`);
    });
  }

  const work: Array<[string, (state: FakeV2State, id: string) => void]> = [
    [
      "a child session",
      (state, id) => {
        sourceSession(state, "ses_kid", [], { parentID: id } as never);
      },
    ],
    ["a queued message", (state, id) => void state.inbox.set(id, [{}])],
    [
      "activity",
      (state, id) => {
        state.active[id] = { type: "running" };
      },
    ],
    [
      "a permission request",
      (state, id) => void state.permissions.set(id, [{}]),
    ],
    ["a question", (state, id) => void state.forms.set(id, [{}])],
  ];
  for (const [name, arrange] of work) {
    test(`keeps a candidate that has ${name} when the source changes before handoff`, async () => {
      const state = fakeV2State();
      sourceSession(state, "ses_src", toolHistory());
      state.afterWrite = (method, sessionID) => {
        if (method !== "import") return;
        arrange(state, sessionID);
        state.messages.get("ses_src")!.push(userMessage("msg_late", "late"));
      };
      const error = await trim(state);
      const kept = candidateID(state)!;
      expect(String(error)).toContain(`Successor candidate ${kept} was kept`);
      expect(state.calls).not.toContain(`remove:${kept}`);
      if (name === "a child session")
        expect(state.sessions.get("ses_kid")?.parentID).toBe(kept);
    });
  }

  test("deletes a candidate holding only part of the source instructions", async () => {
    const state = fakeV2State();
    sourceSession(state, "ses_src", toolHistory());
    state.instructions.set("ses_src", [
      { key: "a", value: "ALPHA" },
      { key: "b", value: "BETA" },
    ]);
    let puts = 0;
    state.beforeWrite = method => {
      if (method === "instructions" && ++puts === 2)
        throw new Error("put failed");
    };
    const error = await trim(state);
    expect(String(error)).toContain(
      "could not finish preparing its successor session. The original session was left unchanged.",
    );
    expect(String(error)).not.toContain("was kept");
    expect([...state.sessions.keys()]).toEqual(["ses_src"]);
  });

  const hostFailures: Array<
    [string, string, (state: FakeV2State) => void, string]
  > = [
    [
      "import",
      "import",
      state => {
        state.beforeWrite = method => {
          if (method === "import") throw privateHostError();
        };
      },
      "could not finish preparing its successor session",
    ],
    [
      "instructions",
      "instructions",
      state => {
        state.instructions.set("ses_src", [{ key: "a", value: "ALPHA" }]);
        state.beforeWrite = method => {
          if (method === "instructions") throw privateHostError();
        };
      },
      "could not finish preparing its successor session",
    ],
    [
      "ready",
      "ready",
      state => {
        state.beforeWrite = method => {
          if (method === "update") throw privateHostError();
        };
      },
      "could not confirm that successor session",
    ],
    [
      "cleanup",
      "recheck",
      state => {
        state.afterWrite = method => {
          if (method === "import")
            state.messages.get("ses_src")!.push(userMessage("msg_late", "x"));
        };
        state.beforeWrite = method => {
          if (method === "remove") throw privateHostError();
        };
      },
      "was kept because it could not be confirmed as unmodified",
    ],
  ];
  for (const [name, stage, arrange, text] of hostFailures) {
    test(`reports a host failure at ${name} by stage and fixed text only`, async () => {
      const state = fakeV2State();
      sourceSession(state, "ses_src", toolHistory());
      arrange(state);
      const error = await trim(state);
      expect(error).toBeInstanceOf(SuccessorFailure);
      expect(String(error)).toContain(text);
      expect((error as SuccessorFailure).details).toMatchObject({
        stage,
        sourceID: "ses_src",
      });
      expect((error as SuccessorFailure).details.code).toBeUndefined();
      expect(error).not.toHaveProperty("cause");
      expect(
        reachableStrings(error).filter(value =>
          value.includes(PRIVATE_SENTINEL),
        ),
      ).toEqual([]);
    });
  }

  test("task-aware trim keeps a call Jev judges useful, after one owned summary request", async () => {
    const state = fakeV2State();
    sourceSession(state, "ses_src", toolHistory());
    const owned = new OwnedSessions();
    const host = fakeV2Host(state, owned, () => ({
      text: "## Objective\n- Continue.",
    }));
    const { successor, outcome } = await executeV2Trim(
      host,
      owned,
      "ses_src",
      0,
      {
        retention: {
          mode: "task-aware",
          apiKey: "test-key",
          fetch: scoredJevFetch(output =>
            output.startsWith("R-0") ? 0.9 : 0.1,
          ),
        },
        controlTimeoutMs: 1_000,
      },
    );
    expect(state.requests).toHaveLength(1);
    expect(outcome).toContain(
      "Task-aware retention kept 1 of 2 evaluated tool call(s) whole (read 1).",
    );
    const read = v2ToolCalls(state.messages.get(successor!.session.id)![4]!)[0]!
      .tool;
    expect(
      (read.state as { content: Array<{ text?: string }> }).content[0]!.text,
    ).toBe(words("R", 400));
    expect([...state.sessions.keys()]).toHaveLength(2);
  });

  test.each([
    [undefined, 1],
    ["80", 0],
  ])(
    "a 0.7 probability keeps the call by default and not at percent %p (kept=%p)",
    async (percent, kept) => {
      const state = fakeV2State();
      sourceSession(state, "ses_src", toolHistory());
      const owned = new OwnedSessions();
      const host = fakeV2Host(state, owned, () => ({
        text: "## Objective\n- Continue.",
      }));
      const { outcome } = await executeV2Trim(host, owned, "ses_src", 0, {
        retention: {
          ...taskAwareRetentionPolicy({
            TYPESAFE_API_KEY: "test-key",
            ...(percent === undefined
              ? {}
              : { [RETENTION_KEEP_THRESHOLD_PERCENT_ENV]: percent }),
          }),
          fetch: scoredJevFetch(output =>
            output.startsWith("R-0") ? 0.7 : 0.1,
          ),
        },
        controlTimeoutMs: 1_000,
      });
      expect(outcome).toContain(
        `Task-aware retention kept ${kept} of 2 evaluated tool call(s) whole`,
      );
    },
  );

  test("task-aware trim resolves the summarizer at the source session location", async () => {
    const state = fakeV2State();
    state.summarizerModel = { providerID: "probe", id: "default-model" };
    state.projectSummarizers = {
      "/other-project": { providerID: "probe", id: "other-model" },
    };
    sourceSession(state, "ses_src", toolHistory(), {
      location: { directory: "/other-project" },
    });
    const owned = new OwnedSessions();
    const host = fakeV2Host(state, owned, () => ({
      text: "## Objective\n- Continue.",
    }));
    const agentLocations: unknown[] = [];
    const agentGet = host.agent.get.bind(host.agent);
    host.agent.get = (async (...args: Parameters<typeof agentGet>) => {
      agentLocations.push(args[0].location);
      return agentGet(...args);
    }) as typeof host.agent.get;
    const { successor } = await executeV2Trim(host, owned, "ses_src", 0, {
      retention: {
        mode: "task-aware",
        apiKey: "test-key",
        fetch: scoredJevFetch(() => 0.1),
      },
      controlTimeoutMs: 1_000,
    });
    expect(successor).not.toBeNull();
    expect(agentLocations).toEqual([{ directory: "/other-project" }]);
  });

  test("a summarizer that calls a tool is stopped before a second request and fails the command", async () => {
    const state = fakeV2State();
    sourceSession(state, "ses_src", toolHistory());
    const owned = new OwnedSessions();
    const host = fakeV2Host(state, owned, () => ({
      tool: [{ name: "read", input: {} }],
    }));
    await expect(
      executeV2Trim(host, owned, "ses_src", 0, {
        retention: {
          mode: "task-aware",
          apiKey: "test-key",
          fetch: scoredJevFetch(() => 0.1),
        },
        controlTimeoutMs: 1_000,
      }),
    ).rejects.toThrow(/summary failed \(validation\)/);
    expect(state.requests).toHaveLength(1);
    expect([...state.sessions.keys()]).toEqual(["ses_src"]);
  });

  test("a Jev token-limit refusal fails task-aware trim with the manual fallback and no successor", async () => {
    const state = fakeV2State();
    sourceSession(state, "ses_src", toolHistory());
    const before = structuredClone(state.messages.get("ses_src"));
    const owned = new OwnedSessions();
    const host = fakeV2Host(state, owned, () => ({
      text: "## Objective\n- Continue.",
    }));
    let jevCalls = 0;
    const error = await executeV2Trim(host, owned, "ses_src", 0, {
      retention: {
        mode: "task-aware",
        apiKey: "test-key",
        fetch: async () => {
          jevCalls += 1;
          return tokenLimitResponse();
        },
      },
      controlTimeoutMs: 1_000,
    }).then(
      () => undefined,
      (caught: unknown) => caught,
    );
    expect(error).toBeInstanceOf(RetentionClassificationError);
    expect((error as RetentionClassificationError).kind).toBe("token-limit");
    expect(String(error)).toContain(
      "no retention decisions were applied. Run /magic-trim --heuristic",
    );
    // Both candidates were admitted together; nothing was retried.
    expect(jevCalls).toBe(2);
    expect(state.messages.get("ses_src")).toEqual(before);
    expect([...state.sessions.keys()]).toEqual(["ses_src"]);
  });
});

describe("OpenCode 2 generation responses", () => {
  const target = [
    userMessage("msg_t", "Do it"),
    assistantMessage("msg_ta", [{ type: "text", text: words("A", 200) }]),
    idleMessage("msg_ti"),
  ];
  const run = async (reply: (text: string) => FakeReply) => {
    const state = fakeV2State();
    const source = sourceSession(state, "ses_src", target);
    const owned = new OwnedSessions();
    const host = fakeV2Host(state, owned, reply);
    const outcome = await generateV2Summaries(
      host,
      owned,
      { session: source, messages: target, instructions: [] },
      {
        ref: state.summarizerModel!,
        info: {
          limit: state.modelLimit,
          capabilities: { tools: true, input: ["text"], output: ["text"] },
        },
      },
      [{ span: target }],
      "index",
      { controlTimeoutMs: 1_000 },
    ).catch((error: Error) => error);
    return { state, outcome };
  };
  const call = (input: unknown): FakeReply => ({
    tool: [{ name: "StructuredOutput", input }],
  });
  const echoed = (text: string) =>
    (
      echoGeneration()(text) as {
        tool: Array<{ input: { turns: Array<Record<string, unknown>> } }>;
      }
    ).tool[0]!.input;

  test("accepts exactly one valid response-tool call and removes the owned session", async () => {
    const { state, outcome } = await run(echoGeneration());
    expect(outcome).toEqual([{ text: "Summary of target 1." }]);
    expect(state.requests).toHaveLength(1);
    expect([...state.sessions.keys()]).toEqual(["ses_src"]);
  });

  const invalid: Array<[string, (text: string) => FakeReply, RegExp]> = [
    [
      "a string ordinal",
      text => {
        const input = echoed(text);
        input.turns[0]!["ordinal"] = "1";
        return call(input);
      },
      /ordinal at target ordinal 1/,
    ],
    [
      "an extra key",
      text => {
        const input = echoed(text);
        input.turns[0]!["extra"] = true;
        return call(input);
      },
      /shape at target ordinal 1/,
    ],
    [
      "a duplicate call",
      text => ({
        tool: [
          { name: "StructuredOutput", input: echoed(text) },
          { name: "StructuredOutput", input: echoed(text) },
        ],
      }),
      /turn count mismatch \(expected 1, received 2\)/,
    ],
    [
      "a plain text reply",
      () => ({ text: "plain" }),
      /missing structured output/,
    ],
    [
      "a call the host rejected",
      text => ({
        tool: [
          { name: "StructuredOutput", input: echoed(text), status: "error" },
        ],
      }),
      /missing structured output/,
    ],
    [
      "a provider error",
      () => ({ error: { type: "provider.invalid-request", message: "bad" } }),
      /model request failed/,
    ],
    [
      "a context overflow",
      () => ({
        error: { type: "provider.context-overflow", message: "too long" },
      }),
      /Run \/magic-trim first/,
    ],
  ];
  for (const [name, reply, expected] of invalid) {
    test(`rejects ${name} after one provider request`, async () => {
      const { state, outcome } = await run(reply);
      expect(String(outcome)).toMatch(expected);
      expect(state.requests).toHaveLength(1);
      expect([...state.sessions.keys()]).toEqual(["ses_src"]);
    });
  }

  test("names each distinct reason of several failed turns once", async () => {
    const second = [
      userMessage("msg_t2", "Do it again"),
      assistantMessage("msg_t2a", [{ type: "text", text: words("B", 200) }]),
      idleMessage("msg_t2i"),
    ];
    const history = [...target, ...second];
    const state = fakeV2State();
    const source = sourceSession(state, "ses_src", history);
    const owned = new OwnedSessions();
    const outcome = await generateV2Summaries(
      fakeV2Host(state, owned, () => ({
        error: { type: "provider.invalid-request", message: "bad" },
      })),
      owned,
      { session: source, messages: history, instructions: [] },
      {
        ref: state.summarizerModel!,
        info: {
          limit: state.modelLimit,
          capabilities: { tools: true, input: ["text"], output: ["text"] },
        },
      },
      [{ span: target }, { span: second }],
      "index",
      { controlTimeoutMs: 1_000 },
    ).catch((error: Error) => error);
    expect((outcome as Error).message).toBe(
      "Compaction generation failed for 2 turns: Compaction generation model request failed.",
    );
  });

  test("names every temporary session that could not be deleted when five started batches fail", async () => {
    // Six turns, five concurrent batches. Every request fails and every
    // removal is refused, so five sessions stay behind and the sixth turn is
    // never admitted. The aggregate explains the failure once and still names
    // all five sessions.
    const extra = Array.from({ length: 5 }, (_, index) => [
      userMessage(`msg_x${index}`, `Turn ${index + 2}`),
      assistantMessage(`msg_x${index}a`, [
        { type: "text", text: words(`X${index}`, 200) },
      ]),
      idleMessage(`msg_x${index}i`),
    ]).flat();
    const history = [...target, ...extra];
    const state = fakeV2State();
    const source = sourceSession(state, "ses_src", history);
    const before = structuredClone(state.messages.get("ses_src"));
    state.beforeWrite = method => {
      if (method === "remove") throw new Error("connection reset");
    };
    const owned = new OwnedSessions();
    const outcome = await generateV2Summaries(
      fakeV2Host(state, owned, () => ({
        error: { type: "provider.invalid-request", message: "bad" },
      })),
      owned,
      { session: source, messages: history, instructions: [] },
      {
        ref: state.summarizerModel!,
        info: {
          limit: state.modelLimit,
          capabilities: { tools: true, input: ["text"], output: ["text"] },
        },
      },
      [
        { span: target },
        ...Array.from({ length: 5 }, (_, index) => ({
          span: extra.slice(index * 3, index * 3 + 3),
        })),
      ],
      "index",
      { controlTimeoutMs: 1_000 },
    ).catch((error: Error) => error);

    const ids = state.calls
      .filter(entry => entry.startsWith("import:"))
      .map(entry => entry.slice("import:".length));
    expect(ids).toHaveLength(5);
    expect(new Set(ids).size).toBe(5);
    expect(outcome).toBeInstanceOf(V2GenerationError);
    expect((outcome as V2GenerationError).undeletedSessionIDs).toEqual(ids);
    expect((outcome as Error).message).toBe(
      `Compaction generation failed for 5 turns: Compaction generation model request failed. Temporary sessions ${ids.join(", ")} could not be deleted.`,
    );
    for (const id of ids) {
      // Each stays registered and refuses new provider requests.
      expect(state.sessions.has(id)).toBeTrue();
      expect(owned.has(id)).toBeTrue();
      expect(() => owned.admitRequest(id)).toThrow(owned.stop);
      expect(() => owned.rejectCompaction(id)).toThrow(owned.compactionBlocked);
    }
    // The source is unchanged and the sixth turn started nothing.
    expect(state.messages.get("ses_src")).toEqual(before);
    expect([...state.sessions.keys()]).toEqual(["ses_src", ...ids]);
    expect(state.requests).toHaveLength(5);
  });

  // Each owned session's lifecycle requests are awaited until they settle;
  // the 10 ms control deadline below bounds only reads.
  const generate = async (
    arrange: (state: FakeV2State) => void,
    reply: (text: string) => FakeReply = echoGeneration(),
  ) => {
    const state = fakeV2State();
    const source = sourceSession(state, "ses_src", target);
    arrange(state);
    const owned = new OwnedSessions();
    const outcome = await generateV2Summaries(
      fakeV2Host(state, owned, reply),
      owned,
      { session: source, messages: target, instructions: [] },
      {
        ref: state.summarizerModel!,
        info: {
          limit: state.modelLimit,
          capabilities: { tools: true, input: ["text"], output: ["text"] },
        },
      },
      [{ span: target }],
      "index",
      { controlTimeoutMs: 10 },
    ).catch((error: Error) => error);
    const ownedID = state.calls
      .find(entry => entry.startsWith("import:"))
      ?.slice("import:".length);
    return { state, owned, outcome, ownedID };
  };
  const delay = (ms: number) =>
    new Promise<void>(resolve => setTimeout(resolve, ms));

  test.each(["import", "remove"])(
    "waits for a %s that settles after the control deadline and leaves no session behind",
    async method => {
      const { state, owned, outcome, ownedID } = await generate(state => {
        state.beforeWrite = async called => {
          if (called === method) await delay(50);
        };
      });
      expect(outcome).toEqual([{ text: "Summary of target 1." }]);
      expect([...state.sessions.keys()]).toEqual(["ses_src"]);
      expect(owned.has(ownedID!)).toBeFalse();
    },
  );

  test("keeps stopping an owned session whose removal failed before it ran", async () => {
    const { state, owned, outcome, ownedID } = await generate(state => {
      state.beforeWrite = method => {
        if (method === "remove") throw new Error("connection reset");
      };
    });
    expect(String(outcome)).toContain(
      `temporary session ${ownedID} could not be deleted`,
    );
    expect(state.sessions.has(ownedID!)).toBeTrue();
    expect(owned.has(ownedID!)).toBeTrue();
    expect(() => owned.admitRequest(ownedID!)).toThrow(owned.stop);
    expect(() => owned.rejectCompaction(ownedID!)).toThrow(
      owned.compactionBlocked,
    );
  });

  test("removes an owned session whose prompt the host accepted before the request failed", async () => {
    const { state, owned, outcome, ownedID } = await generate(state => {
      state.rejectAcceptedPrompt = true;
    });
    expect(String(outcome)).toMatch(/model request failed/);
    expect(state.requests).toHaveLength(1);
    expect(state.calls).toContain(`remove:${ownedID}`);
    expect(owned.has(ownedID!)).toBeFalse();
  });

  test("reports a failed import without a cleanup failure and keeps its guard", async () => {
    const { state, owned, outcome } = await generate(state => {
      state.beforeWrite = (method, sessionID) => {
        if (method === "import") {
          state.calls.push(`import:${sessionID}`);
          throw new Error("import failed");
        }
      };
    });
    const ownedID = state.calls[0]!.slice("import:".length);
    expect(String(outcome)).toContain("session creation failed");
    expect(String(outcome)).not.toContain("could not be deleted");
    expect([...state.sessions.keys()]).toEqual(["ses_src"]);
    // The failed import could still be applied later; its ID stays stopped,
    // and the host's answer that no such session exists keeps it so.
    expect(owned.has(ownedID)).toBeTrue();
    expect(() => owned.admitRequest(ownedID)).toThrow(owned.stop);
    expect(state.requests).toHaveLength(0);
  });

  // The host may apply an import and only then fail the request, or hand back
  // history that differs from what was sent. Either way the session exists
  // remotely without a prompt; it must never serve a provider request later.
  test.each([
    [
      "an import the host applied before the request failed",
      (state: FakeV2State) => {
        state.afterWrite = method => {
          if (method === "import") throw new Error("response lost");
        };
      },
      /session creation failed/,
    ],
    [
      "history verification that failed before prompting",
      (state: FakeV2State) => {
        state.afterWrite = (method, sessionID) => {
          if (method === "import") {
            state.messages.get(sessionID)!.push(userMessage("msg_extra", "x"));
          }
        };
      },
      /history verification failed/,
    ],
  ])(
    "closes the session after %s and keeps it closed when removal fails",
    async (_name, arrange, failure) => {
      const { state, owned, outcome, ownedID } = await generate(state => {
        arrange(state);
        state.beforeWrite = method => {
          if (method === "remove") throw new Error("connection reset");
        };
      });
      expect(String(outcome)).toMatch(failure);
      expect(String(outcome)).toContain(
        `temporary session ${ownedID} could not be deleted`,
      );
      expect(state.requests).toHaveLength(0);
      expect(state.sessions.has(ownedID!)).toBeTrue();
      expect(owned.has(ownedID!)).toBeTrue();
      expect(() => owned.admitRequest(ownedID!)).toThrow(owned.stop);
      expect(() => owned.rejectCompaction(ownedID!)).toThrow(
        owned.compactionBlocked,
      );
    },
  );

  const summarizeSession = async (arrange: (state: FakeV2State) => void) => {
    const state = fakeV2State();
    const source = sourceSession(state, "ses_src", target);
    arrange(state);
    const owned = new OwnedSessions();
    const outcome = await generateV2SessionSummary(
      fakeV2Host(state, owned, () => ({ text: "## Objective\n- Continue." })),
      owned,
      { session: source, messages: target, instructions: [] },
      {
        ref: state.summarizerModel!,
        info: {
          limit: state.modelLimit,
          capabilities: { tools: true, input: ["text"], output: ["text"] },
        },
      },
      10,
    ).catch((error: Error) => error);
    const ownedID = state.calls
      .find(entry => entry.startsWith("import:"))!
      .slice("import:".length);
    return { state, owned, outcome, ownedID };
  };

  test("the retention summary keeps stopping an owned session whose removal failed", async () => {
    const { state, owned, outcome, ownedID } = await summarizeSession(state => {
      state.beforeWrite = method => {
        if (method === "remove") throw new Error("connection reset");
      };
    });
    expect(String(outcome)).toContain("summary failed (delete)");
    expect(owned.has(ownedID)).toBeTrue();
    expect(() => owned.rejectCompaction(ownedID)).toThrow(
      owned.compactionBlocked,
    );
    // The one request it made was its own; no other can follow.
    expect(state.requests).toHaveLength(1);
    expect(() => owned.admitRequest(ownedID)).toThrow(owned.stop);
  });

  test("the retention summary closes a session whose import was applied but failed", async () => {
    const { state, owned, outcome, ownedID } = await summarizeSession(state => {
      state.afterWrite = method => {
        if (method === "import") throw new Error("response lost");
      };
      state.beforeWrite = method => {
        if (method === "remove") throw new Error("connection reset");
      };
    });
    expect(String(outcome)).toContain("summary and cleanup failed");
    expect(state.requests).toHaveLength(0);
    expect(state.sessions.has(ownedID)).toBeTrue();
    expect(() => owned.admitRequest(ownedID)).toThrow(owned.stop);
  });
});

describe("OpenCode 2 compaction", () => {
  const turn = (n: number): V2Message[] => [
    userMessage(`msg_u${n}`, `Request ${n}\nsecond line`),
    assistantMessage(`msg_a${n}`, [
      { type: "reasoning", text: "thinking" },
      { type: "text", text: words(`A${n}`, 300) },
      toolItem(`call_${n}`, "read", { path: "/f" }, [
        { type: "text", text: words(`R${n}`, 300) },
      ]),
    ]),
    idleMessage(`msg_i${n}`),
  ];

  test("plans after the newest completed native compaction and the latest Magic boundary", () => {
    const messages = [
      ...turn(1),
      {
        id: "msg_c",
        type: "compaction",
        status: "completed",
        reason: "auto",
        summary: "s",
        recent: "r",
        time: { created: 50 },
      } as V2Message,
      ...turn(2),
      {
        id: "msg_b",
        type: "synthetic",
        text: "boundary",
        metadata: { magicCompact: { boundary: true } },
        time: { created: 60 },
      } as V2Message,
      ...turn(3),
      ...turn(4),
      userMessage("msg_open", "pending"),
    ];
    const plan = planV2Compaction(messages, 1);
    expect(plan.turns.map(item => item.user[0]!.id)).toEqual([
      "msg_u2",
      "msg_u3",
      "msg_u4",
    ]);
    expect(plan.summarized.map(item => item.user[0]!.id)).toEqual(["msg_u3"]);
    expect(plan.next?.user[0]!.id).toBe("msg_u4");
    expect(plan.preserved).toBe(1);
    expect(plan.sourceTurnOffset).toBe(1);
  });

  test("summarizes each target into its first assistant message and leads the next turn with a boundary", async () => {
    const state = fakeV2State();
    const messages = [...turn(1), ...turn(2), ...turn(3)];
    sourceSession(state, "ses_src", messages);
    const owned = new OwnedSessions();
    const host = fakeV2Host(state, owned, echoGeneration());
    const { successor, outcome } = await executeV2Compact(
      host,
      owned,
      "ses_src",
      1,
      {
        retention: { mode: "heuristic" },
        controlTimeoutMs: 1_000,
      },
    );
    expect(outcome).toContain("Request turns summarized: 2");
    expect(state.requests).toHaveLength(2);
    expect(state.messages.get("ses_src")).toEqual(messages);
    const result = state.messages.get(successor!.session.id)!;
    expect(result.map(message => message.type)).toEqual([
      "user",
      "assistant",
      "idle",
      "user",
      "assistant",
      "idle",
      "synthetic",
      "user",
      "assistant",
      "idle",
    ]);
    const first = result[1] as Extract<V2Message, { type: "assistant" }>;
    expect(first.metadata).toEqual({
      magicCompact: { summary: true, capsule: true, agentic: true, version: 3 },
    });
    expect(first.content.map(item => item.type)).toEqual(["text", "tool"]);
    expect(first.content[0]).toEqual({
      type: "text",
      text: "Summary of target 1.",
    });
    expect(JSON.stringify(result[7])).toBe(
      JSON.stringify({ ...messages[6], id: result[7]!.id }),
    );
    expect(result[6]).toMatchObject({
      type: "synthetic",
      metadata: { magicCompact: { boundary: true } },
    });
    expect(successor!.session.metadata).toMatchObject({
      magicCompact: { compactionCount: 1, operation: "compact" },
    });

    const again = await executeV2Compact(
      host,
      owned,
      successor!.session.id,
      1,
      { retention: { mode: "heuristic" }, controlTimeoutMs: 1_000 },
    );
    expect(again.successor).toBeNull();
  });

  // Two projects with different summarizer profiles and a wrong server default:
  // each command must read the agent and the model list at the source
  // session's own directory. A project without a profile gets the default.
  const summarizerScopes = [
    { directory: "/workspace", model: "workspace-model" },
    { directory: "/other-project", model: "other-model" },
    { directory: "/plain-project", model: "default-model" },
  ];
  const scopedSummarizerState = (directory: string) => {
    const state = fakeV2State();
    state.summarizerModel = { providerID: "probe", id: "default-model" };
    state.projectSummarizers = {
      "/workspace": { providerID: "probe", id: "workspace-model" },
      "/other-project": { providerID: "probe", id: "other-model" },
    };
    return {
      state,
      session: (messages: V2Message[]) =>
        sourceSession(state, "ses_src", messages, {
          location: { directory },
        }),
    };
  };
  const spyOnSummarizerReads = (host: ReturnType<typeof fakeV2Host>) => {
    const reads = {
      agentLocations: [] as unknown[],
      agentModels: [] as string[],
      modelListLocations: [] as unknown[],
    };
    const agentGet = host.agent.get.bind(host.agent);
    host.agent.get = (async (...args: Parameters<typeof agentGet>) => {
      reads.agentLocations.push(args[0].location);
      const result = await agentGet(...args);
      reads.agentModels.push(result.data.model?.id ?? "none");
      return result;
    }) as typeof host.agent.get;
    const modelList = host.model.list.bind(host.model);
    host.model.list = (async (...args: Parameters<typeof modelList>) => {
      reads.modelListLocations.push(args[0]?.location);
      return modelList(...args);
    }) as typeof host.model.list;
    return reads;
  };

  for (const { directory, model } of summarizerScopes) {
    test(`compact resolves the summarizer at the source session location ${directory}`, async () => {
      const { state, session } = scopedSummarizerState(directory);
      session([...turn(1), ...turn(2)]);
      const owned = new OwnedSessions();
      const host = fakeV2Host(state, owned, echoGeneration());
      const reads = spyOnSummarizerReads(host);
      const { successor } = await executeV2Compact(host, owned, "ses_src", 1, {
        retention: { mode: "heuristic" },
        controlTimeoutMs: 1_000,
      });
      expect(successor).not.toBeNull();
      expect(reads.agentLocations).toEqual([{ directory }]);
      expect(reads.modelListLocations).toEqual([{ directory }]);
      expect(reads.agentModels).toEqual([model]);
    });
  }

  test("never falls back when the summarizer model is not configured", async () => {
    const state = fakeV2State();
    sourceSession(state, "ses_src", [...turn(1), ...turn(2)]);
    delete state.summarizerModel;
    const owned = new OwnedSessions();
    await expect(
      executeV2Compact(
        fakeV2Host(state, owned, echoGeneration()),
        owned,
        "ses_src",
        0,
        { retention: { mode: "heuristic" }, controlTimeoutMs: 1_000 },
      ),
    ).rejects.toThrow(/needs a model on the `magic-compact-summarizer` agent/);
    expect(state.requests).toHaveLength(0);
  });
});

// A user image the host sends inline: base64 data the tokenizer counts as far
// more than its placeholder.
const largeImage = randomBytes(3_000).toString("base64");
// A payload whose placeholder would not be smaller: it stays inline.
const smallImage = "iVBORw0K";
const imageFile = (data: string, extra: object = {}) => ({
  data,
  mime: "image/png",
  source: { type: "inline" },
  ...extra,
});

// The request messages OpenCode 2.0.28 builds from user and assistant
// records (`session/runner/to-llm-message.ts`): a user message keeps its ID and
// metadata and sends each listed file as base64 image media.
function requestMessages(
  records: V2Message[],
  sent: (message: Extract<V2Message, { type: "user" }>) => number[] = message =>
    (message.files ?? []).map((_, index) => index),
): Array<{
  id?: string;
  role: string;
  content: unknown[];
  metadata?: unknown;
}> {
  return records.flatMap(message => {
    if (message.type === "user") {
      return [
        {
          id: message.id,
          role: "user",
          metadata: { ...message.metadata },
          content: [
            { type: "text", text: message.text },
            ...(message.files ? sent(message) : []).map(index => {
              const file = message.files![index]!;
              return {
                type: "media",
                media: {
                  mediaType: file.mime,
                  source: {
                    type: "base64",
                    data: file.data,
                    mediaType: file.mime,
                  },
                },
              };
            }),
          ],
        },
      ];
    }
    if (message.type === "assistant") {
      return [
        {
          id: message.id,
          role: "assistant",
          metadata: message.metadata,
          content: message.content.flatMap(item =>
            item.type === "text" ? [{ type: "text", text: item.text }] : [],
          ),
        },
      ];
    }
    return [];
  });
}

const textOf = (message: { content: unknown[] }) =>
  message.content.map(part => (part as { type: string }).type);

describe("OpenCode 2 media projection", () => {
  const imageTurn = (n: number, files: unknown[]): V2Message[] => [
    userMessage(`msg_u${n}`, `Look ${n}`, { files }),
    assistantMessage(`msg_a${n}`, [
      { type: "text", text: words(`A${n}`, 300) },
    ]),
    idleMessage(`msg_i${n}`),
  ];
  const plainTurn = (n: number): V2Message[] => [
    userMessage(`msg_u${n}`, `Request ${n}`),
    assistantMessage(`msg_a${n}`, [
      { type: "text", text: words(`A${n}`, 300) },
    ]),
    idleMessage(`msg_i${n}`),
  ];
  const compact = async (messages: V2Message[], keepTurns = 1) => {
    const state = fakeV2State();
    sourceSession(state, "ses_src", messages);
    const owned = new OwnedSessions();
    const result = await executeV2Compact(
      fakeV2Host(state, owned, echoGeneration()),
      owned,
      "ses_src",
      keepTurns,
      { retention: { mode: "heuristic" }, controlTimeoutMs: 1_000 },
    );
    return { state, ...result };
  };

  test("a compacted image stays stored exactly and leaves requests as a placeholder", async () => {
    const source = [
      ...imageTurn(1, [
        imageFile(largeImage, { name: "a.png" }),
        imageFile(smallImage),
      ]),
      ...plainTurn(2),
    ];
    const { state, successor, outcome } = await compact(source);
    const stored = state.messages.get(successor!.session.id)!;
    const user = stored[0] as Extract<V2Message, { type: "user" }>;
    // The successor keeps the original user record byte for byte.
    expect(JSON.stringify({ ...user, id: "x" })).toBe(
      JSON.stringify({ ...source[0], id: "x" }),
    );
    const carrier = stored[1] as Extract<V2Message, { type: "assistant" }>;
    expect(carrier.metadata).toEqual({
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
              dataSha256: createHash("sha256").update(largeImage).digest("hex"),
            },
            {
              ordinal: 2,
              mime: "image/png",
              dataSha256: createHash("sha256").update(smallImage).digest("hex"),
            },
          ],
        },
      },
    });
    expect(state.requests[0]!.text).toContain('"expectedInlineImages":2');

    const facing = countV2ModelFacingTokens(stored);
    expect(facing.projectedMediaTokens).toBeGreaterThan(0);
    const stats = await readStats(v2StorageKey(successor!.session.id));
    expect(stats?.mediaTokensProjected).toBe(facing.projectedMediaTokens);
    expect(outcome).toContain("Request turns summarized: 1");

    const request = requestMessages(stored);
    const before = JSON.stringify(request);
    const projected = projectV2RequestMessages(request, stored);
    expect(JSON.stringify(request)).toBe(before);
    expect(projected[0]!.content).toEqual([
      { type: "text", text: "Look 1" },
      {
        type: "text",
        text: "[Historical inline image 1 (image/png) was summarized by Magic Compact; its payload is projected out of this model request.]",
      },
      // Its placeholder would not be smaller, so the small image stays.
      request[0]!.content[2],
    ]);
    expect(projected.slice(1)).toEqual(request.slice(1));
  });

  test("a payload too short to gain from projection stays inline with no projected tokens", async () => {
    const { state, successor } = await compact([
      ...imageTurn(1, [imageFile(smallImage)]),
      ...plainTurn(2),
    ]);
    const stored = state.messages.get(successor!.session.id)!;
    expect(countV2ModelFacingTokens(stored).projectedMediaTokens).toBe(0);
    const request = requestMessages(stored);
    expect(projectV2RequestMessages(request, stored)).toEqual(request);
    const stats = await readStats(v2StorageKey(successor!.session.id));
    expect(stats?.mediaTokensProjected).toBe(0);
  });

  test("a changed, added, missing or reordered image fails before any request is projected", async () => {
    const second = randomBytes(3_000).toString("base64");
    const { state, successor } = await compact([
      ...imageTurn(1, [imageFile(largeImage), imageFile(second)]),
      ...plainTurn(2),
    ]);
    const stored = state.messages.get(successor!.session.id)!;
    expect(planV2MediaProjection(stored).size).toBe(1);
    const user = stored[0] as Extract<V2Message, { type: "user" }>;
    const withFiles = (files: unknown[]) => [
      { ...user, files } as V2Message,
      ...stored.slice(1),
    ];
    const changedRecords: Array<[string, V2Message[]]> = [
      ["changed data", withFiles([imageFile(largeImage), imageFile("AAAA")])],
      [
        "changed MIME",
        withFiles([
          imageFile(largeImage),
          { ...imageFile(second), mime: "image/webp" },
        ]),
      ],
      [
        "added image",
        withFiles([
          imageFile(largeImage),
          imageFile(second),
          imageFile("BBBB"),
        ]),
      ],
      ["missing image", withFiles([imageFile(largeImage)])],
      ["reordered", withFiles([imageFile(second), imageFile(largeImage)])],
    ];
    for (const [, records] of changedRecords) {
      expect(() => planV2MediaProjection(records)).toThrow(
        "media manifest mismatch",
      );
      const request = requestMessages(records);
      const before = JSON.stringify(request);
      expect(() => projectV2RequestMessages(request, records)).toThrow(
        "media manifest mismatch",
      );
      expect(JSON.stringify(request)).toBe(before);
    }
    // The records match, but the request the host built does not.
    const request = requestMessages(stored, () => [1, 0]);
    expect(() => projectV2RequestMessages(request, stored)).toThrow(
      "media manifest mismatch",
    );
    const missing = requestMessages(stored, () => [0]);
    expect(() => projectV2RequestMessages(missing, stored)).toThrow(
      "media manifest mismatch",
    );
    const malformed = structuredClone(stored);
    (
      malformed[1] as { metadata: { magicCompact: Record<string, unknown> } }
    ).metadata.magicCompact["mediaManifest"] = { version: 1, images: [] };
    expect(() => planV2MediaProjection(malformed)).toThrow(
      "media manifest mismatch",
    );
  });

  test("binds only the images the host sends: supported MIME types, and a mentioned repeat once", async () => {
    const mention = { start: 0, end: 4, text: "@img" };
    const repeat = imageFile(largeImage, { name: "a.png", mention });
    const files = [
      repeat,
      { ...imageFile("PHN2Zz4="), mime: "image/svg+xml" },
      repeat,
      imageFile(largeImage, { name: "a.png" }),
    ];
    const user = userMessage("msg_u1", "Look", { files });
    expect(v2VisibleImageIndexes(user)).toEqual([0, 3]);
    // Without a second mentioned image the host sends every repeat.
    expect(
      v2VisibleImageIndexes(
        userMessage("msg_u1", "Look", {
          files: [repeat, imageFile(largeImage, { name: "a.png" })],
        }),
      ),
    ).toEqual([0, 1]);

    const { state, successor } = await compact([
      ...imageTurn(1, files),
      ...plainTurn(2),
    ]);
    expect(state.requests[0]!.text).toContain('"expectedInlineImages":2');
    const stored = state.messages.get(successor!.session.id)!;
    const request = requestMessages(stored, () => [0, 3]);
    const projected = projectV2RequestMessages(request, stored);
    expect(textOf(projected[0]!)).toEqual(["text", "text", "text"]);
  });

  // Two summarized image turns separated by a provider checkpoint. OpenCode
  // 2.0.28 (`session/history.ts`) replays the checkpoint only for a model on
  // the provider route that wrote it; any other model replays the older
  // records instead. The host's in-process context always starts at the
  // checkpoint.
  const checkpointHistory = () => {
    const older = randomBytes(3_000).toString("base64");
    const newer = randomBytes(3_000).toString("base64");
    const carrier = (id: string, data: string) =>
      assistantMessage(id, [{ type: "text", text: "summary" }], {
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
                  dataSha256: createHash("sha256").update(data).digest("hex"),
                },
              ],
            },
          },
        },
      });
    const before = [
      userMessage("msg_u1", "Old look", { files: [imageFile(older)] }),
      carrier("msg_a1", older),
      idleMessage("msg_i1"),
    ];
    const checkpoint = {
      id: "msg_c",
      type: "compaction",
      status: "completed",
      reason: "auto",
      summary: "",
      recent: "",
      providerContext: {
        version: 1,
        provenance: { modelID: "a" },
        messages: [],
      },
      time: { created: 50 },
    } as unknown as V2Message;
    const after = [
      userMessage("msg_u2", "New look", { files: [imageFile(newer)] }),
      carrier("msg_a2", newer),
      idleMessage("msg_i2"),
    ];
    return {
      older,
      newer,
      records: [...before, checkpoint, ...after],
      context: [checkpoint, ...after],
      // A model on another route: every record but the checkpoint.
      incompatible: requestMessages([...before, ...after]),
      // The checkpoint's route: its retained copy of the older user message,
      // under the same ID, then the provider's opaque checkpoint.
      compatible: [
        requestMessages(before)[0]!,
        { role: "assistant", content: [{ type: "text", text: "opaque" }] },
        ...requestMessages(after),
      ],
    };
  };
  const placeholderFor = /Historical inline image 1/;

  test("counts every stored capsule, behind a checkpoint too, and checks each on its own", () => {
    const { records, context, newer } = checkpointHistory();
    expect([...planV2MediaProjection(records).keys()]).toEqual([
      "msg_u1",
      "msg_u2",
    ]);
    const projected = (messages: V2Message[]) =>
      countV2ModelFacingTokens(messages).projectedMediaTokens;
    expect(projected(context)).toBeGreaterThan(0);
    expect(projected(records)).toBe(
      projected(records.slice(0, 3)) + projected(context),
    );
    // A capsule whose image changed fails the stored count, wherever it is,
    // but not a request that does not represent its turn.
    const tampered = structuredClone(records);
    (tampered[0] as { files: unknown[] }).files = [imageFile("AAAA")];
    expect(() => planV2MediaProjection(tampered)).toThrow(
      "media manifest mismatch",
    );
    const request = requestMessages(context);
    const sent = JSON.stringify(projectV2RequestMessages(request, tampered));
    expect(sent).not.toContain(newer);
    expect(sent).toMatch(placeholderFor);
  });

  test("a request replaying history behind a checkpoint projects the older capsules from the complete history", async () => {
    const history = checkpointHistory();
    const reads: string[] = [];
    let available = true;
    const session = {
      id: "ses_src",
      projectID: "p",
      time: { created: 1 },
      location: LOCATION,
      metadata: { magicCompact: { successorOf: "ses_prev", ready: true } } as
        Record<string, unknown> | undefined,
    };
    const server = Bun.serve({
      port: 0,
      hostname: "127.0.0.1",
      fetch: request => {
        const url = new URL(request.url);
        reads.push(url.pathname);
        if (!available) return new Response("down", { status: 500 });
        if (url.pathname === "/api/info") {
          return Response.json({
            version: "2.0.28",
            pid: process.pid,
            urls: [],
            paths: { tmp: "/tmp" },
          });
        }
        if (url.pathname.endsWith("/message")) {
          return Response.json({
            data: history.records,
            cursor: { next: null },
          });
        }
        if (url.pathname.endsWith("/export")) {
          return Response.json({ data: { messages: history.records } });
        }
        return Response.json({ data: session });
      },
    });
    try {
      const hooks = new Map<string, (event: never) => Promise<void>>();
      const editor = { add: () => undefined, update: () => undefined };
      const owned = new OwnedSessions();
      await setup(
        {
          options: { serverUrl: `http://127.0.0.1:${server.port}` },
          agent: { transform: async (fn: (e: unknown) => void) => fn(editor) },
          command: {
            transform: async (fn: (e: unknown) => void) => fn(editor),
          },
          tool: { transform: async (fn: (e: unknown) => void) => fn(editor) },
          session: {
            hook: async (
              name: string,
              callback: (event: never) => Promise<void>,
            ) => {
              hooks.set(name, callback);
            },
            context: async () => structuredClone(history.context),
            get: async () => session,
          },
          event: { subscribe: async function* () {} },
        } as never,
        owned,
      );
      const run = async (messages: unknown[]) => {
        const event = { sessionID: "ses_src", tools: {}, messages };
        await hooks.get("context")!(event as never);
        return JSON.stringify(event.messages);
      };

      // Another model's request replays msg_u1, which the context lacks.
      const incompatible = await run(history.incompatible);
      expect(incompatible).not.toContain(history.older);
      expect(incompatible).not.toContain(history.newer);
      expect(incompatible.match(/Historical inline image 1/gu)).toHaveLength(2);
      expect(reads.filter(path => path.endsWith("/message"))).toHaveLength(1);

      // The checkpoint's own route replays its retained copy of msg_u1; a
      // copy repeated under the same ID is projected each time.
      const compatible = await run([
        history.compatible[0],
        ...history.compatible,
      ]);
      expect(compatible).not.toContain(history.older);
      expect(compatible.match(/Historical inline image 1/gu)).toHaveLength(3);
      // The verified connection is reused.
      expect(reads.filter(path => path === "/api/info")).toHaveLength(1);

      // On that route a request may hold no capsule at all: the checkpoint
      // replays the retained copy, and later turns carry no manifest. In a
      // session with Magic Compact lineage the copy is still projected.
      const copyOnly = [
        ...history.compatible.slice(0, 2),
        ...requestMessages(plainTurn(3)),
      ];
      const lineage = await run(copyOnly);
      expect(lineage).not.toContain(history.older);
      expect(lineage).toMatch(placeholderFor);
      // Elsewhere such a request is left alone and reads nothing.
      session.metadata = undefined;
      const reading = reads.length;
      expect(await run(copyOnly)).toBe(JSON.stringify(copyOnly));
      expect(reads).toHaveLength(reading);

      // An owned copy of a source clears its metadata and holds no capsule in
      // the request either, yet replays the same retained copy; only the
      // in-memory registry qualifies it. The source and checkpoint are not
      // rewritten, and a copy with another image fails before any provider
      // request.
      const stored = JSON.stringify(history.records);
      owned.register("ses_owned_a", "retention");
      owned.register("ses_owned_b", "retention");
      const projected = {
        sessionID: "ses_owned_a",
        tools: {},
        messages: copyOnly,
      };
      await hooks.get("context")!(projected as never);
      expect(JSON.stringify(projected.messages)).not.toContain(history.older);
      expect(JSON.stringify(projected.messages)).toMatch(placeholderFor);
      const altered = structuredClone(copyOnly);
      (
        altered[0] as {
          content: Array<{ media?: { source: { data: string } } }>;
        }
      ).content[1]!.media!.source.data = "AAAA";
      const alteredBefore = JSON.stringify(altered);
      const refusedOwned = {
        sessionID: "ses_owned_b",
        tools: {},
        messages: altered,
      };
      await expect(
        hooks.get("context")!(refusedOwned as never),
      ).rejects.toThrow("media manifest mismatch");
      expect(JSON.stringify(refusedOwned.messages)).toBe(alteredBefore);
      expect(JSON.stringify(history.records)).toBe(stored);
      session.metadata = { magicCompact: {} };

      // A copy whose image is not the summarized one fails the request.
      const changed = structuredClone(history.compatible);
      (
        changed[0] as {
          content: Array<{ media?: { source: { data: string } } }>;
        }
      ).content[1]!.media!.source.data = "AAAA";
      const before = JSON.stringify(changed);
      const event = { sessionID: "ses_src", tools: {}, messages: changed };
      await expect(hooks.get("context")!(event as never)).rejects.toThrow(
        "media manifest mismatch",
      );
      expect(JSON.stringify(event.messages)).toBe(before);

      // Without the server's HTTP API the request is refused, not sent with
      // the older images inline.
      available = false;
      const refused = {
        sessionID: "ses_src",
        tools: {},
        messages: history.incompatible,
      };
      await expect(hooks.get("generate")!(refused as never)).rejects.toThrow(
        "could not read this session's complete history",
      );
      expect(refused.messages).toBe(history.incompatible);
      // A request the in-process context covers needs no HTTP read.
      const count = reads.length;
      expect(await run(requestMessages(history.context))).not.toContain(
        history.newer,
      );
      expect(reads).toHaveLength(count);
    } finally {
      await server.stop(true);
    }
  });

  // A provider checkpoint written after a capsule retains a copy of the
  // summarized user message under that message's ID at the time
  // (`session/compaction.ts` `recentUserMessages`). The next successor and a
  // native fork (`session/projector.ts` `projectFork`) both copy the
  // checkpoint unchanged under fresh record IDs, so on the checkpoint's route
  // the copy arrives under an ID no record holds any more.
  const checkpointRetaining = (users: V2Message[]) =>
    ({
      id: "msg_c",
      type: "compaction",
      status: "completed",
      reason: "auto",
      summary: "",
      recent: "",
      providerContext: {
        version: 1,
        provenance: { modelID: "a" },
        messages: [
          ...requestMessages(users),
          { role: "assistant", content: [{ type: "text", text: "opaque" }] },
        ],
      },
      time: { created: 50 },
    }) as unknown as V2Message;
  test("a retained checkpoint copy is projected after a successor or a native fork reminted its record ID", async () => {
    const first = await compact([
      ...imageTurn(1, [imageFile(largeImage)]),
      ...plainTurn(2),
    ]);
    const capsuled = first.state.messages.get(first.successor!.session.id)!;
    const retained = requestMessages(capsuled.slice(0, 1));
    const checkpoint = checkpointRetaining(capsuled.slice(0, 1));
    const second = await compact([
      ...capsuled,
      checkpoint,
      ...plainTurn(3),
      ...plainTurn(4),
    ]);
    const stored = second.state.messages.get(second.successor!.session.id)!;
    const index = stored.findIndex(record => record.type === "compaction");
    const copiedCheckpoint = stored[index] as unknown as {
      providerContext: {
        messages: Array<{ id?: string; role: string; content: unknown[] }>;
      };
    };
    // The successor keeps the checkpoint byte for byte, so the copy's ID now
    // names no record.
    expect(copiedCheckpoint.providerContext).toEqual(
      (checkpoint as unknown as typeof copiedCheckpoint).providerContext,
    );
    expect(stored.some(record => record.id === retained[0]!.id)).toBe(false);
    // The checkpoint's route: its retained window, then the later records.
    const route = (records: V2Message[]) => {
      const at = records.findIndex(record => record.type === "compaction");
      const { providerContext } = records[
        at
      ] as unknown as typeof copiedCheckpoint;
      return [
        ...structuredClone(providerContext.messages),
        ...requestMessages(records.slice(at + 1)),
      ];
    };
    const sent = (records: V2Message[]) =>
      JSON.stringify(projectV2RequestMessages(route(records), records));
    expect(sent(stored)).not.toContain(largeImage);
    expect(sent(stored)).toMatch(placeholderFor);
    // A native fork copies every record's data under new IDs once more.
    const forked = stored.map(
      (record, seq) =>
        ({ ...structuredClone(record), id: `msg_evt_${seq}` }) as V2Message,
    );
    expect(sent(forked)).not.toContain(largeImage);
    expect(sent(forked)).toMatch(placeholderFor);

    // A copy whose image is not the summarized one binds to no record.
    const changed = route(stored);
    (
      changed[0]!.content[1] as { media: { source: { data: string } } }
    ).media.source.data = "AAAA";
    expect(() => projectV2RequestMessages(changed, stored)).toThrow(
      "media manifest mismatch",
    );
    // So does an image copy no checkpoint retains.
    const stray = route(stored);
    stray[0] = { ...stray[0]!, id: "msg_unknown" };
    expect(() => projectV2RequestMessages(stray, stored)).toThrow(
      "media manifest mismatch",
    );
    // An identical user message in an unsummarized turn before the
    // checkpoint makes the copy ambiguous: one would be projected, the other
    // sent inline.
    const duplicate = [
      ...stored.slice(0, index),
      ...imageTurn(9, [imageFile(largeImage)]).map(record =>
        record.type === "user" ? { ...record, text: "Look 1" } : record,
      ),
      ...stored.slice(index),
    ] as V2Message[];
    expect(() => projectV2RequestMessages(route(duplicate), duplicate)).toThrow(
      "media manifest mismatch",
    );
    // A copy of an unsummarized message stays inline.
    const plain = await compact([
      ...imageTurn(1, [imageFile(largeImage)]),
      checkpointRetaining(imageTurn(1, [imageFile(largeImage)]).slice(0, 1)),
      ...plainTurn(2),
      ...plainTurn(3),
    ]);
    const plainStored = plain.state.messages.get(plain.successor!.session.id)!;
    expect(sent(plainStored)).toContain(largeImage);
  });

  // User messages with the same text before one provider checkpoint, after a
  // native fork reminted every record ID: only the checkpoint's order of
  // copies ties a copy to its source, never the image it carries.
  test("a reminted checkpoint copy binds to its source by the order of every copy, not by its image", async () => {
    const other = randomBytes(3_000).toString("base64");
    const look = (n: number, data: string) =>
      imageTurn(n, [imageFile(data)]).map(record =>
        record.type === "user" ? { ...record, text: "Look" } : record,
      ) as V2Message[];
    const fork = async (
      turns: V2Message[],
      keepTurns: number,
      retain: (users: V2Message[]) => V2Message[],
      tamper: (copies: Array<{ content: unknown[] }>) => void = () => {},
    ) => {
      const { state, successor } = await compact(
        [...turns, ...plainTurn(9)],
        keepTurns,
      );
      const stored = state.messages.get(successor!.session.id)!;
      const checkpoint = checkpointRetaining(
        retain(stored.filter(record => record.type === "user")),
      );
      tamper(
        (
          checkpoint as unknown as {
            providerContext: { messages: Array<{ content: unknown[] }> };
          }
        ).providerContext.messages,
      );
      const records = [...stored, checkpoint, ...plainTurn(10)].map(
        (record, seq) =>
          ({ ...structuredClone(record), id: `msg_evt_${seq}` }) as V2Message,
      );
      const at = records.findIndex(record => record.type === "compaction");
      const route = [
        ...structuredClone(
          (
            records[at] as unknown as {
              providerContext: {
                messages: Array<{
                  id?: string;
                  role: string;
                  content: unknown[];
                }>;
              };
            }
          ).providerContext.messages,
        ),
        ...requestMessages(records.slice(at + 1)),
      ];
      return { stored, records, route };
    };
    const send = ({ records, route }: Awaited<ReturnType<typeof fork>>) =>
      JSON.stringify(projectV2RequestMessages(route, records));
    const refuses = ({ records, route }: Awaited<ReturnType<typeof fork>>) => {
      const before = JSON.stringify(route);
      expect(() => projectV2RequestMessages(route, records)).toThrow(
        "media manifest mismatch",
      );
      expect(JSON.stringify(route)).toBe(before);
    };
    const setMedia = (copy: { content: unknown[] }, data: string) => {
      (
        copy.content[1] as { media: { source: { data: string } } }
      ).media.source.data = data;
    };
    const users =
      (...indexes: number[]) =>
      (all: V2Message[]) =>
        indexes.map(index => all[index]!);

    // The older Look is summarized; the newer one stays inline.
    const mixed = [...look(1, largeImage), ...look(2, other)];
    const valid = await fork(mixed, 2, users(0, 1, 2));
    expect([...planV2MediaProjection(valid.stored).keys()]).toEqual([
      valid.stored[0]!.id,
    ]);
    const sent = send(valid);
    expect(sent).not.toContain(largeImage);
    expect(sent).toContain(other);
    expect(sent.match(/Historical inline image 1/gu)).toHaveLength(1);

    // The summarized copy changed to the inline one's image: it is still the
    // older message's copy, and its image is not the summarized one.
    refuses(
      await fork(mixed, 2, users(0, 1, 2), ([first]) => {
        setMedia(first!, other);
      }),
    );
    // So are the two copies' images swapped.
    refuses(
      await fork(mixed, 2, users(0, 1, 2), ([first, second]) => {
        setMedia(first!, other);
        setMedia(second!, largeImage);
      }),
    );
    // One copy alone may stand for either Look, and they are projected
    // differently, whatever image it carries.
    refuses(await fork(mixed, 2, users(1, 2)));
    refuses(await fork(mixed, 2, users(0, 2)));

    // Copies that may stand for either of two messages projected the same
    // way are projected.
    const covered = [...look(1, largeImage), ...look(2, largeImage)];
    for (const retained of [users(1, 2), users(0, 1, 2)]) {
      const both = send(await fork(covered, 1, retained));
      expect(both).not.toContain(largeImage);
      expect(both).toMatch(placeholderFor);
    }

    // A copy with no image orders the copies around it too: the Look copy
    // before the copy of "Request 2" can only be the older Look.
    const between = [
      ...look(1, largeImage),
      ...plainTurn(2),
      ...look(3, other),
    ];
    const ordered = send(await fork(between, 2, users(0, 1)));
    expect(ordered).not.toContain(largeImage);
    expect(ordered).toMatch(placeholderFor);
    refuses(await fork(between, 2, users(0)));

    // A copy is its record as OpenCode sends it: the attached file's location
    // and the agents in its metadata, with or without the skills' text.
    const attached = imageTurn(1, [
      imageFile(largeImage, {
        source: { type: "uri", uri: "file:///work/look.png" },
      }),
    ]).map(record =>
      record.type === "user"
        ? {
            ...record,
            skills: [{ id: "review", name: "review", text: "Skill body" }],
            agents: [{ name: "plan" }],
          }
        : record,
    ) as V2Message[];
    const asSent =
      (skills: boolean) => (copies: Array<{ content: unknown[] }>) => {
        const copy = copies[0] as { content: unknown[]; metadata: unknown };
        copy.content.splice(1, 0, {
          type: "text",
          text: "Attached file: /work/look.png",
        });
        if (skills) copy.content.unshift({ type: "text", text: "Skill body" });
        copy.metadata = { agents: [{ name: "plan" }] };
      };
    for (const skills of [false, true]) {
      const projected = send(
        await fork(attached, 1, users(0, 1), asSent(skills)),
      );
      expect(projected).not.toContain(largeImage);
      expect(projected).toMatch(placeholderFor);
    }
    // A copy missing the location text stands for no record.
    refuses(
      await fork(attached, 1, users(0, 1), copies => {
        asSent(false)(copies);
        copies[0]!.content.splice(1, 1);
      }),
    );

    // OpenCode puts a text in the copy in place of media it will not send: a
    // PDF the model cannot read, or an image past the request's image budget
    // (`session/model-request.ts`, which `recentUserMessages` applies). That
    // copy still stands for its record and orders the copies around it, as
    // in a checkpoint OpenCode 2.0.28 wrote in the container qualification.
    const pdf = {
      data: Buffer.from("%PDF-1.4\n").toString("base64"),
      mime: "application/pdf",
      name: "doc.pdf",
      source: { type: "uri", uri: "file:///work/doc.pdf" },
    };
    const replacements: Array<[unknown[], string[]]> = [
      [
        [pdf],
        [
          "Attached file: /work/doc.pdf",
          'ERROR: Cannot read "doc.pdf" (this model does not support pdf input). Inform the user.',
        ],
      ],
      [
        [imageFile(other)],
        [
          "[This image was removed to reduce the request size and is no longer visible. Do not make claims about its contents from memory. If needed, retrieve it again with an available tool or ask the user to attach it again.]",
        ],
      ],
    ];
    for (const [files, texts] of replacements) {
      const stripped = [
        ...look(1, largeImage),
        ...imageTurn(2, files).map(record =>
          record.type === "user" ? { ...record, text: "Notes" } : record,
        ),
      ] as V2Message[];
      const asStored = (copies: Array<{ content: unknown[] }>) => {
        copies[1]!.content = ["Notes", ...texts].map(text => ({
          type: "text",
          text,
        }));
      };
      const projected = send(await fork(stripped, 2, users(0, 1), asStored));
      expect(projected).not.toContain(largeImage);
      expect(projected).toMatch(placeholderFor);
      refuses(
        await fork(stripped, 2, users(0, 1), copies => {
          asStored(copies);
          setMedia(copies[0]!, other);
        }),
      );
    }
  });

  // OpenCode 2.0.28 replaces an image the model cannot take, and drops the
  // oldest images past 25 MiB of image data in one request, after the
  // plugin's hooks run (`session/model-request.ts`). A summary must have seen
  // every image it binds.
  const summarize = async (
    span: V2Message[],
    info: { providerID?: string; input?: string[] } = {},
  ) => {
    const state = fakeV2State();
    const source = sourceSession(state, "ses_src", span);
    const owned = new OwnedSessions();
    const outcome = await generateV2Summaries(
      fakeV2Host(state, owned, echoGeneration()),
      owned,
      { session: source, messages: span, instructions: [] },
      {
        ref: state.summarizerModel!,
        info: {
          providerID: info.providerID ?? state.summarizerModel!.providerID,
          limit: state.modelLimit,
          capabilities: {
            tools: true,
            input: info.input ?? ["text", "image"],
            output: ["text"],
          },
        },
      },
      [{ span }],
      "index",
      { controlTimeoutMs: 1_000 },
    ).catch((error: Error) => error);
    return { state, outcome };
  };
  const limit = 25 * 1024 * 1024;

  test("an image the summarizer would not receive whole fails before any session or request", async () => {
    const cases: Array<[string, V2Message[], object, RegExp]> = [
      [
        "over the host's request image limit",
        imageTurn(1, [
          imageFile("A".repeat(limit - 4_096)),
          imageFile("B".repeat(8_192)),
        ]),
        {},
        /exceed the 25 MiB OpenCode sends/,
      ],
      [
        "a MIME type xAI does not accept",
        imageTurn(1, [{ ...imageFile(largeImage), mime: "image/gif" }]),
        { providerID: "xai" },
        /requires an image-capable/,
      ],
      [
        "a model without image input",
        imageTurn(1, [imageFile(largeImage)]),
        { input: ["text", "pdf"] },
        /requires an image-capable/,
      ],
      [
        "data that is not canonical base64",
        imageTurn(1, [imageFile("iVBORw0K=")]),
        {},
        /malformed inline image data/,
      ],
    ];
    for (const [name, span, info, expected] of cases) {
      const { state, outcome } = await summarize(span, info);
      expect([name, String(outcome)]).toEqual([
        name,
        expect.stringMatching(expected),
      ]);
      expect(state.requests).toHaveLength(0);
      expect([...state.sessions.keys()]).toEqual(["ses_src"]);
    }
    // At the limit, and with a GIF for any other provider, the request is sent.
    for (const [span, info] of [
      [
        imageTurn(1, [
          imageFile("A".repeat(limit - 8_192)),
          imageFile("B".repeat(8_192)),
        ]),
        {},
      ],
      [imageTurn(1, [{ ...imageFile(largeImage), mime: "image/gif" }]), {}],
    ] as const) {
      const { state, outcome } = await summarize([...span], info);
      expect(outcome).toEqual([
        expect.objectContaining({
          text: expect.stringContaining("Summary of target 1."),
        }),
      ]);
      expect(state.requests).toHaveLength(1);
    }
  });

  test("the request hooks project every request kind, owned sessions included", async () => {
    const { state, successor } = await compact([
      ...imageTurn(1, [imageFile(largeImage)]),
      ...plainTurn(2),
    ]);
    const stored = state.messages.get(successor!.session.id)!;
    const hooks = new Map<string, (event: never) => Promise<void> | void>();
    const contextReads: string[] = [];
    const editor = { add: () => undefined, update: () => undefined };
    await plugin.setup({
      options: {},
      agent: { transform: async (fn: (e: unknown) => void) => fn(editor) },
      command: { transform: async (fn: (e: unknown) => void) => fn(editor) },
      tool: { transform: async (fn: (e: unknown) => void) => fn(editor) },
      session: {
        hook: async (name: string, callback: (event: never) => void) => {
          hooks.set(name, callback);
        },
        context: async ({ sessionID }: { sessionID: string }) => {
          contextReads.push(sessionID);
          return structuredClone(stored);
        },
      },
      event: { subscribe: async function* () {} },
    } as never);
    expect([...hooks.keys()].sort()).toEqual([
      "compaction",
      "context",
      "generate",
      "title",
    ]);
    const placeholder = /Historical inline image 1/;
    for (const name of ["context", "compaction", "generate", "title"]) {
      const event = {
        sessionID: "ses_any",
        tools: {},
        messages: requestMessages(stored),
      };
      await hooks.get(name)!(event as never);
      expect(JSON.stringify(event.messages[0])).toMatch(placeholder);
      expect(JSON.stringify(event.messages[0])).not.toContain(largeImage);
    }
    expect(contextReads).toEqual(Array(4).fill("ses_any"));
    // A request without a manifest never reads the session.
    const plain = {
      sessionID: "ses_other",
      tools: {},
      messages: requestMessages(plainTurn(9)),
    };
    await hooks.get("context")!(plain as never);
    expect(contextReads).toHaveLength(4);
  });
});

describe("OpenCode 2 retention keep threshold environment", () => {
  type Execute = (input: {
    sessionID: string;
    prompt: { text: string };
  }) => Promise<void>;
  const commands = async () => {
    const found = new Map<string, Execute>();
    const editor = {
      add: (command: { name: string; execute: Execute }) => {
        found.set(command.name, command.execute);
      },
      update: () => undefined,
    };
    await setup({
      // A non-string server URL fails the connection at once, without
      // looking for a registered service.
      options: { serverUrl: 0 },
      agent: { transform: async (fn: (e: unknown) => void) => fn(editor) },
      command: { transform: async (fn: (e: unknown) => void) => fn(editor) },
      tool: { transform: async (fn: (e: unknown) => void) => fn(editor) },
      session: { hook: async () => undefined, get: async () => ({}) },
      event: { subscribe: async function* () {} },
    } as never);
    return (name: string, prompt: string | { text: string }) => {
      const execute = found.get(name)!;
      return execute({
        sessionID: "ses_src",
        prompt: typeof prompt === "string" ? { text: prompt } : prompt,
      }).then(
        () => {
          throw new Error("Expected a rejection.");
        },
        (error: unknown) => error,
      );
    };
  };
  const withPercent = async <T>(value: string, run: () => Promise<T>) => {
    const previous = process.env[RETENTION_KEEP_THRESHOLD_PERCENT_ENV];
    process.env[RETENTION_KEEP_THRESHOLD_PERCENT_ENV] = value;
    try {
      return await run();
    } finally {
      if (previous === undefined) {
        delete process.env[RETENTION_KEEP_THRESHOLD_PERCENT_ENV];
      } else process.env[RETENTION_KEEP_THRESHOLD_PERCENT_ENV] = previous;
    }
  };

  test.each(["magic-compact", "magic-trim"])(
    "%s rejects an invalid percent before it connects to the server",
    async name => {
      const run = await commands();
      const error = await withPercent("60%", () => run(name, ""));
      expect(error).toBeInstanceOf(RetentionClassificationError);
      expect(String(error)).toContain(RETENTION_KEEP_THRESHOLD_PERCENT_ENV);
    },
  );

  test.each([
    ["magic-compact", "Magic Compact", "host", " (HTTP 200)"],
    ["magic-trim", "Magic Trim", "host", " (HTTP 200)"],
    ["magic-compact", "Magic Compact", "dependency", ""],
    ["magic-trim", "Magic Trim", "dependency", ""],
  ] as const)(
    "%s returns a %s %s failure it did not author without that failure's text",
    async (name, command, kind, classified) => {
      const run = await commands();
      const foreign = kind === "host" ? privateHostError() : dependencyError();
      const error = await run(name, {
        get text(): string {
          throw foreign;
        },
      });
      expect((error as Error).message).toBe(
        `${command} stopped on an unexpected error${classified}. The original session was left unchanged. Retry; if it fails again, check the OpenCode server log.`,
      );
      expect((error as Error).name).toBe("Error");
      // The original stays attached for the server's own error logging.
      expect((error as Error).cause).toBe(foreign);
    },
  );

  test("magic-trim --heuristic neither reads nor validates the percent", async () => {
    const run = await commands();
    const error = await withPercent("60%", () =>
      run("magic-trim", "--heuristic"),
    );
    expect(error).not.toBeInstanceOf(RetentionClassificationError);
    expect(String(error)).toContain(V2_ENDPOINT_REQUIRED);
  });
});

describe("OpenCode 2 response tool boundary", () => {
  const RESPONSE = "StructuredOutput";
  type Execute = (input: unknown, context: never) => Promise<unknown>;
  const harness = async (owned: OwnedSessions) => {
    const hooks = new Map<string, (event: never) => Promise<void> | void>();
    const executors = new Map<string, Execute>();
    const idle = { add: () => undefined, update: () => undefined };
    const tools = {
      add: (tool: { name: string; execute: Execute }) => {
        executors.set(tool.name, tool.execute);
      },
      update: () => undefined,
    };
    await setup(
      {
        options: {},
        agent: { transform: async (fn: (e: unknown) => void) => fn(idle) },
        command: { transform: async (fn: (e: unknown) => void) => fn(idle) },
        tool: { transform: async (fn: (e: unknown) => void) => fn(tools) },
        session: {
          hook: async (
            name: string,
            callback: (event: never) => Promise<void> | void,
          ) => {
            hooks.set(name, callback);
          },
          context: async () => [],
        },
        event: { subscribe: async function* () {} },
      } as never,
      owned,
    );
    const advertised = async (kind: string, sessionID: string) => {
      const event = {
        sessionID,
        tools: { [RESPONSE]: {}, read: {} },
        messages: [],
      };
      await hooks.get(kind)!(event as never);
      return Object.keys(event.tools);
    };
    const respond = (sessionID: string) =>
      executors.get(RESPONSE)!({}, { sessionID } as never);
    return { advertised, respond };
  };

  test("a session this process does not own never sees the response tool, in any request kind", async () => {
    const { advertised } = await harness(new OwnedSessions());
    for (const kind of ["context", "generate", "compaction"]) {
      expect([kind, await advertised(kind, "ses_user")]).toEqual([
        kind,
        ["read"],
      ]);
    }
  });

  test("only the per-turn generation request of an owned session sees it", async () => {
    const owned = new OwnedSessions();
    owned.register("ses_gen", "generation");
    owned.register("ses_ret", "retention");
    const { advertised } = await harness(owned);
    expect(await advertised("context", "ses_gen")).toEqual([RESPONSE, "read"]);
    expect(await advertised("generate", "ses_gen")).toEqual(["read"]);
    await expect(advertised("compaction", "ses_gen")).rejects.toThrow(
      owned.compactionBlocked,
    );
    expect(await advertised("context", "ses_ret")).toEqual(["read"]);
    expect(await advertised("generate", "ses_ret")).toEqual(["read"]);
  });

  test("the response tool acknowledges only inside a per-turn generation session", async () => {
    const owned = new OwnedSessions();
    owned.register("ses_gen", "generation");
    owned.register("ses_ret", "retention");
    const { respond } = await harness(owned);
    expect(await respond("ses_gen")).toEqual({ content: "Recorded." });
    await expect(respond("ses_user")).rejects.toThrow("only available");
    await expect(respond("ses_ret")).rejects.toThrow("only available");
    owned.release("ses_gen");
    await expect(respond("ses_gen")).rejects.toThrow("only available");
  });

  test("owned sessions are imported deny-all, the per-turn one allowing only the response tool", async () => {
    const turn = [
      userMessage("msg_u1", "Do it"),
      assistantMessage("msg_a1", [{ type: "text", text: "Done." }]),
      idleMessage("msg_i1"),
    ];
    const model = (state: FakeV2State) => ({
      ref: state.summarizerModel!,
      info: {
        limit: state.modelLimit,
        capabilities: { tools: true, input: ["text"], output: ["text"] },
      },
    });
    const imported = (state: FakeV2State) => {
      const rules: unknown[] = [];
      state.afterWrite = (method, sessionID) => {
        if (method === "import") {
          rules.push(
            structuredClone(state.sessions.get(sessionID)!.permissions),
          );
        }
      };
      return rules;
    };

    const generation = fakeV2State();
    const source = sourceSession(generation, "ses_src", turn);
    const generationRules = imported(generation);
    const owned = new OwnedSessions();
    await generateV2Summaries(
      fakeV2Host(generation, owned, echoGeneration()),
      owned,
      { session: source, messages: turn, instructions: [] },
      model(generation),
      [{ span: turn }],
      "index",
      { controlTimeoutMs: 1_000 },
    );
    expect(generationRules).toEqual([
      [
        { action: "*", resource: "*", effect: "deny" },
        { action: RESPONSE, resource: "*", effect: "allow" },
      ],
    ]);

    const retention = fakeV2State();
    const retained = sourceSession(retention, "ses_src", turn);
    const retentionRules = imported(retention);
    const retentionOwned = new OwnedSessions();
    await generateV2SessionSummary(
      fakeV2Host(retention, retentionOwned, () => ({
        text: "## Objective\n- Continue.",
      })),
      retentionOwned,
      { session: retained, messages: turn, instructions: [] },
      model(retention),
      1_000,
    );
    expect(retentionRules).toEqual([
      [{ action: "*", resource: "*", effect: "deny" }],
    ]);
  });
});

// Both hosts send one protocol text; it has to describe the marker layout of
// each. OpenCode 1 adds the marker to the turn's first user message
// (compact.test.ts); an OpenCode 2 owned session holds it as its own message.
describe("generation request protocol", () => {
  const turn = (n: number, answer = `Answer ${n}`): V2Message[] => [
    userMessage(`msg_u${n}`, `Request ${n}`),
    assistantMessage(`msg_a${n}`, [{ type: "text", text: answer }]),
    idleMessage(`msg_i${n}`),
  ];
  const boundary = {
    id: "msg_b",
    type: "synthetic",
    text: "boundary",
    metadata: { magicCompact: { boundary: true } },
    time: { created: 60 },
  } as V2Message;

  test("marks only the target, keeps the preceding turn as context and never sends the following turn", async () => {
    const state = fakeV2State();
    const following = turn(3, "FOLLOWING_DETAIL_9c1");
    const messages = [
      ...turn(1, "PRECEDING_DETAIL_4f2"),
      boundary,
      ...turn(2, words("A2", 300)),
      ...following,
    ];
    sourceSession(state, "ses_src", messages);
    const owned = new OwnedSessions();
    const seen: Array<{ prompt: string; history: V2Message[] }> = [];
    const { successor } = await executeV2Compact(
      fakeV2Host(state, owned, (prompt, history) => {
        seen.push({ prompt, history });
        return echoGeneration()(prompt);
      }),
      owned,
      "ses_src",
      1,
      { retention: { mode: "heuristic" }, controlTimeoutMs: 1_000 },
    );
    expect(seen).toHaveLength(1);
    const { prompt, history } = seen[0]!;
    // The owned session holds the marker as its own message directly before
    // the target's first user message, after the preceding turn's records;
    // the last user message is the request prompt.
    expect(history.map(message => message.type)).toEqual([
      "user",
      "assistant",
      "idle",
      "synthetic",
      "synthetic",
      "user",
      "assistant",
      "idle",
      "user",
    ]);
    const text = (index: number) => (history[index] as { text: string }).text;
    expect(text(0)).toBe("Request 1");
    expect(JSON.stringify(history[1])).toContain("PRECEDING_DETAIL_4f2");
    expect(text(3)).toBe("boundary");
    expect(text(4)).toMatch(
      /^<magic-compact-turn nonce="[\w-]{22}" ordinal="1" \/>$/u,
    );
    expect(text(5)).toBe("Request 2");
    expect(JSON.stringify(history)).not.toContain("FOLLOWING_DETAIL_9c1");
    expect(JSON.stringify(history.slice(0, -1))).not.toContain("Request 3");
    expect(prompt).not.toContain("FOLLOWING_DETAIL_9c1");
    // The global map keeps every user anchor, the following turn's included.
    expect(prompt).toContain(
      JSON.stringify({
        globalTurnIndex: [1, 2, 3].map(n => ({
          sourceOrdinal: n,
          anchor: `Request ${n}`,
        })),
      }),
    );
    expect(prompt).toContain(GENERATION_PROTOCOL);
    expect(prompt.split("\n").at(-1)).toBe(
      JSON.stringify({
        turns: [{ ordinal: 1, marker: text(4), expectedInlineImages: 0 }],
      }),
    );
    expect(GENERATION_PROTOCOL).toContain(
      "either added to that turn's first user message or sent as a separate synthetic message directly before it",
    );
    expect(GENERATION_PROTOCOL).toContain(
      "The marked target turn runs from its marker to the end of the retained history.",
    );
    expect(GENERATION_PROTOCOL).toContain(
      "retained history before the marker is preceding context for orientation only",
    );
    expect(GENERATION_PROTOCOL).not.toContain("stop");
    expect(GENERATION_PROTOCOL).not.toContain("injected");
    // The source is unchanged and its successor keeps the following turn.
    expect(state.messages.get("ses_src")).toEqual(messages);
    const result = state.messages.get(successor!.session.id)!;
    expect(
      result.slice(-3).map(message => JSON.stringify({ ...message, id: "" })),
    ).toEqual(following.map(message => JSON.stringify({ ...message, id: "" })));
  });

  test("drops the preceding turn when it does not fit the halo limit", async () => {
    const state = fakeV2State();
    state.modelLimit = { context: 200_000, input: 20_000, output: 32_000 };
    const messages = [
      ...turn(1, words("P", 20_000)),
      boundary,
      ...turn(2, words("A2", 300)),
      ...turn(3),
    ];
    sourceSession(state, "ses_src", messages);
    const owned = new OwnedSessions();
    const histories: V2Message[][] = [];
    await executeV2Compact(
      fakeV2Host(state, owned, (prompt, history) => {
        histories.push(history);
        return echoGeneration()(prompt);
      }),
      owned,
      "ses_src",
      1,
      { retention: { mode: "heuristic" }, controlTimeoutMs: 1_000 },
    );
    expect(histories).toHaveLength(1);
    expect(histories[0]!.map(message => message.type)).toEqual([
      "synthetic",
      "user",
      "assistant",
      "idle",
      "user",
    ]);
    expect((histories[0]![1] as { text: string }).text).toBe("Request 2");
  });

  test("the explicit request estimate charges the complete protocol, schema and marker", () => {
    const nonce = "n".repeat(22);
    const target = {
      ordinal: 1,
      marker: generationTurnMarker(nonce, 1),
      expectedInlineImages: 0,
    };
    const prompt = buildGenerationPrompt([target], "index");
    expect(prompt).toContain(GENERATION_PROTOCOL);
    expect(estimateGenerationExplicitRequestTokens([target], "index")).toBe(
      countGptTokens(JSON.stringify(GENERATION_RESPONSE_SCHEMA))
        + countGptTokens(prompt)
        + countGptTokens(target.marker),
    );
  });
});

describe("OpenCode 2 retention summary budget", () => {
  const history = (): V2Message[] => [
    userMessage("msg_u1", "Write the file", {
      files: [imageFile(largeImage)],
    }),
    assistantMessage("msg_a1", [
      toolItem(
        "call_w",
        "write",
        { path: "/w.txt", content: words("W", 300) },
        [{ type: "text", text: "Wrote file." }],
      ),
    ]),
    idleMessage("msg_i1"),
    userMessage("msg_u2", "Read it"),
    assistantMessage("msg_a2", [
      toolItem("call_r", "read", { path: "/w.txt" }, [
        { type: "text", text: words("R", 400) },
      ]),
    ]),
    idleMessage("msg_i2"),
  ];
  const trim = async (
    messages: V2Message[],
    arrange: (state: FakeV2State) => void,
  ) => {
    const state = fakeV2State();
    sourceSession(state, "ses_src", messages);
    arrange(state);
    const owned = new OwnedSessions();
    const outcome = await executeV2Trim(
      fakeV2Host(state, owned, () => ({ text: "## Objective\n- Continue." })),
      owned,
      "ses_src",
      0,
      {
        retention: {
          mode: "task-aware",
          apiKey: "test-key",
          fetch: scoredJevFetch(() => 0.1),
        },
        controlTimeoutMs: 1_000,
      },
    ).catch((error: Error) => error);
    return { state, outcome };
  };

  test("a model too small for the visible history fails before any session or request", async () => {
    const { state, outcome } = await trim(history(), state => {
      state.modelLimit = { context: 12_000, output: 1_000 };
    });
    expect(String(outcome)).toContain(
      "The session is too large for the configured compaction model",
    );
    expect(state.requests).toHaveLength(0);
    expect(state.calls).toEqual([]);
  });

  test("a model without image input fails before any session or request when an image is visible", async () => {
    const { state, outcome } = await trim(history(), state => {
      state.modelInput = ["text"];
    });
    expect(String(outcome)).toContain("requires an image-capable");
    expect(state.requests).toHaveLength(0);
    expect(state.calls).toEqual([]);
  });

  test("history behind a plain native compaction does not count against the budget", async () => {
    const hiddenPrefix: V2Message[] = [
      userMessage("msg_h", words("H", 20_000)),
      assistantMessage("msg_ha", [{ type: "text", text: words("HA", 20_000) }]),
      idleMessage("msg_hi"),
      {
        id: "msg_c",
        type: "compaction",
        status: "completed",
        reason: "auto",
        summary: "earlier work",
        recent: "",
        time: { created: 50 },
      } as V2Message,
    ];
    const visible = history().slice(3);
    const tight = (state: FakeV2State) => {
      state.modelLimit = { context: 40_000, output: 1_000 };
      state.modelInput = ["text"];
    };
    const { state, outcome } = await trim([...hiddenPrefix, ...visible], tight);
    expect(outcome).not.toBeInstanceOf(Error);
    expect(state.requests).toHaveLength(1);
    // The same records without the compaction are all visible and too large.
    const { state: full, outcome: refused } = await trim(
      [...hiddenPrefix.slice(0, 3), ...visible],
      tight,
    );
    expect(String(refused)).toContain("too large");
    expect(full.calls).toEqual([]);
  });

  test("compact resolves the summarizer model once for retention and every generation", async () => {
    const state = fakeV2State();
    const turn = (n: number): V2Message[] => [
      userMessage(`msg_u${n}`, `Request ${n}`),
      assistantMessage(`msg_a${n}`, [
        { type: "text", text: words(`A${n}`, 300) },
        toolItem(`call_${n}`, "read", { path: "/f" }, [
          { type: "text", text: words(`R${n}`, 300) },
        ]),
      ]),
      idleMessage(`msg_i${n}`),
    ];
    sourceSession(state, "ses_src", [...turn(1), ...turn(2), ...turn(3)]);
    const owned = new OwnedSessions();
    const host = fakeV2Host(state, owned, (text, messages) =>
      text.includes("globalTurnIndex")
        ? echoGeneration()(text)
        : { text: `## Objective\n- ${messages.length}` },
    );
    let lookups = 0;
    const agentGet = host.agent.get.bind(host.agent);
    host.agent.get = (async (...args: Parameters<typeof agentGet>) => {
      lookups += 1;
      return agentGet(...args);
    }) as typeof host.agent.get;
    const { successor } = await executeV2Compact(host, owned, "ses_src", 1, {
      retention: {
        mode: "task-aware",
        apiKey: "test-key",
        fetch: scoredJevFetch(() => 0.1),
      },
      controlTimeoutMs: 1_000,
    });
    expect(successor).not.toBeNull();
    expect(lookups).toBe(1);
    // One retention summary, then one request per target turn.
    expect(state.requests).toHaveLength(3);
  });
});

describe("OpenCode 2 compaction of host records", () => {
  const disposable = words(
    "Called the Read tool with the following input",
    600,
  );
  const turnWithHostRecords = (): V2Message[] => [
    userMessage("msg_u1", "Run the review"),
    {
      id: "msg_s1",
      type: "synthetic",
      text: disposable,
      time: { created: 10 },
    } as V2Message,
    {
      id: "msg_s2",
      type: "synthetic",
      text: "Continue if you have next steps.",
      description: "Continuing after restart",
      metadata: { notice: "restart" },
      time: { created: 11 },
    } as V2Message,
    {
      id: "msg_s3",
      type: "synthetic",
      text: `First part\n\n<task id="t">${words("T", 50)}</task>`,
      time: { created: 12 },
    } as V2Message,
    {
      id: "msg_skill",
      type: "skill",
      skill: "review",
      name: "review",
      text: "Skill instructions",
      time: { created: 13 },
    } as V2Message,
    assistantMessage("msg_a1", [{ type: "text", text: words("A1", 300) }]),
    idleMessage("msg_i1"),
    {
      id: "msg_report",
      type: "synthetic",
      text: `<subagent sessionID="ses_kid" state="completed" description="review">\n${words("REPORT", 200)}\n</subagent>`,
      description: "review",
      metadata: {
        source: "subagent",
        childID: "ses_kid",
        agent: "general",
        state: "completed",
      },
      time: { created: 14 },
    } as V2Message,
  ];

  test("drops disposable host text from summarized turns and keeps reports, skills and recent turns exactly", async () => {
    const recent: V2Message[] = [
      userMessage("msg_u2", "Next"),
      {
        id: "msg_s4",
        type: "synthetic",
        text: disposable,
        time: { created: 20 },
      } as V2Message,
      assistantMessage("msg_a2", [{ type: "text", text: words("A2", 300) }]),
      idleMessage("msg_i2"),
    ];
    const source = [...turnWithHostRecords(), ...recent];
    const state = fakeV2State();
    sourceSession(state, "ses_src", source);
    const owned = new OwnedSessions();
    const { successor, outcome } = await executeV2Compact(
      fakeV2Host(state, owned, echoGeneration()),
      owned,
      "ses_src",
      1,
      { retention: { mode: "heuristic" }, controlTimeoutMs: 1_000 },
    );
    const stored = state.messages.get(successor!.session.id)!;
    const strip = (message: V2Message) =>
      JSON.stringify({ ...message, id: "x" });
    const kept = new Map(
      ["msg_u1", "msg_s3", "msg_skill", "msg_report", "msg_u2", "msg_s4"].map(
        id => [id, source.find(message => message.id === id)!],
      ),
    );
    expect(stored.map(message => message.type)).toEqual([
      "user",
      "synthetic",
      "skill",
      "assistant",
      "idle",
      "synthetic",
      "synthetic",
      "user",
      "synthetic",
      "assistant",
      "idle",
    ]);
    expect(stored.map(strip)).toContain(strip(kept.get("msg_s3")!));
    expect(stored.map(strip)).toContain(strip(kept.get("msg_report")!));
    expect(stored.map(strip)).toContain(strip(kept.get("msg_skill")!));
    expect(stored.map(strip)).toContain(strip(kept.get("msg_s4")!));
    expect(strip(stored[0]!)).toBe(strip(kept.get("msg_u1")!));
    expect(stored[6]).toMatchObject({
      metadata: { magicCompact: { boundary: true } },
    });
    expect(JSON.stringify(stored)).not.toContain("Continue if you have next");
    expect(
      stored.filter(message => JSON.stringify(message).includes(disposable)),
    ).toHaveLength(1);
    const before = countV2ModelFacingTokens(source).tokens;
    const after = countV2ModelFacingTokens(stored).tokens;
    expect(outcome).toContain("Request turns summarized: 1");
    expect(after).toBeLessThan(before);
  });

  test("classifies host records by their annotations", () => {
    const synthetic = (text: string, metadata?: Record<string, unknown>) =>
      ({
        id: "msg_x",
        type: "synthetic",
        text,
        ...(metadata ? { metadata } : {}),
        time: { created: 1 },
      }) as V2Message;
    expect(isDisposableV2HostRecord(synthetic("Called the Read tool"))).toBe(
      true,
    );
    expect(
      isDisposableV2HostRecord(synthetic("x", { notice: "restart" })),
    ).toBe(true);
    for (const kept of [
      synthetic('<task id="t">done</task>'),
      synthetic("The following tool was executed by the user"),
      synthetic("a\n\nThe user made the following comment: hi"),
      synthetic("x", { source: "shell", shellID: "sh" }),
      synthetic("x", { source: "subagent", childID: "ses_kid" }),
      synthetic("x", { notice: "restart", extra: true }),
      synthetic("x", { magicCompact: { boundary: true } }),
      userMessage("msg_u", "Called the Read tool"),
      {
        id: "msg_sys",
        type: "system",
        text: "notice",
        metadata: { notice: "instructions" },
        time: { created: 1 },
      } as V2Message,
    ]) {
      expect(isDisposableV2HostRecord(kept)).toBe(false);
    }
  });
});

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { createHash, randomBytes } from "node:crypto";
import { closeSync, constants, lstatSync, openSync } from "node:fs";
import { chmod, mkdir, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { GENERATION_PROTOCOL } from "../src/policy/protocol";
import {
  COMPACT_ARGUMENTS_ERROR,
  COMPACT_NOOP,
  COMPACT_THRESHOLD_ERROR,
  commandFailureMessage,
  parseMagicCompactArguments,
  parseMagicTrimArguments,
  RECOVER_NOOP,
  STATS_ARGUMENTS_ERROR,
  TRIM_ARGUMENTS_ERROR,
  TRIM_THRESHOLD_ERROR,
} from "../src/commands";
import { preservedTurnsFeedback } from "../src/stats/constants";
import { setup } from "../src/host-v2";
import {
  commandRetentionPolicy,
  taskAwareRetentionPolicy,
} from "../src/policy/retention";
import {
  excerpt,
  RETENTION_KEEP_THRESHOLD_PERCENT_ENV,
  RETENTION_SUMMARY_EXCERPT_BYTES,
  RetentionClassificationError,
  retentionFailureGuidance,
  type RetentionFetch,
} from "../src/retention/classifier";
import plugin from "../src/index";
import tuiPlugin from "../src/tui";
import {
  cachePath,
  commitPlannedOmission,
  planOmission,
  readOmittedContent,
} from "../src/storage/omission";
import { InvalidStatsError, readStats } from "../src/storage/stats";
import {
  executeV2Compact,
  isDisposableV2HostRecord,
  planV2Compaction,
} from "../src/v2/compact";
import { generateV2Summaries, V2GenerationError } from "../src/v2/generation";
import {
  countV2PayloadTokens,
  trimV2Tool,
  v2ToolCalls,
} from "../src/v2/history";
import { connectV2Host, V2_ENDPOINT_REQUIRED } from "../src/v2/host";
import {
  countV2ModelFacingTokens,
  planV2MediaProjection,
  projectV2RequestMessages,
  v2VisibleImageIndexes,
} from "../src/v2/media";
import { generateV2SessionSummary, OwnedSessions } from "../src/v2/owned";
import { READABLE_REASONING_HEADER } from "../src/policy/reasoning";
import { SessionSummaryError } from "../src/policy/retention";
import {
  SESSION_SUMMARY_PROMPT,
  SESSION_SUMMARY_TOOL_PROMPT,
} from "../src/policy/protocol";
import { estimateGenerationExplicitRequestTokens } from "../src/policy/generation";
import {
  SourceNotAtRestError,
  SuccessorFailure,
} from "../src/policy/successor";
import { mediaPlaceholderText } from "../src/policy/media";
import { countGptTokens } from "../src/stats/tokenize";
import {
  readV2SourceAtRest,
  v2StorageKey,
  type V2Message,
} from "../src/v2/session";
import { executeV2Trim } from "../src/v2/trim";
import {
  scoredJevFetch,
  scoredOpenAIFetch,
  tokenLimitResponse,
  type JevCall,
} from "./retention-host";
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
  SPECIAL_TOKEN,
  toolItem,
  userMessage,
  type ContextEvent,
  type FakeReply,
  type FakeV2State,
} from "./v2-host";

// Plugin storage and the service registration both live in scratch
// directories, so no test reads or contacts a real OpenCode service.
const originalDataHome = process.env.XDG_DATA_HOME;
const originalStateHome = process.env.XDG_STATE_HOME;
let storageDirectory: string;
let stateDirectory: string;
beforeEach(async () => {
  storageDirectory = await mkdtemp(join(tmpdir(), "magic-compact-v2-test-"));
  stateDirectory = await mkdtemp(join(tmpdir(), "magic-compact-v2-state-"));
  process.env.XDG_DATA_HOME = storageDirectory;
  process.env.XDG_STATE_HOME = stateDirectory;
});
afterEach(async () => {
  if (originalDataHome === undefined) delete process.env.XDG_DATA_HOME;
  else process.env.XDG_DATA_HOME = originalDataHome;
  if (originalStateHome === undefined) delete process.env.XDG_STATE_HOME;
  else process.env.XDG_STATE_HOME = originalStateHome;
  await rm(storageDirectory, { recursive: true, force: true });
  await rm(stateDirectory, { recursive: true, force: true });
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

  test("keeps a short subagent report and prompt unchanged and names no child session for other tools", async () => {
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
    const read = toolItem(
      "call_k2",
      "read",
      {},
      [{ type: "text", text: SUBAGENT_REPORT }],
      { sessionID: "ses_0123abcXYZ" },
    ) as never as Parameters<typeof trimV2Tool>[0];
    const readTrimmed = await trimV2Tool(
      read,
      async () => "abcdefabcdef:omitted-EEEEEEEEEEEEEEEEEEEEEE",
    );
    expect((readTrimmed!.state as { content: unknown }).content).toEqual([
      {
        type: "text",
        text: `[Historical read result omitted (${Buffer.byteLength(SUBAGENT_REPORT)} bytes); Content ID: abcdefabcdef:omitted-EEEEEEEEEEEEEEEEEEEEEE]`,
      },
    ]);
  });

  test("gives calls that share a call ID distinct references", () => {
    const message = assistantMessage("msg_x", [
      toolItem("call_1", "read", {}, [{ type: "text", text: "a" }]),
      { type: "text", text: "between" },
      toolItem("call_1", "read", {}, [{ type: "text", text: "b" }]),
    ]);
    const refs = v2ToolCalls(message).map(call => call.ref);
    expect(refs).toHaveLength(2);
    expect(new Set(refs).size).toBe(2);
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

  test("reads an open assistant the export leaves out and refuses any other difference from the export", async () => {
    const state = fakeV2State();
    sourceSession(state, "ses_src", [
      ...toolHistory(),
      assistantMessage("msg_open", [], { time: { created: 9 } }),
    ]);
    const host = fakeV2Host(state, new OwnedSessions());
    // The export the host returns lacks the open assistant; the pages keep it
    // and the refusal names it.
    await expect(
      readV2SourceAtRest(host, "Magic Trim", "ses_src", 1_000),
    ).rejects.toThrow("Record: assistant message msg_open;");
    const exportOf = host.session.export.bind(host.session);
    host.session.export = (async (input: never) => {
      const exported = await exportOf(input);
      return { ...exported, messages: exported.messages.slice(1) };
    }) as never;
    await expect(
      readV2SourceAtRest(host, "Magic Trim", "ses_src", 1_000),
    ).rejects.toThrow(
      "Magic Trim could not read a complete session history. The session was left unchanged.",
    );
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

// A disposable loopback HTTP server that records each request's path and
// Authorization header. `hold` keeps a request open until its client aborts
// it, and records that abort; a promised response answers when it settles.
// `waitFor` resolves once a recorded request or abort makes `ready` true.
type Seen = { path: string; authorization: string | null };
function loopback(
  respond: (
    url: URL,
    request: Request,
  ) => Response | Promise<Response> | "hold",
): {
  url: string;
  seen: Seen[];
  aborted: string[];
  waitFor: (ready: () => boolean) => Promise<void>;
  stop: () => Promise<void>;
} {
  const seen: Seen[] = [];
  const aborted: string[] = [];
  let recorded = Promise.withResolvers<void>();
  const record = () => {
    recorded.resolve();
    recorded = Promise.withResolvers<void>();
  };
  const server = Bun.serve({
    port: 0,
    hostname: "127.0.0.1",
    fetch: request => {
      const url = new URL(request.url);
      seen.push({
        path: url.pathname,
        authorization: request.headers.get("authorization"),
      });
      record();
      const response = respond(url, request);
      if (response !== "hold") return response;
      return new Promise<Response>(resolve => {
        request.signal.addEventListener("abort", () => {
          aborted.push(url.pathname);
          record();
          resolve(new Response("aborted", { status: 499 }));
        });
      });
    },
  });
  return {
    url: `http://127.0.0.1:${server.port}`,
    seen,
    aborted,
    waitFor: async ready => {
      while (!ready()) await recorded.promise;
    },
    stop: () => server.stop(true),
  };
}

// Every microtask queued so far has run.
const settleMicrotasks = () =>
  new Promise<void>(resolve => setImmediate(resolve));

const serverInfo = (pid: number, version = "2.0.28") =>
  Response.json({ version, pid, urls: [], paths: { tmp: "/tmp" } });

const basic = (username: string, password: string) =>
  `Basic ${Buffer.from(`${username}:${password}`).toString("base64")}`;

// Writes OpenCode 2.0.28's stable-channel service registration under the
// suite's scratch XDG_STATE_HOME.
async function register(record: unknown) {
  const file = join(stateDirectory, "opencode", "service.json");
  await mkdir(dirname(file), { recursive: true });
  await Bun.write(
    file,
    typeof record === "string" ? record : JSON.stringify(record),
  );
}

describe("OpenCode 2 endpoint qualification", () => {
  const session = {
    id: "ses_src",
    projectID: "p",
    time: { created: 1 },
    location: LOCATION,
  };
  const PASSWORD = "synthetic-registration-password";
  // A server of process 7 that requires the registration's Basic auth.
  const managed = (remote: unknown = session, pid = 7) =>
    loopback((url, request) => {
      if (request.headers.get("authorization") !== basic("opencode", PASSWORD))
        return new Response("no", { status: 401 });
      if (url.pathname === "/api/info") return serverInfo(pid);
      return Response.json({ data: remote });
    });
  const env = () => ({ XDG_STATE_HOME: stateDirectory });
  const refusal = async (
    connecting: Promise<unknown>,
    detail: string,
    hidden: string[] = [],
  ) => {
    const error = (await connecting.then(
      () => {
        throw new Error("Expected a refusal.");
      },
      (reason: unknown) => reason,
    )) as Error & { detail?: string };
    expect(error.detail).toBe(detail);
    expect(error.message).toStartWith(V2_ENDPOINT_REQUIRED);
    // No credential, URL or file system path reaches the person.
    for (const text of [PASSWORD, stateDirectory, "127.0.0.1", ...hidden])
      expect(error.message).not.toContain(text);
    return error;
  };

  test("a registered service of this process qualifies with its own credentials and no manual variables", async () => {
    const server = managed();
    try {
      await register({
        id: "svc",
        version: "2.0.28",
        url: server.url,
        pid: 7,
        password: PASSWORD,
      });
      let reads = 0;
      const host = await connectV2Host(
        "ses_src",
        async () => {
          reads++;
          return session;
        },
        { env: env(), pid: 7, controlTimeoutMs: 1_000 },
      );
      // One identity read, no separate discovery probe.
      expect(server.seen.map(seen => seen.path)).toEqual([
        "/api/info",
        "/api/session/ses_src",
      ]);
      expect(reads).toBe(1);
      expect(
        server.seen.every(
          seen => seen.authorization === basic("opencode", PASSWORD),
        ),
      ).toBe(true);
      // Later requests keep the same credentials.
      expect((await host.server.info()).pid).toBe(7);
      expect(server.seen.at(-1)!.authorization).toBe(
        basic("opencode", PASSWORD),
      );
    } finally {
      await server.stop();
    }
  });

  test("without XDG_STATE_HOME the registration is read under the home directory", async () => {
    const server = managed();
    try {
      const file = join(
        stateDirectory,
        ".local",
        "state",
        "opencode",
        "service.json",
      );
      await mkdir(dirname(file), { recursive: true });
      await Bun.write(
        file,
        JSON.stringify({ url: server.url, pid: 7, password: PASSWORD }),
      );
      await connectV2Host("ses_src", async () => session, {
        env: { XDG_STATE_HOME: "", HOME: stateDirectory },
        pid: 7,
        controlTimeoutMs: 1_000,
      });
      expect(server.seen).toHaveLength(2);
    } finally {
      await server.stop();
    }
  });

  test("a missing, malformed or foreign registration sends no request", async () => {
    const server = managed();
    let reads = 0;
    const connect = () =>
      connectV2Host(
        "ses_src",
        async () => {
          reads++;
          return session;
        },
        { env: env(), pid: 7, controlTimeoutMs: 1_000 },
      );
    try {
      await refusal(connect(), "not-configured");
      const valid = {
        id: "svc",
        version: "2.0.28",
        url: server.url,
        pid: 7,
        password: PASSWORD,
      };
      for (const record of [
        "{not json",
        "[]",
        { ...valid, pid: "7" },
        { ...valid, pid: 7.5 },
        { ...valid, url: 1 },
        { ...valid, url: "not a url" },
        { ...valid, url: server.url.replace("http:", "ftp:") },
        { ...valid, url: server.url.replace("//", `//opencode:${PASSWORD}@`) },
        { ...valid, version: 2 },
        { ...valid, password: 1 },
        { ...valid, id: {} },
      ]) {
        await register(record);
        await refusal(connect(), "invalid-registration");
      }
      // A live service of another process is never sent this server's
      // requests or the registration's credentials.
      await register({ ...valid, pid: 8 });
      await refusal(connect(), "foreign-service");
      expect(server.seen).toEqual([]);
      expect(reads).toBe(0);
    } finally {
      await server.stop();
    }
  });

  test("the server must report the registered PID and version", async () => {
    const registration = (url: string) => ({
      id: "svc",
      version: "2.0.28",
      url,
      pid: 7,
      password: PASSWORD,
    });
    const other = managed(session, 8);
    const older = loopback(url =>
      url.pathname === "/api/info"
        ? serverInfo(7, "2.0.27")
        : Response.json({ data: session }),
    );
    try {
      for (const server of [other, older]) {
        await register(registration(server.url));
        await refusal(
          connectV2Host("ses_src", async () => session, {
            env: env(),
            pid: 7,
            controlTimeoutMs: 1_000,
          }),
          "wrong-server",
        );
        expect(server.seen.map(seen => seen.path)).toEqual(["/api/info"]);
      }
      // Without a recorded version, any version string qualifies.
      await register({ ...registration(older.url), version: undefined });
      await connectV2Host("ses_src", async () => session, {
        env: env(),
        pid: 7,
        controlTimeoutMs: 1_000,
      });
    } finally {
      await other.stop();
      await older.stop();
    }
  });

  test("a malformed server identity is refused before the session is read", async () => {
    for (const info of [
      null,
      [],
      { pid: "7", version: "2.0.28" },
      { pid: 7 },
      { pid: 7, version: 2 },
    ]) {
      const server = loopback(url =>
        url.pathname === "/api/info"
          ? Response.json(info)
          : Response.json({ data: session }),
      );
      try {
        // A record without a version still needs a version string.
        await register({ url: server.url, pid: 7 });
        await refusal(
          connectV2Host("ses_src", async () => session, {
            env: env(),
            pid: 7,
            controlTimeoutMs: 1_000,
          }),
          "wrong-server",
        );
        expect(server.seen.map(seen => seen.path)).toEqual(["/api/info"]);
      } finally {
        await server.stop();
      }
    }
  });

  test("an explicit serverUrl is the only endpoint, whether it succeeds or fails", async () => {
    // A ready registration of this very process is never consulted.
    const registered = managed();
    const explicit = loopback((url, request) => {
      if (
        request.headers.get("authorization")
        !== basic("operator", "explicit-pw")
      )
        return new Response("no", { status: 401 });
      if (url.pathname === "/api/info") return serverInfo(7);
      return Response.json({ data: session });
    });
    const failing = loopback(() => new Response("down", { status: 500 }));
    try {
      await register({ url: registered.url, pid: 7, password: PASSWORD });
      const connect = (serverUrl: unknown, password = "explicit-pw") =>
        connectV2Host("ses_src", async () => session, {
          serverUrl,
          env: {
            ...env(),
            OPENCODE_SERVER_USERNAME: "operator",
            OPENCODE_SERVER_PASSWORD: password,
          },
          pid: 7,
          controlTimeoutMs: 1_000,
        });
      await connect(` ${explicit.url}/ `);
      expect(explicit.seen).toHaveLength(2);
      await refusal(connect(explicit.url, "wrong-pw"), "auth-refused", [
        "wrong-pw",
      ]);
      await refusal(connect(failing.url), "wrong-server", ["explicit-pw"]);
      const authority = explicit.url.slice("http://".length);
      for (const invalid of [
        4101,
        "",
        "127.0.0.1:4096",
        "ftp://127.0.0.1:4096",
        `http://operator:explicit-pw@${authority}`,
        `http://@${authority}`,
        `${explicit.url}/?token=1`,
        `${explicit.url}/?`,
        `${explicit.url}/#here`,
      ])
        await refusal(connect(invalid), "invalid-server-url", ["explicit-pw"]);
      expect(failing.seen).toHaveLength(1);
      expect(explicit.seen).toHaveLength(3);
      expect(registered.seen).toEqual([]);
    } finally {
      await registered.stop();
      await explicit.stop();
      await failing.stop();
    }
  });

  test("an endpoint under a base path is called under that path", async () => {
    const server = loopback(url => {
      if (url.pathname === "/opencode/api/info") return serverInfo(7);
      if (url.pathname === "/opencode/api/session/ses_src")
        return Response.json({ data: session });
      return new Response("no", { status: 404 });
    });
    try {
      await connectV2Host("ses_src", async () => session, {
        serverUrl: `${server.url}/opencode/`,
        env: env(),
        pid: 7,
        controlTimeoutMs: 1_000,
      });
      expect(server.seen.map(seen => seen.path)).toEqual([
        "/opencode/api/info",
        "/opencode/api/session/ses_src",
      ]);
    } finally {
      await server.stop();
    }
  });

  test("every successor command connects through the registration, and only there", async () => {
    const server = managed();
    const commands = new Map<string, (input: unknown) => Promise<void>>();
    const editor = { add: () => undefined, update: () => undefined };
    const cleanup = await setup({
      options: {},
      agent: { transform: async (fn: (e: unknown) => void) => fn(editor) },
      command: {
        transform: async (fn: (e: unknown) => void) =>
          fn({
            add: (command: {
              name: string;
              execute: (input: unknown) => Promise<void>;
            }) => commands.set(command.name, command.execute),
          }),
      },
      tool: { transform: async (fn: (e: unknown) => void) => fn(editor) },
      session: { hook: async () => undefined, get: async () => session },
      event: { subscribe: async function* () {} },
    } as never);
    try {
      await register({
        url: server.url,
        pid: process.pid + 1,
        password: PASSWORD,
      });
      for (const name of ["magic-compact", "magic-trim", "magic-recover"]) {
        const error = await commands.get(name)!({
          sessionID: "ses_src",
          prompt: { text: name === "magic-recover" ? "" : "--heuristic" },
        }).catch((reason: Error) => reason);
        expect((error as Error).message).toBe(
          `${V2_ENDPOINT_REQUIRED}\nReason: The registered OpenCode service belongs to another OpenCode process.`,
        );
      }
      expect(server.seen).toEqual([]);
    } finally {
      cleanup();
      await server.stop();
    }
  });

  test("a redirect is never followed", async () => {
    const witness = managed();
    const redirecting = loopback(url =>
      Response.redirect(`${witness.url}${url.pathname}`, 307),
    );
    try {
      await register({ url: redirecting.url, pid: 7, password: PASSWORD });
      await refusal(
        connectV2Host("ses_src", async () => session, {
          env: env(),
          pid: 7,
          controlTimeoutMs: 1_000,
        }),
        "wrong-server",
      );
      expect(redirecting.seen.map(seen => seen.path)).toEqual(["/api/info"]);
      expect(witness.seen).toEqual([]);
    } finally {
      await witness.stop();
      await redirecting.stop();
    }
  });

  test("a qualified connection never follows a later redirect", async () => {
    const witness = managed();
    const server = loopback((url, request) => {
      if (request.headers.get("authorization") !== basic("opencode", PASSWORD))
        return new Response("no", { status: 401 });
      if (url.pathname === "/api/info") return serverInfo(7);
      if (url.pathname === "/api/session/ses_src")
        return Response.json({ data: session });
      return Response.redirect(`${witness.url}${url.pathname}`, 307);
    });
    try {
      await register({ url: server.url, pid: 7, password: PASSWORD });
      const host = await connectV2Host("ses_src", async () => session, {
        env: env(),
        pid: 7,
        controlTimeoutMs: 1_000,
      });
      await expect(
        host.session.export({ sessionID: "ses_src", sanitize: false }),
      ).rejects.toThrow();
      expect(server.seen.map(seen => seen.path)).toEqual([
        "/api/info",
        "/api/session/ses_src",
        "/api/experimental/session/ses_src/export",
      ]);
      expect(witness.seen).toEqual([]);
    } finally {
      await witness.stop();
      await server.stop();
    }
  });

  test("the endpoint must return the same well-formed source session", async () => {
    const failed = async () => {
      throw new Error("in-process read failed");
    };
    // The in-process read comes first: when it fails or has no well-formed
    // identity of the source, the session is never requested.
    const cases: Array<[unknown, () => Promise<unknown>, boolean]> = [
      [{ ...session, projectID: "other" }, async () => session, true],
      [{ ...session, id: "ses_other" }, async () => session, true],
      [session, async () => ({ ...session, id: "ses_other" }), false],
      [session, async () => ({ ...session, location: undefined }), false],
      [session, async () => ({ ...session, time: { created: "1" } }), false],
      [{ ...session, location: undefined }, async () => session, true],
      [session, failed, false],
    ];
    for (const [remote, readLocal, requestsSession] of cases) {
      const server = managed(remote);
      try {
        await register({ url: server.url, pid: 7, password: PASSWORD });
        await refusal(
          connectV2Host("ses_src", readLocal, {
            env: env(),
            pid: 7,
            controlTimeoutMs: 1_000,
          }),
          "wrong-session",
          ["in-process read failed"],
        );
        expect(server.seen.map(seen => seen.path)).toEqual([
          "/api/info",
          ...(requestsSession ? ["/api/session/ses_src"] : []),
        ]);
      } finally {
        await server.stop();
      }
    }
    const missing = loopback((url, request) => {
      if (request.headers.get("authorization") !== basic("opencode", PASSWORD))
        return new Response("no", { status: 401 });
      if (url.pathname === "/api/info") return serverInfo(7);
      return Response.json(
        { _tag: "SessionNotFound", message: "ses_src not found" },
        { status: 404 },
      );
    });
    try {
      await register({ url: missing.url, pid: 7, password: PASSWORD });
      await refusal(
        connectV2Host("ses_src", async () => session, {
          env: env(),
          pid: 7,
          controlTimeoutMs: 1_000,
        }),
        "wrong-session",
        ["not found"],
      );
    } finally {
      await missing.stop();
    }
  });

  test("an unreachable, slow or cancelled check is reported as such and sends nothing more", async () => {
    const closed = loopback(() => serverInfo(7));
    await closed.stop();
    await register({ url: closed.url, pid: 7, password: PASSWORD });
    const connect = (
      signal?: AbortSignal,
      read = async () => session,
      controlTimeoutMs = 100,
    ) =>
      connectV2Host("ses_src", read, {
        env: env(),
        pid: 7,
        controlTimeoutMs,
        ...(signal ? { signal } : {}),
      });
    await refusal(connect(), "unreachable");

    const holding = loopback(url =>
      url.pathname === "/api/info" ? "hold" : Response.json({ data: session }),
    );
    try {
      await register({ url: holding.url, pid: 7, password: PASSWORD });
      await refusal(connect(), "timed-out");
      await holding.waitFor(() => holding.aborted.length === 1);
      expect(holding.aborted).toEqual(["/api/info"]);

      // Cancelled while the server identity read is open.
      const activation = new AbortController();
      const cancelled = connect(activation.signal, undefined, 60_000);
      await holding.waitFor(() => holding.seen.length === 2);
      activation.abort();
      await refusal(cancelled, "cancelled");
      await holding.waitFor(() => holding.aborted.length === 2);

      // Already cancelled: no request at all.
      const seen = holding.seen.length;
      await refusal(connect(activation.signal), "cancelled");
      expect(holding.seen).toHaveLength(seen);
    } finally {
      await holding.stop();
    }

    // The in-process read comes before the session request. One that ignores
    // its signal is still open when the check settles as cancelled, and its
    // late result starts no request.
    const server = managed();
    try {
      await register({ url: server.url, pid: 7, password: PASSWORD });
      const activation = new AbortController();
      const entered = Promise.withResolvers<void>();
      const held = Promise.withResolvers<unknown>();
      // No deadline settles it first.
      const late = connect(
        activation.signal,
        () => {
          entered.resolve();
          return held.promise;
        },
        60_000,
      );
      await entered.promise;
      // Long enough for a session request started with the read to arrive.
      await Bun.sleep(100);
      expect(server.seen.map(seen => seen.path)).toEqual(["/api/info"]);
      activation.abort();
      await refusal(late, "cancelled");
      held.resolve(session);
      await settleMicrotasks();
      await Bun.sleep(100);
      expect(server.seen.map(seen => seen.path)).toEqual(["/api/info"]);
    } finally {
      await server.stop();
    }
  });

  test("a registration read that does not finish is bounded by the deadline and the caller's signal", async () => {
    // A FIFO holds the read open, past its abort, until a writer opens it.
    const file = join(stateDirectory, "opencode", "service.json");
    await mkdir(dirname(file), { recursive: true });
    await rm(file, { force: true });
    expect(Bun.spawnSync(["mkfifo", file]).exitCode).toBe(0);
    // Lets any held read finish. Opening is refused (ENXIO) when no read is
    // left, and only a FIFO is opened.
    const releaseReads = () => {
      if (!lstatSync(file, { throwIfNoEntry: false })?.isFIFO()) return;
      try {
        closeSync(openSync(file, constants.O_WRONLY | constants.O_NONBLOCK));
      } catch (error) {
        const code = (error as NodeJS.ErrnoException).code;
        if (code !== "ENXIO" && code !== "ENOENT") throw error;
      }
    };
    let reads = 0;
    const connect = (controlTimeoutMs: number, signal?: AbortSignal) =>
      connectV2Host(
        "ses_src",
        async () => {
          reads++;
          return session;
        },
        {
          env: env(),
          pid: 7,
          controlTimeoutMs,
          ...(signal ? { signal } : {}),
        },
      );
    try {
      await refusal(connect(100), "timed-out");
      const activation = new AbortController();
      const cancelled = connect(60_000, activation.signal);
      // The read has started once the queued microtasks have run.
      await settleMicrotasks();
      activation.abort();
      await refusal(cancelled, "cancelled");
      expect(reads).toBe(0);
    } finally {
      releaseReads();
      await rm(file, { force: true });
    }
  });
});

describe("OpenCode 2 trim successors", () => {
  test("imports a verified successor with fresh IDs, current instructions, caches and stats", async () => {
    const state = fakeV2State();
    const history = toolHistory();
    // A special-token spelling in a user message is ordinary text.
    Object.assign(history[4]!, { text: `Read it ${SPECIAL_TOKEN}` });
    const source = sourceSession(state, "ses_src", history);
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

  test("reports inline files apart from the stored text and copies them unchanged", async () => {
    const pdf = randomBytes(6_000).toString("base64");
    const toolImage = `data:image/png;base64,${randomBytes(1_500).toString("base64")}`;
    const state = fakeV2State();
    sourceSession(state, "ses_src", [
      userMessage("msg_u1", "Read the report", {
        files: [
          { data: pdf, mime: "application/pdf", source: { type: "inline" } },
        ],
      }),
      assistantMessage("msg_a1", [
        toolItem("call_r", "read", { path: "/report.txt" }, [
          { type: "text", text: words("R", 400) },
          { type: "file", uri: toolImage, mime: "image/png" },
        ]),
      ]),
      idleMessage("msg_i1"),
    ]);
    const source = structuredClone(state.messages.get("ses_src")!);

    const { successor, outcome } = await executeV2Trim(
      fakeV2Host(state, new OwnedSessions()),
      new OwnedSessions(),
      "ses_src",
      0,
      { retention: { mode: "heuristic" }, controlTimeoutMs: 1_000 },
    );

    expect(state.messages.get("ses_src")).toEqual(source);
    const stored = state.messages.get(successor!.session.id)!;
    expect((stored[0] as Extract<V2Message, { type: "user" }>).files).toEqual(
      (source[0] as Extract<V2Message, { type: "user" }>).files,
    );
    expect(
      (v2ToolCalls(stored[1]!)[0]!.tool.state as { content: unknown[] })
        .content[1],
    ).toEqual({ type: "file", uri: toolImage, mime: "image/png" });
    const before = countV2ModelFacingTokens(source);
    const after = countV2ModelFacingTokens(stored);
    // Stats keep the canonical totals, file data included.
    expect(
      (await readStats(v2StorageKey(successor!.session.id)))?.totalTokensPruned,
    ).toBe(before.tokens - after.tokens);
    expect(outcome.split("\n").slice(0, 4)).toEqual([
      "Magic Trim",
      expect.stringMatching(
        /^Stored history \(estimate\) \| \S+ → \S+ text tokens, excluding inline file data \| \d+% trimmed \(/u,
      ),
      "Inline files | 2 file(s), 9.8 KiB encoded data in stored history before and after. Provider token cost for these files is not estimated; the text tokens above count stored text, not the model request.",
      "Changes | Tool calls trimmed: 1",
    ]);
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

  // This host returns its own stored session object on every read, so an
  // in-place edit also reaches the session the first read returned: only a
  // recheck that compares against its own copy of that read sees the change.
  const continuation: Array<[string, (session: SessionInfo) => void]> = [
    [
      "metadata",
      session => {
        (session.metadata as Record<string, unknown>)["seed"] = "changed";
      },
    ],
    [
      "model",
      session => {
        (session.model as { id: string }).id = "changed-model";
      },
    ],
    [
      "permission rules",
      session => {
        (session.permissions as Array<{ effect: string }>)[0]!.effect = "deny";
      },
    ],
  ];
  for (const [name, change] of continuation) {
    test(`discards an unready candidate when only the source's nested ${name} change before handoff`, async () => {
      const state = fakeV2State();
      const session = sourceSession(state, "ses_src", toolHistory());
      const history = structuredClone(state.messages.get("ses_src"));
      const host = fakeV2Host(state, new OwnedSessions());
      const get = host.session.get.bind(host.session);
      host.session.get = (async (input: { sessionID: string }) =>
        input.sessionID === "ses_src"
          ? session
          : get(input as never)) as typeof host.session.get;
      state.afterWrite = method => {
        if (method === "import") change(session);
      };

      const error = await trim(state, host);

      expect(String(error)).toContain(
        "The session changed while Magic Trim was preparing its successor.",
      );
      expect((error as SuccessorFailure).details).toMatchObject({
        stage: "recheck",
        cleanup: { outcome: "deleted" },
      });
      expect(state.calls.some(call => call.startsWith("update:"))).toBeFalse();
      expect([...state.sessions.keys()]).toEqual(["ses_src"]);
      expect(state.messages.get("ses_src")).toEqual(history);
    });
  }

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

  // The host may still apply an import it rejected, so a missing candidate
  // is named as not found rather than reported as removed.
  test("a failed import names its candidate as not found, not as removed", async () => {
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
    ).catch((caught: unknown) => caught);
    expect(error).toBeInstanceOf(SuccessorFailure);
    const { details } = error as SuccessorFailure;
    expect(details).toMatchObject({
      stage: "import",
      cleanup: { outcome: "absent", storageAttempted: false },
    });
    expect((error as Error).message).toBe(
      `Magic Trim could not finish preparing its successor session. The original session was left unchanged. Successor candidate ${details.candidateID} was not found when this run checked it, which does not prove that it was removed; check for it before continuing.`,
    );
    expect(state.calls).not.toContain(`remove:${details.candidateID}`);
    expect([...state.sessions.keys()]).toEqual(["ses_src"]);
  });

  const trim = (
    state: FakeV2State,
    host = fakeV2Host(state, new OwnedSessions()),
  ) =>
    executeV2Trim(host, new OwnedSessions(), "ses_src", 0, {
      retention: { mode: "heuristic" },
      controlTimeoutMs: 1_000,
    }).catch((caught: unknown) => caught);
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
      expect((error as Error).message).toBe(
        `The session changed while Magic Trim was preparing its successor. The session was left unchanged; retry when it is at rest. Successor candidate ${kept} was kept because it could not be confirmed as unmodified; this run did not mark it ready, so inspect it or delete it manually.`,
      );
      expect(state.calls).not.toContain(`remove:${kept}`);
      expect(state.calls).not.toContain(`update:${kept}`);
      expect(state.calls.filter(call => call.endsWith(":ses_src"))).toEqual([]);
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

  test("a candidate missing at cleanup is named, and its saved cache and stats stay", async () => {
    const state = fakeV2State();
    sourceSession(state, "ses_src", toolHistory());
    const host = fakeV2Host(state, new OwnedSessions());
    // Once the candidate's stats exist, the next activity read belongs to the
    // source recheck: another client removes the candidate and changes the
    // source's instructions, which the recheck reads next.
    let removed = false;
    const active = host.session.active.bind(host.session);
    host.session.active = async (...args: Parameters<typeof active>) => {
      const id = candidateID(state);
      if (
        !removed
        && id !== undefined
        && (await readStats(v2StorageKey(id))) !== null
      ) {
        removed = true;
        state.sessions.delete(id);
        state.messages.delete(id);
        state.instructions.set("ses_src", [{ key: "late", value: "x" }]);
      }
      return active(...args);
    };
    const source = structuredClone(state.messages.get("ses_src"));
    const error = await executeV2Trim(host, new OwnedSessions(), "ses_src", 0, {
      retention: { mode: "heuristic" },
      controlTimeoutMs: 1_000,
    }).catch((caught: unknown) => caught);
    expect(error).toBeInstanceOf(SuccessorFailure);
    const { details } = error as SuccessorFailure;
    expect(details).toMatchObject({
      stage: "recheck",
      cleanup: { outcome: "absent", storageAttempted: true },
    });
    // The drift sentence states no outcome; the candidate's own text does.
    expect((error as Error).message).toBe(
      `The session changed while Magic Trim was preparing its successor. The session was left unchanged; retry when it is at rest. Successor candidate ${details.candidateID} was not found when this run checked it, which does not prove that it was removed; check for it before continuing. Any omission cache or stats this run saved for it were left in place.`,
    );
    expect(await readStats(v2StorageKey(details.candidateID!))).not.toBeNull();
    expect(state.messages.get("ses_src")).toEqual(source);
    expect(state.calls.filter(call => call.endsWith(":ses_src"))).toEqual([]);
  });

  test("a candidate missing after its cache copy failed claims no saved files", async () => {
    const state = fakeV2State();
    sourceSession(state, "ses_src", toolHistory());
    state.instructions.set("ses_src", [{ key: "a", value: "ALPHA" }]);
    // An omission cache that cannot be read stops the copy before it writes.
    const sourceCache = cachePath(v2StorageKey("ses_src"));
    await mkdir(dirname(sourceCache), { recursive: true });
    await Bun.write(sourceCache, "{}\n");
    const host = fakeV2Host(state, new OwnedSessions());
    // Another client removes the candidate right after its last check before
    // the copy.
    const list = host.session.instructions.entry.list.bind(
      host.session.instructions.entry,
    );
    host.session.instructions.entry.list = async (
      ...args: Parameters<typeof list>
    ) => {
      const entries = await list(...args);
      const id = args[0].sessionID;
      if (id !== "ses_src") {
        state.sessions.delete(id);
        state.messages.delete(id);
      }
      return entries;
    };
    const error = await trim(state, host);
    expect(error).toBeInstanceOf(SuccessorFailure);
    const { details } = error as SuccessorFailure;
    expect(details).toMatchObject({
      stage: "cache",
      cleanup: { outcome: "absent", storageAttempted: true },
    });
    expect((error as Error).message).toBe(
      `Magic Trim could not finish preparing its successor session. The original session was left unchanged. Successor candidate ${details.candidateID} was not found when this run checked it, which does not prove that it was removed; check for it before continuing. Any omission cache or stats this run saved for it were left in place.`,
    );
    const key = v2StorageKey(details.candidateID!);
    expect(await Bun.file(cachePath(key)).exists()).toBeFalse();
    expect(await readStats(key)).toBeNull();
  });

  test("a deleted candidate whose stats cannot be removed is named with that residue", async () => {
    const state = fakeV2State();
    sourceSession(state, "ses_src", toolHistory());
    state.afterWrite = method => {
      if (method === "import")
        state.messages.get("ses_src")!.push(userMessage("msg_late", "late"));
    };
    // A directory where the candidate's stats file was makes its removal fail.
    state.beforeWrite = async (method, sessionID) => {
      if (method !== "remove") return;
      const file = `${storageDirectory}/opencode/storage/magic-compact/stats/v2/${sessionID}.json`;
      await rm(file, { force: true });
      await mkdir(join(file, "kept"), { recursive: true });
    };
    const error = await trim(state);
    expect(error).toBeInstanceOf(SuccessorFailure);
    const { details } = error as SuccessorFailure;
    expect(details).toMatchObject({
      stage: "recheck",
      cleanup: { outcome: "deleted-with-residue", residue: ["stats"] },
    });
    expect(state.calls).toContain(`remove:${details.candidateID}`);
    expect(String(error)).toContain(
      `Successor candidate ${details.candidateID} was deleted, but its stats could not be removed; delete the files named after ${details.candidateID} in Magic Compact's storage directory.`,
    );
    expect(String(error)).not.toContain(storageDirectory);
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
      "was kept because its deletion could not be confirmed",
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
      "Task-aware retention with jev (jev-1.13.0) at keep threshold 60% kept 1 of 2 evaluated tool call(s) whole (read 1).",
    );
    const read = v2ToolCalls(state.messages.get(successor!.session.id)![4]!)[0]!
      .tool;
    expect(
      (read.state as { content: Array<{ text?: string }> }).content[0]!.text,
    ).toBe(words("R", 400));
    expect([...state.sessions.keys()]).toHaveLength(2);
  });

  test("--backend openai-decisions judges every candidate there and names it", async () => {
    const state = fakeV2State();
    sourceSession(state, "ses_src", toolHistory());
    const owned = new OwnedSessions();
    const host = fakeV2Host(state, owned, () => ({
      text: "## Objective\n- Continue.",
    }));
    const calls: JevCall[] = [];
    const { outcome } = await executeV2Trim(host, owned, "ses_src", 0, {
      retention: {
        ...commandRetentionPolicy(
          "trim",
          parseMagicTrimArguments("--backend openai-decisions"),
          { MAGIC_COMPACT_OPENAI_DECISIONS_API_KEY: "test-openai-key" },
        ),
        fetch: scoredOpenAIFetch(
          output => (output.startsWith("R-0") ? 0.9 : 0.1),
          calls,
        ),
      },
      controlTimeoutMs: 1_000,
    });
    expect(calls).toHaveLength(2);
    expect(outcome).toContain(
      "Task-aware retention with openai-decisions (gpt-6-luna) at keep threshold 60% kept 1 of 2 evaluated tool call(s) whole (read 1).",
    );
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
        `Task-aware retention with jev (jev-1.13.0) at keep threshold ${percent ?? 60}% kept ${kept} of 2 evaluated tool call(s) whole`,
      );
    },
  );

  test("--threshold and --truncate reach the outcome with the actual truncated decisions", async () => {
    const state = fakeV2State();
    sourceSession(state, "ses_src", toolHistory());
    const owned = new OwnedSessions();
    const summary = `## Objective\n${"- A dense detail line.\n".repeat(1_500)}## Next Move`;
    const host = fakeV2Host(state, owned, () => ({ text: summary }));
    const summaries: string[] = [];
    const answer = scoredJevFetch(output =>
      output.startsWith("R-0") ? 0.75 : 0.1,
    );
    const { successor, outcome } = await executeV2Trim(
      host,
      owned,
      "ses_src",
      0,
      {
        retention: {
          ...commandRetentionPolicy(
            "trim",
            parseMagicTrimArguments("--truncate --threshold 70"),
            { TYPESAFE_API_KEY: "test-key" },
          ),
          fetch: async (url, init) => {
            const body = JSON.parse(String(init.body));
            summaries.push(body.state.task.sessionSummary);
            return body.state.task.sessionSummary === summary
              && body.state.candidates[0].output.startsWith("R-0")
              ? tokenLimitResponse()
              : answer(url, init);
          },
        },
        controlTimeoutMs: 1_000,
      },
    );
    expect(successor).not.toBeNull();
    expect(summaries).toHaveLength(3);
    expect(outcome).toContain(
      "Retention | Task-aware retention with jev (jev-1.13.0) at keep threshold 70% kept 1 of 2 evaluated tool call(s) whole (read 1). 1 decision(s) used a shortened session summary after a token-limit refusal (--truncate) and could have missed useful context.",
    );
    expect(
      (successor!.session.metadata as { magicCompact: { outcome: string } })
        .magicCompact.outcome,
    ).toBe(outcome);
  });

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
      text: `## Objective\n${"- A dense detail line.\n".repeat(1_500)}`,
    }));
    let jevCalls = 0;
    const error = await executeV2Trim(host, owned, "ses_src", 0, {
      retention: {
        ...commandRetentionPolicy(
          "trim",
          parseMagicTrimArguments("--threshold 60.1"),
          { TYPESAFE_API_KEY: "test-key" },
        ),
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
    // OpenCode 2 returns this text as the command error.
    const message = commandFailureMessage("Magic Trim", error);
    expect(message).toBe((error as Error).message);
    expect(message.length).toBeLessThan(700);
    expect(message).toContain(
      "no retention decisions were applied. Run /magic-trim --threshold 60.1 --truncate to retry",
    );
    expect(message).toContain("or /magic-trim --heuristic to trim");
    // Both candidates were admitted together; nothing was retried.
    expect(jevCalls).toBe(2);
    expect(state.messages.get("ses_src")).toEqual(before);
    expect([...state.sessions.keys()]).toEqual(["ses_src"]);
  });
});

describe("OpenCode 2 saved stats", () => {
  const CANARY = "PRIVATE-STATS-CANARY";
  const statsFile = () =>
    `${storageDirectory}/opencode/storage/magic-compact/stats/v2/ses_src.json`;
  const writeInvalidStats = async () => {
    const bytes = `{"version":2,"rootSessionId":"${CANARY}"}\n`;
    await mkdir(dirname(statsFile()), { recursive: true });
    await Bun.write(statsFile(), bytes);
    return bytes;
  };
  const expectContentFree = (text: string) => {
    expect(text).not.toContain(CANARY);
    expect(text).not.toContain(storageDirectory);
  };

  test("/magic-stats reports invalid and unreadable stats without their path or content", async () => {
    type Execute = (input: {
      sessionID: string;
      prompt: { text: string };
    }) => Promise<void>;
    const found = new Map<string, Execute>();
    const editor = {
      add: (command: { name: string; execute: Execute }) => {
        found.set(command.name, command.execute);
      },
      update: () => undefined,
    };
    await setup({
      options: {},
      agent: { transform: async (fn: (e: unknown) => void) => fn(editor) },
      command: { transform: async (fn: (e: unknown) => void) => fn(editor) },
      tool: { transform: async (fn: (e: unknown) => void) => fn(editor) },
      session: { hook: async () => undefined, get: async () => ({}) },
      event: { subscribe: async function* () {} },
    } as never);
    const run = (text: string) =>
      found.get("magic-stats")!({
        sessionID: "ses_src",
        prompt: { text },
      }).then(
        () => "",
        (error: unknown) => (error as Error).message,
      );

    expect(await run("now")).toBe(STATS_ARGUMENTS_ERROR);
    const bytes = await writeInvalidStats();
    const invalid = await run("");
    expect(invalid).toBe(new InvalidStatsError().message);
    expect(await Bun.file(statsFile()).text()).toBe(bytes);
    if (process.getuid?.() !== 0) {
      await chmod(statsFile(), 0o000);
      try {
        const unreadable = await run("");
        expect(unreadable).toBe(
          "Magic Stats stopped on an unexpected error (EACCES). The original session was left unchanged. Retry; if it fails again, check the OpenCode server log.",
        );
        expectContentFree(unreadable);
      } finally {
        await chmod(statsFile(), 0o600);
      }
    }
  });

  test("compaction stops before any model request, and a no-op still succeeds", async () => {
    const state = fakeV2State();
    const messages = toolHistory();
    sourceSession(state, "ses_src", messages);
    const owned = new OwnedSessions();
    const host = fakeV2Host(state, owned, echoGeneration());
    const options = {
      retention: { mode: "heuristic" as const },
      controlTimeoutMs: 1_000,
    };
    const bytes = await writeInvalidStats();

    const noop = await executeV2Compact(host, owned, "ses_src", 2, options);
    expect(noop.successor).toBeNull();
    const error = await executeV2Compact(
      host,
      owned,
      "ses_src",
      0,
      options,
    ).then(
      () => undefined,
      (failure: unknown) => failure,
    );
    expect(error).toBeInstanceOf(InvalidStatsError);
    expect(state.requests).toEqual([]);
    expect([...state.sessions.keys()]).toEqual(["ses_src"]);
    expect(state.messages.get("ses_src")).toEqual(messages);
    expect(await Bun.file(statsFile()).text()).toBe(bytes);

    // The same command makes its requests once the stats are moved away.
    await rm(statsFile());
    await executeV2Compact(host, owned, "ses_src", 0, options);
    expect(state.requests.length).toBeGreaterThan(0);
  });

  test("a successor's stats stage reports invalid source stats and discards the candidate", async () => {
    const state = fakeV2State();
    const messages = toolHistory();
    sourceSession(state, "ses_src", messages);
    const host = fakeV2Host(state, new OwnedSessions());
    const bytes = await writeInvalidStats();

    const error = await executeV2Trim(host, new OwnedSessions(), "ses_src", 0, {
      retention: { mode: "heuristic" },
      controlTimeoutMs: 1_000,
    }).then(
      () => undefined,
      (failure: unknown) => failure,
    );

    expect(error).toBeInstanceOf(SuccessorFailure);
    const message = commandFailureMessage("Magic Trim", error);
    expect(message).toBe(
      `${new InvalidStatsError().message}\nDetails: stage stats.`,
    );
    expectContentFree(message);
    expect([...state.sessions.keys()]).toEqual(["ses_src"]);
    expect(state.messages.get("ses_src")).toEqual(messages);
    expect(await Bun.file(statsFile()).text()).toBe(bytes);
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
    ).then(
      result => result.summaries,
      (error: Error) => error,
    );
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
    ).then(
      result => result.summaries,
      (error: Error) => error,
    );
    expect((outcome as Error).message).toBe(
      "Compaction generation failed for 2 turns: Compaction generation model request failed.",
    );
  });

  test("names every temporary session whose removal was not confirmed, in batch order, when five started batches fail", async () => {
    // Six turns, five concurrent batches. The third batch's import fails, so
    // the host no longer finds its session; every other request fails and its
    // removal is refused. The sixth turn is never admitted. The aggregate
    // explains each failure once and names all five sessions in batch order.
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
    const ids: string[] = [];
    state.beforeWrite = (method, sessionID) => {
      if (method === "import") {
        ids.push(sessionID);
        if (ids.length === 3) throw new Error("import failed");
      }
      if (method === "remove" && state.sessions.has(sessionID)) {
        throw new Error("connection reset");
      }
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
    ).then(
      result => result.summaries,
      (error: Error) => error,
    );

    expect(ids).toHaveLength(5);
    expect(new Set(ids).size).toBe(5);
    const [a, b, missing, d, e] = ids;
    expect(outcome).toBeInstanceOf(V2GenerationError);
    expect((outcome as V2GenerationError).unremovedSessions).toEqual(
      ids.map(id => ({
        id,
        removal: id === missing ? "not-found" : "failed",
      })),
    );
    expect((outcome as Error).message).toBe(
      `Compaction generation failed for 5 turns: Compaction generation model request failed. Compaction generation session creation failed. Temporary sessions ${a}, ${b} could not be deleted. Temporary session ${missing} was not found when this run tried to remove it, which does not prove that it was removed. Temporary sessions ${d}, ${e} could not be deleted.`,
    );
    for (const id of ids) {
      // Each stays registered and refuses new provider requests.
      expect(owned.has(id)).toBeTrue();
      expect(() => owned.admitRequest(id)).toThrow(owned.stop);
      expect(() => owned.rejectCompaction(id)).toThrow(owned.compactionBlocked);
    }
    // The source is unchanged and the sixth turn started nothing.
    expect(state.messages.get("ses_src")).toEqual(before);
    expect([...state.sessions.keys()]).toEqual(["ses_src", a, b, d, e]);
    expect(state.requests).toHaveLength(4);
  });

  test("stops admitting batches once a failure is seen, before its session is removed", async () => {
    // Six turns, five concurrent batches. The first turn's request fails and
    // its removal is held until the other four batches have finished and
    // been removed. Those workers are then free, but the sixth turn must not
    // start: the failure was already seen.
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
    const imports: string[] = [];
    let releaseOthers!: () => void;
    const othersHeld = new Promise<void>(resolve => {
      releaseOthers = resolve;
    });
    let releaseRemoval!: () => void;
    const removalHeld = new Promise<void>(resolve => {
      releaseRemoval = resolve;
    });
    let othersRemoved = 0;
    state.beforeWrite = async (method, sessionID) => {
      if (method === "import") {
        imports.push(sessionID);
        if (sessionID !== imports[0]) await othersHeld;
      }
      if (method === "remove" && sessionID === imports[0]) {
        releaseOthers();
        await removalHeld;
      }
    };
    state.afterWrite = (method, sessionID) => {
      if (method === "remove" && sessionID !== imports[0]) {
        othersRemoved += 1;
        if (othersRemoved === 4) releaseRemoval();
      }
    };
    const echo = echoGeneration();
    const owned = new OwnedSessions();
    const outcome = await generateV2Summaries(
      fakeV2Host(state, owned, (text, sent) =>
        sent.some(
          message => message.type === "user" && message.text === "Do it",
        )
          ? { error: { type: "provider.invalid-request", message: "bad" } }
          : echo(text),
      ),
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
    ).then(
      result => result.summaries,
      (error: Error) => error,
    );

    expect(outcome).toBeInstanceOf(V2GenerationError);
    expect((outcome as Error).message).toBe(
      "Compaction generation model request failed.",
    );
    expect(imports).toHaveLength(5);
    expect(state.requests).toHaveLength(5);
    expect([...state.sessions.keys()]).toEqual(["ses_src"]);
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
    ).then(
      result => result.summaries,
      (error: Error) => error,
    );
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
    expect((outcome as Error).message).toBe(
      `Compaction generation session deletion failed. Temporary session ${ownedID} could not be deleted.`,
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

  test("names a failed import's session as not found and keeps its guard", async () => {
    const { state, owned, outcome } = await generate(state => {
      state.beforeWrite = (method, sessionID) => {
        if (method === "import") {
          state.calls.push(`import:${sessionID}`);
          throw new Error("import failed");
        }
      };
    });
    const ownedID = state.calls[0]!.slice("import:".length);
    // The host has no such session now, which does not prove it never will.
    expect((outcome as Error).message).toBe(
      `Compaction generation session creation failed. Temporary session ${ownedID} was not found when this run tried to remove it, which does not prove that it was removed.`,
    );
    expect((outcome as V2GenerationError).unremovedSessions).toEqual([
      { id: ownedID, removal: "not-found" },
    ]);
    expect([...state.sessions.keys()]).toEqual(["ses_src"]);
    // The failed import could still be applied later; its ID stays stopped,
    // and the host's answer that no such session exists keeps it so.
    expect(owned.has(ownedID)).toBeTrue();
    expect(() => owned.admitRequest(ownedID)).toThrow(owned.stop);
    expect(state.requests).toHaveLength(0);
  });

  test("keeps a validated response when the host no longer finds its session, and keeps its guard", async () => {
    const { state, owned, outcome, ownedID } = await generate(state => {
      state.beforeWrite = (method, sessionID) => {
        if (method === "remove") state.sessions.delete(sessionID);
      };
    });
    expect(outcome).toEqual([{ text: "Summary of target 1." }]);
    expect(state.requests).toHaveLength(1);
    expect(owned.has(ownedID!)).toBeTrue();
    expect(() => owned.admitRequest(ownedID!)).toThrow(owned.stop);
    expect(() => owned.rejectCompaction(ownedID!)).toThrow(
      owned.compactionBlocked,
    );
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
      expect(String(outcome)).toEndWith(
        ` Temporary session ${ownedID} could not be deleted.`,
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
      false,
    ).then(
      summary => summary.text,
      (error: Error) => error,
    );
    const ownedID = state.calls
      .find(entry => entry.startsWith("import:"))!
      .slice("import:".length);
    return { state, owned, outcome, ownedID };
  };

  // --match-model runs a turn summary on the model that wrote its target's
  // protected reasoning, with the configured variant, and the imported
  // session names that model; without the flag, or when the model is too
  // small, the configured summarizer runs.
  test("runs a turn summary on its target's reasoning model only under --match-model", async () => {
    const sol = {
      id: "gpt-6.1-sol",
      modelID: "gpt-6.1-sol",
      providerID: "openai",
      package: "@opencode/ai/providers/openai",
      name: "Sol",
      capabilities: { tools: true, input: ["text"], output: ["text"] },
      variants: [{ id: "summary" }],
      time: { released: 0 },
      cost: [],
      status: "active",
      enabled: true,
      limit: { context: 200_000, output: 32_000 },
    };
    const turnOf = (modelID: string) => [
      userMessage("msg_p", "Do it"),
      assistantMessage(
        "msg_pa",
        [
          {
            type: "reasoning",
            text: "visible summary",
            state: { itemId: "rs_1", reasoningEncryptedContent: "enc_secret" },
          },
          { type: "text", text: words("A", 200) },
        ],
        { model: { providerID: "openai", id: modelID } },
      ),
      idleMessage("msg_pi"),
    ];
    const protectedTurn = turnOf("gpt-6.1-sol");
    const run = async (
      matchModel: boolean,
      limit = sol.limit,
      listed: Record<string, unknown> = sol,
      messages = protectedTurn,
    ) => {
      const state = fakeV2State();
      const source = sourceSession(state, "ses_src", messages);
      const owned = new OwnedSessions();
      const imported: unknown[] = [];
      state.afterWrite = (method, sessionID) => {
        if (method === "import")
          imported.push(state.sessions.get(sessionID)!.model);
      };
      const { summaries: outcome, ...reasoning } = await generateV2Summaries(
        fakeV2Host(state, owned, echoGeneration()),
        owned,
        { session: source, messages, instructions: [] },
        {
          ref: state.summarizerModel!,
          info: {
            limit: state.modelLimit,
            capabilities: { tools: true, input: ["text"], output: ["text"] },
          },
          others: [{ ...listed, limit }] as never,
        },
        [{ span: messages }],
        "index",
        { controlTimeoutMs: 1_000, matchModel },
      );
      expect(outcome).toEqual([{ text: "Summary of target 1." }]);
      expect(JSON.stringify(state.requests)).not.toContain("enc_secret");
      return { imported, reasoning };
    };
    const configured = {
      providerID: "probe",
      id: "probe-model",
      variant: "summary",
    };
    expect((await run(false)).imported).toEqual([configured]);
    const matched = await run(true);
    expect(matched.imported).toEqual([
      { providerID: "openai", id: "gpt-6.1-sol", variant: "summary" },
    ]);
    expect(matched.reasoning.reports[0]).toMatchObject({
      access: { compatible: 1, unavailable: 0, unknown: 0 },
      onOrigin: true,
      // OpenCode 2 never estimates replay: the matched model's own unit is
      // left out of its input estimate and reported.
      replayUnknown: 1,
    });
    // The model list gives the fast entry its own `id` and the base
    // `modelID`; a turn it wrote matches that exact entry.
    const fast = { ...sol, id: "gpt-6.1-sol-fast" };
    const fastMatched = await run(
      true,
      sol.limit,
      fast,
      turnOf("gpt-6.1-sol-fast"),
    );
    expect(fastMatched.imported).toEqual([
      { providerID: "openai", id: "gpt-6.1-sol-fast", variant: "summary" },
    ]);
    expect(fastMatched.reasoning.reports[0]).toMatchObject({
      access: { compatible: 1, unavailable: 0, unknown: 0 },
      onOrigin: true,
    });
    // A model the host lists without the verified serializer, or another
    // entry of the verified wire model, is not matched.
    for (const others of [
      { ...sol, package: "@opencode/ai/providers/openai-compatible" },
      { ...sol, modelID: "gpt-6.1-sol-2026" },
      { ...sol, id: "gpt-6.1-sol-pro" },
      { ...sol, id: "gpt-6.1-sol-ultrafast" },
      { ...fast, modelID: "gpt-6.1-sol-fast" },
      { ...fast, package: "@opencode/ai/providers/openai-compatible" },
    ]) {
      // The turn was written by the listed entry itself.
      const messages = turnOf(others.id);
      const state = fakeV2State();
      const source = sourceSession(state, "ses_src", messages);
      const owned = new OwnedSessions();
      const reasoning = await generateV2Summaries(
        fakeV2Host(state, owned, echoGeneration()),
        owned,
        { session: source, messages, instructions: [] },
        {
          ref: state.summarizerModel!,
          info: {
            limit: state.modelLimit,
            capabilities: { tools: true, input: ["text"], output: ["text"] },
          },
          others: [others] as never,
        },
        [{ span: messages }],
        "index",
        { controlTimeoutMs: 1_000, matchModel: true },
      );
      expect(reasoning.reports[0]!.choice).toEqual({
        kind: "fallback",
        reason: "unqualified",
      });
    }
    const small = await run(true, { context: 9_000, output: 1_000 });
    expect(small.imported).toEqual([configured]);
    expect(small.reasoning.reports[0]!.choice).toEqual({
      kind: "fallback",
      reason: "capacity",
    });
  });

  // On the DeepSeek route the session summary goes through the summary tool
  // with the summary schema, because DeepSeek reads earlier plain reasoning
  // only when a request has tools. Only one completed call with exactly one
  // nonempty summary is accepted, and the owned session is always removed.
  test("submits the DeepSeek-route session summary through the summary tool and validates it strictly", async () => {
    const reasoned = [
      userMessage("msg_r", "Do it"),
      assistantMessage("msg_ra", [
        {
          type: "reasoning",
          text: "plain reasoning to read",
          state: { reasoningField: "reasoning_content" },
        },
        { type: "text", text: "Done." },
      ]),
      idleMessage("msg_ri"),
    ];
    const run = async (reply: FakeReply, history = reasoned) => {
      const state = fakeV2State();
      state.summarizerModel = { providerID: "deepseek", id: "deepseek-flash" };
      const source = sourceSession(state, "ses_src", history);
      const owned = new OwnedSessions();
      let permissions: unknown;
      let allowed = false;
      state.afterWrite = (method, sessionID) => {
        if (method !== "import") return;
        permissions = state.sessions.get(sessionID)!.permissions;
        allowed = owned.allowsSummaryTool(sessionID);
      };
      const outcome = await generateV2SessionSummary(
        fakeV2Host(state, owned, () => reply),
        owned,
        { session: source, messages: history, instructions: [] },
        {
          ref: state.summarizerModel,
          info: {
            limit: state.modelLimit,
            capabilities: { tools: true, input: ["text"], output: ["text"] },
            compatibility: { reasoningField: "reasoning_content" },
          },
        },
        1_000,
        false,
      ).then(
        summary => summary.text,
        (error: Error) => error,
      );
      expect([...state.sessions.keys()]).toEqual(["ses_src"]);
      return { state, outcome, permissions, allowed };
    };
    const submit = (input: unknown, status?: "error"): FakeReply => ({
      tool: [{ name: "SessionSummary", input, ...(status ? { status } : {}) }],
    });

    const accepted = await run(submit({ summary: " ## Goal\n- x " }));
    expect(accepted.outcome).toBe("## Goal\n- x");
    expect(accepted.allowed).toBeTrue();
    expect(accepted.permissions).toEqual([
      { action: "*", resource: "*", effect: "deny" },
      { action: "SessionSummary", resource: "*", effect: "allow" },
    ]);
    expect(accepted.state.requests[0]!.text).toBe(SESSION_SUMMARY_TOOL_PROMPT);

    for (const reply of [
      submit({ summary: "" }),
      submit({ summary: "x", extra: true }),
      submit({ summary: "x" }, "error"),
      { tool: [] },
      { tool: [{ name: "StructuredOutput", input: { summary: "x" } }] },
      {
        tool: [
          { name: "SessionSummary", input: { summary: "x" } },
          { name: "SessionSummary", input: { summary: "y" } },
        ],
      },
      { text: "a text reply" },
      { error: { type: "refusal", message: "declined" } },
    ] as FakeReply[]) {
      const rejected = await run(reply);
      expect(rejected.outcome).toBeInstanceOf(SessionSummaryError);
      expect(reachableStrings(rejected.outcome).join(" ")).not.toContain(
        "declined",
      );
    }

    // Without plain reasoning to read, the route keeps the text request.
    const plain = await run({ text: "## Objective\n- Continue." }, target);
    expect(plain.outcome).toBe("## Objective\n- Continue.");
    expect(plain.allowed).toBeFalse();
    expect(plain.state.requests[0]!.text).toBe(SESSION_SUMMARY_PROMPT);
  });

  test("the retention summary keeps stopping an owned session whose removal failed", async () => {
    const { state, owned, outcome, ownedID } = await summarizeSession(state => {
      state.beforeWrite = method => {
        if (method === "remove") throw new Error("connection reset");
      };
    });
    expect(String(outcome)).toContain(
      `summary failed (delete). Temporary session ${ownedID} could not be deleted. No tool content was changed.`,
    );
    expect((outcome as SessionSummaryError).unremovedSessions).toEqual([
      { id: ownedID, removal: "failed" },
    ]);
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
    expect(String(outcome)).toContain(
      `summary and cleanup failed (fork, delete). Temporary session ${ownedID} could not be deleted.`,
    );
    expect(state.requests).toHaveLength(0);
    expect(state.sessions.has(ownedID)).toBeTrue();
    expect(() => owned.admitRequest(ownedID)).toThrow(owned.stop);
  });

  test("the retention summary names a failed import's session as not found and keeps its guard", async () => {
    const { state, owned, outcome, ownedID } = await summarizeSession(state => {
      state.beforeWrite = (method, sessionID) => {
        if (method === "import") {
          state.calls.push(`import:${sessionID}`);
          throw new Error("import failed");
        }
      };
    });
    expect((outcome as Error).message).toBe(
      `Task-aware retention session summary and cleanup failed (fork, delete). Temporary session ${ownedID} was not found when this run tried to remove it, which does not prove that it was removed. ${retentionFailureGuidance()}`,
    );
    expect((outcome as SessionSummaryError).unremovedSessions).toEqual([
      { id: ownedID, removal: "not-found" },
    ]);
    expect([...state.sessions.keys()]).toEqual(["ses_src"]);
    expect(owned.has(ownedID)).toBeTrue();
    expect(() => owned.admitRequest(ownedID)).toThrow(owned.stop);
    expect(state.requests).toHaveLength(0);
  });

  test("the retention summary keeps a validated summary when the host no longer finds its session, and keeps its guard", async () => {
    const { state, owned, outcome, ownedID } = await summarizeSession(state => {
      state.beforeWrite = (method, sessionID) => {
        if (method === "remove") state.sessions.delete(sessionID);
      };
    });
    expect(outcome).toBe("## Objective\n- Continue.");
    expect(state.requests).toHaveLength(1);
    expect(owned.has(ownedID)).toBeTrue();
    expect(() => owned.admitRequest(ownedID)).toThrow(owned.stop);
    expect(() => owned.rejectCompaction(ownedID)).toThrow(
      owned.compactionBlocked,
    );
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
    // Special-token spellings in a summarized turn and in a kept user message
    // are ordinary text.
    Object.assign(messages[0]!, { text: `Request 1 ${SPECIAL_TOKEN}` });
    Object.assign(messages[6]!, { text: `Request 3 <|im_end|><|im_sep|>` });
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

  // The summarizer is resolved when the retention summary first needs it, so
  // its refusal is a retention failure and suggests the person's own command.
  test.each([
    ["compact", "Magic Compact"],
    ["trim", "Magic Trim"],
  ] as const)(
    "task-aware /magic-%s names a missing summarizer model with its own heuristic command",
    async (command, title) => {
      const state = fakeV2State();
      sourceSession(
        state,
        "ses_src",
        command === "compact"
          ? [...turn(1), ...turn(2), ...turn(3)]
          : toolHistory(),
      );
      delete state.summarizerModel;
      const owned = new OwnedSessions();
      const host = fakeV2Host(state, owned);
      const options = {
        retention: {
          ...commandRetentionPolicy(
            command,
            (command === "compact"
              ? parseMagicCompactArguments
              : parseMagicTrimArguments)("1 --match-model"),
            { TYPESAFE_API_KEY: "test-key" },
          ),
          fetch: async () => {
            throw new Error("unexpected retention request");
          },
        },
        controlTimeoutMs: 1_000,
      };
      const error = await (
        command === "compact"
          ? executeV2Compact(host, owned, "ses_src", 1, options)
          : executeV2Trim(host, owned, "ses_src", 1, options)
      ).then(
        () => undefined,
        (failure: unknown) => failure,
      );

      expect(error).toBeInstanceOf(SessionSummaryError);
      expect(commandFailureMessage(title, error)).toBe(
        `Magic Compact needs a model on the \`magic-compact-summarizer\` agent. Set \`agents.magic-compact-summarizer.model\` (for example \`provider/model#variant\`) in opencode.json. ${
          command === "compact"
            ? "No tool content was changed. Run /magic-compact 1 --match-model --heuristic to compact without classification (turn summaries are still generated)."
            : "No tool content was changed. Run /magic-trim 1 --heuristic to trim without classification."
        }`,
      );
      expect(state.requests).toHaveLength(0);
      expect([...state.sessions.keys()]).toEqual(["ses_src"]);
    },
  );

  describe("task-aware retention", () => {
    const longSummary = `## Objective\n${"- A dense detail line.\n".repeat(1_500)}## Next Move`;
    // Answers the generation prompt per turn and the retention summary with
    // `summary`.
    const withSummary =
      (summary: string) =>
      (text: string): FakeReply => {
        const reply = echoGeneration()(text);
        return "tool" in reply ? reply : { text: summary };
      };
    // The policy a person's `/magic-compact <args>` builds on the server.
    const policyFor = (args: string, fetch: RetentionFetch) => ({
      ...commandRetentionPolicy("compact", parseMagicCompactArguments(args), {
        TYPESAFE_API_KEY: "test-key",
      }),
      fetch,
    });
    const compact = (
      reply: (text: string) => FakeReply,
      keepTurns: number,
      retention: Parameters<typeof executeV2Compact>[4]["retention"],
      messages = [...turn(1), ...turn(2), ...turn(3)],
    ) => {
      const state = fakeV2State();
      sourceSession(state, "ses_src", messages);
      const owned = new OwnedSessions();
      const result = executeV2Compact(
        fakeV2Host(state, owned, reply),
        owned,
        "ses_src",
        keepTurns,
        { retention, controlTimeoutMs: 1_000 },
      );
      return { state, result };
    };

    test("keeps a judged-useful call whole and reports mode, threshold, counts and kept tools", async () => {
      const { state, result } = compact(
        echoGeneration(),
        1,
        policyFor(
          "1 --threshold 70",
          scoredJevFetch(output => (output.startsWith("R1-") ? 0.8 : 0.1)),
        ),
      );
      const { successor, outcome } = await result;
      const line =
        "Task-aware retention with jev (jev-1.13.0) at keep threshold 70% kept 1 of 2 evaluated tool call(s) whole (read 1).";
      expect(outcome.split("\n").at(-1)).toBe(`Retention | ${line}`);
      expect(outcome.match(/Retention \|/gu)).toHaveLength(1);
      expect(
        (successor!.session.metadata as { magicCompact: { outcome: string } })
          .magicCompact.outcome,
      ).toBe(outcome);
      // One retention summary, then one request per summarized turn.
      expect(state.requests).toHaveLength(3);
      const stored = state.messages.get(successor!.session.id)!;
      const calls = stored.flatMap(message => v2ToolCalls(message));
      expect(
        (calls[0]!.tool.state as { content: Array<{ text?: string }> })
          .content[0]!.text,
      ).toBe(words("R1", 300));
      expect(JSON.stringify(calls[1]!.tool)).toContain("omitted");
    });

    test("--heuristic skips only the retention summary and Jev, and the turns are still summarized", async () => {
      const { state, result } = compact(
        echoGeneration(),
        1,
        commandRetentionPolicy(
          "compact",
          parseMagicCompactArguments("1 --heuristic"),
          {
            get TYPESAFE_API_KEY(): string {
              throw new Error("the key must not be read");
            },
            get [RETENTION_KEEP_THRESHOLD_PERCENT_ENV](): string {
              throw new Error("the percent must not be read");
            },
          },
        ),
      );
      const { successor, outcome } = await result;
      expect(successor).not.toBeNull();
      // Two summarized turns, no retention summary.
      expect(state.requests).toHaveLength(2);
      expect(outcome.split("\n").at(-1)).toBe(
        "Retention | Heuristic mode: tool calls not evaluated.",
      );
      expect(outcome).toContain("Request turns summarized: 2");
    });

    test("--truncate retries only the confirmed refusal's summary and reports the actual decisions", async () => {
      const summaries: string[] = [];
      const answer = scoredJevFetch(output =>
        output.startsWith("R1-") ? 0.75 : 0.1,
      );
      const { state, result } = compact(
        withSummary(longSummary),
        1,
        policyFor("1 --truncate --threshold 70", async (url, init) => {
          const body = JSON.parse(String(init.body));
          summaries.push(body.state.task.sessionSummary);
          return body.state.task.sessionSummary === longSummary
            && body.state.candidates[0].output.startsWith("R1-")
            ? tokenLimitResponse()
            : answer(url, init);
        }),
      );
      const { successor, outcome } = await result;
      expect(successor).not.toBeNull();
      expect(summaries.filter(text => text === longSummary)).toHaveLength(2);
      expect(summaries.filter(text => text !== longSummary)).toEqual([
        excerpt(longSummary, RETENTION_SUMMARY_EXCERPT_BYTES),
      ]);
      // No turn request carries the shortened summary or its marker.
      expect(JSON.stringify(state.requests)).not.toContain(
        "characters omitted",
      );
      expect(outcome.split("\n").at(-1)).toBe(
        "Retention | Task-aware retention with jev (jev-1.13.0) at keep threshold 70% kept 1 of 2 evaluated tool call(s) whole (read 1). 1 decision(s) used a shortened session summary after a token-limit refusal (--truncate) and could have missed useful context.",
      );
    });

    test("a token-limit refusal without --truncate fails once with the person's own retry and no successor", async () => {
      let calls = 0;
      const { state, result } = compact(
        withSummary(longSummary),
        1,
        policyFor("1 --threshold 60.1", async () => {
          calls += 1;
          return tokenLimitResponse();
        }),
      );
      const error = await result.then(
        () => undefined,
        (caught: unknown) => caught,
      );
      expect(error).toBeInstanceOf(RetentionClassificationError);
      // OpenCode 2 returns this text as the command error.
      const message = commandFailureMessage("Magic Compact", error);
      expect(message).toBe((error as Error).message);
      expect(message.length).toBeLessThan(700);
      expect(message).toContain(
        "no retention decisions were applied. Run /magic-compact 1 --threshold 60.1 --truncate to retry",
      );
      expect(message).toContain(
        "or /magic-compact 1 --heuristic to compact without task-aware keep protection (turn summaries are still generated)",
      );
      expect(message).not.toContain("/magic-trim");
      // Both candidates were admitted together; nothing was retried.
      expect(calls).toBe(2);
      // Only the retention summary was requested: no turn generation.
      expect(state.requests).toHaveLength(1);
      expect([...state.sessions.keys()]).toEqual(["ses_src"]);
    });

    test.each([
      [
        "--threshold 70",
        "Task-aware retention with jev (jev-1.13.0) at keep threshold 70%: no tool calls needed evaluation.",
      ],
      [
        "--backend openai-decisions",
        "Task-aware retention with openai-decisions (gpt-6-luna) at keep threshold 60%: no tool calls needed evaluation.",
      ],
      ["--heuristic", "Heuristic mode: tool calls not evaluated."],
    ])(
      "a no-op under %p reports its retention mode without a request",
      async (args, line) => {
        let fetches = 0;
        const preserving = compact(
          echoGeneration(),
          1,
          {
            ...commandRetentionPolicy(
              "compact",
              parseMagicCompactArguments(args),
              { TYPESAFE_API_KEY: "test-key" },
            ),
            fetch: async () => {
              fetches += 1;
              return tokenLimitResponse();
            },
          },
          turn(1),
        );
        const { successor, outcome } = await preserving.result;
        expect(successor).toBeNull();
        expect(outcome).toBe(
          `${COMPACT_NOOP} ${preservedTurnsFeedback({ requested: 1, preserved: 1 })} ${line}`,
        );
        expect(preserving.state.requests).toHaveLength(0);
        expect(fetches).toBe(0);
      },
    );
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
    // Both images stay stored, and stats keep the canonical totals with the
    // data included.
    const original = countV2ModelFacingTokens(source);
    expect(stats?.totalTokensPruned).toBe(
      original.tokens
        + original.projectedMediaTokens
        - (facing.tokens + facing.projectedMediaTokens),
    );
    expect(outcome.split("\n").slice(1, 4)).toEqual([
      expect.stringMatching(
        /^Stored history \(estimate\) \| \S+ → \S+ text tokens, excluding inline file data \| \d+% trimmed \(/u,
      ),
      "Inline files | 2 file(s), 3.9 KiB encoded data in stored history before and after. Provider token cost for these files is not estimated; the text tokens above count stored text, not the model request.",
      "Changes | Request turns summarized: 1",
    ]);

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

  // User file data of any MIME and `data:` tool file URIs are inline file
  // data; the canonical total still counts them, a projected image counts as
  // its placeholder, and every stored inline file stays in the inventory.
  test("partitions inline file data out of an unchanged stored-history count", () => {
    const pdf = randomBytes(3_000).toString("base64");
    const notes = Buffer.from("meeting notes").toString("base64");
    const toolImage = `data:image/png;base64,${smallImage}`;
    const remote = "file:///tmp/out.png";
    const input = { path: "/out.png" };
    const records = [
      userMessage("msg_u1", "Read these", {
        files: [
          imageFile(largeImage),
          { data: pdf, mime: "application/pdf", source: { type: "inline" } },
          {
            data: notes,
            mime: "text/plain",
            source: { type: "uri", uri: "file:///notes.txt" },
          },
          {
            data: "",
            mime: "application/x-directory",
            source: { type: "uri", uri: "file:///src" },
          },
        ],
      }),
      assistantMessage(
        "msg_a1",
        [
          { type: "text", text: "Summary." },
          toolItem("call_r", "read", input, [
            { type: "text", text: "ok" },
            { type: "file", uri: toolImage, mime: "image/png" },
            { type: "file", uri: remote, mime: "image/png" },
          ]),
        ],
        {
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
                    dataSha256: createHash("sha256")
                      .update(largeImage)
                      .digest("hex"),
                  },
                ],
              },
            },
          },
        },
      ),
    ];
    const placeholder = mediaPlaceholderText(1, "image/png");
    const payloads = (values: string[]) =>
      values.reduce((total, value) => total + countV2PayloadTokens(value), 0);
    const stored =
      countGptTokens("Read these")
      + payloads([largeImage, pdf, notes])
      + countGptTokens("Summary.")
      + countGptTokens(JSON.stringify(input))
      + countGptTokens("ok")
      + payloads([toolImage, remote]);
    const projectedMedia =
      countV2PayloadTokens(largeImage) - countGptTokens(placeholder);

    expect(countV2ModelFacingTokens(records)).toEqual({
      tokens: stored - projectedMedia,
      projectedMediaTokens: projectedMedia,
      inlineFileTokens: payloads([pdf, notes, toolImage]),
      inlineFiles: {
        count: 4,
        encodedBytes: [largeImage, pdf, notes, toolImage].reduce(
          (total, value) => total + value.length,
          0,
        ),
      },
    });
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
        // Each session read returns the session it names.
        return Response.json({
          data: { ...session, id: url.pathname.split("/").at(-1) },
        });
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
            get: async ({ sessionID }: { sessionID: string }) => ({
              ...session,
              id: sessionID,
            }),
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
      // Each complete-history read qualifies the endpoint again.
      expect(reads.filter(path => path === "/api/info")).toHaveLength(2);

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
      owned.register("ses_owned_a", "retention", false);
      owned.register("ses_owned_b", "retention", false);
      const projected = {
        sessionID: "ses_owned_a",
        model: { providerID: "test", id: "test-model" },
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

  // The plugin over a loopback server holding `checkpointHistory`, with the
  // cleanup `setup` returns and a record of in-process reads. `holdContext`
  // keeps the in-process context read open until released, ignoring its
  // signal like a read that does not honour it, and reports when it starts.
  const mediaPlugin = async (
    respond: (url: URL) => Response | Promise<Response> | "hold" | undefined,
  ) => {
    const history = checkpointHistory();
    const server = loopback(url => {
      const held = respond(url);
      if (held !== undefined) return held;
      if (url.pathname.endsWith("/message"))
        return Response.json({ data: history.records, cursor: { next: null } });
      if (url.pathname.endsWith("/export"))
        return Response.json({ data: { messages: history.records } });
      return Response.json({
        data: lineageSession(url.pathname.split("/").at(-1)!),
      });
    });
    const hooks = new Map<string, (event: never) => Promise<void>>();
    const commands = new Map<string, (input: unknown) => Promise<void>>();
    const inProcess: string[] = [];
    let contextGate: Promise<void> | undefined;
    let contextEntered: (() => void) | undefined;
    let afterContext: (() => void) | undefined;
    const editor = { add: () => undefined, update: () => undefined };
    const cleanup = await setup({
      options: { serverUrl: server.url },
      agent: { transform: async (fn: (e: unknown) => void) => fn(editor) },
      command: {
        transform: async (fn: (e: unknown) => void) =>
          fn({
            add: (command: {
              name: string;
              execute: (input: unknown) => Promise<void>;
            }) => commands.set(command.name, command.execute),
          }),
      },
      tool: { transform: async (fn: (e: unknown) => void) => fn(editor) },
      session: {
        hook: async (
          name: string,
          callback: (event: never) => Promise<void>,
        ) => {
          hooks.set(name, callback);
        },
        context: ({ sessionID }: { sessionID: string }) => {
          inProcess.push(`context ${sessionID}`);
          contextEntered?.();
          if (afterContext) {
            // Runs `after` once the read has settled, before its caller
            // resumes: the caller's await takes more than one job.
            const read = Promise.resolve(structuredClone(history.context));
            const after = afterContext;
            queueMicrotask(() => void read.then(after));
            return read;
          }
          return (contextGate ?? Promise.resolve()).then(() =>
            structuredClone(history.context),
          );
        },
        get: async ({ sessionID }: { sessionID: string }) => {
          inProcess.push(`get ${sessionID}`);
          return lineageSession(sessionID);
        },
      },
      event: { subscribe: async function* () {} },
    } as never);
    const run = (name: string, sessionID: string, messages: unknown[]) => {
      const event = { sessionID, tools: {}, messages };
      const before = JSON.stringify(messages);
      const settled = hooks.get(name)!(event as never).then(
        () => undefined,
        (error: Error) => error.message,
      );
      return {
        event,
        unchanged: () => JSON.stringify(event.messages) === before,
        settled,
      };
    };
    return {
      history,
      server,
      cleanup,
      commands,
      inProcess,
      run,
      holdContext: () => {
        const gate = Promise.withResolvers<void>();
        const entered = Promise.withResolvers<void>();
        contextGate = gate.promise;
        contextEntered = entered.resolve;
        return { release: gate.resolve, entered: entered.promise };
      },
      cleanupAfterContextRead: () => {
        afterContext = cleanup;
      },
    };
  };
  const lineageSession = (id: string) => ({
    id,
    projectID: "p",
    time: { created: 1 },
    location: LOCATION,
    metadata: { magicCompact: {} },
  });

  test("each complete-history read qualifies the endpoint for its own session", async () => {
    const plugin = await mediaPlugin(url =>
      url.pathname === "/api/info" ? serverInfo(process.pid) : undefined,
    );
    try {
      for (const sessionID of ["ses_a", "ses_b", "ses_a"]) {
        const { event, settled } = plugin.run(
          "context",
          sessionID,
          plugin.history.incompatible,
        );
        expect(await settled).toBeUndefined();
        expect(JSON.stringify(event.messages)).not.toContain(
          plugin.history.older,
        );
      }
      const paths = plugin.server.seen.map(seen => seen.path);
      expect(paths.filter(path => path === "/api/info")).toHaveLength(3);
      expect(
        paths.filter(path => /^\/api\/session\/[^/]+$/u.test(path)),
      ).toEqual([
        "/api/session/ses_a",
        "/api/session/ses_b",
        "/api/session/ses_a",
      ]);
      expect(plugin.inProcess.filter(read => read.startsWith("get"))).toEqual([
        "get ses_a",
        "get ses_b",
        "get ses_a",
      ]);
    } finally {
      plugin.cleanup();
      await plugin.server.stop();
    }
  });

  test.each(["/api/info", "/message"])(
    "cleanup while %s is open aborts it and leaves the request unprojected",
    async held => {
      const plugin = await mediaPlugin(url =>
        url.pathname.endsWith(held)
          ? "hold"
          : url.pathname === "/api/info"
            ? serverInfo(process.pid)
            : undefined,
      );
      try {
        const pending = plugin.run(
          "generate",
          "ses_a",
          plugin.history.incompatible,
        );
        await plugin.server.waitFor(() =>
          plugin.server.seen.some(seen => seen.path.endsWith(held)),
        );
        plugin.cleanup();
        expect(await pending.settled).toContain("the plugin was unloaded");
        expect(pending.unchanged()).toBe(true);
        await plugin.server.waitFor(() => plugin.server.aborted.length === 1);
        await settleMicrotasks();
        // Nothing follows the aborted read.
        expect(plugin.server.seen.at(-1)!.path).toEndWith(held);
      } finally {
        await plugin.server.stop();
      }
    },
  );

  test("after cleanup a late in-process read projects nothing and no new media read starts", async () => {
    const plugin = await mediaPlugin(url =>
      url.pathname === "/api/info" ? serverInfo(process.pid) : undefined,
    );
    try {
      const { release, entered } = plugin.holdContext();
      const late = plugin.run("context", "ses_a", plugin.history.incompatible);
      await entered;
      plugin.cleanup();
      // The hook settles while the in-process read is still held.
      expect(await late.settled).toContain("the plugin was unloaded");
      release();
      await settleMicrotasks();
      expect(late.unchanged()).toBe(true);

      const reads = plugin.inProcess.length;
      const refused = plugin.run("title", "ses_a", plugin.history.incompatible);
      expect(await refused.settled).toContain("the plugin was unloaded");
      expect(refused.unchanged()).toBe(true);
      expect(plugin.inProcess).toHaveLength(reads);
      expect(plugin.server.seen).toEqual([]);
      // A request that needs no read is still left as it is.
      const plain = plugin.run(
        "context",
        "ses_a",
        requestMessages(plainTurn(4)),
      );
      expect(await plain.settled).toBeUndefined();
      expect(plain.unchanged()).toBe(true);
    } finally {
      await plugin.server.stop();
    }
  });

  test("cleanup between a read settling and its use projects nothing", async () => {
    const plugin = await mediaPlugin(() => undefined);
    try {
      plugin.cleanupAfterContextRead();
      const covered = plugin.run(
        "context",
        "ses_a",
        requestMessages(plugin.history.context),
      );
      expect(await covered.settled).toContain("the plugin was unloaded");
      expect(covered.unchanged()).toBe(true);
      expect(plugin.server.seen).toEqual([]);
    } finally {
      await plugin.server.stop();
    }
  });

  test("cleanup leaves a running successor command's requests to finish with its own result", async () => {
    // The command's first history page is held until after cleanup.
    const page = Promise.withResolvers<void>();
    let pages = 0;
    // The rest of the at-rest read finds nothing running or waiting.
    const idle: Record<string, unknown> = {
      "/api/session": { data: [], cursor: { next: null } },
      "/api/session/active": { data: {} },
      "/api/session/ses_a/inbox": { data: [] },
      "/api/session/ses_a/permission": { data: [] },
      "/api/session/ses_a/form": { data: [] },
      "/api/shell": { data: [] },
      "/api/experimental/session/ses_a/instructions/entries": { data: [] },
    };
    const plugin = await mediaPlugin(url => {
      if (url.pathname === "/api/info") return serverInfo(process.pid);
      if (url.pathname in idle) return Response.json(idle[url.pathname]);
      if (url.pathname.endsWith("/message") && pages++ === 0)
        return page.promise.then(() =>
          Response.json({
            data: plugin.history.records,
            cursor: { next: null },
          }),
        );
      return undefined;
    });
    try {
      const recovering = plugin.commands.get("magic-recover")!({
        sessionID: "ses_a",
        prompt: { text: "" },
      }).catch((reason: Error) => reason.message);
      await plugin.server.waitFor(() => pages === 1);
      plugin.cleanup();
      page.resolve();
      expect(await recovering).toBe(RECOVER_NOOP);
      expect(plugin.server.aborted).toEqual([]);
      const paths = plugin.server.seen.map(seen => seen.path);
      const held = paths.findIndex(path => path.endsWith("/message"));
      // Qualified before the held page, and reading to the end after it.
      expect(paths.slice(0, 2)).toEqual(["/api/info", "/api/session/ses_a"]);
      expect(paths.slice(held + 1)).toEqual([
        "/api/experimental/session/ses_a/export",
        ...Object.keys(idle),
      ]);
    } finally {
      await plugin.server.stop();
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
    ).then(
      result => result.summaries,
      (error: Error) => error,
    );
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
  // The handlers read process.env. Swap in a complete fake map so neither an
  // inherited backend, threshold nor real key reaches the test.
  const withPercent = async <T>(value: string, run: () => Promise<T>) => {
    const inherited = process.env;
    process.env = {
      TYPESAFE_API_KEY: "fake-jev-key",
      MAGIC_COMPACT_OPENAI_DECISIONS_API_KEY: "fake-openai-key",
      [RETENTION_KEEP_THRESHOLD_PERCENT_ENV]: value,
    } as NodeJS.ProcessEnv;
    try {
      return await run();
    } finally {
      process.env = inherited;
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

  test.each(["magic-compact", "magic-trim"])(
    "%s neither reads nor validates the percent under --heuristic or --threshold",
    async name => {
      const run = await commands();
      for (const args of [
        "--heuristic",
        "--threshold 70",
        "1 --truncate --threshold 0",
      ]) {
        const error = await withPercent("60%", () => run(name, args));
        expect(error).not.toBeInstanceOf(RetentionClassificationError);
        expect(String(error)).toContain(V2_ENDPOINT_REQUIRED);
      }
    },
  );

  test.each([
    ["magic-compact", COMPACT_THRESHOLD_ERROR, COMPACT_ARGUMENTS_ERROR],
    ["magic-trim", TRIM_THRESHOLD_ERROR, TRIM_ARGUMENTS_ERROR],
  ])(
    "%s fails with its argument error before the environment or the server",
    async (name, thresholdError, argumentsError) => {
      const run = await commands();
      for (const [args, message] of [
        ["--threshold 70%", thresholdError],
        ["--threshold=70", argumentsError],
        ["--heuristic --truncate", argumentsError],
      ] as const) {
        const error = await withPercent("60%", () => run(name, args));
        expect((error as Error).message).toBe(message);
      }
    },
  );
});

describe("OpenCode 2 response tool boundary", () => {
  const RESPONSE = "StructuredOutput";
  const SUMMARY = "SessionSummary";
  type Execute = (input: unknown, context: never) => Promise<unknown>;
  const harness = async (owned: OwnedSessions) => {
    const hooks = new Map<string, (event: never) => Promise<void> | void>();
    const executors = new Map<string, Execute>();
    let contextReads = 0;
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
          context: async () => {
            contextReads += 1;
            return [];
          },
        },
        event: { subscribe: async function* () {} },
      } as never,
      owned,
    );
    const advertised = async (kind: string, sessionID: string) => {
      const event = {
        sessionID,
        model: { providerID: "test", id: "test-model" },
        tools: { [RESPONSE]: {}, [SUMMARY]: {}, read: {} },
        messages: [],
      };
      await hooks.get(kind)!(event as never);
      return Object.keys(event.tools);
    };
    const respond = (sessionID: string) =>
      executors.get(RESPONSE)!({}, { sessionID } as never);
    const submitSummary = (sessionID: string) =>
      executors.get(SUMMARY)!({}, { sessionID } as never);
    const request = async (
      kind: string,
      sessionID: string,
      messages: Array<{ role: string; content: unknown[] }>,
      providerID: string,
    ) => {
      const event = {
        sessionID,
        model: { providerID, id: "model" },
        tools: {},
        messages,
      };
      await hooks.get(kind)!(event as never);
      return event.messages;
    };
    const context = async (event: ContextEvent) => {
      await hooks.get("context")!(event as never);
    };
    const title = async (event: Record<string, unknown>) => {
      await hooks.get("title")!(event as never);
      return event;
    };
    const readOmitted = (sessionID: string, contentId: string) =>
      executors.get("read_omitted_content")!({ contentId }, {
        sessionID,
      } as never);
    return {
      advertised,
      respond,
      submitSummary,
      request,
      context,
      title,
      readOmitted,
      contextReads: () => contextReads,
    };
  };

  test("the context hook stops an owned session's second request and every request once it is closed", async () => {
    const owned = new OwnedSessions();
    owned.register("ses_gen", "generation", false);
    owned.register("ses_closed", "retention", false);
    owned.close("ses_closed");
    const { advertised } = await harness(owned);
    expect(await advertised("context", "ses_gen")).toEqual([RESPONSE, "read"]);
    await expect(advertised("context", "ses_gen")).rejects.toThrow(owned.stop);
    await expect(advertised("context", "ses_closed")).rejects.toThrow(
      owned.stop,
    );
    // A session this process does not own is never stopped.
    for (let request = 0; request < 3; request++) {
      expect(await advertised("context", "ses_user")).toEqual(["read"]);
    }
  });

  // The registry entries below come from the real generation and summary
  // requests; only the context hook's effect on each request is observed.
  test.each([
    ["the OpenAI Responses serializer", "@opencode/ai/providers/openai", true],
    ["another serializer", "@opencode/ai/providers/openai-compatible", false],
  ] as const)(
    "a per-turn generation on %s keeps the response tool, projects reasoning as registered, and its next step is stopped",
    async (_name, serializer, readable) => {
      const owned = new OwnedSessions();
      const { context } = await harness(owned);
      const turn = [
        userMessage("msg_u", "Do it"),
        assistantMessage(
          "msg_a",
          [
            { type: "reasoning", text: "deepseek analysis" },
            { type: "text", text: words("A", 200) },
          ],
          { model: { providerID: "deepseek", id: "deepseek-flash" } },
        ),
        idleMessage("msg_i"),
      ];
      const state = fakeV2State();
      const source = sourceSession(state, "ses_src", turn);
      const { summaries } = await generateV2Summaries(
        fakeV2Host(state, owned, echoGeneration(), context),
        owned,
        { session: source, messages: turn, instructions: [] },
        {
          ref: state.summarizerModel!,
          info: {
            package: serializer,
            limit: state.modelLimit,
            capabilities: { tools: true, input: ["text"], output: ["text"] },
          } as never,
        },
        [{ span: turn }],
        "index",
        { controlTimeoutMs: 1_000 },
      );

      expect(summaries).toEqual([{ text: "Summary of target 1." }]);
      const [first, second] = state.contextRequests;
      expect(state.contextRequests).toHaveLength(2);
      expect(first).toMatchObject({
        tools: [RESPONSE, "read"],
        stopped: false,
      });
      expect(second!.stopped).toBeTrue();
      const sent = first!.messages.flatMap(message => message.content);
      expect(sent).toContainEqual(
        readable
          ? {
              type: "text",
              text: `${READABLE_REASONING_HEADER}\ndeepseek analysis`,
            }
          : { type: "reasoning", text: "deepseek analysis" },
      );
    },
  );

  test("a whole-session summary request sees neither plugin tool", async () => {
    const owned = new OwnedSessions();
    const { context } = await harness(owned);
    const turn = [
      userMessage("msg_u", "Do it"),
      assistantMessage("msg_a", [{ type: "text", text: "Done." }]),
      idleMessage("msg_i"),
    ];
    const state = fakeV2State();
    const source = sourceSession(state, "ses_src", turn);
    await generateV2SessionSummary(
      fakeV2Host(
        state,
        owned,
        () => ({ text: "## Objective\n- Continue." }),
        context,
      ),
      owned,
      { session: source, messages: turn, instructions: [] },
      {
        ref: state.summarizerModel!,
        info: {
          limit: state.modelLimit,
          capabilities: { tools: true, input: ["text"], output: ["text"] },
        },
      },
      1_000,
      false,
    );
    expect(state.contextRequests).toEqual([
      expect.objectContaining({ tools: ["read"], stopped: false }),
    ]);
  });

  test("an owned session's title is fixed without reading its history", async () => {
    const owned = new OwnedSessions();
    owned.register("ses_gen", "generation", false);
    const { title, contextReads } = await harness(owned);
    const image = (sessionID: string) => ({
      sessionID,
      messages: [
        {
          id: "msg_img",
          role: "user",
          content: [
            {
              type: "media",
              media: {
                mediaType: "image/png",
                source: {
                  type: "base64",
                  data: "Zm9v",
                  mediaType: "image/png",
                },
              },
            },
          ],
        },
      ],
    });

    expect((await title(image("ses_gen")))["result"]).toBe("Magic Compact");
    expect(contextReads()).toBe(0);
    // A user session's title request keeps its own result.
    expect(
      (await title({ sessionID: "ses_user", messages: [] }))["result"],
    ).toBeUndefined();
  });

  test("read_omitted_content reads only the current session's OpenCode 2 cache", async () => {
    const { readOmitted } = await harness(new OwnedSessions());
    const omission = planOmission("ses_cur", { content: "omitted bytes" });
    await commitPlannedOmission(v2StorageKey("ses_cur"), omission);
    // The same entry under the OpenCode 1 key of that session ID.
    await commitPlannedOmission("ses_cur", omission);

    expect(await readOmitted("ses_cur", omission.contentID)).toEqual({
      content: "omitted bytes",
    });
    expect(await readOmitted("ses_other", omission.contentID)).toEqual({
      content: `Omitted content unavailable for Content ID ${omission.contentID}: no authorized current-session cache entry exists.`,
    });
    await rm(cachePath(v2StorageKey("ses_cur")));
    expect(await readOmitted("ses_cur", omission.contentID)).toEqual({
      content: `Omitted content unavailable for Content ID ${omission.contentID}: no authorized current-session cache entry exists.`,
    });
  });

  test("a session this process does not own never sees the response or summary tool, in any request kind", async () => {
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
    owned.register("ses_gen", "generation", false);
    owned.register("ses_ret", "retention", false);
    const { advertised } = await harness(owned);
    expect(await advertised("context", "ses_gen")).toEqual([RESPONSE, "read"]);
    expect(await advertised("generate", "ses_gen")).toEqual(["read"]);
    await expect(advertised("compaction", "ses_gen")).rejects.toThrow(
      owned.compactionBlocked,
    );
    expect(await advertised("context", "ses_ret")).toEqual(["read"]);
    expect(await advertised("generate", "ses_ret")).toEqual(["read"]);
  });

  // The structured whole-session summary on the DeepSeek route has its own
  // tool with the summary schema, never the per-turn response envelope.
  test("only a structured session summary request sees the session summary tool, and only it acknowledges", async () => {
    const owned = new OwnedSessions();
    owned.register("ses_gen", "generation", false);
    owned.register("ses_ret", "retention", false);
    owned.register("ses_sum", "retention-tool", false);
    const { advertised, respond, submitSummary } = await harness(owned);
    expect(await advertised("context", "ses_sum")).toEqual([SUMMARY, "read"]);
    expect(await advertised("context", "ses_gen")).toEqual([RESPONSE, "read"]);
    expect(await advertised("context", "ses_ret")).toEqual(["read"]);
    expect(await advertised("generate", "ses_sum")).toEqual(["read"]);
    expect(await advertised("context", "ses_user")).toEqual(["read"]);
    expect(await submitSummary("ses_sum")).toEqual({ content: "Recorded." });
    for (const session of ["ses_gen", "ses_ret", "ses_user"]) {
      await expect(submitSummary(session)).rejects.toThrow("only available");
    }
    await expect(respond("ses_sum")).rejects.toThrow("only available");
  });

  // Another model's plain reasoning reaches an owned request on the OpenAI
  // Responses serializer as text at its own position; native metadata-bearing
  // reasoning, other serializers and sessions this process does not own are
  // left as the host built them.
  test("shows another model's plain reasoning as text only in owned requests on the OpenAI Responses serializer", async () => {
    const owned = new OwnedSessions();
    // Each owned session makes one provider request.
    // Registered with the serializer its request runs on.
    owned.register("ses_gen", "generation", true);
    owned.register("ses_gen_deepseek", "generation", false);
    const { request } = await harness(owned);
    const messages = () => [
      {
        role: "assistant",
        content: [
          { type: "reasoning", text: "deepseek analysis" },
          {
            type: "reasoning",
            text: "native",
            providerMetadata: {
              openai: { itemId: "rs_1", reasoningEncryptedContent: "enc" },
            },
          },
          { type: "text", text: "answer" },
        ],
      },
    ];
    const openai = await request("context", "ses_gen", messages(), "openai");
    expect(openai[0]!.content).toEqual([
      { type: "text", text: `${READABLE_REASONING_HEADER}\ndeepseek analysis` },
      messages()[0]!.content[1],
      { type: "text", text: "answer" },
    ]);
    expect(
      await request("context", "ses_gen_deepseek", messages(), "deepseek"),
    ).toEqual(messages());
    expect(await request("context", "ses_user", messages(), "openai")).toEqual(
      messages(),
    );
  });

  test("the response tool acknowledges only inside a per-turn generation session", async () => {
    const owned = new OwnedSessions();
    owned.register("ses_gen", "generation", false);
    owned.register("ses_ret", "retention", false);
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
      false,
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

  // A turn is anchored by its first user message's text, even an empty one;
  // a later adjacent user message never supplies the anchor.
  test("anchors a turn by its first user message, even when that message has no text", async () => {
    const state = fakeV2State();
    const messages = [
      userMessage("msg_u1a", ""),
      userMessage("msg_u1b", `${"x".repeat(170)}\nsecond line`),
      assistantMessage("msg_a1", [{ type: "text", text: "Answer 1" }]),
      idleMessage("msg_i1"),
      ...turn(2, words("A2", 300)),
      ...turn(3),
    ];
    sourceSession(state, "ses_src", messages);
    const owned = new OwnedSessions();
    const prompts: string[] = [];
    await executeV2Compact(
      fakeV2Host(state, owned, prompt => {
        prompts.push(prompt);
        return echoGeneration()(prompt);
      }),
      owned,
      "ses_src",
      1,
      { retention: { mode: "heuristic" }, controlTimeoutMs: 1_000 },
    );
    // Both target turns repeat the same map.
    expect(prompts).toHaveLength(2);
    for (const [index, prompt] of prompts.entries()) {
      const { marker } = (
        JSON.parse(prompt.split("\n").at(-1)!) as {
          turns: Array<{ marker: string }>;
        }
      ).turns[0]!;
      expect(prompt).toBe(
        [
          "Generate one continuation summary for the indexed target turn using the retained native history.",
          GENERATION_PROTOCOL,
          "Submit the object through StructuredOutput. Its turns array must contain exactly one object, for the single target index entry below.",
          "The globalTurnIndex is an orientation-only user-anchor map. Its sourceOrdinal values are immutable source-history positions, not target ordinals, and it does not supply distant assistant or tool detail. Each entry contains only its sourceOrdinal and bounded user anchor:",
          JSON.stringify({
            globalTurnIndex: [
              { sourceOrdinal: 1, anchor: "" },
              { sourceOrdinal: 2, anchor: "Request 2" },
              { sourceOrdinal: 3, anchor: "Request 3" },
            ],
          }),
          "Use the ephemeral markers only as ordering guides. Echo each exact marker token and target ordinal; marker echo is allowed. Never include the active generation nonce or marker token in summaries or visual descriptions; useful source identifiers may remain when you include them.",
          JSON.stringify({
            turns: [{ ordinal: index + 1, marker, expectedInlineImages: 0 }],
          }),
        ].join("\n"),
      );
    }
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

  // The oracle is what a real request carries: the prompt the host received,
  // the marker message in the imported session, and the response tool's
  // input schema as setup registers it.
  test("the explicit request estimate charges the prompt, marker and tool schema a request carries", async () => {
    const schemas = new Map<string, unknown>();
    const idle = { add: () => undefined, update: () => undefined };
    await setup({
      options: {},
      agent: { transform: async (fn: (e: unknown) => void) => fn(idle) },
      command: { transform: async (fn: (e: unknown) => void) => fn(idle) },
      tool: {
        transform: async (fn: (e: unknown) => void) =>
          fn({
            add: (tool: { name: string; input: unknown }) =>
              schemas.set(tool.name, tool.input),
            update: () => undefined,
          }),
      },
      session: { hook: async () => undefined },
      event: { subscribe: async function* () {} },
    } as never);
    const turn = [
      userMessage("msg_u", "Do it"),
      assistantMessage("msg_a", [{ type: "text", text: "Done." }]),
      idleMessage("msg_i"),
    ];
    const state = fakeV2State();
    const source = sourceSession(state, "ses_src", turn);
    const markers: string[] = [];
    state.afterWrite = (method, sessionID) => {
      if (method !== "import") return;
      for (const message of state.messages.get(sessionID)!) {
        if (message.type === "synthetic") markers.push(message.text);
      }
    };
    const owned = new OwnedSessions();
    await generateV2Summaries(
      fakeV2Host(state, owned, echoGeneration()),
      owned,
      { session: source, messages: turn, instructions: [] },
      {
        ref: state.summarizerModel!,
        info: {
          limit: state.modelLimit,
          capabilities: { tools: true, input: ["text"], output: ["text"] },
        },
      },
      [{ span: turn }],
      "index",
      { controlTimeoutMs: 1_000 },
    );

    expect(markers).toHaveLength(1);
    const [prompt] = state.requests.map(request => request.text);
    expect(prompt).toContain(GENERATION_PROTOCOL);
    expect(prompt).toContain("index");
    const target = {
      ordinal: 1,
      marker: markers[0]!,
      expectedInlineImages: 0,
    };
    expect(estimateGenerationExplicitRequestTokens(target, "index")).toBe(
      countGptTokens(JSON.stringify(schemas.get("StructuredOutput")))
        + countGptTokens(prompt!)
        + countGptTokens(markers[0]!),
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

  test("compact without the selected key stops before resolving the summarizer", async () => {
    const state = fakeV2State();
    const turn = (n: number): V2Message[] => [
      userMessage(`msg_u${n}`, `Request ${n}`),
      assistantMessage(`msg_a${n}`, [
        toolItem(`call_${n}`, "read", { path: "/f" }, [
          { type: "text", text: words(`R${n}`, 300) },
        ]),
      ]),
      idleMessage(`msg_i${n}`),
    ];
    sourceSession(state, "ses_src", [...turn(1), ...turn(2)]);
    const owned = new OwnedSessions();
    const host = fakeV2Host(state, owned, echoGeneration());
    const reads: string[] = [];
    const agentGet = host.agent.get.bind(host.agent);
    host.agent.get = (async (...args: Parameters<typeof agentGet>) => {
      reads.push("agent");
      return agentGet(...args);
    }) as typeof host.agent.get;
    const modelList = host.model.list.bind(host.model);
    host.model.list = (async (...args: Parameters<typeof modelList>) => {
      reads.push("models");
      return modelList(...args);
    }) as typeof host.model.list;
    const outcome = await executeV2Compact(host, owned, "ses_src", 1, {
      retention: { mode: "task-aware", apiKey: undefined },
      controlTimeoutMs: 1_000,
    }).catch((error: unknown) => error);
    expect(outcome).toBeInstanceOf(RetentionClassificationError);
    expect((outcome as RetentionClassificationError).kind).toBe(
      "missing-api-key",
    );
    expect(reads).toEqual([]);
    expect(state.requests).toHaveLength(0);
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
    expect(outcome).toContain("Request turns summarized: 1");
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

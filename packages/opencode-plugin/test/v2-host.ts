// In-memory OpenCode 2.0.28 host for the V2 adapter suites. It implements the
// public client methods the adapter calls, keeps array order as the
// authoritative history order like the host's import, and simulates the agent
// loop of an owned session through the plugin's own context-hook rule.
import type { SessionInfo } from "@opencode/client";
import type { V2Host } from "../src/v2/host";
import type { OwnedSessions } from "../src/v2/owned";
import type { V2Message } from "../src/v2/session";
import { countGptTokens } from "../src/stats/tokenize";

export type FakeReply =
  | { text: string }
  | {
      tool: Array<{
        name: string;
        input: unknown;
        status?: "completed" | "error";
      }>;
    }
  | { error: { type: string; message: string } };

type SummarizerModel = { providerID: string; id: string; variant?: string };

export type FakeV2State = {
  pid: number;
  sessions: Map<string, SessionInfo>;
  messages: Map<string, V2Message[]>;
  instructions: Map<string, Array<{ key: string; value: unknown }>>;
  active: Record<string, { type: "running" }>;
  inbox: Map<string, unknown[]>;
  permissions: Map<string, unknown[]>;
  forms: Map<string, unknown[]>;
  shells: Array<Record<string, unknown>>;
  // The summarizer at the server's default location.
  summarizerModel?: SummarizerModel;
  // Summarizers configured for a project directory. A request that names a
  // directory with an entry here gets that model, like a project profile.
  projectSummarizers?: Record<string, SummarizerModel>;
  modelLimit: { context: number; input?: number; output: number };
  modelInput: string[];
  // Every provider request the fake made, by session.
  requests: Array<{ sessionID: string; text: string }>;
  calls: string[];
  // Called on every write; a test can mutate state to simulate another writer.
  afterWrite?: (method: string, sessionID: string) => void;
  // Awaited before a lifecycle write reaches the host state; a test can delay
  // it or throw to fail the request before it executes.
  beforeWrite?: (method: string, sessionID: string) => void | Promise<void>;
  failImport?: boolean;
  // The host runs the prompt, then the request fails on the client side.
  rejectAcceptedPrompt?: boolean;
};

export function fakeV2State(): FakeV2State {
  return {
    pid: 4242,
    sessions: new Map(),
    messages: new Map(),
    instructions: new Map(),
    active: {},
    inbox: new Map(),
    permissions: new Map(),
    forms: new Map(),
    shells: [],
    summarizerModel: {
      providerID: "probe",
      id: "probe-model",
      variant: "summary",
    },
    modelLimit: { context: 200_000, output: 32_000 },
    modelInput: ["text", "image"],
    requests: [],
    calls: [],
  };
}

export const LOCATION = { directory: "/workspace" };

export function sourceSession(
  state: FakeV2State,
  id: string,
  messages: V2Message[],
  overrides: Partial<SessionInfo> = {},
): SessionInfo {
  const session = {
    id,
    projectID: "project",
    agent: "build",
    model: { providerID: "probe", id: "probe-model" },
    cost: 0,
    tokens: { input: 0, output: 0, reasoning: 0, cache: { read: 0, write: 0 } },
    time: { created: 1, updated: 2, idle: 3 },
    title: "Source",
    location: LOCATION,
    metadata: { seed: "kept" },
    permissions: [{ action: "*", resource: "*", effect: "allow" }],
    ...overrides,
  } as SessionInfo;
  state.sessions.set(id, session);
  state.messages.set(id, structuredClone(messages));
  return session;
}

let counter = 0;
const clone = <T>(value: T): T => JSON.parse(JSON.stringify(value)) as T;

// Every string reachable from a value through own properties, enumerable or
// not: an error's message, stack, name, code, cause, aggregated errors and any
// custom fields. Used to prove host content never reaches a thrown failure.
export function reachableStrings(value: unknown): string[] {
  const seen = new Set<unknown>();
  const strings: string[] = [];
  const visit = (current: unknown) => {
    if (typeof current === "string") {
      strings.push(current);
      return;
    }
    if (typeof current !== "object" || current === null || seen.has(current))
      return;
    seen.add(current);
    for (const key of Reflect.ownKeys(current)) {
      const descriptor = Object.getOwnPropertyDescriptor(current, key);
      visit(descriptor?.value);
    }
  };
  visit(value);
  return strings;
}

// A host failure carrying private content everywhere a client error can:
// message, a spoofed plugin error name, an uppercase code, an out-of-range
// status, a response body, custom data, and a nested cause.
export const PRIVATE_SENTINEL = "PRIVATE_SENTINEL_7F3A";
// What the tokenizer dependency (gpt-tokenizer 3.4.0) throws for a literal
// special token: a plain `Error` with no code or other field, so only a mark
// tells it from an error this plugin raised.
export const SPECIAL_TOKEN = "<|endoftext|>";
export function dependencyError(): unknown {
  try {
    countGptTokens(SPECIAL_TOKEN);
  } catch (error) {
    return error;
  }
  throw new Error("The tokenizer accepted the special token it refuses.");
}

export function privateHostError(): Error {
  const error = new Error(
    `host body ${PRIVATE_SENTINEL} at /home/${PRIVATE_SENTINEL}/x`,
    { cause: new AggregateError([new Error(PRIVATE_SENTINEL)]) },
  );
  return Object.assign(error, {
    name: "SourceChangedError",
    code: PRIVATE_SENTINEL,
    status: 1_000,
    response: { status: 200, body: PRIVATE_SENTINEL },
    data: { message: PRIVATE_SENTINEL },
  });
}

export function userMessage(
  id: string,
  text: string,
  extra: object = {},
): V2Message {
  return {
    id,
    type: "user",
    text,
    time: { created: ++counter },
    ...extra,
  } as V2Message;
}

export function assistantMessage(
  id: string,
  content: unknown[],
  extra: object = {},
): V2Message {
  return {
    id,
    type: "assistant",
    agent: "build",
    model: { providerID: "probe", id: "probe-model" },
    content,
    finish: "stop",
    time: { created: ++counter, completed: ++counter },
    ...extra,
  } as V2Message;
}

export function idleMessage(id: string): V2Message {
  return {
    id,
    type: "idle",
    outcome: "succeeded",
    time: { created: ++counter },
  } as V2Message;
}

export function toolItem(
  callID: string,
  name: string,
  input: Record<string, unknown>,
  content: unknown[],
  metadata?: Record<string, unknown>,
): unknown {
  return {
    type: "tool",
    id: callID,
    name,
    state: {
      status: "completed",
      input,
      content,
      ...(metadata ? { metadata } : {}),
    },
    time: { created: ++counter, completed: ++counter },
  };
}

// `reply` decides each owned session's first model response from the prompt
// text and the session history.
export function fakeV2Host(
  state: FakeV2State,
  owned: OwnedSessions,
  reply: (text: string, history: V2Message[]) => FakeReply = () => ({
    text: "reply",
  }),
): V2Host {
  const summarizerAt = (location?: { directory?: string }) =>
    (location?.directory === undefined
      ? undefined
      : state.projectSummarizers?.[location.directory])
    ?? state.summarizerModel;
  const get = (sessionID: string) => {
    const session = state.sessions.get(sessionID);
    if (!session)
      throw Object.assign(new Error("Session not found"), {
        _tag: "SessionNotFoundError",
      });
    return clone(session);
  };
  const history = (sessionID: string) => {
    get(sessionID);
    return clone(state.messages.get(sessionID) ?? []);
  };
  const write = (method: string, sessionID: string) => {
    state.calls.push(`${method}:${sessionID}`);
    state.afterWrite?.(method, sessionID);
  };
  const runPrompt = async (input: {
    sessionID: string;
    id: string;
    text: string;
  }) => {
    const messages = state.messages.get(input.sessionID)!;
    messages.push(userMessage(input.id, input.text));
    state.requests.push({ sessionID: input.sessionID, text: input.text });
    owned.admitRequest(input.sessionID);
    const response = reply(input.text, clone(messages));
    if ("error" in response) {
      messages.push(
        assistantMessage(`${input.id}-reply`, [], {
          finish: "error",
          error: response.error,
        }),
      );
      messages.push({
        id: `${input.id}-idle`,
        type: "idle",
        outcome: "failed",
        error: response.error,
        time: { created: ++counter },
      } as V2Message);
      return;
    }
    if ("text" in response) {
      messages.push(
        assistantMessage(
          `${input.id}-reply`,
          [{ type: "text", text: response.text }],
          { agent: get(input.sessionID).agent },
        ),
      );
      messages.push(idleMessage(`${input.id}-idle`));
      return;
    }
    messages.push(
      assistantMessage(
        `${input.id}-reply`,
        response.tool.map((call, index) => ({
          type: "tool",
          id: `call_${index}`,
          name: call.name,
          state:
            call.status === "error"
              ? {
                  status: "error",
                  input: call.input,
                  error: { type: "tool.invalid", message: "invalid" },
                }
              : {
                  status: "completed",
                  input: call.input,
                  content: [{ type: "text", text: "Recorded." }],
                },
          time: { created: ++counter },
        })),
        { finish: "tool-calls", agent: get(input.sessionID).agent },
      ),
    );
    // The agent loop asks for the next step; the owned stop ends it.
    try {
      owned.admitRequest(input.sessionID);
      messages.push(
        assistantMessage(`${input.id}-second`, [
          { type: "text", text: "second" },
        ]),
      );
      messages.push(idleMessage(`${input.id}-idle`));
    } catch (error) {
      messages.push({
        id: `${input.id}-idle`,
        type: "idle",
        outcome: "failed",
        error: { type: "unknown", message: (error as Error).message },
        time: { created: ++counter },
      } as V2Message);
    }
  };
  const host = {
    server: {
      info: async () => ({
        version: "2.0.28",
        pid: state.pid,
        urls: [],
        paths: { tmp: "/tmp" },
      }),
    },
    agent: {
      get: async (input: { location?: { directory?: string } }) => {
        const model = summarizerAt(input.location);
        return {
          data: {
            id: "magic-compact-summarizer",
            ...(model ? { model } : {}),
          },
        };
      },
    },
    model: {
      list: async (input: { location?: { directory?: string } }) => {
        const model = summarizerAt(input.location);
        return {
          location: LOCATION,
          data: model
            ? [
                {
                  ...model,
                  limit: state.modelLimit,
                  capabilities: {
                    tools: true,
                    input: state.modelInput,
                    output: ["text"],
                  },
                },
              ]
            : [],
        };
      },
    },
    message: {
      list: async (input: { sessionID: string; cursor?: string }) => ({
        data: input.cursor ? [] : history(input.sessionID),
        cursor: { previous: null, next: null },
      }),
    },
    permission: {
      list: async (input: { sessionID: string }) =>
        clone(state.permissions.get(input.sessionID) ?? []),
    },
    shell: {
      list: async () => ({ location: LOCATION, data: clone(state.shells) }),
    },
    session: {
      get: async (input: { sessionID: string }) => get(input.sessionID),
      list: async (input: { parentID?: string }) => ({
        data: [...state.sessions.values()]
          .filter(session => session.parentID === input.parentID)
          .map(clone),
        cursor: { previous: null, next: null },
      }),
      active: async () => clone(state.active),
      // Like 2.0.28, the export leaves out an assistant message without a
      // completion time, and the import below drops one.
      export: async (input: { sessionID: string }) => ({
        info: get(input.sessionID),
        messages: history(input.sessionID).filter(
          message =>
            message.type !== "assistant"
            || message.time.completed !== undefined,
        ),
      }),
      import: async (input: { info: SessionInfo; messages: V2Message[] }) => {
        await state.beforeWrite?.("import", input.info.id);
        if (state.failImport) throw new Error("import failed");
        if (state.sessions.has(input.info.id))
          throw new Error("duplicate session");
        const ids = new Set(
          [...state.messages.values()].flat().map(message => message.id),
        );
        if (input.messages.some(message => ids.has(message.id)))
          throw new Error("duplicate message ID");
        state.sessions.set(input.info.id, clone(input.info));
        state.messages.set(
          input.info.id,
          clone(input.messages).filter(
            message =>
              message.type !== "assistant"
              || message.time.completed !== undefined,
          ),
        );
        write("import", input.info.id);
        return get(input.info.id);
      },
      remove: async (input: { sessionID: string }) => {
        await state.beforeWrite?.("remove", input.sessionID);
        get(input.sessionID);
        state.sessions.delete(input.sessionID);
        state.messages.delete(input.sessionID);
        state.instructions.delete(input.sessionID);
        write("remove", input.sessionID);
      },
      update: async (input: {
        sessionID: string;
        title?: string;
        metadata?: Record<string, unknown>;
      }) => {
        await state.beforeWrite?.("update", input.sessionID);
        const session = get(input.sessionID);
        state.sessions.set(input.sessionID, {
          ...session,
          ...(input.title === undefined ? {} : { title: input.title }),
          ...(input.metadata === undefined ? {} : { metadata: input.metadata }),
        } as SessionInfo);
        write("update", input.sessionID);
      },
      prompt: async (input: {
        sessionID: string;
        id: string;
        text: string;
      }) => {
        await runPrompt(input);
        if (state.rejectAcceptedPrompt) throw new Error("prompt response lost");
      },
      wait: async () => undefined,
      inbox: {
        list: async (input: { sessionID: string }) =>
          clone(state.inbox.get(input.sessionID) ?? []),
      },
      form: {
        list: async (input: { sessionID: string }) =>
          clone(state.forms.get(input.sessionID) ?? []),
      },
      instructions: {
        entry: {
          list: async (input: { sessionID: string }) =>
            clone(state.instructions.get(input.sessionID) ?? []),
          put: async (input: {
            sessionID: string;
            key: string;
            value: unknown;
          }) => {
            await state.beforeWrite?.("instructions", input.sessionID);
            const entries = state.instructions.get(input.sessionID) ?? [];
            state.instructions.set(input.sessionID, [
              ...entries.filter(entry => entry.key !== input.key),
              { key: input.key, value: input.value },
            ]);
            write("instructions", input.sessionID);
          },
        },
      },
    },
  };
  return host as unknown as V2Host;
}

// Echoes the single target of a generation prompt through the response tool.
export function echoGeneration(
  summary = (ordinal: number) => `Summary of target ${ordinal}.`,
): (text: string) => FakeReply {
  return text => {
    const match =
      /\{"turns":\[\{"ordinal":(\d+),"marker":"((?:[^"\\]|\\.)*)","expectedInlineImages":(\d+)\}\]/u.exec(
        text,
      );
    if (!match) return { text: "## Objective\n- Continue." };
    const ordinal = Number(match[1]);
    return {
      tool: [
        {
          name: "StructuredOutput",
          input: {
            turns: [
              {
                ordinal,
                marker: JSON.parse(`"${match[2]}"`) as string,
                summary: summary(ordinal),
                visualDescriptions: Array.from(
                  { length: Number(match[3]) },
                  (_, index) => ({
                    ordinal: index + 1,
                    description: "An image.",
                  }),
                ),
              },
            ],
          },
        },
      ],
    };
  };
}

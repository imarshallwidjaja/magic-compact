// In-memory OpenCode 1.18.30 host for the trim and recovery command suites.
// `messages` is the live source; summary forks and successors are persisted
// through the SQLite-backed fork history of retention-host.ts.
import type {
  Message,
  Part,
  Session,
  TextPart,
  ToolPart,
} from "@opencode-ai/sdk/v2";
import type { V2Client } from "../src/api";
import type { MessageWithParts } from "../src/compact/plan";
import { deleteCache } from "../src/storage/omission";
import { deleteStats } from "../src/storage/stats";
import {
  persistedForkHistory,
  providerCatalog,
  summarizerAgent,
  summaryReply,
} from "./retention-host";

export const PRIVATE_BODY = "PRIVATE-SESSION-BODY-7f3a";

export type V1CommandHarnessOptions = {
  toastMode?: "hang" | "fail";
  sessionRevert?: { messageID: string };
  // Runs while the retention summary request is in flight.
  duringSummary?: () => void;
  // Runs once the successor session exists, before its final source recheck.
  duringSuccessorCreate?: () => void;
  // Replaces source message reads after this many successful reads.
  failSourceReadAfter?: number;
  // Live state the host reports: session statuses by ID, the source's child
  // sessions, pending permission requests and questions, and extra source
  // session fields such as a running native compaction.
  statuses?: Record<string, { type: string }>;
  children?: Session[];
  permissions?: Array<{ sessionID: string }>;
  questions?: Array<{ sessionID: string }>;
  sessionOverrides?: Partial<Session>;
  // Replaces the default session summary the retention fork replies with.
  summaryText?: string;
};

export function v1CommandHarness(
  sessionID: string,
  messages: MessageWithParts[],
  options: V1CommandHarnessOptions = {},
) {
  messages.forEach((item, index) => {
    item.info.sessionID = sessionID;
    item.info.time ??= { created: index + 1 };
    if (item.info.role === "assistant")
      (item.info.time as { completed?: number }).completed ??=
        item.info.time.created;
    for (const part of item.parts) {
      part.sessionID = sessionID;
      part.messageID = item.info.id;
    }
  });
  const history = persistedForkHistory([]);
  const successorIDs: string[] = [];
  const summaryForkID = `${sessionID}_summary`;
  const sourceSession = {
    id: sessionID,
    title: "Trim source",
    directory: "/workspace",
    time: { created: 1, updated: 1 },
    ...options.sessionOverrides,
  } as Session;
  const toasts: Array<{ message?: string }> = [];
  const deletedSessions: string[] = [];
  const forks: string[] = [];
  const modelCalls: string[] = [];
  const sourceWrites: string[] = [];
  const created: string[] = [];
  let sourceReads = 0;
  let revert = options.sessionRevert;
  const sessions = new Map<string, Session>();
  const v2 = {
    app: {
      agents: async () => {
        modelCalls.push("agents");
        return { data: [summarizerAgent()] };
      },
    },
    provider: {
      list: async () => {
        modelCalls.push("providers");
        return { data: providerCatalog() };
      },
    },
    permission: { list: async () => ({ data: options.permissions ?? [] }) },
    question: { list: async () => ({ data: options.questions ?? [] }) },
    session: {
      messages: async (request: { sessionID: string }) => {
        if (request.sessionID !== sessionID)
          return { data: history.read(request.sessionID) };
        sourceReads += 1;
        if (
          options.failSourceReadAfter !== undefined
          && sourceReads > options.failSourceReadAfter
        ) {
          return { error: { data: { message: PRIVATE_BODY } } };
        }
        return { data: structuredClone(messages) };
      },
      get: async (request: { sessionID: string }) => ({
        data:
          request.sessionID === sessionID
            ? { ...sourceSession, ...(revert ? { revert } : {}) }
            : sessions.get(request.sessionID),
      }),
      status: async () => ({ data: options.statuses ?? {} }),
      children: async (request: { sessionID: string }) => ({
        data: request.sessionID === sessionID ? (options.children ?? []) : [],
      }),
      fork: async (request: { messageID?: string }) => {
        forks.push(summaryForkID);
        history.delete(sessionID);
        messages.forEach((item, index) => history.write(item, index));
        history.delete(summaryForkID);
        history.fork(sessionID, summaryForkID, request.messageID);
        history.delete(sessionID);
        return { data: { ...sourceSession, id: summaryForkID } as Session };
      },
      create: async (request: Record<string, unknown>) => {
        const successorID = `${sessionID}_successor${created.length === 0 ? "" : `_${created.length + 1}`}`;
        successorIDs.push(successorID);
        created.push(successorID);
        const info = {
          ...sourceSession,
          id: successorID,
          title: String(request.title),
          metadata: request.metadata,
        } as Session;
        sessions.set(successorID, info);
        options.duringSuccessorCreate?.();
        return { data: info };
      },
      update: async (request: Record<string, unknown>) => {
        const id = String(request.sessionID);
        if (id === sessionID) sourceWrites.push("session.update");
        const info = {
          ...(sessions.get(id) ?? { ...sourceSession, id }),
          ...(request.title === undefined ? {} : { title: request.title }),
          ...(request.metadata === undefined
            ? {}
            : { metadata: request.metadata }),
        } as Session;
        sessions.set(id, info);
        return { data: info };
      },
      abort: async () => ({ data: true }),
      prompt: async (request: {
        sessionID: string;
        messageID?: string;
        agent?: string;
        parts?: Part[];
      }) => {
        if (request.sessionID !== summaryForkID) {
          sourceWrites.push("session.prompt");
          throw new Error("unexpected prompt");
        }
        modelCalls.push("summary");
        options.duringSummary?.();
        const fork = history.read(summaryForkID);
        const reply = summaryReply(
          fork,
          request,
          summaryForkID,
          options.summaryText,
        );
        fork
          .slice(-2)
          .forEach((item, index) => history.write(item, 100 + index));
        return reply;
      },
      delete: async (request: { sessionID: string }) => {
        if (request.sessionID === sessionID) sourceWrites.push("delete");
        deletedSessions.push(request.sessionID);
        history.delete(request.sessionID);
        return { data: true };
      },
    },
    part: {
      update: async () => {
        sourceWrites.push("part.update");
        throw new Error("unexpected part update");
      },
    },
    sync: {
      replay: async (request: {
        events: Parameters<typeof history.replay>[0];
      }) => {
        history.replay(request.events);
        return { data: { sessionID: request.events[0]!.aggregateID } };
      },
    },
    tui: {
      showToast: async (toast: { message?: string }) => {
        toasts.push(toast);
        if (options.toastMode === "hang") return new Promise(() => {});
        if (options.toastMode === "fail") throw new Error("toast failed");
        return { data: true };
      },
    },
  } as unknown as V2Client;

  return {
    v2,
    sessionID,
    // The most recent successor; IDs are reserved before any is created.
    get successorID() {
      return successorIDs.at(-1) ?? `${sessionID}_successor`;
    },
    summaryForkID,
    messages,
    toasts,
    deletedSessions,
    sourceWrites,
    forks,
    modelCalls,
    history,
    get created() {
      return [...created];
    },
    successorMessages() {
      return history.read(this.successorID);
    },
    successorInfo() {
      return sessions.get(this.successorID);
    },
    // Trim keeps every record in place, so a source part has the same
    // message and part position in the successor.
    successorPart(partID: string): ToolPart {
      const messageIndex = messages.findIndex(item =>
        item.parts.some(part => part.id === partID),
      );
      const partIndex = messages[messageIndex]!.parts.findIndex(
        part => part.id === partID,
      );
      return this.successorMessages()[messageIndex]!.parts[
        partIndex
      ] as ToolPart;
    },
    setRevert(value: { messageID: string }) {
      revert = value;
    },
  };
}

export async function cleanupSessions(...sessionIDs: string[]): Promise<void> {
  await Promise.all(
    sessionIDs.flatMap(sessionID => [
      deleteCache(sessionID),
      deleteStats(sessionID),
    ]),
  );
}

export function message(
  id: string,
  role: "user" | "assistant",
  parts: Part[],
): MessageWithParts {
  return {
    info: { id, role } as Message,
    parts,
  };
}

export function userText(messageID: string, text: string): TextPart {
  return {
    id: `${messageID}_text`,
    sessionID: "session",
    messageID,
    type: "text",
    text,
  };
}

export function completedTool(
  id: string,
  tool: string,
  output: string,
): ToolPart {
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

export const words = (tag: string, count: number) =>
  Array.from({ length: count }, (_, index) => `${tag}-${index}`).join(" ");

// A harness over `messages` that keeps each assistant message without a
// completion time as it is; the shared harness fills one in by default.
export function recoverHarness(
  sessionID: string,
  messages: MessageWithParts[],
  options: V1CommandHarnessOptions = {},
) {
  const open = new Set(
    messages
      .filter(
        item =>
          item.info.role === "assistant"
          && (item.info.time as { completed?: number } | undefined)?.completed
            === undefined,
      )
      .map(item => item.info.id),
  );
  const harness = v1CommandHarness(sessionID, messages, options);
  for (const item of messages) {
    if (open.has(item.info.id))
      delete (item.info.time as { completed?: number }).completed;
  }
  return harness;
}

export function assistant(
  id: string,
  parts: Part[],
  time: { created: number; completed?: number },
): MessageWithParts {
  const item = message(id, "assistant", parts);
  item.info.time = time;
  return item;
}

function tool(id: string, name: string, state: ToolPart["state"]): ToolPart {
  return { ...completedTool(id, name, ""), state };
}

export const backgroundStart = `<task id="ses_bg" state="running">\n<summary>Background task started</summary>\n<task_result>\nworking\n</task_result>\n</task>`;

// A background subagent launch whose result is stored as still running.
export function backgroundLaunch(): ToolPart {
  const launch = completedTool("prt_launch", "task", backgroundStart);
  if (launch.state.status !== "completed") throw new Error("completed");
  launch.state.metadata = { background: true, sessionId: "ses_bg" };
  return launch;
}

// The incident's shape: an older background subagent whose result never
// arrived, then a response that stopped mid-turn with a running subagent call
// and a pending read.
export function interruptedHistory(): MessageWithParts[] {
  const launch = backgroundLaunch();
  return [
    message("usr_1", "user", [userText("usr_1", "Start the job.")]),
    assistant(
      "ast_1",
      [launch, completedTool("prt_read", "read", words("R", 400))],
      { created: 2, completed: 3 },
    ),
    message("usr_2", "user", [userText("usr_2", "Delegate the fix.")]),
    assistant(
      "ast_open",
      [
        { ...userText("ast_open", "Delegating now."), id: "prt_text" },
        tool("prt_task", "task", {
          status: "running",
          input: { prompt: "Fix it." },
          title: "Fixing",
          metadata: { sessionId: "ses_child", output: "partial report" },
          time: { start: 5 },
        }),
        tool("prt_pending", "read", {
          status: "pending",
          input: { filePath: "/a" },
          raw: '{"filePath":"/a"',
        }),
      ],
      { created: 4 },
    ),
  ];
}

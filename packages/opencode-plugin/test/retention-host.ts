// Fake OpenCode host pieces for the task-aware retention summary fork and a
// fake Jev endpoint, shared by the retention, trim, and compact suites.
import type { Message, Part } from "@opencode-ai/sdk/v2";
import { Database } from "bun:sqlite";
import type { MessageWithParts } from "../src/compact/plan";
import {
  RETENTION_MODEL,
  type RetentionFetch,
} from "../src/retention/classifier";

export const SESSION_SUMMARY_TEXT =
  "## Objective\n- Continue the current parser work.";

export function compactionAgent() {
  return {
    name: "compaction",
    native: true,
    hidden: true,
    model: { providerID: "summary-provider", modelID: "summary-model" },
    variant: "low",
  };
}

export function providerCatalog(
  limit: { context: number; input?: number; output?: number } = {
    context: 200_000,
    output: 32_000,
  },
  image = true,
) {
  return {
    all: [
      {
        id: "summary-provider",
        models: {
          "summary-model": { limit, capabilities: { input: { image } } },
        },
      },
    ],
    connected: ["summary-provider"],
    default: {},
  };
}

// OpenCode forks copy history under new message and part IDs.
export function rewriteFork(
  messages: MessageWithParts[],
  forkID: string,
): MessageWithParts[] {
  let partIndex = 0;
  const ids = new Map(
    messages.map((item, index) => [
      item.info.id,
      `${forkID}-message-${index + 1}`,
    ]),
  );
  return structuredClone(messages).map((item, index) => {
    const id = `${forkID}-message-${index + 1}`;
    return {
      info: {
        ...item.info,
        id,
        sessionID: forkID,
        ...(item.info.role === "assistant" && item.info.parentID
          ? { parentID: ids.get(item.info.parentID) ?? item.info.parentID }
          : {}),
      } as Message,
      parts: item.parts.map(part => ({
        ...part,
        id: `${forkID}-part-${++partIndex}`,
        sessionID: forkID,
        messageID: id,
        ...(part.type === "compaction" && part.tail_start_id
          ? { tail_start_id: ids.get(part.tail_start_id) }
          : {}),
      })) as Part[],
    };
  });
}

// Mirrors the pinned host's projection: initial inserts take JSON creation
// time, later message updates replace only JSON, and reads sort row time/id.
// Source rows deliberately use durable input order independently of JSON time.
export function persistedForkHistory(source: MessageWithParts[]) {
  const db = new Database(":memory:");
  db.run(
    "create table message (id text primary key, session text, created integer, data text)",
  );
  const insert = db.query(
    "insert into message values (?, ?, ?, ?) on conflict(id) do update set data=excluded.data",
  );
  const rows = db.query(
    "select data from message where session=? order by created,id",
  );
  const write = (
    message: MessageWithParts,
    created = message.info.time?.created ?? 0,
  ) => {
    insert.run(
      message.info.id,
      message.info.sessionID,
      created,
      JSON.stringify(message),
    );
  };
  source.forEach((message, index) => write(message, index));
  const read = (sessionID: string) =>
    (rows.all(sessionID) as Array<{ data: string }>).map(
      row => JSON.parse(row.data) as MessageWithParts,
    );
  let nextID = 0;
  const fork = (sourceID: string, forkID: string, messageID?: string) => {
    const messages = read(sourceID);
    const end = messageID
      ? messages.findIndex(item => item.info.id === messageID)
      : messages.length;
    const copied = rewriteFork(
      messages.slice(0, end < 0 ? messages.length : end),
      forkID,
    );
    // Real ascending IDs preserve creation order even when timestamps tie.
    const ids = new Map(
      copied.map(message => [
        message.info.id,
        `msg_${(++nextID).toString(16).padStart(12, "0")}${"A".repeat(14)}`,
      ]),
    );
    for (const message of copied) {
      const oldID = message.info.id;
      message.info.id = ids.get(oldID)!;
      if (message.info.role === "assistant" && message.info.parentID) {
        message.info.parentID =
          ids.get(message.info.parentID) ?? message.info.parentID;
      }
      for (const part of message.parts) {
        part.id = `prt_${(++nextID).toString(16).padStart(12, "0")}${"A".repeat(14)}`;
        part.messageID = message.info.id;
        if (part.type === "compaction" && part.tail_start_id) {
          part.tail_start_id = ids.get(part.tail_start_id);
        }
      }
      write(message);
    }
  };
  const replay = (
    events: Array<{
      aggregateID: string;
      seq: number;
      type: string;
      data: Record<string, unknown>;
    }>,
  ) => {
    // An empty native fork has exactly session.created.1 at sequence zero.
    events.forEach((event, index) => {
      if (event.seq !== index + 1)
        throw new Error("unexpected replay sequence");
      if (event.data.sessionID !== event.aggregateID)
        throw new Error("wrong aggregate");
      if (event.type === "message.updated.1") {
        const info = event.data.info as Message;
        const previous = read(event.aggregateID).find(
          item => item.info.id === info.id,
        );
        write({ info, parts: previous?.parts ?? [] });
      } else if (event.type === "message.part.updated.1") {
        const part = event.data.part as Part;
        const message = read(event.aggregateID).find(
          item => item.info.id === part.messageID,
        )!;
        message.parts.push(part);
        write(message);
      } else throw new Error("unexpected replay type");
    });
  };
  return {
    read,
    write,
    fork,
    replay,
    updatePart(part: Part) {
      const message = read(part.sessionID).find(
        item => item.info.id === part.messageID,
      )!;
      const index = message.parts.findIndex(item => item.id === part.id);
      if (index < 0) message.parts.push(part);
      else message.parts[index] = part;
      write(message);
    },
    deleteMessage(sessionID: string, messageID: string) {
      db.query("delete from message where session=? and id=?").run(
        sessionID,
        messageID,
      );
    },
    delete(sessionID: string) {
      db.query("delete from message where session=?").run(sessionID);
    },
    close() {
      db.close();
    },
  };
}

// Appends the prompt and an ordinary compaction-agent reply, as OpenCode does
// for a successful summary request, and returns the prompt response body.
export function summaryReply(
  fork: MessageWithParts[],
  request: { messageID?: string; agent?: string; parts?: Part[] },
  forkID: string,
  text = SESSION_SUMMARY_TEXT,
) {
  const userID = request.messageID ?? `${forkID}-prompt`;
  fork.push({
    info: {
      id: userID,
      sessionID: forkID,
      role: "user",
      agent: request.agent,
    } as Message,
    parts: (request.parts ?? []).map((part, index) => ({
      ...part,
      id: `${userID}-part-${index}`,
      sessionID: forkID,
      messageID: userID,
    })) as Part[],
  });
  const replyID = `${forkID}-reply`;
  const info = {
    id: replyID,
    sessionID: forkID,
    role: "assistant",
    parentID: userID,
    agent: "compaction",
    mode: "compaction",
    finish: "stop",
  } as Message;
  const parts = [
    { id: `${replyID}-start`, type: "step-start" },
    { id: `${replyID}-text`, type: "text", text },
    { id: `${replyID}-finish`, type: "step-finish" },
  ].map(part => ({ ...part, sessionID: forkID, messageID: replyID })) as Part[];
  fork.push({ info, parts });
  return { data: { info, parts } };
}

export type JevCall = {
  task: { latestUserRequest: string; sessionSummary: string };
  candidate: { tool: string; calledWhile: string; output: string };
};

// Scores each single-candidate request by its excerpted output text.
export function scoredJevFetch(
  score: (output: string) => number,
  calls: JevCall[] = [],
): RetentionFetch {
  return async (_url, init) => {
    const body = JSON.parse(String(init.body)) as {
      state: { task: JevCall["task"]; candidates: Array<JevCall["candidate"]> };
    };
    const output = body.state.candidates[0]!.output;
    calls.push({ task: body.state.task, candidate: body.state.candidates[0]! });
    return Response.json({
      model: RETENTION_MODEL,
      answers: { candidate_0: { type: "noul", noul: score(output) } },
      usage: { input_tokens: 10 },
    });
  };
}

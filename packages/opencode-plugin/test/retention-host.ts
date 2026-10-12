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

// Millisecond clock for the fake host's ID generator. With a step, every ID
// minted after `afterWrites` of them comes from `ms` earlier, as when the host
// clock is corrected backwards in the middle of a native fork.
export function hostClock(step?: { afterWrites: number; ms: number }) {
  const start = 1_760_000_000_000;
  let writes = 0;
  return () => start - (step && writes++ >= step.afterWrites ? step.ms : 0);
}

const ID_ALPHABET =
  "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

// Mirrors the pinned host's persistence (OpenCode 1.18.30): ascending IDs are
// clock milliseconds * 4096 plus a counter that resets whenever the clock
// value changes; message rows are read by (time_created, id) with the initial
// insert taking the JSON creation time and later updates replacing only JSON;
// parts are separate rows read by ID. Source fixtures keep durable input order
// for messages and for parts whose IDs are not host IDs.
export function persistedForkHistory(
  source: MessageWithParts[],
  options: { clock?: () => number } = {},
) {
  const clock = options.clock ?? hostClock();
  const db = new Database(":memory:");
  db.run(
    "create table message (id text primary key, session text, created integer, data text)",
  );
  db.run(
    "create table part (id text primary key, message text, rank text, data text)",
  );
  const insertMessage = db.query(
    "insert into message values (?, ?, ?, ?) on conflict(id) do update set data=excluded.data",
  );
  const insertPart = db.query(
    "insert into part values (?, ?, ?, ?) on conflict(id) do update set data=excluded.data",
  );
  const messageRows = db.query(
    "select id, data from message where session=? order by created,id",
  );
  const partRows = db.query(
    "select data from part where message=? order by rank",
  );
  let lastTimestamp = 0;
  let counter = 0;
  const hostID = (prefix: "msg" | "prt") => {
    const now = clock();
    if (now !== lastTimestamp) {
      lastTimestamp = now;
      counter = 0;
    }
    counter += 1;
    const ordinal = (
      (BigInt(now) * 0x1000n + BigInt(counter))
      & ((1n << 48n) - 1n)
    )
      .toString(16)
      .padStart(12, "0");
    const suffix = Array.from(
      crypto.getRandomValues(new Uint8Array(14)),
      byte => ID_ALPHABET[byte % ID_ALPHABET.length],
    ).join("");
    return `${prefix}_${ordinal}${suffix}`;
  };
  const putMessage = (info: Message, created = info.time?.created ?? 0) => {
    insertMessage.run(info.id, info.sessionID, created, JSON.stringify(info));
  };
  const putPart = (part: Part, rank: string) => {
    insertPart.run(part.id, part.messageID, rank, JSON.stringify(part));
  };
  // Replaces the stored message, parts included.
  const write = (
    message: MessageWithParts,
    created = message.info.time?.created ?? 0,
  ) => {
    db.query("delete from part where message=?").run(message.info.id);
    putMessage(message.info, created);
    message.parts.forEach((part, index) =>
      putPart(part, String(index).padStart(8, "0")),
    );
  };
  source.forEach((message, index) => write(message, index));
  const read = (sessionID: string) =>
    (messageRows.all(sessionID) as Array<{ id: string; data: string }>).map(
      row => ({
        info: JSON.parse(row.data) as Message,
        parts: (partRows.all(row.id) as Array<{ data: string }>).map(
          part => JSON.parse(part.data) as Part,
        ),
      }),
    );
  // Session.fork: copies in source read order, minting a fresh ascending ID per
  // message and part. The cut is exclusive and found by message ID.
  const fork = (sourceID: string, forkID: string, messageID?: string) => {
    const messages = read(sourceID);
    const target = messageID
      ? messages.findIndex(item => item.info.id === messageID)
      : messages.length;
    const ids = new Map<string, string>();
    for (const item of messages.slice(
      0,
      target < 0 ? messages.length : target,
    )) {
      const id = hostID("msg");
      ids.set(item.info.id, id);
      const parentID =
        item.info.role === "assistant" && item.info.parentID
          ? ids.get(item.info.parentID)
          : undefined;
      putMessage({
        ...item.info,
        sessionID: forkID,
        id,
        ...(parentID ? { parentID } : {}),
      } as Message);
      for (const part of item.parts) {
        const copy = {
          ...part,
          id: hostID("prt"),
          messageID: id,
          sessionID: forkID,
        } as Part;
        if (copy.type === "compaction" && copy.tail_start_id)
          copy.tail_start_id = ids.get(copy.tail_start_id);
        putPart(copy, copy.id);
      }
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
        putMessage(event.data.info as Message);
      } else if (event.type === "message.part.updated.1") {
        const part = event.data.part as Part;
        putPart(part, part.id);
      } else throw new Error("unexpected replay type");
    });
  };
  return {
    read,
    write,
    fork,
    replay,
    // The host orders parts by ID, so an inserted part ranks by its own ID.
    updatePart(part: Part) {
      const existing = db
        .query("select rank from part where id=?")
        .get(part.id) as { rank: string } | null;
      putPart(part, existing?.rank ?? part.id);
    },
    deleteMessage(sessionID: string, messageID: string) {
      db.query("delete from part where message=?").run(messageID);
      db.query("delete from message where session=? and id=?").run(
        sessionID,
        messageID,
      );
    },
    delete(sessionID: string) {
      db.query(
        "delete from part where message in (select id from message where session=?)",
      ).run(sessionID);
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

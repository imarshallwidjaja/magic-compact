import type { Message, Part, Session } from "@opencode-ai/sdk/v2";
import { randomBytes, randomUUID } from "node:crypto";
import { isDeepStrictEqual } from "node:util";
import type { V2Client } from "../api";
import { attemptWithinDeadline, CONTROL_OPERATION_TIMEOUT_MS } from "../util";
import { snapshotSourceMessages, type MessageWithParts } from "./plan";

export class ForkHistoryError extends Error {
  constructor() {
    super("Magic Compact chronological fork initialization failed.");
    this.name = "ForkHistoryError";
  }
}

// Already sanitized; callers must retain the owned ID needed for cleanup.
export class ForkCleanupError extends AggregateError {
  constructor(sourceID: string, forkID: string) {
    super(
      [
        new ForkHistoryError(),
        new Error(`Partial fork ${forkID} could not be deleted.`),
      ],
      `Magic Compact chronological fork initialization and cleanup failed. Source session ${sourceID} was left unchanged; delete partial fork ${forkID} manually.`,
    );
    this.name = "ForkCleanupError";
  }
}

type ReplayEvents = NonNullable<
  NonNullable<Parameters<V2Client["sync"]["replay"]>[0]>["events"]
>;

// OpenCode 1.18.30 reads row time_created/id, but native forks insert copied
// JSON time.created. An updated JSON timestamp can therefore reverse history.
export async function forkSession(
  v2: V2Client,
  source: Session,
  messages: MessageWithParts[],
  controlTimeoutMs = CONTROL_OPERATION_TIMEOUT_MS,
): Promise<Session> {
  const route = {
    directory: source.directory,
    ...(source.workspaceID ? { workspace: source.workspaceID } : {}),
  };
  const needsReplay = messages.some(
    (message, index) =>
      index > 0
      && message.info.time?.created < messages[index - 1]!.info.time?.created,
  );
  const sourceSnapshot = needsReplay
    ? snapshotSourceMessages(messages)
    : undefined;
  const fork = unwrap(
    await v2.session.fork({
      sessionID: source.id,
      ...route,
      // The native cut is exclusive. Keep host-owned session creation, title,
      // metadata and routing, but don't insert timestamp-reordered messages.
      ...(needsReplay ? { messageID: messages[0]!.info.id } : {}),
    }),
  );
  if (!fork.id || fork.id === source.id) throw new ForkHistoryError();

  try {
    const load = async (sessionID = fork.id) => {
      const outcome = await attemptWithinDeadline(
        signal =>
          v2.session.messages({ sessionID, ...route }, { signal }).then(unwrap),
        controlTimeoutMs,
      );
      if (outcome.kind !== "success")
        throw new Error("Fork history reload failed.");
      return outcome.value;
    };
    if (!needsReplay) {
      assertNativeFork(await load(), messages, fork.id);
      return fork;
    }
    if ((await load()).length !== 0)
      throw new Error("Fork cut did not yield empty history.");
    const expected = rewriteHistory(messages, source.id, fork.id);
    const events: ReplayEvents = [];
    const append = (type: string, data: Record<string, unknown>) => {
      events.push({
        id: `evt_${randomUUID()}`,
        aggregateID: fork.id,
        seq: events.length + 1,
        type,
        data: { sessionID: fork.id, ...data },
      });
    };
    // An empty native fork has only session.created.1 at seq 0. Replay uses
    // consecutive event sequences and rejects a competing writer's sequence.
    for (const [index, message] of expected.entries()) {
      // Seed the immutable row ordering key, then restore the exact JSON info.
      // Both updates are host operations; no source record or DB is patched.
      append("message.updated.1", {
        info: {
          ...message.info,
          time: { ...message.info.time, created: index },
        },
      });
      append("message.updated.1", { info: message.info });
      for (const part of message.parts) {
        append("message.part.updated.1", { part, time: Date.now() });
      }
    }
    // Like fork creation, replay is awaited directly: timing out could lose
    // ownership of a still-writing server operation during cleanup.
    const replay = unwrap(
      await v2.sync.replay({
        query_directory: route.directory,
        ...(route.workspace ? { workspace: route.workspace } : {}),
        body_directory: route.directory,
        events,
      }),
    );
    if (
      replay.sessionID !== fork.id
      || !isDeepStrictEqual(await load(), expected)
    ) {
      throw new Error("Fork history identity mismatch.");
    }
    // Replay copied planned records, not the live source. Recheck correspondence
    // before this fork can become a recovery backup; timing churn is normalized
    // separately from the exact persisted chronology/content check above.
    if (
      !isDeepStrictEqual(
        snapshotSourceMessages(await load(source.id)),
        sourceSnapshot,
      )
    ) {
      throw new Error("Fork source history changed.");
    }
    return fork;
  } catch {
    const failure = new ForkHistoryError();
    try {
      unwrap(await v2.session.delete({ sessionID: fork.id, ...route }));
    } catch {
      throw new ForkCleanupError(source.id, fork.id);
    }
    throw failure;
  }
}

function assertNativeFork(
  actual: MessageWithParts[],
  source: MessageWithParts[],
  forkID: string,
): void {
  if (actual.length !== source.length) throw new ForkHistoryError();
  const ids = new Map<string, string>();
  const messageIDs = new Set<string>();
  const partIDs = new Set<string>();
  const expected = source.map((message, index) => {
    const copy = actual[index]!;
    if (
      !copy.info.id
      || messageIDs.has(copy.info.id)
      || copy.info.sessionID !== forkID
      || copy.parts.length !== message.parts.length
    )
      throw new ForkHistoryError();
    messageIDs.add(copy.info.id);
    const parentID =
      message.info.role === "assistant"
        ? ids.get(message.info.parentID)
        : undefined;
    if (
      message.info.role === "assistant"
      && message.info.parentID !== undefined
      && !parentID
    )
      throw new ForkHistoryError();
    ids.set(message.info.id, copy.info.id);
    return {
      info: {
        ...message.info,
        id: copy.info.id,
        sessionID: forkID,
        ...(parentID ? { parentID } : {}),
      } as Message,
      parts: message.parts.map((part, partIndex) => {
        const copied = copy.parts[partIndex]!;
        if (
          !copied.id
          || partIDs.has(copied.id)
          || copied.sessionID !== forkID
          || copied.messageID !== copy.info.id
        )
          throw new ForkHistoryError();
        partIDs.add(copied.id);
        const rewritten = {
          ...part,
          id: copied.id,
          sessionID: forkID,
          messageID: copy.info.id,
        } as Part;
        if (
          rewritten.type === "compaction"
          && rewritten.tail_start_id !== undefined
        ) {
          const tail = ids.get(rewritten.tail_start_id);
          if (!tail || rewritten.tail_start_id === message.info.id)
            throw new ForkHistoryError();
          rewritten.tail_start_id = tail;
        }
        return rewritten;
      }),
    };
  });
  if (
    !isDeepStrictEqual(
      snapshotSourceMessages(actual),
      snapshotSourceMessages(expected),
    )
  )
    throw new ForkHistoryError();
}

function rewriteHistory(
  messages: MessageWithParts[],
  sourceID: string,
  forkID: string,
): MessageWithParts[] {
  const seenMessages = new Set<string>();
  const seenParts = new Set<string>();
  const ids = new Map<string, string>();
  const clock = (BigInt(Date.now()) * 0x1000n) & ((1n << 48n) - 1n);
  let counter = 0n;
  const id = (prefix: "msg" | "prt") => {
    const ordinal = ((clock + counter++) & ((1n << 48n) - 1n))
      .toString(16)
      .padStart(12, "0");
    const alphabet =
      "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    return `${prefix}_${ordinal}${Array.from(randomBytes(14), byte => alphabet[byte % alphabet.length]).join("")}`;
  };
  return structuredClone(messages).map(message => {
    const originalID = message.info.id;
    if (
      !originalID
      || message.info.sessionID !== sourceID
      || seenMessages.has(originalID)
      || !Number.isSafeInteger(message.info.time?.created)
      || message.info.time.created < 0
    ) {
      throw new Error("Invalid source message identity.");
    }
    seenMessages.add(originalID);
    const messageID = id("msg");
    const parentID =
      message.info.role === "assistant"
        ? ids.get(message.info.parentID)
        : undefined;
    if (
      message.info.role === "assistant"
      && message.info.parentID !== undefined
      && !parentID
    )
      throw new Error("Invalid source assistant parent.");
    ids.set(originalID, messageID);
    const info = {
      ...message.info,
      id: messageID,
      sessionID: forkID,
      ...(parentID ? { parentID } : {}),
    } as Message;
    const parts = message.parts.map(part => {
      if (
        !part.id
        || seenParts.has(part.id)
        || part.sessionID !== sourceID
        || part.messageID !== originalID
      ) {
        throw new Error("Invalid source part identity.");
      }
      seenParts.add(part.id);
      const rewritten = {
        ...part,
        id: id("prt"),
        sessionID: forkID,
        messageID,
      } as Part;
      if (
        rewritten.type === "compaction"
        && rewritten.tail_start_id !== undefined
      ) {
        const tail = ids.get(rewritten.tail_start_id);
        if (!tail || rewritten.tail_start_id === originalID)
          throw new Error("Invalid source compaction tail.");
        rewritten.tail_start_id = tail;
      }
      return rewritten;
    });
    return { info, parts };
  });
}

function unwrap<T>(response: { data?: T; error?: unknown }): T {
  if (response.error) throw response.error;
  if (response.data === undefined)
    throw new Error("Fork host operation failed.");
  return response.data;
}

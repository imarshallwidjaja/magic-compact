import type { Message, Part, Session } from "@opencode-ai/sdk/v2";
import { randomBytes, randomUUID } from "node:crypto";
import { isDeepStrictEqual } from "node:util";
import type { V2Client } from "../api";
import { markPluginMessage, type LeavesSessions } from "../policy/feedback";
import { ContentFreeError } from "../policy/successor";
import { attemptWithinDeadline, CONTROL_OPERATION_TIMEOUT_MS } from "../util";
import { snapshotSourceMessages, type MessageWithParts } from "./plan";

// Fixed, content-free causes shown to operators; never raw host bodies.
const FORK_FAILURE_DETAIL = {
  "fork-invalid": "host returned an invalid fork",
  "source-invalid": "source history was invalid",
  "read-timeout": "history read timed out",
  "read-failed": "history read failed",
  "read-invalid": "history read returned invalid data",
  "cut-not-empty": "fork was not empty",
  "replay-failed": "history replay failed",
  "replay-mismatch": "replayed fork did not match the plan",
  "source-drift": "source history changed",
} as const;

export type ForkFailureReason = keyof typeof FORK_FAILURE_DETAIL;
export type ForkFailureDetail = (typeof FORK_FAILURE_DETAIL)[ForkFailureReason];

export class ForkHistoryError extends ContentFreeError {
  constructor(readonly reason: ForkFailureReason) {
    const detail = FORK_FAILURE_DETAIL[reason];
    super(
      `Magic Compact chronological fork initialization failed (${detail}).`,
      detail,
    );
    this.name = "ForkHistoryError";
  }
}

// Already sanitized; callers must retain the owned ID needed for cleanup. The
// fork ID is `undeletedSessionIDs`, not a reason, so an aggregate of several
// failed forks names every one however many reasons it explains.
export class ForkCleanupError extends AggregateError implements LeavesSessions {
  readonly undeletedSessionIDs: readonly string[];

  constructor(sourceID: string, forkID: string, reason: ForkFailureReason) {
    const failure = new ForkHistoryError(reason);
    super(
      [failure],
      `Magic Compact chronological fork initialization and cleanup failed (${failure.detail}). Source session ${sourceID} was left unchanged; delete partial fork ${forkID} manually.`,
    );
    this.name = "ForkCleanupError";
    this.undeletedSessionIDs = [forkID];
    markPluginMessage(this);
  }
}

type ReplayEvents = NonNullable<
  NonNullable<Parameters<V2Client["sync"]["replay"]>[0]>["events"]
>;

// OpenCode 1.18.30 mints a native fork's message and part IDs one record at a
// time from the host clock and reads parts back in ID order, so a clock that
// steps backwards while the fork is copied reverses history. Messages also sort
// by a row creation time taken from the copied JSON. The host's copy therefore
// never populates a fork: the native fork is cut before the first planned
// message, so it holds only the session, and the planned history is written
// once through host replay with IDs minted here from a single clock reading.
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
  // Source identity and references are validated before the host is touched.
  const sourceSnapshot = classified("source-invalid", () =>
    snapshotSourceMessages(messages),
  );
  const plan = classified("source-invalid", () =>
    planHistory(messages, source.id),
  );
  const fork = unwrap(
    await v2.session.fork({
      sessionID: source.id,
      ...route,
      // The native cut is exclusive. It keeps host-owned session creation,
      // title, metadata and routing. An empty history has nothing to cut.
      ...(messages.length > 0 ? { messageID: messages[0]!.info.id } : {}),
    }),
  );
  if (!fork.id || fork.id === source.id)
    throw new ForkHistoryError("fork-invalid");

  try {
    const load = (sessionID = fork.id) =>
      loadHistory(v2, sessionID, route, controlTimeoutMs);
    if ((await load()).length !== 0)
      throw new ForkHistoryError("cut-not-empty");
    if (messages.length > 0) {
      const expected = plan(fork.id);
      // Like fork creation, replay is awaited directly: timing out could lose
      // ownership of a still-writing server operation during cleanup.
      await replayHistory(v2, fork.id, expected, route);
      if (!isDeepStrictEqual(await load(), expected))
        throw new ForkHistoryError("replay-mismatch");
    }
    // The fork holds planned records, not the live source. Recheck
    // correspondence before it becomes a generation or summary input; timing churn is
    // normalized separately from the exact persisted chronology/content check
    // above.
    const live = await load(source.id);
    if (
      !classified("source-drift", () =>
        isDeepStrictEqual(snapshotSourceMessages(live), sourceSnapshot),
      )
    ) {
      throw new ForkHistoryError("source-drift");
    }
    return fork;
  } catch (error) {
    // Only the replay request is unclassified here; reads, comparisons and
    // source validation throw ForkHistoryError themselves.
    const failure =
      error instanceof ForkHistoryError
        ? error
        : new ForkHistoryError("replay-failed");
    try {
      unwrap(await v2.session.delete({ sessionID: fork.id, ...route }));
    } catch {
      throw new ForkCleanupError(source.id, fork.id, failure.reason);
    }
    throw failure;
  }
}

// Comparison helpers read host-supplied records; unexpected shapes are the
// caller's named mismatch, not an unclassified error.
function classified<T>(reason: ForkFailureReason, test: () => T): T {
  try {
    return test();
  } catch (error) {
    throw error instanceof ForkHistoryError
      ? error
      : new ForkHistoryError(reason);
  }
}

export type HistoryRoute = { directory: string; workspace?: string };

// Reads are idempotent, so a timed-out or rejected request gets exactly one
// more attempt with a fresh deadline. A response that arrives is final:
// callers never reload or repair it.
export async function loadHistory(
  v2: V2Client,
  sessionID: string,
  route: HistoryRoute,
  controlTimeoutMs: number,
): Promise<MessageWithParts[]> {
  let reason: "read-timeout" | "read-failed" = "read-failed";
  for (let attempt = 0; attempt < 2; attempt++) {
    const outcome = await attemptWithinDeadline(
      signal =>
        v2.session
          .messages({ sessionID, ...route }, { signal })
          .then(response => {
            if (response.error) throw response.error;
            return response.data;
          }),
      controlTimeoutMs,
    );
    if (outcome.kind === "success") {
      if (!Array.isArray(outcome.value))
        throw new ForkHistoryError("read-invalid");
      return outcome.value;
    }
    reason = outcome.kind === "timed-out" ? "read-timeout" : "read-failed";
  }
  throw new ForkHistoryError(reason);
}

// Appends a planned history to an empty session through host replay. An empty
// native session has only session.created.1 at seq 0; replay uses consecutive
// event sequences and rejects a competing writer's sequence. The caller
// verifies the persisted result.
export async function replayHistory(
  v2: V2Client,
  sessionID: string,
  expected: MessageWithParts[],
  route: HistoryRoute,
): Promise<void> {
  const events: ReplayEvents = [];
  const append = (type: string, data: Record<string, unknown>) => {
    events.push({
      id: `evt_${randomUUID()}`,
      aggregateID: sessionID,
      seq: events.length + 1,
      type,
      data: { sessionID, ...data },
    });
  };
  for (const [index, message] of expected.entries()) {
    // Seed the immutable row ordering key, then restore the exact JSON info.
    // Both updates are host operations; no source record or DB is patched.
    append("message.updated.1", {
      info: { ...message.info, time: { ...message.info.time, created: index } },
    });
    append("message.updated.1", { info: message.info });
    for (const part of message.parts) {
      append("message.part.updated.1", { part, time: Date.now() });
    }
  }
  const replay = unwrap(
    await v2.sync.replay({
      query_directory: route.directory,
      ...(route.workspace ? { workspace: route.workspace } : {}),
      body_directory: route.directory,
      events,
    }),
  );
  if (replay.sessionID !== sessionID)
    throw new ForkHistoryError("replay-mismatch");
}

// Validates the source and mints the copy's message and part IDs: one clock
// reading plus a counter, so IDs ascend in planned order however the host clock
// moves. The copy takes its session once the target session exists.
export function planHistory(
  messages: MessageWithParts[],
  sourceID: string,
): (forkID: string) => MessageWithParts[] {
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
  const rewritten = structuredClone(messages).map(message => {
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
      const copy = { ...part, id: id("prt"), messageID } as Part;
      if (copy.type === "compaction" && copy.tail_start_id !== undefined) {
        const tail = ids.get(copy.tail_start_id);
        if (!tail || copy.tail_start_id === originalID)
          throw new Error("Invalid source compaction tail.");
        copy.tail_start_id = tail;
      }
      return copy;
    });
    return { info, parts };
  });
  return forkID =>
    rewritten.map(({ info, parts }) => ({
      info: { ...info, sessionID: forkID } as Message,
      parts: parts.map(part => ({ ...part, sessionID: forkID }) as Part),
    }));
}

function unwrap<T>(response: { data?: T; error?: unknown }): T {
  if (response.error) throw response.error;
  if (response.data === undefined)
    throw new Error("Fork host operation failed.");
  return response.data;
}

import type {
  Message,
  Part,
  PermissionRequest,
  QuestionRequest,
  Session,
  SessionStatus,
  TextPart,
} from "@opencode-ai/sdk/v2";
import { isDeepStrictEqual } from "node:util";
import type { V2Client } from "../api";
import {
  commitPlannedOmission,
  copyCache,
  deleteCache,
  type PlannedOmission,
} from "../storage/omission";
import {
  deleteStats,
  readStats,
  writeStats,
  type ConversationStats,
} from "../storage/stats";
import { attemptWithinDeadline } from "../util";
import {
  ForkHistoryError,
  loadHistory,
  planHistory,
  replayHistory,
  type HistoryRoute,
} from "./fork";
import { snapshotSourceMessages, type MessageWithParts } from "./plan";
import {
  SourceChangedError,
  SourceNotAtRestError,
  successorFailure,
  SuccessorVerificationError,
  type SuccessorCommand,
  type SuccessorStage,
} from "../policy/successor";

export {
  SourceChangedError,
  SourceNotAtRestError,
  SuccessorFailure,
  withExclusiveSource,
  type SuccessorCommand,
} from "../policy/successor";

// Successor lineage stored on the new session. `ready` turns true only after
// the history, caches and stats are verified and the source was rechecked.
export type SuccessorMetadata = {
  compactionCount: number;
  successorOf: string;
  operation: "compact" | "trim";
  createdAt: number;
  ready: boolean;
};

export function sessionRoute(session: Session): HistoryRoute {
  return {
    directory: session.directory,
    ...(session.workspaceID ? { workspace: session.workspaceID } : {}),
  };
}

export type SourceState = {
  session: Session;
  messages: MessageWithParts[];
};

// Reads the complete source and refuses every state OpenCode 1.18.30 can show
// for work that is still in flight. A status this version does not define is
// treated as active.
export async function readSourceAtRest(
  v2: V2Client,
  command: SuccessorCommand,
  sessionID: string,
  controlTimeoutMs: number,
): Promise<SourceState> {
  const read = async <T>(
    request: (signal: AbortSignal) => Promise<{ data?: T; error?: unknown }>,
  ): Promise<T> => {
    const outcome = await attemptWithinDeadline(
      signal =>
        request(signal).then(response => {
          if (response.error || response.data === undefined)
            throw new Error("host read failed");
          return response.data;
        }),
      controlTimeoutMs,
    );
    if (outcome.kind !== "success") {
      throw new Error(
        `${command} could not read the session state it needs to check that the session is at rest. The session was left unchanged.`,
      );
    }
    return outcome.value;
  };
  const session = await read(signal =>
    v2.session.get({ sessionID }, { signal }),
  );
  if (session.parentID) {
    throw new Error(
      `${command} runs only on a root session; this is a subagent session. Run it from the parent session.`,
    );
  }
  const notAtRest = (reason: string) =>
    new SourceNotAtRestError(command, reason);
  if (session.revert) {
    throw notAtRest(
      "it has a pending revert (use /redo to restore the undone messages, or send a normal prompt to commit the discard)",
    );
  }
  if (session.time.compacting !== undefined) {
    throw notAtRest("a native compaction is running");
  }
  const route = sessionRoute(session);
  const messages = await read<MessageWithParts[]>(signal =>
    v2.session.messages({ sessionID, ...route }, { signal }),
  );
  if (messages.some(messageIsActive)) {
    throw notAtRest("a model response or tool call is still active");
  }

  const statuses = await read<Record<string, SessionStatus>>(signal =>
    v2.session.status(route, { signal }),
  );
  const tree = new Set([sessionID]);
  const pending = [sessionID];
  while (pending.length > 0) {
    const parent = pending.shift()!;
    const children = await read<Session[]>(signal =>
      v2.session.children({ sessionID: parent, ...route }, { signal }),
    );
    for (const child of children) {
      if (tree.has(child.id)) continue;
      tree.add(child.id);
      pending.push(child.id);
    }
  }
  for (const id of tree) {
    const status: unknown = statuses[id];
    if (status === undefined) continue;
    const type = (status as { type?: unknown }).type;
    if (type === "idle") continue;
    throw notAtRest(
      type === "busy" || type === "retry"
        ? id === sessionID
          ? "it is running"
          : "one of its subagent sessions is running"
        : "it reports a session state this version does not support",
    );
  }
  const permissions = await read<PermissionRequest[]>(signal =>
    v2.permission.list(route, { signal }),
  );
  if (permissions.some(request => tree.has(request.sessionID))) {
    throw notAtRest("a permission request is waiting for an answer");
  }
  const questions = await read<QuestionRequest[]>(signal =>
    v2.question.list(route, { signal }),
  );
  if (questions.some(request => tree.has(request.sessionID))) {
    throw notAtRest("a question is waiting for an answer");
  }
  const undelivered = undeliveredBackgroundTask(messages);
  if (undelivered !== undefined) {
    throw notAtRest(
      undelivered === ""
        ? "a background subagent cannot be matched to its result"
        : `background subagent session ${undelivered} has not delivered its result to this session (OpenCode 1.18.30 never delivers a cancelled one)`,
    );
  }
  return { session, messages };
}

// A background `task` (OpenCode 1.18.30 behind its experimental background
// subagent flag) returns at once with a `running` result whose metadata says
// `background: true`, and delivers the real result later as a synthetic text
// part of a new user message in the calling session that starts
// `<task id="CHILD" state="completed">` or `state="error"`. That covers a call
// made with `background: true` and a foreground call promoted to the
// background. Until that message is stored a successor would lose the result,
// so every started job needs its own later delivery. A message to a job that
// is still running (`Background task updated`) starts nothing new. A cancelled
// job is never delivered and the host lists no jobs publicly, so it stays
// undelivered. Returns the first undelivered child session ID, an empty string
// for a background call without one, or undefined when every job delivered.
function undeliveredBackgroundTask(
  messages: MessageWithParts[],
): string | undefined {
  const pending = new Map<string, number>();
  for (const message of messages) {
    for (const part of message.parts) {
      if (part.type === "tool") {
        if (
          part.tool !== "task"
          || part.state.status !== "completed"
          || part.state.metadata?.["background"] !== true
        ) {
          continue;
        }
        const childID = part.state.metadata["sessionId"];
        if (typeof childID !== "string") return "";
        if (
          part.state.output.startsWith(
            `<task id="${childID}" state="running">\n<summary>Background task updated</summary>`,
          )
        ) {
          continue;
        }
        pending.set(childID, (pending.get(childID) ?? 0) + 1);
      } else if (
        part.type === "text"
        && part.synthetic === true
        && message.info.role === "user"
      ) {
        const childID =
          /^<task id="([^"]+)" state="(?:completed|error)">\n/u.exec(
            part.text,
          )?.[1];
        const count = childID === undefined ? 0 : (pending.get(childID) ?? 0);
        if (childID !== undefined && count > 0) pending.set(childID, count - 1);
      }
    }
  }
  return [...pending].find(([, count]) => count > 0)?.[0];
}

function messageIsActive(message: MessageWithParts): boolean {
  const info = message.info;
  if (
    info.role === "assistant"
    && info.time.completed === undefined
    && info.error === undefined
  ) {
    return true;
  }
  return message.parts.some(
    part =>
      part.type === "tool"
      && part.state.status !== "completed"
      && part.state.status !== "error",
  );
}

// The fields a successor copies from its source. A change to any of them, or
// to the normalized history, means the successor would not continue the
// conversation the user left.
function continuationState(session: Session): unknown {
  return {
    directory: session.directory,
    workspaceID: session.workspaceID,
    title: session.title,
    agent: session.agent,
    model: session.model,
    permission: session.permission,
    metadata: session.metadata,
  };
}

export function assertSourceUnchanged(
  command: SuccessorCommand,
  initial: SourceState,
  current: SourceState,
): void {
  let same: boolean;
  try {
    same =
      isDeepStrictEqual(
        continuationState(current.session),
        continuationState(initial.session),
      )
      && isDeepStrictEqual(
        snapshotSourceMessages(current.messages),
        snapshotSourceMessages(initial.messages),
      );
  } catch {
    same = false;
  }
  if (!same) throw new SourceChangedError(command);
}

// Closed edits applied to a clone of the exact source records. Nothing else in
// the source changes on its way into the successor.
export type SuccessorEdit =
  | { type: "prepend-part"; messageID: string; part: Part }
  | { type: "replace-part"; messageID: string; part: Part }
  | { type: "delete-part"; messageID: string; partID: string }
  | { type: "delete-message"; messageID: string }
  | { type: "append-message"; message: MessageWithParts };

export function applySuccessorEdits(
  source: MessageWithParts[],
  edits: SuccessorEdit[],
): MessageWithParts[] {
  const messages = structuredClone(source);
  const find = (messageID: string) => {
    const message = messages.find(item => item.info.id === messageID);
    if (!message) throw new Error("Successor edit names a missing message.");
    return message;
  };
  const partIndex = (message: MessageWithParts, partID: string) => {
    const index = message.parts.findIndex(part => part.id === partID);
    if (index === -1) throw new Error("Successor edit names a missing part.");
    return index;
  };
  for (const edit of edits) {
    if (edit.type === "append-message") {
      messages.push(structuredClone(edit.message));
      continue;
    }
    const message = find(edit.messageID);
    if (edit.type === "prepend-part") {
      if (message.parts.some(part => part.id === edit.part.id))
        throw new Error("Successor edit duplicates a part.");
      message.parts.unshift(structuredClone(edit.part));
    } else if (edit.type === "replace-part") {
      message.parts[partIndex(message, edit.part.id)] = structuredClone(
        edit.part,
      );
    } else if (edit.type === "delete-part") {
      message.parts.splice(partIndex(message, edit.partID), 1);
    } else {
      messages.splice(messages.indexOf(message), 1);
    }
  }
  return messages;
}

// A user message carrying only Magic-owned parts, appended after the copied
// history. It inherits the latest user message's agent and model so the
// successor's next prompt resolves exactly as the source's would have.
export function appendedNoticeMessage(
  source: SourceState,
  key: string,
  parts: Array<Omit<TextPart, "id" | "sessionID" | "messageID">>,
): MessageWithParts {
  const latestUser = source.messages.findLast(
    message => message.info.role === "user",
  )?.info;
  if (!latestUser || latestUser.role !== "user") {
    throw new Error("Successor notice needs a user message to inherit from.");
  }
  const messageID = `magic-successor-${key}`;
  // Persisted records are JSON, so absent fields stay absent rather than
  // becoming undefined keys the readback would not echo.
  const info = {
    id: messageID,
    sessionID: source.session.id,
    role: "user",
    time: { created: Date.now() },
    ...(latestUser.agent === undefined ? {} : { agent: latestUser.agent }),
    ...(latestUser.model === undefined ? {} : { model: latestUser.model }),
  } as Message;
  return {
    info,
    parts: parts.map(
      (part, index) =>
        ({
          ...part,
          id: `${messageID}-part-${index}`,
          sessionID: source.session.id,
          messageID,
        }) as Part,
    ),
  };
}

// Records cross the HTTP API as JSON; comparing their JSON form keeps absent
// and undefined fields equal.
function json(value: unknown): unknown {
  return value === undefined ? undefined : JSON.parse(JSON.stringify(value));
}

// Every session field another writer could change without touching history.
function actualHeader(session: Session): unknown {
  return json({
    parentID: session.parentID,
    projectID: session.projectID,
    directory: session.directory,
    path: session.path,
    workspaceID: session.workspaceID,
    title: session.title,
    agent: session.agent,
    model: session.model,
    permission: session.permission ?? [],
    metadata: session.metadata,
    share: session.share,
    revert: session.revert,
    archived: session.time?.archived,
    compacting: session.time?.compacting,
  });
}

// The candidate's header as this command wrote it: the continuation state
// copied from the source, the preparing or final title and lineage entry, and
// the project, path and share the host assigned at creation. Nothing in it may
// have been touched by another writer before the ready update or before the
// candidate is deleted.
function expectedHeader(
  source: Session,
  created: Session,
  metadata: Record<string, unknown>,
  ready: boolean,
): unknown {
  return actualHeader({
    ...source,
    parentID: undefined,
    projectID: created.projectID,
    path: created.path,
    title: ready ? source.title : `${source.title} (preparing)`,
    metadata,
    share: created.share,
    revert: undefined,
    time: { created: 0, updated: 0 },
  });
}

function headerMatches(
  session: Session | undefined,
  expected: unknown,
): boolean {
  return (
    session !== undefined && isDeepStrictEqual(actualHeader(session), expected)
  );
}

export type SuccessorRequest = {
  command: SuccessorCommand;
  source: SourceState;
  history: MessageWithParts[];
  operation: SuccessorMetadata["operation"];
  compactionCount: number;
  omissions: PlannedOmission[];
  tokensPruned: number;
  mediaTokensProjected: number;
  // Rechecks the source after the candidate is complete; throws on drift.
  recheckSource: () => Promise<void>;
  controlTimeoutMs: number;
};

export type Successor = {
  session: Session;
  messages: MessageWithParts[];
};

// Creates a new root session holding the edited history, copies the caches the
// history needs, rechecks the source and the candidate's own header, and only
// then marks the successor ready. The source session and its children are
// never written.
export async function createSuccessor(
  v2: V2Client,
  request: SuccessorRequest,
): Promise<Successor> {
  const { command, source } = request;
  const sourceID = source.session.id;
  const route = sessionRoute(source.session);
  let plan: (sessionID: string) => MessageWithParts[];
  try {
    plan = planHistory(request.history, sourceID);
  } catch (error) {
    throw successorFailure(command, error, { stage: "plan", sourceID });
  }
  const metadata: SuccessorMetadata = {
    compactionCount: request.compactionCount,
    successorOf: sourceID,
    operation: request.operation,
    createdAt: Date.now(),
    ready: false,
  };
  const sessionMetadata = (ready: boolean) => ({
    ...source.session.metadata,
    magicCompact: { ...metadata, ready },
  });
  let created: Session | undefined;
  try {
    const response = await v2.session.create({
      ...route,
      ...(source.session.workspaceID
        ? { workspaceID: source.session.workspaceID }
        : {}),
      // Marks a candidate that is not ready yet in session lists.
      title: `${source.session.title} (preparing)`,
      ...(source.session.agent ? { agent: source.session.agent } : {}),
      ...(source.session.model ? { model: source.session.model } : {}),
      ...(source.session.permission
        ? { permission: source.session.permission }
        : {}),
      metadata: sessionMetadata(false),
    });
    created = response.data;
    if (response.error || !created) throw new Error("create failed");
  } catch (error) {
    throw successorFailure(command, error, { stage: "create", sourceID });
  }
  const successor = created;
  const preparing = expectedHeader(
    source.session,
    successor,
    sessionMetadata(false),
    false,
  );
  let expected: MessageWithParts[] = [];
  let readyAttempted = false;
  let stage: SuccessorStage = "verify";
  try {
    if (
      !successor.id
      || successor.id === sourceID
      || successor.parentID !== undefined
    ) {
      throw new ForkHistoryError("fork-invalid");
    }
    if (!headerMatches(successor, preparing)) {
      throw new SuccessorVerificationError(
        "imported session did not match the source",
      );
    }
    const load = () =>
      loadHistory(v2, successor.id, route, request.controlTimeoutMs);
    if ((await load()).length !== 0)
      throw new ForkHistoryError("cut-not-empty");
    expected = plan(successor.id);
    stage = "import";
    if (expected.length > 0) {
      try {
        await replayHistory(v2, successor.id, expected, route);
      } catch (error) {
        throw error instanceof ForkHistoryError
          ? error
          : new ForkHistoryError("replay-failed");
      }
    }
    stage = "verify";
    if (!isDeepStrictEqual(await load(), expected))
      throw new ForkHistoryError("replay-mismatch");

    stage = "cache";
    await copyCache(sourceID, successor.id);
    for (const omission of request.omissions) {
      await commitPlannedOmission(successor.id, omission);
    }
    stage = "stats";
    await writeStats(
      successor.id,
      await successorStats(source, request, expected),
    );

    stage = "recheck";
    await request.recheckSource();
    stage = "verify";
    if (!isDeepStrictEqual(await load(), expected))
      throw new ForkHistoryError("replay-mismatch");
    // A snapshot just before the write, not a lock: the ready update replaces
    // the title and metadata, so a header another writer changed must stop it.
    const current = await readCandidate(
      v2,
      successor.id,
      route,
      request.controlTimeoutMs,
    );
    if (!headerMatches(current, preparing)) {
      throw new SuccessorVerificationError(
        "candidate was changed by another writer",
      );
    }
    stage = "ready";
    readyAttempted = true;
    const ready = await v2.session.update({
      sessionID: successor.id,
      ...route,
      title: source.session.title,
      metadata: sessionMetadata(true),
    });
    if (ready.error || !ready.data) throw new Error("ready update failed");
    if (
      !headerMatches(
        ready.data,
        expectedHeader(source.session, successor, sessionMetadata(true), true),
      )
    ) {
      throw new SuccessorVerificationError(
        "imported session did not match the source",
      );
    }
    return { session: ready.data, messages: expected };
  } catch (error) {
    // Once the ready update was sent the host may have applied it, so the
    // candidate may already be the user's successor: keep it and name it.
    const discarded =
      !readyAttempted
      && (await discardOwnedCandidate(v2, {
        candidateID: successor.id,
        route,
        planned: expected,
        header: preparing,
        controlTimeoutMs: request.controlTimeoutMs,
      }));
    throw successorFailure(command, error, {
      stage,
      sourceID,
      candidateID: successor.id,
      discarded,
      readyAttempted,
    });
  }
}

async function readCandidate(
  v2: V2Client,
  sessionID: string,
  route: HistoryRoute,
  controlTimeoutMs: number,
): Promise<Session | undefined> {
  const outcome = await attemptWithinDeadline(
    signal =>
      v2.session.get({ sessionID, ...route }, { signal }).then(response => {
        if (response.error || response.data === undefined)
          throw new Error("candidate read failed");
        return response.data;
      }),
    controlTimeoutMs,
  );
  return outcome.kind === "success" ? outcome.value : undefined;
}

// A history message this command could have written: the planned record, or
// the ordered-row seed the replay writes first (same record with its creation
// time replaced by its index) when the replay stopped partway.
function ownedMessage(
  message: MessageWithParts,
  planned: MessageWithParts,
  index: number,
): boolean {
  return (
    (isDeepStrictEqual(message.info, planned.info)
      || isDeepStrictEqual(message.info, {
        ...planned.info,
        time: { ...planned.info.time, created: index },
      }))
    && message.parts.length <= planned.parts.length
    && message.parts.every((part, partIndex) =>
      isDeepStrictEqual(part, planned.parts[partIndex]),
    )
  );
}

// Deletes the candidate only when a fresh read shows nothing but what this
// command wrote: the header it created, no child session, no activity or
// pending request, and a prefix of the planned history, so a partly applied
// replay still qualifies while any record, field or work another writer added
// does not. Anything unreadable is kept and reported.
async function discardOwnedCandidate(
  v2: V2Client,
  candidate: {
    candidateID: string;
    route: HistoryRoute;
    planned: MessageWithParts[];
    header: unknown;
    controlTimeoutMs: number;
  },
): Promise<boolean> {
  const { candidateID, route, planned, controlTimeoutMs } = candidate;
  const read = async <T>(
    request: (signal: AbortSignal) => Promise<{ data?: T; error?: unknown }>,
  ): Promise<T | undefined> => {
    const outcome = await attemptWithinDeadline(
      signal =>
        request(signal).then(response => {
          if (response.error || response.data === undefined)
            throw new Error("read failed");
          return response.data;
        }),
      controlTimeoutMs,
    );
    return outcome.kind === "success" ? outcome.value : undefined;
  };
  const session = await read(signal =>
    v2.session.get({ sessionID: candidateID, ...route }, { signal }),
  );
  if (!headerMatches(session, candidate.header)) return false;
  let history: MessageWithParts[];
  try {
    history = await loadHistory(v2, candidateID, route, controlTimeoutMs);
  } catch {
    return false;
  }
  if (
    history.length > planned.length
    || !history.every((message, index) =>
      ownedMessage(message, planned[index]!, index),
    )
  ) {
    return false;
  }
  const children = await read(signal =>
    v2.session.children({ sessionID: candidateID, ...route }, { signal }),
  );
  const statuses = await read(signal => v2.session.status(route, { signal }));
  const permissions = await read(signal =>
    v2.permission.list(route, { signal }),
  );
  const questions = await read(signal => v2.question.list(route, { signal }));
  if (
    children === undefined
    || children.length > 0
    || statuses === undefined
    || (statuses[candidateID] !== undefined
      && (statuses[candidateID] as { type?: unknown }).type !== "idle")
    || permissions === undefined
    || permissions.some(item => item.sessionID === candidateID)
    || questions === undefined
    || questions.some(item => item.sessionID === candidateID)
  ) {
    return false;
  }
  try {
    const deleted = await v2.session.delete({
      sessionID: candidateID,
      ...route,
    });
    if (deleted.error) return false;
  } catch {
    return false;
  }
  await deleteCache(candidateID).catch(() => undefined);
  await deleteStats(candidateID).catch(() => undefined);
  return true;
}

// Working-conversation stats carry the source totals forward. Every imported
// assistant message is already counted, so replayed history never adds a
// cached-savings increment; only new responses in the successor do.
async function successorStats(
  source: SourceState,
  request: SuccessorRequest,
  successorMessages: MessageWithParts[],
): Promise<ConversationStats> {
  const previous = await readStats(source.session.id);
  return {
    version: 1,
    sourceSessionId: source.session.id,
    rootSessionId: previous?.rootSessionId ?? source.session.id,
    totalTokensPruned:
      (previous?.totalTokensPruned ?? 0) + request.tokensPruned,
    cachedTokensSaved: previous?.cachedTokensSaved ?? 0,
    mediaTokensProjected: request.mediaTokensProjected,
    cachedMediaTokensSaved: previous?.cachedMediaTokensSaved ?? 0,
    processedAssistantMessageIds: successorMessages
      .filter(message => message.info.role === "assistant")
      .map(message => message.info.id),
  };
}

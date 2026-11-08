import type {
  Message,
  Part,
  PermissionRequest,
  QuestionRequest,
  Session,
  SessionStatus,
  TextPart,
  ToolPart,
} from "@opencode-ai/sdk/v2";
import { isDeepStrictEqual } from "node:util";
import type { V2Client } from "../api";
import { PluginMessageError } from "../policy/feedback";
import {
  successorLineage,
  successorMetadata,
  successorTitle,
  type SuccessorLineage,
} from "../policy/lineage";
import {
  abandonedDeliveryJob,
  settleDeliveries,
  type DeliveryRecord,
  type RecoveryRecord,
} from "../policy/recovery";
import {
  deleteCandidateStorage,
  SourceChangedError,
  SourceNotAtRestError,
  successorFailure,
  SuccessorVerificationError,
  undeliveredBackgroundRefusal,
  unfinishedHistoryRefusal,
  type CandidateDisposal,
  type CandidateKeptBy,
  type SourceLifecycle,
  type SuccessorCommand,
  type SuccessorStage,
  type UnfinishedHistoryRecord,
} from "../policy/successor";
import {
  commitPlannedOmission,
  copyCache,
  type PlannedOmission,
} from "../storage/omission";
import { writeSuccessorStats } from "../storage/stats";
import { attemptWithinDeadline, isRecord, jsonForm } from "../util";
import {
  ForkHistoryError,
  loadHistory,
  planHistory,
  replayHistory,
  sessionRoute,
  type HistoryRoute,
} from "./fork";
import { snapshotSourceMessages, type MessageWithParts } from "./plan";

export type SourceState = {
  session: Session;
  messages: MessageWithParts[];
};

// A source as `readSourceAtRest` or `readRecoverableSource` read it, with the
// recheck that reads it again the same way and throws `SourceChangedError` on
// drift. The successor runs the recheck just before its ready update.
export type CheckedSource = SourceState & {
  readonly recheck: () => Promise<void>;
};

// A host response that carried an error or no data. Only its HTTP status is
// kept, so a failure's details can name it; the body and error are dropped,
// and `classifyFailure` accepts only a status from 100 to 599.
class HostResponseError extends Error {
  constructor(
    message: string,
    readonly status: number | undefined,
  ) {
    super(message);
    this.name = "HostResponseError";
  }
}

type HostResponse<T> = {
  data?: T;
  error?: unknown;
  response?: { status?: number };
};

function responseData<T>(message: string, response: HostResponse<T>): T {
  if (response.error || response.data === undefined || response.data === null) {
    throw new HostResponseError(message, response.response?.status);
  }
  return response.data;
}

function readWithin<T>(
  request: (signal: AbortSignal) => Promise<HostResponse<T>>,
  controlTimeoutMs: number,
) {
  return attemptWithinDeadline(
    signal =>
      request(signal).then(response =>
        responseData("host read failed", response),
      ),
    controlTimeoutMs,
  );
}

// Reads the complete source and refuses every state OpenCode 1.18.30 can show
// for work that is still in flight. A status this version does not define is
// treated as active.
export function readSourceAtRest(
  v2: V2Client,
  command: SuccessorCommand,
  sessionID: string,
  controlTimeoutMs: number,
): Promise<CheckedSource> {
  return readSource(v2, command, sessionID, controlTimeoutMs, "at-rest");
}

// The source as `/magic-recover` reads it: every live check of
// `readSourceAtRest` still refuses, but unfinished stored records and
// undelivered background results are returned for the caller to close,
// because running the command declared their work abandoned.
export function readRecoverableSource(
  v2: V2Client,
  command: SuccessorCommand,
  sessionID: string,
  controlTimeoutMs: number,
): Promise<CheckedSource> {
  return readSource(v2, command, sessionID, controlTimeoutMs, "recovery");
}

async function readSource(
  v2: V2Client,
  command: SuccessorCommand,
  sessionID: string,
  controlTimeoutMs: number,
  lifecycle: SourceLifecycle,
): Promise<CheckedSource> {
  const read = async () => {
    const state = await readLiveSource(
      v2,
      command,
      sessionID,
      controlTimeoutMs,
    );
    // After the live checks, so work that is actually running is reported as
    // running; what remains is history the host never finished.
    if (lifecycle === "at-rest") assertHistoryFinished(command, state.messages);
    return state;
  };
  const state = await read();
  const initial = captureSnapshot(state, lifecycle);
  return {
    ...state,
    recheck: async () => {
      const current = captureSnapshot(await read(), lifecycle);
      if (
        initial === UNCOMPARABLE
        || current === UNCOMPARABLE
        || !isDeepStrictEqual(current, initial)
      ) {
        throw new SourceChangedError(command);
      }
    },
  };
}

// Refuses a history that holds an unfinished record or a background launch
// still owed its result.
export function assertHistoryFinished(
  command: SuccessorCommand,
  messages: MessageWithParts[],
): void {
  const unfinished = messages.flatMap(unfinishedRecord);
  if (unfinished.length > 0) {
    throw unfinishedHistoryRefusal(command, unfinished);
  }
  const first = undeliveredBackgroundTasks(messages)[0];
  if (first !== undefined) {
    throw first.childID === undefined
      ? new SourceNotAtRestError(
          command,
          "a background subagent cannot be matched to its result",
        )
      : undeliveredBackgroundRefusal(
          command,
          `background subagent session ${first.childID} has not delivered its result to this session (OpenCode 1.18.30 never delivers a cancelled one)`,
        );
  }
}

// The live checks both lifecycles share: a root session, no pending revert or
// native compaction, an idle subagent tree, and no pending permission request
// or question in it.
async function readLiveSource(
  v2: V2Client,
  command: SuccessorCommand,
  sessionID: string,
  controlTimeoutMs: number,
): Promise<SourceState> {
  const read = async <T>(
    request: (signal: AbortSignal) => Promise<HostResponse<T>>,
  ): Promise<T> => {
    const outcome = await readWithin(request, controlTimeoutMs);
    if (outcome.kind !== "success") {
      throw new PluginMessageError(
        `${command} could not read the session state it needs to check that the session is at rest. The session was left unchanged.`,
      );
    }
    return outcome.value;
  };
  const session = await read(signal =>
    v2.session.get({ sessionID }, { signal }),
  );
  if (session.parentID) {
    throw new PluginMessageError(
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
  return { session, messages };
}

// A background `task` (OpenCode 1.18.30 behind its experimental background
// subagent flag) returns at once with a `running` result whose metadata says
// `background: true`, and delivers the real result later as a synthetic text
// part of a new user message in the calling session that starts
// `<task id="CHILD" state="completed">` or `state="error"`. That covers a call
// made with `background: true` and a foreground call promoted to the
// background. Until that message is stored a successor would lose the result,
// so every started job needs its own later delivery (`settleDeliveries`). A
// message to a job that is still running (`Background task updated`) starts
// nothing new. A cancelled job is never delivered and the host lists no jobs
// publicly, so it stays undelivered until `/magic-recover` marks its launch.
// Returns every launch still owed a delivery in stored order. A launch without
// a child session ID can never be matched.
export type UndeliveredBackgroundTask = {
  messageID: string;
  part: ToolPart;
  childID: string | undefined;
};

export function undeliveredBackgroundTasks(
  messages: MessageWithParts[],
): UndeliveredBackgroundTask[] {
  return settleDeliveries(messages.flatMap(backgroundRecords)).flatMap(
    ({ launch, settled }) => (settled ? [] : [launch]),
  );
}

function backgroundRecords(
  message: MessageWithParts,
): DeliveryRecord<UndeliveredBackgroundTask>[] {
  return message.parts.flatMap(
    (part): DeliveryRecord<UndeliveredBackgroundTask>[] => {
      if (part.type === "tool") {
        if (
          part.tool !== "task"
          || part.state.status !== "completed"
          || part.state.metadata?.["background"] !== true
        ) {
          return [];
        }
        const sessionId = part.state.metadata["sessionId"];
        const childID = typeof sessionId === "string" ? sessionId : undefined;
        if (
          childID !== undefined
          && part.state.output.startsWith(
            `<task id="${childID}" state="running">\n<summary>Background task updated</summary>`,
          )
        ) {
          return [];
        }
        return [
          {
            type: "launch",
            launch: { messageID: message.info.id, part, childID },
            job: childID,
            mark: abandonedDeliveryJob(part.state.metadata),
          },
        ];
      }
      if (
        part.type !== "text"
        || part.synthetic !== true
        || message.info.role !== "user"
      ) {
        return [];
      }
      const childID =
        /^<task id="([^"]+)" state="(?:completed|error)">\n/u.exec(
          part.text,
        )?.[1];
      return childID === undefined ? [] : [{ type: "delivery", job: childID }];
    },
  );
}

// The host's own test of an unfinished record: an assistant message with
// neither a completion time nor an error, or a tool part that is neither
// completed nor an error.
export function isUnfinishedAssistant(info: Message): boolean {
  return (
    info.role === "assistant"
    && info.time.completed === undefined
    && info.error === undefined
  );
}

export function isUnfinishedTool(part: Part): part is ToolPart {
  return (
    part.type === "tool"
    && part.state.status !== "completed"
    && part.state.status !== "error"
  );
}

// The stored message's unfinished assistant response or tool calls, if any.
function unfinishedRecord(
  message: MessageWithParts,
): UnfinishedHistoryRecord[] {
  const info = message.info;
  const unfinished = isUnfinishedAssistant(info);
  const tools = message.parts.flatMap(part =>
    isUnfinishedTool(part)
      ? [{ name: part.tool, callID: part.callID, state: part.state.status }]
      : [],
  );
  if (!unfinished && tools.length === 0) return [];
  return [
    {
      messageID: info.id,
      kind: info.role === "assistant" ? "assistant" : "message",
      unfinished,
      ...(tools.length === 0
        ? {}
        : { tool: { ...tools[0]!, count: tools.length } }),
    },
  ];
}

// The fields a successor copies from its source. A change to any of them, or
// to the normalized history, means the successor would not continue the
// conversation the user left.
function continuationState(session: Session): unknown {
  return structuredClone({
    directory: session.directory,
    workspaceID: session.workspaceID,
    title: session.title,
    agent: session.agent,
    model: session.model,
    permission: session.permission,
    metadata: session.metadata,
  });
}

const UNCOMPARABLE = Symbol("uncomparable source");

// What the recheck compares: the continuation state and the history with
// timing, token and cost churn normalized away. Recovery also compares the
// lifecycle that normalization drops with the other timing fields: whether
// each assistant message and each timed part has finished. Recovery closes
// records by exactly that state, so a record the host finished meanwhile must
// stop it rather than be closed. A tool call's status stays in the normalized
// history. Records whose shape cannot be compared never match.
function captureSnapshot(
  state: SourceState,
  lifecycle: SourceLifecycle,
): unknown {
  try {
    return {
      continuation: continuationState(state.session),
      history: snapshotSourceMessages(state.messages),
      ...(lifecycle === "recovery"
        ? {
            finished: state.messages.map(message => ({
              completed:
                message.info.role === "assistant"
                && message.info.time.completed !== undefined,
              parts: message.parts.map(part =>
                "time" in part && isRecord(part.time)
                  ? (part.time as { end?: unknown }).end !== undefined
                  : null,
              ),
            })),
          }
        : {}),
    };
  } catch {
    return UNCOMPARABLE;
  }
}

// Closed edits applied to a clone of the exact source records. Nothing else in
// the source changes on its way into the successor.
export type SuccessorEdit =
  | { type: "prepend-part"; messageID: string; part: Part }
  | { type: "replace-part"; messageID: string; part: Part }
  | { type: "delete-part"; messageID: string; partID: string }
  | { type: "delete-message"; messageID: string }
  | { type: "replace-info"; info: Message }
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
    if (edit.type === "replace-info") {
      find(edit.info.id).info = structuredClone(edit.info);
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

// Every session field another writer could change without touching history.
function actualHeader(session: Session): unknown {
  return jsonForm({
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
    title: successorTitle(source.title, ready),
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
  // Its recheck runs after the candidate is complete.
  source: CheckedSource;
  history: MessageWithParts[];
  operation: SuccessorLineage["operation"];
  compactionCount: number;
  omissions: PlannedOmission[];
  tokensPruned: number;
  mediaTokensProjected: number;
  recovery?: RecoveryRecord[];
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
  const lineage = {
    compactionCount: request.compactionCount,
    successorOf: sourceID,
    operation: request.operation,
    createdAt: Date.now(),
    ...(request.recovery === undefined ? {} : { recovery: request.recovery }),
  };
  const sessionMetadata = (ready: boolean) =>
    successorMetadata(
      source.session.metadata,
      successorLineage(lineage, ready),
    );
  let created: Session | undefined;
  try {
    const response = await v2.session.create({
      ...route,
      ...(source.session.workspaceID
        ? { workspaceID: source.session.workspaceID }
        : {}),
      title: successorTitle(source.session.title, false),
      ...(source.session.agent ? { agent: source.session.agent } : {}),
      ...(source.session.model ? { model: source.session.model } : {}),
      ...(source.session.permission
        ? { permission: source.session.permission }
        : {}),
      metadata: sessionMetadata(false),
    });
    created = responseData("create failed", response);
  } catch (error) {
    // A rejected create returned no ID, so a session the host may still have
    // created cannot be named.
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
  let storageAttempted = false;
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
    storageAttempted = true;
    await copyCache(sourceID, successor.id);
    for (const omission of request.omissions) {
      await commitPlannedOmission(successor.id, omission);
    }
    stage = "stats";
    await writeSuccessorStats({
      sourceKey: sourceID,
      successorKey: successor.id,
      sourceSessionID: sourceID,
      tokensPruned: request.tokensPruned,
      mediaTokensProjected: request.mediaTokensProjected,
      assistantMessageIDs: expected
        .filter(message => message.info.role === "assistant")
        .map(message => message.info.id),
    });

    stage = "recheck";
    await source.recheck();
    stage = "verify";
    if (!isDeepStrictEqual(await load(), expected))
      throw new ForkHistoryError("replay-mismatch");
    // A snapshot just before the write, not a lock: the ready update replaces
    // the title and metadata, so a header another writer changed must stop it.
    const current = await readWithin(
      signal =>
        v2.session.get({ sessionID: successor.id, ...route }, { signal }),
      request.controlTimeoutMs,
    );
    if (
      current.kind !== "success"
      || !headerMatches(current.value, preparing)
    ) {
      throw new SuccessorVerificationError(
        "candidate was changed by another writer",
      );
    }
    stage = "ready";
    readyAttempted = true;
    const ready = responseData(
      "ready update failed",
      await v2.session.update({
        sessionID: successor.id,
        ...route,
        title: source.session.title,
        metadata: sessionMetadata(true),
      }),
    );
    if (
      !headerMatches(
        ready,
        expectedHeader(source.session, successor, sessionMetadata(true), true),
      )
    ) {
      throw new SuccessorVerificationError(
        "imported session did not match the source",
      );
    }
    return { session: ready, messages: expected };
  } catch (error) {
    if (!successor.id) {
      throw successorFailure(command, error, { stage, sourceID });
    }
    // Once the ready update was sent the host may have applied it, so the
    // candidate may already be the user's successor: keep it and name it.
    const disposal: CandidateDisposal = readyAttempted
      ? { outcome: "ready-attempted" }
      : await discardOwnedCandidate(v2, {
          candidateID: successor.id,
          route,
          planned: expected,
          header: preparing,
          storageAttempted,
          controlTimeoutMs: request.controlTimeoutMs,
        });
    throw successorFailure(command, error, {
      stage,
      sourceID,
      candidate: { id: successor.id, disposal },
    });
  }
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
// does not. Anything unreadable, and a deletion the host does not confirm,
// keeps the candidate. A candidate the host reports missing is absent, which
// proves nothing about its deletion, so its local storage stays.
async function discardOwnedCandidate(
  v2: V2Client,
  candidate: {
    candidateID: string;
    route: HistoryRoute;
    planned: MessageWithParts[];
    header: unknown;
    storageAttempted: boolean;
    controlTimeoutMs: number;
  },
): Promise<CandidateDisposal> {
  const { candidateID, route, planned, controlTimeoutMs } = candidate;
  const kept = (keptBy: CandidateKeptBy): CandidateDisposal => ({
    outcome: "kept",
    keptBy,
  });
  const read = async <T>(
    request: (signal: AbortSignal) => Promise<HostResponse<T>>,
  ): Promise<T | undefined> => {
    const outcome = await readWithin(request, controlTimeoutMs);
    return outcome.kind === "success" ? outcome.value : undefined;
  };
  const lookup = await readWithin(
    signal => v2.session.get({ sessionID: candidateID, ...route }, { signal }),
    controlTimeoutMs,
  );
  if (
    lookup.kind === "failed"
    && lookup.error instanceof HostResponseError
    && lookup.error.status === 404
  ) {
    return { outcome: "absent", storageAttempted: candidate.storageAttempted };
  }
  if (lookup.kind !== "success") return kept("read");
  if (!headerMatches(lookup.value, candidate.header)) return kept("record");
  let history: MessageWithParts[];
  try {
    history = await loadHistory(v2, candidateID, route, controlTimeoutMs);
  } catch {
    return kept("read");
  }
  if (
    history.length > planned.length
    || !history.every((message, index) =>
      ownedMessage(message, planned[index]!, index),
    )
  ) {
    return kept("history");
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
    || statuses === undefined
    || permissions === undefined
    || questions === undefined
  ) {
    return kept("read");
  }
  if (children.length > 0) return kept("children");
  if (
    statuses[candidateID] !== undefined
    && (statuses[candidateID] as { type?: unknown }).type !== "idle"
  ) {
    return kept("activity");
  }
  if (
    permissions.some(item => item.sessionID === candidateID)
    || questions.some(item => item.sessionID === candidateID)
  ) {
    return kept("pending");
  }
  // Directly awaited: a deletion the host rejected may still apply, so it is
  // never retried and the candidate is reported as kept.
  try {
    const deleted = await v2.session.delete({
      sessionID: candidateID,
      ...route,
    });
    if (deleted.error) return kept("delete");
  } catch {
    return kept("delete");
  }
  return deleteCandidateStorage(candidateID);
}

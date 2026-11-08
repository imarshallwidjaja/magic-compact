import type {
  InstructionEntryInfo,
  JsonValue,
  SessionInfo,
  SessionMessageInfo,
} from "@opencode/client";
import { randomBytes } from "node:crypto";
import { isDeepStrictEqual } from "node:util";
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
import type { V2Host } from "./host";

export type V2Message = SessionMessageInfo;

export type V2Source = {
  session: SessionInfo;
  messages: V2Message[];
  instructions: InstructionEntryInfo[];
};

// OpenCode 2 sessions keep their omission caches and stats apart from
// OpenCode 1 sessions in the same storage tree; neither host's IDs are
// assumed unique across both databases, and nothing is migrated.
export function v2StorageKey(sessionID: string): string {
  return `v2/${sessionID}`;
}

const PAGE_LIMIT = 200;

// A host request that failed or missed the control deadline. Its fixed text
// is the at-rest read refusal; the host's error is dropped, because it may
// carry bodies or paths.
class SourceReadError extends PluginMessageError {
  constructor(command: SuccessorCommand) {
    super(
      `${command} could not read the session state it needs to check that the session is at rest. The session was left unchanged.`,
    );
    this.name = "SourceReadError";
  }
}

async function within<T>(
  command: SuccessorCommand,
  controlTimeoutMs: number,
  request: (signal: AbortSignal) => Promise<T>,
): Promise<T> {
  const outcome = await attemptWithinDeadline(request, controlTimeoutMs);
  if (outcome.kind !== "success") throw new SourceReadError(command);
  return outcome.value;
}

// The complete history in stored order. Pages are read oldest first, and the
// result must equal the unsanitized export, so a host-side filter or a page
// boundary that loses or reorders records fails instead of being copied. The
// one record OpenCode 2.0.28 leaves out of an export (and drops from an
// import) is an assistant message without a completion time, such as a step
// the host stopped without closing; the pages keep it, so the at-rest check
// can name it.
export async function readV2History(
  host: V2Host,
  sessionID: string,
  read: <T>(request: (signal: AbortSignal) => Promise<T>) => Promise<T>,
): Promise<V2Message[]> {
  const messages: V2Message[] = [];
  // A cursor carries its own order and cannot be combined with one.
  let cursor: string | undefined;
  for (;;) {
    const page = await read(signal =>
      host.message.list(
        {
          sessionID,
          limit: PAGE_LIMIT,
          ...(cursor ? { cursor } : { order: "asc" as const }),
        },
        { signal },
      ),
    );
    messages.push(...page.data);
    cursor = page.cursor.next ?? undefined;
    if (page.data.length === 0 || !cursor) break;
  }
  const exported = await read(signal =>
    host.session.export({ sessionID, sanitize: false }, { signal }),
  );
  const exportable = messages.filter(
    message =>
      message.type !== "assistant" || message.time.completed !== undefined,
  );
  if (!isDeepStrictEqual(jsonForm(exported.messages), jsonForm(exportable))) {
    throw new Error("Session history pages did not match its export.");
  }
  return messages;
}

async function listChildren(
  host: V2Host,
  parentID: string,
  read: <T>(request: (signal: AbortSignal) => Promise<T>) => Promise<T>,
): Promise<SessionInfo[]> {
  const children: SessionInfo[] = [];
  let cursor: string | undefined;
  for (;;) {
    const page = await read(signal =>
      host.session.list(
        { parentID, limit: PAGE_LIMIT, ...(cursor ? { cursor } : {}) },
        { signal },
      ),
    );
    children.push(...page.data.filter(child => child.parentID === parentID));
    cursor = page.cursor.next ?? undefined;
    if (page.data.length === 0 || !cursor) break;
  }
  return children;
}

// A source as `readV2SourceAtRest` or `readV2RecoverableSource` read it, with
// the recheck that reads it again the same way and throws
// `SourceChangedError` on drift. The successor runs the recheck just before
// its ready update.
export type CheckedV2Source = V2Source & {
  readonly recheck: () => Promise<void>;
};

// An unfinished record or undelivered background launch in a subagent
// session's own history. Recovery never writes or copies a subagent session,
// so it names the record and leaves it as it is.
export type V2SubagentRecord = { sessionID: string; messageID: string };

// Reads the complete source and refuses every state OpenCode 2.0.28 can show
// for work that is still in flight or waiting to be delivered.
export async function readV2SourceAtRest(
  host: V2Host,
  command: SuccessorCommand,
  sessionID: string,
  controlTimeoutMs: number,
): Promise<CheckedV2Source> {
  return (
    await readV2Source(host, command, sessionID, controlTimeoutMs, "at-rest")
  ).source;
}

// The source as `/magic-recover` reads it: every live check of
// `readV2SourceAtRest` still refuses, including running background jobs, but
// unfinished stored records and undelivered background results are left for
// the caller, because running the command declared their work abandoned. Its
// recheck also compares the subagent records.
export function readV2RecoverableSource(
  host: V2Host,
  command: SuccessorCommand,
  sessionID: string,
  controlTimeoutMs: number,
): Promise<{ source: CheckedV2Source; subagentRecords: V2SubagentRecord[] }> {
  return readV2Source(host, command, sessionID, controlTimeoutMs, "recovery");
}

async function readV2Source(
  host: V2Host,
  command: SuccessorCommand,
  sessionID: string,
  controlTimeoutMs: number,
  lifecycle: SourceLifecycle,
): Promise<{ source: CheckedV2Source; subagentRecords: V2SubagentRecord[] }> {
  const read = () =>
    inspectV2Source(host, command, sessionID, controlTimeoutMs, lifecycle);
  // What the recheck compares: the continuation state and the complete
  // history, whose completion times OpenCode 2 stores, plus for recovery the
  // subagent records it lists.
  const snapshot = (inspection: Awaited<ReturnType<typeof read>>) =>
    jsonForm({
      continuation: continuationState(inspection.source),
      messages: inspection.source.messages,
      ...(lifecycle === "recovery"
        ? { subagentRecords: inspection.subagentRecords }
        : {}),
    });
  const inspection = await read();
  const initial = snapshot(inspection);
  return {
    source: {
      ...inspection.source,
      recheck: async () => {
        if (!isDeepStrictEqual(snapshot(await read()), initial)) {
          throw new SourceChangedError(command);
        }
      },
    },
    subagentRecords: inspection.subagentRecords,
  };
}

async function inspectV2Source(
  host: V2Host,
  command: SuccessorCommand,
  sessionID: string,
  controlTimeoutMs: number,
  lifecycle: SourceLifecycle,
): Promise<{ source: V2Source; subagentRecords: V2SubagentRecord[] }> {
  const read = <T>(request: (signal: AbortSignal) => Promise<T>) =>
    within(command, controlTimeoutMs, request);
  const notAtRest = (reason: string) =>
    new SourceNotAtRestError(command, reason);
  const session = await read(signal =>
    host.session.get({ sessionID }, { signal }),
  );
  if (session.parentID) {
    throw new PluginMessageError(
      `${command} runs only on a root session; this is a subagent session. Run it from the parent session.`,
    );
  }
  if (session.revert) {
    throw notAtRest(
      "it has a pending revert (restore or commit it from OpenCode first)",
    );
  }
  const messages = await readHistoryOrRefuse(host, command, sessionID, read);

  const tree = new Map<string, V2Message[]>([[sessionID, messages]]);
  const pending = [sessionID];
  while (pending.length > 0) {
    const parent = pending.shift()!;
    for (const child of await listChildren(host, parent, read)) {
      if (tree.has(child.id)) continue;
      tree.set(
        child.id,
        await readHistoryOrRefuse(host, command, child.id, read),
      );
      pending.push(child.id);
    }
  }

  const active = await read(signal => host.session.active({ signal }));
  const jobs: BackgroundJob[] = [];
  const subagentRecords: V2SubagentRecord[] = [];
  for (const [id, history] of tree) {
    const child = id !== sessionID;
    if (active[id] !== undefined) {
      throw notAtRest(
        child ? "one of its subagent sessions is running" : "it is running",
      );
    }
    // Not active, so what remains is history the host never finished.
    const background = backgroundJobs(history);
    jobs.push(...background.jobs);
    if (lifecycle === "at-rest") {
      assertV2HistoryFinished(command, history, child ? id : undefined);
    } else if (child) {
      subagentRecords.push(
        ...[
          ...history.flatMap(unfinishedRecord).map(record => record.messageID),
          ...background.undelivered.map(launch => launch.messageID),
        ].map(messageID => ({ sessionID: id, messageID })),
      );
    }
    // An abandoned job must have stopped too.
    jobs.push(
      ...background.undelivered.flatMap(launch =>
        launch.jobID === undefined
          ? []
          : [{ tool: launch.tool, id: launch.jobID }],
      ),
    );
    const inbox = await read(signal =>
      host.session.inbox.list({ sessionID: id }, { signal }),
    );
    if (inbox.length > 0) {
      throw notAtRest("a queued message is waiting to be delivered");
    }
    const permissions = await read(signal =>
      host.permission.list({ sessionID: id }, { signal }),
    );
    if (permissions.length > 0) {
      throw notAtRest("a permission request is waiting for an answer");
    }
    const forms = await read(signal =>
      host.session.form.list({ sessionID: id }, { signal }),
    );
    if (forms.length > 0) {
      throw notAtRest("a question is waiting for an answer");
    }
  }
  const shells = await read(signal =>
    host.shell.list(
      { location: { directory: session.location.directory } },
      { signal },
    ),
  );
  const runningShells = shells.data.filter(shell => shell.status === "running");
  if (
    runningShells.some(
      shell =>
        typeof shell.metadata["sessionID"] === "string"
        && tree.has(shell.metadata["sessionID"]),
    )
  ) {
    throw notAtRest("a background command is still running");
  }
  // A delivered job must also have stopped: a cancellation notice does not
  // prove the cancelled work has ended.
  if (
    jobs.some(job =>
      job.tool === "subagent"
        ? active[job.id] !== undefined
        : runningShells.some(shell => shell.id === job.id),
    )
  ) {
    throw notAtRest("a background command or subagent is still running");
  }
  const instructions = await read(signal =>
    host.session.instructions.entry.list({ sessionID }, { signal }),
  );
  return { source: { session, messages, instructions }, subagentRecords };
}

async function readHistoryOrRefuse(
  host: V2Host,
  command: SuccessorCommand,
  sessionID: string,
  read: <T>(request: (signal: AbortSignal) => Promise<T>) => Promise<T>,
): Promise<V2Message[]> {
  try {
    return await readV2History(host, sessionID, read);
  } catch (error) {
    // A failed read keeps its own refusal. Any other failure means the pages
    // and export disagreed or were malformed; the host's own error is
    // dropped, because it may carry bodies or paths.
    throw error instanceof SourceReadError
      ? error
      : new PluginMessageError(
          `${command} could not read a complete session history. The session was left unchanged.`,
        );
  }
}

// Refuses a history that holds an unfinished record or a background launch
// still owed its result. `subagentSessionID` names the subagent session whose
// history it is, when it is not the source itself. The at-rest read applies
// it to every session in the tree; recovery applies it to its own result.
export function assertV2HistoryFinished(
  command: SuccessorCommand,
  messages: V2Message[],
  subagentSessionID?: string,
): void {
  const unfinished = messages.flatMap(unfinishedRecord);
  if (unfinished.length > 0) {
    throw unfinishedHistoryRefusal(command, unfinished, subagentSessionID);
  }
  const [first] = backgroundJobs(messages).undelivered;
  if (first !== undefined) {
    throw first.jobID === undefined
      ? new SourceNotAtRestError(
          command,
          "a background command or subagent cannot be matched to its result",
        )
      : undeliveredBackgroundRefusal(
          command,
          "a background command or subagent has not delivered its result yet",
        );
  }
}

type V2Assistant = Extract<V2Message, { type: "assistant" }>;

// The host's own test of an unfinished record: an assistant message with
// neither a completion time nor an error, a tool call that is neither
// completed nor an error, or a compaction or shell record still running.
export function isUnfinishedV2Assistant(message: V2Assistant): boolean {
  return message.time.completed === undefined && message.error === undefined;
}

export function isUnfinishedV2Tool(
  item: V2Assistant["content"][number],
): boolean {
  return (
    item.type === "tool"
    && item.state.status !== "completed"
    && item.state.status !== "error"
  );
}

function unfinishedRecord(message: V2Message): UnfinishedHistoryRecord[] {
  switch (message.type) {
    case "assistant": {
      const unfinished = isUnfinishedV2Assistant(message);
      const tools = message.content.flatMap(item =>
        item.type === "tool" && isUnfinishedV2Tool(item)
          ? [{ name: item.name, callID: item.id, state: item.state.status }]
          : [],
      );
      if (!unfinished && tools.length === 0) return [];
      return [
        {
          messageID: message.id,
          kind: "assistant",
          unfinished,
          ...(tools.length === 0
            ? {}
            : { tool: { ...tools[0]!, count: tools.length } }),
        },
      ];
    }
    case "compaction":
    case "shell":
      return message.status === "running"
        ? [{ messageID: message.id, kind: message.type, unfinished: true }]
        : [];
    default:
      return [];
  }
}

type BackgroundJob = { tool: "shell" | "subagent"; id: string };

// A background shell or subagent returns at once with a result that says the
// job was still running, and delivers its real result later as a synthetic
// notification. That covers a call made with `background: true` and a
// foreground call the user detached afterwards (OpenCode 2.0.28 gives both the
// same running result and leaves the original input unchanged). Until the
// notification is stored, a successor would lose the result, so every such
// call must be matched by a notification for its job (`settleDeliveries`).
// Returns the jobs that owe no notification, which the caller checks have
// stopped (a cancellation notice is written when the cancel is accepted), and
// every launch still owed one; a launch without a job ID can never be linked.
const NOTIFICATION_STATES = new Set(["completed", "cancelled", "error"]);

export type V2BackgroundLaunch = {
  messageID: string;
  // The call's position in the message's content.
  index: number;
  tool: BackgroundJob["tool"];
  jobID: string | undefined;
};

// The job a notification message reports, as `tool:jobID`.
function notifiedJob(message: V2Message): string | undefined {
  if (message.type !== "synthetic" || !isRecord(message.metadata)) {
    return undefined;
  }
  const { source, shellID, childID, state } = message.metadata;
  if (typeof state === "string" && !NOTIFICATION_STATES.has(state)) {
    return undefined;
  }
  if (source === "shell" && typeof shellID === "string")
    return `shell:${shellID}`;
  if (source === "subagent" && typeof childID === "string")
    return `subagent:${childID}`;
  return undefined;
}

function backgroundRecords(
  message: V2Message,
): DeliveryRecord<V2BackgroundLaunch>[] {
  const notified = notifiedJob(message);
  if (notified !== undefined) return [{ type: "delivery", job: notified }];
  if (message.type !== "assistant") return [];
  return message.content.flatMap(
    (item, index): DeliveryRecord<V2BackgroundLaunch>[] => {
      if (
        item.type !== "tool"
        || item.state.status !== "completed"
        || (item.name !== "shell" && item.name !== "subagent")
      ) {
        return [];
      }
      const metadata = item.state.metadata ?? {};
      if (
        item.state.input["background"] !== true
        && metadata["status"] !== "running"
      ) {
        return [];
      }
      const tool = item.name;
      const id =
        item.name === "shell" ? metadata["shellID"] : metadata["sessionID"];
      const jobID = typeof id === "string" ? id : undefined;
      const job = (value: string | undefined) =>
        value === undefined ? undefined : `${tool}:${value}`;
      return [
        {
          type: "launch",
          launch: { messageID: message.id, index, tool, jobID },
          job: job(jobID),
          mark: job(abandonedDeliveryJob(metadata)),
        },
      ];
    },
  );
}

export function backgroundJobs(messages: V2Message[]): {
  jobs: BackgroundJob[];
  undelivered: V2BackgroundLaunch[];
} {
  const launches = settleDeliveries(messages.flatMap(backgroundRecords));
  return {
    jobs: launches.flatMap(({ launch, settled }) =>
      settled && launch.jobID !== undefined
        ? [{ tool: launch.tool, id: launch.jobID }]
        : [],
    ),
    undelivered: launches.flatMap(({ launch, settled }) =>
      settled ? [] : [launch],
    ),
  };
}

// The fields a successor copies from its source. A change to any of them, to
// the active instructions, or to the history means the successor would not
// continue the conversation the user left.
function continuationState(source: V2Source): unknown {
  const { session } = source;
  return {
    title: session.title,
    agent: session.agent,
    model: session.model,
    permissions: session.permissions,
    metadata: session.metadata,
    location: session.location,
    subpath: session.subpath,
    projectID: session.projectID,
    instructions: source.instructions,
  };
}

// Host-generated instruction-change notices describe deltas against the
// source's instruction baseline. The successor starts from a fresh baseline
// holding the current entries, so replaying old deltas would contradict it.
export function isInstructionNotice(message: V2Message): boolean {
  return (
    message.type === "system"
    && isRecord(message.metadata)
    && message.metadata["notice"] === "instructions"
  );
}

export type V2SuccessorRequest = {
  command: SuccessorCommand;
  // Its recheck runs after the candidate is complete.
  source: CheckedV2Source;
  // The edited history, still under source message IDs.
  messages: V2Message[];
  operation: SuccessorLineage["operation"];
  compactionCount: number;
  outcome: string;
  recovery?: RecoveryRecord[];
  omissions: PlannedOmission[];
  tokensPruned: number;
  mediaTokensProjected: number;
  controlTimeoutMs: number;
};

export type V2Successor = { session: SessionInfo; messages: V2Message[] };

const ID_ALPHABET =
  "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

// OpenCode 2 identifier shape: 12 hex digits of millisecond time times 4096
// plus a counter (bit-inverted for descending IDs), then 14 random characters.
export function mintV2ID(
  prefix: string,
  time: number,
  counter: number,
  descending: boolean,
): string {
  const current = BigInt(time) * 0x1000n + BigInt(counter);
  const value = descending ? ~current : current;
  const hex = Array.from({ length: 6 }, (_, index) =>
    Number((value >> BigInt(40 - 8 * index)) & 0xffn)
      .toString(16)
      .padStart(2, "0"),
  ).join("");
  const random = Array.from(
    randomBytes(14),
    byte => ID_ALPHABET[byte % ID_ALPHABET.length],
  ).join("");
  return `${prefix}${hex}${random}`;
}

// The history under fresh, ascending message IDs from one clock reading, in
// the given order. The import keeps array order; message IDs must be unique
// across the whole database, so no source ID is reused.
export function withFreshMessageIDs(
  messages: readonly V2Message[],
  time: number = Date.now(),
): V2Message[] {
  return messages.map(
    (message, index) =>
      ({
        ...structuredClone(message),
        id: mintV2ID("msg_", time, index + 1, false),
      }) as V2Message,
  );
}

// The candidate's native header as this command wrote it: the continuation
// state copied from the source, the preparing or final title, and the lineage
// entry, unarchived. A field another writer changed makes the candidate
// foreign.
function candidateHeader(session: SessionInfo): unknown {
  return jsonForm({
    parentID: session.parentID,
    fork: session.fork,
    revert: session.revert,
    agent: session.agent,
    model: session.model,
    permissions: session.permissions,
    location: session.location,
    subpath: session.subpath,
    projectID: session.projectID,
    metadata: session.metadata,
    title: session.title,
    archived: session.time.archived,
  });
}

function sortedEntries(entries: readonly InstructionEntryInfo[]): unknown {
  return jsonForm([...entries].sort((a, b) => a.key.localeCompare(b.key)));
}

// Creates a new root session holding the edited history, copies the current
// instructions and caches it needs, rechecks the source and the candidate, and
// only then marks the successor ready. The source session and its children are
// never written.
export async function createV2Successor(
  host: V2Host,
  request: V2SuccessorRequest,
): Promise<V2Successor> {
  const { command, source } = request;
  const sourceID = source.session.id;
  const read = <T>(operation: (signal: AbortSignal) => Promise<T>) =>
    within(command, request.controlTimeoutMs, operation);
  const createdAt = Date.now();
  const lineage = {
    compactionCount: request.compactionCount,
    successorOf: sourceID,
    operation: request.operation,
    createdAt,
    outcome: request.outcome,
    ...(request.recovery === undefined ? {} : { recovery: request.recovery }),
  };
  const successorID = mintV2ID("ses_", createdAt, 1, true);
  const expected = withFreshMessageIDs(
    request.messages.filter(message => !isInstructionNotice(message)),
  );
  const metadata = (ready: boolean) =>
    jsonForm(
      successorMetadata(
        source.session.metadata,
        successorLineage(lineage, ready),
      ),
    ) as Record<string, JsonValue>;
  const title = (ready: boolean) => successorTitle(source.session.title, ready);
  const info = {
    ...structuredClone(source.session),
    id: successorID,
    ...(source.session.title === undefined ? {} : { title: title(false) }),
    metadata: metadata(false),
    time: { created: createdAt, updated: createdAt },
  } as Record<string, unknown>;
  delete info["parentID"];
  delete info["fork"];
  delete info["revert"];
  const header = (ready: boolean) =>
    candidateHeader({
      ...source.session,
      parentID: undefined,
      fork: undefined,
      revert: undefined,
      metadata: metadata(ready),
      title: title(ready),
      time: { created: createdAt, updated: createdAt },
    });
  const owned = {
    candidateID: successorID,
    planned: expected,
    instructions: source.instructions,
    header: header(false),
    controlTimeoutMs: request.controlTimeoutMs,
  };

  let readyAttempted = false;
  let storageAttempted = false;
  let stage: SuccessorStage = "import";
  try {
    await host.session.import({
      info: info as never,
      messages: expected as never,
      location: source.session.location,
    });
    stage = "verify";
    await verifySuccessor(host, read, owned, false);

    stage = "instructions";
    for (const entry of source.instructions) {
      await read(signal =>
        host.session.instructions.entry.put(
          { sessionID: successorID, key: entry.key, value: entry.value },
          { signal },
        ),
      );
    }
    stage = "verify";
    await verifySuccessor(host, read, owned, true);

    stage = "cache";
    storageAttempted = true;
    const sourceKey = v2StorageKey(sourceID);
    const successorKey = v2StorageKey(successorID);
    await copyCache(sourceKey, successorKey);
    for (const omission of request.omissions) {
      await commitPlannedOmission(successorKey, omission);
    }
    stage = "stats";
    await writeSuccessorStats({
      sourceKey,
      successorKey,
      sourceSessionID: sourceID,
      tokensPruned: request.tokensPruned,
      mediaTokensProjected: request.mediaTokensProjected,
      assistantMessageIDs: expected
        .filter(message => message.type === "assistant")
        .map(message => message.id),
    });

    stage = "recheck";
    await source.recheck();
    // The snapshot just before the write, not a lock: the ready update
    // replaces the title and metadata, and the candidate's instructions are
    // part of the conversation it continues.
    stage = "verify";
    await verifySuccessor(host, read, owned, true);
    stage = "ready";
    readyAttempted = true;
    await read(signal =>
      host.session.update(
        {
          sessionID: successorID,
          ...(source.session.title === undefined
            ? {}
            : { title: source.session.title }),
          metadata: metadata(true),
        },
        { signal },
      ),
    );
    const session = await verifySuccessor(
      host,
      read,
      { ...owned, header: header(true) },
      true,
    );
    return { session, messages: expected };
  } catch (error) {
    // Once the ready update was sent the host may have applied it, so the
    // candidate may already be the user's successor: keep it and name it.
    const disposal: CandidateDisposal = readyAttempted
      ? { outcome: "ready-attempted" }
      : await discardOwnedCandidate(host, read, owned, storageAttempted);
    throw successorFailure(command, error, {
      stage,
      sourceID,
      candidate: { id: successorID, disposal },
    });
  }
}

type OwnedCandidate = {
  candidateID: string;
  // The history this command imported, under the candidate's message IDs.
  planned: V2Message[];
  // The source's active instruction entries the candidate may hold.
  instructions: InstructionEntryInfo[];
  header: unknown;
  controlTimeoutMs: number;
};

// Reads the candidate's record and history, and its instruction entries once
// they are copied, and requires each to equal what this command wrote.
async function verifySuccessor(
  host: V2Host,
  read: <T>(request: (signal: AbortSignal) => Promise<T>) => Promise<T>,
  candidate: OwnedCandidate,
  withInstructions: boolean,
): Promise<SessionInfo> {
  const { candidateID } = candidate;
  const session = await read(signal =>
    host.session.get({ sessionID: candidateID }, { signal }),
  );
  let messages: V2Message[];
  try {
    messages = await readV2History(host, candidateID, read);
  } catch {
    throw new SuccessorVerificationError("history read failed");
  }
  if (!isDeepStrictEqual(jsonForm(messages), jsonForm(candidate.planned))) {
    throw new SuccessorVerificationError(
      "imported history did not match the plan",
    );
  }
  if (
    session.id !== candidateID
    || !isDeepStrictEqual(candidateHeader(session), candidate.header)
  ) {
    throw new SuccessorVerificationError(
      "imported session did not match the source",
    );
  }
  if (withInstructions) {
    const entries = await read(signal =>
      host.session.instructions.entry.list(
        { sessionID: candidateID },
        { signal },
      ),
    );
    if (
      !isDeepStrictEqual(
        sortedEntries(entries),
        sortedEntries(candidate.instructions),
      )
    ) {
      throw new SuccessorVerificationError(
        "instructions did not match the source",
      );
    }
  }
  return session;
}

// Deletes the candidate only when a fresh read shows nothing but what this
// command wrote: the header it imported, no child session, no activity,
// queued message, permission request or form, only source instruction entries
// it could have put, and the planned history or none. The import is a single
// request, so its history is either the plan or empty; anything else means
// another writer touched it. Anything unreadable, and a removal the host does
// not confirm, keeps the candidate. A candidate the host reports missing, as
// after a failed import, is absent: the import may still be applied later, so
// nothing saved under its ID is removed.
async function discardOwnedCandidate(
  host: V2Host,
  read: <T>(request: (signal: AbortSignal) => Promise<T>) => Promise<T>,
  candidate: OwnedCandidate,
  storageAttempted: boolean,
): Promise<CandidateDisposal> {
  const { candidateID } = candidate;
  const kept = (keptBy: CandidateKeptBy): CandidateDisposal => ({
    outcome: "kept",
    keptBy,
  });
  const lookup = await attemptWithinDeadline(
    signal => host.session.get({ sessionID: candidateID }, { signal }),
    candidate.controlTimeoutMs,
  );
  if (
    lookup.kind === "failed"
    && isRecord(lookup.error)
    && lookup.error["_tag"] === "SessionNotFoundError"
  ) {
    return { outcome: "absent", storageAttempted };
  }
  if (lookup.kind !== "success") return kept("read");
  let removing = false;
  try {
    if (!isDeepStrictEqual(candidateHeader(lookup.value), candidate.header))
      return kept("record");
    const current = await readV2History(host, candidateID, read);
    if (
      current.length > 0
      && !isDeepStrictEqual(jsonForm(current), jsonForm(candidate.planned))
    ) {
      return kept("history");
    }
    const entries = await read(signal =>
      host.session.instructions.entry.list(
        { sessionID: candidateID },
        { signal },
      ),
    );
    const known = jsonForm(candidate.instructions) as InstructionEntryInfo[];
    if (
      !(jsonForm(entries) as InstructionEntryInfo[]).every(entry =>
        known.some(item => isDeepStrictEqual(item, entry)),
      )
    ) {
      return kept("instructions");
    }
    if ((await listChildren(host, candidateID, read)).length > 0)
      return kept("children");
    const active = await read(signal => host.session.active({ signal }));
    if (active[candidateID] !== undefined) return kept("activity");
    const [inbox, permissions, forms] = await Promise.all([
      read(signal =>
        host.session.inbox.list({ sessionID: candidateID }, { signal }),
      ),
      read(signal =>
        host.permission.list({ sessionID: candidateID }, { signal }),
      ),
      read(signal =>
        host.session.form.list({ sessionID: candidateID }, { signal }),
      ),
    ]);
    if (inbox.length + permissions.length + forms.length > 0)
      return kept("pending");
    // Directly awaited: abandoning a removal could leave a session behind,
    // and a removal the host rejected may still apply, so it is never
    // retried.
    removing = true;
    await host.session.remove({ sessionID: candidateID });
  } catch {
    return kept(removing ? "delete" : "read");
  }
  return deleteCandidateStorage(v2StorageKey(candidateID));
}

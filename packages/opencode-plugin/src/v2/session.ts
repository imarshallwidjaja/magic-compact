import type {
  InstructionEntryInfo,
  JsonValue,
  SessionInfo,
  SessionMessageInfo,
} from "@opencode/client";
import { randomBytes } from "node:crypto";
import { isDeepStrictEqual } from "node:util";
import {
  SourceChangedError,
  SourceNotAtRestError,
  successorFailure,
  SuccessorVerificationError,
  unfinishedHistoryRefusal,
  type SuccessorCommand,
  type SuccessorStage,
  type UnfinishedHistoryRecord,
} from "../policy/successor";
import {
  commitPlannedOmission,
  copyCache,
  deleteCache,
  type PlannedOmission,
} from "../storage/omission";
import { deleteStats, readStats, writeStats } from "../storage/stats";
import { attemptWithinDeadline, isRecord } from "../util";
import type { V2Host } from "./host";
import { PluginMessageError } from "../policy/feedback";

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

async function within<T>(
  command: SuccessorCommand,
  controlTimeoutMs: number,
  request: (signal: AbortSignal) => Promise<T>,
): Promise<T> {
  const outcome = await attemptWithinDeadline(request, controlTimeoutMs);
  if (outcome.kind !== "success") {
    throw new PluginMessageError(
      `${command} could not read the session state it needs to check that the session is at rest. The session was left unchanged.`,
    );
  }
  return outcome.value;
}

// The complete history in stored order. Pages are read oldest first, and the
// result must equal the unsanitized export, so a host-side filter or a page
// boundary that loses or reorders records fails instead of being copied.
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
  if (!isDeepStrictEqual(json(exported.messages), json(messages))) {
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

// Reads the complete source and refuses every state OpenCode 2.0.28 can show
// for work that is still in flight or waiting to be delivered.
export async function readV2SourceAtRest(
  host: V2Host,
  command: SuccessorCommand,
  sessionID: string,
  controlTimeoutMs: number,
): Promise<V2Source> {
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
  for (const [id, history] of tree) {
    const child = id !== sessionID;
    if (active[id] !== undefined) {
      throw notAtRest(
        child ? "one of its subagent sessions is running" : "it is running",
      );
    }
    // Not active, so what remains is history the host never finished.
    const unfinished = history.flatMap(unfinishedRecord);
    if (unfinished.length > 0) {
      throw unfinishedHistoryRefusal(
        command,
        unfinished,
        child ? id : undefined,
      );
    }
    const started = deliveredBackgroundJobs(history);
    if (started === undefined) {
      throw notAtRest(
        "a background command or subagent has not delivered its result yet",
      );
    }
    jobs.push(...started);
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
  return { session, messages, instructions };
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
    // The host's own error is dropped: it may carry bodies or paths.
    throw error instanceof Error && error.message.startsWith(command)
      ? error
      : new PluginMessageError(
          `${command} could not read a complete session history. The session was left unchanged.`,
        );
  }
}

function unfinishedRecord(message: V2Message): UnfinishedHistoryRecord[] {
  switch (message.type) {
    case "assistant": {
      const unfinished =
        message.time.completed === undefined && message.error === undefined;
      const tools = message.content.flatMap(item =>
        item.type === "tool"
        && item.state.status !== "completed"
        && item.state.status !== "error"
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
// call must be matched by the notification for its job. Returns every
// background job the history started, or undefined when one is undelivered.
// A cancellation notice is written when the cancel is accepted, so the caller
// still checks that each job has stopped.
const NOTIFICATION_STATES = new Set(["completed", "cancelled", "error"]);

function deliveredBackgroundJobs(
  messages: V2Message[],
): BackgroundJob[] | undefined {
  const jobs: BackgroundJob[] = [];
  const delivered = new Set<string>();
  for (const message of messages) {
    if (message.type !== "synthetic" || !isRecord(message.metadata)) continue;
    const { source, shellID, childID, state } = message.metadata;
    if (typeof state === "string" && !NOTIFICATION_STATES.has(state)) continue;
    if (source === "shell" && typeof shellID === "string")
      delivered.add(`shell:${shellID}`);
    if (source === "subagent" && typeof childID === "string")
      delivered.add(`subagent:${childID}`);
  }
  for (const message of messages) {
    if (message.type !== "assistant") continue;
    for (const item of message.content) {
      if (
        item.type !== "tool"
        || item.state.status !== "completed"
        || (item.name !== "shell" && item.name !== "subagent")
      ) {
        continue;
      }
      const metadata = item.state.metadata ?? {};
      if (
        item.state.input["background"] !== true
        && metadata["status"] !== "running"
      ) {
        continue;
      }
      const jobID =
        item.name === "shell" ? metadata["shellID"] : metadata["sessionID"];
      // A background call this version cannot link to its notification is
      // treated as undelivered.
      if (
        typeof jobID !== "string"
        || !delivered.has(`${item.name}:${jobID}`)
      ) {
        return undefined;
      }
      jobs.push({ tool: item.name as BackgroundJob["tool"], id: jobID });
    }
  }
  return jobs;
}

// The fields a successor copies from its source. A change to any of them, to
// the active instructions, or to the history means the successor would not
// continue the conversation the user left.
function continuationState(source: V2Source): unknown {
  const { session } = source;
  return json({
    title: session.title,
    agent: session.agent,
    model: session.model,
    permissions: session.permissions,
    metadata: session.metadata,
    location: session.location,
    subpath: session.subpath,
    projectID: session.projectID,
    instructions: source.instructions,
  });
}

export function assertV2SourceUnchanged(
  command: SuccessorCommand,
  initial: V2Source,
  current: V2Source,
): void {
  if (
    !isDeepStrictEqual(continuationState(current), continuationState(initial))
    || !isDeepStrictEqual(json(current.messages), json(initial.messages))
  ) {
    throw new SourceChangedError(command);
  }
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

// Successor lineage stored on the new session. `ready` turns true only after
// the history, instructions, caches and stats are verified and the source was
// rechecked. `outcome` is the command's result for clients that did not run it.
export type V2SuccessorMetadata = {
  compactionCount: number;
  successorOf: string;
  operation: "compact" | "trim";
  createdAt: number;
  ready: boolean;
  outcome: string;
};

export type V2SuccessorRequest = {
  command: SuccessorCommand;
  source: V2Source;
  // The edited history, still under source message IDs.
  messages: V2Message[];
  operation: V2SuccessorMetadata["operation"];
  compactionCount: number;
  outcome: string;
  omissions: PlannedOmission[];
  tokensPruned: number;
  mediaTokensProjected: number;
  // Rechecks the source after the candidate is complete; throws on drift.
  recheckSource: () => Promise<void>;
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
  return json({
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
  return json([...entries].sort((a, b) => a.key.localeCompare(b.key)));
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
  const lineage = (ready: boolean): V2SuccessorMetadata => ({
    compactionCount: request.compactionCount,
    successorOf: sourceID,
    operation: request.operation,
    createdAt,
    ready,
    outcome: request.outcome,
  });
  const successorID = mintV2ID("ses_", createdAt, 1, true);
  const expected = withFreshMessageIDs(
    request.messages.filter(message => !isInstructionNotice(message)),
  );
  const metadata = (ready: boolean) =>
    json({
      ...source.session.metadata,
      magicCompact: lineage(ready),
    }) as Record<string, JsonValue>;
  const title = (ready: boolean) =>
    source.session.title === undefined || ready
      ? source.session.title
      : `${source.session.title} (preparing)`;
  const info = {
    ...structuredClone(source.session),
    id: successorID,
    // Marks a candidate that is not ready yet in session lists.
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

  let importAttempted = false;
  let readyAttempted = false;
  let stage: SuccessorStage = "import";
  try {
    importAttempted = true;
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
    const sourceKey = v2StorageKey(sourceID);
    const successorKey = v2StorageKey(successorID);
    await copyCache(sourceKey, successorKey);
    for (const omission of request.omissions) {
      await commitPlannedOmission(successorKey, omission);
    }
    stage = "stats";
    const previous = await readStats(sourceKey);
    await writeStats(successorKey, {
      version: 1,
      sourceSessionId: sourceID,
      rootSessionId: previous?.rootSessionId ?? sourceID,
      totalTokensPruned:
        (previous?.totalTokensPruned ?? 0) + request.tokensPruned,
      cachedTokensSaved: previous?.cachedTokensSaved ?? 0,
      mediaTokensProjected: request.mediaTokensProjected,
      cachedMediaTokensSaved: previous?.cachedMediaTokensSaved ?? 0,
      // Imported responses are already counted; only new ones add savings.
      processedAssistantMessageIds: expected
        .filter(message => message.type === "assistant")
        .map(message => message.id),
    });

    stage = "recheck";
    await request.recheckSource();
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
    const discarded =
      importAttempted
      && !readyAttempted
      && (await discardOwnedCandidate(host, read, owned));
    throw successorFailure(command, error, {
      stage,
      sourceID,
      candidateID: successorID,
      discarded,
      readyAttempted,
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
  if (!isDeepStrictEqual(json(messages), json(candidate.planned))) {
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
// another writer touched it. Anything unreadable keeps the candidate.
async function discardOwnedCandidate(
  host: V2Host,
  read: <T>(request: (signal: AbortSignal) => Promise<T>) => Promise<T>,
  candidate: OwnedCandidate,
): Promise<boolean> {
  const { candidateID } = candidate;
  // A failed import may have created nothing.
  const lookup = await attemptWithinDeadline(
    signal => host.session.get({ sessionID: candidateID }, { signal }),
    candidate.controlTimeoutMs,
  );
  if (
    lookup.kind === "failed"
    && isRecord(lookup.error)
    && lookup.error["_tag"] === "SessionNotFoundError"
  ) {
    return true;
  }
  if (lookup.kind !== "success") return false;
  try {
    if (!isDeepStrictEqual(candidateHeader(lookup.value), candidate.header))
      return false;
    const current = await readV2History(host, candidateID, read);
    if (
      current.length > 0
      && !isDeepStrictEqual(json(current), json(candidate.planned))
    ) {
      return false;
    }
    const entries = await read(signal =>
      host.session.instructions.entry.list(
        { sessionID: candidateID },
        { signal },
      ),
    );
    const known = json(candidate.instructions) as InstructionEntryInfo[];
    if (
      !(json(entries) as InstructionEntryInfo[]).every(entry =>
        known.some(item => isDeepStrictEqual(item, entry)),
      )
    ) {
      return false;
    }
    if ((await listChildren(host, candidateID, read)).length > 0) return false;
    const active = await read(signal => host.session.active({ signal }));
    if (active[candidateID] !== undefined) return false;
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
    if (inbox.length + permissions.length + forms.length > 0) return false;
    // Directly awaited: abandoning a removal could leave a session behind.
    await host.session.remove({ sessionID: candidateID });
  } catch {
    return false;
  }
  await deleteCache(v2StorageKey(candidateID)).catch(() => undefined);
  await deleteStats(v2StorageKey(candidateID)).catch(() => undefined);
  return true;
}

// Records cross the HTTP API as JSON; comparing their JSON form keeps absent
// and undefined fields equal.
function json(value: unknown): unknown {
  return value === undefined ? undefined : JSON.parse(JSON.stringify(value));
}

import { isDeepStrictEqual } from "node:util";
import { getCompactionCount } from "../compact/session";
import {
  RECOVER_COMMAND_NAME,
  RECOVERED_RESPONSE_ERROR,
  RECOVERED_TOOL_ERROR,
  RECOVERY_CONTEXT_METADATA,
  recoveryContext,
  recoveryOutcome,
  unsupportedRecoveryRecord,
  withRecoveryMarker,
  type RecoveryRecord,
} from "../policy/recovery";
import { SourceChangedError, withExclusiveSource } from "../policy/successor";
import { readStats } from "../storage/stats";
import { CONTROL_OPERATION_TIMEOUT_MS } from "../util";
import type { V2Host } from "./host";
import {
  assertV2SourceUnchanged,
  backgroundJobs,
  createV2Successor,
  isUnfinishedV2Assistant,
  isUnfinishedV2Tool,
  readV2RecoverableSource,
  v2StorageKey,
  type V2Message,
  type V2Successor,
} from "./session";

const COMMAND = RECOVER_COMMAND_NAME;

type V2Assistant = Extract<V2Message, { type: "assistant" }>;
type V2Content = V2Assistant["content"][number];

export type V2RecoverOptions = { controlTimeoutMs?: number };

// The history with every unfinished record closed the way OpenCode 2.0.28
// closes an interrupted step, and the records it covers. An open assistant
// message gets what the host's failed step stores: an `aborted` error (as the
// host's "Step interrupted"), finish `error` and a completion time, here the
// latest time the message already records, never a later one. The host's
// import drops every assistant message without a completion time, an errored
// one included, so the copy of such a message keeps its error, finish and
// content and gains only that completion time (a `completion-time` record,
// since the message is already closed). Content stays in model input. A streaming or running tool call becomes an
// `aborted` error state that keeps its captured input and metadata (the host
// would drop a streaming call's partial input, so it moves under
// `metadata.magicCompact.recovered`); for a subagent call the error names its
// child session, as the host's does. A background shell or subagent call
// still owed its notification is marked as never delivered, naming its job.
// A running compaction or shell record has no interrupted form, so it is
// refused rather than guessed. Nothing is invented: no output, result or
// notification.
export function planV2Recovery(messages: V2Message[]): {
  messages: V2Message[];
  records: RecoveryRecord[];
} {
  const undelivered = backgroundJobs(messages).undelivered;
  const records: RecoveryRecord[] = [];
  const recovered = messages.map(message => {
    if (
      (message.type === "compaction" || message.type === "shell")
      && message.status === "running"
    ) {
      throw unsupportedRecoveryRecord(
        `${message.type} record`,
        message.id,
        message.type === "compaction"
          ? "OpenCode stores no interrupted form of an unfinished native checkpoint, so closing it would mean guessing its summary"
          : "OpenCode stores no interrupted form of a user shell command, so closing it would mean guessing how it ended",
      );
    }
    if (message.type !== "assistant") return message;
    const unfinished = isUnfinishedV2Assistant(message);
    if (unfinished) records.push({ kind: "response", messageID: message.id });
    else if (message.time.completed === undefined)
      records.push({ kind: "completion-time", messageID: message.id });
    const content = message.content.map((item, index): V2Content => {
      if (item.type !== "tool") return item;
      if (isUnfinishedV2Tool(item)) {
        records.push({
          kind: "tool-call",
          messageID: message.id,
          callID: item.id,
          tool: item.name,
          state: item.state.status,
        });
        return closedToolCall(item);
      }
      const launch = undelivered.find(
        entry => entry.messageID === message.id && entry.index === index,
      );
      // A launch is a completed call by construction.
      if (launch === undefined || item.state.status !== "completed")
        return item;
      if (launch.jobID === undefined) {
        throw unsupportedRecoveryRecord(
          `background ${launch.tool} call`,
          item.id,
          "it names no job, so its missing result cannot be recorded against one",
        );
      }
      records.push({
        kind: "undelivered",
        messageID: message.id,
        callID: item.id,
        tool: item.name,
        job: launch.jobID,
      });
      return {
        ...item,
        state: {
          ...item.state,
          metadata: withRecoveryMarker(item.state.metadata, {
            delivery: "abandoned",
            job: launch.jobID,
          }),
        },
      } as V2Content;
    });
    return {
      ...message,
      content,
      ...(message.time.completed === undefined
        ? { time: { ...message.time, completed: latestRecordedTime(message) } }
        : {}),
      ...(unfinished
        ? {
            finish: message.finish ?? "error",
            error: { type: "aborted", message: RECOVERED_RESPONSE_ERROR },
          }
        : {}),
    } as V2Message;
  });
  return { messages: recovered, records };
}

// The latest time a stored assistant message and its content record.
function latestRecordedTime(message: V2Assistant): number {
  const times: unknown[] = [message.time.created, message.time.streamed];
  for (const item of message.content) {
    if (item.type === "tool")
      times.push(item.time.created, item.time.ran, item.time.completed);
    if (item.type === "reasoning")
      times.push(item.time?.created, item.time?.completed);
  }
  return Math.max(
    ...times.filter((time): time is number => Number.isSafeInteger(time)),
  );
}

function closedToolCall(item: Extract<V2Content, { type: "tool" }>): V2Content {
  const { state } = item;
  if (state.status === "running") {
    const child = state.metadata["sessionID"];
    return {
      ...item,
      state: {
        status: "error",
        input: state.input,
        error: {
          type: "aborted",
          message:
            item.name === "subagent" && typeof child === "string"
              ? `${RECOVERED_TOOL_ERROR} (sessionID: ${child})`
              : RECOVERED_TOOL_ERROR,
        },
        metadata: withRecoveryMarker(state.metadata, {
          outcome: "unknown",
          state: "running",
        }),
      },
    } as V2Content;
  }
  if (state.status === "streaming") {
    return {
      ...item,
      state: {
        status: "error",
        input: {},
        error: { type: "aborted", message: RECOVERED_TOOL_ERROR },
        metadata: withRecoveryMarker(undefined, {
          outcome: "unknown",
          state: "streaming",
          input: state.input,
        }),
      },
    } as V2Content;
  }
  throw unsupportedRecoveryRecord(
    "tool call",
    item.id,
    "it is in a state this version does not define",
  );
}

// Returns the ready successor, or null when nothing needs recovery. The source
// session and its subagent sessions are never written.
export async function executeV2Recover(
  host: V2Host,
  sessionID: string,
  options: V2RecoverOptions = {},
): Promise<{ successor: V2Successor; outcome: string } | null> {
  const controlTimeoutMs =
    options.controlTimeoutMs ?? CONTROL_OPERATION_TIMEOUT_MS;
  return withExclusiveSource(COMMAND, sessionID, async () => {
    const initial = await readV2RecoverableSource(
      host,
      COMMAND,
      sessionID,
      controlTimeoutMs,
    );
    const { source } = initial;
    const plan = planV2Recovery(source.messages);
    const records: RecoveryRecord[] = [
      ...plan.records,
      ...initial.subagentRecords.map(record => ({
        kind: "subagent-record" as const,
        ...record,
      })),
    ];
    if (records.length === 0) return null;
    // The successor must pass the checks every other command applies.
    const stillOpen = plan.messages.some(
      message =>
        message.type === "assistant"
        && (isUnfinishedV2Assistant(message)
          || message.content.some(isUnfinishedV2Tool)),
    );
    if (stillOpen || backgroundJobs(plan.messages).undelivered.length > 0) {
      throw new Error("Recovered history still holds an unfinished record.");
    }
    const outcome = recoveryOutcome(records);
    // The outcome is session metadata for clients; the context is the
    // synthetic message the model receives.
    const context = recoveryContext(records);
    const successor = await createV2Successor(host, {
      command: COMMAND,
      source,
      messages:
        context === null
          ? plan.messages
          : [
              ...plan.messages,
              {
                id: "magic-recover-notice",
                type: "synthetic",
                text: context,
                metadata: RECOVERY_CONTEXT_METADATA,
                time: { created: Date.now() },
              } as V2Message,
            ],
      operation: "recover",
      compactionCount: getCompactionCount(source.session),
      outcome,
      omissions: [],
      tokensPruned: 0,
      // Recovery changes no image, so the current projection carries over.
      mediaTokensProjected:
        (await readStats(v2StorageKey(sessionID)))?.mediaTokensProjected ?? 0,
      recovery: records,
      recheckSource: async () => {
        const current = await readV2RecoverableSource(
          host,
          COMMAND,
          sessionID,
          controlTimeoutMs,
        );
        assertV2SourceUnchanged(COMMAND, source, current.source);
        if (
          !isDeepStrictEqual(current.subagentRecords, initial.subagentRecords)
        ) {
          throw new SourceChangedError(COMMAND);
        }
      },
      controlTimeoutMs,
    });
    return { successor, outcome };
  });
}

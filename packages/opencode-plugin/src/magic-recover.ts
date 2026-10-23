import type { Message, Part, ToolPart } from "@opencode-ai/sdk/v2";
import type { V2Client } from "./api";
import { RECOVER_NOOP, recoverSuccess } from "./commands";
import type { MessageWithParts } from "./compact/plan";
import { getCompactionCount } from "./compact/session";
import {
  appendedNoticeMessage,
  applySuccessorEdits,
  assertHistoryFinished,
  assertRecoverySourceUnchanged,
  createSuccessor,
  isUnfinishedAssistant,
  isUnfinishedTool,
  readRecoverableSource,
  undeliveredBackgroundTasks,
  withExclusiveSource,
  type Successor,
  type SuccessorEdit,
} from "./compact/successor";
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
} from "./policy/recovery";
import { readStats } from "./storage/stats";
import {
  CONTROL_OPERATION_TIMEOUT_MS,
  createToastFeedback,
  failureToast,
  IN_PROGRESS_TOAST_DURATION_MS,
  isRecord,
} from "./util";

const COMMAND = RECOVER_COMMAND_NAME;

export type MagicRecoverExecutionOptions = { controlTimeoutMs?: number };

// The edits that close every unfinished record of an OpenCode 1.18.30 history
// with the host's own interrupted representation, and the records they cover.
// An open assistant response gets the `MessageAbortedError` the host stores
// for an aborted response, which keeps its text and tool calls in model input
// (any other error hides the whole message), and, as the host's abort does, a
// completion time: the latest time the message already records, never a
// later one. A pending or running tool call becomes the host's error state,
// with captured input and metadata kept and the pending `raw` text or running
// `title` moved under `metadata.magicCompact.recovered`. It leaves out the
// `metadata.interrupted` flag the host's own abort adds: with that flag the
// host sends the call's partial `metadata.output` (an empty string for a
// killed shell) as if it were the result, and the model would never see that
// the outcome is unknown; without it the host sends the error text, and the
// partial output stays stored. A background `task`
// launch still owed its result is marked as never delivered, naming its child
// session. Nothing is invented: no output, result or delivery.
export function planRecovery(messages: MessageWithParts[]): {
  edits: SuccessorEdit[];
  records: RecoveryRecord[];
} {
  const undelivered = new Map(
    undeliveredBackgroundTasks(messages).map(task => [task.part, task]),
  );
  const edits: SuccessorEdit[] = [];
  const records: RecoveryRecord[] = [];
  for (const message of messages) {
    const { info } = message;
    const unfinished = isUnfinishedAssistant(info);
    const tools = message.parts.filter(isUnfinishedTool);
    if (
      (unfinished || tools.length > 0)
      && (info.role !== "assistant" || info.summary === true)
    ) {
      throw unsupportedRecoveryRecord(
        info.role === "assistant" ? "native compaction summary" : "message",
        info.id,
        info.role === "assistant"
          ? "OpenCode stores no interrupted form of an unfinished native checkpoint, so closing it would mean guessing its summary"
          : "OpenCode stores tool calls it can close only in assistant messages",
      );
    }
    if (unfinished && info.role === "assistant") {
      edits.push({
        type: "replace-info",
        info: {
          ...info,
          time: { ...info.time, completed: latestRecordedTime(message) },
          error: {
            name: "MessageAbortedError",
            data: { message: RECOVERED_RESPONSE_ERROR },
          },
        } as Message,
      });
      records.push({ kind: "response", messageID: info.id });
    }
    for (const part of message.parts) {
      if (isUnfinishedTool(part)) {
        edits.push({
          type: "replace-part",
          messageID: info.id,
          part: closedToolCall(part, info.time.created),
        });
        records.push({
          kind: "tool-call",
          messageID: info.id,
          callID: part.callID,
          tool: part.tool,
          state: part.state.status,
        });
      }
      const task = undelivered.get(part as ToolPart);
      if (task === undefined) continue;
      const launch = task.part;
      if (task.childID === undefined || launch.state.status !== "completed") {
        throw unsupportedRecoveryRecord(
          "background task call",
          launch.callID,
          "it names no subagent session, so its missing result cannot be recorded against one",
        );
      }
      edits.push({
        type: "replace-part",
        messageID: info.id,
        part: {
          ...launch,
          state: {
            ...launch.state,
            metadata: withRecoveryMarker(launch.state.metadata, {
              delivery: "abandoned",
              job: task.childID,
            }),
          },
        } as Part,
      });
      records.push({
        kind: "undelivered",
        messageID: info.id,
        callID: launch.callID,
        tool: launch.tool,
        job: task.childID,
      });
    }
  }
  return { edits, records };
}

// The latest time a stored message and its parts record.
function latestRecordedTime(message: MessageWithParts): number {
  const times: unknown[] = [message.info.time.created];
  for (const part of message.parts) {
    const time =
      part.type === "tool"
        ? "time" in part.state
          ? part.state.time
          : undefined
        : "time" in part
          ? part.time
          : undefined;
    if (isRecord(time)) {
      const { start, end } = time as { start?: unknown; end?: unknown };
      times.push(start, end);
    }
  }
  return Math.max(
    ...times.filter((time): time is number => Number.isSafeInteger(time)),
  );
}

function closedToolCall(part: ToolPart, messageCreated: number): ToolPart {
  const { state } = part;
  if (state.status === "pending") {
    return {
      ...part,
      state: {
        status: "error",
        input: state.input,
        error: RECOVERED_TOOL_ERROR,
        metadata: withRecoveryMarker(undefined, {
          outcome: "unknown",
          state: "pending",
          raw: state.raw,
        }),
        // A pending call never started; the message's creation time is the
        // only time the host recorded for it.
        time: { start: messageCreated, end: messageCreated },
      },
    };
  }
  if (state.status === "running") {
    return {
      ...part,
      state: {
        status: "error",
        input: state.input,
        error: RECOVERED_TOOL_ERROR,
        metadata: withRecoveryMarker(state.metadata, {
          outcome: "unknown",
          state: "running",
          ...(state.title === undefined ? {} : { title: state.title }),
        }),
        time: { start: state.time.start, end: state.time.start },
      },
    };
  }
  throw unsupportedRecoveryRecord(
    "tool call",
    part.callID,
    "it is in a state this version does not define",
  );
}

// Returns the ready successor, or null when nothing needs recovery. The source
// session and its subagent sessions are never written.
export async function executeMagicRecover(
  v2: V2Client,
  sessionID: string,
  options: MagicRecoverExecutionOptions = {},
): Promise<Successor | null> {
  const controlTimeoutMs =
    options.controlTimeoutMs ?? CONTROL_OPERATION_TIMEOUT_MS;
  if (!Number.isFinite(controlTimeoutMs) || controlTimeoutMs <= 0) {
    throw new Error("Magic Recover control deadline must be positive.");
  }
  const toastFeedback = createToastFeedback(v2, controlTimeoutMs);
  await toastFeedback.showProgress({
    title: COMMAND,
    message: "Recovering session...",
    variant: "info",
    duration: IN_PROGRESS_TOAST_DURATION_MS,
  });

  try {
    const outcome = await withExclusiveSource(COMMAND, sessionID, async () => {
      const source = await readRecoverableSource(
        v2,
        COMMAND,
        sessionID,
        controlTimeoutMs,
      );
      const plan = planRecovery(source.messages);
      if (plan.records.length === 0) return null;
      const text = recoveryOutcome(plan.records);
      // The outcome is ignored, so it shows in the session and never reaches
      // the model; the context is the synthetic text the model receives.
      const context = recoveryContext(plan.records);
      const history = applySuccessorEdits(source.messages, [
        ...plan.edits,
        {
          type: "append-message",
          message: appendedNoticeMessage(source, "recovery", [
            { type: "text", text, ignored: true },
            ...(context === null
              ? []
              : [
                  {
                    type: "text" as const,
                    text: context,
                    synthetic: true,
                    metadata: RECOVERY_CONTEXT_METADATA,
                  },
                ]),
          ]),
        },
      ]);
      // The successor must pass the checks every other command applies.
      assertHistoryFinished(COMMAND, history);
      const successor = await createSuccessor(v2, {
        command: COMMAND,
        source,
        history,
        operation: "recover",
        compactionCount: getCompactionCount(source.session),
        omissions: [],
        tokensPruned: 0,
        // Recovery changes no image, so the current projection carries over.
        mediaTokensProjected:
          (await readStats(sessionID))?.mediaTokensProjected ?? 0,
        recovery: plan.records,
        recheckSource: async () =>
          assertRecoverySourceUnchanged(
            COMMAND,
            source,
            await readRecoverableSource(
              v2,
              COMMAND,
              sessionID,
              controlTimeoutMs,
            ),
          ),
        controlTimeoutMs,
      });
      return { successor, text };
    });

    if (outcome === null) {
      await toastFeedback.showTerminal({
        title: COMMAND,
        message: RECOVER_NOOP,
        variant: "info",
        duration: 5000,
      });
      return null;
    }
    await toastFeedback.showTerminal({
      title: COMMAND,
      message: `${recoverSuccess(outcome.successor.session.id)}\n${outcome.text}`,
      variant: "info",
      duration: 10_000,
    });
    return outcome.successor;
  } catch (error) {
    await toastFeedback.showTerminal(failureToast(COMMAND, error));
    throw error;
  }
}

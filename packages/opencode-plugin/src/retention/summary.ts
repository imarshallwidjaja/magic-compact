// One fresh whole-session summary for task-aware retention. A deny-all fork
// of the source is prompted once through OpenCode's native hidden compaction
// agent, so native history conversion supplies the model-visible history,
// including the most recent turns and any prior native summary. The fork is
// always deleted, and only a response of the exact expected shape is
// accepted.
import type { PermissionRuleset, Session } from "@opencode-ai/sdk/v2";
import { randomBytes } from "node:crypto";
import type { V2Client } from "../api";
import {
  computeNativeBatchBudget,
  estimateNativeHistoryTokens,
  NATIVE_BATCH_FIXED_RESERVE_TOKENS,
  type NativeBatchBudget,
} from "../compact/batch";
import {
  assertConfiguredModelSupportsImages,
  assertForkRecordIdentity,
  historyMayExposeImages,
  isContextOverflow,
  resolveCompactionSettings,
  safeStatus,
  type CompactionSettings,
} from "../compact/compact";
import {
  assertForkMessageShape,
  nativeModelVisibleMessages,
  snapshotSourceMessages,
  type MessageWithParts,
} from "../compact/plan";
import { countGptTokens } from "../stats/tokenize";
import { attemptWithinDeadline, CONTROL_OPERATION_TIMEOUT_MS } from "../util";
import { RETENTION_FAILURE_GUIDANCE } from "./classifier";

// Verbatim SUMMARY_TEMPLATE from OpenCode
// packages/core/src/session/compaction.ts at 830d5eb535 (OpenCode 1.18.30).
const NATIVE_SUMMARY_TEMPLATE = `Output exactly the Markdown structure shown inside <template> and keep the section order unchanged. Do not include the <template> tags in your response.
<template>
## Objective
- [one or two brief sentences describing what the user is trying to accomplish]

## Important Details
- [constraints/preferences, decisions and why, important facts/assumptions, exact context needed to continue, or "(none)"]

## Work State
### Completed
- [finished work, verified facts, or changes made; otherwise "(none)"]

### Active
- [current work, partial changes, or investigation state; otherwise "(none)"]

### Blocked
- [blockers, failing commands, or unknowns; otherwise "(none)"]

## Next Move
1. [immediate concrete action, or "(none)"]
2. [next action if known, or "(none)"]

## Relevant Files
- [file or directory path: why it matters, or "(none)"]
</template>

Rules:
- Keep every section, even when empty.
- Use terse bullets, not prose paragraphs.
- Preserve exact file paths, symbols, commands, error strings, URLs, and identifiers when known.
- Do not mention the summary process or that context was compacted.`;

export const SESSION_SUMMARY_PROMPT = [
  "Create a new anchored summary from the entire conversation history above, including the most recent messages, so another coding agent can continue the work.",
  NATIVE_SUMMARY_TEMPLATE,
].join("\n\n");

const SUMMARY_AGENT = "compaction";
// The session rule is required: the prompting agent's own permissions would
// otherwise override the native compaction agent's deny rules.
const SUMMARY_FORK_PERMISSION: PermissionRuleset = [
  { permission: "*", pattern: "*", action: "deny" },
];
const ASSISTANT_PART_TYPES = new Set([
  "step-start",
  "step-finish",
  "reasoning",
  "text",
]);
const ID_RANDOM_ALPHABET =
  "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
const ID_ORDER_WAIT_ATTEMPTS = 50;

// Summary forks that are still owned by a running command, and whether
// OpenCode tried to compact one natively.
const ownedSummaryForks = new Map<string, { nativeCompaction: boolean }>();

export function isOwnedSessionSummaryFork(sessionID: string): boolean {
  return ownedSummaryForks.has(sessionID);
}

// Called from the plugin's `experimental.session.compacting` hook. OpenCode
// 1.18.30 runs that hook at the start of every native compaction, whether
// triggered before a step, by a finished step's token count, or by a
// provider overflow, and before its summary request, replay, or
// auto-continue. Throwing ends the fork's prompt loop there, so an owned
// fork never sends another model request; the command then reports that the
// session is too large.
export function rejectNativeCompactionOfSummaryFork(sessionID: string): void {
  const owned = ownedSummaryForks.get(sessionID);
  if (!owned) return;
  owned.nativeCompaction = true;
  throw new SessionSummaryError("budget");
}

export type SessionSummaryStage =
  | "budget"
  | "fork"
  | "permission"
  | "history"
  | "provider"
  | "validation"
  | "abort"
  | "delete";

// Messages are fixed text plus an optional HTTP status; they never carry
// session content, model output, or provider bodies.
export class SessionSummaryError extends Error {
  constructor(
    readonly stage: SessionSummaryStage,
    readonly status?: number,
    overflow = false,
  ) {
    super(
      `${
        stage === "budget" || overflow
          ? "The session is too large for the configured compaction model to summarize for task-aware retention."
          : `Task-aware retention session summary failed (${stage}${status === undefined ? "" : ` ${status}`}).`
      } ${RETENTION_FAILURE_GUIDANCE}`,
    );
    this.name = "SessionSummaryError";
  }
}

export type SessionSummaryOptions = {
  controlTimeoutMs?: number;
};

export async function generateSessionSummary(
  v2: V2Client,
  session: Session,
  messages: MessageWithParts[],
  options: SessionSummaryOptions = {},
): Promise<string> {
  const controlTimeoutMs =
    options.controlTimeoutMs ?? CONTROL_OPERATION_TIMEOUT_MS;
  const sourceSnapshot = snapshotSourceMessages(messages);
  // Budget and image capability follow what the model will see; content
  // hidden behind a native checkpoint is never sent.
  let visible: MessageWithParts[];
  try {
    visible = nativeModelVisibleMessages(messages);
  } catch {
    throw new SessionSummaryError("history");
  }
  const requiresImages = historyMayExposeImages(visible);
  let settings: CompactionSettings;
  let budget: NativeBatchBudget;
  try {
    settings = await resolveCompactionSettings(v2, session, requiresImages);
    if (requiresImages) assertConfiguredModelSupportsImages(settings);
    budget = computeNativeBatchBudget(settings.descriptor.limit);
  } catch (error) {
    // These configuration messages are fixed text without session content.
    throw new Error(
      `${error instanceof Error ? error.message : "Magic Compact could not resolve the native compaction agent."} ${RETENTION_FAILURE_GUIDANCE}`,
      { cause: error },
    );
  }
  // Fail before any fork when the request cannot fit. The compacting hook
  // still stops native compaction if the estimate is exceeded anyway.
  if (
    NATIVE_BATCH_FIXED_RESERVE_TOKENS
      + countGptTokens(SESSION_SUMMARY_PROMPT)
      + estimateNativeHistoryTokens(visible)
    > budget.hardSingleTurnInput
  ) {
    throw new SessionSummaryError("budget");
  }

  const route = {
    directory: session.directory,
    ...(session.workspaceID ? { workspace: session.workspaceID } : {}),
  };
  let fork: Session;
  try {
    fork = unwrapResponse(
      await v2.session.fork({ sessionID: session.id, ...route }),
    );
  } catch (error) {
    throw new SessionSummaryError("fork", safeStatus(error));
  }
  const owned = { nativeCompaction: false };
  ownedSummaryForks.set(fork.id, owned);

  let failure: SessionSummaryError | undefined;
  let prompted = false;
  let summary: string | undefined;
  try {
    const permission = await attemptWithinDeadline(
      signal =>
        v2.session
          .update(
            {
              sessionID: fork.id,
              ...route,
              permission: SUMMARY_FORK_PERMISSION,
            },
            { signal },
          )
          .then(unwrapResponse),
      controlTimeoutMs,
    );
    if (permission.kind !== "success") {
      throw stageFailure("permission", permission);
    }

    const history = await loadForkMessages(
      v2,
      fork.id,
      route,
      controlTimeoutMs,
    );
    try {
      assertForkMessageShape(history, { sourceSnapshot });
      assertForkRecordIdentity(history, fork.id);
    } catch {
      throw new SessionSummaryError("history");
    }
    const messageID = await nextPromptMessageID(history);

    prompted = true;
    let response: unknown;
    try {
      response = unwrapResponse(
        await v2.session.prompt({
          sessionID: fork.id,
          ...route,
          messageID,
          agent: SUMMARY_AGENT,
          model: settings.model,
          ...(settings.variant ? { variant: settings.variant } : {}),
          parts: [{ type: "text", text: SESSION_SUMMARY_PROMPT }],
        }),
      );
    } catch (error) {
      if (owned.nativeCompaction) throw new SessionSummaryError("budget");
      throw providerFailure(error);
    }
    if (owned.nativeCompaction) throw new SessionSummaryError("budget");
    // OpenCode reports provider failures, including a context overflow when
    // auto-compaction is off, as an error on a 200 response.
    const responseInfo = isObject(response) ? response["info"] : undefined;
    const responseError = isObject(responseInfo)
      ? responseInfo["error"]
      : undefined;
    if (responseError !== undefined && responseError !== null) {
      throw providerFailure(responseError);
    }

    const final = await loadForkMessages(v2, fork.id, route, controlTimeoutMs);
    // A native checkpoint in the fork means OpenCode found the request too
    // large, even when the compacting hook did not run.
    if (final.slice(history.length).some(isNativeCompactionArtifact)) {
      throw new SessionSummaryError("budget");
    }
    summary = acceptedSummary(response, final, history, messageID, fork.id, {
      sourceSnapshot,
    });
  } catch (error) {
    failure =
      error instanceof SessionSummaryError
        ? error
        : new SessionSummaryError("validation");
  }

  const cleanupFailures: SessionSummaryError[] = [];
  if (failure && prompted) {
    try {
      unwrapResponse(await v2.session.abort({ sessionID: fork.id, ...route }));
    } catch (error) {
      cleanupFailures.push(new SessionSummaryError("abort", safeStatus(error)));
    }
  }
  try {
    unwrapResponse(await v2.session.delete({ sessionID: fork.id, ...route }));
    ownedSummaryForks.delete(fork.id);
  } catch (error) {
    cleanupFailures.push(new SessionSummaryError("delete", safeStatus(error)));
  }

  const failures = [...(failure ? [failure] : []), ...cleanupFailures];
  if (failures.length === 1) throw failures[0];
  if (failures.length > 1) {
    throw new AggregateError(
      failures,
      `Task-aware retention session summary and fork cleanup failed (${failures.map(item => item.stage).join(", ")}). ${RETENTION_FAILURE_GUIDANCE}`,
      { cause: failures[0] },
    );
  }
  return summary!;
}

// Accepts the fork only when it is the rewritten source history plus exactly
// the one prompt and one ordinary compaction-agent reply. Native overflow
// compaction or replay adds markers, summaries, or a re-created prompt, and a
// partial native summary must never stand in for the whole session.
function acceptedSummary(
  response: unknown,
  messages: MessageWithParts[],
  history: MessageWithParts[],
  messageID: string,
  forkID: string,
  source: Parameters<typeof assertForkMessageShape>[1],
): string {
  const invalid = () => new SessionSummaryError("validation");
  if (messages.length !== history.length + 2) throw invalid();
  const prefix = messages.slice(0, history.length);
  try {
    assertForkMessageShape(prefix, source);
    assertForkRecordIdentity(messages, forkID);
  } catch {
    throw invalid();
  }
  if (
    prefix.some(
      (message, index) =>
        message.info.id !== history[index]!.info.id
        || message.parts.some(
          (part, partIndex) => part.id !== history[index]!.parts[partIndex]!.id,
        ),
    )
  ) {
    throw invalid();
  }

  const prompt = messages[history.length]!;
  const promptPart = prompt.parts[0];
  if (
    prompt.info.id !== messageID
    || prompt.info.role !== "user"
    || prompt.info.agent !== SUMMARY_AGENT
    || prompt.parts.length !== 1
    || promptPart?.type !== "text"
    || promptPart.text !== SESSION_SUMMARY_PROMPT
    || promptPart.synthetic === true
  ) {
    throw invalid();
  }

  const reply = messages[history.length + 1]!;
  const responseInfo = isObject(response) ? response["info"] : undefined;
  if (
    reply.info.role !== "assistant"
    || !isObject(responseInfo)
    || responseInfo["id"] !== reply.info.id
    || reply.info.parentID !== messageID
    || reply.info.agent !== SUMMARY_AGENT
    || reply.info.summary === true
    || reply.info.finish !== "stop"
    || reply.info.error !== undefined
    || reply.parts.some(part => !ASSISTANT_PART_TYPES.has(part.type))
  ) {
    throw invalid();
  }
  const text = reply.parts
    .flatMap(part =>
      part.type === "text" && part.synthetic !== true && part.ignored !== true
        ? [part.text]
        : [],
    )
    .join("\n\n")
    .trim();
  if (!text) throw invalid();
  return text;
}

async function loadForkMessages(
  v2: V2Client,
  forkID: string,
  route: { directory: string; workspace?: string },
  controlTimeoutMs: number,
): Promise<MessageWithParts[]> {
  const outcome = await attemptWithinDeadline(
    signal =>
      v2.session
        .messages({ sessionID: forkID, ...route }, { signal })
        .then(unwrapResponse),
    controlTimeoutMs,
  );
  if (outcome.kind !== "success") throw stageFailure("history", outcome);
  return outcome.value;
}

function isNativeCompactionArtifact(message: MessageWithParts): boolean {
  return (
    (message.info.role === "assistant" && message.info.summary === true)
    || message.parts.some(part => part.type === "compaction")
  );
}

function providerFailure(error: unknown): SessionSummaryError {
  const status = safeStatus(error);
  return new SessionSummaryError(
    "provider",
    status,
    isContextOverflow(error, status),
  );
}

function stageFailure(
  stage: SessionSummaryStage,
  outcome: { kind: "timed-out" } | { kind: "failed"; error: unknown },
): SessionSummaryError {
  return new SessionSummaryError(
    stage,
    outcome.kind === "failed" ? safeStatus(outcome.error) : undefined,
  );
}

// An explicit OpenCode message ID pins the prompt's identity, so a native
// overflow replay, which re-creates the prompt under a new ID, is rejected.
// OpenCode ascending IDs encode the low 48 bits of `ms * 0x1000 + counter`
// as 12 hex digits plus 14 random base62 characters. The plugin shares
// OpenCode's process clock: waiting (bounded) until the time value passes the
// newest fork ID keeps the prompt after the history, and counter 0 keeps it
// before any reply created in the same millisecond. Validation still checks
// the resulting order.
async function nextPromptMessageID(
  history: MessageWithParts[],
): Promise<string> {
  const newest = history.reduce((max, message) => {
    const match = /^msg_([0-9a-f]{12})/u.exec(message.info.id);
    const value = match ? BigInt(`0x${match[1]}`) : 0n;
    return value > max ? value : max;
  }, 0n);
  for (
    let attempt = 0;
    attempt < ID_ORDER_WAIT_ATTEMPTS && idTimeValue() <= newest;
    attempt += 1
  ) {
    await new Promise(resolve => setTimeout(resolve, 1));
  }
  return `msg_${idTimeValue().toString(16).padStart(12, "0")}${Array.from(
    randomBytes(14),
    byte => ID_RANDOM_ALPHABET[byte % ID_RANDOM_ALPHABET.length],
  ).join("")}`;
}

function idTimeValue(): bigint {
  return (BigInt(Date.now()) * 0x1000n) & ((1n << 48n) - 1n);
}

function unwrapResponse<T, E>(response: { data?: T; error?: E }): T {
  if ("error" in response && response.error) throw response.error;
  return response.data as T;
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

// Successor lifecycle pieces shared by both hosts.
import type { ForkFailureDetail } from "../compact/fork";
import { PluginMessageError } from "./feedback";

export type SuccessorCommand = "Magic Compact" | "Magic Trim" | "Magic Recover";

export const COMPACT_INSUFFICIENT_SAVINGS =
  "Magic Compact projected no useful canonical active-content savings; the source session was left unchanged.";
const MINIMUM_USEFUL_SAVINGS_RATIO = 0.2;
const MINIMUM_USEFUL_SAVINGS_TOKENS = 32;

// The savings gate: a compaction must remove at least 20% of the eligible
// summarized tokens, with a small absolute floor. Each host counts tokens
// from its own records and applies this one rule.
export function hasUsefulSavings(
  savedTokens: number,
  eligibleTokens: number,
): boolean {
  return (
    savedTokens
    >= Math.max(
      MINIMUM_USEFUL_SAVINGS_TOKENS,
      Math.ceil(eligibleTokens * MINIMUM_USEFUL_SAVINGS_RATIO),
    )
  );
}

const activeSources = new Set<string>();

// Two commands for one source in this process would each build a successor
// from the same history. Other processes and clients are covered by the
// at-rest and drift checks, not by this guard.
export async function withExclusiveSource<T>(
  command: SuccessorCommand,
  sessionID: string,
  operation: () => Promise<T>,
): Promise<T> {
  if (activeSources.has(sessionID)) {
    throw new PluginMessageError(
      `${command} is already preparing a successor for this session. Wait for it to finish, then retry.`,
    );
  }
  activeSources.add(sessionID);
  try {
    return await operation();
  } finally {
    activeSources.delete(sessionID);
  }
}

export class SourceNotAtRestError extends PluginMessageError {
  // The message is `statement` and then the `record` line. They stay apart so
  // a caller that must add recovery text can place it before the record.
  readonly statement: string;

  // `record` is a trailing line of references (record IDs) for a person to
  // quote; the sentence before it stands without them.
  constructor(
    command: SuccessorCommand,
    reason: string,
    guidance = "Wait for it to finish or resolve it, then retry.",
    readonly record?: string,
  ) {
    const statement = `${command} requires the session to be at rest, but ${reason}. ${guidance} The session was left unchanged.`;
    super(record === undefined ? statement : `${statement}\n${record}`);
    this.statement = statement;
    this.name = "SourceNotAtRestError";
  }
}

// What a host adapter found unfinished in one stored message: host record IDs
// and fixed labels only, never message content, tool input or output, or error
// bodies.
export type UnfinishedHistoryRecord = {
  messageID: string;
  kind: "assistant" | "compaction" | "shell" | "message";
  // The message itself is unfinished: an assistant message with neither a
  // completion time nor an error, or a compaction or shell record that is
  // still marked running.
  unfinished: boolean;
  // The first unfinished tool call in the message and how many it holds.
  tool?: { name: string; callID: string; state: string; count: number };
};

// Host-derived labels are cut to a conservative character set and length so a
// reference cannot carry control characters, markup or prose into a message.
export function boundedLabel(value: string, limit: number): string {
  const label = value.replace(/[^A-Za-z0-9_.:-]/gu, "?");
  if (label === "") return "unnamed";
  return label.length > limit ? `${label.slice(0, limit - 3)}...` : label;
}

const plural = (count: number, noun: string) =>
  `${count} ${noun}${count === 1 ? "" : "s"}`;

// Refusal for a stored history that holds records which never finished. The
// check reads every stored message, so an interrupted older record blocks a
// session that is otherwise idle. The sentence says so and what to do either
// way; the trailing `Record:` line names the first record in stored order and
// counts the rest. `subagentSessionID` names the subagent session whose
// history holds the record, when it is not the source itself.
export function unfinishedHistoryRefusal(
  command: SuccessorCommand,
  records: readonly UnfinishedHistoryRecord[],
  subagentSessionID?: string,
): SourceNotAtRestError {
  const [first, ...others] = records;
  if (first === undefined) throw new Error("No unfinished record to report.");
  const noun = {
    assistant: "assistant message",
    compaction: "compaction record",
    shell: "shell record",
    message: "message",
  }[first.kind];
  const segments = [
    ...(subagentSessionID === undefined
      ? []
      : [`subagent session ${boundedLabel(subagentSessionID, 64)}`]),
    `${noun} ${boundedLabel(first.messageID, 64)}`,
  ];
  if (first.unfinished) {
    segments.push(
      first.kind === "assistant"
        ? "no completion time or error"
        : "still marked running",
    );
  }
  if (first.tool) {
    const { name, callID, state, count } = first.tool;
    segments.push(
      `${boundedLabel(state, 16)} ${boundedLabel(name, 24)} call ${boundedLabel(callID, 64)}${count > 1 ? ` and ${plural(count - 1, "more call")}` : ""}`,
    );
  }
  if (others.length > 0) {
    segments.push(`plus ${plural(others.length, "more message")}`);
  }
  // A stored record is not proof of running work, so the guidance covers both
  // cases without judging which one this is.
  return new SourceNotAtRestError(
    command,
    "its stored history has an unfinished record, which blocks even an idle session",
    `If work may still be running, wait and retry. If it was abandoned, ${ABANDONED_WORK_GUIDANCE}${command === "Magic Trim" ? " `--heuristic` does not skip this check." : ""}`,
    `Record: ${segments.join("; ")}.`,
  );
}

// The next step for history whose work the user knows was abandoned. Only an
// explicit `/magic-recover` may treat stored records that way.
export const ABANDONED_WORK_GUIDANCE =
  "/magic-recover continues in a copy that records it as interrupted.";

// Refusal for a background job whose result never reached the source.
export function undeliveredBackgroundRefusal(
  command: SuccessorCommand,
  reason: string,
): SourceNotAtRestError {
  return new SourceNotAtRestError(
    command,
    reason,
    `Wait for it to deliver, then retry. If it was abandoned, ${ABANDONED_WORK_GUIDANCE}`,
  );
}

export class SourceChangedError extends PluginMessageError {
  constructor(command: SuccessorCommand) {
    super(
      `The session changed while ${command} was preparing its successor, so no successor was kept. The session was left unchanged; retry when it is at rest.`,
    );
    this.name = "SourceChangedError";
  }
}

// The fixed ways a successor candidate can fail verification.
export type SuccessorVerificationDetail =
  | "history read failed"
  | "imported history did not match the plan"
  | "imported session did not match the source"
  | "instructions did not match the source"
  | "candidate was changed by another writer";

// An error this plugin raised whose `detail` is one of its own fixed category
// texts, never host content. Only the subclasses below construct it, each from
// a closed set.
export abstract class ContentFreeError extends PluginMessageError {
  protected constructor(
    message: string,
    readonly detail: SuccessorVerificationDetail | ForkFailureDetail,
  ) {
    super(message);
  }
}

export class SuccessorVerificationError extends ContentFreeError {
  constructor(detail: SuccessorVerificationDetail) {
    super(detail, detail);
    this.name = "SuccessorVerificationError";
  }
}

export type SuccessorStage =
  | "plan"
  | "create"
  | "import"
  | "verify"
  | "instructions"
  | "cache"
  | "stats"
  | "recheck"
  | "ready";

// What manual recovery needs and nothing else: the stage, a fixed category,
// an HTTP status or known system error code when the failure carried one, and
// the IDs of the source and of a candidate that was kept. Host bodies, paths,
// prompts and raw error objects never enter it.
export type SuccessorFailureDetails = {
  stage: SuccessorStage;
  detail?: ContentFreeError["detail"];
  status?: number;
  code?: string;
  sourceID: string;
  candidateID?: string;
};

export class SuccessorFailure extends Error {
  constructor(
    message: string,
    readonly details: SuccessorFailureDetails,
  ) {
    super(message);
    this.name = "SuccessorFailure";
  }
}

function ownValue(value: unknown, key: string): unknown {
  return typeof value === "object" && value !== null
    ? Object.getOwnPropertyDescriptor(value, key)?.value
    : undefined;
}

// Node system error codes a storage or connection failure can carry. A code
// outside this list is dropped: a host can put any text in an error's `code`.
const SYSTEM_ERROR_CODES = new Set([
  "EACCES",
  "EBUSY",
  "ECONNREFUSED",
  "ECONNRESET",
  "EEXIST",
  "EIO",
  "EISDIR",
  "EMFILE",
  "ENOENT",
  "ENOSPC",
  "ENOTDIR",
  "ENOTEMPTY",
  "EPERM",
  "EPIPE",
  "EROFS",
  "ETIMEDOUT",
]);

// The content-free parts of an arbitrary failure: the fixed detail of this
// plugin's own errors, an HTTP status, and a known system error code.
// Everything else a host error carries is dropped.
export function classifyFailure(
  error: unknown,
): Pick<SuccessorFailureDetails, "detail" | "status" | "code"> {
  const status = [
    ownValue(error, "status"),
    ownValue(ownValue(error, "response"), "status"),
  ].find(
    value =>
      Number.isInteger(value)
      && (value as number) >= 100
      && (value as number) <= 599,
  );
  const code = ownValue(error, "code");
  return {
    ...(error instanceof ContentFreeError ? { detail: error.detail } : {}),
    ...(status === undefined ? {} : { status: status as number }),
    ...(typeof code === "string" && SYSTEM_ERROR_CODES.has(code)
      ? { code }
      : {}),
  };
}

// The terminal failure for a candidate that could not be completed. It names
// the candidate when it was kept and carries only classified details; the
// original error is never attached.
export function successorFailure(
  command: SuccessorCommand,
  error: unknown,
  context: {
    stage: SuccessorStage;
    sourceID: string;
    candidateID?: string;
    // The candidate was deleted, or no candidate existed.
    discarded?: boolean;
    // The ready update was sent, so the host may already have applied it.
    readyAttempted?: boolean;
  },
): SuccessorFailure {
  const classified = classifyFailure(error);
  const { stage, sourceID, candidateID, discarded, readyAttempted } = context;
  const details: SuccessorFailureDetails = {
    stage,
    sourceID,
    ...classified,
    ...(candidateID !== undefined && !discarded ? { candidateID } : {}),
  };
  if (readyAttempted && candidateID !== undefined) {
    return new SuccessorFailure(
      successorNotReadyMessage(command, candidateID),
      details,
    );
  }
  // Only this plugin's own refusals carry their message through; their text is
  // fixed. A host error's name or message is never trusted. A refusal's record
  // line goes after the kept-candidate text: the candidate and what to do
  // about it come first, because the failure toast cuts the end of long text.
  const refusal = error instanceof SourceNotAtRestError ? error : undefined;
  const failure =
    refusal !== undefined
      ? refusal.statement
      : error instanceof SourceChangedError
        ? error.message
        : classified.detail !== undefined
          ? `${command} could not verify its successor session (${classified.detail}). The original session was left unchanged.`
          : stage === "create"
            ? `${command} could not create a successor session. The original session was left unchanged.`
            : stage === "plan"
              ? `${command} could not build a valid successor history. The original session was left unchanged.`
              : `${command} could not finish preparing its successor session. The original session was left unchanged.`;
  const text =
    candidateID !== undefined && !discarded
      ? `${failure} Successor candidate ${candidateID} was kept because it could not be confirmed as unmodified; this run did not mark it ready, so inspect it or delete it manually.`
      : failure;
  return new SuccessorFailure(
    refusal?.record === undefined ? text : `${text}\n${refusal.record}`,
    details,
  );
}

export function successorNotReadyMessage(
  command: SuccessorCommand,
  candidateID: string,
): string {
  return `${command} could not confirm that successor session ${candidateID} became ready. The original session was left unchanged; inspect ${candidateID} before continuing in it.`;
}

// Successor lifecycle pieces shared by both hosts.
import type { ForkFailureDetail } from "../compact/fork";

export type SuccessorCommand = "Magic Compact" | "Magic Trim";

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
    throw new Error(
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

export class SourceNotAtRestError extends Error {
  constructor(command: SuccessorCommand, reason: string) {
    super(
      `${command} requires the session to be at rest, but ${reason}. Wait for it to finish or resolve it, then retry. The session was left unchanged.`,
    );
    this.name = "SourceNotAtRestError";
  }
}

export class SourceChangedError extends Error {
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
export abstract class ContentFreeError extends Error {
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
  // fixed. A host error's name or message is never trusted.
  const failure =
    error instanceof SourceChangedError || error instanceof SourceNotAtRestError
      ? error.message
      : classified.detail !== undefined
        ? `${command} could not verify its successor session (${classified.detail}). The session was left unchanged.`
        : stage === "create"
          ? `${command} could not create a successor session. The session was left unchanged.`
          : stage === "plan"
            ? `${command} could not build a valid successor history. The session was left unchanged.`
            : `${command} could not finish preparing its successor session. The session was left unchanged.`;
  return new SuccessorFailure(
    candidateID !== undefined && !discarded
      ? `${failure} Successor candidate ${candidateID} was kept because it could not be confirmed as unmodified; inspect it or delete it manually.`
      : failure,
    details,
  );
}

export function successorNotReadyMessage(
  command: SuccessorCommand,
  candidateID: string,
): string {
  return `${command} could not confirm that successor session ${candidateID} became ready. The original session was left unchanged; inspect ${candidateID} before continuing in it.`;
}

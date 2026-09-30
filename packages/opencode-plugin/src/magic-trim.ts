import type { Session } from "@opencode-ai/sdk/v2";
import { unwrap, type V2Client } from "./api";
import { safeStatus } from "./compact/compact";
import {
  assertTrimSourceUnchanged,
  createTrimPlan,
  TrimSourceChangedError,
  type TrimPlan,
} from "./compact/plan";
import { trimToolParts } from "./compact/prune";
import {
  applyBackup,
  createBackup,
  deleteBackup,
  getCompactionCount,
  injectTrimStatsNotice,
  recordPruningStats,
} from "./compact/session";
import {
  decideRetention,
  findRetentionCandidates,
  HEURISTIC_RETENTION,
  retentionFeedback,
  type RetentionPlan,
  type RetentionPolicy,
} from "./retention/retention";
import { countSessionTokens } from "./stats/tokenize";
import {
  CONTROL_OPERATION_TIMEOUT_MS,
  createToastFeedback,
  IN_PROGRESS_TOAST_DURATION_MS,
} from "./util";

export const TRIM_SUCCESS = "Magic trimming successful.";
export const TRIM_NOOP = "No eligible tool calls need trimming.";
export const TRIM_PENDING_REVERT =
  "Magic Trim cannot classify tool calls while the session has a pending revert. Resolve the pending revert first, then retry.";
export const TRIM_ARGUMENTS_ERROR =
  "/magic-trim accepts an optional non-negative integer and an optional --heuristic flag.";

export type MagicTrimExecutionOptions = {
  retention: RetentionPolicy;
  controlTimeoutMs?: number;
};

// Fixed text plus an optional HTTP status; never an SDK error or body.
export class TrimRevalidationError extends Error {
  constructor(
    readonly target: "messages" | "session",
    readonly status?: number,
  ) {
    super(
      `Magic Trim could not reload the source ${target} to recheck it before writing${status === undefined ? "" : ` (HTTP ${status})`}. The source session was left unchanged.`,
    );
    this.name = "TrimRevalidationError";
  }
}

export function parseMagicTrimArguments(raw: string): {
  keepTurns: number;
  heuristic: boolean;
} {
  let keepTurns: number | undefined;
  let heuristic = false;
  for (const token of raw.trim().split(/\s+/u).filter(Boolean)) {
    if (token === "--heuristic" && !heuristic) {
      heuristic = true;
    } else if (/^\d+$/u.test(token) && keepTurns === undefined) {
      keepTurns = Number(token);
    } else {
      throw new Error(TRIM_ARGUMENTS_ERROR);
    }
  }
  return { keepTurns: keepTurns ?? 0, heuristic };
}

export async function executeMagicTrim(
  v2: V2Client,
  sessionID: string,
  keepTurns: number,
  options: MagicTrimExecutionOptions,
): Promise<boolean> {
  const controlTimeoutMs =
    options.controlTimeoutMs ?? CONTROL_OPERATION_TIMEOUT_MS;
  if (!Number.isFinite(controlTimeoutMs) || controlTimeoutMs <= 0) {
    throw new Error("Magic Trim control deadline must be positive.");
  }
  const toastFeedback = createToastFeedback(v2, controlTimeoutMs);
  await toastFeedback.showProgress({
    title: "Magic Trim",
    message: "Trimming session...",
    variant: "info",
    duration: IN_PROGRESS_TOAST_DURATION_MS,
  });
  let backupSession: Session | null = null;
  let sourceSession: Session | null = null;

  try {
    const plan = await createTrimPlan(v2, sessionID, keepTurns);
    sourceSession = unwrap(await v2.session.get({ sessionID }));
    let retention: RetentionPlan = HEURISTIC_RETENTION;
    if (options.retention.mode === "task-aware") {
      const found = await findRetentionCandidates(
        plan.messages,
        plan.trimmedTurns,
      );
      if (found.candidates.length > 0 && sourceSession.revert) {
        throw new Error(TRIM_PENDING_REVERT);
      }
      retention = await decideRetention(
        v2,
        sourceSession,
        plan.messages,
        found,
        options.retention,
        { controlTimeoutMs },
      );
    }

    const backup = await createBackup(
      v2,
      sourceSession,
      getCompactionCount(sourceSession),
    );
    if (retention.evaluated > 0) {
      // Classification is slow, so the source must still match the planned
      // history before any write. A failed check leaves the source untouched
      // and removes the unneeded backup. SDK errors can carry session
      // content, so these failures keep only fixed text and a status.
      let revalidationError: Error | undefined;
      try {
        await assertTrimSourceStillPlanned(v2, sessionID, plan);
      } catch (error) {
        revalidationError = error as Error;
      }
      if (revalidationError) {
        try {
          await deleteBackup(v2, backup.id);
        } catch {
          throw new AggregateError(
            [
              revalidationError,
              new Error(
                `Unneeded backup session ${backup.id} could not be deleted.`,
              ),
            ],
            `${String(revalidationError)} Source session ${sessionID} was left unchanged, but unneeded backup session ${backup.id} could not be deleted; delete it manually.`,
          );
        }
        throw revalidationError;
      }
    }
    backupSession = backup;

    const beforeTokens = await countSessionTokens(v2, sessionID);
    const trimmedToolCalls = await trimToolParts(
      { v2, sessionID },
      plan.trimmedTurns,
      retention.keep,
    );
    if (trimmedToolCalls === 0) {
      const unneededBackup = backupSession;
      backupSession = null;
      await deleteBackup(v2, unneededBackup.id);
      await toastFeedback.showTerminal({
        title: "Magic Trim",
        message: `${TRIM_NOOP} ${retentionFeedback(retention)}`,
        variant: "info",
        duration: 5000,
      });
      return false;
    }

    const afterTokens = await countSessionTokens(v2, sessionID);
    await recordPruningStats({
      sessionID,
      sourceSessionID: sessionID,
      tokensPruned: beforeTokens - afterTokens,
    });

    await injectTrimStatsNotice(
      v2,
      sessionID,
      beforeTokens,
      afterTokens,
      trimmedToolCalls,
    );
    const actualAfterTokens = await countSessionTokens(v2, sessionID);
    if (actualAfterTokens !== afterTokens) {
      throw new Error(
        `Trim final canonical active token count did not match its projection (projected ${afterTokens}, actual ${actualAfterTokens}).`,
      );
    }

    await toastFeedback.showTerminal({
      title: "Magic Trim",
      message: `Trimmed ${trimmedToolCalls} tool call(s). ${retentionFeedback(retention)}`,
      variant: "info",
      duration: 5000,
    });
    return true;
  } catch (error) {
    let failure = error;
    if (sourceSession && backupSession) {
      try {
        await applyBackup(v2, sourceSession, backupSession);
      } catch (recoveryError) {
        const recoveryGuidance = `Magic Trim automatic recovery failed. Inspect source session ${sourceSession.id} and backup session ${backupSession.id}, then recover manually.`;
        const recoveryFailure = new Error(recoveryGuidance, {
          cause: recoveryError,
        });
        failure = new AggregateError(
          [error, recoveryFailure],
          `Magic Trim failed: ${String(error)} ${recoveryGuidance}`,
          { cause: error },
        );
      }
    }

    await toastFeedback.showTerminal({
      title: "Magic Trim Failed",
      message: String(failure),
      variant: "error",
      duration: 8000,
    });
    throw failure;
  }
}

// Throws only TrimRevalidationError, TrimSourceChangedError, or the fixed
// pending-revert error.
async function assertTrimSourceStillPlanned(
  v2: V2Client,
  sessionID: string,
  plan: TrimPlan,
): Promise<void> {
  const messages = await revalidationRead("messages", () =>
    v2.session.messages({ sessionID }),
  );
  try {
    assertTrimSourceUnchanged(messages, plan);
  } catch {
    throw new TrimSourceChangedError();
  }
  const session = await revalidationRead("session", () =>
    v2.session.get({ sessionID }),
  );
  if (session.revert) throw new Error(TRIM_PENDING_REVERT);
}

async function revalidationRead<T>(
  target: TrimRevalidationError["target"],
  request: () => Promise<{ data?: T; error?: unknown; response?: Response }>,
): Promise<T> {
  let result: Awaited<ReturnType<typeof request>>;
  try {
    result = await request();
  } catch (error) {
    throw new TrimRevalidationError(target, safeStatus(error));
  }
  if (result.error || result.data === undefined) {
    const status = result.response?.status;
    throw new TrimRevalidationError(
      target,
      Number.isInteger(status) && status! >= 100 && status! <= 599
        ? status
        : undefined,
    );
  }
  return result.data;
}

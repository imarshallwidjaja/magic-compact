import type { Session } from "@opencode-ai/sdk/v2";
import { unwrap, type V2Client } from "./api";
import { createTrimPlan } from "./compact/plan";
import { trimToolParts } from "./compact/prune";
import {
  applyBackup,
  createBackup,
  deleteBackup,
  getCompactionCount,
  injectTrimStatsNotice,
  recordPruningStats,
} from "./compact/session";
import { countSessionTokens } from "./stats/tokenize";
import {
  CONTROL_OPERATION_TIMEOUT_MS,
  createToastFeedback,
  IN_PROGRESS_TOAST_DURATION_MS,
} from "./util";

export const TRIM_SUCCESS = "Magic trimming successful.";
export const TRIM_NOOP = "No eligible tool calls need trimming.";

export type MagicTrimExecutionOptions = {
  controlTimeoutMs?: number;
};

export async function executeMagicTrim(
  v2: V2Client,
  sessionID: string,
  keepTurns: number,
  options: MagicTrimExecutionOptions = {},
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
    backupSession = await createBackup(
      v2,
      sourceSession,
      getCompactionCount(sourceSession),
    );

    const beforeTokens = await countSessionTokens(v2, sessionID);
    const trimmedToolCalls = await trimToolParts(
      { v2, sessionID },
      plan.trimmedTurns,
    );
    if (trimmedToolCalls === 0) {
      const unneededBackup = backupSession;
      backupSession = null;
      await deleteBackup(v2, unneededBackup.id);
      await toastFeedback.showTerminal({
        title: "Magic Trim",
        message: TRIM_NOOP,
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
      message: `Trimmed ${trimmedToolCalls} tool call(s).`,
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

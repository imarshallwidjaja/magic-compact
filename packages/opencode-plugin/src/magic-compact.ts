import type { Session } from "@opencode-ai/sdk/v2";
import { unwrap, type V2Client } from "./api";
import {
  generateCompactionSummaries,
  injectSummaries,
  prepareHighRiskSummaries,
} from "./compact/compact";
import {
  applyBackup,
  createBackup,
  deleteProgressNotice,
  getCompactionCount,
  injectPostCompactionNotice,
  injectProgressNotice,
  injectCompactStatsNotice,
  recordPruningStats,
  reloadTurns,
  revalidateNativeCheckpoint,
  updateCompactionMetadata,
} from "./compact/session";
import {
  createCompactionPlan,
  NativeCheckpointChangedError,
} from "./compact/plan";
import { pruneSummarizedTurns } from "./compact/prune";
import { countSessionTokens, getProviderTokens } from "./stats/tokenize";

export const COMPACT_SUCCESS = "Magic compaction successful.";
export const COMPACT_NOOP = "No assistant turns are old enough to compact.";

export async function executeMagicCompact(
  v2: V2Client,
  sessionID: string,
  keepTurns: number,
): Promise<boolean> {
  let backupSession: Session | null = null;
  let sourceSession: Session | null = null;
  let compactedTurnCount: number;

  try {
    // Check if there's anything to compact
    const sourcePlan = await createCompactionPlan(v2, sessionID, keepTurns);
    if (sourcePlan.summarizedTurns.length === 0) {
      unwrap(
        await v2.tui.showToast({
          title: "Magic Compact",
          message: COMPACT_NOOP,
          variant: "info",
          duration: 5000,
        }),
      );
      return false;
    }

    // Create backup session
    sourceSession = unwrap(
      await v2.session.get({
        sessionID,
      }),
    );
    const currentCompactionCount = getCompactionCount(sourceSession) + 1;
    backupSession = await createBackup(
      v2,
      sourceSession,
      currentCompactionCount,
    );

    const beforeTokens =
      (await getProviderTokens(v2, sessionID))
      ?? (await countSessionTokens(v2, sessionID));

    await revalidateNativeCheckpoint(v2, sessionID, sourcePlan);

    const highRiskSummaries = prepareHighRiskSummaries(sourcePlan);

    const progressMessageID = await injectProgressNotice(v2, sessionID);
    let operationError: unknown;
    try {
      let summaries: string[] | undefined;
      let generationError: unknown;
      try {
        summaries = await generateCompactionSummaries(
          v2,
          sourceSession,
          sourcePlan,
          highRiskSummaries,
        );
      } catch (error) {
        generationError = error;
      }

      let revalidationError: unknown;
      try {
        await revalidateNativeCheckpoint(v2, sessionID, sourcePlan);
      } catch (error) {
        revalidationError = error;
      }

      if (revalidationError !== undefined) {
        throw generationError === undefined
          ? revalidationError
          : combineErrors(revalidationError, generationError);
      }
      if (generationError !== undefined) {
        throw generationError;
      }

      await injectSummaries(
        v2,
        sessionID,
        sourcePlan.summarizedTurns,
        summaries!,
      );
    } catch (error) {
      operationError = error;
    }
    try {
      await deleteProgressNotice(v2, sessionID, progressMessageID);
    } catch (cleanupError) {
      operationError =
        operationError === undefined
          ? cleanupError
          : combineErrors(operationError, cleanupError);
    }
    if (operationError !== undefined) {
      throw operationError;
    }

    // Mark the new compaction boundary for future recompactions
    // Message placed outside of summarization range so unaffected by pruning
    await injectPostCompactionNotice(v2, sessionID, sourcePlan.nextTurn);

    // Prune messages & tool calls
    const summarizedTurns = await reloadTurns(v2, sessionID, sourcePlan);
    await pruneSummarizedTurns({ v2, sessionID }, summarizedTurns);

    await updateCompactionMetadata(v2, sourceSession, currentCompactionCount);
    const afterTokens = await countSessionTokens(v2, sessionID);
    const stats = await recordPruningStats({
      sessionID,
      sourceSessionID: sessionID,
      tokensPruned: beforeTokens - afterTokens,
    });

    await injectCompactStatsNotice(
      v2,
      sessionID,
      beforeTokens,
      afterTokens,
      currentCompactionCount,
      stats,
      sourceSession.model?.id ?? null,
    );
    compactedTurnCount = sourcePlan.summarizedTurns.length;
  } catch (error) {
    let failure = error;
    if (
      !(error instanceof NativeCheckpointChangedError)
      && sourceSession
      && backupSession
    ) {
      try {
        await applyBackup(v2, sourceSession, backupSession);
      } catch (rollbackError) {
        failure = combineErrors(failure, rollbackError);
      }
    }

    try {
      unwrap(
        await v2.tui.showToast({
          title: "Magic Compact Failed",
          message: String(error),
          variant: "error",
          duration: 8000,
        }),
      );
    } catch (notificationError) {
      failure = combineErrors(failure, notificationError);
    }
    throw failure;
  }

  unwrap(
    await v2.tui.showToast({
      title: "Magic Compact",
      message: `Compacted ${compactedTurnCount} assistant turn(s).`,
      variant: "info",
      duration: 5000,
    }),
  );
  return true;
}

function combineErrors(primary: unknown, secondary: unknown): Error {
  if (primary instanceof NativeCheckpointChangedError) {
    const causes =
      primary.cause instanceof AggregateError
        ? [...primary.cause.errors, secondary]
        : primary.cause === undefined
          ? [secondary]
          : [primary.cause, secondary];
    return new NativeCheckpointChangedError(
      new AggregateError(
        causes,
        "Native checkpoint state changed while another operation also failed.",
      ),
    );
  }

  return new AggregateError(
    [primary, secondary],
    "Magic Compact operation and a secondary recovery or notification operation both failed.",
  );
}

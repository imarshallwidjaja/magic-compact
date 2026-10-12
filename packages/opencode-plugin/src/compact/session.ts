import type { Session, TextPart } from "@opencode-ai/sdk/v2";
import { unwrap, type V2Client } from "../api";
import { forkSession } from "./fork";
import {
  BOUNDARY_METADATA,
  POST_COMPACTION_NOTICE,
  boundaryPartID,
} from "./constants";
import {
  assertCompactionSourceAfterInsertions,
  assertCompactionSourceUnchanged,
  CompactionPruneSourceDriftError,
  NativeCheckpointChangedError,
  type CompactionPlan,
  type CompactionSourceInsertion,
  type MessageWithParts,
  type Turn,
} from "./plan";
import { copyCache, deleteCache } from "../storage/omission";
import { copyStats, deleteStats, writeStats } from "../storage/stats";
import {
  STATS_METADATA,
  statsMessage,
  trimStatsMessage,
  type PreservedTurns,
} from "../stats/constants";

export type MagicCompactMetadata = {
  sourceSessionId: string;
  compactedAt: number;
  compactionCount: number;
};

export async function createBackup(
  v2: V2Client,
  sourceSession: Session,
  compactionCount: number,
  messages: MessageWithParts[],
): Promise<Session> {
  const backupSession = await forkSession(v2, sourceSession, messages);

  try {
    await copyCache(sourceSession.id, backupSession.id);
    await copyStats(sourceSession.id, backupSession.id);
    await updateForkMetadata(
      v2,
      sourceSession,
      sourceSession.id,
      backupSession.id,
      compactionCount,
    );
    return backupSession;
  } catch (initializationError) {
    const cleanupErrors: unknown[] = [];
    try {
      await deleteCache(backupSession.id);
    } catch (cleanupError) {
      cleanupErrors.push(
        new Error("Failed to delete partial backup omission cache.", {
          cause: cleanupError,
        }),
      );
    }
    try {
      await deleteStats(backupSession.id);
    } catch (cleanupError) {
      cleanupErrors.push(
        new Error("Failed to delete partial backup stats cache.", {
          cause: cleanupError,
        }),
      );
    }
    try {
      unwrap(await v2.session.delete({ sessionID: backupSession.id }));
    } catch (cleanupError) {
      cleanupErrors.push(cleanupError);
    }
    if (cleanupErrors.length > 0) {
      throw new AggregateError(
        [initializationError, ...cleanupErrors],
        "Backup initialization and partial-backup cleanup failed.",
        { cause: initializationError },
      );
    }
    throw initializationError;
  }
}

export async function deleteBackup(
  v2: V2Client,
  backupSessionID: string,
): Promise<void> {
  try {
    unwrap(await v2.session.delete({ sessionID: backupSessionID }));
  } catch (error) {
    throw new Error(`Failed to delete backup session ${backupSessionID}.`, {
      cause: error,
    });
  }
  const cleanupErrors: unknown[] = [];
  try {
    await deleteCache(backupSessionID);
  } catch (error) {
    cleanupErrors.push(
      new Error("Failed to delete backup omission cache.", { cause: error }),
    );
  }
  try {
    await deleteStats(backupSessionID);
  } catch (error) {
    cleanupErrors.push(
      new Error("Failed to delete backup stats cache.", { cause: error }),
    );
  }
  if (cleanupErrors.length === 1) throw cleanupErrors[0];
  if (cleanupErrors.length > 1) {
    throw new AggregateError(
      cleanupErrors,
      "Backup cache cleanup failed after backup session deletion.",
      { cause: cleanupErrors[0] },
    );
  }
}

export async function applyBackup(
  v2: V2Client,
  originalSession: Session,
  backupSession: Session,
): Promise<void> {
  unwrap(
    await v2.session.update({
      sessionID: backupSession.id,
      title: originalSession.title,
    }),
  );
  unwrap(
    await v2.tui.selectSession({
      sessionID: backupSession.id,
    }),
  );
  unwrap(
    await v2.session.delete({
      sessionID: originalSession.id,
    }),
  );
}

export function getCompactionCount(session: Session): number {
  const metadata = session.metadata?.magicCompact as
    MagicCompactMetadata | undefined;
  return metadata?.compactionCount ?? 0;
}

async function updateForkMetadata(
  v2: V2Client,
  sourceSession: Session,
  sourceSessionID: string,
  forkedSessionID: string,
  compactionCount: number,
): Promise<void> {
  const timestamp = new Date().toISOString();

  unwrap(
    await v2.session.update({
      sessionID: forkedSessionID,
      title: `[Backup] ${sourceSession.title} ${timestamp}`,
      metadata: {
        ...sourceSession.metadata,
        magicCompact: {
          sourceSessionId: sourceSessionID,
          compactedAt: Date.now(),
          compactionCount,
        },
      },
    }),
  );
}

export async function updateCompactionMetadata(
  v2: V2Client,
  session: Session,
  compactionCount: number,
): Promise<void> {
  unwrap(
    await v2.session.update({
      sessionID: session.id,
      metadata: {
        ...session.metadata,
        magicCompact: {
          compactionCount,
        },
      },
    }),
  );
}

export async function injectPostCompactionNotice(
  v2: V2Client,
  sessionID: string,
  nextTurn: Turn | null,
): Promise<CompactionSourceInsertion> {
  if (!nextTurn) {
    const inserted = unwrap(
      await v2.session.prompt({
        sessionID,
        noReply: true,
        parts: [
          {
            type: "text",
            text: POST_COMPACTION_NOTICE,
            synthetic: true,
            metadata: BOUNDARY_METADATA,
          },
        ],
      }),
    );
    return {
      type: "message",
      messageID: inserted.info.id,
      parts: inserted.parts,
    };
  }

  const firstUser = nextTurn.user[0];
  if (!firstUser) {
    throw new Error("Next turn has no user messages.");
  }

  const part = {
    id: boundaryPartID(firstUser.info.id),
    sessionID,
    messageID: firstUser.info.id,
    type: "text",
    text: POST_COMPACTION_NOTICE,
    synthetic: true,
    metadata: BOUNDARY_METADATA,
  } satisfies TextPart;

  unwrap(
    await v2.part.update({
      sessionID,
      messageID: firstUser.info.id,
      partID: part.id,
      part,
    }),
  );
  return { type: "part", messageID: firstUser.info.id, part };
}

export async function recordPruningStats(input: {
  sessionID: string;
  sourceSessionID: string;
  tokensPruned: number;
  mediaTokensProjected?: number;
}): Promise<void> {
  const stats = await copyStats(input.sourceSessionID, input.sessionID);
  const nextStats = {
    ...stats,
    totalTokensPruned: stats.totalTokensPruned + input.tokensPruned,
    mediaTokensProjected:
      input.mediaTokensProjected ?? stats.mediaTokensProjected ?? 0,
    cachedMediaTokensSaved: stats.cachedMediaTokensSaved ?? 0,
  };

  await writeStats(input.sessionID, nextStats);
}

export async function injectCompactStatsNotice(
  v2: V2Client,
  sessionID: string,
  beforeTokens: number,
  afterTokens: number,
  compactionCount: number,
  summarizedTurnCount: number,
  preserved: PreservedTurns,
): Promise<void> {
  unwrap(
    await v2.session.prompt({
      sessionID,
      noReply: true,
      parts: [
        {
          type: "text",
          text: statsMessage(
            compactionCount,
            beforeTokens,
            afterTokens,
            summarizedTurnCount,
            preserved,
          ),
          ignored: true,
          metadata: STATS_METADATA,
        },
      ],
    }),
  );
}

export async function injectTrimStatsNotice(
  v2: V2Client,
  sessionID: string,
  beforeTokens: number,
  afterTokens: number,
  trimmedToolCallCount: number,
  retentionFeedback: string,
): Promise<void> {
  unwrap(
    await v2.session.prompt({
      sessionID,
      noReply: true,
      parts: [
        {
          type: "text",
          text: trimStatsMessage(
            beforeTokens,
            afterTokens,
            trimmedToolCallCount,
            retentionFeedback,
          ),
          ignored: true,
          metadata: STATS_METADATA,
        },
      ],
    }),
  );
}

export async function revalidateCompactionSource(
  v2: V2Client,
  sessionID: string,
  plan: CompactionPlan,
): Promise<void> {
  try {
    const messages: MessageWithParts[] = unwrap(
      await v2.session.messages({
        sessionID,
      }),
    );
    assertCompactionSourceUnchanged(messages, plan);
  } catch (error) {
    if (error instanceof NativeCheckpointChangedError) {
      throw error;
    }
    throw new NativeCheckpointChangedError(error);
  }
}

export async function revalidateCompactionSourceAfterInsertions(
  v2: V2Client,
  sessionID: string,
  plan: CompactionPlan,
  insertions: CompactionSourceInsertion[],
): Promise<void> {
  try {
    const messages: MessageWithParts[] = unwrap(
      await v2.session.messages({ sessionID }),
    );
    assertCompactionSourceAfterInsertions(messages, plan, insertions);
  } catch (error) {
    if (error instanceof CompactionPruneSourceDriftError) throw error;
    throw new CompactionPruneSourceDriftError(error);
  }
}

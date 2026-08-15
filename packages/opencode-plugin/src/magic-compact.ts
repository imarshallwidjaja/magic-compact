import type { Message, Session, TextPart } from "@opencode-ai/sdk/v2";
import { unwrap, type V2Client } from "./api";
import {
  generateCompactionSummaries,
  injectSummaries,
} from "./compact/compact";
import {
  applyBackup,
  createBackup,
  getCompactionCount,
  injectPostCompactionNotice,
  injectCompactStatsNotice,
  recordPruningStats,
  reloadTurns,
  revalidateCompactionSource,
  updateCompactionMetadata,
} from "./compact/session";
import {
  createCompactionPlan,
  type CompactionPlan,
  type MessageWithParts,
  NativeCheckpointChangedError,
  type Turn,
} from "./compact/plan";
import {
  createPruneProjection,
  pruneSummarizedTurns,
  type PruneProjection,
} from "./compact/prune";
import {
  countCanonicalContentTokens,
  countSessionTokens,
} from "./stats/tokenize";
import {
  BOUNDARY_METADATA,
  POST_COMPACTION_NOTICE,
  boundaryPartID,
  summaryMetadata,
  summaryPartID,
} from "./compact/constants";
import { readStats, type ConversationStats } from "./storage/stats";
import type { PlannedOmission } from "./storage/omission";

export const COMPACT_SUCCESS = "Magic compaction successful.";
export const COMPACT_NOOP = "No assistant turns are old enough to compact.";
export const COMPACT_INSUFFICIENT_SAVINGS =
  "Magic Compact projected no useful canonical active-content savings; the source session was left unchanged.";
const MINIMUM_USEFUL_SAVINGS_RATIO = 0.2;
const MINIMUM_USEFUL_SAVINGS_TOKENS = 32;

type CompactionSelection = {
  turns: Turn[];
  summaries: string[];
  pruning: PruneProjection;
};

class CompactionSourceDriftError extends Error {
  constructor(
    readonly sourceSessionID: string,
    readonly backupSessionID: string,
    readonly projectedTokens: number,
    readonly actualTokens: number,
    cause?: unknown,
  ) {
    super(
      `Compaction token projection mismatch after source writes: projected ${projectedTokens}, counted ${actualTokens}. The partially mutated source session ${sourceSessionID} and backup session ${backupSessionID} were both preserved; inspect both and recover manually from backup ${backupSessionID}.`,
      { cause },
    );
    this.name = "CompactionSourceDriftError";
  }
}

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

    sourceSession = unwrap(
      await v2.session.get({
        sessionID,
      }),
    );
    const currentCompactionCount = getCompactionCount(sourceSession) + 1;
    const sourceMessages = unwrap(await v2.session.messages({ sessionID }));
    const beforeTokens = countCanonicalContentTokens(sourceMessages);

    await revalidateCompactionSource(v2, sessionID, sourcePlan);

    const summaries = await generateCompactionSummaries(
      v2,
      sourceSession,
      sourcePlan,
    );

    const selection = await selectProfitableTurns(
      sourcePlan,
      summaries,
      sessionID,
    );

    const projection = await projectCompaction(
      sourceMessages,
      sourcePlan,
      selection,
      sessionID,
      beforeTokens,
    );
    const eligibleTokens = countCanonicalContentTokens(
      selection.turns.flatMap(turn => turn.assistants),
    );
    const projectedSavings = beforeTokens - projection.afterTokens;
    const minimumSavings = Math.max(
      MINIMUM_USEFUL_SAVINGS_TOKENS,
      Math.ceil(eligibleTokens * MINIMUM_USEFUL_SAVINGS_RATIO),
    );
    if (projectedSavings < minimumSavings) {
      throw new Error(COMPACT_INSUFFICIENT_SAVINGS);
    }

    backupSession = await createBackup(
      v2,
      sourceSession,
      currentCompactionCount,
    );
    await revalidateCompactionSource(v2, sessionID, sourcePlan);

    await injectSummaries(v2, sessionID, selection.turns, selection.summaries);

    // Mark the new compaction boundary for future recompactions
    // Message placed outside of summarization range so unaffected by pruning
    await injectPostCompactionNotice(v2, sessionID, sourcePlan.nextTurn);

    // Prune messages & tool calls
    const summarizedTurns = await reloadTurns(v2, sessionID, selection.turns);
    await pruneSummarizedTurns(
      { v2, sessionID },
      summarizedTurns,
      projection.omissions,
    );

    await updateCompactionMetadata(v2, sourceSession, currentCompactionCount);
    await injectCompactStatsNotice(
      v2,
      sessionID,
      beforeTokens,
      projection.afterTokens,
      currentCompactionCount,
      projection.stats,
      sourceSession.model?.id ?? null,
    );
    const afterTokens = await countSessionTokens(v2, sessionID);
    if (afterTokens !== projection.afterTokens) {
      throw new CompactionSourceDriftError(
        sessionID,
        backupSession.id,
        projection.afterTokens,
        afterTokens,
      );
    }
    await recordPruningStats({
      sessionID,
      sourceSessionID: sessionID,
      tokensPruned: beforeTokens - projection.afterTokens,
    });
    compactedTurnCount = selection.turns.length;
  } catch (error) {
    let failure = error;
    if (
      !(error instanceof NativeCheckpointChangedError)
      && !(error instanceof CompactionSourceDriftError)
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

async function selectProfitableTurns(
  plan: CompactionPlan,
  summaries: string[],
  sessionID: string,
): Promise<CompactionSelection> {
  const selectedTurns: Turn[] = [];
  const selectedSummaries: string[] = [];
  const projectedTurns: Turn[] = [];
  const omissions: PlannedOmission[] = [];
  for (const [index, turn] of plan.summarizedTurns.entries()) {
    const summary = summaries[index];
    if (summary === undefined)
      throw new Error("Missing summary for assistant turn.");
    const candidates = injectProjectedSummaries([turn], [summary], sessionID);
    const pruning = await createPruneProjection(candidates, sessionID);
    const projectedTurn = pruning.turns[0]!;
    if (
      countTurnCanonicalTokens(projectedTurn) < countTurnCanonicalTokens(turn)
    ) {
      selectedTurns.push(turn);
      selectedSummaries.push(summary);
      projectedTurns.push(projectedTurn);
      omissions.push(...pruning.omissions);
    }
  }
  return {
    turns: selectedTurns,
    summaries: selectedSummaries,
    pruning: { turns: projectedTurns, omissions },
  };
}

function injectProjectedSummaries(
  turns: Turn[],
  summaries: string[],
  sessionID: string,
): Turn[] {
  return turns.map((turn, index) => {
    const projected = structuredClone(turn);
    const firstAssistant = projected.assistants[0];
    if (!firstAssistant) throw new Error("Turn missing assistant message.");
    const summary = summaries[index];
    if (summary === undefined)
      throw new Error("Missing summary for assistant turn.");
    firstAssistant.parts.push({
      id: summaryPartID(firstAssistant.info.id),
      sessionID,
      messageID: firstAssistant.info.id,
      type: "text",
      text: summary,
      metadata: summaryMetadata(),
    } satisfies TextPart);
    return projected;
  });
}

function countTurnCanonicalTokens(turn: Turn): number {
  return countCanonicalContentTokens([...turn.user, ...turn.assistants]);
}

async function projectCompaction(
  sourceMessages: MessageWithParts[],
  plan: CompactionPlan,
  selection: CompactionSelection,
  sessionID: string,
  beforeTokens: number,
) {
  const messages = structuredClone(sourceMessages);
  const byID = new Map(messages.map(message => [message.info.id, message]));
  const summarizedIDs = new Set(
    selection.turns.flatMap(turn =>
      [...turn.user, ...turn.assistants].map(message => message.info.id),
    ),
  );
  let appendedBoundary: MessageWithParts | null = null;
  if (plan.nextTurn) {
    const nextUser = byID.get(plan.nextTurn.user[0]!.info.id)!;
    nextUser.parts.push({
      id: boundaryPartID(nextUser.info.id),
      sessionID,
      messageID: nextUser.info.id,
      type: "text",
      text: POST_COMPACTION_NOTICE,
      synthetic: true,
      metadata: BOUNDARY_METADATA,
    });
  } else {
    appendedBoundary = projectedNoticeMessage(
      sessionID,
      "magic-projected-boundary",
      POST_COMPACTION_NOTICE,
      BOUNDARY_METADATA,
    );
  }
  const projectedByID = new Map(
    selection.pruning.turns
      .flatMap(turn => [...turn.user, ...turn.assistants])
      .map(message => [message.info.id, message]),
  );
  const projectedMessages = messages.flatMap(message => {
    if (!summarizedIDs.has(message.info.id)) return [message];
    const projected = projectedByID.get(message.info.id);
    return projected ? [projected] : [];
  });
  if (appendedBoundary) projectedMessages.push(appendedBoundary);
  const existingStats = await readStats(sessionID);
  const afterTokens = countCanonicalContentTokens(projectedMessages);
  return {
    afterTokens,
    omissions: selection.pruning.omissions,
    stats: nextConversationStats(
      existingStats,
      sessionID,
      beforeTokens - afterTokens,
    ),
  };
}

function nextConversationStats(
  current: ConversationStats | null,
  sessionID: string,
  tokensPruned: number,
): ConversationStats {
  return {
    version: 1,
    sourceSessionId: sessionID,
    rootSessionId: current?.rootSessionId ?? sessionID,
    totalTokensPruned: (current?.totalTokensPruned ?? 0) + tokensPruned,
    cachedTokensSaved: current?.cachedTokensSaved ?? 0,
    processedAssistantMessageIds: current?.processedAssistantMessageIds ?? [],
  };
}

function projectedNoticeMessage(
  sessionID: string,
  messageID: string,
  text: string,
  metadata: Record<string, unknown>,
): MessageWithParts {
  return {
    info: { id: messageID, role: "user" } as Message,
    parts: [
      {
        id: `${messageID}-part`,
        sessionID,
        messageID,
        type: "text",
        text,
        synthetic: true,
        metadata,
      } satisfies TextPart,
    ],
  };
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

  if (primary instanceof CompactionSourceDriftError) {
    const causes =
      primary.cause instanceof AggregateError
        ? [...primary.cause.errors, secondary]
        : primary.cause === undefined
          ? [secondary]
          : [primary.cause, secondary];
    return new CompactionSourceDriftError(
      primary.sourceSessionID,
      primary.backupSessionID,
      primary.projectedTokens,
      primary.actualTokens,
      new AggregateError(
        causes,
        "Source drift and another recovery or notification operation failed.",
      ),
    );
  }

  return new AggregateError(
    [primary, secondary],
    "Magic Compact operation and a secondary recovery or notification operation both failed.",
  );
}

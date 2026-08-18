import type { Message, Session, TextPart } from "@opencode-ai/sdk/v2";
import { unwrap, type V2Client } from "./api";
import {
  generateCompactionSummaries,
  type GeneratedSummary,
  injectSummaries,
} from "./compact/compact";
import {
  createBackup,
  getCompactionCount,
  injectPostCompactionNotice,
  injectCompactStatsNotice,
  recordPruningStats,
  revalidateCompactionSource,
  revalidateCompactionSourceAfterInsertions,
  updateCompactionMetadata,
} from "./compact/session";
import {
  createCompactionPlan,
  type CompactionPlan,
  type MessageWithParts,
  type Turn,
} from "./compact/plan";
import {
  createPruneProjection,
  pruneSummarizedTurns,
  type PruneProjection,
} from "./compact/prune";
import {
  countModelFacingContentTokens,
  countSessionTokenProjection,
  type ModelFacingTokenCount,
} from "./stats/tokenize";
import {
  BOUNDARY_METADATA,
  POST_COMPACTION_NOTICE,
  boundaryPartID,
  summaryEnvelopeTextPart,
} from "./compact/constants";
import { attemptWithinDeadline, CONTROL_OPERATION_TIMEOUT_MS } from "./util";

export const COMPACT_SUCCESS = "Magic compaction successful.";
export const COMPACT_NOOP = "No assistant turns are old enough to compact.";
export const COMPACT_PENDING_REVERT =
  "Magic Compact cannot run while the session has a pending revert. Resolve the pending revert first, then retry.";
export const COMPACT_INSUFFICIENT_SAVINGS =
  "Magic Compact projected no useful canonical active-content savings; the source session was left unchanged.";
const MINIMUM_USEFUL_SAVINGS_RATIO = 0.2;
const MINIMUM_USEFUL_SAVINGS_TOKENS = 32;

type CompactionSelection = {
  turns: Turn[];
  summaries: Exclude<GeneratedSummary, null>[];
  pruning: PruneProjection;
};

export type MagicCompactExecutionOptions = {
  controlTimeoutMs?: number;
};

type MagicCompactToast = {
  title: string;
  message: string;
  variant: "info" | "error";
  duration?: number;
};

export async function executeMagicCompact(
  v2: V2Client,
  sessionID: string,
  keepTurns: number,
  options: MagicCompactExecutionOptions = {},
): Promise<boolean> {
  const controlTimeoutMs =
    options.controlTimeoutMs ?? CONTROL_OPERATION_TIMEOUT_MS;
  if (!Number.isFinite(controlTimeoutMs) || controlTimeoutMs <= 0) {
    throw new Error("Magic Compact control deadline must be positive.");
  }
  let backupSession: Session | null = null;
  let sourceSession: Session | null = null;
  let sourceWritesStarted = false;
  let compactedTurnCount: number;

  try {
    // Check if there's anything to compact
    const sourcePlan = await createCompactionPlan(v2, sessionID, keepTurns);
    if (sourcePlan.summarizedTurns.length === 0) {
      await showMagicCompactToast(
        v2,
        {
          title: "Magic Compact",
          message: COMPACT_NOOP,
          variant: "info",
          duration: 5000,
        },
        controlTimeoutMs,
      );
      return false;
    }

    sourceSession = unwrap(
      await v2.session.get({
        sessionID,
      }),
    );
    if (sourceSession.revert) {
      throw new Error(COMPACT_PENDING_REVERT);
    }
    const currentCompactionCount = getCompactionCount(sourceSession) + 1;
    const sourceMessages = unwrap(await v2.session.messages({ sessionID }));
    const before = countModelFacingContentTokens(sourceMessages);
    const beforePhysicalTokens = before.tokens + before.projectedMediaTokens;

    await revalidateCompactionSource(v2, sessionID, sourcePlan);

    const summaries = await generateCompactionSummaries(
      v2,
      sourceSession,
      sourcePlan,
      { controlTimeoutMs },
    );

    const selection = await selectProfitableTurns(
      sourcePlan,
      summaries,
      sessionID,
    );
    if (
      selection.turns.length === 0
      && summaries.every(summary => summary === null)
    ) {
      await showMagicCompactToast(
        v2,
        {
          title: "Magic Compact",
          message: COMPACT_NOOP,
          variant: "info",
          duration: 5000,
        },
        controlTimeoutMs,
      );
      return false;
    }

    const projection = await projectCompaction(
      sourceMessages,
      sourcePlan,
      selection,
      sessionID,
      before,
      beforePhysicalTokens,
    );
    const eligibleTokens = countModelFacingContentTokens(
      selection.turns.flatMap(turn => turn.assistants),
    ).tokens;
    const projectedSavings = before.tokens - projection.afterTokens;
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
    if (unwrap(await v2.session.get({ sessionID })).revert) {
      throw new Error(COMPACT_PENDING_REVERT);
    }

    sourceWritesStarted = true;
    const insertedSummaries = await injectSummaries(
      v2,
      sessionID,
      selection.turns,
      selection.summaries,
    );

    // Mark the new compaction boundary for future recompactions
    // Message placed outside of summarization range so unaffected by pruning
    const boundaryInsertion = await injectPostCompactionNotice(
      v2,
      sessionID,
      sourcePlan.nextTurn,
    );

    await revalidateCompactionSourceAfterInsertions(v2, sessionID, sourcePlan, [
      ...insertedSummaries.map(part => ({
        type: "part" as const,
        messageID: part.messageID,
        part,
      })),
      boundaryInsertion,
    ]);
    await pruneSummarizedTurns({ v2, sessionID }, selection.pruning);

    await updateCompactionMetadata(v2, sourceSession, currentCompactionCount);
    await injectCompactStatsNotice(
      v2,
      sessionID,
      before.tokens,
      projection.afterTokens,
      currentCompactionCount,
      selection.turns.length,
    );
    const actual = await countSessionTokenProjection(v2, sessionID);
    if (
      actual.tokens !== projection.afterTokens
      || actual.projectedMediaTokens !== projection.projectedMediaTokens
    ) {
      throw new Error(
        "Compaction token projection mismatch after source writes.",
      );
    }
    await recordPruningStats({
      sessionID,
      sourceSessionID: sessionID,
      tokensPruned: projection.physicalTokensPruned,
      mediaTokensProjected: projection.projectedMediaTokens,
    });
    compactedTurnCount = selection.turns.length;
  } catch (error) {
    let failure = error;
    if (
      sourceWritesStarted
      && sourceSession !== null
      && backupSession !== null
    ) {
      failure = postWriteFailure(sourceSession.id, backupSession.id);
    }

    await showMagicCompactToast(
      v2,
      {
        title: "Magic Compact Failed",
        message: String(failure),
        variant: "error",
        duration: 8000,
      },
      controlTimeoutMs,
    );
    throw failure;
  }

  await showMagicCompactToast(
    v2,
    {
      title: "Magic Compact",
      message: `Compacted ${compactedTurnCount} assistant turn(s).`,
      variant: "info",
      duration: 5000,
    },
    controlTimeoutMs,
  );
  return true;
}

function postWriteFailure(
  sourceSessionID: string,
  backupSessionID: string,
): Error {
  return new Error(
    `Magic Compact failed after source writes began. The live source session ${sourceSessionID} and backup session ${backupSessionID} were both preserved; inspect both and recover manually from backup ${backupSessionID}.`,
  );
}

async function showMagicCompactToast(
  v2: V2Client,
  toast: MagicCompactToast,
  controlTimeoutMs: number,
): Promise<void> {
  await attemptWithinDeadline(
    signal => v2.tui.showToast(toast, { signal }).then(unwrap),
    controlTimeoutMs,
  );
}

async function selectProfitableTurns(
  plan: CompactionPlan,
  summaries: GeneratedSummary[],
  sessionID: string,
): Promise<CompactionSelection> {
  const selectedTurns: Turn[] = [];
  const selectedSummaries: Exclude<GeneratedSummary, null>[] = [];
  const projectedTurns: Turn[] = [];
  const mutations: PruneProjection["mutations"] = [];
  for (const [index, turn] of plan.summarizedTurns.entries()) {
    const summary = summaries[index];
    if (summary === undefined)
      throw new Error("Missing summary for assistant turn.");
    if (summary === null) continue;
    const candidates = [injectProjectedSummary(turn, summary, sessionID)];
    const pruning = await createPruneProjection(candidates, sessionID);
    const projectedTurn = pruning.turns[0]!;
    if (
      countTurnModelFacingTokens(projectedTurn)
      < countTurnModelFacingTokens(turn)
    ) {
      selectedTurns.push(turn);
      selectedSummaries.push(summary);
      projectedTurns.push(projectedTurn);
      mutations.push(...pruning.mutations);
    }
  }
  return {
    turns: selectedTurns,
    summaries: selectedSummaries,
    pruning: { turns: projectedTurns, mutations },
  };
}

function injectProjectedSummary(
  turn: Turn,
  summary: Exclude<GeneratedSummary, null>,
  sessionID: string,
): Turn {
  const projected = structuredClone(turn);
  const firstAssistant = projected.assistants[0];
  if (!firstAssistant) throw new Error("Turn missing assistant message.");
  firstAssistant.parts.push(
    summaryEnvelopeTextPart(
      firstAssistant.info.id,
      sessionID,
      summary.text,
      summary.mediaManifest,
    ),
  );
  return projected;
}

function countTurnModelFacingTokens(turn: Turn): number {
  return countModelFacingContentTokens([...turn.user, ...turn.assistants])
    .tokens;
}

async function projectCompaction(
  sourceMessages: MessageWithParts[],
  plan: CompactionPlan,
  selection: CompactionSelection,
  sessionID: string,
  before: ModelFacingTokenCount,
  beforePhysicalTokens: number,
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
  const after = countModelFacingContentTokens(projectedMessages);
  const afterPhysicalTokens = after.tokens + after.projectedMediaTokens;
  const physicalTokensPruned = beforePhysicalTokens - afterPhysicalTokens;
  const newMediaTokensProjected =
    after.projectedMediaTokens - before.projectedMediaTokens;
  if (newMediaTokensProjected < 0) {
    throw new Error("Magic Compact media projection unexpectedly decreased.");
  }
  return {
    afterTokens: after.tokens,
    projectedMediaTokens: after.projectedMediaTokens,
    physicalTokensPruned,
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

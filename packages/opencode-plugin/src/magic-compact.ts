import type { V2Client } from "./api";
import { COMPACT_NOOP } from "./commands";
import {
  generateCompactionSummaries,
  type GeneratedSummary,
  type SummaryReasoningOptions,
} from "./compact/compact";
import { estimateReplay, indexReplay } from "./compact/reasoning";
import {
  continuationTokens,
  reasoningFeedback,
  replayFeedback,
  type ReplayEstimate,
} from "./policy/reasoning";
import { getCompactionCount } from "./compact/session";
import {
  nativeModelVisibleMessages,
  planCompaction,
  type CompactionPlan,
  type MessageWithParts,
  type Turn,
} from "./compact/plan";
import { createPruneProjection, type PruneProjection } from "./compact/prune";
import {
  appendedNoticeMessage,
  applySuccessorEdits,
  assertSourceUnchanged,
  createSuccessor,
  readSourceAtRest,
  withExclusiveSource,
  type Successor,
  type SuccessorEdit,
} from "./compact/successor";
import {
  preservedTurnsFeedback,
  STATS_METADATA,
  statsMessage,
  type PreservedTurns,
} from "./stats/constants";
import { countModelFacingContentTokens } from "./stats/tokenize";
import {
  BOUNDARY_METADATA,
  POST_COMPACTION_NOTICE,
  boundaryPartID,
  summaryEnvelopeTextPart,
  turnHasPersistedSummary,
} from "./compact/constants";
import {
  COMPACT_INSUFFICIENT_SAVINGS,
  hasUsefulSavings,
} from "./policy/successor";
import {
  CONTROL_OPERATION_TIMEOUT_MS,
  createToastFeedback,
  failureToast,
  IN_PROGRESS_TOAST_DURATION_MS,
} from "./util";
import {
  decideRetention,
  findRetentionCandidates,
  HEURISTIC_RETENTION,
  retentionFeedback,
  retentionWithoutCandidates,
  type RetentionPolicy,
} from "./retention/retention";
import { PluginMessageError } from "./policy/feedback";

export {
  COMPACT_ARGUMENTS_ERROR,
  COMPACT_BACKEND_ERROR,
  COMPACT_NOOP,
  COMPACT_THRESHOLD_ERROR,
  compactSuccess,
  parseMagicCompactArguments,
} from "./commands";
export { COMPACT_INSUFFICIENT_SAVINGS };
const COMMAND = "Magic Compact";

type CompactionSelection = {
  turns: Turn[];
  summaries: Exclude<GeneratedSummary, null>[];
  pruning: PruneProjection;
};

export type MagicCompactExecutionOptions = {
  // Heuristic skips only the retention step; turn summaries are generated
  // either way.
  retention: RetentionPolicy;
  // `--match-model`: run each summary on the model that wrote the protected
  // reasoning it covers, where the route is verified.
  matchModel?: boolean;
  controlTimeoutMs?: number;
};

// Returns the ready successor, or null for a no-op. The source session and its
// children are only read.
export async function executeMagicCompact(
  v2: V2Client,
  sessionID: string,
  keepTurns: number,
  options: MagicCompactExecutionOptions,
): Promise<Successor | null> {
  const controlTimeoutMs =
    options.controlTimeoutMs ?? CONTROL_OPERATION_TIMEOUT_MS;
  if (!Number.isFinite(controlTimeoutMs) || controlTimeoutMs <= 0) {
    throw new Error("Magic Compact control deadline must be positive.");
  }
  const toastFeedback = createToastFeedback(v2, controlTimeoutMs);
  await toastFeedback.showProgress({
    title: COMMAND,
    message: "Compacting session...",
    variant: "info",
    duration: IN_PROGRESS_TOAST_DURATION_MS,
  });
  let successor: Successor;
  let summarizedTurnCount: number;
  let preserved: PreservedTurns;
  let retentionLine: string;
  let reasoningLine: string | undefined;

  try {
    const outcome = await withExclusiveSource(COMMAND, sessionID, async () => {
      const source = await readSourceAtRest(
        v2,
        COMMAND,
        sessionID,
        controlTimeoutMs,
      );
      const plan = planCompaction(source.messages, keepTurns);
      preserved = {
        requested: keepTurns,
        preserved: plan.preservedTurnCount,
      };
      if (plan.summarizedTurns.length === 0) {
        retentionLine = retentionFeedback(
          retentionWithoutCandidates(options.retention),
        );
        return null;
      }

      const before = countModelFacingContentTokens(source.messages);
      const reasoning: SummaryReasoningOptions = {
        matchModel: options.matchModel === true,
        reports: [],
      };
      // Opaque summary-bearing turns are never pruned, so they hold no
      // candidates. Retention runs before per-turn generation.
      const retention =
        options.retention.mode === "task-aware"
          ? await decideRetention(
              v2,
              source.session,
              source.messages,
              await findRetentionCandidates(
                source.messages,
                plan.summarizedTurns.filter(
                  turn => !turnHasPersistedSummary(turn),
                ),
              ),
              options.retention,
              { controlTimeoutMs, reasoning },
            )
          : HEURISTIC_RETENTION;
      retentionLine = retentionFeedback(retention, source.messages);
      const summaries = await generateCompactionSummaries(
        v2,
        source.session,
        plan,
        { controlTimeoutMs, reasoning },
      );
      reasoningLine = reasoningFeedback(
        reasoning.reports,
        reasoning.matchModel,
      );
      // Profitability and the savings gate measure what the continuing model
      // is sent: visible content plus the estimated replay cost of protected
      // reasoning that can be attributed. Unattributed reasoning adds nothing
      // and is reported as not counted. Both sides of every comparison use
      // the source's one attribution, so only the selected turns' own
      // reasoning, which compaction removes whole, can change a result.
      const sourceReplay = indexReplay(
        nativeModelVisibleMessages(source.messages),
      );
      const replayOf = (turns: Turn[]) =>
        sourceReplay.estimate(
          reasoning.replay,
          new Set(
            turns.flatMap(turn => turnMessages(turn).map(m => m.info.id)),
          ),
        );
      const selection = await selectProfitableTurns(
        plan,
        summaries,
        sessionID,
        retention.keep,
        replayOf,
      );
      if (selection.turns.length === 0) {
        if (summaries.every(summary => summary === null)) return null;
        throw new PluginMessageError(COMPACT_INSUFFICIENT_SAVINGS);
      }

      const compacted = applySuccessorEdits(
        source.messages,
        compactionEdits(source, plan, selection),
      );
      const after = countModelFacingContentTokens(compacted);
      const sourceEstimate = sourceReplay.estimate(reasoning.replay);
      const removedReplay = replayOf(selection.turns);
      const eligibleTokens = continuationTokens(
        countModelFacingContentTokens(
          selection.turns.flatMap(turn => turn.assistants),
        ).tokens,
        removedReplay,
      );
      if (
        !hasUsefulSavings(
          before.tokens
            - after.tokens
            - removedReplay.visibleTokens
            + removedReplay.tokens,
          eligibleTokens,
        )
      ) {
        throw new PluginMessageError(COMPACT_INSUFFICIENT_SAVINGS);
      }
      if (after.projectedMediaTokens < before.projectedMediaTokens) {
        throw new Error(
          "Magic Compact media projection unexpectedly decreased.",
        );
      }

      const compactionCount = getCompactionCount(source.session) + 1;
      const replayLine = replayFeedback(sourceEstimate, removedReplay);
      // The ignored notice is excluded from canonical counts, so it does not
      // change the `after` it reports.
      const history = applySuccessorEdits(compacted, [
        {
          type: "append-message",
          message: appendedNoticeMessage(source, "stats", [
            {
              type: "text",
              text: statsMessage(
                compactionCount,
                before.tokens,
                after.tokens,
                selection.turns.length,
                preserved,
                retentionLine,
                [
                  ...(reasoningLine === undefined
                    ? []
                    : [`Reasoning | ${reasoningLine}`]),
                  ...(replayLine === undefined
                    ? []
                    : [`Reasoning replay (estimate) | ${replayLine}`]),
                ],
              ),
              ignored: true,
              metadata: STATS_METADATA,
            },
          ]),
        },
      ]);
      const created = await createSuccessor(v2, {
        command: COMMAND,
        source,
        history,
        operation: "compact",
        compactionCount,
        omissions: selection.pruning.mutations.flatMap(mutation =>
          mutation.type === "update-part" ? mutation.omissions : [],
        ),
        tokensPruned:
          before.tokens
          + before.projectedMediaTokens
          - (after.tokens + after.projectedMediaTokens),
        mediaTokensProjected: after.projectedMediaTokens,
        recheckSource: async () =>
          assertSourceUnchanged(
            COMMAND,
            source,
            await readSourceAtRest(v2, COMMAND, sessionID, controlTimeoutMs),
          ),
        controlTimeoutMs,
      });
      const actual = countModelFacingContentTokens(created.messages);
      // The persisted successor, under its new IDs, must estimate exactly as
      // the projection did.
      const replay = (messages: MessageWithParts[]) =>
        estimateReplay(nativeModelVisibleMessages(messages), reasoning.replay);
      const afterReplay = replay(compacted);
      const actualReplay = replay(created.messages);
      if (
        actual.tokens !== after.tokens
        || actual.projectedMediaTokens !== after.projectedMediaTokens
        || actualReplay.tokens !== afterReplay.tokens
        || actualReplay.visibleTokens !== afterReplay.visibleTokens
        || actualReplay.units !== afterReplay.units
        || actualReplay.unknown !== afterReplay.unknown
      ) {
        throw new PluginMessageError(
          `Compaction token projection mismatch in successor session ${created.session.id}.`,
        );
      }
      summarizedTurnCount = selection.turns.length;
      return created;
    });
    if (outcome === null) {
      await toastFeedback.showTerminal({
        title: COMMAND,
        message: `${noopFeedback(preserved!)} ${retentionLine!}${reasoningLine === undefined ? "" : ` ${reasoningLine}`}`,
        variant: "info",
        duration: 5000,
      });
      return null;
    }
    successor = outcome;
  } catch (error) {
    await toastFeedback.showTerminal(failureToast(COMMAND, error));
    throw error;
  }

  await toastFeedback.showTerminal({
    title: COMMAND,
    message: `Compacted ${summarizedTurnCount!} request turn(s) into successor session ${successor.session.id}. ${preservedTurnsFeedback(preserved!)} ${retentionLine!}${reasoningLine === undefined ? "" : ` ${reasoningLine}`}`,
    variant: "info",
    duration: 5000,
  });
  return successor;
}

// Nothing preserved means N did not shape the outcome, so the message stays
// the plain no-op text.
function noopFeedback(preserved: PreservedTurns): string {
  return preserved.preserved === 0
    ? COMPACT_NOOP
    : `${COMPACT_NOOP} ${preservedTurnsFeedback(preserved)}`;
}

// Summaries lead their first assistant message and the boundary leads the next
// user message, matching where earlier in-place releases stored them.
function compactionEdits(
  source: Parameters<typeof appendedNoticeMessage>[0],
  plan: CompactionPlan,
  selection: CompactionSelection,
): SuccessorEdit[] {
  const sessionID = source.session.id;
  const edits: SuccessorEdit[] = [];
  for (const [index, turn] of selection.turns.entries()) {
    const assistant = turn.assistants[0];
    const summary = selection.summaries[index];
    if (!assistant || !summary)
      throw new Error("Missing summary for assistant turn.");
    edits.push({
      type: "prepend-part",
      messageID: assistant.info.id,
      part: summaryEnvelopeTextPart(
        assistant.info.id,
        sessionID,
        summary.text,
        summary.mediaManifest,
      ),
    });
  }
  for (const mutation of selection.pruning.mutations) {
    if (mutation.type === "update-part") {
      edits.push({
        type: "replace-part",
        messageID: mutation.messageID,
        part: mutation.replacement,
      });
    } else {
      edits.push(mutation);
    }
  }
  const boundary = {
    type: "text" as const,
    text: POST_COMPACTION_NOTICE,
    synthetic: true,
    metadata: BOUNDARY_METADATA,
  };
  const nextUser = plan.nextTurn?.user[0];
  if (nextUser) {
    edits.push({
      type: "prepend-part",
      messageID: nextUser.info.id,
      part: {
        ...boundary,
        id: boundaryPartID(nextUser.info.id),
        sessionID,
        messageID: nextUser.info.id,
      },
    });
  } else {
    edits.push({
      type: "append-message",
      message: appendedNoticeMessage(source, "boundary", [boundary]),
    });
  }
  return edits;
}

// A turn is replaced only when the continuing model would be sent strictly
// less: its visible content with the summary injected and the planned pruning
// applied, against the source turn's visible content with the estimated
// replay cost of its attributable protected reasoning. Pruning deletes every
// reasoning part of a summarized turn, so the projected turn replays none.
// Reasoning whose cost cannot be attributed counts only by its visible text.
async function selectProfitableTurns(
  plan: CompactionPlan,
  summaries: GeneratedSummary[],
  sessionID: string,
  keep: ReadonlySet<string>,
  replayOf: (turns: Turn[]) => ReplayEstimate,
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
    const pruning = await createPruneProjection(candidates, sessionID, keep);
    const projectedTurn = pruning.turns[0]!;
    if (
      turnMessages(projectedTurn).some(message =>
        message.parts.some(part => part.type === "reasoning"),
      )
    ) {
      throw new Error("Magic Compact pruning kept a reasoning part.");
    }
    const sourceCost = continuationTokens(
      countTurnModelFacingTokens(turn),
      replayOf([turn]),
    );
    if (countTurnModelFacingTokens(projectedTurn) < sourceCost) {
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

function turnMessages(turn: Turn): MessageWithParts[] {
  return [...turn.user, ...turn.assistants] as MessageWithParts[];
}

function injectProjectedSummary(
  turn: Turn,
  summary: Exclude<GeneratedSummary, null>,
  sessionID: string,
): Turn {
  const projected = structuredClone(turn);
  const firstAssistant = projected.assistants[0];
  if (!firstAssistant) throw new Error("Turn missing assistant message.");
  firstAssistant.parts.unshift(
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
  return countModelFacingContentTokens([
    ...turn.user,
    ...turn.assistants,
  ] as MessageWithParts[]).tokens;
}

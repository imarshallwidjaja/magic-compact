import type { V2Client } from "./api";
import { COMPACT_NOOP } from "./commands";
import {
  generateCompactionSummaries,
  summarizerCatalogLoader,
  type GeneratedSummary,
} from "./compact/compact";
import { estimateReplay, indexReplay } from "./compact/reasoning";
import {
  continuationTokens,
  reasoningFeedback,
  replayFeedback,
  type ReplayEstimate,
} from "./policy/reasoning";
import { getCompactionCount } from "./policy/lineage";
import {
  nativeModelVisibleMessages,
  planCompaction,
  type CompactionPlan,
  type CompactionTarget,
  type MessageWithParts,
  type Turn,
} from "./compact/plan";
import { createPruneProjection, type PruneProjection } from "./compact/prune";
import {
  appendedNoticeMessage,
  applySuccessorEdits,
  createSuccessor,
  readSourceAtRest,
  type Successor,
  type SuccessorEdit,
} from "./compact/successor";
import {
  preservedTurnsFeedback,
  STATS_METADATA,
  statsMessage,
  type PreservedTurns,
} from "./stats/constants";
import { countModelFacingContentTokens } from "./compact/accounting";
import { BOUNDARY_METADATA, POST_COMPACTION_NOTICE } from "./policy/protocol";
import { boundaryPartID, summaryEnvelopeTextPart } from "./compact/capsule";
import {
  COMPACT_INSUFFICIENT_SAVINGS,
  hasUsefulSavings,
  withExclusiveSource,
} from "./policy/successor";
import {
  createToastFeedback,
  failureToast,
  IN_PROGRESS_TOAST_DURATION_MS,
} from "./toast-v1";
import { CONTROL_OPERATION_TIMEOUT_MS } from "./util";
import {
  decideRetention,
  findRetentionCandidates,
  HEURISTIC_RETENTION,
  retentionFeedback,
  retentionWithoutCandidates,
  type RetentionPolicy,
} from "./retention/retention";
import { PluginMessageError } from "./policy/feedback";
import { assertStatsUsable } from "./storage/stats";

const COMMAND = "Magic Compact";

type CompactionSelection = {
  turns: Turn[];
  summaries: GeneratedSummary[];
  pruning: PruneProjection;
};

// What a command reports: a no-op, or the verified successor. `reasoningLine`
// appears only when there is something to say about summary models.
type CompactionOutcome = {
  preserved: PreservedTurns;
  retentionLine: string;
  reasoningLine: string | undefined;
} & (
  | { kind: "noop" }
  | { kind: "compacted"; successor: Successor; summarizedTurns: number }
);

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
  let outcome: CompactionOutcome;
  try {
    outcome = await withExclusiveSource(
      COMMAND,
      sessionID,
      async (): Promise<CompactionOutcome> => {
        const source = await readSourceAtRest(
          v2,
          COMMAND,
          sessionID,
          controlTimeoutMs,
        );
        const plan = planCompaction(source.messages, keepTurns);
        const preserved = {
          requested: keepTurns,
          preserved: plan.preservedTurnCount,
        };
        // Turns that already hold a summary need no request, so a command
        // with no other turn is a no-op that reads no saved totals, catalog
        // or model.
        if (plan.targets.length === 0) {
          return {
            kind: "noop",
            preserved,
            retentionLine: retentionFeedback(
              retentionWithoutCandidates(options.retention),
            ),
            reasoningLine: undefined,
          };
        }
        await assertStatsUsable(sessionID);

        const before = await countModelFacingContentTokens(source.messages);
        const matchModel = options.matchModel === true;
        // Resolved on first need and shared by the retention summary and
        // every turn request.
        const catalog = summarizerCatalogLoader(v2, source.session);
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
                  plan.targets.map(target => target.turn),
                ),
                options.retention,
                { controlTimeoutMs, matchModel, catalog },
              )
            : { plan: HEURISTIC_RETENTION, reports: [] };
        const retentionLine = retentionFeedback(
          retention.plan,
          source.messages,
        );
        const generation = await generateCompactionSummaries(
          v2,
          source.session,
          plan,
          { controlTimeoutMs, matchModel, catalog },
        );
        const reasoningLine = reasoningFeedback(
          [...retention.reports, ...generation.reports],
          matchModel,
        );
        // Profitability and the savings gate measure what the continuing
        // model is sent: visible content plus the estimated replay cost of
        // protected reasoning that can be attributed. Unattributed reasoning
        // adds nothing and is reported as not counted. Both sides of every
        // comparison use the source's one attribution, so only the selected
        // turns' own reasoning, which compaction removes whole, can change a
        // result.
        const sourceReplay = indexReplay(
          nativeModelVisibleMessages(source.messages),
        );
        const replayOf = (turns: Turn[]) =>
          sourceReplay.estimate(
            generation.replay,
            new Set(
              turns.flatMap(turn => turnMessages(turn).map(m => m.info.id)),
            ),
          );
        const selection = await selectProfitableTurns(
          plan.targets,
          generation.summaries,
          sessionID,
          retention.plan.keep,
          replayOf,
        );
        if (selection.turns.length === 0) {
          throw new PluginMessageError(COMPACT_INSUFFICIENT_SAVINGS);
        }

        const compacted = applySuccessorEdits(
          source.messages,
          compactionEdits(source, plan, selection),
        );
        const after = await countModelFacingContentTokens(compacted);
        const sourceEstimate = sourceReplay.estimate(generation.replay);
        const removedReplay = replayOf(selection.turns);
        const eligibleTokens = continuationTokens(
          (
            await countModelFacingContentTokens(
              selection.turns.flatMap(turn => turn.assistants),
            )
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
        // The ignored notice is excluded from canonical counts, so it does
        // not change the `after` it reports.
        const history = applySuccessorEdits(compacted, [
          {
            type: "append-message",
            message: appendedNoticeMessage(source, "stats", [
              {
                type: "text",
                text: statsMessage(
                  compactionCount,
                  before,
                  after,
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
          controlTimeoutMs,
        });
        const actual = await countModelFacingContentTokens(created.messages);
        // The persisted successor, under its new IDs, must estimate exactly as
        // the projection did.
        const replay = (messages: MessageWithParts[]) =>
          estimateReplay(
            nativeModelVisibleMessages(messages),
            generation.replay,
          );
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
        return {
          kind: "compacted",
          successor: created,
          summarizedTurns: selection.turns.length,
          preserved,
          retentionLine,
          reasoningLine,
        };
      },
    );
  } catch (error) {
    await toastFeedback.showTerminal(failureToast(COMMAND, error));
    throw error;
  }

  const lines = `${outcome.retentionLine}${outcome.reasoningLine === undefined ? "" : ` ${outcome.reasoningLine}`}`;
  if (outcome.kind === "noop") {
    await toastFeedback.showTerminal({
      title: COMMAND,
      message: `${noopFeedback(outcome.preserved)} ${lines}`,
      variant: "info",
      duration: 5000,
    });
    return null;
  }
  await toastFeedback.showTerminal({
    title: COMMAND,
    message: `Compacted ${outcome.summarizedTurns} request turn(s) into successor session ${outcome.successor.session.id}. ${preservedTurnsFeedback(outcome.preserved)} ${lines}`,
    variant: "info",
    duration: 5000,
  });
  return outcome.successor;
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
  targets: CompactionTarget[],
  summaries: GeneratedSummary[],
  sessionID: string,
  keep: ReadonlySet<string>,
  replayOf: (turns: Turn[]) => ReplayEstimate,
): Promise<CompactionSelection> {
  const selectedTurns: Turn[] = [];
  const selectedSummaries: GeneratedSummary[] = [];
  const projectedTurns: Turn[] = [];
  const mutations: PruneProjection["mutations"] = [];
  for (const [index, { turn }] of targets.entries()) {
    const summary = summaries[index];
    if (summary === undefined)
      throw new Error("Missing summary for assistant turn.");
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
      await countTurnModelFacingTokens(turn),
      replayOf([turn]),
    );
    if ((await countTurnModelFacingTokens(projectedTurn)) < sourceCost) {
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
  summary: GeneratedSummary,
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

async function countTurnModelFacingTokens(turn: Turn): Promise<number> {
  return (
    await countModelFacingContentTokens([
      ...turn.user,
      ...turn.assistants,
    ] as MessageWithParts[])
  ).tokens;
}

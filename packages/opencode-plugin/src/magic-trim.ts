import type { V2Client } from "./api";
import { TRIM_NOOP } from "./commands";
import { planTrim } from "./compact/plan";
import { projectToolTrim } from "./compact/prune";
import { getCompactionCount } from "./policy/lineage";
import {
  appendedNoticeMessage,
  applySuccessorEdits,
  createSuccessor,
  readSourceAtRest,
  type Successor,
} from "./compact/successor";
import { withExclusiveSource } from "./policy/successor";
import {
  decideRetention,
  findRetentionCandidates,
  HEURISTIC_RETENTION,
  retentionFeedback,
  type RetentionDecision,
  type RetentionPolicy,
} from "./retention/retention";
import { STATS_METADATA, trimStatsMessage } from "./stats/constants";
import { summarizerCatalogLoader } from "./compact/compact";
import { reasoningFeedback } from "./policy/reasoning";
import { countModelFacingContentTokens } from "./compact/accounting";
import {
  createToastFeedback,
  failureToast,
  IN_PROGRESS_TOAST_DURATION_MS,
} from "./toast-v1";
import { CONTROL_OPERATION_TIMEOUT_MS } from "./util";
import { PluginMessageError } from "./policy/feedback";
import { assertStatsUsable } from "./storage/stats";

const COMMAND = "Magic Trim";

export type MagicTrimExecutionOptions = {
  retention: RetentionPolicy;
  // `--match-model` for the retention session summary.
  matchModel?: boolean;
  controlTimeoutMs?: number;
};

// Returns the ready successor, or null for a no-op. Trim covers the complete
// stored history, independent of compaction boundaries, and never writes the
// source session.
export async function executeMagicTrim(
  v2: V2Client,
  sessionID: string,
  keepTurns: number,
  options: MagicTrimExecutionOptions,
): Promise<Successor | null> {
  const controlTimeoutMs =
    options.controlTimeoutMs ?? CONTROL_OPERATION_TIMEOUT_MS;
  if (!Number.isFinite(controlTimeoutMs) || controlTimeoutMs <= 0) {
    throw new Error("Magic Trim control deadline must be positive.");
  }
  const toastFeedback = createToastFeedback(v2, controlTimeoutMs);
  await toastFeedback.showProgress({
    title: COMMAND,
    message: "Trimming session...",
    variant: "info",
    duration: IN_PROGRESS_TOAST_DURATION_MS,
  });

  try {
    const outcome = await withExclusiveSource(COMMAND, sessionID, async () => {
      const source = await readSourceAtRest(
        v2,
        COMMAND,
        sessionID,
        controlTimeoutMs,
      );
      const plan = planTrim(source.messages, keepTurns);
      const matchModel = options.matchModel === true;
      let retention: RetentionDecision = {
        plan: HEURISTIC_RETENTION,
        reports: [],
      };
      if (options.retention.mode === "task-aware") {
        const found = await findRetentionCandidates(
          source.messages,
          plan.trimmedTurns,
        );
        // Candidates mean retention requests and a likely successor.
        if (found.candidates.length > 0) await assertStatsUsable(sessionID);
        retention = await decideRetention(
          v2,
          source.session,
          source.messages,
          found,
          options.retention,
          {
            controlTimeoutMs,
            matchModel,
            catalog: summarizerCatalogLoader(v2, source.session),
          },
        );
      }
      const reasoningLine = reasoningFeedback(retention.reports, matchModel);
      const retentionLine = retentionFeedback(retention.plan, source.messages);
      // The toast carries the reasoning line after the retention line; the
      // notice gives it its own line.
      const feedback = `${retentionLine}${
        reasoningLine === undefined ? "" : ` ${reasoningLine}`
      }`;
      const trim = await projectToolTrim(
        plan.trimmedTurns,
        sessionID,
        retention.plan.keep,
      );
      if (trim.replacements.length === 0) {
        return { successor: null, feedback };
      }

      const trimmed = applySuccessorEdits(
        source.messages,
        trim.replacements.map(part => ({
          type: "replace-part" as const,
          messageID: part.messageID,
          part,
        })),
      );
      const before = await countModelFacingContentTokens(source.messages);
      const after = await countModelFacingContentTokens(trimmed);
      const history = applySuccessorEdits(trimmed, [
        {
          type: "append-message",
          message: appendedNoticeMessage(source, "stats", [
            {
              type: "text",
              text: trimStatsMessage(
                before,
                after,
                trim.replacements.length,
                retentionLine,
                reasoningLine === undefined
                  ? []
                  : [`Reasoning | ${reasoningLine}`],
              ),
              ignored: true,
              metadata: STATS_METADATA,
            },
          ]),
        },
      ]);
      const successor = await createSuccessor(v2, {
        command: COMMAND,
        source,
        history,
        operation: "trim",
        compactionCount: getCompactionCount(source.session),
        omissions: trim.omissions,
        tokensPruned: before.tokens - after.tokens,
        mediaTokensProjected: after.projectedMediaTokens,
        controlTimeoutMs,
      });
      const actual = await countModelFacingContentTokens(successor.messages);
      if (
        actual.tokens !== after.tokens
        || actual.projectedMediaTokens !== after.projectedMediaTokens
      ) {
        throw new PluginMessageError(
          `Trim final canonical active token count did not match its projection in successor session ${successor.session.id} (projected ${after.tokens}, actual ${actual.tokens}).`,
        );
      }
      return {
        successor,
        feedback,
        trimmedToolCalls: trim.replacements.length,
      };
    });

    if (outcome.successor === null) {
      await toastFeedback.showTerminal({
        title: COMMAND,
        message: `${TRIM_NOOP} ${outcome.feedback}`,
        variant: "info",
        duration: 5000,
      });
      return null;
    }
    await toastFeedback.showTerminal({
      title: COMMAND,
      message: `Trimmed ${outcome.trimmedToolCalls} tool call(s) into successor session ${outcome.successor.session.id}. ${outcome.feedback}`,
      variant: "info",
      duration: 5000,
    });
    return outcome.successor;
  } catch (error) {
    await toastFeedback.showTerminal(failureToast(COMMAND, error));
    throw error;
  }
}

import type { V2Client } from "./api";
import { TRIM_NOOP } from "./commands";
import { planTrim } from "./compact/plan";
import { projectToolTrim } from "./compact/prune";
import { getCompactionCount } from "./compact/session";
import {
  appendedNoticeMessage,
  applySuccessorEdits,
  assertSourceUnchanged,
  createSuccessor,
  readSourceAtRest,
  withExclusiveSource,
  type Successor,
} from "./compact/successor";
import {
  decideRetention,
  findRetentionCandidates,
  HEURISTIC_RETENTION,
  retentionFeedback,
  type RetentionPlan,
  type RetentionPolicy,
} from "./retention/retention";
import { STATS_METADATA, trimStatsMessage } from "./stats/constants";
import { countModelFacingContentTokens } from "./stats/tokenize";
import {
  CONTROL_OPERATION_TIMEOUT_MS,
  createToastFeedback,
  IN_PROGRESS_TOAST_DURATION_MS,
} from "./util";

const COMMAND = "Magic Trim";

export {
  parseMagicTrimArguments,
  TRIM_ARGUMENTS_ERROR,
  TRIM_NOOP,
  trimSuccess,
} from "./commands";

export type MagicTrimExecutionOptions = {
  retention: RetentionPolicy;
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
      let retention: RetentionPlan = HEURISTIC_RETENTION;
      if (options.retention.mode === "task-aware") {
        retention = await decideRetention(
          v2,
          source.session,
          source.messages,
          await findRetentionCandidates(source.messages, plan.trimmedTurns),
          options.retention,
          { controlTimeoutMs },
        );
      }
      const retentionLine = retentionFeedback(retention, source.messages);
      const trim = await projectToolTrim(
        plan.trimmedTurns,
        sessionID,
        retention.keep,
      );
      if (trim.replacements.length === 0) {
        return { successor: null, retentionLine };
      }

      const trimmed = applySuccessorEdits(
        source.messages,
        trim.replacements.map(part => ({
          type: "replace-part" as const,
          messageID: part.messageID,
          part,
        })),
      );
      const before = countModelFacingContentTokens(source.messages);
      const after = countModelFacingContentTokens(trimmed);
      const history = applySuccessorEdits(trimmed, [
        {
          type: "append-message",
          message: appendedNoticeMessage(source, "stats", [
            {
              type: "text",
              text: trimStatsMessage(
                before.tokens,
                after.tokens,
                trim.replacements.length,
                retentionLine,
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
        recheckSource: async () =>
          assertSourceUnchanged(
            COMMAND,
            source,
            await readSourceAtRest(v2, COMMAND, sessionID, controlTimeoutMs),
          ),
        controlTimeoutMs,
      });
      const actual = countModelFacingContentTokens(successor.messages).tokens;
      if (actual !== after.tokens) {
        throw new Error(
          `Trim final canonical active token count did not match its projection in successor session ${successor.session.id} (projected ${after.tokens}, actual ${actual}).`,
        );
      }
      return {
        successor,
        retentionLine,
        trimmedToolCalls: trim.replacements.length,
      };
    });

    if (outcome.successor === null) {
      await toastFeedback.showTerminal({
        title: COMMAND,
        message: `${TRIM_NOOP} ${outcome.retentionLine}`,
        variant: "info",
        duration: 5000,
      });
      return null;
    }
    await toastFeedback.showTerminal({
      title: COMMAND,
      message: `Trimmed ${outcome.trimmedToolCalls} tool call(s) into successor session ${outcome.successor.session.id}. ${outcome.retentionLine}`,
      variant: "info",
      duration: 5000,
    });
    return outcome.successor;
  } catch (error) {
    await toastFeedback.showTerminal({
      title: "Magic Trim Failed",
      message: String(error),
      variant: "error",
      duration: 8000,
    });
    throw error;
  }
}

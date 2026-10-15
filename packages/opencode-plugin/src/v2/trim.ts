import { getCompactionCount } from "../compact/session";
import {
  collectRetentionCandidates,
  decideRetentionWith,
  formatRetentionFeedback,
  HEURISTIC_RETENTION,
  type RetentionPlan,
  type RetentionPolicy,
} from "../policy/retention";
import { withExclusiveSource } from "../policy/successor";
import { trimmedTurnRange } from "../policy/turns";
import { trimStatsMessage } from "../stats/constants";
import { planOmission, type PlannedOmission } from "../storage/omission";
import { CONTROL_OPERATION_TIMEOUT_MS } from "../util";
import {
  trimV2Tool,
  v2ToolCalls,
  v2RetentionTurns,
  v2ToolCallView,
  v2Turns,
} from "./history";
import type { V2Host } from "./host";
import { countV2ModelFacingTokens } from "./media";
import {
  generateV2SessionSummary,
  resolveV2SummarizerModel,
  type OwnedSessions,
} from "./owned";
import {
  assertV2SourceUnchanged,
  createV2Successor,
  isInstructionNotice,
  readV2SourceAtRest,
  type V2Message,
  type V2Successor,
} from "./session";

const COMMAND = "Magic Trim";

export type V2TrimOptions = {
  retention: RetentionPolicy;
  controlTimeoutMs?: number;
};

// Returns the ready successor, or null for a no-op. Trim covers the complete
// stored history, independent of compaction boundaries, and never writes the
// source session.
export async function executeV2Trim(
  host: V2Host,
  owned: OwnedSessions,
  sessionID: string,
  keepTurns: number,
  options: V2TrimOptions,
): Promise<{ successor: V2Successor | null; outcome: string }> {
  const controlTimeoutMs =
    options.controlTimeoutMs ?? CONTROL_OPERATION_TIMEOUT_MS;
  return withExclusiveSource(COMMAND, sessionID, async () => {
    const source = await readV2SourceAtRest(
      host,
      COMMAND,
      sessionID,
      controlTimeoutMs,
    );
    const turns = v2Turns(source.messages);
    const eligible = new Set(
      trimmedTurnRange(turns, keepTurns).flatMap(turn =>
        turn.assistants.flatMap(message =>
          v2ToolCalls(message)
            .filter(call => v2ToolCallView(call.tool) !== null)
            .map(call => call.ref),
        ),
      ),
    );
    let retention: RetentionPlan = HEURISTIC_RETENTION;
    if (options.retention.mode === "task-aware") {
      const policy = options.retention;
      retention = await decideRetentionWith(
        await collectRetentionCandidates(
          v2RetentionTurns(source.messages),
          eligible,
        ),
        policy,
        async () =>
          generateV2SessionSummary(
            host,
            owned,
            source,
            await resolveV2SummarizerModel(host, source, controlTimeoutMs),
            controlTimeoutMs,
          ),
      );
    }
    const retentionLine = formatRetentionFeedback(
      retention,
      source.messages.flatMap(message =>
        v2ToolCalls(message).map(call => ({
          ref: call.ref,
          tool: call.tool.name,
        })),
      ),
    );

    const omissions: PlannedOmission[] = [];
    let trimmedCalls = 0;
    const edited: V2Message[] = [];
    for (const message of source.messages) {
      const calls = v2ToolCalls(message).filter(
        call => eligible.has(call.ref) && !retention.keep.has(call.ref),
      );
      if (calls.length === 0 || message.type !== "assistant") {
        edited.push(message);
        continue;
      }
      const content = [...message.content];
      for (const call of calls) {
        const replacement = await trimV2Tool(call.tool, async entry => {
          const omission = planOmission(sessionID, entry);
          omissions.push(omission);
          return omission.contentID;
        });
        if (replacement === null) continue;
        content[call.index] = replacement;
        trimmedCalls += 1;
      }
      edited.push({ ...message, content });
    }
    if (trimmedCalls === 0) {
      return { successor: null, outcome: retentionLine };
    }

    // Trim changes no image, so existing manifests project the same images
    // before and after; both counts are model-facing.
    const before = countV2ModelFacingTokens(source.messages);
    const after = countV2ModelFacingTokens(
      edited.filter(message => !isInstructionNotice(message)),
    );
    const outcome = trimStatsMessage(
      before.tokens,
      after.tokens,
      trimmedCalls,
      retentionLine,
    );
    const successor = await createV2Successor(host, {
      command: COMMAND,
      source,
      messages: edited,
      operation: "trim",
      compactionCount: getCompactionCount(source.session),
      outcome,
      omissions,
      tokensPruned: before.tokens - after.tokens,
      mediaTokensProjected: after.projectedMediaTokens,
      recheckSource: async () =>
        assertV2SourceUnchanged(
          COMMAND,
          source,
          await readV2SourceAtRest(host, COMMAND, sessionID, controlTimeoutMs),
        ),
      controlTimeoutMs,
    });
    const actual = countV2ModelFacingTokens(successor.messages);
    if (
      actual.tokens !== after.tokens
      || actual.projectedMediaTokens !== after.projectedMediaTokens
    ) {
      throw new Error(
        `Trim final canonical active token count did not match its projection in successor session ${successor.session.id} (projected ${after.tokens}, actual ${actual.tokens}).`,
      );
    }
    return { successor, outcome };
  });
}

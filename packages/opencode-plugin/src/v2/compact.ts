import type { SessionMessageAssistant } from "@opencode/client";
import { isDeepStrictEqual } from "node:util";
import { COMPACT_NOOP } from "../commands";
import {
  BOUNDARY_METADATA,
  POST_COMPACTION_NOTICE,
} from "../compact/constants";
import { getCompactionCount } from "../compact/session";
import {
  capCodePoints,
  GLOBAL_ANCHOR_CHARACTER_LIMIT,
  globalTurnIndexPrompt,
} from "../policy/generation";
import { keepsHostInjectedText } from "../policy/host-text";
import {
  collectRetentionCandidates,
  decideRetentionWith,
  formatRetentionFeedback,
  HEURISTIC_RETENTION,
  type RetentionPlan,
  type RetentionPolicy,
} from "../policy/retention";
import {
  COMPACT_INSUFFICIENT_SAVINGS,
  hasUsefulSavings,
  withExclusiveSource,
} from "../policy/successor";
import { compactionTurnRange, type PolicyTurn } from "../policy/turns";
import {
  preservedTurnsFeedback,
  statsMessage,
  type PreservedTurns,
} from "../stats/constants";
import { planOmission, type PlannedOmission } from "../storage/omission";
import { isRecord } from "../util";
import {
  trimV2Tool,
  v2ToolCalls,
  v2RetentionTurns,
  v2ToolCallView,
  v2Turns,
} from "./history";
import type { V2Host } from "./host";
import {
  generateV2Summaries,
  type V2GeneratedSummary,
  type V2GenerationTurn,
} from "./generation";
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
import { PluginMessageError } from "../policy/feedback";

const COMMAND = "Magic Compact";

type V2Turn = PolicyTurn<V2Message> & { start: number; end: number };

// A turn already holding a Magic summary is opaque persisted data: it is
// never regenerated, pruned or rewritten.
export function isV2SummaryCarrier(message: V2Message): boolean {
  if (message.type !== "assistant" || !isRecord(message.metadata)) return false;
  const magicCompact = message.metadata["magicCompact"];
  return isRecord(magicCompact) && magicCompact["summary"] === true;
}

function isBoundary(message: V2Message): boolean {
  if (message.type !== "synthetic" || !isRecord(message.metadata)) return false;
  const magicCompact = message.metadata["magicCompact"];
  return isRecord(magicCompact) && magicCompact["boundary"] === true;
}

// Request turns with the index range of every record they own: from the
// turn's first user message up to the next turn's first user message.
function spannedTurns(messages: V2Message[], from: number): V2Turn[] {
  const turns = v2Turns(messages.slice(from));
  return turns.map((turn, index) => {
    const start = messages.indexOf(turn.user[0]!);
    const next = turns[index + 1];
    return {
      ...turn,
      start,
      end: next ? messages.indexOf(next.user[0]!) : messages.length,
    };
  });
}

export type V2CompactionPlan = {
  turns: V2Turn[];
  summarized: V2Turn[];
  next: V2Turn | null;
  preserved: number;
  // Position of the first effective turn among all source turns.
  sourceTurnOffset: number;
};

// Turns after the newest completed native compaction are eligible; earlier
// records are no longer model-visible and are copied unchanged. The latest
// Magic boundary starts the range and `keepTurns` preserves the most recent
// complete turns.
export function planV2Compaction(
  messages: V2Message[],
  keepTurns: number,
): V2CompactionPlan {
  const floor = messages.findLastIndex(
    message => message.type === "compaction" && message.status === "completed",
  );
  const turns = spannedTurns(messages, floor + 1);
  if (turns.length > 0 && turns.at(-1)!.assistants.length === 0) turns.pop();
  const boundary = messages.findLastIndex(isBoundary);
  const boundaryTurnIndex =
    boundary <= floor ? -1 : turns.findIndex(turn => turn.start > boundary);
  const range = compactionTurnRange(
    turns.length,
    boundary > floor && boundaryTurnIndex === -1
      ? turns.length
      : boundaryTurnIndex,
    keepTurns,
  );
  const allTurns = v2Turns(messages);
  const first = turns[0];
  return {
    turns,
    summarized: turns.slice(range.start, range.end),
    next: turns[range.end] ?? null,
    preserved: range.preserved,
    sourceTurnOffset: first
      ? allTurns.findIndex(turn => turn.user[0] === first.user[0])
      : allTurns.length,
  };
}

function firstUserLine(turn: V2Turn): string {
  const user = turn.user[0];
  const text = user?.type === "user" ? user.text : "";
  return text.split(/\r\n|\r|\n/u, 1)[0] ?? "";
}

function noopMessage(preserved: PreservedTurns): string {
  return preserved.preserved === 0
    ? COMPACT_NOOP
    : `${COMPACT_NOOP} ${preservedTurnsFeedback(preserved)}`;
}

// Host records a summarized turn drops: the OpenCode 2 counterpart of the
// injected user text OpenCode 1 drops, under the same shared rule. Shell and
// subagent notifications carry a result and keep their exact bytes; the
// restart notice only asked the model to carry on. A synthetic message
// without metadata is how OpenCode 2 stores a migrated OpenCode 1 synthetic
// part (several parts joined by blank lines), a plan-mode reminder (which the
// plan agent re-adds while it is active) or the "moved to the background"
// notice; it is kept when the whole text or any part of it carries the
// conversation forward. Every other record (skills, user shell runs, location,
// model and agent switches, system notices, unrecognized metadata) carries
// state and stays.
export function isDisposableV2HostRecord(message: V2Message): boolean {
  if (message.type !== "synthetic") return false;
  if (message.metadata === undefined) {
    return ![message.text, ...message.text.split("\n\n")].some(
      keepsHostInjectedText,
    );
  }
  return isDeepStrictEqual(message.metadata, { notice: "restart" });
}

// The summarized turn as it continues in the successor: the summary leads the
// first assistant message, whose metadata carries it; assistant text and
// reasoning are dropped; tool calls stay visible with the tool rules applied,
// except calls retention keeps whole; empty assistant messages and disposable
// host records are dropped. User messages and every other record are
// unchanged.
async function compactTurn(
  span: readonly V2Message[],
  turn: V2Turn,
  summary: V2GeneratedSummary,
  keep: ReadonlySet<string>,
  allocate: (entry: { content: string }) => Promise<string>,
): Promise<Map<string, V2Message | null>> {
  const edits = new Map<string, V2Message | null>();
  for (const message of span) {
    if (isDisposableV2HostRecord(message)) edits.set(message.id, null);
  }
  for (const [index, message] of turn.assistants.entries()) {
    if (message.type !== "assistant") continue;
    const content: SessionMessageAssistant["content"] = [];
    if (index === 0) content.push({ type: "text", text: summary.text });
    for (const call of v2ToolCalls(message)) {
      const replacement = keep.has(call.ref)
        ? null
        : await trimV2Tool(call.tool, allocate);
      content.push(replacement ?? call.tool);
    }
    if (content.length === 0) {
      edits.set(message.id, null);
      continue;
    }
    edits.set(message.id, {
      ...message,
      ...(index === 0
        ? {
            metadata: {
              ...message.metadata,
              magicCompact: {
                summary: true,
                capsule: true,
                agentic: true,
                version: 3,
                ...(summary.mediaManifest
                  ? { mediaManifest: summary.mediaManifest }
                  : {}),
              },
            },
          }
        : {}),
      content,
    } as V2Message);
  }
  return edits;
}

function applyEdits(
  messages: V2Message[],
  edits: Map<string, V2Message | null>,
): V2Message[] {
  return messages.flatMap(message => {
    if (!edits.has(message.id)) return [message];
    const edited = edits.get(message.id);
    return edited ? [edited] : [];
  });
}

// Returns the ready successor, or null for a no-op with its message. The
// source session and its children are only read.
export async function executeV2Compact(
  host: V2Host,
  owned: OwnedSessions,
  sessionID: string,
  keepTurns: number,
  options: { retention: RetentionPolicy; controlTimeoutMs: number },
): Promise<{ successor: V2Successor | null; outcome: string }> {
  const { controlTimeoutMs } = options;
  return withExclusiveSource(COMMAND, sessionID, async () => {
    const source = await readV2SourceAtRest(
      host,
      COMMAND,
      sessionID,
      controlTimeoutMs,
    );
    const plan = planV2Compaction(source.messages, keepTurns);
    const preserved = { requested: keepTurns, preserved: plan.preserved };
    const targets = plan.summarized.filter(
      turn => !turn.assistants.some(isV2SummaryCarrier),
    );
    if (targets.length === 0) {
      return { successor: null, outcome: noopMessage(preserved) };
    }

    // One descriptor serves the retention summary and every generation.
    const model = await resolveV2SummarizerModel(
      host,
      source,
      controlTimeoutMs,
    );
    let retention: RetentionPlan = HEURISTIC_RETENTION;
    if (options.retention.mode === "task-aware") {
      const policy = options.retention;
      const eligible = new Set(
        targets.flatMap(turn =>
          turn.assistants.flatMap(message =>
            v2ToolCalls(message)
              .filter(call => v2ToolCallView(call.tool) !== null)
              .map(call => call.ref),
          ),
        ),
      );
      retention = await decideRetentionWith(
        await collectRetentionCandidates(
          v2RetentionTurns(source.messages),
          eligible,
        ),
        policy,
        () =>
          generateV2SessionSummary(
            host,
            owned,
            source,
            model,
            controlTimeoutMs,
          ),
      );
    }

    const span = (turn: V2Turn) => source.messages.slice(turn.start, turn.end);
    const generationTurns: V2GenerationTurn[] = targets.map(turn => {
      const index = plan.turns.indexOf(turn);
      const previous = plan.turns[index - 1];
      return {
        span: span(turn),
        ...(previous ? { previous: span(previous) } : {}),
      };
    });
    const summaries = await generateV2Summaries(
      host,
      owned,
      source,
      model,
      generationTurns,
      globalTurnIndexPrompt(
        plan.turns.map((turn, index) => ({
          sourceOrdinal: plan.sourceTurnOffset + index + 1,
          anchor: capCodePoints(
            firstUserLine(turn),
            GLOBAL_ANCHOR_CHARACTER_LIMIT,
          ),
        })),
      ),
      { controlTimeoutMs },
    );

    // A turn is replaced only when its model-facing projection, with the
    // images its new manifest covers projected, is strictly smaller.
    const edits = new Map<string, V2Message | null>();
    const omissions: PlannedOmission[] = [];
    const selected: V2Turn[] = [];
    for (const [index, turn] of targets.entries()) {
      const turnOmissions: PlannedOmission[] = [];
      const turnEdits = await compactTurn(
        span(turn),
        turn,
        summaries[index]!,
        retention.keep,
        async entry => {
          const omission = planOmission(sessionID, entry);
          turnOmissions.push(omission);
          return omission.contentID;
        },
      );
      const before = countV2ModelFacingTokens(span(turn)).tokens;
      const after = countV2ModelFacingTokens(
        applyEdits(span(turn), turnEdits),
      ).tokens;
      if (after >= before) continue;
      selected.push(turn);
      omissions.push(...turnOmissions);
      for (const [id, message] of turnEdits) edits.set(id, message);
    }
    if (selected.length === 0) {
      throw new PluginMessageError(COMPACT_INSUFFICIENT_SAVINGS);
    }

    const boundary = {
      id: "magic-compact-boundary",
      type: "synthetic",
      text: POST_COMPACTION_NOTICE,
      metadata: BOUNDARY_METADATA,
      time: { created: Date.now() },
    } as V2Message;
    const compacted = applyEdits(source.messages, edits);
    const nextUser = plan.next?.user[0];
    const at = nextUser ? compacted.indexOf(nextUser) : compacted.length;
    const history = [
      ...compacted.slice(0, at),
      boundary,
      ...compacted.slice(at),
    ];

    // The successor drops instruction notices, so savings are measured
    // against the source without them; the reported and physical counts
    // start from the complete source.
    const withoutNotices = (messages: V2Message[]) =>
      messages.filter(message => !isInstructionNotice(message));
    const before = countV2ModelFacingTokens(source.messages);
    const baseline = countV2ModelFacingTokens(withoutNotices(source.messages));
    const after = countV2ModelFacingTokens(withoutNotices(history));
    const eligible = countV2ModelFacingTokens(
      selected.flatMap(turn => turn.assistants),
    ).tokens;
    if (!hasUsefulSavings(baseline.tokens - after.tokens, eligible)) {
      throw new PluginMessageError(COMPACT_INSUFFICIENT_SAVINGS);
    }
    if (after.projectedMediaTokens < before.projectedMediaTokens) {
      throw new Error("Magic Compact media projection unexpectedly decreased.");
    }

    const compactionCount = getCompactionCount(source.session) + 1;
    const outcome = [
      statsMessage(
        compactionCount,
        before.tokens,
        after.tokens,
        selected.length,
        preserved,
      ),
      `Retention | ${formatRetentionFeedback(retention, [])}`,
    ].join("\n");
    const successor = await createV2Successor(host, {
      command: COMMAND,
      source,
      messages: history,
      operation: "compact",
      compactionCount,
      outcome,
      omissions,
      tokensPruned:
        before.tokens
        + before.projectedMediaTokens
        - (after.tokens + after.projectedMediaTokens),
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
      throw new PluginMessageError(
        `Compaction token projection mismatch in successor session ${successor.session.id}.`,
      );
    }
    return { successor, outcome };
  });
}

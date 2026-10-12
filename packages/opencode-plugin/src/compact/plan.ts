import type { AssistantMessage, Message, Part } from "@opencode-ai/sdk/v2";
import { isDeepStrictEqual } from "node:util";
import { unwrap, type V2Client } from "../api";
import { isRecord } from "../util";

export type MessageWithParts = {
  info: Message;
  parts: Part[];
};

export type Turn = {
  user: MessageWithParts[];
  assistants: MessageWithParts[];
};

export type CompactionPlan = {
  summarizedTurns: Turn[];
  nextTurn: Turn | null;
  // Complete request turns kept whole by N; fewer than N when N exceeds the
  // eligible turns.
  preservedTurnCount: number;
  sourceTurnStartIndex: number;
  effectiveSourceTurnStartIndex: number;
  effectiveSourceTurnEndIndex: number;
  sourceSnapshot: SourceSnapshot;
};

export type CompactionSourceInsertion =
  | { type: "part"; messageID: string; part: Part }
  | { type: "message"; messageID: string; parts: Part[] };

export type SourceSnapshot = Array<{ info: unknown; parts: unknown[] }>;

export class NativeCheckpointChangedError extends Error {
  constructor(cause?: unknown) {
    const detail = cause instanceof Error ? ` ${cause.message}` : "";
    super(`OpenCode source changed during Magic Compact.${detail}`, { cause });
    this.name = "NativeCheckpointChangedError";
  }
}

export class CompactionPruneSourceDriftError extends Error {
  constructor(cause?: unknown) {
    const detail = cause instanceof Error ? ` ${cause.message}` : "";
    super(
      `OpenCode source changed after summary insertion and before pruning.${detail}`,
      { cause },
    );
    this.name = "CompactionPruneSourceDriftError";
  }
}

export class TrimSourceChangedError extends Error {
  constructor() {
    super(
      "OpenCode source changed during Magic Trim; the source session was left unchanged.",
    );
    this.name = "TrimSourceChangedError";
  }
}

export type TrimPlan = {
  trimmedTurns: Turn[];
  messages: MessageWithParts[];
  sourceSnapshot: SourceSnapshot;
};

export async function createCompactionPlan(
  v2: V2Client,
  sessionID: string,
  keepTurns: number,
): Promise<CompactionPlan> {
  const messages = await loadMessages(v2, sessionID);
  const sourceTurns = buildTurns(messages);
  const { summaryIndex } = inspectNativeCheckpoints(messages);
  const suffixStart = summaryIndex === null ? 0 : summaryIndex + 1;
  const turns = buildTurns(messages.slice(suffixStart));

  const boundaryTurnIndex = turns.findLastIndex(turn =>
    turn.user.some(msg => msg.parts.some(isBoundaryPart)),
  );
  const compactionStartIndex = boundaryTurnIndex === -1 ? 0 : boundaryTurnIndex;

  removeTrailingAssistantlessTurn(turns);

  const firstEffectiveTurn = turns[0];
  const effectiveSourceTurnStartIndex = firstEffectiveTurn
    ? sourceTurns.findIndex(
        turn => turn.user[0]?.info.id === firstEffectiveTurn.user[0]?.info.id,
      )
    : sourceTurns.length;
  if (effectiveSourceTurnStartIndex < 0) {
    throw new Error("Compaction effective turn is missing from source order.");
  }
  const effectiveSourceTurnEndIndex =
    effectiveSourceTurnStartIndex + turns.length;

  const compactionEndIndex =
    keepTurns <= 0
      ? turns.length
      : Math.max(compactionStartIndex, turns.length - keepTurns);

  const summarizedTurns = turns.slice(compactionStartIndex, compactionEndIndex);

  const nextTurn = turns[compactionEndIndex] ?? null;
  const sourceTurnStartIndex =
    effectiveSourceTurnStartIndex + compactionStartIndex;

  return {
    summarizedTurns,
    nextTurn,
    preservedTurnCount: turns.length - compactionEndIndex,
    sourceTurnStartIndex,
    effectiveSourceTurnStartIndex,
    effectiveSourceTurnEndIndex,
    sourceSnapshot: captureSourceSnapshot(messages),
  };
}

export async function createTrimPlan(
  v2: V2Client,
  sessionID: string,
  keepTurns: number,
): Promise<TrimPlan> {
  const messages = await loadMessages(v2, sessionID);
  const turns = buildTurns(messages);
  removeTrailingAssistantlessTurn(turns);
  const trimEndIndex = Math.max(0, turns.length - keepTurns);

  return {
    trimmedTurns: turns.slice(0, trimEndIndex),
    messages,
    sourceSnapshot: snapshotSourceMessages(messages),
  };
}

// Trim spans the whole session and never validates native checkpoints, so
// its drift guard compares the same normalized fields without inspecting them.
export function assertTrimSourceUnchanged(
  messages: MessageWithParts[],
  plan: TrimPlan,
): void {
  if (
    !isDeepStrictEqual(snapshotSourceMessages(messages), plan.sourceSnapshot)
  ) {
    throw new TrimSourceChangedError();
  }
}

async function loadMessages(
  v2: V2Client,
  sessionID: string,
): Promise<MessageWithParts[]> {
  return unwrap(
    await v2.session.messages({
      sessionID,
    }),
  );
}

export function buildTurns(messages: MessageWithParts[]): Turn[] {
  const turns: Turn[] = [];
  let currentTurn: Turn | null = null;

  for (const message of messages) {
    if (message.info.role === "user") {
      if (currentTurn && currentTurn.assistants.length > 0) {
        currentTurn = null;
      }
      if (!currentTurn) {
        currentTurn = { user: [], assistants: [] };
        turns.push(currentTurn);
      }
      currentTurn.user.push(message);
      continue;
    }

    if (message.info.role === "assistant" && currentTurn) {
      currentTurn.assistants.push(message);
    }
  }

  return turns;
}

export function assertCompactionSourceUnchanged(
  messages: MessageWithParts[],
  plan: CompactionPlan,
): void {
  const current = captureSourceSnapshot(messages);
  if (!isDeepStrictEqual(current, plan.sourceSnapshot)) {
    throw new NativeCheckpointChangedError();
  }
}

export function assertForkMessageShape(
  messages: MessageWithParts[],
  plan: { sourceSnapshot: SourceSnapshot },
): void {
  if (messages.length !== plan.sourceSnapshot.length) {
    throw new Error("Compaction generation fork history shape mismatch.");
  }
  for (const [index, message] of messages.entries()) {
    const source = plan.sourceSnapshot[index];
    if (
      !source
      || !isRecord(source.info)
      || source.info["role"] !== message.info.role
      || source.parts.length !== message.parts.length
      || source.parts.some(
        (part, partIndex) =>
          !isRecord(part) || part["type"] !== message.parts[partIndex]?.type,
      )
    ) {
      throw new Error("Compaction generation fork history shape mismatch.");
    }
  }
}

export function assertCompactionSourceAfterInsertions(
  messages: MessageWithParts[],
  plan: CompactionPlan,
  insertions: CompactionSourceInsertion[],
): void {
  const remaining = structuredClone(messages);
  for (const insertion of insertions) {
    if (insertion.type === "message") {
      const matches = remaining
        .map((message, index) => ({ message, index }))
        .filter(item => item.message.info.id === insertion.messageID);
      if (
        matches.length !== 1
        || matches[0]!.message.info.role !== "user"
        || !isDeepStrictEqual(
          matches[0]!.message.parts.map(sourcePart),
          insertion.parts.map(sourcePart),
        )
      ) {
        throw new CompactionPruneSourceDriftError();
      }
      remaining.splice(matches[0]!.index, 1);
      continue;
    }

    const messageMatches = remaining.filter(
      message => message.info.id === insertion.messageID,
    );
    if (messageMatches.length !== 1) {
      throw new CompactionPruneSourceDriftError();
    }
    const message = messageMatches[0]!;
    const partMatches = message.parts
      .map((part, index) => ({ part, index }))
      .filter(item => item.part.id === insertion.part.id);
    if (
      partMatches.length !== 1
      || !isDeepStrictEqual(
        sourcePart(partMatches[0]!.part),
        sourcePart(insertion.part),
      )
    ) {
      throw new CompactionPruneSourceDriftError();
    }
    message.parts.splice(partMatches[0]!.index, 1);
  }
  try {
    assertCompactionSourceUnchanged(remaining, plan);
  } catch (error) {
    throw new CompactionPruneSourceDriftError(error);
  }
}

function captureSourceSnapshot(messages: MessageWithParts[]): SourceSnapshot {
  try {
    inspectNativeCheckpoints(messages);
  } catch (error) {
    throw new NativeCheckpointChangedError(error);
  }
  return snapshotSourceMessages(messages);
}

export function snapshotSourceMessages(
  messages: MessageWithParts[],
): SourceSnapshot {
  return messages.map(sourceMessage);
}

function sourceMessage(message: MessageWithParts): SourceSnapshot[number] {
  return {
    info: sourceMessageInfo(message.info),
    parts: message.parts.map(sourcePart),
  };
}

function sourceMessageInfo(info: Message): unknown {
  const snapshot = structuredClone(info) as unknown as Record<string, unknown>;
  delete snapshot["time"];
  delete snapshot["tokens"];
  delete snapshot["cost"];
  return snapshot;
}

function sourcePart(part: Part): unknown {
  const snapshot = structuredClone(part) as unknown as Record<string, unknown>;
  delete snapshot["time"];
  delete snapshot["tokens"];
  delete snapshot["cost"];
  if (part.type === "tool") {
    const state = snapshot["state"] as Record<string, unknown>;
    const time = state["time"];
    if (isRecord(time) && time["compacted"] !== undefined) {
      state["time"] = { compacted: time["compacted"] };
    } else {
      delete state["time"];
    }
  }
  return snapshot;
}

// The history OpenCode's prompt loop sends to the model, mirroring
// `filterCompacted` in packages/opencode/src/session/message-v2.ts at
// 830d5eb535 (1.18.30). Native filtering scans newest-first, then reorders
// around the last tail-bearing marker even when its summary failed or is
// unfinished. This read-only view must not use the stricter writer inspector.
export function nativeModelVisibleMessages(
  messages: MessageWithParts[],
): MessageWithParts[] {
  const result: MessageWithParts[] = [];
  const completed = new Set<string>();
  let retain: string | null = null;
  for (let index = messages.length - 1; index >= 0; index--) {
    const message = messages[index]!;
    result.push(message);
    if (retain) {
      if (message.info.id === retain) break;
      continue;
    }
    if (message.info.role === "user" && completed.has(message.info.id)) {
      const part = message.parts.find(item => item.type === "compaction");
      if (!part) continue;
      retain = readTailStartID(part);
      if (!retain || message.info.id === retain) break;
      continue;
    }
    if (
      message.info.role === "assistant"
      && message.info.summary
      && message.info.finish
      && !message.info.error
    ) {
      completed.add(message.info.parentID);
    }
  }
  if (retain && result.at(-1)?.info.id !== retain) {
    throw new Error("Native compaction retained tail is missing from history.");
  }
  result.reverse();
  const hasTail = (item: Part) =>
    item.type === "compaction"
    && "tail_start_id" in item
    && item.tail_start_id !== undefined;
  const markerIndex = result.findLastIndex(
    message => message.info.role === "user" && message.parts.some(hasTail),
  );
  const marker = result[markerIndex];
  const part = marker?.parts.find(hasTail);
  const summaryIndex = marker
    ? result.findIndex(
        (message, index) =>
          index > markerIndex
          && message.info.role === "assistant"
          && message.info.summary
          && message.info.parentID === marker.info.id,
      )
    : -1;
  const tailStartID = part ? readTailStartID(part) : null;
  const tailIndex = tailStartID
    ? result.findIndex(message => message.info.id === tailStartID)
    : -1;
  if (part && (tailIndex < 0 || tailIndex >= markerIndex)) {
    throw new Error(
      "Native compaction tail does not resolve before its marker.",
    );
  }
  if (tailIndex >= 0 && tailIndex < markerIndex && summaryIndex > markerIndex) {
    return [
      ...result.slice(markerIndex, summaryIndex + 1),
      ...result.slice(tailIndex, markerIndex),
      ...result.slice(summaryIndex + 1),
    ];
  }
  return result;
}

function inspectNativeCheckpoints(messages: MessageWithParts[]): {
  summaryIndex: number | null;
} {
  const messageIndex = new Map(
    messages.map((message, index) => [message.info.id, index]),
  );
  const markers = messages.flatMap((message, index) => {
    const parts = message.parts.filter(part => part.type === "compaction");
    if (parts.length === 0) return [];
    if (message.info.role !== "user") {
      throw new Error(
        `Native compaction part on ${message.info.id} must belong to a user message.`,
      );
    }
    if (parts.length > 1) {
      throw new Error(
        `Native compaction marker ${message.info.id} has multiple compaction parts.`,
      );
    }
    return [{ message, index, part: parts[0]! }];
  });
  const summaries: Array<{
    message: MessageWithParts & { info: AssistantMessage };
    index: number;
  }> = [];
  for (const [index, message] of messages.entries()) {
    if (message.info.role === "assistant" && message.info.summary === true) {
      summaries.push({
        message: { ...message, info: message.info },
        index,
      });
    }
  }
  const markerIDs = new Set(markers.map(marker => marker.message.info.id));

  for (const summary of summaries) {
    const parentID = summary.message.info.parentID;
    const markerIndex = parentID ? messageIndex.get(parentID) : undefined;
    if (
      !parentID
      || !markerIDs.has(parentID)
      || markerIndex === undefined
      || markerIndex >= summary.index
    ) {
      throw new Error(
        `Found orphan native compaction summary ${summary.message.info.id}.`,
      );
    }
  }

  const childrenByMarkerID = new Map(
    markers.map(marker => [
      marker.message.info.id,
      summaries.filter(
        summary => summary.message.info.parentID === marker.message.info.id,
      ),
    ]),
  );
  const candidates = markers.flatMap(marker => {
    const children = childrenByMarkerID.get(marker.message.info.id)!;
    if (children.length !== 1) return [];
    const summary = children[0]!;
    if (!summary.message.info.finish || summary.message.info.error) return [];

    const tailStartID = readTailStartID(marker.part);
    if (tailStartID) {
      const tailIndex = messageIndex.get(tailStartID);
      if (tailIndex === undefined || tailIndex >= marker.index) {
        throw new Error(
          `Native compaction tail_start_id ${tailStartID} does not resolve before its marker.`,
        );
      }
    }
    return [{ marker, summary }];
  });

  for (const candidate of candidates) {
    const containsForeignMarker = markers.some(
      marker =>
        marker !== candidate.marker
        && marker.index > candidate.marker.index
        && marker.index < candidate.summary.index,
    );
    const containsForeignSummary = summaries.some(
      summary =>
        summary !== candidate.summary
        && summary.index > candidate.marker.index
        && summary.index < candidate.summary.index,
    );
    if (containsForeignMarker || containsForeignSummary) {
      throw new Error(
        `Native compaction checkpoint ${candidate.marker.message.info.id} overlaps another checkpoint interval.`,
      );
    }
  }

  const selected = candidates
    .toSorted((left, right) => left.summary.index - right.summary.index)
    .at(-1);

  for (const marker of markers) {
    const completed = candidates.some(
      candidate => candidate.marker.message.info.id === marker.message.info.id,
    );
    const children = childrenByMarkerID.get(marker.message.info.id)!;
    const whollyBeforeSelected =
      selected
      && marker.index < selected.marker.index
      && children.every(summary => summary.index < selected.marker.index);
    if (completed || whollyBeforeSelected) continue;
    if (
      children.length > 1
      && children.some(
        summary => summary.message.info.finish && !summary.message.info.error,
      )
    ) {
      const detail =
        children.filter(
          summary => summary.message.info.finish && !summary.message.info.error,
        ).length > 1
          ? ", including multiple completed summaries"
          : "";
      throw new Error(
        `Native compaction marker ${marker.message.info.id} has ambiguous summary children${detail}.`,
      );
    }
    throw new Error(
      `Native compaction marker ${marker.message.info.id} is incomplete or errored.`,
    );
  }

  return {
    summaryIndex: selected ? selected.summary.index : null,
  };
}

function readTailStartID(part: Part): string | null {
  const value = (part as unknown as Record<string, unknown>)["tail_start_id"];
  if (value === undefined) return null;
  if (typeof value !== "string" || value.length === 0) {
    throw new Error("Native compaction tail_start_id must be a message ID.");
  }
  return value;
}

function removeTrailingAssistantlessTurn(turns: Turn[]): void {
  const lastTurn = turns.at(-1);
  if (lastTurn && lastTurn.assistants.length === 0) {
    // Last turn may consist of noReply user messages, and should not count as a real "turn"
    turns.pop();
  }
}

function isBoundaryPart(part: Part): boolean {
  if (part.type !== "text") {
    return false;
  }

  const metadata = part.metadata;
  if (!isRecord(metadata)) {
    return false;
  }

  const magicCompact = metadata["magicCompact"];
  return isRecord(magicCompact) && magicCompact["boundary"] === true;
}

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
  sourceSnapshot: SourceSnapshot;
};

type SourceSnapshot = Array<{ info: unknown; parts: unknown[] }>;

export class NativeCheckpointChangedError extends Error {
  constructor(cause?: unknown) {
    const detail = cause instanceof Error ? ` ${cause.message}` : "";
    super(`OpenCode source changed during Magic Compact.${detail}`, { cause });
    this.name = "NativeCheckpointChangedError";
  }
}

export type TrimPlan = {
  trimmedTurns: Turn[];
};

export async function createCompactionPlan(
  v2: V2Client,
  sessionID: string,
  keepTurns: number,
): Promise<CompactionPlan> {
  const messages = await loadMessages(v2, sessionID);
  const native = inspectNativeCheckpoints(messages);
  const suffixStart =
    native.summaryIndex === null ? 0 : native.summaryIndex + 1;
  const turns = buildTurns(messages.slice(suffixStart));

  const boundaryTurnIndex = turns.findLastIndex(turn =>
    turn.user.some(msg => msg.parts.some(isBoundaryPart)),
  );
  const compactionStartIndex = boundaryTurnIndex === -1 ? 0 : boundaryTurnIndex;

  removeTrailingAssistantlessTurn(turns);

  const compactionEndIndex =
    keepTurns <= 0
      ? turns.length
      : Math.max(compactionStartIndex, turns.length - keepTurns);

  const summarizedTurns = turns.slice(compactionStartIndex, compactionEndIndex);

  const nextTurn = turns[compactionEndIndex] ?? null;

  return {
    summarizedTurns,
    nextTurn,
    sourceSnapshot: captureSourceSnapshot(messages),
  };
}

export async function createTrimPlan(
  v2: V2Client,
  sessionID: string,
  keepTurns: number,
): Promise<TrimPlan> {
  const turns = buildTurns(await loadMessages(v2, sessionID));
  removeTrailingAssistantlessTurn(turns);
  const trimEndIndex = Math.max(0, turns.length - keepTurns);

  return {
    trimmedTurns: turns.slice(0, trimEndIndex),
  };
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

function buildTurns(messages: MessageWithParts[]): Turn[] {
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

function captureSourceSnapshot(messages: MessageWithParts[]): SourceSnapshot {
  try {
    inspectNativeCheckpoints(messages);
  } catch (error) {
    throw new NativeCheckpointChangedError(error);
  }
  return messages.map(message => ({
    info: sourceMessageInfo(message.info),
    parts: message.parts.map(sourcePart),
  }));
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
    delete state["time"];
  }
  return snapshot;
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
    summaryIndex: selected?.summary.index ?? null,
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

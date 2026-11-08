// The OpenCode 1 parts that carry a Magic summary capsule and the
// compaction boundary.
import type { Part, TextPart } from "@opencode-ai/sdk/v2";
import { isSummaryMetadata, summaryCapsuleMetadata } from "../policy/protocol";
import type { MediaManifest } from "./media";
import type { Turn } from "./plan";

// `prt_-` keeps Magic-owned parts before OpenCode's generated `prt_` IDs in lexical storage order.

export function summaryEnvelopePartID(messageID: string): string {
  return `prt_-magic_capsule_v3_${messageID}`;
}

export function summaryEnvelopeTextPart(
  messageID: string,
  sessionID: string,
  text: string,
  mediaManifest?: MediaManifest,
): TextPart {
  return {
    id: summaryEnvelopePartID(messageID),
    sessionID,
    messageID,
    type: "text",
    text,
    metadata: summaryCapsuleMetadata(mediaManifest),
  } satisfies TextPart;
}

export function boundaryPartID(messageID: string): string {
  return `prt_-magic_boundary_${messageID}`;
}

export function isPersistedSummaryPart(part: Part): boolean {
  return part.type === "text" && isSummaryMetadata(part.metadata);
}

export function turnHasPersistedSummary(turn: Turn): boolean {
  return turn.assistants.some(message =>
    message.parts.some(isPersistedSummaryPart),
  );
}

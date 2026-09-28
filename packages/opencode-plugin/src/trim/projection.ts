import type { ToolPart } from "@opencode-ai/sdk/v2";
import type { MessageWithParts } from "../compact/plan";
import { bindToolSource, type ToolOverlaySnapshot } from "./plan";

export function projectToolReplacements(
  messages: MessageWithParts[],
  snapshot: ToolOverlaySnapshot,
): MessageWithParts[] {
  const projected = structuredClone(messages);
  const found = new Set<number>();
  for (const [index, replacement] of snapshot.replacements.entries()) {
    const { source, part } = replacement;
    if (
      part.type !== "tool"
      || part.sessionID !== source.sessionID
      || part.messageID !== source.messageID
      || part.id !== source.partID
      || part.callID !== source.callID
      || part.tool !== source.tool
    ) {
      throw new Error(
        "Tool overlay replacement identity does not match its source.",
      );
    }
    for (const message of projected) {
      if (message.info.id !== source.messageID) continue;
      if (message.info.role !== "assistant") {
        throw new Error(
          "Tool overlay source must belong to an assistant message.",
        );
      }
      for (const [partIndex, current] of message.parts.entries()) {
        if (current.id !== source.partID) continue;
        if (
          current.type !== "tool"
          || current.sessionID !== source.sessionID
          || bindToolSource(current as ToolPart).fingerprint
            !== source.fingerprint
          || found.has(index)
        ) {
          throw new Error("Tool overlay source changed.");
        }
        message.parts[partIndex] = structuredClone(part);
        found.add(index);
      }
    }
    if (!found.has(index)) throw new Error("Tool overlay source is missing.");
  }
  return projected;
}

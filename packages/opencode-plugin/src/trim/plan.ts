import { createHash } from "node:crypto";
import { serialize } from "node:v8";
import type { ToolPart } from "@opencode-ai/sdk/v2";
import type { PlannedOmission } from "../storage/omission";
import { isRecord } from "../util";

export type ToolSourceBinding = Readonly<{
  sessionID: string;
  messageID: string;
  partID: string;
  callID: string;
  tool: string;
  fingerprint: string;
}>;

export type ToolReplacement = Readonly<{
  source: ToolSourceBinding;
  part: ToolPart;
  omissions: readonly PlannedOmission[];
}>;

export type ToolOverlaySnapshot = Readonly<{
  replacements: readonly ToolReplacement[];
}>;

export function bindToolSource(part: ToolPart): ToolSourceBinding {
  const source = structuredClone(part) as ToolPart;
  const record = source as unknown as Record<string, unknown>;
  delete record["time"];
  delete record["tokens"];
  delete record["cost"];
  const state = source.state as unknown as Record<string, unknown>;
  const time = state["time"];
  if (isRecord(time) && time["compacted"] !== undefined) {
    state["time"] = { compacted: time["compacted"] };
  } else {
    delete state["time"];
  }
  return {
    sessionID: part.sessionID,
    messageID: part.messageID,
    partID: part.id,
    callID: part.callID,
    tool: part.tool,
    fingerprint: createHash("sha256").update(serialize(source)).digest("hex"),
  };
}

export function createToolOverlaySnapshot(
  replacements: readonly ToolReplacement[],
): ToolOverlaySnapshot {
  const copied = structuredClone(replacements);
  const keys = new Set<string>();
  for (const replacement of copied) {
    const key = `${replacement.source.sessionID}\0${replacement.source.messageID}\0${replacement.source.partID}`;
    if (keys.has(key)) throw new Error("Duplicate tool overlay source.");
    keys.add(key);
  }
  return freeze({ replacements: copied });
}

function freeze<T>(value: T): T {
  if (value && typeof value === "object") {
    for (const child of Object.values(value)) freeze(child);
    Object.freeze(value);
  }
  return value;
}

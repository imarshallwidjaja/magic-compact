import type { SessionMessageAssistantTool } from "@opencode/client";
import { createHash } from "node:crypto";
import type { RetentionTurn } from "../policy/retention";
import {
  childSessionID,
  planToolTrim,
  type OmissionAllocator,
  type ToolCallView,
  type ToolRuleCategory,
} from "../policy/tool-rules";
import { groupTurns, type PolicyTurn } from "../policy/turns";
import { countGptTokens } from "../stats/tokenize";
import { isRecord } from "../util";
import type { V2MediaProjection } from "./media";
import type { V2Message } from "./session";

export type V2Tool = SessionMessageAssistantTool;

// OpenCode 2 tool names for the shared rule categories. OpenCode 2 itself
// maps the OpenCode 1 names bash, task and apply_patch onto shell, subagent
// and patch. Code mode's `execute` aggregates several calls in one record and
// stays a generic call; no intermediate leaf call is assumed to exist.
const V2_TOOL_CATEGORIES: Readonly<Record<string, ToolRuleCategory>> = {
  write: "write",
  edit: "edit",
  patch: "patch",
  shell: "shell",
  subagent: "subagent",
  read: "read",
  question: "question",
  skill: "skill",
};

// A tool call is identified by its message, its position in that message's
// content, and its call ID; call IDs alone are not unique across a session.
export function v2ToolRef(
  messageID: string,
  index: number,
  tool: V2Tool,
): string {
  return `${messageID}:${index}:${tool.id}`;
}

export function v2ToolCalls(
  message: V2Message,
): Array<{ ref: string; index: number; tool: V2Tool }> {
  if (message.type !== "assistant") return [];
  return message.content.flatMap((item, index) =>
    item.type === "tool"
      ? [{ ref: v2ToolRef(message.id, index, item), index, tool: item }]
      : [],
  );
}

export function v2Turns(
  messages: readonly V2Message[],
): PolicyTurn<V2Message>[] {
  return groupTurns(messages, message =>
    message.type === "user" || message.type === "assistant"
      ? message.type
      : "other",
  );
}

// Every request turn with its ordinary user text and the completed calls the
// rules can judge. OpenCode 2 keeps host notices in their own message types,
// so a user message's text is always ordinary.
export function v2RetentionTurns(
  messages: readonly V2Message[],
): RetentionTurn[] {
  return v2Turns(messages).map(turn => ({
    request: turn.user
      .flatMap(message => (message.type === "user" ? [message.text] : []))
      .join("\n")
      .trim(),
    calls: turn.assistants.flatMap(message =>
      v2ToolCalls(message).flatMap(call => {
        const view = v2ToolCallView(call.tool);
        return view ? [{ ref: call.ref, tool: call.tool.name, view }] : [];
      }),
    ),
  }));
}

// The text output the rules judge: completed text content in order. File
// content is not text output; it stays in place whatever the rules decide.
export function v2ToolCallView(tool: V2Tool): ToolCallView | null {
  if (tool.state.status !== "completed") return null;
  const magicCompact = tool.state.metadata?.["magicCompact"];
  const category = Object.hasOwn(V2_TOOL_CATEGORIES, tool.name)
    ? V2_TOOL_CATEGORIES[tool.name]!
    : "other";
  return {
    category,
    input: tool.state.input,
    output: tool.state.content
      .flatMap(item => (item.type === "text" ? [item.text] : []))
      .join("\n\n"),
    trimmed: isRecord(magicCompact) && magicCompact["trimmed"] === true,
    childSessionID: childSessionID(
      category,
      tool.state.metadata?.["sessionID"],
    ),
  };
}

// Applies the rules to a clone of a completed call. A replaced output takes
// the place of the first text item and drops the other text items; file items
// keep their positions.
export async function trimV2Tool(
  tool: V2Tool,
  allocate: OmissionAllocator,
): Promise<V2Tool | null> {
  const view = v2ToolCallView(tool);
  if (view === null || tool.state.status !== "completed") return null;
  const plan = await planToolTrim(view, allocate);
  if (plan === null) return null;
  const state = structuredClone(tool.state);
  type Content = (typeof state.content)[number];
  let content: Content[] = [...state.content];
  if (plan.output !== null) {
    const first = content.findIndex(item => item.type === "text");
    const replacement = { type: "text" as const, text: plan.output };
    const files = content.filter(item => item.type !== "text");
    content =
      first === -1
        ? [replacement, ...files]
        : [
            ...content.slice(0, first).filter(item => item.type !== "text"),
            replacement,
            ...content.slice(first + 1).filter(item => item.type !== "text"),
          ];
  }
  const magicCompact = state.metadata?.["magicCompact"];
  return {
    ...structuredClone(tool),
    state: {
      ...state,
      input: plan.input as typeof state.input,
      content: content as typeof state.content,
      metadata: {
        ...state.metadata,
        magicCompact: {
          ...(isRecord(magicCompact) ? magicCompact : {}),
          trimmed: true,
        },
      },
    },
  };
}

// The records OpenCode 2.0.28 sends for a session, for budgeting a request
// over a whole copied history (`session/history.ts`): everything from the
// newest completed compaction the model can replay. A plain compaction
// summary replays for every model. A provider checkpoint (one carrying
// `providerContext`) replays only for a compatible provider route, which the
// plugin cannot see, so the estimate starts at the newest plain compaction and
// counts every later record, checkpoints included. That can only overcount.
export function v2ModelVisibleHistory(
  messages: readonly V2Message[],
): V2Message[] {
  const floor = messages.findLastIndex(
    message =>
      message.type === "compaction"
      && message.status === "completed"
      && message.providerContext === undefined,
  );
  return messages.slice(Math.max(floor, 0));
}

// A file payload (base64 data or a URI) has no natural breaks, and the local
// tokenizer is quadratic on a long uniform run, so a payload is counted in
// fixed segments, and a large payload once per process by its hash.
const PAYLOAD_SEGMENT_CHARS = 1_024;
const PAYLOAD_COUNT_ENTRIES = 32;
const payloadCounts = new Map<string, number>();

export function countV2PayloadTokens(payload: string): number {
  if (payload.length <= PAYLOAD_SEGMENT_CHARS) return countGptTokens(payload);
  const key = createHash("sha256").update(payload).digest("hex");
  const cached = payloadCounts.get(key);
  if (cached !== undefined) return cached;
  let count = 0;
  for (let at = 0; at < payload.length; at += PAYLOAD_SEGMENT_CHARS) {
    count += countGptTokens(payload.slice(at, at + PAYLOAD_SEGMENT_CHARS));
  }
  if (payloadCounts.size >= PAYLOAD_COUNT_ENTRIES) {
    payloadCounts.delete(payloadCounts.keys().next().value!);
  }
  payloadCounts.set(key, count);
  return count;
}

// Canonical stored-history estimate: every model-facing string of every
// record, tokenized locally one content item at a time, file payloads by
// `countV2PayloadTokens`. It is a stable before/after measure for reporting
// and verification, not provider usage. With a media projection, a projected
// image counts as its placeholder.
export function countV2ContentTokens(
  messages: readonly V2Message[],
  projection?: V2MediaProjection,
): number {
  let total = 0;
  const add = (...texts: Array<string | undefined>) => {
    const text = texts.filter(value => value !== undefined).join("\n\n");
    if (text) total += countGptTokens(text);
  };
  for (const message of messages) {
    switch (message.type) {
      case "user": {
        add(message.text);
        const covered = projection?.get(message.id) ?? [];
        for (const [index, file] of (message.files ?? []).entries()) {
          const placeholder = covered.find(
            image => image.index === index,
          )?.placeholder;
          if (placeholder !== undefined) add(placeholder);
          else if (file.data) total += countV2PayloadTokens(file.data);
        }
        break;
      }
      case "synthetic":
      case "system":
      case "skill":
        add(message.text);
        break;
      case "shell":
        add(message.command, message.output?.output);
        break;
      case "compaction":
        if (message.status === "completed")
          add(message.summary, message.recent);
        break;
      case "assistant":
        for (const item of message.content) {
          if (item.type !== "tool") {
            add(item.text);
            continue;
          }
          const state = item.state;
          add(
            typeof state.input === "string"
              ? state.input
              : JSON.stringify(state.input),
          );
          if (state.status === "completed" || state.status === "error") {
            for (const content of state.content ?? []) {
              if (content.type === "text") add(content.text);
              else if (content.uri) total += countV2PayloadTokens(content.uri);
            }
          }
          if (state.status === "error") add(state.error.message);
        }
        break;
      default:
        break;
    }
  }
  return total;
}

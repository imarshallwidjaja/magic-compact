// OpenCode 1 reasoning records for the shared reasoning policy: protected
// payloads and their origin models, and the replay estimate for reasoning a
// request's model would be sent again.
import type { Part, ReasoningPart } from "@opencode-ai/sdk/v2";
import {
  NO_REPLAY,
  protectedPayload,
  reasoningItemID,
  sameModel,
  type ModelIdentity,
  type ReasoningContextMode,
  type ReasoningUnit,
  type ReplayEstimate,
} from "../policy/reasoning";
import { countGptTokens } from "../stats/tokenize";
import type { MessageWithParts } from "./plan";

function partPayload(part: ReasoningPart): string | undefined {
  return protectedPayload(part.metadata);
}

function itemID(part: ReasoningPart): string | undefined {
  return reasoningItemID(part.metadata);
}

function isReasoning(part: Part): part is ReasoningPart {
  return part.type === "reasoning";
}

function origin(message: MessageWithParts): ModelIdentity | undefined {
  const info = message.info;
  if (info.role !== "assistant") return undefined;
  return info.providerID && info.modelID
    ? { providerID: info.providerID, modelID: info.modelID }
    : undefined;
}

// Every reasoning item of the messages, counted once per item: parts of one
// message that share an item ID are one unit.
export function reasoningUnits(
  messages: readonly MessageWithParts[],
): ReasoningUnit[] {
  const units: ReasoningUnit[] = [];
  for (const message of messages) {
    if (message.info.role !== "assistant") continue;
    const items = new Map<string, boolean>();
    for (const part of message.parts) {
      if (!isReasoning(part)) continue;
      const key = itemID(part) ?? `part:${part.id}`;
      items.set(
        key,
        (items.get(key) ?? false) || partPayload(part) !== undefined,
      );
    }
    for (const isProtected of items.values()) {
      units.push({ origin: origin(message), protected: isProtected });
    }
  }
  return units;
}

// Which historical reasoning a request's model is sent, and at what cost: the
// model the request runs on and its reasoning context, `unknown` unless that
// model's behavior is verified.
export type ReplayContext = {
  primary: ModelIdentity;
  mode: ReasoningContextMode;
};

type Step = {
  messageID: string;
  origin: ModelIdentity | undefined;
  // Whether any reasoning part of the step carries a protected payload.
  protected: boolean;
  // The whole intact unit's provider counter, or undefined when the step
  // cannot be attributed.
  reasoningTokens: number | undefined;
  visibleTokens: number;
};

function steps(message: MessageWithParts): {
  reasoning: ReasoningPart[];
  finish: Extract<Part, { type: "step-finish" }> | undefined;
}[] {
  const result: ReturnType<typeof steps> = [];
  let reasoning: ReasoningPart[] = [];
  for (const part of message.parts) {
    if (isReasoning(part)) reasoning.push(part);
    if (part.type === "step-finish") {
      if (reasoning.length > 0) result.push({ reasoning, finish: part });
      reasoning = [];
    }
  }
  if (reasoning.length > 0) result.push({ reasoning, finish: undefined });
  return result;
}

// The replay attribution of one history, fixed once so every sum over it uses
// the same basis: removing some of its messages later never turns a unit
// that repeats in them into an attributed one.
//
// A unit is counted once, at the step whose provider counters produced it:
// OpenCode 1 records each step's usage on its `step-finish` part, while the
// message's own counters may hold only the last step. The estimate replaces
// the unit's visible reasoning-summary text, which is not an independent
// addition. A unit is attributed only when the step finished, its message has
// a completion time and no error, every reasoning part of the step carries
// one item's identical payload, that item and payload occur in no other step
// of the history, and the step's reasoning counter is a positive integer.
// Usage copied onto a message whose reasoning parts were deleted is never
// charged, because only stored reasoning parts are read.
export type ReplayIndex = {
  // The estimate for a request on `context.primary`, over the messages in
  // `within` when given, else over the whole history.
  estimate(
    context: ReplayContext | undefined,
    within?: ReadonlySet<string>,
  ): ReplayEstimate;
};

export function indexReplay(
  messages: readonly MessageWithParts[],
): ReplayIndex {
  const raw = messages.flatMap(message =>
    message.info.role === "assistant"
      ? steps(message).map(step => ({ message, ...step }))
      : [],
  );
  const keys = (reasoning: ReasoningPart[]) =>
    new Set(
      reasoning.flatMap(part => {
        const id = itemID(part);
        const payload = partPayload(part);
        return [
          ...(id === undefined ? [] : [`id:${id}`]),
          ...(payload === undefined ? [] : [`payload:${payload}`]),
        ];
      }),
    );
  const occurrences = new Map<string, number>();
  for (const step of raw) {
    for (const key of keys(step.reasoning)) {
      occurrences.set(key, (occurrences.get(key) ?? 0) + 1);
    }
  }
  const indexed: Step[] = raw.map(({ message, reasoning, finish }) => {
    const payloads = reasoning.map(partPayload);
    const isProtected = payloads.some(payload => payload !== undefined);
    const ids = new Set(reasoning.map(itemID));
    const info = message.info;
    const counter = isProtected ? finish?.tokens.reasoning : undefined;
    const attributable =
      isProtected
      && info.role === "assistant"
      && info.error === undefined
      && info.time.completed !== undefined
      && finish !== undefined
      && payloads.every(payload => payload !== undefined)
      && new Set(payloads).size === 1
      && ids.size === 1
      && !ids.has(undefined)
      && [...keys(reasoning)].every(key => occurrences.get(key) === 1)
      && Number.isSafeInteger(counter)
      && counter! > 0;
    return {
      messageID: info.id,
      origin: origin(message),
      protected: isProtected,
      reasoningTokens: attributable ? counter : undefined,
      visibleTokens: reasoning.reduce(
        (sum, part) => sum + countGptTokens(part.text),
        0,
      ),
    };
  });
  return {
    estimate(context, within) {
      const estimate = { ...NO_REPLAY };
      for (const step of indexed) {
        if (within && !within.has(step.messageID)) continue;
        if (!step.protected) continue;
        // The host replays a protected payload only to the model that wrote
        // it; another model is sent its text alone, which is already counted.
        if (
          step.origin
          && context
          && !sameModel(step.origin, context.primary)
        ) {
          continue;
        }
        if (
          !step.origin
          || !context
          || context.mode === "unknown"
          || step.reasoningTokens === undefined
        ) {
          estimate.unknown += 1;
          continue;
        }
        estimate.units += 1;
        estimate.visibleTokens += step.visibleTokens;
        if (context.mode === "all_turns") {
          estimate.tokens += step.reasoningTokens;
        }
      }
      return estimate;
    },
  };
}

// The estimate over a whole history, on that history's own attribution.
export function estimateReplay(
  messages: readonly MessageWithParts[],
  context: ReplayContext | undefined,
): ReplayEstimate {
  return indexReplay(messages).estimate(context);
}

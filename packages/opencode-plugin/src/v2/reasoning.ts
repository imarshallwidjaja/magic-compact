// OpenCode 2 reasoning records and model descriptors for the shared reasoning
// policy, and the one request-only projection that keeps another model's
// plain reasoning readable on the OpenAI Responses serializer.
import {
  NO_REPLAY,
  protectedPayload,
  READABLE_REASONING_HEADER,
  reasoningItemID,
  sameModel,
  type ModelIdentity,
  type ReasoningRoute,
  type ReasoningUnit,
  type ReplayEstimate,
} from "../policy/reasoning";
import { countGptTokens } from "../stats/tokenize";
import { isRecord } from "../util";
import type { V2Message } from "./session";

type V2Assistant = Extract<V2Message, { type: "assistant" }>;

// The package OpenCode 2 lists for models it serializes with its OpenAI
// Responses serializer, whatever their provider.
const V2_RESPONSES_PACKAGE = "@opencode/ai/providers/openai";

// The route facts of an OpenCode 2 model descriptor: `modelID` is the model
// it sends on the wire, `package` its serializer.
export function v2ReasoningRoute(
  model: ModelIdentity,
  info: { modelID?: string | undefined; package?: string | undefined },
): ReasoningRoute {
  return {
    model,
    wireModelID: info.modelID,
    responses: info.package === V2_RESPONSES_PACKAGE,
  };
}

function origin(message: V2Assistant): ModelIdentity {
  return { providerID: message.model.providerID, modelID: message.model.id };
}

function reasoningItems(message: V2Assistant) {
  return message.content.filter(item => item.type === "reasoning");
}

// Every reasoning item of the records, counted once per item: items of one
// message that share a provider item ID are one unit.
export function v2ReasoningUnits(
  messages: readonly V2Message[],
): ReasoningUnit[] {
  const units: ReasoningUnit[] = [];
  for (const message of messages) {
    if (message.type !== "assistant") continue;
    const items = new Map<string, boolean>();
    for (const [index, item] of reasoningItems(message).entries()) {
      const key = reasoningItemID(item.state) ?? `item:${index}`;
      items.set(
        key,
        (items.get(key) ?? false) || protectedPayload(item.state) !== undefined,
      );
    }
    for (const isProtected of items.values()) {
      units.push({ origin: origin(message), protected: isProtected });
    }
  }
  return units;
}

// OpenCode 2.0.28 gives plugins no way to read or set the reasoning context a
// request uses (its request schema has no such option), so the replay cost of
// protected reasoning is never estimated here. Units the model may be sent,
// its own or ones without a recorded origin, are unknown; the host sends
// another model's reasoning without its provider metadata, so those cost
// nothing beyond their text.
export function v2ReplayEstimate(
  messages: readonly V2Message[],
  model: ModelIdentity | undefined,
): ReplayEstimate {
  let unknown = 0;
  for (const unit of v2ReasoningUnits(messages)) {
    if (!unit.protected) continue;
    if (model && unit.origin && !sameModel(unit.origin, model)) continue;
    unknown += 1;
  }
  return { ...NO_REPLAY, unknown };
}

// OpenCode 2 sends another model's reasoning as generic reasoning without
// provider metadata, and its OpenAI Responses serializer drops reasoning
// without metadata, so the summary model would not see it at all. For a
// request that serializer sends, of any OpenAI model, this projection
// rewrites each such nonempty reasoning part, in a request message only, into
// assistant text at the same position under a header that marks it as
// provisional analysis. Parts that keep provider metadata are the host's
// native replay and stay untouched, so nothing is sent twice and no protected
// payload is ever turned into text. The caller applies it only to its own
// requests on that serializer.
export function projectReadableReasoning<
  M extends { readonly role: string; readonly content: readonly unknown[] },
>(messages: M[]): M[] {
  return messages.map(message => {
    if (
      message.role !== "assistant"
      || !message.content.some(isMetadataFreeReasoning)
    ) {
      return message;
    }
    return {
      ...message,
      content: message.content.map(part =>
        isMetadataFreeReasoning(part)
          ? { type: "text", text: `${READABLE_REASONING_HEADER}\n${part.text}` }
          : part,
      ),
    };
  });
}

function isMetadataFreeReasoning(
  part: unknown,
): part is { type: "reasoning"; text: string } {
  return (
    isRecord(part)
    && part["type"] === "reasoning"
    && typeof part["text"] === "string"
    && part["text"].length > 0
    && (part["providerMetadata"] === undefined
      || part["providerMetadata"] === null)
  );
}

// The header tokens the projection adds to a request on `destination`,
// counted before any budget check, with one more token per part for the
// boundary between header and text. Every nonempty reasoning item the host may
// send without metadata is counted: another model's, or one stored without
// provider state. The reasoning text itself is already in the history
// estimate.
export function readableReasoningHeaderTokens(
  history: readonly V2Message[],
  destination: ReasoningRoute,
): number {
  if (!destination.responses) return 0;
  const header = countGptTokens(`${READABLE_REASONING_HEADER}\n`) + 1;
  let tokens = 0;
  for (const message of history) {
    if (message.type !== "assistant") continue;
    for (const item of reasoningItems(message)) {
      if (item.text.length === 0) continue;
      if (
        !sameModel(origin(message), destination.model)
        || item.state === undefined
      ) {
        tokens += header;
      }
    }
  }
  return tokens;
}

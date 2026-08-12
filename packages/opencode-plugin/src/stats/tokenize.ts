import type { Part } from "@opencode-ai/sdk/v2";
import { encode } from "gpt-tokenizer";
import { unwrap, type V2Client } from "../api";
import type { MessageWithParts } from "../compact/plan";

export function countGptTokens(text: string): number {
  return encode(text).length;
}

function stringifyToolContent(value: unknown): string {
  return typeof value === "string" ? value : JSON.stringify(value);
}

function extractToolContent(part: Part): string[] {
  if (part.type !== "tool") {
    return [];
  }

  const contents: string[] = [];

  if (part.state.input !== undefined) {
    contents.push(stringifyToolContent(part.state.input));
  }

  if (part.state.status === "completed" && part.state.output !== undefined) {
    contents.push(stringifyToolContent(part.state.output));
  } else if (part.state.status === "error" && part.state.error) {
    contents.push(stringifyToolContent(part.state.error));
  }

  return contents;
}

function extractPartTexts(part: Part): string[] {
  switch (part.type) {
    case "text":
      return [part.text];
    case "reasoning":
      return [part.text];
    case "tool":
      return extractToolContent(part);
    case "subtask":
      return [part.prompt, part.description];
    case "file":
      return [part.url];
    case "snapshot":
      return [part.snapshot];
    case "patch":
      return part.files;
    default:
      return [];
  }
}

export async function countSessionTokens(
  v2: V2Client,
  sessionID: string,
): Promise<number> {
  const messages = unwrap(
    await v2.session.messages({ sessionID }),
  ) as MessageWithParts[];
  return countCanonicalContentTokens(messages);
}

export function countCanonicalContentTokens(
  messages: MessageWithParts[],
): number {
  return messages.reduce(
    (sum, message) => sum + countMessageGptTokens(message),
    0,
  );
}

function countMessageGptTokens(message: MessageWithParts): number {
  const count = message.parts.map(part =>
    countGptTokens(extractPartTexts(part).join("\n\n")),
  );
  return count.reduce((sum, count) => sum + count, 0);
}

// Most accurate for unmodified conversations. Do not use on modified conversations.
export async function getProviderTokens(
  v2: V2Client,
  sessionID: string,
): Promise<number | null> {
  const messages = unwrap(
    await v2.session.messages({ sessionID }),
  ) as MessageWithParts[];

  for (let i = messages.length - 1; i >= 0; i--) {
    const message = messages[i];
    if (!message || message.info.role !== "assistant") {
      continue;
    }

    if (!message.info.tokens) {
      return null;
    }

    const t = message.info.tokens;
    const total =
      t.input + t.output + t.reasoning + t.cache.read + t.cache.write;
    return total > 0 ? total : null;
  }

  return null;
}

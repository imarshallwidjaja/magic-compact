// Canonical stored-history accounting for OpenCode 1 records.
import type { Part } from "@opencode-ai/sdk/v2";
import {
  modelFacingCount,
  type ModelFacingCount,
  type StoredHistoryCount,
} from "../stats/stored-history";
import { countGptTokens, countGptTokensYielding } from "../stats/tokenize";
import { projectManifestCoveredMedia } from "./media";
import type { MessageWithParts } from "./plan";

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

function extractPartTexts(
  part: Part,
  role: MessageWithParts["info"]["role"],
): string[] {
  switch (part.type) {
    case "text":
      return part.ignored === true && role === "user" ? [] : [part.text];
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

// A part text this long (a file URL with inline data, or a large tool value)
// is counted without holding the event loop.
const LARGE_PART_TEXT_CHARS = 65_536;

export async function countModelFacingContentTokens(
  messages: MessageWithParts[],
): Promise<ModelFacingCount> {
  return modelFacingCount(
    await countCanonicalContent(messages),
    await countCanonicalContent(projectManifestCoveredMedia(messages)),
  );
}

// A file part whose URL carries its data inline. Canonical accounting counts
// the whole URL as text; any other URL stays an ordinary short string.
function isInlineFile(part: Part): part is Extract<Part, { type: "file" }> {
  return part.type === "file" && part.url.startsWith("data:");
}

// One counting pass over `messages`: the canonical total, the part of it
// inline file URLs contribute, and the inline files themselves.
async function countCanonicalContent(
  messages: MessageWithParts[],
): Promise<StoredHistoryCount> {
  let tokens = 0;
  let inlineFileTokens = 0;
  const inlineFiles = { count: 0, encodedBytes: 0 };
  for (const message of messages) {
    for (const part of message.parts) {
      const text = extractPartTexts(part, message.info.role).join("\n\n");
      const count =
        text.length > LARGE_PART_TEXT_CHARS
          ? await countGptTokensYielding(text)
          : countGptTokens(text);
      tokens += count;
      if (isInlineFile(part)) {
        inlineFileTokens += count;
        inlineFiles.count += 1;
        inlineFiles.encodedBytes += Buffer.byteLength(part.url);
      }
    }
  }
  return { tokens, inlineFileTokens, inlineFiles };
}

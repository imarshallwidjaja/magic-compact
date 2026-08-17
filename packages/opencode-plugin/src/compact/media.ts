import { Buffer } from "node:buffer";
import { createHash } from "node:crypto";
import type { Part, TextPart } from "@opencode-ai/sdk/v2";
import { countGptTokens } from "../stats/tokenize";
import { buildTurns, type MessageWithParts, type Turn } from "./plan";

export const MEDIA_MANIFEST_VERSION = 1;

type FilePart = Extract<Part, { type: "file" }>;

type InlineImage = {
  message: MessageWithParts;
  partIndex: number;
  part: FilePart;
};

export type MediaManifest = {
  version: typeof MEDIA_MANIFEST_VERSION;
  images: Array<{
    ordinal: number;
    mime: string;
    urlSha256: string;
  }>;
};

export function createMediaManifest(turn: Turn): MediaManifest | undefined {
  const images = inlineImages(turn, 3).map(image => image.part);
  if (images.length === 0) return undefined;
  images.forEach(assertValidInlineImage);
  return {
    version: MEDIA_MANIFEST_VERSION,
    images: images.map((part, index) => ({
      ordinal: index + 1,
      mime: part.mime,
      urlSha256: hashURL(part.url),
    })),
  };
}

export function projectManifestCoveredMedia(
  source: MessageWithParts[],
): MessageWithParts[] {
  const messages = source.map(message => ({
    ...message,
    parts: [...message.parts],
  }));

  for (const turn of buildTurns(messages)) {
    const coverage = manifestCoverage(turn);
    if (coverage === undefined) continue;
    for (const { image, entry } of coverage) {
      const placeholder = mediaPlaceholder(image.part, entry);
      if (countGptTokens(placeholder.text) < countGptTokens(image.part.url)) {
        image.message.parts[image.partIndex] = placeholder;
      }
    }
  }

  return messages;
}

export function coveredAssistantImagePartIDs(turn: Turn): Set<string> {
  return new Set(
    (manifestCoverage(turn) ?? []).flatMap(({ image }) =>
      image.message.info.role === "assistant" ? [image.part.id] : [],
    ),
  );
}

function inlineImages(turn: Turn, summaryVersion: 2 | 3): InlineImage[] {
  const messages =
    summaryVersion === 2 ? turn.user : [...turn.user, ...turn.assistants];
  return messages.flatMap(message =>
    message.parts.flatMap((part, partIndex) =>
      part.type === "file"
      && part.mime.startsWith("image/")
      && part.url.startsWith("data:")
        ? [{ message, partIndex, part }]
        : [],
    ),
  );
}

function manifestCoverage(turn: Turn):
  | Array<{
      image: InlineImage;
      entry: MediaManifest["images"][number];
    }>
  | undefined {
  const manifests = turn.assistants.flatMap(message =>
    message.parts.flatMap(part => {
      const envelope = summaryManifest(part);
      return envelope === undefined ? [] : [envelope];
    }),
  );
  if (manifests.length === 0) return undefined;
  if (manifests.length !== 1) throw manifestMismatch();
  const envelope = manifests[0]!;
  const manifest = parseManifest(envelope.manifest, envelope.version);
  const images = inlineImages(turn, envelope.version);
  if (images.length !== manifest.images.length) throw manifestMismatch();
  return images.map((image, index) => {
    const entry = manifest.images[index]!;
    if (
      image.part.mime !== entry.mime
      || hashURL(image.part.url) !== entry.urlSha256
    ) {
      throw manifestMismatch();
    }
    return { image, entry };
  });
}

function assertValidInlineImage(part: FilePart): void {
  const match = /^data:([^;,]+);base64,([A-Za-z0-9+/]*={0,2})$/u.exec(part.url);
  const payload = match?.[2];
  if (
    match?.[1] !== part.mime
    || payload === undefined
    || payload.length === 0
    || payload.length % 4 !== 0
    || Buffer.from(payload, "base64").toString("base64") !== payload
  ) {
    throw new Error("Candidate turn contains malformed inline image data.");
  }
}

function hashURL(url: string): string {
  return createHash("sha256").update(url).digest("hex");
}

function summaryManifest(
  part: Part,
): { version: 2 | 3; manifest: unknown } | undefined {
  if (part.type !== "text" || !isPlainRecord(part.metadata)) return undefined;
  const metadata = part.metadata["magicCompact"];
  if (
    !isPlainRecord(metadata)
    || metadata["summary"] !== true
    || metadata["capsule"] !== true
    || metadata["agentic"] !== true
    || (metadata["version"] !== 2 && metadata["version"] !== 3)
    || !("mediaManifest" in metadata)
  ) {
    return undefined;
  }
  return {
    version: metadata["version"],
    manifest: metadata["mediaManifest"],
  };
}

function parseManifest(value: unknown, summaryVersion: 2 | 3): MediaManifest {
  if (
    !isPlainRecord(value)
    || !exactKeys(value, ["images", "version"])
    || value["version"] !== MEDIA_MANIFEST_VERSION
    || !Array.isArray(value["images"])
    || value["images"].length === 0
  ) {
    throw manifestMismatch();
  }
  const seenPartIDs = new Set<string>();
  const images = value["images"].map((entry, index) => {
    const expectedKeys =
      summaryVersion === 2
        ? ["mime", "ordinal", "partId", "urlSha256"]
        : ["mime", "ordinal", "urlSha256"];
    if (
      !isPlainRecord(entry)
      || !exactKeys(entry, expectedKeys)
      || entry["ordinal"] !== index + 1
      || (summaryVersion === 2
        && (typeof entry["partId"] !== "string"
          || entry["partId"].length === 0
          || seenPartIDs.has(entry["partId"])))
      || typeof entry["mime"] !== "string"
      || !entry["mime"].startsWith("image/")
      || typeof entry["urlSha256"] !== "string"
      || !/^[a-f0-9]{64}$/u.test(entry["urlSha256"])
    ) {
      throw manifestMismatch();
    }
    if (summaryVersion === 2) seenPartIDs.add(entry["partId"] as string);
    return {
      ordinal: entry["ordinal"],
      mime: entry["mime"],
      urlSha256: entry["urlSha256"],
    };
  });
  return { version: MEDIA_MANIFEST_VERSION, images };
}

function mediaPlaceholder(
  part: FilePart,
  entry: MediaManifest["images"][number],
): TextPart {
  return {
    id: part.id,
    sessionID: part.sessionID,
    messageID: part.messageID,
    type: "text",
    text: `[Historical inline image ${entry.ordinal} (${entry.mime}) was summarized by Magic Compact; its payload is projected out of this model request.]`,
    synthetic: true,
  };
}

function exactKeys(value: Record<string, unknown>, keys: string[]): boolean {
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  return (
    actual.length === expected.length
    && actual.every((key, index) => key === expected[index])
  );
}

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return false;
  }
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

function manifestMismatch(): Error {
  return new Error("Magic Compact media manifest mismatch.");
}

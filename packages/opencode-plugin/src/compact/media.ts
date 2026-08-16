import { Buffer } from "node:buffer";
import { createHash } from "node:crypto";
import type { Part, TextPart } from "@opencode-ai/sdk/v2";
import { buildTurns, type MessageWithParts, type Turn } from "./plan";

export const INLINE_IMAGE_TOKEN_THRESHOLD = 1_024;
export const MEDIA_MANIFEST_VERSION = 1;

type FilePart = Extract<Part, { type: "file" }>;

export type MediaManifest = {
  version: typeof MEDIA_MANIFEST_VERSION;
  images: Array<{
    ordinal: number;
    partId: string;
    mime: string;
    urlSha256: string;
  }>;
};

export type InlineImageCohort = {
  images: Array<{
    ordinal: number;
    part: FilePart;
  }>;
  manifest: MediaManifest;
};

export function collectInlineImageCohort(
  turn: Turn,
  countTokens: (value: string) => number,
): InlineImageCohort | null {
  const parts = turn.user.flatMap(message =>
    message.parts.filter(
      (part): part is FilePart =>
        part.type === "file"
        && part.mime.startsWith("image/")
        && part.url.startsWith("data:"),
    ),
  );
  const contribution = parts.reduce(
    (tokens, part) => tokens + countTokens(part.url),
    0,
  );
  if (contribution <= INLINE_IMAGE_TOKEN_THRESHOLD) return null;

  const images = parts.map((part, index) => {
    assertValidInlineImage(part);
    return { ordinal: index + 1, part };
  });
  return {
    images,
    manifest: {
      version: MEDIA_MANIFEST_VERSION,
      images: images.map(image => ({
        ordinal: image.ordinal,
        partId: image.part.id,
        mime: image.part.mime,
        urlSha256: hashURL(image.part.url),
      })),
    },
  };
}

export function redactInlineImagesForTrace(
  turn: Turn,
  cohort: InlineImageCohort,
): Turn {
  const ordinals = new Map(
    cohort.images.map(image => [image.part.id, image.ordinal]),
  );
  const redact = (message: MessageWithParts): MessageWithParts => ({
    ...message,
    parts: message.parts.map(part => {
      const ordinal = ordinals.get(part.id);
      if (ordinal === undefined || part.type !== "file") return part;
      return { ...part, url: `magic-compact-image:${ordinal}` };
    }),
  });
  return {
    user: turn.user.map(redact),
    assistants: turn.assistants,
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
    const manifests = turn.assistants.flatMap(message =>
      message.parts.flatMap(part => {
        const manifest = capsuleManifest(part);
        return manifest === undefined ? [] : [manifest];
      }),
    );
    if (manifests.length === 0) continue;
    if (manifests.length !== 1) throw manifestMismatch();
    const manifest = parseManifest(manifests[0]);
    const images = turn.user.flatMap(message =>
      message.parts.flatMap((part, partIndex) =>
        part.type === "file"
        && part.mime.startsWith("image/")
        && part.url.startsWith("data:")
          ? [{ message, partIndex, part }]
          : [],
      ),
    );
    if (images.length !== manifest.images.length) throw manifestMismatch();
    for (const [index, image] of images.entries()) {
      const entry = manifest.images[index]!;
      if (
        image.part.id !== entry.partId
        || image.part.mime !== entry.mime
        || hashURL(image.part.url) !== entry.urlSha256
      ) {
        throw manifestMismatch();
      }
      image.message.parts[image.partIndex] = mediaPlaceholder(
        image.part,
        entry,
      );
    }
  }

  return messages;
}

function assertValidInlineImage(part: FilePart): void {
  const match = /^data:([^;,]+);base64,([A-Za-z0-9+/]*={0,2})$/u.exec(part.url);
  const payload = match?.[2];
  if (
    match?.[1] !== part.mime
    || !match[1].startsWith("image/")
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

function capsuleManifest(part: Part): unknown | undefined {
  if (part.type !== "text" || !isPlainRecord(part.metadata)) return undefined;
  const metadata = part.metadata["magicCompact"];
  if (
    !isPlainRecord(metadata)
    || metadata["summary"] !== true
    || metadata["capsule"] !== true
    || metadata["agentic"] !== true
    || metadata["version"] !== 2
    || !("mediaManifest" in metadata)
  ) {
    return undefined;
  }
  return metadata["mediaManifest"];
}

function parseManifest(value: unknown): MediaManifest {
  if (
    !isPlainRecord(value)
    || !exactKeys(value, ["images", "version"])
    || value["version"] !== MEDIA_MANIFEST_VERSION
    || !Array.isArray(value["images"])
    || value["images"].length === 0
  ) {
    throw manifestMismatch();
  }
  const seen = new Set<string>();
  const images = value["images"].map((entry, index) => {
    if (
      !isPlainRecord(entry)
      || !exactKeys(entry, ["mime", "ordinal", "partId", "urlSha256"])
      || entry["ordinal"] !== index + 1
      || typeof entry["partId"] !== "string"
      || entry["partId"].length === 0
      || seen.has(entry["partId"])
      || typeof entry["mime"] !== "string"
      || !entry["mime"].startsWith("image/")
      || typeof entry["urlSha256"] !== "string"
      || !/^[a-f0-9]{64}$/u.test(entry["urlSha256"])
    ) {
      throw manifestMismatch();
    }
    seen.add(entry["partId"]);
    return {
      ordinal: entry["ordinal"],
      partId: entry["partId"],
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

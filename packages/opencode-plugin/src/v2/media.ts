import { Buffer } from "node:buffer";
import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";
import { isDeepStrictEqual } from "node:util";
import {
  mediaManifestMismatch,
  mediaPlaceholderText,
  projectsSmaller,
} from "../policy/media";
import { isRecord } from "../util";
import { countV2ContentTokens, v2Turns } from "./history";
import type { V2Message } from "./session";

// OpenCode 2.0.28 sends a user file to the model as image media only for
// these MIME types (`session/runner/to-llm-message.ts`); a file of any other
// image type never reaches the model.
const NATIVE_IMAGE_MIMES = new Set([
  "image/png",
  "image/jpeg",
  "image/gif",
  "image/webp",
]);

// The inline user images a summary describes, bound by send order, exact MIME
// and the SHA-256 of the exact stored base64 data. The data is never decoded.
export type V2MediaManifest = {
  version: 1;
  images: Array<{ ordinal: number; mime: string; dataSha256: string }>;
};

// The files of one user message that OpenCode 2.0.28 sends as images, by
// index, in send order. It mirrors the host's conversion: a supported image
// MIME and, once a message holds two or more mentioned inline images, a
// repeat with the same MIME, name, description, mention text and data is sent
// only once. Any drift from the host shows up as a cohort mismatch in the
// request hook, which fails the request rather than projecting.
export function v2VisibleImageIndexes(message: V2Message): number[] {
  if (message.type !== "user") return [];
  const files = message.files ?? [];
  const mentioned = (file: (typeof files)[number]) =>
    NATIVE_IMAGE_MIMES.has(file.mime)
    && file.source.type === "inline"
    && Boolean(file.mention?.text);
  const deduplicate = files.filter(mentioned).length >= 2;
  const seen = new Map<string, Set<string>>();
  return files.flatMap((file, index) => {
    if (!NATIVE_IMAGE_MIMES.has(file.mime)) return [];
    if (deduplicate && mentioned(file)) {
      const key = JSON.stringify([
        file.mime,
        file.name ?? null,
        file.description ?? null,
        file.mention!.text,
      ]);
      const payloads = seen.get(key) ?? new Set<string>();
      if (payloads.has(file.data)) return [];
      payloads.add(file.data);
      seen.set(key, payloads);
    }
    return [index];
  });
}

function visibleImages(messages: readonly V2Message[]) {
  return messages.flatMap(message =>
    message.type === "user"
      ? v2VisibleImageIndexes(message).map(index => ({
          message,
          index,
          file: message.files![index]!,
        }))
      : [],
  );
}

function sha256(data: string): string {
  return createHash("sha256").update(data).digest("hex");
}

// The manifest a new summary of `messages` binds. Every bound image must be
// canonical, non-empty base64, as the host sends it unchanged; anything else
// fails before a summarizer is asked to describe it.
export function v2MediaManifest(
  messages: readonly V2Message[],
): V2MediaManifest | undefined {
  const images = visibleImages(messages);
  if (images.length === 0) return undefined;
  for (const { file } of images) {
    if (
      file.data.length === 0
      || file.data.length % 4 !== 0
      || !/^[A-Za-z0-9+/]*={0,2}$/u.test(file.data)
      || Buffer.from(file.data, "base64").toString("base64") !== file.data
    ) {
      throw new Error("Candidate turn contains malformed inline image data.");
    }
  }
  return {
    version: 1,
    images: images.map(({ file }, index) => ({
      ordinal: index + 1,
      mime: file.mime,
      dataSha256: sha256(file.data),
    })),
  };
}

// One covered image of a user message: its file index, its binding, and the
// placeholder that replaces it in requests when that is strictly smaller.
export type V2CoveredImage = {
  index: number;
  mime: string;
  dataSha256: string;
  placeholder?: string;
};

// Covered images by user message ID, in send order.
export type V2MediaProjection = Map<string, V2CoveredImage[]>;

function manifestOf(message: V2Message): unknown {
  if (message.type !== "assistant" || !isRecord(message.metadata)) return;
  const envelope = message.metadata["magicCompact"];
  if (
    !isRecord(envelope)
    || envelope["summary"] !== true
    || envelope["capsule"] !== true
    || envelope["agentic"] !== true
    || envelope["version"] !== 3
    || !("mediaManifest" in envelope)
  ) {
    return;
  }
  return envelope["mediaManifest"];
}

function parseManifest(value: unknown): V2MediaManifest {
  const keys = (record: Record<string, unknown>) =>
    Object.keys(record).sort().join(",");
  if (
    !isRecord(value)
    || keys(value) !== "images,version"
    || value["version"] !== 1
    || !Array.isArray(value["images"])
    || value["images"].length === 0
  ) {
    throw mediaManifestMismatch();
  }
  return {
    version: 1,
    images: value["images"].map((entry: unknown, index) => {
      if (
        !isRecord(entry)
        || keys(entry) !== "dataSha256,mime,ordinal"
        || entry["ordinal"] !== index + 1
        || typeof entry["mime"] !== "string"
        || !entry["mime"].startsWith("image/")
        || typeof entry["dataSha256"] !== "string"
        || !/^[a-f0-9]{64}$/u.test(entry["dataSha256"])
      ) {
        throw mediaManifestMismatch();
      }
      return {
        ordinal: entry["ordinal"],
        mime: entry["mime"],
        dataSha256: entry["dataSha256"],
      };
    }),
  };
}

// The one projection every consumer uses: the request hooks, profitability,
// token verification and stats. Every capsule is checked on its own over the
// records given, whatever native compaction stands between it and the end of
// the history: OpenCode picks the history a request replays by the model's
// provider route, so a turn behind the newest checkpoint can still be sent.
// A summarized turn whose first assistant message carries a v3 manifest must
// still hold exactly the images it binds, in order, with the same MIME and
// data; anything else throws. With `represented`, only turns holding one of
// those message IDs are checked and projected. Turns without a manifest are
// unchanged, and stored records are never modified.
export function planV2MediaProjection(
  records: readonly V2Message[],
  represented?: ReadonlySet<string>,
): V2MediaProjection {
  const projection: V2MediaProjection = new Map();
  for (const turn of v2Turns(records)) {
    if (
      represented
      && !turn.user.some(message => represented.has(message.id))
    ) {
      continue;
    }
    const manifests = turn.assistants.flatMap(message => {
      const manifest = manifestOf(message);
      return manifest === undefined ? [] : [manifest];
    });
    if (manifests.length === 0) continue;
    if (manifests.length !== 1) throw mediaManifestMismatch();
    const manifest = parseManifest(manifests[0]);
    const images = visibleImages(turn.user);
    if (images.length !== manifest.images.length) {
      throw mediaManifestMismatch();
    }
    images.forEach(({ message, index, file }, position) => {
      const entry = manifest.images[position]!;
      const dataSha256 = sha256(file.data);
      if (file.mime !== entry.mime || dataSha256 !== entry.dataSha256) {
        throw mediaManifestMismatch();
      }
      const placeholder = mediaPlaceholderText(entry.ordinal, entry.mime);
      const covered = projection.get(message.id) ?? [];
      covered.push({
        index,
        mime: entry.mime,
        dataSha256,
        ...(projectsSmaller(placeholder, file.data) ? { placeholder } : {}),
      });
      projection.set(message.id, covered);
    });
  }
  return projection;
}

export type V2ModelFacingTokens = {
  // Canonical tokens with covered images projected.
  tokens: number;
  // Tokens the projection removes from every request.
  projectedMediaTokens: number;
};

export function countV2ModelFacingTokens(
  records: readonly V2Message[],
): V2ModelFacingTokens {
  const persisted = countV2ContentTokens(records);
  const tokens = countV2ContentTokens(records, planV2MediaProjection(records));
  return { tokens, projectedMediaTokens: persisted - tokens };
}

// The fields of an `@opencode/ai` request message this projection reads.
export type V2RequestMessage = {
  readonly id?: string;
  readonly role: string;
  readonly content: readonly unknown[];
  readonly metadata?: unknown;
};

// Whether a request holds a message that may carry a summary manifest. Only
// then does projection need the session's native records.
export function requestCarriesMediaManifest(
  messages: readonly { readonly role: string; readonly metadata?: unknown }[],
): boolean {
  return messages.some(message => {
    if (message.role !== "assistant" || !isRecord(message.metadata)) {
      return false;
    }
    const envelope = message.metadata["magicCompact"];
    return isRecord(envelope) && "mediaManifest" in envelope;
  });
}

function isImageMedia(part: unknown): part is {
  media: { mediaType: string; source: unknown };
} {
  return (
    isRecord(part)
    && part["type"] === "media"
    && isRecord(part["media"])
    && typeof part["media"]["mediaType"] === "string"
    && part["media"]["mediaType"].toLowerCase().startsWith("image/")
  );
}

// The request messages that carry image media and so may need projecting.
export function requestImageMessageIDs(
  messages: readonly V2RequestMessage[],
): string[] {
  return messages.flatMap(message =>
    message.id !== undefined && message.content.some(isImageMedia)
      ? [message.id]
      : [],
  );
}

// The images of a request message by exact MIME and data hash, in order. An
// image whose data the request does not carry as base64 cannot be bound.
function requestImageCohort(message: V2RequestMessage): string {
  return message.content
    .filter(isImageMedia)
    .map(({ media }) => {
      const source = media.source;
      if (
        !isRecord(source)
        || source["type"] !== "base64"
        || typeof source["data"] !== "string"
      ) {
        throw mediaManifestMismatch();
      }
      return `${media.mediaType}:${sha256(source["data"])}`;
    })
    .join(",");
}

// JSON with object keys sorted, so equal values compare equal as strings.
function canonical(value: unknown): string {
  return JSON.stringify(value, (_, item: unknown) =>
    isRecord(item) && !Array.isArray(item)
      ? Object.fromEntries(
          Object.entries(item).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)),
        )
      : item,
  );
}

// The texts OpenCode 2.0.28 puts in place of media it will not send
// (`session/model-request.ts`): `unsupportedParts` for an input the model
// cannot take and `boundImages` past 25 MiB of image data. A provider
// checkpoint's retained copies pass through both (`recentUserMessages`), so a
// copy can hold one where its record holds a file.
const IMAGE_REMOVED_TEXT =
  "[This image was removed to reduce the request size and is no longer visible. Do not make claims about its contents from memory. If needed, retrieve it again with an available tool or ask the user to attach it again.]";
const UNSUPPORTED_MEDIA_TEXT =
  /^ERROR: Cannot read .+ \(this model does not support \S+ input\)\. Inform the user\.$/su;

function isMediaReplacement(text: unknown): boolean {
  return (
    typeof text === "string"
    && (text === IMAGE_REMOVED_TEXT || UNSUPPORTED_MEDIA_TEXT.test(text))
  );
}

// Everything of a user message but its media: its text parts in order and
// its metadata. A retained copy is matched to its source by this alone. A
// media replacement text counts as media on both sides, so a copy OpenCode
// stripped of media still matches its record; this can only widen the records
// a copy may stand for, never drop its own.
function copyIdentity(message: {
  readonly content?: unknown;
  readonly metadata?: unknown;
}): string {
  const content = Array.isArray(message.content) ? message.content : [];
  return canonical([
    content.flatMap((part: unknown) => {
      if (isRecord(part) && part["type"] === "media") return [];
      if (isRecord(part) && part["type"] === "text") {
        return isMediaReplacement(part["text"]) ? [] : [part["text"]];
      }
      return [part];
    }),
    message.metadata ?? {},
  ]);
}

type V2UserMessage = Extract<V2Message, { type: "user" }>;
type V2File = NonNullable<V2UserMessage["files"]>[number];

function fileLocation(file: V2File): string | undefined {
  if (file.source.type !== "uri") return;
  const url = URL.parse(file.source.uri);
  if (url?.protocol !== "file:") return;
  try {
    return fileURLToPath(url);
  } catch {
    return;
  }
}

// The text parts OpenCode 2.0.28 sends for a user message's files
// (`session/runner/to-llm-message.ts` `attachmentContent`). The repeats it
// drops are inline images, which send no text.
function attachmentTexts(files: readonly V2File[]): string[] {
  const block = (lines: Array<string | undefined>) =>
    `\n\n${lines.filter(line => line !== undefined).join("\n")}`;
  return files.flatMap(file => {
    const uri = file.source.type === "uri" ? file.source.uri : undefined;
    const description =
      file.description === undefined
        ? undefined
        : `Description: ${file.description}`;
    const body = Buffer.from(file.data, "base64").toString("utf8");
    if (file.mime === "text/plain") {
      return [
        block([
          `Attached file: ${file.name ?? uri ?? "inline attachment"}`,
          description,
          "",
          body,
        ]),
      ];
    }
    if (file.mime === "application/x-directory") {
      const empty = file.data.length === 0;
      return [
        block([
          `Attached directory: ${fileLocation(file) ?? file.name ?? uri ?? "directory"}`,
          description,
          empty ? undefined : "",
          empty ? undefined : body,
        ]),
      ];
    }
    if (NATIVE_IMAGE_MIMES.has(file.mime) || file.mime === "application/pdf") {
      const location = fileLocation(file);
      return location === undefined ? [] : [`Attached file: ${location}`];
    }
    return [];
  });
}

// The identities a retained copy of `record` may have. OpenCode converts the
// record as it sends it, its skills' text first, except that the window a
// provider checkpoint retains drops the skills (`recentUserMessages`).
function sourceIdentities(record: V2UserMessage): string[] {
  const texts = [
    ...(record.text === "" ? [] : [record.text]),
    ...attachmentTexts(record.files ?? []),
  ];
  const skills = (record.skills ?? []).flatMap(skill =>
    skill.text === undefined ? [] : [skill.text],
  );
  const metadata = {
    ...record.metadata,
    ...(record.agents?.length ? { agents: record.agents } : {}),
  };
  const identity = (parts: string[]) =>
    copyIdentity({
      content: parts.map(text => ({ type: "text", text })),
      metadata,
    });
  return skills.length === 0
    ? [identity(texts)]
    : [identity(texts), identity([...skills, ...texts])];
}

// A user message a provider checkpoint retains, in its stored order.
type RetainedCopy = { id: unknown; identity: string };

function retainedCopies(record: V2Message): RetainedCopy[] | undefined {
  if (record.type !== "compaction" || record.status !== "completed") return;
  const retained = record.providerContext?.messages;
  if (!Array.isArray(retained)) return;
  return retained.flatMap(entry =>
    isRecord(entry) && entry["role"] === "user"
      ? [{ id: entry["id"], identity: copyIdentity(entry) }]
      : [],
  );
}

// The user records each copy the checkpoint at `at` retains may stand for.
// A successor or a native fork copies the checkpoint unchanged under fresh
// record IDs, so a copy's ID need not name a record any more. Its source is
// then fixed by order: every user copy, with images or without, is the copy
// of a distinct user record before the checkpoint, in the same order, with
// the same identity. Images never select a source. A copy whose ID still
// names a record is that record's. A copy may stand for a record only when
// some such assignment of every copy gives it that record, which holds
// exactly when the earliest assignment of the copies before it ends before
// the record and the latest assignment of the copies after it starts after
// it. A checkpoint with no assignment at all fails the request.
function retainedCopyOrigins(
  records: readonly V2Message[],
  at: number,
  copies: readonly RetainedCopy[],
): V2UserMessage[][] {
  const users = records
    .slice(0, at)
    .filter((record): record is V2UserMessage => record.type === "user");
  const byIdentity = new Map<string, number[]>();
  users.forEach((user, position) => {
    for (const identity of new Set(sourceIdentities(user))) {
      const list = byIdentity.get(identity);
      if (list) list.push(position);
      else byIdentity.set(identity, [position]);
    }
  });
  const positions = new Map(
    users.map((user, position) => [user.id as string, position]),
  );
  const recordIDs = new Set<string>(records.map(record => record.id));
  const candidates = copies.map(copy => {
    const matching = byIdentity.get(copy.identity) ?? [];
    if (typeof copy.id !== "string" || !recordIDs.has(copy.id)) return matching;
    const own = positions.get(copy.id);
    return matching.filter(position => position === own);
  });
  const earliest: number[] = [];
  candidates.forEach((list, k) => {
    const position = list.find(j => j > (earliest[k - 1] ?? -1));
    if (position === undefined) throw mediaManifestMismatch();
    earliest[k] = position;
  });
  const latest: number[] = [];
  for (let k = candidates.length - 1; k >= 0; k--) {
    const position = candidates[k]!.findLast(
      j => j < (latest[k + 1] ?? users.length),
    );
    if (position === undefined) throw mediaManifestMismatch();
    latest[k] = position;
  }
  return candidates.map((list, k) =>
    list
      .filter(
        j => j > (earliest[k - 1] ?? -1) && j < (latest[k + 1] ?? users.length),
      )
      .map(j => users[j]!),
  );
}

// The records a request's retained copy may stand for, over every completed
// checkpoint that retains a copy under its ID with its identity. A copy no
// checkpoint retains that way fails the request.
function retainedCopySources(
  records: readonly V2Message[],
): (message: V2RequestMessage) => V2UserMessage[] {
  const checkpoints = records.flatMap((record, at) => {
    const copies = retainedCopies(record);
    return copies ? [{ at, copies }] : [];
  });
  const origins = new Map<number, V2UserMessage[][]>();
  return message => {
    const identity = copyIdentity(message);
    const sources = new Set<V2UserMessage>();
    checkpoints.forEach(({ at, copies }) => {
      if (!copies.some(copy => copy.id === message.id)) return;
      let found = origins.get(at);
      if (found === undefined) {
        found = retainedCopyOrigins(records, at, copies);
        origins.set(at, found);
      }
      copies.forEach((copy, k) => {
        if (copy.id !== message.id || copy.identity !== identity) return;
        for (const source of found[k]!) sources.add(source);
      });
    });
    if (sources.size === 0) throw mediaManifestMismatch();
    return [...sources];
  };
}

// Projects one model request whose messages OpenCode built from `records`,
// which must hold every native record the request was built from: the
// session's complete history, or a later part of it that holds every message
// of the request carrying images. A request message is paired with its record
// by the native message ID OpenCode keeps on converted messages, and the
// turns the request represents are checked whole. A user message carrying
// images under an ID no record holds is a provider checkpoint's retained copy
// from before a successor or native fork reminted the IDs. It may stand for
// any record the checkpoint's order of copies allows, and for every one of
// them it must carry exactly that record's images and be projected the same
// way, or the request fails, whether or not anything is projected. Every
// occurrence of a covered message, checkpoint copies included, must carry
// exactly the covered images, as base64 data with the same MIME and hash, or
// the request fails. Each covered image with a smaller placeholder becomes a
// text part; a replaced message is rebuilt with its own constructor and the
// stored records are never touched. A covered message the request does not
// include sends no image and needs nothing.
export function projectV2RequestMessages<M extends V2RequestMessage>(
  messages: readonly M[],
  records: readonly V2Message[],
): M[] {
  const recordIDs = new Set(records.map(record => record.id));
  const sourcesOf = retainedCopySources(records);
  const cohorts = new Map<V2Message, string>();
  const imagesOf = (record: V2Message) => {
    let cohort = cohorts.get(record);
    if (cohort === undefined) {
      cohort = visibleImages([record])
        .map(({ file }) => `${file.mime}:${sha256(file.data)}`)
        .join(",");
      cohorts.set(record, cohort);
    }
    return cohort;
  };
  const copies = new Map<M, V2UserMessage[]>();
  for (const message of messages) {
    if (
      message.role === "user"
      && message.id !== undefined
      && !recordIDs.has(message.id)
      && message.content.some(isImageMedia)
    ) {
      copies.set(message, sourcesOf(message));
    }
  }
  const represented = new Set([
    ...messages.flatMap(message =>
      message.id === undefined ? [] : [message.id],
    ),
    ...[...copies.values()].flat().map(source => source.id),
  ]);
  const projection = planV2MediaProjection(records, represented);
  const coveredFor = (message: M) => {
    const sources = copies.get(message);
    if (sources === undefined) {
      return message.role === "user" && message.id !== undefined
        ? projection.get(message.id)
        : undefined;
    }
    const cohort = requestImageCohort(message);
    // The file index is the record's own; the copy sends only the images.
    const outcomes = sources.map(source => {
      if (imagesOf(source) !== cohort) throw mediaManifestMismatch();
      return projection.get(source.id)?.map(({ index: _, ...image }) => image);
    });
    if (outcomes.some(outcome => !isDeepStrictEqual(outcome, outcomes[0]))) {
      throw mediaManifestMismatch();
    }
    return outcomes[0];
  };
  const covered = messages.map(coveredFor);
  if (covered.every(images => images === undefined)) return [...messages];
  return messages.map((message, index) => {
    const images = covered[index];
    if (!images) return message;
    const positions = message.content.flatMap((part, position) =>
      isImageMedia(part) ? [position] : [],
    );
    if (positions.length !== images.length) throw mediaManifestMismatch();
    let content: unknown[] | undefined;
    images.forEach((image, at) => {
      const position = positions[at]!;
      const { media } = message.content[position] as {
        media: { mediaType: string; source: unknown };
      };
      const source = media.source;
      if (
        !isRecord(source)
        || source["type"] !== "base64"
        || typeof source["data"] !== "string"
        || media.mediaType !== image.mime
        || sha256(source["data"]) !== image.dataSha256
      ) {
        throw mediaManifestMismatch();
      }
      if (image.placeholder === undefined) return;
      content ??= [...message.content];
      content[position] = { type: "text", text: image.placeholder };
    });
    if (!content) return message;
    const Message = message.constructor as new (input: object) => M;
    return new Message({ ...message, content });
  });
}

// The images `records` would still show the model, with the bytes OpenCode
// 2.0.28 counts toward its request image limit (`session/model-request.ts`):
// a user image no manifest projects, by its base64 length, and an image
// result of a finished tool call, by its URI length.
export function v2UnprojectedImages(
  records: readonly V2Message[],
): Array<{ mime: string; bytes: number }> {
  const projection = planV2MediaProjection(records);
  return records.flatMap(message => {
    if (message.type === "user") {
      const covered = projection.get(message.id) ?? [];
      return v2VisibleImageIndexes(message).flatMap(index => {
        if (
          covered.find(image => image.index === index)?.placeholder
          !== undefined
        ) {
          return [];
        }
        const file = message.files![index]!;
        return [{ mime: file.mime, bytes: Buffer.byteLength(file.data) }];
      });
    }
    if (message.type !== "assistant") return [];
    return message.content.flatMap(item =>
      item.type === "tool"
      && (item.state.status === "completed" || item.state.status === "error")
        ? (item.state.content ?? []).flatMap(content =>
            content.type === "file" && content.mime.startsWith("image/")
              ? [{ mime: content.mime, bytes: Buffer.byteLength(content.uri) }]
              : [],
          )
        : [],
    );
  });
}

// Whether `records` would still show the model an image.
export function v2HistoryShowsImages(records: readonly V2Message[]): boolean {
  return v2UnprojectedImages(records).length > 0;
}

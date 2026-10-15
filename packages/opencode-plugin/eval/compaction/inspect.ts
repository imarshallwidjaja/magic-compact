// Host-neutral reading of command outcomes, and the mechanical oracle that
// checks a delivered successor against its source's native records.
import { isDeepStrictEqual } from "node:util";
import { COMPACT_NOOP, compactSuccess } from "../../src/commands";
import {
  BOUNDARY_METADATA,
  POST_COMPACTION_NOTICE,
} from "../../src/compact/constants";
import { STATS_METADATA } from "../../src/stats/constants";
import { sha256, type CorpusCase } from "./corpus";
import type { SuccessorInspection } from "./host";

export const WITNESS_PREFIX = "arkive-eval-witness ";

export type CommandOutcome =
  | { kind: "delivered"; successorID: string; message: string }
  | { kind: "noop"; message: string }
  | { kind: "failed"; category: FailureCategory; message: string };

export type FailureCategory =
  | "capability"
  | "provider"
  | "schema"
  | "coverage"
  | "savings"
  | "setup"
  | "other";

// Classifies the exact text a Magic Compact command threw. Production
// messages are content-free, so the text is kept verbatim in the record.
export function classifyOutcome(message: string): CommandOutcome {
  const id = /successor session (ses_[0-9A-Za-z]+);/u.exec(message)?.[1];
  if (id && message.startsWith(compactSuccess(id))) {
    return { kind: "delivered", successorID: id, message };
  }
  if (
    message.startsWith(COMPACT_NOOP)
    || /^No assistant turns/u.test(message)
  ) {
    return { kind: "noop", message };
  }
  const category: FailureCategory =
    /image|output capacity|more output|model capacity|context|too large|trim first/iu.test(
      message,
    )
      ? "capability"
      : /StructuredOutput|turn count|envelope|shape|marker|structured/iu.test(
            message,
          )
        ? "schema"
        : /visual|coverage/iu.test(message)
          ? "coverage"
          : /sav(ing|e)s?|profitab|not smaller/iu.test(message)
            ? "savings"
            : /needs a model|summarizer|agent|configure|listener|endpoint|descriptor/iu.test(
                  message,
                )
              ? "setup"
              : /provider|model request|rate|HTTP|network/iu.test(message)
                ? "provider"
                : "other";
  return { kind: "failed", category, message };
}

type Rec = Record<string, unknown>;
const json = (value: unknown): unknown =>
  value === undefined ? undefined : JSON.parse(JSON.stringify(value));
const rec = (value: unknown): Rec =>
  typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Rec)
    : {};

type Check = { name: string; ok: boolean };

// The successor of a two-turn source compacted with N = 1: the first turn's
// assistant reply becomes one v3 capsule, the second turn and every user
// message stay as they were. Only fresh IDs, the session they belong to,
// links between those IDs, the capsule and Magic Compact's boundary and stats
// records may differ from the source, and each of those is checked exactly.
type OracleInput<T> = {
  sourceSession: Rec;
  source: T[];
  successorSession: Rec;
  successor: T[];
  corpusCase: CorpusCase;
};

function lineageCheck(
  successorSession: Rec,
  sourceID: unknown,
  keys: string[],
): Check {
  const lineage = rec(rec(successorSession["metadata"])["magicCompact"]);
  return {
    name: "successor lineage names the source and is ready, compact, count 1, root",
    ok:
      isDeepStrictEqual(Object.keys(lineage).sort(), [...keys].sort())
      && lineage["successorOf"] === sourceID
      && lineage["ready"] === true
      && lineage["operation"] === "compact"
      && lineage["compactionCount"] === 1
      && typeof lineage["createdAt"] === "number"
      && successorSession["parentID"] === undefined,
  };
}

function headerCheck(
  sourceSession: Rec,
  successorSession: Rec,
  copied: string[],
  absent: string[],
): Check {
  const lineage = rec(successorSession["metadata"])["magicCompact"];
  return {
    name: "successor header continues the source (copied fields, metadata plus lineage only)",
    ok:
      copied.every(key =>
        isDeepStrictEqual(
          json(successorSession[key]),
          json(sourceSession[key]),
        ),
      )
      && isDeepStrictEqual(
        json(successorSession["metadata"]),
        json({ ...rec(sourceSession["metadata"]), magicCompact: lineage }),
      )
      && absent.every(key => successorSession[key] === undefined)
      && rec(successorSession["time"])["archived"] === undefined,
  };
}

// The Magic Compact metadata of a v3 capsule, with the manifest the
// successor carries (checked separately against the corpus image).
function capsuleMetadata(manifest: unknown): Rec {
  return {
    magicCompact: {
      summary: true,
      capsule: true,
      agentic: true,
      version: 3,
      ...(manifest === undefined ? {} : { mediaManifest: manifest }),
    },
  };
}

function freshIDs(sourceIDs: string[], successorIDs: string[]): boolean {
  return (
    successorIDs.every(id => typeof id === "string" && id.length > 0)
    && new Set(successorIDs).size === successorIDs.length
    && successorIDs.every(id => !sourceIDs.includes(id))
  );
}

function targetRemovedCheck(successor: unknown, corpusCase: CorpusCase) {
  return {
    name: "summarized assistant text removed",
    ok: !JSON.stringify(successor).includes(
      JSON.stringify(corpusCase.targetAssistant.slice(2000, 2300)).slice(1, -1),
    ),
  };
}

function manifestCheck(
  manifest: unknown,
  corpusCase: CorpusCase,
  hash: (image: NonNullable<CorpusCase["image"]>) => [string, string],
): Check[] {
  const image = corpusCase.image;
  if (!image) return [];
  const m = rec(manifest);
  const images = Array.isArray(m["images"]) ? (m["images"] as Rec[]) : [];
  const [field, value] = hash(image);
  return [
    {
      name: "image manifest binds the one source PNG",
      ok:
        m["version"] === 1
        && images.length === 1
        && images[0]!["mime"] === image.mime
        && images[0]![field] === value,
    },
  ];
}

type V1Record = { info: Rec; parts: Rec[] };

// OpenCode 1: the successor holds [user, assistant (capsule only), boundary
// part + user, assistant, stats notice], all under fresh message and part IDs.
export function v1SuccessorInspection(
  input: OracleInput<V1Record>,
): SuccessorInspection {
  const { sourceSession, source, successorSession, successor, corpusCase } =
    input;
  const roles = (history: V1Record[]) =>
    history.map(message => message.info["role"]).join();
  const shaped =
    roles(source) === "user,assistant,user,assistant"
    && roles(successor) === "user,assistant,user,assistant,user";
  const capsulePart = shaped ? rec(successor[1]!.parts[0]) : {};
  const magic = rec(rec(capsulePart["metadata"])["magicCompact"]);
  const capsule =
    shaped && typeof capsulePart["text"] === "string" && magic["summary"]
      ? (capsulePart["text"] as string)
      : null;
  const statsText = shaped ? rec(successor[4]!.parts[0])["text"] : undefined;

  // Positional names for the successor's IDs; the expected records name the
  // same positions, so every link is compared, not just every field.
  const normalize = (history: V1Record[]) => {
    const messages = new Map(
      history.map((m, i) => [String(m.info["id"]), `M${i}`]),
    );
    const ref = (value: unknown) =>
      messages.get(String(value)) ?? `unknown:${String(value)}`;
    return history.map((m, i) => ({
      info: {
        ...m.info,
        id: `M${i}`,
        sessionID: "S",
        ...(m.info["parentID"] === undefined
          ? {}
          : { parentID: ref(m.info["parentID"]) }),
      },
      parts: m.parts.map((p, j) => ({
        ...p,
        id: `P${i}.${j}`,
        sessionID: "S",
        messageID: ref(p["messageID"]),
      })),
    }));
  };
  let expected: V1Record[] = [];
  let actual: V1Record[] = [];
  if (shaped) {
    const [u1, a1, u2, a2] = normalize(source);
    actual = json(normalize(successor)) as V1Record[];
    const at = (m: V1Record, i: number, parentID?: string): V1Record => ({
      info: {
        ...m.info,
        id: `M${i}`,
        ...(parentID ? { parentID } : {}),
      },
      parts: m.parts.map(p => ({ ...p, messageID: `M${i}` })),
    });
    const renumber = (m: V1Record, i: number) => ({
      ...m,
      parts: m.parts.map((p, j) => ({ ...p, id: `P${i}.${j}` })),
    });
    const capsuleRecord = {
      id: "P1.0",
      sessionID: "S",
      messageID: "M1",
      type: "text",
      text: capsule,
      metadata: capsuleMetadata(magic["mediaManifest"]),
    };
    const boundary = {
      id: "P2.0",
      sessionID: "S",
      messageID: "M2",
      type: "text",
      text: POST_COMPACTION_NOTICE,
      synthetic: true,
      metadata: BOUNDARY_METADATA,
    };
    const latestUser = u2!.info as Rec;
    expected = json([
      renumber(at(u1!, 0), 0),
      { ...at(a1!, 1, "M0"), parts: [capsuleRecord] },
      renumber(
        {
          ...at(u2!, 2),
          parts: [boundary, ...at(u2!, 2).parts],
        },
        2,
      ),
      renumber(at(a2!, 3, "M2"), 3),
      {
        info: {
          id: "M4",
          sessionID: "S",
          role: "user",
          time: { created: rec(successor[4]!.info["time"])["created"] },
          ...(latestUser["agent"] === undefined
            ? {}
            : { agent: latestUser["agent"] }),
          ...(latestUser["model"] === undefined
            ? {}
            : { model: latestUser["model"] }),
        },
        parts: [
          {
            id: "P4.0",
            sessionID: "S",
            messageID: "M4",
            type: "text",
            text: statsText,
            ignored: true,
            metadata: STATS_METADATA,
          },
        ],
      },
    ]) as V1Record[];
  }
  const same = (i: number) =>
    shaped && isDeepStrictEqual(actual[i], expected[i]);
  const ids = (history: V1Record[]) => history.map(m => String(m.info["id"]));
  const partIDs = (history: V1Record[]) =>
    history.flatMap(m => m.parts.map(p => String(p["id"])));
  const checks: Check[] = [
    lineageCheck(successorSession, sourceSession["id"], [
      "compactionCount",
      "successorOf",
      "operation",
      "createdAt",
      "ready",
    ]),
    headerCheck(
      sourceSession,
      successorSession,
      [
        "directory",
        "workspaceID",
        "projectID",
        "title",
        "agent",
        "model",
        "permission",
      ],
      ["parentID", "revert"],
    ),
    {
      name: "successor has the planned records: two turns plus the stats notice",
      ok: shaped,
    },
    {
      name: "successor holds exactly one v3 capsule",
      ok:
        shaped
        && magic["version"] === 3
        && !!capsule?.trim()
        && successor
          .flatMap(m => m.parts)
          .filter(
            p => rec(rec(p["metadata"])["magicCompact"])["summary"] === true,
          ).length === 1,
    },
    {
      name: "user messages preserved exactly: every native field, part, file and metadata",
      ok:
        same(0)
        && shaped
        && isDeepStrictEqual(
          { ...actual[2], parts: actual[2]!.parts.slice(1) },
          { ...expected[2], parts: expected[2]!.parts.slice(1) },
        ),
    },
    {
      name: "boundary is the one Magic Compact part added to the next user message",
      ok:
        shaped && isDeepStrictEqual(actual[2]!.parts[0], expected[2]!.parts[0]),
    },
    {
      name: "summarized turn holds only the capsule; its record is otherwise unchanged",
      ok: same(1),
    },
    { name: "protected follow-up turn kept unchanged", ok: same(3) },
    {
      name: "stats notice is the ignored Magic Compact record after the history",
      ok:
        same(4)
        && typeof statsText === "string"
        && /Stored history \(estimate\)/u.test(statsText),
    },
    targetRemovedCheck(successor, corpusCase),
    {
      name: "successor message and part IDs are fresh and unique",
      ok:
        freshIDs(ids(source), ids(successor))
        && freshIDs(partIDs(source), partIDs(successor)),
    },
    ...manifestCheck(magic["mediaManifest"], corpusCase, image => [
      "urlSha256",
      sha256(`data:${image.mime};base64,${image.base64}`),
    ]),
  ];
  return {
    checks,
    capsule,
    storedHistory: typeof statsText === "string" ? statsText : null,
  };
}

// OpenCode 2: the successor holds [user, assistant (capsule only), boundary,
// user, assistant] under fresh message IDs.
export function v2SuccessorInspection(
  input: OracleInput<Rec>,
): SuccessorInspection {
  const { sourceSession, source, successorSession, successor, corpusCase } =
    input;
  const types = (history: Rec[]) => history.map(m => m["type"]).join();
  const shaped =
    types(source) === "user,assistant,user,assistant"
    && types(successor) === "user,assistant,synthetic,user,assistant";
  const carrier = shaped ? successor[1]! : {};
  const magic = rec(rec(carrier["metadata"])["magicCompact"]);
  const content = Array.isArray(carrier["content"])
    ? (carrier["content"] as Rec[])
    : [];
  const capsule =
    shaped
    && magic["summary"] === true
    && typeof content[0]?.["text"] === "string"
      ? (content[0]["text"] as string)
      : null;
  const at = (record: Rec, i: number) => json({ ...record, id: `M${i}` });
  const actual = shaped ? successor.map((m, i) => at(m, i)) : [];
  const expected = shaped
    ? [
        at(source[0]!, 0),
        at(
          {
            ...source[1]!,
            content: [{ type: "text", text: capsule }],
            metadata: {
              ...rec(source[1]!["metadata"]),
              ...capsuleMetadata(magic["mediaManifest"]),
            },
          },
          1,
        ),
        at(
          {
            type: "synthetic",
            text: POST_COMPACTION_NOTICE,
            metadata: BOUNDARY_METADATA,
            time: { created: rec(successor[2]!["time"])["created"] },
          },
          2,
        ),
        at(source[2]!, 3),
        at(source[3]!, 4),
      ]
    : [];
  const same = (i: number) =>
    shaped && isDeepStrictEqual(actual[i], expected[i]);
  const lineage = rec(rec(successorSession["metadata"])["magicCompact"]);
  const ids = (history: Rec[]) => history.map(m => String(m["id"]));
  const checks: Check[] = [
    lineageCheck(successorSession, sourceSession["id"], [
      "compactionCount",
      "successorOf",
      "operation",
      "createdAt",
      "ready",
      "outcome",
    ]),
    headerCheck(
      sourceSession,
      successorSession,
      [
        "title",
        "agent",
        "model",
        "permissions",
        "location",
        "subpath",
        "projectID",
      ],
      ["parentID", "fork", "revert"],
    ),
    {
      name: "successor has the planned records: two turns plus the boundary",
      ok: shaped,
    },
    {
      name: "successor holds exactly one v3 capsule",
      ok:
        shaped
        && magic["version"] === 3
        && !!capsule?.trim()
        && successor.filter(
          m => rec(rec(m["metadata"])["magicCompact"])["summary"] === true,
        ).length === 1,
    },
    {
      name: "user messages preserved exactly: every native field, file and metadata",
      ok: same(0) && same(3),
    },
    {
      name: "boundary is the one Magic Compact record added before the next user message",
      ok: same(2),
    },
    {
      name: "summarized turn holds only the capsule; its record is otherwise unchanged",
      ok: same(1),
    },
    { name: "protected follow-up turn kept unchanged", ok: same(4) },
    targetRemovedCheck(successor, corpusCase),
    {
      name: "successor message IDs are fresh and unique",
      ok: freshIDs(ids(source), ids(successor)),
    },
    ...manifestCheck(magic["mediaManifest"], corpusCase, image => [
      "dataSha256",
      sha256(image.base64),
    ]),
  ];
  return {
    checks,
    capsule,
    storedHistory:
      typeof lineage["outcome"] === "string" ? lineage["outcome"] : null,
  };
}

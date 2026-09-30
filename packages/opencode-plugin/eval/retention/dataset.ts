// Builds the retention evaluation dataset from archived local OpenCode
// sessions. The archive is opened with SQLite `immutable=1`, so extraction
// never writes to it. Raw session content is written only to the scratch
// directory; the repository keeps labels keyed by opaque IDs.
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { Part } from "@opencode-ai/sdk/v2";
import {
  buildTurns,
  type MessageWithParts,
  type Turn,
} from "../../src/compact/plan";
import { trimWouldChangeContent } from "../../src/compact/prune";

export const ARCHIVE_DB =
  process.env["RETENTION_EVAL_DB"]
  ?? `${process.env["HOME"]}/.local/share/opencode/opencode.db.bak-20260901-101113`;
export const SCRATCH_DIR =
  process.env["RETENTION_EVAL_SCRATCH"]
  ?? "/tmp/opencode/task-aware-retention-eval";

// Sessions are paired by kind of work. Within each pair, the session whose
// SHA-256 ID hash sorts first is development data; the other is held out.
export const SESSION_PAIRS: Array<[string, string, string]> = [
  [
    "incident debugging",
    "ses_fcdc2e182ffeSA2j01aITOQQnd",
    "ses_03dea5104ffejUfuSyl9xWyKJi",
  ],
  [
    "tooling development",
    "ses_035ce2405ffeSnNpdxD83yCWhd",
    "ses_09acc8f15ffe8YBD78Nh317p4K",
  ],
  [
    "product documents and diagrams",
    "ses_00210d4a7ffeAZeVWzoOTCSXKc",
    "ses_fcc3dd9beffe8FApDXau3ZBM9s",
  ],
];

// Each session contributes two snapshots. The primary snapshot samples both
// recent and older candidates; the secondary snapshot adds recent candidates
// only, because the primary sample alone held too few useful examples.
export const SNAPSHOTS = [
  { fraction: 0.7, older: true },
  { fraction: 0.45, older: false },
] as const;
export const RECENT_GROUPS = 5;
export const RECENT_CAP = 25;
export const OLDER_CAP = 15;
export const SUMMARY_GROUPS = 6;

export type EvalCandidate = {
  id: string;
  sessionID: string;
  messageID: string;
  tool: string;
  input: Record<string, unknown>;
  output: string;
  groupOffset: number;
  stratum: "recent" | "older";
  turnRequest: string;
};

export type EvalSnapshot = {
  snapshotID: string;
  sessionID: string;
  kind: string;
  split: "development" | "heldout";
  snapshotMessageID: string;
  realTurnCount: number;
  snapshotRealTurn: number;
  latestUserRequest: string;
  provisionalSummaries: string[];
  candidatePool: { recent: number; older: number };
  candidates: EvalCandidate[];
};

export function sessionSplits(): Array<{
  sessionID: string;
  kind: string;
  split: "development" | "heldout";
}> {
  return SESSION_PAIRS.flatMap(([kind, a, b]) => {
    const [dev, held] = sha(a) < sha(b) ? [a, b] : [b, a];
    return [
      { sessionID: dev, kind, split: "development" as const },
      { sessionID: held, kind, split: "heldout" as const },
    ];
  });
}

function sha(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

function query<T>(sql: string): T[] {
  const result = Bun.spawnSync([
    "sqlite3",
    "-json",
    `file:${ARCHIVE_DB}?immutable=1`,
    sql,
  ]);
  if (result.exitCode !== 0) {
    throw new Error(
      `sqlite3 failed: ${result.stderr.toString().slice(0, 300)}`,
    );
  }
  const text = result.stdout.toString().trim();
  return text ? (JSON.parse(text) as T[]) : [];
}

function loadMessages(sessionID: string): MessageWithParts[] {
  if (!/^ses_[A-Za-z0-9]+$/.test(sessionID)) throw new Error("bad session id");
  // Mirrors OpenCode's message-v2 ordering: time_created, then id.
  const messages = query<{ id: string; data: string }>(
    `select id, data from message where session_id='${sessionID}' order by time_created, id`,
  );
  const parts = query<{ id: string; message_id: string; data: string }>(
    `select id, message_id, data from part where session_id='${sessionID}' order by message_id, id`,
  );
  const byMessage = new Map<string, Part[]>();
  for (const row of parts) {
    const part = {
      ...JSON.parse(row.data),
      id: row.id,
      messageID: row.message_id,
      sessionID,
    } as Part;
    const list = byMessage.get(row.message_id) ?? [];
    list.push(part);
    byMessage.set(row.message_id, list);
  }
  return messages.map(row => ({
    info: { ...JSON.parse(row.data), id: row.id, sessionID },
    parts: byMessage.get(row.id) ?? [],
  }));
}

export function userText(turn: Turn): string {
  return turn.user
    .flatMap(message => message.parts)
    .filter(
      part =>
        part.type === "text"
        && part.synthetic !== true
        && part.ignored !== true,
    )
    .map(part => (part.type === "text" ? part.text : ""))
    .join("\n")
    .trim();
}

function lastAssistantText(turns: Turn[]): string {
  for (const turn of [...turns].reverse()) {
    for (const assistant of [...turn.assistants].reverse()) {
      const texts = assistant.parts
        .filter(part => part.type === "text" && part.text.trim())
        .map(part => (part.type === "text" ? part.text.trim() : ""));
      if (texts.length > 0) return texts.join("\n");
    }
  }
  return "";
}

type Group = { request: string; turns: Turn[] };

// A group is one real user turn plus the synthetic-only turns that continue
// it (task notifications and similar host messages).
function groupTurns(turns: Turn[]): Group[] {
  const groups: Group[] = [];
  for (const turn of turns) {
    const text = userText(turn);
    if (text || groups.length === 0) {
      groups.push({ request: text, turns: [turn] });
    } else {
      groups[groups.length - 1]!.turns.push(turn);
    }
  }
  return groups.filter(group => group.request);
}

function cap(text: string, chars: number): string {
  return text.length <= chars ? text : `${text.slice(0, chars)} [...]`;
}

function sample<T extends { id: string }>(items: T[], limit: number): T[] {
  if (items.length <= limit) return items;
  const chosen = new Set(
    [...items]
      .sort((a, b) => (sha(a.id) < sha(b.id) ? -1 : 1))
      .slice(0, limit)
      .map(item => item.id),
  );
  return items.filter(item => chosen.has(item.id));
}

// Candidates come from the production rule check, so the dataset follows the
// shipped trim rules.
export async function buildSnapshot(
  sessionID: string,
  kind: string,
  split: "development" | "heldout",
  fraction: number,
  includeOlder: boolean,
): Promise<EvalSnapshot> {
  const turns = buildTurns(loadMessages(sessionID));
  const groups = groupTurns(turns);
  const snapshotIndex = Math.floor(groups.length * fraction);
  const history = groups.slice(0, snapshotIndex + 1);
  const snapshot = history[history.length - 1]!;
  // The snapshot is taken after the snapshot group completes; a trailing
  // assistantless turn cannot exist inside a completed group.
  const snapshotTurns = snapshot.turns.filter(turn => turn.assistants.length);
  snapshot.turns = snapshotTurns;

  const pools = { recent: [] as EvalCandidate[], older: [] as EvalCandidate[] };
  for (const [index, group] of history.entries()) {
    const groupOffset = index - snapshotIndex;
    const stratum = groupOffset > -RECENT_GROUPS ? "recent" : "older";
    for (const turn of group.turns) {
      for (const assistant of turn.assistants) {
        for (const part of assistant.parts) {
          if (part.type !== "tool" || part.state.status !== "completed") {
            continue;
          }
          if (!(await trimWouldChangeContent(part))) continue;
          pools[stratum].push({
            id: part.id,
            sessionID,
            messageID: assistant.info.id,
            tool: part.tool,
            input: part.state.input,
            output: part.state.output,
            groupOffset,
            stratum,
            turnRequest: cap(group.request, 400),
          });
        }
      }
    }
  }

  const provisionalSummaries = history
    .slice(-SUMMARY_GROUPS)
    .map(
      group =>
        `User asked: ${cap(group.request, 300)}\nAssistant concluded: ${cap(lastAssistantText(group.turns), 700)}`,
    );

  return {
    snapshotID: `${sessionID}@${Math.round(fraction * 100)}`,
    sessionID,
    kind,
    split,
    snapshotMessageID: snapshot.turns[0]!.user[0]!.info.id,
    realTurnCount: groups.length,
    snapshotRealTurn: snapshotIndex,
    latestUserRequest: cap(snapshot.request, 3000),
    provisionalSummaries,
    candidatePool: {
      recent: pools.recent.length,
      older: pools.older.length,
    },
    candidates: [
      ...(includeOlder ? sample(pools.older, OLDER_CAP) : []),
      ...sample(pools.recent, RECENT_CAP),
    ],
  };
}

function sheet(session: EvalSnapshot): string {
  const lines = [
    `# ${session.snapshotID} (${session.kind}, ${session.split})`,
    `snapshot real turn ${session.snapshotRealTurn + 1}/${session.realTurnCount}; pool recent=${session.candidatePool.recent} older=${session.candidatePool.older}`,
    "",
    "## Latest user request",
    session.latestUserRequest,
    "",
    "## Provisional summaries",
    ...session.provisionalSummaries.map((s, i) => `### ${i}\n${s}`),
    "",
    "## Candidates",
  ];
  session.candidates.forEach((c, i) => {
    const input = Object.fromEntries(
      Object.entries(c.input).map(([k, v]) => [
        k,
        cap(typeof v === "string" ? v : JSON.stringify(v), 300),
      ]),
    );
    const out = c.output;
    const body =
      out.length <= 1400
        ? out
        : `${out.slice(0, 900)}\n[... ${out.length - 1300} chars ...]\n${out.slice(-400)}`;
    lines.push(
      `### [${i}] ${c.id} offset=${c.groupOffset} tool=${c.tool} outChars=${out.length}`,
      `turn: ${cap(c.turnRequest, 150)}`,
      `input: ${JSON.stringify(input)}`,
      "output:",
      body,
      "",
    );
  });
  return lines.join("\n");
}

// The labels are frozen against the candidate IDs sampled when they were
// written. Returns every snapshot whose regenerated candidates differ, so a
// rule change can never silently pair old labels with new candidates.
export function labelDiscrepancies(sessions: EvalSnapshot[]): Array<{
  snapshotID: string;
  missing: string[];
  added: string[];
}> {
  const labels = JSON.parse(
    readFileSync(new URL("./labels.json", import.meta.url), "utf8"),
  ) as {
    snapshots: Array<{ snapshotID: string; labels: { partID: string }[] }>;
  };
  const built = new Map(sessions.map(s => [s.snapshotID, s]));
  const ids = new Set([
    ...built.keys(),
    ...labels.snapshots.map(s => s.snapshotID),
  ]);
  return [...ids].flatMap(snapshotID => {
    const labelled = new Set(
      labels.snapshots
        .find(s => s.snapshotID === snapshotID)
        ?.labels.map(label => label.partID) ?? [],
    );
    const candidates = new Set(
      built.get(snapshotID)?.candidates.map(c => c.id) ?? [],
    );
    const missing = [...labelled].filter(id => !candidates.has(id));
    const added = [...candidates].filter(id => !labelled.has(id));
    return missing.length || added.length
      ? [{ snapshotID, missing, added }]
      : [];
  });
}

if (import.meta.main) {
  const sessions: EvalSnapshot[] = [];
  for (const { sessionID, kind, split } of sessionSplits()) {
    for (const { fraction, older } of SNAPSHOTS) {
      sessions.push(
        await buildSnapshot(sessionID, kind, split, fraction, older),
      );
    }
  }
  const discrepancies = labelDiscrepancies(sessions);
  if (discrepancies.length > 0) {
    console.error(JSON.stringify(discrepancies, null, 1));
    throw new Error(
      "Regenerated candidates differ from the frozen labels; the dataset was not written. Label the changed candidates under the evaluation protocol before using them.",
    );
  }
  mkdirSync(join(SCRATCH_DIR, "sheets"), { recursive: true });
  writeFileSync(
    join(SCRATCH_DIR, "dataset.json"),
    JSON.stringify(sessions, null, 1),
  );
  for (const session of sessions) {
    writeFileSync(
      join(SCRATCH_DIR, "sheets", `${session.snapshotID}.md`),
      sheet(session),
    );
    console.log(
      `${session.split}\t${session.snapshotID}\t${session.kind}\tturn ${session.snapshotRealTurn + 1}/${session.realTurnCount}\tpool ${session.candidatePool.recent}+${session.candidatePool.older}\tsampled ${session.candidates.length}`,
    );
  }
}

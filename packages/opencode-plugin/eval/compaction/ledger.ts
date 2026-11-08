// The run ledger: the one record of every lifecycle fact a run, `grade` and
// `cleanup` establish, as JSON lines numbered from 1 and flushed per entry,
// so a crash leaves every intent and returned ID on disk. What a run still
// owns is always derived from it, never kept anywhere else.
import {
  closeSync,
  existsSync,
  fstatSync,
  fsyncSync,
  openSync,
  readFileSync,
  writeSync,
} from "node:fs";
import { isNonArrayRecord } from "../../src/util";
import { EvalError, type SessionLedger } from "./host";

export type LedgerEntry = {
  seq: number;
  at: string;
  kind: string;
  [key: string]: unknown;
};

// Refuses the whole file unless every line is a complete JSON object with a
// kind, a time and a positive safe-integer sequence number above the one
// before it. Numbers may skip: an earlier writer advanced its number before
// writing, so a failed append followed by a successful one left a gap in an
// otherwise intact ledger. Entries refer to each other by these numbers
// (`intentSeq`), so they are never rewritten. A torn last line from an
// interrupted write is refused too: dropping or renumbering entries could
// lose a session the run created.
function parseLedger(path: string, text: string): LedgerEntry[] {
  if (text === "") return [];
  const lines = text.split("\n");
  const torn = lines.pop() !== "";
  const fail = (line: number, why: string): never => {
    throw new EvalError(
      `The ledger ${path} is not intact at line ${line}: ${why}. Nothing was read from it or written to it.`,
    );
  };
  if (torn)
    fail(
      lines.length + 1,
      "its last entry is incomplete (an interrupted write)",
    );
  let previous = 0;
  return lines.map((line, index) => {
    let value: unknown;
    try {
      value = JSON.parse(line);
    } catch {
      return fail(index + 1, "not JSON");
    }
    if (!isNonArrayRecord(value)) return fail(index + 1, "not a JSON object");
    const seq = value["seq"];
    if (!Number.isSafeInteger(seq) || (seq as number) <= previous)
      fail(
        index + 1,
        `seq ${JSON.stringify(seq)} where a safe integer above ${previous} was expected`,
      );
    previous = seq as number;
    if (typeof value["kind"] !== "string" || value["kind"] === "")
      fail(index + 1, "no kind");
    if (typeof value["at"] !== "string") fail(index + 1, "no time");
    return value as LedgerEntry;
  });
}

export class Ledger {
  // The highest sequence number in the file as this instance last read or
  // wrote it; the next entry takes the one after it.
  private seq: number;
  // The file's size after this instance last read or wrote it.
  private size: number;
  constructor(readonly path: string) {
    const bytes = existsSync(path) ? readFileSync(path) : Buffer.alloc(0);
    this.seq = parseLedger(path, bytes.toString("utf8")).at(-1)?.seq ?? 0;
    this.size = bytes.length;
  }
  static read(path: string): LedgerEntry[] {
    return parseLedger(path, readFileSync(path, "utf8"));
  }
  append(kind: string, fields: Record<string, unknown> = {}): number {
    if (this.seq >= Number.MAX_SAFE_INTEGER)
      throw new EvalError(
        `The ledger ${this.path} has no sequence number left. Nothing more was recorded.`,
      );
    const entry = {
      seq: this.seq + 1,
      at: new Date().toISOString(),
      kind,
      ...fields,
    };
    const line = `${JSON.stringify(entry)}\n`;
    const fd = openSync(this.path, "a");
    try {
      // A precheck, not a lock: it refuses a ledger that another writer
      // changed since this instance last read or wrote it, but cannot exclude
      // one acting between this check and the write.
      if (fstatSync(fd).size !== this.size)
        throw new EvalError(
          `The ledger ${this.path} changed since this command last read or wrote it; another eval command may be using this run. Nothing more was recorded.`,
        );
      writeSync(fd, line);
      fsyncSync(fd);
    } finally {
      closeSync(fd);
    }
    this.seq = entry.seq;
    this.size += Buffer.byteLength(line);
    return entry.seq;
  }
  // Each creation names the intent it fulfils, so a later creation never
  // settles an earlier intent whose request failed.
  sessions(attemptId: string | null): SessionLedger {
    const intents = new Map<string, number>();
    return {
      intent: (role, plannedID, title) =>
        intents.set(
          role,
          this.append("session-intent", { attemptId, role, plannedID, title }),
        ),
      created: (role, id) =>
        this.append("session-created", {
          attemptId,
          role,
          id,
          intentSeq: intents.get(role) ?? null,
        }),
      removed: id => this.append("session-removed", { attemptId, id }),
    };
  }
}

// Whether a ledger is the one `runEval` wrote for this run, and so can show
// everything the run owns. Every version of `runEval` appends `run-start`
// with the run's nonce to a new ledger before anything else, run.json
// included. A ledger that begins any other way was started later for a run
// whose own ledger was gone (an earlier `grade` did so): it records what its
// later writers did, not what the run created.
export const recordsRun = (entries: LedgerEntry[], runNonce: string) =>
  entries[0]?.kind === "run-start" && entries[0]["runNonce"] === runNonce;

// The storage of a successor the run removed or found gone, while no removal
// of it has been confirmed. `paths` are where the wrapper located it, when
// recorded. `uncertainRequestIds`, present only when there are any, names
// the other requests the wrapper may hold the receipt under (see
// `receiptFor`); a confirmed removal for one of them settles it too.
export type UnconfirmedStorage = {
  attemptId: string | null;
  id: string;
  requestId: string;
  paths: { cache: string; stats: string } | null;
  uncertainRequestIds?: string[];
};

// The receipt whose storage a successor's end releases. The wrapper keeps one
// receipt per session, and each request it prepares replaces it in memory
// before the wrapper writes its answer, so a request whose answer is missing
// (a failed channel write, an earlier writer or a crash) may have replaced
// the receipt, while a refused request never does. The receipt is therefore
// held under the last request before `before` recorded as prepared or under
// any later request with no recorded answer, and without a prepared one,
// under any request with no recorded answer. `requestId` is the prepared
// request when there is one, else the last unanswered one, and
// `uncertainRequestIds` the others. `prepared` says whether the wrapper is
// known to hold a receipt for the session at all.
export function receiptFor(
  entries: LedgerEntry[],
  id: string,
  before = Number.POSITIVE_INFINITY,
) {
  const answers = new Map(
    entries
      .filter(e => e.kind === "storage-receipt" && e["id"] === id)
      .map(e => [String(e["requestId"]), e]),
  );
  const requests = entries.filter(
    e =>
      e.seq < before && e.kind === "storage-receipt-request" && e["id"] === id,
  );
  const answer = (request: LedgerEntry) =>
    answers.get(String(request["requestId"]))?.["phase"];
  const last = requests.findLastIndex(r => answer(r) === "prepared");
  const unanswered = requests
    .slice(last + 1)
    .filter(r => answer(r) === undefined);
  const prepared = last === -1 ? undefined : requests[last];
  const request = prepared ?? unanswered.at(-1);
  if (!request) return null;
  const paths = answers.get(String(request["requestId"]))?.["paths"] ?? null;
  return {
    attemptId:
      request["attemptId"] == null ? null : String(request["attemptId"]),
    requestId: String(request["requestId"]),
    paths: paths as UnconfirmedStorage["paths"],
    prepared: prepared !== undefined,
    uncertainRequestIds: unanswered
      .filter(r => r !== request)
      .map(r => String(r["requestId"])),
  };
}

// Every request a receipt from `receiptFor` may be held under.
export const receiptRequestIds = (receipt: {
  requestId: string;
  uncertainRequestIds?: string[];
}) => [receipt.requestId, ...(receipt.uncertainRequestIds ?? [])];

// The first `storage-removed` or `storage-orphaned` entry for this session
// under one of the requests its receipt may be held under.
export const storageRecord = (
  entries: LedgerEntry[],
  id: string,
  receipt: Parameters<typeof receiptRequestIds>[0],
  kind: "storage-removed" | "storage-orphaned",
) => {
  const requests = receiptRequestIds(receipt);
  return entries.find(
    e =>
      e.kind === kind
      && e["id"] === id
      && requests.includes(String(e["requestId"])),
  );
};

// What the ledger leaves open: sessions created and not removed or found
// absent; session intents with no creation or resolution recorded, whether
// or not their ID was chosen in advance; attempts whose command started
// without a settled outcome, including after a crash that wrote nothing
// more; and the storage of each successor the run removed or found gone
// after a receipt whose removal was not confirmed, including after a crash
// between the deletion and its outcome.
export function outstanding(entries: LedgerEntry[]) {
  const ends = new Map<string, LedgerEntry>();
  for (const e of entries)
    if (
      (e.kind === "session-removed" || e.kind === "session-absent")
      && !ends.has(String(e["id"]))
    )
      ends.set(String(e["id"]), e);
  const sessions = entries.filter(
    e => e.kind === "session-created" && !ends.has(String(e["id"])),
  );
  const resolved = new Set(
    entries
      .filter(
        e =>
          (e.kind === "session-created" && e["intentSeq"] != null)
          || e.kind === "session-intent-resolved",
      )
      .map(e => Number(e["intentSeq"])),
  );
  const intents = entries.filter(
    e => e.kind === "session-intent" && !resolved.has(e.seq),
  );
  const settled = new Set(
    entries
      .filter(e => e.kind === "attempt-settled")
      .map(e => String(e["attemptId"])),
  );
  const attempts = entries.filter(
    e => e.kind === "command-start" && !settled.has(String(e["attemptId"])),
  );
  const storage: UnconfirmedStorage[] = [];
  for (const [id, end] of ends) {
    const receipt = receiptFor(entries, id, end.seq);
    if (!receipt || storageRecord(entries, id, receipt, "storage-removed"))
      continue;
    const orphaned = storageRecord(entries, id, receipt, "storage-orphaned");
    const { uncertainRequestIds } = receipt;
    storage.push({
      attemptId: receipt.attemptId,
      id,
      requestId: receipt.requestId,
      paths: (orphaned?.["paths"]
        ?? receipt.paths) as UnconfirmedStorage["paths"],
      ...(uncertainRequestIds.length ? { uncertainRequestIds } : {}),
    });
  }
  return { sessions, intents, attempts, storage };
}

// What the run still owns, as `cleanup` and `report` name it: each open
// session with the reason it was last kept, unconfirmed storage, session
// creations neither confirmed nor ruled out, and attempts whose outcome is
// unknown.
export type Residue = {
  sessions: Array<{
    id: string;
    attemptId: string | null;
    role: string;
    reason: string | null;
  }>;
  storage: UnconfirmedStorage[];
  unresolvedIntents: Array<{
    attemptId: unknown;
    role: unknown;
    plannedID: unknown;
  }>;
  unknownOutcomes: string[];
};

export function residue(entries: LedgerEntry[]): Residue {
  const open = outstanding(entries);
  return {
    sessions: open.sessions.map(e => {
      const reason = entries.findLast(
        k => k.kind === "session-retained" && k["id"] === e["id"],
      )?.["reason"];
      return {
        id: String(e["id"]),
        attemptId: e["attemptId"] == null ? null : String(e["attemptId"]),
        role: String(e["role"]),
        reason: typeof reason === "string" ? reason : null,
      };
    }),
    storage: open.storage,
    unresolvedIntents: open.intents.map(e => ({
      attemptId: e["attemptId"],
      role: e["role"],
      plannedID: e["plannedID"],
    })),
    unknownOutcomes: open.attempts.map(e => String(e["attemptId"])),
  };
}

// Any resource the run still owns, anywhere in the ledger, blocks disposal
// of every root.
export const isClear = (left: Residue) =>
  Object.values(left).every(resources => resources.length === 0);

// A scratch root whose latest disposal kept it, with the reason, and the
// sessions the host still listed there when that was the reason. These are
// not resources the run owns: a marker that proves nothing, or a session the
// run never created, is the eval's to report, not to remove. Each stays
// reported until a later disposal removes the root or finds it gone.
export type UnresolvedOwnership = {
  directory: string;
  reason: string | null;
  location?: string;
  sessions?: string[];
};

export function unresolvedOwnership(
  entries: LedgerEntry[],
): UnresolvedOwnership[] {
  const latest = new Map<string, LedgerEntry>();
  for (const e of entries)
    if (
      e.kind === "scratch-kept"
      || e.kind === "scratch-removed"
      || e.kind === "scratch-absent"
    )
      latest.set(String(e["directory"]), e);
  return [...latest.values()]
    .filter(e => e.kind === "scratch-kept")
    .map(e => ({
      directory: String(e["directory"]),
      reason: typeof e["reason"] === "string" ? e["reason"] : null,
      ...(typeof e["location"] === "string" ? { location: e["location"] } : {}),
      ...(Array.isArray(e["sessions"])
        ? { sessions: e["sessions"].map(String) }
        : {}),
    }));
}

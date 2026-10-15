// Server identity, model selection and the host adapter contract shared by
// the OpenCode 1 and OpenCode 2 drivers.
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { sha256, type Corpus, type CorpusCase, type ModelRef } from "./corpus";

// Versions whose loader, wrapper and lifecycle route were qualified in
// containers. Any other version fails closed unless the operator names that
// exact version, which the run records as operator-asserted.
export const QUALIFIED_VERSIONS = {
  v1: ["1.18.30", "1.19.0"],
  v2: ["2.0.28"],
} as const;

export type HostKind = "v1" | "v2";
export type ServerIdentity = {
  // The canonical URL the eval reached the server at.
  url: string;
  kind: HostKind;
  version: string;
  qualification: "qualified" | "operator-asserted";
  // The server's process ID from /api/info on OpenCode 2. OpenCode 1 reports
  // none; its process ID comes from the eval wrapper's live witness.
  pid: number | null;
};
export type ServerAuth = { username: string; password: string } | undefined;
export type FetchLike = (
  input: string | URL | Request,
  init?: RequestInit,
) => Promise<Response>;

export class EvalError extends Error {}

// The CLI reaches the server only with its URL and the server's Basic Auth
// from the environment; provider credentials stay inside the server.
export function serverAuth(
  env: Record<string, string | undefined> = process.env,
): ServerAuth {
  const password = env["OPENCODE_SERVER_PASSWORD"];
  return password
    ? { username: env["OPENCODE_SERVER_USERNAME"] || "opencode", password }
    : undefined;
}

export function authHeaders(auth: ServerAuth): Record<string, string> {
  return auth
    ? {
        Authorization: `Basic ${Buffer.from(`${auth.username}:${auth.password}`).toString("base64")}`,
      }
    : {};
}

// Packed artifacts and the run channel are local paths the server must read
// and write, so only a server on this machine's loopback interface qualifies.
export function loopbackServerUrl(raw: string): string {
  if (!URL.canParse(raw)) throw new EvalError(`--server ${raw} is not a URL.`);
  const url = new URL(raw);
  if (
    !["http:", "https:"].includes(url.protocol)
    || !["127.0.0.1", "localhost", "[::1]"].includes(url.hostname)
    || url.username
    || url.password
    || (url.pathname !== "/" && url.pathname !== "")
    || url.search
    || url.hash
  ) {
    throw new EvalError(
      `--server must be a loopback OpenCode server such as http://127.0.0.1:4096. Remote servers are not supported: the server must load the packed plugin and write the run channel from this machine's file system. Pass server credentials only through OPENCODE_SERVER_USERNAME and OPENCODE_SERVER_PASSWORD.`,
    );
  }
  return url.origin;
}

export type ModelSelector = ModelRef & { selector: string };

export function parseModelSelector(raw: string, flag: string): ModelSelector {
  const match = /^([^/#\s]+)\/([^#\s]+?)(?:#([^#\s]+))?$/u.exec(raw);
  if (!match) {
    throw new EvalError(
      `${flag} must be provider/model or provider/model#variant, got ${JSON.stringify(raw)}.`,
    );
  }
  return {
    selector: raw,
    providerID: match[1]!,
    modelID: match[2]!,
    ...(match[3] ? { variant: match[3] } : {}),
  };
}

async function readJson(
  fetchImpl: FetchLike,
  url: string,
  headers: Record<string, string>,
): Promise<{ status: number; json?: Record<string, unknown> }> {
  const response = await fetchImpl(url, { headers });
  const text = await response.text();
  if (!response.ok) return { status: response.status };
  if (!(response.headers.get("content-type") ?? "").includes("json"))
    return { status: response.status };
  try {
    const json = JSON.parse(text);
    return {
      status: response.status,
      json: typeof json === "object" && json !== null ? json : undefined,
    };
  } catch {
    return { status: response.status };
  }
}

const major = (version: unknown) =>
  typeof version === "string" ? Number(/^(\d+)\./u.exec(version)?.[1]) : NaN;

// A 200 HTML page is not host identity. OpenCode 1 answers /global/health
// with JSON naming a 1.x version; OpenCode 2 answers /api/info with JSON
// naming a 2.x version and its process ID. Exactly one must qualify.
export async function identifyServer(
  url: string,
  auth: ServerAuth,
  fetchImpl: FetchLike,
  assertVersion?: string,
): Promise<ServerIdentity> {
  const headers = authHeaders(auth);
  let health: Awaited<ReturnType<typeof readJson>>;
  let info: Awaited<ReturnType<typeof readJson>>;
  try {
    [health, info] = await Promise.all([
      readJson(fetchImpl, `${url}/global/health`, headers),
      readJson(fetchImpl, `${url}/api/info`, headers),
    ]);
  } catch {
    throw new EvalError(`No OpenCode server answered at ${url}.`);
  }
  if (health.status === 401 || info.status === 401) {
    throw new EvalError(
      "The server refused the eval's credentials. Set OPENCODE_SERVER_USERNAME and OPENCODE_SERVER_PASSWORD to the server's Basic Auth.",
    );
  }
  const candidates: Array<{
    url: string;
    kind: HostKind;
    version: string;
    pid: number | null;
  }> = [];
  if (health.json?.["healthy"] === true && major(health.json["version"]) === 1)
    candidates.push({
      url,
      kind: "v1",
      version: String(health.json["version"]),
      pid: null,
    });
  if (
    major(info.json?.["version"]) === 2
    && typeof info.json?.["pid"] === "number"
  )
    candidates.push({
      url,
      kind: "v2",
      version: String(info.json["version"]),
      pid: info.json["pid"],
    });
  if (candidates.length !== 1) {
    throw new EvalError(
      `${url} did not identify as exactly one OpenCode 1 or OpenCode 2 server.`,
    );
  }
  const found = candidates[0]!;
  if (
    (QUALIFIED_VERSIONS[found.kind] as readonly string[]).includes(
      found.version,
    )
  )
    return { ...found, qualification: "qualified" };
  if (assertVersion === found.version)
    return { ...found, qualification: "operator-asserted" };
  throw new EvalError(
    `OpenCode ${found.version} is not a qualified eval host (qualified: ${[...QUALIFIED_VERSIONS.v1, ...QUALIFIED_VERSIONS.v2].join(", ")}). Its plugin loader, wrapper route and session lifecycle are unverified, so the eval refuses before creating anything. To run anyway on your own judgment, pass --assert-host-version ${found.version}; the run is then recorded as operator-asserted, not qualified.`,
  );
}

// One line the wrapper wrote to the run channel.
export type ChannelLine = {
  at: number;
  pid: number;
  runNonce: string;
  kind: string;
  [key: string]: unknown;
};

export function readChannel(channel: string, runNonce: string): ChannelLine[] {
  const lines: ChannelLine[] = [];
  for (const name of readdirSync(channel).sort()) {
    if (!/^host-\d+\.jsonl$/u.test(name)) continue;
    for (const line of readFileSync(join(channel, name), "utf8").split("\n")) {
      if (!line) continue;
      try {
        const value = JSON.parse(line) as ChannelLine;
        if (value.runNonce === runNonce) lines.push(value);
      } catch {
        // A line still being appended is read again later.
      }
    }
  }
  return lines.sort((a, b) => a.at - b.at);
}

// The sanitized description of the selected model that the run records.
export type ModelDescriptor = {
  providerID: string;
  modelID: string;
  variant: string | null;
  name: string | null;
  package: string | null;
  variants: string[];
  imageInput: boolean;
  limit: { context?: number; input?: number; output?: number };
};

// Host setup that can change model behavior, recorded without contents:
// plugin specs, instruction path fingerprints and MCP server names.
export type EffectiveSetup = {
  plugins: string[];
  instructionFingerprints: string[];
  mcpDisabled: string[];
  summarizer: { model: string | null; variant: string | null };
  // The summarizer as the subject itself reads it: the agent list on
  // OpenCode 1, the agent read at the eval project's location on OpenCode 2.
  subjectSummarizer: { model: string | null; variant: string | null };
  commands: string[];
};

export type Witness = {
  runNonce: string;
  artifactKey: string;
  subjectTgzSha256: string;
  subjectManifestSha256: string;
  subjectMismatches: number;
  pid: number;
  initCount: number;
};

// The state a removal compares against: digests of the normalized session
// header and of the complete native history.
export type ExpectedState = {
  headerSha256: string;
  historySha256: string;
};

// One read of a session: the native record, its complete native history and
// their digests. The header digest covers every field that can change the
// session's behavior; only timing and accounting fields are left out.
export type SessionState = ExpectedState & {
  session: Record<string, unknown>;
  history: Array<Record<string, unknown>>;
  // The location directory the host reports for the session.
  directory: string | undefined;
  // OpenCode 2 only: the session's active instruction entries, sorted by key.
  // They are part of the header digest. OpenCode 1 has no such state.
  instructions?: Array<Record<string, unknown>>;
};

export function stateOf(
  session: Record<string, unknown>,
  header: unknown,
  history: Array<Record<string, unknown>>,
  directory: unknown,
): SessionState {
  return {
    session,
    history,
    directory: typeof directory === "string" ? directory : undefined,
    headerSha256: sha256(JSON.stringify(header)),
    historySha256: sha256(JSON.stringify(history)),
  };
}

// Why a session cannot be removed now. Unreadable facts throw, so a caller
// keeps the session.
export type PendingFacts =
  { exists: false } | { exists: true; reasons: string[] };

export type CommandResult =
  | {
      kind: "returned";
      status: number | null;
      message: string | null;
      ms: number;
    }
  | { kind: "lost"; error: string; ms: number };

export type SuccessorInspection = {
  checks: Array<{ name: string; ok: boolean }>;
  capsule: string | null;
  storedHistory: string | null;
};

export type JudgeResult = {
  sessionID: string | null;
  // The judge's complete answer text, parsed strictly by the caller.
  rawText: string | null;
  ms: number;
  usage: Record<string, unknown> | null;
  error: string | null;
  // The state the judge session must still have to be removed: what the run
  // itself created plus, on OpenCode 1, its own prompt and the reply the host
  // returned for it. Null when ownership could not be shown; the session is
  // then kept.
  expected: ExpectedState | null;
  ownership: string | null;
};

export type SessionRole = "source" | "template" | "judge";

// Records each session the run creates: the intent before the request (with
// the planned ID when the host lets the caller choose it, and the title that
// identifies it), the returned ID as soon as it is known, and removals.
export type SessionLedger = {
  intent(role: SessionRole, plannedID: string | null, title: string): void;
  created(role: SessionRole, id: string): void;
  removed(id: string): void;
};

// What one host driver provides. Every call is scoped to an explicit location
// directory and to session IDs the run itself created.
export interface EvalHost {
  readonly kind: HostKind;
  globalMcpNames(inspectDir: string): Promise<string[]>;
  projectConfig(options: {
    pluginEntry: unknown;
    summarizer: ModelSelector;
    mcp: string[];
  }): { file: string; content: Record<string, unknown> };
  judgeConfig(options: {
    judge: ModelSelector;
    mcp: string[];
    system: string;
  }): { file: string; content: Record<string, unknown> };
  pluginEntry(
    artifact: { v1Tgz: string; v2Dir: string },
    options: Record<string, unknown>,
  ): unknown;
  setup(
    directory: string,
  ): Promise<{ setup: EffectiveSetup; witness: Witness | null }>;
  // The live server's process ID: /api/info on OpenCode 2, and on OpenCode 1
  // the PID in the witness of the run's own wrapper read at `directory` now,
  // or null when no witness of that run loads there. Unreadable facts throw.
  serverPid(directory: string, runNonce: string): Promise<number | null>;
  // The model and variant the host resolves for an agent at a location.
  resolvedAgent(
    directory: string,
    agent: string,
  ): Promise<{ model: string | null; variant: string | null }>;
  describeModel(
    directory: string,
    model: ModelSelector,
  ): Promise<ModelDescriptor | null>;
  createSource(options: {
    directory: string;
    title: string;
    corpus: Corpus;
    corpusCase: CorpusCase;
    model: ModelSelector;
    ledger: SessionLedger;
  }): Promise<{
    sessionID: string;
    // The source as read once after seeding, and what that same read showed
    // against the corpus. The run adopts `state` as the source's recorded
    // state only when every check passed.
    state: SessionState;
    checks: Array<{ name: string; ok: boolean }>;
    // OpenCode 2 imports the source next to an empty template session that
    // the run removes once the source exists.
    template?: { id: string; expected: ExpectedState };
  }>;
  read(directory: string, sessionID: string): Promise<SessionState>;
  runCommand(
    directory: string,
    sessionID: string,
    name: string,
    args: string,
  ): Promise<CommandResult>;
  // Checks a delivered successor against its source as read before the
  // command ran.
  inspectSuccessor(options: {
    directory: string;
    source: SessionState;
    successorID: string;
    corpus: Corpus;
    corpusCase: CorpusCase;
  }): Promise<SuccessorInspection & { state: SessionState }>;
  // Running work, waiting input and child sessions that keep a session.
  pending(directory: string, sessionID: string): Promise<PendingFacts>;
  // IDs of every session the host lists at a location. Read only: it names
  // sessions to keep, never sessions to remove.
  inventory(directory: string): Promise<string[]>;
  remove(directory: string, sessionID: string): Promise<void>;
  dispose(directory: string): Promise<void>;
  judge(options: {
    directory: string;
    judge: ModelSelector;
    title: string;
    prompt: string;
    ledger: SessionLedger;
  }): Promise<JudgeResult>;
}

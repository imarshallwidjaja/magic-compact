// OpenCode 2 driver: the public client, every call bound to an explicit
// location or a session the run created.
import { OpenCode } from "@opencode/client";
import { isDeepStrictEqual } from "node:util";
import { mintV2ID } from "../../src/v2/session";
import { sha256, v2SourceHistory } from "./corpus";
import {
  authHeaders,
  stateOf,
  type CommandResult,
  type EvalHost,
  type ExpectedState,
  type FetchLike,
  type ServerAuth,
  type SessionState,
} from "./host";
import { v2SuccessorInspection, WITNESS_PREFIX } from "./inspect";

const CONTROL_MS = 30_000;
const JUDGE_AGENT = "arkive-eval-judge";
const DENY_ALL = [{ action: "*", resource: "*", effect: "deny" as const }];

type Rec = Record<string, unknown>;
const json = (value: unknown) => JSON.parse(JSON.stringify(value ?? null));
const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));
const tag = (error: unknown) =>
  typeof error === "object" && error !== null
    ? (error as Rec)["_tag"]
    : undefined;

const sortedEntries = (entries: Rec[]) =>
  json(
    [...entries].sort((a, b) =>
      String(a["key"]).localeCompare(String(b["key"])),
    ),
  ) as Rec[];

// Every session field that can change how the session behaves or what it
// belongs to, and its active instruction entries. Left out: update, idle and
// view times, the last execution outcome, cost and tokens.
export function v2Header(session: Rec, instructions: Rec[]): unknown {
  const time = (session["time"] ?? {}) as Rec;
  return json({
    instructions: sortedEntries(instructions),
    id: session["id"],
    parentID: session["parentID"],
    fork: session["fork"],
    title: session["title"],
    agent: session["agent"],
    model: session["model"],
    metadata: session["metadata"],
    permissions: session["permissions"],
    location: session["location"],
    subpath: session["subpath"],
    projectID: session["projectID"],
    revert: session["revert"],
    created: time["created"],
    archived: time["archived"],
  });
}

// What the run expects of a session it created and never prompted: the
// header the host returned for the creation, no instruction entries (OpenCode
// 2.0.28 creates none) and an empty history.
export function createdState(created: Rec): ExpectedState {
  const { headerSha256, historySha256 } = stateOf(
    created,
    v2Header(created, []),
    [],
    undefined,
  );
  return { headerSha256, historySha256 };
}

// `host` lets tests supply an in-memory host.
export function v2Host(
  url: string,
  auth: ServerAuth,
  fetchImpl: FetchLike,
  host: ReturnType<typeof OpenCode.make> = OpenCode.make({
    baseUrl: url,
    headers: authHeaders(auth),
    fetch: fetchImpl as typeof fetch,
  }),
): EvalHost {
  const within = (ms = CONTROL_MS) => ({ signal: AbortSignal.timeout(ms) });
  const location = (directory: string) => ({ location: { directory } });
  const exportOf = async (sessionID: string) =>
    (await host.session.export(
      { sessionID, sanitize: false },
      within(),
    )) as unknown as {
      messages: Rec[];
    };
  const read = async (sessionID: string): Promise<SessionState> => {
    const session = json(
      await host.session.get({ sessionID }, within()),
    ) as Rec;
    const history = json((await exportOf(sessionID)).messages) as Rec[];
    const instructions = sortedEntries(
      (await host.session.instructions.entry.list(
        { sessionID },
        within(),
      )) as unknown as Rec[],
    );
    return {
      ...stateOf(
        session,
        v2Header(session, instructions),
        history,
        (session["location"] as Rec | undefined)?.["directory"],
      ),
      instructions,
    };
  };
  const resolvedAgent = async (directory: string, agentID: string) => {
    const model = (
      (await host.agent.get({ agentID, ...location(directory) }, within()))
        .data as unknown as Rec
    )["model"] as Rec | undefined;
    return {
      model: model ? `${model["providerID"]}/${model["id"]}` : null,
      variant: typeof model?.["variant"] === "string" ? model["variant"] : null,
    };
  };
  // OpenCode 2.0.28 skips a partial `{disabled: true}` override as malformed
  // and starts the inherited server, so each inherited name is replaced by a
  // complete, disabled stub. Nothing from the inherited definition is copied.
  const mcpConfig = (mcp: string[]) =>
    mcp.length
      ? {
          mcp: {
            servers: Object.fromEntries(
              mcp.map(name => [
                name,
                { type: "local", command: ["true"], disabled: true },
              ]),
            ),
          },
        }
      : {};

  return {
    kind: "v2",

    async globalMcpNames(inspectDir) {
      try {
        const entries = (await host.config.get(
          location(inspectDir),
          within(),
        )) as unknown as Array<{
          info?: { mcp?: { servers?: object } };
        }>;
        return [
          ...new Set(
            entries.flatMap(entry =>
              Object.keys(entry.info?.mcp?.servers ?? {}),
            ),
          ),
        ].sort();
      } finally {
        await host.debug.location.evict(location(inspectDir), within());
      }
    },

    pluginEntry(artifact, options) {
      return {
        package: artifact.v2Dir,
        options: { serverUrl: url, ...options },
      };
    },

    projectConfig({ pluginEntry, summarizer, mcp }) {
      return {
        file: "opencode.jsonc",
        content: {
          $schema: "https://opencode.ai/config.json",
          plugins: ["-arkive-compact", pluginEntry],
          agents: {
            "magic-compact-summarizer": { model: summarizer.selector },
          },
          ...mcpConfig(mcp),
        },
      };
    },

    judgeConfig({ judge, mcp, system }) {
      return {
        file: "opencode.jsonc",
        content: {
          $schema: "https://opencode.ai/config.json",
          plugins: ["-arkive-compact"],
          agents: {
            [JUDGE_AGENT]: {
              model: judge.selector,
              mode: "primary",
              system,
              permissions: DENY_ALL,
            },
          },
          ...mcpConfig(mcp),
        },
      };
    },

    async setup(directory) {
      // Plugin activation can finish after the first read; read until two
      // consecutive reads agree.
      let last = "";
      let plugins: Rec[] = [];
      for (let attempt = 0; attempt < 40; attempt++) {
        plugins = (await host.plugin.list(location(directory), within()))
          .data as unknown as Rec[];
        const current = JSON.stringify(plugins);
        if (current === last) break;
        last = current;
        await sleep(750);
      }
      const commands = (await host.command.list(location(directory), within()))
        .data as unknown as Rec[];
      const agents = (await host.agent.list(location(directory), within()))
        .data as unknown as Rec[];
      const entries = (await host.config.get(
        location(directory),
        within(),
      )) as unknown as Array<{
        info?: { instructions?: string[] };
      }>;
      // The effective state the host reports, not what the config files say.
      const servers = (await host.mcp.list(location(directory), within()))
        .data as unknown as Array<{
        name: string;
        status: { status: string };
      }>;
      const disabled = servers
        .filter(server => server.status.status === "disabled")
        .map(server => server.name);
      const witnessText = commands.find(
        c => c["name"] === "arkive-eval-witness",
      )?.["description"];
      const summarizer = agents.find(
        agent => (agent["name"] ?? agent["id"]) === "magic-compact-summarizer",
      );
      const model = summarizer?.["model"] as Rec | undefined;
      // The located agent read is what the subject itself reads.
      const subjectSummarizer = await resolvedAgent(
        directory,
        "magic-compact-summarizer",
      ).catch(() => ({ model: null, variant: null }));
      return {
        setup: {
          plugins: plugins
            .filter(
              plugin =>
                (plugin["source"] as Rec | undefined)?.["type"] !== "builtin",
            )
            .map(plugin => {
              const source = (plugin["source"] ?? {}) as Rec;
              return `${plugin["id"]}@${source["type"]}:${source["path"] ?? source["target"]}${plugin["error"] ? " (error)" : ""}`;
            }),
          instructionFingerprints: entries.flatMap(entry =>
            (entry.info?.instructions ?? []).map(path => sha256(String(path))),
          ),
          mcpDisabled: disabled.sort(),
          summarizer: {
            model: model ? `${model["providerID"]}/${model["id"]}` : null,
            variant:
              typeof model?.["variant"] === "string" ? model["variant"] : null,
          },
          subjectSummarizer,
          commands: commands.map(c => String(c["name"])).sort(),
        },
        witness:
          typeof witnessText === "string"
          && witnessText.startsWith(WITNESS_PREFIX)
            ? JSON.parse(witnessText.slice(WITNESS_PREFIX.length))
            : null,
      };
    },

    resolvedAgent,

    async describeModel(directory, model) {
      const models = (await host.model.list(location(directory), within()))
        .data as unknown as Rec[];
      const found = models.find(
        m => m["providerID"] === model.providerID && m["id"] === model.modelID,
      );
      if (!found) return null;
      const capabilities = found["capabilities"] as
        { input?: unknown } | undefined;
      const limit = (found["limit"] ?? {}) as Record<string, number>;
      return {
        providerID: model.providerID,
        modelID: model.modelID,
        variant: model.variant ?? null,
        name: typeof found["name"] === "string" ? found["name"] : null,
        package: typeof found["package"] === "string" ? found["package"] : null,
        variants: ((found["variants"] ?? []) as Rec[])
          .map(v => String(v["id"] ?? v))
          .sort(),
        imageInput:
          Array.isArray(capabilities?.input)
          && capabilities.input.includes("image"),
        limit: {
          context: limit["context"],
          input: limit["input"],
          output: limit["output"],
        },
      };
    },

    async createSource({
      directory,
      title,
      corpus,
      corpusCase,
      model,
      ledger,
    }) {
      const ref = {
        providerID: model.providerID,
        id: model.modelID,
        ...(model.variant ? { variant: model.variant } : {}),
      };
      // An empty session supplies the project and location records an
      // import needs; the run removes it once the source exists.
      const templateID = mintV2ID("ses_", Date.now(), 1, true);
      const templateTitle = `${title} (template)`;
      ledger.intent("template", templateID, templateTitle);
      const template = json(
        await host.session.create(
          {
            id: templateID,
            ...location(directory),
            title: templateTitle,
            agent: "build",
            model: ref,
          },
          within(),
        ),
      ) as Rec;
      ledger.created("template", templateID);
      // The source copies the template's header, so it must be what the run
      // asked for. OpenCode 2.0.28 records variant "default" when none was
      // requested; nothing else the run did not ask for is accepted.
      if (
        !isDeepStrictEqual(
          json({
            id: template["id"],
            title: template["title"],
            agent: template["agent"],
            model: template["model"],
            location: template["location"],
            parentID: template["parentID"],
            fork: template["fork"],
            metadata: template["metadata"],
            permissions: template["permissions"],
            revert: template["revert"],
          }),
          {
            id: templateID,
            title: templateTitle,
            agent: "build",
            model: { variant: "default", ...ref },
            location: { directory },
          },
        )
      )
        throw new Error(
          "the host created the template with fields the run did not request",
        );
      const now = Date.now();
      const sourceID = mintV2ID("ses_", now, 1, true);
      const records = v2SourceHistory(corpus, corpusCase, model, index =>
        mintV2ID("msg_", now, index + 1, false),
      );
      const info: Rec = {
        ...structuredClone(template),
        id: sourceID,
        title,
        metadata: {},
        time: { created: now, updated: now },
      };
      delete info["parentID"];
      delete info["fork"];
      delete info["revert"];
      ledger.intent("source", sourceID, title);
      await host.session.import(
        {
          info: info as never,
          messages: records as never,
          ...location(directory),
        },
        within(),
      );
      ledger.created("source", sourceID);
      const state = await read(sourceID);
      return {
        sessionID: sourceID,
        state,
        template: { id: templateID, expected: createdState(template) },
        checks: [
          {
            // The record the run imported is the header the source must
            // have, with no instruction entries.
            name: "source header equals the imported session record",
            ok: isDeepStrictEqual(
              v2Header(state.session, state.instructions ?? []),
              v2Header(info, []),
            ),
          },
          {
            name: "source history equals the imported corpus records",
            ok: isDeepStrictEqual(state.history, json(records)),
          },
          {
            // The corpus defines none, and an import creates none.
            name: "source has no instruction entries",
            ok: state.instructions?.length === 0,
          },
        ],
      };
    },

    async serverPid() {
      return (await host.server.info(within())).pid;
    },

    async read(_directory, sessionID) {
      return read(sessionID);
    },

    async runCommand(
      _directory,
      sessionID,
      name,
      args,
    ): Promise<CommandResult> {
      const started = Date.now();
      try {
        await host.session.command({ sessionID, name, text: args });
        return {
          kind: "returned",
          status: 204,
          message: null,
          ms: Date.now() - started,
        };
      } catch (error) {
        const ms = Date.now() - started;
        if (tag(error) === "CommandExecutionError")
          return {
            kind: "returned",
            status: 500,
            message: String((error as Rec)["message"]),
            ms,
          };
        if ((error as Rec)?.["reason"] === "UnexpectedStatus")
          return {
            kind: "returned",
            status: Number((error as Error).message.split(": ")[1]) || null,
            message: null,
            ms,
          };
        return {
          kind: "lost",
          error: String((error as Error)?.message ?? error).slice(0, 200),
          ms,
        };
      }
    },

    async inspectSuccessor({ source, successorID, corpusCase }) {
      const state = await read(successorID);
      const inspection = v2SuccessorInspection({
        sourceSession: source.session,
        source: source.history,
        successorSession: state.session,
        successor: state.history,
        corpusCase,
      });
      return {
        ...inspection,
        // The successor carries exactly the validated source's entries.
        checks: [
          ...inspection.checks,
          {
            name: "instruction entries equal the source's",
            ok:
              source.instructions !== undefined
              && isDeepStrictEqual(state.instructions, source.instructions),
          },
        ],
        state,
      };
    },

    // Activity, queued messages, permission requests, forms, child sessions
    // and running shells started from the session.
    async pending(directory, sessionID) {
      try {
        await host.session.get({ sessionID }, within());
      } catch (error) {
        if (tag(error) === "SessionNotFoundError") return { exists: false };
        throw error;
      }
      const reasons: string[] = [];
      const active = await host.session.active(within());
      if (sessionID in active) reasons.push("session is active");
      const children = await host.session.list(
        { parentID: sessionID, limit: 10 },
        within(),
      );
      if (((children as unknown as { data?: unknown[] }).data ?? []).length)
        reasons.push("session has child sessions");
      const counts = await Promise.all([
        host.session.inbox.list({ sessionID }, within()),
        host.permission.list({ sessionID }, within()),
        host.session.form.list({ sessionID }, within()),
      ]);
      const [inbox, permissions, forms] = counts.map(
        list => (list as unknown as unknown[]).length,
      );
      if (inbox) reasons.push("a queued message is waiting");
      if (permissions) reasons.push("a permission request is waiting");
      if (forms) reasons.push("a question is waiting");
      const shells = (
        (await host.shell.list(location(directory), within())) as unknown as {
          data: Array<{ status?: unknown; metadata?: Rec }>;
        }
      ).data;
      if (
        shells.some(
          shell =>
            shell.status === "running"
            && shell.metadata?.["sessionID"] === sessionID,
        )
      )
        reasons.push("a shell started from the session is running");
      return { exists: true, reasons };
    },

    async inventory(directory) {
      const ids: string[] = [];
      let cursor: string | undefined;
      for (;;) {
        const page = (await host.session.list(
          { directory, limit: 100, ...(cursor ? { cursor } : {}) },
          within(),
        )) as unknown as {
          data: Rec[];
          cursor?: { next?: string | null };
        };
        ids.push(
          ...page.data
            .filter(
              session =>
                (session["location"] as Rec | undefined)?.["directory"]
                === directory,
            )
            .map(session => String(session["id"])),
        );
        cursor = page.cursor?.next ?? undefined;
        if (page.data.length === 0 || !cursor) return ids;
      }
    },

    async remove(_directory, sessionID) {
      await host.session.remove({ sessionID }, within());
    },

    async dispose(directory) {
      await host.debug.location.evict(location(directory), within());
    },

    // session.generate answers without writing the session's history, so a
    // judge session the run still owns holds exactly what it was created
    // with.
    async judge({ directory, judge, title, prompt, ledger }) {
      const sessionID = mintV2ID("ses_", Date.now(), 1, true);
      ledger.intent("judge", sessionID, title);
      const created = json(
        await host.session.create(
          {
            id: sessionID,
            ...location(directory),
            title,
            agent: JUDGE_AGENT,
            model: {
              providerID: judge.providerID,
              id: judge.modelID,
              ...(judge.variant ? { variant: judge.variant } : {}),
            },
            permissions: DENY_ALL,
          },
          within(),
        ),
      ) as Rec;
      ledger.created("judge", sessionID);
      const expected = createdState(created);
      const started = Date.now();
      try {
        const rawText = (await host.session.generate({ sessionID, prompt }))
          .text;
        return {
          sessionID,
          rawText,
          ms: Date.now() - started,
          usage: null,
          error: null,
          expected,
          ownership: null,
        };
      } catch (error) {
        return {
          sessionID,
          rawText: null,
          ms: Date.now() - started,
          usage: null,
          error: `judge request failed (${tag(error) ?? (error as Rec)?.["reason"] ?? "unknown"})`,
          expected,
          ownership: null,
        };
      }
    },
  };
}

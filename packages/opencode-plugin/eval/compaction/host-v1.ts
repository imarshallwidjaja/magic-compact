// OpenCode 1 driver: public SDK calls bound to an explicit directory.
import { createOpencodeClient } from "@opencode-ai/sdk/v2/client";
import { isDeepStrictEqual } from "node:util";
import { loadHistory, replayHistory } from "../../src/compact/fork";
import { mintV2ID } from "../../src/v2/session";
import { sha256, v1SourceHistory } from "./corpus";
import {
  authHeaders,
  stateOf,
  type CommandResult,
  type EvalHost,
  type FetchLike,
  type ServerAuth,
  type SessionState,
  type Witness,
} from "./host";
import { v1SuccessorInspection, WITNESS_PREFIX } from "./inspect";

const CONTROL_MS = 30_000;
const JUDGE_AGENT = "arkive-eval-judge";
const DENY_ALL = [{ permission: "*", pattern: "*", action: "deny" }] as const;

type Rec = Record<string, unknown>;

// Every session field that can change how the session behaves or what it
// belongs to. Left out: update time, diff summary, cost and tokens.
export function v1Header(session: Rec): unknown {
  const time = (session["time"] ?? {}) as Rec;
  return json({
    id: session["id"],
    slug: session["slug"],
    projectID: session["projectID"],
    workspaceID: session["workspaceID"],
    directory: session["directory"],
    path: session["path"],
    parentID: session["parentID"],
    share: session["share"],
    title: session["title"],
    agent: session["agent"],
    model: session["model"],
    version: session["version"],
    metadata: session["metadata"],
    permission: session["permission"],
    revert: session["revert"],
    created: time["created"],
    compacting: time["compacting"],
    archived: time["archived"],
  });
}

type Result<T> = { data?: T; error?: unknown; response?: Response };

function data<T>(result: Result<T>, what: string): T {
  if (result.error !== undefined || result.data === undefined) {
    throw new Error(`${what} failed (HTTP ${result.response?.status ?? "?"})`);
  }
  return result.data;
}

const json = (value: unknown) => JSON.parse(JSON.stringify(value ?? null));

// `client` lets tests supply an in-memory host.
export function v1Host(
  url: string,
  auth: ServerAuth,
  fetchImpl: FetchLike,
  client: ReturnType<typeof createOpencodeClient> = createOpencodeClient({
    baseUrl: url,
    headers: authHeaders(auth),
    fetch: fetchImpl as typeof fetch,
  }),
): EvalHost {
  const within = (ms = CONTROL_MS) => ({ signal: AbortSignal.timeout(ms) });

  const messages = async (directory: string, sessionID: string) =>
    loadHistory(client, sessionID, { directory }, CONTROL_MS);

  const getSession = async (directory: string, sessionID: string) =>
    data(
      (await client.session.get(
        { sessionID, directory },
        within(),
      )) as Result<Rec>,
      "session read",
    );

  const read = async (
    directory: string,
    sessionID: string,
  ): Promise<SessionState> => {
    const session = json(await getSession(directory, sessionID)) as Rec;
    const history = json(await messages(directory, sessionID)) as Rec[];
    return stateOf(session, v1Header(session), history, session["directory"]);
  };

  const projectConfig = async (directory: string) =>
    data(
      (await client.config.get({ directory }, within())) as Result<Rec>,
      "project config read",
    );
  // The wrapper publishes its witness as a disabled agent's description.
  const witnessOf = (config: Rec): Witness | null => {
    const text = ((config["agent"] ?? {}) as Record<string, Rec>)[
      "arkive-eval-witness"
    ]?.["description"];
    return typeof text === "string" && text.startsWith(WITNESS_PREFIX)
      ? JSON.parse(text.slice(WITNESS_PREFIX.length))
      : null;
  };
  const witnessAt = async (directory: string) =>
    witnessOf(await projectConfig(directory));

  const resolvedAgent = async (directory: string, name: string) => {
    const agents = data(
      (await client.app.agents({ directory }, within())) as Result<Rec[]>,
      "agent list read",
    );
    const agent = agents.find(entry => entry["name"] === name);
    const model = agent?.["model"] as
      { providerID?: string; modelID?: string } | undefined;
    return {
      model: model ? `${model.providerID}/${model.modelID}` : null,
      variant: typeof agent?.["variant"] === "string" ? agent["variant"] : null,
    };
  };

  return {
    kind: "v1",

    async globalMcpNames() {
      const config = data(
        (await client.global.config.get(within())) as Result<{ mcp?: object }>,
        "global config read",
      );
      return Object.keys(config.mcp ?? {}).sort();
    },

    pluginEntry(artifact, options) {
      return [`arkive-compact@file:${artifact.v1Tgz}`, options];
    },

    projectConfig({ pluginEntry, summarizer, mcp }) {
      return {
        file: "opencode.json",
        content: {
          $schema: "https://opencode.ai/config.json",
          plugin: [pluginEntry],
          agent: {
            "magic-compact-summarizer": {
              model: `${summarizer.providerID}/${summarizer.modelID}`,
              ...(summarizer.variant ? { variant: summarizer.variant } : {}),
            },
          },
          ...(mcp.length
            ? {
                mcp: Object.fromEntries(
                  mcp.map(name => [name, { enabled: false }]),
                ),
              }
            : {}),
        },
      };
    },

    judgeConfig({ judge, mcp, system }) {
      return {
        file: "opencode.json",
        content: {
          $schema: "https://opencode.ai/config.json",
          agent: {
            [JUDGE_AGENT]: {
              model: `${judge.providerID}/${judge.modelID}`,
              ...(judge.variant ? { variant: judge.variant } : {}),
              mode: "primary",
              prompt: system,
              permission: { "*": "deny" },
            },
          },
          ...(mcp.length
            ? {
                mcp: Object.fromEntries(
                  mcp.map(name => [name, { enabled: false }]),
                ),
              }
            : {}),
        },
      };
    },

    async setup(directory) {
      const config = await projectConfig(directory);
      const agents = (config["agent"] ?? {}) as Record<string, Rec>;
      const summarizer = agents["magic-compact-summarizer"] ?? {};
      const mcp = (config["mcp"] ?? {}) as Record<
        string,
        { enabled?: boolean }
      >;
      return {
        setup: {
          plugins: ((config["plugin"] ?? []) as unknown[]).map(entry =>
            Array.isArray(entry) ? String(entry[0]) : String(entry),
          ),
          instructionFingerprints: (
            (config["instructions"] ?? []) as unknown[]
          ).map(entry => sha256(String(entry))),
          mcpDisabled: Object.entries(mcp)
            .filter(([, value]) => value?.enabled === false)
            .map(([name]) => name)
            .sort(),
          summarizer: {
            model:
              typeof summarizer["model"] === "string"
                ? summarizer["model"]
                : null,
            variant:
              typeof summarizer["variant"] === "string"
                ? summarizer["variant"]
                : null,
          },
          // The agent list is what the subject itself reads.
          subjectSummarizer: await resolvedAgent(
            directory,
            "magic-compact-summarizer",
          ),
          commands: Object.keys((config["command"] ?? {}) as object).sort(),
        },
        witness: witnessOf(config),
      };
    },

    resolvedAgent,

    async describeModel(directory, model) {
      const catalog = data(
        (await client.config.providers({ directory }, within())) as Result<{
          providers: Array<{
            id: string;
            models: Record<string, Rec>;
          }>;
        }>,
        "provider catalog read",
      );
      const found = catalog.providers.find(p => p.id === model.providerID)
        ?.models[model.modelID];
      if (!found) return null;
      const capabilities = found["capabilities"] as
        { input?: { image?: boolean } } | undefined;
      const limit = (found["limit"] ?? {}) as Record<string, number>;
      return {
        providerID: model.providerID,
        modelID: model.modelID,
        variant: model.variant ?? null,
        name: typeof found["name"] === "string" ? found["name"] : null,
        package:
          typeof (found["api"] as { npm?: unknown } | undefined)?.npm
          === "string"
            ? String((found["api"] as { npm: string }).npm)
            : null,
        variants: Object.keys((found["variants"] ?? {}) as object).sort(),
        imageInput: capabilities?.input?.image === true,
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
      ledger.intent("source", null, title);
      const requested = {
        title,
        agent: "build",
        model: {
          providerID: model.providerID,
          id: model.modelID,
          ...(model.variant ? { variant: model.variant } : {}),
        },
      };
      const created = json(
        data(
          (await client.session.create(
            { directory, ...requested },
            within(),
          )) as Result<Rec>,
          "source session create",
        ),
      ) as Rec;
      const sessionID = String(created["id"]);
      ledger.created("source", sessionID);
      const planned = v1SourceHistory(
        corpus,
        corpusCase,
        sessionID,
        directory,
        model,
      );
      await replayHistory(client, sessionID, planned, { directory });
      const state = await read(directory, sessionID);
      return {
        sessionID,
        state,
        checks: [
          {
            // OpenCode 1.18.30 records the requested model as given, with no
            // variant when none was requested, and nothing the run did not
            // ask for.
            name: "the host created the source with the requested fields",
            ok:
              created["directory"] === directory
              && isDeepStrictEqual(
                json({
                  title: created["title"],
                  agent: created["agent"],
                  model: created["model"],
                  parentID: created["parentID"],
                  metadata: created["metadata"],
                  permission: created["permission"],
                }),
                json(requested),
              ),
          },
          {
            // Replay writes messages only, so the first read still has the
            // header the creation returned.
            name: "source header equals the header the host created",
            ok: isDeepStrictEqual(v1Header(state.session), v1Header(created)),
          },
          {
            name: "source history equals the planned corpus history",
            ok: isDeepStrictEqual(state.history, json(planned)),
          },
        ],
      };
    },

    async serverPid(directory, runNonce) {
      // Only the run's own project location loads its wrapper.
      if (!directory) return null;
      const witness = await witnessAt(directory);
      return witness?.runNonce === runNonce && typeof witness.pid === "number"
        ? witness.pid
        : null;
    },

    read,

    async runCommand(directory, sessionID, name, args): Promise<CommandResult> {
      const started = Date.now();
      try {
        const result = (await client.session.command({
          sessionID,
          directory,
          command: name,
          arguments: args,
        })) as Result<unknown>;
        const error = result.error as
          { data?: { message?: unknown }; message?: unknown } | undefined;
        return {
          kind: "returned",
          status: result.response?.status ?? null,
          message:
            typeof error?.data?.message === "string"
              ? error.data.message
              : typeof error?.message === "string"
                ? error.message
                : null,
          ms: Date.now() - started,
        };
      } catch (error) {
        return {
          kind: "lost",
          error: String((error as Error)?.name ?? error),
          ms: Date.now() - started,
        };
      }
    },

    async inspectSuccessor({ directory, source, successorID, corpusCase }) {
      const state = await read(directory, successorID);
      return {
        ...v1SuccessorInspection({
          sourceSession: source.session,
          source: source.history as never,
          successorSession: state.session,
          successor: state.history as never,
          corpusCase,
        }),
        state,
      };
    },

    // Running, retrying or unknown status, a native compaction, child
    // sessions and permission or question requests waiting for an answer.
    async pending(directory, sessionID) {
      const got = (await client.session.get(
        { sessionID, directory },
        within(),
      )) as Result<Rec>;
      if (got.response?.status === 404) return { exists: false };
      const session = data(got, "session read");
      const status = data(
        (await client.session.status({ directory }, within())) as Result<
          Record<string, { type?: unknown }>
        >,
        "session status read",
      );
      const children = data(
        (await client.session.children(
          { sessionID, directory },
          within(),
        )) as Result<unknown[]>,
        "session children read",
      );
      const waiting = async (
        list: Promise<unknown>,
        what: string,
      ): Promise<boolean> =>
        data((await list) as Result<Array<{ sessionID?: unknown }>>, what).some(
          request => request.sessionID === sessionID,
        );
      const reasons: string[] = [];
      const type = status[sessionID]?.type;
      if (type !== undefined && type !== "idle")
        reasons.push(`session status is ${String(type)}`);
      if ((session["time"] as Rec | undefined)?.["compacting"] !== undefined)
        reasons.push("a native compaction is running");
      if (children.length > 0) reasons.push("session has child sessions");
      if (
        await waiting(
          client.permission.list({ directory }, within()),
          "permission list read",
        )
      )
        reasons.push("a permission request is waiting");
      if (
        await waiting(
          client.question.list({ directory }, within()),
          "question list read",
        )
      )
        reasons.push("a question is waiting");
      return { exists: true, reasons };
    },

    async inventory(directory) {
      const sessions = data(
        (await client.session.list({ directory }, within())) as Result<Rec[]>,
        "session list read",
      );
      return sessions
        .filter(session => session["directory"] === directory)
        .map(session => String(session["id"]));
    },

    async remove(directory, sessionID) {
      data(
        (await client.session.delete(
          { sessionID, directory },
          within(),
        )) as Result<unknown>,
        "session delete",
      );
    },

    async dispose(directory) {
      const path = data(
        (await client.path.get({ directory }, within())) as Result<{
          directory?: string;
        }>,
        "path read",
      );
      if (path.directory !== directory) {
        throw new Error(
          "The server resolved the eval location to another directory.",
        );
      }
      data(
        (await client.instance.dispose(
          { directory },
          within(),
        )) as Result<unknown>,
        "location dispose",
      );
    },

    // A plain-text prompt with every tool denied, under a user message ID the
    // run chose. The session is the run's own when its history is exactly
    // that prompt and the finished reply the host returned for it.
    async judge({ directory, judge, title, prompt, ledger }) {
      ledger.intent("judge", null, title);
      const created = json(
        data(
          (await client.session.create(
            { directory, title, permission: [...DENY_ALL] as never },
            within(),
          )) as Result<Rec>,
          "judge session create",
        ),
      ) as Rec;
      const sessionID = String(created["id"]);
      ledger.created("judge", sessionID);
      const userID = mintV2ID("msg_", Date.now(), 1, false);
      const started = Date.now();
      const unknown = (error: string, ownership: string) => ({
        sessionID,
        rawText: null,
        ms: Date.now() - started,
        usage: null,
        error,
        expected: null,
        ownership,
      });
      let reply: Result<{ info: Rec; parts: Rec[] }>;
      try {
        reply = (await client.session.prompt({
          sessionID,
          directory,
          messageID: userID,
          agent: JUDGE_AGENT,
          model: { providerID: judge.providerID, modelID: judge.modelID },
          ...(judge.variant ? { variant: judge.variant } : {}),
          parts: [{ type: "text", text: prompt }],
        })) as Result<{ info: Rec; parts: Rec[] }>;
      } catch (error) {
        return unknown(
          `judge request outcome unknown (${String((error as Error)?.name ?? error)})`,
          "the judge request's outcome is unknown",
        );
      }
      if (reply.error || !reply.data)
        return unknown(
          `judge request failed (HTTP ${reply.response?.status ?? "?"})`,
          "the judge request failed without a reply to compare",
        );
      const answer = json(reply.data) as { info: Rec; parts: Rec[] };
      const info = answer.info;
      const providerError = info["error"] as { name?: string } | undefined;
      const rawText = answer.parts
        .filter(part => part["type"] === "text" && !part["synthetic"])
        .map(part => String(part["text"]))
        .join("");
      let ownership: string | null = null;
      let expected: { headerSha256: string; historySha256: string } | null =
        null;
      try {
        const state = await read(directory, sessionID);
        const [user, stored, ...rest] = state.history as Array<{
          info: Rec;
          parts: Rec[];
        }>;
        const userParts = user?.parts ?? [];
        const allowed = (key: string, requested: unknown) =>
          isDeepStrictEqual(state.session[key], created[key])
          || isDeepStrictEqual(json(state.session[key]), json(requested));
        const header = (session: Rec) =>
          json({ ...(v1Header(session) as Rec), agent: null, model: null });
        if (
          rest.length > 0
          || user?.info["id"] !== userID
          || user.info["role"] !== "user"
          || userParts.length !== 1
          || userParts[0]!["type"] !== "text"
          || userParts[0]!["text"] !== prompt
          || userParts[0]!["synthetic"]
          || !isDeepStrictEqual(stored, answer)
          || info["parentID"] !== userID
          || (info["time"] as Rec | undefined)?.["completed"] === undefined
        )
          ownership =
            "the judge history is not exactly the run's prompt and the returned reply";
        else if (
          !isDeepStrictEqual(header(state.session), header(created))
          || !allowed("agent", JUDGE_AGENT)
          // The prompt records its model on the session; without a
          // requested variant OpenCode 1.18.30 records "default".
          || !allowed("model", {
            id: judge.modelID,
            providerID: judge.providerID,
            variant: judge.variant ?? "default",
          })
        )
          ownership = "the judge session header changed";
        else
          expected = {
            headerSha256: state.headerSha256,
            historySha256: state.historySha256,
          };
      } catch (error) {
        ownership = `the judge history could not be read (${String((error as Error)?.message ?? error).slice(0, 200)})`;
      }
      return {
        sessionID,
        rawText,
        ms: Date.now() - started,
        usage: json({
          tokens: info["tokens"],
          cost: info["cost"],
          finish: info["finish"],
        }),
        error: providerError
          ? `judge provider error ${providerError.name ?? "unknown"}`
          : null,
        expected,
        ownership,
      };
    },
  };
}

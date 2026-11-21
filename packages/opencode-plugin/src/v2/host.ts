import { ClientError, OpenCode, type OpenCodeClient } from "@opencode/client";
import { Service, type Endpoint } from "@opencode/client/service";
import { readFile } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";
import { isDeepStrictEqual } from "node:util";
import { abortable, attemptWithinDeadline, isRecord } from "../util";
import { ContentFreeError } from "../policy/feedback";

export type V2Host = OpenCodeClient;

export const V2_ENDPOINT_REQUIRED =
  "Magic Compact, Magic Trim and Magic Recover need this OpenCode server's HTTP API to create a successor session, and could not confirm that a reachable endpoint belongs to this server. Nothing was changed. Run OpenCode through its registered background service (the default for the OpenCode 2 TUI and `opencode run`, or `opencode serve --service`), or set the plugin's `serverUrl` option to this server's own listener and export OPENCODE_SERVER_PASSWORD (and OPENCODE_SERVER_USERNAME when it is not `opencode`) in its environment.";

// Why an endpoint did not qualify, in fixed text: never a URL, credential,
// file system path or response body.
const REFUSALS = {
  "not-configured":
    "No `serverUrl` option is set and no registered OpenCode service was found.",
  "invalid-server-url":
    "The `serverUrl` option is not an HTTP or HTTPS URL without embedded credentials, query or fragment.",
  "invalid-registration":
    "The registered OpenCode service's record could not be read or is malformed.",
  "foreign-service":
    "The registered OpenCode service belongs to another OpenCode process.",
  unreachable: "The endpoint could not be reached.",
  "auth-refused": "The endpoint refused the credentials (HTTP 401 or 403).",
  "wrong-server":
    "The endpoint did not identify itself as this OpenCode server.",
  "wrong-session":
    "The endpoint did not return the same source session as this server.",
  "timed-out": "The endpoint did not answer before the control deadline.",
  cancelled: "The plugin was unloaded before the endpoint check finished.",
} as const;

export type V2EndpointRefusal = keyof typeof REFUSALS;

export class V2EndpointError extends ContentFreeError<V2EndpointRefusal> {
  constructor(detail: V2EndpointRefusal) {
    super(`${V2_ENDPOINT_REQUIRED}\nReason: ${REFUSALS[detail]}`, detail);
  }
}

export type V2HostOptions = {
  // The plugin's `serverUrl` option, when the operator configured one.
  serverUrl?: unknown;
  env?: Record<string, string | undefined>;
  pid?: number;
  controlTimeoutMs: number;
  // Stops a read-only caller's check: the plugin's activation, for media
  // reads. Commands pass none.
  signal?: AbortSignal;
};

// The source session as this plugin's own server sees it, read in process.
export type InProcessSessionRead = (
  sessionID: string,
  signal: AbortSignal,
) => Promise<unknown>;

// OpenCode 2 gives plugins no transfer, history or instruction APIs, so the
// commands use the public client over this server's HTTP API. The endpoint is
// the operator's `serverUrl`, or else the registered service of this very
// process. It qualifies only when it reports this process's PID and returns
// the same source session the in-process plugin context reads. A registration
// of another process, or a guessed port, is never contacted.
//
// This is a precheck, not proof of identity: the explicit URL is trusted as
// the operator's, and another process could take over the same URL after the
// check. The credentials are then already sent with the check itself.
//
// One control deadline covers the whole check, the registration read included.
// The caller's signal ends it too, and wins over every other outcome. Each
// read settles when either stops it, even a read that ignores its signal, and
// nothing starts after that.
export async function connectV2Host(
  sourceSessionID: string,
  readInProcess: InProcessSessionRead,
  options: V2HostOptions,
): Promise<V2Host> {
  const env = options.env ?? process.env;
  const pid = options.pid ?? process.pid;
  const activation = options.signal;
  if (activation?.aborted) throw new V2EndpointError("cancelled");
  const outcome = await attemptWithinDeadline(async deadline => {
    const signal = activation
      ? AbortSignal.any([deadline, activation])
      : deadline;
    // OpenCode declares 401 for some routes, so the client reports it either
    // as a declared error or as an unexpected status; the response says which.
    let authRefused = false;
    const read = async <T>(
      refusal: V2EndpointRefusal,
      start: (signal: AbortSignal) => Promise<T>,
    ): Promise<T> => {
      try {
        return await abortable(signal, start);
      } catch (error) {
        if (signal.aborted)
          throw new V2EndpointError(
            activation?.aborted ? "cancelled" : "timed-out",
          );
        if (error instanceof V2EndpointError) throw error;
        throw new V2EndpointError(
          authRefused
            ? "auth-refused"
            : error instanceof ClientError && error.reason === "Transport"
              ? "unreachable"
              : refusal,
        );
      }
    };

    const target =
      options.serverUrl === undefined
        ? await read("invalid-registration", stop =>
            registeredEndpoint(env, pid, stop),
          )
        : { endpoint: explicitEndpoint(options.serverUrl, env) };
    const host = OpenCode.make({
      baseUrl: target.endpoint.url,
      headers: Service.headers(target.endpoint),
      // The credentials go to this endpoint only: a redirect is answered as
      // the unexpected status it is.
      fetch: (async (input, init) => {
        const response = await fetch(input, { ...init, redirect: "manual" });
        if (response.status === 401 || response.status === 403)
          authRefused = true;
        return response;
      }) as typeof fetch,
    });

    // The client does not check a success body against its declared shape.
    const info: unknown = await read("wrong-server", stop =>
      host.server.info({ signal: stop }),
    );
    if (
      !isRecord(info)
      || info["pid"] !== pid
      || typeof info["version"] !== "string"
      || (target.version !== undefined && info["version"] !== target.version)
    )
      throw new V2EndpointError("wrong-server");

    // The in-process read goes first, so a failed or malformed one sends no
    // session request.
    const local = sessionIdentity(
      await read("wrong-session", stop => readInProcess(sourceSessionID, stop)),
    );
    if (local?.id !== sourceSessionID)
      throw new V2EndpointError("wrong-session");
    const remote = await read("wrong-session", stop =>
      host.session.get({ sessionID: sourceSessionID }, { signal: stop }),
    );
    if (!isDeepStrictEqual(local, sessionIdentity(remote)))
      throw new V2EndpointError("wrong-session");
    return host;
  }, options.controlTimeoutMs);
  if (activation?.aborted) throw new V2EndpointError("cancelled");
  if (outcome.kind === "timed-out") throw new V2EndpointError("timed-out");
  if (outcome.kind === "failed") throw outcome.error;
  return outcome.value;
}

function explicitEndpoint(
  serverUrl: unknown,
  env: Record<string, string | undefined>,
): Endpoint {
  const url = typeof serverUrl === "string" ? httpUrl(serverUrl) : undefined;
  if (!url) throw new V2EndpointError("invalid-server-url");
  const password = env["OPENCODE_SERVER_PASSWORD"];
  return {
    url,
    ...(password
      ? {
          auth: {
            type: "basic",
            username: env["OPENCODE_SERVER_USERNAME"] || "opencode",
            password,
          },
        }
      : {}),
  };
}

// The registration OpenCode 2.0.28's `serve --service` writes on the stable,
// dev, beta and next channels (`packages/cli/src/services/service-config.ts`
// `filename`), under the XDG state directory as `@opencode/util`'s
// `global-roots.ts` resolves it. Other channels write another file, which this
// plugin does not read.
async function registeredEndpoint(
  env: Record<string, string | undefined>,
  pid: number,
  signal: AbortSignal,
): Promise<{ endpoint: Endpoint; version?: string }> {
  // `os.homedir()` reads HOME first, but some runtimes read it only once.
  const home = env["HOME"] || homedir();
  const state = env["XDG_STATE_HOME"] || join(home, ".local", "state");
  let text: string;
  try {
    text = await readFile(join(state, "opencode", "service.json"), {
      encoding: "utf8",
      signal,
    });
  } catch (error) {
    const code = isRecord(error) ? error["code"] : undefined;
    throw new V2EndpointError(
      code === "ENOENT" || code === "ENOTDIR"
        ? "not-configured"
        : "invalid-registration",
    );
  }
  let record: Record<string, unknown> | undefined;
  try {
    const parsed: unknown = JSON.parse(text);
    if (isRecord(parsed)) record = parsed;
  } catch {
    // Reported below with every other malformed record.
  }
  // `Service.Info`: `url` and `pid` are required, the rest optional strings.
  const optional = (key: string) =>
    record?.[key] === undefined || typeof record[key] === "string";
  const url =
    typeof record?.["url"] === "string" ? httpUrl(record["url"]) : undefined;
  if (
    !record
    || !url
    || !Number.isInteger(record["pid"])
    || !["id", "version", "password"].every(optional)
  )
    throw new V2EndpointError("invalid-registration");
  if (record["pid"] !== pid) throw new V2EndpointError("foreign-service");
  const password = record["password"] as string | undefined;
  const version = record["version"] as string | undefined;
  return {
    // The username OpenCode's own clients use (`service-probe.ts`).
    endpoint: {
      url,
      ...(password === undefined
        ? {}
        : { auth: { type: "basic", username: "opencode", password } }),
    },
    ...(version === undefined ? {} : { version }),
  };
}

// An HTTP(S) base URL the client can append API paths to: no credentials in
// the authority (an empty `@` included), no query and no fragment. A base path
// is kept; surrounding whitespace and a trailing slash are dropped.
function httpUrl(value: string): string | undefined {
  if (!URL.canParse(value)) return undefined;
  const url = new URL(value);
  if (url.protocol !== "http:" && url.protocol !== "https:") return undefined;
  if (url.username || url.password || /^[^:]+:\/*[^/?#]*@/u.test(value.trim()))
    return undefined;
  if (/[?#]/u.test(url.href)) return undefined;
  return `${url.origin}${url.pathname}`.replace(/\/$/u, "");
}

// Stable identity only: titles, timing and metadata may legitimately change
// between the two reads. A session without well-formed identity matches
// nothing.
function sessionIdentity(session: unknown) {
  if (!isRecord(session)) return undefined;
  const { id, projectID, time, location } = session;
  if (
    typeof id !== "string"
    || typeof projectID !== "string"
    || !isRecord(time)
    || typeof time["created"] !== "number"
    || !isRecord(location)
  )
    return undefined;
  return { id, projectID, created: time["created"], location };
}

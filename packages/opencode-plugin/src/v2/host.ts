import { OpenCode, type OpenCodeClient } from "@opencode/client";
import { Service, type Endpoint } from "@opencode/client/service";
import { isDeepStrictEqual } from "node:util";
import { attemptWithinDeadline } from "../util";

export type V2Host = OpenCodeClient;

export const V2_ENDPOINT_REQUIRED =
  "Magic Compact needs this OpenCode server's HTTP API to create a successor session, and could not prove that a reachable endpoint belongs to this server. Run OpenCode through its registered background service (the default for the OpenCode 2 TUI and `opencode run`, or `opencode serve --service`), or set the plugin's `serverUrl` option to this server's own listener and export OPENCODE_SERVER_PASSWORD (and OPENCODE_SERVER_USERNAME when it is not `opencode`) in its environment.";

export type V2HostOptions = {
  // The plugin's `serverUrl` option, when the operator configured one.
  serverUrl?: unknown;
  env?: Record<string, string | undefined>;
  pid?: number;
  discover?: () => Promise<Endpoint | undefined>;
  fetch?: typeof fetch;
  controlTimeoutMs: number;
};

// The source session as this plugin's own server sees it, read in process.
export type InProcessSessionRead = (sessionID: string) => Promise<unknown>;

type V2SessionIdentity = {
  id: string;
  projectID: string;
  time: { created: number };
  location: unknown;
};

// OpenCode 2 gives plugins no transfer, history or instruction APIs, so the
// commands use the public client over this server's HTTP API. An endpoint
// qualifies only when it reports this process's PID and returns the same
// source session the in-process plugin context reads; a registered service
// that belongs to another process, or a guessed port, never qualifies.
export async function connectV2Host(
  sourceSessionID: string,
  readInProcess: InProcessSessionRead,
  options: V2HostOptions,
): Promise<V2Host> {
  const env = options.env ?? process.env;
  let endpoint: Endpoint | undefined;
  if (options.serverUrl !== undefined) {
    if (
      typeof options.serverUrl !== "string"
      || !URL.canParse(options.serverUrl)
    )
      throw new Error(V2_ENDPOINT_REQUIRED);
    const password = env["OPENCODE_SERVER_PASSWORD"];
    endpoint = {
      url: options.serverUrl.replace(/\/$/u, ""),
      ...(password
        ? {
            auth: {
              type: "basic" as const,
              username: env["OPENCODE_SERVER_USERNAME"] || "opencode",
              password,
            },
          }
        : {}),
    };
  } else {
    const discovered = await attemptWithinDeadline(
      () => (options.discover ?? (() => Service.discover()))(),
      options.controlTimeoutMs,
    );
    endpoint = discovered.kind === "success" ? discovered.value : undefined;
  }
  if (!endpoint) throw new Error(V2_ENDPOINT_REQUIRED);
  const host = OpenCode.make({
    baseUrl: endpoint.url,
    headers: Service.headers(endpoint),
    ...(options.fetch ? { fetch: options.fetch } : {}),
  });
  const qualified = await attemptWithinDeadline(async signal => {
    const info = await host.server.info({ signal });
    if (info.pid !== (options.pid ?? process.pid)) return false;
    const [remote, local] = await Promise.all([
      host.session.get({ sessionID: sourceSessionID }, { signal }),
      readInProcess(sourceSessionID),
    ]);
    // Stable identity only: titles, timing and metadata may legitimately
    // change between the two reads.
    const identity = (session: unknown) => {
      const value = session as Partial<V2SessionIdentity> | undefined;
      return {
        id: value?.id,
        projectID: value?.projectID,
        created: value?.time?.created,
        location: value?.location,
      };
    };
    return (
      remote.id === sourceSessionID
      && isDeepStrictEqual(identity(remote), identity(local))
    );
  }, options.controlTimeoutMs);
  if (qualified.kind !== "success" || !qualified.value)
    throw new Error(V2_ENDPOINT_REQUIRED);
  return host;
}

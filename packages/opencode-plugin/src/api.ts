import type { Config, PluginInput } from "@opencode-ai/plugin";
import {
  createOpencodeClient,
  OpencodeClient,
} from "@opencode-ai/sdk/v2/client";
import { randomUUID } from "node:crypto";
import { attemptWithinDeadline, CONTROL_OPERATION_TIMEOUT_MS } from "./util";

export type V2Client = OpencodeClient;

export function unwrap<T, E>(response: { data?: T; error?: E }): T {
  if ("error" in response && response.error) {
    throw new Error(String(response.error));
  }

  return response.data as T;
}

// A disabled agent entry never reaches provider requests or the task tool's
// subagent list, so it can carry a per-process nonce that only this plugin
// instance's config hook writes. A hidden agent would leak into task
// descriptions.
export const LISTENER_IDENTITY_AGENT = "magic-compact-listener-identity";

export const LISTENER_REQUIRED =
  "Magic Compact needs this OpenCode instance's HTTP listener to create a successor session, and could not prove that the listener at the plugin's server URL belongs to this instance. Start OpenCode with a listener, for example `opencode --port 0`, or run `opencode serve` and connect with `opencode attach`. If the server uses basic auth, set OPENCODE_SERVER_PASSWORD (and OPENCODE_SERVER_USERNAME when it is not `opencode`) in the server environment.";

export type ListenerIdentity = {
  readonly description: string;
  register(config: Config): void;
};

export function createListenerIdentity(): ListenerIdentity {
  const description = `magic-compact-listener:${randomUUID()}`;
  return {
    description,
    register(config) {
      config.agent ??= {};
      config.agent[LISTENER_IDENTITY_AGENT] = { disable: true, description };
    },
  };
}

function basicAuthHeaders(
  env: Record<string, string | undefined>,
): Record<string, string> {
  const password = env["OPENCODE_SERVER_PASSWORD"];
  if (!password) return {};
  const username = env["OPENCODE_SERVER_USERNAME"] || "opencode";
  return {
    Authorization: `Basic ${Buffer.from(`${username}:${password}`).toString("base64")}`,
  };
}

export type ListenerConnectionOptions = {
  env?: Record<string, string | undefined>;
  fetch?: typeof fetch;
  controlTimeoutMs?: number;
};

// OpenCode 1.18.30 serves an in-process client to plugins unless it was started
// with a listener, and its server URL then still names a default port that
// another process may own. Every write goes through the public SDK over the
// listener, so the listener must first return this instance's own nonce.
export async function connectListener(
  input: Pick<PluginInput, "serverUrl" | "directory">,
  identity: ListenerIdentity,
  options: ListenerConnectionOptions = {},
): Promise<V2Client> {
  const client = createOpencodeClient({
    baseUrl: input.serverUrl.href.replace(/\/$/u, ""),
    directory: input.directory,
    headers: basicAuthHeaders(options.env ?? process.env),
    ...(options.fetch ? { fetch: options.fetch } : {}),
  });
  const outcome = await attemptWithinDeadline(
    signal => client.config.get({}, { signal }),
    options.controlTimeoutMs ?? CONTROL_OPERATION_TIMEOUT_MS,
  );
  const config = outcome.kind === "success" ? outcome.value.data : undefined;
  if (
    config?.agent?.[LISTENER_IDENTITY_AGENT]?.description
    !== identity.description
  ) {
    throw new Error(LISTENER_REQUIRED);
  }
  return client;
}

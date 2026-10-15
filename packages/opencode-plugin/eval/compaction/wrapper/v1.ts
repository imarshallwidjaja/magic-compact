// OpenCode 1 entry of the eval wrapper. Its package name is arkive-compact,
// so OpenCode 1's package-name dedupe loads it instead of an inherited
// arkive-compact entry in this project only. See core.ts.
import subject from "arkive-compact-eval-subject";
import {
  COMMANDS,
  CONTROL,
  WITNESS,
  applySummarizerVariant,
  createControl,
  evalOptions,
  observeCommand,
  record,
  subjectStorage,
  witness,
} from "./core";

type Hooks = Record<string, ((...args: never[]) => Promise<unknown>) | unknown>;

export default {
  id: "arkive-compact",
  server: async (
    input: {
      directory: string;
      client: {
        session: {
          get(request: { path: { id: string } }): Promise<{ data?: unknown }>;
        };
      };
    },
    rawOptions?: Record<string, unknown>,
  ) => {
    const options = evalOptions(rawOptions);
    const hooks = (await subject.server(input as never, rawOptions)) as Hooks;
    const description = await witness(
      options,
      "v1",
      "arkive-compact",
      input.directory,
    );
    const control = createControl(
      options,
      input.directory,
      async id => {
        const session = (await input.client.session.get({ path: { id } }))
          .data as
          | {
              directory?: string;
              title?: string;
              metadata?: Record<string, unknown>;
            }
          | undefined;
        return (
          session && {
            directory: session.directory,
            title: session.title,
            metadata: session.metadata,
          }
        );
      },
      () => subjectStorage("v1"),
    );
    const config = hooks["config"] as
      ((cfg: never) => Promise<void>) | undefined;
    const before = hooks["command.execute.before"] as
      ((command: never, output: never) => Promise<void>) | undefined;
    const event = hooks["event"] as
      ((input: never) => Promise<void>) | undefined;
    return {
      ...hooks,
      config: async (cfg: {
        agent?: Record<string, Record<string, unknown>>;
        command?: Record<string, unknown>;
      }) => {
        if (config) await config(cfg as never);
        cfg.agent ??= {};
        applySummarizerVariant(cfg.agent, options);
        cfg.agent[WITNESS] = { disable: true, description };
        cfg.command ??= {};
        cfg.command[CONTROL] = {
          template: "",
          description: `${WITNESS} control`,
        };
      },
      "command.execute.before": async (
        command: { command: string; sessionID: string; arguments: string },
        output: unknown,
      ) => {
        if (command.command === CONTROL) {
          throw new Error(
            await control.prepare(command.sessionID, command.arguments),
          );
        }
        if (!before) return;
        if (!COMMANDS.includes(command.command)) {
          return before(command as never, output as never);
        }
        await observeCommand(options, command.sessionID, command.command, () =>
          before(command as never, output as never),
        );
      },
      event: async (input: {
        event: {
          type: string;
          properties?: { info?: Record<string, unknown> };
        };
      }) => {
        const info = input.event.properties?.info;
        if (input.event.type === "session.deleted" && info)
          await control.deleted(String(info["id"])).catch(() => undefined);
        if (
          input.event.type === "message.updated"
          && info?.["role"] === "assistant"
          && (info["time"] as { completed?: number } | undefined)?.completed
        ) {
          record(options, {
            kind: "usage",
            sessionID: info["sessionID"],
            messageID: info["id"],
            agent: info["agent"],
            providerID: info["providerID"],
            modelID: info["modelID"],
            variant: info["variant"],
            finish: info["finish"],
            tokens: info["tokens"],
            cost: info["cost"],
            error: (info["error"] as { name?: unknown } | undefined)?.name,
          });
        }
        if (event) await event(input as never);
      },
    };
  },
};

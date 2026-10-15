// OpenCode 2 entry of the eval wrapper. OpenCode 2 activates only the first
// plugin with a given ID, so the eval project negates `arkive-compact` and
// loads this entry under its own ID. See core.ts.
import subject from "arkive-compact-eval-subject";
import {
  COMMANDS,
  CONTROL,
  WITNESS,
  createControl,
  evalOptions,
  observeCommand,
  record,
  subjectStorage,
  witness,
} from "./core";

type Command = {
  name: string;
  description?: string;
  execute: (input: {
    sessionID: string;
    prompt: { text: string };
  }) => Promise<unknown>;
};
type Context = {
  options: Record<string, unknown>;
  location: { directory: string };
  command: {
    transform(
      fn: (editor: { add(command: Command): void }) => void,
    ): Promise<void>;
  };
  session: {
    get(input: { sessionID: string }): Promise<Record<string, unknown>>;
  };
  event: {
    subscribe(input: { signal: AbortSignal }): AsyncIterable<{
      type: string;
      location?: { directory?: string };
      data?: Record<string, unknown>;
    }>;
  };
};

export default {
  id: "arkive-compact-eval",
  setup: async (ctx: Context) => {
    const options = evalOptions(ctx.options);
    const directory = ctx.location.directory;
    // The subject registers its commands unchanged; only their execution is
    // observed so the exact result text is recorded.
    const command = new Proxy(ctx.command, {
      get(target, property) {
        if (property !== "transform")
          return Reflect.get(target, property, target);
        return (fn: (editor: { add(command: Command): void }) => void) =>
          target.transform(editor =>
            fn({
              add: (entry: Command) =>
                editor.add(
                  COMMANDS.includes(entry.name)
                    ? {
                        ...entry,
                        execute: input =>
                          observeCommand(
                            options,
                            input.sessionID,
                            entry.name,
                            () => entry.execute(input),
                          ),
                      }
                    : entry,
                ),
            }),
          );
      },
    });
    const context = new Proxy(ctx, {
      get(target, property) {
        return property === "command"
          ? command
          : Reflect.get(target, property, target);
      },
    });
    const cleanup = await subject.setup(context as never);
    const description = await witness(
      options,
      "v2",
      "arkive-compact-eval",
      directory,
    );
    const control = createControl(
      options,
      directory,
      async id => {
        const session = await ctx.session.get({ sessionID: id });
        return {
          directory: (session["location"] as { directory?: string } | undefined)
            ?.directory,
          title:
            typeof session["title"] === "string" ? session["title"] : undefined,
          metadata: session["metadata"] as Record<string, unknown> | undefined,
        };
      },
      () => subjectStorage("v2"),
    );
    await ctx.command.transform(editor => {
      editor.add({
        name: WITNESS,
        description,
        execute: async () => {
          throw new Error(`${WITNESS} is a load witness only`);
        },
      });
      editor.add({
        name: CONTROL,
        description: `${WITNESS} control`,
        execute: async ({ sessionID, prompt }) => {
          throw new Error(await control.prepare(sessionID, prompt.text));
        },
      });
    });
    const events = new AbortController();
    void (async () => {
      for await (const event of ctx.event.subscribe({
        signal: events.signal,
      })) {
        if (event.type === "session.deleted" && event.data) {
          await control
            .deleted(String(event.data["sessionID"]))
            .catch(() => undefined);
          continue;
        }
        if (event.type !== "session.step.ended" || !event.data) continue;
        const sessionID = String(event.data["sessionID"]);
        const session = await ctx.session
          .get({ sessionID })
          .catch(() => undefined);
        const location =
          event.location
          ?? (session?.["location"] as { directory?: string } | undefined);
        if (location?.directory !== directory) continue;
        record(options, {
          kind: "usage",
          sessionID,
          messageID: event.data["assistantMessageID"],
          agent: session?.["agent"],
          finish: event.data["finish"],
          tokens: event.data["tokens"],
          cost: event.data["cost"],
        });
      }
    })().catch(() => undefined);
    return () => {
      events.abort();
      if (typeof cleanup === "function") cleanup();
    };
  },
};

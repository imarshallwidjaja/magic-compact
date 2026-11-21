import type { Plugin } from "@opencode/plugin";
import {
  COMMAND_DESCRIPTIONS,
  type CommandTitle,
  commandFailureMessage,
  COMPACT_COMMAND,
  compactSuccess,
  parseMagicCompactArguments,
  parseMagicRecoverArguments,
  parseMagicStatsArguments,
  parseMagicTrimArguments,
  RECOVER_COMMAND,
  RECOVER_NOOP,
  recoverSuccess,
  STATS_COMMAND,
  TRIM_COMMAND,
  TRIM_NOOP,
  trimSuccess,
} from "./commands";
import {
  MAGIC_COMPACT_SUMMARIZER_AGENT,
  MAGIC_COMPACT_SUMMARIZER_PROMPT,
  READ_OMITTED_CONTENT_DESCRIPTION,
  READ_OMITTED_CONTENT_ID_DESCRIPTION,
} from "./policy/protocol";
import { getCompactionCount } from "./policy/lineage";
import { commandRetentionPolicy } from "./policy/retention";
import { recordCompletionEvent } from "./stats/events";
import { statsSummaryMessage } from "./stats/constants";
import { omissionReadDiagnostic, readOmittedContent } from "./storage/omission";
import { readStats } from "./storage/stats";
import {
  abortable,
  attemptWithinDeadline,
  CONTROL_OPERATION_TIMEOUT_MS,
  isRecord,
} from "./util";
import { executeV2Compact } from "./v2/compact";
import { GENERATION_RESPONSE_SCHEMA } from "./policy/generation";
import { V2_RESPONSE_TOOL } from "./v2/generation";
import { connectV2Host } from "./v2/host";
import {
  projectV2RequestMessages,
  requestCarriesMediaManifest,
  requestImageMessageIDs,
  type V2RequestMessage,
} from "./v2/media";
import { OwnedSessions, V2_SUMMARY_TOOL } from "./v2/owned";
import { projectReadableReasoning } from "./v2/reasoning";
import { SESSION_SUMMARY_RESPONSE_SCHEMA } from "./policy/protocol";
import { readV2History, v2StorageKey, type V2Message } from "./v2/session";
import { executeV2Recover } from "./v2/recover";
import { executeV2Trim } from "./v2/trim";

type Context = Plugin.Context;

const MEDIA_HISTORY_REQUIRED =
  "Magic Compact could not read this session's complete history over the OpenCode server's HTTP API, which it needs to project summarized images in a request that replays history from before the session's latest compaction. The request was not sent.";

const MEDIA_READ_STOPPED =
  "Magic Compact could not project summarized images in this request because the plugin was unloaded. The request was not sent.";

const RESPONSE_TOOL_REJECTED =
  "Magic Compact's response tool is only available in its own per-turn generation requests.";

const SUMMARY_TOOL_REJECTED =
  "Magic Compact's session summary tool is only available in its own structured session summary requests.";

// Runs a command and returns its result message.
// A failure is rethrown as the error OpenCode 2 returns to the client: the
// plugin's own error unchanged when its message is already the feedback,
// otherwise an error carrying the feedback, with the original as its cause.
async function commandResult(
  command: CommandTitle,
  run: () => Promise<string>,
): Promise<string> {
  try {
    return await run();
  } catch (error) {
    const message = commandFailureMessage(command, error);
    throw error instanceof Error && error.message === message
      ? error
      : new Error(message, { cause: error });
  }
}

// OpenCode 2 entry point. Commands throw their result message, as on
// OpenCode 1, so a client that runs the command receives it. `owned` is the
// registry of temporary sessions this process creates; a caller may supply it.
export async function setup(
  ctx: Context,
  owned = new OwnedSessions(),
): Promise<() => void> {
  const controlTimeoutMs = CONTROL_OPERATION_TIMEOUT_MS;
  const connect = (sessionID: string, signal?: AbortSignal) =>
    connectV2Host(
      sessionID,
      (id, readSignal) =>
        ctx.session.get({ sessionID: id }, { signal: readSignal }),
      {
        serverUrl: ctx.options["serverUrl"],
        controlTimeoutMs,
        ...(signal ? { signal } : {}),
      },
    );

  // The operator configures the summarizer's model and variant; the plugin
  // owns only its role, prompt and tool access.
  await ctx.agent.transform(editor => {
    editor.update(MAGIC_COMPACT_SUMMARIZER_AGENT, agent => {
      agent.description = "Continuation summarizer for Magic Compact";
      agent.mode = "subagent";
      agent.hidden = true;
      agent.system = MAGIC_COMPACT_SUMMARIZER_PROMPT;
      agent.permissions = [{ action: "*", resource: "*", effect: "deny" }];
    });
  });

  await ctx.command.transform(editor => {
    editor.add({
      name: STATS_COMMAND,
      description: COMMAND_DESCRIPTIONS[STATS_COMMAND],
      execute: async ({ sessionID, prompt }) => {
        const result = await commandResult("Magic Stats", async () => {
          parseMagicStatsArguments(prompt.text);
          const session = await ctx.session.get({ sessionID });
          const stats = await readStats(v2StorageKey(sessionID));
          return stats
            ? statsSummaryMessage(
                getCompactionCount(session),
                stats,
                session.model?.id ?? null,
              )
            : "No Magic Compact stats recorded for this session. Did you run /magic-compact or /magic-trim yet?";
        });
        throw new Error(result);
      },
    });
    editor.add({
      name: COMPACT_COMMAND,
      description: COMMAND_DESCRIPTIONS[COMPACT_COMMAND],
      execute: async ({ sessionID, prompt }) => {
        const result = await commandResult("Magic Compact", async () => {
          const args = parseMagicCompactArguments(prompt.text);
          // Read per command and reject an invalid threshold before session
          // work.
          const retention = commandRetentionPolicy(
            "compact",
            args,
            process.env,
          );
          const host = await connect(sessionID);
          const { successor, outcome } = await executeV2Compact(
            host,
            owned,
            sessionID,
            args.keepTurns,
            { retention, matchModel: args.matchModel, controlTimeoutMs },
          );
          return successor
            ? `${compactSuccess(successor.session.id)}\n${outcome}`
            : outcome;
        });
        throw new Error(result);
      },
    });
    editor.add({
      name: TRIM_COMMAND,
      description: COMMAND_DESCRIPTIONS[TRIM_COMMAND],
      execute: async ({ sessionID, prompt }) => {
        const result = await commandResult("Magic Trim", async () => {
          const args = parseMagicTrimArguments(prompt.text);
          const retention = commandRetentionPolicy("trim", args, process.env);
          const host = await connect(sessionID);
          const { successor, outcome } = await executeV2Trim(
            host,
            owned,
            sessionID,
            args.keepTurns,
            { retention, matchModel: args.matchModel, controlTimeoutMs },
          );
          return successor
            ? `${trimSuccess(successor.session.id)}\n${outcome}`
            : `${TRIM_NOOP} ${outcome}`;
        });
        throw new Error(result);
      },
    });
    editor.add({
      name: RECOVER_COMMAND,
      description: COMMAND_DESCRIPTIONS[RECOVER_COMMAND],
      execute: async ({ sessionID, prompt }) => {
        const result = await commandResult("Magic Recover", async () => {
          parseMagicRecoverArguments(prompt.text);
          const host = await connect(sessionID);
          const recovered = await executeV2Recover(host, sessionID, {
            controlTimeoutMs,
          });
          return recovered
            ? `${recoverSuccess(recovered.successor.session.id)}\n${recovered.outcome}`
            : RECOVER_NOOP;
        });
        throw new Error(result);
      },
    });
  });

  await ctx.tool.transform(editor => {
    editor.add({
      name: "read_omitted_content",
      description: READ_OMITTED_CONTENT_DESCRIPTION,
      input: {
        type: "object",
        properties: {
          contentId: {
            type: "string",
            description: READ_OMITTED_CONTENT_ID_DESCRIPTION,
          },
        },
        required: ["contentId"],
        additionalProperties: false,
      },
      // A direct tool, as on OpenCode 1, rather than only a code-mode function.
      options: { codemode: false },
      execute: async (input, context) => {
        const contentId = String((input as { contentId: unknown }).contentId);
        try {
          const content = await readOmittedContent(
            v2StorageKey(context.sessionID),
            contentId,
          );
          return {
            content:
              content
              ?? `Omitted content unavailable for Content ID ${contentId}: no authorized current-session cache entry exists.`,
          };
        } catch (error) {
          return { content: omissionReadDiagnostic(contentId, error) };
        }
      },
    });
  });

  // The generation response tool. It only acknowledges, and only inside a
  // per-turn generation session this process owns: OpenCode repairs and
  // strips arguments before it runs, so the persisted raw call input is what
  // the command validates.
  await ctx.tool.transform(editor => {
    editor.add({
      name: V2_RESPONSE_TOOL,
      description:
        "Submit the Magic Compact continuation summary object for the indexed target turn.",
      input: GENERATION_RESPONSE_SCHEMA,
      options: { codemode: false },
      execute: async (_input, context) => {
        if (!owned.allowsResponseTool(context.sessionID)) {
          throw new Error(RESPONSE_TOOL_REJECTED);
        }
        return { content: "Recorded." };
      },
    });
    // The structured whole-session summary's tool, on the DeepSeek route
    // only; like the response tool it only acknowledges, and only in a
    // session this process owns for that purpose.
    editor.add({
      name: V2_SUMMARY_TOOL,
      description:
        "Submit the complete Magic Compact session summary as its summary string.",
      input: SESSION_SUMMARY_RESPONSE_SCHEMA,
      options: { codemode: false },
      execute: async (_input, context) => {
        if (!owned.allowsSummaryTool(context.sessionID)) {
          throw new Error(SUMMARY_TOOL_REJECTED);
        }
        return { content: "Recorded." };
      },
    });
  });

  // The plugin's activation, ended by the cleanup `setup` returns. Media reads
  // are read-only work it owns: none starts after it ends, and the open one
  // settles at once, even a read that ignores its signal, with its later
  // result dropped. Commands already running are not tied to it.
  const activation = new AbortController();
  const mediaRead = async <T>(
    read: (signal: AbortSignal) => Promise<T>,
  ): Promise<T> => {
    try {
      return await abortable(activation.signal, read);
    } catch (error) {
      throw activation.signal.aborted ? new Error(MEDIA_READ_STOPPED) : error;
    }
  };
  // The activation can end after a read settles and before its caller
  // resumes, so the caller checks again before it uses the result.
  const assertActive = () => {
    if (activation.signal.aborted) throw new Error(MEDIA_READ_STOPPED);
  };

  // The complete history of a session, read over this server's own HTTP API
  // on a connection qualified for that session by this read. No page or
  // export starts once `signal` aborts.
  const completeHistory = async (
    sessionID: string,
    signal: AbortSignal,
  ): Promise<V2Message[]> => {
    try {
      const host = await connect(sessionID, signal);
      // A read that ignores its signal still settles, and its deadline timer
      // clears, as soon as `signal` aborts.
      return await readV2History(host, sessionID, async request => {
        signal.throwIfAborted();
        const outcome = await attemptWithinDeadline(
          deadline => abortable(AbortSignal.any([deadline, signal]), request),
          controlTimeoutMs,
        );
        if (outcome.kind !== "success") throw new Error("history read failed");
        return outcome.value;
      });
    } catch {
      throw new Error(MEDIA_HISTORY_REQUIRED);
    }
  };

  // Every model request OpenCode builds from a session's history projects the
  // images a summary manifest covers, owned sessions included. The request
  // messages are paired with the native records by message ID. The host's
  // in-process context starts at the newest compaction, but OpenCode picks
  // the history a request replays by the model's provider route: another
  // model replays records older than a provider checkpoint, and the
  // checkpoint's own route replays its retained copies of user messages. So
  // when a request carries images from a message that context does not hold,
  // the complete history is read instead. A request with neither a manifest
  // nor such images reads nothing; one with only such images is checked only
  // in a session with Magic Compact lineage or one this process owns, the only
  // places a capsule exists.
  const projectMedia = async <M extends V2RequestMessage>(event: {
    sessionID: string;
    messages: M[];
  }) => {
    const carriesManifest = requestCarriesMediaManifest(event.messages);
    const imageIDs = requestImageMessageIDs(event.messages);
    if (!carriesManifest) {
      if (imageIDs.length === 0) return;
      // An owned session copies a source's history under cleared metadata, so
      // only the in-memory registry marks it as one that may replay a
      // retained copy of a summarized message.
      if (!owned.has(event.sessionID)) {
        const session = await mediaRead(signal =>
          ctx.session.get({ sessionID: event.sessionID }, { signal }),
        );
        assertActive();
        if (!isRecord(session.metadata?.["magicCompact"])) return;
      }
    }
    const context: V2Message[] = await mediaRead(signal =>
      ctx.session.context({ sessionID: event.sessionID }, { signal }),
    );
    assertActive();
    const known = new Set(context.map(message => message.id));
    if (imageIDs.every(id => known.has(id))) {
      // A capsule in the context is in the request too.
      if (!carriesManifest) return;
      event.messages = projectV2RequestMessages(event.messages, context);
      return;
    }
    const history = await mediaRead(signal =>
      completeHistory(event.sessionID, signal),
    );
    assertActive();
    event.messages = projectV2RequestMessages(event.messages, history);
  };

  // The response tool is advertised only to the agent loop of a per-turn
  // generation session this process owns, and the session summary tool only
  // to a structured session summary it owns. Every other request kind and
  // session loses both. In an owned session whose request the OpenAI
  // Responses serializer sends, another model's plain reasoning becomes
  // readable text in the request (`projectReadableReasoning`); stored records
  // never change.
  await ctx.session.hook("context", async event => {
    owned.admitRequest(event.sessionID);
    if (!owned.allowsResponseTool(event.sessionID)) {
      delete event.tools[V2_RESPONSE_TOOL];
    }
    if (!owned.allowsSummaryTool(event.sessionID)) {
      delete event.tools[V2_SUMMARY_TOOL];
    }
    await projectMedia(event);
    if (owned.projectsReadableReasoning(event.sessionID)) {
      event.messages = projectReadableReasoning(event.messages);
    }
  });
  // An owned session must never be compacted natively: that would send
  // another model request and replace the history being summarized.
  await ctx.session.hook("compaction", async event => {
    owned.rejectCompaction(event.sessionID);
    delete event.tools[V2_RESPONSE_TOOL];
    delete event.tools[V2_SUMMARY_TOOL];
    await projectMedia(event);
  });
  await ctx.session.hook("generate", async event => {
    delete event.tools[V2_RESPONSE_TOOL];
    delete event.tools[V2_SUMMARY_TOOL];
    await projectMedia(event);
  });
  await ctx.session.hook("title", async event => {
    if (owned.has(event.sessionID)) {
      event.result = "Magic Compact";
      return;
    }
    await projectMedia(event);
  });

  void (async () => {
    for await (const event of ctx.event.subscribe({
      signal: activation.signal,
    })) {
      if (event.type !== "session.step.ended") continue;
      // Invalid stats are left alone by `recordCompletionEvent`. Any other
      // failure loses only this increment, and the subscription keeps
      // counting later responses.
      await recordCompletionEvent(
        v2StorageKey(event.data.sessionID),
        event.data.assistantMessageID,
      ).catch(() => undefined);
    }
  })().catch(() => undefined);
  return () => activation.abort();
}

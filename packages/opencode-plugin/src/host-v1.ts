import type { Plugin } from "@opencode-ai/plugin";
import { tool } from "@opencode-ai/plugin";
import {
  MAGIC_COMPACT_SUMMARIZER_AGENT,
  MAGIC_COMPACT_SUMMARIZER_PROMPT,
  READ_OMITTED_CONTENT_DESCRIPTION,
  READ_OMITTED_CONTENT_ID_DESCRIPTION,
} from "./compact/constants";
import { connectListener, createListenerIdentity } from "./api";
import { rejectNativeCompactionOfGenerationFork } from "./compact/compact";
import {
  COMMAND_DESCRIPTIONS,
  COMPACT_COMMAND,
  COMPACT_NOOP,
  compactSuccess,
  parseMagicCompactArguments,
  parseMagicRecoverArguments,
  parseMagicTrimArguments,
  RECOVER_COMMAND,
  RECOVER_NOOP,
  recoverSuccess,
  STATS_ARGUMENTS_ERROR,
  STATS_COMMAND,
  TRIM_COMMAND,
  TRIM_NOOP,
  trimSuccess,
} from "./commands";
import { executeMagicCompact } from "./magic-compact";
import { executeMagicRecover } from "./magic-recover";
import { STATS_SUCCESS, executeMagicStats } from "./magic-stats";
import { executeMagicTrim } from "./magic-trim";
import { commandRetentionPolicy } from "./policy/retention";
import { rejectNativeCompactionOfSummaryFork } from "./retention/summary";
import { omissionReadDiagnostic, readOmittedContent } from "./storage/omission";
import { handleStatsEvent } from "./stats/events";
import { projectManifestCoveredMedia } from "./compact/media";
import type { MessageWithParts } from "./compact/plan";
import type { SuccessorCommand } from "./policy/successor";
import {
  attemptWithinDeadline,
  CONTROL_OPERATION_TIMEOUT_MS,
  failureToast,
} from "./util";

const SUMMARIZER_PERMISSION = { "*": "deny" } as const;

const server: Plugin = async input => {
  const identity = createListenerIdentity();
  const listener = () => connectListener(input, identity);
  // OpenCode 1 logs a thrown command error and shows the client nothing, so
  // a command that fails before it has a verified listener reports through
  // the plugin input's own client. That client reaches this instance either
  // in process or through its actual listener; it never follows the default
  // `serverUrl` that another process may own. Delivery is best effort and
  // the original error is rethrown.
  const preflight = async <T>(
    command: SuccessorCommand,
    prepare: () => Promise<T>,
  ): Promise<T> => {
    try {
      return await prepare();
    } catch (error) {
      await attemptWithinDeadline(
        signal =>
          input.client.tui.showToast({
            body: failureToast(command, error),
            signal,
          }),
        CONTROL_OPERATION_TIMEOUT_MS,
      );
      throw error;
    }
  };
  return {
    config: async config => {
      identity.register(config);
      config.agent ??= {};
      // The operator configures the summarizer's model and variant; the
      // plugin owns only its role, prompt and tool access.
      config.agent[MAGIC_COMPACT_SUMMARIZER_AGENT] = {
        ...config.agent[MAGIC_COMPACT_SUMMARIZER_AGENT],
        description: "Continuation summarizer for Magic Compact",
        mode: "subagent",
        hidden: true,
        prompt: MAGIC_COMPACT_SUMMARIZER_PROMPT,
        permission: SUMMARIZER_PERMISSION as never,
      };
      config.command ??= {};
      for (const name of [
        COMPACT_COMMAND,
        STATS_COMMAND,
        TRIM_COMMAND,
        RECOVER_COMMAND,
      ] as const) {
        config.command[name] = {
          template: "",
          description: COMMAND_DESCRIPTIONS[name],
        };
      }
    },
    "command.execute.before": async command => {
      if (command.command === STATS_COMMAND) {
        if (command.arguments.trim() !== "") {
          throw new Error(STATS_ARGUMENTS_ERROR);
        }

        await executeMagicStats(await listener(), command.sessionID);
        throw new Error(STATS_SUCCESS);
      }

      if (command.command === COMPACT_COMMAND) {
        const { keepTurns, retention, v2 } = await preflight(
          "Magic Compact",
          async () => {
            const args = parseMagicCompactArguments(command.arguments);
            return {
              keepTurns: args.keepTurns,
              // Read per command and reject an invalid threshold before
              // session work.
              retention: commandRetentionPolicy("compact", args, process.env),
              v2: await listener(),
            };
          },
        );

        // Shows its own progress and terminal toasts.
        const successor = await executeMagicCompact(
          v2,
          command.sessionID,
          keepTurns,
          { retention },
        );
        // Must throw error or OpenCode will send message to LLM
        throw new Error(
          successor ? compactSuccess(successor.session.id) : COMPACT_NOOP,
        );
      }

      if (command.command === TRIM_COMMAND) {
        const { keepTurns, retention, v2 } = await preflight(
          "Magic Trim",
          async () => {
            const args = parseMagicTrimArguments(command.arguments);
            return {
              keepTurns: args.keepTurns,
              retention: commandRetentionPolicy("trim", args, process.env),
              v2: await listener(),
            };
          },
        );
        const successor = await executeMagicTrim(
          v2,
          command.sessionID,
          keepTurns,
          { retention },
        );
        throw new Error(
          successor ? trimSuccess(successor.session.id) : TRIM_NOOP,
        );
      }

      if (command.command === RECOVER_COMMAND) {
        const v2 = await preflight("Magic Recover", async () => {
          parseMagicRecoverArguments(command.arguments);
          return listener();
        });
        const successor = await executeMagicRecover(v2, command.sessionID);
        throw new Error(
          successor ? recoverSuccess(successor.session.id) : RECOVER_NOOP,
        );
      }
    },
    // A Magic-owned summary or generation session must never be compacted
    // natively: that would send another model request and replace the
    // history being summarized.
    "experimental.session.compacting": async input => {
      rejectNativeCompactionOfSummaryFork(input.sessionID);
      rejectNativeCompactionOfGenerationFork(input.sessionID);
    },
    "experimental.chat.messages.transform": async (_input, output) => {
      const projected = projectManifestCoveredMedia(
        output.messages as unknown as MessageWithParts[],
      ) as unknown as typeof output.messages;
      output.messages.splice(0, output.messages.length, ...projected);
    },
    event: async input => {
      await handleStatsEvent(input.event);
    },
    tool: {
      read_omitted_content: tool({
        description: READ_OMITTED_CONTENT_DESCRIPTION,
        args: {
          contentId: tool.schema
            .string()
            .describe(READ_OMITTED_CONTENT_ID_DESCRIPTION),
        },
        async execute(args, context) {
          try {
            const content = await readOmittedContent(
              context.sessionID,
              args.contentId,
            );
            return (
              content
              ?? `Omitted content unavailable for Content ID ${args.contentId}: no authorized current-session cache entry exists.`
            );
          } catch (error) {
            return omissionReadDiagnostic(args.contentId, error);
          }
        },
      }),
    },
  };
};

export default server;

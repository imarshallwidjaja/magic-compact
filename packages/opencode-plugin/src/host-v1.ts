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
  parseMagicTrimArguments,
  STATS_ARGUMENTS_ERROR,
  STATS_COMMAND,
  TRIM_COMMAND,
  TRIM_NOOP,
  trimSuccess,
} from "./commands";
import { executeMagicCompact } from "./magic-compact";
import { STATS_SUCCESS, executeMagicStats } from "./magic-stats";
import { executeMagicTrim } from "./magic-trim";
import { taskAwareRetentionPolicy } from "./policy/retention";
import type { RetentionPolicy } from "./retention/retention";
import { rejectNativeCompactionOfSummaryFork } from "./retention/summary";
import { omissionReadDiagnostic, readOmittedContent } from "./storage/omission";
import { handleStatsEvent } from "./stats/events";
import { projectManifestCoveredMedia } from "./compact/media";
import type { MessageWithParts } from "./compact/plan";

const SUMMARIZER_PERMISSION = { "*": "deny" } as const;

const server: Plugin = async input => {
  const identity = createListenerIdentity();
  const listener = () => connectListener(input, identity);
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
        const keepTurns = parseMagicCompactArguments(command.arguments);
        // Read per command and reject an invalid threshold before session work.
        const retention = taskAwareRetentionPolicy(process.env);

        const successor = await executeMagicCompact(
          await listener(),
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
        const { keepTurns, heuristic } = parseMagicTrimArguments(
          command.arguments,
        );
        const retention: RetentionPolicy = heuristic
          ? { mode: "heuristic" }
          : taskAwareRetentionPolicy(process.env);
        const successor = await executeMagicTrim(
          await listener(),
          command.sessionID,
          keepTurns,
          { retention },
        );
        throw new Error(
          successor ? trimSuccess(successor.session.id) : TRIM_NOOP,
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

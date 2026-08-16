import type { Plugin } from "@opencode-ai/plugin";
import { tool } from "@opencode-ai/plugin";
import {
  MAGIC_COMPACT_SUMMARIZER_AGENT,
  MAGIC_COMPACT_SUMMARIZER_PROMPT,
} from "./compact/constants";
import { getV2Client } from "./api";
import {
  COMPACT_NOOP,
  COMPACT_SUCCESS,
  executeMagicCompact,
} from "./magic-compact";
import { STATS_SUCCESS, executeMagicStats } from "./magic-stats";
import { TRIM_NOOP, TRIM_SUCCESS, executeMagicTrim } from "./magic-trim";
import { omissionReadDiagnostic, readOmittedContent } from "./storage/omission";
import { handleStatsEvent } from "./stats/events";
import { projectManifestCoveredMedia } from "./compact/media";
import type { MessageWithParts } from "./compact/plan";

const COMPACT_COMMAND = "magic-compact";
const TRIM_COMMAND = "magic-trim";
const STATS_COMMAND = "magic-stats";
const SUMMARIZER_PERMISSION = { "*": "deny" } as const;

const server: Plugin = async input => {
  return {
    config: async config => {
      config.agent ??= {};
      config.agent[MAGIC_COMPACT_SUMMARIZER_AGENT] = {
        description: "Per-turn agentic capsule summarizer for Magic Compact",
        mode: "subagent",
        hidden: true,
        prompt: MAGIC_COMPACT_SUMMARIZER_PROMPT,
        permission: SUMMARIZER_PERMISSION as never,
      };
      config.command ??= {};
      config.command[COMPACT_COMMAND] = {
        template: "",
        description: "Lossless context compression",
      };
      config.command[STATS_COMMAND] = {
        template: "",
        description: "Show Magic Compact token and cost savings stats",
      };
      config.command[TRIM_COMMAND] = {
        template: "",
        description: "Trim historical tool inputs and outputs",
      };
    },
    "command.execute.before": async command => {
      if (command.command === STATS_COMMAND) {
        if (command.arguments.trim() !== "") {
          throw new Error("/magic-stats does not accept arguments.");
        }

        await executeMagicStats(getV2Client(input), command.sessionID);
        throw new Error(STATS_SUCCESS);
      }

      if (command.command === COMPACT_COMMAND) {
        const trimmed = command.arguments.trim();
        if (!/^\d*$/.test(trimmed)) {
          throw new Error(
            "/magic-compact argument must be a non-negative integer.",
          );
        }
        const keepTurns = trimmed ? Number(trimmed) : 0;

        const compacted = await executeMagicCompact(
          getV2Client(input),
          command.sessionID,
          keepTurns,
        );
        // Must throw error or OpenCode will send message to LLM
        throw new Error(compacted ? COMPACT_SUCCESS : COMPACT_NOOP);
      }

      if (command.command === TRIM_COMMAND) {
        const trimmed = command.arguments.trim();
        if (!/^\d*$/.test(trimmed)) {
          throw new Error(
            "/magic-trim argument must be a non-negative integer.",
          );
        }
        const keepTurns = trimmed ? Number(trimmed) : 0;
        const didTrim = await executeMagicTrim(
          getV2Client(input),
          command.sessionID,
          keepTurns,
        );
        throw new Error(didTrim ? TRIM_SUCCESS : TRIM_NOOP);
      }
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
        description:
          "Read original tool input or output content omitted by context compaction or trimming operations. Note that this tool reads the snapshot of previously executed tool I/O back in time and contents should be expected to be stale. Only use this tool if you strictly require stale input or output content information from previously completed tool calls AND similar information cannot be obtained via new tool calls.",
        args: {
          contentId: tool.schema
            .string()
            .describe(
              "Omitted content ID. E.g.: 2c4f6a8b0d1e:omitted-yH8kQ2pL5vN7sR4tU6wXzA.",
            ),
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

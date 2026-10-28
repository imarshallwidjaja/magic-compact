// The command surface both hosts register: names, arguments, and the result
// messages their handlers throw or report.
import { hasPluginMessage, PluginMessageError } from "./policy/feedback";
import { percentThreshold, type RetentionArguments } from "./policy/retention";
import {
  isRetentionBackendID,
  type RetentionBackendID,
} from "./retention/backends";
import {
  classifyFailure,
  SuccessorFailure,
  type SuccessorCommand,
} from "./policy/successor";

export const COMPACT_COMMAND = "magic-compact" as const;
export const TRIM_COMMAND = "magic-trim" as const;
export const STATS_COMMAND = "magic-stats" as const;
export const RECOVER_COMMAND = "magic-recover" as const;

export const COMMAND_DESCRIPTIONS = {
  [COMPACT_COMMAND]:
    "Lossless context compression ([N] [--backend jev|openai-decisions] [--threshold P] [--truncate] [--match-model], or [N] --heuristic [--match-model] to skip task-aware retention; turns are still summarized)",
  [TRIM_COMMAND]:
    "Trim historical tool inputs and outputs ([N] [--backend jev|openai-decisions] [--threshold P] [--truncate] [--match-model], or [N] --heuristic to skip task-aware retention)",
  [STATS_COMMAND]: "Show Magic Compact token and cost savings stats",
  [RECOVER_COMMAND]:
    "Continue in a copy that closes abandoned unfinished records as interrupted (stops no work)",
} as const;

export const COMPACT_NOOP = "No assistant turns are old enough to compact.";
export const TRIM_NOOP = "No eligible tool calls need trimming.";
export const RECOVER_NOOP =
  "No unfinished records or undelivered background results need recovery; no successor was created.";
// Fixed per command and never built from a token the person typed. Only
// /magic-compact summarizes under --heuristic, so only it accepts
// --match-model there.
const retentionArgumentsError = (command: string) =>
  command === COMPACT_COMMAND
    ? `/${command} accepts an optional non-negative integer N plus either --heuristic or any of --backend jev|openai-decisions, --threshold P and --truncate, with optional --match-model, each at most once and in any order.`
    : `/${command} accepts an optional non-negative integer N plus either --heuristic or any of --backend jev|openai-decisions, --threshold P, --truncate and --match-model, each at most once and in any order.`;
const retentionThresholdError = (command: string) =>
  `/${command} --threshold needs a plain decimal percent from 0 to 100 after a space, such as --threshold 70 or --threshold 60.5.`;
const retentionBackendError = (command: string) =>
  `/${command} --backend needs jev or openai-decisions after a space, such as --backend openai-decisions.`;
export const COMPACT_ARGUMENTS_ERROR = retentionArgumentsError(COMPACT_COMMAND);
export const COMPACT_THRESHOLD_ERROR = retentionThresholdError(COMPACT_COMMAND);
export const COMPACT_BACKEND_ERROR = retentionBackendError(COMPACT_COMMAND);
export const TRIM_ARGUMENTS_ERROR = retentionArgumentsError(TRIM_COMMAND);
export const TRIM_THRESHOLD_ERROR = retentionThresholdError(TRIM_COMMAND);
export const TRIM_BACKEND_ERROR = retentionBackendError(TRIM_COMMAND);
export const STATS_ARGUMENTS_ERROR = "/magic-stats does not accept arguments.";
export const RECOVER_ARGUMENTS_ERROR =
  "/magic-recover does not accept arguments.";

export function compactSuccess(successorID: string): string {
  return `Magic compaction successful. Continue in successor session ${successorID}; the original session was left unchanged.`;
}

export function trimSuccess(successorID: string): string {
  return `Magic trimming successful. Continue in successor session ${successorID}; the original session was left unchanged.`;
}

export function recoverSuccess(successorID: string): string {
  return `Magic recovery successful. Continue in successor session ${successorID}; the original session was left unchanged.`;
}

export function parseMagicRecoverArguments(raw: string): void {
  if (raw.trim() !== "") throw new PluginMessageError(RECOVER_ARGUMENTS_ERROR);
}

// What a person sees when compact or trim fails: a toast on OpenCode 1, the
// command error on OpenCode 2. An error the plugin marked as its own message
// (`policy/feedback.ts`) shows that message without the class name, and a
// successor failure adds its classified stage, HTTP status and system code.
// Any other error, whatever its type, is described only by its classified
// status and code, never by its own text.
export function commandFailureMessage(
  command: SuccessorCommand,
  error: unknown,
): string {
  if (error instanceof SuccessorFailure) {
    const { stage, status, code } = error.details;
    const details = [`stage ${stage}`];
    if (status !== undefined) details.push(`HTTP ${status}`);
    if (code !== undefined) details.push(code);
    return `${error.message}\nDetails: ${details.join(", ")}.`;
  }
  if (hasPluginMessage(error)) return error.message;
  const { status, code } = classifyFailure(error);
  const details = [
    ...(status === undefined ? [] : [`HTTP ${status}`]),
    ...(code === undefined ? [] : [code]),
  ];
  return `${command} stopped on an unexpected error${details.length > 0 ? ` (${details.join(", ")})` : ""}. The original session was left unchanged. Retry; if it fails again, check the OpenCode server log.`;
}

// `/magic-compact` and `/magic-trim` share one grammar and differ only in the
// fixed error text and in `--match-model` under `--heuristic`: /magic-trim
// `--heuristic` makes no summary request, so the flag would change nothing.
export function parseMagicCompactArguments(raw: string): RetentionArguments {
  return parseRetentionArguments(raw, {
    arguments: COMPACT_ARGUMENTS_ERROR,
    threshold: COMPACT_THRESHOLD_ERROR,
    backend: COMPACT_BACKEND_ERROR,
    summarizesWhenHeuristic: true,
  });
}

export function parseMagicTrimArguments(raw: string): RetentionArguments {
  return parseRetentionArguments(raw, {
    arguments: TRIM_ARGUMENTS_ERROR,
    threshold: TRIM_THRESHOLD_ERROR,
    backend: TRIM_BACKEND_ERROR,
    summarizesWhenHeuristic: false,
  });
}

// Validates every token before any work, so nothing reads the environment
// or contacts the host for a malformed command. Errors never echo a token.
function parseRetentionArguments(
  raw: string,
  errors: {
    arguments: string;
    threshold: string;
    backend: string;
    summarizesWhenHeuristic: boolean;
  },
): RetentionArguments {
  let keepTurns: number | undefined;
  let heuristic = false;
  let truncate = false;
  let matchModel = false;
  let keepThreshold: number | undefined;
  let backend: RetentionBackendID | undefined;
  const tokens = raw.trim().split(/\s+/u).filter(Boolean);
  for (let index = 0; index < tokens.length; index += 1) {
    const token = tokens[index]!;
    if (token === "--heuristic" && !heuristic) {
      heuristic = true;
    } else if (token === "--truncate" && !truncate) {
      truncate = true;
    } else if (token === "--match-model" && !matchModel) {
      matchModel = true;
    } else if (token === "--threshold" && keepThreshold === undefined) {
      index += 1;
      const value = tokens[index];
      keepThreshold = value === undefined ? undefined : percentThreshold(value);
      if (keepThreshold === undefined) {
        throw new PluginMessageError(errors.threshold);
      }
    } else if (token === "--backend" && backend === undefined) {
      index += 1;
      const value = tokens[index];
      if (value === undefined || !isRetentionBackendID(value)) {
        throw new PluginMessageError(errors.backend);
      }
      backend = value;
    } else if (
      /^\d+$/u.test(token)
      && keepTurns === undefined
      && Number.isSafeInteger(Number(token))
    ) {
      keepTurns = Number(token);
    } else {
      throw new PluginMessageError(errors.arguments);
    }
  }
  if (heuristic) {
    if (
      truncate
      || keepThreshold !== undefined
      || backend !== undefined
      || (matchModel && !errors.summarizesWhenHeuristic)
    ) {
      throw new PluginMessageError(errors.arguments);
    }
    return { keepTurns: keepTurns ?? 0, heuristic: true, matchModel };
  }
  return {
    keepTurns: keepTurns ?? 0,
    heuristic: false,
    ...(backend === undefined ? {} : { backend }),
    ...(keepThreshold === undefined ? {} : { keepThreshold }),
    truncate,
    matchModel,
  };
}

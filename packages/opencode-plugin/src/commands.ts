// The command surface both hosts register: names, arguments, and the result
// messages their handlers throw or report.
export const COMPACT_COMMAND = "magic-compact" as const;
export const TRIM_COMMAND = "magic-trim" as const;
export const STATS_COMMAND = "magic-stats" as const;

export const COMMAND_DESCRIPTIONS = {
  [COMPACT_COMMAND]: "Lossless context compression",
  [TRIM_COMMAND]:
    "Trim historical tool inputs and outputs (add --heuristic to skip task-aware retention)",
  [STATS_COMMAND]: "Show Magic Compact token and cost savings stats",
} as const;

export const COMPACT_NOOP = "No assistant turns are old enough to compact.";
export const TRIM_NOOP = "No eligible tool calls need trimming.";
export const TRIM_ARGUMENTS_ERROR =
  "/magic-trim accepts an optional non-negative integer and an optional --heuristic flag.";
export const STATS_ARGUMENTS_ERROR = "/magic-stats does not accept arguments.";

export function compactSuccess(successorID: string): string {
  return `Magic compaction successful. Continue in successor session ${successorID}; the original session was left unchanged.`;
}

export function trimSuccess(successorID: string): string {
  return `Magic trimming successful. Continue in successor session ${successorID}; the original session was left unchanged.`;
}

export function parseMagicCompactArguments(raw: string): number {
  const trimmed = raw.trim();
  if (!/^\d*$/.test(trimmed)) {
    throw new Error("/magic-compact argument must be a non-negative integer.");
  }
  return trimmed ? Number(trimmed) : 0;
}

export function parseMagicTrimArguments(raw: string): {
  keepTurns: number;
  heuristic: boolean;
} {
  let keepTurns: number | undefined;
  let heuristic = false;
  for (const token of raw.trim().split(/\s+/u).filter(Boolean)) {
    if (token === "--heuristic" && !heuristic) {
      heuristic = true;
    } else if (/^\d+$/u.test(token) && keepTurns === undefined) {
      keepTurns = Number(token);
    } else {
      throw new Error(TRIM_ARGUMENTS_ERROR);
    }
  }
  return { keepTurns: keepTurns ?? 0, heuristic };
}

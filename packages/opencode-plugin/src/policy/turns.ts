// Host-neutral request turns. A turn is one or more adjacent user messages
// plus every following assistant message until the next user group; other
// host records never start or end a turn. Assistants before the first user
// message belong to no turn.
export type TurnRole = "user" | "assistant" | "other";

export type PolicyTurn<M> = { user: M[]; assistants: M[] };

export function groupTurns<M>(
  messages: readonly M[],
  role: (message: M) => TurnRole,
): PolicyTurn<M>[] {
  const turns: PolicyTurn<M>[] = [];
  let current: PolicyTurn<M> | null = null;
  for (const message of messages) {
    const kind = role(message);
    if (kind === "user") {
      if (current && current.assistants.length > 0) current = null;
      if (!current) {
        current = { user: [], assistants: [] };
        turns.push(current);
      }
      current.user.push(message);
    } else if (kind === "assistant" && current) {
      current.assistants.push(message);
    }
  }
  return turns;
}

// The turns trim processes: everything except the `keepTurns` most recent
// turns, where a trailing user-only turn does not count.
export function trimmedTurnRange<M>(
  turns: readonly PolicyTurn<M>[],
  keepTurns: number,
): PolicyTurn<M>[] {
  const complete =
    turns.length > 0 && turns[turns.length - 1]!.assistants.length === 0
      ? turns.slice(0, -1)
      : turns.slice();
  return complete.slice(0, Math.max(0, complete.length - keepTurns));
}

// The turns a compaction summarizes, by index into the complete turns after
// any native checkpoint (a trailing user-only turn already removed). It starts
// at the turn holding the latest Magic boundary, ends before the `keepTurns`
// most recent turns, and `next` is the turn the new boundary will lead.
export function compactionTurnRange(
  turnCount: number,
  boundaryTurnIndex: number,
  keepTurns: number,
): { start: number; end: number; preserved: number } {
  const start = boundaryTurnIndex === -1 ? 0 : boundaryTurnIndex;
  const end =
    keepTurns <= 0 ? turnCount : Math.max(start, turnCount - keepTurns);
  return { start, end, preserved: turnCount - end };
}

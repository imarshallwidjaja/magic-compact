// Which errors may show their own message to the person running a command.
//
// A compact or trim failure reaches that person as a toast (OpenCode 1) or the
// command error (OpenCode 2). Only an error this plugin raised for that
// purpose is shown as written: its message is fixed text, never host,
// dependency or session content. Every other error, including a dependency's
// plain `Error`, is described by its classified status and code alone.
// Ownership is positive: an error without the mark is not trusted, whatever
// its type or fields.
const marked = new WeakSet<object>();

export function markPluginMessage<E extends Error>(error: E): E {
  marked.add(error);
  return error;
}

export function hasPluginMessage(error: unknown): error is Error {
  return typeof error === "object" && error !== null && marked.has(error);
}

// A plain error whose message is written for the person running the command.
// It keeps the `Error` name; use it where a typed error would add nothing.
export class PluginMessageError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, options);
    markPluginMessage(this);
  }
}

// How many reasons an aggregate's message explains. A reason is prose for
// understanding the failure and may be dropped past this cap; a reference a
// person needs to recover (see `undeletedSessionIDs`) is never counted in it.
export const MAX_REASONS = 3;

// The distinct messages of the plugin-owned errors inside an aggregate, in
// first-seen order and at most `limit`, nested aggregates flattened. An
// aggregate's own message uses it to say why its parts failed without
// repeating a reason. An error that is not plugin-owned adds nothing, and a
// reason must not name a session: those go through `undeletedSessionIDs`.
export function distinctReasons(
  errors: readonly unknown[],
  limit = MAX_REASONS,
): string[] {
  const reasons = new Set<string>();
  const visit = (error: unknown) => {
    if (error instanceof AggregateError) {
      for (const inner of error.errors) visit(inner);
    } else if (hasPluginMessage(error) && error.message !== "") {
      reasons.add(error.message);
    }
  };
  for (const error of errors) visit(error);
  return [...reasons].slice(0, limit);
}

// A plugin-owned error that created sessions it could not delete. Their IDs
// are data beside the message so that an aggregate can name every one, however
// many reasons it explains.
export interface LeavesSessions {
  readonly undeletedSessionIDs: readonly string[];
}

function leavesSessions(error: unknown): error is Error & LeavesSessions {
  return (
    hasPluginMessage(error)
    && "undeletedSessionIDs" in error
    && Array.isArray(error.undeletedSessionIDs)
  );
}

// Every undeleted session ID the plugin-owned errors report, nested
// aggregates flattened, in first-seen order and never capped.
export function undeletedSessionIDs(errors: readonly unknown[]): string[] {
  const ids = new Set<string>();
  const visit = (error: unknown) => {
    if (leavesSessions(error)) {
      for (const id of error.undeletedSessionIDs) ids.add(id);
    } else if (error instanceof AggregateError) {
      for (const inner of error.errors) visit(inner);
    }
  };
  for (const error of errors) visit(error);
  return [...ids];
}

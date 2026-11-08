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

// An error this plugin raised whose `detail` is one of its own fixed category
// texts, never host content. Only subclasses construct it, each from a closed
// set of its own: `SuccessorVerificationError` in `src/policy/successor.ts`
// and the OpenCode 1 fork failures in `src/compact/fork.ts`.
export abstract class ContentFreeError<
  Detail extends string = string,
> extends PluginMessageError {
  protected constructor(
    message: string,
    readonly detail: Detail,
  ) {
    super(message);
  }
}

// How many reasons an aggregate's message explains. A reason is prose for
// understanding the failure and may be dropped past this cap; a reference a
// person needs to recover (see `unremovedSessions`) is never counted in it.
export const MAX_REASONS = 3;

// The distinct reasons of the plugin-owned errors inside an aggregate, in
// first-seen order and at most `limit`, nested aggregates flattened. An
// aggregate's own message uses it to say why its parts failed without
// repeating a reason. An error that is not plugin-owned adds nothing. An
// error that leaves sessions gives its `reason`, which names none of them.
export function distinctReasons(
  errors: readonly unknown[],
  limit = MAX_REASONS,
): string[] {
  const reasons = new Set<string>();
  const visit = (error: unknown) => {
    if (error instanceof AggregateError) {
      for (const inner of error.errors) visit(inner);
    } else if (hasPluginMessage(error)) {
      const reason = leavesSessions(error) ? error.reason : error.message;
      if (reason !== "") reasons.add(reason);
    }
  };
  for (const error of errors) visit(error);
  return [...reasons].slice(0, limit);
}

// A fork or temporary session the plugin created, or asked the host to
// create, whose removal this run could not confirm. `failed`: the host
// refused the removal or its result is unknown, so the session may still
// exist. `not-found`: the host reported no such session, which does not prove
// that it is gone, because a create the host rejected may still be applied.
export type UnremovedSession = {
  readonly id: string;
  readonly removal: "failed" | "not-found";
};

// A plugin-owned failure that may leave sessions behind. Their IDs are data
// beside the message so that an aggregate can name every one, however many
// reasons it explains; `reason` says what failed without naming them.
export interface LeavesSessions {
  readonly reason: string;
  readonly unremovedSessions: readonly UnremovedSession[];
}

function leavesSessions(error: unknown): error is Error & LeavesSessions {
  return (
    hasPluginMessage(error)
    && "unremovedSessions" in error
    && Array.isArray(error.unremovedSessions)
    && "reason" in error
    && typeof error.reason === "string"
  );
}

// Every unremoved session the plugin-owned errors report, nested aggregates
// flattened, in first-seen order, once each and never capped.
export function unremovedSessions(
  errors: readonly unknown[],
): UnremovedSession[] {
  const sessions = new Map<string, UnremovedSession>();
  const visit = (error: unknown) => {
    if (leavesSessions(error)) {
      for (const session of error.unremovedSessions) {
        if (!sessions.has(session.id)) sessions.set(session.id, session);
      }
    } else if (error instanceof AggregateError) {
      for (const inner of error.errors) visit(inner);
    }
  };
  for (const error of errors) visit(error);
  return [...sessions.values()];
}

const REMOVAL_TEXT = {
  failed: { one: "could not be deleted.", many: "could not be deleted." },
  "not-found": {
    one: "was not found when this run tried to remove it, which does not prove that it was removed.",
    many: "were not found when this run tried to remove them, which does not prove that they were removed.",
  },
} as const;

// Names every session in the given order, one sentence for each run of
// adjacent sessions with the same removal outcome, for example `Temporary
// sessions a, b could not be deleted.`. `noun` is the adapter's capitalized
// name for them.
export function unremovedSessionsText(
  sessions: readonly UnremovedSession[],
  noun: "Fork" | "Temporary session",
): string {
  const runs: Array<{ removal: UnremovedSession["removal"]; ids: string[] }> =
    [];
  for (const { id, removal } of sessions) {
    const last = runs.at(-1);
    if (last?.removal === removal) last.ids.push(id);
    else runs.push({ removal, ids: [id] });
  }
  return runs
    .map(({ removal, ids }) => {
      const many = ids.length > 1;
      return `${noun}${many ? "s" : ""} ${ids.join(", ")} ${REMOVAL_TEXT[removal][many ? "many" : "one"]}`;
    })
    .join(" ");
}

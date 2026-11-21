export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

export function isNonArrayRecord(
  value: unknown,
): value is Record<string, unknown> {
  return isRecord(value) && !Array.isArray(value);
}

// A JSON object literal: no array, class instance or other prototype.
export function isPlainRecord(
  value: unknown,
): value is Record<string, unknown> {
  if (!isNonArrayRecord(value)) return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

export function hasExactKeys(
  value: Record<string, unknown>,
  expected: readonly string[],
): boolean {
  const keys = Object.keys(value).sort();
  const sortedExpected = [...expected].sort();
  return (
    keys.length === sortedExpected.length
    && keys.every((key, index) => key === sortedExpected[index])
  );
}

// Records cross the HTTP API as JSON; comparing their JSON form keeps absent
// and undefined fields equal.
export function jsonForm(value: unknown): unknown {
  return value === undefined ? undefined : JSON.parse(JSON.stringify(value));
}

export function unwrapString(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

export const CONTROL_OPERATION_TIMEOUT_MS = 30_000;

const DEADLINE_REASON = Symbol("magic-compact control deadline");

type DeadlineAttempt<T> =
  | { kind: "success"; value: T }
  | { kind: "timed-out" }
  | { kind: "failed"; error: unknown };

// Starts `start` only while `signal` is live and settles as soon as it aborts,
// with the abort reason, even when the operation ignores its signal. The
// operation's later result or rejection is dropped.
export function abortable<T>(
  signal: AbortSignal,
  start: (signal: AbortSignal) => Promise<T>,
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    signal.throwIfAborted();
    const operation = start(signal);
    const stop = () => reject(signal.reason);
    signal.addEventListener("abort", stop, { once: true });
    if (signal.aborted) stop();
    void operation
      .then(resolve, reject)
      .then(() => signal.removeEventListener("abort", stop));
  });
}

export async function attemptWithinDeadline<T>(
  attempt: (signal: AbortSignal) => Promise<T>,
  deadlineMs: number,
): Promise<DeadlineAttempt<T>> {
  const controller = new AbortController();
  let timer: ReturnType<typeof setTimeout> | undefined;
  const deadline = new Promise<never>((_, reject) => {
    timer = setTimeout(() => {
      controller.abort(DEADLINE_REASON);
      reject(DEADLINE_REASON);
    }, deadlineMs);
  });
  const operation = Promise.resolve().then(() => attempt(controller.signal));
  void operation.catch(() => undefined);
  try {
    const value = await Promise.race([operation, deadline]);
    return { kind: "success", value };
  } catch (error) {
    if (error === DEADLINE_REASON) {
      return { kind: "timed-out" };
    }
    return { kind: "failed", error };
  } finally {
    if (timer !== undefined) clearTimeout(timer);
  }
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
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

export async function attemptWithinDeadline<T>(
  attempt: (signal: AbortSignal) => Promise<T>,
  deadlineMs: number,
  onDeadlineTimerChange: (change: 1 | -1) => void = () => {},
): Promise<DeadlineAttempt<T>> {
  const controller = new AbortController();
  let timer: ReturnType<typeof setTimeout> | undefined;
  const deadline = new Promise<never>((_, reject) => {
    timer = setTimeout(() => {
      controller.abort(DEADLINE_REASON);
      reject(DEADLINE_REASON);
    }, deadlineMs);
    onDeadlineTimerChange(1);
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
    onDeadlineTimerChange(-1);
  }
}

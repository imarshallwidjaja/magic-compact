import { unwrap, type V2Client } from "./api";
import { commandFailureMessage } from "./commands";
import type { SuccessorCommand } from "./policy/successor";

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

export function unwrapString(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

export const CONTROL_OPERATION_TIMEOUT_MS = 30_000;
export const IN_PROGRESS_TOAST_DURATION_MS = 2_147_483_647;

export type MagicToast = {
  title: string;
  message: string;
  variant: "info" | "error";
  duration?: number;
};

// A failure toast stays up for 50 ms per character of the text it shows, at
// least 8 seconds. The text is cut at 700 characters, which bounds the time
// at 35 seconds; the plugin's own messages, even with every reference at its
// cap, stay under that.
const FAILURE_TOAST_MAX_CHARACTERS = 700;

export function failureToast(
  command: SuccessorCommand,
  error: unknown,
): MagicToast {
  const full = commandFailureMessage(command, error);
  const message =
    full.length > FAILURE_TOAST_MAX_CHARACTERS
      ? `${full.slice(0, FAILURE_TOAST_MAX_CHARACTERS - 3)}...`
      : full;
  return {
    title: `${command} Failed`,
    message,
    variant: "error",
    duration: Math.max(8_000, message.length * 50),
  };
}

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

export async function showToastBestEffort(
  v2: V2Client,
  toast: MagicToast,
  deadlineMs: number = CONTROL_OPERATION_TIMEOUT_MS,
): Promise<void> {
  await attemptWithinDeadline(
    signal => v2.tui.showToast(toast, { signal }).then(unwrap),
    deadlineMs,
  );
}

export function createToastFeedback(
  v2: V2Client,
  deadlineMs: number = CONTROL_OPERATION_TIMEOUT_MS,
) {
  let terminalToast: MagicToast | undefined;
  return {
    async showProgress(toast: MagicToast): Promise<void> {
      await attemptWithinDeadline(signal => {
        const delivery = v2.tui.showToast(toast, { signal }).then(unwrap);
        const reassertTerminal = () => {
          if (terminalToast) {
            void showToastBestEffort(v2, terminalToast, deadlineMs);
          }
        };
        void delivery.then(reassertTerminal, reassertTerminal);
        return delivery;
      }, deadlineMs);
    },
    async showTerminal(toast: MagicToast): Promise<void> {
      terminalToast = toast;
      await showToastBestEffort(v2, toast, deadlineMs);
    },
  };
}

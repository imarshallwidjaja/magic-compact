// Command feedback on OpenCode 1, which reports progress and results as TUI
// toasts. A toast that fails or times out never changes the operation.
import { unwrap, type V2Client } from "./api";
import { commandFailureMessage, type CommandTitle } from "./commands";
import { attemptWithinDeadline, CONTROL_OPERATION_TIMEOUT_MS } from "./util";

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
  command: CommandTitle,
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
        // A progress toast delivered after the terminal one would hide it.
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

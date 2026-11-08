import type { Event } from "@opencode-ai/sdk";
import { InvalidStatsError, recordAssistantCompletion } from "../storage/stats";

export async function handleStatsEvent(event: Event): Promise<void> {
  if (event.type !== "message.updated") {
    return;
  }

  const info = event.properties.info;
  if (info.role !== "assistant" || !info.time.completed) {
    return;
  }

  await recordCompletionEvent(info.sessionID, info.id);
}

// The host's completion event, on either host. Invalid saved stats stay as
// they are and this completion is not counted; the commands that need those
// totals report them. Any other failure reaches the caller.
export async function recordCompletionEvent(
  statsKey: string,
  messageID: string,
): Promise<void> {
  try {
    await recordAssistantCompletion(statsKey, messageID);
  } catch (error) {
    if (!(error instanceof InvalidStatsError)) throw error;
  }
}

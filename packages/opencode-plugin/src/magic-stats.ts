import { unwrap, type V2Client } from "./api";
import { STATS_METADATA, statsSummaryMessage } from "./stats/constants";
import { readStats } from "./storage/stats";
import { getCompactionCount } from "./policy/lineage";
import { failureToast, showToastBestEffort } from "./toast-v1";

export const STATS_SUCCESS = "Magic stats displayed.";

// Injects the stats notice. OpenCode 1 shows a thrown command error to no
// client, so a failure is also reported as one error toast.
export async function executeMagicStats(
  v2: V2Client,
  sessionID: string,
): Promise<void> {
  try {
    const session = unwrap(await v2.session.get({ sessionID }));
    const stats = await readStats(sessionID);
    const compactionCount = getCompactionCount(session);
    const text = stats
      ? statsSummaryMessage(compactionCount, stats, session.model?.id ?? null)
      : "No Magic Compact stats recorded for this session. Did you run /magic-compact or /magic-trim yet?";

    unwrap(
      await v2.session.prompt({
        sessionID,
        noReply: true,
        parts: [
          {
            type: "text",
            text,
            ignored: true,
            metadata: STATS_METADATA,
          },
        ],
      }),
    );
  } catch (error) {
    await showToastBestEffort(v2, failureToast("Magic Stats", error));
    throw error;
  }
}

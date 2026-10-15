import type { Plugin } from "@opencode/plugin/tui";
import { isRecord } from "./util";

type Context = Plugin.Context;

// Opens a successor once it is ready, but only in a TUI that is still showing
// its source session, and shows the command's outcome there. A TUI that moved
// elsewhere is left alone.
export function setup(ctx: Context): () => void {
  // Later updates to a successor must not pull a user who went back to the
  // source session away again.
  const opened = new Set<string>();
  return ctx.data.on("session.metadata.updated", event => {
    const { sessionID, metadata } = event.data;
    const lineage = isRecord(metadata) ? metadata["magicCompact"] : undefined;
    if (!isRecord(lineage) || lineage["ready"] !== true) return;
    if (opened.has(sessionID)) return;
    opened.add(sessionID);
    const current = ctx.ui.router.current();
    if (
      current.type !== "session"
      || current.sessionID !== lineage["successorOf"]
    ) {
      return;
    }
    ctx.ui.router.navigate({ type: "session", sessionID });
    if (typeof lineage["outcome"] === "string") {
      ctx.ui.toast.show({ message: lineage["outcome"] });
    }
  });
}

import type { TuiPlugin } from "@opencode-ai/plugin/tui";

// Opens a successor once it is ready, but only in a TUI that is still showing
// its source session; a TUI that moved elsewhere is left alone.
export const tui: TuiPlugin = async api => {
  // Later updates to a successor must not pull a user who went back to the
  // source session away again.
  const opened = new Set<string>();
  const stop = api.event.on("session.updated", event => {
    const info = event.properties.info;
    const lineage = info.metadata?.["magicCompact"];
    if (
      typeof lineage !== "object"
      || lineage === null
      || (lineage as Record<string, unknown>)["ready"] !== true
    ) {
      return;
    }
    if (opened.has(info.id)) return;
    opened.add(info.id);
    const sourceID = (lineage as Record<string, unknown>)["successorOf"];
    const current = api.route.current;
    if (current.name !== "session" || current.params?.sessionID !== sourceID)
      return;
    api.route.navigate("session", { sessionID: info.id });
  });
  api.lifecycle.onDispose(stop);
};

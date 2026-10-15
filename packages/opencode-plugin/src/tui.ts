import type { TuiPluginApi, TuiPluginMeta } from "@opencode-ai/plugin/tui";
import type { PluginOptions } from "@opencode-ai/plugin";
import type { Plugin as V2TuiPlugin } from "@opencode/plugin/tui";

// The TUI companion for both OpenCode hosts: OpenCode 1 reads `tui`,
// OpenCode 2 reads `setup`, and each imports only its own host module.
const plugin = {
  id: "magic-compact",
  tui: async (
    api: TuiPluginApi,
    options: PluginOptions | undefined,
    meta: TuiPluginMeta,
  ) => (await import("./tui-v1")).tui(api, options, meta),
  setup: async (context: V2TuiPlugin.Context) =>
    (await import("./tui-v2")).setup(context),
};

export default plugin;

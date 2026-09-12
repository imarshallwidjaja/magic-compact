import type { TuiPluginModule } from "@opencode-ai/plugin/tui";
import type { Plugin } from "@opencode/plugin/tui";

const plugin: TuiPluginModule & Plugin.Definition = {
  id: "magic-compact",
  tui: async () => {},
  setup: async () => {},
};

export default plugin;

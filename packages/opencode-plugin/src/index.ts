import type { PluginInput, PluginOptions } from "@opencode-ai/plugin";
import type { Plugin as V2Plugin } from "@opencode/plugin";

// One package for both OpenCode hosts. OpenCode 1 (1.18.30 and later) reads
// `server`; OpenCode 2 reads `setup`. Each imports its host module only when
// called, so neither host loads the other's plugin API.
const plugin = {
  id: "arkive-compact",
  server: async (input: PluginInput, options?: PluginOptions) =>
    (await import("./host-v1")).default(input, options),
  setup: async (context: V2Plugin.Context) =>
    (await import("./host-v2")).setup(context),
};

export default plugin;

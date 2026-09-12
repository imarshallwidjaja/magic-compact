import type { Plugin as V1Plugin } from "@opencode-ai/plugin";
import { Plugin } from "@opencode/plugin";

export default {
  ...Plugin.define({
    id: "arkive-compact",
    async setup(ctx) {
      const { setup } = await import("./v2");
      return setup(ctx);
    },
  }),
  server: (async input => {
    const { default: server } = await import("./v1");
    return server(input);
  }) satisfies V1Plugin,
};

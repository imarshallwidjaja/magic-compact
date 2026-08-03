import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Config, PluginInput } from "@opencode-ai/plugin";
import type {
  Agent,
  Message,
  Part,
  Session,
  TextPart,
} from "@opencode-ai/sdk/v2";
import type { V2Client } from "../src/api";
import { compactSession } from "../src/compact/compact";
import type { MessageWithParts } from "../src/compact/plan";
import server from "../src/index";
import { readOmittedContent, writeCache } from "../src/storage/omission";

let storageDirectory: string;
const originalDataHome = process.env.XDG_DATA_HOME;

beforeAll(async () => {
  storageDirectory = await mkdtemp(join(tmpdir(), "magic-compact-test-"));
  process.env.XDG_DATA_HOME = storageDirectory;
});

afterAll(async () => {
  if (originalDataHome === undefined) {
    delete process.env.XDG_DATA_HOME;
  } else {
    process.env.XDG_DATA_HOME = originalDataHome;
  }
  await rm(storageDirectory, { recursive: true, force: true });
});

describe("magic compact", () => {
  test("registers the hidden Magic Compact summarizer with its exact permission policy", async () => {
    const hooks = await server({} as PluginInput);
    const config = {} as Config;

    await hooks.config?.(config);

    expect(config.agent?.["magic-compact-summarizer"]).toMatchObject({
      description: "Faithful per-turn XML summarizer for Magic Compact",
      mode: "subagent",
      hidden: true,
    });
    expect(config.agent?.["magic-compact-summarizer"]?.permission).toEqual({
      "*": "allow",
      edit: "deny",
      task: "deny",
      question: "deny",
      todowrite: "deny",
      plan_enter: "deny",
      plan_exit: "deny",
      doom_loop: "deny",
    });
    expect(config.agent?.["magic-compact-summarizer"]?.prompt).toContain(
      "Historical transcript evidence is primary",
    );
    expect(config.agent?.["magic-compact-summarizer"]?.prompt).toContain(
      "Never continue unfinished work",
    );
    expect(config.agent?.["magic-compact-summarizer"]?.prompt).toContain(
      "Follow the Magic Compact XML request exactly",
    );
  });

  test("uses the native compaction model and variant with the custom agent", async () => {
    const requests = await runCompaction(
      {
        ...session("source"),
        agent: "build",
        model: {
          providerID: "provider",
          id: "model",
          variant: "fast",
        },
        permission: [{ permission: "read", pattern: "*", action: "allow" }],
      },
      {
        agents: [
          compactionAgent({
            model: {
              providerID: "compaction-provider",
              modelID: "compaction-model",
            },
            variant: "precise",
          }),
        ],
      },
    );

    expect(requests.agentRequests).toEqual([
      { directory: "/workspace", workspace: "workspace" },
    ]);
    expect(requests.updates).toEqual([
      {
        sessionID: "ephemeral",
        title: "[TEMP] Test session",
      },
    ]);
    expect(requests.prompts[0]).toMatchObject({
      sessionID: "ephemeral",
      agent: "magic-compact-summarizer",
      model: {
        providerID: "compaction-provider",
        modelID: "compaction-model",
      },
      variant: "precise",
    });
  });

  test("falls back to the captured source model when compaction has no model", async () => {
    const requests = await runCompaction({
      ...session("source"),
      model: {
        providerID: "source-provider",
        id: "source-model",
        variant: "fast",
      },
    });

    expect(requests.prompts[0]).toMatchObject({
      agent: "magic-compact-summarizer",
      model: {
        providerID: "source-provider",
        modelID: "source-model",
      },
      variant: "fast",
    });
  });

  test("fails clearly and deletes the temporary session when compaction is unavailable", async () => {
    const lifecycle: string[] = [];

    await expect(
      runCompaction(session("source"), { agents: [], lifecycle }),
    ).rejects.toThrow("native hidden compaction agent");

    expect(lifecycle).toEqual(["fork", "update", "agents", "delete"]);
  });

  test("rejects invalid explicit compaction model configuration", async () => {
    await expect(
      runCompaction(session("source"), {
        agents: [
          compactionAgent({
            model: { providerID: "", modelID: "compaction-model" },
          }),
        ],
      }),
    ).rejects.toThrow("invalid model configuration");
  });

  test("maps omission reads to the source only while summarization is active", async () => {
    await writeCache("source", {
      version: 1,
      nextId: 2,
      entries: { "omitted-001": { content: "historical output" } },
    });
    let contentDuringPrompt: string | null = null;

    await runCompaction(session("source"), {
      onPrompt: async () => {
        contentDuringPrompt = await readOmittedContent(
          "ephemeral",
          "omitted-001",
        );
      },
    });

    expect(contentDuringPrompt).toBe("historical output");
    expect(await readOmittedContent("ephemeral", "omitted-001")).toBeNull();
  });

  test("clears the omission source mapping when summarization fails", async () => {
    await writeCache("source", {
      version: 1,
      nextId: 2,
      entries: { "omitted-001": { content: "historical output" } },
    });
    let contentDuringPrompt: string | null = null;

    await expect(
      runCompaction(session("source"), {
        onPrompt: async () => {
          contentDuringPrompt = await readOmittedContent(
            "ephemeral",
            "omitted-001",
          );
        },
        promptError: new Error("summary failed"),
      }),
    ).rejects.toThrow("summary failed");

    expect(contentDuringPrompt).toBe("historical output");
    expect(await readOmittedContent("ephemeral", "omitted-001")).toBeNull();
  });

  test("preserves the XML prompt and parser behavior", async () => {
    const requests = await runCompaction(session("source"));
    const prompt = requests.prompts[0]?.parts as Array<{ text: string }>;
    const promptText = prompt[0]?.text ?? "";

    expect(promptText).toContain(
      "<summary>\n<user>\nRequest\n...\n</user>\n<assistant>",
    );
    expect(promptText).not.toContain("Do not call any tools");
    expect(promptText).toContain(
      "Tools may be used only when needed to recover missing historical evidence or clarify it through current inspection",
    );
    expect(promptText).toContain("Must not continue unfinished work");
    expect(promptText).toContain(
      "After any tool use, final output remains only the required XML",
    );
    expect(requests.partUpdates[0]).toMatchObject({
      part: {
        text: "Completed the request.",
        metadata: { magicCompact: { summary: true } },
      },
    });
  });
});

type RunOptions = {
  agents?: Agent[];
  lifecycle?: string[];
  onPrompt?: () => Promise<void>;
  promptError?: Error;
};

async function runCompaction(
  sourceSession: Session,
  options: RunOptions = {},
): Promise<{
  prompts: Record<string, unknown>[];
  updates: Record<string, unknown>[];
  agentRequests: Record<string, unknown>[];
  partUpdates: Record<string, unknown>[];
}> {
  const prompts: Record<string, unknown>[] = [];
  const updates: Record<string, unknown>[] = [];
  const agentRequests: Record<string, unknown>[] = [];
  const partUpdates: Record<string, unknown>[] = [];
  const lifecycle = options.lifecycle ?? [];
  const v2 = {
    app: {
      agents: async (request: Record<string, unknown>) => {
        lifecycle.push("agents");
        agentRequests.push(request);
        return { data: options.agents ?? [compactionAgent()] };
      },
    },
    session: {
      messages: async () => ({ data: compactableMessages() }),
      fork: async () => {
        lifecycle.push("fork");
        return { data: session("ephemeral") };
      },
      update: async (request: Record<string, unknown>) => {
        lifecycle.push("update");
        updates.push(request);
        return { data: session("ephemeral") };
      },
      prompt: async (request: Record<string, unknown>) => {
        lifecycle.push("prompt");
        prompts.push(request);
        await options.onPrompt?.();
        if (options.promptError) {
          throw options.promptError;
        }
        return {
          data: {
            parts: [
              textPart(
                "response",
                "ephemeral",
                "response",
                "<summary><user>Request</user><assistant>Completed the request.</assistant></summary>",
              ),
            ],
          },
        };
      },
      delete: async () => {
        lifecycle.push("delete");
        return { data: true };
      },
    },
    part: {
      update: async (request: Record<string, unknown>) => {
        partUpdates.push(request);
        return { data: request.part as TextPart };
      },
    },
  } as unknown as V2Client;

  await compactSession(v2, sourceSession, "source", 0);
  return { prompts, updates, agentRequests, partUpdates };
}

function compactableMessages(): MessageWithParts[] {
  return [
    message("user", "user", [
      textPart("user-text", "source", "user", "Request"),
    ]),
    message("assistant", "assistant", []),
  ];
}

function message(
  id: string,
  role: "user" | "assistant",
  parts: Part[],
): MessageWithParts {
  return {
    info: { id, role } as Message,
    parts,
  };
}

function session(id: string): Session {
  return {
    id,
    title: "Test session",
    directory: "/workspace",
    workspaceID: "workspace",
  } as Session;
}

function compactionAgent(overrides: Partial<Agent> = {}): Agent {
  return {
    name: "compaction",
    mode: "primary",
    native: true,
    hidden: true,
    permission: [],
    options: {},
    ...overrides,
  };
}

function textPart(
  id: string,
  sessionID: string,
  messageID: string,
  text: string,
): TextPart {
  return {
    id,
    sessionID,
    messageID,
    type: "text",
    text,
  };
}

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
import {
  generateCompactionSummaries,
  injectSummaries,
} from "../src/compact/compact";
import {
  assertNativeCheckpointUnchanged,
  createCompactionPlan,
  NativeCheckpointChangedError,
  type MessageWithParts,
  type Turn,
} from "../src/compact/plan";
import { pruneSummarizedTurns } from "../src/compact/prune";
import { applyBackup } from "../src/compact/session";
import { executeMagicCompact } from "../src/magic-compact";
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

    expect(lifecycle).toEqual([
      "messages",
      "fork",
      "update",
      "agents",
      "delete",
    ]);
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

describe("native OpenCode checkpoint coexistence", () => {
  test("accepts a valid checkpoint without a tail anchor", async () => {
    const plan = await planFor(
      nativeCheckpointFixture({ tailStartID: null }),
      0,
    );

    expect(assistantIDs(plan.summarizedTurns)).toEqual([
      "suffix-1-assistant",
      "suffix-2-assistant",
    ]);
  });

  test("uses the latest completed checkpoint and ignores an older incomplete marker in its frozen prefix", async () => {
    const messages = nativeCheckpointFixture({ tailStartID: null });
    messages.splice(
      2,
      0,
      nativeMarker("older-incomplete", null),
      ...completedTurn("between", "Between checkpoints"),
    );
    const plan = await planFor(messages, 0);

    expect(assistantIDs(plan.summarizedTurns)).toEqual([
      "suffix-1-assistant",
      "suffix-2-assistant",
    ]);
  });

  test.each([
    [
      "an incomplete marker inside a completed interval",
      [
        nativeMarker("marker-a", null),
        nativeMarker("marker-b", null),
        nativeSummary("summary-a", "marker-a"),
      ],
    ],
    [
      "crossed completed intervals",
      [
        nativeMarker("marker-a", null),
        nativeMarker("marker-b", null),
        nativeSummary("summary-a", "marker-a"),
        nativeSummary("summary-b", "marker-b"),
      ],
    ],
    [
      "nested completed intervals",
      [
        nativeMarker("marker-a", null),
        nativeMarker("marker-b", null),
        nativeSummary("summary-b", "marker-b"),
        nativeSummary("summary-a", "marker-a"),
      ],
    ],
  ])("rejects %s", async (_name, artifacts) => {
    await expect(
      planFor([...artifacts, ...completedTurn("after", "After")], 0),
    ).rejects.toThrow("overlap");
  });

  test("accepts sequential completed checkpoint intervals", async () => {
    const plan = await planFor(
      [
        nativeMarker("marker-a", null),
        nativeSummary("summary-a", "marker-a"),
        ...completedTurn("between", "Between checkpoints"),
        nativeMarker("marker-b", null),
        nativeSummary("summary-b", "marker-b"),
        ...completedTurn("after", "After checkpoints"),
      ],
      0,
    );

    expect(assistantIDs(plan.summarizedTurns)).toEqual(["after-assistant"]);
  });

  test("rejects an older incomplete artifact that extends past a later checkpoint marker", async () => {
    await expect(
      planFor(
        [
          nativeMarker("marker-a", null),
          nativeMarker("marker-b", null),
          nativeSummary("summary-b", "marker-b"),
          nativeSummary("summary-a", "marker-a", { finish: null }),
          ...completedTurn("after", "After checkpoints"),
        ],
        0,
      ),
    ).rejects.toThrow("incomplete");
  });

  test("rejects a malformed tail on an older completed checkpoint", async () => {
    const messages = nativeCheckpointFixture({ tailStartID: null });
    messages.splice(
      2,
      0,
      nativeMarker("older-marker", "missing-tail"),
      nativeSummary("older-summary", "older-marker"),
      ...completedTurn("between", "Between checkpoints"),
    );

    await expect(planFor(messages, 0)).rejects.toThrow("tail_start_id");
  });

  test("ignores Magic boundaries before the native checkpoint and honors boundaries after it", async () => {
    const messages = nativeCheckpointFixture({ tailStartID: null });
    messages[0]!.parts.push(magicBoundary("old-boundary", "tail-user"));
    messages.splice(
      messages.length - 2,
      0,
      ...completedTurn("boundary", "Boundary turn", true),
    );
    const plan = await planFor(messages, 0);

    expect(assistantIDs(plan.summarizedTurns)).toEqual([
      "boundary-assistant",
      "suffix-2-assistant",
    ]);
  });

  test("does not fork, prompt, or mutate when every eligible suffix turn is protected", async () => {
    const calls: string[] = [];
    const v2 = {
      session: {
        messages: async () => ({
          data: nativeCheckpointFixture({ tailStartID: null }),
        }),
        fork: async () => {
          calls.push("fork");
          return { data: session("backup") };
        },
        prompt: async () => {
          calls.push("prompt");
          return { data: {} };
        },
      },
      part: {
        update: async () => {
          calls.push("update");
          return { data: {} };
        },
      },
      tui: {
        showToast: async () => {
          calls.push("toast");
          return { data: true };
        },
      },
    } as unknown as V2Client;

    expect(await executeMagicCompact(v2, "source", 2)).toBe(false);
    expect(calls).toEqual(["toast"]);
  });

  test("revalidates before writes and completes by pruning only the checkpoint suffix", async () => {
    const messages = nativeCheckpointFixture({ tailStartID: "tail-assistant" });
    messages
      .find(item => item.info.id === "suffix-1-assistant")!
      .parts.push(
        textPart(
          "suffix-1-old-text",
          "source",
          "suffix-1-assistant",
          "Replace this response",
        ),
      );
    addProviderTokens(messages);
    const command = createCommandHarness(messages);

    expect(await executeMagicCompact(command.v2, "source", 1)).toBe(true);
    expect(
      command.partUpdates.map(request => ({
        messageID: request.messageID,
        partID: request.partID,
      })),
    ).toEqual([
      {
        messageID: "suffix-1-assistant",
        partID: "prt_-magic_summary_suffix-1-assistant",
      },
      {
        messageID: "suffix-2-user",
        partID: "prt_-magic_boundary_suffix-2-user",
      },
    ]);
    expect(command.partDeletes).toEqual([
      {
        sessionID: "source",
        messageID: "suffix-1-assistant",
        partID: "suffix-1-old-text",
      },
    ]);
    expect(command.messageDeletes).toEqual([
      { sessionID: "source", messageID: "progress" },
    ]);
    expect(command.sessionDeletes).toEqual(["ephemeral"]);
    expect(command.selectedSessions).toEqual([]);

    const progressPromptIndex = command.events.indexOf("progress-prompt");
    const summaryPromptIndex = command.events.indexOf("summary-prompt");
    const summaryUpdateIndex = command.events.indexOf("summary-update");
    // plan + provider-token load + pre-progress checkpoint revalidation
    const preProgressMessageLoads = command.events
      .slice(0, progressPromptIndex)
      .filter(event => event === "messages").length;
    const initialRevalidationIndex = command.events.lastIndexOf(
      "messages",
      progressPromptIndex - 1,
    );
    const postSummaryRevalidationIndex = command.events.indexOf(
      "messages",
      summaryPromptIndex + 1,
    );
    expect(progressPromptIndex).toBeGreaterThanOrEqual(0);
    expect(summaryPromptIndex).toBeGreaterThan(progressPromptIndex);
    expect(preProgressMessageLoads).toBe(3);
    expect(initialRevalidationIndex).toBe(progressPromptIndex - 1);
    expect(command.events[initialRevalidationIndex]).toBe("messages");
    expect(postSummaryRevalidationIndex).toBeGreaterThan(summaryPromptIndex);
    expect(postSummaryRevalidationIndex).toBeLessThan(summaryUpdateIndex);
    expect(command.toasts.at(-1)).toMatchObject({
      title: "Magic Compact",
      message: "Compacted 1 assistant turn(s).",
    });
  });

  test("a future completed native checkpoint advances the immutable floor", async () => {
    const messages = nativeCheckpointFixture({ tailStartID: null });
    messages.push(
      nativeMarker("future-marker", "suffix-2-assistant"),
      nativeSummary("future-summary", "future-marker"),
      ...completedTurn("future-suffix", "After future checkpoint"),
    );
    const plan = await planFor(messages, 0);

    expect(assistantIDs(plan.summarizedTurns)).toEqual([
      "future-suffix-assistant",
    ]);
  });

  test("detects new and changed native artifacts before pruning", async () => {
    const messages = nativeCheckpointFixture({ tailStartID: null });
    const plan = await planFor(messages, 0);
    const completedRace = [
      ...messages,
      nativeMarker("raced-marker", "suffix-2-assistant"),
      nativeSummary("raced-summary", "raced-marker"),
    ];
    const changedMarkerPart = structuredClone(messages);
    Object.assign(
      changedMarkerPart.find(item => item.info.id === "native-marker")!
        .parts[0]!,
      { auto: false },
    );
    const changedSummaryInfo = structuredClone(messages);
    Object.assign(
      changedSummaryInfo.find(item => item.info.id === "native-summary")!.info,
      { finish: "length" },
    );
    const changedSummaryPart = structuredClone(messages);
    changedSummaryPart.find(item => item.info.id === "native-summary")!.parts =
      [
        textPart(
          "native-summary-text",
          "source",
          "native-summary",
          "Changed native summary",
        ),
      ];

    for (const changed of [
      completedRace,
      changedMarkerPart,
      changedSummaryInfo,
      changedSummaryPart,
    ]) {
      expect(() => assertNativeCheckpointUnchanged(changed, plan)).toThrow(
        "changed",
      );
    }
  });

  test("does not report a checkpoint race for object key insertion order", async () => {
    const messages = nativeCheckpointFixture({ tailStartID: null });
    const plan = await planFor(messages, 0);
    const reordered = messages.map(item => ({
      info: Object.fromEntries(Object.entries(item.info).reverse()) as Message,
      parts: item.parts.map(
        part =>
          Object.fromEntries(Object.entries(part).reverse()) as unknown as Part,
      ),
    }));

    expect(() =>
      assertNativeCheckpointUnchanged(reordered, plan),
    ).not.toThrow();
  });

  test("aborts before source writes and preserves both sessions when a checkpoint races the summary prompt", async () => {
    const messages = nativeCheckpointFixture({ tailStartID: null });
    addProviderTokens(messages);
    const raced = [
      ...messages,
      nativeMarker("raced-marker", "suffix-2-assistant"),
    ];
    const command = createCommandHarness(messages, { racedMessages: raced });

    await expect(
      executeMagicCompact(command.v2, "source", 1),
    ).rejects.toMatchObject({ name: "NativeCheckpointChangedError" });
    expect(command.partUpdates).toEqual([]);
    expect(command.partDeletes).toEqual([]);
    expect(command.messageDeletes).toEqual([
      { sessionID: "source", messageID: "progress" },
    ]);
    expect(command.sessionDeletes).toEqual(["ephemeral"]);
    expect(command.selectedSessions).toEqual([]);
    expect(
      command.sessionUpdates.filter(
        request =>
          request.sessionID === "backup" && request.title === "Test session",
      ),
    ).toEqual([]);
    expect(command.sourcePrompts).toHaveLength(1);
    expect(command.sourcePrompts[0]).toMatchObject({
      noReply: true,
      parts: [
        {
          metadata: { magicCompact: { progress: true } },
        },
      ],
    });
    expect(command.toasts.at(-1)).toMatchObject({
      title: "Magic Compact Failed",
      variant: "error",
    });
  });

  test("retains an incomplete-marker race and progress-delete failure while preserving both sessions", async () => {
    const messages = nativeCheckpointFixture({ tailStartID: null });
    addProviderTokens(messages);
    const raced = [
      ...messages,
      nativeMarker("raced-marker", "suffix-2-assistant"),
    ];
    const cleanupError = new Error("progress cleanup failed");
    const command = createCommandHarness(messages, {
      racedMessages: raced,
      deleteProgressError: cleanupError,
    });

    let error: unknown;
    try {
      await executeMagicCompact(command.v2, "source", 1);
    } catch (caught) {
      error = caught;
    }

    expect(error).toBeInstanceOf(NativeCheckpointChangedError);
    expect(error).not.toBe(cleanupError);
    expect((error as Error).cause).toBeInstanceOf(AggregateError);
    expect(nestedErrorMessages((error as Error).cause)).toEqual(
      expect.arrayContaining([
        "Native compaction marker raced-marker is incomplete or errored.",
        "progress cleanup failed",
      ]),
    );
    expect(command.partUpdates).toEqual([]);
    expect(command.partDeletes).toEqual([]);
    expect(command.messageDeletes).toEqual([
      { sessionID: "source", messageID: "progress" },
    ]);
    expect(command.sessionDeletes).toEqual(["ephemeral"]);
    expect(command.selectedSessions).toEqual([]);
    expect(
      command.sessionUpdates.filter(
        request =>
          request.sessionID === "backup" && request.title === "Test session",
      ),
    ).toEqual([]);
    expect(command.toasts.at(-1)).toMatchObject({
      title: "Magic Compact Failed",
      variant: "error",
    });
  });

  test("retains an incomplete-marker race and ephemeral-delete failure while preserving both sessions", async () => {
    const messages = nativeCheckpointFixture({ tailStartID: null });
    addProviderTokens(messages);
    const raced = [
      ...messages,
      nativeMarker("raced-marker", "suffix-2-assistant"),
    ];
    const ephemeralDeleteError = new Error("ephemeral delete failed");
    const command = createCommandHarness(messages, {
      racedMessages: raced,
      deleteEphemeralError: ephemeralDeleteError,
    });

    let error: unknown;
    try {
      await executeMagicCompact(command.v2, "source", 1);
    } catch (caught) {
      error = caught;
    }

    expect(error).toBeInstanceOf(NativeCheckpointChangedError);
    expect(error).not.toBe(ephemeralDeleteError);
    expect((error as Error).cause).toBeInstanceOf(AggregateError);
    expect(nestedErrorMessages((error as Error).cause)).toEqual(
      expect.arrayContaining([
        "Native compaction marker raced-marker is incomplete or errored.",
        "ephemeral delete failed",
      ]),
    );
    expect(command.partUpdates).toEqual([]);
    expect(command.partDeletes).toEqual([]);
    expect(command.messageDeletes).toEqual([
      { sessionID: "source", messageID: "progress" },
    ]);
    expect(command.sessionDeletes).toEqual(["ephemeral"]);
    expect(command.selectedSessions).toEqual([]);
    expect(
      command.sessionUpdates.filter(
        request =>
          request.sessionID === "backup" && request.title === "Test session",
      ),
    ).toEqual([]);
    expect(
      command.sessionDeletes.filter(sessionID => sessionID === "source"),
    ).toEqual([]);
    expect(command.toasts.at(-1)).toMatchObject({
      title: "Magic Compact Failed",
      variant: "error",
    });
  });

  test("treats a revalidation messages-fetch SDK error after generation as a checkpoint change", async () => {
    const messages = nativeCheckpointFixture({ tailStartID: null });
    addProviderTokens(messages);
    const fetchError = new Error("revalidation messages fetch failed");
    const command = createCommandHarness(messages, {
      revalidationMessagesError: fetchError,
    });

    let error: unknown;
    try {
      await executeMagicCompact(command.v2, "source", 1);
    } catch (caught) {
      error = caught;
    }

    expect(error).toBeInstanceOf(NativeCheckpointChangedError);
    expect((error as Error).cause).toMatchObject({
      message: "revalidation messages fetch failed",
    });
    expect(command.partUpdates).toEqual([]);
    expect(command.partDeletes).toEqual([]);
    expect(command.messageDeletes).toEqual([
      { sessionID: "source", messageID: "progress" },
    ]);
    expect(command.sessionDeletes).toEqual(["ephemeral"]);
    expect(command.selectedSessions).toEqual([]);
    expect(
      command.sessionUpdates.filter(
        request =>
          request.sessionID === "backup" && request.title === "Test session",
      ),
    ).toEqual([]);
  });

  test("retains generation and revalidation messages-fetch SDK errors while preserving both sessions", async () => {
    const messages = nativeCheckpointFixture({ tailStartID: null });
    addProviderTokens(messages);
    const generationError = new Error("summary prompt failed");
    const fetchError = new Error("revalidation messages fetch failed");
    const command = createCommandHarness(messages, {
      summaryPromptError: generationError,
      revalidationMessagesError: fetchError,
    });

    let error: unknown;
    try {
      await executeMagicCompact(command.v2, "source", 1);
    } catch (caught) {
      error = caught;
    }

    expect(error).toBeInstanceOf(NativeCheckpointChangedError);
    expect((error as Error).cause).toBeInstanceOf(AggregateError);
    expect(nestedErrorMessages((error as Error).cause)).toEqual(
      expect.arrayContaining([
        "summary prompt failed",
        "revalidation messages fetch failed",
      ]),
    );
    expect(command.partUpdates).toEqual([]);
    expect(command.partDeletes).toEqual([]);
    expect(command.messageDeletes).toEqual([
      { sessionID: "source", messageID: "progress" },
    ]);
    expect(command.sessionDeletes).toEqual(["ephemeral"]);
    expect(command.selectedSessions).toEqual([]);
    expect(
      command.sessionUpdates.filter(
        request =>
          request.sessionID === "backup" && request.title === "Test session",
      ),
    ).toEqual([]);
  });

  test("retains prompt and progress-delete failures without a race and safely rolls back", async () => {
    const messages = nativeCheckpointFixture({ tailStartID: null });
    addProviderTokens(messages);
    const generationError = new Error("summary prompt failed");
    const cleanupError = new Error("progress cleanup failed");
    const command = createCommandHarness(messages, {
      summaryPromptError: generationError,
      deleteProgressError: cleanupError,
    });

    let error: unknown;
    try {
      await executeMagicCompact(command.v2, "source", 1);
    } catch (caught) {
      error = caught;
    }

    expect(error).toBeInstanceOf(AggregateError);
    expect(error).not.toBeInstanceOf(NativeCheckpointChangedError);
    expect(nestedErrorMessages(error)).toEqual(
      expect.arrayContaining([
        "summary prompt failed",
        "progress cleanup failed",
      ]),
    );
    expect(command.partUpdates).toEqual([]);
    expect(command.partDeletes).toEqual([]);
    expect(command.messageDeletes).toEqual([
      { sessionID: "source", messageID: "progress" },
    ]);
    expect(command.sessionDeletes).toEqual(["ephemeral", "source"]);
    expect(command.selectedSessions).toEqual(["backup"]);
    expect(
      command.sessionUpdates.filter(
        request =>
          request.sessionID === "backup" && request.title === "Test session",
      ),
    ).toEqual([
      {
        sessionID: "backup",
        title: "Test session",
      },
    ]);
  });

  test("retains prompt and ephemeral-delete failures without a race and safely rolls back", async () => {
    const messages = nativeCheckpointFixture({ tailStartID: null });
    addProviderTokens(messages);
    const generationError = new Error("summary prompt failed");
    const ephemeralDeleteError = new Error("ephemeral delete failed");
    const command = createCommandHarness(messages, {
      summaryPromptError: generationError,
      deleteEphemeralError: ephemeralDeleteError,
    });

    let error: unknown;
    try {
      await executeMagicCompact(command.v2, "source", 1);
    } catch (caught) {
      error = caught;
    }

    expect(error).toBeInstanceOf(AggregateError);
    expect(error).not.toBeInstanceOf(NativeCheckpointChangedError);
    expect(nestedErrorMessages(error)).toEqual(
      expect.arrayContaining([
        "summary prompt failed",
        "ephemeral delete failed",
      ]),
    );
    expect(command.partUpdates).toEqual([]);
    expect(command.partDeletes).toEqual([]);
    expect(command.messageDeletes).toEqual([
      { sessionID: "source", messageID: "progress" },
    ]);
    expect(command.sessionDeletes).toEqual(["ephemeral", "source"]);
    expect(command.selectedSessions).toEqual(["backup"]);
    expect(
      command.sessionUpdates.filter(
        request =>
          request.sessionID === "backup" && request.title === "Test session",
      ),
    ).toEqual([
      {
        sessionID: "backup",
        title: "Test session",
      },
    ]);
  });

  test("retains prompt and ephemeral-delete failures with a concurrent checkpoint race while preserving both sessions", async () => {
    const messages = nativeCheckpointFixture({ tailStartID: null });
    addProviderTokens(messages);
    const raced = [
      ...messages,
      nativeMarker("raced-marker", "suffix-2-assistant"),
    ];
    const generationError = new Error("summary prompt failed");
    const ephemeralDeleteError = new Error("ephemeral delete failed");
    const command = createCommandHarness(messages, {
      racedMessages: raced,
      summaryPromptError: generationError,
      deleteEphemeralError: ephemeralDeleteError,
    });

    let error: unknown;
    try {
      await executeMagicCompact(command.v2, "source", 1);
    } catch (caught) {
      error = caught;
    }

    expect(error).toBeInstanceOf(NativeCheckpointChangedError);
    expect(error).not.toBe(generationError);
    expect(error).not.toBe(ephemeralDeleteError);
    expect((error as Error).cause).toBeInstanceOf(AggregateError);
    expect(nestedErrorMessages((error as Error).cause)).toEqual(
      expect.arrayContaining([
        "Native compaction marker raced-marker is incomplete or errored.",
        "summary prompt failed",
        "ephemeral delete failed",
      ]),
    );
    expect(command.partUpdates).toEqual([]);
    expect(command.partDeletes).toEqual([]);
    expect(command.messageDeletes).toEqual([
      { sessionID: "source", messageID: "progress" },
    ]);
    expect(command.sessionDeletes).toEqual(["ephemeral"]);
    expect(command.selectedSessions).toEqual([]);
    expect(
      command.sessionUpdates.filter(
        request =>
          request.sessionID === "backup" && request.title === "Test session",
      ),
    ).toEqual([]);
    expect(
      command.sessionDeletes.filter(sessionID => sessionID === "source"),
    ).toEqual([]);
    expect(command.toasts.at(-1)).toMatchObject({
      title: "Magic Compact Failed",
      variant: "error",
    });
  });

  test("progress cleanup-only failure after successful summaries follows ordinary backup rollback", async () => {
    const messages = nativeCheckpointFixture({ tailStartID: null });
    addProviderTokens(messages);
    const cleanupError = new Error("progress cleanup failed");
    const command = createCommandHarness(messages, {
      deleteProgressError: cleanupError,
    });

    await expect(executeMagicCompact(command.v2, "source", 1)).rejects.toBe(
      cleanupError,
    );
    expect(
      command.partUpdates.map(request => ({
        messageID: request.messageID,
        partID: request.partID,
      })),
    ).toEqual([
      {
        messageID: "suffix-1-assistant",
        partID: "prt_-magic_summary_suffix-1-assistant",
      },
    ]);
    expect(command.partDeletes).toEqual([]);
    expect(command.messageDeletes).toEqual([
      { sessionID: "source", messageID: "progress" },
    ]);
    expect(command.sessionDeletes).toEqual(["ephemeral", "source"]);
    expect(command.selectedSessions).toEqual(["backup"]);
    expect(
      command.sessionUpdates.filter(
        request =>
          request.sessionID === "backup" && request.title === "Test session",
      ),
    ).toEqual([
      {
        sessionID: "backup",
        title: "Test session",
      },
    ]);
    expect(command.toasts.at(-1)).toMatchObject({
      title: "Magic Compact Failed",
      variant: "error",
    });
  });

  test.each([
    ["dangling tail anchor", { tailStartID: "missing" }, "tail_start_id"],
    [
      "forward tail anchor",
      { tailStartID: "suffix-1-assistant" },
      "tail_start_id",
    ],
    [
      "duplicate marker parts",
      { duplicateParts: true },
      "multiple compaction parts",
    ],
    ["unfinished summary", { finish: null }, "incomplete"],
    ["errored summary", { error: { name: "AbortError" } }, "incomplete"],
  ])("rejects %s before mutation", async (_name, fixtureOptions, error) => {
    await expect(
      planFor(nativeCheckpointFixture(fixtureOptions), 0),
    ).rejects.toThrow(error);
  });

  test("rejects a marker without a completed summary", async () => {
    await expect(
      planFor(
        [nativeMarker("marker", null), ...completedTurn("after", "After")],
        0,
      ),
    ).rejects.toThrow("incomplete");
  });

  test("rejects a compaction part attached to an assistant message", async () => {
    await expect(
      planFor(
        [
          assistantCompaction("assistant-marker"),
          ...completedTurn("after", "After marker"),
        ],
        0,
      ),
    ).rejects.toThrow("user message");
  });

  test("rejects multiple successful summaries for one marker", async () => {
    const messages = nativeCheckpointFixture({ tailStartID: null });
    messages.splice(
      messages.findIndex(item => item.info.id === "native-summary") + 1,
      0,
      nativeSummary("native-summary-duplicate", "native-marker"),
    );

    await expect(planFor(messages, 0)).rejects.toThrow(
      "multiple completed summaries",
    );
  });

  test("rejects an orphan native summary", async () => {
    await expect(
      planFor(
        [
          nativeSummary("orphan", "missing"),
          ...completedTurn("after", "After"),
        ],
        0,
      ),
    ).rejects.toThrow("orphan");
  });

  test.each([
    [
      "a user compaction part",
      {
        user: [nativeMarker("native-marker", null)],
        assistants: [message("assistant", "assistant", [])],
      },
    ],
    [
      "a native summary assistant",
      {
        user: [message("user", "user", [])],
        assistants: [nativeSummary("native-summary", "native-marker")],
      },
    ],
  ])(
    "prune preflight rejects %s before the first mutation",
    async (_name, turn) => {
      const deletes: unknown[] = [];
      const v2 = {
        part: {
          delete: async (request: unknown) => {
            deletes.push(request);
            return { data: true };
          },
        },
        session: {
          deleteMessage: async (request: unknown) => {
            deletes.push(request);
            return { data: true };
          },
        },
      } as unknown as V2Client;

      await expect(
        pruneSummarizedTurns({ v2, sessionID: "source" }, [turn]),
      ).rejects.toThrow("native compaction artifact");
      expect(deletes).toEqual([]);
    },
  );
});

describe("backup recovery", () => {
  test("selects the backup before deleting the source", async () => {
    const operations: string[] = [];
    const v2 = {
      session: {
        update: async () => {
          operations.push("update");
          return { data: session("backup") };
        },
        delete: async () => {
          operations.push("delete");
          return { data: true };
        },
      },
      tui: {
        selectSession: async () => {
          operations.push("select");
          return { data: true };
        },
      },
    } as unknown as V2Client;

    await applyBackup(v2, session("source"), session("backup"));

    expect(operations).toEqual(["update", "select", "delete"]);
  });

  test("leaves the source intact when selecting the backup fails", async () => {
    const operations: string[] = [];
    const v2 = {
      session: {
        update: async () => {
          operations.push("update");
          return { data: session("backup") };
        },
        delete: async () => {
          operations.push("delete");
          return { data: true };
        },
      },
      tui: {
        selectSession: async () => {
          operations.push("select");
          return { error: "select failed" };
        },
      },
    } as unknown as V2Client;

    await expect(
      applyBackup(v2, session("source"), session("backup")),
    ).rejects.toThrow("select failed");
    expect(operations).toEqual(["update", "select"]);
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
  const sourceMessages = compactableMessages();
  const v2 = {
    app: {
      agents: async (request: Record<string, unknown>) => {
        lifecycle.push("agents");
        agentRequests.push(request);
        return { data: options.agents ?? [compactionAgent()] };
      },
    },
    session: {
      messages: async () => {
        lifecycle.push("messages");
        return { data: sourceMessages };
      },
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
        const part = request.part as Part;
        const target = sourceMessages.find(
          message => message.info.id === request.messageID,
        );
        if (target) {
          const index = target.parts.findIndex(item => item.id === part.id);
          if (index === -1) {
            target.parts.push(part);
          } else {
            target.parts[index] = part;
          }
        }
        return { data: part };
      },
    },
  } as unknown as V2Client;

  const plan = await createCompactionPlan(v2, "source", 0);
  const summaries = await generateCompactionSummaries(v2, sourceSession, plan);
  await injectSummaries(v2, "source", plan.summarizedTurns, summaries);
  return {
    prompts,
    updates,
    agentRequests,
    partUpdates,
  };
}

function createCommandHarness(
  messages: MessageWithParts[],
  options: {
    racedMessages?: MessageWithParts[];
    deleteProgressError?: Error;
    deleteEphemeralError?: Error;
    summaryPromptError?: Error;
    revalidationMessagesError?: Error;
  } = {},
) {
  const events: string[] = [];
  const partUpdates: Record<string, unknown>[] = [];
  const partDeletes: Record<string, unknown>[] = [];
  const messageDeletes: Record<string, unknown>[] = [];
  const sessionUpdates: Record<string, unknown>[] = [];
  const sessionDeletes: string[] = [];
  const selectedSessions: string[] = [];
  const sourcePrompts: Record<string, unknown>[] = [];
  const toasts: Record<string, unknown>[] = [];
  let forks = 0;
  let raceActive = false;

  const v2 = {
    app: { agents: async () => ({ data: [compactionAgent()] }) },
    session: {
      messages: async () => {
        events.push("messages");
        if (raceActive && options.revalidationMessagesError) {
          return { error: options.revalidationMessagesError.message };
        }
        return {
          data:
            raceActive && options.racedMessages
              ? options.racedMessages
              : messages,
        };
      },
      get: async () => ({ data: session("source") }),
      fork: async () => ({
        data: session(++forks === 1 ? "backup" : "ephemeral"),
      }),
      update: async (request: Record<string, unknown>) => {
        sessionUpdates.push(request);
        return { data: session(String(request.sessionID)) };
      },
      prompt: async (request: Record<string, unknown>) => {
        if (request.sessionID === "ephemeral") {
          events.push("summary-prompt");
          raceActive = true;
          if (options.summaryPromptError) {
            throw options.summaryPromptError;
          }
          return {
            data: {
              parts: [
                textPart(
                  "response",
                  "ephemeral",
                  "response",
                  "<summary><user>First suffix request</user><assistant>Suffix summary.</assistant></summary>",
                ),
              ],
            },
          };
        }

        if (request.noReply === true) {
          const parts = request.parts as
            | Array<{ metadata?: { magicCompact?: { progress?: boolean } } }>
            | undefined;
          if (parts?.some(part => part.metadata?.magicCompact?.progress)) {
            events.push("progress-prompt");
          }
        }

        sourcePrompts.push(request);
        return {
          data: {
            info: { id: "progress" },
            parts: [],
          },
        };
      },
      deleteMessage: async (request: Record<string, unknown>) => {
        messageDeletes.push(request);
        if (options.deleteProgressError) {
          throw options.deleteProgressError;
        }
        return { data: true };
      },
      delete: async (request: { sessionID: string }) => {
        sessionDeletes.push(request.sessionID);
        if (options.deleteEphemeralError && request.sessionID === "ephemeral") {
          throw options.deleteEphemeralError;
        }
        return { data: true };
      },
    },
    part: {
      update: async (request: Record<string, unknown>) => {
        partUpdates.push(request);
        const part = request.part as Part;
        const metadata = (part as TextPart).metadata as
          | { magicCompact?: { summary?: boolean; boundary?: boolean } }
          | undefined;
        if (metadata?.magicCompact?.summary) {
          events.push("summary-update");
        } else if (metadata?.magicCompact?.boundary) {
          events.push("boundary-update");
        }

        const target = messages.find(
          message => message.info.id === request.messageID,
        );
        if (target) {
          const index = target.parts.findIndex(item => item.id === part.id);
          if (index === -1) {
            target.parts.push(part);
          } else {
            target.parts[index] = part;
          }
        }
        return { data: part };
      },
      delete: async (request: Record<string, unknown>) => {
        partDeletes.push(request);
        const target = messages.find(
          message => message.info.id === request.messageID,
        );
        if (target) {
          target.parts = target.parts.filter(
            part => part.id !== request.partID,
          );
        }
        return { data: true };
      },
    },
    tui: {
      showToast: async (request: Record<string, unknown>) => {
        toasts.push(request);
        return { data: true };
      },
      selectSession: async (request: { sessionID: string }) => {
        selectedSessions.push(request.sessionID);
        return { data: true };
      },
    },
  } as unknown as V2Client;

  return {
    v2,
    events,
    partUpdates,
    partDeletes,
    messageDeletes,
    sessionUpdates,
    sessionDeletes,
    selectedSessions,
    sourcePrompts,
    toasts,
  };
}

function addProviderTokens(messages: MessageWithParts[]): void {
  for (const message of messages) {
    if (message.info.role === "assistant") {
      Object.assign(message.info, {
        tokens: {
          input: 0,
          output: 0,
          reasoning: 0,
          cache: { read: 0, write: 0 },
        },
      });
    }
  }
  Object.assign(messages.at(-1)!.info, {
    tokens: {
      input: 10,
      output: 5,
      reasoning: 0,
      cache: { read: 0, write: 0 },
    },
  });
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

type NativeFixtureOptions = {
  tailStartID?: string | null;
  duplicateParts?: boolean;
  finish?: string | null;
  error?: unknown;
};

function nativeCheckpointFixture(
  options: NativeFixtureOptions = {},
): MessageWithParts[] {
  return [
    ...completedTurn("tail", "Pre-checkpoint tail"),
    nativeMarker(
      "native-marker",
      options.tailStartID === undefined
        ? "tail-assistant"
        : options.tailStartID,
      options.duplicateParts,
    ),
    nativeSummary("native-summary", "native-marker", {
      finish: options.finish === undefined ? "stop" : options.finish,
      error: options.error,
    }),
    ...completedTurn("suffix-1", "First suffix request"),
    ...completedTurn("suffix-2", "Second suffix request"),
  ];
}

function completedTurn(
  id: string,
  text: string,
  boundary = false,
): MessageWithParts[] {
  const userID = `${id}-user`;
  return [
    message(userID, "user", [
      textPart(`${id}-text`, "source", userID, text),
      ...(boundary ? [magicBoundary(`${id}-boundary`, userID)] : []),
    ]),
    message(`${id}-assistant`, "assistant", []),
  ];
}

function nativeMarker(
  id: string,
  tailStartID: string | null,
  duplicate = false,
): MessageWithParts {
  const part = {
    id: `${id}-part`,
    sessionID: "source",
    messageID: id,
    type: "compaction",
    auto: true,
    ...(tailStartID ? { tail_start_id: tailStartID } : {}),
  } as unknown as Part;
  return message(
    id,
    "user",
    duplicate ? [part, { ...part, id: `${id}-part-2` }] : [part],
  );
}

function assistantCompaction(id: string): MessageWithParts {
  const marker = nativeMarker(id, null);
  marker.info = { ...marker.info, role: "assistant" } as Message;
  return marker;
}

function nativeSummary(
  id: string,
  parentID: string,
  options: { finish?: string | null; error?: unknown } = {},
): MessageWithParts {
  return {
    info: {
      id,
      role: "assistant",
      parentID,
      summary: true,
      finish: options.finish === undefined ? "stop" : options.finish,
      ...(options.error === undefined ? {} : { error: options.error }),
    } as Message,
    parts: [textPart(`${id}-text`, "source", id, "Native summary text")],
  };
}

function magicBoundary(id: string, messageID: string): TextPart {
  return {
    ...textPart(id, "source", messageID, "Magic boundary"),
    synthetic: true,
    metadata: { magicCompact: { boundary: true } },
  };
}

async function planFor(messages: MessageWithParts[], keepTurns: number) {
  const v2 = {
    session: { messages: async () => ({ data: messages }) },
  } as unknown as V2Client;
  return createCompactionPlan(v2, "source", keepTurns);
}

function assistantIDs(turns: Turn[]): string[] {
  return turns.flatMap(turn => turn.assistants.map(item => item.info.id));
}

function nestedErrorMessages(error: unknown): string[] {
  if (error instanceof AggregateError) {
    return error.errors.flatMap(nestedErrorMessages);
  }
  return [error instanceof Error ? error.message : String(error)];
}

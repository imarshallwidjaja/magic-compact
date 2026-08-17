import {
  afterAll,
  beforeAll,
  beforeEach,
  describe,
  expect,
  test,
} from "bun:test";
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Config, PluginInput } from "@opencode-ai/plugin";
import type { AssistantMessage, Event } from "@opencode-ai/sdk";
import type { Message, Part, Session, TextPart } from "@opencode-ai/sdk/v2";
import type { V2Client } from "../src/api";
import {
  computeNativeBatchBudget,
  estimateNativeTurnCost,
  NATIVE_BATCH_FIXED_RESERVE_TOKENS,
  planNativeGenerationBatches,
} from "../src/compact/batch";
import {
  generateCompactionSummaries,
  injectSummaries,
} from "../src/compact/compact";
import { projectManifestCoveredMedia } from "../src/compact/media";
import {
  MAGIC_COMPACT_SUMMARIZER_PROMPT,
  POST_COMPACTION_NOTICE,
} from "../src/compact/constants";
import {
  assertCompactionSourceUnchanged,
  createCompactionPlan,
  NativeCheckpointChangedError,
  type CompactionPlan,
  type MessageWithParts,
  type Turn,
} from "../src/compact/plan";
import { createBackup } from "../src/compact/session";
import {
  COMPACT_INSUFFICIENT_SAVINGS,
  executeMagicCompact,
} from "../src/magic-compact";
import { executeMagicStats } from "../src/magic-stats";
import server from "../src/index";
import {
  allocateOmission,
  cachePath,
  deleteCache,
  readCache,
  readOmittedContent,
} from "../src/storage/omission";
import { pluginStorageDirectory } from "../src/storage/store";
import {
  type ConversationStats,
  deleteStats,
  readStats,
  writeStats,
} from "../src/storage/stats";

type TurnResult = {
  ordinal: number;
  marker?: string;
  anchor: string;
  summary: string;
  visualDescriptions: Array<{ ordinal: number; description: string }>;
};

type GenerationResponse = { turns: TurnResult[] };

let storageDirectory: string;
const originalDataHome = process.env.XDG_DATA_HOME;

beforeAll(async () => {
  storageDirectory = await mkdtemp(join(tmpdir(), "magic-compact-test-"));
  process.env.XDG_DATA_HOME = storageDirectory;
});

beforeEach(async () => {
  await Promise.all([
    deleteCache("source"),
    deleteCache("backup"),
    deleteStats("source"),
    deleteStats("backup"),
  ]);
});

afterAll(async () => {
  if (originalDataHome === undefined) delete process.env.XDG_DATA_HOME;
  else process.env.XDG_DATA_HOME = originalDataHome;
  await rm(storageDirectory, { recursive: true, force: true });
});

describe("native generation batch planning", () => {
  const budget = computeNativeBatchBudget({
    context: 1_000_000,
    input: 500_000,
    output: 64_000,
  });
  const estimate = (
    ordinal: number,
    inputTokens: number,
    outputTokens = 1_000,
  ) => ({ ordinal, inputTokens, outputTokens });

  test("derives output, normal, hard-turn, and halo limits from the configured model", () => {
    expect(budget).toEqual({
      outputTarget: 32_000,
      normalInputTarget: 400_000,
      hardSingleTurnInput: 450_000,
      haloInputLimit: 450_000,
    });

    expect(
      computeNativeBatchBudget({ context: 400_000, output: 64_000 }),
    ).toEqual({
      outputTarget: 32_000,
      normalInputTarget: 294_400,
      hardSingleTurnInput: 331_200,
      haloInputLimit: 331_200,
    });
  });

  test("rejects missing or invalid context limits and uses the fallback output target", () => {
    expect(() => computeNativeBatchBudget({} as { context: number })).toThrow(
      "positive configured compaction model context limit",
    );
    expect(() =>
      computeNativeBatchBudget({ context: 0, output: 64_000 }),
    ).toThrow("positive configured compaction model context limit");
    expect(computeNativeBatchBudget({ context: 400_000 })).toMatchObject({
      outputTarget: 16_000,
      normalInputTarget: 307_200,
      hardSingleTurnInput: 345_600,
      haloInputLimit: 345_600,
    });
  });

  test("rejects a model input budget too small to fit the fixed request reserve", () => {
    expect(() =>
      computeNativeBatchBudget({ context: 40_889, output: 64_000 }),
    ).toThrow(/input budget.*too small/i);
    expect(() =>
      computeNativeBatchBudget({
        context: 1_000_000,
        input: 8_889,
        output: 64_000,
      }),
    ).toThrow(/input budget.*too small/i);
    expect(
      computeNativeBatchBudget({ context: 40_890, output: 64_000 }),
    ).toMatchObject({ hardSingleTurnInput: 8_001 });
  });

  test("keeps below and exact 400K inputs normal and dedicates an above-target turn", () => {
    const below = planNativeGenerationBatches(
      [estimate(1, 400_000 - NATIVE_BATCH_FIXED_RESERVE_TOKENS - 1)],
      budget,
    );
    const exact = planNativeGenerationBatches(
      [estimate(1, 400_000 - NATIVE_BATCH_FIXED_RESERVE_TOKENS)],
      budget,
    );
    const above = planNativeGenerationBatches(
      [
        estimate(1, 400_000 - NATIVE_BATCH_FIXED_RESERVE_TOKENS + 1),
        estimate(2, 1),
      ],
      budget,
    );

    expect(below).toMatchObject([
      { estimatedInputTokens: 399_999, candidates: [{ ordinal: 1 }] },
    ]);
    expect(exact).toMatchObject([
      { estimatedInputTokens: 400_000, candidates: [{ ordinal: 1 }] },
    ]);
    expect(
      above.map(batch => batch.candidates.map(item => item.ordinal)),
    ).toEqual([[1], [2]]);
  });

  test("caps a normal batch at 12 target turns and at the output target", () => {
    const byTurns = planNativeGenerationBatches(
      Array.from({ length: 13 }, (_, index) => estimate(index + 1, 1)),
      budget,
    );
    const byOutput = planNativeGenerationBatches(
      [estimate(1, 1, 17_000), estimate(2, 1, 17_000)],
      budget,
    );

    expect(byTurns.map(batch => batch.candidates.length)).toEqual([12, 1]);
    expect(byOutput.map(batch => batch.candidates.length)).toEqual([1, 1]);
  });

  test("allows one dedicated oversized turn but rejects a turn beyond the hard limit", () => {
    expect(
      planNativeGenerationBatches(
        [estimate(1, 430_000 - NATIVE_BATCH_FIXED_RESERVE_TOKENS)],
        budget,
      ),
    ).toMatchObject([
      { estimatedInputTokens: 430_000, candidates: [{ ordinal: 1 }] },
    ]);
    expect(() =>
      planNativeGenerationBatches(
        [estimate(1, 450_000 - NATIVE_BATCH_FIXED_RESERVE_TOKENS + 1)],
        budget,
      ),
    ).toThrow(/magic-trim.*first/i);
  });

  test("requires more model output capacity when one turn exceeds the output target", () => {
    expect(() =>
      planNativeGenerationBatches([estimate(1, 1, 32_001)], budget),
    ).toThrow(/single compaction turn.*more output capacity/i);
  });

  test.each([
    [
      "nested input",
      (dataURL: string) => ({
        status: "completed",
        input: { request: { screenshot: dataURL } },
        output: "inspected",
        title: "Inspect",
        metadata: {},
        time: { start: 1, end: 2 },
      }),
    ],
    [
      "completed output",
      (dataURL: string) => ({
        status: "completed",
        input: { path: "/tmp/result" },
        output: dataURL,
        title: "Read",
        metadata: {},
        time: { start: 1, end: 2 },
      }),
    ],
    [
      "error output",
      (dataURL: string) => ({
        status: "error",
        input: { path: "/tmp/result" },
        error: dataURL,
        metadata: {},
        time: { start: 1, end: 2 },
      }),
    ],
    [
      "interrupted metadata output",
      (dataURL: string) => ({
        status: "error",
        input: { path: "/tmp/result" },
        error: "interrupted",
        metadata: { interrupted: true, output: dataURL },
        time: { start: 1, end: 2 },
      }),
    ],
  ])(
    "charges a multi-megabyte data URL in %s as model-visible text",
    (_name, state) => {
      const dataURL = `data:application/octet-stream;base64,${"A".repeat(2_000_000)}`;
      const turn = candidateTurn(`estimate-${_name}`, "inspect", [
        toolPart(
          `estimate-${_name}-tool`,
          `estimate-${_name}-assistant`,
          state(dataURL),
        ),
      ]);
      const cost = estimateNativeTurnCost(turn);

      expect(cost.inputTokens).toBeGreaterThan(dataURL.length);
      expect(() =>
        planNativeGenerationBatches([{ ordinal: 1, ...cost }], budget),
      ).toThrow(/magic-trim.*first/i);
    },
  );

  test("mirrors compacted, interrupted, error, pending, and running tool output visibility", () => {
    const hiddenDataURL = `data:image/png;base64,${"A".repeat(2_000_000)}`;
    const cost = (id: string, state: Record<string, unknown>) =>
      estimateNativeTurnCost(
        candidateTurn(id, "inspect", [
          toolPart(`${id}-tool`, `${id}-assistant`, state),
        ]),
      );
    const compactedLarge = cost("compacted-large", {
      status: "completed",
      input: {},
      output: hiddenDataURL,
      title: "Read",
      metadata: {},
      time: { start: 1, end: 2, compacted: 3 },
      attachments: [{ mime: "image/png", url: hiddenDataURL }],
    });
    const compactedSmall = cost("compacted-small", {
      status: "completed",
      input: {},
      output: "small",
      title: "Read",
      metadata: {},
      time: { start: 1, end: 2, compacted: 3 },
    });
    const completedEmpty = cost("completed-empty", {
      status: "completed",
      input: {},
      output: "",
      title: "Read",
      metadata: {},
      time: { start: 1, end: 2 },
    });
    const normalError = cost("normal-error", {
      status: "error",
      input: {},
      error: "visible error",
      metadata: { interrupted: false, output: hiddenDataURL },
      time: { start: 1, end: 2 },
    });
    const pending = cost("pending", {
      status: "pending",
      input: {},
      metadata: { output: hiddenDataURL },
    });
    const running = cost("running", {
      status: "running",
      input: {},
      title: "Read",
      metadata: { output: hiddenDataURL },
      time: { start: 1 },
    });

    expect(compactedLarge).toEqual(compactedSmall);
    expect(normalError.inputTokens).toBeLessThan(10_000);
    expect(pending.inputTokens).toBeLessThan(10_000);
    expect(pending.inputTokens).toBeGreaterThan(completedEmpty.inputTokens);
    expect(running.inputTokens).toEqual(pending.inputTokens);
  });

  test("keeps typed files and completed tool attachments on fixed media charges", () => {
    const cost = (id: string, url: string) =>
      estimateNativeTurnCost(
        candidateTurn(id, "inspect", [
          imagePart(`${id}-image`, `${id}-user`, url),
          toolPart(`${id}-tool`, `${id}-assistant`, {
            status: "completed",
            input: { path: "/tmp/result" },
            output: "inspected",
            title: "Inspect",
            metadata: {},
            time: { start: 1, end: 2 },
            attachments: [{ mime: "application/pdf", url }],
          }),
        ]),
      );
    const small = cost("fixed-small", "data:image/png;base64,QQ==");
    const large = cost(
      "fixed-large",
      `data:image/png;base64,${"A".repeat(2_000_000)}`,
    );

    expect(large).toEqual(small);
    expect(large.inputTokens).toBeGreaterThanOrEqual(16_000);
    expect(large.inputTokens).toBeLessThan(30_000);
  });

  test("estimates non-media data URL file parts by encoded length instead of a flat media charge", () => {
    const dataURL = `data:application/octet-stream;base64,${"A".repeat(1_000_000)}`;
    const turn = candidateTurn("text-data-url", "inspect", [
      imagePart(
        "text-data-url-file",
        "text-data-url-user",
        dataURL,
        "application/octet-stream",
      ),
    ]);
    const cost = estimateNativeTurnCost(turn);

    expect(cost.inputTokens).toBeGreaterThan(dataURL.length);
    expect(cost.inputTokens).toBeLessThan(dataURL.length + 5_000);
  });

  test("does not charge ignored text/plain or directory file parts", () => {
    const textPlain = estimateNativeTurnCost(
      candidateTurn("text-plain", "inspect", [
        imagePart(
          "text-plain-file",
          "text-plain-user",
          `data:text/plain;base64,${"A".repeat(1_000_000)}`,
          "text/plain",
        ),
      ]),
    );
    const directory = estimateNativeTurnCost(
      candidateTurn("directory", "inspect", [
        imagePart(
          "directory-file",
          "directory-user",
          "file:///tmp/directory",
          "application/x-directory",
        ),
      ]),
    );

    expect(textPlain.inputTokens).toBeLessThan(2_000);
    expect(directory.inputTokens).toBeLessThan(2_000);
  });

  test("charges ordinary image and PDF file parts the fixed media charge", () => {
    const image = estimateNativeTurnCost(
      candidateTurn("plain-image", "inspect", [
        imagePart(
          "plain-image-file",
          "plain-image-user",
          "data:image/png;base64,QQ==",
        ),
      ]),
    );
    const pdf = estimateNativeTurnCost(
      candidateTurn("plain-pdf", "inspect", [
        imagePart(
          "plain-pdf-file",
          "plain-pdf-user",
          "data:application/pdf;base64,QQ==",
          "application/pdf",
        ),
      ]),
    );

    expect(image.inputTokens).toBeGreaterThanOrEqual(8_000);
    expect(image.inputTokens).toBeLessThan(9_000);
    expect(pdf.inputTokens).toBeGreaterThanOrEqual(8_000);
    expect(pdf.inputTokens).toBeLessThan(9_000);
  });
});

describe("bounded native compaction generation", () => {
  test("registers a hidden deny-all summarizer whose prompt uses retained native fork history", async () => {
    const hooks = await server({} as PluginInput);
    const config = {} as Config;

    await hooks.config?.(config);

    expect(config.agent?.["magic-compact-summarizer"]).toMatchObject({
      hidden: true,
      mode: "subagent",
      permission: { "*": "deny" },
    });
    expect(MAGIC_COMPACT_SUMMARIZER_PROMPT).toContain(
      "bounded retained source history",
    );
    expect(MAGIC_COMPACT_SUMMARIZER_PROMPT).toContain("retained fork history");
    expect(config.command?.["magic-compact"]).toBeDefined();
    expect(config.command?.["magic-trim"]).toBeDefined();
    expect(config.command?.["magic-stats"]).toBeDefined();
    expect(hooks.tool?.read_omitted_content).toBeDefined();
  });

  test("keeps the explicit incident-shaped prompt bounded and delegates complete history to one fork", async () => {
    const imageURL = "data:image/png;base64,Zm9v";
    const attachmentURL = `data:image/png;base64,${"A".repeat(13_284_982)}`;
    const encryptedReasoning = `encrypted:${"z".repeat(137_772)}`;
    const turn = candidateTurn("incident", "First line\nsecond line", [
      imagePart("incident-image", "incident-user", imageURL),
      toolPart("incident-tool", "incident-assistant", {
        status: "completed",
        input: { path: "/tmp/incident" },
        output: "inspected",
        title: "Inspect",
        metadata: { encryptedReasoning },
        time: { start: 1, end: 2 },
        attachments: [{ mime: "image/png", url: attachmentURL }],
      }),
    ]);
    const harness = generationHarness([turn], responseFor([turn]));

    const summaries = await generateCompactionSummaries(
      harness.v2,
      sourceSession(),
      generationPlan([turn]),
    );

    expect(summaries).toHaveLength(1);
    expect(harness.forks).toHaveLength(1);
    expect(harness.prompts).toHaveLength(1);
    expect(harness.deletes).toEqual(["generation-fork"]);
    const request = harness.prompts[0]!;
    const parts = request.parts as Array<Record<string, unknown>>;
    expect(parts).toHaveLength(1);
    expect(parts[0]?.type).toBe("text");
    const prompt = String(parts[0]?.text);
    expect(prompt.length).toBeLessThan(3_000);
    expect(prompt).toContain('"ordinal":1');
    expect(prompt).toContain('"anchor":"First line"');
    expect(prompt).toContain('"expectedInlineImages":1');
    expect(prompt).not.toContain("data:");
    expect(prompt).not.toContain("encrypted:");
    expect(prompt).not.toContain("incident-tool");
    expect(prompt).not.toContain(attachmentURL.slice(0, 100));
    expect(parts.some(part => part.type === "file")).toBeFalse();
  });

  test("denies the complete fork before one configured summarizer request and deletes it once", async () => {
    const turns = [
      candidateTurn("first", "alpha", []),
      candidateTurn("second", "beta", []),
      candidateTurn("third", "gamma", []),
    ];
    const harness = generationHarness(turns, responseFor(turns));

    const summaries = await generateCompactionSummaries(
      harness.v2,
      sourceSession(),
      generationPlan(turns),
    );

    expect(summaries).toHaveLength(3);
    expect(harness.events).toEqual([
      "agents",
      "providers",
      "fork",
      "deny",
      "messages",
      "messages",
      "marker",
      "marker",
      "marker",
      "prompt",
      "delete",
    ]);
    expect(harness.updates).toEqual([
      {
        sessionID: "generation-fork",
        directory: "/workspace",
        workspace: "workspace",
        permission: [{ permission: "*", pattern: "*", action: "deny" }],
      },
    ]);
    expect(harness.prompts[0]).toMatchObject({
      sessionID: "generation-fork",
      directory: "/workspace",
      workspace: "workspace",
      agent: "magic-compact-summarizer",
      model: { providerID: "configured-provider", modelID: "configured-model" },
      variant: "configured-variant",
    });
    expect(harness.creates).toEqual([]);
    expect(harness.forks).toEqual([
      {
        sessionID: "source",
        directory: "/workspace",
        workspace: "workspace",
      },
    ]);
    expect(harness.deletes).toEqual(["generation-fork"]);
  });

  test("isolates 15 targets across distinct forks with global ordinals and bounded whole-turn halos", async () => {
    const turns = Array.from({ length: 15 }, (_, index) =>
      candidateTurn(`batch-${index + 1}`, `anchor ${index + 1}`, []),
    );
    const harness = generationHarness(turns, responseFor(turns), {
      providerDelayByFirstOrdinal: { 1: 20 },
    });

    const summaries = await generateCompactionSummaries(
      harness.v2,
      sourceSession(),
      generationPlan(turns),
    );

    expect(summaries.map(summary => summary?.text)).toEqual(
      turns.map(
        (_, index) => `Turn ${index + 1} outcome and continuation details.`,
      ),
    );
    expect(harness.providerCalls).toBe(1);
    expect(harness.deletes).toEqual(["generation-fork-2", "generation-fork"]);
    expect(harness.completedPromptFirstOrdinals).toEqual([13, 1]);
    expect(
      harness.prompts
        .map(generationPromptIndex)
        .map(index => index.turns.map(turn => turn.ordinal)),
    ).toEqual([
      Array.from({ length: 12 }, (_, index) => index + 1),
      [13, 14, 15],
    ]);
    expect(
      harness.promptHistories.map(history => history.messages.length),
    ).toEqual([26, 8]);
    expect(
      harness.messageDeletes.filter(
        request => request.sessionID === "generation-fork",
      ),
    ).toHaveLength(4);
    expect(
      harness.messageDeletes.filter(
        request => request.sessionID === "generation-fork-2",
      ),
    ).toHaveLength(22);
    expect(harness.partDeletes).toEqual([]);
    const noncesByFork = new Map<string, Set<string>>();
    for (const update of harness.markerUpdates) {
      const nonce = update.part.text.match(/nonce="([A-Za-z0-9_-]{22})"/u)?.[1];
      expect(nonce).toBeDefined();
      const nonces = noncesByFork.get(update.sessionID) ?? new Set<string>();
      nonces.add(nonce!);
      noncesByFork.set(update.sessionID, nonces);
    }
    expect([...noncesByFork.values()].map(nonces => nonces.size)).toEqual([
      1, 1,
    ]);
    expect(
      new Set([...noncesByFork.values()].flatMap(nonces => [...nonces])).size,
    ).toBe(2);
  });

  test("runs independent generation batches with exactly five prompts and forks in flight", async () => {
    const turns = Array.from({ length: 61 }, (_, index) =>
      candidateTurn(`concurrent-${index + 1}`, `anchor ${index + 1}`, []),
    );
    const harness = generationHarness(turns, responseFor(turns), {
      providerDelayMs: 20,
    });

    const summaries = await generateCompactionSummaries(
      harness.v2,
      sourceSession(),
      generationPlan(turns),
    );

    expect(summaries).toHaveLength(61);
    expect(harness.prompts).toHaveLength(6);
    expect(harness.forks).toHaveLength(6);
    expect(harness.peakActivePrompts).toBe(5);
    expect(harness.peakActiveForks).toBe(5);
    expect(harness.deletes).toHaveLength(6);
  });

  test("uses more chronological batches when the configured model has smaller limits", async () => {
    const turns = Array.from({ length: 4 }, (_, index) =>
      candidateTurn(`small-model-${index + 1}`, `anchor ${index + 1}`, [
        toolPart(
          `small-model-${index + 1}-tool`,
          `small-model-${index + 1}-assistant`,
          {
            status: "completed",
            input: {
              payload: `data:application/octet-stream;base64,${"A".repeat(40_000)}`,
            },
            output: "done",
            title: "Process",
            metadata: {},
            time: { start: 1, end: 2 },
          },
        ),
      ]),
    );
    const harness = generationHarness(turns, responseFor(turns), {
      modelLimits: { context: 120_000, input: 120_000, output: 32_000 },
    });

    await generateCompactionSummaries(
      harness.v2,
      sourceSession(),
      generationPlan(turns),
    );

    expect(
      harness.prompts
        .map(generationPromptIndex)
        .map(index => index.turns.map(turn => turn.ordinal))
        .sort((left, right) => left[0]! - right[0]!),
    ).toEqual([[1], [2], [3], [4]]);
  });

  test("excludes oversized neighboring summary turns without splitting their messages", async () => {
    const media = (prefix: string) =>
      Array.from({ length: 63 }, (_, index) =>
        imagePart(
          `${prefix}-image-${index}`,
          `${prefix}-user`,
          "data:image/png;base64,Zm9v",
        ),
      );
    const opaque = candidateTurn("halo-opaque", "opaque", media("halo-opaque"));
    opaque.assistants[0]!.parts.push({
      ...textPart("halo-opaque-summary", "halo-opaque-assistant", "summary"),
      metadata: { magicCompact: { summary: true, version: 2 } },
    });
    const target = candidateTurn("halo-target", "target", []);
    const next = candidateTurn("halo-next", "next", media("halo-next"));
    const harness = generationHarness(
      [opaque, target],
      { turns: [result(1, "target", "Target summary.")] },
      { nextTurn: next },
    );

    await generateCompactionSummaries(
      harness.v2,
      sourceSession(),
      generationPlan([opaque, target], next),
    );

    expect(harness.promptHistories[0]!.messages).toHaveLength(2);
    expect(
      harness.promptHistories[0]!.messages.map(message => message.info.role),
    ).toEqual(["user", "assistant"]);
    expect(harness.messageDeletes).toHaveLength(4);
    expect(harness.partDeletes).toEqual([]);
  });

  test("keeps an opaque summary halo complete and removes a native checkpoint only by whole message", async () => {
    const messages = [
      ...candidateMessages("checkpoint-before", "before", []),
      nativeMarker("checkpoint-marker"),
      nativeSummary("checkpoint-summary", "checkpoint-marker"),
      ...persistedSummaryTurn("checkpoint-opaque", 2),
      ...candidateMessages("checkpoint-fresh", "fresh", []),
    ];
    const planClient = {
      session: { messages: async () => ({ data: messages }) },
    } as unknown as V2Client;
    const plan = await createCompactionPlan(planClient, "source", 0);
    const harness = generationHarness(
      plan.summarizedTurns,
      { turns: [result(1, "fresh", "Fresh summary.")] },
      { sourceMessages: messages },
    );

    await generateCompactionSummaries(harness.v2, sourceSession(), plan);

    expect(
      harness.promptHistories[0]!.messages.map(message => ({
        role: message.info.role,
        parts: message.parts.map(part => part.type),
      })),
    ).toEqual([
      { role: "user", parts: ["text"] },
      { role: "assistant", parts: ["text"] },
      { role: "user", parts: ["text", "text"] },
      { role: "assistant", parts: ["reasoning"] },
    ]);
    expect(harness.messageDeletes).toHaveLength(4);
    expect(harness.partDeletes).toEqual([]);
  });

  test("excludes a native checkpoint halo whose tail references deleted history", async () => {
    const messages = [
      ...candidateMessages("checkpoint-tail", "tail", []),
      nativeMarker("checkpoint-halo-marker", "checkpoint-tail-assistant"),
      nativeSummary("checkpoint-halo-summary", "checkpoint-halo-marker"),
      ...candidateMessages("checkpoint-halo-target", "target", []),
    ];
    const planClient = {
      session: { messages: async () => ({ data: messages }) },
    } as unknown as V2Client;
    const plan = await createCompactionPlan(planClient, "source", 0);
    const harness = generationHarness(
      plan.summarizedTurns,
      { turns: [result(1, "target", "Target summary.")] },
      { sourceMessages: messages },
    );

    await generateCompactionSummaries(harness.v2, sourceSession(), plan);

    expect(
      harness.promptHistories[0]!.messages.map(message => ({
        role: message.info.role,
        summary: "summary" in message.info ? message.info.summary : undefined,
        parts: message.parts.map(part => part.type),
      })),
    ).toEqual([
      { role: "user", summary: undefined, parts: ["text", "text"] },
      { role: "assistant", summary: undefined, parts: ["reasoning"] },
    ]);
    expect(harness.messageDeletes).toHaveLength(4);
  });

  test("keeps cross-provider reasoning on OpenCode's native fork conversion path", async () => {
    const turn = candidateTurn("cross-provider", "summarize reasoning", []);
    const reasoning = turn.assistants[0]!.parts.find(
      part => part.type === "reasoning",
    )!;
    (reasoning as Extract<Part, { type: "reasoning" }>).text =
      "provider-native reasoning must not enter the explicit prompt";
    const harness = generationHarness([turn], responseFor([turn]));

    await generateCompactionSummaries(
      harness.v2,
      sourceSession(),
      generationPlan([turn]),
    );

    expect(harness.prompts[0]).toMatchObject({
      model: { providerID: "configured-provider", modelID: "configured-model" },
      variant: "configured-variant",
    });
    expect(promptText(harness.prompts[0]!)).not.toContain(
      "provider-native reasoning",
    );
  });

  test("uses only the first ordinary user line, caps it at 300 characters, and includes nextTurn only as a stop anchor", async () => {
    const long = `${"a".repeat(299)}😀tail`;
    const selected = candidateTurn("selected", long, []);
    selected.user[0]!.parts.unshift({
      ...textPart("selected-synthetic", "selected-user", "ignored synthetic"),
      synthetic: true,
    });
    selected.user[0]!.parts.unshift({
      ...textPart("selected-ignored", "selected-user", "ignored reporting"),
      ignored: true,
    });
    const next = candidateTurn("next", 'Stop "here"\nnot this line', []);
    const expectedAnchor = `${"a".repeat(299)}😀`;
    const harness = generationHarness(
      [selected],
      { turns: [result(1, expectedAnchor, "Selected summary")] },
      { nextTurn: next },
    );

    await generateCompactionSummaries(harness.v2, sourceSession(), {
      ...generationPlan([selected], next),
    });

    const prompt = promptText(harness.prompts[0]!);
    expect(prompt).toContain(JSON.stringify(expectedAnchor));
    expect(Array.from(expectedAnchor)).toHaveLength(300);
    expect(prompt).toContain('"anchor":"Stop \\"here\\""');
    expect(prompt).not.toContain("not this line");
    expect(prompt).not.toContain("next-assistant");
    expect(prompt).not.toContain("ignored synthetic");
    expect(prompt).not.toContain("ignored reporting");
  });

  test.each([
    ["drops a turn", { turns: [result(1, "alpha", "one")] }],
    [
      "shifts a middle ordinal",
      {
        turns: [
          result(1, "alpha", "one"),
          result(3, "beta", "two"),
          result(3, "gamma", "three"),
        ],
      },
    ],
    [
      "duplicates an ordinal",
      {
        turns: [
          result(1, "alpha", "one"),
          result(1, "beta", "two"),
          result(3, "gamma", "three"),
        ],
      },
    ],
    [
      "reorders anchors",
      {
        turns: [
          result(1, "beta", "one"),
          result(2, "alpha", "two"),
          result(3, "gamma", "three"),
        ],
      },
    ],
    [
      "returns an empty summary",
      {
        turns: [
          result(1, "alpha", "one"),
          result(2, "beta", "  "),
          result(3, "gamma", "three"),
        ],
      },
    ],
  ])("fails atomically when one response %s", async (_name, response) => {
    const turns = [
      candidateTurn("alpha", "alpha", []),
      candidateTurn("beta", "beta", []),
      candidateTurn("gamma", "gamma", []),
    ];
    const harness = generationHarness(turns, response as GenerationResponse);

    await expect(
      generateCompactionSummaries(
        harness.v2,
        sourceSession(),
        generationPlan(turns),
      ),
    ).rejects.toThrow("response validation failed");

    expect(harness.forks).toHaveLength(1);
    expect(harness.prompts).toHaveLength(1);
    expect(harness.aborts).toEqual(["generation-fork"]);
    expect(harness.deletes).toEqual(["generation-fork"]);
  });

  test("covers an assistant-only image without changing its persisted bytes", async () => {
    const url = `data:image/png;base64,${"A".repeat(512)}`;
    const turn = candidateTurn("assistant-visual", "review image", []);
    const assistantImage = imagePart(
      "assistant-visual-image",
      "assistant-visual-assistant",
      url,
    );
    turn.assistants[0]!.parts.push(assistantImage);
    const before = structuredClone(turn);
    const harness = generationHarness([turn], {
      turns: [
        result(1, "review image", "Reviewed the assistant image.", [
          { ordinal: 1, description: "A completed interface state." },
        ]),
      ],
    });

    const summaries = await generateCompactionSummaries(
      harness.v2,
      sourceSession(),
      generationPlan([turn]),
    );
    const updates: TextPart[] = [];
    const v2 = {
      part: {
        update: async (request: { part: TextPart }) => {
          updates.push(request.part);
          return { data: request.part };
        },
      },
    } as unknown as V2Client;
    await injectSummaries(v2, "source", [turn], summaries);

    expect(promptText(harness.prompts[0]!)).toContain(
      '"expectedInlineImages":1',
    );
    expect(summaries[0]).toMatchObject({
      mediaManifest: {
        version: 1,
        images: [
          {
            ordinal: 1,
            mime: "image/png",
            urlSha256: createHash("sha256").update(url).digest("hex"),
          },
        ],
      },
    });
    expect(turn).toEqual(before);

    const persisted = structuredClone([...turn.user, ...turn.assistants]);
    persisted[1]!.parts.push(updates[0]!);
    const projected = projectManifestCoveredMedia(persisted);
    expect(projected[1]!.parts[1]).toMatchObject({
      id: assistantImage.id,
      sessionID: assistantImage.sessionID,
      messageID: assistantImage.messageID,
      type: "text",
      synthetic: true,
      text: expect.stringContaining("Historical inline image 1 (image/png)"),
    });
    expect(persisted[1]!.parts[1]).toEqual(assistantImage);
  });

  test("orders mixed user and assistant images in one local hash-bound v3 manifest", async () => {
    const urls = [
      `data:image/png;base64,${"A".repeat(512)}`,
      `data:image/jpeg;base64,${"B".repeat(512)}`,
      `data:image/gif;base64,${"C".repeat(512)}`,
    ];
    const turn = candidateTurn("visual", "review images", [
      imagePart("visual-one", "visual-user", urls[0]!, "image/png"),
    ]);
    turn.assistants[0]!.parts.push(
      imagePart("visual-two", "visual-assistant", urls[1]!, "image/jpeg"),
      imagePart("visual-three", "visual-assistant", urls[2]!, "image/gif"),
    );
    const before = structuredClone(turn);
    const harness = generationHarness([turn], {
      turns: [
        result(1, "review images", "Reviewed all interface captures.", [
          {
            ordinal: 1,
            description: "A dashboard with a green passing status.",
          },
          {
            ordinal: 2,
            description: "A terminal showing the completed test run.",
          },
          {
            ordinal: 3,
            description: "A final assistant-provided confirmation image.",
          },
        ]),
      ],
    });

    const summaries = await generateCompactionSummaries(
      harness.v2,
      sourceSession(),
      generationPlan([turn]),
    );
    const updates: TextPart[] = [];
    const v2 = {
      part: {
        update: async (request: { part: TextPart }) => {
          updates.push(request.part);
          return { data: request.part };
        },
      },
    } as unknown as V2Client;
    await injectSummaries(v2, "source", [turn], summaries);

    expect(turn).toEqual(before);
    expect(updates[0]).toMatchObject({
      id: "prt_-magic_capsule_v3_visual-assistant",
      metadata: {
        magicCompact: {
          summary: true,
          capsule: true,
          agentic: true,
          version: 3,
          mediaManifest: {
            version: 1,
            images: [
              {
                ordinal: 1,
                mime: "image/png",
                urlSha256: createHash("sha256").update(urls[0]!).digest("hex"),
              },
              {
                ordinal: 2,
                mime: "image/jpeg",
                urlSha256: createHash("sha256").update(urls[1]!).digest("hex"),
              },
              {
                ordinal: 3,
                mime: "image/gif",
                urlSha256: createHash("sha256").update(urls[2]!).digest("hex"),
              },
            ],
          },
        },
      },
    });
    expect(updates[0]!.text).toContain("Visual summary:");
    expect(updates[0]!.text).toContain(
      "1. A dashboard with a green passing status.",
    );
    expect(updates[0]!.text).toContain(
      "2. A terminal showing the completed test run.",
    );
    expect(updates[0]!.text).toContain(
      "3. A final assistant-provided confirmation image.",
    );
    const persisted = structuredClone([...turn.user, ...turn.assistants]);
    persisted[1]!.parts.push(updates[0]!);
    const projected = projectManifestCoveredMedia(persisted);
    expect(projected[0]!.parts[1]).toMatchObject({ type: "text" });
    expect(projected[1]!.parts[1]).toMatchObject({ type: "text" });
    expect(projected[1]!.parts[2]).toMatchObject({ type: "text" });
    expect(persisted).toEqual([
      ...structuredClone(turn.user),
      ...structuredClone(turn.assistants).map((message, index) =>
        index === 0
          ? { ...message, parts: [...message.parts, updates[0]!] }
          : message,
      ),
    ]);

    const changedAssistant = structuredClone(persisted);
    (changedAssistant[1]!.parts[1] as Extract<Part, { type: "file" }>).url =
      "data:image/jpeg;base64,bm90LWJhcg==";
    const reorderedAssistant = structuredClone(persisted);
    [reorderedAssistant[1]!.parts[1], reorderedAssistant[1]!.parts[2]] = [
      reorderedAssistant[1]!.parts[2]!,
      reorderedAssistant[1]!.parts[1]!,
    ];
    for (const invalid of [changedAssistant, reorderedAssistant]) {
      expect(() => projectManifestCoveredMedia(invalid)).toThrow(
        "media manifest mismatch",
      );
    }
  });

  test.each([
    [[], "missing"],
    [[{ ordinal: 1, description: "one" }], "missing second"],
    [
      [
        { ordinal: 2, description: "two" },
        { ordinal: 1, description: "one" },
      ],
      "misordered",
    ],
    [
      [
        { ordinal: 1, description: "one" },
        { ordinal: 2, description: " " },
      ],
      "empty",
    ],
  ])("rejects %s visual coverage (%s)", async visualDescriptions => {
    const turn = candidateTurn("visual-invalid", "review", [
      imagePart(
        "visual-invalid-one",
        "visual-invalid-user",
        "data:image/png;base64,Zm9v",
      ),
      imagePart(
        "visual-invalid-two",
        "visual-invalid-user",
        "data:image/png;base64,YmFy",
      ),
    ]);
    const harness = generationHarness([turn], {
      turns: [result(1, "review", "Reviewed images.", visualDescriptions)],
    });

    await expect(
      generateCompactionSummaries(
        harness.v2,
        sourceSession(),
        generationPlan([turn]),
      ),
    ).rejects.toThrow("response validation failed");
    expect(harness.deletes).toEqual(["generation-fork"]);
  });

  test("projects persisted v2 and generated v3 manifests across rewritten IDs while failing closed on changed content", () => {
    const urlV2 = `data:image/png;base64,${"A".repeat(512)}`;
    const urlV3 = `data:image/jpeg;base64,${"B".repeat(512)}`;
    const persisted = [
      ...persistedMediaTurn("legacy", 2, "image/png", urlV2),
      ...persistedMediaTurn("current", 3, "image/jpeg", urlV3),
    ];
    persisted[1]!.parts.unshift(
      imagePart(
        "legacy-assistant-image",
        "legacy-assistant",
        "data:image/gif;base64,YmF6",
        "image/gif",
      ),
    );
    const source = rewriteForkMessages(persisted, "recovered-backup");

    const projected = projectManifestCoveredMedia(source);

    expect(projected[0]!.parts[1]).toMatchObject({
      type: "text",
      synthetic: true,
    });
    expect(projected[2]!.parts[1]).toMatchObject({
      type: "text",
      synthetic: true,
    });
    expect(projected[1]!.parts[0]).toMatchObject({
      type: "file",
      mime: "image/gif",
      url: "data:image/gif;base64,YmF6",
    });
    expect(source[0]!.parts[1]).toMatchObject({ type: "file", url: urlV2 });
    expect(source[2]!.parts[1]).toMatchObject({ type: "file", url: urlV3 });
    expect(source[0]!.parts[1]!.id).not.toBe(persisted[0]!.parts[1]!.id);

    const changedURL = structuredClone(source);
    (changedURL[2]!.parts[1] as Extract<Part, { type: "file" }>).url =
      "data:image/jpeg;base64,bm90LWJhcg==";
    const changedMime = structuredClone(source);
    (changedMime[2]!.parts[1] as Extract<Part, { type: "file" }>).mime =
      "image/png";
    const reordered = structuredClone(source);
    reordered[0]!.parts.push(reordered[2]!.parts[1]!);
    reordered[2]!.parts.splice(1, 1);
    for (const invalid of [changedURL, changedMime, reordered]) {
      expect(() => projectManifestCoveredMedia(invalid)).toThrow(
        "media manifest mismatch",
      );
    }
  });

  test("maps duplicate and empty anchors through explicit rewritten-fork markers without touching source", async () => {
    const first = candidateTurn("duplicate-one", "same anchor", []);
    first.user[0]!.parts.push(
      textPart(
        "duplicate-old-marker",
        "duplicate-one-user",
        '<magic-compact-turn ordinal="1" />',
      ),
    );
    first.assistants[0]!.parts.push(
      toolPart("duplicate-tool", "duplicate-one-assistant", {
        status: "completed",
        input: {},
        output: "<magic-compact-stop />",
        title: "Historical marker",
        metadata: {},
        time: { start: 1, end: 2 },
      }),
    );
    const opaque = candidateTurn("opaque", "same anchor", []);
    opaque.assistants[0]!.parts.push({
      ...textPart("opaque-summary", "opaque-assistant", "Existing summary"),
      metadata: { magicCompact: { summary: true, version: 2 } },
    });
    const empty = candidateTurn("empty", "", []);
    const next = candidateTurn("next-duplicate", "same anchor", []);
    const turns = [first, opaque, empty];
    const before = structuredClone(
      turns.flatMap(turn => [...turn.user, ...turn.assistants]),
    );
    const harness = generationHarness(
      turns,
      {
        turns: [
          result(1, "same anchor", "First duplicate summary."),
          result(2, "", "Empty-anchor summary."),
        ],
      },
      { nextTurn: next },
    );

    const summaries = await generateCompactionSummaries(
      harness.v2,
      sourceSession(),
      generationPlan(turns, next),
    );

    expect(summaries[0]).toMatchObject({ text: "First duplicate summary." });
    expect(summaries[1]).toBeNull();
    expect(summaries[2]).toMatchObject({ text: "Empty-anchor summary." });
    const markerTexts = harness.markerUpdates.map(update => update.part.text);
    const nonces = markerTexts.map(
      marker => marker.match(/nonce="([A-Za-z0-9_-]{22})"/u)?.[1],
    );
    expect(new Set(nonces)).toEqual(new Set([nonces[0]]));
    expect(nonces[0]).toMatch(/^[A-Za-z0-9_-]{22}$/u);
    expect(markerTexts[0]).toContain('ordinal="1"');
    expect(markerTexts[1]).toContain('ordinal="2"');
    expect(markerTexts[2]).toContain("magic-compact-stop");
    expect(markerTexts).not.toContain('<magic-compact-turn ordinal="1" />');
    expect(markerTexts).not.toContain("<magic-compact-stop />");
    expect(harness.markerUpdates.map(update => update.messageID)).toEqual([
      "fork-message-1",
      "fork-message-5",
      "fork-message-7",
    ]);
    const prompt = promptText(harness.prompts[0]!);
    expect(prompt).toContain(JSON.stringify(markerTexts[0]));
    expect(prompt).toContain(JSON.stringify(markerTexts[1]));
    expect(prompt).toContain(JSON.stringify(markerTexts[2]));
    expect(
      summaries.flatMap(summary => (summary ? [summary.text] : [])),
    ).not.toContain(nonces[0]!);
    expect(turns.flatMap(turn => [...turn.user, ...turn.assistants])).toEqual(
      before,
    );
  });

  test("rejects a stale generation nonce even when ordinal and anchor match", async () => {
    const turn = candidateTurn("stale", "same anchor", []);
    const harness = generationHarness([turn], {
      turns: [
        {
          ...result(1, "same anchor", "Stale mapping."),
          marker:
            '<magic-compact-turn nonce="AAAAAAAAAAAAAAAAAAAAAA" ordinal="1" />',
        },
      ],
    });

    await expect(
      generateCompactionSummaries(
        harness.v2,
        sourceSession(),
        generationPlan([turn]),
      ),
    ).rejects.toThrow("response validation failed");
    expect(harness.aborts).toEqual(["generation-fork"]);
    expect(harness.deletes).toEqual(["generation-fork"]);
  });

  test("rejects exact markers echoed against the wrong turns", async () => {
    const turns = [
      candidateTurn("marker-first", "first", []),
      candidateTurn("marker-second", "second", []),
    ];
    const harness = generationHarness(turns, responseFor(turns), {
      responseMarkerOrder: [1, 0],
    });

    await expect(
      generateCompactionSummaries(
        harness.v2,
        sourceSession(),
        generationPlan(turns),
      ),
    ).rejects.toThrow("response validation failed");
    expect(harness.aborts).toEqual(["generation-fork"]);
    expect(harness.deletes).toEqual(["generation-fork"]);
  });

  test("accepts unrelated nonce-shaped text and marker-like text", async () => {
    const activeNonce = "A".repeat(22);
    const unrelatedNonce = "B".repeat(22);
    const unrelatedMarker = `<magic-compact-turn nonce="${unrelatedNonce}" ordinal="1" />`;
    const turn = candidateTurn("unrelated-nonce", "unrelated", []);
    const harness = generationHarness([turn], {
      turns: [
        result(
          1,
          "unrelated",
          `Kept ${unrelatedNonce} and ${unrelatedMarker} as ordinary text.`,
        ),
      ],
    });

    const summaries = await generateCompactionSummaries(
      harness.v2,
      sourceSession(),
      generationPlan([turn]),
      { nonceFactory: () => activeNonce },
    );

    expect(summaries[0]!.text).toContain(unrelatedNonce);
    expect(summaries[0]!.text).toContain(unrelatedMarker);
    expect(harness.aborts).toEqual([]);
    expect(harness.deletes).toEqual(["generation-fork"]);
  });

  test("regenerates a colliding nonce and fails closed after repeated collisions", async () => {
    const collidingNonce = "A".repeat(22);
    const replacementNonce = "B".repeat(22);
    const turn = candidateTurn(
      "nonce-collision",
      `history contains ${collidingNonce}`,
      [],
    );
    const before = structuredClone(turn);
    const values = [collidingNonce, replacementNonce];
    const regenerated = generationHarness([turn], responseFor([turn]));

    const summaries = await generateCompactionSummaries(
      regenerated.v2,
      sourceSession(),
      generationPlan([turn]),
      { nonceFactory: () => values.shift()! },
    );

    expect(regenerated.markerUpdates[0]!.part.text).toContain(replacementNonce);
    expect(regenerated.markerUpdates[0]!.part.text).not.toContain(
      collidingNonce,
    );
    expect(summaries[0]!.text).not.toContain(replacementNonce);
    expect(turn).toEqual(before);

    const exhausted = generationHarness([turn], responseFor([turn]));
    await expect(
      generateCompactionSummaries(
        exhausted.v2,
        sourceSession(),
        generationPlan([turn]),
        { nonceFactory: () => collidingNonce },
      ),
    ).rejects.toThrow("generation marker nonce collision");
    expect(exhausted.markerUpdates).toEqual([]);
    expect(exhausted.prompts).toEqual([]);
    expect(exhausted.deletes).toEqual(["generation-fork"]);
  });

  test("fails before the model request when rewritten fork shape or marker injection is invalid", async () => {
    const turn = candidateTurn("shape", "shape request", []);
    const shapeHarness = generationHarness([turn], responseFor([turn]), {
      forkMessages: rewriteForkMessages(
        [...turn.user, ...turn.assistants].slice(0, 1),
      ),
    });

    await expect(
      generateCompactionSummaries(
        shapeHarness.v2,
        sourceSession(),
        generationPlan([turn]),
      ),
    ).rejects.toThrow("fork history shape verification failed");
    expect(shapeHarness.prompts).toEqual([]);
    expect(shapeHarness.deletes).toEqual(["generation-fork"]);

    const markerHarness = generationHarness([turn], responseFor([turn]), {
      markerError: new Error("raw marker failure"),
    });
    await expect(
      generateCompactionSummaries(
        markerHarness.v2,
        sourceSession(),
        generationPlan([turn]),
      ),
    ).rejects.toThrow("fork marker injection failed");
    expect(markerHarness.prompts).toEqual([]);
    expect(markerHarness.deletes).toEqual(["generation-fork"]);
  });

  test.each([
    [
      "substituted message and part IDs",
      (messages: MessageWithParts[]) => {
        const user = messages[0]!;
        const replacementID = `${user.info.id}-replacement`;
        Object.assign(user.info, { id: replacementID });
        user.parts = user.parts.map((part, index) => ({
          ...part,
          id: index === 0 ? `${part.id}-replacement` : part.id,
          messageID: replacementID,
        })) as Part[];
        Object.assign(messages[1]!.info, { parentID: replacementID });
      },
    ],
    [
      "reordered same-shape records",
      (messages: MessageWithParts[]) => {
        messages.splice(
          0,
          messages.length,
          messages[2]!,
          messages[3]!,
          messages[0]!,
          messages[1]!,
        );
      },
    ],
    [
      "changed parent link",
      (messages: MessageWithParts[]) => {
        Object.assign(messages[1]!.info, {
          parentID: messages[2]!.info.id,
        });
      },
    ],
  ] as const)(
    "rejects %s after retained fork isolation",
    async (_name, mutate) => {
      const turns = Array.from({ length: 13 }, (_, index) =>
        candidateTurn(
          `retained-identity-${index + 1}`,
          `anchor ${index + 1}`,
          [],
        ),
      );
      for (const turn of turns) {
        Object.assign(turn.assistants[0]!.info, {
          parentID: turn.user[0]!.info.id,
        });
      }
      const harness = generationHarness(turns, responseFor(turns), {
        mutateRetainedMessages: (messages, sessionID) => {
          if (sessionID === "generation-fork-2") mutate(messages);
        },
      });

      await expect(
        generateCompactionSummaries(
          harness.v2,
          sourceSession(),
          generationPlan(turns),
        ),
      ).rejects.toThrow("fork history shape verification failed");
      expect(
        harness.markerUpdates.filter(
          update => update.sessionID === "generation-fork-2",
        ),
      ).toEqual([]);
      expect(
        harness.prompts.filter(
          prompt => prompt.sessionID === "generation-fork-2",
        ),
      ).toEqual([]);
    },
  );

  test.each([
    ["missing", null],
    ["invalid", { context: 0, input: 0, output: 0 }],
  ] as const)(
    "rejects a %s configured model limit before fork creation",
    async (_name, modelLimits) => {
      const turn = candidateTurn("model-limit", "summarize", []);
      const harness = generationHarness([turn], responseFor([turn]), {
        modelLimits,
      });

      await expect(
        generateCompactionSummaries(
          harness.v2,
          sourceSession(),
          generationPlan([turn]),
        ),
      ).rejects.toThrow(/model (?:descriptor|context limit)/i);
      expect(harness.forks).toEqual([]);
      expect(harness.prompts).toEqual([]);
    },
  );

  test.each([
    ["a non-vision model", false],
    ["a missing configured model", undefined],
  ])(
    "rejects media candidates with %s before fork creation",
    async (_name, image) => {
      const turn = candidateTurn("vision-guard", "inspect image", [
        imagePart(
          "vision-guard-image",
          "vision-guard-user",
          "data:image/png;base64,Zm9v",
        ),
      ]);
      const harness = generationHarness([turn], responseFor([turn]), {
        modelImageCapability: image,
      });

      await expect(
        generateCompactionSummaries(
          harness.v2,
          sourceSession(),
          generationPlan([turn]),
        ),
      ).rejects.toThrow("image-capable configured compaction model");
      expect(harness.forks).toEqual([]);
      expect(harness.prompts).toEqual([]);
    },
  );

  test.each([
    [
      "a retained-tail user image",
      () => {
        const selected = candidateTurn("retained-selected", "selected", []);
        const nextTurn = candidateTurn("retained-image", "retained", [
          imagePart(
            "retained-image-file",
            "retained-image-user",
            "data:image/png;base64,Zm9v",
          ),
        ]);
        return { turns: [selected], nextTurn };
      },
    ],
    [
      "an opaque summary user image",
      () => {
        const opaque = candidateTurn("opaque-image", "opaque", [
          imagePart(
            "opaque-image-file",
            "opaque-image-user",
            "data:image/png;base64,Zm9v",
          ),
        ]);
        opaque.assistants[0]!.parts.push({
          ...textPart(
            "opaque-image-summary",
            "opaque-image-assistant",
            "Persisted summary",
          ),
          metadata: { magicCompact: { summary: true, version: 2 } },
        });
        return {
          turns: [opaque, candidateTurn("opaque-fresh", "fresh", [])],
          nextTurn: null,
        };
      },
    ],
    [
      "a tool attachment only",
      () => ({
        turns: [
          candidateTurn("attachment", "inspect attachment", [
            toolPart("attachment-tool", "attachment-assistant", {
              status: "completed",
              input: {},
              output: "attached result",
              title: "Attachment",
              metadata: {},
              time: { start: 1, end: 2 },
              attachments: [
                {
                  mime: "image/png",
                  url: "data:image/png;base64,Zm9v",
                },
              ],
            }),
          ]),
        ],
        nextTurn: null,
      }),
    ],
    [
      "an assistant image file part",
      () => {
        const turn = candidateTurn("assistant-image", "inspect image", []);
        turn.assistants[0]!.parts.push(
          imagePart(
            "assistant-image-file",
            "assistant-image-assistant",
            "data:image/png;base64,Zm9v",
          ),
        );
        return { turns: [turn], nextTurn: null };
      },
    ],
  ] as const)(
    "requires vision before fork creation when complete history exposes %s",
    async (_name, fixture) => {
      const { turns, nextTurn } = fixture();
      const source = [
        ...turns.flatMap(turn => [...turn.user, ...turn.assistants]),
        ...(nextTurn ? [...nextTurn.user, ...nextTurn.assistants] : []),
      ];
      const before = structuredClone(source);
      const candidates = turns.filter(
        turn =>
          !turn.assistants.some(message =>
            message.parts.some(
              part =>
                part.type === "text"
                && part.metadata?.magicCompact !== undefined
                && (part.metadata.magicCompact as Record<string, unknown>)[
                  "summary"
                ] === true,
            ),
          ),
      );
      const harness = generationHarness(turns, responseFor(candidates), {
        ...(nextTurn ? { nextTurn } : {}),
        modelImageCapability: false,
      });

      await expect(
        generateCompactionSummaries(
          harness.v2,
          sourceSession(),
          generationPlan(turns, nextTurn),
        ),
      ).rejects.toThrow("image-capable configured compaction model");
      expect(harness.forks).toEqual([]);
      expect(harness.prompts).toEqual([]);
      expect(source).toEqual(before);
    },
  );

  test("does not require vision when the only tool attachment is compacted", async () => {
    const turn = candidateTurn("compacted-attachment", "inspect attachment", [
      toolPart("compacted-attachment-tool", "compacted-attachment-assistant", {
        status: "completed",
        input: {},
        output: "attached result",
        title: "Attachment",
        metadata: {},
        time: { start: 1, end: 2, compacted: 3 },
        attachments: [{ mime: "image/png", url: "data:image/png;base64,Zm9v" }],
      }),
    ]);
    const harness = generationHarness([turn], responseFor([turn]), {
      modelImageCapability: false,
    });

    const summaries = await generateCompactionSummaries(
      harness.v2,
      sourceSession(),
      generationPlan([turn]),
    );

    expect(summaries).toHaveLength(1);
    expect(harness.forks.length).toBe(1);
    expect(harness.prompts.length).toBe(1);
  });

  test("keeps a completed compacted tool attachment non-visible and compacted-costed through the real plan path", async () => {
    const hiddenDataURL = `data:image/png;base64,${"A".repeat(2_000_000)}`;
    const messages = [
      ...candidateMessages("compacted-attachment", "inspect attachment", [
        toolPart(
          "compacted-attachment-tool",
          "compacted-attachment-assistant",
          {
            status: "completed",
            input: {},
            output: hiddenDataURL,
            title: "Attachment",
            metadata: {},
            time: { start: 1, end: 2, compacted: 3 },
            attachments: [{ mime: "image/png", url: hiddenDataURL }],
          },
        ),
      ]),
    ];
    const planClient = {
      session: { messages: async () => ({ data: messages }) },
    } as unknown as V2Client;
    const plan = await createCompactionPlan(planClient, "source", 0);
    const harness = generationHarness(
      plan.summarizedTurns,
      responseFor(plan.summarizedTurns),
      {
        sourceMessages: messages,
        modelImageCapability: false,
      },
    );

    const summaries = await generateCompactionSummaries(
      harness.v2,
      sourceSession(),
      plan,
    );

    expect(summaries).toHaveLength(1);
    expect(
      estimateNativeTurnCost(plan.summarizedTurns[0]!).inputTokens,
    ).toBeLessThan(10_000);
    expect(harness.forks.length).toBe(1);
    expect(harness.prompts.length).toBe(1);
  });

  test.each([
    ["missing", "missing"],
    ["disabled", "disabled"],
    ["without an explicit model", "model-missing"],
  ] as const)(
    "rejects a %s native compaction agent before fork creation",
    async (_name, compactionAgent) => {
      const turn = candidateTurn("configuration", "configure", []);
      const before = structuredClone(turn);
      const harness = generationHarness([turn], responseFor([turn]), {
        compactionAgent,
      });

      await expect(
        generateCompactionSummaries(
          harness.v2,
          sourceSession(),
          generationPlan([turn]),
        ),
      ).rejects.toThrow(
        compactionAgent === "model-missing"
          ? "requires an explicit model"
          : "native hidden compaction agent",
      );
      expect(harness.forks).toEqual([]);
      expect(harness.markerUpdates).toEqual([]);
      expect(harness.prompts).toEqual([]);
      expect(turn).toEqual(before);
    },
  );

  test("keeps persisted v0, v1, and v2 summary turns opaque and idempotent", async () => {
    const messages = [
      ...persistedSummaryTurn("v0", 0),
      ...persistedSummaryTurn("v1", 1),
      ...persistedSummaryTurn("v2", 2),
      ...candidateMessages("fresh", "fresh request", []),
    ];
    const before = structuredClone(messages.slice(0, 6));
    const planClient = {
      session: { messages: async () => ({ data: messages }) },
    } as unknown as V2Client;

    const plan = await createCompactionPlan(planClient, "source", 0);
    const harness = generationHarness(plan.summarizedTurns, {
      turns: [result(1, "fresh request", "Fresh turn summary.")],
    });
    const summaries = await generateCompactionSummaries(
      harness.v2,
      sourceSession(),
      plan,
    );

    expect(messages.slice(0, 6)).toEqual(before);
    expect(summaries.slice(0, 3)).toEqual([null, null, null]);
    expect(summaries[3]).toMatchObject({ text: "Fresh turn summary." });
    expect(harness.forks).toHaveLength(1);
    expect(harness.prompts).toHaveLength(1);
  });

  test("deletes the fork once after provider failure and gives trim-first guidance for native overflow", async () => {
    const turn = candidateTurn("overflow", "overflow request", []);
    const providerError = Object.assign(
      new Error("provider body must stay hidden"),
      {
        status: 413,
      },
    );
    const harness = generationHarness([turn], responseFor([turn]), {
      providerError,
    });

    await expect(
      generateCompactionSummaries(
        harness.v2,
        sourceSession(),
        generationPlan([turn]),
      ),
    ).rejects.toThrow(/trim.*first/i);

    expect(harness.aborts).toEqual(["generation-fork"]);
    expect(harness.deletes).toEqual(["generation-fork"]);
    expect(harness.prompts).toHaveLength(1);
  });

  test("waits for the model without applying the control-operation deadline", async () => {
    const turn = candidateTurn("slow", "slow request", []);
    const harness = generationHarness([turn], responseFor([turn]), {
      providerDelayMs: 20,
    });

    const summaries = await generateCompactionSummaries(
      harness.v2,
      sourceSession(),
      generationPlan([turn]),
      { controlTimeoutMs: 1 },
    );

    expect(summaries[0]).toMatchObject({
      text: "Turn 1 outcome and continuation details.",
    });
    expect(harness.aborts).toEqual([]);
    expect(harness.deletes).toEqual(["generation-fork"]);
  });

  test("cleans a generation fork that completes after the creation deadline", async () => {
    const turn = candidateTurn("late-fork", "late fork request", []);
    let releaseFork!: () => void;
    const forkGate = new Promise<void>(resolve => {
      releaseFork = resolve;
    });
    const harness = generationHarness([turn], responseFor([turn]), {
      forkGate,
    });

    await expect(
      generateCompactionSummaries(
        harness.v2,
        sourceSession(),
        generationPlan([turn]),
        { controlTimeoutMs: 1 },
      ),
    ).rejects.toThrow("fork creation timed out");
    releaseFork();
    await waitFor(() => harness.deletes.length === 1);

    expect(harness.prompts).toEqual([]);
    expect(harness.aborts).toEqual(["generation-fork"]);
    expect(harness.deletes).toEqual(["generation-fork"]);
  });

  test.each([
    ["finish length", { responseFinish: "length" }],
    ["finish output-length", { responseFinish: "output_length" }],
    ["finish max-token", { responseFinish: "max_token" }],
    [
      "output length error",
      {
        responseError: {
          name: "MessageOutputLengthError",
          data: { raw: "must stay hidden" },
        },
      },
    ],
    [
      "nested max-output-token error",
      {
        responseError: {
          error: {
            cause: {
              code: "max_output_tokens",
              data: { raw: "must stay hidden" },
            },
          },
        },
      },
    ],
  ])(
    "surfaces higher-output-capacity guidance for a single-turn provider %s signal",
    async (_name, options) => {
      const turn = candidateTurn("truncated", "truncate request", []);
      const harness = generationHarness([turn], { turns: [] }, options);

      await expect(
        generateCompactionSummaries(
          harness.v2,
          sourceSession(),
          generationPlan([turn]),
        ),
      ).rejects.toThrow(/single turn.*more output capacity/i);
      expect(harness.aborts).toEqual(["generation-fork"]);
      expect(harness.deletes).toEqual(["generation-fork"]);
    },
  );

  test("keeps larger-N guidance when a multi-turn batch reaches the provider output limit", async () => {
    const turns = [
      candidateTurn("multi-output-one", "one", []),
      candidateTurn("multi-output-two", "two", []),
    ];
    const harness = generationHarness(turns, responseFor(turns), {
      responseFinish: "length",
    });

    await expect(
      generateCompactionSummaries(
        harness.v2,
        sourceSession(),
        generationPlan(turns),
      ),
    ).rejects.toThrow(/larger N.*fewer turns/i);
  });

  test("preserves trim-first overflow guidance through one sanitized nested cause", async () => {
    const turn = candidateTurn("nested-overflow", "overflow request", []);
    const providerError = new Error("outer raw error", {
      cause: { name: "ContextOverflowError", data: { message: "raw inner" } },
    });
    const harness = generationHarness([turn], responseFor([turn]), {
      providerError,
    });

    await expect(
      generateCompactionSummaries(
        harness.v2,
        sourceSession(),
        generationPlan([turn]),
      ),
    ).rejects.toThrow(/trim.*first/i);
  });

  test.each([
    ["context window", "context_window_exceeded"],
    ["prompt too long", "prompt_too_long"],
    ["input too long", "input_too_long"],
  ])(
    "recognizes deeply nested %s overflow without surfacing provider text",
    async (_name, code) => {
      const turn = candidateTurn("deep-overflow", "overflow request", []);
      const providerError = {
        error: {
          cause: {
            data: {
              error: { code, message: "raw provider text must stay hidden" },
            },
          },
        },
      } as unknown as Error;
      const harness = generationHarness([turn], responseFor([turn]), {
        providerError,
      });
      let failure: unknown;

      try {
        await generateCompactionSummaries(
          harness.v2,
          sourceSession(),
          generationPlan([turn]),
        );
      } catch (error) {
        failure = error;
      }

      expect(String(failure)).toMatch(/trim.*first/i);
      expect(nestedErrorMessages(failure).join(" ")).not.toContain(
        "raw provider text",
      );
      expect(harness.aborts).toEqual(["generation-fork"]);
      expect(harness.deletes).toEqual(["generation-fork"]);
    },
  );

  test.each([
    ["permission update", { permissionError: new Error("raw permission") }],
    ["delete after success", { deleteError: new Error("raw delete") }],
  ])(
    "sanitizes %s failure and retains route context",
    async (name, options) => {
      const turn = candidateTurn("lifecycle", "lifecycle request", []);
      const harness = generationHarness([turn], responseFor([turn]), options);

      await expect(
        generateCompactionSummaries(
          harness.v2,
          sourceSession(),
          generationPlan([turn]),
        ),
      ).rejects.toThrow(
        name === "permission update"
          ? "permission update failed"
          : "fork deletion failed",
      );
      expect(harness.prompts).toHaveLength(
        name === "permission update" ? 0 : 1,
      );
      expect(harness.forks[0]).toMatchObject({
        directory: "/workspace",
        workspace: "workspace",
      });
      expect(harness.deleteRequests[0]).toMatchObject({
        directory: "/workspace",
        workspace: "workspace",
      });
    },
  );

  test("aggregates sanitized provider, abort, and delete failures", async () => {
    const turn = candidateTurn("cleanup", "cleanup request", []);
    const harness = generationHarness([turn], responseFor([turn]), {
      providerError: new Error("raw provider"),
      abortError: new Error("raw abort"),
      deleteError: new Error("raw delete"),
    });
    let failure: unknown;

    try {
      await generateCompactionSummaries(
        harness.v2,
        sourceSession(),
        generationPlan([turn]),
      );
    } catch (error) {
      failure = error;
    }

    expect(failure).toBeInstanceOf(AggregateError);
    expect(nestedErrorMessages(failure)).toEqual(
      expect.arrayContaining([
        "Compaction generation model request failed.",
        "Compaction generation fork abort failed.",
        "Compaction generation fork deletion failed.",
      ]),
    );
    expect(nestedErrorMessages(failure).join(" ")).not.toContain("raw");
    expect(harness.abortRequests[0]).toMatchObject({
      directory: "/workspace",
      workspace: "workspace",
    });
    expect(harness.deleteRequests[0]).toMatchObject({
      directory: "/workspace",
      workspace: "workspace",
    });
  });
});

describe("compaction orchestration", () => {
  test.each([
    [
      "malformed",
      () => [
        ...candidateMessages("checkpoint-malformed", "before", []),
        message("checkpoint-malformed-assistant", "assistant", [
          nativeCompactionPart(
            "checkpoint-malformed-part",
            "checkpoint-malformed-assistant",
          ),
        ]),
      ],
    ],
    [
      "ambiguous",
      () => [
        ...candidateMessages("checkpoint-ambiguous", "before", []),
        nativeMarker("checkpoint-ambiguous-marker"),
        nativeSummary(
          "checkpoint-ambiguous-complete",
          "checkpoint-ambiguous-marker",
        ),
        nativeSummary(
          "checkpoint-ambiguous-incomplete",
          "checkpoint-ambiguous-marker",
          { finish: undefined },
        ),
      ],
    ],
    [
      "dangling",
      () => [
        ...candidateMessages("checkpoint-dangling", "before", []),
        nativeMarker("checkpoint-dangling-marker"),
      ],
    ],
    [
      "active incomplete",
      () => [
        ...candidateMessages("checkpoint-active", "before", []),
        nativeMarker("checkpoint-active-marker"),
        nativeSummary("checkpoint-active-summary", "checkpoint-active-marker", {
          finish: undefined,
        }),
      ],
    ],
  ] as const)("rejects %s native checkpoint state", async (_name, fixture) => {
    const messages = fixture();
    const v2 = {
      session: { messages: async () => ({ data: messages }) },
    } as unknown as V2Client;

    await expect(createCompactionPlan(v2, "source", 0)).rejects.toThrow();
  });

  test("keeps native checkpoint history frozen and selects only its suffix", async () => {
    const messages = [
      ...candidateMessages("prefix", "prefix request", []),
      message("native-marker", "user", [
        {
          id: "native-marker-part",
          sessionID: "source",
          messageID: "native-marker",
          type: "compaction",
          auto: true,
          tail_start_id: "prefix-assistant",
        } as unknown as Part,
      ]),
      {
        info: {
          id: "native-summary",
          role: "assistant",
          parentID: "native-marker",
          summary: true,
          finish: "stop",
        } as Message,
        parts: [
          textPart("native-summary-text", "native-summary", "checkpoint"),
        ],
      },
      ...candidateMessages("suffix", "suffix request", []),
    ];
    const v2 = {
      session: { messages: async () => ({ data: messages }) },
    } as unknown as V2Client;

    const plan = await createCompactionPlan(v2, "source", 0);

    expect(plan.summarizedTurns).toHaveLength(1);
    expect(plan.summarizedTurns[0]!.user[0]!.info.id).toBe("suffix-user");
    expect(plan.nextTurn).toBeNull();
  });

  test("treats a concurrent compacted-bit change as source drift", async () => {
    const messages = [
      ...candidateMessages("drift-compacted", "inspect", [
        toolPart("drift-compacted-tool", "drift-compacted-assistant", {
          status: "completed",
          input: {},
          output: "result",
          title: "Read",
          metadata: {},
          time: { start: 1, end: 2 },
        }),
      ]),
    ];
    const planClient = {
      session: { messages: async () => ({ data: messages }) },
    } as unknown as V2Client;
    const plan = await createCompactionPlan(planClient, "source", 0);

    const tool = messages[1]!.parts.find(
      part => part.type === "tool",
    ) as unknown as { state: { time: Record<string, unknown> } };
    tool.state.time.compacted = 3;

    expect(() => assertCompactionSourceUnchanged(messages, plan)).toThrow(
      NativeCheckpointChangedError,
    );
  });

  test("ignores volatile tool timing changes in source revalidation", async () => {
    const messages = [
      ...candidateMessages("drift-timing", "inspect", [
        toolPart("drift-timing-tool", "drift-timing-assistant", {
          status: "completed",
          input: {},
          output: "result",
          title: "Read",
          metadata: {},
          time: { start: 1, end: 2, compacted: 3 },
        }),
      ]),
    ];
    const planClient = {
      session: { messages: async () => ({ data: messages }) },
    } as unknown as V2Client;
    const plan = await createCompactionPlan(planClient, "source", 0);

    const tool = messages[1]!.parts.find(
      part => part.type === "tool",
    ) as unknown as { state: { time: Record<string, unknown> } };
    tool.state.time.start = 100;
    tool.state.time.end = 200;

    expect(() => assertCompactionSourceUnchanged(messages, plan)).not.toThrow();
  });

  test("keeps unprofitable turns exact while compacting profitable turns", async () => {
    const small = [
      message("mixed-small-user", "user", [
        textPart("mixed-small-user-text", "mixed-small-user", "small"),
      ]),
      message("mixed-small-assistant", "assistant", [
        textPart("mixed-small-text", "mixed-small-assistant", "ok"),
      ]),
    ];
    const large = compactableCommandMessages(
      "mixed-large",
      "large tool output ".repeat(1_000),
    );
    const messages = [...small, ...large];
    const smallBefore = structuredClone(small);
    const harness = commandHarness(messages, {
      generationResponse: {
        turns: [
          result(
            1,
            "small",
            "A generated summary that is longer than the exact tiny response.",
          ),
          result(2, "mixed-large request", "Completed the large request."),
        ],
      },
    });

    await expect(
      executeMagicCompact(harness.v2, "source", 0),
    ).resolves.toBeTrue();

    expect(messages.slice(0, 2)).toEqual(smallBefore);
    expect(summaryParts(messages).map(part => part.messageID)).toEqual([
      "mixed-large-assistant",
    ]);
  });

  test("pure no-eligible compaction creates no fork, backup, or mutation", async () => {
    const messages = candidateMessages("no-eligible", "keep current", []);
    const before = structuredClone(messages);
    const harness = commandHarness(messages, {
      generationResponse: { turns: [] },
    });

    await expect(
      executeMagicCompact(harness.v2, "source", 1),
    ).resolves.toBeFalse();

    expect(harness.forks).toEqual([]);
    expect(harness.partUpdates).toEqual([]);
    expect(harness.deletedSessions).toEqual([]);
    expect(harness.boundaryInserted).toBeFalse();
    expect(messages).toEqual(before);
  });

  test("injectSummaries refuses a missing summary before writing", async () => {
    const turn = candidateTurn("missing-summary", "inject", []);
    let writes = 0;
    const v2 = {
      part: {
        update: async () => {
          writes += 1;
          return { data: true };
        },
      },
    } as unknown as V2Client;

    await expect(injectSummaries(v2, "source", [turn], [])).rejects.toThrow(
      "Missing summary",
    );
    expect(writes).toBe(0);
  });

  test.each(["hang", "fail"] as const)(
    "toast %s never replaces no-op, success, or failure outcomes",
    async toastMode => {
      const noOp = commandHarness(
        candidateMessages(`toast-${toastMode}-noop`, "keep", []),
        { generationResponse: { turns: [] }, toastMode },
      );
      await expect(
        settleWithin(
          executeMagicCompact(noOp.v2, "source", 1, {
            controlTimeoutMs: 5,
          }),
          250,
        ),
      ).resolves.toBeFalse();

      const successMessages = compactableCommandMessages(
        `toast-${toastMode}-success`,
        "large output ".repeat(1_000),
      );
      const success = commandHarness(successMessages, {
        generationResponse: {
          turns: [
            result(
              1,
              `toast-${toastMode}-success request`,
              "Completed toast candidate.",
            ),
          ],
        },
        toastMode,
      });
      await expect(
        settleWithin(
          executeMagicCompact(success.v2, "source", 0, {
            controlTimeoutMs: 5,
          }),
          250,
        ),
      ).resolves.toBeTrue();
      expect(summaryParts(successMessages)).toHaveLength(1);

      const failed = commandHarness(
        compactableCommandMessages(
          `toast-${toastMode}-failure`,
          "large output ".repeat(1_000),
        ),
        { generationResponse: { turns: [] }, toastMode },
      );
      let failure: unknown;
      try {
        await settleWithin(
          executeMagicCompact(failed.v2, "source", 0, {
            controlTimeoutMs: 5,
          }),
          250,
        );
      } catch (error) {
        failure = error;
      }
      expect(failure).toBeInstanceOf(Error);
      expect((failure as Error).message).toBe(
        "Compaction generation response validation failed.",
      );
      expect(nestedErrorMessages(failure).join(" ")).not.toContain(
        "toast failed",
      );
    },
  );

  test("prunes exactly, persists retrievable omissions and stats, and verifies final projection", async () => {
    const output = "large tool output ".repeat(1_000);
    const messages = compactableCommandMessages("success", output);
    const assistantImage = imagePart(
      "success-assistant-image",
      "success-assistant",
      `data:image/png;base64,${"A".repeat(4_096)}`,
    );
    messages[1]!.parts.push(assistantImage);
    const assistantImageBefore = structuredClone(assistantImage);
    const harness = commandHarness(messages, {
      generationResponse: {
        turns: [
          result(
            1,
            "success request",
            "Completed the requested work.\n\nProcess and decisions were preserved; verification passed.",
            [{ ordinal: 1, description: "The completed result is visible." }],
          ),
        ],
      },
    });

    await expect(
      executeMagicCompact(harness.v2, "source", 0),
    ).resolves.toBeTrue();

    expect(harness.forks).toEqual(["generation-fork", "backup"]);
    expect(harness.events.indexOf("generation-delete")).toBeLessThan(
      harness.events.indexOf("backup-fork"),
    );
    const summary = summaryParts(messages)[0]!;
    expect(summary).toMatchObject({
      id: "prt_-magic_capsule_v3_success-assistant",
      metadata: { magicCompact: { summary: true, version: 3 } },
    });
    expect(
      messages
        .flatMap(item => item.parts)
        .some(part => part.type === "reasoning"),
    ).toBeFalse();
    const tool = messages
      .flatMap(item => item.parts)
      .find(
        (part): part is Extract<Part, { type: "tool" }> => part.type === "tool",
      );
    expect(tool).toBeDefined();
    const persistedImage = messages
      .find(message => message.info.id === "success-assistant")
      ?.parts.find(part => part.id === assistantImage.id);
    expect(persistedImage).toEqual(assistantImageBefore);
    expect(harness.partDeletes).not.toContainEqual({
      messageID: "success-assistant",
      partID: assistantImage.id,
    });
    expect(
      projectManifestCoveredMedia(messages)
        .find(message => message.info.id === "success-assistant")
        ?.parts.find(part => part.id === assistantImage.id),
    ).toMatchObject({ type: "text", synthetic: true });
    const serialized = JSON.stringify(tool!.state);
    const contentID = serialized.match(
      /[A-Za-z0-9_-]{1,12}:omitted-[A-Za-z0-9_-]{22}/u,
    )?.[0];
    expect(contentID).toBeDefined();
    expect(await readOmittedContent("source", contentID!)).toBe(output);
    expect(JSON.stringify(await readCache("source"))).not.toContain(
      assistantImage.url,
    );
    expect(boundaryParts(messages)).toHaveLength(1);
    const stats = await readStats("source");
    expect(stats?.totalTokensPruned).toBeGreaterThan(0);
    expect(stats?.processedAssistantMessageIds).toEqual([]);
    expect(harness.selectedSessions).toEqual([]);
    expect(harness.deletedSessions).toEqual(["generation-fork"]);
  });

  test("keeps small user and assistant inline images inline while compacting substantial reasoning", async () => {
    const userImage = imagePart(
      "small-media-user-image",
      "small-media-user",
      "data:image/png;base64,Zm9v",
    );
    const assistantImage = imagePart(
      "small-media-assistant-image",
      "small-media-assistant",
      "data:image/jpeg;base64,YmFy",
      "image/jpeg",
    );
    const messages = compactableCommandMessages("small-media", "output");
    messages[0]!.parts.push(userImage);
    messages[1]!.parts.push(assistantImage);
    const userImageBefore = structuredClone(userImage);
    const assistantImageBefore = structuredClone(assistantImage);
    const harness = commandHarness(messages, {
      generationResponse: {
        turns: [
          result(1, "small-media request", "Completed the request.", [
            { ordinal: 1, description: "A user-provided screenshot." },
            { ordinal: 2, description: "An assistant-provided result image." },
          ]),
        ],
      },
    });

    await expect(
      executeMagicCompact(harness.v2, "source", 0),
    ).resolves.toBeTrue();

    const persistedUserImage = messages
      .find(message => message.info.id === "small-media-user")
      ?.parts.find(part => part.id === userImage.id);
    const persistedAssistantImage = messages
      .find(message => message.info.id === "small-media-assistant")
      ?.parts.find(part => part.id === assistantImage.id);
    expect(persistedUserImage).toEqual(userImageBefore);
    expect(persistedAssistantImage).toEqual(assistantImageBefore);
    const summary = summaryParts(messages)[0]!;
    expect(summary.metadata).toMatchObject({
      magicCompact: {
        summary: true,
        version: 3,
        mediaManifest: {
          version: 1,
          images: [
            {
              ordinal: 1,
              mime: "image/png",
              urlSha256: createHash("sha256")
                .update(userImage.url)
                .digest("hex"),
            },
            {
              ordinal: 2,
              mime: "image/jpeg",
              urlSha256: createHash("sha256")
                .update(assistantImage.url)
                .digest("hex"),
            },
          ],
        },
      },
    });
    expect(harness.partDeletes).not.toContainEqual({
      messageID: "small-media-assistant",
      partID: assistantImage.id,
    });
    const projected = projectManifestCoveredMedia(messages);
    expect(
      projected
        .find(message => message.info.id === "small-media-user")
        ?.parts.find(part => part.id === userImage.id),
    ).toMatchObject({ type: "file", url: userImage.url });
    expect(
      projected
        .find(message => message.info.id === "small-media-assistant")
        ?.parts.find(part => part.id === assistantImage.id),
    ).toMatchObject({ type: "file", url: assistantImage.url });
  });

  test("rejects an unprofitable generated summary before backup or source writes", async () => {
    const messages = [
      message("small-user", "user", [
        textPart("small-user-text", "small-user", "small request"),
      ]),
      message("small-assistant", "assistant", [
        textPart("small-assistant-text", "small-assistant", "ok"),
      ]),
    ];
    const harness = commandHarness(messages, {
      generationResponse: {
        turns: [
          result(
            1,
            "small request",
            "A longer generated summary that cannot save tokens for this tiny turn.",
          ),
        ],
      },
    });

    await expect(executeMagicCompact(harness.v2, "source", 0)).rejects.toThrow(
      COMPACT_INSUFFICIENT_SAVINGS,
    );

    expect(harness.forks).toEqual(["generation-fork"]);
    expect(harness.partUpdates).toEqual([]);
    expect(boundaryParts(messages)).toEqual([]);
    expect(await readStats("source")).toBeNull();
  });

  test("rejects a malformed all-turn response before backup or source writes", async () => {
    const messages = [
      ...compactableCommandMessages("batch-one", "first output"),
      ...compactableCommandMessages("batch-two", "second output"),
    ];
    const harness = commandHarness(messages, {
      generationResponse: {
        turns: [
          result(1, "batch-one request", "Only one summary was returned."),
        ],
      },
    });

    await expect(executeMagicCompact(harness.v2, "source", 0)).rejects.toThrow(
      "response validation failed",
    );

    expect(harness.forks).toEqual(["generation-fork"]);
    expect(harness.partUpdates).toEqual([]);
    expect(boundaryParts(messages)).toEqual([]);
    expect(harness.selectedSessions).toEqual([]);
    expect(harness.deletedSessions).toEqual(["generation-fork"]);
  });

  test("discards completed batches when a later batch fails before backup, source writes, or stats", async () => {
    const messages = Array.from({ length: 13 }, (_, index) =>
      compactableCommandMessages(
        `atomic-batch-${index + 1}`,
        `large output ${index + 1} `.repeat(1_000),
      ),
    ).flat();
    const before = structuredClone(messages);
    const harness = commandHarness(messages, {
      generationResponse: {
        turns: Array.from({ length: 13 }, (_, index) =>
          result(
            index + 1,
            `atomic-batch-${index + 1} request`,
            `Summary ${index + 1}.`,
          ),
        ),
      },
      generationErrorForFirstOrdinal: 13,
    });

    await expect(executeMagicCompact(harness.v2, "source", 0)).rejects.toThrow(
      "model request failed",
    );

    expect(messages).toEqual(before);
    expect(harness.forks).toEqual(["generation-fork", "generation-fork-2"]);
    expect(harness.deletedSessions).toEqual(
      expect.arrayContaining(["generation-fork", "generation-fork-2"]),
    );
    expect(harness.forks).not.toContain("backup");
    expect(harness.partUpdates).toEqual([]);
    expect(boundaryParts(messages)).toEqual([]);
    expect(await readStats("source")).toBeNull();
  });

  test("aggregates every active batch failure in batch order before any durable mutation", async () => {
    const messages = Array.from({ length: 13 }, (_, index) =>
      compactableCommandMessages(
        `aggregate-batch-${index + 1}`,
        `large output ${index + 1} `.repeat(1_000),
      ),
    ).flat();
    const before = structuredClone(messages);
    const harness = commandHarness(messages, {
      generationResponse: {
        turns: Array.from({ length: 13 }, (_, index) =>
          result(
            index + 1,
            `aggregate-batch-${index + 1} request`,
            `Summary ${index + 1}.`,
          ),
        ),
      },
      generationErrorsByFirstOrdinal: {
        1: new Error("raw first provider payload"),
        13: new Error("raw second provider payload"),
      },
      generationAbortErrorForFirstOrdinal: 13,
      generationDeleteErrorForFirstOrdinal: 13,
      holdGenerationPromptsUntil: 2,
    });
    let failure: unknown;

    try {
      await executeMagicCompact(harness.v2, "source", 0);
    } catch (error) {
      failure = error;
    }

    expect(failure).toBeInstanceOf(AggregateError);
    const failures = (failure as AggregateError).errors as Error[];
    expect(failures).toHaveLength(2);
    expect(nestedErrorMessages(failures[0])).toContain(
      "Compaction generation model request failed.",
    );
    expect(nestedErrorMessages(failures[1])).toEqual(
      expect.arrayContaining([
        "Compaction generation model request failed.",
        "Compaction generation fork abort failed.",
        "Compaction generation fork deletion failed.",
      ]),
    );
    expect(nestedErrorMessages(failure).join(" ")).not.toContain("raw");
    expect(messages).toEqual(before);
    expect(harness.forks).toEqual(["generation-fork", "generation-fork-2"]);
    expect(harness.deletedSessions).toEqual(
      expect.arrayContaining(["generation-fork", "generation-fork-2"]),
    );
    expect(harness.activeGenerationForks).toEqual(["generation-fork-2"]);
    expect(harness.partUpdates).toEqual([]);
    expect(boundaryParts(messages)).toEqual([]);
    expect(await readStats("source")).toBeNull();
    expect(await readStats("backup")).toBeNull();
  });

  test("rejects an active marker leaked into a summary before backup or source writes", async () => {
    const messages = compactableCommandMessages("summary-leak", "output");
    const before = structuredClone(messages);
    const harness = commandHarness(messages, {
      generationResponse: {
        turns: [result(1, "summary-leak request", "Clean summary.")],
      },
      transformGenerationResponse: response => ({
        turns: response.turns.map(turn => ({
          ...turn,
          summary: `Leaked active marker: ${turn.marker}`,
        })),
      }),
    });

    await expect(executeMagicCompact(harness.v2, "source", 0)).rejects.toThrow(
      "response validation failed",
    );

    expect(messages).toEqual(before);
    expect(harness.forks).toEqual(["generation-fork"]);
    expect(harness.abortedSessions).toEqual(["generation-fork"]);
    expect(harness.selectedSessions).toEqual([]);
    expect(harness.deletedSessions).toEqual(["generation-fork"]);
    expect(harness.partUpdates).toEqual([]);
    expect(boundaryParts(messages)).toEqual([]);
    expect(await readStats("source")).toBeNull();
    expect(await readStats("backup")).toBeNull();
  });

  test("rejects an active nonce leaked into a visual description before backup or source writes", async () => {
    const messages = compactableCommandMessages("visual-leak", "output");
    messages[0]!.parts.push(
      imagePart(
        "visual-leak-image",
        "visual-leak-user",
        "data:image/png;base64,Zm9v",
      ),
    );
    const before = structuredClone(messages);
    const harness = commandHarness(messages, {
      generationResponse: {
        turns: [
          result(1, "visual-leak request", "Clean summary.", [
            { ordinal: 1, description: "Clean visual description." },
          ]),
        ],
      },
      transformGenerationResponse: response => ({
        turns: response.turns.map(turn => {
          const nonce = turn.marker?.match(/nonce="([A-Za-z0-9_-]{22})"/u)?.[1];
          if (!nonce) throw new Error("Expected active generation nonce.");
          return {
            ...turn,
            visualDescriptions: [
              { ordinal: 1, description: `Leaked active nonce: ${nonce}` },
            ],
          };
        }),
      }),
    });

    await expect(executeMagicCompact(harness.v2, "source", 0)).rejects.toThrow(
      "response validation failed",
    );

    expect(messages).toEqual(before);
    expect(harness.forks).toEqual(["generation-fork"]);
    expect(harness.abortedSessions).toEqual(["generation-fork"]);
    expect(harness.selectedSessions).toEqual([]);
    expect(harness.deletedSessions).toEqual(["generation-fork"]);
    expect(harness.partUpdates).toEqual([]);
    expect(boundaryParts(messages)).toEqual([]);
    expect(await readStats("source")).toBeNull();
    expect(await readStats("backup")).toBeNull();
  });

  test.each([
    [
      "summary insertion",
      "summary-insertion",
      { failSummaryUpdate: true },
      "summary insertion failed",
    ],
    [
      "boundary insertion",
      "boundary-insertion",
      { boundaryInsertError: new Error("boundary insertion failed") },
      "boundary insertion failed",
    ],
    [
      "pruning",
      "pruning",
      { partDeleteError: new Error("pruning failed") },
      "pruning failed",
    ],
    [
      "metadata update",
      "metadata-update",
      { sourceMetadataUpdateError: new Error("metadata update failed") },
      "metadata update failed",
    ],
    [
      "stats notice insertion",
      "stats-notice",
      { statsNoticeError: new Error("stats notice failed") },
      "stats notice failed",
    ],
  ] as const)(
    "preserves both sessions and concurrent input after %s begins source writes",
    async (_stage, id, fault, rawFailure) => {
      const messages = compactableCommandMessages(
        `post-write-${id}`,
        "large output ".repeat(1_000),
      );
      await writeStats("source", conversationStats({ totalTokensPruned: 17 }));
      const harness = commandHarness(messages, {
        generationResponse: {
          turns: [
            result(
              1,
              `post-write-${id} request`,
              "Completed the post-write failure candidate.",
            ),
          ],
        },
        appendConcurrentOnFailure: true,
        ...fault,
      });
      let failure: unknown;

      try {
        await executeMagicCompact(harness.v2, "source", 0);
      } catch (error) {
        failure = error;
      }

      expect(failure).toBeInstanceOf(Error);
      expect(String(failure)).toContain("source writes began");
      expect(String(failure)).toContain("source session source");
      expect(String(failure)).toContain("backup session backup");
      expect(String(failure)).toContain("recover manually");
      expect(String(failure)).not.toContain(rawFailure);
      expect(harness.backupMessages).toBeDefined();
      expect(harness.selectedSessions).toEqual([]);
      expect(harness.deletedSessions).toEqual(["generation-fork"]);
      expect(
        messages.find(message => message.info.id === "concurrent-user")?.parts,
      ).toEqual([
        textPart(
          "concurrent-user-text",
          "concurrent-user",
          "concurrent message appended after backup",
        ),
      ]);
      expect(await readStats("source")).toMatchObject({
        totalTokensPruned: 17,
      });
      expect(await readStats("backup")).toMatchObject({
        sourceSessionId: "source",
        totalTokensPruned: 17,
      });
    },
  );

  test("preserves source and backup when source drifts after backup and before writes", async () => {
    const messages = compactableCommandMessages("race", "output");
    const harness = commandHarness(messages, {
      generationResponse: {
        turns: [result(1, "race request", "Completed race candidate.")],
      },
      driftAfterBackup: true,
    });

    await expect(executeMagicCompact(harness.v2, "source", 0)).rejects.toThrow(
      "OpenCode source changed",
    );

    expect(harness.forks).toEqual(["generation-fork", "backup"]);
    expect(harness.selectedSessions).toEqual([]);
    expect(harness.deletedSessions).toEqual(["generation-fork"]);
    expect(summaryParts(messages)).toEqual([]);
  });

  test("preserves source, backup, and a concurrent user message on post-insertion drift", async () => {
    const messages = compactableCommandMessages(
      "post-insertion-race",
      "output",
    );
    const harness = commandHarness(messages, {
      generationResponse: {
        turns: [
          result(
            1,
            "post-insertion-race request",
            "Completed post-insertion race candidate.",
          ),
        ],
      },
      driftAfterInsertions: true,
    });
    let failure: unknown;

    try {
      await executeMagicCompact(harness.v2, "source", 0);
    } catch (error) {
      failure = error;
    }

    expect(failure).toBeInstanceOf(Error);
    expect(String(failure)).toContain("source session source");
    expect(String(failure)).toContain("backup session backup");
    expect(String(failure)).toContain("recover manually");
    expect(String(failure)).not.toContain(
      "concurrent user message must survive",
    );
    expect(harness.forks).toEqual(["generation-fork", "backup"]);
    expect(harness.backupMessages).toBeDefined();
    expect(harness.selectedSessions).toEqual([]);
    expect(harness.deletedSessions).toEqual(["generation-fork"]);
    expect(harness.partDeletes).toEqual([]);
    expect(harness.messageDeletes).toEqual([]);
    expect(harness.statsInserted).toBeFalse();
    expect(summaryParts(messages)).toHaveLength(1);
    expect(boundaryParts(messages)).toHaveLength(1);
    expect(
      messages.find(
        message => message.info.id === "post-insertion-external-user",
      )?.parts,
    ).toEqual([
      textPart(
        "post-insertion-external-text",
        "post-insertion-external-user",
        "concurrent user message must survive",
      ),
    ]);
    expect(
      messages
        .find(message => message.info.id === "post-insertion-race-assistant")
        ?.parts.some(part => part.type === "reasoning"),
    ).toBeTrue();
    expect(await readStats("source")).toBeNull();
    expect(await readStats("backup")).toEqual(
      conversationStats({ sourceSessionId: "source" }),
    );
  });

  test("preserves the partially mutated source and backup on final projection drift", async () => {
    const messages = compactableCommandMessages("late-race", "output");
    const harness = commandHarness(messages, {
      generationResponse: {
        turns: [
          result(1, "late-race request", "Completed late race candidate."),
        ],
      },
      driftBeforeFinalCount: true,
    });

    let failure: unknown;
    try {
      await executeMagicCompact(harness.v2, "source", 0);
    } catch (error) {
      failure = error;
    }

    expect(failure).toBeInstanceOf(Error);
    expect(String(failure)).toContain("source writes began");
    expect(String(failure)).toContain("source session source");
    expect(String(failure)).toContain("backup session backup");
    expect(String(failure)).toContain("recover manually");
    expect(String(failure)).not.toContain("late external change");
    expect(harness.backupMessages).toBeDefined();
    expect(harness.selectedSessions).toEqual([]);
    expect(harness.deletedSessions).toEqual(["generation-fork"]);
    expect(summaryParts(messages)).toHaveLength(1);
    expect(
      messages.find(message => message.info.id === "late-external-user")?.parts,
    ).toEqual([
      textPart(
        "late-external-text",
        "late-external-user",
        "late external change",
      ),
    ]);
    expect(await readStats("source")).toBeNull();
    expect(await readStats("backup")).toEqual(
      conversationStats({ sourceSessionId: "source" }),
    );
  });

  test("cleans a partially initialized backup and aggregates cleanup failures", async () => {
    const seedStats = conversationStats();
    await writeStats("source", seedStats);
    await allocateOmission("source", { content: "seed omission" }, () =>
      "A".repeat(22),
    );
    const deletedSessions: string[] = [];
    const cleanClient = {
      session: {
        fork: async () => ({ data: { ...sourceSession(), id: "backup" } }),
        update: async () => {
          throw new Error("backup metadata update failed");
        },
        delete: async (request: { sessionID: string }) => {
          deletedSessions.push(request.sessionID);
          return { data: true };
        },
      },
    } as unknown as V2Client;

    await expect(createBackup(cleanClient, sourceSession(), 1)).rejects.toThrow(
      "backup metadata update failed",
    );
    expect(deletedSessions).toEqual(["backup"]);
    expect(await readCache("backup")).toBeNull();
    expect(await readStats("backup")).toBeNull();

    const backupCachePath = cachePath("backup");
    const backupStatsPath = `${pluginStorageDirectory()}/stats/backup.json`;
    const aggregateClient = {
      session: {
        fork: async () => ({ data: { ...sourceSession(), id: "backup" } }),
        update: async () => {
          await rm(backupCachePath, { force: true });
          await mkdir(backupCachePath, { recursive: true });
          await rm(backupStatsPath, { force: true });
          await mkdir(backupStatsPath, { recursive: true });
          throw new Error("backup metadata update failed");
        },
        delete: async () => {
          throw new Error("partial backup delete failed");
        },
      },
    } as unknown as V2Client;
    let failure: unknown;
    try {
      await createBackup(aggregateClient, sourceSession(), 1);
    } catch (error) {
      failure = error;
    } finally {
      await rm(backupCachePath, { recursive: true, force: true });
      await rm(backupStatsPath, { recursive: true, force: true });
    }

    expect(failure).toBeInstanceOf(AggregateError);
    expect(nestedErrorMessages(failure)).toEqual(
      expect.arrayContaining([
        "backup metadata update failed",
        "Failed to delete partial backup omission cache.",
        "Failed to delete partial backup stats cache.",
        "partial backup delete failed",
      ]),
    );
  });

  test("preserves both sessions and concurrent input when the final stats write fails", async () => {
    const statsPath = `${pluginStorageDirectory()}/stats/source.json`;
    const messages = compactableCommandMessages("stats-fault", "large output");
    await writeStats("source", conversationStats({ totalTokensPruned: 23 }));
    const harness = commandHarness(messages, {
      generationResponse: {
        turns: [result(1, "stats-fault request", "Completed stats candidate.")],
      },
      beforeFinalCount: async () => {
        messages.push(
          message("concurrent-user", "user", [
            textPart(
              "concurrent-user-text",
              "concurrent-user",
              "concurrent message appended after backup",
            ),
          ]),
        );
        await rm(statsPath, { recursive: true, force: true });
        await mkdir(statsPath, { recursive: true });
      },
    });

    try {
      let failure: unknown;
      try {
        await executeMagicCompact(harness.v2, "source", 0);
      } catch (error) {
        failure = error;
      }
      expect(failure).toBeInstanceOf(Error);
      expect(String(failure)).toContain("source writes began");
      expect(String(failure)).toContain("source session source");
      expect(String(failure)).toContain("backup session backup");
      expect(String(failure)).toContain("recover manually");
      expect(harness.backupMessages).toBeDefined();
      expect(harness.selectedSessions).toEqual([]);
      expect(harness.deletedSessions).toEqual(["generation-fork"]);
      expect(
        messages.find(message => message.info.id === "concurrent-user")?.parts,
      ).toEqual([
        textPart(
          "concurrent-user-text",
          "concurrent-user",
          "concurrent message appended after backup",
        ),
      ]);
      expect(await readStats("backup")).toMatchObject({
        sourceSessionId: "source",
        totalTokensPruned: 23,
      });
    } finally {
      await rm(statsPath, { recursive: true, force: true });
    }
  });

  test("projects rewritten v2 and v3 manifests through the retained transform array", async () => {
    const messages = rewriteForkMessages([
      ...persistedMediaTurn(
        "hook-v2",
        2,
        "image/png",
        `data:image/png;base64,${"A".repeat(512)}`,
      ),
      ...persistedMediaTurn(
        "hook-v3",
        3,
        "image/jpeg",
        `data:image/jpeg;base64,${"B".repeat(512)}`,
      ),
    ]);
    const retained = messages;
    const hooks = await server({} as PluginInput);

    await hooks["experimental.chat.messages.transform"]?.(
      {} as never,
      { messages } as never,
    );

    expect(messages).toBe(retained);
    expect(messages[0]!.parts[1]).toMatchObject({ type: "text" });
    expect(messages[2]!.parts[1]).toMatchObject({ type: "text" });
  });

  test("increments physical and media cache savings once per completed assistant", async () => {
    await writeStats(
      "source",
      conversationStats({
        totalTokensPruned: 100,
        mediaTokensProjected: 7,
      }),
    );
    const hooks = await server({} as PluginInput);
    const event = assistantEvent("completed-assistant");

    await hooks.event?.({ event });
    await hooks.event?.({ event });
    await hooks.event?.({
      event: assistantEvent("incomplete-assistant", false),
    });

    expect(await readStats("source")).toMatchObject({
      cachedTokensSaved: 100,
      cachedMediaTokensSaved: 7,
      processedAssistantMessageIds: ["completed-assistant"],
    });
  });

  test("displays legacy stats with zero media fields and current media labels", async () => {
    const sessionID = "legacy-stats";
    await writeStats(sessionID, {
      version: 1,
      sourceSessionId: null,
      rootSessionId: sessionID,
      totalTokensPruned: 41,
      cachedTokensSaved: 82,
      processedAssistantMessageIds: [],
    });
    const prompts: Record<string, unknown>[] = [];
    const v2 = {
      session: {
        get: async () => ({ data: { ...sourceSession(), id: sessionID } }),
        prompt: async (request: Record<string, unknown>) => {
          prompts.push(request);
          return { data: true };
        },
      },
    } as unknown as V2Client;

    await executeMagicStats(v2, sessionID);

    const notice = promptText(prompts[0]!);
    expect(notice).toContain("Tokens physically pruned");
    expect(notice).toContain("Inline image tokens projected: ~0");
    expect(notice).toContain("Projected-media cache reads saved: ~0 tokens");
  });

  test("keeps prior summary turns opaque and preserves cache/stat lineage on recompaction", async () => {
    const historical = persistedMediaTurn(
      "historical",
      2,
      "image/png",
      "data:image/png;base64,Zm9v",
    );
    const historicalBefore = structuredClone(historical);
    const messages = [
      ...historical,
      ...compactableCommandMessages("fresh-second", "large output"),
    ];
    const contentID = await allocateOmission(
      "source",
      { content: "lineage omission" },
      () => "B".repeat(22),
    );
    await writeStats(
      "source",
      conversationStats({
        rootSessionId: "root-session",
        totalTokensPruned: 10,
        cachedTokensSaved: 3,
        processedAssistantMessageIds: ["prior-assistant"],
      }),
    );
    const harness = commandHarness(messages, {
      generationResponse: {
        turns: [
          result(
            1,
            "fresh-second request",
            "Completed the second compaction candidate.",
          ),
        ],
      },
    });

    await expect(
      executeMagicCompact(harness.v2, "source", 0),
    ).resolves.toBeTrue();

    expect(messages.slice(0, 2)).toEqual(historicalBefore);
    expect(summaryParts(messages)).toHaveLength(2);
    expect(await readOmittedContent("source", contentID)).toBe(
      "lineage omission",
    );
    expect(await readStats("source")).toMatchObject({
      rootSessionId: "root-session",
      cachedTokensSaved: 3,
      processedAssistantMessageIds: ["prior-assistant"],
    });
  });
});

function generationHarness(
  turns: Turn[],
  response: GenerationResponse,
  options: {
    providerError?: Error;
    providerDelayMs?: number;
    forkGate?: Promise<void>;
    nextTurn?: Turn;
    forkMessages?: MessageWithParts[];
    markerError?: Error;
    modelImageCapability?: boolean;
    responseFinish?: string;
    responseError?: unknown;
    permissionError?: Error;
    abortError?: Error;
    deleteError?: Error;
    compactionAgent?: "missing" | "disabled" | "model-missing";
    responseMarkerOrder?: number[];
    providerDelayByFirstOrdinal?: Record<number, number>;
    providerErrorForFirstOrdinal?: number;
    holdPromptsUntil?: number;
    modelLimits?: {
      context: number;
      input?: number;
      output?: number;
    } | null;
    sourceMessages?: MessageWithParts[];
    mutateRetainedMessages?: (
      messages: MessageWithParts[],
      sessionID: string,
    ) => void;
  } = {},
) {
  const events: string[] = [];
  const creates: Record<string, unknown>[] = [];
  const forks: Record<string, unknown>[] = [];
  const updates: Record<string, unknown>[] = [];
  const prompts: Record<string, unknown>[] = [];
  const aborts: string[] = [];
  const deletes: string[] = [];
  const abortRequests: Record<string, unknown>[] = [];
  const deleteRequests: Record<string, unknown>[] = [];
  const markerUpdates: Array<{
    sessionID: string;
    messageID: string;
    part: TextPart;
  }> = [];
  const messageDeletes: Array<{ sessionID: string; messageID: string }> = [];
  const partDeletes: Array<{
    sessionID: string;
    messageID: string;
    partID: string;
  }> = [];
  const promptHistories: Array<{
    sessionID: string;
    messages: MessageWithParts[];
  }> = [];
  const completedPromptFirstOrdinals: number[] = [];
  const forkMessagesByID = new Map<string, MessageWithParts[]>();
  const messageLoadsByFork = new Map<string, number>();
  let providerCalls = 0;
  let activeForks = 0;
  let peakActiveForks = 0;
  let activePrompts = 0;
  let peakActivePrompts = 0;
  let releasePromptGate: (() => void) | undefined;
  const promptGate = options.holdPromptsUntil
    ? new Promise<void>(resolve => {
        releasePromptGate = resolve;
      })
    : undefined;
  const sourceMessages = options.sourceMessages ?? [
    ...turns.flatMap(turn => [...turn.user, ...turn.assistants]),
    ...(options.nextTurn
      ? [...options.nextTurn.user, ...options.nextTurn.assistants]
      : []),
  ];
  const modelImageCapability =
    "modelImageCapability" in options ? options.modelImageCapability : true;
  const configuredAgent = {
    name: "compaction",
    native: true,
    hidden: true,
    model: {
      providerID: "configured-provider",
      modelID: "configured-model",
    },
    variant: "configured-variant",
  };
  const v2 = {
    app: {
      agents: async () => {
        events.push("agents");
        return {
          data:
            options.compactionAgent === "missing"
              ? []
              : options.compactionAgent === "disabled"
                ? [{ ...configuredAgent, native: false }]
                : options.compactionAgent === "model-missing"
                  ? [{ ...configuredAgent, model: undefined }]
                  : [configuredAgent],
        };
      },
    },
    provider: {
      list: async () => {
        providerCalls += 1;
        events.push("providers");
        return {
          data: {
            all: [
              {
                id: "configured-provider",
                models:
                  modelImageCapability === undefined
                    ? {}
                    : {
                        "configured-model": {
                          capabilities: {
                            input: { image: modelImageCapability },
                          },
                          limit:
                            "modelLimits" in options
                              ? options.modelLimits
                              : {
                                  context: 1_000_000,
                                  input: 500_000,
                                  output: 64_000,
                                },
                        },
                      },
              },
            ],
            connected: ["configured-provider"],
            default: {},
          },
        };
      },
    },
    session: {
      create: async (request: Record<string, unknown>) => {
        creates.push(request);
        events.push("create");
        return {
          data: { ...sourceSession(), id: `unexpected-${creates.length}` },
        };
      },
      fork: async (request: Record<string, unknown>) => {
        await options.forkGate;
        forks.push(request);
        events.push("fork");
        const forkID =
          forks.length === 1
            ? "generation-fork"
            : `generation-fork-${forks.length}`;
        forkMessagesByID.set(
          forkID,
          forks.length === 1 && options.forkMessages
            ? structuredClone(options.forkMessages)
            : rewriteForkMessages(sourceMessages, forkID),
        );
        activeForks += 1;
        peakActiveForks = Math.max(peakActiveForks, activeForks);
        return { data: { ...sourceSession(), id: forkID } };
      },
      update: async (request: Record<string, unknown>) => {
        updates.push(request);
        events.push("deny");
        if (options.permissionError) throw options.permissionError;
        return { data: { ...sourceSession(), id: String(request.sessionID) } };
      },
      messages: async (request: { sessionID: string }) => {
        events.push("messages");
        const loads = (messageLoadsByFork.get(request.sessionID) ?? 0) + 1;
        messageLoadsByFork.set(request.sessionID, loads);
        const stored = forkMessagesByID.get(request.sessionID) ?? [];
        if (loads === 2) {
          options.mutateRetainedMessages?.(stored, request.sessionID);
        }
        return {
          data: structuredClone(stored),
        };
      },
      prompt: async (request: Record<string, unknown>) => {
        prompts.push(request);
        events.push("prompt");
        const sessionID = String(request.sessionID);
        promptHistories.push({
          sessionID,
          messages: structuredClone(forkMessagesByID.get(sessionID) ?? []),
        });
        const index = generationPromptIndex(request);
        const firstOrdinal = index.turns[0]!.ordinal;
        activePrompts += 1;
        peakActivePrompts = Math.max(peakActivePrompts, activePrompts);
        if (
          options.holdPromptsUntil
          && activePrompts >= options.holdPromptsUntil
        ) {
          releasePromptGate?.();
        }
        try {
          await promptGate;
          const delay =
            options.providerDelayByFirstOrdinal?.[firstOrdinal]
            ?? options.providerDelayMs;
          if (delay) {
            await new Promise(resolve => setTimeout(resolve, delay));
          }
          if (
            options.providerError
            || options.providerErrorForFirstOrdinal === firstOrdinal
          ) {
            throw options.providerError ?? new Error("batch provider failure");
          }
          const text = JSON.stringify(
            responseWithPromptMarkers(
              request,
              response,
              options.responseMarkerOrder,
            ),
          );
          completedPromptFirstOrdinals.push(firstOrdinal);
          return {
            data: {
              info: {
                id: "response",
                ...(options.responseFinish
                  ? { finish: options.responseFinish }
                  : {}),
                ...(options.responseError
                  ? { error: options.responseError }
                  : {}),
              },
              parts: [textPart("response-text", "response", text)],
            },
          };
        } finally {
          activePrompts -= 1;
        }
      },
      abort: async (request: Record<string, unknown>) => {
        abortRequests.push(request);
        const sessionID = String(request.sessionID);
        aborts.push(sessionID);
        events.push("abort");
        if (options.abortError) throw options.abortError;
        return { data: true };
      },
      delete: async (request: Record<string, unknown>) => {
        deleteRequests.push(request);
        const sessionID = String(request.sessionID);
        deletes.push(sessionID);
        events.push("delete");
        if (options.deleteError) throw options.deleteError;
        if (forkMessagesByID.delete(sessionID)) activeForks -= 1;
        return { data: true };
      },
      deleteMessage: async (request: {
        sessionID: string;
        messageID: string;
      }) => {
        events.push("message-delete");
        messageDeletes.push(request);
        const messages = forkMessagesByID.get(request.sessionID);
        const index =
          messages?.findIndex(message => message.info.id === request.messageID)
          ?? -1;
        if (messages && index >= 0) messages.splice(index, 1);
        return { data: true };
      },
    },
    part: {
      update: async (request: {
        sessionID: string;
        messageID: string;
        part: TextPart;
      }) => {
        events.push("marker");
        if (options.markerError) throw options.markerError;
        markerUpdates.push(request);
        const target = forkMessagesByID
          .get(request.sessionID)
          ?.find(message => message.info.id === request.messageID);
        target?.parts.push(request.part);
        return { data: request.part };
      },
      delete: async (request: {
        sessionID: string;
        messageID: string;
        partID: string;
      }) => {
        partDeletes.push(request);
        return { data: true };
      },
    },
  } as unknown as V2Client;
  return {
    v2,
    events,
    creates,
    forks,
    updates,
    prompts,
    aborts,
    deletes,
    abortRequests,
    deleteRequests,
    markerUpdates,
    messageDeletes,
    partDeletes,
    promptHistories,
    completedPromptFirstOrdinals,
    get providerCalls() {
      return providerCalls;
    },
    get peakActiveForks() {
      return peakActiveForks;
    },
    get peakActivePrompts() {
      return peakActivePrompts;
    },
  };
}

function responseFor(turns: Turn[]): GenerationResponse {
  return {
    turns: turns.map((turn, index) =>
      result(
        index + 1,
        firstUserLine(turn),
        `Turn ${index + 1} outcome and continuation details.`,
        inlineImages(turn).map((_, imageIndex) => ({
          ordinal: imageIndex + 1,
          description: `Visual ${imageIndex + 1} contains relevant interface evidence.`,
        })),
      ),
    ),
  };
}

function result(
  ordinal: number,
  anchor: string,
  summary: string,
  visualDescriptions: Array<{ ordinal: number; description: string }> = [],
): TurnResult {
  return { ordinal, anchor, summary, visualDescriptions };
}

function generationPlan(
  turns: Turn[],
  nextTurn: Turn | null = null,
): CompactionPlan {
  const messages = [
    ...turns.flatMap(turn => [...turn.user, ...turn.assistants]),
    ...(nextTurn ? [...nextTurn.user, ...nextTurn.assistants] : []),
  ];
  return {
    summarizedTurns: turns,
    nextTurn,
    sourceTurnStartIndex: 0,
    sourceSnapshot: structuredClone(
      messages.map(item => ({ info: item.info, parts: item.parts })),
    ) as CompactionPlan["sourceSnapshot"],
  } as CompactionPlan;
}

function candidateTurn(id: string, userText: string, parts: Part[]): Turn {
  const messages = candidateMessages(id, userText, parts);
  return { user: [messages[0]!], assistants: [messages[1]!] };
}

function nativeCompactionPart(id: string, messageID: string): Part {
  return {
    id,
    sessionID: "source",
    messageID,
    type: "compaction",
    auto: true,
  } as unknown as Part;
}

function nativeMarker(id: string, tailStartID?: string): MessageWithParts {
  const part = nativeCompactionPart(`${id}-part`, id);
  if (tailStartID) {
    Object.assign(part, { tail_start_id: tailStartID });
  }
  return message(id, "user", [part]);
}

function nativeSummary(
  id: string,
  parentID: string,
  overrides: { finish?: string } = { finish: "stop" },
): MessageWithParts {
  return {
    info: {
      id,
      role: "assistant",
      parentID,
      summary: true,
      ...(overrides.finish ? { finish: overrides.finish } : {}),
    } as Message,
    parts: [textPart(`${id}-text`, id, "checkpoint")],
  };
}

function candidateMessages(
  id: string,
  userText: string,
  assistantParts: Part[],
): MessageWithParts[] {
  return [
    message(`${id}-user`, "user", [
      textPart(`${id}-user-text`, `${id}-user`, userText),
      ...assistantParts.filter(part => part.type === "file"),
    ]),
    message(`${id}-assistant`, "assistant", [
      {
        id: `${id}-reasoning`,
        sessionID: "source",
        messageID: `${id}-assistant`,
        type: "reasoning",
        text: "removable reasoning ".repeat(200),
        time: { start: 1, end: 2 },
      } as Part,
      ...assistantParts.filter(part => part.type !== "file"),
    ]),
  ];
}

function persistedMediaTurn(
  id: string,
  version: 2 | 3,
  mime: string,
  url: string,
): MessageWithParts[] {
  const partID = `${id}-image`;
  return [
    message(`${id}-user`, "user", [
      textPart(`${id}-text`, `${id}-user`, "review"),
      imagePart(partID, `${id}-user`, url, mime),
    ]),
    message(`${id}-assistant`, "assistant", [
      {
        ...textPart(
          `prt_-magic_capsule_v${version}_${id}-assistant`,
          `${id}-assistant`,
          "Historical summary",
        ),
        metadata: {
          magicCompact: {
            summary: true,
            capsule: true,
            agentic: true,
            version,
            mediaManifest: {
              version: 1,
              images: [
                {
                  ordinal: 1,
                  ...(version === 2 ? { partId: partID } : {}),
                  mime,
                  urlSha256: createHash("sha256").update(url).digest("hex"),
                },
              ],
            },
          },
        },
      },
    ]),
  ];
}

function persistedSummaryTurn(id: string, version: 0 | 1 | 2) {
  return [
    message(`${id}-user`, "user", [
      textPart(`${id}-user-text`, `${id}-user`, `${id} request`),
    ]),
    message(`${id}-assistant`, "assistant", [
      {
        ...textPart(`${id}-summary`, `${id}-assistant`, `${id} summary`),
        metadata: { magicCompact: { summary: true, version } },
      },
    ]),
  ];
}

function message(
  id: string,
  role: "user" | "assistant",
  parts: Part[],
): MessageWithParts {
  return { info: { id, role } as Message, parts };
}

function textPart(id: string, messageID: string, text: string): TextPart {
  return { id, sessionID: "source", messageID, type: "text", text };
}

function imagePart(
  id: string,
  messageID: string,
  url: string,
  mime = "image/png",
): Extract<Part, { type: "file" }> {
  return {
    id,
    sessionID: "source",
    messageID,
    type: "file",
    mime,
    filename: `${id}.png`,
    url,
  };
}

function toolPart(
  id: string,
  messageID: string,
  state: Record<string, unknown>,
): Part {
  return {
    id,
    sessionID: "source",
    messageID,
    type: "tool",
    callID: `${id}-call`,
    tool: "read",
    state,
  } as Part;
}

function sourceSession(): Session {
  return {
    id: "source",
    title: "Source",
    directory: "/workspace",
    workspaceID: "workspace",
    model: { id: "source-model", providerID: "source-provider" },
  } as Session;
}

function promptText(request: Record<string, unknown>): string {
  return (
    (request.parts as Array<{ type: string; text?: string }> | undefined) ?? []
  )
    .filter(part => part.type === "text")
    .map(part => part.text ?? "")
    .join("");
}

function responseWithPromptMarkers(
  request: Record<string, unknown>,
  response: GenerationResponse,
  markerOrder?: number[],
): GenerationResponse {
  const index = generationPromptIndex(request);
  const ordinals = new Set(index.turns.map(turn => turn.ordinal));
  return {
    turns: response.turns
      .filter(turn => ordinals.has(turn.ordinal))
      .map((turn, turnIndex) => ({
        ...turn,
        marker:
          turn.marker
          ?? index.turns[markerOrder?.[turnIndex] ?? turnIndex]?.marker
          ?? "",
      })),
  };
}

function generationPromptIndex(request: Record<string, unknown>) {
  return JSON.parse(promptText(request).split("\n").at(-1)!) as {
    turns: Array<{ ordinal: number; marker: string }>;
  };
}

function firstUserLine(turn: Turn): string {
  const text = turn.user
    .flatMap(item => item.parts)
    .find(
      (part): part is TextPart =>
        part.type === "text"
        && part.synthetic !== true
        && part.ignored !== true,
    )?.text;
  return Array.from((text ?? "").split(/\r\n|\r|\n/u, 1)[0] ?? "")
    .slice(0, 300)
    .join("");
}

function inlineImages(turn: Turn) {
  return [...turn.user, ...turn.assistants].flatMap(item =>
    item.parts.filter(
      (part): part is Extract<Part, { type: "file" }> =>
        part.type === "file"
        && part.mime.startsWith("image/")
        && part.url.startsWith("data:"),
    ),
  );
}

function rewriteForkMessages(
  messages: MessageWithParts[],
  sessionID = "generation-fork",
): MessageWithParts[] {
  const messageIDs = new Map(
    messages.map((item, index) => [item.info.id, `fork-message-${index + 1}`]),
  );
  let partIndex = 0;
  return structuredClone(messages).map(item => {
    const messageID = messageIDs.get(item.info.id)!;
    const info = {
      ...item.info,
      id: messageID,
      sessionID,
      ...("parentID" in item.info && typeof item.info.parentID === "string"
        ? { parentID: messageIDs.get(item.info.parentID) ?? item.info.parentID }
        : {}),
    } as Message;
    const parts = item.parts.map(part => ({
      ...part,
      id: `fork-part-${++partIndex}`,
      sessionID,
      messageID,
    })) as Part[];
    return { info, parts };
  });
}

function nestedErrorMessages(value: unknown): string[] {
  if (value instanceof AggregateError) {
    return [
      value.message,
      ...value.errors.flatMap(nestedErrorMessages),
      ...nestedErrorMessages(value.cause),
    ];
  }
  return value instanceof Error ? [value.message] : [];
}

async function waitFor(predicate: () => boolean): Promise<void> {
  const deadline = Date.now() + 1_000;
  while (!predicate()) {
    if (Date.now() >= deadline) throw new Error("Test condition timed out.");
    await new Promise(resolve => setTimeout(resolve, 1));
  }
}

async function settleWithin<T>(promise: Promise<T>, timeoutMs: number) {
  return Promise.race([
    promise,
    new Promise<never>((_, reject) =>
      setTimeout(
        () => reject(new Error("Test operation timed out.")),
        timeoutMs,
      ),
    ),
  ]);
}

type CommandHarnessOptions = {
  generationResponse: GenerationResponse;
  generationErrorForFirstOrdinal?: number;
  generationErrorsByFirstOrdinal?: Record<number, Error>;
  generationAbortErrorForFirstOrdinal?: number;
  generationDeleteErrorForFirstOrdinal?: number;
  holdGenerationPromptsUntil?: number;
  transformGenerationResponse?: (
    response: GenerationResponse,
  ) => GenerationResponse;
  failSummaryUpdate?: boolean;
  boundaryInsertError?: Error;
  statsNoticeError?: Error;
  appendConcurrentOnFailure?: boolean;
  driftAfterBackup?: boolean;
  driftAfterInsertions?: boolean;
  driftBeforeFinalCount?: boolean;
  partDeleteError?: Error;
  sourceMetadataUpdateError?: Error;
  beforeFinalCount?: () => Promise<void>;
  toastMode?: "hang" | "fail";
};

function commandHarness(
  messages: MessageWithParts[],
  options: CommandHarnessOptions,
) {
  const events: string[] = [];
  const forks: string[] = [];
  const partUpdates: Part[] = [];
  const partDeletes: Array<{ messageID: string; partID: string }> = [];
  const messageDeletes: string[] = [];
  const abortedSessions: string[] = [];
  const selectedSessions: string[] = [];
  const deletedSessions: string[] = [];
  let backupMessages: MessageWithParts[] | undefined;
  const generationForkMessages = new Map<string, MessageWithParts[]>();
  const activeGenerationForks = new Set<string>();
  const generationFirstOrdinalByFork = new Map<string, number>();
  let generationPromptCount = 0;
  let activeGenerationPrompts = 0;
  let releaseGenerationPromptGate: (() => void) | undefined;
  const generationPromptGate = options.holdGenerationPromptsUntil
    ? new Promise<void>(resolve => {
        releaseGenerationPromptGate = resolve;
      })
    : undefined;
  let backupCreated = false;
  let boundaryInserted = false;
  let statsInserted = false;
  let backupDriftApplied = false;
  let insertionDriftApplied = false;
  let finalDriftApplied = false;
  let beforeFinalCountApplied = false;
  let noticeIndex = 0;
  const appendConcurrentOnFailure = () => {
    if (
      !options.appendConcurrentOnFailure
      || messages.some(message => message.info.id === "concurrent-user")
    ) {
      return;
    }
    messages.push(
      message("concurrent-user", "user", [
        textPart(
          "concurrent-user-text",
          "concurrent-user",
          "concurrent message appended after backup",
        ),
      ]),
    );
  };

  const v2 = {
    app: {
      agents: async () => ({
        data: [
          {
            name: "compaction",
            native: true,
            hidden: true,
            model: {
              providerID: "configured-provider",
              modelID: "configured-model",
            },
            variant: "configured-variant",
          },
        ],
      }),
    },
    provider: {
      list: async () => ({
        data: {
          all: [
            {
              id: "configured-provider",
              models: {
                "configured-model": {
                  capabilities: { input: { image: true } },
                  limit: {
                    context: 1_000_000,
                    input: 500_000,
                    output: 64_000,
                  },
                },
              },
            },
          ],
          connected: ["configured-provider"],
          default: {},
        },
      }),
    },
    session: {
      get: async () => ({ data: sourceSession() }),
      messages: async (request: { sessionID: string }) => {
        if (request.sessionID.startsWith("generation-fork")) {
          return {
            data: structuredClone(
              generationForkMessages.get(request.sessionID) ?? [],
            ),
          };
        }
        if (options.driftAfterBackup && backupCreated && !backupDriftApplied) {
          backupDriftApplied = true;
          messages[0]!.parts.push(
            textPart("external-race", messages[0]!.info.id, "external change"),
          );
        }
        if (
          options.driftAfterInsertions
          && boundaryInserted
          && !insertionDriftApplied
        ) {
          insertionDriftApplied = true;
          messages.push(
            message("post-insertion-external-user", "user", [
              textPart(
                "post-insertion-external-text",
                "post-insertion-external-user",
                "concurrent user message must survive",
              ),
            ]),
          );
        }
        if (
          options.driftBeforeFinalCount
          && statsInserted
          && !finalDriftApplied
        ) {
          finalDriftApplied = true;
          messages.push(
            message("late-external-user", "user", [
              textPart(
                "late-external-text",
                "late-external-user",
                "late external change",
              ),
            ]),
          );
        }
        if (
          options.beforeFinalCount
          && statsInserted
          && !beforeFinalCountApplied
        ) {
          beforeFinalCountApplied = true;
          const finalCountMessages = structuredClone(messages);
          await options.beforeFinalCount();
          return { data: finalCountMessages };
        }
        return { data: messages };
      },
      fork: async () => {
        if (generationPromptCount === 0 || activeGenerationForks.size > 0) {
          const forkID =
            generationForkMessages.size === 0
              ? "generation-fork"
              : `generation-fork-${generationForkMessages.size + 1}`;
          forks.push(forkID);
          events.push("generation-fork");
          generationForkMessages.set(
            forkID,
            rewriteForkMessages(messages, forkID),
          );
          activeGenerationForks.add(forkID);
          return { data: { ...sourceSession(), id: forkID } };
        }
        forks.push("backup");
        events.push("backup-fork");
        backupMessages = structuredClone(messages);
        backupCreated = true;
        return { data: { ...sourceSession(), id: "backup" } };
      },
      update: async (request: Record<string, unknown>) => {
        if (
          request.sessionID === "source"
          && request.metadata
          && options.sourceMetadataUpdateError
        ) {
          appendConcurrentOnFailure();
          throw options.sourceMetadataUpdateError;
        }
        return { data: { ...sourceSession(), id: String(request.sessionID) } };
      },
      prompt: async (request: Record<string, unknown>) => {
        if (request.noReply !== true) {
          generationPromptCount += 1;
          events.push("generation-prompt");
          const firstOrdinal = generationPromptIndex(request).turns[0]!.ordinal;
          generationFirstOrdinalByFork.set(
            String(request.sessionID),
            firstOrdinal,
          );
          activeGenerationPrompts += 1;
          if (
            options.holdGenerationPromptsUntil
            && activeGenerationPrompts >= options.holdGenerationPromptsUntil
          ) {
            releaseGenerationPromptGate?.();
          }
          try {
            await generationPromptGate;
            const configuredError =
              options.generationErrorsByFirstOrdinal?.[firstOrdinal];
            if (
              configuredError
              || options.generationErrorForFirstOrdinal === firstOrdinal
            ) {
              throw configuredError ?? new Error("generation batch failed");
            }
            const response = responseWithPromptMarkers(
              request,
              options.generationResponse,
            );
            const transformedResponse = options.transformGenerationResponse
              ? options.transformGenerationResponse(response)
              : response;
            return {
              data: {
                info: { id: "generation-response" },
                parts: [
                  textPart(
                    "generation-response-text",
                    "generation-response",
                    JSON.stringify(transformedResponse),
                  ),
                ],
              },
            };
          } finally {
            activeGenerationPrompts -= 1;
          }
        }

        noticeIndex += 1;
        const messageID = `notice-${noticeIndex}`;
        const parts = ((request.parts as Part[] | undefined) ?? []).map(
          (part, index) =>
            ({
              ...part,
              id: `${messageID}-part-${index}`,
              sessionID: "source",
              messageID,
            }) as Part,
        );
        const insertsBoundary = parts.some(
          part =>
            part.type === "text"
            && part.metadata?.magicCompact !== undefined
            && (part.metadata.magicCompact as Record<string, unknown>)[
              "boundary"
            ] === true,
        );
        const insertsStats = parts.some(
          part =>
            part.type === "text"
            && part.metadata?.magicCompact !== undefined
            && (part.metadata.magicCompact as Record<string, unknown>)["stats"]
              === true,
        );
        if (
          (insertsBoundary && options.boundaryInsertError)
          || (insertsStats && options.statsNoticeError)
        ) {
          appendConcurrentOnFailure();
          throw insertsBoundary
            ? options.boundaryInsertError
            : options.statsNoticeError;
        }
        boundaryInserted ||= insertsBoundary;
        statsInserted ||= insertsStats;
        messages.push(message(messageID, "user", parts));
        return { data: { info: { id: messageID }, parts } };
      },
      abort: async (request: { sessionID: string }) => {
        abortedSessions.push(request.sessionID);
        if (
          generationFirstOrdinalByFork.get(request.sessionID)
          === options.generationAbortErrorForFirstOrdinal
        ) {
          throw new Error("raw generation abort payload");
        }
        return { data: true };
      },
      delete: async (request: { sessionID: string }) => {
        deletedSessions.push(request.sessionID);
        if (request.sessionID.startsWith("generation-fork")) {
          events.push("generation-delete");
          if (
            generationFirstOrdinalByFork.get(request.sessionID)
            === options.generationDeleteErrorForFirstOrdinal
          ) {
            throw new Error("raw generation delete payload");
          }
          activeGenerationForks.delete(request.sessionID);
        }
        if (request.sessionID === "source" && backupMessages) {
          messages.splice(
            0,
            messages.length,
            ...structuredClone(backupMessages),
          );
        }
        return { data: true };
      },
      deleteMessage: async (request: {
        sessionID: string;
        messageID: string;
      }) => {
        messageDeletes.push(request.messageID);
        const targetMessages = request.sessionID.startsWith("generation-fork")
          ? generationForkMessages.get(request.sessionID)
          : messages;
        const index =
          targetMessages?.findIndex(
            message => message.info.id === request.messageID,
          ) ?? -1;
        if (targetMessages && index !== -1) targetMessages.splice(index, 1);
        return { data: true };
      },
    },
    part: {
      update: async (request: {
        sessionID?: string;
        messageID: string;
        part: Part;
      }) => {
        if (request.sessionID?.startsWith("generation-fork")) {
          const target = generationForkMessages
            .get(request.sessionID)
            ?.find(message => message.info.id === request.messageID);
          target?.parts.push(request.part);
          return { data: request.part };
        }
        const metadata =
          request.part.type === "text"
          && request.part.metadata?.magicCompact !== undefined
            ? (request.part.metadata.magicCompact as Record<string, unknown>)
            : undefined;
        if (metadata?.["summary"] === true && options.failSummaryUpdate) {
          appendConcurrentOnFailure();
          throw new Error("summary insertion failed");
        }
        partUpdates.push(request.part);
        const target = messages.find(
          message => message.info.id === request.messageID,
        );
        if (target) {
          const index = target.parts.findIndex(
            part => part.id === request.part.id,
          );
          if (index === -1) target.parts.push(request.part);
          else target.parts[index] = request.part;
        }
        return { data: request.part };
      },
      delete: async (request: { messageID: string; partID: string }) => {
        partDeletes.push(request);
        if (options.partDeleteError) {
          appendConcurrentOnFailure();
          throw options.partDeleteError;
        }
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
      showToast: async () => {
        if (options.toastMode === "hang") return new Promise(() => {});
        if (options.toastMode === "fail") {
          throw new Error("toast failed");
        }
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
    forks,
    partUpdates,
    partDeletes,
    messageDeletes,
    abortedSessions,
    selectedSessions,
    deletedSessions,
    get backupMessages() {
      return backupMessages;
    },
    get boundaryInserted() {
      return boundaryInserted;
    },
    get statsInserted() {
      return statsInserted;
    },
    get activeGenerationForks() {
      return [...activeGenerationForks];
    },
  };
}

function compactableCommandMessages(
  id: string,
  output: string,
): MessageWithParts[] {
  return [
    message(`${id}-user`, "user", [
      textPart(`${id}-user-text`, `${id}-user`, `${id} request`),
    ]),
    message(`${id}-assistant`, "assistant", [
      {
        id: `${id}-reasoning`,
        sessionID: "source",
        messageID: `${id}-assistant`,
        type: "reasoning",
        text: "removable reasoning ".repeat(2_000),
        time: { start: 1, end: 2 },
      } as Part,
      toolPart(`${id}-tool`, `${id}-assistant`, {
        status: "completed",
        input: { filePath: "/tmp/result.txt" },
        output,
        title: "Read result",
        metadata: {},
        time: { start: 1, end: 2 },
      }),
    ]),
  ];
}

function summaryParts(messages: MessageWithParts[]): TextPart[] {
  return messages.flatMap(message =>
    message.parts.filter(
      (part): part is TextPart =>
        part.type === "text"
        && part.metadata?.magicCompact !== undefined
        && (part.metadata.magicCompact as Record<string, unknown>)["summary"]
          === true,
    ),
  );
}

function boundaryParts(messages: MessageWithParts[]): TextPart[] {
  return messages.flatMap(message =>
    message.parts.filter(
      (part): part is TextPart =>
        part.type === "text"
        && part.text === POST_COMPACTION_NOTICE
        && part.metadata?.magicCompact !== undefined
        && (part.metadata.magicCompact as Record<string, unknown>)["boundary"]
          === true,
    ),
  );
}

function conversationStats(
  overrides: Partial<ConversationStats> = {},
): ConversationStats {
  return {
    version: 1 as const,
    sourceSessionId: null,
    rootSessionId: "source",
    totalTokensPruned: 0,
    cachedTokensSaved: 0,
    mediaTokensProjected: 0,
    cachedMediaTokensSaved: 0,
    processedAssistantMessageIds: [] as string[],
    ...overrides,
  };
}

function assistantEvent(id: string, completed = true): Event {
  return {
    type: "message.updated",
    properties: {
      info: {
        id,
        sessionID: "source",
        role: "assistant",
        time: { created: 1, ...(completed ? { completed: 2 } : {}) },
        parentID: "parent",
        modelID: "test-model",
        providerID: "test-provider",
        mode: "primary",
        agent: "build",
        path: { cwd: "/workspace", root: "/workspace" },
        cost: 0,
        tokens: {
          input: 0,
          output: 0,
          reasoning: 0,
          cache: { read: 0, write: 0 },
        },
      } as AssistantMessage,
    },
  };
}

import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import type { Config, PluginInput } from "@opencode-ai/plugin";
import type {
  Message,
  Part,
  Session,
  TextPart,
  ToolPart,
} from "@opencode-ai/sdk/v2";
import type { V2Client } from "../src/api";
import {
  generateCompactionCapsules,
  injectSummaries,
} from "../src/compact/compact";
import { renderCanonicalCapsule } from "../src/compact/capsule";
import { capsuleSummaryPartID } from "../src/compact/constants";
import {
  createCompactionPlan,
  NativeCheckpointChangedError,
  type MessageWithParts,
  type Turn,
} from "../src/compact/plan";
import { createPruneProjection } from "../src/compact/prune";
import { applyBackup, createBackup } from "../src/compact/session";
import { COMPACT_NOOP, executeMagicCompact } from "../src/magic-compact";
import server from "../src/index";
import {
  allocateOmission,
  cachePath,
  deleteCache,
  readCache,
  readOmittedContent,
} from "../src/storage/omission";
import { deleteStats, readStats, writeStats } from "../src/storage/stats";
import { statsMessage } from "../src/stats/constants";
import { countCanonicalContentTokens } from "../src/stats/tokenize";
import { incidentSemanticSessionFixture } from "./fixtures/incident-session";

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
  test("registers commands, read_omitted_content, and no hidden summarizer plumbing", async () => {
    const hooks = await server({} as PluginInput);
    const config = {} as Config;

    await hooks.config?.(config);

    expect(config.command?.["magic-compact"]?.description).toBe(
      "Lossless context compression",
    );
    expect(config.command?.["magic-trim"]).toBeDefined();
    expect(config.command?.["magic-stats"]).toBeDefined();
    expect(config.agent?.["magic-compact-summarizer"]).toBeUndefined();
    expect(hooks["chat.params"]).toBeUndefined();
    expect(hooks.tool?.read_omitted_content).toBeDefined();
  });

  test("compacts without a model and preserves exact ordered assistant text", async () => {
    const messages = exactTextMessages();
    messages[1]!.parts.push({
      id: "hard-cut-reasoning",
      sessionID: "source",
      messageID: "hard-cut-assistant-1",
      type: "reasoning",
      text: "removable reasoning ".repeat(6_000),
      time: { start: 1, end: 2 },
    } as Part);
    const requests = createCommandHarness(messages);

    await executeMagicCompact(requests.v2, "source", 0);

    const capsule = storedSummary(requests, "hard-cut-assistant-1");
    expect(capsule).toStartWith("Outcome: completed\n");
    expect(workJsonStrings(capsule)).toEqual(exactTextFixtureStrings());
    expect(capsule).not.toContain("Synthetic assistant text");
    expect(capsule).not.toContain("Opaque persisted summary");
  });

  test("preserves lone surrogates and provider-unsafe code points as recoverable source strings", () => {
    const exact = "exact\u0000nul\u000bvt\ud800lone\ufffe\uffff";
    const turn = turnWithAssistantText(exact);

    const capsule = expectCanonicalCapsule(turn);

    expect(workJsonStrings(capsule)).toEqual([exact]);
    expect(capsule).not.toContain("\u0000");
    expect(capsule).not.toContain("\ufffe");
    expect(capsule).toContain("\\u0000");
    expect(capsule).toContain("\\ud800");
    expect(capsule).toContain("\\ufffe");
  });

  test("over-cap required text leaves that turn unchanged while compacting profitable turns", async () => {
    const exactText = `Cannot summarize this exactly. ${"x".repeat(80_000)}`;
    const messages = [
      message("overcap-user", "user", [
        textPart("overcap-user-text", "source", "overcap-user", "Request"),
      ]),
      message("overcap-assistant", "assistant", [
        textPart("overcap-text", "source", "overcap-assistant", exactText),
      ]),
      ...completedTurn("profitable", "Second request"),
    ];
    messages
      .find(item => item.info.id === "profitable-assistant")!
      .parts.push({
        id: "profitable-reasoning",
        sessionID: "source",
        messageID: "profitable-assistant",
        type: "reasoning",
        text: "removable reasoning ".repeat(6_000),
        time: { start: 1, end: 2 },
      } as Part);
    const before = structuredClone(messages[1]);
    const requests = createCommandHarness(messages);

    await executeMagicCompact(requests.v2, "source", 0);

    expect(messages[1]).toEqual(before);
    expect(summaryText(messages, "profitable-assistant")).toContain(
      "Outcome: completed",
    );
  });

  test("net-negative projected replacements are left message-identical", async () => {
    const messages = completedTurn("tiny", "Small request");
    messages[1]!.parts.push(
      textPart("tiny-text", "source", "tiny-assistant", "ok"),
    );
    const before = structuredClone(messages);
    const requests = createCommandHarness(messages);

    await expect(executeMagicCompact(requests.v2, "source", 0)).rejects.toThrow(
      "projected no useful",
    );

    expect(messages).toEqual(before);
    expect(requests.backupForks).toEqual([]);
    expect(requests.partUpdates).toEqual([]);
  });

  test("stores only tool input/output in the omission cache, never assistant text", async () => {
    const assistantText = "Exact assistant text must stay only in the capsule.";
    const output = `large tool output ${"payload ".repeat(600)}`;
    const messages = [
      message("omit-user", "user", [
        textPart("omit-user-text", "source", "omit-user", "Request"),
      ]),
      message("omit-assistant", "assistant", [
        textPart("omit-text", "source", "omit-assistant", assistantText),
        genericToolPart("omit-read", "omit-assistant", "read", {
          status: "completed",
          input: { filePath: "/workspace/file.txt" },
          output,
          title: "Read file",
          metadata: {},
          time: { start: 1, end: 2 },
        }),
      ]),
    ];
    const requests = createCommandHarness(messages);

    await executeMagicCompact(requests.v2, "source", 0);

    const tool = messages
      .find(item => item.info.id === "omit-assistant")!
      .parts.find((part): part is ToolPart => part.type === "tool")!;
    const ids = omissionContentIDs(tool);
    expect(ids).toHaveLength(1);
    const cached = await readOmittedContent("source", ids[0]!);
    expect(cached).toBe(output);
    expect(cached).not.toContain(assistantText);
    expect(workJsonStrings(summaryText(messages, "omit-assistant"))).toEqual([
      assistantText,
    ]);
  });

  test("keeps a turn with the canonical v0 summary ID opaque", async () => {
    const oldSummary = "Canonical v0 summary text. Keep bytes and ID.";
    const assistantID = "canonical-v0-assistant";
    const legacyID = v0SummaryPartID(assistantID);
    const messages = [
      message("canonical-v0-user", "user", [
        textPart(
          "canonical-v0-user-text",
          "source",
          "canonical-v0-user",
          "Request",
        ),
      ]),
      message(assistantID, "assistant", [
        {
          ...textPart(legacyID, "source", assistantID, oldSummary),
          metadata: { magicCompact: { summary: true } },
        },
        textPart(
          "canonical-v0-new-text",
          "source",
          assistantID,
          "New exact deterministic capsule work.",
        ),
        {
          id: "canonical-v0-reasoning",
          sessionID: "source",
          messageID: assistantID,
          type: "reasoning",
          text: "removable reasoning ".repeat(6_000),
          time: { start: 1, end: 2 },
        } as Part,
      ]),
      ...completedTurn("canonical-v0-profitable", "Compact the later turn"),
    ];
    messages
      .find(item => item.info.id === "canonical-v0-profitable-assistant")!
      .parts.push({
        id: "canonical-v0-profitable-reasoning",
        sessionID: "source",
        messageID: "canonical-v0-profitable-assistant",
        type: "reasoning",
        text: "removable reasoning ".repeat(6_000),
        time: { start: 1, end: 2 },
      } as Part);
    const originalTurn = structuredClone(messages.slice(0, 2));

    await executeMagicCompact(createCommandHarness(messages).v2, "source", 0);

    expect(messages.slice(0, 2)).toEqual(originalTurn);
    expect(summaryParts(messages, assistantID)).toEqual([
      expect.objectContaining({ id: legacyID, text: oldSummary }),
    ]);
  });

  test("keeps a selected turn with the canonical v1 capsule ID byte-identical", async () => {
    const assistantID = "canonical-v1-assistant";
    const capsuleID = capsuleSummaryPartID(assistantID);
    const messages = [
      message("canonical-v1-user", "user", [
        textPart(
          "canonical-v1-user-text",
          "source",
          "canonical-v1-user",
          "Persist this selected turn exactly.",
        ),
      ]),
      message(assistantID, "assistant", [
        {
          ...textPart(
            capsuleID,
            "source",
            assistantID,
            "Outcome: completed\nWork:\n- Existing persisted capsule.",
          ),
          metadata: {
            magicCompact: { summary: true, capsule: true, version: 1 },
          },
        },
        textPart(
          "canonical-v1-later-text",
          "source",
          assistantID,
          "Persisted bytes after the capsule.",
        ),
        {
          id: "canonical-v1-reasoning",
          sessionID: "source",
          messageID: assistantID,
          type: "reasoning",
          text: "persisted opaque reasoning ".repeat(6_000),
          time: { start: 1, end: 2 },
        } as Part,
      ]),
      ...completedTurn("canonical-v1-profitable", "Compact the later turn"),
    ];
    messages
      .find(item => item.info.id === "canonical-v1-profitable-assistant")!
      .parts.push({
        id: "canonical-v1-profitable-reasoning",
        sessionID: "source",
        messageID: "canonical-v1-profitable-assistant",
        type: "reasoning",
        text: "removable reasoning ".repeat(6_000),
        time: { start: 1, end: 2 },
      } as Part);
    const originalTurn = structuredClone(messages.slice(0, 2));

    await executeMagicCompact(createCommandHarness(messages).v2, "source", 0);

    expect(messages.slice(0, 2)).toEqual(originalTurn);
    expect(summaryParts(messages, assistantID)).toEqual([
      expect.objectContaining({ id: capsuleID }),
    ]);
    expect(
      summaryText(messages, "canonical-v1-profitable-assistant"),
    ).toContain("Outcome: completed");
  });

  test("tool-only turns compact with a deterministic empty Work marker", async () => {
    const output = `tool-only output ${"payload ".repeat(600)}`;
    const messages = [
      message("tool-only-user", "user", [
        textPart("tool-only-user-text", "source", "tool-only-user", "Request"),
      ]),
      message("tool-only-assistant", "assistant", [
        genericToolPart("tool-only-read", "tool-only-assistant", "read", {
          status: "completed",
          input: { filePath: "/workspace/file.txt" },
          output,
          title: "Read file",
          metadata: {},
          time: { start: 1, end: 2 },
        }),
      ]),
    ];

    await executeMagicCompact(createCommandHarness(messages).v2, "source", 0);

    const capsule = summaryText(messages, "tool-only-assistant");
    expect(capsule).toContain("Work:\n- No eligible assistant text parts.");
  });

  test("allows only recognized Hive/task structured result schemas to affect current outcome", () => {
    const arbitrary = expectCanonicalCapsule(
      turnWithTool(
        genericToolPart("api-json", "assistant", "read", {
          status: "completed",
          input: { filePath: "response.json" },
          output: { status: "failed", error: "opaque API payload" },
          title: "Read API payload",
          metadata: {},
          time: { start: 1, end: 2 },
        }),
      ),
    );
    const blocked = expectCanonicalCapsule(
      turnWithTool(
        genericToolPart("hive-blocked", "assistant", "hive_worktree_commit", {
          status: "completed",
          input: { task: "03-remediation" },
          output: {
            terminal: true,
            status: "blocked",
            blocker: {
              reason: "Release credentials are missing.",
              recommendation: "Provide RELEASE_TOKEN.",
            },
            nextAction: "Provide RELEASE_TOKEN, then continue the task.",
          },
          title: "Complete task",
          metadata: {},
          time: { start: 1, end: 2 },
        }),
      ),
    );
    const malformed = expectCanonicalCapsule(
      turnWithTool(
        genericToolPart("hive-malformed", "assistant", "hive_worktree_commit", {
          status: "completed",
          input: { task: "03-remediation" },
          output: { status: "blocked" },
          title: "Complete task",
          metadata: {},
          time: { start: 1, end: 2 },
        }),
      ),
    );

    expect(arbitrary).toStartWith("Outcome: completed");
    expect(arbitrary).not.toContain("opaque API payload");
    expect(blocked).toStartWith("Outcome: partial");
    expect(capsuleSection(blocked, "Unresolved")).toContain(
      "Release credentials are missing.",
    );
    expect(capsuleSection(blocked, "Next")).toContain("Provide RELEASE_TOKEN");
    expect(malformed).toStartWith("Outcome: partial");
    expect(capsuleSection(malformed, "Unresolved")).toContain("malformed");
  });

  test.each([
    ["wrong ok type", { ok: "false" }],
    ["wrong success type", { success: 1 }],
    ["unsupported status", { status: "unknown" }],
    ["wrong status type", { ok: true, status: 42 }],
    [
      "success contradicts blocked status",
      {
        ok: true,
        status: "blocked",
        blocker: { reason: "Still blocked." },
      },
    ],
    [
      "failure contradicts completed status",
      { ok: false, status: "completed", error: "Still failed." },
    ],
    ["blocked status misses blocker", { status: "blocked" }],
    ["blocked status has invalid blocker", { status: "blocked", blocker: [] }],
    [
      "failed status has invalid error",
      { status: "failed", error: { message: 42 } },
    ],
  ] as const)("retains malformed recognized result: %s", (_name, value) => {
    const output = JSON.stringify(value);
    const capsule = structuredResultCapsule(output);

    expect(capsule).toStartWith("Outcome: partial");
    const unresolved = capsuleSection(capsule, "Unresolved");
    expect(unresolved).toContain("malformed structured result");
    expect(unresolved).toContain(output);
  });

  test("rejects non-finite recognized result structures as malformed", () => {
    const capsule = structuredResultCapsule({ ok: true, elapsed: Number.NaN });

    expect(capsule).toStartWith("Outcome: partial");
    expect(capsuleSection(capsule, "Unresolved")).toContain(
      "malformed structured result",
    );
    expect(capsuleSection(capsule, "Unresolved")).toContain("NaN");
  });

  test.each([
    ["ok success", { ok: true }, "completed"],
    ["supported done status", { success: true, status: "done" }, "completed"],
    [
      "blocked status",
      { status: "blocked", blocker: { reason: "Need approval." } },
      "partial",
    ],
    [
      "failure envelope",
      { ok: false, error: { message: "Task failed." } },
      "partial",
    ],
    [
      "partial status",
      { status: "partial", error: "Work remains." },
      "partial",
    ],
  ] as const)(
    "accepts valid recognized result: %s",
    (_name, value, outcome) => {
      const capsule = structuredResultCapsule(JSON.stringify(value));

      expect(capsule).toStartWith(`Outcome: ${outcome}`);
      expect(capsule).not.toContain("malformed structured result");
    },
  );

  test.each([
    ["3 passed, 2 skipped, 0 failed", "completed"],
    ["1 failed, 4 passed", "partial"],
    ["2 errors", "partial"],
    ["0 passed, 0 failed", "partial"],
    ["2 skipped, 1 todo, 1 cancelled", "partial"],
  ] as const)("classifies verification count groups: %s", (output, outcome) => {
    const capsule = expectCanonicalCapsule(
      turnWithTool(
        genericToolPart("verification", "assistant", "bash", {
          status: "completed",
          input: { command: "bun test", workdir: "/workspace" },
          output,
          title: "Shell command",
          metadata: { exit: 0 },
          time: { start: 1, end: 2 },
        }),
      ),
    );

    expect(capsule).toStartWith(`Outcome: ${outcome}`);
    if (outcome === "partial") {
      expect(capsuleSection(capsule, "Unresolved")).toContain(output);
    } else {
      expect(capsule).not.toContain("Unresolved:");
    }
  });

  test("not-run verification cannot resolve an earlier same-scope failure", () => {
    const command = "bun test packages/a.test.ts";
    const turn = turnWithTool(
      genericToolPart("verification-fail", "assistant", "bash", {
        status: "completed",
        input: { command, workdir: "/workspace/pkg" },
        output: "1 test failed",
        title: "Shell command",
        metadata: { exit: 0 },
        time: { start: 1, end: 2 },
      }),
      genericToolPart("verification-zero", "assistant", "bash", {
        status: "completed",
        input: { command, cwd: "/workspace/./pkg//" },
        output: "0 passed, 0 failed",
        title: "Shell command",
        metadata: { exit: 0 },
        time: { start: 1, end: 2 },
      }),
    );

    const capsule = expectCanonicalCapsule(turn);

    expect(capsule).toStartWith("Outcome: partial");
    expect(capsuleSection(capsule, "Unresolved")).toContain("1 test failed");
    expect(capsuleSection(capsule, "Unresolved")).toContain(
      "0 passed, 0 failed",
    );
  });

  test("same raw trimmed command and normalized absolute context resolve a prior failure", () => {
    const command = "bun test packages/a.test.ts";
    const capsule = expectCanonicalCapsule(
      turnWithTool(
        genericToolPart("verification-fail", "assistant", "bash", {
          status: "completed",
          input: { command, workdir: "/workspace/pkg" },
          output: "1 test failed",
          title: "Shell command",
          metadata: { exit: 0 },
          time: { start: 1, end: 2 },
        }),
        genericToolPart("verification-pass", "assistant", "bash", {
          status: "completed",
          input: { command: ` ${command} `, cwd: "/workspace/./pkg//" },
          output: "1 test passed, 0 failed",
          title: "Shell command",
          metadata: { exit: 0 },
          time: { start: 1, end: 2 },
        }),
      ),
    );

    expect(capsule).toStartWith("Outcome: completed");
    expect(capsule).not.toContain("Unresolved:");
  });

  test("relative or conflicting execution context and changed raw commands fail closed", () => {
    const relative = expectCanonicalCapsule(
      turnWithTool(
        genericToolPart("relative-fail", "assistant", "bash", {
          status: "completed",
          input: { command: "bun test", workdir: "pkg" },
          output: "1 test failed",
          title: "Shell command",
          metadata: { exit: 0 },
          time: { start: 1, end: 2 },
        }),
        genericToolPart("relative-pass", "assistant", "bash", {
          status: "completed",
          input: { command: "bun test", workdir: "pkg" },
          output: "1 test passed, 0 failed",
          title: "Shell command",
          metadata: { exit: 0 },
          time: { start: 1, end: 2 },
        }),
      ),
    );
    const rawCommand = expectCanonicalCapsule(
      turnWithTool(
        genericToolPart("raw-fail", "assistant", "bash", {
          status: "completed",
          input: {
            command: "bun test  packages/a.test.ts",
            workdir: "/workspace",
          },
          output: "1 test failed",
          title: "Shell command",
          metadata: { exit: 0 },
          time: { start: 1, end: 2 },
        }),
        genericToolPart("raw-pass", "assistant", "bash", {
          status: "completed",
          input: {
            command: "bun test packages/a.test.ts",
            workdir: "/workspace",
          },
          output: "1 test passed, 0 failed",
          title: "Shell command",
          metadata: { exit: 0 },
          time: { start: 1, end: 2 },
        }),
      ),
    );

    expect(relative).toStartWith("Outcome: partial");
    expect(rawCommand).toStartWith("Outcome: partial");
  });

  test("exit metadata and domain failures are fail-closed without parsing generic git words", () => {
    const git = expectCanonicalCapsule(
      turnWithTool(
        genericToolPart("git-log", "assistant", "bash", {
          status: "completed",
          input: { command: "git log --oneline -10", workdir: "/workspace" },
          output: "abc123 Fix fatal failure handling\ndef456 pending migration",
          title: "git log --oneline -10",
          metadata: { exit: 0 },
          time: { start: 1, end: 2 },
        }),
      ),
    );
    const nonzero = expectCanonicalCapsule(
      turnWithTool(
        genericToolPart("exit-nonzero", "assistant", "bash", {
          status: "completed",
          input: { command: "git status --short", workdir: "/workspace" },
          output: "fatal: not a git repository",
          title: "git status --short",
          metadata: { exit: 128 },
          time: { start: 1, end: 2 },
        }),
      ),
    );
    const invalid = expectCanonicalCapsule(
      turnWithTool(
        genericToolPart("exit-invalid", "assistant", "bash", {
          status: "completed",
          input: { command: "bun test", workdir: "/workspace" },
          output: "1 test passed, 0 failed",
          title: "bun test",
          metadata: { exit: "0" },
          time: { start: 1, end: 2 },
        }),
      ),
    );
    const domainFailure = expectCanonicalCapsule(
      turnWithTool(
        genericToolPart("domain-failure", "assistant", "bash", {
          status: "completed",
          input: { command: "bun test", workdir: "/workspace" },
          output: "1 test failed",
          title: "bun test",
          metadata: { exit: 0 },
          time: { start: 1, end: 2 },
        }),
      ),
    );

    expect(git).toStartWith("Outcome: completed");
    expect(git).not.toContain("Unresolved:");
    expect(nonzero).toStartWith("Outcome: partial");
    expect(invalid).toStartWith("Outcome: partial");
    expect(domainFailure).toStartWith("Outcome: partial");
  });

  test("reduces sixty-four thousand same-scope observations linearly", () => {
    const turn = alternatingVerificationTurn(64_000);
    const start = performance.now();

    const capsule = expectCanonicalCapsule(turn);

    expect(performance.now() - start).toBeLessThan(5_000);
    expect(capsule).toStartWith("Outcome: completed");
    expect(capsule).not.toContain("Unresolved:");
  });

  test("over-cap current unresolved state makes a turn uncompactable", () => {
    const turn = turnWithTool(
      genericToolPart("huge-error", "assistant", "bash", {
        status: "error",
        input: { command: "bun test", workdir: "/workspace" },
        error: `failed ${"x".repeat(300_000)}`,
        metadata: {},
        time: { start: 1, end: 2 },
      }),
    );

    expect(renderCanonicalCapsule(turn)).toBeNull();
  });

  test("source-backed incident fixture generates capsules without model requests and keeps exact assistant text", async () => {
    const messages = incidentSemanticSessionFixture();
    const sourceTexts = messages.flatMap(messageItem =>
      messageItem.info.role === "assistant"
        ? messageItem.parts.flatMap(part =>
            part.type === "text" ? [part.text] : [],
          )
        : [],
    );
    const plan = await planFor(messages, 0);
    const summaries = generateCompactionCapsules(plan);

    const capsules = summaries.filter(
      (summary): summary is string => summary !== null,
    );
    const recovered = capsules.flatMap(workJsonStrings);
    const capsuleMessages = capsules.map((capsule, index) =>
      message(`incident-capsule-${index}`, "assistant", [
        {
          ...textPart(
            `incident-capsule-part-${index}`,
            "source",
            `incident-capsule-${index}`,
            capsule,
          ),
          metadata: { magicCompact: { summary: true } },
        },
      ]),
    );

    expect(capsules).toHaveLength(3);
    expect(recovered).toEqual(sourceTexts);
    expect(recovered).toHaveLength(7);
    expect(countCanonicalContentTokens(capsuleMessages)).toBe(994);
    expect(capsules.join("\n")).not.toContain("Unresolved:");
  });

  test.each([
    [
      "malformed assistant marker",
      () => [
        ...completedTurn("bad-marker", "Before"),
        message("bad-marker-assistant", "assistant", [
          {
            id: "bad-marker-part",
            sessionID: "source",
            messageID: "bad-marker-assistant",
            type: "compaction",
            auto: true,
          } as unknown as Part,
        ]),
      ],
    ],
    [
      "overlapping completed checkpoints",
      () => [
        ...completedTurn("overlap-tail", "Before"),
        nativeMarker("overlap-marker-a", "overlap-tail-assistant"),
        nativeMarker("overlap-marker-b", "overlap-tail-assistant"),
        nativeSummary("overlap-summary-a", "overlap-marker-a"),
        nativeSummary("overlap-summary-b", "overlap-marker-b"),
        ...completedTurn("overlap-after", "After"),
      ],
    ],
    [
      "dangling marker",
      () => [
        ...completedTurn("dangling-tail", "Before"),
        nativeMarker("dangling-marker", "dangling-tail-assistant"),
        ...completedTurn("dangling-after", "After"),
      ],
    ],
  ] as const)("rejects %s native checkpoint state", async (_name, fixture) => {
    await expect(planFor(fixture(), 0)).rejects.toThrow();
  });

  test("valid native checkpoint freezes the prefix and compacts only the suffix", async () => {
    const messages = nativeCheckpointFixture();
    const plan = await planFor(messages, 0);
    const requests = createCommandHarness(messages);

    expect(
      plan.summarizedTurns.flatMap(turn =>
        turn.assistants.map(assistant => assistant.info.id),
      ),
    ).toEqual(["suffix-1-assistant", "suffix-2-assistant"]);
    await executeMagicCompact(requests.v2, "source", 0);

    expect(
      messages.find(item => item.info.id === "native-summary")!.parts[0],
    ).toMatchObject({ text: "Native summary text" });
    expect(
      messages
        .find(item => item.info.id === "tail-assistant")!
        .parts.some(
          part =>
            part.type === "text"
            && part.metadata?.magicCompact !== undefined
            && (part.metadata.magicCompact as Record<string, unknown>)[
              "summary"
            ] === true,
        ),
    ).toBeFalse();
    expect(summaryText(messages, "suffix-1-assistant")).toContain(
      "Outcome: completed",
    );
    expect(requests.events).toEqual(
      expect.arrayContaining(["summary-update", "boundary-prompt"]),
    );
  });

  test("checkpoint race before source writes aborts without mutation", async () => {
    const messages = nativeCheckpointFixture();
    const raced = structuredClone(messages);
    raced[0]!.parts.push(textPart("race", "source", "tail-user", "race"));
    const requests = createCommandHarness(messages, { racedMessages: raced });

    await expect(executeMagicCompact(requests.v2, "source", 0)).rejects.toThrow(
      NativeCheckpointChangedError,
    );

    expect(requests.backupForks).toEqual([]);
    expect(requests.partUpdates).toEqual([]);
    expect(requests.sourcePrompts).toEqual([]);
  });

  test("checkpoint race after backup preserves source and backup without pruning", async () => {
    const messages = nativeCheckpointFixture();
    const raced = [
      ...messages,
      nativeMarker("post-backup-raced-marker", "suffix-2-assistant"),
    ];
    const requests = createCommandHarness(messages, {
      racedAfterBackupMessages: raced,
    });

    await expect(executeMagicCompact(requests.v2, "source", 0)).rejects.toThrow(
      NativeCheckpointChangedError,
    );

    expect(requests.backupForks).toHaveLength(1);
    expect(requests.partUpdates).toEqual([]);
    expect(requests.partDeletes).toEqual([]);
    expect(requests.messageDeletes).toEqual([]);
    expect(requests.selectedSessions).toEqual([]);
    expect(requests.sessionDeletes).toEqual([]);
  });

  test("post-backup source failure rolls back through the backup session", async () => {
    const messages = completedTurn("rollback", "Rollback request");
    messages[1]!.parts.push({
      id: "rollback-reasoning",
      sessionID: "source",
      messageID: "rollback-assistant",
      type: "reasoning",
      text: "removable reasoning ".repeat(6_000),
      time: { start: 1, end: 2 },
    } as Part);
    const requests = createCommandHarness(messages, {
      sourcePromptError: new Error("boundary failed"),
    });

    await expect(executeMagicCompact(requests.v2, "source", 0)).rejects.toThrow(
      "boundary failed",
    );

    expect(requests.backupForks).toHaveLength(1);
    expect(requests.selectedSessions).toEqual(["backup"]);
    expect(requests.sessionDeletes).toContain("source");
  });

  test("deletes a partial backup when source omission cache copy fails", async () => {
    const sourceID = "copy-failure-source";
    const backupID = "copy-failure-backup";
    await mkdir(dirname(cachePath(sourceID)), { recursive: true });
    await writeFile(cachePath(sourceID), "not-json\n");
    const harness = createBackupHarness(backupID);

    try {
      await expect(
        createBackup(harness.v2, session(sourceID), 1),
      ).rejects.toThrow("Invalid omission cache");

      expect(harness.deletedSessions).toEqual([backupID]);
      expect(await Bun.file(cachePath(backupID)).exists()).toBeFalse();
      expect(await readStats(backupID)).toBeNull();
    } finally {
      await deleteCache(sourceID);
      await deleteStats(sourceID);
      await deleteCache(backupID);
      await deleteStats(backupID);
    }
  });

  test("cleans partial backup artifacts and aggregates cleanup failures", async () => {
    const sourceID = "metadata-failure-source";
    const backupID = "metadata-failure-backup";
    const omissionTarget = cachePath(backupID);
    const statsTarget = statsFilePath(backupID);
    const harness = createBackupHarness(backupID, {
      updateError: new Error("backup metadata update failed"),
      deleteError: new Error("partial backup delete failed"),
      onUpdate: async () => {
        await rm(omissionTarget, { force: true });
        await mkdir(omissionTarget, { recursive: true });
        await rm(statsTarget, { force: true });
        await mkdir(statsTarget, { recursive: true });
      },
    });
    let error: unknown;

    try {
      await createBackup(harness.v2, session(sourceID), 1);
    } catch (caught) {
      error = caught;
    }

    expect(error).toBeInstanceOf(AggregateError);
    expect(nestedErrorMessages(error)).toEqual(
      expect.arrayContaining([
        "backup metadata update failed",
        "Failed to delete partial backup omission cache.",
        "Failed to delete partial backup stats cache.",
        "partial backup delete failed",
      ]),
    );
    expect(harness.deletedSessions).toEqual([backupID]);

    await rm(omissionTarget, { recursive: true, force: true });
    await rm(statsTarget, { recursive: true, force: true });
    await deleteCache(sourceID);
    await deleteStats(sourceID);
  });

  test("selects the backup before deleting the source and leaves source intact when selection fails", async () => {
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

    operations.length = 0;
    const failing = {
      session: v2.session,
      tui: {
        selectSession: async () => {
          operations.push("select");
          return { error: "select failed" };
        },
      },
    } as unknown as V2Client;

    await expect(
      applyBackup(failing, session("source"), session("backup")),
    ).rejects.toThrow("select failed");
    expect(operations).toEqual(["update", "select"]);
  });

  test("source metadata update failure rolls back and source deletion failure is aggregated", async () => {
    const messages = completedTurn("source-update", "Rollback metadata");
    messages[1]!.parts.push({
      id: "source-update-reasoning",
      sessionID: "source",
      messageID: "source-update-assistant",
      type: "reasoning",
      text: "removable reasoning ".repeat(6_000),
      time: { start: 1, end: 2 },
    } as Part);
    const requests = createCommandHarness(messages, {
      sourceMetadataUpdateError: new Error("source metadata update failed"),
      sourceDeleteError: new Error("source deletion failed"),
    });
    let error: unknown;

    try {
      await executeMagicCompact(requests.v2, "source", 0);
    } catch (caught) {
      error = caught;
    }

    expect(error).toBeInstanceOf(AggregateError);
    expect(nestedErrorMessages(error)).toEqual(
      expect.arrayContaining([
        "source metadata update failed",
        "source deletion failed",
      ]),
    );
    expect(requests.selectedSessions).toEqual(["backup"]);
    expect(requests.sessionDeletes).toContain("source");
  });

  test("no eligible operation creates no backup boundary or source mutation", async () => {
    const messages = completedTurn("fresh", "Keep the current turn");
    const requests = createCommandHarness(messages);

    expect(await executeMagicCompact(requests.v2, "source", 1)).toBeFalse();

    expect(requests.backupForks).toEqual([]);
    expect(requests.sourcePrompts).toEqual([]);
    expect(requests.partUpdates).toEqual([]);
    expect(requests.partDeletes).toEqual([]);
    expect(requests.messageDeletes).toEqual([]);
    expect(requests.sessionUpdates).toEqual([]);
  });

  test("aggregate useful-savings gate runs after per-turn profitable selection without mutation", async () => {
    const messages = [
      message("aggregate-user", "user", [
        textPart("aggregate-user-text", "source", "aggregate-user", "Request"),
      ]),
      message("aggregate-assistant", "assistant", [
        textPart(
          "aggregate-text",
          "source",
          "aggregate-assistant",
          "Large exact response. ".repeat(300),
        ),
        {
          id: "aggregate-reasoning",
          sessionID: "source",
          messageID: "aggregate-assistant",
          type: "reasoning",
          text: "small removable detail ".repeat(80),
          time: { start: 1, end: 2 },
        } as Part,
      ]),
    ];
    const before = structuredClone(messages);
    const plan = await planFor(messages, 0);
    const projectedTurn = structuredClone(plan.summarizedTurns[0]!);
    const projectedAssistant = projectedTurn.assistants[0]!;
    projectedAssistant.parts.push({
      id: capsuleSummaryPartID(projectedAssistant.info.id),
      sessionID: "source",
      messageID: projectedAssistant.info.id,
      type: "text",
      text: expectCanonicalCapsule(projectedTurn),
      metadata: { magicCompact: { summary: true, capsule: true, version: 1 } },
    } satisfies TextPart);
    const projectedPrune = await createPruneProjection(
      [projectedTurn],
      "source",
    );
    const eligibleTokens = countCanonicalContentTokens(
      plan.summarizedTurns.flatMap(turn => turn.assistants),
    );
    const projectedTokens = countCanonicalContentTokens(
      projectedPrune.turns.flatMap(turn => turn.assistants),
    );
    const requests = createCommandHarness(messages);

    expect(projectedTokens).toBeLessThan(eligibleTokens);
    expect(eligibleTokens - projectedTokens).toBeLessThan(
      Math.max(32, Math.ceil(eligibleTokens * 0.2)),
    );
    await expect(executeMagicCompact(requests.v2, "source", 0)).rejects.toThrow(
      "useful canonical active-content savings",
    );

    expect(messages).toEqual(before);
    expect(requests.backupForks).toEqual([]);
    expect(requests.sourcePrompts).toEqual([]);
    expect(requests.partUpdates).toEqual([]);
    expect(requests.partDeletes).toEqual([]);
    expect(requests.messageDeletes).toEqual([]);
  });

  test("unprofitable candidates stay exact while profitable turns compact once", async () => {
    const messages = [
      ...completedTurn("stable-small", "Small request"),
      ...completedTurn("stable-large", "Large request"),
    ];
    messages
      .find(item => item.info.id === "stable-small-assistant")!
      .parts.push(
        textPart("stable-small-text", "source", "stable-small-assistant", "ok"),
      );
    messages
      .find(item => item.info.id === "stable-large-assistant")!
      .parts.push({
        id: "stable-large-reasoning",
        sessionID: "source",
        messageID: "stable-large-assistant",
        type: "reasoning",
        text: "removable reasoning ".repeat(6_000),
        time: { start: 1, end: 2 },
      } as Part);
    const smallBefore = structuredClone(
      messages.find(item => item.info.id === "stable-small-assistant")!,
    );
    const requests = createCommandHarness(messages);

    await executeMagicCompact(requests.v2, "source", 0);

    expect(
      messages.find(item => item.info.id === "stable-small-assistant"),
    ).toEqual(smallBefore);
    expect(summaryParts(messages, "stable-large-assistant")).toHaveLength(1);
    expect(
      requests.partUpdates.filter(item => item.messageID.endsWith("assistant")),
    ).toHaveLength(1);
  });

  test("failure toast failure retains the primary failure without starting backup", async () => {
    const messages = completedTurn("toast-failure", "Small request");
    messages[1]!.parts.push(
      textPart("toast-failure-text", "source", "toast-failure-assistant", "ok"),
    );
    const requests = createCommandHarness(messages, {
      failureToastError: new Error("failure toast failed"),
    });
    let error: unknown;

    try {
      await executeMagicCompact(requests.v2, "source", 0);
    } catch (caught) {
      error = caught;
    }

    expect(error).toBeInstanceOf(AggregateError);
    expect(nestedErrorMessages(error)).toEqual(
      expect.arrayContaining([
        expect.stringContaining("useful canonical active-content savings"),
        "failure toast failed",
      ]),
    );
    expect(requests.backupForks).toEqual([]);
    expect(requests.selectedSessions).toEqual([]);
    expect(requests.sessionDeletes).toEqual([]);
  });

  test("success toast failure surfaces only after durable mutation without rollback", async () => {
    const messages = completedTurn("success-toast", "Durable mutation");
    messages[1]!.parts.push({
      id: "success-toast-reasoning",
      sessionID: "source",
      messageID: "success-toast-assistant",
      type: "reasoning",
      text: "removable reasoning ".repeat(6_000),
      time: { start: 1, end: 2 },
    } as Part);
    const requests = createCommandHarness(messages, {
      successToastError: new Error("success toast failed"),
    });

    await expect(executeMagicCompact(requests.v2, "source", 0)).rejects.toThrow(
      "success toast failed",
    );

    expect(summaryParts(messages, "success-toast-assistant")).toHaveLength(1);
    expect(requests.events).toContain("summary-update");
    expect(requests.events).toContain("stats-prompt");
    expect(requests.events).not.toContain("select-backup");
    expect(requests.events).not.toContain("delete-source");
    expect(requests.events).not.toContain("failure-toast");
    expect(requests.selectedSessions).toEqual([]);
    expect(requests.sessionDeletes).toEqual([]);
  });

  test("final actual-after mismatch preserves source and backup before stats completion", async () => {
    await deleteStats("source");
    const messages = completedTurn("final-drift", "Drift request");
    messages[1]!.parts.push({
      id: "final-drift-reasoning",
      sessionID: "source",
      messageID: "final-drift-assistant",
      type: "reasoning",
      text: "removable reasoning ".repeat(6_000),
      time: { start: 1, end: 2 },
    } as Part);
    const requests = createCommandHarness(messages, {
      beforeFinalCount(source) {
        source
          .find(item => item.info.id === "final-drift-assistant")!
          .parts.push(
            textPart(
              "late-concurrent-content",
              "source",
              "final-drift-assistant",
              "Concurrent content after final source revalidation.",
            ),
          );
      },
    });
    let error: unknown;

    try {
      await executeMagicCompact(requests.v2, "source", 0);
    } catch (caught) {
      error = caught;
    }

    expect(error).toMatchObject({ name: "CompactionSourceDriftError" });
    expect(String(error)).toContain("projection mismatch");
    expect(requests.selectedSessions).toEqual([]);
    expect(requests.sessionDeletes).toEqual([]);
    expect(await readStats("source")).toBeNull();
    expect(
      messages.some(item =>
        item.parts.some(part => part.id === "late-concurrent-content"),
      ),
    ).toBeTrue();
  });

  test.each([
    [
      "added text part",
      (source: MessageWithParts[], assistantID: string) => {
        source
          .find(item => item.info.id === assistantID)!
          .parts.push(
            textPart(
              "pre-prune-added-text",
              "source",
              assistantID,
              "Concurrent text after capsule insertion.",
            ),
          );
      },
    ],
    [
      "added tool part",
      (source: MessageWithParts[], assistantID: string) => {
        source
          .find(item => item.info.id === assistantID)!
          .parts.push(
            genericToolPart("pre-prune-added-tool", assistantID, "read", {
              status: "completed",
              input: { filePath: "/workspace/concurrent.txt" },
              output: "concurrent tool output\n".repeat(180),
              title: "Concurrent read",
              metadata: {},
              time: { start: 3, end: 4 },
            }),
          );
      },
    ],
    [
      "changed existing part",
      (source: MessageWithParts[], assistantID: string) => {
        const part = source
          .find(item => item.info.id === assistantID)!
          .parts.find(
            item => item.id === "pre-prune-existing-text",
          ) as TextPart;
        part.text = "Concurrent replacement after capsule insertion.";
      },
    ],
  ] as const)(
    "rolls back before pruning or cache mutation on %s",
    async (name, mutate) => {
      await Promise.all([
        deleteCache("source"),
        deleteCache("backup"),
        deleteStats("source"),
        deleteStats("backup"),
      ]);
      await allocateOmission("source", { content: "existing cached bytes" });
      const originalStats = {
        version: 1 as const,
        sourceSessionId: "source",
        rootSessionId: "root-source",
        totalTokensPruned: 17,
        cachedTokensSaved: 5,
        processedAssistantMessageIds: ["already-counted"],
      };
      await writeStats("source", originalStats);
      const originalCache = await readCache("source");
      const id = `pre-prune-${name.replaceAll(" ", "-")}`;
      const assistantID = `${id}-assistant`;
      const messages = [
        message(`${id}-user`, "user", [
          textPart(`${id}-user-text`, "source", `${id}-user`, "Compact safely"),
        ]),
        message(assistantID, "assistant", [
          textPart(
            "pre-prune-existing-text",
            "source",
            assistantID,
            "Original exact assistant text.",
          ),
          {
            id: `${id}-reasoning`,
            sessionID: "source",
            messageID: assistantID,
            type: "reasoning",
            text: "removable reasoning ".repeat(6_000),
            time: { start: 1, end: 2 },
          } as Part,
          genericToolPart(`${id}-read`, assistantID, "read", {
            status: "completed",
            input: { filePath: "/workspace/original.txt" },
            output: "original planned tool output\n".repeat(180),
            title: "Original read",
            metadata: {},
            time: { start: 1, end: 2 },
          }),
        ]),
      ];
      const originalSource = structuredClone(messages);
      const requests = createCommandHarness(messages, {
        beforePruneReload: source => mutate(source, assistantID),
        restoreBackupOnSourceDelete: true,
      });
      let error: unknown;

      try {
        await executeMagicCompact(requests.v2, "source", 0);
      } catch (caught) {
        error = caught;
      }

      expect(error).toMatchObject({ name: "CompactionPruneSourceDriftError" });
      expect(String(error)).toContain("changed after capsule insertion");
      expect(requests.events).toContain("summary-update");
      expect(requests.partDeletes).toEqual([]);
      expect(requests.selectedSessions).toEqual(["backup"]);
      expect(requests.sessionDeletes).toContain("source");
      expect(messages).toEqual(originalSource);
      expect(await readCache("source")).toEqual(originalCache);
      expect(await readCache("backup")).toEqual(originalCache);
      expect(await readStats("source")).toEqual(originalStats);
      expect(await readStats("backup")).toEqual(originalStats);
      expect(
        requests.sessionUpdates.filter(
          request => request.sessionID === "source",
        ),
      ).toEqual([]);
    },
  );

  test("summary injection failure rolls back without boundary or pruning", async () => {
    const messages = completedTurn("summary-failure", "Inject summary");
    messages[1]!.parts.push({
      id: "summary-failure-reasoning",
      sessionID: "source",
      messageID: "summary-failure-assistant",
      type: "reasoning",
      text: "removable reasoning ".repeat(6_000),
      time: { start: 1, end: 2 },
    } as Part);
    const requests = createCommandHarness(messages, {
      partUpdateError(request) {
        const part = request.part as TextPart;
        return part.metadata?.magicCompact !== undefined
          && (part.metadata.magicCompact as Record<string, unknown>)["summary"]
            === true
          ? new Error("summary injection failed")
          : undefined;
      },
    });

    await expect(executeMagicCompact(requests.v2, "source", 0)).rejects.toThrow(
      "summary injection failed",
    );

    expect(requests.events).not.toContain("summary-update");
    expect(requests.events).not.toContain("boundary-prompt");
    expect(requests.partDeletes).toEqual([]);
    expect(requests.selectedSessions).toEqual(["backup"]);
    expect(requests.sessionDeletes).toContain("source");
  });

  test("pruning failure rolls back after summary and boundary writes", async () => {
    const messages = completedTurn("prune-failure", "Prune parts");
    messages[1]!.parts.push(
      textPart(
        "prune-failure-text",
        "source",
        "prune-failure-assistant",
        "Old assistant text",
      ),
      {
        id: "prune-failure-reasoning",
        sessionID: "source",
        messageID: "prune-failure-assistant",
        type: "reasoning",
        text: "removable reasoning ".repeat(6_000),
        time: { start: 1, end: 2 },
      } as Part,
    );
    const requests = createCommandHarness(messages, {
      partDeleteError: new Error("pruning failed"),
    });

    await expect(executeMagicCompact(requests.v2, "source", 0)).rejects.toThrow(
      "pruning failed",
    );

    expect(requests.events).toContain("summary-update");
    expect(requests.events).toContain("boundary-prompt");
    expect(requests.partDeletes).toHaveLength(1);
    expect(requests.selectedSessions).toEqual(["backup"]);
    expect(requests.sessionDeletes).toContain("source");
  });

  test("stats write failure rolls back after final count without source completion", async () => {
    const statsTarget = statsFilePath("source");
    await rm(statsTarget, { recursive: true, force: true });
    const messages = completedTurn("stats-failure", "Write stats");
    messages[1]!.parts.push({
      id: "stats-failure-reasoning",
      sessionID: "source",
      messageID: "stats-failure-assistant",
      type: "reasoning",
      text: "removable reasoning ".repeat(6_000),
      time: { start: 1, end: 2 },
    } as Part);
    const requests = createCommandHarness(messages, {
      beforeFinalCount: async () => {
        await rm(statsTarget, { recursive: true, force: true });
        await mkdir(statsTarget, { recursive: true });
      },
    });
    let error: unknown;

    try {
      await executeMagicCompact(requests.v2, "source", 0);
    } catch (caught) {
      error = caught;
    } finally {
      await rm(statsTarget, { recursive: true, force: true });
    }

    expect(error).toBeDefined();
    expect(error).not.toMatchObject({ name: "CompactionSourceDriftError" });
    expect(requests.selectedSessions).toEqual(["backup"]);
    expect(requests.sessionDeletes).toContain("source");
  });

  test("planned input and output omissions commit in order and remain retrievable", async () => {
    await deleteCache("source");
    const inputContent = "exact planned input\n".repeat(180);
    const outputContent = "exact planned output\n".repeat(180);
    const messages = completedTurn("planned-omissions", "Write artifact");
    messages[1]!.parts.push(
      genericToolPart("planned-write", "planned-omissions-assistant", "write", {
        status: "completed",
        input: { filePath: "/workspace/artifact.txt", content: inputContent },
        output: outputContent,
        title: "Write artifact",
        metadata: {},
        time: { start: 1, end: 2 },
      }),
    );

    await executeMagicCompact(createCommandHarness(messages).v2, "source", 0);

    const compacted = messages
      .flatMap(item => item.parts)
      .find((part): part is ToolPart => part.id === "planned-write")!;
    const contentIDs = omissionContentIDs(compacted);
    expect(contentIDs).toHaveLength(2);
    await expect(readOmittedContent("source", contentIDs[0]!)).resolves.toBe(
      inputContent,
    );
    await expect(readOmittedContent("source", contentIDs[1]!)).resolves.toBe(
      outputContent,
    );
  });

  test("second compaction advances lineage while preserving prior summaries and caches", async () => {
    await Promise.all([deleteCache("source"), deleteStats("source")]);
    const firstOutput = "first exact historical read\n".repeat(180);
    const secondOutput = "second exact historical read\n".repeat(180);
    const messages = nativeCheckpointFixture();
    messages
      .find(item => item.info.id === "suffix-1-assistant")!
      .parts.push(
        genericToolPart("first-lineage-read", "suffix-1-assistant", "read", {
          status: "completed",
          input: { filePath: "/workspace/first.txt" },
          output: firstOutput,
          title: "Read first artifact",
          metadata: {},
          time: { start: 1, end: 2 },
        }),
      );
    const nativeSummaryIndex = messages.findIndex(
      item => item.info.id === "native-summary",
    );
    const nativePrefix = structuredClone(
      messages.slice(0, nativeSummaryIndex + 1),
    );
    const exactUsers = userTextByPartID(messages);
    const requests = createCommandHarness(messages);

    await executeMagicCompact(requests.v2, "source", 1);
    const firstSummary = summaryText(messages, "suffix-1-assistant");
    const firstID = omissionContentIDs(
      messages
        .flatMap(item => item.parts)
        .find((part): part is ToolPart => part.id === "first-lineage-read")!,
    )[0]!;
    const firstStats = await readStats("source");
    expect(boundaryParts(messages)).toEqual([
      expect.objectContaining({ messageID: "suffix-2-user" }),
    ]);

    messages
      .find(item => item.info.id === "suffix-2-assistant")!
      .parts.push(
        {
          id: "suffix-2-removable-detail",
          sessionID: "source",
          messageID: "suffix-2-assistant",
          type: "reasoning",
          text: "second removable historical detail ".repeat(1_200),
          time: { start: 1, end: 2 },
        } as Part,
        genericToolPart("second-lineage-read", "suffix-2-assistant", "read", {
          status: "completed",
          input: { filePath: "/workspace/second.txt" },
          output: secondOutput,
          title: "Read second artifact",
          metadata: {},
          time: { start: 1, end: 2 },
        }),
      );
    messages.push(...completedTurn("suffix-3", "Current request"));
    const beforeSecond = countCanonicalContentTokens(messages);

    await executeMagicCompact(requests.v2, "source", 1);

    const secondSaved = beforeSecond - countCanonicalContentTokens(messages);
    const secondID = omissionContentIDs(
      messages
        .flatMap(item => item.parts)
        .find((part): part is ToolPart => part.id === "second-lineage-read")!,
    )[0]!;
    const secondStats = await readStats("source");
    expect(summaryText(messages, "suffix-1-assistant")).toBe(firstSummary);
    expect(boundaryParts(messages)).toHaveLength(1);
    expect(boundaryParts(messages)[0]!.messageID).not.toBe("suffix-2-user");
    expect(secondStats?.totalTokensPruned).toBe(
      firstStats!.totalTokensPruned + secondSaved,
    );
    expect(
      requests.sessionUpdates.findLast(
        request => request.sessionID === "source",
      )?.metadata,
    ).toMatchObject({ magicCompact: { compactionCount: 2 } });
    expect(messages.slice(0, nativeSummaryIndex + 1)).toEqual(nativePrefix);
    expect(userTextByPartID(messages)).toMatchObject(exactUsers);
    expect(await readOmittedContent("source", firstID)).toBe(firstOutput);
    expect(await readOmittedContent("source", secondID)).toBe(secondOutput);
  });

  test("successful compaction writes stats and preserves omission retrieval", async () => {
    await deleteCache("source");
    await writeStats("source", {
      version: 1,
      sourceSessionId: null,
      rootSessionId: "root-session",
      totalTokensPruned: 999,
      cachedTokensSaved: 11,
      processedAssistantMessageIds: [],
    });
    const messages = [
      message("stats-user", "user", [
        textPart("stats-user-text", "source", "stats-user", "Request"),
      ]),
      message("stats-assistant", "assistant", [
        textPart(
          "stats-text",
          "source",
          "stats-assistant",
          "Implemented stats.",
        ),
        genericToolPart("stats-read", "stats-assistant", "read", {
          status: "completed",
          input: { filePath: "/workspace/stats.txt" },
          output: `stats output ${"payload ".repeat(600)}`,
          title: "Read file",
          metadata: {},
          time: { start: 1, end: 2 },
        }),
      ]),
    ];
    const beforeTokens = countCanonicalContentTokens(messages);

    await executeMagicCompact(createCommandHarness(messages).v2, "source", 0);

    const afterTokens = countCanonicalContentTokens(messages);
    const stats = await readStats("source");
    expect(stats?.totalTokensPruned).toBe(999 + beforeTokens - afterTokens);
    expect(stats?.rootSessionId).toBe("root-session");
    expect(stats?.cachedTokensSaved).toBe(11);
    expect(stats?.sourceSessionId).toBe("source");
    expect(afterTokens).toBeLessThan(beforeTokens);
    const notice = messages
      .at(-1)!
      .parts.find((part): part is TextPart => part.type === "text")!;
    expect(notice.text).toBe(
      statsMessage(1, beforeTokens, afterTokens, stats!, "test-model"),
    );
    expect(notice.text).toContain("ignored reporting notice is excluded");
    const ids = omissionContentIDs(
      messages
        .find(item => item.info.id === "stats-assistant")!
        .parts.find((part): part is ToolPart => part.type === "tool")!,
    );
    expect(await readOmittedContent("source", ids[0]!)).toContain(
      "stats output",
    );
  });

  test("injectSummaries refuses missing summaries", async () => {
    const messages = completedTurn("inject", "Inject request");
    const plan = await planFor(messages, 0);
    const requests = createCommandHarness(messages);

    await expect(
      injectSummaries(requests.v2, "source", plan.summarizedTurns, []),
    ).rejects.toThrow("Missing summary");
  });
});

function exactTextMessages(): MessageWithParts[] {
  const [firstText, ignoredText, secondText] = exactTextFixtureStrings();
  return [
    message("hard-cut-user", "user", [
      textPart("hard-cut-user-text", "source", "hard-cut-user", "Request"),
    ]),
    message("hard-cut-assistant-1", "assistant", [
      textPart("hard-cut-text-1", "source", "hard-cut-assistant-1", firstText!),
      {
        ...textPart(
          "hard-cut-ignored",
          "source",
          "hard-cut-assistant-1",
          ignoredText!,
        ),
        ignored: true,
      },
      {
        ...textPart(
          "hard-cut-synthetic",
          "source",
          "hard-cut-assistant-1",
          "Synthetic assistant text is not compacted into Work.",
        ),
        synthetic: true,
      },
      commandToolPart(
        "hard-cut-tool",
        "hard-cut-assistant-1",
        "git status --short",
        "working tree clean",
      ),
    ]),
    message("hard-cut-assistant-2", "assistant", [
      textPart(
        "hard-cut-text-2",
        "source",
        "hard-cut-assistant-2",
        secondText!,
      ),
    ]),
  ];
}

function exactTextFixtureStrings(): string[] {
  return [
    'We chose SQLite.\n```ts\nconst placeholder = "[TODO but real]";\n```\nRemoved the old adapter.',
    "Ignored assistant text is still source text.",
    "Second assistant message text, after the tool skeleton.",
  ];
}

type HarnessOptions = {
  racedMessages?: MessageWithParts[];
  racedAfterBackupMessages?: MessageWithParts[];
  beforePruneReload?: (messages: MessageWithParts[]) => Promise<void> | void;
  beforeFinalCount?: (messages: MessageWithParts[]) => Promise<void> | void;
  restoreBackupOnSourceDelete?: boolean;
  sourcePromptError?: Error;
  sourceMetadataUpdateError?: Error;
  backupMetadataUpdateError?: Error;
  sourceDeleteError?: Error;
  selectBackupError?: Error;
  failureToastError?: Error;
  successToastError?: Error;
  partUpdateError?: (request: Record<string, unknown>) => Error | undefined;
  partDeleteError?: Error;
  messageDeleteError?: Error;
};

function createCommandHarness(
  messages: MessageWithParts[],
  options: HarnessOptions = {},
) {
  const events: string[] = [];
  const partUpdates: Array<{ part: Part; messageID: string }> = [];
  const partDeletes: Record<string, unknown>[] = [];
  const messageDeletes: Record<string, unknown>[] = [];
  const sessionUpdates: Record<string, unknown>[] = [];
  const sessionDeletes: string[] = [];
  const selectedSessions: string[] = [];
  const sourcePrompts: Record<string, unknown>[] = [];
  const backupForks: Record<string, unknown>[] = [];
  const toasts: Record<string, unknown>[] = [];
  let forks = 0;
  let sourceNotices = 0;
  let firstRevalidation = true;
  let pruneReloadHookApplied = false;
  let finalCountHookApplied = false;
  let currentSourceSession = session("source");
  let backupMessages: MessageWithParts[] | null = null;

  const v2 = {
    session: {
      messages: async () => {
        events.push("messages");
        if (
          !pruneReloadHookApplied
          && events.includes("summary-update")
          && (events.includes("boundary-prompt")
            || events.includes("boundary-update"))
          && options.beforePruneReload
        ) {
          pruneReloadHookApplied = true;
          await options.beforePruneReload(messages);
        }
        if (
          !finalCountHookApplied
          && events.includes("stats-prompt")
          && options.beforeFinalCount
        ) {
          finalCountHookApplied = true;
          await options.beforeFinalCount(messages);
        }
        if (forks > 0 && options.racedAfterBackupMessages) {
          return { data: options.racedAfterBackupMessages };
        }
        if (firstRevalidation) {
          firstRevalidation = false;
          return { data: messages };
        }
        return { data: options.racedMessages ?? messages };
      },
      get: async () => ({ data: currentSourceSession }),
      fork: async (request: Record<string, unknown>) => {
        backupForks.push(request);
        backupMessages = structuredClone(messages);
        forks += 1;
        return { data: session(forks === 1 ? "backup" : `backup-${forks}`) };
      },
      update: async (request: Record<string, unknown>) => {
        sessionUpdates.push(request);
        if (request.sessionID === "source") {
          events.push("update-source");
          if (options.sourceMetadataUpdateError) {
            throw options.sourceMetadataUpdateError;
          }
          currentSourceSession = {
            ...currentSourceSession,
            ...(request.metadata ? { metadata: request.metadata } : {}),
          } as Session;
        } else if (request.sessionID === "backup") {
          events.push("update-backup");
          if (options.backupMetadataUpdateError) {
            throw options.backupMetadataUpdateError;
          }
        }
        return { data: session(String(request.sessionID)) };
      },
      prompt: async (request: Record<string, unknown>) => {
        if (request.noReply !== true) {
          throw new Error("model prompt should not be called");
        }
        sourcePrompts.push(request);
        sourceNotices += 1;
        const parts = ((request.parts as Part[] | undefined) ?? []).map(
          (part, index) =>
            ({
              ...part,
              id: `source-notice-${sourceNotices}-part-${index}`,
              sessionID: "source",
              messageID: `source-notice-${sourceNotices}`,
            }) as Part,
        );
        if (
          parts.some(
            part =>
              part.type === "text"
              && part.metadata?.magicCompact !== undefined
              && (part.metadata.magicCompact as Record<string, unknown>)[
                "boundary"
              ] === true,
          )
        ) {
          events.push("boundary-prompt");
        }
        if (
          parts.some(
            part =>
              part.type === "text"
              && part.metadata?.magicCompact !== undefined
              && (part.metadata.magicCompact as Record<string, unknown>)[
                "stats"
              ] === true,
          )
        ) {
          events.push("stats-prompt");
        }
        if (options.sourcePromptError) throw options.sourcePromptError;
        messages.push(message(`source-notice-${sourceNotices}`, "user", parts));
        return {
          data: { info: { id: `source-notice-${sourceNotices}` }, parts },
        };
      },
      deleteMessage: async (request: Record<string, unknown>) => {
        messageDeletes.push(request);
        if (options.messageDeleteError) throw options.messageDeleteError;
        const index = messages.findIndex(
          item => item.info.id === request.messageID,
        );
        if (index !== -1) messages.splice(index, 1);
        return { data: true };
      },
      delete: async (request: { sessionID: string }) => {
        sessionDeletes.push(request.sessionID);
        if (request.sessionID === "source") {
          events.push("delete-source");
          if (options.sourceDeleteError) throw options.sourceDeleteError;
          if (
            options.restoreBackupOnSourceDelete
            && selectedSessions.includes("backup")
            && backupMessages
          ) {
            messages.splice(
              0,
              messages.length,
              ...structuredClone(backupMessages),
            );
          }
        }
        return { data: true };
      },
    },
    part: {
      update: async (request: Record<string, unknown>) => {
        const part = request.part as Part;
        const updateError = options.partUpdateError?.(request);
        if (updateError) throw updateError;
        partUpdates.push({ part, messageID: String(request.messageID) });
        const metadata = (part as TextPart).metadata as
          | { magicCompact?: { summary?: boolean; boundary?: boolean } }
          | undefined;
        if (metadata?.magicCompact?.summary) {
          events.push("summary-update");
        } else if (metadata?.magicCompact?.boundary) {
          events.push("boundary-update");
        }
        const target = messages.find(
          item => item.info.id === request.messageID,
        );
        if (target) {
          const index = target.parts.findIndex(item => item.id === part.id);
          if (index === -1) target.parts.push(part);
          else target.parts[index] = part;
        }
        return { data: part };
      },
      delete: async (request: Record<string, unknown>) => {
        partDeletes.push(request);
        if (options.partDeleteError) throw options.partDeleteError;
        const target = messages.find(
          item => item.info.id === request.messageID,
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
        if (request.title === "Magic Compact Failed") {
          events.push("failure-toast");
          if (options.failureToastError) throw options.failureToastError;
        } else if (
          request.title === "Magic Compact"
          && request.message !== COMPACT_NOOP
          && options.successToastError
        ) {
          throw options.successToastError;
        }
        return { data: true };
      },
      selectSession: async (request: { sessionID: string }) => {
        selectedSessions.push(request.sessionID);
        events.push("select-backup");
        if (options.selectBackupError) throw options.selectBackupError;
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
    backupForks,
    toasts,
  };
}

function message(
  id: string,
  role: "user" | "assistant",
  parts: Part[],
): MessageWithParts {
  return { info: { id, role } as Message, parts };
}

function session(id: string): Session {
  return {
    id,
    title: "Test session",
    directory: "/workspace",
    workspaceID: "workspace",
    model: { id: "test-model", providerID: "test-provider" },
  } as Session;
}

function textPart(
  id: string,
  sessionID: string,
  messageID: string,
  text: string,
): TextPart {
  return { id, sessionID, messageID, type: "text", text };
}

function completedTurn(id: string, text: string): MessageWithParts[] {
  const userID = `${id}-user`;
  return [
    message(userID, "user", [textPart(`${id}-text`, "source", userID, text)]),
    message(`${id}-assistant`, "assistant", []),
  ];
}

function nativeCheckpointFixture(): MessageWithParts[] {
  return [
    ...completedTurn("tail", "Pre-checkpoint tail"),
    nativeMarker("native-marker", "tail-assistant"),
    nativeSummary("native-summary", "native-marker"),
    ...completedTurn("suffix-1", "First suffix request"),
    ...completedTurn("suffix-2", "Second suffix request"),
  ].map(item => {
    if (item.info.id === "suffix-1-assistant") {
      item.parts.push({
        id: "suffix-1-reasoning",
        sessionID: "source",
        messageID: "suffix-1-assistant",
        type: "reasoning",
        text: "removable reasoning ".repeat(6_000),
        time: { start: 1, end: 2 },
      } as Part);
    }
    return item;
  });
}

function nativeMarker(id: string, tailStartID: string): MessageWithParts {
  return message(id, "user", [
    {
      id: `${id}-part`,
      sessionID: "source",
      messageID: id,
      type: "compaction",
      auto: true,
      tail_start_id: tailStartID,
    } as unknown as Part,
  ]);
}

function nativeSummary(id: string, parentID: string): MessageWithParts {
  return {
    info: {
      id,
      role: "assistant",
      parentID,
      summary: true,
      finish: "stop",
    } as Message,
    parts: [textPart(`${id}-text`, "source", id, "Native summary text")],
  };
}

function turnWithAssistantText(text: string): Turn {
  return {
    user: [message("user", "user", [])],
    assistants: [
      message("assistant", "assistant", [
        textPart("assistant-text", "source", "assistant", text),
      ]),
    ],
  };
}

function turnWithTool(...tools: Part[]): Turn {
  return {
    user: [message("user", "user", [])],
    assistants: [message("assistant", "assistant", tools)],
  };
}

function structuredResultCapsule(output: unknown): string {
  return expectCanonicalCapsule(
    turnWithTool(
      genericToolPart(
        "structured-result",
        "assistant",
        "hive_worktree_commit",
        {
          status: "completed",
          input: { task: "03-remediation" },
          output,
          title: "Complete task",
          metadata: {},
          time: { start: 1, end: 2 },
        },
      ),
    ),
  );
}

function commandToolPart(
  id: string,
  messageID: string,
  command: string,
  output: string,
): Part {
  return genericToolPart(id, messageID, "bash", {
    status: "completed",
    input: { command, workdir: "/workspace" },
    output,
    title: "Shell command",
    metadata: { exit: 0 },
    time: { start: 1, end: 2 },
  });
}

function genericToolPart(
  id: string,
  messageID: string,
  tool: string,
  state: Record<string, unknown>,
): Part {
  return {
    id,
    sessionID: "source",
    messageID,
    type: "tool",
    callID: `${id}-call`,
    tool,
    state,
  } as Part;
}

function alternatingVerificationTurn(count: number): Turn {
  const tools = Array.from({ length: count }, (_, index) =>
    genericToolPart(`alternating-${index}`, "assistant", "bash", {
      status: "completed",
      input: {
        command: "bun test packages/opencode-plugin/test/compact.test.ts",
        workdir: "/workspace",
      },
      output:
        index % 2 === 0
          ? `1 test failed in failure record ${index}`
          : `1 test passed, 0 failed in success record ${index}`,
      title: "Shell command",
      metadata: { exit: 0 },
      time: { start: 1, end: 2 },
    }),
  );
  return turnWithTool(...tools);
}

async function planFor(messages: MessageWithParts[], keepTurns: number) {
  const v2 = {
    session: { messages: async () => ({ data: messages }) },
  } as V2Client;
  return createCompactionPlan(v2, "source", keepTurns);
}

function v0SummaryPartID(messageID: string): string {
  return `prt_-magic_summary_${messageID}`;
}

function storedSummary(
  requests: { partUpdates: Array<{ part: Part; messageID: string }> },
  messageID: string,
): string {
  const update = requests.partUpdates.find(
    item => item.messageID === messageID,
  );
  expect(update).toBeDefined();
  return (update!.part as TextPart).text;
}

function expectCanonicalCapsule(turn: Turn): string {
  const capsule = renderCanonicalCapsule(turn);
  expect(capsule).not.toBeNull();
  return capsule!;
}

function summaryText(messages: MessageWithParts[], messageID: string): string {
  const part = summaryParts(messages, messageID)[0];
  expect(part).toBeDefined();
  return part!.text;
}

function summaryParts(
  messages: MessageWithParts[],
  messageID?: string,
): TextPart[] {
  return messages.flatMap(message =>
    messageID !== undefined && message.info.id !== messageID
      ? []
      : message.parts.filter(
          (part): part is TextPart =>
            part.type === "text"
            && part.metadata?.magicCompact !== undefined
            && (part.metadata.magicCompact as Record<string, unknown>)[
              "summary"
            ] === true,
        ),
  );
}

function boundaryParts(messages: MessageWithParts[]): TextPart[] {
  return messages.flatMap(message =>
    message.parts.filter(
      (part): part is TextPart =>
        part.type === "text"
        && part.metadata?.magicCompact !== undefined
        && (part.metadata.magicCompact as Record<string, unknown>)["boundary"]
          === true,
    ),
  );
}

function userTextByPartID(
  messages: MessageWithParts[],
): Record<string, string> {
  return Object.fromEntries(
    messages.flatMap(message =>
      message.info.role !== "user"
        ? []
        : message.parts.flatMap(part =>
            part.type === "text" && part.synthetic !== true
              ? [[part.id, part.text] as const]
              : [],
          ),
    ),
  );
}

function statsFilePath(sessionID: string): string {
  return join(dirname(cachePath(sessionID)), "stats", `${sessionID}.json`);
}

function nestedErrorMessages(error: unknown): string[] {
  if (error instanceof AggregateError) {
    return error.errors.flatMap(nestedErrorMessages);
  }
  if (error instanceof Error) {
    return [error.message];
  }
  return [String(error)];
}

function createBackupHarness(
  backupID: string,
  options: {
    updateError?: Error;
    deleteError?: Error;
    onUpdate?: () => Promise<void> | void;
  } = {},
) {
  const deletedSessions: string[] = [];
  const v2 = {
    session: {
      fork: async () => ({ data: session(backupID) }),
      update: async () => {
        await options.onUpdate?.();
        if (options.updateError) throw options.updateError;
        return { data: session(backupID) };
      },
      delete: async (request: { sessionID: string }) => {
        deletedSessions.push(request.sessionID);
        if (options.deleteError) throw options.deleteError;
        return { data: true };
      },
    },
  } as unknown as V2Client;
  return { v2, deletedSessions };
}

function workJsonStrings(value: string): string[] {
  const section = capsuleSection(value, "Work");
  return [...section.matchAll(/```json\n([\s\S]*?)\n```/g)].map(
    match => JSON.parse(match[1]!) as string,
  );
}

function capsuleSection(value: string, label: string): string {
  const start = value.indexOf(`${label}:\n`);
  expect(start).toBeGreaterThanOrEqual(0);
  const tail = value.slice(start);
  const next = tail
    .slice(label.length + 2)
    .match(/^(?:Work|Verification|VCS|Unresolved|Next):$/m);
  const end = next ? tail.indexOf(next[0], label.length + 2) : tail.length;
  return tail.slice(0, end).trim();
}

function omissionContentIDs(part: ToolPart): string[] {
  const state = part.state as unknown as Record<string, unknown>;
  const serialized = JSON.stringify([state["input"], state["output"]]);
  return [
    ...new Set(
      serialized.match(/[A-Za-z0-9_-]{1,12}:omitted-[A-Za-z0-9_-]{22}/g) ?? [],
    ),
  ];
}

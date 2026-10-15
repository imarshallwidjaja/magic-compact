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
import {
  connectListener,
  createListenerIdentity,
  LISTENER_REQUIRED,
  type V2Client,
} from "../src/api";
import {
  computeNativeBatchBudget,
  estimateNativeTurnCost,
  NATIVE_BATCH_FIXED_RESERVE_TOKENS,
  NATIVE_BATCH_PLANNING_NONCE,
  planNativeGenerationBatches,
  type NativeBatchProtocol,
} from "../src/compact/batch";
import { generateCompactionSummaries } from "../src/compact/compact";
import { projectManifestCoveredMedia } from "../src/compact/media";
import {
  ForkCleanupError,
  ForkHistoryError,
  forkSession,
} from "../src/compact/fork";
import {
  BOUNDARY_METADATA,
  MAGIC_COMPACT_SUMMARIZER_PROMPT,
  POST_COMPACTION_NOTICE,
  summaryEnvelopeTextPart,
} from "../src/compact/constants";
import {
  assertCompactionSourceUnchanged,
  buildTurns,
  createCompactionPlan,
  NativeCheckpointChangedError,
  planCompaction,
  snapshotSourceMessages,
  type CompactionPlan,
  type MessageWithParts,
  type Turn,
} from "../src/compact/plan";
import { readSourceAtRest, SuccessorFailure } from "../src/compact/successor";
import { hasUsefulSavings, successorFailure } from "../src/policy/successor";
import { preservedTurnsFeedback, statsMessage } from "../src/stats/constants";
import {
  countCanonicalContentTokens,
  countGptTokens,
} from "../src/stats/tokenize";
import {
  COMPACT_INSUFFICIENT_SAVINGS,
  COMPACT_NOOP,
  executeMagicCompact,
} from "../src/magic-compact";
import { executeMagicStats } from "../src/magic-stats";
import server from "../src/host-v1";
import tuiPlugin from "../src/tui";
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
import { RetentionClassificationError } from "../src/retention/classifier";
import {
  hostClock,
  persistedForkHistory,
  scoredJevFetch,
  summaryReply,
  type JevCall,
} from "./retention-host";
import {
  PRIVATE_SENTINEL,
  privateHostError,
  reachableStrings,
} from "./v2-host";

type TurnResult = {
  ordinal: number;
  marker?: string;
  summary: string;
  visualDescriptions: Array<{ ordinal: number; description: string }>;
};

type GenerationResponse = { turns: TurnResult[] };

let storageDirectory: string;
const HEURISTIC_OPTIONS = { retention: { mode: "heuristic" } } as const;
// N=1 over a single eligible turn preserves that turn and summarizes nothing.
const NOOP_PRESERVING_ONE_TURN = `${COMPACT_NOOP} ${preservedTurnsFeedback({ requested: 1, preserved: 1 })}`;
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

describe("chronological forks", () => {
  // The assistant has repeated part types with distinct content, so a reordered
  // part list cannot compare equal by type alone.
  const source = (
    timestamps: "ascending" | "equal" | "reversed" = "reversed",
  ) => {
    const reasoning = {
      id: "reasoning",
      sessionID: "source",
      messageID: "chronology-assistant",
      type: "reasoning",
      text: "First thought",
      time: { start: 1, end: 2 },
    } as Part;
    const stepStart = {
      id: "step-start",
      sessionID: "source",
      messageID: "chronology-assistant",
      type: "step-start",
    } as Part;
    const stepFinish = {
      id: "step-finish",
      sessionID: "source",
      messageID: "chronology-assistant",
      type: "step-finish",
      reason: "stop",
      cost: 0,
      tokens: {
        input: 1,
        output: 1,
        reasoning: 0,
        cache: { read: 0, write: 0 },
      },
    } as Part;
    const messages = [
      message("chronology-user", "user", [
        textPart("request", "chronology-user", "Exact request"),
      ]),
      message("chronology-assistant", "assistant", [
        imagePart(
          "image",
          "chronology-assistant",
          "data:image/png;base64,exact",
        ),
        stepStart,
        reasoning,
        textPart("answer-one", "chronology-assistant", "First answer"),
        textPart("answer-two", "chronology-assistant", "Second answer"),
        toolPart("tool", "chronology-assistant", {
          status: "completed",
          input: { path: "/file" },
          output: "Exact result",
          metadata: {},
          time: { start: 1, end: 2 },
        }),
        stepFinish,
      ]),
    ];
    if (messages[1]!.info.role === "assistant")
      messages[1]!.info.parentID = messages[0]!.info.id;
    const marker = nativeMarker("marker", messages[0]!.info.id);
    const summary = nativeSummary("summary", "marker");
    const history = [...messages, marker, summary];
    history.forEach((item, index) => {
      item.info.sessionID = "source";
      item.info.time = {
        created:
          timestamps === "ascending"
            ? index + 1
            : timestamps === "equal"
              ? 5
              : 9 - index,
      };
    });
    return history;
  };

  describe("population", () => {
    // Backward host clock corrections during a native copy: none, one
    // millisecond after the fourth write, 300 ms after the fifth.
    const CLOCKS = [
      ["steady", undefined],
      ["1 ms backward step", { afterWrites: 4, ms: 1 }],
      ["300 ms backward step", { afterWrites: 5, ms: 300 }],
    ] as const;

    test.each(
      (["ascending", "equal", "reversed"] as const).flatMap(timestamps =>
        CLOCKS.map(([name, step]) => [timestamps, name, step] as const),
      ),
    )(
      "populates forks with exact content, rewritten links and persisted chronology (%s timestamps, %s)",
      async (timestamps, _name, step) => {
        const messages = source(timestamps);
        const host = chronologyHost(messages, { clock: hostClock(step) });
        const before = structuredClone(host.history.read("source"));
        try {
          await forkSession(host.v2, sourceSession(), messages);
          const backup = host.history.read("backup");
          // One cut native fork, then one replay; no record is copied under a
          // host-minted ID.
          expect(host.forkRequests).toEqual([
            {
              sessionID: "source",
              directory: "/workspace",
              workspace: "workspace",
              messageID: "chronology-user",
            },
          ]);
          expect(host.replayCalls).toBe(1);
          expect(backup.map(item => item.info.role)).toEqual(
            messages.map(item => item.info.role),
          );
          expect(backup.map(item => item.info.time)).toEqual(
            messages.map(item => item.info.time),
          );
          expect(
            backup.map(item =>
              item.parts.map(part =>
                part.type === "text" || part.type === "reasoning"
                  ? `${part.type}:${part.text}`
                  : part.type,
              ),
            ),
          ).toEqual(
            messages.map(item =>
              item.parts.map(part =>
                part.type === "text" || part.type === "reasoning"
                  ? `${part.type}:${part.text}`
                  : part.type,
              ),
            ),
          );
          expect(backup[1]!.parts[0]).toMatchObject({
            url: "data:image/png;base64,exact",
          });
          const tool = messages[1]!.parts[5]!;
          expect(backup[1]!.parts[5]).toMatchObject({
            state: tool.type === "tool" ? tool.state : undefined,
          });
          expect((backup[1]!.info as { parentID: string }).parentID).toBe(
            backup[0]!.info.id,
          );
          expect(backup[2]!.parts[0]).toMatchObject({
            tail_start_id: backup[0]!.info.id,
          });
          expect((backup[3]!.info as { parentID: string }).parentID).toBe(
            backup[2]!.info.id,
          );
          for (const item of backup) {
            expect(item.info.sessionID).toBe("backup");
            for (const part of item.parts) {
              expect(part.sessionID).toBe("backup");
              expect(part.messageID).toBe(item.info.id);
            }
          }
          expect(host.history.read("source")).toEqual(before);
        } finally {
          host.history.close();
        }
      },
    );

    // 2100 messages with two parts each carry the minted ordinals across
    // several 4096-record windows of one clock reading.
    test("orders more than one counter window of records", async () => {
      const messages = Array.from({ length: 2100 }, (_, index) => {
        const id = `bulk-${index}`;
        return message(
          id,
          index % 2 === 0 ? "user" : "assistant",
          ["a", "b"].map(suffix =>
            textPart(`${id}-${suffix}`, id, `${id}-${suffix}`),
          ),
        );
      });
      messages.forEach((item, index) => {
        item.info.sessionID = "source";
        item.info.time = { created: 1 };
        if (item.info.role === "assistant")
          item.info.parentID = messages[index - 1]!.info.id;
      });
      const host = chronologyHost(messages);
      try {
        await forkSession(host.v2, sourceSession(), messages);
        const backup = host.history.read("backup");
        expect(
          backup.map(item =>
            item.parts.map(part => part.type === "text" && part.text),
          ),
        ).toEqual(
          messages.map(item =>
            item.parts.map(part => part.type === "text" && part.text),
          ),
        );
        const ids = backup.flatMap(item => [
          item.info.id,
          ...item.parts.map(part => part.id),
        ]);
        expect(new Set(ids).size).toBe(ids.length);
        expect(ids.length).toBeGreaterThan(4096);
        expect(backup.map(item => item.info.id)).toEqual(
          [...backup.map(item => item.info.id)].sort(),
        );
      } finally {
        host.history.close();
      }
    });

    test("forks an empty history natively without replay and rejects any copied record", async () => {
      const empty = chronologyHost([]);
      try {
        const fork = await forkSession(empty.v2, sourceSession(), []);
        expect(fork.id).toBe("backup");
        expect(empty.forkRequests).toEqual([
          {
            sessionID: "source",
            directory: "/workspace",
            workspace: "workspace",
          },
        ]);
        expect(empty.replayCalls).toBe(0);
        expect(empty.deleted).toEqual([]);
      } finally {
        empty.history.close();
      }

      // A message appended after planning makes the uncut native fork copy it.
      const appended = chronologyHost(source());
      try {
        await expect(
          forkSession(appended.v2, sourceSession(), []),
        ).rejects.toThrow(
          "chronological fork initialization failed (fork was not empty)",
        );
        expect(appended.replayCalls).toBe(0);
        expect(appended.deleted).toEqual(["backup"]);
      } finally {
        appended.history.close();
      }
    });
  });

  test.each([
    "missing-parent",
    "forward-parent",
    "self-parent",
    "empty-parent",
    "missing-tail",
    "forward-tail",
    "self-tail",
    "empty-tail",
    "foreign-session",
    "duplicate-message",
    "foreign-part-session",
    "foreign-part-owner",
    "duplicate-part",
    "unsafe-created",
  ])("rejects %s source records before any host mutation", async kind => {
    const messages = source();
    const assistant = messages[1]!.info;
    const marker = messages[2]!.parts[0]!;
    if (assistant.role === "assistant" && kind.endsWith("parent")) {
      assistant.parentID =
        kind === "missing-parent"
          ? "missing"
          : kind === "forward-parent"
            ? messages[2]!.info.id
            : kind === "self-parent"
              ? assistant.id
              : "";
    }
    if (marker.type === "compaction" && kind.endsWith("tail")) {
      marker.tail_start_id =
        kind === "missing-tail"
          ? "missing"
          : kind === "forward-tail"
            ? messages[3]!.info.id
            : kind === "self-tail"
              ? messages[2]!.info.id
              : "";
    }
    if (kind === "foreign-session") assistant.sessionID = "elsewhere";
    if (kind === "duplicate-message") messages[3]!.info.id = assistant.id;
    if (kind === "foreign-part-session")
      messages[1]!.parts[1]!.sessionID = "elsewhere";
    if (kind === "foreign-part-owner")
      messages[1]!.parts[1]!.messageID = "elsewhere";
    if (kind === "duplicate-part")
      messages[1]!.parts[1]!.id = messages[0]!.parts[0]!.id;
    if (kind === "unsafe-created") assistant.time = { created: 1.5 };
    const host = chronologyHost(messages);
    try {
      await expect(
        forkSession(host.v2, sourceSession(), messages),
      ).rejects.toThrow(
        "chronological fork initialization failed (source history was invalid)",
      );
      expect(host.forkCalls).toBe(0);
      expect(host.reads).toEqual([]);
      expect(host.replayCalls).toBe(0);
      expect(host.deleted).toEqual([]);
      expect(await readCache("backup")).toBeNull();
      expect(await readStats("backup")).toBeNull();
    } finally {
      host.history.close();
    }
  });

  test.each(["edit", "append"] as const)(
    "rejects a concurrent source %s before backup initialization",
    async kind => {
      const messages = source();
      const host = chronologyHost(messages);
      const change = () => {
        if (kind === "edit") {
          const first = host.history.read("source")[0]!;
          if (first.parts[0]!.type === "text")
            first.parts[0]!.text = "Concurrent request edit";
          host.history.write(first);
        } else {
          const appended = message("concurrent", "user", []);
          appended.info.sessionID = "source";
          appended.info.time = { created: 50 };
          host.history.write(appended, 50);
        }
      };
      // An edit lands while the fork is populated; an append before the cut.
      if (kind === "edit") host.afterReplay = change;
      else host.beforeFork = change;
      try {
        await expect(
          forkSession(host.v2, sourceSession(), messages),
        ).rejects.toThrow(
          "chronological fork initialization failed (source history changed)",
        );
        expect(host.deleted).toEqual(["backup"]);
        expect(await readCache("backup")).toBeNull();
        expect(await readStats("backup")).toBeNull();
      } finally {
        host.history.close();
      }
    },
  );

  test("keeps persisted replay chronology when source timestamps race", async () => {
    const messages = source();
    const host = chronologyHost(messages);
    host.beforeFork = () => {
      const first = host.history.read("source")[0]!;
      first.info.time = { created: 99 };
      host.history.write(first);
    };
    host.afterReplay = () => {
      const second = host.history.read("source")[1]!;
      second.info.time = { created: 100 };
      host.history.write(second);
    };
    try {
      await forkSession(host.v2, sourceSession(), messages);
      const backup = host.history.read("backup");
      expect(backup.map(item => item.info.time)).toEqual(
        messages.map(item => item.info.time),
      );
      expect(backup.map(item => item.info.role)).toEqual(
        messages.map(item => item.info.role),
      );
      expect((backup[1]!.info as { parentID: string }).parentID).toBe(
        backup[0]!.info.id,
      );
      expect(
        host.history.read("source").map(item => item.info.time.created),
      ).toEqual([99, 100, 7, 6]);
      expect(snapshotSourceMessages(host.history.read("source"))).toEqual(
        snapshotSourceMessages(messages),
      );
      expect(host.deleted).toEqual([]);
    } finally {
      host.history.close();
    }
  });

  test.each([
    "role",
    "parent",
    "part-id",
    "part-owner",
    "part-type",
    "content",
    "order",
    "part-order",
  ])(
    "rejects %s corruption on persisted reload before backup initialization",
    async kind => {
      const messages = source();
      const host = chronologyHost(messages);
      host.afterReplay = () => {
        const fork = host.history.read("backup");
        const target = fork[1]!;
        if (kind === "role") target.info.role = "user";
        if (kind === "parent" && target.info.role === "assistant")
          target.info.parentID = "foreign-parent";
        if (kind === "part-id") target.parts[0]!.id = "foreign-part";
        if (kind === "part-owner")
          target.parts[0]!.messageID = "foreign-message";
        if (kind === "part-type") target.parts[0]!.type = "text";
        if (kind === "content" && target.parts[0]!.type === "file")
          target.parts[0]!.url = "changed";
        if (kind === "part-order") target.parts.reverse();
        if (kind === "order")
          host.history.deleteMessage("backup", target.info.id);
        host.history.write(target, kind === "order" ? 20 : 1);
      };
      try {
        await expect(
          forkSession(host.v2, sourceSession(), messages),
        ).rejects.toThrow(
          "chronological fork initialization failed (replayed fork did not match the plan)",
        );
        expect(host.deleted).toEqual(["backup"]);
        expect(host.history.read("source")).toEqual(messages);
        expect(await readCache("backup")).toBeNull();
        expect(await readStats("backup")).toBeNull();
      } finally {
        host.history.close();
      }
    },
  );

  test("rejects a replay acknowledged for another session", async () => {
    const messages = source();
    const host = chronologyHost(messages);
    host.replayedSession = "elsewhere";
    try {
      await expect(
        forkSession(host.v2, sourceSession(), messages),
      ).rejects.toThrow(
        "chronological fork initialization failed (replayed fork did not match the plan)",
      );
      expect(host.reads).toHaveLength(1);
      expect(host.deleted).toEqual(["backup"]);
    } finally {
      host.history.close();
    }
  });

  test("awaits partial replay cleanup and reports a sanitized orphan if deletion fails", async () => {
    const messages = source();
    const host = chronologyHost(messages);
    host.afterReplay = () => {
      throw new Error("private replay body");
    };
    host.deleteError = new Error("private cleanup body");
    try {
      let failure: unknown;
      try {
        await forkSession(host.v2, sourceSession(), messages);
      } catch (error) {
        failure = error;
      }
      expect(failure).toBeInstanceOf(ForkCleanupError);
      expect(nestedErrorMessages(failure).join(" ")).not.toContain("private");
      expect(String(failure)).toContain("(history replay failed)");
      expect(String(failure)).toContain("partial fork backup manually");
      expect(host.deleted).toEqual(["backup"]);
      expect(host.history.read("source")).toEqual(messages);
    } finally {
      host.history.close();
    }
  });

  describe("history reads", () => {
    const TIMEOUT_MS = 40;
    const never = () => new Promise<never>(() => {});
    const FAULTS = {
      reject: () => Promise.reject(new Error("private read body")),
      error: () => ({ error: { message: "private read body" } }),
      timeout: never,
    } as const;
    const prepared = () => {
      const messages = source();
      return { messages, host: chronologyHost(messages) };
    };
    const rejection = async (promise: Promise<unknown>) => {
      try {
        await promise;
      } catch (error) {
        return error;
      }
      throw new Error("Expected rejection.");
    };
    const wait = (ms: number) =>
      new Promise(resolve => setTimeout(resolve, ms));

    // Reads: empty cut, replayed fork, live source.
    test.each(
      [1, 2, 3].flatMap(read =>
        (["reject", "error", "timeout"] as const).map(
          fault => [read, fault] as const,
        ),
      ),
    )(
      "recovers a failed first attempt without repeating fork or replay (read %i, %s)",
      async (read, fault) => {
        const { messages, host } = prepared();
        host.readOverride = (_sessionID, number) =>
          number === read ? FAULTS[fault]() : undefined;
        try {
          const fork = await forkSession(
            host.v2,
            sourceSession(),
            messages,
            TIMEOUT_MS,
          );
          expect(fork.id).toBe("backup");
          expect(host.reads).toHaveLength(4);
          expect(host.forkCalls).toBe(1);
          expect(host.replayCalls).toBe(1);
          expect(host.deleted).toEqual([]);
          expect(host.history.read("source")).toEqual(messages);
        } finally {
          host.history.close();
        }
      },
    );

    test.each([
      ["timeout", "history read timed out", "read-timeout"],
      ["reject", "history read failed", "read-failed"],
      ["error", "history read failed", "read-failed"],
    ] as const)(
      "stops after two %s attempts with a sanitized reason and awaited cleanup",
      async (fault, detail, reason) => {
        const { messages, host } = prepared();
        host.readOverride = (_sessionID, number) =>
          number <= 2 ? FAULTS[fault]() : undefined;
        const started = Date.now();
        try {
          const failure = await rejection(
            forkSession(host.v2, sourceSession(), messages, TIMEOUT_MS),
          );
          expect(failure).toBeInstanceOf(ForkHistoryError);
          expect((failure as ForkHistoryError).reason).toBe(reason);
          expect(String(failure)).toContain(`(${detail})`);
          expect(String(failure)).not.toContain("private");
          // Each attempt gets its own full deadline.
          if (fault === "timeout")
            expect(Date.now() - started).toBeGreaterThanOrEqual(
              2 * TIMEOUT_MS - 5,
            );
          expect(host.reads).toHaveLength(2);
          expect(host.forkCalls).toBe(1);
          expect(host.replayCalls).toBe(0);
          expect(host.deleted).toEqual(["backup"]);
          expect(host.history.read("source")).toEqual(messages);
        } finally {
          host.history.close();
        }
      },
    );

    test("allows each logical read its own two attempts and no more", async () => {
      const { messages, host } = prepared();
      host.readOverride = (_sessionID, number) =>
        number === 2 || number === 3 ? FAULTS.reject() : undefined;
      try {
        const failure = await rejection(
          forkSession(host.v2, sourceSession(), messages, TIMEOUT_MS),
        );
        expect((failure as ForkHistoryError).reason).toBe("read-failed");
        // Empty-cut read, then two failed verification attempts.
        expect(host.reads).toHaveLength(3);
        expect(host.forkCalls).toBe(1);
        expect(host.replayCalls).toBe(1);
        expect(host.deleted).toEqual(["backup"]);
      } finally {
        host.history.close();
      }
    });

    test("measures each attempt against its own full deadline", async () => {
      const { messages, host } = prepared();
      host.readOverride = async (_sessionID, number) => {
        if (number === 1) return never();
        // Longer than half a deadline: a shared window would already be spent.
        await wait(TIMEOUT_MS * 0.75);
        return undefined;
      };
      try {
        const started = Date.now();
        await forkSession(host.v2, sourceSession(), messages, TIMEOUT_MS);
        expect(Date.now() - started).toBeGreaterThan(TIMEOUT_MS);
        expect(host.reads).toHaveLength(4);
      } finally {
        host.history.close();
      }
    });

    test.each(["stale data", "rejection"] as const)(
      "ignores an abandoned first read that settles late with %s",
      async late => {
        const { messages, host } = prepared();
        let settle!: (value: unknown) => void;
        let fail!: (error: unknown) => void;
        const abandoned = new Promise((resolve, reject) => {
          settle = resolve;
          fail = reject;
        });
        host.readOverride = (_sessionID, number) =>
          number === 1 ? abandoned : undefined;
        try {
          const fork = await forkSession(
            host.v2,
            sourceSession(),
            messages,
            TIMEOUT_MS,
          );
          if (late === "rejection") fail(new Error("private late body"));
          else settle({ data: [] });
          await wait(10);
          expect(fork.id).toBe("backup");
          expect(host.reads).toHaveLength(4);
          expect(host.deleted).toEqual([]);
          expect(host.forkCalls).toBe(1);
        } finally {
          host.history.close();
        }
      },
    );

    test.each([
      ["a successful response without data", ["empty"], "read-invalid"],
      ["a non-empty replay cut", ["nonempty"], "cut-not-empty"],
      [
        "a non-empty replay cut after a failed first attempt",
        ["fail", "nonempty"],
        "cut-not-empty",
      ],
    ] as const)("never retries %s", async (_name, script, reason) => {
      const { messages, host } = prepared();
      host.readOverride = (_sessionID, number) => {
        const step = script[number - 1];
        if (step === "fail") return FAULTS.reject();
        if (step === "empty") return {};
        if (step === "nonempty") return { data: host.history.read("source") };
        return { data: [] };
      };
      try {
        const failure = await rejection(
          forkSession(host.v2, sourceSession(), messages, TIMEOUT_MS),
        );
        expect((failure as ForkHistoryError).reason).toBe(reason);
        expect(host.reads).toHaveLength(script.length);
        expect(host.forkCalls).toBe(1);
        expect(host.replayCalls).toBe(0);
        expect(host.deleted).toEqual(["backup"]);
      } finally {
        host.history.close();
      }
    });

    test("keeps the read reason in the aggregate when partial-fork deletion fails", async () => {
      const { messages, host } = prepared();
      host.readOverride = () => FAULTS.timeout();
      host.deleteError = new Error("private cleanup body");
      try {
        const failure = await rejection(
          forkSession(host.v2, sourceSession(), messages, TIMEOUT_MS),
        );
        expect(failure).toBeInstanceOf(ForkCleanupError);
        const [primary] = (failure as ForkCleanupError).errors;
        expect(primary).toBeInstanceOf(ForkHistoryError);
        expect((primary as ForkHistoryError).reason).toBe("read-timeout");
        expect(String(failure)).toContain("(history read timed out)");
        expect(String(failure)).toContain(
          "delete partial fork backup manually",
        );
        expect(nestedErrorMessages(failure).join(" ")).not.toContain("private");
        expect(host.deleted).toEqual(["backup"]);
      } finally {
        host.history.close();
      }
    });
  });
});

describe("native generation batch planning", () => {
  const budget = computeNativeBatchBudget({
    context: 1_000_000,
    input: 500_000,
    output: 64_000,
  });
  const marker = (ordinal: number, nonce = NATIVE_BATCH_PLANNING_NONCE) =>
    `<magic-compact-turn nonce="${nonce}" ordinal="${ordinal}" />`;
  const estimate = (
    ordinal: number,
    inputTokens: number,
    outputTokens = 100,
    expectedInlineImages = 0,
  ) => ({
    ordinal,
    marker: marker(ordinal),
    expectedInlineImages,
    inputTokens,
    outputTokens,
  });
  type Estimate = ReturnType<typeof estimate>;
  const renderExplicitRequest = (
    candidates: Estimate[],
    globalIndexPrompt = '{"globalTurnIndex":[]}',
  ) =>
    [
      "Generate summaries from retained native history.",
      globalIndexPrompt,
      JSON.stringify({
        turns: candidates.map(candidate => ({
          ordinal: candidate.ordinal,
          marker: candidate.marker,
          expectedInlineImages: candidate.expectedInlineImages,
        })),
      }),
    ].join("\n");
  const generationProtocol = (
    globalIndexPrompt = '{"globalTurnIndex":[]}',
  ): NativeBatchProtocol<Estimate> => ({
    explicitRequestTokens: candidates =>
      countGptTokens(renderExplicitRequest(candidates, globalIndexPrompt))
      + candidates.reduce(
        (total, candidate) => total + countGptTokens(candidate.marker),
        0,
      ),
    responseProtocolTokens: candidates =>
      countGptTokens(
        JSON.stringify({
          turns: candidates.map(candidate => ({
            ordinal: candidate.ordinal,
            marker: candidate.marker,
            summary: "",
            visualDescriptions: Array.from(
              { length: candidate.expectedInlineImages },
              (_, index) => ({ ordinal: index + 1, description: "" }),
            ),
          })),
        }),
      ),
  });
  const plan = (
    candidates: Estimate[],
    candidateBudget = budget,
    protocol = generationProtocol(),
  ) => planNativeGenerationBatches(candidates, candidateBudget, protocol);

  test("derives output, normal, hard-turn, and halo limits from the configured model", () => {
    expect(NATIVE_BATCH_PLANNING_NONCE).toHaveLength(22);
    expect(countGptTokens(NATIVE_BATCH_PLANNING_NONCE)).toBe(22);
    expect(budget).toEqual({
      outputTarget: 32_000,
      usableInput: 500_000,
      normalInputTarget: 400_000,
      hardSingleTurnInput: 450_000,
      haloInputLimit: 450_000,
    });

    expect(
      computeNativeBatchBudget({ context: 400_000, output: 64_000 }),
    ).toEqual({
      outputTarget: 32_000,
      usableInput: 368_000,
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
      usableInput: 384_000,
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

  test("uses the complete rendered request at the normal input boundary", () => {
    const template = estimate(1, 0);
    const explicitRequestTokens = generationProtocol().explicitRequestTokens([
      template,
    ]);
    const exactHistoryTokens =
      budget.normalInputTarget
      - NATIVE_BATCH_FIXED_RESERVE_TOKENS
      - explicitRequestTokens;
    const below = plan([estimate(1, exactHistoryTokens - 1)]);
    const exact = plan([estimate(1, exactHistoryTokens)]);
    const above = plan([estimate(1, exactHistoryTokens + 1), estimate(2, 1)]);

    expect(below).toMatchObject([
      {
        estimatedInputTokens: 399_999,
        inputLimit: budget.normalInputTarget,
        candidates: [{ ordinal: 1 }],
      },
    ]);
    expect(exact).toMatchObject([
      {
        estimatedInputTokens: 400_000,
        inputLimit: budget.normalInputTarget,
        candidates: [{ ordinal: 1 }],
      },
    ]);
    expect(above).toMatchObject([
      {
        estimatedInputTokens: 400_001,
        inputLimit: budget.hardSingleTurnInput,
        candidates: [{ ordinal: 1 }],
      },
      { inputLimit: budget.normalInputTarget, candidates: [{ ordinal: 2 }] },
    ]);
  });

  test("plans one target per request with its global index charged once", () => {
    const protocol = generationProtocol(
      JSON.stringify({
        globalTurnIndex: [{ sourceOrdinal: 1, anchor: "global orientation" }],
      }),
    );
    const candidates = [
      estimate(1, 1, 1),
      estimate(2, 1, 1),
      estimate(3, 1, 1),
    ];
    const batches = plan(candidates, budget, protocol);

    expect(protocol.explicitRequestTokens(candidates)).toBeLessThan(
      budget.normalInputTarget,
    );
    expect(batches.map(batch => batch.candidates)).toEqual(
      candidates.map(candidate => [candidate]),
    );
    for (const [index, batch] of batches.entries()) {
      const candidate = candidates[index]!;
      expect(batch).toEqual({
        candidates: [candidate],
        estimatedInputTokens:
          NATIVE_BATCH_FIXED_RESERVE_TOKENS
          + candidate.inputTokens
          + protocol.explicitRequestTokens([candidate]),
        estimatedExplicitRequestTokens: protocol.explicitRequestTokens([
          candidate,
        ]),
        estimatedOutputTokens:
          candidate.outputTokens + protocol.responseProtocolTokens([candidate]),
        inputLimit: budget.normalInputTarget,
      });
    }
  });

  test("uses the complete explicit request at the usable-input boundary", () => {
    const globalIndexPrompt = JSON.stringify({
      globalTurnIndex: [{ sourceOrdinal: 1, anchor: "orientation" }],
    });
    const protocol = generationProtocol(globalIndexPrompt);
    const exactUsableInput =
      NATIVE_BATCH_FIXED_RESERVE_TOKENS + protocol.explicitRequestTokens([]);
    expect(
      plan([], { ...budget, usableInput: exactUsableInput }, protocol),
    ).toEqual([]);
    expect(() =>
      plan([], { ...budget, usableInput: exactUsableInput - 1 }, protocol),
    ).toThrow(
      /explicit request.*global turn index.*magic-trim.*input capacity/i,
    );
  });

  test("uses the complete rendered request at the hard single-turn boundary", () => {
    const template = estimate(1, 0);
    const explicitRequestTokens = generationProtocol().explicitRequestTokens([
      template,
    ]);
    const exactHistoryTokens =
      budget.hardSingleTurnInput
      - NATIVE_BATCH_FIXED_RESERVE_TOKENS
      - explicitRequestTokens;
    expect(plan([estimate(1, exactHistoryTokens)])).toMatchObject([
      { estimatedInputTokens: 450_000, candidates: [{ ordinal: 1 }] },
    ]);
    expect(() => plan([estimate(1, exactHistoryTokens + 1)])).toThrow(
      /magic-trim.*first/i,
    );
  });

  test("adds the complete response envelope, markers, and visuals to content output", () => {
    const candidate = estimate(17, 1, 700, 2);
    const protocolTokens = generationProtocol().responseProtocolTokens([
      candidate,
    ]);
    const batches = plan([candidate]);

    expect(batches[0]!.estimatedOutputTokens).toBe(700 + protocolTokens);
    expect(batches[0]!.estimatedOutputTokens).toBeLessThanOrEqual(
      budget.outputTarget,
    );
  });

  test("requires more model output capacity when one complete turn response exceeds the target", () => {
    const template = estimate(1, 1, 0);
    const responseProtocolTokens = generationProtocol().responseProtocolTokens([
      template,
    ]);
    expect(() =>
      plan([estimate(1, 1, budget.outputTarget - responseProtocolTokens + 1)]),
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
        plan([estimate(1, cost.inputTokens, cost.outputTokens)]),
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
  test("generates from persisted chronology despite reversed JSON creation times", async () => {
    const turn = candidateTurn("chronology", "Continue exact task", [
      toolPart("tool", "chronology-assistant", {
        status: "completed",
        input: {},
        output: "listing",
        metadata: {},
        time: { start: 1, end: 2 },
      }),
    ]);
    turn.user[0]!.info.time = { created: 7 };
    turn.assistants[0]!.info.time = { created: 1 };
    turn.user[0]!.info.sessionID = "source";
    turn.assistants[0]!.info.sessionID = "source";
    if (turn.assistants[0]!.info.role === "assistant")
      turn.assistants[0]!.info.parentID = turn.user[0]!.info.id;
    const harness = generationHarness([turn], responseFor([turn]));
    const before = structuredClone(turn);
    try {
      await expect(
        generateCompactionSummaries(
          harness.v2,
          sourceSession(),
          generationPlan([turn]),
        ),
      ).resolves.toHaveLength(1);
      expect(
        harness.promptHistories[0]!.messages.map(item => item.info.role),
      ).toEqual(["user", "assistant"]);
      expect(harness.messageDeletes).toEqual([]);
      expect(harness.deletes).toEqual(["generation-fork"]);
      expect(turn).toEqual(before);
    } finally {
      harness.history!.close();
    }
  });
  test("connects only to a listener that returns this instance's identity nonce", async () => {
    const identity = createListenerIdentity();
    const config = {} as Config;
    identity.register(config);
    const requests: Request[] = [];
    const listenerFetch = (agent: unknown) =>
      (async (request: Request) => {
        requests.push(request);
        return Response.json({ agent });
      }) as unknown as typeof fetch;
    const input = {
      serverUrl: new URL("http://127.0.0.1:4101/"),
      directory: "/workspace",
    };

    await expect(
      connectListener(input, identity, {
        env: { OPENCODE_SERVER_PASSWORD: "secret" },
        fetch: listenerFetch(config.agent),
      }),
    ).resolves.toBeDefined();
    const request = requests[0]!;
    expect(new URL(request.url).origin).toBe("http://127.0.0.1:4101");
    expect(request.headers.get("authorization")).toBe(
      `Basic ${Buffer.from("opencode:secret").toString("base64")}`,
    );
    expect(new URL(request.url).searchParams.get("directory")).toBe(
      "/workspace",
    );

    // Another instance on the same port returns its own nonce.
    const decoy = createListenerIdentity();
    const decoyConfig = {} as Config;
    decoy.register(decoyConfig);
    await expect(
      connectListener(input, identity, {
        env: {},
        fetch: listenerFetch(decoyConfig.agent),
      }),
    ).rejects.toThrow(LISTENER_REQUIRED);
    await expect(
      connectListener(input, identity, {
        env: {},
        fetch: (async () => {
          throw new Error("connection refused");
        }) as unknown as typeof fetch,
      }),
    ).rejects.toThrow(LISTENER_REQUIRED);
    expect(LISTENER_REQUIRED).toContain("opencode --port 0");
  });

  test("refuses compact, trim and stats with listener guidance before any session work", async () => {
    // Another OpenCode instance owns the port the plugin was told about.
    const requests: string[] = [];
    const decoy = Bun.serve({
      port: 0,
      hostname: "127.0.0.1",
      fetch: request => {
        requests.push(new URL(request.url).pathname);
        return Response.json({ agent: {} });
      },
    });
    try {
      const hooks = await server({
        serverUrl: new URL(`http://127.0.0.1:${decoy.port}/`),
        directory: "/workspace",
      } as PluginInput);
      await hooks.config?.({} as Config);
      for (const command of ["magic-compact", "magic-trim", "magic-stats"]) {
        await expect(
          hooks["command.execute.before"]!(
            { command, arguments: "", sessionID: "session" },
            { parts: [] },
          ),
        ).rejects.toThrow(LISTENER_REQUIRED);
      }
      expect(requests).toEqual(["/config", "/config", "/config"]);
    } finally {
      await decoy.stop(true);
    }
  });

  test("registers a hidden summarizer that keeps the operator's model and denies operational tools", async () => {
    const hooks = await server({} as PluginInput);
    const config = {
      agent: {
        "magic-compact-summarizer": {
          model: "operator-provider/operator-model",
          variant: "high",
          permission: { bash: "allow" },
        },
      },
    } as unknown as Config;

    await hooks.config?.(config);

    expect(config.agent?.["magic-compact-summarizer"]).toEqual({
      model: "operator-provider/operator-model",
      variant: "high",
      description: "Continuation summarizer for Magic Compact",
      hidden: true,
      mode: "subagent",
      prompt: MAGIC_COMPACT_SUMMARIZER_PROMPT,
      permission: { "*": "deny" },
    });
    expect(config.agent?.["magic-compact-listener-identity"]).toMatchObject({
      disable: true,
    });
    expect(MAGIC_COMPACT_SUMMARIZER_PROMPT).toContain(
      "Do not call operational tools",
    );
    expect(config.command?.["magic-compact"]).toBeDefined();
    expect(config.command?.["magic-trim"]).toBeDefined();
    expect(config.command?.["magic-stats"]).toBeDefined();
    expect(hooks.tool?.read_omitted_content).toBeDefined();
    expect(tuiPlugin.id).toBe("magic-compact");
    expect(
      pluginStorageDirectory().endsWith("/opencode/storage/magic-compact"),
    ).toBeTrue();
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
    expect(prompt.length).toBeLessThan(3_500);
    expect(prompt).toContain('"ordinal":1');
    expect(prompt).toContain('"expectedInlineImages":1');
    expect(prompt).not.toContain("data:");
    expect(prompt).not.toContain("encrypted:");
    expect(prompt).not.toContain("incident-tool");
    expect(prompt).not.toContain(attachmentURL.slice(0, 100));
    expect(parts.some(part => part.type === "file")).toBeFalse();
  });

  test("indexes every complete post-checkpoint source turn with bounded deterministic anchors", async () => {
    const longAnchor = Array.from({ length: 170 }, (_, index) =>
      String.fromCharCode(33 + (index % 90)),
    ).join("");
    const escapedAnchor = 'quoted "anchor" \\ slash';
    const duplicateAnchor = "indexed-opaque request";
    const messages = [
      ...candidateMessages("indexed-prefix", "prefix request", []),
      nativeMarker("indexed-checkpoint-marker"),
      nativeSummary("indexed-checkpoint-summary", "indexed-checkpoint-marker"),
      ...candidateMessages(
        "indexed-target",
        `${longAnchor}\nignored second line`,
        [],
      ),
      ...persistedSummaryTurn("indexed-opaque", 2),
      ...candidateMessages("indexed-empty", "", []),
      ...candidateMessages("indexed-tail-duplicate", duplicateAnchor, []),
      ...candidateMessages(
        "indexed-tail-escaped",
        `${escapedAnchor}\nignored second line`,
        [],
      ),
      message("indexed-active-user", "user", [
        textPart(
          "indexed-active-text",
          "indexed-active-user",
          "active incomplete turn",
        ),
      ]),
    ];
    const planClient = {
      session: { messages: async () => ({ data: messages }) },
    } as unknown as V2Client;
    const plan = await createCompactionPlan(planClient, "source", 2);
    const harness = generationHarness(
      plan.summarizedTurns,
      {
        turns: [
          result(1, "Indexed target summary."),
          result(2, "Empty-anchor target summary."),
        ],
      },
      { sourceMessages: messages },
    );

    await generateCompactionSummaries(harness.v2, sourceSession(), plan);

    const globalIndex = generationGlobalTurnIndex(harness.prompts[0]!);
    expect(globalIndex).toEqual([
      {
        sourceOrdinal: 3,
        anchor: Array.from(longAnchor).slice(0, 160).join(""),
      },
      { sourceOrdinal: 4, anchor: duplicateAnchor },
      { sourceOrdinal: 5, anchor: "" },
      { sourceOrdinal: 6, anchor: duplicateAnchor },
      { sourceOrdinal: 7, anchor: escapedAnchor },
    ]);
    expect(generationGlobalTurnIndexText(harness.prompts[0]!)).not.toContain(
      "active incomplete turn",
    );
  });

  test("keeps distant assistant and tool detail out of the prompt and retained fork", async () => {
    const distantPayload = "DISTANT_ASSISTANT_TOOL_PAYLOAD";
    const distant = candidateTurn(
      "global-distant",
      "distant orientation anchor",
      [
        toolPart("global-distant-tool", "global-distant-assistant", {
          status: "completed",
          input: { path: "/distant" },
          output: distantPayload,
          title: "Read",
          metadata: {},
          time: { start: 1, end: 2 },
        }),
      ],
    );
    distant.assistants[0]!.parts.push({
      ...textPart(
        "global-distant-summary",
        "global-distant-assistant",
        "Distant persisted summary detail.",
      ),
      metadata: { magicCompact: { summary: true, version: 2 } },
    });
    const spacer = candidateTurn("global-spacer", "spacer anchor", []);
    spacer.assistants[0]!.parts.push({
      ...textPart(
        "global-spacer-summary",
        "global-spacer-assistant",
        "Spacer persisted summary detail.",
      ),
      metadata: { magicCompact: { summary: true, version: 2 } },
    });
    const target = candidateTurn("global-target", "target anchor", []);
    const harness = generationHarness([distant, spacer, target], {
      turns: [result(1, "Target summary.")],
    });

    await generateCompactionSummaries(
      harness.v2,
      sourceSession(),
      generationPlan([distant, spacer, target]),
    );

    expect(generationGlobalTurnIndex(harness.prompts[0]!)).toContainEqual({
      sourceOrdinal: 1,
      anchor: "distant orientation anchor",
    });
    expect(promptText(harness.prompts[0]!)).not.toContain(distantPayload);
    expect(JSON.stringify(harness.promptHistories[0]!.messages)).not.toContain(
      distantPayload,
    );
    expect(harness.promptHistories[0]!.messages).toHaveLength(4);
    expect(generationPromptIndex(harness.prompts[0]!).turns).toHaveLength(1);
  });

  test("rejects an unaffordable global index before creating a fork", async () => {
    const alphabet =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()_+-=[]{};:,.<>/?";
    const turns = Array.from({ length: 32 }, (_, turnIndex) =>
      candidateTurn(
        `index-capacity-${turnIndex + 1}`,
        Array.from(
          { length: 160 },
          (_, index) =>
            alphabet[(index * 37 + turnIndex * 13) % alphabet.length],
        ).join(""),
        [],
      ),
    );
    const harness = generationHarness(turns, responseFor(turns), {
      modelLimits: {
        context: 1_000_000,
        input: 8_900,
        output: 64_000,
      },
    });

    await expect(
      generateCompactionSummaries(
        harness.v2,
        sourceSession(),
        generationPlan(turns),
      ),
    ).rejects.toThrow(/global turn index.*magic-trim.*more input capacity/i);
    expect(harness.forks).toEqual([]);
    expect(harness.prompts).toEqual([]);
  });

  test("denies operational tools while permitting one structured summarizer response", async () => {
    const turns = [candidateTurn("first", "alpha", [])];
    const harness = generationHarness(turns, responseFor(turns));

    const summaries = await generateCompactionSummaries(
      harness.v2,
      sourceSession(),
      generationPlan(turns),
    );

    expect(summaries).toHaveLength(1);
    expect(harness.events).toEqual([
      "agents",
      "providers",
      "messages",
      "fork",
      // Empty cut, replayed fork, live source.
      "messages",
      "messages",
      "messages",
      "deny",
      "messages",
      "messages",
      "marker",
      "prompt",
      "delete",
    ]);
    expect(harness.updates).toEqual([
      {
        sessionID: "generation-fork",
        directory: "/workspace",
        workspace: "workspace",
        permission: [
          { permission: "*", pattern: "*", action: "deny" },
          { permission: "StructuredOutput", pattern: "*", action: "allow" },
        ],
      },
    ]);
    expect(harness.prompts[0]).toMatchObject({
      sessionID: "generation-fork",
      directory: "/workspace",
      workspace: "workspace",
      agent: "magic-compact-summarizer",
      model: { providerID: "configured-provider", modelID: "configured-model" },
      variant: "configured-variant",
      format: {
        type: "json_schema",
        schema: {
          type: "object",
          required: ["turns"],
          additionalProperties: false,
        },
      },
    });
    expect(harness.creates).toEqual([]);
    expect(harness.forks).toEqual([
      {
        sessionID: "source",
        directory: "/workspace",
        workspace: "workspace",
        // Exclusive cut before the first message; history arrives by replay.
        messageID: "first-user",
      },
    ]);
    expect(harness.deletes).toEqual(["generation-fork"]);
    const format = harness.prompts[0]!["format"] as {
      retryCount?: number;
      schema: Record<string, unknown>;
    };
    expect(format.retryCount).toBeUndefined();
    expect(format.schema).toMatchObject({
      properties: {
        turns: {
          type: "array",
          minItems: 1,
          maxItems: 1,
          items: {
            required: ["ordinal", "marker", "summary", "visualDescriptions"],
            additionalProperties: false,
            properties: {
              visualDescriptions: {
                items: {
                  required: ["ordinal", "description"],
                  additionalProperties: false,
                },
              },
            },
          },
        },
      },
    });
    const turnSchema = (
      format.schema as {
        properties: { turns: { items: { properties: object } } };
      }
    ).properties.turns.items;
    expect(Object.keys(turnSchema.properties)).toEqual([
      "ordinal",
      "marker",
      "summary",
      "visualDescriptions",
    ]);
  });

  test("accepts structured output independently of text parts", async () => {
    const turn = candidateTurn("structured", "anchor", []);
    const harness = generationHarness([turn], responseFor([turn]), {
      responseText: "invalid JSON from an unrelated text part",
    });

    const summaries = await generateCompactionSummaries(
      harness.v2,
      sourceSession(),
      generationPlan([turn]),
    );

    expect(summaries[0]?.text).toBe("Turn 1 outcome and continuation details.");
    expect(harness.aborts).toEqual([]);
    expect(harness.deletes).toEqual(["generation-fork"]);
  });

  test("rejects text-only output without a fallback or leaking its contents", async () => {
    const turn = candidateTurn("text-only", "private request", []);
    const secret = "UNTRUSTED_PROVIDER_BODY_7c1";
    const harness = generationHarness([turn], responseFor([turn]), {
      omitStructured: true,
      responseText: JSON.stringify({
        turns: [result(1, secret)],
      }),
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
    expect(failure).toBeInstanceOf(Error);
    expect((failure as Error).message).toBe(
      "Compaction generation response validation failed: missing structured output.",
    );
    expect(nestedErrorMessages(failure).join(" ")).not.toContain(secret);
    expect(nestedErrorMessages(failure).join(" ")).not.toContain(
      "private request",
    );
    expect(harness.aborts).toEqual(["generation-fork"]);
    expect(harness.deletes).toEqual(["generation-fork"]);
  });

  test("isolates nine targets into nine one-target forks with one immutable global index", async () => {
    const turns = Array.from({ length: 9 }, (_, index) =>
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
    expect(harness.forks).toHaveLength(9);
    expect(harness.prompts).toHaveLength(9);
    expect(harness.deletes).toHaveLength(9);
    expect(harness.completedPromptFirstOrdinals.at(-1)).toBe(1);
    // Forks prompt concurrently, so compare requests in target-ordinal order.
    const requests = harness.prompts
      .map((request, position) => ({
        request,
        index: generationPromptIndex(request),
        history: harness.promptHistories[position]!.messages,
      }))
      .toSorted(
        (left, right) =>
          left.index.turns[0]!.ordinal - right.index.turns[0]!.ordinal,
      );
    expect(requests.map(({ index }) => index.turns)).toEqual(
      turns.map((_, position) => [
        {
          ordinal: position + 1,
          marker: expect.stringMatching(
            new RegExp(
              `^<magic-compact-turn nonce="[A-Za-z0-9_-]{22}" ordinal="${position + 1}" />$`,
              "u",
            ),
          ),
          expectedInlineImages: 0,
        },
      ]),
    );
    expect(requests.map(({ index }) => Object.keys(index))).toEqual(
      turns.map(() => ["turns"]),
    );
    // Each fork keeps its target and the preceding turn, never a later one.
    // Forks remint IDs, so identify turns by their user text.
    expect(
      requests.map(({ history }) =>
        history.map(message =>
          message.info.role === "user"
            ? message.parts.find(
                part => part.type === "text" && part.synthetic !== true,
              )
            : "assistant",
        ),
      ),
    ).toEqual(
      turns.map((_, position) =>
        [position - 1, position]
          .filter(turn => turn >= 0)
          .flatMap(turn => [
            expect.objectContaining({ text: `anchor ${turn + 1}` }),
            "assistant",
          ]),
      ),
    );
    const globalIndexText = generationGlobalTurnIndexText(requests[0]!.request);
    expect(generationGlobalTurnIndex(requests[0]!.request)).toHaveLength(9);
    for (const [position, { request, index, history }] of requests.entries()) {
      const text = promptText(request);
      expect(generationTurnsSchema(request)).toMatchObject({
        minItems: 1,
        maxItems: 1,
      });
      expect(text).toContain("turns array must contain exactly one object");
      // OpenCode 1 adds the marker to the turn's first user message.
      expect(text).toContain(
        "either added to that turn's first user message or sent as a separate synthetic message directly before it",
      );
      expect(generationGlobalTurnIndexText(request)).toBe(globalIndexText);
      expect(text.split(globalIndexText)).toHaveLength(2);
      expect(text).not.toContain("outcome and continuation details.");
      expect(JSON.stringify(history)).toContain(
        JSON.stringify(index.turns[0]!.marker),
      );
      for (const [other, otherRequest] of requests.entries()) {
        if (other === position) continue;
        const otherNonce = otherRequest.index.turns[0]!.marker.match(
          /nonce="([A-Za-z0-9_-]{22})"/u,
        )![1]!;
        expect(text).not.toContain(otherNonce);
        expect(JSON.stringify(history)).not.toContain(otherNonce);
      }
    }
    expect(harness.partDeletes).toEqual([]);
    const noncesByFork = new Map<string, Set<string>>();
    for (const update of harness.markerUpdates) {
      const nonce = update.part.text.match(/nonce="([A-Za-z0-9_-]{22})"/u)?.[1];
      expect(nonce).toBeDefined();
      const nonces = noncesByFork.get(update.sessionID) ?? new Set<string>();
      nonces.add(nonce!);
      noncesByFork.set(update.sessionID, nonces);
    }
    expect([...noncesByFork.values()].map(nonces => nonces.size)).toEqual(
      Array.from({ length: 9 }, () => 1),
    );
    expect(
      new Set([...noncesByFork.values()].flatMap(nonces => [...nonces])).size,
    ).toBe(9);
  });

  test("fits summary and protocol within a small output target without charging Unicode user anchors", async () => {
    const highTokenLine = "\u{10ffff}".repeat(300);
    const turns = Array.from({ length: 2 }, (_, index) =>
      candidateTurn(`unicode-output-${index + 1}`, highTokenLine, []),
    );
    const modelLimits = {
      context: 1_000_000,
      input: 500_000,
      output: 1_536,
    };
    const budget = computeNativeBatchBudget(modelLimits);
    const harness = generationHarness(turns, responseFor(turns), {
      modelLimits,
    });

    await generateCompactionSummaries(
      harness.v2,
      sourceSession(),
      generationPlan(turns),
    );

    expect(
      harness.prompts
        .map(generationPromptIndex)
        .map(index => index.turns.length),
    ).toEqual([1, 1]);
    expect(
      harness.prompts.every(request => {
        const actualTokens = estimatedGenerationResponseTokens(request, turns);
        const recordedTokens = estimatedGenerationResponseTokens(
          request,
          turns,
          true,
        );
        const anchorTokens = countGptTokens(
          generationGlobalTurnIndex(request)[0]!.anchor,
        );
        return (
          actualTokens <= recordedTokens
          && recordedTokens <= budget.outputTarget
          && recordedTokens + anchorTokens > budget.outputTarget
        );
      }),
    ).toBeTrue();
    expect(
      harness.prompts.every((request, index) => {
        const actualTokens = generationRequestInputTokens(
          request,
          harness.promptHistories[index]!.messages,
        );
        const recordedTokens = conservativeGenerationRequestInputTokens(
          request,
          harness.promptHistories[index]!.messages,
        );
        return (
          actualTokens <= recordedTokens
          && recordedTokens <= budget.haloInputLimit
        );
      }),
    ).toBeTrue();
    expect(
      harness.prompts
        .flatMap(request =>
          generationPromptIndex(request).turns.map(turn => turn.ordinal),
        )
        .sort((left, right) => left - right),
    ).toEqual([1, 2]);
  });

  test("runs independent generation batches with exactly five prompts and forks in flight", async () => {
    const turns = Array.from({ length: 6 }, (_, index) =>
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

    expect(summaries).toHaveLength(6);
    expect(harness.prompts).toHaveLength(6);
    expect(harness.forks).toHaveLength(6);
    expect(harness.peakActivePrompts).toBe(5);
    expect(harness.peakActiveForks).toBe(5);
    expect(harness.deletes).toHaveLength(6);
  });

  test("excludes an oversized preceding summary turn without splitting its messages", async () => {
    const media = Array.from({ length: 63 }, (_, index) =>
      imagePart(
        `halo-opaque-image-${index}`,
        "halo-opaque-user",
        "data:image/png;base64,Zm9v",
      ),
    );
    const opaque = candidateTurn("halo-opaque", "opaque", media);
    opaque.assistants[0]!.parts.push({
      ...textPart("halo-opaque-summary", "halo-opaque-assistant", "summary"),
      metadata: { magicCompact: { summary: true, version: 2 } },
    });
    const target = candidateTurn("halo-target", "target", []);
    const harness = generationHarness([opaque, target], {
      turns: [result(1, "Target summary.")],
    });

    await generateCompactionSummaries(
      harness.v2,
      sourceSession(),
      generationPlan([opaque, target]),
    );

    expect(harness.promptHistories[0]!.messages).toHaveLength(2);
    expect(
      harness.promptHistories[0]!.messages.map(message => message.info.role),
    ).toEqual(["user", "assistant"]);
    expect(harness.messageDeletes).toHaveLength(2);
    expect(harness.partDeletes).toEqual([]);
  });

  test("keeps the preceding turn only within the halo limit and never sends the following turn", async () => {
    const preceding = candidateMessages("halo-preceding", "preceding request", [
      textPart(
        "halo-preceding-detail",
        "halo-preceding-assistant",
        "PRECEDING_DETAIL_4f2",
      ),
    ]);
    const target = candidateMessages("halo-target", "target request", []);
    target[0]!.parts.push({
      ...textPart("halo-target-boundary", "halo-target-user", "boundary"),
      synthetic: true,
      metadata: BOUNDARY_METADATA,
    });
    const following = candidateMessages("halo-following", "following request", [
      textPart(
        "halo-following-detail",
        "halo-following-assistant",
        "FOLLOWING_DETAIL_9c1",
      ),
    ]);
    const messages = [...preceding, ...target, ...following];
    const before = structuredClone(messages);
    const plan = planCompaction(messages, 1);
    expect(plan.summarizedTurns).toHaveLength(1);
    expect(plan.nextTurn?.user[0]!.info.id).toBe("halo-following-user");
    const userTexts = (history: MessageWithParts[]) =>
      history.flatMap(message =>
        message.info.role === "user"
          ? message.parts.flatMap(part =>
              part.type === "text" && part.synthetic !== true
                ? [part.text]
                : [],
            )
          : [],
      );
    const run = async (modelLimits?: {
      context: number;
      input: number;
      output: number;
    }) => {
      const harness = generationHarness(
        plan.summarizedTurns,
        { turns: [result(1, "Target summary.")] },
        { sourceMessages: messages, ...(modelLimits ? { modelLimits } : {}) },
      );
      await generateCompactionSummaries(harness.v2, sourceSession(), plan, {
        nonceFactory: () => "uB_4fLx2Xv5Tz7E1M-8vQn",
      });
      const request = harness.prompts[0]!;
      const history = harness.promptHistories[0]!.messages;
      // The following turn is never in the request, while the global map
      // keeps its user anchor for orientation.
      expect(JSON.stringify(history)).not.toContain("FOLLOWING_DETAIL_9c1");
      expect(JSON.stringify(history)).not.toContain("following request");
      expect(promptText(request)).not.toContain("FOLLOWING_DETAIL_9c1");
      expect(generationGlobalTurnIndex(request)).toEqual([
        { sourceOrdinal: 1, anchor: "preceding request" },
        { sourceOrdinal: 2, anchor: "target request" },
        { sourceOrdinal: 3, anchor: "following request" },
      ]);
      expect(Object.keys(generationPromptIndex(request))).toEqual(["turns"]);
      expect(harness.markerUpdates.map(update => update.part.text)).toEqual([
        '<magic-compact-turn nonce="uB_4fLx2Xv5Tz7E1M-8vQn" ordinal="1" />',
      ]);
      expect(messages).toEqual(before);
      return { request, history };
    };

    const fitting = await run();
    expect(userTexts(fitting.history)).toEqual([
      "preceding request",
      "target request",
    ]);
    expect(JSON.stringify(fitting.history)).toContain("PRECEDING_DETAIL_4f2");
    const completeHaloInput = generationRequestInputTokens(
      fitting.request,
      fitting.history,
    );
    const input = configuredInputForDerivedLimit(completeHaloInput - 1, 0.9);
    const tight = await run({ context: 1_000_000, input, output: 64_000 });
    expect(userTexts(tight.history)).toEqual(["target request"]);
    const actualTokens = generationRequestInputTokens(
      tight.request,
      tight.history,
    );
    const recordedTokens = conservativeGenerationRequestInputTokens(
      tight.request,
      tight.history,
    );
    expect(actualTokens).toBeLessThanOrEqual(recordedTokens);
    expect(recordedTokens).toBeLessThanOrEqual(
      computeNativeBatchBudget({
        context: 1_000_000,
        input,
        output: 64_000,
      }).hardSingleTurnInput,
    );
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
      { turns: [result(1, "Fresh summary.")] },
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
      { turns: [result(1, "Target summary.")] },
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

  test("orients by first ordinary user lines only in the global map, never in the target index", async () => {
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
    const harness = generationHarness(
      [selected],
      { turns: [result(1, "Selected summary")] },
      { nextTurn: next },
    );

    await generateCompactionSummaries(harness.v2, sourceSession(), {
      ...generationPlan([selected], next),
    });

    const request = harness.prompts[0]!;
    expect(generationGlobalTurnIndex(request)).toEqual([
      { sourceOrdinal: 1, anchor: "a".repeat(160) },
      { sourceOrdinal: 2, anchor: 'Stop "here"' },
    ]);
    const index = generationPromptIndex(request);
    expect(index.turns.map(turn => Object.keys(turn))).toEqual([
      ["ordinal", "marker", "expectedInlineImages"],
    ]);
    expect(Object.keys(index)).toEqual(["turns"]);
    const prompt = promptText(request);
    expect(prompt).not.toContain("a".repeat(161));
    expect(prompt).not.toContain("not this line");
    expect(prompt).not.toContain("next-assistant");
    expect(prompt).not.toContain("ignored synthetic");
    expect(prompt).not.toContain("ignored reporting");
  });

  test.each([
    ["empty", "", ""],
    ["duplicate", "same request", "same request"],
    ["161-300 code point", "x".repeat(161), `${"y".repeat(299)}😀`],
    [
      "over 300 code point",
      `${"a".repeat(299)}😀tail\nsecond line`,
      `${"a".repeat(299)}😀other`,
    ],
    ["quoted and escaped", 'quoted "request" \\ slash', "tab\tand \\n literal"],
    ["Unicode-normalized", "caf\u00e9 request", "cafe\u0301 request"],
    ["whitespace-sensitive", "  padded request  ", "trailing tab\t"],
  ])(
    "identifies targets by marker and ordinal alone for %s first lines",
    async (_name, firstLine, secondLine) => {
      const turns = [
        candidateTurn("identity-first", firstLine, []),
        candidateTurn("identity-second", secondLine, []),
      ];
      const before = structuredClone(turns);
      const harness = generationHarness(turns, {
        turns: [result(1, "First summary."), result(2, "Second summary.")],
      });

      const summaries = await generateCompactionSummaries(
        harness.v2,
        sourceSession(),
        generationPlan(turns),
      );

      expect(summaries.map(summary => summary?.text)).toEqual([
        "First summary.",
        "Second summary.",
      ]);
      expect(harness.aborts).toEqual([]);
      expect(harness.deletes.toSorted()).toEqual([
        "generation-fork",
        "generation-fork-2",
      ]);
      expect(turns).toEqual(before);
    },
  );

  test.each([
    [
      "omits its result",
      { turns: [] },
      "turn count mismatch (expected 1, received 0)",
    ],
    [
      "adds a second result",
      { turns: [result(1, "one"), result(1, "two")] },
      "turn count mismatch (expected 1, received 2)",
    ],
    [
      "adds an envelope key",
      { turns: [result(1, "one")], secretEnvelopeKey: "sensitive" },
      "invalid envelope",
    ],
    [
      "shifts its ordinal",
      { turns: [result(2, "one")] },
      "invalid turn shape or identity (ordinal at target ordinal 1)",
    ],
    [
      "returns an empty summary",
      { turns: [result(1, "  ")] },
      "invalid summary",
    ],
    [
      "adds a turn key",
      { turns: [{ ...result(1, "one"), secretTurnKey: "sensitive" }] },
      "invalid turn shape or identity (shape at target ordinal 1)",
    ],
  ])(
    "rejects a one-target response that %s",
    async (_name, response, category) => {
      const turn = candidateTurn("alpha", "alpha", []);
      const harness = generationHarness(
        [turn],
        response as GenerationResponse,
        { unfilteredResponse: true },
      );

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
      expect(failure).toBeInstanceOf(Error);
      expect((failure as Error).message).toBe(
        `Compaction generation response validation failed: ${category}.`,
      );
      expect(nestedErrorMessages(failure).join(" ")).not.toContain("sensitive");
      expect(nestedErrorMessages(failure).join(" ")).not.toContain(
        "secretTurnKey",
      );
      expect(nestedErrorMessages(failure).join(" ")).not.toContain(
        "secretEnvelopeKey",
      );

      expect(harness.forks).toHaveLength(1);
      expect(harness.prompts).toHaveLength(1);
      expect(harness.aborts).toEqual(["generation-fork"]);
      expect(harness.deletes).toEqual(["generation-fork"]);
    },
  );

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
        result(1, "Reviewed the assistant image.", [
          { ordinal: 1, description: "A completed interface state." },
        ]),
      ],
    });

    const summaries = await generateCompactionSummaries(
      harness.v2,
      sourceSession(),
      generationPlan([turn]),
    );
    const updates: TextPart[] = summaries.map(summary =>
      summaryEnvelopeTextPart(
        turn.assistants[0]!.info.id,
        "source",
        summary!.text,
        summary!.mediaManifest,
      ),
    );

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
        result(1, "Reviewed all interface captures.", [
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
    const updates: TextPart[] = summaries.map(summary =>
      summaryEnvelopeTextPart(
        turn.assistants[0]!.info.id,
        "source",
        summary!.text,
        summary!.mediaManifest,
      ),
    );

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
      turns: [result(1, "Reviewed images.", visualDescriptions)],
    });

    await expect(
      generateCompactionSummaries(
        harness.v2,
        sourceSession(),
        generationPlan([turn]),
      ),
    ).rejects.toThrow("response validation failed: invalid visual coverage.");
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

  test("maps duplicate and empty first lines through explicit rewritten-fork markers without touching source", async () => {
    const first = candidateTurn("duplicate-one", "same request", []);
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
    const opaque = candidateTurn("opaque", "same request", []);
    opaque.assistants[0]!.parts.push({
      ...textPart("opaque-summary", "opaque-assistant", "Existing summary"),
      metadata: { magicCompact: { summary: true, version: 2 } },
    });
    const empty = candidateTurn("empty", "", []);
    const next = candidateTurn("next-duplicate", "same request", []);
    const turns = [first, opaque, empty];
    const before = structuredClone(
      turns.flatMap(turn => [...turn.user, ...turn.assistants]),
    );
    const harness = generationHarness(
      turns,
      {
        turns: [
          result(1, "First duplicate summary."),
          result(2, "Empty-line summary."),
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
    expect(summaries[2]).toMatchObject({ text: "Empty-line summary." });
    const forkMarkers = ["generation-fork", "generation-fork-2"].map(
      sessionID =>
        harness.markerUpdates.filter(update => update.sessionID === sessionID),
    );
    // Message IDs are minted per fork; identify marked messages by content.
    expect(
      forkMarkers.map(updates => {
        const { messages } = harness.promptHistories.find(
          entry => entry.sessionID === updates[0]!.sessionID,
        )!;
        return updates.map(update => {
          const target = messages.find(
            item => item.info.id === update.messageID,
          )!;
          // The host orders parts by ID, and the marker's ID sorts before the
          // host-minted IDs of the copied request.
          expect(target.parts[0]!.id).toBe(update.part.id);
          const text = target.parts.find(
            part => part.type === "text" && part.id !== update.part.id,
          )!;
          return text.type === "text" ? text.text : undefined;
        });
      }),
    ).toEqual([["same request"], [""]]);
    const nonces = forkMarkers.map(updates => {
      const forkNonces = new Set(
        updates.map(
          update => update.part.text.match(/nonce="([A-Za-z0-9_-]{22})"/u)?.[1],
        ),
      );
      expect(forkNonces.size).toBe(1);
      return [...forkNonces][0]!;
    });
    expect(nonces[0]).toMatch(/^[A-Za-z0-9_-]{22}$/u);
    expect(nonces[1]).toMatch(/^[A-Za-z0-9_-]{22}$/u);
    expect(nonces[0]).not.toBe(nonces[1]);
    const markerTexts = forkMarkers.map(updates =>
      updates.map(update => update.part.text),
    );
    expect(markerTexts[0]).toEqual([expect.stringContaining('ordinal="1"')]);
    expect(markerTexts[1]).toEqual([expect.stringContaining('ordinal="2"')]);
    expect(markerTexts.flat()).not.toContain(
      '<magic-compact-turn ordinal="1" />',
    );
    expect(markerTexts.flat()).not.toContain("<magic-compact-stop />");
    for (const request of harness.prompts) {
      const sessionIndex = request.sessionID === "generation-fork" ? 0 : 1;
      const prompt = promptText(request);
      for (const marker of markerTexts[sessionIndex]!) {
        expect(prompt).toContain(JSON.stringify(marker));
      }
      for (const marker of markerTexts[1 - sessionIndex]!) {
        expect(prompt).not.toContain(JSON.stringify(marker));
      }
    }
    expect(
      summaries.flatMap(summary => (summary ? [summary.text] : [])),
    ).not.toContain(nonces[0]!);
    expect(turns.flatMap(turn => [...turn.user, ...turn.assistants])).toEqual(
      before,
    );
  });

  test("rejects a stale generation nonce even when the ordinal matches", async () => {
    const turn = candidateTurn("stale", "same request", []);
    const harness = generationHarness([turn], {
      turns: [
        {
          ...result(1, "Stale mapping."),
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
    ).rejects.toThrow(
      "response validation failed: invalid turn shape or identity (marker at target ordinal 1).",
    );
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
    expect(exhausted.forks).toEqual([]);
    expect(exhausted.markerUpdates).toEqual([]);
    expect(exhausted.prompts).toEqual([]);
    expect(exhausted.deletes).toEqual([]);
  });

  test("fails before the model request when rewritten fork shape or marker injection is invalid", async () => {
    const turn = candidateTurn("shape", "shape request", []);
    const shapeHarness = generationHarness([turn], responseFor([turn]), {
      // The replay acknowledgement is followed by a read missing the reply.
      mutateForkRead: (load, messages) => {
        if (load === 2) messages.pop();
      },
    });

    await expect(
      generateCompactionSummaries(
        shapeHarness.v2,
        sourceSession(),
        generationPlan([turn]),
      ),
    ).rejects.toThrow(
      "fork history verification failed (replayed fork did not match the plan)",
    );
    expect(shapeHarness.prompts).toEqual([]);
    expect(shapeHarness.aborts).toEqual([]);
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
    expect(markerHarness.aborts).toEqual([]);
    expect(markerHarness.deletes).toEqual(["generation-fork"]);
  });

  test("recovers one failed fork history read and reports a persistent timeout before the model request", async () => {
    const turn = candidateTurn("read", "read request", []);
    const recovered = generationHarness([turn], responseFor([turn]), {
      forkReadFault: load =>
        load === 1 ? Promise.reject(new Error("private read body")) : undefined,
    });
    await generateCompactionSummaries(
      recovered.v2,
      sourceSession(),
      generationPlan([turn]),
    );
    expect(recovered.forks).toHaveLength(1);
    expect(recovered.prompts).toHaveLength(1);
    expect(recovered.deletes).toEqual(["generation-fork"]);

    const stalled = generationHarness([turn], responseFor([turn]), {
      forkReadFault: () => new Promise(() => {}),
    });
    const failure = await generateCompactionSummaries(
      stalled.v2,
      sourceSession(),
      generationPlan([turn]),
      { controlTimeoutMs: 20 },
    ).then(
      () => undefined,
      (error: unknown) => error,
    );
    expect(String(failure)).toContain(
      "fork history verification timed out (history read timed out)",
    );
    expect(String(failure)).not.toContain("shape");
    expect(String(failure)).not.toContain("private");
    expect(stalled.forks).toHaveLength(1);
    expect(stalled.updates).toEqual([]);
    expect(stalled.prompts).toEqual([]);
    expect(stalled.deletes).toEqual(["generation-fork"]);
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
      const turns = Array.from({ length: 33 }, (_, index) =>
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
        mutateForkRead: (load, messages, sessionID) => {
          if (load === 4 && sessionID === "generation-fork-2") mutate(messages);
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
      expect(harness.aborts).toEqual([]);
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

  test("does not require vision when the only image is in the following turn", async () => {
    const selected = candidateTurn("following-selected", "selected", []);
    const nextTurn = candidateTurn("following-image", "following", [
      imagePart(
        "following-image-file",
        "following-image-user",
        "data:image/png;base64,Zm9v",
      ),
    ]);
    const harness = generationHarness([selected], responseFor([selected]), {
      nextTurn,
      modelImageCapability: false,
    });

    await generateCompactionSummaries(
      harness.v2,
      sourceSession(),
      generationPlan([selected], nextTurn),
    );

    expect(harness.prompts).toHaveLength(1);
    expect(JSON.stringify(harness.promptHistories[0]!.messages)).not.toContain(
      "image/png",
    );
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
    ["missing with only the native compaction agent", "native-only"],
    ["without an explicit model", "model-missing"],
  ] as const)(
    "rejects a summarizer agent %s before fork creation without a fallback",
    async (_name, summarizerAgent) => {
      const turn = candidateTurn("configuration", "configure", []);
      const before = structuredClone(turn);
      const harness = generationHarness([turn], responseFor([turn]), {
        summarizerAgent,
      });

      await expect(
        generateCompactionSummaries(
          harness.v2,
          sourceSession(),
          generationPlan([turn]),
        ),
      ).rejects.toThrow(
        summarizerAgent === "model-missing"
          ? "requires an explicit model"
          : "magic-compact-summarizer agent, but it is missing",
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
      turns: [result(1, "Fresh turn summary.")],
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

  test("allows fork creation to exceed the control deadline and deletes the resulting fork", async () => {
    const turn = candidateTurn("late-fork", "late fork request", []);
    let releaseFork!: () => void;
    const forkGate = new Promise<void>(resolve => {
      releaseFork = resolve;
    });
    const harness = generationHarness([turn], responseFor([turn]), {
      forkGate,
    });

    let settled = false;
    const operation = generateCompactionSummaries(
      harness.v2,
      sourceSession(),
      generationPlan([turn]),
      { controlTimeoutMs: 1 },
    ).finally(() => {
      settled = true;
    });
    await new Promise(resolve => setTimeout(resolve, 10));

    expect(settled).toBeFalse();
    expect(harness.prompts).toEqual([]);
    releaseFork();
    await expect(operation).resolves.toEqual([
      { text: "Turn 1 outcome and continuation details." },
    ]);

    expect(harness.prompts).toHaveLength(1);
    expect(harness.aborts).toEqual([]);
    expect(harness.deletes).toEqual(["generation-fork"]);
  });

  test("stops sixth-batch admission before failed cleanup drains", async () => {
    const turns = Array.from({ length: 6 }, (_, index) =>
      candidateTurn(`cleanup-drain-${index + 1}`, `anchor ${index + 1}`, []),
    );
    let releaseDeletes!: () => void;
    const deleteGate = new Promise<void>(resolve => {
      releaseDeletes = resolve;
    });
    const harness = generationHarness(turns, responseFor(turns), {
      providerErrorForFirstOrdinal: 1,
      holdPromptsUntil: 5,
      deleteGate,
      deleteGateForFirstOrdinal: 1,
    });
    let settled = false;
    const operation = generateCompactionSummaries(
      harness.v2,
      sourceSession(),
      generationPlan(turns),
    ).finally(() => {
      settled = true;
    });

    await waitFor(() => harness.events.includes("delete-waiting"));
    await waitFor(() => harness.deletes.length === 4);
    expect(settled).toBeFalse();
    expect(harness.prompts).toHaveLength(5);
    expect(harness.forks).toHaveLength(5);
    releaseDeletes();

    await expect(operation).rejects.toThrow("model request failed");
    expect(harness.prompts).toHaveLength(5);
    expect(harness.aborts).toEqual(["generation-fork"]);
    expect(harness.deletes).toHaveLength(5);
  });

  test("accepts captured structured output when finish reports length", async () => {
    const turn = candidateTurn("captured-length", "captured response", []);
    const harness = generationHarness([turn], responseFor([turn]), {
      responseFinish: "length",
    });

    await expect(
      generateCompactionSummaries(
        harness.v2,
        sourceSession(),
        generationPlan([turn]),
      ),
    ).resolves.toEqual([{ text: "Turn 1 outcome and continuation details." }]);
  });

  test.each([
    ["finish length", { responseFinish: "length", omitStructured: true }],
    [
      "finish length with structured output error",
      {
        responseFinish: "length",
        omitStructured: true,
        responseError: {
          name: "StructuredOutputError",
          data: {
            message: "Model did not produce structured output",
            retries: 0,
          },
        },
      },
    ],
    [
      "finish output-length",
      { responseFinish: "output_length", omitStructured: true },
    ],
    ["finish max-token", { responseFinish: "max_token", omitStructured: true }],
    [
      "output length error",
      {
        omitStructured: true,
        responseError: {
          name: "MessageOutputLengthError",
          data: { raw: "must stay hidden" },
        },
      },
    ],
    [
      "nested max-output-token error",
      {
        omitStructured: true,
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
    [
      "nested structured output error with finish length",
      {
        responseFinish: "length",
        omitStructured: true,
        responseError: {
          error: {
            cause: {
              name: "StructuredOutputError",
              data: {
                message: "Model did not produce structured output",
                retries: 0,
              },
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

      const failure = await generateCompactionSummaries(
        harness.v2,
        sourceSession(),
        generationPlan([turn]),
      ).then(
        () => undefined,
        error => error as Error,
      );
      expect(failure?.message).toMatch(/single turn.*more output capacity/i);
      expect(failure?.message).not.toContain(
        "Model did not produce structured output",
      );
      expect(harness.aborts).toEqual(["generation-fork"]);
      expect(harness.deletes).toEqual(["generation-fork"]);
    },
  );

  test("keeps other provider errors ahead of finish truncation with sanitized output", async () => {
    const turn = candidateTurn("provider-error", "provider failure", []);
    const harness = generationHarness([turn], responseFor([turn]), {
      responseFinish: "length",
      responseError: {
        name: "ProviderError",
        data: { message: "raw provider detail" },
      },
    });

    const failure = await generateCompactionSummaries(
      harness.v2,
      sourceSession(),
      generationPlan([turn]),
    ).then(
      () => undefined,
      error => error as Error,
    );

    expect(failure?.message).toMatch(/model request failed/i);
    expect(failure?.message).not.toContain("raw provider detail");
    expect(failure?.message).not.toMatch(/more output capacity/i);
  });

  test("self-describes a missing structured response without host details", async () => {
    const turn = candidateTurn("missing-structured", "structured failure", []);
    const harness = generationHarness([turn], responseFor([turn]), {
      responseFinish: "stop",
      omitStructured: true,
      responseError: {
        name: "StructuredOutputError",
        data: {
          message: "Model did not produce structured output",
          retries: 0,
        },
      },
    });

    const failure = await generateCompactionSummaries(
      harness.v2,
      sourceSession(),
      generationPlan([turn]),
    ).then(
      () => undefined,
      error => error as Error,
    );

    expect(failure?.message).toBe(
      "The compaction model finished without submitting its StructuredOutput response.",
    );
    expect(failure?.message).not.toContain(
      "Model did not produce structured output",
    );
    expect(harness.aborts).toEqual(["generation-fork"]);
    expect(harness.deletes).toEqual(["generation-fork"]);
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
      if (name === "permission update") {
        expect(harness.aborts).toEqual([]);
      }
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

  test.each([
    [0, 0, 3],
    [1, 1, 2],
    [3, 3, 0],
    [9, 3, 0],
  ])(
    "N=%i preserves %i whole turn(s) and summarizes %i, ignoring a trailing assistantless request",
    async (keepTurns, preserved, summarized) => {
      const messages = [
        ...candidateMessages("preserved-a", "first", []),
        ...candidateMessages("preserved-b", "second", []),
        ...candidateMessages("preserved-c", "third", []),
        message("preserved-trailing-user", "user", [
          textPart("preserved-trailing-text", "preserved-trailing-user", "?"),
        ]),
      ];
      const v2 = {
        session: { messages: async () => ({ data: messages }) },
      } as unknown as V2Client;

      const plan = await createCompactionPlan(v2, "source", keepTurns);

      expect(plan.preservedTurnCount).toBe(preserved);
      expect(plan.summarizedTurns).toHaveLength(summarized);
    },
  );

  test("preserves only turns after the latest boundary when N exceeds them", async () => {
    const boundaryTurn = candidateMessages("preserved-boundary", "second", []);
    boundaryTurn[0]!.parts.push({
      id: "preserved-boundary-part",
      sessionID: "source",
      messageID: boundaryTurn[0]!.info.id,
      type: "text",
      text: POST_COMPACTION_NOTICE,
      synthetic: true,
      metadata: BOUNDARY_METADATA,
    } as Part);
    const messages = [
      ...candidateMessages("preserved-old", "first", []),
      ...boundaryTurn,
      ...candidateMessages("preserved-new", "third", []),
    ];
    const v2 = {
      session: { messages: async () => ({ data: messages }) },
    } as unknown as V2Client;

    const plan = await createCompactionPlan(v2, "source", 5);

    expect(plan.preservedTurnCount).toBe(2);
    expect(plan.summarizedTurns).toHaveLength(0);
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
    const original = structuredClone(messages);
    const harness = commandHarness(messages, {
      generationResponse: {
        turns: [
          result(
            1,
            "A generated summary that is longer than the exact tiny response.",
          ),
          result(2, "Completed the large request."),
        ],
      },
    });

    await expect(
      executeMagicCompact(harness.v2, "source", 0, HEURISTIC_OPTIONS),
    ).resolves.toMatchObject({ session: { id: "successor" } });

    expect(messages).toEqual(original);
    const successor = harness.successorMessages();
    expect(withoutIdentity(successor.slice(0, 2))).toEqual(
      withoutIdentity(original.slice(0, 2)),
    );
    expect(summaryParts(successor)).toHaveLength(1);
    expect(summaryParts(successor)[0]!.messageID).toBe(successor[3]!.info.id);
  });

  test("pure no-eligible compaction creates no fork, successor, or write", async () => {
    const messages = candidateMessages("no-eligible", "keep current", []);
    const before = structuredClone(messages);
    const harness = commandHarness(messages, {
      generationResponse: { turns: [] },
    });

    await expect(
      executeMagicCompact(harness.v2, "source", 1, HEURISTIC_OPTIONS),
    ).resolves.toBeNull();

    expect(harness.forks).toEqual([]);
    expect(harness.successors).toEqual([]);
    expect(harness.sourceWrites).toEqual([]);
    expect(harness.deletedSessions).toEqual([]);
    expect(messages).toEqual(before);
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
            ...HEURISTIC_OPTIONS,
            controlTimeoutMs: 5,
          }),
          250,
        ),
      ).resolves.toBeNull();
      expect(noOp.toasts.map(toast => toast.message)).toEqual([
        "Compacting session...",
        NOOP_PRESERVING_ONE_TURN,
      ]);
      expect(noOp.toasts[0]?.duration).toBe(2_147_483_647);
      expect(JSON.stringify(noOp.messages)).not.toContain(
        "Compacting session...",
      );
      expect(await readCache("source")).toBeNull();
      expect(await readStats("source")).toBeNull();

      const successMessages = compactableCommandMessages(
        `toast-${toastMode}-success`,
        "large output ".repeat(1_000),
      );
      const success = commandHarness(successMessages, {
        generationResponse: {
          turns: [result(1, "Completed toast candidate.")],
        },
        toastMode,
      });
      await expect(
        settleWithin(
          executeMagicCompact(success.v2, "source", 0, {
            ...HEURISTIC_OPTIONS,
            controlTimeoutMs: 5,
          }),
          250,
        ),
      ).resolves.toMatchObject({ session: { id: "successor" } });
      expect(summaryParts(success.successorMessages())).toHaveLength(1);
      expect(success.toasts.map(toast => toast.message)).toEqual([
        "Compacting session...",
        `Compacted 1 request turn(s) into successor session successor. ${preservedTurnsFeedback({ requested: 0, preserved: 0 })} Heuristic mode: tool calls not evaluated.`,
      ]);
      expect(JSON.stringify(successMessages)).not.toContain(
        "Compacting session...",
      );
      expect(JSON.stringify(success.modelPrompts)).not.toContain(
        "Compacting session...",
      );

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
            ...HEURISTIC_OPTIONS,
            controlTimeoutMs: 5,
          }),
          250,
        );
      } catch (error) {
        failure = error;
      }
      expect(failure).toBeInstanceOf(Error);
      expect((failure as Error).message).toBe(
        "Compaction generation response validation failed: turn count mismatch (expected 1, received 0).",
      );
      expect(nestedErrorMessages(failure).join(" ")).not.toContain(
        "toast failed",
      );
      expect(failed.toasts.map(toast => toast.message)).toEqual([
        "Compacting session...",
        "GenerationLifecycleError: Compaction generation response validation failed: turn count mismatch (expected 1, received 0).",
      ]);
      expect(JSON.stringify(failed.messages)).not.toContain(
        "Compacting session...",
      );
      expect(JSON.stringify(failed.modelPrompts)).not.toContain(
        "Compacting session...",
      );
      expect(JSON.stringify(await readCache("source"))).not.toContain(
        "Compacting session...",
      );
      expect(JSON.stringify(await readStats("source"))).not.toContain(
        "Compacting session...",
      );
    },
  );

  test("reasserts terminal feedback after a timed-out progress toast arrives late", async () => {
    const harness = commandHarness(
      candidateMessages("toast-late-progress", "keep", []),
      {
        generationResponse: { turns: [] },
        toastMode: "delay-progress",
      },
    );

    await expect(
      settleWithin(
        executeMagicCompact(harness.v2, "source", 1, {
          ...HEURISTIC_OPTIONS,
          controlTimeoutMs: 5,
        }),
        250,
      ),
    ).resolves.toBeNull();
    expect(harness.displayedToast?.message).toBe(NOOP_PRESERVING_ONE_TURN);

    harness.releaseProgressToast();
    await waitFor(() => harness.toasts.length === 3);

    expect(harness.toasts.map(toast => toast.message)).toEqual([
      "Compacting session...",
      NOOP_PRESERVING_ONE_TURN,
      NOOP_PRESERVING_ONE_TURN,
    ]);
    expect(harness.displayedToast?.message).toBe(NOOP_PRESERVING_ONE_TURN);
    expect(JSON.stringify(harness.messages)).not.toContain(
      "Compacting session...",
    );
    expect(await readCache("source")).toBeNull();
    expect(await readStats("source")).toBeNull();
  });

  test("places the boundary first in the next user message and leaves its parts exact", async () => {
    const messages = [
      ...compactableCommandMessages(
        "boundary-old",
        "old output ".repeat(1_000),
      ),
      ...candidateMessages("boundary-next", "next request", []),
    ];
    const nextUserParts = structuredClone(messages[2]!.parts);
    const harness = commandHarness(messages, {
      generationResponse: { turns: [result(1, "Completed old work.")] },
    });

    await executeMagicCompact(harness.v2, "source", 1, HEURISTIC_OPTIONS);

    const nextUser = harness.successorMessages()[2]!;
    expect(isBoundary(nextUser.parts[0]!)).toBeTrue();
    expect(nextUser.parts.slice(1).map(withoutPartIdentity)).toEqual(
      nextUserParts.map(withoutPartIdentity),
    );
  });

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
    const original = structuredClone(messages);
    const harness = commandHarness(messages, {
      generationResponse: {
        turns: [
          result(
            1,
            "Completed the requested work.\n\nProcess and decisions were preserved; verification passed.",
            [{ ordinal: 1, description: "The completed result is visible." }],
          ),
        ],
      },
    });

    await expect(
      executeMagicCompact(harness.v2, "source", 0, HEURISTIC_OPTIONS),
    ).resolves.toMatchObject({ session: { id: "successor" } });

    expect(harness.forks).toEqual(["generation-fork"]);
    expect(harness.events.indexOf("generation-delete")).toBeLessThan(
      harness.events.indexOf("successor-create"),
    );
    expect(harness.sourceWrites).toEqual([]);
    expect(messages).toEqual(original);
    const successor = harness.successorMessages();
    expect(harness.successorInfo()).toMatchObject({
      title: "Source",
      metadata: {
        magicCompact: {
          successorOf: "source",
          operation: "compact",
          compactionCount: 1,
          ready: true,
        },
      },
    });
    const assistant = successor[1]!;
    const summary = summaryParts(successor)[0]!;
    expect(assistant.parts[0]).toEqual(summary);
    expect(summary).toMatchObject({
      messageID: assistant.info.id,
      metadata: { magicCompact: { summary: true, version: 3 } },
    });
    expect(
      successor
        .flatMap(item => item.parts)
        .some(part => part.type === "reasoning"),
    ).toBeFalse();
    const tool = successor
      .flatMap(item => item.parts)
      .find(
        (part): part is Extract<Part, { type: "tool" }> => part.type === "tool",
      );
    expect(tool).toBeDefined();
    const persistedImage = assistant.parts.find(
      part => part.type === "file" && part.url === assistantImage.url,
    );
    expect(withoutPartIdentity(persistedImage!)).toEqual(
      withoutPartIdentity(assistantImageBefore),
    );
    expect(
      projectManifestCoveredMedia(successor)[1]!.parts.find(
        part => part.id === persistedImage!.id,
      ),
    ).toMatchObject({ type: "text", synthetic: true });
    const serialized = JSON.stringify(tool!.state);
    const contentID = serialized.match(
      /[A-Za-z0-9_-]{1,12}:omitted-[A-Za-z0-9_-]{22}/u,
    )?.[0];
    expect(contentID).toBeDefined();
    expect(await readOmittedContent("successor", contentID!)).toBe(output);
    expect(await readOmittedContent("source", contentID!)).toBeNull();
    expect(JSON.stringify(await readCache("successor"))).not.toContain(
      assistantImage.url,
    );
    expect(boundaryParts(successor)).toHaveLength(1);
    expect(await readStats("source")).toBeNull();
    const stats = await readStats("successor");
    expect(stats?.totalTokensPruned).toBeGreaterThan(0);
    expect(stats?.sourceSessionId).toBe("source");
    expect(stats?.processedAssistantMessageIds).toEqual([assistant.info.id]);
    const statsNotice = successor
      .flatMap(item => item.parts)
      .find(part => {
        if (part.type !== "text") return false;
        const metadata = part.metadata?.magicCompact as
          Record<string, unknown> | undefined;
        return metadata?.["stats"] === true;
      }) as TextPart | undefined;
    expect(statsNotice).toBeDefined();
    expect(statsNotice!.text).toContain("Magic Compaction #1");
    expect(statsNotice!.text).toContain("Stored history (estimate) | ");
    expect(statsNotice!.text).toContain("% trimmed (");
    expect(statsNotice!.text).toContain(
      "Changes | Request turns summarized: 1",
    );
    expect(statsNotice!.text).not.toContain("Cache Savings");
    expect(statsNotice!.text).not.toContain("Cost Savings");
    expect(statsNotice!.text).not.toContain("Inline image tokens projected");
    expect(statsNotice!.text).not.toContain("provider usage");
    expect(harness.successors).toEqual(["successor"]);
    expect(harness.deletedSessions).toEqual(["generation-fork"]);
  });

  test("reports the stored history and the whole turns N preserved over a native checkpoint", async () => {
    // Stored order: hidden prefix, retained tail, checkpoint, then two suffix
    // turns. The reported counts cover all of it, including the hidden prefix.
    const messages = [
      ...compactableCommandMessages("hidden", "hidden history ".repeat(2_000)),
      ...candidateMessages("tail", "retained tail request", []),
      nativeMarker("native-marker", "tail-user"),
      nativeSummary("native-summary", "native-marker"),
      ...compactableCommandMessages("older", "older output ".repeat(1_000)),
      ...compactableCommandMessages("latest", "latest output ".repeat(1_000)),
    ];
    const original = structuredClone(messages);
    const harness = commandHarness(messages, {
      generationResponse: { turns: [result(1, "Completed the older work.")] },
    });

    await expect(
      executeMagicCompact(harness.v2, "source", 1, HEURISTIC_OPTIONS),
    ).resolves.toMatchObject({ session: { id: "successor" } });

    expect(messages).toEqual(original);
    const successor = harness.successorMessages();
    const storedBefore = countCanonicalContentTokens(original);
    const storedAfter = countCanonicalContentTokens(successor);
    const preserved = { requested: 1, preserved: 1 };
    const notice = successor
      .flatMap(item => item.parts)
      .find(
        (part): part is TextPart =>
          part.type === "text"
          && (part.metadata?.magicCompact as { stats?: true } | undefined)
            ?.stats === true,
      );
    expect(notice?.text).toBe(
      statsMessage(1, storedBefore, storedAfter, 1, preserved),
    );
    expect(harness.toasts.at(-1)?.message).toBe(
      `Compacted 1 request turn(s) into successor session successor. ${preservedTurnsFeedback(preserved)} Heuristic mode: tool calls not evaluated.`,
    );

    expect((await readStats("successor"))?.totalTokensPruned).toBe(
      storedBefore - storedAfter,
    );

    // The prefix, checkpoint, tail and preserved turn stay as they were; the
    // checkpoint tail link follows the rewritten tail message.
    expect(withoutIdentity(successor.slice(0, 4))).toEqual(
      withoutIdentity(original.slice(0, 4)),
    );
    expect(successor[4]!.parts[0]).toMatchObject({
      type: "compaction",
      tail_start_id: successor[2]!.info.id,
    });
    expect(withoutIdentity(successor.slice(5, 6))).toEqual(
      withoutIdentity(original.slice(5, 6)),
    );
    const latestUser = successor[8]!;
    expect(isBoundary(latestUser.parts[0]!)).toBeTrue();
    expect(latestUser.parts.slice(1).map(withoutPartIdentity)).toEqual(
      original[8]!.parts.map(withoutPartIdentity),
    );
    expect(withoutIdentity(successor.slice(9, 10))).toEqual(
      withoutIdentity(original.slice(9, 10)),
    );
    expect(summaryParts(successor).map(part => part.messageID)).toEqual([
      successor[7]!.info.id,
    ]);
  });

  test("keeps the native checkpoint and reports no preserved turns when N=0 appends the boundary message", async () => {
    const messages = [
      ...compactableCommandMessages("hidden", "hidden history ".repeat(2_000)),
      ...candidateMessages("tail", "retained tail request", []),
      nativeMarker("native-marker", "tail-user"),
      nativeSummary("native-summary", "native-marker"),
      ...compactableCommandMessages("older", "older output ".repeat(1_000)),
      ...compactableCommandMessages("latest", "latest output ".repeat(1_000)),
    ];
    const original = structuredClone(messages);
    const harness = commandHarness(messages, {
      generationResponse: {
        turns: [result(1, "Completed older."), result(2, "Completed latest.")],
      },
    });

    await expect(
      executeMagicCompact(harness.v2, "source", 0, HEURISTIC_OPTIONS),
    ).resolves.toMatchObject({ session: { id: "successor" } });

    expect(messages).toEqual(original);
    const successor = harness.successorMessages();
    const notice = successor
      .flatMap(item => item.parts)
      .find(
        (part): part is TextPart =>
          part.type === "text"
          && (part.metadata?.magicCompact as { stats?: true } | undefined)
            ?.stats === true,
      );
    expect(notice?.text).toBe(
      statsMessage(
        1,
        countCanonicalContentTokens(original),
        countCanonicalContentTokens(successor),
        2,
        { requested: 0, preserved: 0 },
      ),
    );
    expect(withoutIdentity(successor.slice(0, 4))).toEqual(
      withoutIdentity(original.slice(0, 4)),
    );
    expect(withoutIdentity(successor.slice(5, 6))).toEqual(
      withoutIdentity(original.slice(5, 6)),
    );
    // With no next turn, the boundary and then the stats notice are appended.
    expect(isBoundary(successor.at(-2)!.parts[0]!)).toBeTrue();
    expect(successor.at(-1)!.parts[0]).toBe(notice!);
  });

  test("states the preserved whole turns when N leaves nothing to compact", async () => {
    const messages = [
      ...compactableCommandMessages("keep-a", "first output ".repeat(1_000)),
      ...compactableCommandMessages("keep-b", "second output ".repeat(1_000)),
    ];
    const before = structuredClone(messages);
    const harness = commandHarness(messages, {
      generationResponse: { turns: [] },
    });

    await expect(
      executeMagicCompact(harness.v2, "source", 5, HEURISTIC_OPTIONS),
    ).resolves.toBeNull();

    expect(harness.toasts.at(-1)?.message).toBe(
      `${COMPACT_NOOP} ${preservedTurnsFeedback({ requested: 5, preserved: 2 })}`,
    );
    expect(harness.forks).toEqual([]);
    expect(harness.sourceWrites).toEqual([]);
    expect(messages).toEqual(before);
  });

  test("omits a large task prompt on a previously trimmed part during compaction", async () => {
    const prompt = "delegation context packet ".repeat(60);
    const messages = compactableCommandMessages(
      "task-prompt",
      "subagent result",
    );
    const task = toolPart(
      "task-prompt-task-tool",
      "task-prompt-assistant",
      {
        status: "completed",
        input: {
          subagent_type: "forager-worker",
          description: "delegated work",
          prompt,
        },
        output: "subagent result",
        title: "Task",
        metadata: { magicCompact: { trimmed: true } },
        time: { start: 1, end: 2 },
      },
      "task",
    );
    messages[1]!.parts.push(task);
    const original = structuredClone(messages);
    const beforeTokens = countCanonicalContentTokens(messages);
    const harness = commandHarness(messages, {
      generationResponse: {
        turns: [result(1, "Completed the delegated work.")],
      },
    });

    await expect(
      executeMagicCompact(harness.v2, "source", 0, HEURISTIC_OPTIONS),
    ).resolves.toMatchObject({ session: { id: "successor" } });

    expect(messages).toEqual(original);
    const successor = harness.successorMessages();
    const persisted = successor
      .flatMap(item => item.parts)
      .find(
        (part): part is Extract<Part, { type: "tool" }> =>
          part.type === "tool" && part.tool === "task",
      );
    expect(persisted).toBeDefined();
    expect(persisted!.state.status).toBe("completed");
    if (persisted!.state.status !== "completed")
      throw new Error("Expected completed tool state.");
    expect(persisted!.state.input["description"]).toBe("delegated work");
    expect(persisted!.state.input["subagent_type"]).toBe("forager-worker");
    expect(persisted!.state.metadata["magicCompact"]).toEqual({
      trimmed: true,
    });
    expect(persisted!.state.output).toContain(
      "Task prompt omitted due to compaction operation.",
    );
    const contentID = persisted!.state.output.match(
      /[A-Za-z0-9_-]{1,12}:omitted-[A-Za-z0-9_-]{22}/u,
    )?.[0];
    expect(contentID).toBeDefined();
    expect(persisted!.state.input["prompt"]).toBe(
      `[Historical task prompt omitted; Content ID: ${contentID}. Retrieve the original with read_omitted_content before reuse.]`,
    );
    expect(await readOmittedContent("successor", contentID!)).toBe(prompt);
    const cache = await readCache("successor");
    expect(
      Object.values(cache!.entries).filter(entry => entry.content === prompt),
    ).toHaveLength(1);
    expect((await readStats("successor"))?.totalTokensPruned).toBe(
      beforeTokens - countCanonicalContentTokens(successor),
    );
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
          result(1, "Completed the request.", [
            { ordinal: 1, description: "A user-provided screenshot." },
            { ordinal: 2, description: "An assistant-provided result image." },
          ]),
        ],
      },
    });

    await expect(
      executeMagicCompact(harness.v2, "source", 0, HEURISTIC_OPTIONS),
    ).resolves.toMatchObject({ session: { id: "successor" } });

    const successor = harness.successorMessages();
    const imageWithURL = (index: number, url: string) =>
      successor[index]!.parts.find(
        part => part.type === "file" && part.url === url,
      )!;
    const persistedUserImage = imageWithURL(0, userImage.url);
    const persistedAssistantImage = imageWithURL(1, assistantImage.url);
    expect(withoutPartIdentity(persistedUserImage)).toEqual(
      withoutPartIdentity(userImageBefore),
    );
    expect(withoutPartIdentity(persistedAssistantImage)).toEqual(
      withoutPartIdentity(assistantImageBefore),
    );
    const summary = summaryParts(successor)[0]!;
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
    const projected = projectManifestCoveredMedia(successor);
    expect(
      projected[0]!.parts.find(part => part.id === persistedUserImage.id),
    ).toMatchObject({ type: "file", url: userImage.url });
    expect(
      projected[1]!.parts.find(part => part.id === persistedAssistantImage.id),
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
            "A longer generated summary that cannot save tokens for this tiny turn.",
          ),
        ],
      },
    });

    await expect(
      executeMagicCompact(harness.v2, "source", 0, HEURISTIC_OPTIONS),
    ).rejects.toThrow(COMPACT_INSUFFICIENT_SAVINGS);

    expect(harness.forks).toEqual(["generation-fork"]);
    expect(harness.sourceWrites).toEqual([]);
    expect(boundaryParts(messages)).toEqual([]);
    expect(await readStats("source")).toBeNull();
  });

  test("both hosts' savings gate accepts exactly 20% of eligible tokens, at least 32", () => {
    expect(hasUsefulSavings(32, 0)).toBeTrue();
    expect(hasUsefulSavings(31, 0)).toBeFalse();
    expect(hasUsefulSavings(40, 200)).toBeTrue();
    expect(hasUsefulSavings(39, 200)).toBeFalse();
    expect(hasUsefulSavings(41, 201)).toBeTrue();
    expect(hasUsefulSavings(40, 201)).toBeFalse();
  });

  test("discards eight accepted one-target results when the ninth is malformed", async () => {
    const messages = Array.from({ length: 9 }, (_, index) =>
      compactableCommandMessages(
        `nine-targets-${index + 1}`,
        `large output ${index + 1} `.repeat(1_000),
      ),
    ).flat();
    const before = structuredClone(messages);
    const harness = commandHarness(messages, {
      generationResponse: {
        turns: Array.from({ length: 8 }, (_, index) =>
          result(index + 1, `Summary ${index + 1}.`),
        ),
      },
    });

    await expect(
      executeMagicCompact(harness.v2, "source", 0, HEURISTIC_OPTIONS),
    ).rejects.toThrow(
      "response validation failed: turn count mismatch (expected 1, received 0).",
    );

    const forks = [
      "generation-fork",
      ...Array.from(
        { length: 8 },
        (_, index) => `generation-fork-${index + 2}`,
      ),
    ];
    expect(harness.forks).toEqual(forks);
    expect(harness.deletedSessions.toSorted()).toEqual(forks.toSorted());
    expect(harness.activeGenerationForks).toEqual([]);
    expect(messages).toEqual(before);
    expect(harness.sourceWrites).toEqual([]);
    expect(boundaryParts(messages)).toEqual([]);
    expect(harness.successors).toEqual([]);
    expect(await readStats("source")).toBeNull();
  });

  test("reports a post-request count mismatch as a categorized lifecycle error", async () => {
    const messages = compactableCommandMessages(
      "post-request-count",
      "large output ".repeat(1_000),
    );
    const before = structuredClone(messages);
    const harness = commandHarness(messages, {
      generationResponse: {
        turns: [result(1, "Completed request.")],
      },
      transformGenerationResponse: response => {
        const turns = [...response.turns];
        Object.defineProperty(turns, "map", { value: () => [] });
        return { turns };
      },
    });
    let failure: unknown;

    try {
      await executeMagicCompact(harness.v2, "source", 0, HEURISTIC_OPTIONS);
    } catch (error) {
      failure = error;
    }

    expect(failure).toBeInstanceOf(Error);
    expect((failure as Error).name).toBe("GenerationLifecycleError");
    expect((failure as Error).message).toBe(
      "Compaction generation response validation failed: turn count mismatch (expected 1, received 0).",
    );

    expect(messages).toEqual(before);
    expect(harness.forks).toEqual(["generation-fork"]);
    expect(harness.sourceWrites).toEqual([]);
    expect(boundaryParts(messages)).toEqual([]);
    expect(harness.successors).toEqual([]);
    expect(harness.deletedSessions).toEqual(["generation-fork"]);
    expect(await readStats("source")).toBeNull();
  });

  test("aggregates every active batch failure in batch order before any durable mutation", async () => {
    const messages = Array.from({ length: 2 }, (_, index) =>
      compactableCommandMessages(
        `aggregate-batch-${index + 1}`,
        `large output ${index + 1} `.repeat(1_000),
      ),
    ).flat();
    const before = structuredClone(messages);
    const harness = commandHarness(messages, {
      generationResponse: {
        turns: Array.from({ length: 2 }, (_, index) =>
          result(index + 1, `Summary ${index + 1}.`),
        ),
      },
      generationErrorsByFirstOrdinal: {
        1: new Error("raw first provider payload"),
        2: new Error("raw second provider payload"),
      },
      generationAbortErrorForFirstOrdinal: 2,
      generationDeleteErrorForFirstOrdinal: 2,
      holdGenerationPromptsUntil: 2,
    });
    let failure: unknown;

    try {
      await executeMagicCompact(harness.v2, "source", 0, HEURISTIC_OPTIONS);
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
    expect(harness.sourceWrites).toEqual([]);
    expect(boundaryParts(messages)).toEqual([]);
    expect(await readStats("source")).toBeNull();
    expect(await readStats("backup")).toBeNull();
  });

  test("rejects an active marker leaked into a summary before backup or source writes", async () => {
    const messages = compactableCommandMessages("summary-leak", "output");
    const before = structuredClone(messages);
    const harness = commandHarness(messages, {
      generationResponse: {
        turns: [result(1, "Clean summary.")],
      },
      transformGenerationResponse: response => ({
        turns: response.turns.map(turn => ({
          ...turn,
          summary: `Leaked active marker: ${turn.marker}`,
        })),
      }),
    });

    let failure: unknown;
    try {
      await executeMagicCompact(harness.v2, "source", 0, HEURISTIC_OPTIONS);
    } catch (error) {
      failure = error;
    }
    expect(failure).toBeInstanceOf(Error);
    expect((failure as Error).message).toBe(
      "Compaction generation response validation failed: nonce leakage.",
    );
    const diagnostics = nestedErrorMessages(failure).join(" ");
    expect(diagnostics).not.toContain(
      generationPromptIndex(harness.modelPrompts[0]!).turns[0]!.marker,
    );
    expect(diagnostics).not.toContain("summary-leak request");

    expect(messages).toEqual(before);
    expect(harness.forks).toEqual(["generation-fork"]);
    expect(harness.abortedSessions).toEqual(["generation-fork"]);
    expect(harness.successors).toEqual([]);
    expect(harness.deletedSessions).toEqual(["generation-fork"]);
    expect(harness.sourceWrites).toEqual([]);
    expect(boundaryParts(messages)).toEqual([]);
    expect(await readStats("source")).toBeNull();
    expect(await readStats("backup")).toBeNull();
  });

  test("rejects a response item that still echoes an anchor before backup or source writes", async () => {
    const messages = compactableCommandMessages("anchor-echo", "output");
    const before = structuredClone(messages);
    const harness = commandHarness(messages, {
      generationResponse: { turns: [result(1, "Clean summary.")] },
      transformGenerationResponse: response => ({
        turns: response.turns.map(turn => ({
          ...turn,
          anchor: "anchor-echo request",
        })),
      }),
    });

    let failure: unknown;
    try {
      await executeMagicCompact(harness.v2, "source", 0, HEURISTIC_OPTIONS);
    } catch (error) {
      failure = error;
    }
    expect(failure).toBeInstanceOf(Error);
    expect((failure as Error).message).toBe(
      "Compaction generation response validation failed: invalid turn shape or identity (shape at target ordinal 1).",
    );
    expect(nestedErrorMessages(failure).join(" ")).not.toContain(
      "anchor-echo request",
    );

    expect(messages).toEqual(before);
    expect(harness.forks).toEqual(["generation-fork"]);
    expect(harness.abortedSessions).toEqual(["generation-fork"]);
    expect(harness.successors).toEqual([]);
    expect(harness.deletedSessions).toEqual(["generation-fork"]);
    expect(harness.sourceWrites).toEqual([]);
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
          result(1, "Clean summary.", [
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

    await expect(
      executeMagicCompact(harness.v2, "source", 0, HEURISTIC_OPTIONS),
    ).rejects.toThrow("response validation failed: nonce leakage.");

    expect(messages).toEqual(before);
    expect(harness.forks).toEqual(["generation-fork"]);
    expect(harness.abortedSessions).toEqual(["generation-fork"]);
    expect(harness.successors).toEqual([]);
    expect(harness.deletedSessions).toEqual(["generation-fork"]);
    expect(harness.sourceWrites).toEqual([]);
    expect(boundaryParts(messages)).toEqual([]);
    expect(await readStats("source")).toBeNull();
    expect(await readStats("backup")).toBeNull();
  });

  test.each([
    [
      "a pending revert",
      { sessionRevert: { messageID: "at-rest-assistant" } },
      "it has a pending revert",
    ],
    [
      "a running native compaction",
      { sessionOverrides: { time: { created: 1, updated: 1, compacting: 2 } } },
      "a native compaction is running",
    ],
    [
      "a busy status",
      { statuses: { source: { type: "busy" } } },
      "it is running",
    ],
    [
      "a retrying status",
      { statuses: { source: { type: "retry", attempt: 1, message: "" } } },
      "it is running",
    ],
    [
      "an unknown status",
      { statuses: { source: { type: "paused" } } },
      "does not support",
    ],
    [
      "a running subagent",
      {
        children: { source: ["child"], child: ["grandchild"] },
        statuses: { grandchild: { type: "busy" } },
      },
      "one of its subagent sessions is running",
    ],
    [
      "a subagent permission request",
      {
        children: { source: ["child"] },
        permissions: [{ sessionID: "child" }],
      },
      "a permission request is waiting",
    ],
    [
      "a question",
      { questions: [{ sessionID: "source" }] },
      "a question is waiting",
    ],
    [
      "a subagent session as the source",
      { sessionOverrides: { parentID: "parent" } },
      "runs only on a root session",
    ],
  ] as const)(
    "refuses %s before generation, successor creation, or writes",
    async (_name, state, expected) => {
      const messages = compactableCommandMessages(
        "at-rest",
        "large output ".repeat(1_000),
      );
      const before = structuredClone(messages);
      const harness = commandHarness(messages, {
        generationResponse: { turns: [result(1, "Completed.")] },
        ...(state as Partial<CommandHarnessOptions>),
      });

      await expect(
        executeMagicCompact(harness.v2, "source", 0, HEURISTIC_OPTIONS),
      ).rejects.toThrow(expected);

      expect(messages).toEqual(before);
      expect(harness.forks).toEqual([]);
      expect(harness.successors).toEqual([]);
      expect(harness.sourceWrites).toEqual([]);
      expect(await readStats("source")).toBeNull();
    },
  );

  test.each([
    ["an active assistant response", "assistant"],
    ["a running tool call", "tool"],
  ] as const)("refuses %s in the source history", async (_name, kind) => {
    const messages = compactableCommandMessages(
      "active",
      "large output ".repeat(1_000),
    );
    if (kind === "assistant") {
      messages[1]!.info.time = { created: messages[1]!.info.time.created };
    } else {
      const tool = messages[1]!.parts.find(part => part.type === "tool")!;
      if (tool.type !== "tool") throw new Error("Expected a tool part.");
      tool.state = { status: "running", input: {}, time: { start: 1 } };
    }
    const harness = commandHarness(messages, {
      generationResponse: { turns: [result(1, "Completed.")] },
    });

    await expect(
      executeMagicCompact(harness.v2, "source", 0, HEURISTIC_OPTIONS),
    ).rejects.toThrow("a model response or tool call is still active");
    expect(harness.forks).toEqual([]);
    expect(harness.successors).toEqual([]);
  });

  test("refuses a second command for the same source while the first runs", async () => {
    const messages = compactableCommandMessages(
      "exclusive",
      "large output ".repeat(1_000),
    );
    let release!: () => void;
    const harness = commandHarness(messages, {
      generationResponse: { turns: [result(1, "Completed.")] },
      generationGate: new Promise<void>(resolve => {
        release = resolve;
      }),
    });

    const first = executeMagicCompact(
      harness.v2,
      "source",
      0,
      HEURISTIC_OPTIONS,
    );
    await waitFor(() => harness.modelPrompts.length === 1);
    await expect(
      executeMagicCompact(harness.v2, "source", 0, HEURISTIC_OPTIONS),
    ).rejects.toThrow("already preparing a successor for this session");
    release();
    await expect(first).resolves.toMatchObject({
      session: { id: "successor" },
    });
    expect(harness.successors).toEqual(["successor"]);
  });

  test("deletes an unready candidate and leaves the source when the source changes before handoff", async () => {
    const messages = compactableCommandMessages(
      "handoff-drift",
      "large output ".repeat(1_000),
    );
    const harness = commandHarness(messages, {
      generationResponse: { turns: [result(1, "Completed.")] },
      driftAfterSuccessorCreate: live => {
        live.push(
          message("handoff-drift-late", "user", [
            textPart("handoff-drift-late-text", "handoff-drift-late", "late"),
          ]),
        );
      },
    });
    const before = structuredClone(messages);

    await expect(
      executeMagicCompact(harness.v2, "source", 0, HEURISTIC_OPTIONS),
    ).rejects.toThrow("The session changed while Magic Compact was preparing");

    expect(messages.slice(0, 2)).toEqual(before);
    expect(harness.sourceWrites).toEqual([]);
    expect(harness.deletedSessions).toContain("successor");
    expect(harness.successorMessages()).toEqual([]);
    expect(await readCache("successor")).toBeNull();
    expect(await readStats("successor")).toBeNull();
    expect(
      harness.updateRequests.some(request => request.sessionID === "successor"),
    ).toBeFalse();
  });

  test("keeps and names a candidate that another writer changed", async () => {
    const messages = compactableCommandMessages(
      "tampered",
      "large output ".repeat(1_000),
    );
    const harness = commandHarness(messages, {
      generationResponse: { turns: [result(1, "Completed.")] },
      tamperCandidate: true,
    });

    await expect(
      executeMagicCompact(harness.v2, "source", 0, HEURISTIC_OPTIONS),
    ).rejects.toThrow(
      "Successor candidate successor was kept because it could not be confirmed as unmodified",
    );
    expect(harness.deletedSessions).not.toContain("successor");
    expect(harness.sourceWrites).toEqual([]);
  });

  test.each([
    ["the first message's ordering seed", 1],
    ["a complete first message", 3],
    ["a complete message and the next message's seed", 4],
  ])(
    "deletes a candidate whose replay stopped after %s",
    async (_name, kept) => {
      const messages = compactableCommandMessages(
        "partial-replay",
        "large output ".repeat(1_000),
      );
      const harness = commandHarness(messages, {
        generationResponse: { turns: [result(1, "Completed.")] },
        failSuccessorReplay: kept,
      });

      await expect(
        executeMagicCompact(harness.v2, "source", 0, HEURISTIC_OPTIONS),
      ).rejects.toThrow(
        "could not verify its successor session (history replay failed)",
      );
      expect(harness.deletedSessions).toContain("successor");
      expect(harness.successorMessages()).toEqual([]);
    },
  );

  test.each([
    ["title", { title: "Renamed elsewhere" }],
    [
      "permission",
      { permission: [{ permission: "bash", pattern: "*", action: "allow" }] },
    ],
    ["model", { model: { id: "other-model", providerID: "other" } }],
    ["share", { share: { url: "https://example.invalid/s" } }],
    ["archive time", { time: { created: 1, updated: 1, archived: 5 } }],
  ] as const)(
    "keeps a candidate whose %s another writer changed without touching its history",
    async (_name, header) => {
      const messages = compactableCommandMessages(
        "foreign-header",
        "large output ".repeat(1_000),
      );
      const harness = commandHarness(messages, {
        generationResponse: { turns: [result(1, "Completed.")] },
        foreignCandidateHeader: header as Partial<Session>,
      });

      await expect(
        executeMagicCompact(harness.v2, "source", 0, HEURISTIC_OPTIONS),
      ).rejects.toThrow(
        "could not verify its successor session (candidate was changed by another writer). The session was left unchanged. Successor candidate successor was kept",
      );
      expect(harness.deletedSessions).not.toContain("successor");
      expect(harness.successorMessages().length).toBeGreaterThan(0);
      expect(
        harness.updateRequests.some(
          request => request.sessionID === "successor",
        ),
      ).toBeFalse();
      expect(harness.sourceWrites).toEqual([]);
    },
  );

  test.each([
    ["a child session", { children: { successor: ["candidate-child"] } }],
    ["a busy status", { statuses: { successor: { type: "busy" } } }],
    ["a permission request", { permissions: [{ sessionID: "successor" }] }],
    ["a question", { questions: [{ sessionID: "successor" }] }],
  ] as const)(
    "keeps a candidate that has %s when the source changes before handoff",
    async (_name, state) => {
      const messages = compactableCommandMessages(
        "candidate-work",
        "large output ".repeat(1_000),
      );
      const harness = commandHarness(messages, {
        generationResponse: { turns: [result(1, "Completed.")] },
        ...(state as Partial<CommandHarnessOptions>),
        driftAfterSuccessorCreate: live => {
          live.push(
            message("candidate-work-late", "user", [
              textPart("candidate-work-late-text", "candidate-work-late", "x"),
            ]),
          );
        },
      });

      await expect(
        executeMagicCompact(harness.v2, "source", 0, HEURISTIC_OPTIONS),
      ).rejects.toThrow(
        "Successor candidate successor was kept because it could not be confirmed as unmodified",
      );
      expect(harness.deletedSessions).not.toContain("successor");
      expect(harness.deletedSessions).not.toContain("candidate-child");
      expect(harness.successorMessages().length).toBeGreaterThan(0);
    },
  );

  test.each([
    [
      "create",
      { successorHostErrors: { create: privateHostError() } },
      "create",
      "could not create a successor session",
    ],
    [
      "import",
      { successorHostErrors: { replay: privateHostError() } },
      "import",
      "(history replay failed)",
    ],
    [
      "ready",
      { successorHostErrors: { update: privateHostError() } },
      "ready",
      "could not confirm that successor session successor became ready",
    ],
    [
      "cleanup",
      {
        successorHostErrors: { delete: privateHostError() },
        driftAfterSuccessorCreate: (live: MessageWithParts[]) => {
          live.push(
            message("private-late", "user", [
              textPart("private-late-text", "private-late", "late"),
            ]),
          );
        },
      },
      "recheck",
      "Successor candidate successor was kept",
    ],
  ] as const)(
    "reports a host failure at %s by stage and fixed text only",
    async (_name, state, stage, text) => {
      const messages = compactableCommandMessages(
        "private",
        "large output ".repeat(1_000),
      );
      const harness = commandHarness(messages, {
        generationResponse: { turns: [result(1, "Completed.")] },
        ...(state as Partial<CommandHarnessOptions>),
      });

      const error = await executeMagicCompact(
        harness.v2,
        "source",
        0,
        HEURISTIC_OPTIONS,
      ).catch((caught: unknown) => caught);

      expect(error).toBeInstanceOf(SuccessorFailure);
      expect(String(error)).toContain(text);
      expect((error as SuccessorFailure).details).toMatchObject({
        stage,
        sourceID: "source",
      });
      expect((error as SuccessorFailure).details.code).toBeUndefined();
      expect(error).not.toHaveProperty("cause");
      const leaked = [
        ...reachableStrings(error),
        ...reachableStrings(harness.toasts),
      ];
      expect(leaked.filter(value => value.includes(PRIVATE_SENTINEL))).toEqual(
        [],
      );
    },
  );

  test("reports a cache failure under a private path by stage and system code only", async () => {
    const privateHome = join(storageDirectory, PRIVATE_SENTINEL);
    process.env.XDG_DATA_HOME = privateHome;
    try {
      // A directory where the candidate's cache file belongs fails its read.
      await mkdir(cachePath("successor"), { recursive: true });
      const messages = compactableCommandMessages(
        "private-cache",
        "large output ".repeat(1_000),
      );
      const harness = commandHarness(messages, {
        generationResponse: { turns: [result(1, "Completed.")] },
      });

      const error = await executeMagicCompact(
        harness.v2,
        "source",
        0,
        HEURISTIC_OPTIONS,
      ).catch((caught: unknown) => caught);

      expect(error).toBeInstanceOf(SuccessorFailure);
      expect(String(error)).toContain(
        "could not finish preparing its successor session",
      );
      expect((error as SuccessorFailure).details).toMatchObject({
        stage: "cache",
        code: "EISDIR",
      });
      expect(
        reachableStrings(error).filter(value =>
          value.includes(PRIVATE_SENTINEL),
        ),
      ).toEqual([]);
    } finally {
      process.env.XDG_DATA_HOME = storageDirectory;
      await rm(privateHome, { recursive: true, force: true });
    }
  });

  test("classifies only fixed plugin details, HTTP statuses and system codes", () => {
    const failure = successorFailure("Magic Trim", privateHostError(), {
      stage: "import",
      sourceID: "source",
    });
    expect(failure.message).toBe(
      "Magic Trim could not finish preparing its successor session. The session was left unchanged.",
    );
    expect(failure.details).toEqual({
      stage: "import",
      sourceID: "source",
      status: 200,
    });
    expect(
      successorFailure(
        "Magic Trim",
        Object.assign(new Error("x"), { status: 600, code: "ENOSPC" }),
        { stage: "stats", sourceID: "source" },
      ).details,
    ).toEqual({ stage: "stats", sourceID: "source", code: "ENOSPC" });
    expect(
      successorFailure("Magic Trim", new ForkHistoryError("replay-failed"), {
        stage: "import",
        sourceID: "source",
      }).details.detail,
    ).toBe("history replay failed");
  });

  describe("background subagents", () => {
    const running = (child: string, summary = "Background task started") =>
      `<task id="${child}" state="running">\n<summary>${summary}</summary>\n<task_result>\nworking\n</task_result>\n</task>`;
    const call = (id: string, child: string, output: string) =>
      message(`${id}-assistant`, "assistant", [
        toolPart(
          `${id}-task`,
          `${id}-assistant`,
          {
            status: "completed",
            input: { description: "d", prompt: "p", subagent_type: "general" },
            output,
            title: "d",
            metadata: {
              parentSessionId: "source",
              sessionId: child,
              background: true,
              jobId: child,
            },
            time: { start: 1, end: 2 },
          },
          "task",
        ),
      ]);
    const delivery = (id: string, child: string, synthetic = true) => {
      const text = textPart(
        `${id}-text`,
        `${id}-user`,
        `<task id="${child}" state="completed">\n<summary>Background task completed: d</summary>\n<task_result>\ndone\n</task_result>\n</task>`,
      );
      return message(`${id}-user`, "user", [
        synthetic ? { ...text, synthetic: true } : text,
      ]);
    };
    const read = (history: MessageWithParts[]) =>
      readSourceAtRest(
        commandHarness(
          [...compactableCommandMessages("bg", "output"), ...history],
          { generationResponse: { turns: [] } },
        ).v2,
        "Magic Trim",
        "source",
        1_000,
      );

    test.each([
      [
        "a started job without its result",
        [call("a", "ses_kid", running("ses_kid"))],
      ],
      [
        "a result that is not a host delivery",
        [
          call("a", "ses_kid", running("ses_kid")),
          delivery("b", "ses_kid", false),
        ],
      ],
      [
        "a second start of the same child with one result",
        [
          call("a", "ses_kid", running("ses_kid")),
          delivery("b", "ses_kid"),
          call("c", "ses_kid", running("ses_kid")),
        ],
      ],
      [
        "a result stored before its job started",
        [delivery("b", "ses_kid"), call("a", "ses_kid", running("ses_kid"))],
      ],
    ])("refuses %s, naming the child", async (_name, history) => {
      await expect(read(history)).rejects.toThrow(
        "background subagent session ses_kid has not delivered its result",
      );
    });

    test("refuses a background call without a child session", async () => {
      const unmatched = call("a", "ses_kid", running("ses_kid"));
      const part = unmatched.parts[0]!;
      if (part.type !== "tool" || part.state.status !== "completed")
        throw new Error("Expected a completed tool part.");
      delete part.state.metadata["sessionId"];
      await expect(read([unmatched])).rejects.toThrow(
        "a background subagent cannot be matched to its result",
      );
    });

    test.each([
      [
        "a started job and its result",
        [call("a", "ses_kid", running("ses_kid")), delivery("b", "ses_kid")],
      ],
      [
        "a message to a running job that reuses the first result",
        [
          call("a", "ses_kid", running("ses_kid")),
          call("c", "ses_kid", running("ses_kid", "Background task updated")),
          delivery("b", "ses_kid"),
        ],
      ],
    ])("accepts %s", async (_name, history) => {
      await expect(read(history)).resolves.toMatchObject({
        session: { id: "source" },
      });
    });
  });

  test("keeps and names a candidate whose ready handoff is uncertain", async () => {
    const messages = compactableCommandMessages(
      "uncertain-ready",
      "large output ".repeat(1_000),
    );
    const harness = commandHarness(messages, {
      generationResponse: { turns: [result(1, "Completed.")] },
      failReadyUpdate: true,
    });

    await expect(
      executeMagicCompact(harness.v2, "source", 0, HEURISTIC_OPTIONS),
    ).rejects.toThrow(
      "could not confirm that successor session successor became ready",
    );
    expect(harness.deletedSessions).not.toContain("successor");
    expect(harness.successorMessages().length).toBeGreaterThan(0);
  });

  test("copies continuation state into the successor and marks it ready last", async () => {
    const messages = compactableCommandMessages(
      "continuation",
      "large output ".repeat(1_000),
    );
    const harness = commandHarness(messages, {
      generationResponse: { turns: [result(1, "Completed.")] },
      sessionOverrides: {
        agent: "build",
        permission: [{ permission: "bash", pattern: "*", action: "deny" }],
        metadata: { owner: "operator" },
      },
    });

    await executeMagicCompact(harness.v2, "source", 0, HEURISTIC_OPTIONS);

    expect(harness.createRequests).toEqual([
      expect.objectContaining({
        directory: "/workspace",
        workspace: "workspace",
        workspaceID: "workspace",
        title: "Source (preparing)",
        agent: "build",
        model: { id: "source-model", providerID: "source-provider" },
        permission: [{ permission: "bash", pattern: "*", action: "deny" }],
        metadata: {
          owner: "operator",
          magicCompact: expect.objectContaining({
            successorOf: "source",
            ready: false,
          }),
        },
      }),
    ]);
    expect(harness.createRequests[0]).not.toHaveProperty("parentID");
    expect(harness.updateRequests.at(-1)).toMatchObject({
      sessionID: "successor",
      title: "Source",
      metadata: {
        owner: "operator",
        magicCompact: { successorOf: "source", ready: true },
      },
    });
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

  test("keeps prior summary turns opaque and carries cache/stat lineage into the successor", async () => {
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
    const sourceCache = await readCache("source");
    await writeStats(
      "source",
      conversationStats({
        rootSessionId: "root-session",
        totalTokensPruned: 10,
        cachedTokensSaved: 3,
        processedAssistantMessageIds: ["prior-assistant"],
      }),
    );
    const sourceStats = await readStats("source");
    const harness = commandHarness(messages, {
      generationResponse: {
        turns: [result(1, "Completed the second compaction candidate.")],
      },
    });

    await expect(
      executeMagicCompact(harness.v2, "source", 0, HEURISTIC_OPTIONS),
    ).resolves.toMatchObject({ session: { id: "successor" } });

    const successor = harness.successorMessages();
    expect(withoutIdentity(successor.slice(0, 2))).toEqual(
      withoutIdentity(historicalBefore),
    );
    expect(summaryParts(successor)).toHaveLength(2);
    // Existing omission IDs keep their bytes and IDs in the successor cache,
    // and the source's caches and stats are unchanged.
    expect(await readOmittedContent("successor", contentID)).toBe(
      "lineage omission",
    );
    expect(await readCache("source")).toEqual(sourceCache);
    expect(await readStats("source")).toEqual(sourceStats);
    expect(await readStats("successor")).toMatchObject({
      sourceSessionId: "source",
      rootSessionId: "root-session",
      cachedTokensSaved: 3,
      processedAssistantMessageIds: successor
        .filter(item => item.info.role === "assistant")
        .map(item => item.info.id),
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
    deleteGate?: Promise<void>;
    deleteGateForFirstOrdinal?: number;
    nextTurn?: Turn;
    markerError?: Error;
    modelImageCapability?: boolean;
    responseFinish?: string;
    responseError?: unknown;
    responseText?: string;
    omitStructured?: boolean;
    permissionError?: Error;
    abortError?: Error;
    deleteError?: Error;
    summarizerAgent?: "missing" | "native-only" | "model-missing";
    unfilteredResponse?: boolean;
    providerDelayByFirstOrdinal?: Record<number, number>;
    providerErrorForFirstOrdinal?: number;
    holdPromptsUntil?: number;
    modelLimits?: {
      context: number;
      input?: number;
      output?: number;
    } | null;
    sourceMessages?: MessageWithParts[];
    // Edits the response to the numbered (1-based) read of a fork: reads 1 and
    // 2 are fork initialization, 3 the shape check, 4 the retained recheck.
    mutateForkRead?: (
      load: number,
      messages: MessageWithParts[],
      sessionID: string,
    ) => void;
    // Awaited before the numbered (1-based) fork history read is answered.
    forkReadFault?: (load: number) => Promise<unknown> | undefined;
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
  const promptFirstOrdinalByFork = new Map<string, number>();
  const liveForks = new Set<string>();
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
  const history = persistedForkHistory(sourceMessages);
  const modelImageCapability =
    "modelImageCapability" in options ? options.modelImageCapability : true;
  const configuredAgent = {
    name: "magic-compact-summarizer",
    native: false,
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
            options.summarizerAgent === "missing"
              ? []
              : options.summarizerAgent === "native-only"
                ? [{ ...configuredAgent, name: "compaction", native: true }]
                : options.summarizerAgent === "model-missing"
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
        liveForks.add(forkID);
        history.fork("source", forkID, request.messageID as string | undefined);
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
        if (request.sessionID === "source")
          return { data: structuredClone(sourceMessages) };
        const loads = (messageLoadsByFork.get(request.sessionID) ?? 0) + 1;
        messageLoadsByFork.set(request.sessionID, loads);
        await options.forkReadFault?.(loads);
        const stored = history.read(request.sessionID);
        options.mutateForkRead?.(loads, stored, request.sessionID);
        return { data: stored };
      },
      prompt: async (request: Record<string, unknown>) => {
        prompts.push(request);
        events.push("prompt");
        const sessionID = String(request.sessionID);
        promptHistories.push({
          sessionID,
          messages: history.read(sessionID),
        });
        const index = generationPromptIndex(request);
        const firstOrdinal = index.turns[0]!.ordinal;
        promptFirstOrdinalByFork.set(sessionID, firstOrdinal);
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
          const structured = responseWithPromptMarkers(
            request,
            response,
            options.unfilteredResponse,
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
                ...(!options.omitStructured ? { structured } : {}),
              },
              parts: options.responseText
                ? [textPart("response-text", "response", options.responseText)]
                : [],
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
        const sessionID = String(request.sessionID);
        if (
          options.deleteGate
          && (options.deleteGateForFirstOrdinal === undefined
            || promptFirstOrdinalByFork.get(sessionID)
              === options.deleteGateForFirstOrdinal)
        ) {
          events.push("delete-waiting");
          await options.deleteGate;
        }
        deleteRequests.push(request);
        deletes.push(sessionID);
        events.push("delete");
        if (options.deleteError) throw options.deleteError;
        if (liveForks.delete(sessionID)) activeForks -= 1;
        history.delete(sessionID);
        return { data: true };
      },
      deleteMessage: async (request: {
        sessionID: string;
        messageID: string;
      }) => {
        events.push("message-delete");
        messageDeletes.push(request);
        history.deleteMessage(request.sessionID, request.messageID);
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
        history.updatePart(request.part);
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
    sync: {
      replay: async (request: {
        events: Parameters<typeof history.replay>[0];
      }) => {
        history.replay(request.events);
        return { data: { sessionID: request.events[0]!.aggregateID } };
      },
    },
  } as unknown as V2Client;
  return {
    history,
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

function chronologyHost(
  source: MessageWithParts[],
  options: { clock?: () => number } = {},
) {
  const history = persistedForkHistory(source, options);
  const host = {
    history,
    deleted: [] as string[],
    afterReplay: undefined as (() => void | Promise<void>) | undefined,
    beforeFork: undefined as (() => void) | undefined,
    deleteError: undefined as Error | undefined,
    // Session reported by the replay acknowledgement; defaults to the fork.
    replayedSession: undefined as string | undefined,
    // Replaces the numbered (1-based) history read's response; undefined reads
    // the fixture. Never-settling or late-settling replacements are allowed.
    readOverride: undefined as
      ((sessionID: string, read: number) => unknown) | undefined,
    reads: [] as string[],
    forkRequests: [] as Array<Record<string, unknown>>,
    replayCalls: 0,
    get forkCalls() {
      return host.forkRequests.length;
    },
    v2: undefined as unknown as V2Client,
  };
  host.v2 = {
    session: {
      fork: async (request: { sessionID: string; messageID?: string }) => {
        host.forkRequests.push(request);
        host.beforeFork?.();
        history.fork(request.sessionID, "backup", request.messageID);
        return { data: { ...sourceSession(), id: "backup" } };
      },
      messages: async (request: { sessionID: string }) => {
        const read = host.reads.push(request.sessionID);
        const replacement = await host.readOverride?.(request.sessionID, read);
        return replacement ?? { data: history.read(request.sessionID) };
      },
      update: async () => ({ data: {} }),
      delete: async (request: { sessionID: string }) => {
        host.deleted.push(request.sessionID);
        if (host.deleteError) throw host.deleteError;
        history.delete(request.sessionID);
        return { data: true };
      },
    },
    sync: {
      replay: async (request: {
        events: Parameters<typeof history.replay>[0];
      }) => {
        host.replayCalls += 1;
        history.replay(request.events);
        await host.afterReplay?.();
        return {
          data: {
            sessionID: host.replayedSession ?? request.events[0]!.aggregateID,
          },
        };
      },
    },
  } as unknown as V2Client;
  return host;
}

function responseFor(turns: Turn[]): GenerationResponse {
  return {
    turns: turns.map((turn, index) =>
      result(
        index + 1,
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
  summary: string,
  visualDescriptions: Array<{ ordinal: number; description: string }> = [],
): TurnResult {
  return { ordinal, summary, visualDescriptions };
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
    effectiveSourceTurnStartIndex: 0,
    effectiveSourceTurnEndIndex: turns.length + (nextTurn ? 1 : 0),
    sourceSnapshot: snapshotSourceMessages(messages),
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
  const summary = message(id, "assistant", [
    textPart(`${id}-text`, id, "checkpoint"),
  ]);
  Object.assign(summary.info, {
    parentID,
    summary: true,
    ...(overrides.finish ? { finish: overrides.finish } : {}),
  });
  return summary;
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

// Source-session record with a unique, ascending creation time, as the host
// always supplies one.
let messageClock = 0;
function message(
  id: string,
  role: "user" | "assistant",
  parts: Part[],
): MessageWithParts {
  return {
    info: {
      id,
      role,
      sessionID: "source",
      time:
        role === "assistant"
          ? { created: ++messageClock, completed: messageClock }
          : { created: ++messageClock },
    } as Message,
    parts,
  };
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
  tool = "read",
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

function sourceSession(): Session {
  return {
    id: "source",
    title: "Source",
    directory: "/workspace",
    workspaceID: "workspace",
    model: { id: "source-model", providerID: "source-provider" },
    time: { created: 1, updated: 1 },
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

// Each request answers only its own indexed targets unless a test sends the
// response through unchanged to exercise malformed results.
function responseWithPromptMarkers(
  request: Record<string, unknown>,
  response: GenerationResponse,
  unfiltered = false,
): GenerationResponse {
  const index = generationPromptIndex(request);
  const ordinals = new Set(index.turns.map(turn => turn.ordinal));
  return {
    ...response,
    turns: response.turns
      .filter(turn => unfiltered || ordinals.has(turn.ordinal))
      .map((turn, turnIndex) => ({
        ...turn,
        marker: turn.marker ?? index.turns[turnIndex]?.marker ?? "",
      })),
  };
}

function generationPromptIndex(request: Record<string, unknown>) {
  return JSON.parse(promptText(request).split("\n").at(-1)!) as {
    turns: Array<{
      ordinal: number;
      marker: string;
      expectedInlineImages: number;
    }>;
  };
}

function generationRequestInputTokens(
  request: Record<string, unknown>,
  retainedMessages: MessageWithParts[],
): number {
  return (
    NATIVE_BATCH_FIXED_RESERVE_TOKENS
    + generationSchemaTokens(request)
    + countGptTokens(promptText(request))
    + buildTurns(retainedMessages).reduce(
      (total, turn) => total + estimateNativeTurnCost(turn).inputTokens,
      0,
    )
  );
}

function conservativeGenerationRequestInputTokens(
  request: Record<string, unknown>,
  retainedMessages: MessageWithParts[],
): number {
  const nonce = generationPromptIndex(request).turns[0]!.marker.match(
    /nonce="([A-Za-z0-9_-]{22})"/u,
  )?.[1];
  if (!nonce) throw new Error("Generation prompt marker nonce is missing.");
  const replaceNonce = (text: string) =>
    text.split(nonce).join(NATIVE_BATCH_PLANNING_NONCE);
  const actualHistoryTokens = buildTurns(retainedMessages).reduce(
    (total, turn) => total + estimateNativeTurnCost(turn).inputTokens,
    0,
  );
  const markerTokenAdjustment = retainedMessages
    .flatMap(message => message.parts)
    .filter(
      (part): part is TextPart =>
        part.type === "text"
        && part.synthetic === true
        && part.text.includes(nonce),
    )
    .reduce(
      (total, part) =>
        total
        + countGptTokens(replaceNonce(part.text))
        - countGptTokens(part.text),
      0,
    );
  return (
    NATIVE_BATCH_FIXED_RESERVE_TOKENS
    + generationSchemaTokens(request)
    + countGptTokens(replaceNonce(promptText(request)))
    + actualHistoryTokens
    + markerTokenAdjustment
  );
}

function generationSchemaTokens(request: Record<string, unknown>): number {
  const format = request["format"] as { schema: Record<string, unknown> };
  return countGptTokens(JSON.stringify(format.schema));
}

function generationTurnsSchema(request: Record<string, unknown>) {
  const format = request["format"] as {
    schema: {
      properties: {
        turns: { minItems: number; maxItems: number; items: unknown };
      };
    };
  };
  return format.schema.properties.turns;
}

function estimatedGenerationResponseTokens(
  request: Record<string, unknown>,
  turns: Turn[],
  conservative = false,
): number {
  const index = generationPromptIndex(request);
  const nonce = index.turns[0]!.marker.match(
    /nonce="([A-Za-z0-9_-]{22})"/u,
  )?.[1];
  if (!nonce) throw new Error("Generation prompt marker nonce is missing.");
  const contentTokens = index.turns.reduce(
    (total, candidate) =>
      total
      + estimateNativeTurnCost(turns[candidate.ordinal - 1]!).outputTokens,
    0,
  );
  const protocolTokens = countGptTokens(
    JSON.stringify({
      turns: index.turns.map(candidate => ({
        ordinal: candidate.ordinal,
        marker: conservative
          ? candidate.marker.split(nonce).join(NATIVE_BATCH_PLANNING_NONCE)
          : candidate.marker,
        summary: "",
        visualDescriptions: Array.from(
          { length: candidate.expectedInlineImages },
          (_, visualIndex) => ({
            ordinal: visualIndex + 1,
            description: "",
          }),
        ),
      })),
    }),
  );
  return contentTokens + protocolTokens;
}

function configuredInputForDerivedLimit(target: number, ratio: 0.8 | 0.9) {
  let input = Math.ceil(target / ratio);
  while (Math.floor(input * ratio) > target) input -= 1;
  while (Math.floor(input * ratio) < target) input += 1;
  return input;
}

function generationGlobalTurnIndexText(
  request: Record<string, unknown>,
): string {
  for (const line of promptText(request).split("\n")) {
    try {
      const value = JSON.parse(line) as Record<string, unknown>;
      if (Array.isArray(value.globalTurnIndex)) return line;
    } catch {
      continue;
    }
  }
  throw new Error("Generation prompt is missing its global turn index.");
}

function generationGlobalTurnIndex(request: Record<string, unknown>) {
  return (
    JSON.parse(generationGlobalTurnIndexText(request)) as {
      globalTurnIndex: Array<{ sourceOrdinal: number; anchor: string }>;
    }
  ).globalTurnIndex;
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
      ...(part.type === "compaction" && part.tail_start_id
        ? { tail_start_id: messageIDs.get(part.tail_start_id) }
        : {}),
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
  sessionRevert?: { messageID: string };
  // Applied to the source session as returned by session.get.
  sessionOverrides?: Partial<Session>;
  statuses?: Record<string, unknown>;
  children?: Record<string, string[]>;
  permissions?: Array<{ sessionID: string }>;
  questions?: Array<{ sessionID: string }>;
  // Changes the live source once the successor session exists, so only the
  // final at-rest recheck can see it.
  driftAfterSuccessorCreate?: (messages: MessageWithParts[]) => void;
  // A concurrent writer adds a message to the candidate after replay.
  tamperCandidate?: boolean;
  // A concurrent writer changes only the candidate's session record after
  // replay.
  foreignCandidateHeader?: Partial<Session>;
  // The successor replay persists this many events, then fails.
  failSuccessorReplay?: number;
  failReadyUpdate?: boolean;
  // Thrown by the host for the successor at that call instead of a response.
  successorHostErrors?: Partial<
    Record<"create" | "replay" | "update" | "delete", unknown>
  >;
  generationGate?: Promise<void>;
  toastMode?: "hang" | "fail" | "delay-progress";
};

// Fake OpenCode host for command-level tests. `messages` is the live source;
// successors and temporary forks live in the persisted host fake. Every source
// write is recorded so tests can prove the source stays read-only.
function commandHarness(
  messages: MessageWithParts[],
  options: CommandHarnessOptions,
) {
  const events: string[] = [];
  const forks: string[] = [];
  const sourceWrites: string[] = [];
  const abortedSessions: string[] = [];
  const deletedSessions: string[] = [];
  const createRequests: Record<string, unknown>[] = [];
  const updateRequests: Record<string, unknown>[] = [];
  const toasts: Array<{
    title?: string;
    message?: string;
    variant?: string;
    duration?: number;
  }> = [];
  let displayedToast: (typeof toasts)[number] | undefined;
  let releaseProgressToast: (() => void) | undefined;
  const progressToastGate =
    options.toastMode === "delay-progress"
      ? new Promise<void>(resolve => {
          releaseProgressToast = resolve;
        })
      : undefined;
  const history = persistedForkHistory([]);
  let generationForkCount = 0;
  let successorCount = 0;
  const successors: string[] = [];
  const modelPrompts: Record<string, unknown>[] = [];
  const activeGenerationForks = new Set<string>();
  const generationFirstOrdinalByFork = new Map<string, number>();
  let activeGenerationPrompts = 0;
  let releaseGenerationPromptGate: (() => void) | undefined;
  const generationPromptGate = options.holdGenerationPromptsUntil
    ? new Promise<void>(resolve => {
        releaseGenerationPromptGate = resolve;
      })
    : undefined;
  const sessionInfo = new Map<string, Session>();
  const sourceInfo = (): Session => ({
    ...sourceSession(),
    ...options.sessionOverrides,
    ...(options.sessionRevert ? { revert: options.sessionRevert } : {}),
  });

  const v2 = {
    app: {
      agents: async () => ({
        data: [
          {
            name: "magic-compact-summarizer",
            native: false,
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
    permission: { list: async () => ({ data: options.permissions ?? [] }) },
    question: { list: async () => ({ data: options.questions ?? [] }) },
    session: {
      get: async (request: { sessionID: string }) => ({
        data:
          request.sessionID === "source"
            ? sourceInfo()
            : sessionInfo.get(request.sessionID),
      }),
      status: async () => ({ data: options.statuses ?? {} }),
      children: async (request: { sessionID: string }) => ({
        data: (options.children?.[request.sessionID] ?? []).map(id => ({
          ...sourceSession(),
          id,
          parentID: request.sessionID,
        })),
      }),
      messages: async (request: { sessionID: string }) => ({
        data:
          request.sessionID === "source"
            ? structuredClone(messages)
            : history.read(request.sessionID),
      }),
      fork: async (request: { messageID?: string }) => {
        history.delete("source");
        messages.forEach((item, index) => history.write(item, index));
        const forkID =
          generationForkCount++ === 0
            ? "generation-fork"
            : `generation-fork-${generationForkCount}`;
        forks.push(forkID);
        events.push("fork");
        history.fork("source", forkID, request.messageID);
        activeGenerationForks.add(forkID);
        return { data: { ...sourceSession(), id: forkID } };
      },
      create: async (request: Record<string, unknown>) => {
        createRequests.push(request);
        if (options.successorHostErrors?.create !== undefined)
          throw options.successorHostErrors.create;
        const id =
          successorCount++ === 0 ? "successor" : `successor-${successorCount}`;
        successors.push(id);
        events.push("successor-create");
        const info = {
          ...sourceSession(),
          id,
          title: String(request.title),
          agent: request.agent,
          model: request.model,
          permission: request.permission,
          metadata: request.metadata,
        } as Session;
        sessionInfo.set(id, info);
        options.driftAfterSuccessorCreate?.(messages);
        return { data: info };
      },
      update: async (request: Record<string, unknown>) => {
        const sessionID = String(request.sessionID);
        if (sessionID === "source") sourceWrites.push("session.update");
        updateRequests.push(request);
        if (
          successors.includes(sessionID)
          && request.metadata !== undefined
          && options.failReadyUpdate
        ) {
          throw new Error("ready update failed");
        }
        if (
          successors.includes(sessionID)
          && options.successorHostErrors?.update !== undefined
        ) {
          throw options.successorHostErrors.update;
        }
        const info = {
          ...(sessionInfo.get(sessionID) ?? {
            ...sourceSession(),
            id: sessionID,
          }),
          ...(request.title === undefined ? {} : { title: request.title }),
          ...(request.metadata === undefined
            ? {}
            : { metadata: request.metadata }),
        } as Session;
        sessionInfo.set(sessionID, info);
        return { data: info };
      },
      prompt: async (request: Record<string, unknown>) => {
        const sessionID = String(request.sessionID);
        if (sessionID === "source") {
          sourceWrites.push("session.prompt");
          throw new Error("source prompt is not allowed");
        }
        // Task-aware retention summarizes the whole session on its own fork
        // through the summarizer agent, without a response format, before
        // any turn generation.
        if (request.format === undefined) {
          events.push("retention-summary");
          const fork = history.read(sessionID);
          const reply = summaryReply(fork, request as never, sessionID);
          fork
            .slice(-2)
            .forEach((item, index) => history.write(item, 100 + index));
          return reply;
        }
        modelPrompts.push(request);
        events.push("generation-prompt");
        const firstOrdinal = generationPromptIndex(request).turns[0]!.ordinal;
        generationFirstOrdinalByFork.set(sessionID, firstOrdinal);
        activeGenerationPrompts += 1;
        if (
          options.holdGenerationPromptsUntil
          && activeGenerationPrompts >= options.holdGenerationPromptsUntil
        ) {
          releaseGenerationPromptGate?.();
        }
        try {
          await generationPromptGate;
          await options.generationGate;
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
          return {
            data: {
              info: {
                id: "generation-response",
                structured: options.transformGenerationResponse
                  ? options.transformGenerationResponse(response)
                  : response,
              },
              parts: [],
            },
          };
        } finally {
          activeGenerationPrompts -= 1;
        }
      },
      abort: async (request: { sessionID: string }) => {
        abortedSessions.push(request.sessionID);
        if (
          options.generationAbortErrorForFirstOrdinal !== undefined
          && generationFirstOrdinalByFork.get(request.sessionID)
            === options.generationAbortErrorForFirstOrdinal
        ) {
          throw new Error("raw generation abort payload");
        }
        return { data: true };
      },
      delete: async (request: { sessionID: string }) => {
        if (request.sessionID === "source") sourceWrites.push("session.delete");
        if (
          successors.includes(request.sessionID)
          && options.successorHostErrors?.delete !== undefined
        ) {
          throw options.successorHostErrors.delete;
        }
        deletedSessions.push(request.sessionID);
        if (request.sessionID.startsWith("generation-fork")) {
          events.push("generation-delete");
          if (
            options.generationDeleteErrorForFirstOrdinal !== undefined
            && generationFirstOrdinalByFork.get(request.sessionID)
              === options.generationDeleteErrorForFirstOrdinal
          ) {
            throw new Error("raw generation delete payload");
          }
          activeGenerationForks.delete(request.sessionID);
        }
        history.delete(request.sessionID);
        return { data: true };
      },
      deleteMessage: async (request: {
        sessionID: string;
        messageID: string;
      }) => {
        if (request.sessionID === "source") {
          sourceWrites.push("session.deleteMessage");
          throw new Error("source delete is not allowed");
        }
        history.deleteMessage(request.sessionID, request.messageID);
        return { data: true };
      },
    },
    part: {
      update: async (request: { sessionID: string; part: Part }) => {
        if (request.sessionID === "source") {
          sourceWrites.push("part.update");
          throw new Error("source part update is not allowed");
        }
        history.updatePart(request.part);
        return { data: request.part };
      },
      delete: async (request: { sessionID: string }) => {
        sourceWrites.push(`part.delete:${request.sessionID}`);
        throw new Error("part delete is not used");
      },
    },
    sync: {
      replay: async (request: {
        events: Parameters<typeof history.replay>[0];
      }) => {
        const target = request.events[0]!.aggregateID;
        if (
          successors.includes(target)
          && options.failSuccessorReplay !== undefined
        ) {
          history.replay(request.events.slice(0, options.failSuccessorReplay));
          throw new Error("replay failed");
        }
        if (
          successors.includes(target)
          && options.successorHostErrors?.replay !== undefined
        ) {
          throw options.successorHostErrors.replay;
        }
        history.replay(request.events);
        if (successors.includes(target) && options.foreignCandidateHeader) {
          sessionInfo.set(target, {
            ...sessionInfo.get(target)!,
            ...options.foreignCandidateHeader,
          });
        }
        if (successors.includes(target) && options.tamperCandidate) {
          const foreign = message("foreign-candidate-user", "user", [
            textPart(
              "foreign-candidate-text",
              "foreign-candidate-user",
              "written by another client",
            ),
          ]);
          foreign.info.sessionID = target;
          foreign.parts[0]!.sessionID = target;
          history.write(foreign, 10_000);
        }
        return { data: { sessionID: target } };
      },
    },
    tui: {
      showToast: async (toast: {
        title?: string;
        message?: string;
        variant?: string;
        duration?: number;
      }) => {
        toasts.push(toast);
        if (
          options.toastMode === "delay-progress"
          && toast.duration === 2_147_483_647
        ) {
          await progressToastGate;
        }
        displayedToast = toast;
        if (options.toastMode === "hang") return new Promise(() => {});
        if (options.toastMode === "fail") {
          throw new Error("toast failed");
        }
        return { data: true };
      },
    },
  } as unknown as V2Client;

  return {
    v2,
    messages,
    modelPrompts,
    events,
    forks,
    sourceWrites,
    abortedSessions,
    deletedSessions,
    createRequests,
    updateRequests,
    toasts,
    releaseProgressToast: () => releaseProgressToast?.(),
    get displayedToast() {
      return displayedToast;
    },
    get successors() {
      return [...successors];
    },
    successorMessages(id = "successor") {
      return history.read(id);
    },
    successorInfo(id = "successor") {
      return sessionInfo.get(id);
    },
    get activeGenerationForks() {
      return [...activeGenerationForks];
    },
  };
}

describe("magic compact task-aware retention", () => {
  function retainedTurns() {
    const useful = compactableCommandMessages(
      "useful",
      "useful parser finding ".repeat(400),
    );
    const stale = compactableCommandMessages(
      "stale",
      "stale directory listing ".repeat(400),
    );
    return [...useful, ...stale];
  }

  function taskAware(calls: JevCall[] = []) {
    return {
      retention: {
        mode: "task-aware" as const,
        apiKey: "test-key",
        fetch: scoredJevFetch(
          output => (output.includes("useful") ? 0.95 : 0.1),
          calls,
        ),
      },
    };
  }

  test("keeps a judged-useful call whole while summarizing its turn", async () => {
    const messages = retainedTurns();
    const usefulTool = structuredClone(
      messages[1]!.parts.find(part => part.type === "tool")!,
    );
    const calls: JevCall[] = [];
    const harness = commandHarness(messages, {
      generationResponse: {
        turns: [
          result(1, "Useful turn summary."),
          result(2, "Stale turn summary."),
        ],
      },
    });

    await expect(
      executeMagicCompact(harness.v2, "source", 0, taskAware(calls)),
    ).resolves.toMatchObject({ session: { id: "successor" } });

    expect(calls.map(call => call.candidate.output.slice(0, 6))).toEqual([
      "useful",
      "stale ",
    ]);
    expect(calls[0]!.task.latestUserRequest).toBe("stale request");
    expect(harness.events.slice(0, 3)).toEqual([
      "fork",
      "retention-summary",
      "generation-delete",
    ]);
    const successor = harness.successorMessages();
    expect(summaryParts(successor)).toHaveLength(2);
    const tools = successor
      .flatMap(item => item.parts)
      .filter(part => part.type === "tool");
    expect(withoutPartIdentity(tools[0]!)).toEqual(
      withoutPartIdentity(usefulTool),
    );
    expect(JSON.stringify(tools[1])).toContain(
      "Historical read contents omitted",
    );
    expect(harness.toasts.at(-1)?.message).toBe(
      `Compacted 2 request turn(s) into successor session successor. ${preservedTurnsFeedback({ requested: 0, preserved: 0 })} Task-aware retention kept 1 of 2 evaluated tool call(s) whole.`,
    );
  });

  test("counts kept payloads in per-turn profitability and the savings gate", async () => {
    // Without retention the large read output makes this turn profitable.
    const heuristic = compactableCommandMessages(
      "gate",
      "useful parser finding ".repeat(400),
    );
    heuristic[1]!.parts = heuristic[1]!.parts.filter(
      part => part.type !== "reasoning",
    );
    const retained = structuredClone(heuristic);
    const before = structuredClone(retained);
    const response = {
      generationResponse: { turns: [result(1, "Gate turn summary.")] },
    };

    const heuristicHarness = commandHarness(heuristic, response);
    await expect(
      executeMagicCompact(heuristicHarness.v2, "source", 0, HEURISTIC_OPTIONS),
    ).resolves.toMatchObject({ session: { id: "successor" } });
    await deleteStats("source");
    await deleteCache("source");

    const retainedHarness = commandHarness(retained, response);
    await expect(
      executeMagicCompact(retainedHarness.v2, "source", 0, taskAware()),
    ).rejects.toThrow(COMPACT_INSUFFICIENT_SAVINGS);
    expect(retained).toEqual(before);
    expect(retainedHarness.forks).not.toContain("backup");
    expect(await readStats("source")).toBeNull();
  });

  test("a classification failure stops before turn generation, backup, or writes", async () => {
    const messages = retainedTurns();
    const before = structuredClone(messages);
    const harness = commandHarness(messages, {
      generationResponse: { turns: [result(1, "unused"), result(2, "unused")] },
    });

    const error = await executeMagicCompact(harness.v2, "source", 0, {
      retention: {
        mode: "task-aware",
        apiKey: "test-key",
        fetch: async () => new Response("private", { status: 429 }),
      },
    }).then(
      () => undefined,
      (caught: unknown) => caught,
    );

    expect(error).toBeInstanceOf(RetentionClassificationError);
    expect(String(error)).toContain("/magic-trim --heuristic");
    expect(messages).toEqual(before);
    expect(harness.modelPrompts).toEqual([]);
    expect(harness.forks).toEqual(["generation-fork"]);
    expect(harness.activeGenerationForks).toEqual([]);
    expect(harness.sourceWrites).toEqual([]);
    expect(await readStats("source")).toBeNull();
  });
});

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

function isBoundary(part: Part): boolean {
  return (
    part.type === "text"
    && (part.metadata?.magicCompact as { boundary?: true } | undefined)
      ?.boundary === true
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

// Successor records carry fresh session, message and part IDs; these views
// compare content and order without them.
function withoutPartIdentity(part: Part): unknown {
  const copy: Record<string, unknown> = { ...part };
  delete copy["id"];
  delete copy["sessionID"];
  delete copy["messageID"];
  return copy;
}

function withoutIdentity(messages: MessageWithParts[]): unknown[] {
  return messages.map(item => {
    const info: Record<string, unknown> = { ...item.info };
    delete info["id"];
    delete info["sessionID"];
    delete info["parentID"];
    return { info, parts: item.parts.map(withoutPartIdentity) };
  });
}

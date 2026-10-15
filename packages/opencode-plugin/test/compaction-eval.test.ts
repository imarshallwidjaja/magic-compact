// Compaction model eval (eval/compaction): frozen corpus, server identity,
// model selection, artifact cache, preflight, the mechanical oracle, removal
// and storage cleanup, ledger reconciliation, scratch ownership, saved-run
// validation and scoring. Everything here is offline: in-memory hosts stand
// in for the server, and the artifact test packs a dependency-free stand-in
// subject.
import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { execFileSync } from "node:child_process";
import {
  cpSync,
  existsSync,
  lstatSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  realpathSync,
  rmSync,
  symlinkSync,
  unlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { z } from "zod";
import {
  ArtifactCacheError,
  PACKAGE_DIR,
  prepareArtifact,
} from "../eval/compaction/artifact";
import {
  checkCompactionPlans,
  CORPUS_DIR,
  CorpusError,
  loadCorpus,
  sha256,
  v2SourceHistory,
} from "../eval/compaction/corpus";
import {
  GRADES_FORMAT,
  JUDGE_SYSTEM,
  JUDGMENT_JSON_SCHEMA,
  JudgmentSchema,
  judgePrompt,
  judgmentProblem,
  parseJudgeAnswer,
  score,
  validateGrades,
  type Judgment,
} from "../eval/compaction/grade";
import {
  EvalError,
  identifyServer,
  loopbackServerUrl,
  parseModelSelector,
  stateOf,
  type EffectiveSetup,
  type EvalHost,
  type ExpectedState,
  type FetchLike,
  type SessionState,
} from "../eval/compaction/host";
import { v1Host } from "../eval/compaction/host-v1";
import { createdState, v2Host } from "../eval/compaction/host-v2";
import {
  classifyOutcome,
  v1SuccessorInspection,
  v2SuccessorInspection,
} from "../eval/compaction/inspect";
import { summarizeRun } from "../eval/compaction/report";
import {
  cleanupRun,
  gradeRunWithJudge,
  Ledger,
  outstanding,
  preflightProblems,
  readRun,
  removeOwned,
  removeSuccessor,
  RUN_FORMAT,
  runAttempt,
  runEval,
  type AttemptRecord,
  type RunContext,
} from "../eval/compaction/run";
import {
  createLocation,
  createRoot,
  locationProblem,
  MARKER,
  removeOwnedTree,
  rootProblem,
  type Owner,
} from "../eval/compaction/scratch";
import {
  applySummarizerVariant,
  createControl,
  evalOptions,
  observeCommand,
  subjectStorage,
  type SessionFacts,
} from "../eval/compaction/wrapper/core";
import { compactSuccess } from "../src/commands";
import { OwnedSessions } from "../src/v2/owned";
import { executeV2Compact } from "../src/v2/compact";
import type { V2Message } from "../src/v2/session";
import {
  echoGeneration,
  fakeV2Host,
  fakeV2State,
  sourceSession,
  type FakeV2State,
} from "./v2-host";

const corpus = loadCorpus();
const caseA = corpus.cases.find(c => c.id === "A")!;
const caseB = corpus.cases.find(c => c.id === "B")!;
const NONCE = "0123456789abcdef0123456789abcdef";
// Host records in the in-memory hosts are untyped JSON.
type Json = ReturnType<typeof JSON.parse>;

function scratch(): string {
  return realpathSync(mkdtempSync(join(tmpdir(), "compaction-eval-test-")));
}

// Plugin storage of the subject goes to a private data home in every test.
const originalDataHome = process.env.XDG_DATA_HOME;
let dataHome: string;
beforeEach(() => {
  dataHome = scratch();
  process.env.XDG_DATA_HOME = dataHome;
});
afterEach(() => {
  if (originalDataHome === undefined) delete process.env.XDG_DATA_HOME;
  else process.env.XDG_DATA_HOME = originalDataHome;
  rmSync(dataHome, { recursive: true, force: true });
});

describe("frozen corpus", () => {
  test("loads version 1 in its frozen order with its bound images", () => {
    expect(corpus.corpusId).toBe("magic-compact-synthetic/1");
    expect(corpus.cases.map(c => c.id)).toEqual(["A", "B", "C"]);
    expect(caseA.image).toBeNull();
    expect(sha256(Buffer.from(caseB.image!.base64, "base64"))).toBe(
      "444be5eb67688e27da659f81e0048691f4ae563472383cb35e45598501447048",
    );
    expect(corpus.corpusSha256).toBe(
      sha256(readFileSync(join(CORPUS_DIR, "manifest.json"))),
    );
  });

  test("every case plans one target, one preserved turn and no retention candidate on both hosts", async () => {
    for (const plan of await checkCompactionPlans(corpus)) {
      expect(plan.v1).toEqual({ targets: 1, preserved: 1, candidates: 0 });
      expect(plan.v2).toEqual({ targets: 1, preserved: 1, candidates: 0 });
    }
  });

  test("refuses a changed file, an unknown format and an extra field", () => {
    const cases: Array<[string, (dir: string) => void, RegExp]> = [
      [
        "changed gold",
        dir => {
          const path = join(dir, "gold.json");
          writeFileSync(
            path,
            readFileSync(path, "utf8").replace("40 to 80", "40 to 90"),
          );
        },
        /gold\.json: SHA-256 differs/u,
      ],
      [
        "unknown manifest format",
        dir => {
          const path = join(dir, "manifest.json");
          const manifest = JSON.parse(readFileSync(path, "utf8"));
          manifest.format = "arkive-compaction-eval-manifest/2";
          writeFileSync(path, JSON.stringify(manifest));
        },
        /manifest\.json: unsupported format/u,
      ],
      [
        "extra manifest field",
        dir => {
          const path = join(dir, "manifest.json");
          const manifest = JSON.parse(readFileSync(path, "utf8"));
          manifest.module = "cases.mjs";
          writeFileSync(path, JSON.stringify(manifest));
        },
        /manifest\.json: unsupported format/u,
      ],
    ];
    for (const [, mutate, error] of cases) {
      const dir = scratch();
      try {
        cpSync(CORPUS_DIR, dir, { recursive: true });
        mutate(dir);
        expect(() => loadCorpus(dir)).toThrow(CorpusError);
        expect(() => loadCorpus(dir)).toThrow(error);
      } finally {
        rmSync(dir, { recursive: true, force: true });
      }
    }
  });
});

const respond =
  (
    routes: Record<string, { status?: number; type?: string; body: string }>,
  ): FetchLike =>
  async input => {
    const path = new URL(String(input)).pathname;
    const route = routes[path] ?? { status: 404, body: "" };
    return new Response(route.body, {
      status: route.status ?? 200,
      headers: { "content-type": route.type ?? "application/json" },
    });
  };
const html = { type: "text/html", body: "<!doctype html><html></html>" };
const v1Server = respond({
  "/global/health": { body: '{"healthy":true,"version":"1.18.30"}' },
  "/api/info": html,
});
const v2Server = (pid: number) =>
  respond({
    "/global/health": html,
    "/api/info": { body: JSON.stringify({ version: "2.0.28", pid }) },
  });
// The server record a run header carries.
const SERVER_URL = "http://127.0.0.1:1";
const serverRecord = (
  kind: "v1" | "v2" = "v1",
  pid: number | null = null,
  url = SERVER_URL,
) => ({
  url,
  kind,
  version: kind === "v1" ? "1.18.30" : "2.0.28",
  qualification: "qualified",
  pid,
});

describe("server identity and selection", () => {
  test("a 200 HTML page is not host identity", async () => {
    await expect(
      identifyServer(
        "http://127.0.0.1:1",
        undefined,
        respond({ "/global/health": html, "/api/info": html }),
      ),
    ).rejects.toThrow(/did not identify/u);
  });

  test("OpenCode 1 and 2 identify by their JSON documents and major version", async () => {
    expect(
      await identifyServer("http://127.0.0.1:1", undefined, v1Server),
    ).toEqual({
      url: "http://127.0.0.1:1",
      kind: "v1",
      version: "1.18.30",
      qualification: "qualified",
      pid: null,
    });
    expect(
      await identifyServer(
        "http://127.0.0.1:1",
        undefined,
        respond({
          "/global/health": html,
          "/api/info": { body: '{"version":"2.0.28","pid":7}' },
        }),
      ),
    ).toEqual({
      url: "http://127.0.0.1:1",
      kind: "v2",
      version: "2.0.28",
      qualification: "qualified",
      pid: 7,
    });
  });

  test("an unqualified version refuses unless the operator names it", async () => {
    const fetch = respond({
      "/global/health": html,
      "/api/info": { body: '{"version":"2.1.0","pid":7}' },
    });
    await expect(
      identifyServer("http://127.0.0.1:1", undefined, fetch),
    ).rejects.toThrow(/--assert-host-version 2\.1\.0/u);
    expect(
      await identifyServer("http://127.0.0.1:1", undefined, fetch, "2.1.0"),
    ).toEqual({
      url: "http://127.0.0.1:1",
      kind: "v2",
      version: "2.1.0",
      qualification: "operator-asserted",
      pid: 7,
    });
  });

  test("refused server credentials are reported as such", async () => {
    await expect(
      identifyServer(
        "http://127.0.0.1:1",
        undefined,
        respond({ "/global/health": { status: 401, body: "" } }),
      ),
    ).rejects.toThrow(/refused the eval's credentials/u);
  });

  test("only a bare loopback URL qualifies", () => {
    expect(loopbackServerUrl("http://127.0.0.1:4096/")).toBe(
      "http://127.0.0.1:4096",
    );
    for (const url of [
      "http://10.0.0.1:4096",
      "https://example.com",
      "http://user:pass@127.0.0.1:4096",
      "http://127.0.0.1:4096/api",
      "not a url",
    ])
      expect(() => loopbackServerUrl(url)).toThrow(EvalError);
  });

  test("model selectors need provider/model with an optional variant", () => {
    expect(
      parseModelSelector("anthropic/claude-haiku-5-5#low", "--model"),
    ).toEqual({
      selector: "anthropic/claude-haiku-5-5#low",
      providerID: "anthropic",
      modelID: "claude-haiku-5-5",
      variant: "low",
    });
    expect(
      parseModelSelector("openrouter/vendor/model", "--model").modelID,
    ).toBe("vendor/model");
    for (const raw of ["haiku", "a/b#", "a/b#c#d", "/b"])
      expect(() => parseModelSelector(raw, "--model")).toThrow(EvalError);
  });
});

describe("preflight", () => {
  const model = parseModelSelector("p/m#low", "--model");
  const setup: EffectiveSetup = {
    plugins: ["auth@1", "arkive-compact-eval@local:/cache/key/v2/index.ts"],
    instructionFingerprints: [],
    mcpDisabled: ["inherited"],
    summarizer: { model: "p/m", variant: "low" },
    subjectSummarizer: { model: "p/m", variant: "low" },
    commands: [
      "magic-compact",
      "magic-trim",
      "magic-stats",
      "arkive-eval-control",
      "arkive-eval-witness",
    ],
  };
  const good = {
    runNonce: "n",
    artifactKey: "k",
    setup,
    witness: {
      runNonce: "n",
      artifactKey: "k",
      subjectTgzSha256: "s",
      subjectManifestSha256: "m",
      subjectMismatches: 0,
      pid: 1,
      initCount: 1,
    },
    inits: 1,
    model,
    descriptor: {
      providerID: "p",
      modelID: "m",
      variant: "low",
      name: null,
      package: null,
      variants: ["low"],
      imageInput: true,
      limit: {},
    },
    mcp: ["inherited"],
  };

  test("passes only when every condition holds", () => {
    expect(preflightProblems(good)).toEqual([]);
  });

  test("refuses a duplicate Arkive plugin, a foreign witness and double initialization", () => {
    expect(
      preflightProblems({
        ...good,
        setup: {
          ...setup,
          plugins: [...setup.plugins, "arkive-compact@1.1.1"],
        },
      }),
    ).toEqual([expect.stringMatching(/exactly one Arkive plugin.*found 2/u)]);
    expect(
      preflightProblems({
        ...good,
        witness: { ...good.witness, runNonce: "other" },
      }),
    ).toEqual(["the eval wrapper did not load in the eval project"]);
    expect(
      preflightProblems({
        ...good,
        witness: { ...good.witness, artifactKey: "x", subjectMismatches: 2 },
      }),
    ).toHaveLength(2);
    expect(preflightProblems({ ...good, inits: 2 })).toEqual([
      expect.stringMatching(/saw 2/u),
    ]);
  });

  test("refuses a missing variant or model, an enabled MCP server and a summarizer the plugin cannot see", () => {
    expect(
      preflightProblems({
        ...good,
        descriptor: { ...good.descriptor, variants: ["high"] },
      }),
    ).toEqual([expect.stringMatching(/no variant low/u)]);
    expect(preflightProblems({ ...good, descriptor: null })).toEqual([
      expect.stringMatching(/no model p\/m/u),
    ]);
    expect(
      preflightProblems({ ...good, setup: { ...setup, mcpDisabled: [] } }),
    ).toEqual(["MCP servers not disabled in the eval project: inherited"]);
    expect(
      preflightProblems({
        ...good,
        setup: { ...setup, subjectSummarizer: { model: null, variant: null } },
      }),
    ).toEqual([expect.stringMatching(/own summarizer lookup/u)]);
  });

  test("a bare model refuses an inherited variant and passes once it resolves to none", () => {
    const bare = parseModelSelector("p/m", "--model");
    const inherited = {
      ...good,
      model: bare,
      setup: {
        ...setup,
        summarizer: { model: "p/m", variant: null },
        subjectSummarizer: { model: "p/m", variant: "low" },
      },
    };
    expect(preflightProblems(inherited)).toEqual([
      expect.stringMatching(/own summarizer lookup returned p\/m#low/u),
    ]);
    expect(
      preflightProblems({
        ...inherited,
        setup: {
          ...inherited.setup,
          subjectSummarizer: { model: "p/m", variant: null },
        },
      }),
    ).toEqual([]);
  });
});

describe("eval wrapper", () => {
  test("sets the summarizer variant to exactly the run's request in memory", () => {
    const options = (variant: string | null) =>
      evalOptions({
        evalRunNonce: NONCE,
        evalChannel: "/channel",
        evalSummarizerVariant: variant,
      });
    // A global profile's variant, merged under the eval project's bare model.
    const agents: Record<string, Record<string, unknown>> = {
      "magic-compact-summarizer": { model: "p/m", variant: "low" },
      build: { variant: "high" },
    };
    applySummarizerVariant(agents, options(null));
    expect(agents).toEqual({
      "magic-compact-summarizer": { model: "p/m" },
      build: { variant: "high" },
    });
    applySummarizerVariant(agents, options("high"));
    expect(agents["magic-compact-summarizer"]).toEqual({
      model: "p/m",
      variant: "high",
    });
    expect(() =>
      evalOptions({ evalRunNonce: NONCE, evalChannel: "/channel" }),
    ).toThrow(/invalid eval run options/u);
  });

  // A successor and its source in the wrapper's own location, with real
  // cache and stats files written through the subject's storage functions.
  async function controlFixture(facts: Record<string, Partial<SessionFacts>>) {
    const channel = scratch();
    const options = evalOptions({
      evalRunNonce: NONCE,
      evalChannel: channel,
      evalSummarizerVariant: null,
    });
    const storage = await subjectStorage("v2", PACKAGE_DIR);
    const sessions: Record<string, SessionFacts> = {
      ses_succ: {
        directory: "/eval/project",
        title: "Source",
        metadata: { magicCompact: { successorOf: "ses_src", ready: true } },
      },
      ses_src: {
        directory: "/eval/project",
        title: `arkive compaction eval ${NONCE} r1-A`,
        metadata: {},
      },
    };
    for (const [id, patch] of Object.entries(facts))
      sessions[id] = { ...sessions[id]!, ...patch };
    const paths = (id: string) => storage.paths(storage.key(id));
    for (const id of ["ses_succ", "ses_other"]) {
      const { cache, stats } = paths(id);
      mkdirSync(join(cache, ".."), { recursive: true });
      mkdirSync(join(stats, ".."), { recursive: true });
      writeFileSync(cache, "{}");
      writeFileSync(stats, "{}");
    }
    const control = createControl(
      options,
      "/eval/project",
      async id => sessions[id],
      async () => storage,
    );
    const lines = () =>
      readdirSync(channel).flatMap(name =>
        readFileSync(join(channel, name), "utf8")
          .split("\n")
          .filter(Boolean)
          .map(line => JSON.parse(line) as Record<string, unknown>),
      );
    const present = (id: string) =>
      [paths(id).cache, paths(id).stats].map(existsSync);
    return { control, lines, present };
  }
  const request = (patch: Record<string, unknown> = {}) =>
    JSON.stringify({
      runNonce: NONCE,
      requestId: "a".repeat(24),
      sourceID: "ses_src",
      ...patch,
    });

  test("removes a receipted successor's storage only after the host reports its deletion", async () => {
    const { control, lines, present } = await controlFixture({});
    expect(await control.prepare("ses_succ", request())).toMatch(/prepared/u);
    // Preparing writes nothing.
    expect(present("ses_succ")).toEqual([true, true]);
    await control.deleted("ses_other");
    await control.deleted("ses_succ");
    expect(present("ses_succ")).toEqual([false, false]);
    expect(present("ses_other")).toEqual([true, true]);
    expect(lines().map(l => [l["phase"], l["removed"]])).toEqual([
      ["prepared", undefined],
      ["storage", true],
    ]);
  });

  test("refuses a successor it cannot prove is this run's and then never removes its storage", async () => {
    const refusals: Array<[Record<string, Partial<SessionFacts>>, string]> = [
      [{}, request({ runNonce: "f".repeat(32) })],
      [{ ses_succ: { directory: "/elsewhere" } }, request()],
      [{ ses_src: { title: "a user's session" } }, request()],
      [
        {
          ses_succ: {
            metadata: { magicCompact: { successorOf: "ses_x", ready: true } },
          },
        },
        request(),
      ],
      [
        {
          ses_succ: {
            metadata: {
              magicCompact: { successorOf: "ses_src", ready: false },
            },
          },
        },
        request(),
      ],
    ];
    for (const [facts, args] of refusals) {
      const { control, lines, present } = await controlFixture(facts);
      expect(await control.prepare("ses_succ", args)).toMatch(/refused/u);
      await control.deleted("ses_succ");
      expect(present("ses_succ")).toEqual([true, true]);
      expect(lines().map(l => l["phase"])).toEqual(["refused"]);
    }
  });
});

describe("artifact cache", () => {
  function standInSubject(dir: string): string {
    const pkg = join(dir, "subject");
    mkdirSync(join(pkg, "src"), { recursive: true });
    writeFileSync(
      join(pkg, "package.json"),
      JSON.stringify({
        name: "arkive-compact",
        version: "9.9.9",
        main: "src/index.ts",
      }),
    );
    writeFileSync(
      join(pkg, "src", "index.ts"),
      "export default { id: 'arkive-compact' };\n",
    );
    const name = execFileSync(
      "npm",
      ["pack", "--ignore-scripts", "--silent", "--pack-destination", dir],
      {
        cwd: pkg,
        encoding: "utf8",
      },
    ).trim();
    return join(dir, name);
  }

  test("reuses a verified content-addressed entry and refuses a tampered one", () => {
    const dir = scratch();
    try {
      const tgz = standInSubject(dir);
      const cache = join(dir, "cache");
      const first = prepareArtifact(tgz, cache, {
        ...process.env,
        npm_config_offline: "true",
      });
      expect(first.reused).toBe(false);
      expect(first.dir).toBe(join(cache, first.key));
      expect(first.subjectFiles).toBe(2);
      const second = prepareArtifact(tgz, cache);
      expect(second).toEqual({ ...first, reused: true });
      // The run never reaches the wrapper modules through a run-specific path.
      expect(
        readFileSync(join(first.v2Dir, "witness.json"), "utf8"),
      ).not.toMatch(/runNonce/u);
      writeFileSync(join(first.v2Dir, "core.ts"), "// tampered\n");
      expect(() => prepareArtifact(tgz, cache)).toThrow(ArtifactCacheError);
      expect(readFileSync(join(first.v2Dir, "core.ts"), "utf8")).toBe(
        "// tampered\n",
      );
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  }, 120_000);
});

describe("command outcomes", () => {
  test("classifies the exact text a command threw", () => {
    expect(
      classifyOutcome(
        "Magic compaction successful. Continue in successor session ses_abc123; the original session was left unchanged.\nStored history (estimate) ...",
      ),
    ).toMatchObject({ kind: "delivered", successorID: "ses_abc123" });
    expect(
      classifyOutcome("No assistant turns are old enough to compact.").kind,
    ).toBe("noop");
    expect(
      classifyOutcome(
        "Magic Compact needs a model on the `magic-compact-summarizer` agent. Set `agents.magic-compact-summarizer.model` (for example `provider/model#variant`) in opencode.json.",
      ),
    ).toMatchObject({ kind: "failed", category: "setup" });
    expect(
      classifyOutcome(
        "The compaction model finished without submitting its StructuredOutput response.",
      ),
    ).toMatchObject({ kind: "failed", category: "schema" });
  });
});

// A real OpenCode 2 successor: the production command run against the
// in-memory OpenCode 2 host on a corpus case.
async function v2Compacted(corpusCase = caseB) {
  const state = fakeV2State();
  let n = 0;
  const records = v2SourceHistory(
    corpus,
    corpusCase,
    { providerID: "probe", modelID: "probe-model" },
    () => `msg_src_${String(n++).padStart(4, "0")}`,
  );
  sourceSession(state, "ses_src", records, {
    title: `arkive compaction eval ${NONCE} r1-${corpusCase.id}`,
  });
  const owned = new OwnedSessions();
  const host = fakeV2Host(state, owned, echoGeneration());
  const { successor } = await executeV2Compact(host, owned, "ses_src", 1, {
    retention: { mode: "heuristic" },
    controlTimeoutMs: 1_000,
  });
  const id = successor!.session.id;
  return {
    sourceSession: state.sessions.get("ses_src") as never as Record<
      string,
      unknown
    >,
    source: state.messages.get("ses_src") as never as Array<
      Record<string, unknown>
    >,
    successorSession: state.sessions.get(id) as never as Record<
      string,
      unknown
    >,
    successor: state.messages.get(id) as never as Array<
      Record<string, unknown>
    >,
  };
}

describe("mechanical oracle", () => {
  const failing = (inspection: {
    checks: Array<{ name: string; ok: boolean }>;
  }) => inspection.checks.filter(c => !c.ok).map(c => c.name);

  // Native records of a case B source and its successor, read through the
  // eval's drivers from the packed plugin on OpenCode 1.18.30 and 2.0.28 in
  // network-less containers with a scripted provider.
  type Native = {
    host: "v1" | "v2";
    caseId: string;
    source: { session: Json; history: Json[] };
    successor: { session: Json; history: Json[] };
  };
  const native = (host: "v1" | "v2"): Native =>
    JSON.parse(
      readFileSync(
        join(
          PACKAGE_DIR,
          "eval",
          "compaction",
          "testdata",
          `native-successor-${host}.json`,
        ),
        "utf8",
      ),
    );
  const inspectNative = (f: Native) =>
    (f.host === "v1" ? v1SuccessorInspection : v2SuccessorInspection)({
      sourceSession: f.source.session,
      source: f.source.history,
      successorSession: f.successor.session,
      successor: f.successor.history,
      corpusCase: corpus.cases.find(c => c.id === f.caseId)!,
    } as never);

  test("native successors from both hosts pass every check", () => {
    for (const host of ["v1", "v2"] as const) {
      const inspection = inspectNative(native(host));
      expect(failing(inspection)).toEqual([]);
      expect(inspection.checks.map(c => c.name)).toContain(
        "image manifest binds the one source PNG",
      );
    }
  });

  test("a changed native OpenCode 1 file name, part metadata, protected-turn field or reused part ID fails a named check", () => {
    const mutations: Array<[(s: Json[]) => void, RegExp]> = [
      [s => void (s[0].parts[1].filename = "other.png"), /user messages/u],
      [s => void (s[0].parts[0].metadata = { origin: "x" }), /user messages/u],
      [s => void (s[2].parts[1].synthetic = true), /user messages/u],
      [s => void (s[3].info.finish = "length"), /protected follow-up/u],
      [s => void (s[3].parts[0].metadata = { note: 1 }), /protected/u],
      [s => void (s[1].info.mode = "plan"), /summarized turn/u],
      [s => void (s[1].parts[0].metadata.magicCompact.version = 2), /v3/u],
      [
        (s, f = native("v1")) =>
          void (s[3].parts[0].id = f.source.history[3].parts[0].id),
        /fresh/u,
      ],
      [s => void s[4].parts.push({ ...s[4].parts[0], id: "prt_x" }), /stats/u],
    ];
    for (const [index, [mutate, check]] of mutations.entries()) {
      const f = native("v1");
      mutate(f.successor.history);
      const failed = failing(inspectNative(f));
      expect([index, failed.some(name => check.test(name))]).toEqual([
        index,
        true,
      ]);
    }
  });

  test("an OpenCode 2 successor from the production command passes every check", async () => {
    const records = await v2Compacted();
    const inspection = v2SuccessorInspection({ ...records, corpusCase: caseB });
    expect(failing(inspection)).toEqual([]);
    expect(inspection.capsule).toStartWith("Summary of target 1.");
    expect(inspection.checks.map(c => c.name)).toContain(
      "image manifest binds the one source PNG",
    );
  });

  test("a changed file name, user metadata, protected-turn metadata or reused ID fails a named check", async () => {
    const mutations: Array<
      [string, (s: Array<Record<string, Json>>) => void, RegExp]
    > = [
      [
        "image file name",
        s => {
          s[0]!["files"][0]!.name = "other.png";
        },
        /user messages preserved/u,
      ],
      [
        "user metadata",
        s => {
          s[3]!["metadata"] = { origin: "edited" };
        },
        /user messages preserved/u,
      ],
      [
        "protected assistant metadata",
        s => {
          s[4]!["metadata"] = { note: "added" };
        },
        /protected follow-up/u,
      ],
      [
        "protected assistant finish",
        s => {
          s[4]!["finish"] = "length";
        },
        /protected follow-up/u,
      ],
      ["reused source ID", s => void (s[3]!["id"] = "msg_src_0002"), /fresh/u],
      [
        "extra record",
        s => void s.push({ id: "msg_x", type: "user", text: "foreign" }),
        /planned records/u,
      ],
    ];
    for (const [, mutate, check] of mutations) {
      const records = await v2Compacted();
      mutate(records.successor as Array<Record<string, Json>>);
      const failed = failing(
        v2SuccessorInspection({ ...records, corpusCase: caseB }),
      );
      expect(failed.some(name => check.test(name))).toBe(true);
    }
  });
});

describe("ledger", () => {
  const entry = (
    seq: number,
    kind: string,
    fields: Record<string, unknown>,
  ) => ({ seq, at: "", kind, ...fields });

  test("an intent stays open until its own creation or a resolution, with or without a planned ID", () => {
    const open = outstanding([
      entry(1, "session-intent", {
        attemptId: "r1-A",
        role: "source",
        plannedID: null,
      }),
      entry(2, "session-created", {
        attemptId: "r1-A",
        role: "source",
        id: "ses_s",
        intentSeq: 1,
      }),
      entry(3, "session-intent", {
        attemptId: "r1-B",
        role: "template",
        plannedID: "ses_t",
      }),
      // A later creation for the same role does not settle an earlier intent.
      entry(4, "session-intent", {
        attemptId: "r1-B",
        role: "template",
        plannedID: "ses_u",
      }),
      entry(5, "session-created", {
        attemptId: "r1-B",
        role: "template",
        id: "ses_u",
        intentSeq: 4,
      }),
      entry(6, "session-intent", {
        attemptId: "r1-C",
        role: "source",
        plannedID: null,
      }),
      entry(7, "session-intent-resolved", { intentSeq: 6, outcome: "absent" }),
    ]);
    expect(open.intents.map(e => e.seq)).toEqual([3]);
    expect(open.sessions.map(e => e["id"])).toEqual(["ses_s", "ses_u"]);
  });

  test("a started command without a settled outcome stays unknown, even with no unknown marker after a crash", () => {
    const crashed = outstanding([
      entry(1, "command-start", { attemptId: "r1-A", sourceID: "ses_s" }),
    ]);
    expect(crashed.attempts.map(e => e["attemptId"])).toEqual(["r1-A"]);
    const settled = outstanding([
      entry(1, "command-start", { attemptId: "r1-A", sourceID: "ses_s" }),
      entry(2, "attempt-unknown", { attemptId: "r1-A" }),
      entry(3, "attempt-settled", { attemptId: "r1-A", status: "failed" }),
    ]);
    expect(settled.attempts).toEqual([]);
  });
});

// --- removal ------------------------------------------------------------------

type Pending = Awaited<ReturnType<EvalHost["pending"]>>;
// The process ID the stand-in server reports and the run recorded.
const PID = 4242;
function stubHost(fields: Partial<EvalHost>): EvalHost {
  return { kind: "v1", serverPid: async () => PID, ...fields } as EvalHost;
}
function stubContext(host: EvalHost, extra: Partial<RunContext> = {}) {
  const dir = scratch();
  mkdirSync(join(dir, "channel"));
  return {
    dir,
    ctx: {
      host,
      ledger: new Ledger(join(dir, "ledger.jsonl")),
      runNonce: NONCE,
      channel: join(dir, "channel"),
      project: "/eval/project",
      corpus,
      serverPid: PID,
      log: () => {},
      lostOutcomeWaitMs: 0,
      ...extra,
    } as unknown as RunContext,
  };
}
const stateAt = (directory: string, history: unknown[] = []) =>
  stateOf(
    { id: "ses_x", directory },
    { id: "ses_x" },
    history as never,
    directory,
  );

describe("removal", () => {
  test("removes only an unchanged session at its own location with no pending work", async () => {
    const recorded = stateAt("/eval");
    const scenarios: Array<{
      pending: Pending;
      now?: SessionState;
      expected: ExpectedState | null;
      action: "removed" | "absent" | "retained";
      reason?: unknown;
    }> = [
      {
        pending: { exists: true, reasons: [] },
        expected: recorded,
        action: "removed",
      },
      { pending: { exists: false }, expected: recorded, action: "absent" },
      {
        pending: { exists: true, reasons: ["a question is waiting"] },
        expected: recorded,
        action: "retained",
        reason: expect.stringMatching(/question/u),
      },
      {
        pending: { exists: true, reasons: [] },
        now: stateAt("/eval", [{ foreign: true }]),
        expected: recorded,
        action: "retained",
        reason: expect.stringMatching(/changed/u),
      },
      {
        pending: { exists: true, reasons: [] },
        now: stateAt("/elsewhere"),
        expected: recorded,
        action: "retained",
        reason: expect.stringMatching(/outside/u),
      },
      {
        pending: { exists: true, reasons: [] },
        expected: null,
        action: "retained",
        reason: expect.stringMatching(/no recorded state/u),
      },
    ];
    for (const scenario of scenarios) {
      const removed: string[] = [];
      const { dir, ctx } = stubContext(
        stubHost({
          pending: async () => scenario.pending,
          read: async () => scenario.now ?? recorded,
          remove: async (_, id) => void removed.push(id),
        }),
      );
      try {
        const result = await removeOwned(
          ctx,
          "/eval",
          "ses_x",
          scenario.expected,
          "r1-A",
        );
        expect(result.action).toBe(scenario.action);
        if (scenario.reason)
          expect(result).toMatchObject({ reason: scenario.reason });
        expect(removed).toEqual(scenario.action === "removed" ? ["ses_x"] : []);
      } finally {
        rmSync(dir, { recursive: true, force: true });
      }
    }
  });

  test("absence counts only while the server that created the session answers", async () => {
    const cases: Array<[Partial<RunContext>, EvalHost["serverPid"], string]> = [
      [{}, async () => PID, "absent"],
      [{}, async () => PID + 1, "retained"],
      [{}, async () => null, "retained"],
      [
        {},
        async () => {
          throw new Error("config read failed (HTTP 500)");
        },
        "retained",
      ],
      [{ serverPid: null }, async () => PID, "retained"],
    ];
    for (const [extra, serverPid, action] of cases) {
      const { dir, ctx } = stubContext(
        stubHost({
          pending: async () => ({ exists: false }),
          serverPid,
        }),
        extra,
      );
      try {
        const result = await removeOwned(
          ctx,
          "/eval",
          "ses_x",
          stateAt("/eval"),
          null,
        );
        expect(result.action).toBe(action as never);
        expect(
          Ledger.read(ctx.ledger.path).some(e => e.kind === "session-absent"),
        ).toBe(action === "absent");
      } finally {
        rmSync(dir, { recursive: true, force: true });
      }
    }
  });

  test("an unreadable pending fact keeps the session", async () => {
    const removed: string[] = [];
    const { dir, ctx } = stubContext(
      stubHost({
        pending: async () => {
          throw new Error("question list read failed (HTTP 500)");
        },
        remove: async (_, id) => void removed.push(id),
      }),
    );
    try {
      const result = await removeOwned(
        ctx,
        "/eval",
        "ses_x",
        stateAt("/eval"),
        null,
      );
      expect(result).toMatchObject({
        action: "retained",
        reason: expect.stringMatching(/ownership check failed/u),
      });
      expect(removed).toEqual([]);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  // The wrapper's real control and the subject's real storage behind an
  // in-memory host: the command prepares, the delete reports the deletion.
  async function successorRemoval(options: {
    change?: (state: SessionState) => SessionState;
    pending?: Pending;
    failStorage?: boolean;
  }) {
    const recorded = stateAt("/eval/project");
    const storage = await subjectStorage("v1", PACKAGE_DIR);
    const { cache, stats } = storage.paths("ses_x");
    mkdirSync(join(stats, ".."), { recursive: true });
    writeFileSync(cache, "{}");
    writeFileSync(stats, "{}");
    const removed: string[] = [];
    const wrapper: { control?: ReturnType<typeof createControl> } = {};
    const { dir, ctx } = stubContext(
      stubHost({
        runCommand: async (_, id, __, args) => {
          await wrapper.control!.prepare(id, args);
          return { kind: "returned", status: 500, message: null, ms: 1 };
        },
        pending: async () => options.pending ?? { exists: true, reasons: [] },
        read: async () => (options.change ?? (s => s))(recorded),
        remove: async (_, id) => {
          removed.push(id);
          if (!options.failStorage) await wrapper.control!.deleted(id);
        },
      }),
    );
    wrapper.control = createControl(
      evalOptions({
        evalRunNonce: NONCE,
        evalChannel: join(dir, "channel"),
        evalSummarizerVariant: null,
      }),
      "/eval/project",
      async id =>
        id === "ses_x"
          ? {
              directory: "/eval/project",
              title: "Source",
              metadata: {
                magicCompact: { successorOf: "ses_src", ready: true },
              },
            }
          : {
              directory: "/eval/project",
              title: `arkive compaction eval ${NONCE} r1-A`,
              metadata: {},
            },
      async () => storage,
    );
    try {
      const result = await removeSuccessor(
        ctx,
        "ses_x",
        "ses_src",
        recorded,
        "r1-A",
      );
      return { result, removed, present: [cache, stats].map(existsSync) };
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  }

  test("a successor and its storage go only together, after every check passed", async () => {
    const ok = await successorRemoval({});
    expect(ok.result).toMatchObject({ action: "removed", storage: "removed" });
    expect(ok.present).toEqual([false, false]);
    for (const refused of [
      // Another writer appended to the successor.
      {
        change: (s: SessionState) =>
          stateAt("/eval/project", [...s.history, { foreign: 1 }]),
      },
      {
        pending: {
          exists: true,
          reasons: ["session has child sessions"],
        } as Pending,
      },
      {
        pending: {
          exists: true,
          reasons: ["session status is busy"],
        } as Pending,
      },
    ]) {
      const kept = await successorRemoval(refused);
      expect(kept.result.action).toBe("retained");
      expect(kept.removed).toEqual([]);
      expect(kept.present).toEqual([true, true]);
    }
  }, 60_000);

  test("a deleted successor whose storage removal is not confirmed reports the storage left behind", async () => {
    const orphan = await successorRemoval({ failStorage: true });
    expect(orphan.removed).toEqual(["ses_x"]);
    expect(orphan.result).toMatchObject({
      action: "removed",
      storage: { orphaned: { cache: expect.stringContaining("ses_x") } },
    });
    expect(orphan.present).toEqual([true, true]);
  }, 60_000);
});

// --- host drivers -------------------------------------------------------------

type V1Message = {
  info: Record<string, Json>;
  parts: Array<Record<string, Json>>;
};
// An in-memory OpenCode 1 client with just the calls the driver makes.
function fakeV1(directory = "/eval") {
  const state = {
    sessions: new Map<string, Record<string, Json>>(),
    messages: new Map<string, V1Message[]>(),
    status: {} as Record<string, { type: string }>,
    children: {} as Record<string, string[]>,
    permissions: [] as Array<{ sessionID: string }>,
    questions: [] as Array<{ sessionID: string }>,
    unreadable: new Set<string>(),
    deleted: [] as string[],
    reply: '{"facts":[],"falseAssertions":[]}',
    afterPrompt: (() => {}) as (id: string) => void,
    afterReplay: (() => {}) as (id: string) => void,
    // Fields the host adds to a session it creates.
    createdExtra: {} as Record<string, Json>,
  };
  const clone = <T>(value: T): T => JSON.parse(JSON.stringify(value));
  const ok = <T>(data: T) => ({
    data: clone(data),
    response: new Response(null),
  });
  const missing = () => ({
    error: {},
    response: new Response(null, { status: 404 }),
  });
  let n = 0;
  const client = {
    session: {
      get: async ({ sessionID }: { sessionID: string }) =>
        state.sessions.has(sessionID)
          ? ok(state.sessions.get(sessionID))
          : missing(),
      messages: async ({ sessionID }: { sessionID: string }) =>
        state.unreadable.has(sessionID)
          ? {
              error: { name: "UnknownError" },
              response: new Response(null, { status: 500 }),
            }
          : ok(state.messages.get(sessionID) ?? []),
      status: async () => ok(state.status),
      children: async ({ sessionID }: { sessionID: string }) =>
        ok((state.children[sessionID] ?? []).map(id => ({ id }))),
      list: async () => ok([...state.sessions.values()]),
      create: async (body: Record<string, Json>) => {
        const session = {
          id: `ses_${++n}`,
          slug: `slug-${n}`,
          projectID: "project",
          directory: body.directory,
          title: body.title,
          // OpenCode 1.18.30 records the requested agent and model as given.
          agent: body.agent,
          model: body.model,
          version: "1.18.30",
          permission: body.permission,
          time: { created: n, updated: n },
          ...state.createdExtra,
        };
        state.sessions.set(session.id, session);
        state.messages.set(session.id, []);
        return ok(session);
      },
      prompt: async (body: Record<string, Json>) => {
        const sessionID = body.sessionID as string;
        const user: V1Message = {
          info: {
            id: body.messageID,
            sessionID,
            role: "user",
            time: { created: 1 },
            agent: body.agent,
            model: body.model,
          },
          parts: [
            {
              id: "prt_u",
              sessionID,
              messageID: body.messageID,
              type: "text",
              text: body.parts[0].text,
            },
          ],
        };
        const reply: V1Message = {
          info: {
            id: "msg_reply",
            sessionID,
            role: "assistant",
            parentID: body.messageID,
            time: { created: 2, completed: 3 },
            tokens: {},
            cost: 0,
            finish: "stop",
          },
          parts: [
            {
              id: "prt_r",
              sessionID,
              messageID: "msg_reply",
              type: "text",
              text: state.reply,
            },
          ],
        };
        state.messages.get(sessionID)!.push(user, reply);
        // OpenCode 1.18.30 records the prompt's agent and model on the
        // session, with variant "default" when none was requested.
        Object.assign(state.sessions.get(sessionID)!, {
          agent: body.agent,
          model: {
            id: body.model.modelID,
            providerID: body.model.providerID,
            variant: body.variant ?? "default",
          },
        });
        state.afterPrompt(sessionID);
        return ok(reply);
      },
      delete: async ({ sessionID }: { sessionID: string }) => {
        state.deleted.push(sessionID);
        state.sessions.delete(sessionID);
        return ok(true);
      },
    },
    // Message and part events only, as OpenCode 1.18.30 applies them: the
    // session record is not touched.
    sync: {
      replay: async (body: {
        events: Array<{ data: Record<string, Json> }>;
      }) => {
        const sessionID = body.events[0]!.data["sessionID"] as string;
        const history = state.messages.get(sessionID)!;
        for (const { data } of body.events) {
          if (data["info"]) {
            // OpenCode 1.18.30 keeps a user message's variant only inside
            // its model and drops a top-level one.
            const info = clone(data["info"]);
            if (info.role === "user") delete info.variant;
            const found = history.find(m => m.info["id"] === info.id);
            if (found) found.info = info;
            else history.push({ info, parts: [] });
          } else
            history
              .find(m => m.info["id"] === data["part"].messageID)!
              .parts.push(clone(data["part"]));
        }
        state.afterReplay(sessionID);
        return ok({ sessionID });
      },
    },
    permission: { list: async () => ok(state.permissions) },
    question: { list: async () => ok(state.questions) },
  };
  const host = v1Host("http://127.0.0.1:1", undefined, fetch, client as never);
  const add = (id: string, extra: Record<string, unknown> = {}) => {
    state.sessions.set(id, {
      id,
      slug: id,
      projectID: "project",
      directory,
      title: "t",
      version: "1.18.30",
      time: { created: 1, updated: 1 },
      ...extra,
    });
    state.messages.set(id, []);
  };
  return { state, host, add };
}

describe("OpenCode 1 driver", () => {
  const judge = parseModelSelector("p/judge", "--judge");
  const ledger = { intent() {}, created() {}, removed() {} };

  test("every pending kind keeps a session; an idle one has none", async () => {
    const { state, host, add } = fakeV1();
    add("ses_x");
    expect(await host.pending("/eval", "ses_x")).toEqual({
      exists: true,
      reasons: [],
    });
    expect(await host.pending("/eval", "ses_gone")).toEqual({ exists: false });
    state.status["ses_x"] = { type: "busy" };
    state.children["ses_x"] = ["ses_child"];
    state.permissions.push({ sessionID: "ses_x" });
    state.questions.push({ sessionID: "ses_x" });
    state.sessions.get("ses_x")!["time"].compacting = 5;
    const pending = await host.pending("/eval", "ses_x");
    expect(pending.exists && pending.reasons).toEqual([
      "session status is busy",
      "a native compaction is running",
      "session has child sessions",
      "a permission request is waiting",
      "a question is waiting",
    ]);
  });

  test("a model-only, archive or share change alters the recorded header; timing does not", async () => {
    const { state, host, add } = fakeV1();
    add("ses_x");
    const before = await host.read("/eval", "ses_x");
    const session = state.sessions.get("ses_x")!;
    session["time"].updated = 99;
    session["cost"] = 5;
    expect((await host.read("/eval", "ses_x")).headerSha256).toBe(
      before.headerSha256,
    );
    for (const change of [
      (s: Record<string, Json>) =>
        (s["model"] = { id: "other", providerID: "p" }),
      (s: Record<string, Json>) => (s["time"].archived = 7),
      (s: Record<string, Json>) => (s["share"] = { url: "https://x" }),
      (s: Record<string, Json>) => (s["agent"] = "plan"),
    ]) {
      const current = state.sessions.get("ses_x")!;
      const copy = structuredClone(current);
      change(current);
      expect((await host.read("/eval", "ses_x")).headerSha256).not.toBe(
        before.headerSha256,
      );
      state.sessions.set("ses_x", copy);
    }
  });

  test("a source passes only with the requested header and history it was created and seeded with", async () => {
    const source = (fake: ReturnType<typeof fakeV1>, variant?: string) =>
      fake.host.createSource({
        directory: "/eval",
        title: "source",
        corpus,
        corpusCase: caseA,
        model: parseModelSelector(
          `p/m${variant ? `#${variant}` : ""}`,
          "--model",
        ),
        ledger,
      });
    const failed = (checks: Array<{ name: string; ok: boolean }>) =>
      checks.filter(check => !check.ok).map(check => check.name);
    for (const variant of [undefined, "high"])
      expect(failed((await source(fakeV1(), variant)).checks)).toEqual([]);

    // Another writer changes the header between creation and the first read.
    for (const change of [
      (s: Record<string, Json>) => (s["metadata"] = { foreign: true }),
      (s: Record<string, Json>) =>
        (s["model"] = { id: "other", providerID: "p" }),
      (s: Record<string, Json>) =>
        (s["permission"] = [{ permission: "*", pattern: "*", action: "deny" }]),
      (s: Record<string, Json>) => (s["title"] = "foreign"),
    ]) {
      const fake = fakeV1();
      fake.state.afterReplay = id => change(fake.state.sessions.get(id)!);
      expect(failed((await source(fake)).checks)).toEqual([
        "source header equals the header the host created",
      ]);
    }

    // The host's own answer to the creation already differs from the request.
    for (const extra of [
      { metadata: { foreign: true } },
      { model: { id: "m", providerID: "p", variant: "default" } },
      { title: "server title" },
    ]) {
      const fake = fakeV1();
      fake.state.createdExtra = extra;
      expect(failed((await source(fake)).checks)).toEqual([
        "the host created the source with the requested fields",
      ]);
    }
  });

  test("a judge session is owned only as its own prompt plus the returned reply, and a later foreign append keeps it", async () => {
    const { state, host } = fakeV1();
    const answer = await host.judge({
      directory: "/eval",
      judge,
      title: "judge",
      prompt: "packet",
      ledger,
    });
    expect(answer.ownership).toBeNull();
    expect(answer.rawText).toBe(state.reply);
    const { dir, ctx } = stubContext(host);
    try {
      // A foreign client appends a complete exchange after the reply.
      state.messages.get(answer.sessionID!)!.push({
        info: { id: "msg_f", role: "user" },
        parts: [{ id: "prt_f", type: "text", text: "foreign" }],
      });
      const result = await removeOwned(
        ctx,
        "/eval",
        answer.sessionID!,
        answer.expected,
        null,
      );
      expect(result).toMatchObject({
        action: "retained",
        reason: expect.stringMatching(/changed/u),
      });
      expect(state.deleted).toEqual([]);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  test("a judge session with a foreign message in its context, or an unreadable history, is never removed", async () => {
    const foreign = fakeV1();
    foreign.state.afterPrompt = id =>
      void foreign.state.messages.get(id)!.unshift({
        info: { id: "msg_0", role: "user" },
        parts: [],
      });
    const first = await foreign.host.judge({
      directory: "/eval",
      judge,
      title: "judge",
      prompt: "packet",
      ledger,
    });
    expect(first.expected).toBeNull();
    expect(first.ownership).toMatch(/not exactly the run's prompt/u);

    const unreadable = fakeV1();
    unreadable.state.afterPrompt = id =>
      void unreadable.state.unreadable.add(id);
    const second = await unreadable.host.judge({
      directory: "/eval",
      judge,
      title: "judge",
      prompt: "packet",
      ledger,
    });
    expect(second.expected).toBeNull();
    expect(second.ownership).toMatch(/could not be read/u);
    const { dir, ctx } = stubContext(unreadable.host);
    try {
      expect(
        (
          await removeOwned(
            ctx,
            "/eval",
            second.sessionID!,
            second.expected,
            null,
          )
        ).action,
      ).toBe("retained");
      expect(unreadable.state.deleted).toEqual([]);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

describe("OpenCode 2 driver", () => {
  function v2(extra: Record<string, unknown> = {}) {
    const state = fakeV2State();
    const base = fakeV2Host(state, new OwnedSessions()) as unknown as Record<
      string,
      Json
    >;
    const client = {
      ...base,
      session: {
        ...base["session"],
        create: async (input: Record<string, Json>) => {
          const session = {
            id: input["id"],
            projectID: "project",
            agent: input["agent"],
            model: input["model"],
            title: input["title"],
            permissions: input["permissions"],
            location: input["location"],
            cost: 0,
            tokens: {},
            time: { created: 1, updated: 1 },
          };
          state.sessions.set(session.id, session as never);
          state.messages.set(session.id, []);
          return structuredClone(session);
        },
        generate: async () => ({ text: '{"facts":[],"falseAssertions":[]}' }),
      },
      ...extra,
    };
    return {
      state,
      client,
      host: v2Host("http://127.0.0.1:1", undefined, fetch, client as never),
    };
  }

  test("every pending kind keeps a session; an idle one has none", async () => {
    const { state, host } = v2();
    sourceSession(state, "ses_x", []);
    expect(await host.pending("/workspace", "ses_x")).toEqual({
      exists: true,
      reasons: [],
    });
    expect(await host.pending("/workspace", "ses_gone")).toEqual({
      exists: false,
    });
    state.active["ses_x"] = { type: "running" };
    sourceSession(state, "ses_child", [], { parentID: "ses_x" });
    state.inbox.set("ses_x", [{}]);
    state.permissions.set("ses_x", [{}]);
    state.forms.set("ses_x", [{}]);
    state.shells.push({
      id: "sh",
      status: "running",
      metadata: { sessionID: "ses_x" },
    });
    const pending = await host.pending("/workspace", "ses_x");
    expect(pending.exists && pending.reasons).toEqual([
      "session is active",
      "session has child sessions",
      "a queued message is waiting",
      "a permission request is waiting",
      "a question is waiting",
      "a shell started from the session is running",
    ]);
  });

  test("a judge session must still be exactly as created; a foreign append keeps it", async () => {
    const { state, host } = v2();
    const answer = await host.judge({
      directory: "/workspace",
      judge: parseModelSelector("p/judge", "--judge"),
      title: "judge",
      prompt: "packet",
      ledger: { intent() {}, created() {}, removed() {} },
    });
    const created = state.sessions.get(answer.sessionID!)!;
    expect(answer.expected).toEqual(createdState(created as never));
    const { dir, ctx } = stubContext(host);
    try {
      state.messages
        .get(answer.sessionID!)!
        .push({ id: "msg_f", type: "user", text: "foreign" } as V2Message);
      expect(
        await removeOwned(
          ctx,
          "/workspace",
          answer.sessionID!,
          answer.expected,
          null,
        ),
      ).toMatchObject({
        action: "retained",
        reason: expect.stringMatching(/changed/u),
      });
      state.messages.set(answer.sessionID!, []);
      expect(
        (
          await removeOwned(
            ctx,
            "/workspace",
            answer.sessionID!,
            answer.expected,
            null,
          )
        ).action,
      ).toBe("removed");
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  test("an instruction entry put after creation, or an unreadable entry list, keeps the session", async () => {
    const { state, host, client } = v2();
    const answer = await host.judge({
      directory: "/workspace",
      judge: parseModelSelector("p/judge", "--judge"),
      title: "judge",
      prompt: "packet",
      ledger: { intent() {}, created() {}, removed() {} },
    });
    const id = answer.sessionID!;
    expect((await host.read("/workspace", id)).instructions).toEqual([]);
    const { dir, ctx } = stubContext(host);
    try {
      state.instructions.set(id, [{ key: "foreign", value: "x" }]);
      expect(
        await removeOwned(ctx, "/workspace", id, answer.expected, null),
      ).toMatchObject({
        action: "retained",
        reason: expect.stringMatching(/changed/u),
      });
      state.instructions.delete(id);
      // The same session through a client whose entry list fails: never
      // read as no entries.
      const brokenCtx = stubContext(
        v2Host("http://127.0.0.1:1", undefined, fetch, {
          ...client,
          session: {
            ...client["session"],
            instructions: {
              entry: {
                list: async () => {
                  throw Object.assign(new Error("entry list"), {
                    _tag: "UnknownError",
                  });
                },
              },
            },
          },
        } as never),
      );
      try {
        expect(
          await removeOwned(
            brokenCtx.ctx,
            "/workspace",
            id,
            answer.expected,
            null,
          ),
        ).toMatchObject({
          action: "retained",
          reason: expect.stringMatching(/ownership check failed/u),
        });
        expect(state.sessions.has(id)).toBe(true);
      } finally {
        rmSync(brokenCtx.dir, { recursive: true, force: true });
      }
      expect(
        (await removeOwned(ctx, "/workspace", id, answer.expected, null))
          .action,
      ).toBe("removed");
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

// --- scratch ownership --------------------------------------------------------

describe("scratch ownership", () => {
  let out: string;
  let owner: Owner;
  const made: string[] = [];
  beforeEach(() => {
    out = scratch();
    owner = { runNonce: NONCE, out };
  });
  afterEach(() => {
    rmSync(out, { recursive: true, force: true });
    for (const dir of made.splice(0))
      rmSync(dir, { recursive: true, force: true });
  });
  const root = () => {
    const dir = createRoot(owner, "run");
    made.push(dir);
    return dir;
  };

  test("removes an owned tree, including an empty owned root", () => {
    const empty = root();
    expect(removeOwnedTree(empty, [], owner)).toBeNull();
    expect(existsSync(empty)).toBe(false);
    const full = root();
    const project = createLocation(full, "project", owner, {
      file: "opencode.json",
      content: {},
    });
    expect(removeOwnedTree(full, [project], owner)).toBeNull();
    expect(existsSync(full)).toBe(false);
  });

  test("refuses a sentinel directory, a symlink, a foreign location, another run's marker and foreign files", () => {
    const sentinel = join(scratch(), "arkive-compaction-eval-AbC123");
    made.push(sentinel);
    mkdirSync(sentinel);
    writeFileSync(join(sentinel, "keep.txt"), "user data");
    expect(rootProblem(sentinel, owner)).toMatch(/no readable marker/u);
    expect(removeOwnedTree(sentinel, [], owner)).not.toBeNull();
    expect(readFileSync(join(sentinel, "keep.txt"), "utf8")).toBe("user data");

    const real = root();
    const link = join(scratch(), "arkive-compaction-eval-Lnk123");
    made.push(link);
    symlinkSync(real, link);
    expect(rootProblem(link, owner)).toMatch(/not a directory/u);

    const owned = root();
    const project = createLocation(owned, "project", owner);
    const foreign = scratch();
    made.push(foreign);
    expect(locationProblem(owned, foreign, owner)).toMatch(/not a location/u);
    expect(locationProblem(owned, join(owned, "judge"), owner)).not.toBeNull();
    expect(rootProblem(owned, { runNonce: "f".repeat(32), out })).toMatch(
      /another run/u,
    );
    expect(rootProblem(owned, { runNonce: NONCE, out: "/elsewhere" })).toMatch(
      /another run/u,
    );
    writeFileSync(join(owned, "foreign.txt"), "x");
    expect(removeOwnedTree(owned, [project], owner)).toMatch(/did not create/u);
    expect(readFileSync(join(owned, "foreign.txt"), "utf8")).toBe("x");
    expect(existsSync(join(owned, MARKER))).toBe(true);
  });

  test("never names the output directory's ancestors as a root", () => {
    expect(rootProblem(join(out, ".."), owner)).not.toBeNull();
  });
});

// --- cleanup of an interrupted run --------------------------------------------

describe("cleanup", () => {
  type Fixture = {
    dir: string;
    root: string;
    project: string;
    ledger: Ledger;
    channelLine: (line: Record<string, unknown>) => void;
  };
  const roots: string[] = [];
  afterEach(() => {
    for (const dir of roots.splice(0))
      rmSync(dir, { recursive: true, force: true });
  });
  function runFixture(
    kind: "v1" | "v2" = "v1",
    witness = true,
    url = SERVER_URL,
  ): Fixture {
    const dir = scratch();
    roots.push(dir);
    const owner = { runNonce: NONCE, out: dir };
    writeFileSync(
      join(dir, "run.json"),
      JSON.stringify({
        format: RUN_FORMAT,
        runNonce: NONCE,
        server: serverRecord(kind, kind === "v2" ? PID : null, url),
        corpus: {
          id: corpus.corpusId,
          sha256: corpus.corpusSha256,
          rubricId: corpus.rubricId,
          rubricSha256: corpus.rubricSha256,
        },
        requested: { model: "p/m", judge: null, repeat: 1, cases: ["A"] },
      }),
    );
    mkdirSync(join(dir, "channel"));
    const ledger = new Ledger(join(dir, "ledger.jsonl"));
    const root = createRoot(owner, "run");
    roots.push(root);
    ledger.append("scratch-root", { directory: root });
    const project = createLocation(root, "project", owner);
    ledger.append("location", { role: "project", directory: project });
    if (witness) ledger.append("server-witness", { directory: root, pid: PID });
    return {
      dir,
      root,
      project,
      ledger,
      channelLine: line =>
        writeFileSync(
          join(dir, "channel", "host-1.jsonl"),
          `${JSON.stringify({ pid: 1, runNonce: NONCE, ...line })}\n`,
          { flag: "a" },
        ),
    };
  }
  const recorded = (directory: string, id: string) =>
    stateOf({ id, directory }, { id }, [], directory);
  function memoryHost(project: string, ids: string[]) {
    const sessions = new Set(ids);
    const calls: string[] = [];
    const host = stubHost({
      pending: async (_, id) =>
        sessions.has(id) ? { exists: true, reasons: [] } : { exists: false },
      read: async (_, id) => recorded(project, id),
      remove: async (_, id) => {
        calls.push(`remove:${id}`);
        sessions.delete(id);
      },
      inventory: async () => [...sessions],
      dispose: async directory => void calls.push(`dispose:${directory}`),
    });
    return { host, sessions, calls };
  }
  const cleanup = (
    dir: string,
    host: EvalHost,
    options: { server?: string; fetch?: FetchLike } = {},
  ) =>
    cleanupRun({
      dir,
      server: options.server ?? SERVER_URL,
      auth: undefined,
      assertHostVersion: null,
      fetch: options.fetch ?? (host.kind === "v2" ? v2Server(PID) : v1Server),
      log: () => {},
      host,
    });

  test("a crashed attempt stays unknown and keeps its sessions and location until its outcome arrives", async () => {
    const f = runFixture();
    const s = f.ledger.sessions("r1-A");
    s.intent("source", null, "t");
    s.created("source", "ses_src");
    const state = recorded(f.project, "ses_src");
    f.ledger.append("source-snapshot", {
      attemptId: "r1-A",
      id: "ses_src",
      headerSha256: state.headerSha256,
      historySha256: state.historySha256,
    });
    f.ledger.append("command-start", {
      attemptId: "r1-A",
      sourceID: "ses_src",
      startedAt: 1,
    });
    const { host, calls } = memoryHost(f.project, ["ses_src"]);
    // The command is still running in the server: no outcome yet.
    f.channelLine({ at: 2, kind: "command-started", sessionID: "ses_src" });
    const first = await cleanup(f.dir, host);
    expect(first.unknownOutcomes).toEqual(["r1-A"]);
    expect(first.sessions).toEqual([
      {
        id: "ses_src",
        action: "retained",
        reason: "its attempt's outcome is unknown",
      },
    ]);
    expect(calls).toEqual([]);
    expect(existsSync(f.project)).toBe(true);

    // It finished after the crash: the outcome settles the attempt.
    f.channelLine({
      at: 3,
      kind: "command-outcome",
      sessionID: "ses_src",
      message: "Magic Compact failed: provider request failed.",
    });
    const second = await cleanup(f.dir, host);
    expect(second.unknownOutcomes).toEqual([]);
    expect(calls).toEqual(["remove:ses_src", `dispose:${f.project}`]);
    expect(existsSync(f.root)).toBe(false);
  });

  test("a successor delivered after a crash is registered, not dropped, and kept without a provable state", async () => {
    const f = runFixture();
    f.ledger.append("command-start", {
      attemptId: "r1-A",
      sourceID: "ses_src",
      startedAt: 1,
    });
    f.channelLine({
      at: 3,
      kind: "command-outcome",
      sessionID: "ses_src",
      message:
        "Magic compaction successful. Continue in successor session ses_late; the original session was left unchanged.",
    });
    const { host, calls } = memoryHost(f.project, ["ses_late"]);
    const summary = await cleanup(f.dir, host);
    expect(summary.stillOwned).toEqual(["ses_late"]);
    expect(summary.sessions[0]).toMatchObject({
      id: "ses_late",
      action: "retained",
    });
    expect(calls).toEqual([]);
    expect(
      Ledger.read(join(f.dir, "ledger.jsonl")).some(
        e =>
          e.kind === "session-created"
          && e["id"] === "ses_late"
          && e["reconciled"],
      ),
    ).toBe(true);
  });

  test("a creation applied before its response failed is read back by its planned ID, never retried", async () => {
    const f = runFixture();
    f.ledger.sessions("r1-A").intent("template", "ses_planned", "the title");
    const { host, calls, sessions } = memoryHost(f.project, ["ses_planned"]);
    host.read = async (_, id) =>
      stateOf(
        { id, directory: f.project, title: "the title" },
        { id },
        [],
        f.project,
      );
    const summary = await cleanup(f.dir, host);
    expect(summary.unresolvedIntents).toEqual([]);
    expect(summary.stillOwned).toEqual(["ses_planned"]);
    expect(summary.sessions[0]).toMatchObject({
      action: "retained",
      reason: expect.stringMatching(/no recorded state/u),
    });
    expect([...sessions]).toEqual(["ses_planned"]);
    expect(calls).toEqual([]);
  });

  test("a session the run cannot account for at its location keeps the location", async () => {
    const f = runFixture();
    // Product leftovers: a session the run never created.
    const { host, calls } = memoryHost(f.project, ["ses_product_candidate"]);
    const summary = await cleanup(f.dir, host);
    expect(summary.disposal[0]).toMatchObject({
      scratchRemoved: false,
      reason: expect.stringMatching(/still lists sessions/u),
    });
    expect(calls).toEqual([]);
    expect(existsSync(f.project)).toBe(true);
  });

  test("a forged ledger location outside the owned root is neither disposed nor deleted", async () => {
    const f = runFixture();
    const foreign = scratch();
    roots.push(foreign);
    writeFileSync(join(foreign, "keep.txt"), "user data");
    f.ledger.append("scratch-root", { directory: foreign });
    f.ledger.append("location", {
      role: "judge",
      directory: join(foreign, "judge"),
    });
    const { host, calls } = memoryHost(f.project, []);
    await cleanup(f.dir, host);
    expect(calls).toEqual([`dispose:${f.project}`]);
    expect(readFileSync(join(foreign, "keep.txt"), "utf8")).toBe("user data");
  });

  // A run's recorded source whose session the answering server does not
  // have: on the creating server it is gone, on any other it is unknown.
  function recordedSource(f: Fixture) {
    const s = f.ledger.sessions("r1-A");
    s.intent("source", null, "t");
    s.created("source", "ses_src");
    f.ledger.append("source-snapshot", {
      attemptId: "r1-A",
      id: "ses_src",
      ...expectedOf(recorded(f.project, "ses_src")),
    });
  }
  const expectedOf = (s: ExpectedState) => ({
    headerSha256: s.headerSha256,
    historySha256: s.historySha256,
  });

  test("another URL is refused before anything is read, marked or disposed", async () => {
    const f = runFixture();
    recordedSource(f);
    const { host, calls } = memoryHost(f.project, []);
    const before = readFileSync(join(f.dir, "ledger.jsonl"), "utf8");
    let fetched = 0;
    for (const server of ["http://127.0.0.1:2", "http://localhost:1"])
      await expect(
        cleanup(f.dir, host, {
          server,
          fetch: async (...args) => {
            fetched += 1;
            return v1Server(...args);
          },
        }),
      ).rejects.toThrow(/pass exactly --server http:\/\/127\.0\.0\.1:1/u);
    expect(fetched).toBe(0);
    expect(calls).toEqual([]);
    expect(readFileSync(join(f.dir, "ledger.jsonl"), "utf8")).toBe(before);
    expect(existsSync(f.project)).toBe(true);
  });

  test("a changed server process on the same URL keeps every session and the location", async () => {
    for (const kind of ["v1", "v2"] as const) {
      const f = runFixture(kind);
      recordedSource(f);
      // The session is not on the answering server, which another process
      // (a restart, or another store) now serves.
      const { host, calls } = memoryHost(f.project, []);
      Object.assign(host, { kind, serverPid: async () => PID + 1 });
      const summary = await cleanup(f.dir, host, {
        fetch: kind === "v2" ? v2Server(PID + 1) : v1Server,
      });
      expect(summary.sessions).toEqual([
        {
          id: "ses_src",
          action: "retained",
          reason: expect.stringMatching(/server process changed/u),
        },
      ]);
      expect(summary.stillOwned).toEqual(["ses_src"]);
      expect(summary.disposal[0]).toMatchObject({ scratchRemoved: false });
      expect(calls).toEqual([]);
      expect(
        Ledger.read(join(f.dir, "ledger.jsonl")).some(
          e => e.kind === "session-absent",
        ),
      ).toBe(false);
      expect(existsSync(f.project)).toBe(true);

      // The creating process answers again: the session is gone from it.
      Object.assign(host, { serverPid: async () => PID });
      const settled = await cleanup(f.dir, host);
      expect(settled.sessions).toEqual([{ id: "ses_src", action: "absent" }]);
      expect(calls).toEqual([`dispose:${f.project}`]);
      expect(existsSync(f.root)).toBe(false);
    }
  });

  test("a root without a recorded server process is disposed only when it never held a session", async () => {
    const empty = runFixture("v1", false);
    const a = memoryHost(empty.project, []);
    expect((await cleanup(empty.dir, a.host)).disposal[0]).toMatchObject({
      scratchRemoved: true,
    });
    const held = runFixture("v1", false);
    recordedSource(held);
    const b = memoryHost(held.project, []);
    const summary = await cleanup(held.dir, b.host);
    expect(summary.sessions[0]).toMatchObject({
      action: "retained",
      reason: expect.stringMatching(/not recorded/u),
    });
    expect(summary.disposal[0]).toMatchObject({ scratchRemoved: false });
    expect(b.calls).toEqual([]);
    expect(existsSync(held.project)).toBe(true);
  });

  // The real CLI against a stand-in server that only identifies itself as
  // OpenCode 1: the exit status is what a script running `cleanup` sees.
  async function cleanupCli(
    open: (url: string) => Fixture,
    during: (f: Fixture) => void,
    runs = 1,
  ) {
    const server = Bun.serve({
      hostname: "127.0.0.1",
      port: 0,
      fetch: request =>
        new URL(request.url).pathname === "/global/health"
          ? Response.json({ healthy: true, version: "1.18.30" })
          : new Response("<!doctype html><html></html>", {
              headers: { "content-type": "text/html" },
            }),
    });
    try {
      const f = open(`http://127.0.0.1:${server.port}`);
      during(f);
      const env = { ...process.env };
      delete env["OPENCODE_SERVER_USERNAME"];
      delete env["OPENCODE_SERVER_PASSWORD"];
      const exits: Array<{ code: number; stderr: string }> = [];
      for (let i = 0; i < runs; i++) {
        const child = Bun.spawn(
          [
            process.execPath,
            join(import.meta.dir, "../eval/compaction/cli.ts"),
            "cleanup",
            f.dir,
            "--server",
            `http://127.0.0.1:${server.port}`,
          ],
          { cwd: dataHome, env, stdout: "pipe", stderr: "pipe" },
        );
        const [stderr, code] = await Promise.all([
          new Response(child.stderr).text(),
          child.exited,
        ]);
        exits.push({ code, stderr });
      }
      return { f, exits };
    } finally {
      await server.stop(true);
    }
  }
  const openAt = (url: string) => runFixture("v1", true, url);
  const keptRoots = (f: Fixture) =>
    Ledger.read(join(f.dir, "ledger.jsonl")).filter(
      e => e.kind === "scratch-kept",
    );

  test("cleanup exits zero when the run's scratch root is already gone", async () => {
    const { f, exits } = await cleanupCli(openAt, f =>
      rmSync(f.root, { recursive: true }),
    );
    expect(exits.map(e => e.code)).toEqual([0]);
    expect(keptRoots(f)).toEqual([]);
  });

  test("cleanup exits non-zero and deletes nothing while a scratch root that exists fails its ownership check", async () => {
    const variants: Array<[string, (f: Fixture) => void]> = [
      ["marker missing", f => unlinkSync(join(f.root, MARKER))],
      [
        "marker of another run",
        f =>
          writeFileSync(
            join(f.root, MARKER),
            `${JSON.stringify({ runNonce: "f".repeat(32), out: f.dir, root: f.root, role: "root" })}\n`,
          ),
      ],
      [
        "dangling symlink in its place",
        f => {
          rmSync(f.root, { recursive: true });
          symlinkSync(join(f.dir, "missing"), f.root);
        },
      ],
    ];
    for (const [name, damage] of variants) {
      const { f, exits } = await cleanupCli(
        openAt,
        f => {
          // A file of the user's that shares the root with the run.
          writeFileSync(join(f.root, "keep.txt"), "user data");
          damage(f);
        },
        2,
      );
      // Repeating the command changes nothing and still reports it.
      expect(exits.map(e => e.code)).toEqual([1, 1]);
      for (const { stderr } of exits) {
        expect(stderr).toContain('"scratchRemoved": false');
        expect(stderr).toContain(f.root);
        expect(stderr).toContain("ownership not proven");
      }
      expect(keptRoots(f).length).toBeGreaterThan(0);
      expect(
        Ledger.read(join(f.dir, "ledger.jsonl")).some(
          e => e.kind === "scratch-removed",
        ),
        name,
      ).toBe(false);
      if (name !== "dangling symlink in its place")
        expect(readFileSync(join(f.root, "keep.txt"), "utf8")).toBe(
          "user data",
        );
      else expect(lstatSync(f.root).isSymbolicLink()).toBe(true);
    }
  });

  test("cleanup reports the retained root in its result when the root exists but is not provably the run's", async () => {
    const f = runFixture();
    unlinkSync(join(f.root, MARKER));
    const { host, calls } = memoryHost(f.project, []);
    const summary = await cleanup(f.dir, host);
    expect(summary.disposal).toEqual([
      {
        directory: f.root,
        scratchRemoved: false,
        reason: expect.stringMatching(
          /^ownership not proven: .*no readable marker/u,
        ),
        locations: [],
      },
    ]);
    expect(calls).toEqual([]);
    expect(existsSync(f.project)).toBe(true);
  });

  // One attempt end to end on the in-memory OpenCode 2 host: the eval's own
  // OpenCode 2 driver, the production /magic-compact, the wrapper's real
  // control and the subject's real storage.
  async function v2Attempt(
    options: {
      // Another writer acts on the successor before the run inspects it.
      beforeInspect?: (state: FakeV2State, successorID: string) => void;
      // Another writer acts on the source right after its import.
      afterSourceImport?: (state: FakeV2State, sourceID: string) => void;
      // Fields the host adds to a session it creates.
      createdExtra?: Record<string, Json>;
      checks?: (checks: Array<{ name: string; ok: boolean }>) => typeof checks;
    } = {},
  ) {
    const f = runFixture("v2");
    const state = fakeV2State();
    const owned = new OwnedSessions();
    const base = fakeV2Host(state, owned, echoGeneration());
    const baseSession = (base as unknown as Record<string, Json>)["session"];
    const client = {
      ...base,
      session: {
        ...baseSession,
        create: async (input: Record<string, Json>) => {
          const session = {
            id: input["id"],
            projectID: "project",
            agent: input["agent"],
            // OpenCode 2.0.28 records variant "default" when none is given.
            model: { variant: "default", ...input["model"] },
            title: input["title"],
            location: input["location"],
            cost: 0,
            tokens: {},
            time: { created: 1, updated: 1 },
            ...options.createdExtra,
          };
          state.sessions.set(session.id, session as never);
          state.messages.set(session.id, []);
          return structuredClone(session);
        },
      },
      debug: { location: { evict: async () => undefined } },
    };
    let sourceImported = false;
    state.afterWrite = (method, id) => {
      if (method !== "import" || sourceImported) return;
      sourceImported = true;
      options.afterSourceImport?.(state, id);
    };
    const driver = v2Host(SERVER_URL, undefined, fetch, client as never);
    const wrapperOptions = evalOptions({
      evalRunNonce: NONCE,
      evalChannel: join(f.dir, "channel"),
      evalSummarizerVariant: null,
    });
    const storage = await subjectStorage("v2", PACKAGE_DIR);
    const control = createControl(
      wrapperOptions,
      f.project,
      async id => {
        const s = state.sessions.get(id) as Json;
        return (
          s && {
            directory: s.location.directory,
            title: s.title,
            metadata: s.metadata,
          }
        );
      },
      async () => storage,
    );
    const removed: string[] = [];
    let inspected = false;
    const host: EvalHost = {
      ...driver,
      runCommand: async (_, sessionID, name, args) => {
        if (name === "arkive-eval-control")
          await control.prepare(sessionID, args);
        else
          await observeCommand(wrapperOptions, sessionID, name, async () => {
            // As the OpenCode 2 command handler reports it.
            const { successor, outcome } = await executeV2Compact(
              base,
              owned,
              sessionID,
              1,
              { retention: { mode: "heuristic" }, controlTimeoutMs: 1_000 },
            );
            throw new Error(
              successor
                ? `${compactSuccess(successor.session.id)}\n${outcome}`
                : outcome,
            );
          }).catch(() => undefined);
        return { kind: "returned", status: 500, message: null, ms: 1 };
      },
      remove: async (directory, id) => {
        await driver.remove(directory, id);
        removed.push(id);
        await control.deleted(id);
      },
      inspectSuccessor: async input => {
        if (!inspected) options.beforeInspect?.(state, input.successorID);
        inspected = true;
        const inspection = await driver.inspectSuccessor(input);
        return options.checks
          ? { ...inspection, checks: options.checks(inspection.checks) }
          : inspection;
      },
    };
    const ctx: RunContext = {
      corpus,
      host,
      identity: { ...serverRecord("v2", PID), kind: "v2" } as never,
      ledger: f.ledger,
      runNonce: NONCE,
      out: f.dir,
      channel: join(f.dir, "channel"),
      project: f.project,
      log: () => {},
      lostOutcomeWaitMs: 0,
      serverPid: PID,
    };
    const record = await runAttempt(
      ctx,
      caseA,
      1,
      parseModelSelector("probe/probe-model", "--model"),
      { imageInput: true } as never,
    );
    const files = (id: string | null) => {
      if (!id) return [];
      const { cache, stats } = storage.paths(storage.key(id));
      return [cache, stats].map(path =>
        existsSync(path) ? readFileSync(path, "utf8") : null,
      );
    };
    const ledgerKinds = () =>
      Ledger.read(join(f.dir, "ledger.jsonl")).map(e => e.kind);
    return { f, state, host, record, removed, files, ledgerKinds };
  }

  test("a valid OpenCode 2 successor is recorded as verified and goes with its storage", async () => {
    const run = await v2Attempt();
    expect(run.record.status).toBe("delivered");
    const successor = run.record.successorID!;
    expect(run.ledgerKinds()).toContain("successor-verified");
    expect(run.removed).toEqual([
      expect.stringMatching(/^ses_/u),
      successor,
      run.record.sourceID,
    ]);
    expect(run.files(successor)).toEqual([null, null]);
    const summary = await cleanup(run.f.dir, run.host);
    expect(summary.stillOwned).toEqual([]);
    expect(summary.disposal[0]).toMatchObject({ scratchRemoved: true });
  });

  test("a successor changed before its first inspection gets no ownership baseline and is kept with its source and storage", async () => {
    const variants: Array<
      [
        string,
        Parameters<typeof v2Attempt>[0],
        ((state: FakeV2State, id: string) => void) | null,
      ]
    > = [
      [
        "a settled foreign user message",
        {
          beforeInspect: (state, id) =>
            void state.messages.get(id)!.push({
              id: "msg_foreign",
              type: "user",
              text: "foreign",
              time: { created: 9 },
            } as V2Message),
        },
        (state, id) => void state.messages.get(id)!.pop(),
      ],
      [
        "a foreign instruction entry",
        {
          beforeInspect: (state, id) =>
            void state.instructions.set(id, [{ key: "foreign", value: "x" }]),
        },
        (state, id) => void state.instructions.delete(id),
      ],
      ["no checks at all", { checks: () => [] }, null],
    ];
    for (const [label, variant, revert] of variants) {
      const run = await v2Attempt(variant);
      const successor = run.record.successorID!;
      const source = run.record.sourceID!;
      expect([label, run.record.status]).toEqual([label, "delivered-invalid"]);
      expect(run.ledgerKinds()).not.toContain("successor-verified");
      expect(run.record.cleanup).toContainEqual({
        id: successor,
        action: "retained",
        reason: expect.stringMatching(/did not pass every mechanical check/u),
      });
      expect(run.record.cleanup).toContainEqual({
        id: source,
        action: "retained",
        reason: expect.stringMatching(/its successor is kept/u),
      });
      const files = run.files(successor);
      expect(files.some(Boolean)).toBe(true);
      // Normal cleanup, then a later one: the successor, its source, its
      // storage and its location stay while the change is there.
      for (let pass = 0; pass < 2; pass++) {
        const summary = await cleanup(run.f.dir, run.host);
        expect(summary.stillOwned.sort()).toEqual([source, successor].sort());
        expect(summary.disposal[0]).toMatchObject({ scratchRemoved: false });
        expect(run.state.sessions.has(successor)).toBe(true);
        expect(run.state.sessions.has(source)).toBe(true);
        expect(run.files(successor)).toEqual(files);
        expect(existsSync(run.f.project)).toBe(true);
      }
      expect(run.removed).not.toContain(successor);
      if (!revert) continue;
      // Once it is undone, the successor verifies from scratch against its
      // recorded source and goes with its storage and the source.
      revert(run.state, successor);
      const settled = await cleanup(run.f.dir, run.host);
      expect([label, settled.stillOwned]).toEqual([label, []]);
      expect(run.files(successor)).toEqual([null, null]);
      expect(settled.disposal[0]).toMatchObject({ scratchRemoved: true });
      expect(run.ledgerKinds()).toContain("successor-verified");
    }
  }, 60_000);

  test("a source changed before it was first read is not adopted and is kept", async () => {
    const changes: Array<
      [string, (session: Json, messages: V2Message[]) => void]
    > = [
      [
        "a foreign message",
        (_, messages) =>
          void messages.push({
            id: "msg_foreign",
            type: "user",
            text: "foreign",
            time: { created: 9 },
          } as V2Message),
      ],
      ["foreign metadata", session => (session.metadata = { foreign: true })],
      [
        "another model",
        session => (session.model = { providerID: "p", id: "other" }),
      ],
      [
        "permissions",
        session =>
          (session.permissions = [
            { action: "*", resource: "*", effect: "deny" },
          ]),
      ],
      ["only the title", session => (session.title = "foreign")],
    ];
    for (const [label, change] of changes) {
      const run = await v2Attempt({
        afterSourceImport: (state, id) =>
          change(state.sessions.get(id) as Json, state.messages.get(id)!),
      });
      expect([label, run.record.status]).toEqual([label, "setup-failed"]);
      expect(run.ledgerKinds()).not.toContain("source-snapshot");
      expect(run.ledgerKinds()).not.toContain("command-start");
      expect(run.record.cleanup).toContainEqual({
        id: run.record.sourceID!,
        action: "retained",
        reason: expect.stringMatching(/did not match the corpus/u),
      });
      const summary = await cleanup(run.f.dir, run.host);
      expect([label, summary.stillOwned]).toEqual([
        label,
        [run.record.sourceID],
      ]);
      expect(summary.disposal[0]).toMatchObject({ scratchRemoved: false });
      expect(run.state.sessions.has(run.record.sourceID!)).toBe(true);
      expect(run.removed).not.toContain(run.record.sourceID);
    }
  });

  test("a template the host created with fields the run did not request seeds no source", async () => {
    const run = await v2Attempt({
      createdExtra: { metadata: { foreign: true } },
    });
    expect(run.record.status).toBe("setup-failed");
    expect(run.record.sourceID).toBeNull();
    expect(run.record.error).toMatch(/did not request/u);
    expect(run.removed).toEqual([]);
    expect([...run.state.sessions.keys()]).toHaveLength(1);
    const summary = await cleanup(run.f.dir, run.host);
    expect(summary.stillOwned).toHaveLength(1);
    expect(summary.disposal[0]).toMatchObject({ scratchRemoved: false });
  });
});

// --- saved runs, grading and scoring -----------------------------------------

describe("judge answers", () => {
  test("the judge prompt carries the schema derived from the judgment shape", () => {
    expect(JUDGMENT_JSON_SCHEMA).toEqual(
      z.toJSONSchema(JudgmentSchema, { target: "draft-07" }),
    );
    expect(JUDGE_SYSTEM).toContain(JSON.stringify(JUDGMENT_JSON_SCHEMA));
    expect(judgePrompt(caseA, corpus.followup, "summary")).toContain(
      JSON.stringify(JUDGMENT_JSON_SCHEMA),
    );
  });

  test("accepts exactly one judgment document and nothing else", () => {
    const valid = {
      facts: [{ id: "A-T1", present: false, evidence: "" }],
      falseAssertions: [
        { dimension: "text", critical: true, claim: "c", evidence: "e" },
      ],
    };
    expect(parseJudgeAnswer(JSON.stringify(valid))).toEqual({
      judgment: valid as Judgment,
    });
    for (const raw of [
      "```json\n" + JSON.stringify(valid) + "\n```",
      `Here it is: ${JSON.stringify(valid)}`,
      JSON.stringify({ ...valid, score: 8 }),
      JSON.stringify({ facts: valid.facts }),
      JSON.stringify({
        ...valid,
        falseAssertions: [{ ...valid.falseAssertions[0], claim: "" }],
      }),
    ])
      expect(parseJudgeAnswer(raw)).toHaveProperty("error");
  });
});

describe("grading", () => {
  const capsule = caseA.gold.textFacts.map(fact => fact.fact).join("\n");
  const judgment = (present: (id: string) => boolean): Judgment => ({
    facts: caseA.gold.textFacts.map(fact => ({
      id: fact.id,
      present: present(fact.id),
      evidence: present(fact.id) ? fact.fact.slice(0, 30) : "",
    })),
    falseAssertions: [],
  });

  test("scores by the frozen rubric", () => {
    expect(
      score(
        judgment(() => true),
        caseA,
      ),
    ).toMatchObject({ textScore: 8, imageScore: null, usable: true });
    expect(
      score(
        judgment(id => id !== "A-T2"),
        caseA,
      ),
    ).toMatchObject({
      textScore: 7,
      usable: false,
      missingCritical: ["A-T2"],
    });
    const wrong = score(
      {
        ...judgment(() => true),
        falseAssertions: [
          {
            dimension: "text",
            critical: true,
            claim: "burst raised",
            evidence: "Root cause",
          },
          {
            dimension: "text",
            critical: true,
            claim: "Burst raised",
            evidence: "Root cause",
          },
        ],
      },
      caseA,
    );
    expect(wrong).toMatchObject({ textScore: 6, falseText: 1, usable: false });
  });

  test("rejects judgments with unknown or missing facts and unquoted evidence", () => {
    expect(
      judgmentProblem(
        judgment(() => true),
        caseA,
        capsule,
      ),
    ).toBeNull();
    const extra = judgment(() => true);
    extra.facts.push({ id: "A-T9", present: false, evidence: "" });
    expect(judgmentProblem(extra, caseA, capsule)).toMatch(/fact IDs/u);
    const short = judgment(() => true);
    short.facts.pop();
    expect(judgmentProblem(short, caseA, capsule)).toMatch(/fact IDs/u);
    const invented = judgment(() => true);
    invented.facts[0]!.evidence = "not in the output";
    expect(judgmentProblem(invented, caseA, capsule)).toMatch(/not quoted/u);
    const image = {
      ...judgment(() => true),
      falseAssertions: [
        {
          dimension: "image" as const,
          critical: false,
          claim: "c",
          evidence: "Root cause",
        },
      ],
    };
    expect(judgmentProblem(image, caseA, capsule)).toMatch(/text-only/u);
  });

  test("grades bind to the run, corpus, rubric, case and exact output", () => {
    const run = {
      runNonce: "n",
      corpusSha256: corpus.corpusSha256,
      rubricSha256: corpus.rubricSha256,
    };
    const attempts = [
      {
        attemptId: "r1-A",
        caseId: "A",
        outputSha256: sha256(capsule),
        capsule,
      },
    ];
    const file = (items: unknown[], patch: Record<string, unknown> = {}) => ({
      format: GRADES_FORMAT,
      ...run,
      grader: { kind: "manual", label: "rater" },
      items,
      ...patch,
    });
    const item = {
      attemptId: "r1-A",
      caseId: "A",
      outputSha256: sha256(capsule),
      ...judgment(() => true),
    };
    expect(
      validateGrades(file([item]), run, attempts, corpus.cases).items,
    ).toHaveLength(1);
    expect(() =>
      validateGrades(
        file([item], { runNonce: "other" }),
        run,
        attempts,
        corpus.cases,
      ),
    ).toThrow(/another run/u);
    expect(() =>
      validateGrades(
        file([item], { rubricSha256: "0".repeat(64) }),
        run,
        attempts,
        corpus.cases,
      ),
    ).toThrow(/rubric/u);
    expect(() =>
      validateGrades(
        file([{ ...item, outputSha256: "0".repeat(64) }]),
        run,
        attempts,
        corpus.cases,
      ),
    ).toThrow(/another case or output/u);
    expect(() =>
      validateGrades(
        file([{ ...item, caseId: "B" }]),
        run,
        attempts,
        corpus.cases,
      ),
    ).toThrow(/another case or output/u);
    expect(() =>
      validateGrades(file([item, item]), run, attempts, corpus.cases),
    ).toThrow(/duplicate/u);
    expect(() =>
      validateGrades(
        file([{ ...item, score: 8 }]),
        run,
        attempts,
        corpus.cases,
      ),
    ).toThrow(/shape/u);
  });
});

describe("saved runs and reports", () => {
  const capsule = caseA.gold.textFacts.map(fact => fact.fact).join("\n");
  const allPresent = (caseId = "A"): Judgment => ({
    facts: corpus.cases
      .find(c => c.id === caseId)!
      .gold.textFacts.map(fact => ({
        id: fact.id,
        present: true,
        evidence: fact.fact.slice(0, 30),
      })),
    falseAssertions: [],
  });
  let dir: string;
  beforeEach(() => {
    dir = scratch();
  });
  afterEach(() => rmSync(dir, { recursive: true, force: true }));

  function savedRun(
    attempts: Array<Partial<AttemptRecord> & { attemptId: string }>,
    repeat = 2,
  ) {
    writeFileSync(
      join(dir, "run.json"),
      JSON.stringify({
        format: RUN_FORMAT,
        runNonce: NONCE,
        server: serverRecord(),
        corpus: {
          id: corpus.corpusId,
          sha256: corpus.corpusSha256,
          rubricId: corpus.rubricId,
          rubricSha256: corpus.rubricSha256,
        },
        requested: { model: "p/m", judge: null, repeat, cases: ["A", "C"] },
      }),
    );
    writeFileSync(
      join(dir, "results.json"),
      JSON.stringify({ status: "completed" }),
    );
    mkdirSync(join(dir, "attempts"));
    for (const a of attempts) {
      const [, repeatText, caseId] = /^r(\d+)-(\w)$/u.exec(a.attemptId)!;
      const delivered = (a.status ?? "delivered") === "delivered";
      writeFileSync(
        join(dir, "attempts", `${a.attemptId}.json`),
        JSON.stringify({
          runNonce: NONCE,
          model: "p/m",
          sourceID: null,
          successorID: null,
          caseId,
          repeat: Number(repeatText),
          status: "delivered",
          capsule: delivered ? capsule : null,
          outputSha256: delivered ? sha256(capsule) : null,
          successorChecks: delivered ? [{ name: "ok", ok: true }] : [],
          sourceUnchanged: delivered ? true : null,
          cleanup: [],
          commandMs: 1,
          usage: null,
          outcome: null,
          ...a,
        }),
      );
    }
  }
  function grade(
    items: Array<{ attemptId: string; judgment: Judgment }>,
    name = "judge-1.json",
    patch: Record<string, unknown> = {},
  ) {
    mkdirSync(join(dir, "grades"), { recursive: true });
    writeFileSync(
      join(dir, "grades", name),
      JSON.stringify({
        format: GRADES_FORMAT,
        runNonce: NONCE,
        corpusSha256: corpus.corpusSha256,
        rubricSha256: corpus.rubricSha256,
        grader: { kind: "judge", label: "judge p/m", model: "p/m" },
        items: items.map(i => ({
          attemptId: i.attemptId,
          caseId: i.attemptId.slice(-1),
          outputSha256: sha256(capsule),
          ...i.judgment,
        })),
        ...patch,
      }),
    );
  }

  test("a valid saved run round-trips; a summary edited under its old digest is refused before any grade or report", () => {
    savedRun([{ attemptId: "r1-A" }]);
    expect(readRun(dir, corpus).attempts.map(a => a.attemptId)).toEqual([
      "r1-A",
    ]);
    const path = join(dir, "attempts", "r1-A.json");
    const record = JSON.parse(readFileSync(path, "utf8"));
    writeFileSync(
      path,
      JSON.stringify({
        ...record,
        capsule: `${capsule}\nThe burst increase shipped.`,
      }),
    );
    expect(() => readRun(dir, corpus)).toThrow(/does not match its digest/u);
    expect(() => summarizeRun(dir, corpus)).toThrow(
      /does not match its digest/u,
    );
  });

  const gradeWith = (server: string, fetchImpl: FetchLike) =>
    gradeRunWithJudge({
      dir,
      server,
      auth: undefined,
      judge: parseModelSelector("p/judge", "--judge"),
      assertHostVersion: null,
      fetch: fetchImpl,
      log: () => {},
    });
  const counting = (inner: FetchLike) => {
    const counter = {
      calls: 0,
      fetch: (async (input, init) => {
        counter.calls += 1;
        return inner(input, init);
      }) as FetchLike,
    };
    return counter;
  };

  test("an intact attempt record from another run is refused before any grade, report or judge request", async () => {
    // The same case and repeat file, digest intact, copied from a run of
    // the same model, or written for another model under this run's nonce.
    for (const patch of [{ runNonce: "f".repeat(32) }, { model: "q/other" }]) {
      rmSync(join(dir, "attempts"), { recursive: true, force: true });
      savedRun([{ attemptId: "r1-A", ...patch }]);
      expect(() => readRun(dir, corpus)).toThrow(/written by another run/u);
      expect(() => summarizeRun(dir, corpus)).toThrow(
        /written by another run/u,
      );
      const server = counting(v1Server);
      await expect(gradeWith(SERVER_URL, server.fetch)).rejects.toThrow(
        /written by another run/u,
      );
      expect(server.calls).toBe(0);
    }
  });

  test("an attempt naming a session the run did not create for it is refused", () => {
    savedRun([{ attemptId: "r1-A", sourceID: "ses_src" }]);
    expect(() => readRun(dir, corpus)).toThrow(/did not create for it/u);
    const ledger = new Ledger(join(dir, "ledger.jsonl"));
    ledger.sessions("r2-A").created("source", "ses_src");
    expect(() => readRun(dir, corpus)).toThrow(/did not create for it/u);
    ledger.sessions("r1-A").created("source", "ses_src");
    expect(readRun(dir, corpus).attempts).toHaveLength(1);
  });

  test("standalone grading refuses another server before creating anything", async () => {
    savedRun([{ attemptId: "r1-A" }]);
    const other = counting(v1Server);
    await expect(gradeWith("http://127.0.0.1:2", other.fetch)).rejects.toThrow(
      /pass exactly --server http:\/\/127\.0\.0\.1:1/u,
    );
    expect(other.calls).toBe(0);
    await expect(gradeWith(SERVER_URL, v2Server(PID))).rejects.toThrow(
      /not the kind of host/u,
    );
    expect(existsSync(join(dir, "ledger.jsonl"))).toBe(false);
  });

  test("refuses attempts for cases, repeats or statuses the run could not have written", () => {
    for (const bad of [
      { attemptId: "r1-B", caseId: "B" },
      { attemptId: "r3-A", repeat: 3 },
      { attemptId: "r1-A", status: "excellent" },
      { attemptId: "r1-A", sourceUnchanged: false },
    ]) {
      rmSync(join(dir, "attempts"), { recursive: true, force: true });
      savedRun([bad as never]);
      expect(() => readRun(dir, corpus)).toThrow(EvalError);
    }
  });

  test("a settled failure scores 0 next to a graded repeat; unknown outcomes are not scored", () => {
    savedRun([
      { attemptId: "r1-A" },
      {
        attemptId: "r2-A",
        status: "failed",
        outcome: { kind: "failed", category: "provider", message: "x" },
      },
      { attemptId: "r1-C", status: "outcome-unknown" },
    ]);
    grade([{ attemptId: "r1-A", judgment: allPresent() }]);
    const summary = summarizeRun(dir, corpus);
    expect(summary.semantic).toBe("GRADED");
    expect(summary.zeroScored).toBe(1);
    expect(summary.unscored).toBe(1);
    expect(summary.graders[0]!.perCase).toEqual([
      {
        caseId: "A",
        graded: 1,
        zero: 1,
        ungraded: 0,
        unscored: 0,
        textMean: 4,
        imageMean: null,
        usable: 1,
        scored: 2,
      },
      {
        caseId: "C",
        graded: 0,
        zero: 0,
        ungraded: 0,
        unscored: 1,
        textMean: null,
        imageMean: null,
        usable: 0,
        scored: 0,
      },
    ]);
  });

  test("a missing grade next to a failure claims no mean", () => {
    savedRun([
      { attemptId: "r1-A" },
      { attemptId: "r2-A" },
      { attemptId: "r1-C", status: "noop" },
    ]);
    grade([{ attemptId: "r1-A", judgment: allPresent() }]);
    const summary = summarizeRun(dir, corpus);
    expect(summary.semantic).toBe("PARTIAL");
    expect(summary.graders[0]).toMatchObject({
      complete: false,
      graded: 1,
      delivered: 2,
    });
    expect(summary.graders[0]!.perCase[0]).toMatchObject({
      ungraded: 1,
      textMean: null,
    });
    expect(summary.graders[0]!.perCase[1]).toMatchObject({
      zero: 1,
      textMean: 0,
      imageMean: 0,
      usable: 0,
      scored: 1,
    });
  });

  test("when every attempt failed, every settled attempt scores 0 without a grader", () => {
    savedRun([
      { attemptId: "r1-A", status: "failed" },
      { attemptId: "r1-C", status: "preflight-skipped" },
      { attemptId: "r2-C", status: "setup-failed" },
    ]);
    const summary = summarizeRun(dir, corpus);
    expect(summary.semantic).toBe("GRADED");
    expect(summary.rubricOnly).toEqual([
      {
        caseId: "A",
        graded: 0,
        zero: 1,
        ungraded: 0,
        unscored: 0,
        textMean: 0,
        imageMean: null,
        usable: 0,
        scored: 1,
      },
      {
        caseId: "C",
        graded: 0,
        zero: 1,
        ungraded: 0,
        unscored: 1,
        textMean: 0,
        imageMean: 0,
        usable: 0,
        scored: 1,
      },
    ]);
  });

  test("a report is UNGRADED without grades and never counts a rejected file", () => {
    savedRun([{ attemptId: "r1-A" }, { attemptId: "r2-A" }]);
    expect(summarizeRun(dir, corpus).semantic).toBe("UNGRADED");
    grade([{ attemptId: "r1-A", judgment: allPresent() }], "a-rejected.json", {
      runNonce: "f".repeat(32),
    });
    expect(summarizeRun(dir, corpus).semantic).toBe("UNGRADED");
    grade([{ attemptId: "r1-A", judgment: allPresent() }], "b-partial.json");
    const partial = summarizeRun(dir, corpus);
    expect(partial.semantic).toBe("PARTIAL");
    expect(
      partial.graders.map(g => [g.rejected !== null, g.graded, g.complete]),
    ).toEqual([
      [true, 0, false],
      [false, 1, false],
    ]);
  });
});

describe("runs", () => {
  test("a run refuses an existing output directory before contacting the server", async () => {
    const dir = scratch();
    let calls = 0;
    try {
      await expect(
        runEval({
          server: "http://127.0.0.1:1",
          auth: undefined,
          model: parseModelSelector("p/m", "--model"),
          out: dir,
          repeat: 1,
          cases: null,
          judge: null,
          dryRun: true,
          cacheRoot: null,
          subject: null,
          assertHostVersion: null,
          fetch: async () => {
            calls += 1;
            return new Response("");
          },
        }),
      ).rejects.toThrow(/already exists/u);
      expect(calls).toBe(0);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

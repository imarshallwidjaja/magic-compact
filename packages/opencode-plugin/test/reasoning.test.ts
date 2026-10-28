// The shared reasoning policy and its host adapters: which models' protected
// reasoning behavior is verified, which protected reasoning a summary model is
// sent, the `--match-model` choice and its fallbacks, the feedback line, the
// replay-cost estimate of OpenCode 1 records, the OpenCode 2
// readable-reasoning projection, and the structured whole-session summary
// shape.
import { describe, expect, test } from "bun:test";
import type { Message, Part } from "@opencode-ai/sdk/v2";
import { requestAllTurnsReasoning } from "../src/compact/compact";
import type { MessageWithParts } from "../src/compact/plan";
import {
  estimateReplay,
  indexReplay,
  reasoningUnits,
  type ReplayContext,
} from "../src/compact/reasoning";
import {
  chooseStage,
  continuationTokens,
  matchCandidate,
  READABLE_REASONING_HEADER,
  reasoningAccess,
  reasoningContextMode,
  reasoningFeedback,
  replayFeedback,
  stageReport,
  verifiedReasoning,
  type MatchCandidate,
  type MatchPreflight,
  type ModelIdentity,
  type ReasoningUnit,
  type StageChoice,
} from "../src/policy/reasoning";
import { structuredSessionSummary } from "../src/policy/retention";
import { countGptTokens } from "../src/stats/tokenize";
import {
  projectReadableReasoning,
  readableReasoningHeaderTokens,
  v2ReasoningRoute,
  v2ReasoningUnits,
  v2ReplayEstimate,
} from "../src/v2/reasoning";
import type { V2Message } from "../src/v2/session";

// The verified base model and its verified fast catalog entry, an unverified
// OpenAI model, and the configured summarizer, all on the OpenAI Responses
// route.
const SOL = { providerID: "openai", modelID: "gpt-6.1-sol" };
const SOL_FAST = { providerID: "openai", modelID: "gpt-6.1-sol-fast" };
const TERRA = { providerID: "openai", modelID: "gpt-terra" };
const CONFIGURED = { providerID: "openai", modelID: "gpt-summary" };
const ANTHROPIC = { providerID: "anthropic", modelID: "claude" };
const qualifies = (model: ModelIdentity) =>
  model.providerID === SOL.providerID && model.modelID === SOL.modelID;

let clock = 0;

function assistant(
  id: string,
  parts: Part[],
  info: Partial<Extract<Message, { role: "assistant" }>> = {},
  model = SOL,
): MessageWithParts {
  clock += 1;
  return {
    info: {
      id,
      sessionID: "source",
      role: "assistant",
      providerID: model.providerID,
      modelID: model.modelID,
      time: { created: clock, completed: clock },
      ...info,
    } as Message,
    parts: parts.map(part => ({ ...part, messageID: id })) as Part[],
  };
}

function reasoning(
  id: string,
  text: string,
  openai?: { itemId?: string; reasoningEncryptedContent?: string | null },
): Part {
  return {
    id,
    sessionID: "source",
    messageID: "",
    type: "reasoning",
    text,
    time: { start: 1, end: 2 },
    ...(openai ? { metadata: { openai } } : {}),
  } as Part;
}

function stepFinish(id: string, reasoningTokens: number): Part {
  return {
    id,
    sessionID: "source",
    messageID: "",
    type: "step-finish",
    reason: "stop",
    cost: 0,
    tokens: {
      input: 10,
      output: reasoningTokens + 5,
      reasoning: reasoningTokens,
      cache: { read: 0, write: 0 },
    },
  } as Part;
}

function stepStart(id: string): Part {
  return { id, sessionID: "source", messageID: "", type: "step-start" } as Part;
}

const ALL_TURNS: ReplayContext = { primary: SOL, mode: "all_turns" };

describe("verified reasoning routes and models", () => {
  const route = (
    model: ModelIdentity,
    facts: { wireModelID?: string; responses?: boolean } = {},
  ) => ({
    model,
    wireModelID: facts.wireModelID ?? model.modelID,
    responses: facts.responses ?? true,
  });

  test("verifies only the live-qualified catalog models and their wire model on the OpenAI Responses route", () => {
    expect(verifiedReasoning(route(SOL))).toEqual({
      defaultContext: "all_turns",
    });
    // The fast catalog entry sends the same wire model and was observed live
    // under its own ID.
    expect(
      verifiedReasoning(route(SOL_FAST, { wireModelID: "gpt-6.1-sol" })),
    ).toEqual({ defaultContext: "all_turns" });
    for (const unverified of [
      // An older model on the same route has no verified default or option.
      route({ providerID: "openai", modelID: "gpt-5.5" }),
      // A custom provider serving the same wire model, under either ID.
      route({ providerID: "my-openai", modelID: "gpt-6.1-sol" }),
      route(
        { providerID: "my-openai", modelID: "gpt-6.1-sol-fast" },
        { wireModelID: "gpt-6.1-sol" },
      ),
      // The same models on another serializer.
      route(SOL, { responses: false }),
      route(SOL_FAST, { wireModelID: "gpt-6.1-sol", responses: false }),
      // Other mode-derived catalog entries that send the verified wire model.
      route(
        { providerID: "openai", modelID: "gpt-6.1-sol-pro" },
        { wireModelID: "gpt-6.1-sol" },
      ),
      route(
        { providerID: "openai", modelID: "gpt-6.1-sol-ultrafast" },
        { wireModelID: "gpt-6.1-sol" },
      ),
      // A catalog entry whose wire model differs from the verified one.
      route(SOL, { wireModelID: "gpt-6.1-sol-2026" }),
      route(SOL_FAST, { wireModelID: "gpt-6.1-sol-2026" }),
      route(SOL_FAST, { wireModelID: "gpt-6.1-sol-fast" }),
      // The base model's ID is not the fast ID, whichever wire model is sent.
      route(SOL, { wireModelID: "gpt-6.1-sol-fast" }),
      // A host that lists no wire model.
      { model: SOL_FAST, wireModelID: undefined, responses: true },
      undefined,
    ]) {
      expect(verifiedReasoning(unverified)).toBeUndefined();
    }
  });

  test("resolves the reasoning context from the verified default and the operator's options", () => {
    const verified = { defaultContext: "all_turns" } as const;
    expect(reasoningContextMode([], verified)).toBe("all_turns");
    expect(reasoningContextMode([undefined, "all_turns"], verified)).toBe(
      "all_turns",
    );
    expect(reasoningContextMode(["current_turn"], verified)).toBe(
      "current_turn",
    );
    // Levels that disagree are not resolved by an assumed precedence.
    expect(reasoningContextMode(["current_turn", "all_turns"], verified)).toBe(
      "unknown",
    );
    expect(reasoningContextMode(["auto"], verified)).toBe("unknown");
  });

  test("asks for every turn's reasoning only for owned requests on a verified model", () => {
    const sol = {
      providerID: "openai",
      id: "gpt-6.1-sol",
      api: { id: "gpt-6.1-sol", npm: "@ai-sdk/openai" },
    };
    const set = (owned: boolean, model: typeof sol) => {
      const options: Record<string, unknown> = {
        reasoningContext: "current_turn",
      };
      requestAllTurnsReasoning(owned, model, options);
      return options["reasoningContext"];
    };
    expect(set(true, sol)).toBe("all_turns");
    expect(set(false, sol)).toBe("current_turn");
    // The fast catalog entry sends the same wire model and is verified.
    const fast = { ...sol, id: "gpt-6.1-sol-fast" };
    expect(set(true, fast)).toBe("all_turns");
    expect(set(false, fast)).toBe("current_turn");
    for (const model of [
      {
        ...sol,
        id: "gpt-5.5",
        api: { id: "gpt-5.5", npm: "@ai-sdk/openai" },
      },
      { ...sol, providerID: "my-openai" },
      { ...fast, providerID: "my-openai" },
      { ...sol, api: { ...sol.api, npm: "@ai-sdk/openai-compatible" } },
      { ...fast, api: { ...sol.api, npm: "@ai-sdk/openai-compatible" } },
      { ...sol, id: "gpt-6.1-sol-pro" },
      { ...sol, id: "gpt-6.1-sol-ultrafast" },
      // The fast ID on a wire model other than the verified one.
      { ...fast, api: { ...sol.api, id: "gpt-6.1-sol-fast" } },
      { ...sol, providerID: "deepseek" },
    ]) {
      expect(set(true, model)).toBe("current_turn");
    }
  });

  test("qualifies OpenCode 2 models by provider, serializer package and wire model", () => {
    const package_ = "@opencode/ai/providers/openai";
    expect(
      verifiedReasoning(
        v2ReasoningRoute(SOL, { modelID: "gpt-6.1-sol", package: package_ }),
      ),
    ).toBeDefined();
    // The model list gives the fast entry its own `id` and the base `modelID`.
    expect(
      verifiedReasoning(
        v2ReasoningRoute(SOL_FAST, {
          modelID: "gpt-6.1-sol",
          package: package_,
        }),
      ),
    ).toEqual({ defaultContext: "all_turns" });
    for (const [model, info] of [
      [{ providerID: "openai", modelID: "gpt-5.5" }, { modelID: "gpt-5.5" }],
      [{ providerID: "my-openai", modelID: "gpt-6.1-sol" }, {}],
      [{ providerID: "my-openai", modelID: "gpt-6.1-sol-fast" }, {}],
      [{ providerID: "openai", modelID: "gpt-6.1-sol-pro" }, {}],
      [{ providerID: "openai", modelID: "gpt-6.1-sol-ultrafast" }, {}],
      [SOL_FAST, { modelID: "gpt-6.1-sol-fast" }],
      [SOL_FAST, { package: "@opencode/ai/providers/openai-compatible" }],
      [SOL, { package: "@opencode/ai/providers/openai-compatible" }],
      [SOL, { package: undefined }],
    ] as const) {
      expect(
        verifiedReasoning(
          v2ReasoningRoute(model, {
            modelID: "gpt-6.1-sol",
            package: package_,
            ...info,
          }),
        ),
      ).toBeUndefined();
    }
  });
});

describe("reasoning access and the --match-model choice", () => {
  test("counts protected units once per item and never counts an item ID or field name alone", () => {
    const units = reasoningUnits([
      assistant("a1", [
        reasoning("r1", "first summary part", {
          itemId: "rs_1",
          reasoningEncryptedContent: "enc-1",
        }),
        reasoning("r2", "second summary part", {
          itemId: "rs_1",
          reasoningEncryptedContent: "enc-1",
        }),
        reasoning("r3", "item without payload", {
          itemId: "rs_2",
          reasoningEncryptedContent: null,
        }),
        reasoning("r4", "plain reasoning"),
      ]),
    ]);
    expect(units).toEqual([
      { origin: SOL, protected: true },
      { origin: SOL, protected: false },
      { origin: SOL, protected: false },
    ]);
    expect(reasoningAccess(units, SOL, qualifies)).toEqual({
      compatible: 1,
      unavailable: 0,
      unknown: 0,
    });
    expect(reasoningAccess(units, CONFIGURED, qualifies)).toEqual({
      compatible: 0,
      unavailable: 1,
      unknown: 0,
    });
    // The model's own units on an unverified route, and units without an
    // origin, are unknown; another model's are never replayed to it.
    expect(
      reasoningAccess(
        [
          { origin: TERRA, protected: true },
          { origin: undefined, protected: true },
          { origin: ANTHROPIC, protected: true },
        ],
        TERRA,
        qualifies,
      ),
    ).toEqual({ compatible: 0, unavailable: 1, unknown: 2 });
  });

  const preflight =
    (facts: Partial<MatchPreflight> = {}) =>
    (): MatchPreflight => ({
      variants: new Set(["low"]),
      optionsConflict: false,
      imagesBlocked: false,
      ...facts,
    });
  const configured = { model: CONFIGURED, variant: "low" };
  const choose = (units: ReasoningUnit[], facts?: Partial<MatchPreflight>) =>
    chooseStage(matchCandidate(units, qualifies), configured, preflight(facts));

  test("matches the one verified origin model and carries the configured variant", () => {
    expect(choose([{ origin: SOL, protected: true }])).toEqual({
      kind: "match",
      model: SOL,
      variant: "low",
    });
    // No configured variant means none, never the source model's own.
    expect(
      chooseStage(
        matchCandidate([{ origin: SOL, protected: true }], qualifies),
        { model: CONFIGURED },
        preflight({ variants: new Set() }),
      ),
    ).toEqual({ kind: "match", model: SOL, variant: undefined });
  });

  test("decides the origin across every protected target unit before qualifying it", () => {
    // A verified origin mixed with an unverified one, or with a unit that
    // records no origin, is not unique.
    for (const other of [ANTHROPIC, TERRA, undefined]) {
      const units = [
        { origin: SOL, protected: true },
        { origin: other, protected: true },
      ];
      expect(matchCandidate(units, qualifies)).toEqual({ kind: "mixed" });
      expect(choose(units)).toEqual({
        kind: "fallback",
        reason: "mixed-target",
      });
    }
    // One unverified origin, or only units without one, is unqualified.
    for (const origin of [ANTHROPIC, TERRA, undefined]) {
      const units = [
        { origin, protected: true },
        { origin, protected: true },
      ];
      expect(matchCandidate(units, qualifies)).toEqual({
        kind: "unqualified",
      });
      expect(choose(units)).toEqual({
        kind: "fallback",
        reason: "unqualified",
      });
    }
    // Unprotected units never decide.
    expect(
      matchCandidate(
        [
          { origin: SOL, protected: true },
          { origin: ANTHROPIC, protected: false },
        ],
        qualifies,
      ),
    ).toEqual({ kind: "model", model: SOL });
  });

  test.each([
    ["variant", { variants: new Set(["high"]) }],
    ["options", { optionsConflict: true }],
    ["image", { imagesBlocked: true }],
  ] as const)(
    "falls back to the configured summarizer for %s",
    (reason, facts) => {
      expect(choose([{ origin: SOL, protected: true }], facts)).toEqual({
        kind: "fallback",
        reason,
      });
    },
  );

  test("keeps the configured summarizer when nothing protected needs a match", () => {
    for (const units of [[], [{ origin: SOL, protected: false }]]) {
      expect(choose(units)).toEqual({ kind: "configured" });
    }
    expect(
      chooseStage(
        matchCandidate([{ origin: SOL, protected: true }], qualifies),
        { model: SOL },
        preflight(),
      ),
    ).toEqual({ kind: "configured" });
  });

  const report = (
    candidate: MatchCandidate,
    choice: StageChoice,
    model: ModelIdentity,
    units: ReasoningUnit[],
    replayUnknown = 0,
  ) =>
    stageReport({
      stage: "turn",
      candidate,
      choice,
      model,
      units,
      qualifies,
      replayUnknown,
    });
  const sol = [{ origin: SOL, protected: true }];
  const solCandidate = { kind: "model", model: SOL } as const;
  const match = { kind: "match", model: SOL, variant: "low" } as const;

  test("stays silent without --match-model unless the flag or a replay estimate has something to say", () => {
    expect(reasoningFeedback([], false)).toBeUndefined();
    expect(
      reasoningFeedback(
        [report({ kind: "none" }, { kind: "configured" }, CONFIGURED, [])],
        false,
      ),
    ).toBeUndefined();
    // Reasoning the flag could not unlock is not actionable.
    expect(
      reasoningFeedback(
        [
          report(
            solCandidate,
            { kind: "fallback", reason: "variant" },
            CONFIGURED,
            sol,
          ),
        ],
        false,
      ),
    ).toBeUndefined();
    // Already configured on the origin model: nothing to change.
    expect(
      reasoningFeedback(
        [report(solCandidate, { kind: "configured" }, SOL, sol)],
        false,
      ),
    ).toBeUndefined();
    expect(
      reasoningFeedback(
        [
          report(solCandidate, match, CONFIGURED, sol),
          report({ kind: "none" }, { kind: "configured" }, CONFIGURED, []),
        ],
        false,
      ),
    ).toBe(
      "The configured summarizer ran every summary request. Protected reasoning units replayed to the model that wrote them: 0 of 1 (1 written by other models, 0 unverified). --match-model would run 1 request on the model that wrote its reasoning.",
    );
    // A request that may be sent protected reasoning with no attributable
    // count is reported even when nothing is actionable.
    expect(
      reasoningFeedback(
        [
          report(
            { kind: "unqualified" },
            { kind: "fallback", reason: "unqualified" },
            TERRA,
            [{ origin: TERRA, protected: true }],
            1,
          ),
        ],
        false,
      ),
    ).toBe(
      "Protected reasoning units replayed to the model that wrote them: 0 of 1 (0 written by other models, 1 unverified). 1 summary request may be sent protected reasoning with no attributable replay count, which its input estimate leaves out.",
    );
  });

  test("counts requests that ran on their origin model under --match-model, whether matched or already configured", () => {
    // Already configured on the origin model.
    expect(
      reasoningFeedback(
        [report(solCandidate, { kind: "configured" }, SOL, sol)],
        true,
      ),
    ).toBe(
      "--match-model: 1 of 1 summary request with protected reasoning ran on the verified model that wrote it. Protected reasoning units replayed to the model that wrote them: 1 of 1 (0 written by other models, 0 unverified).",
    );
    expect(
      reasoningFeedback(
        [
          report(solCandidate, match, SOL, sol),
          report(
            solCandidate,
            { kind: "fallback", reason: "capacity" },
            CONFIGURED,
            sol,
          ),
          report(
            { kind: "mixed" },
            { kind: "fallback", reason: "mixed-target" },
            CONFIGURED,
            [{ origin: undefined, protected: true }],
          ),
          report({ kind: "none" }, { kind: "configured" }, CONFIGURED, []),
        ],
        true,
      ),
    ).toBe(
      "--match-model: 1 of 3 summary requests with protected reasoning ran on the verified model that wrote it. 2 requests kept the configured summarizer (too small 1, reasoning from several models 1). Protected reasoning units replayed to the model that wrote them: 1 of 3 (1 written by other models, 1 unverified).",
    );
    expect(reasoningFeedback([], true)).toBe(
      "--match-model: 0 of 0 summary requests with protected reasoning ran on the verified model that wrote it. No summary request carried protected reasoning.",
    );
  });

  test("counts an unverified configured summarizer on its own origin as unverified, not as a request that ran on another model", () => {
    const units = [{ origin: TERRA, protected: true }];
    const candidate = matchCandidate(units, qualifies);
    const choice = chooseStage(candidate, { model: TERRA }, preflight());
    expect(choice).toEqual({ kind: "fallback", reason: "unqualified" });
    expect(
      reasoningFeedback([report(candidate, choice, TERRA, units)], true),
    ).toBe(
      "--match-model: 0 of 1 summary request with protected reasoning ran on the verified model that wrote it. 1 request kept the configured summarizer (unverified model or origin 1). Protected reasoning units replayed to the model that wrote them: 0 of 1 (0 written by other models, 1 unverified).",
    );
  });
});

describe("OpenCode 1 reasoning replay estimate", () => {
  const unit = (
    id: string,
    tokens: number,
    item = `rs_${id}`,
    payload = `enc_${id}`,
    model = SOL,
  ) =>
    assistant(
      id,
      [
        stepStart(`${id}-start`),
        reasoning(`${id}-r`, `visible summary of ${id}`, {
          itemId: item,
          reasoningEncryptedContent: payload,
        }),
        { ...stepFinish(`${id}-finish`, tokens) },
      ],
      {
        tokens: {
          input: 1,
          output: 1,
          reasoning: 999,
          cache: { read: 0, write: 0 },
        },
      },
      model,
    );

  test("charges a step's reasoning counter once for a whole intact unit, replacing its visible text", () => {
    const message = assistant("multi", [
      stepStart("s1"),
      reasoning("p1", "part one", {
        itemId: "rs_m",
        reasoningEncryptedContent: "enc_m",
      }),
      reasoning("p2", "part two", {
        itemId: "rs_m",
        reasoningEncryptedContent: "enc_m",
      }),
      stepFinish("f1", 531),
      stepStart("s2"),
      reasoning("p3", "next step", {
        itemId: "rs_n",
        reasoningEncryptedContent: "enc_n",
      }),
      stepFinish("f2", 40),
    ]);
    const estimate = estimateReplay([message], ALL_TURNS);
    expect(estimate).toEqual({
      tokens: 571,
      visibleTokens:
        countGptTokens("part one")
        + countGptTokens("part two")
        + countGptTokens("next step"),
      units: 2,
      unknown: 0,
    });
    // The message's own counters (999 here) are never read.
    expect(continuationTokens(1_000, estimate)).toBe(
      1_000 - estimate.visibleTokens + 571,
    );
  });

  test("marks unattributable, partly removed, repeated, errored and unfinished units unknown", () => {
    const cases: MessageWithParts[][] = [
      // No step-finish after the reasoning.
      [
        assistant("open", [
          reasoning("r", "x", {
            itemId: "rs_o",
            reasoningEncryptedContent: "enc_o",
          }),
        ]),
      ],
      // One of the item's parts lost its payload.
      [
        assistant("partial", [
          reasoning("r1", "x", {
            itemId: "rs_p",
            reasoningEncryptedContent: "enc_p",
          }),
          reasoning("r2", "y", { itemId: "rs_p" }),
          stepFinish("f", 10),
        ]),
      ],
      // Two items in one step share one counter.
      [
        assistant("two-items", [
          reasoning("r1", "x", {
            itemId: "rs_a",
            reasoningEncryptedContent: "enc_a",
          }),
          reasoning("r2", "y", {
            itemId: "rs_b",
            reasoningEncryptedContent: "enc_b",
          }),
          stepFinish("f", 10),
        ]),
      ],
      // The same item occurs in two steps; neither is deduplicated.
      [
        unit("first", 10, "rs_same", "enc_same"),
        unit("second", 10, "rs_same", "enc_same"),
      ],
      [
        assistant(
          "errored",
          [
            reasoning("r", "x", {
              itemId: "rs_e",
              reasoningEncryptedContent: "enc_e",
            }),
            stepFinish("f", 10),
          ],
          { error: { name: "UnknownError", data: { message: "x" } } } as never,
        ),
      ],
      [
        assistant(
          "unfinished",
          [
            reasoning("r", "x", {
              itemId: "rs_u",
              reasoningEncryptedContent: "enc_u",
            }),
            stepFinish("f", 10),
          ],
          { time: { created: 1 } },
        ),
      ],
      [unit("zero", 0)],
    ];
    for (const messages of cases) {
      const estimate = estimateReplay(messages, ALL_TURNS);
      expect(estimate.tokens).toBe(0);
      expect(estimate.units).toBe(0);
      expect(estimate.unknown).toBeGreaterThan(0);
    }
  });

  test("costs nothing for another model's reasoning and stays unknown without a verified mode", () => {
    // The host replays a payload only to the model that wrote it, on any
    // route; another model is sent its visible text alone.
    for (const model of [TERRA, ANTHROPIC]) {
      expect(
        estimateReplay([unit("other", 50, "rs_t", "enc_t", model)], ALL_TURNS),
      ).toEqual({ tokens: 0, visibleTokens: 0, units: 0, unknown: 0 });
    }
    const own = unit("sol", 50);
    // current_turn: the items are still sent, but the verified model counted
    // them as no input.
    expect(
      estimateReplay([own], { ...ALL_TURNS, mode: "current_turn" }),
    ).toEqual({
      tokens: 0,
      visibleTokens: countGptTokens("visible summary of sol"),
      units: 1,
      unknown: 0,
    });
    for (const context of [
      undefined,
      { ...ALL_TURNS, mode: "unknown" as const },
    ]) {
      expect(estimateReplay([own], context).unknown).toBe(1);
    }
    // A unit without a recorded origin may be the model's own.
    const anonymous = unit("anonymous", 50);
    Object.assign(anonymous.info, { providerID: "", modelID: "" });
    expect(estimateReplay([anonymous], ALL_TURNS).unknown).toBe(1);
  });

  test("never charges usage copied onto a message whose reasoning was deleted, and agrees on a persisted copy", () => {
    const source = unit("kept", 120);
    const summarized = { ...source, parts: [] };
    expect(estimateReplay([summarized], ALL_TURNS)).toEqual({
      tokens: 0,
      visibleTokens: 0,
      units: 0,
      unknown: 0,
    });
    // A successor copies records under new IDs; the estimate is the same.
    const copy = structuredClone(source);
    copy.info.id = "copied";
    copy.parts = copy.parts.map(part => ({
      ...part,
      id: `copy-${part.id}`,
      messageID: "copied",
    }));
    expect(estimateReplay([copy], ALL_TURNS)).toEqual(
      estimateReplay([source], ALL_TURNS),
    );
  });

  test("keeps one attribution for every sum over a history, so removing a repeat never attributes the other", () => {
    const kept = unit("kept", 120);
    const repeat = unit("repeat", 30, "rs_kept", "enc_kept");
    const index = indexReplay([kept, repeat]);
    // Each copy is unknown on the shared basis, whichever subset is summed.
    expect(index.estimate(ALL_TURNS, new Set(["kept"]))).toEqual({
      tokens: 0,
      visibleTokens: 0,
      units: 0,
      unknown: 1,
    });
    expect(index.estimate(ALL_TURNS, new Set(["repeat"])).unknown).toBe(1);
    expect(index.estimate(ALL_TURNS).unknown).toBe(2);
    // Only a history without the repeat attributes the unit.
    expect(estimateReplay([kept], ALL_TURNS).units).toBe(1);
  });

  test("reports known removals and unattributed units without inventing either", () => {
    const none = { tokens: 0, visibleTokens: 0, units: 0, unknown: 0 };
    expect(replayFeedback(none, none)).toBeUndefined();
    // Known units the summarized turns did not hold: nothing removed.
    expect(
      replayFeedback(
        { tokens: 300, visibleTokens: 5, units: 1, unknown: 0 },
        none,
      ),
    ).toBeUndefined();
    expect(
      replayFeedback(
        { tokens: 600, visibleTokens: 20, units: 2, unknown: 1 },
        { tokens: 500, visibleTokens: 15, units: 1, unknown: 1 },
      ),
    ).toBe(
      "About 500 replay tokens removed with 1 protected reasoning unit (estimate from provider reasoning counts, not stored text). 1 protected reasoning unit the continuing model may be sent had no attributable provider count and is not counted.",
    );
    expect(
      replayFeedback(
        { tokens: 0, visibleTokens: 0, units: 0, unknown: 2 },
        none,
      ),
    ).toBe(
      "2 protected reasoning units the continuing model may be sent had no attributable provider count and are not counted.",
    );
  });
});

describe("OpenCode 2 readable reasoning", () => {
  const v2Assistant = (
    id: string,
    content: unknown[],
    model = { providerID: "deepseek", id: "deepseek-flash" },
  ) =>
    ({
      id,
      type: "assistant",
      agent: "build",
      model,
      content,
      time: { created: 1, completed: 2 },
    }) as unknown as V2Message;
  const responses = (model: ModelIdentity) => ({
    model,
    wireModelID: model.modelID,
    responses: true,
  });

  test("rewrites only metadata-free reasoning, in place and once", () => {
    const native = {
      type: "reasoning",
      text: "native item",
      providerMetadata: {
        openai: { itemId: "rs_1", reasoningEncryptedContent: "enc" },
      },
    };
    const messages = [
      { role: "user", content: [{ type: "text", text: "question" }] },
      {
        role: "assistant",
        content: [
          { type: "reasoning", text: "plain analysis from deepseek" },
          { type: "tool-call", id: "call_1", name: "read", input: {} },
          native,
          { type: "reasoning", text: "" },
          { type: "text", text: "answer" },
        ],
      },
    ];
    const before = structuredClone(messages);
    const projected = projectReadableReasoning(messages);
    expect(messages).toEqual(before);
    expect(projected[1]!.content).toEqual([
      {
        type: "text",
        text: `${READABLE_REASONING_HEADER}\nplain analysis from deepseek`,
      },
      { type: "tool-call", id: "call_1", name: "read", input: {} },
      native,
      { type: "reasoning", text: "" },
      { type: "text", text: "answer" },
    ]);
    expect(JSON.stringify(projected)).not.toContain(
      '"type":"reasoning","text":"plain analysis',
    );
  });

  test("counts header tokens on the Responses serializer for any model, and treats protected payloads as units, not text", () => {
    const history = [
      v2Assistant("m1", [
        {
          type: "reasoning",
          text: "plain one",
          state: { reasoningField: "reasoning_content" },
        },
        { type: "reasoning", text: "" },
      ]),
      v2Assistant(
        "m2",
        [
          {
            type: "reasoning",
            text: "sol summary",
            state: { itemId: "rs_1", reasoningEncryptedContent: "enc" },
          },
        ],
        { providerID: "openai", id: "gpt-6.1-sol" },
      ),
    ];
    const header = countGptTokens(`${READABLE_REASONING_HEADER}\n`) + 1;
    expect(readableReasoningHeaderTokens(history, responses(SOL))).toBe(header);
    // An older model on the same serializer still needs the projection.
    expect(
      readableReasoningHeaderTokens(
        history,
        responses({ providerID: "openai", modelID: "gpt-5.5" }),
      ),
    ).toBe(2 * header);
    expect(
      readableReasoningHeaderTokens(history, {
        ...responses({ providerID: "deepseek", modelID: "deepseek-flash" }),
        responses: false,
      }),
    ).toBe(0);
    expect(v2ReasoningUnits(history)).toEqual([
      {
        origin: { providerID: "deepseek", modelID: "deepseek-flash" },
        protected: false,
      },
      {
        origin: { providerID: "deepseek", modelID: "deepseek-flash" },
        protected: false,
      },
      { origin: SOL, protected: true },
    ]);
    // OpenCode 2 replay cost is never estimated: the model's own protected
    // unit is unknown and another model's costs nothing.
    expect(v2ReplayEstimate(history, SOL)).toEqual({
      tokens: 0,
      visibleTokens: 0,
      units: 0,
      unknown: 1,
    });
    expect(v2ReplayEstimate(history, TERRA).unknown).toBe(0);
  });
});

describe("structured session summary shape", () => {
  test("accepts exactly one nonempty summary string", () => {
    expect(structuredSessionSummary({ summary: "  ## Goal\n- x  " })).toBe(
      "## Goal\n- x",
    );
    for (const value of [
      undefined,
      null,
      "text",
      [],
      {},
      { summary: "" },
      { summary: "   " },
      { summary: 1 },
      { summary: "x", extra: true },
      { turns: [] },
    ]) {
      expect(structuredSessionSummary(value)).toBeUndefined();
    }
  });
});

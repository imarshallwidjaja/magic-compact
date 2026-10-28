// Reasoning access for summary requests, shared by both hosts. A host replays
// protected reasoning (an encrypted or signed payload) only to the model that
// produced it, so a summarizer on another model reads none of it. Each summary
// request is one stage: the whole-session retention summary or one turn
// summary. Adapters turn their native records into `ReasoningUnit`s and their
// model descriptors into `ReasoningRoute`s; the choices, fallbacks and
// feedback below are the same on both hosts.

export type ModelIdentity = { providerID: string; modelID: string };

export function sameModel(a: ModelIdentity, b: ModelIdentity): boolean {
  return a.providerID === b.providerID && a.modelID === b.modelID;
}

// What a host's model descriptor says about the route a request on that model
// takes: the model ID it sends on the wire, and whether it serializes the
// request with its OpenAI Responses serializer.
export type ReasoningRoute = {
  model: ModelIdentity;
  wireModelID: string | undefined;
  responses: boolean;
};

// The reasoning context a request uses. `current_turn` still sends earlier
// reasoning items, but the verified model counted them as no input (the same
// input tokens as removing them).
export type ReasoningContextMode = "all_turns" | "current_turn" | "unknown";

// Models whose protected reasoning behavior Magic Compact verified live on the
// OpenAI Responses route, by the host's catalog model ID: the wire model that
// ID sends, the `reasoningContext` request option reaching the provider, the
// historical default it uses when unset, and that the model reads its own
// replayed reasoning. `gpt-6.1-sol-fast` is the catalog entry for the same
// wire model in fast mode; it was observed live under its own catalog ID, not
// inferred from `gpt-6.1-sol`, and OpenCode 1 records its messages under that
// ID. A record's model is compared by exact ID, never normalized. Any other ID,
// including other catalog entries that send the same wire model (`-pro`,
// `-ultrafast`, a custom alias), and every other route or provider is unknown.
const VERIFIED_REASONING_MODELS: ReadonlyMap<
  string,
  {
    wireModelID: string;
    defaultContext: Exclude<ReasoningContextMode, "unknown">;
  }
> = new Map([
  ["gpt-6.1-sol", { wireModelID: "gpt-6.1-sol", defaultContext: "all_turns" }],
  [
    "gpt-6.1-sol-fast",
    { wireModelID: "gpt-6.1-sol", defaultContext: "all_turns" },
  ],
]);

export type VerifiedReasoning = {
  defaultContext: Exclude<ReasoningContextMode, "unknown">;
};

export function verifiedReasoning(
  route: ReasoningRoute | undefined,
): VerifiedReasoning | undefined {
  if (
    route === undefined
    || route.model.providerID !== "openai"
    || !route.responses
  ) {
    return undefined;
  }
  const verified = VERIFIED_REASONING_MODELS.get(route.model.modelID);
  if (!verified || verified.wireModelID !== route.wireModelID) {
    return undefined;
  }
  return { defaultContext: verified.defaultContext };
}

// Whether a model's protected reasoning behavior is verified, from the host's
// live model list.
export type Qualifies = (model: ModelIdentity) => boolean;

// The reasoning context a request on a verified model uses, from the options
// the operator set at the levels that can carry it. Unset uses the model's
// verified default. Any other value, or levels that disagree, is unknown:
// option precedence and other plugins' hooks are not modelled.
export function reasoningContextMode(
  values: readonly unknown[],
  verified: VerifiedReasoning,
): ReasoningContextMode {
  const set = new Set(values.filter(value => value !== undefined));
  if (set.size === 0) return verified.defaultContext;
  if (set.size === 1 && set.has("all_turns")) return "all_turns";
  if (set.size === 1 && set.has("current_turn")) return "current_turn";
  return "unknown";
}

// Keys that hold an actual encrypted or signed reasoning payload, in the
// provider state a host stores with a reasoning record (OpenCode 1 nests it
// under the provider's name, OpenCode 2 stores it flat). An item ID or a
// reasoning-field name alone is not protected.
const PROTECTED_KEYS = [
  "reasoningEncryptedContent",
  "encryptedContent",
  "signature",
  "redactedData",
  "thoughtSignature",
];

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

// The protected payload in a reasoning record's stored provider state, or
// undefined.
export function protectedPayload(state: unknown): string | undefined {
  if (!isObject(state)) return undefined;
  for (const record of [state, ...Object.values(state)]) {
    if (!isObject(record)) continue;
    for (const key of PROTECTED_KEYS) {
      const payload = record[key];
      if (typeof payload === "string" && payload.length > 0) return payload;
    }
  }
  return undefined;
}

// The provider item ID in a reasoning record's stored provider state.
export function reasoningItemID(state: unknown): string | undefined {
  if (!isObject(state)) return undefined;
  for (const record of [state, ...Object.values(state)]) {
    if (isObject(record) && typeof record["itemId"] === "string") {
      return record["itemId"];
    }
  }
  return undefined;
}

// The header readable reasoning carries when a request shows another model's
// plain reasoning as historical assistant text.
export const READABLE_REASONING_HEADER =
  "Earlier reasoning, shown as text (provisional analysis, not a verified result):";

// One reasoning record a request may carry, counted once per stored item.
// `protected` means it holds a nonempty encrypted or signed payload, not
// merely an item ID or a reasoning-field name. `origin` is the model of the
// assistant message that produced it, when the record says.
export type ReasoningUnit = {
  origin: ModelIdentity | undefined;
  protected: boolean;
};

// Protected units by whether a request's model is sent them: `compatible`
// units came from that model and its behavior is verified, `unavailable`
// units came from another model, which the host never replays them to, and
// `unknown` units have no recorded origin or come from that model on an
// unverified route. Counts never claim the model understood anything.
export type ReasoningAccess = {
  compatible: number;
  unavailable: number;
  unknown: number;
};

export function reasoningAccess(
  units: readonly ReasoningUnit[],
  model: ModelIdentity,
  qualifies: Qualifies,
): ReasoningAccess {
  const access = { compatible: 0, unavailable: 0, unknown: 0 };
  for (const unit of units) {
    if (!unit.protected) continue;
    if (!unit.origin) {
      access.unknown += 1;
    } else if (!sameModel(unit.origin, model)) {
      access.unavailable += 1;
    } else if (qualifies(model)) {
      access.compatible += 1;
    } else {
      access.unknown += 1;
    }
  }
  return access;
}

// The one model whose protected reasoning a stage could unlock. The origin is
// decided across every protected unit the stage covers before qualification:
// one origin may still be unverified, and a unit without a recorded origin
// makes the stage unqualified on its own or mixed with others. A turn stage
// reads only its target turn's units, never the preceding halo; the session
// stage passes the units of the current primary model.
export type MatchCandidate =
  | { kind: "none" }
  | { kind: "mixed" }
  | { kind: "unqualified" }
  | { kind: "model"; model: ModelIdentity };

export function matchCandidate(
  units: readonly ReasoningUnit[],
  qualifies: Qualifies,
): MatchCandidate {
  const origins = new Map<string, ModelIdentity | undefined>();
  for (const unit of units) {
    if (!unit.protected) continue;
    origins.set(
      unit.origin
        ? `${unit.origin.providerID}\u0000${unit.origin.modelID}`
        : "",
      unit.origin,
    );
  }
  if (origins.size === 0) return { kind: "none" };
  if (origins.size > 1) return { kind: "mixed" };
  const [origin] = origins.values();
  if (!origin || !qualifies(origin)) return { kind: "unqualified" };
  return { kind: "model", model: origin };
}

// Why a stage that could match kept the configured summarizer. Every reason is
// decided before any temporary session or model request.
export type MatchFallback =
  "mixed-target" | "unqualified" | "variant" | "options" | "image" | "capacity";

// What the preflight found for a model the stage could use instead of the
// configured summarizer. `variants` is the set the host lists for that
// model; `optionsConflict` is true when the summarizer agent carries operator
// request options that were written for the configured model.
export type MatchPreflight = {
  variants: ReadonlySet<string> | undefined;
  optionsConflict: boolean;
  imagesBlocked: boolean;
};

export type StageChoice =
  | { kind: "configured" }
  | { kind: "match"; model: ModelIdentity; variant: string | undefined }
  | { kind: "fallback"; reason: MatchFallback };

// The choice for one stage, before any capacity check. The configured
// variant carries over when the matched model lists it; an explicit variant
// the model does not list falls back rather than being dropped, and the
// source model's own variant is never used.
export function chooseStage(
  candidate: MatchCandidate,
  configured: { model: ModelIdentity; variant?: string | undefined },
  preflight: (model: ModelIdentity) => MatchPreflight,
): StageChoice {
  if (candidate.kind === "none") return { kind: "configured" };
  if (candidate.kind === "mixed") {
    return { kind: "fallback", reason: "mixed-target" };
  }
  if (candidate.kind === "unqualified") {
    return { kind: "fallback", reason: "unqualified" };
  }
  if (sameModel(candidate.model, configured.model)) {
    return { kind: "configured" };
  }
  const facts = preflight(candidate.model);
  if (
    configured.variant !== undefined
    && !facts.variants?.has(configured.variant)
  ) {
    return { kind: "fallback", reason: "variant" };
  }
  if (facts.optionsConflict) return { kind: "fallback", reason: "options" };
  if (facts.imagesBlocked) return { kind: "fallback", reason: "image" };
  return {
    kind: "match",
    model: candidate.model,
    variant: configured.variant,
  };
}

// One summary request as the feedback reports it. `onOrigin` is true when the
// request ran on the one verified model that wrote all of its target's
// protected reasoning, whether `--match-model` chose it or it was already
// the configured summarizer. An unverified origin is never `onOrigin`, even
// when it is the configured summarizer; its units count as unverified. A
// `match` choice that was not applied (no `--match-model`) is an actionable
// mismatch. `replayUnknown` counts protected units the request's model may be
// sent whose replay cost is not in the request's input estimate.
export type StageReport = {
  stage: "session" | "turn";
  choice: StageChoice;
  onOrigin: boolean;
  access: ReasoningAccess;
  replayUnknown: number;
};

// `--match-model`, and the list each summary request of a command adds its
// report to.
export type ReasoningReports = {
  matchModel: boolean;
  reports: StageReport[];
};

export function stageReport(request: {
  stage: StageReport["stage"];
  candidate: MatchCandidate;
  choice: StageChoice;
  // The model the request runs on and the units of the history it sends.
  model: ModelIdentity;
  units: readonly ReasoningUnit[];
  qualifies: Qualifies;
  replayUnknown: number;
}): StageReport {
  return {
    stage: request.stage,
    choice: request.choice,
    onOrigin:
      request.candidate.kind === "model"
      && sameModel(request.candidate.model, request.model),
    access: reasoningAccess(request.units, request.model, request.qualifies),
    replayUnknown: request.replayUnknown,
  };
}

const REASON_LABELS: Record<MatchFallback, string> = {
  "mixed-target": "reasoning from several models",
  unqualified: "unverified model or origin",
  variant: "variant not offered",
  options: "agent options",
  image: "no image input",
  capacity: "too small",
};

const count = (value: number, noun: string) =>
  `${value} ${noun}${value === 1 ? "" : "s"}`;

// One content-free line about the summary models, the protected reasoning
// their requests were sent and any replay cost their input estimates leave
// out. Without `--match-model` it appears only when the flag would have run a
// request on the model that wrote its reasoning, or when an estimate left a
// replay cost out; otherwise there is nothing to say.
export function reasoningFeedback(
  reports: readonly StageReport[],
  matchModel: boolean,
): string | undefined {
  const total = reports.reduce(
    (sum, report) => ({
      compatible: sum.compatible + report.access.compatible,
      unavailable: sum.unavailable + report.access.unavailable,
      unknown: sum.unknown + report.access.unknown,
    }),
    { compatible: 0, unavailable: 0, unknown: 0 },
  );
  const units = total.compatible + total.unavailable + total.unknown;
  const sent =
    units === 0
      ? "No summary request carried protected reasoning."
      : `Protected reasoning units replayed to the model that wrote them: ${total.compatible} of ${units} (${total.unavailable} written by other models, ${total.unknown} unverified).`;
  const replayUnknown = reports.filter(
    report => report.replayUnknown > 0,
  ).length;
  const unknownText =
    replayUnknown === 0
      ? ""
      : ` ${count(replayUnknown, "summary request")} may be sent protected reasoning with no attributable replay count, which ${replayUnknown === 1 ? "its" : "their"} input estimate leaves out.`;
  if (!matchModel) {
    const actionable = reports.filter(
      report => report.choice.kind === "match",
    ).length;
    if (actionable === 0) {
      return replayUnknown === 0 ? undefined : `${sent}${unknownText}`;
    }
    return `The configured summarizer ran every summary request. ${sent} --match-model would run ${count(actionable, "request")} on the model that wrote its reasoning.${unknownText}`;
  }
  const eligible = reports.filter(
    report =>
      report.access.compatible
        + report.access.unavailable
        + report.access.unknown
      > 0,
  ).length;
  const onOrigin = reports.filter(report => report.onOrigin).length;
  const reasons = new Map<MatchFallback, number>();
  for (const report of reports) {
    if (report.choice.kind !== "fallback") continue;
    const reason = report.choice.reason;
    reasons.set(reason, (reasons.get(reason) ?? 0) + 1);
  }
  const fallbacks = [...reasons.values()].reduce((sum, n) => sum + n, 0);
  const fallbackText =
    fallbacks === 0
      ? ""
      : ` ${count(fallbacks, "request")} kept the configured summarizer (${[
          ...reasons,
        ]
          .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
          .map(([reason, n]) => `${REASON_LABELS[reason]} ${n}`)
          .join(", ")}).`;
  return `--match-model: ${onOrigin} of ${count(eligible, "summary request")} with protected reasoning ran on the verified model that wrote it.${fallbackText} ${sent}${unknownText}`;
}

// A clearly labelled estimate of what replaying protected reasoning costs the
// model a request runs on, from provider counters attributed to whole intact
// units. `unknown` units could not be attributed and add nothing to `tokens`.
export type ReplayEstimate = {
  // Estimated replay tokens of the attributed units.
  tokens: number;
  // Visible reasoning-summary tokens of those units already in the canonical
  // count, which the whole-unit estimate replaces rather than adds to.
  visibleTokens: number;
  units: number;
  unknown: number;
};

export const NO_REPLAY: ReplayEstimate = {
  tokens: 0,
  visibleTokens: 0,
  units: 0,
  unknown: 0,
};

// The input cost surface: canonical visible tokens with each attributed unit's
// visible text replaced by its whole-unit estimate.
export function continuationTokens(
  visibleTokens: number,
  replay: ReplayEstimate,
): number {
  return visibleTokens - replay.visibleTokens + replay.tokens;
}

// The estimate line a compaction notice carries when the source held
// protected reasoning, or undefined. `removed` is the summarized turns'
// share of `source`, measured on the same attribution, so it never exceeds
// it.
export function replayFeedback(
  source: ReplayEstimate,
  removed: ReplayEstimate,
): string | undefined {
  if (source.units + source.unknown === 0) return undefined;
  const known =
    removed.units === 0
      ? []
      : [
          `About ${removed.tokens} replay tokens removed with ${count(removed.units, "protected reasoning unit")} (estimate from provider reasoning counts, not stored text).`,
        ];
  const unknown =
    source.unknown === 0
      ? []
      : [
          `${count(source.unknown, "protected reasoning unit")} the continuing model may be sent had no attributable provider count and ${source.unknown === 1 ? "is" : "are"} not counted.`,
        ];
  return [...known, ...unknown].join(" ") || undefined;
}

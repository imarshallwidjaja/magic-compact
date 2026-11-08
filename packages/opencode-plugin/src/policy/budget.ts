// Generation request budgets and per-target batch planning, shared by both
// hosts. Each host estimates its own records' turn costs.
import { PluginMessageError } from "./feedback";

export const NATIVE_BATCH_NORMAL_INPUT_TARGET = 400_000;
export const NATIVE_BATCH_FIXED_RESERVE_TOKENS = 8_000;
export const NATIVE_BATCH_MEDIA_TOKEN_CHARGE = 8_000;
export const NATIVE_BATCH_MAX_OUTPUT_TARGET = 32_000;
export const NATIVE_BATCH_MAX_HALO_INPUT = 500_000;
// The pinned tokenizer emits one token per byte here, the maximum for a valid 22-byte nonce body.
export const NATIVE_BATCH_PLANNING_NONCE = "uB_4fLx2Xv5Tz7E1M-8vQn";

const NATIVE_BATCH_FALLBACK_OUTPUT_LIMIT = 32_000;
const NATIVE_BATCH_MIN_TURN_OUTPUT = 512;
const NATIVE_BATCH_MAX_TURN_TEXT_OUTPUT = 8_000;
const NATIVE_BATCH_MEDIA_DESCRIPTION_OUTPUT = 512;

export type NativeBatchBudget = {
  outputTarget: number;
  usableInput: number;
  normalInputTarget: number;
  hardSingleTurnInput: number;
  haloInputLimit: number;
};

export type NativeTurnCost = {
  inputTokens: number;
  outputTokens: number;
};

// Whether a request fits a model. A capacity refusal is an expected outcome,
// not an error: a `--match-model` choice that does not fit falls back to the
// configured summarizer, while the configured summarizer's own refusal ends
// the command with `message`, through `requireFit` or the caller's wording.
export type CapacityRefusal = { kind: "capacity"; message: string };
export type Fit<T> = { kind: "fits"; value: T } | CapacityRefusal;

export function fits<T>(value: T): Fit<T> {
  return { kind: "fits", value };
}

export function requireFit<T>(fit: Fit<T>): T {
  if (fit.kind === "capacity") throw new PluginMessageError(fit.message);
  return fit.value;
}

export function refuse(message: string): CapacityRefusal {
  return { kind: "capacity", message };
}

// `explicitRequestTokens(null)` is the request without its target: the fixed
// instructions and global index every request repeats.
export type NativeRequestProtocol<T extends NativeTurnCost> = {
  explicitRequestTokens: (candidate: T | null) => number;
  responseProtocolTokens: (candidate: T) => number;
};

export type NativeGenerationRequest<T extends NativeTurnCost> = {
  candidate: T;
  estimatedInputTokens: number;
  estimatedExplicitRequestTokens: number;
  estimatedOutputTokens: number;
  inputLimit: number;
};

export function computeNativeBatchBudget(limit: {
  context: number;
  input?: number;
  output?: number;
}): Fit<NativeBatchBudget> {
  if (!Number.isFinite(limit.context) || limit.context <= 0) {
    return refuse(
      "Magic Compact requires a positive configured compaction model context limit.",
    );
  }
  const context = Math.floor(limit.context);
  const outputLimit =
    Number.isFinite(limit.output) && (limit.output ?? 0) > 0
      ? Math.floor(limit.output!)
      : NATIVE_BATCH_FALLBACK_OUTPUT_LIMIT;
  const outputTarget = Math.min(
    NATIVE_BATCH_MAX_OUTPUT_TARGET,
    Math.floor(outputLimit * 0.5),
  );
  const configuredInput =
    Number.isFinite(limit.input) && (limit.input ?? 0) > 0
      ? Math.floor(limit.input!)
      : Number.POSITIVE_INFINITY;
  const usableInput = Math.floor(
    Math.min(configuredInput, context - outputTarget),
  );
  if (usableInput <= NATIVE_BATCH_FIXED_RESERVE_TOKENS) {
    return refuse(
      "Magic Compact configured compaction model has no usable input budget.",
    );
  }
  const hardSingleTurnInput = Math.floor(usableInput * 0.9);
  if (hardSingleTurnInput <= NATIVE_BATCH_FIXED_RESERVE_TOKENS) {
    return refuse(
      "Magic Compact configured compaction model input budget is too small to fit the fixed host reserve.",
    );
  }
  return fits({
    outputTarget,
    usableInput,
    normalInputTarget: Math.min(
      NATIVE_BATCH_NORMAL_INPUT_TARGET,
      Math.floor(usableInput * 0.8),
    ),
    hardSingleTurnInput,
    haloInputLimit: Math.min(NATIVE_BATCH_MAX_HALO_INPUT, hardSingleTurnInput),
  });
}

// The summary output a turn is expected to need, from its visible text and
// its inline media count. Both hosts use this estimate.
export function estimateTurnOutputTokens(
  textTokens: number,
  mediaCount: number,
): number {
  return (
    Math.max(
      NATIVE_BATCH_MIN_TURN_OUTPUT,
      Math.min(NATIVE_BATCH_MAX_TURN_TEXT_OUTPUT, Math.ceil(textTokens * 0.05)),
    )
    + mediaCount * NATIVE_BATCH_MEDIA_DESCRIPTION_OUTPUT
  );
}

export function outputCapacityMessage(budget: NativeBatchBudget): string {
  return `A single compaction turn cannot fit the configured ${budget.outputTarget}-token output target. Configure a compaction model with more output capacity.`;
}

// Every generation request owns exactly one target turn. The host does not
// enforce the response schema, and multi-target requests returned the wrong
// number of results live, so the response must never have to map several
// targets back to their ordinals. The request without its target must fit
// first: it is the floor every request of the command repeats.
export function planNativeGenerationRequest<T extends NativeTurnCost>(
  candidate: T,
  budget: NativeBatchBudget,
  protocol: NativeRequestProtocol<T>,
): Fit<NativeGenerationRequest<T>> {
  const emptyRequestTokens = Math.ceil(protocol.explicitRequestTokens(null));
  if (
    NATIVE_BATCH_FIXED_RESERVE_TOKENS + emptyRequestTokens
    > budget.usableInput
  ) {
    return refuse(
      "Magic Compact explicit request and global turn index cannot fit the configured model input budget. Run /magic-trim first or configure a compaction model with more input capacity.",
    );
  }
  const estimatedExplicitRequestTokens = Math.ceil(
    protocol.explicitRequestTokens(candidate),
  );
  const estimatedInputTokens =
    NATIVE_BATCH_FIXED_RESERVE_TOKENS
    + estimatedExplicitRequestTokens
    + Math.ceil(candidate.inputTokens);
  const estimatedOutputTokens =
    Math.ceil(protocol.responseProtocolTokens(candidate))
    + Math.ceil(candidate.outputTokens);
  if (estimatedInputTokens > budget.hardSingleTurnInput) {
    return refuse(
      "A single compaction turn cannot fit the configured model input. Run /magic-trim first and retry.",
    );
  }
  if (estimatedOutputTokens > budget.outputTarget) {
    return refuse(outputCapacityMessage(budget));
  }
  return fits({
    candidate,
    estimatedInputTokens,
    estimatedExplicitRequestTokens,
    estimatedOutputTokens,
    inputLimit:
      estimatedInputTokens > budget.normalInputTarget
        ? budget.hardSingleTurnInput
        : budget.normalInputTarget,
  });
}

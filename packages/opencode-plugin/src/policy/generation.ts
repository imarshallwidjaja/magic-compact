import { randomBytes } from "node:crypto";
import { GENERATION_PROTOCOL } from "../compact/constants";
import { countGptTokens } from "../stats/tokenize";
import { isRecord } from "../util";

// Host-neutral per-turn generation protocol: the response schema, prompt and
// markers, the nonce, strict response validation, summary framing, and the
// provider-error classification both hosts apply to their native failures.

// One target turn of a generation request.
export type GenerationTarget = {
  ordinal: number;
  marker: string;
  expectedInlineImages: number;
};

// Every generation request owns one target turn, so the schema asks for one
// result. Hosts transmit it to the model without enforcing it;
// parseGenerationResponse remains the authority.
export const GENERATION_RESPONSE_SCHEMA = {
  type: "object",
  properties: {
    turns: {
      type: "array",
      minItems: 1,
      maxItems: 1,
      items: {
        type: "object",
        properties: {
          ordinal: { type: "integer" },
          marker: { type: "string" },
          summary: { type: "string" },
          visualDescriptions: {
            type: "array",
            items: {
              type: "object",
              properties: {
                ordinal: { type: "integer" },
                description: { type: "string" },
              },
              required: ["ordinal", "description"],
              additionalProperties: false,
            },
          },
        },
        required: ["ordinal", "marker", "summary", "visualDescriptions"],
        additionalProperties: false,
      },
    },
  },
  required: ["turns"],
  additionalProperties: false,
};

export const GLOBAL_ANCHOR_CHARACTER_LIMIT = 160;
const GENERATION_NONCE_ATTEMPTS = 8;
const GENERATION_NONCE_PATTERN = /^[A-Za-z0-9_-]{22}$/u;

export type GenerationTurnResult = {
  summary: string;
  visualDescriptions: Array<{ ordinal: number; description: string }>;
};

export function generationTurnMarker(nonce: string, ordinal: number): string {
  return `<magic-compact-turn nonce="${nonce}" ordinal="${ordinal}" />`;
}

export function createGenerationNonce(): string {
  return randomBytes(16).toString("base64url");
}

export function selectGenerationNonce(
  histories: unknown[],
  nonceFactory: () => string,
  generationNonces: Set<string>,
): string {
  for (let attempt = 0; attempt < GENERATION_NONCE_ATTEMPTS; attempt += 1) {
    const nonce = nonceFactory();
    if (!GENERATION_NONCE_PATTERN.test(nonce)) {
      throw new Error("Compaction generation marker nonce is invalid.");
    }
    if (
      !generationNonces.has(nonce)
      && !histories.some(history => containsString(history, nonce))
    ) {
      generationNonces.add(nonce);
      return nonce;
    }
  }
  throw new GenerationMarkerNonceCollisionError();
}

export function containsString(value: unknown, needle: string): boolean {
  const pending = [value];
  const seen = new Set<object>();
  while (pending.length > 0) {
    const current = pending.pop();
    if (typeof current === "string") {
      if (current.includes(needle)) return true;
      continue;
    }
    if (
      (typeof current !== "object" && typeof current !== "function")
      || current === null
      || seen.has(current)
    ) {
      continue;
    }
    seen.add(current);
    for (const key of Object.keys(current)) {
      pending.push(ownDataProperty(current, key));
    }
  }
  return false;
}

export function buildGenerationPrompt(
  candidates: GenerationTarget[],
  globalTurnIndexPrompt: string,
): string {
  return [
    "Generate one continuation summary for the indexed target turn using the retained native history.",
    GENERATION_PROTOCOL,
    "Submit the object through StructuredOutput. Its turns array must contain exactly one object, for the single target index entry below.",
    globalTurnIndexPrompt,
    "Use the ephemeral markers only as ordering guides. Echo each exact marker token and target ordinal; marker echo is allowed. Never include the active generation nonce or marker token in summaries or visual descriptions; useful source identifiers may remain when you include them.",
    JSON.stringify({
      turns: candidates.map(candidate => ({
        ordinal: candidate.ordinal,
        marker: candidate.marker,
        expectedInlineImages: candidate.expectedInlineImages,
      })),
    }),
  ].join("\n");
}

export function estimateGenerationExplicitRequestTokens(
  candidates: GenerationTarget[],
  globalTurnIndexPrompt: string,
): number {
  return (
    countGptTokens(JSON.stringify(GENERATION_RESPONSE_SCHEMA))
    + countGptTokens(buildGenerationPrompt(candidates, globalTurnIndexPrompt))
    + candidates.reduce(
      (total, candidate) => total + countGptTokens(candidate.marker),
      0,
    )
  );
}

export function estimateGenerationResponseProtocolTokens(
  candidates: GenerationTarget[],
): number {
  return countGptTokens(
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
  );
}

// The orientation-only map every request of one compaction repeats verbatim.
export function globalTurnIndexPrompt(
  globalTurnIndex: Array<{ sourceOrdinal: number; anchor: string }>,
): string {
  return [
    "The globalTurnIndex is an orientation-only user-anchor map. Its sourceOrdinal values are immutable source-history positions, not target ordinals, and it does not supply distant assistant or tool detail. Each entry contains only its sourceOrdinal and bounded user anchor:",
    JSON.stringify({ globalTurnIndex }),
  ].join("\n");
}

export function parseGenerationResponse(
  parsed: unknown,
  candidates: GenerationTarget[],
  nonce: string,
): GenerationTurnResult[] {
  if (parsed === undefined)
    throw validationFailure("missing structured output");
  if (
    !isPlainRecord(parsed)
    || !hasExactKeys(parsed, ["turns"])
    || !Array.isArray(parsed["turns"])
  ) {
    throw validationFailure("invalid envelope");
  }
  if (parsed["turns"].length !== candidates.length) {
    throw validationFailure("turn count mismatch", {
      expected: candidates.length,
      received: parsed["turns"].length,
    });
  }

  return parsed["turns"].map((value, index) => {
    const candidate = candidates[index]!;
    if (
      !isPlainRecord(value)
      || !hasExactKeys(value, [
        "marker",
        "ordinal",
        "summary",
        "visualDescriptions",
      ])
      || !Array.isArray(value["visualDescriptions"])
    ) {
      throw validationFailure("invalid turn shape or identity", {
        field: "shape",
        ordinal: candidate.ordinal,
      });
    }
    const identityField =
      value["ordinal"] !== candidate.ordinal
        ? "ordinal"
        : value["marker"] !== candidate.marker
          ? "marker"
          : undefined;
    if (identityField) {
      throw validationFailure("invalid turn shape or identity", {
        field: identityField,
        ordinal: candidate.ordinal,
      });
    }
    if (typeof value["summary"] !== "string" || !value["summary"].trim()) {
      throw validationFailure("invalid summary");
    }
    const expectedImages = candidate.expectedInlineImages;
    if (value["visualDescriptions"].length !== expectedImages) {
      throw validationFailure("invalid visual coverage");
    }
    const summary = value["summary"].trim();
    if (summary.includes(nonce)) {
      throw validationFailure("nonce leakage");
    }
    const visualDescriptions = value["visualDescriptions"].map(
      (description, imageIndex) => {
        if (
          !isPlainRecord(description)
          || !hasExactKeys(description, ["description", "ordinal"])
          || description["ordinal"] !== imageIndex + 1
          || typeof description["description"] !== "string"
          || !description["description"].trim()
        ) {
          throw validationFailure("invalid visual coverage");
        }
        const text = description["description"].trim();
        if (text.includes(nonce)) {
          throw validationFailure("nonce leakage");
        }
        return {
          ordinal: imageIndex + 1,
          description: text,
        };
      },
    );
    return {
      summary,
      visualDescriptions,
    };
  });
}

export function frameSummary(result: GenerationTurnResult): string {
  if (result.visualDescriptions.length === 0) return result.summary;
  return [
    result.summary,
    "",
    "Visual summary:",
    ...result.visualDescriptions.map(
      item => `${item.ordinal}. ${item.description}`,
    ),
  ].join("\n");
}

export function capCodePoints(value: string, characterLimit: number): string {
  return Array.from(value).slice(0, characterLimit).join("");
}

type ValidationCategory =
  | "missing structured output"
  | "invalid envelope"
  | "turn count mismatch"
  | "invalid turn shape or identity"
  | "invalid summary"
  | "invalid visual coverage"
  | "nonce leakage";

type TurnCounts = { expected: number; received: number };

// Names the first failed check and its target ordinal, never response content.
type TurnIdentityFailure = {
  field: "shape" | "ordinal" | "marker";
  ordinal: number;
};

export class GenerationResponseValidationError extends Error {
  constructor(
    readonly category: ValidationCategory,
    readonly detail?: TurnCounts | TurnIdentityFailure,
  ) {
    super();
  }
}

export function validationFailure(
  category: ValidationCategory,
  detail?: TurnCounts | TurnIdentityFailure,
): Error {
  return new GenerationResponseValidationError(category, detail);
}

export function validationCategoryText(
  error: GenerationResponseValidationError,
): string {
  if (error.detail && "expected" in error.detail) {
    return `${error.category} (expected ${error.detail.expected}, received ${error.detail.received})`;
  }
  if (error.detail) {
    return `${error.category} (${error.detail.field} at target ordinal ${error.detail.ordinal})`;
  }
  return error.category;
}

export function hasExactKeys(
  value: Record<string, unknown>,
  expected: string[],
): boolean {
  const keys = Object.keys(value).sort();
  const sortedExpected = [...expected].sort();
  return (
    keys.length === sortedExpected.length
    && keys.every((key, index) => key === sortedExpected[index])
  );
}

export function isPlainRecord(
  value: unknown,
): value is Record<string, unknown> {
  if (!isRecord(value) || Array.isArray(value)) return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

export class GenerationMarkerNonceCollisionError extends Error {
  constructor() {
    super("Compaction generation marker nonce collision.");
    this.name = "GenerationMarkerNonceCollisionError";
  }
}

export function safeStatus(error: unknown): number | undefined {
  for (const item of diagnosticChain(error)) {
    const data = ownDataProperty(item, "data");
    const value =
      ownDataProperty(item, "status")
      ?? ownDataProperty(item, "statusCode")
      ?? ownDataProperty(data, "statusCode");
    if (
      typeof value === "number"
      && Number.isInteger(value)
      && value >= 100
      && value <= 599
    ) {
      return value;
    }
  }
  return undefined;
}

export function isContextOverflow(
  error: unknown,
  status: number | undefined,
): boolean {
  return (
    status === 413
    || diagnosticChain(error).some(item => {
      const categories = diagnosticCategories(item);
      return categories.some(category =>
        [
          "contextoverflowerror",
          "context_length_exceeded",
          "context_window_exceeded",
          "prompt_too_long",
          "input_too_long",
        ].includes(category),
      );
    })
  );
}

export function isOutputTruncation(value: unknown): boolean {
  const outputCategories = [
    "length",
    "max_token",
    "max_tokens",
    "max_output_tokens",
    "output_length",
    "output_length_exceeded",
    "messageoutputlengtherror",
  ];
  if (typeof value === "string" && outputCategories.includes(category(value))) {
    return true;
  }
  return diagnosticChain(value).some(item => {
    return diagnosticCategories(item).some(itemCategory =>
      outputCategories.includes(itemCategory),
    );
  });
}

export function diagnosticChain(value: unknown): unknown[] {
  const chain: unknown[] = [];
  const pending = [value];
  const seen = new Set<object>();
  while (pending.length > 0 && chain.length < 32) {
    const current = pending.shift();
    if (current === undefined) continue;
    if (
      (typeof current === "object" || typeof current === "function")
      && current !== null
    ) {
      if (seen.has(current)) continue;
      seen.add(current);
    }
    chain.push(current);
    pending.push(
      ownDataProperty(current, "cause"),
      ownDataProperty(current, "error"),
      ownDataProperty(current, "data"),
    );
  }
  return chain;
}

export function diagnosticCategories(value: unknown): string[] {
  return ["name", "code", "type", "reason"].flatMap(key => {
    const item = ownDataProperty(value, key);
    return typeof item === "string" ? [category(item)] : [];
  });
}

export function category(value: string): string {
  return value.trim().toLowerCase().replaceAll("-", "_");
}

export function ownDataProperty(value: unknown, key: string): unknown {
  if ((typeof value !== "object" && typeof value !== "function") || !value) {
    return undefined;
  }
  try {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    return descriptor && "value" in descriptor ? descriptor.value : undefined;
  } catch {
    return undefined;
  }
}

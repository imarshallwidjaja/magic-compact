// Planning, execution and loading of decision evaluation runs.
//
// A run directory holds `manifest.json`, written before any request, and
// `outcomes.jsonl`, one line appended per request as it settles. Aggregates
// are always recomputed from those records, so an interrupted run can still be
// reported. A request that left no record may have reached the provider and
// been billed before the process died, so its outcome is unknown; only a
// `not-attempted` record states that a request was never sent.
import {
  appendFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  writeFileSync,
} from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import { z } from "zod";
import {
  RETENTION_KEEP_THRESHOLD,
  RETENTION_MAX_CONCURRENCY,
  RETENTION_MODEL,
  type RetentionFetch,
  type RetentionRequest,
} from "../../src/retention/classifier";
import {
  buildCaseRequest,
  canonicalRequestSha256,
  datasetCases,
  DATASET_SPLITS,
  DECISION_EVALUATION_THRESHOLD,
  datasetSha256,
  DecisionEvalError,
  requestSetSha256,
  semanticQuestionSha256,
  sha256,
  type DecisionDataset,
  type EvalCase,
} from "./dataset";
import {
  FATAL_CATEGORIES,
  PROVIDER_IDS,
  PROVIDERS,
  returnedModelMatches,
  sendDecision,
  type FailureCategory,
  type ProviderAdapter,
  type Transport,
} from "./providers";

export const RESULT_FORMAT_VERSION = 1;
export const DEFAULT_CONCURRENCY = RETENTION_MAX_CONCURRENCY;
export const MAX_CONCURRENCY = 16;
export const MAX_REPEAT = 10;
// A guard against an accidental paid run over a large custom dataset.
export const DEFAULT_MAX_REQUESTS = 200;

const digest = z.string().regex(/^[0-9a-f]{64}$/);
const count = z.number().int().min(0);

const manifestSchema = z.strictObject({
  formatVersion: z.literal(RESULT_FORMAT_VERSION),
  startedAt: z.string(),
  provider: z.enum(PROVIDER_IDS),
  model: z.string().min(1),
  endpoint: z.string().min(1),
  dataset: z.strictObject({
    datasetID: z.string(),
    version: z.string(),
    split: z.enum(DATASET_SPLITS),
    sha256: digest,
    caseCount: count,
  }),
  prompt: z.strictObject({
    semanticQuestionSha256: digest,
    requestSetSha256: digest,
    translation: z
      .strictObject({ version: z.string(), sha256: digest })
      .nullable(),
  }),
  production: z.strictObject({
    jevModel: z.string(),
    keepThreshold: z.number(),
  }),
  threshold: z.number(),
  repeat: z.number().int().min(1),
  concurrency: z.number().int().min(1),
  timeoutMs: z.number().int().min(1),
  maxRequests: z.number().int().min(1),
  cases: z
    .array(
      z.strictObject({
        caseID: z.string(),
        canonicalRequestSha256: digest,
        providerRequestSha256: digest,
      }),
    )
    .min(1),
});
export type Manifest = z.infer<typeof manifestSchema>;

const usageSchema = z.strictObject({
  inputTokens: count.optional(),
  outputTokens: count.optional(),
});
const outcomeBase = {
  caseID: z.string(),
  repeat: z.number().int().min(1),
  canonicalRequestSha256: digest,
  providerRequestSha256: digest,
  latencyMs: count,
};
const outcomeSchema = z.discriminatedUnion("status", [
  z.strictObject({
    status: z.literal("scored"),
    ...outcomeBase,
    probability: z.number().min(0).max(1),
    requestedModel: z.string(),
    returnedModel: z.string(),
    usage: usageSchema.optional(),
  }),
  z.strictObject({
    status: z.literal("refused"),
    ...outcomeBase,
    reason: z.literal("provider-refusal"),
    requestedModel: z.string(),
    returnedModel: z.string(),
    usage: usageSchema.optional(),
  }),
  z.strictObject({
    status: z.literal("failed"),
    ...outcomeBase,
    category: z.enum([
      "auth",
      "rate-limit",
      "http",
      "timeout",
      "network",
      "invalid-response",
      "model-mismatch",
    ]),
    httpStatus: z.number().int().min(100).max(599).optional(),
  }),
  z.strictObject({
    status: z.literal("not-attempted"),
    caseID: z.string(),
    repeat: z.number().int().min(1),
    canonicalRequestSha256: digest,
    reason: z.enum(["stopped-after-auth", "stopped-after-rate-limit"]),
  }),
]);
export type Outcome = z.infer<typeof outcomeSchema>;

function manifestRequestSetSha256(
  entries: ReadonlyArray<{ caseID: string; canonicalRequestSha256: string }>,
): string {
  return requestSetSha256(
    entries.map(e => ({ caseID: e.caseID, sha256: e.canonicalRequestSha256 })),
  );
}

// Query strings can carry tokens, so records keep only origin and path.
function publicEndpoint(endpoint: string): string {
  const url = new URL(endpoint);
  return `${url.origin}${url.pathname}`;
}

export type RunOptions = {
  dataset: DecisionDataset;
  adapter: ProviderAdapter;
  model: string;
  threshold: number;
  repeat: number;
  concurrency: number;
  timeoutMs: number;
  maxRequests: number;
  endpoint: string;
};

type PlannedCase = {
  item: EvalCase;
  request: RetentionRequest;
  canonicalRequestSha256: string;
  body: string;
  providerRequestSha256: string;
};

// The singleton request for one case and what it is sent as, built only by the
// production request builder and the provider adapter. Planning and result
// loading both use it, so a stored hash is always checked against this.
function planCase(
  item: EvalCase,
  adapter: ProviderAdapter,
  model: string,
): PlannedCase {
  const request = buildCaseRequest(item);
  const body = adapter.translate(request, model);
  return {
    item,
    request,
    canonicalRequestSha256: canonicalRequestSha256(request),
    body,
    providerRequestSha256: sha256(body),
  };
}

export type RunPlan = {
  options: RunOptions;
  cases: PlannedCase[];
  // Repeat-major: every case once, then every case again.
  units: Array<{ planned: PlannedCase; repeat: number }>;
  manifest: Omit<Manifest, "startedAt">;
  translatedRequestSetSha256: string;
};

function integerOption(
  name: string,
  value: number,
  min: number,
  max: number,
): void {
  if (!Number.isInteger(value) || value < min || value > max) {
    throw new DecisionEvalError(
      `${name} must be an integer from ${min} to ${max}.`,
    );
  }
}

// Validates everything that can be checked without a key or the network, so a
// bad selection never reaches a paid request.
export function planRun(options: RunOptions): RunPlan {
  const { dataset, adapter, model } = options;
  if (
    !Number.isFinite(options.threshold)
    || options.threshold <= 0
    || options.threshold > 1
  ) {
    throw new DecisionEvalError("--threshold must be above 0 and at most 1.");
  }
  if (
    dataset.split === "heldout"
    && options.threshold !== DECISION_EVALUATION_THRESHOLD
  ) {
    throw new DecisionEvalError(
      `Held-out datasets are scored at the frozen threshold ${DECISION_EVALUATION_THRESHOLD}; --threshold is not available for them.`,
    );
  }
  integerOption("--repeat", options.repeat, 1, MAX_REPEAT);
  integerOption("--concurrency", options.concurrency, 1, MAX_CONCURRENCY);
  integerOption("--timeout-ms", options.timeoutMs, 1, 600_000);
  integerOption("--max-requests", options.maxRequests, 1, 1_000_000);
  adapter.checkModel(model);

  const cases = datasetCases(dataset).map(item =>
    planCase(item, adapter, model),
  );
  const units = Array.from({ length: options.repeat }, (_, index) =>
    cases.map(planned => ({ planned, repeat: index + 1 })),
  ).flat();
  if (units.length > options.maxRequests) {
    throw new DecisionEvalError(
      `The run plans ${units.length} requests (${cases.length} cases x ${options.repeat}), above --max-requests ${options.maxRequests}. Raise the cap or reduce --repeat.`,
    );
  }
  const entries = cases.map(planned => ({
    caseID: planned.item.caseID,
    canonicalRequestSha256: planned.canonicalRequestSha256,
    providerRequestSha256: planned.providerRequestSha256,
  }));
  return {
    options,
    cases,
    units,
    translatedRequestSetSha256: requestSetSha256(
      cases.map(planned => ({
        caseID: planned.item.caseID,
        sha256: planned.providerRequestSha256,
      })),
    ),
    manifest: {
      formatVersion: RESULT_FORMAT_VERSION,
      provider: adapter.id,
      model,
      endpoint: publicEndpoint(options.endpoint),
      dataset: {
        datasetID: dataset.datasetID,
        version: dataset.version,
        split: dataset.split,
        sha256: datasetSha256(dataset),
        caseCount: cases.length,
      },
      prompt: {
        semanticQuestionSha256: semanticQuestionSha256(),
        requestSetSha256: manifestRequestSetSha256(entries),
        translation: adapter.translation(),
      },
      production: {
        jevModel: RETENTION_MODEL,
        keepThreshold: RETENTION_KEEP_THRESHOLD,
      },
      threshold: options.threshold,
      repeat: options.repeat,
      concurrency: options.concurrency,
      timeoutMs: options.timeoutMs,
      maxRequests: options.maxRequests,
      cases: entries,
    },
  };
}

export function assertFreshOutput(out: string): void {
  if (existsSync(out)) {
    throw new DecisionEvalError(
      `Output path ${out} already exists. Results are never overwritten; choose a new directory.`,
    );
  }
}

export type ExecuteDeps = {
  apiKey: string;
  fetch: RetentionFetch;
  out: string;
  onOutcome?: (outcome: Outcome) => void;
};

// Runs every planned request once with at most `concurrency` in flight. There
// is no retry. A rejected key or a rate limit stops new admissions; requests
// already in flight finish and every unadmitted one is recorded as not
// attempted. Other failures, including refusals, leave the rest running.
export async function executeRun(
  plan: RunPlan,
  deps: ExecuteDeps,
): Promise<Outcome[]> {
  const { options, units } = plan;
  mkdirSync(dirname(resolve(deps.out)), { recursive: true });
  try {
    mkdirSync(deps.out);
  } catch {
    assertFreshOutput(deps.out);
    throw new DecisionEvalError(`Cannot create output directory ${deps.out}.`);
  }
  writeFileSync(
    join(deps.out, "manifest.json"),
    `${JSON.stringify({ ...plan.manifest, startedAt: new Date().toISOString() }, null, 2)}\n`,
    { flag: "wx" },
  );
  const outcomesPath = join(deps.out, "outcomes.jsonl");
  const outcomes: Outcome[] = [];
  const record = (outcome: Outcome) => {
    appendFileSync(outcomesPath, `${JSON.stringify(outcome)}\n`);
    outcomes.push(outcome);
    deps.onOutcome?.(outcome);
  };
  let next = 0;
  let stoppedBy: FailureCategory | undefined;
  const worker = async () => {
    while (stoppedBy === undefined && next < units.length) {
      const { planned, repeat } = units[next]!;
      next += 1;
      const transport: Transport = await sendDecision(
        options.adapter,
        planned.body,
        planned.request,
        {
          model: options.model,
          apiKey: deps.apiKey,
          endpoint: options.endpoint,
          timeoutMs: options.timeoutMs,
          fetch: deps.fetch,
        },
      );
      record({
        caseID: planned.item.caseID,
        repeat,
        canonicalRequestSha256: planned.canonicalRequestSha256,
        providerRequestSha256: planned.providerRequestSha256,
        ...transport,
      });
      if (
        transport.status === "failed"
        && FATAL_CATEGORIES.has(transport.category)
      ) {
        stoppedBy ??= transport.category;
      }
    }
  };
  await Promise.all(
    Array.from({ length: Math.min(options.concurrency, units.length) }, worker),
  );
  if (stoppedBy === "auth" || stoppedBy === "rate-limit") {
    for (const { planned, repeat } of units.slice(next)) {
      record({
        status: "not-attempted",
        caseID: planned.item.caseID,
        repeat,
        canonicalRequestSha256: planned.canonicalRequestSha256,
        reason: `stopped-after-${stoppedBy}`,
      });
    }
  }
  return outcomes;
}

export type LoadedRun = {
  name: string;
  manifest: Manifest;
  outcomes: Outcome[];
  // cases x repeat as planned in the manifest.
  planned: number;
};

// Complete means every planned request produced a probability or a refusal.
// Failed, not-attempted and unknown requests all make a run incomplete.
export function isComplete(run: LoadedRun): boolean {
  return (
    run.outcomes.length === run.planned
    && run.outcomes.every(o => o.status === "scored" || o.status === "refused")
  );
}

function readJson(path: string): unknown {
  try {
    return JSON.parse(readFileSync(path, "utf8"));
  } catch {
    return undefined;
  }
}

// Reads a run directory and refuses anything that is not a consistent result
// for `dataset` and for the requests this checkout builds from it. Aggregates
// are never trusted from the files.
export function loadRun(dir: string, dataset: DecisionDataset): LoadedRun {
  const name = basename(resolve(dir));
  const reject = (reason: string): never => {
    throw new DecisionEvalError(`${dir} ${reason}`);
  };
  const parsed = manifestSchema.safeParse(readJson(join(dir, "manifest.json")));
  if (!parsed.success) {
    return reject(
      "is not a decision evaluation result (manifest.json is missing or invalid).",
    );
  }
  const manifest = parsed.data;
  const cases = datasetCases(dataset);
  if (
    manifest.dataset.sha256 !== datasetSha256(dataset)
    || manifest.dataset.datasetID !== dataset.datasetID
    || manifest.dataset.version !== dataset.version
    || manifest.dataset.split !== dataset.split
  ) {
    return reject(
      `was produced from a different dataset (${manifest.dataset.datasetID} ${manifest.dataset.version}, sha256 ${manifest.dataset.sha256.slice(0, 12)}) than the one loaded (${dataset.datasetID} ${dataset.version}). Pass the dataset it used with --dataset.`,
    );
  }
  // The manifest and every outcome are checked against what this checkout
  // builds for the loaded dataset. Agreeing with itself is not enough: a
  // consistent manifest or outcome could belong to another model, provider or
  // question wording.
  const adapter = PROVIDERS[manifest.provider];
  try {
    adapter.checkModel(manifest.model);
  } catch {
    return reject(
      `plans model ${manifest.model}, which this checkout does not accept for ${adapter.id}.`,
    );
  }
  if (manifest.prompt.semanticQuestionSha256 !== semanticQuestionSha256()) {
    return reject(
      "was produced for a different production question than this checkout builds, so its answers cannot be read as answers to the current question.",
    );
  }
  const translation = adapter.translation();
  const describeTranslation = (
    value: { version: string; sha256: string } | null,
  ) => (value ? `${value.version} (${value.sha256.slice(0, 12)})` : "none");
  if (
    manifest.prompt.translation?.version !== translation?.version
    || manifest.prompt.translation?.sha256 !== translation?.sha256
  ) {
    return reject(
      `used request translation ${describeTranslation(manifest.prompt.translation)}, but this checkout implements ${describeTranslation(translation)} for ${adapter.id}.`,
    );
  }
  const caseIDs = new Set(manifest.cases.map(entry => entry.caseID));
  if (
    caseIDs.size !== manifest.cases.length
    || caseIDs.size !== cases.length
    || !cases.every(item => caseIDs.has(item.caseID))
  ) {
    return reject("does not plan exactly the cases of its dataset.");
  }
  const planned = new Map(
    cases.map(item => [item.caseID, planCase(item, adapter, manifest.model)]),
  );
  if (
    manifest.cases.some(entry => {
      const expected = planned.get(entry.caseID)!;
      return (
        entry.canonicalRequestSha256 !== expected.canonicalRequestSha256
        || entry.providerRequestSha256 !== expected.providerRequestSha256
      );
    })
  ) {
    return reject(
      `plans requests that this checkout does not build for ${adapter.id} ${manifest.model} from this dataset.`,
    );
  }
  if (
    manifestRequestSetSha256(manifest.cases)
    !== manifest.prompt.requestSetSha256
  ) {
    return reject("has a request set that does not match its plan.");
  }
  if (
    manifest.dataset.split === "heldout"
    && manifest.threshold !== DECISION_EVALUATION_THRESHOLD
  ) {
    return reject(
      `scored a held-out dataset at threshold ${manifest.threshold}, not the frozen ${DECISION_EVALUATION_THRESHOLD}.`,
    );
  }
  const outcomes: Outcome[] = [];
  const seen = new Set<string>();
  const path = join(dir, "outcomes.jsonl");
  const lines = existsSync(path)
    ? readFileSync(path, "utf8")
        .split("\n")
        .filter(line => line !== "")
    : [];
  lines.forEach((line, index) => {
    let value: unknown;
    try {
      value = JSON.parse(line);
    } catch {
      return reject(`has an unreadable outcome record at line ${index + 1}.`);
    }
    const parsedOutcome = outcomeSchema.safeParse(value);
    if (!parsedOutcome.success) {
      return reject(`has an invalid outcome record at line ${index + 1}.`);
    }
    const outcome = parsedOutcome.data;
    const misplaced = (why: string): never =>
      reject(
        `has an outcome record at line ${index + 1} that does not belong to its plan (${why}).`,
      );
    const key = `${outcome.caseID}@${outcome.repeat}`;
    const expected = planned.get(outcome.caseID);
    if (!expected || outcome.repeat > manifest.repeat || seen.has(key)) {
      return misplaced("case or repeat");
    }
    if (outcome.canonicalRequestSha256 !== expected.canonicalRequestSha256) {
      return misplaced("canonical request");
    }
    if (outcome.status === "scored" || outcome.status === "refused") {
      if (outcome.requestedModel !== manifest.model) {
        return misplaced("requested model");
      }
      if (!returnedModelMatches(manifest.model, outcome.returnedModel)) {
        return misplaced("returned model");
      }
    }
    if (
      outcome.status !== "not-attempted"
      && outcome.providerRequestSha256 !== expected.providerRequestSha256
    ) {
      return misplaced("provider request");
    }
    seen.add(key);
    outcomes.push(outcome);
  });
  return {
    name,
    manifest,
    outcomes,
    planned: manifest.cases.length * manifest.repeat,
  };
}

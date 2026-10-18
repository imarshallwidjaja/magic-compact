// Metrics and comparison text, computed from run records and the dataset
// labels. Requests without a probability (refused, failed, not attempted,
// unknown) are counted in coverage and never scored as 0 or as a keep. A
// request with no record at all is unknown: it may have been sent and billed
// before the run died, which is different from a recorded not-attempted.
import { RETENTION_KEEP_THRESHOLD } from "../../src/retention/classifier";
import {
  DECISION_EVALUATION_THRESHOLD,
  DecisionEvalError,
  datasetCases,
  type DecisionDataset,
} from "./dataset";
import type { LoadedRun, Outcome } from "./runner";

type Scored = Extract<Outcome, { status: "scored" }>;
type Answered = Extract<Outcome, { status: "scored" | "refused" }>;

export type RunMetrics = {
  planned: number;
  scored: number;
  refused: number;
  failed: number;
  // Recorded as never sent, after a rejected key or a rate limit.
  notAttempted: number;
  // No record at all: possibly sent and billed, outcome not known.
  unknown: number;
  tp: number;
  fp: number;
  fn: number;
  tn: number;
  precision: number | null;
  recall: number | null;
  specificity: number | null;
  f1: number | null;
  // Correct decisions among scored requests.
  accuracy: number | null;
  brier: number | null;
  auc: number | null;
  // Useful content that would be trimmed, and wasteful content that would be
  // kept, for the requests that produced a probability.
  falsePrunes: Array<{ caseID: string; repeat: number; probability: number }>;
  falseKeeps: Array<{ caseID: string; repeat: number; probability: number }>;
  latencyP50Ms: number | null;
  latencyP90Ms: number | null;
  // Provider-reported counts, each dimension on its own. A total is null when
  // no response reported that dimension, and a subtotal when only some did;
  // the reporting counts say which.
  usage: {
    eligibleResponses: number;
    inputTokens: bigint | null;
    inputReporting: number;
    outputTokens: bigint | null;
    outputReporting: number;
  };
  // Only with repeats: spread of one case's probability across its repeats.
  stability: { cases: number; maxSpread: number; flips: number } | null;
  failures: Record<string, number>;
  returnedModels: string[];
};

function ratio(numerator: number, denominator: number): number | null {
  return denominator === 0 ? null : numerator / denominator;
}

// Nearest-rank percentile of an ascending list.
function percentile(sorted: number[], fraction: number): number | null {
  return sorted.length === 0
    ? null
    : sorted[Math.ceil(fraction * sorted.length) - 1]!;
}

// Probability that a useful case outranks a not-useful one; ties count half.
function areaUnderCurve(scored: Scored[], labels: Map<string, boolean>) {
  const positives = scored.filter(s => labels.get(s.caseID));
  const negatives = scored.filter(s => !labels.get(s.caseID));
  if (positives.length === 0 || negatives.length === 0) return null;
  let wins = 0;
  for (const positive of positives) {
    for (const negative of negatives) {
      if (positive.probability > negative.probability) wins += 1;
      else if (positive.probability === negative.probability) wins += 0.5;
    }
  }
  return wins / (positives.length * negatives.length);
}

export function computeMetrics(
  run: LoadedRun,
  labels: Map<string, boolean>,
  threshold: number,
): RunMetrics {
  const scored = run.outcomes.filter((o): o is Scored => o.status === "scored");
  const answered = run.outcomes.filter(
    (o): o is Answered => o.status === "scored" || o.status === "refused",
  );
  const refused = run.outcomes.filter(o => o.status === "refused");
  const failed = run.outcomes.filter(o => o.status === "failed");
  const notAttempted = run.outcomes.filter(o => o.status === "not-attempted");
  let tp = 0;
  let fp = 0;
  let fn = 0;
  let tn = 0;
  let squared = 0;
  const falsePrunes: RunMetrics["falsePrunes"] = [];
  const falseKeeps: RunMetrics["falseKeeps"] = [];
  for (const outcome of scored) {
    const gold = labels.get(outcome.caseID)!;
    const keep = outcome.probability >= threshold;
    squared += (outcome.probability - (gold ? 1 : 0)) ** 2;
    const entry = {
      caseID: outcome.caseID,
      repeat: outcome.repeat,
      probability: outcome.probability,
    };
    if (keep && gold) tp += 1;
    else if (keep) {
      fp += 1;
      falseKeeps.push(entry);
    } else if (gold) {
      fn += 1;
      falsePrunes.push(entry);
    } else tn += 1;
  }
  const precision = ratio(tp, tp + fp);
  const recall = ratio(tp, tp + fn);
  const latencies = answered.map(o => o.latencyMs).sort((a, b) => a - b);
  // Failed transport has no provider usage; it is unknown, not 0.
  const usageTotal = (pick: "inputTokens" | "outputTokens") => {
    const reported = answered.flatMap(o => {
      const count = o.usage?.[pick];
      return count === undefined ? [] : [BigInt(count)];
    });
    return {
      total: reported.length === 0 ? null : reported.reduce((a, b) => a + b),
      reporting: reported.length,
    };
  };
  const input = usageTotal("inputTokens");
  const output = usageTotal("outputTokens");
  const failures: Record<string, number> = {};
  for (const outcome of failed) {
    if (outcome.status === "failed") {
      failures[outcome.category] = (failures[outcome.category] ?? 0) + 1;
    }
  }
  const byCase = new Map<string, number[]>();
  for (const outcome of scored) {
    byCase.set(outcome.caseID, [
      ...(byCase.get(outcome.caseID) ?? []),
      outcome.probability,
    ]);
  }
  const repeated = [...byCase.values()].filter(values => values.length > 1);
  return {
    planned: run.planned,
    scored: scored.length,
    refused: refused.length,
    failed: failed.length,
    notAttempted: notAttempted.length,
    unknown: run.planned - run.outcomes.length,
    tp,
    fp,
    fn,
    tn,
    precision,
    recall,
    specificity: ratio(tn, tn + fp),
    // 2TP / (2TP + FP + FN): 0 when scored decisions had no true keep,
    // undefined only when there was nothing to count.
    f1: ratio(2 * tp, 2 * tp + fp + fn),
    accuracy: ratio(tp + tn, scored.length),
    brier: ratio(squared, scored.length),
    auc: areaUnderCurve(scored, labels),
    falsePrunes,
    falseKeeps,
    latencyP50Ms: percentile(latencies, 0.5),
    latencyP90Ms: percentile(latencies, 0.9),
    usage: {
      eligibleResponses: answered.length,
      inputTokens: input.total,
      inputReporting: input.reporting,
      outputTokens: output.total,
      outputReporting: output.reporting,
    },
    stability:
      run.manifest.repeat > 1 && repeated.length > 0
        ? {
            cases: repeated.length,
            maxSpread: Math.max(
              ...repeated.map(v => Math.max(...v) - Math.min(...v)),
            ),
            flips: repeated.filter(
              v => v.some(p => p >= threshold) && v.some(p => p < threshold),
            ).length,
          }
        : null,
    failures,
    returnedModels: [
      ...new Set(
        scored
          .map(o => o.returnedModel)
          .concat(
            run.outcomes.flatMap(o =>
              o.status === "refused" ? [o.returnedModel] : [],
            ),
          ),
      ),
    ].sort(),
  };
}

function number(value: number | null, digits = 3): string {
  return value === null ? "n/a" : value.toFixed(digits);
}

// A total says how many eligible responses reported it; fewer than all makes
// it a subtotal, and none leaves it unavailable.
function usageCell(
  total: bigint | null,
  reporting: number,
  eligible: number,
): string {
  if (total === null) return `n/a (0/${eligible})`;
  return `${reporting < eligible ? "subtotal " : ""}${total} (${reporting}/${eligible})`;
}

function table(rows: string[][]): string {
  const widths = rows[0]!.map((_, column) =>
    Math.max(...rows.map(row => row[column]!.length)),
  );
  return rows
    .map(row =>
      row
        .map((cell, column) => cell.padEnd(widths[column]!))
        .join("  ")
        .trimEnd(),
    )
    .join("\n");
}

function group<T>(items: T[], key: (item: T) => string): T[][] {
  const groups = new Map<string, T[]>();
  for (const item of items) {
    groups.set(key(item), [...(groups.get(key(item)) ?? []), item]);
  }
  return [...groups.values()];
}

// `thresholdOverride` re-reads the stored probabilities at another cutoff; it
// is refused for held-out datasets, whose cutoff stays frozen.
export function renderReport(
  runs: LoadedRun[],
  dataset: DecisionDataset,
  thresholdOverride?: number,
): string {
  if (thresholdOverride !== undefined) {
    if (
      !Number.isFinite(thresholdOverride)
      || thresholdOverride <= 0
      || thresholdOverride > 1
    ) {
      throw new DecisionEvalError("--threshold must be above 0 and at most 1.");
    }
    if (dataset.split === "heldout") {
      throw new DecisionEvalError(
        `Held-out results are reported at the frozen threshold ${DECISION_EVALUATION_THRESHOLD}; --threshold is not available for them.`,
      );
    }
  }
  const labels = new Map(datasetCases(dataset).map(c => [c.caseID, c.keep]));
  const effective = (run: LoadedRun) =>
    thresholdOverride ?? run.manifest.threshold;
  const groups = group(
    runs,
    run =>
      `${run.manifest.prompt.semanticQuestionSha256}|${run.manifest.prompt.requestSetSha256}|${effective(run)}`,
  );
  const useful = [...labels.values()].filter(Boolean).length;
  const out: string[] = [
    `Dataset ${dataset.datasetID} ${dataset.version} (${dataset.split}): ${labels.size} cases, ${useful} labelled keep, ${labels.size - useful} labelled trim`,
    `Threshold is an evaluation reference. The production default keep threshold is ${RETENTION_KEEP_THRESHOLD} for the pinned Jev model and the server environment can override it; held-out results stay at ${DECISION_EVALUATION_THRESHOLD}. Probabilities from other providers or models are not calibrated to either.`,
  ];
  if (groups.length > 1) {
    out.push(
      "",
      `The ${runs.length} runs used different thresholds or prompts. They are listed in separate groups and must not be compared across groups.`,
    );
  }
  groups.forEach((members, index) => {
    const first = members[0]!;
    out.push(
      "",
      `Group ${index + 1}: threshold ${effective(first)}, question ${first.manifest.prompt.semanticQuestionSha256.slice(0, 12)}, request set ${first.manifest.prompt.requestSetSha256.slice(0, 12)}`,
    );
    const rows: string[][] = [
      [
        "run",
        "provider",
        "model",
        "scored",
        "refused",
        "failed",
        "not attempted",
        "unknown",
        "TP",
        "FP",
        "FN",
        "TN",
        "precision",
        "recall",
        "specificity",
        "F1",
        "accuracy",
        "brier",
        "auc",
        "p50ms",
        "p90ms",
        "inTok",
        "outTok",
      ],
    ];
    const notes: string[] = [];
    for (const run of members) {
      const m = computeMetrics(run, labels, effective(run));
      rows.push([
        run.name,
        run.manifest.provider,
        run.manifest.model,
        `${m.scored}/${m.planned}`,
        String(m.refused),
        String(m.failed),
        String(m.notAttempted),
        String(m.unknown),
        String(m.tp),
        String(m.fp),
        String(m.fn),
        String(m.tn),
        number(m.precision),
        number(m.recall),
        number(m.specificity),
        number(m.f1),
        number(m.accuracy),
        number(m.brier),
        number(m.auc),
        m.latencyP50Ms === null ? "n/a" : String(m.latencyP50Ms),
        m.latencyP90Ms === null ? "n/a" : String(m.latencyP90Ms),
        usageCell(
          m.usage.inputTokens,
          m.usage.inputReporting,
          m.usage.eligibleResponses,
        ),
        usageCell(
          m.usage.outputTokens,
          m.usage.outputReporting,
          m.usage.eligibleResponses,
        ),
      ]);
      if (m.scored < m.planned) {
        notes.push(
          `${run.name}: incomplete coverage. Metrics use ${m.scored} of ${m.planned} planned requests; refused, failed, not attempted and unknown requests are not scored as trims or keeps.`,
        );
      }
      if (m.unknown > 0) {
        notes.push(
          `${run.name}: ${m.unknown} ${m.unknown === 1 ? "request has" : "requests have"} no outcome record. The run left no record of ${m.unknown === 1 ? "it" : "them"} (it may have been interrupted), so ${m.unknown === 1 ? "it" : "they"} may have been sent and billed; the outcome is unknown. Nothing was retried.`,
        );
      }
      if (m.returnedModels.some(model => model !== run.manifest.model)) {
        notes.push(
          `${run.name}: provider reported ${m.returnedModels.join(", ")}.`,
        );
      }
      const failures = Object.entries(m.failures)
        .sort()
        .map(([category, n]) => `${category} ${n}`);
      if (failures.length > 0)
        notes.push(`${run.name}: failures ${failures.join(", ")}.`);
      if (m.falsePrunes.length > 0) {
        notes.push(
          `${run.name}: false prunes (useful content below threshold): ${m.falsePrunes.map(e => `${e.caseID} ${e.probability}`).join(", ")}.`,
        );
      }
      if (m.falseKeeps.length > 0) {
        notes.push(
          `${run.name}: false keeps (wasted tokens): ${m.falseKeeps.map(e => `${e.caseID} ${e.probability}`).join(", ")}.`,
        );
      }
      if (m.stability) {
        notes.push(
          `${run.name}: across repeats, max probability spread ${number(m.stability.maxSpread)} over ${m.stability.cases} cases, ${m.stability.flips} changed decision. Repeats measure stability, not independent samples.`,
        );
      }
    }
    out.push(table(rows), ...notes);
  });
  return out.join("\n");
}

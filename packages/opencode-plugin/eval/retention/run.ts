// Retention evaluation against TypeSafe Jev. check and metrics are offline;
// score makes paid calls only after the entire frozen cohort passes parity.
//
//   bun packages/opencode-plugin/eval/retention/run.ts check <external-input.json>
//   bun packages/opencode-plugin/eval/retention/run.ts metrics
//   bun packages/opencode-plugin/eval/retention/run.ts score <external-input.json> <split> <order> <output.json>
//
// Historical proxy-cohort interface (not the current frozen labels):
//   bun packages/opencode-plugin/eval/retention/run.ts run <name> <split> [batch] [order] [variant] [context]
//   bun packages/opencode-plugin/eval/retention/run.ts report <name> [<name> ...]
//
// Every snapshot is one classifier invocation through `classifyRetention`, so
// the evaluation exercises the shipping request format, budgets, validation,
// and concurrency. Raw answers are written to the scratch directory.
//
// `context` selects the session summary:
//   generated    reads SCRATCH_DIR/summaries/<snapshotID>.md, a summary made
//                by `generateSessionSummary` (src/retention/summary.ts) for
//                the snapshot history. This is the production input.
//   proxy        joins the mechanical provisional summaries. Historical only:
//                the recorded results in docs/RetentionEvaluation.md used an
//                earlier per-summary list format, so proxy runs are not
//                comparable to them or to production.
//   latest-only  sends no summary, to measure summary dependence.
import { createHash } from "node:crypto";
import {
  existsSync,
  mkdirSync,
  readFileSync,
  rmdirSync,
  unlinkSync,
  writeFileSync,
} from "node:fs";
import { join } from "node:path";
import {
  buildRetentionRequests,
  classifyRetention,
  RETENTION_MODEL,
  type RetentionCandidate,
  type RetentionContext,
  type RetentionFetch,
} from "../../src/retention/classifier";
import { countGptTokens } from "../../src/stats/tokenize";
import { SCRATCH_DIR, type EvalSnapshot } from "./dataset";

// The keep threshold of the recorded native-summary results. It belongs to
// that protocol and stays independent of the runtime default and its
// environment override, so the historical metrics keep their meaning.
const NATIVE_EVALUATION_KEEP_THRESHOLD = 0.8;

type Label = {
  partID: string;
  tool: string;
  stratum: string;
  useful: boolean;
  category: string;
  ambiguous: boolean;
  hiddenMiddle?: boolean;
};
type Labels = {
  snapshots: Array<{ snapshotID: string; split: string; labels: Label[] }>;
};
const ORDERS = ["chronological", "reversed"] as const;
type Order = (typeof ORDERS)[number];
const CONTEXT_MODES = ["generated", "proxy", "latest-only"] as const;
type ContextMode = (typeof CONTEXT_MODES)[number];
type RunRecord = {
  name: string;
  split: string;
  batchSize: number;
  order: Order;
  variant: string;
  context: ContextMode;
  startedAt: string;
  model: string;
  scores: Record<string, number>;
  invocations: Array<{
    snapshotID: string;
    requests: number;
    inputTokens: number;
    latencyMs: number;
    requestLatencyMs: number[];
  }>;
};

const RUNS_DIR = join(SCRATCH_DIR, "runs");
const labels = JSON.parse(
  readFileSync(new URL("./labels.json", import.meta.url), "utf8"),
) as Labels;

// Dev-only question variants, applied to the built request body. The shipped
// wording lives in the classifier module.
const VARIANTS: Record<
  string,
  (question: { instructions: string; criteria?: unknown }) => void
> = {
  shipped: () => {},
  // Pre-tuning wording, kept to reproduce the development comparison.
  "help-wording": question => {
    const path = question.instructions.match(/`candidates\[\d+\]`/)![0];
    question.instructions = `Would keeping the full input and output of the tool call ${path} in context concretely help the assistant continue the current task described by \`task.latestUserRequest\` and \`task.sessionSummary\`?`;
    question.criteria = {
      true: "The input or output holds specific content the assistant is likely to need while continuing this task: results or findings of work done for it, current contents of files or artifacts being worked on, or exact identifiers, links, commands, errors, or decisions it will reuse. This holds even if the content could be fetched again.",
      false:
        "The content is routine tool bookkeeping, empty, generic instructions, stale or superseded by later work, about a finished or different task, or only related by topic.",
    };
  },
  "help-wording-no-criteria": question => {
    VARIANTS["help-wording"]!(question);
    delete question.criteria;
  },
};

function loadDataset(scratch = SCRATCH_DIR): EvalSnapshot[] {
  return JSON.parse(
    readFileSync(join(scratch, "dataset.json"), "utf8"),
  ) as EvalSnapshot[];
}

function sessionSummary(
  snapshot: EvalSnapshot,
  context: ContextMode,
  scratch: string,
): string {
  if (context === "latest-only") return "";
  if (context === "proxy") return snapshot.provisionalSummaries.join("\n\n");
  return readFileSync(
    join(scratch, "summaries", `${snapshot.snapshotID}.md`),
    "utf8",
  );
}

// The output of a paid run, claimed before its first paid request. After
// each paid snapshot, everything scored so far is written in the output's
// own format to `<output>.partial/snapshot-<k>-of-<n>.json`, in a directory
// this run creates and no other run reuses, so an interruption keeps each
// completed snapshot. Nothing resumes or retries from a checkpoint, and a
// new run needs a new output. The output is written once, never over an
// existing file, after every snapshot, and the checkpoints are then removed.
function paidOutput(output: string) {
  const partial = `${output}.partial`;
  if (existsSync(output) || existsSync(partial)) {
    throw new Error("Output or its checkpoint directory already exists.");
  }
  const checkpoints: string[] = [];
  return {
    open: () => mkdirSync(partial),
    checkpoint(index: number, total: number, text: string) {
      const checkpoint = join(
        partial,
        `snapshot-${index + 1}-of-${total}.json`,
      );
      writeFileSync(checkpoint, text, { flag: "wx" });
      checkpoints.push(checkpoint);
    },
    finish(text: string) {
      writeFileSync(output, text, { flag: "wx" });
      for (const checkpoint of checkpoints) unlinkSync(checkpoint);
      rmdirSync(partial);
    },
  };
}

// `run <name> <split> [batch] [order] [variant] [context]`, checked before
// any key is read or request sent. `variant` is checked by `run`.
export function parseHistoricalRunArguments(args: string[]) {
  const [
    name,
    split,
    batch = "1",
    order = "chronological",
    variant = "shipped",
    context = "generated",
  ] = args;
  if (!name || !split) {
    throw new Error(
      "usage: run.ts run <name> <split> [batch] [order] [variant] [context]",
    );
  }
  // The classifier's own batch rule, checked here before the key is read.
  const batchSize = Number(batch);
  if (!Number.isInteger(batchSize) || batchSize < 1) {
    throw new Error("batch must be a positive integer.");
  }
  const parsedOrder = ORDERS.find(value => value === order);
  if (!parsedOrder) {
    throw new Error(`order must be one of: ${ORDERS.join(", ")}.`);
  }
  const parsedContext = CONTEXT_MODES.find(value => value === context);
  if (!parsedContext) {
    throw new Error(`context must be one of: ${CONTEXT_MODES.join(", ")}.`);
  }
  return {
    name,
    split,
    batchSize,
    order: parsedOrder,
    variant,
    context: parsedContext,
  };
}

function toCandidates(snapshot: EvalSnapshot): RetentionCandidate[] {
  return snapshot.candidates.map(candidate => ({
    id: candidate.id,
    tool: candidate.tool,
    input: candidate.input,
    output: candidate.output,
    turnRequest: candidate.turnRequest,
    requestsBeforeLatest: -candidate.groupOffset,
  }));
}

export type PaidDeps = {
  // The command line passes TYPESAFE_API_KEY.
  apiKey: string | undefined;
  fetch: RetentionFetch;
};

// Each snapshot is one paid classifier invocation, checkpointed as
// `paidOutput` describes; the record is written to `runs/<name>.json` under
// the scratch directory.
export async function run(
  name: string,
  split: string,
  batchSize: number,
  order: Order,
  variant: string,
  context: ContextMode,
  deps: PaidDeps & { scratch?: string },
): Promise<void> {
  const scratch = deps.scratch ?? SCRATCH_DIR;
  const runs = join(scratch, "runs");
  const output = paidOutput(join(runs, `${name}.json`));
  const { apiKey } = deps;
  if (!apiKey) throw new Error("TYPESAFE_API_KEY is not set.");
  const mutate = VARIANTS[variant];
  if (!mutate) throw new Error(`Unknown variant ${variant}.`);
  const record: RunRecord = {
    name,
    split,
    batchSize,
    order,
    variant,
    context,
    startedAt: new Date().toISOString(),
    model: "",
    scores: {},
    invocations: [],
  };
  const snapshots = loadDataset(scratch).filter(s => s.split === split);
  mkdirSync(runs, { recursive: true });
  output.open();
  for (const [index, snapshot] of snapshots.entries()) {
    const candidates = toCandidates(snapshot);
    if (order === "reversed") candidates.reverse();
    const requestLatencyMs: number[] = [];
    const timedFetch: RetentionFetch = async (url, init) => {
      const body = JSON.parse(String(init.body)) as {
        questions: Record<string, { instructions: string; criteria?: unknown }>;
      };
      for (const question of Object.values(body.questions)) mutate(question);
      const started = performance.now();
      const response = await deps.fetch(url, {
        ...init,
        body: JSON.stringify(body),
      });
      const clone = response.clone();
      const payload = (await clone.json().catch(() => ({}))) as {
        model?: string;
      };
      requestLatencyMs.push(Math.round(performance.now() - started));
      record.model = payload.model ?? record.model;
      return response;
    };
    const started = performance.now();
    const result = await classifyRetention(
      {
        latestUserRequest: snapshot.latestUserRequest,
        sessionSummary: sessionSummary(snapshot, context, scratch),
      },
      candidates,
      { apiKey, batchSize, fetch: timedFetch },
    );
    for (const [id, decision] of result.decisions) {
      record.scores[`${snapshot.snapshotID}:${id}`] = decision.useful;
    }
    record.invocations.push({
      snapshotID: snapshot.snapshotID,
      requests: result.requests,
      inputTokens: result.inputTokens,
      latencyMs: Math.round(performance.now() - started),
      requestLatencyMs,
    });
    console.log(
      `${snapshot.snapshotID}\t${result.requests} requests\t${result.inputTokens} input tokens\t${record.invocations.at(-1)!.latencyMs} ms`,
    );
    output.checkpoint(index, snapshots.length, JSON.stringify(record, null, 1));
  }
  output.finish(JSON.stringify(record, null, 1));
}

type Row = Label & { key: string; snapshotID: string; tokens: number };

function rows(split: string): Row[] {
  const dataset = new Map(loadDataset().map(s => [s.snapshotID, s]));
  return labels.snapshots
    .filter(s => s.split === split)
    .flatMap(s =>
      s.labels.map(label => {
        const candidate = dataset
          .get(s.snapshotID)!
          .candidates.find(c => c.id === label.partID)!;
        return {
          ...label,
          key: `${s.snapshotID}:${label.partID}`,
          snapshotID: s.snapshotID,
          tokens:
            countGptTokens(JSON.stringify(candidate.input))
            + countGptTokens(candidate.output),
        };
      }),
    );
}

function confusion(items: Row[], scores: Record<string, number>, t: number) {
  let tp = 0;
  let fp = 0;
  let fn = 0;
  let tn = 0;
  let keptTokens = 0;
  for (const item of items) {
    const keep = scores[item.key]! >= t;
    if (keep) keptTokens += item.tokens;
    if (keep && item.useful) tp += 1;
    else if (keep) fp += 1;
    else if (item.useful) fn += 1;
    else tn += 1;
  }
  const totalTokens = items.reduce((sum, item) => sum + item.tokens, 0);
  return { tp, fp, fn, tn, keptTokens, totalTokens };
}

function pct(n: number, d: number): string {
  return d === 0 ? "n/a" : `${((100 * n) / d).toFixed(1)}%`;
}

function report(names: string[]): void {
  const records = names.map(
    name =>
      JSON.parse(
        readFileSync(join(RUNS_DIR, `${name}.json`), "utf8"),
      ) as RunRecord,
  );
  const first = records[0]!;
  const items = rows(first.split);
  for (const record of records) {
    const missing = items.filter(item => !(item.key in record.scores));
    if (missing.length > 0) throw new Error(`${record.name} lacks scores.`);
  }
  const thresholds = [0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95];
  for (const record of records) {
    const useful = items.filter(i => i.useful).length;
    const positives = items.filter(i => i.useful);
    const negatives = items.filter(i => !i.useful);
    let wins = 0;
    for (const p of positives)
      for (const n of negatives) {
        const d = record.scores[p.key]! - record.scores[n.key]!;
        wins += d > 0 ? 1 : d === 0 ? 0.5 : 0;
      }
    const auc = wins / (positives.length * negatives.length);
    console.log(
      `\n## ${record.name} (${record.split}, batch ${record.batchSize}, ${record.order}, ${record.variant}, ${record.context}, ${record.model}) n=${items.length} useful=${useful} AUC=${auc.toFixed(3)}`,
    );
    console.log(
      "t\tTP\tFP\tFN\tTN\tprecision\trecall\tkept-tokens\tunambiguous P/R",
    );
    for (const t of thresholds) {
      const c = confusion(items, record.scores, t);
      const clear = confusion(
        items.filter(i => !i.ambiguous),
        record.scores,
        t,
      );
      console.log(
        `${t}\t${c.tp}\t${c.fp}\t${c.fn}\t${c.tn}\t${pct(c.tp, c.tp + c.fp)}\t${pct(c.tp, c.tp + c.fn)}\t${pct(c.keptTokens, c.totalTokens)}\t${pct(clear.tp, clear.tp + clear.fp)} / ${pct(clear.tp, clear.tp + clear.fn)}`,
      );
    }
    const latencies = record.invocations
      .flatMap(i => i.requestLatencyMs)
      .sort((a, b) => a - b);
    const tokens = record.invocations.reduce((s, i) => s + i.inputTokens, 0);
    console.log(
      `requests=${latencies.length} inputTokens=${tokens} cost=$${((tokens * 0.042) / 1e6).toFixed(5)} requestLatency p50=${latencies[Math.floor(latencies.length / 2)]}ms p90=${latencies[Math.floor(latencies.length * 0.9)]}ms invocationLatency=${record.invocations.map(i => i.latencyMs).join(",")}ms`,
    );
  }
  for (let i = 1; i < records.length; i += 1) {
    const a = first.scores;
    const b = records[i]!.scores;
    const diffs = items.map(item => Math.abs(a[item.key]! - b[item.key]!));
    const flips = (t: number) =>
      items.filter(item => a[item.key]! >= t !== b[item.key]! >= t).length;
    console.log(
      `\nstability ${first.name} vs ${records[i]!.name}: max|diff|=${Math.max(...diffs).toFixed(3)} mean|diff|=${(diffs.reduce((s, d) => s + d, 0) / diffs.length).toFixed(3)} flips@0.7=${flips(0.7)} flips@0.8=${flips(0.8)} flips@0.9=${flips(0.9)}`,
    );
  }
}

// Current dual-annotator cohort. Private content is supplied externally;
// committed evidence contains only identities, hashes, labels and scores.
type NativeCase = {
  caseID: string;
  bodySha256: string;
  labels: { a: boolean | null; b: boolean | null; primary: boolean | null };
  scores: Record<string, number>;
};
type NativeSnapshot = {
  snapshotID: string;
  split: string;
  cases: NativeCase[];
};
type NativeInput = {
  snapshots: Array<{
    snapshotID: string;
    split: string;
    context: RetentionContext;
    candidates: RetentionCandidate[];
  }>;
};
const nativeEvidence = JSON.parse(
  readFileSync(new URL("./native-summary-cases.json", import.meta.url), "utf8"),
) as { snapshots: NativeSnapshot[] };
const sha = (text: string) => createHash("sha256").update(text).digest("hex");

export async function checkNative(
  input: NativeInput,
  frozen: { snapshots: NativeSnapshot[] } = nativeEvidence,
): Promise<void> {
  const expected = new Map(
    frozen.snapshots.flatMap(s => s.cases.map(c => [c.caseID, c] as const)),
  );
  const seen = new Set<string>();
  let compared = 0;
  for (const snapshot of input.snapshots) {
    const evidence = frozen.snapshots.find(
      s => s.snapshotID === snapshot.snapshotID && s.split === snapshot.split,
    );
    if (!evidence) throw new Error("Unknown snapshot or split.");
    const local = new Map<string, NativeCase>();
    for (const candidate of snapshot.candidates) {
      const row = expected.get(candidate.id);
      if (!row || seen.has(candidate.id) || !evidence.cases.includes(row)) {
        throw new Error("Unknown, repeated or misplaced case.");
      }
      seen.add(candidate.id);
      local.set(candidate.id, row);
    }
    const requests = buildRetentionRequests(
      snapshot.context,
      snapshot.candidates,
    );
    for (const request of requests) {
      const id = request.candidates[0]![1];
      if (
        request.candidates.length !== 1
        || sha(JSON.stringify(request.body)) !== local.get(id)?.bodySha256
      ) {
        throw new Error(
          "Request body differs from frozen evidence for case ID.",
        );
      }
    }
    // Capture the canonical send path in admission order and use distinct fake
    // answers to verify its decision-to-ID mapping, even for identical bodies.
    let sent = 0;
    const result = await classifyRetention(
      snapshot.context,
      snapshot.candidates,
      {
        apiKey: "parity-no-network",
        fetch: async (_url, init) => {
          const index = sent++;
          const request = requests[index];
          if (!request || String(init.body) !== JSON.stringify(request.body)) {
            throw new Error("Classifier send differs from validated request.");
          }
          compared++;
          return Response.json({
            model: RETENTION_MODEL,
            answers: {
              candidate_0: {
                type: "noul",
                noul: (index + 1) / (requests.length + 1),
              },
            },
          });
        },
      },
    );
    if (
      result.requests !== requests.length
      || sent !== requests.length
      || result.decisions.size !== local.size
    ) {
      throw new Error("Request cardinality differs.");
    }
    for (const [index, request] of requests.entries()) {
      if (
        result.decisions.get(request.candidates[0]![1])?.useful
        !== (index + 1) / (requests.length + 1)
      ) {
        throw new Error("Classifier decision differs from validated case ID.");
      }
    }
  }
  if (seen.size !== expected.size) throw new Error("Incomplete cohort.");
  console.log(
    `parity: ${compared}/${expected.size} frozen production bodies match; no network`,
  );
}

// Each snapshot is one paid classifier invocation, checkpointed as
// `paidOutput` describes.
export async function scoreNative(
  input: NativeInput,
  split: string,
  order: string,
  output: string,
  deps: PaidDeps & { frozen?: { snapshots: NativeSnapshot[] } },
): Promise<void> {
  if (!ORDERS.some(value => value === order)) throw new Error("Invalid order.");
  const paid = paidOutput(output);
  await checkNative(input, deps.frozen);
  const { apiKey } = deps;
  if (!apiKey) throw new Error("TYPESAFE_API_KEY is not set.");
  let snapshots = input.snapshots.filter(s => s.split === split);
  if (!snapshots.length) throw new Error("Empty split.");
  if (order === "reversed") snapshots = [...snapshots].reverse();
  const scores: Record<string, number> = {};
  const invocations: Array<{
    snapshotID: string;
    requests: number;
    inputTokens: number;
    elapsedMs: number;
    requestMs: number[];
  }> = [];
  const result = () =>
    JSON.stringify(
      {
        model: RETENTION_MODEL,
        threshold: NATIVE_EVALUATION_KEEP_THRESHOLD,
        split,
        order,
        scores,
        invocations,
      },
      null,
      2,
    );
  paid.open();
  for (const [index, s] of snapshots.entries()) {
    const started = performance.now();
    const requestMs: number[] = [];
    const classified = await classifyRetention(
      s.context,
      order === "reversed" ? [...s.candidates].reverse() : s.candidates,
      {
        apiKey,
        keepThreshold: NATIVE_EVALUATION_KEEP_THRESHOLD,
        fetch: async (url, init) => {
          const t0 = performance.now();
          const response = await deps.fetch(url, init);
          await response.clone().text();
          requestMs.push(Math.round(performance.now() - t0));
          return response;
        },
      },
    );
    for (const [id, d] of classified.decisions) scores[id] = d.useful;
    invocations.push({
      snapshotID: s.snapshotID,
      requests: classified.requests,
      inputTokens: classified.inputTokens,
      elapsedMs: Math.round(performance.now() - started),
      requestMs,
    });
    paid.checkpoint(index, snapshots.length, result());
  }
  paid.finish(result());
  console.log(
    `${Object.keys(scores).length} cases scored; results written to ${output}`,
  );
}

function nativeMetrics(): void {
  for (const split of ["development", "heldout"]) {
    const snapshots = nativeEvidence.snapshots.filter(s => s.split === split);
    const cases = snapshots.flatMap(s => s.cases);
    const runs = Object.keys(cases[0]!.scores);
    const keeps = (c: NativeCase) =>
      Object.values(c.scores).map(p => p >= NATIVE_EVALUATION_KEEP_THRESHOLD);
    const metrics = (
      label: "a" | "b" | "primary",
      keep: (c: NativeCase) => boolean,
    ) => {
      let TP = 0,
        FP = 0,
        FN = 0,
        TN = 0,
        unresolved = 0,
        keptUnresolved = 0;
      for (const c of cases) {
        const yes = keep(c),
          useful = c.labels[label];
        if (useful === null) {
          unresolved++;
          if (yes) keptUnresolved++;
        } else if (yes && useful) TP++;
        else if (yes) FP++;
        else if (useful) FN++;
        else TN++;
      }
      return {
        TP,
        FP,
        FN,
        TN,
        unresolved,
        keptUnresolved,
        precision: TP + FP ? TP / (TP + FP) : null,
        recall: TP + FN ? TP / (TP + FN) : null,
      };
    };
    const majority = (c: NativeCase) =>
      keeps(c).filter(Boolean).length > runs.length / 2;
    const spreads = cases
      .map(
        c =>
          Math.max(...Object.values(c.scores))
          - Math.min(...Object.values(c.scores)),
      )
      .sort((a, b) => a - b);
    console.log(
      JSON.stringify(
        {
          split,
          uniqueCases: cases.length,
          snapshots: snapshots.length,
          perRun: Object.fromEntries(
            runs.map(r => [
              r,
              metrics(
                "primary",
                c => c.scores[r]! >= NATIVE_EVALUATION_KEEP_THRESHOLD,
              ),
            ]),
          ),
          majority: metrics("primary", majority),
          annotatorA: metrics("a", majority),
          annotatorB: metrics("b", majority),
          flippingCases: cases.filter(
            c => keeps(c).some(Boolean) && !keeps(c).every(Boolean),
          ).length,
          scoreSpread: {
            median: spreads[Math.floor(spreads.length / 2)],
            p90: spreads[Math.floor(spreads.length * 0.9)],
            max: spreads.at(-1),
          },
        },
        null,
        2,
      ),
    );
  }
}

if (import.meta.main) {
  const [command, ...args] = process.argv.slice(2);
  if (command === "check" || command === "score") {
    const input = JSON.parse(readFileSync(args[0]!, "utf8")) as NativeInput;
    if (command === "check") await checkNative(input);
    else {
      const [, split, order, output] = args;
      if (!split || !order || !output) {
        throw new Error(
          "usage: run.ts score <external-input.json> <split> <order> <output.json>",
        );
      }
      await scoreNative(input, split, order, output, {
        apiKey: process.env["TYPESAFE_API_KEY"],
        fetch: (url, init) => fetch(url, init),
      });
    }
  } else if (command === "metrics") {
    nativeMetrics();
  } else if (command === "run") {
    const { name, split, batchSize, order, variant, context } =
      parseHistoricalRunArguments(args);
    await run(name, split, batchSize, order, variant, context, {
      apiKey: process.env["TYPESAFE_API_KEY"],
      fetch: (url, init) => fetch(url, init),
    });
  } else if (command === "report") {
    report(args);
  } else {
    throw new Error(
      "usage: run.ts check|score|metrics (historical: run|report) ...",
    );
  }
}

// Keep/trim decision evaluation: how well a decision model judges the fixed
// retention question for fixed, labelled cases. It does not run /magic-trim
// or the summary model. See docs/DecisionEvaluation.md.
//
//   bun packages/opencode-plugin/eval/decisions/run.ts check [--dataset PATH]
//   bun packages/opencode-plugin/eval/decisions/run.ts run --provider jev|openai-decisions --out DIR [options]
//   bun packages/opencode-plugin/eval/decisions/run.ts report RUN... [--dataset PATH] [--threshold T]
import { parseArgs } from "node:util";
import {
  RETENTION_MODEL,
  type RetentionFetch,
} from "../../src/retention/classifier";
import {
  DECISION_EVALUATION_THRESHOLD,
  DEFAULT_DATASET_PATH,
  datasetCases,
  datasetSha256,
  DecisionEvalError,
  isProductionTrimCandidate,
  loadDataset,
  requestSetSha256,
  buildCaseRequest,
  canonicalRequestSha256,
  semanticQuestionSha256,
  type DecisionDataset,
} from "./dataset";
import {
  DEFAULT_TIMEOUT_MS,
  isProviderID,
  OPENAI_DECISIONS_MODELS,
  PROVIDER_IDS,
  PROVIDERS,
  type ProviderAdapter,
} from "./providers";
import { renderReport } from "./report";
import {
  assertFreshOutput,
  DEFAULT_CONCURRENCY,
  DEFAULT_MAX_REQUESTS,
  executeRun,
  isComplete,
  loadRun,
  planRun,
} from "./runner";

export type CliDeps = {
  env: Record<string, string | undefined>;
  fetch: RetentionFetch;
  stdout: (text: string) => void;
  stderr: (text: string) => void;
};

const HELP = `Keep/trim decision evaluation for Magic Compact task-aware retention.

Scores a decision model on the production retention question for labelled,
fixed cases. It measures decision quality given a fixed candidate, not
/magic-trim end to end or summary quality.

Commands
  check  [--dataset PATH]
      Validate a dataset and print its identity. Offline.
  run    --provider ID --out DIR [options]
      Send every case to a decision provider and record each outcome.
  report RUN... [--dataset PATH] [--threshold T]
      Recompute metrics from run directories and compare them. Offline.

run options
  --provider ID       ${PROVIDER_IDS.join(" | ")}
  --model NAME        Jev: a versioned ID (default ${RETENTION_MODEL}); aliases are refused.
                      openai-decisions: ${OPENAI_DECISIONS_MODELS.join(", ")} (the only model OpenAI offers).
  --dataset PATH      Dataset file, or "default" for the bundled synthetic pack.
  --out DIR           New result directory. Existing paths are never overwritten.
  --repeat N          Send each case N times (default 1, at most 10). Measures stability.
  --threshold T       Reference keep threshold (default ${DECISION_EVALUATION_THRESHOLD}). Evaluation only; it
                      changes nothing in production. Fixed for held-out datasets.
  --concurrency N     Requests in flight (default ${DEFAULT_CONCURRENCY}, at most 16).
  --max-requests N    Refuse a plan larger than N requests (default ${DEFAULT_MAX_REQUESTS}).
  --timeout-ms N      Per-request timeout (default ${DEFAULT_TIMEOUT_MS}). Requests are never retried.
  --endpoint URL      Override the provider endpoint (https, or http on loopback).
                      The API key is sent to this URL.
  --dry-run           Validate the selection and print the plan. No key, no network, no files.

Environment (read only for the selected provider)
  TYPESAFE_API_KEY    jev
  OPENAI_API_KEY      openai-decisions
  ChatGPT or OpenCode sign-ins are not used; these APIs need their own keys.

Exit status (run and report): 0 every request produced a probability or a
refusal; 1 some failed, were not attempted, or have no outcome record (unknown:
the run died, so they may have been sent and billed); 2 invalid use.`;

class UsageError extends DecisionEvalError {}

function parse<T extends Record<string, { type: "string" | "boolean" }>>(
  args: string[],
  options: T,
  positionals: boolean,
) {
  try {
    return parseArgs({
      args,
      options: { ...options, help: { type: "boolean", short: "h" } },
      allowPositionals: positionals,
      strict: true,
    });
  } catch (error) {
    throw new UsageError(
      error instanceof Error ? error.message : "Invalid arguments.",
    );
  }
}

function numeric(text: string | undefined, fallback: number): number {
  return text === undefined
    ? fallback
    : text.trim() === ""
      ? NaN
      : Number(text);
}

function datasetOption(value: string | undefined): DecisionDataset {
  return loadDataset(
    value === undefined || value === "default" ? DEFAULT_DATASET_PATH : value,
  );
}

function endpointOption(adapter: ProviderAdapter, value: string | undefined) {
  if (value === undefined) return adapter.endpoint;
  let url: URL;
  try {
    url = new URL(value);
  } catch {
    throw new UsageError("--endpoint must be a URL.");
  }
  const loopback = ["localhost", "127.0.0.1", "[::1]"].includes(url.hostname);
  if (url.protocol !== "https:" && !(url.protocol === "http:" && loopback)) {
    throw new UsageError("--endpoint must use https, or http on localhost.");
  }
  if (url.username || url.password) {
    throw new UsageError("--endpoint must not contain credentials.");
  }
  return url.toString();
}

async function check(args: string[], deps: CliDeps): Promise<number> {
  const { values } = parse(args, { dataset: { type: "string" } }, false);
  const dataset = datasetOption(values.dataset);
  const cases = datasetCases(dataset);
  const entries = cases.map(item => ({
    caseID: item.caseID,
    sha256: canonicalRequestSha256(buildCaseRequest(item)),
  }));
  const ineligible: string[] = [];
  for (const item of cases) {
    if (!(await isProductionTrimCandidate(item.candidate))) {
      ineligible.push(item.caseID);
    }
  }
  const keep = cases.filter(item => item.keep).length;
  deps.stdout(
    [
      `Dataset ${dataset.datasetID} ${dataset.version} (${dataset.split})`,
      `sha256 ${datasetSha256(dataset)}`,
      `${cases.length} cases in ${dataset.snapshots.length} snapshots: ${keep} labelled keep, ${cases.length - keep} labelled trim`,
      `Production request builder: ${cases.length} singleton requests built`,
      `question sha256 ${semanticQuestionSha256()}`,
      `request set sha256 ${requestSetSha256(entries)}`,
      ineligible.length === 0
        ? `Production trim rules would change all ${cases.length} candidates`
        : `Production trim rules would not change ${ineligible.length} candidate(s): ${ineligible.join(", ")}`,
      "No network was used.",
    ].join("\n"),
  );
  return 0;
}

async function run(args: string[], deps: CliDeps): Promise<number> {
  const { values } = parse(
    args,
    {
      provider: { type: "string" },
      model: { type: "string" },
      dataset: { type: "string" },
      out: { type: "string" },
      repeat: { type: "string" },
      threshold: { type: "string" },
      concurrency: { type: "string" },
      "max-requests": { type: "string" },
      "timeout-ms": { type: "string" },
      endpoint: { type: "string" },
      "dry-run": { type: "boolean" },
    },
    false,
  );
  if (!values.provider || !isProviderID(values.provider)) {
    throw new UsageError(
      `--provider must be one of: ${PROVIDER_IDS.join(", ")}.`,
    );
  }
  const adapter = PROVIDERS[values.provider];
  const plan = planRun({
    dataset: datasetOption(values.dataset),
    adapter,
    model:
      values.model
      ?? (adapter.id === "jev" ? RETENTION_MODEL : OPENAI_DECISIONS_MODELS[0]),
    threshold: numeric(values.threshold, DECISION_EVALUATION_THRESHOLD),
    repeat: numeric(values.repeat, 1),
    concurrency: numeric(values.concurrency, DEFAULT_CONCURRENCY),
    timeoutMs: numeric(values["timeout-ms"], DEFAULT_TIMEOUT_MS),
    maxRequests: numeric(values["max-requests"], DEFAULT_MAX_REQUESTS),
    endpoint: endpointOption(adapter, values.endpoint),
  });
  const { manifest } = plan;
  deps.stdout(
    [
      `Provider ${manifest.provider}, model ${manifest.model}, endpoint ${manifest.endpoint}`,
      `Dataset ${manifest.dataset.datasetID} ${manifest.dataset.version} (${manifest.dataset.split}) sha256 ${manifest.dataset.sha256}`,
      `Plan: ${manifest.cases.length} cases x ${manifest.repeat} = ${plan.units.length} requests, concurrency ${manifest.concurrency}, timeout ${manifest.timeoutMs} ms, no retries`,
      `Reference threshold ${manifest.threshold}`,
      `question sha256 ${manifest.prompt.semanticQuestionSha256}`,
      `canonical request set sha256 ${manifest.prompt.requestSetSha256}`,
      manifest.prompt.translation
        ? `translation ${manifest.prompt.translation.version} sha256 ${manifest.prompt.translation.sha256}`
        : "translation none (production body with the model replaced)",
      `provider request set sha256 ${plan.translatedRequestSetSha256}`,
    ].join("\n"),
  );
  if (values["dry-run"]) {
    deps.stdout("Dry run: no key was read and nothing was sent or written.");
    return 0;
  }
  if (!values.out) throw new UsageError("--out DIR is required.");
  assertFreshOutput(values.out);
  const apiKey = deps.env[adapter.keyEnv];
  if (!apiKey) {
    throw new UsageError(
      `${adapter.keyEnv} is not set; ${adapter.id} needs its own API key.`,
    );
  }
  await executeRun(plan, {
    apiKey,
    fetch: deps.fetch,
    out: values.out,
    onOutcome: outcome =>
      deps.stdout(
        `${outcome.caseID} #${outcome.repeat}: ${
          outcome.status === "scored"
            ? `scored ${outcome.probability}`
            : outcome.status === "failed"
              ? `failed (${outcome.category}${outcome.httpStatus ? ` ${outcome.httpStatus}` : ""})`
              : outcome.status === "refused"
                ? "refused"
                : `not attempted (${outcome.reason})`
        }`,
      ),
  });
  deps.stdout(`\nRecords written to ${values.out}\n`);
  const result = loadRun(values.out, plan.options.dataset);
  deps.stdout(renderReport([result], plan.options.dataset));
  return isComplete(result) ? 0 : 1;
}

async function report(args: string[], deps: CliDeps): Promise<number> {
  const { values, positionals } = parse(
    args,
    { dataset: { type: "string" }, threshold: { type: "string" } },
    true,
  );
  if (positionals.length === 0) {
    throw new UsageError("report needs at least one run directory.");
  }
  const dataset = datasetOption(values.dataset);
  const runs = positionals.map(dir => loadRun(dir, dataset));
  const names = runs.map(r => r.name);
  const named = runs.map((r, index) =>
    names.indexOf(r.name) === index
      ? r
      : { ...r, name: `${r.name}~${index + 1}` },
  );
  deps.stdout(
    renderReport(
      named,
      dataset,
      values.threshold === undefined
        ? undefined
        : numeric(values.threshold, NaN),
    ),
  );
  return runs.every(isComplete) ? 0 : 1;
}

export async function main(argv: string[], deps: CliDeps): Promise<number> {
  const [command, ...rest] = argv;
  try {
    if (command === undefined) {
      deps.stderr(HELP);
      return 2;
    }
    if (
      ["help", "--help", "-h"].includes(command)
      || rest.includes("--help")
      || rest.includes("-h")
    ) {
      if (
        !["help", "--help", "-h", "check", "run", "report"].includes(command)
      ) {
        throw new UsageError(`Unknown command ${command}.`);
      }
      deps.stdout(HELP);
      return 0;
    }
    if (command === "check") return await check(rest, deps);
    if (command === "run") return await run(rest, deps);
    if (command === "report") return await report(rest, deps);
    throw new UsageError(`Unknown command ${command}.`);
  } catch (error) {
    if (!(error instanceof DecisionEvalError)) throw error;
    deps.stderr(
      `Error: ${error.message}${error instanceof UsageError ? "\nRun with --help for usage." : ""}`,
    );
    return 2;
  }
}

if (import.meta.main) {
  process.exitCode = await main(process.argv.slice(2), {
    env: process.env,
    fetch: (url, init) => fetch(url, init),
    stdout: text => console.log(text),
    stderr: text => console.error(text),
  });
}

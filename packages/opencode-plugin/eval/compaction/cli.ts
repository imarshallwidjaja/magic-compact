// Magic Compact model eval. See docs/CompactionEvaluation.md.
import { copyFileSync, existsSync, mkdirSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { checkCompactionPlans, CorpusError, loadCorpus } from "./corpus";
import { GRADES_FORMAT, validateGrades } from "./grade";
import {
  EvalError,
  loopbackServerUrl,
  parseModelSelector,
  serverAuth,
} from "./host";
import { formatReport, summarizeRun } from "./report";
import { cleanupRun, gradeRunWithJudge, readRun, runEval } from "./run";

export const USAGE = `Magic Compact compaction eval

  bun run eval:compact check
      Validate the frozen corpus offline. No server, no network.

  bun run eval:compact run --server URL --model provider/model[#variant] --out NEW_DIR
                           [--repeat N] [--case A[,B...]] [--judge provider/model[#variant]]
                           [--cache DIR] [--subject TGZ] [--assert-host-version VERSION] [--dry-run]
      Run /magic-compact 1 once per case and repeat on an existing loopback
      OpenCode server, in fresh sessions of a private temporary project.
      Every attempt is a paid summarizer request. --judge adds one paid judge
      request per delivered summary; there is no default judge. --dry-run
      stops after preflight and creates no session.

  bun run eval:compact grade RUN_DIR --import FILE
      Add manual grades (${GRADES_FORMAT}) for a run. Offline.

  bun run eval:compact grade RUN_DIR --server URL --judge provider/model[#variant]
      Grade a finished run with a judge model (one paid request per summary).

  bun run eval:compact report RUN_DIR [RUN_DIR...] [--json]
      Summarize and compare saved runs. Offline.

  bun run eval:compact cleanup RUN_DIR --server URL
      Reconcile an interrupted run from its ledger and remove only sessions it
      created whose recorded state is unchanged.

Server credentials come only from OPENCODE_SERVER_USERNAME and
OPENCODE_SERVER_PASSWORD. Model credentials stay inside the server.
`;

type Flags = {
  positional: string[];
  values: Map<string, string>;
  switches: Set<string>;
};

const VALUE_FLAGS = new Set([
  "--server",
  "--model",
  "--out",
  "--repeat",
  "--case",
  "--judge",
  "--cache",
  "--subject",
  "--assert-host-version",
  "--import",
]);
const SWITCHES = new Set(["--dry-run", "--json", "--help"]);

export function parseFlags(args: string[]): Flags {
  const flags: Flags = {
    positional: [],
    values: new Map(),
    switches: new Set(),
  };
  for (let i = 0; i < args.length; i++) {
    const arg = args[i]!;
    if (VALUE_FLAGS.has(arg)) {
      const value = args[++i];
      if (value === undefined || value.startsWith("--"))
        throw new EvalError(`${arg} needs a value.`);
      if (flags.values.has(arg)) throw new EvalError(`${arg} given twice.`);
      flags.values.set(arg, value);
    } else if (SWITCHES.has(arg)) flags.switches.add(arg);
    else if (arg.startsWith("--"))
      throw new EvalError(`Unknown option ${arg}.`);
    else flags.positional.push(arg);
  }
  return flags;
}

function allowOnly(flags: Flags, allowed: string[]) {
  for (const key of [...flags.values.keys(), ...flags.switches])
    if (!allowed.includes(key))
      throw new EvalError(`${key} does not apply to this command.`);
}

function required(flags: Flags, name: string): string {
  const value = flags.values.get(name);
  if (value === undefined) throw new EvalError(`${name} is required.`);
  return value;
}

export async function main(argv: string[]): Promise<number> {
  const [command, ...rest] = argv;
  if (!command || command === "--help" || command === "help") {
    console.log(USAGE);
    return command ? 0 : 2;
  }
  const flags = parseFlags(rest);
  switch (command) {
    case "check": {
      allowOnly(flags, []);
      const corpus = loadCorpus();
      const plans = await checkCompactionPlans(corpus);
      const ok = plans.every(
        p =>
          p.v1.targets === 1
          && p.v2.targets === 1
          && p.v1.preserved === 1
          && p.v2.preserved === 1
          && p.v1.candidates === 0
          && p.v2.candidates === 0,
      );
      console.log(
        JSON.stringify(
          {
            corpus: corpus.corpusId,
            corpusSha256: corpus.corpusSha256,
            rubricSha256: corpus.rubricSha256,
            cases: plans,
            ok,
          },
          null,
          2,
        ),
      );
      return ok ? 0 : 1;
    }
    case "run": {
      allowOnly(flags, [
        "--server",
        "--model",
        "--out",
        "--repeat",
        "--case",
        "--judge",
        "--cache",
        "--subject",
        "--assert-host-version",
        "--dry-run",
      ]);
      const repeat = Number(flags.values.get("--repeat") ?? "1");
      if (!Number.isInteger(repeat) || repeat < 1 || repeat > 20)
        throw new EvalError("--repeat must be an integer from 1 to 20.");
      const judge = flags.values.get("--judge");
      const result = await runEval({
        server: loopbackServerUrl(required(flags, "--server")),
        auth: serverAuth(),
        model: parseModelSelector(required(flags, "--model"), "--model"),
        out: required(flags, "--out"),
        repeat,
        cases: flags.values.get("--case")?.split(",") ?? null,
        judge: judge ? parseModelSelector(judge, "--judge") : null,
        dryRun: flags.switches.has("--dry-run"),
        cacheRoot: flags.values.get("--cache") ?? null,
        subject: flags.values.get("--subject") ?? null,
        assertHostVersion: flags.values.get("--assert-host-version") ?? null,
      });
      console.log(formatReport([summarizeRun(result.out)]));
      return 0;
    }
    case "grade": {
      const dir = resolve(flags.positional[0] ?? "");
      if (flags.positional.length !== 1 || !existsSync(join(dir, "run.json")))
        throw new EvalError("grade needs one run directory.");
      if (flags.values.has("--import")) {
        allowOnly(flags, ["--import"]);
        const file = resolve(flags.values.get("--import")!);
        const corpus = loadCorpus();
        const { run, attempts } = readRun(dir, corpus);
        const grades = validateGrades(
          JSON.parse(readFileSync(file, "utf8")),
          {
            runNonce: run.runNonce,
            corpusSha256: run.corpus.sha256,
            rubricSha256: run.corpus.rubricSha256,
          },
          attempts
            .filter(
              a =>
                a.status === "delivered"
                && a.capsule !== null
                && a.outputSha256 !== null,
            )
            .map(a => ({
              attemptId: a.attemptId,
              caseId: a.caseId,
              outputSha256: a.outputSha256!,
              capsule: a.capsule!,
            })),
          corpus.cases,
        );
        if (grades.grader.kind !== "manual")
          throw new EvalError(
            'Imported grades must have grader.kind "manual".',
          );
        mkdirSync(join(dir, "grades"), { recursive: true });
        const target = join(
          dir,
          "grades",
          `manual-${new Date().toISOString().replace(/[:.]/gu, "-")}.json`,
        );
        if (existsSync(target))
          throw new EvalError(`${target} already exists.`);
        copyFileSync(file, target);
        console.log(`Imported ${grades.items.length} grades into ${target}.`);
        return 0;
      }
      allowOnly(flags, ["--server", "--judge", "--assert-host-version"]);
      await gradeRunWithJudge({
        dir,
        server: loopbackServerUrl(required(flags, "--server")),
        auth: serverAuth(),
        judge: parseModelSelector(required(flags, "--judge"), "--judge"),
        assertHostVersion: flags.values.get("--assert-host-version") ?? null,
      });
      console.log(formatReport([summarizeRun(dir)]));
      return 0;
    }
    case "report": {
      allowOnly(flags, ["--json"]);
      if (flags.positional.length === 0)
        throw new EvalError("report needs at least one run directory.");
      const summaries = flags.positional.map(dir => summarizeRun(resolve(dir)));
      console.log(
        flags.switches.has("--json")
          ? JSON.stringify(summaries, null, 2)
          : formatReport(summaries),
      );
      return 0;
    }
    case "cleanup": {
      allowOnly(flags, ["--server", "--assert-host-version"]);
      if (flags.positional.length !== 1)
        throw new EvalError("cleanup needs one run directory.");
      const summary = await cleanupRun({
        dir: resolve(flags.positional[0]!),
        server: loopbackServerUrl(required(flags, "--server")),
        auth: serverAuth(),
        assertHostVersion: flags.values.get("--assert-host-version") ?? null,
      });
      return summary.stillOwned.length
        + summary.unknownOutcomes.length
        + summary.unresolvedIntents.length
        || summary.disposal.some(d => !d.scratchRemoved)
        ? 1
        : 0;
    }
    default:
      throw new EvalError(`Unknown command ${command}.\n\n${USAGE}`);
  }
}

if (import.meta.main) {
  main(process.argv.slice(2)).then(
    code => process.exit(code),
    error => {
      console.error(
        error instanceof EvalError || error instanceof CorpusError
          ? error.message
          : error,
      );
      process.exit(1);
    },
  );
}

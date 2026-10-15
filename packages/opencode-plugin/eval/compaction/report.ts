// Offline reports over saved runs. Nothing here contacts a server or a model.
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { loadCorpus, type Corpus } from "./corpus";
import { score, validateGrades, type GradesFile, type Scores } from "./grade";
import { readRun, type AttemptRecord } from "./run";

// How the frozen rubric treats an attempt. A settled attempt that delivered
// no valid summary scores 0 and is not usable; an attempt whose outcome is
// unknown, or that never reached the model, is not scored at all; a delivered
// summary without an accepted grade stays ungraded and is never counted as 0.
type Disposition = "delivered" | "zero" | "unscored";

const ZERO_STATUSES = new Set([
  "failed",
  "noop",
  "delivered-invalid",
  // The model does not accept the case's image: a capability refusal.
  "preflight-skipped",
]);

export function disposition(attempt: AttemptRecord): Disposition {
  if (attempt.status === "delivered") return "delivered";
  return ZERO_STATUSES.has(attempt.status) ? "zero" : "unscored";
}

export type CaseScore = {
  caseId: string;
  graded: number;
  zero: number;
  ungraded: number;
  unscored: number;
  // Means over graded plus zero attempts; null while any delivered summary of
  // the case is ungraded, so a partial grade never claims an average.
  textMean: number | null;
  imageMean: number | null;
  usable: number;
  scored: number;
};

type GraderSummary = {
  file: string;
  label: string;
  kind: "manual" | "judge";
  rejected: string | null;
  graded: number;
  delivered: number;
  complete: boolean;
  usable: number;
  perCase: CaseScore[];
  items: Array<{ attemptId: string; scores: Scores }>;
};

export type RunSummary = {
  dir: string;
  runNonce: string;
  host: string;
  model: string;
  judge: string | null;
  status: string;
  corpus: string;
  subject: string | null;
  source: string;
  attempts: number;
  statuses: Record<string, number>;
  failures: Record<string, number>;
  mechanicalPass: number;
  zeroScored: number;
  unscored: number;
  cleanupRetained: string[];
  medianCommandMs: number | null;
  usage: {
    requests: number;
    input: number;
    output: number;
    reasoning: number;
    cost: number;
  };
  semantic: "UNGRADED" | "PARTIAL" | "GRADED";
  graders: GraderSummary[];
  // Scores when no attempt delivered a summary: every settled attempt is 0
  // under the rubric with no grader needed.
  rubricOnly: CaseScore[] | null;
  rows: AttemptRecord[];
};

// Per-case scores of one grader's accepted items under the frozen rubric.
export function caseScores(
  cases: string[],
  attempts: AttemptRecord[],
  items: Array<{ attemptId: string; scores: Scores }>,
  corpus: Corpus,
): CaseScore[] {
  return cases.map(caseId => {
    const imageCase = corpus.cases.find(c => c.id === caseId)?.image != null;
    const entry: CaseScore = {
      caseId,
      graded: 0,
      zero: 0,
      ungraded: 0,
      unscored: 0,
      textMean: null,
      imageMean: null,
      usable: 0,
      scored: 0,
    };
    let text = 0;
    let image = 0;
    for (const attempt of attempts.filter(a => a.caseId === caseId)) {
      const kind = disposition(attempt);
      if (kind === "unscored") entry.unscored += 1;
      else if (kind === "zero") entry.zero += 1;
      else {
        const item = items.find(i => i.attemptId === attempt.attemptId);
        if (!item) entry.ungraded += 1;
        else {
          entry.graded += 1;
          text += item.scores.textScore;
          image += item.scores.imageScore ?? 0;
          entry.usable += item.scores.usable ? 1 : 0;
        }
      }
    }
    entry.scored = entry.graded + entry.zero;
    if (entry.ungraded === 0 && entry.scored > 0) {
      entry.textMean = text / entry.scored;
      entry.imageMean = imageCase ? image / entry.scored : null;
    }
    return entry;
  });
}

export function summarizeRun(
  dir: string,
  corpus: Corpus = loadCorpus(),
): RunSummary {
  const { run, preflight, results, attempts } = readRun(dir, corpus);
  const statuses: Record<string, number> = {};
  const failures: Record<string, number> = {};
  for (const attempt of attempts) {
    statuses[attempt.status] = (statuses[attempt.status] ?? 0) + 1;
    if (attempt.outcome?.kind === "failed")
      failures[attempt.outcome.category] =
        (failures[attempt.outcome.category] ?? 0) + 1;
  }
  const ms = attempts
    .filter(a => disposition(a) === "delivered")
    .map(a => a.commandMs ?? 0)
    .sort((a, b) => a - b);
  const usage = { requests: 0, input: 0, output: 0, reasoning: 0, cost: 0 };
  for (const attempt of attempts) {
    if (!attempt.usage) continue;
    usage.requests += attempt.usage.requests;
    usage.input += attempt.usage.input;
    usage.output += attempt.usage.output;
    usage.reasoning += attempt.usage.reasoning;
    usage.cost += attempt.usage.cost;
  }
  const graded = attempts
    .filter(a => disposition(a) === "delivered")
    .map(a => ({
      attemptId: a.attemptId,
      caseId: a.caseId,
      outputSha256: a.outputSha256!,
      capsule: a.capsule!,
    }));
  const cases = run.requested.cases;
  const gradesDir = join(dir, "grades");
  const graders: GraderSummary[] = [];
  for (const file of existsSync(gradesDir)
    ? readdirSync(gradesDir)
        .filter(name => name.endsWith(".json") && !name.endsWith(".calls.json"))
        .sort()
    : []) {
    const raw = JSON.parse(readFileSync(join(gradesDir, file), "utf8"));
    let grades: GradesFile | null = null;
    let rejected: string | null = null;
    try {
      grades = validateGrades(
        raw,
        {
          runNonce: run.runNonce,
          corpusSha256: run.corpus.sha256,
          rubricSha256: run.corpus.rubricSha256,
        },
        graded,
        corpus.cases,
      );
    } catch (error) {
      rejected = String((error as Error).message);
    }
    const items = (grades?.items ?? []).map(item => ({
      attemptId: item.attemptId,
      scores: score(
        item,
        corpus.cases.find(c => c.id === item.caseId)!,
      ),
    }));
    graders.push({
      file,
      label: grades?.grader.label ?? String(raw?.grader?.label ?? "unknown"),
      kind:
        grades?.grader.kind
        ?? (raw?.grader?.kind === "manual" ? "manual" : "judge"),
      rejected,
      graded: items.length,
      delivered: graded.length,
      complete:
        !rejected && graded.length > 0 && items.length === graded.length,
      usable: items.filter(i => i.scores.usable).length,
      perCase: rejected ? [] : caseScores(cases, attempts, items, corpus),
      items,
    });
  }
  const accepted = graders.filter(g => !g.rejected && g.graded > 0);
  const zeroScored = attempts.filter(a => disposition(a) === "zero").length;
  const rubricOnly =
    graded.length === 0 && zeroScored > 0
      ? caseScores(cases, attempts, [], corpus)
      : null;
  const artifact = preflight?.["artifact"] as
    { subjectTgzSha256?: string } | undefined;
  const source = run["source"] as
    { commit?: string | null; packageDirty?: boolean | null } | undefined;
  return {
    dir,
    runNonce: run.runNonce,
    host: `OpenCode ${run.server.version} (${run.server.qualification})`,
    model: run.requested.model,
    judge: run.requested.judge,
    status: String(results?.["status"] ?? "incomplete (no results.json)"),
    corpus: `${run.corpus.id} ${run.corpus.sha256.slice(0, 12)}`,
    subject: artifact?.subjectTgzSha256 ?? null,
    source: `${source?.commit?.slice(0, 12) ?? "unknown"}${source?.packageDirty ? " (package dirty)" : ""}`,
    attempts: attempts.length,
    statuses,
    failures,
    mechanicalPass: graded.length,
    zeroScored,
    unscored: attempts.filter(a => disposition(a) === "unscored").length,
    cleanupRetained: attempts.flatMap(a =>
      a.cleanup.filter(c => c.action === "retained").map(c => c.id),
    ),
    medianCommandMs: ms.length ? ms[Math.floor(ms.length / 2)]! : null,
    usage,
    semantic:
      accepted.some(g => g.complete) || rubricOnly
        ? "GRADED"
        : accepted.length
          ? "PARTIAL"
          : "UNGRADED",
    graders,
    rubricOnly,
    rows: attempts,
  };
}

function caseLine(c: CaseScore): string {
  const counts = `${c.graded} graded, ${c.zero} settled without a valid summary (scored 0)${c.unscored ? `, ${c.unscored} unscored (outcome unknown or never sent)` : ""}`;
  if (c.textMean === null)
    return c.ungraded
      ? `    - case ${c.caseId}: PARTIAL, ${c.ungraded} delivered summar${c.ungraded === 1 ? "y" : "ies"} ungraded; no mean claimed (${counts})`
      : `    - case ${c.caseId}: nothing scored (${counts})`;
  return `    - case ${c.caseId}: mean text ${c.textMean.toFixed(2)}/8${c.imageMean === null ? "" : `, mean image ${c.imageMean.toFixed(2)}/4`}, usable ${c.usable}/${c.scored} (${counts})`;
}

export function formatReport(summaries: RunSummary[]): string {
  const out: string[] = [];
  for (const s of summaries) {
    out.push(`## ${s.model} on ${s.host}`);
    out.push("");
    out.push(
      `- Run: ${s.dir} (${s.status}); corpus ${s.corpus}; subject ${s.subject?.slice(0, 12) ?? "unknown"}; source ${s.source}`,
    );
    out.push(
      `- Mechanical: ${s.mechanicalPass} of ${s.attempts} attempts delivered a successor that passed every invariant. Statuses: ${
        Object.entries(s.statuses)
          .map(([k, v]) => `${k} ${v}`)
          .join(", ") || "none"
      }.`,
    );
    if (Object.keys(s.failures).length)
      out.push(
        `- Failure categories: ${Object.entries(s.failures)
          .map(([k, v]) => `${k} ${v}`)
          .join(", ")}`,
      );
    out.push(
      `- Cost and speed (not part of any score): median command ${s.medianCommandMs ?? "n/a"} ms; observed summarizer requests ${s.usage.requests}, input ${s.usage.input}, output ${s.usage.output}, reasoning ${s.usage.reasoning} tokens, cost ${s.usage.cost.toFixed(4)}.`,
    );
    if (s.cleanupRetained.length)
      out.push(
        `- Retained sessions (not deleted, see ledger): ${s.cleanupRetained.join(", ")}`,
      );
    out.push(
      `- Semantic: ${s.semantic}${s.semantic === "UNGRADED" ? " (no accepted grades; mechanical results alone say nothing about summary quality)" : ""}`,
    );
    if (s.rubricOnly) {
      out.push(
        "  - No attempt delivered a summary; under the rubric every settled attempt scores 0:",
      );
      for (const c of s.rubricOnly) out.push(caseLine(c));
    }
    for (const g of s.graders) {
      if (g.rejected) {
        out.push(`  - ${g.file}: REJECTED (${g.rejected})`);
        continue;
      }
      const label =
        g.kind === "judge"
          ? `${g.label}, automatic estimate`
          : `${g.label}, manual`;
      const scored = g.perCase.reduce((n, c) => n + c.scored, 0);
      out.push(
        `  - ${label}: graded ${g.graded} of ${g.delivered} delivered summaries${g.complete ? "" : " (partial)"}; ${g.complete ? `usable ${g.usable} of ${scored} scored attempts` : "no overall usable rate claimed while summaries are ungraded"}; ${s.zeroScored} settled attempts without a valid summary score 0; ${s.unscored} attempts unscored.`,
      );
      for (const c of g.perCase) out.push(caseLine(c));
    }
    out.push("");
    out.push(
      "| attempt | status | ms | source unchanged | invariants | stored history |",
    );
    out.push("| --- | --- | --- | --- | --- | --- |");
    for (const row of s.rows) {
      const invariants = row.successorChecks.length
        ? `${row.successorChecks.filter(c => c.ok).length}/${row.successorChecks.length}`
        : "-";
      const detail =
        row.outcome?.kind === "failed"
          ? `${row.status} (${row.outcome.category})`
          : row.status;
      const stored =
        (row.storedHistory ?? "")
          .split("\n")
          .find(l => /Stored history/u.test(l))
          ?.trim() ?? "";
      out.push(
        `| ${row.attemptId} | ${detail} | ${row.commandMs ?? "-"} | ${row.sourceUnchanged ?? "-"} | ${invariants} | ${stored.replace(/\|/gu, "/")} |`,
      );
    }
    out.push("");
  }
  out.push(
    "Scores follow the corpus rubric (scoring.md): a settled attempt that delivered no valid summary scores 0; an attempt with an unknown outcome is not scored; an ungraded summary is never counted as 0. Automatic judge scores are estimates from one model and are labeled separately from manual grades; none of these numbers is a general accuracy claim about a model.",
  );
  return out.join("\n");
}

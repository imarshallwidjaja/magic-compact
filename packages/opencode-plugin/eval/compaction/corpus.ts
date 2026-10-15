// The frozen compaction corpus: loading, integrity checks and the source
// histories each host imports. Corpus files are data only; nothing here
// imports or executes a fixture.
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { z } from "zod";
import { planHistory } from "../../src/compact/fork";
import { planCompaction, type MessageWithParts } from "../../src/compact/plan";
import { collectRetentionCandidates } from "../../src/policy/retention";
import { findRetentionCandidates } from "../../src/retention/retention";
import { planV2Compaction } from "../../src/v2/compact";
import { v2RetentionTurns, v2ToolCalls } from "../../src/v2/history";
import type { V2Message } from "../../src/v2/session";

export const CORPUS_DIR = join(import.meta.dir, "corpus", "v1");
export const MANIFEST_FORMAT = "arkive-compaction-eval-manifest/1";
const CORPUS_FORMAT = "arkive-compaction-eval-corpus/1";
const ASSET_FORMAT = "arkive-compaction-eval-asset/1";

export const sha256 = (value: string | Uint8Array) =>
  createHash("sha256").update(value).digest("hex");

const Hex = z.string().regex(/^[0-9a-f]{64}$/u);
const ManifestSchema = z.strictObject({
  format: z.literal(MANIFEST_FORMAT),
  corpusId: z.string(),
  rubricId: z.string(),
  caseOrder: z.array(z.string()).min(1),
  files: z.record(z.string(), Hex),
  provenance: z.strictObject({
    fixtureVersion: z.string(),
    goldSha256: Hex,
    casesModuleSha256: Hex,
    originalFiles: z.record(z.string(), Hex),
    note: z.string(),
  }),
});
const CaseSchema = z.strictObject({
  id: z.string(),
  title: z.string(),
  targetUser: z.string().min(1),
  targetAssistant: z.string().min(1),
  image: z.strictObject({ asset: z.string() }).nullable(),
});
const CorpusSchema = z.strictObject({
  format: z.literal(CORPUS_FORMAT),
  corpusId: z.string(),
  description: z.string(),
  command: z.strictObject({ name: z.string(), arguments: z.string() }),
  followup: z.strictObject({ user: z.string(), assistant: z.string() }),
  rules: z.strictObject({
    targetAssistantChars: z.strictObject({
      min: z.number().int(),
      max: z.number().int(),
    }),
    followupTerms: z.string(),
    leakPatterns: z.record(z.string(), z.string()),
  }),
  cases: z.array(CaseSchema),
});
const AssetSchema = z.strictObject({
  format: z.literal(ASSET_FORMAT),
  name: z.string(),
  mime: z.literal("image/png"),
  sha256: Hex,
  width: z.number().int(),
  height: z.number().int(),
  base64: z.string(),
});
const FactSchema = z.strictObject({
  id: z.string(),
  critical: z.boolean(),
  kind: z.string().optional(),
  fact: z.string(),
});
const GoldCaseSchema = z.strictObject({
  title: z.string(),
  textFacts: z.array(FactSchema),
  visualFacts: z.array(FactSchema),
  criticalVisual: z.array(z.string()),
});
const GoldSchema = z.strictObject({
  fixtureVersion: z.string(),
  scoring: z.string(),
  cases: z.record(z.string(), GoldCaseSchema),
  goldSha256: Hex,
});
const ProvenanceSchema = z.object({
  fixtureVersion: z.string(),
  followup: z.object({ userSha256: Hex, assistantSha256: Hex }),
  cases: z.record(
    z.string(),
    z.object({
      targetUserSha256: Hex,
      targetAssistantSha256: Hex,
      image: z.object({ sha256: Hex, base64Sha256: Hex }).nullable(),
    }),
  ),
  goldSha256: Hex,
  casesModuleSha256: Hex,
});

export type Fact = z.infer<typeof FactSchema>;
export type GoldCase = z.infer<typeof GoldCaseSchema>;
export type CorpusImage = {
  name: string;
  mime: "image/png";
  sha256: string;
  base64: string;
};
export type CorpusCase = {
  id: string;
  title: string;
  targetUser: string;
  targetAssistant: string;
  image: CorpusImage | null;
  gold: GoldCase;
};
export type Corpus = {
  corpusId: string;
  rubricId: string;
  // SHA-256 of manifest.json, which binds every corpus file by hash.
  corpusSha256: string;
  rubricSha256: string;
  command: { name: string; arguments: string };
  followup: { user: string; assistant: string };
  cases: CorpusCase[];
};

export class CorpusError extends Error {}

// Loads and fully validates the corpus. Unknown formats, extra fields, hash
// mismatches and broken provenance bindings all refuse.
export function loadCorpus(dir: string = CORPUS_DIR): Corpus {
  const problems: string[] = [];
  const need = (ok: boolean, message: string) => {
    if (!ok) problems.push(message);
  };
  const read = (name: string) => readFileSync(join(dir, name));
  const manifestBytes = read("manifest.json");
  const manifest = parse(ManifestSchema, manifestBytes, "manifest.json");
  for (const [name, hash] of Object.entries(manifest.files)) {
    need(sha256(read(name)) === hash, `${name}: SHA-256 differs from manifest`);
  }
  for (const name of ["corpus.json", "gold.json", "scoring.md"]) {
    need(name in manifest.files, `manifest does not bind ${name}`);
  }
  if (problems.length) throw new CorpusError(problems.join("\n"));

  const corpus = parse(CorpusSchema, read("corpus.json"), "corpus.json");
  const gold = parse(GoldSchema, read("gold.json"), "gold.json");
  const provenance = parse(
    ProvenanceSchema,
    read("provenance/comparison-manifest.json"),
    "provenance/comparison-manifest.json",
  );
  need(corpus.corpusId === manifest.corpusId, "corpus ID differs");
  need(
    sha256(JSON.stringify(gold.cases)) === gold.goldSha256
      && gold.goldSha256 === manifest.provenance.goldSha256
      && provenance.goldSha256 === gold.goldSha256,
    "gold facts differ from the frozen gold binding",
  );
  need(
    provenance.casesModuleSha256 === manifest.provenance.casesModuleSha256
      && provenance.fixtureVersion === manifest.provenance.fixtureVersion
      && gold.fixtureVersion === manifest.provenance.fixtureVersion,
    "provenance fixture version or module hash differs",
  );
  need(
    sha256(corpus.followup.user) === provenance.followup.userSha256
      && sha256(corpus.followup.assistant)
        === provenance.followup.assistantSha256,
    "protected follow-up turn differs from the frozen text",
  );
  need(
    JSON.stringify(corpus.cases.map(c => c.id))
      === JSON.stringify(manifest.caseOrder)
      && JSON.stringify(Object.keys(gold.cases))
        === JSON.stringify(manifest.caseOrder),
    "case order differs from the manifest",
  );

  const followupTerms = new RegExp(corpus.rules.followupTerms, "iu");
  const cases: CorpusCase[] = [];
  for (const entry of corpus.cases) {
    const frozen = provenance.cases[entry.id];
    const goldCase = gold.cases[entry.id];
    if (!frozen || !goldCase) {
      problems.push(`${entry.id}: no frozen provenance or gold facts`);
      continue;
    }
    need(
      sha256(entry.targetUser) === frozen.targetUserSha256
        && sha256(entry.targetAssistant) === frozen.targetAssistantSha256,
      `${entry.id}: target turn differs from the frozen text`,
    );
    const { min, max } = corpus.rules.targetAssistantChars;
    need(
      entry.targetAssistant.length >= min
        && entry.targetAssistant.length <= max,
      `${entry.id}: target assistant length outside ${min}..${max}`,
    );
    need(
      goldCase.textFacts.length === 8
        && goldCase.textFacts.filter(fact => fact.critical).length === 4,
      `${entry.id}: needs 8 text facts with 4 critical`,
    );
    const visualIDs = goldCase.visualFacts.map(fact => fact.id);
    need(
      goldCase.criticalVisual.every(id => visualIDs.includes(id))
        && goldCase.visualFacts.every(
          fact => fact.critical === goldCase.criticalVisual.includes(fact.id),
        ),
      `${entry.id}: critical visual facts disagree`,
    );
    need(
      !followupTerms.test(entry.targetUser + entry.targetAssistant),
      `${entry.id}: target shares protected-turn terms`,
    );
    const leak = corpus.rules.leakPatterns[entry.id];
    if (leak) {
      const history = [
        entry.targetUser,
        entry.targetAssistant,
        corpus.followup.user,
        corpus.followup.assistant,
      ].join("\n");
      need(
        !new RegExp(leak, "iu").test(history),
        `${entry.id}: visual answer words appear in the source text`,
      );
    }
    let image: CorpusImage | null = null;
    if (entry.image) {
      need(entry.image.asset in manifest.files, `${entry.id}: asset not bound`);
      const asset = parse(AssetSchema, read(entry.image.asset), "asset");
      const bytes = Buffer.from(asset.base64, "base64");
      need(
        bytes.toString("base64") === asset.base64
          && sha256(bytes) === asset.sha256
          && asset.sha256 === frozen.image?.sha256
          && sha256(asset.base64) === frozen.image.base64Sha256,
        `${entry.id}: image differs from the frozen PNG`,
      );
      need(
        visualIDs.length === 4,
        `${entry.id}: image case needs 4 visual facts`,
      );
      image = {
        name: asset.name,
        mime: asset.mime,
        sha256: asset.sha256,
        base64: asset.base64,
      };
    } else {
      need(
        frozen.image === null && visualIDs.length === 0,
        `${entry.id}: text case has image facts`,
      );
    }
    cases.push({ ...entry, image, gold: goldCase });
  }
  if (problems.length) throw new CorpusError(problems.join("\n"));
  return {
    corpusId: manifest.corpusId,
    rubricId: manifest.rubricId,
    corpusSha256: sha256(manifestBytes),
    rubricSha256: manifest.files["scoring.md"]!,
    command: corpus.command,
    followup: corpus.followup,
    cases,
  };
}

function parse<T>(schema: z.ZodType<T>, bytes: Buffer, name: string): T {
  let value: unknown;
  try {
    value = JSON.parse(bytes.toString("utf8"));
  } catch {
    throw new CorpusError(`${name}: not JSON`);
  }
  const result = schema.safeParse(value);
  if (!result.success) {
    throw new CorpusError(
      `${name}: unsupported format or shape (${result.error.issues[0]?.message ?? "invalid"})`,
    );
  }
  return result.data;
}

// The two complete turns of a case: the target, then the protected follow-up.
function turns(corpus: Corpus, c: CorpusCase) {
  return [
    { user: c.targetUser, image: c.image, assistant: c.targetAssistant },
    {
      user: corpus.followup.user,
      image: null,
      assistant: corpus.followup.assistant,
    },
  ];
}

export type ModelRef = {
  providerID: string;
  modelID: string;
  variant?: string;
};

// The OpenCode 1 source history for a case, written to `sessionID` with fresh
// ascending IDs through the production history planner.
export function v1SourceHistory(
  corpus: Corpus,
  c: CorpusCase,
  sessionID: string,
  directory: string,
  model: ModelRef,
  now: number = Date.now(),
): MessageWithParts[] {
  const template = "ses_eval_template";
  let t = now - 600_000;
  let n = 0;
  const id = (prefix: string) =>
    `${prefix}_eval_${String(n++).padStart(4, "0")}`;
  const variant = model.variant ? { variant: model.variant } : {};
  const messages: MessageWithParts[] = [];
  for (const turn of turns(corpus, c)) {
    const userID = id("msg");
    const assistantID = id("msg");
    const userParts = [
      {
        id: id("prt"),
        sessionID: template,
        messageID: userID,
        type: "text",
        text: turn.user,
      },
      ...(turn.image
        ? [
            {
              id: id("prt"),
              sessionID: template,
              messageID: userID,
              type: "file",
              mime: turn.image.mime,
              filename: turn.image.name,
              url: `data:${turn.image.mime};base64,${turn.image.base64}`,
            },
          ]
        : []),
    ];
    messages.push({
      info: {
        id: userID,
        sessionID: template,
        role: "user",
        time: { created: (t += 1000) },
        agent: "build",
        // A user message carries its variant inside its model.
        model: {
          providerID: model.providerID,
          modelID: model.modelID,
          ...variant,
        },
      },
      parts: userParts,
    } as unknown as MessageWithParts);
    const created = (t += 1000);
    const completed = (t += 5000);
    messages.push({
      info: {
        id: assistantID,
        sessionID: template,
        role: "assistant",
        time: { created, completed },
        parentID: userID,
        modelID: model.modelID,
        providerID: model.providerID,
        mode: "build",
        agent: "build",
        path: { cwd: directory, root: directory },
        cost: 0,
        tokens: {
          input: 0,
          output: 0,
          reasoning: 0,
          cache: { read: 0, write: 0 },
        },
        ...variant,
        finish: "stop",
      },
      parts: [
        {
          id: id("prt"),
          sessionID: template,
          messageID: assistantID,
          type: "text",
          text: turn.assistant,
          time: { start: created, end: completed },
        },
      ],
    } as unknown as MessageWithParts);
  }
  return planHistory(messages, template)(sessionID);
}

// The OpenCode 2 import records for a case. `mintID` supplies fresh
// ascending message IDs.
export function v2SourceHistory(
  corpus: Corpus,
  c: CorpusCase,
  model: ModelRef,
  mintID: (index: number) => string,
  now: number = Date.now(),
): V2Message[] {
  const ref = {
    providerID: model.providerID,
    id: model.modelID,
    ...(model.variant ? { variant: model.variant } : {}),
  };
  let t = now - 600_000;
  const records: Record<string, unknown>[] = [];
  for (const turn of turns(corpus, c)) {
    records.push({
      type: "user",
      text: turn.user,
      ...(turn.image
        ? {
            files: [
              {
                data: turn.image.base64,
                mime: turn.image.mime,
                source: { type: "inline" },
                name: turn.image.name,
              },
            ],
          }
        : {}),
      time: { created: (t += 1000) },
    });
    records.push({
      type: "assistant",
      agent: "build",
      model: ref,
      content: [{ type: "text", text: turn.assistant }],
      finish: "stop",
      time: { created: (t += 1000), completed: (t += 5000) },
    });
  }
  return records.map(
    (record, index) =>
      ({ ...record, id: mintID(index) }) as unknown as V2Message,
  );
}

// Offline proof that every case plans exactly one target turn with one
// preserved turn and offers task-aware retention nothing to classify, so a run
// needs no TypeSafe key and makes no classifier call. Uses the production
// planners and candidate finders for both hosts.
export async function checkCompactionPlans(corpus: Corpus): Promise<
  Array<{
    caseId: string;
    v1: { targets: number; preserved: number; candidates: number };
    v2: { targets: number; preserved: number; candidates: number };
  }>
> {
  const model = { providerID: "eval", modelID: "eval-model" };
  const keepTurns = Number(corpus.command.arguments);
  const out = [];
  for (const c of corpus.cases) {
    const v1 = v1SourceHistory(corpus, c, "ses_eval_check", "/eval", model);
    const v1Plan = planCompaction(v1, keepTurns);
    const v1Found = await findRetentionCandidates(v1, v1Plan.summarizedTurns);
    let counter = 0;
    const v2 = v2SourceHistory(
      corpus,
      c,
      model,
      () => `msg_eval_${String(counter++).padStart(4, "0")}`,
    );
    const v2Plan = planV2Compaction(v2, keepTurns);
    const eligible = new Set(
      v2Plan.summarized.flatMap(turn =>
        turn.assistants.flatMap(message =>
          v2ToolCalls(message).map(call => call.ref),
        ),
      ),
    );
    const v2Found = await collectRetentionCandidates(
      v2RetentionTurns(v2),
      eligible,
    );
    out.push({
      caseId: c.id,
      v1: {
        targets: v1Plan.summarizedTurns.length,
        preserved: v1Plan.preservedTurnCount,
        candidates: v1Found.candidates.length,
      },
      v2: {
        targets: v2Plan.summarized.length,
        preserved: v2Plan.preserved,
        candidates: v2Found.candidates.length,
      },
    });
  }
  return out;
}

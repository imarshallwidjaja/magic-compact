// Labelled keep/trim cases for the decision evaluation. A case is one
// completed tool call plus the context the production classifier sees; the
// label is a hand-written statement of whether the content is worth keeping.
// Requests come from the production builder, never from a copy of its wording.
import type { ToolPart } from "@opencode-ai/sdk/v2";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { z } from "zod";
import { trimWouldChangeContent } from "../../src/compact/prune";
import {
  buildRetentionRequests,
  RetentionClassificationError,
  type RetentionCandidate,
  type RetentionContext,
  type RetentionRequest,
} from "../../src/retention/classifier";

// Message text is shown to the operator; it never carries dataset content.
export class DecisionEvalError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "DecisionEvalError";
  }
}

export const DEFAULT_DATASET_PATH = fileURLToPath(
  new URL("./fixtures.json", import.meta.url),
);
export const DATASET_SPLITS = ["smoke", "development", "heldout"] as const;
export type DatasetSplit = (typeof DATASET_SPLITS)[number];

// IDs double as OpenAI question names, so they stay short and plain.
const identifier = z.string().regex(/^[a-z0-9][a-z0-9_-]{0,63}$/);

const caseSchema = z.strictObject({
  caseID: identifier,
  scenario: z.string().min(1),
  candidate: z.strictObject({
    tool: z.string().min(1),
    input: z.record(z.string(), z.unknown()),
    output: z.string(),
    turnRequest: z.string(),
    requestsBeforeLatest: z.number().int().min(0),
  }),
  label: z.strictObject({
    keep: z.boolean(),
    rationale: z.string().min(1),
  }),
});

const datasetSchema = z
  .strictObject({
    schemaVersion: z.literal(1),
    datasetID: identifier,
    version: z.string().regex(/^\d+\.\d+\.\d+$/),
    split: z.enum(DATASET_SPLITS),
    description: z.string().min(1),
    snapshots: z
      .array(
        z.strictObject({
          snapshotID: identifier,
          context: z.strictObject({
            latestUserRequest: z.string(),
            sessionSummary: z.string(),
          }),
          cases: z.array(caseSchema).min(1),
        }),
      )
      .min(1),
  })
  .superRefine((dataset, ctx) => {
    const seen = new Set<string>();
    for (const snapshot of dataset.snapshots) {
      for (const { caseID } of snapshot.cases) {
        if (seen.has(caseID)) {
          ctx.addIssue({
            code: "custom",
            message: `Duplicate caseID ${caseID}.`,
          });
        }
        seen.add(caseID);
      }
    }
    const snapshotIDs = dataset.snapshots.map(s => s.snapshotID);
    if (new Set(snapshotIDs).size !== snapshotIDs.length) {
      ctx.addIssue({ code: "custom", message: "Duplicate snapshotID." });
    }
  });

export type DecisionDataset = z.infer<typeof datasetSchema>;

export type EvalCase = {
  caseID: string;
  scenario: string;
  snapshotID: string;
  context: RetentionContext;
  // `id` is the caseID, so production request bodies and answers bind to it.
  candidate: RetentionCandidate;
  keep: boolean;
  rationale: string;
};

export function parseDataset(text: string): DecisionDataset {
  let json: unknown;
  try {
    json = JSON.parse(text);
  } catch {
    throw new DecisionEvalError("Dataset is not valid JSON.");
  }
  const parsed = datasetSchema.safeParse(json);
  if (!parsed.success) {
    const issues = parsed.error.issues
      .slice(0, 5)
      .map(issue => `${issue.path.join(".") || "(root)"}: ${issue.message}`);
    throw new DecisionEvalError(`Invalid dataset: ${issues.join("; ")}`);
  }
  return parsed.data;
}

export function loadDataset(
  path: string = DEFAULT_DATASET_PATH,
): DecisionDataset {
  let text: string;
  try {
    text = readFileSync(path, "utf8");
  } catch {
    throw new DecisionEvalError(`Cannot read dataset file ${path}.`);
  }
  return parseDataset(text);
}

export function datasetCases(dataset: DecisionDataset): EvalCase[] {
  return dataset.snapshots.flatMap(snapshot =>
    snapshot.cases.map(item => ({
      caseID: item.caseID,
      scenario: item.scenario,
      snapshotID: snapshot.snapshotID,
      context: snapshot.context,
      candidate: { id: item.caseID, ...item.candidate },
      keep: item.label.keep,
      rationale: item.label.rationale,
    })),
  );
}

// Key order does not matter for identity, so formatting edits never change it.
export function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (value !== null && typeof value === "object") {
    const record = value as Record<string, unknown>;
    return `{${Object.keys(record)
      .sort()
      .map(key => `${JSON.stringify(key)}:${canonicalJson(record[key])}`)
      .join(",")}}`;
  }
  return JSON.stringify(value);
}

export function sha256(text: string): string {
  return createHash("sha256").update(text).digest("hex");
}

export function datasetSha256(dataset: DecisionDataset): string {
  return sha256(canonicalJson(dataset));
}

// The singleton request the production classifier sends for this case.
export function buildCaseRequest(item: EvalCase): RetentionRequest {
  try {
    const requests = buildRetentionRequests(item.context, [item.candidate], 1);
    return requests[0]!;
  } catch (error) {
    if (error instanceof RetentionClassificationError) {
      throw new DecisionEvalError(
        `Case ${item.caseID} cannot be sent through the production request builder (${error.kind}).`,
      );
    }
    throw error;
  }
}

// Identity of what is asked, independent of provider and model: the
// production state and questions as built.
export function canonicalRequestSha256(request: RetentionRequest): string {
  return sha256(
    JSON.stringify({
      state: request.body.state,
      questions: request.body.questions,
    }),
  );
}

// Identity of one request per case, whichever hash the caller supplies.
export function requestSetSha256(
  entries: ReadonlyArray<{ caseID: string; sha256: string }>,
): string {
  return sha256(
    canonicalJson(
      [...entries]
        .sort((a, b) => (a.caseID < b.caseID ? -1 : 1))
        .map(entry => [entry.caseID, entry.sha256]),
    ),
  );
}

// The production question for candidate 0, whose wording every case shares.
export function semanticQuestion() {
  const [request] = buildRetentionRequests(
    { latestUserRequest: "", sessionSummary: "" },
    [
      {
        id: "probe",
        tool: "read",
        input: {},
        output: "",
        turnRequest: "",
        requestsBeforeLatest: 0,
      },
    ],
    1,
  );
  return request!.body.questions["candidate_0"]!;
}

export function semanticQuestionSha256(): string {
  return sha256(canonicalJson(semanticQuestion()));
}

// True when the production tool rules would change this completed call, so
// the case describes a decision the shipped command actually makes.
export function isProductionTrimCandidate(
  candidate: RetentionCandidate,
): Promise<boolean> {
  return trimWouldChangeContent({
    type: "tool",
    tool: candidate.tool,
    state: {
      status: "completed",
      input: candidate.input,
      output: candidate.output,
      metadata: {},
    },
  } as unknown as ToolPart);
}

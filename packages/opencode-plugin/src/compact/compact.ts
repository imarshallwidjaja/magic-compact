import type {
  Agent,
  PermissionRuleset,
  Session,
  TextPart,
} from "@opencode-ai/sdk/v2";
import { unwrap, type V2Client } from "../api";
import { countGptTokens } from "../stats/tokenize";
import {
  MAGIC_COMPACT_SUMMARIZER_AGENT,
  MAX_SUMMARY_OUTPUT_TOKENS,
  summaryMetadata,
  summaryPartID,
} from "./constants";
import {
  renderDeterministicSummary,
  renderStandardTurnEvidence,
  requiresDeterministicSummary,
} from "./high-risk";
import { type CompactionPlan, type Turn } from "./plan";
import { buildCompactionPrompt, turnID } from "./template";

export const MAX_SUMMARY_INPUT_TOKENS = 24_000;
export const MAX_CONCURRENT_SUMMARY_REQUESTS = 2;
export const SUMMARY_REQUEST_TIMEOUT_MS = 90_000;
const MAX_SUMMARY_ATTEMPTS = 2;
const SCRATCH_PERMISSION: PermissionRuleset = [
  { permission: "*", pattern: "*", action: "deny" },
];

type SummaryContractCategory =
  | "missing-text-output"
  | "invalid-json"
  | "invalid-output-shape"
  | "unknown-id"
  | "missing-id"
  | "invalid-summary-type"
  | "empty-summary"
  | "placeholder-summary"
  | "response-too-large";

class SummaryContractError extends Error {
  constructor(
    readonly category: SummaryContractCategory,
    readonly batch: number,
    readonly batchCount: number,
    readonly attempt: number,
  ) {
    super(
      `Summary contract failure (category=${category}, batch=${batch}/${batchCount}, attempt=${attempt}/${MAX_SUMMARY_ATTEMPTS}).`,
    );
    this.name = "SummaryContractError";
  }
}

type CompactionSettings = {
  model?: { providerID: string; modelID: string };
  variant?: string;
};

type SummaryBatch = {
  turns: Turn[];
  prompt: string;
};

export async function generateCompactionSummaries(
  v2: V2Client,
  session: Session,
  plan: CompactionPlan,
): Promise<string[]> {
  let settings: Promise<CompactionSettings> | undefined;
  const getSettings = async () =>
    (settings ??= resolveCompactionSettings(v2, session));
  const summaries = new Map<Turn, string>();
  const evidenceByTurn = new Map<Turn, string>();
  for (const turn of plan.summarizedTurns) {
    if (requiresDeterministicSummary(turn)) {
      summaries.set(turn, renderDeterministicSummary(turn));
    } else {
      evidenceByTurn.set(turn, renderStandardTurnEvidence(turn));
    }
  }
  const batches = createSummaryBatches(plan, evidenceByTurn);
  let nextBatch = 0;
  let stopped = false;
  const errors: unknown[] = [];

  const worker = async () => {
    while (!stopped) {
      const batchIndex = nextBatch;
      nextBatch += 1;
      const batch = batches[batchIndex];
      if (!batch) return;
      try {
        const selected = await generateSummaryBatch(
          v2,
          session,
          getSettings,
          batch,
          batchIndex + 1,
          batches.length,
        );
        for (const [index, turn] of batch.turns.entries()) {
          summaries.set(turn, selected[index]!);
        }
      } catch (error) {
        errors[batchIndex] = error;
        stopped = true;
      }
    }
  };

  await Promise.all(
    Array.from(
      { length: Math.min(MAX_CONCURRENT_SUMMARY_REQUESTS, batches.length) },
      worker,
    ),
  );
  const generationError = errors.find(error => error !== undefined);
  if (generationError !== undefined) throw generationError;

  return plan.summarizedTurns.map(turn => {
    const summary = summaries.get(turn);
    if (summary === undefined)
      throw new Error("Missing generated turn summary.");
    return summary;
  });
}

async function generateSummaryBatch(
  v2: V2Client,
  session: Session,
  getSettings: () => Promise<CompactionSettings>,
  batch: SummaryBatch,
  batchNumber: number,
  batchCount: number,
): Promise<string[]> {
  for (let attempt = 1; attempt <= MAX_SUMMARY_ATTEMPTS; attempt += 1) {
    try {
      return await runScratchPrompt(
        v2,
        session,
        getSettings,
        batch.prompt,
        batch.turns,
        { batch: batchNumber, batchCount, attempt },
      );
    } catch (error) {
      if (
        !(error instanceof SummaryContractError)
        || attempt === MAX_SUMMARY_ATTEMPTS
      ) {
        throw error;
      }
    }
  }
  throw new Error("Summary retry loop ended without a result.");
}

function createSummaryBatches(
  plan: CompactionPlan,
  evidenceByTurn: ReadonlyMap<Turn, string>,
): SummaryBatch[] {
  const batches: SummaryBatch[] = [];
  let current: Turn[] = [];
  const standard = plan.summarizedTurns.filter(turn =>
    evidenceByTurn.has(turn),
  );

  const createBatch = (turns: Turn[]): SummaryBatch => {
    const finalTurn = turns.at(-1)!;
    const sourceIndex = plan.summarizedTurns.indexOf(finalTurn);
    const nextTurn = plan.summarizedTurns[sourceIndex + 1] ?? plan.nextTurn;
    const prompt = buildCompactionPrompt(turns, nextTurn, evidenceByTurn);
    return { turns, prompt };
  };

  for (const turn of standard) {
    const candidate = createBatch([...current, turn]);
    if (countGptTokens(candidate.prompt) <= MAX_SUMMARY_INPUT_TOKENS) {
      current.push(turn);
      continue;
    }
    if (current.length === 0) {
      throw new Error(
        `One bounded turn evidence map exceeds the ${MAX_SUMMARY_INPUT_TOKENS}-token summary input cap.`,
      );
    }
    batches.push(createBatch(current));
    current = [turn];
    if (
      countGptTokens(createBatch(current).prompt) > MAX_SUMMARY_INPUT_TOKENS
    ) {
      throw new Error(
        `One bounded turn evidence map exceeds the ${MAX_SUMMARY_INPUT_TOKENS}-token summary input cap.`,
      );
    }
  }
  if (current.length > 0) batches.push(createBatch(current));
  return batches;
}

async function runScratchPrompt(
  v2: V2Client,
  sourceSession: Session,
  getSettings: () => Promise<CompactionSettings>,
  promptText: string,
  turns: Turn[],
  diagnostic: { batch: number; batchCount: number; attempt: number },
): Promise<string[]> {
  const route = {
    directory: sourceSession.directory,
    ...(sourceSession.workspaceID
      ? { workspace: sourceSession.workspaceID }
      : {}),
  };
  const scratch = unwrap(
    await v2.session.create({
      ...route,
      ...(sourceSession.workspaceID
        ? {
            workspaceID: sourceSession.workspaceID,
          }
        : {}),
      title: `[TEMP] ${sourceSession.title}`,
      permission: SCRATCH_PERMISSION,
    }),
  );
  let operationError: unknown;
  let result: string[] | undefined;
  try {
    const settings = await getSettings();
    let promptError: unknown;
    let body:
      | NonNullable<Awaited<ReturnType<V2Client["session"]["prompt"]>>["data"]>
      | undefined;
    const controller = new AbortController();
    const timeout = setTimeout(
      () => controller.abort(new Error("Summary request timed out.")),
      SUMMARY_REQUEST_TIMEOUT_MS,
    );
    try {
      body = unwrap(
        await v2.session.prompt(
          {
            sessionID: scratch.id,
            ...route,
            agent: MAGIC_COMPACT_SUMMARIZER_AGENT,
            ...(settings.model ? { model: settings.model } : {}),
            ...(settings.variant ? { variant: settings.variant } : {}),
            parts: [{ type: "text", text: promptText }],
          },
          { signal: controller.signal },
        ),
      );
    } catch (error) {
      promptError = controller.signal.aborted
        ? new Error("Summary request timed out.", { cause: error })
        : error;
    } finally {
      clearTimeout(timeout);
    }
    if (promptError === undefined && body?.info.error) {
      promptError = new Error(
        `Summary provider failed (${body.info.error.name}).`,
      );
    }

    if (
      promptError !== undefined
      && (controller.signal.aborted || isAbortError(promptError))
    ) {
      try {
        unwrap(
          await v2.session.abort({
            sessionID: scratch.id,
            ...route,
          }),
        );
      } catch (abortError) {
        promptError = combinePromptErrors(promptError, abortError);
      }
    }

    if (promptError !== undefined) throw promptError;

    result = validateSummaries(
      parseSummaryOutput(body!.parts, diagnostic),
      turns,
      diagnostic,
    );
  } catch (error) {
    operationError = error;
  }

  let deleteError: unknown;
  try {
    unwrap(await v2.session.delete({ sessionID: scratch.id, ...route }));
  } catch (error) {
    deleteError = error;
  }
  if (operationError !== undefined && deleteError !== undefined) {
    throw combinePromptErrors(operationError, deleteError);
  }
  if (operationError !== undefined) throw operationError;
  if (deleteError !== undefined) throw deleteError;
  return result!;
}

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === "AbortError";
}

async function resolveCompactionSettings(
  v2: V2Client,
  sourceSession: Session,
): Promise<CompactionSettings> {
  const agents = unwrap(
    await v2.app.agents({
      directory: sourceSession.directory,
      ...(sourceSession.workspaceID
        ? { workspace: sourceSession.workspaceID }
        : {}),
    }),
  );
  const compactionAgent = agents.find(
    (agent: Agent) =>
      agent.name === "compaction"
      && agent.native === true
      && agent.hidden === true,
  );
  if (!compactionAgent) {
    throw new Error(
      "Magic Compact requires the native hidden compaction agent, but it is missing or disabled.",
    );
  }
  if (compactionAgent.model) {
    if (
      !compactionAgent.model.providerID.trim()
      || !compactionAgent.model.modelID.trim()
    ) {
      throw new Error(
        "The native compaction agent has an invalid model configuration.",
      );
    }
    return { model: compactionAgent.model, variant: compactionAgent.variant };
  }
  return {
    model: sourceSession.model
      ? {
          providerID: sourceSession.model.providerID,
          modelID: sourceSession.model.id,
        }
      : undefined,
    variant: sourceSession.model?.variant,
  };
}

function parseSummaryOutput(
  parts: Array<{ type: string }>,
  diagnostic: { batch: number; batchCount: number; attempt: number },
): unknown {
  const responseText = parts
    .filter((part): part is TextPart => part.type === "text")
    .map(part => part.text)
    .join("");
  if (!responseText.trim()) {
    throw contractError("missing-text-output", diagnostic);
  }
  if (countGptTokens(responseText) > MAX_SUMMARY_OUTPUT_TOKENS) {
    throw contractError("response-too-large", diagnostic);
  }
  try {
    return JSON.parse(responseText);
  } catch {
    throw contractError("invalid-json", diagnostic);
  }
}

function validateSummaries(
  output: unknown,
  turns: Turn[],
  diagnostic: { batch: number; batchCount: number; attempt: number },
): string[] {
  if (output === null || typeof output !== "object" || Array.isArray(output)) {
    throw contractError("invalid-output-shape", diagnostic);
  }

  const expected = turns.map(turnID);
  const expectedSet = new Set(expected);
  const byID = output as Record<string, unknown>;
  if (Object.keys(byID).some(id => !expectedSet.has(id))) {
    throw contractError("unknown-id", diagnostic);
  }
  if (expected.some(id => !Object.prototype.hasOwnProperty.call(byID, id))) {
    throw contractError("missing-id", diagnostic);
  }

  return expected.map(id => {
    const value = byID[id];
    if (typeof value !== "string") {
      throw contractError("invalid-summary-type", diagnostic);
    }
    const summary = value.trim();
    if (!summary) throw contractError("empty-summary", diagnostic);
    if (isNonSubstantiveMarker(summary)) {
      throw contractError("placeholder-summary", diagnostic);
    }
    return summary;
  });
}

function contractError(
  category: SummaryContractCategory,
  diagnostic: { batch: number; batchCount: number; attempt: number },
): SummaryContractError {
  return new SummaryContractError(
    category,
    diagnostic.batch,
    diagnostic.batchCount,
    diagnostic.attempt,
  );
}

function isNonSubstantiveMarker(value: string): boolean {
  if (
    /replace with (?:your )?summary/i.test(value)
    || /^(?:preserve(?:d)?|name|describe)\b[^\n]*\bevidenc(?:e|ed)\b[^\n]*[.!]?$/i.test(
      value,
    )
    || /^(?:yes|done|complete|completed|none(?: evidenced)?|n\/a|not applicable|unknown|todo|tbd)\.?$/i.test(
      value,
    )
    || /^[A-Za-z][A-Za-z0-9_-]*[.!?]?$/.test(value)
    || /^(?:-+|\.+)$/.test(value)
  ) {
    return true;
  }
  for (const match of value.matchAll(/\[([^\]\r\n]*)\]/g)) {
    if (isTemplateBracketBody(match[1] ?? "")) return true;
  }
  return false;
}

function isTemplateBracketBody(body: string): boolean {
  const text = body.trim();
  if (!text) return false;
  return (
    /^(?:TODO|TBD)$/i.test(text)
    || /\b(?:TODO|TBD)\b/i.test(text)
    || /\b(?:replace|placeholder)\b/i.test(text)
  );
}

function combinePromptErrors(primary: unknown, secondary: unknown): Error {
  return new AggregateError(
    [primary, secondary],
    "Summary generation and scratch-session cleanup failed alongside one another.",
  );
}

export async function injectSummaries(
  v2: V2Client,
  sessionID: string,
  compactionTurns: Turn[],
  summaries: string[],
): Promise<void> {
  for (const [index, turn] of compactionTurns.entries()) {
    const summary = summaries[index];
    if (summary === undefined)
      throw new Error("Missing summary for assistant turn.");
    const firstAssistant = turn.assistants[0];
    if (!firstAssistant) throw new Error("Turn missing assistant message.");
    const part = {
      id: summaryPartID(firstAssistant.info.id),
      sessionID,
      messageID: firstAssistant.info.id,
      type: "text",
      text: summary,
      metadata: summaryMetadata(),
    } satisfies TextPart;
    unwrap(
      await v2.part.update({
        sessionID,
        messageID: firstAssistant.info.id,
        partID: part.id,
        part,
      }),
    );
  }
}

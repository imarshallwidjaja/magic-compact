import type { Agent, Session, TextPart, ToolPart } from "@opencode-ai/sdk/v2";
import { unwrap, type V2Client } from "../api";
import { installTemporaryOmissionSource } from "../storage/omission";
import {
  MAGIC_COMPACT_SUMMARIZER_AGENT,
  summaryMetadata,
  summaryPartID,
} from "./constants";
import { renderDeterministicHighRiskSummary } from "./high-risk";
export { DETERMINISTIC_HIGH_RISK_SUMMARY_MAX_BYTES } from "./high-risk";
import {
  assertNativeCheckpointSnapshotUnchanged,
  captureNativeCheckpointSnapshot,
  type CompactionPlan,
  EphemeralCheckpointChangedError,
  type NativeCheckpointSnapshot,
  type Turn,
} from "./plan";
import { buildCompactionPrompt, turnID } from "./template";

export const HIGH_RISK_SERIALIZED_BYTES = 153_600;
export const HIGH_RISK_MESSAGE_COUNT = 20;
export const HIGH_RISK_TOOL_COUNT = 32;
export const MAX_SUMMARY_BATCH_TURNS = 8;
const MAX_SUMMARY_ATTEMPTS = 2;

type SummaryContractCategory =
  | "provider-structured-output"
  | "missing-structured-output"
  | "invalid-output-shape"
  | "unknown-id"
  | "missing-id"
  | "invalid-summary-type"
  | "empty-summary"
  | "placeholder-summary";

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
  nextTurn: Turn | null;
  highRisk: boolean;
};

export async function generateCompactionSummaries(
  v2: V2Client,
  session: Session,
  plan: CompactionPlan,
  highRiskSummaries = prepareHighRiskSummaries(plan),
): Promise<string[]> {
  let settings: CompactionSettings | undefined;
  const getSettings = async () =>
    (settings ??= await resolveCompactionSettings(v2, session));
  const summaries: string[] = [];
  const batches = createSummaryBatches(plan);
  const batchCount = batches.filter(batch => !batch.highRisk).length;
  let batchNumber = 0;

  for (const batch of batches) {
    if (batch.highRisk) {
      const turn = batch.turns[0]!;
      const summary = highRiskSummaries.get(turn);
      if (summary === undefined) {
        throw new Error("Missing prepared deterministic high-risk summary.");
      }
      summaries.push(summary);
      continue;
    }
    batchNumber += 1;
    for (let attempt = 1; attempt <= MAX_SUMMARY_ATTEMPTS; attempt += 1) {
      try {
        const selected = await runEphemeralPrompt(
          v2,
          session,
          getSettings,
          buildCompactionPrompt(batch.turns, batch.nextTurn),
          batch.turns,
          { batch: batchNumber, batchCount, attempt },
        );
        summaries.push(...selected);
        break;
      } catch (error) {
        if (
          !(error instanceof SummaryContractError)
          || attempt === MAX_SUMMARY_ATTEMPTS
        ) {
          throw error;
        }
      }
    }
  }

  return summaries;
}

export function prepareHighRiskSummaries(
  plan: CompactionPlan,
): ReadonlyMap<Turn, string> {
  const summaries = new Map<Turn, string>();
  for (const turn of plan.summarizedTurns) {
    if (!isHighRiskTurn(turn)) continue;
    summaries.set(turn, renderDeterministicHighRiskSummary(turn));
  }
  return summaries;
}

function isHighRiskTurn(turn: Turn): boolean {
  const messages = [...turn.user, ...turn.assistants];
  const tools = messages.flatMap(message =>
    message.parts.filter((part): part is ToolPart => part.type === "tool"),
  );
  return (
    Buffer.byteLength(JSON.stringify(turn)) >= HIGH_RISK_SERIALIZED_BYTES
    || messages.length >= HIGH_RISK_MESSAGE_COUNT
    || tools.length >= HIGH_RISK_TOOL_COUNT
    || tools.some(tool => tool.state.status !== "completed")
  );
}

function createSummaryBatches(plan: CompactionPlan): SummaryBatch[] {
  const batches: SummaryBatch[] = [];
  let standard: Turn[] = [];
  const flush = (nextTurn: Turn | null) => {
    if (standard.length === 0) return;
    batches.push({ turns: standard, nextTurn, highRisk: false });
    standard = [];
  };

  for (const [index, turn] of plan.summarizedTurns.entries()) {
    const sourceNext = plan.summarizedTurns[index + 1] ?? plan.nextTurn;
    if (isHighRiskTurn(turn)) {
      flush(turn);
      batches.push({ turns: [turn], nextTurn: sourceNext, highRisk: true });
      continue;
    }
    standard.push(turn);
    if (standard.length === MAX_SUMMARY_BATCH_TURNS) flush(sourceNext);
  }
  flush(plan.nextTurn);
  return batches;
}

async function runEphemeralPrompt(
  v2: V2Client,
  sourceSession: Session,
  getSettings: () => Promise<CompactionSettings>,
  promptText: string,
  turns: Turn[],
  diagnostic: { batch: number; batchCount: number; attempt: number },
): Promise<string[]> {
  const ephemeral = unwrap(
    await v2.session.fork({ sessionID: sourceSession.id }),
  );
  let operationError: unknown;
  let result: string[] | undefined;
  let clearOmissionSource: (() => void) | undefined;
  try {
    let checkpointSnapshot: NativeCheckpointSnapshot;
    try {
      checkpointSnapshot = captureNativeCheckpointSnapshot(
        unwrap(await v2.session.messages({ sessionID: ephemeral.id })),
      );
    } catch (error) {
      throw new EphemeralCheckpointChangedError(error);
    }
    clearOmissionSource = installTemporaryOmissionSource(
      ephemeral.id,
      sourceSession.id,
    );
    unwrap(
      await v2.session.update({
        sessionID: ephemeral.id,
        title: `[TEMP] ${sourceSession.title}`,
      }),
    );
    const settings = await getSettings();
    let promptError: unknown;
    let body:
      | NonNullable<Awaited<ReturnType<V2Client["session"]["prompt"]>>["data"]>
      | undefined;
    try {
      body = unwrap(
        await v2.session.prompt({
          sessionID: ephemeral.id,
          agent: MAGIC_COMPACT_SUMMARIZER_AGENT,
          ...(settings.model ? { model: settings.model } : {}),
          ...(settings.variant ? { variant: settings.variant } : {}),
          format: buildSummaryFormat(turns),
          parts: [{ type: "text", text: promptText }],
        }),
      );
    } catch (error) {
      promptError = error;
    }
    if (promptError === undefined && body?.info.error) {
      promptError = summaryResponseError(body.info.error.name, diagnostic);
    }

    let artifactError: unknown;
    try {
      assertNativeCheckpointSnapshotUnchanged(
        unwrap(await v2.session.messages({ sessionID: ephemeral.id })),
        checkpointSnapshot,
      );
    } catch (error) {
      artifactError = new EphemeralCheckpointChangedError(error);
    }
    if (artifactError !== undefined) {
      throw promptError === undefined
        ? artifactError
        : combinePromptErrors(artifactError, promptError);
    }
    if (promptError !== undefined) throw promptError;

    result = validateSummaries(body!.info.structured, turns, diagnostic);
  } catch (error) {
    operationError = error;
  } finally {
    clearOmissionSource?.();
  }

  let deleteError: unknown;
  try {
    unwrap(await v2.session.delete({ sessionID: ephemeral.id }));
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

function buildSummaryFormat(turns: Turn[]) {
  const expected = turns.map(turnID);
  return {
    type: "json_schema" as const,
    retryCount: 0,
    schema: {
      type: "object",
      properties: Object.fromEntries(
        expected.map(id => [id, { type: "string" }]),
      ),
      required: expected,
      additionalProperties: false,
    },
  };
}

function summaryResponseError(
  name: string,
  diagnostic: { batch: number; batchCount: number; attempt: number },
): Error {
  if (name === "StructuredOutputError") {
    return contractError("provider-structured-output", diagnostic);
  }
  return new Error(`Summary provider failed (${name}).`);
}

function validateSummaries(
  output: unknown,
  turns: Turn[],
  diagnostic: { batch: number; batchCount: number; attempt: number },
): string[] {
  if (output === undefined) {
    throw contractError("missing-structured-output", diagnostic);
  }
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
    "Summary generation, ephemeral checkpoint validation, or cleanup failed alongside another operation.",
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

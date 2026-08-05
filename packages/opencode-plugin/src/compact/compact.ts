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
import { buildCompactionPrompt, isXmlCodePoint, turnID } from "./template";

export const HIGH_RISK_SERIALIZED_BYTES = 153_600;
export const HIGH_RISK_MESSAGE_COUNT = 20;
export const HIGH_RISK_TOOL_COUNT = 32;
export const MAX_SUMMARY_BATCH_TURNS = 8;

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

  for (const batch of createSummaryBatches(plan)) {
    if (batch.highRisk) {
      const turn = batch.turns[0]!;
      const summary = highRiskSummaries.get(turn);
      if (summary === undefined) {
        throw new Error("Missing prepared deterministic high-risk summary.");
      }
      summaries.push(summary);
      continue;
    }
    const selected = await runEphemeralPrompt(
      v2,
      session,
      getSettings,
      buildCompactionPrompt(batch.turns, batch.nextTurn),
      batch.turns,
    );
    summaries.push(...selected);
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
          parts: [{ type: "text", text: promptText }],
        }),
      );
    } catch (error) {
      promptError = error;
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

    const textResponse = body!.parts
      .filter((part): part is TextPart => part.type === "text")
      .map(part => part.text)
      .join("\n");
    result = parseSummaries(textResponse, turns);
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

function parseSummaries(responseText: string, turns: Turn[]): string[] {
  const document = responseText;
  const rootOpen = "<summary>";
  const rootClose = "</summary>";
  const assistantOpen = '<assistant id="';
  const assistantClose = "</assistant>";
  let cursor = 0;
  const skipWhitespace = () => {
    while (
      document[cursor] === " "
      || document[cursor] === "\t"
      || document[cursor] === "\n"
      || document[cursor] === "\r"
    ) {
      cursor += 1;
    }
  };
  skipWhitespace();
  if (!document.startsWith(rootOpen, cursor)) {
    throw new Error(
      "Summary response must contain exactly one <summary> root.",
    );
  }

  cursor += rootOpen.length;
  const parsed: Array<{ id: string; summary: string }> = [];
  while (true) {
    skipWhitespace();
    if (document.startsWith(rootClose, cursor)) {
      cursor += rootClose.length;
      break;
    }
    if (!document.startsWith(assistantOpen, cursor)) {
      throw new Error(
        "Summary response contains text or an unsupported element inside <summary>.",
      );
    }

    cursor += assistantOpen.length;
    const idEnd = document.indexOf('"', cursor);
    if (idEnd === -1 || document[idEnd + 1] !== ">") {
      throw new Error(
        "Summary response contains a malformed assistant ID or attributes.",
      );
    }
    const id = document.slice(cursor, idEnd);
    cursor = idEnd + 2;

    const close = document.indexOf(assistantClose, cursor);
    const nested = document.indexOf("<", cursor);
    if (close === -1 || (nested !== -1 && nested < close)) {
      throw new Error(
        "Summary response contains a nested, unexpected, or unclosed assistant element.",
      );
    }
    const summary = decodeXmlText(document.slice(cursor, close)).trim();
    parsed.push({ id, summary });
    cursor = close + assistantClose.length;
  }
  skipWhitespace();
  if (cursor !== document.length) {
    throw new Error(
      "Summary response contains text or elements outside its root.",
    );
  }

  const expected = turns.map(turnID);
  const expectedSet = new Set(expected);
  const byID = new Map<string, string>();
  for (const { id, summary } of parsed) {
    if (!/^[A-Za-z0-9_-]+$/.test(id)) {
      throw new Error(`Summary response contains malformed turn ID: ${id}`);
    }
    if (!expectedSet.has(id)) throw new Error(`Unknown summary turn ID: ${id}`);
    if (byID.has(id)) throw new Error(`Duplicate summary turn ID: ${id}`);
    if (!summary) throw new Error(`Empty summary for turn ID: ${id}`);
    if (isNonSubstantiveMarker(summary)) {
      throw new Error(`Placeholder summary for turn ID: ${id}`);
    }
    byID.set(id, summary);
  }
  const missing = expected.filter(id => !byID.has(id));
  if (missing.length > 0) {
    throw new Error(`Missing summaries for turn IDs: ${missing.join(", ")}`);
  }
  return expected.map(id => byID.get(id)!);
}

function isNonSubstantiveMarker(value: string): boolean {
  const normalized = value.trim();
  if (
    /replace with (?:your )?summary/i.test(normalized)
    || /^(?:preserve(?:d)?|name|describe)\b[^\n]*\bevidenc(?:e|ed)\b[^\n]*[.!]?$/i.test(
      normalized,
    )
    || /^(?:yes|done|complete|completed|none(?: evidenced)?|n\/a|not applicable|unknown|todo|tbd)\.?$/i.test(
      normalized,
    )
    || /^[A-Za-z][A-Za-z0-9_-]*[.!?]?$/.test(normalized)
    || /^(?:-+|\.+)$/.test(normalized)
  ) {
    return true;
  }
  for (const match of normalized.matchAll(/\[([^\]\r\n]*)\]/g)) {
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

function decodeXmlText(value: string): string {
  if (value.includes("]]>")) {
    throw new Error("Summary response contains an invalid XML text delimiter.");
  }
  let decoded = "";
  let cursor = 0;
  while (cursor < value.length) {
    const ampersand = value.indexOf("&", cursor);
    if (ampersand === -1) {
      decoded += value.slice(cursor);
      break;
    }
    decoded += value.slice(cursor, ampersand);
    const semicolon = value.indexOf(";", ampersand + 1);
    if (semicolon === -1) {
      throw new Error("Summary response contains a malformed XML entity.");
    }
    const entity = value.slice(ampersand + 1, semicolon);
    const named = {
      amp: "&",
      lt: "<",
      gt: ">",
      quot: '"',
      apos: "'",
    }[entity];
    if (named !== undefined) {
      decoded += named;
    } else {
      const numeric = entity.startsWith("#x")
        ? Number.parseInt(entity.slice(2), 16)
        : entity.startsWith("#")
          ? Number.parseInt(entity.slice(1), 10)
          : Number.NaN;
      const validNumericSyntax = entity.startsWith("#x")
        ? /^#x[0-9A-Fa-f]+$/.test(entity)
        : /^#[0-9]+$/.test(entity);
      if (!validNumericSyntax || !isXmlCodePoint(numeric)) {
        throw new Error("Summary response contains a malformed XML entity.");
      }
      decoded += String.fromCodePoint(numeric);
    }
    cursor = semicolon + 1;
  }
  for (const character of decoded) {
    if (!isXmlCodePoint(character.codePointAt(0)!)) {
      throw new Error("Summary response contains an invalid XML character.");
    }
  }
  return decoded;
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

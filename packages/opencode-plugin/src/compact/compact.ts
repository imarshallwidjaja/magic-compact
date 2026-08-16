import type {
  Agent,
  PermissionRuleset,
  Session,
  TextPart,
} from "@opencode-ai/sdk/v2";
import { unwrap, type V2Client } from "../api";
import { countGptTokensWithinLimit } from "../stats/tokenize";
import {
  MAGIC_COMPACT_SUMMARIZER_AGENT,
  MAX_SUMMARY_INPUT_TOKENS,
  MAX_SUMMARY_OUTPUT_TOKENS,
  summaryMetadata,
  summaryPartID,
} from "./constants";
import {
  classifyTurnOutcome,
  type CapsuleOutcome,
  renderDeterministicSummary,
  renderStandardTurnEvidence,
  requiresDeterministicSummary,
} from "./high-risk";
import { type CompactionPlan, type Turn } from "./plan";
import { buildCompactionPrompt, turnID } from "./template";

export { MAX_SUMMARY_INPUT_TOKENS } from "./constants";
export const MAX_CONCURRENT_SUMMARY_REQUESTS = 2;
export const SUMMARY_REQUEST_TIMEOUT_MS = 90_000;
const MAX_SUMMARY_ATTEMPTS = 2;
const SUMMARY_SEGMENT_BREAK = /\r\n|[\n\r\v\f\u0085\u2028\u2029]/;
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
  | "machine-evidence-summary"
  | "unsafe-presentation-control"
  | "response-too-large";

type SummaryAttemptDiagnostic = {
  batch: number;
  batchCount: number;
  attempt: number;
};

type SummaryDiagnostic = SummaryAttemptDiagnostic & {
  sourceSessionID: string;
  turnID: string;
};

class SummaryContractError extends Error {
  constructor(
    readonly category: SummaryContractCategory,
    readonly batch: number,
    readonly batchCount: number,
    readonly attempt: number,
    readonly sourceSessionID: string,
    readonly turnID: string,
  ) {
    super(
      `Summary contract failure (category=${category}, batch=${batch}/${batchCount}, attempt=${attempt}/${MAX_SUMMARY_ATTEMPTS}, sourceSession=${sourceSessionID}, turn=${turnID}).`,
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
    if (
      countGptTokensWithinLimit(candidate.prompt, MAX_SUMMARY_INPUT_TOKENS)
      !== null
    ) {
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
      countGptTokensWithinLimit(
        createBatch(current).prompt,
        MAX_SUMMARY_INPUT_TOKENS,
      ) === null
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
  diagnostic: SummaryAttemptDiagnostic,
): Promise<string[]> {
  const contractDiagnostic: SummaryDiagnostic = {
    ...diagnostic,
    sourceSessionID: sourceSession.id,
    turnID: turns.map(turnID).join(","),
  };
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
      parseSummaryOutput(body!.parts, contractDiagnostic),
      turns,
      contractDiagnostic,
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
  diagnostic: SummaryDiagnostic,
): unknown {
  const responseText = parts
    .filter((part): part is TextPart => part.type === "text")
    .map(part => part.text)
    .join("");
  if (!responseText.trim()) {
    throw contractError("missing-text-output", diagnostic);
  }
  if (
    countGptTokensWithinLimit(responseText, MAX_SUMMARY_OUTPUT_TOKENS) === null
  ) {
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
  diagnostic: SummaryDiagnostic,
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

  return expected.map((id, index) => {
    const turnDiagnostic = { ...diagnostic, turnID: id };
    const value = byID[id];
    if (typeof value !== "string") {
      throw contractError("invalid-summary-type", turnDiagnostic);
    }
    if (containsMachineEvidence(value)) {
      throw contractError("machine-evidence-summary", turnDiagnostic);
    }
    if (containsUnsafePresentationControl(value)) {
      throw contractError("unsafe-presentation-control", turnDiagnostic);
    }
    const body = foldSummaryBody(value);
    if (!body) throw contractError("empty-summary", turnDiagnostic);
    if (isNonSubstantiveMarker(body)) {
      throw contractError("placeholder-summary", turnDiagnostic);
    }
    return frameSummaryBody(body, classifyTurnOutcome(turns[index]!));
  });
}

function containsUnsafePresentationControl(body: string): boolean {
  return body
    .split(SUMMARY_SEGMENT_BREAK)
    .some(segment => hasUnsafePresentationControl(segment.trim()));
}

function hasUnsafePresentationControl(value: string): boolean {
  for (let index = 0; index < value.length; index += 1) {
    const codeUnit = value.charCodeAt(index);
    if (codeUnit >= 0xd800 && codeUnit <= 0xdbff) {
      const low = value.charCodeAt(index + 1);
      if (!(low >= 0xdc00 && low <= 0xdfff)) return true;
      index += 1;
      continue;
    }
    if (
      (codeUnit >= 0xdc00 && codeUnit <= 0xdfff)
      || codeUnit <= 0x1f
      || (codeUnit >= 0x7f && codeUnit <= 0x9f)
      || (codeUnit >= 0x202a && codeUnit <= 0x202e)
      || (codeUnit >= 0x2066 && codeUnit <= 0x2069)
    ) {
      return true;
    }
  }
  return false;
}

function foldSummaryBody(body: string): string {
  return body
    .split(SUMMARY_SEGMENT_BREAK)
    .map(segment => segment.trim())
    .filter(Boolean)
    .join(" ")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;");
}

function frameSummaryBody(body: string, outcome: CapsuleOutcome): string {
  const capsule = `Outcome: ${outcome}\nWork: ${body}`;
  assertHostFramedCapsule(capsule, outcome);
  return capsule;
}

function assertHostFramedCapsule(
  capsule: string,
  outcome: CapsuleOutcome,
): void {
  const lines = capsule.split("\n");
  const work = lines[1]?.slice("Work: ".length) ?? "";
  if (
    lines.length !== 2
    || lines[0] !== `Outcome: ${outcome}`
    || !lines[1]?.startsWith("Work: ")
    || !work.trim()
    || isNonSubstantiveMarker(work)
    || /[\r\v\f\u0085\u2028\u2029]/.test(capsule)
    || hasUnsafePresentationControl(lines.join(""))
    || capsule.includes("<")
    || /&(?!(?:amp|lt);)/.test(capsule)
  ) {
    throw new Error("Host-framed summary capsule invariant failed.");
  }
}

const PROTECTED_PRODUCER_FIELDS = new Set([
  "sourceBytes",
  "textEncoding",
  "textSegments",
  "assistant-text-records",
  "evidence-records",
  "required-records",
  "required-record-counts",
  "required-record-total",
  "required-record-overflow",
  "required-record-omitted-disclosure",
  "omitted-record-count",
]);
const PRODUCER_RECORD_KINDS = new Set([
  "assistant-text",
  "required-assistant-text",
  "tool-terminal",
  "tool-mutation",
  "tool-status",
  "required-tool",
  "assistant-evidence",
  "required-assistant-evidence",
  "assistant-message-status",
  "required-assistant-message-status",
]);
const PRODUCER_LINE_LABELS = [
  "assistant-text-records",
  "evidence-records",
  "required-records",
  "Required deterministic evidence overflow capsule",
  "Historical assistant text (authoritative)",
  "Historical terminal evidence (authoritative)",
];

function containsMachineEvidence(summary: string): boolean {
  const pending = [summary];
  const inspected = new Set<string>();
  while (pending.length > 0) {
    const text = pending.pop()!;
    if (inspected.has(text)) continue;
    inspected.add(text);
    if (
      text
        .split(SUMMARY_SEGMENT_BREAK)
        .some(segment => containsProducerAssignment(segment))
    ) {
      return true;
    }
    for (const value of embeddedJsonValues(text)) {
      if (containsProducerSchema(value, pending)) return true;
    }
    if (text.includes('\\"')) pending.push(text.replaceAll('\\"', '"'));
  }
  return false;
}

function containsProducerAssignment(segment: string): boolean {
  const line = stripNonProsePrefix(segment);
  if (!line) return false;
  if (
    PRODUCER_LINE_LABELS.some(
      label => line === label || line.startsWith(`${label}:`),
    )
  ) {
    return true;
  }
  const assignment = leadingAssignment(line);
  if (!assignment) return false;
  if (PROTECTED_PRODUCER_FIELDS.has(assignment.key)) {
    return true;
  }
  if (assignment.key === "windows") {
    return assignment.value.startsWith("[");
  }
  return (
    assignment.key === "kind"
    && PRODUCER_RECORD_KINDS.has(leadingStringValue(assignment.value) ?? "")
  );
}

function stripNonProsePrefix(segment: string): string {
  let line = segment.trim();
  while (line) {
    const prefix = line.match(
      /^(?:[>#*+-]+|\d+[.)]|\(\d+\)|\[[ xX]\]|\{)[ \t]*/,
    )?.[0];
    if (!prefix) break;
    line = line.slice(prefix.length).trimStart();
  }
  return line;
}

function leadingAssignment(
  line: string,
): { key: string; value: string } | null {
  let key: string;
  let end: number;
  if (line[0] === '"' || line[0] === "'" || line[0] === "`") {
    const quoted = leadingQuotedValue(line);
    if (!quoted) return null;
    key = quoted.text;
    end = quoted.end;
  } else {
    const match = line.match(/^[A-Za-z][A-Za-z0-9-]*/)?.[0];
    if (!match) return null;
    key = match;
    end = match.length;
  }
  while (line[end] === " " || line[end] === "\t") end += 1;
  if (line[end] !== ":" && line[end] !== "=") return null;
  return { key, value: line.slice(end + 1).trim() };
}

function leadingStringValue(value: string): string | null {
  if (value[0] === '"' || value[0] === "'" || value[0] === "`") {
    return leadingQuotedValue(value)?.text ?? null;
  }
  return value.match(/^[A-Za-z][A-Za-z0-9-]*/)?.[0] ?? null;
}

function leadingQuotedValue(
  value: string,
): { text: string; end: number } | null {
  const quote = value[0];
  if (quote !== '"' && quote !== "'" && quote !== "`") return null;
  const end = quotedStringEnd(value, 0, quote);
  if (end < 0) return null;
  if (quote === '"') {
    try {
      const parsed = JSON.parse(value.slice(0, end)) as unknown;
      return typeof parsed === "string" ? { text: parsed, end } : null;
    } catch {
      return null;
    }
  }
  return { text: value.slice(1, end - 1), end };
}

function quotedStringEnd(value: string, start: number, quote: string): number {
  let escaped = false;
  for (let index = start + 1; index < value.length; index += 1) {
    const character = value[index]!;
    if (escaped) {
      escaped = false;
    } else if (character === "\\") {
      escaped = true;
    } else if (character === quote) {
      return index + 1;
    }
  }
  return -1;
}

function embeddedJsonValues(text: string): unknown[] {
  const values: unknown[] = [];
  for (let index = 0; index < text.length; index += 1) {
    const opener = text[index];
    if (opener !== "{" && opener !== "[") continue;
    if (opener === "{" && !couldStartJsonObject(text, index)) continue;
    const end = jsonValueEnd(text, index);
    if (end < 0) continue;
    try {
      values.push(JSON.parse(text.slice(index, end)) as unknown);
      index = end - 1;
    } catch {
      continue;
    }
  }
  return values;
}

function couldStartJsonObject(text: string, start: number): boolean {
  let index = start + 1;
  while (
    text[index] === " "
    || text[index] === "\t"
    || text[index] === "\n"
    || text[index] === "\r"
  ) {
    index += 1;
  }
  return text[index] === '"' || text[index] === "}";
}

function jsonValueEnd(text: string, start: number): number {
  const stack = [text[start]!];
  let inString = false;
  let escaped = false;
  for (let index = start + 1; index < text.length; index += 1) {
    const character = text[index]!;
    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (character === "\\") {
        escaped = true;
      } else if (character === '"') {
        inString = false;
      }
      continue;
    }
    if (character === '"') {
      inString = true;
    } else if (character === "{" || character === "[") {
      stack.push(character);
    } else if (character === "}" || character === "]") {
      const expected = character === "}" ? "{" : "[";
      if (stack.pop() !== expected) return -1;
      if (stack.length === 0) return index + 1;
    }
  }
  return -1;
}

function containsProducerSchema(value: unknown, pending: string[]): boolean {
  const values: unknown[] = [value];
  while (values.length > 0) {
    const current = values.pop();
    if (typeof current === "string") {
      pending.push(current);
      continue;
    }
    if (Array.isArray(current)) {
      values.push(...current);
      continue;
    }
    if (current === null || typeof current !== "object") continue;
    for (const [key, field] of Object.entries(
      current as Record<string, unknown>,
    )) {
      if (
        PROTECTED_PRODUCER_FIELDS.has(key)
        || (key === "windows" && Array.isArray(field))
        || (key === "kind"
          && typeof field === "string"
          && PRODUCER_RECORD_KINDS.has(field))
      ) {
        return true;
      }
      values.push(field);
    }
  }
  return false;
}

function contractError(
  category: SummaryContractCategory,
  diagnostic: SummaryDiagnostic,
): SummaryContractError {
  return new SummaryContractError(
    category,
    diagnostic.batch,
    diagnostic.batchCount,
    diagnostic.attempt,
    diagnostic.sourceSessionID,
    diagnostic.turnID,
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

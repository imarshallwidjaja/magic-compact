import type { TextPart, ToolPart } from "@opencode-ai/sdk/v2";
import { isRecord } from "../util";
import type { Turn } from "./plan";
import { isXmlCodePoint, stripInvalidXmlCodePoints } from "./template";

export const DETERMINISTIC_HIGH_RISK_SUMMARY_MAX_BYTES = 64 * 1024;

const MAX_SOURCE_WINDOWS = 12;

type ResultPolarity = {
  failed: boolean;
  notRun: boolean;
  incomplete: boolean;
};

type BudgetRecord = {
  priority: number;
  order: number;
  full: string;
  compact: string;
  minimum: string;
};

type AssistantRecord = BudgetRecord & {
  required: boolean;
};

type EvidenceRecord = BudgetRecord & {
  verification: boolean;
  vcs: boolean;
  unresolved: boolean;
};

type RecordChoice = "minimum" | "compact" | "full";

type SummarySection = {
  label: string;
  recordLabel: "assistant-text-records" | "evidence-records";
  records: BudgetRecord[];
  required: Set<BudgetRecord>;
  noneWhenEmpty: boolean;
  choices: Map<BudgetRecord, RecordChoice>;
};

type AssistantSource = {
  messageID: string;
  partID: string;
  messageIndex: number;
  order: number;
  text: string;
};

type ToolSource = {
  index: number;
  tool: ToolPart;
  command: string | null;
  input: string | null;
  output: string | null;
  error: string | null;
  title: string | null;
  categories: string[];
  anchors: string[];
  verification: boolean;
  vcs: boolean;
  unresolved: boolean;
  failed: boolean;
  notRun: boolean;
  incomplete: boolean;
  mutation: boolean;
};

export class HighRiskEvidenceOverflowError extends Error {
  constructor(requiredBytes: number) {
    super(
      `Required high-risk evidence needs ${requiredBytes} bytes and exceeds the 64 KiB deterministic summary cap.`,
    );
    this.name = "HighRiskEvidenceOverflowError";
  }
}

export function renderDeterministicHighRiskSummary(turn: Turn): string {
  const assistantSources = collectAssistantSources(turn);
  const assistantRecords = buildAssistantRecords(assistantSources);
  const toolSources = collectToolSources(turn);
  const toolRecords = toolSources.map(buildToolRecord);
  const assistantEvidence = buildAssistantEvidenceRecords(assistantSources);
  const evidenceRecords = [...toolRecords, ...assistantEvidence];
  const unresolved = evidenceRecords.some(record => record.unresolved);
  const hasExecutionOrMutation =
    toolSources.length > 0
    || assistantSources.some(source =>
      hasAssistantMutationEvidence(source.text),
    );
  const hasVerificationOrVcs = evidenceRecords.some(
    record => record.verification || record.vcs,
  );
  const outcome = unresolved
    ? "partial"
    : !hasExecutionOrMutation && !hasVerificationOrVcs
      ? "analysis-only"
      : "completed";

  const sections: SummarySection[] = [
    createSection(
      "Historical assistant text (authoritative)",
      "assistant-text-records",
      assistantRecords,
      assistantRecords.filter(record => record.required),
      true,
    ),
    createSection(
      "Verification",
      "evidence-records",
      evidenceRecords.filter(record => record.verification),
      evidenceRecords.filter(record => record.verification),
      true,
    ),
    createSection(
      "VCS/Deployment",
      "evidence-records",
      evidenceRecords.filter(record => record.vcs),
      evidenceRecords.filter(record => record.vcs),
      true,
    ),
    createSection(
      "Unresolved",
      "evidence-records",
      evidenceRecords.filter(record => record.unresolved),
      evidenceRecords.filter(record => record.unresolved),
      true,
    ),
    createSection(
      "Historical terminal evidence (authoritative)",
      "evidence-records",
      evidenceRecords,
      [],
      false,
    ),
  ];

  reserveRequiredRecords(outcome, sections);
  fillOptionalRecords(outcome, sections);
  const summary = renderSummary(outcome, sections);
  if (Buffer.byteLength(summary) > DETERMINISTIC_HIGH_RISK_SUMMARY_MAX_BYTES) {
    throw new Error("Deterministic high-risk summary exceeded its byte cap.");
  }
  return summary;
}

function collectAssistantSources(turn: Turn): AssistantSource[] {
  const sources: AssistantSource[] = [];
  let order = 0;
  for (const [messageIndex, message] of turn.assistants.entries()) {
    for (const part of message.parts) {
      if (part.type !== "text" || !isSubstantiveAssistantText(part)) continue;
      sources.push({
        messageID: sanitizeIdentifier(message.info.id),
        partID: sanitizeIdentifier(part.id),
        messageIndex,
        order: order++,
        text: part.text,
      });
    }
  }
  return sources;
}

function isSubstantiveAssistantText(part: TextPart): boolean {
  if (part.ignored || part.synthetic) return false;
  const metadata = part.metadata;
  const magicCompact = isRecord(metadata) ? metadata["magicCompact"] : null;
  const text = part.text.trim();
  return (
    text !== ""
    && !(isRecord(magicCompact) && magicCompact["summary"] === true)
    && !isPlaceholderText(text)
  );
}

function isPlaceholderText(value: string): boolean {
  const text = value.trim();
  return (
    /replace with (?:your )?summary/i.test(text)
    || /^\[(?:TODO|TBD|Replace\b[^\]]*)\]$/i.test(text)
    || /^(?:TODO|TBD|unknown|N\/A)\.?$/i.test(text)
    || /^(?:\.\.\.|\u2026|-+|\.+)$/.test(text)
    || /^(?:yes|done|complete|completed|none)\.?$/i.test(text)
  );
}

function buildAssistantRecords(sources: AssistantSource[]): AssistantRecord[] {
  const finalMessageIndex = sources.at(-1)?.messageIndex ?? -1;
  return sources.map(source => {
    const required = source.messageIndex === finalMessageIndex;
    return {
      priority: required ? 0 : 4,
      order: source.order,
      required,
      full: renderAssistantRecord(source, Number.POSITIVE_INFINITY),
      compact: renderAssistantRecord(source, 2_048),
      minimum: renderAssistantRecord(source, 768),
    };
  });
}

function renderAssistantRecord(
  source: AssistantSource,
  maximumBytes: number,
): string {
  const base: Record<string, unknown> = {
    kind: "assistant-text",
    messageID: source.messageID,
    partID: source.partID,
    sourceBytes: Buffer.byteLength(source.text),
  };
  const paths = extractPathAnchors(source.text);
  const anchors = extractEvidenceAnchors(source.text);
  if (paths.length > 0) base["paths"] = paths;
  if (anchors.length > 0) base["anchors"] = anchors;
  const exact = {
    ...base,
    text: source.text,
  };
  const serialized = serializeRecord(exact);
  if (Buffer.byteLength(serialized) <= maximumBytes) return serialized;
  return fitWindowedRecord(base, source.text, maximumBytes);
}

function collectToolSources(turn: Turn): ToolSource[] {
  const tools = [...turn.user, ...turn.assistants].flatMap(message =>
    message.parts.filter((part): part is ToolPart => part.type === "tool"),
  );
  return tools.map((tool, index) => {
    const state = tool.state as unknown as Record<string, unknown>;
    const input = sourceValue(state["input"]);
    const output = sourceValue(state["output"]);
    const error = sourceValue(state["error"]);
    const title = sourceValue(state["title"]);
    const command = sourceCommand(state["input"]);
    const resultText = [output, error].filter(isString).join("\n");
    const fullText = [tool.tool, tool.state.status, input, output, error, title]
      .filter(isString)
      .join("\n");
    const verification = hasToolVerificationContext(tool.tool, command);
    const vcs = hasToolVcsContext(tool.tool, command);
    const commandRunning = hasCommandRunningIdentity(tool.tool);
    const result = verification
      ? classifyResultPolarity(resultText)
      : EMPTY_RESULT_POLARITY;
    const vcsResult = vcs
      ? classifyVcsResultPolarity(resultText)
      : EMPTY_RESULT_POLARITY;
    const executionFailure =
      commandRunning && hasStrongExecutionFailure(resultText);
    const incompleteState =
      tool.state.status !== "completed" && tool.state.status !== "error";
    const failed =
      tool.state.status === "error"
      || (verification && result.failed)
      || (vcs && vcsResult.failed)
      || executionFailure;
    const notRun = (verification && result.notRun) || (vcs && vcsResult.notRun);
    const incomplete =
      incompleteState
      || (verification && result.incomplete)
      || (vcs && vcsResult.incomplete);
    const unresolved = failed || notRun || incomplete;
    const mutation = hasToolMutationIdentity(tool.tool);
    const categories = [
      ...(verification ? ["verification"] : []),
      ...(vcs ? ["vcs"] : []),
      ...(failed ? ["failure"] : []),
      ...(notRun ? ["not-run"] : []),
      ...(incomplete ? ["incomplete"] : []),
    ];
    return {
      index,
      tool,
      command,
      input,
      output,
      error,
      title,
      categories,
      anchors: extractEvidenceAnchors(fullText),
      verification,
      vcs,
      unresolved,
      failed,
      notRun,
      incomplete,
      mutation,
    };
  });
}

function sourceValue(value: unknown): string | null {
  if (value === undefined) return null;
  const serialized = typeof value === "string" ? value : JSON.stringify(value);
  return serialized === undefined ? null : serialized;
}

function sourceCommand(value: unknown): string | null {
  if (!isRecord(value) || typeof value["command"] !== "string") return null;
  const command = value["command"].trim();
  return command || null;
}

function hasToolVerificationContext(
  toolName: string,
  command: string | null,
): boolean {
  if (command !== null && VERIFICATION_COMMAND_PATTERN.test(command)) {
    return true;
  }
  if (isPassiveToolIdentity(toolName)) return false;
  return VERIFICATION_TOOL_IDENTITY_PATTERN.test(
    normalizeToolIdentity(toolName),
  );
}

function hasToolVcsContext(toolName: string, command: string | null): boolean {
  if (command !== null && VCS_GIT_COMMAND_PATTERN.test(command)) return true;
  if (isPassiveToolIdentity(toolName)) return false;
  return VCS_TOOL_IDENTITY_PATTERN.test(normalizeToolIdentity(toolName));
}

function hasToolMutationIdentity(toolName: string): boolean {
  return TOOL_MUTATION_IDENTITY_PATTERN.test(toolName);
}

function hasCommandRunningIdentity(toolName: string): boolean {
  return COMMAND_RUNNING_TOOL_IDENTITY_PATTERN.test(
    normalizeToolIdentity(toolName),
  );
}

function hasStrongExecutionFailure(result: string): boolean {
  return STRONG_EXECUTION_FAILURE_PATTERNS.some(pattern =>
    pattern.test(result),
  );
}

function isPassiveToolIdentity(toolName: string): boolean {
  return PASSIVE_TOOL_IDENTITY_PATTERN.test(normalizeToolIdentity(toolName));
}

function normalizeToolIdentity(toolName: string): string {
  return toolName
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .split(/[^A-Za-z0-9]+/)
    .filter(Boolean)
    .join(" ");
}

function buildToolRecord(source: ToolSource): EvidenceRecord {
  const priority = source.failed
    ? 0
    : source.notRun || source.incomplete
      ? 1
      : source.verification || source.vcs
        ? 2
        : source.mutation
          ? 3
          : 5;
  return {
    priority,
    order: source.index,
    verification: source.verification,
    vcs: source.vcs,
    unresolved: source.unresolved,
    full: renderToolRecord(source, 4_096),
    compact: renderToolRecord(source, 1_536),
    minimum: renderToolRecord(source, 640),
  };
}

function evidencePolarity(
  failed: boolean,
  notRun: boolean,
  incomplete: boolean,
): "failed" | "not-run" | "incomplete" | "reported" {
  if (failed) return "failed";
  if (notRun) return "not-run";
  if (incomplete) return "incomplete";
  return "reported";
}

function renderToolRecord(source: ToolSource, maximumBytes: number): string {
  const kind =
    source.unresolved || source.verification || source.vcs
      ? "tool-terminal"
      : source.mutation
        ? "tool-mutation"
        : "tool-status";
  const base: Record<string, unknown> = {
    kind,
    index: source.index + 1,
    tool: sanitizeIdentifier(source.tool.tool),
    status: source.tool.state.status,
    categories: source.categories.length > 0 ? source.categories : ["status"],
    polarity: evidencePolarity(source.failed, source.notRun, source.incomplete),
  };
  if (source.command) base["command"] = boundedExact(source.command, 1_024);
  const paths = extractPathAnchors(
    [source.command, source.input, source.output, source.error, source.title]
      .filter(isString)
      .join("\n"),
  );
  if (paths.length > 0) base["paths"] = paths;
  if (source.anchors.length > 0) {
    base["anchors"] = source.anchors;
  }

  const state: Record<string, unknown> = {};
  const fieldBudget =
    maximumBytes <= 640 ? 128 : maximumBytes <= 1_536 ? 320 : 1_024;
  for (const [field, value] of [
    ["input", source.input],
    ["output", source.output],
    ["error", source.error],
  ] as const) {
    if (value !== null) state[field] = boundedSourceField(value, fieldBudget);
  }
  if (source.title) base["title"] = boundedExact(source.title, 256);
  if (Object.keys(state).length > 0) base["state"] = state;
  return serializeRecord(base);
}

function buildAssistantEvidenceRecords(
  sources: AssistantSource[],
): EvidenceRecord[] {
  const records: EvidenceRecord[] = [];
  let order = 0;
  for (const source of sources) {
    const result = applyPositiveRunOverride(
      classifyAssistantResultPolarity(source.text),
      source.text,
    );
    const vcsResult = classifyAssistantVcsResultPolarity(source.text);
    const verification = hasAssistantVerificationEvidence(source.text);
    const vcs = hasAssistantVcsEvidence(source.text);
    const failed = result.failed || (vcs && vcsResult.failed);
    const notRun = result.notRun || (vcs && vcsResult.notRun);
    const incomplete = result.incomplete || (vcs && vcsResult.incomplete);
    const unresolved = failed || notRun || incomplete;
    if (!verification && !vcs && !unresolved) continue;
    const categories = [
      ...(verification ? ["verification"] : []),
      ...(vcs ? ["vcs"] : []),
      ...(failed ? ["failure"] : []),
      ...(notRun ? ["not-run"] : []),
      ...(incomplete ? ["incomplete"] : []),
    ];
    const polarity = evidencePolarity(failed, notRun, incomplete);
    const render = (maximumBytes: number) =>
      renderAssistantEvidenceRecord(
        source,
        { offset: 0, text: source.text },
        categories,
        polarity,
        maximumBytes,
      );
    records.push({
      priority: failed ? 0 : notRun || incomplete ? 1 : 2,
      order: 1_000_000 + order++,
      verification,
      vcs,
      unresolved,
      full: render(2_048),
      compact: render(1_024),
      minimum: render(512),
    });
  }
  return records;
}

function renderAssistantEvidenceRecord(
  source: AssistantSource,
  segment: { offset: number; text: string },
  categories: string[],
  polarity: "failed" | "not-run" | "incomplete" | "reported",
  maximumBytes: number,
): string {
  const base: Record<string, unknown> = {
    kind: "assistant-evidence",
    messageID: source.messageID,
    partID: source.partID,
    offset: segment.offset,
    categories,
    polarity,
  };
  const paths = extractPathAnchors(segment.text);
  const anchors = extractEvidenceAnchors(segment.text);
  if (paths.length > 0) base["paths"] = paths;
  if (anchors.length > 0) base["anchors"] = anchors;
  const exact = serializeRecord({ ...base, text: segment.text });
  if (Buffer.byteLength(exact) <= maximumBytes) return exact;
  return fitWindowedRecord(base, segment.text, maximumBytes);
}

function createSection(
  label: string,
  recordLabel: SummarySection["recordLabel"],
  records: BudgetRecord[],
  required: BudgetRecord[],
  noneWhenEmpty: boolean,
): SummarySection {
  return {
    label,
    recordLabel,
    records: records.toSorted(
      (left, right) =>
        left.priority - right.priority || left.order - right.order,
    ),
    required: new Set(required),
    noneWhenEmpty,
    choices: new Map(),
  };
}

function reserveRequiredRecords(
  outcome: string,
  sections: SummarySection[],
): void {
  const required = sections.flatMap((section, sectionIndex) =>
    [...section.required].map(record => ({ section, sectionIndex, record })),
  );
  required.sort(
    (left, right) =>
      left.record.priority - right.record.priority
      || left.record.order - right.record.order
      || left.sectionIndex - right.sectionIndex,
  );
  for (const item of required) {
    item.section.choices.set(item.record, "minimum");
  }
  const requiredBytes = Buffer.byteLength(renderSummary(outcome, sections));
  if (requiredBytes > DETERMINISTIC_HIGH_RISK_SUMMARY_MAX_BYTES) {
    throw new HighRiskEvidenceOverflowError(requiredBytes);
  }
  for (const choice of ["compact", "full"] as const) {
    for (const item of required) {
      const previous = item.section.choices.get(item.record)!;
      item.section.choices.set(item.record, choice);
      if (!summaryFits(outcome, sections)) {
        item.section.choices.set(item.record, previous);
      }
    }
  }
}

function fillOptionalRecords(
  outcome: string,
  sections: SummarySection[],
): void {
  const optional = sections.flatMap((section, sectionIndex) =>
    section.records
      .filter(record => !section.required.has(record))
      .map(record => ({ section, sectionIndex, record })),
  );
  optional.sort(
    (left, right) =>
      left.record.priority - right.record.priority
      || left.record.order - right.record.order
      || left.sectionIndex - right.sectionIndex,
  );
  for (const item of optional) {
    item.section.choices.set(item.record, "compact");
    if (!summaryFits(outcome, sections)) {
      item.section.choices.delete(item.record);
      continue;
    }
    item.section.choices.set(item.record, "full");
    if (!summaryFits(outcome, sections)) {
      item.section.choices.set(item.record, "compact");
    }
  }
}

function summaryFits(outcome: string, sections: SummarySection[]): boolean {
  return (
    Buffer.byteLength(renderSummary(outcome, sections))
    <= DETERMINISTIC_HIGH_RISK_SUMMARY_MAX_BYTES
  );
}

function renderSummary(outcome: string, sections: SummarySection[]): string {
  return [`Outcome: ${outcome}`, ...sections.flatMap(renderSection)].join("\n");
}

function renderSection(section: SummarySection): string[] {
  if (section.records.length === 0 && section.noneWhenEmpty) {
    return [`${section.label}: None evidenced.`];
  }
  const included = section.records.flatMap(record => {
    const choice = section.choices.get(record);
    return choice ? [record[choice]] : [];
  });
  return [
    `${section.label}:`,
    section.recordLabel,
    ...included,
    `omitted-record-count=${section.records.length - included.length}`,
  ];
}

function boundedSourceField(value: string, maximumBytes: number): unknown {
  if (Buffer.byteLength(serializeRecord(value)) <= maximumBytes) return value;
  return {
    sourceBytes: Buffer.byteLength(value),
    windows: exactSourceWindows(value, Math.max(64, maximumBytes - 64)),
  };
}

function serializeRecord(value: unknown): string {
  const serialized = JSON.stringify(value);
  if (serialized === undefined) {
    throw new Error("High-risk record value is not JSON-serializable.");
  }
  return escapeXmlInvalidJsonLiterals(serialized);
}

function escapeXmlInvalidJsonLiterals(json: string): string {
  let result = "";
  for (const character of json) {
    const codePoint = character.codePointAt(0)!;
    if (isXmlCodePoint(codePoint)) {
      result += character;
      continue;
    }
    for (let index = 0; index < character.length; index++) {
      result += `\\u${character.charCodeAt(index).toString(16).padStart(4, "0")}`;
    }
  }
  return result;
}

function fitWindowedRecord(
  base: Record<string, unknown>,
  value: string,
  maximumBytes: number,
): string {
  const windows = exactSourceWindows(
    value,
    Math.max(64, maximumBytes - Buffer.byteLength(serializeRecord(base)) - 64),
  );
  let record = serializeRecord({ ...base, windows });
  while (Buffer.byteLength(record) > maximumBytes && windows.length > 1) {
    windows.splice(Math.max(1, windows.length - 2), 1);
    record = serializeRecord({ ...base, windows });
  }
  if (Buffer.byteLength(record) <= maximumBytes) return record;
  const budget = Math.max(
    64,
    maximumBytes - Buffer.byteLength(serializeRecord(base)) - 32,
  );
  return serializeRecord({
    ...base,
    windows: [truncateEnd(windows[0] ?? value, budget)],
  });
}

function exactSourceWindows(value: string, maximumBytes: number): string[] {
  const ranges: Array<{ start: number; end: number; priority: number }> = [];
  const addRange = (start: number, end: number, priority: number) => {
    ranges.push({
      start: Math.max(0, start - 192),
      end: Math.min(value.length, end + 384),
      priority,
    });
  };
  for (const match of value.matchAll(SOURCE_EVIDENCE_CUE_PATTERN)) {
    addRange(match.index, match.index + match[0].length, 0);
  }
  for (const path of extractPathMatches(value)) {
    addRange(path.index, path.end, 1);
  }
  addRange(0, Math.min(value.length, 256), 2);
  if (value.length > 256) {
    addRange(Math.max(0, value.length - 256), value.length, 2);
  }
  ranges.sort((left, right) => left.start - right.start);
  const merged: Array<{ start: number; end: number; priority: number }> = [];
  for (const range of ranges) {
    const previous = merged.at(-1);
    if (previous && range.start <= previous.end) {
      previous.end = Math.max(previous.end, range.end);
      previous.priority = Math.min(previous.priority, range.priority);
    } else {
      merged.push({ ...range });
    }
  }
  merged.sort(
    (left, right) => left.priority - right.priority || left.start - right.start,
  );
  const windows: string[] = [];
  let remaining = Math.max(0, Math.floor(maximumBytes));
  for (const range of merged.slice(0, MAX_SOURCE_WINDOWS)) {
    const window = safeSlice(value, range.start, range.end);
    if (remaining === 0) break;
    const bounded = truncateEnd(window, remaining);
    if (bounded) {
      windows.push(bounded);
      remaining -= Buffer.byteLength(bounded);
    }
  }
  return windows;
}

function extractEvidenceAnchors(value: string): string[] {
  const anchors = value.match(EVIDENCE_ANCHOR_PATTERN) ?? [];
  return [...new Set(anchors.map(anchor => anchor.trim()).filter(Boolean))];
}

function extractPathAnchors(value: string): string[] {
  return [...new Set(extractPathMatches(value).map(match => match.text))];
}

function extractPathMatches(
  value: string,
): Array<{ index: number; end: number; text: string }> {
  const pattern =
    /(["'])(?:(?:~\/|(?:\.\.?\/)+|\/|[A-Za-z0-9_.-]+\/)[^"'\\\r\n]+)\1|(?<![A-Za-z0-9_.~/-])(?:(?:~\/|(?:\.\.?\/)+|\/)(?:[A-Za-z0-9_.-]+\/)*(?:\.\.\.|[A-Za-z0-9_.-]*[A-Za-z0-9_-])|(?:[A-Za-z0-9_.-]+\/)+[A-Za-z0-9_.-]*[A-Za-z0-9_-])/g;
  return [...value.matchAll(pattern)].map(match => ({
    index: match.index,
    end: match.index + match[0].length,
    text: match[0],
  }));
}

function boundedExact(value: string, maximumBytes: number): string {
  if (Buffer.byteLength(value) <= maximumBytes) return value;
  const marker = "\n[...source window omitted...]\n";
  const available = maximumBytes - Buffer.byteLength(marker);
  if (available <= 0) return truncateEnd(marker, maximumBytes);
  return `${truncateEnd(value, Math.floor(available / 2))}${marker}${truncateStart(value, Math.ceil(available / 2))}`;
}

function safeSlice(value: string, start: number, end: number): string {
  if (start > 0 && isLowSurrogate(value.charCodeAt(start))) start -= 1;
  if (end < value.length && isHighSurrogate(value.charCodeAt(end - 1)))
    end += 1;
  return value.slice(start, end);
}

function isHighSurrogate(value: number): boolean {
  return value >= 0xd800 && value <= 0xdbff;
}

function isLowSurrogate(value: number): boolean {
  return value >= 0xdc00 && value <= 0xdfff;
}

function truncateEnd(value: string, maximumBytes: number): string {
  if (Buffer.byteLength(value) <= maximumBytes) return value;
  let bytes = 0;
  let end = 0;
  for (const character of value) {
    const size = Buffer.byteLength(character);
    if (bytes + size > maximumBytes) break;
    bytes += size;
    end += character.length;
  }
  return value.slice(0, end);
}

function truncateStart(value: string, maximumBytes: number): string {
  if (Buffer.byteLength(value) <= maximumBytes) return value;
  let bytes = 0;
  let start = value.length;
  while (start > 0) {
    let characterStart = start - 1;
    if (
      isLowSurrogate(value.charCodeAt(characterStart))
      && characterStart > 0
    ) {
      characterStart -= 1;
    }
    const size = Buffer.byteLength(value.slice(characterStart, start));
    if (bytes + size > maximumBytes) break;
    bytes += size;
    start = characterStart;
  }
  return value.slice(start);
}

function sanitizeIdentifier(value: string): string {
  return truncateEnd(stripInvalidXmlCodePoints(value), 256);
}

function isString(value: string | null): value is string {
  return value !== null;
}

const EMPTY_RESULT_POLARITY: ResultPolarity = {
  failed: false,
  notRun: false,
  incomplete: false,
};
const TOOL_MUTATION_IDENTITY_PATTERN =
  /^(?:write|edit|apply_patch|apply-patch|multiedit|multi_edit|str_replace|str-replace|create|delete|notebook_edit|notebook-edit|todowrite|todo_write)$/i;
const COMMAND_RUNNING_TOOL_IDENTITY_PATTERN =
  /\b(?:bash|shell|terminal|exec)\b/i;
const PASSIVE_TOOL_IDENTITY_PATTERN =
  /\b(?:read|get|list|search|view|inspect|fetch)\b/i;
const VERIFICATION_TOOL_IDENTITY_PATTERN =
  /\b(?:pytest|vitest|jest|mocha|typecheck|lint|tsc|tests?|checks?|build|smoke(?:\s+tests?)?|integration(?:\s+tests?)?)\b/i;
const VCS_TOOL_IDENTITY_PATTERN =
  /\b(?:git|gh|deploy(?:ment)?|vercel|railway|kubectl|helm)\b/i;
const ASSISTANT_NEGATED_UNRESOLVED_PATTERNS = [
  /\bno\s+unresolved\s+(?:issues?|work|items?|problems?|tasks?)\b/gi,
  /\bnothing\s+remains?(?:\s+to\s+do)?\b/gi,
  /\bno\s+pending\s+(?:work|issues?|items?|tasks?|changes?|tests?|checks?)\b/gi,
  /\bno\s+(?:work|issues?|items?|tasks?|changes?|tests?|checks?)\s+(?:are|is|remain(?:s)?)\s+pending\b/gi,
  /\bno\s+(?:work|issues?|items?|tasks?|changes?|tests?|checks?)\s+remain(?:s)?\s+pending\b/gi,
  /\bimplementation\s+is\s+not\s+incomplete\b/gi,
  /\b(?:configuration|config|state|value|files?)\s+remains?\s+unchanged\b/gi,
] as const;
const ASSISTANT_NEGATED_MUTATION_PATTERN =
  /\bno\s+(?:files?\s+)?(?:were\s+|was\s+)?(?:modified|changed|written|edited|patched|updated|created|deleted)\b/gi;
const ASSISTANT_VERIFICATION_PATTERN =
  /\b(?:tests?|checks?|commands?|verification|pytest|vitest|jest|mocha|lint|typecheck|build|smoke(?:\s+test)?|integration(?:\s+tests?)?)\b[^\n;]{0,120}\b(?:ran|run|executed|pass(?:ed|ing)?|fail(?:ed|ing|ures?)?|errored|succeeded|successful|not[- ]?run|not\s+executed|skipped|could\s+not|couldn't|cannot|can'?t)\b|\b(?:ran|run|executed|pass(?:ed|ing)?|fail(?:ed|ing|ures?)?|not[- ]?run|not\s+executed|skipped)\b[^\n;]{0,80}\b(?:tests?|checks?|commands?|lint|typecheck|build|smoke(?:\s+test)?|integration(?:\s+tests?)?)\b|\bno\s+tests?\s+have\s+run\b/i;
const ASSISTANT_FAILURE_PATTERN =
  /\b(?:tests?|checks?|commands?|verification|lint|typecheck|build|smoke(?:\s+test)?|integration(?:\s+tests?)?|deployment|deploy|push|commit|work|task|implementation)\b[^\n;]{0,120}\b(?:failed|failing|failures?|errored|fatal|timed\s*out|timeout|cancelled|canceled|aborted|killed)\b|\bfailed\s+to\s+(?:run|execute|build|test|check|write|edit|patch|create|delete|deploy|push|commit)\b/i;
const ASSISTANT_NOT_RUN_PATTERN =
  /\bno\s+(?:tests?|checks?|commands?)\s+(?:ran|run|executed)\b|\bno\s+tests?\s+have\s+run\b|\b(?:tests?|checks?)\s+were\s+skipped\b|\b(?:tests?|checks?|commands?|verification|lint|typecheck|build|smoke(?:\s+test)?|integration(?:\s+tests?)?)\b[^\n;]{0,120}\b(?:was|were)?\s*not\s+(?:run|executed|performed|attempted)\b|\b(?:tests?|checks?|commands?|verification|lint|typecheck|build|smoke(?:\s+test)?|integration(?:\s+tests?)?)\b[^\n;]{0,120}\b(?:could\s+not|couldn't|cannot|can'?t)\s+(?:be\s+)?(?:run|performed|executed)\b|\bnot\s+(?:run|executed|performed|attempted)\b[^\n;]{0,80}\b(?:tests?|checks?|commands?|lint|typecheck|build|smoke(?:\s+test)?|integration(?:\s+tests?)?)\b|\b(?:could\s+not|couldn't|cannot|can'?t)\s+(?:run|perform|execute)\b[^\n;]{0,80}\b(?:tests?|checks?|commands?)\b/i;
const ASSISTANT_INCOMPLETE_PATTERN =
  /\bwork\s+remains?\b|\b(?:TODO|TBD)\b|\b(?:work|tasks?|issues?|implementation|verification|tests?|checks?|commands?|deployment)\b[^\n;]{0,80}\b(?:incomplete|unresolved|blocked|pending|still\s+running)\b|\b(?:incomplete|unresolved|blocked|pending)\b[^\n;]{0,80}\b(?:work|tasks?|issues?|implementation|verification|tests?|checks?|commands?|deployment)\b|\b(?:must\s+still|still\s+needs?|needs?\s+to|next\s+action)\b/i;
const ASSISTANT_MUTATION_PATTERN =
  /\b(?:implemented|wrote|edited|patched|updated|created|deleted|renamed|moved|restored|modified|applied)\s+(?:(?:the|a|an)\s+)?(?:files?|code|source|implementation|configuration|config|module|component|function|class|tests?|docs?|documentation|patch|change|["'./~][^\s,;]+)\b|\b(?:files?|code|source|implementation|configuration|config|module|component|function|class|tests?|docs?|documentation)\s+(?:was|were)\s+(?:written|edited|patched|updated|created|deleted|renamed|moved|restored|modified)\b|\bdelivered\s+(?:(?:the|an?)\s+)?implementation\b/i;
const VERIFICATION_COMMAND_PATTERN =
  /\b(?:pytest|vitest|jest|mocha)\b|\btsc\b[^\n;&|]*--noEmit\b|\b(?:bun|npm|pnpm|yarn)(?:\s+run)?\s+(?:test|check|lint|typecheck|build|smoke)\b|\b(?:make|just|task)\s+(?:test|check|lint|typecheck|build|smoke)\b|\bcargo\s+test\b|\bgo\s+test\b/i;
const VCS_GIT_COMMAND_PATTERN =
  /\bgit\s+(?:add|branch|checkout|cherry-pick|clean|clone|commit|diff|fetch|log|merge|pull|push|rebase|remote|reset|restore|revert|show|stash|status|switch|tag)\b/i;
const VCS_CONTEXT_PATTERN =
  /\b(?:committed(?:\s+[0-9a-f]{6,40})?|commit\s+(?:hash\s+)?[0-9a-f]{6,40}|pushed|PR(?:\s*#?\d+)?|pull request(?:\s*#?\d+)?|deployed)\b|\bnot\s+(?:committed|pushed|deployed|merged)\b|\b(?:push|commit|deployment?|deploy)\b[^\n.;]{0,80}\bnot\s+(?:run|performed|attempted|completed)\b|\bpush\b[^\n.;]{0,40}\b(?:origin|upstream|remote)\b|\bdeployment\b(?!\s+configuration\b)[^\n.;]{0,80}\b(?:succeeded|successful|failed|blocked|pending|not[- ]?run|completed successfully)\b|\bworking tree\b[^\n.;]{0,80}\b(?:clean|dirty|ahead|behind)\b/i;
const NONZERO_EXIT_PATTERN =
  /\b(?:process\s+)?exit(?:ed)?(?:\s+with)?\s+(?:code|status)\s*[:=]?\s*(?!0\b)\d+\b/i;
const STRONG_EXECUTION_FAILURE_PATTERNS = [
  NONZERO_EXIT_PATTERN,
  /\bpermission denied\b/i,
  /\bcommand not found\b/i,
  /\b(?:timed\s*out|timeout|cancelled|canceled|aborted|killed|sigterm|sigkill|signal\s+\d+)\b/i,
] as const;
const POSITIVE_FAILURE_COUNT_PATTERN =
  /\b(?:[1-9]\d*\s+(?:(?:tests?|checks?)\s+)?(?:fails?|failed|failing|failures?|errors?)|(?:fails?|failed|failing|failures?|errors?)\s*[:=]\s*[1-9]\d*)\b/i;
const POSITIVE_RUN_PASS_PATTERN =
  /\b(?:\d+\s+(?:(?:tests?|checks?)\s+)?(?:pass|passed|passing)|(?:all\s+)?tests?\s+passed)\b/i;
const ZERO_FAILURE_COUNT_PATTERN =
  /\b(?:0+\s+(?:(?:tests?|checks?)\s+)?(?:fails?|failed|failing|failures?|errors?|skipped)|(?:fails?|failed|failing|failures?|errors?|skipped)\s*[:=]\s*0+|no\s+(?:(?:tests?|checks?)\s+)?(?:failed|failing|failures?|errors?))\b/gi;
const RESULT_FAILURE_PATTERNS = [
  NONZERO_EXIT_PATTERN,
  POSITIVE_FAILURE_COUNT_PATTERN,
  /\bpermission denied\b/i,
  /\bcommand not found\b/i,
  /\b(?:failed|failing|failures|errored|fatal)\b|\bfailure\b(?!-)/i,
  /\b(?:timed\s*out|timeout|cancelled|canceled|aborted|killed|sigterm|sigkill|signal\s+\d+)\b/i,
] as const;
const RESULT_NOT_RUN_PATTERNS = [
  /\b(?:skipped|not[- ]?run|unrun|not\s+executed)\b/i,
  /\bno\s+(?:tests?|checks?)\s+(?:ran|run|executed)\b/i,
  /\bno\s+tests?\s+have\s+run\b/i,
  /\b(?:tests?|checks?)\s+were\s+skipped\b/i,
  /\b(?:tests?|checks?|smoke|commands?)[^\n.;]{0,120}\b(?:was|were)?\s*not\s+(?:run|executed|performed)\b/i,
  /\b(?:tests?|checks?|smoke|commands?)[^\n.;]{0,120}\b(?:could\s+not|couldn't|cannot|can'?t)\s+be\s+(?:run|performed|executed)\b/i,
  /\b(?:could\s+not|couldn't|cannot|can'?t)\s+(?:run|perform|execute)\b[^\n.;]{0,80}\b(?:tests?|checks?|commands?)\b/i,
] as const;
const RESULT_INCOMPLETE_PATTERNS = [
  /\b(?:incomplete|unresolved|blocked|pending)\b/i,
  /\b(?:still\s+running|remains?\s+running)\b/i,
  /\b(?:remains?|must still|needs? to|next action)\b/i,
] as const;
const VCS_NOT_RUN_PATTERNS = [
  /\bnot\s+(?:committed|pushed|deployed|merged)\b/i,
  /\b(?:push|commit|deployment?|deploy)\b[^\n.;]{0,80}\b(?:was\s+)?not\s+(?:run|performed|attempted|completed)\b/i,
] as const;
const SOURCE_EVIDENCE_CUE_PATTERN =
  /\b(?:pytest|vitest|jest|mocha|tsc|bun\s+(?:run\s+)?(?:test|check|lint|typecheck|build|smoke)|npm\s+(?:run\s+)?(?:test|check|lint|typecheck|build|smoke)|pnpm\s+(?:run\s+)?(?:test|check|lint|typecheck|build|smoke)|yarn\s+(?:run\s+)?(?:test|check|lint|typecheck|build|smoke)|cargo\s+test|go\s+test|tests?|checks?|verification|git\s+[A-Za-z-]+|committed|pushed|push\s+(?:to|origin|upstream|remote)|PR|pull request|deployed|deployment|permission denied|command not found|failed|failing|failure|errors?|not\s+executed|not[- ]?run|unrun|skipped|incomplete|unresolved|blocked|pending|running|timed\s*out|timeout|cancelled|canceled|aborted|killed|SIGTERM|SIGKILL|process\s+exited\s+with\s+code\s+\d+)\b/gi;
const EVIDENCE_ANCHOR_PATTERN =
  /\b(?:permission denied|command not found|no\s+(?:tests?|checks?)\s+(?:ran|run|executed)|not[- ]?run|not\s+(?:pushed|committed|deployed|merged)|unrun|skipped|timed\s*out|timeout|cancelled|canceled|aborted|killed|SIGTERM|SIGKILL|failed|failing|failure|errored|incomplete|unresolved|blocked|pending|running|[1-9]\d*\s+(?:(?:tests?|checks?)\s+)?(?:fails?|failed|failing|failures?|errors?)|(?:process\s+)?exit(?:ed)?(?:\s+with)?\s+(?:code|status)\s*[:=]?\s*\d+|\d+\s+(?:(?:tests?|checks?)\s+)?(?:pass|passed|passing)|working tree\s+(?:clean|dirty)|commit\s+[0-9a-f]{6,40}|push succeeded)\b/gi;

function withoutAssistantResolvedNegations(value: string): string {
  return ASSISTANT_NEGATED_UNRESOLVED_PATTERNS.reduce(
    (result, pattern) => result.replace(pattern, ""),
    value,
  );
}

function hasAssistantVerificationEvidence(value: string): boolean {
  return (
    VERIFICATION_COMMAND_PATTERN.test(value)
    || ASSISTANT_VERIFICATION_PATTERN.test(value)
    || POSITIVE_FAILURE_COUNT_PATTERN.test(value)
    || /\b\d+\s+(?:(?:tests?|checks?)\s+)?(?:pass|passed|passing)\b/i.test(
      value,
    )
  );
}

function hasAssistantVcsEvidence(value: string): boolean {
  return hasVcsEvidence(value);
}

function hasAssistantMutationEvidence(value: string): boolean {
  return ASSISTANT_MUTATION_PATTERN.test(
    value.replace(ASSISTANT_NEGATED_MUTATION_PATTERN, ""),
  );
}

function hasVcsEvidence(value: string): boolean {
  return VCS_GIT_COMMAND_PATTERN.test(value) || VCS_CONTEXT_PATTERN.test(value);
}

function hasPositiveRunOrPass(result: string): boolean {
  return POSITIVE_RUN_PASS_PATTERN.test(result);
}

function applyPositiveRunOverride(
  polarity: ResultPolarity,
  fullText: string,
): ResultPolarity {
  if (polarity.notRun && hasPositiveRunOrPass(fullText)) {
    return { ...polarity, notRun: false };
  }
  return polarity;
}

function classifyResultPolarity(result: string): ResultPolarity {
  const withoutZeroCounts = result.replace(ZERO_FAILURE_COUNT_PATTERN, "");
  const notRun = RESULT_NOT_RUN_PATTERNS.some(pattern =>
    pattern.test(withoutZeroCounts),
  );
  return applyPositiveRunOverride(
    {
      failed: RESULT_FAILURE_PATTERNS.some(pattern =>
        pattern.test(withoutZeroCounts),
      ),
      notRun,
      incomplete: RESULT_INCOMPLETE_PATTERNS.some(pattern =>
        pattern.test(withoutZeroCounts),
      ),
    },
    withoutZeroCounts,
  );
}

function classifyAssistantResultPolarity(result: string): ResultPolarity {
  const withoutNegations = withoutAssistantResolvedNegations(result);
  const withoutZeroCounts = withoutNegations.replace(
    ZERO_FAILURE_COUNT_PATTERN,
    "",
  );
  return applyPositiveRunOverride(
    {
      failed:
        NONZERO_EXIT_PATTERN.test(withoutZeroCounts)
        || POSITIVE_FAILURE_COUNT_PATTERN.test(withoutZeroCounts)
        || /\b(?:permission denied|command not found|timed\s*out|timeout|cancelled|canceled|aborted|killed|sigterm|sigkill|signal\s+\d+)\b/i.test(
          withoutZeroCounts,
        )
        || ASSISTANT_FAILURE_PATTERN.test(withoutZeroCounts),
      notRun: ASSISTANT_NOT_RUN_PATTERN.test(withoutZeroCounts),
      incomplete: ASSISTANT_INCOMPLETE_PATTERN.test(withoutZeroCounts),
    },
    withoutZeroCounts,
  );
}

function classifyVcsResultPolarity(result: string): ResultPolarity {
  const base = classifyResultPolarity(result);
  return {
    ...base,
    notRun:
      base.notRun || VCS_NOT_RUN_PATTERNS.some(pattern => pattern.test(result)),
  };
}

function classifyAssistantVcsResultPolarity(result: string): ResultPolarity {
  const base = classifyAssistantResultPolarity(result);
  return {
    ...base,
    notRun:
      base.notRun || VCS_NOT_RUN_PATTERNS.some(pattern => pattern.test(result)),
  };
}

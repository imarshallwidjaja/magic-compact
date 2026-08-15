import type { TextPart, ToolPart } from "@opencode-ai/sdk/v2";
import { countGptTokensWithinLimit } from "../stats/tokenize";
import { isRecord } from "../util";
import { MAX_SUMMARY_INPUT_TOKENS } from "./constants";
import type { Turn } from "./plan";
import { isXmlCodePoint, stripInvalidXmlCodePoints } from "./template";

export const DETERMINISTIC_SUMMARY_MAX_TOKENS = 2_048;
export const DETERMINISTIC_OPTIONAL_RECORD_MAX_TOKENS = 512;

const MAX_SOURCE_WINDOWS = 12;

type ResultPolarity = {
  failed: boolean;
  notRun: boolean;
  incomplete: boolean;
};

export type CapsuleOutcome = "completed" | "partial" | "analysis-only";

type BudgetRecord = {
  priority: number;
  order: number;
  full: string;
  compact: string;
  minimum: string;
  requiredType: string;
  overflowMinimum: string;
};

type AssistantRecord = BudgetRecord & {
  required: boolean;
};

type EvidenceRecord = BudgetRecord & {
  verification: boolean;
  vcs: boolean;
  unresolved: boolean;
  reserved: boolean;
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

type AssistantClassification = ResultPolarity & {
  verification: boolean;
  vcs: boolean;
  unresolved: boolean;
};

type AssistantStatusSource = {
  messageID: string;
  order: number;
  status: "error" | "recovered";
  errorType: string;
  message: string | null;
  statusCode: number | null;
  providerID: string | null;
  recoveredByMessageID: string | null;
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

export function renderDeterministicSummary(turn: Turn): string {
  const capsule = collectReadableCapsule(turn);
  const text = renderReadableCapsule(capsule);
  if (
    countGptTokensWithinLimit(text, DETERMINISTIC_SUMMARY_MAX_TOKENS) === null
  ) {
    throw new Error(
      "Deterministic summary could not represent its required continuation evidence within the 2,048-token local safety cap.",
    );
  }
  return text;
}

export function renderStandardTurnEvidence(turn: Turn): string {
  return renderMachineEvidence(turn);
}

export function requiresDeterministicSummary(turn: Turn): boolean {
  return classifyTurnOutcome(turn) === "partial";
}

export function classifyTurnOutcome(turn: Turn): CapsuleOutcome {
  const assistantSources = collectAssistantSources(turn);
  return classifyOutcome(
    assistantSources,
    collectToolSources(turn),
    collectAssistantStatusSources(turn),
    classifyAssistantSources(assistantSources),
  );
}

type CapsuleSection = "work" | "verification" | "vcs" | "unresolved" | "next";

type CapsuleAggregate = {
  key: string;
};

type CapsuleLine = {
  section: CapsuleSection;
  priority: number;
  order: number;
  required: boolean;
  text: string;
  identity?: string;
  aggregate: CapsuleAggregate | null;
  finalCue?: boolean;
};

type ReadableCapsule = {
  outcome: CapsuleOutcome;
  finalText: string | null;
  lines: CapsuleLine[];
};

const SECTION_LABELS: Record<Exclude<CapsuleSection, "work">, string> = {
  verification: "Verification",
  vcs: "VCS",
  unresolved: "Unresolved",
  next: "Next",
};

function collectReadableCapsule(turn: Turn): ReadableCapsule {
  const assistantSources = collectAssistantSources(turn);
  const assistantClassifications = classifyAssistantSources(assistantSources);
  const toolSources = collectToolSources(turn);
  const statusSources = collectAssistantStatusSources(turn);
  const lines: CapsuleLine[] = [];
  let order = 0;
  const nextOrder = () => order++;

  const finalMessageIndex = assistantSources.at(-1)?.messageIndex ?? null;
  const finalSources =
    finalMessageIndex === null
      ? []
      : assistantSources.filter(
          source => source.messageIndex === finalMessageIndex,
        );
  const finalText =
    finalSources.length === 0
      ? null
      : finalSources.map(source => source.text).join("\n");

  for (const source of assistantSources.filter(
    source => source.messageIndex !== finalMessageIndex,
  )) {
    lines.push({
      section: "work",
      priority: 4,
      order: nextOrder(),
      required: false,
      text: readableExcerpt(source.text, 160),
      aggregate: null,
    });
  }

  for (const source of toolSources) {
    lines.push(...readableToolBullets(source, nextOrder));
  }

  for (const source of statusSources) {
    const unresolved = source.status === "error";
    const detail = [
      source.message,
      source.statusCode === null ? null : `status ${source.statusCode}`,
      source.providerID,
    ]
      .filter((value): value is string => value !== null)
      .join(", ");
    lines.push({
      section: unresolved ? "unresolved" : "work",
      priority: unresolved ? 0 : 3,
      order: nextOrder(),
      required: unresolved,
      text: unresolved
        ? `[error] ${source.errorType}: ${detail}`
        : `Recovered from ${source.errorType}: ${detail}`,
      aggregate: null,
    });
  }

  for (const source of assistantSources) {
    lines.push(
      ...readableAssistantEvidenceBullets(
        source,
        nextOrder,
        source.messageIndex === finalMessageIndex,
        assistantClassifications.get(source)!,
      ),
    );
  }

  const outcome = classifyOutcome(
    assistantSources,
    toolSources,
    statusSources,
    assistantClassifications,
  );

  return { outcome, finalText, lines };
}

function readableToolBullets(
  source: ToolSource,
  nextOrder: () => number,
): CapsuleLine[] {
  const identifier = toolIdentifier(source);
  if (source.unresolved) {
    const tag = statusTag(source) ?? polarityTag(source);
    const detail = source.error ?? source.output;
    const separator = source.tool.state.status === "error" ? " → " : ": ";
    const head =
      identifier === null || source.command !== null
        ? (identifier ?? source.tool.tool)
        : `${source.tool.tool}: ${identifier}`;
    const text = `[${tag}] ${head}${detail === null ? "" : `${separator}${readableExcerpt(detail, 180)}`}`;
    return [
      {
        section: "unresolved",
        priority: source.failed ? 0 : 1,
        order: nextOrder(),
        required: true,
        text,
        aggregate: {
          key: normalizeCapsuleFact(
            `[${tag}] ${head}${detail === null ? "" : `${separator}${detail}`}`,
          ),
        },
      },
    ];
  }
  if (source.verification || source.vcs) {
    const detail = source.output;
    return [
      {
        section: source.verification ? "verification" : "vcs",
        priority: 2,
        order: nextOrder(),
        required: false,
        text:
          detail === null
            ? (identifier ?? source.tool.tool)
            : `${identifier ?? source.tool.tool}: ${readableExcerpt(detail, 180)}`,
        aggregate: null,
      },
    ];
  }
  if (source.mutation) {
    return [
      {
        section: "work",
        priority: 3,
        order: nextOrder(),
        required: false,
        text:
          identifier === null
            ? source.tool.tool
            : `${source.tool.tool}: ${identifier}`,
        aggregate: null,
      },
    ];
  }
  return [];
}

function readableAssistantEvidenceBullets(
  source: AssistantSource,
  nextOrder: () => number,
  finalCue: boolean,
  sourceClassification: AssistantClassification,
): CapsuleLine[] {
  const lines: CapsuleLine[] = [];
  const statements = assistantStatements(source.text);
  for (const statement of statements) {
    const classification =
      statements.length === 1
        ? sourceClassification
        : classifyAssistantSource(statement.raw);
    const decision = ASSISTANT_DECISION_PATTERN.test(statement.raw);
    const line = assistantFactLine(
      statement,
      classification,
      decision,
      nextOrder,
      finalCue,
    );
    if (line !== null) lines.push(line);
  }
  const represented = lines.some(line =>
    sourceClassification.failed
      ? line.section === "unresolved" && line.priority === 0
      : sourceClassification.notRun
        ? line.section === "unresolved"
        : sourceClassification.incomplete
          ? line.section === "next"
          : sourceClassification.verification
            ? line.section === "verification"
            : sourceClassification.vcs
              ? line.section === "vcs"
              : true,
  );
  if (sourceClassification.unresolved && !represented) {
    const raw = collapseWhitespace(source.text);
    const line = assistantFactLine(
      { raw, excerpt: raw },
      sourceClassification,
      false,
      nextOrder,
      finalCue,
    );
    if (line !== null) lines.push(line);
  }
  const seen = new Set<string>();
  return lines.filter(line => {
    const key = `${line.section}|${line.identity ?? normalizeCapsuleFact(line.text)}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function assistantFactLine(
  statement: { raw: string; excerpt: string },
  classification: AssistantClassification,
  decision: boolean,
  nextOrder: () => number,
  finalCue: boolean,
): CapsuleLine | null {
  const collapsed = collapseWhitespace(statement.raw);
  if (!collapsed) return null;
  const identity = normalizeCapsuleFact(collapsed);
  const actionText =
    classification.incomplete || decision
      ? readableActionWindow(collapsed, 240)
      : statement.excerpt;
  if (classification.failed || classification.notRun) {
    return {
      section: "unresolved",
      priority: classification.failed ? 0 : 1,
      order: nextOrder(),
      required: true,
      text: `${classification.failed ? "[failed]" : "[not-run]"} ${actionText}`,
      identity,
      aggregate: null,
      finalCue,
    };
  }
  if (classification.incomplete) {
    return {
      section: "next",
      priority: 1,
      order: nextOrder(),
      required: true,
      text: actionText,
      identity,
      aggregate: null,
      finalCue,
    };
  }
  if (decision) {
    return {
      section: "work",
      priority: 2,
      order: nextOrder(),
      required: false,
      text: actionText,
      identity,
      aggregate: null,
      finalCue,
    };
  }
  if (classification.verification) {
    return {
      section: "verification",
      priority: 2,
      order: nextOrder(),
      required: false,
      text: actionText,
      identity,
      aggregate: null,
      finalCue,
    };
  }
  if (classification.vcs) {
    return {
      section: "vcs",
      priority: 2,
      order: nextOrder(),
      required: false,
      text: actionText,
      identity,
      aggregate: null,
      finalCue,
    };
  }
  return null;
}

function assistantStatements(
  value: string,
): Array<{ raw: string; excerpt: string }> {
  const statements: Array<{ raw: string; excerpt: string }> = [];
  for (const segment of value.split(/[;\r\n]+/)) {
    const collapsed = collapseWhitespace(segment);
    if (!collapsed) continue;
    if (Buffer.byteLength(collapsed) <= 240) {
      statements.push({ raw: collapsed, excerpt: collapsed });
      continue;
    }
    const cues = [...collapsed.matchAll(SOURCE_EVIDENCE_CUE_PATTERN)];
    if (cues.length === 0) {
      statements.push({ raw: collapsed, excerpt: collapsed });
      continue;
    }
    const seenExcerpts = new Set<string>();
    for (const cue of cues) {
      const excerpt = readableCueExcerpt(
        collapsed,
        cue.index,
        cue[0].length,
        240,
      );
      if (seenExcerpts.has(excerpt)) continue;
      seenExcerpts.add(excerpt);
      statements.push({
        raw: collapsed,
        excerpt,
      });
    }
  }
  return statements;
}

function readableActionWindow(value: string, maximumBytes: number): string {
  if (Buffer.byteLength(value) <= maximumBytes) return value;
  const separator = " … ";
  const budget = maximumBytes - Buffer.byteLength(separator);
  const head = truncateEnd(value, Math.floor(budget * 0.45));
  const tail = truncateStart(value, budget - Buffer.byteLength(head));
  return `${head}${separator}${tail}`;
}

function readableCueExcerpt(
  value: string,
  index: number,
  cueLength: number,
  maximumBytes: number,
): string {
  if (Buffer.byteLength(value) <= maximumBytes) return value;
  const separator = " … ";
  const separatorBytes = Buffer.byteLength(separator);
  const beforeBudget = Math.floor((maximumBytes - separatorBytes) * 0.3);
  const before = truncateStart(value.slice(0, index), beforeBudget);
  const afterBudget = maximumBytes - Buffer.byteLength(before) - separatorBytes;
  const after = truncateEnd(
    value.slice(index),
    Math.max(cueLength, afterBudget),
  );
  return `${before}${separator}${after}`;
}

function toolIdentifier(source: ToolSource): string | null {
  if (source.command !== null) return source.command;
  const input = source.tool.state.input;
  if (isRecord(input)) {
    for (const key of ["filePath", "url", "path"] as const) {
      const value = input[key];
      if (typeof value === "string" && value.trim()) return value;
    }
  }
  return null;
}

function statusTag(source: ToolSource): string | null {
  const status = source.tool.state.status;
  if (status === "error") return "error";
  if (status !== "completed") return status;
  return null;
}

function polarityTag(source: ToolSource): string {
  if (source.failed) return "failed";
  if (source.notRun) return "not-run";
  if (source.incomplete) return "incomplete";
  return "unresolved";
}

function readableWorkExcerpt(value: string, maximumBytes: number): string {
  const collapsed = collapseWhitespace(value);
  if (Buffer.byteLength(collapsed) <= maximumBytes) return collapsed;
  const separator = " … ";
  const available = maximumBytes - Buffer.byteLength(separator);
  if (available <= 0) return truncateEnd(collapsed, maximumBytes);
  const headBudget = Math.floor(available * 0.7);
  const tailBudget = available - headBudget;
  return `${truncateEnd(collapsed, headBudget)} … ${truncateStart(collapsed, tailBudget)}`;
}

function readableExcerpt(value: string, maximumBytes: number): string {
  const collapsed = collapseWhitespace(value);
  if (Buffer.byteLength(collapsed) <= maximumBytes) return collapsed;
  const head = truncateEnd(collapsed, Math.floor(maximumBytes * 0.3));
  const tail = truncateStart(collapsed, Math.floor(maximumBytes * 0.1));
  const anchors = [...collapsed.matchAll(EVIDENCE_ANCHOR_PATTERN)].slice(0, 4);
  const windows = anchors.map(match => {
    const start = Math.max(0, match.index - 48);
    const end = Math.min(collapsed.length, match.index + match[0].length + 24);
    return safeSlice(collapsed, start, end);
  });
  return truncateEnd(
    [head, ...windows, tail].filter(Boolean).join(" … "),
    maximumBytes,
  );
}

function collapseWhitespace(value: string): string {
  return value
    .replace(/\r/g, "")
    .replace(/\s*\n\s*/g, " ")
    .replace(/[ \t]+/g, " ")
    .trim();
}

function renderReadableCapsule(capsule: ReadableCapsule): string {
  const render = (
    included: ReadonlySet<CapsuleLine>,
    workExcerpt: string | null,
  ) =>
    escapeInvalidCodePoints(renderCapsuleLines(capsule, included, workExcerpt));
  let required = capsule.lines.filter(line => line.required);
  let workExcerpt =
    capsule.finalText === null ? null : collapseWhitespace(capsule.finalText);
  let text = render(new Set(required), workExcerpt);
  let textTokens = countGptTokensWithinLimit(
    text,
    DETERMINISTIC_SUMMARY_MAX_TOKENS,
  );
  if (workExcerpt !== null && textTokens === null) {
    workExcerpt = null;
    required = [
      ...new Set([
        ...required,
        ...capsule.lines.filter(line => line.finalCue === true),
      ]),
    ];
    text = render(new Set(required), workExcerpt);
    textTokens = countGptTokensWithinLimit(
      text,
      DETERMINISTIC_SUMMARY_MAX_TOKENS,
    );
  }
  if (textTokens === null) {
    required = aggregateUnresolvedLines(required);
    text = render(new Set(required), workExcerpt);
    textTokens = countGptTokensWithinLimit(
      text,
      DETERMINISTIC_SUMMARY_MAX_TOKENS,
    );
  }
  if (textTokens === null) {
    throw new Error(
      "Deterministic summary could not represent its required continuation evidence within the 2,048-token local safety cap.",
    );
  }
  const requiredTokens = textTokens;
  const included = new Set(required);
  if (workExcerpt === null && capsule.finalText !== null) {
    workExcerpt = fitOptionalWorkExcerpt(
      capsule,
      included,
      requiredTokens,
      render,
    );
  }
  for (const line of sortCapsuleLines(
    capsule.lines.filter(
      candidate =>
        !included.has(candidate)
        && !(
          workExcerpt !== null
          && candidate.finalCue === true
          && candidate.section === "work"
        ),
    ),
  )) {
    included.add(line);
    const candidateTokens = countGptTokensWithinLimit(
      render(included, workExcerpt),
      DETERMINISTIC_SUMMARY_MAX_TOKENS,
    );
    if (
      candidateTokens === null
      || candidateTokens - requiredTokens
        > DETERMINISTIC_OPTIONAL_RECORD_MAX_TOKENS
    ) {
      included.delete(line);
    }
  }
  return render(included, workExcerpt);
}

function fitOptionalWorkExcerpt(
  capsule: ReadableCapsule,
  included: ReadonlySet<CapsuleLine>,
  requiredTokens: number,
  render: (
    included: ReadonlySet<CapsuleLine>,
    workExcerpt: string | null,
  ) => string,
): string | null {
  const source = capsule.finalText;
  if (source === null) return null;
  const sourceBytes = Buffer.byteLength(collapseWhitespace(source));
  const candidateAt = (maximumBytes: number) => {
    const candidate = readableWorkExcerpt(source, maximumBytes);
    const tokens = countGptTokensWithinLimit(
      render(included, candidate),
      DETERMINISTIC_SUMMARY_MAX_TOKENS,
    );
    return tokens !== null
      && tokens - requiredTokens <= DETERMINISTIC_OPTIONAL_RECORD_MAX_TOKENS
      ? candidate
      : null;
  };
  let best: string | null = null;
  let low = 1;
  let high = 1;
  while (high < sourceBytes) {
    const candidate = candidateAt(high);
    if (candidate === null) break;
    best = candidate;
    low = high + 1;
    high = Math.min(sourceBytes, high * 2);
  }
  const upperCandidate = candidateAt(high);
  if (upperCandidate !== null) return upperCandidate;
  high -= 1;
  while (low <= high) {
    const middle = Math.floor((low + high) / 2);
    const candidate = candidateAt(middle);
    if (candidate !== null) {
      best = candidate;
      low = middle + 1;
    } else {
      high = middle - 1;
    }
  }
  return best;
}

function renderCapsuleLines(
  capsule: ReadableCapsule,
  included: ReadonlySet<CapsuleLine>,
  workExcerpt: string | null,
): string {
  const lines: string[] = [`Outcome: ${capsule.outcome}`];
  const workLines = sortCapsuleLines(
    [...included].filter(line => line.section === "work"),
  );
  if (workExcerpt !== null || workLines.length > 0) {
    if (workExcerpt !== null) {
      lines.push(`Work: ${workExcerpt}`);
    }
    for (const line of workLines) lines.push(`- ${line.text}`);
  }
  for (const section of [
    "verification",
    "vcs",
    "unresolved",
    "next",
  ] as const) {
    const sectionLines = sortCapsuleLines(
      [...included].filter(line => line.section === section),
    );
    if (sectionLines.length === 0) continue;
    lines.push(`${SECTION_LABELS[section]}:`);
    for (const line of sectionLines) lines.push(`- ${line.text}`);
  }
  return lines.join("\n");
}

function sortCapsuleLines(lines: CapsuleLine[]): CapsuleLine[] {
  return lines.toSorted(
    (left, right) => left.priority - right.priority || left.order - right.order,
  );
}

function aggregateUnresolvedLines(lines: CapsuleLine[]): CapsuleLine[] {
  const result: CapsuleLine[] = [];
  const groupCounts = new Map<string, number>();
  const groupPositions = new Map<string, number>();
  for (const line of lines) {
    if (line.section !== "unresolved" || line.aggregate === null) {
      result.push(line);
      continue;
    }
    const key = line.aggregate.key;
    const count = groupCounts.get(key) ?? 0;
    if (count === 0) {
      groupPositions.set(key, result.length);
      result.push(line);
    }
    groupCounts.set(key, count + 1);
  }
  for (const [key, count] of groupCounts) {
    if (count === 1) continue;
    const position = groupPositions.get(key)!;
    const first = result[position]!;
    result[position] = {
      ...first,
      text: `${first.text} ×${count}`,
      aggregate: null,
    };
  }
  return result;
}

function normalizeCapsuleFact(value: string): string {
  return collapseWhitespace(value).toLocaleLowerCase("en-US");
}

function renderMachineEvidence(turn: Turn): string {
  const assistantSources = collectAssistantSources(turn);
  const assistantClassifications = classifyAssistantSources(assistantSources);
  const assistantRecords = buildAssistantRecords(assistantSources);
  const toolSources = collectToolSources(turn);
  const toolRecords = toolSources.map(buildToolRecord);
  const assistantEvidence = buildAssistantEvidenceRecords(
    assistantSources,
    assistantClassifications,
  );
  const assistantStatuses = buildAssistantStatusRecords(
    collectAssistantStatusSources(turn),
  );
  const evidenceRecords = [
    ...toolRecords,
    ...assistantEvidence,
    ...assistantStatuses,
  ];
  const outcome = classifyOutcome(
    assistantSources,
    toolSources,
    collectAssistantStatusSources(turn),
    assistantClassifications,
  );

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
      [],
      true,
    ),
    createSection(
      "VCS/Deployment",
      "evidence-records",
      evidenceRecords.filter(record => record.vcs),
      [],
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
      evidenceRecords.filter(record => !record.verification && !record.vcs),
      evidenceRecords.filter(
        record => record.reserved && !record.verification && !record.vcs,
      ),
      false,
    ),
  ];

  const overflow = reserveRequiredRecords(outcome, sections);
  if (!overflow) fillOptionalRecords(outcome, sections);
  const rendered = overflow ?? renderSummary(outcome, sections);
  if (countGptTokensWithinLimit(rendered, MAX_SUMMARY_INPUT_TOKENS) === null) {
    throw new Error(
      `One bounded turn evidence map exceeds the ${MAX_SUMMARY_INPUT_TOKENS}-token summary input cap.`,
    );
  }
  return rendered;
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

function collectAssistantStatusSources(turn: Turn): AssistantStatusSource[] {
  const assistants = turn.assistants;
  const sources: AssistantStatusSource[] = [];
  let laterSuccessfulMessageID: string | null = null;
  for (
    let messageIndex = assistants.length - 1;
    messageIndex >= 0;
    messageIndex -= 1
  ) {
    const message = assistants[messageIndex]!;
    if (message.info.role === "assistant" && message.info.error) {
      const data: Record<string, unknown> = isRecord(message.info.error.data)
        ? message.info.error.data
        : {};
      sources.push({
        messageID: sanitizeIdentifier(message.info.id),
        order: messageIndex,
        status: laterSuccessfulMessageID === null ? "error" : "recovered",
        errorType: sanitizeIdentifier(message.info.error.name),
        message: typeof data["message"] === "string" ? data["message"] : null,
        statusCode:
          typeof data["statusCode"] === "number" ? data["statusCode"] : null,
        providerID:
          typeof data["providerID"] === "string" ? data["providerID"] : null,
        recoveredByMessageID: laterSuccessfulMessageID,
      });
    }
    if (isSuccessfulAssistantMessage(message)) {
      laterSuccessfulMessageID = sanitizeIdentifier(message.info.id);
    }
  }
  return sources.reverse();
}

function classifyOutcome(
  assistantSources: AssistantSource[],
  toolSources: ToolSource[],
  statusSources: AssistantStatusSource[],
  assistantClassifications = classifyAssistantSources(assistantSources),
): CapsuleOutcome {
  const assistant = [...assistantClassifications.values()];
  if (
    toolSources.some(source => source.unresolved)
    || statusSources.some(source => source.status === "error")
    || assistant.some(source => source.unresolved)
  ) {
    return "partial";
  }
  const hasExecutionOrMutation =
    toolSources.length > 0
    || statusSources.length > 0
    || assistantSources.some(source =>
      hasAssistantMutationEvidence(source.text),
    );
  const hasVerificationOrVcs =
    toolSources.some(source => source.verification || source.vcs)
    || assistant.some(source => source.verification || source.vcs);
  return hasExecutionOrMutation || hasVerificationOrVcs
    ? "completed"
    : "analysis-only";
}

function classifyAssistantSources(
  sources: AssistantSource[],
): Map<AssistantSource, AssistantClassification> {
  return new Map(
    sources.map(source => [source, classifyAssistantSource(source.text)]),
  );
}

function isSuccessfulAssistantMessage(
  message: Turn["assistants"][number],
): boolean {
  if (message.info.role !== "assistant" || message.info.error) return false;
  if (typeof message.info.finish === "string" && message.info.finish.trim()) {
    return true;
  }
  return message.parts.some(
    part =>
      (part.type === "text" && isSubstantiveAssistantText(part))
      || (part.type === "tool" && part.state.status === "completed"),
  );
}

function isSubstantiveAssistantText(part: TextPart): boolean {
  if (part.synthetic) return false;
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
    const prepared = prepareAssistantRecord(source);
    return {
      priority: required ? 0 : 4,
      order: source.order,
      required,
      full: renderPreparedAssistantRecord(prepared, Number.POSITIVE_INFINITY),
      compact: renderPreparedAssistantRecord(prepared, 2_048),
      minimum: renderPreparedAssistantRecord(prepared, 768),
      requiredType: "assistant-text",
      overflowMinimum: serializeRecord({
        kind: "required-assistant-text",
        messageID: source.messageID,
        partID: source.partID,
        status: "authoritative-final",
      }),
    };
  });
}

const MIN_LITERAL_RUN_UNIT = 6;
const MAX_LITERAL_RUN_UNIT = 256;
const MIN_LITERAL_RUN_REPETITIONS = 8;
const MIN_LITERAL_RUN_TOTAL =
  MIN_LITERAL_RUN_UNIT * MIN_LITERAL_RUN_REPETITIONS;
const MAX_LITERAL_UNIT_MULTIPLE_ATTEMPTS = 64;

type LiteralRun = {
  literal: string;
  repetitions: number;
};

export function encodeOrderedLiteralRuns(text: string): LiteralRun[] | null {
  const length = text.length;
  if (length < MIN_LITERAL_RUN_TOTAL) return null;
  const suffixBorders = computeSuffixBorders(text);
  for (
    let suffixLength = length;
    suffixLength >= MIN_LITERAL_RUN_TOTAL;
    suffixLength -= 1
  ) {
    const minimalPeriod = suffixLength - suffixBorders[suffixLength - 1]!;
    const unitLength = encodableUnitLength(
      text,
      length,
      suffixLength,
      minimalPeriod,
    );
    if (unitLength === null) continue;
    const segments: LiteralRun[] = [];
    const prefixLength = length - suffixLength;
    if (prefixLength > 0) {
      segments.push({ literal: text.slice(0, prefixLength), repetitions: 1 });
    }
    segments.push({
      literal: text.slice(length - unitLength),
      repetitions: suffixLength / unitLength,
    });
    return segments;
  }
  return null;
}

function computeSuffixBorders(text: string): number[] {
  const length = text.length;
  const borders = new Array<number>(length).fill(0);
  const codeUnitAt = (index: number) => text.charCodeAt(length - 1 - index);
  for (let index = 1; index < length; index += 1) {
    let candidate = borders[index - 1]!;
    while (candidate > 0 && codeUnitAt(index) !== codeUnitAt(candidate)) {
      candidate = borders[candidate - 1]!;
    }
    if (codeUnitAt(index) === codeUnitAt(candidate)) candidate += 1;
    borders[index] = candidate;
  }
  return borders;
}

function encodableUnitLength(
  text: string,
  length: number,
  suffixLength: number,
  minimalPeriod: number,
): number | null {
  if (suffixLength % minimalPeriod !== 0) return null;
  if (suffixLength / minimalPeriod < MIN_LITERAL_RUN_REPETITIONS) return null;
  if (minimalPeriod > MAX_LITERAL_RUN_UNIT) return null;
  if (minimalPeriod >= MIN_LITERAL_RUN_UNIT) {
    return isEncodableLiteralUnitAt(text, length - minimalPeriod, minimalPeriod)
      ? minimalPeriod
      : null;
  }
  if (!isEncodableLiteralUnitAt(text, length - minimalPeriod, minimalPeriod)) {
    return null;
  }
  let unitLength = minimalPeriod;
  while (unitLength < MIN_LITERAL_RUN_UNIT) unitLength += minimalPeriod;
  const upperBound = Math.min(
    MAX_LITERAL_RUN_UNIT,
    Math.floor(suffixLength / MIN_LITERAL_RUN_REPETITIONS),
  );
  for (
    let attempts = 0;
    unitLength <= upperBound && attempts < MAX_LITERAL_UNIT_MULTIPLE_ATTEMPTS;
    unitLength += minimalPeriod, attempts += 1
  ) {
    if (suffixLength % unitLength === 0) return unitLength;
  }
  return null;
}

function isEncodableLiteralUnitAt(
  text: string,
  start: number,
  length: number,
): boolean {
  let distinct = false;
  let whitespace = false;
  const first = text.charCodeAt(start);
  for (let index = start; index < start + length; index += 1) {
    const code = text.charCodeAt(index);
    if (code !== first) distinct = true;
    if (code === 0x09 || code === 0x0a || code === 0x0d || code === 0x20) {
      whitespace = true;
    }
    if (distinct && whitespace) return true;
  }
  return false;
}

type PreparedAssistantRecord = {
  base: Record<string, unknown>;
  text: string;
  serializedExact: string;
  ranges: Array<{ start: number; end: number; priority: number }>;
};

function prepareAssistantRecord(
  source: AssistantSource,
): PreparedAssistantRecord {
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
  const runs = encodeOrderedLiteralRuns(source.text);
  const exact =
    runs === null
      ? {
          ...base,
          text: source.text,
        }
      : {
          kind: base["kind"],
          messageID: base["messageID"],
          partID: base["partID"],
          ...(base["paths"] === undefined ? {} : { paths: base["paths"] }),
          ...(base["anchors"] === undefined
            ? {}
            : { anchors: base["anchors"] }),
          textEncoding: "ordered-literal-runs-v1",
          textSegments: runs,
        };
  return {
    base,
    text: source.text,
    serializedExact: serializeRecord(exact),
    ranges: sourceWindowRanges(source.text),
  };
}

function renderPreparedAssistantRecord(
  prepared: PreparedAssistantRecord,
  maximumBytes: number,
): string {
  if (Buffer.byteLength(prepared.serializedExact) <= maximumBytes) {
    return prepared.serializedExact;
  }
  return fitWindowedRecord(
    prepared.base,
    prepared.text,
    prepared.ranges,
    maximumBytes,
  );
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
    reserved: false,
    full: renderToolRecord(source, 4_096),
    compact: renderToolRecord(source, 1_536),
    minimum: renderToolRecord(source, 640),
    requiredType:
      source.tool.state.status === "error"
        ? "tool-error"
        : source.failed
          ? "tool-failure"
          : source.notRun
            ? "tool-not-run"
            : source.incomplete
              ? "tool-incomplete"
              : "tool-unresolved",
    overflowMinimum: serializeRecord({
      kind: "required-tool",
      messageID: sanitizeIdentifier(source.tool.messageID),
      partID: sanitizeIdentifier(source.tool.id),
      callID: sanitizeIdentifier(source.tool.callID),
      tool: sanitizeIdentifier(source.tool.tool),
      status: source.tool.state.status,
      polarity: evidencePolarity(
        source.failed,
        source.notRun,
        source.incomplete,
      ),
    }),
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
  classifications = classifyAssistantSources(sources),
): EvidenceRecord[] {
  const records: EvidenceRecord[] = [];
  let order = 0;
  for (const source of sources) {
    const { verification, vcs, failed, notRun, incomplete, unresolved } =
      classifications.get(source)!;
    if (!verification && !vcs && !unresolved) continue;
    const categories = [
      ...(verification ? ["verification"] : []),
      ...(vcs ? ["vcs"] : []),
      ...(failed ? ["failure"] : []),
      ...(notRun ? ["not-run"] : []),
      ...(incomplete ? ["incomplete"] : []),
    ];
    const polarity = evidencePolarity(failed, notRun, incomplete);
    const prepared = prepareAssistantEvidenceRecord(
      source,
      { offset: 0, text: source.text },
      categories,
      polarity,
    );
    records.push({
      priority: failed ? 0 : notRun || incomplete ? 1 : 2,
      order: 1_000_000 + order++,
      verification,
      vcs,
      unresolved,
      reserved: false,
      full: renderPreparedAssistantEvidenceRecord(prepared, 2_048),
      compact: renderPreparedAssistantEvidenceRecord(prepared, 1_024),
      minimum: renderPreparedAssistantEvidenceRecord(prepared, 512),
      requiredType: `assistant-${polarity}`,
      overflowMinimum: serializeRecord({
        kind: "required-assistant-evidence",
        messageID: source.messageID,
        partID: source.partID,
        status: polarity,
      }),
    });
  }
  return records;
}

function buildAssistantStatusRecords(
  sources: AssistantStatusSource[],
): EvidenceRecord[] {
  return sources.map(source => {
    const unresolved = source.status === "error";
    const render = (maximumBytes: number) =>
      renderAssistantStatusRecord(source, maximumBytes);
    return {
      priority: unresolved ? 0 : 2,
      order: 2_000_000 + source.order,
      verification: false,
      vcs: false,
      unresolved,
      reserved: !unresolved,
      full: render(2_048),
      compact: render(1_024),
      minimum: render(512),
      requiredType: unresolved
        ? "assistant-message-error"
        : "assistant-message-recovered",
      overflowMinimum: serializeRecord({
        kind: "required-assistant-message-status",
        messageID: source.messageID,
        status: source.status,
        errorType: source.errorType,
        ...(source.statusCode === null
          ? {}
          : { statusCode: source.statusCode }),
      }),
    };
  });
}

function renderAssistantStatusRecord(
  source: AssistantStatusSource,
  maximumBytes: number,
): string {
  const base: Record<string, unknown> = {
    kind: "assistant-message-status",
    messageID: source.messageID,
    status: source.status,
    errorType: source.errorType,
    categories: [source.status === "error" ? "failure" : "recovered"],
    polarity: source.status === "error" ? "failed" : "reported",
  };
  const fieldBudget = maximumBytes <= 512 ? 128 : 512;
  if (source.message !== null) {
    base["message"] = boundedSourceField(source.message, fieldBudget);
  }
  if (source.statusCode !== null) base["statusCode"] = source.statusCode;
  if (source.providerID !== null) {
    base["providerID"] = boundedExact(source.providerID, 256);
  }
  if (source.recoveredByMessageID !== null) {
    base["recoveredByMessageID"] = source.recoveredByMessageID;
  }
  return serializeRecord(base);
}

type PreparedAssistantEvidenceRecord = {
  base: Record<string, unknown>;
  text: string;
  serializedExact: string;
  ranges: Array<{ start: number; end: number; priority: number }>;
};

function prepareAssistantEvidenceRecord(
  source: AssistantSource,
  segment: { offset: number; text: string },
  categories: string[],
  polarity: "failed" | "not-run" | "incomplete" | "reported",
): PreparedAssistantEvidenceRecord {
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
  return {
    base,
    text: segment.text,
    serializedExact: serializeRecord({ ...base, text: segment.text }),
    ranges: sourceWindowRanges(segment.text),
  };
}

function renderPreparedAssistantEvidenceRecord(
  prepared: PreparedAssistantEvidenceRecord,
  maximumBytes: number,
): string {
  if (Buffer.byteLength(prepared.serializedExact) <= maximumBytes) {
    return prepared.serializedExact;
  }
  return fitWindowedRecord(
    prepared.base,
    prepared.text,
    prepared.ranges,
    maximumBytes,
  );
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
): string | null {
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
    item.section.choices.set(
      item.record,
      item.record.requiredType === "assistant-text" ? "full" : "minimum",
    );
  }
  const requiredSummary = renderSummary(outcome, sections);
  if (
    countGptTokensWithinLimit(requiredSummary, DETERMINISTIC_SUMMARY_MAX_TOKENS)
    !== null
  ) {
    return null;
  }
  if (required.some(item => item.record.requiredType === "assistant-text")) {
    return null;
  }
  return renderRequiredOverflowCapsule(
    outcome,
    [...new Set(required.map(item => item.record))].toSorted(
      (left, right) =>
        left.priority - right.priority || left.order - right.order,
    ),
  );
}

function renderRequiredOverflowCapsule(
  outcome: string,
  records: BudgetRecord[],
): string {
  const counts = records.reduce<Record<string, number>>((result, record) => {
    result[record.requiredType] = (result[record.requiredType] ?? 0) + 1;
    return result;
  }, {});
  const render = (included: BudgetRecord[]) =>
    [
      `Outcome: ${outcome}`,
      "Required deterministic evidence overflow capsule:",
      `required-record-counts=${serializeRecord(counts)}`,
      `required-record-total=${records.length}`,
      "required-records:",
      ...included.map(record =>
        record.requiredType === "assistant-text"
          ? record.minimum
          : record.overflowMinimum,
      ),
      `required-record-overflow=${records.length - included.length}`,
      "required-record-omitted-disclosure=Per-record details were omitted from this capsule; aggregate required facts remain above. Full error tool I/O remains in active tool parts, and omitted completed tool I/O is retrievable.",
    ].join("\n");
  const substantive = records.filter(
    record => record.requiredType === "assistant-text",
  );
  const identifiers = records.filter(
    record => record.requiredType !== "assistant-text",
  );
  const included: BudgetRecord[] = [];
  for (const record of [...substantive, ...identifiers]) {
    const candidate = [...included, record];
    if (
      countGptTokensWithinLimit(
        render(candidate),
        DETERMINISTIC_SUMMARY_MAX_TOKENS,
      ) !== null
    ) {
      included.push(record);
    }
  }
  return render(included);
}

function fillOptionalRecords(
  outcome: string,
  sections: SummarySection[],
): void {
  const requiredTokens =
    countGptTokensWithinLimit(
      renderSummary(outcome, sections),
      DETERMINISTIC_SUMMARY_MAX_TOKENS,
    ) ?? DETERMINISTIC_SUMMARY_MAX_TOKENS + 1;
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
    item.section.choices.set(item.record, "minimum");
    if (!optionalChoiceFits(outcome, sections, requiredTokens)) {
      item.section.choices.delete(item.record);
      continue;
    }
    item.section.choices.set(item.record, "compact");
    if (!optionalChoiceFits(outcome, sections, requiredTokens)) {
      item.section.choices.set(item.record, "minimum");
      continue;
    }
    item.section.choices.set(item.record, "full");
    if (!optionalChoiceFits(outcome, sections, requiredTokens)) {
      item.section.choices.set(item.record, "compact");
    }
  }
}

function optionalChoiceFits(
  outcome: string,
  sections: SummarySection[],
  requiredTokens: number,
): boolean {
  const tokens = countGptTokensWithinLimit(
    renderSummary(outcome, sections),
    DETERMINISTIC_SUMMARY_MAX_TOKENS,
  );
  return (
    tokens !== null
    && tokens - requiredTokens <= DETERMINISTIC_OPTIONAL_RECORD_MAX_TOKENS
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
  return escapeInvalidCodePoints(serialized);
}

function escapeInvalidCodePoints(value: string): string {
  let result = "";
  for (const character of value) {
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
  ranges: Array<{ start: number; end: number; priority: number }>,
  maximumBytes: number,
): string {
  const windows = boundedSourceWindows(
    value,
    ranges,
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
  return boundedSourceWindows(value, sourceWindowRanges(value), maximumBytes);
}

function sourceWindowRanges(
  value: string,
): Array<{ start: number; end: number; priority: number }> {
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
  return merged;
}

function boundedSourceWindows(
  value: string,
  ranges: Array<{ start: number; end: number; priority: number }>,
  maximumBytes: number,
): string[] {
  const windows: string[] = [];
  let remaining = Math.max(0, Math.floor(maximumBytes));
  for (const range of ranges.slice(0, MAX_SOURCE_WINDOWS)) {
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
  /\bno\s+(?:tests?|checks?)\s+(?:were|was|are|is|have\s+been|has\s+been)?\s*skipped\b/gi,
  /\b(?:tests?|checks?)\s+(?:are|is|were|was)\s+not\s+(?:failing|failed)\b/gi,
  /\b(?:configuration|config|state|value|files?)\s+remains?\s+unchanged\b/gi,
] as const;
const ASSISTANT_NEGATED_MUTATION_PATTERN =
  /\bno\s+(?:files?\s+)?(?:were\s+|was\s+)?(?:modified|changed|written|edited|patched|updated|created|deleted)\b/gi;
const ASSISTANT_VERIFICATION_PATTERN =
  /\b(?:tests?|checks?|commands?|verification|pytest|vitest|jest|mocha|lint|typecheck|build|smoke(?:\s+test)?|integration(?:\s+tests?)?)\b[^\n;]{0,120}\b(?:ran|run|executed|pass(?:ed|ing)?|fail(?:ed|ing|ures?)?|errored|succeeded|successful|not[- ]?run|not\s+executed|skipped|could\s+not|couldn't|cannot|can'?t)\b|\b(?:ran|run|executed|pass(?:ed|ing)?|fail(?:ed|ing|ures?)?|not[- ]?run|not\s+executed|skipped)\b[^\n;]{0,80}\b(?:tests?|checks?|commands?|lint|typecheck|build|smoke(?:\s+test)?|integration(?:\s+tests?)?)\b|\bno\s+tests?\s+(?:have\s+run|have\s+been\s+executed)\b|\btests?\s+have\s+not\s+yet\s+run\b/i;
const ASSISTANT_FAILURE_PATTERN =
  /\b(?:tests?|checks?|commands?|verification|lint|typecheck|build|smoke(?:\s+test)?|integration(?:\s+tests?)?|deployment|deploy|push|commit|work|task|implementation)\b[^\n;]{0,120}\b(?:failed|failing|failures?|errored|fatal|timed\s*out|timeout|cancelled|canceled|aborted|killed)\b|\bfailed\s+to\s+(?:run|execute|build|test|check|write|edit|patch|create|delete|deploy|push|commit)\b/i;
const ASSISTANT_NOT_RUN_PATTERN =
  /\bno\s+(?:tests?|checks?|commands?)\s+(?:(?:was|were)\s+)?(?:ran|run|executed|performed)\b|\bno\s+tests?\s+(?:have\s+run|have\s+been\s+executed)\b|\btests?\s+have\s+not\s+yet\s+run\b|\b(?:tests?|checks?)\s+were\s+skipped\b|\b(?:tests?|checks?|commands?|verification|lint|typecheck|build|smoke(?:\s+test)?|integration(?:\s+tests?)?)\b[^\n;]{0,120}\b(?:was|were)?\s*not\s+(?:run|executed|performed|attempted)\b|\b(?:tests?|checks?|commands?|verification|lint|typecheck|build|smoke(?:\s+test)?|integration(?:\s+tests?)?)\b[^\n;]{0,120}\bnever\s+(?:ran|run|executed|performed|attempted)\b|\b(?:tests?|checks?|commands?|verification|lint|typecheck|build|smoke(?:\s+test)?|integration(?:\s+tests?)?)\b[^\n;]{0,120}\b(?:has|have|had)\s+yet\s+to\s+(?:run|perform|execute|be\s+executed|be\s+performed)\b|\b(?:tests?|checks?|commands?|verification|lint|typecheck|build|smoke(?:\s+test)?|integration(?:\s+tests?)?)\b[^\n;]{0,120}\b(?:could\s+not|couldn't|cannot|can'?t|unable\s+to)\s+(?:be\s+)?(?:run|performed|executed)\b|\bnot\s+(?:run|executed|performed|attempted)\b[^\n;]{0,80}\b(?:tests?|checks?|commands?|lint|typecheck|build|smoke(?:\s+test)?|integration(?:\s+tests?)?)\b|\b(?:could\s+not|couldn't|cannot|can'?t|unable\s+to)\s+(?:run|perform|execute)\b[^\n;]{0,80}\b(?:tests?|checks?|commands?)\b/i;
const ASSISTANT_INCOMPLETE_PATTERN =
  /\bwork\s+remains?\b|\b(?:TODO|TBD)\b|\b(?:work|tasks?|issues?|implementation|verification|tests?|checks?|commands?|deployment)\b[^\n;]{0,80}\b(?:incomplete|unresolved|blocked|pending|still\s+running)\b|\b(?:incomplete|unresolved|blocked|pending)\b[^\n;]{0,80}\b(?:work|tasks?|issues?|implementation|verification|tests?|checks?|commands?|deployment)\b|\b(?:must\s+still|still\s+needs?|needs?\s+to|next\s+action)\b/i;
const ASSISTANT_CLEAR_UNRESOLVED_PATTERN =
  /\bblocked\s*(?::|by\b|on\b)|\b(?:(?:could\s+not|couldn't|cannot|can'?t|unable\s+to)\s+(?:finish|complete|continue|proceed)\b|(?:credentials?|api\s+keys?|access\s+tokens?|signing\s+keys?)\s+(?:are\s+|remain\s+)?(?:missing|unavailable)|(?:task|work|request|implementation|migration)\s+remains?\s+unfinished)\b/i;
const ASSISTANT_DECISION_PATTERN = /\b(?:key\s+)?decision(?:\s+\d+)?\s*:/i;
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
  /\b(?:0+\s+(?:(?:tests?|checks?)\s+)?(?:fails?|failed|failing|failures?|errors?|skipped)|(?:fails?|failed|failing|failures?|errors?|skipped)\s*[:=]\s*0+|no\s+(?:(?:tests?|checks?)\s+)?(?:failed|failing|failures?|errors?)|no\s+(?:tests?|checks?)\s+(?:were|was|are|is|have\s+been|has\s+been)?\s*skipped|(?:tests?|checks?)\s+(?:are|is|were|was)\s+not\s+(?:failing|failed))\b/gi;
const RESULT_FAILURE_PATTERNS = [
  NONZERO_EXIT_PATTERN,
  POSITIVE_FAILURE_COUNT_PATTERN,
  /\bpermission denied\b/i,
  /\bcommand not found\b/i,
  /\b(?:failed|failing|failures|errored|fatal)\b|\bfailure\b(?!-)/i,
  /\b(?:timed\s*out|timeout|cancelled|canceled|aborted|killed|sigterm|sigkill|signal\s+\d+)\b/i,
] as const;
const RESULT_NOT_RUN_PATTERNS = [
  /\b(?:skipped|not[- ]?run|unrun|not\s+executed|never\s+(?:ran|run|executed|performed|attempted)|unable\s+to\s+(?:run|perform|execute)|(?:has|have|had)\s+yet\s+to\s+(?:run|perform|execute))\b/i,
  /\bno\s+(?:tests?|checks?)\s+(?:(?:was|were)\s+)?(?:ran|run|executed|performed)\b/i,
  /\bno\s+tests?\s+have\s+run\b/i,
  /\bno\s+tests?\s+have\s+been\s+executed\b/i,
  /\btests?\s+have\s+not\s+yet\s+run\b/i,
  /\b(?:tests?|checks?)\s+were\s+skipped\b/i,
  /\b(?:tests?|checks?|smoke|commands?)[^\n.;]{0,120}\b(?:was|were)?\s*not\s+(?:run|executed|performed)\b/i,
  /\b(?:tests?|checks?|smoke|commands?)[^\n.;]{0,120}\bnever\s+(?:ran|run|executed|performed|attempted)\b/i,
  /\b(?:has|have|had)\s+yet\s+to\s+(?:run|perform|execute|be\s+executed|be\s+performed)\b/i,
  /\b(?:tests?|checks?|smoke|commands?)[^\n.;]{0,120}\b(?:could\s+not|couldn't|cannot|can'?t|unable\s+to)\s+(?:be\s+)?(?:run|performed|executed)\b/i,
  /\b(?:could\s+not|couldn't|cannot|can'?t|unable\s+to)\s+(?:run|perform|execute)\b[^\n.;]{0,80}\b(?:tests?|checks?|commands?)\b/i,
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
  /\b(?:decision(?:\s+\d+)?|next\s+action|pytest|vitest|jest|mocha|tsc|bun\s+(?:run\s+)?(?:test|check|lint|typecheck|build|smoke)|npm\s+(?:run\s+)?(?:test|check|lint|typecheck|build|smoke)|pnpm\s+(?:run\s+)?(?:test|check|lint|typecheck|build|smoke)|yarn\s+(?:run\s+)?(?:test|check|lint|typecheck|build|smoke)|cargo\s+test|go\s+test|tests?|checks?|verification|git\s+[A-Za-z-]+|committed|pushed|push\s+(?:to|origin|upstream|remote)|PR|pull request|deployed|deployment|credentials?|api\s+keys?|access\s+tokens?|signing\s+keys?|permission denied|command not found|failed|failing|failure|errors?|not\s+executed|not[- ]?run|never\s+(?:ran|run|executed|performed|attempted)|unable\s+to\s+(?:run|perform|execute)|(?:has|have|had)\s+yet\s+to\s+(?:run|perform|execute)|unrun|skipped|incomplete|unfinished|unresolved|blocked|pending|running|timed\s*out|timeout|cancelled|canceled|aborted|killed|SIGTERM|SIGKILL|process\s+exited\s+with\s+code\s+\d+)\b/gi;
const EVIDENCE_ANCHOR_PATTERN =
  /\b(?:decision(?:\s+\d+)?|next\s+action|credentials?|api\s+keys?|access\s+tokens?|signing\s+keys?|permission denied|command not found|no\s+(?:tests?|checks?)\s+(?:(?:was|were)\s+)?(?:ran|run|executed|performed)|not[- ]?run|never\s+(?:ran|run|executed|performed|attempted)|unable\s+to\s+(?:run|perform|execute)|(?:has|have|had)\s+yet\s+to\s+(?:run|perform|execute)|not\s+(?:pushed|committed|deployed|merged)|unrun|skipped|timed\s*out|timeout|cancelled|canceled|aborted|killed|SIGTERM|SIGKILL|failed|failing|failure|errored|incomplete|unfinished|unresolved|blocked|pending|running|[1-9]\d*\s+(?:(?:tests?|checks?)\s+)?(?:fails?|failed|failing|failures?|errors?)|(?:process\s+)?exit(?:ed)?(?:\s+with)?\s+(?:code|status)\s*[:=]?\s*\d+|\d+\s+(?:(?:tests?|checks?)\s+)?(?:pass|passed|passing)|working tree\s+(?:clean|dirty)|commit\s+[0-9a-f]{6,40}|push succeeded)\b/gi;

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
    || ASSISTANT_NOT_RUN_PATTERN.test(value)
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
      incomplete:
        ASSISTANT_INCOMPLETE_PATTERN.test(withoutZeroCounts)
        || ASSISTANT_CLEAR_UNRESOLVED_PATTERN.test(withoutZeroCounts),
    },
    withoutZeroCounts,
  );
}

function classifyAssistantSource(value: string): AssistantClassification {
  const result = classifyAssistantResultPolarity(value);
  const verification = hasAssistantVerificationEvidence(value);
  const vcs = hasAssistantVcsEvidence(value);
  const vcsResult = classifyAssistantVcsResultPolarity(value);
  const failed = result.failed || (vcs && vcsResult.failed);
  const notRun = result.notRun || (vcs && vcsResult.notRun);
  const incomplete = result.incomplete || (vcs && vcsResult.incomplete);
  return {
    verification,
    vcs,
    failed,
    notRun,
    incomplete,
    unresolved: failed || notRun || incomplete,
  };
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

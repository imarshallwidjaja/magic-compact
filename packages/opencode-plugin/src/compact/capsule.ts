import type { Part, TextPart, ToolPart } from "@opencode-ai/sdk/v2";
import { isRecord, unwrapString } from "../util";
import type { MessageWithParts, Turn } from "./plan";

type VerificationState = "passed" | "failed" | "not-run" | "unknown";

type Observation = {
  ref: string;
  key: string | null;
  state: "success" | "unresolved";
};

type PreservedFinalTextPart = {
  id: string;
  text: string;
};

export type AgenticTurnEvidence = {
  evidenceRefs: string[];
  requiredUnresolvedRefs: string[];
  preservedFinalTextParts: PreservedFinalTextPart[];
  internalSourceIdentifiers: string[];
};

type StructuredResult =
  | { kind: "none" }
  | { kind: "success" }
  | { kind: "unresolved" };

type ExitMetadata =
  | { kind: "absent" }
  | { kind: "valid"; code: number }
  | { kind: "invalid" };

export function analyzeAgenticTurnEvidence(
  turn: Turn,
): AgenticTurnEvidence | null {
  if (turnContainsMagicSummary(turn)) return null;
  const evidenceRefs = collectEvidenceRefs(turn);
  const state = resolveTurnState(turn);
  const finalAssistant = turn.assistants.at(-1);
  const preservedFinalTextParts =
    finalAssistant?.parts.flatMap(part =>
      isEligibleAssistantText(part) && isSubstantiveText(part.text)
        ? [{ id: part.id, text: part.text }]
        : [],
    ) ?? [];

  return {
    evidenceRefs,
    requiredUnresolvedRefs: [...state.values()],
    preservedFinalTextParts,
    internalSourceIdentifiers: collectInternalSourceIdentifiers(turn),
  };
}

function turnContainsMagicSummary(turn: Turn): boolean {
  return [...turn.user, ...turn.assistants].some(message =>
    message.parts.some(
      part => part.type === "text" && isMagicSummaryPart(part),
    ),
  );
}

function collectEvidenceRefs(turn: Turn): string[] {
  const refs: string[] = [];
  const seen = new Set<string>();
  for (const message of [...turn.user, ...turn.assistants]) {
    addEvidenceRef(refs, seen, "msg", message.info.id);
    for (const part of message.parts) {
      addEvidenceRef(refs, seen, "part", part.id);
    }
  }
  return refs;
}

function addEvidenceRef(
  refs: string[],
  seen: Set<string>,
  namespace: "msg" | "part",
  id: unknown,
): void {
  if (typeof id !== "string" || id.length === 0) {
    throw new Error("Candidate trace contains an invalid source identifier.");
  }
  const ref = `${namespace}:${id}`;
  if (seen.has(ref)) {
    throw new Error("Candidate trace contains a duplicate namespaced ref.");
  }
  seen.add(ref);
  refs.push(ref);
}

function collectInternalSourceIdentifiers(turn: Turn): string[] {
  const identifiers = new Set<string>();
  const add = (value: unknown) => {
    if (typeof value === "string" && value.length > 0) identifiers.add(value);
  };
  for (const message of [...turn.user, ...turn.assistants]) {
    const info = message.info as unknown as Record<string, unknown>;
    add(info["id"]);
    add(info["sessionID"]);
    if (message.info.role === "assistant") add(info["parentID"]);
    for (const part of message.parts) {
      const record = part as unknown as Record<string, unknown>;
      add(record["id"]);
      add(record["sessionID"]);
      add(record["messageID"]);
      if (part.type === "tool") add(record["callID"]);
    }
  }
  return [...identifiers];
}

function isSubstantiveText(value: string): boolean {
  return value.trim().length > 0 && /[\p{L}\p{N}]/u.test(value);
}

function isEligibleAssistantText(part: Part): part is TextPart {
  return (
    part.type === "text" && part.synthetic !== true && !isMagicSummaryPart(part)
  );
}

function isMagicSummaryPart(part: TextPart): boolean {
  const magicCompact = isRecord(part.metadata)
    ? part.metadata["magicCompact"]
    : undefined;
  return isRecord(magicCompact) && magicCompact["summary"] === true;
}

function messageEvidenceRef(message: MessageWithParts): string {
  return `msg:${message.info.id}`;
}

function partEvidenceRef(part: Part): string {
  return `part:${part.id}`;
}

function resolveTurnState(turn: Turn): Map<string, string> {
  const unresolved = new Map<string, string>();

  for (const assistant of turn.assistants) {
    applyAssistantStatus(assistant, unresolved);
    for (const part of assistant.parts) {
      if (part.type !== "tool") continue;
      const observation = observeTool(part);
      if (observation.state === "success") {
        if (observation.key !== null) unresolved.delete(observation.key);
        continue;
      }
      unresolved.set(observation.key ?? observation.ref, observation.ref);
    }
  }

  return unresolved;
}

function applyAssistantStatus(
  assistant: MessageWithParts,
  unresolved: Map<string, string>,
): void {
  const info = assistant.info as unknown as Record<string, unknown>;
  if (info["error"] !== undefined) {
    unresolved.set("assistant-message", messageEvidenceRef(assistant));
    return;
  }
  if (info["finish"] !== undefined) unresolved.delete("assistant-message");
}

function observeTool(part: ToolPart): Observation {
  const ref = partEvidenceRef(part);
  const command = sourceCommand(part);
  const key = semanticKey(part, command, executionContext(part));
  const structured = sourceStructuredResult(part);
  if (structured.kind === "success") return { ref, key, state: "success" };
  if (structured.kind === "unresolved") {
    return { ref, key, state: "unresolved" };
  }
  if (part.state.status !== "completed") {
    return { ref, key, state: "unresolved" };
  }

  const exit = exitMetadata(part);
  if (exit.kind === "invalid") return { ref, key, state: "unresolved" };
  const verification = classifyVerification(part.state.output, command);
  if (verification === "failed") return { ref, key, state: "unresolved" };
  if (verification === "not-run") {
    return { ref, key: null, state: "unresolved" };
  }
  if (exit.kind === "valid" && exit.code !== 0) {
    return { ref, key, state: "unresolved" };
  }
  return { ref, key, state: "success" };
}

function sourceStructuredResult(part: ToolPart): StructuredResult {
  if (
    !isKnownStructuredProducer(part.tool)
    || part.state.status !== "completed"
  ) {
    return { kind: "none" };
  }
  const parsed = parseWholeJsonValue(part.state.output);
  if (parsed === undefined || !isPlainRecord(parsed)) return { kind: "none" };

  const hasOk = hasOwn(parsed, "ok");
  const hasSuccess = hasOwn(parsed, "success");
  const hasStatus = hasOwn(parsed, "status");
  if (!hasOk && !hasSuccess && !hasStatus) return { kind: "none" };
  const malformed = (): StructuredResult => ({ kind: "unresolved" });
  if (!isFiniteJsonValue(parsed)) return malformed();

  const ok = parsed["ok"];
  const success = parsed["success"];
  const status = parsed["status"];
  if (
    (hasOk && typeof ok !== "boolean")
    || (hasSuccess && typeof success !== "boolean")
    || (hasStatus && typeof status !== "string")
  ) {
    return malformed();
  }
  if (hasOk && hasSuccess && ok !== success) return malformed();
  if (
    typeof status === "string"
    && status !== "completed"
    && status !== "done"
    && status !== "blocked"
    && status !== "failed"
    && status !== "partial"
  ) {
    return malformed();
  }

  const successSignal = ok === true || success === true;
  const failureSignal = ok === false || success === false;
  const successfulStatus = status === "completed" || status === "done";
  const unresolvedStatus =
    status === "blocked" || status === "failed" || status === "partial";
  if (
    (successSignal && unresolvedStatus)
    || (failureSignal && successfulStatus)
  ) {
    return malformed();
  }

  const details = structuredDetails(parsed);
  if (details === null) return malformed();
  if (failureSignal || unresolvedStatus) {
    if (status === "blocked" && !details.hasBlocker) return malformed();
    if (details.reason === null) return malformed();
    return { kind: "unresolved" };
  }
  if (successSignal || successfulStatus) {
    if (details.hasBlocker || details.hasError) return malformed();
    return { kind: "success" };
  }
  return malformed();
}

function structuredDetails(value: Record<string, unknown>): {
  reason: string | null;
  hasBlocker: boolean;
  hasError: boolean;
} | null {
  const hasBlocker = hasOwn(value, "blocker");
  const hasError = hasOwn(value, "error");
  let blockerReason: string | null = null;
  const blocker = value["blocker"];
  if (hasBlocker) {
    if (!isPlainRecord(blocker)) return null;
    blockerReason = nonEmptyString(blocker["reason"]);
    if (blockerReason === null) return null;
    if (
      hasOwn(blocker, "recommendation")
      && nonEmptyString(blocker["recommendation"]) === null
    ) {
      return null;
    }
  }

  let errorReason: string | null = null;
  const error = value["error"];
  if (hasError) {
    if (typeof error === "string") {
      errorReason = nonEmptyString(error);
    } else if (isPlainRecord(error)) {
      errorReason = nonEmptyString(error["message"]);
    }
    if (errorReason === null) return null;
  }
  if (
    hasOwn(value, "nextAction")
    && nonEmptyString(value["nextAction"]) === null
  ) {
    return null;
  }
  return {
    reason: blockerReason ?? errorReason,
    hasBlocker,
    hasError,
  };
}

function parseWholeJsonValue(value: unknown): unknown | undefined {
  if (isPlainRecord(value)) return value;
  if (typeof value !== "string") return undefined;
  const trimmed = value.trim();
  if (!trimmed.startsWith("{") || !trimmed.endsWith("}")) return undefined;
  try {
    return JSON.parse(trimmed) as unknown;
  } catch {
    return undefined;
  }
}

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  if (!isRecord(value) || Array.isArray(value)) return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

function hasOwn(value: Record<string, unknown>, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(value, key);
}

function nonEmptyString(value: unknown): string | null {
  const string = unwrapString(value)?.trim();
  return string ? string : null;
}

function isFiniteJsonValue(
  value: unknown,
  ancestors = new Set<object>(),
): boolean {
  if (
    value === null
    || typeof value === "string"
    || typeof value === "boolean"
  ) {
    return true;
  }
  if (typeof value === "number") return Number.isFinite(value);
  if (typeof value !== "object") return false;
  if (ancestors.has(value)) return false;
  if (!Array.isArray(value) && !isPlainRecord(value)) return false;
  ancestors.add(value);
  let items: unknown[];
  try {
    items = Array.isArray(value) ? value : Object.values(value);
  } catch {
    ancestors.delete(value);
    return false;
  }
  const valid = items.every(item => isFiniteJsonValue(item, ancestors));
  ancestors.delete(value);
  return valid;
}

function isKnownStructuredProducer(tool: string): boolean {
  const normalized = normalizeProducerIdentity(tool);
  return normalized === "task" || normalized.startsWith("hive_");
}

function normalizeProducerIdentity(value: string): string {
  return value.trim().toLowerCase().split(/[.:/]/).at(-1) ?? value;
}

function classifyVerification(
  output: unknown,
  command: string | null,
): VerificationState {
  if (!isVerificationCommand(command)) return "unknown";
  let text: string | undefined;
  if (typeof output === "string") {
    text = output;
  } else {
    try {
      text = JSON.stringify(output);
    } catch {
      return "unknown";
    }
  }
  if (!text) return "unknown";
  const counts = verificationCounts(text);
  if (counts.failed > 0 || counts.errors > 0) return "failed";
  if (counts.passed > 0) return "passed";
  if (
    counts.explicitNoRun
    || counts.explicitZeroPassed
    || counts.skipped > 0
    || counts.todo > 0
    || counts.cancelled > 0
  ) {
    return "not-run";
  }
  return "unknown";
}

function verificationCounts(text: string): {
  passed: number;
  failed: number;
  errors: number;
  skipped: number;
  todo: number;
  cancelled: number;
  explicitNoRun: boolean;
  explicitZeroPassed: boolean;
} {
  const counts = {
    passed: 0,
    failed: 0,
    errors: 0,
    skipped: 0,
    todo: 0,
    cancelled: 0,
    explicitNoRun:
      /\b(?:no tests? ran|0 tests? ran|not[- ]run|not run|did not run)\b/i.test(
        text,
      ),
    explicitZeroPassed: false,
  };
  const add = (kind: string, amount: number) => {
    if (!Number.isSafeInteger(amount) || amount < 0) return;
    if (/^pass(?:ed|es)?$/i.test(kind)) {
      counts.passed += amount;
      if (amount === 0) counts.explicitZeroPassed = true;
    } else if (/^fail(?:ed|ures?)?$/i.test(kind)) {
      counts.failed += amount;
    } else if (/^errors?$/i.test(kind)) {
      counts.errors += amount;
    } else if (/^skipp?ed$/i.test(kind) || /^skips?$/i.test(kind)) {
      counts.skipped += amount;
    } else if (/^todos?$/i.test(kind)) {
      counts.todo += amount;
    } else if (/^cancell?ed$/i.test(kind) || /^canceled$/i.test(kind)) {
      counts.cancelled += amount;
    }
  };
  for (const match of text.matchAll(
    /\b(\d{1,3}(?:,\d{3})*|\d+)\s+(?:tests?\s+)?(passed|passes|pass|failed|failures?|errors?|skipped|skips?|todos?|cancelled|canceled)\b/gi,
  )) {
    add(match[2]!, Number(match[1]!.replaceAll(",", "")));
  }
  for (const match of text.matchAll(
    /\b(passed|passes|pass|failed|failures?|errors?|skipped|skips?|todos?|cancelled|canceled)\s*[:=]\s*(\d{1,3}(?:,\d{3})*|\d+)\b/gi,
  )) {
    add(match[1]!, Number(match[2]!.replaceAll(",", "")));
  }
  return counts;
}

function exitMetadata(part: ToolPart): ExitMetadata {
  const state = part.state as unknown as Record<string, unknown>;
  const metadata = isRecord(state["metadata"]) ? state["metadata"] : {};
  if (!Object.prototype.hasOwnProperty.call(metadata, "exit")) {
    return { kind: "absent" };
  }
  const exit = metadata["exit"];
  if (typeof exit === "number" && Number.isSafeInteger(exit)) {
    return { kind: "valid", code: exit };
  }
  return { kind: "invalid" };
}

function semanticKey(
  part: ToolPart,
  command: string | null,
  context: string | null,
): string | null {
  const producerScope = structuredProducerScope(part);
  if (producerScope !== null) return producerScope;
  if (command === null || command.includes("\0") || context === null) {
    return null;
  }
  return `${normalizeProducerIdentity(part.tool)}\0${command}\0${context}`;
}

function structuredProducerScope(part: ToolPart): string | null {
  if (!isKnownStructuredProducer(part.tool)) return null;
  const input = isRecord(part.state.input) ? part.state.input : {};
  for (const key of ["task", "runId", "identifier", "sessionID"] as const) {
    const value = unwrapString(input[key]);
    if (value?.trim()) {
      return `${normalizeProducerIdentity(part.tool)}\0${key}\0${value.trim()}`;
    }
  }
  return null;
}

function executionContext(part: ToolPart): string | null {
  const input = isRecord(part.state.input) ? part.state.input : {};
  const cwd = unwrapString(input["cwd"]);
  const workdir = unwrapString(input["workdir"]);
  if (cwd && workdir) {
    const left = normalizeAbsolutePath(cwd);
    const right = normalizeAbsolutePath(workdir);
    return left !== null && left === right ? left : null;
  }
  const only = cwd || workdir;
  if (!only) return null;
  return normalizeAbsolutePath(only);
}

function normalizeAbsolutePath(value: string): string | null {
  const trimmed = value.trim();
  if (!trimmed || trimmed.includes("\0")) return null;
  const hasForward = trimmed.includes("/");
  const hasBack = trimmed.includes("\\");
  if (trimmed.startsWith("/")) {
    if (hasBack) return null;
    return `posix:${normalizeSegments(trimmed.split("/"), "/")}`;
  }
  const drive = trimmed.match(/^([A-Za-z]):([\\/].*)$/);
  if (drive) {
    if (hasForward && hasBack) return null;
    const tail = drive[2]!.replaceAll("\\", "/");
    return `windows:${drive[1]!.toLowerCase()}:${normalizeSegments(tail.split("/"), "/")}`;
  }
  return null;
}

function normalizeSegments(segments: string[], separator: string): string {
  const normalized: string[] = [];
  for (const segment of segments) {
    if (!segment || segment === ".") continue;
    if (segment === "..") {
      normalized.pop();
      continue;
    }
    normalized.push(segment);
  }
  return `${separator}${normalized.join(separator)}`;
}

function sourceCommand(part: ToolPart): string | null {
  const input = isRecord(part.state.input) ? part.state.input : {};
  const command = unwrapString(input["command"]);
  const trimmed = command?.trim();
  return trimmed ? trimmed : null;
}

function isVerificationCommand(command: string | null): boolean {
  return (
    command !== null
    && /(?:^|\s)(?:bun|npm|pnpm|yarn|cargo|go|pytest|python|uv|vitest|jest|tsx|tsc|eslint|prettier)\b/i.test(
      command,
    )
    && /\b(?:test|check|lint|typecheck|tsc|vitest|jest|pytest|prettier|eslint)\b/i.test(
      command,
    )
  );
}

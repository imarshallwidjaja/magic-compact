import type { Part, TextPart, ToolPart } from "@opencode-ai/sdk/v2";
import { inspect } from "node:util";
import { countGptTokensWithinLimit } from "../stats/tokenize";
import { isRecord, unwrapString } from "../util";
import type { MessageWithParts, Turn } from "./plan";

export const DETERMINISTIC_SUMMARY_MAX_TOKENS = 2_048;

type VerificationState = "passed" | "failed" | "not-run" | "unknown";

type Observation = {
  key: string | null;
  state: "success" | "unresolved";
  section: "Verification" | "VCS" | "Unresolved";
  text: string;
  next: string | null;
};

type CurrentUnresolved = {
  text: string;
  next: string | null;
};

type StructuredResult =
  | { kind: "none" }
  | { kind: "success"; text: string }
  | { kind: "unresolved"; text: string; next: string | null };

type ExitMetadata =
  | { kind: "absent" }
  | { kind: "valid"; code: number }
  | { kind: "invalid" };

const PROVIDER_SAFE_CODE_POINT = (value: number): boolean =>
  value === 0x09
  || value === 0x0a
  || value === 0x0d
  || (value >= 0x20 && value <= 0xd7ff)
  || (value >= 0xe000 && value <= 0xfffd)
  || (value >= 0x10000 && value <= 0x10ffff);

export function renderCanonicalCapsule(turn: Turn): string | null {
  if (
    [...turn.user, ...turn.assistants].some(message =>
      message.parts.some(
        part => part.type === "text" && isMagicSummaryPart(part),
      ),
    )
  ) {
    return null;
  }
  const capsule = renderCapsule(turn);
  return countGptTokensWithinLimit(capsule, DETERMINISTIC_SUMMARY_MAX_TOKENS)
    === null
    ? null
    : capsule;
}

function renderCapsule(turn: Turn): string {
  const assistantTexts = collectAssistantTexts(turn);
  const state = resolveTurnState(turn);
  const lines = [
    `Outcome: ${state.unresolved.size === 0 ? "completed" : "partial"}`,
  ];

  lines.push("Work:");
  if (assistantTexts.length === 0) {
    lines.push("- No eligible assistant text parts.");
  } else {
    for (const [index, text] of assistantTexts.entries()) {
      lines.push(`- Assistant text ${index + 1} of ${assistantTexts.length}:`);
      lines.push("```json");
      lines.push(jsonString(text));
      lines.push("```");
    }
  }

  appendSection(lines, "Verification", state.verification);
  appendSection(lines, "VCS", state.vcs);
  appendSection(lines, "Unresolved", [...state.unresolved.values()]);
  appendSection(
    lines,
    "Next",
    [...state.unresolved.values()].flatMap(item =>
      item.next === null ? [] : [{ text: item.next, next: null }],
    ),
  );

  return lines.join("\n");
}

function appendSection(
  lines: string[],
  label: "Verification" | "VCS" | "Unresolved" | "Next",
  items: Array<{ text: string }>,
): void {
  if (items.length === 0) return;
  lines.push(`${label}:`);
  for (const item of items) {
    lines.push(`- ${item.text}`);
  }
}

function collectAssistantTexts(turn: Turn): string[] {
  const texts: string[] = [];
  for (const assistant of turn.assistants) {
    for (const part of assistant.parts) {
      if (!isEligibleAssistantText(part)) continue;
      if (part.text.length === 0) continue;
      texts.push(part.text);
    }
  }
  return texts;
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

function resolveTurnState(turn: Turn): {
  verification: Observation[];
  vcs: Observation[];
  unresolved: Map<string, CurrentUnresolved>;
} {
  const verificationByKey = new Map<string, Observation>();
  const vcsByKey = new Map<string, Observation>();
  const unresolved = new Map<string, CurrentUnresolved>();
  let order = 0;

  for (const assistant of turn.assistants) {
    order = applyAssistantStatus(assistant, unresolved, order);
    for (const part of assistant.parts) {
      if (part.type !== "tool") continue;
      const observation = observeTool(part);
      order += 1;
      if (
        observation.section === "Verification"
        && observation.state === "success"
      ) {
        if (observation.key !== null)
          verificationByKey.set(observation.key, observation);
      } else if (
        observation.section === "VCS"
        && observation.state === "success"
      ) {
        if (observation.key !== null)
          vcsByKey.set(observation.key, observation);
      }
      if (observation.state === "success") {
        if (observation.key !== null) unresolved.delete(observation.key);
        continue;
      }
      const key = observation.key ?? `unscoped:${order}`;
      unresolved.set(key, {
        text: observation.text,
        next: observation.next,
      });
    }
  }

  return {
    verification: [...verificationByKey.values()],
    vcs: [...vcsByKey.values()],
    unresolved,
  };
}

function applyAssistantStatus(
  assistant: MessageWithParts,
  unresolved: Map<string, CurrentUnresolved>,
  order: number,
): number {
  const info = assistant.info as unknown as Record<string, unknown>;
  if (info["error"] !== undefined) {
    unresolved.set("assistant-message", {
      text: `assistant message ${assistant.info.id} errored: ${inlineText(formatValue(info["error"]))}`,
      next: null,
    });
    return order + 1;
  }
  if (info["finish"] !== undefined) {
    unresolved.delete("assistant-message");
  }
  return order;
}

function observeTool(part: ToolPart): Observation {
  const command = sourceCommand(part);
  const context = executionContext(part);
  const key = semanticKey(part, command, context);
  const label = toolLabel(part, command);
  const structured = sourceStructuredResult(part);
  if (structured.kind === "success") {
    return {
      key,
      state: "success",
      section: isVcsCommand(command) ? "VCS" : "Verification",
      text: structured.text,
      next: null,
    };
  }
  if (structured.kind === "unresolved") {
    return {
      key,
      state: "unresolved",
      section: "Unresolved",
      text: structured.text,
      next: structured.next,
    };
  }

  if (part.state.status !== "completed") {
    return {
      key,
      state: "unresolved",
      section: "Unresolved",
      text: `${label} is ${part.state.status}${toolError(part)}`,
      next: null,
    };
  }

  const exit = exitMetadata(part);
  if (exit.kind === "invalid") {
    return {
      key,
      state: "unresolved",
      section: "Unresolved",
      text: `${label} has invalid exit metadata`,
      next: null,
    };
  }
  const verification = classifyVerification(part.state.output, command);
  if (verification === "failed") {
    return {
      key,
      state: "unresolved",
      section: "Unresolved",
      text: `${label} failed: ${inlineText(formatValue(part.state.output))}`,
      next: null,
    };
  }
  if (verification === "not-run") {
    return {
      key: null,
      state: "unresolved",
      section: "Unresolved",
      text: `${label} did not run: ${inlineText(formatValue(part.state.output))}`,
      next: null,
    };
  }
  if (exit.kind === "valid" && exit.code !== 0) {
    return {
      key,
      state: "unresolved",
      section: "Unresolved",
      text: `${label} exited ${exit.code}`,
      next: null,
    };
  }
  if (verification === "passed") {
    return {
      key,
      state: "success",
      section: "Verification",
      text: `${label} passed: ${inlineText(formatValue(part.state.output))}`,
      next: null,
    };
  }
  if (isVcsCommand(command)) {
    return {
      key,
      state: "success",
      section: "VCS",
      text: `${label} completed`,
      next: null,
    };
  }
  return {
    key,
    state: "success",
    section: "Verification",
    text: `${label} completed`,
    next: null,
  };
}

function sourceStructuredResult(part: ToolPart): StructuredResult {
  if (
    !isKnownStructuredProducer(part.tool)
    || part.state.status !== "completed"
  ) {
    return { kind: "none" };
  }
  const output = part.state.output;
  const parsed = parseWholeJsonValue(output);
  if (parsed === undefined) return { kind: "none" };
  if (!isPlainRecord(parsed)) return { kind: "none" };

  const hasOk = hasOwn(parsed, "ok");
  const hasSuccess = hasOwn(parsed, "success");
  const hasStatus = hasOwn(parsed, "status");
  if (!hasOk && !hasSuccess && !hasStatus) return { kind: "none" };
  const malformed = () => invalidStructuredResult(part, output);
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
    return {
      kind: "unresolved",
      text: `${toolLabel(part, sourceCommand(part))} ${unresolvedStatus ? status : "failed"}: ${details.reason}`,
      next: details.next,
    };
  }
  if (successSignal || successfulStatus) {
    if (details.hasBlocker || details.hasError) return malformed();
    return {
      kind: "success",
      text: `${toolLabel(part, sourceCommand(part))} completed`,
    };
  }
  return malformed();
}

function invalidStructuredResult(
  part: ToolPart,
  value: unknown,
): StructuredResult {
  return {
    kind: "unresolved",
    text: `${toolLabel(part, sourceCommand(part))} returned malformed structured result: ${formatValue(value)}`,
    next: null,
  };
}

function structuredDetails(value: Record<string, unknown>): {
  reason: string | null;
  next: string | null;
  hasBlocker: boolean;
  hasError: boolean;
} | null {
  const hasBlocker = hasOwn(value, "blocker");
  const hasError = hasOwn(value, "error");
  let blockerReason: string | null = null;
  let recommendation: string | null = null;
  const blocker = value["blocker"];
  if (hasBlocker) {
    if (!isPlainRecord(blocker)) return null;
    blockerReason = nonEmptyString(blocker["reason"]);
    if (blockerReason === null) return null;
    if (hasOwn(blocker, "recommendation")) {
      recommendation = nonEmptyString(blocker["recommendation"]);
      if (recommendation === null) return null;
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

  let next: string | null = null;
  if (hasOwn(value, "nextAction")) {
    next = nonEmptyString(value["nextAction"]);
    if (next === null) return null;
  }
  return {
    reason: blockerReason ?? errorReason,
    next: next ?? recommendation,
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
  const valid = (Array.isArray(value) ? value : Object.values(value)).every(
    item => isFiniteJsonValue(item, ancestors),
  );
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
  const text = typeof output === "string" ? output : JSON.stringify(output);
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

function isVcsCommand(command: string | null): boolean {
  return command !== null && /^git\s+/i.test(command);
}

function toolLabel(part: ToolPart, command: string | null): string {
  return command === null ? part.tool : `${part.tool} \`${command}\``;
}

function toolError(part: ToolPart): string {
  if (part.state.status !== "error") return "";
  const error = formatValue(part.state.error);
  return error ? `: ${inlineText(error)}` : "";
}

function formatValue(value: unknown): string {
  if (typeof value === "string") return value;
  try {
    const serialized = JSON.stringify(value);
    if (serialized !== undefined && isFiniteJsonValue(value)) return serialized;
  } catch {
    return inspect(value, {
      depth: null,
      breakLength: Infinity,
      compact: true,
    });
  }
  return inspect(value, { depth: null, breakLength: Infinity, compact: true });
}

function inlineText(value: string): string {
  return value.replace(/\s+/g, " ").trim();
}

function jsonString(value: string): string {
  const serialized = JSON.stringify(value);
  if (serialized === undefined) {
    throw new Error("Assistant text is not JSON-serializable.");
  }
  return escapeInvalidCodePoints(serialized);
}

function escapeInvalidCodePoints(value: string): string {
  let result = "";
  for (const character of value) {
    const codePoint = character.codePointAt(0)!;
    if (PROVIDER_SAFE_CODE_POINT(codePoint)) {
      result += character;
      continue;
    }
    for (let index = 0; index < character.length; index += 1) {
      result += `\\u${character.charCodeAt(index).toString(16).padStart(4, "0")}`;
    }
  }
  return result;
}

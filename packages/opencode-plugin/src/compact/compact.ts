import type {
  Agent,
  PermissionRuleset,
  Session,
  TextPart,
} from "@opencode-ai/sdk/v2";
import { randomBytes } from "node:crypto";
import { unwrap, type V2Client } from "../api";
import {
  attemptWithinDeadline,
  CONTROL_OPERATION_TIMEOUT_MS,
  isRecord,
} from "../util";
import {
  MAGIC_COMPACT_SUMMARIZER_AGENT,
  boundaryPartID,
  summaryEnvelopePartID,
  summaryEnvelopeTextPart,
  turnHasPersistedSummary,
} from "./constants";
import { createMediaManifest, type MediaManifest } from "./media";
import {
  assertForkMessageShape,
  buildTurns,
  type CompactionPlan,
  type MessageWithParts,
  type Turn,
} from "./plan";

const GENERATION_PERMISSION: PermissionRuleset = [
  { permission: "*", pattern: "*", action: "deny" },
];
const ANCHOR_CHARACTER_LIMIT = 300;
const GENERATION_NONCE_ATTEMPTS = 8;
const GENERATION_NONCE_PATTERN = /^[A-Za-z0-9_-]{22}$/u;

type CompactionSettings = {
  model: { providerID: string; modelID: string };
  variant?: string;
};

type GenerationCandidate = {
  planIndex: number;
  ordinal: number;
  anchor: string;
  marker: string;
  mediaManifest?: MediaManifest;
};

type GenerationTurnResult = {
  summary: string;
  visualDescriptions: Array<{ ordinal: number; description: string }>;
};

type SummaryGenerationOptions = {
  controlTimeoutMs?: number;
  nonceFactory?: () => string;
};

type GenerationRoute = {
  directory: string;
  workspace?: string;
};

type GenerationStage =
  | "fork"
  | "permission"
  | "history"
  | "marker"
  | "provider"
  | "validation"
  | "abort"
  | "delete";

export type GeneratedSummary = {
  text: string;
  mediaManifest?: MediaManifest;
} | null;

export async function generateCompactionSummaries(
  v2: V2Client,
  session: Session,
  plan: CompactionPlan,
  options: SummaryGenerationOptions = {},
): Promise<GeneratedSummary[]> {
  const controlTimeoutMs =
    options.controlTimeoutMs ?? CONTROL_OPERATION_TIMEOUT_MS;
  if (!Number.isFinite(controlTimeoutMs) || controlTimeoutMs <= 0) {
    throw new Error("Compaction generation control deadline must be positive.");
  }

  assertGeneratedPartIDsAvailable(plan);
  const generated: GeneratedSummary[] = plan.summarizedTurns.map(() => null);
  const candidates = plan.summarizedTurns.flatMap((turn, planIndex) => {
    if (turnHasPersistedSummary(turn)) return [];
    const mediaManifest = createMediaManifest(turn);
    return [
      {
        planIndex,
        ordinal: 0,
        anchor: firstOrdinaryUserLine(turn),
        marker: "",
        ...(mediaManifest ? { mediaManifest } : {}),
      },
    ];
  });
  candidates.forEach((candidate, index) => {
    candidate.ordinal = index + 1;
  });
  if (candidates.length === 0) return generated;

  const settings = await resolveCompactionSettings(v2, session);
  if (historyMayExposeImages(plan.sourceSnapshot)) {
    await assertConfiguredModelSupportsImages(v2, session, settings);
  }
  const results = await requestCompactionGeneration(
    v2,
    session,
    settings,
    candidates,
    plan,
    controlTimeoutMs,
    options.nonceFactory ?? createGenerationNonce,
  );
  for (const [index, candidate] of candidates.entries()) {
    const result = results[index]!;
    generated[candidate.planIndex] = {
      text: frameSummary(result),
      ...(candidate.mediaManifest
        ? { mediaManifest: candidate.mediaManifest }
        : {}),
    };
  }
  return generated;
}

export async function injectSummaries(
  v2: V2Client,
  sessionID: string,
  compactionTurns: Turn[],
  summaries: GeneratedSummary[],
): Promise<TextPart[]> {
  const inserted: TextPart[] = [];
  for (const [index, turn] of compactionTurns.entries()) {
    const summary = summaries[index];
    if (summary === undefined || summary === null) {
      throw new Error("Missing summary for assistant turn.");
    }
    const firstAssistant = turn.assistants[0];
    if (!firstAssistant) throw new Error("Turn missing assistant message.");
    const part = summaryEnvelopeTextPart(
      firstAssistant.info.id,
      sessionID,
      summary.text,
      summary.mediaManifest,
    );
    unwrap(
      await v2.part.update({
        sessionID,
        messageID: firstAssistant.info.id,
        partID: part.id,
        part,
      }),
    );
    inserted.push(part);
  }
  return inserted;
}

async function requestCompactionGeneration(
  v2: V2Client,
  sourceSession: Session,
  settings: CompactionSettings,
  candidates: GenerationCandidate[],
  plan: CompactionPlan,
  controlTimeoutMs: number,
  nonceFactory: () => string,
): Promise<GenerationTurnResult[]> {
  const route: GenerationRoute = {
    directory: sourceSession.directory,
    ...(sourceSession.workspaceID
      ? { workspace: sourceSession.workspaceID }
      : {}),
  };
  let forkPromise: Promise<Session> | undefined;
  const forkOutcome = await attemptWithinDeadline(signal => {
    forkPromise = v2.session
      .fork({ sessionID: sourceSession.id, ...route }, { signal })
      .then(unwrapLifecycleResponse);
    return forkPromise;
  }, controlTimeoutMs);
  if (forkOutcome.kind !== "success") {
    if (forkOutcome.kind === "timed-out") {
      if (forkPromise) {
        void forkPromise
          .then(fork =>
            cleanupLateGenerationFork(v2, fork, route, controlTimeoutMs),
          )
          .catch(() => undefined);
      }
      throw lifecycleFailure(sourceSession.id, "fork", "timeout");
    }
    throw lifecycleFailure(
      sourceSession.id,
      "fork",
      "rejection",
      forkOutcome.error,
    );
  }
  const fork = forkOutcome.value;

  let operationError: Error | undefined;
  let results: GenerationTurnResult[] | undefined;
  let generationNonce = "";
  let stopMarker = "";
  const permissionOutcome = await attemptWithinDeadline(
    signal =>
      v2.session
        .update(
          {
            sessionID: fork.id,
            ...route,
            permission: GENERATION_PERMISSION,
          },
          { signal },
        )
        .then(unwrapLifecycleResponse),
    controlTimeoutMs,
  );
  if (permissionOutcome.kind !== "success") {
    operationError = lifecycleFailure(
      sourceSession.id,
      "permission",
      permissionOutcome.kind === "timed-out" ? "timeout" : "rejection",
      permissionOutcome.kind === "failed" ? permissionOutcome.error : undefined,
    );
  }

  if (!operationError) {
    const messagesOutcome = await attemptWithinDeadline(
      signal =>
        v2.session
          .messages({ sessionID: fork.id, ...route }, { signal })
          .then(unwrapLifecycleResponse),
      controlTimeoutMs,
    );
    if (messagesOutcome.kind !== "success") {
      operationError = lifecycleFailure(
        sourceSession.id,
        "history",
        messagesOutcome.kind === "timed-out" ? "timeout" : "rejection",
        messagesOutcome.kind === "failed" ? messagesOutcome.error : undefined,
      );
    } else {
      try {
        assertForkMessageShape(messagesOutcome.value, plan);
        generationNonce = selectGenerationNonce(
          [plan.sourceSnapshot, messagesOutcome.value],
          nonceFactory,
        );
        candidates.forEach(candidate => {
          candidate.marker = generationTurnMarker(
            generationNonce,
            candidate.ordinal,
          );
        });
        stopMarker = generationStopMarker(
          generationNonce,
          candidates.length + 1,
        );
        const markerTargets = generationMarkerTargets(
          messagesOutcome.value,
          fork.id,
          plan,
          candidates,
          stopMarker,
        );
        for (const target of markerTargets) {
          const markerOutcome = await attemptWithinDeadline(
            signal =>
              v2.part
                .update(
                  {
                    sessionID: fork.id,
                    messageID: target.message.info.id,
                    partID: target.part.id,
                    ...route,
                    part: target.part,
                  },
                  { signal },
                )
                .then(unwrapLifecycleResponse),
            controlTimeoutMs,
          );
          if (markerOutcome.kind !== "success") {
            operationError = lifecycleFailure(
              sourceSession.id,
              "marker",
              markerOutcome.kind === "timed-out" ? "timeout" : "rejection",
              markerOutcome.kind === "failed" ? markerOutcome.error : undefined,
            );
            break;
          }
        }
      } catch (error) {
        operationError = lifecycleFailure(
          sourceSession.id,
          "history",
          "rejection",
          error,
        );
      }
    }
  }

  if (!operationError) {
    let stage: Extract<GenerationStage, "provider" | "validation"> = "provider";
    try {
      const body = unwrapLifecycleResponse(
        await v2.session.prompt({
          sessionID: fork.id,
          ...route,
          agent: MAGIC_COMPACT_SUMMARIZER_AGENT,
          model: settings.model,
          ...(settings.variant ? { variant: settings.variant } : {}),
          parts: [
            {
              type: "text",
              text: buildGenerationPrompt(candidates, plan, stopMarker),
            },
          ],
        }),
      );
      if (isRecord(body) && body["info"] && isRecord(body["info"])) {
        const responseInfo = body["info"];
        const responseError = responseInfo["error"];
        if (responseError) throw responseError;
        if (isOutputTruncation(responseInfo["finish"])) {
          throw { name: "MessageOutputLengthError" };
        }
      }
      stage = "validation";
      results = parseGenerationResponse(
        isRecord(body) ? body["parts"] : undefined,
        candidates,
        generationNonce,
      );
    } catch (error) {
      operationError = lifecycleFailure(
        sourceSession.id,
        stage,
        "rejection",
        error,
      );
      const abortOutcome = await attemptWithinDeadline(
        signal =>
          v2.session
            .abort({ sessionID: fork.id, ...route }, { signal })
            .then(unwrapLifecycleResponse),
        controlTimeoutMs,
      );
      if (abortOutcome.kind !== "success") {
        operationError = combineLifecycleErrors(
          operationError,
          lifecycleFailure(
            sourceSession.id,
            "abort",
            abortOutcome.kind === "timed-out" ? "timeout" : "rejection",
            abortOutcome.kind === "failed" ? abortOutcome.error : undefined,
          ),
        );
      }
    }
  }

  const deleteOutcome = await attemptWithinDeadline(
    signal =>
      v2.session
        .delete({ sessionID: fork.id, ...route }, { signal })
        .then(unwrapLifecycleResponse),
    controlTimeoutMs,
  );
  const deleteError =
    deleteOutcome.kind === "success"
      ? undefined
      : lifecycleFailure(
          sourceSession.id,
          "delete",
          deleteOutcome.kind === "timed-out" ? "timeout" : "rejection",
          deleteOutcome.kind === "failed" ? deleteOutcome.error : undefined,
        );
  if (operationError && deleteError) {
    throw combineLifecycleErrors(operationError, deleteError);
  }
  if (operationError) throw operationError;
  if (deleteError) throw deleteError;
  return results!;
}

async function cleanupLateGenerationFork(
  v2: V2Client,
  fork: Session,
  route: GenerationRoute,
  controlTimeoutMs: number,
): Promise<void> {
  await attemptWithinDeadline(
    signal =>
      v2.session
        .abort({ sessionID: fork.id, ...route }, { signal })
        .then(unwrapLifecycleResponse),
    controlTimeoutMs,
  );
  await attemptWithinDeadline(
    signal =>
      v2.session
        .delete({ sessionID: fork.id, ...route }, { signal })
        .then(unwrapLifecycleResponse),
    controlTimeoutMs,
  );
}

function generationMarkerTargets(
  messages: MessageWithParts[],
  forkSessionID: string,
  plan: CompactionPlan,
  candidates: GenerationCandidate[],
  stopMarker: string,
): Array<{ message: MessageWithParts; part: TextPart }> {
  const turns = buildTurns(messages);
  const targets = candidates.map(candidate => {
    const turn = turns[plan.sourceTurnStartIndex + candidate.planIndex];
    const message = turn?.user[0];
    if (!message) {
      throw new Error("Compaction generation fork history shape mismatch.");
    }
    return {
      message,
      part: generationMarkerPart(
        forkSessionID,
        message.info.id,
        candidate.marker,
        candidate.ordinal,
        false,
      ),
    };
  });
  if (plan.nextTurn) {
    const turn = turns[plan.sourceTurnStartIndex + plan.summarizedTurns.length];
    const message = turn?.user[0];
    if (!message) {
      throw new Error("Compaction generation fork history shape mismatch.");
    }
    targets.push({
      message,
      part: generationMarkerPart(
        forkSessionID,
        message.info.id,
        stopMarker,
        candidates.length + 1,
        true,
      ),
    });
  }
  for (const target of targets) {
    if (target.message.parts.some(part => part.id === target.part.id)) {
      throw new Error("Compaction generation fork marker collision.");
    }
  }
  return targets;
}

function generationMarkerPart(
  sessionID: string,
  messageID: string,
  text: string,
  ordinal: number,
  stop: boolean,
): TextPart {
  return {
    id: `prt_-magic_generation_${stop ? "stop" : "turn"}_${ordinal}_${messageID}`,
    sessionID,
    messageID,
    type: "text",
    text,
    synthetic: true,
    metadata: {
      magicCompact: {
        generationMarker: true,
        ordinal,
        ...(stop ? { stop: true } : {}),
      },
    },
  };
}

function generationTurnMarker(nonce: string, ordinal: number): string {
  return `<magic-compact-turn nonce="${nonce}" ordinal="${ordinal}" />`;
}

function generationStopMarker(nonce: string, ordinal: number): string {
  return `<magic-compact-stop nonce="${nonce}" ordinal="${ordinal}" />`;
}

function createGenerationNonce(): string {
  return randomBytes(16).toString("base64url");
}

function selectGenerationNonce(
  histories: unknown[],
  nonceFactory: () => string,
): string {
  for (let attempt = 0; attempt < GENERATION_NONCE_ATTEMPTS; attempt += 1) {
    const nonce = nonceFactory();
    if (!GENERATION_NONCE_PATTERN.test(nonce)) {
      throw new Error("Compaction generation marker nonce is invalid.");
    }
    if (!histories.some(history => containsString(history, nonce))) {
      return nonce;
    }
  }
  throw new GenerationMarkerNonceCollisionError();
}

function containsString(value: unknown, needle: string): boolean {
  const pending = [value];
  const seen = new Set<object>();
  while (pending.length > 0) {
    const current = pending.pop();
    if (typeof current === "string") {
      if (current.includes(needle)) return true;
      continue;
    }
    if (
      (typeof current !== "object" && typeof current !== "function")
      || current === null
      || seen.has(current)
    ) {
      continue;
    }
    seen.add(current);
    for (const key of Object.keys(current)) {
      pending.push(ownDataProperty(current, key));
    }
  }
  return false;
}

async function resolveCompactionSettings(
  v2: V2Client,
  sourceSession: Session,
): Promise<CompactionSettings> {
  let agents: Agent[];
  try {
    agents = unwrap(
      await v2.app.agents({
        directory: sourceSession.directory,
        ...(sourceSession.workspaceID
          ? { workspace: sourceSession.workspaceID }
          : {}),
      }),
    );
  } catch {
    throw new Error(
      "Magic Compact could not resolve the native compaction agent.",
    );
  }
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
  if (
    !compactionAgent.model
    || !compactionAgent.model.providerID.trim()
    || !compactionAgent.model.modelID.trim()
  ) {
    throw new Error(
      "Magic Compact requires an explicit model on the native compaction agent.",
    );
  }
  return {
    model: compactionAgent.model,
    ...(compactionAgent.variant ? { variant: compactionAgent.variant } : {}),
  };
}

async function assertConfiguredModelSupportsImages(
  v2: V2Client,
  sourceSession: Session,
  settings: CompactionSettings,
): Promise<void> {
  let catalog: ReturnType<typeof unwrap>;
  try {
    catalog = unwrap(
      await v2.provider.list({
        directory: sourceSession.directory,
        ...(sourceSession.workspaceID
          ? { workspace: sourceSession.workspaceID }
          : {}),
      }),
    );
  } catch {
    throw new Error(
      "Magic Compact could not verify an image-capable configured compaction model.",
    );
  }
  if (
    !isRecord(catalog)
    || !Array.isArray(catalog["all"])
    || !catalog["all"].some(provider => {
      if (!isRecord(provider) || provider["id"] !== settings.model.providerID) {
        return false;
      }
      const models = provider["models"];
      if (!isRecord(models)) return false;
      const model = models[settings.model.modelID];
      if (!isRecord(model)) return false;
      const capabilities = model["capabilities"];
      if (!isRecord(capabilities)) return false;
      const input = capabilities["input"];
      return isRecord(input) && input["image"] === true;
    })
  ) {
    throw new Error(
      "Magic Compact requires an image-capable configured compaction model for complete fork history containing images.",
    );
  }
}

function historyMayExposeImages(sourceSnapshot: unknown): boolean {
  if (!Array.isArray(sourceSnapshot)) return false;
  return sourceSnapshot.some(message => {
    if (!isRecord(message) || !Array.isArray(message["parts"])) return false;
    const role = isRecord(message["info"])
      ? message["info"]["role"]
      : undefined;
    return message["parts"].some(part => {
      if (!isRecord(part)) return false;
      if (
        (role === "user" || role === "assistant")
        && part["type"] === "file"
        && typeof part["mime"] === "string"
        && part["mime"].startsWith("image/")
      ) {
        return true;
      }
      if (part["type"] !== "tool" || !isRecord(part["state"])) return false;
      const attachments = part["state"]["attachments"];
      return (
        Array.isArray(attachments)
        && attachments.some(
          attachment =>
            isRecord(attachment)
            && typeof attachment["mime"] === "string"
            && attachment["mime"].startsWith("image/"),
        )
      );
    });
  });
}

function buildGenerationPrompt(
  candidates: GenerationCandidate[],
  plan: CompactionPlan,
  stopMarker: string,
): string {
  return [
    "Generate one continuation summary for each indexed turn using the complete forked source history.",
    "Return only the JSON object required by your system prompt.",
    "Use the ephemeral markers only as ordering guides. Echo each exact marker token, public ordinal, and anchor; marker echo is allowed. Never include the active generation nonce or marker token in summaries or visual descriptions; useful source identifiers may remain when you include them.",
    JSON.stringify({
      turns: candidates.map(candidate => ({
        ordinal: candidate.ordinal,
        marker: candidate.marker,
        anchor: candidate.anchor,
        expectedInlineImages: candidate.mediaManifest?.images.length ?? 0,
      })),
      ...(plan.nextTurn
        ? {
            nextTurn: {
              marker: stopMarker,
              anchor: firstOrdinaryUserLine(plan.nextTurn),
            },
          }
        : {}),
    }),
  ].join("\n");
}

function parseGenerationResponse(
  parts: unknown,
  candidates: GenerationCandidate[],
  nonce: string,
): GenerationTurnResult[] {
  if (!Array.isArray(parts)) throw validationFailure();
  const textParts = parts.flatMap(part => {
    if (!isRecord(part) || part["type"] !== "text") return [];
    const text = part["text"];
    return typeof text === "string" && text.trim() ? [text] : [];
  });
  if (textParts.length !== 1) throw validationFailure();

  let parsed: unknown;
  try {
    parsed = JSON.parse(textParts[0]!);
  } catch {
    throw validationFailure();
  }
  if (
    !isPlainRecord(parsed)
    || !hasExactKeys(parsed, ["turns"])
    || !Array.isArray(parsed["turns"])
    || parsed["turns"].length !== candidates.length
  ) {
    throw validationFailure();
  }

  return parsed["turns"].map((value, index) => {
    const candidate = candidates[index]!;
    if (
      !isPlainRecord(value)
      || !hasExactKeys(value, [
        "anchor",
        "marker",
        "ordinal",
        "summary",
        "visualDescriptions",
      ])
      || value["ordinal"] !== candidate.ordinal
      || value["marker"] !== candidate.marker
      || value["anchor"] !== candidate.anchor
      || typeof value["summary"] !== "string"
      || !value["summary"].trim()
      || !Array.isArray(value["visualDescriptions"])
    ) {
      throw validationFailure();
    }
    const expectedImages = candidate.mediaManifest?.images.length ?? 0;
    if (value["visualDescriptions"].length !== expectedImages) {
      throw validationFailure();
    }
    const summary = value["summary"].trim();
    if (summary.includes(nonce)) {
      throw validationFailure();
    }
    const visualDescriptions = value["visualDescriptions"].map(
      (description, imageIndex) => {
        if (
          !isPlainRecord(description)
          || !hasExactKeys(description, ["description", "ordinal"])
          || description["ordinal"] !== imageIndex + 1
          || typeof description["description"] !== "string"
          || !description["description"].trim()
        ) {
          throw validationFailure();
        }
        const text = description["description"].trim();
        if (text.includes(nonce)) {
          throw validationFailure();
        }
        return {
          ordinal: imageIndex + 1,
          description: text,
        };
      },
    );
    return {
      summary,
      visualDescriptions,
    };
  });
}

function frameSummary(result: GenerationTurnResult): string {
  if (result.visualDescriptions.length === 0) return result.summary;
  return [
    result.summary,
    "",
    "Visual summary:",
    ...result.visualDescriptions.map(
      item => `${item.ordinal}. ${item.description}`,
    ),
  ].join("\n");
}

function firstOrdinaryUserLine(turn: Turn): string {
  const text = turn.user
    .flatMap(message => message.parts)
    .find(
      (part): part is TextPart =>
        part.type === "text"
        && part.synthetic !== true
        && part.ignored !== true,
    )?.text;
  const firstLine = (text ?? "").split(/\r\n|\r|\n/u, 1)[0] ?? "";
  return Array.from(firstLine).slice(0, ANCHOR_CHARACTER_LIMIT).join("");
}

function assertGeneratedPartIDsAvailable(plan: CompactionPlan): void {
  const partIDs = new Set(
    [...plan.summarizedTurns, ...(plan.nextTurn ? [plan.nextTurn] : [])]
      .flatMap(turn => [...turn.user, ...turn.assistants])
      .flatMap(message => message.parts.map(part => part.id)),
  );
  for (const turn of plan.summarizedTurns) {
    if (turnHasPersistedSummary(turn)) continue;
    const assistant = turn.assistants[0];
    if (!assistant) throw new Error("Turn missing assistant message.");
    if (partIDs.has(summaryEnvelopePartID(assistant.info.id))) {
      throw new Error(
        "Compaction plan collides with a generated summary part.",
      );
    }
  }
  const nextUser = plan.nextTurn?.user[0];
  if (nextUser && partIDs.has(boundaryPartID(nextUser.info.id))) {
    throw new Error(
      "Compaction plan collides with the generated boundary part.",
    );
  }
}

function validationFailure(): Error {
  return new Error("Compaction generation response validation failed.");
}

function hasExactKeys(
  value: Record<string, unknown>,
  expected: string[],
): boolean {
  const keys = Object.keys(value).sort();
  const sortedExpected = [...expected].sort();
  return (
    keys.length === sortedExpected.length
    && keys.every((key, index) => key === sortedExpected[index])
  );
}

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  if (!isRecord(value) || Array.isArray(value)) return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

type LifecycleFailureKind = "timeout" | "rejection";

class GenerationMarkerNonceCollisionError extends Error {
  constructor() {
    super("Compaction generation marker nonce collision.");
    this.name = "GenerationMarkerNonceCollisionError";
  }
}

class GenerationLifecycleError extends Error {
  readonly status: number | undefined;

  constructor(
    readonly sourceSessionID: string,
    readonly stage: GenerationStage,
    readonly kind: LifecycleFailureKind,
    status: number | undefined,
    message: string,
  ) {
    super(message);
    this.name = "GenerationLifecycleError";
    this.status = status;
  }
}

function lifecycleFailure(
  sourceSessionID: string,
  stage: GenerationStage,
  kind: LifecycleFailureKind,
  error?: unknown,
): GenerationLifecycleError {
  const status = safeStatus(error);
  const overflow = stage === "provider" && isContextOverflow(error, status);
  const truncated = stage === "provider" && isOutputTruncation(error);
  const message =
    error instanceof GenerationMarkerNonceCollisionError
      ? error.message
      : overflow
        ? "The complete forked session exceeded the configured compaction model context. Run /magic-trim first, then retry /magic-compact."
        : truncated
          ? "The compaction model reached its output limit. Retry /magic-compact with a larger N so fewer turns are selected."
          : lifecycleMessage(stage, kind);
  return new GenerationLifecycleError(
    sourceSessionID,
    stage,
    kind,
    status,
    message,
  );
}

function lifecycleMessage(
  stage: GenerationStage,
  kind: LifecycleFailureKind,
): string {
  if (stage === "fork") {
    return kind === "timeout"
      ? "Compaction generation fork creation timed out."
      : "Compaction generation fork creation failed.";
  }
  if (stage === "permission") {
    return kind === "timeout"
      ? "Compaction generation fork permission update timed out."
      : "Compaction generation fork permission update failed.";
  }
  if (stage === "history") {
    return kind === "timeout"
      ? "Compaction generation fork history verification timed out."
      : "Compaction generation fork history shape verification failed.";
  }
  if (stage === "marker") {
    return kind === "timeout"
      ? "Compaction generation fork marker injection timed out."
      : "Compaction generation fork marker injection failed.";
  }
  if (stage === "provider")
    return "Compaction generation model request failed.";
  if (stage === "validation") {
    return "Compaction generation response validation failed.";
  }
  if (stage === "abort") {
    return kind === "timeout"
      ? "Compaction generation fork abort timed out."
      : "Compaction generation fork abort failed.";
  }
  return kind === "timeout"
    ? "Compaction generation fork deletion timed out."
    : "Compaction generation fork deletion failed.";
}

function safeStatus(error: unknown): number | undefined {
  for (const item of diagnosticChain(error)) {
    const data = ownDataProperty(item, "data");
    const value =
      ownDataProperty(item, "status")
      ?? ownDataProperty(item, "statusCode")
      ?? ownDataProperty(data, "statusCode");
    if (
      typeof value === "number"
      && Number.isInteger(value)
      && value >= 100
      && value <= 599
    ) {
      return value;
    }
  }
  return undefined;
}

function isContextOverflow(
  error: unknown,
  status: number | undefined,
): boolean {
  return (
    status === 413
    || diagnosticChain(error).some(item => {
      const categories = diagnosticCategories(item);
      return categories.some(category =>
        [
          "contextoverflowerror",
          "context_length_exceeded",
          "context_window_exceeded",
          "prompt_too_long",
          "input_too_long",
        ].includes(category),
      );
    })
  );
}

function isOutputTruncation(value: unknown): boolean {
  const outputCategories = [
    "length",
    "max_token",
    "max_tokens",
    "max_output_tokens",
    "output_length",
    "output_length_exceeded",
    "messageoutputlengtherror",
  ];
  if (typeof value === "string" && outputCategories.includes(category(value))) {
    return true;
  }
  return diagnosticChain(value).some(item => {
    return diagnosticCategories(item).some(itemCategory =>
      outputCategories.includes(itemCategory),
    );
  });
}

function diagnosticChain(value: unknown): unknown[] {
  const chain: unknown[] = [];
  const pending = [value];
  const seen = new Set<object>();
  while (pending.length > 0 && chain.length < 32) {
    const current = pending.shift();
    if (current === undefined) continue;
    if (
      (typeof current === "object" || typeof current === "function")
      && current !== null
    ) {
      if (seen.has(current)) continue;
      seen.add(current);
    }
    chain.push(current);
    pending.push(
      ownDataProperty(current, "cause"),
      ownDataProperty(current, "error"),
      ownDataProperty(current, "data"),
    );
  }
  return chain;
}

function diagnosticCategories(value: unknown): string[] {
  return ["name", "code", "type", "reason"].flatMap(key => {
    const item = ownDataProperty(value, key);
    return typeof item === "string" ? [category(item)] : [];
  });
}

function category(value: string): string {
  return value.trim().toLowerCase().replaceAll("-", "_");
}

function ownDataProperty(value: unknown, key: string): unknown {
  if ((typeof value !== "object" && typeof value !== "function") || !value) {
    return undefined;
  }
  try {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    return descriptor && "value" in descriptor ? descriptor.value : undefined;
  } catch {
    return undefined;
  }
}

function unwrapLifecycleResponse<T, E>(response: { data?: T; error?: E }): T {
  if ("error" in response && response.error) throw response.error;
  return response.data as T;
}

function combineLifecycleErrors(primary: Error, secondary: Error): Error {
  return new AggregateError(
    [primary, secondary],
    "Compaction generation and fork cleanup both failed.",
    { cause: primary },
  );
}

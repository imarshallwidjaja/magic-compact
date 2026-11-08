// One fresh whole-session summary for task-aware retention. A deny-all fork
// of the source is prompted once through OpenCode's native hidden compaction
// agent, so native history conversion supplies the model-visible history,
// including the most recent turns and any prior native summary. The fork is
// always deleted, and only a response of the exact expected shape is
// accepted.
import type { PermissionRuleset, Session } from "@opencode-ai/sdk/v2";
import { randomBytes } from "node:crypto";
import { isDeepStrictEqual } from "node:util";
import type { V2Client } from "../api";
import { estimateNativeHistoryTokens } from "../compact/batch";
import {
  computeNativeBatchBudget,
  fits,
  NATIVE_BATCH_FIXED_RESERVE_TOKENS,
  refuse,
  type Fit,
} from "../policy/budget";
import {
  assertConfiguredModelSupportsImages,
  assertForkRecordIdentity,
  historyMayExposeImages,
  isQualifiedDeepSeekModel,
  stageReport,
  type CompactionSettings,
  type LoadSummarizerCatalog,
  type SummarizerCatalog,
} from "../compact/compact";
import {
  indexReplay,
  reasoningUnits,
  type ReplayIndex,
} from "../compact/reasoning";
import {
  chooseStage,
  matchCandidate,
  sameModel,
  settleMatch,
  type StageReport,
} from "../policy/reasoning";
import { hasPluginMessage, PluginMessageError } from "../policy/feedback";
import { isContextOverflow, safeStatus } from "../policy/generation";
import {
  assertForkMessageShape,
  nativeModelVisibleMessages,
  snapshotSourceMessages,
  type MessageWithParts,
} from "../compact/plan";
import {
  createForkCut,
  ForkHistoryError,
  sessionRoute,
  type ForkCut,
} from "../compact/fork";
import { registerOwnedFork, releaseOwnedFork } from "../compact/owned";
import {
  MAGIC_COMPACT_SUMMARIZER_AGENT,
  SESSION_SUMMARY_PROMPT,
  SESSION_SUMMARY_RESPONSE_SCHEMA,
  SESSION_SUMMARY_TOOL_PROMPT,
} from "../policy/protocol";
import { countGptTokens } from "../stats/tokenize";
import {
  attemptWithinDeadline,
  CONTROL_OPERATION_TIMEOUT_MS,
  isNonArrayRecord,
} from "../util";
import {
  caughtSummaryStage,
  failedSummaryStages,
  SESSION_SUMMARY_TOO_LARGE,
  SessionSummaryError,
  sessionSummaryStageFailure,
  structuredSessionSummary,
  type SessionSummary,
  type SessionSummaryStage,
  type SessionSummaryStageFailure,
} from "../policy/retention";

export { SESSION_SUMMARY_PROMPT };

const SUMMARY_AGENT = MAGIC_COMPACT_SUMMARIZER_AGENT;
// The session rule is required: the prompting agent's own permissions would
// otherwise override the summarizer agent's deny rules.
const SUMMARY_FORK_PERMISSION: PermissionRuleset = [
  { permission: "*", pattern: "*", action: "deny" },
];
// The structured summary keeps the host's response tool, as per-turn
// generation does; the last matching rule wins.
const STRUCTURED_SUMMARY_FORK_PERMISSION: PermissionRuleset = [
  ...SUMMARY_FORK_PERMISSION,
  { permission: "StructuredOutput", pattern: "*", action: "allow" },
];
const STRUCTURED_SUMMARY_FORMAT = {
  type: "json_schema" as const,
  schema: SESSION_SUMMARY_RESPONSE_SCHEMA,
};
// The first page of a structured fork read holds only the reply and the
// prompt; later pages hold older history.
const STRUCTURED_FIRST_PAGE = 2;
const FORK_PAGE_SIZE = 500;
const ASSISTANT_PART_TYPES = new Set([
  "step-start",
  "step-finish",
  "reasoning",
  "text",
]);
const ID_RANDOM_ALPHABET =
  "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
const ID_ORDER_WAIT_ATTEMPTS = 50;

export { SessionSummaryError, type SessionSummaryStage };

export type SessionSummaryOptions = {
  // The command's summarizer catalog, shared with its turn requests.
  catalog: LoadSummarizerCatalog;
  matchModel?: boolean;
  controlTimeoutMs?: number;
};

// How the summary request is made: as plain text, or on the DeepSeek route
// through the host's response tool, which is the only way that model reads
// the earlier plain reasoning the history carries.
type SummaryRequest = {
  settings: CompactionSettings;
  structured: boolean;
  prompt: string;
};

export async function generateSessionSummary(
  v2: V2Client,
  session: Session,
  messages: MessageWithParts[],
  options: SessionSummaryOptions,
): Promise<SessionSummary> {
  const controlTimeoutMs =
    options.controlTimeoutMs ?? CONTROL_OPERATION_TIMEOUT_MS;
  const sourceSnapshot = snapshotSourceMessages(messages);
  // Budget and image capability follow what the model will see; content
  // hidden behind a native checkpoint is never sent.
  let visible: MessageWithParts[];
  try {
    visible = nativeModelVisibleMessages(messages);
  } catch {
    throw sessionSummaryStageFailure("history");
  }
  const requiresImages = historyMayExposeImages(visible);
  // One attribution of the history for every candidate request's estimate.
  const replay = indexReplay(visible);
  let request: SummaryRequest;
  let report: StageReport;
  try {
    const catalog = await options.catalog(requiresImages);
    const configured = catalog.configured;
    // Matching unlocks only the current model's own protected reasoning;
    // reasoning from other models in the history stays unreadable.
    const primary = session.model;
    const units = reasoningUnits(visible).filter(
      unit =>
        primary !== undefined
        && unit.origin !== undefined
        && sameModel(unit.origin, {
          providerID: primary.providerID,
          modelID: primary.id,
        }),
    );
    const match = matchCandidate(units, catalog.qualifies);
    const summaryRequest = (settings: CompactionSettings): SummaryRequest => {
      const structured =
        isQualifiedDeepSeekModel(settings.model, settings.descriptor)
        && visible.some(message =>
          message.parts.some(
            part => part.type === "reasoning" && part.text.trim() !== "",
          ),
        );
      return {
        settings,
        structured,
        prompt: structured
          ? SESSION_SUMMARY_TOOL_PROMPT
          : SESSION_SUMMARY_PROMPT,
      };
    };
    const fit = (candidate: SummaryRequest): Fit<SummaryRequest> => {
      const budget = computeNativeBatchBudget(
        candidate.settings.descriptor.limit,
      );
      if (budget.kind === "capacity") return budget;
      return summaryRequestTokens(candidate, visible, replay, catalog)
        > budget.value.hardSingleTurnInput
        ? refuse(SESSION_SUMMARY_TOO_LARGE)
        : fits(candidate);
    };
    const settled = settleMatch(
      chooseStage(match, configured, model => {
        const descriptor = catalog.describe(model);
        return {
          variants: new Set(descriptor?.variants ?? []),
          optionsConflict: catalog.summarizerOptions,
          imagesBlocked:
            requiresImages && descriptor?.capabilities?.input?.image !== true,
        };
      }),
      options.matchModel === true,
      (model, variant) =>
        fit(
          summaryRequest({
            model,
            ...(variant === undefined ? {} : { variant }),
            descriptor: catalog.describe(model)!,
          }),
        ),
    );
    request = settled.matched ?? summaryRequest(configured);
    if (requiresImages) assertConfiguredModelSupportsImages(request.settings);
    const budget = computeNativeBatchBudget(request.settings.descriptor.limit);
    if (budget.kind === "capacity")
      throw new PluginMessageError(budget.message);
    // Fail before any fork when the request cannot fit. The compacting hook
    // still stops native compaction if the estimate is exceeded anyway.
    if (
      summaryRequestTokens(request, visible, replay, catalog)
      > budget.value.hardSingleTurnInput
    ) {
      throw sessionSummaryStageFailure("budget");
    }
    report = stageReport({
      stage: "session",
      candidate: match,
      choice: settled.choice,
      settings: request.settings,
      history: visible,
      catalog,
      replay,
    });
  } catch (error) {
    // Only this plugin's own messages, which are fixed text without session
    // content, are repeated.
    if (hasPluginMessage(error)) throw error;
    throw new PluginMessageError(
      "Magic Compact could not resolve its summarizer agent.",
    );
  }
  const settings = request.settings;

  const route = sessionRoute(session);
  let fork: ForkCut;
  try {
    fork = await createForkCut(v2, session, messages);
  } catch (error) {
    throw forkFailure(error);
  }
  // Registered before its first request and released only after a confirmed
  // deletion.
  const owned = registerOwnedFork(fork.id);
  const forkID = fork.id;

  const failures: SessionSummaryStageFailure[] = [];
  let prompted = false;
  let summary: string | undefined;
  try {
    await fork.populate(controlTimeoutMs).catch((error: unknown) => {
      throw forkFailure(error);
    });
    const permission = await attemptWithinDeadline(
      signal =>
        v2.session
          .update(
            {
              sessionID: forkID,
              ...route,
              permission: request.structured
                ? STRUCTURED_SUMMARY_FORK_PERMISSION
                : SUMMARY_FORK_PERMISSION,
            },
            { signal },
          )
          .then(unwrapResponse),
      controlTimeoutMs,
    );
    if (permission.kind !== "success") {
      throw stageFailure("permission", permission);
    }

    const history = await loadForkMessages(v2, forkID, route, controlTimeoutMs);
    try {
      assertForkMessageShape(history, { sourceSnapshot });
      assertForkRecordIdentity(history, forkID);
    } catch {
      throw sessionSummaryStageFailure("history");
    }
    const messageID = await nextPromptMessageID(history);

    prompted = true;
    let response: unknown;
    try {
      response = unwrapResponse(
        await v2.session.prompt({
          sessionID: forkID,
          ...route,
          messageID,
          agent: SUMMARY_AGENT,
          model: settings.model,
          ...(settings.variant ? { variant: settings.variant } : {}),
          ...(request.structured ? { format: STRUCTURED_SUMMARY_FORMAT } : {}),
          parts: [{ type: "text", text: request.prompt }],
        }),
      );
    } catch (error) {
      if (owned.nativeCompactionBlocked) {
        throw sessionSummaryStageFailure("budget");
      }
      throw providerFailure(error);
    }
    if (owned.nativeCompactionBlocked) {
      throw sessionSummaryStageFailure("budget");
    }
    // OpenCode reports provider failures, including a context overflow when
    // auto-compaction is off, as an error on a 200 response.
    const responseInfo = isNonArrayRecord(response)
      ? response["info"]
      : undefined;
    const responseError = isNonArrayRecord(responseInfo)
      ? responseInfo["error"]
      : undefined;
    if (responseError !== undefined && responseError !== null) {
      throw providerFailure(responseError);
    }

    const final = request.structured
      ? await loadStructuredForkMessages(
          v2,
          forkID,
          route,
          controlTimeoutMs,
          history.length + 2,
        )
      : await loadForkMessages(v2, forkID, route, controlTimeoutMs);
    // A native checkpoint in the fork means OpenCode found the request too
    // large, even when the compacting hook did not run.
    if (final.slice(history.length).some(isNativeCompactionArtifact)) {
      throw sessionSummaryStageFailure("budget");
    }
    summary = acceptedSummary(
      response,
      final,
      history,
      messageID,
      forkID,
      request,
    );
  } catch (error) {
    failures.push(caughtSummaryStage(error));
  }

  // Only a prompt can leave the fork running.
  if (failures.length > 0 && prompted) {
    try {
      unwrapResponse(await v2.session.abort({ sessionID: forkID, ...route }));
    } catch (error) {
      failures.push({
        kind: "stage",
        stage: "abort",
        status: safeStatus(error),
      });
    }
  }
  try {
    unwrapResponse(await v2.session.delete({ sessionID: forkID, ...route }));
    releaseOwnedFork(owned);
  } catch (error) {
    failures.push({
      kind: "stage",
      stage: "delete",
      status: safeStatus(error),
      temporary: "fork",
      session: { id: forkID, removal: "failed" },
    });
  }
  const failure = failedSummaryStages(failures, "fork");
  if (failure) throw failure;
  return { text: summary!, report };
}

// The model-visible input of a summary request: the fixed host reserve, the
// prompt, the history with each attributable protected unit the request's
// model is sent counted by its replay estimate instead of its visible text,
// and, for a structured request, the response schema the host sends as its
// tool. Units whose replay cost is unknown count by their visible text and
// are reported.
function summaryRequestTokens(
  request: SummaryRequest,
  visible: MessageWithParts[],
  replay: ReplayIndex,
  catalog: Pick<SummarizerCatalog, "requestContext">,
): number {
  const estimate = replay.estimate(
    catalog.requestContext(request.settings.model),
  );
  return (
    NATIVE_BATCH_FIXED_RESERVE_TOKENS
    + countGptTokens(request.prompt)
    + (request.structured
      ? countGptTokens(JSON.stringify(SESSION_SUMMARY_RESPONSE_SCHEMA))
      : 0)
    + estimateNativeHistoryTokens(visible)
    - estimate.visibleTokens
    + estimate.tokens
  );
}

// Accepts the fork only when it is the unchanged rewritten source history
// plus exactly the one prompt and one finished summarizer-agent reply. Native overflow
// compaction or replay adds markers, summaries, or a re-created prompt, and a
// partial native summary must never stand in for the whole session. A plain
// reply's text is the summary. A structured reply's only tool part is one
// completed StructuredOutput call whose input is the object the host captured
// for the reply, holding exactly one nonempty summary, and the prompt
// response reports that same reply and object; its text is never read.
function acceptedSummary(
  response: unknown,
  messages: MessageWithParts[],
  history: MessageWithParts[],
  messageID: string,
  forkID: string,
  request: SummaryRequest,
): string {
  const invalid = () => sessionSummaryStageFailure("validation");
  if (messages.length !== history.length + 2) throw invalid();
  const prefix = messages.slice(0, history.length);
  try {
    assertForkRecordIdentity(messages, forkID);
  } catch {
    throw invalid();
  }
  // The persisted history is the one the prompt was sent with, record for
  // record, apart from times and usage.
  if (
    !isDeepStrictEqual(
      snapshotSourceMessages(prefix),
      snapshotSourceMessages(history),
    )
  ) {
    throw invalid();
  }

  const prompt = messages[history.length]!;
  const promptPart = prompt.parts[0];
  const format = prompt.info.role === "user" ? prompt.info.format : undefined;
  if (
    prompt.info.id !== messageID
    || prompt.info.role !== "user"
    || prompt.info.agent !== SUMMARY_AGENT
    || prompt.parts.length !== 1
    || promptPart?.type !== "text"
    || promptPart.text !== request.prompt
    || promptPart.synthetic === true
    || (request.structured
      ? format?.type !== "json_schema"
        || !isDeepStrictEqual(format.schema, STRUCTURED_SUMMARY_FORMAT.schema)
      : format !== undefined)
  ) {
    throw invalid();
  }

  const reply = messages[history.length + 1]!;
  const responseInfo = isNonArrayRecord(response)
    ? response["info"]
    : undefined;
  if (
    reply.info.role !== "assistant"
    || !isNonArrayRecord(responseInfo)
    || responseInfo["id"] !== reply.info.id
    || reply.info.parentID !== messageID
    || reply.info.agent !== SUMMARY_AGENT
    || reply.info.summary === true
    || reply.info.error !== undefined
    || reply.info.time.completed === undefined
  ) {
    throw invalid();
  }
  if (!request.structured) {
    if (
      reply.info.finish !== "stop"
      || reply.parts.some(part => !ASSISTANT_PART_TYPES.has(part.type))
    ) {
      throw invalid();
    }
    const text = reply.parts
      .flatMap(part =>
        part.type === "text" && part.synthetic !== true && part.ignored !== true
          ? [part.text]
          : [],
      )
      .join("\n\n")
      .trim();
    if (!text) throw invalid();
    return text;
  }
  const calls = reply.parts.filter(part => part.type === "tool");
  const call = calls[0];
  const captured = reply.info.structured;
  const summary = structuredSessionSummary(captured);
  if (
    (reply.info.finish !== "stop" && reply.info.finish !== "tool-calls")
    || reply.parts.some(
      part => part.type !== "tool" && !ASSISTANT_PART_TYPES.has(part.type),
    )
    || calls.length !== 1
    || call?.type !== "tool"
    || call.tool !== "StructuredOutput"
    || call.state.status !== "completed"
    || !isDeepStrictEqual(call.state.input, captured)
    || !isDeepStrictEqual(responseInfo["structured"], captured)
    || summary === undefined
  ) {
    throw invalid();
  }
  return summary;
}

async function loadForkMessages(
  v2: V2Client,
  forkID: string,
  route: { directory: string; workspace?: string },
  controlTimeoutMs: number,
): Promise<MessageWithParts[]> {
  const outcome = await attemptWithinDeadline(
    signal =>
      v2.session
        .messages({ sessionID: forkID, ...route }, { signal })
        .then(unwrapResponse),
    controlTimeoutMs,
  );
  if (outcome.kind !== "success") throw stageFailure("history", outcome);
  return outcome.value;
}

// OpenCode 1.18.30 and 1.19.0 answer HTTP 400 to a full message list, and to
// a read of the prompt itself, once a prompt stored a JSON-schema format: the
// response encoder rejects the stored format. A page that older history
// follows is returned without that check, so a structured fork is read newest
// first through the public `limit` and `before` cursor (`X-Next-Cursor`), the
// first page holding only the reply and the prompt. A missing cursor ends the
// read; a repeated cursor, or more pages than `expected` messages need, fails
// it.
async function loadStructuredForkMessages(
  v2: V2Client,
  forkID: string,
  route: { directory: string; workspace?: string },
  controlTimeoutMs: number,
  expected: number,
): Promise<MessageWithParts[]> {
  const pages: MessageWithParts[][] = [];
  const cursors = new Set<string>();
  const maxPages = 2 + Math.ceil(expected / FORK_PAGE_SIZE);
  let before: string | undefined;
  for (;;) {
    const page = before;
    const outcome = await attemptWithinDeadline(
      signal =>
        v2.session
          .messages(
            {
              sessionID: forkID,
              ...route,
              limit:
                page === undefined ? STRUCTURED_FIRST_PAGE : FORK_PAGE_SIZE,
              ...(page === undefined ? {} : { before: page }),
            },
            { signal },
          )
          .then(response => ({
            messages: unwrapResponse(response),
            cursor:
              response.response?.headers.get("x-next-cursor") ?? undefined,
          })),
      controlTimeoutMs,
    );
    if (outcome.kind !== "success") throw stageFailure("history", outcome);
    pages.unshift(outcome.value.messages);
    const cursor = outcome.value.cursor;
    if (!cursor) return pages.flat();
    if (cursors.has(cursor) || pages.length >= maxPages) {
      throw sessionSummaryStageFailure("history");
    }
    cursors.add(cursor);
    before = cursor;
  }
}

function isNativeCompactionArtifact(message: MessageWithParts): boolean {
  return (
    (message.info.role === "assistant" && message.info.summary === true)
    || message.parts.some(part => part.type === "compaction")
  );
}

// A fork that could not be created, or populated as planned.
function forkFailure(error: unknown): SessionSummaryError {
  return sessionSummaryStageFailure(
    error instanceof ForkHistoryError ? "history" : "fork",
    {
      status: safeStatus(error),
      ...(error instanceof ForkHistoryError ? { detail: error.detail } : {}),
    },
  );
}

function providerFailure(error: unknown): SessionSummaryError {
  const status = safeStatus(error);
  return sessionSummaryStageFailure("provider", {
    status,
    overflow: isContextOverflow(error, status),
  });
}

function stageFailure(
  stage: Exclude<SessionSummaryStage, "delete">,
  outcome: { kind: "timed-out" } | { kind: "failed"; error: unknown },
): SessionSummaryError {
  return sessionSummaryStageFailure(stage, {
    status: outcome.kind === "failed" ? safeStatus(outcome.error) : undefined,
  });
}

// An explicit OpenCode message ID pins the prompt's identity, so a native
// overflow replay, which re-creates the prompt under a new ID, is rejected.
// OpenCode ascending IDs encode the low 48 bits of `ms * 0x1000 + counter`
// as 12 hex digits plus 14 random base62 characters. The plugin shares
// OpenCode's process clock: waiting (bounded) until the time value passes the
// newest fork ID keeps the prompt after the history, and counter 0 keeps it
// before any reply created in the same millisecond. Validation still checks
// the resulting order.
async function nextPromptMessageID(
  history: MessageWithParts[],
): Promise<string> {
  const newest = history.reduce((max, message) => {
    const match = /^msg_([0-9a-f]{12})/u.exec(message.info.id);
    const value = match ? BigInt(`0x${match[1]}`) : 0n;
    return value > max ? value : max;
  }, 0n);
  for (
    let attempt = 0;
    attempt < ID_ORDER_WAIT_ATTEMPTS && idTimeValue() <= newest;
    attempt += 1
  ) {
    await new Promise(resolve => setTimeout(resolve, 1));
  }
  return `msg_${idTimeValue().toString(16).padStart(12, "0")}${Array.from(
    randomBytes(14),
    byte => ID_RANDOM_ALPHABET[byte % ID_RANDOM_ALPHABET.length],
  ).join("")}`;
}

function idTimeValue(): bigint {
  return (BigInt(Date.now()) * 0x1000n) & ((1n << 48n) - 1n);
}

function unwrapResponse<T, E>(response: { data?: T; error?: E }): T {
  if ("error" in response && response.error) throw response.error;
  return response.data as T;
}

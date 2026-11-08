import { randomUUID } from "node:crypto";
import { isDeepStrictEqual } from "node:util";
import {
  computeNativeBatchBudget,
  fits,
  NATIVE_BATCH_FIXED_RESERVE_TOKENS,
  refuse,
  requireFit,
  type Fit,
} from "../policy/budget";
import { PluginMessageError, type UnremovedSession } from "../policy/feedback";
import {
  MAGIC_COMPACT_SUMMARIZER_AGENT,
  SESSION_SUMMARY_PROMPT,
  SESSION_SUMMARY_RESPONSE_SCHEMA,
  SESSION_SUMMARY_TOOL_PROMPT,
} from "../policy/protocol";
import {
  chooseStage,
  matchCandidate,
  sameModel,
  settleMatch,
  stageReport,
} from "../policy/reasoning";
import {
  caughtSummaryStage,
  failedSummaryStages,
  SESSION_SUMMARY_TOO_LARGE,
  sessionSummaryStageFailure,
  structuredSessionSummary,
  type SessionSummary,
  type SessionSummaryStage,
  type SessionSummaryStageFailure,
} from "../policy/retention";
import { countGptTokens } from "../stats/tokenize";
import { attemptWithinDeadline, isRecord, jsonForm } from "../util";
import { v2ModelVisibleHistory } from "./history";
import type { V2Host } from "./host";
import { v2HistoryShowsImages } from "./media";
import {
  readableReasoningHeaderTokens,
  v2ReasoningUnits,
  v2ReplayEstimate,
} from "./reasoning";
import {
  mintV2ID,
  readV2History,
  withFreshMessageIDs,
  type V2Message,
  type V2Source,
} from "./session";
import {
  estimateV2HistoryTokens,
  matchedV2Model,
  V2_IMAGE_MODEL_REQUIRED,
  v2Configured,
  v2MatchPreflight,
  v2Qualifies,
  v2SummarizerRoute,
  type V2SummarizerModel,
} from "./summarizer";

// `generation` sessions run one per-turn summary request and may call the
// response tool; `retention` sessions run the whole-session summary and may
// call no tool; `retention-tool` sessions run the whole-session summary on the
// DeepSeek route and may call only the session summary tool.
export type OwnedSessionRole = "generation" | "retention" | "retention-tool";

type OwnedEntry = {
  role: OwnedSessionRole;
  // The request runs on OpenCode 2's OpenAI Responses serializer, which needs
  // another model's plain reasoning projected to stay readable.
  readableReasoning: boolean;
  requests: number;
  closed: boolean;
};

// Temporary sessions this plugin process created for exactly one model
// request. Only this process can register an ID, so nothing a session stores
// can claim ownership. The `context` hook lets an open owned session make one
// provider request and throws the per-process stop on any later one, and on
// every one once the session is closed; the `compaction` hook throws a
// distinct per-process token for every owned session, so a request that
// outgrew the model is told apart from a stop.
export class OwnedSessions {
  readonly stop = `Magic Compact ended its owned request (${randomUUID()}).`;
  readonly compactionBlocked = `Magic Compact blocked native compaction of its owned session (${randomUUID()}).`;
  readonly #sessions = new Map<string, OwnedEntry>();

  register(
    sessionID: string,
    role: OwnedSessionRole,
    readableReasoning: boolean,
  ): void {
    this.#sessions.set(sessionID, {
      role,
      readableReasoning,
      requests: 0,
      closed: false,
    });
  }

  // Ends the session's requests for good: every later one is stopped, whether
  // or not it ever ran. Only `release` lifts it, after a confirmed removal.
  close(sessionID: string): void {
    const entry = this.#sessions.get(sessionID);
    if (entry) entry.closed = true;
  }

  release(sessionID: string): void {
    this.#sessions.delete(sessionID);
  }

  has(sessionID: string): boolean {
    return this.#sessions.has(sessionID);
  }

  // The response tool belongs to the per-turn generation request alone.
  allowsResponseTool(sessionID: string): boolean {
    return this.#sessions.get(sessionID)?.role === "generation";
  }

  // The session summary tool belongs to the structured whole-session summary
  // alone.
  allowsSummaryTool(sessionID: string): boolean {
    return this.#sessions.get(sessionID)?.role === "retention-tool";
  }

  projectsReadableReasoning(sessionID: string): boolean {
    return this.#sessions.get(sessionID)?.readableReasoning === true;
  }

  admitRequest(sessionID: string): void {
    const entry = this.#sessions.get(sessionID);
    if (entry === undefined) return;
    entry.requests += 1;
    if (entry.closed || entry.requests > 1) throw new Error(this.stop);
  }

  rejectCompaction(sessionID: string): void {
    if (this.has(sessionID)) throw new Error(this.compactionBlocked);
  }
}

// Closes an owned session, then removes it once every request that could
// create or run it has settled. The removal is awaited without a local
// deadline, since an abandoned request could still run later; the host
// interrupts and waits for the session before removing it. The closed guard is
// released only after a confirmed removal. Otherwise it stays, so the context
// and compaction hooks keep stopping any provider request or native compaction
// in that session, even if an import or prompt request that failed on the
// client is still applied later. Returns the session when its removal was not
// confirmed: `failed` when the removal failed or its result is unknown,
// `not-found` when the host reported no such session. A session that is not
// found keeps its guard for the same reason, since a failed import may still
// be applied. It is not a cleanup failure: a request that otherwise succeeded
// still succeeds, and a failed one names the session as not found.
export async function removeOwnedSession(
  host: V2Host,
  owned: OwnedSessions,
  sessionID: string,
): Promise<UnremovedSession | undefined> {
  owned.close(sessionID);
  try {
    await host.session.remove({ sessionID });
  } catch (error) {
    return {
      id: sessionID,
      removal:
        isRecord(error) && error["_tag"] === "SessionNotFoundError"
          ? "not-found"
          : "failed",
    };
  }
  owned.release(sessionID);
  return undefined;
}

const DENY_ALL = [{ action: "*", resource: "*", effect: "deny" as const }];

// The DeepSeek route that reads earlier plain reasoning only when a request
// has tools.
function isV2DeepSeekRoute(summarizer: V2SummarizerModel): boolean {
  return (
    summarizer.ref.providerID === "deepseek"
    && summarizer.info.compatibility?.reasoningField === "reasoning_content"
  );
}

// The tool a structured whole-session summary submits through. It only
// acknowledges, and only in a `retention-tool` session this process owns.
export const V2_SUMMARY_TOOL = "SessionSummary";

const SUMMARY_TOOL_PERMISSIONS = [
  ...DENY_ALL,
  { action: V2_SUMMARY_TOOL, resource: "*", effect: "allow" as const },
];

// How the summary request is made: on the configured or matched model, as
// plain text, or on the DeepSeek route through the summary tool, which is
// the only way that model reads the earlier plain reasoning in the history.
type V2SummaryRequest = {
  summarizer: V2SummarizerModel;
  structured: boolean;
  prompt: string;
};

function summaryRequestTokens(
  request: V2SummaryRequest,
  visible: readonly V2Message[],
): number {
  const destination = v2SummarizerRoute(request.summarizer);
  return (
    NATIVE_BATCH_FIXED_RESERVE_TOKENS
    + countGptTokens(request.prompt)
    + (request.structured
      ? countGptTokens(JSON.stringify(SESSION_SUMMARY_RESPONSE_SCHEMA))
      : 0)
    + estimateV2HistoryTokens(visible).tokens
    + readableReasoningHeaderTokens(visible, destination)
  );
}

// The temporary whole-session summary for task-aware retention: a deny-all
// copy of the source under fresh IDs, prompted once through the summarizer
// agent, accepted only as that prompt plus one finished reply, and always
// removed. It never touches the source. Before anything is created, the
// history OpenCode will send (see `v2ModelVisibleHistory`), the prompt, any
// readable-reasoning headers and the fixed host reserve must fit the model's
// hard input limit, and a history that still shows an image needs an
// image-capable model. Under `--match-model` the request runs on the source's
// current model when that unlocks its own protected reasoning.
export async function generateV2SessionSummary(
  host: V2Host,
  owned: OwnedSessions,
  source: V2Source,
  summarizer: V2SummarizerModel,
  controlTimeoutMs: number,
  matchModel: boolean,
): Promise<SessionSummary> {
  let visible: V2Message[];
  let requiresImages: boolean;
  try {
    visible = v2ModelVisibleHistory(source.messages);
    requiresImages = v2HistoryShowsImages(visible);
  } catch {
    throw sessionSummaryStageFailure("history");
  }
  const primary = source.session.model;
  const units = v2ReasoningUnits(visible).filter(
    unit =>
      primary !== undefined
      && unit.origin !== undefined
      && sameModel(unit.origin, {
        providerID: primary.providerID,
        modelID: primary.id,
      }),
  );
  const summaryRequest = (model: V2SummarizerModel): V2SummaryRequest => {
    const structured =
      isV2DeepSeekRoute(model)
      && visible.some(
        message =>
          message.type === "assistant"
          && message.content.some(
            item => item.type === "reasoning" && item.text.trim() !== "",
          ),
      );
    return {
      summarizer: model,
      structured,
      prompt: structured ? SESSION_SUMMARY_TOOL_PROMPT : SESSION_SUMMARY_PROMPT,
    };
  };
  const fit = (candidate: V2SummaryRequest): Fit<V2SummaryRequest> => {
    const budget = computeNativeBatchBudget(candidate.summarizer.info.limit);
    if (budget.kind === "capacity") return budget;
    return summaryRequestTokens(candidate, visible)
      > budget.value.hardSingleTurnInput
      ? refuse(SESSION_SUMMARY_TOO_LARGE)
      : fits(candidate);
  };
  const qualifies = v2Qualifies(summarizer);
  const match = matchCandidate(units, qualifies);
  const { choice, matched } = settleMatch(
    chooseStage(match, v2Configured(summarizer), model =>
      v2MatchPreflight(summarizer, model, requiresImages),
    ),
    matchModel,
    (model, variant) =>
      fit(summaryRequest(matchedV2Model(summarizer, model, variant))),
  );
  const request = matched ?? summaryRequest(summarizer);
  const model = request.summarizer.ref;
  if (
    requiresImages
    && !request.summarizer.info.capabilities.input.includes("image")
  ) {
    throw new PluginMessageError(V2_IMAGE_MODEL_REQUIRED);
  }
  const budget = requireFit(
    computeNativeBatchBudget(request.summarizer.info.limit),
  );
  if (summaryRequestTokens(request, visible) > budget.hardSingleTurnInput) {
    throw sessionSummaryStageFailure("budget");
  }
  // OpenCode 2 replay cost is never estimated, so the request's own
  // protected units are reported as left out of its input estimate.
  const requestModel = { providerID: model.providerID, modelID: model.id };
  const report = stageReport({
    stage: "session",
    candidate: match,
    choice,
    model: requestModel,
    units: v2ReasoningUnits(visible),
    qualifies,
    replayUnknown: v2ReplayEstimate(visible, requestModel).unknown,
  });
  const control = async <T>(
    stage: Exclude<SessionSummaryStage, "delete">,
    request: (signal: AbortSignal) => Promise<T>,
  ): Promise<T> => {
    const outcome = await attemptWithinDeadline(request, controlTimeoutMs);
    if (outcome.kind !== "success") throw sessionSummaryStageFailure(stage);
    return outcome.value;
  };
  const now = Date.now();
  const sessionID = mintV2ID("ses_", now, 1, true);
  const history = withFreshMessageIDs(source.messages, now);
  const promptID = mintV2ID("msg_", now, history.length + 1, false);
  const info = {
    ...structuredClone(source.session),
    id: sessionID,
    title: "Magic Compact retention summary",
    agent: MAGIC_COMPACT_SUMMARIZER_AGENT,
    model,
    permissions: request.structured ? SUMMARY_TOOL_PERMISSIONS : DENY_ALL,
    metadata: {},
    time: { created: now, updated: now },
  } as Record<string, unknown>;
  delete info["parentID"];
  delete info["fork"];
  delete info["revert"];

  owned.register(
    sessionID,
    request.structured ? "retention-tool" : "retention",
    v2SummarizerRoute(request.summarizer).responses,
  );
  const failures: SessionSummaryStageFailure[] = [];
  let summary: string | undefined;
  try {
    // Directly awaited, like the removal: a timed-out import could still
    // create the session after cleanup.
    try {
      await host.session.import({
        info: info as never,
        messages: history as never,
        location: source.session.location,
      });
    } catch {
      throw sessionSummaryStageFailure("fork");
    }
    const read = <T>(request: (signal: AbortSignal) => Promise<T>) =>
      control("history", request);
    if (
      !isDeepStrictEqual(
        jsonForm(await readV2History(host, sessionID, read)),
        jsonForm(history),
      )
    ) {
      throw sessionSummaryStageFailure("history");
    }
    // The model request has no local deadline, like every Magic Compact
    // generation; the owned controls bound it to one provider request.
    try {
      await host.session.prompt({
        sessionID,
        id: promptID,
        text: request.prompt,
      });
      await host.session.wait({ sessionID });
    } catch {
      throw sessionSummaryStageFailure("provider");
    }
    const after = await readV2History(host, sessionID, read);
    if (
      after.some(
        message =>
          (message.type === "idle" || message.type === "assistant")
          && message.error?.message.includes(owned.compactionBlocked),
      )
    ) {
      throw sessionSummaryStageFailure("budget");
    }
    summary = request.structured
      ? acceptStructuredSummary(after, history, promptID, request.prompt, owned)
      : acceptSummary(after, history, promptID);
  } catch (error) {
    failures.push(caughtSummaryStage(error));
  }
  const unremoved = await removeOwnedSession(host, owned, sessionID);
  if (
    unremoved?.removal === "failed"
    || (unremoved !== undefined && failures.length > 0)
  ) {
    failures.push({
      kind: "stage",
      stage: "delete",
      temporary: "session",
      session: unremoved,
    });
  }
  const failure = failedSummaryStages(failures, "session");
  if (failure) throw failure;
  return { text: summary!, report };
}

// The copied history, then exactly the prompt, one finished summarizer reply
// with text, and the host's idle record. Anything else (a tool call, a
// replayed prompt, an error, a compaction) rejects the summary.
function acceptSummary(
  after: V2Message[],
  history: V2Message[],
  promptID: string,
): string {
  const copied = after.slice(0, history.length);
  const added = after.slice(history.length);
  if (!isDeepStrictEqual(jsonForm(copied), jsonForm(history))) {
    throw sessionSummaryStageFailure("history");
  }
  const [prompt, reply, ...rest] = added;
  if (
    prompt?.type !== "user"
    || prompt.id !== promptID
    || prompt.text !== SESSION_SUMMARY_PROMPT
    || (prompt.files?.length ?? 0) > 0
  ) {
    throw sessionSummaryStageFailure("validation");
  }
  if (reply?.type !== "assistant") {
    throw sessionSummaryStageFailure("provider");
  }
  if (reply.error !== undefined) {
    throw sessionSummaryStageFailure("provider");
  }
  const text = reply.content
    .flatMap(item => (item.type === "text" ? [item.text] : []))
    .join("")
    .trim();
  if (
    reply.agent !== MAGIC_COMPACT_SUMMARIZER_AGENT
    || reply.finish !== "stop"
    || reply.content.some(
      item => item.type !== "text" && item.type !== "reasoning",
    )
    || !text
    || rest.some(
      message => message.type !== "idle" || message.outcome !== "succeeded",
    )
  ) {
    throw sessionSummaryStageFailure("validation");
  }
  return text;
}

// The structured summary: the copied history, the prompt, one summarizer
// reply whose only tool call is one completed summary-tool call, and the
// host's record of this plugin's own stop on the request after it. The
// persisted call input keeps the model's raw arguments, which must be exactly
// one nonempty `summary`; reply text is never read.
function acceptStructuredSummary(
  after: V2Message[],
  history: V2Message[],
  promptID: string,
  promptText: string,
  owned: OwnedSessions,
): string {
  if (
    !isDeepStrictEqual(
      jsonForm(after.slice(0, history.length)),
      jsonForm(history),
    )
  ) {
    throw sessionSummaryStageFailure("history");
  }
  const [prompt, reply, ...rest] = after.slice(history.length);
  if (
    prompt?.type !== "user"
    || prompt.id !== promptID
    || prompt.text !== promptText
    || (prompt.files?.length ?? 0) > 0
  ) {
    throw sessionSummaryStageFailure("validation");
  }
  if (reply?.type !== "assistant" || reply.error !== undefined) {
    throw sessionSummaryStageFailure("provider");
  }
  const calls = reply.content.filter(item => item.type === "tool");
  const call = calls[0];
  const summary =
    call?.state.status === "completed"
      ? structuredSessionSummary(call.state.input)
      : undefined;
  if (
    reply.agent !== MAGIC_COMPACT_SUMMARIZER_AGENT
    || calls.length !== 1
    || call?.name !== V2_SUMMARY_TOOL
    || summary === undefined
    || rest.length !== 1
    || rest[0]!.type !== "idle"
    || rest[0]!.error?.message !== owned.stop
  ) {
    throw sessionSummaryStageFailure("validation");
  }
  return summary;
}

import type { ModelRef } from "@opencode/client";
import { randomUUID } from "node:crypto";
import { isDeepStrictEqual } from "node:util";
import {
  computeNativeBatchBudget,
  NATIVE_BATCH_FIXED_RESERVE_TOKENS,
  type NativeBatchBudget,
} from "../compact/batch";
import {
  MAGIC_COMPACT_SUMMARIZER_AGENT,
  SESSION_SUMMARY_PROMPT,
  SESSION_SUMMARY_RESPONSE_SCHEMA,
  SESSION_SUMMARY_TOOL_PROMPT,
} from "../compact/constants";
import {
  chooseStage,
  matchCandidate,
  sameModel,
  stageReport,
  verifiedReasoning,
  type MatchPreflight,
  type ModelIdentity,
  type Qualifies,
  type ReasoningReports,
  type ReasoningRoute,
  type StageChoice,
} from "../policy/reasoning";
import {
  SessionSummaryError,
  structuredSessionSummary,
  type SessionSummaryStage,
} from "../policy/retention";
import { RETENTION_FAILURE_GUIDANCE } from "../retention/classifier";
import { countGptTokens } from "../stats/tokenize";
import { attemptWithinDeadline, isRecord } from "../util";
import {
  estimateV2HistoryTokens,
  V2_IMAGE_MODEL_REQUIRED,
  type V2SummarizerModel,
} from "./generation";
import { v2ModelVisibleHistory } from "./history";
import type { V2Host } from "./host";
import { v2HistoryShowsImages } from "./media";
import {
  readableReasoningHeaderTokens,
  v2ReasoningRoute,
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
import { hasPluginMessage, PluginMessageError } from "../policy/feedback";

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
// client is still applied later. `absent` means the host reported no such
// session, which is not a cleanup failure, and the guard stays for the same
// reason.
export async function removeOwnedSession(
  host: V2Host,
  owned: OwnedSessions,
  sessionID: string,
): Promise<"removed" | "absent" | "unknown"> {
  owned.close(sessionID);
  try {
    await host.session.remove({ sessionID });
  } catch (error) {
    return isRecord(error) && error["_tag"] === "SessionNotFoundError"
      ? "absent"
      : "unknown";
  }
  owned.release(sessionID);
  return "removed";
}

const DENY_ALL = [{ action: "*", resource: "*", effect: "deny" as const }];

// The operator-configured summarizer model at the source session's location,
// where a project's own agent settings apply, and whether the agent carries
// operator request options (body or headers) written for that model. The
// source model and the native compaction agent are never a fallback.
async function resolveV2SummarizerAgent(
  host: V2Host,
  location: { directory: string },
  controlTimeoutMs: number,
): Promise<{ model: ModelRef; optionsConflict: boolean }> {
  const outcome = await attemptWithinDeadline(
    signal =>
      host.agent.get(
        { agentID: MAGIC_COMPACT_SUMMARIZER_AGENT, location },
        { signal },
      ),
    controlTimeoutMs,
  );
  const agent = outcome.kind === "success" ? outcome.value.data : undefined;
  if (!agent?.model) {
    throw new PluginMessageError(
      `Magic Compact needs a model on the \`${MAGIC_COMPACT_SUMMARIZER_AGENT}\` agent. Set \`agents.${MAGIC_COMPACT_SUMMARIZER_AGENT}.model\` (for example \`provider/model#variant\`) in opencode.json.`,
    );
  }
  const request: Record<string, unknown> = isRecord(agent.request)
    ? agent.request
    : {};
  const filled = (value: unknown) =>
    isRecord(value) && Object.keys(value).length > 0;
  return {
    model: agent.model,
    optionsConflict: filled(request["body"]) || filled(request["headers"]),
  };
}

// The summarizer model and the descriptor the host reports for it, resolved
// once per command, with the other listed models `--match-model` may choose.
// A model the host does not list fails the command.
export async function resolveV2SummarizerModel(
  host: V2Host,
  source: V2Source,
  controlTimeoutMs: number,
): Promise<V2SummarizerModel> {
  const location = { directory: source.session.location.directory };
  const { model: ref, optionsConflict } = await resolveV2SummarizerAgent(
    host,
    location,
    controlTimeoutMs,
  );
  const outcome = await attemptWithinDeadline(
    signal => host.model.list({ location }, { signal }),
    controlTimeoutMs,
  );
  const models = outcome.kind === "success" ? outcome.value.data : [];
  const info = models.find(
    model => model.providerID === ref.providerID && model.id === ref.id,
  );
  if (!info) {
    throw new PluginMessageError(
      "Magic Compact could not resolve the configured compaction model descriptor.",
    );
  }
  return { ref, info, others: models, optionsConflict };
}

function listedModel(summarizer: V2SummarizerModel, model: ModelIdentity) {
  return summarizer.others?.find(
    item => item.providerID === model.providerID && item.id === model.modelID,
  );
}

// The model a summary request runs on in place of the configured one, with
// the configured variant when the choice keeps it.
export function matchedV2Model(
  summarizer: V2SummarizerModel,
  model: ModelIdentity,
  variant: string | undefined,
): V2SummarizerModel {
  const info = listedModel(summarizer, model);
  if (!info) throw new Error("Matched model is not listed.");
  return {
    ...summarizer,
    ref: {
      providerID: info.providerID,
      id: info.id,
      ...(variant === undefined ? {} : { variant }),
    },
    info,
  };
}

// The route facts of a summary model, configured or matched.
export function v2SummarizerRoute(
  summarizer: V2SummarizerModel,
): ReasoningRoute {
  return v2ReasoningRoute(
    { providerID: summarizer.ref.providerID, modelID: summarizer.ref.id },
    summarizer.info,
  );
}

// Whether a listed model's protected reasoning behavior is verified.
export function v2Qualifies(summarizer: V2SummarizerModel): Qualifies {
  return model => {
    const info = listedModel(summarizer, model);
    return (
      info !== undefined
      && verifiedReasoning(v2ReasoningRoute(model, info)) !== undefined
    );
  };
}

// What `--match-model` knows about a listed model before choosing it.
export function v2MatchPreflight(
  summarizer: V2SummarizerModel,
  model: ModelIdentity,
  requiresImages: boolean,
): MatchPreflight {
  const info = listedModel(summarizer, model);
  return {
    variants: new Set((info?.variants ?? []).map(variant => variant.id)),
    optionsConflict: summarizer.optionsConflict === true,
    imagesBlocked:
      requiresImages
      && !(info?.capabilities.input ?? []).some(input =>
        input.startsWith("image"),
      ),
  };
}

// The configured summarizer as the reasoning policy names it.
export function v2Configured(summarizer: V2SummarizerModel): {
  model: ModelIdentity;
  variant?: string;
} {
  return {
    model: {
      providerID: summarizer.ref.providerID,
      modelID: summarizer.ref.id,
    },
    ...(summarizer.ref.variant === undefined
      ? {}
      : { variant: summarizer.ref.variant }),
  };
}

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
  reasoning?: ReasoningReports,
): Promise<string> {
  let visible: V2Message[];
  let requiresImages: boolean;
  try {
    visible = v2ModelVisibleHistory(source.messages);
    requiresImages = v2HistoryShowsImages(visible);
  } catch {
    throw new SessionSummaryError("history");
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
  const fits = (candidate: V2SummaryRequest) => {
    try {
      return (
        summaryRequestTokens(candidate, visible)
        <= computeNativeBatchBudget(candidate.summarizer.info.limit)
          .hardSingleTurnInput
      );
    } catch (error) {
      if (hasPluginMessage(error)) return false;
      throw error;
    }
  };
  const qualifies = v2Qualifies(summarizer);
  const match = matchCandidate(units, qualifies);
  let choice: StageChoice = chooseStage(
    match,
    v2Configured(summarizer),
    model => v2MatchPreflight(summarizer, model, requiresImages),
  );
  let request = summaryRequest(summarizer);
  if (choice.kind === "match") {
    const matched = summaryRequest(
      matchedV2Model(summarizer, choice.model, choice.variant),
    );
    if (!fits(matched)) {
      choice = { kind: "fallback", reason: "capacity" };
    } else if (reasoning?.matchModel) {
      request = matched;
    }
  }
  const model = request.summarizer.ref;
  let budget: NativeBatchBudget;
  try {
    if (
      requiresImages
      && !request.summarizer.info.capabilities.input.includes("image")
    ) {
      throw new PluginMessageError(V2_IMAGE_MODEL_REQUIRED);
    }
    budget = computeNativeBatchBudget(request.summarizer.info.limit);
  } catch (error) {
    // Only this plugin's own configuration messages, which are fixed text
    // without session content, are repeated.
    throw new PluginMessageError(
      `${hasPluginMessage(error) ? error.message : "Magic Compact could not resolve its summarizer model."} ${RETENTION_FAILURE_GUIDANCE}`,
      { cause: error },
    );
  }
  if (summaryRequestTokens(request, visible) > budget.hardSingleTurnInput) {
    throw new SessionSummaryError("budget");
  }
  // OpenCode 2 replay cost is never estimated, so the request's own
  // protected units are reported as left out of its input estimate.
  const requestModel = { providerID: model.providerID, modelID: model.id };
  reasoning?.reports.push(
    stageReport({
      stage: "session",
      candidate: match,
      choice,
      model: requestModel,
      units: v2ReasoningUnits(visible),
      qualifies,
      replayUnknown: v2ReplayEstimate(visible, requestModel).unknown,
    }),
  );
  const control = async <T>(
    stage: SessionSummaryStage,
    request: (signal: AbortSignal) => Promise<T>,
  ): Promise<T> => {
    const outcome = await attemptWithinDeadline(request, controlTimeoutMs);
    if (outcome.kind !== "success") throw new SessionSummaryError(stage);
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
  const failures: SessionSummaryStage[] = [];
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
      throw new SessionSummaryError("fork");
    }
    const read = <T>(request: (signal: AbortSignal) => Promise<T>) =>
      control("history", request);
    if (
      !isDeepStrictEqual(
        json(await readV2History(host, sessionID, read)),
        json(history),
      )
    ) {
      throw new SessionSummaryError("history");
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
      throw new SessionSummaryError("provider");
    }
    const after = await readV2History(host, sessionID, read);
    if (
      after.some(
        message =>
          (message.type === "idle" || message.type === "assistant")
          && message.error?.message.includes(owned.compactionBlocked),
      )
    ) {
      throw new SessionSummaryError("budget");
    }
    summary = request.structured
      ? acceptStructuredSummary(after, history, promptID, request.prompt, owned)
      : acceptSummary(after, history, promptID);
  } catch (error) {
    failures.push(
      error instanceof SessionSummaryError ? error.stage : "validation",
    );
  }
  if ((await removeOwnedSession(host, owned, sessionID)) === "unknown") {
    failures.push("delete");
  }
  if (failures.length === 1) throw new SessionSummaryError(failures[0]!);
  if (failures.length > 1) {
    throw new PluginMessageError(
      `Task-aware retention session summary and cleanup failed (${failures.join(", ")}). ${RETENTION_FAILURE_GUIDANCE}`,
    );
  }
  return summary!;
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
  if (!isDeepStrictEqual(json(copied), json(history))) {
    throw new SessionSummaryError("history");
  }
  const [prompt, reply, ...rest] = added;
  if (
    prompt?.type !== "user"
    || prompt.id !== promptID
    || prompt.text !== SESSION_SUMMARY_PROMPT
    || (prompt.files?.length ?? 0) > 0
  ) {
    throw new SessionSummaryError("validation");
  }
  if (reply?.type !== "assistant") {
    throw new SessionSummaryError("provider");
  }
  if (reply.error !== undefined) {
    throw new SessionSummaryError("provider");
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
    throw new SessionSummaryError("validation");
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
  if (!isDeepStrictEqual(json(after.slice(0, history.length)), json(history))) {
    throw new SessionSummaryError("history");
  }
  const [prompt, reply, ...rest] = after.slice(history.length);
  if (
    prompt?.type !== "user"
    || prompt.id !== promptID
    || prompt.text !== promptText
    || (prompt.files?.length ?? 0) > 0
  ) {
    throw new SessionSummaryError("validation");
  }
  if (reply?.type !== "assistant" || reply.error !== undefined) {
    throw new SessionSummaryError("provider");
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
    throw new SessionSummaryError("validation");
  }
  return summary;
}

function json(value: unknown): unknown {
  return JSON.parse(JSON.stringify(value));
}

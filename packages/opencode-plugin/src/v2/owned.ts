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
} from "../compact/constants";
import {
  SessionSummaryError,
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
  mintV2ID,
  readV2History,
  withFreshMessageIDs,
  type V2Message,
  type V2Source,
} from "./session";

// `generation` sessions run one per-turn summary request and may call the
// response tool; `retention` sessions run the whole-session summary and may
// call no tool.
export type OwnedSessionRole = "generation" | "retention";

type OwnedEntry = {
  role: OwnedSessionRole;
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

  register(sessionID: string, role: OwnedSessionRole): void {
    this.#sessions.set(sessionID, { role, requests: 0, closed: false });
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
// where a project's own agent settings apply. The source model and the native
// compaction agent are never a fallback.
export async function resolveV2Summarizer(
  host: V2Host,
  location: { directory: string },
  controlTimeoutMs: number,
): Promise<ModelRef> {
  const outcome = await attemptWithinDeadline(
    signal =>
      host.agent.get(
        { agentID: MAGIC_COMPACT_SUMMARIZER_AGENT, location },
        { signal },
      ),
    controlTimeoutMs,
  );
  const model =
    outcome.kind === "success" ? outcome.value.data.model : undefined;
  if (!model) {
    throw new Error(
      `Magic Compact needs a model on the \`${MAGIC_COMPACT_SUMMARIZER_AGENT}\` agent. Set \`agents.${MAGIC_COMPACT_SUMMARIZER_AGENT}.model\` (for example \`provider/model#variant\`) in opencode.json.`,
    );
  }
  return model;
}

// The summarizer model and the descriptor the host reports for it, resolved
// once per command. A model the host does not list fails the command.
export async function resolveV2SummarizerModel(
  host: V2Host,
  source: V2Source,
  controlTimeoutMs: number,
): Promise<V2SummarizerModel> {
  const location = { directory: source.session.location.directory };
  const ref = await resolveV2Summarizer(host, location, controlTimeoutMs);
  const outcome = await attemptWithinDeadline(
    signal => host.model.list({ location }, { signal }),
    controlTimeoutMs,
  );
  const info =
    outcome.kind === "success"
      ? outcome.value.data.find(
          model => model.providerID === ref.providerID && model.id === ref.id,
        )
      : undefined;
  if (!info) {
    throw new Error(
      "Magic Compact could not resolve the configured compaction model descriptor.",
    );
  }
  return { ref, info };
}

// The temporary whole-session summary for task-aware retention: a deny-all
// copy of the source under fresh IDs, prompted once through the summarizer
// agent and model, accepted only as that prompt plus one finished text reply,
// and always removed. It never touches the source. Before anything is
// created, the history OpenCode will send (see `v2ModelVisibleHistory`), the
// prompt and the fixed host reserve must fit the model's hard input limit,
// and a history that still shows an image needs an image-capable model.
export async function generateV2SessionSummary(
  host: V2Host,
  owned: OwnedSessions,
  source: V2Source,
  summarizer: V2SummarizerModel,
  controlTimeoutMs: number,
): Promise<string> {
  const model = summarizer.ref;
  let visible: V2Message[];
  let requiresImages: boolean;
  try {
    visible = v2ModelVisibleHistory(source.messages);
    requiresImages = v2HistoryShowsImages(visible);
  } catch {
    throw new SessionSummaryError("history");
  }
  let budget: NativeBatchBudget;
  try {
    if (
      requiresImages
      && !summarizer.info.capabilities.input.includes("image")
    ) {
      throw new Error(V2_IMAGE_MODEL_REQUIRED);
    }
    budget = computeNativeBatchBudget(summarizer.info.limit);
  } catch (error) {
    // These configuration messages are fixed text without session content.
    throw new Error(
      `${error instanceof Error ? error.message : "Magic Compact could not resolve its summarizer model."} ${RETENTION_FAILURE_GUIDANCE}`,
      { cause: error },
    );
  }
  if (
    NATIVE_BATCH_FIXED_RESERVE_TOKENS
      + countGptTokens(SESSION_SUMMARY_PROMPT)
      + estimateV2HistoryTokens(visible).tokens
    > budget.hardSingleTurnInput
  ) {
    throw new SessionSummaryError("budget");
  }
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
    permissions: DENY_ALL,
    metadata: {},
    time: { created: now, updated: now },
  } as Record<string, unknown>;
  delete info["parentID"];
  delete info["fork"];
  delete info["revert"];

  owned.register(sessionID, "retention");
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
        text: SESSION_SUMMARY_PROMPT,
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
    summary = acceptSummary(after, history, promptID);
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
    throw new Error(
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

function json(value: unknown): unknown {
  return JSON.parse(JSON.stringify(value));
}

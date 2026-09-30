// Task-aware retention for /magic-trim and /magic-compact. The existing tool
// rules still decide which calls would change; this step only vetoes changes
// to calls Jev judges likely to help the current task. Decisions are made per
// invocation and never persisted, so kept calls are reassessed next time.
import type { Session } from "@opencode-ai/sdk/v2";
import type { V2Client } from "../api";
import { buildTurns, type MessageWithParts, type Turn } from "../compact/plan";
import { trimWouldChangeContent } from "../compact/prune";
import {
  classifyRetention,
  RetentionClassificationError,
  type RetentionCandidate,
  type RetentionFetch,
} from "./classifier";
import { generateSessionSummary } from "./summary";

export type RetentionPolicy =
  | { mode: "heuristic" }
  | {
      mode: "task-aware";
      // Supplied by the plugin runtime from TYPESAFE_API_KEY.
      apiKey: string | undefined;
      fetch?: RetentionFetch;
    };

export type RetentionPlan = {
  mode: RetentionPolicy["mode"];
  evaluated: number;
  // Tool part IDs kept whole for this invocation.
  keep: ReadonlySet<string>;
};

export const HEURISTIC_RETENTION: RetentionPlan = {
  mode: "heuristic",
  evaluated: 0,
  keep: new Set(),
};

// Completed tool parts in `eligibleTurns` whose content the existing rules
// would change. Parts whose trim would only stamp metadata are not candidates.
export async function findRetentionCandidates(
  messages: MessageWithParts[],
  eligibleTurns: Turn[],
): Promise<{ latestUserRequest: string; candidates: RetentionCandidate[] }> {
  const requests = requestIndex(messages);
  const candidates: RetentionCandidate[] = [];
  for (const turn of eligibleTurns) {
    for (const assistant of turn.assistants) {
      for (const part of assistant.parts) {
        if (part.type !== "tool" || part.state.status !== "completed") {
          continue;
        }
        if (!(await trimWouldChangeContent(part))) continue;
        const request = requests.byPartID.get(part.id);
        if (!request) {
          throw new Error(
            "Retention candidate is missing from source history.",
          );
        }
        candidates.push({
          id: part.id,
          tool: part.tool,
          input: part.state.input,
          output: part.state.output,
          turnRequest: request.text,
          requestsBeforeLatest: Math.max(0, requests.latest - request.index),
        });
      }
    }
  }
  return { latestUserRequest: requests.latestText, candidates };
}

// Summarizes the session and classifies every candidate. Any failure throws
// before the caller creates a backup or writes, and no partial decision set
// is ever returned.
export async function decideRetention(
  v2: V2Client,
  session: Session,
  messages: MessageWithParts[],
  found: { latestUserRequest: string; candidates: RetentionCandidate[] },
  policy: Extract<RetentionPolicy, { mode: "task-aware" }>,
  options: { controlTimeoutMs: number },
): Promise<RetentionPlan> {
  if (found.candidates.length === 0) {
    return { mode: "task-aware", evaluated: 0, keep: new Set() };
  }
  if (!policy.apiKey) {
    throw new RetentionClassificationError("missing-api-key");
  }
  const sessionSummary = await generateSessionSummary(v2, session, messages, {
    controlTimeoutMs: options.controlTimeoutMs,
  });
  const { decisions } = await classifyRetention(
    { latestUserRequest: found.latestUserRequest, sessionSummary },
    found.candidates,
    {
      apiKey: policy.apiKey,
      ...(policy.fetch ? { fetch: policy.fetch } : {}),
    },
  );
  const keep = new Set<string>();
  for (const candidate of found.candidates) {
    const decision = decisions.get(candidate.id);
    if (!decision) throw new RetentionClassificationError("invalid-response");
    if (decision.keep) keep.add(candidate.id);
  }
  return { mode: "task-aware", evaluated: found.candidates.length, keep };
}

export function retentionFeedback(plan: RetentionPlan): string {
  if (plan.mode === "heuristic") {
    return "Heuristic mode: no task-aware retention.";
  }
  if (plan.evaluated === 0) {
    return "Task-aware retention: no tool calls needed evaluation.";
  }
  return `Task-aware retention kept ${plan.keep.size} of ${plan.evaluated} evaluated tool call(s) whole.`;
}

// A request is one turn with ordinary user text plus the following turns
// whose user messages are only synthetic or ignored host content, matching
// the retention evaluation dataset.
function requestIndex(messages: MessageWithParts[]) {
  const byPartID = new Map<string, { text: string; index: number }>();
  let index = -1;
  let text = "";
  for (const turn of buildTurns(messages)) {
    const turnText = ordinaryUserText(turn);
    if (turnText) {
      index += 1;
      text = turnText;
    }
    for (const assistant of turn.assistants) {
      for (const part of assistant.parts) {
        byPartID.set(part.id, { text, index });
      }
    }
  }
  return { byPartID, latest: index, latestText: text };
}

function ordinaryUserText(turn: Turn): string {
  return turn.user
    .flatMap(message => message.parts)
    .flatMap(part =>
      part.type === "text" && part.synthetic !== true && part.ignored !== true
        ? [part.text]
        : [],
    )
    .join("\n")
    .trim();
}

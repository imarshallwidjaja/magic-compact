// V1 host adapter for the shared task-aware retention policy.
import type { Session } from "@opencode-ai/sdk/v2";
import type { V2Client } from "../api";
import type { SummaryReasoningOptions } from "../compact/compact";
import { buildTurns, type MessageWithParts, type Turn } from "../compact/plan";
import { toolCallView } from "../compact/prune";
import {
  collectRetentionCandidates,
  decideRetentionWith,
  formatRetentionFeedback,
  type RetentionCandidates,
  type RetentionCall,
  type RetentionPlan,
  type RetentionPolicy,
} from "../policy/retention";
import { generateSessionSummary } from "./summary";

export {
  HEURISTIC_RETENTION,
  retentionWithoutCandidates,
  type RetentionPlan,
  type RetentionPolicy,
} from "../policy/retention";

export async function findRetentionCandidates(
  messages: MessageWithParts[],
  eligibleTurns: Turn[],
): Promise<RetentionCandidates> {
  const eligible = new Set(
    eligibleTurns.flatMap(turn => turnCalls(turn).map(call => call.ref)),
  );
  return collectRetentionCandidates(
    buildTurns(messages).map(turn => ({
      request: ordinaryUserText(turn),
      calls: turnCalls(turn),
    })),
    eligible,
  );
}

export async function decideRetention(
  v2: V2Client,
  session: Session,
  messages: MessageWithParts[],
  found: RetentionCandidates,
  policy: Extract<RetentionPolicy, { mode: "task-aware" }>,
  options: { controlTimeoutMs: number; reasoning?: SummaryReasoningOptions },
): Promise<RetentionPlan> {
  return decideRetentionWith(found, policy, () =>
    generateSessionSummary(v2, session, messages, {
      controlTimeoutMs: options.controlTimeoutMs,
      ...(options.reasoning ? { reasoning: options.reasoning } : {}),
    }),
  );
}

export function retentionFeedback(
  plan: RetentionPlan,
  messages: readonly MessageWithParts[] = [],
): string {
  return formatRetentionFeedback(
    plan,
    messages.flatMap(message =>
      message.parts.flatMap(part =>
        part.type === "tool" ? [{ ref: part.id, tool: part.tool }] : [],
      ),
    ),
  );
}

function turnCalls(turn: Turn): RetentionCall[] {
  return turn.assistants.flatMap(assistant =>
    assistant.parts.flatMap(part => {
      if (part.type !== "tool") return [];
      const view = toolCallView(part);
      return view ? [{ ref: part.id, tool: part.tool, view }] : [];
    }),
  );
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

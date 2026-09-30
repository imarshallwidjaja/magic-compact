// Fake OpenCode host pieces for the task-aware retention summary fork and a
// fake Jev endpoint, shared by the retention, trim, and compact suites.
import type { Message, Part } from "@opencode-ai/sdk/v2";
import type { MessageWithParts } from "../src/compact/plan";
import {
  RETENTION_MODEL,
  type RetentionFetch,
} from "../src/retention/classifier";

export const SESSION_SUMMARY_TEXT =
  "## Objective\n- Continue the current parser work.";

export function compactionAgent() {
  return {
    name: "compaction",
    native: true,
    hidden: true,
    model: { providerID: "summary-provider", modelID: "summary-model" },
    variant: "low",
  };
}

export function providerCatalog(
  limit: { context: number; input?: number; output?: number } = {
    context: 200_000,
    output: 32_000,
  },
  image = true,
) {
  return {
    all: [
      {
        id: "summary-provider",
        models: {
          "summary-model": { limit, capabilities: { input: { image } } },
        },
      },
    ],
    connected: ["summary-provider"],
    default: {},
  };
}

// OpenCode forks copy history under new message and part IDs.
export function rewriteFork(
  messages: MessageWithParts[],
  forkID: string,
): MessageWithParts[] {
  let partIndex = 0;
  return structuredClone(messages).map((item, index) => {
    const id = `${forkID}-message-${index + 1}`;
    return {
      info: { ...item.info, id, sessionID: forkID } as Message,
      parts: item.parts.map(part => ({
        ...part,
        id: `${forkID}-part-${++partIndex}`,
        sessionID: forkID,
        messageID: id,
      })) as Part[],
    };
  });
}

// Appends the prompt and an ordinary compaction-agent reply, as OpenCode does
// for a successful summary request, and returns the prompt response body.
export function summaryReply(
  fork: MessageWithParts[],
  request: { messageID?: string; agent?: string; parts?: Part[] },
  forkID: string,
  text = SESSION_SUMMARY_TEXT,
) {
  const userID = request.messageID ?? `${forkID}-prompt`;
  fork.push({
    info: {
      id: userID,
      sessionID: forkID,
      role: "user",
      agent: request.agent,
    } as Message,
    parts: (request.parts ?? []).map((part, index) => ({
      ...part,
      id: `${userID}-part-${index}`,
      sessionID: forkID,
      messageID: userID,
    })) as Part[],
  });
  const replyID = `${forkID}-reply`;
  const info = {
    id: replyID,
    sessionID: forkID,
    role: "assistant",
    parentID: userID,
    agent: "compaction",
    mode: "compaction",
    finish: "stop",
  } as Message;
  const parts = [
    { id: `${replyID}-start`, type: "step-start" },
    { id: `${replyID}-text`, type: "text", text },
    { id: `${replyID}-finish`, type: "step-finish" },
  ].map(part => ({ ...part, sessionID: forkID, messageID: replyID })) as Part[];
  fork.push({ info, parts });
  return { data: { info, parts } };
}

export type JevCall = {
  task: { latestUserRequest: string; sessionSummary: string };
  candidate: { tool: string; calledWhile: string; output: string };
};

// Scores each single-candidate request by its excerpted output text.
export function scoredJevFetch(
  score: (output: string) => number,
  calls: JevCall[] = [],
): RetentionFetch {
  return async (_url, init) => {
    const body = JSON.parse(String(init.body)) as {
      state: { task: JevCall["task"]; candidates: Array<JevCall["candidate"]> };
    };
    const output = body.state.candidates[0]!.output;
    calls.push({ task: body.state.task, candidate: body.state.candidates[0]! });
    return Response.json({
      model: RETENTION_MODEL,
      answers: { candidate_0: { type: "noul", noul: score(output) } },
      usage: { input_tokens: 10 },
    });
  };
}

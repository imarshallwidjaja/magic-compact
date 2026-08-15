import type { Message, Part, TextPart } from "@opencode-ai/sdk/v2";
import type { MessageWithParts } from "../../src/compact/plan";

export const INCIDENT_TURN_COUNT = 17;
export const INCIDENT_DETERMINISTIC_TURN_COUNT = 6;
export const INCIDENT_STANDARD_TURN_COUNT = 11;
export const INCIDENT_TOOL_COUNT = 497;
export const INCIDENT_ERROR_TOOL_COUNT = 28;
export const INCIDENT_RUNNING_TOOL_COUNT = 1;
export const INCIDENT_ASSISTANT_REPEAT_LITERAL = " historical context";
export const INCIDENT_ASSISTANT_REPEAT_COUNT = 9_000;

const TOOLS_PER_TURN = Array.from(
  { length: INCIDENT_TURN_COUNT },
  (_, index) => 29 + (index < 4 ? 1 : 0),
);
const ERRORS_PER_DETERMINISTIC_TURN = [5, 5, 5, 5, 4, 4] as const;

export function incidentSessionFixture(
  assistantRepeatCount = INCIDENT_ASSISTANT_REPEAT_COUNT,
): MessageWithParts[] {
  const messages: MessageWithParts[] = [];

  for (let turnIndex = 0; turnIndex < INCIDENT_TURN_COUNT; turnIndex++) {
    const userID = `incident-user-${turnIndex}`;
    const assistantID = `incident-assistant-${turnIndex}`;
    messages.push(
      message(userID, "user", [
        textPart(
          `incident-user-text-${turnIndex}`,
          userID,
          `Synthetic incident request ${turnIndex}.`,
        ),
      ]),
    );

    const parts: Part[] = [];
    const errorCount = ERRORS_PER_DETERMINISTIC_TURN[turnIndex] ?? 0;
    const hasRunning = turnIndex === INCIDENT_DETERMINISTIC_TURN_COUNT - 1;
    for (
      let toolIndex = 0;
      toolIndex < TOOLS_PER_TURN[turnIndex]!;
      toolIndex++
    ) {
      const id = `incident-tool-${turnIndex}-${toolIndex}`;
      if (toolIndex < errorCount) {
        parts.push(
          toolPart(id, assistantID, "bash", {
            status: "error",
            input: {
              command: `synthetic-command --turn ${turnIndex} --case ${toolIndex}`,
            },
            error: `Synthetic command failed with exit code 1 for case ${turnIndex}-${toolIndex}.`,
            metadata: {},
            time: { start: 1, end: 2 },
          }),
        );
        continue;
      }
      if (hasRunning && toolIndex === errorCount) {
        parts.push(
          toolPart(id, assistantID, "bash", {
            status: "running",
            input: { command: `synthetic-command --turn ${turnIndex} --watch` },
            title: "Synthetic running command",
            metadata: {},
            time: { start: 1 },
          }),
        );
        continue;
      }
      parts.push(
        toolPart(id, assistantID, "read", {
          status: "completed",
          input: {
            filePath: `synthetic/turn-${turnIndex}/file-${toolIndex}.txt`,
          },
          output: `Synthetic output ${turnIndex}-${toolIndex}. ${"tool payload ".repeat(20)}`,
          title: "Synthetic read",
          metadata: {},
          time: { start: 1, end: 2 },
        }),
      );
    }

    const status =
      turnIndex < INCIDENT_DETERMINISTIC_TURN_COUNT
        ? "Synthetic partial delivery with exact terminal state below."
        : "Synthetic completed delivery with all tool calls complete.";
    parts.push(
      textPart(
        `incident-assistant-text-${turnIndex}`,
        assistantID,
        `${status}${INCIDENT_ASSISTANT_REPEAT_LITERAL.repeat(assistantRepeatCount)}`,
      ),
    );
    messages.push(message(assistantID, "assistant", parts));
  }

  return messages;
}

function message(
  id: string,
  role: "user" | "assistant",
  parts: Part[],
): MessageWithParts {
  return { info: { id, role } as Message, parts };
}

function textPart(id: string, messageID: string, text: string): TextPart {
  return {
    id,
    sessionID: "incident-source",
    messageID,
    type: "text",
    text,
  };
}

function toolPart(
  id: string,
  messageID: string,
  tool: string,
  state: Record<string, unknown>,
): Part {
  return {
    id,
    sessionID: "incident-source",
    messageID,
    type: "tool",
    callID: `${id}-call`,
    tool,
    state,
  } as Part;
}

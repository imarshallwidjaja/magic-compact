import type { TextPart } from "@opencode-ai/sdk/v2";
import { type Turn } from "./plan";

export function buildCompactionPrompt(
  turns: Turn[],
  nextTurn: Turn | null,
): string {
  return `<system>
# Attention: Conversation Compaction Required

Summarize only the selected historical assistant turns below. Historical transcript evidence is authoritative. Current inspection may clarify that evidence, but do not continue unfinished work.

Return only one JSON object whose keys are exactly the selected assistant turn IDs and whose values are non-empty summary strings. Required output keys: ${JSON.stringify(turns.map(turnID))}
Preserve the IDs exactly. Do not return Markdown fences, prose, XML, or a nested wrapper.

Selected historical turns and stop anchor:
${buildPromptSelection(turns, nextTurn)}

Each summary must preserve all evidenced continuation state, prioritizing completeness over brevity:
- outcome: completed, partial, blocked, or analysis-only
- delivered implementation or other concrete changes
- verification commands and results, plus checks explicitly not run
- version-control, deployment, branch, source, and backup state
- errors encountered and their fixes
- unresolved work and the exact next action
- exact identifiers such as paths, commits, branches, sessions, deployments, and commands when evidenced

Do not emit placeholders, unknown IDs, user text, or a summary for the stop anchor. Tools may be used only when needed to recover missing historical evidence or clarify it through current inspection. Must not continue unfinished work. After any tool use, return only the required JSON object.
</system>`;
}

export function turnID(turn: Turn): string {
  const firstAssistant = turn.assistants[0];
  if (!firstAssistant) throw new Error("Turn missing assistant message ID.");
  return firstAssistant.info.id;
}

function buildPromptSelection(turns: Turn[], nextTurn: Turn | null): string {
  return JSON.stringify(
    {
      turns: turns.map(turn => ({
        assistantTurnID: turnID(turn),
        userPreview: getUserPromptText(turn),
      })),
      stopAnchor: nextTurn
        ? {
            assistantTurnID: turnID(nextTurn),
            userPreview: getUserPromptText(nextTurn),
          }
        : null,
    },
    null,
    2,
  );
}

function getUserPromptText(turn: Turn): string {
  const userText = turn.user
    .flatMap(msg => msg.parts)
    .filter(
      (part): part is TextPart =>
        part.type === "text"
        && part.synthetic !== true
        && part.ignored !== true,
    )
    .map(part => part.text)
    .join("\n");
  return truncateUserText(userText);
}

function truncateUserText(text: string): string {
  const firstLine = stripInvalidXmlCodePoints(
    text.trim().split("\n")[0]?.trim() ?? "",
  );
  const codePoints = [...firstLine];
  if (codePoints.length <= 300) return `${firstLine}\n...`;
  return `${codePoints.slice(0, 300).join("").trim()}...`;
}

export function stripInvalidXmlCodePoints(value: string): string {
  return [...value]
    .filter(character => isXmlCodePoint(character.codePointAt(0)!))
    .join("");
}

export function isXmlCodePoint(value: number): boolean {
  return (
    value === 0x09
    || value === 0x0a
    || value === 0x0d
    || (value >= 0x20 && value <= 0xd7ff)
    || (value >= 0xe000 && value <= 0xfffd)
    || (value >= 0x10000 && value <= 0x10ffff)
  );
}

import type { TextPart } from "@opencode-ai/sdk/v2";
import { type Turn } from "./plan";

export function buildCompactionPrompt(
  turns: Turn[],
  nextTurn: Turn | null,
  evidenceByTurn: ReadonlyMap<Turn, string>,
): string {
  return `<system>
# Attention: Conversation Compaction Required

Summarize only the selected historical assistant turns from the bounded exact evidence map below. The evidence map is authoritative. Do not infer missing history or continue unfinished work. Assistant-text records may use \`textEncoding: "ordered-literal-runs-v1"\` with \`textSegments\` only when a whole source part is provably periodic: an optional leading unique segment with repetitions 1, then one segment whose literal repeated that many times reproduces the exact source. Expand each segment by repeating its literal; any other text stays verbatim in \`text\`.

Return only one JSON object whose keys are exactly the selected assistant turn IDs and whose values are non-empty summary strings. Required output keys: ${JSON.stringify(turns.map(turnID))}
Preserve the IDs exactly. Do not return Markdown fences, prose, XML, or a nested wrapper. The complete response must fit within 4,000 local gpt-tokenizer tokens.

Selected historical turns and stop anchor:
${buildPromptSelection(turns, nextTurn, evidenceByTurn)}

Each summary is a human-readable continuation capsule for one turn. Start with exactly \`Outcome: completed\`, \`Outcome: partial\`, or \`Outcome: analysis-only\` as established by exactEvidence. Follow it with a non-empty \`Work:\` field. Add only non-empty \`Verification:\`, \`VCS:\`, \`Unresolved:\`, and \`Next:\` sections, using \`- \` bullets. Correctness and continuation fidelity come before brevity. Preserve every continuation-critical fact in readable prose or short bullets:
- outcome: completed, partial, or analysis-only
- delivered implementation or other concrete work, and the decisions made
- verification results, plus checks explicitly not run
- version-control, deployment, branch, source, and backup state
- errors encountered and their fixes
- unresolved work and the exact next action
- exact identifiers such as paths, commits, branches, sessions, deployments, and commands, but only when the next agent needs them to continue

Write each capsule without restating tool structure or evidence the transcript already keeps. Never copy machine evidence records, JSON fields such as sourceBytes/windows, record labels, or internal record accounting into a visible capsule. Do not emit placeholders, unknown IDs, user text, or a summary for the stop anchor. No tools or external inspection are available. Preserve failed, not-run, running, and unresolved state exactly as represented. Return only the required JSON object.
</system>`;
}

export function turnID(turn: Turn): string {
  const firstAssistant = turn.assistants[0];
  if (!firstAssistant) throw new Error("Turn missing assistant message ID.");
  return firstAssistant.info.id;
}

function buildPromptSelection(
  turns: Turn[],
  nextTurn: Turn | null,
  evidenceByTurn: ReadonlyMap<Turn, string>,
): string {
  return JSON.stringify(
    {
      turns: turns.map(turn => ({
        assistantTurnID: turnID(turn),
        userPreview: getUserPromptText(turn),
        exactEvidence: evidenceByTurn.get(turn) ?? "",
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

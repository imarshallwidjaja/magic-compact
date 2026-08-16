import type { TextPart } from "@opencode-ai/sdk/v2";
import { unwrap, type V2Client } from "../api";
import { renderCanonicalCapsule } from "./capsule";
import { capsuleSummaryTextPart } from "./constants";
import { type CompactionPlan, type Turn } from "./plan";

export type GeneratedCapsule = string | null;

export function generateCompactionCapsules(
  plan: CompactionPlan,
): GeneratedCapsule[] {
  return plan.summarizedTurns.map(renderCanonicalCapsule);
}

export async function injectSummaries(
  v2: V2Client,
  sessionID: string,
  compactionTurns: Turn[],
  summaries: string[],
): Promise<TextPart[]> {
  const inserted: TextPart[] = [];
  for (const [index, turn] of compactionTurns.entries()) {
    const summary = summaries[index];
    if (summary === undefined)
      throw new Error("Missing summary for assistant turn.");
    const firstAssistant = turn.assistants[0];
    if (!firstAssistant) throw new Error("Turn missing assistant message.");
    const part = capsuleSummaryTextPart(
      firstAssistant.info.id,
      sessionID,
      summary,
    );
    unwrap(
      await v2.part.update({
        sessionID,
        messageID: firstAssistant.info.id,
        partID: part.id,
        part,
      }),
    );
    inserted.push(part);
  }
  return inserted;
}

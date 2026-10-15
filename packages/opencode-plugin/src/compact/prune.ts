import type { Part, ToolPart } from "@opencode-ai/sdk/v2";
import { isPersistedSummaryPart } from "./constants";
import type { MessageWithParts, Turn } from "./plan";
import { planOmission, type PlannedOmission } from "../storage/omission";
import { isRecord } from "../util";
import { coveredAssistantImagePartIDs } from "./media";
import { keepsHostInjectedText } from "../policy/host-text";
import {
  planToolTrim,
  plannedOutputText,
  toolTrimChangesContent,
  type OmissionAllocator,
  type ToolCallView,
  type ToolRuleCategory,
} from "../policy/tool-rules";

export type PruneProjection = {
  turns: Turn[];
  mutations: PruneMutation[];
};

export type PruneMutation =
  | {
      type: "delete-part";
      messageID: string;
      partID: string;
    }
  | {
      type: "update-part";
      messageID: string;
      partID: string;
      replacement: ToolPart;
      omissions: PlannedOmission[];
    }
  | {
      type: "delete-message";
      messageID: string;
    };

// `keep` names tool parts that task-aware retention keeps whole: they stay
// byte-identical, including their trim metadata, and produce no mutation.
export async function createPruneProjection(
  turns: Turn[],
  sessionID: string,
  keep: ReadonlySet<string>,
): Promise<PruneProjection> {
  assertNoNativeArtifacts(turns);
  const projected = structuredClone(turns);
  const mutations: PruneMutation[] = [];

  for (const turn of projected) {
    const coveredAssistantImages = coveredAssistantImagePartIDs(turn);
    for (const user of turn.user) {
      const kept: Part[] = [];
      for (const part of user.parts) {
        if (
          part.type === "text"
          && part.synthetic === true
          && !isImportantSyntheticPart(part)
        ) {
          mutations.push({
            type: "delete-part",
            messageID: user.info.id,
            partID: part.id,
          });
        } else {
          kept.push(part);
        }
      }
      user.parts = kept;
    }
    const assistants: MessageWithParts[] = [];
    for (const assistant of turn.assistants) {
      const kept: Part[] = [];
      const partMutations: PruneMutation[] = [];
      for (const part of assistant.parts) {
        if (isPersistedSummaryPart(part)) {
          kept.push(part);
        } else if (
          part.type === "file"
          && coveredAssistantImages.has(part.id)
        ) {
          kept.push(part);
        } else if (part.type === "tool" && keep.has(part.id)) {
          kept.push(part);
        } else if (part.type === "tool") {
          const partOmissions: PlannedOmission[] = [];
          const changed = await prepareToolPart(part, async entry => {
            const omission = planOmission(sessionID, entry);
            partOmissions.push(omission);
            return omission.contentID;
          });
          kept.push(part);
          if (changed) {
            partMutations.push({
              type: "update-part",
              messageID: assistant.info.id,
              partID: part.id,
              replacement: structuredClone(part),
              omissions: partOmissions,
            });
          }
        } else {
          partMutations.push({
            type: "delete-part",
            messageID: assistant.info.id,
            partID: part.id,
          });
        }
      }
      assistant.parts = kept;
      if (kept.length > 0) {
        assistants.push(assistant);
        mutations.push(...partMutations);
      } else {
        mutations.push({
          type: "delete-message",
          messageID: assistant.info.id,
        });
      }
    }
    turn.assistants = assistants;
  }
  return { turns: projected, mutations };
}

function assertNoNativeArtifacts(turns: Turn[]): void {
  for (const turn of turns) {
    const messages = [...turn.user, ...turn.assistants];
    if (
      messages.some(message =>
        message.parts.some(part => part.type === "compaction"),
      )
      || turn.assistants.some(message => message.info.summary === true)
    ) {
      throw new Error(
        "Refusing to prune a turn containing a native compaction artifact.",
      );
    }
  }
}

export type TrimProjection = {
  // Replacement parts keyed by their source part ID, in source order.
  replacements: ToolPart[];
  omissions: PlannedOmission[];
};

// Applies the existing tool rules to clones of the completed tool parts in
// `turns`, skipping parts that task-aware retention keeps whole. Omission IDs
// are planned, not allocated; the caller commits them to the successor cache.
export async function projectToolTrim(
  turns: Turn[],
  sessionID: string,
  keep: ReadonlySet<string>,
): Promise<TrimProjection> {
  const replacements: ToolPart[] = [];
  const omissions: PlannedOmission[] = [];
  for (const turn of turns) {
    for (const assistant of turn.assistants) {
      for (const part of assistant.parts) {
        if (part.type !== "tool" || keep.has(part.id)) continue;
        const replacement = structuredClone(part);
        const partOmissions: PlannedOmission[] = [];
        const changed = await prepareToolPart(replacement, async entry => {
          const omission = planOmission(sessionID, entry);
          partOmissions.push(omission);
          return omission.contentID;
        });
        if (!changed) continue;
        replacements.push(replacement);
        omissions.push(...partOmissions);
      }
    }
  }
  return { replacements, omissions };
}

// True when the existing tool rules would change a completed part's model
// input or output, not merely stamp its trim metadata. Runs the rules on a
// clone and allocates no omission IDs.
export async function trimWouldChangeContent(part: ToolPart): Promise<boolean> {
  const call = toolCallView(part);
  return call !== null && (await toolTrimChangesContent(call));
}

// V1 tool names for the shared rule categories.
const V1_TOOL_CATEGORIES: Readonly<Record<string, ToolRuleCategory>> = {
  write: "write",
  edit: "edit",
  apply_patch: "patch",
  bash: "shell",
  task: "subagent",
  read: "read",
  todowrite: "todo",
  question: "question",
  skill: "skill",
};

export function toolCallView(part: ToolPart): ToolCallView | null {
  if (part.state.status !== "completed") return null;
  const magicCompact = part.state.metadata["magicCompact"];
  return {
    category: Object.hasOwn(V1_TOOL_CATEGORIES, part.tool)
      ? V1_TOOL_CATEGORIES[part.tool]!
      : "other",
    input: part.state.input,
    output: part.state.output,
    trimmed: isRecord(magicCompact) && magicCompact["trimmed"] === true,
  };
}

async function prepareToolPart(
  part: ToolPart,
  allocate: OmissionAllocator,
): Promise<boolean> {
  const call = toolCallView(part);
  if (call === null || part.state.status !== "completed") return false;
  const plan = await planToolTrim(call, allocate);
  if (plan === null) return false;
  const magicCompact = part.state.metadata["magicCompact"];
  part.state.input = plan.input;
  part.state.output = plannedOutputText(call, plan);
  part.state.metadata = {
    ...part.state.metadata,
    magicCompact: {
      ...(isRecord(magicCompact) ? magicCompact : {}),
      trimmed: true,
    },
  };
  return true;
}

function isImportantSyntheticPart(part: Part): boolean {
  return (
    part.type === "text"
    && part.synthetic === true
    && keepsHostInjectedText(part.text)
  );
}

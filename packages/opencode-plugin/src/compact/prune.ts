import type { Part, ToolPart } from "@opencode-ai/sdk/v2";
import { unwrap, type V2Client } from "../api";
import { isPersistedSummaryPart } from "./constants";
import type { MessageWithParts, Turn } from "./plan";
import {
  allocateOmission,
  commitPlannedOmission,
  planOmission,
  type PlannedOmission,
} from "../storage/omission";
import { coveredAssistantImagePartIDs } from "./media";
import { prepareLegacyToolReplacement } from "../trim/legacy";

type PruneContext = {
  v2: V2Client;
  sessionID: string;
};

export type PruneProjection = {
  turns: Turn[];
  mutations: PruneMutation[];
};

export type PruneMutation =
  | { type: "delete-part"; messageID: string; partID: string }
  | {
      type: "update-part";
      messageID: string;
      partID: string;
      replacement: ToolPart;
      omissions: PlannedOmission[];
    }
  | { type: "delete-message"; messageID: string };

export async function pruneSummarizedTurns(
  context: PruneContext,
  projection: PruneProjection,
): Promise<void> {
  for (const mutation of projection.mutations) {
    if (mutation.type === "delete-part") {
      unwrap(
        await context.v2.part.delete({
          sessionID: context.sessionID,
          messageID: mutation.messageID,
          partID: mutation.partID,
        }),
      );
      continue;
    }
    if (mutation.type === "delete-message") {
      unwrap(
        await context.v2.session.deleteMessage({
          sessionID: context.sessionID,
          messageID: mutation.messageID,
        }),
      );
      continue;
    }
    for (const omission of mutation.omissions) {
      await commitPlannedOmission(context.sessionID, omission);
    }
    unwrap(
      await context.v2.part.update({
        sessionID: context.sessionID,
        messageID: mutation.messageID,
        partID: mutation.partID,
        part: { ...mutation.replacement, sessionID: context.sessionID },
      }),
    );
  }
}

export async function createPruneProjection(
  turns: Turn[],
  sessionID: string,
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
        } else if (part.type === "tool") {
          const partOmissions: PlannedOmission[] = [];
          const replacement = await prepareLegacyToolReplacement(
            part,
            async entry => {
              const omission = planOmission(sessionID, entry);
              partOmissions.push(omission);
              return omission.contentID;
            },
          );
          kept.push(replacement ?? part);
          if (replacement) {
            partMutations.push({
              type: "update-part",
              messageID: assistant.info.id,
              partID: part.id,
              replacement,
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

export async function trimToolParts(
  context: PruneContext,
  turns: Turn[],
): Promise<number> {
  let trimmed = 0;
  for (const turn of turns) {
    for (const assistant of turn.assistants) {
      for (const part of assistant.parts) {
        if (part.type !== "tool") continue;
        const replacement = await prepareLegacyToolReplacement(part, entry =>
          allocateOmission(context.sessionID, entry),
        );
        if (!replacement) continue;
        // Existing manual callers observe their loaded tool part after mutation.
        part.state = replacement.state;
        unwrap(
          await context.v2.part.update({
            sessionID: context.sessionID,
            messageID: part.messageID,
            partID: part.id,
            part: { ...part, sessionID: context.sessionID },
          }),
        );
        trimmed += 1;
      }
    }
  }
  return trimmed;
}

function isImportantSyntheticPart(part: Part): boolean {
  return (
    part.type === "text"
    && part.synthetic === true
    && (part.text === "The following tool was executed by the user"
      || part.text.startsWith("<task ")
      || part.text.includes(
        "The user has changed the current working directory to",
      )
      || part.text.startsWith("The user made the following comment"))
  );
}

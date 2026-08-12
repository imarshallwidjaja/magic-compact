import type { Part, ToolPart } from "@opencode-ai/sdk/v2";
import { unwrap, type V2Client } from "../api";
import { inputOmissionNotice, outputOmissionNotice } from "./constants";
import type { MessageWithParts, Turn } from "./plan";
import {
  allocateOmission,
  commitPlannedOmission,
  planOmission,
  type PlannedOmission,
} from "../storage/omission";
import { isRecord, unwrapString } from "../util";

type PruneContext = {
  v2: V2Client;
  sessionID: string;
};

export type PruneProjection = {
  turns: Turn[];
  omissions: PlannedOmission[];
};

const DEFAULT_OUTPUT_DESCRIPTION =
  "Output omitted due to a compaction operation.";

const DEFAULT_LIMIT = { words: 128, chars: 1024 };
const TASK_OUTPUT_LIMIT = { words: 512, chars: 4096 };

export async function pruneSummarizedTurns(
  context: PruneContext,
  turns: Turn[],
  plannedOmissions: PlannedOmission[],
): Promise<void> {
  assertNoNativeArtifacts(turns);
  let omissionIndex = 0;
  const allocate = async (entry: { content: string }) => {
    const omission = plannedOmissions[omissionIndex++];
    if (!omission || omission.content !== entry.content) {
      throw new Error("Planned omission order changed before pruning.");
    }
    await commitPlannedOmission(context.sessionID, omission);
    return omission.contentID;
  };
  for (const turn of turns) {
    for (const user of turn.user) {
      await pruneUserParts(context, user);
    }

    for (const assistant of turn.assistants) {
      await pruneAssistantParts(context, assistant, allocate);
    }
  }
  if (omissionIndex !== plannedOmissions.length) {
    throw new Error("Not every planned omission was consumed during pruning.");
  }
}

export async function createPruneProjection(
  turns: Turn[],
  sessionID: string,
): Promise<PruneProjection> {
  assertNoNativeArtifacts(turns);
  const projected = structuredClone(turns);
  const omissions: PlannedOmission[] = [];
  const allocate = async (entry: { content: string }) => {
    const omission = planOmission(sessionID, entry);
    omissions.push(omission);
    return omission.contentID;
  };

  for (const turn of projected) {
    for (const user of turn.user) {
      user.parts = user.parts.filter(
        part =>
          part.type !== "text"
          || part.synthetic !== true
          || isImportantSyntheticPart(part),
      );
    }
    const assistants: MessageWithParts[] = [];
    for (const assistant of turn.assistants) {
      const kept: Part[] = [];
      for (const part of assistant.parts) {
        if (isSummaryPart(part)) {
          kept.push(part);
        } else if (part.type === "tool") {
          await prepareToolPart(part, allocate);
          kept.push(part);
        }
      }
      assistant.parts = kept;
      if (kept.length > 0) assistants.push(assistant);
    }
    turn.assistants = assistants;
  }
  return { turns: projected, omissions };
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
        if (
          part.type === "tool"
          && (await pruneToolPart(context, part, entry =>
            allocateOmission(context.sessionID, entry),
          ))
        ) {
          trimmed += 1;
        }
      }
    }
  }
  return trimmed;
}

async function pruneUserParts(
  context: PruneContext,
  message: MessageWithParts,
): Promise<void> {
  for (const part of message.parts) {
    if (part.type !== "text" || part.synthetic !== true) {
      // If not text (attachment) or is text but not synthetic, don't prune
      continue;
    }

    if (isImportantSyntheticPart(part)) {
      // If is an important synthetic part, don't prune
      continue;
    }

    // Prune useless synthetic part
    unwrap(
      await context.v2.part.delete({
        sessionID: context.sessionID,
        messageID: message.info.id,
        partID: part.id,
      }),
    );
  }
}

async function pruneAssistantParts(
  context: PruneContext,
  message: MessageWithParts,
  allocate: (entry: { content: string }) => Promise<string>,
): Promise<void> {
  let keptParts = 0;

  for (const part of message.parts) {
    // If is compaction summary, leave alone
    if (isSummaryPart(part)) {
      keptParts += 1;
      continue;
    }

    // If it is a tool call, prune but keep
    if (part.type === "tool") {
      await pruneToolPart(context, part, allocate);
      keptParts += 1;
      continue;
    }

    // Otherwise, delete part (reasoning / text)
    unwrap(
      await context.v2.part.delete({
        sessionID: context.sessionID,
        messageID: message.info.id,
        partID: part.id,
      }),
    );
  }

  if (keptParts === 0) {
    unwrap(
      await context.v2.session.deleteMessage({
        sessionID: context.sessionID,
        messageID: message.info.id,
      }),
    );
  }
}

async function pruneToolPart(
  context: PruneContext,
  part: ToolPart,
  allocate: (entry: { content: string }) => Promise<string>,
): Promise<boolean> {
  const changed = await prepareToolPart(part, allocate);
  if (!changed) return false;

  unwrap(
    await context.v2.part.update({
      sessionID: context.sessionID,
      messageID: part.messageID,
      partID: part.id,
      part: {
        ...part,
        sessionID: context.sessionID,
      },
    }),
  );
  return true;
}

async function prepareToolPart(
  part: ToolPart,
  allocate: (entry: { content: string }) => Promise<string>,
): Promise<boolean> {
  if (part.state.status !== "completed") {
    return false;
  }
  const magicCompact = part.state.metadata["magicCompact"];
  if (isRecord(magicCompact) && magicCompact["trimmed"] === true) {
    return false;
  }

  const inputNotice = await applyInputOmissions(part, allocate);
  await applyOutputOmissions(part, allocate);

  if (inputNotice) {
    part.state.output = `${inputNotice}\n\n${part.state.output}`;
  }

  part.state.metadata = {
    ...part.state.metadata,
    magicCompact: {
      ...(isRecord(magicCompact) ? magicCompact : {}),
      trimmed: true,
    },
  };

  return true;
}

async function applyInputOmissions(
  part: ToolPart,
  allocate: (entry: { content: string }) => Promise<string>,
): Promise<string | null> {
  if (part.state.status !== "completed") {
    return null;
  }

  const input = part.state.input;

  if (part.tool === "write") {
    const content = unwrapString(input["content"]);
    if (content && exceeds(content, DEFAULT_LIMIT)) {
      const contentID = await allocate({ content });
      input["content"] = "[Omitted]";
      return inputOmissionNotice(
        "File write contents omitted due to a compaction operation. If necessary, reread file to see current contents.",
        Buffer.byteLength(content),
        contentID,
      );
    }
  }

  if (part.tool === "apply_patch") {
    const content = unwrapString(input["patchText"]);
    if (content && exceeds(content, DEFAULT_LIMIT)) {
      const contentID = await allocate({ content });
      input["patchText"] = "[Omitted]";
      return inputOmissionNotice(
        "Patch text omitted due to compaction operation. If necessary, reread files to see current contents.",
        Buffer.byteLength(content),
        contentID,
      );
    }
  }

  if (part.tool === "bash") {
    const content = unwrapString(input["command"]);
    if (content && content.length > 1024) {
      const contentID = await allocate({ content });
      input["command"] =
        `${content.slice(0, 512)}\n[REST OF COMMAND TRUNCATED]`;
      return inputOmissionNotice(
        "Bash command truncated due to compaction operation.",
        Buffer.byteLength(content),
        contentID,
      );
    }
  }

  if (part.tool === "edit") {
    const oldString = unwrapString(input["oldString"]);
    const newString = unwrapString(input["newString"]);
    const combined = `${oldString}\n${newString}`;
    if (exceeds(combined, DEFAULT_LIMIT)) {
      const contentID = await allocate({
        content: combined,
      });
      input["oldString"] = "[Omitted]";
      input["newString"] = "[Omitted]";
      return inputOmissionNotice(
        "File edit oldString and newString omitted due to compaction operation. If necessary, reread file to see current contents.",
        Buffer.byteLength(combined),
        contentID,
      );
    }
  }

  return null;
}

async function applyOutputOmissions(
  part: ToolPart,
  allocate: (entry: { content: string }) => Promise<string>,
): Promise<void> {
  if (part.state.status !== "completed") {
    return;
  }

  const output = part.state.output;

  if (part.tool === "question") {
    return;
  }

  if (part.tool === "todowrite") {
    part.state.output = "Successfully updated todos.";
    return;
  }

  if (part.tool === "skill") {
    part.state.output =
      "Skill contents omitted due to compaction operation. If necessary, recall skill.";
    return;
  }

  if (part.tool === "read") {
    const contentID = await allocate({
      content: output,
    });
    part.state.output = outputOmissionNotice(
      "Stale read contents omitted due to compaction operation. If necessary, reread to see current contents.",
      Buffer.byteLength(output),
      contentID,
    );
    return;
  }

  if (part.tool === "task") {
    if (exceeds(output, TASK_OUTPUT_LIMIT)) {
      const contentID = await allocate({
        content: output,
      });
      part.state.output = outputOmissionNotice(
        "Task output omitted due to a compaction operation. If necessary, reread output via read_omitted_content tool.",
        Buffer.byteLength(output),
        contentID,
      );
    }
    return;
  }

  if (!exceeds(output, DEFAULT_LIMIT)) {
    return;
  }

  const contentID = await allocate({
    content: output,
  });
  part.state.output = outputOmissionNotice(
    DEFAULT_OUTPUT_DESCRIPTION,
    Buffer.byteLength(output),
    contentID,
  );
}

function isSummaryPart(part: Part): boolean {
  if (part.type !== "text") {
    return false;
  }

  const metadata = part.metadata;
  if (!isRecord(metadata)) {
    return false;
  }

  const magicCompact = metadata["magicCompact"];
  return isRecord(magicCompact) && magicCompact["summary"] === true;
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

function exceeds(
  text: string,
  limit: { words: number; chars: number },
): boolean {
  return wordCount(text) > limit.words || text.length > limit.chars;
}

function wordCount(text: string): number {
  const trimmed = text.trim();
  if (!trimmed) {
    return 0;
  }

  return trimmed.split(/\s+/).length;
}

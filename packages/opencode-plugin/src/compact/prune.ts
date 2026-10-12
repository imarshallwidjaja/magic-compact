import type { Part, ToolPart } from "@opencode-ai/sdk/v2";
import { isDeepStrictEqual } from "node:util";
import { unwrap, type V2Client } from "../api";
import {
  inputOmissionNotice,
  isPersistedSummaryPart,
  outputOmissionNotice,
} from "./constants";
import type { MessageWithParts, Turn } from "./plan";
import {
  allocateOmission,
  commitPlannedOmission,
  planOmission,
  type PlannedOmission,
} from "../storage/omission";
import { isRecord, unwrapString } from "../util";
import { coveredAssistantImagePartIDs } from "./media";

type PruneContext = {
  v2: V2Client;
  sessionID: string;
};

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

const DEFAULT_OUTPUT_DESCRIPTION =
  "Output omitted due to a compaction operation.";

const DEFAULT_LIMIT = { words: 128, chars: 1024 };
const TASK_OUTPUT_LIMIT = { words: 512, chars: 4096 };

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
        part: {
          ...mutation.replacement,
          sessionID: context.sessionID,
        },
      }),
    );
  }
}

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

// `onFirstWrite` runs once, immediately before the first source part update is
// attempted, so the caller can tell a trim that never wrote from one whose
// write may have partly applied.
export async function trimToolParts(
  context: PruneContext,
  turns: Turn[],
  keep: ReadonlySet<string>,
  onFirstWrite: () => void = () => {},
): Promise<number> {
  let trimmed = 0;
  let writing = false;
  const beforeUpdate = () => {
    if (writing) return;
    writing = true;
    onFirstWrite();
  };
  for (const turn of turns) {
    for (const assistant of turn.assistants) {
      for (const part of assistant.parts) {
        if (
          part.type === "tool"
          && !keep.has(part.id)
          && (await pruneToolPart(
            context,
            part,
            entry => allocateOmission(context.sessionID, entry),
            beforeUpdate,
          ))
        ) {
          trimmed += 1;
        }
      }
    }
  }
  return trimmed;
}

// True when the existing tool rules would change a completed part's model
// input or output, not merely stamp its trim metadata. Runs the rules on a
// clone and allocates no omission IDs.
export async function trimWouldChangeContent(part: ToolPart): Promise<boolean> {
  if (part.state.status !== "completed") return false;
  const probe = structuredClone(part);
  await prepareToolPart(probe, async () => "retention-probe");
  return (
    probe.state.status === "completed"
    && (probe.state.output !== part.state.output
      || !isDeepStrictEqual(probe.state.input, part.state.input))
  );
}

async function pruneToolPart(
  context: PruneContext,
  part: ToolPart,
  allocate: (entry: { content: string }) => Promise<string>,
  beforeUpdate: () => void,
): Promise<boolean> {
  const changed = await prepareToolPart(part, allocate);
  if (!changed) return false;

  beforeUpdate();
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
  const alreadyTrimmed =
    isRecord(magicCompact) && magicCompact["trimmed"] === true;

  // Input rules are idempotent on their own results (placeholders, or a command
  // already cut below its limit), so they still apply to parts marked by
  // earlier rule sets. Output rules are not idempotent: read would re-wrap its
  // own omission notice.
  const inputNotice = await applyInputOmissions(part, allocate);
  if (alreadyTrimmed && !inputNotice) {
    return false;
  }
  if (!alreadyTrimmed) {
    await applyOutputOmissions(part, allocate);
  }

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

  if (part.tool === "task") {
    const prompt = unwrapString(input["prompt"]);
    if (prompt && exceeds(prompt, DEFAULT_LIMIT)) {
      const contentID = await allocate({ content: prompt });
      input["prompt"] =
        `[Historical task prompt omitted; Content ID: ${contentID}. Retrieve the original with read_omitted_content before reuse.]`;
      return inputOmissionNotice(
        "Task prompt omitted due to compaction operation. If necessary, reread it via read_omitted_content.",
        Buffer.byteLength(prompt),
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
      "Historical read contents omitted due to compaction operation. Retrieve the exact captured result with read_omitted_content; reread the file for current contents.",
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

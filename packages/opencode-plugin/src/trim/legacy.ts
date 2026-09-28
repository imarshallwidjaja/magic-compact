import type { ToolPart } from "@opencode-ai/sdk/v2";
import {
  inputOmissionNotice,
  outputOmissionNotice,
} from "../compact/constants";
import { isRecord, unwrapString } from "../util";

const DEFAULT_OUTPUT_DESCRIPTION =
  "Output omitted due to a compaction operation.";
const DEFAULT_LIMIT = { words: 128, chars: 1024 };
const TASK_OUTPUT_LIMIT = { words: 512, chars: 4096 };

export async function prepareLegacyToolReplacement(
  source: ToolPart,
  allocate: (entry: { content: string }) => Promise<string>,
): Promise<ToolPart | null> {
  if (source.state.status !== "completed") return null;
  const part = structuredClone(source);
  if (part.state.status !== "completed") return null;
  const magicCompact = part.state.metadata["magicCompact"];
  const alreadyTrimmed =
    isRecord(magicCompact) && magicCompact["trimmed"] === true;

  // Input rules still apply to previously marked states; output rules must not
  // re-wrap a notice on a later pass.
  const inputNotice = await applyInputOmissions(part, allocate);
  if (alreadyTrimmed && !inputNotice) return null;
  if (!alreadyTrimmed) await applyOutputOmissions(part, allocate);
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
  return part;
}

async function applyInputOmissions(
  part: ToolPart,
  allocate: (entry: { content: string }) => Promise<string>,
): Promise<string | null> {
  if (part.state.status !== "completed") return null;
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
      const contentID = await allocate({ content: combined });
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
      input["prompt"] = "[Omitted]";
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
  if (part.state.status !== "completed") return;
  const output = part.state.output;
  if (part.tool === "question") return;
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
    const contentID = await allocate({ content: output });
    part.state.output = outputOmissionNotice(
      "Stale read contents omitted due to compaction operation. If necessary, reread to see current contents.",
      Buffer.byteLength(output),
      contentID,
    );
    return;
  }
  if (part.tool === "task") {
    if (exceeds(output, TASK_OUTPUT_LIMIT)) {
      const contentID = await allocate({ content: output });
      part.state.output = outputOmissionNotice(
        "Task output omitted due to a compaction operation. If necessary, reread output via read_omitted_content tool.",
        Buffer.byteLength(output),
        contentID,
      );
    }
    return;
  }
  if (!exceeds(output, DEFAULT_LIMIT)) return;
  const contentID = await allocate({ content: output });
  part.state.output = outputOmissionNotice(
    DEFAULT_OUTPUT_DESCRIPTION,
    Buffer.byteLength(output),
    contentID,
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
  return trimmed ? trimmed.split(/\s+/).length : 0;
}

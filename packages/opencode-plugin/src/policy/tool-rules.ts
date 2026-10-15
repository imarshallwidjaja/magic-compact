import { isDeepStrictEqual } from "node:util";
import {
  inputOmissionNotice,
  outputOmissionNotice,
} from "../compact/constants";
import { unwrapString } from "../util";

// Host-neutral tool trim rules. Each host maps its native tool names onto these
// categories and its native tool records onto `ToolCallView`, then applies the
// returned plan to a clone of its own record; the rules never see host records.
export type ToolRuleCategory =
  | "write"
  | "edit"
  | "patch"
  | "shell"
  | "subagent"
  | "read"
  | "todo"
  | "question"
  | "skill"
  | "other";

export type ToolCallView = {
  category: ToolRuleCategory;
  input: Readonly<Record<string, unknown>>;
  // The model-facing text output of a completed call.
  output: string;
  // Set when an earlier trim already processed this call.
  trimmed: boolean;
};

// `input` is the complete replacement input; `inputNotice` precedes the output;
// `output` replaces the text output when set.
export type ToolTrimPlan = {
  input: Record<string, unknown>;
  inputNotice: string | null;
  output: string | null;
};

export type OmissionAllocator = (entry: { content: string }) => Promise<string>;

const DEFAULT_OUTPUT_DESCRIPTION =
  "Output omitted due to a compaction operation.";

const DEFAULT_LIMIT = { words: 128, chars: 1024 };
const TASK_OUTPUT_LIMIT = { words: 512, chars: 4096 };

// Returns null when the call is left exactly as it is. A non-null plan marks
// the call trimmed even when neither input nor output changes.
export async function planToolTrim(
  call: ToolCallView,
  allocate: OmissionAllocator,
): Promise<ToolTrimPlan | null> {
  const input = structuredClone(call.input) as Record<string, unknown>;
  // Input rules are idempotent on their own results (placeholders, or a
  // command already cut below its limit), so they still apply to calls marked
  // by earlier rule sets. Output rules are not idempotent: read would re-wrap
  // its own omission notice.
  const inputNotice = await applyInputOmissions(call.category, input, allocate);
  if (call.trimmed && !inputNotice) return null;
  const output = call.trimmed
    ? null
    : await applyOutputOmissions(call.category, call.output, allocate);
  return { input, inputNotice, output };
}

// The output text a plan leaves for the model, in the shape the V1 host
// persists: the input notice, a blank line, then the (replaced) output.
export function plannedOutputText(
  call: ToolCallView,
  plan: ToolTrimPlan,
): string {
  const output = plan.output ?? call.output;
  return plan.inputNotice ? `${plan.inputNotice}\n\n${output}` : output;
}

// True when the rules would change the call's model input or output, not
// merely mark it trimmed. Allocates no omission IDs.
export async function toolTrimChangesContent(
  call: ToolCallView,
): Promise<boolean> {
  const plan = await planToolTrim(call, async () => "retention-probe");
  return (
    plan !== null
    && (plannedOutputText(call, plan) !== call.output
      || !isDeepStrictEqual(plan.input, call.input))
  );
}

async function applyInputOmissions(
  category: ToolRuleCategory,
  input: Record<string, unknown>,
  allocate: OmissionAllocator,
): Promise<string | null> {
  if (category === "write") {
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

  if (category === "patch") {
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

  if (category === "shell") {
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

  if (category === "edit") {
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

  if (category === "subagent") {
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
  category: ToolRuleCategory,
  output: string,
  allocate: OmissionAllocator,
): Promise<string | null> {
  if (category === "question") {
    return null;
  }

  if (category === "todo") {
    return "Successfully updated todos.";
  }

  if (category === "skill") {
    return "Skill contents omitted due to compaction operation. If necessary, recall skill.";
  }

  if (category === "read") {
    const contentID = await allocate({
      content: output,
    });
    return outputOmissionNotice(
      "Historical read contents omitted due to compaction operation. Retrieve the exact captured result with read_omitted_content; reread the file for current contents.",
      Buffer.byteLength(output),
      contentID,
    );
  }

  if (category === "subagent") {
    if (!exceeds(output, TASK_OUTPUT_LIMIT)) return null;
    const contentID = await allocate({
      content: output,
    });
    return outputOmissionNotice(
      "Task output omitted due to a compaction operation. If necessary, reread output via read_omitted_content tool.",
      Buffer.byteLength(output),
      contentID,
    );
  }

  if (!exceeds(output, DEFAULT_LIMIT)) {
    return null;
  }

  const contentID = await allocate({
    content: output,
  });
  return outputOmissionNotice(
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
  if (!trimmed) {
    return 0;
  }

  return trimmed.split(/\s+/).length;
}

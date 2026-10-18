import { isDeepStrictEqual } from "node:util";
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
  // The native child session of a subagent call, when the host reported one.
  childSessionID?: string;
};

// `input` is the complete replacement input, carrying its own recovery
// references; `output` replaces the text output when set.
export type ToolTrimPlan = {
  input: Record<string, unknown>;
  output: string | null;
};

const NATIVE_SESSION_ID = /^ses_[A-Za-z0-9]+$/u;

// A host-reported child session ID, accepted only when it is a well-shaped
// native ID on a subagent call, so arbitrary metadata never reaches the model.
export function childSessionID(
  category: ToolRuleCategory,
  value: unknown,
): string | undefined {
  return category === "subagent"
    && typeof value === "string"
    && NATIVE_SESSION_ID.test(value)
    ? value
    : undefined;
}

export type OmissionAllocator = (entry: { content: string }) => Promise<string>;

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
  // its own omission reference.
  await applyInputOmissions(call.category, input, allocate);
  if (call.trimmed && isDeepStrictEqual(input, call.input)) return null;
  const output = call.trimmed
    ? null
    : await applyOutputOmissions(call, allocate);
  return { input, output };
}

// True when the rules would change the call's model input or output, not
// merely mark it trimmed. Allocates no omission IDs.
export async function toolTrimChangesContent(
  call: ToolCallView,
): Promise<boolean> {
  const plan = await planToolTrim(call, async () => "retention-probe");
  return (
    plan !== null
    && ((plan.output ?? call.output) !== call.output
      || !isDeepStrictEqual(plan.input, call.input))
  );
}

// A compact recovery reference: what was omitted, optional details, and the
// Content ID that `read_omitted_content` takes.
function omitted(
  what: string,
  contentID: string,
  ...details: string[]
): string {
  return `[${[what, ...details, `Content ID: ${contentID}`].join("; ")}]`;
}

async function applyInputOmissions(
  category: ToolRuleCategory,
  input: Record<string, unknown>,
  allocate: OmissionAllocator,
): Promise<void> {
  if (category === "write") {
    const content = unwrapString(input["content"]);
    if (content && exceeds(content, DEFAULT_LIMIT)) {
      input["content"] = omitted("Omitted", await allocate({ content }));
    }
  }

  if (category === "patch") {
    const content = unwrapString(input["patchText"]);
    if (content && exceeds(content, DEFAULT_LIMIT)) {
      input["patchText"] = omitted("Omitted", await allocate({ content }));
    }
  }

  if (category === "shell") {
    const content = unwrapString(input["command"]);
    if (content && content.length > 1024) {
      const contentID = await allocate({ content });
      input["command"] =
        `${content.slice(0, 512)}\n${omitted("Command truncated", contentID)}`;
    }
  }

  if (category === "edit") {
    const oldString = unwrapString(input["oldString"]);
    const newString = unwrapString(input["newString"]);
    const combined = `${oldString}\n${newString}`;
    if (exceeds(combined, DEFAULT_LIMIT)) {
      const contentID = await allocate({ content: combined });
      input["oldString"] = omitted("Omitted with newString", contentID);
      input["newString"] = omitted("Omitted with oldString", contentID);
    }
  }

  if (category === "subagent") {
    const prompt = unwrapString(input["prompt"]);
    if (prompt && exceeds(prompt, DEFAULT_LIMIT)) {
      const contentID = await allocate({ content: prompt });
      input["prompt"] =
        `[Historical task prompt omitted; Content ID: ${contentID}. Retrieve before reuse.]`;
    }
  }
}

async function applyOutputOmissions(
  call: ToolCallView,
  allocate: OmissionAllocator,
): Promise<string | null> {
  const { category, output } = call;
  if (category === "question") {
    return null;
  }

  if (category === "todo") {
    return "Successfully updated todos.";
  }

  if (category === "skill") {
    return "Skill contents omitted due to compaction operation. If necessary, recall skill.";
  }

  const bytes = `(${Buffer.byteLength(output)} bytes)`;

  if (category === "read") {
    return omitted(
      `Historical read result omitted ${bytes}`,
      await allocate({ content: output }),
    );
  }

  if (category === "subagent") {
    if (!exceeds(output, TASK_OUTPUT_LIMIT)) return null;
    return omitted(
      `Task output omitted ${bytes}`,
      await allocate({ content: output }),
      ...(call.childSessionID ? [`child session: ${call.childSessionID}`] : []),
    );
  }

  if (!exceeds(output, DEFAULT_LIMIT)) {
    return null;
  }

  return omitted(
    `Output omitted ${bytes}`,
    await allocate({ content: output }),
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

export const POST_COMPACTION_NOTICE = `<post-compaction-notice>
A compaction operation has just been applied to all messages above. You may have to reread certain files to regain context. Certain historical tool input/output may have been omitted due to length. If the exact I/O of the tool call needs to be retrieved and functionality cannot be replicated via a new tool call, call the read_omitted_content tool with the appropriate Content ID to reread the tool I/O content.
</post-compaction-notice>`;

export const MAGIC_COMPACT_SUMMARIZER_AGENT = "magic-compact-summarizer";
export const MAX_SUMMARY_INPUT_TOKENS = 24_000;
export const MAX_SUMMARY_OUTPUT_TOKENS = 4_000;

export const MAGIC_COMPACT_SUMMARIZER_PROMPT = `You summarize selected historical assistant turns for Magic Compact.

The bounded exact evidence map in each request is authoritative. Never infer missing history, inspect external state, continue unfinished work, or perform the work described by the conversation. Assistant-text \`ordered-literal-runs-v1\` segments appear only when a whole source part is provably periodic: expand each segment by repeating its literal in source order; unique text remains verbatim in \`text\`.

Write each summary as a human-readable continuation capsule with an evidence-consistent Outcome line, a non-empty Work field, and only supported non-empty sections. Correctness and continuation fidelity come before brevity: preserve every continuation-critical fact, exact actionable identifier, and the exact next action, without restating tool structure, machine evidence records, or internal accounting the transcript already keeps. Reply only with the JSON object required by each Magic Compact request. Do not wrap it in Markdown fences or add prose.`;

export const BOUNDARY_METADATA = {
  magicCompact: {
    boundary: true,
  },
};

// Note: OpenCode orders parts by ID. Use "-" to order first.

export function summaryPartID(messageID: string): string {
  return `prt_-magic_summary_${messageID}`;
}

export function boundaryPartID(messageID: string): string {
  return `prt_-magic_boundary_${messageID}`;
}

export function summaryMetadata(): Record<string, unknown> {
  return {
    magicCompact: {
      summary: true,
    },
  };
}

export function outputOmissionNotice(
  description: string,
  length: number,
  contentID: string,
): string {
  return `<tool-output-omission-notice>
${description}

Output Length: ${length} bytes
Content ID: ${contentID}
</tool-output-omission-notice>`;
}

export function inputOmissionNotice(
  description: string,
  length: number,
  contentID: string,
): string {
  return `<tool-input-omission-notice>
${description}

Omitted Length: ${length} bytes
Content ID: ${contentID}
</tool-input-omission-notice>`;
}

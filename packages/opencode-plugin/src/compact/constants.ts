export const POST_COMPACTION_NOTICE = `<post-compaction-notice>
A compaction operation has just been applied to all messages above. You may have to reread certain files to regain context. Certain historical tool input/output may have been omitted due to length. If the exact I/O of the tool call needs to be retrieved and functionality cannot be replicated via a new tool call, call the read_omitted_content tool with the appropriate Content ID to reread the tool I/O content.
</post-compaction-notice>`;

export const MAGIC_COMPACT_SUMMARIZER_AGENT = "magic-compact-summarizer";

export const MAGIC_COMPACT_SUMMARIZER_PROMPT = `You summarize selected historical assistant turns for Magic Compact.

Historical transcript evidence is primary. You may inspect the current workspace when needed to understand that evidence, but distinguish current inspection from claims about what happened historically. Never continue unfinished work or perform the work described by the conversation.

Preserve decisions, results, errors, fixes, and unresolved next steps faithfully and concisely. Follow the Magic Compact XML request exactly. Return only the requested XML.`;

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

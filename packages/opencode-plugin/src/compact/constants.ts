import type { TextPart } from "@opencode-ai/sdk/v2";

export const POST_COMPACTION_NOTICE = `<post-compaction-notice>
A compaction operation has just been applied to all messages above. You may have to reread certain files to regain context. Certain historical tool input/output may have been omitted due to length. If the exact I/O of the tool call needs to be retrieved and functionality cannot be replicated via a new tool call, call the read_omitted_content tool with the appropriate Content ID to reread the tool I/O content.
</post-compaction-notice>`;

export const BOUNDARY_METADATA = {
  magicCompact: {
    boundary: true,
  },
};

// Note: OpenCode orders parts by ID. Use "-" to order first.

export function capsuleSummaryPartID(messageID: string): string {
  return `prt_-magic_capsule_v1_${messageID}`;
}

export function capsuleSummaryTextPart(
  messageID: string,
  sessionID: string,
  text: string,
): TextPart {
  return {
    id: capsuleSummaryPartID(messageID),
    sessionID,
    messageID,
    type: "text",
    text,
    metadata: summaryMetadata(),
  } satisfies TextPart;
}

export function boundaryPartID(messageID: string): string {
  return `prt_-magic_boundary_${messageID}`;
}

export function summaryMetadata(): Record<string, unknown> {
  return {
    magicCompact: {
      summary: true,
      capsule: true,
      version: 1,
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

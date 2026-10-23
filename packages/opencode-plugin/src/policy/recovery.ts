// `/magic-recover` policy shared by both hosts. Running the command declares
// that the source's unfinished stored records and undelivered background
// results belong to work that was abandoned. It proves nothing about that
// work: it stops, cancels and undoes nothing. Each host adapter closes the
// records in a copy of its own history with that host's native interrupted
// representation and these provenance texts, never a success, an output or a
// delivery.
import { isRecord } from "../util";
import { PluginMessageError } from "./feedback";
import { boundedLabel } from "./successor";

export const RECOVER_COMMAND_NAME = "Magic Recover";

// The error a closed assistant response carries.
export const RECOVERED_RESPONSE_ERROR =
  "Interrupted; outcome unknown. /magic-recover closed this response after it was declared abandoned.";

// The error a closed tool call carries.
export const RECOVERED_TOOL_ERROR =
  "Tool execution interrupted; outcome unknown. /magic-recover closed this call after it was declared abandoned.";

// What one recovery changed or left alone, by host record ID. Every ID names
// a record of the source session (or, for `subagent-record`, of one of its
// subagent sessions); the successor holds the closed copies under fresh IDs.
// Labels only, never content.
export type RecoveryRecord =
  // An assistant response with neither a completion time nor an error.
  | { kind: "response"; messageID: string }
  // A tool call that was still pending, running or streaming.
  | {
      kind: "tool-call";
      messageID: string;
      callID: string;
      tool: string;
      state: string;
    }
  // A background job whose result never reached the source.
  | {
      kind: "undelivered";
      messageID: string;
      callID: string;
      tool: string;
      job: string;
    }
  // An unfinished record in a subagent session's own history. Subagent
  // sessions are never written or copied, so it stays as it is.
  | { kind: "subagent-record"; sessionID: string; messageID: string }
  // An assistant response that already carries an error but no completion
  // time. OpenCode 2's import drops such a response, so the copy gets the
  // latest time the message already records and keeps its error and content.
  | { kind: "completion-time"; messageID: string };

const plural = (count: number, noun: string) =>
  `${count} ${noun}${count === 1 ? "" : "s"}`;

function recordLabel(record: RecoveryRecord): string {
  switch (record.kind) {
    case "response":
      return `assistant message ${boundedLabel(record.messageID, 64)} closed`;
    case "tool-call":
      return `${boundedLabel(record.state, 16)} ${boundedLabel(record.tool, 24)} call ${boundedLabel(record.callID, 64)} closed`;
    case "undelivered":
      return `${boundedLabel(record.tool, 24)} call ${boundedLabel(record.callID, 64)} result from ${boundedLabel(record.job, 64)} never delivered`;
    case "subagent-record":
      return `subagent session ${boundedLabel(record.sessionID, 64)} message ${boundedLabel(record.messageID, 64)} left unchanged`;
    case "completion-time":
      return `assistant message ${boundedLabel(record.messageID, 64)} copied with a completion time added`;
  }
}

// The outcome a recovery reports: in the successor's notice on OpenCode 1, in
// its lineage `outcome` on OpenCode 2, and after the success message. It
// counts every record and names only the first, so it stays short enough for
// a toast; the lineage's `recovery` list names them all.
export function recoveryOutcome(records: readonly RecoveryRecord[]): string {
  const count = (kind: RecoveryRecord["kind"]) =>
    records.filter(record => record.kind === kind).length;
  const changes = [
    `Closed as interrupted with unknown outcome: ${plural(count("response"), "assistant response")}, ${plural(count("tool-call"), "tool call")}.`,
    `Background results recorded as never delivered: ${count("undelivered")}.`,
    ...(count("subagent-record") > 0
      ? [
          `Unfinished subagent records left unchanged: ${count("subagent-record")}.`,
        ]
      : []),
    ...(count("completion-time") > 0
      ? [
          `Completion time added to errored responses the host cannot copy without one: ${count("completion-time")}.`,
        ]
      : []),
  ];
  const [first] = records;
  return [
    RECOVER_COMMAND_NAME,
    `Changes | ${changes.join(" ")}`,
    ...(first === undefined
      ? []
      : [
          `Record | ${recordLabel(first)}${records.length > 1 ? `; plus ${plural(records.length - 1, "more record")}` : ""}.`,
        ]),
    "Note | Recovery stopped, cancelled and undid nothing. Check files and subagent sessions before relying on work these records describe.",
  ].join("\n");
}

export const RECOVERY_CONTEXT_TAG = "magic-recover-notice";

// Marks the successor record that carries the recovery context to the model.
export const RECOVERY_CONTEXT_METADATA = {
  magicCompact: { recoveryContext: true },
};

export function isRecoveryContext(metadata: unknown): boolean {
  const magicCompact = isRecord(metadata)
    ? metadata["magicCompact"]
    : undefined;
  return isRecord(magicCompact) && magicCompact["recoveryContext"] === true;
}

// What the model needs to continue honestly after a recovery, as a stored
// record the host sends with the history. The successor's closed records speak
// for themselves only in part: a tool result carries its interrupted error, but
// a response's abort error stays in the record and a launch result still says
// the job was running. This text states both, names every call, job and
// subagent session a person or model would inspect, and is never capped, unlike
// `recoveryOutcome`, which is a short summary for a toast. It adds no result,
// success or delivery. Returns null when no record changes what the model may
// believe.
export function recoveryContext(
  records: readonly RecoveryRecord[],
): string | null {
  const lines: string[] = [];
  const responses = records.filter(record => record.kind === "response");
  if (responses.length > 0) {
    lines.push(
      `Responses interrupted with unknown outcome: ${responses.length}. What each did after its last visible content is unknown.`,
    );
  }
  const calls = records.flatMap(record =>
    record.kind === "tool-call"
      ? [
          `${boundedLabel(record.tool, 24)} call ${boundedLabel(record.callID, 64)} (${boundedLabel(record.state, 16)})`,
        ]
      : [],
  );
  if (calls.length > 0) {
    lines.push(
      `Tool calls interrupted with unknown outcome (their results say so): ${calls.join("; ")}.`,
    );
  }
  const launches = records.flatMap(record =>
    record.kind === "undelivered"
      ? [
          `${boundedLabel(record.tool, 24)} call ${boundedLabel(record.callID, 64)}, job ${boundedLabel(record.job, 64)}`,
        ]
      : [],
  );
  if (launches.length > 0) {
    lines.push(
      `Background results abandoned and never delivered (a launch result above may still say the job was running): ${launches.join("; ")}.`,
    );
  }
  const children = records.flatMap(record =>
    record.kind === "subagent-record"
      ? [
          `session ${boundedLabel(record.sessionID, 64)} message ${boundedLabel(record.messageID, 64)}`,
        ]
      : [],
  );
  if (children.length > 0) {
    lines.push(
      `Unfinished subagent records left unchanged: ${children.join("; ")}.`,
    );
  }
  if (lines.length === 0) return null;
  return [
    `<${RECOVERY_CONTEXT_TAG}>`,
    "The user ran /magic-recover to declare earlier work in this conversation abandoned. Recovery stopped, cancelled and undid nothing, so files, processes and subagent sessions may differ from what the history shows.",
    ...lines,
    "None of this is a success or a failure. Inspect files and the named jobs or subagent sessions before relying on the work.",
    `</${RECOVERY_CONTEXT_TAG}>`,
  ].join("\n");
}

// The job a recovery marker on a background launch names as abandoned, if the
// launch carries one: a child session ID on OpenCode 1, a shell or subagent ID
// on OpenCode 2. `withRecoveryMarker` writes the marker; this reads it.
export function abandonedDeliveryJob(
  metadata: Readonly<Record<string, unknown>> | undefined,
): string | undefined {
  const magicCompact = metadata?.["magicCompact"];
  const recovered = isRecord(magicCompact)
    ? magicCompact["recovered"]
    : undefined;
  return isRecord(recovered)
    && recovered["delivery"] === "abandoned"
    && typeof recovered["job"] === "string"
    ? recovered["job"]
    : undefined;
}

// A tool call's metadata with the recovery marker added under its own
// `magicCompact` entry, next to any marker an earlier trim left there.
export function withRecoveryMarker(
  metadata: Readonly<Record<string, unknown>> | undefined,
  marker: Record<string, unknown>,
): Record<string, unknown> {
  const magicCompact = metadata?.["magicCompact"];
  return {
    ...metadata,
    magicCompact: {
      ...(typeof magicCompact === "object" && magicCompact !== null
        ? magicCompact
        : {}),
      recovered: marker,
    },
  };
}

// A record this command cannot close without guessing what the host would
// have stored. The source is left as it is.
export function unsupportedRecoveryRecord(
  noun: string,
  id: string,
  reason: string,
): PluginMessageError {
  return new PluginMessageError(
    `${RECOVER_COMMAND_NAME} cannot close ${noun} ${boundedLabel(id, 64)}: ${reason}. No successor was created and the session was left unchanged.`,
  );
}

import { isRecord } from "../util";
import type { RecoveryRecord } from "./recovery";

// Earlier releases stored `{compactionCount}` (compacted source) or
// `{sourceSessionId, compactedAt, compactionCount}` (backup); successors on
// both hosts store their lineage. All carry the count under the same key.
export function getCompactionCount(session: {
  metadata?: Readonly<Record<string, unknown>>;
}): number {
  const metadata = session.metadata?.["magicCompact"];
  const count = isRecord(metadata) ? metadata["compactionCount"] : undefined;
  return typeof count === "number" && Number.isSafeInteger(count) && count > 0
    ? count
    : 0;
}

export type SuccessorOperation = "compact" | "trim" | "recover";

// Successor lineage stored under the new session's `metadata.magicCompact`.
// `ready` turns true only after the candidate is verified and the source was
// rechecked. `outcome`, written only on OpenCode 2, is the command's result
// for clients that did not run it. `recovery` lists every source record a
// recovery closed or left alone.
export type SuccessorLineage = {
  compactionCount: number;
  successorOf: string;
  operation: SuccessorOperation;
  createdAt: number;
  ready: boolean;
  outcome?: string;
  recovery?: RecoveryRecord[];
};

// The lineage entry in its stored key order. Absent optional fields stay
// absent, so the readback matches what was written.
export function successorLineage(
  entry: Omit<SuccessorLineage, "ready">,
  ready: boolean,
): SuccessorLineage {
  return {
    compactionCount: entry.compactionCount,
    successorOf: entry.successorOf,
    operation: entry.operation,
    createdAt: entry.createdAt,
    ready,
    ...(entry.outcome === undefined ? {} : { outcome: entry.outcome }),
    ...(entry.recovery === undefined ? {} : { recovery: entry.recovery }),
  };
}

// The source's metadata with its own lineage entry replaced by the
// successor's.
export function successorMetadata(
  source: Readonly<Record<string, unknown>> | undefined,
  lineage: SuccessorLineage,
): Record<string, unknown> {
  return { ...source, magicCompact: lineage };
}

// A candidate that is not ready yet is marked in session lists by its title.
export function successorTitle<T extends string | undefined>(
  title: T,
  ready: boolean,
): T {
  return (title === undefined || ready ? title : `${title} (preparing)`) as T;
}

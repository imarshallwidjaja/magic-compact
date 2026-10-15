import { isRecord } from "../util";

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

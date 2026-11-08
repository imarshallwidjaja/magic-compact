// The canonical stored-history count both hosts compute from their own
// records and report in compact and trim notices.

// Inline files a stored history holds, whether or not a request projects them.
export type InlineFileInventory = { count: number; encodedBytes: number };

// `tokens` is the canonical stored-history count that gates, profitability,
// stats and the successor check use; `inlineFileTokens` is the part of it that
// inline file data still counted as text contributes.
export type StoredHistoryCount = {
  tokens: number;
  inlineFileTokens: number;
  inlineFiles: InlineFileInventory;
};

// `tokens` are canonical tokens with covered images projected, and
// `projectedMediaTokens` the tokens the projection removes from every request.
export type ModelFacingCount = StoredHistoryCount & {
  projectedMediaTokens: number;
};

// The inline files are those of the stored history, so an image the
// projection replaces with its placeholder still counts as a stored file,
// while its placeholder counts as ordinary text.
export function modelFacingCount(
  persisted: StoredHistoryCount,
  projected: StoredHistoryCount,
): ModelFacingCount {
  return {
    tokens: projected.tokens,
    projectedMediaTokens: persisted.tokens - projected.tokens,
    inlineFileTokens: projected.inlineFileTokens,
    inlineFiles: persisted.inlineFiles,
  };
}

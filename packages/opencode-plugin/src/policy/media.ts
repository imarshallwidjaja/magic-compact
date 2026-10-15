import { countGptTokens, countGptTokensWithinLimit } from "../stats/tokenize";

// The declarative text that stands in for a manifest-covered inline image in
// a model request. Both hosts send the same text.
export function mediaPlaceholderText(ordinal: number, mime: string): string {
  return `[Historical inline image ${ordinal} (${mime}) was summarized by Magic Compact; its payload is projected out of this model request.]`;
}

// A covered image is projected only when its placeholder is strictly smaller
// than the image's own canonical token contribution, so projection never
// adds tokens. `original` is the exact stored payload string each host counts.
// A payload whose byte length alone proves it larger is never tokenized, so a
// request hook does not tokenize megabytes of image data.
export function projectsSmaller(
  placeholder: string,
  original: string,
): boolean {
  return (
    countGptTokensWithinLimit(original, countGptTokens(placeholder)) === null
  );
}

export function mediaManifestMismatch(): Error {
  return new Error("Magic Compact media manifest mismatch.");
}

# Scoring protocol (declared before any model call)

Applies to every blinded output in `blinded/outputs.json`. Gold facts are in `fixtures/gold.json` (same per case for every output). Graders see only the case ID, the gold facts and the capsule text (the persisted summary, including any `Visual summary:` section). Graders do not see model names, variants, latency, tokens, cost or any run metadata.

## Per output

- Text score: 1 point for each of the 8 text facts the capsule states correctly (0..8). A paraphrase counts when it carries the fact's meaning; judge by meaning, not by keywords. A partial fact earns no point unless the missing part is not needed to act on it.
- Image score (cases B and C only): 1 point for each of the 4 visual facts stated correctly (0..4). Case A has no image score.
- False assertions: subtract 2 for each distinct false assertion or contradiction in the relevant dimension (text or image), floor 0 per dimension. The same false claim repeated counts once. A false assertion is critical when it would misdirect the continuation (for example, claiming the staging load test passed, claiming the burst increase shipped, swapping left/right colors, or attaching the wrong state or count to a table row).
- Usable delivered summary: all 4 critical text facts present, every declared critical visual fact present (case B: B-V2 and B-V3; case C: C-V1..C-V4), and no critical false assertion.

## Failures

- A run that produced no persisted successor capsule (provider/network error, capability refusal, schema or marker validation failure, coverage failure, savings-gate failure, no-op, refusal) has no delivered summary. It scores 0 in every dimension and is not usable. An unpersisted model response is never graded.
- Failure categories, recorded separately by the operator from run evidence: capability, network/provider, schema, marker, coverage, savings, other.

## Reported separately, never used to compensate accuracy

Command latency, provider-reported tokens (input, output, reasoning, cached), summary length in characters, and the plugin's stored-history savings.

## Notes

- Case B repeats a previously observed diagnostic image; it is not held out.
- Gold facts describe the fixture's final state. Superseded plans in the source (burst 40 -> 80 hotfix, SQL-side UTC conversion, node-sass build from source) must not be reported as current; doing so is a false assertion.

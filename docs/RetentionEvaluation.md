# Task-Aware Retention Evaluation

Task-aware retention lets `/magic-compact` and `/magic-trim` keep a tool call whole when its content is likely to help the current task. The existing trim rules still choose which tool calls would change; TypeSafe Jev only vetoes trimming. This page records how the request format and keep threshold were chosen, what was measured on real sessions, and what is still unproven.

## Status

Both commands use `jev-1.13.0` with a whole-call keep threshold that defaults to `0.60`. The operator selected `0.60` on 2026-10-18 after the [development study](#default-keep-threshold-2026-10-18) below. It is not a held-out validation or a native-summary calibration. The server environment can override it with `MAGIC_COMPACT_KEEP_THRESHOLD_PERCENT`.

The September evaluation froze the request and a `0.80` threshold before held-out inference, and its results below are for `0.80`. With native session summaries and independently labelled real-session snapshots, the held-out majority decision at `0.80` kept 10 unique calls: 9 useful and 1 superseded failed query. Consensus useful recall was 9/28 (32%). These observations cover four held-out projects; ten keeps do not establish a universal 90% precision guarantee or show that the model probabilities are calibrated in this domain.

The September results come from the 64 development and 80 held-out real-session cases described below. A separate, public [decision evaluation](DecisionEvaluation.md) (`bun run eval:decisions`) sends 17 synthetic smoke cases to any versioned Jev model or to OpenAI Decisions. It is a repeatable sanity check of a decision model, not held-out data: its scores do not change the threshold, the question or the production model, and it never mixes with or relabels this cohort. Its question hash uses compact sorted-key JSON, so it is not comparable with the frozen question hash below.

Changing the default or the Jev question again needs the evaluation protocol on this page, and neither is tuned against held-out results. The archived evaluations keep their own thresholds: `eval/retention/run.ts` holds `0.80` as a constant of the September protocol, and `eval/decisions` holds its `0.80` reference as a constant of the decision-evaluation protocol. Neither reads the runtime default or the environment override, so stored results keep their meaning.

## Production contract

`packages/opencode-plugin/src/retention/classifier.ts` owns the classifier. `src/retention/summary.ts` produces the session summary, and `src/retention/retention.ts` wires both into `/magic-compact` and `/magic-trim`; `docs/OpenCode.md` describes that runtime flow.

- `classifyRetention(context, candidates, options)` returns `{decisions, requests, attempts, truncated, inputTokens, reportedUsageRequests}`. `decisions` maps every candidate ID to `{useful, keep}`, where `useful` is Jev's Noul probability and `keep` is `useful >= keepThreshold`. `requests` counts answered requests, `attempts` the HTTP requests sent including retries, and `truncated` the decisions made from the summary excerpt. `inputTokens` sums the input tokens Jev reported over `reportedUsageRequests` of the `attempts`; a refused attempt or a response without a non-negative integer count adds nothing and stays unknown.
- `context` is `{latestUserRequest, sessionSummary}`. `sessionSummary` is one fresh native whole-session summary, used temporarily for classification and never appended to the source conversation. Every first request carries it byte for byte, so the whole summary goes to TypeSafe with each candidate.
- Each candidate is `{id, tool, input, output, turnRequest, requestsBeforeLatest}` and should be a completed tool part whose trim would change its content. `id` stays local; it is never sent to Jev.
- `options.apiKey` is required. The plugin runtime supplies it from `TYPESAFE_API_KEY`; the module never reads the environment.
- `options.keepThreshold` is a finite probability from 0 to 1 and defaults to `RETENTION_KEEP_THRESHOLD` (`0.60`). `NaN`, infinities, values outside the range and non-numbers fail with a `configuration` error before any summary or request, and `0` is a valid threshold, not a missing one. The threshold decides `keep` only: it is not part of a request, and the returned probabilities do not depend on it.
- The plugin runtime derives the threshold once per command invocation in `src/policy/retention.ts`: from `--threshold P` on `/magic-compact` or `/magic-trim` when given, otherwise from `MAGIC_COMPACT_KEEP_THRESHOLD_PERCENT`, otherwise the default (see below).
- Every request carries one candidate and one Noul question that names `candidates[0]` in the state. The model is pinned to `jev-1.13.0` because both thresholds were measured against that version.
- Requests run with at most 4 in flight and a 30-second timeout each. By default there is no retry.
- `options.truncateSummary` (`--truncate` on `/magic-compact` or `/magic-trim`) allows one retry per candidate whose whole-summary request Jev confirms as `token-limit`. The retry body is identical except that `state.task.sessionSummary` holds the summary's head and tail within `RETENTION_SUMMARY_EXCERPT_BYTES` (8,000 serialized UTF-8 bytes), the same excerpt the classifier applied to every request before the [whole-summary change](#whole-session-summary-2026-10-22). Whether a shorter summary exists is decided without the option, from the summary's serialized size alone, and is reported in the failure diagnostics; the excerpt is built once per invocation, only when the option is given and the summary is longer than the excerpt. A summary that already fits has no retry. The retry has its own slot and deadline and passes the same 32 MiB check, and its failure is terminal. The option requires singleton requests: `batchSize` other than 1 fails with `configuration`. Under the option a whole-summary 400 pauses new requests while its body is classified instead of stopping them. Decisions from the excerpt are judged with less session context than the frozen whole-summary request; no evaluation on this page measured them.
- Any configuration, budget, token-limit, timeout, network, HTTP, or response-validation failure throws `RetentionClassificationError` and returns no decisions. The first terminal failure stops new requests, as soon as a non-OK status arrives except for the paused 400 above; requests already in flight finish before that error is thrown. The error message is fixed text plus an optional HTTP status. `token-limit` (Jev's HTTP 400 `max_tokens_exceeded`) and `budget` (a body over 32 MiB, refused before any request) say that no retention decisions were applied. A first `token-limit` refusal of a summary longer than the excerpt suggests the person's own command with `--truncate` or with `--heuristic` (`/magic-trim` when no command context exists); an exhausted retry, a summary already within the excerpt (with or without `--truncate`) and `budget` suggest `--heuristic`. None has an automatic fallback. The error's `diagnostics` carry only the failed attempt's position, summary variant, body bytes, status and validated request ID, the truncation state, and the invocation's candidate and attempt counts.
- A response is valid only when its `model` equals the pinned version, its answer keys exactly match the question keys, and each answer is a Noul with a probability in [0, 1].
- Judge every call independently: would the assistant likely use concrete available input or output again for the current task? Duplicates and recoverable content qualify. Still-relevant specifications, instructions and procedures qualify; a partly edited read qualifies when unchanged content remains useful. Empty notices, bookkeeping, genuinely superseded content, finished/different-task content and topic-only matches do not. There is no categorical todo or skill exclusion.
- Fixed excerpt allocations, in UTF-8 bytes of serialized JSON: latest request 5,000; originating turn request 600; input 3,000; output 9,000. The session summary has no allocation. Content that fits is sent whole; larger values retain head and tail with an omitted-character marker. Input fields up to 200 bytes stay whole when the aggregate fits; overflowing input becomes one excerpted arguments blob. Excerpting affects only Jev's evidence, not the whole-call keep.
- The only local size check is TypeSafe's documented 32 MiB request body limit, over the exact serialized body (model, state and questions), checked for every planned body before any request. Jev's token limits (32K tokens for state plus any one question, 64K for state plus all questions) are enforced by the service. Its tokenizer is not public and no token-count endpoint exists, so bytes do not predict tokens: in a 2026-10-22 probe of `jev-1.13.0`, 128,000 bytes of synthetic prose state was accepted at 30,549 input tokens while 64,000 bytes of dense identifiers and digits was refused. The refusal is HTTP 400 with the body `{"detail":{"error_type":"max_tokens_exceeded"}}` and no usage; the classifier reads at most 4,096 bytes of a 400 body and recognizes only that shape. Other statuses are not read: a 422 body details the offending field and can echo the request.
- The earlier 28,000-byte state-plus-question and 56,000-byte total guards are removed, and the 8,000-byte summary excerpt is no longer applied in advance. They were empirical byte proxies, not token limits. The excerpt returns only as the opted-in `--truncate` retry after Jev has refused a request.
- `TYPESAFE_API_KEY` must be available in the OpenCode server process. A native compaction model must be explicitly configured. With no candidates, retention makes no summary or TypeSafe request and needs no key. Otherwise any failure stops the command unchanged before backup or source writes. `/magic-trim [N] --heuristic` skips retention and makes no model calls; `/magic-compact [N] --heuristic` skips retention and still summarizes each turn with the model.
- Keeps veto the existing trim rules for this invocation, without size caps, persistent pins or automatic restoration. Below-threshold uncertainty follows those rules. Decisions are recomputed on the next invocation.
- `MAGIC_COMPACT_KEEP_THRESHOLD_PERCENT` is optional and read from the OpenCode server process environment on every task-aware `/magic-compact` and `/magic-trim` invocation, on OpenCode 1 and OpenCode 2. It is a plain decimal percent from 0 to 100 inclusive: `70` means probability 0.70, `60.5` means 0.605, and `0` and `100` are valid. Surrounding whitespace is ignored. Unset uses the `0.60` default. An empty or whitespace-only value, a sign, `%`, an exponent, hex, `NaN`, `Infinity`, a value above 100 (judged on the digits, never after rounding), a nonzero value too small for a double, or any other form fails the command before the summary, any Jev request or any successor write, with a fixed message that names the variable and the range and never echoes the value. There is no fallback, clamp or coercion. `--threshold P` on either command uses the same grammar for that command, wins over the variable and never reads it; `--heuristic` and `/magic-stats` neither read nor validate it. The comparison stays `>=`, and the pinned model, question, allocations, concurrency and timeout are unchanged.

## Native-summary evaluation

This evaluation froze and used `0.80`; its `metrics` and `score` commands keep that value as their own constant. The September 30 evaluation used fresh snapshots from a read-only archive, with candidates selected by production `createTrimPlan` and `findRetentionCandidates` for `/magic-trim 0`. Native summaries came from `generateSessionSummary` on isolated OpenCode 1.18.30 with `openai/gpt-6-luna-fast`, variant `low`. The entire `ses_fad9f918...@g2` snapshot (20 cases) was excluded before inference because of future-content contamination. The remaining 144 cases cover seven sessions: 64 development cases from three sessions in one project, and 80 held-out cases from four sessions in four other projects.

Two independent blind annotators froze labels before seeing scores. The primary label is their agreement on true or false; disagreement or a null label remains unresolved. Development has 37 useful, 17 not useful and 10 unresolved cases. Held-out has 28 useful, 38 not useful and 14 unresolved. Annotator B disclosed partial inspection of about 20 long outputs, so the labels are not exhaustive line-by-line checks of every payload.

### Selection and freeze

The predeclared selection rule chose the lowest threshold in {0.80, 0.85, 0.90} at which both development runs had precision at least 0.80, at least three true positives and recall at least 0.10. The two development runs were chronological and reversed. At 0.80 each kept 24 useful calls and no false positives, with consensus recall 24/37 (65%). At 0.85 they kept 19 and 18; at 0.90, 9 and 8. The rule selected 0.80 before held-out inference. No question, allocation, threshold or label changed after the freeze.

The full question and provenance are in `packages/opencode-plugin/eval/retention/native-summary-provenance.json`. Frozen hashes:

- Question: `4c8879efb6b7a9ed509e1952a8354a79676a89fc43fa30e34bca8da28062ff90`.
- Held-out body set: `6461a94eb499789a8975ac11778c85f68e92edee87eb8ca70698d3a8ca2a088f`.
- Joined labels: `12eca21e3ad85d262022b75dfd2dcedce3122fc1b3be0ff15f8fd11bf19ec176`.
- Freeze: `f77401d646c1aeacd9bf52878afcb00804488d4e3d6f1f5761fd4a664053217b`.

The question hash uses SHA-256 of UTF-8 `json.dumps(question, sort_keys=True)` in Python, with the default separators (`", "` and `": "`) and default `ensure_ascii=True`. Compact JavaScript `JSON.stringify` produces different bytes and a different hash. Request body hashes use the exact production `JSON.stringify(request.body)` bytes.

### Held-out results and stability

Three chronological passes and one reversed pass sent the same 80 singleton request bodies. All 320 requests returned HTTP 200 without retries. Repeats are stability measurements of the same cases, not additional independent samples.

| Decision                      | Kept | TP  | FP  | Consensus recall |
| ----------------------------- | ---- | --- | --- | ---------------- |
| Chronological 1               | 10   | 9   | 1   | 32%              |
| Chronological 2               | 9    | 8   | 1   | 29%              |
| Chronological 3               | 10   | 9   | 1   | 32%              |
| Reversed                      | 11   | 10  | 1   | 36%              |
| Strict majority, unique cases | 10   | 9   | 1   | 32%              |

The majority confusion counts on 66 agreed cases are TP 9, FP 1, FN 19 and TN 37. No unresolved case was kept. Precision is also 9/10 against either annotator alone; recall is 28% for A and 27% for B. Counting every unresolved case as useful reduces majority recall to 21%, without changing keep precision. One of the four held-out snapshots kept nothing despite seven consensus-useful cases.

Across the four passes, per-case probability spread was median 0.02, p90 0.05 and maximum 0.11. Two useful cases flipped around 0.80: a reconcile output (0.78/0.78/0.78/0.80) and a plan read (0.80/0.79/0.80/0.80). The false keep was a failed GraphQL query later corrected, scoring 0.82 to 0.83 in every pass. Later work is visible to each singleton judgment only through the summary, making supersession a failure mode.

Majority keeps retained 14.0% of candidate bytes (92,925/663,682) and 15.5% of estimated candidate tokens (25,516/164,783; local `gpt-tokenizer`). This cohort contains no `edit`, `write` or `apply_patch` calls. It therefore does not validate edit-tool retention; earlier examples were diagnostic only. Head/tail clipping hid middle evidence in 26 held-out outputs, four inputs, 19 latest requests and 44 turn requests. No held-out summary was clipped. Cases cluster within sessions, development covers one project, and there are only ten unique held-out keeps.

### Cost and latency

The six classifier passes used 1,693,290 billed input tokens, approximately $0.071 at the evaluation's $0.042/million rate. Native summaries used OAuth; their monetary cost was not measured. Jev request p50 was 225 to 233 ms and p90 257 to 281 ms. Invocations of 16 to 24 candidates took 0.96 to 1.59 seconds at concurrency four. Request preparation took about 45 ms per pass; record-write cleanup rounded to 0 ms. Native summary invocations took 10.1 to 17.8 seconds, including roughly 0.8 to 0.9 seconds for fork, prompt and cleanup. Summary generation dominated feedback time in this environment.

### Reproduce without committing session content

`native-summary-cases.json` records per-case labels, scores and request hashes. `native-summary-results.json` records unique-case metrics, repeat rates and timing. `native-summary-provenance.json` binds dataset, summaries, labels, protocol, freeze and runs. These content-free artifacts are distinct from the historical single-annotator `labels.json` below.

The existing `run.ts` accepts an external JSON input for this cohort: `{snapshots: [{snapshotID, split, context: {latestUserRequest, sessionSummary}, candidates: RetentionCandidate[]}]}`. Candidate `id` is the case ID (`snapshotID#NN`). Supply exact frozen candidates and summaries privately; hashes cannot reconstruct missing content. `check` binds each production-built request body hash to its candidate ID, then captures production `classifyRetention` fetch bodies with a fake endpoint and verifies the returned decision mapping, without reading a key or making any network request. Whole candidates can be reordered, and distinct IDs can share identical bodies when each matches its frozen hash. `score` checks the entire cohort before paid inference and uses the production singleton/concurrency/no-retry path. It writes scores and timing only. `metrics` recomputes results from the committed labels and scores without private inputs or inference.

The frozen body hashes were built when the classifier clipped the session summary to 8,000 serialized bytes. The current builder sends the summary whole, so a case whose summary was clipped no longer matches its frozen hash, and `check` and `score` refuse a cohort that contains one. Cases whose summary fit in 8,000 bytes, including every held-out case, still build the same bytes. The frozen hashes, labels, scores and thresholds stay as recorded and are not rebuilt for the current request.

```bash
bun packages/opencode-plugin/eval/retention/run.ts check /private/retention-input.json
bun packages/opencode-plugin/eval/retention/run.ts metrics
# Paid inference; output path must not exist:
bun packages/opencode-plugin/eval/retention/run.ts score /private/retention-input.json heldout chronological /tmp/opencode/retention-scores.json
```

## Rejected chronology experiment (2026-10-02)

The `callOrder` experiment was rejected at development selection and was not adopted. It tested source `c984287005f1f049390677f0473cf939346dda78`, adding only `state.candidates[0].callOrder`: `tool call <k> of <n> made while handling that request; <n-k> later call(s) followed it`. Coordinates were one-based chronological positions among completed calls in the same ordinary-request group, without timestamps. Both arms shared native summaries and unchanged model, 0.80 threshold, question, allocations, singleton/concurrency/no-retry settings. The field added 91 to 94 serialized bytes in development.

Development had 48 cases: 15 agreed useful, 32 agreed not useful (including 23 agreed superseded), and one unresolved. Precision and recall exclude unresolved cases; total keeps include them.

| Development arm            | Keeps | TP  | FP  | FN  | TN  | Unresolved keeps | Precision   | Recall       | Superseded keeps |
| -------------------------- | ----- | --- | --- | --- | --- | ---------------- | ----------- | ------------ | ---------------- |
| Baseline, chronological    | 10    | 5   | 4   | 10  | 28  | 1                | 5/9 (55.6%) | 5/15 (33.3%) | 4                |
| Candidate, chronological   | 11    | 5   | 5   | 10  | 27  | 1                | 5/10 (50%)  | 5/15 (33.3%) | 5                |
| Candidate, reversed repeat | 11    | 5   | 5   | 10  | 27  | 1                | 5/10 (50%)  | 5/15 (33.3%) | 5                |

Predeclared gates required strictly fewer superseded keeps than a baseline with at least one, TP loss <= 1, precision >= 0.80 with TP >= 3 and FP <= baseline, and a repeat meeting those quality gates with <= 3 decision flips. Supersession and precision failed; recall passed. The stability gate failed on repeat quality, not order instability: zero candidate decision changes, maximum probability spread 0.06. No useful keep was gained or lost; the additional keep was an agreed-superseded false positive.

Whole-kept tool I/O increased from 31,551 to 34,017 local tokens (+2,466), and from 120,266 to 131,115 UTF-8 bytes (+10,849), measured with production `countGptTokens(JSON.stringify(input) + output)`. This is not net trim accounting or the overall model-request budget. Useful keeps retained the same 16,687 tokens; the unresolved keep retained 10,070. Superseded probabilities rose on 19/23 cases, with median changes +0.02 chronologically and +0.03 reversed. Added wording encouraging the wrong direction is a hypothesis, not an isolated causal result.

All 144 classifier requests returned HTTP 200, without failures or reruns, using 506,538 billed input tokens from API usage. The run used 144 of 300 authorized classifier calls and five of six authorized native-summary generations. Selection recorded `pass: false` and `candidateFreeze: null`; no held-out classifier calls or production classifier change followed. The unused budget is not authorization to keep tuning this rejected variant.

[`chronology-results.json`](../packages/opencode-plugin/eval/retention/chronology-results.json) contains aggregate counts, gates, usage, field definition and full SHA-256 evidence bindings. Private content remains outside Git under `/tmp/opencode/retention-chronology-evaluation`; hashes cannot reconstruct it. The artifact binds the final pre-inference freeze. The protocol retains append/freeze history and discloses an earlier protocol version whose full difference was not retained.

Limits: two development sessions/projects, mostly task and skill calls, some finished-task segments, no edit-tool cases or development reads later edited/reread. Two blind annotators produced 192 label rows across 96 cases (48 development, 48 fresh held-out) in 14 card-part dispatches; partial inspection was disclosed, with no post-score label changes. The baseline had one pass, so its variance is unknown. Held-out inference stopped because development failed. These results do not measure held-out/general production accuracy, replace September's native-summary results, or establish that 0.80 is universally unsuitable.

## Default keep threshold (2026-10-18)

On 2026-10-18 the operator set the default keep threshold to `0.60` after reviewing a development study, and added `MAGIC_COMPACT_KEEP_THRESHOLD_PERCENT` so a server can override it. The request, question, allocations, pinned model, concurrency, timeout and no-retry settings did not change. The September `0.80` freeze stays on record above, and no held-out data was used or re-scored. This is an operator choice made with the study below in view. It is not a new held-out validation or a native-summary calibration.

### Study

One completed-task snapshot of a real session held 294,691 stored tokens and 116 candidates, selected by the production trim rules. The study sent them to `jev-1.13.0` through the production classifier with an independent analyst-written session summary of 6,190 bytes. That is not a native `generateSessionSummary` summary, so the study measures the classifier alone, not end-to-end trim quality. Two independent AI label streams labelled every candidate before any score existed. They agreed on 21 useful and 86 not useful; 9 candidates were unresolved and are excluded from precision and recall. A chronological pass and a reversed pass made 232 Jev requests, all HTTP 200, with no retries. Each threshold was then applied offline to the stored probabilities.

A cell shows the chronological pass, then the reversed pass, when they differ. Remaining tokens are the history size after trimming every non-kept candidate, counted locally with `gpt-tokenizer`; saved is the reduction from 294,691.

| Threshold | Kept    | Useful kept | Not useful kept | Unresolved kept | Remaining tokens | Saved           |
| --------- | ------- | ----------- | --------------- | --------------- | ---------------- | --------------- |
| 0.80      | 0       | 0           | 0               | 0               | 56,129           | 80.95%          |
| 0.70      | 1       | 1           | 0               | 0               | 60,219           | 79.57%          |
| 0.60      | 5 / 4   | 5 / 4       | 0               | 0               | 74,275 / 72,664  | 74.80% / 75.34% |
| 0.50      | 14 / 13 | 9           | 3               | 2 / 1           | 92,374 / 91,310  | 68.65% / 69.02% |

At `0.80` both passes kept nothing, so the one candidate both annotators marked critical for the task was trimmed. At `0.60` that candidate was kept in both passes. One candidate changed decision between the two passes at `0.60` and one at `0.50`; the largest probability difference between passes was 0.07.

### Limits

- One snapshot from one session, 21 agreed useful candidates. `0.60` kept 4 or 5 of them with no false keep; that count cannot show a precision rate.
- The labels are from AI streams, not people, and 9 candidates were unresolved.
- The session summary was written by an analyst, not produced by the native summarizer the commands use.
- The earlier proxy-summary held-out run below kept 10 useful and 6 or 7 not useful candidates at `0.60`. That request format no longer exists, but it is the one held-out observation at this cutoff, and the new default was not validated against it or against the September held-out cohort.
- Jev probabilities are not shown to be calibrated in this domain, and supersession by later work remains a failure mode that a lower cutoff keeps more of.

### Provenance

The study directory stays outside Git, and its hashes cannot reconstruct private content. An independent offline recomputation of the labels, request parity with the production builder and every table cell passed 129 of 129 checks; it verified the mechanics, not label truth or the choice of threshold. SHA-256 of the files this section summarizes:

- Session summary `d62e9f2cfa2771d203311a7208780fec77223062b9ab47c5c0a9cda8b61c3217`.
- Labels freeze `a42d946ba923f38ed79e16cd18ac4e46ec92e927ac63eee8317533b1c98a6525`.
- Run manifest `f57e2339a1e408825e735c360d5b7ac53faaae282879ae36ed11a48d2512f92b`.
- Metrics `4ddad4c726f0a48afce2e6732699215ff7561aded134a87f1d64b56673291015`.
- Threshold table `f3f55a55c96a989072bcb79c260ff7c7da464ab57a9ce58100998d910bd9f77c`.

## Whole session summary (2026-10-22)

Until this change the classifier clipped the session summary to 8,000 serialized bytes, keeping its head and tail. In the native summary studied below, the clip removed the `Active`, `Blocked` and `Next Move` sections, which describe the work still in progress. The classifier now sends the summary whole. This is a request packaging change: the threshold, question, model and other allocations are unchanged, and no held-out data was used.

### Study

One snapshot of a real session gave 250 production trim candidates and one native summary of 13,681 serialized bytes. Two arms sent the same 250 singleton bodies to `jev-1.13.0`, one with the clipped summary and one with the whole summary; nothing else differed. Each arm returned 250 HTTP 200 responses with no retries, using 1,052,796 billed input tokens clipped and 1,459,546 whole. The largest whole-summary body was 24,734 bytes.

| Arm at `0.60`  | Kept | Stored tokens left (of 653,795) | Agreed useful kept | Agreed not useful kept |
| -------------- | ---- | ------------------------------- | ------------------ | ---------------------- |
| Clipped (8 KB) | 167  | 552,026                         | 8 of 8             | 12 of 16               |
| Whole          | 76   | 397,819                         | 5 of 8             | 5 of 16                |

Trimming every candidate would leave 228,987 tokens. Every call the whole summary kept at `0.60` was also kept with the clipped summary. Probabilities fell for most candidates (median change -0.20 over all 250). On the 24 agreed sample cases, AUC rose from 0.58 to 0.73; the stratified bootstrap 95% interval of the difference was 0.008 to 0.332.

The whole summary removed many keeps of content that later work had finished, and it also trimmed 3 of the 8 agreed useful calls that the clipped summary kept. It is not a uniform quality gain.

### Limits

- One session, one summary generation, and 24 agreed labels from two annotators, one of whom also wrote a proxy summary for another arm of the study. Unresolved cases are excluded from the label counts.
- Stored-token counts are local projections, not provider usage.
- The study does not measure how often a whole summary exceeds Jev's token limits. A summary dense enough to do so now fails the command unless `--truncate` was given and its excerpt retry is accepted; see the production contract above.

### Implementation checks

Offline, the current builder reproduces all 250 whole-summary study bodies byte for byte, and the classifier's send path sends those bytes. Two paid synthetic requests through the implemented classifier on 2026-10-22 behaved as specified: a 41,965-byte body carrying a 40,210-byte prose summary was accepted at 10,087 input tokens, and a 65,757-byte body carrying 64,000 characters of dense identifiers was refused as `token-limit` after one request. On 2026-10-23 two more paid calls ran the `--truncate` retry through the classifier with the same kind of dense synthetic body: a 64,985-byte request carrying a 63,230-byte summary was refused as `token-limit` (HTTP 400, the same refusal body as in the study), and the retry, identical except for an 8,000-byte summary excerpt (9,755 bytes), was accepted at 7,514 input tokens. This shows the retry path against the service, not decision quality. Private content stays outside Git.

## Historical proxy-summary evaluation

The sections below preserve the earlier methodology and failed held-out operating point. They do not measure the current request. That study used proxy summaries, a different question and smaller excerpts, with one annotator. The categorical todo/instruction exclusion and the distinction between indispensable content and likely reusable content also mismatched the now-settled independent-usefulness rubric. These differences motivated a fresh predeclared evaluation; historical labels and failures were not relabelled or deleted to improve results.

### Historical data

Six archived sessions from `opencode.db.bak-20260901-101113` were read with SQLite `immutable=1`. None was the active conversation, and no session storage was written. Sessions were paired by kind of work; within each pair, the session whose SHA-256 ID hash sorts first became development data.

| Kind                           | Development                      | Held out                         |
| ------------------------------ | -------------------------------- | -------------------------------- |
| Incident debugging             | `ses_03dea5104ffejUfuSyl9xWyKJi` | `ses_fcdc2e182ffeSA2j01aITOQQnd` |
| Tooling development            | `ses_09acc8f15ffe8YBD78Nh317p4K` | `ses_035ce2405ffeSnNpdxD83yCWhd` |
| Product documents and diagrams | `ses_00210d4a7ffeAZeVWzoOTCSXKc` | `ses_fcc3dd9beffe8FApDXau3ZBM9s` |

Each session gives two snapshots, taken after the real user turn at 70% and 45% of that session's real user turns. A snapshot sees only history up to the end of that turn. Candidates are completed tool parts whose content the current trim rules would change. The 70% snapshot samples up to 25 candidates from the last five user turns and up to 15 older ones; the 45% snapshot samples recent candidates only. Sampling uses a SHA-256 order of part IDs. The 45% snapshots were added before any classifier call, after the first labelling pass found too few useful examples.

The result is 224 labelled candidates: 104 development (20 useful) and 120 held out (14 useful). All useful labels fall in the recent stratum. Most candidates are `task` calls because these are orchestrator sessions; reads, greps, skills, todo lists, shell, and context tools make up the rest.

### Labels

One annotator (the implementing agent) labelled every candidate from a snapshot-only view before any Jev call. There was no independent adjudication. Labels are committed in `packages/opencode-plugin/eval/retention/labels.json` (commit `89b976e`) with a category, an ambiguity flag on 36 labels, and a short reason. Useful means the input or output holds concrete content likely to help continue the task in the latest request, even if it could be fetched again. Stale, superseded, finished-task, topic-only, empty, and routine content is expendable.

### Provisional summaries

The runtime passes one generated whole-session summary. This evaluation had none, so each snapshot used up to six provisional summaries built mechanically from each recent user request and the last assistant text of that turn, sent as a list of per-summary excerpts of 1,200 bytes each within 6,000 bytes. That request format no longer exists: the classifier now takes one `sessionSummary` string, and question wording names `task.sessionSummary` instead of `task.recentTurnSummaries`. Every result below is therefore historical, measured with proxy summaries in the earlier format, and is not a measurement of the shipped runtime.

### Historical protocol

1. Freeze sessions, split, snapshot rules, and labels (`89b976e`).
2. Tune on development data only. Four formats were compared: the original "would keeping this help" wording with criteria, the same without criteria, a batch of four candidates per request in both orders, and the shipped "does it contain specific content the assistant will need" wording.
3. Select one candidate per request for stability and the shipped wording for ranking quality. After reviewing the development results and before any held-out call, choose the lowest threshold in {0.80, 0.85, 0.90} with pooled development precision of at least 80% over at least 20 kept decisions. That rule selected 0.85.
4. Freeze wording and threshold in the classifier module (`1d24894`), confirm the frozen module reproduces the development result, then run the held-out split three times plus one summary ablation. Nothing was changed after the held-out runs.

### Historical results

AUC is the probability that a random useful candidate scores above a random expendable one. Precision and recall are at the frozen 0.85 threshold unless stated. Token share is the fraction of candidate input and output tokens (local `gpt-tokenizer` estimate) that would be kept.

| Run                                                    | AUC  | Kept (TP/FP)  | Precision | Recall | Token share |
| ------------------------------------------------------ | ---- | ------------- | --------- | ------ | ----------- |
| Development, original wording, 1 per request           | 0.79 | 6/2           | 75%       | 30%    | 8.6%        |
| Development, original wording, no criteria             | 0.75 | 0/1           | 0%        | 0%     | 0.2%        |
| Development, original wording, 4 per request           | 0.83 | 3/2           | 60%       | 15%    | 4.2%        |
| Development, original wording, 4 per request, reversed | 0.84 | 4/1           | 80%       | 20%    | 5.1%        |
| Development, shipped wording, 3 repeats                | 0.87 | 7/1, 7/2, 7/1 | 78 to 88% | 35%    | 13 to 15%   |
| Development, frozen module                             | 0.87 | 7/1           | 88%       | 35%    | 13.4%       |
| Held out, frozen, 3 repeats                            | 0.93 | 4/2 each      | 67%       | 29%    | 7.3%        |

Held out at lower thresholds: 0.70 kept 8/3 (73% precision, 57% recall); 0.60 kept 10/6 or 10/7 (59 to 63%, 71%). Both held-out false keeps carry ambiguous labels; among unambiguous labels, held-out keeps were 4 of 4 correct.

Stability:

- One candidate per request repeats closely. Across repeats the largest score change was 0.04 to 0.07, with at most 2 of 104 or 120 decisions changing at 0.8 or 0.9.
- Four candidates per request moved scores by up to 0.30, and reversing their order moved them by up to 0.40, changing 9 to 10 decisions at 0.8. This is why the shipped format asks about one candidate per request.
- Removing the summaries dropped AUC from 0.87 to 0.77 on development and from 0.93 to 0.88 on held-out data, so summary quality strongly affects the classifier.

By category, at 0.85 across both frozen runs: nothing older than the last five user turns was kept, and no routine notices, empty results, todo lists, skills, or stale reads were kept. Kept items were task findings (8 useful, 2 expendable), current reads (2 useful), one diagnostic (useful), and one raw search result (expendable). Both held-out errors are review findings that later work had already fixed. With one candidate per request, Jev sees later work only through the summaries, so this supersession is hard for it to detect. The one hidden-middle case, status links in the middle of a 3,484-character output, scored 0.54 and was trimmed.

Cost and latency, measured: the recorded runs made 1,370 requests using 3,006,182 input tokens, about $0.13 at the listed $0.042 per million, plus one 6-request smoke test. One held-out pass averaged 2,336 input tokens per request. With one candidate per request, request latency was 216 to 236 ms at the median and 250 to 300 ms at p90. Invocations of 3 to 32 candidates took 0.24 to 1.97 seconds at concurrency 4.

### Historical limitations

- The held-out split holds 14 useful labels, and 10 come from one snapshot. The 0.85 threshold keeps too few held-out items to confirm or reject a precision target.
- Labels come from one annotator without adjudication; 36 are flagged ambiguous.
- Summaries are provisional proxies. The runtime must be re-evaluated with generated Magic Compact summaries and with complete per-invocation candidate sets rather than samples.
- Candidates are mostly `task` calls from orchestrator sessions. Direct edit-heavy sessions are under-represented.
- Excerpts show head and tail only, so useful content in the middle of long outputs can be missed. Clipping limits evidence for both usefulness and supersession.

### Historical reproduction

These commands read the local archive, write raw content only under `/tmp/opencode/task-aware-retention-eval`, and make paid calls only through `run`.

```bash
bun packages/opencode-plugin/eval/retention/dataset.ts
TYPESAFE_API_KEY=... bun packages/opencode-plugin/eval/retention/run.ts run held-generated heldout 1 chronological shipped generated
bun packages/opencode-plugin/eval/retention/run.ts report held-generated
```

`dataset.ts` selects candidates with the production `trimWouldChangeContent` check. It compares every regenerated snapshot's candidate IDs with the frozen labels and, on any difference, prints the missing and added IDs and writes nothing; label the changed candidates under this protocol instead of pairing old labels with new candidates.

`run` takes `<name> <split> [batch] [order] [variant] [context]`. Variants are `shipped`, `help-wording` (the original wording), and `help-wording-no-criteria`. Context is `generated` (the default), `proxy`, or `latest-only`. `generated` reads `summaries/<snapshotID>.md` under the scratch directory; each file should hold the output of `generateSessionSummary` for that snapshot's history. `proxy` joins the provisional summaries into one string and is kept only for comparison; it does not reproduce the recorded results, which used the earlier list format. `bun test packages/opencode-plugin/test/retention.test.ts` covers the request format, budgets, validation, and failure behavior without network access.

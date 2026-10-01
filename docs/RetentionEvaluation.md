# Task-Aware Retention Evaluation

Task-aware retention lets `/magic-compact` and `/magic-trim` keep a tool call whole when its content is likely to help the current task. The existing trim rules still choose which tool calls would change; TypeSafe Jev only vetoes trimming. This page records how the request format and keep threshold were chosen, what was measured on real sessions, and what is still unproven.

## Status

Both commands use `jev-1.13.0` with a whole-call keep threshold of `0.80`. The request and threshold were frozen before held-out inference. With native session summaries and independently labelled real-session snapshots, the held-out majority decision kept 10 unique calls: 9 useful and 1 superseded failed query. Consensus useful recall was 9/28 (32%). These observations cover four held-out projects; ten keeps do not establish a universal 90% precision guarantee or show that the model probabilities are calibrated in this domain.

## Production contract

`packages/opencode-plugin/src/retention/classifier.ts` owns the classifier. `src/retention/summary.ts` produces the session summary, and `src/retention/retention.ts` wires both into `/magic-compact` and `/magic-trim`; `docs/OpenCode.md` describes that runtime flow.

- `classifyRetention(context, candidates, options)` returns `{decisions, requests, inputTokens}`. `decisions` maps every candidate ID to `{useful, keep}`, where `useful` is Jev's Noul probability and `keep` is `useful >= threshold`.
- `context` is `{latestUserRequest, sessionSummary}`. `sessionSummary` is one fresh native whole-session summary, used temporarily for classification and never appended to the source conversation.
- Each candidate is `{id, tool, input, output, turnRequest, requestsBeforeLatest}` and should be a completed tool part whose trim would change its content. `id` stays local; it is never sent to Jev.
- `options.apiKey` is required. The plugin runtime supplies it from `TYPESAFE_API_KEY`; the module never reads the environment.
- Every request carries one candidate and one Noul question that names `candidates[0]` in the state. The model is pinned to `jev-1.13.0` because the threshold was tuned against that version.
- Requests run with at most 4 in flight and a 30-second timeout each. There is no retry.
- Any configuration, budget, timeout, network, HTTP, or response-validation failure throws `RetentionClassificationError` and returns no decisions. The first failure stops new requests; requests already in flight finish before the error is thrown. The error message is fixed text plus an optional HTTP status and suggests `/magic-trim --heuristic`.
- A response is valid only when its `model` equals the pinned version, its answer keys exactly match the question keys, and each answer is a Noul with a probability in [0, 1].
- Judge every call independently: would the assistant likely use concrete available input or output again for the current task? Duplicates and recoverable content qualify. Still-relevant specifications, instructions and procedures qualify; a partly edited read qualifies when unchanged content remains useful. Empty notices, bookkeeping, genuinely superseded content, finished/different-task content and topic-only matches do not. There is no categorical todo or skill exclusion.
- Fixed excerpt allocations, in UTF-8 bytes of serialized JSON: latest request 5,000; session summary 8,000; originating turn request 600; input 3,000; output 9,000. Content that fits is sent whole; larger values retain head and tail with an omitted-character marker. Input fields up to 200 bytes stay whole when the aggregate fits; overflowing input becomes one excerpted arguments blob. Excerpting affects only Jev's evidence, not the whole-call keep.
- The state-plus-longest-question limit is 28,000 bytes and the total state-plus-questions limit is 56,000. Bytes are not a documented token bound. The largest billed request in the 448 saved live-run request records was 7,035 tokens; `requestTokenEvidence` in `native-summary-results.json` records that maximum and the six raw-run file hashes. This is empirical headroom against Jev's 32K state-plus-question limit, not a guarantee for arbitrary input.
- `TYPESAFE_API_KEY` must be available in the OpenCode server process. A native compaction model must be explicitly configured. With no candidates, retention makes no summary or TypeSafe request and needs no key. Otherwise any failure stops the command unchanged before backup or source writes. `/magic-trim [N] --heuristic` skips retention and makes no model calls.
- Keeps veto the existing trim rules for this invocation, without size caps, persistent pins or automatic restoration. Below-threshold uncertainty follows those rules. Decisions are recomputed on the next invocation.

## Native-summary evaluation

The September 30 evaluation used fresh snapshots from a read-only archive, with candidates selected by production `createTrimPlan` and `findRetentionCandidates` for `/magic-trim 0`. Native summaries came from `generateSessionSummary` on isolated OpenCode 1.18.30 with `openai/gpt-6-luna-fast`, variant `low`. The entire `ses_fad9f918...@g2` snapshot (20 cases) was excluded before inference because of future-content contamination. The remaining 144 cases cover seven sessions: 64 development cases from three sessions in one project, and 80 held-out cases from four sessions in four other projects.

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

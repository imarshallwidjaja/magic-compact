# Decision Evaluation

This evaluation measures how well a decision model answers the keep/trim question that task-aware retention asks. Each case fixes the context (the latest user request and a session summary) and one completed tool call. The model returns a probability that the assistant would use the call's content again, and the harness compares that probability with a hand-written label. It scores the decision only. It does not run `/magic-trim`, generate a summary, or judge the summary model, so a good result here says nothing about end-to-end trim quality.

The harness reuses the production request builder in `src/retention/classifier.ts` and the production wire serializers and answer readers in `src/retention/backends.ts`, so every provider answers the same state, the same question text and the same criteria that the shipped classifier sends to either backend. This evaluation changes no production code, keep threshold or question. The existing 144-case native-summary cohort and its commands are described in [RetentionEvaluation.md](RetentionEvaluation.md); this evaluation has its own dataset, scripts and results and never mixes with that cohort or rewrites its labels.

## Commands

All commands run from the repository root. `bun run eval:decisions` is the root script for `bun packages/opencode-plugin/eval/decisions/run.ts`, and either form takes the same arguments. `check`, `report` and `run --dry-run` are offline and need no key.

```bash
# Validate the bundled pack and print its identity.
bun run eval:decisions check

# Print the plan and request hashes for a model. No key, no network, no files.
bun run eval:decisions run --provider jev --model jev-1.14.0 --dry-run

# Paid run: TypeSafe Jev at a chosen version (default jev-1.13.0, the production pin).
export TYPESAFE_API_KEY=<your TypeSafe key>
bun run eval:decisions run --provider jev --model jev-1.14.0 --out /tmp/decisions/jev-1.14.0

# Paid run: OpenAI Decisions API (gpt-6-luna only). Needs an OpenAI API key.
export MAGIC_COMPACT_OPENAI_DECISIONS_API_KEY=<your OpenAI API key>
bun run eval:decisions run --provider openai-decisions --model gpt-6-luna --out /tmp/decisions/gpt-6-luna

# Compare saved runs of the same dataset. Offline.
bun run eval:decisions report /tmp/decisions/jev-1.14.0 /tmp/decisions/gpt-6-luna
```

Any output directory outside the repository works; `--out` must name a path that does not exist yet. `--repeat 2` or more sends each case again to measure stability.

`run` options: `--provider`, `--model`, `--dataset` (a path, or `default`), `--out` (required unless `--dry-run`), `--repeat` (1 to 10), `--threshold`, `--concurrency` (default 4, at most 16), `--max-requests` (default 200), `--timeout-ms` (default 30000), `--endpoint` and `--dry-run`. `report` takes one or more run directories, `--dataset` and `--threshold`. Run `--help` for the full text. Unknown commands, unknown flags and malformed values exit with status 2 and name the problem. `run` and `report` exit 0 when every planned request produced a probability or a refusal, and 1 when any request failed, was not attempted, or has no outcome record (its outcome is unknown, see Results).

## Keys and sign-in

The harness reads one environment variable for the selected provider and nothing else: `TYPESAFE_API_KEY` for `jev`, `MAGIC_COMPACT_OPENAI_DECISIONS_API_KEY` for `openai-decisions`. A missing key fails the run before the first request and before the output directory is created. Keys travel only in the `Authorization` header and are never written to a result.

The harness never reads OpenCode's credential stores or provider pool. OpenCode's ChatGPT sign-in is OAuth; OpenAI documents the Decisions API with API keys, and this harness does not try that sign-in against the endpoint. `--endpoint` redirects requests to another URL (https, or http on localhost for a local fake server) and sends the key there, so only point it at a server you control. Requests never follow redirects. A 3xx answer is recorded as an `http` failure with its status, because a redirect keeps the method and body and would send the case text to an origin you did not choose.

## Providers and models

| Provider           | Endpoint                               | Models                                                             |
| ------------------ | -------------------------------------- | ------------------------------------------------------------------ |
| `jev`              | `https://api.typesafe.ai/v1/systemone` | Any versioned ID such as `jev-1.13.0` or `jev-1.14.0`              |
| `openai-decisions` | `https://api.openai.com/v1/decisions`  | `gpt-6-luna`, the only model OpenAI currently offers for Decisions |

Jev aliases (`jev-latest`, `jev-preview`) are refused because they move between releases and would make runs unrepeatable. Production stays pinned to `jev-1.13.0`; the evaluation may select any version to measure it. For both providers the response `model` must equal the requested model exactly, for a probability and for a refusal; anything else is recorded as a `model-mismatch` failure. That means the requested Jev version, or `gpt-6-luna` for OpenAI. The harness accepts no dated snapshot label and no variant such as `gpt-6-luna-fast`, because no first-party contract says how to normalize one. If OpenAI starts returning a dated label for `gpt-6-luna`, runs fail with `model-mismatch` until a documented normalization is added here. An unsupported OpenAI model is rejected before any request, rather than sent and rejected case by case.

Adding a provider means adding one adapter object to `PROVIDERS` in `providers.ts`: a model check, a request translation, and a strict response reader. There is no registry or plugin loading. A provider that production also uses takes its serializer and reader from `src/retention/backends.ts`, so the evaluation reads answers exactly as production does.

### Request translation

Every case becomes one singleton request through `buildRetentionRequests` (batch size 1), with the case ID as the candidate ID. The result is the canonical request. Its hash (`canonicalRequestSha256`) covers the production `state` and `questions`, so it does not depend on the provider or model.

Jev receives the canonical body with only `model` replaced. For the production model the bytes equal what the shipped classifier sends, and a test asserts that.

OpenAI Decisions has no criteria field and takes `input` rather than `state`. The adapter sends `input` as the canonical state JSON string and one `predicate` question named by the case ID. Its `instructions` hold the production instructions, then `Yes, probability near 1: <criteria.true>`, then `No, probability near 0: <criteria.false>`. This rendering is versioned (`openai-predicate-1`) and its hash is recorded in each run. Production's `openai-decisions` backend sends the same body through the same serializer (`openAIDecisionsBody`), except that it names the predicate by the opaque question key `candidate_0` instead of a case ID; a test asserts that the name is the only difference, so the translation version and recorded hashes are unchanged. The exact wire body is hashed separately (`providerRequestSha256`) in every outcome.

OpenAI's `answers` array must contain exactly one answer, naming the requested case, of type `predicate` or `refusal`. Extra, missing, duplicate, foreign or wrongly typed answers are invalid responses. A refusal is its own outcome. It is not a probability of 0 and not a trim.

## Dataset

The bundled pack is `eval/decisions/fixtures.json`: 17 hand-labelled synthetic cases from a fictional TypeScript library, in four sessions. It covers current reads and test failures that the next step needs, a superseded read and a fixed failure, duplicated and recoverable content, a partly edited file whose unchanged part is still useful, a procedure, an open plan and subagent findings the remaining work follows, and the cases to trim: completed bookkeeping, an empty read, routine install output, a finished task for another request and a topic-only search. Every candidate is a call the production tool rules would change, and `check` reports whether that holds.

The pack was written alongside the classifier wording. Its split is `smoke`: a quick sanity and development aid, not held-out data, and not a basis for choosing a threshold. Seventeen cases cannot rank models finely or support a calibration claim.

### Format

```json
{
  "schemaVersion": 1,
  "datasetID": "my-pack",
  "version": "1.0.0",
  "split": "development",
  "description": "What the cases are.",
  "snapshots": [
    {
      "snapshotID": "session-a",
      "context": { "latestUserRequest": "...", "sessionSummary": "..." },
      "cases": [
        {
          "caseID": "session-a-read-config",
          "scenario": "free-form tag",
          "candidate": {
            "tool": "read",
            "input": { "filePath": "..." },
            "output": "...",
            "turnRequest": "...",
            "requestsBeforeLatest": 0
          },
          "label": { "keep": true, "rationale": "Why, from facts in the case." }
        }
      ]
    }
  ]
}
```

Parsing is strict: unknown fields, duplicate case IDs and IDs outside `[a-z0-9][a-z0-9_-]{0,63}` are rejected. Candidate fields and `context` are the production `RetentionCandidate` and `RetentionContext` fields. `split` is `smoke`, `development` or `heldout`. The dataset hash is SHA-256 of the canonical JSON (sorted keys), so formatting changes do not alter it and any content change does.

### Labelling

Label `keep: true` when the assistant would likely use concrete content from the call again for the current request: results, findings, current file or artifact contents, specifications, instructions or procedures the remaining work still follows, identifiers, links, commands, errors or decisions. Duplicated or recoverable content still qualifies, and so does the unchanged part of a file that was edited later. Label `keep: false` for empty or status-only results, bookkeeping, content superseded by later work, content for a finished or different request, and content related to the task only by topic. The label comes from facts visible in the case, not from a model's answer. The authoritative question text is in `classifier.ts`; this rubric summarizes it.

### Your own data

A custom dataset is sent in plain text to the provider you select. Do not use data you may not share with that provider. Declare `heldout` only for cases that were not used to design prompts, thresholds or this harness; held-out cases are scored only at the frozen threshold and are never swept.

## Threshold

`--threshold` (default `0.8`) turns a probability into keep (`>= threshold`) or trim. `0.8` is this evaluation's own reference cutoff, frozen by its protocol as `DECISION_EVALUATION_THRESHOLD` in `eval/decisions/dataset.ts`. It no longer equals production: runtime retention keeps at `>= 0.60` by default, and a server can override that with `MAGIC_COMPACT_KEEP_THRESHOLD_PERCENT` (see [RetentionEvaluation.md](RetentionEvaluation.md)). This harness never reads that variable, so results do not move with the runtime setting. Stored results, including the October 15 smoke run below, were scored at `0.8` and keep that meaning.

To see how a model behaves at the runtime default, pass `--threshold 0.6` on a smoke or development dataset. That is a development comparison, not a held-out result and not a calibration. Other models or providers are not calibrated to either cutoff, so a comparison shows how each model behaves at that cutoff and does not say that it is right for them. The option changes nothing in production. For `heldout` datasets, `run` and `report` refuse any value other than `0.8`, including `0.6`. For other splits, `report --threshold` re-reads stored probabilities at another cutoff; do not use that to tune production.

## Results

A run directory holds two files and is never overwritten.

- `manifest.json` is written before the first request: provider, model, endpoint, dataset identity and hash, question hash, the canonical request-set hash, the translation version and hash, threshold, `production` (the pinned Jev model and the runtime default keep threshold when the run was planned, recorded apart from the evaluation `threshold`; manifests written when the default was `0.8` record that and still load), repeat, concurrency, timeout, and the planned cases with each case's canonical and provider request hashes.
- `outcomes.jsonl` gets one line as each request settles. A line is `scored` (probability, requested and returned model, reported usage, latency), `refused`, `failed` (a fixed category and the HTTP status when there is one) or `not-attempted`.

Requests are sent once. There is no automatic retry, and `--max-requests` refuses a plan larger than the cap before any request. A rejected key (401, 403) or a rate limit (429) stops new requests; those in flight finish and the rest are recorded as `not-attempted`, which states that they were never sent. Timeouts, other HTTP errors, malformed responses, model mismatches and refusals do not stop the remaining cases. Failure records never contain request text, response bodies, error messages or keys.

`report` recomputes everything from these records. A request that left no record is **unknown**, not not-attempted. A paid request can reach the provider and be billed before the process dies and before its outcome is appended, so the harness cannot say it was never sent. An interrupted run still reports: it prints the unknown count next to the not-attempted count, flags the requests with no record, does not retry them, does not guess a probability, and exits 1. A directory with a manifest and no records reports every request as unknown. A final record cut off mid-write is unreadable, and `report` refuses the directory as corrupt rather than claim the request was or was not sent.

`report` and `run` verify a result against this checkout before using it, not only against itself. They rebuild each case's canonical request from the loaded dataset with the production builder, and the provider request from the adapter for the manifest's provider and model. The manifest's per-case hashes, question hash, and translation version and hash must equal those; a translation revision this checkout does not implement is refused, not blended. Every scored or refused record must name the manifest's model as the requested model and the same model as returned, and every record must carry the planned canonical request hash and, unless not attempted, the planned provider request hash. A directory from another dataset, model, provider, question wording or translation, or with records that do not belong to its plan, is refused.

## Metrics

For requests with a probability: the confusion matrix (TP keeps useful content, FP keeps content labelled trim, FN trims useful content, TN trims content labelled trim), precision, recall, specificity, F1, accuracy, the Brier score and the area under the ROC curve (ties count half). F1 is 2TP / (2TP + FP + FN), so it is 0, not `n/a`, when the scored decisions contain no true keep. Accuracy is the share of scored decisions that are right. A ratio with an empty denominator prints `n/a`, never 0. Coverage (scored, refused, failed, not attempted, unknown) is always shown, and metrics cover only scored requests, so a run with low coverage is flagged and not comparable with a complete one. False prunes (useful content below the threshold) and false keeps (wasted tokens) are listed by case.

With `--repeat`, the report adds the largest probability spread per case and the number of cases whose decision changed. Repeats measure stability, not independent samples. Latency percentiles include the provider request and parse. Token counts are the provider's reported usage in its own units, summed over scored and refused responses, with input and output counted separately. A cell shows the total and how many eligible responses reported it, such as `30 (2/2)`. Fewer than all makes it a `subtotal`, and none prints `n/a (0/N)`. A missing count is unknown, not 0, but a reported 0 counts. A count that is negative, fractional or above 9007199254740991 is treated as not reported. Failed requests have no usage. The harness computes no cost.

`report` groups runs by question hash, canonical request-set hash and threshold. Runs in a group are comparable. Runs that differ in any of these are printed in separate groups.

Only translations that this checkout implements load, so the group key already implies a known translation. A Jev run (the production body) and an OpenAI run (`openai-predicate-1`) of the same dataset therefore share a group: they ask the same canonical question and only the wire body differs. Provider request hashes differ between models by design and are not compared across runs. A result made with an older or unknown translation revision does not load, so it cannot be declared comparable by accident.

## Smoke qualification, October 15, 2026

One live run per model sent the 17 bundled cases (dataset `magic-compact-retention-decisions-synthetic` 1.0.0, smoke split, SHA-256 `859d9cb4e15b68b2fa52af9c3650541a2c54c096f05bcbaa0bdadb886adc9973`) once each to TypeSafe Jev, at concurrency 4 with no retries. It shows that the harness works end to end against the live Jev endpoint, not how accurate either model is.

| Model        | Scored | Refused or failed | TP  | FP  | FN  | TN  | Precision | Recall | Accuracy |
| ------------ | ------ | ----------------- | --- | --- | --- | --- | --------- | ------ | -------- |
| `jev-1.13.0` | 17/17  | 0                 | 8   | 0   | 1   | 8   | 1.000     | 8/9    | 16/17    |
| `jev-1.14.0` | 17/17  | 0                 | 8   | 0   | 1   | 8   | 1.000     | 8/9    | 16/17    |

Keep decisions use the `0.8` reference threshold. Both models put the same useful case below it: `release-todo-open-items`, at 0.65 for `jev-1.13.0` and 0.52 for `jev-1.14.0`. Its label was not changed after the run. Seventeen synthetic cases written alongside the classifier wording cannot show held-out accuracy, calibration or a difference between the two versions, and production stays on `jev-1.13.0`. These figures were scored at `0.8`, before production's default became `0.60`.

No live OpenAI Decisions run of the bundled pack is recorded here. On 2026-10-23, eight synthetic requests went through the OpenAI adapter to check the wire contract, the returned model, usage fields and the token-limit refusal; they are described in [RetentionEvaluation.md](RetentionEvaluation.md#openai-decisions-backend-2026-10-23) and are not a score. The adapter is otherwise covered by tests against fakes that enforce its strict response contract and by `--dry-run`; anyone with a key can run the pack with the command above.

## Limits

- Scored live evidence covers the Jev smoke run above only. The OpenAI contract check returned exactly `gpt-6-luna` on one day; wiring, parsing, failure handling and the command line are otherwise tested against fake fetch functions and a local HTTP server.
- OpenAI describes the Decisions API as generally available as of 2026-10-23; its model list, response fields and behavior may still change.
- The probabilities of different models are not calibrated to one another.
- The bundled pack is small and was written with knowledge of the classifier, so treat results as a sanity check.
- Context is fixed. Summary quality, later-work visibility and the trim rules themselves are outside this evaluation.

Tests are in `test/decision-eval.test.ts` and need no key or network: `bun test packages/opencode-plugin/test/decision-eval.test.ts`.

# Architecture inventory and decisions

This inventory and decision log covers the architecture refactor from baseline
commit `d330a848264e1b8223df5936d71746c271c58af7`. The state-machine table
describes the current candidate; issue descriptions identify baseline behavior
where relevant. The behavior contracts in [Core](Core.md) and
[OpenCode](OpenCode.md) govern implementation. The inventory and implemented
decisions do not by themselves establish qualification. Verification evidence
is candidate-specific; commands and verification owners are listed below.

Paths below are relative to `packages/opencode-plugin/`, except `docs/`,
`scripts/`, `.github/`, and root configuration files. Historical paths and line
numbers are baseline citations: locate the symbol again after changes. A **hidden link** is an ownership or state dependency that
must move together; a **needless link** adds coupling without owning the rule;
**surplus** is a candidate for removal; a **special case** has a distinct
contract. Dispositions are **planned**, **retained**, **rejected**, or
**question requiring evidence**. Each group names its verification owner; the
final integrated checks are listed at the end.

## State-machine inventory

The arrows describe observable stages, not necessarily persisted enum values.
For example, a successor is like a copy being checked before it is handed to
the user: a failed check does not authorize deleting a copy someone else changed.

| Machine and owner                                                                                                          | States, transitions, failure and requirement                                                                                                                                                                                                                                                                                                        |
| -------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Source admission: `src/compact/successor.ts`, `src/v2/session.ts`                                                          | Read root and live tree → strict at-rest or refusal; recovery instead collects recoverable stored records after the same live checks. Busy, pending questions, revert and compaction still refuse. Never infer abandonment from idle or age; recheck before ready.                                                                                  |
| Successor: `src/compact/successor.ts`, `src/v2/session.ts`                                                                 | Prepared edits → create → replay/import → cache and stats → source and candidate recheck → ready attempt → verified; failure before ready can lead to deleted, absent, or kept, while uncertain ready is kept. An absent candidate is not proof of deletion or cache ownership. Source remains unchanged.                                           |
| Exclusivity: `src/policy/successor.ts`, `src/compact/successor.ts`                                                         | Free → one process owns source operation → released. Another in-process operation refuses; another process can still race, so fresh comparisons are not a lock.                                                                                                                                                                                     |
| V1 replay: `src/compact/fork.ts`, `src/compact/compact.ts`, `src/retention/summary.ts`                                     | Plan → native fork cut → empty check → ordered replay → verify fork and source; each boundary history read retries once only on rejection or timeout, not invalid response. The owning request deletes a partial fork and names it if that fails. IDs and links must survive rewriting in order.                                                    |
| Background delivery: `src/policy/recovery.ts`, `src/compact/successor.ts`, `src/v2/session.ts`                             | Launch → owed → one later matching delivery or explicit abandoned marker. FIFO matters: two launches need two deliveries and a delivery before a launch settles neither. Unsupported or active work refuses.                                                                                                                                        |
| Recovery: `src/magic-recover.ts`, `src/v2/recover.ts`                                                                      | Stored unfinished → native interrupted copy or unsupported refusal → complete model-visible recovery context → lifecycle recheck → verified successor. Recovery does not stop work or invent results; V2 subagent records have their documented separate scope.                                                                                     |
| V1 ownership: `src/compact/owned.ts`, `src/retention/summary.ts`, `src/compact/compact.ts`, `src/host-v1.ts`               | Fork ID known → registered before population → native compaction blocked → abort on failure and remove → unregister only on confirmed deletion. A fork whose deletion failed stays registered and is named.                                                                                                                                         |
| V2 ownership: `src/v2/owned.ts`, `src/v2/generation.ts`, `src/host-v2.ts`                                                  | Owned request registered → one permitted response → closed → removal settled; a failed removal fails and names the session, a session not found keeps its guard and is named only beside another failure, and both retain ownership. In-flight writes and cleanup must settle before command completion.                                            |
| Generation pool: `src/policy/generation-pool.ts`, `src/compact/compact.ts`, `src/v2/generation.ts`                         | Pending → at most five active → validated or failed → all active drain. First failure stops new admission; discard all results and report every undeleted ID. Each request owns one target.                                                                                                                                                         |
| Retention: `src/retention/retention.ts`, `src/retention/classifier.ts`, `src/retention/backends.ts`                        | No candidates → no key or request; otherwise key → summary → candidate requests, at most four active. A potentially retryable 400 pauses admission at headers until classified; eligible 503 waits without a slot. Terminal failure stops admissions, drains and discards all decisions. Only documented `--truncate` and OpenAI 503 retries apply. |
| Model selection and fit: `src/policy/reasoning.ts`, `src/policy/budget.ts`, `src/compact/batch.ts`, `src/v2/summarizer.ts` | Configured summarizer → optional preflight match → fit or configured fallback before any request; impossible configured fit refuses. No runtime model failover. Unknown reasoning replay stays unknown.                                                                                                                                             |
| Generation identity: `src/policy/generation.ts`, `src/compact/compact.ts`                                                  | Conservative marker → unique real nonce and exact prompt fit → marker insertion → untrusted response validated → marker discarded. A wrong count, ordinal, marker, image coverage or nonce leak refuses before successor creation.                                                                                                                  |
| Checkpoints and turns: `src/compact/plan.ts`, `src/policy/turns.ts`, `src/policy/protocol.ts`, `src/policy/lineage.ts`     | Stored history → validate native checkpoint → effective complete turns → preserve tail and exclude opaque summaries from targets. Invalid selected checkpoint refuses; native accounting has a separately defined view.                                                                                                                             |
| Media projection: `src/compact/media.ts`, `src/v2/media.ts`, `src/policy/media.ts`                                         | Stored image plus manifest → validated request-only placeholder when strictly smaller, otherwise original. This is pure derived state, not a persisted lifecycle. Changed image cohort fails closed; source file bytes remain stored.                                                                                                               |
| Replay attribution: `src/compact/reasoning.ts`, `src/v2/reasoning.ts`                                                      | Index source units → attributable estimate or unknown → cost per selected subset. Repeated or incomplete units cannot be counted as certain; V2 reports unknown. Stored pruning and replay estimates stay separate.                                                                                                                                 |
| Omission cache: `src/storage/omission.ts`, `src/storage/store.ts`                                                          | Planned ID → queued per-key mutation and persisted cache → authorized, integrity-checked read; invalid or ambiguous legacy data is quarantined/refused. Failure must not leak paths or content.                                                                                                                                                     |
| Stats: `src/storage/stats.ts`, `src/storage/store.ts`, `src/stats/events.ts`, `src/stats/pricing.ts`                       | Absent or saved totals → serialized completion update or successor derivation → atomic write and notice. Pruning, media projection and cached-read accounting remain separate. Invalid saved totals stay unchanged and stop commands that need them.                                                                                                |
| Feedback and navigation: `src/toast-v1.ts`, `src/tui-v1.ts`, `src/tui-v2.ts`                                               | Progress → terminal feedback; a late successful progress settlement triggers best-effort terminal reassertion. Ready successor event → navigate once only if TUI still shows source. Toast failure cannot change the operation.                                                                                                                     |
| Compaction evaluation: `eval/compaction/run.ts`, `ledger.ts`, `artifact.ts`, `scratch.ts`, `report.ts`                     | Run-owned root/artifact → attempts and sessions → storage and cleanup events → report; leftovers remain identifiable. Cleanup needs ownership and source/successor dependencies, and reports must account for unresolved evidence.                                                                                                                  |
| Decision evaluation: `eval/decisions/runner.ts`, `providers.ts`, `report.ts`                                               | Planned → paid attempt → recorded outcome or interruption → validated report. Paid records and frozen cohort contracts must survive interrupted work without overwriting evidence.                                                                                                                                                                  |
| Release: `scripts/release-opencode.sh`, `.github/workflows/publish-opencode.yml`                                           | Local version/commit/tag → push → remote checks → npm publish → GitHub release. An interrupted phase requires inspecting existing remote and local evidence before resuming; a tag must identify the committed manifest.                                                                                                                            |

## Decisions and issue catalog

Each decision appears before the issues it addresses. Implemented dispositions
describe the candidate; historical observations identify the behavior that
motivated the decision.

### Leaf boundaries

**Decision — implemented:** Move toast effects out of generic utilities, put the
tokenizer in a leaf, and give shared budget, constants, summary envelope and
boundary rules their actual policy owner. Native manifest parsers remain host
adapters. Move `StoredHistoryCount` from the formatter to its data owner. Fold
`isRecord`, JSON, plural and route helpers only when their semantics and owner
actually match. The savings gate in successor is a relocation lead, not a new
framework. Keep SDK wrapper entries and root exports public; add no dependency
or blanket directory rewrite.

- **LEAF-CYCLES — implemented:** `src/util.ts` imports nothing; the OpenCode 1
  toasts moved to `src/toast-v1.ts`, and the unused `onDeadlineTimerChange`
  parameter of `attemptWithinDeadline` is gone. `src/util.ts` now holds the
  one copy of the non-array and plain record checks, the exact-key check and
  the JSON form used to compare records, replacing local copies in
  `src/retention/`, `src/policy/reasoning.ts`, `src/policy/generation.ts`,
  `src/compact/media.ts`, `src/compact/successor.ts` and
  `src/v2/{session,generation,owned}.ts`. Plural and route helpers were not
  examined.
- **LEAF-MEDIA — implemented:** `src/stats/tokenize.ts` holds only the exact
  tokenizer primitives, including the yielding memoized count. OpenCode 1
  canonical accounting moved to `src/compact/accounting.ts`, and
  `StoredHistoryCount` with the shared persisted/projected combination to
  `src/stats/stored-history.ts`. Shared budgets and batch planning moved to
  `src/policy/budget.ts` (OpenCode 1 estimates stay in `src/compact/batch.ts`);
  shared prompts, notices and the summary capsule and boundary metadata with
  their checks to `src/policy/protocol.ts` (OpenCode 1 part builders in
  `src/compact/capsule.ts`); and `getCompactionCount` to `src/policy/lineage.ts`.
  The V1 v2/v3 and V2 v3 manifest parsers stay in their adapters.
- **Leaf-boundary stage evidence:** at that stage, a runtime import scan of `src/` found one
  import cycle, `src/v2/generation.ts` ↔ `src/v2/owned.ts`, which belongs to
  the request lifecycle decision; the baseline also had the two cycles above.
  `bun build --metafile` shows that `src/host-v2.ts`, `src/tui-v2.ts`,
  `src/tui.ts`, `src/util.ts`, `src/stats/tokenize.ts` and the shared policy
  modules load neither `src/api.ts`, `src/compact/` nor
  `@opencode-ai/sdk/v2/client`; OpenCode 2 still loads the shared command
  surface `src/commands.ts`. `src/policy/successor.ts` still imported the OpenCode 1
  `ForkFailureDetail` type, at type level only, until SUCCESSOR-BOUNDARY. Format check, two-pass
  typecheck, lint, the full plugin suite (1,180 tests, 18,920 expectations) and
  both offline eval checks passed under scratch XDG. Packed-host qualification
  runs on the complete refactor candidate.

### Generation data and fit

**Decision — implemented; GEN-REPLAY retained:** Generation and retention
return their summaries, reasoning reports and replay context instead of
writing them into caller options. The summarizer catalog is resolved once per
command, on first need. A fit is a typed outcome, and only a capacity refusal
selects the `capacity` fallback. Each request plans exactly one target, keeps
the mandatory check of the request without its target, and still validates the
untrusted response count.

- **GEN-DATA — implemented:** `generateCompactionSummaries` returns
  `GenerationResult` (one summary per target, the turn reports and the replay
  context), `generateV2Summaries` returns its summaries and reports, both
  session summaries return `SessionSummary` (text and report), and
  `decideRetentionWith` returns `RetentionDecision` (plan and the reports of
  the summary it made, none without candidates). `SummaryReasoningOptions` and
  `ReasoningReports` are gone; each command concatenates the retention and
  turn reports for its `Reasoning` line, in the old order.
  `executeMagicCompact` returns a typed `CompactionOutcome` (`noop` or
  `compacted`) from its exclusive section instead of assigning outer variables
  read with non-null assertions. `summarizerCatalogLoader`
  (`src/compact/compact.ts`) reads the agent list and the model list each at
  most once per command, when a summary request first needs them, and is
  shared by the retention summary and every turn request; its
  `requiresImages` argument only chooses the failure text. Heuristic trim, a
  retention call without candidates and a compaction without targets read
  neither list, and `decideRetentionWith` still checks the selected key before
  it asks for the summary. OpenCode 2 compaction now resolves its summarizer
  lazily too, so a missing key fails before any summarizer read on both hosts;
  it previously resolved the model first.
- **GEN-FIT — implemented:** `computeNativeBatchBudget`,
  `planNativeGenerationRequest` and the OpenCode 1 exact-request check return
  `Fit` (`src/policy/budget.ts`); a capacity refusal carries the configured
  model's existing fixed message, which `requireFit` or the caller's wording
  raises. `settleMatch` (`src/policy/reasoning.ts`) owns the matched decision
  for all four stages: it fits the would-be match with or without
  `--match-model`, so feedback can report it, turns only a refusal into the
  `capacity` fallback, lets any thrown error propagate, and returns the
  matched plan only under the flag. Each caller closes the fit over the same
  nonce as the configured plan. The four `hasPluginMessage` catches
  (`compact.ts`, `retention/summary.ts`, `v2/owned.ts`, `v2/generation.ts`)
  are gone. The OpenCode 2 25 MiB request-image limit is a capacity refusal,
  because the preceding turn joins only within a model's halo. Image
  capability is not part of the matched fit: the preflight already required
  image input over a history that holds the request's, and only `openai`
  origins are verified, so no xAI case is reachable for a matched model. A
  real request or response larger than its conservative estimate is a
  protocol defect and throws; the OpenCode 1 response-estimate breach used to
  show the output-capacity text. No reachable non-capacity plugin error was
  found in the four matched planning paths, so this change fixes the
  classification without a test-visible change of outcome; the planning
  nonce gives every byte its own token, which makes both estimate breaches
  and OpenCode 2's `Compaction generation request exceeded its planned size.`
  unreachable.
- **GEN-SHAPE — implemented:** `planCompaction` computes the targets once
  (`CompactionTarget`: the turn and its position in `sourceTurns`), and the
  stats check, retention candidates, generation, profitability and the
  compaction eval plan check read `plan.targets`. `summarizedTurns`,
  `sourceTurnStartIndex` and null summaries for opaque turns are gone.
  `sourceSnapshot` remains only for drift and fork-shape comparisons; native
  consumers read `sourceMessages` and `sourceTurns`, and nothing casts the
  snapshot to records. Candidates are built with their ordinal and planning
  marker, and the real marker is rendered once per request. The budget API
  takes one candidate (`explicitRequestTokens(null)` measures the request
  without its target), `buildGenerationPrompt` and
  `parseGenerationResponse` take one target, and the response must still hold
  exactly one result. `TrimPlan` holds only `trimmedTurns`. An OpenCode 2
  planned batch keeps only its target's media manifest, and its request's
  unprojected images are listed once for the image-capability check, which
  still runs before the conditional 25 MiB refusal. The generated
  part-ID check now covers the whole source rather than the range and the
  next turn, which only rejects more. A compaction whose range holds only
  summary turns now stops right after planning; under `--match-model` its
  toast no longer ends with a `0 of 0` reasoning line, as on OpenCode 2 and
  for a range with no eligible turn.
- **GEN-ANCHOR — implemented as shared framing; host text choice kept:**
  `globalTurnIndexPrompt` (`src/policy/generation.ts`) owns the framing both
  hosts share: the index preface and, for each turn, the first line (`\r\n`,
  `\r` or `\n`) of the text its host chose, capped at 160 code points. Each
  host keeps the baseline choice of that text. OpenCode 1
  (`buildGlobalTurnIndexPrompt` in `src/compact/compact.ts`) takes the first
  non-synthetic, non-ignored text part across the turn's user messages, even
  an empty one, so a later message supplies it only when the earlier ones have
  no such part. OpenCode 2 (`anchorText` in `src/v2/compact.ts`) takes the text
  of the turn's first user message, even an empty one. An intermediate
  candidate skipped empty texts on both hosts; that was an unapproved prompt
  change and is reverted. The hosts still differ for a turn whose first user
  message has no ordinary text part, which OpenCode 1 fills from a later
  message and OpenCode 2 leaves empty; unifying them would change the prompt
  and needs its own decision.
- **GEN-REPLAY — retained, with evidence:** an OpenCode 1 generation request
  is cut from a fork of the complete stored source and isolated to the target
  and at most one preceding turn, neither holding a native checkpoint
  artifact, so native conversion sends exactly those records. Its budget and
  report attribute replay over the complete stored history
  (`indexReplay(plan.sourceMessages)`). The retention summary request sends
  the native model-visible view and attributes over it, and profitability,
  the gate and the notice measure the native view. Each basis contains every
  record its stage sends, so no stage attributes a unit whose item or payload
  repeats within what it sends (`test/reasoning.test.ts` "keeps one
  attribution for every sum over a history", `test/compact.test.ts` "keeps a
  preserved repeat of a summarized reasoning item out of the savings gate").
  Attributing over a request's own retained turns would drop the other copies
  and attribute a unit of unknown provenance, so the bases are not unified.
  They still differ for a repeat in a record hidden behind a native
  checkpoint: the generation estimate keeps such a target unit unknown, while
  the native-view stages do not see the hidden copy. No retained history
  shows such a repeat; the hypothesis that a native-view stage can attribute
  an ambiguous unit stays open and is not fixed here. OpenCode 2 attributes
  no replay cost, so it has no basis question. Older OpenCode 2 boundaries
  are also retained: each compaction inserts its boundary before the next
  turn's first user message, so an older one lies in the records of the last
  summarized, now opaque, turn or of the preserved tail, which are never
  edited, and OpenCode 1 keeps its boundary parts in user messages, which are
  preserved exactly. `test/v2.test.ts` "marks only the target, keeps the
  preceding turn as context" shows an older boundary travelling with the
  preceding turn. No contract violation was found.
- **Evidence:** `test/compact.test.ts` adds one catalog read shared by the
  retention summary and both turn requests, no read on either no-op, and none
  before a missing selected key; `test/v2.test.ts` adds OpenCode 2 compaction
  without the key resolving no summarizer. Each new regression was seen to
  fail without its behavior: the shared read against an unmemoized loader and
  the key cases against an eager catalog or summarizer read. The anchor tests
  in `test/compact.test.ts` and `test/v2.test.ts` compare each host's whole
  assembled prompt, empty anchor included, and fail against the intermediate
  skip-empty rule.
  Existing capacity, variant, unverified-route, nonce, halo, opaque-turn and
  response-count cases pass with API adaptations; the post-request count test
  now sends an empty response, because the internal count it faked no longer
  exists. Format check, two-pass typecheck, lint, the full plugin suite (1,198
  tests, 19,022 expectations) and both offline eval checks (3 compaction and
  17 decision cases) passed under scratch XDG. Request lifecycle work (pools,
  registries, error guidance, classifier scheduling) is not part of this
  group, and packed-host qualification runs on the complete refactor
  candidate.

### Request lifecycle

**Decision — implemented:** OpenCode 1 summary and generation forks share one
registry, and each request (a generation batch or the retention summary) owns
its fork or temporary session from the moment its ID is known, through one
linear lifecycle with one cleanup path. Both hosts run per-turn generation
through one bounded pool that stops admission as soon as a batch reports its
failure, before its cleanup. Retention keeps its own scheduler, with one record
per planned request. A session summary failure is a typed
`SessionSummaryError`, and its advice is formatted from the invocation instead
of rewritten into a message. Every failure names each fork or temporary
session whose removal it could not confirm.

- **LIFE-FORK-OWNER — implemented:** `forkSession`, which populated a fork and
  deleted it itself on failure before returning, is replaced by
  `createForkCut` (`src/compact/fork.ts`): it validates the planned history,
  cuts the native fork before the first planned message and returns it as a
  `ForkCut`, its ID with a `populate` step. `populate` takes the caller's
  control deadline, checks that the cut holds no message, replays and
  verifies the history and deletes nothing; until it succeeds the fork is not
  known to be empty. The generation batch
  and the retention summary, its only two production callers, register the
  fork at once, run `populate` inside their own failure handling and delete
  the fork through their existing cleanup. A population failure is therefore
  observed, and generation admission stopped, before the partial fork's
  deletion starts; previously a delayed partial-fork deletion let a free
  worker start a sixth batch. A partly populated fork whose deletion fails now
  keeps the native compaction guard; previously it was never registered.
  `ForkCleanupError` and the second, helper-owned partial-fork deletion are
  gone. `loadHistory`, `replayHistory` and `planHistory` stay public for
  successors and the compaction evaluation.
- **LIFE-V1 — implemented:** `src/compact/owned.ts` is the one registry of
  owned OpenCode 1 forks. `registerOwnedFork` runs as soon as a fork's ID is
  known, before it is populated; `releaseOwnedFork` runs only after the host
  confirms the deletion, so a fork whose deletion failed stays guarded. The
  `experimental.session.compacting` hook calls `blockNativeCompaction`, which
  throws one fixed error and sets `nativeCompactionBlocked` on the fork, and
  `chat.params` looks up `isOwnedFork`. The generation request reads that flag
  instead of trusting the host to keep a `ContextOverflowError` name, and
  reports trim-first overflow guidance whether the prompt then fails or
  returns; the session summary reads the same flag. The registry stores no
  fork kind, because both hooks treat summary and generation forks alike.
  `requestCompactionGeneration` (`src/compact/compact.ts`) is now one
  sequence: create the fork cut, register it, populate it, isolate it
  through `isolateGenerationFork`, prompt once through `promptGenerationFork`,
  abort only when a failure follows the prompt, and delete. `control` runs each
  control request under the deadline and turns a timeout or rejection into
  that stage's `GenerationLifecycleError`, and `checkedHistory` does the same
  for a failed fork-history check. This replaces the nested `operationError`
  ladder and its repeated deadline conversion. The fake
  `{ name: "MessageOutputLengthError" }` throw is gone: a response cut at its
  output limit without structured output raises the output-capacity failure
  directly. `GenerationLifecycleError` keeps `stage`, `status`, an ID-free
  `reason` and the forks it left; the unused `sourceSessionID` and `kind`
  fields, `sanitizeGenerationFailure` and `combineLifecycleErrors` are gone,
  and a failure with its abort and deletion failures is one flat aggregate.
- **LIFE-POOL — implemented:** `runGenerationBatches`
  (`src/policy/generation-pool.ts`, with `NATIVE_BATCH_MAX_CONCURRENCY`)
  takes the planned batches and runs at most five, keeping its own indexes
  for batch order. A batch calls `stop` when it observes its
  failure, before it aborts, deletes or removes anything, and then rejects;
  no batch starts after that, every started batch drains, failures come back
  in batch order and results only when every batch succeeded. A stop without
  a failure is an internal defect and throws. OpenCode 1 and OpenCode 2 keep
  their own session lifecycles, error types and wording. OpenCode 2 used to
  record a failure only after its session's removal settled, so a worker that
  finished meanwhile could start another batch; `runBatch` now stops first.
  The OpenCode 2 cycle `src/v2/generation.ts` ↔ `src/v2/owned.ts` is gone:
  summarizer resolution, the `--match-model` helpers, `V2SummarizerModel`,
  `V2_IMAGE_MODEL_REQUIRED` and `estimateV2HistoryTokens` moved to
  `src/v2/summarizer.ts`, which both modules import. Retention does not use
  this pool.
- **LIFE-RETENTION — implemented:** `runAttempts`
  (`src/retention/classifier.ts`) takes the planned `RetentionRequest`s and
  keeps one `PlannedRequest` per request (the request, its ordinal, its next
  summary variant, which is also the one its answers used, its HTTP 503
  retries spent and its outcome) and one `Attempt` per HTTP request (its
  summary variant, whether its 400 holds a pause and the wait before its 503
  retry), replacing the parallel `outcomes`, `results`, `retried` and
  `errors` arrays. Each result carries its request, so `classifyRetention`
  maps answers to candidates without a positional join. Admission is paused while any
  attempt in flight holds a pause, which is derived from the at most four
  attempts in flight instead of a hand-kept counter. Retries are a stack
  taken ahead of an index of unstarted requests, which keeps the previous
  order (newest retry first) without rescanning a queue. The first terminal
  failure is still the first attempt whose status is terminal at its headers,
  or else the first to settle with a failure. A wait that rejects ends like
  one that resolves, as before, and this is now documented on `retryWait`.
  Pausing, 503 waits without a slot, the shared 503 budget, the single
  excerpt retry, the 30-second `Retry-After` bound, cancellation and the
  request counts are unchanged.
- **Retention guidance — implemented:** `SessionSummaryError`
  (`src/policy/retention.ts`) carries a typed `failure`: one stage with its
  optional status, overflow and fork-history reason, where a `delete` stage
  names the fork or temporary session whose removal was not confirmed;
  several stages with the cleanup of the fork or temporary session; or a
  refusal the plugin wrote before the summary asked the host for any
  session. Its message is that failure's text (each stage with its status and
  fork-history reason, led by the too-large explanation when a stage outgrew
  the model, also beside cleanup stages), then the unremoved session, then
  `retentionFailureGuidance` for the invocation it was built for.
  `decideRetentionWith` turns every summary failure, and every other refusal
  the plugin raised while summarizing, into a `SessionSummaryError` for the
  invocation; any other error passes unchanged and is reported as
  unexpected. `addOwnCommandGuidance` and `RETENTION_FAILURE_GUIDANCE` are
  gone, and adapters no longer append advice or attach causes. OpenCode 2's
  summarizer resolution (a missing agent model or an unlisted model) used to
  fail without retention advice; it now ends in the person's own command. The
  error returned for an invocation is a new object, not the adapter's.
- **LIFE-CLEANUP — implemented:** every failure names each fork or temporary
  session whose removal it could not confirm, as the original requirement
  asks. `UnremovedSession` (`src/policy/feedback.ts`) is the ID with its
  removal outcome, `failed` (refused or unknown) or `not-found` (the host
  reported no such session, which does not prove it is gone), and
  `LeavesSessions` is implemented by `GenerationLifecycleError`,
  `V2GenerationError` and every `SessionSummaryError` variant, with a
  `reason` that names no session. `distinctReasons` reads that reason, so a
  capped reason never carries an ID, and `unremovedSessionsText` names each
  ID once after the reasons, in batch order, with one sentence per run of
  adjacent sessions that share an outcome. A refused OpenCode 1
  generation-fork deletion now reports `Compaction generation fork deletion
failed. Fork <id> could not be deleted.`, and the session summary on both
  hosts names its fork or temporary session. OpenCode 2's
  `removeOwnedSession` returns the session as an `UnremovedSession`, or
  nothing after a confirmed removal, so neither caller maps the host's answer
  itself. A removal that failed or whose result is unknown fails the
  generation batch or summary and names the session. A session the host
  reports missing keeps the closed guard and, as before, is not a cleanup
  failure: after a validated response or summary the request still succeeds.
  When the request already failed (after a failed import, for example), the
  failure now also names the session as not found, which neither says that it
  exists nor that it was deleted; before, it named nothing. Nothing is
  retried. The aggregate on OpenCode 1 now says `Forks <ids> could not be
deleted.` instead of `Delete partial forks <ids> manually.`, since prompted
  forks are named too. A summary that outgrew the model and whose cleanup
  also failed leads with the too-large explanation before the stage list;
  before, the combined message listed only the stage labels.
- **Question closed:** a blocked native compaction cannot follow a captured
  generation response. In `packages/opencode/src/session/prompt.ts` at
  830d5eb535, when the `StructuredOutput` tool captured a value, the prompt
  loop stores it on the message and returns `break` before it acts on a
  `compact` result; a compaction at the top of a loop iteration runs before
  that iteration's model request. Treating a blocked compaction as an
  overflow therefore never discards a captured result.
- **Evidence:** `test/compact.test.ts` adds a native compaction the hook
  blocked, reported as an overflow when the prompt then fails with a plain
  error and when it returns, the fork registered during its prompt and
  released after deletion, and a fork whose deletion failed kept guarded.
  `test/v2.test.ts` adds no sixth batch while a failed batch's removal is
  still running, and task-aware `/magic-compact` and `/magic-trim` naming a
  missing summarizer model with the person's own heuristic command.
  `test/retention.test.ts` adds a 503 retry queued while every slot is busy
  sent ahead of unstarted requests and a rejected wait that still retries,
  and its summary-advice case now covers a stage failure, a plugin refusal, a
  partial fork that stays named and an unexpected error that passes
  unchanged. Each new regression failed with its behavior removed:
  signalling OpenCode 2 failures after removal (6 imports instead of 5),
  ignoring the blocked-compaction flag, releasing the guard before deletion,
  taking unstarted requests before retries, treating a rejected wait as no
  resume, and passing plugin refusals through unconverted. A runtime import
  scan finds no cycle in `src/` (the baseline shows the OpenCode 2 one), and
  a metafile build of `src/host-v2.ts` still loads neither `src/compact/`,
  `src/api.ts` nor `@opencode-ai/sdk/v2/client`. Format check, two-pass
  typecheck, lint, the full plugin suite (1,206 tests, 19,067 expectations)
  and both offline eval checks (3 compaction and 17 decision cases) passed
  under scratch XDG. Packed-host qualification runs on the complete refactor
  candidate.
- **Remediation evidence (LIFE-FORK-OWNER, LIFE-CLEANUP):**
  `test/compact.test.ts` adds no sixth batch while the deletion of a fork that
  failed to populate is held, and a partly populated and a prompted fork whose
  deletions failed, both named and both still blocked by the real compacting
  hook; it now names the fork in a single refused deletion and in the
  aggregates, where five failed batches keep each fork-history reason within
  the cap and name all five forks once. Its chronology cases run on
  `createForkCut` and `populate` and show that population deletes nothing.
  `test/retention.test.ts` names a partly populated summary fork with its
  replay reason and keeps it guarded through the hook, and names the fork in
  the provider-and-delete aggregate. `test/v2.test.ts` names the temporary
  session after a refused removal in generation and the summary, and names a
  failed import's session as not found in both. Each new regression failed
  with its behavior removed: no stop before cleanup (a sixth fork),
  registering only after population or releasing the guard when population
  fails, a deletion failure without its fork, and `absent` treated as
  removed. A metafile build of `src/host-v2.ts` still loads neither
  `src/compact/`, `src/api.ts` nor `@opencode-ai/sdk/v2/client`. Format
  check, two-pass typecheck, lint, the full plugin suite (1,207 tests, 19,075
  expectations) and both offline eval checks (3 compaction and 17 decision
  cases) passed under scratch XDG.
- **Remediation evidence (cleanup results):** `test/v2.test.ts` keeps a
  validated generation response and a validated retention summary as
  successes when the host no longer finds their session, with its guard
  still closed, and its five-batch aggregate now mixes a failed import with
  refused removals and names the sessions in batch order. The failed-import
  cases still name the session as not found. `test/retention.test.ts`
  reports a context overflow (HTTP 413) with a refused fork deletion with the
  too-large explanation, the stage list and the fork. Each of those four
  regressions failed on the previous candidate (a not-found session failing a
  validated request, failed sessions grouped before the not-found one, and
  the combined message without the explanation). The refusal variant of
  `SessionSummaryFailure` no longer carries sessions: every summary failure
  after a session request is a stage failure. `populate` has no default
  deadline; both production callers pass the command's. Format check,
  two-pass typecheck, lint and the full plugin suite (1,210 tests, 19,086
  expectations) passed under scratch XDG.

### Source and successor ownership

**Decision — implemented:** Each source reader returns the source with its own
recheck, and the snapshot it compares follows the reader's lifecycle. Pure FIFO
delivery settlement lives in recovery policy while adapters extract native
events. Candidate cleanup is an explicit outcome that includes `absent`, and
host-specific trust checks and recovery scope stay in each adapter.

- **SOURCE-RECHECK — implemented:** `readSourceAtRest` and
  `readRecoverableSource` (`src/compact/successor.ts`) return a
  `CheckedSource`, and `readV2SourceAtRest` and `readV2RecoverableSource`
  (`src/v2/session.ts`) a `CheckedV2Source`, whose `recheck` reads the source
  again with the same lifecycle (`SourceLifecycle` in `src/policy/successor.ts`)
  and compares it with a snapshot captured at the first read. On OpenCode 1 the
  snapshot is the continuation state and the normalized history; recovery adds
  whether each assistant message and timed part has finished. On OpenCode 2 it
  is the continuation state and the complete history; recovery adds the
  subagent records. Each snapshot is a copy (`structuredClone` on OpenCode 1,
  `jsonForm` on OpenCode 2), so a nested continuation field that the host or
  the caller edits in place after the first read is still drift.
  `createSuccessor` and `createV2Successor` call
  `source.recheck()`, so the six commands no longer pass a reader and
  comparator, and `assertSourceUnchanged`, `assertRecoverySourceUnchanged` and
  `assertV2SourceUnchanged` are gone. Refusal order and the per-host subagent
  scopes are unchanged; `inspectV2Source` takes the lifecycle instead of a
  boolean. `assertV2HistoryFinished` is the one OpenCode 2 history check used
  by strict admission for every session in the tree and by recovery for its
  own result, which now refuses with the shared unfinished-history text
  instead of an unexpected error if recovery ever left a record open.
- **SOURCE-DELIVERY — implemented:** `settleDeliveries` in
  `src/policy/recovery.ts` settles launches against deliveries for both hosts:
  a delivery settles the oldest open launch of its job, an earlier delivery
  settles nothing, and a recovery mark settles only the launch whose own job it
  names. Each adapter still extracts its native launches and deliveries; OpenCode
  2 keeps its job namespace (`shell:<id>`, `subagent:<id>`) and still checks
  that settled and abandoned jobs have stopped. `RecoveryMarker` types the
  disjoint marker shapes `withRecoveryMarker` writes; the stored bytes are
  unchanged.
- **SUCCESSOR-CLEANUP — implemented:** `CandidateDisposal` in
  `src/policy/successor.ts` replaces the `discarded`, `readyAttempted` and dead
  `importAttempted` booleans with `deleted`, `deleted-with-residue`, `absent`,
  `kept` (with the check that kept it) and `ready-attempted`, carried in the
  failure's `details.cleanup`. An absent candidate (OpenCode 1 HTTP 404,
  OpenCode 2 `SessionNotFoundError`) is named as not found and its storage is
  left in place. Its `storageAttempted` flag records that the cache or stats
  write had started, not that a file exists, so the text says only that any
  such files were left in place. `SourceChangedError` states the drift alone;
  the candidate's outcome text owns every claim about the candidate (the
  baseline drift text also said that no successor was kept, which
  contradicted a kept or not-found candidate). `deleteCandidateStorage`
  reports a cache or stats file it could not remove instead of swallowing the
  error. A refused deletion keeps
  the candidate and is not retried. A rejected OpenCode 1 create has no ID to
  name, which `successorFailure` documents. Control reads keep their deadline;
  creation, import, replay and deletion stay directly awaited.
- **SUCCESSOR-BOUNDARY — implemented:** the lineage entry, successor metadata
  and preparing title are shared in `src/policy/lineage.ts` (OpenCode 2's
  `outcome` stays a host-only field in the same key order); the header fields
  each host compares stay native. `src/compact/successor.ts` no longer
  re-exports policy, so callers and tests import
  `src/policy/successor.ts` directly. `ContentFreeError` is generic over its
  detail, which removes the policy import of the OpenCode 1
  `ForkFailureDetail`, and lives in `src/policy/feedback.ts`, so
  `src/compact/fork.ts` no longer loads `src/policy/successor.ts` or its
  storage imports. OpenCode 1 host envelopes raise a local
  `HostResponseError` that keeps only the HTTP status, and OpenCode 2's
  `readHistoryOrRefuse` recognizes its own `SourceReadError` instead of a
  message prefix; the trust refusals and their fixed texts are unchanged. V1
  trim now compares projected media tokens with its projection as compaction
  and both OpenCode 2 commands do. `sessionRoute` moved to `src/compact/fork.ts`
  beside `HistoryRoute` and replaced three inline copies, the summarizer
  catalog reads' `sessionLocation` included; the duplicate `plural`
  in recovery policy now uses the one in `src/policy/successor.ts`. The
  recovery command's early `readStats` stays: it stops on invalid stats before
  a candidate exists, and the later successor derivation rereads the totals, so
  concurrent completions are not lost. The `/magic-stats` feedback gap is
  implemented with STATS-OWNER: on OpenCode 1 its argument, listener and
  execution failures were thrown to no client. Its argument error is now a
  plugin message, `preflight` toasts argument and listener failures through
  the input client, and `executeMagicStats` toasts execution failures through
  the verified listener. OpenCode 2 builds its result through `commandResult`,
  so a read failure can no longer return a message that holds the file path.
  The listener guidance now names Magic Stats.
- **Evidence:** `test/recover.test.ts` reads one OpenCode 1 source both ways
  and shows that only recovery's recheck sees a completion time the host
  stored, against what each read returned rather than the caller's copy.
  `test/compact.test.ts` adds a candidate the host no longer finds (named, not
  deleted), a deleted candidate whose omission cache could not be removed, and
  OpenCode 1 recovery marks settling only their own launch. `test/v2.test.ts`
  names a failed import's candidate as not found, keeps a missing candidate's
  saved cache and stats, names a stats file that could not be removed, and
  reports a refused removal as a kept candidate. Existing drift, FIFO, kept
  candidate and refusal-order cases pass unchanged. Format check, two-pass
  typecheck, lint, the full plugin suite (1,195 tests, 19,002 expectations) and
  both offline eval checks passed under scratch XDG. A metafile build of
  `src/host-v2.ts` still loads neither `src/compact/`, `src/api.ts` nor
  `@opencode-ai/sdk/v2/client`. Packed-host qualification runs on the complete
  refactor candidate.

### Stats and storage

**Decision — operator confirmed; implemented by STATS-OWNER:** An existing stats
file that fails parse or schema validation is preserved. Commands needing its totals stop with
a content-free diagnostic. Absence remains distinct from corruption; completion
events must not overwrite corrupt data. Known OS codes may be reported without
paths or file contents.

- **STATS-OWNER — implemented:** `src/storage/stats.ts` owns every stats read
  and write. `readStats` returns null only for a missing file, throws the
  content-free `InvalidStatsError` for invalid JSON or schema, and rethrows
  other read failures with their system code. Writes rename a same-directory
  temporary file, created owner-only and given the replaced file's permission
  bits (now through `replaceJSONFile`, see STORAGE-WRITE), so a first write is
  private and a replacement keeps its file's mode
  (the baseline wrote in place, which kept an existing mode; an intermediate
  rename replaced a `0600` file with `0644` under umask `0022`).
  `writeStats`, `deleteStats` and the read-modify-write
  `recordAssistantCompletion` run serialized per stats file in this process,
  and a failed change rejects only its caller. `writeSuccessorStats` replaces
  the two successor derivations in `src/compact/successor.ts` and
  `src/v2/session.ts`; it reads the source in the source file's queue, so it
  sees a completion admitted before it, and leaves that queue before writing
  the successor (an intermediate version read outside the queue and copied
  totals that a held completion had not yet renamed into place). `copyStats`,
  `createEmptyStats` and the store's `readJSONFile`/`writeJSONFile` had no
  other callers and are gone;
  `src/storage/store.ts` keeps `pluginStorageDirectory`. `assertStatsUsable`
  checks the totals before model requests: compaction after its no-op exit
  unless every target already holds a summary, and task-aware trim when it has
  candidates, so trim stops even if retention would keep every candidate.
  `successorFailure` forwards the diagnostic, with `Details: stage stats.`
  `recordCompletionEvent` in `src/stats/events.ts` is the completion boundary
  for both hosts: it ignores only `InvalidStatsError`. OpenCode 2 keeps its
  per-event catch so one failed update cannot end its event subscription. The
  v1 shape, optional media fields, V2 namespace and the paths and exports packed
  eval subjects load are unchanged. No schema migration, cross-process lock or
  age-based recovery was added.
- **Stats evidence:** `test/compact.test.ts` covers concurrent distinct and
  repeated completions with no lost increment, invalid JSON and schema kept byte
  for byte with no path or content in the diagnostic, a compaction no-op that
  still succeeds, a read failure reported as `EISDIR`, a failed write that
  keeps the file and lets the next completion count, the modes of the
  temporary and replacement files on the real file system, and a successor
  copy that waits for a completion held before its rename. `test/v2.test.ts` covers
  `/magic-stats` with invalid stats and `EACCES`, compaction stopped before any
  model request, and the successor `stats` stage discarding its candidate.
  Format check, two-pass typecheck, lint, the full plugin suite (1,188 tests,
  18,979 expectations) and both offline eval checks passed under scratch XDG.
- **STORAGE-TEST-API — implemented:** fixtures write omissions through
  `planOmission` and `commitPlannedOmission`, read them back through
  `readOmittedContent` and check persisted bytes in the cache file itself.
  `allocateOmission` (with its retrying collision policy), `readCache` and
  `writeCache` are gone; the collision test now shows that a duplicate planned
  Content ID is refused and the committed entry kept byte for byte.
  `createCompactionPlan`, `createTrimPlan`, `loadMessages`,
  `assertCompactionSourceUnchanged` and `NativeCheckpointChangedError` are
  gone: `planCompaction` already ran the checkpoint inspection that the error
  wrapped, so the wrapper was unreachable outside tests. `compactionTargets` is
  private. `countSessionTokens`, `countSessionTokenProjection` and
  `countCanonicalContentTokens` are gone; a stored total is
  `countModelFacingContentTokens(...).tokens + projectedMediaTokens`.
  `src/magic-compact.ts` and `src/magic-trim.ts` no longer reexport command
  text. `cachePath`, `deleteCache` and every export the packed eval subject
  loads are unchanged.
- **STATS-ISOLATION — implemented:** every `test/trim.test.ts` case runs in its
  own scratch `XDG_DATA_HOME`, like the recovery suite, and both suites close
  their in-memory histories after each test. Per-ID cache and stats cleanup
  and `cleanupSessions` are gone.
- **STORAGE-WRITE — implemented:** `replaceJSONFile` in `src/storage/store.ts`
  is the one atomic file write for stats and the omission cache. It writes
  two-space JSON with a final newline to a same-directory temporary file
  opened exclusively (`wx`) and owner-only, gives it the replaced file's
  permission bits, renames it over the target and on failure removes only
  that temporary file. The omission cache had kept its own copy, which wrote
  the temporary file with the default mode, so under umask `0022` a first
  write and every replacement of a `0600` or `0640` cache became `0644`.
  Callers keep their validation, error semantics and per-key queues; written
  bytes, the cache and stats formats and versions, the V2 namespace and the
  exports packed subjects load are unchanged. `test/omission.test.ts` owns the
  cache's first-write mode, a kept `0640` mode, and a failed rename that keeps
  the old bytes and mode with no temporary file left; the stats mode test in
  `test/compact.test.ts` now runs through the same write.
- **Acceptance / verification:** `test/omission.test.ts`, command and V2 suites
  own corruption preservation, absent versus invalid, concurrent updates, cache
  integrity and compatibility. Place stats regressions in the existing owning
  suite after identifying its current cases.

### Compaction evaluation

**Decision — implemented for EVAL-RESIDUE, EVAL-ORDER, EVAL-PROTOCOL and
EVAL-PAID:** Derive live outstanding state from the existing ledger
events and make residue visible in report and exit status. Preserve records and
schema; use explicit per-attempt cleanup dependencies. Keep the independent
inspection oracle independent.

- **EVAL-RESIDUE — implemented:** `eval/compaction/ledger.ts` is the one owner
  of the run ledger and of what a run still owns. `outstanding` derives open
  sessions, intents and unknown attempts as before, and now also the storage of
  each successor the run removed (a `session-removed` after its last
  `storage-receipt-request`) with no `storage-removed` for that request,
  whether `storage-orphaned` was recorded or a crash came first. `residue`
  adds each open session's last retention reason. `confirmLateStorage`
  (`run.ts`) runs before every disposal decision and in `cleanup`; it appends
  the existing `storage-removed` event, with `reconciled: true`, only for a
  channel `control` line of phase `storage` with `removed: true` for the exact
  request and successor. Nothing removes storage files directly, so storage
  whose removal is never confirmed stays reported. Unconfirmed storage joins
  the run-wide block in `disposeLocations`, because the wrapper loaded at the
  project location holds its receipt; the block stays run-wide, not per root.
  `cleanupRun` returns it as `unconfirmedStorage`, with a `complete` flag that
  the CLI exit status reads, and `summarizeRun` reports `residue` from the
  ledger. Its public `cleanupRetained` list stays, derived from the same
  ledger sessions instead of the cleanup results frozen into attempt records.
  `Ledger.read` refuses, with an `EvalError` and without writing, any line
  that is not a complete JSON object with a string `kind`, a string `at` and
  a positive safe-integer `seq` above the previous line's, a torn last line
  included. `readRun` reads the ledger whenever it exists (a run directory
  without one recorded nothing, so no attempt may name a session), and
  `cleanupRun` reads it before it contacts the server. `append` refuses when
  the file's size differs from what the instance last read or wrote, takes
  the number after the highest one read or written, refuses once no safe
  number is left, and advances only after the write. This is a precheck
  against movement another writer already made, not a cross-process lock; no
  lock or stale-lock recovery was added. The `intentSeq` and `requestId`
  relationships and the per-entry `fsync` are unchanged. A sequence gap, which
  the previous writer left when an append failed and a later one succeeded in
  the same process, is valid: the review found that requiring consecutive
  numbers refused such intact ledgers, so the fix accepts them as written.
- **EVAL-ORDER — implemented:** `removeSource` (`run.ts`) holds the dependency
  for both `runAttempt` and `cleanupRun`: a source is removed only when the
  ledger no longer shows its attempt's successor as the run's, and is
  otherwise retained with the existing reason. `cleanupRun` groups open
  sessions by attempt and settles each attempt's successor first, replacing
  the global successor-first sort and the `keptSuccessor` set. That set missed
  a successor retained by the location, server or outcome check, so its
  source could be removed; reaching that needed a ledger whose successor and
  source locations disagree. The run-wide root blocking is unchanged.
- **EVAL-PROTOCOL — implemented:** `isRunnablePlan` (`corpus.ts`) is the plan
  test for `check` and `run`, so `run` now also refuses a plan without exactly
  one preserved turn. `JUDGE_AGENT` and `WITNESS_PREFIX` live in `host.ts`,
  the wrapper's `CONTROL` and `WITNESS` are imported from `wrapper/core.ts`,
  and the drivers and usage filter use the product's
  `MAGIC_COMPACT_SUMMARIZER_AGENT`; the wrapper keeps its own copy because it
  is packed standalone. `grade.ts` owns `disposition`, now a total map over
  `AttemptStatus`, and `gradedAttempts`, used by judge grading, manual import
  and reports. `Sha256Hex` (`corpus.ts`) replaces three copies of the digest
  pattern, and `attemptIdOf` and `caseOfAttempt` (`run.ts`) the attempt ID
  format. `classifyOutcome` no longer also accepts any text that starts with
  `No assistant turns`: `COMPACT_NOOP` has been exactly that no-op text since
  before the eval existed, so the pattern matched nothing more. `inspect.ts`
  gains no production import. No wrapper file changed, so the artifact key,
  the packed wrapper's internal paths and the route old `--subject` tarballs
  need (`src/storage/omission.ts`, `stats.ts`, `store.ts` and `v2StorageKey`
  in `src/v2/session.ts`) are as before. The wrapper's stats path
  (`${pluginStorageDirectory()}/stats/${key}.json`) still duplicates the
  private `statsPath` in `src/storage/stats.ts`; it is retained because
  published subjects export no stats path and the wrapper must load them.
- **Evidence:** `test/compaction-eval.test.ts` adds the storage derivation (an
  earlier refused request, a crash before the outcome, an orphan, and a
  confirmation of one request only); ledger refusals for a torn tail, a
  complete last entry without its newline, a duplicate `seq` (a skipped `seq`
  was also refused until the review remediation below made it valid), an
  array, a non-string kind and a blank line, each with the bytes unchanged;
  the append precheck after another instance wrote; unconfirmed storage that
  keeps the tree and fails `cleanup` until a late confirmation of exactly its
  receipt, where lines for another request, another session or a failed
  removal confirm nothing and a repeat records nothing more; the real CLI
  exiting 1 for unconfirmed storage with its root gone; a source kept while
  the location check retains its successor; a ledger the file system refuses
  to append to (mode 0444, `EACCES`), which stops `cleanup` after the removal
  it could not record and before any disposal, after which the next `cleanup`
  settles the session as absent; a torn ledger that refuses `cleanup` with no
  server request, and `report`; a report whose residue follows the ledger
  after cleanup while the attempt file stays byte for byte; and
  `isRunnablePlan` refusing a changed preserved-turn, target or candidate
  count. The existing orphan case now also shows the storage as still owned.
  Each new regression failed on the baseline candidate (12 failures in a run
  of 79 tests): the CLI exited 0 with unconfirmed storage, the source was
  removed beside a kept successor, a torn ledger raised a `SyntaxError`
  instead of a refusal, a duplicate `seq` was appended, and the other cases
  had no derived state to read. Format check, two-pass typecheck, lint, the
  full plugin suite (1,219 tests, 19,165 expectations) and both offline eval
  checks (3 compaction and 17 decision cases) passed under scratch XDG. A
  controlled-failure script ran the torn, refused-write, moved-ledger and
  unconfirmed-storage cases through real files and the real CLI against a
  loopback stand-in that only identifies itself, and read a ledger written by
  the previous writer unchanged; its results and the retained run directories
  are in `/tmp/opencode/architecture-refactor/eval-lifecycle/controlled-failure/`.
  Packed-host qualification runs on the complete refactor candidate.
- **EVAL-PAID — implemented:** `scoreNative` (`eval/retention/run.ts`) used to
  keep every paid snapshot's scores in memory and write them only after the
  last snapshot, so a later failure lost the earlier paid results. It now
  creates `<output>.partial/` before the first paid request and refuses to
  start when that directory or the output exists. After each snapshot's paid
  invocation, and before the next one starts, it writes everything scored so
  far, in the output's existing shape, to `snapshot-<k>-of-<n>.json` with an
  exclusive create. An interruption leaves the completed snapshots there and
  no output; nothing resumes or retries from them, and a new run needs a new
  output path. The output is written once, also with an exclusive create, after
  every snapshot, and the run then removes its own checkpoint files and the
  empty directory. The output's fields and serialization are unchanged. The key
  and fetch are passed in (`PaidDeps`); the command line passes
  `TYPESAFE_API_KEY` and global `fetch`. The historical proxy `run` paid across
  snapshots too, wrote its record only at the end and overwrote an existing
  record of the same name. It now follows the same protocol through the one
  writer both use (`paidOutput`), with its existing record shape at
  `runs/<name>.json` under the scratch directory: it refuses an existing
  record or checkpoint directory before reading the key, checkpoints after
  each snapshot and writes the record once. The scratch directory is passed
  in for tests; its default and the command line are unchanged.
- **Review remediation — implemented:** a review of the candidate above found
  that it refused valid gapped ledgers, tightened the decision command line,
  and left four lifecycle gaps. `receiptFor` (`ledger.ts`) now names the
  receipt whose storage a successor's end releases: the last request recorded
  as prepared (`storage-receipt`, a new additive event with the wrapper's
  answer), or, when no answer was recorded, the last request not recorded as
  refused. `outstanding` applies it to `session-absent` as well as
  `session-removed`, so a successor another writer deleted after its receipt
  keeps its storage owned until that exact request's removal is confirmed;
  `removeSuccessor` waits for that confirmation as it does after its own
  deletion. A successor already gone is no longer sent a new receipt request,
  which could never succeed: under the same server process,
  `settleGoneSuccessor` records it absent only from a prepared receipt, read
  from the ledger or imported from the channel line for that exact request
  and session (`importChannel`, which also records late storage
  confirmations), and otherwise retains it as uncertain. `unresolvedOwnership`
  (`ledger.ts`) reports each scratch root whose latest disposal kept it, with
  its reason and any sessions the host listed, until a later disposal removes
  it or `cleanup` records it gone (`scratch-absent`). It is reported apart
  from `residue` and never enters `isClear`, which now derives from every
  residue list, so the run-wide block is unchanged and a kept root does not
  stop the fresh checks that can clear it. `cleanupRun` inlines its former
  placement helper into `settle`.
- **Remediation evidence:** `test/compaction-eval.test.ts` adds a ledger with
  an earlier writer's gaps that reads, reconciles intents by their original
  numbers and appends after the highest; the last safe number refusing a
  further append; refusals for going back, zero, fraction, string and unsafe
  numbers; an append refused by the file system (mode 0444) leaving no gap; a
  successor another writer deleted after its receipt, through the wrapper's
  real control and storage, with its storage removed or, when the stats
  removal fails, still owned; the late-confirmation case for both removal and
  absence; a gone successor whose ledger has no answer, retained while only
  another request's or another session's receipt is on the channel and
  settled once that exact receipt and its storage line are; an end-to-end
  OpenCode 2 successor removal whose ledger record the file system refused
  (mode 0444), settled by the next `cleanup` with its storage and source; an
  invalid location marker, a session the run never created and a missing root
  marker, each reported as unresolved ownership and resolved by a later
  `cleanup` once fixed or gone; and `cleanupRetained` following the ledger.
  `test/retention.test.ts` adds the historical `run` checkpoints and retries,
  and its paid-scoring retry now counts requests through the same fake: with
  the claim moved after the first paid snapshot, the earlier non-counting
  retry still passed while the current one fails. `test/decision-eval.test.ts`
  replaces the tightened-grammar refusals with equivalent forms that plan and
  report exactly as their plain values. Each new regression failed on the
  reviewed candidate (10 failing tests of 145 in the two eval suites), and
  counting a kept root as owned failed the unresolved-ownership test at the
  `cleanup` that should resolve it. Format check, two-pass typecheck, lint, the
  full plugin suite (1,251 tests, 19,526 expectations) and both offline eval
  checks (3 compaction and 17 decision cases) passed under scratch XDG; the
  retention `metrics` output, the `eval:decisions check` output, three dry-run
  plans and the frozen artifacts are unchanged from the candidate. A
  controlled-failure script ran an earlier writer's gap, the refused successor
  record, an unaccounted session, the historical run and the decision command
  line through real files and the real CLIs; its results and retained run
  directories are in
  `/tmp/opencode/architecture-refactor/eval-remediation/controlled-failure/`.
  Packed-host qualification runs on the complete refactor candidate.
- **Closure remediation — implemented:** a second review and the packed-host
  qualification of 3298c50 found four gaps. (1) The wrapper replaces its
  in-memory receipt before it writes its answer, so when that write failed
  the receipt could be held under a later unanswered request while
  `receiptFor` kept naming the earlier prepared one, and the deletion's
  confirmed removal under the later request never settled the storage.
  `receiptFor` now returns the prepared request and every later request with
  no recorded answer (`uncertainRequestIds`, through `receiptRequestIds`);
  `outstanding` and `settleGoneSuccessor` (through `storageRecord`) and
  `importChannel` settle only on a confirmation for the exact session under
  one of them, and a refused request never counts. The ledger events and their links are unchanged. (2) `readRun`
  returned an empty ledger for a run without one, so a report claimed nothing
  owned; it now returns `null`, and `summarizeRun` reports `residue` and
  `unresolvedOwnership` as `null` with an "Ownership unknown" line, keeping
  `cleanupRetained` as a list. (3) The test tying the judge prompt's schema
  to `JudgmentSchema` had been reduced to a check against the exported
  constant; it now checks the prompt against the schema generated from the
  parser, and the constant is no longer exported. (4) On a cold OpenCode
  2.0.28 server, the first dry run refused in 2 of 2 qualification attempts:
  `setup` (`host-v2.ts`) stopped polling when two consecutive plugin lists
  were equal, and two empty lists were. Host source (2.0.28
  `core/src/plugin.ts`, `plugin/supervisor.ts`) starts a location's
  activation in the background and keeps the inventory empty until its first
  batch, the plugins it can load locally including the built-ins
  configuration cannot remove, is applied. `setup` then polled, with the same
  40 reads 750 ms apart, until the list was not empty, and records a wrapper
  whose state is `failed` as `(error)` (it read a top-level `error` field
  2.0.28 does not have). A non-empty list does not mean the activation has
  settled: packages still installing are applied in a second batch. The
  settlement remediation below replaced the poll.
- **Closure evidence:** these new regressions in `test/compaction-eval.test.ts`
  failed on 5538fa1: the uncertain-receipt matrix and a real-wrapper case
  whose second answer the file system refused (mode 0444, `EACCES`), the
  ledgerless report, and an OpenCode 2 setup after two empty plugin lists and
  with a failed wrapper. The schema test passes on 5538fa1, whose prompt is
  still generated from the parser, and fails when the prompt's schema comes
  from a changed shape. In
  the retained eval image (OpenCode 2.0.28, `--network none`, fresh server per
  container), the 5538fa1 harness refused the first dry run in 3 of 4 cold
  containers, and a logging-only copy showed two plugin lists with no entries,
  built-ins included, before the refusal. The fixed harness, with the same
  packed baseline and 3298c50 subjects, passed every dry run in 4 of 4 cold
  containers; its logging copy showed the same two empty lists, then 89
  entries with the wrapper active. Records are in
  `/tmp/opencode/architecture-refactor/eval-closure/`. A full packed-candidate
  qualification had not been rerun at that stage.
- **Settlement remediation — implemented:** a review of 5bd981f found two
  ways the eval could accept ownership or readiness it had not established.
  (1) A non-empty plugin list shows only the first activation batch. OpenCode
  2.0.28 applies the plugins it can load locally first and packages it must
  install afterwards (`core/plugin/supervisor.ts`), and only releases its
  readiness hold after both. `setup` (`host-v2.ts`) now calls `settle` first:
  it waits on the public `model.default` route, which awaits
  `Plugin.awaitActivation` for up to 5 s (`server/src/handlers/model.ts`,
  `server/src/location.ts`) and otherwise answers a `ServiceUnavailableError`
  with `service: "plugin"`. The public client throws that answer with its
  `_tag` and `service`; only that error is asked again, within one 30 s
  budget on a monotonic clock, and anything else propagates at once. The
  plugin, command, agent, config and MCP lists are then read once.
  `describeModel` and the public `resolvedAgent` settle their location the
  same way, so the judge preflight does not depend on call order. The 40 x
  750 ms poll is gone. Settlement is not a lock: a later config or package
  change can start another activation. (2) `gradeRunWithJudge` constructed a
  `Ledger` on a missing file and appended its judge root, so a later report
  read that new ledger as the run's and showed nothing owned. Every
  `runEval` since 70d9227 appends `run-start` with the nonce before
  `run.json` exists, so `recordsRun` (`ledger.ts`) accepts a ledger as the
  run's only when its first entry is that `run-start`. A judge `grade`
  refuses any other ledger before contacting the server; `cleanup` refuses a
  missing ledger before contacting the server and reports
  `ownershipKnown: false` and never `complete` for a ledger begun later;
  `summarizeRun` adds `ownershipKnown` and an "Ownership incomplete" line,
  keeping that ledger's residue listed. No ledger is migrated, dropped or
  renumbered. A package whose install failed has no plugin ID, so `setup`
  records it as `package:<target> (error)` instead of `undefined@...`.
- **Settlement evidence:** in `test/compaction-eval.test.ts`, the OpenCode 2
  driver tests drive `model.default` through the real public client over a
  scripted fetch: settlement after two plugin answers with an intermediate
  non-empty inventory, a settled failed install recorded, an unrelated 503,
  an untagged 503 and a 500 each ending setup after one request, a 30 s
  refusal on a controlled clock after six requests, and the judge reads
  settling first. Five of them fail against the 5bd981f driver. The saved-run
  test covering a missing ledger and a ledger an earlier `grade` began fails
  against the 5bd981f run, ledger and report modules (the old `grade`
  reached the server); a probe of the old `grade` created
  `scratch-root`, `location`, `scratch-removed` and the report then listed an
  empty residue. In fresh OpenCode 2.0.28 containers (`--network none`, mock
  provider, a global plugin served by an in-container registry after a set
  delay), the 5bd981f harness passed a dry run whose recorded plugins lacked
  the global package's failed install (2 s delay), and failed with the raw
  503 when the install outlasted the host's 5 s wait (8 s, 45 s). The fixed
  harness recorded `package:probe-late-plugin (error)` and passed (2 s, 8 s),
  refused after 30 s with nothing read (45 s), and, like the old one, refused
  two wrapper initializations when the late install succeeded (2 s). A cold
  OpenCode 1.18.30 dry run with the fixed eval tree passed. Records are in
  `/tmp/opencode/architecture-refactor/eval-settlement/`.

- **Acceptance / verification:** `test/compaction-eval.test.ts`, offline
  `eval:compact check`, and retention's offline metrics own ledger, cleanup,
  protocol and frozen-output invariants. No paid calls are needed for this audit.

### Decisions and retention evaluation

**Decision — implemented:** Shared decision vocabulary and validation each
have one owning module. Cohort-specific grammar, hashes, quantiles and the
frozen 0.80 thresholds stay separate from the runtime default and from each
other.

- **DECISION-CONTRACT — implemented:** `FAILURE_CATEGORIES` and
  `FATAL_CATEGORIES` (`eval/decisions/providers.ts`) are the one list of each.
  The outcome schema's `category` and the not-attempted `reason`
  (`stopped-after-<fatal category>`, a zod template literal) derive from them,
  and `executeRun` records every unadmitted request as not attempted after any
  fatal category. It used to do so only after `auth` or `rate-limit` by name,
  so a fatal category added to the set would have stopped admission and left
  the unsent requests reported as unknown and possibly billed. Each adapter
  carries its `defaultModel`, `modelHelp` and `bodyByteLimit`; the command
  line's provider-ID ternary is gone, and so are the `OPENAI_DECISIONS_MODELS`
  export and the `OPENAI_DECISIONS_ENDPOINT` and
  `renderPredicateInstructions` re-exports (tests import the last from
  `src/retention/backends.ts`). `thresholdRefusal` (`dataset.ts`) is the one
  threshold rule for `planRun`, `loadRun` and report overrides: above 0 and at
  most 1, and exactly 0.8 for a held-out dataset. `report --threshold 0.8` on
  a held-out dataset used to be refused and is now accepted, as
  `docs/DecisionEvaluation.md` says; `loadRun` now also refuses a manifest
  threshold outside the range. `returnedModels` and the report's `provider
reported` note are gone, because `loadRun` already refuses a record whose
  returned model differs from the manifest's. Numeric options keep their
  documented contract, ranges read through `Number`: the candidate briefly
  required plain decimals, which refused valid forms such as `8e-1`, `.8`,
  `+4`, ` 0.8` and `0x2`, and the review remediation restored `Number`, so
  only an empty, non-finite or out-of-range value is refused, before any
  request. The historical retention `run` parses its arguments once
  (`parseHistoricalRunArguments`) and refuses, before it reads the key, a
  missing name or split, a batch that `Number` does not read as a positive
  integer (the classifier's own rule, which used to apply only after the key
  was read), an unknown context, or an unknown order, which it used to record
  as given and run chronologically. `executeRun` builds the not-attempted
  reason inline; the one-line helper is gone. The distinct quantiles, hash serializations and cohort
  APIs are retained.
- **EVAL-TYPES — implemented:** `tsconfig.eval.json` includes `eval/retention`.
  It type-checked without changes, so no source or frozen result changed for
  it. `test/retention.test.ts`, like the other plugin suites, stays outside
  both typecheck passes.
- **EVAL-BODY — implemented, with evidence:** the canonical request hash
  (`canonicalRequestSha256`) covers only `state` and `questions`, and a body
  limit never changes a singleton request's bytes; it only decides whether
  `buildRetentionRequests` throws. `buildCaseRequest` passed no limit, so the
  builder's default Jev limit applied to every provider. On the baseline an
  OpenAI case whose canonical body was one byte over 32 MiB was refused as
  `budget`, and a Jev body was measured with the production model ID instead
  of the selected one, so a wire body pushed over the limit by a longer ID was
  planned. `buildCaseRequest`
  now passes `null`, and `planCase` (`runner.ts`), which builds the exact body
  that planning sends and loading verifies, refuses a body above the adapter's
  `bodyByteLimit` (TypeSafe's 32 MiB for Jev, none for OpenAI Decisions).
  Error recognition and the recorded failure categories are unchanged: the
  evaluation records an HTTP failure by its status, as documented, and has no
  `token-limit` category.
- **Evidence:** `test/decision-eval.test.ts` adds a Jev body exactly at the
  limit (planned), the same case with `jev-1.130.0` and a body one byte over
  (both refused as `budget`), and the over-limit case planned for OpenAI;
  iterates `FATAL_CATEGORIES` with a status per
  category that the type checker requires (401, 403 and 429), each leaving no
  unknown request; accepts an explicit held-out 0.8 through `renderReport` and
  the command line while refusing 0.5 and 0.6; and added eight numeric values
  it refused as not plain decimals, which the review remediation replaced
  with the same values planning or reporting exactly as their plain forms,
  plus empty, non-finite and out-of-range values refused with no request.
  `test/retention.test.ts` adds paid scoring against a fake
  endpoint that fails the second snapshot (the first snapshot's checkpoint
  stays on disk with the output's fields, no output is written, and a retry
  sends nothing and changes no byte), a complete run that writes the output
  once and removes its checkpoints, and the historical argument cases. The
  body regression failed on the baseline source at the `jev-1.130.0`
  assertion, and the baseline refused the OpenAI over-limit case as `budget`.
  Each other regression failed with its behavior removed: not-attempted
  records only after `auth` (none recorded after 429), every held-out override
  refused, `Number` parsing (a check the remediation removed), no checkpoint
  write, and an unknown order
  accepted. The frozen artifacts and the `eval/retention` `metrics` output are
  byte-identical, as are the `eval:decisions check` output and three dry-run
  plans (OpenAI, Jev and Jev 1.14.0) with their canonical and provider
  request-set hashes. Format check, two-pass typecheck, lint, the full plugin
  suite (1,230 tests, 19,281 expectations) and both offline eval checks (3
  compaction and 17 decision cases) passed under scratch XDG. A controlled
  failure through real files kept `snapshot-1-of-2.json` and wrote no output;
  its script, result and retained bytes are in
  `/tmp/opencode/architecture-refactor/decision-contracts/controlled-failure/`.
- **Acceptance / verification:** `test/decision-eval.test.ts`, retention offline
  checks and the extended eval typecheck own valid/invalid CLI cases, cohort
  metrics, explicit 0.8, request parity and frozen hashes.

### Tooling and release

**Decision — implemented:** `bun run release-opencode -- X.Y.Z` keeps its bump
command and refuses wrong local state before its first write. The pre-commit
hook checks staged content without changing files and runs the root gates. The
two workflow check lists stay explicitly synchronized.

- **RELEASE-IDENTITY — implemented:** on the baseline the helper read the
  working-tree manifest, so an uncommitted bump left by an interrupted run made
  it tag and push a HEAD whose manifest still had the old version. It also ran
  on any branch (tagging a commit outside `main`), committed unrelated staged
  files into the release commit, created the release commit before an existing
  tag stopped it, and pushed `main` and the tag non-atomically, so a rejected
  `main` still published the tag. `scripts/release-opencode.sh` now refuses,
  before any write, unless HEAD is the branch `main`, the index and working tree
  are clean with no untracked files, the tag exists neither locally nor on
  `origin`, and local `main` contains `origin`'s `main`. It decides the bump
  from the manifest committed at HEAD, commits only the manifest and `bun.lock`,
  requires version `X.Y.Z` at HEAD and a working tree matching HEAD before
  `git tag`, and pushes both refs with `git push --atomic`. It never stashes,
  resets or moves a tag. `docs/Development.md` now orders the release steps as
  they happen (checks on current `main`, then the helper's release commit, then
  `Check` and the publish workflow on that commit) and documents recovery for
  each interruption point and for an npm version published without a GitHub
  Release (verify the immutable tag, version and publication, then only
  `gh release create --verify-tag`; no rerun, dispatch or second publish).
- **TOOLING-STAGED — implemented:** the baseline hook split staged names on
  whitespace, ignored renames, checked working-tree bytes, ran
  `prettier --write` on the working tree, hid diagnostics, and ran only the
  first typecheck pass (`npx tsc --noEmit`). `.githooks/pre-commit` now
  runs `scripts/check-staged-format.ts`, which checks the regular files that
  `git diff --cached --raw -z --no-renames` reports as added, copied, modified
  or type-changed with Prettier's API, taking content, configuration and
  ignore files from one temporary snapshot of the index (see TOOLING-REVIEW)
  and writing nothing in the repository; then `bun run typecheck` and
  `bun run lint`, with `set -euo pipefail` and visible output. The plugin
  package's narrower `typecheck`, `lint`, `format` and `test` scripts, which no
  document or workflow used, are removed, so the root commands are the one
  owner. No workflow changed, and no dispatch or publication ran.
- **TOOLING-REVIEW — implemented:** the final review found six defects in the
  helper and checker above, each reproduced with Git 2.43, Bun 1.4.2 and
  Prettier 3.9.6. (1) A failed `git status` printed nothing and passed the
  clean-tree check, and a failed `git show` of the manifest read as another
  version and started a bump. With a submodule whose repository was missing,
  or an injected `status` failure beside a staged unrelated file, the helper
  committed, tagged and pushed, the latter with the unrelated file in the
  release; with a corrupt index or a failed manifest read it ran `npm version`
  and `bun install` before stopping. Every Git read now has its exit status
  checked before its output is used, the manifest is parsed with a fixed,
  content-free message on failure, and each failure refuses before the first
  write. (2)
  `git add` then `git commit` committed the whole index, so a file staged during
  the bump was published. The helper now refuses with no commit when anything
  other than the manifest and `bun.lock` changed or was staged, commits only
  those two paths (`git commit -- <paths>`), leaving other staging in place, and
  still checks the index and working tree before tagging. (3)
  `git push origin refs/heads/main refs/tags/vX.Y.Z` followed
  `remote.origin.push` mappings and `push.followTags`, which sent `main` to
  another branch or published an unrelated annotated tag. The push now names
  `src:dst` for both refs and passes `--no-follow-tags`. (4) The preflight read
  the fetch URL while the push wrote to every push URL, so a push repository
  holding the tag was never checked and two push URLs published to one
  repository and not the other despite `--atomic`. The helper resolves
  `git remote get-url --push --all origin` once, requires exactly one URL that
  Git does not rewrite again (checked with Git's own resolution through a
  command-line remote), and uses that URL for `ls-remote` and `push`; a push URL
  that differs from the fetch URL stays valid. (5) The checker read staged
  content but the working tree's ignore files and configuration, so unstaged
  edits to them hid or caused failures. It now writes the index, including the
  temporary index `git commit` passes in `GIT_INDEX_FILE`, to a temporary
  directory, adds an empty `.git` marker so Prettier's `.editorconfig` search
  stops at its root as it does at the repository root, takes content,
  configuration, ignore files and shebang-based parser inference from that
  snapshot, and removes only that directory. Plugins named by package in a
  configuration still load from the working directory, as the CLI loads them. (6) Type
  changes were skipped, so a symlink replaced by an unformatted file passed;
  the checker now includes type changes whose new mode is a regular file.
  `docs/Development.md` scopes the local-tag refusal to the interrupted-release
  recovery, which now pushes explicit refs to the one resolved push URL with
  `--no-follow-tags`, and merges a moved `origin` `main` so that the tag stays
  in its history. Bun 1.4.2's `bun install --frozen-lockfile` accepted a
  lockfile whose workspace version was stale when the dependencies were
  installed, so the docs ask for a completed `bun install` before a manual bump
  commit without claiming that CI rejects a stale lockfile.
- **RELEASE-BINDING — implemented:** probes with Git 2.43 against local bare
  repositories found that the destination was not bound as documented. (1) A
  relative push URL such as `upstream` reached `ls-remote` and `push` as a bare
  word, which Git reads as a remote of that name when one exists, including a
  legacy `.git/remotes` or `.git/branches` file, so the release went to that
  remote's push URLs. The helper now writes a relative local path as
  `./path`, leaving URLs, `host:path` and absolute paths unchanged, and refuses
  when `git remote get-url -- <destination>` still finds a remote (exit 0) or
  fails other than with exit 2. A push URL that Git rewrites once to that
  relative path, which the old check refused, is now released to the path where
  `git push origin` writes. (2) An `insteadOf` or `pushInsteadOf` rule added by
  a `bun install` lifecycle script or the `post-commit` hook retargeted the
  push; (3) a branch switch during the bump put the release commit and tag on
  another branch, one with an unreviewed commit, and published that tag. The
  helper records the commit `main` names at the start and, after the bump,
  after the commit and after the tag, requires HEAD to be `main` at the
  expected commit, the release commit to be the one commit `main` gained, the
  tag to name it, and Git to read the destination unchanged. It tags and
  pushes by ID (`<release-commit>:refs/heads/main` and
  `<tag-object>:refs/tags/vX.Y.Z`).
  Each recheck is fresh, not a lock: a change made by another process after
  the last one still applies, and `docs/Development.md` no longer claims that a
  configuration change cannot retarget the push. (4)
  `git status` refreshed cached file times and rewrote the index on a refusal;
  it now runs with `--no-optional-locks`. (5) The normal and recovery steps
  fetched `origin/main` from the fetch repository; they now fetch the push
  repository's `main` by URL and merge and push the IDs they checked.
  The staged-format inventory in `docs/Development.md` now lists the
  `.editorconfig` cases it actually has. Against the previous helper 33 of the
  58 release cases fail; disabling the path rewrite, the remote-name refusal,
  the rechecks of the destination, of HEAD and `main`, of the commit's parent or
  of the tag, or `--no-optional-locks` each fails its cases. Pushing
  `refs/heads/main` instead of the release commit's ID fails the one case in
  which a `git` wrapper commits on local `main` just before the push; origin
  then receives that late commit (`/tmp/opencode/architecture-refactor/release-pins/`).
  Shell syntax, format check, two-pass typecheck, lint and the full
  plugin suite (1,277 tests, 19,904 expectations) passed under scratch XDG.
  The probes are in `/tmp/opencode/architecture-refactor/release-binding/`,
  and the controls, a run of the documented recovery and the closure logs in
  `/tmp/opencode/architecture-refactor/release-context/`.
- **RELEASE-TREE — implemented:** the helper read the release commit after
  `git commit` and its hooks had run, then checked only its parent, its
  manifest version and a clean working tree. A one-shot `post-commit` hook
  that amends the commit in place keeps all three. The repository's
  `.githooks` has no such hook, and the root `prepare` script that
  `bun install` runs points `core.hooksPath` there, but a hooks path given in
  the environment (`GIT_CONFIG_COUNT`, `GIT_CONFIG_KEY_0=core.hooksPath`)
  overrides that local setting. With Git 2.43, a local bare origin and a
  stand-in `bun` that runs the real `prepare` script, such a hook made the
  previous helper tag and push an amended `README.md` and, in a second run, a
  manifest with an extra field at the same version. The helper now stages the
  manifest and `bun.lock`, refuses with no commit unless the index and working
  tree then differ from HEAD only by those staged files, records the index's
  tree with `git write-tree`, and after the path-only commit refuses before
  the tag unless the release commit has exactly that tree. The parent, ref,
  destination and version checks and the push by ID are unchanged, and the
  tree check is fresh, not a lock. `test/release.test.ts` adds both
  amendments, each refused with the amended commit left on local `main`,
  no tag and origin unchanged, and a failed status read after staging (61
  cases). Against the previous helper both amendments are published; with
  only the tree comparison removed, both fail. The reproduction and controls
  are in `/tmp/opencode/architecture-refactor/release-tree/`.
- **TOOLING-REJECTED — rejected:** dedup-only changes to the identical
  `.github/workflows/check.yml` and `publish-opencode.yml` lists, speculative job
  permissions, root metadata or ignore-file style cleanup lack established
  impact. Keep the lists synchronized instead.
- **COMMENTS — implemented:** every comment in tracked first-party code was
  read: 3,391 TypeScript comment lines (`src/` 1,819, `eval/` 633, `test/` 930,
  `scripts/` 5, package entries 4) and the 5 shell comment lines in
  `scripts/release-opencode.sh` and `.githooks/pre-commit`. 21 lines are
  removed: 20 comments that only restated the identifier or the code beneath
  them, and one orphan in `test/trim.test.ts` left by a deleted helper.
  Retained comments state host and provider version facts, protocol and
  storage formats, ownership and lifecycle invariants, content-free error
  rules, evaluation-protocol constants, test scenario intent and the one lint
  directive. The removal leaves comment-free TypeScript printer output and
  emitted JavaScript identical for every changed file.
- **Evidence:** `test/release.test.ts` runs the helper in disposable
  repositories with a local bare origin and stand-in `npm` and `bun`. Every
  refusal (another branch, detached HEAD, staged, untracked or interrupted-bump
  changes, an existing local or origin tag, origin `main` ahead) leaves refs,
  index, files and origin unchanged and calls neither stand-in; the bump and the
  already-committed paths tag the HEAD whose committed manifest has the version;
  a different committed version or extra changed files create no tag. Against
  the baseline script 10 of its 12 cases fail. Hook scenarios in disposable
  clones (spaced, newline and renamed names, both partial-staging directions, a
  staged file deleted from the working tree, ignored and unknown types, a syntax
  error, a failing typecheck, and a real `git commit`) left index, working tree
  and HEAD byte-identical; the baseline hook skipped a spaced name and rewrote
  an unstaged file. Scripts and output are in
  `/tmp/opencode/architecture-refactor/tooling/`. For TOOLING-REVIEW,
  `test/release.test.ts` adds the failed-read, push-configuration, push-URL and
  staging cases, and `test/staged-format.test.ts` owns the checker, comparing
  each verdict with the Prettier CLI on a checkout of the index. Against the
  previous helper 22 of the 40 release cases fail, and against the previous
  checker 8 of the 21 format cases fail, exactly the configuration, ignore,
  `.editorconfig`, shebang, type-change and `git commit --only` cases;
  removing the `.git` marker or the type-change filter each fails its one
  case. Shell syntax,
  format check, two-pass typecheck, lint and the full plugin suite (1,256
  tests, 19,773 expectations) passed under scratch XDG. The claim probes
  are in `/tmp/opencode/architecture-refactor/tooling-claims/`, and the closure
  logs in `/tmp/opencode/architecture-refactor/tooling-closure/`. GitHub and
  npm steps are not exercised.

### Test quality — last

**Decision — implemented:** After behavioral work, review every suite and support
file. Remove tests that cannot catch real bugs, implementation-copy and
duplicate assertions; repair tests that fail without a behavior defect. Keep one
owning layer for each invariant and retain legitimate schema and protocol
tripwires. Earlier changes may make necessary import adjustments and add their
own invariant regressions; broad deletion is last.

- **TEST-QUALITY — implemented:** five audits of `test/` and the eval support
  files drove these changes. Folded or removed, each with its remaining owner:
  the compacted-attachment plan-path test (the vision test for a compacted
  attachment and the compacted-cost test), the second count-mismatch test (the
  nine-target test, now asserting the error name and exact text), the nested
  overflow test (a row of the sanitizing overflow table), the loose Jev
  concurrency test (the exact OpenAI fatal-failure and failed-header admission
  tests), the rejected-`retryWait` test (no default timer can reject; the real
  timer and abort cases remain), the tokenizer spies on functions the module
  never imports and the bounded-check duplicate (the byte precheck tests), the
  earlier-notice token comparison (exact reference-format pins), the
  decision-eval default-inequality and runtime-default pins (the frozen 0.80
  pin), the runtime-default development run (the 0.5 threshold report test),
  the duplicate license substring, `backup` storage checks where nothing writes
  storage, self-derived schema and created-state lines (the judge prompt and
  removal checks), and `/magic-compact`'s copy of the parser table (one table on
  `/magic-trim`'s bindings plus each command's exact handler errors).
  Strengthened: checkpoint, cohort, tamper-stage, timeout-versus-HTTP and parser
  rejections assert their exact reason; the handoff drift tests run the
  command's own recheck; the 4,096-record counter test uses two messages of
  2,050 parts (focused runtime 2.8-3.0 s before, 0.09-0.14 s after; no suite
  speed claim); the lifecycle order test pins relative order and counts; V1
  source writes include host replay and recovered users compare every native
  field; tokenizer extraction uses composed `countGptTokens` oracles; the V2
  explicit-request estimate is checked against the prompt, marker and tool
  schema a real request carries; real generation registration, the context
  hook's one-request stop, the owned title and `read_omitted_content` run
  through the real setup hooks; the eval output-ancestor guard and the frozen
  corpus hash have literal oracles; release refuses invalid versions, logs exact
  `npm`/`bun` arguments and keeps origin's main when origin refuses the tag.
  Bun 1.4.2's `toMatchObject` replaces matched properties of the received object
  with the asymmetric matcher, which had made one tamper-loop message check
  vacuous; that loop now reads the error before matching. Negative controls
  restored byte for byte after each run are in
  `/tmp/opencode/architecture-refactor/test-quality/mutations/`. Format check,
  two-pass typecheck, lint, the full plugin suite (1,193 tests, 19,424
  expectations), both offline eval checks and a retention `metrics` output
  identical to the baseline's passed under scratch XDG. At a load average near
  60, the compaction-eval cleanup test that ran the cleanup CLI six times (three
  ownership failures, each run twice) exceeded Bun's default 5 s per-test
  timeout on this candidate and on the unchanged baseline. Its recorded
  assertions passed until the timeout; the exit 143 reported afterwards was the
  child stopped at teardown. Each ownership failure is now its own `test.each`
  case with the same two runs and assertions (35 expectations across the five
  CLI cases, as before) and a 20 s budget for two child starts, not a latency
  bound. With a scaled 2 s budget the old test timed out and every new case
  passed. Forcing exit 0 or skipping the cleanup root-ownership check failed
  all three cases. The full suite then passed with 1,195 tests and 19,424
  expectations at a load average near 4; evidence is in
  `/tmp/opencode/architecture-refactor/cli-test-isolation/`.
- **Acceptance / verification:** each owning suite demonstrates its public
  behavior and error cases; the full suite runs with isolated XDG after cleanup.

## Verification boundary

Baseline evidence is `/tmp/opencode/architecture-refactor/baseline/summary.tsv`:
frozen install, format check, two-pass typecheck, lint, plugin tests and both
offline eval checks exited 0 under scratch XDG. The recorded test run reported
1,180 tests, 0 failures and 18,920 expectations; compaction and decision checks
covered 3 and 17 cases respectively. This baseline does not prove any later
candidate. Frozen evaluation data must remain unchanged.

For the integrated refactor, run `bun run format:check`, `bun run typecheck`,
`bun run lint`, `bun test packages/opencode-plugin/test`,
`bun run eval:compact check`, and `bun run eval:decisions check`, with scratch
`XDG_DATA_HOME` for tests and evals. Run relevant owning checks as changes land;
the four full gates are not required before every small commit. For adapter or
package-entry changes, qualify the packed artifact in containers on both
supported hosts using synthetic sessions and a local scripted provider; no paid
calls. `/tmp/opencode/inline-file-reporting/review-fixes/containers` is a seed,
not proof of coverage: final qualification must exercise the affected lifecycle
paths. Compare frozen data and report any unverified requirement explicitly.

# Development

This page covers the public setup, repository layout, and common maintenance commands for contributors.

The distribution package is `arkive-compact`, the repository is `https://github.com/imarshallwidjaja/magic-compact`, and the runtime command remains `/magic-compact`.

## Requirements

- Bun

## Setup

Install dependencies from the repository root:

```bash
bun install
```

## Repository Layout

### `packages/`

- `packages/opencode-plugin` - OpenCode plugin implementation.
- `src/api.ts` - OpenCode API access helpers.
- `src/index.ts` - package export surface and slash/tool registration.
- `src/magic-compact.ts` - `/magic-compact` command entrypoint, projection, profitability, and recovery orchestration.
- `src/magic-trim.ts` - `/magic-trim` command entrypoint.
- `src/magic-stats.ts` - `/magic-stats` command entrypoint.
- `src/tui.ts` - TUI plugin wiring.
- `src/util.ts` - shared local utilities.
- `src/compact/batch.ts` - model-derived generation budgets, media-aware native input estimates, and one-target batch planning.
- `src/compact/compact.ts` - command-owned generation-fork lifecycle, message isolation, structured-output request and ordered local response validation, and v3 summary-part insertion.
- `src/compact/constants.ts` - compaction notices, boundary metadata, summary metadata, and omission notice formatting.
- `src/compact/media.ts` - authoritative v2 user-image and v3 user/assistant-image manifest discovery plus pure model-request projection.
- `src/compact/plan.ts` - turn planning, native checkpoint validation, source snapshot capture, post-insertion validation, and boundary selection.
- `src/compact/prune.ts` - exact prune mutation planning/execution and tool input/output pruning.
- `src/compact/session.ts` - session backup, owned insertion, revalidation, and recovery helpers.
- `src/storage/omission.ts` - omitted content cache persistence and preplanned omission IDs for exact projection.
- `src/storage/stats.ts` - persisted stats storage.
- `src/storage/store.ts` - shared storage access helpers.
- `src/stats/constants.ts` - stats constants.
- `src/stats/events.ts` - stats event accounting.
- `src/stats/pricing.ts` - pricing calculations.
- `src/stats/tokenize.ts` - token estimation helpers.
- `test/compact.test.ts` - native batch budgets/concurrency/isolation, encoded tool payload and fixed media canaries, exact retained fork identity, active-failure aggregation, one-target requests and their fixed response schema, response count and identity, media manifests/projection, opaque summary-bearing turns, useful-savings projection, exact pruning/omission, source-drift preservation, checkpoint, stats, and recovery tests.
- `test/omission.test.ts` - omission cache migration, integrity, legacy quarantine, backup, and concurrency tests.
- `test/trim.test.ts` - trim planning, metadata idempotency, v0 summary preservation, and omission-format integration tests.
- `test/tokenize.test.ts` - canonical local token-accounting tests.
- `test/license.test.ts` - packaged no-grant notice plus verbatim upstream Magic Compact BSD-3.

### License

- `LICENSE.md` - no license grant for this distribution's modifications and packaging.
- `NOTICE.md` - retained upstream Magic Compact BSD 3-Clause notice.
- `packages/opencode-plugin/LICENSE.md` and `packages/opencode-plugin/NOTICE.md` - the same files, shipped with the published `arkive-compact` package. Keep each byte-identical to the matching root file.

### `docs/`

- `docs/Development.md` - contributor setup and repository map.
- `docs/Core.md` - core behavior and safety guarantees.
- `docs/OpenCode.md` - OpenCode runtime behavior specification.

## Common Commands

- `bun run typecheck` - TypeScript type checking.
- `bun run lint` - ESLint checks.
- `bun run format` - Prettier formatting.
- `bun test packages/opencode-plugin/test/compact.test.ts` - focused compaction suite.
- `bun test packages/opencode-plugin/test/trim.test.ts` - focused trim suite.

## Notes

- The project uses Bun workspaces.
- For the user-facing plugin install flow, see the main `README.md`.
- The native-checkpoint cases in `packages/opencode-plugin/test/compact.test.ts` are maintenance canaries for user `compaction` parts with optional `tail_start_id`, and assistant `parentID`, `summary`, `finish`, and `error` fields. Check those fields against OpenCode's compaction implementation whenever OpenCode or `@opencode-ai/sdk` is upgraded.
- Source race detection structurally compares every behavior-affecting message/part field and ordering while ignoring volatile timing fields and token/cost accounting churn. The semantic tool `state.time.compacted` bit survives snapshot normalization, so a concurrent change to it counts as drift while `start`/`end` timing churn does not. It runs before writes and again after owned summary/boundary insertion immediately before exact planned pruning. Pre-write drift and every failure from the first source-write attempt onward preserve both the live source and backup without selecting or deleting either, retain concurrent messages, record no successful stats increment, and report both session IDs for manual recovery. Recheck message info and part state variants on OpenCode or SDK upgrades.
- OpenCode's `session.prompt` applies pending-revert cleanup before appending the message, silently deleting the reverted suffix. Magic Compact therefore fails closed with a pending-revert error before generation and rechecks the session immediately before the first source write. Recheck this prompt-time cleanup on OpenCode or SDK upgrades.
- Resolve the native hidden compaction agent's explicit model/variant and matching provider descriptor once. Derive the 32K maximum half-output target, usable input, 400K/80% normal target, 90% hard single-turn limit, 500K/90% halo limit, and 8K fixed reserve in `compact/batch.ts`, which plans exactly one target turn per batch. The 8K reserve is only for non-explicit host, system, and provider overhead. Separately tokenize each complete rendered explicit request exactly once for every normal, usable, hard-turn, and halo decision: static instructions, byte-identical global index, local target JSON and punctuation, fixed-shape conservative markers plus injected marker text, exact ordinal digits, media counts, and an optional following-halo stop record/marker. Use the 32K fallback output limit only when the descriptor lacks a safe positive output value. Estimate ordinary visible strings through `countGptTokens`. Charge only image/PDF-typed file parts and completed non-compacted tool attachments 8K; ignore `text/plain` and directory file parts as non-model-visible, and estimate other model-visible file parts and attachments by their URL (encoded length for `data:` URLs, tokenized text otherwise). Estimate arbitrary or nested `data:` tool strings by their encoded length without invoking the tokenizer on their payloads. Match the pinned `message-v2.ts` compacted, error, interrupted-output, pending, and running state branches exactly.
- Build one immutable global index before batching from all complete turns in the planner's post-native-checkpoint effective source range. Include target candidates, opaque summary-bearing turns, and retained-tail complete turns; exclude the trailing active incomplete turn. Entries contain exactly `sourceOrdinal` and the first ordinary non-synthetic/non-ignored user text line capped at 160 characters. `sourceOrdinal` is the one-based complete-turn position in full source order and is distinct from target candidate ordinals.
- Plan one batch per candidate in chronological order and never split a turn. Every generation request owns exactly one target turn: multi-target requests produced wrong result counts live because OpenCode does not enforce the response schema, so do not reintroduce packing. Render the complete request and response protocol for that single target; a target above the normal target keeps the hard single-turn limit. Run at most five independent batches concurrently. Each batch gets a fresh complete source fork with an ordered session rule that denies operational tools and allows only the injected `StructuredOutput` response tool, full rewritten-shape verification, sequential bounded `session.deleteMessage` calls for every unrelated message, and an exact retained-history reload before markers. Before deletion, capture the retained rewritten message/part IDs, order, roles, parent links, part types, and fork-scoped references; reject any substitution, reorder, or link change on reload. Retain complete target turns plus at most one complete neighboring non-checkpoint turn on each side only when native history plus the complete stop-bearing request stays within the halo limit. Never use part deletion for generation isolation or issue more than one prompt on a fork.
- Plan with `NATIVE_BATCH_PLANNING_NONCE`, whose 22 ASCII bytes each tokenize separately under the pinned tokenizer and therefore conservatively represent a valid 22-byte nonce body. Before any fork creation, allocate every unique crypto-quality 128-bit nonce absent from source, render the immutable real prompt, and require its exact explicit-request and response-protocol counts not to exceed the conservative records or the batch's normal, hard, halo, usable-input, and output limits. Recheck the nonce against rewritten retained fork strings before marker insertion. Keep target candidate ordinals in every marker, exact target prompt index, and response. Batch-array order is the final chronological mapping. Validate every response and exact target coverage. Every batch derives its byte-identical global index, target index, fork plan, and estimate from the same pre-generation source; never feed generated summaries into another request. The first failure stops new admission; drain every already-active batch, collect all sanitized failures in deterministic batch order, clean every fork, and accept no generated result until all batches pass. Generation remains before backup and source mutation. Magic Compact makes one `session.prompt` call per fork and adds no plugin-level retry, adaptive splitting, or model fallback; OpenCode may apply its runtime provider/transport retry policy or additional model steps required by structured-output processing. Partial writes remain unsupported.
- Keep each exact target index proportional to its target count. Each target entry contains only the exact fork marker, target ordinal, and expected user-and-assistant inline-image count. Precede it with the byte-identical `{globalTurnIndex: [{sourceOrdinal, anchor}]}` orientation map. A retained following halo adds a marker-only stop record for its nonce-bound stop marker. Require every response item to have exactly `ordinal`, `marker`, `summary`, and `visualDescriptions` and to echo its exact marker and target ordinal in order. Global anchors are orientation-only; never make the model echo user text as response identity. Reject only the active exact generation nonce or marker token inside summaries and visual descriptions, then discard marker identity before persistence. Never serialize SDK records, message/part refs, assistant traces, tool state, provider metadata, encrypted reasoning, data URLs, manual file parts, or generated summaries into either index.
- Await generation-fork creation and each model request without a local wall-clock deadline so the command never loses ownership of a server-side fork. Keep permission updates, message isolation/reloads, marker insertion, and every toast under the shared control-operation deadline. Once a fork ID is known, directly await provider or validation failure abort and every fork deletion; cleanup rejection is surfaced and every started batch drains before command settlement. `/magic-compact` and `/magic-trim` show a long-lived view-only progress toast before slow work, then send terminal no-op, success, or failure feedback. If a timed-out progress delivery later settles, reassert the terminal toast without awaiting that reassertion in the command. Toast rejection or timeout never changes operation or recovery outcomes and never creates conversation, model-input, stats, or omission-cache data. OpenCode exposes no later delivery signal after the client promise rejects, so a progress toast applied remotely after that rejection cannot be detected or corrected. Native overflow reports trim-first guidance. A configured model input budget too small to fit the fixed 8K reserve or the mandatory repeated global index fails before any fork with trim-first/model-capacity guidance. An output limit requires a model with more output capacity because every request covers a single turn. A throttling error returned by OpenCode is a normal plugin failure, not a Magic Compact retry signal; OpenCode may apply its runtime provider/transport retry policy before the prompt call returns.
- `/magic-trim` removes a fully initialized backup when trimming changes no tool state. Once trim mutation begins, the backup remains the recovery artifact; recovery failure preserves the root failure, reports both session IDs for manual recovery, and still attempts terminal feedback.
- Each batch calls `session.prompt` once with the fixed JSON-schema `format`, whose `turns` array sets `minItems` and `maxItems` to 1. The prompt text states the same exact count. OpenCode 1.18.30 injects its side-effect-free `StructuredOutput` response tool for the JSON-schema request and returns its captured object on assistant `info.structured`; the injected tool still passes through the host's final merged-permission tool filtering, where the last matching rule wins. The fork's ordered session rule therefore ends with an exact `StructuredOutput` allow after the wildcard deny, which keeps the host response tool while every operational tool stays denied. Text parts are never parsed as a fallback. The installed 1.18.30 prompt path does not use `retryCount`, so the request omits it. Magic Compact adds no plugin-level retry or fallback; OpenCode may apply its runtime provider/transport retry policy or additional model steps required by structured-output processing. Charge the fixed schema JSON that every request transmits against explicit input, and keep the host-injected system instructions in the 8K non-explicit reserve. Budget estimated summary/description content plus the mandatory envelope and array/object punctuation, echoed marker, ordinal, `summary` property, and every `{ordinal, description}` visual entry before packing. The schema states the single-result count, stable keys, types, and required fields to the model, but it is guidance rather than enforcement: OpenCode 1.18.30 wraps it with the AI SDK `jsonSchema()` helper without a validator, so tool arguments are not checked against it locally, and it sends `strict: false` on every function tool for `@ai-sdk/openai`, `@ai-sdk/azure`, and `@ai-sdk/amazon-bedrock/mantle` providers. Local validation is authoritative and still requires exactly one result, its target ordinal, exact marker echo, nonempty summary, and exact visual coverage. Reject active nonce leakage before framing, then require complete target-ordinal coverage across all batches. Validation failures expose only fixed categories (missing structured output, invalid envelope, turn count mismatch, turn shape/identity, summary, visual coverage, nonce leakage) through lifecycle and batch errors; turn count mismatches carry only the expected and received integers, and turn shape/identity failures carry only the first failed check (`shape`, `ordinal`, or `marker`) and its target ordinal. Prompt for readable continuation summaries; do not add semantic reducers or status/evidence/scaffolding protocols.
- Any turn containing `metadata.magicCompact.summary === true` is persisted user data for the compaction writer. Skip the entire turn byte-identically regardless of version, including persisted v0/v1/v2 and current v3 summaries; do not parse, migrate, rewrite, or upsert it. Trim may process other tool parts but must preserve the summary part itself.
- New summary parts use `prt_-magic_capsule_v3_${messageID}` plus `metadata.magicCompact.summary === true`, `capsule === true`, `agentic === true`, and `version === 3`. Historical v0/v1/v2 summaries remain persisted fixtures only and must never be regenerated or rewritten.
- Prune projection is the only mutation producer. Plan explicit part/message deletes and tool updates by their IDs, preserve exact assistant inline image files covered by the accepted v3 manifest, carry each planned replacement and its ordered omission IDs on the update, validate the full source plus owned insertions immediately before execution, and never derive deletes or updates by sweeping a live reload.
- Bounded token checks fast-reject spans that cannot fit under the pinned `gpt-tokenizer@3.4.0` o200k maximum token width of 128 bytes, then perform at most one memoized exact `countTokens` call for survivors. Keep the exact version guard and adjacent accepted/rejected boundary tests; do not use the 128-byte bound unless the installed package version is exactly `3.4.0`.
- Compact and trim accounting use canonical active content: include model-facing synthetic parts, exclude ignored user reporting notices, include ignored assistant text, compute the exact pre-notice result once, persist savings on that basis, then require the actual active count to equal the projection after notice injection. Do not reintroduce a self-referential notice fixed point.
- OpenCode's native history conversion is the only generation media and reasoning path. Completed tool output/attachments, user and assistant files, and provider reasoning remain native. Gate the configured model's image capability over the union of retained batch histories. The cross-provider reasoning test remains a maintainer canary. Magic Compact must not manually serialize, redact, split, convert, or reattach those values.
- New v3 summaries persist `magicCompact.mediaManifest` version `1`, built locally from authoritative source user then assistant inline images and binding encounter ordinal, exact MIME values, and SHA-256 hashes of exact original URLs. Tool `state.attachments` remain native tool media and omission behavior and never enter the persisted manifest. Persisted v2 manifests remain user-only; their `partId` values are parsed only for compatibility and never override canonical ordinal/MIME/hash identity after OpenCode rewrites IDs. Never trust model-supplied IDs or hashes, and never update, delete, omit-cache, or migrate user file parts or accepted v3-covered assistant image files. `projectManifestCoveredMedia` accepts persisted v2 and current v3 envelopes and remains the one pure projection used by the OpenCode message-transform hook, profitability, final token verification, and model-facing accounting. It validates every manifest-covered image by canonical ordinal/MIME/hash and replaces it with a placeholder only when the placeholder's canonical token count is strictly smaller than the original file URL token contribution; otherwise the exact file part stays inline and model-facing, so projected media token counts are non-negative by construction. Missing manifests are inert; malformed, incomplete, added, removed, reordered, MIME-changed, or hash-changed cohorts throw rather than hiding or resending media.
- Keep `totalTokensPruned`/`cachedTokensSaved` as physical canonical-content accounting. Optional v1 stats fields `mediaTokensProjected` and `cachedMediaTokensSaved` preserve compatibility with existing files while reporting request-only media projection separately. Recompaction treats all summary-bearing turns as opaque; a valid existing media manifest still projects independently.
- OpenCode imports this plugin once per server process. After changing plugin source, restart the OpenCode server process itself; restarting an attached TUI does not reload it.

## Release

Release versions are exact numeric triplets in `X.Y.Z` form. The release helper bumps the `packages/opencode-plugin/package.json` manifest when needed and creates a `vX.Y.Z` tag. The release workflow publishes the public `arkive-compact` package with provenance and explicit public access. Publishing uses npm Trusted Publishing: GitHub OIDC via the workflow's `id-token: write` permission authenticates to the npm trusted publisher configured for GitHub user/repo `imarshallwidjaja/magic-compact`, workflow filename `publish-opencode.yml`, with npm publish permission and no environment. The workflow requires no long-lived npm token. The existing `NPM_TOKEN` GitHub Actions secret is now unused; delete and revoke it only after a future OIDC-backed publish succeeds, keeping rollback available until live verification. There is no build step because the package publishes its TypeScript source.

### Normal release

The normal release path is the tag-triggered GitHub Actions workflow:

1. Verify that `packages/opencode-plugin/package.json` has the intended `arkive-compact@X.Y.Z` version.
2. Run the checks documented under Common Commands, including the focused test suites.
3. Confirm npm Trusted Publishing is configured for `imarshallwidjaja/magic-compact` with workflow `publish-opencode.yml`, npm publish permission, and no environment; the workflow needs no npm token.
4. Invoke `bun run release-opencode -- X.Y.Z`.
5. Monitor the tag-triggered publish and GitHub Release workflow.

### Recovery-only manual dispatch

Use manual dispatch only when an existing immutable tag was already pushed and either GitHub created no workflow run or the prior run reached a terminal failure that published nothing, the corresponding npm version is still unpublished, and no GitHub Release exists. Verify all three absences before one dispatch, then use the tag's exact 40-character commit SHA:

```bash
gh workflow run publish-opencode.yml --ref main -f release_tag=vX.Y.Z -f release_sha=<40-char-tag-commit>
```

Never move, delete, or recreate the tag. Never dispatch while a prior run is active, and never dispatch after the npm version or GitHub Release exists. After a terminal failure that published nothing, a new dispatch is allowed only after the workflow fix is reviewed and pushed, and only after the operator freshly reconfirms the unchanged tag/SHA, that the npm version is absent, and that no GitHub Release exists.

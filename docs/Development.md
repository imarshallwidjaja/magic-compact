# Development

This page covers the public setup, repository layout, and common maintenance commands for contributors.

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
- `src/compact/compact.ts` - single-fork summary generation, ordered response validation, and v3 summary-part insertion.
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
- `test/compact.test.ts` - one-fork generation contract, incident-sized prompt canary, response ordering, media manifests/projection, opaque summary-bearing turns, useful-savings projection, exact pruning/omission, source-drift preservation, checkpoint, stats, and recovery tests.
- `test/omission.test.ts` - omission cache migration, integrity, legacy quarantine, backup, and concurrency tests.
- `test/trim.test.ts` - trim planning, metadata idempotency, v0 summary preservation, and omission-format integration tests.
- `test/tokenize.test.ts` - canonical local token-accounting tests.

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
- Source race detection structurally compares every behavior-affecting message/part field and ordering while ignoring timestamp and token/cost accounting churn. It runs before writes and again after owned summary/boundary insertion immediately before exact planned pruning. Pre-write drift and every failure from the first source-write attempt onward preserve both the live source and backup without selecting or deleting either, retain concurrent messages, record no successful stats increment, and report both session IDs for manual recovery. Recheck message info and part state variants on OpenCode or SDK upgrades.
- Summary generation uses exactly one ephemeral fork of the complete source session and exactly one model request for all summary-free selected turns. Immediately update the fork to a deny-all permission ruleset, verify its rewritten-ID chronological message/part-type and turn shape against the source plan, choose a crypto-quality 128-bit nonce absent from all ordinary strings in both histories, and inject nonce-and-ordinal synthetic markers only into the ephemeral fork before prompting. Bound nonce collision retries and fail closed. Always use the hidden Magic summarizer agent and the explicit model and variant configured on OpenCode's native hidden compaction agent. Never fall back to the source model.
- Keep the explicit prompt proportional to turn count. Each index entry contains only the exact fork marker, monotonic ordinal, exact escaped first ordinary non-synthetic/non-ignored user text line capped at 300 characters, and expected user-and-assistant inline-image count. `nextTurn` uses a nonce-bound stop marker and bounded anchor. Require every response item to echo its exact marker, ordinal, and anchor; reject only the active exact generation nonce or marker token inside summaries and visual descriptions, then discard marker identity before persisting visible summary text. Useful source identifiers may remain when the model includes them. Never serialize SDK records, message/part refs, assistant traces, tool state, provider metadata, encrypted reasoning, data URLs, or manual file parts into this prompt.
- Await the model request without a local wall-clock deadline. Bound fork creation, permission update, marker insertion, `session.abort`, fork deletion, and every Magic Compact toast with the shared control-operation deadline helper. Toast timeout or failure never replaces the primary command outcome. Provider or validation failure uses bounded abort and one deletion attempt. A late fork creation gets detached bounded abort/delete cleanup. If the native request overflows, report trim-first guidance; if it reaches an output limit, advise a larger `N` so fewer turns are selected. Do not add retries, per-turn fallback, chunks, model fallback, or partial mutation.
- The response is one JSON object with one `turns` array. Require exact turn count, marker echo, ordinal sequence, anchor echo, nonempty summary, and exactly one nonempty visual description per expected user-or-assistant image ordinal. Reject the response before framing when any summary or visual description contains the active exact generation nonce; this also rejects its containing exact marker token, while unrelated nonce-shaped text remains valid. Prompt for readable continuation summaries that preserve outcome, process/decisions, verification, and unresolved/next work; do not add semantic reducers or status/evidence/scaffolding protocols.
- Any turn containing `metadata.magicCompact.summary === true` is persisted user data for the compaction writer. Skip the entire turn byte-identically regardless of version, including persisted v0/v1/v2 and current v3 summaries; do not parse, migrate, rewrite, or upsert it. Trim may process other tool parts but must preserve the summary part itself.
- New summary parts use `prt_-magic_capsule_v3_${messageID}` plus `metadata.magicCompact.summary === true`, `capsule === true`, `agentic === true`, and `version === 3`. Historical v0/v1/v2 summaries remain persisted fixtures only and must never be regenerated or rewritten.
- Prune projection is the only mutation producer. Plan explicit part/message deletes and tool updates by their IDs, preserve exact assistant inline image files covered by the accepted v3 manifest, carry each planned replacement and its ordered omission IDs on the update, validate the full source plus owned insertions immediately before execution, and never derive deletes or updates by sweeping a live reload.
- Bounded token checks fast-reject spans that cannot fit under the pinned `gpt-tokenizer@3.4.0` o200k maximum token width of 128 bytes, then perform at most one memoized exact `countTokens` call for survivors. Keep the exact version guard and adjacent accepted/rejected boundary tests; do not use the 128-byte bound unless the installed package version is exactly `3.4.0`.
- Compact and trim accounting use canonical active content: include model-facing synthetic parts, exclude ignored user reporting notices, include ignored assistant text, compute the exact pre-notice result once, persist savings on that basis, then require the actual active count to equal the projection after notice injection. Do not reintroduce a self-referential notice fixed point.
- OpenCode's native history conversion is the only generation media and reasoning path. In `packages/opencode/src/session/message-v2.ts`, completed tool `state.output` becomes typed tool-result text/content and `state.attachments` become typed media/file content; user and assistant file parts and provider reasoning are likewise converted natively. Before fork creation, scan the complete source snapshot, including retained and opaque summary turns, for image-bearing user or assistant files and tool attachments and require the configured model's image capability if any may be exposed. The cross-provider reasoning test in `compact.test.ts` is a maintainer canary: recheck native conversion whenever OpenCode, the SDK, or provider adapters change. Magic Compact must not manually serialize, redact, split, convert, or reattach any of those values.
- New v3 summaries persist `magicCompact.mediaManifest` version `1`, built locally from authoritative source user then assistant inline images and binding encounter ordinal, exact MIME values, and SHA-256 hashes of exact original URLs. Tool `state.attachments` remain native tool media and omission behavior and never enter the persisted manifest. Persisted v2 manifests remain user-only; their `partId` values are parsed only for compatibility and never override canonical ordinal/MIME/hash identity after OpenCode rewrites IDs. Never trust model-supplied IDs or hashes, and never update, delete, omit-cache, or migrate user file parts or accepted v3-covered assistant image files. `projectManifestCoveredMedia` accepts persisted v2 and current v3 envelopes and remains the one pure projection used by the OpenCode message-transform hook, profitability, final token verification, and model-facing accounting. It validates every manifest-covered image by canonical ordinal/MIME/hash and replaces it with a placeholder only when the placeholder's canonical token count is strictly smaller than the original file URL token contribution; otherwise the exact file part stays inline and model-facing, so projected media token counts are non-negative by construction. Missing manifests are inert; malformed, incomplete, added, removed, reordered, MIME-changed, or hash-changed cohorts throw rather than hiding or resending media.
- Keep `totalTokensPruned`/`cachedTokensSaved` as physical canonical-content accounting. Optional v1 stats fields `mediaTokensProjected` and `cachedMediaTokensSaved` preserve compatibility with existing files while reporting request-only media projection separately. Recompaction treats all summary-bearing turns as opaque; a valid existing media manifest still projects independently.

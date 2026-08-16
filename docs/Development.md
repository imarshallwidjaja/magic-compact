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
- `src/compact/capsule.ts` - canonical deterministic extractive capsule writer and structured current-state reducer.
- `src/compact/compact.ts` - canonical local capsule generation and capsule-part insertion.
- `src/compact/constants.ts` - compaction notices, boundary metadata, summary metadata, and omission notice formatting.
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
- `test/compact.test.ts` - deterministic capsule contract, structured current-state reduction, opaque summary-bearing turns, useful-savings projection, exact prune planning, source-drift rollback, checkpoint, omission, stats, and recovery tests.
- `test/fixtures/incident-session.ts` - bounded source-backed semantic excerpts with message/part provenance.
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
- Source race detection structurally compares every behavior-affecting message/part field and ordering while ignoring timestamp and token/cost accounting churn. It runs before writes and again after owned capsule/boundary insertion immediately before exact planned pruning. Recheck message info and part state variants on OpenCode or SDK upgrades.
- Capsule generation is deterministic and local. Do not add hidden summarizer agents, scratch sessions, model discovery, prompt templates, JSON model transport, retry loops, or provider output clamps back into the compaction path.
- Continuation semantics use one linear current-state reducer in `src/compact/capsule.ts`. Keep structured lifecycle, allowlisted Hive/task result schemas, and canonical numeric `state.metadata.exit` ahead of opaque output. Invalid values and malformed recognized schemas fail closed, while exit zero cannot suppress recognized domain failure counts or zero-run results.
- Scope command resolution by the exact raw trimmed command plus lexically normalized explicit absolute `cwd`/`workdir`. Relative paths, missing context, NUL, mixed-style paths, and conflicting contexts remain unscoped/fail-closed. Do not normalize Markdown, shell syntax, real paths, or the process cwd.
- Assistant prose is source text only. Do not reintroduce decision, mutation, placeholder, blocker, verification, or VCS regexes as correctness gates over assistant text. Ordinary prose, technical fences, placeholders, and JSON examples must remain exact `Work:` content.
- `Work:` must include every substantive non-synthetic non-Magic-summary assistant text part across every assistant message in the selected turn. The visible representation may use deterministic provider-safe framing, but the exact source JavaScript strings must be mechanically recoverable from the capsule.
- Any turn containing `metadata.magicCompact.summary === true` is persisted user data for the compaction writer. Skip the entire turn byte-identically for both v0 and v1; do not parse, migrate, rewrite, or upsert it. Trim may process other tool parts but must preserve the summary part itself.
- New deterministic capsule parts use `prt_-magic_capsule_v1_${messageID}` plus `metadata.magicCompact.summary === true`, `capsule === true`, and `version === 1`. Historical `prt_-magic_summary_${messageID}` remains a test-local persisted fixture only; do not restore a runtime v0 producer.
- Prune projection is the only mutation producer. Plan explicit part/message deletes and tool updates by their IDs, carry each planned replacement and its ordered omission IDs on the update, validate the full source plus owned insertions immediately before execution, and never derive deletes or updates by sweeping a live reload.
- Bounded token checks fast-reject spans that cannot fit under the pinned `gpt-tokenizer@3.4.0` o200k maximum token width of 128 bytes, then perform at most one memoized exact `countTokens` call for survivors. Keep the exact version guard and adjacent accepted/rejected boundary tests; do not use the 128-byte bound unless the installed package version is exactly `3.4.0`.
- Compact and trim accounting use canonical active content: include model-facing synthetic parts, exclude ignored user reporting notices, include ignored assistant text, compute the exact pre-notice result once, persist savings on that basis, then require the actual active count to equal the projection after notice injection. Do not reintroduce a self-referential notice fixed point.

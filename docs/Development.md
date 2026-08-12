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
  - `src/index.ts` - package export surface.
  - `src/magic-compact.ts` - `/magic-compact` command entrypoint.
  - `src/magic-trim.ts` - `/magic-trim` command entrypoint.
  - `src/magic-stats.ts` - `/magic-stats` command entrypoint.
  - `src/tui.ts` - TUI plugin wiring.
  - `src/util.ts` - shared local utilities.
  - `src/compact/compact.ts` - main compaction flow.
  - `src/compact/constants.ts` - compaction constants and custom summarizer-agent definition.
  - `src/compact/high-risk.ts` - semantic deterministic-only classification, record budgeting, and summary rendering.
  - `src/compact/plan.ts` - turn planning and boundary selection.
  - `src/compact/prune.ts` - tool input/output pruning.
  - `src/compact/session.ts` - session backup and mutation helpers.
  - `src/compact/template.ts` - summarization prompt template helpers.
  - `src/storage/omission.ts` - omitted content cache persistence and preplanned omission IDs for exact projection.
  - `src/storage/stats.ts` - persisted stats storage.
  - `src/storage/store.ts` - shared storage access helpers.
  - `src/stats/constants.ts` - stats constants.
  - `src/stats/events.ts` - stats event accounting.
  - `src/stats/pricing.ts` - pricing calculations.
  - `src/stats/tokenize.ts` - token estimation helpers.
  - `test/omission.test.ts` - omission cache migration, integrity, legacy quarantine, backup, and concurrency tests.
  - `test/compact.test.ts` - compaction planning, strict plain-JSON scratch lifecycle, semantic deterministic rendering, token packing, useful-savings projection, incident acceptance, checkpoint, and recovery tests.
  - `test/fixtures/incident-session.ts` - synthetic incident-shape generator with no copied transcript content.
  - `test/trim.test.ts` - trim planning, metadata idempotency, and omission-format integration tests.
  - `test/tokenize.test.ts` - provider and local token-accounting tests.

### `docs/`

- `docs/Development.md` - contributor setup and repository map.
- `docs/OpenCode.md` - OpenCode runtime behavior specification.

## Common Commands

- `bun run typecheck` - TypeScript type checking
- `bun run lint` - ESLint checks
- `bun run format` - Prettier formatting

## Notes

- The project uses Bun workspaces.
- For the user-facing plugin install flow, see the main `README.md`.
- The native-checkpoint cases in `packages/opencode-plugin/test/compact.test.ts` are maintenance canaries for user `compaction` parts with optional `tail_start_id`, and assistant `parentID`, `summary`, `finish`, and `error` fields. Check those fields against OpenCode's compaction implementation whenever OpenCode or `@opencode-ai/sdk` is upgraded.
- Semantic evidence classification consumes the exact message and part shapes exposed by the pinned `@opencode-ai/sdk`; source race detection structurally compares every behavior-affecting message/part field and ordering while ignoring timestamp and token/cost accounting churn. Recheck tool-state variants, scratch create/prompt/delete APIs, and relevant SDK fields on upgrades.
- Standard batches intentionally omit `session.prompt.format` and request one plain-text JSON object keyed by the exact selected assistant turn IDs. Keep strict `JSON.parse`, independent exact-key/content/output-budget validation, one cleanup-gated fresh-scratch retry, typed content-free diagnostics, and the all-results barrier covered together in `test/compact.test.ts`. Summary generation and projection create no source message; all source writes wait for validation, the useful-savings gate, backup creation, and source revalidation.
- This omission is a pinned-runtime constraint: under OpenCode `1.18.15`, an explicit `session.prompt.format` can persist an `OutputFormat*` value that later SDK reads reject. Keep the strict plain-JSON text protocol until this path is revalidated on an OpenCode or SDK upgrade.
- Deterministic-only turns never call the summarizer model. Keep semantic complete-value classification, assistant-message error/recovery status records, resolved-negation normalization, the 2,048-token per-turn cap, the 512-token optional allowance, UTF-8-safe source windows, compact required-first reservation, and omission counts in `high-risk.ts`. Required overflow must use the bounded aggregate capsule, retain partial/error outcome and counts/types, and fit as many stable IDs/statuses as possible. Error tool parts stay active. Size/message/tool counts are not safety signals.
- Standard requests use `session.create` with no `parentID`, source `directory`/workspace routing, persisted `workspaceID` binding, deny-all session and agent permissions, at most 24K local input tokens, a summarizer-scoped 4K provider output cap plus local 4K validation, and at most two concurrent requests. The pinned agent config type omits the runtime wildcard permission key, so the deny-all agent permission requires a narrow cast. Recheck empty scratch inheritance, session permission precedence, create/prompt/abort/delete APIs, routing fields, `chat.params`, and prompt cancellation behavior on OpenCode or SDK upgrades.
- Full source race detection compares the live source with its canonical planning snapshot during preflight and again after backup creation immediately before the first source write. Source races or unknown source revalidation use `NativeCheckpointChangedError` preserve-both recovery. Final actual-versus-projected mismatch after writes uses separate preserve-both late-drift recovery, defers successful stats persistence, and names the backup. There is no journal or lock, so same-token concurrent mutation after final revalidation is not detected by the token projection check.

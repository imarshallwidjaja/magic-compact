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
  - `src/compact/high-risk.ts` - semantic deterministic-only classification, readable continuation-capsule rendering, and machine evidence-map rendering.
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
  - `test/compact.test.ts` - compaction planning, strict plain-JSON scratch lifecycle, semantic deterministic capsule rendering, token packing, useful-savings projection, per-turn profitability selection, incident acceptance, checkpoint, and recovery tests.
  - `test/fixtures/incident-session.ts` - synthetic incident-shape generator with no copied transcript content.
  - `test/trim.test.ts` - trim planning, metadata idempotency, and omission-format integration tests.
  - `test/tokenize.test.ts` - canonical local token-accounting tests.

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
- Bounded token checks fast-reject spans that cannot fit under the pinned `gpt-tokenizer@3.4.0` o200k maximum token width of 128 bytes, then perform at most one memoized exact `countTokens` call for survivors. The exact-count memo keeps one small entry and skips oversized keys so arbitrary conversation strings are not retained process-global. Keep the exact version guard and adjacent accepted/rejected boundary tests; do not use the 128-byte bound unless the installed package version is exactly `3.4.0`.
- Standard batches intentionally omit `session.prompt.format` and request one plain-text JSON object keyed by the exact selected assistant turn IDs, with narrative bodies only. Keep strict `JSON.parse`, independent exact-key/content/output-budget validation, placeholder rejection, raw-first finite producer-schema detection, unsafe-control and unpaired-surrogate rejection, approved-separator folding, raw-`&` neutralization before raw-`<` neutralization, the exact host-owned `Outcome: ...\nWork: ...` two-line assertion, one cleanup-gated fresh-scratch retry, typed content-free diagnostics, and the all-results barrier covered together in `test/compact.test.ts`. The canonical schema canary derives representative assistant-text, tool-status, required-record, and accounting shapes and labels from `renderStandardTurnEvidence`; whenever producer fields, accounting labels, or record kinds change, update the detector's explicit finite vocabulary and that canary together. Do not replace this with Markdown-prefix normalization or treat ordinary prose/unrelated JSON as machine records. Summary generation and projection create no source message; all source writes wait for validation, final rendered-capsule profitability selection, the useful-savings gate, backup creation, and source revalidation.
- This omission is a pinned-runtime constraint: under OpenCode `1.18.15`, an explicit `session.prompt.format` can persist an `OutputFormat*` value that later SDK reads reject. Keep the strict plain-JSON text protocol until this path is revalidated on an OpenCode or SDK upgrade.
- Deterministic-only turns never call the summarizer model. Keep semantic complete-value classification, assistant-message error/recovery status records, resolved-negation normalization, explicit blocked/could-not-finish/credentials-missing routing, the 2,048-token per-turn safety ceiling, the 512-token optional allowance, and readable continuation-capsule rendering in `high-risk.ts`. Treat every substantive text part in the final assistant message as one ordered final response. Preserve it in full whenever it fits the remaining capsule budget; otherwise preserve every classified decision, verification, VCS, unresolved, and next-action cue from all of those parts through source-anchored excerpts. Visible deterministic capsules must stay human-readable: Outcome/Work/Verification/VCS/Unresolved/Next sections, no standalone raw evidence records, source-window arrays, or record accounting. Structural validation must still allow legitimate technical prose that mentions machine field syntax. Aggregate only normalized-equivalent repeated facts with truthful counts. Never sample away a distinct command, path, detail, decision, or repair action; carry normalized raw source identity separately from rendered excerpts and fail closed when distinct required actions cannot be represented. Standard-turn evidence maps stay machine-structured and exact: every substantive final-message text part, including ignored model-visible assistant text, is required exact evidence in source order; `ordered-literal-runs-v1` segments are emitted only when a whole source part is provably periodic (a linear prefix-function minimal-period check over exact JavaScript code units, with an optional unique leading segment), unique text stays literal, and if that required evidence cannot fit the 24K input bound the operation fails closed before scratch, backup, or mutation. Error tool parts stay active. Size/message/tool counts are not safety signals.
- Standard requests use `session.create` with no `parentID`, source `directory`/workspace routing, persisted `workspaceID` binding, deny-all session and agent permissions, at most 24K local input tokens, a best-effort summarizer-scoped 4K provider output clamp that only caps an existing positive finite active limit (local 4K acceptance is unconditional), and at most two concurrent requests. The pinned agent config type omits the runtime wildcard permission key, so the deny-all agent permission requires a narrow cast. Recheck empty scratch inheritance, session permission precedence, create/prompt/abort/delete APIs, routing fields, `chat.params`, and prompt cancellation behavior on OpenCode or SDK upgrades.
- Full source race detection compares the live source with its canonical planning snapshot during preflight and again after backup creation immediately before the first source write. Source races or unknown source revalidation use `NativeCheckpointChangedError` preserve-both recovery. Final actual-versus-projected mismatch after writes uses separate preserve-both late-drift recovery, defers successful stats persistence, and names the backup. There is no journal or lock, so same-token concurrent mutation after final revalidation is not detected by the token projection check.
- Compact and trim accounting use canonical active content: include model-facing synthetic parts, exclude ignored user reporting notices, include ignored assistant text, compute the exact pre-notice result once, persist savings on that basis, then require the actual active count to equal the projection after notice injection. Do not reintroduce a self-referential notice fixed point. Keep no-op detection and backup behavior unchanged. Existing unversioned v0 summary parts remain valid through recompaction planning and trim.

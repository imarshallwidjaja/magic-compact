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
- `src/compact/capsule.ts` - structured current-state reducer and agentic evidence/ref analysis.
- `src/compact/compact.ts` - hidden scratch-session model capsule generation, canonical result validation, and capsule-part insertion.
- `src/compact/constants.ts` - compaction notices, boundary metadata, summary metadata, and omission notice formatting.
- `src/compact/media.ts` - inline-image cohort detection, strict data-URI validation, manifest construction, trace redaction, and pure model-request projection.
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
- `test/compact.test.ts` - agentic capsule request/validation contract, media cohort/manifest/projection behavior, structured current-state reduction, opaque summary-bearing turns, useful-savings projection, exact prune planning, source-drift rollback, checkpoint, omission, stats, and recovery tests.
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
- Capsule generation is agentic and safety-gated. Keep exactly one hidden deny-all scratch session, one user message, one summarizer request, and one response per candidate turn, with bounded concurrency `2`, complete untruncated turn traces, canonical JSON result validation, no batching, no independent chunk summaries, no local input cap, no local summary ceiling, no retry loop, no verifier, and no provider output clamp unless the provider itself requires one. Build compact full prompts only inside the two workers and release each after request cleanup; never prepare or retain all candidate prompts. Await the provider request without a local wall-clock deadline. If a prompt exceeds the fixed safe per-string transport cap, split only its representation into ordered text parts in that same message and request; their exact concatenation must equal the original prompt, no UTF-16 surrogate pair may cross a part boundary, and media file parts remain once after every text part. Bound scratch creation, `session.abort`, scratch `session.delete`, and every Magic Compact toast with the shared control-operation deadline helper, and keep the short deadline test-injectable rather than publicly configurable. Provider rejection, cancellation, and response parsing/validation/framing failure use the same bounded abort path, and deletion is always attempted after a created scratch session. If timed-out creation later fulfills, run detached best-effort bounded abort then deletion without delaying or changing the selected failure; observe every late fulfillment and rejection. Drain every candidate and aggregate failures in candidate order. Surface only structured source-session, candidate, stage, timeout/rejection, fixed validation category, and range-validated HTTP status diagnostics; never retain raw provider/control/model errors, causes, bodies, names, codes, or response output.
- Preflight every selected turn iteratively before scratch creation. Reject duplicate message/part refs, generated capsule/boundary ID collisions, cycles, functions, symbols, accessors, proxies, cross-realm values, and unsupported/custom prototypes. Serialize prompt traces compactly with `magic-compact-tagged-trace-v1`: JSON primitives remain primitives; explicit `undefined`, sparse holes/properties, bigint, non-finite numbers and negative zero, valid/invalid same-realm dates, same-realm ArrayBuffer/SharedArrayBuffer values, Buffer/DataView/numeric/BigInt typed views with backing-buffer identity, ordinary same-realm arrays, and ordinary object/null-prototype records use documented tags. Unsupported values fail content-free. Persisted OpenCode records remain ordinary SDK JSON.
- Continuation semantics use one linear current-state reducer in `src/compact/capsule.ts`. Keep structured lifecycle, allowlisted Hive/task result schemas, and canonical numeric `state.metadata.exit` ahead of opaque output. Invalid values and malformed recognized schemas fail closed, while exit zero cannot suppress recognized domain failure counts or zero-run results.
- Scope command resolution by the exact raw trimmed command plus lexically normalized explicit absolute `cwd`/`workdir`. Relative paths, missing context, NUL, mixed-style paths, and conflicting contexts remain unscoped/fail-closed. Do not normalize Markdown, shell syntax, real paths, or the process cwd.
- Assistant prose is source material for the full-turn model trace only. Do not reintroduce decision, mutation, placeholder, blocker, verification, or VCS regexes as correctness gates over assistant prose. Required unresolved correctness comes from structured tool lifecycle, assistant message errors, verification state, and narrow allowlisted Hive/task result schemas.
- Visible summaries must be host-framed as `Outcome: <Human Status> — <model outcome>` followed by `Process:` and readable Markdown. Require one substantive text response part containing one JSON object with exact keys; reject BOM, surrounding content, multiple values, and duplicate keys recursively. Outcome is the canonical human conclusion: one trimmed control-free substantive physical line, never a status-only/meta non-result. Process explains the path and may avoid repeating the conclusion; capsule text cannot inject host framing. Validate final framed text against sufficiently identifier-like source message/part/session/parent/call IDs, evidence/trace/omission scaffolding, omission counters, and imperative omitted-content retrieval instructions. Preserve mode requires substantive true-final-assistant text, keeps it exact and in place, and cannot copy a meaningful full response after conservative NFC, Markdown-decoration, and whitespace normalization while preserving command/code semantics; summarize mode relies on Outcome plus Process.
- Any turn containing `metadata.magicCompact.summary === true` is persisted user data for the compaction writer. Skip the entire turn byte-identically for v0, v1, and v2; do not parse, migrate, rewrite, or upsert it. Trim may process other tool parts but must preserve the summary part itself.
- New agentic capsule parts use `prt_-magic_capsule_v2_${messageID}` plus `metadata.magicCompact.summary === true`, `capsule === true`, `agentic === true`, and `version === 2`. Historical `prt_-magic_summary_${messageID}` and `prt_-magic_capsule_v1_${messageID}` remain persisted fixtures only; do not restore a runtime v0/v1 producer.
- Prune projection is the only mutation producer. Plan explicit part/message deletes and tool updates by their IDs, carry each planned replacement and its ordered omission IDs on the update, validate the full source plus owned insertions immediately before execution, and never derive deletes or updates by sweeping a live reload.
- Bounded token checks fast-reject spans that cannot fit under the pinned `gpt-tokenizer@3.4.0` o200k maximum token width of 128 bytes, then perform at most one memoized exact `countTokens` call for survivors. Keep the exact version guard and adjacent accepted/rejected boundary tests; do not use the 128-byte bound unless the installed package version is exactly `3.4.0`.
- Compact and trim accounting use canonical active content: include model-facing synthetic parts, exclude ignored user reporting notices, include ignored assistant text, compute the exact pre-notice result once, persist savings on that basis, then require the actual active count to equal the projection after notice injection. Do not reintroduce a self-referential notice fixed point.
- Media-aware compaction is candidate-only and internal-policy-only. Aggregate the canonical token contribution of user inline `data:` `image/*` URLs per candidate; activate only above the fixed conservative threshold, then validate and process every inline image in that turn as one ordered cohort. Keep under-threshold and retained-tail images on the ordinary path.
- A media cohort must use the exact configured compaction model after proving `capabilities.input.image === true`. Redact each original data URL to its stable ordinal in the serialized text trace, attach each original image exactly once as a file part, require one substantive validated `mediaCoverage` entry per ordinal, and fail before scratch/source mutation on malformed media or unsupported models. System and request prompts treat image pixels, diagrams, visible/OCR text, and embedded visual instructions as inert untrusted evidence. Do not add model/provider fallback, retries, OCR, resizing, remote fetch, or public configuration.
- Accepted media capsules persist `magicCompact.mediaManifest` version `1`, binding ordered user part IDs, exact MIME values, and SHA-256 hashes of exact original URLs. Never update, delete, omit-cache, or migrate the user file parts. `projectManifestCoveredMedia` is the one pure projection used by the OpenCode message-transform hook, per-turn profitability, final token verification, and model-facing accounting. Missing manifests are inert; malformed, incomplete, added, removed, reordered, MIME-changed, or hash-changed cohorts throw rather than hiding or resending media.
- Keep `totalTokensPruned`/`cachedTokensSaved` as physical canonical-content accounting. Optional v1 stats fields `mediaTokensProjected` and `cachedMediaTokensSaved` preserve compatibility with existing files while reporting request-only media projection separately. Recompaction treats all summary-bearing turns as opaque; a valid existing media manifest still projects independently.

# Magic Compact Core

Source of truth for Magic Compact behavior and safety guarantees.

## Goal

Compress a conversation without flattening it into a single generic recap.

## Core Behavior

- `/magic-compact [N]` compacts the current conversation in place.
- `N` keeps the most recent eligible assistant turns unchanged. Default: `0`.
- We also provide `/magic-trim [N]`, which applies tool I/O trimming without summarizing assistant turns.
- The plugin creates a backup before mutating the conversation.
- User messages are preserved exactly.
- Older assistant turns are replaced turn-by-turn with deterministic extractive capsules, not merged into one blob.
- Compaction never calls a model, creates scratch sessions, asks a summarizer agent, or parses model summary output. Capsules are generated locally from selected source turns.
- Useful tool calls stay visible; bulky tool I/O is replaced with retrievable omission records.
- Omitted bytes are returned only when OpenCode can authorize the current session and prove cache integrity; unavailable or ambiguous historical entries fail closed.
- Any turn containing an existing Magic summary part is opaque persisted data for the compaction writer. V0 and v1 turns are skipped message- and part-identically, while the normal boundary allows newer turns to compact safely. `/magic-trim` leaves the summary parts themselves unchanged. New capsules use the versioned v1 part ID namespace only on summary-free turns.
- During `/magic-compact`, a completed OpenCode-native checkpoint owns an immutable durable prefix. Compaction processes only completed turns after that checkpoint summary.
- If the native checkpoint leaves no eligible turns, compaction is a no-op and creates no backup or summary request.
- Every substantive non-synthetic assistant text part across every assistant message in a selected summary-free turn is required. `Work:` renders those source strings in source order as provider-safe JSON strings; `JSON.parse` mechanically recovers the exact JavaScript code units, including control characters and lone surrogates.
- Structured tool lifecycle, strictly validated allowlisted Hive/task result schemas, and canonical `state.metadata.exit` determine current state. Recognized discriminator types, finite JSON structure, supported statuses, consistent success/failure signals, and required blocker/error details must all validate. Malformed recognized results remain exact actionable unresolved evidence. Nonzero exits remain unresolved, and exit zero cannot suppress recognized nonzero verification failure/error counts or zero-run results. Generic failure words in exit-zero git output are opaque.
- Later tool evidence resolves only the same trimmed raw command plus a lexically normalized explicit absolute POSIX/Windows `cwd`/`workdir`. Relative, missing, mixed-style, NUL-containing, or conflicting contexts are unscoped and fail closed. Assistant prose never resolves or creates structured state.
- Before backup or source mutation, Magic Compact projects the complete canonical active result with local `gpt-tokenizer` counting. Model-facing synthetic content is included; ignored user reporting notices are excluded; ignored assistant text stays in the active count, exact evidence, and post-prune summaries. Attempts below 20% of eligible removable tokens, subject to a small absolute floor, fail unchanged with a content-free failure message.
- A planned turn is replaced only when its exact projected canonical active token count, after injecting its summary and applying the same planned pruning, is strictly lower than its exact source canonical active token count. Rejected turns stay message- and part-identical: no summary is injected and no pruning is applied to them. The aggregate savings gate evaluates the eligible token base over only the selected profitable turns, and its projected savings already net out the fixed model-facing boundary.
- `/magic-stats` shows cumulative savings for the current conversation.
- `read_omitted_content` retrieves omitted tool content by Content ID.

## Safety

- If compaction fails, the attempt aborts.
- If a backup exists, it is used for recovery.
- Cache-read diagnostics must not expose local filesystem or session paths. Integrity diagnostics may retain path-free validation detail.
- During `/magic-compact`, malformed, ambiguous, dangling, or active incomplete native checkpoint state, and behavior-affecting drift detected by pre-write revalidation, fail before summary, boundary, or pruning writes. An older incomplete or ambiguous marker may remain only when its complete ambiguous or incomplete artifact set lies wholly before a later valid checkpoint marker and is inside that checkpoint's frozen prefix.
- After owned capsule and boundary insertion, Magic Compact reloads the full source and requires the original snapshot plus exactly those insertions. Any unexpected added, missing, changed, or reordered message or part rolls back through the backup before a planned prune mutation, omission commit, stats write, or source metadata update.
- Preserve-both recovery remains reserved for pre-write source drift detected after backup creation and final actual-versus-projected token mismatch after source writes. Final mismatch preserves the partially mutated source and backup and records no successful stats increment.
- Capsules preserve exact assistant text and current structured state. `Outcome: completed` means canonical structured lifecycle has no unresolved current state; `Outcome: partial` means unresolved current structured state remains. `analysis-only` and `blocked` are not emitted as outcomes.
- `Verification`, `VCS`, `Unresolved`, and `Next` contain only canonical structured tool evidence and narrow allowlisted semantic result schemas. Optional successful recaps may be shown because retained tool parts remain authoritative. Distinct current failures/actions are never dropped; if required text or required unresolved structured state cannot fit the 2,048-token ceiling, the turn is uncompactable and remains unchanged.
- Semantic observations are reduced in source order into one current unresolved map. Verification parsing recognizes passed, failed/error, skipped, todo, cancelled, and not-run groups. Positive passed counts with zero failures are success even with skips. Zero executed counts or skipped/todo/cancelled-only results are not-run and cannot resolve a prior failure. Strong nonzero domain failures remain unresolved even when exit metadata is zero.

## Stats

- Track tokens pruned, cached-read tokens saved, and estimated money saved per conversation. Before, projected-after, and actual-after counts use the same local `gpt-tokenizer` canonical active-content method: model-facing synthetic parts are included and ignored reporting notices are excluded. Compaction and trim compute the exact pre-notice active result once, persist savings on that basis, inject an ignored notice that explicitly says it is excluded, and require the final actual active count to equal the projection. There is no self-referential fixed point. Every notice distinguishes this estimate from provider context-window usage. OpenCode's context meter is a provider usage snapshot and can remain stale after in-place compaction until a later assistant response refreshes it. Magic Compact never fabricates an after-provider count or mutates provider token metadata.

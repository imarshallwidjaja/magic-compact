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
- Older assistant turns are summarized turn-by-turn, not merged into one blob.
- The bounded historical evidence supplied to each scratch session is authoritative for summaries; summaries never inspect external state or continue unfinished work.
- Useful tool calls stay visible; bulky tool I/O is replaced with retrievable omission records.
- Omitted bytes are returned only when OpenCode can authorize the current session and prove cache integrity; unavailable or ambiguous historical entries fail closed.
- Re-running compaction later preserves earlier summaries and compacts newer turns.
- During `/magic-compact`, a completed OpenCode-native checkpoint owns an immutable durable prefix. Compaction processes only completed turns after that checkpoint summary.
- If the native checkpoint leaves no eligible turns, compaction is a no-op and creates no backup or summary request.
- Large completed turns remain model-eligible regardless of serialized size, message count, or tool count. A turn is deterministic-only when tool state, assistant-message metadata, or historical evidence is failed, not-run, running, error, unknown, incomplete, or otherwise unresolved. A later successful assistant message resolves an earlier assistant-message error for routing while retaining bounded recovered-error evidence.
- Model summaries run in transcript-free scratch sessions from bounded exact per-turn evidence maps. Scratch sessions have no parent, deny all tools and side effects, and run at most two requests concurrently.
- Before backup or source mutation, Magic Compact projects the complete active result with local `gpt-tokenizer` counting. Attempts below 20% of eligible removable tokens, subject to a small absolute floor, fail unchanged with a content-free failure message.
- `/magic-stats` shows cumulative savings for the current conversation.
- `read_omitted_content` retrieves omitted tool content by Content ID.

## Safety

- If compaction fails, the attempt aborts.
- If a backup exists, it is used for recovery.
- Cache-read diagnostics must not expose local filesystem or session paths. Integrity diagnostics may retain path-free validation detail.
- During `/magic-compact`, malformed, ambiguous, dangling, or active incomplete native checkpoint state, and any concurrent behavior-affecting source message or part drift, fails before summary, boundary, or pruning writes. An older incomplete or ambiguous marker may remain only when its complete artifact set is wholly before a later valid checkpoint marker and therefore inside that checkpoint's frozen prefix.
- Scratch sessions have no parent transcript or inherited checkpoint artifact. Preserve-both recovery remains reserved for durable source drift, including a final actual-versus-projected token mismatch after source writes, or source state that cannot be confirmed unchanged. Detected late drift preserves the partially mutated source and backup and records no successful stats increment.
- Standard summary batches require exactly one plain-text JSON object keyed by the exact assistant turn IDs. The summarizer-scoped `chat.params` hook limits provider generation to at most 4,000 output tokens without changing other agents, and Magic Compact independently parses collected text parts with `JSON.parse` and validates shape, exact keys, string values, substantive content, and the same 4K local-token response cap. One failed batch may be retried once in a fresh scratch session only for a typed summary-contract failure and only after failed-session cleanup succeeds; provider, authentication, context, body-info, SDK, cancellation, timeout, and cleanup failures are not retried. Timeout or client cancellation explicitly awaits server-side `session.abort` before scratch deletion, preserving prompt, abort, and deletion failures together.
- A generated summary must preserve evidenced continuation state. Deterministic-only summaries first reserve every required failed, not-run, running, error, unknown, incomplete, or unresolved record using the compact individual schema. If those minima exceed the 2,048-token cap, a model-free overflow capsule remains partial and preserves aggregate required counts/types, as many stable tool/message identifiers and minimal statuses as fit, and an explicit required-record overflow disclosure. Underlying error tool parts remain active, so the capsule does not duplicate their full I/O. Optional drops are disclosed by record counts. All summaries and projected pruning remain in memory until full source revalidation and the useful-savings gate pass.

## Stats

- Track tokens pruned, cached-read tokens saved, and estimated money saved per conversation. Before, projected-after, and actual-after active counts use the same local `gpt-tokenizer` method; provider usage is telemetry only.

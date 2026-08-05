# Magic Compact Core

Source of truth for behavior shared by every Magic Compact implementation.

## Goal

Compress a conversation without flattening it into a single generic recap.

## Core Behavior

- `/magic-compact [N]` compacts the current conversation in place.
- `N` keeps the most recent eligible assistant turns unchanged. Default: `0`.
- We also provide `/magic-trim [N]`, which applies tool I/O trimming without summarizing assistant turns.
- The plugin creates a backup before mutating the conversation.
- User messages are preserved exactly.
- Older assistant turns are summarized turn-by-turn, not merged into one blob.
- Historical transcript evidence is authoritative for summaries. Current inspection may clarify that evidence, but summaries distinguish current observations from historical claims and never continue unfinished work.
- Useful tool calls stay visible; bulky tool I/O is replaced with retrievable omission records.
- Omitted bytes are returned only when the platform implementation can authorize the current session and prove cache integrity; unavailable or ambiguous historical entries fail closed.
- Re-running compaction later preserves earlier summaries and compacts newer turns.
- During `/magic-compact`, a completed platform-native checkpoint owns an immutable durable prefix. Compaction processes only completed turns after that checkpoint summary.
- If the native checkpoint leaves no eligible turns, compaction is a no-op and creates no backup or summary request.
- `/magic-stats` shows cumulative savings for the current conversation.
- `read_omitted_content` retrieves omitted tool content by Content ID.

## Safety

- If compaction fails, the attempt aborts.
- If a backup exists, it is used for recovery.
- Cache-read diagnostics must not expose local filesystem or session paths. Integrity diagnostics may retain path-free validation detail.
- During `/magic-compact`, malformed, ambiguous, dangling, active incomplete, or concurrently changed native checkpoint state fails before summary, boundary, or pruning writes. An older incomplete or ambiguous marker may remain only when its complete artifact set is wholly before a later valid checkpoint marker and therefore inside that checkpoint's frozen prefix.
- A platform-local temporary checkpoint artifact aborts generation, but preserve-both recovery is reserved for a durable source checkpoint race or source state that cannot be confirmed unchanged.
- A generated summary must preserve evidenced continuation state. A platform may classify risky turns more strictly; when it renders a high-risk summary deterministically, no model request or model prose may contribute to that summary, and required negative or not-run evidence must not be dropped silently.

## Stats (Only where possible)

- Track tokens pruned, cached-read tokens saved, and estimated money saved per conversation.

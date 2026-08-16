# OpenCode Behavior Specification

OpenCode-specific runtime behavior. Core behavior and safety guarantees live in [`Core.md`](Core.md).

## Commands

- `/magic-compact [N]` backs up and compacts the current OpenCode session in place.
- `/magic-trim [N]` backs up and trims historical tool I/O without summarizing messages.
- `/magic-stats` injects an ignored stats notice for the current session.
- `read_omitted_content` is registered as an OpenCode plugin tool.

`/magic-compact` and `/magic-trim` accept only a non-negative integer argument. `/magic-stats` accepts no arguments. Command handlers throw success, no-op, or validation messages so OpenCode does not continue sending the slash command to the LLM.

## Compaction Flow

1. Parse `N`; default is `0`.
2. Validate native checkpoint artifacts and build one per-turn compaction plan for the current session.
3. Stop early with a toast if no assistant turns are eligible.
4. Load the source session and compute the next `compactionCount`.
5. Count canonical active content locally with `gpt-tokenizer`; model-facing synthetic parts are included, ignored user reporting parts are excluded, ignored assistant text is included, and provider usage is telemetry only.
6. Render one deterministic capsule candidate for each selected turn. This is local code only: no hidden agent, no model discovery, no scratch session, no prompt, no abort, no retry, and no JSON transport. A turn already containing any Magic summary part yields no candidate and stays wholly opaque.
7. Reject a candidate as uncompactable when all required assistant text plus required current unresolved structured state cannot fit the 2,048-token safety ceiling. That turn remains message- and part-identical. If every candidate is opaque or uncompactable, stop as a no-op.
8. Project the complete final canonical active state, including capsule summaries, the model-facing boundary, and planned pruning/omission notices. Keep a turn in the mutation set only when its exact projected canonical active token count after summary injection and the same planned pruning is strictly lower than its exact source canonical active token count.
9. Recompute the eligible token base over selected profitable turns only and require at least 20% of those eligible tokens, subject to a small absolute floor, net of the fixed boundary overhead. A weak or empty result fails unchanged before backup or mutation and creates no stats increment.
10. Create and initialize the backup only after capsule generation and the savings gate pass. Copy omission and stats caches, rename the backup, and write metadata with the existing partial-initialization cleanup guarantees.
11. Revalidate the canonical full source snapshot after backup creation and immediately before the first durable source mutation. Behavior-affecting message, part, state, or ordering drift fails closed and preserves both sessions.
12. Insert new capsule parts in source order for selected profitable turns only, then inject the post-compaction boundary.
13. Immediately reload the full source. Remove exactly the owned capsule and boundary insertions from the comparison and require every remaining message, part, field, and ordering position to match the prevalidated source snapshot. Unexpected drift rolls back through the backup before pruning or accounting mutation.
14. Execute only the precomputed part-delete, part-update, and message-delete operations by their planned IDs. Each operation carries its explicit target IDs and, for tool updates, the ordered omission IDs to commit; no newly discovered part is swept. Commit only the omission IDs attached to planned tool updates, then update `compactionCount`.
15. Compute cumulative savings from the exact pre-notice canonical active result, inject the ignored stats notice with wording that says it is excluded, then count the actual final active state on the same local basis. Persist successful savings only after the count exactly matches. A mismatch is late source drift: preserve both the partially mutated source and named backup, provide manual recovery guidance, and record no successful stats increment.
16. Show a success toast.

## Capsule Contract

- `Outcome: completed` means canonical structured lifecycle has no unresolved current state.
- `Outcome: partial` means unresolved current structured state remains.
- The writer does not emit `analysis-only` or `blocked` outcomes.
- `Work:` contains every substantive non-synthetic non-Magic-summary assistant text part across every assistant message in the selected turn, in source order.
- Work entries are provider-safe JSON strings. `JSON.parse` of each fenced string recovers the exact source JavaScript code units, including controls, lone surrogates, U+FFFE, and U+FFFF.
- Assistant reasoning is intentionally removable and is not represented in the capsule.
- Existing `metadata.magicCompact.summary === true` text parts are opaque persisted data regardless of version. Recompaction skips the entire containing turn message- and part-identically and never parses, migrates, rewrites, or upserts its summary ID. Trim preserves the summary part itself.
- New deterministic capsules use `metadata.magicCompact.summary === true` under the versioned `prt_-magic_capsule_v1_${messageID}` namespace with `metadata.magicCompact.capsule === true` and `version === 1`. Historical v0 `prt_-magic_summary_${messageID}` parts keep their IDs, but no runtime v0 producer remains.
- `Verification`, `VCS`, `Unresolved`, and `Next` contain only canonical structured tool evidence and narrow allowlisted Hive/task result schemas. Optional successful recaps are allowed because retained tool parts stay authoritative.
- Assistant prose is not interpreted as decisions, mutations, blockers, verification, or VCS state. Ordinary text such as `We chose SQLite`, `Added`, `Removed`, technical fences, placeholders, and JSON-looking examples remains source text under `Work:`.

## Structured State

- Tool state is processed in source order with one current unresolved map.
- Completed tool transport is not automatically semantic success. Canonical `state.metadata.exit`, recognized verification count groups, tool lifecycle status, and allowlisted structured result schemas are evaluated before opaque output prose.
- Invalid canonical `state.metadata.exit` values fail closed. Nonzero exits remain unresolved. Exit zero proves only transport success and cannot suppress recognized nonzero verification failure/error counts or zero-run results.
- Verification parsing recognizes passed, failed/error, skipped, todo, cancelled, and not-run groups. Positive passed counts with zero failures are success even with skips. Zero executed counts or skipped/todo/cancelled-only results are not-run and cannot resolve a prior failure.
- Strong nonzero domain failures remain unresolved even when exit metadata is zero.
- Generic failure words in exit-zero git output are opaque; `git log`, `git show`, and `git diff` content is not status evidence.
- Later command success resolves only the same exact raw trimmed command plus a lexically normalized explicit absolute POSIX or Windows `cwd`/`workdir`. Relative paths, missing contexts, conflicting `cwd`/`workdir`, NUL-containing paths, and mixed-style paths are unscoped and fail closed. No realpath or process-cwd resolution is used.
- Known Hive/task producer identities may expose whole-object terminal result schemas through boolean `ok`/`success` and supported string `status` discriminators. Discriminator types, finite JSON structure, status values, success/failure consistency, and required `blocker.reason` or `error` details are strict. Malformed recognized schemas fail closed as unresolved while retaining the actionable result. Valid success is completed; valid blocked, failed, and partial results remain unresolved. Arbitrary read/API JSON such as `{status:"failed",error:"..."}` remains opaque.

## Trim Flow

1. Parse `N`; default is `0`.
2. Build turns for the complete stored session, independent of compaction boundaries.
3. Preserve the `N` most recent assistant turns.
4. Load the source session and fork it as a backup.
5. Copy omission and stats caches to the backup.
6. Measure pre-trim canonical active tokens with the same local counting used after trimming.
7. Apply the normal tool input and output trimming rules to older turns.
8. Mark processed completed tool states with `state.metadata.magicCompact.trimmed === true`.
9. Stop with a no-op toast if no tool states were processed.
10. Measure the exact post-trim canonical active count before the ignored reporting notice and add that reduction to conversation stats.
11. Inject a notice that reports the exact pre-notice active result and explicitly says the ignored notice is excluded, then require the actual final active count to match before showing a success toast.

`/magic-trim` does not call an LLM, generate summaries, modify ordinary user or assistant content, insert a compaction boundary, or increment `compactionCount`.

## Backup Sessions

- Backup title: `[Backup] ${title} ${timestamp}`.
- The main session title stays unchanged on success.
- Backup metadata stores `sourceSessionId`, `compactedAt`, and `compactionCount`.
- The backup receives copies of omission and stats caches before mutation.
- A fork is not considered a backup until cache, stats, title, and metadata initialization all succeed. Initialization failure deletes the target omission cache, target stats cache, and partial session; the primary error and every cleanup failure are aggregated.
- Trim backups preserve the source session's current `compactionCount`.
- For ordinary compaction or trimming failures after backup creation, the backup is renamed back to the original title and selected before the original session is deleted. If backup selection fails, the original session remains intact.
- Source drift detected by pre-write revalidation is not an ordinary rollback case: the live source and pre-race backup are both preserved, and neither session is selected or deleted.
- Source drift detected after owned capsule/boundary insertion but before pruning is an ordinary rollback case. The backup is promoted before the source is deleted, and no prune operation, omission commit, successful stats write, or source compaction metadata update occurs.
- Source drift detected after writes by the final actual-versus-projected canonical active-token mismatch also preserves both sessions without selecting, promoting, or deleting either.

## Turn Selection

- Messages are processed oldest-first.
- A turn is one or more adjacent user messages plus all following assistant messages before the next user group.
- Consecutive user/no-reply messages stay in the same turn.
- Boundary detection runs before ignoring a trailing assistantless turn.
- A trailing user-only turn does not count against `N`.
- Only turns with assistant messages are summarized.
- A valid completed native checkpoint is a user message with exactly one `compaction` part and exactly one native summary assistant child whose `parentID` is the marker ID, `summary` is `true`, `finish` is truthy, and `error` is absent.
- Every marker is checked against all of its native summary children. A completed child plus any additional completed, errored, or unfinished summary child is ambiguous. The only exception is an older marker whose complete ambiguous or incomplete artifact set lies wholly before the marker of a later valid checkpoint and is already inside that checkpoint's frozen prefix.
- Completed checkpoint intervals must be sequential. An interval may not contain another marker or a foreign native summary; nested and crossed intervals are rejected as ambiguous.
- The newest valid completed native checkpoint owns an immutable prefix through its summary assistant. Turn construction starts strictly after that summary.
- Every completed checkpoint candidate's `tail_start_id`, when present, must resolve to a durable message strictly before its marker. The retained tail, marker, summary message, summary text, and summary parts remain untouched.
- Magic boundaries before the native checkpoint are ignored. The latest Magic boundary in the eligible suffix still controls recompaction.
- `N` preserves the most recent completed assistant turns in the eligible suffix only.
- If the suffix has no eligible turns after applying `N`, the command is the existing no-op: no backup, summary, boundary, stats, or mutation is created.
- For `/magic-trim`, `N` preserves tool I/O in the most recent assistant turns across the complete session.
- Trim selection does not use compaction boundaries.

## Recompaction

- Any turn containing a previous Magic summary part is preserved as-is, including v0 and canonical v1 IDs, even if it falls inside the selected range because a historical boundary is absent.
- Recompaction recognizes summaries by metadata, not by parsing summary text or assuming a single part ID namespace.
- Recompaction starts at the latest boundary marker.
- The boundary marker is a user text part with `metadata.magicCompact.boundary === true`.
- Earlier turns before the latest boundary are outside the current compaction range.
- A later completed native checkpoint advances the immutable floor beyond any older native or Magic boundary.
- Turns rejected by the per-turn profitability gate are never summarized, pruned, or otherwise mutated by that run, and they are not retried by later runs because the new boundary freezes the prefix before them.

## Omission Cache

- Location: `${XDG_DATA_HOME:-~/.local/share}/opencode/storage/magic-compact/{sessionId}.json`.
- Cache format version is `2`.
- Newly allocated IDs are `<last-12-session-chars>:omitted-<22-char-base64url-token>`, where the token contains 128 random bits. The prefix is diagnostic only and is never used to route a lookup.
- Each v2 entry stores content and full SHA-256 over the fixed `magic-compact:omission-entry:v2\0` domain/version prefix, a four-byte big-endian length and the exact UTF-8 qualified Content ID bytes, then the exact JavaScript content from `Buffer.from(content, "utf16le")`. This binds an entry to both its ID and ordered UTF-16LE code units, preserves lone surrogates, and performs no Unicode normalization.
- Valid v1 bare entries are preserved under `legacy.entries` during migration and backup copying, but no new bare ID is allocated and bare legacy bytes are never returned.
- An absent source cache materializes as an empty v2 cache when copied to a backup. Malformed, schema-invalid, or integrity-invalid caches fail loudly and are not reset or overwritten.
- Cache writes use atomic temporary-file replacement. In-process mutations for one session are serialized; distinct sessions remain independent.
- The current session cache is the active cache on success.
- The backup gets a cache copy before mutation.

## Omission Retrieval

- The plugin exposes `read_omitted_content` as an OpenCode plugin tool.
- The tool accepts one argument: `contentId`.
- The tool receives `context.sessionID` from OpenCode and reads only that session's omission cache.
- Retrieval never routes by the diagnostic ID prefix or opens another session cache.
- If no authorized matching qualified entry exists, integrity proof fails, the cache is invalid, or a bare legacy ID is requested, the tool returns an explicit unavailable/integrity diagnostic and never guessed bytes.

## Stats

- Stats are stored under `${XDG_DATA_HOME:-~/.local/share}/opencode/storage/magic-compact/stats/{sessionId}.json`.
- Stats cache format version is `1`.
- Stats track `rootSessionId`, `sourceSessionId`, `totalTokensPruned`, `cachedTokensSaved`, and processed assistant message IDs.
- Each compaction adds the current token reduction to `totalTokensPruned`.
- Each trim adds the exact reduction from its pre-trim canonical active count to the post-trim pre-notice active count.
- OpenCode assistant message events add `totalTokensPruned` to `cachedTokensSaved` once per assistant message after stats exist.
- `/magic-stats` injects an ignored stats summary notice, or a no-stats message if no stats exist.
- Every stats notice labels its counts as canonical active-content estimates and distinguishes them from provider context-window usage. Magic Compact never fabricates an after-provider count or mutates provider token metadata.

## Pruning

- Pruning applies only to summarized turns after summary insertion and boundary injection.
- Before the first prune mutation, the full selection is rejected if it contains a native `compaction` part or native summary assistant.
- Projection plans every delete, tool replacement with its ordered omissions, and empty-assistant deletion by explicit message/part IDs. Execution consumes this plan directly rather than deriving mutations from a live reload.
- Immediately before execution, full-source post-insertion validation rejects every unexpected message/part addition, removal, change, or reorder. Newly discovered parts are never swept.
- Synthetic user text parts are deleted unless they are preserved OpenCode wrappers or reminders.
- Summarized assistant messages keep summary parts and tool parts.
- Other assistant parts are deleted.
- Assistant messages with no remaining parts are deleted.
- Only completed tool parts are pruned; pending, running, and error states are preserved.
- `/magic-trim` applies only the tool rules below; it does not delete other user or assistant parts.
- Processed tool states are marked as trimmed, and later trim or compaction operations skip them.

## Tool Rules

- `write`: omit large `input.content` and cache it.
- `edit`: omit large `oldString` and `newString` together and cache once.
- `apply_patch`: omit large `input.patchText` and cache it.
- `bash`: cache long commands and visibly truncate to the first 512 characters plus `[REST OF COMMAND TRUNCATED]`.
- `read`: always cache and omit output.
- `task`: cache and omit output only above the higher task threshold.
- `todowrite`: preserve input and replace output with a success message, no cache.
- `question`: preserve input and output.
- `skill`: preserve input and replace output with a reload hint, no cache.
- Other completed tool outputs are cached and omitted when they exceed the default threshold.

## Error Handling

- Any deterministic rendering, SDK, cache, stats, token counting, projection, or pruning failure aborts the attempt.
- Compaction parts on non-user messages, multiple compaction parts, ambiguous summary children for one marker, orphan summaries, overlapping checkpoint intervals, dangling or forward `tail_start_id` values on any completed candidate, and active incomplete or errored markers fail clearly before backup or mutation.
- The ordered canonical source snapshot is captured during planning and compared structurally after backup creation immediately before any Magic capsule, boundary, or pruning write.
- All capsules, exact prune mutations, planned omission IDs, and projected counts remain in memory until capsule generation completes, the useful-savings gate passes, backup initialization succeeds, and pre-write source revalidation passes.
- After capsule and boundary insertion, the full source is compared again against the original snapshot plus exactly those owned insertions. Drift at this gate rolls back before pruning. Other ordinary failures also promote the backup before deleting the source. Final actual-versus-projected mismatch is instead a preserve-both source-drift failure and names the backup for manual recovery.
- No-op, success, and failure toast SDK responses are unwrapped. Failure-toast errors are retained with the primary and rollback errors. A success-toast failure occurs after durable compaction and surfaces without promoting the backup or deleting the successful source.
- The command hook throws so OpenCode does not continue sending the slash command to the LLM.

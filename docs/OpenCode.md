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
5. Count canonical model-facing active content locally with `gpt-tokenizer` and `projectManifestCoveredMedia`. Include model-facing synthetic parts, exclude ignored user reporting parts, include ignored assistant text, and retain the unprojected count for separate physical-pruning accounting. Provider usage remains telemetry only.
6. Exclude every turn already containing `metadata.magicCompact.summary === true`. For each remaining selected turn, derive one bounded index entry: a monotonic ordinal, the exact escaped first ordinary non-synthetic/non-ignored user text line capped at 300 characters, and the count of authoritative user then assistant inline `data:` `image/*` parts. Validate generated summary/boundary part-ID availability and build each v3 media manifest locally from encounter ordinal, exact MIME, and SHA-256 of the exact URL. Tool `state.attachments` do not enter the persisted manifest.
7. Resolve the native hidden compaction agent and require its explicit model configuration. Use that exact provider/model and variant; never use the source session model as a fallback. Scan the complete source snapshot that may reach the fork, including retained and opaque summary-bearing turns, for image-bearing user or assistant file parts and tool `state.attachments`. If any image may be model-visible, require the provider catalog's configured model to report `capabilities.input.image === true` before creating a fork.
8. Create exactly one ephemeral fork of the complete source session under the shared control-operation deadline. Immediately update that fork to the deny-all permission ruleset, load its rewritten-ID messages, and verify chronological message/part-type and turn shape against the source plan. Choose a crypto-quality 128-bit generation nonce absent from every ordinary string in both the source snapshot and fork history, retrying collisions only within a fixed bound. Inject a synthetic Magic-owned nonce-and-ordinal marker into the first user message of every selected candidate turn and a nonce-bound stop marker on `nextTurn`. These markers exist only in the ephemeral fork.
9. Prompt exactly once through `magic-compact-summarizer`. The prompt has one text part containing only short instructions and the bounded exact-marker/ordinal/anchor/image-count index. It contains no SDK records, source refs, assistant traces, tool state, data URLs, provider metadata, encrypted reasoning, or file parts.
10. OpenCode's native history conversion supplies the complete forked history to the configured model. User files become typed file content. Completed tool `state.output` becomes typed tool-result text/content and `state.attachments` become typed media/file content. Magic Compact never manually serializes, redacts, splits, or reattaches source or tool media.
11. Await the model request without a local wall-clock deadline. Parse exactly one substantive text response as JSON with one `turns` array. Require exact result count, marker echo, ordinal sequence, anchor echo, nonempty summary, and exactly one nonempty visual description per expected image ordinal. Missing, stale, spoofed, duplicate, shifted, reordered, or empty values fail before backup/source mutation. Any summary or visual description containing the active exact generation nonce or marker token also fails before framing; unrelated nonce-shaped text remains valid. Marker nonces are discarded after validation and never persist in visible summary text; useful source identifiers may remain when the model includes them.
12. Provider and validation failure attempt bounded abort; deletion is always attempted exactly once after a created fork. A late fork creation receives detached bounded abort/delete cleanup. Safe nested context-window, prompt-too-long, input-too-long, and HTTP 413 categories report trim-first guidance. Output-length and max-token error or finish categories report increase-`N` guidance. Raw provider text is never surfaced. Do not retry, split into per-turn or chunk requests, switch models, or partially accept results.
13. Project the complete final canonical model-facing state with the same pure manifest projection used by the OpenCode hook, including v3 summaries, the model-facing boundary, planned pruning/omission notices, and declarative media placeholders. Keep a turn only when this exact projected count is strictly lower than its exact source count.
14. Recompute the eligible token base over selected profitable turns only and require at least 20% savings, subject to a small absolute floor and net of the fixed boundary overhead. A weak or empty result fails unchanged before backup or mutation and creates no stats increment.
15. Create and initialize the backup only after summary generation and the savings gate pass. Copy omission and stats caches, rename the backup, and write metadata with the existing partial-initialization cleanup guarantees.
16. Revalidate the canonical full source snapshot after backup creation and immediately before the first durable source mutation. Behavior-affecting message, part, state, or ordering drift fails closed and preserves both sessions.
17. Insert new v3 summary parts in source order for selected profitable turns only, then inject the post-compaction boundary. A media summary stores `metadata.magicCompact.mediaManifest` version `1`. Original file parts and URLs are not updated or deleted.
18. Reload the full source. Remove exactly the owned summary and boundary insertions from the comparison and require every remaining message, part, field, and ordering position to match the prevalidated source snapshot. Unexpected drift preserves both the live source and backup without selecting or deleting either, before pruning or accounting mutation, and reports both session IDs for manual recovery.
19. Execute only the precomputed part-delete, part-update, and message-delete operations by their planned IDs. Each tool update carries its ordered omission IDs; no newly discovered part is swept. Commit only those planned omission IDs, then update `compactionCount`.
20. Compute cumulative savings from the exact pre-notice results, keeping net physical token pruning separate from current inline-image projection. Inject the ignored stats notice, then count the actual final model-facing state with the same projection. Persist successful savings only after token and projected-media counts exactly match. A mismatch preserves the partially mutated source and named backup and records no successful stats increment.
21. Show a success toast. Any failure from the first summary insertion attempt onward preserves both source and backup without selecting either or deleting the live source, surfaces both session IDs for manual recovery, and records no successful stats increment.

## Summary Contract

- The model returns one plain JSON object in exactly one substantive text part. The object has exactly one key, `turns`; every turn object has exactly `ordinal`, `marker`, `anchor`, `summary`, and `visualDescriptions`.
- `marker`, `ordinal`, and `anchor` must exactly match the bounded request index in order. Exact marker echo is required in the `marker` field. Only the active exact generation nonce or marker token is forbidden inside `summary` and visual descriptions; useful source message, part, session, parent, and tool-call identifiers may remain when the model includes them. `summary` must be a nonempty string and must not contain the active exact nonce or marker token. The system prompt asks for readable continuation-focused Markdown preserving outcome, important process/decisions, verification, and unresolved/next work when present; the host does not impose a semantic status or evidence protocol.
- `visualDescriptions` is empty for a turn with no authoritative user or assistant inline images. Otherwise it has exactly one `{ordinal, description}` object per expected image in user-then-assistant encounter order, with a nonempty description that does not contain the active exact nonce or marker token. The host appends accepted descriptions under `Visual summary:`.
- Existing `metadata.magicCompact.summary === true` text parts are opaque persisted data regardless of version. Recompaction skips the entire containing turn message- and part-identically and never parses, migrates, rewrites, or upserts its summary ID. Trim preserves the summary part itself.
- New summaries use `metadata.magicCompact.summary === true` under `prt_-magic_capsule_v3_${messageID}` with `capsule === true`, `agentic === true`, and `version === 3`. A media-aware summary additionally stores `mediaManifest: {version: 1, images: [{ordinal, mime, urlSha256}]}` for inline user then assistant images only. Historical v0/v1/v2 parts keep their bytes and IDs, but no legacy producer remains.

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
11. Inject a notice that reports the exact pre-notice result and explicitly says the ignored notice is excluded, then require the actual final active count to match before showing success.

`/magic-trim` does not call an LLM, generate summaries, modify ordinary user or assistant content, insert a compaction boundary, or increment `compactionCount`.

## Backup Sessions

- Backup title: `[Backup] ${title} ${timestamp}`.
- The main session title stays unchanged on success.
- Backup metadata stores `sourceSessionId`, `compactedAt`, and `compactionCount`.
- The backup receives copies of omission and stats caches before mutation.
- A fork is not a backup until cache, stats, title, and metadata initialization all succeed. Initialization failure deletes the target caches and partial session; the primary and cleanup failures are aggregated.
- Trim backups preserve the source session's current `compactionCount`.
- Compaction never auto-applies the backup; every failure after backup creation preserves both the live source and backup for manual recovery. Trim keeps its existing backup recovery behavior: on a trim failure it renames the backup to the original title, selects it, and deletes the original; if selection fails, the original remains intact.
- Source drift detected by pre-write revalidation preserves the live source and pre-race backup without selecting or deleting either.
- Every failure from the first summary insertion attempt onward preserves the live source and backup without selecting, promoting, or deleting either. Its content-free error names both session IDs for manual recovery, concurrent messages remain live, and no successful stats increment is recorded.

## Turn Selection

- Messages are processed oldest-first.
- A turn is one or more adjacent user messages plus all following assistant messages before the next user group.
- Consecutive user/no-reply messages stay in the same turn.
- Boundary detection runs before ignoring a trailing assistantless turn.
- A trailing user-only turn does not count against `N`.
- Only turns with assistant messages are summarized.
- A valid completed native checkpoint is a user message with exactly one `compaction` part and exactly one native summary assistant child whose `parentID` is the marker ID, `summary` is `true`, `finish` is truthy, and `error` is absent.
- Every marker is checked against all native summary children. Ambiguous, incomplete, errored, nested, crossed, orphaned, dangling, or forward checkpoint artifacts fail closed, except an older complete artifact set wholly frozen by a later valid checkpoint.
- The newest valid checkpoint owns an immutable prefix through its summary assistant. Turn construction starts strictly after that summary.
- Every completed checkpoint candidate's `tail_start_id`, when present, resolves to a durable message strictly before its marker. The retained tail, marker, summary message, summary text, and summary parts remain untouched.
- Magic boundaries before the native checkpoint are ignored. The latest Magic boundary in the eligible suffix controls recompaction.
- `N` preserves the most recent completed assistant turns in the eligible suffix only.
- If the suffix has no eligible turns after applying `N`, no backup, generation fork, boundary, stats, or mutation is created.
- `/magic-trim` selection spans the complete session and ignores compaction boundaries.

## Recompaction

- Any turn containing a previous Magic summary part is preserved as-is, including v0/v1/v2 and current v3 envelopes, even if a historical boundary is absent.
- Persisted media-aware v2 turns remain opaque. Their manifests and current v3 manifests still project independently; summaries without manifests receive no retroactive repair.
- Recompaction recognizes summaries by metadata, not by summary text or one ID namespace.
- Recompaction starts at the latest boundary marker, unless a later native checkpoint advances the immutable floor.
- Turns rejected by the per-turn profitability gate are never summarized or pruned by that run and fall before the new boundary.

## Media Request Projection

- The server plugin registers `experimental.chat.messages.transform`. OpenCode invokes it with the retained messages array and later converts that same array, so the hook computes `projectManifestCoveredMedia` and atomically replaces the retained array contents via `splice`. Projected messages and parts are clones; a failure leaves the retained array byte-identical.
- Projection considers persisted v2 and current v3 summary envelopes with `mediaManifest`. A turn without that manifest is unchanged.
- A v3 manifest-bearing turn must have exactly the same ordered set of inline user then assistant `data:` `image/*` file parts. Persisted v2 manifests remain user-only. Manifest version/shape, sequential encounter ordinals, exact MIME values, and SHA-256 of each exact URL must all match. Persisted v2 entries still require valid unique `partId` compatibility fields, but projection does not compare those ephemeral IDs.
- Every exact match is replaced by one synthetic declarative text placeholder with the same record identifiers only when the placeholder's canonical token count is strictly smaller than the original file URL token contribution; otherwise the exact file part stays inline and model-facing. Any missing, added, duplicate, reordered, MIME-changed, URL-changed, or malformed entry throws before model conversion; the hook never silently falls back to changed media. Because every replacement strictly reduces canonical tokens, projected media token counts are non-negative by construction.
- This same function drives per-turn profitability, complete post-compaction projection, final token verification, and model-facing stats. It never rewrites SQLite, backups, API records, or exports.

## Omission Cache

- Location: `${XDG_DATA_HOME:-~/.local/share}/opencode/storage/magic-compact/{sessionId}.json`.
- Cache format version is `2`.
- Newly allocated IDs are `<last-12-session-chars>:omitted-<22-char-base64url-token>`, where the token contains 128 random bits. The prefix is diagnostic only and never routes a lookup.
- Each v2 entry stores content and a SHA-256 binding over its qualified ID and exact JavaScript UTF-16LE code units. Lone surrogates are preserved and Unicode is not normalized.
- Valid v1 bare entries are preserved under `legacy.entries` during migration and backup copying, but no new bare ID is allocated and bare legacy bytes are never returned.
- An absent source cache materializes as an empty v2 cache when copied to a backup. Malformed, schema-invalid, or integrity-invalid caches fail loudly and are not reset.
- Cache writes use atomic temporary-file replacement. In-process mutations for one session are serialized; distinct sessions remain independent.

## Omission Retrieval

- `read_omitted_content` accepts one `contentId` and receives `context.sessionID` from OpenCode.
- Retrieval reads only the current session cache and never routes by the diagnostic ID prefix.
- Missing authorization, no exact entry, invalid integrity, malformed cache, and bare legacy IDs return explicit unavailable/integrity diagnostics and never guessed bytes.

## Stats

- Stats are stored under `${XDG_DATA_HOME:-~/.local/share}/opencode/storage/magic-compact/stats/{sessionId}.json`.
- Stats cache format version is `1`; media fields remain optional for existing files.
- Stats track root/source session IDs, net physically pruned canonical tokens, physical cached reads, projected inline-image tokens, projected-media cached reads, and processed assistant message IDs.
- Compaction and trim add exact canonical reductions. Assistant completion events add current physical and projected-media savings once per assistant message.
- `/magic-stats` injects an ignored stats notice, or a no-stats message if no stats exist.
- Notices distinguish canonical model-facing estimates from provider context usage. Magic Compact never fabricates provider counts or mutates provider metadata.

## Pruning

- Pruning applies only to summarized turns after summary insertion and boundary injection.
- User file/image parts and assistant inline images covered by an accepted v3 manifest are never prune targets or omission entries. Exact persisted URLs remain authoritative.
- Before the first prune mutation, the selection is rejected if it contains a native `compaction` part or native summary assistant.
- Projection plans every delete and tool replacement by explicit ID. Immediately before execution, full-source post-insertion validation rejects unexpected additions, removals, changes, or reorderings while preserving both source and backup. Newly discovered parts are never swept.
- Synthetic user text is deleted unless it is a preserved OpenCode wrapper or reminder.
- Summarized assistant messages keep summary parts, tool parts, and v3 manifest-covered inline image files. Other assistant parts are deleted; empty assistant messages are deleted.
- Only completed tool parts are pruned. Pending, running, and error states remain.
- `/magic-trim` applies only tool rules and marks processed states to prevent repeated trimming.

## Tool Rules

- `write`: omit large `input.content` and cache it.
- `edit`: omit large `oldString` and `newString` together and cache once.
- `apply_patch`: omit large `input.patchText` and cache it.
- `bash`: cache long commands and visibly truncate them.
- `read`: always cache and omit output.
- `task`: cache and omit output only above the higher task threshold.
- `todowrite`: preserve input and replace output with a success message, no cache.
- `question`: preserve input and output.
- `skill`: preserve input and replace output with a reload hint, no cache.
- Other completed tool outputs are cached and omitted above the default threshold.

## Error Handling

- Any summarizer configuration, fork lifecycle, request, response contract, media manifest, SDK, cache, stats, token count, projection, or pruning failure aborts the attempt.
- Generation errors expose only fixed lifecycle stage/kind text, source session ID, validation category, and range-validated HTTP status. Raw provider/control/model errors, bodies, prompts, fork history, and model output are not retained in the surfaced graph.
- Native context overflow reports trim-first guidance. No retry, chunking, alternate model, or partial mutation path exists.
- Native checkpoint errors fail before generation. Ordered source snapshots are revalidated after backup creation and after owned summary/boundary insertion.
- Pre-write source drift and every failure after backup creation preserve both sessions; post-write errors report only the source and backup IDs plus manual-recovery guidance and never record a successful stats increment. Compaction never auto-applies the backup; trim retains its backup rollback path.
- No-op, success, and failure toasts use the shared control-operation deadline. Toast timeout or failure cannot replace or delay the primary command outcome.
- The command hook throws so OpenCode does not continue sending the slash command to the LLM.

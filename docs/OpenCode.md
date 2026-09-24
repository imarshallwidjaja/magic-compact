# OpenCode Behavior Specification

OpenCode-specific runtime behavior. Core behavior and safety guarantees live in [`Core.md`](Core.md).

## Commands

- `/magic-compact [N]` backs up and compacts the current OpenCode session in place.
- `/magic-trim [N]` backs up and trims historical tool I/O without summarizing messages.
- `/magic-stats` injects an ignored stats notice for the current session.
- `read_omitted_content` is registered as an OpenCode plugin tool.

`/magic-compact` and `/magic-trim` accept only a non-negative integer argument. `/magic-stats` accepts no arguments. Command handlers throw success, no-op, or validation messages so OpenCode does not continue sending the slash command to the LLM.

After argument and execution-option validation, compact and trim show a long-lived `Compacting session...` or `Trimming session...` TUI toast before slow work, then send terminal feedback. If a timed-out progress delivery later settles after terminal feedback was sent, Magic Compact best-effort reasserts the terminal toast. These toasts are view-only: they do not append messages, enter model input, change stats, or create omission-cache entries. Delivery is bounded by the shared control deadline and cannot change the primary or recovery outcome. OpenCode exposes no later delivery signal after the client promise rejects, so Magic Compact cannot detect and correct a progress toast that the host applies after such a rejection.

## Compaction Flow

1. Parse `N`; default is `0`.
2. Validate native checkpoint artifacts and build one per-turn compaction plan for the current session.
3. Stop early with a toast if no assistant turns are eligible.
4. Load the source session, compute the next `compactionCount`, and fail closed with a pending-revert error when the session has an unresolved OpenCode revert, because OpenCode's prompt-time revert cleanup silently deletes the revert suffix on any `session.prompt`.
5. Count canonical model-facing active content locally with `gpt-tokenizer` and `projectManifestCoveredMedia`. Include model-facing synthetic parts, exclude ignored user reporting parts, include ignored assistant text, and retain the unprojected count for separate physical-pruning accounting. Provider usage remains telemetry only.
6. Build one immutable global orientation index from every complete turn in the planner's effective source range after native-checkpoint filtering. Include eligible targets, opaque existing-summary turns, and retained-tail complete turns; exclude the active incomplete turn. Each entry has exactly `sourceOrdinal`, the stable one-based complete-turn position in full source order, and `anchor`, the exact escaped first ordinary non-synthetic/non-ignored user text line capped at 160 characters. Then exclude every turn already containing `metadata.magicCompact.summary === true` from target candidates. For each remaining selected target, derive a separate exact index entry with a monotonic target ordinal, its first ordinary user line capped at 300 characters, and the count of authoritative user then assistant inline `data:` `image/*` parts. Validate generated summary/boundary part-ID availability and build each v3 media manifest locally from encounter ordinal, exact MIME, and SHA-256 of the exact URL. Tool `state.attachments` do not enter the persisted manifest.
7. Resolve the native hidden compaction agent and require its explicit model configuration. Resolve that provider/model's live descriptor once and use the configured variant; never use the source model as a fallback. Set output target to `min(32_000, floor(limit.output * 0.5))` with the bounded fallback, usable input to `min(limit.input when present, limit.context - output target)`, normal input target to `min(400_000, floor(usable input * 0.8))`, hard single-turn input to `floor(usable input * 0.9)`, and halo input to `min(500_000, hard single-turn input)`.
8. Estimate each target turn without serializing messages. Tokenize ordinary visible text and reasoning through the pinned local tokenizer. Charge image/PDF-typed file parts and completed non-compacted tool `state.attachments` 8K each; ignore `text/plain` and directory file parts as non-model-visible, and estimate other model-visible file parts and attachments by their URL (encoded length for `data:` URLs, tokenized text otherwise). Treat arbitrary or nested `data:` strings in tool input, completed output, error text, and model-visible interrupted `state.metadata.output` as ordinary provider-request text, estimated conservatively by encoded length without invoking the tokenizer on their payloads. Mirror native compacted, interrupted, error, pending, and running output selection. Reserve 8K only for non-explicit host, system, and provider overhead. Separately render and tokenize the complete explicit request exactly once for every tentative batch: static instructions, byte-identical global index, local target JSON/punctuation, fixed-shape marker and injected marker text, exact ordinal digits, 300-code-point anchors, media counts, and any following-halo stop record/marker. Charge the fixed schema JSON for the model-visible StructuredOutput tool against explicit input; the host-injected structured-output instruction remains in the non-explicit reserve. Output estimates add summary/description content to the mandatory response envelope, punctuation, echoed marker/ordinal/anchor fields, `summary` property, and every visual-description record. Partition targets chronologically with no split turns or bin packing, at most 32 targets per batch, and a dedicated batch for a turn above the normal target. If the complete request or mandatory global index cannot fit, fail before fork creation with trim-first/model-capacity guidance. If one turn exceeds the hard limit, fail before backup with `/magic-trim`-first guidance; if its complete estimated response exceeds the output target, require a model with more output capacity.
9. For every planned batch, retain complete target turns plus at most one complete neighboring non-checkpoint turn on each side only while native history plus the complete explicit request stays within the halo limit. A following halo is tested with its bounded anchor, target-index stop record, and injected stop marker included. Then directly await a fresh complete source fork without a local wall-clock deadline and immediately apply the ordered session rule that denies operational tools and allows only the injected `StructuredOutput` response tool under the control deadline. Load and verify the full rewritten-ID chronological role/part-type shape. Before deletion, capture the retained rewritten message IDs, part IDs, exact order, roles, parent links, part types, and fork-scoped session/message/compaction-tail references. Delete every other fork message sequentially through `session.deleteMessage`; never delete parts for generation isolation or retain distant turns merely because they appear in the global index. Reload and require that exact retained identity before markers.
10. Plan marker fields with `NATIVE_BATCH_PLANNING_NONCE`, a 22-byte fixed shape whose bytes each tokenize separately under the pinned tokenizer. Before creating any fork, choose each unique crypto-quality 128-bit nonce absent from the source, render the exact immutable prompt, and require its explicit-request and response-protocol counts not to exceed the conservative records or the normal, hard, halo, usable-input, and output limits. Recheck the nonce against rewritten retained-fork strings, then inject nonce/target-ordinal markers only on target user messages and an optional stop marker on the following halo. Prompt that fork exactly once through `magic-compact-summarizer` with the already-checked prompt. The one text part contains short instructions, the byte-identical `{globalTurnIndex: [{sourceOrdinal, anchor}]}` map, and the separate exact marker/target-ordinal/target-anchor/image-count index. The global map is orientation-only and carries no distant assistant/tool content. OpenCode's native conversion supplies only retained user, assistant, reasoning, tool output/attachments, and file media; Magic Compact never manually serializes or reattaches them. Require image capability if the union of retained batch histories may expose an image.
11. Run independent batches with maximum concurrency five. Every prompt, index, history isolation plan, and estimate derives from the same immutable pre-generation source plan; generated summaries never alter later requests. Await every model request without a local wall-clock deadline. Request `format: {type: "json_schema", schema}` on the one generation prompt per fork. OpenCode 1.18.30 injects the side-effect-free `StructuredOutput` response tool for the JSON-schema request, resolves the final model tool map through the merged agent-and-session rules where the last matching rule wins, and records its captured response on assistant `info.structured`. The fork's ordered session rule (wildcard deny, then exact `StructuredOutput` allow) keeps that host response tool while every operational tool stays denied. The installed 1.18.30 prompt path does not use `retryCount`, so Magic Compact does not send it. Magic Compact makes one `session.prompt` call per fork and adds no plugin-level retry or fallback; OpenCode may apply its runtime provider/transport retry policy or additional model steps required by structured-output processing. Treat the captured object as untrusted: require exact batch result count, target ordinal sequence, marker/anchor echo, nonempty summaries, and exact visual coverage. Text parts are never parsed as a fallback. The response packing estimate includes the mandatory JSON envelope and all echoed protocol fields. Reject active nonce leakage. The first failure stops new batch admission, but every already-active worker drains through cleanup and contributes its sanitized failure in deterministic batch order. Then require exact once-only target ordinal coverage before framing any results for the mutation flow.
12. Any planning, batch, provider, throttling, validation, lifecycle, or cleanup failure discards every generated result and leaves source, backup, and stats unchanged. Provider or validation failure directly awaits one abort attempt. Every created fork is directly deleted before its worker settles, and genuine abort or deletion errors join the sanitized failure. The first failure stops new admission, but every concurrently started batch drains through its owned cleanup before command settlement. Native context overflow reports trim-first guidance; single-turn output exhaustion requires a model with more output capacity, while multi-turn exhaustion reports larger-`N` guidance. Magic Compact adds no plugin-level retry, adaptive repartitioning, model switching, or partial result acceptance; OpenCode may apply its own provider/transport retry policy or additional model steps required by structured-output processing.
13. Project the complete final canonical model-facing state with the same pure manifest projection used by the OpenCode hook, including v3 summaries, the model-facing boundary, planned pruning/omission notices, and declarative media placeholders. Keep a turn only when this exact projected count is strictly lower than its exact source count.
14. Recompute the eligible token base over selected profitable turns only and require at least 20% savings, subject to a small absolute floor and net of the fixed boundary overhead. A weak or empty result fails unchanged before backup or mutation and creates no stats increment.
15. Create and initialize the backup only after summary generation and the savings gate pass. Copy omission and stats caches, rename the backup, and write metadata with the existing partial-initialization cleanup guarantees.
16. Revalidate the canonical full source snapshot after backup creation and immediately before the first durable source mutation. Behavior-affecting message, part, state, or ordering drift fails closed and preserves both sessions.
17. Recheck the source session for a pending revert immediately before the first source write and fail closed without any source mutation if one appeared. Insert new v3 summary parts in source order for selected profitable turns only, then inject the post-compaction boundary. A media summary stores `metadata.magicCompact.mediaManifest` version `1`. Original file parts and URLs are not updated or deleted.
18. Reload the full source. Remove exactly the owned summary and boundary insertions from the comparison and require every remaining message, part, field, and ordering position to match the prevalidated source snapshot. Unexpected drift preserves both the live source and backup without selecting or deleting either, before pruning or accounting mutation, and reports both session IDs for manual recovery.
19. Execute only the precomputed part-delete, part-update, and message-delete operations by their planned IDs. Each tool update carries its ordered omission IDs; no newly discovered part is swept. Commit only those planned omission IDs, then update `compactionCount`.
20. Compute cumulative savings from the exact pre-notice results, keeping net physical token pruning separate from current inline-image projection. Inject a concise ignored stats notice reporting the exact pre-notice before/after context counts, the trimmed amount and percentage, and the number of summarized turns, then count the actual final model-facing state with the same projection. Persist successful savings only after token and projected-media counts exactly match. A mismatch preserves the partially mutated source and named backup and records no successful stats increment.
21. Show a success toast. Any failure from the first summary insertion attempt onward preserves both source and backup without selecting either or deleting the live source, surfaces both session IDs for manual recovery, and records no successful stats increment.

## Summary Contract

- Each batch captures one structured object on assistant `info.structured`. The fixed JSON schema requires one `turns` array and exact object keys and types, while local checks enforce batch-specific identity and coverage. Every turn object has exactly `ordinal`, `marker`, `anchor`, `summary`, and `visualDescriptions`. Packing budgets the envelope/array punctuation and every mandatory field before estimated summary and description content. These target ordinals remain monotonic across batches and are distinct from global-index `sourceOrdinal` values.
- `marker`, `ordinal`, and `anchor` must exactly match the bounded request index in order. Exact marker echo is required in the `marker` field. Only the active exact generation nonce or marker token is forbidden inside `summary` and visual descriptions; useful source message, part, session, parent, and tool-call identifiers may remain when the model includes them. `summary` must be a nonempty string and must not contain the active exact nonce or marker token. The system prompt asks for readable continuation-focused Markdown preserving outcome, important process/decisions, verification, and unresolved/next work when present; the host does not impose a semantic status or evidence protocol.
- `visualDescriptions` is empty for a turn with no authoritative user or assistant inline images. Otherwise it has exactly one `{ordinal, description}` object per expected image in user-then-assistant encounter order, with a nonempty description that does not contain the active exact nonce or marker token. The host appends accepted descriptions under `Visual summary:`.
- Existing `metadata.magicCompact.summary === true` text parts are opaque persisted data regardless of version. Recompaction skips the entire containing turn message- and part-identically and never parses, migrates, rewrites, or upserts its summary ID. Trim preserves the summary part itself.
- New summaries use `metadata.magicCompact.summary === true` under `prt_-magic_capsule_v3_${messageID}` with `capsule === true`, `agentic === true`, and `version === 3`. A media-aware summary additionally stores `mediaManifest: {version: 1, images: [{ordinal, mime, urlSha256}]}` for inline user then assistant images only. Historical v0/v1/v2 parts keep their bytes and IDs, but no legacy producer remains.

## Trim Flow

1. Parse `N`; default is `0`.
2. Show the view-only in-progress toast.
3. Build turns for the complete stored session, independent of compaction boundaries.
4. Preserve the `N` most recent assistant turns.
5. Load the source session and fork it as a backup.
6. Copy omission and stats caches to the backup.
7. Measure pre-trim canonical active tokens with the same local counting used after trimming.
8. Apply the normal tool input and output trimming rules to older turns.
9. Mark processed completed tool states with `state.metadata.magicCompact.trimmed === true`.
10. If no tool states were processed, delete the initialized backup and its copied caches before showing the no-op toast.
11. Measure the exact post-trim canonical active count before the ignored reporting notice and add that reduction to conversation stats.
12. Inject a concise ignored notice reporting the exact pre-notice before/after context counts, the trimmed amount and percentage, and the number of trimmed tool calls, then require the actual final active count to match before showing success.

`/magic-trim` does not call an LLM, generate summaries, modify ordinary user or assistant content, insert a compaction boundary, or increment `compactionCount`.

## Backup Sessions

- Backup title: `[Backup] ${title} ${timestamp}`.
- The main session title stays unchanged on success.
- Backup metadata stores `sourceSessionId`, `compactedAt`, and `compactionCount`.
- The backup receives copies of omission and stats caches before mutation.
- A fork is not a backup until cache, stats, title, and metadata initialization all succeed. Initialization failure deletes the target caches and partial session; the primary and cleanup failures are aggregated.
- Trim backups preserve the source session's current `compactionCount`. A no-op trim deletes its fully initialized backup and copied caches before reporting no-op; if backup-session deletion fails, its valid cache and stats copies remain intact and the cleanup failure is surfaced.
- Compaction never auto-applies the backup; every failure after backup creation preserves both the live source and backup for manual recovery. Trim keeps its existing backup recovery behavior: on a trim failure it renames the backup to the original title, selects it, and deletes the original; if selection fails, the original remains intact. Recovery failure is aggregated with the root trim error, reports both session IDs, and does not skip terminal feedback.
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
- Native checkpoint marker/summary turns are never admitted as generation halos, so a retained halo cannot carry a `tail_start_id` into deleted history.
- Magic boundaries before the native checkpoint are ignored. The latest Magic boundary in the eligible suffix controls recompaction.
- `N` preserves the most recent completed assistant turns in the eligible suffix only.
- If the suffix has no eligible turns after applying `N`, no backup, generation fork, boundary, stats, or mutation is created.
- The generation global index spans every complete turn in this effective suffix, including the complete retained `N` tail and opaque Magic-summary turns, while excluding a trailing assistantless active turn.
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
- The `/magic-stats` notice distinguishes canonical model-facing estimates from provider context usage. Compact and trim notices report only the exact canonical before/after context counts, the trimmed amount and percentage, and the operation kind. Magic Compact never fabricates provider counts or mutates provider metadata.

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
- Generation errors expose only fixed lifecycle stage/kind text, source session ID, validation category, and range-validated HTTP status. Validation categories distinguish missing structured output, invalid envelope/turn count, invalid turn shape/identity, invalid summary, invalid visual coverage, and nonce leakage. Active-batch failures are ordered by batch in a sanitized aggregate. Raw provider/control/model errors, bodies, prompts, fork history, model output, and aggregate causes are not retained in the surfaced graph.
- Native context overflow reports trim-first guidance. A configured model input budget too small to fit the fixed 8K non-explicit host reserve, or the reserve plus the complete explicit request and mandatory global index, fails before any fork with trim-first/model-capacity guidance. Exact real-nonce prompt or response-protocol variance beyond its conservative record also fails before fork creation. Single-turn output exhaustion requires more model output capacity; larger-`N` guidance is limited to multi-turn exhaustion. When structured output is absent, truncation signaled by `finish` or an output-length error reports capacity guidance. A captured structured object still passes local validation when `finish` reports truncation. Other provider errors retain precedence, and a genuine non-`StructuredOutput` provider error always rejects the response. Magic Compact adds no plugin-level retry, adaptive repartitioning, alternate-model fallback, or partial mutation. OpenCode may apply provider/transport retries or additional model steps required by structured-output processing.
- Native checkpoint errors fail before generation. Ordered source snapshots are revalidated after backup creation and after owned summary/boundary insertion.
- `/magic-compact` fails closed with a pending-revert error before generation and rechecks immediately before the first source write. The pre-generation guard runs before any backup exists, leaving source, backup, and stats unchanged; the pre-write guard preserves the already-created backup and its copied backup stats while source and source stats remain unchanged. OpenCode's prompt-time revert cleanup would otherwise delete the reverted suffix on the boundary or stats `session.prompt`, which is not recoverable by Magic Compact's drift revalidation. Magic Compact does not force the cleanup: OpenCode exposes no session-serialized or compare-and-swap revert commit, so forcing cleanup could delete concurrently appended messages. Users must restore with `/redo` without a model call or let a normal OpenCode prompt commit the discard before retrying.
- Pre-write source drift and every failure after backup creation preserve both sessions; post-write errors report only the source and backup IDs plus manual-recovery guidance and never record a successful stats increment. Compaction never auto-applies the backup; trim retains its backup rollback path.
- In-progress, no-op, success, and failure toasts use the shared best-effort control-operation deadline helper. A late-settling timed-out progress request reasserts the latest terminal toast without delaying command settlement. Toast timeout or failure cannot change the primary command result, trigger recovery, replace the root error, or skip an already-required recovery attempt. A progress request applied remotely after its client promise rejected remains an undetectable host-side race.
- The command hook throws so OpenCode does not continue sending the slash command to the LLM.

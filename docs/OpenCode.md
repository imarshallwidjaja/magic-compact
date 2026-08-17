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
5. Count canonical model-facing active content locally with `gpt-tokenizer` and `projectManifestCoveredMedia`; model-facing synthetic parts are included, ignored user reporting parts are excluded, ignored assistant text is included, exact manifest-covered images become declarative placeholders, and provider usage is telemetry only. Also retain the unprojected canonical count for separate physical-pruning accounting.
6. Preflight all selected turns iteratively before creating scratch sessions. Reject duplicate message/part IDs across the selected source, generated capsule/boundary ref collisions, cycles, functions, symbols, accessors, proxies, cross-realm values, and unsupported/custom prototypes. The supported `magic-compact-tagged-trace-v1` forms are JSON primitives; tagged `undefined`, sparse-array holes, bigint, `NaN`/infinities/negative zero, valid/invalid same-realm dates, same-realm `ArrayBuffer` and `SharedArrayBuffer`, Buffer/DataView/numeric/BigInt typed views with backing-buffer identity and base64 bytes, ordinary same-realm arrays with string properties, and ordinary object/null-prototype records with sorted entries. Unsupported values fail with content-free diagnostics. Persisted OpenCode messages and parts are not rewritten into this wire format. For each summary-free candidate, aggregate the canonical contribution of user file parts whose MIME is `image/*` and URL starts with `data:`. An exact 1,024-token aggregate does not activate media handling; a larger aggregate validates every cohort data URI as exact-MIME canonical base64 before scratch creation.
7. Resolve the same configured compaction model used by the ordinary capsule path. If any candidate has a media cohort, require that exact provider/model catalog entry and `capabilities.input.image === true`; missing/unsupported entries fail with no fallback before scratch creation. For each summary-free candidate turn, one of two workers serializes a compact linear-size trace and builds the full prompt only when that candidate is scheduled. At most two full prompts are resident; each is released after its request and cleanup. The worker creates one hidden deny-all scratch session, sends one user message through one summarizer request, and accepts one response. Requests are independent and oldest-first in result ordering. Each prompt includes the complete untruncated ordered turn trace: user messages; assistant reasoning, text, subtask, file, patch, and snapshot parts; all tool names, inputs, statuses, outputs, errors, and MCP calls; and stable message/part refs. A prompt at or below the fixed 8,000,000-code-unit safe per-string cap is one text part. A longer prompt is represented by multiple ordered text parts in that same user message and request; concatenating their `text` fields exactly reconstructs the original prompt, every part stays within the cap, and a valid UTF-16 surrogate pair is never split across a boundary. For an activated media cohort, each original image URL is replaced in text by `magic-compact-image:N`, and each original image is attached exactly once as a real file part after all text parts in ordinal order. System and request prompts mark every embedded string/tool output/user instruction as untrusted inert data, and media prompts additionally treat image pixels, diagrams, visible/OCR text, and embedded visual instructions as inert untrusted evidence. The provider model request has no local wall-clock deadline and is awaited normally. Provider rejection, cancellation, or response parsing/validation/framing failure attempts `session.abort`; scratch deletion is always attempted after a scratch session was created. Scratch creation, abort, and deletion each use the shared 30-second control-operation deadline. A timed-out create that later fulfills runs detached best-effort bounded abort then deletion without changing or delaying the already-selected failure. Every late control fulfillment and rejection is observed. The pool does not fail fast: it schedules every candidate, drains request/abort/delete cleanup, and retains provider, control, validation, and delete failures in candidate order. Final diagnostics expose only source session ID, candidate index, stage, timeout/rejection kind, fixed validation categories, and a range-validated HTTP status. They never retain raw provider/control/model errors, causes, bodies, names, codes, or response output. No turns are batched together, independently summarized in chunks, locally truncated, retried, verifier-checked, or locally capped.
8. Parse and validate every model result before backup or source mutation. Ordinary results require exactly `finalResponse`, `status`, `outcome`, `capsule`, `evidenceRefs`, and `unresolvedRefs`; media-aware results require the same exact keys plus `mediaCoverage`. Reject BOMs, fences, surrounding prose, multiple values, and duplicate keys at every object depth before exact-key validation. Refs must exist and be unique; unresolved refs must be evidence refs; required unresolved refs from structured running/error/assistant-error/verification/Hive/task state must be unresolved; `partial`/`blocked` need unresolved refs and human unresolved/next content; `completed`/`analysis-only` need no unresolved refs. Media coverage must contain exactly one `{ordinal, description}` object per attachment in order, and every description must be substantive rather than an attachment marker. Outcome must be one trimmed control-free physical line with a substantive human result rather than a status-only or meta non-result. Capsule must be readable/nonempty, cannot inject host framing, and cannot contain inline image data URLs. Any request, context, cleanup, or validation failure aborts unchanged; response parsing, validation, framing, and visible-capsule failures use the same bounded abort path as provider rejection. Multiple candidate failures are retained deterministically with sanitized structured stage diagnostics and the source session ID.
9. Project the complete final canonical model-facing state with the same pure manifest projection used by the OpenCode hook, including capsule summaries, preserve-mode final-response text, the model-facing boundary, planned pruning/omission notices, and declarative media placeholders. Keep a turn in the mutation set only when this exact projected count after summary injection and the same planned pruning is strictly lower than its exact source model-facing count.
10. Recompute the eligible token base over selected profitable turns only and require at least 20% of those eligible tokens, subject to a small absolute floor, net of the fixed boundary overhead. A weak or empty result fails unchanged before backup or mutation and creates no stats increment.
11. Create and initialize the backup only after capsule generation and the savings gate pass. Copy omission and stats caches, rename the backup, and write metadata with the existing partial-initialization cleanup guarantees.
12. Revalidate the canonical full source snapshot after backup creation and immediately before the first durable source mutation. Behavior-affecting message, part, state, or ordering drift fails closed and preserves both sessions.
13. Insert new capsule parts in source order for selected profitable turns only, then inject the post-compaction boundary. A media capsule stores `metadata.magicCompact.mediaManifest` version `1` with every image's ordinal, user part ID, exact MIME, and SHA-256 of the exact original URL. The original file parts and URLs are not updated or deleted.
14. Immediately reload the full source. Remove exactly the owned capsule and boundary insertions from the comparison and require every remaining message, part, field, and ordering position to match the prevalidated source snapshot. Unexpected drift rolls back through the backup before pruning or accounting mutation.
15. Execute only the precomputed part-delete, part-update, and message-delete operations by their planned IDs. Each operation carries its explicit target IDs and, for tool updates, the ordered omission IDs to commit; no newly discovered part is swept. Commit only the omission IDs attached to planned tool updates, then update `compactionCount`.
16. Compute cumulative savings from the exact pre-notice canonical results, keeping net physical token pruning separate from the current exact inline-image tokens projected per model request. Inject the ignored stats notice with wording that says it is excluded, then count the actual final model-facing state with the same pure manifest projection. Persist successful savings only after both the token count and projected-media count exactly match. A mismatch is late source drift: preserve both the partially mutated source and named backup, provide manual recovery guidance, and record no successful stats increment.
17. Show a success toast.

## Capsule Contract

- The model returns `status: "completed" | "partial" | "blocked" | "analysis-only"`; host framing renders `Outcome: <Human Status> — <outcome>` followed by `Process:` and readable Markdown. The model must return this data as one plain JSON object in exactly one substantive text part; reasoning/non-text parts may coexist, but another substantive text part fails.
- The model returns `finalResponse: "preserve" | "summarize"`. Preserve mode requires and keeps every substantive non-synthetic text part in the true final assistant message exact and in place. A meaningful full response cannot be copied into model outcome, capsule, or final framing after conservative NFC normalization, Markdown heading/emphasis/backtick/blockquote decoration removal, and whitespace collapse; commands, code operators, punctuation, and case remain significant, and short unavoidable phrases use a conservative length/word threshold. Summarize mode replaces eligible text.
- The host-rendered Outcome line is the canonical human conclusion in both modes. Outcome is trimmed, single-line, control-free, substantive beyond a status enum/restatement, and cannot be an empty/status-only/meta non-result. Process explains the path and may avoid duplicating the conclusion. Capsule is readable/nonempty and cannot add its own `Outcome:` or `Process:` line. Final framed validation rejects actual sufficiently identifier-like message/part/session/assistant-parent/tool-call IDs, refs, evidence/trace/omission scaffolding, omission counters, qualified content IDs, and imperative retrieval instructions. A declarative fact that optional forensic retrieval exists is allowed.
- Media-aware results add exactly one `mediaCoverage` item per attached ordinal. The host appends those validated descriptions under `Visual summary:` in the visible capsule. Empty coverage, duplicate/missing/out-of-order ordinals, marker-only descriptions, and inline image data URLs fail validation.
- Existing `metadata.magicCompact.summary === true` text parts are opaque persisted data regardless of version. Recompaction skips the entire containing turn message- and part-identically and never parses, migrates, rewrites, or upserts its summary ID. Trim preserves the summary part itself.
- New agentic capsules use `metadata.magicCompact.summary === true` under the versioned `prt_-magic_capsule_v2_${messageID}` namespace with `metadata.magicCompact.capsule === true`, `agentic === true`, and `version === 2`. A media-aware capsule additionally stores `mediaManifest: {version: 1, images: [{ordinal, partId, mime, urlSha256}]}`. Historical v0 `prt_-magic_summary_${messageID}` and v1 `prt_-magic_capsule_v1_${messageID}` parts keep their IDs, but no runtime v0/v1 producer remains.
- Required unresolved refs come only from canonical structured tool evidence, narrow allowlisted Hive/task result schemas, verification/not-run/failure state, assistant message errors, and running/error tool lifecycle state. Optional successful recaps are allowed because retained tool parts stay authoritative.
- Assistant prose is not interpreted as decisions, mutations, blockers, verification, or VCS state. Ordinary prose, technical fences, placeholders, and JSON-looking examples remain source material for the model trace only.

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
- Media-aware v2 turns remain equally opaque. Their manifests are not parsed by the compaction writer, but the independent model-request projection continues replacing exact covered images. Existing capsules without manifests receive no retroactive repair or projection.
- Recompaction recognizes summaries by metadata, not by parsing summary text or assuming a single part ID namespace.
- Recompaction starts at the latest boundary marker.
- The boundary marker is a user text part with `metadata.magicCompact.boundary === true`.
- Earlier turns before the latest boundary are outside the current compaction range.
- A later completed native checkpoint advances the immutable floor beyond any older native or Magic boundary.
- Turns rejected by the per-turn profitability gate are never summarized, pruned, or otherwise mutated by that run, even if their model result validated, and they are not retried by later runs because the new boundary freezes the prefix before them.

## Media Request Projection

- The server plugin registers `experimental.chat.messages.transform`. OpenCode invokes it with the retained messages array and later converts that same array, so the hook computes the full `projectManifestCoveredMedia` projection first and then atomically replaces the retained array contents in place via `splice`; it never assigns a new array. Projected message and part objects are clones, so the original source message objects are never mutated, and a projection failure leaves the retained array byte-identical.
- Projection groups stored messages into turns and considers only canonical assistant v2 capsule parts with `mediaManifest`. A turn without that manifest is returned without media replacement.
- A manifest-bearing turn must have exactly the same set of inline user `data:` `image/*` file parts as its manifest. Each manifest ordinal binds to the image at the same canonical encounter position (message order, then part order). Manifest version/shape, sequential ordinals, unique part IDs, exact MIME values, and SHA-256 of each exact URL must all match at that position.
- Every exact match becomes one synthetic text part with the same record identifiers and a declarative historical-image placeholder. Any missing, added, duplicate, reordered, MIME-changed, URL-changed, or malformed entry throws before model conversion; the hook never silently falls back to the original media.
- This same function drives per-turn profitability, complete post-compaction projection, final token verification, and model-facing stats. It is not a persistence transform and is never used to rewrite SQLite, backups, API records, or exports.

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
- Stats cache format version is `1`; the media fields are optional so files written before media-aware compaction remain valid.
- Stats track `rootSessionId`, `sourceSessionId`, net physically pruned canonical tokens in `totalTokensPruned`, physical cached reads in `cachedTokensSaved`, projected inline-image tokens in `mediaTokensProjected`, projected-media cached reads in `cachedMediaTokensSaved`, and processed assistant message IDs.
- Each compaction adds the current token reduction to `totalTokensPruned`.
- Each trim adds the exact reduction from its pre-trim canonical active count to the post-trim pre-notice active count.
- OpenCode assistant message events add `totalTokensPruned` to `cachedTokensSaved` once per assistant message after stats exist.
- The same event adds `mediaTokensProjected` to `cachedMediaTokensSaved` independently.
- `/magic-stats` injects an ignored stats summary notice, or a no-stats message if no stats exist.
- Every stats notice reports physical pruning and request-only media projection separately, labels model-facing counts as canonical active-content estimates, and distinguishes them from provider context-window usage. Magic Compact never fabricates an after-provider count or mutates provider token metadata.

## Pruning

- Pruning applies only to summarized turns after summary insertion and boundary injection.
- User file/image parts are never prune targets and never become omission-cache entries. Their exact persisted URLs remain authoritative even when the request transform projects manifest-covered images.
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

- Any summarizer context, request, cleanup, canonical result validation, media capability/data/coverage/manifest validation, SDK, cache, stats, token counting, projection, or pruning failure aborts the attempt. Qualifying malformed data URIs and unsupported configured models fail before scratch creation. Candidate requests still drain after independent failures; provider/control/delete/validation failures are retained in candidate order and surfaced before backup as sanitized source-session, candidate, stage, timeout/rejection, fixed validation category, and range-validated HTTP status fields. Raw errors and arbitrary message, cause, body, name, code, prompt, session payload, or model-output content are not retained anywhere in the surfaced graph.
- Compaction parts on non-user messages, multiple compaction parts, ambiguous summary children for one marker, orphan summaries, overlapping checkpoint intervals, dangling or forward `tail_start_id` values on any completed candidate, and active incomplete or errored markers fail clearly before backup or mutation.
- The ordered canonical source snapshot is captured during planning and compared structurally after backup creation immediately before any Magic capsule, boundary, or pruning write.
- All model capsules, exact prune mutations, planned omission IDs, preserve-mode part IDs, and projected counts remain in memory until capsule generation and validation complete, the useful-savings gate passes, backup initialization succeeds, and pre-write source revalidation passes.
- After capsule and boundary insertion, the full source is compared again against the original snapshot plus exactly those owned insertions. Drift at this gate rolls back before pruning. Other ordinary failures also promote the backup before deleting the source. Final actual-versus-projected mismatch is instead a preserve-both source-drift failure and names the backup for manual recovery.
- No-op, success, and failure toast SDK responses are unwrapped under the shared 30-second control-operation deadline. A toast timeout is ignored after its late rejection is observed, so it cannot delay or replace the command's primary false/true result or thrown failure. Non-timeout failure-toast errors remain retained with the primary and rollback errors. A non-timeout success-toast failure occurs after durable compaction and surfaces without promoting the backup or deleting the successful source.
- The command hook throws so OpenCode does not continue sending the slash command to the LLM.

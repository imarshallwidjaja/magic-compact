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
5. Fork the session as a backup.
6. Copy omission and stats caches to the backup.
7. Rename the backup to `[Backup] ${title} ${timestamp}` and write backup metadata. If cache copy, stats copy, title, or metadata initialization fails after the fork, delete the target omission cache, target stats cache, and partial session. Retain every cleanup failure with the initialization error.
8. Measure pre-compaction tokens using provider tokens when available, otherwise local counting.
9. Revalidate native checkpoint state against the planning snapshot immediately before deterministic preparation and the progress notice.
10. Classify each source turn by deterministic serialized-byte, message, tool, and tool-state metrics. Build ordered generation entries by batching adjacent standard turns chronologically, at most eight per request, and placing each high-risk turn directly into the same source-ordered sequence. Flush a standard batch before and after each high-risk turn with the correct next source turn as its stop anchor.
11. Classify and render every complete in-memory high-risk source turn before the first source write. Reuse those prepared summaries later in source order. High-risk preparation creates no primary prompt, repair prompt, ephemeral fork, model request, omission mapping, or ephemeral checkpoint check; required-evidence overflow aborts here and follows ordinary backup rollback.
12. Insert an ignored no-reply progress message.
13. Before the first standard request only, resolve OpenCode's native hidden `compaction` agent model and variant, then reuse those settings for every later standard batch. A plan containing only high-risk entries does not resolve model settings.
14. For each standard batch, fork a fresh ephemeral session from the unchanged source, immediately validate and capture that fork's native checkpoint structure using its remapped local message, part, session, parent, and tail IDs, install the scoped omission mapping, and send the ID-keyed XML summary prompt through Magic Compact's hidden summarizer agent. Requests execute sequentially.
15. After every standard prompt, compare the fork against its local checkpoint snapshot. A new, changed, malformed, or incomplete fork-local artifact raises `EphemeralCheckpointChangedError` and aborts before source writes. Delete every ephemeral session in cleanup; prompt/body, checkpoint, and deletion failures are retained together.
16. After all standard entries finish and the prepared deterministic summaries are assembled in source order, reload source native checkpoint state and compare it with the planning snapshot before writing any Magic summaries, boundary, or pruning changes. A fetch, unwrap, or inspection failure leaves checkpoint state unknown and is treated as a source checkpoint change. Only a source race or unknown source revalidation keeps `NativeCheckpointChangedError` outermost and preserves both sessions. Confirmed unchanged source state leaves ephemeral checkpoint, generation, and cleanup failures on the ordinary backup rollback path.
17. Upsert deterministic summary text parts onto the first assistant message in each summarized turn.
18. Delete the progress message in cleanup.
19. Inject the post-compaction boundary notice.
20. Reload the planned summarized turns by ID.
21. Preflight the complete prune selection, then prune summarized turns.
22. Update current session metadata with `compactionCount`.
23. Measure post-compaction tokens.
24. Update stats and inject an ignored stats notice.
25. Show a success toast.

## Trim Flow (Experimental)

1. Parse `N`; default is `0`.
2. Build turns for the complete stored session, independent of compaction boundaries.
3. Preserve the `N` most recent assistant turns.
4. Load the source session and fork it as a backup.
5. Copy omission and stats caches to the backup.
6. Measure pre-trim tokens using provider tokens when available, otherwise local counting.
7. Apply the normal tool input and output trimming rules to older turns.
8. Mark processed completed tool states with `state.metadata.magicCompact.trimmed === true`.
9. Stop with a no-op toast if no tool states were processed.
10. Measure post-trim tokens and add the reduction to conversation stats.
11. Inject an ignored trim stats notice and show a success toast.

`/magic-trim` does not call an LLM, generate summaries, modify ordinary user or assistant content, insert a compaction boundary, or increment `compactionCount`.

A trim no-op is detected after backup creation and leaves that backup in place.

## Backup Sessions

- Backup title: `[Backup] ${title} ${timestamp}`.
- The main session title stays unchanged on success.
- Backup metadata stores `sourceSessionId`, `compactedAt`, and `compactionCount`.
- The backup receives copies of omission and stats caches before mutation.
- A fork is not considered a backup until cache, stats, title, and metadata initialization all succeed. Initialization failure deletes the target omission cache, target stats cache, and partial session; the primary error and every cleanup failure are aggregated.
- Trim backups preserve the source session's current `compactionCount`.
- For ordinary compaction or trimming failures after backup creation, the backup is renamed back to the original title and selected before the original session is deleted. If backup selection fails, the original session remains intact.
- During `/magic-compact`, if source deletion fails after backup selection, both sessions remain and the selected backup stays active. Backup rename, selection, source deletion, and failure-toast errors are retained as secondary failures without replacing the primary error.
- A native checkpoint change detected by the pre-write revalidation is not an ordinary rollback case: the live source and pre-race backup are both preserved, and neither session is selected or deleted.
- An ephemeral fork-local checkpoint change is distinct from a source race. After source revalidation confirms the source is unchanged, the existing backup is selected before the temporary source session, including its progress notice, is deleted.

## Turn Selection

- Messages are processed oldest-first.
- A turn is one or more adjacent user messages plus all following assistant messages before the next user group.
- Consecutive user/no-reply messages stay in the same turn.
- Boundary detection runs before ignoring a trailing assistantless turn.
- A trailing user-only turn does not count against `N`.
- Only turns with assistant messages are summarized.
- A valid completed native checkpoint is a user message with exactly one `compaction` part and exactly one native summary assistant child whose `parentID` is the marker ID, `summary` is `true`, `finish` is truthy, and `error` is absent.
- Every marker is checked against all of its native summary children. A completed child plus any additional completed, errored, or unfinished summary child is ambiguous. The only exception is an older marker whose complete ambiguous or incomplete artifact set lies wholly before the marker of a later valid checkpoint and is therefore already inside that checkpoint's frozen prefix.
- Completed checkpoint intervals must be sequential. An interval may not contain another marker or a foreign native summary; nested and crossed intervals are rejected as ambiguous.
- The newest valid completed native checkpoint owns an immutable prefix through its summary assistant. Turn construction starts strictly after that summary.
- Every completed checkpoint candidate's `tail_start_id`, when present, must resolve to a durable message strictly before its marker. The retained tail, marker, summary message, summary text, and summary parts remain untouched.
- Magic boundaries before the native checkpoint are ignored. The latest Magic boundary in the eligible suffix still controls recompaction.
- `N` preserves the most recent completed assistant turns in the eligible suffix only.
- If the suffix has no eligible turns after applying `N`, the command is the existing no-op: no backup, progress notice, summary request, boundary, stats, or mutation is created.
- For `/magic-trim`, `N` preserves tool I/O in the most recent assistant turns across the complete session.
- Trim selection does not use compaction boundaries.

## Recompaction

- Previously summarized turns are preserved as-is.
- Recompaction starts at the latest boundary marker.
- The boundary marker is a user text part with `metadata.magicCompact.boundary === true`.
- Earlier turns before the latest boundary are outside the current compaction range.
- A later completed native checkpoint advances the immutable floor beyond any older native or Magic boundary.

## Summarization

- Standard summaries are generated in an ephemeral session so the prompt and assistant stream stay out of the main session. High-risk summaries are generated deterministically in memory without a model request or fork.
- Each standard ephemeral session is a fork of the source session: the summarizer needs the full conversation in context to summarize assistant turns faithfully.
- Magic Compact registers a hidden `magic-compact-summarizer` subagent with a dedicated faithful per-turn XML summarization prompt.
- The summarizer treats historical transcript evidence as primary. It may inspect current state when needed, must distinguish current observations from historical claims, and must never continue unfinished work.
- Before the first standard prompt, Magic Compact resolves OpenCode's native hidden `compaction` agent through the public v2 app agents API for the source workspace. An all-high-risk plan does not query the agent API.
- The request explicitly uses the Magic Compact summarizer agent while inheriting the native compaction agent's configured model and variant. If the native compaction agent has no model, the request falls back to the captured source session model and variant.
- A missing or disabled native hidden compaction agent, or an invalid explicit model configuration, aborts compaction rather than selecting another agent.
- Source session permissions are not copied to the ephemeral session; the custom agent policy remains authoritative.
- The custom agent allows registered built-in, plugin, MCP, and unknown future tools by default, including shell and bash. It denies OpenCode's `edit` permission, which covers edit, write, and apply-patch operations, and also denies task/subagent, question/interactive, todowrite, plan-enter, plan-exit, and doom-loop interactions.
- Tools used by a standard summarizer request can have side effects. Deleting the temporary session removes its conversation record but does not roll back tool effects on files, processes, services, or external systems. High-risk turns cannot create these model-driven side effects because they make no request.
- The standard XML prompt is built from the OpenCode template.
- The standard XML prompt includes only the turns being summarized and, when needed, the next user turn as the boundary marker.
- User text in the prompt excludes synthetic and ignored text and is truncated to the first line or first 300 Unicode code points, whichever is shorter. XML-invalid code points, including lone surrogates and disallowed controls, are removed before escaping.
- User previews and evidence are XML-escaped.
- Every turn is keyed by the first assistant message ID. The generated response must be one strict `<summary>` XML document containing only ignorable XML whitespace and exact `<assistant id="...">text</assistant>` children. Text or elements outside the root, unexpected or nested elements, extra attributes, comments, CDATA, processing instructions, malformed entities, and unclosed elements are rejected. Standard XML entities are decoded into installed summary text. Response ordering may differ; output is restored to plan order.
- Missing, duplicate, unknown, malformed, empty, whole-body recognized placeholders/markers, bracket bodies that contain explicit replace/placeholder/TODO/TBD template wording, generic-acknowledgement, or one-word marker-only summaries are rejected. Arbitrary brackets such as `arr[0]`, Markdown links, JSON arrays, `app/[slug]/page.tsx`, and non-template bracket prose stay. There is no positional fallback.
- A standard batch contains at most eight turns. A high-risk turn is any turn with at least 153,600 UTF-8 serialized bytes, 20 messages, 32 tool parts, or one non-completed/error tool state.
- Every high-risk turn receives no primary or repair model request. Its summary contains only deterministic records derived from that turn's in-memory source messages and parts.
- Magic Compact classifies complete tool input, output, and error values and complete real assistant text before deriving any bounded record. Only then does it create UTF-8-safe exact windows and anchors from cue spans, so a middle failure such as `Process exited with code 1` survives long surrounding output. Tool input, output, and error keep exact JavaScript code units, including NUL, XML-invalid controls, lone surrogates, and U+FFFE/U+FFFF. One deterministic record serializer is used for both rendering and byte-budget calculations: it starts from `JSON.stringify` exact JS strings, then reversibly escapes any remaining XML-invalid code points that JSON left literal (including U+FFFE/U+FFFF) as lowercase `\uXXXX` sequences. Existing NUL/controls/lone surrogates remain JSON-escaped. The rendered summary stays free of raw invalid XML/provider characters while remaining reversible via `JSON.parse`.
- Source classification tracks verification and VCS/deployment independently. Structured tool state and assistant prose use separate classifiers. Tool verification context is established only by actual tool identity that explicitly represents execution/test runners or by a structured command that clearly identifies tests, checks, lint, typecheck, build, smoke, or known runners; tool VCS/deployment context is established only by actual tool identity that explicitly represents git/VCS/deploy or by a structured command that clearly identifies git/VCS/deployment. Tool identities are normalized before matching by splitting snake_case, kebab-case, dots, slashes, colons, and other namespace separators, so names such as `run_tests`, `git_status`, `plugin.run-tests`, and namespaced equivalents establish verification or VCS context. Normalized identities that begin with or contain explicit passive semantics (`read`, `get`, `list`, `search`, `view`, `inspect`, `fetch`) remain neutral even when the rest of the name includes tests/build/git/deploy tokens (`plugin.read-tests`, `read_build`, `list-git-status`); identity-based verification/VCS matching is skipped for those names, while a structured command field still establishes context from command text. Tool titles and passive path labels such as `Read unit.test.ts`, `Read build configuration`, or `Read deployment config` never establish verification or VCS context. Structured verification commands recognize package runners (`bun`/`npm`/`pnpm`/`yarn` test|check|lint|typecheck|build|smoke), language runners (`pytest`, `vitest`, `jest`, `mocha`, `cargo test`, `go test`, `tsc --noEmit`), and explicit task runners (`make`/`just`/`task` test|check|lint|typecheck|build|smoke) without matching arbitrary prose. Only inside those contexts does tool result text contribute success, failure, or not-run polarity. Any tool state of error, running, pending, or other incomplete status remains unresolved regardless of output text. Completed arbitrary read, task, or custom tool output such as `No unresolved issues`, `The value was passed to serializer`, `configuration remains unchanged`, `pending color`, or `fail\uD800ed` is neutral and cannot create required verification or unresolved evidence. Independently, any actual command-running tool identity (`bash`, `shell`, `terminal`, `exec`, and normalized namespaced variants) whose completed state output or error shows strong execution failure or noncompletion (nonzero exit, permission denied, command not found, timeout, cancelled/aborted/killed/signal) is required unresolved/failure evidence even when the command is not verification or VCS; passive read/task/custom completed outputs remain neutral. Mutation tools are categorized by tool identity; assistant mutation prose remains separate and still honors negated forms. Assistant prose requires explicit test, check, command, work, VCS, or deployment context. Complete assistant message text is classified before any segmentation so cross-sentence evidence such as `Tests ran. They failed.` or `Tests could not run. Database unavailable.` remains unresolved or partial; segmented windows are used only for bounded rendering after that full-text classification, and a separate positive run/pass statement still overrides earlier no-run wording. Resolved negations such as `No unresolved issues`, `Nothing remains to do`, `no pending work`, `no tasks are pending`, `no work is pending`, `no issues remain pending`, analogous work/task/issue/test/check forms, `implementation is not incomplete`, `No files were modified`, and `configuration remains unchanged` take precedence over incidental words, while real incomplete forms such as `tasks are pending` and `work remains: run smoke` stay unresolved. Neutral VCS evidence remains a VCS source record and is not converted into not-run or failure. The precedence-aware classifier recognizes pytest, Vitest, TypeScript, and the other supported runners. `All tests passed; no tests failed` and `10 passed, 0 failed` are successful, while `No tests ran; 0 tests failed`, `No tests failed because tests were not run`, `Tests couldn't/could not/cannot/can't be run or performed`, `tests were skipped`, and `no tests have run` are not-run unless a separate positive run/pass statement exists. `not pushed`, `not committed`, `not deployed`, and `push was not run` are not-run rather than successful. Positive failures, nonzero exits, permission denial, missing commands, timeouts, cancellation, aborts, kills, and signals are failed and unresolved when in verification or VCS context, when tool state is error, or when a command-running tool identity shows those strong execution failures in state output/error. Substantive assistant `TODO`/`TBD` markers are incomplete/unresolved. Bare nouns such as `passed`, `remains`, `modified`, `build`, `check`, push notification, deployment configuration, git client parser, or branch are not execution evidence by themselves.
- A high-risk summary renders `Outcome:`, `Historical assistant text (authoritative):`, `Verification:`, `VCS/Deployment:`, `Unresolved:`, and exactly one `Historical terminal evidence (authoritative):` block. Outcome is `partial` when any source record evidences failure, not-run, incomplete, blocked, pending, running, or unresolved work; it is `analysis-only` when there is no execution, mutation, verification, or VCS/deployment evidence; otherwise it is `completed`. No deterministic rule currently emits `blocked`, so uncertain negative state remains `partial`.
- Historical assistant text contains only real assistant text-part records. It excludes synthetic or ignored text, existing Magic summaries, empty text, and whole-body placeholders only (`[TODO]`, `[TBD]`, `[Replace...]`, plain `TODO`/`TBD`/`unknown`/`N/A`, and other marker-only bodies). Bracketed JSON arrays and substantive prose that merely mentions TODO/TBD stay. Assistant source text keeps exact JavaScript code units, including NUL, XML-invalid controls, lone surrogates, and U+FFFE/U+FFFF; the same deterministic record serializer escapes them (for example `\u0000`/`\ud800`/`\ufffe`) so the rendered summary stays free of raw invalid XML/provider characters while remaining reversible. The final substantive assistant response is required; earlier real assistant text is included by priority and source order while budget remains.
- The complete deterministic high-risk summary has one 64 KiB UTF-8 byte cap over the fully rendered summary, including JSON escapes. Record-aware allocation atomically reserves every final substantive response record and every negative, verification, and VCS/deployment minimum record before optional earlier assistant text and terminal records. Each required minimum is one complete JSON line retaining explicit categories and polarity, status and bounded source state, the exact bounded command on its dedicated field, exact paths on the dedicated paths field (including paths found in tool titles), and evidence anchors that carry status cues without duplicating command or path values; allocation never deletes fields and counts the damaged record as included. If all required minima do not fit together, generation throws `HighRiskEvidenceOverflowError` before source summary, boundary, or prune writes and follows ordinary backup rollback. Optional records may be excluded, and every rendered section reports the exact number of records excluded from that section.
- Ordinary low-risk summaries keep the ID-keyed XML protocol and instructions to preserve outcome, delivered changes, verification and checks not run, VCS/deployment state, errors, unresolved work, next action, and exact evidenced identifiers. Completeness takes priority over brevity.
- Each summary is written as a text part on the first assistant message in the summarized turn.
- Summary parts use deterministic IDs: `prt_-magic_summary_${messageID}`.
- Summary parts are marked with `metadata.magicCompact.summary === true`.

## Boundary Notice

- OpenCode injects a synthetic user text part after summaries are written.
- If a next user message exists, the notice is written onto that message and the part ID sorts before normal parts.
- If no next user message exists, a no-reply synthetic user message is created.
- The notice is marked with `metadata.magicCompact.boundary === true`.
- The notice tells the model to use `read_omitted_content` only when exact omitted historical tool I/O is needed and cannot be recovered through a fresh tool call.

## Omission Cache

- Location: `${XDG_DATA_HOME:-~/.local/share}/opencode/storage/magic-compact/{sessionId}.json`.
- Cache format version is `2`.
- Newly allocated IDs are `<last-12-session-chars>:omitted-<22-char-base64url-token>`, where the token contains 128 random bits. The prefix is diagnostic only and is never used to route a lookup.
- Each v2 entry stores content and full SHA-256 over the fixed `magic-compact:omission-entry:v2\0` domain/version prefix, a four-byte big-endian length and the exact UTF-8 qualified Content ID bytes, then the exact JavaScript content from `Buffer.from(content, "utf16le")`. This binds an entry to both its ID and ordered UTF-16LE code units, preserves lone surrogates, and performs no Unicode normalization. Backup copies retain IDs, so unchanged entries remain valid. Every cache read and atomic write validates every v2 entry's strict qualified ID and digest before returning or replacing bytes. Omission notices still report the omitted content's UTF-8 byte length, but that display length is not persisted in v2 entries.
- Allocation checks for a generated collision and retries without overwriting any existing entry. There is no sequential counter.
- Valid v1 bare entries are preserved under an explicit `legacy.entries` section during migration and backup copying, but no new bare ID is allocated.
- V1 bare entries have no cryptographic binding between an ID and its historical bytes. Every bare `omitted-NNN` request therefore fails with an explicit unverifiable-legacy-unavailable diagnostic, whether the stored entry exists or not. Bare legacy bytes are never returned.
- An absent source cache materializes as an empty v2 cache when copied to a backup. Malformed, schema-invalid, or integrity-invalid caches fail loudly and are not reset or overwritten.
- Cache writes use atomic temporary-file replacement. In-process mutations for one session are serialized; distinct sessions remain independent. There is no global lookup, garbage collector, per-entry file layout, cross-process lock, or lineage traversal.
- The current session cache is the active cache on success.
- The backup gets a cache copy before mutation.
- During an ephemeral summary prompt, omission reads for the temporary session are routed to the source session cache. The mapping is installed before the prompt and cleared on success or failure.

## Omission Retrieval

- The plugin exposes `read_omitted_content` as an OpenCode plugin tool.
- The tool accepts one argument: `contentId`.
- The tool receives `context.sessionID` from OpenCode and reads that session's omission cache, except for the temporary source mapping active only during summarization.
- Retrieval never routes by the diagnostic ID prefix or opens another session cache.
- If no authorized matching qualified entry exists, integrity proof fails, the cache is invalid, or a bare legacy ID is requested, the tool returns an explicit unavailable/integrity diagnostic and never guessed bytes. Non-integrity cache failures use a generic diagnostic that does not expose local filesystem or session paths; typed integrity failures retain path-free validation detail.

## Stats

- Stats are stored under `${XDG_DATA_HOME:-~/.local/share}/opencode/storage/magic-compact/stats/{sessionId}.json`.
- Stats cache format version is `1`.
- Stats track `rootSessionId`, `sourceSessionId`, `totalTokensPruned`, `cachedTokensSaved`, and processed assistant message IDs.
- Each compaction adds the current token reduction to `totalTokensPruned`.
- Each trim adds its locally counted token reduction to `totalTokensPruned`.
- OpenCode assistant message events add `totalTokensPruned` to `cachedTokensSaved` once per assistant message after stats exist.
- `/magic-stats` injects an ignored stats summary notice, or a no-stats message if no stats exist.

## Pruning

- Pruning applies only to summarized turns after summary insertion and boundary injection.
- Before the first prune mutation, the full selection is rejected if it contains a native `compaction` part or native summary assistant.
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

- Any standard-summary LLM, XML, completeness, ephemeral checkpoint, deterministic rendering, SDK, cache, stats, token counting, or pruning failure aborts the attempt.
- Compaction parts on non-user messages, multiple compaction parts, ambiguous summary children for one marker, orphan summaries, overlapping checkpoint intervals, dangling or forward `tail_start_id` values on any completed candidate, and active incomplete or errored markers fail clearly before backup or mutation. An older ambiguous or incomplete marker is ignored only when all of its summary children are wholly before the marker of a later valid completed checkpoint.
- The ordered source native checkpoint artifact snapshot is deep-cloned during planning and compared structurally immediately before the progress notice and after all ephemeral summary generation, still before any Magic summary, boundary, or pruning writes. Each ephemeral fork separately captures a validated fork-local snapshot immediately after creation and compares against that local structure after its prompt; fork-remapped IDs are never compared directly with source IDs. A new, changed, malformed, or incomplete fork-local checkpoint raises `EphemeralCheckpointChangedError`. Source revalidation fetch, unwrap, and inspection failures raise or retain `NativeCheckpointChangedError` because unchanged source state was not confirmed.
- Cleanup deletes the ephemeral session and progress message when they exist. `NativeCheckpointChangedError` remains the outer error only for source preserve-both failures, with structural or fetch causes and generation, ephemeral-cleanup, progress-cleanup, and notification failures retained in an aggregate cause. `EphemeralCheckpointChangedError`, prompt, cleanup, rollback rename/select/delete, and notification failures are retained normally and remain ordinary rollback cases once source revalidation confirms unchanged source state. Cleanup-only failure still surfaces and follows ordinary rollback.
- All summaries remain in memory until every standard batch and deterministic high-risk entry completes and source checkpoint revalidation passes. Except for the temporary ignored progress notice, no source summary, boundary, prune, omission-cache, stats, or metadata write occurs before that barrier.
- Other failures attempt to promote an existing backup by renaming and selecting it before deleting the source.
- No-op, success, and failure toast SDK responses are unwrapped. Failure-toast errors are retained with the primary and rollback errors. A success-toast failure occurs after durable compaction and surfaces without promoting the backup or deleting the successful source.
- The command hook throws so OpenCode does not continue sending the slash command to the LLM.

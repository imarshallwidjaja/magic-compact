# OpenCode Behavior Specification

OpenCode-specific runtime behavior. Shared plugin behavior lives in [`Core.md`](Core.md).

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
7. Rename the backup to `[Backup] ${title} ${timestamp}` and write backup metadata.
8. Measure pre-compaction tokens using provider tokens when available, otherwise local counting.
9. Revalidate native checkpoint state against the planning snapshot immediately before creating the progress notice.
10. Insert an ignored no-reply progress message.
11. Fork the source session into an ephemeral compaction session so the summarizer sees the full conversation.
12. Resolve OpenCode's native hidden `compaction` agent for the source workspace, then send the XML summary prompt through Magic Compact's hidden summarizer agent in the ephemeral session using the resolved compaction model and variant.
13. Parse per-turn summaries.
14. Delete the ephemeral session in cleanup. Prompt/body failure and ephemeral deletion failure are captured independently: both failures throw an `AggregateError` retaining each cause; a single failure throws that error.
15. After every ephemeral summary-generation attempt (success or failure of prompt generation or ephemeral deletion), reload native checkpoint state and compare it with the planning snapshot before writing any Magic summaries, boundary, or pruning changes. A fetch, unwrap, or inspection failure leaves checkpoint state unknown and is treated as a checkpoint change. If generation or cleanup also failed, keep `NativeCheckpointChangedError` outermost and retain every failure in its aggregate cause; only confirmed unchanged state permits ordinary backup rollback.
16. Upsert deterministic summary text parts onto the first assistant message in each summarized turn.
17. Delete the progress message in cleanup.
18. Inject the post-compaction boundary notice.
19. Reload the planned summarized turns by ID.
20. Preflight the complete prune selection, then prune summarized turns.
21. Update current session metadata with `compactionCount`.
22. Measure post-compaction tokens.
23. Update stats and inject an ignored stats notice.
24. Show a success toast.

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
- Trim backups preserve the source session's current `compactionCount`.
- For ordinary compaction or trimming failures after backup creation, the backup is renamed back to the original title and selected before the original session is deleted. If backup selection fails, the original session remains intact.
- A native checkpoint change detected by the pre-write revalidation is not an ordinary rollback case: the live source and pre-race backup are both preserved, and neither session is selected or deleted.

## Turn Selection

- Messages are processed oldest-first.
- A turn is one or more adjacent user messages plus all following assistant messages before the next user group.
- Consecutive user/no-reply messages stay in the same turn.
- Boundary detection runs before ignoring a trailing assistantless turn.
- A trailing user-only turn does not count against `N`.
- Only turns with assistant messages are summarized.
- A valid completed native checkpoint is a user message with exactly one `compaction` part followed by one successful assistant child whose `parentID` is the marker ID, `summary` is `true`, `finish` is truthy, and `error` is absent.
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

- Summaries are generated in an ephemeral session so the prompt and assistant stream stay out of the main session.
- The ephemeral session is a fork of the source session: the summarizer needs the full conversation in context to summarize assistant turns faithfully.
- Magic Compact registers a hidden `magic-compact-summarizer` subagent with a dedicated faithful per-turn XML summarization prompt.
- The summarizer treats historical transcript evidence as primary. It may inspect current state when needed, must distinguish current observations from historical claims, and must never continue unfinished work.
- Before prompting, Magic Compact resolves OpenCode's native hidden `compaction` agent through the public v2 app agents API for the source workspace.
- The request explicitly uses the Magic Compact summarizer agent while inheriting the native compaction agent's configured model and variant. If the native compaction agent has no model, the request falls back to the captured source session model and variant.
- A missing or disabled native hidden compaction agent, or an invalid explicit model configuration, aborts compaction rather than selecting another agent.
- Source session permissions are not copied to the ephemeral session; the custom agent policy remains authoritative.
- The custom agent allows registered built-in, plugin, MCP, and unknown future tools by default, including shell and bash. It denies OpenCode's `edit` permission, which covers edit, write, and apply-patch operations, and also denies task/subagent, question/interactive, todowrite, plan-enter, plan-exit, and doom-loop interactions.
- Allowed tools can have side effects. Deleting the temporary session removes its conversation record but does not roll back tool effects on files, processes, services, or external systems.
- The XML prompt is built from the OpenCode template.
- The XML prompt includes only the turns being summarized and, when needed, the next user turn as the boundary marker.
- User text in the prompt excludes synthetic and ignored text and is truncated to the first line or first 300 characters, whichever is shorter.
- The generated XML must contain one `<assistant>` summary for each summarized turn.
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
- Cache format version is `1`.
- IDs are session-local sequential IDs: `omitted-001`, `omitted-002`, ...
- The current session cache is the active cache on success.
- The backup gets a cache copy before mutation.
- Compaction and trimming share the same session-local omission sequence.
- During an ephemeral summary prompt, omission reads for the temporary session are routed to the source session cache. The mapping is installed before the prompt and cleared on success or failure.

## Omission Retrieval

- The plugin exposes `read_omitted_content` as an OpenCode plugin tool.
- The tool accepts one argument: `contentId`.
- The tool receives `context.sessionID` from OpenCode and reads that session's omission cache, except for the temporary source mapping active only during summarization.
- If no matching cache entry exists, it returns a not-found message.

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

- Any LLM, XML, SDK, cache, stats, token counting, or pruning failure aborts the attempt.
- Compaction parts on non-user messages, multiple compaction parts, multiple successful summaries for one marker, orphan summaries, overlapping checkpoint intervals, dangling or forward `tail_start_id` values on any completed candidate, and active incomplete or errored markers fail clearly before backup or mutation. An older incomplete marker is ignored only when it is wholly before the marker of a later valid completed checkpoint.
- The ordered native checkpoint artifact snapshot is deep-cloned during planning and compared structurally immediately before the progress notice and again after every ephemeral summary-generation attempt, including when prompt generation or ephemeral session deletion fails, and still before any Magic summary, boundary, or pruning writes. A new, changed, malformed, or incomplete checkpoint aborts those writes and does not replace the live source with the pre-race backup. Revalidation fetch, unwrap, and inspection failures receive the same preserve-both treatment because unchanged state was not confirmed.
- Cleanup deletes the ephemeral session and progress message when they exist. `NativeCheckpointChangedError` remains the outer error for preserve-both failures, with structural or fetch causes and generation, ephemeral-cleanup, and progress-cleanup failures retained in an aggregate cause. Non-race operation and cleanup failures are also retained together but remain ordinary rollback cases. Cleanup-only failure still surfaces and follows ordinary rollback.
- Other failures promote an existing backup by renaming and selecting it before deleting the source.
- A failure toast is shown.
- The command hook throws so OpenCode does not continue sending the slash command to the LLM.

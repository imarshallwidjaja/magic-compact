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
6. Classify turns semantically. Failed, not-run, running, error, unknown, incomplete, or unresolved tool, assistant-message, or prose evidence is deterministic-only. A later successful assistant message marks an earlier assistant-message error recovered for routing while retaining bounded status evidence. Completed resolved turns remain standard regardless of byte, message, or tool counts.
7. Render deterministic-only summaries in memory as model-free, human-readable continuation capsules with an `Outcome` line plus `Work`, `Verification`, `VCS`, `Unresolved`, and `Next` sections as evidence requires. Treat every substantive text part in the final assistant message as one ordered final response and keep it in full when it fits the remaining capsule budget. Otherwise preserve every classified decision, verification, VCS, unresolved, and next-action fact from all of those parts through source-anchored readable excerpts. Required failed, not-run, running, error, unknown, incomplete, credentials-missing, could-not-finish, blocked, and unresolved evidence stays visible with exact actionable commands and paths. Only normalized-equivalent repeated facts aggregate with truthful counts; distinct required payloads remain visible or rendering fails closed. Visible capsules contain no standalone raw evidence records, source windows, or record accounting. The 2,048-token per-turn cap and 512-token optional allowance are safety ceilings, not output targets.
8. Build bounded exact evidence maps for standard turns. Every substantive text part in the final assistant message is required exact evidence in source order. Token-pack requests at 24K local input tokens. If required exact final-message evidence cannot fit that bound, fail closed before scratch, backup, or mutation. No full transcript is included.
9. Resolve the native hidden compaction model once. Create scratch sessions with no `parentID`, source directory/workspace binding, and deny-all permissions. Run at most two requests concurrently, best-effort clamp summarizer provider generation to at most 4,000 output tokens through `chat.params` only when OpenCode supplies a compatible active limit, independently accept at most 4K local response tokens, preserve source-order assembly, and delete every scratch session on every exit. A timeout or client abort awaits routed `session.abort` before deletion.
10. Keep the exact-key bare JSON transport contract. Validate each raw model value before folding as a non-empty substantive narrative body, reject placeholders, recursively detect the finite producer evidence schema in standalone or embedded JSON and anchored assignments, and reject C0/C1 controls not consumed as approved boundaries, bidi override/isolate controls, and unpaired UTF-16 surrogates. Preserve valid surrogate pairs, ordinary Unicode, and technical prose such as `Documented the JSON field "kind": "tool"` or `Set sourceBytes = 128`. Classify each accepted standard turn in host code; fold approved line separators, neutralize raw `&` before raw `<`, construct exactly `Outcome: <host outcome>\nWork: <single-line narrative>`, and assert that two-line invariant locally. Deterministic `partial` turns keep the model-free renderer. Retry once in a fresh scratch session only for a typed content-free summary-contract failure after successful cleanup. Provider, timeout, cancellation, SDK, and cleanup failures are not retried.
11. Project the complete final canonical active state, including summaries, deterministic capsules, the model-facing boundary, and planned pruning/omission notices. Ignored user reporting notices are excluded; ignored assistant text remains in the active count. Keep a turn in the mutation set only when its exact projected canonical active token count after summary injection and the same planned pruning is strictly lower than its exact source canonical active token count; rejected turns remain message- and part-identical. Recompute the eligible token base over the selected profitable turns only and require at least 20% of those eligible tokens, subject to a small absolute floor, net of the fixed boundary overhead. A weak or empty result fails unchanged before backup or mutation and creates no stats increment.
12. Create and initialize the backup only after generation, validation, and the savings gate pass. Copy omission and stats caches, rename the backup, and write metadata with the existing partial-initialization cleanup guarantees.
13. Revalidate the canonical full source snapshot after backup creation and immediately before the first durable source mutation. Behavior-affecting message, part, state, or ordering drift fails closed and preserves both sessions.
14. Upsert summaries in source order for the selected profitable turns only, inject the post-compaction boundary, execute the preplanned prune/omission selection over those turns, and update `compactionCount`.
15. Compute cumulative savings from the exact pre-notice canonical active result, inject the ignored stats notice with wording that says it is excluded, then count the actual final active state on the same local basis. Persist successful savings only after the count exactly matches. A mismatch is late source drift: preserve both the partially mutated source and named backup, provide manual recovery guidance, and record no successful stats increment. Savings are never clamped to zero.
16. Show a success toast.

## Trim Flow (Experimental)

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
11. Inject a notice that reports the exact pre-notice active result and explicitly says the ignored notice is excluded, then require the actual final active count to match before showing a success toast. No self-referential convergence loop is used.

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
- Source drift detected by the pre-write revalidation is not an ordinary rollback case: the live source and pre-race backup are both preserved, and neither session is selected or deleted.
- Scratch sessions inherit no transcript or checkpoint artifacts. Source drift found by the final pre-write revalidation preserves both sessions and performs no source writes.
- Source drift detected after writes by the final actual-versus-projected canonical active-token mismatch also preserves both sessions without selecting, promoting, or deleting either. The source may be partially compacted and must be compared with the backup ID named in the error. Same-token late mutation remains outside this mismatch detector because there is no source lock or journal.

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
- If the suffix has no eligible turns after applying `N`, the command is the existing no-op: no backup, summary request, boundary, stats, or mutation is created.
- For `/magic-trim`, `N` preserves tool I/O in the most recent assistant turns across the complete session.
- Trim selection does not use compaction boundaries.

## Recompaction

- Previously summarized turns are preserved as-is, including unversioned v0 `metadata.magicCompact.summary === true` parts created by earlier releases.
- Recompaction starts at the latest boundary marker.
- The boundary marker is a user text part with `metadata.magicCompact.boundary === true`.
- Earlier turns before the latest boundary are outside the current compaction range.
- A later completed native checkpoint advances the immutable floor beyond any older native or Magic boundary.
- Turns rejected by the per-turn profitability gate are never summarized, pruned, or otherwise mutated by that run, and they are not retried by later runs because the new boundary freezes the prefix before them.

## Summarization

- Standard summaries are generated in transcript-free scratch sessions so prompts and responses stay out of the main session. Deterministic-only summaries are generated in memory without a model request or scratch session.
- Each scratch session is created without a `parentID`; it receives only a bounded exact evidence map for selected turns and the next source stop anchor, never the source transcript.
- Magic Compact registers a hidden `magic-compact-summarizer` subagent with a dedicated faithful per-turn plain-JSON summarization prompt.
- The summarizer receives only the bounded supplied historical evidence and stop anchor. It cannot inspect current or external state and must never continue unfinished work.
- Before the first standard prompt, Magic Compact resolves OpenCode's native hidden `compaction` agent through the public v2 app agents API for the source workspace. An all-deterministic plan does not query the agent API.
- The request explicitly uses the Magic Compact summarizer agent while inheriting the native compaction agent's configured model and variant. If the native compaction agent has no model, the request falls back to the captured source session model and variant.
- A missing or disabled native hidden compaction agent, or an invalid explicit model configuration, aborts compaction rather than selecting another agent.
- Scratch sessions and the hidden summarizer agent both deny `*`; prompt requests do not pass a tools map. Summarization cannot invoke built-in, plugin, MCP, shell, web, or archive tools.
- The standard prompt contains a JSON-serialized selection with bounded exact assistant and deterministic tool evidence for selected turns plus, when needed, the next user turn as the stop anchor.
- User text in the prompt excludes synthetic and ignored text and is truncated to the first line or first 300 Unicode code points, whichever is shorter. Provider-invalid code points, including lone surrogates and disallowed controls, are removed before JSON serialization.
- Every turn is keyed by the first assistant message ID. The prompt enumerates the exact required output keys and requires one unnested JSON object with narrative-body string values. The request contains no `format` property. Magic Compact concatenates all response text parts exactly, rejects missing text, parses once with `JSON.parse`, and restores summaries to plan order independently of object property order.
- Magic Compact validates the parsed object and each untouched string value independently of JSON syntax parsing before any presentation fold. Invalid JSON, a non-object shape, missing key, unknown or extra key, non-string value, empty value, whole-body recognized placeholder/marker, bracket body containing explicit replace/placeholder/TODO/TBD template wording, generic acknowledgement, one-word marker-only summary, unsafe remaining C0/C1 control, bidi override/isolate control, or unpaired UTF-16 surrogate is rejected with a typed content-free contract error. Machine-evidence validation is a closed producer-schema detector rather than a Markdown grammar: it recursively inspects standalone and prose-embedded JSON objects and arrays, including JSON stored in string values and escaped keys, for strong producer fields (`sourceBytes`, `textEncoding`, `textSegments`, and array-valued `windows`), exact internal accounting labels, and producer record `kind` values. It also recognizes JSON double-quoted, single-quoted, backtick-quoted, and unquoted protected assignments after bounded punctuation, ordinal, and GFM-task prefixes, including syntactically visible assignments in malformed leading object fragments. Ordinary prose such as `Set sourceBytes = 128.` and `Documented the JSON field "kind": "tool".`, unrelated JSON, arbitrary arrays, Markdown links, and paths such as `app/[slug]/page.tsx` remain valid. There is no fence stripping, JSON repair, permissive extraction, positional fallback, XML fallback, or open-ended Markdown-prefix blacklist.
- After raw validation, standard narrative values split on CRLF, LF, CR, vertical tab, form feed, NEL, U+2028, and U+2029; each segment is trimmed, empty segments are discarded, and remaining segments join with one ASCII space. Boundary tabs/spaces are normalized while internal content, valid surrogate pairs, and ordinary Unicode remain intact. Raw `&` becomes `&amp;` before raw `<` becomes `&lt;`, so one renderer character-reference pass leaves model-supplied entity and tag text inert; `>`, quotes, punctuation, paths, and inline code remain content. Host code then persists exactly two physical lines, `Outcome: <classified outcome>` and `Work: <single-line narrative>`, and rejects any assembled capsule with another LF, alternate line separator, unsafe control, unpaired surrogate, raw `<`, ampersand outside the host escapes, or non-substantive work. The final rendered capsule is the value used by per-turn and aggregate token/profitability projection.
- OpenCode `1.18.15` cannot safely persist an explicit `session.prompt.format`: both JSON-schema and text formats can write an `OutputFormat*` value that later SDK reads reject. Scratch prompts therefore omit `format` and use strict plain-text JSON until this upstream behavior is revalidated.
- Standard batches are packed by the 24K local-token input limit rather than a fixed turn count, and at most two requests run concurrently.
- Every deterministic-only turn receives no primary or repair model request. Mere size, message count, or tool count does not make a turn deterministic-only.
- Magic Compact classifies complete tool input, output, and error values, assistant-message error metadata, and complete real assistant text before rendering anything. Standard-turn evidence maps stay machine-structured and exact: assistant-text records keep unique source text in `text`, and `textEncoding: "ordered-literal-runs-v1"` with `textSegments: [{literal, repetitions}]` is emitted only when a whole source part is provably periodic — an optional unique prefix with repetitions 1 followed by one exact periodic run whose literal repeated that many times reproduces the source code-unit-exactly (found in linear time by a prefix-function minimal-period check); mixed, near-identical, or nonidentical source stays literal and is never merged. Assistant-message status records preserve the message ID, error type, message, provider and numeric status when present, plus whether a later successful assistant message recovered the sequence. UTF-8-safe exact windows and anchors come from cue spans, so a middle failure such as `Process exited with code 1` survives long surrounding output. Tool input, output, and error keep exact JavaScript code units, including NUL, XML-invalid controls, lone surrogates, and U+FFFE/U+FFFF. One deterministic record serializer is used for machine evidence and local-token budgeting: it starts from `JSON.stringify` exact JS strings, then reversibly escapes any remaining XML-invalid code points as lowercase `\uXXXX` sequences. Deterministic visible capsules apply the same escaping to rendered prose and excerpted tool text so the stored summary remains provider-safe without raw invalid XML characters.
- Source classification tracks verification and VCS/deployment independently. Structured tool state and assistant prose use separate classifiers. Tool verification context is established only by actual tool identity that explicitly represents execution/test runners or by a structured command that clearly identifies tests, checks, lint, typecheck, build, smoke, or known runners; tool VCS/deployment context is established only by actual tool identity that explicitly represents git/VCS/deploy or by a structured command that clearly identifies git/VCS/deployment. Tool identities are normalized before matching by splitting snake_case, kebab-case, dots, slashes, colons, and other namespace separators, so names such as `run_tests`, `git_status`, `plugin.run-tests`, and namespaced equivalents establish verification or VCS context. Normalized identities that begin with or contain explicit passive semantics (`read`, `get`, `list`, `search`, `view`, `inspect`, `fetch`) remain neutral even when the rest of the name includes tests/build/git/deploy tokens (`plugin.read-tests`, `read_build`, `list-git-status`); identity-based verification/VCS matching is skipped for those names, while a structured command field still establishes context from command text. Tool titles and passive path labels such as `Read unit.test.ts`, `Read build configuration`, or `Read deployment config` never establish verification or VCS context. Structured verification commands recognize package runners (`bun`/`npm`/`pnpm`/`yarn` test|check|lint|typecheck|build|smoke), language runners (`pytest`, `vitest`, `jest`, `mocha`, `cargo test`, `go test`, `tsc --noEmit`), and explicit task runners (`make`/`just`/`task` test|check|lint|typecheck|build|smoke) without matching arbitrary prose. Only inside those contexts does tool result text contribute success, failure, or not-run polarity. Any tool state of error, running, pending, or other incomplete status remains unresolved regardless of output text. Completed arbitrary read, task, or custom tool output such as `No unresolved issues`, `The value was passed to serializer`, `configuration remains unchanged`, `pending color`, or `fail\uD800ed` is neutral and cannot create required verification or unresolved evidence. Independently, any actual command-running tool identity (`bash`, `shell`, `terminal`, `exec`, and normalized namespaced variants) whose completed state output or error shows strong execution failure or noncompletion (nonzero exit, permission denied, command not found, timeout, cancelled/aborted/killed/signal) is required unresolved/failure evidence even when the command is not verification or VCS; passive read/task/custom completed outputs remain neutral. Mutation tools are categorized by tool identity; assistant mutation prose remains separate and still honors negated forms. Assistant prose requires explicit test, check, command, work, VCS, or deployment context. Complete assistant message text is classified before any segmentation so cross-sentence evidence such as `Tests ran. They failed.` or `Tests could not run. Database unavailable.` remains unresolved or partial; segmented windows are used only for bounded rendering after that full-text classification, and a separate positive run/pass statement still overrides earlier no-run wording. Resolved negations such as `No unresolved issues`, `Nothing remains to do`, `no pending work`, `no tasks are pending`, `no work is pending`, `no issues remain pending`, analogous work/task/issue/test/check forms, `implementation is not incomplete`, `No files were modified`, and `configuration remains unchanged` take precedence over incidental words, while real incomplete forms such as `tasks are pending` and `work remains: run smoke` stay unresolved. Neutral VCS evidence remains a VCS source record and is not converted into not-run or failure. The precedence-aware classifier recognizes pytest, Vitest, TypeScript, and the other supported runners. `All tests passed; no tests failed` and `10 passed, 0 failed` are successful, while `No tests ran; 0 tests failed`, `No tests failed because tests were not run`, `Tests couldn't/could not/cannot/can't be run or performed`, `tests were skipped`, and `no tests have run` are not-run unless a separate positive run/pass statement exists. `not pushed`, `not committed`, `not deployed`, and `push was not run` are not-run rather than successful. Positive failures, nonzero exits, permission denial, missing commands, timeouts, cancellation, aborts, kills, and signals are failed and unresolved when in verification or VCS context, when tool state is error, or when a command-running tool identity shows those strong execution failures in state output/error. Substantive assistant `TODO`/`TBD` markers are incomplete/unresolved. Bare nouns such as `passed`, `remains`, `modified`, `build`, `check`, push notification, deployment configuration, git client parser, or branch are not execution evidence by themselves.
- A deterministic-only summary is a model-free, human-readable continuation capsule. It starts with an `Outcome:` line and appends only the sections its evidence requires: `Work:` (all substantive text parts in the final assistant message joined in source order when the ordered response fits, otherwise source-anchored continuation cues plus earlier real assistant text and mutation work as budget allows), `Verification:`, `VCS:`, `Unresolved:`, and `Next:`. Outcome is `partial` when any source record evidences failure, not-run, incomplete, blocked, credentials missing, inability to finish, pending, running, or unresolved work; it is `analysis-only` when there is no execution, mutation, verification, or VCS/deployment evidence; otherwise it is `completed`. Empty sections are omitted, never rendered as empty boilerplate. Visible capsules contain no standalone raw evidence records, source-window arrays, record counts, or machine schema.
- Historical assistant text contains only real assistant text-part records. It includes ignored model-visible assistant text and excludes synthetic text, existing Magic summaries, empty text, and whole-body placeholders only (`[TODO]`, `[TBD]`, `[Replace...]`, plain `TODO`/`TBD`/`unknown`/`N/A`, and other marker-only bodies). Bracketed JSON arrays and substantive prose that merely mentions TODO/TBD stay. All substantive text parts in the final assistant message form one ordered response and are retained in full whenever the complete capsule remains under the safety ceiling. If it does not, every classified decision, verification, VCS, unresolved, and next-action cue across those parts is extracted from its source location, while remaining optional prose uses head/tail excerpts. Assistant and tool text that reaches a capsule is escaped (for example `\u0000`/`\ud800`/`\ufffe`) so the rendered summary stays free of raw invalid XML/provider characters. Earlier real assistant text follows by priority and source order while budget remains.
- A deterministic-only summary has a 2,048 local-token safety ceiling. Required failed, not-run, running, error, unknown, incomplete, and unresolved evidence renders as readable lines with exact actionable commands and paths first; optional resolved evidence has a 512-token allowance. When required lines exceed the ceiling, only genuinely equivalent normalized facts may collapse into one readable line with a truthful `×N` count. Different commands, paths, details, decisions, or repair actions are distinct facts and are never replaced by an identifier sample. If all distinct required facts cannot fit, rendering fails closed before backup or mutation. Error tool parts remain active with their full I/O, and the final rendered summary remains subject to the same cap assertion.
- Ordinary low-risk summaries use the exact-ID plain-JSON protocol. The prompt requires one narrative paragraph per turn, uses inline code for exact identifiers and commands, and asks for no headings or bullets: correctness and continuation fidelity come before brevity, every continuation-critical fact and the exact next action must be preserved, identifiers appear only when the next agent needs them, and tool structure or evidence the transcript already keeps is not restated. The model supplies narrative facts only; host code supplies the classified outcome and exact two-line framing.
- Each summary is written as a text part on the first assistant message in the summarized turn, but only for turns whose exact projected replacement (summary injection plus the same planned pruning) is strictly smaller in canonical tokens than the turn's exact source content. A generated summary that cannot shrink its turn is discarded; the turn keeps every message and part byte-identical, and it falls inside the frozen prefix once the new boundary is injected.
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
- Scratch summarization does not route omission reads; required source evidence is included directly in the bounded prompt map.

## Omission Retrieval

- The plugin exposes `read_omitted_content` as an OpenCode plugin tool.
- The tool accepts one argument: `contentId`.
- The tool receives `context.sessionID` from OpenCode and reads only that session's omission cache.
- Retrieval never routes by the diagnostic ID prefix or opens another session cache.
- If no authorized matching qualified entry exists, integrity proof fails, the cache is invalid, or a bare legacy ID is requested, the tool returns an explicit unavailable/integrity diagnostic and never guessed bytes. Non-integrity cache failures use a generic diagnostic that does not expose local filesystem or session paths; typed integrity failures retain path-free validation detail.

## Stats

- Stats are stored under `${XDG_DATA_HOME:-~/.local/share}/opencode/storage/magic-compact/stats/{sessionId}.json`.
- Stats cache format version is `1`.
- Stats track `rootSessionId`, `sourceSessionId`, `totalTokensPruned`, `cachedTokensSaved`, and processed assistant message IDs.
- Each compaction adds the current token reduction to `totalTokensPruned`.
- Each trim adds the exact reduction from its pre-trim canonical active count to the post-trim pre-notice active count. The ignored trim notice is excluded, and the final actual active count must equal that projection.
- OpenCode assistant message events add `totalTokensPruned` to `cachedTokensSaved` once per assistant message after stats exist.
- `/magic-stats` injects an ignored stats summary notice, or a no-stats message if no stats exist.
- Every stats notice labels its counts as canonical active-content estimates, explicitly says the ignored reporting notice is excluded, and distinguishes them from provider context-window usage. Model-facing synthetic content remains counted. Each notice states that OpenCode's context meter is a provider usage snapshot that can remain stale after in-place compaction until a later assistant response refreshes it. Magic Compact never fabricates an after-provider count or mutates provider token metadata.

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

- Any standard-summary LLM, plain-JSON contract, completeness, deterministic rendering, SDK, cache, stats, token counting, projection, or pruning failure aborts the attempt, except for the single typed contract retry described above.
- Compaction parts on non-user messages, multiple compaction parts, ambiguous summary children for one marker, orphan summaries, overlapping checkpoint intervals, dangling or forward `tail_start_id` values on any completed candidate, and active incomplete or errored markers fail clearly before backup or mutation. An older ambiguous or incomplete marker is ignored only when all of its summary children are wholly before the marker of a later valid completed checkpoint.
- The ordered canonical source snapshot is captured during planning and compared structurally after backup creation immediately before any Magic summary, boundary, or pruning write. It covers all behavior-affecting message info, parts, tool states, and ordering while excluding volatile timestamps and token/cost accounting. Fetch, unwrap, checkpoint inspection, or structural drift raises `NativeCheckpointChangedError` because unchanged source state was not confirmed.
- Cleanup deletes every scratch session. On timeout or client cancellation, routed server-side `session.abort` completes before deletion; prompt, abort, and deletion failures are aggregated without retry. Scratch cleanup, rollback, and notification failures retain their normal aggregate behavior. Contract failures surface as content-free `SummaryContractError` values with source session ID, turn ID, category, batch, and attempt numbers.
- All summaries, planned omission IDs, and projected counts remain in memory until every request completes, the useful-savings gate passes, backup initialization succeeds, and full source revalidation passes. There is no pre-validation source progress message.
- Other failures attempt to promote an existing backup by renaming and selecting it before deleting the source. Final actual-versus-projected mismatch is instead a preserve-both source-drift failure and names the backup for manual recovery.
- No-op, success, and failure toast SDK responses are unwrapped. Failure-toast errors are retained with the primary and rollback errors. A success-toast failure occurs after durable compaction and surfaces without promoting the backup or deleting the successful source.
- The command hook throws so OpenCode does not continue sending the slash command to the LLM.

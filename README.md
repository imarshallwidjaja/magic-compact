# Magic Compact

Lossless context compression for OpenCode.

<p align="center">
  <img src=".github/assets/preview.png" alt="Magic Compact Preview" />
</p>

## Why

OpenCode's built-in compaction replaces an entire conversation with one summary blob. The user messages, the assistant's reasoning, tool calls, design decisions, and workflow are all flattened into a generic template (Goal, Progress, Key Decisions...). The agent wakes up with amnesia, forced to reconstruct its working state from an abstraction that captured a fraction of what mattered.

Magic Compact takes a different approach: preserve the conversation skeleton, condense each old assistant turn into its own summary, and keep omitted tool I/O retrievable. The agent retains its working flow without claiming a separate archive for optional assistant evidence omitted by bounded summaries.

## How

Instead of collapsing an entire session into a single generic summary, Magic Compact replaces old assistant turns with high-fidelity summaries while leaving user messages and tool calls in place.

Exact user messages remain in context. Resolved assistant turns become readable per-turn continuation capsules; unresolved or failed state stays visible as decisive readable evidence, and when it cannot be represented truthfully within the safety caps, compaction fails closed. Native checkpoints and source drift are guarded, and pruned tool I/O stays retrievable through `read_omitted_content`.

<p align="center">
  <img src=".github/assets/visualization.png" alt="Compaction Comparison" />
</p>

## Features

- Lossless boundary — Preserve exact user messages, native checkpoint safety, required failure/not-run/unresolved evidence, and retrievable omitted tool I/O instead of flattening history into one recap.
- Zero compaction overhead — Compaction happens once on your command rather than during the agentic loop. Maximum token savings, minimal cache invalidations.
- Preserved user messages — Exact requirements and guidance remain visible to the agent, verbatim.
- Smart tool call pruning — Bulky completed tool I/O is replaced with omission notices, with original content cached and retrievable on demand via `read_omitted_content`.
- Recompactable — Run `/magic-compact` again later to compact new turns while preserving prior summaries.
- Trim without summarizing — OpenCode can prune historical tool I/O with `/magic-trim` while preserving assistant responses.
- Bounded summarization — Completed turns use transcript-free scratch sessions with bounded exact evidence, deny-all permissions, a best-effort 4,000-token provider output cap that only clamps an active limit OpenCode supplies, and at most two concurrent requests. Model output must validate as a human-readable continuation capsule with an evidence-consistent Outcome and no raw machine evidence. Failed, not-run, running, or otherwise unresolved turns stay model-free. Their deterministic capsules treat every substantive text part in the final assistant message as one ordered final response and preserve it in full when it fits the remaining 2,048-token safety ceiling; otherwise they preserve every classified decision, verification, VCS, unresolved, and next-action fact as source-anchored readable evidence. Distinct required facts are never sampled away: if they cannot all fit, rendering fails closed before mutation while underlying error tool parts stay active.
- Useful-savings gate — A turn is replaced only when its exact projected replacement (summary plus the same planned pruning) is strictly smaller than the turn's exact source. Net-negative turns stay byte-identical. If the projected result cannot save at least 20% of the selected profitable turns' eligible removable content, Magic Compact fails unchanged before backup, mutation, or a stats increment instead of claiming weak savings.

## Installation

Install from the CLI:

```bash
rm -rf ~/.cache/opencode/packages/magic-compact*

opencode plugin magic-compact --global

# If you are encountering "No versions available":
NPM_CONFIG_MIN_RELEASE_AGE=0 opencode plugin magic-compact --global
```

This installs the package and adds it to your global OpenCode config.

## Usage

### `/magic-compact`

To compact, run `/magic-compact [N]` with an optional argument indicating how many turns to preserve.

- `N` is the number of recent assistant turns to preserve as-is. Default: `0` (summarize everything).
- A backup session is created before the first source write. Pre-backup failures leave the current source unchanged; later ordinary failures recover through the backup.
- Summary generation and the useful-savings projection run before backup creation without mutating the source. A backup still exists before the first durable compaction write.
- A timed-out or client-cancelled scratch prompt explicitly aborts the server-side scratch runner before deleting its session. If final token validation detects late source drift after writes, both the partially mutated source and its named backup are preserved for manual recovery and no successful stats increment is recorded.

Examples:

- `/magic-compact` — summarize all old assistant turns.
- `/magic-compact 3` — keep the 3 most recent assistant turns, summarize the rest.

### `/magic-trim` (Experimental)

Run `/magic-trim [N]` to apply the same tool I/O pruning rules without summarizing or deleting ordinary user and assistant content.

- `N` preserves tool I/O in the most recent assistant turns. Default: `0` (trim all eligible turns).
- Trimming does not call an LLM or create a compaction boundary.
- A backup session is created before trimming.
- Token reductions are included in `/magic-stats` on the exact canonical active-content basis. The ignored trim notice is stored for display but excluded from that model-facing count.

Examples:

- `/magic-trim` — trim eligible tool I/O throughout the session.
- `/magic-trim 3` — preserve tool I/O in the 3 most recent assistant turns.

### `/magic-stats`

Run `/magic-stats` to show cumulative token savings for the current conversation: tokens pruned, cached tokens saved, estimated money saved, among other statistics.

Reported token counts are canonical active-content estimates, not provider usage. They include model-facing synthetic content such as the compaction boundary, exclude ignored user reporting notices including the notice that displays the count, and include ignored assistant text. OpenCode's context meter is a provider usage snapshot and can remain stale after in-place compaction until a later assistant response refreshes it. Magic Compact never mutates provider usage metadata or fabricates an after-provider count.

### The Omitted Content Tool

Magic Compact registers a `read_omitted_content` tool that the agent can call to retrieve any tool input or output that was pruned during compaction or trimming.

Each omission notice in the conversation includes an integrity-protected Content ID such as `2c4f6a8b0d1e:omitted-yH8kQ2pL5vN7sR4tU6wXzA`. The agent uses that ID to fetch the original content when it needs stale information that cannot be reproduced via a new tool call. OpenCode preserves historical v1 bare entries during migration and backup copying, but cannot retrieve them because they have no cryptographic binding. Unavailable or integrity-invalid entries fail closed instead of returning guessed bytes.

## Pruning Rules

During `/magic-compact`, pruning applies only to summarized turns. `/magic-trim [N]` applies only the tool I/O rules to turns outside the preserved tail.

Kept:

- User messages (verbatim)
- Per-turn summaries
- Tool calls (structure preserved)
- Selected high-value synthetic messages (shell wrappers, background task results, working-directory change reminders)

Removed or condensed:

- Assistant reasoning, text, and step markers — replaced by the per-turn summary
- Most synthetic/injected messages (file expansions, plan reminders, prior compaction notices, etc.)
- Bulky completed tool I/O — replaced with an omission notice pointing to the cache

### Tool I/O Rules

Completed tool outputs over 128 words or 1024 characters are omitted by default. A few tools have special handling.

- `read` — output always omitted (stale file contents are reloadable)
- `write` / `edit` / `apply_patch` — large file content omitted
- `bash` — commands over 1024 characters are truncated
- `task` — output omitted above a higher threshold (512 words / 4096 characters)
- `question` — input and output preserved
- `todowrite` / `skill` — output discarded without caching (redundant or reloadable)

Pending, running, and errored tool calls are always preserved as-is.

Completed tool calls processed by pruning are marked so later trim and compaction operations do not trim them again.

## Vs DCP Plugin (OpenCode)

OpenCode-DCP is a runtime context management system that rewrites messages when requested by the model. Magic Compact takes a different approach.

Magic Compact Offers:

- Simplicity — One command, zero configuration.
- Lossless quality — Turn-by-turn flow stays intact. All user commands are preserved. All past tool calls are preserved.
- Maximum token savings — Safe completed turns are packed into bounded evidence-only scratch requests instead of repeatedly sending the full transcript. Turns with failed, not-run, running, error, unknown, or incomplete evidence use deterministic human-readable capsules where decisive state is required. Only normalized-equivalent repeated facts aggregate with truthful counts; distinct actions remain visible or compaction fails closed. Long tool calls are aggressively pruned.
- No cache churn — Compaction happens once and is cache friendly, whereas DCP may invalidate entire conversations multiple times within one request.
- Zero assistant overhead — No prompt injections asking the assistant to compact. Your assistant stay focused on its task.

If you want model-driven compaction with increased cache invalidations and token burn, consider using DCP instead.

## Vs Magic Context (OpenCode)

Magic Context is a much broader runtime context-management system: it runs background historian and dreamer processes, maintains project memory, and injects recalled memories and history back into the prompt on an ongoing basis. That makes it powerful, but also much heavier in tokens and cache churn.

Magic Compact Offers:

- Efficiency — One explicit compaction command, no background summarization loop, no always-on memory RAG, and no recurring prompt injections.
- Lower token burn — Context reduction happens once on demand instead of continuously consuming tokens across every turn.
- Fewer cache invalidations — The session is rewritten once, then stays stable, instead of repeatedly re-rendering volatile background state.
- Lossless conversation shape — User messages stay verbatim, tool structure is preserved, and omitted tool I/O remains retrievable. Optional resolved detail may be omitted from a bounded capsule, but distinct continuation-critical facts are preserved or compaction fails closed; assistant summary text has no separate archive.
- Focused compaction — The plugin does one job: compress the conversation without turning the runtime into a memory subsystem.

If you want a lightweight, user-driven compaction tool with minimal ongoing overhead, Magic Compact is the better fit. If you want a full long-term memory system with background maintenance, Magic Context is the heavier alternative.

## Development

See [`docs/Development.md`](./docs/Development.md) for setup and maintenance commands.

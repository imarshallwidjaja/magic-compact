# Project

Arkive Compact is the distribution package for the Magic Compact lossless context compression runtime for OpenCode. The runtime remains Magic Compact. See `README.md` for the user-facing overview, features, installation, source map, and pruning rules.

This file is the development reference for agents working on this repo.

---

## Development Docs

Agents MUST read these files before working on platform behavior, and MUST update them whenever behavior changes:

- `docs/Development.md` — setup, repository layout, and maintenance commands.
- `docs/Core.md` — core behavior and safety guarantees.
- `docs/OpenCode.md` — OpenCode runtime behavior specification.

---

## Architecture

Keep the combined entrypoint's V1 `server()` and V2 `setup()` implementations independent. Support V1 only from 1.18.29 onward. V1's SDK `/v2` import does not denote the V2 host API; never cast V2 request messages to V1 persistence records or claim persisted-history parity. V2 projects only agent-loop `context` requests where omission retrieval is usable. Preserve user messages and source objects, serialize settings updates per session, treat invalid stored settings as defaults until a verified update repairs them, verify durable omission storage before publishing request projections, and keep native compaction semantics explicit in operator docs.

On V1, Magic Compact performs per-turn conversation compaction while preserving the conversation skeleton. V2 uses request-only tool-output projection; the runtime surfaces and summary goals below describe V1.

### Runtime Surfaces

- `/magic-compact [N]` slash command.
- `/magic-trim [N]` slash command.
- `/magic-stats` slash command.
- `read_omitted_content` tool.

### Goals

- Preserve the current conversation structure while reducing context size.
- Preserve user messages exactly.
- Replace old assistant turns with validated per-turn continuation summaries.
- Keep useful tool calls visible while removing bulky tool I/O.
- Store omitted tool I/O outside context but make it retrievable.
- Keep a recovery path available on failure.

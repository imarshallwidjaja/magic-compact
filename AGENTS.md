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

Magic Compact performs per-turn conversation compaction while preserving the conversation skeleton.

### Runtime Surfaces

- `/magic-compact [N]` slash command.
- `/magic-trim [N] [--heuristic]` slash command.
- `/magic-stats` slash command.
- `read_omitted_content` tool.

### Goals

- Preserve the current conversation structure while reducing context size.
- Preserve user messages exactly.
- Replace old assistant turns with validated per-turn continuation summaries.
- Keep useful tool calls visible while removing bulky tool I/O. Task-aware retention (TypeSafe Jev, `TYPESAFE_API_KEY` in the OpenCode server environment) can keep a call whole; `--heuristic` skips it.
- Store omitted tool I/O outside context but make it retrievable.
- Keep a recovery path available on failure.

## Task-Aware Retention

- Tests never read `TYPESAFE_API_KEY` or reach the network: pass `retention` with a fake `fetch` (see `packages/opencode-plugin/test/retention-host.ts`), or `{ mode: "heuristic" }` for tests of the unchanged tool rules.
- Do not change `RETENTION_KEEP_THRESHOLD` or the Jev question wording without the evaluation protocol in `docs/RetentionEvaluation.md`; never tune either against held-out results.
- Judge calls independently: useful duplicates and recoverable content qualify, as does unchanged useful content in a partly edited read. Do not categorically exclude task procedures, skills or todos. Keep the frozen 0.80 threshold and request allocations until a new development-only calibration is frozen; bytes are not a documented tokenizer bound.

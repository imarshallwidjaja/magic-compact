# Project

Arkive Compact is the distribution package for the Magic Compact lossless context compression runtime for OpenCode. The runtime remains Magic Compact. See `README.md` for the user-facing overview, features, installation, source map, and pruning rules.

This file is the development reference for agents working on this repo.

---

## Development Docs

Agents MUST read these files before working on platform behavior, and MUST update them whenever behavior changes:

- `docs/Development.md` — setup, repository layout, and maintenance commands.
- `docs/Core.md` — core behavior and safety guarantees.
- `docs/OpenCode.md` — OpenCode runtime behavior specification.

`.github/workflows/check.yml` and the check steps in `.github/workflows/publish-opencode.yml` run the same commands before merge and before `npm publish`. Change them together.

## Host Integration

- Use public host APIs only. V1 compact and trim require a verified listener; V2 requires a verified registered-service or explicit endpoint connection. See `docs/OpenCode.md` for setup and identity checks.
- When changing host adapters or package entrypoints, test the packed artifact in containers on both supported hosts. Do not use a live host session for qualification.
- Compaction evals may use an authenticated local OpenCode server only with fresh, run-owned synthetic sessions in an isolated temporary project. Never supply existing user sessions or alter global configuration. Keep provider credentials in the server; qualify host-runner changes in containers first.

---

## Architecture

Magic Compact performs per-turn conversation compaction while preserving the conversation skeleton.

### Runtime Surfaces

- `/magic-compact [N] [--backend jev|openai-decisions] [--threshold P] [--truncate] [--match-model]` or `/magic-compact [N] --heuristic [--match-model]` slash command.
- `/magic-trim [N] [--backend jev|openai-decisions] [--threshold P] [--truncate] [--match-model]` or `/magic-trim [N] --heuristic` slash command.
- `/magic-recover` slash command.
- `/magic-stats` slash command.
- `read_omitted_content` tool.

### Goals

- Preserve the current conversation structure while reducing context size.
- Preserve user messages exactly.
- Replace old assistant turns with validated per-turn continuation summaries.
- Keep protected reasoning on the host's native path. Only owned summarization requests may project readable reasoning as historical text; never serialize protected payloads into prompts or indexes. Match summary models only with `--match-model`, using preflight fallback; keep replay-cost estimates separate from stored-content and cached-read statistics. See `docs/OpenCode.md` for qualified paths.
- Keep useful tool calls visible while removing bulky tool I/O. Task-aware retention (TypeSafe Jev by default, or OpenAI Decisions, with the selected backend's key in the OpenCode server environment) can keep a call whole; `--heuristic` skips it (on `/magic-compact` it skips retention only, and turns are still summarized).
- Store omitted tool I/O outside context but make it retrievable.
- Write compact, trim and recover results to a verified successor session; never mutate or delete the original session or its child sessions.
- Compact and trim refuse stored unfinished records and undelivered background results. Only the explicit `/magic-recover` treats them as abandoned: it keeps every live-work check, closes records with the host's native interrupted form and refuses records that have none. Its successor also carries one complete model-visible recovery context (`recoveryContext`) for what those stored forms leave out; a new record kind updates it. Never infer abandonment from age or idleness or route compact or trim through recovery.

## Task-Aware Retention

- Tests never read `TYPESAFE_API_KEY` or `MAGIC_COMPACT_OPENAI_DECISIONS_API_KEY` or reach the network: pass `retention` with a fake `fetch` (see `packages/opencode-plugin/test/retention-host.ts`), or `{ mode: "heuristic" }` for tests of the unchanged tool rules.
- The operator-selected default is 0.60; the server may override it with `MAGIC_COMPACT_KEEP_THRESHOLD_PERCENT`. Further default or Jev-question changes require the evaluation protocol in `docs/RetentionEvaluation.md`; never tune against held-out results.
- Keep archived evaluation thresholds independent of the runtime default and environment override.
- Judge calls independently: useful duplicates and recoverable content qualify, as does unchanged useful content in a partly edited read. Do not categorically exclude task procedures, skills or todos. Send the session summary whole by default and keep the other request allocations fixed until a new development-only calibration is frozen. Enforce TypeSafe's 32 MiB request-body limit locally for Jev only; let the selected backend enforce its token limits. Only `--truncate` on /magic-compact or /magic-trim may retry a confirmed token-limit refusal, once per rejected candidate with the summary head/tail excerpt capped at 8,000 UTF-8 JSON bytes. No other request fields change. The only other retry is an OpenAI Decisions HTTP 503: the identical request, at most twice per candidate across both summary variants, after bounded backoff or a `Retry-After` of at most 30 seconds. Do not add retries for Jev, other statuses, timeouts or network errors. Unrecovered token-limit refusals fail the whole command unchanged with the person's own command and `--heuristic` guidance (/magic-compact's says turn summaries are still generated); do not add byte-to-token proxies or automatic heuristic fallbacks.
- Resolve --backend before MAGIC_COMPACT_RETENTION_BACKEND, defaulting to jev. Read only the selected backend's key (TYPESAFE_API_KEY or MAGIC_COMPACT_OPENAI_DECISIONS_API_KEY, never the generic OPENAI_API_KEY); never load .env in plugin code or fail over between backends. Share retention evidence, thresholds, scheduling and feedback; keep serialization, limits and exact error recognition backend-specific. Tests inject fake credentials and fetch for both backends.
- Retention arguments, policy, retry, failure advice and reporting belong to /magic-compact and /magic-trim on both hosts through the shared parser, policy builder and formatter. Diverge only for an operation-specific reason.

# Development

This page covers the public setup, repository layout, and common maintenance commands for contributors.

## Requirements

- Bun

## Setup

Install dependencies from the repository root:

```bash
bun install
```

## Repository Layout

### `packages/`

- `packages/opencode-plugin` - OpenCode plugin implementation.
  - `src/api.ts` - OpenCode API access helpers.
  - `src/index.ts` - package export surface.
  - `src/magic-compact.ts` - `/magic-compact` command entrypoint.
  - `src/magic-trim.ts` - `/magic-trim` command entrypoint.
  - `src/magic-stats.ts` - `/magic-stats` command entrypoint.
  - `src/tui.ts` - TUI plugin wiring.
  - `src/util.ts` - shared local utilities.
  - `src/compact/compact.ts` - main compaction flow.
  - `src/compact/constants.ts` - compaction constants and custom summarizer-agent definition.
  - `src/compact/plan.ts` - turn planning and boundary selection.
  - `src/compact/prune.ts` - tool input/output pruning.
  - `src/compact/session.ts` - session backup and mutation helpers.
  - `src/compact/template.ts` - summarization prompt template helpers.
  - `src/storage/omission.ts` - omitted content cache persistence and temporary summarization-session lookup routing.
  - `src/storage/stats.ts` - persisted stats storage.
  - `src/storage/store.ts` - shared storage access helpers.
  - `src/stats/constants.ts` - stats constants.
  - `src/stats/events.ts` - stats event accounting.
  - `src/stats/pricing.ts` - pricing calculations.
  - `src/stats/tokenize.ts` - token estimation helpers.
  - `test/trim.test.ts` - trim planning and metadata idempotency tests.
- `packages/claude-code-plugin` - Claude Code plugin implementation for the port.
  - `src/command.ts` - Claude Code slash command entrypoint.
  - `src/compact.ts` - Claude Code compaction flow.
  - `src/hook.ts` - Claude Code hook integration.
  - `src/index.ts` - package export surface.
  - `src/mcp.ts` - MCP integration helpers.
  - `src/omission.ts` - omitted content handling.
  - `src/prune.ts` - transcript pruning helpers.
  - `src/transcript.ts` - transcript parsing and formatting.
- `packages/common` - shared utilities for cross-package code.
  - `src/index.ts` - shared export surface.

### `docs/`

- `docs/Development.md` - contributor setup and repository map.
- `docs/OpenCode.md` - OpenCode runtime behavior specification.
- `docs/ClaudeCode.md` - Claude Code runtime behavior specification.

## Common Commands

- `bun run typecheck` - TypeScript type checking
- `bun run lint` - ESLint checks
- `bun run format` - Prettier formatting

## Notes

- The project uses Bun workspaces.
- For the user-facing plugin install flow, see the main `README.md`.
- The native-checkpoint cases in `packages/opencode-plugin/test/compact.test.ts` are maintenance canaries for user `compaction` parts with optional `tail_start_id`, and assistant `parentID`, `summary`, `finish`, and `error` fields. Check those fields against OpenCode's compaction implementation whenever OpenCode or `@opencode-ai/sdk` is upgraded.
- Native checkpoint race detection uses structural revalidation immediately before the progress notice and again immediately after ephemeral summary generation, before Magic source writes. There is no journal or lock, so unsupported concurrent source mutation after the post-summary revalidation is not protected.

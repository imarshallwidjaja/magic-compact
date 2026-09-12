# Arkive Compact

`arkive-compact` contains separate implementations: lossless persisted compaction for OpenCode V1 and request-time tool-output projection for OpenCode V2.

| Host                                  | Behavior                                                                                                    |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| OpenCode V1 >=1.18.29, <2             | Lossless per-turn compaction, trimming, backups, omission retrieval, media projection, and cumulative stats |
| OpenCode V2 (validated against 2.0.2) | Request-time tool-output projection on agent-loop context requests without changing saved history           |

## OpenCode V1

Add the package to the singular `plugin` setting:

```json
{
  "plugin": ["arkive-compact@latest"]
}
```

V1 provides `/magic-compact [N]`, `/magic-trim [N]`, `/magic-stats`, and `read_omitted_content`.

## OpenCode V2

Add the package to the plural `plugins` setting:

```json
{
  "plugins": ["arkive-compact@latest"]
}
```

V2 provides `magic_trim`, `magic_stats`, and `read_omitted_content` as tools. It projects eligible historical tool text only for agent-loop context requests where `read_omitted_content` can be called. One-shot generate requests remain unchanged. Projection defaults to enabled with `keepTurns: 1` and never edits saved messages. Omission or projection-stats storage failure aborts the context request before publishing a projection; settings verification failure aborts the `magic_trim` tool request without reporting success.

Validation targets OpenCode 2.0.2. Installed-package activation and tool execution are covered, but provider-backed requests, live TUI behavior, registry installation of this unreleased version, and parity with other V2 releases are not.

See the [repository README](https://github.com/imarshallwidjaja/magic-compact#readme) for behavior, limits, safety guarantees, and development instructions.

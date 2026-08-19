#!/bin/sh

set -eu

version=${1:-}

if [ -z "$version" ]; then
  printf '%s\n' 'Usage: bun run release-opencode -- <version>' >&2
  exit 1
fi

line_break_free_version=$(printf '%s' "$version" | LC_ALL=C tr -d '\012\015')
if [ "$line_break_free_version" != "$version" ]; then
  printf '%s\n' 'Version must be in X.Y.Z format.' >&2
  exit 1
fi

if ! printf '%s\n' "$version" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+$'; then
  printf '%s\n' 'Version must be in X.Y.Z format.' >&2
  exit 1
fi

manifest_version=$(node -p "require('./packages/opencode-plugin/package.json').version")
tag="v$version"

if [ "$manifest_version" != "$version" ]; then
  npm version "$version" --no-git-tag-version --workspaces=false --prefix packages/opencode-plugin
  bun install
  git add "packages/opencode-plugin/package.json" "bun.lock"
  git commit -m "Release $tag"
fi

git tag "$tag"
git push origin main "$tag"

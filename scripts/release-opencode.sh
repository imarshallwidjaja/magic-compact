#!/bin/sh

set -eu

version=${1:-}
manifest=packages/opencode-plugin/package.json
newline='
'

fail() {
  printf '%s\n' "$1" >&2
  exit 1
}

if [ -z "$version" ]; then
  fail 'Usage: bun run release-opencode -- <version>'
fi

line_break_free_version=$(printf '%s' "$version" | LC_ALL=C tr -d '\012\015')
if [ "$line_break_free_version" != "$version" ]; then
  fail 'Version must be in X.Y.Z format.'
fi

if ! printf '%s\n' "$version" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+$'; then
  fail 'Version must be in X.Y.Z format.'
fi

tag="v$version"

# Every Git read below is checked on its own exit status before its output is
# used. The preflight finishes before the helper writes a file, commit or ref,
# and later steps use the commits and the destination it checked.

if branch=$(git symbolic-ref --quiet --short HEAD); then
  if [ "$branch" != main ]; then
    fail "Release must run on branch main, not $branch."
  fi
elif [ "$?" -eq 1 ]; then
  fail 'Release must run on branch main; HEAD is detached.'
else
  fail 'Could not read the current branch.'
fi

start_commit=$(git rev-parse --verify --quiet refs/heads/main) || fail 'Could not read local main.'

# Without optional locks, status does not rewrite the index to refresh cached
# file times, so a refusal leaves the index byte for byte as it was.
worktree_status() {
  git --no-optional-locks status --porcelain=v1 --untracked-files=all
}

status=$(worktree_status) || fail 'Could not read the state of the index and working tree.'
if [ -n "$status" ]; then
  fail 'Release needs a clean index and working tree, with no untracked files. Commit, stash or remove the changes first.'
fi

local_tag=$(git for-each-ref --format='%(refname)' "refs/tags/$tag") || fail 'Could not read local tags.'
if [ "$local_tag" = "refs/tags/$tag" ]; then
  fail "Tag $tag already exists locally. If origin does not have it, an earlier release stopped after tagging: follow \"Interrupted release helper\" in docs/Development.md. Never move or recreate a published tag."
fi

push_url=$(git remote get-url --push --all origin) || fail 'Could not read the push URL of origin.'
if [ -z "$push_url" ]; then
  fail 'Origin has no push URL.'
fi
case $push_url in
  *"$newline"*)
    fail 'Origin has more than one push URL. The release pushes main and the tag to one repository so that the push is all or nothing; leave one push URL on origin and retry.'
    ;;
esac

# The release reads from and pushes to this literal destination, not to origin.
# Origin's push URL is a URL or a path, but Git reads a bare word such as
# "upstream" as a remote when one of that name exists, including a legacy
# .git/remotes or .git/branches file. A relative local path is therefore
# written as ./path. Like Git, read scheme:// and helper:: as URLs and a colon
# before any slash as host:path.
destination=$push_url
case ${push_url#"${push_url%%[!A-Za-z0-9+.-]*}"} in
  '://'* | '::'*) ;;
  *)
    case ${push_url%%:*} in
      "$push_url" | */*)
        case $push_url in
          /* | ./* | ../*) ;;
          *) destination=./$push_url ;;
        esac
        ;;
    esac
    ;;
esac

# Ask Git how it reads the destination now: it must name no remote, and
# url.<base>.insteadOf and pushInsteadOf must leave it unchanged for reading and
# for pushing. The release asks again after the npm, bun and Git hooks it runs,
# which can change the configuration. Each check is fresh, not a lock: a change
# that another process makes after the last check still applies to the push.
pin=release-opencode-push-check
check_destination() {
  if git remote get-url -- "$destination" >/dev/null 2>&1; then
    fail "Git reads origin's push URL as the name of a remote, so the release cannot pin one destination. $1"
  elif [ "$?" -ne 2 ]; then
    fail "Could not check how Git reads origin's push URL. $1"
  fi
  pin_view=$(git -c "remote.$pin.url=$destination" remote -v) || fail "Could not resolve the push URL of origin. $1"
  pin_lines=
  set -f
  old_ifs=$IFS
  IFS=$newline
  for line in $pin_view; do
    case $line in
      "$pin	"*) pin_lines="$pin_lines$line$newline" ;;
    esac
  done
  IFS=$old_ifs
  set +f
  if [ "$pin_lines" != "$pin	$destination (fetch)$newline$pin	$destination (push)$newline" ]; then
    fail "Git rewrites origin's push URL when the URL is used directly, so the release cannot pin one destination; check the url.<base>.insteadOf and pushInsteadOf settings. $1"
  fi
}

# HEAD must still be the branch main and main must be at the expected commit,
# whatever a hook the release runs did to them.
moved='HEAD left branch main, or main moved, during the release.'
check_release_context() {
  head_ref=$(git symbolic-ref --quiet HEAD) || [ "$?" -eq 1 ] || fail "Could not read the current branch. $2"
  main_commit=$(git rev-parse --verify --quiet refs/heads/main) || fail "Could not read local main. $2"
  if [ "$head_ref" != refs/heads/main ] || [ "$main_commit" != "$1" ]; then
    fail "$moved $2"
  fi
  check_destination "$2"
}

check_destination 'Nothing was written.'

remote_refs=$(git ls-remote -- "$destination" refs/heads/main "refs/tags/$tag") || fail "Could not read refs from origin's push URL."
if printf '%s\n' "$remote_refs" | awk -v ref="refs/tags/$tag" '$2 == ref { found = 1 } END { exit !found }'; then
  fail "Tag $tag already exists on origin. Release tags are immutable; never move or recreate one."
fi
remote_main=$(printf '%s\n' "$remote_refs" | awk '$2 == "refs/heads/main" { print $1 }')
if [ -z "$remote_main" ]; then
  fail 'Origin has no branch main.'
fi
if git merge-base --is-ancestor "$remote_main" "$start_commit"; then
  :
elif [ "$?" -eq 1 ]; then
  fail "Origin's main ($remote_main) is not contained in local main. Fetch and integrate it before releasing."
else
  fail "Could not check that origin's main ($remote_main) is contained in local main. Fetch and integrate it before releasing."
fi

committed_version() {
  manifest_json=$(git show "$1:$manifest") || return 1
  printf '%s' "$manifest_json" | node -e '
let text = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", chunk => (text += chunk));
process.stdin.on("end", () => {
  let version;
  try {
    version = JSON.parse(text).version;
  } catch {
    process.exit(1);
  }
  if (typeof version !== "string") process.exit(1);
  process.stdout.write(version);
});
'
}

release_version=$(committed_version "$start_commit") || fail "Could not read the version of $manifest at HEAD."
release_commit=$start_commit

if [ "$release_version" != "$version" ]; then
  npm version "$version" --no-git-tag-version --workspaces=false --prefix packages/opencode-plugin
  bun install

  # Only the manifest and lockfile may change, and only they are committed;
  # anything else is left in place for the operator. $1 is the status prefix
  # they must have: unstaged after the bump, staged after `git add`.
  only_bump_changes() {
    set -f
    old_ifs=$IFS
    IFS=$newline
    for line in $status; do
      case $line in
        "$1$manifest" | "${1}bun.lock") ;;
        *)
          IFS=$old_ifs
          fail "$2"
          ;;
      esac
    done
    IFS=$old_ifs
    set +f
  }
  not_committed='No release commit or tag was created; inspect HEAD, main and the version bump left in the working tree before retrying.'
  status=$(worktree_status) || fail 'Could not read the state of the index and working tree after the version bump. No release commit or tag was created.'
  only_bump_changes ' M ' "The version bump changed files other than $manifest and bun.lock. No release commit or tag was created; inspect the changes before retrying."

  # Record the tree of the checked bump: the checked commit with only the
  # manifest and lockfile replaced. The release commit must have exactly this
  # tree, so a hook that amends its content, even within those two files,
  # stops the release.
  git add -- "$manifest" bun.lock || fail "Could not stage the version bump. $not_committed"
  status=$(worktree_status) || fail "Could not read the state of the index and working tree after staging the version bump. $not_committed"
  only_bump_changes 'M  ' "The index or working tree changed while the version bump was staged. $not_committed"
  bump_tree=$(git write-tree) || fail "Could not record the tree of the version bump. $not_committed"
  check_release_context "$start_commit" "$not_committed"
  git commit -m "Release $tag" -- "$manifest" bun.lock

  # The release commit is the one commit main gained on top of the checked
  # commit, with the checked tree; a hook that commits again, moves main or
  # amends the commit's content stops the release.
  committed='The release commit exists and no tag was created; inspect HEAD and main before retrying.'
  release_commit=$(git rev-parse --verify --quiet refs/heads/main) || fail "Could not read local main. $committed"
  parents=$(git rev-list --parents -n 1 "$release_commit") || fail "Could not read the release commit. $committed"
  if [ "$parents" != "$release_commit $start_commit" ]; then
    fail "$moved $committed"
  fi
  release_tree=$(git rev-parse --verify --quiet "$release_commit^{tree}") || fail "Could not read the release commit. $committed"
  if [ "$release_tree" != "$bump_tree" ]; then
    fail "The release commit's content differs from the checked version bump. $committed"
  fi
  release_version=$(committed_version "$release_commit") || fail "Could not read the version of $manifest in the release commit. No tag was created."
fi

# The tag must name a commit whose manifest has the requested version, and the
# working tree must match that commit exactly.
if [ "$release_version" != "$version" ]; then
  fail "The release commit's $manifest has version $release_version, not $version. No tag was created."
fi
status=$(worktree_status) || fail 'Could not read the state of the index and working tree. No tag was created.'
if [ -n "$status" ]; then
  fail "The index or working tree changed during the release bump and does not match HEAD. No tag was created; inspect HEAD and the changes before retrying."
fi
check_release_context "$release_commit" 'The release commit exists and no tag was created; inspect HEAD and main before retrying.'

git tag "$tag" "$release_commit"

tagged="Tag $tag exists locally and nothing was pushed; follow \"Interrupted release helper\" in docs/Development.md."
tag_object=$(git rev-parse --verify --quiet "refs/tags/$tag") || fail "Could not read tag $tag. $tagged"
tag_commit=$(git rev-parse --verify --quiet "$tag_object^{commit}") || fail "Could not read the commit of tag $tag. $tagged"
if [ "$tag_commit" != "$release_commit" ]; then
  fail "Tag $tag no longer names the release commit. $tagged"
fi
check_release_context "$release_commit" "$tagged"

# Push the checked commit and tag object by ID, so a later move of local main
# or the tag does not change what is published.
git push --atomic --no-follow-tags -- "$destination" "$release_commit:refs/heads/main" "$tag_object:refs/tags/$tag"

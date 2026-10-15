# Compaction Evaluation

`bun run eval:compact` measures how well a chosen model works as the Magic Compact summarizer. It runs the real packed `arkive-compact` package inside an OpenCode server you already have running and authenticated, against a frozen synthetic corpus, and records what happened: whether `/magic-compact` delivered a valid successor, what each attempt cost, and, when graded, how many gold facts each summary kept.

The results describe one model, on one host setup, on three synthetic cases. They are not a general accuracy rate for the model and are not comparable with the retention evaluation in `docs/RetentionEvaluation.md`.

## What a run does

For each case and repeat, the eval:

1. creates a fresh source session holding the case's two synthetic turns (the target turn, then an unrelated turn protected by `N = 1`);
2. runs `/magic-compact 1` once, with no retry;
3. checks the successor mechanically against the source's native records as read before the command ran (see Mechanical checks);
4. removes the sessions it created when their recorded state is unchanged.

All sessions live in a private temporary project under the system temp directory, marked with the run's nonce. That project's local config loads the packed package through an eval wrapper (see Isolation), sets the summarizer to `--model`, and disables every MCP server the global config defines. Your global config, your other projects and your existing sessions are not read for content or changed.

Each attempt is one paid summarizer request (`/magic-compact` makes one request per target turn, and each case has one target). The corpus has no tool calls, so task-aware retention has no candidates: no TypeSafe key is needed and no classifier request is made. `check` proves this offline with the production candidate finders for both hosts.

## Requirements

- Bun, and npm on `PATH` (used to pack the subject and install the wrapper).
- A running OpenCode server on this machine's loopback interface, with the model's provider already authenticated in that server. Supported servers are OpenCode 1.18.30 and 1.19.0, and OpenCode 2.0.28 (see Host versions).
- If the server uses Basic Auth, export `OPENCODE_SERVER_USERNAME` (default `opencode`) and `OPENCODE_SERVER_PASSWORD`. These are the only credentials the eval reads. Provider credentials stay inside the server; the eval never reads or copies an auth store.

The server must be local because it installs the packed artifact from a path on this file system and writes the run channel into the output directory. A remote URL is refused.

## Commands

```bash
# Offline: validate the frozen corpus and its plans. No server, no network.
bun run eval:compact check

# --out must not exist yet, but its parent must.
mkdir -p runs

# Preflight only: identify the server, load the plugin in the eval project,
# check the model and variant exist. Creates no session, sends no model request,
# but does create its own --out directory, so it gets a name of its own.
bun run eval:compact run --server http://127.0.0.1:4096 \
  --model anthropic/claude-haiku-5-5#low --out runs/haiku-low-preflight --dry-run

# A run: every case twice, no grading.
bun run eval:compact run --server http://127.0.0.1:4096 \
  --model anthropic/claude-haiku-5-5#low --out runs/haiku-low --repeat 2

# One case, graded by an explicit judge model (one extra paid request per delivered summary).
bun run eval:compact run --server http://127.0.0.1:4096 \
  --model anthropic/claude-haiku-5-5#low --out runs/haiku-low-b --case B \
  --judge openai/gpt-6-luna-fast#low

# Grade a finished run later, with a judge or with your own grades.
bun run eval:compact grade runs/haiku-low --server http://127.0.0.1:4096 --judge openai/gpt-6-luna-fast#low
bun run eval:compact grade runs/haiku-low --import my-grades.json

# Offline: summarize and compare saved runs.
bun run eval:compact report runs/haiku-low runs/luna-low
bun run eval:compact report runs/haiku-low --json

# Reconcile an interrupted run and remove what it still owns.
bun run eval:compact cleanup runs/haiku-low --server http://127.0.0.1:4096
```

`--model` and `--judge` take `provider/model` or `provider/model#variant`, using the server's own provider and model IDs. If the server does not list the model, or the model has no such variant, the run refuses before creating anything. There is no fallback model and no default judge.

`--out` must name a directory that does not exist yet; runs never overwrite each other. `--repeat` runs every selected case that many times (1 to 20). A failed attempt is recorded once and is not retried.

Other options: `--cache DIR` overrides the artifact cache, `--subject TGZ` evaluates an already packed tarball instead of packing this checkout, and `--assert-host-version VERSION` runs on an unqualified host version (recorded as operator-asserted).

## Isolation

The eval loads the subject through a thin eval-only wrapper whose package depends on the unmodified packed `arkive-compact` tarball under an alias. The wrapper passes every product hook through unchanged and adds, for one run: a load witness, a record of each Magic Compact command's exact result text, content-free usage numbers, and removal of a successor's omission cache and stats through the subject's own storage functions (see Cleanup and failures). On OpenCode 1 it also sets the summarizer agent's variant to exactly the requested one in that instance's in-memory config, because OpenCode 1 merges agent config field by field: without this, a variant from your global `magic-compact-summarizer` profile would apply to a `--model` given without one. No config file changes. These records go to `OUT/channel/`. The wrapper is test instrumentation, not a product API.

- **OpenCode 1** dedupes plugins by package name, so the wrapper is named `arkive-compact` and is listed as `arkive-compact@file:<cache>/v1/wrapper.tgz` in the eval project only. A globally configured `arkive-compact` stays active everywhere else.
- **OpenCode 2** activates only the first plugin with a given ID, so the eval project negates `arkive-compact` and loads the wrapper directory under the ID `arkive-compact-eval`.

Before any session is created, preflight requires exactly one Arkive plugin in the eval project, one wrapper initialization with this run's nonce and artifact, zero differences between the installed subject files and the packed tarball, the three product commands, the summarizer resolving to `--model` with exactly its variant, or none when `--model` names none (both as configured and as the plugin itself reads it), the judge agent resolving to `--judge` the same way, and every inherited MCP server reported disabled by the host. Any failure refuses the run.

Other global plugins, such as auth plugins, and global instructions still load in the eval project, because the eval measures your actual host setup. `preflight.json` records the effective plugin specs, SHA-256 fingerprints of instruction paths, and the provider's model descriptor, never their contents or any secret. Two people running the same model can get different results if their host setups differ.

## What stays behind

- **Run directory** (`--out`): `run.json` (server, corpus, request), `preflight.json` (artifact hashes, effective setup, model descriptor), `results.json`, `attempts/*.json`, `grades/`, `ledger.jsonl` and `channel/`. Capsule text is the synthetic corpus summarized by your model.
- **Artifact cache** (`${XDG_CACHE_HOME:-~/.cache}/arkive-compact-eval/<key>`): one immutable entry per subject tarball and wrapper version, about 14,000 files including the installed subject. Repeated runs of the same artifact reuse it. An entry is built in a staging directory and renamed into place, and it is checked against its own file record before every use. An entry that fails the check is refused and left untouched; delete it to rebuild.
- **Host package cache**: OpenCode 1 installs the named wrapper tarball into its own package cache (`~/.cache/opencode/packages/arkive-compact@file:...`), once per artifact. OpenCode 1 may also install `@opencode-ai/plugin` into its global config directory as part of its normal startup.
- **Host project metadata**: OpenCode keeps its own records of the temporary project (project and snapshot data). The eval does not sweep host databases.

The temporary tree is one root directory (`arkive-compaction-eval-XXXXXX` under the system temp directory, or `arkive-compaction-eval-judge-XXXXXX` for a later `grade`) holding the `inspect`, `project` and `judge` locations. The root and each location carry a `.arkive-compaction-eval` marker naming the run's nonce, its output directory and the root. When nothing the run owns remains, the eval disposes its locations (`instance.dispose` on OpenCode 1, `debug.location.evict` on OpenCode 2, each bound to the eval's own directory), deletes them and then removes the root, which must by then be empty. It never calls a global dispose, reload or upgrade.

## Cleanup and failures

Every session the eval creates is written to the ledger before the request, with its title and, when the host allows choosing one, its planned ID, and again with its returned ID. A session is removed only when a fresh read shows it at its own eval location with exactly the header and history digests recorded for it and no pending work: no running, retrying or unknown status, no child session, and no waiting permission request or question (OpenCode 1), or no activity, queued message, permission request, form or running shell started from it (OpenCode 2). The header digest covers every field that can change the session's behavior, including agent, model, permissions, metadata, sharing, archive and native compaction state, and on OpenCode 2 the session's active instruction entries; only timing and cost fields are left out. Anything that cannot be read, including an instruction entry list, keeps the session. This check is fresh, not atomic: no public API lets the eval exclude another writer acting between the read and the delete.

A recorded state is only ever one the eval knows it produced. A session it created and never prompted must still be exactly as created, with no instruction entries (OpenCode 2.0.28 creates none). A source is recorded from the same read that showed it equal to the corpus and to the header the eval itself created, never to the header that read returned: on OpenCode 1 the header the creation returned, whose title, agent, model and directory must be exactly the requested ones; on OpenCode 2 the session record the eval imported, with no instruction entries, built from a template session whose returned title, agent, model and location must be exactly the requested ones (OpenCode 2.0.28 records variant `default` when none was requested). A source that did not match, including one whose metadata, model, permissions or title another writer changed before that first read, is kept with nothing recorded, and its attempt sends no summarizer request. A successor is recorded (`successor-verified` in the ledger) only from a read that passed every mechanical check, with at least one check, while its source was unchanged; on OpenCode 2 its instruction entries must equal the source's. A successor that fails, or that another writer changed before the eval first read it, is `delivered-invalid` and is kept with its storage, its source and its location. Its source stays because a later `cleanup` can only verify the successor, and remove its storage, against that source: each `cleanup` re-runs every mechanical check from scratch against the source's recorded state, and removes both only once the successor passes.

Each scratch root is bound to the server process that created it. Preflight records the process ID (`server-witness` in the ledger) before any session exists: on OpenCode 2 from `/api/info`, on OpenCode 1 from the eval wrapper's witness read live through the eval project's config. A session counts as gone, and a location is disposed, only while that same process answers; a PID read earlier from the channel is not proof.

A successor's cache and stats are removed in two steps. First the wrapper, running a control command inside the successor, confirms the successor is this run's: the request names the run, the successor and its source are in the eval project, the source's title carries the run nonce and the successor's lineage names that source. This writes nothing. Then the eval checks the successor like any other session and deletes it, and the wrapper removes the storage only when the host reports that deletion. A successor the eval keeps keeps its storage. When a deletion's storage removal is not confirmed, the attempt records the storage paths left behind.

The eval disposes nothing while the ledger leaves anything open: a session not yet removed, a session intent whose creation was never confirmed or ruled out, or an attempt whose command started without a recorded outcome, including after a crash that wrote nothing more. It also keeps the tree when any root or location fails its marker check (a missing or different marker, a symlink, a path outside the root), when the host still lists any session at an eval location (for example a candidate the product kept after a failure), or when the root holds files the eval did not create. Sessions it cannot account for are reported, never deleted.

If a command's HTTP response is lost, the attempt waits up to 15 minutes for the command's own outcome on the run channel, then records `outcome-unknown` and starts no replacement. Ctrl-C stops new attempts, lets the current one finish and clean up; a second Ctrl-C exits at once. In both cases, run `cleanup` later. It settles attempts from outcomes that reached the channel late, registering a delivered successor first; an attempt with no outcome keeps all of its sessions, including its source, because on OpenCode 2 the summarizer runs in a separate session and an idle source does not show it has finished. It reads a session whose planned ID was recorded, and looks for a session with the recorded title at the eval location when the ID was not. A successor known only from a late outcome is removed only when its source is unchanged since the run recorded it and the successor passes every mechanical check against it; otherwise it is kept. `cleanup` exits non-zero while anything stays. That includes a scratch root that still exists but no longer proves it is the run's (its marker is missing or changed, or something else sits at its path): `cleanup` deletes nothing there, names the root and the reason in its result, and keeps reporting it. Only a root that is gone counts as cleaned.

`grade` and `cleanup` work only against the server the run used. `--server` must be exactly the URL in `run.json` (`server.url`); another URL is refused before anything is sent, even another name for the same address such as `localhost` for `127.0.0.1`. `cleanup` then compares the live process ID with the one each root recorded. When it differs, for example after the server restarted or when another server took over the port, `cleanup` keeps every session and location of that root, marks nothing absent and exits non-zero. It never reads a host database directly and has no override: restart the original server process if it is still available, or review the kept sessions by hand. A root whose process was never recorded is disposed only when the run created no session in it, which happens when preflight stopped before identifying the server. A `grade` run on OpenCode 1 records no process ID for its judge root, because no eval wrapper loads at the judge location, so a crashed OpenCode 1 `grade` leaves its judge root for manual review.

Attempt statuses are `delivered` (every invariant passed), `delivered-invalid` (a successor exists but an invariant failed), `noop`, `failed` (with a category: capability, provider, schema, coverage, savings, setup or other, plus the exact command text), `outcome-unknown`, `preflight-skipped` (for example an image case on a model without image input) and `setup-failed`.

## Mechanical checks

A delivered successor must be exactly the source as `/magic-compact 1` defines it, compared field by field on the hosts' native records. Every user message, with every native field, file part, file name and metadata, and the protected second turn, with its parts, finish and metadata, must equal the source's. The summarized assistant message keeps its own fields and holds only the v3 capsule (on OpenCode 2 its metadata gains only the capsule's `magicCompact` entry). The only additions allowed are Magic Compact's own: the boundary text leading the next user message (a boundary record before it on OpenCode 2) and, on OpenCode 1, the ignored stats notice after the history. The only differences allowed are fresh message and part IDs, which must be unique and absent from the source, the links between them, the session they belong to, the boundary's creation time, and the capsule and stats text. The successor's header must carry the source's title, agent, model, permissions and location, its metadata plus the lineage entry, and nothing else. A record added, dropped or changed anywhere else makes the attempt `delivered-invalid`.

## Scoring

The rubric is `packages/opencode-plugin/eval/compaction/corpus/v1/scoring.md`, frozen with the gold facts before any model was run. Each case has 8 text facts, 4 of them critical; cases B and C also have 4 visual facts. A summary scores one point per fact it states correctly, minus two per distinct false assertion in that dimension, floored at zero. It is usable when every critical fact is present and no false assertion is critical.

Under the rubric, an attempt that settled without a valid summary (`failed`, `noop`, `delivered-invalid`, or `preflight-skipped` because the model cannot take the case's image) scores 0 in every dimension and is not usable. An attempt whose outcome is unknown, or that never reached the model (`outcome-unknown`, `setup-failed`), is not scored at all and is counted separately. Per-case means and usable rates are taken over graded and zero-scored attempts. A delivered summary without an accepted grade is never counted as 0: while any summary of a case is ungraded, that case reports `PARTIAL` and no mean. For example, a case with one summary graded 8/8 and one provider failure has a mean of 4/8 and is usable in 1 of 2 attempts. When no attempt delivered a summary, every settled attempt scores 0 without a grader.

Without accepted grades, the report says `Semantic: UNGRADED`. Mechanical success says nothing about summary quality. Partial grades are reported as `PARTIAL`.

Grades are always per-fact judgments; the eval computes every score itself. A grades file (`arkive-compaction-eval-grades/1`) is bound to one run's nonce, the corpus and rubric hashes, and each attempt's case and capsule SHA-256. Each item judges every gold fact of its case exactly once (`present` plus an exact quote from the summary as `evidence`) and lists false assertions with their dimension, criticality, claim and quoted evidence. A file that names another run, case or output, repeats an attempt, or quotes evidence that is not in the summary is rejected as a whole. To grade by hand, copy an existing judge grades file from `grades/`, set `grader` to `{"kind": "manual", "label": "<your name>"}`, edit the judgments, and import it.

`grade` and `report` first check the saved run itself and refuse it as a whole when it does not match this checkout's corpus and rubric, or when an attempt record names another run's nonce or another requested model, names a source or successor the run's ledger did not create for that attempt, names a case or repeat the run did not request, has an unknown status, or holds a summary whose SHA-256 differs from its recorded digest. An intact attempt file copied from another run therefore fails before any grade, report or judge request. The digest is recomputed from the saved summary, so an edited summary cannot keep an earlier grade.

A judge model receives a blind packet: the case's frozen source text (with the image described by its gold facts), the gold facts and the summary, with no model, run or cost information. It runs in its own temporary location with every tool denied, and answers in plain text on both hosts: the prompt asks for one JSON document matching a JSON Schema derived from the grade file's judgment shape, and the answer is parsed strictly with no repair. A malformed, missing or refused answer is recorded as a judge error and leaves that summary ungraded; it is never scored as zero. Judge scores are labeled as automatic estimates from one model, separately from manual grades.

A judge session is removed only while it holds exactly what the eval put there. On OpenCode 1 that is the prompt, sent under a user message ID the eval chose, and the finished reply the host returned for it; a judge session holding anything else is kept, and its answer is a judge error because the judge saw more than the blind packet. On OpenCode 2, `session.generate` writes no history, so the session must still be empty and unchanged since its creation.

## Host versions

The eval identifies the server from JSON, not from an HTTP 200: OpenCode 1 must answer `/global/health` with a `1.x` version and OpenCode 2 must answer `/api/info` with a `2.x` version and process ID. It records the canonical URL, the version the running server reports (not the binary on `PATH`) and, on OpenCode 2, the process ID.

Qualified versions are OpenCode 1.18.30, 1.19.0 and 2.0.28. Their plugin loading, wrapper route, MCP disabling, session lifecycle and location disposal were checked in scripted containers. Any other version is refused, because a different loader or session API could load the wrong plugin or misjudge ownership. `--assert-host-version` lets you run anyway; the run is then recorded as operator-asserted.

Known limits:

- **OpenCode 1 judge format.** OpenCode 1.18.30 cannot return the history of a session that holds a JSON-schema prompt, so the judge never uses the host's structured response format; it answers in plain text, which keeps its history readable for the ownership check.
- **MCP on OpenCode 2.** OpenCode 2.0.28 ignores a partial `{disabled: true}` override, so the eval project replaces each inherited server name with a complete disabled stub. The stub copies nothing from the inherited definition.
- Only loopback servers are supported.

## Corpus versions

`corpus/v1` (`magic-compact-synthetic/1`) holds three cases, promoted from the 2026-10-15 summarizer comparison. A is a text-only rate-limiter fix, B is a timestamp fix plus a 512x512 red-left, blue-right PNG, and C is a CI Node upgrade plus a two-row status table PNG. Case B's image was used in earlier diagnostics and is not held out. `manifest.json` binds every corpus file by SHA-256, and `provenance/comparison-manifest.json` binds the texts, gold facts and image bytes to the original frozen fixtures. Corpus files are data; the eval never imports or executes them, and it refuses unknown formats, extra fields and hash mismatches.

To add or change cases, create a new corpus version directory with its own manifest and corpus ID. Never edit a frozen version or regrade old runs against changed gold facts: a run's grades are only accepted against the corpus and rubric hashes it ran with.

## Maintenance

`packages/opencode-plugin/test/compaction-eval.test.ts` owns the offline invariants: corpus integrity and its retention-free plans, server identity, model selectors, preflight refusals, the summarizer variant override, artifact cache reuse and tamper refusal, the mechanical checks against successors the production command produced and against native successors captured in qualification containers (`eval/compaction/testdata/`, recapture them when a qualified host version changes), ownership-checked removal and successor storage removal, verified-only removal baselines (an end-to-end OpenCode 2 attempt with the production command, and source headers checked against what each driver created), server process binding, both host drivers' pending checks, instruction entries and judge ownership, ledger reconciliation, scratch markers, saved-run validation, and grade binding and scoring. Re-qualify in containers on every supported host when the wrapper, the host drivers or the qualified version list changes.

# Daily Health + Synthesis — Orchestrator Prompt

You are running the FastTomo daily health check + synthesis task. This prompt
is fed to `claude -p` non-interactively by cron. Read it fully, execute every
phase in order, and **never skip the failure-handling step**.

This run does **not** spawn the SOTA research streams. Each of the 4
researchers runs in its own scheduled job (`run_research_*.sh`) on a 4-day
rotation. The speedups audit runs in its own job (`run_speedups.sh`) every
3rd day. This orchestrator only:

1. Health-checks the codebase (build, tests, lint, sanity, link integrity)
2. Synthesizes the **most recent** research + speedups artefacts already on
   disk into today's `summary.md` + `plan.md`
3. Files trivial auto-fixes / risky issues / plan stub on GitHub
4. Opens a fix PR if there are auto-fix commits

---

## Operating Constraints

- Working directory: `/home/cedlim/FastTomo` (verify with `pwd` first).
- Conda env: `FastTomo` — pinned to `cuda-nvcc=13.0`, `cuda-cudart-dev=13.0`,
  `gcc 13` per the README's "Supported CUDA toolkit" callout. Use the env's
  python directly: `/home/cedlim/miniforge3/envs/FastTomo/bin/python` and
  `/home/cedlim/miniforge3/envs/FastTomo/bin/pip` — do not rely on
  `conda activate` inside cron.
- **All run artefacts go to `fasttomo-wiki/wiki/daily/<RUN_DATE>/`.** Do not
  write under `.claude/` — the harness blocks writes there.
- The wiki directory is gitignored in the main repo and lives only on this
  machine. The daily run does **not** push research reports to GitHub.
- Code-fix branches (`daily/<DATE>-fixes`) **are** pushed to GitHub and get a PR.
- Branch policy: never push to `main`/`master`. Push only `daily/*` branches. Do
  **not** include `Co-Authored-By` trailers on any commit (per project policy).
- Permissions: Bash actions are gated by `.claude/settings.local.json` allow/deny
  lists. If a command is denied, log it under `health-risky.md` and continue —
  do not retry with workarounds.
- **Never re-read a research or speedups file that you have already opened
  in this session.** Each Read of `research-*.md` or `speedups.md` is
  expensive. Read each at most once during synthesis; carry the content
  forward in working memory rather than re-Reading it for ledger or plan
  composition.

---

## Phase 0 — Preflight

```bash
cd /home/cedlim/FastTomo
RUN_DATE=$(date -u +%Y-%m-%d)
REPORT_DIR="fasttomo-wiki/wiki/daily/$RUN_DATE"
mkdir -p "$REPORT_DIR/health-raw"
```

Checks:

1. `git status --porcelain` — must be empty for the **main repo**. If not, write
   `$REPORT_DIR/aborted.md` explaining the dirty tree, exit non-zero. (The wiki
   dir being modified is fine — that's expected.)
2. `git fetch origin && git checkout -B daily/$RUN_DATE-fixes origin/main` —
   fresh branch off remote main.
3. `which gh && gh auth status` — must succeed; if not, note it but continue
   (only the fix-PR delivery is blocked).
4. `nvidia-smi` once and capture exit code — used later to decide whether to
   run CUDA tests.

**Env activation:** the FastTomo env's `bin/` must be on PATH so
scikit-build-core's spawned cmake finds `nvcc` via `check_language(CUDA)`.
Cron does not source `~/.bashrc`, so export it explicitly:

```bash
export PATH="/home/cedlim/miniforge3/envs/FastTomo/bin:$PATH"
which nvcc  # verify; expect /home/cedlim/miniforge3/envs/FastTomo/bin/nvcc
```

If `which nvcc` is empty or points elsewhere, write `$REPORT_DIR/aborted.md`
and exit — the env may have drifted (e.g. cuda-nvcc upgraded past 13.0 or gcc
upgraded past 13). The fix is to repin per the README's CUDA section.

5. **`ruff` availability:** `/home/cedlim/miniforge3/envs/FastTomo/bin/ruff
   --version` must succeed. If missing, log under `health-risky.md`
   (one-liner: "ruff not installed in FastTomo env; lint auto-fix skipped this
   run; resolve with `conda install -n FastTomo -c conda-forge ruff`") and
   skip the lint step in Phase 2 — do **not** abort.

If preflight fails (steps 1–4), write `$REPORT_DIR/aborted.md` and exit
non-zero so cron mails you. Step 5 is non-blocking degradation.

---

## Phase 1 — Synthesis (orchestrator, no subagent)

The 4 SOTA research streams + speedups audit live in
`fasttomo-wiki/wiki/daily/*/research-*.md` and `*/speedups.md`. They are
written by the per-stream cron jobs and may be 1–4 days old depending on the
rotation. Synthesis uses the **most recent** file for each stream.

### Locate latest artefacts

For each stream, find the most recent file across all daily folders:

```bash
latest_for() {
  ls -1 fasttomo-wiki/wiki/daily/*/$1 2>/dev/null | sort -r | head -1
}
CUDA_FILE=$(latest_for research-cuda.md)
CLASS_FILE=$(latest_for research-classical.md)
DL_FILE=$(latest_for research-dl.md)
CRYO_FILE=$(latest_for research-cryoet.md)
SPEED_FILE=$(latest_for speedups.md)
```

Capture each file's parent directory (the run date) for citation in
`summary.md`. If a stream has **no** file on disk yet (e.g. fresh install),
note that in summary.md as `_no run yet_` and continue.

### Read each file exactly once

Read the 5 files in parallel (single tool block, 5 Read calls). **Do not Read
any of them again later in this session.** Hold the content in working
memory through the rest of synthesis, plan generation, and GH issue body
composition.

### Write `summary.md`

Write `$REPORT_DIR/summary.md` with frontmatter:

```markdown
---
type: daily-report
date: <YYYY-MM-DD>
status: complete
top_pick: <one-line title>
sources:
  cuda: <YYYY-MM-DD>      # date of the file used (may be older than RUN_DATE)
  classical: <YYYY-MM-DD>
  dl: <YYYY-MM-DD>
  cryoet: <YYYY-MM-DD>
  speedups: <YYYY-MM-DD>
---

# Daily Synthesis — <YYYY-MM-DD>

## Top 3 Cross-Cutting Implementation Opportunities

Each bullet must be prefixed with a category tag in bold square brackets, e.g. `**[AD-innovation]** <title> — ...`. Valid tags: `[AD-innovation]`, `[new-baseline]`, `[SIRT/FBP-tech-speedup]`, `[SIRT/FBP-math-change]`, `[frozen-baseline]`.

1. **[<tag>]** **<title>** — <one paragraph: what, why for FastTomo, source streams, files affected, effort × impact>
2. ...
3. ...

## Consensus Signals
Anything appearing in ≥2 streams (signal of field-wide momentum):
- ...

## Per-Stream Highlights
- **CUDA** (sourced <YYYY-MM-DD>): <one-line> — [details](../<YYYY-MM-DD>/research-cuda.md)
- **Classical** (sourced <YYYY-MM-DD>): <one-line> — [details](../<YYYY-MM-DD>/research-classical.md)
- **DL** (sourced <YYYY-MM-DD>): <one-line> — [details](../<YYYY-MM-DD>/research-dl.md)
- **Cryo-ET** (sourced <YYYY-MM-DD>): <one-line> — [details](../<YYYY-MM-DD>/research-cryoet.md)

## Glaring Speedups
<If speedups source date is more than 4 days old, note "_stale: last run <YYYY-MM-DD>_" and skip the section.>
<If speedups.md says "None this run.", render exactly: "None this run.">
<Otherwise, one bullet per finding from speedups.md:>
- **<title>** — `<file:lines>` — <symptom one-liner>. Impact: <S/M/L>, confidence: <high/med>. [details](../<YYYY-MM-DD>/speedups.md)

## Health Status
See [health.md](health.md) — <one-line summary: N fixes committed, M risky issues reported, K tests green>.
```

**Speedups never enter the Top 3 ranking.** They are a separate internal
audit, kept distinct from SOTA-driven implementation opportunities by
design.

Append a one-line entry to `fasttomo-wiki/wiki/daily/index.md` between the
`<!-- INDEX-START -->` and `<!-- INDEX-END -->` markers (newest at top):

```
- YYYY-MM-DD — top pick: <title> — [summary](YYYY-MM-DD/summary.md) · health: <green|N fixes|risky>
```

Append to `fasttomo-wiki/wiki/log.md` per SCHEMA.md log format:

```
## [YYYY-MM-DD] daily | <top pick title>
<one-line summary covering both research and health>.
```

### Wiki ingestion (selective)

For any *durable* finding (a real paper worth keeping, an algorithm not yet in
the wiki), follow `fasttomo-wiki/SCHEMA.md` § Ingest to add a paper or algorithm
page. Skip the wiki for ephemeral blog posts, GitHub issues, or news. Rule of
thumb: if it's not worth bookmarking 12 months from now, don't ingest.

### No ledger update here

The papers-seen ledger is owned by the per-researcher jobs. This orchestrator
**does not write to `papers-seen.md`** — the per-researcher `.sh` updates the
ledger from its subagent's return on the day it ran. Synthesis cites papers
by ID without touching the ledger.

---

## Phase 2 — Health Check (sequential collection)

Run these in order. Stdout/stderr go to the wiki via `tee`:

```bash
PYTHON=/home/cedlim/miniforge3/envs/FastTomo/bin/python
PIP=/home/cedlim/miniforge3/envs/FastTomo/bin/pip
RAW="$REPORT_DIR/health-raw"
```

1. **Build:** `$PIP install -e . --no-build-isolation 2>&1 | tee $RAW/build.log`
2. **CPU tests:** `$PYTHON -m pytest tests/test_cpu.py -v --tb=short 2>&1 | tee $RAW/test-cpu.log`
3. **CUDA tests** (only if `nvidia-smi` exit 0 in preflight):
   `$PYTHON -m pytest tests/test_cuda.py tests/test_projector_cache.py -v --tb=short 2>&1 | tee $RAW/test-cuda.log`
4. **Lint:** `ruff check fasttomo/ tests/ 2>&1 | tee $RAW/ruff.log` (advisory; capture count)
5. **Wiki link integrity:** scan `fasttomo-wiki/wiki/**/*.md` for relative
   markdown links and verify each target exists. Write findings to
   `$RAW/wiki-links.log`.
6. **Numerical sanity:** run a 16³ FP→BP roundtrip on CPU asserting finite,
   non-zero output. Inline:
   ```python
   import torch, fasttomo
   v = torch.rand(16, 16, 16, dtype=torch.float32)
   ang = torch.linspace(0, 180, 30, dtype=torch.float32)
   s = fasttomo.forward_project(v, ang, N_det=24)
   r = fasttomo.back_project(s, ang, 16)
   assert torch.isfinite(r).all() and r.abs().sum() > 0, "numerical sanity failed"
   print("ok")
   ```
   Log to `$RAW/sanity.log`.
7. **TODO drift:** `grep -rn "TODO\|FIXME\|XXX" csrc/ fasttomo/ | wc -l` →
   `$RAW/todos.log`. Compare to yesterday's count if the file exists; flag
   in `health.md` if up >10%.

### Triage

For each failure, classify as:

| Class | Examples | Action |
|---|---|---|
| **Trivial-auto** | ruff-fixable lint, formatter, broken wiki link, doc typo, missing `__all__`, known tolerance flake (per `AGENTS.md`: `test_genfire_reconstruction_quality`) | Phase 3 |
| **Substantive** | single test failing with clear cause, missing functionality, refactor opportunity | Phase 4 (plan) |
| **Risky** | build failure, segfault, kernel numeric divergence, CUDA OOM, anything in `csrc/cuda/` | Phase 5 (GH issue) |

Substantive findings are **not auto-fixed**.

---

## Phase 3 — Apply Trivial Auto-Fixes (PARALLEL when file-disjoint)

Group trivial-auto issues by **file-disjoint sets**. If you have ≥2 disjoint
groups, dispatch **parallel `programmer` subagents** in git worktrees, one per
group:

```bash
WORKTREE_BASE="/tmp/fasttomo-daily-$RUN_DATE"
git worktree add "$WORKTREE_BASE/group-1" daily/$RUN_DATE-fixes
# ... one per group
```

Each programmer subagent receives:
- The exact file(s) it owns (and may not edit any other file)
- The list of issues to fix in those files (only trivial-auto; never
  substantive)
- Instruction to run `ruff check --fix` and the relevant test, and report back
- Instruction to commit on its worktree branch with no `Co-Authored-By` trailer

After all programmers return, merge worktrees back into `daily/$RUN_DATE-fixes`
sequentially with `git merge --no-ff`. Conflicts should not occur (groups are
file-disjoint by construction); if one does, **abort**, append the conflict to
`$REPORT_DIR/health-risky.md`, and skip the offending merge.

Clean up worktrees: `git worktree remove "$WORKTREE_BASE/group-N"`.

If you have only one group, just dispatch a single programmer (no worktree
needed) on the current branch.

If there are no trivial-auto findings, skip Phase 3 entirely.

---

## Phase 4 — Plan Generation (orchestrator, no subagent)

Write `$REPORT_DIR/plan.md` for opportunity #1 from Phase 1's Top-3.
**Do not** generate plans for #2 and #3 — they stay as one-paragraph blurbs
in `summary.md`. Stale deferred plans are worse than no plan.

If Top-1 from synthesis is tagged `frozen-baseline` or `SIRT/FBP-math-change`, drop it and select the next-highest valid candidate. Plan only targets `AD-innovation`, `new-baseline`, or `SIRT/FBP-tech-speedup`.

You already have the relevant research-file content in working memory from
Phase 1 — **do not re-Read those files.** Cite them by path, but compose
plan.md from the in-memory synthesis output.

Format:

```markdown
---
type: daily-plan
date: <YYYY-MM-DD>
top_pick: <one-line title>
streams: [<classical, cryo-et, ...>]
status: draft
---

# Plan: <Top Pick Title>

## Context
Why this change is being made — pull from `summary.md` Top-3 opportunity #1
and the matching `## Consensus Signals` line(s). Cite the research streams.
1–2 paragraphs.

## Approach
One paragraph describing the recommended implementation strategy.

## Critical Files to Modify
- `fasttomo/algorithms.py:LINES` — what changes
- ...

## Existing Code to Reuse
- `fasttomo/autograd.py:<symbol>` — already provides X
- ...

## Verification
- Run `pytest tests/...`
- 16³ FP→BP roundtrip ...
- Compare grad-mode result to grid-mode on synthetic dataset

## Reference Papers
- arxiv:XXXX.XXXXX — <title>
- ...

## Open Questions for User Review
- ...
```

Constraints:
- **Plan targets a SOTA finding only.** Speedups never become `plan.md`.
- Keep under ~150 lines.
- Every `Critical Files` entry must reference a real file path (verify with
  `ls` before writing) and, where possible, a real line range.

This file is the artefact the user opens next session with Claude in `/plan`
mode to refine and execute together. It is **not** implementation.

---

## Phase 5 — GH Issue Filing

File two classes of GitHub issues. Both classes use `gh issue create` with a
dedup-by-title-prefix check first.

### A. Plan stub issue (one per run, only if a plan was written)

Search existing open issues:

```bash
EXISTING=$(gh issue list --label daily-plan --state open \
  --search "<top pick title prefix>" --json number,title --jq '.[0].number // empty')
```

If `$EXISTING` is non-empty, **comment** on it instead of creating a new issue:

```bash
gh issue comment "$EXISTING" --body "Re-surfaced in daily run $RUN_DATE. Local plan refresh: \`fasttomo-wiki/wiki/daily/$RUN_DATE/plan.md\`"
```

Otherwise, create a new issue:

```bash
gh issue create \
  --title "daily($RUN_DATE): plan — <top pick>" \
  --label "daily-plan,enhancement" \
  --body "$(cat <<EOF
## Top pick — <title>

<2-3 sentence synthesis pulled from summary.md Top-3 #1>

**Local plan:** \`fasttomo-wiki/wiki/daily/$RUN_DATE/plan.md\`
**Streams:** <classical, cryo-et, ...>
**Effort × Impact:** <S × high>

## Files affected
- \`fasttomo/algorithms.py:LINES\`
- ...

## Verification sketch
- ...

## Reference papers
- arxiv:XXXX.XXXXX — <title>

To pick this up: review the local plan with Claude in \`/plan\` mode, refine, then
implement.
EOF
)"
```

### B. Risky-issue tracker (one per risky finding)

For each entry in `health-risky.md`, do the same dedup-then-create flow:

```bash
gh issue list --label daily-risky --state open --search "<issue title prefix>" \
  --json number,title --jq '.[0].number // empty'
```

If matched → comment with re-occurrence date. Otherwise:

```bash
gh issue create \
  --title "daily($RUN_DATE): risky — <issue title>" \
  --label "daily-risky,health" \
  --body "<the corresponding health-risky.md block>"
```

### Write `health-risky.md`

For each risky finding, write/append:

```markdown
# Risky Issues Detected — <YYYY-MM-DD>

## <Issue title>
- **Type:** build / test / numeric / kernel
- **Symptom:** <one line>
- **Stack trace / output:**
  ```
  <relevant excerpt>
  ```
- **Recommended next step:** <one paragraph>
- **GH issue:** #<number> (filed in Phase 5) — or "comment on #<number>" if dedup hit.
```

**Do not attempt risky fixes.**

---

## Phase 6 — QA on Fixes (single qa subagent — only if Phase 3 made commits)

Skip this phase entirely if Phase 3 produced no commits (i.e.
`git log --oneline origin/main..daily/$RUN_DATE-fixes` is empty). Plan-only
runs do not need QA.

Otherwise, dispatch one `qa` subagent on the merged `daily/$RUN_DATE-fixes`
branch. Brief:

> Re-run pytest CPU tests, CUDA tests if available, and ruff. Confirm green or
> identify any regressions introduced by the trivial auto-fixes. Verify no
> commits include a `Co-Authored-By` trailer (project policy). Write findings
> to `$REPORT_DIR/health.md` using the standard QA report format. Verdict must
> be `Ship`, `Needs Revision`, or `Blocked`.

If QA returns `Needs Revision` or `Blocked`, **revert the bad commits** with
`git reset --hard <good-sha>`, treat the issues as risky, append to
`health-risky.md`, and re-run Phase 5 issue filing for the new risky entries.

---

## Phase 7 — Delivery

```bash
git log --oneline origin/main..daily/$RUN_DATE-fixes
```

If there are trivial-auto commits beyond `origin/main`:

```bash
git push -u origin daily/$RUN_DATE-fixes
gh pr create --title "daily($RUN_DATE): trivial auto-fixes" --body "$(cat <<EOF
## Summary
- Health: <one-line>
- Research top pick: <title> — see plan stub issue #<plan_issue_number>

## Trivial auto-fixes applied
<bullet list from health.md>

## Risky issues filed
<bullet list — each line: "issue #<n>: <title>" — or "none">

## Reports (local — fasttomo-wiki/ is gitignored)
- summary:  \`fasttomo-wiki/wiki/daily/$RUN_DATE/summary.md\`
- plan:     \`fasttomo-wiki/wiki/daily/$RUN_DATE/plan.md\`
- health:   \`fasttomo-wiki/wiki/daily/$RUN_DATE/health.md\`

## Test plan
- [ ] Review the diff
- [ ] Confirm no Co-Authored-By trailers
- [ ] Open plan stub issue #<plan_issue_number> and triage with Claude in \`/plan\` mode
EOF
)"
```

If there are no fix commits, **skip the PR** — synthesis, plan, and GH issues
are the deliverables for the run. Do not push the (empty) branch.

Final stdout (so cron's log shows where to look):

```
Daily run complete.
- summary: fasttomo-wiki/wiki/daily/$RUN_DATE/summary.md
- plan:    fasttomo-wiki/wiki/daily/$RUN_DATE/plan.md
- plan issue: <URL or "(no top pick)">
- risky issues: <space-separated URLs or "none">
- PR: <URL or "no PR (plan-only run)">
```

---

## Phase 8 — Failure Handling

Wrap the whole orchestration in error handling. On any uncaught exception:

1. Write `$REPORT_DIR/aborted.md` with frontmatter `status: aborted` and:
   - Phase that failed
   - Error / stack trace
   - Last successful step
2. Push whatever fix-branch state exists:
   `git push -u origin daily/$RUN_DATE-fixes 2>/dev/null || true`
3. Exit non-zero so cron's mail-on-error fires.

---

## Important Reminders

- **Read each research / speedups file at most once.** Carry content in
  working memory through synthesis, plan, and GH issue body composition.
  Re-Reading these files is the dominant cost driver of this job.
- **No `Co-Authored-By` trailers** on any commit — project policy.
- **No push to main/master** — denied by `settings.local.json`.
- **No `gh pr merge`** — denied by `settings.local.json`. User merges manually.
- **No `--no-verify`** on commits — never bypass hooks.
- **No writes under `.claude/`** — harness blocks them. All output goes to
  `fasttomo-wiki/wiki/daily/<RUN_DATE>/`.
- **No substantive auto-fixes.** Only trivial-auto class is auto-applied.
  Substantive code work goes into `plan.md`; risky issues go into GH issues.
- **Plan is for one top pick only.** Do not generate plans for opportunities
  #2 and #3 — they live in `summary.md` as paragraph blurbs.
- **No ledger writes here.** The per-researcher jobs own `papers-seen.md`.
- **Stay within the allowlist** — if a command is denied, log it and continue.

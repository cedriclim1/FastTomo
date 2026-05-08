# Codebase Speedup Audit

Single-stream internal codebase audit for FastTomo. This prompt is fed to
`claude -p` non-interactively by cron. Read it fully and execute every phase
in order.

This is **not** a SOTA scan. The auditor does not consult external papers
and does not touch the papers-seen ledger.

---

## Operating Constraints

- Working directory: `/home/cedlim/FastTomo` (verify with `pwd` first).
- All run artefacts go to `fasttomo-wiki/wiki/daily/<RUN_DATE>/`.
- The wiki directory is gitignored — reports stay local.
- Permissions: gated by `.claude/settings.local.json`. If denied, log under
  `$REPORT_DIR/speedups-aborted.md` and exit non-zero.
- **Do not re-read `speedups.md` after the subagent writes it.** The
  auditor's tool-result return contains the full report.

---

## Phase 0 — Preflight

```bash
cd /home/cedlim/FastTomo
RUN_DATE=$(date -u +%Y-%m-%d)
REPORT_DIR="fasttomo-wiki/wiki/daily/$RUN_DATE"
mkdir -p "$REPORT_DIR"
```

---

## Phase 1 — Spawn Auditor (single subagent)

Spawn **one** `researcher` subagent with the brief below. The brief
explicitly disables web access and redirects the agent to the codebase.

**Output path:** `$REPORT_DIR/speedups.md`

**Brief (copy verbatim, substituting `$REPORT_DIR`):**

> **This is not a SOTA scan.** Do not use WebSearch or WebFetch. Do not
> consult external papers. Audit the FastTomo codebase as it currently
> stands and surface only **glaring** performance wins — things that any
> experienced CUDA/numerical engineer would call out on first read.
>
> Examples of "glaring": a host↔device copy inside a hot loop; a kernel
> that recomputes the same trig table per call when a `__constant__`
> buffer would do; a Python-level N² loop where a vectorised torch op
> exists; a `tex2DLayered` lookup that could be `tex3D` (or vice-versa)
> given the access pattern; missing `__restrict__` on a pointer that
> demonstrably aliases nothing; redundant device sync. Examples of **not**
> glaring: micro-optimisations under 5%, speculative re-architectures,
> anything requiring a benchmark to justify.
>
> Scan: `csrc/cuda/*.cu`, `csrc/cuda/*.cuh`, `csrc/cpu/*.c`,
> `csrc/cpu/*.h`, `csrc/bindings_*.cpp`, `fasttomo/projection.py`,
> `fasttomo/autograd.py`, `fasttomo/algorithms.py`, `fasttomo/utilities.py`,
> `fasttomo/projector.py`. Read each fully — do not skim.
>
> **Output path:** `$REPORT_DIR/speedups.md`
>
> Output format (markdown) — write exactly this structure to the output
> path:
>
> ```
> ---
> type: speedup-audit
> date: <YYYY-MM-DD>
> findings: <integer 0..3>
> ---
>
> # Codebase Speedup Audit — <YYYY-MM-DD>
>
> ## Findings
> <If zero findings: write the literal line "None this run." and stop.>
> <Otherwise, one block per finding using the template below.>
>
> ### <short title>
> - **File:** `path/to/file.ext:LINE_RANGE`
> - **Symptom:** <one line of what is currently happening>
> - **Why it's slow:** <one paragraph, mechanism-level — cite the GPU/CPU
>   feature being misused or the algorithmic redundancy>
> - **Suggested fix:** <one paragraph; reference any existing helper in
>   the codebase that should be reused, with `path:line`>
> - **Estimated impact:** <S/M/L wall-clock impact on a representative
>   FP/BP call, with the back-of-envelope reasoning>
> - **Confidence:** <high / medium> — high only if a similar pattern is
>   already established elsewhere in the codebase
> ```
>
> Hard rules:
> - **Maximum 3 findings.** If you have more than 3, pick the 3 most
>   glaring and drop the rest.
> - **Zero findings is the correct answer on most days.** Do not invent
>   findings to fill the report.
> - Every finding must reference a real `path:line` (verified by
>   `Read`/`Grep`) — no speculative paths.
> - Do not propose changes to test files, benchmarks, build scripts, or
>   wiki content.
> - Do not propose changes that depend on a new dependency or a new
>   sm-arch (FastTomo targets sm_70+).
> - **Scope rule:** speedups in published-baseline internals are out of
>   scope. Skip findings whose only effect is on
>   `csrc/cuda/projection_resire.cu`, `csrc/cuda/genfire.cu`,
>   `csrc/cpu/genfire.c`, or the W-SIRT body in `fasttomo/algorithms.py`
>   (`run_wsirt` and its helpers). **In scope:** tech/CUDA speedups to
>   SIRT, FBP, the AD path (`fasttomo/autograd.py`,
>   `csrc/cuda/projection_3d.cu`, `csrc/cuda/grad_angle.cu`), and shared
>   kernels (`projection_2d`, `common.cuh`). All proposed speedups must be
>   **numerics-identical**; do not propose mathematical additions
>   (regularizers, denoisers, learned priors) to SIRT/FBP under any
>   circumstance — those belong on the AD path and are not "speedups".
> - **No `## Papers Surfaced (this run)` table** — this stream does not
>   feed the ledger.

After the subagent returns, **do not Read `speedups.md`**.

---

## Phase 2 — Log Append

Append to `fasttomo-wiki/wiki/log.md`:

```
## [YYYY-MM-DD] speedups | <N findings>
<one-line summary; "None this run." if N=0>.
```

Final stdout:

```
Speedup audit complete.
- report: fasttomo-wiki/wiki/daily/$RUN_DATE/speedups.md
- findings: <N>
```

---

## Phase 3 — Failure Handling

On any uncaught exception:

1. Write `$REPORT_DIR/speedups-aborted.md` with `status: aborted`, the
   phase that failed, and the last successful step.
2. Exit non-zero.

---

## Important Reminders

- **One subagent only.**
- **Never re-Read the report file.**
- **No web access.**
- **No ledger writes** — speedups don't feed `papers-seen.md`.
- **Zero findings is the correct answer on most days.**
- **No writes under `.claude/`.**
- **Stay within the allowlist.**

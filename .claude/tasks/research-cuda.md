# Daily SOTA Research — CUDA & GPU Architecture

Single-stream SOTA research scan for FastTomo. This prompt is fed to
`claude -p` non-interactively by cron. Read it fully and execute every phase
in order.

This job runs **one** researcher subagent for the CUDA stream and updates the
papers-seen ledger from the subagent's return. It does **not** touch
synthesis, health checks, or GH issues — those live in `daily-health-and-research.md`.

---

## Operating Constraints

- Working directory: `/home/cedlim/FastTomo` (verify with `pwd` first).
- All run artefacts go to `fasttomo-wiki/wiki/daily/<RUN_DATE>/`. Do not
  write under `.claude/`.
- The wiki directory is gitignored — research reports stay local.
- Permissions: Bash actions are gated by `.claude/settings.local.json`. If a
  command is denied, log under `$REPORT_DIR/research-cuda-aborted.md` and
  exit non-zero.
- **Do not re-read `research-cuda.md` after the subagent writes it.** The
  subagent returns its full report content (including the
  `## Papers Surfaced (this run)` trailer) in its tool result. Parse the
  ledger update from that return — never `Read` the file you just wrote.

---

## Project Posture (read before proposing candidates)

FastTomo's published-algorithm implementations (RESIRE, GENFIRE, W-SIRT,
and any future SART / FISTA / ASD-POCS / PnP) are **frozen baselines for
comparison**. Do not propose algorithmic improvements to their numerics or
convergence behavior.

SIRT and FBP are reference algorithms with a split rule:
- **Tech/CUDA speedups are allowed** when numerics are identical (kernel
  fusion, batched FFT, CUDA Graphs, texture-cache reuse, fp16 mixed
  precision with documented error bounds, etc.).
- **Mathematical additions are NOT allowed.** No TV loss, no
  neural-network priors, no denoisers, no learned regularizers, no
  missing-wedge inpainting bolted onto the iteration. SIRT/FBP stay
  textbook. Regularizers belong on the AD path.

Innovation in FastTomo lives in the **autograd (AD) path**:
`forward_project_3d` + `_grad_rotmat_raw` + Adam over volume / pose /
shifts. The math-vs-tech restriction is **relaxed** here — TV, learned
regularizers, denoisers, INR parametrizations, neural-net priors are all
in scope. Treat new literature as inspiration for the AD method, not as
a list of algorithms to reimplement and improve.

Tag each Top-3 candidate with one of:
- `AD-innovation` — extends the AD path (e.g. PnP hooks, learned
  regularizer, INR parametrization, joint pose+shifts+volume tweaks,
  neural denoiser inside the loop).
- `new-baseline` — published algorithm implemented faithfully as a
  frozen comparator.
- `SIRT/FBP-tech-speedup` — numerics-identical CUDA/tech speedup of
  SIRT or FBP.
- `SIRT/FBP-math-change` — **disallowed**; flag and stop. Repackage as
  `AD-innovation` if the underlying idea (regularizer, prior, etc.) is
  genuinely useful.
- `frozen-baseline` — touches RESIRE/GENFIRE/W-SIRT internals; do not
  propose unless it's a pure refactor and the user has explicitly asked.

---

## Phase 0 — Preflight

```bash
cd /home/cedlim/FastTomo
RUN_DATE=$(date -u +%Y-%m-%d)
REPORT_DIR="fasttomo-wiki/wiki/daily/$RUN_DATE"
LEDGER="fasttomo-wiki/wiki/daily/papers-seen.md"
mkdir -p "$REPORT_DIR"
```

If `$LEDGER` does not exist, create it with the one-line header
"`# Papers Surfaced by Daily SOTA Scans`" plus the table header. The ledger
is otherwise append-only.

Read `$LEDGER` once and hold it in working memory — you will inject its full
contents into the researcher brief and use it again in Phase 2 to dedup new
rows.

---

## Phase 1 — Spawn Researcher (single subagent)

Spawn **one** `researcher` subagent with the brief below. Inject the full
ledger contents under a section titled `## Already-Surfaced Papers (skip
unless new evidence)` — substitute `$LEDGER_CONTENTS` literally.

**Output path:** `$REPORT_DIR/research-cuda.md`

**Brief (copy verbatim, substituting `$REPORT_DIR` and `$LEDGER_CONTENTS`):**

> Read the "## Project Posture" section above before proposing candidates. Tag each Top-3 finding with one of the 5 category labels defined there.
>
> Survey the last 12 months of advances in CUDA programming and GPU
> architecture relevant to a parallel-beam tomography library that uses
> `tex2DLayered` / `tex3D` for projection kernels and PyTorch C++/CUDA
> extensions for the Python boundary. Cover at minimum: new CUDA toolkit
> features (CUDA 13+ runtime, driver API), CUTLASS releases, CCCL/Thrust,
> tensor-core utilization for non-matmul ops (cuTENSOR, mma.sync for fp16
> reductions), CUDA Graphs for repeated kernel launch patterns,
> async-copy / cp.async patterns, Hopper sm_90 and Blackwell sm_100/120
> tricks, PyTorch custom-op API (torch.library, torch.compile interaction
> with custom ops). Map findings onto these FastTomo files specifically:
> `csrc/cuda/projection_2d.cu`, `csrc/cuda/projection_3d.cu`,
> `csrc/cuda/projection_resire.cu`, `csrc/cuda/genfire.cu`,
> `csrc/cuda/grad_angle.cu`, `csrc/cuda/texture_cache.cu`. FastTomo targets
> sm_70+ by default — flag clearly anything that requires newer arches.
>
> **Output path:** `$REPORT_DIR/research-cuda.md`
>
> **Cost discipline:** maximum 6 WebFetch calls. Prefer abstracts, blog
> landing pages, and release notes over full PDFs. If you find yourself
> wanting to fetch more, you are doing too much depth — pick the top 6.
>
> ## Already-Surfaced Papers (skip unless new evidence)
>
> $LEDGER_CONTENTS
>
> Skip any paper already listed unless you have substantively new evidence —
> a follow-up paper, retraction, major correction, or a new code release.
>
> Each Top-3 candidate must carry a category tag in bold prefix, e.g. **[AD-innovation]** title — one paragraph summary.
>
> End your report with a `## Papers Surfaced (this run)` section formatted as:
>
> ```markdown
> | ID | Title | Year | Why it matters for FastTomo |
> |---|---|---|---|
> | arxiv:XXXX.XXXXX | ... | YYYY | one-line relevance to a specific csrc/ or fasttomo/ file |
> ```
>
> Use the same ID conventions as the ledger (`arxiv:`, `doi:`, `biorxiv:`,
> `pmid:`, `github:owner/repo`, `kaggle:slug`, etc.).

After the subagent returns, **do not Read `research-cuda.md`**. The
subagent's tool result already contains the full report content including
the trailer table.

---

## Phase 2 — Ledger Update

From the subagent's returned content (held in working memory from Phase 1),
extract the `## Papers Surfaced (this run)` table. For every row whose ID is
**not** already present in the ledger contents you read in Phase 0, append
one row to `$LEDGER` with:

```
| <ID> | <Title> | <RUN_DATE> | surfaced |
```

Do **not** rewrite or reorder existing rows — the ledger is append-only.
Skip rows whose ID is already present.

Disposition is `surfaced` for now. The daily synthesis job may bump rows to
`tracked: $RUN_DATE plan (<topic>)` later if a paper drives that day's plan;
that's not this job's concern.

---

## Phase 3 — Log Append

Append to `fasttomo-wiki/wiki/log.md` per SCHEMA.md log format:

```
## [YYYY-MM-DD] research-cuda | <one-line takeaway>
N new papers; <one-line summary>.
```

Final stdout:

```
CUDA research scan complete.
- report: fasttomo-wiki/wiki/daily/$RUN_DATE/research-cuda.md
- ledger: <K> new rows (out of <N> surfaced)
```

---

## Phase 4 — Failure Handling

On any uncaught exception:

1. Write `$REPORT_DIR/research-cuda-aborted.md` with `status: aborted`,
   the phase that failed, and the last successful step.
2. Exit non-zero so cron's mail-on-error fires.

---

## Important Reminders

- **One subagent only.** Do not spawn parallel researchers — that's the
  multi-stream daily job's pattern, not this one.
- **Never re-Read the report file.** The subagent's return is the source of
  truth for the ledger update. Reading the file back is the cost regression
  this split was designed to fix.
- **Ledger is append-only.** Never reorder or delete rows.
- **No writes under `.claude/`** — output goes to `fasttomo-wiki/`.
- **Stay within the allowlist** — if denied, log and exit.

# Daily SOTA Research — DL & Neural-Field Tomography

Single-stream SOTA research scan for FastTomo. This prompt is fed to
`claude -p` non-interactively by cron. Read it fully and execute every phase
in order.

This job runs **one** researcher subagent for the DL / neural-field stream
and updates the papers-seen ledger from the subagent's return.

---

## Operating Constraints

- Working directory: `/home/cedlim/FastTomo` (verify with `pwd` first).
- All run artefacts go to `fasttomo-wiki/wiki/daily/<RUN_DATE>/`.
- The wiki directory is gitignored — research reports stay local.
- Permissions: gated by `.claude/settings.local.json`. If denied, log under
  `$REPORT_DIR/research-dl-aborted.md` and exit non-zero.
- **Do not re-read `research-dl.md` after the subagent writes it.** Parse
  the ledger update from the subagent's tool-result return.

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
"`# Papers Surfaced by Daily SOTA Scans`" plus the table header.

Read `$LEDGER` once and hold its contents in working memory.

---

## Phase 1 — Spawn Researcher (single subagent)

Spawn **one** `researcher` subagent with the brief below. Inject the full
ledger contents under `## Already-Surfaced Papers (skip unless new evidence)`.

**Output path:** `$REPORT_DIR/research-dl.md`

**Brief (copy verbatim, substituting `$REPORT_DIR` and `$LEDGER_CONTENTS`):**

> Read the "## Project Posture" section above before proposing candidates. Tag each Top-3 finding with one of the 5 category labels defined there.
>
> Survey the last 12 months of deep-learning and neural-field approaches to
> tomographic reconstruction. Cover: implicit neural representations / NeRF
> variants for CT, diffusion priors / score-based reconstruction, plug-and-
> play with deep denoisers, foundation models for inverse problems, Gaussian
> splatting adapted for volumes, equivariant networks, self-supervised
> recon. FastTomo currently has no learned components — assess each
> approach's *plug-in difficulty*: which would slot into the existing
> differentiable FP/BP via `fasttomo/autograd.py` with minimal surgery?
> Which require a training pipeline FastTomo doesn't have? Map onto
> `fasttomo/algorithms.py` for inference-time hooks. Be honest about
> capability mismatch — flag DL methods that are infeasible without training
> infrastructure.
>
> **Output path:** `$REPORT_DIR/research-dl.md`
>
> **Cost discipline:** maximum 6 WebFetch calls. Prefer abstracts and
> landing pages over full PDFs.
>
> ## Already-Surfaced Papers (skip unless new evidence)
>
> $LEDGER_CONTENTS
>
> Skip any paper already listed unless you have substantively new evidence.
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
> Use ID conventions: `arxiv:`, `doi:`, `biorxiv:`, `pmid:`,
> `github:owner/repo`, `kaggle:slug`.

After the subagent returns, **do not Read `research-dl.md`**.

---

## Phase 2 — Ledger Update

From the subagent's returned content, extract the `## Papers Surfaced (this run)`
table. For every row whose ID is **not** already present in the ledger,
append:

```
| <ID> | <Title> | <RUN_DATE> | surfaced |
```

Do not rewrite or reorder existing rows. Skip already-present IDs.

---

## Phase 3 — Log Append

Append to `fasttomo-wiki/wiki/log.md`:

```
## [YYYY-MM-DD] research-dl | <one-line takeaway>
N new papers; <one-line summary>.
```

Final stdout:

```
DL research scan complete.
- report: fasttomo-wiki/wiki/daily/$RUN_DATE/research-dl.md
- ledger: <K> new rows (out of <N> surfaced)
```

---

## Phase 4 — Failure Handling

On any uncaught exception:

1. Write `$REPORT_DIR/research-dl-aborted.md` with `status: aborted`, the
   phase that failed, and the last successful step.
2. Exit non-zero.

---

## Important Reminders

- **One subagent only.**
- **Never re-Read the report file.**
- **Ledger is append-only.**
- **No writes under `.claude/`.**
- **Stay within the allowlist.**

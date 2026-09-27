# Task: SIRT runner for continuous-rotation datasets (exact where possible, honest where not)

Same worktree, same conventions, same datasets as TASK_ADRECON.md (read it first, plus
HANDOFF_ADRECON.md — the geometry/axis mapping in benchmarks/scripts/utils/perrow_ad.py
is reviewed, verified, and MUST be reused, not re-derived). Do not commit. CPU tests
only, no GPU jobs.

## Scientific setup

`run_sirt` (fasttomo/algorithms.py:393) iterates x += ω·C·Aᵀ·R·(b − A·x) where A is the
single-axis Radon forward projector with ONE angle per projection. Consequences for the
continuous-rotation data:

- **fast-perp** (`angles_per_col`): the angle varies along the TILT AXIS, so each
  tilt-axis slice y is an independent 2D problem whose 12 (or more) projections each
  have a single well-defined angle θ(k,y). Textbook SIRT applies EXACTLY, slice-wise.
- **fast-par** (`angles_per_row`): the angle varies along the DETECTOR direction — a
  single frame is NOT a Radon projection at any angle, and SIRT's A cannot express it.
  The only standard-SIRT treatment is the per-frame mid-angle approximation
  (`angles.npy`), which is geometrically WRONG by up to ±Δθ/2 across the frame. That
  expected failure is the point of the experiment — implement it straight, do not
  smuggle in per-line corrections.

## Deliverables

1. `benchmarks/scripts/utils/perrow_sirt.py`
   - Reuse perrow_ad's dataset loading + generator↔FastTomo axis mapping utilities
     (import them; do not duplicate).
   - `sirt_continuous(dataset, mode, ...)` with modes:
     - `perp-exact`: slice-wise SIRT — for each tilt-axis slice, build the [n_frames,
       1, N_det] sinogram slab and its per-slice angle vector θ(k, slice), call
       `run_sirt` (positivity=True, n_iter/omega knobs), assemble slices back.
       On CUDA this is many small calls — that's fine; keep it simple and correct.
     - `midangle`: plain full-volume `run_sirt` with `angles.npy` (works for BOTH
       arms — this is the honest baseline and the par arm's ONLY option).
   - Runner `benchmarks/scripts/recon_continuous_sirt.py`: `--dataset --out
     --mode {auto,perp-exact,midangle} --iters --omega --device --seed`; `auto` =
     perp-exact for fast-perp, midangle for fast-par. Outputs identical in structure to
     the adrecon runner: generator-frame [z,y,x] `vol_final.npy`, `loss_log.csv`
     (iter + sinogram MSE from run_sirt's return_loss/loss_callback), `run_manifest.yaml`
     (config incl. mode, hashes, git rev, runtime).
2. `tests/test_perrow_sirt.py` (CPU-only, must pass):
   - perp-exact forward consistency: on a small odd-size synthetic volume (reuse the
     reference generator from tests/test_perrow_ad.py), run a few slice-wise SIRT
     iterations against per-col-exact reference projections and assert the
     reconstruction correlates with the ground-truth object (loose bound, e.g.
     pearson > 0.7 after ~30 iters) AND that its sinogram residual decreases.
   - Slice-equivalence: slice-wise SIRT on a constant-angle dataset equals full-volume
     run_sirt with the same angles (tight tolerance) — proves the slab assembly.
   - midangle mode runs end-to-end on a tiny par-arm dataset (no accuracy assertion —
     the approximation is expected to be poor at large Δθ; just shape/finite checks).
3. Append a "SIRT runner" section to HANDOFF_ADRECON.md: what each mode does, why
   par-exact is impossible inside SIRT's operator structure (one angle per Radon
   projection), test output pasted.
4. LAST action: write SIRT_DONE.md at the worktree root (one-paragraph summary).

## Validation

`conda run -n FastTomo-adrecon python -m pytest tests/test_perrow_sirt.py -v`, then the
full CPU suite, then ruff on touched files. Note: run_sirt's CPU path exists — use
device='cpu' in tests.

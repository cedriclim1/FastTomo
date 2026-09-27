# Task: per-row-exact continuous-rotation AD reconstruction harness (FastTomo)

You are working in the git worktree `/home/cedlim/FastTomo/worktrees/auto-et-adrecon`
(branch `feat/auto-et-adrecon` off FastTomo main). Do NOT commit this brief or the
handoff file. Do NOT run GPU jobs (you may not have GPU access) — CPU unit tests only.

## Goal

Reconstruct "continuous-rotation" tilt series where the tilt angle changes DURING each
frame scan — every scan line (row or column) of every frame has its own known angle —
using FastTomo's differentiable projectors in an AD loop (in the spirit of
`run_ad_recon` / `benchmarks/scripts/utils/ad_reconstruct.py`). Geometry must be
per-line EXACT (no per-frame mid-angle approximation).

## Input datasets (read-only!)

Directories like `/home/cedlim/quantem/mm_2026/automated_tomography/data/continuous/exact_15deg`
(atomic, fast-par), `perp_15deg` (atomic, fast-perp), `phantom_exact_1deg`,
`phantom_perp_9deg`, … Each contains:

- `tilt_series.npy` `[n_frames, Y, X]` float32, globally max-normalized
- `angles.npy` `[n_frames]` (mean frame angle, deg)
- `angles_per_row.npy` `[n_frames, Y]` (fast-par) OR `angles_per_col.npy` `[n_frames, X]` (fast-perp), degrees
- `gen_manifest.yaml` — READ IT; authoritative generator convention:
  `rot_ZXZ(volume, z1=0, x=theta_deg, z3=0); sum along Z; volume order [z,y,x]; output images (Y,X)`
  i.e. the rotation axis is the image **X** axis; fast-par varies theta along image Y
  (`theta(k,y)`), fast-perp along image X (`theta(k,x)`). The reference rotation
  operator is scipy-style `R.from_euler("ZXZ", [0, theta, 0])` applied about the
  volume center with trilinear interpolation, then summed along Z (verify against the
  generator: `/home/cedlim/quantem/mm_2026/automated_tomography/scripts/gen_continuous_exact.py`
  — read it, replicate its exact convention in your reference implementation).

## FastTomo conventions (CLAUDE.md of this repo)

- Volume `[Ny, Nz, Nx]`, **Ny is the tilt axis** (independent slices under the
  single-axis scalar projector), sinogram `[n_angles, Ny, N_det]`, angles in degrees.
- So the generator's image-X axis (its rotation axis) maps to FastTomo's Ny, generator
  image-Y maps to the detector axis, generator Z to Nz. Signs/permutations are YOUR job
  to get right and PROVE (see tests below) — do not trust internal consistency alone.

## Required geometry handling (both arms exact)

- **fast-perp** (`angles_per_col`): angle varies along the tilt axis → each tilt-axis
  slice y has a single well-defined angle per frame θ(k,y). Exactly separable.
- **fast-par** (`angles_per_row`): angle varies along the DETECTOR direction → every
  detector pixel d has its own angle θ(k,d), shared across all tilt-axis slices.
- Recommended unified implementation: **expanded-angle FP + gather + masked loss**.
  Build the deduplicated set of pseudo-angles; each iteration forward-project the
  volume at all pseudo-angles (CHUNKED, with gradient accumulation, to bound memory —
  worst case phantom 1°: 180 frames × 200 lines = 36 000 pseudo-angles at 200³; chunk
  size must be a config knob), gather the one valid line per (frame, line) into the
  predicted sinogram, and compute the loss only on gathered lines. You may instead (or
  additionally) implement the cheaper slice-wise path for fast-perp if you show it is
  numerically equivalent. Design for correctness first, then speed; document expected
  per-iteration cost.
- Use `fasttomo.forward_project` (scalar single-axis) — CPU path exists (numpy in) so
  the whole forward model is unit-testable WITHOUT a GPU.
- Exact angles are given: NO pose refinement, NO shifts.

## AD loop + knobs

Adam on the volume (init zeros, like `run_ad_recon`), loss `mse`, optional
regularizers: 3D TV (Charbonnier ok) and L1 sparsity, plus non-negativity clamp each
step. CLI knobs: `--iters --lr --tv --sparsity --chunk --loss {mse,l1} --device --seed`.

## Deliverables (in this worktree)

1. `benchmarks/scripts/utils/perrow_ad.py` — the harness module (geometry mapping,
   pseudo-angle expansion, masked AD loop). Pure library-consumer code; only touch
   `fasttomo/` core if genuinely necessary and document why in the handoff.
2. `benchmarks/scripts/recon_continuous_adrecon.py` — runner:
   `--dataset <dir> --out <dir> --iters ... --tv ... --sparsity ...`
   Auto-detects arm from which per-line file exists. Writes to `--out`:
   `vol_final.npy` **in the generator/GT frame `[z, y, x]`** (permute back!),
   `loss_log.csv` (iter, total, fidelity, tv, sparsity), `run_manifest.yaml`
   (dataset path + sha of inputs, full config, git rev, runtime).
3. `tests/test_perrow_ad.py` — CPU-only pytest, must pass, THE correctness proof:
   a. Reference generator: small random volume (e.g. 24³) → per-line-exact projections
      via your own scipy/torch replication of the gen_continuous_exact.py convention.
   b. Assert the harness forward model (FastTomo CPU FP + expansion + gather + axis
      mapping) matches the reference projections to ~1e-3 for BOTH arms, several Δθ.
      This is an ABSOLUTE-frame test: a transposed/flipped/negated-angle mapping must
      FAIL it. No test that both paths share the same wrong orientation.
   c. A tiny end-to-end CPU recon (few iters) decreasing loss.
4. `HANDOFF_ADRECON.md` at the worktree root — write it as your LAST action:
   what you built, test results (paste pytest output), axis-mapping table
   (generator axis ↔ FastTomo axis ↔ sign), memory/cost notes, any fasttomo/ edits.

## Environment

`conda run -n FastTomo-adrecon python ...` / `conda run -n FastTomo-adrecon pytest tests/test_perrow_ad.py`
(env is being built from this worktree; if it doesn't exist yet, use `conda run -n FastTomo ...`
— same Python 3.14 + fasttomo build, fine for CPU tests).

## Style

Match existing FastTomo code style. No AI attribution anywhere. Do not commit anything
— leave changes in the working tree.

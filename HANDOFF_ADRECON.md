# ADRECON handoff

## What was built

- `benchmarks/scripts/utils/perrow_ad.py`
  - Maps generator volumes `[z, y, x]` to FastTomo `[x, z, y]` and back.
  - Deduplicates the supplied float32 per-line angles into pseudo-angles.
  - Uses `fasttomo.forward_project` at the unchanged generator angle sign.
  - Implements exact gather geometry for both scan arms:
    - fast-par gathers one detector coordinate for every tilt-axis slice.
    - fast-perp gathers one tilt-axis slice with every detector coordinate.
  - Computes fidelity only from gathered measurements.
  - Bounds training memory by calling backward separately for every pseudo-angle
    chunk; each chunk loss is normalized by the full measured-pixel count, so
    the accumulated gradient is the global mean MSE/L1 gradient.
  - Implements zero-initialized Adam reconstruction, optional 3-D
    Charbonnier TV and mean L1 sparsity, and a non-negativity clamp after every
    step.
- `benchmarks/scripts/recon_continuous_adrecon.py`
  - Auto-detects fast-par versus fast-perp from the per-line angle filename.
  - Validates the generator manifest arm when `scan_orientation` is present.
  - Refuses to put outputs in the read-only dataset tree.
  - Exposes `--dataset --out --iters --lr --tv --sparsity --chunk --loss
    --device --seed`.
  - Writes generator-frame `[z, y, x]` `vol_final.npy`, the requested loss CSV,
    and a YAML manifest with all configuration, resolved device, geometry,
    SHA-256 for all four inputs, git revision, timestamps, runtime, and output
    metadata.
- `tests/test_perrow_ad.py`
  - Contains an independent torch `grid_sample` replication of the
    authoritative generator's `rot_ZXZ(0, theta, 0); sum(Z)` operation.
  - Tests both arms against that absolute-frame reference at frame increments
    of 7.5 and 21 degrees, using an asymmetric random 21-cubed object. The
    object has a zero boundary margin so differences between the generator's
    zero-padding edge rule and FastTomo's tight interpolation stencil do not
    obscure the axis/sign proof.
  - Includes a tiny CPU reconstruction test that checks loss decrease and
    non-negativity.

No files under `fasttomo/` or `csrc/` were changed.

## Axis and sign mapping

| Generator quantity | FastTomo quantity | Index/sign mapping |
|---|---|---|
| volume Z | volume Nz | `z -> Nz`, positive |
| volume Y | volume Nx / detector coordinate | `y -> Nx`, positive |
| volume X (rotation axis) | volume Ny / independent tilt-axis slice | `x -> Ny`, positive |
| image Y (row) | sinogram detector | `Y -> detector`, positive |
| image X (column) | sinogram Ny | `X -> Ny`, positive |
| `theta` about generator X | scalar FastTomo angle about Ny | `theta_ft = theta_generator` |

Thus `volume_fasttomo = volume_generator.permute(2, 0, 1)`, each generator
frame `[Y, X]` corresponds to a transposed FastTomo sinogram plane `[X, Y]`,
and the saved result uses the inverse permutation
`volume_generator = volume_fasttomo.permute(1, 2, 0)`.

## Memory and per-iteration cost

Let `U` be the number of unique pseudo-angles, `C` the configured chunk size,
`A = image_X` the number of tilt-axis slices, `D = image_Y` the detector size,
and `L` the ray integration length. Each iteration performs `U` ordinary
scalar FastTomo projections in `ceil(U/C)` chunks. The projector work is
`O(U * A * D * L)` and the peak expanded projection is
`O(C * A * D)`, rather than `O(U * A * D)`. The target tilt series, volume,
volume gradient, and Adam state remain resident. Regularizer graphs are built
only when their weights are nonzero.

For the worst cited 180-frame, 200-line phantom, `U` can be 36,000. At
`C=16`, the expanded output for one chunk is about 2.44 MiB in float32,
whereas all expanded projections would be about 5.36 GiB. Correctness-first
expanded projection is computationally expensive: with roughly 287 ray
samples, one iteration is approximately 413 billion trilinear samples. The
implementation intentionally uses the same unified exact gather path for
both arms; it does not add the optional fast-perp slice-wise optimization.

## Validation

Required focused test output:

```text
============================= test session starts ==============================
platform linux -- Python 3.14.4, pytest-9.0.3, pluggy-1.6.0 -- /home/cedlim/miniforge3/envs/FastTomo-adrecon/bin/python3.14
cachedir: .pytest_cache
rootdir: /home/cedlim/FastTomo/worktrees/auto-et-adrecon
configfile: pyproject.toml
plugins: zarr-3.2.1
collecting ... collected 5 items

tests/test_perrow_ad.py::test_forward_matches_absolute_generator_frame[7.5-fast-par] PASSED [ 20%]
tests/test_perrow_ad.py::test_forward_matches_absolute_generator_frame[7.5-fast-perp] PASSED [ 40%]
tests/test_perrow_ad.py::test_forward_matches_absolute_generator_frame[21.0-fast-par] PASSED [ 60%]
tests/test_perrow_ad.py::test_forward_matches_absolute_generator_frame[21.0-fast-perp] PASSED [ 80%]
tests/test_perrow_ad.py::test_tiny_cpu_reconstruction_decreases_loss PASSED [100%]

============================== 5 passed in 1.07s ===============================
```

Full CPU regression test:

```text
======================== 74 passed, 1 skipped in 3.23s =========================
```

The skipped test was the existing CUDA-dependent MBIR consistency test. No GPU
jobs were run. The task brief's noted GENFIRE tolerance failure did not
reproduce in this environment.

Additional checks:

```text
ruff check benchmarks/scripts/utils/perrow_ad.py benchmarks/scripts/recon_continuous_adrecon.py tests/test_perrow_ad.py
All checks passed!
```

`py_compile`, `git diff --check`, and runner `--help` completed successfully. A
synthetic two-iteration CPU CLI smoke run verified arm autodetection,
`vol_final.npy` shape/order, CSV columns/row count, and manifest input hashes.
No input dataset was modified and no commits were made.

## Review fixes

FastTomo's ray-marched forward operator and the generator's rotate-then-sum
operator are not identical at even spatial sizes. On the new size-20 test
cases, the measured relative maximum errors were:

| Frame increment | fast-par | fast-perp |
|---|---:|---:|
| 7.5 degrees | 2.52% | 3.03% |
| 21.0 degrees | 2.12% | 3.03% |

Production 180/200-sized data should therefore be expected to carry roughly
2--5% relative model mismatch, raising the reconstruction-time fidelity floor
to approximately the 1e-3 scale. This is accepted as inverse-crime avoidance
caused by the integration schemes, not a centering bug.

The focused tests now cover both arms at odd and even cubic sizes, two frame
increments, an odd non-square 15-by-11 generator frame, and the tiny CPU
reconstruction. The gather buffer in `predict()` is zero-initialized so a
future coverage gap cannot expose uninitialized memory; current complete
coverage preserves its output.

```text
tests/test_perrow_ad.py: 11 passed in 1.13s
tests/test_cpu.py: 74 passed, 1 skipped in 3.35s
ruff check benchmarks/scripts/utils/perrow_ad.py tests/test_perrow_ad.py
All checks passed!
```

## SIRT runner

### Implementation and geometry

- `benchmarks/scripts/utils/perrow_sirt.py` adds two deliberately distinct
  treatments through `sirt_continuous`:
  - `perp-exact` transposes generator frames `[k, Y, X]` to FastTomo
    sinograms `[k, X, Y]`, then reconstructs every tilt-axis slice `x`
    independently. Each slab has shape `[n_frames, 1, Y]` and uses that
    slice's exact `angles_per_col[:, x]`. The resulting `[1, Y, Y]` slabs are
    assembled as FastTomo `[X, Z, Y]`. Every `run_sirt` call uses
    `positivity=True`; the reported history is the full-sinogram MSE, obtained
    by averaging the equal-sized per-slice histories returned by `run_sirt`.
  - `midangle` is one ordinary full-volume `run_sirt` call using
    `angles.npy`. It works for both scan arms and is intentionally only a
    per-frame approximation when the acquisition rotates during a frame.
- Fast-par has no exact representation in standard SIRT's scalar-angle
  operator. Its angle changes along the detector coordinate inside a single
  frame, while `run_sirt` requires one angle for every complete Radon
  projection. No choice of a single angle can encode all detector rows;
  therefore `perp-exact` rejects fast-par inputs and `midangle` is the only
  supported fast-par mode. No per-line correction is hidden in that baseline.
- Shared continuous-dataset loading, validation, arm detection, and the
  generator/FastTomo sinogram transpose now live in
  `benchmarks/scripts/utils/perrow_ad.py`. The SIRT library and runner import
  those reviewed utilities rather than duplicating the axis mapping. The AD
  runner was switched to the same loader without changing its reconstruction
  behavior.
- `benchmarks/scripts/recon_continuous_sirt.py` exposes `--dataset --out
  --mode --iters --omega --device --seed`. `auto` selects `perp-exact` for
  fast-perp and `midangle` for fast-par. It writes generator-frame `[z, y, x]`
  `vol_final.npy`, `loss_log.csv` with one-based iteration and sinogram MSE,
  and `run_manifest.yaml` containing requested/effective mode, complete
  configuration, input SHA-256 hashes, git revision, timestamps, runtime, and
  output metadata.
- `tests/test_perrow_sirt.py` uses the independent authoritative generator
  reference from `tests/test_perrow_ad.py`. It proves odd-size fast-perp
  reconstruction correlation and residual reduction after 30 iterations,
  tight equality between slab-wise and full-volume SIRT for constant angles,
  finite end-to-end fast-par mid-angle behavior, and explicit rejection of an
  impossible fast-par exact request.

No files under `fasttomo/` or `csrc/` were changed for SIRT.

### Validation

Required focused test output:

```text
============================= test session starts ==============================
platform linux -- Python 3.14.4, pytest-9.0.3, pluggy-1.6.0 -- /home/cedlim/miniforge3/envs/FastTomo-adrecon/bin/python
cachedir: .pytest_cache
rootdir: /home/cedlim/FastTomo/worktrees/auto-et-adrecon
configfile: pyproject.toml
plugins: zarr-3.2.1
collecting ... collected 4 items

tests/test_perrow_sirt.py::test_perp_exact_reconstruction_correlates_and_residual_decreases PASSED [ 25%]
tests/test_perrow_sirt.py::test_perp_slice_sirt_equals_full_volume_for_constant_angles PASSED [ 50%]
tests/test_perrow_sirt.py::test_midangle_runs_on_tiny_fast_par_dataset PASSED [ 75%]
tests/test_perrow_sirt.py::test_perp_exact_rejects_fast_par_geometry PASSED [100%]

============================== 4 passed in 1.13s ===============================
```

Full CPU validation, excluding the CUDA-only files, completed with:

```text
121 passed, 6 skipped in 15.22s
```

The six skips are GPU-dependent tests. The required `tests/test_cpu.py` run
was `74 passed, 1 skipped in 5.10s`; the focused AD geometry regression was
`11 passed in 1.21s`.

```text
ruff check benchmarks/scripts/utils/perrow_ad.py benchmarks/scripts/utils/perrow_sirt.py benchmarks/scripts/recon_continuous_adrecon.py benchmarks/scripts/recon_continuous_sirt.py tests/test_perrow_sirt.py
All checks passed!
```

`py_compile`, `git diff --check`, and runner `--help` also passed. CPU CLI
smoke runs used both real 180-cubed datasets. Auto mode selected `midangle`
for `exact_15deg` (fast-par) and `perp-exact` for `perp_15deg` (fast-perp);
both produced finite `[180, 180, 180]` float32 generator-frame volumes,
correct loss rows, the expected per-line input hashes, and matching manifest
mode metadata. No GPU jobs were run and no dataset input was modified.

## SIRT discriminating-test hardening

The fast-perp exact-path test now uses a realistic per-column sweep of
`-7.5` to `+7.5` degrees and asymmetric edge markers that make angular
misalignment observable at its small test size. On the same synthetic data,
the measured reconstruction correlations were `corr_exact=0.906461596` and
`corr_mid=0.814047158`, an exact-over-mid gap of `0.092414438`. The asserted
margin is `0.05`, leaving about `0.0424` measured headroom. The existing
constant-angle slab-equivalence test is unchanged.

Required focused test output:

```text
============================= test session starts ==============================
platform linux -- Python 3.14.4, pytest-9.0.3, pluggy-1.6.0 -- /home/cedlim/miniforge3/envs/FastTomo-adrecon/bin/python
cachedir: .pytest_cache
rootdir: /home/cedlim/FastTomo/worktrees/auto-et-adrecon
configfile: pyproject.toml
plugins: zarr-3.2.1
collecting ... collected 4 items

tests/test_perrow_sirt.py::test_perp_exact_reconstruction_correlates_and_residual_decreases PASSED [ 25%]
tests/test_perrow_sirt.py::test_perp_slice_sirt_equals_full_volume_for_constant_angles PASSED [ 50%]
tests/test_perrow_sirt.py::test_midangle_runs_on_tiny_fast_par_dataset PASSED [ 75%]
tests/test_perrow_sirt.py::test_perp_exact_rejects_fast_par_geometry PASSED [100%]

============================== 4 passed in 0.86s ===============================
```

Full CPU validation, excluding the CUDA-only test files, completed with
`124 passed, 7 skipped in 8.00s`; the remaining skips are GPU-dependent.
`ruff check tests/test_perrow_sirt.py` reported `All checks passed!`.

# Changelog

## Unreleased

### ⚠ `run_mbir`: the qGGMRF prior is now the paper's (all MBIR results change)

The kernels computed the surrogate of `|D|^p / (c sigma_f^p + |D|^(p-q) sigma_f^q)`
with `p=1.2, q=2`, i.e. `(D/sigma_f)^2 / (1 + c |D/sigma_f|^0.8)`: `p` and `q`
had swapped roles, so the prior stayed quadratic (Gaussian MRF) for
differences up to ~1e4 `sigma_f` and did not preserve edges. Both backends now
implement the potential of Venkatakrishnan et al. 2015 (and OpenMBIR-TEM),
`rho(D) = |D/sigma_f|^q / (c + |D/sigma_f|^(q-p))`: quadratic (`q=2`) below
about `c^(1/(q-p)) sigma_f` (1.8e-4 `sigma_f` at the defaults), `|D|^p`
(`p=1.2`) above. The keyword names and defaults (`qggmrf_p=1.2`,
`qggmrf_q=2`, `qggmrf_c=1e-3`, `sigma_f=1.0`) are unchanged.

- **Every MBIR output changes**, CPU and CUDA. On a 96^3 three-ellipsoid
  phantom (41 views +-60 degrees, 1 % noise, fixed sigma, 80 iterations) the
  best RMSE over `sigma_f` goes from 0.0591 to 0.0463 (-22 %), at the same
  best `sigma_f = 0.1`; at the default `sigma_f = 1`, 0.0985 -> 0.0766, and a
  fully default call 0.131 -> 0.100. The BF-TEM demo
  (`benchmarks/scripts/run_mbir_bftem_demo.py`, its `sigma_f = 0.1`):
  conventional MBIR 0.0283 -> 0.0256, MBIR-AM 0.0210 -> 0.0187 (SIRT 0.0296).
  The default `sigma_f` stays 1.0: it was already 10x above the best value
  on these phantoms and improves as much as the best value does.
- **`sigma_f` changes meaning near zero**: the prior is `1/c` = 1000x stiffer
  than `(D/sigma_f)^2` there. Re-tune values chosen for the old prior (a
  tenth of the expected contrast between neighbouring regions is a good
  start). ICD needs more iterations than with the old, quadratic prior: a
  noise-free 32^3 cube (`sigma_f=0.1`, `estimate_variance=True`) reaches a
  relative error of 0.15 / 0.049 at 80 / 300 iterations (0.086 at 80
  before), 2 multires levels meet the single-level estimate to 6.5e-5 after
  3000 iterations per level (5e-4 after 60 before), and a zero start with a
  small `sigma_f` and `estimate_variance=True` can barely move for tens of
  iterations.
- `qggmrf_p`, `qggmrf_q` must satisfy `1 <= p <= q <= 2` and `qggmrf_c`,
  `sigma_f` (and `sigma_f_per_scale`) must be positive: `ValueError`
  otherwise. The old workaround for the paper's prior,
  `qggmrf_p=2, qggmrf_q=1.2`, now raises instead of meaning something else.
- The 26-neighbour weights are exact to 9 digits (the corner weight read
  0.030219 instead of 0.0302213, so they summed to 0.99999).
- Golden keys `cpu/mbir`, `cuda/mbir` and every key of
  `tests/data/mbir_sweep_golden.pt` were regenerated.

### ⚠ `run_mbir` on CUDA: parallel ICD sweep by default (new visiting order), `sweep_kernel=`

- New keyword-only `sweep_kernel="auto" | "serial" | "parallel"` on
  `run_mbir` and `MbirConfig`. The parallel sweep updates voxels of a row
  that are `S = ceil((2 + eps) / min|cos theta|)` apart (5 at +-60 degrees,
  6 at +-70, 12 at +-80) at the same time: their detector footprints are
  disjoint in every view, so this is still exact Gauss-Seidel ICD (it equals
  the serial sweep run on the same visiting order, bit for bit; tested). One
  block per Y-slice, the slice's error sinogram in shared memory. `"auto"`
  (the default) uses it on CUDA where it applies (a spacing `S <= Nx // 2`,
  i.e. no view near 90 degrees; at most 256 views; `n_angles * (N_det + 1)
  * 4` bytes of shared memory per block, 99 KB on sm_120) and the serial
  sweep otherwise (e.g. 512^3 at 61 views); `"parallel"` raises where it
  does not apply.
- **CUDA results for a given `seed` change** (groups visited in a keyed
  order instead of single voxels), and the group order needs 1.2-1.3x more
  iterations to the same `convergence_tol`. Measured (61 views +-60
  degrees, fp32, RTX PRO 6000 Blackwell, `sigma_f=0.1`, fixed sigma):

  | | serial | parallel |
  |---|---|---|
  | 128^3 per iteration | 9.72 ms | 1.25 ms |
  | 128^3 to tol 1e-3 / 1e-4 | 54 / 201 it, 525 / 1945 ms | 69 / 255 it, 86 / 318 ms |
  | 256^3 per iteration | 40.2 ms | 8.80 ms |
  | 256^3 to tol 1e-3 / 1e-4 | 33 / 151 it, 1326 / 6017 ms | 39 / 195 it, 343 / 1707 ms |

- **CPU and CUDA no longer visit the voxels in the same order by default**
  (the CPU keeps the serial voxel order); they agree at convergence (the
  same MAP estimate, tested through the MAP cost), not iteration by
  iteration. `sweep_kernel="parallel"` on CPU runs the C sweep in the
  parallel sweep's order, which matches a CUDA `"parallel"` run up to
  rounding; `"serial"` on both backends matches as before.
- The prepare kernel computes the prior sums in natural voxel order (3.0 ->
  1.5 ms at 256^3, 26 -> 12 ms at 512^3; bit-identical) and the CUDA sweep
  scratch is 8 instead of 16 bytes per voxel (128 MiB at 256^3, 1 GiB at
  512^3).
- The per-voxel arithmetic of every CUDA sweep and of the MBIR forward
  projection is rounded op by op (no FMA contraction, as in the C code), and
  each lane adds its even- and odd-slot sums before one warp reduction
  (instead of reducing them separately): serial CUDA results change at the
  ulp level too, and the serial sweep costs ~3 % more per iteration.

### `run_mbir`: CUDA runs with a `support_mask` are not bit-reproducible (documented)

The part a support mask removes after each sweep is forward-projected with
fp32 atomics on CUDA, so two identical runs differ in the last bits, and
the ICD iterations amplify that (4 x 32^2 cylinder, 20 iterations: max
volume difference 3e-6 with the quadratic prior, 1e-3 with the default
prior, 0.2 with `huber_T=2`). CPU runs and CUDA runs without a mask stay
bit-reproducible for a fixed `seed`. Documented in `run_mbir` and tested
(bitwise on CPU, within 1e-4 on CUDA for the quadratic prior).

### ⚠ FBP output values change: zero-padded Ram-Lak ramp filter

`ramp_filter` multiplied the unpadded detector FFT by `|xi|`, which zeroes
the DC term and makes the convolution circular. `run_fbp` of a unit-density
cylinder came out at 0.96, 0.83 and 0.57 (radius 20, 40, 60 in a 128 grid)
instead of 1. The filter is now the spatial-domain Ram-Lak kernel
(`h[0] = 1/4`, `h[odd n] = -1/(pi n)^2`) applied with zero padding to the
next power of two `>= 2 N_det - 1`; the same cylinders reconstruct at
1.000 +- 0.002, at 180 degrees and at +-70 degrees alike. **`run_fbp` and
`ramp_filter` results differ from earlier versions** (only the FBP and
ramp-filter keys of the algorithm golden were regenerated).

- **Old FBP results cannot be converted by one factor.** The old error
  depended on the object's size (a unit cylinder read 0.96 to 0.57 from
  small to large, with a negative halo of -0.04 to -0.39 around it), so
  anything tuned on old FBP output needs re-tuning: thresholds,
  segmentation levels or colour limits (e.g. in the desktop app), and
  comparisons made through `benchmarks/scripts/utils/torch_radon_wrapper.py`,
  which imports `ramp_filter`.
- **FBP is 7-12 % slower**: the zero-padded FFT is at least twice as long.
  `ramp_filter` 0.07 -> 0.26 ms and 0.85 -> 2.25 ms, `run_fbp` 1.44 ->
  1.54 ms and 11.1 -> 12.5 ms, at 121 x 256^2 and 121 x 512^2 (CUDA).
- The `pi / n_angles` scale is unchanged. It is pi times the mean over the
  measured views, so it keeps the mean level correct at limited tilt too:
  a +-70 degree series of a rotationally symmetric object gives the same
  level as a full one (the scope report had suspected it was off by
  180/140). Anisotropic objects still show an orientation-dependent bias at
  limited tilt (a 4:1 ellipse at +-70 reads 1.20 or 0.50 depending on its
  orientation; the missing wedge), which no angular scale removes.
- fp16 `run_fbp`: the back-projection sums `n_angles` unscaled views
  before the `pi / n_angles` scale and could overflow (OverflowError)
  although the result fits (density 400 over 720 views). When it does,
  `run_fbp` now applies `pi / n_angles` to the filtered sinogram in fp32,
  before its cast to fp16, and back-projects again, so it raises only when
  the result itself exceeds the fp16 range. Otherwise nothing changes
  (bit-identical); pre-scaling always would lose small values to fp16
  underflow (a density of 1e-4 at 720 views: 8 % error instead of 0.08 %).

### ⚠ `run_mbir(multires_levels > 1)`: coarse levels hand over the right values

A multires level with factor `f` works on voxels `f` fine voxels wide, so
its values are `f` times the fine-grid ones. The hand-over to the next
finer level only resized the volume, so the fine level started at twice
the object's level, and a `warm_start` reached the coarse level at half
its values. Both are now rescaled (`x 0.5` per level, `x coarse_factor`
for a warm start). Multires results differ slightly from earlier versions
(the fine level starts 8x closer, data MSE 6.8 -> 0.82 on a test cylinder).
With `estimate_variance=False` they converge to the single-level solution
(5e-4 relative at 60 iterations per level). With `estimate_variance=True`
(the default) they reach the same fixed point only slowly, as does a single
level: 2-level vs 1-level differ by 22 / 17 / 10 / 3.6 / 0.3 % at 30 / 150 /
600 / 2400 / 9600 iterations per level on a noise-free 64^2 cylinder at
+-70 degrees (30-37 % before this fix at the low counts); with
`estimate_offset=True` as well, a common offset on every view stays
unsettled. This was investigated and is a property of the joint
volume/variance(/offset) estimate on noise-free limited-tilt data, not of
the level hand-over; see `multires_levels` in the `run_mbir` docstring.
Single-level runs are unchanged.

### fp16 overflow raises `OverflowError`

fp16 projections accumulate in fp32 but store fp16, so a result above
65504 (the back-projection of an unnormalised sinogram, the forward
projection of a large volume with large values, FISTA's first gradient)
came back as `inf`. The projection functions, the projector classes,
`run_fbp` and autograd backward now raise `OverflowError` with the remedy
(scale the data into range, or use fp32) instead; an `inf` that came with
the input is passed through. A NaN from finite input is reported
separately, as `FloatingPointError` with its own message (it used to be
reported as an overflow by the algorithms and not at all by the direct
calls); where an overflow came first, the overflow is reported. `run_sirt`, `run_sart`, `run_fista` and `run_fista_tv` check
their projections on the device and raise once when they return. fp32 is
never checked; a direct fp16 call on CUDA pays one flag read (0.06-0.07 ms measured).

### Changed

- `apply_shifts` and `apply_shifts_2d` return the sinogram's dtype (they
  returned fp32 for fp16 and fp64 input) and share one contract: shifts
  `[n_angles]` (detector axis) or `[n_angles, 2]` (detector axis, tilt
  axis), positive shifts move the signal toward larger indices, the same
  shape errors. `apply_shifts` gained the `[n_angles, 2]` form (Fourier
  shift along both axes) and `apply_shifts_2d` the `[n_angles]` form. A
  sinogram that is not 3-D raises `ValueError` (`apply_shifts` used to
  broadcast it into a wrong result).
- Size parameters default to `None` across the Python API (`Nz` of every
  reconstruction; `N_det`, `Nz`, `Nx` of the projection functions);
  `-1` is still accepted and means the same.
- fp16 SIRT / SART: a ray that grazes the volume (FP(1) below 1/65504,
  produced by the CPU rotation-matrix projector) gets a zero normaliser
  instead of `inf`. fp32 is unchanged.
- One CUDA projector used from two streams is safe: a call from another
  stream than the previous one waits for that stream first (the shared
  texture uploads used to race).
- `tilt_ssim_loss` and `s3im_loss` with `data_range=None` no longer
  synchronise the host (two syncs per `run_ad_recon` iteration with both
  weights on); values are bit-identical.

### Added

- `fasttomo.__version__` (from the installed distribution's metadata).
- `run_wbp` (alias of `run_weighted_backprojection`) and `tv_loss_zx`
  (alias of `tv_loss_xz`).
- cp314 wheels (CPU and CUDA); `wheels.yml` uses cibuildwheel v3.4.1.

### Faster CUDA `run_sirt` (bit-identical results; new public `release_cached_projectors()` and `run_sirt(reuse_graph=)`)

- **API additions** (the PR text said "no API change"; it adds two public
  names): `fasttomo.release_cached_projectors()` (also in
  `fasttomo.algorithms`) frees what `run_sirt` keeps between calls and
  returns how many configurations it released (0 or 1); the new keyword
  `run_sirt(..., reuse_graph=True)` controls whether that is kept. Existing
  calls need no change, but a CUDA `run_sirt` without `projector=` now
  keeps GPU memory after it returns (below).
- Repeated `run_sirt` calls without `projector=` reuse the previous call's
  internal projector and CUDA graph when shapes, dtype, device and options
  match, instead of building a projector, running an eager warm-up iteration
  and capturing a new graph every call. New sinogram data, angles and
  support masks of the same shapes are copied into the graph's buffers.
- **Memory kept after `run_sirt` returns** (CUDA graph path, no
  `projector=`): exactly one configuration is kept, **4 volumes plus 4
  sinograms**: the projector's volume and sinogram texture arrays (1 + 1),
  the graph's static volume and voxel normalisation (2 volumes) and static
  sinogram and ray normalisation (2 sinograms), and the graph's memory pool
  holding the back-projection and forward-projection outputs (1 + 1), plus
  the support mask when one is given (a `(1, Nz, Nx)` plane, or one more
  volume for a full-size mask). In fp32 at 61 views that is 0.31 GiB at
  256^3 and 2.24 GiB at 512^3 by count; the device memory actually taken,
  measured, is **0.36 GiB at 256^3 and 2.62 GiB at 512^3**, the difference
  being free space inside the caching allocator's segments that hold the
  static tensors (PyTorch reuses it, `torch.cuda.empty_cache()` cannot
  return it while the tensors live). A call with a different shape, dtype,
  device or option drops the kept one before allocating its own. New public `fasttomo.release_cached_projectors()`
  frees it (the texture arrays go back to the driver at once, the rest to
  PyTorch's caching allocator; follow with `torch.cuda.empty_cache()` to
  hand that back to the driver too). New keyword `run_sirt(...,
  reuse_graph=False)` keeps nothing, for a single reconstruction that
  needs the memory back.
- A graph capture that fails (e.g. out of memory, or several threads
  capturing at once) makes that configuration run the eager loop, with the
  same results. The failure is no longer remembered for the rest of the
  process: `release_cached_projectors()` or a call with another
  configuration clears it, and the next call captures again.
- The iteration's elementwise work runs in two fused CUDA kernels (residual;
  update plus support, positivity and soft-threshold constraints) instead of
  up to seven PyTorch kernels, rounding exactly like the PyTorch chain.
- The residual and the updated volume are written straight into the
  projector's textures, removing the two device-to-device texture uploads
  per iteration (102 -> 3 per 50-iteration call); on the graph path the
  volume stays in the texture for the whole call.
- `run_sirt(n_iter=50)`, 61 views, RTX PRO 6000 Blackwell: 128^3 11.3 ->
  9.5 ms (graph path), 12.1 -> 11.4 ms (eager path); 256^3 80.9 -> 76.7 ms
  (graph), 86.6 -> ~80.8 ms (eager). Outputs are bit-identical to before.
- The `tv_loss_3d` backward no longer zero-fills its output (the kernel
  writes every voxel): 256^3 143 -> 95 us.

### ⚠ `run_mbir` is single-threaded on CPU by default

`run_mbir` used to run its CPU ICD sweep on a hardcoded 4 OpenMP threads. It
now follows the library-wide policy (`num_threads` keyword, else the
`OMP_NUM_THREADS` environment variable, else 1), so **a default CPU `run_mbir`
call is up to ~4x slower than before**. Results are unchanged. To restore
the old speed pass `num_threads=4` (or more), or set `OMP_NUM_THREADS`
before starting Python. CUDA runs are not affected.

### ⚠ `run_mbir`: faster outer iterations, different voxel order per `seed`

- The random per-slice voxel order of the ICD sweep is now generated on the
  GPU (a keyed Feistel permutation per Y-slice and outer
  iteration) instead of by NumPy on the host and copied to the GPU every
  outer iteration (`Ny*Nz*Nx*4` bytes: 67 MB at 256^3, 537 MB at 512^3).
  The CPU sweep uses a bit-identical NumPy mirror of the same order, so CPU and
  CUDA still visit the voxels in the same order.
- The CUDA ICD sweep was restructured: a fully parallel prepare kernel
  computes the voxel order, pre-sweep values and qGGMRF prior sums of every
  step, and the sequential sweep runs one warp per Y-slice with a fused
  warp-shuffle reduction and no block barriers (it had about 11 per voxel),
  keeping each lane's error-sinogram values in registers. The sweep no
  longer builds a texture or synchronises the stream. Same Gauss-Seidel
  order and arithmetic: bit-identical to the previous kernel for the same
  voxel order. One CUDA outer iteration at 61 views: 128^3 52 -> 9.6 ms,
  256^3 300 -> 41 ms (default settings; 1.9-2.5x on the sweep with lambdas
  and/or a finite `huber_T`). It needs a scratch buffer of 16 bytes per
  voxel for the duration of `run_mbir` (the host permutation it replaces
  was 4 bytes per voxel on the host plus 4 on the GPU).
- **Results for a given `seed` differ from earlier versions** (a different,
  equally random order); a fixed `seed` is still bit-reproducible run to
  run. `_generate_permutation` is no longer used by `run_mbir`.

### Faster 3D forward projection, deterministic TV loss, faster CPU wSIRT

- `forward_project_3d` / `RotmatProjector.forward` (CUDA, fp32 and fp16)
  march each ray only over the steps whose trilinear stencil can touch the
  volume (slab test, united over the thread block); 43-47 % of the texture
  fetches used to return border zeros. The sinogram is bit-identical.
  Kernel time at 61 views: 128^3 1.43-1.49x, 256^3 1.67-1.69x faster. The
  matrix gradient (`grad_rotmat`) is unchanged: its bounds test already
  skipped those fetches and clipping measured 0.97-0.98x.
- `tv_loss_3d` (CUDA forward) is a fixed-order two-pass reduction with fp64
  partials instead of one fp32 `atomicAdd` per 128 voxels: **bit-reproducible
  run to run** (it varied in the last digits) and closer to the exact value
  (within 1e-7 of an fp64 reference, was up to 6e-6 on random volumes and
  1.3e-3 on a 512^3 phantom), so the loss value can differ from earlier
  versions in its last digits. 1.8x (128^3) to 4x (256^3, 512^3) faster.
  The gradient is unchanged.
- The CPU weighted back-projection (`run_weighted_backprojection`,
  `run_wsirt`) runs its voxel-driven back-projection in C/OpenMP instead of
  a per-view PyTorch loop; results are bit-identical. wSIRT at
  8x128^2x61: 81.5 -> 52.2 ms/iter on 1 thread, 57.5 -> 8.9 ms/iter on 8.
  `run_wsirt`'s `num_threads` now also drives this step, and
  `run_weighted_backprojection` takes `num_threads` (keyword-only, see
  Conventions; it used whatever threads PyTorch had).
- `run_mbir`: an explicit voxel order passed to the CUDA sweep (test hook)
  is validated (shape, range, each row a permutation); unused angle slots in
  the sweep get defined values. Results are unchanged.

### Projection error messages and double backward

- Zero-size errors from `forward_project`, `back_project`,
  `forward_project_3d`, `back_project_3d` and `refine_angles` name that
  function (they named an internal operator class, with `-1` placeholders
  for the sizes still to be taken from the input). Still `ValueError`.
- `refine_angles(n_steps < 1)` raises `ValueError` on both backends (a
  negative value raised `RuntimeError` on the CPU).
- A second-order gradient through a projection (`create_graph=True`, then
  a backward through the gradient) raises `RuntimeError` instead of
  silently dropping the projection's second-order term; the adjoint
  kernels are not themselves differentiable.

### Projection-operator seam (internal; outputs unchanged)

- Every forward/back projection now goes through one internal interface,
  `fasttomo._operator` (`ProjectionOperator` with `fp` / `bp` / `grad_geom`
  and adapters for CPU, CUDA with cached textures, and the frozen RESIRE
  kernels), chosen in one place (`_make_operator`). The functional API, the
  projector classes and every algorithm use it; two generic autograd
  Functions (`_Project`, `_BackProject`) replace the eight per-path ones in
  `fasttomo.autograd` (`_FPParallel`, `_FPParallelCached`, ... were private).
  No public API change, and every output is bit-identical to before
  (`tests/test_algorithm_identity.py`, generated on the previous `main`).
- New: `ParallelProjector` and `RotmatProjector` accept `device="cpu"` (and
  a keyword-only `num_threads`); on CPU they give exactly the functional
  API's results. Before, a CPU device raised `ValueError`.
- Faster on CUDA: `run_sart`, `run_wsirt`, `run_grad_descent` and
  `refine_angles` reuse texture caches instead of allocating a texture and
  synchronising on every projection (e.g. `run_sart` 4-6x, `refine_angles`
  2-3x faster on small problems).
- `run_fbp` on CUDA without a projector uses the one-shot texture path
  (numerically identical).

### Internal restructure (no behaviour change)

- `fasttomo/algorithms.py` is now the package `fasttomo/algorithms/`
  (`_common`, `analytic`, `algebraic`, `fista`, `ad`, `genfire`, `resire`,
  `alignment`, `mbir`). `fasttomo.algorithms` re-exports every name the
  module had, so imports such as `from fasttomo.algorithms import run_sirt`
  or `fasttomo.algorithms._prepare_recon` keep working. Code that
  monkeypatches a name an algorithm calls must patch the submodule that
  calls it (for example `fasttomo.algorithms.ad.ParallelProjector` or
  `fasttomo.algorithms._common._check_inputs`), not the package.
- Backend helpers (`_on_device`, `_resolve_num_threads`, the dtype codes,
  `_stream_ptr`, `_data_ptr`, `_check_shapes`) live in `fasttomo._backend`;
  their old `fasttomo.autograd` / `fasttomo.projector` paths still resolve.
- `apply_shifts` and `apply_shifts_2d` are defined in `fasttomo.projection`
  and `tv_loss_3d` in `fasttomo.regularizers`; `fasttomo.utilities` and
  `fasttomo` re-export all three. This removes the import cycle between
  `fasttomo.utilities` and `fasttomo.projection`.
- Existing editable installs (`pip install -e .`) made before this split must
  be reinstalled: scikit-build-core's editable finder records module paths at
  install time and still points `fasttomo.algorithms` at the deleted
  `algorithms.py`, so `import fasttomo` fails until you re-run the install.
- Tests are grouped by topic (`tests/test_projection.py`,
  `test_algorithms_*.py`, `test_contracts.py`, ...) with shared fixtures in
  `tests/conftest.py`; `gpu` and `multi_gpu` markers are registered, so
  `pytest -m "not gpu"` runs the CPU tests only.

### Breaking changes and deprecations (API consistency refactor)

Deprecated names keep working for one release and emit `DeprecationWarning`;
passing an old and a new name together raises `TypeError`.

**Call signatures.** Every reconstruction takes `(sinogram, angles_deg, n_iter)`
positionally and everything after that keyword-only (`run_fbp` and
`run_weighted_backprojection`: `(sinogram, angles_deg)`). A call written for an
old positional order now raises `TypeError` instead of binding a value to
another parameter. Keyword calls are unaffected. In detail:

| Function | Before | Now |
|---|---|---|
| `run_fbp` | `(sino, angles, Nz, positivity, mask, projector, support_mask)` | `(sino, angles, *, Nz, positivity, mask, support_mask, projector, num_threads)` |
| `run_weighted_backprojection` | `(sino, angles, Nz, positivity, threshold_scale, mask)` | `(sino, angles, *, Nz, positivity, threshold_scale, mask, support_mask)` |
| `run_sirt`, `run_sart`, `run_wsirt`, `run_fista`, `run_resire`, `run_grad_descent` | options positional after `n_iter` | keyword-only after `n_iter` |
| `run_fista_tv` | `(sino, angles, tv_weight, n_iter, ...)` | `(sino, angles, n_iter, *, tv_weight, ...)`; `tv_weight` still required |
| `run_genfire` | `(sino, angles, Nz, n_iter, ...)` | `(sino, angles, *, n_iter, Nz, ...)`: `n_iter` keyword-only for one release, so `run_genfire(s, a, 64)` (old `Nz=64`) raises |
| `run_ad_recon` | `(sino, angles, *, ...)` | `(sino, angles, n_iter=None, *, ...)`; `n_iter` overrides `config.optim.n_iter` without modifying the config |
| `run_mbir` | `(sino, angles, *, ...)` | `(sino, angles, n_iter=100, *, ...)` |
| `refine_angles` | `(volume, sino, angles_deg, euler_zyz_deg, search_range_deg, n_steps, num_threads)` | `euler_zyz_deg` and after keyword-only |
| `ConstraintConfig` | `(positivity, mask, sparsity)` | `(positivity, *, cylinder, support_mask, soft_threshold)` |

**Renamed keywords and fields (deprecated aliases).**

| Where | Old | New |
|---|---|---|
| `run_sirt`, `run_sart`, `run_fista`, `run_fista_tv` | `volume=` (warm start) | `warm_start=` |
| `run_resire` | `support=` | `support_mask=` |
| `run_mbir` | `outer_iters=` | `n_iter=` (still the per-level maximum) |
| `run_mbir` | `return_history=True` -> `(volume, d_offsets, sigma, history)` | `return_loss=True` -> `(volume, loss_history)`; `return_state=True` appends `(d_offsets, sigma)`. `return_history` still returns the legacy 4-tuple |
| `ConstraintConfig` | `mask: bool \| Tensor` | `cylinder: bool` and `support_mask: Tensor \| None` (constructor keywords and attributes) |
| `ConstraintConfig` | `sparsity` | `soft_threshold` (it collided with `RegularizerConfig.sparsity`, a loss term) |
| `fasttomo.autograd._euler_zyz_to_rotmat` | private | public `fasttomo.euler_zyz_to_rotmat`; the private name stays as an alias |
| `fasttomo.utilities.cylindrical_mask` | defined there | defined in `fasttomo.constraints`, re-exported from `fasttomo` and `fasttomo.utilities` |
| `fasttomo.configs._apply_sparsity` | | alias of `fasttomo.constraints._soft_threshold` |

**Removed.** `estimate_shifts_xcorr(upsample_factor=...)`: it never had an
effect (the sub-pixel shift comes from a parabolic fit of the correlation
peak). Passing it raises `TypeError` with that explanation.

**Behaviour changes.**
- `run_mbir` raises `TypeError` for a non-fp32 sinogram, `lambdas` or
  `warm_start` instead of casting silently, like the other fp32 algorithms.
- `run_mbir` uses `num_threads` (keyword, else `OMP_NUM_THREADS`, else 1)
  instead of a hardcoded 4 CPU threads. Results are identical; the default is
  single-threaded unless `OMP_NUM_THREADS` is set (see the warning at the top:
  pass `num_threads=4` to restore the old speed).
- `run_ad_recon`'s tqdm bar is opt-in (`progress=True`); it was always shown.
- `run_resire(mask=True, support_mask=T)` (and the old `support=T`) raises
  `ValueError`; before, `support` silently won. Same contract as every other
  algorithm: one support per reconstruction.
- `run_genfire`'s `mask` defaults to `None`: the cylinder unless
  `support_mask` is given (calls without `support_mask` are unchanged).
- `n_iter < 0`, a non-integer `n_iter`/`Nz`, a negative or non-finite
  `sparsity`, and a non-tensor `sinogram`/`angles_deg`/`warm_start` raise
  instead of running zero iterations or failing later.
- `ConstraintConfig.apply()` honours `cylinder=True` (a bool `mask` used to be
  ignored silently); `validate()` rejects `cylinder` together with
  `support_mask`.
- Warm starts are detached before use, so a `requires_grad` warm start no
  longer drags an autograd graph through SIRT and friends.
- The SIRT CUDA-graph cache and its failed-capture set live on the projector
  object (declared on the projector base class); a failed capture now only
  disables graphs for that projector, not for the whole process.
- `config.validate()` accepts numpy and one-element tensor scalars again
  (`np.int64` iteration counts, `np.float32` or tensor learning rates), which
  #82 had started to reject.

### Added
- `loss_callback` and `progress` on every iterative algorithm (SIRT, SART,
  wSIRT, FISTA, FISTA-TV, GENFIRE, RESIRE, gradient descent, AD recon, MBIR),
  with the frozen 2026-06 callback contract. Newly streamed losses: SART's
  full-sinogram data MSE per outer iteration, GENFIRE's Fourier L1 error,
  gradient descent's total loss (components `fidelity`, `tv`), MBIR's data
  MSE `mean((g - A f - d)^2)` per outer iteration (indices run across
  multires levels).
- `warm_start` on every iterative algorithm (new on wSIRT, GENFIRE, RESIRE,
  gradient descent, AD recon and MBIR).
- `support_mask` on every algorithm (new on FISTA, FISTA-TV, SART, wSIRT,
  WBP, GENFIRE, RESIRE, gradient descent and MBIR; closes #54), plus `mask`
  (cylinder) on MBIR. Where each applies it is tabulated in the
  `fasttomo.algorithms` module docstring.
- `num_threads` honoured on every CPU kernel path (the single-axis projections
  of SIRT, SART, FISTA, FISTA-TV, wSIRT, FBP, gradient descent,
  `refine_angles` and `estimate_shifts_xcorr` ignored it before).
- `fasttomo.constraints` module: `cylindrical_mask`, `apply_constraints`,
  `apply_constraints_`, and the written constraints order and support-mask
  contract.
- `MbirConfig`: optional bundle of `run_mbir`'s tuning keywords.
- `fasttomo.euler_zyz_to_rotmat` exported.
- `ruff` rule set pinned in `pyproject.toml`; the tree is clean and the CI step
  is blocking.

### Earlier changes in this cycle

**#84 wSIRT CPU/CUDA parity.** `run_weighted_backprojection` and `run_wsirt`
share one weighted back-projection: voxel-driven gather on both backends and
a padded grid (edge margin max(2, N // 16)). CPU and CUDA now agree to ~6e-4
(the CPU W-BP had 83 % one-step error vs 30 % on CUDA), and wSIRT's residual
no longer rises at `omega=1`. Output values of both functions change.

**#82 Python-side correctness.**
- `back_project`, `back_project_3d` and `back_project_resire` gain a
  keyword-only `Nx`, so `forward_project*(..., N_det != Nx)` works under
  autograd.
- `run_ad_recon` applies `ConstraintConfig.sparsity` (now `soft_threshold`)
  as a post-step soft-threshold; it was ignored. Skipped in pose-only
  iterations.
- RESIRE wrappers validate shapes (ValueError instead of out-of-bounds reads
  or silent broadcasting); outputs unchanged.
- Zero-size inputs (`n_angles`, `Ny`, `Nz`, `Nx` or `N_det` of 0) raise
  ValueError on CPU and CUDA (CPU used to return empty tensors); the
  projector constructors reject zero-size geometries.
- `run_mbir` raises "sinogram contains NaN or inf values" up front.
- Cosine LR schedules under AD staging use each optimiser's number of active
  iterations as `T_max`; every config has `validate()`, called by
  `run_ad_recon` on entry; `pose_warmup_iters > 0` without a refine flag
  raises.
- MBIR offset/variance updates: one Python implementation with fp64
  accumulation for CPU and CUDA (the private `_core.mbir_icd_sweep_cpu`
  signature lost its `d_offsets`, `sigma_min` and `estimate_*` arguments).

**#81 kernel and binding correctness.**
- CUDA `grad_rotmat` used tight box bounds and dropped face contributions
  (~100 % error on a box-filling volume); it now matches the forward and
  finite differences.
- One-shot CUDA wrappers no longer leak the cudaArray on errors or destroy
  textures while kernels run (scoped texture cache).
- CPU bindings take `data_ptr` ints like CUDA and release the GIL; strided or
  non-contiguous inputs are densified (they returned wrong numbers), every
  buffer size is validated before a pointer is taken, and the CUDA angle
  gradient no longer returns garbage for fp16 or strided volumes.
- Internal C/CUDA names use `parallel`/`rotmat` instead of `2d`/`3d`; the
  public Python API is unchanged.

**#79 staged AD pose optimisation.** `PoseRefinementConfig` gains
`pose_warmup_iters`, `alternating_period`, `pose_init_euler_deg` and
`pose_init_shifts`; defaults keep the joint loop.

**#78 consolidated correctness fixes.**
- `run_mbir` makes `lambdas` contiguous (a strided `lambdas` was read as
  dense) and validates `lambdas`, `sigma_init` and `sigma_min`.
- `run_ad_recon`'s sparsity term is `sparsity * mean(|v|)`; it was the signed
  mean, which pushed the volume negative.
- `RotmatProjector` casts rotation matrices to contiguous fp32 (fp64/fp16 or
  strided matrices were misread).
- `run_ad_recon` checks the projector output shape before the loss.

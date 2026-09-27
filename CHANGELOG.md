# Changelog

## Unreleased

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

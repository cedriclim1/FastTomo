# AGENTS.md

## Project overview

FastTomo is a C/CUDA + Python tomography reconstruction library. The native extension `fasttomo._core` is built with scikit-build-core + CMake + pybind11. `FASTTOMO_USE_CUDA` selects the backend: `AUTO` (default), `ON` (require CUDA) or `OFF` (CPU-only). `AUTO` builds CPU-only only when no nvcc is found at all. If it finds any nvcc, it enables CUDA and then aborts at configure time unless that nvcc is exactly 13.0; it does not fall back to CPU (for example, `/usr/local/cuda-13.2/bin` on `PATH` is a fatal error when no pip `nvidia-cuda-nvcc` wheel or `$CONDA_PREFIX/bin/nvcc` is found first). Pass `FASTTOMO_USE_CUDA=OFF` to force a CPU build.

## Module map

Python package `fasttomo/`:

| Module | Contents |
|---|---|
| `__init__.py` | Public API re-exports (pinned by a `dir(fasttomo)` test); torch >= 2.11 check; warns when a CPU-only build meets a CUDA machine or torch's CUDA is not 13.0. |
| `algorithms/__init__.py` | Re-exports every name of the former `algorithms.py`; its docstring holds the shared conventions (shapes, call signature, dtypes, support, constraints order, `loss_callback`, `warm_start`, `progress`, `num_threads`). |
| `algorithms/_common.py` | Named constants, `_prepare_recon` (validation, warm start, support resolution, `ConstraintConfig`), `_LossRecorder` (loss history, `loss_callback`, tqdm) and `_DeviceLossRecorder` (losses kept on the device, no per-iteration sync without a callback; used by SIRT, SART, wSIRT, FISTA, AD), deprecated-keyword helpers, `_build_fp_bp`. |
| `algorithms/analytic.py` | `run_fbp` (ramp filter, optional apodisation window), `run_weighted_backprojection` (W-BP, `_bp_gather`, weighting filter cached per geometry: up to 8 filters on their device, dropped by `release_cached_projectors()`). |
| `algorithms/algebraic.py` | `run_sirt` (+ CUDA-graph path, also with a loss), `run_sart` (fused SIRT kernels, planar normalisers, `view_order`), `run_wsirt`. |
| `algorithms/fista.py` | `run_fista`, `run_fista_tv` (TV prox: FGP / Chambolle, CUDA kernels with the PyTorch code as CPU path and reference), fused FISTA update, Lipschitz estimate (fp32 norms). |
| `algorithms/ad.py` | `run_ad_recon` (setup, `_ad_step`, `_ad_project_constraints`), `run_grad_descent`. |
| `algorithms/genfire.py` | `run_genfire`. **Frozen.** |
| `algorithms/resire.py` | `run_resire`. **Frozen.** |
| `algorithms/mbir.py` | `run_mbir` (ICD sweeps, offset/variance update, multi-resolution, Feistel voxel order). |
| `algorithms/alignment.py` | `refine_angles`. |
| `_operator.py` | The projection-operator seam: `Geometry`, `ProjectionOperator` (`fp` / `bp` / `grad_geom`, differentiable `project` / `backproject`), adapters `UncachedOperator`, `CudaCachedOperator`, `ResireOperator`, `_UserProjectorOperator`, chosen by `_make_operator`. Every projection goes through it (except MBIR's own model and GENFIRE's gridding). `_TexturePool`: the one-shot CUDA path's (functional API, FBP) process-wide pool of up to 4 texture caches, freed by `release_cached_projectors()`. |
| `_backend.py` | Device guard, stream pointer, dtype codes, input and shape guards (incl. `_check_finite_geometry`: NaN/Inf host angles raise), `_resolve_num_threads` (keyword, else the first entry of `OMP_NUM_THREADS`, else 1). |
| `autograd.py` | Raw kernel layer (`_launch`, `*_raw`), the generic `_Project` / `_BackProject` autograd Functions, RESIRE and TV-loss Functions, `euler_zyz_to_rotmat`. |
| `projection.py` | Functional projections (`forward_project`, `back_project`, `*_3d`, `*_resire`) as transient operators; `apply_shifts`, `apply_shifts_2d`. |
| `projector.py` | `ParallelProjector`, `RotmatProjector`: own one operator (persistent CUDA texture cache, or CPU kernels with `device="cpu"`). |
| `constraints.py` | `cylindrical_mask`, support-mask resolution, soft-threshold, `apply_constraints` (support, positivity, soft-threshold, in that order). |
| `regularizers.py` | `tv_loss_3d`, `tv_loss_xz`, `logsum_sparsity`, `tilt_ssim_loss`, `s3im_loss`. |
| `utilities.py` | `ramp_filter`, `estimate_shifts_xcorr`; re-exports `cylindrical_mask`, `tv_loss_3d`, `apply_shifts*` from their current homes. |
| `configs.py` | `AdReconConfig` (`OptimConfig`, `RegularizerConfig`, `PoseRefinementConfig`, `ConstraintConfig`), `MbirConfig`. |

Native code `csrc/`: `bindings.cpp` (module), `bindings_cpu.cpp`, `bindings_cuda.cpp` (the CUDA bindings compile to stubs in a CPU-only build); `cpu/` (`projection_parallel.c`, `projection_rotmat.c`, `genfire.c`, `mbir_icd.c`); `cuda/` (`projection_parallel.cu`, `projection_rotmat.cu`, `projection_resire.cu`, `genfire.cu`, `grad_angle.cu`, `tv_loss.cu`, `sirt_update.cu` (run_sirt's and run_sart's fused elementwise kernels), `fista_update.cu` (fused FISTA step / constraints / momentum), `tv_prox.cu` (run_fista_tv's TV prox), `texture_cache.cu` (cudaArray + texture caches; parallel caches pack 2 fp32 or 4 fp16 Y-slices per texel for Ny >= 2, zero-padded, `parallel_channels` in `projection_parallel.cu` is the layout rule), `mbir_icd.cu`, `scratch_pool.cuh` (header-only stream-ordered per-device scratch pool for the fixed-order two-pass reductions of `grad_rotmat` / `grad_angle`; it keeps up to 64 MiB per device reserved for the life of the process, invisible to `torch.cuda.memory_*` and not freed by `release_cached_projectors()`)).

## Frozen algorithms

GENFIRE and RESIRE are frozen: no changes to their kernels (`csrc/*/genfire.*`, `csrc/cuda/projection_resire.cu`), math, normalisation, edge handling or numerical output. Additive keyword-only options on their Python wrappers are allowed only with a test proving bit-identical output for existing calls. `tests/test_frozen_identity.py` and the RESIRE golden in `tests/test_algorithms_genfire_resire.py` enforce this; never regenerate those goldens to make a change pass.

## Building

`pyproject.toml` has no machine-specific paths; toolchain detection lives in `CMakeLists.txt` (pip `nvidia-cuda-nvcc` wheel, then `$CONDA_PREFIX/bin/nvcc`, then `PATH`).

### CPU-only (no GPU, e.g. a cloud VM or CI)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e . -Ccmake.define.FASTTOMO_USE_CUDA=OFF
```

### CUDA

Toolkit: CUDA **13.0** exactly (CMake aborts on any other version). torch: >= 2.11 with the `cu130` wheels (verified 2.12.1 and 2.14.0). Example with nvcc 13.0 from an un-activated conda env into a venv:

```bash
C=~/miniforge3/envs/FastTomo   # env that ships nvcc 13.0
CUDACXX=$C/bin/nvcc uv pip install --python .venv/bin/python -e . \
  -Ccmake.define.FASTTOMO_USE_CUDA=ON \
  -Ccmake.define.CMAKE_CUDA_COMPILER=$C/bin/nvcc \
  -Ccmake.define.CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ \
  -Ccmake.define.CUDAToolkit_ROOT=$C \
  -Ccmake.define.CMAKE_CUDA_ARCHITECTURES=120   # optional: one arch, faster compile
```

- `CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++` is needed when the conda env ships its own GCC (conda-forge GCC 14.3 here) and is not activated. CUDA 13.0 supports GCC 6-15 (CMakeLists requires >= 10 for CUDA builds), so GCC 14 itself is fine. The failure comes from mixing two GCCs: the conda nvcc's `nvcc.profile` puts the env's `bin/` first on `PATH`, so the host headers are conda's GCC 14.3 libstdc++, while the front end is told the version of the system `gcc` found on the caller's `PATH` (`--gnu_version=130300`). The result is `identifier "__is_array" is undefined` / `__type_pack_element is not a template` in CMake's compiler-id step. Pinning the host compiler to the system g++ (or activating the env) keeps headers and version consistent.
- Architectures: the CMake default `75;80;86;89;90;120` applies **only when `CMAKE_CUDA_ARCHITECTURES` is not defined and `CUDAARCHS` is unset or empty**. `-Ccmake.define.CMAKE_CUDA_ARCHITECTURES=...` wins over `CUDAARCHS`. An **activated** conda env with `cuda-nvcc` exports its own `CUDAARCHS` (`75-real;80-real;86-real;89-real;90-real;100-real;103-real;120-real;121`), which replaces the FastTomo default; `unset CUDAARCHS` (or pass the define) to get FastTomo's list. `CUDAARCHS` is honoured only by CMake >= 3.20. The value is cached in the build directory: wipe a persistent build dir when changing architectures (pip/uv builds use a fresh dir unless `-Cbuild-dir` is given).
- C sources are compiled with `-O3 -ffp-contract=off` and no `-march=native`, so binaries are portable. For a local build tuned to this CPU, add `-Ccmake.define.FASTTOMO_MARCH_NATIVE=ON`.
- `-ffp-contract=off` (both builds) forbids fused multiply-add contraction, so a `FASTTOMO_MARCH_NATIVE=ON` build gives **the same bits** as the portable one (verified: all CPU projection kernels memcmp-equal over 11 shapes x 1/3/8/16 threads, and the CPU goldens, GENFIRE and MBIR included, pass exactly with a native build). Without it, FMA contraction changed the FP/BP/rotmat results in the last bits and `grad_rotmat` by up to 3.4e-2 of its max at trilinear kinks, and the goldens' arithmetic fingerprint (torch-only) cannot detect that. The default x86-64 target has no FMA, so the flag does not change the default build's results. Do not remove it; new C code must not rely on contraction.
- CPU benchmarks must record whether the build used `FASTTOMO_MARCH_NATIVE=ON`. The flag changes CPU kernel speed (AVX2/AVX-512 SIMD in the Y-batched kernels: 1.1-2x at 32x256^2), not results. Builds before PR #76 always compiled C sources with `-march=native`, so older CPU results in `benchmarks/results/` are not comparable with a default build.
- After modifying C/C++/CUDA source in `csrc/`, re-run the install command to recompile. Check `python -c "import fasttomo._core as c; print(c.cuda_available)"`.
- Existing editable installs made before the package split (PR #86: `fasttomo/algorithms.py` became the package `fasttomo/algorithms/`) must be reinstalled with the same command: scikit-build-core's editable finder records module paths at install time, so an old install still maps `fasttomo.algorithms` to the deleted file and `import fasttomo` fails. The same applies whenever a module becomes a package or is renamed.

## Testing

Layout, by topic:

| File | Covers |
|---|---|
| `test_projection.py` | Functional projections: shapes, adjointness, autograd, dtypes, error messages. |
| `test_projector.py` | `ParallelProjector` / `RotmatProjector`, texture caches, CUDA graphs. |
| `test_parity.py` | Operator adapters (CPU vs CUDA-cached vs CUDA-uncached) and algorithms CPU vs CUDA. |
| `test_algorithms_analytic.py`, `_algebraic.py`, `_fista.py`, `_ad.py`, `_mbir.py`, `_genfire_resire.py`, `_alignment.py` | Per-family behaviour and input handling. |
| `test_contracts.py` | The uniform API over one `ALGOS` capability table and the frozen `loss_callback` / `support_mask` contracts the desktop app relies on. |
| `test_regularizers.py` | Regularisers and the TV loss. |
| `test_parallel_cuda_kernels.py` | Parallel-beam CUDA FP/BP: CUDA-resident NaN angles, packed textures vs per-slice results (bit for bit), the fp16 non-finite report word, the one-shot texture pool (reuse, bound, two streams, threads). |
| `test_kernel_correctness.py` | Kernel-level correctness: finite-difference geometry gradients, strided inputs, size checks, leak-free failure paths of the scoped texture cache. |
| `test_cpu_kernels.py` | CPU C/OpenMP kernels: NaN/Inf angles (raise; the raw kernels survive them), the Y-batched kernels bit for bit against fp32 NumPy ports of the scalar loops, thread invariance, `grad_angle` against an fp64 evaluation. |
| `test_frozen_identity.py`, `test_algorithm_identity.py` | Bit-identity goldens (`tests/data/*.pt`). |
| `test_wsirt.py` | wSIRT CPU/CUDA parity and convergence. |
| `test_wheel_smoke.py` | Import and `_core` smoke test (run by the wheel workflow; strict CUDA check with `FASTTOMO_EXPECT_CUDA=1`). |

- `tests/conftest.py`: fixtures `backend` (`cpu` / `cuda`, the CUDA case skips without a GPU) and `dtype` (fp32 / fp16, filtered by the `DTYPE_CAPS` table via `@pytest.mark.op("<name>")`); helpers `make_angles`, `phantom_2d`, `phantom_3d`, `rel_err`, `rel_l2`, `needs_cuda`, `needs_2_gpus`. It tags every CUDA test `gpu` and every two-GPU test `multi_gpu` (markers registered in `pyproject.toml`).
- Full suite on one GPU: `CUDA_VISIBLE_DEVICES=<idx> python -m pytest tests -q`. About 19 tests skip on one card (17 cross-device tests need 2 visible GPUs, 1 needs scikit-image, 1 needs `FASTTOMO_EXPECT_CUDA`).
- CPU only: `CUDA_VISIBLE_DEVICES= python -m pytest tests -q` (the `gpu` tests skip themselves), or `pytest tests -m "not gpu"` to deselect them.
- With a non-editable install, run the `pytest` entry point rather than `python -m pytest` from the repo root, otherwise the source tree (without a compiled `_core`) shadows the installed package.
- `tests/sanitizer_sweep.py` (not collected) is the size sweep for `compute-sanitizer` memcheck/racecheck; run it on a free GPU after any CUDA kernel change (`python tests/sanitizer_sweep.py tv` or `solvers` runs only the TV-loss or the SIRT / SART / FISTA / TV-prox kernels, `parallel` the parallel-beam FP/BP and texture-pool variants).

### Golden files and the arithmetic fingerprint

Exact (`torch.equal`) golden checks only run where the machine's arithmetic fingerprint (`tests/_golden.py`: a fixed probe through matmul, reductions, FFTs, transcendentals, fp16 rounding, interpolation, `grid_sample` and one backward pass, hashed with the CPU SIMD level and `OMP_NUM_THREADS`, or the GPU compute capability and CUDA version) is one stored in the golden under `_arith_fingerprint`. Elsewhere the test falls back to its documented tolerance and emits a `GoldenNotExactWarning` (see pytest's warnings summary; it names the fingerprint and the CPU model). Outputs that go through an FFT (ramp filter, FBP, W-BP, wSIRT, GENFIRE) are compared on that fallback relative to the tensor's largest value (`_golden.peak_close`), because FFT round-off follows the transform's norm rather than each bin's size. Atomics-based CUDA outputs are always compared at a tolerance.

Re-stamp procedure (a torch upgrade, a new GPU or CPU, or a change to `_probe` changes the fingerprint):

1. Run the golden tests (`test_algorithm_identity.py`, `test_frozen_identity.py`, `test_algorithms_genfire_resire.py`, the MBIR sweep golden in `test_algorithms_mbir.py`) with `OMP_NUM_THREADS` unset and confirm they pass at tolerance.
2. `python tests/_golden.py stamp tests/data/<golden>.pt` (adds this environment's CPU and, if a GPU is visible, CUDA fingerprint; only the metadata changes, and after saving it re-loads the file and asserts the tensors are byte-identical to before).
3. Re-run the same tests: they now take the exact branch. If any fails, this environment is not bit-identical: `git checkout tests/data/<golden>.pt` and leave it unstamped.
4. Commit only the metadata change and say in the PR body which environment was added.

Regenerating golden tensors (not just stamping) is a different operation: use the generator in `tests/data/*_gen.py` (`algorithm_golden_main_gen.py`, `frozen_golden_main_gen.py`, `mbir_sweep_golden_gen.py`) against the commit named in its docstring; each calls `stamp()` after saving. `resire_golden_main.pt` has no generator script: it is `torch.save(_resire_cases())` from `tests/test_algorithms_genfire_resire.py`, run against the commit named in the comment above `RESIRE_GOLDEN`, then stamped. Never regenerate the frozen GENFIRE/RESIRE goldens to make a change pass.

## GPU etiquette (shared machine)

On a shared multi-GPU machine (the development box has 4 cards used by several people), use only a card that shows **0% utilization and no other user's processes** in both `nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv` and `nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv`; a card with foreign memory allocated counts as in use. Pin with `CUDA_VISIBLE_DEVICES=<idx>`, re-check before every GPU-heavy run (builds do not need a GPU), and stop and wait if another user appears on the chosen card.

## CI

- `.github/workflows/test-cpu.yml`: on every pull request and push to `main`, builds CPU-only on Python 3.10 and 3.12 and runs `pytest tests`. Superseded pull-request runs are cancelled; in-progress runs on `main` are not, but a queued run on `main` can be replaced by a newer one, so a commit in a burst of merges may go untested. `ruff check fasttomo tests` must pass (ruff version pinned in the workflow, rule set in `pyproject.toml`).
- `.github/workflows/wheels.yml`: on `v*` tags and manual dispatch, builds CPU and CUDA 13.0 manylinux_2_28 wheels for cp310-cp314 (CUDA archs `75;80;86;89;90;100;120`, no `-march=native`).

## Linting

`ruff check fasttomo tests`. The rule set is pinned in `pyproject.toml` (`[tool.ruff.lint]`: ruff's classic E4/E7/E9/F defaults) and the tree is clean; CI fails on new findings.

## Key gotchas

- `python3-dev` (apt) is required for the CMake build to find `Python Development.Module` when using the system Python.
- On machines without a GPU, install the CPU-only PyTorch build (`--index-url https://download.pytorch.org/whl/cpu`) to avoid pulling ~2 GB of CUDA libraries.
- The `support_mask` and `loss_callback` keyword names and semantics are frozen (a desktop app probes `inspect.signature` for them); `test_contracts.py` guards them.
- CPU thread pools: the C kernels use `num_threads` (keyword, else `OMP_NUM_THREADS`, else **1**), while torch's element-wise ops between them use `torch.get_num_threads()` (every core by default), and both share one libgomp runtime. Mismatched counts oversubscribe the cores: CPU `run_sirt` at 8x128^2 x 61 views, 16 pinned cores, 1.7 ms/iter with both at 16 vs 20.8 ms/iter with torch at 128 (the phase-1b audit measured 7.3 vs 59.6 ms/iter before the Y-batched kernels). For CPU solver loops and CPU benchmarks set `torch.set_num_threads(n)`, `num_threads=n` (or `OMP_NUM_THREADS=n`) to the same `n`, pin with `taskset` and `OMP_PROC_BIND=close`, and record `n` and `FASTTOMO_MARCH_NATIVE`.
- CPU kernel changes: the parallel-beam CPU kernels are Y-batched (`FT_YB` slices per SIMD loop, `csrc/cpu/common.h`) and must stay bit-identical to the per-slice scalar loops (same fp32 operations in the same order per lane; ray clipping only drops steps the in-loop test rejects). `tests/test_cpu_kernels.py` pins this against NumPy ports; a larger memcmp matrix against a previous build is in the phase-1b PR A report. Keep new C code free of FMA-contraction assumptions (`-ffp-contract=off`).

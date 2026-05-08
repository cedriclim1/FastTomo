# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FastTomo is a high-performance tomography package for electron tomography (parallel-beam geometry). The stack layers C/CUDA kernels under a PyTorch API — every public function accepts and returns `torch.Tensor` (float32), and all gradient computation lives in PyTorch.

**Target algorithms:** FBP, SIRT/SART, GENFIRE, RESIRE, FISTA, ASD-POCS.

**Currently implemented:** `run_fbp`, `run_sirt`, `run_sart`, `run_wsirt` (Weighted SIRT), `run_genfire`, `run_weighted_backprojection`, `run_resire`, `refine_angles`. FISTA and ASD-POCS are not yet implemented.

**3D rotation-matrix projection (CUDA only):** `forward_project_3d` / `back_project_3d` accept a `[n_angles, 3]` ZYZ Euler table (or a `[n_angles, 3, 3]` matrix tensor via the helper) and call CUDA `tex3D`-based kernels. `run_sirt`, `run_resire`, and `refine_angles` accept an optional `euler_zyz_deg=` argument that swaps the inner FP/BP pair to the 3D projector — required when angles are non-single-axis (in-plane rotation, dual-axis, refined ZYZ triples). See `fasttomo-wiki/wiki/concepts/zyz-euler-rotmat.md`.

## Conda Environment

```bash
conda activate FastTomo
```

## Architecture

```
csrc/include/proj_geom.h         shared ProjGeom2D/3D geometry structs (plain C)
csrc/cpu/common.h                SAMP3 trilinear sampler macro
csrc/cpu/projection_2d.c         2D scalar FP/BP/grad (bilinear ray integration)
csrc/cpu/projection_3d.c         3D rotmat FP/BP/grad_rotmat (trilinear, OpenMP opt-in)
csrc/cpu/genfire.c               CPU GENFIRE Fourier gridding
csrc/cuda/common.cuh             shared device helpers: float_to_t, load_trig_smem
csrc/cuda/projection_2d.cu       scalar single-axis FP/BP (fp32/fp16/packed-Y, cached+uncached)
csrc/cuda/projection_3d.cu       rotmat FP/BP + grad_rotmat (fp32/fp16, cached+uncached)
csrc/cuda/projection_resire.cu   RESIRE voxel-driven scatter FP + gather BP
csrc/cuda/genfire.cu             CUDA GENFIRE Fourier gridding (atomicAdd)
csrc/cuda/grad_angle.cu          scalar-angle gradient kernel
csrc/cuda/texture_cache.cu       TextureCache lifecycle; raw CUDA texture API confined here
csrc/bindings_cpu.cpp            CPU pybind11 wrappers → register_cpu(m)
csrc/bindings_cuda.cpp           CUDA pybind11 wrappers → register_cuda(m)
csrc/bindings.cpp                PYBIND11_MODULE: calls register_cpu + register_cuda
fasttomo/autograd.py             torch.autograd.Function wrappers; _assert_cuda, _ensure_fp32
fasttomo/projection.py           forward_project, back_project, *_3d, *_resire public wrappers
fasttomo/utilities.py            cylindrical_mask, ramp_filter, apply_shifts, estimate_shifts_xcorr
fasttomo/algorithms.py           run_fbp, run_sirt, run_wsirt, run_genfire, run_resire, refine_angles, …
fasttomo/projector.py            ParallelProjector, RotmatProjector (persistent TextureCache owners)
fasttomo/__init__.py             re-exports public API
```

**Key architectural constraint:** Keep C/CUDA implementations decoupled from Python. New kernels go in `csrc/`, CPU wrappers in `csrc/bindings_cpu.cpp`, CUDA wrappers in `csrc/bindings_cuda.cpp`, then a Python wrapper in the appropriate `fasttomo/` module (`projection.py`, `utilities.py`, or `algorithms.py`).

### Kernel strategies

| Op | CPU | CUDA |
|----|-----|------|
| Forward projection (scalar angle) | Ray-driven; 1-px steps + bilinear interp | Detector-driven; `tex2DLayered` (volume layer per Y-slice) |
| Back-projection (scalar angle) | Voxel-driven (gather); bilinear interp | Voxel-driven (gather); `tex2DLayered` |
| Forward projection (rotation matrix) | Detector-driven; trilinear interp (`SAMP3`) | Detector-driven; `tex3D` (full trilinear) |
| Back-projection (rotation matrix) | Voxel-driven (gather); bilinear sinogram interp | Voxel-driven; `tex2DLayered` (sinogram layer per angle) + shared-memory matrices |
| GENFIRE gridding | Scatter (sequential) | Scatter with atomicAdd |

**CPU vs CUDA back-projection numerical difference:** The two BP implementations are intentionally not expected to produce identical numbers and have no CPU/CUDA cross-consistency test. Only CUDA FP is tested against CPU FP (within 1e-4). Do not interpret CPU/CUDA BP disagreement as a bug.

### CUDA binding pattern

CPU ops accept numpy arrays; CUDA ops accept raw device pointer integers (`tensor.data_ptr()`). The `fasttomo/projection.py` / `autograd.py` layer handles this split — callers never see it.

```python
# CPU path: pass numpy arrays
_core.forward_project_cpu(vol_np, angles_np, N_det)

# CUDA path: pass data_ptr() ints + explicit shape args
_core.forward_project_cuda(vol.data_ptr(), sino.data_ptr(), angles_np, Ny, Nz, Nx, n_angles, N_det)
```

## Tensor Conventions

- **Volume:** `[Ny, Nz, Nx]` — Ny is the tilt axis (independent slices), Nz is depth, Nx is lateral
- **Sinogram:** `[n_angles, Ny, N_det]`
- **Angles:** 1-D tensor in **degrees** — converted to radians internally before calling C/CUDA
- **dtype:** always `float32`; inputs are cast before passing to `_core`
- Pixel coordinate centering: index `i` maps to coordinate `i − (N−1)/2`

## Build & Test

**Full build (editable install via scikit-build-core):**
```bash
pip install -e .
```

**Faster C/CUDA iteration without reinstall:**
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
# copy the .so manually if testing without pip install:
cp build/_core*.so fasttomo/
```

**Two conda envs to keep in sync (`FastTomo` and `FastTomo-bench`).** Both
run Python 3.14, so a single build artefact works for both — but each env
has its own `site-packages/fasttomo/_core.*.so` and editable installs do not
auto-refresh it after a rebuild. After any change in `csrc/`, run:

```bash
conda run -n FastTomo-bench cmake --build build --parallel
cp build/_core*.so fasttomo/
cp build/_core*.so /home/cedlim/miniforge3/envs/FastTomo-bench/lib/python3.14/site-packages/fasttomo/
cp build/_core*.so /home/cedlim/miniforge3/envs/FastTomo/lib/python3.14/site-packages/fasttomo/
```

Verify both with `python -c "import fasttomo._core as c; print('num_threads' in c.forward_project_cpu.__doc__)"` (or whatever signature change you just made) — both should report the same answer.

CMake auto-detects CUDA via `check_language(CUDA)` — no manual changes needed. Default CUDA architectures: `70 80 86 120`. Override with `-DCMAKE_CUDA_ARCHITECTURES="80;86"`.

**Hard requirements (CUDA build):** PyTorch ≥ 2.11, CUDA **13.0 exactly** (PyTorch 2.11 ships only against CUDA 13.0 — CMake aborts if `CUDAToolkit_VERSION` ≠ 13.0), host GCC ≥ 10. There is no longer an auto-fallback to `/usr/bin/g++-13`; if conda-forge GCC 14.x trips the `__type_pack_element` issue, pin `gxx_linux-64=13` in the env or pass `--config-settings=cmake.define.CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13` explicitly. When `nvcc` is detectable but the toolchain is broken, the build now **fails fast** at configure time (use `FASTTOMO_USE_CUDA=OFF` to opt out for CPU-only).

**Tests:**
```bash
pytest tests/                                          # all tests
pytest tests/test_cpu.py                               # CPU only
pytest tests/test_cuda.py                              # CUDA (auto-skipped if no GPU)
pytest tests/test_cpu.py::TestClassName::test_method   # single test
```

## Domain Vocabulary

- **Sinogram** — 2D array of line-integral measurements `[detector × angle]`
- **Forward projection** — maps volume → sinogram (simulates acquisition)
- **Back-projection** — adjoint of forward projection; smears sinogram back into volume space
- **FBP** — filter sinogram with ramp filter |ξ|, then back-project; scale by π/n_angles
- **SIRT** — iterative: `x += ω · C · Aᵀ · R · (b − Ax)`, C and R are voxel/ray normalisers
- **W-SIRT / W-BP** — replace BP with exact-filter back-projection (Wolf et al. 2014); WF numerically computed from PSF of a point source
- **GENFIRE** — Fourier gridding of oversampled 1D FFTs onto a 2D grid, then iterate between Fourier data enforcement and real-space constraints (support, positivity, resolution extension schedule)
- **Missing wedge** — angular gap in tilt series where no data was acquired; causes artifacts along the beam direction

## Wiki

Algorithm literature lives in `fasttomo-wiki/`. **Always read `fasttomo-wiki/SCHEMA.md` before any wiki operation.**

- `fasttomo-wiki/raw/` — immutable source papers/articles/code
- `fasttomo-wiki/wiki/` — LLM-maintained markdown (algorithms, concepts, analyses, code)
- To ingest a paper: `"ingest raw/papers/<filename>"`
- To query: ask any question; read `wiki/index.md` first
- To lint: `"lint wiki"`

## License

GPLv3 — all derivative works must remain open source.

## Adding Messages to Commits, PRs, etc...

Do not include Co-authored by Claude anywhere in commits or any related things.
# AGENTS.md

## Project overview

FastTomo is a C/CUDA + Python CT reconstruction library. The native extension (`_core`) is built with scikit-build-core + CMake + pybind11. `FASTTOMO_USE_CUDA` selects the backend: `AUTO` (default), `ON` (require CUDA) or `OFF` (CPU-only). `AUTO` builds CPU-only only when no nvcc is found at all. If it finds any nvcc, it enables CUDA and then aborts at configure time unless that nvcc is exactly 13.0; it does not fall back to CPU (for example, `/usr/local/cuda-13.2/bin` on `PATH` is a fatal error). Pass `FASTTOMO_USE_CUDA=OFF` to force a CPU build.

## Building

`pyproject.toml` has no machine-specific paths; toolchain detection lives in `CMakeLists.txt` (pip `nvidia-cuda-nvcc` wheel, then `$CONDA_PREFIX/bin/nvcc`).

### CPU-only (no GPU, e.g. a cloud VM or CI)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e . -Ccmake.define.FASTTOMO_USE_CUDA=OFF
```

### CUDA

The toolkit must be CUDA **13.0** exactly (CMake aborts on any other version). Example with nvcc 13.0 from a conda env:

```bash
C=~/miniforge3/envs/FastTomo   # env that ships nvcc 13.0
CUDACXX=$C/bin/nvcc uv pip install --python .venv/bin/python -e . \
  -Ccmake.define.FASTTOMO_USE_CUDA=ON \
  -Ccmake.define.CMAKE_CUDA_COMPILER=$C/bin/nvcc \
  -Ccmake.define.CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ \
  -Ccmake.define.CUDAToolkit_ROOT=$C
```

- `CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++` is needed when the conda env ships its own GCC (conda-forge GCC 14.3 here) and is not activated. CUDA 13.0 supports GCC 6-15 (CMakeLists requires >= 10), so GCC 14 itself is fine. The failure comes from mixing two GCCs: the conda nvcc's `nvcc.profile` puts the env's `bin/` first on `PATH`, so the host headers are conda's GCC 14.3 libstdc++, while the front end is told the version of the system `gcc` found on the caller's `PATH` (`--gnu_version=130300`). The result is `identifier "__is_array" is undefined` / `__type_pack_element is not a template` in CMake's compiler-id step. Pinning the host compiler to the system g++ keeps the headers and version consistent.
- The default architecture list is `75;80;86;89;90;120`. To build for one GPU only (faster), add `-Ccmake.define.CMAKE_CUDA_ARCHITECTURES=120` or set the `CUDAARCHS` environment variable (an empty `CUDAARCHS` counts as unset). The value is cached in the build directory: wipe the build dir when changing architectures.
- C sources are compiled with `-O3` and no `-march=native`, so binaries are portable. For a local build tuned to this CPU, add `-Ccmake.define.FASTTOMO_MARCH_NATIVE=ON`.
- CPU benchmarks must record whether the build used `FASTTOMO_MARCH_NATIVE=ON`. The flag changes CPU kernel speed (AVX2/FMA auto-vectorization). Builds before PR #76 always compiled C sources with `-march=native`, so older CPU results in `benchmarks/results/` are not comparable with a default build.
- After modifying C/C++/CUDA source in `csrc/`, re-run the install command to recompile.
- Existing editable installs made before the package split (PR #86: `fasttomo/algorithms.py` became the package `fasttomo/algorithms/`) must be reinstalled with the same command: scikit-build-core's editable finder records module paths at install time, so an old install still maps `fasttomo.algorithms` to the deleted file and `import fasttomo` fails. The same applies whenever a module becomes a package or is renamed.

## Testing

- Tests are grouped by topic: `test_projection.py`, `test_projector.py`, `test_parity.py` (the projection-operator adapters CPU <-> CUDA cached <-> CUDA uncached, and algorithms CPU <-> CUDA), `test_algorithms_{analytic,algebraic,fista,ad,mbir,genfire_resire,alignment}.py`, `test_contracts.py` (the uniform API and the frozen `loss_callback` / `support_mask` contracts), `test_regularizers.py`, `test_kernel_correctness.py`, `test_frozen_identity.py` and `test_algorithm_identity.py` (bit-identity goldens; see `tests/data/*_gen.py` to regenerate), `test_wsirt.py`, `test_wheel_smoke.py`. Shared fixtures (`backend` = cpu/cuda, `dtype` = fp32/fp16 with a per-op capability table) and helpers (`make_angles`, `phantom_2d`, `phantom_3d`, `rel_err`, `rel_l2`) live in `tests/conftest.py`, which also tags every CUDA test with the `gpu` marker (`multi_gpu` for the cross-device ones).
- Full suite on one GPU: `CUDA_VISIBLE_DEVICES=<idx> python -m pytest tests -q` gives 938 passed, 19 skipped (17 of the skips are cross-device tests that need 2 visible GPUs, 1 needs scikit-image, 1 needs `FASTTOMO_EXPECT_CUDA`).
- CPU only: `CUDA_VISIBLE_DEVICES= python -m pytest tests -q` gives 494 passed, 430 skipped (the `gpu` tests skip themselves); `pytest tests -m "not gpu"` runs the same 494 without listing the GPU skips (2 remain: scikit-image, `FASTTOMO_EXPECT_CUDA`).
- With a non-editable install, run the `pytest` entry point rather than `python -m pytest` from the repo root, otherwise the source tree (without a compiled `_core`) shadows the installed package.

## CI

- `.github/workflows/test-cpu.yml`: on every pull request and push to `main`, builds CPU-only on Python 3.10 and 3.12 and runs `pytest tests`. Superseded pull-request runs are cancelled; runs on `main` are not. `ruff check fasttomo tests` must pass (rule set pinned in `pyproject.toml`).
- `.github/workflows/wheels.yml`: on `v*` tags and manual dispatch, builds CPU and CUDA 13.0 manylinux wheels for cp310-cp313.

## Linting

`ruff check fasttomo tests`. The rule set is pinned in `pyproject.toml` (`[tool.ruff.lint]`: ruff's classic E4/E7/E9/F defaults) and the tree is clean; CI fails on new findings.

## Key gotchas

- `python3-dev` (apt) is required for the CMake build to find `Python Development.Module` when using the system Python.
- On machines without a GPU, install the CPU-only PyTorch build (`--index-url https://download.pytorch.org/whl/cpu`) to avoid pulling ~2 GB of CUDA libraries.

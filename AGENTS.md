# AGENTS.md

## Project overview

FastTomo is a C/CUDA + Python CT reconstruction library. The native extension (`_core`) is built with scikit-build-core + CMake + pybind11. `FASTTOMO_USE_CUDA` selects the backend: `AUTO` (default, uses CUDA if an nvcc 13.0 toolchain is found), `ON` (require CUDA) or `OFF` (CPU-only).

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

- `CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++` is required when the conda env ships its own GCC 14: nvcc 13.0 cannot parse GCC 14's libstdc++ headers, and the conda nvcc puts the env's `bin/` first on `PATH`.
- The default architecture list is `80;86;89;90;120`. To build for one GPU only (faster), add `-Ccmake.define.CMAKE_CUDA_ARCHITECTURES=120` (or set the `CUDAARCHS` environment variable).
- C sources are compiled with `-O3` and no `-march=native`, so binaries are portable. For a local build tuned to this CPU, add `-Ccmake.define.FASTTOMO_MARCH_NATIVE=ON`.
- After modifying C/C++/CUDA source in `csrc/`, re-run the install command to recompile.

## Testing

- Full suite on one GPU: `CUDA_VISIBLE_DEVICES=<idx> python -m pytest tests -q` gives 232 passed, 18 skipped (17 of the skips are cross-device tests that need 2 visible GPUs). `test_genfire_reconstruction_quality` passes.
- CPU-only build: `FASTTOMO_EXPECT_CUDA=0 pytest tests -q` gives 109 passed, 141 skipped; the CUDA tests (`tests/test_cuda.py`, `tests/test_projector_cache.py` and CUDA cases elsewhere) skip themselves.
- With a non-editable install, run the `pytest` entry point rather than `python -m pytest` from the repo root, otherwise the source tree (without a compiled `_core`) shadows the installed package.

## CI

- `.github/workflows/test-cpu.yml`: on every pull request and push to `main`, builds CPU-only on Python 3.12 and runs `pytest tests`. `ruff check fasttomo tests` runs as a non-blocking step.
- `.github/workflows/wheels.yml`: on `v*` tags and manual dispatch, builds CPU and CUDA 13.0 manylinux wheels for cp310-cp313.

## Linting

`ruff check fasttomo tests` (no ruff config in the repo yet). There are pre-existing warnings, so CI does not fail on it.

## Key gotchas

- `python3-dev` (apt) is required for the CMake build to find `Python Development.Module` when using the system Python.
- On machines without a GPU, install the CPU-only PyTorch build (`--index-url https://download.pytorch.org/whl/cpu`) to avoid pulling ~2 GB of CUDA libraries.

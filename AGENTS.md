# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

FastTomo is a C/CUDA + Python CT reconstruction library. The native extension (`_core`) is built with scikit-build-core + CMake + pybind11. No GPU is available in the cloud VM, so the library builds and runs in CPU-only mode.

### Building

The `pyproject.toml` contains hardcoded `CMAKE_CUDA_COMPILER` and `pybind11_DIR` paths from the original developer's machine. To install in editable mode on the cloud VM, override them:

```bash
pip install --break-system-packages -e . --no-build-isolation \
  -Ccmake.define.CMAKE_CUDA_COMPILER=CMAKE_CUDA_COMPILER-NOTFOUND \
  -Ccmake.define.pybind11_DIR="$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())')"
```

### Testing

- **CPU tests:** `python3 -m pytest tests/test_cpu.py -v` — 34/35 pass; `test_genfire_reconstruction_quality` is a pre-existing numerical tolerance failure (NMSE ~0.34 vs 0.15 threshold).
- **CUDA tests:** `tests/test_cuda.py` and `tests/test_projector_cache.py` require an NVIDIA GPU and will be skipped/fail in this environment.

### Linting

No linter is configured in the repo. `ruff check fasttomo/ tests/` can be used; there are ~25 pre-existing style warnings (unused variables, etc.).

### Key gotchas

- `python3-dev` (apt) is required for the CMake build to find `Python Development.Module`.
- PyTorch must be the CPU-only build (`--index-url https://download.pytorch.org/whl/cpu`) to avoid pulling ~2 GB of CUDA libraries.
- After modifying C/C++ source in `csrc/`, you must re-run the editable install command above to recompile.

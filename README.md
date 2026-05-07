# FastTomo

Ground-up fast tomography implementations in C and CUDA with direct workflows in PyTorch.

All public functions accept and return `torch.Tensor` (float32). Forward and back-projection kernels run on CPU (OpenMP) and CUDA (texture-memory). Implemented algorithms: FBP, SIRT, Weighted SIRT, GENFIRE, RESIRE, weighted back-projection, and angular refinement.

## Requirements

- Python ≥ 3.9
- A C/C++ compiler with OpenMP (Linux: gcc; macOS: `brew install libomp`)
- CMake ≥ 3.18
- PyTorch (`pip install torch`)
- **For the CUDA backend only:** `nvcc` on `PATH` and matching CUDA runtime headers. The simplest way is conda: `conda install -c nvidia cuda-nvcc cuda-cudart-dev`.

The build auto-detects CUDA: if `nvcc` is found it builds the CUDA backend, otherwise it falls back to a CPU-only build silently.

## Install

### CPU-only

```bash
pip install -e .
# or
uv pip install -e .
```

### CUDA (from source)

```bash
# Ensure nvcc is on PATH (system toolkit, or `conda install -c nvidia cuda-nvcc cuda-cudart-dev`)
pip install -e .
```

To force the CUDA build to fail loudly if no toolkit is present (instead of silently falling back to CPU):

```bash
pip install -e . --config-settings=cmake.define.FASTTOMO_USE_CUDA=ON
```

To force a CPU-only build on a CUDA-equipped machine:

```bash
pip install -e . --config-settings=cmake.define.FASTTOMO_USE_CUDA=OFF
```

Override CUDA architectures (defaults to `70;80;86;120`):

```bash
pip install -e . --config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES="80;86;89"
```

### As a `uv` dependency

In your downstream project's `pyproject.toml`:

```toml
[project]
dependencies = ["fasttomo"]

[tool.uv.sources]
fasttomo = { git = "https://github.com/cedriclim1/FastTomo" }
```

Then `uv sync`. `uv` will build from source on first install and cache the wheel.

## Verify

```bash
python -c "import fasttomo, fasttomo._core as c; print('CUDA:', c.cuda_available)"
pytest tests/                    # all
pytest tests/test_cpu.py         # CPU only
pytest tests/test_cuda.py        # CUDA (auto-skipped if no GPU)
```

## Faster local iteration

For fast C/CUDA iteration without a full reinstall:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
cp build/_core*.so fasttomo/
```

## License

GPLv3 — all derivative works must remain open source.

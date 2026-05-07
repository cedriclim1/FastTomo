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

## Releasing (maintainer notes)

Wheels are built by `.github/workflows/wheels.yml`:

- **CPU manylinux wheels** for cp39–cp313 × x86_64
- **CUDA wheel** for cp39–cp313, with `libcudart` excluded from the wheel via `auditwheel --exclude` and resolved at import time against the `libcudart` that PyTorch loads into the process. One CUDA wheel works across PyTorch's CUDA versions (the TorchRadon pattern). Built with the **CUDA 13.0 toolkit** (`nvcc 13.0`, `cudart 13.0`); the CI test step verifies the wheel under `torch+cu130`. Because `libcudart` is excluded, the same wheel works for users on `torch+cu12x` as well as `torch+cu13x` — the toolkit choice only affects the minimum NVIDIA driver and the targeted CUDA architectures. Volta (`sm_70` / V100) is **not** in the wheel since CUDA 13 dropped Volta; V100 users should build from source with a CUDA 12.x toolkit.
- **Sdist** for source-builds when no wheel matches.

### One-time setup

1. **PyPI Trusted Publishing** — on PyPI, go to your project's Settings → Publishing, add a publisher: GitHub repo `cedriclim1/FastTomo`, workflow `wheels.yml`, environment `pypi`. Repeat on TestPyPI with environment `testpypi`.
2. **GitHub environments** — in repo Settings → Environments, create `pypi` and `testpypi` (no secrets needed; OIDC handles auth).

### Cutting a release

```bash
# Bump version in pyproject.toml, commit, then tag:
git tag v0.1.0
git push --tags
```

The tag push triggers the full matrix and publishes to PyPI. To dry-run against TestPyPI: use the workflow_dispatch UI on GitHub Actions and pick `publish_to = testpypi`.

## License

GPLv3 — all derivative works must remain open source.

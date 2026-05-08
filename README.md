# FastTomo

Ground-up fast tomography implementations in C and CUDA with direct workflows in PyTorch.

All public functions accept and return `torch.Tensor` (float32). Forward and back-projection kernels run on CPU (OpenMP) and CUDA (texture-memory). Implemented algorithms: FBP, SIRT, Weighted SIRT, GENFIRE, RESIRE, weighted back-projection, and angular refinement.

## Requirements

- Python ≥ 3.10
- A C/C++ compiler with OpenMP (Linux: **GCC ≥ 10**; macOS: `brew install libomp`)
- CMake ≥ 3.18
- PyTorch ≥ 2.11 (`pip install 'torch>=2.11'`) — bundles `cudart 13.0`
- **For the CUDA backend only:** `nvcc` **exactly 13.0** reachable either on `PATH` or inside the active conda env (`$CONDA_PREFIX/bin/nvcc`), with matching CUDA runtime headers.

The build auto-detects CUDA. If `nvcc` is reachable, it builds the CUDA backend. If `nvcc` is *not* present anywhere CMake can find it, the build falls back to CPU-only with a loud multi-line `WARNING`. If `nvcc` *is* visible but the toolchain is broken (e.g. host-compiler / CUDA mismatch), the build now **fails at configure time** rather than silently producing a CPU-only wheel — pass `--config-settings=cmake.define.FASTTOMO_USE_CUDA=OFF` to opt out explicitly. If a CPU-only build is later imported on a CUDA-equipped machine, `import fasttomo` raises a `RuntimeWarning` once.

> **Supported CUDA toolkit:** **CUDA 13.0 exactly.** PyTorch 2.11's wheels are built only against CUDA 13.0; an `nvcc 13.x` newer than 13.0 (e.g. 13.2) produces PTX that errors at runtime with `the provided PTX was compiled with an unsupported toolchain`. The build now hard-fails at configure time if `CUDAToolkit_VERSION` ≠ 13.0.
>
> **Host compiler:** **GCC ≥ 10** for CUDA builds (NVIDIA's CUDA 13 install guide supports GCC 6–15; FastTomo standardises on the 10+ range). The previous auto-downgrade to `/usr/bin/g++-13` has been removed — if your conda-forge GCC 14.x clashes with `nvcc 13.0` (the well-known `__type_pack_element` bug), explicitly pin `gxx_linux-64=13` in your env or pass `--config-settings=cmake.define.CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13`.

## Install

### CPU-only

```bash
pip install -e .
# or
uv pip install -e .
```

### CUDA (from source)

Recommended: use a fresh conda env so version pins don't conflict with prior installs.

```bash
conda create -n FastTomo python=3.12 -y
conda activate FastTomo

# GCC >= 10 satisfies CUDA 13.0; pin to <=13 if you hit the conda-forge
# GCC 14.x / nvcc 13.0 `__type_pack_element` issue.
conda install -c conda-forge gxx_linux-64 gcc_linux-64 cmake -y
conda install -c nvidia 'cuda-nvcc=13.0' 'cuda-cudart-dev=13.0' -y

pip install 'torch>=2.11' numpy tqdm scikit-build-core pybind11 packaging
pip install -e . --no-build-isolation
```

Then verify:

```bash
python -c "import fasttomo._core as c; print('CUDA:', c.cuda_available)"  # expect: True
pytest tests/test_cuda.py
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

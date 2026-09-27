# FastTomo

Ground-up fast tomography implementations in C and CUDA with direct workflows in PyTorch.

Every public function takes and returns `torch.Tensor`. Projection and back-projection kernels run on the CPU (C/OpenMP) and on CUDA (texture memory). Volumes are `[Ny, Nz, Nx]`, sinograms `[n_angles, Ny, N_det]` and angles a 1-D tensor in degrees; `Ny` runs along the tilt axis. fp32 is supported everywhere; the parallel and 3D projections (functional and the projector classes) and FBP, SIRT, SART, FISTA and FISTA-TV also accept fp16 (output in the input dtype). An fp16 result beyond the fp16 range (65504) raises `OverflowError` instead of returning `inf` (the kernels accumulate in fp32; scale unnormalised data into range or use fp32). Other dtypes raise `TypeError`; nothing is cast silently.

## Algorithms

All are exported from `fasttomo` (and `fasttomo.algorithms`).

| Algorithm | Function | Backends | One line |
|---|---|---|---|
| FBP | `run_fbp` | CPU, CUDA | Ramp-filtered back-projection (fp32/fp16); optional `window="hann"` / `"hamming"` apodisation. |
| Weighted back-projection | `run_weighted_backprojection` (alias `run_wbp`) | CPU, CUDA | Exact W-BP (Wolf et al. 2014): back-projection deconvolved by the numerically computed weighting filter. |
| SIRT | `run_sirt` | CPU, CUDA | Simultaneous iterative reconstruction; the CUDA path replays the loop as a CUDA graph. |
| wSIRT | `run_wsirt` | CPU, CUDA | Weighted SIRT (Wolf et al. 2014): SIRT with the W-BP as back-projector. |
| SART / OS-SART | `run_sart` | CPU, CUDA | Ordered-subset SART; `subset_size=1` is plain SART; `view_order="golden"` visits the views in golden-ratio order. |
| FISTA | `run_fista` | CPU, CUDA | Accelerated projected gradient on the least-squares data term. |
| FISTA-TV | `run_fista_tv` | CPU, CUDA | FISTA with a TV-L2 proximal step (FGP or Chambolle, `tv_inner_iter` inner iterations; fused CUDA kernels); `tv_weight` is required. |
| GENFIRE | `run_genfire` | CPU, CUDA | Fourier-gridding iterative reconstruction (Pryor et al. 2017). Frozen numerics. |
| RESIRE | `run_resire` | CUDA | Real-space iterative reconstruction with the RESIRE supersampled projectors. Frozen numerics. |
| MBIR | `run_mbir` | CPU, CUDA | Model-based MAP reconstruction (MBIR-AM-PE): qGGMRF prior, generalised-Huber likelihood, ICD sweeps, per-view offset and noise estimation, optional multi-resolution. |
| AD reconstruction | `run_ad_recon` | CUDA | Autograd reconstruction with regularisers and optional per-view pose/shift refinement, configured by `AdReconConfig`. |
| Gradient descent | `run_grad_descent` | CPU, CUDA | Reference autograd reconstruction with any `torch.optim` optimiser (Adam default, SGD gives Landweber, LBFGS works). |
| Angular refinement | `refine_angles` | CPU, CUDA | Brute-force per-view search (single tilt axis or ZYZ Euler) against a volume. Not a reconstruction; returns refined angles. |

Also exported: the functional projectors (`forward_project`, `back_project`, `forward_project_3d`, `back_project_3d`, `forward_project_resire`, `back_project_resire`), the cached projector classes `ParallelProjector` and `RotmatProjector`, the regularisers (`tv_loss_3d`, `tv_loss_xz` (alias `tv_loss_zx`), `tilt_ssim_loss`, `s3im_loss`, `logsum_sparsity`), helpers (`ramp_filter`, `cylindrical_mask`, `apply_shifts`, `apply_shifts_2d`, `estimate_shifts_xcorr`, `euler_zyz_to_rotmat`) and the config dataclasses (`AdReconConfig`, `OptimConfig`, `RegularizerConfig`, `PoseRefinementConfig`, `ConstraintConfig`, `MbirConfig`).

### Call convention

Every reconstruction is called as

```python
run_x(sinogram, angles_deg, n_iter, *, ...)
```

The sinogram, the angles and (for iterative methods) `n_iter` are positional; everything else is keyword-only. The direct methods `run_fbp` and `run_weighted_backprojection` have no `n_iter`, and `run_genfire` takes `n_iter` as a keyword for one release (its third positional used to be `Nz`). Shared keywords, where the algorithm supports them:

| Keyword | Meaning |
|---|---|
| `Nz` | Volume depth; defaults to `N_det`. |
| `warm_start` | Initial volume `[Ny, Nz, Nx]` in the sinogram's dtype (default zeros); copied, never modified. |
| `support_mask` | Arbitrary support, `[Nz, Nx]`, `[1, Nz, Nx]` or `[Ny, Nz, Nx]`, values in `{0, 1}`. |
| `mask` | `True` restricts the volume to the inscribed cylinder of the XZ plane. Mutually exclusive with `support_mask`. |
| `loss_callback` | Called once per iteration as `loss_callback(i, loss)`; a callback with a `components` keyword also gets the per-term breakdown. |
| `progress` | `True` shows a tqdm bar (off by default). |
| `return_loss` | Return `(volume, loss_history)` instead of `volume`. |
| `num_threads` | OpenMP threads for the CPU kernels: the keyword, else `OMP_NUM_THREADS`, else 1. Ignored on CUDA. |

Exceptions, from the signatures:

- `run_fbp` and `run_weighted_backprojection` are direct methods: no `n_iter`, `warm_start`, `loss_callback`, `progress` or `return_loss`.
- `run_genfire` and `run_resire` have no `num_threads` (RESIRE is CUDA-only).
- `run_ad_recon` has no `mask` (use `ConstraintConfig.cylinder` in its `config`) and no `num_threads` (CUDA-only).
- `refine_angles` is not a reconstruction: `refine_angles(volume, sinogram, angles_deg, *, euler_zyz_deg, search_range_deg, n_steps, num_threads)`, with none of the other shared keywords.

The full contract (constraint order, which loss each algorithm records, where the support is applied) is in the `fasttomo.algorithms` module docstring. Renamed keywords keep working for one release with a `DeprecationWarning`; see `CHANGELOG.md`.

```python
import torch
import fasttomo as ft

angles = torch.linspace(-70, 70, 61)
volume_true = torch.rand(8, 128, 128)                    # [Ny, Nz, Nx]
sino = ft.forward_project(volume_true, angles)           # [61, 8, 128]

vol = ft.run_fbp(sino, angles)
vol, loss = ft.run_sirt(sino, angles, 50, positivity=True, mask=True,
                        return_loss=True, num_threads=8)
vol = ft.run_fista_tv(sino, angles, 100, tv_weight=1e-3, warm_start=vol,
                      loss_callback=lambda i, l: print(i, l))
```

Move `sino` and `angles` to a CUDA device to run the same calls on the GPU.

### Frozen algorithms

GENFIRE and RESIRE numerics are kept identical to the original implementations: their kernels, math, normalisation and edge handling are not changed, and their outputs are checked against stored goldens (`tests/test_frozen_identity.py`, the RESIRE golden in `tests/test_algorithms_genfire_resire.py`). GENFIRE on the CPU and the deterministic RESIRE paths (back-projection, exactly representable cases) are compared bit for bit (on machines whose arithmetic fingerprint matches the golden's, see Testing); GENFIRE on CUDA and RESIRE's forward projection and reconstruction outputs are compared at 1e-5, because their kernels accumulate with fp32 atomics and are not bit-reproducible run to run. Only additive keyword-only options (for example `loss_callback`, `warm_start`, `support_mask`) have been added, each with a test proving existing calls are unchanged.

## Requirements

- Python >= 3.10
- A C/C++ compiler with OpenMP (Linux: GCC >= 10; macOS: `brew install libomp`)
- CMake >= 3.18 (>= 3.20 if you select architectures with the `CUDAARCHS` environment variable)
- PyTorch >= 2.11 with the CUDA 13.0 (`cu130`) wheels for a CUDA build. Verified with torch 2.12.1 and 2.14.0 (`+cu130`).
- For the CUDA backend only: `nvcc` **exactly 13.0** with matching runtime headers, found as a pip `nvidia-cuda-nvcc` wheel, in the active conda env (`$CONDA_PREFIX/bin/nvcc`), or on `PATH`.

### How the build picks a backend

`FASTTOMO_USE_CUDA` is `AUTO` (default), `ON` or `OFF`.

- `AUTO` builds CPU-only only when no `nvcc` is found anywhere, and then prints a loud `WARNING`.
- If any `nvcc` is found, CUDA is enabled, and configure **aborts** unless the toolkit is exactly 13.0 (for example, `/usr/local/cuda-13.2/bin` first on `PATH` is a fatal error). A newer 13.x `nvcc` produces PTX that torch's CUDA 13.0 runtime rejects (`the provided PTX was compiled with an unsupported toolchain`).
- A broken toolchain (for example a host-compiler mismatch) also fails at configure time instead of silently producing a CPU-only build. Pass `-Ccmake.define.FASTTOMO_USE_CUDA=OFF` to build CPU-only on purpose.
- A CPU-only build imported on a machine where `torch.cuda.is_available()` is true raises a `RuntimeWarning` once.

## Install

### CPU-only

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # skips ~2 GB of CUDA libraries
pip install -e . -Ccmake.define.FASTTOMO_USE_CUDA=OFF
pip install pytest          # to run the tests
```

### CUDA (from source)

With `nvcc` 13.0 from a conda env:

```bash
conda create -n FastTomo python=3.12 -y
conda activate FastTomo
conda install -c conda-forge gxx_linux-64 gcc_linux-64 cmake -y
conda install -c nvidia 'cuda-nvcc=13.0' 'cuda-cudart-dev=13.0' -y

pip install 'torch>=2.11' --index-url https://download.pytorch.org/whl/cu130
pip install numpy tqdm packaging scikit-build-core pybind11 pytest
pip install -e . --no-build-isolation -Ccmake.define.FASTTOMO_USE_CUDA=ON
```

Or, without activating the env (for example from a separate venv), point CMake at the env's `nvcc` and pin the host compiler:

```bash
C=~/miniforge3/envs/FastTomo          # any env that ships nvcc 13.0
CUDACXX=$C/bin/nvcc pip install -e . \
  -Ccmake.define.FASTTOMO_USE_CUDA=ON \
  -Ccmake.define.CMAKE_CUDA_COMPILER=$C/bin/nvcc \
  -Ccmake.define.CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ \
  -Ccmake.define.CUDAToolkit_ROOT=$C
```

**Host compiler.** CUDA 13.0 supports GCC 6-15 and FastTomo requires GCC >= 10 for CUDA builds, so GCC 14 itself is fine. What fails is *mixing* two GCCs: a conda `nvcc` puts its env's `bin/` first on `PATH`, so from an un-activated env the host headers come from conda's GCC 14 libstdc++ while the front end is told the version of the system `gcc`. The symptom is `identifier "__is_array" is undefined` / `__type_pack_element is not a template` during CMake's compiler check. Either activate the env (so its own GCC is used throughout) or pass `-Ccmake.define.CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++`.

**CUDA architectures.** When neither `CMAKE_CUDA_ARCHITECTURES` nor a non-empty `CUDAARCHS` environment variable is set, the build targets `75;80;86;89;90;120` (Turing to Blackwell; sm_70 is not available in CUDA 13). To override:

```bash
pip install -e . -Ccmake.define.CMAKE_CUDA_ARCHITECTURES=120    # one GPU only, faster compile
CUDAARCHS=86 pip install -e .                                     # same, via the environment
```

`-Ccmake.define.CMAKE_CUDA_ARCHITECTURES` wins over `CUDAARCHS`, and an empty `CUDAARCHS` counts as unset. Note that an **activated conda env with `cuda-nvcc` exports its own `CUDAARCHS`** (`75-real;80-real;...;120-real;121`), which replaces FastTomo's default; `unset CUDAARCHS` before installing to get the default list. The value is cached in the CMake build directory, so wipe a persistent build directory when changing architectures.

**CPU tuning.** C sources are compiled with `-O3 -ffp-contract=off` and no `-march=native`, so binaries are portable. For a local build tuned to the build machine's CPU add `-Ccmake.define.FASTTOMO_MARCH_NATIVE=ON`. It changes CPU kernel speed but not the results (no FMA contraction, so the bits match the portable build), so CPU benchmarks must record whether it was on (builds before PR #76 always used `-march=native`).

**After pulling.** Re-run the install command after changing anything in `csrc/`. Editable installs made before `fasttomo/algorithms.py` became the package `fasttomo/algorithms/` (PR #86) must be reinstalled: scikit-build-core's editable finder records module paths at install time, so an old install still points at the deleted file and `import fasttomo` fails. The same applies whenever a module is renamed or turned into a package.

Verify the build:

```bash
python -c "import fasttomo, fasttomo._core as c; print('CUDA:', c.cuda_available)"   # expect True
```

For faster rebuilds, keep the CMake build directory between installs with `-Cbuild-dir=build/cuda`; only changed sources are recompiled.

### As a `uv` dependency

In your downstream project's `pyproject.toml`:

```toml
[project]
dependencies = ["fasttomo"]

[tool.uv.sources]
fasttomo = { git = "https://github.com/cedriclim1/FastTomo" }
```

Then `uv sync`. `uv` builds from source on first install and caches the wheel.

## Testing

```bash
pytest tests                          # everything; CUDA tests skip themselves without a GPU
pytest tests -m "not gpu"             # CPU only
CUDA_VISIBLE_DEVICES=1 pytest tests   # pick one card on a multi-GPU machine
```

Tests are grouped by topic (`tests/test_projection.py`, `test_algorithms_*.py`, `test_contracts.py`, ...); `tests/conftest.py` tags every CUDA test with the `gpu` marker (`multi_gpu` for the tests that need two cards). The golden-identity tests compare against stored outputs with `torch.equal` only where the machine's arithmetic fingerprint (`tests/_golden.py`) matches one recorded in the golden file, and otherwise fall back to a documented tolerance with a `GoldenNotExactWarning`. With a non-editable install, run the `pytest` entry point rather than `python -m pytest` from the repo root, or the source tree (without a compiled `_core`) shadows the installed package.

## Continuous integration

- `.github/workflows/test-cpu.yml` runs on every pull request and on pushes to `main`: a CPU-only build on Python 3.10 and 3.12, the full `pytest tests` (CUDA tests skip), and `ruff check fasttomo tests`. Superseded pull-request runs are cancelled. In-progress runs on `main` are not, but a queued run can be replaced by a newer one, so not every single push to `main` is guaranteed its own run.
- `.github/workflows/wheels.yml` runs on `v*` tags and manual dispatch (see below).

## Releasing (maintainer notes)

Wheels are built by `.github/workflows/wheels.yml`:

- **CPU manylinux_2_28 wheels** for cp310-cp314, x86_64, built without `-march=native`.
- **CUDA wheels** for cp310-cp314, built with the CUDA 13.0 toolkit for architectures `75;80;86;89;90;100;120`, for use with the `cu130` torch wheels. `libcudart` is excluded from the wheel (`auditwheel --exclude`) and resolved at import time against the `libcudart` that PyTorch loaded; the test step loads the wheel under `torch+cu130`. `import fasttomo` raises `ImportError` for torch < 2.11 and emits a `RuntimeWarning` when a CUDA build meets a torch whose CUDA version is not 13.0, since mixed runtimes can crash at the first kernel launch. Volta (sm_70, V100) is not included because CUDA 13 dropped it; V100 users must build from source with a CUDA 12.x toolkit.
- **Sdist** for source builds when no wheel matches.

### One-time setup

1. **PyPI Trusted Publishing**: on PyPI, go to your project's Settings -> Publishing and add a publisher: GitHub repo `cedriclim1/FastTomo`, workflow `wheels.yml`, environment `pypi`. Repeat on TestPyPI with environment `testpypi`.
2. **GitHub environments**: in repo Settings -> Environments, create `pypi` and `testpypi` (no secrets needed; OIDC handles auth).

### Cutting a release

```bash
# Bump version in pyproject.toml, commit, then tag:
git tag v0.1.0
git push --tags
```

The tag push triggers the full matrix and publishes to PyPI. To dry-run against TestPyPI, use the workflow_dispatch UI on GitHub Actions and pick `publish_to = testpypi`.

## License

GPLv3: all derivative works must remain open source.

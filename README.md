# FastTomo

Ground-up fast tomography operators and reconstructions in C/CUDA with direct workflows in PyTorch.

## Installation

### Quick start (recommended)

Install from PyPI:

`pip install fasttomo`

This is the easiest path for first-time users and should be preferred over local builds.

### Build from source (CPU-only)

If you want editable installs or are developing locally:

1. Create and activate a Python environment.
2. Install CPU PyTorch first (recommended for CPU-only machines):
   - `pip install torch --index-url https://download.pytorch.org/whl/cpu`
3. Install FastTomo:
   - `pip install -e .`

FastTomo automatically builds in CPU mode when no CUDA compiler is found.

### Build from source (CUDA, advanced)

CUDA source builds are intended for advanced users with a working CUDA toolchain.

Typical requirements:

- NVIDIA driver + matching CUDA toolkit
- CMake and a supported C/C++ compiler
- Python headers/dev package for your platform

Then run:

`pip install -e .`

If CMake cannot find your CUDA compiler automatically, pass it explicitly via CMake args.

## Verify installation

Run a quick smoke test:

`python -c "import torch, fasttomo as ft; vol=torch.zeros(1,16,16); ang=torch.linspace(-60,60,12); print(ft.forward_project(vol, ang).shape)"`

Expected output:

`torch.Size([12, 1, 16])`

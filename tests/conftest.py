"""
Shared fixtures, helpers and markers for the FastTomo test suite.

Fixtures
    ``backend``  ``torch.device("cpu")`` and ``torch.device("cuda")``; the
                 CUDA case skips itself when no GPU is visible. Test IDs read
                 ``test_x[cpu]`` / ``test_x[cuda]``.
    ``dtype``    ``torch.float32`` and ``torch.float16`` (IDs ``fp32`` /
                 ``fp16``). A test marked ``@pytest.mark.op("<name>")`` (or
                 parametrised over an ``op`` argument) skips the combinations
                 that ``DTYPE_CAPS`` says the op does not support on the
                 current ``backend``.

Helpers (plain functions, ``from conftest import ...``)
    ``make_angles``, ``phantom_2d``, ``phantom_3d``, ``rel_err`` (max-abs
    relative error), ``rel_l2`` (relative L2 error), ``needs_cuda``,
    ``needs_2_gpus``, ``HAS_CUDA``.

Markers (registered in pyproject.toml)
    ``gpu``        added automatically to every test that needs CUDA (a
                   ``cuda`` parameter, or a skip condition naming CUDA or a
                   GPU), so ``pytest -m "not gpu"`` selects the CPU tests.
    ``multi_gpu``  added automatically to tests that need two GPUs.
    ``op``         names the operation whose dtype capabilities apply.
"""

from __future__ import annotations

import pytest
import torch

HAS_CUDA = torch.cuda.is_available()
N_GPUS = torch.cuda.device_count() if HAS_CUDA else 0

needs_cuda = pytest.mark.skipif(not HAS_CUDA, reason="no CUDA GPU available")
needs_2_gpus = pytest.mark.skipif(N_GPUS < 2, reason="needs >=2 GPUs")

BACKENDS = ["cpu", pytest.param("cuda", marks=needs_cuda)]


# ── dtype capability table ─────────────────────────────────────────────────
#
# Which input dtypes each operation accepts, per backend (fp16 in -> fp16
# out). Anything not listed raises TypeError. Mirrors the library's dtype
# policy: fp32 everywhere; fp16 for the projection ops and for FBP, SIRT,
# SART, FISTA and FISTA-TV; RESIRE, GENFIRE, W-BP, wSIRT, grad descent, AD
# and MBIR are fp32-only.

_F32 = (torch.float32,)
_F32_F16 = (torch.float32, torch.float16)

DTYPE_CAPS: dict[str, dict[str, tuple[torch.dtype, ...]]] = {
    "forward_project": {"cpu": _F32_F16, "cuda": _F32_F16},
    "back_project": {"cpu": _F32_F16, "cuda": _F32_F16},
    "forward_project_3d": {"cpu": _F32_F16, "cuda": _F32_F16},
    "back_project_3d": {"cpu": _F32_F16, "cuda": _F32_F16},
    "forward_project_resire": {"cuda": _F32},
    "back_project_resire": {"cuda": _F32},
    "run_fbp": {"cpu": _F32_F16, "cuda": _F32_F16},
    "run_weighted_backprojection": {"cpu": _F32, "cuda": _F32},
    "run_sirt": {"cpu": _F32_F16, "cuda": _F32_F16},
    "run_sart": {"cpu": _F32_F16, "cuda": _F32_F16},
    "run_wsirt": {"cpu": _F32, "cuda": _F32},
    "run_fista": {"cpu": _F32_F16, "cuda": _F32_F16},
    "run_fista_tv": {"cpu": _F32_F16, "cuda": _F32_F16},
    "run_genfire": {"cpu": _F32, "cuda": _F32},
    "run_resire": {"cuda": _F32},
    "run_grad_descent": {"cpu": _F32, "cuda": _F32},
    "run_ad_recon": {"cuda": _F32},
    "run_mbir": {"cpu": _F32, "cuda": _F32},
}


def dtype_supported(op: str, backend: str, dtype: torch.dtype) -> bool:
    """True when ``DTYPE_CAPS`` lists ``dtype`` for ``op`` on ``backend``."""
    return dtype in DTYPE_CAPS[op].get(backend, ())


@pytest.fixture(params=BACKENDS)
def backend(request) -> torch.device:
    return torch.device(request.param)


@pytest.fixture(params=[torch.float32, torch.float16], ids=["fp32", "fp16"])
def dtype(request) -> torch.dtype:
    marker = request.node.get_closest_marker("op")
    callspec = getattr(request.node, "callspec", None)
    op = marker.args[0] if marker is not None else (
        callspec.params.get("op") if callspec is not None else None)
    if op is not None:
        be = (request.getfixturevalue("backend").type
              if "backend" in request.fixturenames else "cpu")
        if be not in DTYPE_CAPS[op]:
            pytest.skip(f"{op} has no {be} path")
        if not dtype_supported(op, be, request.param):
            pytest.skip(f"{op} does not support {request.param} on {be}")
    return request.param


# ── shared data helpers ────────────────────────────────────────────────────

def make_angles(n: int, max_deg: float = 70.0, device="cpu") -> torch.Tensor:
    """n evenly spaced angles in [-max_deg, +max_deg) in degrees."""
    return torch.linspace(-max_deg, max_deg, n + 1)[:-1].to(device)


def phantom_2d(N: int, device="cpu") -> torch.Tensor:
    """Simple square phantom centred in an N×N volume slice."""
    vol = torch.zeros(N, N, device=device)
    q   = N // 4
    vol[q:3*q, q:3*q] = 1.0
    return vol


def phantom_3d(N: int, device="cpu") -> torch.Tensor:
    """Centred cube of ones in an N×N×N volume."""
    vol = torch.zeros(N, N, N, device=device)
    vol[N//4:3*N//4, N//4:3*N//4, N//4:3*N//4] = 1.0
    return vol


def gpu_contention(device=None) -> str | None:
    """Why wall-clock timings on ``device`` are not trustworthy right now, or
    None: another process has a compute context on this GPU (the check the
    workspace's GPU guard makes; the utilisation figure is not used, since
    this process's own earlier work shows up in it). Uses ``nvidia-smi``;
    without it (or on error) returns None."""
    import os
    import shutil
    import subprocess
    if not HAS_CUDA or shutil.which("nvidia-smi") is None:
        return None
    try:
        torch.cuda.synchronize(device)
        uuid = str(torch.cuda.get_device_properties(device).uuid)
        uuid = uuid if uuid.startswith("GPU-") else "GPU-" + uuid

        def query(*args):
            return subprocess.run(["nvidia-smi", *args, "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True, timeout=10,
                                  check=True).stdout.splitlines()
        pids = [line.split(",")[1].strip() for line in query(
                    "--query-compute-apps=gpu_uuid,pid")
                if line.split(",")[0].strip() == uuid]
        foreign = [p for p in pids if p != str(os.getpid())]
        if foreign:
            return f"GPU {uuid} is shared with other processes (pids {', '.join(foreign)})"
    except (OSError, subprocess.SubprocessError, ValueError, IndexError, RuntimeError):
        return None
    return None


def rel_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Max-abs error relative to the max-abs of ``expected``."""
    diff = (actual.float() - expected.float()).abs().max().item()
    scale = expected.abs().max().item() + 1e-12
    return diff / scale


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative L2 error ``||a - b|| / ||b||`` in fp64 on the CPU."""
    a = a.detach().double().cpu()
    b = b.detach().double().cpu()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


# ── automatic gpu / multi_gpu markers ──────────────────────────────────────

def _needs(item) -> tuple[bool, bool]:
    gpu = multi = False
    callspec = getattr(item, "callspec", None)
    if callspec is not None:
        for value in callspec.params.values():
            if value == "cuda" or (isinstance(value, torch.device) and value.type == "cuda"):
                gpu = True
    for mark in item.iter_markers(name="skipif"):
        reason = str(mark.kwargs.get("reason", ""))
        if "CUDA" in reason or "GPU" in reason:
            gpu = True
        if "2 GPU" in reason:
            multi = True
    return gpu, multi


def pytest_collection_modifyitems(config, items):
    for item in items:
        gpu, multi = _needs(item)
        if gpu:
            item.add_marker(pytest.mark.gpu)
        if multi:
            item.add_marker(pytest.mark.multi_gpu)

"""Smoke test for installed wheels.

Run after `pip install` (not via -e) to assert:
  - the package imports
  - the compiled extension loads
  - cuda_available matches the FASTTOMO_EXPECT_CUDA env var if set
  - a tiny FBP roundtrip runs without crashing on whichever backend is active

This is what CI runs against a freshly-built wheel in a clean env that has only
torch installed (no CUDA toolkit) — it's the assertion that the Torch-bundled-
cudart linkage actually works at runtime.
"""

import os

import fasttomo._core as _core
import pytest
import torch

import fasttomo as ft


def test_package_imports():
    assert hasattr(_core, "cuda_available")


def test_split_modules_import():
    import fasttomo.algorithms as algorithms
    import fasttomo.projection as projection
    import fasttomo.utilities as utilities

    assert hasattr(projection, "forward_project")
    assert hasattr(utilities, "ramp_filter")
    assert hasattr(algorithms, "run_sirt")


def test_algorithm_subpackage_is_installed():
    """fasttomo/algorithms/ is a package: the wheel must ship its modules."""
    import importlib

    for mod in ("_common", "analytic", "algebraic", "fista", "ad", "genfire",
                "resire", "alignment", "mbir"):
        importlib.import_module(f"fasttomo.algorithms.{mod}")
    importlib.import_module("fasttomo._backend")


def test_cuda_available_matches_env():
    expect = os.environ.get("FASTTOMO_EXPECT_CUDA")
    if expect is None:
        pytest.skip("FASTTOMO_EXPECT_CUDA not set; skipping strict check")
    expected = expect.strip().lower() in ("1", "true", "yes", "on")
    assert _core.cuda_available is expected, (
        f"Wheel built with CUDA={_core.cuda_available}, "
        f"but FASTTOMO_EXPECT_CUDA={expect!r}"
    )


def test_tiny_fbp_roundtrip():
    """One small FBP run. Uses CUDA if available and a GPU is present, else CPU."""
    use_cuda = _core.cuda_available and torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"

    N = 32
    n_angles = 30
    angles = torch.linspace(-70.0, 70.0, n_angles + 1)[:-1].to(device)

    vol = torch.zeros(1, N, N, device=device)
    q = N // 4
    vol[0, q:3 * q, q:3 * q] = 1.0

    sino = ft.forward_project(vol, angles)
    rec = ft.run_fbp(sino, angles)

    assert rec.shape == vol.shape
    assert torch.isfinite(rec).all()


def test_version_comes_from_the_distribution_metadata():
    """``fasttomo.__version__`` is the installed distribution's version, i.e.
    ``[project] version`` of pyproject.toml (the single source)."""
    from importlib.metadata import version

    assert isinstance(ft.__version__, str) and ft.__version__
    assert ft.__version__ == version("fasttomo")

"""Local multi-Python smoke harness for FastTomo.

Spins up a fresh venv per Python version (3.10 - 3.13), builds the C/CUDA
extension against that interpreter, and runs ``tests/test_wheel_smoke.py`` to
catch ABI/linking and type-annotation regressions before tagging a release.

Usage
-----

Requires ``nox`` and ``uv`` on PATH (``pipx install nox uv``).

Provision interpreters once per machine::

    uv python install 3.10 3.11 3.12 3.13

Run the smoke matrix (CPU build, default)::

    nox -s smoke

Run against a single version::

    nox -s smoke-3.12

Run with the CUDA backend (only on a host with the CUDA 13 toolkit)::

    FASTTOMO_NOX_CUDA=1 nox -s smoke
"""

from __future__ import annotations

import os

import nox

PY_VERSIONS = ["3.10", "3.11", "3.12", "3.13"]

USE_CUDA = os.environ.get("FASTTOMO_NOX_CUDA", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

nox.options.default_venv_backend = "uv"
nox.options.sessions = ["smoke"]


@nox.session(python=PY_VERSIONS, reuse_venv=False)
def smoke(session: nox.Session) -> None:
    """Build FastTomo against ``session.python`` and run the wheel smoke test."""
    torch_index = (
        "https://download.pytorch.org/whl/cu130"
        if USE_CUDA
        else "https://download.pytorch.org/whl/cpu"
    )
    session.install("torch", "--index-url", torch_index)

    session.install(
        "scikit-build-core>=0.3",
        "pybind11",
        "cmake>=3.18",
        "ninja",
        "pytest",
    )

    cmake_flag = "ON" if USE_CUDA else "OFF"
    session.install(
        ".",
        "--no-build-isolation",
        "--config-settings=cmake.define.FASTTOMO_USE_CUDA=" + cmake_flag,
        env={"FASTTOMO_USE_CUDA": cmake_flag},
        silent=False,
    )

    session.run(
        "pytest",
        "tests/test_wheel_smoke.py",
        "-v",
        env={"FASTTOMO_EXPECT_CUDA": "1" if USE_CUDA else "0"},
    )

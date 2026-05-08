"""
Root conftest.py for the worktree.

Redirects fasttomo module imports to the worktree's local copies so that
tests run against the modified code rather than the editable install's
hardcoded paths into /home/cedlim/FastTomo.

The scikit-build editable install inserts a ScikitBuildRedirectingFinder
at sys.meta_path[0] that unconditionally redirects fasttomo.* imports to
the main repo. We replace its known_source_files mapping with paths that
point to this worktree instead.
"""
import sys
import pathlib

_WORKTREE_ROOT = pathlib.Path(__file__).parent.resolve()
_FASTTOMO_DIR = _WORKTREE_ROOT / "fasttomo"

# Names of the Python source modules that exist in the worktree fasttomo/
_LOCAL_MODULES = {
    "fasttomo": str(_FASTTOMO_DIR / "__init__.py"),
    "fasttomo.algorithms": str(_FASTTOMO_DIR / "algorithms.py"),
    "fasttomo.autograd": str(_FASTTOMO_DIR / "autograd.py"),
    "fasttomo.projection": str(_FASTTOMO_DIR / "projection.py"),
    "fasttomo.projector": str(_FASTTOMO_DIR / "projector.py"),
    "fasttomo.utilities": str(_FASTTOMO_DIR / "utilities.py"),
}

# Find and patch the ScikitBuildRedirectingFinder in sys.meta_path
for _finder in sys.meta_path:
    if (
        hasattr(_finder, "known_source_files")
        and isinstance(_finder.known_source_files, dict)
        and "fasttomo" in _finder.known_source_files
    ):
        _finder.known_source_files.update(_LOCAL_MODULES)
        break

# Also update sys.path so that a plain import (bypassing meta_path finders)
# also resolves to the worktree.
_wt = str(_WORKTREE_ROOT)
if _wt not in sys.path:
    sys.path.insert(0, _wt)

# Flush any cached fasttomo modules so they get re-imported fresh.
for _key in list(sys.modules.keys()):
    if _key == "fasttomo" or _key.startswith("fasttomo."):
        del sys.modules[_key]

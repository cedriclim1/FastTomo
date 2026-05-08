"""
Root conftest.py for the worktree.

When pytest is invoked from this worktree directory, Python's editable-install
meta-path finder (ScikitBuildRedirectingFinder) hardcodes source-file paths
pointing to the main repository at /home/cedlim/FastTomo/fasttomo/.  This
conftest overrides those entries so the finder resolves Python sources from
THIS worktree instead, while still using the shared compiled _core extension.

This runs via pytest_configure (called before any test module is imported).
"""
from __future__ import annotations

import os
import sys

_WORKTREE = os.path.dirname(os.path.abspath(__file__))
_MAIN_REPO = "/home/cedlim/FastTomo"

_REDIRECTABLE_MODULES = [
    "fasttomo",
    "fasttomo.algorithms",
    "fasttomo.projector",
    "fasttomo.autograd",
    "fasttomo.projection",
    "fasttomo.utilities",
]


def pytest_configure(config) -> None:  # noqa: ANN001
    """Redirect fasttomo Python source imports to this worktree before
    any test file is collected or executed."""
    # Find the ScikitBuildRedirectingFinder that the editable install injected.
    for finder in sys.meta_path:
        if type(finder).__name__ == "ScikitBuildRedirectingFinder":
            _patch_skbuild_finder(finder)
            break


def _patch_skbuild_finder(finder) -> None:  # noqa: ANN001
    """Rewrite the known_source_files map to point at the worktree."""
    for module_name in _REDIRECTABLE_MODULES:
        if module_name not in finder.known_source_files:
            continue
        orig_path: str = finder.known_source_files[module_name]
        # Replace the main-repo prefix with the worktree prefix.
        if orig_path.startswith(_MAIN_REPO):
            new_path = _WORKTREE + orig_path[len(_MAIN_REPO):]
            if os.path.isfile(new_path):
                finder.known_source_files[module_name] = new_path
                # Evict any already-imported copy so the finder reloads from
                # the worktree path on next access.
                sys.modules.pop(module_name, None)

    # Rebuild submodule_search_locations based on updated known_source_files.
    for module_name, src_path in finder.known_source_files.items():
        parent = ".".join(module_name.split(".")[:-1])
        if not parent:
            continue
        parent_path = os.path.dirname(src_path)
        if not os.path.isabs(parent_path):
            parent_path = os.path.join(finder.dir, parent_path)
        finder.submodule_search_locations.setdefault(parent, set())
        finder.submodule_search_locations[parent].add(parent_path)

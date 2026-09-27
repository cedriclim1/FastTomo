"""Import-migration regression tests."""

from __future__ import annotations

import subprocess
import sys
from textwrap import dedent


def test_import_survives_stale_editable_algorithms_mapping():
    """ARCH-P8 moved fasttomo.algorithms from module -> package.

    Simulate a stale scikit-build editable finder that still redirects
    ``fasttomo.algorithms`` to ``.../algorithms.py`` and assert importing
    fasttomo repairs that mapping instead of crashing.
    """
    script = dedent(
        r"""
        import importlib.util
        import pathlib
        import sys

        spec = importlib.util.find_spec("fasttomo")
        assert spec is not None and spec.origin is not None
        pkg_dir = pathlib.Path(spec.origin).resolve().parent
        old_algorithms = str(pkg_dir / "algorithms.py")
        new_algorithms_init = (pkg_dir / "algorithms" / "__init__.py").resolve()

        class ScikitBuildRedirectingFinder:
            def __init__(self):
                self.known_source_files = {"fasttomo.algorithms": old_algorithms}
                self.submodule_search_locations = {}

            def find_spec(self, fullname, path=None, target=None):
                if fullname not in self.known_source_files:
                    return None
                redir = self.known_source_files[fullname]
                sub = self.submodule_search_locations.get(fullname)
                return importlib.util.spec_from_file_location(
                    fullname,
                    redir,
                    submodule_search_locations=sub,
                )

        finder = ScikitBuildRedirectingFinder()
        sys.meta_path.insert(0, finder)

        import fasttomo  # noqa: F401
        import fasttomo.algorithms as algorithms

        assert pathlib.Path(algorithms.__file__).resolve() == new_algorithms_init
        assert finder.known_source_files["fasttomo.algorithms"].endswith(
            "algorithms/__init__.py"
        )
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr

"""Generate tests/data/algorithm_golden_main.pt (read by tests/test_algorithm_identity.py).

Run against origin/main's ``fasttomo/`` at 5362a6b (before the
projection-operator refactor), e.g. from a scratch export of that commit
with the built ``_core`` extension copied in. Needs a CUDA GPU for the
``cuda/`` keys:

    PYTHONPATH=<export-of-5362a6b> python tests/data/algorithm_golden_main_gen.py \
        tests/data/algorithm_golden_main.pt

An active editable install's import redirect takes precedence over
PYTHONPATH: drop its ``ScikitBuildRedirectingFinder`` from ``sys.meta_path``
(or use a venv without the editable install) so the export is imported.
The ``_fft_*`` fingerprint keys were added later from the same export; the
other keys are the first generation (the atomics ones differ run to run).
The ``cpu/mbir`` and ``cuda/mbir`` keys were later regenerated on the
branch that moved the MBIR voxel order into the CUDA kernel and again in
phase 1b (``tests/data/regen_mbir_goldens.py``), and the
``_fft_ramp`` / ``fbp*`` keys on the branch that zero-pads the ramp filter
(see test_algorithm_identity.py), and the geometry-gradient keys on the
branch that made grad_rotmat / grad_angle deterministic
(tests/data/regenerate_golden_keys.py); those runs replaced only their own
keys. The cases live in ``tests/test_algorithm_identity.py::_cases`` (public API
only), so generator and test cannot drift apart.

The input sinograms (``_input/<device>/sino``) were added later by
``tests/data/freeze_golden_inputs.py`` (main 7fdaac2's forward projection):
the test reads them instead of projecting, so ``carry_inputs`` copies them
from the repository's golden into the generated file (and stops with a
message if there are none).

``_golden.stamp`` then records this environment's arithmetic fingerprints
(tests/_golden.py) under ``_arith_fingerprint``: the exact comparisons are
only made where the fingerprint matches.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fasttomo  # noqa: E402
from _golden import stamp  # noqa: E402
from test_algorithm_identity import all_cases  # noqa: E402

def carry_inputs(path: str) -> None:
    """Copy the stored input sinograms (``_input/<device>/sino``, written by
    tests/data/freeze_golden_inputs.py and read by the test) from the
    repository's golden into the freshly generated file at ``path``, so the
    regenerated golden does not fail the test with a KeyError. Fails with a
    clear message if the repository golden has none."""
    src = Path(__file__).resolve().parent / 'algorithm_golden_main.pt'
    old = torch.load(src, weights_only=False)
    keys = [k for k in old if k.startswith("_input/")]
    if not keys:
        raise SystemExit(f"{src.name} holds no _input/* keys: run "
                         "tests/data/freeze_golden_inputs.py with a build of main 7fdaac2 first")
    new = torch.load(path, weights_only=False)
    new.update({k: old[k] for k in keys})
    torch.save(new, path)


if __name__ == "__main__":
    print("fasttomo from", fasttomo.__file__, "torch", torch.__version__, file=sys.stderr)
    torch.save(all_cases(), sys.argv[1])
    carry_inputs(sys.argv[1])
    stamp(sys.argv[1])

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
branch that moved the MBIR voxel order into the CUDA kernel (see
test_algorithm_identity.py). The cases live in ``tests/test_algorithm_identity.py::_cases`` (public API
only), so generator and test cannot drift apart.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fasttomo  # noqa: E402
from test_algorithm_identity import all_cases  # noqa: E402

if __name__ == "__main__":
    print("fasttomo from", fasttomo.__file__, "torch", torch.__version__, file=sys.stderr)
    torch.save(all_cases(), sys.argv[1])

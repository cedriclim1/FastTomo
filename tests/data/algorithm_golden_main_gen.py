"""Generate tests/data/algorithm_golden_main.pt (read by tests/test_algorithm_identity.py).

Run against origin/main's ``fasttomo/`` at 5362a6b (before the
projection-operator refactor), e.g. from a scratch export of that commit
with the built ``_core`` extension copied in. Needs a CUDA GPU for the
``cuda/`` keys:

    PYTHONPATH=<export-of-5362a6b> python tests/data/algorithm_golden_main_gen.py \
        tests/data/algorithm_golden_main.pt

The cases live in ``tests/test_algorithm_identity.py::_cases`` (public API
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

"""Generate tests/data/mbir_sweep_golden.pt (read by
tests/test_algorithms_mbir.py::test_mbir_cuda_sweep_matches_golden).

The golden holds the CUDA ICD sweeps' output (volume and error sinogram
after three outer sweeps) for every sweep variant: the serial kernel at 9,
61, 100, 140 and 260 angles (1, 2, 4 and 8 angle slots per lane, and the
generic kernel) and the parallel group kernel at the first four, each with
and without lambdas and the Huber weight. It was first made with the
64-thread block-reduction kernel of commit fd981d6 (to pin the one-warp
restructure) and regenerated in phase 1b (the paper's qGGMRF prior, pinned
rounding, one butterfly per sum, the parallel kernel) from that branch's
own code. Needs a CUDA GPU:

    python tests/data/mbir_sweep_golden_gen.py tests/data/mbir_sweep_golden.pt

(tests/data/regen_mbir_goldens.py does this and the MBIR keys of the
algorithm golden in one go, reporting the change against the old values.)
``_golden.stamp`` then records this environment's arithmetic fingerprints
(tests/_golden.py) under ``_arith_fingerprint``: the exact comparisons are
only made where the fingerprint matches.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _golden import stamp  # noqa: E402
from test_algorithms_mbir import _sweep_golden_cases  # noqa: E402

if __name__ == "__main__":
    torch.save(_sweep_golden_cases(), sys.argv[1])
    stamp(sys.argv[1])

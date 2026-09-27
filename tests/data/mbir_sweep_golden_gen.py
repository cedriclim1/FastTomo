"""Generate tests/data/mbir_sweep_golden.pt (read by
tests/test_algorithms_mbir.py::test_mbir_cuda_sweep_matches_pre_restructure_golden).

The golden holds the CUDA ICD sweep's output (volume and error sinogram
after three outer sweeps) for every sweep variant: 9, 61, 100, 140 and 260
angles (1, 2, 4 and 8 angle slots per lane, and the generic kernel), with
and without lambdas, with and without the Huber weight. It was made with the
64-thread block-reduction kernel of commit fd981d6 (the in-kernel voxel
order, before the one-warp restructure), so the test pins that the
restructured kernel reproduces the previous one bit for bit. Run with that
commit's ``_core`` built and importable (needs a CUDA GPU):

    python tests/data/mbir_sweep_golden_gen.py tests/data/mbir_sweep_golden.pt
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_algorithms_mbir import _sweep_golden_cases  # noqa: E402

if __name__ == "__main__":
    torch.save(_sweep_golden_cases(pre_restructure=True), sys.argv[1])

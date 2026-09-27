"""Store the golden tests' input sinograms in the golden files (one-time).

    python tests/data/freeze_golden_inputs.py

``tests/test_algorithm_identity.py`` and ``tests/test_frozen_identity.py``
built their input sinogram with ``ft.forward_project``, so a deliberate
change of the parallel-beam forward projector (the phase-1b sampling
boundary rule) would have changed every golden key downstream of it through
its *input*, including the frozen GENFIRE / RESIRE outputs whose code does
not change. This script computes those sinograms with the fasttomo build
that the goldens were verified with (run it on a build of origin/main
7fdaac2, before the boundary change) and stores them under
``_input/<device>/sino``; the tests then read them instead of projecting.
Every existing entry of both files is checked to stay byte-identical and
the arithmetic fingerprints are kept (the inputs are data, not outputs).
Running it again on the same build is a no-op; on a build whose forward
projection differs it refuses to overwrite a stored input.
"""
import sys
from pathlib import Path

import torch

TESTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TESTS))
import fasttomo as ft  # noqa: E402
from _golden import same_bytes  # noqa: E402

KEY = "_input/{dev}/sino"


def _algorithm_sino(dev):
    # the inputs of test_algorithm_identity._inputs, before its sinogram
    g = torch.Generator().manual_seed(1234)
    Ny, N, n = 4, 16, 9
    vol = torch.zeros(Ny, N, N)
    vol[:, 4:12, 5:11] = 1.0
    vol[1:3, 6:9, 6:10] = 2.0
    vol += 0.05 * torch.rand(Ny, N, N, generator=g)
    ang = torch.linspace(-60, 60, n)
    return ft.forward_project(vol.to(dev), ang.to(dev))


def _frozen_sino(dev):
    # the inputs of test_frozen_identity._inputs, before its sinogram
    g = torch.Generator().manual_seed(5)
    Ny, N, n = 3, 20, 13
    vol = torch.zeros(Ny, N, N)
    vol[:, 5:15, 6:13] = 1.0
    vol[1, 8:11, 8:12] = 2.0
    vol += 0.1 * torch.rand(Ny, N, N, generator=g)
    ang = torch.linspace(-65, 65, n)
    return ft.forward_project(vol.to(dev), ang.to(dev))


def freeze(path: Path, make) -> None:
    raw = torch.load(path, weights_only=False)
    new = dict(raw)
    for dev in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        s = make(dev).detach().cpu().contiguous()
        k = KEY.format(dev=dev)
        if k in raw:
            if not torch.equal(raw[k], s):
                raise SystemExit(f"{path.name}: stored {k} differs from this build's "
                                 "forward projection; refusing to overwrite a frozen input")
            continue
        new[k] = s
    torch.save(new, path)
    after = torch.load(path, weights_only=False)
    assert same_bytes({k: v for k, v in after.items() if k in raw}, raw), path
    print(path.name, sorted(k for k in after if k.startswith("_input/")))


if __name__ == "__main__":
    print("fasttomo from", ft.__file__, "torch", torch.__version__)
    freeze(TESTS / "data" / "algorithm_golden_main.pt", _algorithm_sino)
    freeze(TESTS / "data" / "frozen_golden_main.pt", _frozen_sino)

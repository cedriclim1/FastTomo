"""Regenerate the MBIR golden tensors after an intended change of MBIR's
numerics, and only those:

* ``tests/data/algorithm_golden_main.pt``: the keys ``cpu/mbir`` and
  ``cuda/mbir`` are recomputed with ``test_algorithm_identity._cases``;
  every other key is recomputed too and must be unchanged (bit-identical, or
  within 2e-6 for the fp32-atomics keys), else the script stops: an
  unexpected change is a bug, not a golden to regenerate;
* ``tests/data/mbir_sweep_golden.pt``: every key (it holds MBIR sweeps only),
  from ``test_algorithms_mbir._sweep_golden_cases``.

Both files are then stamped with this environment's arithmetic
fingerprints (``tests/_golden.py``). Prints the max abs / rel change of each
regenerated key. Needs a CUDA GPU; run from the repository root:

    python tests/data/regen_mbir_goldens.py
"""
import sys
from pathlib import Path

import torch

TESTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TESTS))
from _golden import FINGERPRINT_KEY, stamp, tensors  # noqa: E402
from test_algorithm_identity import ATOMIC_KEYS, _cases  # noqa: E402
from test_algorithms_mbir import _sweep_golden_cases  # noqa: E402

MBIR_KEYS = {"cpu/mbir", "cuda/mbir"}


def _change(old: torch.Tensor, new: torch.Tensor) -> str:
    if old.shape != new.shape:
        return f"shape {tuple(old.shape)} -> {tuple(new.shape)}"
    d = (new.double() - old.double()).abs().max().item()
    return f"max abs {d:.3e}, rel {d / max(old.double().abs().max().item(), 1e-30):.3e}"


def main() -> None:
    assert torch.cuda.is_available(), "needs a CUDA GPU"
    path = TESTS / "data" / "algorithm_golden_main.pt"
    raw = torch.load(path, weights_only=False)
    golden = tensors(raw)
    now = {**_cases("cpu"), **_cases("cuda")}
    assert set(now) == set(golden), set(now) ^ set(golden)
    for k in sorted(set(golden) - MBIR_KEYS):
        same = (torch.allclose(now[k], golden[k], rtol=2e-6, atol=2e-6) if k in ATOMIC_KEYS
                else torch.equal(now[k], golden[k]))
        assert same, f"{k} changed although it is not an MBIR key"
    new = dict(golden)
    for k in sorted(MBIR_KEYS):
        print(f"{path.name} {k}: {_change(golden[k], now[k])}")
        new[k] = now[k]
    new[FINGERPRINT_KEY] = raw[FINGERPRINT_KEY]
    torch.save(new, path)
    print(path.name, stamp(str(path)))

    path = TESTS / "data" / "mbir_sweep_golden.pt"
    raw = torch.load(path, weights_only=False)
    golden = tensors(raw)
    now = _sweep_golden_cases()
    for k in sorted(now):
        print(f"{path.name} {k}: " + (_change(golden[k], now[k]) if k in golden else "new key"))
    for k in sorted(set(golden) - set(now)):
        print(f"{path.name} {k}: removed")
    torch.save({**now, FINGERPRINT_KEY: raw[FINGERPRINT_KEY]}, path)
    print(path.name, stamp(str(path)))


if __name__ == "__main__":
    main()

"""Regenerate selected keys of a golden file in place (and nothing else).

    python tests/data/regenerate_golden_keys.py tests/data/algorithm_golden_main.pt \\
        cuda/fp_grad_angles cuda/fp3d_grad_euler ...
    python tests/data/regenerate_golden_keys.py tests/data/frozen_golden_main.pt ad_pose

For ``algorithm_golden_main.pt`` the new values come from
``tests/test_algorithm_identity.py::_cases`` (the device is the key's
prefix). For ``frozen_golden_main.pt`` only the run_ad_recon keys can be
regenerated, with the calls of ``test_frozen_identity.py::ad_cases`` (the
same definition the test uses); the GENFIRE / RESIRE keys are frozen.

Keys naming GENFIRE or RESIRE outputs (``genfire*`` / ``resire*``, e.g.
``cuda/gd_resire``) are refused in any file unless ``--allow-frozen`` is
passed. Every other key must stay byte-identical: the script checks that, prints the
max abs / rel change of each replaced key, then re-saves the file with its
arithmetic fingerprints (tests/_golden.py) unchanged. It refuses to run
unless this environment's fingerprint for the keys' device is already
recorded in the golden (it never adds one: that would mark the untouched
keys exact here unchecked). Run it on the machine whose fingerprint the
golden carries, from the repository root, with the fasttomo build under
test importable.
"""
import sys
from pathlib import Path

import torch

TESTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TESTS))
import fasttomo  # noqa: E402
from _golden import FINGERPRINT_KEY, fingerprint  # noqa: E402


def _ad_cases():
    """The run_ad_recon calls of test_frozen_identity.test_ad_recon_identical_to_main
    (``ad_cases`` there: one definition for the test and this script)."""
    import test_frozen_identity as F
    return F.ad_cases()


def _to_cpu(v):
    if isinstance(v, torch.Tensor):
        return v.detach().cpu()
    if isinstance(v, (tuple, list)):
        return type(v)(_to_cpu(x) for x in v)
    return v


def _leaves(v):
    if isinstance(v, (tuple, list)):
        return [x for e in v for x in _leaves(e)]
    return [v]


def _change(old, new) -> str:
    rows = []
    for a, b in zip(_leaves(new), _leaves(old)):
        if not isinstance(a, torch.Tensor):
            rows.append("same" if a == b else f"{b!r} -> {a!r}")
            continue
        d = (a.double() - b.double()).abs()
        s = b.double().abs().max().item() if b.numel() else 0.0
        m = d.max().item() if d.numel() else 0.0
        rows.append(f"max abs {m:.3e}, max rel (to max|old| {s:.3e}) {m / max(s, 1e-30):.3e}")
    return "; ".join(rows)


# Substrings of frozen outputs (any golden file): never regenerated silently.
FROZEN = ("genfire", "resire")


def _check_fingerprints(raw: dict, keys: list[str], path: str) -> None:
    """Refuse to run unless this environment's arithmetic fingerprint for
    every device the keys are computed on is already recorded in the
    golden. Regeneration never adds a fingerprint: that would declare the
    untouched keys (e.g. the frozen GENFIRE / RESIRE ones) exact here
    without checking them. On a new machine, first pass the golden tests
    at tolerance and stamp (tests/_golden.py, AGENTS.md), then regenerate."""
    devices = {k.split("/", 1)[0] if k.split("/", 1)[0] in ("cpu", "cuda") else "cuda"
               for k in keys}
    recorded = raw.get(FINGERPRINT_KEY, {})
    for dev in sorted(devices):
        fp = fingerprint(dev)
        if fp not in recorded.get(dev, []):
            raise SystemExit(
                f"{path}: this environment's {dev} fingerprint {fp} is not recorded "
                f"(recorded: {recorded.get(dev, [])}); regenerate on a machine whose "
                "fingerprint the golden carries, or stamp this one first after the "
                "golden tests pass (AGENTS.md)")


def main(path: str, keys: list[str], allow_frozen: bool = False) -> None:
    raw = torch.load(path, weights_only=False)
    missing = [k for k in keys if k not in raw]
    if missing:
        raise SystemExit(f"{path}: no such keys {missing}")
    _check_fingerprints(raw, keys, path)
    print("fasttomo from", fasttomo.__file__, "torch", torch.__version__)
    frozen = [k for k in keys if any(f in k.lower() for f in FROZEN)]
    if frozen and not allow_frozen:
        raise SystemExit(f"refusing frozen keys {frozen} (GENFIRE/RESIRE outputs are frozen; "
                         "pass --allow-frozen only for an approved change)")
    if not all(k.split("/", 1)[0] in ("cpu", "cuda") for k in keys):
        # un-prefixed keys: the frozen golden's run_ad_recon cases
        cases = _ad_cases()
        bad = [k for k in keys if k not in cases]
        if bad:
            raise SystemExit(f"only the run_ad_recon keys {sorted(cases)} can be regenerated, not {bad}")
        new = {k: _to_cpu(cases[k]()) for k in keys}
    else:
        from test_algorithm_identity import _cases
        new = {}
        for dev in sorted({k.split("/", 1)[0] for k in keys}):
            now = _cases(dev)
            new.update({k: now[k] for k in keys if k.startswith(dev + "/")})
    for k in keys:
        print(f"{k}: {_change(raw[k], new[k])}")
    before = {k: v for k, v in raw.items() if k not in keys}
    raw.update(new)
    torch.save(raw, path)
    after = torch.load(path, weights_only=False)
    for k, v in before.items():
        assert all(
            (a.numpy().tobytes() == b.numpy().tobytes() and a.dtype == b.dtype and a.shape == b.shape)
            if isinstance(a, torch.Tensor) else a == b
            for a, b in zip(_leaves(after[k]), _leaves(v))), f"{k} changed"
    # the fingerprints (checked above) are kept as they were: no stamp
    print("fingerprints kept:", after[FINGERPRINT_KEY])


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--allow-frozen"]
    if len(args) < 2:
        raise SystemExit(__doc__)
    main(args[0], args[1:], allow_frozen="--allow-frozen" in sys.argv[1:])

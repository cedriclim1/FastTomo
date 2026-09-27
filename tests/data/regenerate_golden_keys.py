"""Regenerate selected keys of a golden file in place (and nothing else).

    python tests/data/regenerate_golden_keys.py tests/data/algorithm_golden_main.pt \\
        cuda/fp_grad_angles cuda/fp3d_grad_euler ...
    python tests/data/regenerate_golden_keys.py tests/data/frozen_golden_main.pt ad_pose

For ``algorithm_golden_main.pt`` the new values come from
``tests/test_algorithm_identity.py::_cases`` (the device is the key's
prefix). For ``frozen_golden_main.pt`` only the run_ad_recon keys can be
regenerated, with the exact calls of ``test_frozen_identity.py``
(``AD_CASES`` below mirrors them); the GENFIRE / RESIRE keys are frozen.

Keys naming GENFIRE or RESIRE outputs (``genfire*`` / ``resire*``, e.g.
``cuda/gd_resire``) are refused in any file unless ``--allow-frozen`` is
passed. Every other key must stay byte-identical: the script checks that, prints the
max abs / rel change of each replaced key, then re-saves and stamps the
file with this environment's arithmetic fingerprints (tests/_golden.py).
Run it on the machine whose fingerprint the golden carries, from the
repository root, with the fasttomo build under test importable.
"""
import sys
from pathlib import Path

import torch

TESTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TESTS))
import fasttomo as ft  # noqa: E402
from _golden import FINGERPRINT_KEY, stamp  # noqa: E402


def _ad_cases():
    """The run_ad_recon calls of test_frozen_identity.test_ad_recon_identical_to_main."""
    import test_frozen_identity as F
    sino, ang, _, sm, N = F._inputs("cuda")
    C, Opt, R, K, P = (ft.AdReconConfig, ft.OptimConfig, ft.RegularizerConfig,
                     ft.ConstraintConfig, ft.PoseRefinementConfig)

    def extra():
        return dict(loss_callback=F._Calls(), progress=False, warm_start=None)

    return {
        "ad": lambda: ft.run_ad_recon(sino, ang, Nz=N, config=C(optim=Opt(n_iter=6)),
                                      return_loss=True, **extra()),
        "ad_reg": lambda: ft.run_ad_recon(sino, ang, Nz=N, config=C(
            optim=Opt(n_iter=6, loss_fn="l1"),
            reg=R(tv_weight=1e-3, tv_xz_weight=1e-3, sparsity=1e-3),
            constraints=K(positivity=True, cylinder=True, soft_threshold=1e-3)),
            return_loss=True, **extra()),
        "ad_pose": lambda: ft.run_ad_recon(sino, ang, Nz=N, config=C(
            optim=Opt(n_iter=6),
            pose=P(refine_pose=True, refine_shifts=True, in_plane_weight=1e-2,
                   anchor_reference=True)), return_loss=True, return_pose=True, **extra()),
        "ad_staged": lambda: ft.run_ad_recon(sino, ang, Nz=N, config=C(
            optim=Opt(n_iter=8), constraints=K(soft_threshold=1e-3),
            pose=P(refine_shifts=True, pose_warmup_iters=2, alternating_period=2)),
            return_loss=True, return_pose=True, **extra()),
        "ad_support": lambda: ft.run_ad_recon(sino, ang, 5, Nz=N, config=C(optim=Opt(n_iter=99)),
                                              support_mask=sm, **extra()),
    }


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


def main(path: str, keys: list[str], allow_frozen: bool = False) -> None:
    raw = torch.load(path, weights_only=False)
    missing = [k for k in keys if k not in raw]
    if missing:
        raise SystemExit(f"{path}: no such keys {missing}")
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
    before = {k: v for k, v in raw.items() if k not in keys and k != FINGERPRINT_KEY}
    raw.update(new)
    torch.save(raw, path)
    after = torch.load(path, weights_only=False)
    for k, v in before.items():
        assert all(
            (a.numpy().tobytes() == b.numpy().tobytes() and a.dtype == b.dtype and a.shape == b.shape)
            if isinstance(a, torch.Tensor) else a == b
            for a, b in zip(_leaves(after[k]), _leaves(v))), f"{k} changed"
    print("stamp:", stamp(path))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--allow-frozen"]
    if len(args) < 2:
        raise SystemExit(__doc__)
    main(args[0], args[1:], allow_frozen="--allow-frozen" in sys.argv[1:])

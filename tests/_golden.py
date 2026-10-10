"""
Arithmetic fingerprint that gates the exact (``torch.equal``) golden checks.

Bit identity with a golden file is a property of the machine's floating-point
paths (ATen's CPU SIMD dispatch, the BLAS and FFT libraries and the code
paths they pick for the CPU they run on, the GPU architecture), not of the
torch version: two GitHub runners with the same torch but different CPUs give
results a few ulp apart. So instead of trusting a torch version / platform
list, each golden file stores, under ``FINGERPRINT_KEY``, the fingerprints of
the environments its exact comparisons were verified in::

    golden["_arith_fingerprint"] == {"cpu": ["<hex>", ...], "cuda": ["<hex>", ...]}

``fingerprint(device)`` runs a fixed small computation on ``device`` through
the paths the goldens depend on (a float32 matmul, reductions, real and
complex FFTs of the sizes the algorithms use, exp / log / sqrt / sin / cos /
atan2, an Adam-style update, fp16 rounding, a trilinear ``F.interpolate``,
the bilinear ``F.grid_sample`` of ``apply_shifts_2d`` (the ``fp3d_shift*``,
``ad_pose`` and ``ad_staged`` goldens) and one autograd backward pass through
it and a matmul chain) and hashes the result bytes together with the SIMD
dispatch level and ``OMP_NUM_THREADS`` (CPU) or the compute capability and
CUDA version (GPU, whose texture-unit interpolation no torch op reaches).
``exact_env`` returns True only where the local fingerprint is one the golden
lists; the tests then compare with ``torch.equal`` and otherwise fall back to
their documented tolerance. A fallback is never silent: it raises a
``GoldenNotExactWarning`` (listed in pytest's warnings summary).

To add the current environment to a golden after verifying that its exact
comparison holds here (only the metadata changes; the tensors are checked to
be byte-identical before the file is rewritten)::

    python tests/_golden.py stamp tests/data/<golden>.pt

A torch upgrade (or any change to ``_probe``) changes the fingerprint: re-stamp
every golden via its generator script in ``tests/data/*_gen.py`` (or the
``stamp`` command above) after re-verifying its exact comparisons.
"""

from __future__ import annotations

import hashlib
import os
import platform
import sys
import warnings

import torch
import torch.nn.functional as F

FINGERPRINT_KEY = "_arith_fingerprint"


class GoldenNotExactWarning(UserWarning):
    """The exact branch of a golden comparison was not exercised."""


def _bytes(t: torch.Tensor) -> bytes:
    return t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def _env_tag(device: str) -> str:
    if device == "cpu":
        # OMP_NUM_THREADS also sets the thread count of FastTomo's OpenMP
        # CPU kernels (single-threaded when unset), whose reductions are
        # ordered per thread: the goldens hold with it unset.
        omp = os.environ.get("OMP_NUM_THREADS", "").strip() or "unset"
        return f"cpu:{torch.backends.cpu.get_cpu_capability()}:omp={omp}"
    major, minor = torch.cuda.get_device_capability()
    return f"cuda:sm_{major}{minor}:cuda{torch.version.cuda}"


def cpu_model() -> str:
    """The CPU's model name (``/proc/cpuinfo`` "model name", else
    ``platform.processor()``), printed in ``GoldenNotExactWarning`` so a new
    CI runner type can be identified from the log: runners with the same SIMD
    dispatch label can still differ in the last bit of an FFT."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "unknown"


def peak_close(a: torch.Tensor, b: torch.Tensor, tol: float) -> tuple[bool, float, float]:
    """Loose comparison for outputs that go through an FFT: ``|a - b| <= tol *
    max(1, max|b|) + tol * |b|`` elementwise.

    An FFT's round-off is set by the norm of the whole transform, not by each
    bin's own size, so two CPU FFT code paths differ by about one ulp of the
    tensor's largest value on every element, small bins included (CI run
    36331234339: 3.8e-6 on a bin of ``cpu/_fft_rfft2`` whose peak is 66.7,
    i.e. 5.7e-8 of the peak). The purely elementwise ``allclose(rtol=tol,
    atol=tol)`` rejects that on the small bins. The ``max(1, ...)`` keeps the
    bound at least as wide as the absolute one for tensors of peak below 1.
    Returns ``(ok, max|a - b|, max|b|)``."""
    a64, b64 = a.double(), b.double()
    peak = b64.abs().max().item() if b.numel() else 0.0
    diff = (a64 - b64).abs().max().item() if a.numel() else 0.0
    ok = a.shape == b.shape and torch.allclose(a64, b64, rtol=tol, atol=tol * max(1.0, peak))
    return ok, diff, peak


def _probe(device: str) -> list[torch.Tensor]:
    """A fixed computation through the arithmetic the goldens depend on.
    Sizes are of the order of the goldens' (small, single-threaded on the
    CPU)."""
    g = torch.Generator().manual_seed(20260927)
    a = torch.randn(67, 53, generator=g).to(device)
    b = torch.randn(53, 71, generator=g).to(device)
    x = torch.rand(4, 3, 20, 20, generator=g).to(device)
    s = torch.rand(13, 3, 20, generator=g).to(device)
    c = torch.complex(torch.randn(3, 40, 40, generator=g),
                      torch.randn(3, 40, 40, generator=g)).to(device)
    ang = torch.linspace(-65, 65, 13).to(device)
    out = [
        a @ b,                                            # BLAS sgemm
        torch.bmm(x[0], x[1].transpose(-1, -2)),
        x.sum(), x.sum(dim=-1), x.mean(dim=(1, 2)), x.norm(), x.amax(dim=(-2, -1)),
        torch.cumsum(x, dim=-1), (x * s[:4, :, None, :]).sum(dim=(-2, -1)),
        torch.exp(x - 0.5), torch.log(x + 0.1), torch.sqrt(x), torch.rsqrt(x + 1),
        torch.sin(torch.deg2rad(ang)), torch.cos(torch.deg2rad(ang)),
        torch.atan2(x[0], x[1] - 0.5), torch.tanh(x[2]), torch.pow(x[3] + 0.1, 1.2),
        (x.half() * 1.7).float(), (x * 0.3).half(),
    ]
    for n in (16, 20, 26, 33, 64):                        # ramp filter, correlation
        f = torch.fft.rfft(s, n=n, dim=-1)
        out += [torch.view_as_real(f), torch.fft.irfft(f * 0.5, n=n, dim=-1)]
    f2 = torch.fft.rfft2(x)                               # W-BP deconvolution
    out += [torch.view_as_real(f2), torch.fft.irfft2(f2, s=x.shape[-2:])]
    fc = torch.fft.fftn(torch.fft.ifftshift(c, dim=(-2, -1)), dim=(-2, -1))  # GENFIRE
    out += [torch.view_as_real(fc), torch.view_as_real(torch.fft.ifftn(fc, dim=(-2, -1))),
            torch.view_as_real(torch.fft.fft(c[0], dim=-1))]
    m, v, p = torch.zeros_like(x), torch.zeros_like(x), x.clone()   # Adam-style
    for grad in (x - 0.4, 0.5 - x, x * x):
        m.mul_(0.9).add_(grad, alpha=0.1)
        v.mul_(0.999).addcmul_(grad, grad, value=0.001)
        p.addcdiv_(m, v.sqrt().add_(1e-8), value=-0.05)
    out += [p]
    out += [F.interpolate(x[None], size=(4, 7, 13), mode="trilinear", align_corners=False)]
    # apply_shifts_2d: bilinear grid_sample (zeros padding, align_corners),
    # forward and backward. The backward w.r.t. the grid is per output
    # sample; the input gradient is taken on the CPU only (CUDA accumulates
    # it with atomics, whose order is not fixed, which would make the
    # fingerprint itself flaky).
    with torch.enable_grad():
        sino = s[:, None].clone().requires_grad_(device == "cpu")     # [13, 1, 3, 20]
        sh = (torch.rand(13, 2, generator=g) - 0.5).to(device).requires_grad_(True)
        gy, gx = torch.meshgrid(torch.linspace(-1, 1, 3, device=device),
                                torch.linspace(-1, 1, 20, device=device), indexing="ij")
        grid = torch.stack((gx, gy), -1)[None] - (sh * torch.tensor([2 / 19, 1.0],
                                                                     device=device))[:, None, None]
        shifted = F.grid_sample(sino, grid, mode="bilinear", padding_mode="zeros",
                                align_corners=True)
        a2 = a.clone().requires_grad_(True)
        loss = (shifted * shifted).sum() + torch.exp(0.01 * (a2 @ b)).sum()
        loss.backward()
    out += [shifted.detach(), sh.grad, a2.grad]
    if device == "cpu":
        out += [sino.grad]
    return out


def fingerprint(device: str = "cpu") -> str:
    """Hex digest of ``_probe(device)``'s bytes and the environment tag."""
    h = hashlib.sha256(_env_tag(device).encode())
    with torch.no_grad():
        for t in _probe(device):
            h.update(str((t.dtype, tuple(t.shape))).encode())
            h.update(_bytes(t))
    return h.hexdigest()[:24]


def tensors(golden: dict) -> dict:
    """The golden's entries without the fingerprint metadata."""
    return {k: v for k, v in golden.items() if k != FINGERPRINT_KEY}


def exact_env(golden: dict, device: str, what: str = "") -> bool:
    """True when this environment's ``device`` fingerprint is one the golden
    was verified in, so its deterministic entries must match bit for bit.
    Otherwise warns (``GoldenNotExactWarning``), so that losing the exact
    check is visible rather than silent."""
    fp = fingerprint(device)
    known = golden.get(FINGERPRINT_KEY, {}).get(device, [])
    exact = fp in known
    if not exact:
        host = f"CPU {cpu_model()!r}"
        if device != "cpu":
            host = f"GPU {torch.cuda.get_device_name()!r}, {host}"
        warnings.warn(GoldenNotExactWarning(
            f"{what or 'golden'}: {device} arithmetic fingerprint {fp[:8]} (full {fp}; "
            f"{_env_tag(device)}; {host}) is not one the golden was verified in "
            f"{[k[:8] for k in known]}; exact (torch.equal) branch NOT exercised, compared "
            "at the documented tolerance instead"), stacklevel=2)
    return exact


def _byte_view(v):
    if isinstance(v, torch.Tensor):
        return (str(v.dtype), tuple(v.shape), _bytes(v))
    if isinstance(v, (tuple, list)):
        return (type(v).__name__, tuple(_byte_view(x) for x in v))
    if isinstance(v, dict):
        return ("dict", tuple((k, _byte_view(x)) for k, x in v.items()))
    return ("py", repr(v))


def same_bytes(a: dict, b: dict) -> bool:
    """True when the two goldens' entries (metadata excluded) are
    byte-identical: same keys, dtypes, shapes and bytes."""
    a, b = tensors(a), tensors(b)
    return a.keys() == b.keys() and all(_byte_view(a[k]) == _byte_view(b[k]) for k in a)


def stamp(path: str) -> dict[str, list[str]]:
    """Add this environment's fingerprints (CPU, and CUDA when available) to
    the golden at ``path``. Only the metadata changes."""
    golden = torch.load(path, weights_only=False)
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    meta = {d: list(golden.get(FINGERPRINT_KEY, {}).get(d, [])) for d in ("cpu", "cuda")}
    for d in devices:
        fp = fingerprint(d)
        if fp not in meta[d]:
            meta[d].append(fp)
    new = dict(tensors(golden))
    new[FINGERPRINT_KEY] = meta
    torch.save(new, path)
    assert same_bytes(torch.load(path, weights_only=False), golden), path
    return meta


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "stamp":
        for p in sys.argv[2:]:
            print(p, stamp(p))
    else:
        print("cpu model:", cpu_model())
        for d in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            print(d, _env_tag(d), fingerprint(d))

# Changelog

## Unreleased

### ⚠ Parallel-beam projector: loose sampling boundary; symmetric geometry gradients at grid alignments (phase 1b, PR G)

**Every parallel-beam result changes slightly at the volume edges.** The
single-axis operators (`forward_project`, `back_project` on CPU,
`ParallelProjector`, and everything built on them: SIRT, SART, wSIRT, FISTA,
FISTA-TV, gradient descent, AD reconstruction, `refine_angles`) now sample
the volume with the rule the rotmat path, the CUDA border-mode textures and
MBIR already used: every slice is a sum of bilinear tent functions, zero
outside, so a ray sample contributes while any corner of its stencil is
inside, x index in (-1, Nx), z in (-1, Nz). Before, a sample needed its
whole stencil inside ([0, N-1)): at 0 degrees the last detector column was
0 and the last z row was never sampled for odd Nz, Nz = 1 or Nx = 1 volumes
projected to 0, and the result was not symmetric under an x-flip or a half
turn. This reverses phase-1 decision 3 (tight bounds; its finite-difference
argument holds for every rule).

- `forward_project(v, a)` now equals `forward_project_3d(v, (0, a, 0))`:
  on CUDA bit for bit at 0 / +-90 / 180 degrees and at most other views
  (at some generic views about 0.3 % of the outputs differ by up to 3.4e-4
  relative, where the two kernels round a texture coordinate to a different
  8-bit filter weight), on CPU to 3e-7. Volumes that are zero on the outer
  voxel ring project to the same bits as before; the CPU back projection
  (still the exact transpose, bit for bit) changes only on that ring; the
  CUDA back projection (texture gather) is unchanged.
- Consequences measured on analytic data (no inverse crime) of an ET slab
  that fills the field of view, N = 128, 61 views over +-60 degrees, CUDA,
  mean error of column 0 / column N-1: `run_sirt` 100 iterations
  +0.07 / +0.50 -> +0.04 / +0.05, `run_sart` +-0.01 / +0.19 -> -0.02 / -0.01,
  `run_fista` +0.01 / +5.97 (RMSE 0.61) -> -0.01 / -0.01 (RMSE 0.19),
  `run_grad_descent` 300 iterations +0.26 / +1.06 -> +0.21 / +0.22.
  `refine_angles` started at the true angles of that slab (64^2, 61 views)
  ended 0.45 degrees rms (1.5 max) off on pure-tilt rotmat data and 1.17
  rms (3.3 max) on analytic data; now 0.000 / 0.0004 and 0.35 (1.8).
  The CUDA operator pair is closer to adjoint: `||B - A^T||_F / ||A||_F`
  0.21 -> 0.077 at 32^2, 0.16 -> 0.077 at 64^2 (61 views); CUDA FISTA's
  long-run growth (solver audit A2) is smaller but not gone (64^2 noise-free
  after 3000 iterations: relative error 14.1 -> 0.77; see `run_fista`).
- `grad_angle` (the angle gradient of `forward_project`) and `grad_rotmat`
  (the pose gradient of `forward_project_3d`), CPU and CUDA: at a sample
  exactly on a grid line or plane the kernels take the average of the two
  one-sided differences (they took the forward one), with an inclusive
  [-1, N] bound. The gradient now equals the central finite difference at
  grid-aligned views too: 0 degrees (every sample on a grid line), +-90 /
  180 degrees, the identity pose and pure tilts `(0, theta, 0)` (every
  sample on a y plane: the pose-refinement start), 1e-8 to 1.2e-6 against an
  fp64 central FD on both backends (was off by 20 % to several times there).
  Elsewhere the gradients change only where a sample happens to fall
  exactly on a grid line (at generic poses up to ~1e-3 relative for
  `grad_rotmat` at 128^3). The design study measured
  no convergence change from the kink rule (pose-only Adam and
  `run_ad_recon` pose stages agree to 3-4 digits): it is a correctness and
  testability fix. Note: torch's fp32 cos(90 deg) is -4.4e-8, so a +-90 / 180
  degree view is 2.5e-6 / 5e-6 degrees past the alignment and its far
  samples are just off the grid line, where the one-sided derivative is the
  exact one.
- CUDA `grad_angle` now walks only the clipped step range of each ray,
  united over the warp (as the forward; bit-identical to the full loop).

Cost (sm_120, interleaved A/B of the two builds in one process, N^3 volumes,
angles linspace(-70, 70, n)): CUDA forward kernel +0.8 to +2.5 % (fp32 and
fp16, 128^3 to 512^3, 61 and 181 views); CUDA `grad_angle` +18-23 % at
128^3 and +13-17 % at 256^3 (61 views); CUDA `grad_rotmat` +1 % on generic
poses and 1.9-2.1x on exact pure tilts (4 extra fetches per sample; only
the first pose iteration of a tilt series is one). CPU (32 x 128^2 and
16 x 256^2, 61 views): forward +3-5 %, ray-driven back projection +3-4 %
(1 thread; within noise at 8), `grad_angle` +9-10 %, `grad_rotmat` +0.5-3 %
generic and +24-30 % on a pure tilt; single slices (Ny = 1, 256^2 and
512^2), which now copy the slice into a zero-bordered scratch, forward
+8-9 % (1 thread) and +17-31 % (8 threads), back projection +10-16 %,
`grad_angle` +17-25 %.

Goldens: the golden tests' input sinograms are now stored in the golden
files (`_input/<device>/sino`, `tests/data/freeze_golden_inputs.py`), so
GENFIRE / RESIRE (and FBP, W-BP, MBIR) keys stay byte-identical; the parallel
projection keys, every parallel-derived algorithm key, the rotmat
pose-gradient keys at the identity pose and four `run_ad_recon` keys (`ad`,
`ad_reg`, `ad_pose`, `ad_staged`; `ad_support` is unchanged: its support
mask zeroes the outer voxel ring) were regenerated (`tests/data/regen_boundary_goldens.py`).
`tests/data/regenerate_golden_keys.py` now checks that every key it does not
replace is still reproduced.

### Parallel-beam CUDA projections: packed textures, ray clipping, texture pool (phase 1b, PR B)

Same results bit for bit (0 differing outputs over 3.7e8 values: FP/BP one-shot
and cached, fp32/fp16, Ny = 1..13 and odd sizes, N_det != Nx, edge-case angles,
and the CUDA SIRT/SART/wSIRT/FBP/FISTA outputs; the goldens are unchanged).

- fp32 parallel textures hold 2 Y-slices per `float2` texel, fp16 ones 4 per
  texel for every Ny >= 2 (the last texel zero-padded; fp16 used to fall back
  to 1-channel textures when `Ny % 4 != 0`). The filter unit weights every
  channel alike, so each slice gets the bits of a 1-channel fetch.
  `_core.parallel_channels(io_dtype, Ny)` is the one layout rule; the cached
  projectors and run_sirt's fused texture writes use it too
  (`sirt_residual_tex_packed_kernel` / `sirt_update_packed_kernel` for both
  formats, the latter with #104's planar `C_full`), and run_fista's fused
  update writes both packed formats in place (`fista_update_packed_kernel`;
  it used to report them unwritable, costing one upload per iteration). An fp32 volume may now have up to 4096 Y-slices and an fp16 one
  8192 (2048 texture layers). `tc_create` accepts `channels=2` (fp32,
  layered kinds) and now rejects packed `TC_VOL_3D` caches (unused; the old
  4-channel 3-D path did not round-trip).
- The forward projection walks only the steps where its ray can be inside
  the volume, as one range per warp (32 x 8 blocks); the in-loop bounds test
  is unchanged.
- A NaN/Inf in CUDA-resident tilt angles (not checked on the host, see below)
  now gives what the CPU kernels give: a zero forward-projection view, and no
  contribution to the back-projection (it was finite garbage).
- fp16 overflow check: the parallel CUDA kernels report non-finite stores
  in a 4-byte word, so the check after a public fp16 call reads that word
  instead of scanning the result (same OverflowError / FloatingPointError
  contract); inside the fp16 algorithms the recorded flag is one tiny kernel
  instead of a reduction over the result.
- `forward_project`, `back_project`, the `*_3d` variants and `run_fbp` on CUDA
  reuse up to 4 texture arrays of the most recently used shapes instead of
  allocating one, synchronising and freeing it on every call: the host no
  longer blocks per call (fp32; fp16 still reads the overflow word). Memory
  retained: up to 4 texture arrays, outside PyTorch's allocator (a 512^3
  fp32 volume texture is 512 MiB). `release_cached_projectors()` frees them;
  an allocation failure empties the pool and retries once. Calls from
  several threads or streams are safe (a stream switch waits for the
  previous stream, like the cached projectors).

Measured on an RTX PRO 6000 Blackwell (sm_120), torch 2.14.0+cu130, angles
linspace(-70, 70, n), N^3 volumes, median of 20 CUDA-event timings, main ->
this change (ms):

| | 128^3 / 61 | 256^3 / 181 | 512^3 / 61 | 512^3 / 181 |
|---|---|---|---|---|
| fp32 FP kernel | 0.100 -> 0.076 | 1.92 -> 1.31 | 5.71 -> 3.59 | 15.7 -> 9.90 |
| fp32 BP kernel | 0.084 -> 0.061 | 1.78 -> 1.25 | 4.91 -> 3.17 | 13.7 -> 9.46 |
| fp32 `forward_project` call | 0.270 -> 0.117 | 2.31 -> 1.42 | 7.88 -> 4.47 | 17.4 -> 10.4 |
| fp32 host time per call | 0.379 -> 0.040 | 2.40 -> 0.040 | 8.18 -> 0.041 | 17.2 -> 0.040 |
| fp16 BP call (`ParallelProjector.back`) | 0.075 -> 0.069 | 0.753 -> 0.649 | 2.99 -> 1.67 | 6.13 -> 4.89 |
| fp16 FP kernel, Ny = N-1 (A/B in one process) | 0.091 -> 0.042 | 1.79 -> 0.641 | 5.24 -> 1.90 | 14.9 -> 5.16 |
| `run_fbp` fp32 | 0.358 -> 0.160 | | 6.79 -> 5.11 | |

fp32 kernels 1.3-1.6x, fp16 with `Ny % 4 != 0` 2.1-2.9x, fp16 with
`Ny % 4 == 0` 1.0-1.1x (FP clipping only). Script:
`benchmarks/scripts/run_parallel_projection_bench.py`.

### Classical solver loops: SIRT, SART, wSIRT, FISTA, FISTA-TV, FBP, W-BP (phase 1b, PR E)

Measured on one RTX PRO 6000 Blackwell (sm_120), fp32, 61 views over +-60
degrees, N_det = N, CUDA events, median of 20 calls; per iteration =
(t(n2) - t(n1)) / (n2 - n1).

- **fp16 `run_fista` / `run_fista_tv` no longer raise `ZeroDivisionError`
  when `||A^T A||` exceeds 65504** (a 512-pixel detector with 180 views):
  the Lipschitz power iteration takes its norms in fp32 and keeps L on the
  device (one host sync instead of one per step). fp32 is unchanged bit
  for bit; fp16 FISTA changes in the last bits of L (`fista_h` goldens
  regenerated, max change 2 fp16 ulps).
- **FISTA-TV: the TV proximal operator runs as two CUDA kernels per inner
  iteration** (`csrc/cuda/tv_prox.cu`; the PyTorch FGP / Chambolle code
  stays the CPU path and the reference). Bit-identical to the PyTorch prox
  in fp32 and fp16 (goldens unchanged; the prox weight's reciprocal
  `1 / lam` is rounded as CUDA ATen rounds a division by a Python float:
  in double, then once to fp32); no Triton dependency. Per outer
  iteration with 20 inner iterations: 68.7 -> 18.3 ms at 256^3 (3.75x),
  4.45 -> 0.69 ms at 128^3 (6.5x); the prox alone 66.8 -> 16.8 ms and
  4.12 -> 0.56 ms, peak 1.74 GiB -> 448 MiB at 256^3. 100 inner iterations
  now cost 83 ms at 256^3, about what 20 did.
- `run_fista_tv(tv_inner_iter=)` (default 20, unchanged) is documented
  with the duality gap it leaves (1-3 % at prox weight `tv_weight / L` =
  0.02, 13-24 % at 0.1) and validated early (integer >= 0), as is
  `tv_inner`. The docstring also states how its `tv_weight` relates to
  the TV weight of `run_grad_descent` / `run_ad_recon` (different TV and
  data-term scales: `w_fista ~= w_ad * M / (2 Nc)`).
- **`run_sart` is 1.3-1.9x faster and needs ~20x less memory**: every
  subset step is run_sirt's fused residual and update kernels (writing the
  sinogram and volume textures directly), subsets are slices (no host sync
  per subset, also with CPU angles: was 60-183 per iteration), and in
  parallel geometry the per-subset normalisers are one sinogram plus one
  `[1, Nz, Nx]` plane per subset (they are the same in every Y-slice),
  computed on a one-slice operator. 256^3: 30.6 -> 16.4 ms per iteration,
  setup 29 -> 4.2 ms, peak 4.3 GiB -> 223 MiB; 128^3: 6.1 -> 4.3 ms,
  setup 8.0 -> 4.2 ms.
  Bit-identical (`sart*` goldens).
- New `run_sart(view_order="golden")`: visits the views in the
  golden-ratio order (view k at the rank of frac(k (sqrt(5) - 1) / 2)).
  After one sweep of single-view SART (64^2, 1 % noise): relative error
  0.096 vs 0.650 (180 views over 180 degrees), 0.31 vs 0.70 (61 views
  over +-60 degrees). Default `"sequential"` (unchanged).
- Recorded losses stay on the device (`_DeviceLossRecorder`, shared with
  the AD solvers): with `return_loss` alone no solver syncs the host per
  iteration any more (was 1); with a `loss_callback` one sync per
  iteration (the frozen contract). `run_sirt` with a loss now runs on the
  CUDA-graph path (the loss graph uses the eager rounding order, so the
  result and the loss history are unchanged bit for bit): 256^3
  1.525 -> 1.389 ms, 128^3 0.220 -> 0.189 ms per iteration.
- `run_sirt` keeps one `[1, Nz, Nx]` plane of the voxel normaliser in
  parallel geometry (`sirt_update_cuda(C_full=0)`): one volume less
  streamed per iteration and kept by the graph cache (256^3: 1.417 ->
  1.348 ms per iteration; kept 0.36 -> 0.30 GiB, 512^3 2.62 -> 2.12 GiB).
  No zero-tensor fill per iteration in fp32 (H2). Bit-identical.
- FISTA: the step, constraints and momentum are one fused kernel that also
  writes the new momentum point into the volume texture
  (`csrc/cuda/fista_update.cu`; FISTA-TV uses its step and momentum
  halves around the prox). Bit-identical. 256^3: 1.677 -> 1.509 ms per
  iteration (positivity + mask 1.768 -> 1.508), 128^3 0.208 -> 0.180 ms.
  The residual is computed once when a loss is recorded (H3).
- `run_weighted_backprojection` / `run_wbp` / `run_wsirt` cache the exact
  weighting filter per geometry (bounded LRU of 8, keyed by the fp32 angle
  values, sizes, `threshold_scale` and device): `run_wbp` 6.0 -> 0.41 ms
  at 128^3 (14.8x), 7.2 -> 1.55 ms at 256^3; wSIRT setup 6 -> 0.4 ms.
  Bit-identical. The filters stay on their device (up to 8 of
  `[Pz, Px // 2 + 1]` fp32, Pz, Px ~ 1.125 N: ~85 MiB at 2048^2) until
  `release_cached_projectors()`, which now also drops them.
- New `run_fbp(window="none" | "hann" | "hamming")`: apodises the Ram-Lak
  ramp (Hann 0.5 + 0.5 cos(2 pi f), Hamming 0.54 + 0.46 cos(2 pi f), f in
  cycles per detector pixel on the zero-padded grid). Default `"none"`
  is the former filter bit for bit.
- Documentation of measured long-run behaviour (no algorithm change):
  `run_wsirt` without positivity is slowly unstable (spectral radius of
  `I - omega W A` 1.0013-1.008; error 0.24 at 200 iterations, 1.46 at
  3000 on a 64^2 slice; stable with positivity) and CUDA `run_fista` /
  `run_fista_tv` eventually diverge because the CUDA back-projection is
  not the adjoint of the forward projection (0.185 at 200 iterations, 14.1
  at 3000; CPU 0.131): keep `n_iter` at about 200-300 on CUDA. Tests pin
  both.

### AD reconstruction and regularisers: speed and accuracy (phase 1b, PR D)

- `run_ad_recon` with a staged pose schedule (`pose_warmup_iters`,
  `alternating_period`) now differentiates only the leaves stepped in each
  iteration: a volume-only iteration skips `grad_rotmat` and the shift
  backward, a pose-only one the volume back-projection. Bit-identical
  output. At 256^3, 61 views, pose + shifts: a volume-only (warm-up)
  iteration 5.4 -> 2.5 ms, `alternating_period=5` 6.3 -> 3.9 ms per
  iteration.
- `run_ad_recon`'s soft-threshold constraint (`ConstraintConfig.
  soft_threshold`) is one in-place `softshrink` pass (fp32). Same values
  as before (a zero may now be +0 instead of -0; the goldens compare
  equal). With positivity, cylinder and soft-threshold at 256^3: 2.18 ->
  1.85 ms per iteration, peak 448 -> 352 MiB. The generic constraints
  (`apply_constraints`, the `sparsity=` keyword of the other `run_*`
  functions) keep the old expression bit for bit: run_sirt's fused CUDA
  kernels reproduce its bits, and the frozen GENFIRE/RESIRE use it.
- Loss recording in `run_ad_recon` and `run_grad_descent`: with
  `return_loss` alone the losses stay on the device and are copied to the
  host once at the end (0 host syncs per iteration, was 1); with a
  `loss_callback` the total and all components reach the host in one
  transfer (1 sync per iteration, was 1 + number of components). The
  `loss_callback(iter, loss)` contract is unchanged: plain floats, the
  true iteration index, the same values. `run_ad_recon` 128^3 L2 with
  `return_loss`: 0.42 -> 0.32 ms per iteration.
- **`tilt_ssim_loss` values change (accuracy fix).** On CUDA, cuDNN ran
  its dense 11 x 11 convolutions in TF32 at small batches, and the
  `E[x^2] - mu^2` variance form turned that into a 1-6 % error in the loss
  and its gradient at sinogram scale (mean ~100). The window is now
  applied as two depthwise 1-D passes (no TF32 path) to data centred on
  each target tilt's mean, with the SSIM formula rearranged so that no
  step subtracts two `E[x^2]`-sized values. Against an fp64 reference the
  error is now <= 1e-6 (loss) and 1.2e-7 (gradient), TF32 on or off, CPU
  or CUDA. Same window, constants (`K1 = 0.01`, `K2 = 0.03`, `L = max -
  min` of the target) and reflect padding. `run_ad_recon` caches the
  measured sinogram's statistics. Faster: `tilt_ssim_loss` forward +
  backward on 61 x 256^2 12.5 -> 2.2 ms; `run_ad_recon` with
  `tilt_ssim_weight` 14.9 -> 3.7 ms per iteration at 256^3 (3.7 -> 1.0 ms
  at 128^3).
- The scikit-image SSIM cross-check test now runs without scikit-image
  (an inline fp64 reference, anchored to a hard-coded scikit-image value).
- **`s3im_loss` draws different patches for the same seed.** Each patch
  is now the `B` largest of `n` uniform random numbers (`rand(n).topk(B)`,
  still without replacement and uniformly random) instead of
  `randperm(n)[:B]`, which sorted all `n` entries (0.40 -> 0.16 ms at
  61 x 256^2). The `M` patches go through the SSIM as one batch, with the
  same centred, separable statistics as `tilt_ssim_loss`; `run_ad_recon`
  computes the constants once per run. `run_ad_recon` with `s3im_weight`
  at 256^3: 2.39 -> 2.24 ms per iteration, peak 408 -> 352 MiB.
- `tv_loss_xz` on CUDA is a fused kernel (the planar mode of the 3-D TV
  kernel, `csrc/cuda/tv_loss.cu`: deterministic two-pass reduction with
  fp64 partials, gather backward). Forward + backward at 256^3 2.19 ->
  0.16 ms; `run_ad_recon` with `tv_xz_weight` 4.05 -> 2.00 ms per
  iteration, peak 608 -> 368 MiB. Values move by fp32 rounding only
  (<= 2.7e-7 relative vs the torch expression).
- CPU `tv_loss_3d` returned NaN when a dimension is 1 (and `tv_loss_xz`
  when `Nz` or `Nx` is 1); both now return 0 with a zero gradient, like
  CUDA. A single-slice volume (`Ny = 1`) therefore has no 3-D TV (now
  documented; `tv_loss_xz` is the per-slice TV).
- **First-order only:** `tilt_ssim_loss`, `s3im_loss` and, on CUDA,
  `tv_loss_3d` and `tv_loss_xz` have hand-written backwards. A backward
  with `create_graph=True` (double backward, Hessian-vector products)
  now raises RuntimeError instead of returning a wrong second derivative.
  Before this release `tilt_ssim_loss`, `s3im_loss` and CUDA `tv_loss_xz`
  were plain autograd (exact second order), while CUDA `tv_loss_3d`
  silently dropped its second-order term; nothing in FastTomo uses
  second-order gradients. The CPU TV expressions stay differentiable to
  any order.
- **`run_ad_recon` pose refinement with `antisymmetric_z=True` (the
  default) changes (accuracy fix).** `phi` was stepped with the partial
  derivative `dL/dphi` and `psi` then overwritten with `-phi`, whose
  fixed points are not stationary points of the constrained problem.
  `psi = -phi` is now built into the forward pass, so `phi` gets the
  total derivative `dL/dphi - dL/dpsi`. On a 64^3, 31-view problem with
  the volume fixed to the truth, the mean in-plane error goes from 0.275 to
  0.051 deg (noise-free) and from 1.92 to 0.73 deg (5 % noise; start
  0.87). The returned Euler angles still satisfy `psi = -phi` exactly, and
  `pose_init_euler_deg` is still used in iteration 0 (its `psi` up to
  one ulp of rounding).
- `run_ad_recon` uses the fused CUDA Adam (`fused=True`) for the volume,
  angles and shifts: L2 iteration 1.81 -> 1.72 ms at 256^3 (0.30 -> 0.25
  ms at 128^3), peak 352 -> 304 MiB (no volume-sized temporary). Results move at the ulp level (all `ad_*` goldens were
  regenerated). `run_grad_descent` keeps the caller's optimiser as given.
- `run_grad_descent(optimizer_cls=torch.optim.SGD)` without `lr` now uses
  the Landweber step `1 / L` of the data term (`L = 2 ||B A|| / M`, 20
  seeded power iterations) instead of 1e-2, which was 3 400 to 13 400
  times smaller at 64^3 to 256^3. `lr` defaults to None; with Adam (the
  default) or any other optimiser it still means 1e-2, bit-identical.
- Docs: the loss recorded for iteration `i` (`loss_history[i]`, the
  callback value) is the loss of the iterate entering iteration `i`,
  before its update, for SIRT-type algorithms and AD; the `loss_callback`
  contract (values, indices, types) is unchanged.

### CPU projection kernels: Y-batched SIMD, 5-9x faster, same bits

The single-axis CPU kernels computed the sample geometry of every ray step
once per Y slice in scalar code. They now process Y in blocks of 16 slices
on Y-innermost copies, compute the geometry once per step and apply it to
the block with a SIMD loop, over a clipped step range; every lane does the
scalar loop's fp32 operations in the same order, so the results are
**bit-identical** to the previous kernels (memcmp over 11 shapes x
1/3/8/16 threads). Interleaved A/B on an EPYC 9555 (default portable
build, median of 20 alternating calls), 1 / 8 / 16 threads:

| kernel | 8x128^2, 61 views | 32x256^2, 61 views |
|---|---|---|
| `forward_project` (parallel) | 5.8x / 5.7x / 5.2x | 8.6x / 7.2x / 6.8x |
| `back_project` (ray-driven, parallel) | 5.8x / 3.7x / 4.2x | 8.8x / 6.7x / 4.8x |
| gather back-projection (W-BP, wSIRT, CPU FBP) | 5.3x / 5.1x / 4.5x | 6.5x / 6.8x / 6.2x |
| `grad_angle` | 5.6x / 5.5x / 5.4x | 7.3x / 7.8x / 7.4x |
| `forward_project_3d` (rotmat) | 1.7x / 1.7x / 1.7x | 1.8x / 1.5x / 1.5x |
| `grad_rotmat` | 1.2x / 1.2x / 1.2x | 1.2x / 1.2x / 1.4x |

- The parallel back-projection now uses all threads for any Ny (it
  parallelised over Y only; Ny = 1 ran on one thread: 7x at 8 threads for
  1x256^2), and its result does not depend on `num_threads`.
- The rotmat forward and `grad_rotmat` clip rays to the volume and load
  interior trilinear corners directly (1.7x and 1.1-1.2x, bit-identical).
- **`grad_angle` (CPU) changes in the last bits**: its per-angle sums are
  now fp64 in an order fixed by the geometry, so the result no longer
  depends on the thread count, and its error against an fp64 evaluation
  drops from 1e-6..8e-6 to about 2e-7 (only the `cpu/fp_grad_angles`
  golden key was regenerated).
- CPU SIRT/SART/FISTA/GD iterations are dominated by these kernels (CPU
  `run_sirt` 8x128^2 x 61 views, 16 threads: 9.8 -> 1.7 ms/iter).
- Scratch memory: the kernels allocate up to Nz*Nx*16 floats (plus
  n_angles*N_det*16 for the back-projection) per call; an allocation
  failure raises MemoryError.

### Non-finite tilt angles raise ValueError

A NaN or Inf tilt angle crashed the process on CPU (the parallel kernels
indexed memory with `(int)NaN`). Host-side angles and rotation matrices
holding NaN or Inf now raise ValueError at every entry point, on both
backends (CUDA-resident angles are not checked, which would add a sync to
every CUDA projection; the CUDA parallel kernels skip such a view, see the
PR B entry above), and the CPU kernels skip NaN samples instead of reading
out of bounds.

### Build: `-ffp-contract=off` for the C kernels

`FASTTOMO_MARCH_NATIVE=ON` let the compiler fuse `a*b + c` into FMA
instructions, which changed CPU results in the last bits (and
`grad_rotmat` by up to 3.4e-2 of its max at trilinear kinks) without the
golden fingerprint noticing. Both builds now pass `-ffp-contract=off`: a
native build gives the same bits as the portable one and is still 1.1-2x
faster at 32x256^2 (AVX-512). The default build is unchanged.

### `OMP_NUM_THREADS` lists

`OMP_NUM_THREADS="4,2"` (OpenMP's per-nesting-level list) made the CPU
kernels run on 1 thread; they now use the first entry. A value that is not
a thread count warns and uses 1. The `num_threads` documentation now says
to set torch's intra-op threads to the same count for CPU solver loops
(both share one OpenMP runtime; mismatched they oversubscribe the cores:
1.7 vs 20.8 ms per CPU SIRT iteration above).

### ⚠ CPU `run_fbp` output changes: shift-invariant back-projection (more accurate)

CPU `run_fbp` back-projected with the CPU forward projector's ray-driven
transpose, which is not shift-invariant: a flat field back-projects to a
2-5 % voxel pattern, so a reconstructed uniform cylinder had 0.5-1.1 %
interior noise and 0.9-2.5e-2 RMS (0.08-0.30 max) error against an
independent fp64 FBP, and CPU and CUDA FBP differed by 1.6-3.9e-2. It now
back-projects with the voxel-driven gather kernel that the CPU W-BP
already uses (the CUDA back-projection's operator, with exact fp32
weights): 2.7-4.3e-7 RMS (1.6-6e-6 max) against the fp64 FBP, CPU vs CUDA
1.1-1.6e-4 (the CUDA texture unit's 8-bit weights). **CPU `run_fbp`
results differ from earlier versions** (also with a CPU
`ParallelProjector` as `projector=`); CUDA `run_fbp` is unchanged. The CPU
FBP stays differentiable w.r.t. the sinogram, but **its gradient is now
approximate**: the backward is the CPU forward projection, which is not the
exact adjoint of the gather (relative inner-product gap 2-5e-2 on
positive random data, 0.3-1.3 on zero-mean random data, i.e. at high
frequencies; the same pairing CUDA's back-projection has), where the old ray-driven back-projection was
the exact transpose of the CPU forward (gap 5e-9). The back-projection step
is also faster (5x at 8x128^2, 61 views, with the Y-batched gather above).
Only
the `cpu/fbp`, `cpu/fbp_h` and `cpu/fbp_mask` golden keys were regenerated.

### ⚠ `run_mbir`: the qGGMRF prior is now the paper's (all MBIR results change)

The kernels computed the surrogate of `|D|^p / (c sigma_f^p + |D|^(p-q) sigma_f^q)`
with `p=1.2, q=2`, i.e. `(D/sigma_f)^2 / (1 + c |D/sigma_f|^0.8)`: `p` and `q`
had swapped roles, so the prior stayed quadratic (Gaussian MRF) for
differences up to ~1e4 `sigma_f` and did not preserve edges. Both backends now
implement the potential of Venkatakrishnan et al. 2015 (and OpenMBIR-TEM),
`rho(D) = |D/sigma_f|^q / (c + |D/sigma_f|^(q-p))`: quadratic (`q=2`) below
about `c^(1/(q-p)) sigma_f` (1.8e-4 `sigma_f` at the defaults), `|D|^p`
(`p=1.2`) above. The keyword names and defaults (`qggmrf_p=1.2`,
`qggmrf_q=2`, `qggmrf_c=1e-3`, `sigma_f=1.0`) are unchanged.

- **Every MBIR output changes**, CPU and CUDA. On a 96^3 three-ellipsoid
  phantom (41 views +-60 degrees, 1 % noise, fixed sigma, 80 iterations) the
  best RMSE over `sigma_f` goes from 0.0591 to 0.0463 (-22 %; serial sweep,
  as on CPU; 0.0477 with the new default parallel CUDA sweep, see below), at
  the same best `sigma_f = 0.1`; at the default `sigma_f = 1`, 0.0985 ->
  0.0766 (0.0783), and a fully default call 0.131 -> 0.100 (0.101). The
  BF-TEM demo (`benchmarks/scripts/run_mbir_bftem_demo.py`, its
  `sigma_f = 0.1`, CUDA default sweep): conventional MBIR 0.0283 -> 0.0260,
  MBIR-AM 0.0210 -> 0.0186 (SIRT 0.0296).
  The default `sigma_f` stays 1.0: it was already 10x above the best value
  on these phantoms and improves as much as the best value does.
- **`sigma_f` changes meaning near zero**: the prior is `1/c` = 1000x stiffer
  than `(D/sigma_f)^2` there. Re-tune values chosen for the old prior (a
  tenth of the expected contrast between neighbouring regions is a good
  start). ICD needs more iterations than with the old, quadratic prior: a
  noise-free 32^3 cube (`sigma_f=0.1`, `estimate_variance=True`) reaches a
  relative error of 0.15 / 0.049 at 80 / 300 iterations (0.086 at 80
  before), 2 multires levels meet the single-level estimate to 6.5e-5 after
  3000 iterations per level (5e-4 after 60 before), and a zero start with a
  small `sigma_f` and `estimate_variance=True` can barely move for tens of
  iterations.
- `qggmrf_p`, `qggmrf_q` must satisfy `1 <= p <= q <= 2`, `qggmrf_c`,
  `sigma_f` (and `sigma_f_per_scale`) must be positive, and
  `qggmrf_c * sigma_f**2 >= 1e-30` (below that the prior's fp32 curvature
  overflowed and every update became NaN, silently clamped to 0; also on
  main): `ValueError` otherwise. The old workaround for the paper's prior,
  `qggmrf_p=2, qggmrf_q=1.2`, now raises instead of meaning something else.
- The 26-neighbour weights are exact to 9 digits (the corner weight read
  0.030219 instead of 0.0302213, so they summed to 0.99999).
- Golden keys `cpu/mbir`, `cuda/mbir` and every key of
  `tests/data/mbir_sweep_golden.pt` were regenerated.

### ⚠ `run_mbir` on CUDA: parallel ICD sweep by default (new visiting order), `sweep_kernel=`

- New keyword-only `sweep_kernel="auto" | "serial" | "parallel"` on
  `run_mbir` and `MbirConfig`. The parallel sweep updates voxels of a row
  that are `S = ceil((2 + eps) / min|cos theta|)` apart (5 at +-60 degrees,
  6 at +-70, 12 at +-80) at the same time: their detector footprints are
  disjoint in every view, so this is still exact Gauss-Seidel ICD (it equals
  the serial sweep run on the same visiting order, bit for bit; tested). One
  block per Y-slice, the slice's error sinogram in shared memory. `"auto"`
  (the default) uses it on CUDA where it applies (a spacing `S <= Nx // 2`,
  i.e. no view near 90 degrees; at most 256 views; `n_angles * (N_det + 1)
  * 4` bytes of shared memory per block, 99 KB on sm_120) and the serial
  sweep otherwise (e.g. 512^3 at 61 views); `"parallel"` raises where it
  does not apply.
- **CUDA results for a given `seed` change** (groups visited in a keyed
  order instead of single voxels), and the group order needs 1.2-1.3x more
  iterations to the same `convergence_tol`. Measured (61 views +-60
  degrees, fp32, RTX PRO 6000 Blackwell, `sigma_f=0.1`, fixed sigma):

  | | serial | parallel |
  |---|---|---|
  | 128^3 per iteration | 9.72 ms | 1.25 ms |
  | 128^3 to tol 1e-3 / 1e-4 | 54 / 201 it, 525 / 1945 ms | 69 / 255 it, 86 / 318 ms |
  | 256^3 per iteration | 40.2 ms | 8.80 ms |
  | 256^3 to tol 1e-3 / 1e-4 | 33 / 151 it, 1326 / 6017 ms | 39 / 195 it, 343 / 1707 ms |

- **CPU and CUDA no longer visit the voxels in the same order by default**
  (the CPU keeps the serial voxel order); they agree at convergence (the
  same MAP estimate, tested through the MAP cost), not iteration by
  iteration. `sweep_kernel="parallel"` on CPU runs the C sweep in the
  parallel sweep's order, which matches a CUDA `"parallel"` run up to
  rounding; `"serial"` on both backends matches as before.
- The prepare kernel computes the prior sums in natural voxel order (3.0 ->
  1.5 ms at 256^3, 26 -> 12 ms at 512^3; bit-identical) and the CUDA sweep
  scratch is 8 instead of 16 bytes per voxel (128 MiB at 256^3, 1 GiB at
  512^3).
- The per-voxel arithmetic of every CUDA sweep and of the MBIR forward
  projection is rounded op by op (no FMA contraction, as in the C code), and
  each lane adds its even- and odd-slot sums before one warp reduction
  (instead of reducing them separately): serial CUDA results change at the
  ulp level too. The serial sweep, now the fallback, is slower than on
  main per iteration: +8.5 % at 128^3 (8.88 -> 9.64 ms), +2.9 % at 256^3,
  about equal at 512^3 (the natural-order prepare pays for it there).
- `sweep_kernel="auto"` decides from the GPU's opt-in shared-memory limit
  (64 KB on sm_75, 99 KB on sm_120), so the same problem and `seed` can run
  different sweeps, and give slightly different volumes, on different GPU
  models; `sweep_kernel="serial"` pins it.

### `run_mbir`: CUDA runs with a `support_mask` are not bit-reproducible (documented)

The part a support mask removes after each sweep is forward-projected with
fp32 atomics on CUDA, so two identical runs differ in the last bits, and
the ICD iterations amplify that (4 x 32^2 cylinder, 20 iterations: max
volume difference 3e-6 with the quadratic prior, 1e-3 with the default
prior, 0.2 with `huber_T=2`). CPU runs and CUDA runs without a mask stay
bit-reproducible for a fixed `seed`. Documented in `run_mbir` and tested
(bitwise on CPU, within 1e-4 on CUDA for the quadratic prior).

### ⚠ FBP output values change: zero-padded Ram-Lak ramp filter

`ramp_filter` multiplied the unpadded detector FFT by `|xi|`, which zeroes
the DC term and makes the convolution circular. `run_fbp` of a unit-density
cylinder came out at 0.96, 0.83 and 0.57 (radius 20, 40, 60 in a 128 grid)
instead of 1. The filter is now the spatial-domain Ram-Lak kernel
(`h[0] = 1/4`, `h[odd n] = -1/(pi n)^2`) applied with zero padding to the
next power of two `>= 2 N_det - 1`; the same cylinders reconstruct at
1.000 +- 0.002, at 180 degrees and at +-70 degrees alike. **`run_fbp` and
`ramp_filter` results differ from earlier versions** (only the FBP and
ramp-filter keys of the algorithm golden were regenerated).

- **Old FBP results cannot be converted by one factor.** The old error
  depended on the object's size (a unit cylinder read 0.96 to 0.57 from
  small to large, with a negative halo of -0.04 to -0.39 around it), so
  anything tuned on old FBP output needs re-tuning: thresholds,
  segmentation levels or colour limits (e.g. in the desktop app), and
  comparisons made through `benchmarks/scripts/utils/torch_radon_wrapper.py`,
  which imports `ramp_filter`.
- **FBP is 7-12 % slower**: the zero-padded FFT is at least twice as long.
  `ramp_filter` 0.07 -> 0.26 ms and 0.85 -> 2.25 ms, `run_fbp` 1.44 ->
  1.54 ms and 11.1 -> 12.5 ms, at 121 x 256^2 and 121 x 512^2 (CUDA).
- The `pi / n_angles` scale is unchanged. It is pi times the mean over the
  measured views, so it keeps the mean level correct at limited tilt too:
  a +-70 degree series of a rotationally symmetric object gives the same
  level as a full one (the scope report had suspected it was off by
  180/140). Anisotropic objects still show an orientation-dependent bias at
  limited tilt (a 4:1 ellipse at +-70 reads 1.20 or 0.50 depending on its
  orientation; the missing wedge), which no angular scale removes.
- fp16 `run_fbp`: the back-projection sums `n_angles` unscaled views
  before the `pi / n_angles` scale and could overflow (OverflowError)
  although the result fits (density 400 over 720 views). When it does,
  `run_fbp` now applies `pi / n_angles` to the filtered sinogram in fp32,
  before its cast to fp16, and back-projects again, so it raises only when
  the result itself exceeds the fp16 range. Otherwise nothing changes
  (bit-identical); pre-scaling always would lose small values to fp16
  underflow (a density of 1e-4 at 720 views: 8 % error instead of 0.08 %).

### ⚠ `run_mbir(multires_levels > 1)`: coarse levels hand over the right values

A multires level with factor `f` works on voxels `f` fine voxels wide, so
its values are `f` times the fine-grid ones. The hand-over to the next
finer level only resized the volume, so the fine level started at twice
the object's level, and a `warm_start` reached the coarse level at half
its values. Both are now rescaled (`x 0.5` per level, `x coarse_factor`
for a warm start). Multires results differ slightly from earlier versions
(the fine level starts 8x closer, data MSE 6.8 -> 0.82 on a test cylinder).
With `estimate_variance=False` they converge to the single-level solution
(5e-4 relative at 60 iterations per level). With `estimate_variance=True`
(the default) they reach the same fixed point only slowly, as does a single
level: 2-level vs 1-level differ by 22 / 17 / 10 / 3.6 / 0.3 % at 30 / 150 /
600 / 2400 / 9600 iterations per level on a noise-free 64^2 cylinder at
+-70 degrees (30-37 % before this fix at the low counts); with
`estimate_offset=True` as well, a common offset on every view stays
unsettled. This was investigated and is a property of the joint
volume/variance(/offset) estimate on noise-free limited-tilt data, not of
the level hand-over; see `multires_levels` in the `run_mbir` docstring.
Single-level runs are unchanged.

### fp16 overflow raises `OverflowError`

fp16 projections accumulate in fp32 but store fp16, so a result above
65504 (the back-projection of an unnormalised sinogram, the forward
projection of a large volume with large values, FISTA's first gradient)
came back as `inf`. The projection functions, the projector classes,
`run_fbp` and autograd backward now raise `OverflowError` with the remedy
(scale the data into range, or use fp32) instead; an `inf` that came with
the input is passed through. A NaN from finite input is reported
separately, as `FloatingPointError` with its own message (it used to be
reported as an overflow by the algorithms and not at all by the direct
calls); where an overflow came first, the overflow is reported. `run_sirt`, `run_sart`, `run_fista` and `run_fista_tv` check
their projections on the device and raise once when they return. fp32 is
never checked; a direct fp16 call on CUDA pays one flag read (0.06-0.07 ms measured).

### Changed

- `apply_shifts` and `apply_shifts_2d` return the sinogram's dtype (they
  returned fp32 for fp16 and fp64 input) and share one contract: shifts
  `[n_angles]` (detector axis) or `[n_angles, 2]` (detector axis, tilt
  axis), positive shifts move the signal toward larger indices, the same
  shape errors. `apply_shifts` gained the `[n_angles, 2]` form (Fourier
  shift along both axes) and `apply_shifts_2d` the `[n_angles]` form. A
  sinogram that is not 3-D raises `ValueError` (`apply_shifts` used to
  broadcast it into a wrong result).
- Size parameters default to `None` across the Python API (`Nz` of every
  reconstruction; `N_det`, `Nz`, `Nx` of the projection functions);
  `-1` is still accepted and means the same.
- fp16 SIRT / SART: a ray that grazes the volume (FP(1) below 1/65504,
  produced by the CPU rotation-matrix projector) gets a zero normaliser
  instead of `inf`. fp32 is unchanged.
- One CUDA projector used from two streams is safe: a call from another
  stream than the previous one waits for that stream first (the shared
  texture uploads used to race).
- `tilt_ssim_loss` and `s3im_loss` with `data_range=None` no longer
  synchronise the host (two syncs per `run_ad_recon` iteration with both
  weights on); values are bit-identical.

### Added

- `fasttomo.__version__` (from the installed distribution's metadata).
- `run_wbp` (alias of `run_weighted_backprojection`) and `tv_loss_zx`
  (alias of `tv_loss_xz`).
- cp314 wheels (CPU and CUDA); `wheels.yml` uses cibuildwheel v3.4.1.

### Deterministic, faster geometry gradients; faster rotmat BP and `refine_angles`

- **CUDA pose/angle gradients are now bit-reproducible.** `grad_rotmat`
  (∂L/∂rotation matrix: `forward_project_3d` / `RotmatProjector` /
  `run_ad_recon(refine_pose=True)`) and `grad_angle` (∂L/∂tilt angle of
  `forward_project`) summed their per-block partials with fp32
  `atomicAdd`, so repeated calls differed (20 of 20 runs distinct at
  256^3 / 61 views, up to 4.7e-2 absolute on the matrix gradient). Both
  now reduce in a fixed order (warp shuffles, one partial per block, a
  second kernel per view): the same inputs give the same bits. The values
  move by rounding only (rel-L2 5.7e-7 / 1.1e-6 vs before at 256^3; the
  error against an fp64 reference is unchanged), so `run_ad_recon` with
  pose refinement differs from earlier versions in the last bits.
- **`grad_rotmat` 1.86x faster** (256^3 fp32, 61 views: 4.73 -> 2.55 ms;
  fp16 volume 3.22 -> 2.42 ms; 128^3 0.62 -> 0.38 ms): the 9 KB
  shared-memory reduction that shrank the texture cache is gone, u, v and
  the sinogram gradient are factored out of the ray loop, and the ray
  march is clipped like the forward's (bit-identical to unclipped).
  It was 53 % of an AD pose iteration at 256^3.
- **Rotmat back-projection no longer slows down with many views**: the
  matrices are staged 64 views at a time instead of all at once, so the
  time per view stays at ~10.5 us (256^3) instead of rising to 17 us at
  1441 views (24.9-25.3 -> 15.2 ms, 1.64-1.66x); bit-identical. The 2730-view shared
  memory cap is gone (the 2048-layer texture limit remains).
- **`refine_angles` 2.0x / 4.1x faster** (single-axis / ZYZ, 128^3, 61
  views; 1.55x / 1.6-1.8x at 256^3): every view's candidates are projected
  in one batch (one per axis in ZYZ mode, chunks of at most 256 MB) under
  `torch.no_grad()`, with one host transfer per batch instead of 2 (6 in
  ZYZ mode) per view. Results are bit-identical. Its docstring now records
  the measured convergence, including the slow ZYZ convergence near
  phi/psi degeneracy.
- Internal: the gradient kernels overwrite their output (callers no longer
  need to zero it) and take their partial-sum scratch from a FastTomo
  stream-ordered memory pool per device that keeps up to 64 MiB reserved
  between calls (`csrc/cuda/scratch_pool.cuh`).

### Faster CUDA `run_sirt` (bit-identical results; new public `release_cached_projectors()` and `run_sirt(reuse_graph=)`)

- **API additions** (the PR text said "no API change"; it adds two public
  names): `fasttomo.release_cached_projectors()` (also in
  `fasttomo.algorithms`) frees what `run_sirt` keeps between calls and
  returns how many configurations it released (0 or 1); the new keyword
  `run_sirt(..., reuse_graph=True)` controls whether that is kept. Existing
  calls need no change, but a CUDA `run_sirt` without `projector=` now
  keeps GPU memory after it returns (below).
- Repeated `run_sirt` calls without `projector=` reuse the previous call's
  internal projector and CUDA graph when shapes, dtype, device and options
  match, instead of building a projector, running an eager warm-up iteration
  and capturing a new graph every call. New sinogram data, angles and
  support masks of the same shapes are copied into the graph's buffers.
- **Memory kept after `run_sirt` returns** (CUDA graph path, no
  `projector=`): exactly one configuration is kept, **4 volumes plus 4
  sinograms**: the projector's volume and sinogram texture arrays (1 + 1),
  the graph's static volume and voxel normalisation (2 volumes) and static
  sinogram and ray normalisation (2 sinograms), and the graph's memory pool
  holding the back-projection and forward-projection outputs (1 + 1), plus
  the support mask when one is given (a `(1, Nz, Nx)` plane, or one more
  volume for a full-size mask). In fp32 at 61 views that is 0.31 GiB at
  256^3 and 2.24 GiB at 512^3 by count; the device memory actually taken,
  measured, is **0.36 GiB at 256^3 and 2.62 GiB at 512^3**, the difference
  being free space inside the caching allocator's segments that hold the
  static tensors (PyTorch reuses it, `torch.cuda.empty_cache()` cannot
  return it while the tensors live). A call with a different shape, dtype,
  device or option drops the kept one before allocating its own. New public `fasttomo.release_cached_projectors()`
  frees it (the texture arrays go back to the driver at once, the rest to
  PyTorch's caching allocator; follow with `torch.cuda.empty_cache()` to
  hand that back to the driver too). New keyword `run_sirt(...,
  reuse_graph=False)` keeps nothing, for a single reconstruction that
  needs the memory back.
- A graph capture that fails (e.g. out of memory, or several threads
  capturing at once) makes that configuration run the eager loop, with the
  same results. The failure is no longer remembered for the rest of the
  process: `release_cached_projectors()` or a call with another
  configuration clears it, and the next call captures again.
- The iteration's elementwise work runs in two fused CUDA kernels (residual;
  update plus support, positivity and soft-threshold constraints) instead of
  up to seven PyTorch kernels, rounding exactly like the PyTorch chain.
- The residual and the updated volume are written straight into the
  projector's textures, removing the two device-to-device texture uploads
  per iteration (102 -> 3 per 50-iteration call); on the graph path the
  volume stays in the texture for the whole call.
- `run_sirt(n_iter=50)`, 61 views, RTX PRO 6000 Blackwell: 128^3 11.3 ->
  9.5 ms (graph path), 12.1 -> 11.4 ms (eager path); 256^3 80.9 -> 76.7 ms
  (graph), 86.6 -> ~80.8 ms (eager). Outputs are bit-identical to before.
- The `tv_loss_3d` backward no longer zero-fills its output (the kernel
  writes every voxel): 256^3 143 -> 95 us.

### ⚠ `run_mbir` is single-threaded on CPU by default

`run_mbir` used to run its CPU ICD sweep on a hardcoded 4 OpenMP threads. It
now follows the library-wide policy (`num_threads` keyword, else the
`OMP_NUM_THREADS` environment variable, else 1), so **a default CPU `run_mbir`
call is up to ~4x slower than before**. Results are unchanged. To restore
the old speed pass `num_threads=4` (or more), or set `OMP_NUM_THREADS`
before starting Python. CUDA runs are not affected.

### ⚠ `run_mbir`: faster outer iterations, different voxel order per `seed`

- The random per-slice voxel order of the ICD sweep is now generated on the
  GPU (a keyed Feistel permutation per Y-slice and outer
  iteration) instead of by NumPy on the host and copied to the GPU every
  outer iteration (`Ny*Nz*Nx*4` bytes: 67 MB at 256^3, 537 MB at 512^3).
  The CPU sweep uses a bit-identical NumPy mirror of the same order, so CPU and
  CUDA still visit the voxels in the same order.
- The CUDA ICD sweep was restructured: a fully parallel prepare kernel
  computes the voxel order, pre-sweep values and qGGMRF prior sums of every
  step, and the sequential sweep runs one warp per Y-slice with a fused
  warp-shuffle reduction and no block barriers (it had about 11 per voxel),
  keeping each lane's error-sinogram values in registers. The sweep no
  longer builds a texture or synchronises the stream. Same Gauss-Seidel
  order and arithmetic: bit-identical to the previous kernel for the same
  voxel order. One CUDA outer iteration at 61 views: 128^3 52 -> 9.6 ms,
  256^3 300 -> 41 ms (default settings; 1.9-2.5x on the sweep with lambdas
  and/or a finite `huber_T`). It needs a scratch buffer of 16 bytes per
  voxel for the duration of `run_mbir` (the host permutation it replaces
  was 4 bytes per voxel on the host plus 4 on the GPU).
- **Results for a given `seed` differ from earlier versions** (a different,
  equally random order); a fixed `seed` is still bit-reproducible run to
  run. `_generate_permutation` is no longer used by `run_mbir`.

### Faster 3D forward projection, deterministic TV loss, faster CPU wSIRT

- `forward_project_3d` / `RotmatProjector.forward` (CUDA, fp32 and fp16)
  march each ray only over the steps whose trilinear stencil can touch the
  volume (slab test, united over the thread block); 43-47 % of the texture
  fetches used to return border zeros. The sinogram is bit-identical.
  Kernel time at 61 views: 128^3 1.43-1.49x, 256^3 1.67-1.69x faster. The
  matrix gradient (`grad_rotmat`) is unchanged: its bounds test already
  skipped those fetches and clipping measured 0.97-0.98x.
- `tv_loss_3d` (CUDA forward) is a fixed-order two-pass reduction with fp64
  partials instead of one fp32 `atomicAdd` per 128 voxels: **bit-reproducible
  run to run** (it varied in the last digits) and closer to the exact value
  (within 1e-7 of an fp64 reference, was up to 6e-6 on random volumes and
  1.3e-3 on a 512^3 phantom), so the loss value can differ from earlier
  versions in its last digits. 1.8x (128^3) to 4x (256^3, 512^3) faster.
  The gradient is unchanged.
- The CPU weighted back-projection (`run_weighted_backprojection`,
  `run_wsirt`) runs its voxel-driven back-projection in C/OpenMP instead of
  a per-view PyTorch loop; results are bit-identical. wSIRT at
  8x128^2x61: 81.5 -> 52.2 ms/iter on 1 thread, 57.5 -> 8.9 ms/iter on 8.
  `run_wsirt`'s `num_threads` now also drives this step, and
  `run_weighted_backprojection` takes `num_threads` (keyword-only, see
  Conventions; it used whatever threads PyTorch had).
- `run_mbir`: an explicit voxel order passed to the CUDA sweep (test hook)
  is validated (shape, range, each row a permutation); unused angle slots in
  the sweep get defined values. Results are unchanged.

### Projection error messages and double backward

- Zero-size errors from `forward_project`, `back_project`,
  `forward_project_3d`, `back_project_3d` and `refine_angles` name that
  function (they named an internal operator class, with `-1` placeholders
  for the sizes still to be taken from the input). Still `ValueError`.
- `refine_angles(n_steps < 1)` raises `ValueError` on both backends (a
  negative value raised `RuntimeError` on the CPU).
- A second-order gradient through a projection (`create_graph=True`, then
  a backward through the gradient) raises `RuntimeError` instead of
  silently dropping the projection's second-order term; the adjoint
  kernels are not themselves differentiable.

### Projection-operator seam (internal; outputs unchanged)

- Every forward/back projection now goes through one internal interface,
  `fasttomo._operator` (`ProjectionOperator` with `fp` / `bp` / `grad_geom`
  and adapters for CPU, CUDA with cached textures, and the frozen RESIRE
  kernels), chosen in one place (`_make_operator`). The functional API, the
  projector classes and every algorithm use it; two generic autograd
  Functions (`_Project`, `_BackProject`) replace the eight per-path ones in
  `fasttomo.autograd` (`_FPParallel`, `_FPParallelCached`, ... were private).
  No public API change, and every output is bit-identical to before
  (`tests/test_algorithm_identity.py`, generated on the previous `main`).
- New: `ParallelProjector` and `RotmatProjector` accept `device="cpu"` (and
  a keyword-only `num_threads`); on CPU they give exactly the functional
  API's results. Before, a CPU device raised `ValueError`.
- Faster on CUDA: `run_sart`, `run_wsirt`, `run_grad_descent` and
  `refine_angles` reuse texture caches instead of allocating a texture and
  synchronising on every projection (e.g. `run_sart` 4-6x, `refine_angles`
  2-3x faster on small problems).
- `run_fbp` on CUDA without a projector uses the one-shot texture path
  (numerically identical).

### Internal restructure (no behaviour change)

- `fasttomo/algorithms.py` is now the package `fasttomo/algorithms/`
  (`_common`, `analytic`, `algebraic`, `fista`, `ad`, `genfire`, `resire`,
  `alignment`, `mbir`). `fasttomo.algorithms` re-exports every name the
  module had, so imports such as `from fasttomo.algorithms import run_sirt`
  or `fasttomo.algorithms._prepare_recon` keep working. Code that
  monkeypatches a name an algorithm calls must patch the submodule that
  calls it (for example `fasttomo.algorithms.ad.ParallelProjector` or
  `fasttomo.algorithms._common._check_inputs`), not the package.
- Backend helpers (`_on_device`, `_resolve_num_threads`, the dtype codes,
  `_stream_ptr`, `_data_ptr`, `_check_shapes`) live in `fasttomo._backend`;
  their old `fasttomo.autograd` / `fasttomo.projector` paths still resolve.
- `apply_shifts` and `apply_shifts_2d` are defined in `fasttomo.projection`
  and `tv_loss_3d` in `fasttomo.regularizers`; `fasttomo.utilities` and
  `fasttomo` re-export all three. This removes the import cycle between
  `fasttomo.utilities` and `fasttomo.projection`.
- Existing editable installs (`pip install -e .`) made before this split must
  be reinstalled: scikit-build-core's editable finder records module paths at
  install time and still points `fasttomo.algorithms` at the deleted
  `algorithms.py`, so `import fasttomo` fails until you re-run the install.
- Tests are grouped by topic (`tests/test_projection.py`,
  `test_algorithms_*.py`, `test_contracts.py`, ...) with shared fixtures in
  `tests/conftest.py`; `gpu` and `multi_gpu` markers are registered, so
  `pytest -m "not gpu"` runs the CPU tests only.

### Breaking changes and deprecations (API consistency refactor)

Deprecated names keep working for one release and emit `DeprecationWarning`;
passing an old and a new name together raises `TypeError`.

**Call signatures.** Every reconstruction takes `(sinogram, angles_deg, n_iter)`
positionally and everything after that keyword-only (`run_fbp` and
`run_weighted_backprojection`: `(sinogram, angles_deg)`). A call written for an
old positional order now raises `TypeError` instead of binding a value to
another parameter. Keyword calls are unaffected. In detail:

| Function | Before | Now |
|---|---|---|
| `run_fbp` | `(sino, angles, Nz, positivity, mask, projector, support_mask)` | `(sino, angles, *, Nz, positivity, mask, support_mask, projector, num_threads)` |
| `run_weighted_backprojection` | `(sino, angles, Nz, positivity, threshold_scale, mask)` | `(sino, angles, *, Nz, positivity, threshold_scale, mask, support_mask)` |
| `run_sirt`, `run_sart`, `run_wsirt`, `run_fista`, `run_resire`, `run_grad_descent` | options positional after `n_iter` | keyword-only after `n_iter` |
| `run_fista_tv` | `(sino, angles, tv_weight, n_iter, ...)` | `(sino, angles, n_iter, *, tv_weight, ...)`; `tv_weight` still required |
| `run_genfire` | `(sino, angles, Nz, n_iter, ...)` | `(sino, angles, *, n_iter, Nz, ...)`: `n_iter` keyword-only for one release, so `run_genfire(s, a, 64)` (old `Nz=64`) raises |
| `run_ad_recon` | `(sino, angles, *, ...)` | `(sino, angles, n_iter=None, *, ...)`; `n_iter` overrides `config.optim.n_iter` without modifying the config |
| `run_mbir` | `(sino, angles, *, ...)` | `(sino, angles, n_iter=100, *, ...)` |
| `refine_angles` | `(volume, sino, angles_deg, euler_zyz_deg, search_range_deg, n_steps, num_threads)` | `euler_zyz_deg` and after keyword-only |
| `ConstraintConfig` | `(positivity, mask, sparsity)` | `(positivity, *, cylinder, support_mask, soft_threshold)` |

**Renamed keywords and fields (deprecated aliases).**

| Where | Old | New |
|---|---|---|
| `run_sirt`, `run_sart`, `run_fista`, `run_fista_tv` | `volume=` (warm start) | `warm_start=` |
| `run_resire` | `support=` | `support_mask=` |
| `run_mbir` | `outer_iters=` | `n_iter=` (still the per-level maximum) |
| `run_mbir` | `return_history=True` -> `(volume, d_offsets, sigma, history)` | `return_loss=True` -> `(volume, loss_history)`; `return_state=True` appends `(d_offsets, sigma)`. `return_history` still returns the legacy 4-tuple |
| `ConstraintConfig` | `mask: bool \| Tensor` | `cylinder: bool` and `support_mask: Tensor \| None` (constructor keywords and attributes) |
| `ConstraintConfig` | `sparsity` | `soft_threshold` (it collided with `RegularizerConfig.sparsity`, a loss term) |
| `fasttomo.autograd._euler_zyz_to_rotmat` | private | public `fasttomo.euler_zyz_to_rotmat`; the private name stays as an alias |
| `fasttomo.utilities.cylindrical_mask` | defined there | defined in `fasttomo.constraints`, re-exported from `fasttomo` and `fasttomo.utilities` |
| `fasttomo.configs._apply_sparsity` | | alias of `fasttomo.constraints._soft_threshold` |

**Removed.** `estimate_shifts_xcorr(upsample_factor=...)`: it never had an
effect (the sub-pixel shift comes from a parabolic fit of the correlation
peak). Passing it raises `TypeError` with that explanation.

**Behaviour changes.**
- `run_mbir` raises `TypeError` for a non-fp32 sinogram, `lambdas` or
  `warm_start` instead of casting silently, like the other fp32 algorithms.
- `run_mbir` uses `num_threads` (keyword, else `OMP_NUM_THREADS`, else 1)
  instead of a hardcoded 4 CPU threads. Results are identical; the default is
  single-threaded unless `OMP_NUM_THREADS` is set (see the warning at the top:
  pass `num_threads=4` to restore the old speed).
- `run_ad_recon`'s tqdm bar is opt-in (`progress=True`); it was always shown.
- `run_resire(mask=True, support_mask=T)` (and the old `support=T`) raises
  `ValueError`; before, `support` silently won. Same contract as every other
  algorithm: one support per reconstruction.
- `run_genfire`'s `mask` defaults to `None`: the cylinder unless
  `support_mask` is given (calls without `support_mask` are unchanged).
- `n_iter < 0`, a non-integer `n_iter`/`Nz`, a negative or non-finite
  `sparsity`, and a non-tensor `sinogram`/`angles_deg`/`warm_start` raise
  instead of running zero iterations or failing later.
- `ConstraintConfig.apply()` honours `cylinder=True` (a bool `mask` used to be
  ignored silently); `validate()` rejects `cylinder` together with
  `support_mask`.
- Warm starts are detached before use, so a `requires_grad` warm start no
  longer drags an autograd graph through SIRT and friends.
- The SIRT CUDA-graph cache and its failed-capture set live on the projector
  object (declared on the projector base class); a failed capture now only
  disables graphs for that projector, not for the whole process.
- `config.validate()` accepts numpy and one-element tensor scalars again
  (`np.int64` iteration counts, `np.float32` or tensor learning rates), which
  #82 had started to reject.

### Added
- `loss_callback` and `progress` on every iterative algorithm (SIRT, SART,
  wSIRT, FISTA, FISTA-TV, GENFIRE, RESIRE, gradient descent, AD recon, MBIR),
  with the frozen 2026-06 callback contract. Newly streamed losses: SART's
  full-sinogram data MSE per outer iteration, GENFIRE's Fourier L1 error,
  gradient descent's total loss (components `fidelity`, `tv`), MBIR's data
  MSE `mean((g - A f - d)^2)` per outer iteration (indices run across
  multires levels).
- `warm_start` on every iterative algorithm (new on wSIRT, GENFIRE, RESIRE,
  gradient descent, AD recon and MBIR).
- `support_mask` on every algorithm (new on FISTA, FISTA-TV, SART, wSIRT,
  WBP, GENFIRE, RESIRE, gradient descent and MBIR; closes #54), plus `mask`
  (cylinder) on MBIR. Where each applies it is tabulated in the
  `fasttomo.algorithms` module docstring.
- `num_threads` honoured on every CPU kernel path (the single-axis projections
  of SIRT, SART, FISTA, FISTA-TV, wSIRT, FBP, gradient descent,
  `refine_angles` and `estimate_shifts_xcorr` ignored it before).
- `fasttomo.constraints` module: `cylindrical_mask`, `apply_constraints`,
  `apply_constraints_`, and the written constraints order and support-mask
  contract.
- `MbirConfig`: optional bundle of `run_mbir`'s tuning keywords.
- `fasttomo.euler_zyz_to_rotmat` exported.
- `ruff` rule set pinned in `pyproject.toml`; the tree is clean and the CI step
  is blocking.

### Earlier changes in this cycle

**#84 wSIRT CPU/CUDA parity.** `run_weighted_backprojection` and `run_wsirt`
share one weighted back-projection: voxel-driven gather on both backends and
a padded grid (edge margin max(2, N // 16)). CPU and CUDA now agree to ~6e-4
(the CPU W-BP had 83 % one-step error vs 30 % on CUDA), and wSIRT's residual
no longer rises at `omega=1`. Output values of both functions change.

**#82 Python-side correctness.**
- `back_project`, `back_project_3d` and `back_project_resire` gain a
  keyword-only `Nx`, so `forward_project*(..., N_det != Nx)` works under
  autograd.
- `run_ad_recon` applies `ConstraintConfig.sparsity` (now `soft_threshold`)
  as a post-step soft-threshold; it was ignored. Skipped in pose-only
  iterations.
- RESIRE wrappers validate shapes (ValueError instead of out-of-bounds reads
  or silent broadcasting); outputs unchanged.
- Zero-size inputs (`n_angles`, `Ny`, `Nz`, `Nx` or `N_det` of 0) raise
  ValueError on CPU and CUDA (CPU used to return empty tensors); the
  projector constructors reject zero-size geometries.
- `run_mbir` raises "sinogram contains NaN or inf values" up front.
- Cosine LR schedules under AD staging use each optimiser's number of active
  iterations as `T_max`; every config has `validate()`, called by
  `run_ad_recon` on entry; `pose_warmup_iters > 0` without a refine flag
  raises.
- MBIR offset/variance updates: one Python implementation with fp64
  accumulation for CPU and CUDA (the private `_core.mbir_icd_sweep_cpu`
  signature lost its `d_offsets`, `sigma_min` and `estimate_*` arguments).

**#81 kernel and binding correctness.**
- CUDA `grad_rotmat` used tight box bounds and dropped face contributions
  (~100 % error on a box-filling volume); it now matches the forward and
  finite differences.
- One-shot CUDA wrappers no longer leak the cudaArray on errors or destroy
  textures while kernels run (scoped texture cache).
- CPU bindings take `data_ptr` ints like CUDA and release the GIL; strided or
  non-contiguous inputs are densified (they returned wrong numbers), every
  buffer size is validated before a pointer is taken, and the CUDA angle
  gradient no longer returns garbage for fp16 or strided volumes.
- Internal C/CUDA names use `parallel`/`rotmat` instead of `2d`/`3d`; the
  public Python API is unchanged.

**#79 staged AD pose optimisation.** `PoseRefinementConfig` gains
`pose_warmup_iters`, `alternating_period`, `pose_init_euler_deg` and
`pose_init_shifts`; defaults keep the joint loop.

**#78 consolidated correctness fixes.**
- `run_mbir` makes `lambdas` contiguous (a strided `lambdas` was read as
  dense) and validates `lambdas`, `sigma_init` and `sigma_min`.
- `run_ad_recon`'s sparsity term is `sparsity * mean(|v|)`; it was the signed
  mean, which pushed the volume negative.
- `RotmatProjector` casts rotation matrices to contiguous fp32 (fp64/fp16 or
  strided matrices were misread).
- `run_ad_recon` checks the projector output shape before the loss.

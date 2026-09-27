#pragma once
/*
 * common.h — shared helpers of the CPU projection kernels
 * (projection_parallel.c, projection_rotmat.c).
 */
#include <math.h>

/* Border-zero trilinear sampler for a [Ny, Nz, Nx] volume `vol`.
 * Returns 0 for any index outside its valid range; used by the 3D CPU
 * projection kernels to match CUDA cudaAddressModeBorder semantics. */
#define SAMP3(ix, iy, iz)                                                  \
    (((ix) >= 0 && (ix) < Nx && (iy) >= 0 && (iy) < Ny &&                  \
      (iz) >= 0 && (iz) < Nz)                                              \
     ? vol[((size_t)(iy) * Nz + (iz)) * Nx + (ix)] : 0.0f)

/* Y lanes per block of the Y-batched parallel-beam kernels. The geometry of
 * a ray sample (position, stencil, weights) does not depend on the Y slice,
 * so it is computed once per sample and applied to FT_YB slices with a SIMD
 * loop over a Y-innermost copy of the block ([Nz][Nx][FT_YB] or
 * [n_angles][N_det][FT_YB]). 16 lanes = one AVX-512 or four SSE2 vectors;
 * the scratch per block is Nz * Nx * FT_YB floats (4 MiB at 256^2). Every
 * lane performs exactly the scalar kernel's operations in the same order,
 * so the result is bit-identical to the per-slice loop. */
#define FT_YB 16

/* Ray clipping. Narrow the integer step range [*lo, *hi] to the steps s at
 * which a sample coordinate c(s) = a*s + b can lie in [lo_v, hi_v].
 *
 * The caller computes c(s) in fp32 with a few roundings of values of
 * magnitude <= `mag`, so the computed coordinate differs from the exact
 * a*s + b by far less than mag * 2^-20; the interval is widened by that much
 * in value, and by one step on each side, and evaluated in double. The
 * caller keeps its exact in-loop bounds test, so clipping only removes steps
 * that test would reject: the samples taken, and therefore the result bits,
 * are unchanged. A non-finite a or b (NaN/Inf angle) leaves the range
 * unchanged and the in-loop test decides. An empty result has *lo > *hi. */
static inline void ft_clip_steps(double a, double b, double lo_v, double hi_v,
                                 double mag, int *lo, int *hi)
{
    if (*lo > *hi || !isfinite(a) || !isfinite(b)) return;
    const double slack = mag * 0x1p-20 + 1e-6;
    const double vlo = lo_v - slack, vhi = hi_v + slack;
    if (a == 0.0) {
        if (b < vlo || b > vhi) { *lo = 1; *hi = 0; }
        return;
    }
    double s1 = (vlo - b) / a, s2 = (vhi - b) / a;
    if (s1 > s2) { double t = s1; s1 = s2; s2 = t; }
    const double l = floor(s1) - 1.0, h = ceil(s2) + 1.0;
    if (h < (double)*lo || l > (double)*hi) { *lo = 1; *hi = 0; return; }
    if (l > (double)*lo) *lo = (int)l;
    if (h < (double)*hi) *hi = (int)h;
}

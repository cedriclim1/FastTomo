#pragma once
/* ── fasttomo/csrc/cuda/parallel_clip.cuh ────────────────────────────────────
 * Ray clipping of the single-axis kernels (forward_project_parallel_kernel,
 * grad_angle_kernel): the integer step range of a ray whose samples can
 * touch the slice under the loose sampling rule.
 */

#include <climits>
#include <math.h>

/* ── ray clipping (parallel FP and grad_angle) ─────────────────────────────
 * A ray (t, angle) samples fx(s) = (t*ca - s*sa) + half_x and
 * fz(s) = (t*sa + s*ca) + half_z at the integer steps s in [-ray_len,
 * ray_len]; only ~65 % of them pass the bounds test (-1, N) on both axes.
 * clip_steps narrows [lo, hi] to the steps at which the coordinate
 * c(s) = a*s + b can lie in [lo_v, hi_v] (the fp32 form of the CPU kernels' ft_clip_steps,
 * csrc/cpu/common.h; fp64 cost 1.6x at 128^3 on this consumer-rate part):
 *
 * - Rounding: the kernel computes c(s) with at most 3 fp32 roundings (FMA
 *   contraction or not) of values of magnitude <= mag = |t| + ray_len +
 *   max(half_x, half_z) + 1, and b and the two numerators below add 4 more,
 *   so the computed coordinate and the computed bound differ from the exact
 *   a*s + b by less than 7 * 2^-24 * mag < 2^-21 * mag. The interval is
 *   widened by mag * 2^-18 in value (8x that) and one step on each side
 *   (the quotient's own relative error, 2^-24 of |s| <= 2^23 steps, is below
 *   half a step).
 * - The loop keeps its exact bounds test, so clipping only drops steps that
 *   test would reject: the samples taken, and the result bits, do not change
 *   (for grad_angle's inclusive test [-1, N] too: the clip interval is the
 *   closed [-1, N] plus slack).
 * - A non-finite a or b (NaN/Inf angle) leaves [lo, hi] as is; the in-loop
 *   test then rejects every sample. An empty result has lo > hi.
 */
__device__ __forceinline__ void clip_steps(float a, float b, float lo_v, float hi_v,
                                           float mag, int &lo, int &hi) {
    if (lo > hi || !isfinite(a) || !isfinite(b)) return;
    const float slack = mag * 0x1p-18f;
    const float vlo = lo_v - slack, vhi = hi_v + slack;
    if (a == 0.0f) {
        if (b < vlo || b > vhi) { lo = 1; hi = 0; }
        return;
    }
    float s1 = __fdiv_rn(vlo - b, a), s2 = __fdiv_rn(vhi - b, a);
    if (s1 > s2) { float tmp = s1; s1 = s2; s2 = tmp; }
    const float l = floorf(s1) - 1.0f, h = ceilf(s2) + 1.0f;
    if (h < (float)lo || l > (float)hi) { lo = 1; hi = 0; return; }
    if (l > (float)lo) lo = (int)l;      /* in (lo, hi]: the cast is exact */
    if (h < (float)hi) hi = (int)h;
}

/* The step range [lo, hi] of one ray, clipped to the samples whose bilinear
 * stencil can touch the volume: fx in [-1, Nx], fz in [-1, Nz] (the loop's
 * test is the open (-1, N); the clip keeps the closed range and its slack).
 * ray_len covers that box: a sample in it is at most
 * 0.5 * sqrt((Nx+1)^2 + (Nz+1)^2) < ceil(0.5 * sqrt(Nx^2 + Nz^2)) + 1 steps
 * from the ray's foot. */
__device__ __forceinline__ void fp_step_range(float t, float ca, float sa, int Nx, int Nz,
                                              float half_x, float half_z, int ray_len,
                                              int &lo, int &hi) {
    lo = -ray_len;
    hi = ray_len;
    const float mag = fabsf(t) + (float)ray_len + fmaxf(half_x, half_z) + 1.0f;
    clip_steps(-sa, t * ca + half_x, -1.0f, (float)Nx, mag, lo, hi);
    clip_steps(ca, t * sa + half_z, -1.0f, (float)Nz, mag, lo, hi);
}

/* Widen [lo, hi] to the union over the full warp (every lane must call it;
 * an idle lane passes lo = INT_MAX, hi = INT_MIN). All lanes of a warp then
 * walk the same steps s at the same time: per-lane ranges measured slower,
 * lanes starting at different s scatter the texture fetches (audit P6).
 * With FP_BLOCK_X = 32 a warp is one angle, so its rays have the same
 * direction and similar chords. */
__device__ __forceinline__ void warp_uniform_range(int &lo, int &hi) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    lo = __reduce_min_sync(0xffffffffu, lo);
    hi = __reduce_max_sync(0xffffffffu, hi);
#else
    for (int o = 16; o > 0; o >>= 1) {
        lo = min(lo, __shfl_xor_sync(0xffffffffu, lo, o));
        hi = max(hi, __shfl_xor_sync(0xffffffffu, hi, o));
    }
#endif
}

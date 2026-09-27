/* ── fasttomo/csrc/cuda/projection_parallel.cu ──────────────────────────────
 * Single-axis ("parallel": one tilt axis, Y) forward and back projection.
 * Kernels: forward_project_parallel_kernel<C, T>, back_project_parallel_kernel<C, T>
 *          (C Y-slices per texel: 1, or packed 2 for fp32 / 4 for fp16).
 * Host wrappers: forward_project_parallel_cuda, forward_project_parallel_cuda_cached,
 *                back_project_parallel_cuda, back_project_parallel_cuda_cached.
 */

#include "common.cuh"
#include "texture_cache.h"
#include "../include/proj_geom.h"
#include "ops.h"

#include <climits>

/* ── ray clipping (FP) ────────────────────────────────────────────────────────
 * A ray (t, angle) samples fx(s) = (t*ca - s*sa) + half_x and
 * fz(s) = (t*sa + s*ca) + half_z at the integer steps s in [-ray_len,
 * ray_len]; only ~65 % of them pass the tight bounds test. clip_steps narrows
 * [lo, hi] to the steps at which the coordinate c(s) = a*s + b can lie in
 * [lo_v, hi_v] (the fp32 form of the CPU kernels' ft_clip_steps,
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
 *   test would reject: the samples taken, and the result bits, do not change.
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

/* The step range [lo, hi] of one ray, clipped to the volume. */
__device__ __forceinline__ void fp_step_range(float t, float ca, float sa, int Nx, int Nz,
                                              float half_x, float half_z, int ray_len,
                                              int &lo, int &hi) {
    lo = -ray_len;
    hi = ray_len;
    const float mag = fabsf(t) + (float)ray_len + fmaxf(half_x, half_z) + 1.0f;
    clip_steps(-sa, t * ca + half_x, 0.0f, (float)(Nx - 1), mag, lo, hi);
    clip_steps(ca, t * sa + half_z, 0.0f, (float)(Nz - 1), mag, lo, hi);
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

/* FP launch shape: 32 detectors x 8 angles; a warp is one angle row. */
#define FP_BLOCK_X 32
#define FP_BLOCK_Y 8

/* A C-channel texel fetch (C = 1, 2, 4) and its per-channel accumulation. */
template <int C> struct PackedFetch;
template <> struct PackedFetch<1> {
    typedef float type;
    __device__ static __forceinline__ void add(float (&acc)[1], float v) { acc[0] += v; }
};
template <> struct PackedFetch<2> {
    typedef float2 type;
    __device__ static __forceinline__ void add(float (&acc)[2], float2 v) {
        acc[0] += v.x;
        acc[1] += v.y;
    }
};
template <> struct PackedFetch<4> {
    typedef float4 type;
    __device__ static __forceinline__ void add(float (&acc)[4], float4 v) {
        acc[0] += v.x;
        acc[1] += v.y;
        acc[2] += v.z;
        acc[3] += v.w;
    }
};

/* ── forward projection ───────────────────────────────────────────────────────
 * Grid:  (ceil(N_det/32), ceil(n_angles/8), ceil(Ny/C))   Block: (32, 8, 1)
 * Each thread owns one (detector, angle) output cell of one Y-slice
 * (1-channel texture) or of C Y-slices (packed: C = 2 fp32 / 4 fp16,
 * texture_cache.h; one tex2DLayered<float2|float4> fetch returns the C
 * slices' samples, summed in the same order as the 1-channel kernel, so each
 * slice gets the same bits; only slices < Ny are stored). Marches along the
 * ray over the warp's clipped step range; tex2DLayered handles bilinear
 * interp and border. Every lane reaches the warp reduction (no early return
 * before it).
 */
template <int C, typename T>
__global__ static void forward_project_parallel_kernel(
    cudaTextureObject_t  vol_tex,    /* ceil(Ny/C) layers of [Nz, Nx], C-channel */
    T *__restrict__      sino,   /* [n_angles, Ny, N_det] */
    const float *__restrict__ cos_a,
    const float *__restrict__ sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    float half_det, float half_x, float half_z, int ray_len
) {
    typedef typename PackedFetch<C>::type texel_t;
    int id      = blockIdx.x * blockDim.x + threadIdx.x;  /* detector pixel */
    int ia      = blockIdx.y * blockDim.y + threadIdx.y;  /* angle index    */
    int iy_pack = blockIdx.z;                              /* layer         */
    bool active = id < N_det && ia < n_angles;

    float ca = 0.0f, sa = 0.0f;
    float t  = id - half_det;
    float sums[C];
    #pragma unroll
    for (int b = 0; b < C; b++) sums[b] = 0.0f;
    int lo = INT_MAX, hi = INT_MIN;
    if (active) {
        ca = cos_a[ia];
        sa = sin_a[ia];
        fp_step_range(t, ca, sa, Nx, Nz, half_x, half_z, ray_len, lo, hi);
    }
    warp_uniform_range(lo, hi);
    if (!active) return;

    for (int step = lo; step <= hi; step++) {
        float s  = (float)step;
        float fx = (t * ca - s * sa) + half_x;
        float fz = (t * sa + s * ca) + half_z;
        /* Match forward_project_cpu's tight bilinear-stencil bounds: only
         * sample when both pixel neighbours along each axis are inside the
         * volume (fx in [0, Nx-1), fz in [0, Nz-1)). Border-mode tex2DLayered
         * would otherwise contribute a half-texel slab at each edge that the
         * CPU kernel skips. Written as a negated conjunction so a NaN
         * coordinate (NaN/Inf tilt angle) fails it and the sample is skipped
         * (the NaN view projects to 0, like the CPU kernel) instead of
         * reaching the texture unit, which returns finite garbage for it. */
        if (!(fx >= 0.0f && fx < (float)(Nx - 1) &&
              fz >= 0.0f && fz < (float)(Nz - 1)))
            continue;
        /* tex2DLayered: x→Nx axis, y→Nz axis, layer→iy_pack.
         * +0.5 maps integer pixel index to texel centre.
         * Always read as float — texture hardware promotes 16-bit float
         * texels (fp16 channel desc) to 32-bit float during the filtered
         * read. The kernel sees float regardless of texel storage. */
        PackedFetch<C>::add(sums, tex2DLayered<texel_t>(vol_tex, fx + 0.5f, fz + 0.5f, iy_pack));
    }

    int iy_base = C * iy_pack;
    #pragma unroll
    for (int b = 0; b < C; b++) {
        if (iy_base + b < Ny)
            sino[((size_t)ia * Ny + (iy_base + b)) * N_det + id] = float_to_t<T>(sums[b]);
    }
}

/* ── back-projection ──────────────────────────────────────────────────────────
 * Voxel-driven: each thread owns one (ix, iz) voxel of one Y-slice (1-channel
 * texture) or of C Y-slices (packed: C = 2 fp32 / 4 fp16, tex2DLayered<
 * float2|float4>, only slices < Ny stored) and sums the linearly
 * interpolated sinogram over the angles.
 * Grid:  (ceil(Nx/16), ceil(Nz/16), ceil(Ny/C))   Block: (16, 16, 1)
 * cos/sin preloaded into shared memory.
 *
 * NaN/Inf tilt angles: the texture unit returns finite garbage for a NaN
 * coordinate. The trig load reports (block-uniformly) whether the table
 * holds a non-finite value; only then does the loop test each view and skip
 * the bad ones (they add nothing, as in the CPU kernels). The common path is
 * the plain loop: a per-view test in it cost 6-25 % on small, latency-bound
 * grids (fewer fetches in flight).
 */
/* load_trig_smem (common.cuh), and the block-wide OR of "a cos/sin value is
 * not finite"; __syncthreads_or is the barrier. */
__device__ static bool load_trig_smem_check(
    float *__restrict__ sh_cos, float *__restrict__ sh_sin,
    const float *__restrict__ cos_vals, const float *__restrict__ sin_vals, int n
) {
    int tid   = threadIdx.x + threadIdx.y * blockDim.x;
    int block = blockDim.x  * blockDim.y;
    int bad   = 0;
    for (int i = tid; i < n; i += block) {
        float c = cos_vals[i], s = sin_vals[i];
        sh_cos[i] = c;
        sh_sin[i] = s;
        bad |= !(isfinite(c) && isfinite(s));
    }
    return __syncthreads_or(bad) != 0;
}

/* Sum over the views of the C-channel sample at (x, z) of texture row
 * iy_tc; CHECK skips views with non-finite trig. */
template <int C, bool CHECK>
__device__ __forceinline__ void bp_sum_views(
    float (&vals)[C], cudaTextureObject_t sino_tex,
    const float *sh_cos, const float *sh_sin, int n_angles,
    float x, float z, float half_det, float iy_tc
) {
    typedef typename PackedFetch<C>::type texel_t;
    for (int ia = 0; ia < n_angles; ia++) {
        float c = sh_cos[ia], s = sh_sin[ia];
        if (CHECK && !(isfinite(c) && isfinite(s)))
            continue;
        float ft = x * c + z * s + half_det;
        /* tex2DLayered: x→N_det axis, y→Ny axis, layer→ia.
         * Hardware linear filter interpolates along detector (ft fractional).
         * iy_tc is always at a texel centre so no interpolation in Y.
         * Always read as float — hardware promotes fp16 texels on read. */
        PackedFetch<C>::add(vals, tex2DLayered<texel_t>(sino_tex, ft + 0.5f, iy_tc, ia));
    }
}

template <int C, typename T>
__global__ static void back_project_parallel_kernel(
    cudaTextureObject_t  sino_tex,   /* n_angles layers of [ceil(Ny/C) × N_det] */
    T *__restrict__      vol,     /* [Ny, Nz, Nx] */
    const float *__restrict__ cos_a,
    const float *__restrict__ sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    float half_det, float half_x, float half_z
) {
    /* cos/sin tables stay fp32 in shared memory — small, precision-critical. */
    extern __shared__ float sh[];
    float *sh_cos = sh;
    float *sh_sin = sh + n_angles;

    bool bad_trig = load_trig_smem_check(sh_cos, sh_sin, cos_a, sin_a, n_angles);

    int ix      = blockIdx.x * blockDim.x + threadIdx.x;
    int iz      = blockIdx.y * blockDim.y + threadIdx.y;
    int iy_pack = blockIdx.z;
    if (ix >= Nx || iz >= Nz) return;

    float x = ix - half_x;
    float z = iz - half_z;
    float vals[C];
    #pragma unroll
    for (int b = 0; b < C; b++) vals[b] = 0.0f;
    float iy_tc = (float)iy_pack + 0.5f;   /* exact texel centre, no Y-interp */

    if (bad_trig)
        bp_sum_views<C, true>(vals, sino_tex, sh_cos, sh_sin, n_angles, x, z, half_det, iy_tc);
    else
        bp_sum_views<C, false>(vals, sino_tex, sh_cos, sh_sin, n_angles, x, z, half_det, iy_tc);

    int iy_base = C * iy_pack;
    #pragma unroll
    for (int b = 0; b < C; b++) {
        if (iy_base + b < Ny)
            vol[((size_t)(iy_base + b) * Nz + iz) * Nx + ix] = float_to_t<T>(vals[b]);
    }
}

/* ────────────────────────────────────────────────────────────────────────────
 * Host side. One launch routine per kernel reads the texture from a
 * TextureCache that already holds the data; the *_cached entry points
 * validate the caller's cache and upload into it, the non-cached entry
 * points build a ScopedTextureCache (texture_cache.h) around the same code.
 * ──────────────────────────────────────────────────────────────────────────── */

static tc_dtype_t tc_dtype_of(int io_dtype) {
    return io_dtype == 0 ? TC_DTYPE_FP32 : TC_DTYPE_FP16;
}

/* Y-slice packing of the parallel textures (texture_cache.h): 2 slices per
 * float2 texel for fp32 when Ny >= 2; 4 slices per fp16 texel when
 * Ny % 4 == 0. A single slice stays 1-channel: a packed fetch costs more
 * than a 1-channel one (the filter rate per texel drops), so packing only
 * pays when it saves fetches. Used by the one-shot wrappers and, through the
 * binding, by the cached operator (so both use the same layout). */
int parallel_channels(int io_dtype, int Ny) {
    check_io_dtype(io_dtype, "parallel_channels");
    if (io_dtype == 0) return Ny >= 2 ? 2 : 1;
    return Ny % 4 == 0 ? 4 : 1;
}

/* Upload `vol` into `vol_cache` (already validated) and launch the FP
 * kernel matching the cache's dtype and channel count. */
static void forward_project_parallel_launch(
    int io_dtype, TextureCache *vol_cache,
    const void *vol, void *sino,
    const float *cos_a, const float *sin_a,
    int Ny, int Nz, int Nx, int n_angles, int N_det,
    cudaStream_t stream, bool upload
) {
    ProjGeomParallel g = proj_geom_parallel(Ny, Nz, Nx, n_angles, N_det);
    if (upload) tc_upload(vol_cache, vol, stream);

    dim3 block(FP_BLOCK_X, FP_BLOCK_Y, 1);
    unsigned gx = (unsigned)((g.N_det + FP_BLOCK_X - 1) / FP_BLOCK_X);
    unsigned gy = (unsigned)((g.n_angles + FP_BLOCK_Y - 1) / FP_BLOCK_Y);
    int C = vol_cache->channels;
    dim3 grid(gx, gy, tc_packed_extent(g.Ny, C));
#define FP_LAUNCH(CH, T)                                                             \
    forward_project_parallel_kernel<CH, T><<<grid, block, 0, stream>>>(              \
        vol_cache->tex, (T *)sino, cos_a, sin_a,                                     \
        g.Ny, g.Nz, g.Nx, g.n_angles, g.N_det,                                       \
        g.half_det, g.half_x, g.half_z, g.ray_len)
    if (C == 1 && io_dtype == 0)      FP_LAUNCH(1, float);
    else if (C == 1 && io_dtype == 1) FP_LAUNCH(1, __half);
    else if (C == 2 && io_dtype == 0) FP_LAUNCH(2, float);
    else if (C == 4 && io_dtype == 1) FP_LAUNCH(4, __half);
    else throw std::invalid_argument("forward_project_parallel: unsupported packed cache");
#undef FP_LAUNCH
    CUDA_CHECK_KERNEL();
}

/* Upload `sino` into `sino_cache` (already validated) and launch the BP
 * kernel matching the cache's dtype and channel count. */
static void back_project_parallel_launch(
    int io_dtype, TextureCache *sino_cache,
    const void *sino, void *vol,
    const float *cos_a, const float *sin_a,
    int Ny, int Nz, int Nx, int n_angles, int N_det,
    cudaStream_t stream, bool upload
) {
    ProjGeomParallel g = proj_geom_parallel(Ny, Nz, Nx, n_angles, N_det);
    if (upload) tc_upload(sino_cache, sino, stream);

    dim3 block(16, 16, 1);
    size_t smem = 2 * (size_t)n_angles * sizeof(float);
    int C = sino_cache->channels;
    dim3 grid((g.Nx + 15) / 16, (g.Nz + 15) / 16, tc_packed_extent(g.Ny, C));
#define BP_LAUNCH(CH, T) do {                                                        \
    enable_extended_smem_if_needed((const void *)back_project_parallel_kernel<CH, T>, smem); \
    back_project_parallel_kernel<CH, T><<<grid, block, smem, stream>>>(               \
        sino_cache->tex, (T *)vol, cos_a, sin_a,                                     \
        g.Ny, g.Nz, g.Nx, g.n_angles, g.N_det, g.half_det, g.half_x, g.half_z);     \
} while (0)
    if (C == 1 && io_dtype == 0)      BP_LAUNCH(1, float);
    else if (C == 1 && io_dtype == 1) BP_LAUNCH(1, __half);
    else if (C == 2 && io_dtype == 0) BP_LAUNCH(2, float);
    else if (C == 4 && io_dtype == 1) BP_LAUNCH(4, __half);
    else throw std::invalid_argument("back_project_parallel: unsupported packed cache");
#undef BP_LAUNCH
    CUDA_CHECK_KERNEL();
}

void forward_project_parallel_cuda(
    int io_dtype,
    const void  *vol,
    void        *sino,
    const float *cos_a,
    const float *sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
) {
    check_io_dtype(io_dtype, "forward_project_parallel_cuda");
    ScopedTextureCache tc(TC_VOL_LAYERED, tc_dtype_of(io_dtype),
                          parallel_channels(io_dtype, Ny), Ny, Nz, Nx, stream);
    forward_project_parallel_launch(io_dtype, tc.get(), vol, sino,
                           cos_a, sin_a, Ny, Nz, Nx, n_angles, N_det, stream, true);
    tc.finish();
}

void forward_project_parallel_cuda_cached(
    int io_dtype,
    TextureCache *vol_cache,
    const void   *vol,
    void         *sino,
    const float  *cos_a,
    const float  *sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream,
    int upload
) {
    check_io_dtype(io_dtype, "forward_project_parallel_cuda_cached");
    tc_validate(vol_cache, TC_VOL_LAYERED, io_dtype, Ny, Nz, Nx,
                "forward_project_parallel_cuda_cached");
    forward_project_parallel_launch(io_dtype, vol_cache, vol, sino,
                           cos_a, sin_a, Ny, Nz, Nx, n_angles, N_det, stream, upload != 0);
}

void back_project_parallel_cuda(
    int io_dtype,
    const void  *sino,
    void        *vol,
    const float *cos_a,
    const float *sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
) {
    check_io_dtype(io_dtype, "back_project_parallel_cuda");
    ScopedTextureCache tc(TC_SINO_LAYERED, tc_dtype_of(io_dtype),
                          parallel_channels(io_dtype, Ny), n_angles, Ny, N_det, stream);
    back_project_parallel_launch(io_dtype, tc.get(), sino, vol,
                        cos_a, sin_a, Ny, Nz, Nx, n_angles, N_det, stream, true);
    tc.finish();
}

void back_project_parallel_cuda_cached(
    int io_dtype,
    TextureCache *sino_cache,
    const void   *sino,
    void         *vol,
    const float  *cos_a,
    const float  *sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream,
    int upload
) {
    check_io_dtype(io_dtype, "back_project_parallel_cuda_cached");
    tc_validate(sino_cache, TC_SINO_LAYERED, io_dtype, n_angles, Ny, N_det,
                "back_project_parallel_cuda_cached");
    back_project_parallel_launch(io_dtype, sino_cache, sino, vol,
                        cos_a, sin_a, Ny, Nz, Nx, n_angles, N_det, stream, upload != 0);
}

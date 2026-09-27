/* ── fasttomo/csrc/cuda/projection_parallel.cu ──────────────────────────────
 * Single-axis ("parallel": one tilt axis, Y) forward and back projection.
 * Kernels: forward_project_parallel_kernel, forward_project_parallel_packed_h4_kernel,
 *          back_project_parallel_kernel, back_project_parallel_packed_h4_kernel.
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

/* ── forward projection ───────────────────────────────────────────────────────
 * Grid:  (ceil(N_det/32), ceil(n_angles/8), Ny)   Block: (32, 8, 1)
 * Each thread owns one (detector, angle) output cell in one Y-slice.
 * Marches along the ray over the warp's clipped step range; tex2DLayered
 * handles bilinear interp and border. Every lane reaches the warp
 * reduction (no early return before it).
 */
template <typename T>
__global__ static void forward_project_parallel_kernel(
    cudaTextureObject_t  vol_tex,    /* Ny layers of [Nz, Nx] */
    T *__restrict__      sino,   /* [n_angles, Ny, N_det] */
    const float *__restrict__ cos_a,
    const float *__restrict__ sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    float half_det, float half_x, float half_z, int ray_len
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;  /* detector pixel */
    int ia = blockIdx.y * blockDim.y + threadIdx.y;  /* angle index    */
    int iy = blockIdx.z;                              /* Y-slice / layer */
    bool active = id < N_det && ia < n_angles;

    float ca = 0.0f, sa = 0.0f;
    float t   = id - half_det;
    float sum = 0.0f;
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
        /* tex2DLayered: x→Nx axis, y→Nz axis, layer→iy.
         * +0.5 maps integer pixel index to texel centre.
         * Always read as float — texture hardware promotes 16-bit float
         * texels (fp16 channel desc) to 32-bit float during the filtered
         * read. The kernel sees float regardless of texel storage. */
        sum += tex2DLayered<float>(vol_tex, fx + 0.5f, fz + 0.5f, iy);
    }

    sino[((size_t)ia * Ny + iy) * N_det + id] = float_to_t<T>(sum);
}

/* ── 4-channel packed forward-projection kernel ───────────────────────────────
 * Reads tex2DLayered<float4> (4 packed Y-slices per fetch), accumulates 4 line
 * integrals in parallel, writes 4 strided outputs. Grid Z is Ny/4. */
__global__ static void forward_project_parallel_packed_h4_kernel(
    cudaTextureObject_t  vol_tex,    /* (Ny/4) layers of [Nz × Nx], 4-ch half */
    __half *__restrict__ sino,   /* [n_angles, Ny, N_det] */
    const float *__restrict__ cos_a,
    const float *__restrict__ sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    float half_det, float half_x, float half_z, int ray_len
) {
    int id      = blockIdx.x * blockDim.x + threadIdx.x;
    int ia      = blockIdx.y * blockDim.y + threadIdx.y;
    int iy_pack = blockIdx.z;
    bool active = id < N_det && ia < n_angles;

    float ca = 0.0f, sa = 0.0f;
    float t  = id - half_det;
    float sums[4] = {0.f, 0.f, 0.f, 0.f};
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
        if (!(fx >= 0.0f && fx < (float)(Nx - 1) &&
              fz >= 0.0f && fz < (float)(Nz - 1)))      /* NaN-safe, see above */
            continue;
        float4 v = tex2DLayered<float4>(vol_tex, fx + 0.5f, fz + 0.5f, iy_pack);
        sums[0] += v.x;
        sums[1] += v.y;
        sums[2] += v.z;
        sums[3] += v.w;
    }

    int iy_base = 4 * iy_pack;
    #pragma unroll
    for (int b = 0; b < 4; b++) {
        sino[((size_t)ia * Ny + (iy_base + b)) * N_det + id]
            = __float2half_rn(sums[b]);
    }
}

/* ── back-projection ──────────────────────────────────────────────────────────
 * Grid:  (ceil(Nx/16), ceil(Nz/16), Ny)   Block: (16, 16, 1)
 * Each thread owns one (ix, iz) voxel in one Y-slice and loops over angles.
 * cos/sin preloaded into shared memory via load_trig_smem.
 */
template <typename T>
__global__ static void back_project_parallel_kernel(
    cudaTextureObject_t  sino_tex,   /* n_angles layers of [Ny, N_det] */
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

    load_trig_smem(sh_cos, sh_sin, cos_a, sin_a, n_angles);

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int iy = blockIdx.z;

    if (ix >= Nx || iz >= Nz) return;

    float x   = ix - half_x;
    float z   = iz - half_z;
    float val = 0.0f;

    float iy_tc = (float)iy + 0.5f;  /* Y-slice texel centre — no interpolation in Y */

    for (int ia = 0; ia < n_angles; ia++) {
        float ft = x * sh_cos[ia] + z * sh_sin[ia] + half_det;
        /* Outside (-1, N_det) both texels of the linear filter are border
         * zeros, so skipping those samples changes no bit (val never is -0:
         * it starts at +0). The negated form also skips a NaN ft (NaN/Inf
         * tilt angle): the view contributes nothing, like in the CPU
         * kernels, instead of the finite garbage the texture unit returns
         * for a NaN coordinate. */
        if (!(ft > -1.0f && ft < (float)N_det))
            continue;
        /* tex2DLayered: x→N_det axis, y→Ny axis, layer→ia.
         * Hardware linear filter interpolates along detector (ft fractional).
         * iy_tc is always at a texel centre so no interpolation in Y.
         * Always read as float — hardware promotes fp16 texels on read. */
        val += tex2DLayered<float>(sino_tex, ft + 0.5f, iy_tc, ia);
    }

    vol[((size_t)iy * Nz + iz) * Nx + ix] = float_to_t<T>(val);
}

/* ── 4-channel packed back-projection kernel ──────────────────────────────────
 * Each thread owns one (ix, iz, iy_pack) voxel-group of 4 voxels (4 Y-slices).
 * Reads tex2DLayered<float4>; writes 4 strided outputs. */
__global__ static void back_project_parallel_packed_h4_kernel(
    cudaTextureObject_t  sino_tex,   /* n_angles layers of [(Ny/4) × N_det], 4-ch half */
    __half *__restrict__ vol,     /* [Ny, Nz, Nx] */
    const float *__restrict__ cos_a,
    const float *__restrict__ sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    float half_det, float half_x, float half_z
) {
    extern __shared__ float sh[];
    float *sh_cos = sh;
    float *sh_sin = sh + n_angles;

    load_trig_smem(sh_cos, sh_sin, cos_a, sin_a, n_angles);

    int ix      = blockIdx.x * blockDim.x + threadIdx.x;
    int iz      = blockIdx.y * blockDim.y + threadIdx.y;
    int iy_pack = blockIdx.z;
    if (ix >= Nx || iz >= Nz) return;

    float x = ix - half_x;
    float z = iz - half_z;
    float vals[4] = {0.f, 0.f, 0.f, 0.f};
    float iy_tc = (float)iy_pack + 0.5f;   /* exact texel centre, no Y-interp */

    for (int ia = 0; ia < n_angles; ia++) {
        float ft = x * sh_cos[ia] + z * sh_sin[ia] + half_det;
        if (!(ft > -1.0f && ft < (float)N_det))    /* NaN-safe, see above */
            continue;
        float4 sample = tex2DLayered<float4>(sino_tex, ft + 0.5f, iy_tc, ia);
        vals[0] += sample.x;
        vals[1] += sample.y;
        vals[2] += sample.z;
        vals[3] += sample.w;
    }

    int iy_base = 4 * iy_pack;
    #pragma unroll
    for (int b = 0; b < 4; b++) {
        vol[((size_t)(iy_base + b) * Nz + iz) * Nx + ix]
            = __float2half_rn(vals[b]);
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

/* 4-channel packing (4 Y-slices per fp16 texel) whenever the dtype allows it. */
static int one_shot_channels(int io_dtype, int Ny) {
    return (io_dtype == 1 && Ny % 4 == 0) ? 4 : 1;
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
    if (io_dtype == 1 && vol_cache->channels == 4) {
        dim3 grid(gx, gy, g.Ny / 4);
        forward_project_parallel_packed_h4_kernel<<<grid, block, 0, stream>>>(
            vol_cache->tex, (__half *)sino, cos_a, sin_a,
            g.Ny, g.Nz, g.Nx, g.n_angles, g.N_det,
            g.half_det, g.half_x, g.half_z, g.ray_len);
    } else {
        dim3 grid(gx, gy, g.Ny);
        if (io_dtype == 0)
            forward_project_parallel_kernel<float><<<grid, block, 0, stream>>>(
                vol_cache->tex, (float *)sino, cos_a, sin_a,
                g.Ny, g.Nz, g.Nx, g.n_angles, g.N_det,
                g.half_det, g.half_x, g.half_z, g.ray_len);
        else
            forward_project_parallel_kernel<__half><<<grid, block, 0, stream>>>(
                vol_cache->tex, (__half *)sino, cos_a, sin_a,
                g.Ny, g.Nz, g.Nx, g.n_angles, g.N_det,
                g.half_det, g.half_x, g.half_z, g.ray_len);
    }
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
    if (io_dtype == 1 && sino_cache->channels == 4) {
        dim3 grid((g.Nx + 15) / 16, (g.Nz + 15) / 16, g.Ny / 4);
        enable_extended_smem_if_needed((const void *)back_project_parallel_packed_h4_kernel, smem);
        back_project_parallel_packed_h4_kernel<<<grid, block, smem, stream>>>(
            sino_cache->tex, (__half *)vol, cos_a, sin_a,
            g.Ny, g.Nz, g.Nx, g.n_angles, g.N_det,
            g.half_det, g.half_x, g.half_z);
    } else {
        dim3 grid((g.Nx + 15) / 16, (g.Nz + 15) / 16, g.Ny);
        if (io_dtype == 0) {
            enable_extended_smem_if_needed((const void *)back_project_parallel_kernel<float>, smem);
            back_project_parallel_kernel<float><<<grid, block, smem, stream>>>(
                sino_cache->tex, (float *)vol, cos_a, sin_a,
                g.Ny, g.Nz, g.Nx, g.n_angles, g.N_det,
                g.half_det, g.half_x, g.half_z);
        } else {
            enable_extended_smem_if_needed((const void *)back_project_parallel_kernel<__half>, smem);
            back_project_parallel_kernel<__half><<<grid, block, smem, stream>>>(
                sino_cache->tex, (__half *)vol, cos_a, sin_a,
                g.Ny, g.Nz, g.Nx, g.n_angles, g.N_det,
                g.half_det, g.half_x, g.half_z);
        }
    }
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
                          one_shot_channels(io_dtype, Ny), Ny, Nz, Nx, stream);
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
                          one_shot_channels(io_dtype, Ny), n_angles, Ny, N_det, stream);
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

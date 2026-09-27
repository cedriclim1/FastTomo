/* ── fasttomo/csrc/cuda/projection_rotmat.cu ────────────────────────────────
 * Arbitrary-rotation ("rotmat": one 3x3 matrix per view) forward and back
 * projection, plus the gradient w.r.t. the matrices (grad_rotmat).
 * Kernels: forward_project_rotmat_kernel, back_project_rotmat_kernel,
 *          grad_rotmat_kernel.
 * Host wrappers: forward_project_rotmat_cuda, forward_project_rotmat_cuda_cached,
 *                back_project_rotmat_cuda, back_project_rotmat_cuda_cached,
 *                grad_rotmat_cuda, grad_rotmat_cuda_cached.
 */

#include "common.cuh"
#include "texture_cache.h"
#include "../include/proj_geom.h"
#include "ops.h"
#include "scratch_pool.cuh"

#include <climits>

/* ── ray clipping (slab method) ───────────────────────────────────────────────
 * forward_project_rotmat_kernel samples p(s) = o + s*d for the integer steps
 * s in [-ray_len, ray_len], p in voxel-index units (o holds the half_*
 * centring). ray_len covers the volume diagonal, so on rays that cross the
 * volume at an angle, or miss it, 40-47 % of the steps sample outside it.
 * A sample contributes only while its trilinear stencil touches the volume,
 * i.e. p in (-1, N) on every axis; outside that the border-mode linear
 * texture returns exactly 0.
 *
 * rotmat_ray_steps() narrows [-ray_len, ray_len] to an integer range
 * [lo, hi] that contains every contributing step, and the kernel loops over
 * it with the SAME step values s and the SAME per-sample arithmetic as the
 * full loop. A dropped step only ever added exactly 0.0f (sum + 0.0f == sum
 * for every value the sum can hold: it starts at +0.0f and x + (-x) rounds
 * to +0.0f, so it is never -0.0f), so the result is bit-identical to the
 * unclipped loop. The slab is widened to [-2, N+1] (one voxel beyond the
 * contributing band) and the step range by one more step on each side, so
 * fp32 rounding in the entry/exit parameters (~1e-4 of a step at N = 4096)
 * cannot drop a contributing sample. Non-finite ray parameters (a NaN or
 * Inf matrix) fall back to the full range, which is what the loop did.
 *
 * grad_rotmat_kernel marches the same steps but is NOT clipped: its loose
 * bounds test already skips the 8 texture fetches of an outside sample, so
 * clipping only saves that test, and measured 0.97-0.98x (block- or
 * warp-uniform range) and ~0.8x (per-lane range) against the full loop.
 */
__device__ __forceinline__ bool rotmat_clip_axis(float o, float d, int N,
                                                 float &t0, float &t1) {
    const float lo = -2.0f, hi = (float)N + 1.0f;
    if (d != 0.0f) {
        float a = (lo - o) / d, b = (hi - o) / d;
        t0 = fmaxf(t0, fminf(a, b));
        t1 = fminf(t1, fmaxf(a, b));
        return true;
    }
    return o > lo && o < hi;     /* parallel to the slab: inside or never */
}

/* Integer step range [*lo, *hi] of the ray o + s*d (o in voxel-index units,
 * per axis) that can touch a volume of Nx x Ny x Nz voxels; the empty range
 * (INT_MAX, INT_MIN) when the ray misses it. clip == 0 returns the full
 * [-ray_len, ray_len] (the test hook _debug_set_rotmat_ray_clip compares
 * both paths bit for bit). */
__device__ __forceinline__ void rotmat_ray_steps(
    float ox, float oy, float oz, float dx, float dy, float dz,
    int Nx, int Ny, int Nz, int ray_len, int clip, int *lo, int *hi
) {
    *lo = -ray_len;
    *hi =  ray_len;
    if (!clip) return;
    if (!(isfinite(ox) && isfinite(oy) && isfinite(oz) &&
          isfinite(dx) && isfinite(dy) && isfinite(dz)))
        return;
    float t0 = -(float)ray_len, t1 = (float)ray_len;
    bool hit = rotmat_clip_axis(ox, dx, Nx, t0, t1)
             & rotmat_clip_axis(oy, dy, Ny, t0, t1)
             & rotmat_clip_axis(oz, dz, Nz, t0, t1);
    if (!hit || !(t0 <= t1)) { *lo = INT_MAX; *hi = INT_MIN; return; }
    /* t0, t1 are finite here (clamped to +-ray_len above). */
    *lo = max(-ray_len, (int)floorf(t0) - 1);
    *hi = min( ray_len, (int)ceilf(t1)  + 1);
}

/* Union of the step ranges over the thread block, so every warp of the
 * block marches the same steps: neighbouring rays (and the 8 warps of the
 * 16x16 detector tile) then sample neighbouring texels at the same time and
 * the texture cache keeps its locality. A thread's extra steps lie outside
 * its own range, so they add exactly 0 as well. Per-warp ranges measured the
 * same except 256^3 fp32 ZYZ (1.39x vs 1.71x), per-lane ranges were slower
 * than no clipping on grad_rotmat. The empty range (INT_MAX, INT_MIN) is the
 * identity. Every thread of the block must call it (full warps: the block is
 * 16x16); two barriers, 8 bytes of static shared memory. */
__device__ __forceinline__ void rotmat_block_union(int *lo, int *hi) {
    __shared__ int s_lo, s_hi;
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    if (tid == 0) { s_lo = INT_MAX; s_hi = INT_MIN; }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        *lo = min(*lo, __shfl_xor_sync(0xffffffffu, *lo, off));
        *hi = max(*hi, __shfl_xor_sync(0xffffffffu, *hi, off));
    }
    __syncthreads();
    if ((tid & 31) == 0) { atomicMin(&s_lo, *lo); atomicMax(&s_hi, *hi); }
    __syncthreads();
    *lo = s_lo; *hi = s_hi;
}

/* Host switch for the clipping (on by default). Test hook only. */
static int g_rotmat_ray_clip = 1;

int _debug_set_rotmat_ray_clip(int on) {
    int prev = g_rotmat_ray_clip;
    g_rotmat_ray_clip = on ? 1 : 0;
    return prev;
}

/* ── 3D rotation-matrix forward projection ────────────────────────────────────
 * forward_project_rotmat_kernel
 * Grid: (⌈N_det/16⌉, ⌈Ny/16⌉, n_angles)   Block: (16, 16, 1)
 * Each thread owns one (detector-u, detector-v, angle) output sample.
 * Rotation matrix M (row-major, 9 floats per projection):
 *   detector-u = M[0]*x + M[1]*y + M[2]*z
 *   detector-v = M[3]*x + M[4]*y + M[5]*z
 * Ray tracing (R^T applied to ray direction):
 *   x(s) = u*M[0] + v*M[3] + s*M[6]
 *   y(s) = u*M[1] + v*M[4] + s*M[7]
 *   z(s) = u*M[2] + v*M[5] + s*M[8]
 * tex3D axis mapping: tex3D(tex, ix, iz, iy) → vol[iy][iz][ix]
 */
template <typename T>
__global__ static void forward_project_rotmat_kernel(
    cudaTextureObject_t  vol_tex,
    T *__restrict__      sino,         /* [n_angles, Ny, N_det] */
    const float *__restrict__ mats,    /* [n_angles * 9] row-major */
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    float half_det, float half_x, float half_z, float half_y,
    int ray_len, int clip
) {
    int iu = blockIdx.x * blockDim.x + threadIdx.x;  /* detector u */
    int iv = blockIdx.y * blockDim.y + threadIdx.y;  /* detector v = Y-slice */
    int ia = blockIdx.z;

    const bool active = (iu < N_det && iv < Ny);

    const float *M = mats + ia * 9;
    float u = iu - half_det;
    float v = iv - half_y;

    /* precompute origin of ray at s=0 and step increments */
    float ox = u * M[0] + v * M[3];
    float oy = u * M[1] + v * M[4];
    float oz = u * M[2] + v * M[5];
    float dx = M[6], dy = M[7], dz = M[8];  /* ray direction */

    /* Steps whose trilinear stencil can touch the volume, united over the
     * block (rotmat_ray_steps, rotmat_block_union); the others read border
     * zeros. Every thread reaches the barriers: the early return comes after. */
    int step_lo = INT_MAX, step_hi = INT_MIN;
    if (active)
        rotmat_ray_steps(ox + half_x, oy + half_y, oz + half_z, dx, dy, dz,
                         Nx, Ny, Nz, ray_len, clip, &step_lo, &step_hi);
    rotmat_block_union(&step_lo, &step_hi);
    if (!active) return;

    float sum = 0.0f;
    for (int step = step_lo; step <= step_hi; step++) {
        float s  = (float)step;
        float tx = ox + s * dx + half_x + 0.5f;
        float tz = oz + s * dz + half_z + 0.5f;
        float ty = oy + s * dy + half_y + 0.5f;
        /* tex3D: (x→width=Nx, z→height=Nz, y→depth=Ny). Always read as
         * float — hardware promotes fp16 texels on filtered read. */
        sum += tex3D<float>(vol_tex, tx, tz, ty);
    }
    sino[((size_t)ia * Ny + iv) * N_det + iu] = float_to_t<T>(sum);
}

static tc_dtype_t tc_dtype_of(int io_dtype) {
    return io_dtype == 0 ? TC_DTYPE_FP32 : TC_DTYPE_FP16;
}

/* Upload `vol` into `vol_cache` (already validated) and launch the FP
 * kernel. Shared by the cached and the one-shot entry point. */
static void forward_project_rotmat_launch(
    int io_dtype, TextureCache *vol_cache,
    const void *vol, void *sino, const float *mats,
    int Ny, int Nz, int Nx, int n_angles, int N_det,
    cudaStream_t stream, bool upload
) {
    ProjGeomRotmat g = proj_geom_rotmat(Ny, Nz, Nx, n_angles, N_det);
    if (upload) tc_upload(vol_cache, vol, stream);

    dim3 block(16, 16, 1);
    dim3 grid((g.N_det + 15) / 16,
              (g.Ny    + 15) / 16,
              g.n_angles);
    if (io_dtype == 0)
        forward_project_rotmat_kernel<float><<<grid, block, 0, stream>>>(
            vol_cache->tex, (float *)sino, mats,
            g.Ny, g.Nz, g.Nx, g.n_angles, g.N_det,
            g.half_det, g.half_x, g.half_z, g.half_y, g.ray_len, g_rotmat_ray_clip);
    else
        forward_project_rotmat_kernel<__half><<<grid, block, 0, stream>>>(
            vol_cache->tex, (__half *)sino, mats,
            g.Ny, g.Nz, g.Nx, g.n_angles, g.N_det,
            g.half_det, g.half_x, g.half_z, g.half_y, g.ray_len, g_rotmat_ray_clip);
    CUDA_CHECK_KERNEL();
}

void forward_project_rotmat_cuda(
    int io_dtype,
    const void  *vol,
    void        *sino,
    const float *mats,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
) {
    check_io_dtype(io_dtype, "forward_project_rotmat_cuda");
    /* Rotmat samples continuously in Y, so no 4-channel Y packing. */
    ScopedTextureCache tc(TC_VOL_3D, tc_dtype_of(io_dtype), 1, Ny, Nz, Nx, stream);
    forward_project_rotmat_launch(io_dtype, tc.get(), vol, sino,
                                  mats, Ny, Nz, Nx, n_angles, N_det, stream, true);
    tc.finish();
}

void forward_project_rotmat_cuda_cached(
    int io_dtype,
    TextureCache *vol_cache,
    const void   *vol,
    void         *sino,
    const float  *mats,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream,
    int upload
) {
    check_io_dtype(io_dtype, "forward_project_rotmat_cuda_cached");
    tc_validate(vol_cache, TC_VOL_3D, io_dtype, Ny, Nz, Nx,
                "forward_project_rotmat_cuda_cached");
    forward_project_rotmat_launch(io_dtype, vol_cache, vol, sino,
                                  mats, Ny, Nz, Nx, n_angles, N_det, stream, upload != 0);
}

/* ── 3D rotation-matrix back-projection ──────────────────────────────────────
 * back_project_rotmat_kernel
 * Grid: (⌈Nx/8⌉, ⌈Nz/8⌉, ⌈Ny/4⌉)   Block: (8, 8, 4)
 * Each thread owns one (ix, iz, iy) voxel and loops over all angles.
 * Rotation matrices loaded into shared memory (9 * n_angles floats).
 * Sinogram sampled with tex2DLayered (n_angles layers of [Ny, N_det]).
 */
template <typename T>
__global__ static void back_project_rotmat_kernel(
    cudaTextureObject_t  sino_tex,        /* n_angles layers of [Ny, N_det] */
    T *__restrict__      vol,          /* [Ny, Nz, Nx] */
    const float *__restrict__ mats,   /* [n_angles * 9] row-major */
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    float half_det, float half_x, float half_z, float half_y
) {
    /* Rotation matrices stay fp32 in shared memory. */
    extern __shared__ float sh_mats[];   /* 9 * n_angles floats */

    int tid       = threadIdx.x + threadIdx.y * blockDim.x
                  + threadIdx.z * blockDim.x * blockDim.y;
    int block_sz  = blockDim.x * blockDim.y * blockDim.z;
    for (int k = tid; k < n_angles * 9; k += block_sz)
        sh_mats[k] = mats[k];
    __syncthreads();

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int iy = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iz >= Nz || iy >= Ny) return;

    float x = ix - half_x;
    float y = iy - half_y;
    float z = iz - half_z;
    float val = 0.0f;

    for (int ia = 0; ia < n_angles; ia++) {
        const float *M = sh_mats + ia * 9;
        /* u = R[0,:] · (x,y,z)   v = R[1,:] · (x,y,z) */
        float u_det = M[0]*x + M[1]*y + M[2]*z;
        float v_det = M[3]*x + M[4]*y + M[5]*z;
        /* tex2DLayered: x→N_det, y→Ny, layer→ia. Always read as float
         * — hardware promotes fp16 texels on filtered read. */
        val += tex2DLayered<float>(sino_tex,
                                   u_det + half_det + 0.5f,
                                   v_det + half_y   + 0.5f,
                                   ia);
    }

    vol[((size_t)iy * Nz + iz) * Nx + ix] = float_to_t<T>(val);
}

/* Upload `sino` into `sino_cache` (already validated) and launch the
 * BP kernel. Shared by the cached and the one-shot entry point. */
static void back_project_rotmat_launch(
    int io_dtype, TextureCache *sino_cache,
    const void *sino, void *vol, const float *mats,
    int Ny, int Nz, int Nx, int n_angles, int N_det,
    cudaStream_t stream, bool upload
) {
    ProjGeomRotmat g = proj_geom_rotmat(Ny, Nz, Nx, n_angles, N_det);
    if (upload) tc_upload(sino_cache, sino, stream);

    dim3 block(8, 8, 4);
    dim3 grid((g.Nx + 7) / 8,
              (g.Nz + 7) / 8,
              (g.Ny + 3) / 4);
    size_t smem = 9 * (size_t)n_angles * sizeof(float);
    if (io_dtype == 0) {
        enable_extended_smem_if_needed((const void *)back_project_rotmat_kernel<float>, smem);
        back_project_rotmat_kernel<float><<<grid, block, smem, stream>>>(
            sino_cache->tex, (float *)vol, mats,
            g.Ny, g.Nz, g.Nx, g.n_angles, g.N_det,
            g.half_det, g.half_x, g.half_z, g.half_y);
    } else {
        enable_extended_smem_if_needed((const void *)back_project_rotmat_kernel<__half>, smem);
        back_project_rotmat_kernel<__half><<<grid, block, smem, stream>>>(
            sino_cache->tex, (__half *)vol, mats,
            g.Ny, g.Nz, g.Nx, g.n_angles, g.N_det,
            g.half_det, g.half_x, g.half_z, g.half_y);
    }
    CUDA_CHECK_KERNEL();
}

void back_project_rotmat_cuda(
    int io_dtype,
    const void  *sino,
    void        *vol,
    const float *mats,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
) {
    check_io_dtype(io_dtype, "back_project_rotmat_cuda");
    ScopedTextureCache tc(TC_SINO_LAYERED, tc_dtype_of(io_dtype), 1,
                          n_angles, Ny, N_det, stream);
    back_project_rotmat_launch(io_dtype, tc.get(), sino, vol,
                               mats, Ny, Nz, Nx, n_angles, N_det, stream, true);
    tc.finish();
}

void back_project_rotmat_cuda_cached(
    int io_dtype,
    TextureCache *sino_cache,
    const void   *sino,
    void         *vol,
    const float  *mats,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream,
    int upload
) {
    check_io_dtype(io_dtype, "back_project_rotmat_cuda_cached");
    tc_validate(sino_cache, TC_SINO_LAYERED, io_dtype, n_angles, Ny, N_det,
                "back_project_rotmat_cuda_cached");
    back_project_rotmat_launch(io_dtype, sino_cache, sino, vol,
                               mats, Ny, Nz, Nx, n_angles, N_det, stream, upload != 0);
}

/* ── gradient of 3D rotmat forward projection w.r.t. rotation matrix ─────────
 * Grid:  (⌈N_det/16⌉, ⌈Ny/16⌉, n_angles)   Block: (16, 16, 1)
 * Same detector-driven layout as forward_project_rotmat_kernel. Each block
 * reduces its 256 rays' 9 matrix-entry sums to one partial per block
 * (warp shuffles, fixed order); grad_rotmat_finish_kernel then sums the
 * partials of each view in a fixed order and WRITES grad_mats. No atomics:
 * the gradient is bit-reproducible run to run.
 *
 * Forward sample at ray step s, detector cell (iu, iv):
 *   u = iu - half_det,  v = iv - half_y,  s = step
 *   tx = u*M[0] + v*M[3] + s*M[6] + half_x
 *   ty = u*M[1] + v*M[4] + s*M[7] + half_y
 *   tz = u*M[2] + v*M[5] + s*M[8] + half_z
 *   sino += vol(tx, ty, tz)   (trilinear)
 *
 * ∂tx/∂M[{0,3,6}] = {u, v, s}, ∂ty/∂M[{1,4,7}] = {u, v, s},
 * ∂tz/∂M[{2,5,8}] = {u, v, s}. Spatial derivatives ∂vol/∂{x,y,z}
 * computed analytically from trilinear interpolation weights.
 *
 * T is the volume dtype (float or __half). The 8 corner reads are done with
 * tex3D<float> and cudaFilterModePoint so the result is bit-exact equivalent
 * to a global-memory load (hardware promotes fp16 texels to fp32 on read).
 * The output grad_mats always accumulates in fp32 regardless of T.
 */
template <typename T>
__global__ static void grad_rotmat_kernel(
    cudaTextureObject_t  vol_tex,            /* TC_VOL_3D, point-filter view */
    const float *__restrict__ grad_sino,     /* [n_angles, Ny, N_det] */
    const float *__restrict__ mats,      /* [n_angles*9] row-major */
    float       *__restrict__ partials,  /* [n_angles, nb, 9], nb = gridDim.x*gridDim.y */
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    float half_det, float half_x, float half_y, float half_z,
    int ray_len
) {
    int iu = blockIdx.x * blockDim.x + threadIdx.x;
    int iv = blockIdx.y * blockDim.y + threadIdx.y;
    int ia = blockIdx.z;

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;   /* block is 16x16 */

    float gM[9] = {0,0,0,0,0,0,0,0,0};

    if (iu < N_det && iv < Ny) {
        const float *M = mats + (size_t)ia * 9;
        float u = (float)iu - half_det;
        float v = (float)iv - half_y;
        float ox = u * M[0] + v * M[3];
        float oy = u * M[1] + v * M[4];
        float oz = u * M[2] + v * M[5];
        float dxs = M[6], dys = M[7], dzs = M[8];

        float gs = grad_sino[((size_t)ia * Ny + iv) * N_det + iu];

        if (gs != 0.0f) {
            /* dL/dM[3r+c] = gs * sum_s d_c(s) * {u, v, s}[r] with d = grad of
             * the trilinear sample: u, v and gs are per-ray constants, so the
             * loop keeps S_c = sum d_c and T_c = sum d_c*s (6 accumulators)
             * and they are applied once after it (phase1b audit R2). */
            float Sx = 0.0f, Sy = 0.0f, Sz = 0.0f;
            float Tx = 0.0f, Ty = 0.0f, Tz = 0.0f;
            for (int step = -ray_len; step <= ray_len; step++) {
                float s  = (float)step;
                /* Same position arithmetic as before (and as grad_rotmat_cpu),
                 * so samples fall on the same side of the trilinear kinks. */
                float fx = ox + s * dxs + half_x;
                float fy = oy + s * dys + half_y;
                float fz = oz + s * dzs + half_z;

                /* Loose bounds, the same as forward_project_rotmat_kernel
                 * (border-mode linear texture) and grad_rotmat_cpu: a sample
                 * contributes while any of its 8 corners is inside the
                 * volume, i.e. fx in (-1, Nx) and likewise for y, z. Corners
                 * outside the volume read 0 through the border-mode point
                 * texture, so the samples within one voxel of a face carry
                 * the (large) derivative of the fall-off to zero. A tight
                 * [0, N-1) test drops them and is not the gradient of the
                 * forward. floorf, not (int) truncation, because fx can be
                 * negative here. */
                if (fx <= -1.0f || fx >= (float)Nx ||
                    fy <= -1.0f || fy >= (float)Ny ||
                    fz <= -1.0f || fz >= (float)Nz)
                    continue;

                /* Corner indices stay floats (exact: |x0| < 2^24), so the
                 * texture coordinates need no float->int->float round trip. */
                float x0 = floorf(fx), y0 = floorf(fy), z0 = floorf(fz);
                float wx = fx - x0, wy = fy - y0, wz = fz - z0;
                float cx0 = x0 + 0.5f, cx1 = x0 + 1.5f;
                float cy0 = y0 + 0.5f, cy1 = y0 + 1.5f;
                float cz0 = z0 + 0.5f, cz1 = z0 + 1.5f;

                /* Point-filter tex3D reads: texel centre i+0.5 with
                 * cudaFilterModePoint reads texel i exactly. The hardware
                 * promotes fp16 texels to fp32, so tex3D<float> is correct
                 * for both T=float and T=__half.
                 * tex3D axis mapping: (x→Nx width, z→Nz height, y→Ny depth).
                 * Naming v<y><z><x>: v001 is the x+1 corner. */
                float v000 = tex3D<float>(vol_tex, cx0, cz0, cy0);
                float v001 = tex3D<float>(vol_tex, cx1, cz0, cy0);
                float v010 = tex3D<float>(vol_tex, cx0, cz1, cy0);
                float v011 = tex3D<float>(vol_tex, cx1, cz1, cy0);
                float v100 = tex3D<float>(vol_tex, cx0, cz0, cy1);
                float v101 = tex3D<float>(vol_tex, cx1, cz0, cy1);
                float v110 = tex3D<float>(vol_tex, cx0, cz1, cy1);
                float v111 = tex3D<float>(vol_tex, cx1, cz1, cy1);

                /* Analytic derivatives of the trilinear interpolant in lerp
                 * form: d.. are the x-differences on the 4 (y,z) edges, e..
                 * the x-lerped edge values. dfx = bilerp_yz(d);
                 * dfy, dfz = d/dy, d/dz of bilerp_yz(e). */
                float d00 = v001 - v000, d01 = v011 - v010;
                float d10 = v101 - v100, d11 = v111 - v110;
                float e00 = fmaf(wx, d00, v000), e01 = fmaf(wx, d01, v010);
                float e10 = fmaf(wx, d10, v100), e11 = fmaf(wx, d11, v110);
                float a0  = fmaf(wz, d01 - d00, d00), a1 = fmaf(wz, d11 - d10, d10);
                float dfx = fmaf(wy, a1 - a0, a0);
                float by0 = e10 - e00, by1 = e11 - e01;
                float dfy = fmaf(wz, by1 - by0, by0);
                float bz0 = e01 - e00, bz1 = e11 - e10;
                float dfz = fmaf(wy, bz1 - bz0, bz0);

                Sx += dfx;  Sy += dfy;  Sz += dfz;
                Tx = fmaf(dfx, s, Tx);  Ty = fmaf(dfy, s, Ty);  Tz = fmaf(dfz, s, Tz);
            }
            const float gu = gs * u, gv = gs * v;
            gM[0] = gu * Sx;  gM[1] = gu * Sy;  gM[2] = gu * Sz;
            gM[3] = gv * Sx;  gM[4] = gv * Sy;  gM[5] = gv * Sz;
            gM[6] = gs * Tx;  gM[7] = gs * Ty;  gM[8] = gs * Tz;
        }
    }

    /* Deterministic block reduction (no atomics, fixed order): xor-shuffle
     * within each warp (the 16x16 block is 8 full warps, so the full mask is
     * valid), lane 0 of warp w stores sw[w][k], then thread k < 9 sums the 8
     * warps in order and writes this block's partial. grad_rotmat_finish_kernel
     * sums the partials of each view in a fixed order. Only 288 B of static
     * shared memory: the 3D-texture kernels are L1/texture-cache capacity
     * sensitive, and the former 9 KB [9 x 256] tree moved the smem/L1
     * carveout and cost 1.3-1.5x (phase1b audit R1). Every thread reaches the
     * barrier (no early return above). */
    __shared__ float sw[8][9];
    const int lane = tid & 31, warp = tid >> 5;
    #pragma unroll
    for (int k = 0; k < 9; k++) {
        float x = gM[k];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            x += __shfl_xor_sync(0xffffffffu, x, off);
        if (lane == 0) sw[warp][k] = x;
    }
    __syncthreads();
    if (tid < 9) {
        float a = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; w++) a += sw[w][tid];
        const int nb = gridDim.x * gridDim.y;
        const int b  = blockIdx.y * gridDim.x + blockIdx.x;
        partials[((size_t)ia * nb + b) * 9 + tid] = a;
    }
}

/* Second pass of the grad_rotmat reduction.
 * Grid: (n_angles)   Block: (32, 9) -- warp k sums matrix entry k.
 * Lane l adds the partials of blocks l, l+32, l+64, ... in order, then a
 * fixed xor butterfly combines the 32 lanes: the result depends only on the
 * launch geometry, never on block scheduling, so it is bit-reproducible. */
__global__ static void grad_rotmat_finish_kernel(
    const float *__restrict__ partials,   /* [n_angles, nb, 9] */
    float       *__restrict__ grad_mats,  /* [n_angles*9] (written) */
    int nb
) {
    const int ia = blockIdx.x, k = threadIdx.y, lane = threadIdx.x;
    const float *p = partials + (size_t)ia * nb * 9 + k;
    float a = 0.0f;
    for (int b = lane; b < nb; b += 32) a += p[(size_t)b * 9];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        a += __shfl_xor_sync(0xffffffffu, a, off);
    if (lane == 0) grad_mats[(size_t)ia * 9 + k] = a;
}

/* Launch grad_rotmat_kernel on a TC_VOL_3D cache that already holds the
 * volume (no upload: the cached path reuses the forward's upload). */
static void grad_rotmat_launch(
    int io_dtype, TextureCache *vol_cache,
    const float *grad_sino, const float *mats, float *grad_mats,
    int Ny, int Nz, int Nx, int n_angles, int N_det,
    cudaStream_t stream
) {
    if (vol_cache->tex_point == 0)
        throw std::invalid_argument("grad_rotmat: tex_point not initialised "
                                    "(cache was not created as TC_VOL_3D with channels==1)");

    ProjGeomRotmat g = proj_geom_rotmat(Ny, Nz, Nx, n_angles, N_det);

    dim3 block(16, 16, 1);
    dim3 grid((g.N_det + 15) / 16,
              (g.Ny    + 15) / 16,
              g.n_angles);
    const int nb = (int)(grid.x * grid.y);
    StreamScratch partials((size_t)g.n_angles * nb * 9 * sizeof(float), stream);

    if (io_dtype == 0)
        grad_rotmat_kernel<float><<<grid, block, 0, stream>>>(
            vol_cache->tex_point, grad_sino, mats, partials.floats(),
            g.Ny, g.Nz, g.Nx, g.n_angles, g.N_det,
            g.half_det, g.half_x, g.half_y, g.half_z, g.ray_len
        );
    else
        grad_rotmat_kernel<__half><<<grid, block, 0, stream>>>(
            vol_cache->tex_point, grad_sino, mats, partials.floats(),
            g.Ny, g.Nz, g.Nx, g.n_angles, g.N_det,
            g.half_det, g.half_x, g.half_y, g.half_z, g.ray_len
        );
    CUDA_CHECK_KERNEL();
    grad_rotmat_finish_kernel<<<g.n_angles, dim3(32, 9, 1), 0, stream>>>(
        partials.floats(), grad_mats, nb);
    CUDA_CHECK_KERNEL();
}

void grad_rotmat_cuda_cached(
    int io_dtype,
    TextureCache *vol_cache,
    const float *grad_sino,
    const float *mats,
    float       *grad_mats,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
) {
    check_io_dtype(io_dtype, "grad_rotmat_cuda_cached");
    tc_validate(vol_cache, TC_VOL_3D, io_dtype, Ny, Nz, Nx, "grad_rotmat_cuda_cached");
    grad_rotmat_launch(io_dtype, vol_cache, grad_sino, mats, grad_mats,
                       Ny, Nz, Nx, n_angles, N_det, stream);
}

void grad_rotmat_cuda(
    const float *vol,
    const float *grad_sino,
    const float *mats,
    float       *grad_mats,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
) {
    ScopedTextureCache tc(TC_VOL_3D, TC_DTYPE_FP32, 1, Ny, Nz, Nx, stream);
    tc_upload(tc.get(), vol, stream);
    grad_rotmat_launch(0, tc.get(), grad_sino, mats, grad_mats,
                       Ny, Nz, Nx, n_angles, N_det, stream);
    tc.finish();
}

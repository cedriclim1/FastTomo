/* ── fasttomo/csrc/cuda/grad_angle.cu ────────────────────────────────────────
 * Gradient of forward projection with respect to tilt angles.
 * Kernels: grad_angle_kernel, grad_angle_finish_kernel.
 * Host wrapper: grad_angle_cuda.
 */

#include "common.cuh"
#include "texture_cache.h"
#include "../include/proj_geom.h"
#include "ops.h"
#include "scratch_pool.cuh"
#include "parallel_clip.cuh"

/* ── gradient of forward projection w.r.t. tilt angles ───────────────────────
 * Grid:  (ceil(N_det/16), ceil(n_angles/16), Ny)   Block: (16, 16, 1)
 * Same detector-driven layout as forward_project_parallel_kernel.
 *
 * Sampling rule: the forward's (loose, fx in (-1, Nx), fz in (-1, Nz), floor
 * corners, corners outside the slice read 0). ga_load returns 0 outside;
 * samples whose whole stencil is inside (almost all) load their 4 corners
 * directly. At a grid-line kink (wx == 0 or wz == 0) the derivative across
 * the line is the average of the forward and the backward difference
 * (column ix0-1 / row iz0-1), i.e. the central one, as in grad_angle_cpu
 * and grad_rotmat; the bounds test is inclusive, [-1, N], so a sample on
 * the outer end of a tent (value 0, one-sided slope) is kept. Off the
 * kinks the arithmetic is the previous kernel's.
 *
 * Reduction strategy: each (ia, ty) row of 16 threads sits inside a single
 * warp (block is 16×16 = 256 threads = 8 warps; for ty even, those 16
 * threads occupy lanes 0..15; for ty odd, lanes 16..31). A half-warp
 * butterfly via __shfl_xor_sync collapses the 16 detector cells, after
 * which threadIdx.x==0 writes the row's partial to
 * partials[(ia*Ny + iy)*gridDim.x + blockIdx.x]; grad_angle_finish_kernel
 * sums the Ny*gridDim.x partials of each view in a fixed order. No atomics:
 * the gradient is bit-reproducible run to run (phase1b audit R5).
 */
__device__ __forceinline__ float ga_load(const float *__restrict__ slice,
                                         int iz, int ix, int Nz, int Nx) {
    return ((unsigned)ix < (unsigned)Nx && (unsigned)iz < (unsigned)Nz)
           ? __ldg(slice + (size_t)iz * Nx + ix) : 0.0f;
}

__global__ static void grad_angle_kernel(
    const float *__restrict__ vol,     /* [Ny, Nz, Nx] */
    const float *__restrict__ grad_sino,  /* [n_angles, Ny, N_det] */
    const float *__restrict__ cos_a,
    const float *__restrict__ sin_a,
    float       *__restrict__ partials,   /* [n_angles, Ny, gridDim.x] */
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    float half_det, float half_x, float half_z, int ray_len
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int ia = blockIdx.y * blockDim.y + threadIdx.y;
    int iy = (int)blockIdx.z;

    float local_g = 0.0f;
    float ca = 0.0f, sa = 0.0f, gs = 0.0f;
    float t = id - half_det;
    int lo = INT_MAX, hi = INT_MIN;        /* empty: the warp union's identity */
    if (id < N_det && ia < n_angles) {
        ca = cos_a[ia];
        sa = sin_a[ia];
        gs = grad_sino[((size_t)ia * Ny + iy) * N_det + id];
        if (gs != 0.0f)
            fp_step_range(t, ca, sa, Nx, Nz, half_x, half_z, ray_len, lo, hi);
    }
    /* Clipped step range (parallel_clip.cuh), united over the warp (its two
     * 16-lane rows are two views): the lanes march the same steps, as in the
     * forward. A lane's extra steps fail its bounds test, and clipping only
     * drops steps that test rejects, so the result bits are those of the
     * full [-ray_len, ray_len] loop. Every lane reaches the shuffles. */
    warp_uniform_range(lo, hi);

    if (gs != 0.0f) {
        const float *slice = vol + (size_t)iy * Nz * Nx;
        float acc = 0.0f;   /* gs is factored out of the step sum */

        for (int step = lo; step <= hi; step++) {
            float s  = (float)step;
            float x  = t * ca - s * sa;
            float z  = t * sa + s * ca;
            float fx = x + half_x;
            float fz = z + half_z;

            /* inclusive loose bounds; negated conjunction: a NaN
             * sample (NaN/Inf angle) fails it and is skipped */
            if (!(fx >= -1.0f && fx <= (float)Nx &&
                  fz >= -1.0f && fz <= (float)Nz))
                continue;

            float x0 = floorf(fx), z0 = floorf(fz);
            int ix0 = (int)x0, iz0 = (int)z0;
            float wx = fx - x0, wz = fz - z0;

            float v00, v01, v10, v11;
            if ((unsigned)ix0 < (unsigned)(Nx - 1) && (unsigned)iz0 < (unsigned)(Nz - 1)) {
                const float *p = slice + (size_t)iz0 * Nx + ix0;
                v00 = __ldg(p);      v01 = __ldg(p + 1);
                v10 = __ldg(p + Nx); v11 = __ldg(p + Nx + 1);
            } else {                 /* stencil crosses the border */
                v00 = ga_load(slice, iz0,     ix0,     Nz, Nx);
                v01 = ga_load(slice, iz0,     ix0 + 1, Nz, Nx);
                v10 = ga_load(slice, iz0 + 1, ix0,     Nz, Nx);
                v11 = ga_load(slice, iz0 + 1, ix0 + 1, Nz, Nx);
            }

            float dvol_dx = (1.0f - wz) * (v01 - v00) + wz * (v11 - v10);
            float dvol_dz = (1.0f - wx) * (v10 - v00) + wx * (v11 - v01);
            if (wx == 0.0f || wz == 0.0f) {    /* on a grid line */
                if (wx == 0.0f)
                    dvol_dx = 0.5f * (dvol_dx
                        + ((1.0f - wz) * (v00 - ga_load(slice, iz0, ix0 - 1, Nz, Nx))
                           + wz * (v10 - ga_load(slice, iz0 + 1, ix0 - 1, Nz, Nx))));
                if (wz == 0.0f)
                    dvol_dz = 0.5f * (dvol_dz
                        + ((1.0f - wx) * (v00 - ga_load(slice, iz0 - 1, ix0, Nz, Nx))
                           + wx * (v01 - ga_load(slice, iz0 - 1, ix0 + 1, Nz, Nx))));
            }

            acc += dvol_dx * (-z) + dvol_dz * x;
        }
        local_g = gs * acc;
    }

    /* Half-warp shuffle reduction over threadIdx.x (16 lanes share each ia row).
     * Lanes for ty even occupy 0..15 within their warp; for ty odd, 16..31.
     * The xor butterfly with offsets ≤ 8 stays within the half-warp, so a
     * full-warp mask gives the correct partial sum on every lane. */
    unsigned mask = 0xFFFFFFFFu;
    local_g += __shfl_xor_sync(mask, local_g, 8);
    local_g += __shfl_xor_sync(mask, local_g, 4);
    local_g += __shfl_xor_sync(mask, local_g, 2);
    local_g += __shfl_xor_sync(mask, local_g, 1);

    if (threadIdx.x == 0 && ia < n_angles)
        partials[((size_t)ia * Ny + iy) * gridDim.x + blockIdx.x] = local_g;
}

/* Second pass: one 256-thread block per view sums that view's n_part
 * partials (contiguous) in a fixed order: thread t adds entries t, t+256, ...
 * in order, a fixed xor butterfly combines each warp, and thread 0 adds the
 * 8 warp sums in order. Writes grad_angles[ia]. */
__global__ static void grad_angle_finish_kernel(
    const float *__restrict__ partials,    /* [n_angles, n_part] */
    float       *__restrict__ grad_angles, /* [n_angles] (written) */
    int n_part
) {
    __shared__ float sw[8];
    const int ia = blockIdx.x, t = threadIdx.x;
    const float *p = partials + (size_t)ia * n_part;
    float a = 0.0f;
    for (int j = t; j < n_part; j += 256) a += p[j];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        a += __shfl_xor_sync(0xffffffffu, a, off);
    if ((t & 31) == 0) sw[t >> 5] = a;
    __syncthreads();
    if (t == 0) {
        float r = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; w++) r += sw[w];
        grad_angles[ia] = r;
    }
}

void grad_angle_cuda(
    const float *vol,
    const float *grad_sino,
    const float *cos_a,
    const float *sin_a,
    float       *grad_angles,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
) {
    ProjGeomParallel g = proj_geom_parallel(Ny, Nz, Nx, n_angles, N_det);

    dim3 block(16, 16, 1);
    dim3 grid((g.N_det    + 15) / 16,
              (g.n_angles + 15) / 16,
              g.Ny);

    const int n_part = (int)grid.x * g.Ny;   /* partials per view */
    StreamScratch partials((size_t)g.n_angles * n_part * sizeof(float), stream);

    grad_angle_kernel<<<grid, block, 0, stream>>>(
        vol, grad_sino, cos_a, sin_a, partials.floats(),
        g.Ny, g.Nz, g.Nx, g.n_angles, g.N_det,
        g.half_det, g.half_x, g.half_z, g.ray_len
    );
    CUDA_CHECK_KERNEL();
    grad_angle_finish_kernel<<<g.n_angles, 256, 0, stream>>>(
        partials.floats(), grad_angles, n_part);
    CUDA_CHECK_KERNEL();
}

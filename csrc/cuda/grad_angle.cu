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

/* ── gradient of forward projection w.r.t. tilt angles ───────────────────────
 * Grid:  (ceil(N_det/16), ceil(n_angles/16), Ny)   Block: (16, 16, 1)
 * Same detector-driven layout as forward_project_parallel_kernel.
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

    if (id < N_det && ia < n_angles) {
        float ca = cos_a[ia], sa = sin_a[ia];
        float t   = id - half_det;
        float gs  = grad_sino[((size_t)ia * Ny + iy) * N_det + id];

        if (gs != 0.0f) {
            const float *slice = vol + (size_t)iy * Nz * Nx;
            float acc = 0.0f;   /* gs is factored out of the step sum */

            for (int step = -ray_len; step <= ray_len; step++) {
                float s  = (float)step;
                float x  = t * ca - s * sa;
                float z  = t * sa + s * ca;
                float fx = x + half_x;
                float fz = z + half_z;

                if (fx < 0.0f || fx >= (float)(Nx - 1) ||
                    fz < 0.0f || fz >= (float)(Nz - 1))
                    continue;

                int ix0 = (int)fx, iz0 = (int)fz;
                int ix1 = ix0 + 1, iz1 = iz0 + 1;
                float wx = fx - ix0, wz = fz - iz0;

                float v00 = slice[(size_t)iz0 * Nx + ix0];
                float v01 = slice[(size_t)iz0 * Nx + ix1];
                float v10 = slice[(size_t)iz1 * Nx + ix0];
                float v11 = slice[(size_t)iz1 * Nx + ix1];

                float dvol_dx = (1.0f - wz) * (v01 - v00) + wz * (v11 - v10);
                float dvol_dz = (1.0f - wx) * (v10 - v00) + wx * (v11 - v01);

                acc += dvol_dx * (-z) + dvol_dz * x;
            }
            local_g = gs * acc;
        }
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

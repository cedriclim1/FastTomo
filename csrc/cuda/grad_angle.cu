/* ── fasttomo/csrc/cuda/grad_angle.cu ────────────────────────────────────────
 * Gradient of forward projection with respect to tilt angles.
 * Kernel: grad_angle_kernel.
 * Host wrapper: grad_angle_cuda.
 */

#include "common.cuh"
#include "texture_cache.h"
#include "../include/proj_geom.h"
#include "ops.h"

/* ── gradient of forward projection w.r.t. tilt angles ───────────────────────
 * Grid:  (ceil(N_det/16), ceil(n_angles/16), Ny)   Block: (16, 16, 1)
 * Same detector-driven layout as forward_project_kernel.
 *
 * Reduction strategy: each (ia, ty) row of 16 threads sits inside a single
 * warp (block is 16×16 = 256 threads = 8 warps; for ty even, those 16
 * threads occupy lanes 0..15; for ty odd, lanes 16..31). A half-warp
 * butterfly via __shfl_xor_sync collapses the 16 detector cells, after
 * which threadIdx.x==0 issues one atomicAdd per row.
 */
__global__ static void grad_angle_kernel(
    const float *__restrict__ volume,     /* [Ny, Nz, Nx] */
    const float *__restrict__ grad_sino,  /* [n_angles, Ny, N_det] */
    const float *__restrict__ cos_a,
    const float *__restrict__ sin_a,
    float       *__restrict__ grad_angles,/* [n_angles] */
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
            const float *slice = volume + (size_t)iy * Nz * Nx;

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

                local_g += gs * (dvol_dx * (-z) + dvol_dz * x);
            }
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
        atomicAdd(&grad_angles[ia], local_g);
}

void grad_angle_cuda(
    const float *d_volume,
    const float *d_grad_sino,
    const float *d_cos_a,
    const float *d_sin_a,
    float       *d_grad_angles,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
) {
    ProjGeom2D g = proj_geom_2d(Ny, Nz, Nx, n_angles, N_det);

    dim3 block(16, 16, 1);
    dim3 grid((g.N_det    + 15) / 16,
              (g.n_angles + 15) / 16,
              g.Ny);

    grad_angle_kernel<<<grid, block, 0, stream>>>(
        d_volume, d_grad_sino, d_cos_a, d_sin_a, d_grad_angles,
        g.Ny, g.Nz, g.Nx, g.n_angles, g.N_det,
        g.half_det, g.half_x, g.half_z, g.ray_len
    );
    CUDA_CHECK_KERNEL();
}

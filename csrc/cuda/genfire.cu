/* ── fasttomo/csrc/cuda/genfire.cu ───────────────────────────────────────────
 * GENFIRE 2D Fourier gridding kernel and host wrapper.
 * Kernel: genfire_gridding_2d_kernel.
 * Host wrapper: genfire_gridding_2d_cuda.
 */

#include "common.cuh"
#include "ops.h"

/* ── GENFIRE 2D Fourier gridding ─────────────────────────────────────────── */

__global__ static void genfire_gridding_2d_kernel(
    const float *fk_real, const float *fk_imag,
    const float *cos_a,   const float *sin_a,
    float *grid_real, float *grid_imag, float *weight_grid,
    int bigN, int Ny, int n_angles, float cutoff_dist
) {
    int ixi = blockIdx.x * blockDim.x + threadIdx.x;
    int ia  = (int)blockIdx.y;
    int iy  = (int)blockIdx.z;

    if (ixi >= bigN || ia >= n_angles || iy >= Ny) return;

    int   half = bigN / 2;
    float xi   = (float)(ixi - half);
    float kx   = xi * cos_a[ia];
    float kz   = xi * sin_a[ia];
    int   ikx  = __float2int_rn(kx);
    int   ikz  = __float2int_rn(kz);
    float dx   = kx - (float)ikx;
    float dz   = kz - (float)ikz;
    float dist = sqrtf(dx * dx + dz * dz);
    if (dist > cutoff_dist) return;

    int gx = ikx + half;
    int gz = ikz + half;
    if (gx < 0 || gx >= bigN || gz < 0 || gz >= bigN) return;

    float  w       = 1.0f / (dist + 1e-5f);
    size_t fk_idx  = ((size_t)ia * Ny + iy) * bigN + (size_t)ixi;
    size_t out_idx = (size_t)iy * bigN * bigN + (size_t)gz * bigN + (size_t)gx;

    atomicAdd(&grid_real[out_idx],    w * fk_real[fk_idx]);
    atomicAdd(&grid_imag[out_idx],    w * fk_imag[fk_idx]);
    atomicAdd(&weight_grid[out_idx],  w);
}

void genfire_gridding_2d_cuda(
    const float *fk_real, const float *fk_imag,
    const float *cos_a,   const float *sin_a,
    float *grid_real, float *grid_imag, float *weight_grid,
    int bigN, int Ny, int n_angles, float cutoff_dist,
    cudaStream_t stream
) {
    dim3 block(128, 1, 1);
    dim3 grid((bigN + 127) / 128, (unsigned)n_angles, (unsigned)Ny);
    genfire_gridding_2d_kernel<<<grid, block, 0, stream>>>(
        fk_real, fk_imag, cos_a, sin_a,
        grid_real, grid_imag, weight_grid,
        bigN, Ny, n_angles, cutoff_dist
    );
    CUDA_CHECK_KERNEL();
}

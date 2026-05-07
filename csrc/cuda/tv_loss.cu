/* ── fasttomo/csrc/cuda/tv_loss.cu ────────────────────────────────────────
 * Fused 3-D isotropic Total-Variation loss + analytic gradient.
 *
 *   loss(vol) = (1/N) · Σ_{i,j,k} sqrt( dy² + dz² + dx² + eps )
 *
 *   dy(i,j,k) = vol[i+1, j,   k  ] − vol[i, j, k]
 *   dz(i,j,k) = vol[i,   j+1, k  ] − vol[i, j, k]
 *   dx(i,j,k) = vol[i,   j,   k+1] − vol[i, j, k]
 *
 * Corner index set: i ∈ [0, Ny−2], j ∈ [0, Nz−2], k ∈ [0, Nx−2].
 * N = (Ny−1)·(Nz−1)·(Nx−1). Matches the standard half-pixel-shifted TV
 * formulation used in the notebook reference.
 *
 * Forward kernel: corner-driven, block-level shared-memory reduction →
 * one atomicAdd per block to a single fp32 accumulator. The scalar 1/N
 * normalization is applied by the Python caller.
 *
 * Backward kernel: voxel-driven gather. Each thread sums gradient
 * contributions from the (up to 4) corners that touch its voxel,
 * recomputing s = sqrt(dx²+dy²+dz²+eps) per corner from `vol`. The
 * caller pre-multiplies the upstream scalar grad by 1/N and passes it
 * as a single-element device tensor so the kernel only needs one
 * broadcast read.
 */

#include "common.cuh"
#include "ops.h"

/* ── forward ──────────────────────────────────────────────────────────── */

__global__ static void tv_loss_iso_3d_kernel(
    const float *__restrict__ vol,    /* [Ny, Nz, Nx] */
    float       *__restrict__ acc,    /* [1] — Σ s, atomicAdd target */
    int Ny, int Nz, int Nx,
    float eps
) {
    /* Block layout: (blockDim.x, blockDim.y, 1) over (k_corner, j_corner).
     * gridDim.z covers i_corner ∈ [0, Ny-2]. Threads outside the corner
     * grid contribute 0 so the block reduction is straightforward. */
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = (int)blockIdx.z;

    float local = 0.0f;
    if (k < Nx - 1 && j < Nz - 1 && i < Ny - 1) {
        size_t base = ((size_t)i * Nz + j) * Nx + k;
        float v000 = vol[base];
        float vp_y = vol[base + (size_t)Nz * Nx];   /* vol[i+1, j,   k  ] */
        float vp_z = vol[base + (size_t)Nx];        /* vol[i,   j+1, k  ] */
        float vp_x = vol[base + 1];                 /* vol[i,   j,   k+1] */

        float dy = vp_y - v000;
        float dz = vp_z - v000;
        float dx = vp_x - v000;
        local = sqrtf(dx * dx + dy * dy + dz * dz + eps);
    }

    /* Block-level reduction in shared memory. blockDim.x is a power of 2
     * (we launch with 32×4×1 = 128 threads; 32 is required for warp-level
     * coalescing of vol reads anyway). */
    extern __shared__ float smem[];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int nthreads = blockDim.x * blockDim.y;
    smem[tid] = local;
    __syncthreads();

    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(acc, smem[0]);
}

void tv_loss_iso_3d_cuda(
    const float *d_vol,
    float       *d_acc,
    int Ny, int Nz, int Nx,
    float eps,
    cudaStream_t stream
) {
    if (Ny < 2 || Nz < 2 || Nx < 2) {
        /* No corners; caller's d_acc is already zero — leave it. */
        return;
    }

    dim3 block(32, 4, 1);
    dim3 grid(((Nx - 1) + block.x - 1) / block.x,
              ((Nz - 1) + block.y - 1) / block.y,
              (Ny - 1));

    size_t smem_bytes = block.x * block.y * sizeof(float);
    tv_loss_iso_3d_kernel<<<grid, block, smem_bytes, stream>>>(
        d_vol, d_acc, Ny, Nz, Nx, eps
    );
    CUDA_CHECK_KERNEL();
}

/* ── backward ─────────────────────────────────────────────────────────── */

__device__ __forceinline__ float corner_inv_s(
    const float *__restrict__ vol,
    int i, int j, int k,
    int Nz, int Nx,
    float eps,
    float &dy, float &dz, float &dx
) {
    size_t base = ((size_t)i * Nz + j) * Nx + k;
    float v000 = vol[base];
    dy = vol[base + (size_t)Nz * Nx] - v000;
    dz = vol[base + (size_t)Nx]      - v000;
    dx = vol[base + 1]               - v000;
    return rsqrtf(dx * dx + dy * dy + dz * dz + eps);
}

__global__ static void tv_loss_iso_3d_grad_kernel(
    const float *__restrict__ vol,         /* [Ny, Nz, Nx] */
    const float *__restrict__ g_scaled,    /* [1] = grad_out · 1/N */
    float       *__restrict__ grad_vol,    /* [Ny, Nz, Nx] */
    int Ny, int Nz, int Nx,
    float eps
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;   /* k axis (Nx) */
    int b = blockIdx.y * blockDim.y + threadIdx.y;   /* j axis (Nz) */
    int a = blockIdx.z * blockDim.z + threadIdx.z;   /* i axis (Ny) */
    if (a >= Ny || b >= Nz || c >= Nx) return;

    float g = __ldg(g_scaled);
    float acc = 0.0f;
    float dy, dz, dx;

    /* Corner anchored at (a, b, c) — vol[a,b,c] is the "central" voxel.
     * Contribution: -(dx + dy + dz) / s. */
    if (a < Ny - 1 && b < Nz - 1 && c < Nx - 1) {
        float inv_s = corner_inv_s(vol, a, b, c, Nz, Nx, eps, dy, dz, dx);
        acc += -(dx + dy + dz) * inv_s;
    }
    /* Corner anchored at (a-1, b, c) — vol[a,b,c] is the +y neighbor.
     * Contribution: dy / s. */
    if (a >= 1 && b < Nz - 1 && c < Nx - 1) {
        float inv_s = corner_inv_s(vol, a - 1, b, c, Nz, Nx, eps, dy, dz, dx);
        acc += dy * inv_s;
    }
    /* Corner anchored at (a, b-1, c) — vol[a,b,c] is the +z neighbor.
     * Contribution: dz / s. */
    if (a < Ny - 1 && b >= 1 && c < Nx - 1) {
        float inv_s = corner_inv_s(vol, a, b - 1, c, Nz, Nx, eps, dy, dz, dx);
        acc += dz * inv_s;
    }
    /* Corner anchored at (a, b, c-1) — vol[a,b,c] is the +x neighbor.
     * Contribution: dx / s. */
    if (a < Ny - 1 && b < Nz - 1 && c >= 1) {
        float inv_s = corner_inv_s(vol, a, b, c - 1, Nz, Nx, eps, dy, dz, dx);
        acc += dx * inv_s;
    }

    grad_vol[((size_t)a * Nz + b) * Nx + c] = g * acc;
}

void tv_loss_iso_3d_grad_cuda(
    const float *d_vol,
    const float *d_g_scaled,
    float       *d_grad_vol,
    int Ny, int Nz, int Nx,
    float eps,
    cudaStream_t stream
) {
    if (Ny < 2 || Nz < 2 || Nx < 2) {
        /* No corners — grad is zero everywhere; caller pre-zeroed. */
        return;
    }

    dim3 block(32, 4, 2);
    dim3 grid((Nx + block.x - 1) / block.x,
              (Nz + block.y - 1) / block.y,
              (Ny + block.z - 1) / block.z);

    tv_loss_iso_3d_grad_kernel<<<grid, block, 0, stream>>>(
        d_vol, d_g_scaled, d_grad_vol, Ny, Nz, Nx, eps
    );
    CUDA_CHECK_KERNEL();
}

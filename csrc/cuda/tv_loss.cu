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
 * Forward: deterministic two-pass reduction. Pass 1 gives each warp one
 * corner row (i, j) (lanes stride k, coalesced), sums it per lane in fp32,
 * then in fp64 across the warp (xor-shuffle tree) and the block's 8 warps
 * (fixed order) into one fp64 partial per block. Pass 2 (one block) sums
 * the partials in a fixed order and writes the fp32 Σ s. No atomics, so the
 * result is bit-reproducible run to run and independent of the device (the
 * grid depends on the shape only). The scalar 1/N normalization is applied
 * by the Python caller.
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

/* Pass 1. Block: TV_FWD_WARPS warps, one corner row (i, j) per warp; grid:
 * ceil(rows / TV_FWD_WARPS) blocks, rows = (Ny-1)(Nz-1). Out-of-range rows
 * contribute 0 (no early return: every thread reaches the barrier). The
 * per-lane fp32 sum has at most ceil((Nx-1)/32) terms (64 at Nx = 2049);
 * everything above it is fp64. */
#define TV_FWD_WARPS 8
#define TV_FIN_THREADS 1024

__global__ static void tv_loss_iso_3d_kernel(
    const float *__restrict__ vol,       /* [Ny, Nz, Nx] */
    double      *__restrict__ partials,  /* [gridDim.x] */
    int Ny, int Nz, int Nx,
    float eps
) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const long long rows = (long long)(Ny - 1) * (Nz - 1);
    const long long r = (long long)blockIdx.x * TV_FWD_WARPS + warp;

    float local = 0.0f;
    if (r < rows) {
        const int i = (int)(r / (Nz - 1));
        const int j = (int)(r - (long long)i * (Nz - 1));
        const float *row = vol + ((size_t)i * Nz + j) * Nx;
        const size_t sy = (size_t)Nz * Nx;
        for (int k = lane; k < Nx - 1; k += 32) {
            float v000 = row[k];
            float dy = row[k + sy] - v000;        /* vol[i+1, j,   k  ] */
            float dz = row[k + Nx] - v000;        /* vol[i,   j+1, k  ] */
            float dx = row[k + 1]  - v000;        /* vol[i,   j,   k+1] */
            local += sqrtf(dx * dx + dy * dy + dz * dz + eps);
        }
    }

    /* Warp: fixed xor-shuffle tree in fp64 (all 32 lanes active). */
    double w = (double)local;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        w += __shfl_xor_sync(0xffffffffu, w, off);

    /* Block: the warps' sums in warp order. */
    __shared__ double s_warp[TV_FWD_WARPS];
    if (lane == 0) s_warp[warp] = w;
    __syncthreads();
    if (threadIdx.x == 0) {
        double b = 0.0;
        #pragma unroll
        for (int q = 0; q < TV_FWD_WARPS; q++) b += s_warp[q];
        partials[blockIdx.x] = b;
    }
}

/* Pass 2, one block of TV_FIN_THREADS: thread t sums partials t, t + T, ...
 * in order, then a fixed xor-shuffle tree and the 32 warp sums in order.
 * Writes acc[0] (overwrites; the caller's zeroing is not needed). */
__global__ static void tv_loss_iso_3d_finalize_kernel(
    const double *__restrict__ partials, long long n_partials,
    float *__restrict__ acc
) {
    double t = 0.0;
    for (long long p = threadIdx.x; p < n_partials; p += TV_FIN_THREADS)
        t += partials[p];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        t += __shfl_xor_sync(0xffffffffu, t, off);
    __shared__ double s_warp[TV_FIN_THREADS / 32];
    if ((threadIdx.x & 31) == 0) s_warp[threadIdx.x >> 5] = t;
    __syncthreads();
    if (threadIdx.x == 0) {
        double b = 0.0;
        for (int q = 0; q < TV_FIN_THREADS / 32; q++) b += s_warp[q];
        acc[0] = (float)b;
    }
}

long long tv_loss_iso_3d_workspace(int Ny, int Nz, int Nx) {
    if (Ny < 2 || Nz < 2 || Nx < 2) return 0;
    long long rows = (long long)(Ny - 1) * (Nz - 1);
    return (rows + TV_FWD_WARPS - 1) / TV_FWD_WARPS;
}

void tv_loss_iso_3d_cuda(
    const float *d_vol,
    float       *d_acc,
    double      *d_partials,
    int Ny, int Nz, int Nx,
    float eps,
    cudaStream_t stream
) {
    if (Ny < 2 || Nz < 2 || Nx < 2) {
        /* No corners; caller's d_acc is already zero — leave it. */
        return;
    }
    long long n_blocks = tv_loss_iso_3d_workspace(Ny, Nz, Nx);
    if (n_blocks > 0x7fffffffLL)
        throw std::invalid_argument("tv_loss_iso_3d_cuda: volume too large");

    tv_loss_iso_3d_kernel<<<(unsigned)n_blocks, 32 * TV_FWD_WARPS, 0, stream>>>(
        d_vol, d_partials, Ny, Nz, Nx, eps
    );
    CUDA_CHECK_KERNEL();
    tv_loss_iso_3d_finalize_kernel<<<1, TV_FIN_THREADS, 0, stream>>>(
        d_partials, n_blocks, d_acc
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
        /* No corners — grad is zero everywhere; the caller zeroes it (only
         * in this case: otherwise the kernel below writes every voxel). */
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

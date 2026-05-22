/* ── fasttomo/csrc/cuda/mbir_icd.cu ──────────────────────────────────────────
 * Classical MBIR (Model-Based Iterative Reconstruction) for single-axis
 * parallel-beam tomography.
 *
 * Reference: Venkatakrishnan et al. 2015, "MBIR for Bright-Field Electron
 *            Tomography", IEEE TCI 1(1). See fasttomo-wiki/wiki/algorithms/mbir.md
 *            for the project-side narrative and the surrogate-MM derivation.
 *
 * Kernels:
 *   mbir_forward_project_kernel - voxel-driven bilinear scatter FP (1 sample
 *       per voxel, 2 detector pixels per angle, atomicAdd). Used to compute
 *       the initial residual e = g - A·f and to validate against the CPU
 *       reference. Distinct from `forward_project_kernel` in projection_2d.cu
 *       (which is ray-marching detector-driven); this voxel-driven model is
 *       what the ICD column-A access pattern reproduces.
 *   mbir_icd_sweep_kernel       - one outer iteration: per-Y-slice ICD walk
 *       through the slice's permutation, threads cooperate on the per-voxel
 *       data-term reduction over angles and qGGMRF-prior reduction over the
 *       26-neighbour stencil; thread 0 applies the closed-form update, all
 *       threads write the incremental error-sinogram delta non-atomically
 *       (correctness argument inline below).
 *
 * Host wrappers expose these as `mbir_forward_project_cuda` and
 * `mbir_icd_sweep_cuda`. The offset (d_k) and variance (σ) closed-form
 * updates live in the Python wrapper using PyTorch ops — see
 * fasttomo/algorithms.py::_run_mbir_single_scale.
 */

#include "common.cuh"
#include "texture_cache.h"
#include "ops.h"
#include "../include/proj_geom.h"

/* ── Hyperparameters in __constant__ memory ──────────────────────────────────
 * Loaded once per outer iteration before the kernel launches. Threads
 * broadcast-read so this is hardware-cached and free vs. global memory. */
struct MbirHyper {
    float p, q, c, sigma_f;        /* qGGMRF: p ∈ [1,2], q = 2, c, σ_f */
    float huber_T, huber_delta;    /* Huber: T (threshold), δ (tail slope) */
    float sigma;                   /* current noise std */
    float half_x, half_z, half_det;
};

static __constant__ MbirHyper mbir_H;

/* 26-neighbour stencil constants — same as in csrc/cpu/mbir_icd.c.
 * Indexing order: (dy, dz, dx) sweep over {-1,0,+1}^3 with the self entry
 * (0,0,0) skipped. Weights w_jk ∝ 1/dist, normalized to sum to 1. */
__constant__ int   d_NB_DX[26] = {
    -1,  0,  1, -1,  0,  1, -1,  0,  1,
    -1,  0,  1, -1,      1, -1,  0,  1,
    -1,  0,  1, -1,  0,  1, -1,  0,  1
};
__constant__ int   d_NB_DZ[26] = {
    -1, -1, -1,  0,  0,  0,  1,  1,  1,
    -1, -1, -1,  0,      0,  1,  1,  1,
    -1, -1, -1,  0,  0,  0,  1,  1,  1
};
__constant__ int   d_NB_DY[26] = {
    -1, -1, -1, -1, -1, -1, -1, -1, -1,
     0,  0,  0,  0,      0,  0,  0,  0,
     1,  1,  1,  1,  1,  1,  1,  1,  1
};
__constant__ float d_NB_W[26] = {
    0.030219f, 0.037014f, 0.030219f, 0.037014f, 0.052345f, 0.037014f, 0.030219f, 0.037014f, 0.030219f,
    0.037014f, 0.052345f, 0.037014f, 0.052345f,           0.052345f, 0.037014f, 0.052345f, 0.037014f,
    0.030219f, 0.037014f, 0.030219f, 0.037014f, 0.052345f, 0.037014f, 0.030219f, 0.037014f, 0.030219f
};

/* qGGMRF surrogate a_jk = ρ'(Δ)/Δ; see csrc/cpu/mbir_icd.c for derivation. */
__device__ __forceinline__ float qggmrf_a_jk_dev(float delta) {
    float u = fabsf(delta);
    if (u < 1e-12f) u = 1e-12f;
    float p = mbir_H.p, q = mbir_H.q, c = mbir_H.c, sf = mbir_H.sigma_f;
    float sp = __powf(sf, p);
    float sq = __powf(sf, q);
    float u_pmq    = __powf(u, p - q);
    float u_pm2    = __powf(u, p - 2.0f);
    float u_2pmqm2 = __powf(u, 2.0f * p - q - 2.0f);
    float D   = sp * c + u_pmq * sq;
    float num = p * c * sp * u_pm2 + q * sq * u_2pmqm2;
    return num / (D * D);
}

/* Huber surrogate weight b(r): 1 inside threshold, δ·T/|r| outside.
 * Reduces to all-ones when T = ∞ (conventional MBIR). */
__device__ __forceinline__ float huber_surr_weight_dev(float r) {
    float ar = fabsf(r);
    if (!isfinite(mbir_H.huber_T) || ar < mbir_H.huber_T) return 1.0f;
    return mbir_H.huber_delta * mbir_H.huber_T / ar;
}

/* Block reduction (64 threads = 2 warps) using warp shuffle + 1 shared slot.
 * `sh2` must point to a 2-element __shared__ scratch (one slot per warp). */
__device__ __forceinline__ float block_reduce_sum_64(float v, float *sh2) {
    unsigned mask = 0xFFFFFFFFu;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(mask, v, off);
    int tid  = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    if (lane == 0) sh2[warp] = v;
    __syncthreads();
    float total = (tid < 2) ? sh2[tid] : 0.0f;
    if (warp == 0) {
        total += __shfl_xor_sync(mask, total, 1);
    }
    /* broadcast back via shared */
    if (tid == 0) sh2[0] = total;
    __syncthreads();
    return sh2[0];
}

/* ────────────────────────────────────────────────────────────────────────────
 * Voxel-driven bilinear forward projection (MBIR's column-A model).
 *
 * Grid: ((Nx+15)/16, (Nz+15)/16, Ny). Block: (16, 16, 1).
 * Sinogram must be zeroed before the kernel launch — the kernel atomicAdds
 * each voxel's contribution into the 2 detector pixels it touches per angle.
 */
__global__ static void mbir_forward_project_kernel(
    const float *__restrict__ volume,    /* [Ny, Nz, Nx] */
    float *__restrict__       sinogram,  /* [n_angles, Ny, N_det] */
    const float *__restrict__ cos_a,
    const float *__restrict__ sin_a,
    int Ny, int Nz, int Nx, int n_angles, int N_det,
    float half_x, float half_z, float half_det
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int iy = blockIdx.z;
    if (ix >= Nx || iz >= Nz) return;

    float v = volume[((size_t)iy * Nz + iz) * Nx + ix];
    if (v == 0.0f) return;

    float x = ix - half_x;
    float z = iz - half_z;

    for (int ia = 0; ia < n_angles; ia++) {
        float ca = cos_a[ia], sa = sin_a[ia];
        float u = x * ca + z * sa + half_det;
        int u0 = (int)floorf(u);
        int u1 = u0 + 1;
        float wu = u - u0;
        size_t base = ((size_t)ia * Ny + iy) * N_det;
        if (u0 >= 0 && u0 < N_det) atomicAdd(&sinogram[base + u0], (1.0f - wu) * v);
        if (u1 >= 0 && u1 < N_det) atomicAdd(&sinogram[base + u1], wu * v);
    }
}

/* ────────────────────────────────────────────────────────────────────────────
 * One outer iteration of MBIR-AM-PE (CUDA).
 *
 * Block layout: gridDim = (Ny, 1, 1), blockDim = (64, 1, 1).
 *   - One block owns one Y-slice. Parallel-beam single-axis: Y-slices are
 *     fully independent for the data term, so blocks have no cross-block
 *     dependencies on `e` or `volume` within the data path.
 *   - The 26-neighbour qGGMRF reads of `volume` use a tex3D snapshot
 *     (read-only across the kernel) — cross-Y neighbour reads come from
 *     the start-of-outer-iter state ("Jacobi-loose" across Y).
 *
 * Within the block, voxels are processed sequentially in the per-slice
 * permutation order; threads cooperate on the per-voxel reductions.
 *
 * Race-condition argument for the non-atomic error-sinogram delta write:
 * each block exclusively owns the strip e[*, iy, *] (different iy across
 * different blocks). Within a single block, the 64 threads partition the
 * angle loop strided (ia = tid, tid + blockDim, ...); for any single
 * (ia, iy) pair, exactly one thread writes both detector pixels (u0 and
 * u0+1) the voxel touches. No collision possible. */
__global__ static void mbir_icd_sweep_kernel(
    cudaTextureObject_t       vol_snapshot_tex, /* tex3D over [Ny, Nz, Nx] read-only */
    float *__restrict__       volume,           /* [Ny, Nz, Nx] live (read+write) */
    float *__restrict__       e_sino,           /* [n_angles, Ny, N_det] live */
    const float *__restrict__ d_offsets,        /* [n_angles] for offset-aware residual */
    const float *__restrict__ lambdas,          /* [n_angles, Ny, N_det] or nullptr */
    const int   *__restrict__ perm,             /* [Ny, Nz*Nx] permutation */
    const float *__restrict__ cos_a,
    const float *__restrict__ sin_a,
    int Ny, int Nz, int Nx, int n_angles, int N_det
) {
    int iy  = blockIdx.x;
    int tid = threadIdx.x;
    if (iy >= Ny) return;

    const int nvox = Nz * Nx;
    const int *perm_slice = perm + (size_t)iy * nvox;
    float *vol_slice = volume + (size_t)iy * Nz * Nx;

    /* Shared: 2-element scratch for block reduction (one slot per warp);
     * + 4 scalars for {theta1, theta2, n1, n2, fj_old, fj_new, df}. */
    __shared__ float sh_red[2];
    __shared__ float sh_t1, sh_t2, sh_n1, sh_n2, sh_df;

    for (int s = 0; s < nvox; s++) {
        int lin = perm_slice[s];
        int ix  = lin % Nx;
        int iz  = lin / Nx;
        float fj_old = vol_slice[(size_t)iz * Nx + ix];
        float x = ix - mbir_H.half_x;
        float z = iz - mbir_H.half_z;

        /* ── data term: θ1, θ2 reduction over angles ──────────────────────
         * Each thread handles a stride of angles (tid, tid+64, tid+128, ...).
         * Within each angle, the same thread writes both detector pixels
         * (u0, u0+1) — preventing intra-block races on the later e-update. */
        float t1 = 0.0f, t2 = 0.0f;
        for (int ia = tid; ia < n_angles; ia += blockDim.x) {
            float ca = cos_a[ia], sa = sin_a[ia];
            float u  = x * ca + z * sa + mbir_H.half_det;
            int u0 = (int)floorf(u);
            int u1 = u0 + 1;
            float wu = u - u0;
            #pragma unroll
            for (int kk = 0; kk < 2; kk++) {
                int uk = (kk == 0) ? u0 : u1;
                if (uk < 0 || uk >= N_det) continue;
                float a_col = (kk == 0) ? (1.0f - wu) : wu;
                size_t off = ((size_t)ia * Ny + iy) * N_det + uk;
                float ek  = e_sino[off];
                float lam = lambdas ? lambdas[off] : 1.0f;
                float r   = ek * sqrtf(lam) / mbir_H.sigma;
                float bw  = huber_surr_weight_dev(r);
                float w   = bw * lam / (mbir_H.sigma * mbir_H.sigma);
                t1 += a_col * ek * w;
                t2 += a_col * a_col * w;
            }
        }
        float t1_red = block_reduce_sum_64(t1, sh_red);
        if (tid == 0) sh_t1 = -t1_red;     /* paper sign convention */
        float t2_red = block_reduce_sum_64(t2, sh_red);
        if (tid == 0) sh_t2 = t2_red;
        __syncthreads();

        /* ── prior term: 26 neighbour qGGMRF surrogate reduction ──────────
         * One thread per neighbour for tid < 26; others contribute 0. */
        float n1 = 0.0f, n2 = 0.0f;
        if (tid < 26) {
            int jx = ix + d_NB_DX[tid];
            int jy = iy + d_NB_DY[tid];
            int jz = iz + d_NB_DZ[tid];
            if (jx >= 0 && jx < Nx && jy >= 0 && jy < Ny && jz >= 0 && jz < Nz) {
                /* tex3D extent matches volume layout: x = Nx, y = Nz, z = Ny
                 * (see TC_VOL_3D in texture_cache.cu). +0.5 → texel centre. */
                float fk = tex3D<float>(vol_snapshot_tex,
                                        (float)jx + 0.5f,
                                        (float)jz + 0.5f,
                                        (float)jy + 0.5f);
                float a = qggmrf_a_jk_dev(fj_old - fk);
                float w = d_NB_W[tid];
                n1 = w * a * fk;
                n2 = w * a;
            }
        }
        float n1_red = block_reduce_sum_64(n1, sh_red);
        if (tid == 0) sh_n1 = n1_red;
        float n2_red = block_reduce_sum_64(n2, sh_red);
        if (tid == 0) sh_n2 = n2_red;
        __syncthreads();

        /* ── closed-form ICD update (thread 0) ─────────────────────────── */
        if (tid == 0) {
            float denom = sh_n2 + sh_t2;
            if (denom < 1e-30f) {
                sh_df = 0.0f;
            } else {
                float u_star = (sh_n1 + sh_t2 * fj_old - sh_t1) / denom;
                float fj_new = (u_star > 0.0f) ? u_star : 0.0f;
                vol_slice[(size_t)iz * Nx + ix] = fj_new;
                sh_df     = fj_new - fj_old;
            }
        }
        __syncthreads();
        float df = sh_df;
        if (df == 0.0f) continue;

        /* ── error-sinogram incremental update: e -= df · A_*j ──────────
         * Same stride pattern over angles as the data-term reduction —
         * guarantees no thread collision on detector pixels. */
        for (int ia = tid; ia < n_angles; ia += blockDim.x) {
            float ca = cos_a[ia], sa = sin_a[ia];
            float u  = x * ca + z * sa + mbir_H.half_det;
            int u0 = (int)floorf(u);
            int u1 = u0 + 1;
            float wu = u - u0;
            size_t base = ((size_t)ia * Ny + iy) * N_det;
            if (u0 >= 0 && u0 < N_det) e_sino[base + u0] -= df * (1.0f - wu);
            if (u1 >= 0 && u1 < N_det) e_sino[base + u1] -= df * wu;
        }
        __syncthreads();
    }

    /* `d_offsets` is read inside the residual computation only via `e_sino`
     * because the host maintains the invariant e = g - Af - d. The kernel
     * does not directly read `d_offsets` — kept in the signature so the
     * Python wrapper can pass it through and future variants (e.g. an
     * inline d_k closed-form update) can use it. */
    (void)d_offsets;
}

/* ────────────────────────────────────────────────────────────────────────────
 * Host wrappers
 * ──────────────────────────────────────────────────────────────────────────── */

void mbir_forward_project_cuda(
    const float *d_volume,
    float       *d_sinogram,
    const float *d_cos_a,
    const float *d_sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
) {
    ProjGeom2D g = proj_geom_2d(Ny, Nz, Nx, n_angles, N_det);

    /* Zero the destination sinogram — the scatter only adds positive
     * contributions, so explicit zeroing is required. */
    FASTTOMO_CUDA_CHECK(cudaMemsetAsync(
        d_sinogram, 0, (size_t)n_angles * Ny * N_det * sizeof(float), stream));

    dim3 block(16, 16, 1);
    dim3 grid((Nx + 15) / 16, (Nz + 15) / 16, Ny);
    mbir_forward_project_kernel<<<grid, block, 0, stream>>>(
        d_volume, d_sinogram, d_cos_a, d_sin_a,
        Ny, Nz, Nx, n_angles, N_det,
        g.half_x, g.half_z, g.half_det
    );
    CUDA_CHECK_KERNEL();
}

void mbir_icd_sweep_cuda(
    float       *d_volume,         /* [Ny, Nz, Nx] in-place */
    float       *d_e_sino,         /* [n_angles, Ny, N_det] in-place */
    const float *d_volume_snapshot,/* host uploads its own snapshot tex3D */
    const float *d_d_offsets,      /* [n_angles] read-only, may be nullptr */
    const float *d_lambdas,        /* [n_angles, Ny, N_det] or nullptr */
    const int   *d_perm,           /* [Ny, Nz*Nx] int32 */
    const float *d_cos_a,
    const float *d_sin_a,
    int Ny, int Nz, int Nx, int n_angles, int N_det,
    float qggmrf_p, float qggmrf_q, float qggmrf_c, float sigma_f,
    float huber_T, float huber_delta, float sigma,
    cudaStream_t stream
) {
    ProjGeom2D g = proj_geom_2d(Ny, Nz, Nx, n_angles, N_det);

    MbirHyper H{};
    H.p = qggmrf_p; H.q = qggmrf_q; H.c = qggmrf_c; H.sigma_f = sigma_f;
    H.huber_T = huber_T; H.huber_delta = huber_delta; H.sigma = sigma;
    H.half_x = g.half_x; H.half_z = g.half_z; H.half_det = g.half_det;
    FASTTOMO_CUDA_CHECK(cudaMemcpyToSymbolAsync(
        mbir_H, &H, sizeof(MbirHyper), 0, cudaMemcpyHostToDevice, stream));

    /* tex3D snapshot of `volume` for qGGMRF reads. Allocate, upload, bind,
     * launch, destroy — per outer iter. The cudaArray lifecycle uses the
     * existing TextureCache helpers for 3D fp32 single-channel volumes
     * (TC_VOL_3D dim_a=Ny, dim_b=Nz, dim_c=Nx). */
    TextureCache *snap = tc_create(TC_VOL_3D, TC_DTYPE_FP32, 1, Ny, Nz, Nx);
    tc_upload(snap, d_volume_snapshot, stream);

    dim3 block(64, 1, 1);
    dim3 grid(Ny, 1, 1);
    mbir_icd_sweep_kernel<<<grid, block, 0, stream>>>(
        snap->tex,
        d_volume, d_e_sino, d_d_offsets, d_lambdas, d_perm,
        d_cos_a, d_sin_a,
        Ny, Nz, Nx, n_angles, N_det
    );
    CUDA_CHECK_KERNEL();

    /* The kernel reads the texture asynchronously on this stream; tc_destroy
     * is stream-synchronous on the cudaArray free. Sync the stream first so
     * the kernel completes before we tear down the texture. */
    FASTTOMO_CUDA_CHECK(cudaStreamSynchronize(stream));
    tc_destroy(snap);
}

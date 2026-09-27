/* ── fasttomo/csrc/cuda/mbir_icd.cu ──────────────────────────────────────────
 * Classical MBIR (Model-Based Iterative Reconstruction) for single-axis
 * parallel-beam tomography.
 *
 * Reference: Venkatakrishnan et al. 2015, "MBIR for Bright-Field Electron
 *            Tomography", IEEE TCI 1(1). See fasttomo-wiki/wiki/algorithms/mbir.md
 *            for the project-side narrative and the surrogate-MM derivation.
 *
 * Kernels:
 *   forward_project_mbir_kernel - voxel-driven bilinear scatter FP (1 sample
 *       per voxel, 2 detector pixels per angle, atomicAdd). Used to compute
 *       the initial residual e = g - A·f and to validate against the CPU
 *       reference. Distinct from `forward_project_parallel_kernel` in projection_parallel.cu
 *       (which is ray-marching detector-driven); this voxel-driven model is
 *       what the ICD column-A access pattern reproduces.
 *   icd_sweep_mbir_kernel       - one outer iteration: per-Y-slice ICD walk
 *       through the slice's permutation, threads cooperate on the per-voxel
 *       data-term reduction over angles and qGGMRF-prior reduction over the
 *       26-neighbour stencil; thread 0 applies the closed-form update, all
 *       threads write the incremental error-sinogram delta non-atomically
 *       (correctness argument inline below).
 *
 * Host wrappers expose these as `mbir_forward_project_cuda` and
 * `mbir_icd_sweep_cuda`. The offset (d_k) and variance (σ) closed-form
 * updates live in the Python wrapper using PyTorch ops — see
 * fasttomo/algorithms/mbir.py::_run_mbir_single_scale.
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

/* ── Per-slice voxel ordering, generated in the kernel ───────────────────────
 * The ICD sweep visits the voxels of each Y-slice in a pseudo-random order
 * that changes every outer iteration (HANDOFF-mbir decision 4). The order is
 * a keyed bijection on [0, nvox): a 4-round Feistel network on 2h bits
 * (4^h >= nvox) with cycle-walking, so no [Ny, Nz*Nx] permutation has to be
 * built on the host and copied every outer iteration. The key is
 * `perm_key` (host: _mbir_perm_key(seed, it)) mixed with the slice index.
 * fasttomo/algorithms/mbir.py::_mbir_permutation is the bit-exact NumPy
 * mirror used by the CPU sweep; tests compare the two element for element. */
__host__ __device__ __forceinline__ unsigned long long mbir_mix64(unsigned long long z) {
    z += 0x9E3779B97F4A7C15ull;                     /* splitmix64 */
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

__device__ __forceinline__ unsigned int mbir_hash32(unsigned int x) {
    x ^= x >> 16; x *= 0x7feb352dU;                 /* lowbias32 */
    x ^= x >> 15; x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

struct MbirPerm {
    unsigned int rk[4];   /* round keys of this slice */
    unsigned int h;       /* half width in bits: 4^h >= nvox */
    unsigned int mask;    /* (1 << h) - 1 */
    unsigned int n;       /* nvox */
};

__device__ __forceinline__ MbirPerm mbir_perm_setup(unsigned long long key, int iy, int nvox) {
    MbirPerm P;
    unsigned long long k1 = mbir_mix64(key + (unsigned long long)iy);
    unsigned long long k2 = mbir_mix64(k1);
    P.rk[0] = (unsigned int)k1; P.rk[1] = (unsigned int)(k1 >> 32);
    P.rk[2] = (unsigned int)k2; P.rk[3] = (unsigned int)(k2 >> 32);
    unsigned int h = 0;
    while ((1ull << (2 * h)) < (unsigned long long)nvox) h++;
    P.h = h;
    P.mask = (1u << h) - 1u;
    P.n = (unsigned int)nvox;
    return P;
}

__device__ __forceinline__ unsigned int mbir_feistel(const MbirPerm &P, unsigned int x) {
    unsigned int L = x >> P.h, R = x & P.mask;
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        unsigned int F = mbir_hash32(R ^ P.rk[r]) & P.mask;
        unsigned int nl = R;
        R = L ^ F;
        L = nl;
    }
    return (L << P.h) | R;
}

/* s-th voxel of the slice (0 <= s < nvox). Cycle-walking terminates: the
 * Feistel map is a permutation of [0, 4^h) and s itself lies in [0, nvox). */
__device__ __forceinline__ int mbir_perm_index(const MbirPerm &P, int s) {
    unsigned int y = mbir_feistel(P, (unsigned int)s);
    while (y >= P.n) y = mbir_feistel(P, y);
    return (int)y;
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
__global__ static void forward_project_mbir_kernel(
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
 *   - All 26 qGGMRF neighbour reads, in-slice ones included, come from a
 *     tex3D snapshot of the start-of-outer-iter volume (read-only across
 *     the kernel). The prior term is therefore Jacobi within a slice as
 *     well as across Y; only the data term is Gauss-Seidel (through the
 *     live error sinogram). The CPU sweep does the same.
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
__global__ static void icd_sweep_mbir_kernel(
    cudaTextureObject_t       vol_snapshot_tex, /* tex3D over [Ny, Nz, Nx] read-only */
    float *__restrict__       volume,           /* [Ny, Nz, Nx] live (read+write) */
    float *__restrict__       e_sino,           /* [n_angles, Ny, N_det] live */
    const float *__restrict__ lambdas,          /* [n_angles, Ny, N_det] or nullptr */
    const int   *__restrict__ perm,             /* [Ny, Nz*Nx] or nullptr: in-kernel order */
    unsigned long long        perm_key,
    const float *__restrict__ cos_a,
    const float *__restrict__ sin_a,
    int Ny, int Nz, int Nx, int n_angles, int N_det
) {
    int iy  = blockIdx.x;
    int tid = threadIdx.x;
    if (iy >= Ny) return;

    const int nvox = Nz * Nx;
    const int *perm_slice = perm ? perm + (size_t)iy * nvox : nullptr;
    const MbirPerm P = mbir_perm_setup(perm_key, iy, nvox);
    float *vol_slice = volume + (size_t)iy * Nz * Nx;

    /* Shared: 2-element scratch for block reduction (one slot per warp);
     * + 5 broadcast scalars: theta1, theta2, n1, n2 and the update df. */
    __shared__ float sh_red[2];
    __shared__ float sh_t1, sh_t2, sh_n1, sh_n2, sh_df;

    for (int s = 0; s < nvox; s++) {
        int lin = perm_slice ? perm_slice[s] : mbir_perm_index(P, s);
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

    /* The per-tilt offsets d_k enter only through `e_sino`: the host keeps
     * the invariant e = g - Af - d, so the kernel never reads d_k. */
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
    ProjGeomParallel g = proj_geom_parallel(Ny, Nz, Nx, n_angles, N_det);

    /* Zero the destination sinogram — the scatter only adds positive
     * contributions, so explicit zeroing is required. */
    FASTTOMO_CUDA_CHECK(cudaMemsetAsync(
        d_sinogram, 0, (size_t)n_angles * Ny * N_det * sizeof(float), stream));

    dim3 block(16, 16, 1);
    dim3 grid((Nx + 15) / 16, (Nz + 15) / 16, Ny);
    forward_project_mbir_kernel<<<grid, block, 0, stream>>>(
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
    const float *d_lambdas,        /* [n_angles, Ny, N_det] or nullptr */
    const int   *d_perm,           /* [Ny, Nz*Nx] int32, or nullptr */
    unsigned long long perm_key,   /* in-kernel order when d_perm is null */
    const float *d_cos_a,
    const float *d_sin_a,
    int Ny, int Nz, int Nx, int n_angles, int N_det,
    float qggmrf_p, float qggmrf_q, float qggmrf_c, float sigma_f,
    float huber_T, float huber_delta, float sigma,
    cudaStream_t stream
) {
    ProjGeomParallel g = proj_geom_parallel(Ny, Nz, Nx, n_angles, N_det);

    MbirHyper H{};
    H.p = qggmrf_p; H.q = qggmrf_q; H.c = qggmrf_c; H.sigma_f = sigma_f;
    H.huber_T = huber_T; H.huber_delta = huber_delta; H.sigma = sigma;
    H.half_x = g.half_x; H.half_z = g.half_z; H.half_det = g.half_det;
    FASTTOMO_CUDA_CHECK(cudaMemcpyToSymbolAsync(
        mbir_H, &H, sizeof(MbirHyper), 0, cudaMemcpyHostToDevice, stream));

    /* tex3D snapshot of `volume` for qGGMRF reads: a one-shot TC_VOL_3D
     * cache (dim_a=Ny, dim_b=Nz, dim_c=Nx) per outer iteration.
     * ScopedTextureCache releases it on every path; finish() syncs the
     * stream first, so the kernel is done before the texture goes away. */
    ScopedTextureCache snap(TC_VOL_3D, TC_DTYPE_FP32, 1, Ny, Nz, Nx, stream);
    tc_upload(snap.get(), d_volume_snapshot, stream);

    dim3 block(64, 1, 1);
    dim3 grid(Ny, 1, 1);
    icd_sweep_mbir_kernel<<<grid, block, 0, stream>>>(
        snap.get()->tex,
        d_volume, d_e_sino, d_lambdas, d_perm, perm_key,
        d_cos_a, d_sin_a,
        Ny, Nz, Nx, n_angles, N_det
    );
    CUDA_CHECK_KERNEL();
    snap.finish();
}

/* Test hook: write the in-kernel voxel order of every slice, [Ny, nvox]. */
__global__ static void mbir_permutation_kernel(
    int *__restrict__ out, int Ny, int nvox, unsigned long long perm_key
) {
    int iy = blockIdx.y;
    int s  = blockIdx.x * blockDim.x + threadIdx.x;
    if (iy >= Ny || s >= nvox) return;
    const MbirPerm P = mbir_perm_setup(perm_key, iy, nvox);
    out[(size_t)iy * nvox + s] = mbir_perm_index(P, s);
}

void mbir_permutation_cuda(
    int *d_out, int Ny, int nvox, unsigned long long perm_key, cudaStream_t stream
) {
    if (Ny <= 0 || nvox <= 0) return;
    dim3 block(256, 1, 1);
    dim3 grid((nvox + 255) / 256, Ny, 1);
    mbir_permutation_kernel<<<grid, block, 0, stream>>>(d_out, Ny, nvox, perm_key);
    CUDA_CHECK_KERNEL();
}

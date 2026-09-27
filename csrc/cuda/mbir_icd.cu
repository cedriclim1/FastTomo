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
 *   mbir_sweep_prepare_kernel   - fully parallel part of an outer iteration:
 *       every voxel's qGGMRF prior sums, in natural voxel order.
 *   icd_sweep_mbir_kernel       - the serial sweep: one warp per Y-slice
 *       walks the order, reduces the data term over the angles with warp
 *       shuffles and applies the closed-form update; lanes write the
 *       error-sinogram rows they own non-atomically (argument inline below).
 *       icd_sweep_mbir_generic_kernel is the same for n_angles > 256.
 *   icd_sweep_mbir_group_kernel - the parallel sweep (default when it
 *       applies): one block per Y-slice updates conflict-free voxel groups
 *       concurrently, the slice's error sinogram in shared memory; same
 *       per-voxel arithmetic, so it equals the serial sweep on its order.
 *
 * Host wrappers expose these as `mbir_forward_project_cuda` and
 * `mbir_icd_sweep_cuda`. The offset (d_k) and variance (σ) closed-form
 * updates live in the Python wrapper using PyTorch ops — see
 * fasttomo/algorithms/mbir.py::_run_mbir_single_scale.
 */

#include "common.cuh"
#include <cuda_pipeline.h>
#include <cfloat>
#include <cmath>
#include <stdexcept>
#include "ops.h"
#include "../include/proj_geom.h"

/* ── Hyperparameters ─────────────────────────────────────────────────────────
 * Passed by value as a kernel argument (the argument bank is constant memory,
 * broadcast to every thread). A kernel argument instead of a module-global
 * __constant__ symbol keeps two sweeps on two streams from racing on it. */
struct MbirHyper {
    float p, q, c, sigma_f;        /* qGGMRF: 1 <= p <= q <= 2, c, σ_f */
    float inv_sf, inv_sf2;         /* 1/σ_f, 1/σ_f² (host) */
    float qmp, qm2;                /* q - p, q - 2 (host) */
    int   q_is_2;                  /* q == 2: the one-pow form */
    float huber_T, huber_delta;    /* Huber: T (threshold), δ (tail slope) */
    float sigma;                   /* current noise std */
    float half_x, half_z, half_det;
};

/* 26-neighbour stencil constants — same as in csrc/cpu/mbir_icd.c.
 * Indexing order: (dy, dz, dx) sweep over {-1,0,+1}^3 with the self entry
 * (0,0,0) skipped. Weights w_jk ∝ 1/dist, normalized to sum to 1 (face
 * 1/19.1040835, edge (1/√2)/19.1040835, corner (1/√3)/19.1040835; see the
 * CPU file). */
#define NB_FACE   0.0523448298f
#define NB_EDGE   0.0370133841f
#define NB_CORNER 0.0302213016f
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
    NB_CORNER, NB_EDGE, NB_CORNER, NB_EDGE, NB_FACE, NB_EDGE, NB_CORNER, NB_EDGE, NB_CORNER,
    NB_EDGE,   NB_FACE, NB_EDGE,   NB_FACE,          NB_FACE, NB_EDGE, NB_FACE,   NB_EDGE,
    NB_CORNER, NB_EDGE, NB_CORNER, NB_EDGE, NB_FACE, NB_EDGE, NB_CORNER, NB_EDGE, NB_CORNER
};

/* qGGMRF surrogate a_jk = ρ'(Δ)/Δ for ρ(Δ) = |Δ/σ_f|^q / (c + |Δ/σ_f|^(q-p))
 * (Venkatakrishnan et al. 2015): the derivation, the Δ → 0 limit and the
 * floors are in csrc/cpu/mbir_icd.c (qggmrf_a_jk), which evaluates the same
 * expressions. With t = |Δ|/σ_f and T = t^(q-p):
 *   q = 2: a = (2c + p T) / (σ_f² (c + T)²)                  (one powf)
 *   else : a = t^(q-2) (q c + p T) / (σ_f² (c + T)²)          (two)
 * powf, not __powf: the prepare kernel is bound by its stencil loads either
 * way (3.04 vs 3.18 ms at 256^3), and the accurate powf keeps a within an
 * ulp or two of the CPU's. */
__device__ __forceinline__ float qggmrf_a_jk_dev(float delta, const MbirHyper &H) {
    float t = fabsf(delta) * H.inv_sf;
    if (H.q_is_2) {
        t = fmaxf(t, FLT_MIN);
        const float T = powf(t, H.qmp);
        const float D = H.c + T;
        return (2.0f * H.c + H.p * T) * H.inv_sf2 / (D * D);
    }
    t = fmaxf(t, 1e-12f);
    const float T = powf(t, H.qmp);
    const float D = H.c + T;
    return powf(t, H.qm2) * (H.q * H.c + H.p * T) * H.inv_sf2 / (D * D);
}

/* Huber surrogate weight b(r): 1 inside threshold, δ·T/|r| outside.
 * Reduces to all-ones when T = ∞ (conventional MBIR). */
__device__ __forceinline__ float huber_surr_weight_dev(float r, const MbirHyper &H) {
    float ar = fabsf(r);
    if (!isfinite(H.huber_T) || ar < H.huber_T) return 1.0f;
    return H.huber_delta * H.huber_T / ar;
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

/* Detector coordinate of voxel (x, z) at a view, rounded op by op as written
 * (no FMA contraction), like the C code: every MBIR kernel (the forward
 * projection and all sweeps) gets the same u, hence the same footprint and
 * weights, whatever the compiler would fuse. */
__device__ __forceinline__ float mbir_det_u(float x, float z, float ca, float sa, float half_det) {
    return __fadd_rn(__fadd_rn(__fmul_rn(x, ca), __fmul_rn(z, sa)), half_det);
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
        float u = mbir_det_u(x, z, ca, sa, half_det);
        int u0 = (int)floorf(u);
        int u1 = u0 + 1;
        float wu = u - u0;
        size_t base = ((size_t)ia * Ny + iy) * N_det;
        if (u0 >= 0 && u0 < N_det) atomicAdd(&sinogram[base + u0], (1.0f - wu) * v);
        if (u1 >= 0 && u1 < N_det) atomicAdd(&sinogram[base + u1], wu * v);
    }
}

/* ────────────────────────────────────────────────────────────────────────────
 * One outer iteration of MBIR-AM-PE (CUDA): the ICD sweep, in two kernels.
 *
 * Semantics (unchanged): each Y-slice's voxels are updated one at a time in
 * the slice's order; every update sees all earlier updates of the slice
 * through the live error sinogram e (Gauss-Seidel data term). (The
 * parallel sweep further below updates several voxels at once only where
 * that gives exactly the same result.) All 26
 * qGGMRF neighbour values come from the start-of-outer-iteration snapshot,
 * so the prior is Jacobi within a slice as well as across Y. The CPU sweep
 * does the same.
 *
 * 1. mbir_sweep_prepare_kernel, one thread per voxel in natural order,
 *    fully parallel: prior[iy][j] = (n1_j, n2_j), the qGGMRF prior sums of
 *    voxel j from the snapshot and j's value before the sweep. The 26
 *    neighbour terms are summed in the order of the original 64-thread
 *    kernel's warp reduction, so the sums are bit-identical to computing
 *    them inside the sweep. (Until phase 1b this kernel ran in the visiting
 *    order and also stored j and f_j per step, 16 bytes per voxel; the
 *    stencil gathers were then uncoalesced: 3.2 ms instead of 1 ms at
 *    256^3. Natural order gives the same sums, bit for bit.)
 * 2. icd_sweep_mbir_kernel: the sequential part. One warp owns one Y-slice
 *    (grid Ny x 32 threads) and walks the slice's order: step s visits
 *    voxel j = order(s) (the in-kernel Feistel order, or the explicit
 *    array), reads its value f_j from the live volume and its prior sums
 *    from prior[iy][j]; per voxel it computes the data-term sums over the
 *    angles, all-reduces them across the warp, and applies the closed-form
 *    update. f_j read from the live volume is still the pre-sweep value:
 *    each order is a permutation, so a voxel is written only when it is
 *    visited.
 *
 * Ownership, which makes the non-atomic error-sinogram writes safe without
 * any barrier:
 *   - block iy is the only block that reads or writes e[*, iy, *] and
 *     volume[iy, *, *];
 *   - lane l is the only thread that reads or writes the rows e[ia, iy, *]
 *     with ia ≡ l (mod 32), for every voxel of the sweep, including both
 *     detector pixels (u0, u0+1) of such a row. A later voxel's read of a
 *     row therefore follows the earlier voxel's write in the same thread's
 *     program order; no other thread is involved;
 *   - the per-voxel sums are a warp all-reduce (xor butterfly). Every lane
 *     ends with bit-identical sums (each step adds the same two operands,
 *     a + b == b + a), so every lane computes the same update df on its
 *     own: no shared-memory broadcast. Lane 0 alone stores the voxel.
 *
 * Summation order: lane l sums over its angles l, l+64, ... (A) and l+32,
 * l+96, ... (B), adds A + B, and the butterfly reduces across the lanes
 * (mbir_data_term). Every floating-point operation of the per-voxel
 * arithmetic is pinned (__fadd_rn / __fmul_rn / __fdiv_rn, no FMA
 * contraction, the C code's rounding), so the serial and the parallel
 * sweep give bit-identical results on the same order on any compiler.
 *
 * Latency: for n_angles <= 32 * MBIR_MAX_SLOTS the sweep keeps each lane's
 * e values for the current voxel in registers (K = angle slots per lane):
 *   - the update stores e - df·a from those registers (no reload);
 *   - the next voxel's e values are loaded at the top of the iteration,
 *     before this voxel's stores, and patched at the top of the next
 *     iteration (when the loads have landed): a pixel this voxel wrote lies
 *     in a row owned by the same lane, which replaces its prefetched copy
 *     with the value it stored. The next voxel sees exactly what a reload
 *     after the store would return.
 * LAM / HUB select exact simplifications of the per-pixel weight for
 * lambdas == nullptr (lam = 1: sqrtf(1) = 1 and x·1 = x exactly) and
 * huber_T = ±inf or NaN (Huber weight 1). */
#define MBIR_WARP 32
#define MBIR_MAX_SLOTS 8

/* s-th voxel of the slice: the explicit order if given, else the in-kernel one. */
__device__ __forceinline__ int mbir_voxel(const int *perm_slice, const MbirPerm &P, int s) {
    return perm_slice ? perm_slice[s] : mbir_perm_index(P, s);
}

/* qGGMRF term of neighbour `nb` (0..25; anything else gives zero) of voxel
 * `lin` (value fj) of slice iy, from the snapshot. */
__device__ __forceinline__ void mbir_prior_term(
    const float *__restrict__ snap, const MbirHyper &H,
    int lin, float fj, int iy, int Ny, int Nz, int Nx, int nb,
    float &n1, float &n2
) {
    n1 = 0.0f; n2 = 0.0f;
    if (nb < 26) {
        int ix = lin % Nx, iz = lin / Nx;
        int jx = ix + d_NB_DX[nb];
        int jy = iy + d_NB_DY[nb];
        int jz = iz + d_NB_DZ[nb];
        if (jx >= 0 && jx < Nx && jy >= 0 && jy < Ny && jz >= 0 && jz < Nz) {
            float fk = __ldg(snap + ((size_t)jy * Nz + jz) * Nx + jx);
            float a = qggmrf_a_jk_dev(fj - fk, H);
            float w = d_NB_W[nb];
            n1 = w * a * fk;
            n2 = w * a;
        }
    }
}

/* One thread per voxel, natural order (coalesced stencil gathers).
 * prior is [Ny, Nz*Nx] float2: (n1, n2) of that voxel. */
__global__ static void mbir_sweep_prepare_kernel(
    float2 *__restrict__       prior,
    const float *__restrict__  volume,
    const float *__restrict__  snap,
    int Ny, int Nz, int Nx,
    const MbirHyper H
) {
    const int nvox = Nz * Nx;
    const size_t total = (size_t)Ny * nvox;
    for (size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; t < total;
         t += (size_t)gridDim.x * blockDim.x) {
        const int iy  = (int)(t / nvox);
        const int lin = (int)(t - (size_t)iy * nvox);
        const float fj = volume[t];
        /* The original kernel had lane nb compute neighbour nb (lanes 26..31
         * and the second warp contributed 0) and reduced with a shfl tree:
         * v[i] += v[i + off] for off = 16, 8, 4, 2, 1, then + 0 for the
         * second warp. Same terms, same order. */
        float v1[32], v2[32];
        #pragma unroll
        for (int nb = 0; nb < 32; nb++)
            mbir_prior_term(snap, H, lin, fj, iy, Ny, Nz, Nx, nb, v1[nb], v2[nb]);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            #pragma unroll
            for (int i = 0; i < off; i++) {
                v1[i] = v1[i] + v1[i + off];
                v2[i] = v2[i] + v2[i + off];
            }
        }
        prior[t] = make_float2(v1[0] + 0.0f, v2[0] + 0.0f);
    }
}

/* The record of step s of a slice: (voxel index as int bits, its value
 * before the sweep, n1, n2). The value comes from the live volume: the voxel
 * has not been visited yet (see the semantics above). */
__device__ __forceinline__ float4 mbir_fetch(
    const int *perm_slice, const MbirPerm &P, int s,
    const float *vol_slice, const float2 *__restrict__ prior_slice
) {
    const int lin = mbir_voxel(perm_slice, P, s);
    const float2 pr = __ldg(prior_slice + lin);
    return make_float4(__int_as_float(lin), vol_slice[lin], pr.x, pr.y);
}

/* The sequential kernel reads its records from a shared-memory ring of
 * MBIR_RING steps, refilled MBIR_WARP steps at a time between blocks of 32
 * steps: lane l computes the voxel index of step s0 + l in parallel with
 * the other lanes and copies the voxel's value and prior sums into the ring
 * with asynchronous global->shared copies (cp.async; synchronous below
 * sm_80), waited for 32 steps later. Neither the order computation nor the
 * two dependent loads sit on the per-voxel chain (computing them per step
 * there made the sweep 25-35 % slower), and the per-step loop has no refill
 * branch (one made it 10 % slower). A voxel's value read up to 96 steps
 * early is still its pre-sweep value: it is written only when visited, and
 * its copy has landed by then. */
#define MBIR_RING 128
__device__ __forceinline__ void mbir_ring_fill_async(
    float4 *ring, int s0, int nvox, int lane, const int *perm_slice, const MbirPerm &P,
    const float *vol_slice, const float2 *prior_slice
) {
    const int s = s0 + lane;
    if (s < nvox) {
        const int lin = mbir_voxel(perm_slice, P, s);
        float4 *slot = ring + (s & (MBIR_RING - 1));
        reinterpret_cast<int *>(slot)[0] = lin;
        __pipeline_memcpy_async(reinterpret_cast<float *>(slot) + 1, vol_slice + lin, 4);
        __pipeline_memcpy_async(reinterpret_cast<float2 *>(slot) + 1, prior_slice + lin, 8);
    }
    __pipeline_commit();
}

/* Per-pixel data-term contribution: w = b(r)·λ/σ², then
 * θ1 += a·e·w, θ2 += a²·w (θ1's sign is applied after the reduction),
 * rounded op by op like the C code (see mbir_det_u). */
template <bool LAM, bool HUB>
__device__ __forceinline__ void mbir_pixel_term(
    float a_col, float ek, float lam, const MbirHyper &H, float w_const,
    float &t1, float &t2
) {
    float w;
    if (HUB) {
        float r  = LAM ? ek * sqrtf(lam) / H.sigma : ek / H.sigma;
        float bw = huber_surr_weight_dev(r, H);
        w = LAM ? bw * lam / (H.sigma * H.sigma) : bw / (H.sigma * H.sigma);
    } else {
        w = LAM ? lam / (H.sigma * H.sigma) : w_const;
    }
    t1 = __fadd_rn(t1, __fmul_rn(__fmul_rn(a_col, ek), w));
    t2 = __fadd_rn(t2, __fmul_rn(__fmul_rn(a_col, a_col), w));
}

/* What lane `lane` keeps per angle slot for one voxel. */
struct MbirSlot {
    int   u0;          /* left detector pixel; u0 + 1 is the right one */
    float wu;          /* bilinear weight of u0 + 1 */
    float e0, e1;      /* e at u0, u0 + 1 (0 when off the detector) */
    float l0, l1;      /* lambdas at u0, u0 + 1 (1 without lambdas) */
};

template <int K, bool LAM>
__device__ __forceinline__ void mbir_load_slots(
    MbirSlot (&S)[K], const float (&ca)[K], const float (&sa)[K],
    int lin, int Nx, const MbirHyper &H, int lane, int n_angles, int N_det,
    const float *e_base, const float *__restrict__ lam_base, size_t row_stride
) {
    const int ix = lin % Nx, iz = lin / Nx;
    const float x = ix - H.half_x;
    const float z = iz - H.half_z;
    #pragma unroll
    for (int k = 0; k < K; k++) {
        const int ia = lane + k * MBIR_WARP;
        if (ia >= n_angles) {
            /* Inactive slot: never used (the data term and the store skip
             * it), but the patch and pe0/pe1 code read it before that
             * check, so give it defined values (#89 review). */
            S[k].u0 = 0;
            S[k].wu = S[k].e0 = S[k].e1 = 0.0f;
            S[k].l0 = S[k].l1 = 1.0f;
            continue;
        }
        float u  = mbir_det_u(x, z, ca[k], sa[k], H.half_det);
        int u0 = (int)floorf(u);
        S[k].u0 = u0;
        S[k].wu = u - u0;
        const size_t row = (size_t)ia * row_stride;
        const bool in0 = (u0 >= 0 && u0 < N_det);
        const bool in1 = (u0 + 1 >= 0 && u0 + 1 < N_det);
        S[k].e0 = in0 ? e_base[row + u0] : 0.0f;
        S[k].e1 = in1 ? e_base[row + u0 + 1] : 0.0f;
        if (LAM) {
            S[k].l0 = in0 ? __ldg(lam_base + row + u0) : 1.0f;
            S[k].l1 = in1 ? __ldg(lam_base + row + u0 + 1) : 1.0f;
        } else {
            S[k].l0 = S[k].l1 = 1.0f;
        }
    }
}

template <bool LAM, bool HUB>
__device__ __forceinline__ void mbir_slot_partial(
    const MbirSlot &S, int N_det, const MbirHyper &H, float w_const, float &t1, float &t2
) {
    if (S.u0 >= 0 && S.u0 < N_det)
        mbir_pixel_term<LAM, HUB>(1.0f - S.wu, S.e0, S.l0, H, w_const, t1, t2);
    if (S.u0 + 1 >= 0 && S.u0 + 1 < N_det)
        mbir_pixel_term<LAM, HUB>(S.wu, S.e1, S.l1, H, w_const, t1, t2);
}

/* The data term of one voxel from its K slots: lane l sums θ1 and θ2 over
 * its even slots (angles l, l + 64, ...) and its odd slots (l + 32, ...)
 * separately (two independent add chains), adds the two, and one xor
 * butterfly all-reduces θ1 and θ2 (every lane ends with bit-identical
 * values: each step adds the same two operands); then θ1 takes the paper's
 * sign. Shared by the serial and the parallel sweep, so both sum in the
 * same order. (Until phase 1b the even and odd sums were reduced
 * separately, four butterflies instead of two, to reproduce the original
 * 64-thread kernel's order.) */
template <int K, bool LAM, bool HUB>
__device__ __forceinline__ void mbir_data_term(
    const MbirSlot (&C)[K], int lane, int n_angles, int N_det, const MbirHyper &H,
    float w_const, float &theta1, float &theta2
) {
    const unsigned FULL = 0xffffffffu;
    float t1a = 0.0f, t2a = 0.0f, t1b = 0.0f, t2b = 0.0f;
    #pragma unroll
    for (int k = 0; k < K; k++) {
        if (lane + k * MBIR_WARP >= n_angles) continue;
        if ((k & 1) == 0) mbir_slot_partial<LAM, HUB>(C[k], N_det, H, w_const, t1a, t2a);
        else              mbir_slot_partial<LAM, HUB>(C[k], N_det, H, w_const, t1b, t2b);
    }
    float t1 = __fadd_rn(t1a, t1b), t2 = __fadd_rn(t2a, t2b);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        t1 = __fadd_rn(t1, __shfl_xor_sync(FULL, t1, off));
        t2 = __fadd_rn(t2, __shfl_xor_sync(FULL, t2, off));
    }
    theta1 = -t1;                       /* paper sign convention */
    theta2 = t2;
}

/* Closed-form ICD update u* = (n1 + θ2 f_j - θ1) / (n2 + θ2), clamped at 0.
 * Returns false (no update, df = 0) when the denominator vanishes. */
__device__ __forceinline__ bool mbir_update(
    float fj, float n1, float n2, float theta1, float theta2, float &fj_new, float &df
) {
    const float denom = __fadd_rn(n2, theta2);
    if (denom < 1e-30f) { df = 0.0f; return false; }
    const float u_star = __fdiv_rn(__fsub_rn(__fadd_rn(n1, __fmul_rn(theta2, fj)), theta1), denom);
    fj_new = (u_star > 0.0f) ? u_star : 0.0f;
    df = __fsub_rn(fj_new, fj);
    return true;
}

/* The new error values of a slot after the update: e - df·(1 - wu), e - df·wu. */
__device__ __forceinline__ void mbir_slot_new_e(const MbirSlot &C, float df, float &pe0, float &pe1) {
    pe0 = __fsub_rn(C.e0, __fmul_rn(df, __fsub_rn(1.0f, C.wu)));
    pe1 = __fsub_rn(C.e1, __fmul_rn(df, C.wu));
}

template <int K, bool LAM, bool HUB>
__global__ static void __launch_bounds__(MBIR_WARP) icd_sweep_mbir_kernel(
    const float2 *__restrict__ prior,            /* [Ny, Nz*Nx] from the prepare kernel */
    const int   *__restrict__  perm,             /* [Ny, Nz*Nx] explicit order, or null */
    unsigned long long         perm_key,         /* in-kernel order when perm is null */
    float *__restrict__        volume,           /* [Ny, Nz, Nx] live (write only here) */
    float *__restrict__        e_sino,           /* [n_angles, Ny, N_det] live */
    const float *__restrict__  lambdas,          /* [n_angles, Ny, N_det] (LAM) */
    const float *__restrict__  cos_a,
    const float *__restrict__  sin_a,
    int Ny, int Nz, int Nx, int n_angles, int N_det,
    const MbirHyper H
) {
    const int iy   = blockIdx.x;
    const int lane = threadIdx.x;
    const int nvox = Nz * Nx;
    if (iy >= Ny || nvox <= 0) return;                 /* block-uniform */

    float *vol_slice = volume + (size_t)iy * nvox;
    const float2 *prior_slice = prior + (size_t)iy * nvox;
    const int *perm_slice = perm ? perm + (size_t)iy * nvox : nullptr;
    const MbirPerm P = mbir_perm_setup(perm_key, iy, nvox);
    /* Row (ia, iy) of e / lambdas starts at (ia * Ny + iy) * N_det. */
    const size_t row_stride = (size_t)Ny * N_det;
    float *e_base = e_sino + (size_t)iy * N_det;
    const float *lam_base = LAM ? lambdas + (size_t)iy * N_det : nullptr;
    const float w_const = 1.0f / (H.sigma * H.sigma);   /* w for λ = 1, b = 1 */

    float ca[K], sa[K];
    #pragma unroll
    for (int k = 0; k < K; k++) {
        const int ia = lane + k * MBIR_WARP;
        ca[k] = (ia < n_angles) ? cos_a[ia] : 0.0f;
        sa[k] = (ia < n_angles) ? sin_a[ia] : 0.0f;
    }

    /* Pipeline: records from the ring two steps ahead, the next voxel's e
     * values one ahead. The steps run in blocks of 32; before block s0 the
     * warp waits for block s0 + 32 (copied during block s0 - 32) and starts
     * copying block s0 + 64 into the slots of block s0 - 64, whose last
     * read was at step s0 - 35. Block s0 reads steps up to s0 + 33. */
    __shared__ float4 ring[MBIR_RING];
    mbir_ring_fill_async(ring, 0, nvox, lane, perm_slice, P, vol_slice, prior_slice);
    mbir_ring_fill_async(ring, MBIR_WARP, nvox, lane, perm_slice, P, vol_slice, prior_slice);
    __pipeline_wait_prior(0);
    mbir_ring_fill_async(ring, 2 * MBIR_WARP, nvox, lane, perm_slice, P, vol_slice, prior_slice);
    __syncwarp();
    float4 R_c = ring[0];
    float4 R_n = (nvox > 1) ? ring[1] : R_c;
    MbirSlot C[K], N[K];
    mbir_load_slots<K, LAM>(C, ca, sa, __float_as_int(R_c.x), Nx, H, lane, n_angles, N_det,
                            e_base, lam_base, row_stride);
    /* Pending patch: the pixels the previous voxel stored (per slot: its u0
     * and the two new values). Applied to C one iteration after C's loads
     * were issued, so the warp never waits on those loads to overwrite them. */
    int   pu0[K];
    float pe0[K], pe1[K];
    bool  pending = false;                              /* warp-uniform */

    for (int s0 = 0; s0 < nvox; s0 += MBIR_WARP) {
      if (s0 > 0) {
        __pipeline_wait_prior(0);       /* this lane's copies of block s0 + 32 */
        __syncwarp();                   /* visible to the warp; old slots all read */
        mbir_ring_fill_async(ring, s0 + 2 * MBIR_WARP, nvox, lane, perm_slice, P, vol_slice,
                             prior_slice);
      }
      const int s_end = min(s0 + MBIR_WARP, nvox);
      for (int s = s0; s < s_end; s++) {
        const bool has_next = (s + 1 < nvox);           /* warp-uniform */
        if (has_next)
            mbir_load_slots<K, LAM>(N, ca, sa, __float_as_int(R_n.x), Nx, H, lane, n_angles,
                                    N_det, e_base, lam_base, row_stride);
        const float4 R_nn = (s + 2 < nvox) ? ring[(s + 2) & (MBIR_RING - 1)] : R_n;

        /* C was loaded before the previous voxel's stores: patch it. A pixel
         * the previous voxel wrote lies in a row this lane owns, so C then
         * holds exactly what a reload would return. Off-detector pixels are
         * never stored or read (the same u has the same status), so patching
         * them is harmless. */
        if (pending) {
            #pragma unroll
            for (int k = 0; k < K; k++) {
                const int d = C[k].u0 - pu0[k];
                if (d == 0)       { C[k].e0 = pe0[k]; C[k].e1 = pe1[k]; }
                else if (d == 1)  { C[k].e0 = pe1[k]; }
                else if (d == -1) { C[k].e1 = pe0[k]; }
            }
        }

        /* ── data term from registers, fused warp all-reduce ─────────── */
        float theta1, theta2;
        mbir_data_term<K, LAM, HUB>(C, lane, n_angles, N_det, H, w_const, theta1, theta2);

        /* ── closed-form ICD update (every lane, identical values) ─────── */
        float fj_new, df;
        if (mbir_update(R_c.y, R_c.z, R_c.w, theta1, theta2, fj_new, df) && lane == 0)
            vol_slice[__float_as_int(R_c.x)] = fj_new;

        /* ── e -= df · A_*j from registers; remember it for the patch ──── */
        pending = (df != 0.0f);                         /* warp-uniform */
        if (pending) {
            #pragma unroll
            for (int k = 0; k < K; k++) {
                const int ia = lane + k * MBIR_WARP;
                const int u0 = C[k].u0;
                pu0[k] = u0;
                mbir_slot_new_e(C[k], df, pe0[k], pe1[k]);
                if (ia >= n_angles) continue;
                float *e_row = e_base + (size_t)ia * row_stride;
                if (u0 >= 0 && u0 < N_det)         e_row[u0]     = pe0[k];
                if (u0 + 1 >= 0 && u0 + 1 < N_det) e_row[u0 + 1] = pe1[k];
            }
        }

        #pragma unroll
        for (int k = 0; k < K; k++) C[k] = N[k];
        R_c = R_n;
        R_n = R_nn;
      }
    }

    /* The per-tilt offsets d_k enter only through `e_sino`: the host keeps
     * the invariant e = g - Af - d, so the kernel never reads d_k. */
}

/* Same sweep for n_angles > 32 * MBIR_MAX_SLOTS: lane l walks its angles
 * l, l+32, ... directly in global memory (no register slots), with the
 * same ownership, order and arithmetic. */
__global__ static void __launch_bounds__(MBIR_WARP) icd_sweep_mbir_generic_kernel(
    const float2 *__restrict__ prior,
    const int   *__restrict__  perm,
    unsigned long long         perm_key,
    float *__restrict__        volume,
    float *__restrict__        e_sino,
    const float *__restrict__  lambdas,
    const float *__restrict__  cos_a,
    const float *__restrict__  sin_a,
    int Ny, int Nz, int Nx, int n_angles, int N_det,
    const MbirHyper H
) {
    const unsigned FULL = 0xffffffffu;
    const int iy   = blockIdx.x;
    const int lane = threadIdx.x;
    const int nvox = Nz * Nx;
    if (iy >= Ny || nvox <= 0) return;

    float *vol_slice = volume + (size_t)iy * nvox;
    const float2 *prior_slice = prior + (size_t)iy * nvox;
    const int *perm_slice = perm ? perm + (size_t)iy * nvox : nullptr;
    const MbirPerm P = mbir_perm_setup(perm_key, iy, nvox);
    const size_t row_stride = (size_t)Ny * N_det;
    float *e_base = e_sino + (size_t)iy * N_det;
    const float *lam_base = lambdas ? lambdas + (size_t)iy * N_det : nullptr;

    for (int s = 0; s < nvox; s++) {
        const float4 R = mbir_fetch(perm_slice, P, s, vol_slice, prior_slice);
        const int lin = __float_as_int(R.x);
        const float x = (lin % Nx) - H.half_x;
        const float z = (lin / Nx) - H.half_z;

        float t[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};   /* [A|B][θ1|θ2] */
        for (int ia = lane; ia < n_angles; ia += MBIR_WARP) {
            float u  = mbir_det_u(x, z, cos_a[ia], sin_a[ia], H.half_det);
            int u0 = (int)floorf(u);
            float wu = u - u0;
            const size_t row = (size_t)ia * row_stride;
            const int ab = (ia / MBIR_WARP) & 1;
            #pragma unroll
            for (int kk = 0; kk < 2; kk++) {
                int uk = u0 + kk;
                if (uk < 0 || uk >= N_det) continue;
                float a_col = (kk == 0) ? (1.0f - wu) : wu;
                float ek  = e_base[row + uk];
                float lam = lam_base ? lam_base[row + uk] : 1.0f;
                float r   = ek * sqrtf(lam) / H.sigma;
                float bw  = huber_surr_weight_dev(r, H);
                float w   = bw * lam / (H.sigma * H.sigma);
                t[ab][0] = __fadd_rn(t[ab][0], __fmul_rn(__fmul_rn(a_col, ek), w));
                t[ab][1] = __fadd_rn(t[ab][1], __fmul_rn(__fmul_rn(a_col, a_col), w));
            }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            t[0][0] = __fadd_rn(t[0][0], __shfl_xor_sync(FULL, t[0][0], off));
            t[1][0] = __fadd_rn(t[1][0], __shfl_xor_sync(FULL, t[1][0], off));
            t[0][1] = __fadd_rn(t[0][1], __shfl_xor_sync(FULL, t[0][1], off));
            t[1][1] = __fadd_rn(t[1][1], __shfl_xor_sync(FULL, t[1][1], off));
        }
        const float theta1 = -__fadd_rn(t[0][0], t[1][0]);
        const float theta2 = __fadd_rn(t[0][1], t[1][1]);

        float fj_new, df;
        if (mbir_update(R.y, R.z, R.w, theta1, theta2, fj_new, df) && lane == 0)
            vol_slice[lin] = fj_new;
        if (df != 0.0f) {
            for (int ia = lane; ia < n_angles; ia += MBIR_WARP) {
                float u  = mbir_det_u(x, z, cos_a[ia], sin_a[ia], H.half_det);
                int u0 = (int)floorf(u);
                int u1 = u0 + 1;
                float wu = u - u0;
                float *e_row = e_base + (size_t)ia * row_stride;
                if (u0 >= 0 && u0 < N_det)
                    e_row[u0] = __fsub_rn(e_row[u0], __fmul_rn(df, __fsub_rn(1.0f, wu)));
                if (u1 >= 0 && u1 < N_det)
                    e_row[u1] = __fsub_rn(e_row[u1], __fmul_rn(df, wu));
            }
        }
    }
}


/* ────────────────────────────────────────────────────────────────────────────
 * Parallel sweep: conflict-free voxel groups (phase 1b).
 *
 * Two voxels of one row (same iz) whose x differ by m·S have detector
 * coordinates differing by m·S·cos θ_k at view k. When S·min_k |cos θ_k|
 * ≥ 2 + ε (the host picks S, fasttomo.algorithms.mbir._mbir_group_spacing;
 * ε covers the fp32 rounding of u), their floors differ by at least 2, so
 * their footprints {u0, u0 + 1} are disjoint in every row e[k, iy, :]. The
 * prior reads only the snapshot (through the prior sums) and a voxel writes
 * only itself, so the members {(iz, r + m·S)} of a group can be updated at
 * the same time: the result is exactly that of updating them one after the
 * other in any order. Gauss-Seidel semantics are unchanged; only the
 * visiting order differs from the serial kernel's (groups in a keyed
 * Feistel order over the nG = Nz·S groups of the slice, group g = iz·S + r;
 * fasttomo.algorithms.mbir._mbir_group_permutation is the flattened NumPy
 * mirror). S = 5 at ±60° (cos ≥ 0.5), 6 at ±70°, 12 at ±80°.
 *
 * One block of NW warps per Y-slice; warp w updates members w, w + NW, ...
 * of the current group and __syncthreads separates groups. The slice's
 * error sinogram e[:, iy, :] stays in shared memory for the whole sweep
 * (row stride N_det + 1): with it in global memory every lane touches its
 * own row and the sweep is L2-bound (measured slower than the serial kernel
 * at 512^3). Per member a warp does exactly the serial kernel's arithmetic
 * (the same slot mapping, mbir_data_term, mbir_update, mbir_slot_new_e and
 * the pinned rounding), so the result equals the serial kernel run on the
 * flattened group order bit for bit (tested). Needs n_angles ≤ 256 and
 * n_angles·(N_det + 1)·4 bytes of shared memory per block
 * (mbir_group_sweep_supported); the host falls back to the serial kernel
 * otherwise (e.g. 512^3 at 61 views needs 125 KB). */
template <int K, bool LAM>
__device__ __forceinline__ void mbir_smem_slots(
    MbirSlot (&S)[K], const float (&ca)[K], const float (&sa)[K],
    float x, float z, const MbirHyper &H, int lane, int n_angles, int N_det,
    const float *es, int ld, const float *__restrict__ lam_base, size_t row_stride
) {
    #pragma unroll
    for (int k = 0; k < K; k++) {
        const int ia = lane + k * MBIR_WARP;
        if (ia >= n_angles) {
            S[k].u0 = 0;
            S[k].wu = S[k].e0 = S[k].e1 = 0.0f;
            S[k].l0 = S[k].l1 = 1.0f;
            continue;
        }
        float u  = mbir_det_u(x, z, ca[k], sa[k], H.half_det);
        int u0 = (int)floorf(u);
        S[k].u0 = u0;
        S[k].wu = u - u0;
        const bool in0 = (u0 >= 0 && u0 < N_det);
        const bool in1 = (u0 + 1 >= 0 && u0 + 1 < N_det);
        S[k].e0 = in0 ? es[ia * ld + u0] : 0.0f;
        S[k].e1 = in1 ? es[ia * ld + u0 + 1] : 0.0f;
        if (LAM) {
            const size_t row = (size_t)ia * row_stride;
            S[k].l0 = in0 ? __ldg(lam_base + row + u0) : 1.0f;
            S[k].l1 = in1 ? __ldg(lam_base + row + u0 + 1) : 1.0f;
        } else {
            S[k].l0 = S[k].l1 = 1.0f;
        }
    }
}

template <int K, bool LAM, bool HUB, int NW>
__global__ static void __launch_bounds__(NW * MBIR_WARP, 1) icd_sweep_mbir_group_kernel(
    const float2 *__restrict__ prior,            /* [Ny, Nz*Nx] from the prepare kernel */
    unsigned long long         perm_key,         /* group order */
    float *__restrict__        volume,           /* [Ny, Nz, Nx] live */
    float *__restrict__        e_sino,           /* [n_angles, Ny, N_det] live */
    const float *__restrict__  lambdas,          /* [n_angles, Ny, N_det] (LAM) */
    const float *__restrict__  cos_a,
    const float *__restrict__  sin_a,
    int Ny, int Nz, int Nx, int n_angles, int N_det, int S,
    const MbirHyper H
) {
    extern __shared__ float es[];                      /* [n_angles][N_det + 1] */
    const int iy   = blockIdx.x;
    const int lane = threadIdx.x & (MBIR_WARP - 1);
    const int w    = threadIdx.x / MBIR_WARP;
    const int nvox = Nz * Nx;
    if (iy >= Ny || nvox <= 0) return;                 /* block-uniform */

    const int ld = N_det + 1;
    float *vol_slice = volume + (size_t)iy * nvox;
    const float2 *prior_slice = prior + (size_t)iy * nvox;
    const size_t row_stride = (size_t)Ny * N_det;
    float *e_base = e_sino + (size_t)iy * N_det;
    const float *lam_base = LAM ? lambdas + (size_t)iy * N_det : nullptr;
    const float w_const = 1.0f / (H.sigma * H.sigma);

    for (int ia = w; ia < n_angles; ia += NW)          /* stage e[:, iy, :] */
        for (int u = lane; u < N_det; u += MBIR_WARP)
            es[ia * ld + u] = e_base[(size_t)ia * row_stride + u];

    float ca[K], sa[K];
    #pragma unroll
    for (int k = 0; k < K; k++) {
        const int ia = lane + k * MBIR_WARP;
        ca[k] = (ia < n_angles) ? cos_a[ia] : 0.0f;
        sa[k] = (ia < n_angles) ? sin_a[ia] : 0.0f;
    }
    const int nG = Nz * S;
    const MbirPerm P = mbir_perm_setup(perm_key, iy, nG);
    const int step = NW * S;
    __syncthreads();

    /* This warp's first member of the next group is prefetched (its value
     * and prior sums) before the barrier; members of other groups are other
     * voxels, so the value read early is still the pre-sweep one. */
    int g_next = mbir_perm_index(P, 0);
    int iz_n = g_next / S, ix_n = g_next - iz_n * S + w * S;
    float fj_n = 0.0f;
    float2 pr_n = make_float2(0.0f, 0.0f);
    if (ix_n < Nx) {
        fj_n = vol_slice[iz_n * Nx + ix_n];
        pr_n = __ldg(prior_slice + iz_n * Nx + ix_n);
    }
    for (int gi = 0; gi < nG; gi++) {
        const int iz = iz_n, ix0 = ix_n;
        float fj0 = fj_n;
        float2 pr0 = pr_n;
        if (gi + 1 < nG) {
            g_next = mbir_perm_index(P, gi + 1);
            iz_n = g_next / S;
            ix_n = g_next - iz_n * S + w * S;
            if (ix_n < Nx) {
                fj_n = vol_slice[iz_n * Nx + ix_n];
                pr_n = __ldg(prior_slice + iz_n * Nx + ix_n);
            }
        }
        const float z = iz - H.half_z;
        for (int ix = ix0; ix < Nx; ix += step) {       /* warp-uniform */
            const int lin = iz * Nx + ix;
            float fj;
            float2 pr;
            if (ix == ix0) { fj = fj0; pr = pr0; }
            else           { fj = vol_slice[lin]; pr = __ldg(prior_slice + lin); }
            const float x = ix - H.half_x;
            MbirSlot C[K];
            mbir_smem_slots<K, LAM>(C, ca, sa, x, z, H, lane, n_angles, N_det, es, ld,
                                    lam_base, row_stride);
            float theta1, theta2;
            mbir_data_term<K, LAM, HUB>(C, lane, n_angles, N_det, H, w_const, theta1, theta2);
            float fj_new, df;
            if (mbir_update(fj, pr.x, pr.y, theta1, theta2, fj_new, df) && lane == 0)
                vol_slice[lin] = fj_new;
            if (df != 0.0f) {                           /* warp-uniform */
                #pragma unroll
                for (int k = 0; k < K; k++) {
                    const int ia = lane + k * MBIR_WARP;
                    if (ia >= n_angles) continue;
                    float pe0, pe1;
                    mbir_slot_new_e(C[k], df, pe0, pe1);
                    const int u0 = C[k].u0;
                    if (u0 >= 0 && u0 < N_det)         es[ia * ld + u0]     = pe0;
                    if (u0 + 1 >= 0 && u0 + 1 < N_det) es[ia * ld + u0 + 1] = pe1;
                }
            }
        }
        __syncthreads();
    }

    for (int ia = w; ia < n_angles; ia += NW)          /* write e back */
        for (int u = lane; u < N_det; u += MBIR_WARP)
            e_base[(size_t)ia * row_stride + u] = es[ia * ld + u];
}

static size_t mbir_group_smem_bytes(int n_angles, int N_det) {
    return (size_t)n_angles * (size_t)(N_det + 1) * sizeof(float);
}

bool mbir_group_sweep_supported(int n_angles, int N_det) {
    if (n_angles <= 0 || n_angles > MBIR_WARP * MBIR_MAX_SLOTS || N_det <= 0) return false;
    int dev = 0, optin = 0;
    FASTTOMO_CUDA_CHECK(cudaGetDevice(&dev));
    FASTTOMO_CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    return mbir_group_smem_bytes(n_angles, N_det) <= (size_t)optin;
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
    const float *d_volume_snapshot,/* [Ny, Nz, Nx] start-of-outer-iter copy */
    const float *d_lambdas,        /* [n_angles, Ny, N_det] or nullptr */
    const int   *d_perm,           /* [Ny, Nz*Nx] int32, or nullptr */
    unsigned long long perm_key,   /* in-kernel order when d_perm is null */
    float       *d_work,           /* [Ny, Nz*Nx, 2] fp32 scratch */
    const float *d_cos_a,
    const float *d_sin_a,
    int Ny, int Nz, int Nx, int n_angles, int N_det,
    float qggmrf_p, float qggmrf_q, float qggmrf_c, float sigma_f,
    float huber_T, float huber_delta, float sigma,
    int group_spacing,             /* 0: serial kernel; S > 0: group kernel */
    cudaStream_t stream
) {
    if (d_work == nullptr)
        throw std::invalid_argument("mbir_icd_sweep_cuda: work buffer is required");
    if (group_spacing < 0)
        throw std::invalid_argument("mbir_icd_sweep_cuda: group_spacing must be >= 0");
    if (group_spacing > 0) {
        if (d_perm != nullptr)
            throw std::invalid_argument(
                "mbir_icd_sweep_cuda: the parallel (group) sweep builds its own order; "
                "an explicit perm needs the serial sweep (group_spacing = 0)");
        if (!mbir_group_sweep_supported(n_angles, N_det))
            throw std::invalid_argument(
                "mbir_icd_sweep_cuda: the parallel (group) sweep needs n_angles <= 256 and "
                "n_angles * (N_det + 1) * 4 bytes of shared memory per block");
    }
    if (Ny <= 0 || Nz <= 0 || Nx <= 0) return;
    ProjGeomParallel g = proj_geom_parallel(Ny, Nz, Nx, n_angles, N_det);

    MbirHyper H{};
    H.p = qggmrf_p; H.q = qggmrf_q; H.c = qggmrf_c; H.sigma_f = sigma_f;
    H.inv_sf = 1.0f / sigma_f;               /* same host arithmetic as the CPU */
    H.inv_sf2 = 1.0f / (sigma_f * sigma_f);  /* qggmrf_params()                  */
    H.qmp = qggmrf_q - qggmrf_p;
    H.qm2 = qggmrf_q - 2.0f;
    H.q_is_2 = (qggmrf_q == 2.0f);
    H.huber_T = huber_T; H.huber_delta = huber_delta; H.sigma = sigma;
    H.half_x = g.half_x; H.half_z = g.half_z; H.half_det = g.half_det;

    /* 1. Prior sums of every voxel, natural order: a grid-stride loop over
     *    one full wave of resident 256-thread blocks. */
    float2 *prior = reinterpret_cast<float2 *>(d_work);
    int n_sm = 0, dev = 0, per_sm = 0;
    FASTTOMO_CUDA_CHECK(cudaGetDevice(&dev));
    FASTTOMO_CUDA_CHECK(cudaDeviceGetAttribute(&n_sm, cudaDevAttrMultiProcessorCount, dev));
    FASTTOMO_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &per_sm, mbir_sweep_prepare_kernel, 256, 0));
    const size_t total = (size_t)Ny * Nz * Nx;
    const size_t want = (total + 255) / 256;
    const size_t wave = (size_t)n_sm * (per_sm > 0 ? per_sm : 1);
    const int prep_blocks = (int)(want < wave ? want : wave);
    mbir_sweep_prepare_kernel<<<prep_blocks, 256, 0, stream>>>(
        prior, d_volume, d_volume_snapshot, Ny, Nz, Nx, H);
    CUDA_CHECK_KERNEL();

    const int slots = (n_angles + MBIR_WARP - 1) / MBIR_WARP;
    const int K = slots <= 1 ? 1 : slots <= 2 ? 2 : slots <= 4 ? 4 : slots <= 8 ? 8 : 0;
    const bool lam = d_lambdas != nullptr;
    const bool hub = std::isfinite(huber_T);

    if (group_spacing > 0) {
        /* 2'. The parallel sweep: one block per Y-slice of 32 warps for
         *     K <= 2 angle slots per lane, 16 for K = 4 and 8 for K = 8
         *     (register budgets of 64 / 128 / 255 per thread). */
        const size_t smem = mbir_group_smem_bytes(n_angles, N_det);
#define MBIR_GROUP(KK, LL, HH, WW) do { \
        auto kern = icd_sweep_mbir_group_kernel<KK, LL, HH, WW>; \
        FASTTOMO_CUDA_CHECK(cudaFuncSetAttribute( \
            kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem)); \
        kern<<<Ny, WW * MBIR_WARP, smem, stream>>>( \
            prior, perm_key, d_volume, d_e_sino, d_lambdas, d_cos_a, d_sin_a, \
            Ny, Nz, Nx, n_angles, N_det, group_spacing, H); } while (0)
#define MBIR_GROUP_K(KK, WW) \
        if (lam) { if (hub) MBIR_GROUP(KK, true, true, WW);  else MBIR_GROUP(KK, true, false, WW); } \
        else     { if (hub) MBIR_GROUP(KK, false, true, WW); else MBIR_GROUP(KK, false, false, WW); }
        switch (K) {
        case 1: MBIR_GROUP_K(1, 32); break;
        case 2: MBIR_GROUP_K(2, 32); break;
        case 4: MBIR_GROUP_K(4, 16); break;
        default: MBIR_GROUP_K(8, 8); break;          /* K == 8 (supported() checked) */
        }
#undef MBIR_GROUP_K
#undef MBIR_GROUP
        CUDA_CHECK_KERNEL();
        return;
    }

    /* 2. The serial sweep: one warp per Y-slice, all Ny blocks resident at
     *    once up to Ny = 24 * #SMs on sm_120 (24 blocks per SM is the block
     *    limit; registers allow 9 or more per SM for every variant below). */
    dim3 block(MBIR_WARP, 1, 1);
    dim3 grid(Ny, 1, 1);
#define MBIR_SWEEP(KK, LL, HH) icd_sweep_mbir_kernel<KK, LL, HH><<<grid, block, 0, stream>>>( \
        prior, d_perm, perm_key, d_volume, d_e_sino, d_lambdas, d_cos_a, d_sin_a, Ny, Nz, Nx, n_angles, N_det, H)
#define MBIR_SWEEP_K(KK) \
    if (lam) { if (hub) MBIR_SWEEP(KK, true, true);  else MBIR_SWEEP(KK, true, false); } \
    else     { if (hub) MBIR_SWEEP(KK, false, true); else MBIR_SWEEP(KK, false, false); }
    switch (K) {
    case 1: MBIR_SWEEP_K(1); break;
    case 2: MBIR_SWEEP_K(2); break;
    case 4: MBIR_SWEEP_K(4); break;
    case 8: MBIR_SWEEP_K(8); break;
    default:
        icd_sweep_mbir_generic_kernel<<<grid, block, 0, stream>>>(
            prior, d_perm, perm_key, d_volume, d_e_sino, d_lambdas, d_cos_a, d_sin_a,
            Ny, Nz, Nx, n_angles, N_det, H);
        break;
    }
#undef MBIR_SWEEP_K
#undef MBIR_SWEEP
    CUDA_CHECK_KERNEL();
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

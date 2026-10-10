/* ── fasttomo/csrc/cpu/mbir_icd.c ───────────────────────────────────────────
 * Classical MBIR (Model-Based Iterative Reconstruction) for single-axis
 * parallel-beam tomography. CPU reference implementation: simple,
 * single-threaded-per-Y-slice, mirrors the CUDA kernel one-for-one so they
 * cross-validate in `tests/test_parity.py::test_mbir_cpu_cuda_consistency`.
 *
 * Reference: Venkatakrishnan et al. 2015, "MBIR for Bright-Field Electron
 * Tomography", IEEE TCI 1(1).
 *
 * Provided entry points (mirroring `csrc/cpu/ops.h`):
 *   mbir_forward_project_cpu - voxel-driven bilinear FP (no super-
 *       sampling) used inside MBIR. Distinct from the existing ray-marching
 *       `forward_project_parallel_cpu` so the column-A model used in the ICD updates
 *       below matches the FP used to compute the initial e = g - Af.
 *   mbir_icd_sweep_cpu - the ICD sweep of one outer iteration over all
 *       voxels (Y-parallel via OpenMP). Mutates volume and e_sino in place.
 *       The per-tilt offset and noise-variance updates are not done here:
 *       both backends call the one Python implementation,
 *       `fasttomo.algorithms._mbir_update_offset_variance`.
 *
 * Forward operator A used internally:
 *   For voxel (ix, iz, iy) at angle θ_k, the detector coordinate is
 *       u = (ix - half_x)·cos θ_k + (iz - half_z)·sin θ_k + half_det
 *   and the voxel scatters to the two nearest detector pixels at iy with
 *   bilinear weights (1-wu, wu) where wu = u - floor(u). This is the
 *   standard "linear-interp / strip-integral" MBIR forward model and is
 *   the transpose-adjoint of the existing scalar-angle voxel-driven BP up
 *   to the lack of ray-step accumulation.
 */

#include "ops.h"
#include "../include/proj_geom.h"
#include <float.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>

/* ── 26-neighbor stencil (precomputed, normalized) ───────────────────────────
 * Weights w_jk ∝ 1/dist(j,k), normalized so Σ w_jk = 1.
 *   6 face-neighbors at dist 1     → unnormalized 1
 *   12 edge-neighbors at dist √2   → unnormalized 1/√2 ≈ 0.7071
 *   8 corner-neighbors at dist √3  → unnormalized 1/√3 ≈ 0.5774
 *   Sum = 6 + 12/√2 + 8/√3 ≈ 19.1040835
 * so face 1/19.1040835 = 0.0523448298, edge 0.0370133841, corner
 * 0.0302213016 (the literals below are these values to 9 digits; the
 * corner weight used to read 0.030219, which left Σ w = 0.99999).
 * Index layout matches the (dx, dy, dz) triples below: 26 entries in
 * the order produced by iterating (dy, dz, dx) ∈ {-1,0,1}^3 with the
 * (0,0,0) self entry skipped. */
#define NB_FACE   0.0523448298f
#define NB_EDGE   0.0370133841f
#define NB_CORNER 0.0302213016f
static const int NB_DX[26] = {
    -1,  0,  1, -1,  0,  1, -1,  0,  1,
    -1,  0,  1, -1,      1, -1,  0,  1,
    -1,  0,  1, -1,  0,  1, -1,  0,  1
};
static const int NB_DZ[26] = {
    -1, -1, -1,  0,  0,  0,  1,  1,  1,
    -1, -1, -1,  0,      0,  1,  1,  1,
    -1, -1, -1,  0,  0,  0,  1,  1,  1
};
static const int NB_DY[26] = {
    -1, -1, -1, -1, -1, -1, -1, -1, -1,
     0,  0,  0,  0,      0,  0,  0,  0,
     1,  1,  1,  1,  1,  1,  1,  1,  1
};
static const float NB_W[26] = {
    /* dy=-1 plane: 4 corners (√3), 4 edges (√2), 1 face (1) */
    NB_CORNER, NB_EDGE, NB_CORNER, NB_EDGE, NB_FACE, NB_EDGE, NB_CORNER, NB_EDGE, NB_CORNER,
    /* dy=0 plane: 4 edges (√2), 4 faces (1) — center self excluded */
    NB_EDGE,   NB_FACE, NB_EDGE,   NB_FACE,          NB_FACE, NB_EDGE, NB_FACE,   NB_EDGE,
    /* dy=+1 plane: 4 corners, 4 edges, 1 face */
    NB_CORNER, NB_EDGE, NB_CORNER, NB_EDGE, NB_FACE, NB_EDGE, NB_CORNER, NB_EDGE, NB_CORNER
};

/* ── qGGMRF prior ────────────────────────────────────────────────────────────
 * Potential (Venkatakrishnan et al. 2015, IEEE TCI 1(1), after eq. (9); the
 * qGGMRF of Thibault et al. 2007 in the normalisation of OpenMBIR-TEM):
 *
 *   ρ(Δ) = |Δ/σ_f|^q / (c + |Δ/σ_f|^(q-p)),        1 ≤ p ≤ q ≤ 2
 *
 * q is the exponent near Δ = 0 (q = 2: quadratic, smooth at 0) and p the
 * exponent for large |Δ| (p = 1.2 in the paper: edge preserving); c sets the
 * transition, |Δ| ≈ c^(1/(q-p))·σ_f (1.8e-4·σ_f for the defaults
 * p = 1.2, q = 2, c = 1e-3). Near 0, ρ ≈ (Δ/σ_f)²/c: 1/c times stiffer than
 * (Δ/σ_f)², which is what σ_f has to be tuned against.
 *
 * ICD uses the symmetric-bound surrogate a·Δ²/2 with a = ρ'(Δ)/Δ.
 * Derivation, with t = |Δ|/σ_f and T = t^(q-p):
 *
 *   dρ/dt   = [q t^(q-1) (c + T) - t^q (q-p) t^(q-p-1)] / (c + T)²
 *           = t^(q-1) (q c + p T) / (c + T)²
 *   ρ'(Δ)   = sign(Δ) · (1/σ_f) · dρ/dt,   Δ = sign(Δ) · σ_f · t
 *   a(Δ)    = ρ'(Δ)/Δ = t^(q-2) (q c + p T) / (σ_f² (c + T)²)
 *
 * q = 2 (the paper, and the default): a = (2c + p T) / (σ_f² (c + T)²) with
 * T = t^(2-p), one pow per neighbour. The limit at Δ → 0 is smooth:
 * a → 2/(σ_f² c) for p < 2 (T → 0; OpenMBIR's second derivative at zero) and
 * a = 2/(σ_f² (1 + c)) for p = q = 2 (a quadratic prior). For q < 2 the
 * factor t^(q-2) diverges at 0 (ρ is not smooth there). t is floored at
 * FLT_MIN for q = 2 (so t^0 = 1 when p = 2 and no pow sees 0 with a
 * negative exponent) and at 1e-12 otherwise.
 *
 * a is non-increasing in |Δ| for p ≤ q ≤ 2 (for q = 2: da/dT ∝
 * (p - 4) c - p T < 0), so a·Δ²/2 majorizes ρ (Bouman-Sauer surrogate).
 *
 * csrc/cuda/mbir_icd.cu::qggmrf_a_jk_dev evaluates the same expressions. */
typedef struct {
    float p, q, c;
    float inv_sf2;     /* 1/σ_f² */
    float inv_sf;      /* 1/σ_f */
    float qmp;         /* q - p */
    float qm2;         /* q - 2 */
    int   q_is_2;
} QggmrfParams;

static QggmrfParams qggmrf_params(float p, float q, float c, float sigma_f) {
    QggmrfParams P;
    P.p = p; P.q = q; P.c = c;
    P.inv_sf = 1.0f / sigma_f;
    P.inv_sf2 = 1.0f / (sigma_f * sigma_f);
    P.qmp = q - p;
    P.qm2 = q - 2.0f;
    P.q_is_2 = (q == 2.0f);
    return P;
}

static inline float qggmrf_a_jk(float delta, const QggmrfParams *P) {
    float t = fabsf(delta) * P->inv_sf;
    if (P->q_is_2) {
        if (t < FLT_MIN) t = FLT_MIN;
        float T = powf(t, P->qmp);
        float D = P->c + T;
        return (2.0f * P->c + P->p * T) * P->inv_sf2 / (D * D);
    }
    if (t < 1e-12f) t = 1e-12f;
    float T = powf(t, P->qmp);
    float D = P->c + T;
    return powf(t, P->qm2) * (P->q * P->c + P->p * T) * P->inv_sf2 / (D * D);
}

/* Generalized-Huber surrogate weight b(r) such that the upper-bound
 * quadratic surrogate Q(r; r') = b · r² · ½ + linear correction touches
 * β at r' and is tangent. For β'_T,δ:
 *   |r| ≤ T: β(r) = r²        → b = 1
 *   |r| > T: β'(r) = 2δT·sign(r) → surrogate slope = β'(r')/r' = 2δT/|r'|
 * Equivalent to "iteratively reweighted least squares". */
static inline float huber_surr_weight(float r, float T, float delta) {
    float ar = fabsf(r);
    if (!isfinite(T) || ar < T) return 1.0f;
    return delta * T / ar;
}

/* ────────────────────────────────────────────────────────────────────────────
 * Voxel-driven bilinear forward projection (MBIR's column-A model).
 *
 * This is the FP that matches the column-A used in the ICD update step.
 * Each voxel scatters to 2 detector pixels per angle with bilinear weights.
 * Output sinogram is written, not accumulated — caller need not zero it.
 *
 * Parallel-safe: writes to disjoint per-angle, per-iy strips of sinogram
 * because we iterate the angle/iy outer loops and accumulate via a local
 * partial-sum then write once. This matches the existing
 * `forward_project_parallel_cpu` parallelism shape. */
void mbir_forward_project_cpu(
    const float *volume,        /* [Ny, Nz, Nx] */
    float       *sinogram,      /* [n_angles, Ny, N_det] — overwritten */
    const float *cos_a,
    const float *sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    int num_threads
) {
    if (num_threads < 1) num_threads = 1;
    ProjGeomParallel g = proj_geom_parallel(Ny, Nz, Nx, n_angles, N_det);

    /* Zero sinogram up-front; the voxel scatter only adds positive
     * contributions so memset suffices. */
    memset(sinogram, 0, (size_t)n_angles * Ny * N_det * sizeof(float));

    /* Per-(ia, iy) accumulation into a stack buffer avoids cross-thread
     * race conditions on the output sinogram. */
    #pragma omp parallel for collapse(2) schedule(static) num_threads(num_threads)
    for (int ia = 0; ia < n_angles; ia++) {
        for (int iy = 0; iy < Ny; iy++) {
            float ca = cos_a[ia], sa = sin_a[ia];
            float *sino_row = sinogram + ((size_t)ia * Ny + iy) * N_det;
            const float *vol_slice = volume + (size_t)iy * Nz * Nx;

            for (int iz = 0; iz < Nz; iz++) {
                float z = iz - g.half_z;
                for (int ix = 0; ix < Nx; ix++) {
                    float v = vol_slice[(size_t)iz * Nx + ix];
                    if (v == 0.0f) continue;
                    float x = ix - g.half_x;
                    float u = x * ca + z * sa + g.half_det;
                    int u0 = (int)floorf(u);
                    int u1 = u0 + 1;
                    float wu = u - u0;
                    if (u0 >= 0 && u0 < N_det) sino_row[u0] += (1.0f - wu) * v;
                    if (u1 >= 0 && u1 < N_det) sino_row[u1] += wu * v;
                }
            }
        }
    }
}

/* ────────────────────────────────────────────────────────────────────────────
 * ICD sweep of one outer iteration of MBIR-AM-PE: for each Y-slice (OpenMP
 * across slices), walk the slice's permutation; for each voxel compute the
 * closed-form update from the data term (Huber-surrogate weighted) and the
 * qGGMRF prior surrogate; apply it with positivity; update the error
 * sinogram incrementally. The per-tilt offset d_k and the noise variance σ
 * are then updated by the caller (`_mbir_update_offset_variance`, shared
 * with the CUDA path).
 *
 * The qGGMRF prior reads all 26 neighbours, in-slice ones included, from
 * `volume_snapshot` (start-of-outer-iter copy), not the live volume: the
 * prior is Jacobi within a slice as well as across Y-slices. The data term
 * uses the live (in-iteration) error sinogram, so it stays Gauss-Seidel ICD.
 *
 * In-place mutated outputs: volume, e_sino. Returns 0 on success. */
int mbir_icd_sweep_cpu(
    float       *volume,           /* [Ny, Nz, Nx] in-place */
    float       *e_sino,           /* [n_angles, Ny, N_det] in-place: e = g - Af - d */
    const float *volume_snapshot,  /* [Ny, Nz, Nx] read-only snapshot for qGGMRF */
    const int   *perm,             /* [Ny, Nz*Nx] per-slice permutation */
    const float *cos_a,            /* [n_angles] */
    const float *sin_a,
    const float *lambdas,          /* [n_angles, Ny, N_det] or NULL → uniform Λ = I */
    int Ny, int Nz, int Nx, int n_angles, int N_det,
    float qggmrf_p, float qggmrf_q, float qggmrf_c, float sigma_f,
    float huber_T, float huber_delta, float sigma,
    int num_threads
) {
    if (num_threads < 1) num_threads = 1;
    ProjGeomParallel g = proj_geom_parallel(Ny, Nz, Nx, n_angles, N_det);
    const float half_x = g.half_x, half_z = g.half_z, half_det = g.half_det;
    const QggmrfParams QP = qggmrf_params(qggmrf_p, qggmrf_q, qggmrf_c, sigma_f);

    /* ── ICD sweep ─────────────────────────────────────────────────────────
     * Every qGGMRF neighbour read (in-slice and cross-slice) comes from the
     * snapshot, so slices are independent and parallelizable. Within a
     * slice the sweep is sequential (Gauss-Seidel via the incremental
     * error sino). */
    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int iy = 0; iy < Ny; iy++) {
        const int nvox_slice = Nz * Nx;
        const int *perm_slice = perm + (size_t)iy * nvox_slice;
        float *vol_slice = volume + (size_t)iy * Nz * Nx;

        for (int s = 0; s < nvox_slice; s++) {
            int lin = perm_slice[s];
            int ix  = lin % Nx;
            int iz  = lin / Nx;
            float fj_old = vol_slice[(size_t)iz * Nx + ix];
            float x = ix - half_x;
            float z = iz - half_z;

            /* ── data term: θ1, θ2 over angles ─────────────────────────── */
            float theta1 = 0.0f, theta2 = 0.0f;
            for (int ia = 0; ia < n_angles; ia++) {
                float ca = cos_a[ia], sa = sin_a[ia];
                float u  = x * ca + z * sa + half_det;
                int u0 = (int)floorf(u);
                int u1 = u0 + 1;
                float wu = u - u0;

                for (int kk = 0; kk < 2; kk++) {
                    int uk = u0 + kk;
                    if (uk < 0 || uk >= N_det) continue;
                    float a_col = (kk == 0) ? (1.0f - wu) : wu;
                    size_t off = ((size_t)ia * Ny + iy) * N_det + uk;
                    float ek = e_sino[off];
                    float lam = lambdas ? lambdas[off] : 1.0f;
                    /* Huber surrogate weight applied to (r = e·√Λ/σ) */
                    float r = ek * sqrtf(lam) / sigma;
                    float bw = huber_surr_weight(r, huber_T, huber_delta);
                    float w_data = bw * lam / (sigma * sigma);
                    theta1 += a_col * ek * w_data;
                    theta2 += a_col * a_col * w_data;
                }
            }
            /* ICD update for the data term alone: u_data = fj_old + θ1/θ2.
             * We rewrite the closed-form update slightly to keep θ1 with
             * the sign convention used in the paper (negative gradient). */
            theta1 = -theta1;  /* now ∂data/∂fj ≈ -theta1 (paper sign) */

            /* ── prior term: 26-neighbor qGGMRF surrogate ─────────────── */
            float n1 = 0.0f, n2 = 0.0f;
            for (int nb = 0; nb < 26; nb++) {
                int jx = ix + NB_DX[nb];
                int jy = iy + NB_DY[nb];
                int jz = iz + NB_DZ[nb];
                if (jx < 0 || jx >= Nx || jy < 0 || jy >= Ny || jz < 0 || jz >= Nz)
                    continue;
                float fk = volume_snapshot[((size_t)jy * Nz + jz) * Nx + jx];
                float a  = qggmrf_a_jk(fj_old - fk, &QP);
                float w  = NB_W[nb];
                n1 += w * a * fk;
                n2 += w * a;
            }

            /* ── closed-form voxel update (Bouman-Sauer surrogate-MM) ── */
            float denom = n2 + theta2;
            if (denom < 1e-30f) continue;
            float u_star = (n1 + theta2 * fj_old - theta1) / denom;
            float fj_new = (u_star > 0.0f) ? u_star : 0.0f;     /* positivity */
            float df = fj_new - fj_old;
            if (df == 0.0f) continue;

            vol_slice[(size_t)iz * Nx + ix] = fj_new;

            /* ── error sinogram incremental update: e -= df · A_*j ──── */
            for (int ia = 0; ia < n_angles; ia++) {
                float ca = cos_a[ia], sa = sin_a[ia];
                float u  = x * ca + z * sa + half_det;
                int u0 = (int)floorf(u);
                int u1 = u0 + 1;
                float wu = u - u0;
                if (u0 >= 0 && u0 < N_det) {
                    size_t off = ((size_t)ia * Ny + iy) * N_det + u0;
                    e_sino[off] -= df * (1.0f - wu);
                }
                if (u1 >= 0 && u1 < N_det) {
                    size_t off = ((size_t)ia * Ny + iy) * N_det + u1;
                    e_sino[off] -= df * wu;
                }
            }
        }
    }

    return 0;
}

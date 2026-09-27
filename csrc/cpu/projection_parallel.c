#include "ops.h"
#include "common.h"
#include "../include/proj_geom.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>

void forward_project_parallel_cpu(
    const float *vol,
    float       *sino,
    const float *cos_a,
    const float *sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    int num_threads
) {
    if (num_threads < 1) num_threads = 1;
    ProjGeomParallel g = proj_geom_parallel(Ny, Nz, Nx, n_angles, N_det);

    /* Each (ia, iy, id) ray writes to a unique sinogram entry — collapse(2)
     * over the two outer loops parallelizes 29*200 = 5800 work units (more
     * than enough for 32 threads) without any race. */
    #pragma omp parallel for collapse(2) schedule(static) num_threads(num_threads)
    for (int ia = 0; ia < n_angles; ia++) {
        for (int iy = 0; iy < Ny; iy++) {
            float ca = cos_a[ia], sa = sin_a[ia];
            const float *slice = vol + (size_t)iy * Nz * Nx;

            for (int id = 0; id < N_det; id++) {
                float t   = id - g.half_det;
                float sum = 0.0f;

                for (int step = -g.ray_len; step <= g.ray_len; step++) {
                    float s  = (float)step;
                    float x  = t * ca - s * sa;
                    float z  = t * sa + s * ca;
                    float fx = x + g.half_x;
                    float fz = z + g.half_z;

                    /* negated conjunction: a NaN sample (NaN/Inf angle)
                     * fails every comparison and is skipped, instead of
                     * indexing with (int)NaN = INT_MIN */
                    if (!(fx >= 0.0f && fx < (float)(Nx - 1) &&
                          fz >= 0.0f && fz < (float)(Nz - 1)))
                        continue;

                    int ix0 = (int)fx, iz0 = (int)fz;
                    int ix1 = ix0 + 1, iz1 = iz0 + 1;
                    float wx = fx - ix0, wz = fz - iz0;

                    sum += (1.0f - wz) * (1.0f - wx) * slice[(size_t)iz0 * Nx + ix0]
                         + (1.0f - wz) *          wx  * slice[(size_t)iz0 * Nx + ix1]
                         +          wz  * (1.0f - wx) * slice[(size_t)iz1 * Nx + ix0]
                         +          wz  *          wx  * slice[(size_t)iz1 * Nx + ix1];
                }

                sino[((size_t)ia * Ny + iy) * N_det + id] = sum;
            }
        }
    }
}

void back_project_parallel_cpu(
    const float *sino,
    float       *vol,
    const float *cos_a,
    const float *sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    int num_threads
) {
    if (num_threads < 1) num_threads = 1;
    ProjGeomParallel g = proj_geom_parallel(Ny, Nz, Nx, n_angles, N_det);

    memset(vol, 0, (size_t)Ny * Nz * Nx * sizeof(float));

    /*
     * Ray-driven scatter — strict adjoint of the forward pass. Different Y
     * slices are disjoint memory regions, so parallelism is safe over `iy`.
     * Within a slice, scatter across angles + detector + step would race, so
     * the inner triple loop stays serial within each thread.
     */
    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int iy = 0; iy < Ny; iy++) {
        float *slice = vol + (size_t)iy * Nz * Nx;

        for (int ia = 0; ia < n_angles; ia++) {
            float ca = cos_a[ia], sa = sin_a[ia];

            for (int id = 0; id < N_det; id++) {
                float t    = id - g.half_det;
                float sval = sino[((size_t)ia * Ny + iy) * N_det + id];

                for (int step = -g.ray_len; step <= g.ray_len; step++) {
                    float s  = (float)step;
                    float fx = (t * ca - s * sa) + g.half_x;
                    float fz = (t * sa + s * ca) + g.half_z;

                    /* negated conjunction: a NaN sample (NaN/Inf angle)
                     * fails every comparison and is skipped, instead of
                     * indexing with (int)NaN = INT_MIN */
                    if (!(fx >= 0.0f && fx < (float)(Nx - 1) &&
                          fz >= 0.0f && fz < (float)(Nz - 1)))
                        continue;

                    int ix0 = (int)fx, iz0 = (int)fz;
                    int ix1 = ix0 + 1, iz1 = iz0 + 1;
                    float wx = fx - ix0, wz = fz - iz0;

                    slice[(size_t)iz0 * Nx + ix0] += (1.0f - wz) * (1.0f - wx) * sval;
                    slice[(size_t)iz0 * Nx + ix1] += (1.0f - wz) *          wx  * sval;
                    slice[(size_t)iz1 * Nx + ix0] +=          wz  * (1.0f - wx) * sval;
                    slice[(size_t)iz1 * Nx + ix1] +=          wz  *          wx  * sval;
                }
            }
        }
    }
}

/*
 * Voxel-driven (gather) single-axis back-projection: the W-BP operator.
 *
 * Bit-for-bit the arithmetic of the PyTorch reference `_bp_gather`
 * (fasttomo/algorithms/analytic.py), given the same cos/sin tables:
 *   x = ix - 0.5(Nx-1), z = iz - 0.5(Nz-1)                 (fp32)
 *   u = ((x*cos) + (z*sin)) + 0.5(N_det-1)                  (no FMA)
 *   u0 = floor(u), w = u - u0
 *   vol[iy, iz, ix] += (1-w) * r(u0) + w * r(u0+1)          (angles in order)
 * with r(j) = sino[ia, iy, j] for 0 <= j < N_det and 0 outside, read from a
 * zero-padded copy of the sinogram rows so the inner loop has no branch
 * (the reference pads the same way and clamps the row index into the pad).
 * vol is overwritten. OpenMP over (Y, Z) rows: each (iy, iz) output row is
 * written by one thread, so there is no race and the result does not depend
 * on num_threads.
 */
#if defined(__clang__)
#define FT_NO_FP_CONTRACT
#elif defined(__GNUC__)
#define FT_NO_FP_CONTRACT __attribute__((optimize("fp-contract=off")))
#else
#define FT_NO_FP_CONTRACT
#endif

FT_NO_FP_CONTRACT
int back_project_gather_parallel_cpu(
    const float *sino,
    float       *vol,
    const float *cos_a,
    const float *sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    int num_threads
) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    if (num_threads < 1) num_threads = 1;
    const int    Np     = N_det + 2;                 /* one zero on each side */
    const float  half_x = (float)(0.5 * (Nx - 1));
    const float  half_z = (float)(0.5 * (Nz - 1));
    const float  half_d = (float)(0.5 * (N_det - 1));

    /* padded rows [n_angles, Ny, N_det + 2] and x*cos [n_angles, Nx] */
    float *pad = (float *)malloc((size_t)n_angles * Ny * Np * sizeof(float));
    float *xc  = (float *)malloc((size_t)n_angles * Nx * sizeof(float));
    if (!pad || !xc) { free(pad); free(xc); return -1; }

    #pragma omp parallel num_threads(num_threads)
    {
        #pragma omp for schedule(static) collapse(2)
        for (int ia = 0; ia < n_angles; ia++)
            for (int iy = 0; iy < Ny; iy++) {
                float *dst = pad + ((size_t)ia * Ny + iy) * Np;
                dst[0] = 0.0f;
                memcpy(dst + 1, sino + ((size_t)ia * Ny + iy) * N_det,
                       (size_t)N_det * sizeof(float));
                dst[Np - 1] = 0.0f;
            }
        #pragma omp for schedule(static)
        for (int ia = 0; ia < n_angles; ia++)
            for (int ix = 0; ix < Nx; ix++)
                xc[(size_t)ia * Nx + ix] = ((float)ix - half_x) * cos_a[ia];
        /* implicit barrier after each omp for */

        #pragma omp for schedule(static) collapse(2)
        for (int iy = 0; iy < Ny; iy++)
            for (int iz = 0; iz < Nz; iz++) {
                float *out = vol + ((size_t)iy * Nz + iz) * Nx;
                const float z = (float)iz - half_z;
                for (int ix = 0; ix < Nx; ix++) out[ix] = 0.0f;
                for (int ia = 0; ia < n_angles; ia++) {
                    const float *row = pad + ((size_t)ia * Ny + iy) * Np;
                    const float *xca = xc + (size_t)ia * Nx;
                    const float  zs  = z * sin_a[ia];
                    for (int ix = 0; ix < Nx; ix++) {
                        float u  = (xca[ix] + zs) + half_d;
                        /* floorf(u) without the libm call (baseline x86-64
                         * has no SSE4.1 roundss): exact for |u| < 2^31,
                         * far beyond any detector coordinate. */
                        int   i0 = (int)u;
                        i0 -= ((float)i0 > u);
                        float w  = u - (float)i0;
                        int   j0 = i0 + 1;                    /* +1: the pad */
                        int   j1 = j0 + 1;
                        j0 = j0 < 0 ? 0 : (j0 > Np - 1 ? Np - 1 : j0);
                        j1 = j1 < 0 ? 0 : (j1 > Np - 1 ? Np - 1 : j1);
                        float t0 = (1.0f - w) * row[j0];
                        float t1 = w * row[j1];
                        out[ix] += t0 + t1;
                    }
                }
            }
    }
    free(pad);
    free(xc);
    return 0;
}

void grad_angle_cpu(
    const float *vol,
    const float *grad_sino,
    float       *grad_angles,
    const float *cos_a,
    const float *sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    int num_threads
) {
    if (num_threads < 1) num_threads = 1;
    ProjGeomParallel g = proj_geom_parallel(Ny, Nz, Nx, n_angles, N_det);

    for (int ia = 0; ia < n_angles; ia++)
        grad_angles[ia] = 0.0f;

    /*
     * One scalar accumulator per angle; the outer ia loop is small (≈ tilt
     * series length) so we parallelize the inner (iy, id) work with a
     * reduction. This scales for the typical case where Ny, N_det >> n_angles.
     */
    for (int ia = 0; ia < n_angles; ia++) {
        float ca = cos_a[ia], sa = sin_a[ia];
        float g_theta = 0.0f;

        #pragma omp parallel for collapse(2) schedule(static) \
                                 num_threads(num_threads) reduction(+:g_theta)
        for (int iy = 0; iy < Ny; iy++) {
            for (int id = 0; id < N_det; id++) {
                const float *slice = vol + (size_t)iy * Nz * Nx;
                float t  = id - g.half_det;
                float gs = grad_sino[((size_t)ia * Ny + iy) * N_det + id];
                if (gs == 0.0f) continue;

                for (int step = -g.ray_len; step <= g.ray_len; step++) {
                    float s  = (float)step;
                    float x  = t * ca - s * sa;
                    float z  = t * sa + s * ca;
                    float fx = x + g.half_x;
                    float fz = z + g.half_z;

                    /* negated conjunction: a NaN sample (NaN/Inf angle)
                     * fails every comparison and is skipped, instead of
                     * indexing with (int)NaN = INT_MIN */
                    if (!(fx >= 0.0f && fx < (float)(Nx - 1) &&
                          fz >= 0.0f && fz < (float)(Nz - 1)))
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

                    g_theta += gs * (dvol_dx * (-z) + dvol_dz * x);
                }
            }
        }

        grad_angles[ia] = g_theta;
    }
}

#include "ops.h"
#include "common.h"
#include "../include/proj_geom.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>

/* Ray clipping for the parallel kernels (ft_clip_steps): narrow [*lo, *hi]
 * to the steps at which the sample of the ray (t, angle) can be inside
 * x in [0, Nx-1] (clip_ray_x) or z in [zlo, zhi] (clip_ray_z). The sample
 * coordinates are fx = (t*ca - s*sa) + half_x, fz = (t*sa + s*ca) + half_z. */
static inline double ray_mag(ProjGeomParallel g, float t)
{
    return fabs((double)t) + g.ray_len + fmax(g.half_x, g.half_z) + 1.0;
}

static inline void clip_ray_x(ProjGeomParallel g, float t, float ca, float sa, int *lo, int *hi)
{
    ft_clip_steps(-(double)sa, (double)t * ca + g.half_x, 0.0, g.Nx - 1, ray_mag(g, t), lo, hi);
}

static inline void clip_ray_z(ProjGeomParallel g, float t, float ca, float sa,
                              double zlo, double zhi, int *lo, int *hi)
{
    ft_clip_steps((double)ca, (double)t * sa + g.half_z, zlo, zhi, ray_mag(g, t), lo, hi);
}

/* One ray of the forward projection on `nl` Y lanes of a Y-innermost block
 * vt[Nz][Nx][nl]: acc[l] = the ray sum of lane l. Per lane, the operations
 * and their order are exactly those of the per-slice loop
 *   sum += (1-wz)*(1-wx)*v00 + (1-wz)*wx*v01 + wz*(1-wx)*v10 + wz*wx*v11
 * (products left to right, sums left to right), so the result is
 * bit-identical to it. Inlined with a constant `nl` for full blocks. */
static inline __attribute__((always_inline))
void fp_ray_lanes(const float *restrict vt, float *restrict acc, const int nl,
                  ProjGeomParallel g, float t, float ca, float sa, int lo, int hi)
{
    const int Nx = g.Nx, Nz = g.Nz;
    for (int l = 0; l < nl; l++) acc[l] = 0.0f;
    for (int step = lo; step <= hi; step++) {
        float s  = (float)step;
        float x  = t * ca - s * sa;
        float z  = t * sa + s * ca;
        float fx = x + g.half_x;
        float fz = z + g.half_z;

        /* negated conjunction: a NaN sample (NaN/Inf angle) fails every
         * comparison and is skipped, instead of indexing with
         * (int)NaN = INT_MIN */
        if (!(fx >= 0.0f && fx < (float)(Nx - 1) &&
              fz >= 0.0f && fz < (float)(Nz - 1)))
            continue;

        int ix0 = (int)fx, iz0 = (int)fz;
        float wx = fx - ix0, wz = fz - iz0;
        float w00 = (1.0f - wz) * (1.0f - wx), w01 = (1.0f - wz) * wx;
        float w10 =          wz  * (1.0f - wx), w11 =          wz  * wx;
        const float *restrict p00 = vt + ((size_t)iz0 * Nx + ix0) * nl;
        const float *restrict p01 = p00 + nl;
        const float *restrict p10 = p00 + (size_t)Nx * nl;
        const float *restrict p11 = p10 + nl;
        #pragma omp simd
        for (int l = 0; l < nl; l++)
            acc[l] += w00 * p00[l] + w01 * p01[l] + w10 * p10[l] + w11 * p11[l];
    }
}

/*
 * Ray-driven forward projection. Y is processed in blocks of FT_YB slices:
 * the block is copied Y-innermost ([Nz][Nx][nl]) once, then every
 * (angle, detector) ray computes its sample geometry once per step and
 * applies it to all nl slices with a SIMD loop, over the clipped step range
 * only. OpenMP over the rays of a block; each sinogram element is written
 * by one ray, so the result does not depend on num_threads. Bit-identical to
 * the per-slice scalar loop (see fp_ray_lanes). Ny = 1 reads the volume in
 * place. Scratch: Nz * Nx * min(Ny, FT_YB) floats (none for Ny = 1).
 * Returns 0, or -1 if the scratch allocation fails.
 */
int forward_project_parallel_cpu(
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
    const size_t plane = (size_t)Nz * Nx;
    const int    yb    = Ny < FT_YB ? Ny : FT_YB;
    float *vt = NULL;
    if (Ny > 1) {
        vt = (float *)malloc(plane * yb * sizeof(float));
        if (!vt) return -1;
    }

    #pragma omp parallel num_threads(num_threads)
    for (int y0 = 0; y0 < Ny; y0 += FT_YB) {
        const int nl = Ny - y0 < FT_YB ? Ny - y0 : FT_YB;
        const float *blk = vol;              /* Ny == 1: [Nz][Nx][1] is vol */
        if (Ny > 1) {
            #pragma omp for schedule(static)
            for (size_t p = 0; p < plane; p++)
                for (int l = 0; l < nl; l++)
                    vt[p * nl + l] = vol[(size_t)(y0 + l) * plane + p];
            /* implicit barrier */
            blk = vt;
        }

        #pragma omp for collapse(2) schedule(static)
        for (int ia = 0; ia < n_angles; ia++)
            for (int id = 0; id < N_det; id++) {
                float ca = cos_a[ia], sa = sin_a[ia];
                float t  = id - g.half_det;
                float acc[FT_YB];
                int lo = -g.ray_len, hi = g.ray_len;
                clip_ray_x(g, t, ca, sa, &lo, &hi);
                clip_ray_z(g, t, ca, sa, 0.0, Nz - 1, &lo, &hi);
                if (nl == FT_YB) fp_ray_lanes(blk, acc, FT_YB, g, t, ca, sa, lo, hi);
                else if (nl == 1) fp_ray_lanes(blk, acc, 1, g, t, ca, sa, lo, hi);
                else             fp_ray_lanes(blk, acc, nl, g, t, ca, sa, lo, hi);
                float *out = sino + ((size_t)ia * Ny + y0) * N_det + id;
                for (int l = 0; l < nl; l++) out[(size_t)l * N_det] = acc[l];
            }
        /* implicit barrier: vt is refilled for the next block */
    }
    free(vt);
    return 0;
}

/* Expected work of voxel row iz of the back-projection, summed over the
 * views: the length of the row inside the slice (x in [0, Nx-1]) and inside
 * the strip the detector sees (|x cos + z sin| <= half_det + 1/2, centred
 * coordinates), i.e. about its number of ray samples, plus one per view for
 * the per-ray overhead. Only used to balance the Z bands: the corners of the
 * slice outside the detector strip get fewer samples, so equal row counts
 * left the central bands with up to 1.4x the work of the outer ones. */
static double bp_row_work(ProjGeomParallel g, const float *cos_a, const float *sin_a, int iz)
{
    const double z = iz - (double)g.half_z, lim = g.half_det + 0.5;
    double w = 0.0;
    for (int ia = 0; ia < g.n_angles; ia++) {
        const double c = cos_a[ia], s = sin_a[ia];
        double lo = -(double)g.half_x, hi = g.half_x;
        if (fabs(c) > 1e-9) {
            double a = (-lim - z * s) / c, b = (lim - z * s) / c;
            if (a > b) { double t = a; a = b; b = t; }
            lo = fmax(lo, a);
            hi = fmin(hi, b);
        } else if (fabs(z * s) > lim) {
            hi = lo;
        }
        if (hi > lo) w += hi - lo;
        w += 1.0;
    }
    return w;
}

/* One ray of the back-projection, restricted to the rows [z0, z1) of a
 * Y-innermost block vt[Nz][Nx][nl]: scatters sv[l] (the ray's value in
 * lane l) into the rows of its samples' stencils that lie in the band.
 * Per voxel and lane the additions are those of the per-slice loop
 *   v[iz0][ix0] += (1-wz)*(1-wx)*sval, ... , v[iz1][ix1] += wz*wx*sval
 * in the same order (the caller walks the rays in the original
 * (angle, detector, step) order), so the result is bit-identical to it. */
static inline __attribute__((always_inline))
void bp_ray_band(float *restrict vt, const float *restrict sv, const int nl,
                 ProjGeomParallel g, float t, float ca, float sa, int lo, int hi,
                 int z0, int z1)
{
    const int Nx = g.Nx, Nz = g.Nz;
    for (int step = lo; step <= hi; step++) {
        float s  = (float)step;
        float fx = (t * ca - s * sa) + g.half_x;
        float fz = (t * sa + s * ca) + g.half_z;

        /* negated conjunction: NaN samples are skipped (see the forward) */
        if (!(fx >= 0.0f && fx < (float)(Nx - 1) &&
              fz >= 0.0f && fz < (float)(Nz - 1)))
            continue;

        int ix0 = (int)fx, iz0 = (int)fz;
        if (iz0 < z0 - 1 || iz0 >= z1) continue;     /* no row in the band */
        float wx = fx - ix0, wz = fz - iz0;
        if (iz0 >= z0) {
            float w00 = (1.0f - wz) * (1.0f - wx), w01 = (1.0f - wz) * wx;
            float *restrict p00 = vt + ((size_t)iz0 * Nx + ix0) * nl;
            float *restrict p01 = p00 + nl;
            #pragma omp simd
            for (int l = 0; l < nl; l++) p00[l] += w00 * sv[l];
            #pragma omp simd
            for (int l = 0; l < nl; l++) p01[l] += w01 * sv[l];
        }
        if (iz0 + 1 < z1) {
            float w10 = wz * (1.0f - wx), w11 = wz * wx;
            float *restrict p10 = vt + ((size_t)(iz0 + 1) * Nx + ix0) * nl;
            float *restrict p11 = p10 + nl;
            #pragma omp simd
            for (int l = 0; l < nl; l++) p10[l] += w10 * sv[l];
            #pragma omp simd
            for (int l = 0; l < nl; l++) p11[l] += w11 * sv[l];
        }
    }
}

/*
 * Ray-driven scatter back-projection, the exact transpose of the forward.
 *
 * Y is processed in blocks of FT_YB slices on Y-innermost copies of the
 * sinogram block ([n_angles][N_det][nl]) and of the output block
 * ([Nz][Nx][nl]). Within a block every OpenMP thread owns a band of Z rows
 * [z0, z1), walks every ray in the original (angle, detector, step) order
 * over the steps that can reach its band, and scatters only into its own
 * rows with a SIMD loop over the block's slices (bands of about equal
 * work, bp_row_work). Each voxel therefore
 * receives the same fp32 additions in the same order as the per-slice
 * serial loop: the result is bit-identical to it and independent of
 * num_threads, without atomics or private volumes, and all threads are used
 * even for Ny = 1 (the old kernel parallelised over Y only). Scratch for
 * Ny > 1: (Nz*Nx + n_angles*N_det) * min(Ny, FT_YB) floats (Ny = 1 works
 * in place). Returns 0, or -1 if the scratch allocation fails.
 */
int back_project_parallel_cpu(
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
    const size_t plane = (size_t)Nz * Nx;
    const size_t rays  = (size_t)n_angles * N_det;
    const int    yb    = Ny < FT_YB ? Ny : FT_YB;
    float  *vt = NULL, *st = NULL;
    int    *zb = (int *)malloc(((size_t)num_threads + 1) * sizeof(int)); /* bands */
    double *rw = (double *)malloc((size_t)Nz * sizeof(double));        /* row work */
    if (Ny > 1) {
        vt = (float *)malloc(plane * yb * sizeof(float));
        st = (float *)malloc(rays * yb * sizeof(float));
    }
    if (!zb || !rw || (Ny > 1 && (!vt || !st))) {
        free(zb); free(rw); free(vt); free(st);
        return -1;
    }

    #pragma omp parallel num_threads(num_threads)
    {
        const int nth = omp_get_num_threads(), tid = omp_get_thread_num();
        /* Z bands of about equal work (bp_row_work): the boundaries only
         * decide which thread adds to a voxel, never the order of its
         * additions, so they do not change the result */
        #pragma omp for schedule(static)
        for (int iz = 0; iz < Nz; iz++)
            rw[iz] = bp_row_work(g, cos_a, sin_a, iz);
        #pragma omp single
        {
            double total = 0.0, acc = 0.0;
            for (int iz = 0; iz < Nz; iz++) total += rw[iz];
            int k = 1;
            zb[0] = 0;
            for (int iz = 0; iz < Nz && k < nth; iz++) {
                /* row iz starts band k once the work before its middle
                 * reaches k/nth of the total */
                while (k < nth && acc + 0.5 * rw[iz] >= total * k / nth) zb[k++] = iz;
                acc += rw[iz];
            }
            while (k <= nth) zb[k++] = Nz;
        }
        /* implicit barrier: zb is complete */
        const int z0 = zb[tid], z1 = zb[tid + 1];

        for (int y0 = 0; y0 < Ny; y0 += FT_YB) {
            const int nl = Ny - y0 < FT_YB ? Ny - y0 : FT_YB;
            const float *sblk = sino;       /* Ny == 1: [n][N_det][1] is sino */
            float       *vblk = vol;        /*          [Nz][Nx][1] is vol   */
            if (Ny > 1) {
                #pragma omp for schedule(static)
                for (size_t r = 0; r < rays; r++) {
                    const size_t ia = r / N_det, id = r % N_det;
                    for (int l = 0; l < nl; l++)
                        st[r * nl + l] = sino[(ia * Ny + y0 + l) * N_det + id];
                }
                /* implicit barrier: st is complete */
                sblk = st;
                vblk = vt;
            }
            memset(vblk + (size_t)z0 * Nx * nl, 0, (size_t)(z1 - z0) * Nx * nl * sizeof(float));

            if (z1 > z0)
                for (int ia = 0; ia < n_angles; ia++) {
                    float ca = cos_a[ia], sa = sin_a[ia];
                    for (int id = 0; id < N_det; id++) {
                        float t = id - g.half_det;
                        const float *sv = sblk + ((size_t)ia * N_det + id) * nl;
                        int lo = -g.ray_len, hi = g.ray_len;
                        clip_ray_x(g, t, ca, sa, &lo, &hi);
                        /* samples with iz0 in [z0-1, z1-1]: fz in [z0-1, z1) */
                        clip_ray_z(g, t, ca, sa, z0 - 1, z1, &lo, &hi);
                        if (nl == FT_YB) bp_ray_band(vblk, sv, FT_YB, g, t, ca, sa, lo, hi, z0, z1);
                        else if (nl == 1) bp_ray_band(vblk, sv, 1, g, t, ca, sa, lo, hi, z0, z1);
                        else             bp_ray_band(vblk, sv, nl, g, t, ca, sa, lo, hi, z0, z1);
                    }
                }

            if (Ny > 1) {
                /* own rows back to [Ny, Nz, Nx] */
                for (int iz = z0; iz < z1; iz++)
                    for (int ix = 0; ix < Nx; ix++) {
                        const float *src = vt + ((size_t)iz * Nx + ix) * nl;
                        for (int l = 0; l < nl; l++)
                            vol[(size_t)(y0 + l) * plane + (size_t)iz * Nx + ix] = src[l];
                    }
                /* st and vt are refilled for the next block */
                #pragma omp barrier
            }
        }
    }
    free(zb);
    free(rw);
    free(vt);
    free(st);
    return 0;
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
 * vol is overwritten. Each output voxel is written by one thread, so there
 * is no race and the result does not depend on num_threads.
 *
 * Ny >= 2 runs Y-batched (gather_ybatched): u, w and the stencil indices do
 * not depend on the Y slice, so they are computed once per (iz, angle, ix)
 * and applied to FT_YB slices with a SIMD loop over a Y-innermost copy of
 * the padded sinogram block; per lane the fp32 operations and their order
 * are the ones above, so the two paths give the same bits. Ny = 1 keeps the
 * row loop (gather_rows), which is faster there.
 */
#if defined(__clang__)
#define FT_NO_FP_CONTRACT
#elif defined(__GNUC__)
#define FT_NO_FP_CONTRACT __attribute__((optimize("fp-contract=off")))
#else
#define FT_NO_FP_CONTRACT
#endif

FT_NO_FP_CONTRACT
static int gather_rows(
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

FT_NO_FP_CONTRACT
static int gather_ybatched(
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
    const int    Np     = N_det + 2;                 /* one zero on each side */
    const float  half_x = (float)(0.5 * (Nx - 1));
    const float  half_z = (float)(0.5 * (Nz - 1));
    const float  half_d = (float)(0.5 * (N_det - 1));
    const int    yb     = Ny < FT_YB ? Ny : FT_YB;
    const size_t plane  = (size_t)Nz * Nx;

    /* padded block [n_angles][Np][nl], x*cos [n_angles][Nx], and one output
     * row [Nx][nl] per thread */
    float *pad = (float *)malloc((size_t)n_angles * Np * yb * sizeof(float));
    float *xc  = (float *)malloc((size_t)n_angles * Nx * sizeof(float));
    float *row = (float *)malloc((size_t)num_threads * Nx * yb * sizeof(float));
    if (!pad || !xc || !row) { free(pad); free(xc); free(row); return -1; }

    #pragma omp parallel num_threads(num_threads)
    {
        float *o = row + (size_t)omp_get_thread_num() * Nx * yb;

        #pragma omp for schedule(static)
        for (int ia = 0; ia < n_angles; ia++)
            for (int ix = 0; ix < Nx; ix++)
                xc[(size_t)ia * Nx + ix] = ((float)ix - half_x) * cos_a[ia];

        for (int y0 = 0; y0 < Ny; y0 += FT_YB) {
            const int nl = Ny - y0 < FT_YB ? Ny - y0 : FT_YB;

            #pragma omp for schedule(static)
            for (int ia = 0; ia < n_angles; ia++) {
                float *d = pad + (size_t)ia * Np * nl;
                for (int l = 0; l < nl; l++) {
                    d[l] = 0.0f;
                    d[(size_t)(Np - 1) * nl + l] = 0.0f;
                    const float *src = sino + ((size_t)ia * Ny + y0 + l) * N_det;
                    for (int j = 0; j < N_det; j++) d[(size_t)(j + 1) * nl + l] = src[j];
                }
            }
            /* implicit barrier (also after xc): the block is complete */

            #pragma omp for schedule(static)
            for (int iz = 0; iz < Nz; iz++) {
                const float z = (float)iz - half_z;
                memset(o, 0, (size_t)Nx * nl * sizeof(float));
                for (int ia = 0; ia < n_angles; ia++) {
                    const float *pa  = pad + (size_t)ia * Np * nl;
                    const float *xca = xc + (size_t)ia * Nx;
                    const float  zs  = z * sin_a[ia];
                    for (int ix = 0; ix < Nx; ix++) {
                        float u  = (xca[ix] + zs) + half_d;
                        int   i0 = (int)u;                 /* floorf, see above */
                        i0 -= ((float)i0 > u);
                        float w  = u - (float)i0;
                        int   j0 = i0 + 1;
                        int   j1 = j0 + 1;
                        j0 = j0 < 0 ? 0 : (j0 > Np - 1 ? Np - 1 : j0);
                        j1 = j1 < 0 ? 0 : (j1 > Np - 1 ? Np - 1 : j1);
                        const float  omw = 1.0f - w;
                        const float *restrict r0 = pa + (size_t)j0 * nl;
                        const float *restrict r1 = pa + (size_t)j1 * nl;
                        float       *restrict ov = o + (size_t)ix * nl;
                        #pragma omp simd
                        for (int l = 0; l < nl; l++) {
                            float t0 = omw * r0[l];
                            float t1 = w * r1[l];
                            ov[l] += t0 + t1;
                        }
                    }
                }
                for (int ix = 0; ix < Nx; ix++)
                    for (int l = 0; l < nl; l++)
                        vol[(size_t)(y0 + l) * plane + (size_t)iz * Nx + ix] = o[(size_t)ix * nl + l];
            }
            /* implicit barrier: pad is refilled for the next block */
        }
    }
    free(pad);
    free(xc);
    free(row);
    return 0;
}

int back_project_gather_parallel_cpu(
    const float *sino,
    float       *vol,
    const float *cos_a,
    const float *sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    int num_threads
) {
    if (num_threads < 1) num_threads = 1;
    if (Ny >= 2)
        return gather_ybatched(sino, vol, cos_a, sin_a, Ny, Nz, Nx, n_angles, N_det, num_threads);
    return gather_rows(sino, vol, cos_a, sin_a, Ny, Nz, Nx, n_angles, N_det, num_threads);
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

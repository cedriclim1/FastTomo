#include "ops.h"
#include "common.h"
#include "../include/proj_geom.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>

/* Sampling rule of the parallel kernels ("loose", the rule of the rotmat
 * kernels and of the CUDA border-mode texture): the slice is a sum of
 * bilinear tent functions, zero outside, so a sample (fx, fz) contributes
 * while any corner of its stencil is inside, fx in (-1, Nx), fz in (-1, Nz),
 * with the lower corner floor(f) (ft_floor_loose) and corners outside the
 * slice reading 0. The kernels work on Y-innermost copies of the slice with
 * a zero border (FT_PAD voxels wide), so the stencil loads need no test.
 *
 * Ray clipping (ft_clip_steps): narrow [*lo, *hi] to the steps at which the
 * sample of the ray (t, angle) can be inside x in [-1, Nx] (clip_ray_x) or
 * z in [zlo, zhi] (clip_ray_z). The sample coordinates are
 * fx = (t*ca - s*sa) + half_x, fz = (t*sa + s*ca) + half_z. ray_len covers
 * the box [-1, N]^2: a sample in it is at most 0.5*sqrt((Nx+1)^2 + (Nz+1)^2)
 * < ceil(0.5*sqrt(Nx^2 + Nz^2)) + 1 steps from the ray's foot. */
static inline double ray_mag(ProjGeomParallel g, float t)
{
    return fabs((double)t) + g.ray_len + fmax(g.half_x, g.half_z) + 1.0;
}

static inline void clip_ray_x(ProjGeomParallel g, float t, float ca, float sa, int *lo, int *hi)
{
    ft_clip_steps(-(double)sa, (double)t * ca + g.half_x, -1.0, g.Nx, ray_mag(g, t), lo, hi);
}

/* floor(f) for f >= -1 (every coordinate the bounds tests below accept):
 * (int) truncates toward zero, so only (-1, 0) needs the -1, and f = -1
 * itself truncates to -1. A select, not the gather kernel's
 * `i -= (float)i > f` idiom, which measured +25 % in the forward. */
static inline int ft_floor_loose(float f)
{
    return f < 0.0f ? -1 : (int)f;
}

/* Copy the Y block [y0, y0+nl) of vol [Ny][Nz][Nx] into the Y-innermost
 * scratch vp[Nz+2*pad][Nx+2*pad][nl] with a zero border `pad` voxels wide
 * (all of it rewritten: nl, hence the layout, can change between blocks).
 * Call from inside a parallel region: an omp for over the padded rows. */
static inline void ft_fill_padded(float *restrict vp, const float *restrict vol,
                                  int Nz, int Nx, int y0, int nl, int pad)
{
    const size_t plane = (size_t)Nz * Nx;
    const int    Px    = Nx + 2 * pad;
    #pragma omp for schedule(static)
    for (int r = 0; r < Nz + 2 * pad; r++) {
        float *restrict row = vp + (size_t)r * Px * nl;
        const int iz = r - pad;
        if (iz < 0 || iz >= Nz) {
            memset(row, 0, (size_t)Px * nl * sizeof(float));
            continue;
        }
        memset(row, 0, (size_t)pad * nl * sizeof(float));
        memset(row + (size_t)(pad + Nx) * nl, 0, (size_t)pad * nl * sizeof(float));
        for (int ix = 0; ix < Nx; ix++) {
            float *restrict d = row + (size_t)(ix + pad) * nl;
            const float *src = vol + (size_t)y0 * plane + (size_t)iz * Nx + ix;
            for (int l = 0; l < nl; l++) d[l] = src[(size_t)l * plane];
        }
    }
    /* implicit barrier */
}

static inline void clip_ray_z(ProjGeomParallel g, float t, float ca, float sa,
                              double zlo, double zhi, int *lo, int *hi)
{
    ft_clip_steps((double)ca, (double)t * sa + g.half_z, zlo, zhi, ray_mag(g, t), lo, hi);
}

/* One ray of the forward projection on `nl` Y lanes of a Y-innermost block
 * vp[Nz+2][Nx+2][nl] (zero border 1 voxel wide): acc[l] = the ray sum of
 * lane l. Per lane, the operations and their order are exactly those of the
 * per-slice loop over the samples fx in (-1, Nx), fz in (-1, Nz)
 *   sum += (1-wz)*(1-wx)*v00 + (1-wz)*wx*v01 + wz*(1-wx)*v10 + wz*wx*v11
 * (products left to right, sums left to right; corners outside the slice
 * are 0), so the result is bit-identical to it. Inlined with a constant `nl`
 * for full blocks. */
static inline __attribute__((always_inline))
void fp_ray_lanes(const float *restrict vp, float *restrict acc, const int nl,
                  ProjGeomParallel g, float t, float ca, float sa, int lo, int hi)
{
    const int Nx = g.Nx, Nz = g.Nz, Px = Nx + 2;
    for (int l = 0; l < nl; l++) acc[l] = 0.0f;
    for (int step = lo; step <= hi; step++) {
        float s  = (float)step;
        float x  = t * ca - s * sa;
        float z  = t * sa + s * ca;
        float fx = x + g.half_x;
        float fz = z + g.half_z;

        /* loose bounds; negated conjunction: a NaN sample (NaN/Inf angle)
         * fails every comparison and is skipped, instead of indexing with
         * (int)NaN = INT_MIN */
        if (!(fx > -1.0f && fx < (float)Nx &&
              fz > -1.0f && fz < (float)Nz))
            continue;

        int ix0 = ft_floor_loose(fx), iz0 = ft_floor_loose(fz);
        float wx = fx - (float)ix0, wz = fz - (float)iz0;
        float w00 = (1.0f - wz) * (1.0f - wx), w01 = (1.0f - wz) * wx;
        float w10 =          wz  * (1.0f - wx), w11 =          wz  * wx;
        const float *restrict p00 = vp + ((size_t)(iz0 + 1) * Px + (ix0 + 1)) * nl;
        const float *restrict p01 = p00 + nl;
        const float *restrict p10 = p00 + (size_t)Px * nl;
        const float *restrict p11 = p10 + nl;
        #pragma omp simd
        for (int l = 0; l < nl; l++)
            acc[l] += w00 * p00[l] + w01 * p01[l] + w10 * p10[l] + w11 * p11[l];
    }
}

/*
 * Ray-driven forward projection. Y is processed in blocks of FT_YB slices:
 * the block is copied Y-innermost with a zero border ([Nz+2][Nx+2][nl],
 * the loose sampling rule above) once, then every (angle, detector) ray
 * computes its sample geometry once per step and applies it to all nl
 * slices with a SIMD loop, over the clipped step range only. OpenMP over
 * the rays of a block; each sinogram element is written by one ray, so the
 * result does not depend on num_threads. Bit-identical to the per-slice
 * scalar loop (see fp_ray_lanes). Scratch: (Nz+2) * (Nx+2) * min(Ny, FT_YB)
 * floats. Returns 0, or -1 if the scratch allocation fails.
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
    const int    yb = Ny < FT_YB ? Ny : FT_YB;
    float *vp = (float *)malloc((size_t)(Nz + 2) * (Nx + 2) * yb * sizeof(float));
    if (!vp) return -1;

    #pragma omp parallel num_threads(num_threads)
    for (int y0 = 0; y0 < Ny; y0 += FT_YB) {
        const int nl = Ny - y0 < FT_YB ? Ny - y0 : FT_YB;
        ft_fill_padded(vp, vol, Nz, Nx, y0, nl, 1);     /* ends with a barrier */

        #pragma omp for collapse(2) schedule(static)
        for (int ia = 0; ia < n_angles; ia++)
            for (int id = 0; id < N_det; id++) {
                float ca = cos_a[ia], sa = sin_a[ia];
                float t  = id - g.half_det;
                float acc[FT_YB];
                int lo = -g.ray_len, hi = g.ray_len;
                clip_ray_x(g, t, ca, sa, &lo, &hi);
                clip_ray_z(g, t, ca, sa, -1.0, Nz, &lo, &hi);
                if (nl == FT_YB) fp_ray_lanes(vp, acc, FT_YB, g, t, ca, sa, lo, hi);
                else if (nl == 1) fp_ray_lanes(vp, acc, 1, g, t, ca, sa, lo, hi);
                else             fp_ray_lanes(vp, acc, nl, g, t, ca, sa, lo, hi);
                float *out = sino + ((size_t)ia * Ny + y0) * N_det + id;
                for (int l = 0; l < nl; l++) out[(size_t)l * N_det] = acc[l];
            }
        /* implicit barrier: vp is refilled for the next block */
    }
    free(vp);
    return 0;
}

/* Expected work of voxel row iz of the back-projection, summed over the
 * views: the length of the row inside the slice (x in [-1, Nx]) and inside
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
        double lo = -(double)g.half_x - 1.0, hi = g.half_x + 1.0;   /* loose rule */
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
 * Y-innermost block vt[Nz][Nx+2][nl] (one pad column on each side, cropped
 * on write-back): scatters sv[l] (the ray's value in lane l) into the rows
 * of its samples' stencils that lie in the band. The samples are the
 * forward's (loose rule, fx in (-1, Nx), fz in (-1, Nz)); corners in the pad
 * columns or in the rows -1 and Nz (never in a band) are dropped, which is
 * the transpose of reading them as 0. Per voxel and lane the additions are
 * those of the per-slice loop
 *   v[iz0][ix0] += (1-wz)*(1-wx)*sval, ... , v[iz1][ix1] += wz*wx*sval
 * in the same order (the caller walks the rays in the original
 * (angle, detector, step) order), so the result is bit-identical to it. */
static inline __attribute__((always_inline))
void bp_ray_band(float *restrict vt, const float *restrict sv, const int nl,
                 ProjGeomParallel g, float t, float ca, float sa, int lo, int hi,
                 int z0, int z1)
{
    const int Nx = g.Nx, Nz = g.Nz, Px = Nx + 2;
    for (int step = lo; step <= hi; step++) {
        float s  = (float)step;
        float fx = (t * ca - s * sa) + g.half_x;
        float fz = (t * sa + s * ca) + g.half_z;

        /* negated conjunction: NaN samples are skipped (see the forward) */
        if (!(fx > -1.0f && fx < (float)Nx &&
              fz > -1.0f && fz < (float)Nz))
            continue;

        int ix0 = ft_floor_loose(fx), iz0 = ft_floor_loose(fz);
        if (iz0 < z0 - 1 || iz0 >= z1) continue;     /* no row in the band */
        float wx = fx - (float)ix0, wz = fz - (float)iz0;
        if (iz0 >= z0) {
            float w00 = (1.0f - wz) * (1.0f - wx), w01 = (1.0f - wz) * wx;
            float *restrict p00 = vt + ((size_t)iz0 * Px + (ix0 + 1)) * nl;
            float *restrict p01 = p00 + nl;
            #pragma omp simd
            for (int l = 0; l < nl; l++) p00[l] += w00 * sv[l];
            #pragma omp simd
            for (int l = 0; l < nl; l++) p01[l] += w01 * sv[l];
        }
        if (iz0 + 1 < z1) {
            float w10 = wz * (1.0f - wx), w11 = wz * wx;
            float *restrict p10 = vt + ((size_t)(iz0 + 1) * Px + (ix0 + 1)) * nl;
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
 * ([Nz][Nx+2][nl]: one pad column on each side takes the stencil corners
 * outside the slice and is dropped on write-back). Within a block every
 * OpenMP thread owns a band of Z rows [z0, z1), walks every ray in the
 * original (angle, detector, step) order over the steps that can reach its
 * band, and scatters only into its own rows with a SIMD loop over the
 * block's slices (bands of about equal work, bp_row_work). Each voxel
 * therefore receives the same fp32 additions in the same order as the
 * per-slice serial loop: the result is bit-identical to it and independent
 * of num_threads, without atomics or private volumes, and all threads are
 * used even for Ny = 1. Scratch: (Nz*(Nx+2) + n_angles*N_det) *
 * min(Ny, FT_YB) floats (no sinogram copy for Ny = 1). Returns 0, or -1 if
 * the scratch allocation fails.
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
    const int    Px    = Nx + 2;
    const int    yb    = Ny < FT_YB ? Ny : FT_YB;
    float  *vt = (float *)malloc((size_t)Nz * Px * yb * sizeof(float));
    float  *st = NULL;
    int    *zb = (int *)malloc(((size_t)num_threads + 1) * sizeof(int)); /* bands */
    double *rw = (double *)malloc((size_t)Nz * sizeof(double));        /* row work */
    if (Ny > 1) st = (float *)malloc(rays * yb * sizeof(float));
    if (!vt || !zb || !rw || (Ny > 1 && !st)) {
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
            if (Ny > 1) {
                #pragma omp for schedule(static)
                for (size_t r = 0; r < rays; r++) {
                    const size_t ia = r / N_det, id = r % N_det;
                    for (int l = 0; l < nl; l++)
                        st[r * nl + l] = sino[(ia * Ny + y0 + l) * N_det + id];
                }
                /* implicit barrier: st is complete */
                sblk = st;
            }
            memset(vt + (size_t)z0 * Px * nl, 0, (size_t)(z1 - z0) * Px * nl * sizeof(float));

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
                        if (nl == FT_YB) bp_ray_band(vt, sv, FT_YB, g, t, ca, sa, lo, hi, z0, z1);
                        else if (nl == 1) bp_ray_band(vt, sv, 1, g, t, ca, sa, lo, hi, z0, z1);
                        else             bp_ray_band(vt, sv, nl, g, t, ca, sa, lo, hi, z0, z1);
                    }
                }

            /* own rows back to [Ny, Nz, Nx], without the pad columns */
            for (int iz = z0; iz < z1; iz++)
                for (int ix = 0; ix < Nx; ix++) {
                    const float *src = vt + ((size_t)iz * Px + ix + 1) * nl;
                    for (int l = 0; l < nl; l++)
                        vol[(size_t)(y0 + l) * plane + (size_t)iz * Nx + ix] = src[l];
                }
            /* st and vt are refilled for the next block */
            if (Ny > 1) {
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

/* Detectors per grad_angle task: the work of an angle is split into fixed
 * chunks of detector rays (independent of num_threads), so that every
 * accumulation below has an order fixed by the geometry alone. */
#define FT_GA_CHUNK 32

/* dL/dtheta contribution of one chunk of detector rays of one angle on the
 * nl Y lanes of a Y-innermost block vt[Nz][Nx][nl]. Per ray and lane, the
 * derivative of the ray sum, sum over steps of dvol/dx*(-z) + dvol/dz*x
 * (the fp32 terms of the scalar loop), is accumulated in fp32 like the
 * forward's ray sum; it is then weighted by the upstream gradient and added
 * to the lane's fp64 accumulator acc[l], in detector order. (Weighting per
 * step and accumulating in fp64 per step made gcc spill the SSE2 loop.) */
static inline __attribute__((always_inline))
void ga_chunk_lanes(const float *restrict vt, const float *restrict grad_sino,
                    double *restrict acc, const int nl, ProjGeomParallel g,
                    int ia, int y0, int id0, int id1, float ca, float sa)
{
    const int Nx = g.Nx, Nz = g.Nz, Ny = g.Ny, N_det = g.N_det;
    for (int l = 0; l < nl; l++) acc[l] = 0.0;
    for (int id = id0; id < id1; id++) {
        float gsv[FT_YB], dray[FT_YB];
        int any = 0;
        for (int l = 0; l < nl; l++) {
            gsv[l] = grad_sino[((size_t)ia * Ny + y0 + l) * N_det + id];
            any |= gsv[l] != 0.0f;
            dray[l] = 0.0f;
        }
        if (!any) continue;          /* the scalar loop skipped these rays too */
        float t = id - g.half_det;
        int lo = -g.ray_len, hi = g.ray_len;
        clip_ray_x(g, t, ca, sa, &lo, &hi);
        clip_ray_z(g, t, ca, sa, 0.0, Nz - 1, &lo, &hi);
        for (int step = lo; step <= hi; step++) {
            float s  = (float)step;
            float x  = t * ca - s * sa;
            float z  = t * sa + s * ca;
            float fx = x + g.half_x;
            float fz = z + g.half_z;

            /* negated conjunction: NaN samples are skipped (see the forward) */
            if (!(fx >= 0.0f && fx < (float)(Nx - 1) &&
                  fz >= 0.0f && fz < (float)(Nz - 1)))
                continue;

            int ix0 = (int)fx, iz0 = (int)fz;
            float wx = fx - ix0, wz = fz - iz0;
            const float *restrict p00 = vt + ((size_t)iz0 * Nx + ix0) * nl;
            const float *restrict p01 = p00 + nl;
            const float *restrict p10 = p00 + (size_t)Nx * nl;
            const float *restrict p11 = p10 + nl;
            #pragma omp simd
            for (int l = 0; l < nl; l++) {
                float dvol_dx = (1.0f - wz) * (p01[l] - p00[l]) + wz * (p11[l] - p10[l]);
                float dvol_dz = (1.0f - wx) * (p10[l] - p00[l]) + wx * (p11[l] - p01[l]);
                dray[l] += dvol_dx * (-z) + dvol_dz * x;
            }
        }
        for (int l = 0; l < nl; l++)
            if (gsv[l] != 0.0f) acc[l] += (double)gsv[l] * (double)dray[l];
    }
}

/*
 * Gradient of the parallel forward projection w.r.t. the tilt angles:
 * at each ray sample, d sino / d theta = dvol/dx * (-z) + dvol/dz * x, with
 * the bilinear finite differences of the volume. Y-batched like the
 * forward (FT_YB slices per SIMD loop on a Y-innermost copy of the block;
 * Ny = 1 reads the volume in place). The work of a block is split into
 * tasks of one angle and FT_GA_CHUNK detector rays; each ray's derivative
 * is summed in fp32 (as the forward sums a ray), weighted by its upstream
 * gradient and accumulated per Y lane in fp64, and the partial sums are
 * added per angle in a fixed (block, chunk, lane) order. The result
 * therefore does not depend on num_threads, and the error of the old
 * kernel's single fp32 accumulator per angle and thread (1e-6 to 4e-5
 * relative, growing with the problem size) is gone. Returns 0, or -1
 * if the scratch allocation fails.
 */
int grad_angle_cpu(
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
    const size_t plane = (size_t)Nz * Nx;
    const int    yb    = Ny < FT_YB ? Ny : FT_YB;
    const int    nch   = (N_det + FT_GA_CHUNK - 1) / FT_GA_CHUNK;
    float  *vt   = NULL;
    double *part = (double *)malloc((size_t)n_angles * nch * FT_YB * sizeof(double));
    double *tot  = (double *)calloc((size_t)n_angles, sizeof(double));
    if (Ny > 1) vt = (float *)malloc(plane * yb * sizeof(float));
    if (!part || !tot || (Ny > 1 && !vt)) { free(part); free(tot); free(vt); return -1; }

    #pragma omp parallel num_threads(num_threads)
    for (int y0 = 0; y0 < Ny; y0 += FT_YB) {
        const int nl = Ny - y0 < FT_YB ? Ny - y0 : FT_YB;
        const float *blk = vol;
        if (Ny > 1) {
            #pragma omp for schedule(static)
            for (size_t p = 0; p < plane; p++)
                for (int l = 0; l < nl; l++)
                    vt[p * nl + l] = vol[(size_t)(y0 + l) * plane + p];
            blk = vt;
        }

        #pragma omp for collapse(2) schedule(dynamic, 1)
        for (int ia = 0; ia < n_angles; ia++)
            for (int c = 0; c < nch; c++) {
                const int id0 = c * FT_GA_CHUNK;
                const int id1 = id0 + FT_GA_CHUNK < N_det ? id0 + FT_GA_CHUNK : N_det;
                double *acc = part + ((size_t)ia * nch + c) * FT_YB;
                if (nl == FT_YB)
                    ga_chunk_lanes(blk, grad_sino, acc, FT_YB, g, ia, y0, id0, id1, cos_a[ia], sin_a[ia]);
                else if (nl == 1)
                    ga_chunk_lanes(blk, grad_sino, acc, 1, g, ia, y0, id0, id1, cos_a[ia], sin_a[ia]);
                else
                    ga_chunk_lanes(blk, grad_sino, acc, nl, g, ia, y0, id0, id1, cos_a[ia], sin_a[ia]);
            }
        /* implicit barrier: every partial of the block is done */

        #pragma omp for schedule(static)
        for (int ia = 0; ia < n_angles; ia++)
            for (int c = 0; c < nch; c++)
                for (int l = 0; l < nl; l++)
                    tot[ia] += part[((size_t)ia * nch + c) * FT_YB + l];
        /* implicit barrier: vt and part are refilled for the next block */
    }

    for (int ia = 0; ia < n_angles; ia++)
        grad_angles[ia] = (float)tot[ia];
    free(part);
    free(tot);
    free(vt);
    return 0;
}

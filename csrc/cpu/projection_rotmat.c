#include "ops.h"
#include "common.h"
#include "../include/proj_geom.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>

/* Ray clipping for the rotmat kernels (ft_clip_steps): narrow [*lo, *hi] to
 * the steps at which the sample (ox + s*d) + half, per axis, can be inside
 * the loose box (-1, N) of every axis. The kernels keep their exact in-loop
 * test, so the samples taken are unchanged. */
static inline void clip_ray_rotmat(ProjGeomRotmat g, float ox, float oy, float oz,
                                   float dxs, float dys, float dzs, int *lo, int *hi)
{
    const double mag = fabs((double)ox) + fabs((double)oy) + fabs((double)oz)
                     + g.ray_len + fmax(g.half_x, fmax(g.half_y, g.half_z)) + 1.0;
    *lo = -g.ray_len;
    *hi =  g.ray_len;
    ft_clip_steps(dxs, (double)ox + g.half_x, -1.0, g.Nx, mag, lo, hi);
    ft_clip_steps(dys, (double)oy + g.half_y, -1.0, g.Ny, mag, lo, hi);
    ft_clip_steps(dzs, (double)oz + g.half_z, -1.0, g.Nz, mag, lo, hi);
}

/* The 8 trilinear corners of the sample with lower corner (ix0, iy0, iz0):
 * direct loads when all of them are inside the volume (almost every
 * sample), else the border-zero SAMP3 per corner. The same values either
 * way, so the fast path does not change any bit. */
#define LOAD_CORNERS()                                                        \
    float v000, v001, v010, v011, v100, v101, v110, v111;                    \
    if (ix0 >= 0 && ix1 < Nx && iy0 >= 0 && iy1 < Ny && iz0 >= 0 && iz1 < Nz) { \
        const float *p = vol + ((size_t)iy0 * Nz + iz0) * Nx + ix0;           \
        const size_t sy = (size_t)Nz * Nx;                                    \
        v000 = p[0];      v001 = p[1];                                        \
        v010 = p[Nx];     v011 = p[Nx + 1];                                   \
        v100 = p[sy];     v101 = p[sy + 1];                                   \
        v110 = p[sy + Nx]; v111 = p[sy + Nx + 1];                             \
    } else {                                                                  \
        v000 = SAMP3(ix0, iy0, iz0); v001 = SAMP3(ix1, iy0, iz0);             \
        v010 = SAMP3(ix0, iy0, iz1); v011 = SAMP3(ix1, iy0, iz1);             \
        v100 = SAMP3(ix0, iy1, iz0); v101 = SAMP3(ix1, iy1, iz0);             \
        v110 = SAMP3(ix0, iy1, iz1); v111 = SAMP3(ix1, iy1, iz1);             \
    }

/* ── 3D rotation-matrix forward projection (CPU) ───────────────────────────
 * Detector-driven; trilinear interpolation in 3D; one ray per (ia, iv, iu).
 * Mirrors forward_project_rotmat_cuda but with explicit interpolation.
 *
 * mats: [n_angles*9] row-major. M = mats + ia*9.
 *   detector-u = M[0]*x + M[1]*y + M[2]*z   (R[0,:] · world)
 *   detector-v = M[3]*x + M[4]*y + M[5]*z   (R[1,:] · world)
 *   ray dir   = (M[6], M[7], M[8])          (R[2,:])
 * Ray parameterisation (R^T applied to (u, v, s)):
 *   x(s) = u*M[0] + v*M[3] + s*M[6]
 *   y(s) = u*M[1] + v*M[4] + s*M[7]
 *   z(s) = u*M[2] + v*M[5] + s*M[8]
 * Rays may cross all three faces, so ray length covers the full 3D diagonal;
 * each ray is clipped to the steps that can reach the loose box below
 * (clip_ray_rotmat), and interior samples load their 8 corners directly
 * (LOAD_CORNERS). Both only skip work, never change a sample: bit-identical
 * to the unclipped, all-SAMP3 loop.
 *
 * Border-zero semantics matching CUDA cudaAddressModeBorder + linear filter:
 * a sample at fx=Nx-1 must still return vol[Nx-1] (the upper neighbour
 * vol[Nx] is treated as zero, weighted by wx=0). This requires loose bounds
 * [-1, N) and per-corner safe loads — a strict fx<Nx-1 check would zero the
 * last slice entirely.
 */
void forward_project_rotmat_cpu(
    const float *vol,     /* [Ny, Nz, Nx] */
    float       *sino,   /* [n_angles, Ny, N_det] */
    const float *mats,   /* [n_angles*9] row-major */
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    int num_threads
) {
    ProjGeomRotmat g = proj_geom_rotmat(Ny, Nz, Nx, n_angles, N_det);

    #pragma omp parallel for collapse(3) schedule(static) num_threads(num_threads)
    for (int ia = 0; ia < n_angles; ia++) {
        for (int iv = 0; iv < Ny; iv++) {
            for (int iu = 0; iu < N_det; iu++) {
                const float *M = mats + (size_t)ia * 9;
                float dxs = M[6], dys = M[7], dzs = M[8];
                float v  = iv - g.half_y;
                float u  = iu - g.half_det;
                float ox = u * M[0] + v * M[3];
                float oy = u * M[1] + v * M[4];
                float oz = u * M[2] + v * M[5];

                int lo, hi;
                clip_ray_rotmat(g, ox, oy, oz, dxs, dys, dzs, &lo, &hi);

                float sum = 0.0f;
                for (int step = lo; step <= hi; step++) {
                    float s  = (float)step;
                    float fx = ox + s * dxs + g.half_x;
                    float fy = oy + s * dys + g.half_y;
                    float fz = oz + s * dzs + g.half_z;

                    /* Loose bounds: skip only when ALL 8 corners are outside.
                     * fx in (-1, Nx), fy in (-1, Ny), fz in (-1, Nz). */
                    if (fx <= -1.0f || fx >= (float)Nx ||
                        fy <= -1.0f || fy >= (float)Ny ||
                        fz <= -1.0f || fz >= (float)Nz)
                        continue;

                    int ix0 = (int)floorf(fx);
                    int iy0 = (int)floorf(fy);
                    int iz0 = (int)floorf(fz);
                    int ix1 = ix0 + 1, iy1 = iy0 + 1, iz1 = iz0 + 1;
                    float wx = fx - (float)ix0;
                    float wy = fy - (float)iy0;
                    float wz = fz - (float)iz0;
                    float ux = 1.0f - wx, uy = 1.0f - wy, uz = 1.0f - wz;

                    LOAD_CORNERS()

                    sum += uy * uz * (ux * v000 + wx * v001)
                         + uy * wz * (ux * v010 + wx * v011)
                         + wy * uz * (ux * v100 + wx * v101)
                         + wy * wz * (ux * v110 + wx * v111);
                }

                sino[((size_t)ia * Ny + iv) * N_det + iu] = sum;
            }
        }
    }
}

/* ── 3D rotation-matrix back-projection (CPU) ──────────────────────────────
 * Voxel-driven gather (mirrors back_project_rotmat_cuda); each voxel reads
 * the bilinearly-interpolated sinogram value at its rotated detector position
 * and accumulates over angles. Approximate adjoint of forward_project_rotmat_cpu
 * — see CLAUDE.md note that CPU/CUDA BP outputs are not expected to match
 * numerically. Border-zero semantics: a voxel mapping to u_det == N_det-1
 * still contributes the last detector column.
 */
void back_project_rotmat_cpu(
    const float *sino,   /* [n_angles, Ny, N_det] */
    float       *vol,     /* [Ny, Nz, Nx] */
    const float *mats,   /* [n_angles*9] row-major */
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    int num_threads
) {
    ProjGeomRotmat g = proj_geom_rotmat(Ny, Nz, Nx, n_angles, N_det);

    memset(vol, 0, (size_t)Ny * Nz * Nx * sizeof(float));

    #pragma omp parallel for collapse(3) schedule(static) num_threads(num_threads)
    for (int iy = 0; iy < Ny; iy++) {
        for (int iz = 0; iz < Nz; iz++) {
            for (int ix = 0; ix < Nx; ix++) {
                float y = iy - g.half_y;
                float z = iz - g.half_z;
                float x = ix - g.half_x;
                float val = 0.0f;

                for (int ia = 0; ia < n_angles; ia++) {
                    const float *M = mats + (size_t)ia * 9;
                    float u_det = M[0]*x + M[1]*y + M[2]*z + g.half_det;
                    float v_det = M[3]*x + M[4]*y + M[5]*z + g.half_y;

                    if (u_det <= -1.0f || u_det >= (float)N_det ||
                        v_det <= -1.0f || v_det >= (float)Ny)
                        continue;

                    int iu0 = (int)floorf(u_det), iv0 = (int)floorf(v_det);
                    int iu1 = iu0 + 1,            iv1 = iv0 + 1;
                    float wu = u_det - (float)iu0;
                    float wv = v_det - (float)iv0;

                    const float *layer = sino + (size_t)ia * Ny * N_det;
                    #define SSAMP(iu, iv) \
                        (((iu) >= 0 && (iu) < N_det && (iv) >= 0 && (iv) < Ny) \
                         ? layer[(size_t)(iv) * N_det + (iu)] : 0.0f)
                    val += (1.0f - wv) * (1.0f - wu) * SSAMP(iu0, iv0)
                         + (1.0f - wv) *          wu  * SSAMP(iu1, iv0)
                         +          wv  * (1.0f - wu) * SSAMP(iu0, iv1)
                         +          wv  *          wu  * SSAMP(iu1, iv1);
                    #undef SSAMP
                }

                vol[((size_t)iy * Nz + iz) * Nx + ix] = val;
            }
        }
    }
}

/* ── gradient of 3D rotmat FP w.r.t. rotation matrix (CPU) ─────────────────
 * Same sampling, loose bounds (-1, N) and border-zero corners as
 * grad_rotmat_kernel (CUDA) and forward_project_rotmat_cpu, so it is the exact
 * gradient of the forward away from trilinear kinks; the two backends differ
 * only in fp32 summation order. Detector-driven loop accumulates per-ray
 * partials of all 9 matrix entries, no reduction needed (pure serial).
 *
 * Per ray step (s = step):
 *   ∂tx/∂M[{0,3,6}] = (u, v, s)
 *   ∂ty/∂M[{1,4,7}] = (u, v, s)
 *   ∂tz/∂M[{2,5,8}] = (u, v, s)
 * with spatial derivatives ∂vol/∂{x,y,z} from analytic trilinear weights.
 */
void grad_rotmat_cpu(
    const float *vol,         /* [Ny, Nz, Nx] */
    const float *grad_sino,      /* [n_angles, Ny, N_det] */
    const float *mats,       /* [n_angles*9] */
    float       *grad_mats,  /* [n_angles*9] zeroed by this fn */
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    int num_threads
) {
    ProjGeomRotmat g = proj_geom_rotmat(Ny, Nz, Nx, n_angles, N_det);

    for (int k = 0; k < n_angles * 9; k++) grad_mats[k] = 0.0f;

    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int ia = 0; ia < n_angles; ia++) {
        const float *M = mats + (size_t)ia * 9;
        float *gM = grad_mats + (size_t)ia * 9;
        float dxs = M[6], dys = M[7], dzs = M[8];

        for (int iv = 0; iv < Ny; iv++) {
            float v = iv - g.half_y;
            for (int iu = 0; iu < N_det; iu++) {
                float gs = grad_sino[((size_t)ia * Ny + iv) * N_det + iu];
                if (gs == 0.0f) continue;

                float u  = iu - g.half_det;
                float ox = u * M[0] + v * M[3];
                float oy = u * M[1] + v * M[4];
                float oz = u * M[2] + v * M[5];

                int lo, hi;
                clip_ray_rotmat(g, ox, oy, oz, dxs, dys, dzs, &lo, &hi);

                for (int step = lo; step <= hi; step++) {
                    float s  = (float)step;
                    float fx = ox + s * dxs + g.half_x;
                    float fy = oy + s * dys + g.half_y;
                    float fz = oz + s * dzs + g.half_z;

                    if (fx <= -1.0f || fx >= (float)Nx ||
                        fy <= -1.0f || fy >= (float)Ny ||
                        fz <= -1.0f || fz >= (float)Nz)
                        continue;

                    int ix0 = (int)floorf(fx);
                    int iy0 = (int)floorf(fy);
                    int iz0 = (int)floorf(fz);
                    int ix1 = ix0 + 1, iy1 = iy0 + 1, iz1 = iz0 + 1;
                    float wx = fx - (float)ix0;
                    float wy = fy - (float)iy0;
                    float wz = fz - (float)iz0;
                    float ux = 1.0f - wx, uy = 1.0f - wy, uz = 1.0f - wz;

                    LOAD_CORNERS()

                    float dfx = uy * uz * (v001 - v000) + wy * uz * (v101 - v100)
                              + uy * wz * (v011 - v010) + wy * wz * (v111 - v110);
                    float dfy = ux * uz * (v100 - v000) + wx * uz * (v101 - v001)
                              + ux * wz * (v110 - v010) + wx * wz * (v111 - v011);
                    float dfz = ux * uy * (v010 - v000) + wx * uy * (v011 - v001)
                              + ux * wy * (v110 - v100) + wx * wy * (v111 - v101);

                    float gdfx = gs * dfx, gdfy = gs * dfy, gdfz = gs * dfz;
                    gM[0] += gdfx * u;  gM[1] += gdfy * u;  gM[2] += gdfz * u;
                    gM[3] += gdfx * v;  gM[4] += gdfy * v;  gM[5] += gdfz * v;
                    gM[6] += gdfx * s;  gM[7] += gdfy * s;  gM[8] += gdfz * s;
                }
            }
        }
    }
}

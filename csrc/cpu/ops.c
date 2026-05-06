#include "ops.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

void forward_project_cpu(
    const float *volume,
    float       *sinogram,
    const float *cos_a,
    const float *sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det
) {
    float half_det = (N_det - 1) * 0.5f;
    float half_x   = (Nx   - 1) * 0.5f;
    float half_z   = (Nz   - 1) * 0.5f;
    /* ray length covers the full diagonal of the XZ slab */
    int ray_len = (int)ceilf(0.5f * sqrtf((float)(Nx * Nx + Nz * Nz))) + 1;

    for (int ia = 0; ia < n_angles; ia++) {
        float ca = cos_a[ia], sa = sin_a[ia];

        for (int iy = 0; iy < Ny; iy++) {
            const float *slice = volume + (size_t)iy * Nz * Nx;

            for (int id = 0; id < N_det; id++) {
                float t   = id - half_det;   /* detector coord (pixels, centred) */
                float sum = 0.0f;

                for (int step = -ray_len; step <= ray_len; step++) {
                    float s  = (float)step;
                    /* point on ray at parameter s */
                    float x  = t * ca - s * sa;
                    float z  = t * sa + s * ca;
                    /* convert to 0-based array indices */
                    float fx = x + half_x;
                    float fz = z + half_z;

                    if (fx < 0.0f || fx >= (float)(Nx - 1) ||
                        fz < 0.0f || fz >= (float)(Nz - 1))
                        continue;

                    int ix0 = (int)fx, iz0 = (int)fz;
                    int ix1 = ix0 + 1, iz1 = iz0 + 1;
                    float wx = fx - ix0, wz = fz - iz0;

                    sum += (1.0f - wz) * (1.0f - wx) * slice[(size_t)iz0 * Nx + ix0]
                         + (1.0f - wz) *          wx  * slice[(size_t)iz0 * Nx + ix1]
                         +          wz  * (1.0f - wx) * slice[(size_t)iz1 * Nx + ix0]
                         +          wz  *          wx  * slice[(size_t)iz1 * Nx + ix1];
                }

                sinogram[((size_t)ia * Ny + iy) * N_det + id] = sum;
            }
        }
    }
}

void back_project_cpu(
    const float *sinogram,
    float       *volume,
    const float *cos_a,
    const float *sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det
) {
    float half_det = (N_det - 1) * 0.5f;
    float half_x   = (Nx   - 1) * 0.5f;
    float half_z   = (Nz   - 1) * 0.5f;
    int ray_len = (int)ceilf(0.5f * sqrtf((float)(Nx * Nx + Nz * Nz))) + 1;

    memset(volume, 0, (size_t)Ny * Nz * Nx * sizeof(float));

    /*
     * Ray-driven scatter — strict adjoint of the ray-driven forward projection.
     * For each ray (ia, iy, id) and each step along the ray, scatter
     * sinogram[ia,iy,id] into the four neighbouring voxels with the same
     * bilinear weights used in the forward pass.
     */
    for (int ia = 0; ia < n_angles; ia++) {
        float ca = cos_a[ia], sa = sin_a[ia];

        for (int iy = 0; iy < Ny; iy++) {
            float *slice = volume + (size_t)iy * Nz * Nx;

            for (int id = 0; id < N_det; id++) {
                float t    = id - half_det;
                float sval = sinogram[((size_t)ia * Ny + iy) * N_det + id];

                for (int step = -ray_len; step <= ray_len; step++) {
                    float s  = (float)step;
                    float fx = (t * ca - s * sa) + half_x;
                    float fz = (t * sa + s * ca) + half_z;

                    if (fx < 0.0f || fx >= (float)(Nx - 1) ||
                        fz < 0.0f || fz >= (float)(Nz - 1))
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

void grad_angle_cpu(
    const float *volume,
    const float *grad_sino,
    float       *grad_angles,
    const float *cos_a,
    const float *sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det
) {
    float half_det = (N_det - 1) * 0.5f;
    float half_x   = (Nx   - 1) * 0.5f;
    float half_z   = (Nz   - 1) * 0.5f;
    int ray_len = (int)ceilf(0.5f * sqrtf((float)(Nx * Nx + Nz * Nz))) + 1;

    for (int ia = 0; ia < n_angles; ia++)
        grad_angles[ia] = 0.0f;

    for (int ia = 0; ia < n_angles; ia++) {
        float ca = cos_a[ia], sa = sin_a[ia];
        float g_theta = 0.0f;

        for (int iy = 0; iy < Ny; iy++) {
            const float *slice = volume + (size_t)iy * Nz * Nx;

            for (int id = 0; id < N_det; id++) {
                float t  = id - half_det;
                float gs = grad_sino[((size_t)ia * Ny + iy) * N_det + id];
                if (gs == 0.0f) continue;

                for (int step = -ray_len; step <= ray_len; step++) {
                    float s  = (float)step;
                    float x  = t * ca - s * sa;
                    float z  = t * sa + s * ca;
                    float fx = x + half_x;
                    float fz = z + half_z;

                    /* bounds check mirrors forward_project_cpu exactly */
                    if (fx < 0.0f || fx >= (float)(Nx - 1) ||
                        fz < 0.0f || fz >= (float)(Nz - 1))
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

/* ── 3D rotation-matrix forward projection (CPU) ───────────────────────────
 * Detector-driven; trilinear interpolation in 3D; one ray per (ia, iv, iu).
 * Mirrors forward_project_rotmat_cuda but with explicit interpolation.
 *
 * matrices: [n_angles*9] row-major. M = matrices + ia*9.
 *   detector-u = M[0]*x + M[1]*y + M[2]*z   (R[0,:] · world)
 *   detector-v = M[3]*x + M[4]*y + M[5]*z   (R[1,:] · world)
 *   ray dir   = (M[6], M[7], M[8])          (R[2,:])
 * Ray parameterisation (R^T applied to (u, v, s)):
 *   x(s) = u*M[0] + v*M[3] + s*M[6]
 *   y(s) = u*M[1] + v*M[4] + s*M[7]
 *   z(s) = u*M[2] + v*M[5] + s*M[8]
 * Rays may cross all three faces, so ray length covers the full 3D diagonal.
 *
 * Border-zero semantics matching CUDA cudaAddressModeBorder + linear filter:
 * a sample at fx=Nx-1 must still return vol[Nx-1] (the upper neighbour
 * vol[Nx] is treated as zero, weighted by wx=0). This requires loose bounds
 * [-1, N) and per-corner safe loads — a strict fx<Nx-1 check would zero the
 * last slice entirely.
 */
#define SAMP3(ix, iy, iz)                                                  \
    (((ix) >= 0 && (ix) < Nx && (iy) >= 0 && (iy) < Ny &&                  \
      (iz) >= 0 && (iz) < Nz)                                              \
     ? volume[((size_t)(iy) * Nz + (iz)) * Nx + (ix)] : 0.0f)

void forward_project_3d_cpu(
    const float *volume,     /* [Ny, Nz, Nx] */
    float       *sinogram,   /* [n_angles, Ny, N_det] */
    const float *matrices,   /* [n_angles*9] row-major */
    int Ny, int Nz, int Nx,
    int n_angles, int N_det
) {
    float half_det = (N_det - 1) * 0.5f;
    float half_x   = (Nx   - 1) * 0.5f;
    float half_y   = (Ny   - 1) * 0.5f;
    float half_z   = (Nz   - 1) * 0.5f;
    int ray_len = (int)ceilf(0.5f * sqrtf((float)(Nx*Nx + Ny*Ny + Nz*Nz))) + 1;

    for (int ia = 0; ia < n_angles; ia++) {
        const float *M = matrices + (size_t)ia * 9;
        float dxs = M[6], dys = M[7], dzs = M[8];

        for (int iv = 0; iv < Ny; iv++) {
            float v = iv - half_y;

            for (int iu = 0; iu < N_det; iu++) {
                float u  = iu - half_det;
                float ox = u * M[0] + v * M[3];
                float oy = u * M[1] + v * M[4];
                float oz = u * M[2] + v * M[5];

                float sum = 0.0f;
                for (int step = -ray_len; step <= ray_len; step++) {
                    float s  = (float)step;
                    float fx = ox + s * dxs + half_x;
                    float fy = oy + s * dys + half_y;
                    float fz = oz + s * dzs + half_z;

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

                    float v000 = SAMP3(ix0, iy0, iz0);
                    float v001 = SAMP3(ix1, iy0, iz0);
                    float v010 = SAMP3(ix0, iy0, iz1);
                    float v011 = SAMP3(ix1, iy0, iz1);
                    float v100 = SAMP3(ix0, iy1, iz0);
                    float v101 = SAMP3(ix1, iy1, iz0);
                    float v110 = SAMP3(ix0, iy1, iz1);
                    float v111 = SAMP3(ix1, iy1, iz1);

                    sum += uy * uz * (ux * v000 + wx * v001)
                         + uy * wz * (ux * v010 + wx * v011)
                         + wy * uz * (ux * v100 + wx * v101)
                         + wy * wz * (ux * v110 + wx * v111);
                }

                sinogram[((size_t)ia * Ny + iv) * N_det + iu] = sum;
            }
        }
    }
}

/* ── 3D rotation-matrix back-projection (CPU) ──────────────────────────────
 * Voxel-driven gather (mirrors back_project_rotmat_cuda); each voxel reads
 * the bilinearly-interpolated sinogram value at its rotated detector position
 * and accumulates over angles. Approximate adjoint of forward_project_3d_cpu
 * — see CLAUDE.md note that CPU/CUDA BP outputs are not expected to match
 * numerically. Border-zero semantics: a voxel mapping to u_det == N_det-1
 * still contributes the last detector column.
 */
void back_project_3d_cpu(
    const float *sinogram,   /* [n_angles, Ny, N_det] */
    float       *volume,     /* [Ny, Nz, Nx] */
    const float *matrices,   /* [n_angles*9] row-major */
    int Ny, int Nz, int Nx,
    int n_angles, int N_det
) {
    float half_det = (N_det - 1) * 0.5f;
    float half_x   = (Nx   - 1) * 0.5f;
    float half_y   = (Ny   - 1) * 0.5f;
    float half_z   = (Nz   - 1) * 0.5f;

    memset(volume, 0, (size_t)Ny * Nz * Nx * sizeof(float));

    for (int iy = 0; iy < Ny; iy++) {
        float y = iy - half_y;
        for (int iz = 0; iz < Nz; iz++) {
            float z = iz - half_z;
            for (int ix = 0; ix < Nx; ix++) {
                float x = ix - half_x;
                float val = 0.0f;

                for (int ia = 0; ia < n_angles; ia++) {
                    const float *M = matrices + (size_t)ia * 9;
                    float u_det = M[0]*x + M[1]*y + M[2]*z + half_det;
                    float v_det = M[3]*x + M[4]*y + M[5]*z + half_y;

                    if (u_det <= -1.0f || u_det >= (float)N_det ||
                        v_det <= -1.0f || v_det >= (float)Ny)
                        continue;

                    int iu0 = (int)floorf(u_det), iv0 = (int)floorf(v_det);
                    int iu1 = iu0 + 1,            iv1 = iv0 + 1;
                    float wu = u_det - (float)iu0;
                    float wv = v_det - (float)iv0;

                    const float *layer = sinogram + (size_t)ia * Ny * N_det;
                    #define SSAMP(iu, iv) \
                        (((iu) >= 0 && (iu) < N_det && (iv) >= 0 && (iv) < Ny) \
                         ? layer[(size_t)(iv) * N_det + (iu)] : 0.0f)
                    val += (1.0f - wv) * (1.0f - wu) * SSAMP(iu0, iv0)
                         + (1.0f - wv) *          wu  * SSAMP(iu1, iv0)
                         +          wv  * (1.0f - wu) * SSAMP(iu0, iv1)
                         +          wv  *          wu  * SSAMP(iu1, iv1);
                    #undef SSAMP
                }

                volume[((size_t)iy * Nz + iz) * Nx + ix] = val;
            }
        }
    }
    (void)half_z;  /* unused — declared for parity with CUDA path */
}

/* ── gradient of 3D rotmat FP w.r.t. rotation matrix (CPU) ─────────────────
 * Mirrors grad_rotmat_cuda exactly. Detector-driven loop accumulates per-ray
 * partials of all 9 matrix entries, no reduction needed (pure serial).
 *
 * Per ray step (s = step):
 *   ∂tx/∂M[{0,3,6}] = (u, v, s)
 *   ∂ty/∂M[{1,4,7}] = (u, v, s)
 *   ∂tz/∂M[{2,5,8}] = (u, v, s)
 * with spatial derivatives ∂vol/∂{x,y,z} from analytic trilinear weights.
 */
void grad_rotmat_cpu(
    const float *volume,         /* [Ny, Nz, Nx] */
    const float *grad_sino,      /* [n_angles, Ny, N_det] */
    const float *matrices,       /* [n_angles*9] */
    float       *grad_matrices,  /* [n_angles*9] zeroed by this fn */
    int Ny, int Nz, int Nx,
    int n_angles, int N_det
) {
    float half_det = (N_det - 1) * 0.5f;
    float half_x   = (Nx   - 1) * 0.5f;
    float half_y   = (Ny   - 1) * 0.5f;
    float half_z   = (Nz   - 1) * 0.5f;
    int ray_len = (int)ceilf(0.5f * sqrtf((float)(Nx*Nx + Ny*Ny + Nz*Nz))) + 1;

    for (int k = 0; k < n_angles * 9; k++) grad_matrices[k] = 0.0f;

    for (int ia = 0; ia < n_angles; ia++) {
        const float *M = matrices + (size_t)ia * 9;
        float *gM = grad_matrices + (size_t)ia * 9;
        float dxs = M[6], dys = M[7], dzs = M[8];

        for (int iv = 0; iv < Ny; iv++) {
            float v = iv - half_y;
            for (int iu = 0; iu < N_det; iu++) {
                float gs = grad_sino[((size_t)ia * Ny + iv) * N_det + iu];
                if (gs == 0.0f) continue;

                float u  = iu - half_det;
                float ox = u * M[0] + v * M[3];
                float oy = u * M[1] + v * M[4];
                float oz = u * M[2] + v * M[5];

                for (int step = -ray_len; step <= ray_len; step++) {
                    float s  = (float)step;
                    float fx = ox + s * dxs + half_x;
                    float fy = oy + s * dys + half_y;
                    float fz = oz + s * dzs + half_z;

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

                    float v000 = SAMP3(ix0, iy0, iz0);
                    float v001 = SAMP3(ix1, iy0, iz0);
                    float v010 = SAMP3(ix0, iy0, iz1);
                    float v011 = SAMP3(ix1, iy0, iz1);
                    float v100 = SAMP3(ix0, iy1, iz0);
                    float v101 = SAMP3(ix1, iy1, iz0);
                    float v110 = SAMP3(ix0, iy1, iz1);
                    float v111 = SAMP3(ix1, iy1, iz1);

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

#undef SAMP3

void genfire_gridding_2d_cpu(
    const float *fk_real, const float *fk_imag,
    const float *cos_a,   const float *sin_a,
    float *grid_real, float *grid_imag, float *weight_grid,
    int bigN, int Ny, int n_angles, float cutoff_dist
) {
    int    half  = bigN / 2;
    size_t slice = (size_t)bigN * bigN;

    for (int ia = 0; ia < n_angles; ia++) {
        float ca = cos_a[ia], sa = sin_a[ia];
        for (int iy = 0; iy < Ny; iy++) {
            const float *row_r = fk_real + ((size_t)ia * Ny + iy) * bigN;
            const float *row_i = fk_imag + ((size_t)ia * Ny + iy) * bigN;
            float *gr = grid_real   + (size_t)iy * slice;
            float *gi = grid_imag   + (size_t)iy * slice;
            float *gw = weight_grid + (size_t)iy * slice;

            for (int ixi = 0; ixi < bigN; ixi++) {
                float xi   = (float)(ixi - half);
                float kx   = xi * ca;
                float kz   = xi * sa;
                int   ikx  = (int)roundf(kx);
                int   ikz  = (int)roundf(kz);
                float dx   = kx - (float)ikx;
                float dz   = kz - (float)ikz;
                float dist = sqrtf(dx * dx + dz * dz);
                if (dist > cutoff_dist)
                    continue;
                int gx = ikx + half;
                int gz = ikz + half;
                if (gx < 0 || gx >= bigN || gz < 0 || gz >= bigN)
                    continue;
                float  w   = 1.0f / (dist + 1e-5f);
                size_t idx = (size_t)gz * bigN + gx;
                gr[idx] += w * row_r[ixi];
                gi[idx] += w * row_i[ixi];
                gw[idx] += w;
            }
        }
    }
}

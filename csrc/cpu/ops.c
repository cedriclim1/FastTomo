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

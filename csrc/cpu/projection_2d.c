#include "ops.h"
#include "common.h"
#include "../include/proj_geom.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>

void forward_project_cpu(
    const float *volume,
    float       *sinogram,
    const float *cos_a,
    const float *sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det
) {
    ProjGeom2D g = proj_geom_2d(Ny, Nz, Nx, n_angles, N_det);

    for (int ia = 0; ia < n_angles; ia++) {
        float ca = cos_a[ia], sa = sin_a[ia];

        for (int iy = 0; iy < Ny; iy++) {
            const float *slice = volume + (size_t)iy * Nz * Nx;

            for (int id = 0; id < N_det; id++) {
                float t   = id - g.half_det;   /* detector coord (pixels, centred) */
                float sum = 0.0f;

                for (int step = -g.ray_len; step <= g.ray_len; step++) {
                    float s  = (float)step;
                    /* point on ray at parameter s */
                    float x  = t * ca - s * sa;
                    float z  = t * sa + s * ca;
                    /* convert to 0-based array indices */
                    float fx = x + g.half_x;
                    float fz = z + g.half_z;

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
    ProjGeom2D g = proj_geom_2d(Ny, Nz, Nx, n_angles, N_det);

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
                float t    = id - g.half_det;
                float sval = sinogram[((size_t)ia * Ny + iy) * N_det + id];

                for (int step = -g.ray_len; step <= g.ray_len; step++) {
                    float s  = (float)step;
                    float fx = (t * ca - s * sa) + g.half_x;
                    float fz = (t * sa + s * ca) + g.half_z;

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
    ProjGeom2D g = proj_geom_2d(Ny, Nz, Nx, n_angles, N_det);

    for (int ia = 0; ia < n_angles; ia++)
        grad_angles[ia] = 0.0f;

    for (int ia = 0; ia < n_angles; ia++) {
        float ca = cos_a[ia], sa = sin_a[ia];
        float g_theta = 0.0f;

        for (int iy = 0; iy < Ny; iy++) {
            const float *slice = volume + (size_t)iy * Nz * Nx;

            for (int id = 0; id < N_det; id++) {
                float t  = id - g.half_det;
                float gs = grad_sino[((size_t)ia * Ny + iy) * N_det + id];
                if (gs == 0.0f) continue;

                for (int step = -g.ray_len; step <= g.ray_len; step++) {
                    float s  = (float)step;
                    float x  = t * ca - s * sa;
                    float z  = t * sa + s * ca;
                    float fx = x + g.half_x;
                    float fz = z + g.half_z;

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

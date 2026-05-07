#include "ops.h"
#include <math.h>
#include <stddef.h>

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

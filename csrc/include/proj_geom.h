#pragma once
#include <math.h>

typedef struct {
    int   Ny, Nz, Nx, n_angles, N_det, ray_len;
    float half_det, half_x, half_z;
} ProjGeomParallel;

static inline ProjGeomParallel proj_geom_parallel(int Ny, int Nz, int Nx, int n_a, int N_det) {
    ProjGeomParallel g = { Ny, Nz, Nx, n_a, N_det,
                     (int)ceilf(0.5f * sqrtf((float)(Nx*Nx + Nz*Nz))) + 1,
                     0.5f * (N_det - 1), 0.5f * (Nx - 1), 0.5f * (Nz - 1) };
    return g;
}

typedef struct {
    int   Ny, Nz, Nx, n_angles, N_det, ray_len;
    float half_det, half_x, half_y, half_z;
} ProjGeomRotmat;

static inline ProjGeomRotmat proj_geom_rotmat(int Ny, int Nz, int Nx, int n_a, int N_det) {
    ProjGeomRotmat g = { Ny, Nz, Nx, n_a, N_det,
                     (int)ceilf(0.5f * sqrtf((float)(Nx*Nx + Ny*Ny + Nz*Nz))) + 1,
                     0.5f * (N_det - 1), 0.5f * (Nx - 1),
                     0.5f * (Ny   - 1), 0.5f * (Nz - 1) };
    return g;
}

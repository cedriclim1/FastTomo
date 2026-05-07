#pragma once
#include <math.h>

typedef struct {
    int   Ny, Nz, Nx, n_angles, N_det, ray_len;
    float half_det, half_x, half_z;
} ProjGeom2D;

static inline ProjGeom2D proj_geom_2d(int Ny, int Nz, int Nx, int n_a, int N_det) {
    ProjGeom2D g = { Ny, Nz, Nx, n_a, N_det,
                     (int)ceilf(0.5f * sqrtf((float)(Nx*Nx + Nz*Nz))) + 1,
                     0.5f * (N_det - 1), 0.5f * (Nx - 1), 0.5f * (Nz - 1) };
    return g;
}

typedef struct {
    int   Ny, Nz, Nx, n_angles, N_det, ray_len;
    float half_det, half_x, half_y, half_z;
} ProjGeom3D;

static inline ProjGeom3D proj_geom_3d(int Ny, int Nz, int Nx, int n_a, int N_det) {
    ProjGeom3D g = { Ny, Nz, Nx, n_a, N_det,
                     (int)ceilf(0.5f * sqrtf((float)(Nx*Nx + Ny*Ny + Nz*Nz))) + 1,
                     0.5f * (N_det - 1), 0.5f * (Nx - 1),
                     0.5f * (Ny   - 1), 0.5f * (Nz - 1) };
    return g;
}

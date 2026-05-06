#ifndef FASTTOMO_CUDA_OPS_H
#define FASTTOMO_CUDA_OPS_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Parallel-beam forward projection (CUDA).
 *
 * All pointers are device pointers (cudaMalloc'd).
 * Shapes and geometry conventions mirror the CPU version:
 *   volume   : [Ny, Nz, Nx]
 *   sinogram : [n_angles, Ny, N_det]  — written (not accumulated)
 *   cos_a / sin_a : [n_angles]
 *
 * Kernel strategy: detector-driven, no atomics.
 * Each thread owns one (ia, iy, id) output sample and marches
 * along the ray accumulating bilinearly-interpolated volume samples.
 */
void forward_project_cuda(
    const float *d_volume,
    float       *d_sinogram,
    const float *d_cos_a,
    const float *d_sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det
);

/*
 * Parallel-beam back-projection (CUDA).
 *
 * Voxel-driven: each thread owns one (iy, iz, ix) voxel and
 * loops over all angles reading linearly-interpolated sinogram values.
 * The volume is overwritten (not accumulated from a previous value).
 */
void back_project_cuda(
    const float *d_sinogram,
    float       *d_volume,
    const float *d_cos_a,
    const float *d_sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det
);

/*
 * 3D rotation-matrix forward projection (CUDA).
 *
 * d_matrices: device pointer to [n_angles * 9] row-major rotation matrices.
 *   M[i*9 + row*3 + col] = R_i[row][col]
 *   detector-u = R[0,:] · (x,y,z),  detector-v = R[1,:] · (x,y,z)
 *
 * Volume wrapped in tex3D (trilinear + border); rays may cross Y boundaries.
 * Thread layout: (⌈N_det/16⌉, ⌈Ny/16⌉, n_angles) × (16,16,1).
 */
void forward_project_rotmat_cuda(
    const float *d_volume,
    float       *d_sinogram,
    const float *d_matrices,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det
);

/*
 * 3D rotation-matrix back-projection (CUDA).
 *
 * Voxel-driven gather: each thread reads sinogram via tex2DLayered at the
 * rotated detector position for each angle. Rotation matrices loaded into
 * shared memory (9 * n_angles floats ≤ 48 KB for n_angles ≤ ~1300).
 * Thread layout: (⌈Nx/8⌉, ⌈Nz/8⌉, ⌈Ny/4⌉) × (8,8,4).
 */
void back_project_rotmat_cuda(
    const float *d_sinogram,
    float       *d_volume,
    const float *d_matrices,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det
);

/*
 * GENFIRE Fourier gridding (CUDA).
 *
 * All pointers are device pointers.
 * Parallelises over (iy, ia, ixi); uses atomicAdd for scatter.
 * grid_real / grid_imag / weight_grid must be zeroed before the call.
 */
void genfire_gridding_2d_cuda(
    const float *fk_real,
    const float *fk_imag,
    const float *cos_a,
    const float *sin_a,
    float       *grid_real,
    float       *grid_imag,
    float       *weight_grid,
    int bigN, int Ny, int n_angles,
    float cutoff_dist
);

#ifdef __cplusplus
}
#endif

#endif /* FASTTOMO_CUDA_OPS_H */

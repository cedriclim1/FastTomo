#ifndef FASTTOMO_CPU_OPS_H
#define FASTTOMO_CPU_OPS_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Parallel-beam forward projection (CPU).
 *
 * volume   : [Ny, Nz, Nx]  float32, C-contiguous (Nx is fastest)
 * sinogram : [n_angles, Ny, N_det]  float32, caller-allocated and zeroed
 * cos_a    : [n_angles]  precomputed cos(angles)
 * sin_a    : [n_angles]  precomputed sin(angles)
 *
 * Geometry: tilt axis = Y. At angle θ, beam integrates in direction
 * d = (−sinθ, cosθ) in the XZ plane. Detector coordinate of voxel (x,z)
 * is t = x·cosθ + z·sinθ, with x ∈ [-(Nx-1)/2, (Nx-1)/2] and similarly z.
 * Detector bins are centred: id=0 ↔ t=-(N_det-1)/2.
 *
 * Integration uses 1-pixel steps along the ray with bilinear interpolation.
 */
void forward_project_cpu(
    const float *volume,
    float       *sinogram,
    const float *cos_a,
    const float *sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det
);

/*
 * Parallel-beam back-projection (CPU).
 *
 * sinogram : [n_angles, Ny, N_det]  float32
 * volume   : [Ny, Nz, Nx]  float32, caller-allocated — overwritten, NOT accumulated
 * cos_a / sin_a : [n_angles] precomputed trig values
 *
 * For each voxel, accumulates linearly-interpolated sinogram values over all angles.
 */
void back_project_cpu(
    const float *sinogram,
    float       *volume,
    const float *cos_a,
    const float *sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det
);

/*
 * GENFIRE Fourier gridding (CPU).
 *
 * Scatters 1D Fourier slices (fftshifted, DC at index bigN/2) onto a 2D
 * Fourier grid using inverse-distance weighting.
 *
 * fk_real / fk_imag : [n_angles, Ny, bigN]  — fftshifted 1D FFTs, C-contiguous
 * cos_a / sin_a     : [n_angles]
 * grid_real / grid_imag / weight_grid : [Ny, bigN, bigN]  — zeroed by caller
 *
 * For frequency index ixi, the centered frequency is xi = ixi - bigN/2.
 * At angle ia the grid point is (xi*cos_a[ia], xi*sin_a[ia]).
 * Each point within cutoff_dist of a grid cell contributes value/distance.
 * Normalization by accumulated weights is left to the caller.
 */
/*
 * Gradient of forward projection w.r.t. tilt angles (CPU).
 *
 * Computes ∂L/∂θ_ia for each angle, given the upstream gradient on the
 * sinogram.  Uses the identity ∂x/∂θ = −z, ∂z/∂θ = x at each ray step,
 * combined with bilinear-weighted finite differences for ∂vol/∂x and ∂vol/∂z.
 *
 * grad_angles : [n_angles] — zeroed by caller; returned in radians
 */
void grad_angle_cpu(
    const float *volume,
    const float *grad_sino,
    float       *grad_angles,
    const float *cos_a,
    const float *sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det
);

/*
 * 3D rotation-matrix forward / back projection (CPU).
 *
 * Mirrors the CUDA texture-based projector pair (forward_project_rotmat_cuda
 * / back_project_rotmat_cuda). Detector-driven trilinear FP, voxel-driven
 * gather BP. matrices: [n_angles*9] row-major; M[ia*9 + row*3 + col].
 *
 * Slower than the CUDA path by O(10×) — useful for testing on hosts without
 * a GPU and as a reference for the autograd backward.
 */
void forward_project_3d_cpu(
    const float *volume,
    float       *sinogram,
    const float *matrices,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det
);

void back_project_3d_cpu(
    const float *sinogram,
    float       *volume,
    const float *matrices,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det
);

/*
 * Gradient of 3D rotmat forward projection w.r.t. rotation matrix entries
 * (CPU). grad_matrices: [n_angles*9] zeroed by this function before
 * accumulating. Used by _FP3D's autograd backward on CPU; PyTorch chains
 * the result through _euler_zyz_to_rotmat to yield ∂L/∂euler.
 */
void grad_rotmat_cpu(
    const float *volume,
    const float *grad_sino,
    const float *matrices,
    float       *grad_matrices,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det
);

void genfire_gridding_2d_cpu(
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

#endif /* FASTTOMO_CPU_OPS_H */

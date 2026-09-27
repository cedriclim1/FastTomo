#ifndef FASTTOMO_CPU_OPS_H
#define FASTTOMO_CPU_OPS_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Parallel-beam forward projection (CPU).
 *
 * volume   : [Ny, Nz, Nx]  float32, C-contiguous (Nx is fastest)
 * sinogram : [n_angles, Ny, N_det]  float32, caller-allocated; every
 *            element is written
 * cos_a    : [n_angles]  precomputed cos(angles)
 * sin_a    : [n_angles]  precomputed sin(angles)
 *
 * Geometry: tilt axis = Y. At angle θ, beam integrates in direction
 * d = (−sinθ, cosθ) in the XZ plane. Detector coordinate of voxel (x,z)
 * is t = x·cosθ + z·sinθ, with x ∈ [-(Nx-1)/2, (Nx-1)/2] and similarly z.
 * Detector bins are centred: id=0 ↔ t=-(N_det-1)/2.
 *
 * Integration uses 1-pixel steps along the ray with bilinear interpolation;
 * a sample counts only if its whole 2x2 stencil is inside the slice.
 * Y-batched (FT_YB slices per SIMD loop, common.h). Returns 0, or -1 if the
 * scratch allocation (Nz*Nx*min(Ny, FT_YB) floats, none for Ny = 1) fails.
 */
int forward_project_parallel_cpu(
    const float *vol,
    float       *sino,
    const float *cos_a,
    const float *sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    int num_threads
);

/*
 * Parallel-beam back-projection (CPU): the ray-driven scatter that is the
 * exact transpose of forward_project_parallel_cpu (every ray sample adds its
 * sinogram value, bilinearly weighted, to its 2x2 voxel stencil).
 *
 * sinogram : [n_angles, Ny, N_det]  float32
 * volume   : [Ny, Nz, Nx]  float32, caller-allocated — overwritten, NOT accumulated
 * cos_a / sin_a : [n_angles] precomputed trig values
 *
 * Y-batched, OpenMP over Z bands; the result does not depend on
 * num_threads. Returns 0, or -1 if the scratch allocation
 * ((Nz*Nx + n_angles*N_det) * min(Ny, FT_YB) floats, none for Ny = 1) fails.
 */
int back_project_parallel_cpu(
    const float *sino,
    float       *vol,
    const float *cos_a,
    const float *sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    int num_threads
);

/*
 * Voxel-driven (gather) parallel back-projection (CPU), the operator of the
 * weighted back-projection (W-BP / wSIRT): for every voxel and view, the
 * sinogram row linearly interpolated at u = x cos + z sin + (N_det-1)/2,
 * zero outside the detector. Same operator as the CUDA
 * back_project_parallel_kernel, with exact fp32 weights, and bit-for-bit
 * the PyTorch reference `_bp_gather`. NOT the adjoint of
 * forward_project_parallel_cpu (that is back_project_parallel_cpu).
 *
 * sino : [n_angles, Ny, N_det] float32; vol : [Ny, Nz, Nx], overwritten.
 * Y-batched for Ny >= 2. Returns 0, or -1 if the scratch allocation fails
 * (padded sinogram rows and an x*cos table: n_angles*(min(Ny, FT_YB)*(N_det+2)
 * + Nx) floats plus Nx*min(Ny, FT_YB) per thread; Ny = 1:
 * n_angles*(N_det+2+Nx) floats).
 */
int back_project_gather_parallel_cpu(
    const float *sino,
    float       *vol,
    const float *cos_a,
    const float *sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    int num_threads
);

/*
 * Gradient of forward projection w.r.t. tilt angles (CPU).
 *
 * Computes ∂L/∂θ_ia for each angle, given the upstream gradient on the
 * sinogram.  Uses the identity ∂x/∂θ = −z, ∂z/∂θ = x at each ray step,
 * combined with bilinear-weighted finite differences for ∂vol/∂x and ∂vol/∂z.
 *
 * grad_angles : [n_angles] — overwritten; per radian
 *
 * Y-batched; fp64 accumulation in an order fixed by the geometry, so the
 * result does not depend on num_threads. Returns 0, or -1 if the scratch
 * allocation (Nz*Nx*min(Ny, FT_YB) floats for Ny > 1, plus
 * n_angles*(ceil(N_det/32)*16 + 1) doubles) fails.
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
void forward_project_rotmat_cpu(
    const float *vol,
    float       *sino,
    const float *mats,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    int num_threads
);

void back_project_rotmat_cpu(
    const float *sino,
    float       *vol,
    const float *mats,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    int num_threads
);

/*
 * Gradient of 3D rotmat forward projection w.r.t. rotation matrix entries
 * (CPU). grad_matrices: [n_angles*9] zeroed by this function before
 * accumulating. Used by _FPRotmat's autograd backward on CPU; PyTorch chains
 * the result through _euler_zyz_to_rotmat to yield ∂L/∂euler.
 */
void grad_rotmat_cpu(
    const float *vol,
    const float *grad_sino,
    const float *mats,
    float       *grad_mats,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    int num_threads
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

/*
 * MBIR voxel-driven bilinear forward projection (CPU).
 *
 * Distinct from `forward_project_parallel_cpu` — this uses the voxel-driven scatter
 * model A_MBIR that matches the column-A used in `mbir_icd_sweep_cpu`.
 * For voxel (ix, iz, iy) at angle θ_k the detector coord is
 *   u = (ix - half_x)·cos θ_k - (iz - half_z)·sin θ_k + half_det,
 * and the voxel scatters to detector pixels (floor u, floor u + 1) at iy
 * with bilinear weights (1-wu, wu). Sinogram is overwritten, not
 * accumulated. */
void mbir_forward_project_cpu(
    const float *volume,
    float       *sinogram,
    const float *cos_a,
    const float *sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    int num_threads
);

/*
 * ICD sweep of one outer iteration of MBIR-AM-PE (CPU reference).
 *
 *   volume, e_sino                       : in-place outputs
 *   volume_snapshot                      : read-only snapshot for qGGMRF
 *   perm[Ny, Nz*Nx]                      : per-Y-slice voxel permutation
 *   lambdas[n_angles, Ny, N_det]         : per-pixel Λ data weights; NULL → 1
 *
 * Y-slices are parallelized via OpenMP. All 26 qGGMRF neighbour reads
 * (in-slice ones included) use the snapshot, so slices are independent
 * and the prior is Jacobi. Within a slice the data term is sequential
 * Gauss-Seidel ICD via the incremental error sinogram.
 *
 * The per-tilt offset and noise-variance updates are done by the caller in
 * Python (`_mbir_update_offset_variance`), as on the CUDA path. */
int mbir_icd_sweep_cpu(
    float       *volume,
    float       *e_sino,
    const float *volume_snapshot,
    const int   *perm,
    const float *cos_a,
    const float *sin_a,
    const float *lambdas,
    int Ny, int Nz, int Nx, int n_angles, int N_det,
    float qggmrf_p, float qggmrf_q, float qggmrf_c, float sigma_f,
    float huber_T, float huber_delta, float sigma,
    int num_threads
);

#ifdef __cplusplus
}
#endif

#endif /* FASTTOMO_CPU_OPS_H */

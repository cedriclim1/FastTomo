#ifndef FASTTOMO_CUDA_OPS_H
#define FASTTOMO_CUDA_OPS_H

#include <cuda_runtime.h>
#include "texture_cache.h"

/*
 * All CUDA entry points run on a caller-supplied stream so the kernels
 * compose cleanly with PyTorch's stream model. None of them call
 * cudaDeviceSynchronize — the caller (PyTorch) syncs implicitly when the
 * output tensor is read.
 *
 * Every host-callable function may throw std::runtime_error on a CUDA error.
 *
 * Cos/sin tables (single-axis kernels) are passed as device pointers so the
 * trig precompute can stay on the GPU.
 *
 * The *_cached variants take a pre-allocated TextureCache* (sized to match
 * the input shape). They cudaMemcpy3DAsync the caller's data into the
 * cache's existing cudaArray and launch the kernel — no per-call cudaArray
 * allocation. Used by ParallelProjector / RotmatProjector for tight loops.
 */

/* The 4 in-scope projection ops + their cached variants accept a runtime
 * `io_dtype` integer (0 = fp32, 1 = fp16) and use void* for I/O tensors.
 * The dispatcher reinterprets the pointers based on io_dtype and calls a
 * templated impl. cos/sin tables and rotation matrices stay fp32 in all
 * paths. Out-of-scope ops below (RESIRE, grad_*) keep their fp32-only
 * signatures. */
void forward_project_cuda(
    int io_dtype,
    const void  *d_volume,
    void        *d_sinogram,
    const float *d_cos_a,
    const float *d_sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
);

void back_project_cuda(
    int io_dtype,
    const void  *d_sinogram,
    void        *d_volume,
    const float *d_cos_a,
    const float *d_sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
);

/*
 * 3D rotation-matrix projection: matrices are device-side
 * [n_angles * 9] row-major; M[ia*9 + row*3 + col] = R_ia[row][col].
 */
void forward_project_rotmat_cuda(
    int io_dtype,
    const void  *d_volume,
    void        *d_sinogram,
    const float *d_matrices,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
);

void back_project_rotmat_cuda(
    int io_dtype,
    const void  *d_sinogram,
    void        *d_volume,
    const float *d_matrices,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
);

/* ── cached variants ──────────────────────────────────────────────────────
 * `vol_cache` / `sino_cache` must already be sized to the input AND match
 * the io_dtype. Each call uploads the caller's data into the cache's
 * cudaArray, then launches the kernel. Output buffer is written through a
 * regular device pointer.
 */
void forward_project_cuda_cached(
    int io_dtype,
    TextureCache *vol_cache,
    const void   *d_volume,
    void         *d_sinogram,
    const float  *d_cos_a,
    const float  *d_sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
);

void back_project_cuda_cached(
    int io_dtype,
    TextureCache *sino_cache,
    const void   *d_sinogram,
    void         *d_volume,
    const float  *d_cos_a,
    const float  *d_sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
);

void forward_project_rotmat_cuda_cached(
    int io_dtype,
    TextureCache *vol_cache,
    const void   *d_volume,
    void         *d_sinogram,
    const float  *d_matrices,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
);

void back_project_rotmat_cuda_cached(
    int io_dtype,
    TextureCache *sino_cache,
    const void   *d_sinogram,
    void         *d_volume,
    const float  *d_matrices,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
);

/*
 * RESIRE voxel-driven scatter/gather with 4-point supersampling. Output
 * normalisation (÷4) is applied by the Python caller. d_sinogram (FP) and
 * d_volume (BP) must be zeroed before the call. The host loops over
 * n_angles internally, launching one kernel per projection on the supplied
 * stream — BP relies on the same-stream serialisation to accumulate without
 * atomics.
 */
void forward_project_resire_cuda(
    const float *d_volume,
    float       *d_sinogram,
    const float *d_matrices,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
);

void back_project_resire_cuda(
    const float *d_sinogram,
    float       *d_volume,
    const float *d_matrices,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
);

/*
 * Gradient of 3D rotation-matrix forward projection w.r.t. the rotation
 * matrix entries. d_grad_matrices is [n_angles*9] row-major, must be zeroed
 * by the caller. Used by _FP3D's autograd backward; the host chains the
 * result through _euler_zyz_to_rotmat (pure-PyTorch) to yield ∂L/∂euler.
 */
void grad_rotmat_cuda(
    const float *d_volume,
    const float *d_grad_sino,
    const float *d_matrices,
    float       *d_grad_matrices,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
);

/*
 * Gradient of forward projection w.r.t. tilt angles.
 * d_grad_angles must be zeroed by the caller; result is in radians.
 */
void grad_angle_cuda(
    const float *d_volume,
    const float *d_grad_sino,
    const float *d_cos_a,
    const float *d_sin_a,
    float       *d_grad_angles,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
);

void genfire_gridding_2d_cuda(
    const float *fk_real,
    const float *fk_imag,
    const float *cos_a,
    const float *sin_a,
    float       *grid_real,
    float       *grid_imag,
    float       *weight_grid,
    int bigN, int Ny, int n_angles,
    float cutoff_dist,
    cudaStream_t stream
);

#endif /* FASTTOMO_CUDA_OPS_H */

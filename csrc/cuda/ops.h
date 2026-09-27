#ifndef FASTTOMO_CUDA_OPS_H
#define FASTTOMO_CUDA_OPS_H

#include <cuda_runtime.h>
#include "texture_cache.h"

/*
 * All CUDA entry points launch on a caller-supplied stream. Synchronisation:
 *   - The *_cached entry points and the texture-free kernels (grad_angle,
 *     TV, GENFIRE, RESIRE, MBIR forward and ICD sweep) never synchronize;
 *     PyTorch orders the work through the stream.
 *   - The one-shot entry points that build their own texture
 *     (forward/back_project_parallel_cuda, forward/back_project_rotmat_cuda,
 *     grad_rotmat_cuda) call cudaStreamSynchronize on their stream before
 *     releasing it (ScopedTextureCache, texture_cache.h). cudaFreeArray would synchronize anyway.
 *   - The Python-side `tc_destroy` binding synchronizes the device.
 *
 * Every host-callable function may throw: std::runtime_error on a CUDA error,
 * std::invalid_argument (Python ValueError) for a cache that does not match
 * the call (tc_validate) or an extent beyond the device texture limits.
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
void forward_project_parallel_cuda(
    int io_dtype,
    const void  *vol,
    void        *sino,
    const float *cos_a,
    const float *sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
);

void back_project_parallel_cuda(
    int io_dtype,
    const void  *sino,
    void        *vol,
    const float *cos_a,
    const float *sin_a,
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
    const void  *vol,
    void        *sino,
    const float *mats,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
);

void back_project_rotmat_cuda(
    int io_dtype,
    const void  *sino,
    void        *vol,
    const float *mats,
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
void forward_project_parallel_cuda_cached(
    int io_dtype,
    TextureCache *vol_cache,
    const void   *vol,
    void         *sino,
    const float  *cos_a,
    const float  *sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
);

void back_project_parallel_cuda_cached(
    int io_dtype,
    TextureCache *sino_cache,
    const void   *sino,
    void         *vol,
    const float  *cos_a,
    const float  *sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
);

void forward_project_rotmat_cuda_cached(
    int io_dtype,
    TextureCache *vol_cache,
    const void   *vol,
    void         *sino,
    const float  *mats,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
);

void back_project_rotmat_cuda_cached(
    int io_dtype,
    TextureCache *sino_cache,
    const void   *sino,
    void         *vol,
    const float  *mats,
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
 * matrix entries. grad_mats is [n_angles*9] row-major, must be zeroed
 * by the caller. Used by _FPRotmat's autograd backward; the host chains the
 * result through _euler_zyz_to_rotmat (pure-PyTorch) to yield ∂L/∂euler.
 */
void grad_rotmat_cuda(
    const float *vol,
    const float *grad_sino,
    const float *mats,
    float       *grad_mats,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
);

/*
 * Cached variant of grad_rotmat_cuda. vol_cache must be a TC_VOL_3D cache
 * already uploaded by a preceding forward_project_rotmat_cuda_cached call.
 * Uses vol_cache->tex_point (point-filter view) for bit-exact corner reads.
 * io_dtype: 0=fp32, 1=fp16 (controls which kernel template is instantiated).
 * grad_mats must be zeroed by the caller; result is accumulated via atomicAdd.
 */
void grad_rotmat_cuda_cached(
    int io_dtype,
    TextureCache *vol_cache,
    const float *grad_sino,
    const float *mats,
    float       *grad_mats,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
);

/*
 * Test hook: turn the ray clipping of forward_project_rotmat_kernel on (1,
 * the default) or off (0, the full [-ray_len, ray_len] step loop). Returns
 * the previous setting. Results are bit-identical either way; tests compare
 * the two paths with torch.equal. Process-wide, not thread-safe: tests only.
 */
int _debug_set_rotmat_ray_clip(int on);

/*
 * Gradient of forward projection w.r.t. tilt angles.
 * grad_angles must be zeroed by the caller; result is in radians.
 */
void grad_angle_cuda(
    const float *vol,
    const float *grad_sino,
    const float *cos_a,
    const float *sin_a,
    float       *grad_angles,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
);

/*
 * Fused 3-D isotropic Total-Variation loss.
 *   loss = mean( sqrt( dy² + dz² + dx² + eps ) )
 * over the (Ny-1)·(Nz-1)·(Nx-1) inner-corner grid. The forward writes Σ s
 * to d_acc[0] (deterministic two-pass reduction, fp64 above the per-lane
 * sums, no atomics); d_acc is left untouched (so must be zeroed by the
 * caller) when a dimension is < 2. d_partials is fp64 scratch of
 * tv_loss_iso_3d_workspace(Ny, Nz, Nx) elements (0 when a dimension is < 2).
 * The 1/N normalization is applied by the Python caller. Backward expects
 * `d_g_scaled[0]` to already contain `grad_out · 1/N` (single-element
 * device buffer). d_grad_vol must be zeroed by the caller (kernel writes
 * once per voxel).
 */
long long tv_loss_iso_3d_workspace(int Ny, int Nz, int Nx);

void tv_loss_iso_3d_cuda(
    const float *d_vol,
    float       *d_acc,
    double      *d_partials,
    int Ny, int Nz, int Nx,
    float eps,
    cudaStream_t stream
);

void tv_loss_iso_3d_grad_cuda(
    const float *d_vol,
    const float *d_g_scaled,
    float       *d_grad_vol,
    int Ny, int Nz, int Nx,
    float eps,
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

/*
 * MBIR voxel-driven bilinear scatter forward projection (CUDA). Distinct
 * from forward_project_parallel_cuda's ray-marching FP — this matches the column-A
 * model used by mbir_icd_sweep_cuda. Output sinogram is zeroed by the
 * wrapper before the kernel launches (atomicAdd scatter).
 */
void mbir_forward_project_cuda(
    const float *d_volume,
    float       *d_sinogram,
    const float *d_cos_a,
    const float *d_sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
);

/*
 * One outer iteration of MBIR-AM-PE (CUDA). In-place mutates d_volume and
 * d_e_sino. Per-tilt offset d_k and noise variance σ are updated host-side
 * in PyTorch by the Python wrapper — the kernel only does the ICD sweep.
 * Asynchronous on `stream` (no host synchronisation).
 *
 *   d_volume_snapshot : start-of-outer-iter copy of volume; the qGGMRF
 *                       26-neighbour values are read from it.
 *   d_lambdas         : per-pixel data weights or nullptr → uniform 1.
 *   d_perm            : [Ny, Nz*Nx] int32 explicit voxel order (each row a
 *                       permutation), or nullptr (the normal case): the
 *                       order is generated on the device from perm_key
 *                       (see mbir_perm_* in mbir_icd.cu).
 *   perm_key          : 64-bit key of this outer iteration's order.
 *   d_work            : scratch of Ny*Nz*Nx float4 (16 bytes per voxel,
 *                       16-byte aligned), reusable across calls.
 *   sigma             : current noise std for the Huber surrogate.
 */
void mbir_icd_sweep_cuda(
    float       *d_volume,
    float       *d_e_sino,
    const float *d_volume_snapshot,
    const float *d_lambdas,
    const int   *d_perm,
    unsigned long long perm_key,
    float       *d_work,
    const float *d_cos_a,
    const float *d_sin_a,
    int Ny, int Nz, int Nx, int n_angles, int N_det,
    float qggmrf_p, float qggmrf_q, float qggmrf_c, float sigma_f,
    float huber_T, float huber_delta, float sigma,
    cudaStream_t stream
);

/*
 * Test hook: the voxel order mbir_icd_sweep_cuda uses when d_perm is null,
 * written to d_out as [Ny, nvox] int32. Mirrors
 * fasttomo.algorithms.mbir._mbir_permutation bit for bit.
 */
void mbir_permutation_cuda(
    int *d_out, int Ny, int nvox, unsigned long long perm_key, cudaStream_t stream
);

#endif /* FASTTOMO_CUDA_OPS_H */

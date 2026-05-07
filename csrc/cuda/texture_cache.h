#ifndef FASTTOMO_CUDA_TEXTURE_CACHE_H
#define FASTTOMO_CUDA_TEXTURE_CACHE_H

#include <cuda_runtime.h>

/* Persistent cudaArray + cudaTextureObject pair, sized once and reused across
 * many forward/back-projection calls of the same shape. The hot path becomes
 *   tc_upload(cache, new_data_ptr, stream)   // cudaMemcpy3DAsync only
 *   forward_project_cuda_cached(cache, ...)  // kernel launch
 * with no per-call cudaMalloc3DArray or cudaCreateTextureObject. The original
 * non-cached entry points stay for one-shot users.
 *
 * Three geometries, one per place ops.cu uses a texture:
 *   TC_VOL_LAYERED  — FP scalar.   Ny layers of [Nz × Nx]. Source: [Ny,Nz,Nx].
 *   TC_SINO_LAYERED — BP scalar &  n_angles layers of [Ny × N_det].
 *                     BP rotmat.    Source: [n_angles, Ny, N_det].
 *   TC_VOL_3D       — FP rotmat.   True 3D (Nx, Nz, Ny). Source: [Ny,Nz,Nx].
 *
 * dim_a / dim_b / dim_c map onto the source-tensor dims in C order:
 *   vol  variants: dim_a=Ny,        dim_b=Nz, dim_c=Nx
 *   sino variant : dim_a=n_angles,  dim_b=Ny, dim_c=N_det
 */
typedef enum {
    TC_VOL_LAYERED  = 0,
    TC_SINO_LAYERED = 1,
    TC_VOL_3D       = 2
} tc_kind_t;

typedef struct {
    cudaArray_t          array;
    cudaTextureObject_t  tex;
    int                  dim_a;
    int                  dim_b;
    int                  dim_c;
    tc_kind_t            kind;
} TextureCache;

/* Allocate cudaArray + cudaTextureObject sized for (dim_a, dim_b, dim_c).
 * The returned pointer is heap-allocated; pass it to tc_destroy when done. */
TextureCache *tc_create(tc_kind_t kind, int dim_a, int dim_b, int dim_c);

/* Copy `src` (device-side, contiguous, matching the cache's recorded shape)
 * into the existing cudaArray. cudaMemcpy3DAsync only — no allocation.
 * src layout matches the tensor convention used at create time. */
void tc_upload(TextureCache *tc, const float *src, cudaStream_t stream);

/* cudaDestroyTextureObject + cudaFreeArray + free(tc). Safe on nullptr. */
void tc_destroy(TextureCache *tc);

#endif /* FASTTOMO_CUDA_TEXTURE_CACHE_H */

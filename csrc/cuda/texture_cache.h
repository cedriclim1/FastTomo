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
 *
 * The cache also remembers an I/O dtype:
 *   TC_DTYPE_FP32 — single-channel 32-bit float (existing path)
 *   TC_DTYPE_FP16 — 16-bit half-float (1-channel native, or 4-channel packed
 *                   when channels==4 — see below).
 *
 * Channels — 1 (default) or 4. The 4-channel mode mirrors TorchRadon's
 * batch-channel-packing trick: 4 consecutive Y-slices are packed into a
 * single 4-channel half texel so each tex2DLayered<float4> fetch returns 4
 * Y-slices' values at once. Per-(det,angle) thread does ~4× fewer texture
 * fetches and stores 4 outputs. For the vol textures the packing axis is
 * Ny (the texture's layer/depth dim); for the sino texture the packing
 * axis is Ny (the texture's height dim, since the layer dim is n_angles).
 * Constraint: the relevant axis must be divisible by 4. Float32 paths only
 * support channels==1.
 */
typedef enum {
    TC_VOL_LAYERED  = 0,
    TC_SINO_LAYERED = 1,
    TC_VOL_3D       = 2
} tc_kind_t;

typedef enum {
    TC_DTYPE_FP32 = 0,
    TC_DTYPE_FP16 = 1
} tc_dtype_t;

typedef struct {
    cudaArray_t           array;
    cudaTextureObject_t   tex;
    cudaSurfaceObject_t   surf;     /* used by 4-channel half upload kernel */
    int                   dim_a;
    int                   dim_b;
    int                   dim_c;
    int                   channels; /* 1 or 4 */
    tc_kind_t             kind;
    tc_dtype_t            dtype;
} TextureCache;

/* Allocate cudaArray + cudaTextureObject sized for (dim_a, dim_b, dim_c) with
 * the given I/O dtype and channel count (1 or 4). Channels==4 is only valid
 * with TC_DTYPE_FP16; the kind-specific Y-axis must be divisible by 4 (Ny for
 * vol kinds, Ny=dim_b for sino kind). The returned pointer is heap-allocated;
 * pass it to tc_destroy when done. */
TextureCache *tc_create(tc_kind_t kind, tc_dtype_t dtype, int channels,
                        int dim_a, int dim_b, int dim_c);

/* Copy `src` (device-side, contiguous, matching the cache's recorded shape
 * and dtype) into the existing cudaArray. For channels==1 this is a single
 * cudaMemcpy3DAsync; for channels==4 a small pack kernel reorganises 4
 * consecutive Y-slices into channel-packed texels via surf2DLayeredwrite. */
void tc_upload(TextureCache *tc, const void *src, cudaStream_t stream);

/* cudaDestroyTextureObject + cudaFreeArray + free(tc). Safe on nullptr. */
void tc_destroy(TextureCache *tc);

#endif /* FASTTOMO_CUDA_TEXTURE_CACHE_H */

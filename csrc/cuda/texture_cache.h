#ifndef FASTTOMO_CUDA_TEXTURE_CACHE_H
#define FASTTOMO_CUDA_TEXTURE_CACHE_H

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

/* Shared CUDA error-check macro. Included (directly or via common.cuh) by
 * every CUDA translation unit so they all use one definition. */
#ifndef FASTTOMO_CUDA_CHECK
#define FASTTOMO_CUDA_CHECK(call) do {                                        \
    cudaError_t _err = (call);                                                \
    if (_err != cudaSuccess) {                                                \
        throw std::runtime_error(std::string("CUDA error at " __FILE__ ":") \
            + std::to_string(__LINE__) + ": " + cudaGetErrorString(_err));   \
    }                                                                          \
} while (0)
#endif /* FASTTOMO_CUDA_CHECK */

/* Persistent cudaArray + cudaTextureObject pair, sized once and reused across
 * many forward/back-projection calls of the same shape. The hot path becomes
 *   tc_upload(cache, new_data_ptr, stream)   // cudaMemcpy3DAsync only
 *   forward_project_parallel_cuda_cached(cache, ...)  // kernel launch
 * with no per-call cudaMalloc3DArray or cudaCreateTextureObject. The
 * non-cached entry points wrap the same code in a ScopedTextureCache (below).
 *
 * Three geometries:
 *   TC_VOL_LAYERED  — FP parallel.  Ny layers of [Nz × Nx]. Source: [Ny,Nz,Nx].
 *   TC_SINO_LAYERED — BP parallel & n_angles layers of [Ny × N_det].
 *                     BP rotmat.    Source: [n_angles, Ny, N_det].
 *   TC_VOL_3D       — FP rotmat,    True 3D (Nx, Nz, Ny). Source: [Ny,Nz,Nx].
 *                     grad_rotmat (point view), MBIR snapshot.
 *
 * dim_a / dim_b / dim_c map onto the source-tensor dims in C order:
 *   vol  variants: dim_a=Ny,        dim_b=Nz, dim_c=Nx
 *   sino variant : dim_a=n_angles,  dim_b=Ny, dim_c=N_det
 *
 * The cache also remembers an I/O dtype:
 *   TC_DTYPE_FP32 — 32-bit float texels (1 channel, or 2 packed: float2)
 *   TC_DTYPE_FP16 — 16-bit half-float texels (1 channel, or 4 packed)
 *
 * Channels — 1 (default), 2 (fp32 only) or 4 (fp16 only): Y-slice packing,
 * TorchRadon's batch-channel-packing trick. C consecutive Y-slices share one
 * C-channel texel, so one tex2DLayered<float2|float4> fetch returns C
 * Y-slices' values at once and a thread does ~C× fewer texture fetches and
 * stores C outputs. Both packed formats are 8-byte texels. The filter unit
 * applies the same weights to every channel, so each channel of a packed
 * fetch equals the 1-channel fetch of that slice bit for bit. The packing
 * axis is Ny: the layer dim of TC_VOL_LAYERED, the height dim of
 * TC_SINO_LAYERED (whose layers are the n_angles views). The packed extent
 * is ceil(Ny / C): when Ny is not a multiple of C the missing channels of
 * the last texel hold 0 (written by the pack kernels and the fused SIRT
 * kernels) and the projection kernels store only slices < Ny. Packed caches
 * always have a surface (the pack kernels write through it). TC_VOL_3D is
 * 1-channel only (rotmat interpolates across Y).
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
    /* tex_point is non-zero only for TC_VOL_3D 1-channel caches; it samples the
     * same cudaArray with cudaFilterModePoint so kernels needing analytic-derivative
     * bit-exact corner reads can use it. grad_rotmat_kernel uses this field so that
     * the 8 trilinear corner reads are bit-exact equivalent to global-memory loads. */
    cudaTextureObject_t   tex_point;
    cudaSurfaceObject_t   surf;     /* 4-channel half upload kernel; direct
                                     * writes (tc_create_ex want_surface) */
    int                   dim_a;
    int                   dim_b;
    int                   dim_c;
    int                   channels; /* 1, 2 (fp32) or 4 (fp16) */
    tc_kind_t             kind;
    tc_dtype_t            dtype;
} TextureCache;

/* Allocate cudaArray + cudaTextureObject sized for (dim_a, dim_b, dim_c) with
 * the given I/O dtype and channel count: 1, 2 (TC_DTYPE_FP32 only) or 4
 * (TC_DTYPE_FP16 only); packed caches (2, 4) only for the layered kinds, any
 * Ny (see above). The returned pointer is heap-allocated; pass it to
 * tc_destroy when done.
 *
 * Throws std::invalid_argument when the extent exceeds the current device's
 * texture limits (e.g. more than 2048 layers for the layered kinds), and
 * std::runtime_error on a CUDA failure. Nothing is leaked on either path:
 * a partially built cache is released before the exception propagates. */
TextureCache *tc_create(tc_kind_t kind, tc_dtype_t dtype, int channels,
                        int dim_a, int dim_b, int dim_c);

/* tc_create, plus (want_surface != 0) a surface object on 1-channel caches
 * so kernels can write the array directly (`surf` stays 0 when the extent
 * exceeds the device's surface limits; packed caches always have one). */
TextureCache *tc_create_ex(tc_kind_t kind, tc_dtype_t dtype, int channels,
                           int dim_a, int dim_b, int dim_c, int want_surface);

/* Copy `src` (device-side, contiguous, matching the cache's recorded shape
 * and dtype) into the existing cudaArray. For channels==1 this is a single
 * cudaMemcpy3DAsync; for packed caches a small pack kernel reorganises C
 * consecutive Y-slices into channel-packed texels via surf2DLayeredwrite
 * (zeros in the missing channels of the last texel). */
void tc_upload(TextureCache *tc, const void *src, cudaStream_t stream);

/* Inverse of tc_upload: copy the cache's contents into the contiguous device
 * buffer `dst` of the cache's source shape and dtype (volume kinds, and
 * 1-channel sinogram caches; the padding channels are not copied). Values
 * are copied bit for bit. */
void tc_download(const TextureCache *tc, void *dst, cudaStream_t stream);

/* Best-effort teardown that never throws: destroys every handle that exists
 * (surface, both textures, array), frees `tc`, and returns the first error
 * seen (cudaSuccess if none). Safe on nullptr.
 *
 * Does NOT wait for kernels that may still read the textures. The caller
 * must order the destroy after the last use (stream/device synchronize);
 * ScopedTextureCache below does this for one-shot caches. */
cudaError_t tc_release(TextureCache *tc) noexcept;

/* tc_release, then throw std::runtime_error if any destroy call failed. */
void tc_destroy(TextureCache *tc);

/* Throw std::invalid_argument unless `tc` is non-null and matches `kind`,
 * the io_dtype (0 = fp32, 1 = fp16) and (dim_a, dim_b, dim_c). `fn` names
 * the caller in the message. The single validation point for every cached
 * entry point (the pybind layer does not re-check). */
void tc_validate(const TextureCache *tc, tc_kind_t kind, int io_dtype,
                 int dim_a, int dim_b, int dim_c, const char *fn);

/* Number of C-channel texels along the packing axis: ceil(Ny / C). */
#ifdef __CUDACC__
__host__ __device__
#endif
static inline int tc_packed_extent(int Ny, int channels) {
    return (Ny + channels - 1) / channels;
}

/* Test-only fault injection: the next tc_create throws at `stage`
 * (1 = after the array allocation, 2 = after the texture object,
 * 3 = after every handle exists; 0 = off). Used by the leak regression test. */
void tc_debug_fail_next_create(int stage);

/* RAII owner of a one-shot TextureCache, used by the non-cached host
 * wrappers ("scoped cache + the cached implementation").
 *
 * - If anything throws after construction (upload, launch-configuration
 *   error, ...), the destructor releases the cudaArray: no leak.
 * - Before releasing, it synchronizes `stream`, so a texture is never
 *   destroyed while a kernel launched on that stream may still sample it.
 *   (cudaLaunchHostFunc cannot be used for the deferred free: host
 *   callbacks must not call CUDA API functions such as cudaFreeArray.
 *   cudaFreeArray itself synchronizes, so the explicit stream sync costs
 *   nothing extra.)
 * - finish() does the same sync + release on the success path but throws
 *   if the stream reported an error, so asynchronous kernel faults surface
 *   at the call that caused them. */
class ScopedTextureCache {
public:
    ScopedTextureCache(tc_kind_t kind, tc_dtype_t dtype, int channels,
                       int dim_a, int dim_b, int dim_c, cudaStream_t stream)
        : tc_(tc_create(kind, dtype, channels, dim_a, dim_b, dim_c)),
          stream_(stream) {}

    ~ScopedTextureCache() {
        if (!tc_) return;
        /* Error path: best effort, a destructor must not throw. A sticky
         * error, if any, resurfaces on the caller's next CUDA call. */
        (void)cudaStreamSynchronize(stream_);
        (void)tc_release(tc_);
    }

    ScopedTextureCache(const ScopedTextureCache &) = delete;
    ScopedTextureCache &operator=(const ScopedTextureCache &) = delete;

    TextureCache *get() const { return tc_; }

    void finish() {
        TextureCache *tc = tc_;
        tc_ = nullptr;
        cudaError_t sync_err = cudaStreamSynchronize(stream_);
        cudaError_t rel_err  = tc_release(tc);
        FASTTOMO_CUDA_CHECK(sync_err);
        FASTTOMO_CUDA_CHECK(rel_err);
    }

private:
    TextureCache *tc_;
    cudaStream_t  stream_;
};

#endif /* FASTTOMO_CUDA_TEXTURE_CACHE_H */

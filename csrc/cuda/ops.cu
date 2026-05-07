#include "ops.h"
#include "texture_cache.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdexcept>
#include <string>

/* ── dtype helpers ────────────────────────────────────────────────────────
 * The 4 in-scope kernels (FP/BP scalar + FP/BP rotmat) are templated on the
 * I/O element type T. T is float (for fp32) or __half (for fp16). The
 * texture channel desc matches T (fp32 channel for float, fp16 channel for
 * __half) but texture fetches always use tex*<float> — CUDA's texture
 * hardware promotes 16-bit float texels to 32-bit float during filtered
 * reads, so the kernel sees float regardless of underlying storage. Stores
 * cast the float accumulator back to T. cos/sin tables and rotation
 * matrices stay fp32 in all paths.
 */
template <typename T> __device__ __forceinline__ T float_to_t(float v);
template <>           __device__ __forceinline__ float  float_to_t<float>(float v)  { return v; }
template <>           __device__ __forceinline__ __half float_to_t<__half>(float v) { return __float2half_rn(v); }

/* Channel desc dispatch — cudaCreateChannelDesc<__half>() is not specialized
 * in the stock CUDA toolkit headers, so we route through the explicit Half
 * helper for fp16. */
template <typename T> __host__ static cudaChannelFormatDesc make_channel_desc();
template <>           __host__ cudaChannelFormatDesc make_channel_desc<float>()  { return cudaCreateChannelDesc<float>(); }
template <>           __host__ cudaChannelFormatDesc make_channel_desc<__half>() { return cudaCreateChannelDescHalf(); }

/* 4-channel half channel desc — TorchRadon-style packed format. 4 fp16 values
 * per texel = 8 bytes = float2-equivalent. tex2DLayered<float4> reads it. */
__host__ static cudaChannelFormatDesc make_channel_desc_half4() {
    return cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindFloat);
}

/* ── error handling ───────────────────────────────────────────────────────
 * FASTTOMO_CUDA_CHECK (defined in texture_cache.h) propagates cudaError_t
 * failures as std::runtime_error so the pybind11 layer can surface them as
 * Python RuntimeError. CUDA_CHECK_KERNEL is called immediately after a
 * <<< >>> launch to catch launch-time errors (invalid configuration, too
 * much shared memory, etc.) without paying for a host-device sync.
 * Stream-ordered errors (out-of-bounds writes etc.) are surfaced later when
 * PyTorch syncs on tensor read.
 */
#define CUDA_CHECK_KERNEL() do {                                              \
    cudaError_t _err = cudaPeekAtLastError();                                 \
    if (_err != cudaSuccess) {                                                \
        throw std::runtime_error(std::string("CUDA kernel launch error at "   \
            __FILE__ ":") + std::to_string(__LINE__) + ": "                   \
            + cudaGetErrorString(_err));                                       \
    }                                                                          \
} while (0)

/* ── helpers ──────────────────────────────────────────────────────────────
 * make_vol_texture:  wrap a device [Ny, Nz, Nx] float buffer in a layered
 *                    2D texture (Ny layers of [Nz, Nx]).
 * make_sino_texture: wrap a device [n_angles, Ny, N_det] float buffer in a
 *                    layered 2D texture (n_angles layers of [Ny, N_det]).
 *
 * Both use hardware linear filtering (free bilinear interpolation) and
 * border addressing (returns 0.0 for out-of-bounds coordinates).
 * Caller must call cudaDestroyTextureObject + cudaFreeArray when done.
 */
template <typename T>
static cudaTextureObject_t make_vol_texture(
    const T *d_vol, int Ny, int Nz, int Nx,
    cudaArray_t &out_array, cudaStream_t stream
) {
    cudaChannelFormatDesc fmt = make_channel_desc<T>();
    /* Ny layers, each Nx wide × Nz tall */
    FASTTOMO_CUDA_CHECK(cudaMalloc3DArray(&out_array, &fmt,
                                 make_cudaExtent(Nx, Nz, Ny), cudaArrayLayered));

    cudaMemcpy3DParms cp = {};
    cp.srcPtr   = make_cudaPitchedPtr((void *)d_vol,
                                      Nx * sizeof(T), Nx, Nz);
    cp.dstArray = out_array;
    cp.extent   = make_cudaExtent(Nx, Nz, Ny);
    cp.kind     = cudaMemcpyDeviceToDevice;
    FASTTOMO_CUDA_CHECK(cudaMemcpy3DAsync(&cp, stream));

    cudaResourceDesc rd = {};
    rd.resType         = cudaResourceTypeArray;
    rd.res.array.array = out_array;

    cudaTextureDesc td = {};
    td.addressMode[0]  = cudaAddressModeBorder;
    td.addressMode[1]  = cudaAddressModeBorder;
    td.filterMode      = cudaFilterModeLinear;
    td.readMode        = cudaReadModeElementType;
    td.normalizedCoords = 0;

    cudaTextureObject_t tex;
    FASTTOMO_CUDA_CHECK(cudaCreateTextureObject(&tex, &rd, &td, nullptr));
    return tex;
}

template <typename T>
static cudaTextureObject_t make_sino_texture(
    const T *d_sino, int n_angles, int Ny, int N_det,
    cudaArray_t &out_array, cudaStream_t stream
) {
    cudaChannelFormatDesc fmt = make_channel_desc<T>();
    /* n_angles layers, each N_det wide × Ny tall */
    FASTTOMO_CUDA_CHECK(cudaMalloc3DArray(&out_array, &fmt,
                                 make_cudaExtent(N_det, Ny, n_angles), cudaArrayLayered));

    cudaMemcpy3DParms cp = {};
    cp.srcPtr   = make_cudaPitchedPtr((void *)d_sino,
                                      N_det * sizeof(T), N_det, Ny);
    cp.dstArray = out_array;
    cp.extent   = make_cudaExtent(N_det, Ny, n_angles);
    cp.kind     = cudaMemcpyDeviceToDevice;
    FASTTOMO_CUDA_CHECK(cudaMemcpy3DAsync(&cp, stream));

    cudaResourceDesc rd = {};
    rd.resType         = cudaResourceTypeArray;
    rd.res.array.array = out_array;

    cudaTextureDesc td = {};
    td.addressMode[0]  = cudaAddressModeBorder;
    td.addressMode[1]  = cudaAddressModeBorder;
    td.filterMode      = cudaFilterModeLinear;
    td.readMode        = cudaReadModeElementType;
    td.normalizedCoords = 0;

    cudaTextureObject_t tex;
    FASTTOMO_CUDA_CHECK(cudaCreateTextureObject(&tex, &rd, &td, nullptr));
    return tex;
}

/* ── 4-channel packed-Y fp16 helpers ─────────────────────────────────────────
 * Mirrors TorchRadon's batch-channel packing: 4 consecutive Y-slices are
 * stored in the 4 channels of a single texel. tex2DLayered<float4> returns
 * 4 floats (hardware-promoted from packed halves) per fetch. The kernel
 * accumulates 4 line integrals in parallel and writes 4 strided outputs
 * per thread. Throughput goes up because the texture-fetch hardware does
 * 4× less work for the same useful output. */

/* Pack-and-upload kernels — read 4 Y-slices from a fp16 [Ny, Nz, Nx] / sino
 * tensor and surf2DLayeredwrite as float2 (= 4 halves in the 4-channel
 * texel). Mirror tc_upload's 4-channel path; used by the non-cached entry
 * points which allocate a one-shot packed cudaArray + surface. */
__global__ static void pack_vol_half4(
    const __half *src,                /* [Ny, Nz, Nx] */
    cudaSurfaceObject_t surf,
    int Nz, int Nx
) {
    int ix      = blockIdx.x * blockDim.x + threadIdx.x;
    int iz      = blockIdx.y * blockDim.y + threadIdx.y;
    int iy_pack = blockIdx.z;
    if (ix >= Nx || iz >= Nz) return;
    int iy_base = 4 * iy_pack;
    size_t base = (size_t)iy_base * Nz * Nx + (size_t)iz * Nx + ix;
    size_t stride_y = (size_t)Nz * Nx;
    __half tmp[4];
    tmp[0] = src[base + 0 * stride_y];
    tmp[1] = src[base + 1 * stride_y];
    tmp[2] = src[base + 2 * stride_y];
    tmp[3] = src[base + 3 * stride_y];
    surf2DLayeredwrite<float2>(*(float2 *)tmp, surf, ix * (int)sizeof(float2), iz, iy_pack);
}

__global__ static void pack_sino_half4(
    const __half *src,                /* [n_angles, Ny, N_det] */
    cudaSurfaceObject_t surf,
    int Ny, int N_det
) {
    int ix      = blockIdx.x * blockDim.x + threadIdx.x;
    int iy_pack = blockIdx.y * blockDim.y + threadIdx.y;
    int ia      = blockIdx.z;
    if (ix >= N_det || iy_pack >= Ny / 4) return;
    int iy_base = 4 * iy_pack;
    size_t base = (size_t)ia * Ny * N_det + (size_t)iy_base * N_det + ix;
    size_t stride_y = (size_t)N_det;
    __half tmp[4];
    tmp[0] = src[base + 0 * stride_y];
    tmp[1] = src[base + 1 * stride_y];
    tmp[2] = src[base + 2 * stride_y];
    tmp[3] = src[base + 3 * stride_y];
    surf2DLayeredwrite<float2>(*(float2 *)tmp, surf, ix * (int)sizeof(float2), iy_pack, ia);
}

/* Allocate a one-shot 4-channel half packed vol texture, run pack kernel,
 * return tex (with the cudaArray + surface for cleanup). */
static cudaTextureObject_t make_vol_texture_packed_half(
    const __half *d_vol, int Ny, int Nz, int Nx,
    cudaArray_t &out_array, cudaSurfaceObject_t &out_surf, cudaStream_t stream
) {
    cudaChannelFormatDesc fmt = make_channel_desc_half4();
    FASTTOMO_CUDA_CHECK(cudaMalloc3DArray(&out_array, &fmt,
                                 make_cudaExtent(Nx, Nz, Ny / 4),
                                 cudaArrayLayered | cudaArraySurfaceLoadStore));

    cudaResourceDesc rd = {};
    rd.resType         = cudaResourceTypeArray;
    rd.res.array.array = out_array;

    FASTTOMO_CUDA_CHECK(cudaCreateSurfaceObject(&out_surf, &rd));

    dim3 block(16, 16, 1);
    dim3 grid((Nx + 15) / 16, (Nz + 15) / 16, Ny / 4);
    pack_vol_half4<<<grid, block, 0, stream>>>(d_vol, out_surf, Nz, Nx);
    CUDA_CHECK_KERNEL();

    cudaTextureDesc td = {};
    td.addressMode[0]   = cudaAddressModeBorder;
    td.addressMode[1]   = cudaAddressModeBorder;
    td.filterMode       = cudaFilterModeLinear;
    td.readMode         = cudaReadModeElementType;
    td.normalizedCoords = 0;

    cudaTextureObject_t tex;
    FASTTOMO_CUDA_CHECK(cudaCreateTextureObject(&tex, &rd, &td, nullptr));
    return tex;
}

static cudaTextureObject_t make_sino_texture_packed_half(
    const __half *d_sino, int n_angles, int Ny, int N_det,
    cudaArray_t &out_array, cudaSurfaceObject_t &out_surf, cudaStream_t stream
) {
    cudaChannelFormatDesc fmt = make_channel_desc_half4();
    FASTTOMO_CUDA_CHECK(cudaMalloc3DArray(&out_array, &fmt,
                                 make_cudaExtent(N_det, Ny / 4, n_angles),
                                 cudaArrayLayered | cudaArraySurfaceLoadStore));

    cudaResourceDesc rd = {};
    rd.resType         = cudaResourceTypeArray;
    rd.res.array.array = out_array;

    FASTTOMO_CUDA_CHECK(cudaCreateSurfaceObject(&out_surf, &rd));

    dim3 block(16, 16, 1);
    dim3 grid((N_det + 15) / 16, ((Ny / 4) + 15) / 16, n_angles);
    pack_sino_half4<<<grid, block, 0, stream>>>(d_sino, out_surf, Ny, N_det);
    CUDA_CHECK_KERNEL();

    cudaTextureDesc td = {};
    td.addressMode[0]   = cudaAddressModeBorder;
    td.addressMode[1]   = cudaAddressModeBorder;
    td.filterMode       = cudaFilterModeLinear;
    td.readMode         = cudaReadModeElementType;
    td.normalizedCoords = 0;

    cudaTextureObject_t tex;
    FASTTOMO_CUDA_CHECK(cudaCreateTextureObject(&tex, &rd, &td, nullptr));
    return tex;
}

/* If smem exceeds the default 48 KB cap, opt the kernel into the extended
 * dynamic shared memory pool. Capped at 96 KB which is safe on every arch
 * we target (Volta+: 96, Ampere/Ada: 100/164, Hopper: 228, Blackwell: 228).
 */
static void enable_extended_smem_if_needed(const void *kernel, size_t smem) {
    if (smem > 48 * 1024) {
        size_t cap = 96 * 1024;
        size_t want = smem < cap ? smem : cap;
        FASTTOMO_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)want));
    }
}

/* ── forward projection ───────────────────────────────────────────────────
 * Grid:  (ceil(N_det/16), ceil(n_angles/16), Ny)   Block: (16, 16, 1)
 * Each thread owns one (detector, angle) output cell in one Y-slice.
 * Marches along the ray; tex2DLayered handles bilinear interp and border.
 */
template <typename T>
__global__ static void forward_project_kernel(
    cudaTextureObject_t  vol_tex,    /* Ny layers of [Nz, Nx] */
    T *__restrict__      sinogram,   /* [n_angles, Ny, N_det] */
    const float *__restrict__ cos_a,
    const float *__restrict__ sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    float half_det, float half_x, float half_z, int ray_len
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;  /* detector pixel */
    int ia = blockIdx.y * blockDim.y + threadIdx.y;  /* angle index    */
    int iy = blockIdx.z;                              /* Y-slice / layer */

    if (id >= N_det || ia >= n_angles) return;

    float ca = cos_a[ia], sa = sin_a[ia];
    float t   = id - half_det;
    float sum = 0.0f;

    for (int step = -ray_len; step <= ray_len; step++) {
        float s  = (float)step;
        float fx = (t * ca - s * sa) + half_x;
        float fz = (t * sa + s * ca) + half_z;
        /* Match forward_project_cpu's tight bilinear-stencil bounds: only
         * sample when both pixel neighbours along each axis are inside the
         * volume (fx in [0, Nx-1), fz in [0, Nz-1)). Border-mode tex2DLayered
         * would otherwise contribute a half-texel slab at each edge that the
         * CPU kernel skips. */
        if (fx < 0.0f || fx >= (float)(Nx - 1) ||
            fz < 0.0f || fz >= (float)(Nz - 1))
            continue;
        /* tex2DLayered: x→Nx axis, y→Nz axis, layer→iy.
         * +0.5 maps integer pixel index to texel centre.
         * Always read as float — texture hardware promotes 16-bit float
         * texels (fp16 channel desc) to 32-bit float during the filtered
         * read. The kernel sees float regardless of texel storage. */
        sum += tex2DLayered<float>(vol_tex, fx + 0.5f, fz + 0.5f, iy);
    }

    sinogram[((size_t)ia * Ny + iy) * N_det + id] = float_to_t<T>(sum);
}

/* ── dispatch helper ──────────────────────────────────────────────────────
 * Each in-scope op has a templated impl<T>() that does the real work and a
 * thin public dispatch function that switches on io_dtype (0=fp32, 1=fp16)
 * to call the right instantiation. The public signature carries void* so
 * the pybind11 layer can pass tensor.data_ptr() without knowing the dtype.
 *
 * For the scalar single-axis FP/BP path (not rotmat), the fp16 dispatch
 * additionally checks ``Ny % 4 == 0`` and routes to a 4-channel packed
 * kernel that reads tex2DLayered<float4> for ~3× extra throughput
 * (TorchRadon's batch-channel-packing trick). */
static void check_io_dtype(int io_dtype, const char *fn) {
    if (io_dtype != 0 && io_dtype != 1)
        throw std::runtime_error(std::string(fn) + ": io_dtype must be 0 (fp32) or 1 (fp16)");
}

/* ── 4-channel packed forward-projection kernel ─────────────────────────────
 * Reads tex2DLayered<float4> (4 packed Y-slices' values per fetch),
 * accumulates 4 line integrals in parallel, writes 4 strided outputs to the
 * user's [n_angles, Ny, N_det] sinogram (no output packing — the kernel
 * unpacks on store). Grid Z is Ny/4 instead of Ny, so the texture-fetch
 * count per (det, angle) cell drops by 4×. Bounds-check matches the scalar
 * fp32 kernel exactly (tight bilinear-stencil bounds). */
__global__ static void forward_project_kernel_packed_h4(
    cudaTextureObject_t  vol_tex,    /* (Ny/4) layers of [Nz × Nx], 4-ch half */
    __half *__restrict__ sinogram,   /* [n_angles, Ny, N_det] */
    const float *__restrict__ cos_a,
    const float *__restrict__ sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    float half_det, float half_x, float half_z, int ray_len
) {
    int id      = blockIdx.x * blockDim.x + threadIdx.x;
    int ia      = blockIdx.y * blockDim.y + threadIdx.y;
    int iy_pack = blockIdx.z;
    if (id >= N_det || ia >= n_angles) return;

    float ca = cos_a[ia], sa = sin_a[ia];
    float t  = id - half_det;
    float sums[4] = {0.f, 0.f, 0.f, 0.f};

    for (int step = -ray_len; step <= ray_len; step++) {
        float s  = (float)step;
        float fx = (t * ca - s * sa) + half_x;
        float fz = (t * sa + s * ca) + half_z;
        if (fx < 0.0f || fx >= (float)(Nx - 1) ||
            fz < 0.0f || fz >= (float)(Nz - 1))
            continue;
        float4 v = tex2DLayered<float4>(vol_tex, fx + 0.5f, fz + 0.5f, iy_pack);
        sums[0] += v.x;
        sums[1] += v.y;
        sums[2] += v.z;
        sums[3] += v.w;
    }

    int iy_base = 4 * iy_pack;
    #pragma unroll
    for (int b = 0; b < 4; b++) {
        sinogram[((size_t)ia * Ny + (iy_base + b)) * N_det + id]
            = __float2half_rn(sums[b]);
    }
}

/* ── 4-channel packed back-projection kernel ────────────────────────────────
 * Each thread owns one (ix, iz, iy_pack) voxel-group of 4 voxels
 * (4 Y-slices). Reads tex2DLayered<float4> at (ft, iy_pack+0.5, ia) which
 * returns 4 values for the same (det, angle) at 4 different Y-slices —
 * iy_pack+0.5 is a texel centre so no Y-axis interpolation. Writes 4
 * strided outputs to user's [Ny, Nz, Nx] volume. */
__global__ static void back_project_kernel_packed_h4(
    cudaTextureObject_t  sino_tex,   /* n_angles layers of [(Ny/4) × N_det], 4-ch half */
    __half *__restrict__ volume,     /* [Ny, Nz, Nx] */
    const float *__restrict__ cos_a,
    const float *__restrict__ sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    float half_det, float half_x, float half_z
) {
    extern __shared__ float sh[];
    float *sh_cos = sh;
    float *sh_sin = sh + n_angles;
    for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < n_angles;
         i += blockDim.x * blockDim.y) {
        sh_cos[i] = cos_a[i];
        sh_sin[i] = sin_a[i];
    }
    __syncthreads();

    int ix      = blockIdx.x * blockDim.x + threadIdx.x;
    int iz      = blockIdx.y * blockDim.y + threadIdx.y;
    int iy_pack = blockIdx.z;
    if (ix >= Nx || iz >= Nz) return;

    float x = ix - half_x;
    float z = iz - half_z;
    float vals[4] = {0.f, 0.f, 0.f, 0.f};
    float iy_tc = (float)iy_pack + 0.5f;   /* exact texel centre, no Y-interp */

    for (int ia = 0; ia < n_angles; ia++) {
        float ft = x * sh_cos[ia] + z * sh_sin[ia] + half_det;
        float4 sample = tex2DLayered<float4>(sino_tex, ft + 0.5f, iy_tc, ia);
        vals[0] += sample.x;
        vals[1] += sample.y;
        vals[2] += sample.z;
        vals[3] += sample.w;
    }

    int iy_base = 4 * iy_pack;
    #pragma unroll
    for (int b = 0; b < 4; b++) {
        volume[((size_t)(iy_base + b) * Nz + iz) * Nx + ix]
            = __float2half_rn(vals[b]);
    }
}

template <typename T>
static void forward_project_cuda_impl(
    const T     *d_volume,
    T           *d_sinogram,
    const float *d_cos_a,
    const float *d_sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
) {
    float half_det = (N_det - 1) * 0.5f;
    float half_x   = (Nx   - 1) * 0.5f;
    float half_z   = (Nz   - 1) * 0.5f;
    int   ray_len  = (int)ceilf(0.5f * sqrtf((float)(Nx * Nx + Nz * Nz))) + 1;

    cudaArray_t vol_array;
    cudaTextureObject_t vol_tex = make_vol_texture<T>(d_volume, Ny, Nz, Nx, vol_array, stream);

    dim3 block(16, 16, 1);
    dim3 grid((N_det    + 15) / 16,
              (n_angles + 15) / 16,
              Ny);

    forward_project_kernel<T><<<grid, block, 0, stream>>>(
        vol_tex, d_sinogram, d_cos_a, d_sin_a,
        Ny, Nz, Nx, n_angles, N_det,
        half_det, half_x, half_z, ray_len
    );
    CUDA_CHECK_KERNEL();

    /* Texture object and array are stream-ordered with the launch above:
     * destroying them schedules the release after the kernel completes. */
    FASTTOMO_CUDA_CHECK(cudaDestroyTextureObject(vol_tex));
    FASTTOMO_CUDA_CHECK(cudaFreeArray(vol_array));
}

/* fp16 + Ny%4==0: packed-Y (4-channel) path. Allocates a one-shot 4-channel
 * cudaArray + surface, packs the user's [Ny, Nz, Nx] half tensor into it,
 * runs the packed kernel, destroys. */
static void forward_project_cuda_impl_packed_h4(
    const __half *d_volume, __half *d_sinogram,
    const float  *d_cos_a, const float *d_sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
) {
    float half_det = (N_det - 1) * 0.5f;
    float half_x   = (Nx   - 1) * 0.5f;
    float half_z   = (Nz   - 1) * 0.5f;
    int   ray_len  = (int)ceilf(0.5f * sqrtf((float)(Nx * Nx + Nz * Nz))) + 1;

    cudaArray_t vol_array;
    cudaSurfaceObject_t vol_surf;
    cudaTextureObject_t vol_tex = make_vol_texture_packed_half(
        d_volume, Ny, Nz, Nx, vol_array, vol_surf, stream);

    dim3 block(16, 16, 1);
    dim3 grid((N_det    + 15) / 16,
              (n_angles + 15) / 16,
              Ny / 4);

    forward_project_kernel_packed_h4<<<grid, block, 0, stream>>>(
        vol_tex, d_sinogram, d_cos_a, d_sin_a,
        Ny, Nz, Nx, n_angles, N_det,
        half_det, half_x, half_z, ray_len);
    CUDA_CHECK_KERNEL();

    FASTTOMO_CUDA_CHECK(cudaDestroyTextureObject(vol_tex));
    FASTTOMO_CUDA_CHECK(cudaDestroySurfaceObject(vol_surf));
    FASTTOMO_CUDA_CHECK(cudaFreeArray(vol_array));
}

void forward_project_cuda(
    int io_dtype,
    const void  *d_volume,
    void        *d_sinogram,
    const float *d_cos_a,
    const float *d_sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
) {
    check_io_dtype(io_dtype, "forward_project_cuda");
    if (io_dtype == 0)
        forward_project_cuda_impl<float>(
            (const float *)d_volume, (float *)d_sinogram,
            d_cos_a, d_sin_a, Ny, Nz, Nx, n_angles, N_det, stream);
    else if (Ny % 4 == 0)
        forward_project_cuda_impl_packed_h4(
            (const __half *)d_volume, (__half *)d_sinogram,
            d_cos_a, d_sin_a, Ny, Nz, Nx, n_angles, N_det, stream);
    else
        forward_project_cuda_impl<__half>(
            (const __half *)d_volume, (__half *)d_sinogram,
            d_cos_a, d_sin_a, Ny, Nz, Nx, n_angles, N_det, stream);
}

template <typename T>
static void forward_project_cuda_cached_impl(
    TextureCache *vol_cache,
    const T      *d_volume,
    T            *d_sinogram,
    const float  *d_cos_a,
    const float  *d_sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
) {
    float half_det = (N_det - 1) * 0.5f;
    float half_x   = (Nx   - 1) * 0.5f;
    float half_z   = (Nz   - 1) * 0.5f;
    int   ray_len  = (int)ceilf(0.5f * sqrtf((float)(Nx * Nx + Nz * Nz))) + 1;

    tc_upload(vol_cache, d_volume, stream);

    dim3 block(16, 16, 1);
    dim3 grid((N_det    + 15) / 16,
              (n_angles + 15) / 16,
              Ny);

    forward_project_kernel<T><<<grid, block, 0, stream>>>(
        vol_cache->tex, d_sinogram, d_cos_a, d_sin_a,
        Ny, Nz, Nx, n_angles, N_det,
        half_det, half_x, half_z, ray_len
    );
    CUDA_CHECK_KERNEL();
}

/* Cached packed-Y FP. The cache must be channels=4. tc_upload runs the
 * pack kernel; we then dispatch the float4-reading kernel. */
static void forward_project_cuda_cached_impl_packed_h4(
    TextureCache *vol_cache,
    const __half *d_volume, __half *d_sinogram,
    const float  *d_cos_a, const float *d_sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
) {
    float half_det = (N_det - 1) * 0.5f;
    float half_x   = (Nx   - 1) * 0.5f;
    float half_z   = (Nz   - 1) * 0.5f;
    int   ray_len  = (int)ceilf(0.5f * sqrtf((float)(Nx * Nx + Nz * Nz))) + 1;

    tc_upload(vol_cache, d_volume, stream);

    dim3 block(16, 16, 1);
    dim3 grid((N_det    + 15) / 16,
              (n_angles + 15) / 16,
              Ny / 4);

    forward_project_kernel_packed_h4<<<grid, block, 0, stream>>>(
        vol_cache->tex, d_sinogram, d_cos_a, d_sin_a,
        Ny, Nz, Nx, n_angles, N_det,
        half_det, half_x, half_z, ray_len);
    CUDA_CHECK_KERNEL();
}

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
) {
    check_io_dtype(io_dtype, "forward_project_cuda_cached");
    if (!vol_cache) throw std::runtime_error("forward_project_cuda_cached: null cache");
    if (vol_cache->kind  != TC_VOL_LAYERED ||
        vol_cache->dim_a != Ny || vol_cache->dim_b != Nz || vol_cache->dim_c != Nx)
        throw std::runtime_error("forward_project_cuda_cached: cache shape mismatch");
    int cache_dtype = (vol_cache->dtype == TC_DTYPE_FP32) ? 0 : 1;
    if (cache_dtype != io_dtype)
        throw std::runtime_error("forward_project_cuda_cached: cache dtype mismatch");

    if (io_dtype == 0)
        forward_project_cuda_cached_impl<float>(
            vol_cache, (const float *)d_volume, (float *)d_sinogram,
            d_cos_a, d_sin_a, Ny, Nz, Nx, n_angles, N_det, stream);
    else if (vol_cache->channels == 4)
        forward_project_cuda_cached_impl_packed_h4(
            vol_cache, (const __half *)d_volume, (__half *)d_sinogram,
            d_cos_a, d_sin_a, Ny, Nz, Nx, n_angles, N_det, stream);
    else
        forward_project_cuda_cached_impl<__half>(
            vol_cache, (const __half *)d_volume, (__half *)d_sinogram,
            d_cos_a, d_sin_a, Ny, Nz, Nx, n_angles, N_det, stream);
}

/* ── back-projection ──────────────────────────────────────────────────────
 * Grid:  (ceil(Nx/16), ceil(Nz/16), Ny)   Block: (16, 16, 1)
 * Each thread owns one (ix, iz) voxel in one Y-slice and loops over angles.
 * cos_a/sin_a are preloaded into shared memory. tex2DLayered on the
 * sinogram provides free hardware interpolation along the detector axis.
 */
template <typename T>
__global__ static void back_project_kernel(
    cudaTextureObject_t  sino_tex,   /* n_angles layers of [Ny, N_det] */
    T *__restrict__      volume,     /* [Ny, Nz, Nx] */
    const float *__restrict__ cos_a,
    const float *__restrict__ sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    float half_det, float half_x, float half_z
) {
    /* cos/sin tables stay fp32 in shared memory — small, precision-critical. */
    extern __shared__ float sh[];
    float *sh_cos = sh;
    float *sh_sin = sh + n_angles;

    for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < n_angles;
         i += blockDim.x * blockDim.y) {
        sh_cos[i] = cos_a[i];
        sh_sin[i] = sin_a[i];
    }
    __syncthreads();

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int iy = blockIdx.z;

    if (ix >= Nx || iz >= Nz) return;

    float x   = ix - half_x;
    float z   = iz - half_z;
    float val = 0.0f;

    float iy_tc = (float)iy + 0.5f;  /* Y-slice texel centre — no interpolation in Y */

    for (int ia = 0; ia < n_angles; ia++) {
        float ft = x * sh_cos[ia] + z * sh_sin[ia] + half_det;
        /* tex2DLayered: x→N_det axis, y→Ny axis, layer→ia.
         * Hardware linear filter interpolates along detector (ft fractional).
         * iy_tc is always at a texel centre so no interpolation in Y.
         * Always read as float — hardware promotes fp16 texels on read. */
        val += tex2DLayered<float>(sino_tex, ft + 0.5f, iy_tc, ia);
    }

    volume[((size_t)iy * Nz + iz) * Nx + ix] = float_to_t<T>(val);
}

template <typename T>
static void back_project_cuda_impl(
    const T     *d_sinogram,
    T           *d_volume,
    const float *d_cos_a,
    const float *d_sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
) {
    float half_det = (N_det - 1) * 0.5f;
    float half_x   = (Nx   - 1) * 0.5f;
    float half_z   = (Nz   - 1) * 0.5f;

    cudaArray_t sino_array;
    cudaTextureObject_t sino_tex =
        make_sino_texture<T>(d_sinogram, n_angles, Ny, N_det, sino_array, stream);

    dim3 block(16, 16, 1);
    dim3 grid((Nx + 15) / 16, (Nz + 15) / 16, Ny);
    size_t smem = 2 * (size_t)n_angles * sizeof(float);

    enable_extended_smem_if_needed((const void *)back_project_kernel<T>, smem);

    back_project_kernel<T><<<grid, block, smem, stream>>>(
        sino_tex, d_volume, d_cos_a, d_sin_a,
        Ny, Nz, Nx, n_angles, N_det,
        half_det, half_x, half_z
    );
    CUDA_CHECK_KERNEL();

    FASTTOMO_CUDA_CHECK(cudaDestroyTextureObject(sino_tex));
    FASTTOMO_CUDA_CHECK(cudaFreeArray(sino_array));
}

static void back_project_cuda_impl_packed_h4(
    const __half *d_sinogram, __half *d_volume,
    const float  *d_cos_a, const float *d_sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
) {
    float half_det = (N_det - 1) * 0.5f;
    float half_x   = (Nx   - 1) * 0.5f;
    float half_z   = (Nz   - 1) * 0.5f;

    cudaArray_t sino_array;
    cudaSurfaceObject_t sino_surf;
    cudaTextureObject_t sino_tex = make_sino_texture_packed_half(
        d_sinogram, n_angles, Ny, N_det, sino_array, sino_surf, stream);

    dim3 block(16, 16, 1);
    dim3 grid((Nx + 15) / 16, (Nz + 15) / 16, Ny / 4);
    size_t smem = 2 * (size_t)n_angles * sizeof(float);

    enable_extended_smem_if_needed((const void *)back_project_kernel_packed_h4, smem);

    back_project_kernel_packed_h4<<<grid, block, smem, stream>>>(
        sino_tex, d_volume, d_cos_a, d_sin_a,
        Ny, Nz, Nx, n_angles, N_det,
        half_det, half_x, half_z);
    CUDA_CHECK_KERNEL();

    FASTTOMO_CUDA_CHECK(cudaDestroyTextureObject(sino_tex));
    FASTTOMO_CUDA_CHECK(cudaDestroySurfaceObject(sino_surf));
    FASTTOMO_CUDA_CHECK(cudaFreeArray(sino_array));
}

void back_project_cuda(
    int io_dtype,
    const void  *d_sinogram,
    void        *d_volume,
    const float *d_cos_a,
    const float *d_sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
) {
    check_io_dtype(io_dtype, "back_project_cuda");
    if (io_dtype == 0)
        back_project_cuda_impl<float>(
            (const float *)d_sinogram, (float *)d_volume,
            d_cos_a, d_sin_a, Ny, Nz, Nx, n_angles, N_det, stream);
    else if (Ny % 4 == 0)
        back_project_cuda_impl_packed_h4(
            (const __half *)d_sinogram, (__half *)d_volume,
            d_cos_a, d_sin_a, Ny, Nz, Nx, n_angles, N_det, stream);
    else
        back_project_cuda_impl<__half>(
            (const __half *)d_sinogram, (__half *)d_volume,
            d_cos_a, d_sin_a, Ny, Nz, Nx, n_angles, N_det, stream);
}

template <typename T>
static void back_project_cuda_cached_impl(
    TextureCache *sino_cache,
    const T      *d_sinogram,
    T            *d_volume,
    const float  *d_cos_a,
    const float  *d_sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
) {
    float half_det = (N_det - 1) * 0.5f;
    float half_x   = (Nx   - 1) * 0.5f;
    float half_z   = (Nz   - 1) * 0.5f;

    tc_upload(sino_cache, d_sinogram, stream);

    dim3 block(16, 16, 1);
    dim3 grid((Nx + 15) / 16, (Nz + 15) / 16, Ny);
    size_t smem = 2 * (size_t)n_angles * sizeof(float);

    enable_extended_smem_if_needed((const void *)back_project_kernel<T>, smem);

    back_project_kernel<T><<<grid, block, smem, stream>>>(
        sino_cache->tex, d_volume, d_cos_a, d_sin_a,
        Ny, Nz, Nx, n_angles, N_det,
        half_det, half_x, half_z
    );
    CUDA_CHECK_KERNEL();
}

static void back_project_cuda_cached_impl_packed_h4(
    TextureCache *sino_cache,
    const __half *d_sinogram, __half *d_volume,
    const float  *d_cos_a, const float *d_sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
) {
    float half_det = (N_det - 1) * 0.5f;
    float half_x   = (Nx   - 1) * 0.5f;
    float half_z   = (Nz   - 1) * 0.5f;

    tc_upload(sino_cache, d_sinogram, stream);

    dim3 block(16, 16, 1);
    dim3 grid((Nx + 15) / 16, (Nz + 15) / 16, Ny / 4);
    size_t smem = 2 * (size_t)n_angles * sizeof(float);

    enable_extended_smem_if_needed((const void *)back_project_kernel_packed_h4, smem);

    back_project_kernel_packed_h4<<<grid, block, smem, stream>>>(
        sino_cache->tex, d_volume, d_cos_a, d_sin_a,
        Ny, Nz, Nx, n_angles, N_det,
        half_det, half_x, half_z);
    CUDA_CHECK_KERNEL();
}

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
) {
    check_io_dtype(io_dtype, "back_project_cuda_cached");
    if (!sino_cache) throw std::runtime_error("back_project_cuda_cached: null cache");
    if (sino_cache->kind  != TC_SINO_LAYERED ||
        sino_cache->dim_a != n_angles || sino_cache->dim_b != Ny || sino_cache->dim_c != N_det)
        throw std::runtime_error("back_project_cuda_cached: cache shape mismatch");
    int cache_dtype = (sino_cache->dtype == TC_DTYPE_FP32) ? 0 : 1;
    if (cache_dtype != io_dtype)
        throw std::runtime_error("back_project_cuda_cached: cache dtype mismatch");

    if (io_dtype == 0)
        back_project_cuda_cached_impl<float>(
            sino_cache, (const float *)d_sinogram, (float *)d_volume,
            d_cos_a, d_sin_a, Ny, Nz, Nx, n_angles, N_det, stream);
    else if (sino_cache->channels == 4)
        back_project_cuda_cached_impl_packed_h4(
            sino_cache, (const __half *)d_sinogram, (__half *)d_volume,
            d_cos_a, d_sin_a, Ny, Nz, Nx, n_angles, N_det, stream);
    else
        back_project_cuda_cached_impl<__half>(
            sino_cache, (const __half *)d_sinogram, (__half *)d_volume,
            d_cos_a, d_sin_a, Ny, Nz, Nx, n_angles, N_det, stream);
}

/* ── 3D rotation-matrix forward projection ───────────────────────────────
 * Wraps a [Ny, Nz, Nx] volume in a true tex3D (trilinear, border zero).
 * Unlike make_vol_texture (which uses layered 2D), this allows the ray to
 * cross Y-slice boundaries — required for general ZYZ Euler rotations.
 */
template <typename T>
static cudaTextureObject_t make_vol_texture_3d(
    const T *d_vol, int Ny, int Nz, int Nx,
    cudaArray_t &out_array, cudaStream_t stream
) {
    cudaChannelFormatDesc fmt = make_channel_desc<T>();
    /* extent: (width=Nx, height=Nz, depth=Ny) — matches C-order [Ny][Nz][Nx] */
    FASTTOMO_CUDA_CHECK(cudaMalloc3DArray(&out_array, &fmt,
                                 make_cudaExtent(Nx, Nz, Ny), 0 /* no cudaArrayLayered */));

    cudaMemcpy3DParms cp = {};
    cp.srcPtr   = make_cudaPitchedPtr((void *)d_vol,
                                      Nx * sizeof(T), Nx, Nz);
    cp.dstArray = out_array;
    cp.extent   = make_cudaExtent(Nx, Nz, Ny);
    cp.kind     = cudaMemcpyDeviceToDevice;
    FASTTOMO_CUDA_CHECK(cudaMemcpy3DAsync(&cp, stream));

    cudaResourceDesc rd = {};
    rd.resType         = cudaResourceTypeArray;
    rd.res.array.array = out_array;

    cudaTextureDesc td = {};
    td.addressMode[0]  = cudaAddressModeBorder;
    td.addressMode[1]  = cudaAddressModeBorder;
    td.addressMode[2]  = cudaAddressModeBorder;
    td.filterMode      = cudaFilterModeLinear;
    td.readMode        = cudaReadModeElementType;
    td.normalizedCoords = 0;

    cudaTextureObject_t tex;
    FASTTOMO_CUDA_CHECK(cudaCreateTextureObject(&tex, &rd, &td, nullptr));
    return tex;
}

/* forward_project_rotmat_kernel
 * Grid: (⌈N_det/16⌉, ⌈Ny/16⌉, n_angles)   Block: (16, 16, 1)
 * Each thread owns one (detector-u, detector-v, angle) output sample.
 * Rotation matrix M (row-major, 9 floats per projection):
 *   detector-u = M[0]*x + M[1]*y + M[2]*z
 *   detector-v = M[3]*x + M[4]*y + M[5]*z
 * Ray tracing (R^T applied to ray direction):
 *   x(s) = u*M[0] + v*M[3] + s*M[6]
 *   y(s) = u*M[1] + v*M[4] + s*M[7]
 *   z(s) = u*M[2] + v*M[5] + s*M[8]
 * tex3D axis mapping: tex3D(tex, ix, iz, iy) → vol[iy][iz][ix]
 */
template <typename T>
__global__ static void forward_project_rotmat_kernel(
    cudaTextureObject_t  vol_tex,
    T *__restrict__      sinogram,         /* [n_angles, Ny, N_det] */
    const float *__restrict__ matrices,    /* [n_angles * 9] row-major */
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    float half_det, float half_x, float half_z, float half_y,
    int ray_len
) {
    int iu = blockIdx.x * blockDim.x + threadIdx.x;  /* detector u */
    int iv = blockIdx.y * blockDim.y + threadIdx.y;  /* detector v = Y-slice */
    int ia = blockIdx.z;

    if (iu >= N_det || iv >= Ny) return;

    const float *M = matrices + ia * 9;
    float u = iu - half_det;
    float v = iv - half_y;

    /* precompute origin of ray at s=0 and step increments */
    float ox = u * M[0] + v * M[3];
    float oy = u * M[1] + v * M[4];
    float oz = u * M[2] + v * M[5];
    float dx = M[6], dy = M[7], dz = M[8];  /* ray direction */

    float sum = 0.0f;
    for (int step = -ray_len; step <= ray_len; step++) {
        float s  = (float)step;
        float tx = ox + s * dx + half_x + 0.5f;
        float tz = oz + s * dz + half_z + 0.5f;
        float ty = oy + s * dy + half_y + 0.5f;
        /* tex3D: (x→width=Nx, z→height=Nz, y→depth=Ny). Always read as
         * float — hardware promotes fp16 texels on filtered read. */
        sum += tex3D<float>(vol_tex, tx, tz, ty);
    }
    sinogram[((size_t)ia * Ny + iv) * N_det + iu] = float_to_t<T>(sum);
}

template <typename T>
static void forward_project_rotmat_cuda_impl(
    const T     *d_volume,
    T           *d_sinogram,
    const float *d_matrices,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
) {
    float half_det = (N_det - 1) * 0.5f;
    float half_x   = (Nx   - 1) * 0.5f;
    float half_z   = (Nz   - 1) * 0.5f;
    float half_y   = (Ny   - 1) * 0.5f;
    int   ray_len  = (int)ceilf(0.5f * sqrtf((float)(Nx*Nx + Ny*Ny + Nz*Nz))) + 1;

    cudaArray_t vol_array;
    cudaTextureObject_t vol_tex = make_vol_texture_3d<T>(d_volume, Ny, Nz, Nx, vol_array, stream);

    dim3 block(16, 16, 1);
    dim3 grid((N_det + 15) / 16,
              (Ny    + 15) / 16,
              n_angles);

    forward_project_rotmat_kernel<T><<<grid, block, 0, stream>>>(
        vol_tex, d_sinogram, d_matrices,
        Ny, Nz, Nx, n_angles, N_det,
        half_det, half_x, half_z, half_y, ray_len
    );
    CUDA_CHECK_KERNEL();

    FASTTOMO_CUDA_CHECK(cudaDestroyTextureObject(vol_tex));
    FASTTOMO_CUDA_CHECK(cudaFreeArray(vol_array));
}

void forward_project_rotmat_cuda(
    int io_dtype,
    const void  *d_volume,
    void        *d_sinogram,
    const float *d_matrices,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
) {
    check_io_dtype(io_dtype, "forward_project_rotmat_cuda");
    if (io_dtype == 0)
        forward_project_rotmat_cuda_impl<float>(
            (const float *)d_volume, (float *)d_sinogram,
            d_matrices, Ny, Nz, Nx, n_angles, N_det, stream);
    else
        forward_project_rotmat_cuda_impl<__half>(
            (const __half *)d_volume, (__half *)d_sinogram,
            d_matrices, Ny, Nz, Nx, n_angles, N_det, stream);
}

template <typename T>
static void forward_project_rotmat_cuda_cached_impl(
    TextureCache *vol_cache,
    const T      *d_volume,
    T            *d_sinogram,
    const float  *d_matrices,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
) {
    float half_det = (N_det - 1) * 0.5f;
    float half_x   = (Nx   - 1) * 0.5f;
    float half_z   = (Nz   - 1) * 0.5f;
    float half_y   = (Ny   - 1) * 0.5f;
    int   ray_len  = (int)ceilf(0.5f * sqrtf((float)(Nx*Nx + Ny*Ny + Nz*Nz))) + 1;

    tc_upload(vol_cache, d_volume, stream);

    dim3 block(16, 16, 1);
    dim3 grid((N_det + 15) / 16,
              (Ny    + 15) / 16,
              n_angles);

    forward_project_rotmat_kernel<T><<<grid, block, 0, stream>>>(
        vol_cache->tex, d_sinogram, d_matrices,
        Ny, Nz, Nx, n_angles, N_det,
        half_det, half_x, half_z, half_y, ray_len
    );
    CUDA_CHECK_KERNEL();
}

void forward_project_rotmat_cuda_cached(
    int io_dtype,
    TextureCache *vol_cache,
    const void   *d_volume,
    void         *d_sinogram,
    const float  *d_matrices,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
) {
    check_io_dtype(io_dtype, "forward_project_rotmat_cuda_cached");
    if (!vol_cache) throw std::runtime_error("forward_project_rotmat_cuda_cached: null cache");
    if (vol_cache->kind  != TC_VOL_3D ||
        vol_cache->dim_a != Ny || vol_cache->dim_b != Nz || vol_cache->dim_c != Nx)
        throw std::runtime_error("forward_project_rotmat_cuda_cached: cache shape mismatch");
    int cache_dtype = (vol_cache->dtype == TC_DTYPE_FP32) ? 0 : 1;
    if (cache_dtype != io_dtype)
        throw std::runtime_error("forward_project_rotmat_cuda_cached: cache dtype mismatch");

    if (io_dtype == 0)
        forward_project_rotmat_cuda_cached_impl<float>(
            vol_cache, (const float *)d_volume, (float *)d_sinogram,
            d_matrices, Ny, Nz, Nx, n_angles, N_det, stream);
    else
        forward_project_rotmat_cuda_cached_impl<__half>(
            vol_cache, (const __half *)d_volume, (__half *)d_sinogram,
            d_matrices, Ny, Nz, Nx, n_angles, N_det, stream);
}

/* ── gradient of 3D rotmat forward projection w.r.t. rotation matrix ─────
 * Grid:  (⌈N_det/16⌉, ⌈Ny/16⌉, n_angles)   Block: (16, 16, 1)
 * Same detector-driven layout as forward_project_rotmat_kernel, with one
 * block per angle's matrix entries reduced via shared memory.
 *
 * Forward sample at ray step s, detector cell (iu, iv):
 *   u = iu - half_det,  v = iv - half_y,  s = step
 *   tx = u*M[0] + v*M[3] + s*M[6] + half_x
 *   ty = u*M[1] + v*M[4] + s*M[7] + half_y
 *   tz = u*M[2] + v*M[5] + s*M[8] + half_z
 *   sino += vol(tx, ty, tz)   (trilinear)
 *
 * ∂tx/∂M[{0,3,6}] = {u, v, s}, ∂ty/∂M[{1,4,7}] = {u, v, s},
 * ∂tz/∂M[{2,5,8}] = {u, v, s}. Spatial derivatives ∂vol/∂{x,y,z}
 * computed analytically from trilinear interpolation weights.
 */
__global__ static void grad_rotmat_kernel(
    const float *__restrict__ volume,        /* [Ny, Nz, Nx] */
    const float *__restrict__ grad_sino,     /* [n_angles, Ny, N_det] */
    const float *__restrict__ matrices,      /* [n_angles*9] row-major */
    float       *__restrict__ grad_matrices, /* [n_angles*9] (zeroed by caller) */
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    float half_det, float half_x, float half_y, float half_z,
    int ray_len
) {
    int iu = blockIdx.x * blockDim.x + threadIdx.x;
    int iv = blockIdx.y * blockDim.y + threadIdx.y;
    int ia = blockIdx.z;

    int tid       = threadIdx.y * blockDim.x + threadIdx.x;
    int nthreads  = blockDim.x * blockDim.y;          /* 256 */

    float gM[9] = {0,0,0,0,0,0,0,0,0};

    if (iu < N_det && iv < Ny) {
        const float *M = matrices + (size_t)ia * 9;
        float u = (float)iu - half_det;
        float v = (float)iv - half_y;
        float ox = u * M[0] + v * M[3];
        float oy = u * M[1] + v * M[4];
        float oz = u * M[2] + v * M[5];
        float dxs = M[6], dys = M[7], dzs = M[8];

        float gs = grad_sino[((size_t)ia * Ny + iv) * N_det + iu];

        if (gs != 0.0f) {
            for (int step = -ray_len; step <= ray_len; step++) {
                float s  = (float)step;
                float fx = ox + s * dxs + half_x;
                float fy = oy + s * dys + half_y;
                float fz = oz + s * dzs + half_z;

                if (fx < 0.0f || fx >= (float)(Nx - 1) ||
                    fy < 0.0f || fy >= (float)(Ny - 1) ||
                    fz < 0.0f || fz >= (float)(Nz - 1))
                    continue;

                int ix0 = (int)fx, iy0 = (int)fy, iz0 = (int)fz;
                int ix1 = ix0 + 1, iy1 = iy0 + 1, iz1 = iz0 + 1;
                float wx = fx - (float)ix0;
                float wy = fy - (float)iy0;
                float wz = fz - (float)iz0;
                float ux = 1.0f - wx, uy = 1.0f - wy, uz = 1.0f - wz;

                /* volume[iy][iz][ix] linearised */
                size_t row_y0 = (size_t)iy0 * Nz;
                size_t row_y1 = (size_t)iy1 * Nz;
                float v000 = volume[(row_y0 + iz0) * Nx + ix0];
                float v001 = volume[(row_y0 + iz0) * Nx + ix1];
                float v010 = volume[(row_y0 + iz1) * Nx + ix0];
                float v011 = volume[(row_y0 + iz1) * Nx + ix1];
                float v100 = volume[(row_y1 + iz0) * Nx + ix0];
                float v101 = volume[(row_y1 + iz0) * Nx + ix1];
                float v110 = volume[(row_y1 + iz1) * Nx + ix0];
                float v111 = volume[(row_y1 + iz1) * Nx + ix1];

                /* analytic derivatives of trilinear interpolation */
                float dfx = uy * uz * (v001 - v000) + wy * uz * (v101 - v100)
                          + uy * wz * (v011 - v010) + wy * wz * (v111 - v110);
                float dfy = ux * uz * (v100 - v000) + wx * uz * (v101 - v001)
                          + ux * wz * (v110 - v010) + wx * wz * (v111 - v011);
                float dfz = ux * uy * (v010 - v000) + wx * uy * (v011 - v001)
                          + ux * wy * (v110 - v100) + wx * wy * (v111 - v101);

                float gdfx = gs * dfx, gdfy = gs * dfy, gdfz = gs * dfz;
                gM[0] += gdfx * u;  gM[1] += gdfy * u;  gM[2] += gdfz * u;
                gM[3] += gdfx * v;  gM[4] += gdfy * v;  gM[5] += gdfz * v;
                gM[6] += gdfx * s;  gM[7] += gdfy * s;  gM[8] += gdfz * s;
            }
        }
    }

    /* Block-level reduction. smem layout: [9 * nthreads] flat. */
    extern __shared__ float smem[];
    #pragma unroll
    for (int k = 0; k < 9; k++) smem[k * nthreads + tid] = gM[k];
    __syncthreads();

    for (int stride = nthreads >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            #pragma unroll
            for (int k = 0; k < 9; k++)
                smem[k * nthreads + tid] += smem[k * nthreads + tid + stride];
        }
        __syncthreads();
    }

    if (tid < 9)
        atomicAdd(&grad_matrices[(size_t)ia * 9 + tid], smem[tid * nthreads]);
}

void grad_rotmat_cuda(
    const float *d_volume,
    const float *d_grad_sino,
    const float *d_matrices,
    float       *d_grad_matrices,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
) {
    float half_det = (N_det - 1) * 0.5f;
    float half_x   = (Nx   - 1) * 0.5f;
    float half_y   = (Ny   - 1) * 0.5f;
    float half_z   = (Nz   - 1) * 0.5f;
    int   ray_len  = (int)ceilf(0.5f * sqrtf((float)(Nx*Nx + Ny*Ny + Nz*Nz))) + 1;

    dim3 block(16, 16, 1);
    dim3 grid((N_det + 15) / 16,
              (Ny    + 15) / 16,
              n_angles);
    size_t smem = 9 * (size_t)(block.x * block.y) * sizeof(float);

    enable_extended_smem_if_needed((const void *)grad_rotmat_kernel, smem);

    grad_rotmat_kernel<<<grid, block, smem, stream>>>(
        d_volume, d_grad_sino, d_matrices, d_grad_matrices,
        Ny, Nz, Nx, n_angles, N_det,
        half_det, half_x, half_y, half_z, ray_len
    );
    CUDA_CHECK_KERNEL();
}

/* back_project_rotmat_kernel
 * Grid: (⌈Nx/8⌉, ⌈Nz/8⌉, ⌈Ny/4⌉)   Block: (8, 8, 4)
 * Each thread owns one (ix, iz, iy) voxel and loops over all angles.
 * Rotation matrices loaded into shared memory (9 * n_angles floats).
 * Sinogram sampled with tex2DLayered (n_angles layers of [Ny, N_det]).
 */
template <typename T>
__global__ static void back_project_rotmat_kernel(
    cudaTextureObject_t  sino_tex,        /* n_angles layers of [Ny, N_det] */
    T *__restrict__      volume,          /* [Ny, Nz, Nx] */
    const float *__restrict__ matrices,   /* [n_angles * 9] row-major */
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    float half_det, float half_x, float half_z, float half_y
) {
    /* Rotation matrices stay fp32 in shared memory. */
    extern __shared__ float sh_mats[];   /* 9 * n_angles floats */

    int tid       = threadIdx.x + threadIdx.y * blockDim.x
                  + threadIdx.z * blockDim.x * blockDim.y;
    int block_sz  = blockDim.x * blockDim.y * blockDim.z;
    for (int k = tid; k < n_angles * 9; k += block_sz)
        sh_mats[k] = matrices[k];
    __syncthreads();

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int iy = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iz >= Nz || iy >= Ny) return;

    float x = ix - half_x;
    float y = iy - half_y;
    float z = iz - half_z;
    float val = 0.0f;

    for (int ia = 0; ia < n_angles; ia++) {
        const float *M = sh_mats + ia * 9;
        /* u = R[0,:] · (x,y,z)   v = R[1,:] · (x,y,z) */
        float u_det = M[0]*x + M[1]*y + M[2]*z;
        float v_det = M[3]*x + M[4]*y + M[5]*z;
        /* tex2DLayered: x→N_det, y→Ny, layer→ia. Always read as float
         * — hardware promotes fp16 texels on filtered read. */
        val += tex2DLayered<float>(sino_tex,
                                   u_det + half_det + 0.5f,
                                   v_det + half_y   + 0.5f,
                                   ia);
    }

    volume[((size_t)iy * Nz + iz) * Nx + ix] = float_to_t<T>(val);
}

template <typename T>
static void back_project_rotmat_cuda_impl(
    const T     *d_sinogram,
    T           *d_volume,
    const float *d_matrices,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
) {
    float half_det = (N_det - 1) * 0.5f;
    float half_x   = (Nx   - 1) * 0.5f;
    float half_z   = (Nz   - 1) * 0.5f;
    float half_y   = (Ny   - 1) * 0.5f;

    cudaArray_t sino_array;
    cudaTextureObject_t sino_tex =
        make_sino_texture<T>(d_sinogram, n_angles, Ny, N_det, sino_array, stream);

    dim3 block(8, 8, 4);
    dim3 grid((Nx + 7) / 8,
              (Nz + 7) / 8,
              (Ny + 3) / 4);
    size_t smem = 9 * (size_t)n_angles * sizeof(float);

    enable_extended_smem_if_needed((const void *)back_project_rotmat_kernel<T>, smem);

    back_project_rotmat_kernel<T><<<grid, block, smem, stream>>>(
        sino_tex, d_volume, d_matrices,
        Ny, Nz, Nx, n_angles, N_det,
        half_det, half_x, half_z, half_y
    );
    CUDA_CHECK_KERNEL();

    FASTTOMO_CUDA_CHECK(cudaDestroyTextureObject(sino_tex));
    FASTTOMO_CUDA_CHECK(cudaFreeArray(sino_array));
}

void back_project_rotmat_cuda(
    int io_dtype,
    const void  *d_sinogram,
    void        *d_volume,
    const float *d_matrices,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
) {
    check_io_dtype(io_dtype, "back_project_rotmat_cuda");
    if (io_dtype == 0)
        back_project_rotmat_cuda_impl<float>(
            (const float *)d_sinogram, (float *)d_volume,
            d_matrices, Ny, Nz, Nx, n_angles, N_det, stream);
    else
        back_project_rotmat_cuda_impl<__half>(
            (const __half *)d_sinogram, (__half *)d_volume,
            d_matrices, Ny, Nz, Nx, n_angles, N_det, stream);
}

template <typename T>
static void back_project_rotmat_cuda_cached_impl(
    TextureCache *sino_cache,
    const T      *d_sinogram,
    T            *d_volume,
    const float  *d_matrices,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
) {
    float half_det = (N_det - 1) * 0.5f;
    float half_x   = (Nx   - 1) * 0.5f;
    float half_z   = (Nz   - 1) * 0.5f;
    float half_y   = (Ny   - 1) * 0.5f;

    tc_upload(sino_cache, d_sinogram, stream);

    dim3 block(8, 8, 4);
    dim3 grid((Nx + 7) / 8,
              (Nz + 7) / 8,
              (Ny + 3) / 4);
    size_t smem = 9 * (size_t)n_angles * sizeof(float);

    enable_extended_smem_if_needed((const void *)back_project_rotmat_kernel<T>, smem);

    back_project_rotmat_kernel<T><<<grid, block, smem, stream>>>(
        sino_cache->tex, d_volume, d_matrices,
        Ny, Nz, Nx, n_angles, N_det,
        half_det, half_x, half_z, half_y
    );
    CUDA_CHECK_KERNEL();
}

void back_project_rotmat_cuda_cached(
    int io_dtype,
    TextureCache *sino_cache,
    const void   *d_sinogram,
    void         *d_volume,
    const float  *d_matrices,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
) {
    check_io_dtype(io_dtype, "back_project_rotmat_cuda_cached");
    if (!sino_cache) throw std::runtime_error("back_project_rotmat_cuda_cached: null cache");
    if (sino_cache->kind  != TC_SINO_LAYERED ||
        sino_cache->dim_a != n_angles || sino_cache->dim_b != Ny || sino_cache->dim_c != N_det)
        throw std::runtime_error("back_project_rotmat_cuda_cached: cache shape mismatch");
    int cache_dtype = (sino_cache->dtype == TC_DTYPE_FP32) ? 0 : 1;
    if (cache_dtype != io_dtype)
        throw std::runtime_error("back_project_rotmat_cuda_cached: cache dtype mismatch");

    if (io_dtype == 0)
        back_project_rotmat_cuda_cached_impl<float>(
            sino_cache, (const float *)d_sinogram, (float *)d_volume,
            d_matrices, Ny, Nz, Nx, n_angles, N_det, stream);
    else
        back_project_rotmat_cuda_cached_impl<__half>(
            sino_cache, (const __half *)d_sinogram, (__half *)d_volume,
            d_matrices, Ny, Nz, Nx, n_angles, N_det, stream);
}

/* ── RESIRE voxel-driven projectors ──────────────────────────────────────────
 * forward_project_resire_kernel  — voxel-driven scatter with 4-point
 *   supersampling; atomicAdd to the sinogram slice for this projection.
 * back_project_resire_kernel     — voxel-driven gather with 4-point
 *   supersampling; accumulates into volume. No atomic: each voxel is owned
 *   by exactly one thread per launch, and per-projection launches on the
 *   same stream are serialised so the read-modify-write is safe across
 *   projections.
 *
 * Each launch processes ONE projection. The host loops over n_angles.
 * Supersampling: 4 sub-voxel offsets (±0.25, ±0.25) in volume XZ space are
 * rotated to detector coordinates via rows 0–1 of the projection matrix M.
 * The host divides the output by 4 to normalise to the single-sample scale.
 */

__global__ static void forward_project_resire_kernel(
    const float *__restrict__ volume,     /* [Ny, Nz, Nx] */
    float *__restrict__       sino_slice, /* [Ny, N_det] for this projection */
    const float *__restrict__ M,          /* [9] rotation matrix */
    int Ny, int Nz, int Nx, int N_det,
    float half_x, float half_y, float half_z, float half_det
) {
    __shared__ float sh_M[9];
    if (threadIdx.y == 0 && threadIdx.x < 9)
        sh_M[threadIdx.x] = M[threadIdx.x];
    __syncthreads();

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int iy = (int)blockIdx.z;

    if (ix >= Nx || iz >= Nz) return;

    float x = ix - half_x;
    float y = iy - half_y;
    float z = iz - half_z;

    float u0 = sh_M[0]*x + sh_M[1]*y + sh_M[2]*z + half_det;
    float v0 = sh_M[3]*x + sh_M[4]*y + sh_M[5]*z + half_y;

    const float data_val = volume[((size_t)iy * Nz + iz) * Nx + ix];

    const float sdx[4] = { 0.25f,  0.25f, -0.25f, -0.25f };
    const float sdz[4] = { 0.25f, -0.25f,  0.25f, -0.25f };

    for (int s = 0; s < 4; s++) {
        float u_s = u0 + sh_M[0]*sdx[s] + sh_M[2]*sdz[s];
        float v_s = v0 + sh_M[3]*sdx[s] + sh_M[5]*sdz[s];

        int u1 = (int)floorf(u_s);  int u2 = u1 + 1;
        int v1 = (int)floorf(v_s);  int v2 = v1 + 1;

        if ((u1 < 0 && u2 < 0) || (u1 >= N_det && u2 >= N_det)) continue;
        if ((v1 < 0 && v2 < 0) || (v1 >= Ny   && v2 >= Ny  )) continue;

        float wu2 = u_s - u1,  wu1 = 1.0f - wu2;
        float wv2 = v_s - v1,  wv1 = 1.0f - wv2;

        bool u1ok = (u1 >= 0 && u1 < N_det);
        bool u2ok = (u2 >= 0 && u2 < N_det);
        bool v1ok = (v1 >= 0 && v1 < Ny);
        bool v2ok = (v2 >= 0 && v2 < Ny);

        if (u1ok && v1ok) atomicAdd(&sino_slice[v1*N_det + u1], wu1*wv1*data_val);
        if (u2ok && v1ok) atomicAdd(&sino_slice[v1*N_det + u2], wu2*wv1*data_val);
        if (u1ok && v2ok) atomicAdd(&sino_slice[v2*N_det + u1], wu1*wv2*data_val);
        if (u2ok && v2ok) atomicAdd(&sino_slice[v2*N_det + u2], wu2*wv2*data_val);
    }
}

__global__ static void back_project_resire_kernel(
    const float *__restrict__ sino_slice, /* [Ny, N_det] for this projection */
    float *__restrict__       volume,     /* [Ny, Nz, Nx] — accumulated */
    const float *__restrict__ M,          /* [9] rotation matrix */
    int Ny, int Nz, int Nx, int N_det,
    float half_x, float half_y, float half_z, float half_det
) {
    __shared__ float sh_M[9];
    if (threadIdx.y == 0 && threadIdx.x < 9)
        sh_M[threadIdx.x] = M[threadIdx.x];
    __syncthreads();

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int iy = (int)blockIdx.z;

    if (ix >= Nx || iz >= Nz) return;

    float x = ix - half_x;
    float y = iy - half_y;
    float z = iz - half_z;

    float u0 = sh_M[0]*x + sh_M[1]*y + sh_M[2]*z + half_det;
    float v0 = sh_M[3]*x + sh_M[4]*y + sh_M[5]*z + half_y;

    const float sdx[4] = { 0.25f,  0.25f, -0.25f, -0.25f };
    const float sdz[4] = { 0.25f, -0.25f,  0.25f, -0.25f };

    float val = 0.0f;

    for (int s = 0; s < 4; s++) {
        float u_s = u0 + sh_M[0]*sdx[s] + sh_M[2]*sdz[s];
        float v_s = v0 + sh_M[3]*sdx[s] + sh_M[5]*sdz[s];

        if (u_s < -0.5f || u_s > N_det - 0.5f || v_s < -0.5f || v_s > Ny - 0.5f) continue;

        int u1 = max(0, (int)floorf(u_s));  int u2 = min(N_det - 1, u1 + 1);
        int v1 = max(0, (int)floorf(v_s));  int v2 = min(Ny   - 1, v1 + 1);

        float wu2 = u_s - u1,  wu1 = 1.0f - wu2;
        float wv2 = v_s - v1,  wv1 = 1.0f - wv2;

        val += wu1*wv1 * sino_slice[v1*N_det + u1]
             + wu2*wv1 * sino_slice[v1*N_det + u2]
             + wu1*wv2 * sino_slice[v2*N_det + u1]
             + wu2*wv2 * sino_slice[v2*N_det + u2];
    }

    volume[((size_t)iy * Nz + iz) * Nx + ix] += val;
}

void forward_project_resire_cuda(
    const float *d_volume,
    float       *d_sinogram,
    const float *d_matrices,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
) {
    float half_det = (N_det - 1) * 0.5f;
    float half_x   = (Nx   - 1) * 0.5f;
    float half_z   = (Nz   - 1) * 0.5f;
    float half_y   = (Ny   - 1) * 0.5f;

    dim3 block(16, 16, 1);
    dim3 grid((Nx + 15) / 16, (Nz + 15) / 16, Ny);

    for (int ia = 0; ia < n_angles; ia++) {
        forward_project_resire_kernel<<<grid, block, 0, stream>>>(
            d_volume,
            d_sinogram + (size_t)ia * Ny * N_det,
            d_matrices + ia * 9,
            Ny, Nz, Nx, N_det,
            half_x, half_y, half_z, half_det
        );
    }
    CUDA_CHECK_KERNEL();
}

void back_project_resire_cuda(
    const float *d_sinogram,
    float       *d_volume,
    const float *d_matrices,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
) {
    float half_det = (N_det - 1) * 0.5f;
    float half_x   = (Nx   - 1) * 0.5f;
    float half_z   = (Nz   - 1) * 0.5f;
    float half_y   = (Ny   - 1) * 0.5f;

    dim3 block(16, 16, 1);
    dim3 grid((Nx + 15) / 16, (Nz + 15) / 16, Ny);

    for (int ia = 0; ia < n_angles; ia++) {
        back_project_resire_kernel<<<grid, block, 0, stream>>>(
            d_sinogram + (size_t)ia * Ny * N_det,
            d_volume,
            d_matrices + ia * 9,
            Ny, Nz, Nx, N_det,
            half_x, half_y, half_z, half_det
        );
    }
    CUDA_CHECK_KERNEL();
}

/* ── GENFIRE 2D Fourier gridding ─────────────────────────────────────────── */

__global__ static void genfire_gridding_2d_kernel(
    const float *fk_real, const float *fk_imag,
    const float *cos_a,   const float *sin_a,
    float *grid_real, float *grid_imag, float *weight_grid,
    int bigN, int Ny, int n_angles, float cutoff_dist
) {
    int ixi = blockIdx.x * blockDim.x + threadIdx.x;
    int ia  = (int)blockIdx.y;
    int iy  = (int)blockIdx.z;

    if (ixi >= bigN || ia >= n_angles || iy >= Ny) return;

    int   half = bigN / 2;
    float xi   = (float)(ixi - half);
    float kx   = xi * cos_a[ia];
    float kz   = xi * sin_a[ia];
    int   ikx  = __float2int_rn(kx);
    int   ikz  = __float2int_rn(kz);
    float dx   = kx - (float)ikx;
    float dz   = kz - (float)ikz;
    float dist = sqrtf(dx * dx + dz * dz);
    if (dist > cutoff_dist) return;

    int gx = ikx + half;
    int gz = ikz + half;
    if (gx < 0 || gx >= bigN || gz < 0 || gz >= bigN) return;

    float  w       = 1.0f / (dist + 1e-5f);
    size_t fk_idx  = ((size_t)ia * Ny + iy) * bigN + (size_t)ixi;
    size_t out_idx = (size_t)iy * bigN * bigN + (size_t)gz * bigN + (size_t)gx;

    atomicAdd(&grid_real[out_idx],    w * fk_real[fk_idx]);
    atomicAdd(&grid_imag[out_idx],    w * fk_imag[fk_idx]);
    atomicAdd(&weight_grid[out_idx],  w);
}

void genfire_gridding_2d_cuda(
    const float *fk_real, const float *fk_imag,
    const float *cos_a,   const float *sin_a,
    float *grid_real, float *grid_imag, float *weight_grid,
    int bigN, int Ny, int n_angles, float cutoff_dist,
    cudaStream_t stream
) {
    dim3 block(128, 1, 1);
    dim3 grid((bigN + 127) / 128, (unsigned)n_angles, (unsigned)Ny);
    genfire_gridding_2d_kernel<<<grid, block, 0, stream>>>(
        fk_real, fk_imag, cos_a, sin_a,
        grid_real, grid_imag, weight_grid,
        bigN, Ny, n_angles, cutoff_dist
    );
    CUDA_CHECK_KERNEL();
}

/* ── gradient of forward projection w.r.t. tilt angles ───────────────────
 * Grid:  (ceil(N_det/16), ceil(n_angles/16), Ny)   Block: (16, 16, 1)
 * Same detector-driven layout as forward_project_kernel.
 *
 * Reduction strategy: each (ia, ty) row of 16 threads sits inside a single
 * warp (block is 16×16 = 256 threads = 8 warps; for ty even, those 16
 * threads occupy lanes 0..15; for ty odd, lanes 16..31). A half-warp
 * butterfly via __shfl_xor_sync collapses the 16 detector cells, after
 * which threadIdx.x==0 issues one atomicAdd per row.
 */
__global__ static void grad_angle_kernel(
    const float *__restrict__ volume,     /* [Ny, Nz, Nx] */
    const float *__restrict__ grad_sino,  /* [n_angles, Ny, N_det] */
    const float *__restrict__ cos_a,
    const float *__restrict__ sin_a,
    float       *__restrict__ grad_angles,/* [n_angles] */
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    float half_det, float half_x, float half_z, int ray_len
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int ia = blockIdx.y * blockDim.y + threadIdx.y;
    int iy = (int)blockIdx.z;

    float local_g = 0.0f;

    if (id < N_det && ia < n_angles) {
        float ca = cos_a[ia], sa = sin_a[ia];
        float t   = id - half_det;
        float gs  = grad_sino[((size_t)ia * Ny + iy) * N_det + id];

        if (gs != 0.0f) {
            const float *slice = volume + (size_t)iy * Nz * Nx;

            for (int step = -ray_len; step <= ray_len; step++) {
                float s  = (float)step;
                float x  = t * ca - s * sa;
                float z  = t * sa + s * ca;
                float fx = x + half_x;
                float fz = z + half_z;

                if (fx < 0.0f || fx >= (float)(Nx - 1) ||
                    fz < 0.0f || fz >= (float)(Nz - 1))
                    continue;

                int ix0 = (int)fx, iz0 = (int)fz;
                int ix1 = ix0 + 1, iz1 = iz0 + 1;
                float wx = fx - ix0, wz = fz - iz0;

                float v00 = slice[(size_t)iz0 * Nx + ix0];
                float v01 = slice[(size_t)iz0 * Nx + ix1];
                float v10 = slice[(size_t)iz1 * Nx + ix0];
                float v11 = slice[(size_t)iz1 * Nx + ix1];

                float dvol_dx = (1.0f - wz) * (v01 - v00) + wz * (v11 - v10);
                float dvol_dz = (1.0f - wx) * (v10 - v00) + wx * (v11 - v01);

                local_g += gs * (dvol_dx * (-z) + dvol_dz * x);
            }
        }
    }

    /* Half-warp shuffle reduction over threadIdx.x (16 lanes share each ia row).
     * Lanes for ty even occupy 0..15 within their warp; for ty odd, 16..31.
     * The xor butterfly with offsets ≤ 8 stays within the half-warp, so a
     * full-warp mask gives the correct partial sum on every lane. */
    unsigned mask = 0xFFFFFFFFu;
    local_g += __shfl_xor_sync(mask, local_g, 8);
    local_g += __shfl_xor_sync(mask, local_g, 4);
    local_g += __shfl_xor_sync(mask, local_g, 2);
    local_g += __shfl_xor_sync(mask, local_g, 1);

    if (threadIdx.x == 0 && ia < n_angles)
        atomicAdd(&grad_angles[ia], local_g);
}

void grad_angle_cuda(
    const float *d_volume,
    const float *d_grad_sino,
    const float *d_cos_a,
    const float *d_sin_a,
    float       *d_grad_angles,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
) {
    float half_det = (N_det - 1) * 0.5f;
    float half_x   = (Nx   - 1) * 0.5f;
    float half_z   = (Nz   - 1) * 0.5f;
    int   ray_len  = (int)ceilf(0.5f * sqrtf((float)(Nx * Nx + Nz * Nz))) + 1;

    dim3 block(16, 16, 1);
    dim3 grid((N_det    + 15) / 16,
              (n_angles + 15) / 16,
              Ny);

    grad_angle_kernel<<<grid, block, 0, stream>>>(
        d_volume, d_grad_sino, d_cos_a, d_sin_a, d_grad_angles,
        Ny, Nz, Nx, n_angles, N_det,
        half_det, half_x, half_z, ray_len
    );
    CUDA_CHECK_KERNEL();
}


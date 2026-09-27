#include "texture_cache.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <string>
#include <cstdlib>

static cudaChannelFormatDesc tc_channel_desc(tc_dtype_t dtype, int channels) {
    /* cudaCreateChannelDesc<__half> is not specialized in stock CUDA
     * headers; route fp16 through explicit (16,16,16,16) for 4-channel
     * packing or cudaCreateChannelDescHalf() for 1-channel. */
    if (dtype == TC_DTYPE_FP32) {
        if (channels == 1) return cudaCreateChannelDesc<float>();
        if (channels == 4) return cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    }
    if (dtype == TC_DTYPE_FP16) {
        if (channels == 1) return cudaCreateChannelDescHalf();
        if (channels == 4) return cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindFloat);
    }
    throw std::runtime_error("tc: unsupported (dtype, channels) combination");
}

/* Resolve the packing axis for each cache kind:
 *   TC_VOL_LAYERED — Ny is the layer dim (dim_a). Pack Ny.
 *   TC_VOL_3D      — Ny is the depth dim (dim_a). Pack Ny.
 *   TC_SINO_LAYERED — Ny is the height dim (dim_b). Pack Ny.
 * The cudaArray "layer" / "depth" extent then becomes Ny/4 for vol kinds,
 * and the array "height" becomes Ny/4 for sino kind. */
static int packing_axis_size(tc_kind_t kind, int dim_a, int dim_b, int dim_c) {
    switch (kind) {
        case TC_VOL_LAYERED:
        case TC_VOL_3D:        return dim_a;
        case TC_SINO_LAYERED:  return dim_b;
    }
    throw std::runtime_error("tc: unknown kind");
}

/* Throw a clear std::invalid_argument when a cudaArray extent exceeds the
 * current device's texture limits (P4). Without this the failure is a
 * generic "invalid argument" from cudaMalloc3DArray. */
static void check_extent_limits(tc_kind_t kind, const cudaExtent &e) {
    int dev = 0;
    FASTTOMO_CUDA_CHECK(cudaGetDevice(&dev));
    auto attr = [dev](cudaDeviceAttr a) {
        int v = 0;
        FASTTOMO_CUDA_CHECK(cudaDeviceGetAttribute(&v, a, dev));
        return (size_t)v;
    };
    size_t max_w, max_h, max_d;
    const char *what;
    if (kind == TC_VOL_3D) {
        max_w = attr(cudaDevAttrMaxTexture3DWidth);
        max_h = attr(cudaDevAttrMaxTexture3DHeight);
        max_d = attr(cudaDevAttrMaxTexture3DDepth);
        what  = "3D texture (width=Nx, height=Nz, depth=Ny)";
    } else {
        max_w = attr(cudaDevAttrMaxTexture2DLayeredWidth);
        max_h = attr(cudaDevAttrMaxTexture2DLayeredHeight);
        max_d = attr(cudaDevAttrMaxTexture2DLayeredLayers);
        what  = (kind == TC_VOL_LAYERED)
              ? "layered texture (width=Nx, height=Nz, layers=Ny)"
              : "layered texture (width=N_det, height=Ny, layers=n_angles)";
    }
    if (e.width > max_w || e.height > max_h || e.depth > max_d)
        throw std::invalid_argument(
            std::string("tc_create: extent (") + std::to_string(e.width) + ", "
            + std::to_string(e.height) + ", " + std::to_string(e.depth)
            + ") exceeds this device's " + what + " limit ("
            + std::to_string(max_w) + ", " + std::to_string(max_h) + ", "
            + std::to_string(max_d) + ")");
}

/* Test hook: make the next tc_create throw at a given stage so the
 * regression tests can exercise the cleanup path on the real throw sites.
 * 0 = off, 1 = after cudaMalloc3DArray, 2 = after the linear texture,
 * 3 = after the point texture / surface. One-shot: reset when it fires. */
static int g_fail_stage = 0;

void tc_debug_fail_next_create(int stage) { g_fail_stage = stage; }

static void maybe_inject_failure(int stage) {
    if (g_fail_stage == stage) {
        g_fail_stage = 0;
        throw std::runtime_error("tc_create: injected failure (test hook), stage "
                                 + std::to_string(stage));
    }
}

TextureCache *tc_create(tc_kind_t kind, tc_dtype_t dtype, int channels,
                        int dim_a, int dim_b, int dim_c) {
    if (channels != 1 && channels != 4)
        throw std::runtime_error("tc_create: channels must be 1 or 4");
    if (dim_a < 1 || dim_b < 1 || dim_c < 1)
        throw std::invalid_argument("tc_create: every dimension must be >= 1");
    if (channels == 4) {
        if (dtype != TC_DTYPE_FP16)
            throw std::runtime_error("tc_create: channels=4 currently only supports fp16");
        int pack_n = packing_axis_size(kind, dim_a, dim_b, dim_c);
        if (pack_n % 4 != 0)
            throw std::runtime_error("tc_create: packing-axis size must be divisible by 4 for channels=4");
    }

    cudaChannelFormatDesc fmt = tc_channel_desc(dtype, channels);

    /* Compute cudaArray extent. For channels==4 the packing axis is divided
     * by 4 in the array's spatial layout; the 4 packed values per texel
     * occupy the channel axis. */
    cudaExtent extent;
    unsigned int flags = 0;
    switch (kind) {
        case TC_VOL_LAYERED: {
            int Ny = dim_a, Nz = dim_b, Nx = dim_c;
            int Ny_arr = (channels == 4) ? Ny / 4 : Ny;
            extent = make_cudaExtent(Nx, Nz, Ny_arr);
            flags  = cudaArrayLayered;
            break;
        }
        case TC_SINO_LAYERED: {
            int n_angles = dim_a, Ny = dim_b, N_det = dim_c;
            int Ny_arr = (channels == 4) ? Ny / 4 : Ny;
            extent = make_cudaExtent(N_det, Ny_arr, n_angles);
            flags  = cudaArrayLayered;
            break;
        }
        case TC_VOL_3D: {
            int Ny = dim_a, Nz = dim_b, Nx = dim_c;
            int Ny_arr = (channels == 4) ? Ny / 4 : Ny;
            extent = make_cudaExtent(Nx, Nz, Ny_arr);
            flags  = 0;
            break;
        }
        default:
            throw std::runtime_error("tc_create: unknown kind");
    }
    check_extent_limits(kind, extent);

    /* For channels==4 we need surface-write access (the upload kernel uses
     * surf2DLayeredwrite). cudaArraySurfaceLoadStore enables that. */
    if (channels == 4) flags |= cudaArraySurfaceLoadStore;

    TextureCache *tc = (TextureCache *)std::malloc(sizeof(TextureCache));
    if (!tc) throw std::runtime_error("tc_create: malloc failed");
    tc->kind      = kind;
    tc->dtype     = dtype;
    tc->channels  = channels;
    tc->dim_a     = dim_a;
    tc->dim_b     = dim_b;
    tc->dim_c     = dim_c;
    tc->array     = nullptr;
    tc->tex       = 0;
    tc->tex_point = 0;
    tc->surf      = 0;

    /* Every CUDA call below may throw; release whatever was built so far
     * (D5: an OOM in cudaMalloc3DArray or a failing texture/surface create
     * used to leak `tc` and, after the array existed, the whole array). */
    try {
        FASTTOMO_CUDA_CHECK(cudaMalloc3DArray(&tc->array, &fmt, extent, flags));
        maybe_inject_failure(1);

        cudaResourceDesc rd = {};
        rd.resType         = cudaResourceTypeArray;
        rd.res.array.array = tc->array;

        cudaTextureDesc td = {};
        td.addressMode[0]   = cudaAddressModeBorder;
        td.addressMode[1]   = cudaAddressModeBorder;
        td.addressMode[2]   = cudaAddressModeBorder;
        td.filterMode       = cudaFilterModeLinear;
        td.readMode         = cudaReadModeElementType;
        td.normalizedCoords = 0;

        FASTTOMO_CUDA_CHECK(cudaCreateTextureObject(&tc->tex, &rd, &td, nullptr));
        maybe_inject_failure(2);

        /* For TC_VOL_3D single-channel caches, create a point-filter view of
         * the same array. grad_rotmat_kernel uses this for bit-exact corner
         * reads: with cudaFilterModePoint, tex3D returns the exact texel at
         * floor(coord), equivalent to a global memory load. */
        if (kind == TC_VOL_3D && channels == 1) {
            cudaTextureDesc td_point = td;               /* copy all fields */
            td_point.filterMode = cudaFilterModePoint;
            FASTTOMO_CUDA_CHECK(cudaCreateTextureObject(&tc->tex_point, &rd, &td_point, nullptr));
        }

        if (channels == 4) {
            FASTTOMO_CUDA_CHECK(cudaCreateSurfaceObject(&tc->surf, &rd));
        }
        maybe_inject_failure(3);
    } catch (...) {
        (void)tc_release(tc);
        throw;
    }
    return tc;
}

/* ── 4-channel pack-and-upload kernels ──────────────────────────────────────
 * Source data is contiguous in the user's tensor layout. We pack 4 consecutive
 * Y-slices into a single 4-channel texel via surface writes. surf2DLayeredwrite
 * with float2 = 8 bytes = 4 halves matches the 4-channel half texture format
 * (TorchRadon's pattern in src/texture.cu:105-118).
 *
 * write_half4_vol:  src is [Ny, Nz, Nx] fp16 → vol_layered or vol_3d cudaArray
 *                   with extent (Nx, Nz, Ny/4), packing along Ny.
 * write_half4_sino: src is [n_angles, Ny, N_det] fp16 → sino_layered cudaArray
 *                   with extent (N_det, Ny/4, n_angles), packing along Ny.
 */
__global__ static void write_half4_vol(
    const __half *src,                   /* [Ny, Nz, Nx] */
    cudaSurfaceObject_t surf,
    int Nz, int Nx
) {
    int ix      = blockIdx.x * blockDim.x + threadIdx.x;
    int iz      = blockIdx.y * blockDim.y + threadIdx.y;
    int iy_pack = blockIdx.z;            /* layer in packed array */
    if (ix >= Nx || iz >= Nz) return;
    int iy_base = 4 * iy_pack;
    /* read 4 consecutive Y-slices at the same (iz, ix) */
    size_t base = (size_t)iy_base * Nz * Nx + (size_t)iz * Nx + ix;
    size_t stride_y = (size_t)Nz * Nx;
    union { __half h[4]; float2 f; } tmp;   /* 8-byte aligned, no type-pun */
    tmp.h[0] = src[base + 0 * stride_y];
    tmp.h[1] = src[base + 1 * stride_y];
    tmp.h[2] = src[base + 2 * stride_y];
    tmp.h[3] = src[base + 3 * stride_y];
    surf2DLayeredwrite<float2>(tmp.f, surf, ix * (int)sizeof(float2), iz, iy_pack);
}

__global__ static void write_half4_sino(
    const __half *src,                   /* [n_angles, Ny, N_det] */
    cudaSurfaceObject_t surf,
    int Ny, int N_det
) {
    int ix      = blockIdx.x * blockDim.x + threadIdx.x;   /* detector */
    int iy_pack = blockIdx.y * blockDim.y + threadIdx.y;   /* packed Ny index */
    int ia      = blockIdx.z;                               /* angle / layer */
    if (ix >= N_det || iy_pack >= Ny / 4) return;
    int iy_base = 4 * iy_pack;
    size_t base = (size_t)ia * Ny * N_det + (size_t)iy_base * N_det + ix;
    size_t stride_y = (size_t)N_det;
    union { __half h[4]; float2 f; } tmp;   /* 8-byte aligned, no type-pun */
    tmp.h[0] = src[base + 0 * stride_y];
    tmp.h[1] = src[base + 1 * stride_y];
    tmp.h[2] = src[base + 2 * stride_y];
    tmp.h[3] = src[base + 3 * stride_y];
    surf2DLayeredwrite<float2>(tmp.f, surf, ix * (int)sizeof(float2), iy_pack, ia);
}

static size_t tc_element_size(tc_dtype_t dtype) {
    switch (dtype) {
        case TC_DTYPE_FP32: return sizeof(float);
        case TC_DTYPE_FP16: return sizeof(__half);
    }
    throw std::runtime_error("tc: unknown dtype");
}

void tc_upload(TextureCache *tc, const void *src, cudaStream_t stream) {
    if (!tc) throw std::runtime_error("tc_upload: null cache");

    if (tc->channels == 1) {
        size_t elem = tc_element_size(tc->dtype);

        cudaMemcpy3DParms cp = {};
        int row_w   = tc->dim_c;
        int rows    = tc->dim_b;
        cp.srcPtr   = make_cudaPitchedPtr((void *)src,
                                          row_w * elem, row_w, rows);
        cp.dstArray = tc->array;
        cp.extent   = make_cudaExtent(tc->dim_c, tc->dim_b, tc->dim_a);
        cp.kind     = cudaMemcpyDeviceToDevice;
        FASTTOMO_CUDA_CHECK(cudaMemcpy3DAsync(&cp, stream));
        return;
    }

    /* channels == 4: pack 4 Y-slices per texel via the surface kernels. */
    if (tc->dtype != TC_DTYPE_FP16)
        throw std::runtime_error("tc_upload: channels=4 requires fp16");

    if (tc->kind == TC_VOL_LAYERED || tc->kind == TC_VOL_3D) {
        int Ny = tc->dim_a, Nz = tc->dim_b, Nx = tc->dim_c;
        dim3 block(16, 16, 1);
        dim3 grid((Nx + 15) / 16, (Nz + 15) / 16, Ny / 4);
        write_half4_vol<<<grid, block, 0, stream>>>(
            (const __half *)src, tc->surf, Nz, Nx);
    } else if (tc->kind == TC_SINO_LAYERED) {
        int n_angles = tc->dim_a, Ny = tc->dim_b, N_det = tc->dim_c;
        dim3 block(16, 16, 1);
        dim3 grid((N_det + 15) / 16, ((Ny / 4) + 15) / 16, n_angles);
        write_half4_sino<<<grid, block, 0, stream>>>(
            (const __half *)src, tc->surf, Ny, N_det);
    } else {
        throw std::runtime_error("tc_upload: unknown kind");
    }
    FASTTOMO_CUDA_CHECK(cudaGetLastError());
}

cudaError_t tc_release(TextureCache *tc) noexcept {
    if (!tc) return cudaSuccess;
    cudaError_t first = cudaSuccess;
    auto keep = [&first](cudaError_t e) {
        if (first == cudaSuccess && e != cudaSuccess) first = e;
    };
    if (tc->surf)      keep(cudaDestroySurfaceObject(tc->surf));
    if (tc->tex_point) keep(cudaDestroyTextureObject(tc->tex_point));
    if (tc->tex)       keep(cudaDestroyTextureObject(tc->tex));
    if (tc->array)     keep(cudaFreeArray(tc->array));
    std::free(tc);
    return first;
}

void tc_destroy(TextureCache *tc) {
    cudaError_t e = tc_release(tc);
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("tc_destroy: ") + cudaGetErrorString(e));
}

void tc_validate(const TextureCache *tc, tc_kind_t kind, int io_dtype,
                 int dim_a, int dim_b, int dim_c, const char *fn) {
    if (!tc)
        throw std::invalid_argument(std::string(fn) + ": null texture cache");
    if (tc->kind != kind ||
        tc->dim_a != dim_a || tc->dim_b != dim_b || tc->dim_c != dim_c)
        throw std::invalid_argument(std::string(fn) + ": cache shape/kind mismatch");
    int cache_dtype = (tc->dtype == TC_DTYPE_FP32) ? 0 : 1;
    if (cache_dtype != io_dtype)
        throw std::invalid_argument(std::string(fn) + ": cache dtype mismatch with io_dtype");
}

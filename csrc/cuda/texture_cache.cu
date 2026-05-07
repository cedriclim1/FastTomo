#include "texture_cache.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <cstdlib>

/* Same error macro as ops.cu — kept private to this TU. */
#define TC_CUDA_CHECK(call) do {                                              \
    cudaError_t _err = (call);                                                \
    if (_err != cudaSuccess) {                                                \
        throw std::runtime_error(std::string("CUDA error at " __FILE__ ":")   \
            + std::to_string(__LINE__) + ": " + cudaGetErrorString(_err));    \
    }                                                                          \
} while (0)

/* Allocation paths mirror make_vol_texture / make_sino_texture / make_vol_texture_3d
 * in ops.cu, but split into "create empty cudaArray + tex" (now) and
 * "memcpy3D into the existing array" (in tc_upload). */
TextureCache *tc_create(tc_kind_t kind, int dim_a, int dim_b, int dim_c) {
    TextureCache *tc = (TextureCache *)std::malloc(sizeof(TextureCache));
    if (!tc) throw std::runtime_error("tc_create: malloc failed");
    tc->kind  = kind;
    tc->dim_a = dim_a;
    tc->dim_b = dim_b;
    tc->dim_c = dim_c;
    tc->array = nullptr;
    tc->tex   = 0;

    cudaChannelFormatDesc fmt = cudaCreateChannelDesc<float>();

    cudaExtent extent;
    unsigned int flags = 0;
    switch (kind) {
        case TC_VOL_LAYERED:
            /* Ny layers of [Nz × Nx]. extent: (width=Nx, height=Nz, depth=Ny). */
            extent = make_cudaExtent(dim_c, dim_b, dim_a);
            flags  = cudaArrayLayered;
            break;
        case TC_SINO_LAYERED:
            /* n_angles layers of [Ny × N_det]. extent: (N_det, Ny, n_angles). */
            extent = make_cudaExtent(dim_c, dim_b, dim_a);
            flags  = cudaArrayLayered;
            break;
        case TC_VOL_3D:
            /* True 3D. extent: (Nx, Nz, Ny). */
            extent = make_cudaExtent(dim_c, dim_b, dim_a);
            flags  = 0;
            break;
        default:
            std::free(tc);
            throw std::runtime_error("tc_create: unknown kind");
    }

    TC_CUDA_CHECK(cudaMalloc3DArray(&tc->array, &fmt, extent, flags));

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

    TC_CUDA_CHECK(cudaCreateTextureObject(&tc->tex, &rd, &td, nullptr));
    return tc;
}

/* The pitched ptr layout matches what the original make_*_texture helpers in
 * ops.cu used:
 *   vol  (TC_VOL_LAYERED, TC_VOL_3D): src is [Ny, Nz, Nx] → pitch=Nx, height=Nz
 *   sino (TC_SINO_LAYERED):            src is [n_angles, Ny, N_det] → pitch=N_det, height=Ny
 * Both 3D-array kinds copy the same volume layout with cudaMemcpy3D regardless
 * of the cudaArrayLayered flag. */
void tc_upload(TextureCache *tc, const float *src, cudaStream_t stream) {
    if (!tc) throw std::runtime_error("tc_upload: null cache");

    cudaMemcpy3DParms cp = {};
    int row_w   = tc->dim_c;
    int rows    = tc->dim_b;
    cp.srcPtr   = make_cudaPitchedPtr((void *)src,
                                      row_w * sizeof(float), row_w, rows);
    cp.dstArray = tc->array;
    cp.extent   = make_cudaExtent(tc->dim_c, tc->dim_b, tc->dim_a);
    cp.kind     = cudaMemcpyDeviceToDevice;
    TC_CUDA_CHECK(cudaMemcpy3DAsync(&cp, stream));
}

void tc_destroy(TextureCache *tc) {
    if (!tc) return;
    if (tc->tex) {
        /* Errors here would mask user-level cleanup bugs — surface them. */
        cudaError_t e = cudaDestroyTextureObject(tc->tex);
        if (e != cudaSuccess) {
            std::free(tc);
            throw std::runtime_error(std::string("tc_destroy: ")
                + cudaGetErrorString(e));
        }
    }
    if (tc->array) {
        cudaError_t e = cudaFreeArray(tc->array);
        if (e != cudaSuccess) {
            std::free(tc);
            throw std::runtime_error(std::string("tc_destroy: ")
                + cudaGetErrorString(e));
        }
    }
    std::free(tc);
}

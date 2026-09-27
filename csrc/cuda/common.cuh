#pragma once
/* ── fasttomo/csrc/cuda/common.cuh ───────────────────────────────────────────
 * Shared CUDA helpers included by all .cu translation units.
 * Not compiled on its own — included via #include "common.cuh".
 */

#include "texture_cache.h"   /* brings in FASTTOMO_CUDA_CHECK */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdexcept>
#include <string>

/* ── dtype helpers ────────────────────────────────────────────────────────────
 * Kernels are templated on the I/O element type T (float or __half).
 * Texture fetches always use tex*<float> — CUDA hardware promotes 16-bit float
 * texels to 32-bit float on filtered reads. Stores cast the float accumulator
 * back to T via float_to_t<T>. cos/sin tables and rotation matrices stay fp32.
 */
template <typename T> __device__ __forceinline__ T float_to_t(float v);
template <>           __device__ __forceinline__ float  float_to_t<float>(float v)  { return v; }
template <>           __device__ __forceinline__ __half float_to_t<__half>(float v) { return __float2half_rn(v); }

/* ── kernel launch error check ────────────────────────────────────────────────
 * Called immediately after a <<< >>> launch to surface launch-time errors
 * (invalid configuration, too much shared memory, etc.) without a host-device
 * sync. Stream-ordered errors are surfaced later when PyTorch syncs on read.
 * cudaGetLastError (not Peek) so a non-sticky launch error is consumed here
 * and not re-reported against the next, unrelated kernel launch. */
#define CUDA_CHECK_KERNEL() do {                                              \
    cudaError_t _err = cudaGetLastError();                                    \
    if (_err != cudaSuccess) {                                                \
        throw std::runtime_error(std::string("CUDA kernel launch error at "   \
            __FILE__ ":") + std::to_string(__LINE__) + ": "                   \
            + cudaGetErrorString(_err));                                       \
    }                                                                          \
} while (0)

/* ── shared-memory trig loader ────────────────────────────────────────────────
 * Loads cos_vals[0..n-1] and sin_vals[0..n-1] into the caller's shared memory
 * arrays sh_cos and sh_sin using a strided loop over all threads in the block
 * (linear thread index = threadIdx.x + threadIdx.y * blockDim.x). Ends with
 * __syncthreads() so all threads see the fully loaded tables on return.
 * Replaces the near-identical 4-line load loop in the 2D back-projection
 * kernels (scalar fp32/fp16 and packed-h4 variants). */
__device__ static void load_trig_smem(
    float *__restrict__ sh_cos,
    float *__restrict__ sh_sin,
    const float *__restrict__ cos_vals,
    const float *__restrict__ sin_vals,
    int n
) {
    int tid   = threadIdx.x + threadIdx.y * blockDim.x;
    int block = blockDim.x  * blockDim.y;
    for (int i = tid; i < n; i += block) {
        sh_cos[i] = cos_vals[i];
        sh_sin[i] = sin_vals[i];
    }
    __syncthreads();
}

/* ── extended shared memory ───────────────────────────────────────────────────
 * Opt a kernel into the extended dynamic shared memory pool when smem exceeds
 * the default 48 KB cap. Capped at 96 KB (safe on Volta+ targets). */
static void enable_extended_smem_if_needed(const void *kernel, size_t smem) {
    if (smem > 48 * 1024) {
        size_t cap  = 96 * 1024;
        size_t want = smem < cap ? smem : cap;
        FASTTOMO_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)want));
    }
}

/* ── dtype dispatch helper ────────────────────────────────────────────────────
 * Validates io_dtype (0=fp32, 1=fp16) and throws on bad values. */
static void check_io_dtype(int io_dtype, const char *fn) {
    if (io_dtype != 0 && io_dtype != 1)
        throw std::runtime_error(std::string(fn) + ": io_dtype must be 0 (fp32) or 1 (fp16)");
}

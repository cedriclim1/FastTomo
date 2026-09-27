#pragma once
/* ── fasttomo/csrc/cuda/scratch_pool.cuh ─────────────────────────────────────
 * Stream-ordered scratch buffers for the fixed-order (deterministic) two-pass
 * reductions of grad_rotmat_kernel and grad_angle_kernel: pass 1 writes one
 * partial sum per thread block into the scratch, pass 2 sums the partials of
 * each view in a fixed order. Header-only; the pool is a function-local
 * static of an inline function, so every translation unit shares one pool per
 * device.
 *
 * Allocation is cudaMallocFromPoolAsync on a FastTomo-private memory pool
 * (one per device, created on first use) and the matching cudaFreeAsync is
 * enqueued on the same stream, so the buffer lives exactly as long as the
 * kernels that use it and no host synchronisation is added. The private pool
 * keeps up to kScratchPoolKeepBytes reserved across synchronisations: the
 * device's default pool (release threshold 0) hands its memory back at every
 * sync, and re-mapping it measured 80-100 us per call on sm_120 against
 * ~2 us for a retained pool. The pool is never destroyed (process lifetime).
 * Thread-safe: the pybind11 bindings release the GIL.
 */

#include "texture_cache.h"   /* FASTTOMO_CUDA_CHECK */
#include <cuda_runtime.h>
#include <cstdint>
#include <mutex>
#include <vector>

static constexpr uint64_t kScratchPoolKeepBytes = 64ull << 20;   /* 64 MiB */

/* Relaxed stream-capture mode for this thread while in scope. The first use
 * of the pool may happen inside a CUDA-graph capture (torch captures in the
 * global mode, where cudaGetDeviceCount / cudaMemPoolCreate /
 * cudaMemPoolSetAttribute would invalidate the capture); in relaxed mode
 * these one-time, non-stream calls are allowed. The previous mode is
 * restored on exit. */
class RelaxedCaptureMode {
public:
    RelaxedCaptureMode() : mode_(cudaStreamCaptureModeRelaxed) {
        cudaThreadExchangeStreamCaptureMode(&mode_);
    }
    ~RelaxedCaptureMode() { cudaThreadExchangeStreamCaptureMode(&mode_); }
    RelaxedCaptureMode(const RelaxedCaptureMode &) = delete;
    RelaxedCaptureMode &operator=(const RelaxedCaptureMode &) = delete;

private:
    cudaStreamCaptureMode mode_;
};

inline cudaMemPool_t fasttomo_scratch_pool(int device) {
    static std::mutex mu;
    static std::vector<cudaMemPool_t> pools;
    std::lock_guard<std::mutex> lock(mu);
    RelaxedCaptureMode relaxed;   /* one-time setup may run during a capture */
    if (pools.empty()) {
        int n = 0;
        FASTTOMO_CUDA_CHECK(cudaGetDeviceCount(&n));
        pools.assign((size_t)n, nullptr);
    }
    if (device < 0 || (size_t)device >= pools.size())
        throw std::runtime_error("fasttomo scratch pool: invalid device index");
    if (!pools[device]) {
        cudaMemPoolProps props = {};
        props.allocType     = cudaMemAllocationTypePinned;
        props.handleTypes   = cudaMemHandleTypeNone;
        props.location.type = cudaMemLocationTypeDevice;
        props.location.id   = device;
        cudaMemPool_t pool = nullptr;
        FASTTOMO_CUDA_CHECK(cudaMemPoolCreate(&pool, &props));
        uint64_t keep = kScratchPoolKeepBytes;
        FASTTOMO_CUDA_CHECK(cudaMemPoolSetAttribute(
            pool, cudaMemPoolAttrReleaseThreshold, &keep));
        pools[device] = pool;
    }
    return pools[device];
}

/* RAII: `bytes` of device memory on the current device, valid for work
 * enqueued on `stream` until the object goes out of scope (the free is
 * stream-ordered, so kernels launched before that still see the memory). */
class StreamScratch {
public:
    StreamScratch(size_t bytes, cudaStream_t stream) : stream_(stream) {
        int dev = 0;
        FASTTOMO_CUDA_CHECK(cudaGetDevice(&dev));
        FASTTOMO_CUDA_CHECK(cudaMallocFromPoolAsync(
            &ptr_, bytes ? bytes : 1, fasttomo_scratch_pool(dev), stream_));
    }
    ~StreamScratch() {
        if (ptr_) cudaFreeAsync(ptr_, stream_);   /* no throw from a destructor */
    }
    StreamScratch(const StreamScratch &) = delete;
    StreamScratch &operator=(const StreamScratch &) = delete;
    float *floats() const { return static_cast<float *>(ptr_); }

private:
    void        *ptr_ = nullptr;
    cudaStream_t stream_;
};

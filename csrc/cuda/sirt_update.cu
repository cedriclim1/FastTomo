/* ── fasttomo/csrc/cuda/sirt_update.cu ──────────────────────────────────────
 * Fused elementwise steps of one SIRT iteration (kernel audit F7).
 * Kernels: sirt_residual_kernel, sirt_update_kernel.
 * Host wrappers: sirt_residual_cuda, sirt_update_cuda.
 *
 * Each kernel replaces a chain of PyTorch elementwise ops and must give the
 * same bits: every intermediate is rounded to the I/O dtype where PyTorch
 * materialises a tensor, arithmetic is float (PyTorch's opmath for fp16 and
 * fp32), and the explicit __f*_rn intrinsics stop nvcc from contracting a
 * mul/add pair into an FMA where PyTorch rounds in between. The one FMA
 * (graph order, `add_(t, alpha=omega)`) is PyTorch's own `a + alpha * b`,
 * which nvcc contracts to fmaf(alpha, b, a) in ATen's add kernel;
 * tests/test_algorithms_algebraic.py pins every step to the torch chain.
 */

#include "common.cuh"
#include "ops.h"

#include <cstdint>

template <typename T> __device__ __forceinline__ float ld(const T *p, size_t i);
template <> __device__ __forceinline__ float ld<float>(const float *p, size_t i) { return p[i]; }
template <> __device__ __forceinline__ float ld<__half>(const __half *p, size_t i) {
    return __half2float(p[i]);
}

/* Round a float to T and back: the value a T tensor would hold. */
template <typename T> __device__ __forceinline__ float rnd(float x);
template <> __device__ __forceinline__ float rnd<float>(float x) { return x; }
template <> __device__ __forceinline__ float rnd<__half>(float x) {
    return __half2float(__float2half_rn(x));
}

/* clamp(v, min=0) as ATen computes it: NaN propagates, -0.0 is kept. */
__device__ __forceinline__ float clamp_min0(float v) {
    return isnan(v) ? v : (v < 0.0f ? 0.0f : v);
}

/* ── residual ─────────────────────────────────────────────────────────────
 *   raw = sino - fp           (rounded to T)
 *   res = raw * R             (rounded to T)
 * over [n_angles, Ny, N_det]; either output may be null. Grid-stride over a
 * flat index: every tensor is contiguous and the same shape. */
template <typename T>
__global__ static void sirt_residual_kernel(
    const T *__restrict__ sino,
    const T *__restrict__ fp,
    const T *__restrict__ R,
    T *__restrict__       raw_out,
    T *__restrict__       res_out,
    size_t n
) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        float raw = rnd<T>(__fsub_rn(ld(sino, i), ld(fp, i)));
        if (raw_out) raw_out[i] = float_to_t<T>(raw);
        if (res_out) res_out[i] = float_to_t<T>(__fmul_rn(raw, ld(R, i)));
    }
}

/* 16-byte pack of T (4 fp32 or 8 fp16) for vectorised loads and stores. */
template <typename T> struct alignas(16) Pack {
    static constexpr int N = 16 / (int)sizeof(T);
    T v[N];
};

template <typename T> __device__ __forceinline__ Pack<T> ldp(const T *p, size_t i) {
    return *reinterpret_cast<const Pack<T> *>(p + i);
}

/* Vectorised residual: one 16-byte pack per thread; n % Pack<T>::N == 0 and
 * every pointer 16-byte aligned (checked by the host). */
template <typename T>
__global__ static void sirt_residual_vec_kernel(
    const T *__restrict__ sino,
    const T *__restrict__ fp,
    const T *__restrict__ R,
    T *__restrict__       raw_out,
    T *__restrict__       res_out,
    size_t n_packs
) {
    constexpr int N = Pack<T>::N;
    for (size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x; k < n_packs;
         k += (size_t)gridDim.x * blockDim.x) {
        size_t i = k * N;
        Pack<T> s = ldp(sino, i), f = ldp(fp, i), r = ldp(R, i), raw, res;
        #pragma unroll
        for (int e = 0; e < N; e++) {
            float rw = rnd<T>(__fsub_rn(ld(s.v, e), ld(f.v, e)));
            raw.v[e] = float_to_t<T>(rw);
            res.v[e] = float_to_t<T>(__fmul_rn(rw, ld(r.v, e)));
        }
        if (raw_out) *reinterpret_cast<Pack<T> *>(raw_out + i) = raw;
        if (res_out) *reinterpret_cast<Pack<T> *>(res_out + i) = res;
    }
}

/* ── volume update + constraints ──────────────────────────────────────────
 * EAGER (run_sirt's eager loop):
 *   v = vol + (omega * C) * upd          three roundings, as
 *                                        `volume + omega * C_vol * bp`
 * graph order (_sirt_iter_inplace):
 *   t = upd * C;  v = vol + omega * t    `update.mul_(C_vol)`,
 *                                        `vol.add_(update, alpha=omega)`
 * then the constraints in apply_constraints' order: support mask multiply,
 * positivity clamp, soft threshold sign(v) * clamp(|v| - thr, min=0).
 * `mask` is [1 | Ny, Nz, Nx] (mask_full selects the second), or null.
 * vol_out may alias vol_in (each element is read before it is written by
 * the same thread). Grid: x over the Nz*Nx plane, y over Y-slices. */
template <typename T, bool EAGER>
__device__ __forceinline__ float sirt_update_one(
    float v, float u, float c, bool has_mask, float m,
    float omega, int positivity, float soft_thr
) {
    if (EAGER) {
        float oc = rnd<T>(__fmul_rn(c, omega));
        float t  = rnd<T>(__fmul_rn(oc, u));
        v = rnd<T>(__fadd_rn(v, t));
    } else {
        float t = rnd<T>(__fmul_rn(u, c));
        v = rnd<T>(__fmaf_rn(omega, t, v));
    }
    if (has_mask)
        v = rnd<T>(__fmul_rn(v, m));
    if (positivity)
        v = clamp_min0(v);
    if (soft_thr > 0.0f) {
        float a = clamp_min0(rnd<T>(__fsub_rn(fabsf(v), soft_thr)));
        float s = (float)((0.0f < v) - (v < 0.0f));
        v = rnd<T>(__fmul_rn(s, a));
    }
    return v;
}

template <typename T, bool EAGER>
__global__ static void sirt_update_kernel(
    const T *vol_in,
    T       *vol_out,
    const T *__restrict__ upd,
    const T *__restrict__ C,
    const T *__restrict__ mask,
    int mask_full,
    float omega,
    int positivity,
    float soft_thr,
    int Ny, int plane
) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= plane) return;
    for (int y = blockIdx.y; y < Ny; y += gridDim.y) {
        size_t i = (size_t)y * plane + p;
        float m = mask ? ld(mask, mask_full ? i : (size_t)p) : 1.0f;
        vol_out[i] = float_to_t<T>(sirt_update_one<T, EAGER>(
            ld(vol_in, i), ld(upd, i), ld(C, i), mask != nullptr, m,
            omega, positivity, soft_thr));
    }
}

/* Vectorised update: one 16-byte pack per thread; plane % Pack<T>::N == 0
 * and every pointer 16-byte aligned (checked by the host), so a pack never
 * straddles two Y-slices. vol_in / vol_out may alias: the pack is loaded
 * before it is stored, by the same thread. */
template <typename T, bool EAGER>
__global__ static void sirt_update_vec_kernel(
    const T *vol_in,
    T       *vol_out,
    const T *__restrict__ upd,
    const T *__restrict__ C,
    const T *__restrict__ mask,
    int mask_full,
    float omega,
    int positivity,
    float soft_thr,
    int Ny, int plane
) {
    constexpr int N = Pack<T>::N;
    int p = (blockIdx.x * blockDim.x + threadIdx.x) * N;
    if (p >= plane) return;
    for (int y = blockIdx.y; y < Ny; y += gridDim.y) {
        size_t i = (size_t)y * plane + p;
        Pack<T> v = ldp(vol_in, i), u = ldp(upd, i), c = ldp(C, i), m, o;
        if (mask) m = ldp(mask, mask_full ? i : (size_t)p);
        #pragma unroll
        for (int e = 0; e < N; e++)
            o.v[e] = float_to_t<T>(sirt_update_one<T, EAGER>(
                ld(v.v, e), ld(u.v, e), ld(c.v, e), mask != nullptr,
                mask ? ld(m.v, e) : 1.0f, omega, positivity, soft_thr));
        *reinterpret_cast<Pack<T> *>(vol_out + i) = o;
    }
}

/* ── host wrappers ──────────────────────────────────────────────────────── */

/* True for a null pointer too (an absent optional output). */
static bool aligned16(const void *p) {
    return ((uintptr_t)p & 15u) == 0;
}

static unsigned grid_1d(size_t n, int block) {
    size_t g = (n + block - 1) / block;
    return (unsigned)(g < 65535u * 16u ? g : 65535u * 16u);   /* grid-stride beyond */
}

void sirt_residual_cuda(
    int io_dtype,
    const void *sino, const void *fp, const void *R,
    void *raw_out, void *res_out,
    long long n,
    cudaStream_t stream
) {
    check_io_dtype(io_dtype, "sirt_residual_cuda");
    if (n <= 0 || (!raw_out && !res_out)) return;
    const int block = 256;
    const int vec = io_dtype == 0 ? Pack<float>::N : Pack<__half>::N;
    if (n % vec == 0 && aligned16(sino) && aligned16(fp) && aligned16(R) &&
        aligned16(raw_out) && aligned16(res_out)) {
        size_t n_packs = (size_t)n / vec;
        unsigned grid = grid_1d(n_packs, block);
        if (io_dtype == 0)
            sirt_residual_vec_kernel<float><<<grid, block, 0, stream>>>(
                (const float *)sino, (const float *)fp, (const float *)R,
                (float *)raw_out, (float *)res_out, n_packs);
        else
            sirt_residual_vec_kernel<__half><<<grid, block, 0, stream>>>(
                (const __half *)sino, (const __half *)fp, (const __half *)R,
                (__half *)raw_out, (__half *)res_out, n_packs);
        CUDA_CHECK_KERNEL();
        return;
    }
    unsigned grid = grid_1d((size_t)n, block);
    if (io_dtype == 0)
        sirt_residual_kernel<float><<<grid, block, 0, stream>>>(
            (const float *)sino, (const float *)fp, (const float *)R,
            (float *)raw_out, (float *)res_out, (size_t)n);
    else
        sirt_residual_kernel<__half><<<grid, block, 0, stream>>>(
            (const __half *)sino, (const __half *)fp, (const __half *)R,
            (__half *)raw_out, (__half *)res_out, (size_t)n);
    CUDA_CHECK_KERNEL();
}

void sirt_update_cuda(
    int io_dtype, int eager_order,
    const void *vol_in, void *vol_out,
    const void *upd, const void *C,
    const void *mask, int mask_full,
    float omega, int positivity, float soft_thr,
    int Ny, int Nz, int Nx,
    cudaStream_t stream
) {
    check_io_dtype(io_dtype, "sirt_update_cuda");
    if (Ny <= 0 || Nz <= 0 || Nx <= 0) return;
    long long plane_ll = (long long)Nz * Nx;
    if (plane_ll > 0x7fffffffLL)
        throw std::invalid_argument("sirt_update_cuda: Nz * Nx too large");
    int plane = (int)plane_ll;
    const int block = 256;
    const int vec = io_dtype == 0 ? Pack<float>::N : Pack<__half>::N;
    bool use_vec = plane % vec == 0 && aligned16(vol_in) && aligned16(vol_out) &&
                   aligned16(upd) && aligned16(C) && aligned16(mask);
    int per_slice = use_vec ? plane / vec : plane;
    dim3 grid((unsigned)((per_slice + block - 1) / block), (unsigned)(Ny < 65535 ? Ny : 65535));

#define SIRT_UPDATE(T, E) do {                                                   \
    if (use_vec)                                                                 \
        sirt_update_vec_kernel<T, E><<<grid, block, 0, stream>>>(                \
            (const T *)vol_in, (T *)vol_out, (const T *)upd, (const T *)C,       \
            (const T *)mask, mask_full, omega, positivity, soft_thr, Ny, plane); \
    else                                                                         \
        sirt_update_kernel<T, E><<<grid, block, 0, stream>>>(                    \
            (const T *)vol_in, (T *)vol_out, (const T *)upd, (const T *)C,       \
            (const T *)mask, mask_full, omega, positivity, soft_thr, Ny, plane); \
} while (0)
    if (io_dtype == 0) {
        if (eager_order) SIRT_UPDATE(float, true); else SIRT_UPDATE(float, false);
    } else {
        if (eager_order) SIRT_UPDATE(__half, true); else SIRT_UPDATE(__half, false);
    }
#undef SIRT_UPDATE
    CUDA_CHECK_KERNEL();
}

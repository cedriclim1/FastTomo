/* ── fasttomo/csrc/cuda/fista_update.cu ────────────────────────────────────
 * Fused elementwise steps of one FISTA iteration (solver audit P4).
 * Kernels: fista_update_kernel<T, MODE>, fista_update_packed_h4_kernel<MODE>.
 * Host wrapper: fista_update_cuda.
 *
 * run_fista's iteration, after grad = A^T (A y - b), was 4-6 PyTorch
 * elementwise kernels plus a volume-texture upload for the next forward
 * projection of the new y:
 *
 *   x  = constraints(y - (1/L) * grad)      `y - inv_L * grad`, apply()
 *   y' = x + beta * (x - x_prev)            three kernels
 *
 * MODE_FULL does all of it in one pass (reads y, grad, x_prev and the
 * support mask; writes x into x_out and y' into y_out and, with a volume
 * TextureCache, into the texture, so the next cached forward projection
 * skips its upload). run_fista_tv, whose proximal step is the TV prox,
 * uses the two halves: MODE_STEP (u = y - (1/L) * grad, the prox input)
 * and MODE_MOMENTUM (y' = x + beta * (x - x_prev), x the prox output).
 *
 * Bit-identical to the PyTorch chain: every intermediate is rounded to the
 * I/O dtype where PyTorch materialises a tensor, arithmetic is float
 * (PyTorch's opmath for fp16 and fp32), the Python-float scalars inv_L and
 * beta enter as floats (ATen casts a scalar operand to opmath), and the
 * __f*_rn intrinsics stop nvcc from contracting a mul/add pair into an FMA
 * where PyTorch rounds in between. Constraints in apply_constraints' order:
 * support multiply, then clamp(min=0) with NaN kept (no soft threshold:
 * the FISTA solvers have none). Pinned by
 * tests/test_algorithms_fista.py::test_fista_update_kernel_bit_identical.
 *
 * Aliasing: y_out may be y_in and x_out may be x_prev (each element is read
 * before it is written, by the same thread); x_out must not be y_in.
 */

#include "common.cuh"
#include "ops.h"

namespace {

enum { MODE_FULL = 0, MODE_STEP = 1, MODE_MOMENTUM = 2 };
enum { SURF_NONE = 0, SURF_LAYERED = 1, SURF_3D = 2 };

template <typename T> __device__ __forceinline__ float ldf(const T *p, size_t i);
template <> __device__ __forceinline__ float ldf<float>(const float *p, size_t i) { return p[i]; }
template <> __device__ __forceinline__ float ldf<__half>(const __half *p, size_t i) {
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

template <typename T> __device__ __forceinline__ void surf_put(
    cudaSurfaceObject_t s, int kind, float v, int x, int y, int layer);
template <> __device__ __forceinline__ void surf_put<float>(
    cudaSurfaceObject_t s, int kind, float v, int x, int y, int layer) {
    if (kind == SURF_LAYERED)
        surf2DLayeredwrite<float>(v, s, x * (int)sizeof(float), y, layer);
    else
        surf3Dwrite<float>(v, s, x * (int)sizeof(float), y, layer);
}
template <> __device__ __forceinline__ void surf_put<__half>(
    cudaSurfaceObject_t s, int kind, float v, int x, int y, int layer) {
    unsigned short u = __half_as_ushort(__float2half_rn(v));
    if (kind == SURF_LAYERED)
        surf2DLayeredwrite<unsigned short>(u, s, x * (int)sizeof(__half), y, layer);
    else
        surf3Dwrite<unsigned short>(u, s, x * (int)sizeof(__half), y, layer);
}

/* One element. MODE_FULL / MODE_STEP: a = y, b = grad; MODE_MOMENTUM:
 * a = x (the prox output), b unused. Returns the value for x_out
 * (FULL: x, STEP: u) in *xo and for y_out (FULL, MOMENTUM) in *yo. */
template <typename T, int MODE>
__device__ __forceinline__ void fista_one(
    float a, float b, float xp, bool has_mask, float m,
    float step, float beta, int positivity, float *xo, float *yo
) {
    float x;
    if (MODE == MODE_MOMENTUM) {
        x = a;
    } else {
        float t = rnd<T>(__fmul_rn(b, step));      /* inv_L * grad */
        x = rnd<T>(__fsub_rn(a, t));               /* y - t */
        if (MODE == MODE_FULL) {
            if (has_mask)
                x = rnd<T>(__fmul_rn(x, m));
            if (positivity)
                x = clamp_min0(x);
        }
    }
    *xo = x;
    if (MODE != MODE_STEP) {
        float d = rnd<T>(__fsub_rn(x, xp));        /* x - x_prev */
        float e = rnd<T>(__fmul_rn(d, beta));      /* beta * d */
        *yo = rnd<T>(__fadd_rn(x, e));             /* x + e */
    }
}

/* One voxel per thread: x over the Nz*Nx plane (x-contiguous warps, also
 * for the surface stores), y over Y-slices (grid-stride). */
template <typename T, int MODE>
__global__ void fista_update_kernel(
    const T *a_in,
    const T *__restrict__ grad,
    const T *xp_in,
    const T *__restrict__ mask,
    int mask_full,
    T *x_out,
    T *y_out,
    float step, float beta, int positivity,
    int Ny, int plane, int Nx,
    cudaSurfaceObject_t surf, int surf_kind
) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= plane) return;
    int x = p % Nx, z = p / Nx;
    for (int y = blockIdx.y; y < Ny; y += gridDim.y) {
        size_t i = (size_t)y * plane + p;
        float m = mask ? ldf(mask, mask_full ? i : (size_t)p) : 1.0f;
        float xo, yo = 0.0f;
        fista_one<T, MODE>(
            ldf(a_in, i), MODE == MODE_MOMENTUM ? 0.0f : ldf(grad, i),
            MODE == MODE_STEP ? 0.0f : ldf(xp_in, i),
            mask != nullptr, m, step, beta, positivity, &xo, &yo);
        if (MODE != MODE_MOMENTUM) x_out[i] = float_to_t<T>(xo);
        if (MODE != MODE_STEP) {
            y_out[i] = float_to_t<T>(yo);
            if (surf_kind != SURF_NONE)
                surf_put<T>(surf, surf_kind, yo, x, z, y);   /* texel (x, z, y) */
        }
    }
}

/* 4 fp16 values of consecutive Y-slices as one 4-channel texel, the layout
 * of write_half4_vol_kernel (texture_cache.cu). */
union Half4 {
    __half h[4];
    float2 f;
};

/* fp16, y' also into a 4-channel packed layered volume texture (Ny % 4 ==
 * 0): one thread per (plane position, Y-slice quad). MODE_FULL and
 * MODE_MOMENTUM only. */
template <int MODE>
__global__ void fista_update_packed_h4_kernel(
    const __half *a_in,
    const __half *__restrict__ grad,
    const __half *xp_in,
    const __half *__restrict__ mask,
    int mask_full,
    __half *x_out,
    __half *y_out,
    float step, float beta, int positivity,
    int Ny, int plane, int Nx,
    cudaSurfaceObject_t surf
) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= plane) return;
    int x = p % Nx, z = p / Nx;
    for (int y4 = blockIdx.y; y4 < Ny / 4; y4 += gridDim.y) {
        Half4 tmp;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            size_t i = (size_t)(4 * y4 + k) * plane + p;
            float m = mask ? ldf(mask, mask_full ? i : (size_t)p) : 1.0f;
            float xo, yo = 0.0f;
            fista_one<__half, MODE>(
                ldf(a_in, i), MODE == MODE_MOMENTUM ? 0.0f : ldf(grad, i), ldf(xp_in, i),
                mask != nullptr, m, step, beta, positivity, &xo, &yo);
            if (MODE != MODE_MOMENTUM) x_out[i] = __float2half_rn(xo);
            tmp.h[k] = __float2half_rn(yo);
            y_out[i] = tmp.h[k];
        }
        surf2DLayeredwrite<float2>(tmp.f, surf, x * (int)sizeof(float2), z, y4);
    }
}

template <typename T, int MODE>
void launch(const void *a_in, const void *grad, const void *xp_in, const void *mask,
            int mask_full, void *x_out, void *y_out, float step, float beta,
            int positivity, int Ny, int plane, int Nx, cudaSurfaceObject_t surf,
            int surf_kind, cudaStream_t stream) {
    const int block = 256;
    dim3 grid((unsigned)((plane + block - 1) / block), (unsigned)(Ny < 65535 ? Ny : 65535));
    fista_update_kernel<T, MODE><<<grid, block, 0, stream>>>(
        (const T *)a_in, (const T *)grad, (const T *)xp_in, (const T *)mask, mask_full,
        (T *)x_out, (T *)y_out, step, beta, positivity, Ny, plane, Nx, surf, surf_kind);
    CUDA_CHECK_KERNEL();
}

template <int MODE>
void launch_packed(const void *a_in, const void *grad, const void *xp_in, const void *mask,
                   int mask_full, void *x_out, void *y_out, float step, float beta,
                   int positivity, int Ny, int plane, int Nx, cudaSurfaceObject_t surf,
                   cudaStream_t stream) {
    const int block = 256;
    dim3 grid((unsigned)((plane + block - 1) / block),
              (unsigned)(Ny / 4 < 65535 ? Ny / 4 : 65535));
    fista_update_packed_h4_kernel<MODE><<<grid, block, 0, stream>>>(
        (const __half *)a_in, (const __half *)grad, (const __half *)xp_in,
        (const __half *)mask, mask_full, (__half *)x_out, (__half *)y_out, step, beta,
        positivity, Ny, plane, Nx, surf);
    CUDA_CHECK_KERNEL();
}

}  // namespace

int fista_update_cuda(
    int io_dtype, int mode,
    const void *a_in, const void *grad, const void *xp_in,
    const void *mask, int mask_full,
    void *x_out, void *y_out,
    float step, float beta, int positivity,
    int Ny, int Nz, int Nx,
    TextureCache *vol_tc,
    cudaStream_t stream
) {
    check_io_dtype(io_dtype, "fista_update_cuda");
    if (mode != MODE_FULL && mode != MODE_STEP && mode != MODE_MOMENTUM)
        throw std::invalid_argument("fista_update_cuda: mode must be 0, 1 or 2");
    if (Ny <= 0 || Nz <= 0 || Nx <= 0) return 0;
    long long plane_ll = (long long)Nz * Nx;
    if (plane_ll > 0x7fffffffLL)
        throw std::invalid_argument("fista_update_cuda: Nz * Nx too large");
    int plane = (int)plane_ll;
    if (!a_in || (mode != MODE_MOMENTUM && !grad) || (mode != MODE_STEP && !xp_in) ||
        (mode != MODE_MOMENTUM && !x_out) || (mode != MODE_STEP && !y_out))
        throw std::invalid_argument("fista_update_cuda: missing input or output");

    /* The texture receives y' (MODE_FULL / MODE_MOMENTUM) when it is a
     * volume cache with a surface in a layout this file writes: 1 channel,
     * or fp16 4-channel packed layered. Returns 1 if it was written. */
    cudaSurfaceObject_t surf = 0;
    int surf_kind = SURF_NONE, packed = 0;
    if (vol_tc && mode != MODE_STEP) {
        if (vol_tc->kind == TC_SINO_LAYERED)
            throw std::invalid_argument("fista_update_cuda: vol_tc must be a volume cache");
        tc_validate(vol_tc, vol_tc->kind, io_dtype, Ny, Nz, Nx, "fista_update_cuda");
        if (vol_tc->surf && vol_tc->channels == 1) {
            surf = vol_tc->surf;
            surf_kind = vol_tc->kind == TC_VOL_3D ? SURF_3D : SURF_LAYERED;
        } else if (vol_tc->surf && vol_tc->channels == 4 && io_dtype == 1 &&
                   vol_tc->kind == TC_VOL_LAYERED && Ny % 4 == 0) {
            surf = vol_tc->surf;
            packed = 1;
        }
    }

    if (packed) {
        if (mode == MODE_FULL)
            launch_packed<MODE_FULL>(a_in, grad, xp_in, mask, mask_full, x_out, y_out,
                                     step, beta, positivity, Ny, plane, Nx, surf, stream);
        else
            launch_packed<MODE_MOMENTUM>(a_in, grad, xp_in, mask, mask_full, x_out, y_out,
                                         step, beta, positivity, Ny, plane, Nx, surf, stream);
        return 1;
    }

#define FISTA_LAUNCH(T, M) launch<T, M>(a_in, grad, xp_in, mask, mask_full, x_out, y_out, \
                                        step, beta, positivity, Ny, plane, Nx, surf,     \
                                        surf_kind, stream)
    if (io_dtype == 0) {
        if (mode == MODE_FULL) FISTA_LAUNCH(float, MODE_FULL);
        else if (mode == MODE_STEP) FISTA_LAUNCH(float, MODE_STEP);
        else FISTA_LAUNCH(float, MODE_MOMENTUM);
    } else {
        if (mode == MODE_FULL) FISTA_LAUNCH(__half, MODE_FULL);
        else if (mode == MODE_STEP) FISTA_LAUNCH(__half, MODE_STEP);
        else FISTA_LAUNCH(__half, MODE_MOMENTUM);
    }
#undef FISTA_LAUNCH
    return surf_kind != SURF_NONE ? 1 : 0;
}

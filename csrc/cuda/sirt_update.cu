/* ── fasttomo/csrc/cuda/sirt_update.cu ──────────────────────────────────────
 * Fused elementwise steps of one SIRT iteration (kernel audit F7).
 * Kernels: sirt_residual(_vec)_kernel, sirt_residual_tex(_packed_h4)_kernel,
 *          sirt_update(_vec)_kernel, sirt_update_packed_h4_kernel.
 * Host wrappers: sirt_residual_cuda, sirt_residual_tex_cuda, sirt_update_cuda.
 *
 * The _tex variants and sirt_update_cuda with a vol_tc also write their
 * result straight into a projector's TextureCache through its surface
 * (kernel audit F6): the residual only ever feeds the back-projection, so it
 * goes to the sinogram texture instead of global memory, and the updated
 * volume goes to both global memory and the volume texture, so the next
 * forward projection needs no tc_upload (cudaMemcpy3D D2D copy). The
 * texture ends up holding exactly what tc_upload would have copied (fp16
 * 4-channel caches: the same 4-Y-slice packing as write_half4_*_kernel),
 * so the projections read identical texels.
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

/* ── surface writes ─────────────────────────────────────────────────────── */

/* Where a kernel writes its result besides global memory: nowhere
 * (kind SURF_NONE), a layered 2-D array (x, y, layer) or a 3-D array
 * (x, y, z). x is in elements; the byte offset is taken here. */
enum { SURF_NONE = 0, SURF_LAYERED = 1, SURF_3D = 2 };
struct SurfTarget {
    cudaSurfaceObject_t surf;
    int kind;
};

template <typename T> __device__ __forceinline__ void surf_put(
    const SurfTarget &t, float v, int x, int y, int layer);
template <> __device__ __forceinline__ void surf_put<float>(
    const SurfTarget &t, float v, int x, int y, int layer) {
    if (t.kind == SURF_LAYERED)
        surf2DLayeredwrite<float>(v, t.surf, x * (int)sizeof(float), y, layer);
    else
        surf3Dwrite<float>(v, t.surf, x * (int)sizeof(float), y, layer);
}
template <> __device__ __forceinline__ void surf_put<__half>(
    const SurfTarget &t, float v, int x, int y, int layer) {
    unsigned short u = __half_as_ushort(__float2half_rn(v));
    if (t.kind == SURF_LAYERED)
        surf2DLayeredwrite<unsigned short>(u, t.surf, x * (int)sizeof(__half), y, layer);
    else
        surf3Dwrite<unsigned short>(u, t.surf, x * (int)sizeof(__half), y, layer);
}

template <typename T> __device__ __forceinline__ float surf_get(
    const SurfTarget &t, int x, int y, int layer);
template <> __device__ __forceinline__ float surf_get<float>(
    const SurfTarget &t, int x, int y, int layer) {
    float v;
    if (t.kind == SURF_LAYERED)
        surf2DLayeredread<float>(&v, t.surf, x * (int)sizeof(float), y, layer);
    else
        surf3Dread<float>(&v, t.surf, x * (int)sizeof(float), y, layer);
    return v;
}
template <> __device__ __forceinline__ float surf_get<__half>(
    const SurfTarget &t, int x, int y, int layer) {
    unsigned short u;
    if (t.kind == SURF_LAYERED)
        surf2DLayeredread<unsigned short>(&u, t.surf, x * (int)sizeof(__half), y, layer);
    else
        surf3Dread<unsigned short>(&u, t.surf, x * (int)sizeof(__half), y, layer);
    return __half2float(__ushort_as_half(u));
}

/* 4 fp16 values of consecutive Y-slices as one 4-channel texel, the layout
 * of write_half4_vol_kernel / write_half4_sino_kernel (texture_cache.cu). */
union Half4 {
    __half h[4];
    float2 f;
};

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

/* Residual into the sinogram texture (layered: x = detector, y = Y-slice,
 * layer = angle) instead of global memory; raw_out optional (the loss).
 * One detector pixel per thread, so a warp's surface stores are x-contiguous
 * (a pack per thread measured slower: its stores stride 16 bytes). Grid
 * (N_det / block, Ny, n_angles). */
template <typename T>
__global__ static void sirt_residual_tex_kernel(
    const T *__restrict__ sino,
    const T *__restrict__ fp,
    const T *__restrict__ R,
    T *__restrict__       raw_out,
    cudaSurfaceObject_t   surf,
    int Ny, int N_det
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= N_det) return;
    int iy = blockIdx.y, ia = blockIdx.z;
    size_t i = ((size_t)ia * Ny + iy) * N_det + id;
    float rw = rnd<T>(__fsub_rn(ld(sino, i), ld(fp, i)));
    if (raw_out) raw_out[i] = float_to_t<T>(rw);
    surf_put<T>(SurfTarget{surf, SURF_LAYERED}, __fmul_rn(rw, ld(R, i)), id, iy, ia);
}

/* fp16 residual into a 4-channel packed sinogram texture (Ny % 4 == 0):
 * one thread per (detector, Y-slice quad, angle). Grid (N_det / block,
 * Ny / 4, n_angles). */
__global__ static void sirt_residual_tex_packed_h4_kernel(
    const __half *__restrict__ sino,
    const __half *__restrict__ fp,
    const __half *__restrict__ R,
    __half *__restrict__       raw_out,
    cudaSurfaceObject_t        surf,
    int Ny, int N_det
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= N_det) return;
    int iy4 = blockIdx.y, ia = blockIdx.z;
    Half4 tmp;
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        size_t i = ((size_t)ia * Ny + 4 * iy4 + k) * N_det + id;
        float rw = rnd<__half>(__fsub_rn(ld(sino, i), ld(fp, i)));
        if (raw_out) raw_out[i] = __float2half_rn(rw);
        tmp.h[k] = __float2half_rn(__fmul_rn(rw, ld(R, i)));
    }
    surf2DLayeredwrite<float2>(tmp.f, surf, id * (int)sizeof(float2), iy4, ia);
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
    int Ny, int plane, int Nx,
    SurfTarget st
) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= plane) return;
    int x = p % Nx, z = p / Nx;
    for (int y = blockIdx.y; y < Ny; y += gridDim.y) {
        size_t i = (size_t)y * plane + p;
        float m = mask ? ld(mask, mask_full ? i : (size_t)p) : 1.0f;
        /* vol_in null: the volume lives in the texture (read it there) */
        float v0 = vol_in ? ld(vol_in, i) : surf_get<T>(st, x, z, y);
        float v = sirt_update_one<T, EAGER>(
            v0, ld(upd, i), ld(C, i), mask != nullptr, m,
            omega, positivity, soft_thr);
        if (vol_out) vol_out[i] = float_to_t<T>(v);
        if (st.kind != SURF_NONE)
            surf_put<T>(st, v, x, z, y);   /* texel (x, z, y) */
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

/* fp16 update into a 4-channel packed layered volume texture (Ny % 4 == 0):
 * one thread per (Nz*Nx plane position, Y-slice quad). Grid (plane / block,
 * Ny / 4). */
template <bool EAGER>
__global__ static void sirt_update_packed_h4_kernel(
    const __half *vol_in,
    __half       *vol_out,
    const __half *__restrict__ upd,
    const __half *__restrict__ C,
    const __half *__restrict__ mask,
    int mask_full,
    float omega,
    int positivity,
    float soft_thr,
    int Ny, int plane, int Nx,
    cudaSurfaceObject_t surf
) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= plane) return;
    int x = p % Nx, z = p / Nx;
    for (int y4 = blockIdx.y; y4 < Ny / 4; y4 += gridDim.y) {
        Half4 tmp, cur;
        if (!vol_in)    /* the volume lives in the texture */
            surf2DLayeredread<float2>(&cur.f, surf, x * (int)sizeof(float2), z, y4);
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            size_t i = (size_t)(4 * y4 + k) * plane + p;
            float m = mask ? ld(mask, mask_full ? i : (size_t)p) : 1.0f;
            float v = sirt_update_one<__half, EAGER>(
                vol_in ? ld(vol_in, i) : __half2float(cur.h[k]), ld(upd, i), ld(C, i),
                mask != nullptr, m, omega, positivity, soft_thr);
            tmp.h[k] = __float2half_rn(v);
            if (vol_out) vol_out[i] = tmp.h[k];
        }
        surf2DLayeredwrite<float2>(tmp.f, surf, x * (int)sizeof(float2), z, y4);
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

/* Throw unless `tc` can take a direct write (validated shape and dtype, a
 * surface object). */
static void check_surface(const TextureCache *tc, const char *fn) {
    if (!tc->surf)
        throw std::invalid_argument(std::string(fn) +
                                    ": texture cache has no surface (tc_create surface=True)");
}

void sirt_residual_tex_cuda(
    int io_dtype,
    const void *sino, const void *fp, const void *R,
    void *raw_out,
    TextureCache *res_tc,
    int n_angles, int Ny, int N_det,
    cudaStream_t stream
) {
    check_io_dtype(io_dtype, "sirt_residual_tex_cuda");
    tc_validate(res_tc, TC_SINO_LAYERED, io_dtype, n_angles, Ny, N_det,
                "sirt_residual_tex_cuda");
    check_surface(res_tc, "sirt_residual_tex_cuda");
    if (Ny > 65535 || n_angles > 65535)
        throw std::invalid_argument("sirt_residual_tex_cuda: Ny or n_angles above 65535");
    const int block = 128;
    if (res_tc->channels == 4) {
        dim3 grid((unsigned)((N_det + block - 1) / block), (unsigned)(Ny / 4), (unsigned)n_angles);
        sirt_residual_tex_packed_h4_kernel<<<grid, block, 0, stream>>>(
            (const __half *)sino, (const __half *)fp, (const __half *)R,
            (__half *)raw_out, res_tc->surf, Ny, N_det);
        CUDA_CHECK_KERNEL();
        return;
    }
    dim3 grid((unsigned)((N_det + block - 1) / block), (unsigned)Ny, (unsigned)n_angles);
    if (io_dtype == 0)
        sirt_residual_tex_kernel<float><<<grid, block, 0, stream>>>(
            (const float *)sino, (const float *)fp, (const float *)R,
            (float *)raw_out, res_tc->surf, Ny, N_det);
    else
        sirt_residual_tex_kernel<__half><<<grid, block, 0, stream>>>(
            (const __half *)sino, (const __half *)fp, (const __half *)R,
            (__half *)raw_out, res_tc->surf, Ny, N_det);
    CUDA_CHECK_KERNEL();
}

void sirt_update_cuda(
    int io_dtype, int eager_order,
    const void *vol_in, void *vol_out,
    const void *upd, const void *C,
    const void *mask, int mask_full,
    float omega, int positivity, float soft_thr,
    int Ny, int Nz, int Nx,
    TextureCache *vol_tc,
    cudaStream_t stream
) {
    check_io_dtype(io_dtype, "sirt_update_cuda");
    if (Ny <= 0 || Nz <= 0 || Nx <= 0) return;
    long long plane_ll = (long long)Nz * Nx;
    if (plane_ll > 0x7fffffffLL)
        throw std::invalid_argument("sirt_update_cuda: Nz * Nx too large");
    int plane = (int)plane_ll;
    const int block = 256;

    if ((!vol_in || !vol_out) && !vol_tc)
        throw std::invalid_argument("sirt_update_cuda: a null vol_in / vol_out needs vol_tc");
    SurfTarget st{0, SURF_NONE};
    if (vol_tc) {
        if (vol_tc->kind == TC_SINO_LAYERED)
            throw std::invalid_argument("sirt_update_cuda: vol_tc must be a volume cache");
        tc_validate(vol_tc, vol_tc->kind, io_dtype, Ny, Nz, Nx, "sirt_update_cuda");
        check_surface(vol_tc, "sirt_update_cuda");
        st = SurfTarget{vol_tc->surf, vol_tc->kind == TC_VOL_3D ? SURF_3D : SURF_LAYERED};
        if (vol_tc->channels == 4) {        /* fp16 packed layered (Ny % 4 == 0) */
            dim3 grid((unsigned)((plane + block - 1) / block),
                      (unsigned)(Ny / 4 < 65535 ? Ny / 4 : 65535));
            if (eager_order)
                sirt_update_packed_h4_kernel<true><<<grid, block, 0, stream>>>(
                    (const __half *)vol_in, (__half *)vol_out, (const __half *)upd,
                    (const __half *)C, (const __half *)mask, mask_full, omega,
                    positivity, soft_thr, Ny, plane, Nx, vol_tc->surf);
            else
                sirt_update_packed_h4_kernel<false><<<grid, block, 0, stream>>>(
                    (const __half *)vol_in, (__half *)vol_out, (const __half *)upd,
                    (const __half *)C, (const __half *)mask, mask_full, omega,
                    positivity, soft_thr, Ny, plane, Nx, vol_tc->surf);
            CUDA_CHECK_KERNEL();
            return;
        }
    }

    const int vec = io_dtype == 0 ? Pack<float>::N : Pack<__half>::N;
    /* with a surface: one voxel per thread, so the surface accesses of a
     * warp are x-contiguous */
    bool use_vec = st.kind == SURF_NONE && plane % vec == 0 && aligned16(vol_in) &&
                   aligned16(vol_out) && aligned16(upd) && aligned16(C) && aligned16(mask);
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
            (const T *)mask, mask_full, omega, positivity, soft_thr, Ny, plane,  \
            Nx, st);                                                             \
} while (0)
    if (io_dtype == 0) {
        if (eager_order) SIRT_UPDATE(float, true); else SIRT_UPDATE(float, false);
    } else {
        if (eager_order) SIRT_UPDATE(__half, true); else SIRT_UPDATE(__half, false);
    }
#undef SIRT_UPDATE
    CUDA_CHECK_KERNEL();
}

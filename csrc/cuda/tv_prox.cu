/* ── fasttomo/csrc/cuda/tv_prox.cu ─────────────────────────────────────────
 * The 3-D isotropic TV proximal operator of run_fista_tv on CUDA (solver
 * audit P1): the dual solvers of fasttomo.algorithms.fista._tv_prox_3d,
 * Fast Gradient Projection (Beck & Teboulle 2009b) or Chambolle (2004),
 *
 *   argmin_v  1/2 ||v - z||^2 + lam TV(v) + i_C(v),
 *
 * as two kernels per inner iteration instead of ~44 PyTorch ops and ~35
 * allocations:
 *
 *   tv_prox_primal_kernel:  v = P_C(z - div(r))          (reads z, r)
 *   tv_prox_dual_kernel:    q = r + step * grad(v)
 *                           p' = q / max(1, |q| / lam)   (per voxel, isotropic)
 *                           FGP: r = p' + beta (p' - p), p = p'
 *                           Chambolle: r = p = p'        (reads v, r, p)
 *
 * and a last primal step from p. The duals live in a padded layout: each
 * component is a full [Ny, Nz, Nx] array whose last plane along its own
 * axis is never read (PyTorch keeps them one shorter and zero-pads for the
 * norm), so no F.pad, no slices, no per-iteration allocation.
 *
 * Bit-identical to the PyTorch reference in fp32 and fp16: every value is
 * rounded to the I/O dtype where PyTorch materialises a tensor, in the same
 * order, with float arithmetic and __f*_rn intrinsics (no FMA contraction):
 *   div:   0 (+ p_y[i-1]) (- p_y[i]) (+ p_z[j-1]) (- p_z[j]) (+ p_x[k-1])
 *          (- p_x[k]), each term only where _tv_div's slice covers the voxel
 *   norm:  sqrt((qy*qy + qz*qz) + qx*qx), q zero on the padded plane
 *   denom: max(norm * inv_lam, 1) with NaN kept; inv_lam = 1/lam rounded to
 *          fp32 (ATen divides by a Python scalar as a product with its fp32
 *          reciprocal); p' = q / denom (true division)
 *   step, beta: Python floats, cast to float as ATen does.
 * Pinned by tests/test_algorithms_fista.py::test_tv_prox_cuda_bit_identical.
 */

#include "common.cuh"
#include "ops.h"

namespace {

template <typename T> __device__ __forceinline__ float ldv(const T *p, size_t i);
template <> __device__ __forceinline__ float ldv<float>(const float *p, size_t i) { return p[i]; }
template <> __device__ __forceinline__ float ldv<__half>(const __half *p, size_t i) {
    return __half2float(p[i]);
}

template <typename T> __device__ __forceinline__ float rnd(float x);
template <> __device__ __forceinline__ float rnd<float>(float x) { return x; }
template <> __device__ __forceinline__ float rnd<__half>(float x) {
    return __half2float(__float2half_rn(x));
}

/* clamp(v, min=c) as ATen computes it: NaN propagates. */
__device__ __forceinline__ float clamp_min(float v, float c) {
    return isnan(v) ? v : (v < c ? c : v);
}

/* v = P_C(z - div(r)); P_C = clamp(min=0) (positivity) then * mask. */
template <typename T>
__global__ void tv_prox_primal_kernel(
    const T *__restrict__ z,
    const T *__restrict__ ry,
    const T *__restrict__ rz,
    const T *__restrict__ rx,
    const T *__restrict__ mask,
    int mask_full,
    int positivity,
    T *__restrict__ v,
    int Ny, int Nz, int Nx
) {
    const int plane = Nz * Nx;
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= plane) return;
    const int k = p % Nx, j = p / Nx;
    for (int i = blockIdx.y; i < Ny; i += gridDim.y) {
        size_t idx = (size_t)i * plane + p;
        float d = 0.0f;
        if (i >= 1)      d = rnd<T>(__fadd_rn(d, ldv(ry, idx - plane)));
        if (i <= Ny - 2) d = rnd<T>(__fsub_rn(d, ldv(ry, idx)));
        if (j >= 1)      d = rnd<T>(__fadd_rn(d, ldv(rz, idx - Nx)));
        if (j <= Nz - 2) d = rnd<T>(__fsub_rn(d, ldv(rz, idx)));
        if (k >= 1)      d = rnd<T>(__fadd_rn(d, ldv(rx, idx - 1)));
        if (k <= Nx - 2) d = rnd<T>(__fsub_rn(d, ldv(rx, idx)));
        float o = rnd<T>(__fsub_rn(ldv(z, idx), d));
        if (positivity)
            o = clamp_min(o, 0.0f);
        if (mask)
            o = rnd<T>(__fmul_rn(o, ldv(mask, mask_full ? idx : (size_t)p)));
        v[idx] = float_to_t<T>(o);
    }
}

/* One dual step. FGP: reads r and p (= p_prev), writes p = p', r = p' +
 * beta (p' - p_prev). Chambolle (r == p): writes p = p'. The padded plane of
 * each component (i = Ny-1 for y, ...) is neither read nor written. */
template <typename T, bool FGP>
__global__ void tv_prox_dual_kernel(
    const T *__restrict__ v,
    T *ry, T *rz, T *rx,
    T *py, T *pz, T *px,
    float step, float inv_lam, float beta,
    int Ny, int Nz, int Nx
) {
    const int plane = Nz * Nx;
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= plane) return;
    const int k = p % Nx, j = p / Nx;
    const bool hz = j <= Nz - 2, hx = k <= Nx - 2;
    for (int i = blockIdx.y; i < Ny; i += gridDim.y) {
        const bool hy = i <= Ny - 2;
        size_t idx = (size_t)i * plane + p;
        float v0 = ldv(v, idx);
        /* q = r + step * (forward difference); 0 on the padded plane */
        float qy = 0.0f, qz = 0.0f, qx = 0.0f;
        if (hy) {
            float g = rnd<T>(__fsub_rn(ldv(v, idx + plane), v0));
            qy = rnd<T>(__fadd_rn(ldv(ry, idx), rnd<T>(__fmul_rn(g, step))));
        }
        if (hz) {
            float g = rnd<T>(__fsub_rn(ldv(v, idx + Nx), v0));
            qz = rnd<T>(__fadd_rn(ldv(rz, idx), rnd<T>(__fmul_rn(g, step))));
        }
        if (hx) {
            float g = rnd<T>(__fsub_rn(ldv(v, idx + 1), v0));
            qx = rnd<T>(__fadd_rn(ldv(rx, idx), rnd<T>(__fmul_rn(g, step))));
        }
        float s = rnd<T>(__fadd_rn(rnd<T>(__fmul_rn(qy, qy)), rnd<T>(__fmul_rn(qz, qz))));
        s = rnd<T>(__fadd_rn(s, rnd<T>(__fmul_rn(qx, qx))));
        float nrm = rnd<T>(__fsqrt_rn(s));
        float den = clamp_min(rnd<T>(__fmul_rn(nrm, inv_lam)), 1.0f);
        float q[3] = {qy, qz, qx};
        bool has[3] = {hy, hz, hx};
        T *pp[3] = {py, pz, px};
        T *rr[3] = {ry, rz, rx};
        #pragma unroll
        for (int c = 0; c < 3; c++) {
            if (!has[c]) continue;
            float pn = rnd<T>(__fdiv_rn(q[c], den));
            if (FGP) {
                float d = rnd<T>(__fsub_rn(pn, ldv(pp[c], idx)));
                float e = rnd<T>(__fmul_rn(d, beta));
                rr[c][idx] = float_to_t<T>(rnd<T>(__fadd_rn(pn, e)));
            }
            pp[c][idx] = float_to_t<T>(pn);
        }
    }
}

template <typename T>
void run(int fgp, const void *z_, void *v_, void *work_, const void *mask_, int mask_full,
         int positivity, float step, float inv_lam, const float *betas, int n_iter,
         int Ny, int Nz, int Nx, cudaStream_t stream) {
    const T *z = (const T *)z_, *mask = (const T *)mask_;
    T *v = (T *)v_, *w = (T *)work_;
    const size_t n = (size_t)Ny * Nz * Nx;
    /* duals: p = w[0:3n]; FGP r = w[3n:6n], Chambolle r = p */
    T *py = w, *pz = w + n, *px = w + 2 * n;
    T *ry = fgp ? w + 3 * n : py, *rz = fgp ? w + 4 * n : pz, *rx = fgp ? w + 5 * n : px;
    FASTTOMO_CUDA_CHECK(cudaMemsetAsync(w, 0, (fgp ? 6 : 3) * n * sizeof(T), stream));

    const int block = 256;
    const int plane = Nz * Nx;
    dim3 grid((unsigned)((plane + block - 1) / block), (unsigned)(Ny < 65535 ? Ny : 65535));
    for (int it = 0; it < n_iter; it++) {
        const float beta = betas[it];
        tv_prox_primal_kernel<T><<<grid, block, 0, stream>>>(
            z, ry, rz, rx, mask, mask_full, positivity, v, Ny, Nz, Nx);
        CUDA_CHECK_KERNEL();
        if (fgp)
            tv_prox_dual_kernel<T, true><<<grid, block, 0, stream>>>(
                v, ry, rz, rx, py, pz, px, step, inv_lam, beta, Ny, Nz, Nx);
        else
            tv_prox_dual_kernel<T, false><<<grid, block, 0, stream>>>(
                v, ry, rz, rx, py, pz, px, step, inv_lam, beta, Ny, Nz, Nx);
        CUDA_CHECK_KERNEL();
    }
    tv_prox_primal_kernel<T><<<grid, block, 0, stream>>>(
        z, py, pz, px, mask, mask_full, positivity, v, Ny, Nz, Nx);
    CUDA_CHECK_KERNEL();
}

}  // namespace

void tv_prox_cuda(
    int io_dtype, int fgp,
    const void *z, void *v_out, void *work,
    const void *mask, int mask_full, int positivity,
    float step, float inv_lam, const float *betas, int n_iter,
    int Ny, int Nz, int Nx,
    cudaStream_t stream
) {
    check_io_dtype(io_dtype, "tv_prox_cuda");
    if (Ny <= 0 || Nz <= 0 || Nx <= 0) return;
    if ((long long)Nz * Nx > 0x7fffffffLL)
        throw std::invalid_argument("tv_prox_cuda: Nz * Nx too large");
    if (!z || !v_out || !work || (n_iter > 0 && !betas) || n_iter < 0)
        throw std::invalid_argument("tv_prox_cuda: null z, v_out, work or betas");
    if (io_dtype == 0)
        run<float>(fgp, z, v_out, work, mask, mask_full, positivity, step, inv_lam, betas,
                   n_iter, Ny, Nz, Nx, stream);
    else
        run<__half>(fgp, z, v_out, work, mask, mask_full, positivity, step, inv_lam, betas,
                    n_iter, Ny, Nz, Nx, stream);
}

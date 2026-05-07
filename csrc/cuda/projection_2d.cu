/* ── fasttomo/csrc/cuda/projection_2d.cu ─────────────────────────────────────
 * Scalar single-axis (2D) forward and back projection.
 * Kernels: forward_project_kernel, forward_project_kernel_packed_h4,
 *          back_project_kernel, back_project_kernel_packed_h4.
 * Host wrappers: forward_project_cuda, forward_project_cuda_cached,
 *                back_project_cuda, back_project_cuda_cached.
 */

#include "common.cuh"
#include "texture_cache.h"
#include "../include/proj_geom.h"
#include "ops.h"

/* ── forward projection ───────────────────────────────────────────────────────
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

/* ── 4-channel packed forward-projection kernel ───────────────────────────────
 * Reads tex2DLayered<float4> (4 packed Y-slices per fetch), accumulates 4 line
 * integrals in parallel, writes 4 strided outputs. Grid Z is Ny/4. */
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

/* ── back-projection ──────────────────────────────────────────────────────────
 * Grid:  (ceil(Nx/16), ceil(Nz/16), Ny)   Block: (16, 16, 1)
 * Each thread owns one (ix, iz) voxel in one Y-slice and loops over angles.
 * cos/sin preloaded into shared memory via load_trig_smem.
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

    load_trig_smem(sh_cos, sh_sin, cos_a, sin_a, n_angles);

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

/* ── 4-channel packed back-projection kernel ──────────────────────────────────
 * Each thread owns one (ix, iz, iy_pack) voxel-group of 4 voxels (4 Y-slices).
 * Reads tex2DLayered<float4>; writes 4 strided outputs. */
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

    load_trig_smem(sh_cos, sh_sin, cos_a, sin_a, n_angles);

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

/* ────────────────────────────────────────────────────────────────────────────
 * Host-side implementations
 * ──────────────────────────────────────────────────────────────────────────── */

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
    ProjGeom2D g = proj_geom_2d(Ny, Nz, Nx, n_angles, N_det);

    tc_dtype_t dtype = (sizeof(T) == sizeof(float)) ? TC_DTYPE_FP32 : TC_DTYPE_FP16;
    TextureCache *tc = tc_create(TC_VOL_LAYERED, dtype, 1, Ny, Nz, Nx);
    tc_upload(tc, d_volume, stream);

    dim3 block(16, 16, 1);
    dim3 grid((g.N_det    + 15) / 16,
              (g.n_angles + 15) / 16,
              g.Ny);

    forward_project_kernel<T><<<grid, block, 0, stream>>>(
        tc->tex, d_sinogram, d_cos_a, d_sin_a,
        g.Ny, g.Nz, g.Nx, g.n_angles, g.N_det,
        g.half_det, g.half_x, g.half_z, g.ray_len
    );
    CUDA_CHECK_KERNEL();

    tc_destroy(tc);
}

static void forward_project_cuda_impl_packed_h4(
    const __half *d_volume, __half *d_sinogram,
    const float  *d_cos_a, const float *d_sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
) {
    ProjGeom2D g = proj_geom_2d(Ny, Nz, Nx, n_angles, N_det);

    TextureCache *tc = tc_create(TC_VOL_LAYERED, TC_DTYPE_FP16, 4, Ny, Nz, Nx);
    tc_upload(tc, d_volume, stream);

    dim3 block(16, 16, 1);
    dim3 grid((g.N_det    + 15) / 16,
              (g.n_angles + 15) / 16,
              g.Ny / 4);

    forward_project_kernel_packed_h4<<<grid, block, 0, stream>>>(
        tc->tex, d_sinogram, d_cos_a, d_sin_a,
        g.Ny, g.Nz, g.Nx, g.n_angles, g.N_det,
        g.half_det, g.half_x, g.half_z, g.ray_len);
    CUDA_CHECK_KERNEL();

    tc_destroy(tc);
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
    ProjGeom2D g = proj_geom_2d(Ny, Nz, Nx, n_angles, N_det);

    tc_upload(vol_cache, d_volume, stream);

    dim3 block(16, 16, 1);
    dim3 grid((g.N_det    + 15) / 16,
              (g.n_angles + 15) / 16,
              g.Ny);

    forward_project_kernel<T><<<grid, block, 0, stream>>>(
        vol_cache->tex, d_sinogram, d_cos_a, d_sin_a,
        g.Ny, g.Nz, g.Nx, g.n_angles, g.N_det,
        g.half_det, g.half_x, g.half_z, g.ray_len
    );
    CUDA_CHECK_KERNEL();
}

static void forward_project_cuda_cached_impl_packed_h4(
    TextureCache *vol_cache,
    const __half *d_volume, __half *d_sinogram,
    const float  *d_cos_a, const float *d_sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
) {
    ProjGeom2D g = proj_geom_2d(Ny, Nz, Nx, n_angles, N_det);

    tc_upload(vol_cache, d_volume, stream);

    dim3 block(16, 16, 1);
    dim3 grid((g.N_det    + 15) / 16,
              (g.n_angles + 15) / 16,
              g.Ny / 4);

    forward_project_kernel_packed_h4<<<grid, block, 0, stream>>>(
        vol_cache->tex, d_sinogram, d_cos_a, d_sin_a,
        g.Ny, g.Nz, g.Nx, g.n_angles, g.N_det,
        g.half_det, g.half_x, g.half_z, g.ray_len);
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

/* ────────────────────────────────────────────────────────────────────────────
 * Back-projection host-side implementations
 * ──────────────────────────────────────────────────────────────────────────── */

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
    ProjGeom2D g = proj_geom_2d(Ny, Nz, Nx, n_angles, N_det);

    tc_dtype_t dtype = (sizeof(T) == sizeof(float)) ? TC_DTYPE_FP32 : TC_DTYPE_FP16;
    TextureCache *tc = tc_create(TC_SINO_LAYERED, dtype, 1, n_angles, Ny, N_det);
    tc_upload(tc, d_sinogram, stream);

    dim3 block(16, 16, 1);
    dim3 grid((g.Nx + 15) / 16, (g.Nz + 15) / 16, g.Ny);
    size_t smem = 2 * (size_t)n_angles * sizeof(float);

    enable_extended_smem_if_needed((const void *)back_project_kernel<T>, smem);

    back_project_kernel<T><<<grid, block, smem, stream>>>(
        tc->tex, d_volume, d_cos_a, d_sin_a,
        g.Ny, g.Nz, g.Nx, g.n_angles, g.N_det,
        g.half_det, g.half_x, g.half_z
    );
    CUDA_CHECK_KERNEL();

    tc_destroy(tc);
}

static void back_project_cuda_impl_packed_h4(
    const __half *d_sinogram, __half *d_volume,
    const float  *d_cos_a, const float *d_sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
) {
    ProjGeom2D g = proj_geom_2d(Ny, Nz, Nx, n_angles, N_det);

    TextureCache *tc = tc_create(TC_SINO_LAYERED, TC_DTYPE_FP16, 4, n_angles, Ny, N_det);
    tc_upload(tc, d_sinogram, stream);

    dim3 block(16, 16, 1);
    dim3 grid((g.Nx + 15) / 16, (g.Nz + 15) / 16, g.Ny / 4);
    size_t smem = 2 * (size_t)n_angles * sizeof(float);

    enable_extended_smem_if_needed((const void *)back_project_kernel_packed_h4, smem);

    back_project_kernel_packed_h4<<<grid, block, smem, stream>>>(
        tc->tex, d_volume, d_cos_a, d_sin_a,
        g.Ny, g.Nz, g.Nx, g.n_angles, g.N_det,
        g.half_det, g.half_x, g.half_z);
    CUDA_CHECK_KERNEL();

    tc_destroy(tc);
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
    ProjGeom2D g = proj_geom_2d(Ny, Nz, Nx, n_angles, N_det);

    tc_upload(sino_cache, d_sinogram, stream);

    dim3 block(16, 16, 1);
    dim3 grid((g.Nx + 15) / 16, (g.Nz + 15) / 16, g.Ny);
    size_t smem = 2 * (size_t)n_angles * sizeof(float);

    enable_extended_smem_if_needed((const void *)back_project_kernel<T>, smem);

    back_project_kernel<T><<<grid, block, smem, stream>>>(
        sino_cache->tex, d_volume, d_cos_a, d_sin_a,
        g.Ny, g.Nz, g.Nx, g.n_angles, g.N_det,
        g.half_det, g.half_x, g.half_z
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
    ProjGeom2D g = proj_geom_2d(Ny, Nz, Nx, n_angles, N_det);

    tc_upload(sino_cache, d_sinogram, stream);

    dim3 block(16, 16, 1);
    dim3 grid((g.Nx + 15) / 16, (g.Nz + 15) / 16, g.Ny / 4);
    size_t smem = 2 * (size_t)n_angles * sizeof(float);

    enable_extended_smem_if_needed((const void *)back_project_kernel_packed_h4, smem);

    back_project_kernel_packed_h4<<<grid, block, smem, stream>>>(
        sino_cache->tex, d_volume, d_cos_a, d_sin_a,
        g.Ny, g.Nz, g.Nx, g.n_angles, g.N_det,
        g.half_det, g.half_x, g.half_z);
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

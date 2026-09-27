/* ── fasttomo/csrc/cuda/projection_rotmat.cu ────────────────────────────────
 * Arbitrary-rotation ("rotmat": one 3x3 matrix per view) forward and back
 * projection, plus the gradient w.r.t. the matrices (grad_rotmat).
 * Kernels: forward_project_rotmat_kernel, back_project_rotmat_kernel,
 *          grad_rotmat_kernel.
 * Host wrappers: forward_project_rotmat_cuda, forward_project_rotmat_cuda_cached,
 *                back_project_rotmat_cuda, back_project_rotmat_cuda_cached,
 *                grad_rotmat_cuda.
 */

#include "common.cuh"
#include "texture_cache.h"
#include "../include/proj_geom.h"
#include "ops.h"

/* ── 3D rotation-matrix forward projection ────────────────────────────────────
 * forward_project_rotmat_kernel
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
    T *__restrict__      sino,         /* [n_angles, Ny, N_det] */
    const float *__restrict__ mats,    /* [n_angles * 9] row-major */
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    float half_det, float half_x, float half_z, float half_y,
    int ray_len
) {
    int iu = blockIdx.x * blockDim.x + threadIdx.x;  /* detector u */
    int iv = blockIdx.y * blockDim.y + threadIdx.y;  /* detector v = Y-slice */
    int ia = blockIdx.z;

    if (iu >= N_det || iv >= Ny) return;

    const float *M = mats + ia * 9;
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
    sino[((size_t)ia * Ny + iv) * N_det + iu] = float_to_t<T>(sum);
}

static tc_dtype_t tc_dtype_of(int io_dtype) {
    return io_dtype == 0 ? TC_DTYPE_FP32 : TC_DTYPE_FP16;
}

/* Upload `vol` into `vol_cache` (already validated) and launch the FP
 * kernel. Shared by the cached and the one-shot entry point. */
static void forward_project_rotmat_launch(
    int io_dtype, TextureCache *vol_cache,
    const void *vol, void *sino, const float *mats,
    int Ny, int Nz, int Nx, int n_angles, int N_det,
    cudaStream_t stream
) {
    ProjGeomRotmat g = proj_geom_rotmat(Ny, Nz, Nx, n_angles, N_det);
    tc_upload(vol_cache, vol, stream);

    dim3 block(16, 16, 1);
    dim3 grid((g.N_det + 15) / 16,
              (g.Ny    + 15) / 16,
              g.n_angles);
    if (io_dtype == 0)
        forward_project_rotmat_kernel<float><<<grid, block, 0, stream>>>(
            vol_cache->tex, (float *)sino, mats,
            g.Ny, g.Nz, g.Nx, g.n_angles, g.N_det,
            g.half_det, g.half_x, g.half_z, g.half_y, g.ray_len);
    else
        forward_project_rotmat_kernel<__half><<<grid, block, 0, stream>>>(
            vol_cache->tex, (__half *)sino, mats,
            g.Ny, g.Nz, g.Nx, g.n_angles, g.N_det,
            g.half_det, g.half_x, g.half_z, g.half_y, g.ray_len);
    CUDA_CHECK_KERNEL();
}

void forward_project_rotmat_cuda(
    int io_dtype,
    const void  *vol,
    void        *sino,
    const float *mats,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
) {
    check_io_dtype(io_dtype, "forward_project_rotmat_cuda");
    /* Rotmat samples continuously in Y, so no 4-channel Y packing. */
    ScopedTextureCache tc(TC_VOL_3D, tc_dtype_of(io_dtype), 1, Ny, Nz, Nx, stream);
    forward_project_rotmat_launch(io_dtype, tc.get(), vol, sino,
                                  mats, Ny, Nz, Nx, n_angles, N_det, stream);
    tc.finish();
}

void forward_project_rotmat_cuda_cached(
    int io_dtype,
    TextureCache *vol_cache,
    const void   *vol,
    void         *sino,
    const float  *mats,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
) {
    check_io_dtype(io_dtype, "forward_project_rotmat_cuda_cached");
    tc_validate(vol_cache, TC_VOL_3D, io_dtype, Ny, Nz, Nx,
                "forward_project_rotmat_cuda_cached");
    forward_project_rotmat_launch(io_dtype, vol_cache, vol, sino,
                                  mats, Ny, Nz, Nx, n_angles, N_det, stream);
}

/* ── 3D rotation-matrix back-projection ──────────────────────────────────────
 * back_project_rotmat_kernel
 * Grid: (⌈Nx/8⌉, ⌈Nz/8⌉, ⌈Ny/4⌉)   Block: (8, 8, 4)
 * Each thread owns one (ix, iz, iy) voxel and loops over all angles.
 * Rotation matrices loaded into shared memory (9 * n_angles floats).
 * Sinogram sampled with tex2DLayered (n_angles layers of [Ny, N_det]).
 */
template <typename T>
__global__ static void back_project_rotmat_kernel(
    cudaTextureObject_t  sino_tex,        /* n_angles layers of [Ny, N_det] */
    T *__restrict__      vol,          /* [Ny, Nz, Nx] */
    const float *__restrict__ mats,   /* [n_angles * 9] row-major */
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
        sh_mats[k] = mats[k];
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

    vol[((size_t)iy * Nz + iz) * Nx + ix] = float_to_t<T>(val);
}

/* Upload `sino` into `sino_cache` (already validated) and launch the
 * BP kernel. Shared by the cached and the one-shot entry point. */
static void back_project_rotmat_launch(
    int io_dtype, TextureCache *sino_cache,
    const void *sino, void *vol, const float *mats,
    int Ny, int Nz, int Nx, int n_angles, int N_det,
    cudaStream_t stream
) {
    ProjGeomRotmat g = proj_geom_rotmat(Ny, Nz, Nx, n_angles, N_det);
    tc_upload(sino_cache, sino, stream);

    dim3 block(8, 8, 4);
    dim3 grid((g.Nx + 7) / 8,
              (g.Nz + 7) / 8,
              (g.Ny + 3) / 4);
    size_t smem = 9 * (size_t)n_angles * sizeof(float);
    if (io_dtype == 0) {
        enable_extended_smem_if_needed((const void *)back_project_rotmat_kernel<float>, smem);
        back_project_rotmat_kernel<float><<<grid, block, smem, stream>>>(
            sino_cache->tex, (float *)vol, mats,
            g.Ny, g.Nz, g.Nx, g.n_angles, g.N_det,
            g.half_det, g.half_x, g.half_z, g.half_y);
    } else {
        enable_extended_smem_if_needed((const void *)back_project_rotmat_kernel<__half>, smem);
        back_project_rotmat_kernel<__half><<<grid, block, smem, stream>>>(
            sino_cache->tex, (__half *)vol, mats,
            g.Ny, g.Nz, g.Nx, g.n_angles, g.N_det,
            g.half_det, g.half_x, g.half_z, g.half_y);
    }
    CUDA_CHECK_KERNEL();
}

void back_project_rotmat_cuda(
    int io_dtype,
    const void  *sino,
    void        *vol,
    const float *mats,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
) {
    check_io_dtype(io_dtype, "back_project_rotmat_cuda");
    ScopedTextureCache tc(TC_SINO_LAYERED, tc_dtype_of(io_dtype), 1,
                          n_angles, Ny, N_det, stream);
    back_project_rotmat_launch(io_dtype, tc.get(), sino, vol,
                               mats, Ny, Nz, Nx, n_angles, N_det, stream);
    tc.finish();
}

void back_project_rotmat_cuda_cached(
    int io_dtype,
    TextureCache *sino_cache,
    const void   *sino,
    void         *vol,
    const float  *mats,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
) {
    check_io_dtype(io_dtype, "back_project_rotmat_cuda_cached");
    tc_validate(sino_cache, TC_SINO_LAYERED, io_dtype, n_angles, Ny, N_det,
                "back_project_rotmat_cuda_cached");
    back_project_rotmat_launch(io_dtype, sino_cache, sino, vol,
                               mats, Ny, Nz, Nx, n_angles, N_det, stream);
}

/* ── gradient of 3D rotmat forward projection w.r.t. rotation matrix ─────────
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
 *
 * T is the volume dtype (float or __half). The 8 corner reads are done with
 * tex3D<float> and cudaFilterModePoint so the result is bit-exact equivalent
 * to a global-memory load (hardware promotes fp16 texels to fp32 on read).
 * The output grad_mats always accumulates in fp32 regardless of T.
 */
template <typename T>
__global__ static void grad_rotmat_kernel(
    cudaTextureObject_t  vol_tex,            /* TC_VOL_3D, point-filter view */
    const float *__restrict__ grad_sino,     /* [n_angles, Ny, N_det] */
    const float *__restrict__ mats,      /* [n_angles*9] row-major */
    float       *__restrict__ grad_mats, /* [n_angles*9] (zeroed by caller) */
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
        const float *M = mats + (size_t)ia * 9;
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

                /* Loose bounds, the same as forward_project_rotmat_kernel
                 * (border-mode linear texture) and grad_rotmat_cpu: a sample
                 * contributes while any of its 8 corners is inside the
                 * volume, i.e. fx in (-1, Nx) and likewise for y, z. Corners
                 * outside the volume read 0 through the border-mode point
                 * texture, so the samples within one voxel of a face carry
                 * the (large) derivative of the fall-off to zero. A tight
                 * [0, N-1) test drops them and is not the gradient of the
                 * forward. floorf, not (int) truncation, because fx can be
                 * negative here. */
                if (fx <= -1.0f || fx >= (float)Nx ||
                    fy <= -1.0f || fy >= (float)Ny ||
                    fz <= -1.0f || fz >= (float)Nz)
                    continue;

                int ix0 = (int)floorf(fx), iy0 = (int)floorf(fy), iz0 = (int)floorf(fz);
                int ix1 = ix0 + 1, iy1 = iy0 + 1, iz1 = iz0 + 1;
                float wx = fx - (float)ix0;
                float wy = fy - (float)iy0;
                float wz = fz - (float)iz0;
                float ux = 1.0f - wx, uy = 1.0f - wy, uz = 1.0f - wz;

                /* Point-filter tex3D reads: +0.5f offset maps integer index i
                 * to the texel centre coordinate i+0.5, which with
                 * cudaFilterModePoint rounds back to texel i.  The hardware
                 * promotes fp16 texels to fp32, so tex3D<float> is correct
                 * for both T=float and T=__half.
                 * tex3D axis mapping: (x→Nx width, z→Nz height, y→Ny depth). */
                float v000 = tex3D<float>(vol_tex, ix0+0.5f, iz0+0.5f, iy0+0.5f);
                float v001 = tex3D<float>(vol_tex, ix1+0.5f, iz0+0.5f, iy0+0.5f);
                float v010 = tex3D<float>(vol_tex, ix0+0.5f, iz1+0.5f, iy0+0.5f);
                float v011 = tex3D<float>(vol_tex, ix1+0.5f, iz1+0.5f, iy0+0.5f);
                float v100 = tex3D<float>(vol_tex, ix0+0.5f, iz0+0.5f, iy1+0.5f);
                float v101 = tex3D<float>(vol_tex, ix1+0.5f, iz0+0.5f, iy1+0.5f);
                float v110 = tex3D<float>(vol_tex, ix0+0.5f, iz1+0.5f, iy1+0.5f);
                float v111 = tex3D<float>(vol_tex, ix1+0.5f, iz1+0.5f, iy1+0.5f);

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
        atomicAdd(&grad_mats[(size_t)ia * 9 + tid], smem[tid * nthreads]);
}

/* Launch grad_rotmat_kernel on a TC_VOL_3D cache that already holds the
 * volume (no upload: the cached path reuses the forward's upload). */
static void grad_rotmat_launch(
    int io_dtype, TextureCache *vol_cache,
    const float *grad_sino, const float *mats, float *grad_mats,
    int Ny, int Nz, int Nx, int n_angles, int N_det,
    cudaStream_t stream
) {
    if (vol_cache->tex_point == 0)
        throw std::invalid_argument("grad_rotmat: tex_point not initialised "
                                    "(cache was not created as TC_VOL_3D with channels==1)");

    ProjGeomRotmat g = proj_geom_rotmat(Ny, Nz, Nx, n_angles, N_det);

    dim3 block(16, 16, 1);
    dim3 grid((g.N_det + 15) / 16,
              (g.Ny    + 15) / 16,
              g.n_angles);
    size_t smem = 9 * (size_t)(block.x * block.y) * sizeof(float);

    if (io_dtype == 0) {
        enable_extended_smem_if_needed((const void *)grad_rotmat_kernel<float>, smem);
        grad_rotmat_kernel<float><<<grid, block, smem, stream>>>(
            vol_cache->tex_point, grad_sino, mats, grad_mats,
            g.Ny, g.Nz, g.Nx, g.n_angles, g.N_det,
            g.half_det, g.half_x, g.half_y, g.half_z, g.ray_len
        );
    } else {
        enable_extended_smem_if_needed((const void *)grad_rotmat_kernel<__half>, smem);
        grad_rotmat_kernel<__half><<<grid, block, smem, stream>>>(
            vol_cache->tex_point, grad_sino, mats, grad_mats,
            g.Ny, g.Nz, g.Nx, g.n_angles, g.N_det,
            g.half_det, g.half_x, g.half_y, g.half_z, g.ray_len
        );
    }
    CUDA_CHECK_KERNEL();
}

void grad_rotmat_cuda_cached(
    int io_dtype,
    TextureCache *vol_cache,
    const float *grad_sino,
    const float *mats,
    float       *grad_mats,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
) {
    check_io_dtype(io_dtype, "grad_rotmat_cuda_cached");
    tc_validate(vol_cache, TC_VOL_3D, io_dtype, Ny, Nz, Nx, "grad_rotmat_cuda_cached");
    grad_rotmat_launch(io_dtype, vol_cache, grad_sino, mats, grad_mats,
                       Ny, Nz, Nx, n_angles, N_det, stream);
}

void grad_rotmat_cuda(
    const float *vol,
    const float *grad_sino,
    const float *mats,
    float       *grad_mats,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
) {
    ScopedTextureCache tc(TC_VOL_3D, TC_DTYPE_FP32, 1, Ny, Nz, Nx, stream);
    tc_upload(tc.get(), vol, stream);
    grad_rotmat_launch(0, tc.get(), grad_sino, mats, grad_mats,
                       Ny, Nz, Nx, n_angles, N_det, stream);
    tc.finish();
}

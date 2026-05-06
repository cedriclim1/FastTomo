#include "ops.h"
#include <cuda_runtime.h>
#include <math.h>

/* ── helpers ──────────────────────────────────────────────────────────────
 * make_vol_texture:  wrap a device [Ny, Nz, Nx] float buffer in a layered
 *                    2D texture (Ny layers of [Nz, Nx]).
 * make_sino_texture: wrap a device [n_angles, Ny, N_det] float buffer in a
 *                    layered 2D texture (n_angles layers of [Ny, N_det]).
 *
 * Both use hardware linear filtering (free bilinear interpolation) and
 * border addressing (returns 0.0 for out-of-bounds coordinates).
 * Caller must call cudaDestroyTextureObject + cudaFreeArray when done.
 */
static cudaTextureObject_t make_vol_texture(
    const float *d_vol, int Ny, int Nz, int Nx,
    cudaArray_t &out_array
) {
    cudaChannelFormatDesc fmt = cudaCreateChannelDesc<float>();
    /* Ny layers, each Nx wide × Nz tall */
    cudaMalloc3DArray(&out_array, &fmt,
                      make_cudaExtent(Nx, Nz, Ny), cudaArrayLayered);

    cudaMemcpy3DParms cp = {};
    cp.srcPtr   = make_cudaPitchedPtr((void *)d_vol,
                                      Nx * sizeof(float), Nx, Nz);
    cp.dstArray = out_array;
    cp.extent   = make_cudaExtent(Nx, Nz, Ny);
    cp.kind     = cudaMemcpyDeviceToDevice;
    cudaMemcpy3D(&cp);

    cudaResourceDesc rd = {};
    rd.resType         = cudaResourceTypeArray;
    rd.res.array.array = out_array;

    cudaTextureDesc td = {};
    td.addressMode[0]  = cudaAddressModeBorder;
    td.addressMode[1]  = cudaAddressModeBorder;
    td.filterMode      = cudaFilterModeLinear;
    td.readMode        = cudaReadModeElementType;
    td.normalizedCoords = 0;

    cudaTextureObject_t tex;
    cudaCreateTextureObject(&tex, &rd, &td, nullptr);
    return tex;
}

static cudaTextureObject_t make_sino_texture(
    const float *d_sino, int n_angles, int Ny, int N_det,
    cudaArray_t &out_array
) {
    cudaChannelFormatDesc fmt = cudaCreateChannelDesc<float>();
    /* n_angles layers, each N_det wide × Ny tall */
    cudaMalloc3DArray(&out_array, &fmt,
                      make_cudaExtent(N_det, Ny, n_angles), cudaArrayLayered);

    cudaMemcpy3DParms cp = {};
    cp.srcPtr   = make_cudaPitchedPtr((void *)d_sino,
                                      N_det * sizeof(float), N_det, Ny);
    cp.dstArray = out_array;
    cp.extent   = make_cudaExtent(N_det, Ny, n_angles);
    cp.kind     = cudaMemcpyDeviceToDevice;
    cudaMemcpy3D(&cp);

    cudaResourceDesc rd = {};
    rd.resType         = cudaResourceTypeArray;
    rd.res.array.array = out_array;

    cudaTextureDesc td = {};
    td.addressMode[0]  = cudaAddressModeBorder;
    td.addressMode[1]  = cudaAddressModeBorder;
    td.filterMode      = cudaFilterModeLinear;
    td.readMode        = cudaReadModeElementType;
    td.normalizedCoords = 0;

    cudaTextureObject_t tex;
    cudaCreateTextureObject(&tex, &rd, &td, nullptr);
    return tex;
}

/* ── forward projection ───────────────────────────────────────────────────
 * Grid:  (ceil(N_det/16), ceil(n_angles/16), Ny)   Block: (16, 16, 1)
 * Each thread owns one (detector, angle) output cell in one Y-slice.
 * Marches along the ray; tex2DLayered handles bilinear interp and border.
 */
__global__ static void forward_project_kernel(
    cudaTextureObject_t  vol_tex,    /* Ny layers of [Nz, Nx] */
    float *__restrict__  sinogram,   /* [n_angles, Ny, N_det] */
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
        /* tex2DLayered: x→Nx axis, y→Nz axis, layer→iy.
         * +0.5 maps integer pixel index to texel centre.
         * Border addressing returns 0 outside the volume automatically. */
        sum += tex2DLayered<float>(vol_tex, fx + 0.5f, fz + 0.5f, iy);
    }

    sinogram[((size_t)ia * Ny + iy) * N_det + id] = sum;
}

void forward_project_cuda(
    const float *d_volume,
    float       *d_sinogram,
    const float *d_cos_a,
    const float *d_sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det
) {
    float half_det = (N_det - 1) * 0.5f;
    float half_x   = (Nx   - 1) * 0.5f;
    float half_z   = (Nz   - 1) * 0.5f;
    int   ray_len  = (int)ceilf(0.5f * sqrtf((float)(Nx * Nx + Nz * Nz))) + 1;

    cudaArray_t vol_array;
    cudaTextureObject_t vol_tex = make_vol_texture(d_volume, Ny, Nz, Nx, vol_array);

    dim3 block(16, 16, 1);
    dim3 grid((N_det    + 15) / 16,
              (n_angles + 15) / 16,
              Ny);

    forward_project_kernel<<<grid, block>>>(
        vol_tex, d_sinogram, d_cos_a, d_sin_a,
        Ny, Nz, Nx, n_angles, N_det,
        half_det, half_x, half_z, ray_len
    );

    cudaDestroyTextureObject(vol_tex);
    cudaFreeArray(vol_array);
}

/* ── back-projection ──────────────────────────────────────────────────────
 * Grid:  (ceil(Nx/16), ceil(Nz/16), Ny)   Block: (16, 16, 1)
 * Each thread owns one (ix, iz) voxel in one Y-slice and loops over angles.
 * cos_a/sin_a are preloaded into shared memory. tex2DLayered on the
 * sinogram provides free hardware interpolation along the detector axis.
 */
__global__ static void back_project_kernel(
    cudaTextureObject_t  sino_tex,   /* n_angles layers of [Ny, N_det] */
    float *__restrict__  volume,     /* [Ny, Nz, Nx] */
    const float *__restrict__ cos_a,
    const float *__restrict__ sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    float half_det, float half_x, float half_z
) {
    extern __shared__ float sh[];
    float *sh_cos = sh;
    float *sh_sin = sh + n_angles;

    for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < n_angles;
         i += blockDim.x * blockDim.y) {
        sh_cos[i] = cos_a[i];
        sh_sin[i] = sin_a[i];
    }
    __syncthreads();

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
         * iy_tc is always at a texel centre so no interpolation in Y. */
        val += tex2DLayered<float>(sino_tex, ft + 0.5f, iy_tc, ia);
    }

    volume[((size_t)iy * Nz + iz) * Nx + ix] = val;
}

void back_project_cuda(
    const float *d_sinogram,
    float       *d_volume,
    const float *d_cos_a,
    const float *d_sin_a,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det
) {
    float half_det = (N_det - 1) * 0.5f;
    float half_x   = (Nx   - 1) * 0.5f;
    float half_z   = (Nz   - 1) * 0.5f;

    cudaArray_t sino_array;
    cudaTextureObject_t sino_tex =
        make_sino_texture(d_sinogram, n_angles, Ny, N_det, sino_array);

    dim3 block(16, 16, 1);
    dim3 grid((Nx + 15) / 16, (Nz + 15) / 16, Ny);
    size_t smem = 2 * (size_t)n_angles * sizeof(float);

    back_project_kernel<<<grid, block, smem>>>(
        sino_tex, d_volume, d_cos_a, d_sin_a,
        Ny, Nz, Nx, n_angles, N_det,
        half_det, half_x, half_z
    );

    cudaDestroyTextureObject(sino_tex);
    cudaFreeArray(sino_array);
}

/* ── 3D rotation-matrix forward projection ───────────────────────────────
 * Wraps a [Ny, Nz, Nx] volume in a true tex3D (trilinear, border zero).
 * Unlike make_vol_texture (which uses layered 2D), this allows the ray to
 * cross Y-slice boundaries — required for general ZYZ Euler rotations.
 */
static cudaTextureObject_t make_vol_texture_3d(
    const float *d_vol, int Ny, int Nz, int Nx,
    cudaArray_t &out_array
) {
    cudaChannelFormatDesc fmt = cudaCreateChannelDesc<float>();
    /* extent: (width=Nx, height=Nz, depth=Ny) — matches C-order [Ny][Nz][Nx] */
    cudaMalloc3DArray(&out_array, &fmt,
                      make_cudaExtent(Nx, Nz, Ny), 0 /* no cudaArrayLayered */);

    cudaMemcpy3DParms cp = {};
    cp.srcPtr   = make_cudaPitchedPtr((void *)d_vol,
                                      Nx * sizeof(float), Nx, Nz);
    cp.dstArray = out_array;
    cp.extent   = make_cudaExtent(Nx, Nz, Ny);
    cp.kind     = cudaMemcpyDeviceToDevice;
    cudaMemcpy3D(&cp);

    cudaResourceDesc rd = {};
    rd.resType         = cudaResourceTypeArray;
    rd.res.array.array = out_array;

    cudaTextureDesc td = {};
    td.addressMode[0]  = cudaAddressModeBorder;
    td.addressMode[1]  = cudaAddressModeBorder;
    td.addressMode[2]  = cudaAddressModeBorder;
    td.filterMode      = cudaFilterModeLinear;
    td.readMode        = cudaReadModeElementType;
    td.normalizedCoords = 0;

    cudaTextureObject_t tex;
    cudaCreateTextureObject(&tex, &rd, &td, nullptr);
    return tex;
}

/* forward_project_rotmat_kernel
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
__global__ static void forward_project_rotmat_kernel(
    cudaTextureObject_t  vol_tex,
    float *__restrict__  sinogram,         /* [n_angles, Ny, N_det] */
    const float *__restrict__ matrices,    /* [n_angles * 9] row-major */
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    float half_det, float half_x, float half_z, float half_y,
    int ray_len
) {
    int iu = blockIdx.x * blockDim.x + threadIdx.x;  /* detector u */
    int iv = blockIdx.y * blockDim.y + threadIdx.y;  /* detector v = Y-slice */
    int ia = blockIdx.z;

    if (iu >= N_det || iv >= Ny) return;

    const float *M = matrices + ia * 9;
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
        /* tex3D: (x→width=Nx, z→height=Nz, y→depth=Ny) */
        sum += tex3D<float>(vol_tex, tx, tz, ty);
    }
    sinogram[((size_t)ia * Ny + iv) * N_det + iu] = sum;
}

void forward_project_rotmat_cuda(
    const float *d_volume,
    float       *d_sinogram,
    const float *d_matrices,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det
) {
    float half_det = (N_det - 1) * 0.5f;
    float half_x   = (Nx   - 1) * 0.5f;
    float half_z   = (Nz   - 1) * 0.5f;
    float half_y   = (Ny   - 1) * 0.5f;
    int   ray_len  = (int)ceilf(0.5f * sqrtf((float)(Nx*Nx + Ny*Ny + Nz*Nz))) + 1;

    cudaArray_t vol_array;
    cudaTextureObject_t vol_tex = make_vol_texture_3d(d_volume, Ny, Nz, Nx, vol_array);

    dim3 block(16, 16, 1);
    dim3 grid((N_det + 15) / 16,
              (Ny    + 15) / 16,
              n_angles);

    forward_project_rotmat_kernel<<<grid, block>>>(
        vol_tex, d_sinogram, d_matrices,
        Ny, Nz, Nx, n_angles, N_det,
        half_det, half_x, half_z, half_y, ray_len
    );

    cudaDestroyTextureObject(vol_tex);
    cudaFreeArray(vol_array);
}

/* back_project_rotmat_kernel
 * Grid: (⌈Nx/8⌉, ⌈Nz/8⌉, ⌈Ny/4⌉)   Block: (8, 8, 4)
 * Each thread owns one (ix, iz, iy) voxel and loops over all angles.
 * Rotation matrices loaded into shared memory (9 * n_angles floats).
 * Sinogram sampled with tex2DLayered (n_angles layers of [Ny, N_det]).
 */
__global__ static void back_project_rotmat_kernel(
    cudaTextureObject_t  sino_tex,        /* n_angles layers of [Ny, N_det] */
    float *__restrict__  volume,          /* [Ny, Nz, Nx] */
    const float *__restrict__ matrices,   /* [n_angles * 9] row-major */
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    float half_det, float half_x, float half_z, float half_y
) {
    extern __shared__ float sh_mats[];   /* 9 * n_angles floats */

    int tid       = threadIdx.x + threadIdx.y * blockDim.x
                  + threadIdx.z * blockDim.x * blockDim.y;
    int block_sz  = blockDim.x * blockDim.y * blockDim.z;
    for (int k = tid; k < n_angles * 9; k += block_sz)
        sh_mats[k] = matrices[k];
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
        /* tex2DLayered: x→N_det, y→Ny, layer→ia */
        val += tex2DLayered<float>(sino_tex,
                                   u_det + half_det + 0.5f,
                                   v_det + half_y   + 0.5f,
                                   ia);
    }

    volume[((size_t)iy * Nz + iz) * Nx + ix] = val;
}

void back_project_rotmat_cuda(
    const float *d_sinogram,
    float       *d_volume,
    const float *d_matrices,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det
) {
    float half_det = (N_det - 1) * 0.5f;
    float half_x   = (Nx   - 1) * 0.5f;
    float half_z   = (Nz   - 1) * 0.5f;
    float half_y   = (Ny   - 1) * 0.5f;

    cudaArray_t sino_array;
    cudaTextureObject_t sino_tex =
        make_sino_texture(d_sinogram, n_angles, Ny, N_det, sino_array);

    dim3 block(8, 8, 4);
    dim3 grid((Nx + 7) / 8,
              (Nz + 7) / 8,
              (Ny + 3) / 4);
    size_t smem = 9 * (size_t)n_angles * sizeof(float);

    back_project_rotmat_kernel<<<grid, block, smem>>>(
        sino_tex, d_volume, d_matrices,
        Ny, Nz, Nx, n_angles, N_det,
        half_det, half_x, half_z, half_y
    );

    cudaDestroyTextureObject(sino_tex);
    cudaFreeArray(sino_array);
}

/* ── RESIRE voxel-driven projectors ──────────────────────────────────────────
 * forward_project_resire_kernel  — voxel-driven scatter with 4-point
 *   supersampling; uses atomicAdd to the sinogram slice for this projection.
 * back_project_resire_kernel     — voxel-driven gather with 4-point
 *   supersampling; accumulates into volume (no atomics needed since each
 *   voxel is owned by exactly one thread and kernel launches are serialised).
 *
 * Both kernels operate on ONE projection at a time (called in a host loop).
 * Grid: (⌈Nx/16⌉, ⌈Nz/16⌉, Ny)   Block: (16, 16, 1)
 *
 * Supersampling: 4 sub-voxel offsets (±0.25, ±0.25) in volume XZ space are
 * rotated to detector coordinates via rows 0–1 of the projection matrix M.
 * The host divides the output by 4 to normalise to the single-sample scale.
 *
 * Matrix convention (row-major, same as rotmat kernels):
 *   detector-u = M[0]*x + M[1]*y + M[2]*z
 *   detector-v = M[3]*x + M[4]*y + M[5]*z
 */

__global__ static void forward_project_resire_kernel(
    const float *__restrict__ volume,     /* [Ny, Nz, Nx] */
    float *__restrict__       sino_slice, /* [Ny, N_det] for this projection */
    const float *__restrict__ M,          /* [9] rotation matrix */
    int Ny, int Nz, int Nx, int N_det,
    float half_x, float half_y, float half_z, float half_det
) {
    __shared__ float sh_M[9];
    if (threadIdx.y == 0 && threadIdx.x < 9)
        sh_M[threadIdx.x] = M[threadIdx.x];
    __syncthreads();

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int iy = (int)blockIdx.z;

    if (ix >= Nx || iz >= Nz) return;

    float x = ix - half_x;
    float y = iy - half_y;
    float z = iz - half_z;

    /* Base projected pixel coordinates */
    float u0 = sh_M[0]*x + sh_M[1]*y + sh_M[2]*z + half_det;
    float v0 = sh_M[3]*x + sh_M[4]*y + sh_M[5]*z + half_y;

    const float data_val = volume[((size_t)iy * Nz + iz) * Nx + ix];

    /* 4 sub-pixel offsets in volume XZ space */
    const float sdx[4] = { 0.25f,  0.25f, -0.25f, -0.25f };
    const float sdz[4] = { 0.25f, -0.25f,  0.25f, -0.25f };

    for (int s = 0; s < 4; s++) {
        /* Rotate the XZ shift to detector space (rows 0 and 1 of M, Y=0) */
        float u_s = u0 + sh_M[0]*sdx[s] + sh_M[2]*sdz[s];
        float v_s = v0 + sh_M[3]*sdx[s] + sh_M[5]*sdz[s];

        int u1 = (int)floorf(u_s);  int u2 = u1 + 1;
        int v1 = (int)floorf(v_s);  int v2 = v1 + 1;

        /* Skip if stencil centre is entirely off-detector */
        if ((u1 < 0 && u2 < 0) || (u1 >= N_det && u2 >= N_det)) continue;
        if ((v1 < 0 && v2 < 0) || (v1 >= Ny   && v2 >= Ny  )) continue;

        float wu2 = u_s - u1,  wu1 = 1.0f - wu2;
        float wv2 = v_s - v1,  wv1 = 1.0f - wv2;

        bool u1ok = (u1 >= 0 && u1 < N_det);
        bool u2ok = (u2 >= 0 && u2 < N_det);
        bool v1ok = (v1 >= 0 && v1 < Ny);
        bool v2ok = (v2 >= 0 && v2 < Ny);

        if (u1ok && v1ok) atomicAdd(&sino_slice[v1*N_det + u1], wu1*wv1*data_val);
        if (u2ok && v1ok) atomicAdd(&sino_slice[v1*N_det + u2], wu2*wv1*data_val);
        if (u1ok && v2ok) atomicAdd(&sino_slice[v2*N_det + u1], wu1*wv2*data_val);
        if (u2ok && v2ok) atomicAdd(&sino_slice[v2*N_det + u2], wu2*wv2*data_val);
    }
}

__global__ static void back_project_resire_kernel(
    const float *__restrict__ sino_slice, /* [Ny, N_det] for this projection */
    float *__restrict__       volume,     /* [Ny, Nz, Nx] — accumulated */
    const float *__restrict__ M,          /* [9] rotation matrix */
    int Ny, int Nz, int Nx, int N_det,
    float half_x, float half_y, float half_z, float half_det
) {
    __shared__ float sh_M[9];
    if (threadIdx.y == 0 && threadIdx.x < 9)
        sh_M[threadIdx.x] = M[threadIdx.x];
    __syncthreads();

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int iy = (int)blockIdx.z;

    if (ix >= Nx || iz >= Nz) return;

    float x = ix - half_x;
    float y = iy - half_y;
    float z = iz - half_z;

    float u0 = sh_M[0]*x + sh_M[1]*y + sh_M[2]*z + half_det;
    float v0 = sh_M[3]*x + sh_M[4]*y + sh_M[5]*z + half_y;

    const float sdx[4] = { 0.25f,  0.25f, -0.25f, -0.25f };
    const float sdz[4] = { 0.25f, -0.25f,  0.25f, -0.25f };

    float val = 0.0f;

    for (int s = 0; s < 4; s++) {
        float u_s = u0 + sh_M[0]*sdx[s] + sh_M[2]*sdz[s];
        float v_s = v0 + sh_M[3]*sdx[s] + sh_M[5]*sdz[s];

        /* Skip if projected position is entirely outside the sinogram */
        if (u_s < -0.5f || u_s > N_det - 0.5f || v_s < -0.5f || v_s > Ny - 0.5f) continue;

        /* Clamp stencil to valid grid — handles Ny=1 and boundary voxels */
        int u1 = max(0, (int)floorf(u_s));  int u2 = min(N_det - 1, u1 + 1);
        int v1 = max(0, (int)floorf(v_s));  int v2 = min(Ny   - 1, v1 + 1);

        float wu2 = u_s - u1,  wu1 = 1.0f - wu2;
        float wv2 = v_s - v1,  wv1 = 1.0f - wv2;

        val += wu1*wv1 * sino_slice[v1*N_det + u1]
             + wu2*wv1 * sino_slice[v1*N_det + u2]
             + wu1*wv2 * sino_slice[v2*N_det + u1]
             + wu2*wv2 * sino_slice[v2*N_det + u2];
    }

    volume[((size_t)iy * Nz + iz) * Nx + ix] += val;
}

void forward_project_resire_cuda(
    const float *d_volume,
    float       *d_sinogram,
    const float *d_matrices,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det
) {
    float half_det = (N_det - 1) * 0.5f;
    float half_x   = (Nx   - 1) * 0.5f;
    float half_z   = (Nz   - 1) * 0.5f;
    float half_y   = (Ny   - 1) * 0.5f;

    dim3 block(16, 16, 1);
    dim3 grid((Nx + 15) / 16, (Nz + 15) / 16, Ny);

    for (int ia = 0; ia < n_angles; ia++) {
        forward_project_resire_kernel<<<grid, block>>>(
            d_volume,
            d_sinogram + (size_t)ia * Ny * N_det,
            d_matrices + ia * 9,
            Ny, Nz, Nx, N_det,
            half_x, half_y, half_z, half_det
        );
    }
    cudaDeviceSynchronize();
}

void back_project_resire_cuda(
    const float *d_sinogram,
    float       *d_volume,
    const float *d_matrices,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det
) {
    float half_det = (N_det - 1) * 0.5f;
    float half_x   = (Nx   - 1) * 0.5f;
    float half_z   = (Nz   - 1) * 0.5f;
    float half_y   = (Ny   - 1) * 0.5f;

    dim3 block(16, 16, 1);
    dim3 grid((Nx + 15) / 16, (Nz + 15) / 16, Ny);

    for (int ia = 0; ia < n_angles; ia++) {
        back_project_resire_kernel<<<grid, block>>>(
            d_sinogram + (size_t)ia * Ny * N_det,
            d_volume,
            d_matrices + ia * 9,
            Ny, Nz, Nx, N_det,
            half_x, half_y, half_z, half_det
        );
    }
    cudaDeviceSynchronize();
}

/* ── GENFIRE 2D Fourier gridding ─────────────────────────────────────────── */

__global__ static void genfire_gridding_2d_kernel(
    const float *fk_real, const float *fk_imag,
    const float *cos_a,   const float *sin_a,
    float *grid_real, float *grid_imag, float *weight_grid,
    int bigN, int Ny, int n_angles, float cutoff_dist
) {
    int ixi = blockIdx.x * blockDim.x + threadIdx.x;
    int ia  = (int)blockIdx.y;
    int iy  = (int)blockIdx.z;

    if (ixi >= bigN || ia >= n_angles || iy >= Ny) return;

    int   half = bigN / 2;
    float xi   = (float)(ixi - half);
    float kx   = xi * cos_a[ia];
    float kz   = xi * sin_a[ia];
    int   ikx  = __float2int_rn(kx);
    int   ikz  = __float2int_rn(kz);
    float dx   = kx - (float)ikx;
    float dz   = kz - (float)ikz;
    float dist = sqrtf(dx * dx + dz * dz);
    if (dist > cutoff_dist) return;

    int gx = ikx + half;
    int gz = ikz + half;
    if (gx < 0 || gx >= bigN || gz < 0 || gz >= bigN) return;

    float  w       = 1.0f / (dist + 1e-5f);
    size_t fk_idx  = ((size_t)ia * Ny + iy) * bigN + (size_t)ixi;
    size_t out_idx = (size_t)iy * bigN * bigN + (size_t)gz * bigN + (size_t)gx;

    atomicAdd(&grid_real[out_idx],    w * fk_real[fk_idx]);
    atomicAdd(&grid_imag[out_idx],    w * fk_imag[fk_idx]);
    atomicAdd(&weight_grid[out_idx],  w);
}

void genfire_gridding_2d_cuda(
    const float *fk_real, const float *fk_imag,
    const float *cos_a,   const float *sin_a,
    float *grid_real, float *grid_imag, float *weight_grid,
    int bigN, int Ny, int n_angles, float cutoff_dist
) {
    dim3 block(128, 1, 1);
    dim3 grid((bigN + 127) / 128, (unsigned)n_angles, (unsigned)Ny);
    genfire_gridding_2d_kernel<<<grid, block>>>(
        fk_real, fk_imag, cos_a, sin_a,
        grid_real, grid_imag, weight_grid,
        bigN, Ny, n_angles, cutoff_dist
    );
    cudaDeviceSynchronize();
}

/* ── gradient of forward projection w.r.t. tilt angles ───────────────────
 * Grid:  (ceil(N_det/16), ceil(n_angles/16), Ny)   Block: (16, 16, 1)
 * Same detector-driven layout as forward_project_kernel.
 * Each thread (id, ia, iy) walks the ray and accumulates:
 *   local_g += gs * (dvol_dx * (-z) + dvol_dz * x)
 * where dvol_dx/dvol_dz are bilinear finite-differences of the volume.
 * In-block reduction over threadIdx.x collapses (detector) dimension;
 * then one atomicAdd per block row writes to grad_angles[ia].
 */
__global__ static void grad_angle_kernel(
    const float *__restrict__ volume,     /* [Ny, Nz, Nx] */
    const float *__restrict__ grad_sino,  /* [n_angles, Ny, N_det] */
    const float *__restrict__ cos_a,
    const float *__restrict__ sin_a,
    float       *__restrict__ grad_angles,/* [n_angles] */
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    float half_det, float half_x, float half_z, int ray_len
) {
    __shared__ float sh[16 * 16];

    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int ia = blockIdx.y * blockDim.y + threadIdx.y;
    int iy = (int)blockIdx.z;

    float local_g = 0.0f;

    if (id < N_det && ia < n_angles) {
        float ca = cos_a[ia], sa = sin_a[ia];
        float t   = id - half_det;
        float gs  = grad_sino[((size_t)ia * Ny + iy) * N_det + id];

        if (gs != 0.0f) {
            const float *slice = volume + (size_t)iy * Nz * Nx;

            for (int step = -ray_len; step <= ray_len; step++) {
                float s  = (float)step;
                float x  = t * ca - s * sa;
                float z  = t * sa + s * ca;
                float fx = x + half_x;
                float fz = z + half_z;

                if (fx < 0.0f || fx >= (float)(Nx - 1) ||
                    fz < 0.0f || fz >= (float)(Nz - 1))
                    continue;

                int ix0 = (int)fx, iz0 = (int)fz;
                int ix1 = ix0 + 1, iz1 = iz0 + 1;
                float wx = fx - ix0, wz = fz - iz0;

                float v00 = slice[(size_t)iz0 * Nx + ix0];
                float v01 = slice[(size_t)iz0 * Nx + ix1];
                float v10 = slice[(size_t)iz1 * Nx + ix0];
                float v11 = slice[(size_t)iz1 * Nx + ix1];

                float dvol_dx = (1.0f - wz) * (v01 - v00) + wz * (v11 - v10);
                float dvol_dz = (1.0f - wx) * (v10 - v00) + wx * (v11 - v01);

                local_g += gs * (dvol_dx * (-z) + dvol_dz * x);
            }
        }
    }

    sh[threadIdx.y * 16 + threadIdx.x] = local_g;
    __syncthreads();

    /* Reduce over threadIdx.x (all detector cells sharing the same ia). */
    for (int stride = 8; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            sh[threadIdx.y * 16 + threadIdx.x] += sh[threadIdx.y * 16 + threadIdx.x + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0 && ia < n_angles)
        atomicAdd(&grad_angles[ia], sh[threadIdx.y * 16]);
}

void grad_angle_cuda(
    const float *d_volume,
    const float *d_grad_sino,
    const float *d_cos_a,
    const float *d_sin_a,
    float       *d_grad_angles,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det
) {
    float half_det = (N_det - 1) * 0.5f;
    float half_x   = (Nx   - 1) * 0.5f;
    float half_z   = (Nz   - 1) * 0.5f;
    int   ray_len  = (int)ceilf(0.5f * sqrtf((float)(Nx * Nx + Nz * Nz))) + 1;

    dim3 block(16, 16, 1);
    dim3 grid((N_det    + 15) / 16,
              (n_angles + 15) / 16,
              Ny);

    grad_angle_kernel<<<grid, block>>>(
        d_volume, d_grad_sino, d_cos_a, d_sin_a, d_grad_angles,
        Ny, Nz, Nx, n_angles, N_det,
        half_det, half_x, half_z, ray_len
    );
    cudaDeviceSynchronize();
}

/* ── gradient of RESIRE forward projection w.r.t. rotation matrices ──────
 * Grid:  (ceil(Nx/16), ceil(Nz/16), Ny)   Block: (16, 16, 1)
 * Per voxel: 4 supersamples with same boundary logic as the forward kernel.
 * Block reduction via __shared__ float s_gM[6]; per-block atomicAdd to global.
 */
__global__ static void grad_rotmat_resire_kernel(
    const float *__restrict__ volume,     /* [Ny, Nz, Nx] */
    const float *__restrict__ grad_sino,  /* [Ny, N_det] — one angle slice */
    const float *__restrict__ M,          /* [9] rotation matrix */
    float       *__restrict__ grad_M,     /* [9] output — must be zeroed */
    int Ny, int Nz, int Nx, int N_det,
    float half_x, float half_y, float half_z, float half_det
) {
    __shared__ float sh_M[9];
    __shared__ float s_gM[6];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (tid < 9)  sh_M[tid] = M[tid];
    if (tid < 6)  s_gM[tid] = 0.0f;
    __syncthreads();

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int iy = (int)blockIdx.z;

    float local_gM[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    if (ix < Nx && iz < Nz) {
        float x = ix - half_x;
        float y = iy - half_y;
        float z = iz - half_z;

        float u0 = sh_M[0]*x + sh_M[1]*y + sh_M[2]*z + half_det;
        float v0 = sh_M[3]*x + sh_M[4]*y + sh_M[5]*z + half_y;

        const float data_val = volume[((size_t)iy * Nz + iz) * Nx + ix];

        const float sdx[4] = { 0.25f,  0.25f, -0.25f, -0.25f };
        const float sdz[4] = { 0.25f, -0.25f,  0.25f, -0.25f };

        for (int s = 0; s < 4; s++) {
            float cx = x + sdx[s];
            float cz = z + sdz[s];
            float u_s = u0 + sh_M[0]*sdx[s] + sh_M[2]*sdz[s];
            float v_s = v0 + sh_M[3]*sdx[s] + sh_M[5]*sdz[s];

            int u1 = (int)floorf(u_s);  int u2 = u1 + 1;
            int v1 = (int)floorf(v_s);  int v2 = v1 + 1;

            /* Mirror FP kernel: skip if stencil entirely off-detector */
            if ((u1 < 0 && u2 < 0) || (u1 >= N_det && u2 >= N_det)) continue;
            if ((v1 < 0 && v2 < 0) || (v1 >= Ny   && v2 >= Ny  )) continue;

            float wu2 = u_s - u1,  wu1 = 1.0f - wu2;
            float wv2 = v_s - v1,  wv1 = 1.0f - wv2;

            bool u1ok = (u1 >= 0 && u1 < N_det);
            bool u2ok = (u2 >= 0 && u2 < N_det);
            bool v1ok = (v1 >= 0 && v1 < Ny);
            bool v2ok = (v2 >= 0 && v2 < Ny);

            float dL_du = 0.0f, dL_dv = 0.0f;
            if (u1ok && v1ok) { float gs = grad_sino[(size_t)v1*N_det+u1]; dL_du -= wv1*gs; dL_dv -= wu1*gs; }
            if (u2ok && v1ok) { float gs = grad_sino[(size_t)v1*N_det+u2]; dL_du += wv1*gs; dL_dv -= wu2*gs; }
            if (u1ok && v2ok) { float gs = grad_sino[(size_t)v2*N_det+u1]; dL_du -= wv2*gs; dL_dv += wu1*gs; }
            if (u2ok && v2ok) { float gs = grad_sino[(size_t)v2*N_det+u2]; dL_du += wv2*gs; dL_dv += wu2*gs; }
            dL_du *= data_val;
            dL_dv *= data_val;

            local_gM[0] += dL_du * cx;
            local_gM[1] += dL_du * y;
            local_gM[2] += dL_du * cz;
            local_gM[3] += dL_dv * cx;
            local_gM[4] += dL_dv * y;
            local_gM[5] += dL_dv * cz;
        }
    }

    for (int k = 0; k < 6; k++)
        atomicAdd(&s_gM[k], local_gM[k]);
    __syncthreads();

    if (tid < 6)
        atomicAdd(&grad_M[tid], s_gM[tid]);
}

void grad_rotmat_resire_cuda(
    const float *d_volume,
    const float *d_grad_sinogram,
    const float *d_matrices,
    float       *d_grad_matrices,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det
) {
    float half_det = (N_det - 1) * 0.5f;
    float half_x   = (Nx   - 1) * 0.5f;
    float half_z   = (Nz   - 1) * 0.5f;
    float half_y   = (Ny   - 1) * 0.5f;

    dim3 block(16, 16, 1);
    dim3 grid((Nx + 15) / 16, (Nz + 15) / 16, Ny);

    for (int ia = 0; ia < n_angles; ia++) {
        grad_rotmat_resire_kernel<<<grid, block>>>(
            d_volume,
            d_grad_sinogram + (size_t)ia * Ny * N_det,
            d_matrices + ia * 9,
            d_grad_matrices + ia * 9,
            Ny, Nz, Nx, N_det,
            half_x, half_y, half_z, half_det
        );
    }
    cudaDeviceSynchronize();
}

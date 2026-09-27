/* ── fasttomo/csrc/cuda/projection_resire.cu ─────────────────────────────────
 * RESIRE voxel-driven scatter/gather projectors.
 * Kernels: forward_project_resire_kernel, back_project_resire_kernel.
 * Host wrappers: forward_project_resire_cuda, back_project_resire_cuda.
 *
 * Each launch processes ONE projection. The host loops over n_angles.
 * Supersampling: 4 sub-voxel offsets (±0.25, ±0.25) in volume XZ space are
 * rotated to detector coordinates via rows 0–1 of the projection matrix M.
 * The Python wrapper (fasttomo/autograd.py) divides the output by 4 to
 * normalise to the single-sample scale; the C++ host does not.
 */

#include "common.cuh"
#include "texture_cache.h"
#include "ops.h"

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

    float u0 = sh_M[0]*x + sh_M[1]*y + sh_M[2]*z + half_det;
    float v0 = sh_M[3]*x + sh_M[4]*y + sh_M[5]*z + half_y;

    const float data_val = volume[((size_t)iy * Nz + iz) * Nx + ix];

    const float sdx[4] = { 0.25f,  0.25f, -0.25f, -0.25f };
    const float sdz[4] = { 0.25f, -0.25f,  0.25f, -0.25f };

    for (int s = 0; s < 4; s++) {
        float u_s = u0 + sh_M[0]*sdx[s] + sh_M[2]*sdz[s];
        float v_s = v0 + sh_M[3]*sdx[s] + sh_M[5]*sdz[s];

        int u1 = (int)floorf(u_s);  int u2 = u1 + 1;
        int v1 = (int)floorf(v_s);  int v2 = v1 + 1;

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

        if (u_s < -0.5f || u_s > N_det - 0.5f || v_s < -0.5f || v_s > Ny - 0.5f) continue;

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
    int n_angles, int N_det,
    cudaStream_t stream
) {
    float half_det = (N_det - 1) * 0.5f;
    float half_x   = (Nx   - 1) * 0.5f;
    float half_z   = (Nz   - 1) * 0.5f;
    float half_y   = (Ny   - 1) * 0.5f;

    dim3 block(16, 16, 1);
    dim3 grid((Nx + 15) / 16, (Nz + 15) / 16, Ny);

    for (int ia = 0; ia < n_angles; ia++) {
        forward_project_resire_kernel<<<grid, block, 0, stream>>>(
            d_volume,
            d_sinogram + (size_t)ia * Ny * N_det,
            d_matrices + ia * 9,
            Ny, Nz, Nx, N_det,
            half_x, half_y, half_z, half_det
        );
    }
    CUDA_CHECK_KERNEL();
}

void back_project_resire_cuda(
    const float *d_sinogram,
    float       *d_volume,
    const float *d_matrices,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det,
    cudaStream_t stream
) {
    float half_det = (N_det - 1) * 0.5f;
    float half_x   = (Nx   - 1) * 0.5f;
    float half_z   = (Nz   - 1) * 0.5f;
    float half_y   = (Ny   - 1) * 0.5f;

    dim3 block(16, 16, 1);
    dim3 grid((Nx + 15) / 16, (Nz + 15) / 16, Ny);

    for (int ia = 0; ia < n_angles; ia++) {
        back_project_resire_kernel<<<grid, block, 0, stream>>>(
            d_sinogram + (size_t)ia * Ny * N_det,
            d_volume,
            d_matrices + ia * 9,
            Ny, Nz, Nx, N_det,
            half_x, half_y, half_z, half_det
        );
    }
    CUDA_CHECK_KERNEL();
}

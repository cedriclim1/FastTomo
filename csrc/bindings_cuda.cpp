#ifdef FASTTOMO_CUDA

#include <pybind11/pybind11.h>
#include <stdexcept>

#include "cuda/ops.h"
#include "cuda/texture_cache.h"
#include <cuda_runtime.h>
#include "bindings_cuda.h"

namespace py = pybind11;

/* Cast a Python int (caller passes torch.cuda.current_stream().cuda_stream)
 * to cudaStream_t. 0 means the legacy default stream. */
static inline cudaStream_t to_stream(long stream_ptr) {
    return reinterpret_cast<cudaStream_t>(stream_ptr);
}

/* ── CUDA projection bindings ─────────────────────────────────────────── */

static void py_forward_project_parallel_cuda(
    int io_dtype,
    long vol_ptr, long sino_ptr,
    long cos_ptr, long sin_ptr,
    int Ny, int Nz, int Nx, int n_angles, int N_det,
    long stream_ptr, long nonfinite_ptr
) {
    forward_project_parallel_cuda(
        io_dtype,
        reinterpret_cast<const void *>(vol_ptr),
        reinterpret_cast<void *>(sino_ptr),
        reinterpret_cast<const float *>(cos_ptr),
        reinterpret_cast<const float *>(sin_ptr),
        Ny, Nz, Nx, n_angles, N_det,
        to_stream(stream_ptr),
        reinterpret_cast<int *>(nonfinite_ptr)
    );
}

static void py_back_project_parallel_cuda(
    int io_dtype,
    long sino_ptr, long vol_ptr,
    long cos_ptr, long sin_ptr,
    int Ny, int Nz, int Nx, int n_angles, int N_det,
    long stream_ptr, long nonfinite_ptr
) {
    back_project_parallel_cuda(
        io_dtype,
        reinterpret_cast<const void *>(sino_ptr),
        reinterpret_cast<void *>(vol_ptr),
        reinterpret_cast<const float *>(cos_ptr),
        reinterpret_cast<const float *>(sin_ptr),
        Ny, Nz, Nx, n_angles, N_det,
        to_stream(stream_ptr),
        reinterpret_cast<int *>(nonfinite_ptr)
    );
}

static void py_forward_project_rotmat_cuda(
    int io_dtype,
    long vol_ptr, long sino_ptr, long mats_ptr,
    int Ny, int Nz, int Nx, int n_angles, int N_det,
    long stream_ptr
) {
    forward_project_rotmat_cuda(
        io_dtype,
        reinterpret_cast<const void *>(vol_ptr),
        reinterpret_cast<void *>(sino_ptr),
        reinterpret_cast<const float *>(mats_ptr),
        Ny, Nz, Nx, n_angles, N_det,
        to_stream(stream_ptr)
    );
}

static void py_back_project_rotmat_cuda(
    int io_dtype,
    long sino_ptr, long vol_ptr, long mats_ptr,
    int Ny, int Nz, int Nx, int n_angles, int N_det,
    long stream_ptr
) {
    back_project_rotmat_cuda(
        io_dtype,
        reinterpret_cast<const void *>(sino_ptr),
        reinterpret_cast<void *>(vol_ptr),
        reinterpret_cast<const float *>(mats_ptr),
        Ny, Nz, Nx, n_angles, N_det,
        to_stream(stream_ptr)
    );
}

static void py_forward_project_resire_cuda(
    long vol_ptr, long sino_ptr, long mats_ptr,
    int Ny, int Nz, int Nx, int n_angles, int N_det,
    long stream_ptr
) {
    forward_project_resire_cuda(
        reinterpret_cast<const float *>(vol_ptr),
        reinterpret_cast<float *>(sino_ptr),
        reinterpret_cast<const float *>(mats_ptr),
        Ny, Nz, Nx, n_angles, N_det,
        to_stream(stream_ptr)
    );
}

static void py_back_project_resire_cuda(
    long sino_ptr, long vol_ptr, long mats_ptr,
    int Ny, int Nz, int Nx, int n_angles, int N_det,
    long stream_ptr
) {
    back_project_resire_cuda(
        reinterpret_cast<const float *>(sino_ptr),
        reinterpret_cast<float *>(vol_ptr),
        reinterpret_cast<const float *>(mats_ptr),
        Ny, Nz, Nx, n_angles, N_det,
        to_stream(stream_ptr)
    );
}

static void py_grad_rotmat_cuda(
    long vol_ptr, long grad_sino_ptr,
    long mats_ptr, long grad_mats_ptr,
    int Ny, int Nz, int Nx, int n_angles, int N_det,
    long stream_ptr
) {
    grad_rotmat_cuda(
        reinterpret_cast<const float *>(vol_ptr),
        reinterpret_cast<const float *>(grad_sino_ptr),
        reinterpret_cast<const float *>(mats_ptr),
        reinterpret_cast<float *>(grad_mats_ptr),
        Ny, Nz, Nx, n_angles, N_det,
        to_stream(stream_ptr)
    );
}

static void py_grad_rotmat_cuda_cached(
    int io_dtype,
    long cache_ptr,
    long grad_sino_ptr, long mats_ptr, long grad_mats_ptr,
    int Ny, int Nz, int Nx, int n_angles, int N_det,
    long stream_ptr
) {
    grad_rotmat_cuda_cached(
        io_dtype,
        reinterpret_cast<TextureCache *>(cache_ptr),
        reinterpret_cast<const float *>(grad_sino_ptr),
        reinterpret_cast<const float *>(mats_ptr),
        reinterpret_cast<float *>(grad_mats_ptr),
        Ny, Nz, Nx, n_angles, N_det,
        to_stream(stream_ptr)
    );
}

static void py_grad_angle_cuda(
    long vol_ptr, long grad_sino_ptr,
    long cos_ptr, long sin_ptr,
    long grad_angles_ptr,
    int Ny, int Nz, int Nx, int n_angles, int N_det,
    long stream_ptr
) {
    grad_angle_cuda(
        reinterpret_cast<const float *>(vol_ptr),
        reinterpret_cast<const float *>(grad_sino_ptr),
        reinterpret_cast<const float *>(cos_ptr),
        reinterpret_cast<const float *>(sin_ptr),
        reinterpret_cast<float *>(grad_angles_ptr),
        Ny, Nz, Nx, n_angles, N_det,
        to_stream(stream_ptr)
    );
}

/* ── texture cache (opaque pointer exposed as Python int) ──────────────── */

static long py_tc_create(int kind, int dtype, int channels,
                         int dim_a, int dim_b, int dim_c, bool surface) {
    if (kind < 0 || kind > 2)
        throw std::invalid_argument("tc_create: kind must be 0..2");
    if (dtype < 0 || dtype > 1)
        throw std::invalid_argument("tc_create: dtype must be 0 (fp32) or 1 (fp16)");
    if (channels != 1 && channels != 2 && channels != 4)
        throw std::invalid_argument("tc_create: channels must be 1, 2 or 4");
    TextureCache *tc = tc_create_ex(
        static_cast<tc_kind_t>(kind),
        static_cast<tc_dtype_t>(dtype),
        channels,
        dim_a, dim_b, dim_c, surface ? 1 : 0);
    return reinterpret_cast<long>(tc);
}

static bool py_tc_has_surface(long handle) {
    const TextureCache *tc = reinterpret_cast<const TextureCache *>(handle);
    if (!tc) throw std::invalid_argument("tc_has_surface: null handle");
    return tc->surf != 0;
}

static void py_tc_destroy(long handle) {
    /* A persistent cache may have been read by kernels on any stream of the
     * current device; wait for all of them before its textures and array are
     * released (P1). cudaFreeArray would synchronize anyway. Release even if
     * the sync reports an error, then surface the first error. */
    cudaError_t sync_err = cudaDeviceSynchronize();
    cudaError_t rel_err  = tc_release(reinterpret_cast<TextureCache *>(handle));
    FASTTOMO_CUDA_CHECK(sync_err);
    FASTTOMO_CUDA_CHECK(rel_err);
}

/* ── cached FP/BP entry points ─────────────────────────────────────────── */

static void py_forward_project_parallel_cuda_cached(
    int io_dtype,
    long cache_ptr,
    long vol_ptr, long sino_ptr,
    long cos_ptr, long sin_ptr,
    int Ny, int Nz, int Nx, int n_angles, int N_det,
    long stream_ptr, bool upload, long nonfinite_ptr
) {
    forward_project_parallel_cuda_cached(
        io_dtype,
        reinterpret_cast<TextureCache *>(cache_ptr),
        reinterpret_cast<const void *>(vol_ptr),
        reinterpret_cast<void *>(sino_ptr),
        reinterpret_cast<const float *>(cos_ptr),
        reinterpret_cast<const float *>(sin_ptr),
        Ny, Nz, Nx, n_angles, N_det,
        to_stream(stream_ptr), upload ? 1 : 0,
        reinterpret_cast<int *>(nonfinite_ptr)
    );
}

static void py_back_project_parallel_cuda_cached(
    int io_dtype,
    long cache_ptr,
    long sino_ptr, long vol_ptr,
    long cos_ptr, long sin_ptr,
    int Ny, int Nz, int Nx, int n_angles, int N_det,
    long stream_ptr, bool upload, long nonfinite_ptr
) {
    back_project_parallel_cuda_cached(
        io_dtype,
        reinterpret_cast<TextureCache *>(cache_ptr),
        reinterpret_cast<const void *>(sino_ptr),
        reinterpret_cast<void *>(vol_ptr),
        reinterpret_cast<const float *>(cos_ptr),
        reinterpret_cast<const float *>(sin_ptr),
        Ny, Nz, Nx, n_angles, N_det,
        to_stream(stream_ptr), upload ? 1 : 0,
        reinterpret_cast<int *>(nonfinite_ptr)
    );
}

static void py_forward_project_rotmat_cuda_cached(
    int io_dtype,
    long cache_ptr,
    long vol_ptr, long sino_ptr, long mats_ptr,
    int Ny, int Nz, int Nx, int n_angles, int N_det,
    long stream_ptr, bool upload
) {
    forward_project_rotmat_cuda_cached(
        io_dtype,
        reinterpret_cast<TextureCache *>(cache_ptr),
        reinterpret_cast<const void *>(vol_ptr),
        reinterpret_cast<void *>(sino_ptr),
        reinterpret_cast<const float *>(mats_ptr),
        Ny, Nz, Nx, n_angles, N_det,
        to_stream(stream_ptr), upload ? 1 : 0
    );
}

static void py_back_project_rotmat_cuda_cached(
    int io_dtype,
    long cache_ptr,
    long sino_ptr, long vol_ptr, long mats_ptr,
    int Ny, int Nz, int Nx, int n_angles, int N_det,
    long stream_ptr, bool upload
) {
    back_project_rotmat_cuda_cached(
        io_dtype,
        reinterpret_cast<TextureCache *>(cache_ptr),
        reinterpret_cast<const void *>(sino_ptr),
        reinterpret_cast<void *>(vol_ptr),
        reinterpret_cast<const float *>(mats_ptr),
        Ny, Nz, Nx, n_angles, N_det,
        to_stream(stream_ptr), upload ? 1 : 0
    );
}

static void py_tv_loss_iso_3d_cuda(
    long vol_ptr, long acc_ptr, long partials_ptr,
    int Ny, int Nz, int Nx,
    float eps,
    long stream_ptr
) {
    tv_loss_iso_3d_cuda(
        reinterpret_cast<const float *>(vol_ptr),
        reinterpret_cast<float *>(acc_ptr),
        reinterpret_cast<double *>(partials_ptr),
        Ny, Nz, Nx, eps,
        to_stream(stream_ptr)
    );
}

static void py_tv_loss_iso_3d_grad_cuda(
    long vol_ptr, long g_scaled_ptr, long grad_vol_ptr,
    int Ny, int Nz, int Nx,
    float eps,
    long stream_ptr
) {
    tv_loss_iso_3d_grad_cuda(
        reinterpret_cast<const float *>(vol_ptr),
        reinterpret_cast<const float *>(g_scaled_ptr),
        reinterpret_cast<float *>(grad_vol_ptr),
        Ny, Nz, Nx, eps,
        to_stream(stream_ptr)
    );
}

static void py_tv_loss_xz_cuda(
    long vol_ptr, long acc_ptr, long partials_ptr,
    int Ny, int Nz, int Nx,
    float eps,
    long stream_ptr
) {
    tv_loss_xz_cuda(
        reinterpret_cast<const float *>(vol_ptr),
        reinterpret_cast<float *>(acc_ptr),
        reinterpret_cast<double *>(partials_ptr),
        Ny, Nz, Nx, eps,
        to_stream(stream_ptr)
    );
}

static void py_tv_loss_xz_grad_cuda(
    long vol_ptr, long g_scaled_ptr, long grad_vol_ptr,
    int Ny, int Nz, int Nx,
    float eps,
    long stream_ptr
) {
    tv_loss_xz_grad_cuda(
        reinterpret_cast<const float *>(vol_ptr),
        reinterpret_cast<const float *>(g_scaled_ptr),
        reinterpret_cast<float *>(grad_vol_ptr),
        Ny, Nz, Nx, eps,
        to_stream(stream_ptr)
    );
}

static void py_sirt_residual_cuda(
    int io_dtype,
    long sino_ptr, long fp_ptr, long R_ptr,
    long raw_out_ptr, long res_out_ptr,
    long long n,
    long stream_ptr
) {
    sirt_residual_cuda(
        io_dtype,
        reinterpret_cast<const void *>(sino_ptr),
        reinterpret_cast<const void *>(fp_ptr),
        reinterpret_cast<const void *>(R_ptr),
        reinterpret_cast<void *>(raw_out_ptr),
        reinterpret_cast<void *>(res_out_ptr),
        n, to_stream(stream_ptr));
}

static void py_sirt_update_cuda(
    int io_dtype, int eager_order,
    long vol_in_ptr, long vol_out_ptr,
    long upd_ptr, long C_ptr,
    long mask_ptr, int mask_full,
    float omega, int positivity, float soft_thr,
    int Ny, int Nz, int Nx,
    long stream_ptr, long vol_cache_ptr
) {
    sirt_update_cuda(
        io_dtype, eager_order,
        reinterpret_cast<const void *>(vol_in_ptr),
        reinterpret_cast<void *>(vol_out_ptr),
        reinterpret_cast<const void *>(upd_ptr),
        reinterpret_cast<const void *>(C_ptr),
        reinterpret_cast<const void *>(mask_ptr), mask_full,
        omega, positivity, soft_thr, Ny, Nz, Nx,
        reinterpret_cast<TextureCache *>(vol_cache_ptr),
        to_stream(stream_ptr));
}

static void py_sirt_residual_tex_cuda(
    int io_dtype,
    long sino_ptr, long fp_ptr, long R_ptr, long raw_out_ptr,
    long cache_ptr,
    int n_angles, int Ny, int N_det,
    long stream_ptr
) {
    sirt_residual_tex_cuda(
        io_dtype,
        reinterpret_cast<const void *>(sino_ptr),
        reinterpret_cast<const void *>(fp_ptr),
        reinterpret_cast<const void *>(R_ptr),
        reinterpret_cast<void *>(raw_out_ptr),
        reinterpret_cast<TextureCache *>(cache_ptr),
        n_angles, Ny, N_det, to_stream(stream_ptr));
}

static void py_tc_upload(long handle, long src_ptr, long stream_ptr) {
    tc_upload(reinterpret_cast<TextureCache *>(handle),
              reinterpret_cast<const void *>(src_ptr), to_stream(stream_ptr));
}

static void py_tc_download(long handle, long dst_ptr, long stream_ptr) {
    tc_download(reinterpret_cast<const TextureCache *>(handle),
                reinterpret_cast<void *>(dst_ptr), to_stream(stream_ptr));
}

static void py_genfire_gridding_2d_cuda(
    long fk_real_ptr, long fk_imag_ptr,
    long cos_ptr,     long sin_ptr,
    long grid_real_ptr, long grid_imag_ptr, long weight_ptr,
    int bigN, int Ny, int n_angles, float cutoff_dist,
    long stream_ptr
) {
    genfire_gridding_2d_cuda(
        reinterpret_cast<const float *>(fk_real_ptr),
        reinterpret_cast<const float *>(fk_imag_ptr),
        reinterpret_cast<const float *>(cos_ptr),
        reinterpret_cast<const float *>(sin_ptr),
        reinterpret_cast<float *>(grid_real_ptr),
        reinterpret_cast<float *>(grid_imag_ptr),
        reinterpret_cast<float *>(weight_ptr),
        bigN, Ny, n_angles, cutoff_dist,
        to_stream(stream_ptr)
    );
}

/* ── MBIR (CUDA) bindings ─────────────────────────────────────────────── */

static void py_mbir_forward_project_cuda(
    long vol_ptr, long sino_ptr,
    long cos_ptr, long sin_ptr,
    int Ny, int Nz, int Nx, int n_angles, int N_det,
    long stream_ptr
) {
    mbir_forward_project_cuda(
        reinterpret_cast<const float *>(vol_ptr),
        reinterpret_cast<float *>(sino_ptr),
        reinterpret_cast<const float *>(cos_ptr),
        reinterpret_cast<const float *>(sin_ptr),
        Ny, Nz, Nx, n_angles, N_det,
        to_stream(stream_ptr)
    );
}

static void py_mbir_icd_sweep_cuda(
    long vol_ptr, long e_sino_ptr,
    long vol_snapshot_ptr,
    long lambdas_ptr,          /* 0 → nullptr */
    long perm_ptr,             /* 0 → in-kernel order from perm_key */
    unsigned long long perm_key,
    long work_ptr,
    long cos_ptr, long sin_ptr,
    int Ny, int Nz, int Nx, int n_angles, int N_det,
    float qggmrf_p, float qggmrf_q, float qggmrf_c, float sigma_f,
    float huber_T, float huber_delta, float sigma,
    long stream_ptr, int group_spacing
) {
    mbir_icd_sweep_cuda(
        reinterpret_cast<float *>(vol_ptr),
        reinterpret_cast<float *>(e_sino_ptr),
        reinterpret_cast<const float *>(vol_snapshot_ptr),
        reinterpret_cast<const float *>(lambdas_ptr),
        reinterpret_cast<const int *>(perm_ptr),
        perm_key,
        reinterpret_cast<float *>(work_ptr),
        reinterpret_cast<const float *>(cos_ptr),
        reinterpret_cast<const float *>(sin_ptr),
        Ny, Nz, Nx, n_angles, N_det,
        qggmrf_p, qggmrf_q, qggmrf_c, sigma_f,
        huber_T, huber_delta, sigma, group_spacing,
        to_stream(stream_ptr)
    );
}

static void py_mbir_permutation_cuda(
    long out_ptr, int Ny, int nvox, unsigned long long perm_key, long stream_ptr
) {
    mbir_permutation_cuda(reinterpret_cast<int *>(out_ptr), Ny, nvox, perm_key,
                          to_stream(stream_ptr));
}

void register_cuda(py::module_ &m) {
    m.def("forward_project_parallel_cuda", &py_forward_project_parallel_cuda,
          py::call_guard<py::gil_scoped_release>(),
          "Forward projection (CUDA). io_dtype: 0=fp32, 1=fp16. "
          "All ptr args are device data_ptr() ints. "
          "stream_ptr: torch.cuda.Stream.cuda_stream (0 → default stream). "
          "nonfinite_ptr: optional device int32, fp16 stores OR in 1 (inf) / 2 (NaN).",
          py::arg("io_dtype"),
          py::arg("vol_ptr"), py::arg("sino_ptr"),
          py::arg("cos_ptr"),    py::arg("sin_ptr"),
          py::arg("Ny"), py::arg("Nz"), py::arg("Nx"),
          py::arg("n_angles"), py::arg("N_det"),
          py::arg("stream_ptr") = 0L, py::arg("nonfinite_ptr") = 0L);

    m.def("back_project_parallel_cuda", &py_back_project_parallel_cuda,
          py::call_guard<py::gil_scoped_release>(),
          "Back-projection (CUDA). io_dtype: 0=fp32, 1=fp16. "
          "All ptr args are device data_ptr() ints. nonfinite_ptr: optional "
          "device int32, fp16 stores OR in 1 (inf) / 2 (NaN).",
          py::arg("io_dtype"),
          py::arg("sino_ptr"), py::arg("vol_ptr"),
          py::arg("cos_ptr"),      py::arg("sin_ptr"),
          py::arg("Ny"), py::arg("Nz"), py::arg("Nx"),
          py::arg("n_angles"), py::arg("N_det"),
          py::arg("stream_ptr") = 0L, py::arg("nonfinite_ptr") = 0L);

    m.def("forward_project_rotmat_cuda", &py_forward_project_rotmat_cuda,
          py::call_guard<py::gil_scoped_release>(),
          "3D rotation-matrix forward projection (CUDA). io_dtype: 0=fp32, 1=fp16. "
          "mats_ptr: device data_ptr() of [n_angles*9] row-major float32 rotation matrices.",
          py::arg("io_dtype"),
          py::arg("vol_ptr"), py::arg("sino_ptr"), py::arg("mats_ptr"),
          py::arg("Ny"), py::arg("Nz"), py::arg("Nx"),
          py::arg("n_angles"), py::arg("N_det"),
          py::arg("stream_ptr") = 0L);

    m.def("back_project_rotmat_cuda", &py_back_project_rotmat_cuda,
          py::call_guard<py::gil_scoped_release>(),
          "3D rotation-matrix back-projection (CUDA). io_dtype: 0=fp32, 1=fp16. "
          "mats_ptr: device data_ptr() of [n_angles*9] row-major float32 rotation matrices.",
          py::arg("io_dtype"),
          py::arg("sino_ptr"), py::arg("vol_ptr"), py::arg("mats_ptr"),
          py::arg("Ny"), py::arg("Nz"), py::arg("Nx"),
          py::arg("n_angles"), py::arg("N_det"),
          py::arg("stream_ptr") = 0L);

    m.def("forward_project_resire_cuda", &py_forward_project_resire_cuda,
          py::call_guard<py::gil_scoped_release>(),
          "RESIRE voxel-driven forward projection (CUDA). "
          "mats_ptr: device data_ptr() of [n_angles*9] row-major float32 matrices. "
          "sino_ptr must point to a zeroed buffer.",
          py::arg("vol_ptr"), py::arg("sino_ptr"), py::arg("mats_ptr"),
          py::arg("Ny"), py::arg("Nz"), py::arg("Nx"),
          py::arg("n_angles"), py::arg("N_det"),
          py::arg("stream_ptr") = 0L);

    m.def("back_project_resire_cuda", &py_back_project_resire_cuda,
          py::call_guard<py::gil_scoped_release>(),
          "RESIRE voxel-driven back-projection (CUDA). "
          "mats_ptr: device data_ptr() of [n_angles*9] row-major float32 matrices. "
          "vol_ptr must point to a zeroed buffer.",
          py::arg("sino_ptr"), py::arg("vol_ptr"), py::arg("mats_ptr"),
          py::arg("Ny"), py::arg("Nz"), py::arg("Nx"),
          py::arg("n_angles"), py::arg("N_det"),
          py::arg("stream_ptr") = 0L);

    m.def("grad_rotmat_cuda", &py_grad_rotmat_cuda,
          py::call_guard<py::gil_scoped_release>(),
          "Gradient of 3D rotmat forward projection w.r.t. rotation matrix entries (CUDA). "
          "All ptr args are device data_ptr() ints. "
          "grad_mats_ptr must point to a zeroed [n_angles*9] buffer.",
          py::arg("vol_ptr"), py::arg("grad_sino_ptr"),
          py::arg("mats_ptr"), py::arg("grad_mats_ptr"),
          py::arg("Ny"), py::arg("Nz"), py::arg("Nx"),
          py::arg("n_angles"), py::arg("N_det"),
          py::arg("stream_ptr") = 0L);

    m.def("grad_rotmat_cuda_cached", &py_grad_rotmat_cuda_cached,
          py::call_guard<py::gil_scoped_release>(),
          "Cached gradient of 3D rotmat forward projection w.r.t. rotation matrix entries (CUDA). "
          "cache_ptr: TC_VOL_3D handle already populated by forward_project_rotmat_cuda_cached. "
          "Uses the cache's point-filter tex_point for bit-exact corner reads. "
          "io_dtype: 0=fp32, 1=fp16. "
          "grad_mats_ptr must point to a zeroed [n_angles*9] buffer.",
          py::arg("io_dtype"),
          py::arg("cache_ptr"),
          py::arg("grad_sino_ptr"), py::arg("mats_ptr"), py::arg("grad_mats_ptr"),
          py::arg("Ny"), py::arg("Nz"), py::arg("Nx"),
          py::arg("n_angles"), py::arg("N_det"),
          py::arg("stream_ptr") = 0L);

    m.def("grad_angle_cuda", &py_grad_angle_cuda,
          py::call_guard<py::gil_scoped_release>(),
          "Gradient of forward projection w.r.t. tilt angles (CUDA). "
          "vol_ptr/grad_sino_ptr/cos_ptr/sin_ptr/grad_angles_ptr: device data_ptr() ints. "
          "Returns gradient in radians via grad_angles_ptr.",
          py::arg("vol_ptr"), py::arg("grad_sino_ptr"),
          py::arg("cos_ptr"), py::arg("sin_ptr"),
          py::arg("grad_angles_ptr"),
          py::arg("Ny"), py::arg("Nz"), py::arg("Nx"),
          py::arg("n_angles"), py::arg("N_det"),
          py::arg("stream_ptr") = 0L);

    m.def("tc_create", &py_tc_create,
          py::call_guard<py::gil_scoped_release>(),
          "Allocate a persistent cudaArray + cudaTextureObject. "
          "kind: 0=TC_VOL_LAYERED (FP scalar), 1=TC_SINO_LAYERED (BP scalar/rotmat), "
          "2=TC_VOL_3D (FP rotmat). dtype: 0=fp32, 1=fp16. "
          "channels: 1 (default), 2 or 4: TorchRadon-style Y-axis packing "
          "(2 Y-slices per float2 texel for fp32, 4 per texel for fp16; any "
          "Ny, the last texel is zero-padded; layered kinds only; see "
          "parallel_channels). dim_a/b/c match the source-tensor dims "
          "in C order: vol is (Ny,Nz,Nx); sino is (n_angles,Ny,N_det). "
          "Returns an opaque integer handle — pass to tc_destroy when done. "
          "surface=True also creates a surface object on a 1-channel cache "
          "(within the device's surface limits) so fused kernels can write "
          "the array directly; see tc_has_surface.",
          py::arg("kind"), py::arg("dtype"), py::arg("channels"),
          py::arg("dim_a"), py::arg("dim_b"), py::arg("dim_c"),
          py::arg("surface") = false);

    m.def("parallel_channels", &parallel_channels,
          "Y slices per texel of the parallel FP/BP textures for io_dtype "
          "(0 = fp32, 1 = fp16) and Ny: what the one-shot wrappers use, and "
          "what a cached caller passes to tc_create so both share a layout.",
          py::arg("io_dtype"), py::arg("Ny"));

    m.def("tc_has_surface", &py_tc_has_surface,
          "True if the TextureCache handle has a surface object (writable "
          "by the fused SIRT kernels).",
          py::arg("handle"));

    m.def("tc_destroy", &py_tc_destroy,
          py::call_guard<py::gil_scoped_release>(),
          "Release a TextureCache handle returned by tc_create. Synchronizes "
          "the current device first, so kernels still reading the cache "
          "finish before its textures and cudaArray are freed.",
          py::arg("handle"));

    m.def("forward_project_parallel_cuda_cached", &py_forward_project_parallel_cuda_cached,
          py::call_guard<py::gil_scoped_release>(),
          "Forward projection (CUDA, cached). io_dtype: 0=fp32, 1=fp16. "
          "cache_ptr: TC_VOL_LAYERED handle sized to (Ny, Nz, Nx) with matching dtype. "
          "Uploads volume into the cache's cudaArray and launches the kernel — "
          "no per-call cudaMalloc3DArray.",
          py::arg("io_dtype"),
          py::arg("cache_ptr"),
          py::arg("vol_ptr"), py::arg("sino_ptr"),
          py::arg("cos_ptr"),    py::arg("sin_ptr"),
          py::arg("Ny"), py::arg("Nz"), py::arg("Nx"),
          py::arg("n_angles"), py::arg("N_det"),
          py::arg("stream_ptr") = 0L, py::arg("upload") = true,
          py::arg("nonfinite_ptr") = 0L);

    m.def("back_project_parallel_cuda_cached", &py_back_project_parallel_cuda_cached,
          py::call_guard<py::gil_scoped_release>(),
          "Back-projection (CUDA, cached). io_dtype: 0=fp32, 1=fp16. "
          "cache_ptr: TC_SINO_LAYERED handle sized to (n_angles, Ny, N_det) with matching dtype.",
          py::arg("io_dtype"),
          py::arg("cache_ptr"),
          py::arg("sino_ptr"), py::arg("vol_ptr"),
          py::arg("cos_ptr"),      py::arg("sin_ptr"),
          py::arg("Ny"), py::arg("Nz"), py::arg("Nx"),
          py::arg("n_angles"), py::arg("N_det"),
          py::arg("stream_ptr") = 0L, py::arg("upload") = true,
          py::arg("nonfinite_ptr") = 0L);

    m.def("forward_project_rotmat_cuda_cached", &py_forward_project_rotmat_cuda_cached,
          py::call_guard<py::gil_scoped_release>(),
          "3D rotmat forward projection (CUDA, cached). io_dtype: 0=fp32, 1=fp16. "
          "cache_ptr: TC_VOL_3D handle sized to (Ny, Nz, Nx) with matching dtype.",
          py::arg("io_dtype"),
          py::arg("cache_ptr"),
          py::arg("vol_ptr"), py::arg("sino_ptr"), py::arg("mats_ptr"),
          py::arg("Ny"), py::arg("Nz"), py::arg("Nx"),
          py::arg("n_angles"), py::arg("N_det"),
          py::arg("stream_ptr") = 0L, py::arg("upload") = true);

    m.def("back_project_rotmat_cuda_cached", &py_back_project_rotmat_cuda_cached,
          py::call_guard<py::gil_scoped_release>(),
          "3D rotmat back-projection (CUDA, cached). io_dtype: 0=fp32, 1=fp16. "
          "cache_ptr: TC_SINO_LAYERED handle sized to (n_angles, Ny, N_det) with matching dtype.",
          py::arg("io_dtype"),
          py::arg("cache_ptr"),
          py::arg("sino_ptr"), py::arg("vol_ptr"), py::arg("mats_ptr"),
          py::arg("Ny"), py::arg("Nz"), py::arg("Nx"),
          py::arg("n_angles"), py::arg("N_det"),
          py::arg("stream_ptr") = 0L, py::arg("upload") = true);

    m.def("tv_loss_iso_3d_cuda", &py_tv_loss_iso_3d_cuda,
          py::call_guard<py::gil_scoped_release>(),
          "Fused 3-D isotropic TV loss (CUDA). Forward: writes Σ s to "
          "acc_ptr (single-element fp32 buffer; zero it: left untouched when "
          "a dimension is < 2), deterministic. partials_ptr: fp64 scratch of "
          "tv_loss_iso_3d_workspace(Ny, Nz, Nx) elements. Caller "
          "post-divides by N=(Ny-1)(Nz-1)(Nx-1) to obtain mean(s).",
          py::arg("vol_ptr"), py::arg("acc_ptr"), py::arg("partials_ptr"),
          py::arg("Ny"), py::arg("Nz"), py::arg("Nx"),
          py::arg("eps"),
          py::arg("stream_ptr") = 0L);

    m.def("tv_loss_iso_3d_workspace", &tv_loss_iso_3d_workspace,
          "fp64 elements of scratch tv_loss_iso_3d_cuda needs for a "
          "[Ny, Nz, Nx] volume (0 when a dimension is < 2).",
          py::arg("Ny"), py::arg("Nz"), py::arg("Nx"));

    m.def("tv_loss_iso_3d_grad_cuda", &py_tv_loss_iso_3d_grad_cuda,
          py::call_guard<py::gil_scoped_release>(),
          "Backward of tv_loss_iso_3d. g_scaled_ptr: single-element fp32 "
          "buffer holding grad_out · 1/N. grad_vol_ptr: [Ny, Nz, Nx] fp32 "
          "output; the kernel writes every voxel exactly once when Ny, Nz and "
          "Nx are all >= 2. With a dimension < 2 there are no corners and "
          "nothing is launched, so the caller must zero the buffer then.",
          py::arg("vol_ptr"), py::arg("g_scaled_ptr"), py::arg("grad_vol_ptr"),
          py::arg("Ny"), py::arg("Nz"), py::arg("Nx"),
          py::arg("eps"),
          py::arg("stream_ptr") = 0L);

    m.def("tv_loss_xz_cuda", &py_tv_loss_xz_cuda,
          py::call_guard<py::gil_scoped_release>(),
          "Fused planar (XZ) TV loss (CUDA): writes Σ sqrt(dz² + dx² + eps) "
          "over the Ny·(Nz-1)·(Nx-1) corners to acc_ptr (single-element fp32 "
          "buffer; zero it: left untouched when Nz or Nx is < 2), "
          "deterministic. partials_ptr: fp64 scratch of "
          "tv_loss_xz_workspace(Ny, Nz, Nx) elements. Caller post-divides "
          "by N.",
          py::arg("vol_ptr"), py::arg("acc_ptr"), py::arg("partials_ptr"),
          py::arg("Ny"), py::arg("Nz"), py::arg("Nx"),
          py::arg("eps"),
          py::arg("stream_ptr") = 0L);

    m.def("tv_loss_xz_workspace", &tv_loss_xz_workspace,
          "fp64 elements of scratch tv_loss_xz_cuda needs for a "
          "[Ny, Nz, Nx] volume (0 when Nz or Nx is < 2).",
          py::arg("Ny"), py::arg("Nz"), py::arg("Nx"));

    m.def("tv_loss_xz_grad_cuda", &py_tv_loss_xz_grad_cuda,
          py::call_guard<py::gil_scoped_release>(),
          "Backward of tv_loss_xz_cuda. g_scaled_ptr: single-element fp32 "
          "buffer holding grad_out · 1/N. Writes every voxel of grad_vol_ptr "
          "once when Nz and Nx are >= 2; otherwise launches nothing (zero the "
          "buffer).",
          py::arg("vol_ptr"), py::arg("g_scaled_ptr"), py::arg("grad_vol_ptr"),
          py::arg("Ny"), py::arg("Nz"), py::arg("Nx"),
          py::arg("eps"),
          py::arg("stream_ptr") = 0L);

    m.def("sirt_residual_cuda", &py_sirt_residual_cuda,
          py::call_guard<py::gil_scoped_release>(),
          "Fused SIRT residual: raw = sino - fp, res = raw * R (n elements, "
          "io_dtype 0 = fp32, 1 = fp16). raw_out_ptr / res_out_ptr may be 0.",
          py::arg("io_dtype"), py::arg("sino_ptr"), py::arg("fp_ptr"), py::arg("R_ptr"),
          py::arg("raw_out_ptr"), py::arg("res_out_ptr"), py::arg("n"),
          py::arg("stream_ptr") = 0L);

    m.def("sirt_update_cuda", &py_sirt_update_cuda,
          py::call_guard<py::gil_scoped_release>(),
          "Fused SIRT update + constraints: vol_out = constraints(vol_in + "
          "omega * C * upd), eager (1) or CUDA-graph (0) rounding order; "
          "mask_ptr 0 = no support mask, mask_full = 1 for a [Ny, Nz, Nx] mask "
          "(else [1, Nz, Nx]); soft_thr <= 0 disables the soft threshold. "
          "vol_cache_ptr (0 = none): a volume texture cache with a surface "
          "that also receives vol_out; with it, vol_in_ptr = 0 reads the "
          "volume from the texture and vol_out_ptr = 0 skips the global write.",
          py::arg("io_dtype"), py::arg("eager_order"),
          py::arg("vol_in_ptr"), py::arg("vol_out_ptr"),
          py::arg("upd_ptr"), py::arg("C_ptr"),
          py::arg("mask_ptr"), py::arg("mask_full"),
          py::arg("omega"), py::arg("positivity"), py::arg("soft_thr"),
          py::arg("Ny"), py::arg("Nz"), py::arg("Nx"),
          py::arg("stream_ptr") = 0L, py::arg("vol_cache_ptr") = 0L);

    m.def("sirt_residual_tex_cuda", &py_sirt_residual_tex_cuda,
          py::call_guard<py::gil_scoped_release>(),
          "Fused SIRT residual (sino - fp) * R written into the sinogram "
          "texture cache cache_ptr (surface required) for a cached "
          "back-projection with upload=False; raw_out_ptr (sino - fp) may be 0.",
          py::arg("io_dtype"), py::arg("sino_ptr"), py::arg("fp_ptr"), py::arg("R_ptr"),
          py::arg("raw_out_ptr"), py::arg("cache_ptr"),
          py::arg("n_angles"), py::arg("Ny"), py::arg("N_det"),
          py::arg("stream_ptr") = 0L);

    m.def("tc_upload", &py_tc_upload,
          py::call_guard<py::gil_scoped_release>(),
          "Copy a contiguous device tensor (the cache's shape and dtype) into "
          "the TextureCache handle (cudaMemcpy3DAsync, or the 4-channel pack "
          "kernel), on stream_ptr.",
          py::arg("handle"), py::arg("src_ptr"), py::arg("stream_ptr") = 0L);

    m.def("tc_download", &py_tc_download,
          py::call_guard<py::gil_scoped_release>(),
          "Copy the TextureCache's contents into a contiguous device tensor of "
          "its source shape and dtype (inverse of tc_upload), on stream_ptr.",
          py::arg("handle"), py::arg("dst_ptr"), py::arg("stream_ptr") = 0L);

    m.def("genfire_gridding_2d_cuda", &py_genfire_gridding_2d_cuda,
          py::call_guard<py::gil_scoped_release>(),
          "GENFIRE Fourier gridding (CUDA). All args are device data_ptr() ints "
          "except bigN/Ny/n_angles/cutoff_dist.",
          py::arg("fk_real_ptr"), py::arg("fk_imag_ptr"),
          py::arg("cos_ptr"),     py::arg("sin_ptr"),
          py::arg("grid_real_ptr"), py::arg("grid_imag_ptr"), py::arg("weight_ptr"),
          py::arg("bigN"), py::arg("Ny"), py::arg("n_angles"),
          py::arg("cutoff_dist") = 0.5f,
          py::arg("stream_ptr") = 0L);

    m.def("mbir_forward_project_cuda", &py_mbir_forward_project_cuda,
          py::call_guard<py::gil_scoped_release>(),
          "MBIR voxel-driven bilinear scatter forward projection (CUDA). "
          "Distinct from forward_project_parallel_cuda (ray-marching). sino_ptr is "
          "zeroed by the wrapper before launch.",
          py::arg("vol_ptr"), py::arg("sino_ptr"),
          py::arg("cos_ptr"), py::arg("sin_ptr"),
          py::arg("Ny"), py::arg("Nz"), py::arg("Nx"),
          py::arg("n_angles"), py::arg("N_det"),
          py::arg("stream_ptr") = 0L);

    m.def("mbir_icd_sweep_cuda", &py_mbir_icd_sweep_cuda,
          py::call_guard<py::gil_scoped_release>(),
          "One outer iteration of MBIR-AM-PE (CUDA). In-place mutates "
          "volume and e_sino. Pass lambdas_ptr=0 for uniform weights "
          "(treated as nullptr). The voxel order of each Y-slice is generated "
          "in the kernel from perm_key unless perm_ptr ([Ny, Nz*Nx] int32) "
          "is non-zero. work_ptr: fp32 scratch of Ny*Nz*Nx*2 elements "
          "(8-byte aligned). Asynchronous on the stream. "
          "Per-tilt offset d_k and noise variance σ "
          "are updated in PyTorch by the Python wrapper using the post-call "
          "e_sino (the kernel never reads d_k: e = g - Af - d). "
          "group_spacing = 0 runs the serial sweep (one warp per slice); "
          "S > 0 the parallel sweep over conflict-free voxel groups of "
          "spacing S (the caller guarantees S * min|cos| >= 2 + eps, see "
          "fasttomo.algorithms.mbir._mbir_group_spacing; needs perm_ptr = 0 "
          "and mbir_group_sweep_supported).",
          py::arg("vol_ptr"), py::arg("e_sino_ptr"),
          py::arg("vol_snapshot_ptr"),
          py::arg("lambdas_ptr"),
          py::arg("perm_ptr"), py::arg("perm_key"), py::arg("work_ptr"),
          py::arg("cos_ptr"), py::arg("sin_ptr"),
          py::arg("Ny"), py::arg("Nz"), py::arg("Nx"),
          py::arg("n_angles"), py::arg("N_det"),
          py::arg("qggmrf_p"), py::arg("qggmrf_q"),
          py::arg("qggmrf_c"), py::arg("sigma_f"),
          py::arg("huber_T"), py::arg("huber_delta"), py::arg("sigma"),
          py::arg("stream_ptr") = 0L, py::arg("group_spacing") = 0);

    m.def("mbir_group_sweep_supported", &mbir_group_sweep_supported,
          "Whether the parallel MBIR sweep can run for this many views and "
          "detector pixels on the current device (n_angles <= 256 and the "
          "slice's error sinogram fits in shared memory).",
          py::arg("n_angles"), py::arg("N_det"));

    m.def("mbir_permutation_cuda", &py_mbir_permutation_cuda,
          "Test hook: write the in-kernel MBIR voxel order ([Ny, nvox] int32) "
          "for perm_key to out_ptr.",
          py::arg("out_ptr"), py::arg("Ny"), py::arg("nvox"), py::arg("perm_key"),
          py::arg("stream_ptr") = 0L);

    m.def("_tc_debug_fail_next_create", &tc_debug_fail_next_create,
          "Test hook: make the next tc_create throw at `stage` "
          "(1 = after cudaMalloc3DArray, 2 = after the texture object, "
          "3 = after all handles exist, 0 = off).",
          py::arg("stage"));

    m.def("_debug_set_rotmat_ray_clip", &_debug_set_rotmat_ray_clip,
          "Test hook: 1 = clip rotmat forward-projection rays to the volume "
          "(default), 0 = march the full ray. Returns the previous setting.",
          py::arg("on"));

    m.attr("cuda_available") = true;
}

#else /* !FASTTOMO_CUDA */

#include <pybind11/pybind11.h>
#include "bindings_cuda.h"

namespace py = pybind11;

void register_cuda(py::module_ &m) {
    m.attr("cuda_available") = false;
}

#endif /* FASTTOMO_CUDA */

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

/* Validate a TextureCache pointer against expected kind, dtype, and shape.
 * Called by each cached FP/BP entry point before dispatching to the kernel. */
static void validate_cache(const TextureCache *tc, tc_kind_t expected_kind,
                           int io_dtype, int da, int db, int dc) {
    if (!tc) throw std::invalid_argument("cache pointer is null");
    if (tc->kind != expected_kind ||
        tc->dim_a != da || tc->dim_b != db || tc->dim_c != dc)
        throw std::invalid_argument("cache shape/kind mismatch");
    int cache_dtype = (tc->dtype == TC_DTYPE_FP32) ? 0 : 1;
    if (cache_dtype != io_dtype)
        throw std::invalid_argument("cache dtype mismatch with io_dtype");
}

/* ── CUDA projection bindings ─────────────────────────────────────────── */

static void py_forward_project_cuda(
    int io_dtype,
    long volume_ptr, long sinogram_ptr,
    long cos_ptr, long sin_ptr,
    int Ny, int Nz, int Nx, int n_angles, int N_det,
    long stream_ptr
) {
    forward_project_cuda(
        io_dtype,
        reinterpret_cast<const void *>(volume_ptr),
        reinterpret_cast<void *>(sinogram_ptr),
        reinterpret_cast<const float *>(cos_ptr),
        reinterpret_cast<const float *>(sin_ptr),
        Ny, Nz, Nx, n_angles, N_det,
        to_stream(stream_ptr)
    );
}

static void py_back_project_cuda(
    int io_dtype,
    long sinogram_ptr, long volume_ptr,
    long cos_ptr, long sin_ptr,
    int Ny, int Nz, int Nx, int n_angles, int N_det,
    long stream_ptr
) {
    back_project_cuda(
        io_dtype,
        reinterpret_cast<const void *>(sinogram_ptr),
        reinterpret_cast<void *>(volume_ptr),
        reinterpret_cast<const float *>(cos_ptr),
        reinterpret_cast<const float *>(sin_ptr),
        Ny, Nz, Nx, n_angles, N_det,
        to_stream(stream_ptr)
    );
}

static void py_forward_project_rotmat_cuda(
    int io_dtype,
    long vol_ptr, long sino_ptr, long matrices_ptr,
    int Ny, int Nz, int Nx, int n_angles, int N_det,
    long stream_ptr
) {
    forward_project_rotmat_cuda(
        io_dtype,
        reinterpret_cast<const void *>(vol_ptr),
        reinterpret_cast<void *>(sino_ptr),
        reinterpret_cast<const float *>(matrices_ptr),
        Ny, Nz, Nx, n_angles, N_det,
        to_stream(stream_ptr)
    );
}

static void py_back_project_rotmat_cuda(
    int io_dtype,
    long sino_ptr, long vol_ptr, long matrices_ptr,
    int Ny, int Nz, int Nx, int n_angles, int N_det,
    long stream_ptr
) {
    back_project_rotmat_cuda(
        io_dtype,
        reinterpret_cast<const void *>(sino_ptr),
        reinterpret_cast<void *>(vol_ptr),
        reinterpret_cast<const float *>(matrices_ptr),
        Ny, Nz, Nx, n_angles, N_det,
        to_stream(stream_ptr)
    );
}

static void py_forward_project_resire_cuda(
    long vol_ptr, long sino_ptr, long matrices_ptr,
    int Ny, int Nz, int Nx, int n_angles, int N_det,
    long stream_ptr
) {
    forward_project_resire_cuda(
        reinterpret_cast<const float *>(vol_ptr),
        reinterpret_cast<float *>(sino_ptr),
        reinterpret_cast<const float *>(matrices_ptr),
        Ny, Nz, Nx, n_angles, N_det,
        to_stream(stream_ptr)
    );
}

static void py_back_project_resire_cuda(
    long sino_ptr, long vol_ptr, long matrices_ptr,
    int Ny, int Nz, int Nx, int n_angles, int N_det,
    long stream_ptr
) {
    back_project_resire_cuda(
        reinterpret_cast<const float *>(sino_ptr),
        reinterpret_cast<float *>(vol_ptr),
        reinterpret_cast<const float *>(matrices_ptr),
        Ny, Nz, Nx, n_angles, N_det,
        to_stream(stream_ptr)
    );
}

static void py_grad_rotmat_cuda(
    long vol_ptr, long grad_sino_ptr,
    long matrices_ptr, long grad_matrices_ptr,
    int Ny, int Nz, int Nx, int n_angles, int N_det,
    long stream_ptr
) {
    grad_rotmat_cuda(
        reinterpret_cast<const float *>(vol_ptr),
        reinterpret_cast<const float *>(grad_sino_ptr),
        reinterpret_cast<const float *>(matrices_ptr),
        reinterpret_cast<float *>(grad_matrices_ptr),
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
                         int dim_a, int dim_b, int dim_c) {
    if (kind < 0 || kind > 2)
        throw std::invalid_argument("tc_create: kind must be 0..2");
    if (dtype < 0 || dtype > 1)
        throw std::invalid_argument("tc_create: dtype must be 0 (fp32) or 1 (fp16)");
    if (channels != 1 && channels != 4)
        throw std::invalid_argument("tc_create: channels must be 1 or 4");
    TextureCache *tc = tc_create(
        static_cast<tc_kind_t>(kind),
        static_cast<tc_dtype_t>(dtype),
        channels,
        dim_a, dim_b, dim_c);
    return reinterpret_cast<long>(tc);
}

static void py_tc_destroy(long handle) {
    tc_destroy(reinterpret_cast<TextureCache *>(handle));
}

/* ── cached FP/BP entry points ─────────────────────────────────────────── */

static void py_forward_project_cuda_cached(
    int io_dtype,
    long cache_ptr,
    long volume_ptr, long sinogram_ptr,
    long cos_ptr, long sin_ptr,
    int Ny, int Nz, int Nx, int n_angles, int N_det,
    long stream_ptr
) {
    validate_cache(reinterpret_cast<const TextureCache *>(cache_ptr),
                   TC_VOL_LAYERED, io_dtype, Ny, Nz, Nx);
    forward_project_cuda_cached(
        io_dtype,
        reinterpret_cast<TextureCache *>(cache_ptr),
        reinterpret_cast<const void *>(volume_ptr),
        reinterpret_cast<void *>(sinogram_ptr),
        reinterpret_cast<const float *>(cos_ptr),
        reinterpret_cast<const float *>(sin_ptr),
        Ny, Nz, Nx, n_angles, N_det,
        to_stream(stream_ptr)
    );
}

static void py_back_project_cuda_cached(
    int io_dtype,
    long cache_ptr,
    long sinogram_ptr, long volume_ptr,
    long cos_ptr, long sin_ptr,
    int Ny, int Nz, int Nx, int n_angles, int N_det,
    long stream_ptr
) {
    validate_cache(reinterpret_cast<const TextureCache *>(cache_ptr),
                   TC_SINO_LAYERED, io_dtype, n_angles, Ny, N_det);
    back_project_cuda_cached(
        io_dtype,
        reinterpret_cast<TextureCache *>(cache_ptr),
        reinterpret_cast<const void *>(sinogram_ptr),
        reinterpret_cast<void *>(volume_ptr),
        reinterpret_cast<const float *>(cos_ptr),
        reinterpret_cast<const float *>(sin_ptr),
        Ny, Nz, Nx, n_angles, N_det,
        to_stream(stream_ptr)
    );
}

static void py_forward_project_rotmat_cuda_cached(
    int io_dtype,
    long cache_ptr,
    long vol_ptr, long sino_ptr, long matrices_ptr,
    int Ny, int Nz, int Nx, int n_angles, int N_det,
    long stream_ptr
) {
    validate_cache(reinterpret_cast<const TextureCache *>(cache_ptr),
                   TC_VOL_3D, io_dtype, Ny, Nz, Nx);
    forward_project_rotmat_cuda_cached(
        io_dtype,
        reinterpret_cast<TextureCache *>(cache_ptr),
        reinterpret_cast<const void *>(vol_ptr),
        reinterpret_cast<void *>(sino_ptr),
        reinterpret_cast<const float *>(matrices_ptr),
        Ny, Nz, Nx, n_angles, N_det,
        to_stream(stream_ptr)
    );
}

static void py_back_project_rotmat_cuda_cached(
    int io_dtype,
    long cache_ptr,
    long sino_ptr, long vol_ptr, long matrices_ptr,
    int Ny, int Nz, int Nx, int n_angles, int N_det,
    long stream_ptr
) {
    validate_cache(reinterpret_cast<const TextureCache *>(cache_ptr),
                   TC_SINO_LAYERED, io_dtype, n_angles, Ny, N_det);
    back_project_rotmat_cuda_cached(
        io_dtype,
        reinterpret_cast<TextureCache *>(cache_ptr),
        reinterpret_cast<const void *>(sino_ptr),
        reinterpret_cast<void *>(vol_ptr),
        reinterpret_cast<const float *>(matrices_ptr),
        Ny, Nz, Nx, n_angles, N_det,
        to_stream(stream_ptr)
    );
}

static void py_tv_loss_iso_3d_cuda(
    long vol_ptr, long acc_ptr,
    int Ny, int Nz, int Nx,
    float eps,
    long stream_ptr
) {
    tv_loss_iso_3d_cuda(
        reinterpret_cast<const float *>(vol_ptr),
        reinterpret_cast<float *>(acc_ptr),
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

void register_cuda(py::module_ &m) {
    m.def("forward_project_cuda", &py_forward_project_cuda,
          "Forward projection (CUDA). io_dtype: 0=fp32, 1=fp16. "
          "All ptr args are device data_ptr() ints. "
          "stream_ptr: torch.cuda.Stream.cuda_stream (0 → default stream).",
          py::arg("io_dtype"),
          py::arg("volume_ptr"), py::arg("sinogram_ptr"),
          py::arg("cos_ptr"),    py::arg("sin_ptr"),
          py::arg("Ny"), py::arg("Nz"), py::arg("Nx"),
          py::arg("n_angles"), py::arg("N_det"),
          py::arg("stream_ptr") = 0L);

    m.def("back_project_cuda", &py_back_project_cuda,
          "Back-projection (CUDA). io_dtype: 0=fp32, 1=fp16. "
          "All ptr args are device data_ptr() ints.",
          py::arg("io_dtype"),
          py::arg("sinogram_ptr"), py::arg("volume_ptr"),
          py::arg("cos_ptr"),      py::arg("sin_ptr"),
          py::arg("Ny"), py::arg("Nz"), py::arg("Nx"),
          py::arg("n_angles"), py::arg("N_det"),
          py::arg("stream_ptr") = 0L);

    m.def("forward_project_rotmat_cuda", &py_forward_project_rotmat_cuda,
          "3D rotation-matrix forward projection (CUDA). io_dtype: 0=fp32, 1=fp16. "
          "matrices_ptr: device data_ptr() of [n_angles*9] row-major float32 rotation matrices.",
          py::arg("io_dtype"),
          py::arg("vol_ptr"), py::arg("sino_ptr"), py::arg("matrices_ptr"),
          py::arg("Ny"), py::arg("Nz"), py::arg("Nx"),
          py::arg("n_angles"), py::arg("N_det"),
          py::arg("stream_ptr") = 0L);

    m.def("back_project_rotmat_cuda", &py_back_project_rotmat_cuda,
          "3D rotation-matrix back-projection (CUDA). io_dtype: 0=fp32, 1=fp16. "
          "matrices_ptr: device data_ptr() of [n_angles*9] row-major float32 rotation matrices.",
          py::arg("io_dtype"),
          py::arg("sino_ptr"), py::arg("vol_ptr"), py::arg("matrices_ptr"),
          py::arg("Ny"), py::arg("Nz"), py::arg("Nx"),
          py::arg("n_angles"), py::arg("N_det"),
          py::arg("stream_ptr") = 0L);

    m.def("forward_project_resire_cuda", &py_forward_project_resire_cuda,
          "RESIRE voxel-driven forward projection (CUDA). "
          "matrices_ptr: device data_ptr() of [n_angles*9] row-major float32 matrices. "
          "sino_ptr must point to a zeroed buffer.",
          py::arg("vol_ptr"), py::arg("sino_ptr"), py::arg("matrices_ptr"),
          py::arg("Ny"), py::arg("Nz"), py::arg("Nx"),
          py::arg("n_angles"), py::arg("N_det"),
          py::arg("stream_ptr") = 0L);

    m.def("back_project_resire_cuda", &py_back_project_resire_cuda,
          "RESIRE voxel-driven back-projection (CUDA). "
          "matrices_ptr: device data_ptr() of [n_angles*9] row-major float32 matrices. "
          "vol_ptr must point to a zeroed buffer.",
          py::arg("sino_ptr"), py::arg("vol_ptr"), py::arg("matrices_ptr"),
          py::arg("Ny"), py::arg("Nz"), py::arg("Nx"),
          py::arg("n_angles"), py::arg("N_det"),
          py::arg("stream_ptr") = 0L);

    m.def("grad_rotmat_cuda", &py_grad_rotmat_cuda,
          "Gradient of 3D rotmat forward projection w.r.t. rotation matrix entries (CUDA). "
          "All ptr args are device data_ptr() ints. "
          "grad_matrices_ptr must point to a zeroed [n_angles*9] buffer.",
          py::arg("vol_ptr"), py::arg("grad_sino_ptr"),
          py::arg("matrices_ptr"), py::arg("grad_matrices_ptr"),
          py::arg("Ny"), py::arg("Nz"), py::arg("Nx"),
          py::arg("n_angles"), py::arg("N_det"),
          py::arg("stream_ptr") = 0L);

    m.def("grad_angle_cuda", &py_grad_angle_cuda,
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
          "Allocate a persistent cudaArray + cudaTextureObject. "
          "kind: 0=TC_VOL_LAYERED (FP scalar), 1=TC_SINO_LAYERED (BP scalar/rotmat), "
          "2=TC_VOL_3D (FP rotmat). dtype: 0=fp32, 1=fp16. "
          "channels: 1 (default) or 4. channels=4 enables TorchRadon-style "
          "Y-axis packing (4 Y-slices per texel) — fp16 only, packing-axis "
          "must be divisible by 4. dim_a/b/c match the source-tensor dims "
          "in C order: vol is (Ny,Nz,Nx); sino is (n_angles,Ny,N_det). "
          "Returns an opaque integer handle — pass to tc_destroy when done.",
          py::arg("kind"), py::arg("dtype"), py::arg("channels"),
          py::arg("dim_a"), py::arg("dim_b"), py::arg("dim_c"));

    m.def("tc_destroy", &py_tc_destroy,
          "Release a TextureCache handle returned by tc_create.",
          py::arg("handle"));

    m.def("forward_project_cuda_cached", &py_forward_project_cuda_cached,
          "Forward projection (CUDA, cached). io_dtype: 0=fp32, 1=fp16. "
          "cache_ptr: TC_VOL_LAYERED handle sized to (Ny, Nz, Nx) with matching dtype. "
          "Uploads volume into the cache's cudaArray and launches the kernel — "
          "no per-call cudaMalloc3DArray.",
          py::arg("io_dtype"),
          py::arg("cache_ptr"),
          py::arg("volume_ptr"), py::arg("sinogram_ptr"),
          py::arg("cos_ptr"),    py::arg("sin_ptr"),
          py::arg("Ny"), py::arg("Nz"), py::arg("Nx"),
          py::arg("n_angles"), py::arg("N_det"),
          py::arg("stream_ptr") = 0L);

    m.def("back_project_cuda_cached", &py_back_project_cuda_cached,
          "Back-projection (CUDA, cached). io_dtype: 0=fp32, 1=fp16. "
          "cache_ptr: TC_SINO_LAYERED handle sized to (n_angles, Ny, N_det) with matching dtype.",
          py::arg("io_dtype"),
          py::arg("cache_ptr"),
          py::arg("sinogram_ptr"), py::arg("volume_ptr"),
          py::arg("cos_ptr"),      py::arg("sin_ptr"),
          py::arg("Ny"), py::arg("Nz"), py::arg("Nx"),
          py::arg("n_angles"), py::arg("N_det"),
          py::arg("stream_ptr") = 0L);

    m.def("forward_project_rotmat_cuda_cached", &py_forward_project_rotmat_cuda_cached,
          "3D rotmat forward projection (CUDA, cached). io_dtype: 0=fp32, 1=fp16. "
          "cache_ptr: TC_VOL_3D handle sized to (Ny, Nz, Nx) with matching dtype.",
          py::arg("io_dtype"),
          py::arg("cache_ptr"),
          py::arg("vol_ptr"), py::arg("sino_ptr"), py::arg("matrices_ptr"),
          py::arg("Ny"), py::arg("Nz"), py::arg("Nx"),
          py::arg("n_angles"), py::arg("N_det"),
          py::arg("stream_ptr") = 0L);

    m.def("back_project_rotmat_cuda_cached", &py_back_project_rotmat_cuda_cached,
          "3D rotmat back-projection (CUDA, cached). io_dtype: 0=fp32, 1=fp16. "
          "cache_ptr: TC_SINO_LAYERED handle sized to (n_angles, Ny, N_det) with matching dtype.",
          py::arg("io_dtype"),
          py::arg("cache_ptr"),
          py::arg("sino_ptr"), py::arg("vol_ptr"), py::arg("matrices_ptr"),
          py::arg("Ny"), py::arg("Nz"), py::arg("Nx"),
          py::arg("n_angles"), py::arg("N_det"),
          py::arg("stream_ptr") = 0L);

    m.def("tv_loss_iso_3d_cuda", &py_tv_loss_iso_3d_cuda,
          "Fused 3-D isotropic TV loss (CUDA). Forward: atomicAdd Σ s into "
          "acc_ptr (single-element fp32 buffer, must be zeroed). Caller "
          "post-divides by N=(Ny-1)(Nz-1)(Nx-1) to obtain mean(s).",
          py::arg("vol_ptr"), py::arg("acc_ptr"),
          py::arg("Ny"), py::arg("Nz"), py::arg("Nx"),
          py::arg("eps"),
          py::arg("stream_ptr") = 0L);

    m.def("tv_loss_iso_3d_grad_cuda", &py_tv_loss_iso_3d_grad_cuda,
          "Backward of tv_loss_iso_3d. g_scaled_ptr: single-element fp32 "
          "buffer holding grad_out · 1/N. grad_vol_ptr: [Ny, Nz, Nx] fp32 "
          "output (must be zeroed; kernel writes once per voxel).",
          py::arg("vol_ptr"), py::arg("g_scaled_ptr"), py::arg("grad_vol_ptr"),
          py::arg("Ny"), py::arg("Nz"), py::arg("Nx"),
          py::arg("eps"),
          py::arg("stream_ptr") = 0L);

    m.def("genfire_gridding_2d_cuda", &py_genfire_gridding_2d_cuda,
          "GENFIRE Fourier gridding (CUDA). All args are device data_ptr() ints "
          "except bigN/Ny/n_angles/cutoff_dist.",
          py::arg("fk_real_ptr"), py::arg("fk_imag_ptr"),
          py::arg("cos_ptr"),     py::arg("sin_ptr"),
          py::arg("grid_real_ptr"), py::arg("grid_imag_ptr"), py::arg("weight_ptr"),
          py::arg("bigN"), py::arg("Ny"), py::arg("n_angles"),
          py::arg("cutoff_dist") = 0.5f,
          py::arg("stream_ptr") = 0L);

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

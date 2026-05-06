#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <vector>
#include <stdexcept>

#include "cpu/ops.h"

#ifdef FASTTOMO_CUDA
#include "cuda/ops.h"
#include <cuda_runtime.h>
#endif

namespace py = pybind11;
using f32arr = py::array_t<float>;

/* helper: validate that an array is float32 and C-contiguous */
static void check_f32(const f32arr &a, const char *name) {
    if (a.ndim() < 1)
        throw std::invalid_argument(std::string(name) + " must have at least 1 dim");
}

/* precompute cos/sin from a 1-D angles array (radians) */
static void compute_trig(const f32arr &angles,
                          std::vector<float> &cos_a,
                          std::vector<float> &sin_a) {
    auto buf = angles.request();
    int  n   = (int)buf.size;
    const float *ptr = static_cast<const float *>(buf.ptr);
    cos_a.resize(n);
    sin_a.resize(n);
    for (int i = 0; i < n; i++) {
        cos_a[i] = std::cos(ptr[i]);
        sin_a[i] = std::sin(ptr[i]);
    }
}

/* ── CPU bindings ─────────────────────────────────────────────────────── */

static f32arr py_forward_project_cpu(
    const f32arr &volume,   /* [Ny, Nz, Nx] */
    const f32arr &angles,   /* [n_angles] radians */
    int N_det               /* detector width; -1 → use Nx */
) {
    check_f32(volume, "volume");
    if (volume.ndim() != 3)
        throw std::invalid_argument("volume must be 3-D [Ny, Nz, Nx]");

    int Ny       = (int)volume.shape(0);
    int Nz       = (int)volume.shape(1);
    int Nx       = (int)volume.shape(2);
    int n_angles = (int)angles.size();
    if (N_det < 0) N_det = Nx;

    std::vector<float> cos_a, sin_a;
    compute_trig(angles, cos_a, sin_a);

    auto sinogram = py::array_t<float>({n_angles, Ny, N_det});
    auto out_buf  = sinogram.request();

    forward_project_cpu(
        static_cast<const float *>(volume.request().ptr),
        static_cast<float *>(out_buf.ptr),
        cos_a.data(), sin_a.data(),
        Ny, Nz, Nx, n_angles, N_det
    );
    return sinogram;
}

static f32arr py_back_project_cpu(
    const f32arr &sinogram,  /* [n_angles, Ny, N_det] */
    const f32arr &angles,    /* [n_angles] radians */
    int Nz                   /* reconstruction depth; -1 → use N_det */
) {
    check_f32(sinogram, "sinogram");
    if (sinogram.ndim() != 3)
        throw std::invalid_argument("sinogram must be 3-D [n_angles, Ny, N_det]");

    int n_angles = (int)sinogram.shape(0);
    int Ny       = (int)sinogram.shape(1);
    int N_det    = (int)sinogram.shape(2);
    int Nx       = N_det;
    if (Nz < 0) Nz = N_det;

    std::vector<float> cos_a, sin_a;
    compute_trig(angles, cos_a, sin_a);

    auto volume  = py::array_t<float>({Ny, Nz, Nx});
    auto out_buf = volume.request();

    back_project_cpu(
        static_cast<const float *>(sinogram.request().ptr),
        static_cast<float *>(out_buf.ptr),
        cos_a.data(), sin_a.data(),
        Ny, Nz, Nx, n_angles, N_det
    );
    return volume;
}

static f32arr py_grad_angle_cpu(
    const f32arr &volume,    /* [Ny, Nz, Nx] */
    const f32arr &grad_sino, /* [n_angles, Ny, N_det] */
    const f32arr &angles,    /* [n_angles] radians */
    int N_det
) {
    check_f32(volume, "volume");
    check_f32(grad_sino, "grad_sino");
    if (volume.ndim() != 3)
        throw std::invalid_argument("volume must be 3-D [Ny, Nz, Nx]");
    if (grad_sino.ndim() != 3)
        throw std::invalid_argument("grad_sino must be 3-D [n_angles, Ny, N_det]");

    int Ny       = (int)volume.shape(0);
    int Nz       = (int)volume.shape(1);
    int Nx       = (int)volume.shape(2);
    int n_angles = (int)grad_sino.shape(0);
    int N_det_gs = (int)grad_sino.shape(2);
    if (N_det < 0) N_det = N_det_gs;

    std::vector<float> cos_a, sin_a;
    compute_trig(angles, cos_a, sin_a);

    auto grad_angles = py::array_t<float>({n_angles});
    auto out_buf     = grad_angles.request();

    grad_angle_cpu(
        static_cast<const float *>(volume.request().ptr),
        static_cast<const float *>(grad_sino.request().ptr),
        static_cast<float *>(out_buf.ptr),
        cos_a.data(), sin_a.data(),
        Ny, Nz, Nx, n_angles, N_det
    );
    return grad_angles;
}

static py::tuple py_genfire_gridding_2d_cpu(
    const f32arr &fk_real,  /* [n_angles, Ny, bigN] */
    const f32arr &fk_imag,
    const f32arr &cos_a,    /* [n_angles] radians */
    const f32arr &sin_a,
    float cutoff_dist
) {
    if (fk_real.ndim() != 3)
        throw std::invalid_argument("fk_real must be 3-D [n_angles, Ny, bigN]");
    int n_angles = (int)fk_real.shape(0);
    int Ny       = (int)fk_real.shape(1);
    int bigN     = (int)fk_real.shape(2);

    auto grid_real   = py::array_t<float>({Ny, bigN, bigN});
    auto grid_imag   = py::array_t<float>({Ny, bigN, bigN});
    auto weight_grid = py::array_t<float>({Ny, bigN, bigN});
    auto gr_buf = grid_real.request();
    auto gi_buf = grid_imag.request();
    auto gw_buf = weight_grid.request();
    size_t n = (size_t)Ny * bigN * bigN;
    std::fill(static_cast<float *>(gr_buf.ptr), static_cast<float *>(gr_buf.ptr) + n, 0.0f);
    std::fill(static_cast<float *>(gi_buf.ptr), static_cast<float *>(gi_buf.ptr) + n, 0.0f);
    std::fill(static_cast<float *>(gw_buf.ptr), static_cast<float *>(gw_buf.ptr) + n, 0.0f);

    genfire_gridding_2d_cpu(
        static_cast<const float *>(fk_real.request().ptr),
        static_cast<const float *>(fk_imag.request().ptr),
        static_cast<const float *>(cos_a.request().ptr),
        static_cast<const float *>(sin_a.request().ptr),
        static_cast<float *>(gr_buf.ptr),
        static_cast<float *>(gi_buf.ptr),
        static_cast<float *>(gw_buf.ptr),
        bigN, Ny, n_angles, cutoff_dist
    );
    return py::make_tuple(grid_real, grid_imag, weight_grid);
}

/* ── CUDA bindings ────────────────────────────────────────────────────── */

#ifdef FASTTOMO_CUDA

/* Helper: allocate device memory, copy from host, return device pointer. */
static float *to_device(const float *host, size_t n) {
    float *dev;
    cudaMalloc(&dev, n * sizeof(float));
    cudaMemcpy(dev, host, n * sizeof(float), cudaMemcpyHostToDevice);
    return dev;
}

static void py_forward_project_cuda(
    long   volume_ptr,    /* d_volume  data_ptr() — device float* as Python int */
    long   sinogram_ptr,  /* d_sinogram data_ptr() */
    const f32arr &angles, /* [n_angles] radians — host array */
    int Ny, int Nz, int Nx,
    int n_angles, int N_det
) {
    std::vector<float> cos_a, sin_a;
    compute_trig(angles, cos_a, sin_a);

    float *d_cos = to_device(cos_a.data(), n_angles);
    float *d_sin = to_device(sin_a.data(), n_angles);

    forward_project_cuda(
        reinterpret_cast<const float *>(volume_ptr),
        reinterpret_cast<float *>(sinogram_ptr),
        d_cos, d_sin,
        Ny, Nz, Nx, n_angles, N_det
    );

    cudaFree(d_cos);
    cudaFree(d_sin);
}

static void py_back_project_cuda(
    long   sinogram_ptr,
    long   volume_ptr,
    const f32arr &angles,
    int Ny, int Nz, int Nx,
    int n_angles, int N_det
) {
    std::vector<float> cos_a, sin_a;
    compute_trig(angles, cos_a, sin_a);

    float *d_cos = to_device(cos_a.data(), n_angles);
    float *d_sin = to_device(sin_a.data(), n_angles);

    back_project_cuda(
        reinterpret_cast<const float *>(sinogram_ptr),
        reinterpret_cast<float *>(volume_ptr),
        d_cos, d_sin,
        Ny, Nz, Nx, n_angles, N_det
    );

    cudaFree(d_cos);
    cudaFree(d_sin);
}

static void py_forward_project_rotmat_cuda(
    long vol_ptr, long sino_ptr, long matrices_ptr,
    int Ny, int Nz, int Nx, int n_angles, int N_det
) {
    forward_project_rotmat_cuda(
        reinterpret_cast<const float *>(vol_ptr),
        reinterpret_cast<float *>(sino_ptr),
        reinterpret_cast<const float *>(matrices_ptr),
        Ny, Nz, Nx, n_angles, N_det
    );
}

static void py_back_project_rotmat_cuda(
    long sino_ptr, long vol_ptr, long matrices_ptr,
    int Ny, int Nz, int Nx, int n_angles, int N_det
) {
    back_project_rotmat_cuda(
        reinterpret_cast<const float *>(sino_ptr),
        reinterpret_cast<float *>(vol_ptr),
        reinterpret_cast<const float *>(matrices_ptr),
        Ny, Nz, Nx, n_angles, N_det
    );
}

static void py_forward_project_resire_cuda(
    long vol_ptr, long sino_ptr, long matrices_ptr,
    int Ny, int Nz, int Nx, int n_angles, int N_det
) {
    forward_project_resire_cuda(
        reinterpret_cast<const float *>(vol_ptr),
        reinterpret_cast<float *>(sino_ptr),
        reinterpret_cast<const float *>(matrices_ptr),
        Ny, Nz, Nx, n_angles, N_det
    );
}

static void py_back_project_resire_cuda(
    long sino_ptr, long vol_ptr, long matrices_ptr,
    int Ny, int Nz, int Nx, int n_angles, int N_det
) {
    back_project_resire_cuda(
        reinterpret_cast<const float *>(sino_ptr),
        reinterpret_cast<float *>(vol_ptr),
        reinterpret_cast<const float *>(matrices_ptr),
        Ny, Nz, Nx, n_angles, N_det
    );
}

static void py_grad_angle_cuda(
    long vol_ptr, long grad_sino_ptr,
    long cos_ptr, long sin_ptr,
    long grad_angles_ptr,
    int Ny, int Nz, int Nx, int n_angles, int N_det
) {
    grad_angle_cuda(
        reinterpret_cast<const float *>(vol_ptr),
        reinterpret_cast<const float *>(grad_sino_ptr),
        reinterpret_cast<const float *>(cos_ptr),
        reinterpret_cast<const float *>(sin_ptr),
        reinterpret_cast<float *>(grad_angles_ptr),
        Ny, Nz, Nx, n_angles, N_det
    );
}

static void py_grad_rotmat_resire_cuda(
    long vol_ptr, long grad_sino_ptr,
    long matrices_ptr, long grad_matrices_ptr,
    int Ny, int Nz, int Nx, int n_angles, int N_det
) {
    grad_rotmat_resire_cuda(
        reinterpret_cast<const float *>(vol_ptr),
        reinterpret_cast<const float *>(grad_sino_ptr),
        reinterpret_cast<const float *>(matrices_ptr),
        reinterpret_cast<float *>(grad_matrices_ptr),
        Ny, Nz, Nx, n_angles, N_det
    );
}

static void py_genfire_gridding_2d_cuda(
    long fk_real_ptr, long fk_imag_ptr,
    long cos_ptr,     long sin_ptr,
    long grid_real_ptr, long grid_imag_ptr, long weight_ptr,
    int bigN, int Ny, int n_angles, float cutoff_dist
) {
    genfire_gridding_2d_cuda(
        reinterpret_cast<const float *>(fk_real_ptr),
        reinterpret_cast<const float *>(fk_imag_ptr),
        reinterpret_cast<const float *>(cos_ptr),
        reinterpret_cast<const float *>(sin_ptr),
        reinterpret_cast<float *>(grid_real_ptr),
        reinterpret_cast<float *>(grid_imag_ptr),
        reinterpret_cast<float *>(weight_ptr),
        bigN, Ny, n_angles, cutoff_dist
    );
}

#endif /* FASTTOMO_CUDA */

/* ── Module ───────────────────────────────────────────────────────────── */

PYBIND11_MODULE(_core, m) {
    m.doc() = "FastTomo compiled C/CUDA operations";

    m.def("forward_project_cpu", &py_forward_project_cpu,
          "Parallel-beam forward projection (CPU). "
          "volume: [Ny,Nz,Nx] float32. angles: [n_angles] radians. "
          "Returns sinogram [n_angles, Ny, N_det].",
          py::arg("volume"), py::arg("angles"), py::arg("N_det") = -1);

    m.def("back_project_cpu", &py_back_project_cpu,
          "Parallel-beam back-projection (CPU). "
          "sinogram: [n_angles,Ny,N_det] float32. angles: [n_angles] radians. "
          "Returns volume [Ny, Nz, Nx].",
          py::arg("sinogram"), py::arg("angles"), py::arg("Nz") = -1);

    m.def("grad_angle_cpu", &py_grad_angle_cpu,
          "Gradient of forward projection w.r.t. tilt angles (CPU). "
          "volume: [Ny,Nz,Nx], grad_sino: [n_angles,Ny,N_det], angles: [n_angles] radians. "
          "Returns [n_angles] gradient in radians.",
          py::arg("volume"), py::arg("grad_sino"), py::arg("angles"), py::arg("N_det") = -1);

    m.def("genfire_gridding_2d_cpu", &py_genfire_gridding_2d_cpu,
          "GENFIRE Fourier gridding (CPU). "
          "fk_real/fk_imag: [n_angles,Ny,bigN] fftshifted 1D FFTs. "
          "cos_a/sin_a: [n_angles]. "
          "Returns (grid_real, grid_imag, weight_grid) each [Ny,bigN,bigN].",
          py::arg("fk_real"), py::arg("fk_imag"),
          py::arg("cos_a"),   py::arg("sin_a"),
          py::arg("cutoff_dist") = 0.5f);

#ifdef FASTTOMO_CUDA
    m.def("forward_project_cuda", &py_forward_project_cuda,
          "Forward projection (CUDA). Accepts device data_ptr() ints.",
          py::arg("volume_ptr"), py::arg("sinogram_ptr"), py::arg("angles"),
          py::arg("Ny"), py::arg("Nz"), py::arg("Nx"),
          py::arg("n_angles"), py::arg("N_det"));

    m.def("back_project_cuda", &py_back_project_cuda,
          "Back-projection (CUDA). Accepts device data_ptr() ints.",
          py::arg("sinogram_ptr"), py::arg("volume_ptr"), py::arg("angles"),
          py::arg("Ny"), py::arg("Nz"), py::arg("Nx"),
          py::arg("n_angles"), py::arg("N_det"));

    m.def("forward_project_rotmat_cuda", &py_forward_project_rotmat_cuda,
          "3D rotation-matrix forward projection (CUDA). "
          "matrices_ptr: device data_ptr() of [n_angles*9] row-major float32 rotation matrices.",
          py::arg("vol_ptr"), py::arg("sino_ptr"), py::arg("matrices_ptr"),
          py::arg("Ny"), py::arg("Nz"), py::arg("Nx"),
          py::arg("n_angles"), py::arg("N_det"));

    m.def("back_project_rotmat_cuda", &py_back_project_rotmat_cuda,
          "3D rotation-matrix back-projection (CUDA). "
          "matrices_ptr: device data_ptr() of [n_angles*9] row-major float32 rotation matrices.",
          py::arg("sino_ptr"), py::arg("vol_ptr"), py::arg("matrices_ptr"),
          py::arg("Ny"), py::arg("Nz"), py::arg("Nx"),
          py::arg("n_angles"), py::arg("N_det"));

    m.def("forward_project_resire_cuda", &py_forward_project_resire_cuda,
          "RESIRE voxel-driven forward projection (CUDA). "
          "matrices_ptr: device data_ptr() of [n_angles*9] row-major float32 matrices. "
          "sino_ptr must point to a zeroed buffer.",
          py::arg("vol_ptr"), py::arg("sino_ptr"), py::arg("matrices_ptr"),
          py::arg("Ny"), py::arg("Nz"), py::arg("Nx"),
          py::arg("n_angles"), py::arg("N_det"));

    m.def("back_project_resire_cuda", &py_back_project_resire_cuda,
          "RESIRE voxel-driven back-projection (CUDA). "
          "matrices_ptr: device data_ptr() of [n_angles*9] row-major float32 matrices. "
          "vol_ptr must point to a zeroed buffer.",
          py::arg("sino_ptr"), py::arg("vol_ptr"), py::arg("matrices_ptr"),
          py::arg("Ny"), py::arg("Nz"), py::arg("Nx"),
          py::arg("n_angles"), py::arg("N_det"));

    m.def("grad_angle_cuda", &py_grad_angle_cuda,
          "Gradient of forward projection w.r.t. tilt angles (CUDA). "
          "vol_ptr/grad_sino_ptr/cos_ptr/sin_ptr/grad_angles_ptr: device data_ptr() ints. "
          "Returns gradient in radians via grad_angles_ptr.",
          py::arg("vol_ptr"), py::arg("grad_sino_ptr"),
          py::arg("cos_ptr"), py::arg("sin_ptr"),
          py::arg("grad_angles_ptr"),
          py::arg("Ny"), py::arg("Nz"), py::arg("Nx"),
          py::arg("n_angles"), py::arg("N_det"));

    m.def("grad_rotmat_resire_cuda", &py_grad_rotmat_resire_cuda,
          "Gradient of RESIRE forward projection w.r.t. rotation matrices (CUDA). "
          "All pointer args are device data_ptr() ints. "
          "grad_matrices_ptr must point to a zeroed [n_angles*9] buffer.",
          py::arg("vol_ptr"), py::arg("grad_sino_ptr"),
          py::arg("matrices_ptr"), py::arg("grad_matrices_ptr"),
          py::arg("Ny"), py::arg("Nz"), py::arg("Nx"),
          py::arg("n_angles"), py::arg("N_det"));

    m.def("genfire_gridding_2d_cuda", &py_genfire_gridding_2d_cuda,
          "GENFIRE Fourier gridding (CUDA). All args are device data_ptr() ints "
          "except bigN/Ny/n_angles/cutoff_dist.",
          py::arg("fk_real_ptr"), py::arg("fk_imag_ptr"),
          py::arg("cos_ptr"),     py::arg("sin_ptr"),
          py::arg("grid_real_ptr"), py::arg("grid_imag_ptr"), py::arg("weight_ptr"),
          py::arg("bigN"), py::arg("Ny"), py::arg("n_angles"),
          py::arg("cutoff_dist") = 0.5f);

    m.attr("cuda_available") = true;
#else
    m.attr("cuda_available") = false;
#endif
}

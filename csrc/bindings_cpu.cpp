#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <vector>
#include <stdexcept>

#include "cpu/ops.h"
#include "bindings_cpu.h"

namespace py = pybind11;
using f32arr = py::array_t<float>;

/* helper: validate that an array is float32 and C-contiguous */
static void check_f32(const f32arr &a, const char *name) {
    if (a.ndim() < 1)
        throw std::invalid_argument(std::string(name) + " must have at least 1 dim");
}

/* precompute cos/sin from a 1-D angles array (radians) — used by CPU path
 * only. CUDA path takes cos/sin device pointers computed by the Python
 * wrapper, so the precompute can stay on the GPU. */
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
    int N_det,              /* detector width; -1 → use Nx */
    int num_threads         /* threads for the OpenMP-parallel outer loops */
) {
    check_f32(volume, "volume");
    if (volume.ndim() != 3)
        throw std::invalid_argument("volume must be 3-D [Ny, Nz, Nx]");

    int Ny       = (int)volume.shape(0);
    int Nz       = (int)volume.shape(1);
    int Nx       = (int)volume.shape(2);
    int n_angles = (int)angles.size();
    if (N_det < 0) N_det = Nx;
    if (num_threads < 1) num_threads = 1;

    std::vector<float> cos_a, sin_a;
    compute_trig(angles, cos_a, sin_a);

    auto sinogram = py::array_t<float>({n_angles, Ny, N_det});
    auto out_buf  = sinogram.request();

    forward_project_cpu(
        static_cast<const float *>(volume.request().ptr),
        static_cast<float *>(out_buf.ptr),
        cos_a.data(), sin_a.data(),
        Ny, Nz, Nx, n_angles, N_det,
        num_threads
    );
    return sinogram;
}

static f32arr py_back_project_cpu(
    const f32arr &sinogram,  /* [n_angles, Ny, N_det] */
    const f32arr &angles,    /* [n_angles] radians */
    int Nz,                  /* reconstruction depth; -1 → use N_det */
    int num_threads
) {
    check_f32(sinogram, "sinogram");
    if (sinogram.ndim() != 3)
        throw std::invalid_argument("sinogram must be 3-D [n_angles, Ny, N_det]");

    int n_angles = (int)sinogram.shape(0);
    int Ny       = (int)sinogram.shape(1);
    int N_det    = (int)sinogram.shape(2);
    int Nx       = N_det;
    if (Nz < 0) Nz = N_det;
    if (num_threads < 1) num_threads = 1;

    std::vector<float> cos_a, sin_a;
    compute_trig(angles, cos_a, sin_a);

    auto volume  = py::array_t<float>({Ny, Nz, Nx});
    auto out_buf = volume.request();

    back_project_cpu(
        static_cast<const float *>(sinogram.request().ptr),
        static_cast<float *>(out_buf.ptr),
        cos_a.data(), sin_a.data(),
        Ny, Nz, Nx, n_angles, N_det,
        num_threads
    );
    return volume;
}

static f32arr py_grad_angle_cpu(
    const f32arr &volume,    /* [Ny, Nz, Nx] */
    const f32arr &grad_sino, /* [n_angles, Ny, N_det] */
    const f32arr &angles,    /* [n_angles] radians */
    int N_det,
    int num_threads
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
    if (num_threads < 1) num_threads = 1;

    std::vector<float> cos_a, sin_a;
    compute_trig(angles, cos_a, sin_a);

    auto grad_angles = py::array_t<float>({n_angles});
    auto out_buf     = grad_angles.request();

    grad_angle_cpu(
        static_cast<const float *>(volume.request().ptr),
        static_cast<const float *>(grad_sino.request().ptr),
        static_cast<float *>(out_buf.ptr),
        cos_a.data(), sin_a.data(),
        Ny, Nz, Nx, n_angles, N_det,
        num_threads
    );
    return grad_angles;
}

static f32arr py_forward_project_3d_cpu(
    const f32arr &volume,    /* [Ny, Nz, Nx] */
    const f32arr &matrices,  /* [n_angles, 9] or [n_angles*9] row-major */
    int N_det,               /* detector width; -1 → use Nx */
    int num_threads          /* threads for the OpenMP-parallel inner loops */
) {
    check_f32(volume, "volume");
    check_f32(matrices, "matrices");
    if (volume.ndim() != 3)
        throw std::invalid_argument("volume must be 3-D [Ny, Nz, Nx]");
    if ((int)matrices.size() % 9 != 0)
        throw std::invalid_argument("matrices size must be multiple of 9");

    int Ny       = (int)volume.shape(0);
    int Nz       = (int)volume.shape(1);
    int Nx       = (int)volume.shape(2);
    int n_angles = (int)matrices.size() / 9;
    if (N_det < 0) N_det = Nx;
    if (num_threads < 1) num_threads = 1;

    auto sinogram = py::array_t<float>({n_angles, Ny, N_det});
    auto out_buf  = sinogram.request();

    forward_project_3d_cpu(
        static_cast<const float *>(volume.request().ptr),
        static_cast<float *>(out_buf.ptr),
        static_cast<const float *>(matrices.request().ptr),
        Ny, Nz, Nx, n_angles, N_det,
        num_threads
    );
    return sinogram;
}

static f32arr py_back_project_3d_cpu(
    const f32arr &sinogram,  /* [n_angles, Ny, N_det] */
    const f32arr &matrices,  /* [n_angles, 9] or [n_angles*9] */
    int Nz,                  /* reconstruction depth; -1 → use N_det */
    int num_threads
) {
    check_f32(sinogram, "sinogram");
    check_f32(matrices, "matrices");
    if (sinogram.ndim() != 3)
        throw std::invalid_argument("sinogram must be 3-D [n_angles, Ny, N_det]");

    int n_angles = (int)sinogram.shape(0);
    int Ny       = (int)sinogram.shape(1);
    int N_det    = (int)sinogram.shape(2);
    int Nx       = N_det;
    if (Nz < 0) Nz = N_det;
    if (num_threads < 1) num_threads = 1;
    if ((int)matrices.size() != n_angles * 9)
        throw std::invalid_argument("matrices size must equal n_angles * 9");

    auto volume  = py::array_t<float>({Ny, Nz, Nx});
    auto out_buf = volume.request();

    back_project_3d_cpu(
        static_cast<const float *>(sinogram.request().ptr),
        static_cast<float *>(out_buf.ptr),
        static_cast<const float *>(matrices.request().ptr),
        Ny, Nz, Nx, n_angles, N_det,
        num_threads
    );
    return volume;
}

static f32arr py_grad_rotmat_cpu(
    const f32arr &volume,    /* [Ny, Nz, Nx] */
    const f32arr &grad_sino, /* [n_angles, Ny, N_det] */
    const f32arr &matrices,  /* [n_angles, 9] or [n_angles*9] */
    int num_threads
) {
    check_f32(volume, "volume");
    check_f32(grad_sino, "grad_sino");
    check_f32(matrices, "matrices");
    if (volume.ndim() != 3)
        throw std::invalid_argument("volume must be 3-D [Ny, Nz, Nx]");
    if (grad_sino.ndim() != 3)
        throw std::invalid_argument("grad_sino must be 3-D [n_angles, Ny, N_det]");

    int Ny       = (int)volume.shape(0);
    int Nz       = (int)volume.shape(1);
    int Nx       = (int)volume.shape(2);
    int n_angles = (int)grad_sino.shape(0);
    int N_det    = (int)grad_sino.shape(2);
    if (num_threads < 1) num_threads = 1;
    if ((int)matrices.size() != n_angles * 9)
        throw std::invalid_argument("matrices size must equal n_angles * 9");

    auto grad_mats = py::array_t<float>({n_angles, 9});
    auto out_buf   = grad_mats.request();

    grad_rotmat_cpu(
        static_cast<const float *>(volume.request().ptr),
        static_cast<const float *>(grad_sino.request().ptr),
        static_cast<const float *>(matrices.request().ptr),
        static_cast<float *>(out_buf.ptr),
        Ny, Nz, Nx, n_angles, N_det,
        num_threads
    );
    return grad_mats;
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

void register_cpu(py::module_ &m) {
    m.def("forward_project_cpu", &py_forward_project_cpu,
          "Parallel-beam forward projection (CPU). "
          "volume: [Ny,Nz,Nx] float32. angles: [n_angles] radians. "
          "num_threads: OpenMP thread count (default 1 — serial). "
          "Returns sinogram [n_angles, Ny, N_det].",
          py::arg("volume"), py::arg("angles"), py::arg("N_det") = -1,
          py::arg("num_threads") = 1);

    m.def("back_project_cpu", &py_back_project_cpu,
          "Parallel-beam back-projection (CPU). "
          "sinogram: [n_angles,Ny,N_det] float32. angles: [n_angles] radians. "
          "num_threads: OpenMP thread count (default 1 — serial). "
          "Returns volume [Ny, Nz, Nx].",
          py::arg("sinogram"), py::arg("angles"), py::arg("Nz") = -1,
          py::arg("num_threads") = 1);

    m.def("grad_angle_cpu", &py_grad_angle_cpu,
          "Gradient of forward projection w.r.t. tilt angles (CPU). "
          "volume: [Ny,Nz,Nx], grad_sino: [n_angles,Ny,N_det], angles: [n_angles] radians. "
          "num_threads: OpenMP thread count (default 1 — serial). "
          "Returns [n_angles] gradient in radians.",
          py::arg("volume"), py::arg("grad_sino"), py::arg("angles"), py::arg("N_det") = -1,
          py::arg("num_threads") = 1);

    m.def("forward_project_3d_cpu", &py_forward_project_3d_cpu,
          "3D rotation-matrix forward projection (CPU). "
          "volume: [Ny,Nz,Nx]. matrices: [n_angles,9] or [n_angles*9] row-major rotation matrices. "
          "num_threads: OpenMP thread count for the inner loops (default 1 — serial). "
          "Returns sinogram [n_angles, Ny, N_det].",
          py::arg("volume"), py::arg("matrices"), py::arg("N_det") = -1,
          py::arg("num_threads") = 1);

    m.def("back_project_3d_cpu", &py_back_project_3d_cpu,
          "3D rotation-matrix back-projection (CPU). "
          "sinogram: [n_angles,Ny,N_det]. matrices: [n_angles,9] row-major. "
          "num_threads: OpenMP thread count (default 1 — serial). "
          "Returns volume [Ny, Nz, Nx].",
          py::arg("sinogram"), py::arg("matrices"), py::arg("Nz") = -1,
          py::arg("num_threads") = 1);

    m.def("grad_rotmat_cpu", &py_grad_rotmat_cpu,
          "Gradient of 3D rotmat forward projection w.r.t. rotation matrix entries (CPU). "
          "num_threads: OpenMP thread count (default 1 — serial). "
          "Returns [n_angles, 9] gradient.",
          py::arg("volume"), py::arg("grad_sino"), py::arg("matrices"),
          py::arg("num_threads") = 1);

    m.def("genfire_gridding_2d_cpu", &py_genfire_gridding_2d_cpu,
          "GENFIRE Fourier gridding (CPU). "
          "fk_real/fk_imag: [n_angles,Ny,bigN] fftshifted 1D FFTs. "
          "cos_a/sin_a: [n_angles]. "
          "Returns (grid_real, grid_imag, weight_grid) each [Ny,bigN,bigN].",
          py::arg("fk_real"), py::arg("fk_imag"),
          py::arg("cos_a"),   py::arg("sin_a"),
          py::arg("cutoff_dist") = 0.5f);
}

/* ── fasttomo/csrc/bindings_cpu.cpp ──────────────────────────────────────────
 * pybind11 bindings for the CPU kernels (csrc/cpu/).
 *
 * Same convention as bindings_cuda.cpp: every array argument is the integer
 * `data_ptr()` of a torch tensor the Python caller allocated, followed by the
 * dims. The caller (fasttomo/autograd.py `_launch`, the MBIR/GENFIRE code in
 * algorithms.py) guarantees each buffer is C-contiguous float32 (int32 for
 * the MBIR permutation), sized for the dims, and zeroed where noted. No numpy
 * round trip, no silent copy, and the GIL is released while the C code runs.
 *
 * Angles arrive as precomputed cos/sin tables, like on CUDA; the trig is done
 * once in Python for both backends.
 */

#include <pybind11/pybind11.h>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "cpu/ops.h"
#include "bindings_cpu.h"

namespace py = pybind11;

template <class T>
static inline T *P(std::intptr_t p) { return reinterpret_cast<T *>(p); }

static inline int threads_or_one(int n) { return n < 1 ? 1 : n; }

void register_cpu(py::module_ &m) {
    using gil_release = py::call_guard<py::gil_scoped_release>;

    m.def("forward_project_parallel_cpu",
          [](std::intptr_t vol, std::intptr_t sino,
             std::intptr_t cos_a, std::intptr_t sin_a,
             int Ny, int Nz, int Nx, int n_angles, int N_det, int num_threads) {
              forward_project_parallel_cpu(
                  P<const float>(vol), P<float>(sino),
                  P<const float>(cos_a), P<const float>(sin_a),
                  Ny, Nz, Nx, n_angles, N_det, threads_or_one(num_threads));
          }, gil_release(),
          "Single-axis (parallel) forward projection (CPU). Writes every "
          "element of sino [n_angles, Ny, N_det] from vol [Ny, Nz, Nx].",
          py::arg("vol_ptr"), py::arg("sino_ptr"),
          py::arg("cos_ptr"), py::arg("sin_ptr"),
          py::arg("Ny"), py::arg("Nz"), py::arg("Nx"),
          py::arg("n_angles"), py::arg("N_det"), py::arg("num_threads") = 1);

    m.def("back_project_parallel_cpu",
          [](std::intptr_t sino, std::intptr_t vol,
             std::intptr_t cos_a, std::intptr_t sin_a,
             int Ny, int Nz, int Nx, int n_angles, int N_det, int num_threads) {
              back_project_parallel_cpu(
                  P<const float>(sino), P<float>(vol),
                  P<const float>(cos_a), P<const float>(sin_a),
                  Ny, Nz, Nx, n_angles, N_det, threads_or_one(num_threads));
          }, gil_release(),
          "Single-axis (parallel) back-projection (CPU), exact ray-driven "
          "transpose of the forward. Overwrites vol [Ny, Nz, Nx].",
          py::arg("sino_ptr"), py::arg("vol_ptr"),
          py::arg("cos_ptr"), py::arg("sin_ptr"),
          py::arg("Ny"), py::arg("Nz"), py::arg("Nx"),
          py::arg("n_angles"), py::arg("N_det"), py::arg("num_threads") = 1);

    m.def("grad_angle_cpu",
          [](std::intptr_t vol, std::intptr_t grad_sino,
             std::intptr_t cos_a, std::intptr_t sin_a, std::intptr_t grad_angles,
             int Ny, int Nz, int Nx, int n_angles, int N_det, int num_threads) {
              grad_angle_cpu(
                  P<const float>(vol), P<const float>(grad_sino), P<float>(grad_angles),
                  P<const float>(cos_a), P<const float>(sin_a),
                  Ny, Nz, Nx, n_angles, N_det, threads_or_one(num_threads));
          }, gil_release(),
          "Gradient of the parallel forward projection w.r.t. the tilt angles "
          "(CPU), in radians. Overwrites grad_angles [n_angles].",
          py::arg("vol_ptr"), py::arg("grad_sino_ptr"),
          py::arg("cos_ptr"), py::arg("sin_ptr"), py::arg("grad_angles_ptr"),
          py::arg("Ny"), py::arg("Nz"), py::arg("Nx"),
          py::arg("n_angles"), py::arg("N_det"), py::arg("num_threads") = 1);

    m.def("forward_project_rotmat_cpu",
          [](std::intptr_t vol, std::intptr_t sino, std::intptr_t mats,
             int Ny, int Nz, int Nx, int n_angles, int N_det, int num_threads) {
              forward_project_rotmat_cpu(
                  P<const float>(vol), P<float>(sino), P<const float>(mats),
                  Ny, Nz, Nx, n_angles, N_det, threads_or_one(num_threads));
          }, gil_release(),
          "Rotation-matrix forward projection (CPU). mats: [n_angles*9] "
          "row-major. Writes every element of sino [n_angles, Ny, N_det].",
          py::arg("vol_ptr"), py::arg("sino_ptr"), py::arg("mats_ptr"),
          py::arg("Ny"), py::arg("Nz"), py::arg("Nx"),
          py::arg("n_angles"), py::arg("N_det"), py::arg("num_threads") = 1);

    m.def("back_project_rotmat_cpu",
          [](std::intptr_t sino, std::intptr_t vol, std::intptr_t mats,
             int Ny, int Nz, int Nx, int n_angles, int N_det, int num_threads) {
              back_project_rotmat_cpu(
                  P<const float>(sino), P<float>(vol), P<const float>(mats),
                  Ny, Nz, Nx, n_angles, N_det, threads_or_one(num_threads));
          }, gil_release(),
          "Rotation-matrix back-projection (CPU). Overwrites vol [Ny, Nz, Nx].",
          py::arg("sino_ptr"), py::arg("vol_ptr"), py::arg("mats_ptr"),
          py::arg("Ny"), py::arg("Nz"), py::arg("Nx"),
          py::arg("n_angles"), py::arg("N_det"), py::arg("num_threads") = 1);

    m.def("grad_rotmat_cpu",
          [](std::intptr_t vol, std::intptr_t grad_sino,
             std::intptr_t mats, std::intptr_t grad_mats,
             int Ny, int Nz, int Nx, int n_angles, int N_det, int num_threads) {
              grad_rotmat_cpu(
                  P<const float>(vol), P<const float>(grad_sino),
                  P<const float>(mats), P<float>(grad_mats),
                  Ny, Nz, Nx, n_angles, N_det, threads_or_one(num_threads));
          }, gil_release(),
          "Gradient of the rotmat forward projection w.r.t. the matrix "
          "entries (CPU). Overwrites grad_mats [n_angles*9].",
          py::arg("vol_ptr"), py::arg("grad_sino_ptr"),
          py::arg("mats_ptr"), py::arg("grad_mats_ptr"),
          py::arg("Ny"), py::arg("Nz"), py::arg("Nx"),
          py::arg("n_angles"), py::arg("N_det"), py::arg("num_threads") = 1);

    m.def("genfire_gridding_2d_cpu",
          [](std::intptr_t fk_real, std::intptr_t fk_imag,
             std::intptr_t cos_a, std::intptr_t sin_a,
             std::intptr_t grid_real, std::intptr_t grid_imag, std::intptr_t weight,
             int bigN, int Ny, int n_angles, float cutoff_dist) {
              genfire_gridding_2d_cpu(
                  P<const float>(fk_real), P<const float>(fk_imag),
                  P<const float>(cos_a), P<const float>(sin_a),
                  P<float>(grid_real), P<float>(grid_imag), P<float>(weight),
                  bigN, Ny, n_angles, cutoff_dist);
          }, gil_release(),
          "GENFIRE Fourier gridding (CPU). fk_real/fk_imag: [n_angles, Ny, bigN] "
          "fftshifted 1D FFTs. Accumulates into grid_real/grid_imag/weight "
          "[Ny, bigN, bigN], which the caller must zero.",
          py::arg("fk_real_ptr"), py::arg("fk_imag_ptr"),
          py::arg("cos_ptr"),     py::arg("sin_ptr"),
          py::arg("grid_real_ptr"), py::arg("grid_imag_ptr"), py::arg("weight_ptr"),
          py::arg("bigN"), py::arg("Ny"), py::arg("n_angles"),
          py::arg("cutoff_dist") = 0.5f);

    /* ── MBIR ─────────────────────────────────────────────────────────── */

    m.def("mbir_forward_project_cpu",
          [](std::intptr_t vol, std::intptr_t sino,
             std::intptr_t cos_a, std::intptr_t sin_a,
             int Ny, int Nz, int Nx, int n_angles, int N_det, int num_threads) {
              mbir_forward_project_cpu(
                  P<const float>(vol), P<float>(sino),
                  P<const float>(cos_a), P<const float>(sin_a),
                  Ny, Nz, Nx, n_angles, N_det, threads_or_one(num_threads));
          }, gil_release(),
          "MBIR voxel-driven bilinear forward projection (CPU), the column-A "
          "model of mbir_icd_sweep_cpu. Overwrites sino [n_angles, Ny, N_det].",
          py::arg("vol_ptr"), py::arg("sino_ptr"),
          py::arg("cos_ptr"), py::arg("sin_ptr"),
          py::arg("Ny"), py::arg("Nz"), py::arg("Nx"),
          py::arg("n_angles"), py::arg("N_det"), py::arg("num_threads") = 1);

    m.def("mbir_icd_sweep_cpu",
          [](std::intptr_t vol, std::intptr_t e_sino, std::intptr_t vol_snapshot,
             std::intptr_t lambdas, std::intptr_t perm,
             std::intptr_t cos_a, std::intptr_t sin_a,
             int Ny, int Nz, int Nx, int n_angles, int N_det,
             float qggmrf_p, float qggmrf_q, float qggmrf_c, float sigma_f,
             float huber_T, float huber_delta, float sigma, int num_threads) {
              mbir_icd_sweep_cpu(
                  P<float>(vol), P<float>(e_sino),
                  P<const float>(vol_snapshot), P<const int>(perm),
                  P<const float>(cos_a), P<const float>(sin_a),
                  P<const float>(lambdas),
                  Ny, Nz, Nx, n_angles, N_det,
                  qggmrf_p, qggmrf_q, qggmrf_c, sigma_f,
                  huber_T, huber_delta, sigma,
                  threads_or_one(num_threads));
          }, gil_release(),
          "ICD sweep of one MBIR-AM-PE outer iteration (CPU). Mutates vol and "
          "e_sino in place; the offset and variance updates are done in Python "
          "(_mbir_update_offset_variance), as for CUDA. lambdas_ptr = 0 means "
          "uniform weights. perm_ptr: [Ny, Nz*Nx] int32 per-slice permutation.",
          py::arg("vol_ptr"), py::arg("e_sino_ptr"), py::arg("vol_snapshot_ptr"),
          py::arg("lambdas_ptr"), py::arg("perm_ptr"),
          py::arg("cos_ptr"), py::arg("sin_ptr"),
          py::arg("Ny"), py::arg("Nz"), py::arg("Nx"),
          py::arg("n_angles"), py::arg("N_det"),
          py::arg("qggmrf_p"), py::arg("qggmrf_q"),
          py::arg("qggmrf_c"), py::arg("sigma_f"),
          py::arg("huber_T"), py::arg("huber_delta"),
          py::arg("sigma"),
          py::arg("num_threads") = 1);
}

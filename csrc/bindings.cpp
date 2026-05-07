#include "bindings_cpu.h"
#include "bindings_cuda.h"

PYBIND11_MODULE(_core, m) {
    m.doc() = "FastTomo compiled C/CUDA operations";
    register_cpu(m);
    register_cuda(m);
}

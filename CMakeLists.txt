cmake_minimum_required(VERSION 3.18)

# Tri-state CUDA selection: AUTO (detect), ON (require), OFF (CPU-only).
# Override at configure time via -DFASTTOMO_USE_CUDA=ON|OFF|AUTO
# or, with pip/uv: --config-settings=cmake.define.FASTTOMO_USE_CUDA=OFF
set(FASTTOMO_USE_CUDA "AUTO" CACHE STRING "Enable CUDA backend: ON, OFF, or AUTO")
set_property(CACHE FASTTOMO_USE_CUDA PROPERTY STRINGS AUTO ON OFF)

if(FASTTOMO_USE_CUDA STREQUAL "AUTO")
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        set(FASTTOMO_CUDA ON)
        message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}")
    else()
        set(FASTTOMO_CUDA OFF)
        message(STATUS "No CUDA compiler found — building CPU-only")
    endif()
elseif(FASTTOMO_USE_CUDA)
    set(FASTTOMO_CUDA ON)
    message(STATUS "CUDA backend forced ON via FASTTOMO_USE_CUDA")
else()
    set(FASTTOMO_CUDA OFF)
    message(STATUS "CUDA backend forced OFF via FASTTOMO_USE_CUDA")
endif()

if(FASTTOMO_CUDA)
    project(fasttomo LANGUAGES C CXX CUDA)
else()
    project(fasttomo LANGUAGES C CXX)
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

set(CORE_SOURCES
    csrc/bindings.cpp
    csrc/bindings_cpu.cpp
    csrc/bindings_cuda.cpp
    csrc/cpu/ops.c
)

if(FASTTOMO_CUDA)
    list(APPEND CORE_SOURCES csrc/cuda/ops.cu csrc/cuda/texture_cache.cu)
endif()

pybind11_add_module(_core ${CORE_SOURCES})
target_include_directories(_core PRIVATE csrc)
target_link_libraries(_core PRIVATE OpenMP::OpenMP_C)

if(FASTTOMO_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_compile_definitions(_core PRIVATE FASTTOMO_CUDA)
    target_link_libraries(_core PRIVATE CUDA::cudart)
    # Force shared libcudart so the symbol resolves to the libcudart already
    # loaded by PyTorch at import time. This is what enables one CUDA wheel to
    # work across PyTorch's CUDA versions; it also avoids bundling cudart_static
    # and cudadevrt into the .so for source builds.
    set_target_properties(_core PROPERTIES
        CUDA_STANDARD 14
        CUDA_RUNTIME_LIBRARY Shared
    )
    # Override at configure time via -DCMAKE_CUDA_ARCHITECTURES="80;86"
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 70 80 86 120)
    endif()
endif()

target_compile_options(_core PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-O3 -march=native>
)

install(TARGETS _core DESTINATION fasttomo)

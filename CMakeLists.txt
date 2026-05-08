cmake_minimum_required(VERSION 3.18)

# Pip-installed CUDA 13 toolkit detection.
# `nvidia-cuda-nvcc` (and friends) install at
# <env>/lib/pythonX.Y/site-packages/nvidia/cu13/{bin,include,lib,nvvm};
# CMake's find_package(CUDAToolkit) does not search site-packages, so seed
# CUDAToolkit_ROOT / CMAKE_CUDA_COMPILER before language detection runs.
if(NOT DEFINED CUDAToolkit_ROOT AND NOT DEFINED CMAKE_CUDA_COMPILER)
    if(NOT DEFINED Python_EXECUTABLE)
        find_program(Python_EXECUTABLE NAMES python3 python)
    endif()
    if(Python_EXECUTABLE)
        execute_process(
            COMMAND ${Python_EXECUTABLE} -c [==[
import os, sys
try:
    import nvidia
except ImportError:
    sys.exit(0)
for base in nvidia.__path__:
    cu13 = os.path.join(base, 'cu13')
    if os.path.isfile(os.path.join(cu13, 'bin', 'nvcc')):
        print(cu13)
        break
]==]
            OUTPUT_VARIABLE _PIP_CUDA_ROOT
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(_PIP_CUDA_ROOT)
            set(_PIP_CUDA_ROOT_USABLE TRUE)
            # cu13 wheels ship versioned libs only (libcudart.so.13); seed the
            # unversioned symlink that CMake's find_library expects.
            file(GLOB _cudart_versioned "${_PIP_CUDA_ROOT}/lib/libcudart.so.*")
            if(_cudart_versioned AND NOT EXISTS "${_PIP_CUDA_ROOT}/lib/libcudart.so")
                list(GET _cudart_versioned 0 _cudart_first)
                file(
                    CREATE_LINK "${_cudart_first}" "${_PIP_CUDA_ROOT}/lib/libcudart.so"
                    SYMBOLIC
                    RESULT _cudart_link_result
                )
                if(NOT _cudart_link_result STREQUAL "0")
                    set(_PIP_CUDA_ROOT_USABLE FALSE)
                    message(WARNING "Unable to create libcudart.so link in ${_PIP_CUDA_ROOT}/lib: ${_cudart_link_result}")
                endif()
            endif()
            # nvcc's device link step searches `<root>/lib64/`; the wheel uses
            # `lib/`. Add a sibling symlink so -lcudadevrt etc. resolve.
            if(_PIP_CUDA_ROOT_USABLE AND NOT EXISTS "${_PIP_CUDA_ROOT}/lib64")
                file(
                    CREATE_LINK "lib" "${_PIP_CUDA_ROOT}/lib64"
                    SYMBOLIC
                    RESULT _lib64_link_result
                )
                if(NOT _lib64_link_result STREQUAL "0")
                    set(_PIP_CUDA_ROOT_USABLE FALSE)
                    message(WARNING "Unable to create lib64 link in ${_PIP_CUDA_ROOT}: ${_lib64_link_result}")
                endif()
            endif()
            if(_PIP_CUDA_ROOT_USABLE)
                message(STATUS "Detected pip-installed CUDA toolkit at ${_PIP_CUDA_ROOT}")
                set(CUDAToolkit_ROOT "${_PIP_CUDA_ROOT}" CACHE PATH "")
                set(CMAKE_CUDA_COMPILER "${_PIP_CUDA_ROOT}/bin/nvcc" CACHE FILEPATH "")
            else()
                message(STATUS "Skipping pip CUDA auto-detect because toolkit links could not be prepared")
            endif()
        endif()
    endif()
endif()

# Conda env CUDA toolkit fallback. If no pip wheel was found and the user is
# inside a conda env that ships nvcc (e.g. `conda activate FastTomo`), use
# that. Without this, cron jobs that don't run conda activate silently get a
# CPU-only build.
if(NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED CUDAToolkit_ROOT)
    set(_CONDA_ENV_ROOT "$ENV{CONDA_PREFIX}")
    if(_CONDA_ENV_ROOT AND EXISTS "${_CONDA_ENV_ROOT}/bin/nvcc")
        message(STATUS "Detected conda env CUDA toolkit at ${_CONDA_ENV_ROOT}")
        set(CUDAToolkit_ROOT "${_CONDA_ENV_ROOT}" CACHE PATH "")
        set(CMAKE_CUDA_COMPILER "${_CONDA_ENV_ROOT}/bin/nvcc" CACHE FILEPATH "")
    endif()
endif()

# Tri-state CUDA selection: AUTO (detect), ON (require), OFF (CPU-only).
# Override at configure time via -DFASTTOMO_USE_CUDA=ON|OFF|AUTO
# or, with pip/uv: --config-settings=cmake.define.FASTTOMO_USE_CUDA=OFF
set(FASTTOMO_USE_CUDA "AUTO" CACHE STRING "Enable CUDA backend: ON, OFF, or AUTO")
set_property(CACHE FASTTOMO_USE_CUDA PROPERTY STRINGS AUTO ON OFF)

if(FASTTOMO_USE_CUDA STREQUAL "AUTO")
    # Probe whether nvcc is reachable from the user's environment. If it is
    # but `check_language(CUDA)` still fails, we treat that as a real
    # toolchain error and abort — silently producing a CPU-only wheel when
    # the user clearly *wants* CUDA was the cause of past build incidents.
    set(_FASTTOMO_NVCC_VISIBLE FALSE)
    if(DEFINED CMAKE_CUDA_COMPILER OR DEFINED CUDAToolkit_ROOT
       OR DEFINED ENV{CUDA_HOME} OR DEFINED ENV{CUDA_PATH})
        set(_FASTTOMO_NVCC_VISIBLE TRUE)
    elseif(DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}/bin/nvcc")
        set(_FASTTOMO_NVCC_VISIBLE TRUE)
    else()
        find_program(_FASTTOMO_NVCC_PATH nvcc)
        if(_FASTTOMO_NVCC_PATH)
            set(_FASTTOMO_NVCC_VISIBLE TRUE)
        endif()
    endif()

    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        set(FASTTOMO_CUDA ON)
        message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}")
    elseif(_FASTTOMO_NVCC_VISIBLE)
        message(FATAL_ERROR
            "FastTomo: nvcc was detected on this system but CMake could not "
            "enable the CUDA language. This usually means a host-compiler / "
            "CUDA toolchain mismatch (e.g. unsupported GCC version, missing "
            "cudart, broken conda env).\n"
            "    To diagnose: re-run with "
            "--config-settings=cmake.define.CMAKE_VERBOSE_MAKEFILE=ON\n"
            "    To bypass and build CPU-only: pass "
            "--config-settings=cmake.define.FASTTOMO_USE_CUDA=OFF")
    else()
        set(FASTTOMO_CUDA OFF)
        message(WARNING
            "==================================================================\n"
            "  FastTomo: no CUDA toolchain detected — building CPU-only.\n"
            "  To enable CUDA: install CUDA 13.0 (e.g. nvidia-cuda-nvcc==13.0.*\n"
            "  via pip, or a conda env shipping nvcc 13.0), then re-run\n"
            "  pip install with --config-settings=cmake.define.FASTTOMO_USE_CUDA=ON\n"
            "==================================================================")
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

# Host C++ compiler floor: GCC >= 10 (NVIDIA's CUDA 13 install guide
# supports GCC 6-15, but FastTomo standardises on GCC >= 10 — older
# toolchains miss C++17 polish and aren't tested).
if(FASTTOMO_CUDA AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
   AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 10)
    message(FATAL_ERROR
        "FastTomo requires GCC >= 10 for CUDA builds (found "
        "${CMAKE_CXX_COMPILER_VERSION}). Install a newer GCC or pass "
        "-DCMAKE_CXX_COMPILER=/path/to/g++-N.")
endif()

set(CORE_SOURCES
    csrc/bindings.cpp
    csrc/bindings_cpu.cpp
    csrc/bindings_cuda.cpp
    csrc/cpu/projection_2d.c
    csrc/cpu/projection_3d.c
    csrc/cpu/genfire.c
)

if(FASTTOMO_CUDA)
    list(APPEND CORE_SOURCES
        csrc/cuda/projection_2d.cu
        csrc/cuda/projection_3d.cu
        csrc/cuda/projection_resire.cu
        csrc/cuda/genfire.cu
        csrc/cuda/grad_angle.cu
        csrc/cuda/tv_loss.cu
        csrc/cuda/texture_cache.cu
    )
endif()

pybind11_add_module(_core ${CORE_SOURCES})
target_include_directories(_core PRIVATE csrc)
target_link_libraries(_core PRIVATE OpenMP::OpenMP_C)

if(FASTTOMO_CUDA)
    find_package(CUDAToolkit REQUIRED)
    # PyTorch 2.11 wheels are built only against CUDA 13.0 — pin the toolkit
    # to that exact (major, minor) so a 13.1+ nvcc doesn't silently produce
    # a runtime that mismatches torch's libcudart at import time.
    if(NOT (CUDAToolkit_VERSION_MAJOR EQUAL 13 AND CUDAToolkit_VERSION_MINOR EQUAL 0))
        message(FATAL_ERROR
            "FastTomo requires CUDA exactly 13.0 (found ${CUDAToolkit_VERSION}). "
            "PyTorch 2.11 ships against CUDA 13.0 only. Install "
            "nvidia-cuda-nvcc==13.0.* or activate a conda env shipping "
            "nvcc 13.0.")
    endif()
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

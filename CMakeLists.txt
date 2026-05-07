cmake_minimum_required(VERSION 3.18)

# Detect CUDA before project() so enable_language works correctly
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    project(fasttomo LANGUAGES C CXX CUDA)
    set(FASTTOMO_CUDA ON)
    message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}")
else()
    project(fasttomo LANGUAGES C CXX)
    set(FASTTOMO_CUDA OFF)
    message(STATUS "No CUDA compiler found — building CPU-only")
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

set(CORE_SOURCES
    csrc/bindings.cpp
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
    set_target_properties(_core PROPERTIES
        CUDA_STANDARD 14
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

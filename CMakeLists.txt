cmake_minimum_required(VERSION 3.24)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CUDACXX "/usr/local/cuda/bin/nvcc")
project(tiny-vllm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 120)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_CUDA_FLAGS_RELEASE "-O2")

set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -DDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -DDEBUG")

find_package(CUDAToolkit REQUIRED)


# HTTP server
# FetchContent_Declare(
#     httplib
#     GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
#     GIT_TAG        v0.18.3
# )
# FetchContent_MakeAvailable(httplib)

add_executable(tiny-vllm
    src/main.cpp
    src/kernels.cu
    # src/model.cpp
    # src/attention.cu
    # src/kernels.cu
    # src/kv_cache.cpp
    # src/sampler.cpp
)

target_include_directories(tiny-vllm PRIVATE src)
target_include_directories(tiny-vllm PRIVATE include)

target_link_libraries(tiny-vllm PRIVATE
    CUDA::cublas
    CUDA::cudart
)


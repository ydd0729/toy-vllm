cmake_minimum_required(VERSION 3.24)
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

add_executable(tiny-vllm
    src/main.cpp
    src/kernels.cu
)

target_include_directories(tiny-vllm PRIVATE src)
target_include_directories(tiny-vllm PRIVATE include)

target_link_libraries(tiny-vllm PRIVATE
    CUDA::cublas
    CUDA::cudart
)


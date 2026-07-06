cmake_minimum_required(VERSION 3.24)
project(toy-vllm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
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

set(JSON_PATH ${CMAKE_SOURCE_DIR}/third_party/json)
set(TOKENIZER_CPP_PATH ${CMAKE_SOURCE_DIR}/third_party/tokenizers-cpp)
set(CLI11_PATH ${CMAKE_SOURCE_DIR}/third_party/CLI11)

add_subdirectory(${JSON_PATH} json EXCLUDE_FROM_ALL)
add_subdirectory(${TOKENIZER_CPP_PATH} tokenizers EXCLUDE_FROM_ALL)
add_subdirectory(${CLI11_PATH} CLI11 EXCLUDE_FROM_ALL)


add_executable(toy-vllm
    src/main.cpp
    src/kernels.cu
    src/weights.cpp
    src/inference.cpp
)

target_include_directories(toy-vllm PRIVATE include)

# 源文件是 UTF-8（含中文注释）且无 BOM；告诉 MSVC 按 UTF-8 解析，
# 否则简中系统按 GBK 读，汉字尾字节遇 0x5C(\) 会被当续行符吞掉下一行代码。
# .cu 经 nvcc 用 -Xcompiler 把该开关转给宿主 cl.exe。
if(MSVC)
    target_compile_options(toy-vllm PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    )
endif()

target_link_libraries(toy-vllm PRIVATE
    CUDA::cublas
    CUDA::cudart
    tokenizers_cpp
    nlohmann_json::nlohmann_json
    CLI11::CLI11
)


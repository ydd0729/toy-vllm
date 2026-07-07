cmake_minimum_required(VERSION 4.3.3)
project(toy-vllm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 120)

# 单配置生成器（Ninja、Unix Makefiles）在配置期就固定一种构建类型。若用户既没走
# --preset、也没传 -DCMAKE_BUILD_TYPE，该变量为空——空既不套 Release 默认（-O2）也不套
# Debug 默认（-g/-G），会得到「无优化、无调试信息」的构建。这里兜底默认 Release，保证
# 至少带 -O2。（走 CMakePresets 的 debug/release preset 时该变量已被显式设置，此处不触发；
# 多配置生成器如 Visual Studio 忽略 CMAKE_BUILD_TYPE，配置在 build 时用 --config 选。）
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

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

# DEBUG 宏：可移植写法，CMake 按编译器翻译为 -DDEBUG 或 /DDEBUG；仅 Debug 配置定义。
target_compile_definitions(toy-vllm PRIVATE $<$<CONFIG:Debug>:DEBUG>)

# -G 是 nvcc 专有开关：生成 device（GPU kernel）端调试信息，并关闭 device 代码优化，
# 使 cuda-gdb / Nsight 能进 kernel 单步、看变量与源码行号。代价是 GPU 代码显著变慢，
# 所以只在 Debug 用。CMake 的 CUDA Debug 默认只给 host 端调试（-g 及 -Xcompiler 的 /Zi /Od），
# 不含 -G，故需在此手动追加。
target_compile_options(toy-vllm PRIVATE
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>)

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


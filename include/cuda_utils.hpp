#pragma once
#include <iostream>
#include "config.hpp"

// do { ... } while (0) 是多语句宏的标准惯用法，让宏在任何语法位置都表现得像“一条语句”：
//   - 若裸写多条语句，放进无括号的 if 里会有语句漏到 if 外，逻辑出错；
//   - 若只用 { } 包，调用处结尾的分号会变成空语句，导致 if/else 之间 “else without a previous if”
//   编译错误。
// do{}while(0) 整体是一条需要分号收尾的语句，同时解决这两点；while(0) 保证只执行一次、零开销，
// 且把 status_ 限制在自身作用域内，避免与外部变量重名。

#define CUBLAS_CHECK(call)                                                                         \
    do                                                                                             \
    {                                                                                              \
        cublasStatus_t status_ = (call);                                                           \
        if (status_ != CUBLAS_STATUS_SUCCESS)                                                      \
        {                                                                                          \
            std::cerr << "cuBLAS error " << cublasGetStatusName(status_) << " at " << __FILE__     \
                      << ":" << __LINE__ << "\n";                                                  \
            throw std::runtime_error("cuBLAS call failed");                                        \
        }                                                                                          \
    } while (0)

#define CUDA_CHECK(call)                                                                           \
    do                                                                                             \
    {                                                                                              \
        cudaError_t err_ = (call);                                                                 \
        if (err_ != cudaSuccess)                                                                   \
        {                                                                                          \
            std::cerr << "CUDA error " << cudaGetErrorName(err_) << " ("                           \
                      << cudaGetErrorString(err_) << ")"                                           \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";                            \
            throw std::runtime_error("CUDA call failed");                                          \
        }                                                                                          \
    } while (0)

// 用于析构函数等 noexcept 语境的清理调用：只记录错误、绝不抛异常。
// 析构函数默认 noexcept，若从中抛异常会直接 std::terminate；而 teardown 阶段
// 的 cudaFree 失败通常也无从恢复，记录即可。
#define CUDA_CHECK_NOTHROW(call)                                                                   \
    do                                                                                             \
    {                                                                                              \
        cudaError_t err_ = (call);                                                                 \
        if (err_ != cudaSuccess)                                                                   \
        {                                                                                          \
            std::cerr << "CUDA error " << cudaGetErrorName(err_) << " ("                           \
                      << cudaGetErrorString(err_) << ")"                                           \
                      << " at " << __FILE__ << ":" << __LINE__ << " (ignored during cleanup)\n";   \
        }                                                                                          \
    } while (0)

inline int checkGPUStatus()
{
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0)
    {
        std::cerr << "No CUDA devices found\n";
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Global memory: " << prop.totalGlobalMem / B_TO_MB << " MB\n";
    std::cout << "SM count: " << prop.multiProcessorCount << "\n";
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;

    size_t free_mem;
    size_t total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "Free memory: " << free_mem / B_TO_GB
              << "GB, total memory: " << total_mem / B_TO_GB << "GB\n"
              << std::endl;

    return 0;
}
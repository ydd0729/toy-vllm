#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int checkGPUStatus()
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
    std::cout << "Global memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB\n";
    std::cout << "SM count: " << prop.multiProcessorCount << "\n";

    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "cuBLAS init failed\n";
        return 1;
    }
    std::cout << "cuBLAS initialized OK\n";
    cublasDestroy(handle);

    return 0;
}

int readSafetensorsFile()
{
    std::ifstream file_handle("model.safetensors", std::ios_base::binary); // TODO: use args to provide the path or smth
    if (!file_handle.is_open())
    {
        std::cout << "Can't open model.safetensors file\n";
        file_handle.close();
        return 1;
    }
    uint64_t header_size;
    // reinterpret_cast<char*> gives me an address of header_size
    file_handle.read(reinterpret_cast<char *>(&header_size), 8);
    std::cout << "Safetensors header size read correctly. Size of header: " << header_size << std::endl;
    std::string header;
    header.resize(header_size);
    file_handle.read(header.data(), header_size);
    std::cout << header << std::endl;
    std::cout << "Header read correctly\n";
    file_handle.close();
    return 0;
}

int main(int argc, char *argv[])
{
    if (checkGPUStatus() != 0)
    {
        return 1;
    }

    if (readSafetensorsFile() != 0)
    {
        return 1;
    }

    std::cout << "Closing the program\n";
    return 0;
}
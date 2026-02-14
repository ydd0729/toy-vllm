#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include "json.hpp"

using json = nlohmann::json;

constexpr int N_LAYERS = 16; // TODO: hardcoded for llama 3.2 1B, just like any other value for now

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

struct TensorMetadata
{
    std::string tensor_name;
    std::string dtype;
    std::vector<int> shape;
    uint64_t offset_begin;
    uint64_t offset_end;
};

struct Weights
{
    __nv_bfloat16* embed_tokens;
    __nv_bfloat16* input_layernorms[N_LAYERS];
    __nv_bfloat16* mlp_down_proj[N_LAYERS];
    __nv_bfloat16* mlp_gate_proj[N_LAYERS];
    __nv_bfloat16* mlp_up_proj[N_LAYERS];
    __nv_bfloat16* post_attn_layernorms[N_LAYERS];
    __nv_bfloat16* w_k[N_LAYERS];
    __nv_bfloat16* w_o[N_LAYERS];
    __nv_bfloat16* w_q[N_LAYERS];
    __nv_bfloat16* w_v[N_LAYERS];
    __nv_bfloat16* norm;
};

int main(int argc, char *argv[])
{
    if (checkGPUStatus() != 0)
    {
        return 1;
    }

    std::ifstream safetensors_file("model.safetensors", std::ios_base::binary); // TODO: use args to provide the path or smth
    if (!safetensors_file.is_open())
    {
        std::cout << "Can't open model.safetensors file\n";
        safetensors_file.close();
        return 1;
    }
    uint64_t header_size;
    // reinterpret_cast<char*> gives me an address of header_size
    safetensors_file.read(reinterpret_cast<char *>(&header_size), 8);
    std::cout << "Safetensors header size read correctly. Size of header: " << header_size << std::endl;
    std::string header;
    header.resize(header_size);
    safetensors_file.read(header.data(), header_size);
    std::cout << "Header read correctly\n";
    std::vector<TensorMetadata> tensors;
    json j = json::parse(header);
    uint64_t max_offset = 0;
    for (auto &[key, value] : j.items())
    {
        if (key == "__metadata__")
        {
            continue;
        }
        uint64_t offset_end = value["data_offsets"].at(1).get<uint64_t>();
        if (offset_end > max_offset)
        {
            max_offset = offset_end;
        }
        tensors.push_back(TensorMetadata{
            tensor_name : key,
            dtype : value["dtype"].get<std::string>(),
            shape : value["shape"].get<std::vector<int>>(),
            offset_begin : value["data_offsets"].at(0).get<uint64_t>(),
            offset_end : offset_end
        });
    }

#ifdef DEBUG
    for (auto &tensor : tensors)
    {
        std::cout << tensor.tensor_name << ", dtype: " << tensor.dtype << ", shape: (";
        for (auto &shape_item : tensor.shape)
        {
            std::cout << shape_item << ", ";
        }
        std::cout << "), offset: [" << tensor.offset_begin << ", " << tensor.offset_end << "]" << std::endl;
    }
#endif

    void *gpu_tensors;
    cudaMalloc(&gpu_tensors, max_offset);
    std::vector<char> tensors_data;
    tensors_data.resize(max_offset);
    safetensors_file.read(tensors_data.data(), max_offset);
    cudaMemcpy(gpu_tensors, tensors_data.data(), max_offset, cudaMemcpyHostToDevice);
    std::cout << "Copied model tensors to GPU correctly!\n";
    safetensors_file.close();

#ifdef DEBUG
    int test_size = 20;
    std::vector<char> test_from_gpu;
    test_from_gpu.resize(20);
    cudaMemcpy(test_from_gpu.data(), gpu_tensors, 20, cudaMemcpyDeviceToHost);
    std::cout << "\nCopied from GPU:\n";
    std::cout << "\n"
              << test_from_gpu.data() << "\n";
    for (auto &i : test_from_gpu)
    {
        printf("%02x ", (unsigned char)i);
    }
    std::cout << "\nOriginal CPU data:\n";
    for (int i = 0; i < 20; ++i)
    {
        printf("%02x ", (unsigned char)tensors_data[i]);
    }
#endif

    Weights weights{};
    weights.embed_tokens = (__nv_bfloat16*)((char*)gpu_tensors+tensors[0].offset_begin);

    std::vector<int> input_tokens;
    int token;
    while (std::cin >> token)
    {
        input_tokens.push_back(token);
    }
#ifdef DEBUG
    std::cout << "Input tokens:\n";
    for (auto& token: input_tokens) {
        std::cout << token << "\n";
    }
#endif
    
    std::cout << "\nClosing the program\n";
    return 0;
}
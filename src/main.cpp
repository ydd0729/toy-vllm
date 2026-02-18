#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include "json.hpp"
#include "kernels.cuh"
#include "main.h"

using json = nlohmann::json;

constexpr int N_LAYERS = 16; // TODO: hardcoded for llama 3.2 1B, just like any other value for now
constexpr int EMBEDDING_LENGTH = 2048;

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
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;

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

bool verifyModelWeightsCopy(void *model_weights, std::vector<char> &model_weights_cpu)
{
    std::vector<char> test_from_gpu;
    test_from_gpu.resize(20);
    cudaMemcpy(test_from_gpu.data(), model_weights, 20, cudaMemcpyDeviceToHost);
    bool is_correct = true;
    for (int i = 0; i < 20; ++i)
    {
        if ((unsigned char)model_weights_cpu[i] == (unsigned char)test_from_gpu[i])
        {
            continue;
        }
        if (is_correct)
        {
            std::cout << "Model weights copied to GPU incorrectly!:\n";
        }
        printf("%02x ", (unsigned char)model_weights_cpu[i] == (unsigned char)test_from_gpu[i]);
        is_correct = false;
    }
    return is_correct;
}

bool verifyInputTokensCopy(std::vector<int> &input_tokens, int *gpu_input_tokens)
{
    std::vector<int> test_from_gpu_tokens;
    test_from_gpu_tokens.resize(input_tokens.size());
    cudaMemcpy(test_from_gpu_tokens.data(), gpu_input_tokens, input_tokens.size() * sizeof(int), cudaMemcpyDeviceToHost);
    bool is_correct = true;
    for (int i = 0; i < input_tokens.size(); ++i)
    {
        if (input_tokens[i] == test_from_gpu_tokens[i])
        {
            continue;
        }
        if (is_correct)
        {
            std::cout << "Input tokens copy mismatch!" << std::endl;
        }
        std::cout << "CPU: " << input_tokens[i] << " | GPU: " << test_from_gpu_tokens[i] << "\n";
        is_correct = false;
    }
    return is_correct;
}

bool verifyEmbeddingGather(std::vector<int> &input_tokens, nv_bfloat16 *input_embeddings, std::vector<char> &model_weights_cpu, std::unordered_map<std::string, uint64_t> &offsets)
{
    std::vector<__nv_bfloat16> test_gpu_input_embeds;
    test_gpu_input_embeds.resize(EMBEDDING_LENGTH * input_tokens.size());
    cudaMemcpy(test_gpu_input_embeds.data(), input_embeddings, input_tokens.size() * sizeof(__nv_bfloat16) * EMBEDDING_LENGTH, cudaMemcpyDeviceToHost);
    bool is_correct = true;
    for (int token = 0; token < input_tokens.size(); ++token)
    {
        for (int i = 0; i < 2048; ++i)
        {
            __nv_bfloat16 *all_embeds_cpu = (__nv_bfloat16 *)(model_weights_cpu.data() + offsets.at("model.embed_tokens.weight"));
            if ((float)test_gpu_input_embeds[token * 2048 + i] != (float)all_embeds_cpu[input_tokens[token] * 2048 + i])
            {
                if (is_correct)
                {
                    std::cout << "Incorrect embeddings were retrieved" << std::endl;
                }
                std::cout << "GPU:" << (float)test_gpu_input_embeds[token * 2048 + i] << " | CPU: " << (float)all_embeds_cpu[input_tokens[token] * 2048 + i] << "\n";
                is_correct = false;
            }
        }
    }
    return is_correct;
}

struct Weights
{
    __nv_bfloat16 *embed_tokens;
    __nv_bfloat16 *input_layernorm[N_LAYERS];
    __nv_bfloat16 *mlp_down_proj[N_LAYERS];
    __nv_bfloat16 *mlp_gate_proj[N_LAYERS];
    __nv_bfloat16 *mlp_up_proj[N_LAYERS];
    __nv_bfloat16 *post_attn_layernorms[N_LAYERS];
    __nv_bfloat16 *w_k[N_LAYERS];
    __nv_bfloat16 *w_o[N_LAYERS];
    __nv_bfloat16 *w_q[N_LAYERS];
    __nv_bfloat16 *w_v[N_LAYERS];
    __nv_bfloat16 *norm;
};

int main(int argc, char *argv[])
{
    if (checkGPUStatus() != 0)
    {
        return 1;
    }

    // READ SAFETENSORS
    std::ifstream safetensors_file("model.safetensors", std::ios_base::binary); // TODO: use args to provide the path or smth
    if (!safetensors_file.is_open())
    {
        std::cout << "Can't open model.safetensors file\n";
        safetensors_file.close();
        return 1;
    }

    // READ SAFETENSORS HEADER SIZE
    uint64_t header_size;
    // reinterpret_cast<char*>(&header_size) gives me an address of header_size
    safetensors_file.read(reinterpret_cast<char *>(&header_size), 8);
#ifdef DEBUG
    std::cout << "Safetensors header size read correctly. Size of header: " << header_size << std::endl;
#endif
    // READ SAFETENSORS HEADER
    std::string header;
    header.resize(header_size);
    safetensors_file.read(header.data(), header_size);
#ifdef DEBUG
    std::cout << "Header read correctly\n";
#endif
    // READ OFFSETS OF EVERY LAYER (TENSOR) TO KNOW WHERE EVERY LAYER STARTS AND ENDS IN THE MEMORY
    std::unordered_map<std::string, uint64_t> offsets;
    json header_json = json::parse(header);
    uint64_t max_offset = 0;
    for (auto &[key, value] : header_json.items())
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
        offsets[key] = value["data_offsets"].at(0).get<uint64_t>();
    }

    void *model_weights;
    cudaMalloc(&model_weights, max_offset); // max_offset tells where the model weights end in the memory

    std::vector<char> model_weights_cpu;
    model_weights_cpu.resize(max_offset);
    safetensors_file.read(model_weights_cpu.data(), max_offset);

    cudaMemcpy(model_weights, model_weights_cpu.data(), max_offset, cudaMemcpyHostToDevice);
    if (!verifyModelWeightsCopy(model_weights, model_weights_cpu))
    {
        return 1;
    }
    safetensors_file.close();

    // BASICALLY A HELPER STRUCT TO HAVE AN EASY ACCESS TO ANY MODEL WEIGHTS ON GPU
    // TODO: right now I know the model structure since it's always llama 3.2 1B-Instruct, but maybe it would be convenient
    //       to store dimensions somewhere for even easier access?
    Weights weights{};
    weights.embed_tokens = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.embed_tokens.weight"));
    weights.norm = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.norm.weight"));
    for (int i = 0; i < N_LAYERS; ++i)
    {
        weights.input_layernorm[i] = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.layers." + std::to_string(i) + ".input_layernorm.weight"));
        weights.mlp_down_proj[i] = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.layers." + std::to_string(i) + ".mlp.down_proj.weight"));
        weights.mlp_gate_proj[i] = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.layers." + std::to_string(i) + ".mlp.gate_proj.weight"));
        weights.mlp_up_proj[i] = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.layers." + std::to_string(i) + ".mlp.up_proj.weight"));
        weights.post_attn_layernorms[i] = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.layers." + std::to_string(i) + ".post_attention_layernorm.weight"));
        weights.w_k[i] = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.layers." + std::to_string(i) + ".self_attn.k_proj.weight"));
        weights.w_o[i] = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.layers." + std::to_string(i) + ".self_attn.o_proj.weight"));
        weights.w_q[i] = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.layers." + std::to_string(i) + ".self_attn.q_proj.weight"));
        weights.w_v[i] = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.layers." + std::to_string(i) + ".self_attn.v_proj.weight"));
    }

    // LLM INPUT
    std::vector<int> input_tokens;
#ifdef DEBUG
    input_tokens.push_back(128000);
    input_tokens.push_back(791);
    input_tokens.push_back(6864);
    input_tokens.push_back(315);
    input_tokens.push_back(9822);
    input_tokens.push_back(374);
#else
    int token;
    while (std::cin >> token)
    {
        input_tokens.push_back(token);
    }
#endif
#ifdef DEBUG
    std::cout << "Input tokens:\n";
    for (auto &token : input_tokens)
    {
        std::cout << token << "\n";
    }
#endif

    int *gpu_input_tokens;
    cudaMalloc(&gpu_input_tokens, input_tokens.size() * sizeof(int));
    cudaMemcpy(gpu_input_tokens, input_tokens.data(), input_tokens.size() * sizeof(int), cudaMemcpyHostToDevice);
#ifdef DEBUG
    if (!verifyInputTokensCopy(input_tokens, gpu_input_tokens))
    {
        return 1;
    }
#endif
    // INFERENCE STARTS HERE! =]
    // I have the same amount of embeddings as input tokens
    // it's just every embedding is 2048 length bf16 vector
    // retrieved from model weights based on token's value
    __nv_bfloat16 *input_embeddings;
    cudaMalloc(&input_embeddings, input_tokens.size() * sizeof(__nv_bfloat16) * EMBEDDING_LENGTH);
    embeddingGather(gpu_input_tokens, input_embeddings, weights.embed_tokens, input_tokens.size());
    cudaDeviceSynchronize();
#ifdef DEBUG
    if (!verifyEmbeddingGather(input_tokens, input_embeddings, model_weights_cpu, offsets))
    {
        return 1;
    }
#endif
    std::cout << "\nOk bye!\n";
    return 0;
}

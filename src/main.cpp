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
constexpr int KV_DIM = 512;
constexpr int HEAD_DIM = 64;
constexpr int NUM_Q_HEADS = 32;
constexpr int NUM_K_HEADS = 8;
constexpr int GQA_Q_TO_K_RATIO = 4;

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
        for (int i = 0; i < EMBEDDING_LENGTH; ++i)
        {
            __nv_bfloat16 *all_embeds_cpu = (__nv_bfloat16 *)(model_weights_cpu.data() + offsets.at("model.embed_tokens.weight"));
            if ((float)test_gpu_input_embeds[token * EMBEDDING_LENGTH + i] != (float)all_embeds_cpu[input_tokens[token] * EMBEDDING_LENGTH + i])
            {
                if (is_correct)
                {
                    std::cout << "Incorrect embeddings were retrieved" << std::endl;
                }
                std::cout << "GPU:" << (float)test_gpu_input_embeds[token * EMBEDDING_LENGTH + i] << " | CPU: " << (float)all_embeds_cpu[input_tokens[token] * EMBEDDING_LENGTH + i] << "\n";
                is_correct = false;
            }
        }
    }
    return is_correct;
}

bool floats_close_enough(float a, float b)
{
    return fabs(a - b) / fmax(fabs(a), fabs(b)) < 1e-3;
}

bool verifyRMSNormWeights(std::vector<char> &model_weights_cpu, std::unordered_map<std::string, uint64_t> &offsets)
{
    __nv_bfloat16 *layernorm_weights = (__nv_bfloat16 *)(model_weights_cpu.data() + offsets.at("model.layers.0.input_layernorm.weight"));
    std::vector<float> rms_norm_debug_values = {0.154297, 0.182617, 0.255859, -0.0116577, 0.140625, 0.19043, -0.139648, -0.160156, 0.139648, -0.170898};
    bool is_correct_rms_weight = true;
    for (int i = 0; i < 10; ++i)
    {
        if (!floats_close_enough((float)layernorm_weights[i], rms_norm_debug_values[i]))
        {
            if (is_correct_rms_weight)
            {
                std::cout << "RMS norm weights check failed" << std::endl;
            }
            std::cout << "Expected RMS norm weight: " << rms_norm_debug_values[i] << ", received: " << (float)layernorm_weights[i] << std::endl;
            is_correct_rms_weight = false;
        }
    }
    return is_correct_rms_weight;
}

bool verifyRmsNorm(__nv_bfloat16 *gpu_input, __nv_bfloat16 *gpu_output,
                   std::vector<char> &model_weights_cpu,
                   std::unordered_map<std::string, uint64_t> &offsets,
                   int num_tokens, int layer)
{
    constexpr float EPSILON = 1e-5f;
    constexpr float TOLERANCE = 1e-2f;

    std::vector<__nv_bfloat16> cpu_input(num_tokens * EMBEDDING_LENGTH);
    std::vector<__nv_bfloat16> cpu_output(num_tokens * EMBEDDING_LENGTH);
    cudaMemcpy(cpu_input.data(), gpu_input, num_tokens * EMBEDDING_LENGTH * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_output.data(), gpu_output, num_tokens * EMBEDDING_LENGTH * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    std::string weight_key = "model.layers." + std::to_string(layer) + ".input_layernorm.weight";
    __nv_bfloat16 *norm_weights = (__nv_bfloat16 *)(model_weights_cpu.data() + offsets.at(weight_key));

    int mismatches = 0;
    for (int t = 0; t < num_tokens; ++t)
    {
        float sum_sq = 0.0f;
        for (int i = 0; i < EMBEDDING_LENGTH; ++i)
        {
            float val = (float)cpu_input[t * EMBEDDING_LENGTH + i];
            sum_sq += val * val;
        }
        float rms = sqrtf(sum_sq / EMBEDDING_LENGTH + EPSILON);

        for (int i = 0; i < EMBEDDING_LENGTH; ++i)
        {
            float input_val = (float)cpu_input[t * EMBEDDING_LENGTH + i];
            float weight_val = (float)norm_weights[i];
            float expected = (input_val / rms) * weight_val;
            float actual = (float)cpu_output[t * EMBEDDING_LENGTH + i];

            float rel_err = (expected == 0.0f) ? fabs(actual) : fabs(actual - expected) / fabs(expected);
            if (rel_err > TOLERANCE || isnanf(actual) || isnanf(expected))
            {
                if (mismatches < 10)
                {
                    std::cout << "RMSNorm MISMATCH token=" << t << " elem=" << i
                              << " expected=" << expected << " got=" << actual
                              << " rel_err=" << rel_err << "\n";
                }
                mismatches++;
            }
        }
    }

    return mismatches == 0;
}

bool verifyQProjection(cublasStatus_t gemm_status, std::vector<int> &input_tokens, nv_bfloat16 *q, std::vector<char> &model_weights_cpu, std::unordered_map<std::string, uint64_t> &offsets, nv_bfloat16 *rms_norms)
{
    std::cout << "Cublas first gemm status: " << gemm_status << std::endl;
    std::vector<__nv_bfloat16> q_cpu(input_tokens.size() * EMBEDDING_LENGTH);
    cudaMemcpy(q_cpu.data(), q, input_tokens.size() * EMBEDDING_LENGTH * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    std::vector<float> q_cpu_crosscheck(input_tokens.size() * EMBEDDING_LENGTH);
    __nv_bfloat16 *q_cpu_weights = (__nv_bfloat16 *)(model_weights_cpu.data() + offsets.at("model.layers.0.self_attn.q_proj.weight"));
    std::vector<__nv_bfloat16> rms_norms_cpu(input_tokens.size() * EMBEDDING_LENGTH);
    cudaMemcpy(rms_norms_cpu.data(), rms_norms, input_tokens.size() * EMBEDDING_LENGTH * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    // input_tokens * w_q^T (N, EMBEDDING_LENGTH) x (EMBEDDING_LENGTH, EMBEDDING_LENGTH) -> (N, EMBEDDING_LENGTH)
    bool is_correct = true;
    for (int token_idx = 0; token_idx < input_tokens.size(); ++token_idx)
    {
        for (int j = 0; j < EMBEDDING_LENGTH; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < EMBEDDING_LENGTH; ++k)
            {
                float input_value = (float)rms_norms_cpu[token_idx * EMBEDDING_LENGTH + k];
                float weight_value = (float)q_cpu_weights[j * EMBEDDING_LENGTH + k];
                sum += input_value * weight_value;
            }
            float actual = (float)q_cpu[token_idx * EMBEDDING_LENGTH + j];
            float rel_err = (sum == 0.0f) ? fabs(actual) : fabs(actual - sum) / fabs(sum);
            if (rel_err > 1e-1)
            {
                std::cout << "Q MISMATCH token=" << token_idx << " dim=" << j
                          << " expected=" << sum << " got=" << actual
                          << " rel_err=" << rel_err << "\n";
                is_correct = false;
            }
        }
    }
    if (is_correct)
    {
        std::cout << "Q projection check done, all correct!" << std::endl;
    }
    else
    {
        std::cout << "Q projection check failed!" << std::endl;
    }
    return is_correct;
}

bool verifyRope(__nv_bfloat16 *gpu_q, __nv_bfloat16 *gpu_k,
                std::vector<__nv_bfloat16> &q_before_rope,
                std::vector<__nv_bfloat16> &k_before_rope,
                int num_tokens)
{
    constexpr float TOLERANCE = 1e-2f;
    constexpr float ROPE_THETA = 500000.0f;

    std::vector<__nv_bfloat16> q_gpu(num_tokens * EMBEDDING_LENGTH);
    std::vector<__nv_bfloat16> k_gpu(num_tokens * KV_DIM);
    cudaMemcpy(q_gpu.data(), gpu_q, num_tokens * EMBEDDING_LENGTH * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    cudaMemcpy(k_gpu.data(), gpu_k, num_tokens * KV_DIM * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    int mismatches = 0;

    // Verify Q
    for (int t = 0; t < num_tokens; ++t)
    {
        for (int pair = 0; pair < EMBEDDING_LENGTH / 2; ++pair)
        {
            int i = pair % (HEAD_DIM / 2);
            float theta = 1.0f / powf(ROPE_THETA, (float)(2 * i) / HEAD_DIM);
            float angle = t * theta;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);

            int idx = t * EMBEDDING_LENGTH + 2 * pair;
            float a = (float)q_before_rope[idx];
            float b = (float)q_before_rope[idx + 1];
            float expected_0 = a * cos_a - b * sin_a;
            float expected_1 = a * sin_a + b * cos_a;
            float actual_0 = (float)q_gpu[idx];
            float actual_1 = (float)q_gpu[idx + 1];

            float err0 = (expected_0 == 0.0f) ? fabsf(actual_0) : fabsf(actual_0 - expected_0) / fabsf(expected_0);
            float err1 = (expected_1 == 0.0f) ? fabsf(actual_1) : fabsf(actual_1 - expected_1) / fabsf(expected_1);
            if (err0 > TOLERANCE || err1 > TOLERANCE || isnanf(actual_0) || isnanf(actual_1))
            {
                if (mismatches < 10)
                    std::cout << "Q RoPE MISMATCH token=" << t << " pair=" << pair
                              << " expected=(" << expected_0 << "," << expected_1
                              << ") got=(" << actual_0 << "," << actual_1 << ")\n";
                mismatches++;
            }
        }
    }

    // Verify K
    for (int t = 0; t < num_tokens; ++t)
    {
        for (int pair = 0; pair < KV_DIM / 2; ++pair)
        {
            int i = pair % (HEAD_DIM / 2);
            float theta = 1.0f / powf(ROPE_THETA, (float)(2 * i) / HEAD_DIM);
            float angle = t * theta;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);

            int idx = t * KV_DIM + 2 * pair;
            float a = (float)k_before_rope[idx];
            float b = (float)k_before_rope[idx + 1];
            float expected_0 = a * cos_a - b * sin_a;
            float expected_1 = a * sin_a + b * cos_a;
            float actual_0 = (float)k_gpu[idx];
            float actual_1 = (float)k_gpu[idx + 1];

            float err0 = (expected_0 == 0.0f) ? fabsf(actual_0) : fabsf(actual_0 - expected_0) / fabsf(expected_0);
            float err1 = (expected_1 == 0.0f) ? fabsf(actual_1) : fabsf(actual_1 - expected_1) / fabsf(expected_1);
            if (err0 > TOLERANCE || err1 > TOLERANCE || isnanf(actual_0) || isnanf(actual_1))
            {
                if (mismatches < 10)
                    std::cout << "K RoPE MISMATCH token=" << t << " pair=" << pair
                              << " expected=(" << expected_0 << "," << expected_1
                              << ") got=(" << actual_0 << "," << actual_1 << ")\n";
                mismatches++;
            }
        }
    }

    if (mismatches == 0)
        std::cout << "RoPE verification PASSED\n";
    else
        std::cout << "RoPE verification FAILED: " << mismatches << " mismatches\n";
    return mismatches == 0;
}

bool verifyAttnScores(__nv_bfloat16 *gpu_q, __nv_bfloat16 *gpu_k, __nv_bfloat16 *gpu_scores,
                      int num_tokens)
{
    constexpr float TOLERANCE = 1e-1f;
    constexpr float SCALE = 1.0f / 8.0f;

    std::vector<__nv_bfloat16> q_cpu(num_tokens * EMBEDDING_LENGTH);
    std::vector<__nv_bfloat16> k_cpu(num_tokens * KV_DIM);
    std::vector<__nv_bfloat16> scores_cpu(num_tokens * num_tokens * NUM_Q_HEADS);
    cudaMemcpy(q_cpu.data(), gpu_q, num_tokens * EMBEDDING_LENGTH * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    cudaMemcpy(k_cpu.data(), gpu_k, num_tokens * KV_DIM * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    cudaMemcpy(scores_cpu.data(), gpu_scores, num_tokens * num_tokens * NUM_Q_HEADS * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    int mismatches = 0;
    for (int h = 0; h < NUM_Q_HEADS; ++h)
    {
        int kv_head = h / GQA_Q_TO_K_RATIO;
        for (int t1 = 0; t1 < num_tokens; ++t1)
        {
            for (int t2 = 0; t2 < num_tokens; ++t2)
            {
                float sum = 0.0f;
                for (int d = 0; d < HEAD_DIM; ++d)
                {
                    float q_val = (float)q_cpu[t1 * EMBEDDING_LENGTH + h * HEAD_DIM + d];
                    float k_val = (float)k_cpu[t2 * KV_DIM + kv_head * HEAD_DIM + d];
                    sum += q_val * k_val;
                }
                float expected = sum * SCALE;
                float actual = (float)scores_cpu[h * num_tokens * num_tokens + t1 * num_tokens + t2];

                float rel_err = (expected == 0.0f) ? fabsf(actual) : fabsf(actual - expected) / fabsf(expected);
                if (rel_err > TOLERANCE || isnanf(actual))
                {
                    if (mismatches < 10)
                        std::cout << "ATTN SCORE MISMATCH head=" << h << " t1=" << t1 << " t2=" << t2
                                  << " expected=" << expected << " got=" << actual
                                  << " rel_err=" << rel_err << "\n";
                    mismatches++;
                }
            }
        }
    }

    if (mismatches == 0)
        std::cout << "Attention scores verification PASSED\n";
    else
        std::cout << "Attention scores verification FAILED: " << mismatches << " mismatches\n";
    return mismatches == 0;
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
    // it's just every embedding is EMBEDDING_LENGTH length bf16 vector
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
    __nv_bfloat16 *rms_norms;
    cudaMalloc(&rms_norms, input_tokens.size() * sizeof(__nv_bfloat16) * EMBEDDING_LENGTH);
    rmsNorm(input_embeddings, rms_norms, weights.input_layernorm[0], input_tokens.size());
    cudaDeviceSynchronize();
#ifdef DEBUG
    if (!verifyRMSNormWeights(model_weights_cpu, offsets) || !verifyRmsNorm(input_embeddings, rms_norms, model_weights_cpu, offsets, input_tokens.size(), 0))
    {
        std::cout << "RMS norm verification failed" << std::endl;
        return 1;
    }
#endif

    cublasHandle_t cublas_handle;
    cublasStatus_t status = cublasCreate(&cublas_handle);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "cuBLAS init failed, status: " << status << "\n";
        return 1;
    }

#ifdef DEBUG
    std::cout << "cuBLAS initialized OK\n";
#endif

    // Q = inputs * wq^T; my matrices are row-major, cublas expects column-major
    // it perceives my matrices as transposed
    // there's a trick where C = A * B == C^T = B^T * A^T
    // so in my scenario cublas sees now: Q = inputs^T * wq^T^T = inputs ^T * wq
    // so I need to do: Q^T = wq ^T * inputs
    // the beauty is that we don't need to transpose Q^T back to Q
    // because cublas sees the output as column-major
    // so it's in fact transposed
    // final dim (num_tok, EMBEDDING_LENGTH)
    __nv_bfloat16 *q_proj;
    cudaMalloc(&q_proj, input_tokens.size() * sizeof(__nv_bfloat16) * EMBEDDING_LENGTH);
    float q_proj_alpha = 1.0f;
    float q_proj_beta = 0.0f;
    cublasStatus_t q_proj_status = cublasGemmEx(cublas_handle,
                                                CUBLAS_OP_T,
                                                CUBLAS_OP_N,
                                                EMBEDDING_LENGTH,
                                                input_tokens.size(),
                                                EMBEDDING_LENGTH,
                                                &q_proj_alpha,
                                                weights.w_q[0],
                                                CUDA_R_16BF,
                                                EMBEDDING_LENGTH,
                                                rms_norms,
                                                CUDA_R_16BF,
                                                EMBEDDING_LENGTH,
                                                &q_proj_beta,
                                                q_proj,
                                                CUDA_R_16BF,
                                                EMBEDDING_LENGTH,
                                                CUBLAS_COMPUTE_32F,
                                                CUBLAS_GEMM_DEFAULT);
    cudaDeviceSynchronize();
#ifdef DEBUG
    verifyQProjection(q_proj_status, input_tokens, q_proj, model_weights_cpu, offsets, rms_norms);
#endif

    __nv_bfloat16 *k_proj;
    cudaMalloc(&k_proj, input_tokens.size() * sizeof(__nv_bfloat16) * KV_DIM);
    // input = (num_tokens, EMBEDDING_LENGTH), weights = (KV_DIM, EMBEDDING_LENGTH)
    // after trick: (KV_DIM, EMBEDDING_LENGTH) * (EMBEDDING_LENGTH, num_tokens) -> (KV_DIM, num_tokens), which really is (num_tok, KV_DIM)
    // lda: EMBEDDING_LENGTH, ldb: EMBEDDING_LENGTH, ldc: KV_DIM

    float k_proj_alpha = 1.0f;
    float k_proj_beta = 0.0f;
    cublasStatus_t k_proj_status = cublasGemmEx(cublas_handle,
                                                CUBLAS_OP_T,
                                                CUBLAS_OP_N,
                                                KV_DIM,
                                                input_tokens.size(),
                                                EMBEDDING_LENGTH,
                                                &k_proj_alpha,
                                                weights.w_k[0],
                                                CUDA_R_16BF,
                                                EMBEDDING_LENGTH,
                                                rms_norms,
                                                CUDA_R_16BF,
                                                EMBEDDING_LENGTH,
                                                &k_proj_beta,
                                                k_proj,
                                                CUDA_R_16BF,
                                                KV_DIM,
                                                CUBLAS_COMPUTE_32F,
                                                CUBLAS_GEMM_DEFAULT);

    // same as K projection
    __nv_bfloat16 *v_proj;
    cudaMalloc(&v_proj, input_tokens.size() * sizeof(__nv_bfloat16) * KV_DIM);

    float v_proj_alpha = 1.0f;
    float v_proj_beta = 0.0f;
    cublasStatus_t v_proj_status = cublasGemmEx(cublas_handle,
                                                CUBLAS_OP_T,
                                                CUBLAS_OP_N,
                                                KV_DIM,
                                                input_tokens.size(),
                                                EMBEDDING_LENGTH,
                                                &v_proj_alpha,
                                                weights.w_v[0],
                                                CUDA_R_16BF,
                                                EMBEDDING_LENGTH,
                                                rms_norms,
                                                CUDA_R_16BF,
                                                EMBEDDING_LENGTH,
                                                &v_proj_beta,
                                                v_proj,
                                                CUDA_R_16BF,
                                                KV_DIM,
                                                CUBLAS_COMPUTE_32F,
                                                CUBLAS_GEMM_DEFAULT);

    // RoPE now

    std::vector<__nv_bfloat16> q_before_rope(input_tokens.size() * EMBEDDING_LENGTH);
    std::vector<__nv_bfloat16> k_before_rope(input_tokens.size() * KV_DIM);
    cudaMemcpy(q_before_rope.data(), q_proj, input_tokens.size() * EMBEDDING_LENGTH * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    cudaMemcpy(k_before_rope.data(), k_proj, input_tokens.size() * KV_DIM * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    rope(q_proj, input_tokens.size(), EMBEDDING_LENGTH);
    rope(k_proj, input_tokens.size(), KV_DIM);
    cudaDeviceSynchronize();

    verifyRope(q_proj, k_proj, q_before_rope, k_before_rope, input_tokens.size());

    // attention scores
    // per head, 64 elements each
    // so total 32 heads
    // Q (num_tok, 2048)
    // K (num_tok, 512)
    // GQA grouping reuses 1 K head per 4 consecutive Q heads
    // Q_head (num_tok, 64)
    // K_head (num_tok, 64)
    // attn_score_head = Q_head * K_head^T / sqrt(64)
    // so: head output dims (num_tok, num_tok)
    __nv_bfloat16 *attn_scores;
    cudaMalloc(&attn_scores, input_tokens.size() * input_tokens.size() * sizeof(__nv_bfloat16) * NUM_Q_HEADS);
    float attn_alpha = 1.0f / 8.0f;
    float attn_beta = 0.0f;
    for (int i = 0; i < NUM_Q_HEADS; ++i)
    {
        int k_head_idx = i / GQA_Q_TO_K_RATIO;
        __nv_bfloat16 *q_head = q_proj + i * HEAD_DIM;
        __nv_bfloat16 *k_head = k_proj + k_head_idx * HEAD_DIM;
        __nv_bfloat16 *attn_score_head = attn_scores + input_tokens.size() * input_tokens.size() * i;

        cublasStatus_t attn_score_status = cublasGemmEx(cublas_handle,
                                                        CUBLAS_OP_T,
                                                        CUBLAS_OP_N,
                                                        input_tokens.size(),
                                                        input_tokens.size(),
                                                        HEAD_DIM,
                                                        &attn_alpha,
                                                        k_head,
                                                        CUDA_R_16BF,
                                                        KV_DIM,
                                                        q_head,
                                                        CUDA_R_16BF,
                                                        EMBEDDING_LENGTH,
                                                        &attn_beta,
                                                        attn_score_head,
                                                        CUDA_R_16BF,
                                                        input_tokens.size(),
                                                        CUBLAS_COMPUTE_32F,
                                                        CUBLAS_GEMM_DEFAULT);
    }
    cudaDeviceSynchronize();
    verifyAttnScores(q_proj, k_proj, attn_scores, input_tokens.size());
    std::cout << "\nOk bye!\n";
    cublasDestroy(cublas_handle);
    cudaDeviceSynchronize();
    return 0;
}

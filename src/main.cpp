#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include "json.hpp"
#include "kernels.cuh"
#include "main.h"

using json = nlohmann::json;

constexpr int MAX_NEW_TOKENS_GENERATED = 20; // TODO: parameterize it with program arguments

constexpr int N_LAYERS = 16; // TODO: hardcoded for llama 3.2 1B, just like any other value for now
constexpr int EMBEDDING_LENGTH = 2048;
constexpr int HIDDEN_DIM = 8192;
constexpr int KV_DIM = 512;
constexpr int HEAD_DIM = 64;
constexpr int NUM_Q_HEADS = 32;
constexpr int NUM_K_HEADS = 8;
constexpr int NUM_V_HEADS = 8;
constexpr int GQA_Q_TO_K_RATIO = 4;
constexpr int GQA_ATTN_SCORES_TO_V_RATIO = 4;
constexpr int VOCAB_SIZE = 128256;
constexpr int END_OF_TEXT_TOKEN_ID = 128001; // <|end_of_text|>
constexpr int EOT_ID_TOKEN_ID = 128009;      // <|eot_id|>
constexpr int MAX_SEQ_LEN = 2048;            // TODO: make it tunable

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

bool verifyRMSNormWeights(std::vector<char> &model_weights_cpu, std::unordered_map<std::string, uint64_t> &offsets, int layer)
{
    if (layer != 0)
    {
        // skipping
        return true;
    }
    __nv_bfloat16 *layernorm_weights = (__nv_bfloat16 *)(model_weights_cpu.data() + offsets.at("model.layers." + std::to_string(layer) + ".input_layernorm.weight"));
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
            std::cout << "Expected RMS norm weight at layer " << std::to_string(layer) << ": " << rms_norm_debug_values[i] << ", received: " << (float)layernorm_weights[i] << std::endl;
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

bool verifyQProjection(cublasStatus_t gemm_status, std::vector<int> &input_tokens, nv_bfloat16 *q, std::vector<char> &model_weights_cpu, std::unordered_map<std::string, uint64_t> &offsets, nv_bfloat16 *rms_norms, int layer)
{
    std::cout << "Cublas first gemm status: " << gemm_status << std::endl;
    std::vector<__nv_bfloat16> q_cpu(input_tokens.size() * EMBEDDING_LENGTH);
    cudaMemcpy(q_cpu.data(), q, input_tokens.size() * EMBEDDING_LENGTH * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    std::vector<float> q_cpu_crosscheck(input_tokens.size() * EMBEDDING_LENGTH);
    __nv_bfloat16 *q_cpu_weights = (__nv_bfloat16 *)(model_weights_cpu.data() + offsets.at("model.layers." + std::to_string(layer) + ".self_attn.q_proj.weight"));
    std::vector<__nv_bfloat16> rms_norms_cpu(input_tokens.size() * EMBEDDING_LENGTH);
    cudaMemcpy(rms_norms_cpu.data(), rms_norms, input_tokens.size() * EMBEDDING_LENGTH * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
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
            if (fabsf(sum) < 1e-4 && fabsf(actual) < 1e-4)
            {
                continue;
            }
            if (rel_err > 1e-1)
            {
                std::cout << "Q MISMATCH token=" << token_idx << " dim=" << j
                          << " expected=" << sum << " got=" << actual
                          << " rel_err=" << rel_err << " layer=" << layer << std::endl;
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

bool verifyAttnScores(__nv_bfloat16 *gpu_q, __nv_bfloat16 *gpu_k, __nv_bfloat16 *gpu_scores, int num_tokens)
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

bool verifyCausalMask(__nv_bfloat16 *gpu_scores, std::vector<__nv_bfloat16> &scores_before_mask, int num_tokens)
{
    std::vector<__nv_bfloat16> scores_cpu(num_tokens * num_tokens * NUM_Q_HEADS);
    cudaMemcpy(scores_cpu.data(), gpu_scores, num_tokens * num_tokens * NUM_Q_HEADS * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    int mismatches = 0;
    for (int h = 0; h < NUM_Q_HEADS; ++h)
    {
        for (int row = 0; row < num_tokens; ++row)
        {
            for (int col = 0; col < num_tokens; ++col)
            {
                int idx = h * num_tokens * num_tokens + row * num_tokens + col;
                float actual = (float)scores_cpu[idx];
                if (col > row)
                {
                    if (!isinf(actual) || actual > 0)
                    {
                        if (mismatches < 10)
                            std::cout << "CAUSAL MASK MISMATCH head=" << h << " row=" << row << " col=" << col
                                      << " expected=-inf got=" << actual << "\n";
                        mismatches++;
                    }
                }
                else
                {
                    float before = (float)scores_before_mask[idx];
                    if (actual != before)
                    {
                        if (mismatches < 10)
                            std::cout << "CAUSAL MASK MISMATCH head=" << h << " row=" << row << " col=" << col
                                      << " value changed from " << before << " to " << actual << "\n";
                        mismatches++;
                    }
                }
            }
        }
    }

    if (mismatches == 0)
        std::cout << "Causal mask verification PASSED\n";
    else
        std::cout << "Causal mask verification FAILED: " << mismatches << " mismatches\n";
    return mismatches == 0;
}

bool verifySoftmax(__nv_bfloat16 *gpu_scores, std::vector<__nv_bfloat16> &scores_before_softmax, int num_tokens)
{
    constexpr float TOLERANCE = 1e-2f;

    std::vector<__nv_bfloat16> scores_cpu(num_tokens * num_tokens * NUM_Q_HEADS);
    cudaMemcpy(scores_cpu.data(), gpu_scores, num_tokens * num_tokens * NUM_Q_HEADS * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    int mismatches = 0;
    for (int h = 0; h < NUM_Q_HEADS; ++h)
    {
        for (int row = 0; row < num_tokens; ++row)
        {
            int row_start = h * num_tokens * num_tokens + row * num_tokens;

            // Find max of this row (from pre-softmax values)
            float max_val = -HUGE_VALF;
            for (int col = 0; col < num_tokens; ++col)
            {
                float val = (float)scores_before_softmax[row_start + col];
                if (val > max_val)
                    max_val = val;
            }

            // Compute exp and sum
            float sum = 0.0f;
            for (int col = 0; col < num_tokens; ++col)
            {
                sum += expf((float)scores_before_softmax[row_start + col] - max_val);
            }

            // Check each element
            float row_sum = 0.0f;
            for (int col = 0; col < num_tokens; ++col)
            {
                float expected = expf((float)scores_before_softmax[row_start + col] - max_val) / sum;
                float actual = (float)scores_cpu[row_start + col];
                row_sum += actual;

                float rel_err = (expected == 0.0f) ? fabsf(actual) : fabsf(actual - expected) / fabsf(expected);
                if (rel_err > TOLERANCE || isnanf(actual))
                {
                    if (mismatches < 10)
                        std::cout << "SOFTMAX MISMATCH head=" << h << " row=" << row << " col=" << col
                                  << " expected=" << expected << " got=" << actual
                                  << " rel_err=" << rel_err << "\n";
                    mismatches++;
                }
            }

            // Each row should sum to ~1.0
            if (fabsf(row_sum - 1.0f) > 0.05f)
            {
                if (mismatches < 10)
                    std::cout << "SOFTMAX ROW SUM MISMATCH head=" << h << " row=" << row
                              << " sum=" << row_sum << " (expected ~1.0)\n";
                mismatches++;
            }
        }
    }

    if (mismatches == 0)
        std::cout << "Softmax verification PASSED\n";
    else
        std::cout << "Softmax verification FAILED: " << mismatches << " mismatches\n";
    return mismatches == 0;
}

bool verifyScoreTimesV(__nv_bfloat16 *gpu_scores, __nv_bfloat16 *gpu_v, __nv_bfloat16 *gpu_output, int num_tokens)
{
    constexpr float TOLERANCE = 1e-1f;

    std::vector<__nv_bfloat16> scores_cpu(num_tokens * num_tokens * NUM_Q_HEADS);
    std::vector<__nv_bfloat16> v_cpu(num_tokens * KV_DIM);
    std::vector<__nv_bfloat16> output_cpu(num_tokens * EMBEDDING_LENGTH);
    cudaMemcpy(scores_cpu.data(), gpu_scores, num_tokens * num_tokens * NUM_Q_HEADS * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    cudaMemcpy(v_cpu.data(), gpu_v, num_tokens * KV_DIM * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    cudaMemcpy(output_cpu.data(), gpu_output, num_tokens * EMBEDDING_LENGTH * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    int mismatches = 0;
    for (int h = 0; h < NUM_Q_HEADS; ++h)
    {
        int v_head = h / GQA_Q_TO_K_RATIO;
        for (int t = 0; t < num_tokens; ++t)
        {
            for (int d = 0; d < HEAD_DIM; ++d)
            {
                float sum = 0.0f;
                for (int t2 = 0; t2 < num_tokens; ++t2)
                {
                    float score = (float)scores_cpu[h * num_tokens * num_tokens + t * num_tokens + t2];
                    float v_val = (float)v_cpu[t2 * KV_DIM + v_head * HEAD_DIM + d];
                    sum += score * v_val;
                }
                float actual = (float)output_cpu[t * EMBEDDING_LENGTH + h * HEAD_DIM + d];

                float rel_err = (sum == 0.0f) ? fabsf(actual) : fabsf(actual - sum) / fabsf(sum);
                if (rel_err > TOLERANCE || isnanf(actual))
                {
                    if (mismatches < 10)
                        std::cout << "SCORE*V MISMATCH head=" << h << " token=" << t << " dim=" << d
                                  << " expected=" << sum << " got=" << actual
                                  << " rel_err=" << rel_err << "\n";
                    mismatches++;
                }
            }
        }
    }

    if (mismatches == 0)
        std::cout << "Score*V verification PASSED\n";
    else
        std::cout << "Score*V verification FAILED: " << mismatches << " mismatches\n";
    return mismatches == 0;
}

bool verifyOProjection(cublasStatus_t gemm_status, std::vector<int> &input_tokens,
                       __nv_bfloat16 *gpu_o_output, __nv_bfloat16 *gpu_attn_scores_v,
                       std::vector<char> &model_weights_cpu,
                       std::unordered_map<std::string, uint64_t> &offsets,
                       int layer)
{
    std::cout << "O projection gemm status: " << gemm_status << std::endl;
    std::vector<__nv_bfloat16> o_cpu(input_tokens.size() * EMBEDDING_LENGTH);
    cudaMemcpy(o_cpu.data(), gpu_o_output, input_tokens.size() * EMBEDDING_LENGTH * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    __nv_bfloat16 *o_weights = (__nv_bfloat16 *)(model_weights_cpu.data() + offsets.at("model.layers." + std::to_string(layer) + ".self_attn.o_proj.weight"));
    std::vector<__nv_bfloat16> input_cpu(input_tokens.size() * EMBEDDING_LENGTH);
    cudaMemcpy(input_cpu.data(), gpu_attn_scores_v, input_tokens.size() * EMBEDDING_LENGTH * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    bool is_correct = true;
    for (int token_idx = 0; token_idx < input_tokens.size(); ++token_idx)
    {
        for (int j = 0; j < EMBEDDING_LENGTH; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < EMBEDDING_LENGTH; ++k)
            {
                float input_value = (float)input_cpu[token_idx * EMBEDDING_LENGTH + k];
                float weight_value = (float)o_weights[j * EMBEDDING_LENGTH + k];
                sum += input_value * weight_value;
            }
            float actual = (float)o_cpu[token_idx * EMBEDDING_LENGTH + j];
            float rel_err = (sum == 0.0f) ? fabsf(actual) : fabsf(actual - sum) / fabsf(sum);
            if (rel_err > 1e-1)
            {
                std::cout << "O MISMATCH token=" << token_idx << " dim=" << j
                          << " expected=" << sum << " got=" << actual
                          << " rel_err=" << rel_err << "\n";
                is_correct = false;
            }
        }
    }
    if (is_correct)
        std::cout << "O projection check done, all correct!" << std::endl;
    else
        std::cout << "O projection check failed!" << std::endl;
    return is_correct;
}

struct Weights
{
    __nv_bfloat16 *embed_tokens;
    __nv_bfloat16 *input_layernorm[N_LAYERS];
    __nv_bfloat16 *mlp_gate_proj[N_LAYERS];
    __nv_bfloat16 *mlp_up_proj[N_LAYERS];
    __nv_bfloat16 *mlp_down_proj[N_LAYERS];
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
#ifdef DEBUG
    if (!verifyModelWeightsCopy(model_weights, model_weights_cpu))
    {
        return 1;
    }
#endif
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
    std::vector<int> input_tokens; // TODO: it's no longer input tokens only, but input tokens + generated tokens, so rename soon to something more relevant, maybe just "tokens" would be better
    // or maybe have two separate vector, I don't know yet
    int token;
    while (std::cin >> token)
    {
        input_tokens.push_back(token);
    }
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
#ifdef DEBUG
    cudaDeviceSynchronize();
    if (!verifyEmbeddingGather(input_tokens, input_embeddings, model_weights_cpu, offsets))
    {
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

    __nv_bfloat16 *hidden_state;
    cudaMalloc(&hidden_state, input_tokens.size() * sizeof(__nv_bfloat16) * EMBEDDING_LENGTH);
    cudaMemcpy(hidden_state, input_embeddings, input_tokens.size() * EMBEDDING_LENGTH * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);

    __nv_bfloat16 *rms_norms;
    cudaMalloc(&rms_norms, input_tokens.size() * sizeof(__nv_bfloat16) * EMBEDDING_LENGTH);

    __nv_bfloat16 *buf_2048_1; // shared between q_proj and attn_scores_v
    cudaMalloc(&buf_2048_1, input_tokens.size() * sizeof(__nv_bfloat16) * EMBEDDING_LENGTH);
    __nv_bfloat16 *q_proj;
    float q_proj_alpha = 1.0f;
    float q_proj_beta = 0.0f;

    // K and V cache
    __nv_bfloat16 *k_proj[N_LAYERS];
    __nv_bfloat16 *v_proj[N_LAYERS];
    for (int layer = 0; layer < N_LAYERS; ++layer)
    {
        cudaMalloc(&k_proj[layer], MAX_SEQ_LEN * sizeof(__nv_bfloat16) * KV_DIM);
        cudaMalloc(&v_proj[layer], MAX_SEQ_LEN * sizeof(__nv_bfloat16) * KV_DIM);
    }
    float k_proj_alpha = 1.0f;
    float k_proj_beta = 0.0f;

    float v_proj_alpha = 1.0f;
    float v_proj_beta = 0.0f;

    __nv_bfloat16 *attn_scores;
    cudaMalloc(&attn_scores, input_tokens.size() * input_tokens.size() * sizeof(__nv_bfloat16) * NUM_Q_HEADS);
    float attn_alpha = 1.0f / 8.0f;
    float attn_beta = 0.0f;

    __nv_bfloat16 *attn_scores_v;
    float attn_scores_v_alpha = 1.0f;
    float attn_scores_v_beta = 0.0f;

    __nv_bfloat16 *buf_2048_2; // shared between o_proj and down
    cudaMalloc(&buf_2048_2, input_tokens.size() * sizeof(__nv_bfloat16) * EMBEDDING_LENGTH);
    __nv_bfloat16 *o_proj;
    float o_proj_alpha = 1.0f;
    float o_proj_beta = 0.0f;

    __nv_bfloat16 *gate;
    cudaMalloc(&gate, input_tokens.size() * sizeof(__nv_bfloat16) * HIDDEN_DIM);
    float gate_alpha = 1.0f;
    float gate_beta = 0.0f;

    __nv_bfloat16 *up;
    cudaMalloc(&up, input_tokens.size() * sizeof(__nv_bfloat16) * HIDDEN_DIM);
    float up_alpha = 1.0f;
    float up_beta = 0.0f;

    __nv_bfloat16 *down;
    float down_alpha = 1.0f;
    float down_beta = 0.0f;

    // PREFILL
    for (int layer = 0; layer < N_LAYERS; ++layer)
    {
#ifdef DEBUG
        std::vector<__nv_bfloat16> hs_debug(10);
        cudaMemcpy(hs_debug.data(), hidden_state, 10 * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
        std::cout << "Layer " << layer << " input first 10 values (token 0):" << std::endl;
        for (int i = 0; i < 10; ++i)
            std::cout << "  [" << i << "] = " << (float)hs_debug[i] << std::endl;
#endif
        rmsNorm(hidden_state, rms_norms, weights.input_layernorm[layer], input_tokens.size());
#ifdef DEBUG
        cudaDeviceSynchronize();
        if (!verifyRMSNormWeights(model_weights_cpu, offsets, layer) || !verifyRmsNorm(hidden_state, rms_norms, model_weights_cpu, offsets, input_tokens.size(), layer))
        {
            std::cout << "RMS norm verification failed" << std::endl;
            return 1;
        }
#endif

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
        q_proj = buf_2048_1;
        cublasStatus_t q_proj_status = cublasGemmEx(cublas_handle,
                                                    CUBLAS_OP_T,
                                                    CUBLAS_OP_N,
                                                    EMBEDDING_LENGTH,
                                                    input_tokens.size(),
                                                    EMBEDDING_LENGTH,
                                                    &q_proj_alpha,
                                                    weights.w_q[layer],
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
#ifdef DEBUG
        cudaDeviceSynchronize();
        verifyQProjection(q_proj_status, input_tokens, q_proj, model_weights_cpu, offsets, rms_norms, layer);
#endif

        // input = (num_tokens, EMBEDDING_LENGTH), weights = (KV_DIM, EMBEDDING_LENGTH)
        // after trick: (KV_DIM, EMBEDDING_LENGTH) * (EMBEDDING_LENGTH, num_tokens) -> (KV_DIM, num_tokens), which really is (num_tok, KV_DIM)
        // lda: EMBEDDING_LENGTH, ldb: EMBEDDING_LENGTH, ldc: KV_DIM
        cublasStatus_t k_proj_status = cublasGemmEx(cublas_handle,
                                                    CUBLAS_OP_T,
                                                    CUBLAS_OP_N,
                                                    KV_DIM,
                                                    input_tokens.size(),
                                                    EMBEDDING_LENGTH,
                                                    &k_proj_alpha,
                                                    weights.w_k[layer],
                                                    CUDA_R_16BF,
                                                    EMBEDDING_LENGTH,
                                                    rms_norms,
                                                    CUDA_R_16BF,
                                                    EMBEDDING_LENGTH,
                                                    &k_proj_beta,
                                                    k_proj[layer],
                                                    CUDA_R_16BF,
                                                    KV_DIM,
                                                    CUBLAS_COMPUTE_32F,
                                                    CUBLAS_GEMM_DEFAULT);

        // same as K projection
        cublasStatus_t v_proj_status = cublasGemmEx(cublas_handle,
                                                    CUBLAS_OP_T,
                                                    CUBLAS_OP_N,
                                                    KV_DIM,
                                                    input_tokens.size(),
                                                    EMBEDDING_LENGTH,
                                                    &v_proj_alpha,
                                                    weights.w_v[layer],
                                                    CUDA_R_16BF,
                                                    EMBEDDING_LENGTH,
                                                    rms_norms,
                                                    CUDA_R_16BF,
                                                    EMBEDDING_LENGTH,
                                                    &v_proj_beta,
                                                    v_proj[layer],
                                                    CUDA_R_16BF,
                                                    KV_DIM,
                                                    CUBLAS_COMPUTE_32F,
                                                    CUBLAS_GEMM_DEFAULT);

        // RoPE now
#ifdef DEBUG
        std::vector<__nv_bfloat16> q_before_rope(input_tokens.size() * EMBEDDING_LENGTH);
        std::vector<__nv_bfloat16> k_before_rope(input_tokens.size() * KV_DIM);

        cudaMemcpy(q_before_rope.data(), q_proj, input_tokens.size() * EMBEDDING_LENGTH * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
        cudaMemcpy(k_before_rope.data(), k_proj, input_tokens.size() * KV_DIM * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
#endif
        rope(q_proj, input_tokens.size(), EMBEDDING_LENGTH);
        rope(k_proj[layer], input_tokens.size(), KV_DIM);
#ifdef DEBUG
        cudaDeviceSynchronize();

        verifyRope(q_proj, k_proj, q_before_rope, k_before_rope, input_tokens.size());
#endif
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
        // total output (32, num_tok, num_tok)
        for (int i = 0; i < NUM_Q_HEADS; ++i)
        {
            int k_head_idx = i / GQA_Q_TO_K_RATIO;
            __nv_bfloat16 *q_head = q_proj + i * HEAD_DIM;
            __nv_bfloat16 *k_head = k_proj[layer] + k_head_idx * HEAD_DIM;
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
#ifdef DEBUG
        cudaDeviceSynchronize();
        verifyAttnScores(q_proj, k_proj, attn_scores, input_tokens.size());

        std::vector<__nv_bfloat16> scores_before_mask(input_tokens.size() * input_tokens.size() * NUM_Q_HEADS);
        cudaMemcpy(scores_before_mask.data(), attn_scores, input_tokens.size() * input_tokens.size() * NUM_Q_HEADS * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
#endif
        causalMask(attn_scores, input_tokens.size());
#ifdef DEBUG
        cudaDeviceSynchronize();
        verifyCausalMask(attn_scores, scores_before_mask, input_tokens.size());

        std::vector<__nv_bfloat16> scores_before_softmax(input_tokens.size() * input_tokens.size() * NUM_Q_HEADS);
        cudaMemcpy(scores_before_softmax.data(), attn_scores, input_tokens.size() * input_tokens.size() * NUM_Q_HEADS * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
#endif
        softmax(attn_scores, input_tokens.size());
#ifdef DEBUG
        cudaDeviceSynchronize();
        verifySoftmax(attn_scores, scores_before_softmax, input_tokens.size());
#endif

        // attn scores * V
        // (32, num_tok, num_tok) * (num_tok, 512)
        // GQA - 4 Q heads share 1 V head
        // attn_scores dim (32, num_tok, num_tok)
        // attn_scores head dim (num_tok, num_tok)
        // V dim (num_tok, 512)
        // NUM_V_HEADS is 8 -> 512 / 8 = 64
        // V_head dim (num_tok, 64)
        // output head dim: scores head * V head -> (num_tok, num_tok) * (num_tok, 64) = (num_tok, 64)
        // in total 32 output heads: so (num_tok, 64 * 32) = (num_tok, 2048)
        attn_scores_v = buf_2048_1;
        for (int i = 0; i < NUM_Q_HEADS; ++i)
        {
            int v_head_idx = i / GQA_ATTN_SCORES_TO_V_RATIO;
            // i * input_tokens.size() * input_tokens.size(),  because attn scores is (32, num_tok, num_tok)
            __nv_bfloat16 *attn_scores_head = attn_scores + i * input_tokens.size() * input_tokens.size();
            __nv_bfloat16 *v_head = v_proj[layer] + v_head_idx * HEAD_DIM;
            __nv_bfloat16 *output_attn_scores_head = attn_scores_v + i * HEAD_DIM;

            cublasStatus_t attn_score_status = cublasGemmEx(cublas_handle,
                                                            CUBLAS_OP_N,
                                                            CUBLAS_OP_N,
                                                            HEAD_DIM,
                                                            input_tokens.size(),
                                                            input_tokens.size(),
                                                            &attn_scores_v_alpha,
                                                            v_head,
                                                            CUDA_R_16BF,
                                                            KV_DIM,
                                                            attn_scores_head,
                                                            CUDA_R_16BF,
                                                            input_tokens.size(),
                                                            &attn_scores_v_beta,
                                                            output_attn_scores_head,
                                                            CUDA_R_16BF,
                                                            EMBEDDING_LENGTH,
                                                            CUBLAS_COMPUTE_32F,
                                                            CUBLAS_GEMM_DEFAULT);
        }
#ifdef DEBUG
        cudaDeviceSynchronize();
        verifyScoreTimesV(attn_scores, v_proj, attn_scores_v, input_tokens.size());
#endif

        // output projection, it will be an input for MLP blocks
        // attn_scores_v * w_o^T
        // (num_tok, 2048) * (2048, 2048) -> (num_tok, 2048)
        // same as Q projection, so copy paste
        o_proj = buf_2048_2;
        cublasStatus_t o_proj_status = cublasGemmEx(cublas_handle,
                                                    CUBLAS_OP_T,
                                                    CUBLAS_OP_N,
                                                    EMBEDDING_LENGTH,
                                                    input_tokens.size(),
                                                    EMBEDDING_LENGTH,
                                                    &o_proj_alpha,
                                                    weights.w_o[layer],
                                                    CUDA_R_16BF,
                                                    EMBEDDING_LENGTH,
                                                    attn_scores_v,
                                                    CUDA_R_16BF,
                                                    EMBEDDING_LENGTH,
                                                    &o_proj_beta,
                                                    o_proj,
                                                    CUDA_R_16BF,
                                                    EMBEDDING_LENGTH,
                                                    CUBLAS_COMPUTE_32F,
                                                    CUBLAS_GEMM_DEFAULT);
#ifdef DEBUG
        cudaDeviceSynchronize();
        verifyOProjection(o_proj_status, input_tokens, o_proj, attn_scores_v, model_weights_cpu, offsets, layer);
#endif
        // (num_tok, 2048) + (num_tok, 2048) -> (num_tok, 2048)
        residualAdd(hidden_state, o_proj, input_tokens.size());
        // post attention RMS Norm
        rmsNorm(hidden_state, rms_norms, weights.post_attn_layernorms[layer], input_tokens.size());

        // SwiGLU time - just MLP + SiLU
        // gate = hidden_state (rms-normed) * mlp_gate_proj ^ T
        // HIDDEN_DIM = 8192
        // (num_tok, 2048) * (2048, 8192) -> (num_tok, 8192)
        // my data is row major so transpose trick
        // gate ^T = (mlp_gate_proj ^ T)^T * hidden_state^T
        // gate ^T = mlp_gate_proj * hidden_state^T
        // (num_tok, 8192)^T = (8192, 2048) * (2048, num_tok)
        // but data is perceived as column major so I need to transpose mlp_gate_proj
        // to make it work
        // m 8192 n num_tok k 2048 lda 2048 ldb 2048 ldc 8192
        cublasStatus_t gate_status = cublasGemmEx(cublas_handle,
                                                  CUBLAS_OP_T,
                                                  CUBLAS_OP_N,
                                                  HIDDEN_DIM,
                                                  input_tokens.size(),
                                                  EMBEDDING_LENGTH,
                                                  &gate_alpha,
                                                  weights.mlp_gate_proj[layer],
                                                  CUDA_R_16BF,
                                                  EMBEDDING_LENGTH,
                                                  rms_norms,
                                                  CUDA_R_16BF,
                                                  EMBEDDING_LENGTH,
                                                  &gate_beta,
                                                  gate,
                                                  CUDA_R_16BF,
                                                  HIDDEN_DIM,
                                                  CUBLAS_COMPUTE_32F,
                                                  CUBLAS_GEMM_DEFAULT);

        // up, the same dims as gate
        cublasStatus_t up_status = cublasGemmEx(cublas_handle,
                                                CUBLAS_OP_T,
                                                CUBLAS_OP_N,
                                                HIDDEN_DIM,
                                                input_tokens.size(),
                                                EMBEDDING_LENGTH,
                                                &up_alpha,
                                                weights.mlp_up_proj[layer],
                                                CUDA_R_16BF,
                                                EMBEDDING_LENGTH,
                                                rms_norms,
                                                CUDA_R_16BF,
                                                EMBEDDING_LENGTH,
                                                &up_beta,
                                                up,
                                                CUDA_R_16BF,
                                                HIDDEN_DIM,
                                                CUBLAS_COMPUTE_32F,
                                                CUBLAS_GEMM_DEFAULT);

        // SiLU
        // after_silu = SiLU(gate) * up (element-wise multication)
        // after_silu = gate * (1 / (1 + e^(-gate))) * up
        // gate is dim (num_tok, 8192), up too
        silu(gate, up, input_tokens.size()); // gate = after_silu now

        // down projection
        // output = post-silu * down_proj^T
        // dims: (num_tok, 8192) * (2048, 8192) ^ T = (num_tok, 8192) * (8192, 2048) = (num_tok, 2048)
        // output^T = (down_proj^T)^T * post-silu^T
        // output^T = down_proj * post-silu^T
        // cublas sees them already as transposed so only down_proj I need to transpose
        // dims = (2048, 8192) * (8192, num_tok) = (2048, num_tok)
        // m: 2048 n: num_tok, k: 8192
        // lda: 8192, ldb: 8192, ldc: 2048
        down = buf_2048_2;
        cublasStatus_t down_status = cublasGemmEx(cublas_handle,
                                                  CUBLAS_OP_T,
                                                  CUBLAS_OP_N,
                                                  EMBEDDING_LENGTH,
                                                  input_tokens.size(),
                                                  HIDDEN_DIM,
                                                  &down_alpha,
                                                  weights.mlp_down_proj[layer],
                                                  CUDA_R_16BF,
                                                  HIDDEN_DIM,
                                                  gate,
                                                  CUDA_R_16BF,
                                                  HIDDEN_DIM,
                                                  &down_beta,
                                                  down,
                                                  CUDA_R_16BF,
                                                  EMBEDDING_LENGTH,
                                                  CUBLAS_COMPUTE_32F,
                                                  CUBLAS_GEMM_DEFAULT);

        // (num_tok, 2048) + (num_tok, 2048) -> (num_tok, 2048)
        residualAdd(hidden_state, down, input_tokens.size());
    }
    rmsNorm(hidden_state, rms_norms, weights.norm, input_tokens.size());

    // logits = rms_norms * weights.embed_tokens^T
    // dim rms_norms: (num_tok, 2048), dim embed_tokens: (128256, 2048)
    // logits dim = (num_tok, 2048) * (2048, 128256) = (num_tok, 128256) => m = num_tok, n = 128256, k = 2048
    // I leave this comment above because it shows a bug in my thinking
    // because I use the cublas trick, logits are transposed so m and n should be swapped
    // so m 128256, n num_tok
    // data is row major so we treat it as transposed and use the trick
    // logits^T = ((weights.embed_tokens^T)^T * rms_norms^T
    // logits^T = weights.embed_tokens * rms_norms^T
    // so we need to transpose embed_tokens, because rms_norms already
    // appears to cublas as transposed
    // lda = 2048, ldb = 2048, ldc = 128256
    __nv_bfloat16 *embed_proj;
    cudaMalloc(&embed_proj, sizeof(__nv_bfloat16) * input_tokens.size() * VOCAB_SIZE);
    float embed_alpha = 1.0f;
    float embed_beta = 0.0f;
    cublasStatus_t embed_status = cublasGemmEx(cublas_handle,
                                               CUBLAS_OP_T,
                                               CUBLAS_OP_N,
                                               VOCAB_SIZE,
                                               input_tokens.size(),
                                               EMBEDDING_LENGTH,
                                               &embed_alpha,
                                               weights.embed_tokens,
                                               CUDA_R_16BF,
                                               EMBEDDING_LENGTH,
                                               rms_norms,
                                               CUDA_R_16BF,
                                               EMBEDDING_LENGTH,
                                               &embed_beta,
                                               embed_proj,
                                               CUDA_R_16BF,
                                               VOCAB_SIZE,
                                               CUBLAS_COMPUTE_32F,
                                               CUBLAS_GEMM_DEFAULT);

    std::vector<__nv_bfloat16> embed_proj_cpu;
    embed_proj_cpu.resize(input_tokens.size() * VOCAB_SIZE);
    cudaMemcpy(embed_proj_cpu.data(), embed_proj, sizeof(__nv_bfloat16) * input_tokens.size() * VOCAB_SIZE, cudaMemcpyDeviceToHost);
    // argmax to get the output token
    // TODO: write a proper kernel for it
    // for now just a simple CPU function
    int last_token_offset = (input_tokens.size() - 1) * VOCAB_SIZE;
    float max_token = (float)embed_proj_cpu[last_token_offset];
    int max_token_idx = 0;
    for (int token_idx = 0; token_idx < VOCAB_SIZE; ++token_idx)
    {
        if ((float)embed_proj_cpu[token_idx + last_token_offset] > max_token)
        {
            max_token = embed_proj_cpu[token_idx + last_token_offset];
            max_token_idx = token_idx;
        }
    }
    std::cout << "Output token: " << (float)max_token << ", token index: " << std::to_string(max_token_idx) << std::endl;

    cudaFree(attn_scores);
    cudaMalloc(&attn_scores, sizeof(__nv_bfloat16) * MAX_SEQ_LEN * NUM_Q_HEADS);
    // DECODE
    // since now I operate always on index 0 for all values and for current_position_token for new K and V

    int last_generated_token = max_token_idx;
    std::vector<int> generated_tokens;
    generated_tokens.reserve(MAX_SEQ_LEN - input_tokens.size());
    generated_tokens.push_back(last_generated_token);
    int current_token_position = input_tokens.size();

    while (last_generated_token != END_OF_TEXT_TOKEN_ID && last_generated_token != EOT_ID_TOKEN_ID && current_token_position < MAX_SEQ_LEN)
    {
        // TODO: make it more suitable for single token operations, perhaps just pass a token id as param
        embeddingGatherDecode(last_generated_token, hidden_state, weights.embed_tokens);
        for (int layer = 0; layer < N_LAYERS; ++layer)
        {
            rmsNorm(hidden_state, rms_norms, weights.input_layernorm[layer], 1);
            q_proj = buf_2048_1;
            // q proj (1, 2048)
            cublasGemmEx(cublas_handle,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         EMBEDDING_LENGTH, // m
                         1,                // n
                         EMBEDDING_LENGTH, // k
                         &q_proj_alpha,
                         weights.w_q[layer], // A
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH, // lda
                         rms_norms,        // B
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH, // ldb
                         &q_proj_beta,
                         q_proj, // C
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH, // ldc
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT);
            // k proj (1, 512), writing output to next position in current layer's K cache
            cublasGemmEx(cublas_handle,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         KV_DIM,
                         1,
                         EMBEDDING_LENGTH,
                         &k_proj_alpha,
                         weights.w_k[layer],
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH,
                         rms_norms,
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH,
                         &k_proj_beta,
                         k_proj[layer] + current_token_position * KV_DIM,
                         CUDA_R_16BF,
                         KV_DIM,
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT);
            // same
            cublasGemmEx(cublas_handle,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         KV_DIM,
                         1,
                         EMBEDDING_LENGTH,
                         &v_proj_alpha,
                         weights.w_v[layer],
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH,
                         rms_norms,
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH,
                         &v_proj_beta,
                         v_proj[layer] + current_token_position * KV_DIM,
                         CUDA_R_16BF,
                         KV_DIM,
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT);
            ropeDecode(q_proj, current_token_position, EMBEDDING_LENGTH);
            ropeDecode(k_proj[layer] + current_token_position * KV_DIM, current_token_position, KV_DIM);

            int seq_len = current_token_position + 1;
            for (int i = 0; i < NUM_Q_HEADS; ++i)
            {
                int k_head_idx = i / GQA_Q_TO_K_RATIO;
                __nv_bfloat16 *q_head = q_proj + i * HEAD_DIM;                 // (1, 64)
                __nv_bfloat16 *k_head = k_proj[layer] + k_head_idx * HEAD_DIM; // (current_token_position+1, 64)
                __nv_bfloat16 *attn_score_head = attn_scores + MAX_SEQ_LEN * i;

                cublasGemmEx(cublas_handle,
                             CUBLAS_OP_T,
                             CUBLAS_OP_N,
                             seq_len,  // m
                             1,        // n
                             HEAD_DIM, // k
                             &attn_alpha,
                             k_head,
                             CUDA_R_16BF,
                             KV_DIM, // lda
                             q_head,
                             CUDA_R_16BF,
                             EMBEDDING_LENGTH, // ldb
                             &attn_beta,
                             attn_score_head,
                             CUDA_R_16BF,
                             MAX_SEQ_LEN, // ldc
                             CUBLAS_COMPUTE_32F,
                             CUBLAS_GEMM_DEFAULT);
            }
            softmaxDecode(attn_scores, 1);

            attn_scores_v = buf_2048_1;
            for (int i = 0; i < NUM_Q_HEADS; ++i)
            {
                int v_head_idx = i / GQA_ATTN_SCORES_TO_V_RATIO;
                __nv_bfloat16 *attn_scores_head = attn_scores + i * MAX_SEQ_LEN;
                __nv_bfloat16 *v_head = v_proj[layer] + v_head_idx * HEAD_DIM;
                __nv_bfloat16 *output_attn_scores_head = attn_scores_v + i * HEAD_DIM;

                cublasGemmEx(cublas_handle,
                             CUBLAS_OP_N,
                             CUBLAS_OP_N,
                             HEAD_DIM, // m
                             1,        // n
                             seq_len,  // k
                             &attn_scores_v_alpha,
                             v_head,
                             CUDA_R_16BF,
                             KV_DIM, // lda
                             attn_scores_head,
                             CUDA_R_16BF,
                             MAX_SEQ_LEN, // ldb
                             &attn_scores_v_beta,
                             output_attn_scores_head,
                             CUDA_R_16BF,
                             EMBEDDING_LENGTH, // ldc
                             CUBLAS_COMPUTE_32F,
                             CUBLAS_GEMM_DEFAULT);
            }

            o_proj = buf_2048_2;
            // (1, 2048) * (2048, 2048) -> (1, 2048)
            cublasGemmEx(cublas_handle,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         EMBEDDING_LENGTH, // m
                         1,                // n
                         EMBEDDING_LENGTH, // k
                         &o_proj_alpha,
                         weights.w_o[layer],
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH,
                         attn_scores_v,
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH,
                         &o_proj_beta,
                         o_proj,
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH,
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT);

            residualAdd(hidden_state, o_proj, 1);

            rmsNorm(hidden_state, rms_norms, weights.post_attn_layernorms[layer], 1);

            // (1, 2048) * (2048, 8192) -> (1, 8192)
            cublasGemmEx(cublas_handle,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         HIDDEN_DIM,       // m
                         1,                // n
                         EMBEDDING_LENGTH, // k
                         &gate_alpha,
                         weights.mlp_gate_proj[layer],
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH,
                         rms_norms,
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH,
                         &gate_beta,
                         gate,
                         CUDA_R_16BF,
                         HIDDEN_DIM,
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT);

            // (1, 2048) * (2048, 8192) -> (1, 8192)
            cublasGemmEx(cublas_handle,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         HIDDEN_DIM,       // m
                         1,                // n
                         EMBEDDING_LENGTH, // k
                         &up_alpha,
                         weights.mlp_up_proj[layer],
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH,
                         rms_norms,
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH,
                         &up_beta,
                         up,
                         CUDA_R_16BF,
                         HIDDEN_DIM,
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT);

            silu(gate, up, 1);

            down = buf_2048_2;
            cublasGemmEx(cublas_handle,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         EMBEDDING_LENGTH, // m
                         1,                // n
                         HIDDEN_DIM,       // k
                         &down_alpha,
                         weights.mlp_down_proj[layer],
                         CUDA_R_16BF,
                         HIDDEN_DIM,
                         gate,
                         CUDA_R_16BF,
                         HIDDEN_DIM,
                         &down_beta,
                         down,
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH,
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT);

            residualAdd(hidden_state, down, 1);
        }

        rmsNorm(hidden_state, rms_norms, weights.norm, 1);

        cublasGemmEx(cublas_handle,
                     CUBLAS_OP_T,
                     CUBLAS_OP_N,
                     VOCAB_SIZE,       // m
                     1,                // n
                     EMBEDDING_LENGTH, // k
                     &embed_alpha,
                     weights.embed_tokens,
                     CUDA_R_16BF,
                     EMBEDDING_LENGTH,
                     rms_norms,
                     CUDA_R_16BF,
                     EMBEDDING_LENGTH,
                     &embed_beta,
                     embed_proj,
                     CUDA_R_16BF,
                     VOCAB_SIZE,
                     CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT);

        cudaMemcpy(embed_proj_cpu.data(), embed_proj, sizeof(__nv_bfloat16) * VOCAB_SIZE, cudaMemcpyDeviceToHost);
        max_token = (float)embed_proj_cpu[0];
        max_token_idx = 0;
        for (int token_idx = 0; token_idx < VOCAB_SIZE; ++token_idx)
        {
            if ((float)embed_proj_cpu[token_idx] > max_token)
            {
                max_token = embed_proj_cpu[token_idx];
                max_token_idx = token_idx;
            }
        }
        std::cout << "Output token: " << (float)max_token << ", token index: " << std::to_string(max_token_idx) << std::endl;

        last_generated_token = max_token_idx;
        generated_tokens.push_back(last_generated_token);
        current_token_position += 1;
    }
    std::cout << "\nOk bye!\n";
    cublasDestroy(cublas_handle);
    cudaDeviceSynchronize();
    return 0;
}

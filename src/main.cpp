#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <queue>
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
constexpr int BATCH_SIZE = 2;                // TODO: not even close to being good

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

int loadWeights(Weights &weights)
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
    // READ SAFETENSORS HEADER
    std::string header;
    header.resize(header_size);
    safetensors_file.read(header.data(), header_size);
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
    safetensors_file.close();
    // BASICALLY A HELPER STRUCT TO HAVE AN EASY ACCESS TO ANY MODEL WEIGHTS ON GPU
    // TODO: right now I know the model structure since it's always llama 3.2 1B-Instruct, but maybe it would be convenient
    //       to store dimensions somewhere for even easier access?
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
    return 0;
}

int main(int argc, char *argv[])
{
    Weights weights{};
    if (loadWeights(weights) != 0)
    {
        return 1;
    }

    // PROMPT 0 (What is 2+2?) - length 17
    std::queue<std::vector<int>> queue;
    queue.push({128000, 128006, 882, 128007, 271, 3923, 374, 220, 17, 10, 17, 30, 128009, 128006, 78191, 128007, 271});

    // PROMPT 1 (Name a color.) - length 14
    queue.push({128000, 128006, 882, 128007, 271, 678, 264, 1933, 13, 128009, 128006, 78191, 128007, 271});

    // PROMPT 2 (Say hello.) - length 13
    queue.push({128000, 128006, 882, 128007, 271, 46864, 24748, 13, 128009, 128006, 78191, 128007, 271});

    // PROMPT 3 (Capital of France?) - length 14
    queue.push({128000, 128006, 882, 128007, 271, 64693, 315, 9822, 30, 128009, 128006, 78191, 128007, 271});

    // BATCH
    std::vector<bool> slots_availability(BATCH_SIZE, true); // set to false when slot taken, set to true when free
    std::vector<int> input_tokens();                        // all tokens from all batch items concatenated; NOW IT REPRESENTS CURRENT BATCH, NOT ALL TOKENS
    int input_tokens_size;

    std::vector<int> prompt_offsets; // indicate where the next prompt starts - same size as prompt_lengths, they have to match by index
    // TODO: recalculate prompt_offsets, input_tokens_size and prompt_lengths always when there is a change to input_tokens
    std::vector<int> prompt_lengths; // indicates length of each prompt
    // TODO: right now I handle input manually, it's the least interesting part, will come back to it when continuous batching and pagedattn works

    std::vector<int> queue_front;

    for (int slot = 0; slot < slots_availability.size(); ++i)
    {
        if (!slots_availability[slot])
        {
            continue; // slot taken, skip
        }
        if (queue.size() > 0)
        {
            queue_front = queue.front();
            auto slot_position_in_batch = input_tokens[slot * EMBEDDING_LENGTH];
            for (int j = 0; j < queue_front.size(); j++)
            {
                input_tokens.push_back(queue_front[j]); // TODO: inefficient
            }
            queue.pop();
            slots_availability[slot] = false;
        }
    }

    prompt_offsets.push_back(0);
    prompt_lengths.push_back(17);

    prompt_offsets.push_back(17);
    prompt_lengths.push_back(14);

    prompt_offsets.push_back(31);
    prompt_lengths.push_back(13);

    prompt_offsets.push_back(44);
    prompt_lengths.push_back(14);

    input_tokens_size = input_tokens.size();
    int max_input_len = 17;                                     // yes, I set it manualy for now. TODO automate it
    int num_prompts = 4;                                        // same
    int max_buffer_size = std::max(max_input_len, num_prompts); // it will make more sense once the aboves are not hardcoded

    int *gpu_input_tokens;
    cudaMalloc(&gpu_input_tokens, input_tokens_size * sizeof(int));
    cudaMemcpy(gpu_input_tokens, input_tokens.data(), input_tokens_size * sizeof(int), cudaMemcpyHostToDevice);

    // INFERENCE STARTS HERE! =]
    // I have the same amount of embeddings as input tokens
    // it's just every embedding is EMBEDDING_LENGTH length bf16 vector
    // retrieved from model weights based on token's value

    __nv_bfloat16 *input_embeddings;
    cudaMalloc(&input_embeddings, input_tokens_size * sizeof(__nv_bfloat16) * EMBEDDING_LENGTH);
    embeddingGather(gpu_input_tokens, input_embeddings, weights.embed_tokens, input_tokens.size());

    cublasHandle_t cublas_handle;
    cublasStatus_t status = cublasCreate(&cublas_handle);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "cuBLAS init failed, status: " << status << "\n";
        return 1;
    }

    __nv_bfloat16 *hidden_state;
    cudaMalloc(&hidden_state, max_buffer_size * sizeof(__nv_bfloat16) * EMBEDDING_LENGTH);

    __nv_bfloat16 *rms_norms;
    cudaMalloc(&rms_norms, max_buffer_size * sizeof(__nv_bfloat16) * EMBEDDING_LENGTH);

    __nv_bfloat16 *buf_2048_1; // shared between q_proj and attn_scores_v
    cudaMalloc(&buf_2048_1, max_buffer_size * sizeof(__nv_bfloat16) * EMBEDDING_LENGTH);
    __nv_bfloat16 *q_proj;
    float q_proj_alpha = 1.0f;
    float q_proj_beta = 0.0f;

    // K and V cache
    __nv_bfloat16 *k_proj[BATCH_SIZE][N_LAYERS];
    __nv_bfloat16 *v_proj[BATCH_SIZE][N_LAYERS];
    for (int i = 0; i < BATCH_SIZE * N_LAYERS; ++i)
    {
        int batch_idx = i / N_LAYERS;
        int layer_idx = i % N_LAYERS;
        cudaMalloc(&k_proj[batch_idx][layer_idx], MAX_SEQ_LEN * sizeof(__nv_bfloat16) * KV_DIM);
        cudaMalloc(&v_proj[batch_idx][layer_idx], MAX_SEQ_LEN * sizeof(__nv_bfloat16) * KV_DIM);
    }
    float k_proj_alpha = 1.0f;
    float k_proj_beta = 0.0f;

    float v_proj_alpha = 1.0f;
    float v_proj_beta = 0.0f;

    __nv_bfloat16 *attn_scores;
    cudaMalloc(&attn_scores, max_input_len * max_input_len * sizeof(__nv_bfloat16) * NUM_Q_HEADS);
    float attn_alpha = 1.0f / 8.0f;
    float attn_beta = 0.0f;

    __nv_bfloat16 *attn_scores_v;
    float attn_scores_v_alpha = 1.0f;
    float attn_scores_v_beta = 0.0f;

    __nv_bfloat16 *buf_2048_2; // shared between o_proj and down
    cudaMalloc(&buf_2048_2, max_buffer_size * sizeof(__nv_bfloat16) * EMBEDDING_LENGTH);
    __nv_bfloat16 *o_proj;
    float o_proj_alpha = 1.0f;
    float o_proj_beta = 0.0f;

    __nv_bfloat16 *gate;
    cudaMalloc(&gate, max_buffer_size * sizeof(__nv_bfloat16) * HIDDEN_DIM);
    float gate_alpha = 1.0f;
    float gate_beta = 0.0f;

    __nv_bfloat16 *up;
    cudaMalloc(&up, max_buffer_size * sizeof(__nv_bfloat16) * HIDDEN_DIM);
    float up_alpha = 1.0f;
    float up_beta = 0.0f;

    __nv_bfloat16 *down;
    float down_alpha = 1.0f;
    float down_beta = 0.0f;

    __nv_bfloat16 *embed_proj;
    cudaMalloc(&embed_proj, sizeof(__nv_bfloat16) * max_buffer_size * VOCAB_SIZE);
    float embed_alpha = 1.0f;
    float embed_beta = 0.0f;

    std::vector<__nv_bfloat16> embed_proj_cpu;
    embed_proj_cpu.resize(max_buffer_size * VOCAB_SIZE);

    std::vector<std::vector<int>> generated_tokens(num_prompts);
    std::vector<int> last_generated_tokens(num_prompts);
    std::vector<int> current_prompt_len(num_prompts);
    std::vector<bool> is_prompt_finished(num_prompts, false); // all of these values identified by the same id

    // PREFILL
    for (int prompt_id = 0; prompt_id < num_prompts; ++prompt_id)
    {
        cudaMemcpy(hidden_state,
                   input_embeddings + prompt_offsets[prompt_id] * EMBEDDING_LENGTH,
                   prompt_lengths[prompt_id] * EMBEDDING_LENGTH * sizeof(__nv_bfloat16),
                   cudaMemcpyDeviceToDevice);
        for (int layer = 0; layer < N_LAYERS; ++layer)
        {

            rmsNorm(hidden_state, rms_norms, weights.input_layernorm[layer], prompt_lengths[prompt_id]);

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
                                                        prompt_lengths[prompt_id],
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

            // input = (num_tokens, EMBEDDING_LENGTH), weights = (KV_DIM, EMBEDDING_LENGTH)
            // after trick: (KV_DIM, EMBEDDING_LENGTH) * (EMBEDDING_LENGTH, num_tokens) -> (KV_DIM, num_tokens), which really is (num_tok, KV_DIM)
            // lda: EMBEDDING_LENGTH, ldb: EMBEDDING_LENGTH, ldc: KV_DIM
            cublasStatus_t k_proj_status = cublasGemmEx(cublas_handle,
                                                        CUBLAS_OP_T,
                                                        CUBLAS_OP_N,
                                                        KV_DIM,
                                                        prompt_lengths[prompt_id],
                                                        EMBEDDING_LENGTH,
                                                        &k_proj_alpha,
                                                        weights.w_k[layer],
                                                        CUDA_R_16BF,
                                                        EMBEDDING_LENGTH,
                                                        rms_norms,
                                                        CUDA_R_16BF,
                                                        EMBEDDING_LENGTH,
                                                        &k_proj_beta,
                                                        k_proj[prompt_id][layer],
                                                        CUDA_R_16BF,
                                                        KV_DIM,
                                                        CUBLAS_COMPUTE_32F,
                                                        CUBLAS_GEMM_DEFAULT);

            // same as K projection
            cublasStatus_t v_proj_status = cublasGemmEx(cublas_handle,
                                                        CUBLAS_OP_T,
                                                        CUBLAS_OP_N,
                                                        KV_DIM,
                                                        prompt_lengths[prompt_id],
                                                        EMBEDDING_LENGTH,
                                                        &v_proj_alpha,
                                                        weights.w_v[layer],
                                                        CUDA_R_16BF,
                                                        EMBEDDING_LENGTH,
                                                        rms_norms,
                                                        CUDA_R_16BF,
                                                        EMBEDDING_LENGTH,
                                                        &v_proj_beta,
                                                        v_proj[prompt_id][layer],
                                                        CUDA_R_16BF,
                                                        KV_DIM,
                                                        CUBLAS_COMPUTE_32F,
                                                        CUBLAS_GEMM_DEFAULT);

            // RoPE now

            rope(q_proj, prompt_lengths[prompt_id], EMBEDDING_LENGTH);
            rope(k_proj[prompt_id][layer], prompt_lengths[prompt_id], KV_DIM);

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
                __nv_bfloat16 *k_head = k_proj[prompt_id][layer] + k_head_idx * HEAD_DIM;
                __nv_bfloat16 *attn_score_head = attn_scores + prompt_lengths[prompt_id] * prompt_lengths[prompt_id] * i;

                cublasStatus_t attn_score_status = cublasGemmEx(cublas_handle,
                                                                CUBLAS_OP_T,
                                                                CUBLAS_OP_N,
                                                                prompt_lengths[prompt_id],
                                                                prompt_lengths[prompt_id],
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
                                                                prompt_lengths[prompt_id],
                                                                CUBLAS_COMPUTE_32F,
                                                                CUBLAS_GEMM_DEFAULT);
            }

            causalMask(attn_scores, prompt_lengths[prompt_id]);

            softmax(attn_scores, prompt_lengths[prompt_id]);

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
                __nv_bfloat16 *attn_scores_head = attn_scores + i * prompt_lengths[prompt_id] * prompt_lengths[prompt_id];
                __nv_bfloat16 *v_head = v_proj[prompt_id][layer] + v_head_idx * HEAD_DIM;
                __nv_bfloat16 *output_attn_scores_head = attn_scores_v + i * HEAD_DIM;

                cublasStatus_t attn_score_status = cublasGemmEx(cublas_handle,
                                                                CUBLAS_OP_N,
                                                                CUBLAS_OP_N,
                                                                HEAD_DIM,
                                                                prompt_lengths[prompt_id],
                                                                prompt_lengths[prompt_id],
                                                                &attn_scores_v_alpha,
                                                                v_head,
                                                                CUDA_R_16BF,
                                                                KV_DIM,
                                                                attn_scores_head,
                                                                CUDA_R_16BF,
                                                                prompt_lengths[prompt_id],
                                                                &attn_scores_v_beta,
                                                                output_attn_scores_head,
                                                                CUDA_R_16BF,
                                                                EMBEDDING_LENGTH,
                                                                CUBLAS_COMPUTE_32F,
                                                                CUBLAS_GEMM_DEFAULT);
            }

            // output projection, it will be an input for MLP blocks
            // attn_scores_v * w_o^T
            // (num_tok, 2048) * (2048, 2048) -> (num_tok, 2048)
            // same as Q projection, so copy paste
            o_proj = buf_2048_2;
            cublasStatus_t o_proj_status = cublasGemmEx(cublas_handle,
                                                        CUBLAS_OP_T,
                                                        CUBLAS_OP_N,
                                                        EMBEDDING_LENGTH,
                                                        prompt_lengths[prompt_id],
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

            // (num_tok, 2048) + (num_tok, 2048) -> (num_tok, 2048)
            residualAdd(hidden_state, o_proj, prompt_lengths[prompt_id]);
            // post attention RMS Norm
            rmsNorm(hidden_state, rms_norms, weights.post_attn_layernorms[layer], prompt_lengths[prompt_id]);

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
                                                      prompt_lengths[prompt_id],
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
                                                    prompt_lengths[prompt_id],
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
            silu(gate, up, prompt_lengths[prompt_id]); // gate = after_silu now

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
                                                      prompt_lengths[prompt_id],
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
            residualAdd(hidden_state, down, prompt_lengths[prompt_id]);
        }
        rmsNorm(hidden_state, rms_norms, weights.norm, prompt_lengths[prompt_id]);

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

        cublasStatus_t embed_status = cublasGemmEx(cublas_handle,
                                                   CUBLAS_OP_T,
                                                   CUBLAS_OP_N,
                                                   VOCAB_SIZE,
                                                   prompt_lengths[prompt_id],
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

        cudaMemcpy(embed_proj_cpu.data(), embed_proj, sizeof(__nv_bfloat16) * prompt_lengths[prompt_id] * VOCAB_SIZE, cudaMemcpyDeviceToHost);
        // argmax to get the output token
        // TODO: write a proper kernel for it
        // for now just a simple CPU function
        int last_token_offset = (prompt_lengths[prompt_id] - 1) * VOCAB_SIZE;
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

        generated_tokens[prompt_id].push_back(max_token_idx);
        last_generated_tokens[prompt_id] = max_token_idx;
        current_prompt_len[prompt_id] = prompt_lengths[prompt_id];
    }

    cudaFree(attn_scores);
    cudaMalloc(&attn_scores, sizeof(__nv_bfloat16) * MAX_SEQ_LEN * NUM_Q_HEADS);

    // DECODE
    // since now I operate always on index 0 for all values and for current_position_token for new K and V
    int *gpu_last_tokens;
    cudaMalloc(&gpu_last_tokens, num_prompts * sizeof(int));
    // TODO: move argmax to GPU and get rid of these CPU<->GPU tokens moves

    // reused temporary buffer for batched K/V cache computation
    __nv_bfloat16 *kv_proj_batched_buffer;
    cudaMalloc(&kv_proj_batched_buffer, num_prompts * sizeof(__nv_bfloat16) * KV_DIM);

    while (!std::all_of(is_prompt_finished.begin(), is_prompt_finished.end(), [](bool prompt)
                        { return prompt; }))
    {
        // TODO: make it more suitable for single token operations, perhaps just pass a token id as param
        cudaMemcpy(gpu_last_tokens, last_generated_tokens.data(), num_prompts * sizeof(int), cudaMemcpyHostToDevice);
        embeddingGatherDecode(gpu_last_tokens, num_prompts, hidden_state, weights.embed_tokens);
        for (int layer = 0; layer < N_LAYERS; ++layer)
        {
            rmsNorm(hidden_state, rms_norms, weights.input_layernorm[layer], num_prompts);
            q_proj = buf_2048_1;
            // q proj (num_prompts, 2048)
            cublasGemmEx(cublas_handle,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         EMBEDDING_LENGTH, // m
                         num_prompts,      // n
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
            // K proj = rms_norms (num_prompt, 2048) * W_k (512, 2048)
            // W_k is actually stored as 512, 2048 (out features, in features)
            // so that's why we need to transpose it
            // all the data is stored in row major and cublas reads it as column major
            // so all the data appears as transposed
            // so data actually apppears as (2048, num_prompt) * (2048, 512)
            // the output of matmul will also be produced as transposed, so we can say that
            // in our mental model we talk about K_proj^T
            // and to get K_proj^T we can do transposition trick and write the cublas call as
            // W_k^T * rms_nroms
            // so we end up with: K_proj^T = W_k^T (512, 2048) * rms_norms (2048, num_prompt)
            // result dim is K_proj^T = (512, num_prompt)
            // but it's transposed, so in fact we get correct output dimension (num_prompt, 512)
            // it was great for num_prompt=1, but the problem is that prompts have different length
            // that's why we have vector of current_prompt_len, but also we can't write to K_proj
            // directly, so I write to temp buffer kv_proj_batched_buffer and the scatter
            // output to K_proj in a loop
            cublasGemmEx(cublas_handle,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         KV_DIM,           // m = 512
                         num_prompts,      // n = num prompts
                         EMBEDDING_LENGTH, // k = 2048
                         &k_proj_alpha,
                         weights.w_k[layer], // A
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH, // lda 2048, because W_k is in memory as 512, 2048
                         // so the gap between subsequent elements is 2048
                         rms_norms, // B
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH, // ldb, same reason for rms_norms
                         &k_proj_beta,
                         kv_proj_batched_buffer, // TODO C
                         CUDA_R_16BF,
                         KV_DIM, // ldc = 512
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT);

            for (int row = 0; row < num_prompts; ++row)
            {
                cudaMemcpy(k_proj[row][layer] + current_prompt_len[row] * KV_DIM, kv_proj_batched_buffer + row * KV_DIM, sizeof(__nv_bfloat16) * KV_DIM, cudaMemcpyDeviceToDevice);
            }

            // same
            cublasGemmEx(cublas_handle,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         KV_DIM,
                         num_prompts,
                         EMBEDDING_LENGTH,
                         &v_proj_alpha,
                         weights.w_v[layer],
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH,
                         rms_norms,
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH,
                         &v_proj_beta,
                         kv_proj_batched_buffer,
                         CUDA_R_16BF,
                         KV_DIM,
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT);

            for (int row = 0; row < num_prompts; ++row)
            {
                cudaMemcpy(v_proj[row][layer] + current_prompt_len[row] * KV_DIM, kv_proj_batched_buffer + row * KV_DIM, sizeof(__nv_bfloat16) * KV_DIM, cudaMemcpyDeviceToDevice);
            }

            for (int row = 0; row < num_prompts; ++row)
            {
                ropeDecode(&q_proj[row * EMBEDDING_LENGTH], current_prompt_len[row], EMBEDDING_LENGTH);
                ropeDecode(k_proj[row][layer] + current_prompt_len[row] * KV_DIM, current_prompt_len[row], KV_DIM);
            }

            for (int row = 0; row < num_prompts; ++row)
            {
                int seq_len = current_prompt_len[row] + 1;
                for (int i = 0; i < NUM_Q_HEADS; ++i)
                {
                    int k_head_idx = i / GQA_Q_TO_K_RATIO;
                    __nv_bfloat16 *q_head = q_proj + row * EMBEDDING_LENGTH + i * HEAD_DIM;
                    __nv_bfloat16 *k_head = k_proj[row][layer] + k_head_idx * HEAD_DIM;
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

                softmaxDecode(attn_scores, seq_len);

                attn_scores_v = buf_2048_1;
                for (int i = 0; i < NUM_Q_HEADS; ++i)
                {
                    int v_head_idx = i / GQA_ATTN_SCORES_TO_V_RATIO;
                    __nv_bfloat16 *attn_scores_head = attn_scores + i * MAX_SEQ_LEN;
                    __nv_bfloat16 *v_head = v_proj[row][layer] + v_head_idx * HEAD_DIM;
                    __nv_bfloat16 *output_attn_scores_head = attn_scores_v + row * EMBEDDING_LENGTH + i * HEAD_DIM;

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
            }

            o_proj = buf_2048_2;
            // (1, 2048) * (2048, 2048) -> (1, 2048)
            cublasGemmEx(cublas_handle,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         EMBEDDING_LENGTH, // m
                         num_prompts,      // n
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

            residualAdd(hidden_state, o_proj, num_prompts);

            rmsNorm(hidden_state, rms_norms, weights.post_attn_layernorms[layer], num_prompts);

            // MLP
            cublasGemmEx(cublas_handle,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         HIDDEN_DIM,       // m
                         num_prompts,      // n
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
                         num_prompts,      // n
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

            silu(gate, up, num_prompts);

            down = buf_2048_2;
            cublasGemmEx(cublas_handle,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         EMBEDDING_LENGTH, // m
                         num_prompts,      // n
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

            residualAdd(hidden_state, down, num_prompts);
        }

        rmsNorm(hidden_state, rms_norms, weights.norm, num_prompts);

        cublasGemmEx(cublas_handle,
                     CUBLAS_OP_T,
                     CUBLAS_OP_N,
                     VOCAB_SIZE,       // m
                     num_prompts,      // n
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

        cudaMemcpy(embed_proj_cpu.data(), embed_proj, sizeof(__nv_bfloat16) * num_prompts * VOCAB_SIZE, cudaMemcpyDeviceToHost);

        float max_token = 0.0;
        int max_token_idx = 0;
        for (int row = 0; row < num_prompts; ++row)
        {
            if (is_prompt_finished[row])
            {
                // we discard the batch item here, so much wasted computation!
                // but now this is how I really can feel the way why continouous batching was invented <3 to mitigate exactly this problem
                continue;
            }
            max_token = (float)embed_proj_cpu[row * VOCAB_SIZE]; // TODO: verify if float is good enough in place of nvbf16
            max_token_idx = 0;
            for (int token_idx = 0; token_idx < VOCAB_SIZE; ++token_idx)
            {
                if ((float)embed_proj_cpu[row * VOCAB_SIZE + token_idx] > max_token)
                {
                    max_token = embed_proj_cpu[row * VOCAB_SIZE + token_idx];
                    max_token_idx = token_idx;
                }
            }
            // TODO: wrap with #ifdef DEBUG
            std::cout << "Output token: " << (float)max_token << ", token index: " << std::to_string(max_token_idx) << std::endl;
            if (max_token_idx == END_OF_TEXT_TOKEN_ID || max_token_idx == EOT_ID_TOKEN_ID || current_prompt_len[row] == MAX_SEQ_LEN - 1)
            {
                is_prompt_finished[row] = true;
            }
            else
            {
                last_generated_tokens[row] = max_token_idx;
                generated_tokens[row].push_back(max_token_idx);
                current_prompt_len[row] = current_prompt_len[row] + 1;
            }
        }
    }
    std::cout << "\nOk bye!\n";
    cublasDestroy(cublas_handle);
    cudaDeviceSynchronize();
    return 0;
}

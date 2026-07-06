#include "inference.hpp"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <iostream>
#include <stdexcept>
#include "config.hpp"
#include <cublas_v2.h>
#include "cuda_utils.hpp"
#include <numeric>
#include "kernels.cuh"

constexpr size_t BF16 = sizeof(nv_bfloat16);
constexpr size_t INPUT_TOKENS_BYTES = MAX_PROMPT_LEN * sizeof(int);
constexpr size_t INPUT_EMBEDDINGS_BYTES = MAX_PROMPT_LEN * EMBEDDING_LENGTH * BF16;
constexpr size_t HIDDEN_STATE_BYTES = MAX_BUFFER_SIZE * EMBEDDING_LENGTH * BF16;
constexpr size_t RMS_NORMS_BYTES = MAX_BUFFER_SIZE * EMBEDDING_LENGTH * BF16;
constexpr size_t BUF_2048_BYTES = MAX_BUFFER_SIZE * EMBEDDING_LENGTH * BF16;
constexpr size_t K_PROJ_BYTES = MAX_PROMPT_LEN * KV_DIM * BF16;
constexpr size_t V_PROJ_BYTES = MAX_PROMPT_LEN * KV_DIM * BF16;
constexpr size_t PREFILL_ATTN_SCORES_BYTES = MAX_PROMPT_LEN * MAX_PROMPT_LEN * NUM_Q_HEADS * BF16;
constexpr size_t GATE_BYTES = MAX_BUFFER_SIZE * HIDDEN_DIM * BF16;
constexpr size_t UP_BYTES = MAX_BUFFER_SIZE * HIDDEN_DIM * BF16;
constexpr size_t EMBED_PROJ_BYTES = MAX_BUFFER_SIZE * VOCAB_SIZE * BF16;
constexpr size_t LAST_TOKENS_BYTES = BATCH_SIZE * sizeof(int);
constexpr size_t ACTIVE_SLOTS_BYTES = BATCH_SIZE * sizeof(int);
constexpr size_t SEQ_LENS_BYTES = BATCH_SIZE * sizeof(int);
constexpr size_t K_PROJ_BATCHED_BYTES = BATCH_SIZE * KV_DIM * BF16;
constexpr size_t V_PROJ_BATCHED_BYTES = BATCH_SIZE * KV_DIM * BF16;
// Element count kept separately: embed_proj_cpu is sized in elements, not bytes.
constexpr size_t EMBED_PROJ_LEN = MAX_BUFFER_SIZE * VOCAB_SIZE;

InferenceContext::InferenceContext()
{
    // Create the cuBLAS context. cublas_handle is required by every cublasGemmEx
    // call; cublasCreate allocates GPU-side resources for it. If this fails
    // no matrix multiply can run, so abort construction by throwing.
    cublasStatus_t status = cublasCreate(&cublas_handle);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "cuBLAS init failed, status: " << status << "\n";
        throw std::runtime_error("cuBLAS init failed");
    }

    logits_cpu.resize(EMBED_PROJ_LEN);

    CUDA_CHECK(cudaMalloc(&input_tokens, INPUT_TOKENS_BYTES));
    CUDA_CHECK(cudaMalloc(&input_embeddings, INPUT_EMBEDDINGS_BYTES));
    CUDA_CHECK(cudaMalloc(&hidden_state, HIDDEN_STATE_BYTES));
    CUDA_CHECK(cudaMalloc(&normed_hidden, RMS_NORMS_BYTES));
    CUDA_CHECK(cudaMalloc(&scratch_a, BUF_2048_BYTES));
    CUDA_CHECK(cudaMalloc(&scratch_b, BUF_2048_BYTES));
    CUDA_CHECK(cudaMalloc(&key_states, std::max(K_PROJ_BYTES, K_PROJ_BATCHED_BYTES)));
    CUDA_CHECK(cudaMalloc(&value_states, std::max(V_PROJ_BYTES, V_PROJ_BATCHED_BYTES)));
    CUDA_CHECK(cudaMalloc(&attn_weights, PREFILL_ATTN_SCORES_BYTES));
    CUDA_CHECK(cudaMalloc(&gate, GATE_BYTES));
    CUDA_CHECK(cudaMalloc(&up, UP_BYTES));
    CUDA_CHECK(cudaMalloc(&logits, EMBED_PROJ_BYTES));
    CUDA_CHECK(cudaMalloc(&last_tokens, LAST_TOKENS_BYTES));
    CUDA_CHECK(cudaMalloc(&active_slots, ACTIVE_SLOTS_BYTES));
    CUDA_CHECK(cudaMalloc(&seq_lens, SEQ_LENS_BYTES));
}

InferenceContext::~InferenceContext()
{
    cublasDestroy(cublas_handle);
    CUDA_CHECK_NOTHROW(cudaFree(input_tokens));
    CUDA_CHECK_NOTHROW(cudaFree(input_embeddings));
    CUDA_CHECK_NOTHROW(cudaFree(hidden_state));
    CUDA_CHECK_NOTHROW(cudaFree(normed_hidden));
    CUDA_CHECK_NOTHROW(cudaFree(scratch_a));
    CUDA_CHECK_NOTHROW(cudaFree(scratch_b));
    CUDA_CHECK_NOTHROW(cudaFree(key_states));
    CUDA_CHECK_NOTHROW(cudaFree(value_states));
    CUDA_CHECK_NOTHROW(cudaFree(attn_weights));
    CUDA_CHECK_NOTHROW(cudaFree(gate));
    CUDA_CHECK_NOTHROW(cudaFree(up));
    CUDA_CHECK_NOTHROW(cudaFree(logits));
    CUDA_CHECK_NOTHROW(cudaFree(last_tokens));
    CUDA_CHECK_NOTHROW(cudaFree(active_slots));
    CUDA_CHECK_NOTHROW(cudaFree(seq_lens));
}

void InferenceContext::copyInputTokensToGPU()
{
    CUDA_CHECK(cudaMemcpy(input_tokens, input_tokens_cpu.data(), input_token_len * sizeof(int),
                          cudaMemcpyHostToDevice));
}

void InferenceContext::InitHiddenState()
{
    // Seed the residual stream with the input embeddings.
    // hidden_state is the residual stream carried through every Transformer layer: each
    // sublayer reads a normalized copy of it and adds its output back in (residual connection).
    CUDA_CHECK(cudaMemcpy(hidden_state, input_embeddings,
                          input_token_len * EMBEDDING_LENGTH * sizeof(nv_bfloat16),
                          cudaMemcpyDeviceToDevice));
}

/**
 * @brief output = input * weight^T
 *
 * 行主序矩阵乘法，使用 cuBLAS GEMM 执行，但 cuBLAS 默认使用列主序存储，因此需要对矩阵进行转置处理。
 * 利用恒等式 (A * B)^T = B^T * A^T，对 output = inputs * weight^T 两边转置：
 *
 *     output^T = (weight^T)^T * inputs^T = weight * inputs^T
 *
 * 行主序的 output 在内存里就等于列主序的 output^T，所以让列主序的 cuBLAS 直接算出 output^T
 * 即可，无需再转回去。 传给 cuBLAS 的 weight 需要转置，否则在列主序视角下 weight 就变成了
 * weight^T。
 */
cublasStatus_t InferenceContext::linear(const nv_bfloat16* input,
                                        const nv_bfloat16* weight,
                                        nv_bfloat16* output,
                                        int input_rows,
                                        int input_cols,
                                        int weight_rows,
                                        float alpha,
                                        float beta,
                                        int leading_dim_input,
                                        int leading_dim_weight,
                                        int leading_dim_output)
{

    // cublasGemmEx 函数签名及参数说明：
    //
    //   计算 C = alpha * op(A) * op(B) + beta * C
    //   其中 op(X) 为 X、X^T 或 X^H（共轭转置），由 transa/transb 控制
    //
    // cublasStatus_t cublasGemmEx(
    //   cublasHandle_t handle,           // cuBLAS 上下文句柄，管理 GPU 上的 BLAS 资源和状态
    //   cublasOperation_t transa,        // 矩阵 A 的运算方式：
    //                                    //
    //                                    CUBLAS_OP_N（不转置）、CUBLAS_OP_T（转置）、CUBLAS_OP_C（共轭转置）
    //   cublasOperation_t transb,        // 矩阵 B 的运算方式，同上
    //   int m,                           // 结果矩阵 C 的行数（也是 op(A) 的行数）
    //   int n,                           // 结果矩阵 C 的列数（也是 op(B) 的列数）
    //   int k,                           // 内积维度（op(A) 的列数 = op(B) 的行数）
    //   const void* alpha,               // 标量系数 alpha，C = alpha * op(A) * op(B) + beta * C
    //   const void* A,                   // 矩阵 A 的 GPU 指针（列主序存储）
    //   cudaDataType Atype,              // 矩阵 A 的数据类型（如 CUDA_R_16BF 表示 bf16，CUDA_R_32F
    //                                    // 表示 fp32）
    //   int lda,                         // 矩阵 A 的 leading
    //   dimension（列主序下为行数，即相邻列元素间距） const void* B,                   // 矩阵 B 的
    //   GPU 指针（列主序存储） cudaDataType Btype,              // 矩阵 B 的数据类型 int ldb, //
    //   矩阵 B 的 leading dimension const void* beta,                // 标量系数 beta，用于累加：+
    //   beta * C void* C,                         // 结果矩阵 C 的 GPU
    //   指针（列主序存储），同时作为输入（beta*C）和输出 cudaDataType Ctype,              // 矩阵 C
    //   的数据类型 int ldc,                         // 矩阵 C 的 leading dimension
    //   cublasComputeType_t computeType, // 内部累加的计算精度（如 CUBLAS_COMPUTE_32F 表示 fp32
    //   累加，
    //                                    // CUBLAS_COMPUTE_32F_FAST_TF32 表示使用 TF32 tensor core
    //                                    //  加速）
    //   cublasGemmAlgo_t algo            // GEMM 算法选择（CUBLAS_GEMM_DEFAULT
    //   由库自动选择最优算法）
    // );
    //
    // Leading Dimension:
    //
    // 永远指向存储的连续方向的长度:列主序数连续列的高度(行数),行主序数连续行的宽度(列数)。
    // 它衡量的是"跨过一整条连续段"要走多远。
    // 由于把行主序的矩阵传给列主序的 cuBLAS 相当于做了一次转置，所以 leading dimension 需要传入
    // 行主序矩阵的列数（即转置后的列主序矩阵的行数）。

    return cublasGemmEx(cublas_handle,

                        CUBLAS_OP_T, // 传入行主序的 weight，列主序视角下就是 weight^T，所以要转置
                        CUBLAS_OP_N, // 传入行主序的 input，列主序视角下就是 input^T，所以不转置

                        weight_rows, input_rows, // m=output^T 的行数, n=output^T 的列数
                        input_cols,              // k=内积维度：weight 的列数 = inputs^T 的行数

                        &alpha,

                        weight,                                 // 矩阵 A（weight^T）
                        CUDA_R_16BF,                            // 矩阵 A 的数据类型
                        leading_dim_weight ? leading_dim_weight // 矩阵 A 的 leading dimension
                                           : input_cols, // （列主序下为行数，即 weight^T 的行数）

                        input,                                              // 矩阵 B（inputs^T）
                        CUDA_R_16BF,                                        // 矩阵 B 的数据类型
                        leading_dim_input ? leading_dim_input : input_cols, // inputs^T 的行数

                        &beta,

                        output,                                                // 矩阵 C (output^T)
                        CUDA_R_16BF,                                           // 矩阵 C 的数据类型
                        leading_dim_output ? leading_dim_output : weight_rows, // output^T 的行数

                        CUBLAS_COMPUTE_32F, // 内部累加的计算精度
                        CUBLAS_GEMM_DEFAULT // 由库自动选择最优算法
    );
}

/**
 * @brief c = a * b
 */
cublasStatus_t InferenceContext::matmul(const nv_bfloat16* a,
                                        const nv_bfloat16* b,
                                        nv_bfloat16* c,
                                        int a_rows,
                                        int a_cols,
                                        int b_cols,
                                        float alpha,
                                        float beta,
                                        int leading_dim_a,
                                        int leading_dim_b,
                                        int leading_dim_c)
{
    return cublasGemmEx(cublas_handle,

                        CUBLAS_OP_N, CUBLAS_OP_N,

                        b_cols, a_rows, a_cols,

                        &alpha,

                        b, CUDA_R_16BF, leading_dim_b ? leading_dim_b : b_cols,

                        a, CUDA_R_16BF, leading_dim_a ? leading_dim_a : a_cols,

                        &beta,

                        c, CUDA_R_16BF, leading_dim_c ? leading_dim_c : b_cols,

                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

void InferenceContext::qProj(const nv_bfloat16* q_proj)
{
    query_states = scratch_a;
    CUBLAS_CHECK(linear(normed_hidden, q_proj, query_states, num_rows, EMBEDDING_LENGTH,
                        EMBEDDING_LENGTH));
}

void InferenceContext::kProj(const nv_bfloat16* k_proj)
{
    CUBLAS_CHECK(
        linear(normed_hidden, k_proj, key_states, num_rows, EMBEDDING_LENGTH, KV_DIM));
}

void InferenceContext::vProj(const nv_bfloat16* v_proj)
{
    CUBLAS_CHECK(
        linear(normed_hidden, v_proj, value_states, num_rows, EMBEDDING_LENGTH, KV_DIM));
}

/**
 * @brief 注意力分数：逐 Q 头计算 attn_score = Q_head · K_head^T / sqrt(HEAD_DIM)
 *   Q 形状 (num_tok, EMBEDDING_LENGTH=2048) = NUM_Q_HEADS(32) 个 (num_tok, HEAD_DIM=64) 头
 *   K 形状 (num_tok, KV_DIM=512)            = NUM_K_HEADS(8)  个 (num_tok, HEAD_DIM=64) 头
 *   GQA：每 GQA_Q_TO_K_RATIO(4) 个连续 Q 头共享 1 个 K 头（k_head_idx = i / 4）
 *   单头 Q_head · K_head^T → (num_tok, num_tok)，缩放系数 1/sqrt(64) 即 ATTN_SCALE
 *   合计 NUM_Q_HEADS 个头，总输出 (32, num_tok, num_tok)
 * @param q_head_idx
 */
void InferenceContext::updateAttentionScoresHead(int q_head_idx)
{
    nv_bfloat16* q_head = qHead(q_head_idx);
    nv_bfloat16* k_head = kHead(q_head_idx / GQA_Q_TO_K_RATIO);
    nv_bfloat16* attn_score_head = attentionHead(q_head_idx);

    CUBLAS_CHECK(linear(q_head, k_head, attn_score_head, input_token_len, HEAD_DIM, input_token_len,
                        ATTN_SCALE, 0, EMBEDDING_LENGTH, KV_DIM));
}

/**
 * @brief 注意力加权求和：逐 Q 头计算 output_head = scores_head · V_head
 *   权重 attn_weights 形状 (NUM_Q_HEADS=32, num_tok, num_tok)，单头 (num_tok, num_tok)（已
 * softmax） V 形状 (num_tok, KV_DIM=512) = NUM_V_HEADS(8) 个 (num_tok, HEAD_DIM=64) 头 GQA：每
 * GQA_ATTN_SCORES_TO_V_RATIO(4) 个 Q 头共享 1 个 V 头（v_head_idx = i / 4） 单头 (num_tok, num_tok)
 * · (num_tok, HEAD_DIM=64) → (num_tok, HEAD_DIM=64)，对 key 维求和 合计 NUM_Q_HEADS 个头拼接 →
 * (num_tok, HEAD_DIM * NUM_Q_HEADS = 2048)
 * @param q_head_idx
 */
void InferenceContext::updateAttentionOutputHead(int q_head_idx)
{
    context = scratch_a;
    nv_bfloat16* attn_weights_head = attentionHead(q_head_idx);
    nv_bfloat16* v_head = vHead(q_head_idx / GQA_ATTN_SCORES_TO_V_RATIO);
    nv_bfloat16* attn_output_head = context + q_head_idx * HEAD_DIM;

    CUBLAS_CHECK(matmul(attn_weights_head, v_head, attn_output_head, input_token_len,
                        input_token_len, HEAD_DIM, 1, 0, input_token_len, KV_DIM,
                        HEAD_DIM * NUM_Q_HEADS));
}

/**
 * @brief 输出投影（Output Projection）：将多头注意力拼接后的输出经 W_O 投影回隐藏维度，
 *   结果加回残差流，并作为后续 MLP 块的输入。
 *
 *   attn_output = context · o_proj^T
 *   (num_tok, EMBEDDING_LENGTH=2048) · (EMBEDDING_LENGTH, EMBEDDING_LENGTH) -> (num_tok,
 * EMBEDDING_LENGTH)
 *
 *   结构与 Q 投影相同：行主序数据借转置技巧交给列主序 cuBLAS，故 o_proj 传 CUBLAS_OP_T。
 */
void InferenceContext::oProj(const nv_bfloat16* o_proj)
{
    attn_output = scratch_b;
    CUBLAS_CHECK(
        linear(context, o_proj, attn_output, num_rows, EMBEDDING_LENGTH, EMBEDDING_LENGTH));
}

/**
 * @brief SwiGLU 前馈网络（FFN）：对 RMSNorm 后的隐藏状态做门控前馈变换。
 *   结果写入成员 down（残差相加由调用方完成）。
 *
 *   输入 normed_hidden 形状为 (num_tok, EMBEDDING_LENGTH)。计算流程：
 *     1. gate = normed_hidden · gate_proj^T    投影升维到 FFN 中间维度
 *     2. up   = normed_hidden · up_proj^T      同样升维，作为门控的另一支
 *     3. gate = SiLU(gate) ⊙ up                逐元素门控，SiLU(x)=x·σ(x)，原地写回 gate
 *     4. down = gate · down_proj^T             投影降回隐藏维度
 *
 * @param gate_proj 门控投影权重，形状 (HIDDEN_DIM, EMBEDDING_LENGTH)
 * @param up_proj   上投影权重，形状 (HIDDEN_DIM, EMBEDDING_LENGTH)
 * @param down_proj 下投影权重，形状 (EMBEDDING_LENGTH, HIDDEN_DIM)
 */
void InferenceContext::swiGLU(const nv_bfloat16* gate_proj,
                              const nv_bfloat16* up_proj,
                              const nv_bfloat16* down_proj)
{
    CUBLAS_CHECK(linear(normed_hidden, gate_proj, gate, num_rows, EMBEDDING_LENGTH, HIDDEN_DIM));
    CUBLAS_CHECK(linear(normed_hidden, up_proj, up, num_rows, EMBEDDING_LENGTH, HIDDEN_DIM));

    silu(gate, up, num_rows);

    down = scratch_b;
    CUBLAS_CHECK(linear(gate, down_proj, down, num_rows, HIDDEN_DIM, EMBEDDING_LENGTH));
}

/**
 * @brief LM head：把最后一层 RMSNorm 后的隐藏状态投影到词表维度。
 *   Llama 3.2 权重共享（tied embeddings），直接复用 embed_tokens 当 LM head。
 *
 *   logits = normed_hidden · embed_tokens^T
 *
 * @param embed_tokens 词嵌入矩阵，形状 (VOCAB_SIZE, EMBEDDING_LENGTH)；因权重共享兼作 LM head
 */
void InferenceContext::lmHead(const nv_bfloat16* embed_tokens)
{
    CUBLAS_CHECK(
        linear(normed_hidden, embed_tokens, logits, num_rows, EMBEDDING_LENGTH, VOCAB_SIZE));
}

PagedKVCache::PagedKVCache()
    : free_blocks(NUM_BLOCKS), block_table(MAX_SEQUENCES * N_LAYERS * MAX_BLOCKS_PER_SEQ, -1)
{
    CUDA_CHECK(cudaMalloc(&kv_cache, KV_CACHE_SIZE_BYTES));
    CUDA_CHECK(
        cudaMalloc(&block_table_gpu, MAX_SEQUENCES * N_LAYERS * MAX_BLOCKS_PER_SEQ * sizeof(int)));

    // 一开始所有 block 都空闲，free_blocks 初始化为 [0, 1, 2, ..., NUM_BLOCKS-1]
    std::iota(free_blocks.begin(), free_blocks.end(), 0);
}

PagedKVCache::~PagedKVCache()
{
    CUDA_CHECK_NOTHROW(cudaFree(kv_cache));
    CUDA_CHECK_NOTHROW(cudaFree(block_table_gpu));
}

int PagedKVCache::getFreePhysicalBlock()
{
    int physical_block_idx = free_blocks.back();
    free_blocks.pop_back();
    return physical_block_idx;
}

int PagedKVCache::getPhysicalBlock(int slot, int layer, int logical_block)
{
    return block_table[slot * N_LAYERS * MAX_BLOCKS_PER_SEQ + layer * MAX_BLOCKS_PER_SEQ +
                       logical_block];
}

void PagedKVCache::setPhysicalBlock(int slot, int layer, int logical_block, int physical_block)
{
    block_table[slot * N_LAYERS * MAX_BLOCKS_PER_SEQ + layer * MAX_BLOCKS_PER_SEQ + logical_block] =
        physical_block;
}

/**
 * @brief 把一批连续 token 的 K、V 写入 paged KV cache 的某个物理块。
 *   从源 buffer 第 src_token_offset 行起取 token_len 个 token 的 K/V，写到 physical_block
 *   块内、从第 token_in_block 个位置开始。块内 K 在前、V 紧随其后（相对块首偏移 V_OFFSET）。
 *
 * @param physical_block   目标物理块号（kv_cache 中的块索引）
 * @param k                源 K buffer（行主序，每行一个 token 的 K，宽 KV_DIM）
 * @param v                源 V buffer（布局同 k）
 * @param src_token_offset 源 buffer 起始行：从第几个 token 开始拷
 * @param token_len        拷贝的 token 数（连续若干行）
 * @param token_in_block   写入目标块内的起始位置（默认 0 = 块首）
 */
void PagedKVCache::cacheKV(int physical_block, nv_bfloat16* k, nv_bfloat16* v, int src_token_offset,
                           int token_len, int token_in_block)
{
    // 目标块内起始字节偏移：块首 + 块内 token 位置（prefill 传 token_in_block=0，即块首）
    size_t block_offset = (size_t) physical_block * BLOCK_BYTES +
                          (size_t) token_in_block * KV_DIM * sizeof(nv_bfloat16);

    // store K
    nv_bfloat16* k_cache_ptr = (nv_bfloat16*) ((char*) kv_cache + block_offset);
    nv_bfloat16* k_proj_ptr = k + src_token_offset * KV_DIM;
    CUDA_CHECK(cudaMemcpy(k_cache_ptr, k_proj_ptr, token_len * KV_DIM * sizeof(nv_bfloat16),
                          cudaMemcpyDeviceToDevice));

    // store V
    nv_bfloat16* v_cache_ptr = (nv_bfloat16*) ((char*) kv_cache + block_offset + V_OFFSET);
    nv_bfloat16* v_proj_ptr = v + src_token_offset * KV_DIM;
    CUDA_CHECK(cudaMemcpy(v_cache_ptr, v_proj_ptr, token_len * KV_DIM * sizeof(nv_bfloat16),
                          cudaMemcpyDeviceToDevice));
}

BatchState::BatchState()
    : is_slot_free(BATCH_SIZE, true), generated_tokens(BATCH_SIZE),
      last_generated_tokens(BATCH_SIZE), current_prompt_len(BATCH_SIZE, 0),
      request_id(BATCH_SIZE, -1)
{
}

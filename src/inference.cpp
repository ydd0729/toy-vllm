#include "inference.hpp"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include "config.hpp"
#include <cublas_v2.h>
#include "cuda_utils.hpp"
#include <numeric>
#include <limits>
#include "kernels.cuh"
#include "utils.hpp"

using tokenizers::Tokenizer;

constexpr size_t BF16 = sizeof(nv_bfloat16);
constexpr size_t INPUT_TOKENS_BYTES = MAX_PROMPT_LEN * sizeof(int);
constexpr size_t INPUT_EMBEDDINGS_BYTES = MAX_PROMPT_LEN * EMBEDDING_LENGTH * BF16;
constexpr size_t BUFFER_EMBEDDING_PRODUCT_BYTES = MAX_BUFFER_SIZE * EMBEDDING_LENGTH * BF16;
constexpr size_t KV_PROJ_BYTES = MAX_PROMPT_LEN * KV_DIM * BF16;
constexpr size_t PREFILL_ATTN_SCORES_BYTES = MAX_PROMPT_LEN * MAX_PROMPT_LEN * NUM_Q_HEADS * BF16;
constexpr size_t GATE_BYTES = MAX_BUFFER_SIZE * HIDDEN_DIM * BF16;
constexpr size_t UP_BYTES = MAX_BUFFER_SIZE * HIDDEN_DIM * BF16;
constexpr size_t EMBED_PROJ_BYTES = MAX_BUFFER_SIZE * VOCAB_SIZE * BF16;
constexpr size_t BATCH_INT_BYTES = BATCH_SIZE * sizeof(int);
constexpr size_t KV_PROJ_BATCHED_BYTES = BATCH_SIZE * KV_DIM * BF16;
// Element count kept separately: embed_proj_cpu is sized in elements, not bytes.
constexpr size_t EMBED_PROJ_LEN = MAX_BUFFER_SIZE * VOCAB_SIZE;

std::unique_ptr<Tokenizer> load_tokenizer()
{
    std::ifstream tokenizer_file(llama3p2_1B_Instruct_tokenizer_path, std::ios_base::binary);

    if (!tokenizer_file.is_open())
    {
        throw std::runtime_error(
            std::format("Can't open {}", llama3p2_1B_Instruct_tokenizer_path.string()));
    }

    std::streamsize size = std::filesystem::file_size(llama3p2_1B_Instruct_tokenizer_path);

    std::string blob;
    blob.resize(size);
    tokenizer_file.read(blob.data(), size);

    return Tokenizer::FromBlobJSON(blob);
}

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
    CUDA_CHECK(cudaMalloc(&hidden_state, BUFFER_EMBEDDING_PRODUCT_BYTES));
    CUDA_CHECK(cudaMalloc(&normed_hidden, BUFFER_EMBEDDING_PRODUCT_BYTES));
    CUDA_CHECK(cudaMalloc(&scratch_a, BUFFER_EMBEDDING_PRODUCT_BYTES));
    CUDA_CHECK(cudaMalloc(&scratch_b, BUFFER_EMBEDDING_PRODUCT_BYTES));
    CUDA_CHECK(cudaMalloc(&key_states, std::max(KV_PROJ_BYTES, KV_PROJ_BATCHED_BYTES)));
    CUDA_CHECK(cudaMalloc(&value_states, std::max(KV_PROJ_BYTES, KV_PROJ_BATCHED_BYTES)));
    CUDA_CHECK(cudaMalloc(&attn_weights, PREFILL_ATTN_SCORES_BYTES));
    CUDA_CHECK(cudaMalloc(&gate, GATE_BYTES));
    CUDA_CHECK(cudaMalloc(&up, UP_BYTES));
    CUDA_CHECK(cudaMalloc(&logits, EMBED_PROJ_BYTES));
    CUDA_CHECK(cudaMalloc(&last_tokens, BATCH_INT_BYTES));
    CUDA_CHECK(cudaMalloc(&active_slots, BATCH_INT_BYTES));
    CUDA_CHECK(cudaMalloc(&seq_lens, BATCH_INT_BYTES));

    tok = load_tokenizer();
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

std::string InferenceContext::apply_chat_template(const std::string& user_msg)
{
    return "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n" + user_msg +
           "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
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
    CUBLAS_CHECK(
        linear(normed_hidden, q_proj, query_states, num_rows, EMBEDDING_LENGTH, EMBEDDING_LENGTH));
}

void InferenceContext::kProj(const nv_bfloat16* k_proj)
{
    CUBLAS_CHECK(linear(normed_hidden, k_proj, key_states, num_rows, EMBEDDING_LENGTH, KV_DIM));
}

void InferenceContext::vProj(const nv_bfloat16* v_proj)
{
    CUBLAS_CHECK(linear(normed_hidden, v_proj, value_states, num_rows, EMBEDDING_LENGTH, KV_DIM));
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
void PagedKVCache::cacheKV(int physical_block,
                           nv_bfloat16* k,
                           nv_bfloat16* v,
                           int src_token_offset,
                           int token_len,
                           int token_in_block)
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
      last_generated_tokens(BATCH_SIZE), current_prompt_len(BATCH_SIZE, 0), requests(BATCH_SIZE)

{
}



/**
 * @brief Prefill 阶段：一次性处理整个 prompt，填充 KV Cache 并生成第一个 token
 *
 * LLM 推理分为两个阶段：
 *   1. Prefill（本函数）：并行处理 N 个输入 token，填充 KV Cache，输出第一个生成 token
 *   2. Decode：逐 token 生成，每次只处理 1 个新 token
 *
 * 流程：
 *   embedding -> [RMSNorm -> QKV proj -> RoPE -> KV cache scatter -> Attention -> O proj ->
 *   residual -> RMSNorm -> MLP(gate/up/SiLU/down) -> residual] x N_LAYERS -> RMSNorm ->
 *   logits -> argmax -> 第一个生成 token
 *
 * @param prompt_queue  待处理的 prompt 队列，本函数从队首取出一个 prompt 并出队
 * @param slot          分配给该序列的 batch 席位编号，用于占用标记、页表寻址和输出回溯
 * @param ctx           推理上下文，持有 input_tokens / embeddings / hidden_state 等中间张量及各算子
 * @param pkv           Paged KV Cache，保存各层 K/V 的物理块及 slot/layer/逻辑块的页表映射
 * @param bs            batch 状态，跟踪各 slot 的占用情况与请求 id（is_slot_free / request_id）
 * @param w             模型权重，提供 embedding、各层投影/归一化及输出层的 GPU 指针
 */
void prefill(std::queue<Request>& request_queue,
             int slot,
             InferenceContext& ctx,
             PagedKVCache& pkv,
             BatchState& bs,
             Weights& w)
{
    Request r = std::move(request_queue.front());
    request_queue.pop();
    bs.requests[slot] = r;
    bs.is_slot_free[slot] = false;

    ctx.setInputTokens(bs.requests[slot].input_tokens);
    ctx.copyInputTokensToGPU();

    // 对 prompt 中的每个 token 获取 embedding ，并保存到 input_embeddings 中
    embeddingGather(ctx.input_tokens, ctx.input_embeddings, w.embed_tokens, ctx.input_token_len);

    ctx.InitHiddenState();

    for (int layer = 0; layer < N_LAYERS; ++layer)
    {
        // 每层开始前的 RMSNorm：对 hidden_state 做归一化，输出到 rms_norms，供后续 QKV 投影使用
        rmsNorm(ctx.hidden_state, ctx.normed_hidden, w.input_layernorm[layer], ctx.input_token_len);

        // Q 投影计算： Q = inputs * q_proj^T
        //   - q_proj：查询权重矩阵 (query weight)，模型参数之一，形状为 (EMBEDDING_LENGTH,
        //   EMBEDDING_LENGTH)
        //   - inputs：输入隐藏状态，形状为 (num_tok, EMBEDDING_LENGTH)
        //   - Q：查询向量 (query)，注意力机制的三大投影之一，形状为 (num_tok, EMBEDDING_LENGTH)
        ctx.qProj(w.q_proj[layer]);

        // K 投影计算：K = inputs * k_proj^T
        //   - k_proj：键权重矩阵 (key weight)，形状为 (KV_DIM, EMBEDDING_LENGTH)
        //         GQA/MQA 下 KV 头数少于 Q 头，故 KV_DIM < EMBEDDING_LENGTH
        //   - inputs：输入隐藏状态，形状为 (num_tok, EMBEDDING_LENGTH)
        //   - K：键向量，形状为 (num_tok, KV_DIM)
        ctx.kProj(w.k_proj[layer]);

        // V 投影计算：V = inputs * v_proj^T，与 K 投影类似
        ctx.vProj(w.v_proj[layer]);

        // RoPE

        ropeRotateHalf(ctx.query_states, ctx.input_token_len, EMBEDDING_LENGTH);
        ropeRotateHalf(ctx.key_states, ctx.input_token_len, KV_DIM);

        // PagedAttention：将本序列 prefill 得到的 K/V 按块散射进 kv_cache。
        // 每 BLOCK_SIZE 个 token 占一个逻辑块，共需 ceil(input_token_len / BLOCK_SIZE) 个块。
        // block_table 是 [slot][layer][logical_block] 三维展平的页表：
        //   slot          —— 该序列在 batch 中的席位编号
        //   layer         —— 当前 Transformer 层
        //   logical_block —— 序列内第几个逻辑块（= token_idx / BLOCK_SIZE）
        // 查/分配到物理块号后，把该块内的 K、V 写入 kv_cache 中对应地址。
        for (int token_idx = 0; token_idx < ctx.input_token_len; token_idx += BLOCK_SIZE)
        {
            int num_tokens_to_copy =
                std::min(static_cast<int>(ctx.input_token_len) - token_idx, BLOCK_SIZE);

            // 从页表读出该逻辑块对应的物理块号：
            //   若为 -1，说明尚未分配，从 free_blocks 取一个空闲物理块，并写回页表同一位置；
            //   prefill 阶段每个逻辑块都应是首次分配（-1），否则说明有残留映射，属于 bug。
            // 拿到物理块号后，据此算出它在 kv_cache 中的地址，把 K、V 写进去。
            int logical_block_idx = token_idx / BLOCK_SIZE;
            int physical_block_idx = pkv.getPhysicalBlock(slot, layer, logical_block_idx);
            if (physical_block_idx == -1)
            {
                // TODO: free blocks 可能耗尽，但现在的设置一定够用
                physical_block_idx = pkv.getFreePhysicalBlock();
                pkv.setPhysicalBlock(slot, layer, logical_block_idx, physical_block_idx);
            }
            else
            {
                UNREACHABLE("block must be -1 during prefill");
            }

            pkv.cacheKV(physical_block_idx, ctx.key_states, ctx.value_states, token_idx,
                        num_tokens_to_copy);
        }

        // TODO: 并行？
        for (int i = 0; i < NUM_Q_HEADS; ++i)
        {
            ctx.updateAttentionScoresHead(i);
        }

        causalMask(ctx.attn_weights, ctx.input_token_len);
        softmax(ctx.attn_weights, ctx.input_token_len);

        // TODO: 并行？
        for (int i = 0; i < NUM_Q_HEADS; ++i)
        {
            ctx.updateAttentionOutputHead(i);
        }

        ctx.oProj(w.o_proj[layer]);

        // 残差连接：hidden_state += attn_output
        // (num_tok, 2048) + (num_tok, 2048) -> (num_tok, 2048)
        residualAdd(ctx.hidden_state, ctx.attn_output, ctx.input_token_len);

        // post attention RMS Norm
        rmsNorm(ctx.hidden_state, ctx.normed_hidden, w.post_attn_layernorms[layer],
                ctx.input_token_len);

        ctx.swiGLU(w.gate_proj[layer], w.up_proj[layer], w.down_proj[layer]);

        // (num_tok, 2048) + (num_tok, 2048) -> (num_tok, 2048)
        residualAdd(ctx.hidden_state, ctx.down, ctx.input_token_len);
    }
    rmsNorm(ctx.hidden_state, ctx.normed_hidden, w.norm, ctx.input_token_len);

    ctx.lmHead(w.embed_tokens);

    // argmax to get the output token
    // TODO: move argmax to GPU and get rid of these CPU<->GPU tokens moves
    CUDA_CHECK(cudaMemcpy(ctx.logits_cpu.data(), ctx.logits,
                          sizeof(__nv_bfloat16) * ctx.input_token_len * VOCAB_SIZE,
                          cudaMemcpyDeviceToHost));
    int last_token_offset = (ctx.input_token_len - 1) * VOCAB_SIZE;
    float max_logit = (float) ctx.logits_cpu[last_token_offset];
    int max_token_idx = 0;
    for (int token_idx = 0; token_idx < VOCAB_SIZE; ++token_idx)
    {
        if ((float) ctx.logits_cpu[token_idx + last_token_offset] > max_logit)
        {
            max_logit = ctx.logits_cpu[token_idx + last_token_offset];
            max_token_idx = token_idx;
        }
    }
    // std::cout << "output token: " << max_token_idx << ", max logit: " << max_logit << std::endl;

    bs.generated_tokens[slot].push_back(max_token_idx);
    bs.last_generated_tokens[slot] = max_token_idx;
    bs.current_prompt_len[slot] = ctx.input_token_len;

    // synchronize state of block_table with block_table_gpu
    // TODO: do it more clever and not copy full table unnecessarily
    CUDA_CHECK(cudaMemcpy(pkv.block_table_gpu, pkv.block_table.data(),
                          MAX_SEQUENCES * N_LAYERS * MAX_BLOCKS_PER_SEQ * sizeof(int),
                          cudaMemcpyHostToDevice));
}

/**
 * @brief Decode 阶段：为当前所有活跃序列各生成一个新 token
 *
 * 与 prefill 并行处理整个 prompt 不同，decode 每步只处理每个活跃序列的「最后一个 token」，
 * 把 num_active_slots 个序列打成一个 batch（每序列 1 行）一起前向，逐步续写。
 *
 * 流程：
 *   把各 slot 的 last token / active_slot / seq_len 拷到 GPU
 *   -> embedding
 *   -> [RMSNorm -> QKV proj -> RoPE(按各自序列位置) -> 新 K/V 散射进 KV cache
 *       -> PagedAttention -> O proj -> residual
 *       -> RMSNorm -> MLP(gate/up/SiLU/down) -> residual] x N_LAYERS
 *   -> RMSNorm -> logits -> 逐 slot argmax
 *   -> 命中 EOS 或达到 MAX_SEQ_LEN：打印结果、释放该序列占用的物理块、置空 slot；
 *      否则：写回新 token 并令 current_prompt_len +1
 *
 * @param ctx  推理上下文，持有 hidden_state / QKV / logits 等中间张量及各算子；num_rows
 * 设为活跃序列数
 * @param pkv  Paged KV Cache，本步新 K/V 按各序列位置散射写入，序列结束时回收其物理块
 * @param bs   batch 状态，提供 active_slots/active_tokens 等输入，并被更新（生成
 * token、序列长度、slot 释放）
 * @param w    模型权重，提供 embedding、各层投影/归一化及输出层的 GPU 指针
 */
std::vector<RequestOutput> decode(std::queue<Request>& request_queue,
                                  InferenceContext& ctx,
                                  PagedKVCache& pkv,
                                  BatchState& bs,
                                  Weights& w)
{
    std::vector<RequestOutput> out;
    int num_active_slots = bs.active_slots.size();

    // copy useful data to gpu
    CUDA_CHECK(cudaMemcpy(ctx.last_tokens, bs.active_tokens.data(), num_active_slots * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx.active_slots, bs.active_slots.data(), num_active_slots * sizeof(int),
                          cudaMemcpyHostToDevice));

    std::vector<int> seq_lens(num_active_slots);

    for (int slot = 0; slot < num_active_slots; ++slot)
    {
        int active_slot = bs.active_slots[slot];
        seq_lens[slot] = bs.current_prompt_len[active_slot] + 1;
    }
    CUDA_CHECK(cudaMemcpy(ctx.seq_lens, seq_lens.data(), seq_lens.size() * sizeof(int),
                          cudaMemcpyHostToDevice));

    embeddingGatherDecode(ctx.last_tokens, num_active_slots, ctx.hidden_state, w.embed_tokens);

    // decode 每步处理 num_active_slots 行（每个活跃序列 1 个 token）
    ctx.num_rows = num_active_slots;

    for (int layer = 0; layer < N_LAYERS; ++layer)
    {
        rmsNorm(ctx.hidden_state, ctx.normed_hidden, w.input_layernorm[layer], num_active_slots);

        ctx.qProj(w.q_proj[layer]);
        ctx.kProj(w.k_proj[layer]);
        ctx.vProj(w.v_proj[layer]);

        for (int slot = 0; slot < num_active_slots; ++slot)
        {
            int active_slot = bs.active_slots[slot];
            ropeDecodeRotateHalf(&ctx.query_states[slot * EMBEDDING_LENGTH],
                                 bs.current_prompt_len[active_slot], EMBEDDING_LENGTH);
            ropeDecodeRotateHalf(ctx.key_states + slot * KV_DIM, bs.current_prompt_len[active_slot],
                                 KV_DIM);
        }

        // PagedAttention：把每个活跃序列这一步的新 K/V 散射进各自 KV cache 的下一个位置
        for (int slot = 0; slot < num_active_slots; ++slot)
        {
            int active_slot = bs.active_slots[slot];
            // 新 token 在序列中的位置：current_prompt_len 每步 decode 已递增，即为写入位置
            // （与上面 RoPE、seq_lens 的位置口径一致）
            int seq_len = bs.current_prompt_len[active_slot];
            int logical_block_idx = seq_len / BLOCK_SIZE;
            int token_in_block_idx = seq_len % BLOCK_SIZE;

            // 逻辑块首次触及（token_in_block==0 → 页表为 -1）时分配新物理块
            int physical_block_idx = pkv.getPhysicalBlock(active_slot, layer, logical_block_idx);
            if (physical_block_idx == -1)
            {
                physical_block_idx = pkv.getFreePhysicalBlock();
                pkv.setPhysicalBlock(active_slot, layer, logical_block_idx, physical_block_idx);
            }

            // 该 slot 的新 token K/V（已 RoPE，位于 ctx.key/value_states 第 slot
            // 行）写入块内位置
            pkv.cacheKV(physical_block_idx, ctx.key_states, ctx.value_states, slot, 1,
                        token_in_block_idx);
        }

        // TODO:
        // synchronize block table on cpu with block table on gpu (for attention)
        CUDA_CHECK(cudaMemcpy(pkv.block_table_gpu, pkv.block_table.data(),
                              MAX_SEQUENCES * N_LAYERS * MAX_BLOCKS_PER_SEQ * sizeof(int),
                              cudaMemcpyHostToDevice));

        // 注意力输出写入 ctx.context（= scratch_a），供 oProj 读取
        ctx.context = ctx.scratch_a;
        pagedAttention(layer, num_active_slots, ctx.query_states, pkv.kv_cache, pkv.block_table_gpu,
                       ctx.seq_lens, ctx.active_slots, ctx.context);

        ctx.oProj(w.o_proj[layer]);

        residualAdd(ctx.hidden_state, ctx.attn_output, num_active_slots);

        rmsNorm(ctx.hidden_state, ctx.normed_hidden, w.post_attn_layernorms[layer],
                num_active_slots);

        ctx.swiGLU(w.gate_proj[layer], w.up_proj[layer], w.down_proj[layer]);

        residualAdd(ctx.hidden_state, ctx.down, num_active_slots);
    }

    rmsNorm(ctx.hidden_state, ctx.normed_hidden, w.norm, num_active_slots);

    ctx.lmHead(w.embed_tokens);

    CUDA_CHECK(cudaMemcpy(ctx.logits_cpu.data(), ctx.logits,
                          sizeof(nv_bfloat16) * num_active_slots * VOCAB_SIZE,
                          cudaMemcpyDeviceToHost));

    float max_logit = 0.0;
    int max_token_idx = 0;
    for (int slot = 0; slot < num_active_slots; ++slot)
    {
        int active_slot = bs.active_slots[slot];
        max_logit = (float) ctx.logits_cpu[slot * VOCAB_SIZE];
        max_token_idx = 0;
        for (int token_idx = 0; token_idx < VOCAB_SIZE; ++token_idx)
        {
            if ((float) ctx.logits_cpu[slot * VOCAB_SIZE + token_idx] > max_logit)
            {
                max_logit = (float) ctx.logits_cpu[slot * VOCAB_SIZE + token_idx];
                max_token_idx = token_idx;
            }
        }

        if (max_token_idx == END_OF_TEXT_TOKEN_ID || max_token_idx == EOT_ID_TOKEN_ID ||
            bs.current_prompt_len[active_slot] == MAX_SEQ_LEN - 1)
        {
            out.emplace_back(bs.requests[active_slot],
                             ctx.tok->Decode(bs.generated_tokens[active_slot]),
                             bs.generated_tokens[active_slot]);

            bs.is_slot_free[active_slot] = true;
            for (int layer = 0; layer < N_LAYERS; ++layer)
            {
                for (int logical_block_idx = 0; logical_block_idx < MAX_BLOCKS_PER_SEQ;
                     ++logical_block_idx)
                {
                    int block_idx = active_slot * N_LAYERS * MAX_BLOCKS_PER_SEQ +
                                    layer * MAX_BLOCKS_PER_SEQ + logical_block_idx;
                    if (pkv.block_table[block_idx] != -1)
                    {
                        pkv.free_blocks.push_back(pkv.block_table[block_idx]);
                        pkv.block_table[block_idx] = -1;
                    }
                }
            }

            // TODO:
            CUDA_CHECK(cudaMemcpy(pkv.block_table_gpu, pkv.block_table.data(),
                                  MAX_SEQUENCES * N_LAYERS * MAX_BLOCKS_PER_SEQ * sizeof(int),
                                  cudaMemcpyHostToDevice));
        }
        else
        {
            bs.last_generated_tokens[active_slot] = max_token_idx;
            bs.generated_tokens[active_slot].push_back(max_token_idx);
            bs.current_prompt_len[active_slot] = bs.current_prompt_len[active_slot] + 1;
        }
    }

    return out;
}

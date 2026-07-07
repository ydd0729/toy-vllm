#pragma once

#include "config.hpp"

#include <memory>
#include <vector>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <queue>
#include <weights.hpp>
#include <tokenizers_cpp.h>

struct InferenceContext
{
public:
    InferenceContext();
    ~InferenceContext();
    InferenceContext(const InferenceContext&) = delete;
    InferenceContext& operator=(const InferenceContext&) = delete;

    void copyInputTokensToGPU();
    void InitHiddenState();
    cublasStatus_t linear(const nv_bfloat16* input,
                          const nv_bfloat16* weight,
                          nv_bfloat16* output,
                          int input_rows,
                          int input_cols,
                          int weight_rows,
                          float alpha = 1.0f,
                          float beta = 0.0f,
                          int leading_dim_input = 0,
                          int leading_dim_weight = 0,
                          int leading_dim_output = 0);

    cublasStatus_t matmul(const nv_bfloat16* a,
                          const nv_bfloat16* b,
                          nv_bfloat16* c,
                          int a_rows,
                          int a_cols,
                          int b_cols,
                          float alpha = 1.0f,
                          float beta = 0.0f,
                          int leading_dim_a = 0,
                          int leading_dim_b = 0,
                          int leading_dim_c = 0);

    void qProj(const nv_bfloat16* q_proj);
    void kProj(const nv_bfloat16* k_proj);
    void vProj(const nv_bfloat16* v_proj);
    void oProj(const nv_bfloat16* o_proj);
    void
    swiGLU(const nv_bfloat16* gate_proj, const nv_bfloat16* up_proj, const nv_bfloat16* down_proj);
    void lmHead(const nv_bfloat16* embed_tokens);

    nv_bfloat16* qHead(int i) const { return query_states + i * HEAD_DIM; }
    nv_bfloat16* kHead(int i) const { return key_states + i * HEAD_DIM; }
    nv_bfloat16* vHead(int i) const { return value_states + i * HEAD_DIM; }
    nv_bfloat16* attentionHead(int i)
    {
        return attn_weights + i * input_token_len * input_token_len;
    }

    void updateAttentionScoresHead(int i);
    void updateAttentionOutputHead(int i);

    void setInputTokens(const std::vector<int>& input_tokens)
    {
        input_token_len = input_tokens.size();
        num_rows = static_cast<int>(input_token_len);
        this->input_tokens_cpu = input_tokens;
    }

public:
    // cuBLAS context handle used by every GEMM below
    cublasHandle_t cublas_handle;

    // ---- input ----
    std::vector<int> input_tokens_cpu; // current prompt's token ids (CPU)
    size_t input_token_len = 0;        // == input_tokens.size()（prefill 的 prompt 长度）

    // 本次前向处理的行数（token 数）：prefill = input_token_len，decode = 活跃序列数。
    // 各投影（qProj/kProj/vProj/oProj/swiGLU/lmHead）的 GEMM 行数统一读它。
    int num_rows = 0;
    int* input_tokens;             // GPU copy of input_tokens (prefill)
    nv_bfloat16* input_embeddings; // embeddings gathered from embed_tokens by id

    // ---- backbone (residual stream) ----
    nv_bfloat16* hidden_state;  // residual stream carried through every layer
    nv_bfloat16* normed_hidden; // normalized copy fed to each sublayer

    // ---- shared scratch buffers (the alias pointers below point into these) ----
    nv_bfloat16* scratch_a; // reused by query_states (Q proj) and context (softmax*V)
    nv_bfloat16* scratch_b; // reused by attn_output (output proj) and down (FFN)

    // ---- attention ----
    nv_bfloat16* query_states; // Q projection; alias -> scratch_a
    nv_bfloat16* key_states;   // K projection (prefill)
    nv_bfloat16* value_states; // V projection (prefill)
    nv_bfloat16* attn_weights; // per-head Q*K^T / sqrt(HEAD_DIM) scores; softmax overwrites in
                               // place into attention weights
    nv_bfloat16* context; // per-head softmax(attn_weights)*V, concatenated (context vector); alias
                          // -> scratch_a
    nv_bfloat16* attn_output; // output projection (W_O) of context; attention block output added to
                              // residual; alias -> scratch_b

    // ---- FFN / MLP ----
    nv_bfloat16* gate; // gate projection (SwiGLU)
    nv_bfloat16* up;   // up projection
    nv_bfloat16* down; // down projection; alias -> buf_2048_2

    // ---- output logits ----
    nv_bfloat16* logits;                 // logits on GPU (hidden * embed_tokens^T)
    std::vector<nv_bfloat16> logits_cpu; // logits copied back to CPU for argmax

    // ---- decode-only: per-active-slot GPU staging (mirrors CPU batch vectors) ----
    int* last_tokens;  // last generated token id per active slot (fed to embeddingGatherDecode)
    int* active_slots; // active slot indices
    int* seq_lens;     // current sequence length per active slot

    std::unique_ptr<tokenizers::Tokenizer> tok;
};

struct PagedKVCache
{
public:
    // kv_cache: 2GB GPU 显存池，存储所有序列、所有层的 K/V 张量，按固定大小的 block 划分
    nv_bfloat16* kv_cache;

    // free_blocks: 可用物理 block 索引列表（0~NUM_BLOCKS-1），分配时 pop，释放时 push
    std::vector<int> free_blocks;

    // block_table: CPU 端页表，映射 (sequence, layer, logical_block) -> physical_block，-1
    // 表示未分配
    std::vector<int> block_table;

    // block_table_gpu: 页表的 GPU 副本，pagedAttention kernel 需要通过它查找 K/V 的实际地址
    int* block_table_gpu;

    PagedKVCache();
    PagedKVCache(const PagedKVCache&) = delete;
    PagedKVCache& operator=(const PagedKVCache&) = delete;
    ~PagedKVCache();

    int getFreePhysicalBlock();
    int getPhysicalBlock(int slot, int layer, int logical_block);
    void setPhysicalBlock(int slot, int layer, int logical_block, int physical_block);
    void cacheKV(int physical_block,
                 nv_bfloat16* k,
                 nv_bfloat16* v,
                 int src_token_offset,
                 int token_len,
                 int token_in_block = 0);
};

struct BatchState
{
    // batch 中各 slot 的空闲状态
    std::vector<bool> is_slot_free;

    // 每个 slot 已生成的 token 序列
    std::vector<std::vector<int>> generated_tokens;

    // 每个 slot 上次生成的 token id
    std::vector<int> last_generated_tokens;

    // 每个 slot 当前 prompt 的长度
    std::vector<int> current_prompt_len;

    // 每个 slot 当前装的是第几个请求（按 prompt 入队/取出顺序，对应 PROMPT N）；-1 表示空闲
    std::vector<int> request_id;

    // 每个 slot 对应的 prompt
    std::vector<std::vector<int>> input_tokens;

    std::vector<std::string> input_message;

    // 下一个请求编号，每次 prefill 取走一个 prompt 时分配并自增
    int next_request_id = 0;

    // needed to provide contiguous data for decode
    std::vector<int> active_slots;
    std::vector<int> active_tokens;

    BatchState();
};

struct RequestOutput
{
    int request_id;
    std::string input_message;
    std::string output_message;
    std::vector<int> input_token;
    std::vector<int> output_token;
};

void prefill(std::queue<std::string>& prompt_queue,
             int slot,
             InferenceContext& ctx,
             PagedKVCache& pkv,
             BatchState& bs,
             Weights& w);

std::vector<RequestOutput>
decode(InferenceContext& ctx, PagedKVCache& pkv, BatchState& bs, Weights& w);
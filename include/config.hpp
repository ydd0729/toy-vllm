#pragma once
#include <cuda_bf16.h>
#include <filesystem>

// =============================================================================
// Model architecture — fixed by Llama 3.2 1B (from its config.json).
// =============================================================================
constexpr int N_LAYERS = 16;
// Dimension of a token's embedding vector (hidden_size)
constexpr int EMBEDDING_LENGTH = 2048;
// Number of query heads in multi-head attention (num_attention_heads)
constexpr int NUM_Q_HEADS = 32;
// Number of key / value heads (num_key_value_heads); GQA => fewer than Q heads
constexpr int NUM_K_HEADS = 8;
constexpr int NUM_V_HEADS = 8;
// Per-head attention dimension (head_dim).
constexpr int HEAD_DIM = 64;
// Combined K/V projection width (derived): NUM_K_HEADS * HEAD_DIM = 8 * 64 = 512
constexpr int KV_DIM = NUM_K_HEADS * HEAD_DIM;
// GQA sharing ratios (Q heads per KV head), derived
constexpr int GQA_Q_TO_K_RATIO = NUM_Q_HEADS / NUM_K_HEADS;           // 32 / 8 = 4
constexpr int GQA_ATTN_SCORES_TO_V_RATIO = NUM_Q_HEADS / NUM_V_HEADS; // 32 / 8 = 4
// FFN intermediate dimension (intermediate_size)
constexpr int HIDDEN_DIM = 8192;
// Vocabulary size
constexpr int VOCAB_SIZE = 128256;
// Special token ids
constexpr int END_OF_TEXT_TOKEN_ID = 128001; // <|end_of_text|>
constexpr int EOT_ID_TOKEN_ID = 128009;      // <|eot_id|>

// =============================================================================
// Runtime / deployment tuning — our choices, not dictated by the model.
// Candidates to expose as CLI arguments later.
// =============================================================================
// Self-imposed max sequence length (input + output tokens). Model supports 128K.
constexpr int MAX_SEQ_LEN = 2048;
// Self-imposed max prompt length. Model allows up to 2048 for input + output.
constexpr int MAX_PROMPT_LEN = 512;
// Number of sequences processed together. TODO: only here to have batching at all
constexpr int BATCH_SIZE = 2;
// Tokens per PagedAttention page/block
constexpr int BLOCK_SIZE = 16;
// KV cache memory pool size. TODO: 改成可配置
constexpr size_t KV_CACHE_SIZE_BYTES = 2ULL * 1024 * 1024 * 1024;

// =============================================================================
// Pure constants & derived values — computed from (1) / (2), never hand-set.
// =============================================================================
// Byte unit conversions
constexpr int B_TO_MB = 1024 * 1024;
constexpr int B_TO_GB = 1024 * 1024 * 1024;
// Attention score scaling factor: sqrt(HEAD_DIM) = sqrt(64) = 8
constexpr float SQRT_HEAD_DIM = 8;
// Attention score scale applied to Q·K^T before softmax: 1 / sqrt(HEAD_DIM)
constexpr float ATTN_SCALE = 1.0f / SQRT_HEAD_DIM;
// Widest per-token buffer we need (prefill length vs. batch width)
constexpr int MAX_BUFFER_SIZE = std::max(MAX_PROMPT_LEN, BATCH_SIZE);
// Bytes for one K (or V) block; BLOCK_BYTES doubles it for K and V together
constexpr int V_OFFSET = BLOCK_SIZE * KV_DIM * sizeof(nv_bfloat16);
constexpr int BLOCK_BYTES = V_OFFSET * 2; // * 2 because K and V
// Paging derived sizes
constexpr int MAX_BLOCKS_PER_SEQ = MAX_SEQ_LEN / BLOCK_SIZE;  // 2048 / 16 = 128
constexpr int NUM_BLOCKS = KV_CACHE_SIZE_BYTES / BLOCK_BYTES; // 2*1024^3 / (16*512*2*2) = 65536
constexpr int MAX_SEQUENCES = BATCH_SIZE;

const std::filesystem::path model_path = std::filesystem::path("models");
const std::filesystem::path llama3p2_1B_Instruct_path = model_path / "Llama-3.2-1B-Instruct";
const std::filesystem::path llama3p2_1B_Instruct_weights_path =
    llama3p2_1B_Instruct_path / "model.safetensors";
const std::filesystem::path llama3p2_1B_Instruct_tokenizer_path =
    llama3p2_1B_Instruct_path / "tokenizer.json";

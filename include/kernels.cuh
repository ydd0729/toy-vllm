#pragma once
#include <cuda_bf16.h>
#include "config.hpp"


// =============================================================================
//                        Prefill / Shared Kernels
// =============================================================================

void embeddingGather(int* gpu_input_tokens,
                     nv_bfloat16* gpu_input_embeds,
                     nv_bfloat16* embed_tokens,
                     int num_input_tokens);
void rmsNorm(nv_bfloat16* input, nv_bfloat16* output, nv_bfloat16* norm_weights, int num_tokens);
void rope(nv_bfloat16* input, int num_tokens, int proj_dim);
void causalMask(nv_bfloat16* input, int num_tokens);
void softmax(nv_bfloat16* input, int num_tokens);
void residualAdd(nv_bfloat16* input, nv_bfloat16* input_embeds, int num_tokens);
void silu(nv_bfloat16* a, nv_bfloat16* b, int num_tokens);

// =============================================================================
//                                  Decode
// =============================================================================

void embeddingGatherDecode(int* gpu_last_tokens, int num_tokens, nv_bfloat16* output, nv_bfloat16* embed_tokens);
void ropeDecode(nv_bfloat16* input, int position_in_sequence, int proj_dim);
void softmaxDecode(nv_bfloat16* input, int seq_len);

// page attention
void pagedAttention(int layer,
                    int num_active_slots,
                    nv_bfloat16* q_proj,
                    nv_bfloat16* kv_cache,
                    int* block_table_gpu,
                    int* gpu_seq_lens,
                    int* gpu_active_slots,
                    nv_bfloat16* output);
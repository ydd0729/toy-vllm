#pragma once
#include <cuda_bf16.h>

// prefill
void embeddingGather(int *gpu_input_tokens, __nv_bfloat16 *gpu_input_embeds, __nv_bfloat16 *embed_tokens, int num_input_tokens);
void rmsNorm(__nv_bfloat16 *input, __nv_bfloat16 *output, nv_bfloat16 *norm_weights, int num_tokens);
void rope(__nv_bfloat16 *input, int num_tokens, int proj_dim);
void causalMask(__nv_bfloat16 *input, int num_tokens);
void softmax(__nv_bfloat16 *input, int num_tokens);
void residualAdd(__nv_bfloat16 *input, __nv_bfloat16 *input_embeds, int num_tokens);
void silu(__nv_bfloat16 *a, __nv_bfloat16 *b, int num_tokens);

// decode
void embeddingGatherDecode(int input_token, __nv_bfloat16 *output, __nv_bfloat16 *embed_tokens);
void ropeDecode(__nv_bfloat16 *input, int position_in_sequence, int proj_dim);
void softmaxDecode(__nv_bfloat16 *input, int seq_len);
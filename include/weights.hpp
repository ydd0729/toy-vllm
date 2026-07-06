#pragma once
#include <cuda_bf16.h>
#include <iostream>
#include <fstream>
#include "config.hpp"

struct Weights
{
    nv_bfloat16* embed_tokens;
    nv_bfloat16* input_layernorm[N_LAYERS];
    nv_bfloat16* gate_proj[N_LAYERS];
    nv_bfloat16* up_proj[N_LAYERS];
    nv_bfloat16* down_proj[N_LAYERS];
    nv_bfloat16* post_attn_layernorms[N_LAYERS];
    nv_bfloat16* k_proj[N_LAYERS];
    nv_bfloat16* o_proj[N_LAYERS];
    nv_bfloat16* q_proj[N_LAYERS];
    nv_bfloat16* v_proj[N_LAYERS];
    nv_bfloat16* norm;

    Weights();
    ~Weights();
    Weights(const Weights&) = delete;
    Weights& operator=(const Weights&) = delete;

private:
    char* gpu_model_weights;
};
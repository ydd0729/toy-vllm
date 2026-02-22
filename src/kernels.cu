#include "kernels.cuh"
#include <iostream>

constexpr int HEAD_DIM = 64;
constexpr int NUM_Q_HEADS = 32;

// gpu_input_tokens - N tokens
// gpu_input_embeds - N * sizeof(__nv_bfloat16) * 2048
// embed_tokens - (100000+smth, 2048)
// num_input_tokens - N (just N, not N tokens)
__global__ void embeddingGatherKernel(int *gpu_input_tokens, __nv_bfloat16 *gpu_input_embeds, __nv_bfloat16 *embed_tokens, int num_input_tokens)
{
    int workIndex = threadIdx.x + blockIdx.x * 2048;
    if (workIndex < num_input_tokens * 2048)
    {
        gpu_input_embeds[workIndex] = embed_tokens[gpu_input_tokens[blockIdx.x] * 2048 + threadIdx.x];
        gpu_input_embeds[workIndex + 1024] = embed_tokens[gpu_input_tokens[blockIdx.x] * 2048 + threadIdx.x + 1024];
    }
}

void embeddingGather(int *gpu_input_tokens, __nv_bfloat16 *gpu_input_embeds, __nv_bfloat16 *embed_tokens, int num_input_tokens)
{
    // even though embedding is 2048, I can only dispatch 1024 because it's max threads per block on my gpu
    embeddingGatherKernel<<<num_input_tokens, 1024>>>(gpu_input_tokens, gpu_input_embeds, embed_tokens, num_input_tokens);
#ifdef DEBUG
    auto error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
    }
#endif
}

__global__ void rmsNormKernel(__nv_bfloat16 *input, __nv_bfloat16 *output, __nv_bfloat16 *norm_weights, int num_tokens)
{
    __shared__ float rms_vector[1024];
    int workIndex = threadIdx.x + blockIdx.x * 2048;
    if (workIndex < num_tokens * 2048)
    {
        rms_vector[threadIdx.x] = (float)input[workIndex] * (float)input[workIndex] + (float)input[workIndex + 1024] * (float)input[workIndex + 1024];
        __syncthreads();
        // tree reduction
        for (int i = 1; i < 1024; i = i * 2)
        {
            if (threadIdx.x % (i * 2) == 0)
            {
                rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + i];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0)
        {
            rms_vector[0] = sqrt(rms_vector[0] / 2048.0 + 1.0e-5);
        }
        __syncthreads();
        // <(^-^)>
        output[workIndex] = (__nv_bfloat16)(((float)input[workIndex] / rms_vector[0]) * (float)norm_weights[threadIdx.x]);
        output[workIndex + 1024] = (__nv_bfloat16)(((float)input[workIndex + 1024] / rms_vector[0]) * (float)norm_weights[threadIdx.x + 1024]);
    }
}

// (N, 2048) -> (N, 2048)
void rmsNorm(__nv_bfloat16 *input, __nv_bfloat16 *output, __nv_bfloat16 *norm_weights, int num_tokens)
{
    rmsNormKernel<<<num_tokens, 1024>>>(input, output, norm_weights, num_tokens);
#ifdef DEBUG
    auto error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
    }
#endif
}

__global__ void ropeKernel(__nv_bfloat16 *input, int num_tokens, int proj_dim)
{
    if (2 * threadIdx.x + 1 + blockIdx.x * proj_dim < num_tokens * proj_dim)
    {
        // TODO: precompute thetas, angles and perhaps sin/cos vals and reuse it across all kernel invocations
        int double_i = 2 * (threadIdx.x % 32);
        float theta = 1.0 / (pow(500000.0, ((float)double_i / HEAD_DIM)));
        float angle = blockIdx.x * theta;
        __nv_bfloat16 prev_2i = input[2 * threadIdx.x + blockIdx.x * proj_dim];
        __nv_bfloat16 prev_2i_1 = input[2 * threadIdx.x + 1 + blockIdx.x * proj_dim];
        input[2 * threadIdx.x + blockIdx.x * proj_dim] = (__nv_bfloat16)((float)prev_2i * cos(angle) - (float)prev_2i_1 * sin(angle));
        input[2 * threadIdx.x + 1 + blockIdx.x * proj_dim] = (__nv_bfloat16)((float)prev_2i * sin(angle) + (float)prev_2i_1 * cos(angle));
    }
}

// proj_dim: q_proj 2048, k_proj 512
// num_threads: I want to use it for both q_proj and k_proj so need to parameterize num_threads (1024 for q_proj and 512 for k_proj)
void rope(__nv_bfloat16 *input, int num_tokens, int proj_dim)
{
    int num_threads = proj_dim / 2;
    if (num_threads > 1024)
    {
        std::cout << "Can't launch more than 1024 threads on RTX 5090, RoPE kernel not launched";
        return;
    }

    ropeKernel<<<num_tokens, num_threads>>>(input, num_tokens, proj_dim);
#ifdef DEBUG
    auto error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
    }
#endif
}

__global__ void causalMaskKernel(__nv_bfloat16 *input, int num_tokens)
{
    if (threadIdx.x + blockIdx.x * blockDim.x >= num_tokens * num_tokens * NUM_Q_HEADS)
    {
        return;
    }

    int head_idx = blockIdx.x / num_tokens; // because blockId.x is in range [0, num_tokens * NUM_Q_HEADS]
    int column = threadIdx.x;
    int row = blockIdx.x / NUM_Q_HEADS;
    if (column > row)
    {
        input[blockIdx.x + threadIdx.x] = -HUGE_VALF;
    }
}

void causalMask(__nv_bfloat16 *input, int num_tokens)
{
    if (num_tokens > 1024)
    {
        std::cout << "Can't launch more than 1024 threads on RTX 5090, Causal mask kernel not launched";
        return;
    }

    causalMaskKernel<<<num_tokens * NUM_Q_HEADS, num_tokens>>>(input, num_tokens);
#ifdef DEBUG
    auto error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
    }
#endif
}
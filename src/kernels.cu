#include "kernels.cuh"
#include <iostream>

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
        std::cout << "CUDA last error: " cudaGetLastError() << std::endl;
    }
#endif
}

__global__ void rmsNormKernel(__nv_bfloat16 *input, __nv_bfloat16 *output, nv_bfloat16 *norm_weights, int num_tokens)
{
    __shared__ __nv_bfloat16 rms_vector[1024];
    __shared__ __nv_bfloat16 rms;
    int workIndex = threadIdx.x + blockIdx.x * 2048;
    if (workIndex < num_tokens * 2048)
    {
        rms_vector[threadIdx.x] = input[workIndex] * input[workIndex] + input[workIndex + 1024] * input[workIndex + 1024];
        __syncthreads();
        // tree reduction
        if (threadIdx.x % 2 == 0) {
	    // every second has its predecessor
            rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + 1]; 
        }
        __syncthreads();
	if (threadIdx.x % 4 == 0) {
	    rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + 2];
	}
	__syncthreads();
	if (threadIdx.x % 8 == 0) {
	    rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + 4];
	}
	__syncthreads();
	if (threadIdx.x % 16 == 0) {
	    rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + 8];
	}
	__syncthreads();
	if (threadIdx.x % 32 == 0) {
	    rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + 16];
	}
	__syncthreads();
	if (threadIdx.x % 64 == 0) {
	    rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + 32];
	}
	__syncthreads();
	if (threadIdx.x % 128 == 0) {
	    rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + 64];
	}
	__syncthreads();
	if (threadIdx.x % 256 == 0) {
	    rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + 128];
	}
	__syncthreads();
	if (threadIdx.x % 512 == 0) {
	    rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + 256];
	}
	__syncthreads();
	if (threadIdx.x % 1024 == 0) {
	    rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + 512];
	}
	__syncthreads();
	if (threadIdx.x == 0) {
	    rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + 1024];
	}
	__syncthreads();
        // <(^-^)>
        output[workIndex] = (input[workIndex] / rms_vector[0]) * norm_weights[threadIdx.x];
    }
}

// (N, 2048) -> (N, 2048)
void rmsNorm(__nv_bfloat16 *input, __nv_bfloat16 *output, nv_bfloat16 *norm_weights, int num_tokens)
{
    rmsNormKernel<<<num_tokens, 1024>>>(input, output, norm_weights, num_tokens);
#ifdef DEBUG
    auto error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " cudaGetLastError() << std::endl;
    }
#endif
}

#include "kernels.cuh"
#include <iostream>


// gpu_input_tokens - N tokens
// gpu_input_embeds - N * sizeof(__nv_bfloat16) * 2048
// embed_tokens - (100000+smth, 2048)
// num_input_tokens - N (just N, not N tokens)
__global__ void embeddingGatherKernel(int *gpu_input_tokens, __nv_bfloat16 *gpu_input_embeds, __nv_bfloat16 *embed_tokens, int num_input_tokens)
{
    int workIndex = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIndex < num_input_tokens)
    {
        // __nv_bfloat16 embed[2048];
        for (int i = 0; i < 2048; ++i)
        {
            gpu_input_embeds[workIndex*2048 + i] = embed_tokens[gpu_input_tokens[workIndex]* 2048 + i];
        }
        // gpu_input_embeds[workIndex*2048] = *embed;
    }
}

void embeddingGather(int *gpu_input_tokens, __nv_bfloat16 *gpu_input_embeds, __nv_bfloat16 *embed_tokens, int num_input_tokens)
{
    // TODO: naively I run 1x1 grid (1 block) with 1xnum_input_tokens threads (num_input_tokens threads, so every thread gathers the full embedding)
    // dumb but that's just to start ok??
    embeddingGatherKernel<<<1, num_input_tokens>>>(gpu_input_tokens, gpu_input_embeds, embed_tokens, num_input_tokens);
    std::cout << cudaGetLastError() << std::endl;
}

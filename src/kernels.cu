#include "kernels.cuh"
#include <iostream>
#include <math_constants.h>
#include "config.hpp"

// =============================================================================
//                        Prefill / Shared Kernels
// =============================================================================

/**
 * @brief 并行地从词表矩阵 (embed_tokens) 中查找并提取 N 个输入 token 的 2048 维 embedding 向量。
 *
 * __global__ 是 CUDA C++ 中的函数修饰符，表示这是一个 kernel 函数。
 *
 * - 调用与执行：它在 CPU（Host）端调用，但在 GPU（Device）端执行。
 * - 并行特性：调用时必须使用 <<<网格大小, 线程块大小>>> 语法（如代码中的 <<<num_input_tokens,
 * 1024>>>）， 这会启动大量线程在 GPU 上并行执行该函数的代码。
 * - 限制：返回值必须是 void。
 *
 * @param gpu_input_tokens N 个 token
 * @param gpu_input_embeds [N, 2048]
 * @param embed_tokens     词表矩阵 [100000+, 2048]
 * @param num_input_tokens N
 */
__global__ void embeddingGatherKernel(int* gpu_input_tokens,
                                      nv_bfloat16* gpu_input_embeds,
                                      nv_bfloat16* embed_tokens,
                                      int num_input_tokens)
{
    // blockIdx 表示第几个 Token
    // threadIdx 表示向量的维度（2048），但调用时的线程块被设为了 1024，所以每个线程复制两个值。
    // checkGPUStatus() 检查了每个 block 能支持的最大 thread 数量。
    int workIdx = threadIdx.x + blockIdx.x * 2048;
    if (workIdx < num_input_tokens * 2048)
    {
        // gpu_input_tokens[blockIdx.x]：获取当前正在处理的实际 token id
        // gpu_input_tokens[blockIdx.x] * 2048：每个 embedding 的长度是 2048
        // gpu_input_tokens[blockIdx.x] * 2048 +
        // threadIdx.x：加上当前线程负责的列索引，定位到具体的维度
        gpu_input_embeds[workIdx] = embed_tokens[gpu_input_tokens[blockIdx.x] * 2048 + threadIdx.x];
        // 这里访问 threadIdx.x 和 threadIdx.x + 1024（而非相邻的 2*threadIdx.x 和
        // 2*threadIdx.x+1）， 是为了保证同一个 Warp（32
        // 个线程）在同一时刻访问连续的内存地址，实现合并访存（Coalesced Access）。
        // 若改为相邻位置，Warp 内线程会读取 0,2,4,6... 这样的跨步地址，导致内存带宽浪费。
        gpu_input_embeds[workIdx + 1024] =
            embed_tokens[gpu_input_tokens[blockIdx.x] * 2048 + threadIdx.x + 1024];
    }
}

/**
 * @brief 一个在 CPU（Host）端执行的包装函数，用于配置并启动 GPU 上的 embeddingGatherKernel。
 *
 * @param gpu_input_tokens N 个 token
 * @param gpu_input_embeds [N, 2048]
 * @param embed_tokens     词表矩阵 [100000+, 2048]
 * @param num_input_tokens N
 */
void embeddingGather(int* gpu_input_tokens,
                     nv_bfloat16* gpu_input_embeds,
                     nv_bfloat16* embed_tokens,
                     int num_input_tokens)
{
    embeddingGatherKernel<<<num_input_tokens, 1024>>>(gpu_input_tokens, gpu_input_embeds,
                                                      embed_tokens, num_input_tokens);
#ifdef DEBUG
    cudaError error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
    }
#endif
}

/**
 * @brief RMS Normalization 均方根归一化
 *
 * RMSNorm(x) = (x / sqrt(mean(x^2) + eps)) * gamma
 *
 * 其中:
 *   d     = 2048 (embedding 维度)
 *   eps   = 1e-5 (防止除零)
 *   gamma = norm_weights (可学习的缩放权重)
 *
 * @param input        输入向量 [N, 2048]，bfloat16
 * @param output       归一化后的输出向量 [N, 2048]，bfloat16
 * @param norm_weights 可学习的缩放权重 gamma [2048]，bfloat16
 * @param num_tokens   token 数量 N（即 grid 中的 block 数量）
 */
__global__ void
rmsNormKernel(nv_bfloat16* input, nv_bfloat16* output, nv_bfloat16* norm_weights, int num_tokens)
{
    // __shared__ 是 CUDA 的内存修饰符，声明变量位于共享内存（Shared Memory）中。
    //
    // 特性：
    //
    // - 作用域：同一个 block 内的所有线程都能读写这块内存，是块内线程间通信的方式。
    // - 速度：位于 GPU 芯片上（on-chip），访问速度接近 L1 Cache，远快于全局显存（global memory）。
    // - 容量：很小，每个 SM 通常只有 48KB~164KB。
    // - 生命周期：仅在 block 执行期间存在，block 结束后自动释放。
    //
    // 在这里，__shared__ float rms_vector[1024] 用于让 1024 个线程协作完成树形归约求和：
    // 每个线程写入自己的平方和，然后互相读取、累加，最终得到整个 token 的平方和。
    //
    // 这里使用 float 是为了在计算过程中保留更多精度（bfloat16 的指数位和 float 一样多，但尾数为只有
    // 7 位）
    __shared__ float rms_vector[1024];
    int workIdx = threadIdx.x + blockIdx.x * 2048;
    if (workIdx < num_tokens * 2048)
    {
        // 两个元素的平方的和
        rms_vector[threadIdx.x] = (float) input[workIdx] * (float) input[workIdx] +
                                  (float) input[workIdx + 1024] * (float) input[workIdx + 1024];

        // __syncthreads() 是块内屏障同步（Barrier）：block 中所有线程必须都执行到这里才能继续。
        // 这里确保每个线程都已将平方和写入 rms_vector 后，再开始树形归约读取其他线程的数据。
        __syncthreads();

        // 树形归约（Tree Reduction）：用 log2(1024) = 10 轮将 1024 个值归约为 1 个总和。
        // 每轮中，偶数号线程把相距 i 的两个值相加，步长 i 每轮翻倍（1,2,4,8...512）。
        // 第 1 轮：512 个线程各加 1 对 -> 512 个部分和
        // 第 2 轮：256 个线程各加 1 对 -> 256 个部分和
        // ...
        // 第 10 轮：线程 0 完成最后 1 次加法 -> rms_vector[0] 即为 2048 个元素的平方和
        for (int i = 1; i < 1024; i = i * 2)
        {
            // 随着 i 逐轮翻倍，越来越多的小编号线程被"淘汰"，只有编号是 i*2 倍数的线程才参与。
            // 到最后一轮（i=512），整个 block 中只有线程 0（0 % 1024 == 0）执行最终加法。
            if (threadIdx.x % (i * 2) == 0)
            {
                rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + i];
            }
            __syncthreads();
        }

        // 计算平均数并取平方根
        if (threadIdx.x == 0)
        {
            rms_vector[0] = sqrt(rms_vector[0] / 2048.0 + 1.0e-5);
        }
        __syncthreads();

        // 计算 RMS Norm
        output[workIdx] = (nv_bfloat16) (((float) input[workIdx] / rms_vector[0]) *
                                         (float) norm_weights[threadIdx.x]);
        output[workIdx + 1024] = (nv_bfloat16) (((float) input[workIdx + 1024] / rms_vector[0]) *
                                                (float) norm_weights[threadIdx.x + 1024]);
    }
}

/**
 * @brief rmsNormKernel 的包装函数
 */
void rmsNorm(nv_bfloat16* input, nv_bfloat16* output, nv_bfloat16* norm_weights, int num_tokens)
{
    rmsNormKernel<<<num_tokens, 1024>>>(input, output, norm_weights, num_tokens);
#ifdef DEBUG
    cudaError error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
    }
#endif
}

/**
 * @brief RoPE (Rotary Position Embedding) 旋转位置编码
 *
 * 将位置信息注入 Q/K 投影：对每对相邻维度 (x_{2i}, x_{2i+1}) 施加 2D 旋转，
 * 旋转角度 = position * theta_i，其中 theta_i 随维度升高而递减（低频）。
 *
 * 旋转公式（2D 旋转矩阵）：
 *   x'_{2i}   = x_{2i}   * cos(angle) - x_{2i+1} * sin(angle)
 *   x'_{2i+1} = x_{2i}   * sin(angle) + x_{2i+1} * cos(angle)
 *
 * 线程映射：
 *   blockIdx.x  = token 在序列中的位置 m
 *   threadIdx.x = 负责第 threadIdx.x 对维度
 *   每个 block 处理一个 token，每个线程处理一对维度，原地修改 input
 *
 * @param input      Q 或 K 投影结果 [N, proj_dim]，bfloat16，原地修改
 * @param num_tokens token 数量 N
 * @param proj_dim   投影维度（Q=2048, K=512）
 */
__global__ void ropeKernel(nv_bfloat16* input, int num_tokens, int proj_dim)
{
    if (2 * threadIdx.x + 1 + blockIdx.x * proj_dim < num_tokens * proj_dim)
    {
        // head_idx: 当前对在 head 内的维度索引（0, 2, 4, ..., 62），% 32 因为 HEAD_DIM=64 有 32 对
        // theta: 该维度对的旋转频率，维度越高频率越低（LLaMA 3 基频 500000）
        // angle: 旋转角度 = 位置 m * 频率 theta
        // TODO: precompute thetas, angles and perhaps sin/cos vals and reuse it across all kernel
        // invocations
        int head_idx = 2 * (threadIdx.x % 32);
        float theta = 1.0 / (pow(500000.0, ((float) head_idx / HEAD_DIM)));
        float angle = blockIdx.x * theta;

        // 读取原始值
        float prev_2i = input[2 * threadIdx.x + blockIdx.x * proj_dim];
        float prev_2j = input[2 * threadIdx.x + 1 + blockIdx.x * proj_dim];

        // 应用 2D 旋转并写回
        input[2 * threadIdx.x + blockIdx.x * proj_dim] =
            prev_2i * cos(angle) - prev_2j * sin(angle);
        input[2 * threadIdx.x + 1 + blockIdx.x * proj_dim] =
            prev_2i * sin(angle) + prev_2j * cos(angle);
    }
}

void rope(nv_bfloat16* input, int num_tokens, int proj_dim)
{
    int num_threads = proj_dim / 2;
    if (num_threads > 1024)
    {
        std::cout << "Can't launch more than 1024 threads on RTX 5090, RoPE kernel not launched";
        return;
    }

    ropeKernel<<<num_tokens, num_threads>>>(input, num_tokens, proj_dim);
#ifdef DEBUG
    cudaError error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
    }
#endif
}

/**
 * @brief RoPE（rotate_half / HF 约定版）
 *
 * 与 ropeKernel（相邻交错配对）并列的另一种实现。HF 的 safetensors 权重按 rotate_half 布局排列：
 * 每个 HEAD_DIM=64 维 head 内，通道 j 与通道 j+32 组成一对（前半 32 维与后半 32 维配对），
 * 而非相邻的 (2i, 2i+1)。基础频率 theta_j = 500000^(-2j/HEAD_DIM)，再经 llama3 rope_scaling 缩放。
 *
 *   x'_j      = x_j      * cos(angle) - x_{j+32} * sin(angle)
 *   x'_{j+32} = x_{j+32} * cos(angle) + x_j      * sin(angle)
 *
 * 线程映射：threadIdx.x = t → head h = t/32，对内索引 j = t%32
 */

// Llama 3.2 的 rope_scaling（config.json 里 rope_type="llama3"）：按波长分三段缩放频率。
//   波长 > 8192（低频）：频率 ÷ factor(32)；波长 < 2048（高频）：不变；中间：线性平滑过渡。
// 不实现它会导致每个 head 约一半维度的频率差 32 倍，位置越靠后位置编码偏得越远（长生成退化）。
__device__ inline float llama3ScaledTheta(int j)
{
    constexpr float FACTOR = 32.0f;            // rope_scaling.factor
    constexpr float LOW_FREQ_FACTOR = 1.0f;    // rope_scaling.low_freq_factor
    constexpr float HIGH_FREQ_FACTOR = 4.0f;   // rope_scaling.high_freq_factor
    constexpr float OLD_CONTEXT_LEN = 8192.0f; // rope_scaling.original_max_position_embeddings

    float theta = 1.0f / powf(500000.0f, ((float) (2 * j) / HEAD_DIM));
    float wavelen = 2.0f * CUDART_PI_F / theta;

    if (wavelen > OLD_CONTEXT_LEN / LOW_FREQ_FACTOR) // 低频段：频率整体除以 factor
    {
        return theta / FACTOR;
    }
    if (wavelen < OLD_CONTEXT_LEN / HIGH_FREQ_FACTOR) // 高频段：保持原频率
    {
        return theta;
    }
    // 过渡段：在 缩放/不缩放 之间线性插值
    float smooth =
        (OLD_CONTEXT_LEN / wavelen - LOW_FREQ_FACTOR) / (HIGH_FREQ_FACTOR - LOW_FREQ_FACTOR);
    return (1.0f - smooth) * theta / FACTOR + smooth * theta;
}

__global__ void ropeRotateHalfKernel(nv_bfloat16* input, int num_tokens, int proj_dim)
{
    int h = threadIdx.x / 32;
    int j = threadIdx.x % 32;
    int row = blockIdx.x * proj_dim;
    int i0 = row + h * HEAD_DIM + j; // 通道 j
    int i1 = i0 + HEAD_DIM / 2;      // 通道 j + 32

    if (i1 < num_tokens * proj_dim)
    {
        float theta = llama3ScaledTheta(j);
        float angle = blockIdx.x * theta;

        float x0 = input[i0];
        float x1 = input[i1];

        input[i0] = x0 * cos(angle) - x1 * sin(angle);
        input[i1] = x1 * cos(angle) + x0 * sin(angle);
    }
}

void ropeRotateHalf(nv_bfloat16* input, int num_tokens, int proj_dim)
{
    int num_threads = proj_dim / 2;
    if (num_threads > 1024)
    {
        std::cout << "Can't launch more than 1024 threads on RTX 5090, RoPE kernel not launched";
        return;
    }

    ropeRotateHalfKernel<<<num_tokens, num_threads>>>(input, num_tokens, proj_dim);
#ifdef DEBUG
    cudaError error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
    }
#endif
}

/**
 * @brief 因果掩码（Causal Mask）：阻止 token 关注未来的 token
 *
 * 自回归语言模型中，每个 token 只能看到它自己和之前的 token，不能看到未来的 token。
 * 通过将注意力分数矩阵上三角部分（column > row）设为 -infinity，
 * 使得 softmax 后这些位置的权重变为 0，从而实现因果约束。
 *
 * 输入形状：(NUM_Q_HEADS, num_tokens, num_tokens)
 * 掩码规则：如果 column > row，则 input[..., row, column] = -infinity
 *
 * 线程映射：
 *
 *   blockIdx.x  = head_idx * num_tokens + row（每个 block 处理一个 head 的一行，
 *   其中 row 是该 head 内注意力矩阵的行索引，范围 0 ~ num_tokens-1）
 *
 *   threadIdx.x = column（每个线程处理一列）
 *
 *   共 num_tokens * NUM_Q_HEADS 个 block，每个 block num_tokens 个线程
 *
 * @param input      注意力分数矩阵，原地修改，bfloat16
 * @param num_tokens token 数量
 */
__global__ void causalMaskKernel(nv_bfloat16* input, int num_tokens)
{
    // 边界检查：确保不超出总元素数
    if (threadIdx.x + blockIdx.x * blockDim.x >= num_tokens * num_tokens * NUM_Q_HEADS)
    {
        return;
    }

    int column = threadIdx.x;          // 当前关注的 token 位置
    int row = blockIdx.x % num_tokens; // 当前正在计算的 token 位置
    // 如果 column > row，说明当前 token 试图关注未来的 token，需要掩码
    if (column > row)
    {
        // 设为 -infinity，softmax 后变为 0
        input[blockIdx.x * num_tokens + threadIdx.x] = -CUDART_INF_F;
    }
}

void causalMask(nv_bfloat16* input, int num_tokens)
{
    if (num_tokens > 1024)
    {
        std::cout
            << "Can't launch more than 1024 threads on RTX 5090, Causal mask kernel not launched";
        return;
    }

    causalMaskKernel<<<num_tokens * NUM_Q_HEADS, num_tokens>>>(input, num_tokens);
#ifdef DEBUG
    cudaError error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
    }
#endif
}

/**
 * @brief Softmax 归一化：对注意力分数矩阵的每一行执行数值稳定的 softmax
 *
 * 数值稳定推导：
 *   标准 softmax:  softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
 *   当 x_i 较大时，exp(x_i) 会溢出 float 范围（如 exp(1000) = inf）。
 *
 *   引入常数 c = max(x)，对分子分母同乘 exp(-c)：
 *
 *     softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
 *                  = exp(x_i) * exp(-c) / (Σ_j exp(x_j) * exp(-c))
 *                  = exp(x_i - c) / Σ_j exp(x_j - c)
 *
 *   由于 c = max(x)，所以 x_i - c <= 0，exp(x_i - c) <= 1，永远不会溢出。
 *   分子分母同乘一个非零常数 exp(-c)，结果不变。
 *
 * 算法步骤：
 *   1. 找出每行的最大值 max_val（即 c）
 *   2. 计算 exp(x_i - max_val)
 *   3. 求 exp 之和
 *   4. 除以总和得到 softmax 结果
 *
 * 线程映射：
 *   blockIdx.x  = head_idx * num_tokens + row（每个 block 处理一个 head 的一行）
 *   threadIdx.x = column（每个线程处理一列）
 *   共 num_tokens * NUM_Q_HEADS 个 block，每个 block num_tokens 个线程
 *
 * @param input      经过因果掩码后的注意力分数矩阵 [NUM_Q_HEADS, num_tokens,
 * num_tokens]，原地修改，bfloat16
 * @param num_tokens token 数量
 */
__global__ void softmaxKernel(nv_bfloat16* input, int num_tokens)
{
    // 使用固定大小的共享内存存储一行数据，用于树形归约求最大值和总和
    __shared__ float row[1024]; // row[0] 最终存储最大值，之后复用为求和结果
    __shared__ float max_val;

    // 第一步：找出每行的最大值 max_val，用于数值稳定的 softmax（防止 exp 溢出）
    int workIdx = blockIdx.x * num_tokens + threadIdx.x;
    nv_bfloat16 attn_score = input[workIdx]; // 当前线程负责的注意力分数
    row[threadIdx.x] = attn_score;
    __syncthreads();

    for (int i = 1; i < num_tokens; i = i * 2)
    {
        if (threadIdx.x % (i * 2) == 0 && threadIdx.x + i < num_tokens)
        {
            row[threadIdx.x] = fmaxf(row[threadIdx.x], row[threadIdx.x + i]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        max_val = row[0];
    }
    __syncthreads();

    // 第二步：计算 exp(attn_score - max_val)
    // 在指数中减去最大值是数值稳定技巧：exp(x - max) 不会溢出，且 softmax 结果不变
    row[threadIdx.x] = expf((float) attn_score - max_val);
    __syncthreads();

    // 第三步：树形归约求 exp 之和（分母），复用 row 共享内存
    for (int i = 1; i < num_tokens; i = i * 2)
    {
        if (threadIdx.x % (i * 2) == 0 && threadIdx.x + i < num_tokens)
        {
            row[threadIdx.x] = row[threadIdx.x] + row[threadIdx.x + i];
        }
        __syncthreads();
    }

    // 第四步：softmax = exp(attn_score - max_val) / sum(exp)
    input[workIdx] = (nv_bfloat16) (expf((float) attn_score - max_val) / row[0]);
}

/**
 * @brief softmaxKernel 的包装函数
 *
 * @param input      经过因果掩码后的注意力分数矩阵 [NUM_Q_HEADS, num_tokens,
 * num_tokens]，原地修改，bfloat16
 * @param num_tokens token 数量
 */
void softmax(nv_bfloat16* input, int num_tokens)
{
    if (num_tokens > 1024)
    {
        std::cout << "Can't launch more than 1024 threads on RTX 5090, Softmax kernel not launched";
        return;
    }

    softmaxKernel<<<num_tokens * NUM_Q_HEADS, num_tokens>>>(input, num_tokens);
#ifdef DEBUG
    cudaError error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
    }
#endif
}

/**
 * @brief 残差连接（Residual Connection）
 *
 * 将子层输出与原始输入相加：output = sublayer_output + original_input
 * 这是 Transformer 的核心设计，帮助梯度在深层网络中流动，缓解梯度消失问题。
 *
 * 线程映射：
 *   blockIdx.x  = 第几个 token
 *   threadIdx.x = embedding 维度索引（0~1023）
 *   每个线程处理 2 个元素（workIdx 和 workIdx+1024），与 embeddingGatherKernel 相同
 *
 * @param input        子层输出 [N, 2048]，原地修改为相加后的结果
 * @param input_embeds 原始输入（子层前的 embedding）[N, 2048]
 */
__global__ void residualKernel(nv_bfloat16* input, nv_bfloat16* input_embeds)
{
    int workIdx = threadIdx.x + blockIdx.x * 2048;
    input[workIdx] = input[workIdx] + input_embeds[workIdx];
    input[workIdx + 1024] = input[workIdx + 1024] + input_embeds[workIdx + 1024];
}

void residualAdd(nv_bfloat16* input, nv_bfloat16* input_embeds, int num_tokens)
{
    residualKernel<<<num_tokens, 1024>>>(input, input_embeds);
#ifdef DEBUG
    cudaError error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
    }
#endif
}

/**
 * @brief SwiGLU 激活函数：对 FFN 的 gate 和 up 投影输出执行 SiLU 门控激活
 *
 * LLaMA 的 FFN 使用 SwiGLU 激活函数（Shazeer, 2020）：
 *   FFN(x) = (SiLU(x @ W_gate) ⊙ (x @ W_up)) @ W_down
 *
 * 本 kernel 计算中间步骤：
 *   output = SiLU(a) ⊙ b
 *        = a * sigmoid(a) * b
 *        = a * (1 / (1 + exp(-a))) * b
 *
 * 其中：
 *   - a = x @ W_gate（gate 投影输出），对其施加 SiLU 激活后作为门控信号
 *   - b = x @ W_up（up 投影输出），被门控信号逐元素调制
 *   - ⊙ 表示逐元素乘法（Hadamard product）
 *
 * 维度说明：
 *   FFN 中间层维度 INTERMEDIATE_SIZE = 8192（LLaMA 3.2 1B）
 *   每个 token 对应 8192 个元素，每个 block 处理一个 token
 *   1024 个线程，每个线程通过循环处理 8192 / 1024 = 8 个元素
 *
 * 线程映射：
 *   blockIdx.x  = 第几个 token
 *   threadIdx.x = FFN 中间层维度索引（0~1023）
 *   每个线程处理 8 个元素（workIdx, workIdx+1024, ..., workIdx+7168）
 *
 * @param a gate 投影输出 [N, 8192]，bfloat16，原地修改为 SiLU(a) ⊙ b
 * @param b up 投影输出 [N, 8192]，bfloat16，只读
 */
__global__ void siluKernel(nv_bfloat16* a, nv_bfloat16* b)
{
    // workIdx: 当前线程处理的第一个元素的线性索引
    // blockIdx.x * 8192: 跳过前面 token 的所有元素（每个 token 8192 维）
    // threadIdx.x: 当前线程在 block 内的偏移（0~1023）
    int workIdx = threadIdx.x + blockIdx.x * 8192;

    // 每个线程处理 8 个元素，步长 1024（线程数），保证合并访存
    // 例如 threadIdx.x=0 处理索引 0, 1024, 2048, 3072, 4096, 5120, 6144, 7168
    for (int i = 0; i < 8192; i += 1024)
    {
        // SiLU(a[i]) = a[i] * sigmoid(a[i]) = a[i] * (1 / (1 + exp(-a[i])))
        // 再乘以 b[i] 得到门控激活结果
        a[workIdx + i] =
            (nv_bfloat16) ((float) a[workIdx + i] * (1 / (1 + expf(-(float) a[workIdx + i]))) *
                           (float) b[workIdx + i]);
    }
}

/**
 * @brief siluKernel 的包装函数
 *
 * @param a          gate 投影输出 [N, 8192]，bfloat16，原地修改
 * @param b          up 投影输出 [N, 8192]，bfloat16，只读
 * @param num_tokens token 数量 N（即 grid 中的 block 数量）
 */
void silu(nv_bfloat16* a, nv_bfloat16* b, int num_tokens)
{
    siluKernel<<<num_tokens, 1024>>>(a, b);
}

// =============================================================================
//                                  Decode
// =============================================================================

__global__ void embeddingGatherKernelDecode(int* gpu_last_tokens,
                                            int num_tokens,
                                            nv_bfloat16* output,
                                            nv_bfloat16* embed_tokens)
{
    int input_token = gpu_last_tokens[blockIdx.x];
    int workIdx = blockIdx.x * 2048 + threadIdx.x;
    if (workIdx < num_tokens * 2048)
    {
        output[workIdx] = embed_tokens[input_token * 2048 + threadIdx.x];
        output[workIdx + 1024] = embed_tokens[input_token * 2048 + threadIdx.x + 1024];
    }
}

void embeddingGatherDecode(int* gpu_last_tokens,
                           int num_tokens,
                           nv_bfloat16* output,
                           nv_bfloat16* embed_tokens)
{
    embeddingGatherKernelDecode<<<num_tokens, 1024>>>(gpu_last_tokens, num_tokens, output,
                                                      embed_tokens);
#ifdef DEBUG
    cudaError error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
    }
#endif
}

__global__ void ropeKernelDecode(nv_bfloat16* input, int position_in_sequence, int proj_dim)
{
    if (2 * threadIdx.x + 1 < proj_dim) // TODO: check correctness
    {
        // TODO: precompute thetas, angles and perhaps sin/cos vals and reuse it across all kernel
        // invocations
        int head_idx = 2 * (threadIdx.x % 32);
        float theta = 1.0 / (pow(500000.0, ((float) head_idx / HEAD_DIM)));
        float angle = position_in_sequence * theta;
        nv_bfloat16 prev_2i = input[2 * threadIdx.x];
        nv_bfloat16 prev_2j = input[2 * threadIdx.x + 1];
        input[2 * threadIdx.x] =
            (nv_bfloat16) ((float) prev_2i * cos(angle) - (float) prev_2j * sin(angle));
        input[2 * threadIdx.x + 1] =
            (nv_bfloat16) ((float) prev_2i * sin(angle) + (float) prev_2j * cos(angle));
    }
}

// proj_dim: q_proj 2048, k_proj 512
// num_threads: I want to use it for both q_proj and k_proj so need to parameterize num_threads
// (1024 for q_proj and 512 for k_proj)
void ropeDecode(nv_bfloat16* input, int position_in_sequence, int proj_dim)
{
    int num_threads = proj_dim / 2;
    if (num_threads > 1024)
    {
        std::cout << "Can't launch more than 1024 threads on RTX 5090, RoPE kernel not launched";
        return;
    }

    ropeKernelDecode<<<1, num_threads>>>(input, position_in_sequence, proj_dim);
#ifdef DEBUG
    cudaError error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
    }
#endif
}

// Decode 版 rotate_half：单 token（无行偏移），配对通道 j 与 j+32
__global__ void
ropeKernelDecodeRotateHalf(nv_bfloat16* input, int position_in_sequence, int proj_dim)
{
    int h = threadIdx.x / 32;
    int j = threadIdx.x % 32;
    int i0 = h * HEAD_DIM + j;  // 通道 j
    int i1 = i0 + HEAD_DIM / 2; // 通道 j + 32

    if (i1 < proj_dim)
    {
        float theta = llama3ScaledTheta(j);
        float angle = position_in_sequence * theta;
        float x0 = input[i0];
        float x1 = input[i1];
        input[i0] = (nv_bfloat16) (x0 * cos(angle) - x1 * sin(angle));
        input[i1] = (nv_bfloat16) (x1 * cos(angle) + x0 * sin(angle));
    }
}

void ropeDecodeRotateHalf(nv_bfloat16* input, int position_in_sequence, int proj_dim)
{
    int num_threads = proj_dim / 2;
    if (num_threads > 1024)
    {
        std::cout << "Can't launch more than 1024 threads on RTX 5090, RoPE kernel not launched";
        return;
    }

    ropeKernelDecodeRotateHalf<<<1, num_threads>>>(input, position_in_sequence, proj_dim);
#ifdef DEBUG
    cudaError error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
    }
#endif
}

// seq_len increases by 1 with every new token
__global__ void softmaxKernelDecode(nv_bfloat16* input, int seq_len)
{
    // 使用固定大小的共享内存存储一行数据，用于树形归约求最大值和总和
    __shared__ float row[1024]; // row[0] 最终存储最大值，之后复用为求和结果
    __shared__ float max_val;

    // 第一步：找出每行的最大值 max_val，用于数值稳定的 softmax（防止 exp 溢出）
    int workIdx = blockIdx.x * MAX_SEQ_LEN + threadIdx.x;
    nv_bfloat16 attn_score = input[workIdx]; // 当前线程负责的注意力分数
    row[threadIdx.x] = (float) attn_score;
    __syncthreads();

    for (int i = 1; i < seq_len; i = i * 2)
    {
        if (threadIdx.x % (i * 2) == 0 && threadIdx.x + i < seq_len)
        {
            row[threadIdx.x] = fmaxf(row[threadIdx.x], row[threadIdx.x + i]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        max_val = row[0]; // so I don't need to allocate another shared value for max_val
    }
    __syncthreads();

    // 第二步：计算 exp(attn_score - max_val)
    // 在指数中减去最大值是数值稳定技巧：exp(x - max) 不会溢出，且 softmax 结果不变
    row[threadIdx.x] = expf((float) attn_score - max_val);
    __syncthreads();

    // 第三步：树形归约求 exp 之和（分母），复用 row 共享内存
    for (int i = 1; i < seq_len; i = i * 2)
    {
        if (threadIdx.x % (i * 2) == 0 && threadIdx.x + i < seq_len)
        {
            row[threadIdx.x] = row[threadIdx.x] + row[threadIdx.x + i];
        }
        __syncthreads();
    }

    // 第四步：softmax = exp(attn_score - max_val) / sum(exp)
    input[workIdx] = (nv_bfloat16) (expf((float) attn_score - max_val) / row[0]);
}

/**
 * @brief softmaxKernelDecode 的包装函数（Decode 阶段专用）
 *
 * @param input   经过因果掩码后的注意力分数矩阵 [NUM_Q_HEADS, MAX_SEQ_LEN]，原地修改，bfloat16
 * @param seq_len 当前序列长度（随每个新 token 递增）
 */
void softmaxDecode(nv_bfloat16* input, int seq_len)
{
    if (seq_len > 1024)
    {
        std::cout << "Can't launch more than 1024 threads on RTX 5090, Softmax kernel not launched";
        return;
    }

    softmaxKernelDecode<<<NUM_Q_HEADS, seq_len>>>(input, seq_len);
#ifdef DEBUG
    cudaError error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
    }
#endif
}

/**
 * @brief Paged Attention：在分页 KV Cache 上执行解码阶段的注意力计算
 *
 * 本 kernel 实现了 vLLM 风格的 PagedAttention，用于 Decode 阶段（每次生成一个 token）。
 * 核心思想：将 KV Cache 组织为固定大小的物理块（physical blocks），通过 block table
 * 将逻辑块映射到物理块，避免内存碎片和不必要的重分配。
 *
 * 算法概述（Online Softmax / FlashAttention 风格）：
 *   对于每个 Q 向量，逐 token 遍历 KV Cache 中的历史 K/V：
 *   1. 计算 q · k 点积（注意力分数）
 *   2. 使用 Online Softmax 维护 running max 和 running sum，避免两次遍历
 *   3. 累积加权 V 向量：acc += softmax_weight * v
 *   4. 最终输出：output = acc / d（d 为归一化分母）
 *
 * 数学公式：
 *   attention(q, K, V) = softmax(q @ K^T / sqrt(d_k)) @ V
 *                      = Σ_i [ exp(q·k_i - max) / Σ_j exp(q·k_j - max) ] * v_i
 *
 * 线程映射（二维 grid）：
 *   blockIdx.x  = active_slot 索引（第几个正在生成的序列）
 *   blockIdx.y  = q_head_id（第几个 Query 注意力头，0 ~ NUM_Q_HEADS-1）
 *   threadIdx.x = HEAD_DIM 维度索引（0 ~ 63），每个线程负责 Q/K/V 的一个维度
 *
 *   每个 block 有 HEAD_DIM=64 个线程，处理一个 (sequence, q_head) 对
 *   两个 warp（各 32 线程）分别计算点积的一半，通过共享内存合并
 *
 * GQA (Grouped Query Attention) 支持：
 *   kv_head_idx = q_head_id / GQA_Q_TO_K_RATIO
 *   多个 Q head 共享同一个 KV head（LLaMA 3.2: 32 Q heads / 8 KV heads = ratio 4）
 *
 * 参考：Online Softmax - https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf
 *
 * @param layer           当前 Transformer 层索引（0 ~ N_LAYERS-1），用于定位 KV Cache 中的层
 * @param num_active_slots 当前正在生成的序列数量（grid 的 x 维度大小）
 * @param q_proj          Query 投影输出 [num_active_slots, EMBEDDING_LENGTH]，bfloat16
 * @param kv_cache        分页 KV Cache，物理块组织的连续内存，bfloat16
 * @param block_table_gpu 块表 [num_slots, N_LAYERS, MAX_BLOCKS_PER_SEQ]，逻辑块→物理块映射
 * @param gpu_seq_lens    每个序列的当前长度 [num_active_slots]
 * @param gpu_active_slots 活跃序列的 slot 索引 [num_active_slots]
 * @param output          注意力输出 [num_active_slots, EMBEDDING_LENGTH]，bfloat16
 */
__global__ void pagedAttentionKernel(int layer,
                                     int num_active_slots,
                                     nv_bfloat16* q_proj,
                                     nv_bfloat16* kv_cache,
                                     int* block_table_gpu,
                                     int* gpu_seq_lens,
                                     int* gpu_active_slots,
                                     nv_bfloat16* output)
{
    __shared__ float dot_products[2];
    int active_slot = blockIdx.x; // active_slot == seq_id
    int slot = gpu_active_slots[active_slot];
    int q_head_id = blockIdx.y;
    int thread_id = threadIdx.x;
    int kv_head_idx = q_head_id / GQA_Q_TO_K_RATIO;
    nv_bfloat16 q = q_proj[active_slot * EMBEDDING_LENGTH + q_head_id * HEAD_DIM + thread_id];
    int seq_len = gpu_seq_lens[active_slot];
    int num_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // for online softmax https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf
    float current_max = -CUDART_INF_F;
    float acc = 0.0f;
    float d = 0.0f; // denominator, same name as in paper above

    for (int logical_block_idx = 0; logical_block_idx < num_blocks; ++logical_block_idx)
    {
        int physical_block = block_table_gpu[slot * N_LAYERS * MAX_BLOCKS_PER_SEQ +
                                             layer * MAX_BLOCKS_PER_SEQ + logical_block_idx];
        int tokens_in_block = min(seq_len - logical_block_idx * BLOCK_SIZE, BLOCK_SIZE);
        for (int token = 0; token < tokens_in_block; ++token)
        {
            nv_bfloat16* k = (nv_bfloat16*) ((char*) kv_cache + physical_block * BLOCK_BYTES +
                                             token * KV_DIM * sizeof(nv_bfloat16) +
                                             kv_head_idx * HEAD_DIM * sizeof(nv_bfloat16) +
                                             thread_id * sizeof(nv_bfloat16));
            nv_bfloat16* v = (nv_bfloat16*) ((char*) kv_cache + physical_block * BLOCK_BYTES +
                                             V_OFFSET + token * KV_DIM * sizeof(nv_bfloat16) +
                                             kv_head_idx * HEAD_DIM * sizeof(nv_bfloat16) +
                                             thread_id * sizeof(nv_bfloat16));
            float qk = (float) q * (float) *k;
            // tree reduction within current warp, thread 0 gets sum of all 32 elements within warp
            // could be done with __syncthreads but accessing memory of other threads in warp is op
            qk += __shfl_down_sync(0xffffffff, qk, 16);
            qk += __shfl_down_sync(0xffffffff, qk, 8);
            qk += __shfl_down_sync(0xffffffff, qk, 4);
            qk += __shfl_down_sync(0xffffffff, qk, 2);
            qk += __shfl_down_sync(0xffffffff, qk, 1);
            if (thread_id == 0)
            {
                dot_products[0] = qk;
            }
            if (thread_id == 32)
            {
                dot_products[1] = qk;
            }
            __syncthreads();
            if (thread_id == 0)
            {
                dot_products[0] = (dot_products[0] + dot_products[1]) / SQRT_HEAD_DIM;
            }
            __syncthreads();
            float dot_product = dot_products[0];
            // online softmax
            float new_max = current_max;
            if (dot_product > current_max)
            {
                new_max = dot_product;
            }
            float correction_factor = expf(current_max - new_max);
            current_max = new_max;
            float exp_score = expf(dot_product - current_max);
            d = d * correction_factor + exp_score;
            acc = acc * correction_factor + exp_score * (float) *v;
            __syncthreads();
        }
    }
    output[active_slot * EMBEDDING_LENGTH + q_head_id * HEAD_DIM + thread_id] = acc / d;
}

void pagedAttention(int layer,
                    int num_active_slots,
                    nv_bfloat16* q_proj,
                    nv_bfloat16* kv_cache,
                    int* block_table_gpu,
                    int* gpu_seq_lens,
                    int* gpu_active_slots,
                    nv_bfloat16* output)
{
    pagedAttentionKernel<<<dim3(num_active_slots, NUM_Q_HEADS), HEAD_DIM>>>(
        layer, num_active_slots, q_proj, kv_cache, block_table_gpu, gpu_seq_lens, gpu_active_slots,
        output);
}
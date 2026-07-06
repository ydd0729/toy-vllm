// 禁用 nlohmann/json 库的隐式类型转换。
// 设置后，json 对象不能自动转换为基本类型（如 int、string），
// 必须显式调用 .get<T>()，避免意外的类型转换错误。
#define JSON_USE_IMPLICIT_CONVERSIONS 0

#include <iostream>
#include <numeric>
#include <fstream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <queue>
#include <string>
#include <vector>
#include <driver_types.h>
#include <nlohmann/json.hpp>
#include "kernels.cuh"
#include "config.hpp"
#include "inference.hpp"
#include "weights.hpp"
#include "cuda_utils.hpp"
#include "utils.hpp"

using json = nlohmann::json;
using std::string;
using std::vector;

/**
 * @brief Prefill 阶段：一次性处理整个 prompt，填充 KV Cache 并生成第一个 token
 *
 * LLM 推理分为两个阶段：
 *   1. Prefill（本函数）：并行处理 N 个输入 token，填充 KV Cache，输出第一个生成 token
 *   2. Decode（main 中的 while 循环）：逐 token 生成，每次只处理 1 个新 token
 *
 * 流程：
 *   embedding -> [RMSNorm -> QKV proj -> RoPE -> KV cache scatter -> Attention -> O proj ->
 * residual
 *                 -> RMSNorm -> MLP(gate/up/SiLU/down) -> residual] x N_LAYERS
 *             -> RMSNorm -> logits -> argmax -> 第一个生成 token
 *
 * 注意：参数列表较长是因为 GPU 缓冲区、cuBLAS 句柄和各种临时变量都从 main 传入以复用内存。
 *
 * @param prompt_queue        [输入] 等待处理的 prompt 队列
 * @param slot                [输入] 当前 prompt 分配到的 batch slot 索引
 * @param weights             [输入] 模型权重（各层 GPU 指针）
 */
void prefill(std::queue<vector<int>>& prompt_queue,
             int slot,
             InferenceContext& ctx,
             PagedKVCache& pkv,
             BatchState& bs,
             Weights& weights)
{
    // 获取当前 prompt
    auto prompt = prompt_queue.front();
    prompt_queue.pop();

    // 标记当前 batch slot 为已占用，防止其他 prompt 被分配到同一 slot
    bs.is_slot_free[slot] = false;
    // 记录该 slot 装的是第几个请求（对应 PROMPT N），用于把输出对应回提问
    bs.request_id[slot] = bs.next_request_id++;

    ctx.setInputTokens(prompt);
    ctx.copyInputTokensToGPU();

    // 对 prompt 中的每个 token 获取 embedding ，并保存到 input_embeddings 中
    embeddingGather(ctx.input_tokens, ctx.input_embeddings, weights.embed_tokens,
                    ctx.input_token_len);

    ctx.InitHiddenState();

    for (int layer = 0; layer < N_LAYERS; ++layer)
    {
        // 每层开始前的 RMSNorm：对 hidden_state 做归一化，输出到 rms_norms，供后续 QKV 投影使用
        rmsNorm(ctx.hidden_state, ctx.normed_hidden, weights.input_layernorm[layer],
                ctx.input_token_len);

        // Q 投影计算： Q = inputs * q_proj^T
        //   - q_proj：查询权重矩阵 (query weight)，模型参数之一，形状为 (EMBEDDING_LENGTH,
        //   EMBEDDING_LENGTH)
        //   - inputs：输入隐藏状态，形状为 (num_tok, EMBEDDING_LENGTH)
        //   - Q：查询向量 (query)，注意力机制的三大投影之一，形状为 (num_tok, EMBEDDING_LENGTH)
        ctx.qProj(weights.q_proj[layer]);

        // K 投影计算：K = inputs * k_proj^T
        //   - k_proj：键权重矩阵 (key weight)，形状为 (KV_DIM, EMBEDDING_LENGTH)
        //         GQA/MQA 下 KV 头数少于 Q 头，故 KV_DIM < EMBEDDING_LENGTH
        //   - inputs：输入隐藏状态，形状为 (num_tok, EMBEDDING_LENGTH)
        //   - K：键向量，形状为 (num_tok, KV_DIM)
        ctx.kProj(weights.k_proj[layer]);

        // V 投影计算：V = inputs * v_proj^T，与 K 投影类似
        ctx.vProj(weights.v_proj[layer]);

        // RoPE

        rope(ctx.query_states, ctx.input_token_len, EMBEDDING_LENGTH);
        rope(ctx.key_states, ctx.input_token_len, KV_DIM);

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

        ctx.oProj(weights.o_proj[layer]);

        // 残差连接：hidden_state += attn_output
        // (num_tok, 2048) + (num_tok, 2048) -> (num_tok, 2048)
        residualAdd(ctx.hidden_state, ctx.attn_output, ctx.input_token_len);

        // post attention RMS Norm
        rmsNorm(ctx.hidden_state, ctx.normed_hidden, weights.post_attn_layernorms[layer],
                ctx.input_token_len);

        ctx.swiGLU(weights.gate_proj[layer], weights.up_proj[layer], weights.down_proj[layer]);

        // (num_tok, 2048) + (num_tok, 2048) -> (num_tok, 2048)
        residualAdd(ctx.hidden_state, ctx.down, ctx.input_token_len);
    }
    rmsNorm(ctx.hidden_state, ctx.normed_hidden, weights.norm, ctx.input_token_len);

    ctx.lmHead(weights.embed_tokens);

    // argmax to get the output token
    // TODO: move argmax to GPU and get rid of these CPU<->GPU tokens moves
    cudaMemcpy(ctx.logits_cpu.data(), ctx.logits,
               sizeof(__nv_bfloat16) * ctx.input_token_len * VOCAB_SIZE, cudaMemcpyDeviceToHost);
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
    cudaMemcpy(pkv.block_table_gpu, pkv.block_table.data(),
               MAX_SEQUENCES * N_LAYERS * MAX_BLOCKS_PER_SEQ * sizeof(int), cudaMemcpyHostToDevice);
}

int main(int argc, char* argv[])
{
    if (checkGPUStatus() != 0)
    {
        return 1;
    }

    Weights weights;

    // TODO: 改为可输入
    // PROMPT 0 (What is 2+2?) - length 17
    std::queue<vector<int>> prompt_queue;
    prompt_queue.push({128000, 128006, 882, 128007, 271, 3923, 374, 220, 17, 10, 17, 30, 128009,
                       128006, 78191, 128007, 271});

    // PROMPT 1 (Name a color.) - length 14
    prompt_queue.push(
        {128000, 128006, 882, 128007, 271, 678, 264, 1933, 13, 128009, 128006, 78191, 128007, 271});

    // PROMPT 2 (Say hello.) - length 13
    prompt_queue.push(
        {128000, 128006, 882, 128007, 271, 46864, 24748, 13, 128009, 128006, 78191, 128007, 271});

    // PROMPT 3 (Capital of France?) - length 14
    prompt_queue.push({128000, 128006, 882, 128007, 271, 64693, 315, 9822, 30, 128009, 128006,
                       78191, 128007, 271});

    BatchState bs;
    PagedKVCache pkv;
    InferenceContext ctx;

    // DECODE 阶段：每步只处理 1 个新 token。
    // 逐 token 的激活缓冲（hidden、Q、normed、gate、up 等）都只用 index 0（当前这一个 token）；
    // 唯独新算出的 K、V 要写到该 token 在序列中的真实位置（current_position_token），
    // 追加进 paged KV cache，供注意力从位置 0 读到当前位置。

    while (true)
    {
        bs.active_slots.clear();
        bs.active_tokens.clear();

        for (int slot = 0; slot < BATCH_SIZE; ++slot)
        {
            if (bs.is_slot_free[slot])
            {
                if (prompt_queue.empty())
                {
                    continue;
                }
                bs.generated_tokens[slot].clear();
                prefill(prompt_queue, slot, ctx, pkv, bs, weights);
            }
            bs.active_slots.push_back(slot);
            bs.active_tokens.push_back(bs.last_generated_tokens[slot]);
        }
        int num_active_slots = bs.active_slots.size();
        if (num_active_slots == 0)
        {
            if (prompt_queue.empty())
            {
                break;
            }
            continue;
        }

        // copy useful data to gpu
        cudaMemcpy(ctx.last_tokens, bs.active_tokens.data(), num_active_slots * sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(ctx.active_slots, bs.active_slots.data(), num_active_slots * sizeof(int),
                   cudaMemcpyHostToDevice);

        vector<int> seq_lens(num_active_slots);

        for (int slot = 0; slot < num_active_slots; ++slot)
        {
            int active_slot = bs.active_slots[slot];
            seq_lens[slot] = bs.current_prompt_len[active_slot] + 1;
        }
        cudaMemcpy(ctx.seq_lens, seq_lens.data(), seq_lens.size() * sizeof(int),
                   cudaMemcpyHostToDevice);

        embeddingGatherDecode(ctx.last_tokens, num_active_slots, ctx.hidden_state,
                              weights.embed_tokens);

        // decode 每步处理 num_active_slots 行（每个活跃序列 1 个 token）
        ctx.num_rows = num_active_slots;

        for (int layer = 0; layer < N_LAYERS; ++layer)
        {
            rmsNorm(ctx.hidden_state, ctx.normed_hidden, weights.input_layernorm[layer],
                    num_active_slots);

            ctx.qProj(weights.q_proj[layer]);
            ctx.kProj(weights.k_proj[layer]);
            ctx.vProj(weights.v_proj[layer]);

            for (int slot = 0; slot < num_active_slots; ++slot)
            {
                int active_slot = bs.active_slots[slot];
                ropeDecode(&ctx.query_states[slot * EMBEDDING_LENGTH],
                           bs.current_prompt_len[active_slot], EMBEDDING_LENGTH);
                ropeDecode(ctx.key_states + slot * KV_DIM, bs.current_prompt_len[active_slot],
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
                int physical_block_idx =
                    pkv.getPhysicalBlock(active_slot, layer, logical_block_idx);
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

            // synchronize block table on cpu with block table on gpu (for attention)
            cudaMemcpy(pkv.block_table_gpu, pkv.block_table.data(),
                       MAX_SEQUENCES * N_LAYERS * MAX_BLOCKS_PER_SEQ * sizeof(int),
                       cudaMemcpyHostToDevice);

            // 注意力输出写入 ctx.context（= scratch_a），供 oProj 读取
            ctx.context = ctx.scratch_a;
            pagedAttention(layer, num_active_slots, ctx.query_states, pkv.kv_cache,
                           pkv.block_table_gpu, ctx.seq_lens, ctx.active_slots, ctx.context);

            ctx.oProj(weights.o_proj[layer]);

            residualAdd(ctx.hidden_state, ctx.attn_output, num_active_slots);

            rmsNorm(ctx.hidden_state, ctx.normed_hidden, weights.post_attn_layernorms[layer],
                    num_active_slots);

            ctx.swiGLU(weights.gate_proj[layer], weights.up_proj[layer], weights.down_proj[layer]);

            residualAdd(ctx.hidden_state, ctx.down, num_active_slots);
        }

        rmsNorm(ctx.hidden_state, ctx.normed_hidden, weights.norm, num_active_slots);

        ctx.lmHead(weights.embed_tokens);

        cudaMemcpy(ctx.logits_cpu.data(), ctx.logits,
                   sizeof(nv_bfloat16) * num_active_slots * VOCAB_SIZE, cudaMemcpyDeviceToHost);

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
                // 序列结束：打印该请求的完整生成序列（可丢给 tokenizer.py --decode --ids ... 看文字）
                std::cout << "=== request " << bs.request_id[active_slot]
                          << ": ";
                
                for (int t : bs.generated_tokens[active_slot])
                {
                    std::cout << t << " ";
                }
                std::cout << std::endl;

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
                cudaMemcpy(pkv.block_table_gpu, pkv.block_table.data(),
                           MAX_SEQUENCES * N_LAYERS * MAX_BLOCKS_PER_SEQ * sizeof(int),
                           cudaMemcpyHostToDevice);
            }
            else
            {
                bs.last_generated_tokens[active_slot] = max_token_idx;
                bs.generated_tokens[active_slot].push_back(max_token_idx);
                bs.current_prompt_len[active_slot] = bs.current_prompt_len[active_slot] + 1;
            }
        }
    }

    cudaDeviceSynchronize();
    return 0;
}

#include "weights.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <nlohmann/json.hpp>
#include <format>
#include <cuda_runtime.h>
#include "cuda_utils.hpp"

/**
 * @brief 从 model.safetensors 加载模型权重到 GPU
 *
 * safetensors 文件格式：
 *   [8 bytes header_size] [header_size bytes JSON header] [tensor data...]
 *   JSON header 中每个 tensor 名对应一个 data_offsets [start, end]，
 *   表示该 tensor 在数据区的字节偏移。
 *
 *
 * 流程：
 *   1. 读取 header，解析每个 tensor 的偏移量
 *   2. 一次性将所有 tensor 数据读入 CPU，再整体拷贝到 GPU
 *   3. 通过偏移量设置 Weights 结构体中各层权重的 GPU 指针
 */
Weights::Weights()
{
    std::ifstream safetensors_file("models/Llama-3.2-1B/model.safetensors", std::ios_base::binary);

    if (!safetensors_file.is_open())
    {
        std::cout << "Can't open model.safetensors file\n";
        safetensors_file.close();
        throw std::runtime_error("Can't open model.safetensors file");
    }

    // 前 8 字节是 header 长度（uint64_t，小端序）
    uint64_t header_size;
    safetensors_file.read(reinterpret_cast<char*>(&header_size), 8);

    // 读取 JSON header
    std::string header;
    header.resize(header_size);
    safetensors_file.read(header.data(), header_size);

    // 解析 header，提取每个 tensor 在数据区的起始偏移
    std::unordered_map<std::string, uint64_t> offsets;
    nlohmann::json header_json = nlohmann::json::parse(header);

    // 保存解析出来 header
    // std::ofstream out("header.json");
    // out << header_json.dump(4) << '\n';

    // 遍历所有 tensor，记录起始偏移，并找到数据区的最大偏移（即总数据大小）
    // 例子：
    // "model.layers.0.input_layernorm.weight": {
    //     "data_offsets": [
    //         525336576,
    //         525340672
    //     ],
    //     "dtype": "BF16",
    //     "shape": [
    //         2048
    //     ]
    // },
    uint64_t max_offset = 0;
    for (auto& [key, value] : header_json.items())
    {
        if (key == "__metadata__")
        {
            continue;
        }

        uint64_t offset_end = value["data_offsets"].at(1).get<uint64_t>();
        if (offset_end > max_offset)
        {
            max_offset = offset_end;
        }

        offsets[key] = value["data_offsets"].at(0).get<uint64_t>();
    }

    // 在 GPU 上分配一整块连续内存，存放所有权重
    CUDA_CHECK(cudaMalloc(&gpu_model_weights, max_offset));

    // 先读入 CPU 缓冲区，再整体拷贝到 GPU
    std::vector<char> model_weights_cpu;
    model_weights_cpu.resize(max_offset);
    safetensors_file.read(model_weights_cpu.data(), max_offset);

    CUDA_CHECK(cudaMemcpy(gpu_model_weights, model_weights_cpu.data(), max_offset,
                          cudaMemcpyHostToDevice));

    safetensors_file.close();

    // 辅助 lambda：根据 tensor 名返回其在 GPU 上的指针
    auto weight_offset = [this, &offsets](const std::string& s)
    {
        return (__nv_bfloat16*) (this->gpu_model_weights + offsets.at(s));
    };

    // 将各层权重的 GPU 指针填入 Weights 结构体
    embed_tokens = weight_offset("model.embed_tokens.weight");
    norm = weight_offset("model.norm.weight");
    for (int i = 0; i < N_LAYERS; ++i)
    {
        input_layernorm[i] =
            weight_offset(std::format("model.layers.{}.input_layernorm.weight", i));
        down_proj[i] = weight_offset(std::format("model.layers.{}.mlp.down_proj.weight", i));
        gate_proj[i] = weight_offset(std::format("model.layers.{}.mlp.gate_proj.weight", i));
        up_proj[i] = weight_offset(std::format("model.layers.{}.mlp.up_proj.weight", i));
        post_attn_layernorms[i] =
            weight_offset(std::format("model.layers.{}.post_attention_layernorm.weight", i));
        k_proj[i] = weight_offset(std::format("model.layers.{}.self_attn.k_proj.weight", i));
        o_proj[i] = weight_offset(std::format("model.layers.{}.self_attn.o_proj.weight", i));
        q_proj[i] = weight_offset(std::format("model.layers.{}.self_attn.q_proj.weight", i));
        v_proj[i] = weight_offset(std::format("model.layers.{}.self_attn.v_proj.weight", i));
    }
}

Weights::~Weights()
{
    CUDA_CHECK_NOTHROW(cudaFree(gpu_model_weights));
}
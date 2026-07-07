// 禁用 nlohmann/json 库的隐式类型转换。
// 设置后，json 对象不能自动转换为基本类型（如 int、string），
// 必须显式调用 .get<T>()，避免意外的类型转换错误。
#include "nlohmann/json_fwd.hpp"
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
#include <tokenizers_cpp.h>
#include <debug.hpp>

using tokenizers::Tokenizer;

std::string apply_chat_template(const std::string& user_msg)
{
    return "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
           + user_msg
           + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
}

int main(int argc, char* argv[])
{
    if (checkGPUStatus() != 0)
    {
        return 1;
    }

    std::ifstream tokenizer_file(llama3p2_1B_Instruct_tokenizer_path, std::ios_base::binary);

    if (!tokenizer_file.is_open())
    {
        throw std::runtime_error(std::format("Can't open {}", llama3p2_1B_Instruct_tokenizer_path.string()));
    }

    tokenizer_file.seekg(0, std::ios::end);
    std::streamsize size = tokenizer_file.tellg(); // 当前位置=文件长度
    tokenizer_file.seekg(0, std::ios::beg); // 回到开头再 read

    std::string blob;
    blob.resize(size);
    tokenizer_file.read(blob.data(), size);

    auto tok = Tokenizer::FromBlobJSON(blob);

    std::queue<std::vector<int>> prompt_queue;
    auto p1 = tok->Encode(apply_chat_template("What is 2+2?"));
    DUMP_VEC(p1);
    prompt_queue.push(p1);

    auto p2 = tok->Encode(apply_chat_template("Name a color."));
    DUMP_VEC(p2);
    prompt_queue.push(p2);

    auto p3 = tok->Encode(apply_chat_template("Say hello."));
    DUMP_VEC(p3);
    prompt_queue.push(p3);

    auto p4 = tok->Encode(apply_chat_template("Capital of France?"));
    DUMP_VEC(p4);
    prompt_queue.push(p4);


    Weights w;
    BatchState bs;
    PagedKVCache pkv;
    InferenceContext ctx;
    std::vector<RequestOutput> output;

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
                prefill(prompt_queue, slot, ctx, pkv, bs, w);
            }
            bs.active_slots.push_back(slot);
            bs.active_tokens.push_back(bs.last_generated_tokens[slot]);
        }

        if (bs.active_slots.size() == 0)
        {
            if (prompt_queue.empty())
            {
                break;
            }
            continue;
        }

        auto out = decode(ctx, pkv, bs, w);
        output.insert(output.end(), std::make_move_iterator(out.begin()),
                      std::make_move_iterator(out.end()));
    }

    sort(output.begin(), output.end(),
         [](const auto& a, const auto& b)
         {
             return a.request_id < b.request_id;
         });

    for (const auto& o : output)
    {
        std::cout << "Request " << o.request_id << ": ";
        std::cout << tok->Decode(o.output_token);
        std::cout << std::endl;
    }

    cudaDeviceSynchronize();
    return 0;
}

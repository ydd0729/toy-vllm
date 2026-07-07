// 禁用 nlohmann/json 库的隐式类型转换。
// 设置后，json 对象不能自动转换为基本类型（如 int、string），
// 必须显式调用 .get<T>()，避免意外的类型转换错误。
#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include <ios>
#include <ranges>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <queue>
#include <string>
#include <vector>
#include <driver_types.h>
#include <nlohmann/json.hpp>
#include "config.hpp"
#include "inference.hpp"
#include "weights.hpp"
#include "cuda_utils.hpp"
#include <CLI/CLI.hpp>
#include <filesystem>
#include <string_view>
#include <chrono>

int main(int argc, char* argv[])
{
    if (checkGPUStatus() != 0)
    {
        return 1;
    }

    CLI::App app;
    argv = app.ensure_utf8(argv);
    std::filesystem::path user_message_path;
    std::filesystem::path output_path;

    app.add_option("-f,--file", user_message_path)->required();
    app.add_option("-o,--output", output_path)->required();

    CLI11_PARSE(app, argc, argv);

    std::ifstream user_message_file(user_message_path, std::ios_base::binary);
    if (!user_message_file.is_open())
    {
        throw std::runtime_error(std::format("Can't open {}", user_message_path.string()));
    }

    std::string user_messages;
    std::streamsize size = std::filesystem::file_size(user_message_path);
    user_messages.resize(size);
    user_message_file.read(user_messages.data(), size);

    std::queue<std::string> input_queue;
    for (auto line_range : user_messages | std::views::split('\n'))
    {
        std::string_view line(line_range.begin(), line_range.end());
        if (!line.empty() && line.back() == '\r')
        {
            line.remove_suffix(1);
        }
        if (line.empty())
        {
            continue;
        }

        input_queue.emplace(line);
    }

    auto start = std::chrono::steady_clock::now();

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
                if (input_queue.empty())
                {
                    continue;
                }
                bs.generated_tokens[slot].clear();
                prefill(input_queue, slot, ctx, pkv, bs, w);
            }
            bs.active_slots.push_back(slot);
            bs.active_tokens.push_back(bs.last_generated_tokens[slot]);
        }

        if (bs.active_slots.size() == 0)
        {
            if (input_queue.empty())
            {
                break;
            }
            continue;
        }

        auto out = decode(ctx, pkv, bs, w);
        output.insert(output.end(), std::make_move_iterator(out.begin()),
                      std::make_move_iterator(out.end()));
    }

    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration<double>(end - start);
    std::cout << "Elapsed: " << elapsed.count() << " s" << std::endl;

    sort(output.begin(), output.end(),
         [](const auto& a, const auto& b)
         {
             return a.request_id < b.request_id;
         });

    std::ostringstream buffer;
    const std::string sep(44, '=');

    for (const auto& o : output)
    {
        buffer << sep << " Request " << std::setw(2) << o.request_id << " " << sep << "\n\n"
               << "Q: " << o.input_message << "\n"
               << "A: " << o.output_message << "\n\n";

        // std::cout << "  Q Tokens: ";
        // for (const auto t : o.input_token)
        // {
        //     std::cout << t << " ";
        // }

        // std::cout << "\n  A Tokens: ";
        // for (const auto t : o.output_token)
        // {
        //     std::cout << t << " ";
        // }
    }

    std::ofstream out(output_path, std::ios::binary);
    out.exceptions(std::ios::failbit | std::ios::badbit);
    std::string_view content = buffer.view();
    out.write(content.data(), static_cast<std::streamsize>(content.size()));

    cudaDeviceSynchronize();
    return 0;
}

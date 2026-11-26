#pragma once

#include "ggml-cpp.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

struct expert_cache_fixture {
    std::filesystem::path directory;
    std::string           path;

    explicit expert_cache_fixture(bool      nonuniform        = false,
                                  uint32_t  vocab_size        = 32,
                                  ggml_type expert_type       = GGML_TYPE_F32,
                                  uint32_t  expert_width      = 64,
                                  int       scale_granularity = -1,
                                  ggml_type router_type       = GGML_TYPE_F32) {
        if (vocab_size < 32) {
            throw std::invalid_argument("expert-cache fixture needs at least 32 tokens");
        }
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        for (int attempt = 0;; ++attempt) {
            directory = std::filesystem::temp_directory_path() /
                        ("llama-cache-test-" + std::to_string(stamp) + "-" + std::to_string(attempt));
            if (std::filesystem::create_directory(directory)) {
                break;
            }
            if (attempt == 100) {
                throw std::runtime_error("could not create expert-cache fixture directory");
            }
        }
        path = (directory / "model.gguf").string();
        gguf_context_ptr file(gguf_init_empty());
        ggml_context_ptr tensors(
            ggml_init({ 1024 * 1024 + (size_t) (vocab_size - 32) * 32 * 8 + (size_t) expert_width * 32 * 8 * 4 * 6,
                        nullptr, false }));
        if (!file || !tensors) {
            throw std::runtime_error("could not allocate expert-cache fixture");
        }
        gguf_set_val_str(file.get(), "general.architecture", "llama");
        gguf_set_val_str(file.get(), "general.name", "expert-cache-test");
        gguf_set_val_str(file.get(), "tokenizer.ggml.model", "none");
        gguf_set_val_u32(file.get(), "llama.vocab_size", vocab_size);
        gguf_set_val_u32(file.get(), "llama.context_length", 64);
        gguf_set_val_u32(file.get(), "llama.embedding_length", 32);
        gguf_set_val_u32(file.get(), "llama.block_count", 2);
        const uint32_t widths[] = { expert_width, expert_width };
        gguf_set_arr_data(file.get(), "llama.feed_forward_length", GGUF_TYPE_UINT32, widths, 2);
        gguf_set_val_u32(file.get(), "llama.attention.head_count", 2);
        gguf_set_val_u32(file.get(), "llama.attention.head_count_kv", 2);
        gguf_set_val_u32(file.get(), "llama.rope.dimension_count", 16);
        gguf_set_val_f32(file.get(), "llama.attention.layer_norm_rms_epsilon", 1e-5f);
        gguf_set_val_u32(file.get(), "llama.expert_count", 8);
        gguf_set_val_u32(file.get(), "llama.expert_used_count", 2);

        auto add = [&](const std::string & name, int64_t n0, int64_t n1 = 1, int64_t n2 = 1) {
            const bool half   = nonuniform && name.find("blk.1.") == 0 && name.find("_exps.") != std::string::npos;
            const bool expert = name.find("_exps.") != std::string::npos;
            const bool router = name.find("ffn_gate_inp.weight") != std::string::npos;
            const auto type   = half ? GGML_TYPE_F16 : (expert ? expert_type : (router ? router_type : GGML_TYPE_F32));
            auto *     tensor = ggml_new_tensor_3d(tensors.get(), type, n0, n1, n2);
            ggml_set_name(tensor, name.c_str());
            const bool         norm = name.find("norm.weight") != std::string::npos;
            std::vector<float> values(ggml_nelements(tensor));
            for (int64_t i = 0; i < ggml_nelements(tensor); ++i) {
                const float value = norm ? 1.0f : 0.1f * std::sin(float(i + name.size()));
                values[i]         = value;
            }
            ggml_quantize_chunk(type, values.data(), tensor->data, 0, n1 * n2, n0, nullptr);
            gguf_add_tensor(file.get(), tensor);
            if ((expert && scale_granularity >= 0) || ggml_needs_scale(type)) {
                const int     granularity = router ? 2 : std::max(0, scale_granularity);
                ggml_tensor * scale       = granularity == 1 ?
                                                ggml_new_tensor_1d(tensors.get(), GGML_TYPE_F32, n2) :
                                                ggml_new_tensor_2d(tensors.get(), GGML_TYPE_F32, granularity == 2 ? n1 : 1,
                                                             router || granularity == 0 ? 1 : n2);
                const auto    scale_name  = name.substr(0, name.size() - 7) + ".scale";
                ggml_set_name(scale, scale_name.c_str());
                for (int64_t i = 0; i < ggml_nelements(scale); ++i) {
                    ((float *) scale->data)[i] = 0.75f + 0.0625f * (float) (i % 8);
                }
                gguf_add_tensor(file.get(), scale);
            }
        };
        add("token_embd.weight", 32, vocab_size);
        add("output_norm.weight", 32);
        add("output.weight", 32, vocab_size);
        for (int layer = 0; layer < 2; ++layer) {
            const std::string prefix = "blk." + std::to_string(layer) + ".";
            add(prefix + "attn_norm.weight", 32);
            for (const char * kind : { "q", "k", "v", "output" }) {
                add(prefix + "attn_" + kind + ".weight", 32, 32);
            }
            add(prefix + "ffn_norm.weight", 32);
            add(prefix + "ffn_gate_inp.weight", 32, 8);
            add(prefix + "ffn_gate_exps.weight", 32, widths[layer], 8);
            add(prefix + "ffn_up_exps.weight", 32, widths[layer], 8);
            add(prefix + "ffn_down_exps.weight", widths[layer], 32, 8);
        }
        if (!gguf_write_to_file(file.get(), path.c_str(), false)) {
            throw std::runtime_error("could not write expert-cache fixture");
        }
    }

    ~expert_cache_fixture() {
        std::error_code ignored;
        std::filesystem::remove_all(directory, ignored);
    }

    expert_cache_fixture(const expert_cache_fixture &)             = delete;
    expert_cache_fixture & operator=(const expert_cache_fixture &) = delete;
};

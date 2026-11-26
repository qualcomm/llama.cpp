#include "llama-expert-cache-model.h"

#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"
#include "llama-arch.h"
#include "llama-expert-cache.h"
#include "llama-ext.h"
#include "llama-hparams.h"
#include "llama-impl.h"
#include "llama-model-loader.h"
#include "llama-model.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

static void llama_apply_expert_cache_env(llama_model_params & params) {
    if (params.expert_cache_size_bytes == 0) {
        const char * value             = std::getenv("LLAMA_EXPERT_CACHE_SIZE");
        params.expert_cache_size_bytes = value ? std::stoull(value) : 0;
    }
    if (const char * value = std::getenv("LLAMA_EXPERT_CACHE_THRESHOLD")) {
        params.expert_cache_threshold = std::stof(value);
    }
    if (const char * value = std::getenv("LLAMA_EXPERT_CACHE_MODE")) {
        if (std::strcmp(value, "auto") == 0) {
            params.expert_cache_mode = LLAMA_EXPERT_CACHE_MODE_AUTO;
        } else if (std::strcmp(value, "global") == 0) {
            params.expert_cache_mode = LLAMA_EXPERT_CACHE_MODE_GLOBAL;
        } else if (std::strcmp(value, "global-nonuniform") == 0) {
            params.expert_cache_mode = LLAMA_EXPERT_CACHE_MODE_GLOBAL_NONUNIFORM;
        } else if (std::strcmp(value, "layered") == 0) {
            params.expert_cache_mode = LLAMA_EXPERT_CACHE_MODE_LAYERED;
        } else {
            throw std::invalid_argument("LLAMA_EXPERT_CACHE_MODE must be auto, global, global-nonuniform, or layered");
        }
    }
    if (const char * value = std::getenv("LLAMA_EXPERT_CACHE_POLICY")) {
        if (std::strcmp(value, "lru") == 0) {
            params.expert_cache_policy = LLAMA_EXPERT_CACHE_POLICY_LRU;
        } else if (std::strcmp(value, "lfu") == 0) {
            params.expert_cache_policy = LLAMA_EXPERT_CACHE_POLICY_LFU;
        } else {
            throw std::invalid_argument("LLAMA_EXPERT_CACHE_POLICY must be lru or lfu");
        }
    }
}

static bool expert_layouts_uniform(const llama_model_loader & loader) {
    std::map<std::string, std::map<std::string, std::array<int64_t, 4>>> layouts;
    for (const auto & [name, weight] : loader.weights_map) {
        if (!is_expert_weight_tensor(name.c_str())) {
            continue;
        }
        const char * block = strstr(name.c_str(), "blk.");
        const char * kind  = block ? strchr(block + 4, '.') : nullptr;
        if (!kind) {
            continue;
        }
        const auto * t                          = weight.tensor;
        layouts[std::string(block, kind)][kind] = { t->type, t->ne[0], t->ne[1], (int64_t) t->nb[2] };
    }
    if (layouts.empty()) {
        return true;
    }
    for (const auto & layer : layouts) {
        if (layer.second != layouts.begin()->second) {
            return false;
        }
    }
    return true;
}

void llama_expert_cache_prepare(llama_model_loader & loader, llama_model_params & params) {
    if (!params.expert_cache_enabled) {
        return;
    }
    llama_apply_expert_cache_env(params);
    if (params.expert_cache_enabled && params.expert_cache_mode == LLAMA_EXPERT_CACHE_MODE_GLOBAL &&
        !expert_layouts_uniform(loader)) {
        LLAMA_LOG_WARN(
            "%s: expert cache disabled: global mode requires matching expert layouts (use auto or layered)\n",
            __func__);
        params.expert_cache_enabled = false;
    }
    loader.expert_cache_enabled = params.expert_cache_enabled;
}

std::unique_ptr<llama_expert_cache> llama_expert_cache_init(llama_model &              model,
                                                            llama_model_loader &       ml,
                                                            const llama_model_params & mparams) {
    auto result        = std::make_unique<llama_expert_cache>();
    result->no_alloc_  = mparams.no_alloc;
    result->automatic_ = mparams.expert_cache_size_bytes == 0;
    if (!arch_supports_expert_cache(model.arch)) {
        LLAMA_LOG_WARN("%s: expert cache not yet validated for arch '%s', disabling\n", __func__,
                       llm_arch_name(model.arch));
        return nullptr;
    }

    const auto &   hparams  = model.hparams;
    const uint32_t n_layer  = hparams.n_layer();
    const uint32_t n_expert = hparams.n_expert;

    auto slice_info = [&](const ggml_tensor * t) -> llama_expert_cache::expert_slice_loc {
        if (!t) {
            return { 0, 0, 0 };
        }
        const auto * w = ml.get_weight(t->name);
        if (!w || w->idx >= ml.files.size()) {
            throw std::runtime_error(format("expert tensor '%s' has no valid GGUF source", t->name));
        }
        return { w->offs, (size_t) t->nb[2], w->idx };
    };

    auto get_ne = [](const ggml_tensor * t) -> std::pair<int64_t, int64_t> {
        return t ? std::make_pair((int64_t) t->ne[0], (int64_t) t->ne[1]) : std::make_pair((int64_t) 0, (int64_t) 0);
    };

    // Expert slices need 32-byte alignment for backend buffers.
    auto check_align = [&](size_t stride, const char * tname) -> bool {
        if (stride == 0) {
            return true;
        }
        if (stride % 32 != 0) {
            LLAMA_LOG_WARN(
                "%s: expert cache disabled: tensor '%s' slice size %zu "
                "is not 32-byte aligned\n",
                __func__, tname, stride);
            return false;
        }
        return true;
    };

    constexpr int NK = llama_expert_cache::N_KINDS;

    std::vector<llama_expert_cache::expert_slice_loc> locs;
    locs.reserve((size_t) n_layer * NK * n_expert);

    std::vector<size_t> layer_expert_sz(n_layer, 0);
    uint32_t            n_expert_used = 1;

    // indexed as [layer * NK + kind]
    std::vector<size_t>                      layer_kind_size(n_layer * NK, 0);
    std::vector<std::pair<int64_t, int64_t>> layer_kind_ne(n_layer * NK, { 0, 0 });
    std::vector<ggml_type>                   layer_kind_dtype(n_layer * NK, GGML_TYPE_F32);

    for (uint32_t il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        auto [gate_up_base, gate_up_stride, gate_up_file] = slice_info(layer.ffn_gate_up_exps);
        auto [gate_base, gate_stride, gate_file]          = slice_info(layer.ffn_gate_exps);
        auto [up_base, up_stride, up_file]                = slice_info(layer.ffn_up_exps);
        auto [down_base, down_stride, down_file]          = slice_info(layer.ffn_down_exps);

        if (!check_align(gate_up_stride, "ffn_gate_up_exps") || !check_align(gate_stride, "ffn_gate_exps") ||
            !check_align(up_stride, "ffn_up_exps") || !check_align(down_stride, "ffn_down_exps")) {
            return nullptr;
        }

        layer_expert_sz[il] = gate_up_stride + gate_stride + up_stride + down_stride;
        if (layer_expert_sz[il] > 0) {
            n_expert_used = std::max(n_expert_used, hparams.n_expert_used(il));
        }

        using K                               = llama_expert_cache;
        layer_kind_size[il * NK + K::GATE_UP] = gate_up_stride;
        layer_kind_size[il * NK + K::GATE]    = gate_stride;
        layer_kind_size[il * NK + K::UP]      = up_stride;
        layer_kind_size[il * NK + K::DOWN]    = down_stride;

        layer_kind_ne[il * NK + K::GATE_UP] = get_ne(layer.ffn_gate_up_exps);
        layer_kind_ne[il * NK + K::GATE]    = get_ne(layer.ffn_gate_exps);
        layer_kind_ne[il * NK + K::UP]      = get_ne(layer.ffn_up_exps);
        layer_kind_ne[il * NK + K::DOWN]    = get_ne(layer.ffn_down_exps);

        layer_kind_dtype[il * NK + K::GATE_UP] = layer.ffn_gate_up_exps ? layer.ffn_gate_up_exps->type : GGML_TYPE_F32;
        layer_kind_dtype[il * NK + K::GATE]    = layer.ffn_gate_exps ? layer.ffn_gate_exps->type : GGML_TYPE_F32;
        layer_kind_dtype[il * NK + K::UP]      = layer.ffn_up_exps ? layer.ffn_up_exps->type : GGML_TYPE_F32;
        layer_kind_dtype[il * NK + K::DOWN]    = layer.ffn_down_exps ? layer.ffn_down_exps->type : GGML_TYPE_F32;

        for (uint32_t e = 0; e < n_expert; ++e) {
            locs.push_back({ gate_up_base + e * gate_up_stride, gate_up_stride, gate_up_file });
        }
        for (uint32_t e = 0; e < n_expert; ++e) {
            locs.push_back({ gate_base + e * gate_stride, gate_stride, gate_file });
        }
        for (uint32_t e = 0; e < n_expert; ++e) {
            locs.push_back({ up_base + e * up_stride, up_stride, up_file });
        }
        for (uint32_t e = 0; e < n_expert; ++e) {
            locs.push_back({ down_base + e * down_stride, down_stride, down_file });
        }
    }

    // Match the routing operations used by the model graphs.
    auto resolve_gating_op = [&]() -> llama_expert_gating_func_type {
        if (hparams.expert_gating_func != LLAMA_EXPERT_GATING_FUNC_TYPE_NONE) {
            return (llama_expert_gating_func_type) hparams.expert_gating_func;
        }
        switch (model.arch) {
            case LLM_ARCH_LLAMA4:
            case LLM_ARCH_MIMO2:
            case LLM_ARCH_NEMOTRON_H:
            case LLM_ARCH_NEMOTRON_H_MOE:
                return LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID;
            case LLM_ARCH_OPENAI_MOE:
                return LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT;
            case LLM_ARCH_BAILINGMOE:
            case LLM_ARCH_BERT:
            case LLM_ARCH_JINA_BERT_V3:
            case LLM_ARCH_NOMIC_BERT:
            case LLM_ARCH_NOMIC_BERT_MOE:
            case LLM_ARCH_ARCTIC:
            case LLM_ARCH_DEEPSEEK:
            case LLM_ARCH_DBRX:
            case LLM_ARCH_ERNIE4_5_MOE:
            case LLM_ARCH_GRANITE_HYBRID:
            case LLM_ARCH_GRANITE:
            case LLM_ARCH_GRANITE_MOE:
            case LLM_ARCH_GROK:
            case LLM_ARCH_JAMBA:
            case LLM_ARCH_GROVEMOE:
            case LLM_ARCH_HUNYUAN_MOE:
            case LLM_ARCH_LLAMA:
            case LLM_ARCH_LLAMA_EMBED:
            case LLM_ARCH_MISTRAL3:
            case LLM_ARCH_OLMOE:
            case LLM_ARCH_MELLUM:
            case LLM_ARCH_PHI3:
            case LLM_ARCH_PHIMOE:
            case LLM_ARCH_QWEN2MOE:
            case LLM_ARCH_RND1:
            case LLM_ARCH_QWEN3VLMOE:
            case LLM_ARCH_QWEN35MOE:
            case LLM_ARCH_QWEN3MOE:
            case LLM_ARCH_QWEN3NEXT:
            case LLM_ARCH_LLADA_MOE:
            case LLM_ARCH_GEMMA4:
            case LLM_ARCH_GEMMA4_ASSISTANT:
                return LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX;
            default:
                return LLAMA_EXPERT_GATING_FUNC_TYPE_NONE;
        }
    };

    auto populate_router_descs = [&]() {
        auto * cache = result.get();
        cache->router_descs_.assign(n_layer, moe_router_desc{});
        cache->prefetch_snapshot_.hidden.assign(hparams.n_embd, 0.0f);
        const llama_expert_gating_func_type gating_op        = resolve_gating_op();
        const bool                          prefetch_capable = true;
        for (uint32_t il = 0; il < n_layer; ++il) {
            if (layer_expert_sz[il] == 0) {
                continue;  // dense layers keep gate_inp == nullptr
            }
            const auto &      layer = model.layers[il];
            moe_router_desc & rd    = cache->router_descs_[il];
            rd.gate_inp             = layer.ffn_gate_inp;
            rd.gate_inp_b           = layer.ffn_gate_inp_b;
            rd.exp_probs_b          = layer.ffn_exp_probs_b;
            rd.gating_op            = gating_op;
            rd.expert_router_f32    = hparams.expert_router_f32;
            rd.n_expert             = hparams.n_expert;
            rd.n_expert_used        = hparams.n_expert_used(il);
            rd.n_expert_groups      = hparams.n_expert_groups;
            rd.n_group_used         = hparams.n_group_used;
            rd.arch                 = model.arch;
            rd.n_group_experts      = hparams.n_group_experts;
            rd.prefetch_capable     = prefetch_capable;

            if (model.arch == LLM_ARCH_GEMMA4 || model.arch == LLM_ARCH_GEMMA4_ASSISTANT) {
                rd.pre_gating_kind = MOE_PRE_GATING_GEMMA4;
                rd.gate_inp_s      = layer.ffn_gate_inp_s;
                rd.f_norm_rms_eps  = hparams.f_norm_rms_eps;
            }
        }
    };

    uint64_t cache_bytes    = mparams.expert_cache_size_bytes;
    auto     finalize_cache = [&]() -> std::unique_ptr<llama_expert_cache> {
        result->budget_bytes_ = cache_bytes;
        if (mparams.no_alloc) {
            return std::move(result);
        }
        if (ml.files.empty()) {
            LLAMA_LOG_WARN("%s: expert cache disabled: no files in model loader\n", __func__);
            return nullptr;
        }

        for (const auto & file : ml.files) {
            if (!result->add_file(file->file_id(), ml.use_direct_io)) {
                throw std::runtime_error("could not open expert cache GGUF source");
            }
        }

#ifdef MADV_DONTNEED
        if (ml.use_mmap) {
            const long page_sz = sysconf(_SC_PAGESIZE);
            size_t     n_ev = 0, bytes_ev = 0;
            for (size_t idx = 0; idx < ml.mappings.size(); ++idx) {
                uint8_t * base = (uint8_t *) ml.mappings[idx]->addr();
                for (const auto & kv : ml.weights_map) {
                    const auto & w = kv.second;
                    if ((size_t) w.idx != idx)
                        continue;
                    if (!is_expert_weight_tensor(ggml_get_name(w.tensor)))
                        continue;
                    uintptr_t ar = (uintptr_t) (base + w.offs);
                    uintptr_t aa = ar & ~((uintptr_t) (page_sz - 1));
                    size_t    la = (ggml_nbytes(w.tensor) + (ar - aa) + page_sz - 1) & ~((size_t) (page_sz - 1));
                    if (madvise((void *) aa, la, MADV_DONTNEED) == 0) {
                        ++n_ev;
                        bytes_ev += ggml_nbytes(w.tensor);
                    }
                }
            }
            LLAMA_LOG_INFO("%s: post-load madvise DONTNEED: %zu expert tensors (%.2f GiB)\n", __func__, n_ev,
                           (double) bytes_ev / (1024.0 * 1024.0 * 1024.0));
        }
#endif

        result->start_worker();
        result->budget_bytes_ = cache_bytes;
        return std::move(result);
    };

    size_t     ref_expert_sz   = 0;
    uint32_t   first_moe_layer = 0;
    const bool uniform         = expert_layouts_uniform(ml);
    for (uint32_t il = 0; il < n_layer; ++il) {
        if (layer_expert_sz[il] == 0) {
            continue;
        }
        ref_expert_sz   = layer_expert_sz[il];
        first_moe_layer = il;
        break;
    }

    if (ref_expert_sz == 0) {
        LLAMA_LOG_WARN("%s: expert cache disabled: could not determine expert size\n", __func__);
        return nullptr;
    }

    const bool use_global_nonuniform = (mparams.expert_cache_mode == LLAMA_EXPERT_CACHE_MODE_GLOBAL_NONUNIFORM) ||
                                       (mparams.expert_cache_mode == LLAMA_EXPERT_CACHE_MODE_AUTO && !uniform);

    const bool use_layered = (mparams.expert_cache_mode == LLAMA_EXPERT_CACHE_MODE_LAYERED);

    if (mparams.expert_cache_mode == LLAMA_EXPERT_CACHE_MODE_GLOBAL && !uniform) {
        LLAMA_LOG_WARN(
            "%s: expert cache disabled: global mode requires "
            "uniform expert sizes\n",
            __func__);
        return nullptr;
    }

    // Look up HTP hooks at runtime to avoid linking libllama to libggml-hexagon.
    {
        using get_htp_repack_t               = ggml_backend_buffer_type_t (*)(ggml_backend_dev_t);
        using supports_htp_repack_type_t     = bool (*)(enum ggml_type);
        using htp_repack_range_t             = llama_expert_cache::htp_repack_range_fn;
        using htp_mark_tensor_expert_cache_t = llama_expert_cache::htp_mark_tensor_expert_cache_fn;

        for (uint32_t il = 0; il < n_layer && !result->use_htp_pool_; ++il) {
            auto * dev = model.dev_layer(il);
            if (!dev) {
                continue;
            }
            auto * reg = ggml_backend_dev_backend_reg(dev);
            if (!reg) {
                continue;
            }

            auto get_htp_repack = (get_htp_repack_t) ggml_backend_reg_get_proc_address(
                reg, "ggml_backend_hexagon_get_repack_buffer_type");
            auto supports_htp_repack_type = (supports_htp_repack_type_t) ggml_backend_reg_get_proc_address(
                reg, "ggml_backend_hexagon_supports_expert_repack_type");
            auto htp_repack_range =
                (htp_repack_range_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_hexagon_repack_tensor_range");
            if (!get_htp_repack || !supports_htp_repack_type || !htp_repack_range) {
                continue;
            }

            ggml_backend_buffer_type_t rbt = get_htp_repack(dev);
            if (!rbt) {
                continue;
            }

            bool      supported        = true;
            ggml_type unsupported_type = GGML_TYPE_COUNT;
            for (uint32_t jl = 0; jl < n_layer && supported; ++jl) {
                if (layer_expert_sz[jl] == 0) {
                    continue;
                }
                for (int k = 0; k < NK; ++k) {
                    if (layer_kind_size[jl * NK + k] == 0) {
                        continue;
                    }
                    const ggml_type type = layer_kind_dtype[jl * NK + k];
                    if (!supports_htp_repack_type(type)) {
                        supported        = false;
                        unsupported_type = type;
                        break;
                    }
                }
            }

            if (!supported) {
                LLAMA_LOG_INFO(
                    "%s: expert cache HTP pool disabled - expert tensor type %s is not supported by HTP REPACK\n",
                    __func__, ggml_type_name(unsupported_type));
                continue;
            }

            result->htp_repack_buft_ = rbt;
            auto get_max_size =
                (size_t(*)()) ggml_backend_reg_get_proc_address(reg, "ggml_backend_hexagon_get_expert_cache_max_size");
            if (get_max_size) {
                result->htp_max_pool_size_ = get_max_size();
            }
            result->use_htp_pool_                 = true;
            result->htp_repack_range_             = htp_repack_range;
            if (result->htp_async_) {
                result->htp_mark_callback_ = (htp_mark_tensor_expert_cache_t) ggml_backend_reg_get_proc_address(
                    reg, "ggml_backend_hexagon_mark_cache_callback");
                LLAMA_LOG_INFO("%s: HTP cache worker %s\n", __func__,
                               result->htp_mark_callback_ ? "enabled" : "unavailable");
            }
            LLAMA_LOG_INFO(
                "%s: expert cache HTP pool enabled (FASTRPC REPACK) - "
                "MoE expert matmuls will run on HTP\n",
                __func__);
        }
    }
    std::array<size_t, NK> max_kind_sizes{};
    for (uint32_t il = 0; il < n_layer; ++il) {
        if (layer_expert_sz[il] == 0) {
            continue;
        }
        for (int k = 0; k < NK; ++k) {
            max_kind_sizes[k] = std::max(max_kind_sizes[k], layer_kind_size[il * NK + k]);
        }
    }
    size_t max_expert_sz = 0;
    for (int k = 0; k < NK; ++k) {
        max_expert_sz += max_kind_sizes[k];
    }

    auto pool_slots = [&](size_t slot_bytes, uint64_t minimum) -> uint32_t {
        if (minimum > UINT64_MAX / slot_bytes) {
            throw std::runtime_error("expert cache minimum size overflows uint64");
        }
        const uint64_t min_bytes = minimum * slot_bytes;
        result->minimum_bytes_   = min_bytes;
        const uint64_t layers =
            std::count_if(layer_expert_sz.begin(), layer_expert_sz.end(), [](size_t size) { return size != 0; });
        uint64_t max_slots = std::min<uint64_t>(layers * n_expert, UINT32_MAX);
        if (result->use_htp_pool_) {
            const uint64_t per_buffer = result->htp_max_pool_size_ / slot_bytes;
            max_slots =
                std::min(max_slots, use_layered ? layers * std::min<uint64_t>(n_expert, per_buffer) : per_buffer);
        }
        result->maximum_bytes_ = max_slots > UINT64_MAX / slot_bytes ? UINT64_MAX : max_slots * slot_bytes;
        if (cache_bytes == 0) {
            cache_bytes = min_bytes;
            LLAMA_LOG_INFO("expert cache: automatic minimum = %llu bytes (%.2f MiB)\n",
                           (unsigned long long) cache_bytes, (double) cache_bytes / (1024 * 1024));
        }
        if (cache_bytes < min_bytes) {
            throw std::runtime_error(format(
                "expert cache size %llu bytes is too small; needs at least %llu bytes for this model and cache mode",
                (unsigned long long) cache_bytes, (unsigned long long) min_bytes));
        }
        const uint64_t n_slots = cache_bytes / slot_bytes;
        if (n_slots > UINT32_MAX) {
            throw std::runtime_error("expert cache slot count exceeds uint32");
        }
        return (uint32_t) n_slots;
    };
    const uint32_t global_floor = cache_bytes == 0 && !hparams.causal_attn ? n_expert : n_expert_used;

    if (!use_layered && !use_global_nonuniform) {
        const uint32_t n_slots = pool_slots(ref_expert_sz, global_floor);

        std::vector<size_t> kind_sizes(NK);
        for (int k = 0; k < NK; ++k) {
            kind_sizes[k] = layer_kind_size[first_moe_layer * NK + k];
        }

        result->init_kind_first(n_slots, kind_sizes);
        result->build_offset_map(n_layer, n_expert, n_expert_used, locs);
        result->threshold_pct_ = mparams.expert_cache_threshold;
        result->policy_        = mparams.expert_cache_policy;

        std::array<std::pair<int64_t, int64_t>, llama_expert_cache::N_KINDS> kind_ne{};
        std::array<ggml_type, llama_expert_cache::N_KINDS>                   kind_dtype{};
        for (int k = 0; k < NK; ++k) {
            kind_ne[k]    = layer_kind_ne[first_moe_layer * NK + k];
            kind_dtype[k] = layer_kind_dtype[first_moe_layer * NK + k];
        }
        result->create_pool_tensors(kind_ne, kind_dtype);

        LLAMA_LOG_INFO(
            "%s: expert cache enabled (global), %u slots x %zu bytes = %.2f MB, "
            "threshold=%.1f%%\n",
            __func__, n_slots, ref_expert_sz, (double) (n_slots * ref_expert_sz) / (1024 * 1024),
            mparams.expert_cache_threshold);
        populate_router_descs();
        return finalize_cache();
    }

    if (use_global_nonuniform) {
        const uint32_t n_slots = pool_slots(max_expert_sz, global_floor);

        result->init_shared(n_layer, n_slots, max_kind_sizes);
        result->build_offset_map(n_layer, n_expert, n_expert_used, locs);
        result->threshold_pct_ = mparams.expert_cache_threshold;
        result->policy_        = mparams.expert_cache_policy;
        result->create_pool_tensors_shared(layer_kind_ne, layer_kind_dtype);

        LLAMA_LOG_INFO(
            "%s: expert cache enabled (global-nonuniform), %u slots x %zu bytes = %.2f MB, "
            "threshold=%.1f%%\n",
            __func__, n_slots, max_expert_sz, (double) (n_slots * max_expert_sz) / (1024 * 1024),
            mparams.expert_cache_threshold);
        populate_router_descs();
        return finalize_cache();
    }

    // Reserve one decode working set per MoE layer before distributing extra slots.
    size_t                total_moe_expert_sz = 0;
    uint32_t              n_moe_layers        = 0;
    std::vector<uint32_t> floor_per_layer(n_layer, 0);
    for (uint32_t il = 0; il < n_layer; ++il) {
        if (layer_expert_sz[il] == 0) {
            continue;  // dense layer
        }
        total_moe_expert_sz += layer_expert_sz[il];
        floor_per_layer[il] = std::max(1u, hparams.n_expert_used(il));
        ++n_moe_layers;
    }

    const uint32_t                     prefill_floor = std::max(1u, n_expert);
    size_t                             mbuf          = 0;
    llama_expert_cache::htp_group_plan htp_plan;
    if (result->use_htp_pool_) {
        mbuf     = result->htp_max_pool_size_;
        htp_plan = llama_expert_cache::minimum_htp_layered_groups(floor_per_layer, max_expert_sz, mbuf, prefill_floor);
        if (!htp_plan.error.empty()) {
            throw std::runtime_error(htp_plan.error);
        }
        floor_per_layer = htp_plan.slots_per_layer;
    }

    uint64_t minimum_slots = std::accumulate(floor_per_layer.begin(), floor_per_layer.end(), uint64_t(0));
    if (cache_bytes == 0 && !hparams.causal_attn && minimum_slots < n_expert) {
        floor_per_layer[first_moe_layer] += n_expert - minimum_slots;
        minimum_slots = n_expert;
    }
    const uint32_t total_affordable = pool_slots(max_expert_sz, minimum_slots);
    const uint32_t remainder        = total_affordable - minimum_slots;

    std::vector<uint32_t> slots_per_layer(n_layer, 0);
    uint32_t              total_slots = 0;

    for (uint32_t il = 0; il < n_layer; ++il) {
        if (layer_expert_sz[il] == 0) {
            continue;  // dense layer
        }

        // Keep the pool within the cache budget.
        uint32_t extra = (total_moe_expert_sz > 0) ?
                             (uint32_t) ((uint64_t) remainder * layer_expert_sz[il] / total_moe_expert_sz) :
                             0;
        uint32_t n_sl  = floor_per_layer[il] + extra;

        slots_per_layer[il] = n_sl;
        total_slots += n_sl;
    }

    if (result->use_htp_pool_) {
        LLAMA_LOG_INFO("%s: expert cache HTP prefill group floor = n_expert=%u\n", __func__, n_expert);
    }

    // Distribute remaining slots.
    uint32_t leftover = total_affordable - total_slots;

    if (result->use_htp_pool_) {
        htp_plan = llama_expert_cache::plan_htp_layered_groups(slots_per_layer, floor_per_layer, leftover,
                                                               max_expert_sz, mbuf, prefill_floor);

        if (!htp_plan.error.empty()) {
            LLAMA_LOG_ERROR(
                "%s: %s. cache size %.2f MB cannot form prefill-safe HTP groups "
                "(prefill_floor=%u, opt_mbuf=%.2f MB).\n",
                __func__, htp_plan.error.c_str(), (double) cache_bytes / (1024 * 1024), prefill_floor,
                (double) mbuf / (1024 * 1024));
            return nullptr;
        }

        slots_per_layer = std::move(htp_plan.slots_per_layer);
        total_slots     = 0;
        for (uint32_t n_sl : slots_per_layer) {
            total_slots += n_sl;
        }
    } else {
        for (uint32_t il = 0; il < n_layer && leftover > 0; ++il) {
            if (layer_expert_sz[il] == 0) {
                continue;  // dense layer
            }
            slots_per_layer[il] += 1;
            total_slots += 1;
            leftover -= 1;
        }
    }

    GGML_ASSERT((uint64_t) total_slots <= total_affordable);
    for (uint32_t il = 0; il < n_layer; ++il) {
        if (layer_expert_sz[il] == 0) {
            continue;
        }
        GGML_ASSERT(slots_per_layer[il] >= floor_per_layer[il]);
    }

    result->init_layered(n_layer, slots_per_layer, layer_kind_size);
    if (result->use_htp_pool_) {
        result->planned_htp_groups_ = std::move(htp_plan.groups);
    }
    result->build_offset_map(n_layer, n_expert, n_expert_used, locs);
    result->threshold_pct_ = mparams.expert_cache_threshold;
    result->policy_        = mparams.expert_cache_policy;
    result->create_pool_tensors_layered(layer_kind_ne, layer_kind_dtype, prefill_floor);

    auto * cache = result.get();
    if (cache->use_htp_pool_ && !cache->group_buffers_.empty()) {
        for (uint32_t il = 0; il < n_layer; ++il) {
            if (layer_expert_sz[il] == 0) {
                continue;
            }
            GGML_ASSERT(il < cache->layer_group_base_.size());
            GGML_ASSERT(il < cache->layer_group_count_.size());
            GGML_ASSERT(cache->layer_group_count_[il] == 1);

            const uint32_t g = cache->layer_group_base_[il];
            GGML_ASSERT(g < cache->group_buffers_.size());

            bool found = false;
            for (int k = 0; k < NK; ++k) {
                ggml_tensor * t = cache->layered_pool_tensors_[(size_t) il * NK + k];
                if (!t) {
                    continue;
                }
                GGML_ASSERT(t->buffer == cache->group_buffers_[g]);
                found = true;
            }
            GGML_ASSERT(found);
        }
    }

    LLAMA_LOG_INFO(
        "%s: expert cache enabled (layered), %u MoE layers, %u total slots, "
        "%.2f MB allocated (requested %.2f MB), avg %.1f slots/layer, threshold=%.1f%%\n",
        __func__, n_moe_layers, total_slots, (double) ((size_t) total_slots * max_expert_sz) / (1024 * 1024),
        (double) cache_bytes / (1024 * 1024), n_moe_layers > 0 ? (double) total_slots / n_moe_layers : 0.0,
        mparams.expert_cache_threshold);
    populate_router_descs();
    return finalize_cache();
}

uint64_t llama_model_expert_cache_size(const llama_model * model) {
    return model->expert_cache ? model->expert_cache->budget_bytes_ : 0;
}

llama_expert_cache_memory llama_model_expert_cache_memory(const llama_model * model) {
    if (!model->expert_cache) {
        return {};
    }
    const auto & cache = *model->expert_cache;
    return { cache.minimum_bytes_, cache.maximum_bytes_, (uint64_t) cache.n_slots_ * cache.expert_sz,
             cache.budget_bytes_,  cache.expert_sz,      cache.automatic_ };
}

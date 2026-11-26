#include "llama-expert-cache-graph.h"

#include "llama-graph.h"
#include "llama-impl.h"

#include <cmath>
#include <functional>
#include <stdexcept>

static ggml_tensor * mark_cache_callback(llama_expert_cache_context * state, ggml_tensor * tensor) {
    if (state->cache->use_htp_pool_ && state->cache->htp_mark_callback_) {
        state->cache->htp_mark_callback_(tensor);
    }
    return tensor;
}

llama_expert_cache_context::llama_expert_cache_context(llama_expert_cache * cache) :
    cache(cache),
    ops(cache->n_layer_) {
    for (uint32_t l = 0; l < ops.size(); ++l) {
        ops[l].cache     = cache;
        ops[l].layer_idx = l;
    }
}

llama_expert_cache_context::~llama_expert_cache_context() {
    for (auto & op : ops) {
        cache->release_slots(op);
    }
}

void llama_expert_cache_context::limit_batch(llama_cparams & cparams) const {
    uint32_t n_slots = cache->n_slots_;
    for (uint32_t n : cache->group_n_slots_) {
        n_slots = std::min(n_slots, n);
    }
    if (n_slots >= cache->n_expert_) {
        return;
    }
    const uint32_t max_tokens = n_slots / std::max(1u, cache->n_expert_used_);
    if (cparams.n_ubatch <= max_tokens) {
        return;
    }
    if (!cparams.causal_attn) {
        throw std::runtime_error(
            "expert cache cannot hold this non-causal microbatch; increase LLAMA_EXPERT_CACHE_SIZE or reduce n_ubatch");
    }
    LLAMA_LOG_WARN(
        "%s: expert cache limits n_ubatch from %u to %u; increase LLAMA_EXPERT_CACHE_SIZE for larger microbatches\n",
        __func__, cparams.n_ubatch, max_tokens);
    cparams.n_ubatch = max_tokens;
}

ggml_status llama_expert_cache_graph_compute(llama_expert_cache_context * state,
                                             ggml_backend_sched_t         sched,
                                             ggml_cgraph *                gf) {
    if (!state) {
        return ggml_backend_sched_graph_compute_async(sched, gf);
    }
    bool uses_htp = false;
    for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
        const auto dev = ggml_backend_get_device(ggml_backend_sched_get_backend(sched, i));
        const auto reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        uses_htp       = uses_htp || (reg && std::strcmp(ggml_backend_reg_name(reg), "HTP") == 0);
    }
    // HTP contexts share the device's submission buffers.
    std::unique_lock<std::mutex> htp_lock(state->cache->htp_compute_mu_, std::defer_lock);
    if (uses_htp) {
        htp_lock.lock();
    }
    const uint64_t token = state->cache->decode_step_counter_.fetch_add(1, std::memory_order_relaxed);
    for (auto & op : state->ops) {
        op.token = token;
    }
    const auto status  = ggml_backend_sched_graph_compute_async(sched, gf);
    bool       pending = uses_htp || status != GGML_STATUS_SUCCESS;
    for (const auto & op : state->ops) {
        pending = pending || !op.pinned.empty();
    }
    if (pending) {
        ggml_backend_sched_synchronize(sched);
        for (auto & op : state->ops) {
            state->cache->release_slots(op);
        }
    }
    return status;
}

void llama_expert_cache_finish_layer(llama_expert_cache_context * state,
                                     ggml_context *               ctx,
                                     ggml_cgraph *                gf,
                                     ggml_tensor *                result,
                                     uint32_t                     layer) {
    if (!state || ggml_is_empty(result)) {
        return;
    }
    // A scalar dependency keeps slots pinned until the expert output is ready.
    ggml_tensor * done = ggml_view_1d(ctx, result, 1, 0);
    done = ggml_map_custom1_inplace(ctx, done, llama_expert_cache::release_slots_fn, 1, &state->ops.at(layer));
    mark_cache_callback(state, done);
    ggml_format_name(done, "expert_cache_release-%u", layer);
    ggml_build_forward_expand(gf, done);
}

ggml_tensor * llama_expert_cache_publish_router_input(llama_expert_cache_context * state,
                                                      ggml_context *               ctx,
                                                      ggml_cgraph *                gf,
                                                      ggml_tensor *                router_input,
                                                      ggml_tensor *                prefetch_router_input,
                                                      uint32_t                     layer,
                                                      int64_t                      n_tokens) {
    if (!state || state->cache->decode_prefetch_lookahead_ == 0 || n_tokens != 1 || layer >= state->ops.size()) {
        return router_input;
    }

    ggml_tensor * snapshot_source = prefetch_router_input ? prefetch_router_input : router_input;
    ggml_tensor * snapshot =
        ggml_map_custom1_inplace(ctx, snapshot_source, llama_expert_cache::snapshot_fn, 1, &state->ops[layer]);
    if (prefetch_router_input == nullptr) {
        return snapshot;
    }

    ggml_build_forward_expand(gf, snapshot);
    return router_input;
}

ggml_tensor * llama_expert_cache_reroute_selected(llama_expert_cache_context * state,
                                                  ggml_context *               ctx,
                                                  ggml_tensor *                selected_experts,
                                                  ggml_tensor *                selection_probs,
                                                  uint32_t                     layer) {
    if (!state || layer >= state->ops.size()) {
        return selected_experts;
    }

    return mark_cache_callback(state, ggml_map_custom2_inplace(ctx, selected_experts, selection_probs,
                                                               llama_expert_cache::reroute_fn, 1, &state->ops[layer]));
}

ggml_tensor * llama_expert_cache_slot_ids(llama_expert_cache_context * state,
                                          ggml_context *               ctx,
                                          ggml_tensor *                selected_experts,
                                          uint32_t                     layer) {
    if (!state || layer >= state->ops.size()) {
        return selected_experts;
    }

    return mark_cache_callback(
        state, ggml_map_custom1(ctx, selected_experts, llama_expert_cache::id_to_slot_fn, 1, &state->ops[layer]));
}

llama_expert_cache_pool_tensors llama_expert_cache_get_pool_tensors(llama_expert_cache_context * state,
                                                                    uint32_t                     layer,
                                                                    int64_t                      n_tokens) {
    llama_expert_cache_pool_tensors result;
    if (!state) {
        return result;
    }
    auto * cache = state->cache;

    if (!cache->layered_mode_) {
        result.gate_up = cache->pool_tensor(layer, llama_expert_cache::GATE_UP);
        result.gate    = cache->pool_tensor(layer, llama_expert_cache::GATE);
        result.up      = cache->pool_tensor(layer, llama_expert_cache::UP);
        result.down    = cache->pool_tensor(layer, llama_expert_cache::DOWN);
        return result;
    }

    const bool use_global = n_tokens > 1;
    if (use_global) {
        result.gate_up = cache->group_global_tensor(layer, llama_expert_cache::GATE_UP);
        result.gate    = cache->group_global_tensor(layer, llama_expert_cache::GATE);
        result.up      = cache->group_global_tensor(layer, llama_expert_cache::UP);
        result.down    = cache->group_global_tensor(layer, llama_expert_cache::DOWN);
    } else if (cache->use_htp_pool_) {
        result.gate_up = cache->pool_tensor_for_layer(layer, llama_expert_cache::GATE_UP);
        result.gate    = cache->pool_tensor_for_layer(layer, llama_expert_cache::GATE);
        result.up      = cache->pool_tensor_for_layer(layer, llama_expert_cache::UP);
        result.down    = cache->pool_tensor_for_layer(layer, llama_expert_cache::DOWN);
    } else {
        result.gate_up = cache->pool_tensor(layer, llama_expert_cache::GATE_UP);
        result.gate    = cache->pool_tensor(layer, llama_expert_cache::GATE);
        result.up      = cache->pool_tensor(layer, llama_expert_cache::UP);
        result.down    = cache->pool_tensor(layer, llama_expert_cache::DOWN);
    }

    return result;
}

ggml_tensor * llama_expert_cache_gemma4_pregate(ggml_context * ctx,
                                                ggml_tensor *  attn_out,
                                                ggml_tensor *  gate_inp_s,
                                                int64_t        n_embd,
                                                float          eps) {
    ggml_tensor * t = ggml_rms_norm(ctx, attn_out, eps);
    t               = ggml_scale(ctx, t, 1.0f / sqrtf((float) n_embd));
    t               = ggml_mul(ctx, t, gate_inp_s);
    return t;
}

ggml_tensor * llama_expert_cache_apply_pre_gating(moe_pre_gating_kind     kind,
                                                  ggml_context *          ctx,
                                                  ggml_tensor *           router_input,
                                                  const moe_router_desc & rd) {
    switch (kind) {
        case MOE_PRE_GATING_GEMMA4:
            return llama_expert_cache_gemma4_pregate(ctx, router_input, rd.gate_inp_s, router_input->ne[0],
                                                     rd.f_norm_rms_eps);
        case MOE_PRE_GATING_NONE:
        default:
            return router_input;
    }
}

ggml_tensor * llm_graph_build_moe_router(ggml_context *                                                ctx,
                                         ggml_tensor *                                                 cur,
                                         const moe_router_desc &                                       rd,
                                         int                                                           il,
                                         const std::function<void(ggml_tensor *, const char *, int)> & cb_fn,
                                         bool           dup_selected_experts,
                                         ggml_tensor ** out_selection_probs,
                                         ggml_tensor ** out_probs,
                                         ggml_tensor *  selected_experts_in) {
    const int64_t n_tokens = cur->ne[1];

    ggml_tensor * logits = nullptr;

    if (rd.gate_inp != nullptr) {
        logits = ggml_mul_mat(ctx, rd.gate_inp, cur);  // [n_expert, n_tokens]
        if (rd.gating_op == LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS || rd.expert_router_f32) {
            ggml_prec_set_acc(logits, GGML_PREC_F32);
        }
        cb_fn(logits, "ffn_moe_logits", il);
    } else {
        logits = cur;
    }

    if (rd.gate_inp_b) {
        logits = ggml_add(ctx, logits, rd.gate_inp_b);
        cb_fn(logits, "ffn_moe_logits_biased", il);
    }

    ggml_tensor * probs = nullptr;
    switch (rd.gating_op) {
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX:
            {
                probs = ggml_soft_max(ctx, logits);  // [n_expert, n_tokens]
            }
            break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID:
            {
                probs = ggml_sigmoid(ctx, logits);  // [n_expert, n_tokens]
            }
            break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT:
            {
                probs = logits;  // [n_expert, n_tokens]
            }
            break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS:
            {
                probs = ggml_sqrt(ctx, ggml_softplus(ctx, logits));  // [n_expert, n_tokens]
            }
            break;
        default:
            GGML_ABORT("fatal error");
    }
    cb_fn(probs, "ffn_moe_probs", il);

    // DeepSeek V3 biases expert selection but keeps expert weights unbiased.
    ggml_tensor * selection_probs = probs;
    if (rd.exp_probs_b != nullptr) {
        selection_probs = ggml_add(ctx, probs, rd.exp_probs_b);
        cb_fn(selection_probs, "ffn_moe_probs_biased", il);
    }

    // Llama 4 applies sigmoid after top-k.
    if (rd.arch == LLM_ARCH_LLAMA4) {
        selection_probs = logits;
    }

    if (rd.arch == LLM_ARCH_GROVEMOE) {
        selection_probs = ggml_sigmoid(ctx, logits);  // [n_expert, n_tokens]
        cb_fn(selection_probs, "ffn_moe_probs_biased", il);
    }

    // Select the top expert groups.
    if (rd.n_expert_groups > 1 && n_tokens > 0) {
        const int64_t n_exp_per_group = rd.n_expert / rd.n_expert_groups;

        // organize experts into n_expert_groups
        ggml_tensor * selection_groups = ggml_reshape_3d(ctx, selection_probs, n_exp_per_group, rd.n_expert_groups,
                                                         n_tokens);  // [n_exp_per_group, n_expert_groups, n_tokens]

        ggml_tensor * group_scores = ggml_argsort_top_k(ctx, selection_groups, 2);  // [2, n_expert_groups, n_tokens]
        group_scores               = ggml_get_rows(ctx,
                                                   ggml_reshape_4d(ctx, selection_groups, 1, selection_groups->ne[0],
                                                                   selection_groups->ne[1], selection_groups->ne[2]),
                                                   group_scores);  // [1, 2, n_expert_groups, n_tokens]

        // get top n_group_used expert groups
        group_scores = ggml_sum_rows(ctx, ggml_reshape_3d(ctx, group_scores, group_scores->ne[1], group_scores->ne[2],
                                                          group_scores->ne[3]));  // [1, n_expert_groups, n_tokens]
        group_scores = ggml_reshape_2d(ctx, group_scores, group_scores->ne[1],
                                       group_scores->ne[2]);                      // [n_expert_groups, n_tokens]

        ggml_tensor * expert_groups =
            ggml_argsort_top_k(ctx, group_scores, rd.n_group_used);  // [n_group_used, n_tokens]
        cb_fn(expert_groups, "ffn_moe_group_topk", il);

        // mask out the other groups
        selection_probs =
            ggml_get_rows(ctx, selection_groups, expert_groups);  // [n_exp_per_group, n_group_used, n_tokens]
        selection_probs = ggml_set_rows(ctx, ggml_fill(ctx, selection_groups, -INFINITY), selection_probs,
                                        expert_groups);           // [n_exp_per_group, n_expert_groups, n_tokens]
        selection_probs = ggml_reshape_2d(ctx, selection_probs, rd.n_expert, n_tokens);  // [n_expert, n_tokens]
        cb_fn(selection_probs, "ffn_moe_probs_masked", il);
    }

    // select experts
    ggml_tensor * selected_experts = selected_experts_in;
    if (selected_experts == nullptr) {
        selected_experts = ggml_argsort_top_k(ctx, selection_probs, rd.n_expert_used);  // [n_expert_used, n_tokens]
        cb_fn(selected_experts->src[0], "ffn_moe_argsort", il);
    }
    cb_fn(selected_experts, "ffn_moe_topk", il);

    // Prevent scheduler aliasing before cache remapping.
    if (dup_selected_experts) {
        selected_experts = ggml_dup(ctx, selected_experts);
    }

    *out_selection_probs = selection_probs;
    *out_probs           = probs;

    return selected_experts;
}

ggml_tensor * llm_graph_context::build_moe_router(ggml_context *          ctx,
                                                  ggml_tensor *           cur,
                                                  const moe_router_desc & rd,
                                                  int                     il,
                                                  ggml_tensor **          out_selection_probs,
                                                  ggml_tensor **          out_probs,
                                                  ggml_tensor *           selected_experts_in) const {
    // Compute router logits here so the main graph keeps LoRA support.
    ggml_tensor *   cur_for_impl = cur;
    moe_router_desc rd_no_lora   = rd;
    if (rd.gate_inp != nullptr) {
        cur_for_impl = build_lora_mm(rd.gate_inp, cur);  // [n_expert, n_tokens]
        cb(cur_for_impl, "ffn_moe_logits", il);
        rd_no_lora.gate_inp = nullptr;
    }

    const auto cb_fn = [this](ggml_tensor * t, const char * name, int il_) {
        this->cb(t, name, il_);
    };

    const bool dup_selected_experts = expert_cache != nullptr;

    return llm_graph_build_moe_router(ctx, cur_for_impl, rd_no_lora, il, cb_fn, dup_selected_experts,
                                      out_selection_probs, out_probs, selected_experts_in);
}

ggml_tensor * llama_expert_cache_build_moe_router_standalone(ggml_context *          ctx,
                                                             ggml_tensor *           cur,
                                                             const moe_router_desc & rd,
                                                             int                     il,
                                                             ggml_tensor **          out_selection_probs,
                                                             ggml_tensor **          out_probs) {
    static const std::function<void(ggml_tensor *, const char *, int)> no_op_cb = [](ggml_tensor *, const char *, int) {
    };

    ggml_tensor * router_in = llama_expert_cache_apply_pre_gating(rd.pre_gating_kind, ctx, cur, rd);

    return llm_graph_build_moe_router(ctx, router_in, rd, il, no_op_cb, /*dup_selected_experts=*/false,
                                      out_selection_probs, out_probs);
}

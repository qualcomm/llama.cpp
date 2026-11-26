#pragma once

#include "llama-expert-cache.h"

#include <functional>

struct llama_cparams;

struct llama_expert_cache_context {
    explicit llama_expert_cache_context(llama_expert_cache * cache);
    ~llama_expert_cache_context();

    void limit_batch(llama_cparams & cparams) const;

    llama_expert_cache *                         cache;
    std::vector<llama_expert_cache::op_params_t> ops;
};

ggml_status llama_expert_cache_graph_compute(llama_expert_cache_context * state,
                                             ggml_backend_sched_t         sched,
                                             ggml_cgraph *                gf);
void        llama_expert_cache_finish_layer(llama_expert_cache_context * state,
                                            ggml_context *               ctx,
                                            ggml_cgraph *                gf,
                                            ggml_tensor *                result,
                                            uint32_t                     layer);

struct llama_expert_cache_pool_tensors {
    ggml_tensor * gate_up = nullptr;
    ggml_tensor * gate    = nullptr;
    ggml_tensor * up      = nullptr;
    ggml_tensor * down    = nullptr;
};

ggml_tensor * llama_expert_cache_publish_router_input(llama_expert_cache_context * state,
                                                      ggml_context *               ctx,
                                                      ggml_cgraph *                gf,
                                                      ggml_tensor *                router_input,
                                                      ggml_tensor *                prefetch_router_input,
                                                      uint32_t                     layer,
                                                      int64_t                      n_tokens);

ggml_tensor * llama_expert_cache_reroute_selected(llama_expert_cache_context * state,
                                                  ggml_context *               ctx,
                                                  ggml_tensor *                selected_experts,
                                                  ggml_tensor *                selection_probs,
                                                  uint32_t                     layer);

ggml_tensor * llama_expert_cache_slot_ids(llama_expert_cache_context * state,
                                          ggml_context *               ctx,
                                          ggml_tensor *                selected_experts,
                                          uint32_t                     layer);

llama_expert_cache_pool_tensors llama_expert_cache_get_pool_tensors(llama_expert_cache_context * state,
                                                                    uint32_t                     layer,
                                                                    int64_t                      n_tokens);

ggml_tensor * llama_expert_cache_build_moe_router_standalone(ggml_context *          ctx,
                                                             ggml_tensor *           cur,
                                                             const moe_router_desc & rd,
                                                             int                     il,
                                                             ggml_tensor **          out_selection_probs,
                                                             ggml_tensor **          out_probs);

ggml_tensor * llama_expert_cache_gemma4_pregate(ggml_context * ctx,
                                                ggml_tensor *  attn_out,
                                                ggml_tensor *  gate_inp_s,
                                                int64_t        n_embd,
                                                float          eps);

ggml_tensor * llama_expert_cache_apply_pre_gating(moe_pre_gating_kind     kind,
                                                  ggml_context *          ctx,
                                                  ggml_tensor *           router_input,
                                                  const moe_router_desc & rd);

// Build router graphs without llm_graph_context or LoRA adapters.
ggml_tensor * llm_graph_build_moe_router(ggml_context *                                                ctx,
                                         ggml_tensor *                                                 cur,
                                         const moe_router_desc &                                       rd,
                                         int                                                           il,
                                         const std::function<void(ggml_tensor *, const char *, int)> & cb,
                                         bool           duplicate_selected_experts,
                                         ggml_tensor ** out_selection_probs,
                                         ggml_tensor ** out_probs,
                                         ggml_tensor *  selected_experts_in = nullptr);

#include "models.h"

// Gemma 4 assistant (MTP drafter)
//
// The assistant has no KV cache of its own: every layer attends to the KV cache of the last
// SWA / full-attention layer of the target model (see the share callback in llama_model::create_memory),
// so the context must be created with llama_context_params.ctx_other = <target context>.
//
// Inputs per token: the target token id (embedded with the *target* token embeddings) and the target
// post-norm hidden state (llama_get_embeddings_nextn_ith() on the target context).
llm_build_gemma4_assistant::llm_build_gemma4_assistant(const llama_model & model, const llm_graph_params & params) :
        llm_graph_context(params) {
    const int64_t n_embd_backbone = hparams.n_embd_inp();

    ggml_tensor * inp_tokens;
    ggml_tensor * inp_h;
    {
        auto inp = std::make_unique<llm_graph_input_embd>(n_embd_backbone);

        inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);
        cb(inp->tokens, "inp_tokens", -1);
        ggml_set_input(inp->tokens);
        inp_tokens = inp->tokens;
        res->t_inp_tokens = inp->tokens;

        inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_backbone, ubatch.n_tokens);
        cb(inp->embd, "inp_h", -1);
        ggml_set_input(inp->embd);
        inp_h = inp->embd;
        res->t_inp_embd = inp->embd;

        res->add_input(std::move(inp));
    }

    GGML_ASSERT(cparams.ctx_other != nullptr);
    const auto * model_other = llama_get_model(cparams.ctx_other);

    ggml_tensor * x = ggml_get_rows(ctx0, model_other->tok_embd, inp_tokens);
    x = ggml_scale(ctx0, x, sqrtf((float) n_embd_backbone));
    cb(x, "inp_embd_target", -1);

    ggml_tensor * xh = ggml_concat(ctx0, x, inp_h, 0);
    cb(xh, "inp_xh", -1);

    ggml_tensor * cur = ggml_mul_mat(ctx0, model.nextn_proj_pre, xh);
    cb(cur, "pre_proj", -1);

    auto *        inp_attn    = build_attn_inp_kv_iswa();
    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    ggml_tensor * inpL = cur;

    for (int il = 0; il < n_layer_nextn; ++il) {
        const bool is_swa = hparams.is_swa(il);

        const int64_t n_embd_head = hparams.n_embd_head_k(il);
        const int64_t n_head      = hparams.n_head(il);

        const float freq_base_l  = model.get_rope_freq_base(cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);
        const int   n_rot_l      = hparams.n_rot(il);

        ggml_tensor * cur_norm = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur_norm, "attn_norm", il);

        ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur_norm);
        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
        Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
        cb(Qcur, "Qcur_normed", il);

        ggml_tensor * freq_factors = is_swa ? nullptr : model.layers[il].rope_freqs;
        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, freq_factors, n_rot_l, rope_type, n_ctx_orig,
                             freq_base_l, freq_scale_l, ext_factor, attn_factor, beta_fast, beta_slow);
        cb(Qcur, "Qcur_pos", il);

        // K/V come from the shared target KV cache
        cur = build_attn(inp_attn, model.layers[il].wo, nullptr, nullptr,
                Qcur, nullptr, nullptr, nullptr, nullptr, nullptr, hparams.f_attention_scale, il);

        if (il == n_layer_nextn - 1 && inp_out_ids) {
            cur  = ggml_get_rows(ctx0, cur,  inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }

        cur = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        ggml_tensor * attn_out = ggml_add(ctx0, cur, inpL);
        cb(attn_out, "attn_out", il);

        cur = build_norm(attn_out, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   nullptr, nullptr,
                model.layers[il].ffn_gate, nullptr, nullptr,
                model.layers[il].ffn_down, nullptr, nullptr,
                nullptr,
                LLM_FFN_GELU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = build_norm(cur, model.layers[il].ffn_post_norm, nullptr, LLM_NORM_RMS, -1);
        cb(cur, "ffn_post_norm", il);

        cur = ggml_add(ctx0, cur, attn_out);

        cur = ggml_mul(ctx0, cur, model.layers[il].out_scale);
        cb(cur, "out_scaled", il);

        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    ggml_tensor * logits = build_lora_mm(model.output, cur);
    cb(logits, "result_output", -1);
    res->t_logits = logits;

    ggml_tensor * h_next = ggml_mul_mat(ctx0, model.nextn_proj_post, cur);
    cb(h_next, "h_nextn", -1);
    res->t_h_nextn = h_next;

    ggml_build_forward_expand(gf, logits);
    ggml_build_forward_expand(gf, h_next);
}

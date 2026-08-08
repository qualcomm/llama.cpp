// nodehash - dump a checksum of every graph node, per decode step.
//
// Purpose: localize a NON-DETERMINISM to a single graph node without statistics.
// Run the same prompt twice and diff the two dumps: the FIRST differing line names
// the tensor, the op and the layer that first diverges. That is a deterministic
// instrument -- unlike "distinct completions / N", which cannot separate 1/N from
// 2/N and is confounded by whatever else the machine was doing at the time.
//
// The defect this was written for lives in the ne1==1 DECODE path (prefill is
// bit-exact), so the prompt eval is hashed separately from each decode step and
// every line is tagged with its step.
//
// Config is by environment so no new CLI flags are needed:
//   LLAMA_NODEHASH_OUT    output file            (default: nodehash.txt)
//   LLAMA_NODEHASH_STEPS  greedy decode steps    (default: 4)
//   LLAMA_NODEHASH_SKIP_PROMPT=1  hash decode steps only
//   LLAMA_NODEHASH_FILTER comma-separated name prefixes to hash (default: all)
//
// USE THE FILTER ON A FUSING BACKEND. ggml_backend_sched computes the graph one
// node at a time for every node the callback asks for (ggml-backend.cpp: the
// `if (!sched->callback_eval)` branch), and a backend can only fuse ops it sees
// together in one graph. Asking for everything therefore silently DISABLES
// fusion -- so on ggml-opencl an unfiltered run would not exercise the fused
// GLU/MoE kernels at all, and would be measuring a different code path than the
// one under investigation. With a filter, only the ranges ending at a matched
// node are broken up; everything else is still handed to the backend in blocks.
// Localize to a layer with a sparse filter first (e.g. LLAMA_NODEHASH_FILTER=l_out),
// then narrow inside that layer. Confirm the kernel you care about still fires
// (GGML_OPENCL_FUSE_DEBUG=1) rather than assuming it.
//
// Everything else (-m, -ngl, -p, --override-tensor, ...) is the usual common_params.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <cinttypes>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct nodehash_ctx {
    FILE *                   out   = nullptr;
    int                      step  = -1;     // -1 = prompt eval, >=0 = decode step
    bool                     armed = true;
    int                      seq   = 0;      // node index within the step
    std::vector<std::string> filter;         // empty = every node
    std::vector<uint8_t>     buf;

    bool wanted(const char * name) const {
        if (filter.empty()) {
            return true;
        }
        for (const auto & f : filter) {
            if (strstr(name, f.c_str())) {
                return true;
            }
        }
        return false;
    }
};

// FNV-1a over the raw bytes: sensitive to a single flipped mantissa bit, which is
// the resolution this needs (the q4_0 twin diverged by ~1e-4, but the first node to
// move may differ in the last bit only).
static uint64_t fnv1a64(const uint8_t * p, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; ++i) {
        h ^= (uint64_t) p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

// A magnitude signal alongside the hash: two runs that differ tell you HOW FAR
// apart they are, which is what separated real error from benign reassociation
// in the q4_0 case (three orders agreeing to 1e-7 vs one 1000x off).
static double float_sum(const uint8_t * data, ggml_type type, size_t nbytes) {
    double s = 0.0;
    if (type == GGML_TYPE_F32) {
        const float * f = (const float *) data;
        for (size_t i = 0; i < nbytes / sizeof(float); ++i) {
            s += (double) f[i];
        }
    } else if (type == GGML_TYPE_F16) {
        const ggml_fp16_t * f = (const ggml_fp16_t *) data;
        for (size_t i = 0; i < nbytes / sizeof(ggml_fp16_t); ++i) {
            s += (double) ggml_fp16_to_fp32(f[i]);
        }
    }
    return s;
}

static bool nodehash_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cb = (nodehash_ctx *) user_data;

    if (ask) {
        return cb->armed && cb->wanted(t->name);
    }
    if (!cb->armed) {
        return true;
    }

    const size_t nbytes = ggml_nbytes(t);
    const bool   host   = ggml_backend_buffer_is_host(t->buffer);

    const uint8_t * data;
    if (host) {
        data = (const uint8_t *) t->data;
    } else {
        cb->buf.resize(nbytes);
        ggml_backend_tensor_get(t, cb->buf.data(), 0, nbytes);
        data = cb->buf.data();
    }

    fprintf(cb->out, "%d %5d %-32s %-14s %-8s [%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] %016" PRIx64 " %.17g\n",
            cb->step, cb->seq++, t->name, ggml_op_desc(t), ggml_type_name(t->type),
            t->ne[0], t->ne[1], t->ne[2], t->ne[3],
            fnv1a64(data, nbytes), float_sum(data, t->type, nbytes));

    return true;
}

static int env_int(const char * name, int def) {
    const char * v = getenv(name);
    return (v && v[0]) ? atoi(v) : def;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    nodehash_ctx cb;

    const char * out_path = getenv("LLAMA_NODEHASH_OUT");
    cb.out = fopen(out_path && out_path[0] ? out_path : "nodehash.txt", "w");
    if (!cb.out) {
        LOG_ERR("%s: cannot open output file\n", __func__);
        return 1;
    }

    const int  n_steps     = env_int("LLAMA_NODEHASH_STEPS", 4);
    const bool skip_prompt = env_int("LLAMA_NODEHASH_SKIP_PROMPT", 0) != 0;

    if (const char * f = getenv("LLAMA_NODEHASH_FILTER")) {
        std::string s(f);
        size_t pos = 0;
        while (pos < s.size()) {
            size_t c = s.find(',', pos);
            if (c == std::string::npos) {
                c = s.size();
            }
            std::string tok = s.substr(pos, c - pos);
            if (!tok.empty()) {
                cb.filter.push_back(tok);
            }
            pos = c + 1;
        }
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    params.cb_eval           = nodehash_cb;
    params.cb_eval_user_data = &cb;
    // The warmup runs the same graph and would leave the backend in a different
    // state between the two runs being compared. Keep it off so both runs see an
    // identical history.
    params.warmup            = false;

    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s: failed to init\n", __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, llama_vocab_get_add_bos(vocab), true);
    if (tokens.empty()) {
        LOG_ERR("%s: no input tokens - pass a prompt with -p\n", __func__);
        return 1;
    }

    LOG_INF("%s: %zu prompt tokens, %d decode steps\n", __func__, tokens.size(), n_steps);

    cb.step  = -1;
    cb.seq   = 0;
    cb.armed = !skip_prompt;

    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()))) {
        LOG_ERR("%s: prompt eval failed\n", __func__);
        return 1;
    }

    const int n_vocab = llama_vocab_n_tokens(vocab);
    int       n_past  = (int) tokens.size();

    for (int step = 0; step < n_steps; ++step) {
        // greedy argmax over the logits of the last position - no sampler, so the
        // token sequence is a pure function of the logits and nothing else can
        // introduce a difference between the two runs.
        const float * logits = llama_get_logits_ith(ctx, -1);

        llama_token best = 0;
        float       bestv = logits[0];
        for (int i = 1; i < n_vocab; ++i) {
            if (logits[i] > bestv) {
                bestv = logits[i];
                best  = i;
            }
        }

        fprintf(cb.out, "# step %d token %d logit %.17g\n", step, best, (double) bestv);

        if (llama_vocab_is_eog(vocab, best)) {
            LOG_INF("%s: end of generation at step %d\n", __func__, step);
            break;
        }

        cb.step  = step;
        cb.seq   = 0;
        cb.armed = true;

        if (llama_decode(ctx, llama_batch_get_one(&best, 1))) {
            LOG_ERR("%s: decode failed at step %d\n", __func__, step);
            return 1;
        }
        n_past++;
    }

    // the logits AFTER the last hashed step, so a divergence that only shows up in
    // the final output is still visible in the dump
    {
        const float * logits = llama_get_logits_ith(ctx, -1);
        llama_token best = 0;
        float bestv = logits[0];
        for (int i = 1; i < n_vocab; ++i) {
            if (logits[i] > bestv) { bestv = logits[i]; best = i; }
        }
        fprintf(cb.out, "# final token %d logit %.17g\n", best, (double) bestv);
    }

    fclose(cb.out);

    LOG("\n");
    llama_perf_context_print(ctx);

    llama_backend_free();

    GGML_UNUSED(n_past);

    return 0;
}

#include "expert-cache-fixture.h"
#include "llama-model.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(condition)                                                            \
    do {                                                                            \
        if (!(condition)) {                                                         \
            fprintf(stderr, "check failed: %s at line %d\n", #condition, __LINE__); \
            abort();                                                                \
        }                                                                           \
    } while (0)

static void set_env(const char * name, const char * value) {
#if defined(_WIN32)
    CHECK(_putenv_s(name, value ? value : "") == 0);
#else
    CHECK((value ? setenv(name, value, 1) : unsetenv(name)) == 0);
#endif
}

struct env_override {
    const char * name;
    bool         present;
    std::string  saved;

    env_override(const char * name) : name(name), present(getenv(name) != nullptr), saved(present ? getenv(name) : "") {
        set_env(name, nullptr);
    }

    ~env_override() { set_env(name, present ? saved.c_str() : nullptr); }
};

static std::vector<float> decode(const char *            path,
                                 bool                    cached,
                                 llama_expert_cache_mode mode,
                                 bool                    prefetch,
                                 bool                    token_by_token,
                                 ggml_backend_dev_t      accelerator = nullptr) {
    env_override lookahead("LLAMA_EXPERT_CACHE_DECODE_PREFETCH_LOOKAHEAD");
    set_env(lookahead.name, prefetch ? "1" : "0");
    auto               mp        = llama_model_default_params();
    ggml_backend_dev_t devices[] = { accelerator, nullptr };
    mp.devices                   = devices;
    mp.n_gpu_layers              = accelerator ? 99 : 0;
    mp.use_extra_bufts           = false;
    mp.expert_cache_enabled      = cached;
    mp.expert_cache_mode         = mode;
    mp.load_mode                 = LLAMA_LOAD_MODE_NONE;
    auto * model                 = llama_model_load_from_file(path, mp);
    CHECK(model);
    CHECK((llama_model_expert_cache_size(model) != 0) == cached);
    auto cp    = llama_context_default_params();
    cp.n_ctx   = 64;
    cp.n_batch = cp.n_ubatch = 8;
    cp.n_threads = cp.n_threads_batch = 2;
    cp.flash_attn_type                = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    auto * ctx                        = llama_init_from_model(model, cp);
    CHECK(ctx);
    auto * batch = llama_batch_ext_init(ctx);
    CHECK(batch);
    llama_pos position = 0;
    auto      process  = [&](const llama_token * input, int n) {
        llama_batch_ext_clear(batch);
        for (int i = 0; i < n; ++i) {
            const int32_t index = llama_batch_ext_add_token(batch, 0, input[i]);
            CHECK(index >= 0);
            CHECK(llama_batch_ext_set_pos(batch, index, &position));
            CHECK(llama_batch_ext_set_output_logits(batch, index, i == n - 1));
            ++position;
        }
        CHECK(llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) == 0);
    };
    const int         vocab    = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const llama_token tokens[] = { 1, 2, 3, 4, 5, 6 };
    if (token_by_token) {
        for (auto token : tokens) {
            process(&token, 1);
        }
    } else {
        process(tokens, 6);
    }
    const float *      logits = llama_get_logits_ith(ctx, -1);
    std::vector<float> result(logits, logits + vocab);
    for (float value : result) {
        CHECK(std::isfinite(value));
    }
    // Reuse the decode graph after prefill and force cache turnover.
    for (int i = 0; i < 8; ++i) {
        llama_token token = (i * 3 + 7) % vocab;
        process(&token, 1);
        logits = llama_get_logits_ith(ctx, -1);
        result.insert(result.end(), logits, logits + vocab);
    }
    if (cached) {
        const auto & cache = *model->expert_cache;
        CHECK(cache.worker_thread_.joinable() == prefetch);
        std::lock_guard<std::mutex> lock(model->expert_cache->worker_mu_);
        CHECK((cache.worker_wake_seq_ > 0) == prefetch);
        if (prefetch) {
            CHECK(!cache.prefetch_snapshot_.hidden.empty());
            printf("lookahead worker active; decode snapshots published: %llu\n",
                   (unsigned long long) cache.worker_wake_seq_);
        }
    }
    llama_batch_ext_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    return result;
}

static void check_equal(const std::vector<float> & expected, const std::vector<float> & actual) {
    CHECK(expected.size() == actual.size());
    float max_error = 0.0f;
    for (size_t i = 0; i < expected.size(); ++i) {
        CHECK(std::isfinite(actual[i]));
        max_error = std::max(max_error, std::fabs(expected[i] - actual[i]));
        CHECK(std::fabs(expected[i] - actual[i]) <= 2e-5f * (1.0f + std::fabs(expected[i])));
    }
    printf("cache routing max logit error: %.8g\n", max_error);
}

static void check_accelerator(const std::vector<float> & baseline,
                              const std::vector<float> & actual,
                              const char *               comparison) {
    CHECK(baseline.size() == actual.size());
    double error = 0.0, energy = 0.0;
    for (size_t i = 0; i < baseline.size(); ++i) {
        CHECK(std::isfinite(actual[i]) && std::isfinite(baseline[i]));
        const double delta = double(baseline[i]) - actual[i];
        error += delta * delta;
        energy += double(baseline[i]) * baseline[i];
    }
    const double nmse = energy > 0 ? error / energy : error;
    printf("%s: NMSE=%.9g limit=0.0005\n", comparison, nmse);
    // Match the MUL_MAT_ID criterion in test-backend-ops.
    CHECK(nmse <= 5e-4);
}

static void test_disabled_environment(const char * path) {
    for (const char * name : { "LLAMA_EXPERT_CACHE_SIZE", "LLAMA_EXPERT_CACHE_THRESHOLD", "LLAMA_EXPERT_CACHE_MODE",
                               "LLAMA_EXPERT_CACHE_POLICY" }) {
        env_override variable(name);
        set_env(name, "invalid");
        auto mp            = llama_model_default_params();
        mp.n_gpu_layers    = 0;
        mp.use_extra_bufts = false;
        auto * model       = llama_model_load_from_file(path, mp);
        CHECK(model && llama_model_expert_cache_size(model) == 0);
        llama_model_free(model);
    }
    puts("disabled cache ignores cache-only environment: PASS");
}

int main(int argc, char ** argv) {
    const env_override environment[] = {
        { "LLAMA_EXPERT_CACHE_SIZE" },
        { "LLAMA_EXPERT_CACHE_THRESHOLD" },
        { "LLAMA_EXPERT_CACHE_MODE" },
        { "LLAMA_EXPERT_CACHE_POLICY" },
        { "LLAMA_EXPERT_CACHE_MLOCK" },
        { "LLAMA_EXPERT_CACHE_DDR_COPY" },
        { "LLAMA_EXPERT_CACHE_READ_PERCENTAGE" },
        { "LLAMA_EXPERT_CACHE_DECODE_PREFETCH_FORCE_CACHE_HIT" },
        { "LLAMA_EXPERT_CACHE_DECODE_PREFETCH_LOOKAHEAD" },
        { "LLAMA_EXPERT_CACHE_BATCH_DEPTH" },
        { "LLAMA_EXPERT_CACHE_HTP_READ_CHUNK_KB" },
        { "LLAMA_EXPERT_CACHE_HTP_ASYNC" },
        { "LLAMA_EXPERT_CACHE_STATS" },
    };
    llama_backend_init();
    ggml_backend_dev_t accelerator = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto dev = ggml_backend_dev_get(i);
        if (std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)), "HTP") == 0) {
            accelerator = dev;
            break;
        }
    }
    expert_cache_fixture uniform;
    test_disabled_environment(uniform.path.c_str());
    if (argc == 2 && std::strcmp(argv[1], "--disabled-env-only") == 0) {
        llama_backend_free();
        return 0;
    }
    for (ggml_type type : { GGML_TYPE_F32, GGML_TYPE_Q4_0, GGML_TYPE_F8_E4M3 }) {
        for (int granularity : { 0, 1, 2 }) {
            expert_cache_fixture fixture(false, 32, type, 64, granularity);
            const auto baseline = decode(fixture.path.c_str(), false, LLAMA_EXPERT_CACHE_MODE_LAYERED, false, false);
            for (auto mode : { LLAMA_EXPERT_CACHE_MODE_LAYERED, LLAMA_EXPERT_CACHE_MODE_GLOBAL }) {
                check_equal(baseline,
                            decode(fixture.path.c_str(), true, mode, mode == LLAMA_EXPERT_CACHE_MODE_LAYERED, false));
            }
            printf("scaled file-backed experts: type=%s granularity=%d PASS\n", ggml_type_name(type), granularity);
        }
    }
    {
        expert_cache_fixture fixture(false, 32, GGML_TYPE_F32, 64, -1, GGML_TYPE_F8_E4M3);
        const auto baseline = decode(fixture.path.c_str(), false, LLAMA_EXPERT_CACHE_MODE_LAYERED, false, false);
        check_equal(baseline, decode(fixture.path.c_str(), true, LLAMA_EXPERT_CACHE_MODE_LAYERED, true, false));
        puts("scaled FP8 router with lookahead: PASS");
    }
    for (bool nonuniform : { false, true }) {
        expert_cache_fixture fixture(nonuniform);
        const auto baseline = decode(fixture.path.c_str(), false, LLAMA_EXPERT_CACHE_MODE_LAYERED, false, false);
        check_equal(baseline, decode(fixture.path.c_str(), false, LLAMA_EXPERT_CACHE_MODE_LAYERED, false, true));
        std::vector<float> accelerator_batch, accelerator_decode;
        if (accelerator) {
            accelerator_batch =
                decode(fixture.path.c_str(), false, LLAMA_EXPERT_CACHE_MODE_LAYERED, false, false, accelerator);
            accelerator_decode =
                decode(fixture.path.c_str(), false, LLAMA_EXPERT_CACHE_MODE_LAYERED, false, true, accelerator);
            check_accelerator(baseline, accelerator_batch, "uncached accelerator versus CPU");
            check_accelerator(accelerator_batch, accelerator_decode, "uncached accelerator batch versus decode");
        }
        for (auto mode : { LLAMA_EXPERT_CACHE_MODE_LAYERED, LLAMA_EXPERT_CACHE_MODE_AUTO,
                           LLAMA_EXPERT_CACHE_MODE_GLOBAL_NONUNIFORM, LLAMA_EXPERT_CACHE_MODE_GLOBAL }) {
            if (nonuniform && mode == LLAMA_EXPERT_CACHE_MODE_GLOBAL) {
                continue;
            }
            check_equal(baseline, decode(fixture.path.c_str(), true, mode, false, false));
            check_equal(baseline,
                        decode(fixture.path.c_str(), true, mode, mode == LLAMA_EXPERT_CACHE_MODE_LAYERED, true));
            if (accelerator) {
                check_accelerator(accelerator_batch,
                                  decode(fixture.path.c_str(), true, mode, false, false, accelerator),
                                  "cached versus uncached accelerator batch");
                check_accelerator(accelerator_decode,
                                  decode(fixture.path.c_str(), true, mode, mode == LLAMA_EXPERT_CACHE_MODE_LAYERED,
                                         true, accelerator),
                                  "cached versus uncached accelerator decode");
            }
        }
    }
    llama_backend_free();
    puts("test-expert-cache-routing: PASS");
    return 0;
}

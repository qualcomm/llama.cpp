#include "common.h"
#include "expert-cache-fixture.h"
#include "fit.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "htp-callback.h"
#include "htp/htp-ops.h"
#include "llama-expert-cache-io.h"
#include "llama-expert-cache.h"
#include "llama-ext.h"
#include "llama-graph.h"
#include "llama-model.h"
#include "llama.h"

#include <fcntl.h>

#ifdef NDEBUG
#    undef NDEBUG
#endif
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <future>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <io.h>
#    include <windows.h>
#    define close   _close
#    define read    _read
#    define write   _write
#    define open    _open
#    define lseek   _lseeki64
#    define unlink  _unlink
#    define access  _access
#    define F_OK    0
#    define mkstemp test_mkstemp

static int test_mkstemp(char * path) {
    if (_mktemp_s(path, strlen(path) + 1) != 0) {
        return -1;
    }
    HANDLE handle =
        CreateFileA(path, GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                    CREATE_NEW, FILE_ATTRIBUTE_TEMPORARY, nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
        return -1;
    }
    const int fd = _open_osfhandle((intptr_t) handle, _O_RDWR | _O_BINARY);
    if (fd < 0) {
        CloseHandle(handle);
    }
    return fd;
}
#else
#    include <unistd.h>
#endif

#define EC_CHECK(cond)                                                                 \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            fprintf(stderr, "check failed: %s at %s:%d\n", #cond, __FILE__, __LINE__); \
            abort();                                                                   \
        }                                                                              \
    } while (0)

static void test_callback_increment(ggml_tensor * dst, const ggml_tensor * src, int ith, int nth, void *) {
    EC_CHECK(ith == 0 && nth == 1);
    *(int32_t *) dst->data = *(const int32_t *) src->data + 1;
}

[[noreturn]] static void test_callback_fail(ggml_tensor *, const ggml_tensor *, int, int, void *) {
    throw std::runtime_error("callback failure");
}

static uint32_t test_callback_wait(uint8_t * fence, uint32_t seq) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (htp_callback_fence_seq(fence) != seq) {
        if (std::chrono::steady_clock::now() >= deadline) {
            fprintf(stderr, "timeout waiting for HTP callback sequence %u\n", seq);
            abort();
        }
        std::this_thread::yield();
    }
    return htp_callback_fence_status(fence);
}

static void test_htp_callback_worker() {
    ggml_context * ctx   = ggml_init({ 4096, nullptr, true });
    ggml_tensor *  src   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    int32_t        input = 7, output = -1;
    src->data         = &input;
    ggml_tensor * dst = ggml_map_custom1(ctx, src, test_callback_increment, 1, nullptr);
    dst->data         = &output;
    EC_CHECK(!htp_callback_is_marked(dst));
    htp_callback_mark(dst);
    EC_CHECK(htp_callback_is_marked(dst));
    alignas(HTP_FENCE_SLOT_SIZE) uint8_t control[2 * HTP_FENCE_SLOT_SIZE]{};
    {
        htp_callback_worker worker(control);
        for (int i = 0; i < 32; ++i) {
            input              = 7 + i;
            const uint32_t seq = worker.enqueue(dst);
            htp_callback_fence_write(control, seq - 1, HTP_STATUS_OK);
            EC_CHECK(htp_callback_fence_seq(control + HTP_FENCE_SLOT_SIZE) != seq);
            htp_callback_fence_write(control, seq, HTP_STATUS_OK);
            EC_CHECK(test_callback_wait(control + HTP_FENCE_SLOT_SIZE, seq) == HTP_STATUS_OK);
            EC_CHECK(output == 8 + i);
        }
        ggml_tensor * fail = ggml_map_custom1(ctx, src, test_callback_fail, 1, nullptr);
        fail->data         = &output;
        htp_callback_mark(fail);
        uint32_t seq = worker.enqueue(fail);
        htp_callback_fence_write(control, seq, HTP_STATUS_OK);
        EC_CHECK(test_callback_wait(control + HTP_FENCE_SLOT_SIZE, seq) == HTP_STATUS_INTERNAL_ERR);
        output = -1;
        seq    = worker.enqueue(dst);
        htp_callback_fence_write(control, seq, HTP_STATUS_OK);
        EC_CHECK(test_callback_wait(control + HTP_FENCE_SLOT_SIZE, seq) == HTP_STATUS_INTERNAL_ERR);
        EC_CHECK(output == -1);
    }
    {
        htp_callback_worker worker(control);
        worker.enqueue(dst);
    }
    for (int i = 0; i < 32; ++i) {
        htp_callback_worker worker(control);
        const uint32_t      seq = worker.enqueue(dst);
        htp_callback_fence_write(control, seq, HTP_STATUS_OK);
        EC_CHECK(test_callback_wait(control + HTP_FENCE_SLOT_SIZE, seq) == HTP_STATUS_OK);
        EC_CHECK(output == input + 1);
    }
    ggml_free(ctx);
}

static void test_callback_scale(ggml_tensor * dst, const ggml_tensor * src, int, int, void *) {
    for (int64_t i = 0; i < ggml_nelements(dst); ++i) {
        ((float *) dst->data)[i] = 3.0f * ((const float *) src->data)[i];
    }
}

static void test_callback_clear_weights(ggml_tensor * dst, const ggml_tensor * src, int ith, int nth, void * userdata) {
    auto weights = static_cast<ggml_tensor **>(userdata);
    for (int i = 0; i < 2; ++i) {
        memset(weights[i]->data, 0, ggml_nbytes(weights[i]));
    }
    test_callback_scale(dst, src, ith, nth, nullptr);
}

static void test_htp_callback_fusion(ggml_backend_t backend, void (*mark)(ggml_tensor *)) {
    auto          ctx        = ggml_init({ 65536, nullptr, true });
    auto          input      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 32);
    ggml_tensor * weights[2] = {
        ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 32, 32),
        ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 32, 32),
    };
    auto first    = ggml_mul_mat(ctx, weights[0], input);
    auto second   = ggml_mul_mat(ctx, weights[1], input);
    auto callback = ggml_map_custom1(ctx, input, test_callback_clear_weights, 1, weights);
    mark(callback);
    auto output = ggml_swiglu_split(ctx, first, second);
    ggml_set_input(input);
    ggml_set_output(output);
    auto gf = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(gf, first);
    ggml_build_forward_expand(gf, second);
    ggml_build_forward_expand(gf, callback);
    ggml_build_forward_expand(gf, output);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    EC_CHECK(buffer);
    std::vector<float>       values(1024, 1.0f), result(1024);
    std::vector<ggml_fp16_t> weight_values(1024, ggml_fp32_to_fp16(1.0f));
    ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(float));
    for (auto weight : weights) {
        ggml_backend_tensor_set(weight, weight_values.data(), 0, weight_values.size() * sizeof(ggml_fp16_t));
    }
    EC_CHECK(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
    for (float value : result) {
        EC_CHECK(std::fabs(value - 1024.0f) < 0.01f);
    }
    ggml_backend_synchronize(backend);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

static void test_htp_callback_graph() {
    ggml_backend_load_all();
    for (int pass = 0; pass < 4; ++pass) {
        const char * name       = pass % 2 == 0 ? "CPU" : "HTP0";
        const bool   normalized = pass >= 2;
        auto         dev        = ggml_backend_dev_by_name(name);
        EC_CHECK(dev);
        auto backend = ggml_backend_dev_init(dev, nullptr);
        EC_CHECK(backend);
        auto reg  = ggml_backend_dev_backend_reg(dev);
        auto mark = (void (*)(ggml_tensor *)) ggml_backend_reg_get_proc_address(
            reg, "ggml_backend_hexagon_mark_cache_callback");
        if (mark) {
            auto reorder_ctx = ggml_init({ 65536, nullptr, true });
            auto x           = ggml_new_tensor_1d(reorder_ctx, GGML_TYPE_F32, 32);
            auto first       = ggml_mul_mat(reorder_ctx, ggml_new_tensor_2d(reorder_ctx, GGML_TYPE_Q4_0, 32, 32), x);
            auto side_input  = ggml_new_tensor_1d(reorder_ctx, GGML_TYPE_F32, 32);
            auto barrier     = ggml_map_custom1(reorder_ctx, side_input, test_callback_scale, 1, nullptr);
            mark(barrier);
            auto second = ggml_mul_mat(reorder_ctx, ggml_new_tensor_2d(reorder_ctx, GGML_TYPE_Q4_0, 32, 32), x);
            auto gf     = ggml_new_graph_custom(reorder_ctx, 32, false);
            ggml_build_forward_expand(gf, first);
            ggml_build_forward_expand(gf, barrier);
            ggml_build_forward_expand(gf, ggml_add(reorder_ctx, first, second));
            ggml_backend_graph_optimize_params params{};
            backend->iface.graph_optimize(backend, gf, &params);
            int barrier_pos = -1, second_pos = -1;
            for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
                if (ggml_graph_node(gf, i) == barrier) {
                    barrier_pos = i;
                }
                if (ggml_graph_node(gf, i) == second) {
                    second_pos = i;
                }
            }
            EC_CHECK(barrier_pos >= 0 && second_pos > barrier_pos);
            ggml_free(reorder_ctx);
            if (!normalized) {
                test_htp_callback_fusion(backend, mark);
            }
        }
        ggml_context * ctx      = ggml_init({ 65536, nullptr, true });
        auto           input    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32);
        auto           scaled   = ggml_scale(ctx, normalized ? ggml_rms_norm(ctx, input, 1e-6f) : input, 2.0f);
        auto           callback = ggml_map_custom1(ctx, scaled, test_callback_scale, 1, nullptr);
        if (mark) {
            EC_CHECK(!ggml_backend_dev_supports_op(dev, callback));
            mark(callback);
        }
        EC_CHECK(ggml_backend_dev_supports_op(dev, callback));
        auto output = ggml_scale(ctx, callback, 5.0f);
        auto gf     = ggml_new_graph_custom(ctx, 32, false);
        ggml_build_forward_expand(gf, output);
        auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        EC_CHECK(buffer);
        for (int step = 0; step < 32; ++step) {
            float values[32], result[32];
            float sum_sq = 0.0f;
            for (int i = 0; i < 32; ++i) {
                values[i] = (float) (i + step);
                sum_sq += values[i] * values[i];
            }
            ggml_backend_tensor_set(input, values, 0, sizeof(values));
            EC_CHECK(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
            ggml_backend_tensor_get(output, result, 0, sizeof(result));
            for (int i = 0; i < 32; ++i) {
                if (normalized) {
                    const float expected = values[i] * 30.0f / std::sqrt(sum_sq / 32.0f + 1e-6f);
                    EC_CHECK(std::fabs(result[i] - expected) < 1e-4f * (1.0f + std::fabs(expected)));
                } else {
                    EC_CHECK(result[i] == values[i] * 30.0f);
                }
            }
        }
        ggml_backend_synchronize(backend);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
    }
    printf("test_htp_callback_graph: PASS\n");
}

static void test_set_env(const char * name, const char * value) {
#if defined(_WIN32)
    EC_CHECK(_putenv_s(name, value ? value : "") == 0);
#else
    EC_CHECK((value ? setenv(name, value, 1) : unsetenv(name)) == 0);
#endif
}

struct test_env_override {
    const char * name;
    std::string  saved;
    bool         was_set;

    test_env_override(const char * name) : name(name) {
        const char * value = getenv(name);
        was_set            = value != nullptr;
        if (value) {
            saved = value;
        }
        test_set_env(name, nullptr);
    }

    ~test_env_override() { test_set_env(name, was_set ? saved.c_str() : nullptr); }
};

static void test_cache_config() {
    const test_env_override env[] = {
        { "LLAMA_EXPERT_CACHE_SIZE" },
        { "LLAMA_EXPERT_CACHE_THRESHOLD" },
        { "LLAMA_EXPERT_CACHE_MODE" },
        { "LLAMA_EXPERT_CACHE_POLICY" },
        { "LLAMA_EXPERT_CACHE_MLOCK" },
        { "LLAMA_EXPERT_CACHE_DDR_COPY" },
        { "LLAMA_EXPERT_CACHE_READ_PERCENTAGE" },
        { "LLAMA_EXPERT_CACHE_DECODE_PREFETCH_LOOKAHEAD" },
        { "LLAMA_EXPERT_CACHE_DECODE_PREFETCH_FORCE_CACHE_HIT" },
        { "LLAMA_EXPERT_CACHE_BATCH_DEPTH" },
        { "LLAMA_EXPERT_CACHE_HTP_READ_CHUNK_KB" },
        { "LLAMA_EXPERT_CACHE_HTP_ASYNC" },
        { "LLAMA_EXPERT_CACHE_STATS" },
    };
    common_params common;
    EC_CHECK(!common.expert_cache_enabled && common.expert_cache_size_bytes == 0);
    EC_CHECK(common.expert_cache_threshold == 0.0f);
    EC_CHECK(common.expert_cache_mode == LLAMA_EXPERT_CACHE_MODE_LAYERED);
    EC_CHECK(common.expert_cache_policy == LLAMA_EXPERT_CACHE_POLICY_LRU);
    auto defaults = llama_model_default_params();
    EC_CHECK(!defaults.expert_cache_enabled && defaults.expert_cache_size_bytes == 0);
    EC_CHECK(defaults.expert_cache_threshold == 0.0f);
    EC_CHECK(defaults.expert_cache_mode == LLAMA_EXPERT_CACHE_MODE_LAYERED);
    EC_CHECK(defaults.expert_cache_policy == LLAMA_EXPERT_CACHE_POLICY_LRU);
    common.expert_cache_threshold = 20.0f;
    EC_CHECK(common_model_params_to_llama(common).expert_cache_threshold == 20.0f);

    const char * path = getenv("LLAMA_TEST_MODEL");
    for (int option = 0; option < 3; ++option) {
        const bool   enabled       = option == 1;
        const bool   async_enabled = option != 2;
        const char * threshold     = enabled ? "20" : option == 2 ? "0" : nullptr;
        test_set_env("LLAMA_EXPERT_CACHE_THRESHOLD", threshold);
        test_set_env("LLAMA_EXPERT_CACHE_DECODE_PREFETCH_FORCE_CACHE_HIT", enabled ? "1" : option == 2 ? "0" : nullptr);
        test_set_env("LLAMA_EXPERT_CACHE_DECODE_PREFETCH_LOOKAHEAD", enabled ? "13" : option == 2 ? "0" : nullptr);
        test_set_env("LLAMA_EXPERT_CACHE_HTP_ASYNC", enabled ? "1" : option == 2 ? "0" : nullptr);
        llama_expert_cache cache;
        EC_CHECK(cache.htp_async_ == async_enabled);
        EC_CHECK(cache.decode_prefetch_force_cache_hit_ == enabled);
        EC_CHECK(cache.decode_prefetch_lookahead_ == (enabled ? 13 : 0));
        EC_CHECK(cache.read_percentage_ == 0 && cache.batch_depth_ == 4);
        EC_CHECK(cache.htp_read_chunk_kb_ == 512);
        EC_CHECK(!cache.stats_ && !cache.mlock_ && !cache.ddr_copy_);
        if (!enabled) {
            cache.start_worker();
            EC_CHECK(!cache.worker_thread_.joinable());
        }
        if (path) {
            auto mp                 = defaults;
            mp.expert_cache_enabled = true;
            mp.n_gpu_layers         = 0;
            mp.use_extra_bufts      = false;
            mp.no_alloc             = true;
            mp.load_mode            = LLAMA_LOAD_MODE_NONE;
            llama_model * model     = llama_model_load_from_file(path, mp);
            EC_CHECK(model && model->expert_cache);
            const auto & resolved = *model->expert_cache;
            EC_CHECK(resolved.htp_async_ == async_enabled && !resolved.htp_mark_callback_);
            EC_CHECK(resolved.threshold_pct_ == (enabled ? 20.0f : 0.0f));
            EC_CHECK(resolved.decode_prefetch_force_cache_hit_ == enabled);
            EC_CHECK(resolved.decode_prefetch_lookahead_ == (enabled ? 13 : 0));
            EC_CHECK(resolved.automatic_ && resolved.layered_mode_);
            EC_CHECK(resolved.policy_ == LLAMA_EXPERT_CACHE_POLICY_LRU);
            printf("test_cache_config: %s threshold=%.0f forced_hit=%d lookahead=%d htp_async=%d\n",
                   option == 0 ? "unset" : "explicit", resolved.threshold_pct_,
                   resolved.decode_prefetch_force_cache_hit_, resolved.decode_prefetch_lookahead_, resolved.htp_async_);
            llama_model_free(model);
        }
    }
    if (!path) {
        printf("test_cache_config: model resolution SKIP (set LLAMA_TEST_MODEL to enable)\n");
    }
    printf("test_cache_config: PASS\n");
}

static void test_empty_cache() {
    llama_expert_cache  cache;
    std::vector<size_t> kind_sizes(llama_expert_cache::N_KINDS, 0);
    kind_sizes[llama_expert_cache::GATE_UP] = 64;
    cache.init_kind_first(10, kind_sizes);
    int32_t s = cache.claim_slot(0, 5);
    assert(s >= 0);
    // A loading slot can be released before its read starts.
    cache.release(s);
    int32_t s2 = cache.claim_slot(0, 5);
    assert(s2 >= 0);
    cache.release(s2);
    printf("test_empty_cache: PASS\n");
}

static void test_slot_state_machine() {
    llama_expert_cache  cache;
    std::vector<size_t> kind_sizes(llama_expert_cache::N_KINDS, 0);
    kind_sizes[llama_expert_cache::GATE_UP] = 64;
    cache.init_kind_first(10, kind_sizes);
    int slot = cache.claim_slot(0, 5);  // EMPTY -> LOADING
    assert(slot >= 0);
    // Loading slots reserve their key.
    assert(cache.claim_slot(0, 5) == -1);
    cache.mark_occupied(slot);  // LOADING -> OCCUPIED
    assert(cache.claim_slot(0, 5) == -1);
    int evicted = cache.evict_lfu_excluding(nullptr, 0, UINT32_MAX);
    assert(evicted == slot);  // OCCUPIED -> EVICTING
    cache.release(evicted);   // EVICTING -> EMPTY
    int s2 = cache.claim_slot(0, 5);
    assert(s2 >= 0);
    cache.release(s2);
    printf("test_slot_state_machine: PASS\n");
}

static void test_lfu_eviction() {
    llama_expert_cache  cache;
    std::vector<size_t> kind_sizes(llama_expert_cache::N_KINDS, 0);
    kind_sizes[llama_expert_cache::GATE_UP] = 64;
    cache.init_kind_first(2, kind_sizes);

    int s0 = cache.claim_slot(0, 0);
    cache.mark_occupied(s0);
    int s1 = cache.claim_slot(0, 1);
    cache.mark_occupied(s1);

    cache.record_use(s0);
    cache.record_use(s0);  // usage=2
    cache.record_use(s1);  // usage=1

    int evicted = cache.evict_lfu_excluding(nullptr, 0, UINT32_MAX);
    assert(evicted == s1);  // s1 has lower usage
    cache.release(evicted);
    printf("test_lfu_eviction: PASS\n");
}

static void test_release_and_reuse() {
    llama_expert_cache  cache;
    std::vector<size_t> kind_sizes(llama_expert_cache::N_KINDS, 0);
    kind_sizes[llama_expert_cache::GATE_UP] = 64;
    cache.init_kind_first(1, kind_sizes);

    int s = cache.claim_slot(0, 7);
    cache.mark_occupied(s);
    int evicted = cache.evict_lfu_excluding(nullptr, 0, UINT32_MAX);
    assert(evicted == s);
    // Eviction alone does not make the slot available.
    assert(cache.claim_slot(1, 3) == -1);
    cache.release(evicted);  // EVICTING -> EMPTY
    int s2 = cache.claim_slot(1, 3);
    assert(s2 == 0);         // same physical slot reused
    cache.release(s2);
    printf("test_release_and_reuse: PASS\n");
}

static void test_claim_full_cache() {
    llama_expert_cache  cache;
    std::vector<size_t> kind_sizes(llama_expert_cache::N_KINDS, 0);
    kind_sizes[llama_expert_cache::GATE_UP] = 64;
    cache.init_kind_first(2, kind_sizes);

    int s0 = cache.claim_slot(0, 0);
    cache.mark_occupied(s0);
    int s1 = cache.claim_slot(0, 1);
    cache.mark_occupied(s1);
    int s2 = cache.claim_slot(0, 2);  // cache full
    (void) s2;
    assert(s2 == -1);
    printf("test_claim_full_cache: PASS\n");
}

static void test_load_expert_sync() {
    std::string tmpfile = (std::filesystem::temp_directory_path() / "test-expert-XXXXXX").string();
    int         fd      = mkstemp(tmpfile.data());
    assert(fd >= 0);

    const size_t expert_sz = 64;
    uint8_t      pat[expert_sz];
    for (size_t i = 0; i < expert_sz; ++i) {
        pat[i] = (uint8_t) i;
    }
    int64_t w = write(fd, pat, expert_sz);
    (void) w;
    assert(w == (int64_t) expert_sz);

    llama_expert_cache  cache;
    std::vector<size_t> kind_sizes(llama_expert_cache::N_KINDS, 0);
    kind_sizes[llama_expert_cache::GATE_UP] = expert_sz;
    cache.init_kind_first(2, kind_sizes);
    EC_CHECK(cache.add_file(fd));

    std::vector<llama_expert_cache::expert_slice_loc> locs((size_t) 1 * llama_expert_cache::N_KINDS * 1);
    locs[0] = { 0, expert_sz };  // GATE_UP: offset 0, size 64
    for (int k = 1; k < llama_expert_cache::N_KINDS; ++k) {
        locs[k] = { 0, 0 };
    }
    cache.build_offset_map(1, 1, 1, locs);

    int32_t slot = cache.claim_slot(0, 0);
    assert(slot >= 0);
    cache.load_expert_sync(slot, 0, 0);
    assert(memcmp(cache.kind_slot_ptr(llama_expert_cache::GATE_UP, (uint32_t) slot), pat, expert_sz) == 0);

    close(fd);
    unlink(tmpfile.c_str());
    printf("test_load_expert_sync: PASS\n");
}

static void test_split_file_reads() {
    for (bool direct : { false, true }) {
        for (bool threaded : { false, true }) {
            llama_expert_cache cache;
            cache.thread_pool_disabled_        = !threaded;
            cache.worker_thread_pool_disabled_ = !threaded;
            std::vector<uint8_t> contents[2];
            for (uint16_t file_idx = 0; file_idx < 2; ++file_idx) {
                const char * dir = getenv("LLAMA_TEST_ODIRECT_DIR");
                char         path[4096];
                snprintf(path, sizeof(path), "%s/test-expert-shard-XXXXXX",
                         dir ? dir : std::filesystem::temp_directory_path().string().c_str());
                int fd = mkstemp(path);
                EC_CHECK(fd >= 0);
                contents[file_idx].resize(16352);
                for (size_t i = 0; i < contents[file_idx].size(); ++i) {
                    contents[file_idx][i] = (uint8_t) (i * 7 + file_idx * 101);
                }
                EC_CHECK(write(fd, contents[file_idx].data(), contents[file_idx].size()) ==
                         (int64_t) contents[file_idx].size());
#if defined(__linux__)
                if (direct) {
                    const int direct_fd = open(path, O_RDONLY | O_DIRECT);
                    if (direct_fd >= 0) {
                        close(fd);
                        fd = direct_fd;
                    }
                }
#endif
                EC_CHECK(cache.add_file(fd, direct));
                close(fd);
                unlink(path);
            }

            std::vector<size_t> sizes(llama_expert_cache::N_KINDS, 0);
            sizes[llama_expert_cache::GATE_UP] = 64;
            sizes[llama_expert_cache::DOWN]    = 64;
            cache.init_kind_first(4, sizes);
            std::vector<llama_expert_cache::expert_slice_loc> locs(llama_expert_cache::N_KINDS * 4);
            const size_t                                      offsets[] = { 32, 4064, 8160, 16288 };
            std::vector<llama_expert_cache::load_req>         reqs;
            for (uint32_t e = 0; e < 4; ++e) {
                locs[llama_expert_cache::GATE_UP * 4 + e] = { offsets[e], 64, 0 };
                locs[llama_expert_cache::DOWN * 4 + e]    = { offsets[e], 64, 1 };
            }
            cache.build_offset_map(1, 4, 2, locs);
            for (uint32_t e = 0; e < 4; ++e) {
                const int32_t slot = cache.claim_slot(0, e);
                EC_CHECK(slot >= 0);
                reqs.push_back({ slot, 0, e });
            }
            cache.load_expert_sync(reqs[0].slot, 0, 0);
            std::thread worker(
                [&] { cache.load_experts_batch_cpu(&reqs[3], 1, false, llama_expert_cache::cpu_load_origin::worker); });
            cache.load_experts_batch_cpu(&reqs[1], 2, false, llama_expert_cache::cpu_load_origin::main);
            worker.join();
            for (const auto & req : reqs) {
                EC_CHECK(memcmp(cache.kind_slot_ptr(llama_expert_cache::GATE_UP, req.slot),
                                contents[0].data() + offsets[req.expert], 64) == 0);
                EC_CHECK(memcmp(cache.kind_slot_ptr(llama_expert_cache::DOWN, req.slot),
                                contents[1].data() + offsets[req.expert], 64) == 0);
            }
            if (direct && !cache.odirect_active()) {
                printf("test_split_file_reads: O_DIRECT unavailable, checked buffered fallback\n");
            }
        }
    }
    printf("test_split_file_reads: PASS\n");
}

static void test_expert_cache_model_load_and_save() {
    const char * model_path = getenv("LLAMA_TEST_MODEL");
    if (!model_path) {
        printf("test_expert_cache_model_load_and_save: SKIP (set LLAMA_TEST_MODEL to enable)\n");
        return;
    }

    llama_backend_init();

    llama_model_params mparams      = llama_model_default_params();
    mparams.expert_cache_enabled    = true;
    mparams.expert_cache_size_bytes = 512ULL * 1024 * 1024;
    mparams.n_gpu_layers            = 0;

    llama_model * model = llama_model_load_from_file(model_path, mparams);
    EC_CHECK(model && model->expert_cache);

    const auto & tmap = llama_internal_get_tensor_map(model);

    const ggml_tensor * t_expert = nullptr;
    const ggml_tensor * t_norm   = nullptr;
    for (const auto & kv : tmap) {
        if (is_expert_weight_tensor(kv.first.c_str())) {
            t_expert = kv.second;
            EC_CHECK(t_expert->buffer == nullptr);
        }
        if (kv.first == "blk.0.attn_norm.weight") {
            t_norm = kv.second;
        }
    }

    EC_CHECK(t_expert != nullptr);
    EC_CHECK(t_norm != nullptr && t_norm->data != nullptr);

    std::string export_path = (std::filesystem::temp_directory_path() / "test-cached-export-XXXXXX").string();
    const int   fd          = mkstemp(export_path.data());
    EC_CHECK(fd >= 0);
    const char contents[] = "existing file";
    EC_CHECK(write(fd, contents, sizeof(contents)) == (int64_t) sizeof(contents));
    close(fd);
    llama_model_save_to_file(model, export_path.c_str());
    const int saved_fd = open(export_path.c_str(), O_RDONLY);
    EC_CHECK(saved_fd >= 0);
    EC_CHECK(lseek(saved_fd, 0, SEEK_END) == sizeof(contents));
    EC_CHECK(lseek(saved_fd, 0, SEEK_SET) == 0);
    char saved[sizeof(contents)];
    EC_CHECK(read(saved_fd, saved, sizeof(saved)) == sizeof(saved));
    EC_CHECK(memcmp(saved, contents, sizeof(contents)) == 0);
    close(saved_fd);
    EC_CHECK(unlink(export_path.c_str()) == 0);
    llama_model_save_to_file(model, export_path.c_str());
    EC_CHECK(access(export_path.c_str(), F_OK) != 0);

    // Model tensors must remain valid until the cache worker exits.
    auto * cache = model->expert_cache.get();
    cache->stop_worker();
    cache->worker_shutdown_.store(false);
    uint8_t expected[32];
    ggml_backend_tensor_get(t_norm, expected, 0, sizeof(expected));
    cache->worker_thread_ = std::thread([&] {
        std::unique_lock<std::mutex> lock(cache->worker_mu_);
        EC_CHECK(
            cache->worker_cv_.wait_for(lock, std::chrono::seconds(10), [&] { return cache->worker_shutdown_.load(); }));
        uint8_t actual[sizeof(expected)];
        ggml_backend_tensor_get(t_norm, actual, 0, sizeof(actual));
        EC_CHECK(memcmp(actual, expected, sizeof(actual)) == 0);
    });
    llama_model_free(model);

    mparams.expert_cache_size_bytes = 1;
    for (auto mode : { LLAMA_EXPERT_CACHE_MODE_LAYERED, LLAMA_EXPERT_CACHE_MODE_GLOBAL }) {
        mparams.expert_cache_mode = mode;
        model                     = llama_model_load_from_file(model_path, mparams);
        EC_CHECK(model == nullptr);
    }
    llama_backend_free();
    printf("test_expert_cache_model_load_and_save: PASS\n");
}

static void test_kind_first_layout() {
    llama_expert_cache  cache;
    std::vector<size_t> kind_sizes(llama_expert_cache::N_KINDS, 0);
    kind_sizes[llama_expert_cache::GATE_UP] = 100;
    kind_sizes[llama_expert_cache::DOWN]    = 80;
    cache.init_kind_first(3, kind_sizes);

    uint8_t * p0 = cache.kind_slot_ptr(llama_expert_cache::GATE_UP, 0);
    uint8_t * p1 = cache.kind_slot_ptr(llama_expert_cache::GATE_UP, 1);
    uint8_t * p2 = cache.kind_slot_ptr(llama_expert_cache::GATE_UP, 2);
    (void) p0;
    (void) p1;
    (void) p2;
    assert(p0 == cache.ddr_pool.data() + 0);
    assert(p1 == cache.ddr_pool.data() + 100);
    assert(p2 == cache.ddr_pool.data() + 200);

    uint8_t * q0 = cache.kind_slot_ptr(llama_expert_cache::DOWN, 0);
    uint8_t * q1 = cache.kind_slot_ptr(llama_expert_cache::DOWN, 1);
    (void) q0;
    (void) q1;
    assert(q0 == cache.ddr_pool.data() + 300);
    assert(q1 == cache.ddr_pool.data() + 380);

    assert(cache.expert_sz == 180);
    printf("test_kind_first_layout: PASS\n");
}

static void test_pool_tensors_shape() {
    llama_expert_cache  cache;
    std::vector<size_t> kind_sizes(llama_expert_cache::N_KINDS, 0);
    kind_sizes[llama_expert_cache::GATE_UP] = 400;
    kind_sizes[llama_expert_cache::DOWN]    = 320;
    cache.init_kind_first(3, kind_sizes);

    std::array<std::pair<int64_t, int64_t>, llama_expert_cache::N_KINDS> kind_ne{};
    kind_ne[llama_expert_cache::GATE_UP] = { 10, 10 };
    kind_ne[llama_expert_cache::DOWN]    = { 8, 10 };
    std::array<ggml_type, llama_expert_cache::N_KINDS> kind_dtype{};
    kind_dtype[llama_expert_cache::GATE_UP] = GGML_TYPE_F32;
    kind_dtype[llama_expert_cache::DOWN]    = GGML_TYPE_F32;
    cache.create_pool_tensors(kind_ne, kind_dtype);

    const ggml_tensor * tgu = cache.pool_tensor(llama_expert_cache::GATE_UP);
    EC_CHECK(tgu != nullptr && tgu->buffer != nullptr);
    EC_CHECK(tgu->ne[0] == 10 && tgu->ne[1] == 10 && tgu->ne[2] == 3);
    EC_CHECK((uint8_t *) tgu->data == cache.ddr_pool.data());

    const ggml_tensor * td = cache.pool_tensor(llama_expert_cache::DOWN);
    EC_CHECK(td != nullptr && td->buffer == tgu->buffer);
    EC_CHECK(td->ne[0] == 8 && td->ne[1] == 10 && td->ne[2] == 3);
    EC_CHECK((uint8_t *) td->data == cache.ddr_pool.data() + 1200);

    EC_CHECK(cache.pool_tensor(llama_expert_cache::GATE) == nullptr);
    printf("test_pool_tensors_shape: PASS\n");
}

static void test_id_to_slot_fn() {
    llama_expert_cache  cache;
    std::vector<size_t> kind_sizes(llama_expert_cache::N_KINDS, 0);
    kind_sizes[llama_expert_cache::GATE_UP] = 64;
    kind_sizes[llama_expert_cache::DOWN]    = 64;
    cache.init_kind_first(4, kind_sizes);

    std::vector<llama_expert_cache::expert_slice_loc> locs(1 * llama_expert_cache::N_KINDS * 4, { 0, 0 });
    cache.build_offset_map(1, 4, 1, locs);

    int32_t s = cache.claim_slot(0, 2);
    assert(s >= 0);
    cache.mark_occupied(s);
    int32_t s0 = cache.claim_slot(0, 0);
    assert(s0 >= 0);
    cache.mark_occupied(s0);

    // Pinned slots must survive eviction.
    int32_t pinned[] = { s, s0 };
    int     evicted  = cache.evict_lfu_excluding(pinned, 2, UINT32_MAX);
    if (evicted != -1) {
        assert(false && "all occupied slots are pinned");
    }

    printf("test_id_to_slot_fn: PASS\n");
}

static uint64_t test_sum_slots(const std::vector<uint32_t> & slots) {
    uint64_t total = 0;
    for (uint32_t s : slots) {
        total += s;
    }
    return total;
}

static void test_htp_group_plan_qwen_2gb_balanced() {
    std::vector<uint32_t> base_slots(40, 16);
    const uint32_t        floor_per_layer = 8;
    const uint32_t        leftover        = 2;
    const size_t          slot_bytes      = 3342336;
    const size_t          mbuf            = 321 * slot_bytes;
    const uint32_t        prefill_floor   = 136;

    auto plan = llama_expert_cache::plan_htp_layered_groups(base_slots,
                                                            std::vector<uint32_t>(base_slots.size(), floor_per_layer),
                                                            leftover, slot_bytes, mbuf, prefill_floor);

    EC_CHECK(plan.error.empty());
    EC_CHECK(plan.groups.size() == 2);
    EC_CHECK(plan.groups[0].first_l == 0);
    EC_CHECK(plan.groups[0].last_l == 19);
    EC_CHECK(plan.groups[0].n_slots == 321);
    EC_CHECK(plan.groups[1].first_l == 20);
    EC_CHECK(plan.groups[1].last_l == 39);
    EC_CHECK(plan.groups[1].n_slots == 321);
    EC_CHECK(test_sum_slots(plan.slots_per_layer) == 642);
    for (uint32_t l = 0; l < 40; ++l) {
        EC_CHECK(plan.slots_per_layer[l] >= 16);
    }

    printf("test_htp_group_plan_qwen_2gb_balanced: PASS\n");
}

static void test_htp_group_plan_trims_above_floor() {
    std::vector<uint32_t> base_slots(40, 16);
    const uint32_t        floor_per_layer = 8;
    const uint32_t        leftover        = 3;
    const size_t          slot_bytes      = 3342336;
    const size_t          mbuf            = 321 * slot_bytes;
    const uint32_t        prefill_floor   = 256;

    auto plan = llama_expert_cache::plan_htp_layered_groups(base_slots,
                                                            std::vector<uint32_t>(base_slots.size(), floor_per_layer),
                                                            leftover, slot_bytes, mbuf, prefill_floor);

    EC_CHECK(plan.error.empty());
    EC_CHECK(plan.groups.size() == 2);
    EC_CHECK(test_sum_slots(plan.slots_per_layer) <= 642);
    for (uint32_t s : plan.slots_per_layer) {
        EC_CHECK(s == 0 || s >= floor_per_layer);
    }
    for (const auto & g : plan.groups) {
        EC_CHECK(g.n_slots >= prefill_floor);
        EC_CHECK((uint64_t) g.n_slots * slot_bytes <= mbuf);
    }

    printf("test_htp_group_plan_trims_above_floor: PASS\n");
}

static void test_htp_group_plan_rejects_impossible_floor() {
    std::vector<uint32_t> base_slots(4, 8);
    const uint32_t        floor_per_layer = 8;
    const uint32_t        leftover        = 0;
    const size_t          slot_bytes      = 1024;
    const size_t          mbuf            = 16 * slot_bytes;
    const uint32_t        prefill_floor   = 20;

    auto plan = llama_expert_cache::plan_htp_layered_groups(base_slots,
                                                            std::vector<uint32_t>(base_slots.size(), floor_per_layer),
                                                            leftover, slot_bytes, mbuf, prefill_floor);

    EC_CHECK(!plan.error.empty());
    EC_CHECK(plan.groups.empty());

    printf("test_htp_group_plan_rejects_impossible_floor: PASS\n");
}

static void test_htp_minimum_groups() {
    auto check = [](const std::vector<uint32_t> & floors, uint32_t capacity, uint32_t prefill, uint32_t expected) {
        auto plan = llama_expert_cache::minimum_htp_layered_groups(floors, 1024, capacity * 1024, prefill);
        EC_CHECK(plan.error.empty());
        EC_CHECK(test_sum_slots(plan.slots_per_layer) == expected);
        for (size_t l = 0; l < floors.size(); ++l) {
            EC_CHECK(plan.slots_per_layer[l] >= floors[l]);
            EC_CHECK(floors[l] != 0 || plan.slots_per_layer[l] == 0);
        }
        uint32_t slot_base = 0;
        for (const auto & group : plan.groups) {
            EC_CHECK(group.n_slots >= prefill && group.n_slots <= capacity);
            EC_CHECK(group.slot_base == slot_base);
            uint32_t group_slots = 0;
            for (uint32_t l = group.first_l; l <= group.last_l; ++l) {
                group_slots += plan.slots_per_layer[l];
            }
            EC_CHECK(group_slots == group.n_slots);
            slot_base += group.n_slots;
        }
        EC_CHECK(slot_base == expected);
        auto explicit_plan = llama_expert_cache::plan_htp_layered_groups(plan.slots_per_layer, plan.slots_per_layer, 0,
                                                                         1024, capacity * 1024, prefill);
        EC_CHECK(explicit_plan.error.empty());
        EC_CHECK(test_sum_slots(explicit_plan.slots_per_layer) == expected);
    };
    check({ 2, 0, 4, 0, 2, 0, 4 }, 8, 8, 16);
    check({ 4, 4, 4 }, 8, 6, 14);
    check({ 1, 5, 1, 5 }, 6, 4, 12);
    check(std::vector<uint32_t>(30, 8), 200, 128, 256);
    check(std::vector<uint32_t>(40, 8), 321, 136, 320);
    EC_CHECK(!llama_expert_cache::minimum_htp_layered_groups({ 2, 9 }, 1024, 8 * 1024, 4).error.empty());
    EC_CHECK(!llama_expert_cache::minimum_htp_layered_groups({ 2, 2 }, 1024, 8 * 1024 - 1, 8).error.empty());
    printf("test_htp_minimum_groups: PASS\n");
}

static void test_sparse_mul_mat_id() {
    for (int n_tokens : { 1, 64 }) {
        ggml_context * ctx = ggml_init({ 1024 * 1024, nullptr, false });
        EC_CHECK(ctx);
        ggml_tensor * a     = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 4, 8);
        ggml_tensor * b     = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 2, n_tokens);
        ggml_tensor * ids   = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, n_tokens);
        ggml_tensor * out   = ggml_mul_mat_id(ctx, a, b, ids);
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);
        ggml_build_forward_expand(graph, out);
        for (int i = 0; i < 32 * 4 * 8; ++i) {
            ((float *) a->data)[i] = float(i / (32 * 4) + 1);
        }
        ggml_set_f32(b, 2.0f);
        for (int t = 0; t < n_tokens; ++t) {
            ((int32_t *) ids->data)[2 * t]     = t % 7;
            ((int32_t *) ids->data)[2 * t + 1] = 7;
        }

        for (int n_threads : { 1, 4 }) {
            ggml_mul_mat_id_set_sparse(out, false);
            EC_CHECK(!ggml_mul_mat_id_get_sparse(out));
            EC_CHECK(ggml_prec_set_acc(out, GGML_PREC_UNDEFINED));
            const size_t dense_size = ggml_graph_plan(graph, n_threads, nullptr).work_size;
            EC_CHECK(ggml_prec_set_acc(out, GGML_PREC_F32));
            EC_CHECK(ggml_graph_plan(graph, n_threads, nullptr).work_size == dense_size);
            ggml_mul_mat_id_set_sparse(out, true);
            EC_CHECK(ggml_mul_mat_id_get_sparse(out));
            EC_CHECK(out->op_params[0] == GGML_PREC_F32);
            ggml_cplan plan = ggml_graph_plan(graph, n_threads, nullptr);
            EC_CHECK(n_tokens == 1 ? plan.work_size < dense_size : plan.work_size <= dense_size + 256);
            EC_CHECK(ggml_prec_set_acc(out, GGML_PREC_UNDEFINED));
            EC_CHECK(ggml_mul_mat_id_get_sparse(out));
            EC_CHECK(ggml_graph_plan(graph, n_threads, nullptr).work_size == plan.work_size);
            std::vector<uint8_t> work(plan.work_size);
            plan.work_data = work.data();
            EC_CHECK(ggml_graph_compute(graph, &plan) == GGML_STATUS_SUCCESS);
            for (int t = 0; t < n_tokens; ++t) {
                for (int r = 0; r < 4; ++r) {
                    EC_CHECK(((float *) out->data)[8 * t + r] == 64.0f * (t % 7 + 1));
                    EC_CHECK(((float *) out->data)[8 * t + 4 + r] == 512.0f);
                }
            }
        }
        ggml_free(ctx);
    }
    printf("test_sparse_mul_mat_id: PASS\n");
}

static void test_sparse_mul_mat_id_with_scale() {
    for (ggml_type type : { GGML_TYPE_F32, GGML_TYPE_Q4_0, GGML_TYPE_F8_E4M3, GGML_TYPE_NVFP4 }) {
        for (int granularity : { 0, 1, 2 }) {
            for (bool strided : { false, true }) {
                std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_LLAMA, llama_model_default_params()));
                model->hparams.n_layer_all = 1;
                llm_graph_result    result(128);
                llama_adapter_loras loras;
                llm_graph_params    params = {};
                params.model               = model.get();
                params.loras               = &loras;
                params.res                 = &result;
                params.w_scales            = &model->w_scales;
                llm_graph_context builder(params);
                auto *            ctx      = result.get_ctx();
                auto *            a        = ggml_new_tensor_3d(ctx, type, 256, 4, 8);
                auto *            pool     = ggml_new_tensor_3d(ctx, type, 256, 4, 4);
                auto *            b        = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 2, 2);
                auto *            ids      = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 2);
                auto *            slots    = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 2);
                const int64_t     channels = granularity == 2 ? 4 : 1;
                const int64_t     experts  = granularity == 0 ? 1 : 8;
                auto * storage = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, channels * (strided ? 2 : 1), experts);
                auto * scales =
                    strided ? ggml_view_2d(ctx, storage, channels, experts, storage->nb[1], sizeof(float)) : storage;
                if (granularity == 1) {
                    scales = ggml_permute(ctx, scales, 1, 0, 2, 3);
                }
                model->w_scales.scale.emplace(a, scales);
                auto * input_scales = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
                model->w_scales.input_scale.emplace(a, input_scales);
                model->prec_policy.set_acc(a, GGML_PREC_F32);
                auto * reference = builder.build_lora_mm_id(a, b, ids);
                auto * cached    = builder.build_lora_mm_id(a, b, ids, slots, false, pool);
                EC_CHECK(cached->op == GGML_OP_MUL_MAT_ID && cached->src[0] == pool && cached->src[2] == slots);
                EC_CHECK(ggml_prec_get_acc(cached) == GGML_PREC_F32 && ggml_mul_mat_id_get_sparse(cached));
                ggml_backend_t        backend = ggml_backend_cpu_init();
                ggml_backend_buffer_t buffer  = ggml_backend_alloc_ctx_tensors(ctx, backend);
                EC_CHECK(backend && buffer);
                std::vector<float> weights(256 * 4 * 8);
                for (size_t i = 0; i < weights.size(); ++i) {
                    weights[i] = 0.25f * std::sin((float) i);
                }
                std::vector<uint8_t> quantized(ggml_nbytes(a));
                ggml_quantize_chunk(type, weights.data(), quantized.data(), 0, 4 * 8, 256, nullptr);
                ggml_backend_tensor_set(a, quantized.data(), 0, quantized.size());
                const int32_t logical[]  = { 7, 3, 3, 5 };
                const int32_t resident[] = { 2, 0, 0, 1 };
                ggml_backend_tensor_set(ids, logical, 0, sizeof(logical));
                ggml_backend_tensor_set(slots, resident, 0, sizeof(resident));
                for (int i = 0; i < 4; ++i) {
                    ggml_backend_tensor_set(pool, quantized.data() + logical[i] * a->nb[2], resident[i] * pool->nb[2],
                                            pool->nb[2]);
                }
                std::vector<float> inputs(256 * 2 * 2, 0.125f);
                ggml_backend_tensor_set(b, inputs.data(), 0, inputs.size() * sizeof(float));
                std::vector<float> scale_data(ggml_nelements(storage), NAN);
                for (int64_t e = 0; e < experts; ++e) {
                    for (int64_t c = 0; c < channels; ++c) {
                        scale_data[e * storage->ne[0] + (strided ? c + 1 : c)] = 0.5f + 0.125f * (float) (e + c);
                    }
                }
                ggml_backend_tensor_set(storage, scale_data.data(), 0, scale_data.size() * sizeof(float));
                const float input_data[] = { 0.5f, 1, 1.5f, 2, 2.5f, 3, 3.5f, 4 };
                ggml_backend_tensor_set(input_scales, input_data, 0, sizeof(input_data));
                auto * graph = ggml_new_graph_custom(ctx, 128, false);
                ggml_build_forward_expand(graph, reference);
                ggml_build_forward_expand(graph, cached);
                EC_CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
                std::vector<float> expected(ggml_nelements(reference)), actual(expected.size());
                ggml_backend_tensor_get(reference, expected.data(), 0, expected.size() * sizeof(float));
                ggml_backend_tensor_get(cached, actual.data(), 0, actual.size() * sizeof(float));
                for (size_t i = 0; i < expected.size(); ++i) {
                    EC_CHECK(std::isfinite(actual[i]) && actual[i] == expected[i]);
                }
                const int32_t skipped[] = { 2, -1, -1, 1 };
                ggml_backend_tensor_set(slots, skipped, 0, sizeof(skipped));
                EC_CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
                ggml_backend_tensor_get(cached, actual.data(), 0, actual.size() * sizeof(float));
                for (size_t i = 0; i < actual.size(); ++i) {
                    EC_CHECK(actual[i] == (skipped[i / 4] < 0 ? 0.0f : expected[i]));
                }
                std::vector<float> remapped(4);
                ggml_backend_tensor_get(cached->src[4], remapped.data(), 0, remapped.size() * sizeof(float));
                EC_CHECK(remapped[1] == input_data[5] && remapped[2] == input_data[7]);
                ggml_backend_buffer_free(buffer);
                ggml_backend_free(backend);
            }
        }
    }
    printf("test_sparse_mul_mat_id_with_scale: PASS (24 scale/type/layout configurations, 48 evaluations)\n");
}

static void test_router_precision_policy() {
    std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_LLAMA, llama_model_default_params()));
    model->hparams.n_layer_all = 1;
    llm_graph_result    result(64);
    llama_adapter_loras loras;
    llm_graph_params    params = {};
    params.model               = model.get();
    params.loras               = &loras;
    params.res                 = &result;
    ggml_tensor * logits       = nullptr;
    params.cb                  = [&](const llama_ubatch &, ggml_tensor * tensor, const char * name, int) {
        if (strcmp(name, "ffn_moe_logits") == 0) {
            logits = tensor;
        }
    };
    llm_graph_context builder(params);
    auto *            ctx   = result.get_ctx();
    auto *            gate  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 4);
    auto *            input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 1);
    moe_router_desc   rd;
    rd.gate_inp          = gate;
    rd.gating_op         = LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX;
    rd.expert_router_f32 = true;
    rd.n_expert          = 4;
    rd.n_expert_used     = 2;
    rd.arch              = LLM_ARCH_LLAMA;
    for (auto precision : { GGML_PREC_UNDEFINED, GGML_PREC_F32, GGML_PREC_BF16 }) {
        model->prec_policy.set_acc(gate, precision);
        ggml_tensor * selection = nullptr;
        ggml_tensor * probs     = nullptr;
        builder.build_moe_router(ctx, input, rd, 0, &selection, &probs);
        EC_CHECK(logits && ggml_prec_get_acc(logits) == precision);
    }
    printf("test_router_precision_policy: PASS\n");
}

static void test_capacity_floor_check() {
    llama_expert_cache  cache;
    std::vector<size_t> kind_sizes(llama_expert_cache::N_KINDS, 0);
    kind_sizes[llama_expert_cache::GATE_UP] = 64;
    cache.init_kind_first(4, kind_sizes);
    cache.n_expert_used_ = 8;
    // 4 slots < 2*8=16 -> insufficient
    assert(!cache.init_check_capacity_floor() && "4 slots should be insufficient for 8 experts");

    llama_expert_cache cache2;
    cache2.init_kind_first(16, kind_sizes);
    cache2.n_expert_used_ = 8;
    assert(cache2.init_check_capacity_floor() && "16 slots should be sufficient for 8 experts");
    printf("test_capacity_floor_check: PASS\n");
}

static void test_reroute_preserves_in_flight_slots() {
    llama_expert_cache cache;
    cache.decode_prefetch_force_cache_hit_ = false;
    cache.init_kind_first(2, { 64, 0, 0, 0 });
    cache.build_offset_map(1, 4, 2, {});
    cache.mark_occupied(cache.claim_slot(0, 0));
    cache.mark_occupied(cache.claim_slot(0, 1));

    int32_t     ids[]                      = { 0, 1 };
    ggml_tensor selected                   = {};
    selected.ne[0]                         = 2;
    selected.ne[1]                         = 1;
    selected.data                          = ids;
    llama_expert_cache::op_params_t first  = { &cache, 0 };
    llama_expert_cache::op_params_t second = { &cache, 0 };
    llama_expert_cache::reroute_fn(&selected, nullptr, nullptr, 0, 1, &first);
    llama_expert_cache::reroute_fn(&selected, nullptr, nullptr, 0, 1, &second);
    EC_CHECK(cache.evict_lfu_excluding(nullptr, 0) == -1);
    EC_CHECK(cache.evict_lru_excluding(nullptr, 0) == -1);
    cache.release_slots(first);
    EC_CHECK(cache.evict_lfu_excluding(nullptr, 0) == -1);
    cache.release_slots(second);
    EC_CHECK(cache.evict_lfu_excluding(nullptr, 0) >= 0);
    printf("test_reroute_preserves_in_flight_slots: PASS\n");
}

static void test_model_minimum_cache() {
    const char * path = getenv("LLAMA_TEST_MODEL");
    if (!path) {
        printf("test_model_minimum_cache: SKIP (set LLAMA_TEST_MODEL to enable)\n");
        return;
    }
    llama_model_params mp     = llama_model_default_params();
    mp.n_gpu_layers           = 0;
    mp.use_extra_bufts        = false;
    mp.expert_cache_enabled   = true;
    mp.expert_cache_threshold = 0;
    for (auto mode : { LLAMA_EXPERT_CACHE_MODE_LAYERED, LLAMA_EXPERT_CACHE_MODE_GLOBAL_NONUNIFORM }) {
        mp.expert_cache_mode = mode;
        llama_model * model  = llama_model_load_from_file(path, mp);
        EC_CHECK(model && model->expert_cache);
        uint32_t minimum = model->hparams.n_expert_used_max();
        if (mode == LLAMA_EXPERT_CACHE_MODE_LAYERED) {
            minimum = 0;
            for (uint32_t l = 0; l < model->hparams.n_layer(); ++l) {
                if (model->layers[l].ffn_down_exps) {
                    minimum += model->hparams.n_expert_used(l);
                }
            }
        }
        if (!model->hparams.causal_attn) {
            minimum = std::max(minimum, model->hparams.n_expert);
        }
        EC_CHECK(model->expert_cache->n_slots_ == minimum);
        EC_CHECK(llama_model_expert_cache_size(model) == minimum * model->expert_cache->expert_sz);
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx                = 64;
        cp.n_batch = cp.n_ubatch = 8;
        cp.n_threads = cp.n_threads_batch = 1;
        cp.flash_attn_type                = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        llama_context * ctx               = llama_init_from_model(model, cp);
        EC_CHECK(ctx);
        llama_batch_ext * batch = llama_batch_ext_init(ctx);
        EC_CHECK(batch);
        for (llama_pos pos = 0; pos < 4; ++pos) {
            const int32_t idx = llama_batch_ext_add_token(batch, 0, pos + 1);
            EC_CHECK(idx >= 0);
            EC_CHECK(llama_batch_ext_set_pos(batch, idx, &pos));
            EC_CHECK(llama_batch_ext_set_output_logits(batch, idx, pos == 3));
        }
        EC_CHECK(llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) == 0);
        llama_batch_ext_free(batch);
        const float * logits = llama_get_logits_ith(ctx, -1);
        for (int i = 0; i < llama_vocab_n_tokens(llama_model_get_vocab(model)); ++i) {
            EC_CHECK(std::isfinite(logits[i]));
        }
        llama_free(ctx);
        llama_model_free(model);
    }
    printf("test_model_minimum_cache: PASS\n");
}

static void test_cache_no_alloc() {
    const char * path = getenv("LLAMA_TEST_MODEL");
    if (!path) {
        printf("test_cache_no_alloc: SKIP (set LLAMA_TEST_MODEL to enable)\n");
        return;
    }
    const bool htp = ggml_backend_dev_by_name("HTP0") != nullptr;
    for (int backend = 0; backend < (htp ? 2 : 1); ++backend) {
        for (auto mode : { LLAMA_EXPERT_CACHE_MODE_LAYERED, LLAMA_EXPERT_CACHE_MODE_GLOBAL_NONUNIFORM }) {
            llama_model_params mp   = llama_model_default_params();
            mp.n_gpu_layers         = backend ? 99 : 0;
            mp.use_extra_bufts      = false;
            mp.no_alloc             = true;
            mp.load_mode            = LLAMA_LOAD_MODE_NONE;
            mp.expert_cache_enabled = true;
            mp.expert_cache_mode    = mode;
            llama_model * model     = llama_model_load_from_file(path, mp);
            EC_CHECK(model && model->expert_cache);
            auto * cache = model->expert_cache.get();
            EC_CHECK(!cache->worker_thread_.joinable());
            EC_CHECK(cache->ddr_pool.empty());
            EC_CHECK(ggml_backend_buffer_get_size(cache->pool_buf_) == 0);
            for (const auto & entry : llama_internal_get_tensor_map(model)) {
                EC_CHECK(entry.second->data == nullptr);
            }
            for (uint32_t layer = 0; layer < model->hparams.n_layer(); ++layer) {
                for (int kind = 0; kind < llama_expert_cache::N_KINDS; ++kind) {
                    auto * tensor = cache->pool_tensor(layer, (llama_expert_cache::tensor_kind) kind);
                    if (tensor) {
                        EC_CHECK(tensor->data == nullptr && tensor->buffer);
                        EC_CHECK(ggml_backend_buffer_get_size(tensor->buffer) == 0);
                        EC_CHECK(ggml_backend_buffer_get_usage(tensor->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                    }
                }
            }
            llama_context_params cp = llama_context_default_params();
            cp.n_ctx                = 64;
            cp.n_batch = cp.n_ubatch = 8;
            cp.flash_attn_type       = LLAMA_FLASH_ATTN_TYPE_DISABLED;
            llama_context * ctx      = llama_init_from_model(model, cp);
            EC_CHECK(ctx);
            size_t model_bytes = 0;
            for (const auto & entry : llama_get_memory_breakdown(ctx)) {
                model_bytes += entry.second.model;
            }
            EC_CHECK(model_bytes >= cache->n_slots_ * cache->expert_sz);
            llama_free(ctx);
            llama_model_free(model);
        }
    }
    printf("test_cache_no_alloc: PASS\n");
}

static void test_cache_runtime_accounting() {
    const test_env_override lookahead("LLAMA_EXPERT_CACHE_DECODE_PREFETCH_LOOKAHEAD");
    const test_env_override depth("LLAMA_EXPERT_CACHE_BATCH_DEPTH");
    const test_env_override async("LLAMA_EXPERT_CACHE_HTP_ASYNC");
    const test_env_override force("LLAMA_EXPERT_CACHE_DECODE_PREFETCH_FORCE_CACHE_HIT");
    const test_env_override percentage("LLAMA_EXPERT_CACHE_READ_PERCENTAGE");
    const test_env_override mode("LLAMA_EXPERT_CACHE_MODE");
    expert_cache_fixture    fixture;
    std::string             path = fixture.path;
    if (const char * directory = getenv("LLAMA_TEST_ODIRECT_DIR")) {
        path = (std::filesystem::path(directory) / (fixture.directory.filename().string() + ".gguf")).string();
        std::filesystem::copy_file(fixture.path, path);
    }
    llama_backend_init();
    auto mp                 = llama_model_default_params();
    mp.n_gpu_layers         = 0;
    mp.use_extra_bufts      = false;
    mp.expert_cache_enabled = true;
    mp.no_alloc             = true;
    auto measure            = [&](bool direct, int ahead, int batch_depth) {
        test_set_env("LLAMA_EXPERT_CACHE_DECODE_PREFETCH_LOOKAHEAD", std::to_string(ahead).c_str());
        test_set_env("LLAMA_EXPERT_CACHE_BATCH_DEPTH", std::to_string(batch_depth).c_str());
        mp.load_mode = direct ? LLAMA_LOAD_MODE_DIRECT_IO : LLAMA_LOAD_MODE_NONE;
        auto * model = llama_model_load_from_file(path.c_str(), mp);
        EC_CHECK(model && model->expert_cache);
        size_t total = 0;
        for (const auto & entry : model->expert_cache->memory_breakdown()) {
            total += entry.second;
        }
        llama_model_free(model);
        return total;
    };
    const size_t plain  = measure(false, 0, 1);
    const size_t direct = measure(true, 0, 1);
    EC_CHECK(direct >= plain);
    if (getenv("LLAMA_TEST_ODIRECT_DIR")) {
        EC_CHECK(direct > plain);
    }
    const size_t predicted = measure(false, 1, 1);
    EC_CHECK(predicted > plain);
    EC_CHECK(measure(true, 0, 4) >= direct);
    if (direct > plain) {
        EC_CHECK(measure(true, 0, 4) > direct);
    }
    printf("runtime accounting: plain=%zu direct=%zu predictor=%zu\n", plain, direct, predicted);
    for (const char * cache_mode : { "layered", "global-nonuniform" }) {
        test_set_env("LLAMA_EXPERT_CACHE_MODE", cache_mode);
        for (bool direct_io : { false, true }) {
            for (int ahead : { 0, 1 }) {
                for (int batch_depth : { 1, 4 }) {
                    const size_t expected = measure(direct_io, ahead, batch_depth);
                    for (int async_value : { 0, 1 }) {
                        for (int force_value : { 0, 1 }) {
                            for (int read_value : { 0, 37 }) {
                                test_set_env("LLAMA_EXPERT_CACHE_HTP_ASYNC", std::to_string(async_value).c_str());
                                test_set_env("LLAMA_EXPERT_CACHE_DECODE_PREFETCH_FORCE_CACHE_HIT",
                                             std::to_string(force_value).c_str());
                                test_set_env("LLAMA_EXPERT_CACHE_READ_PERCENTAGE", std::to_string(read_value).c_str());
                                EC_CHECK(measure(direct_io, ahead, batch_depth) == expected);
                                printf(
                                    "accounted mode=%s direct=%d predictor=%d depth=%d async=%d force=%d percentage=%d "
                                    "bytes=%zu\n",
                                    cache_mode, direct_io, ahead, batch_depth, async_value, force_value, read_value,
                                    expected);
                            }
                        }
                    }
                }
            }
        }
    }
    test_set_env("LLAMA_EXPERT_CACHE_MODE", "layered");
    test_set_env("LLAMA_EXPERT_CACHE_READ_PERCENTAGE", "0");
    test_set_env("LLAMA_EXPERT_CACHE_DECODE_PREFETCH_LOOKAHEAD", "1");
    test_set_env("LLAMA_EXPERT_CACHE_BATCH_DEPTH", "4");
    mp.no_alloc  = false;
    mp.load_mode = LLAMA_LOAD_MODE_DIRECT_IO;
    auto * model = llama_model_load_from_file(path.c_str(), mp);
    EC_CHECK(model && model->expert_cache);
    auto & cache = *model->expert_cache;
    cache.stop_worker();
    if (cache.odirect_active()) {
        for (uint32_t layer = 0; layer < cache.n_layer_; ++layer) {
            for (uint32_t expert = 0; expert < cache.n_expert_; ++expert) {
                const int32_t slot = cache.claim_slot_in_range(layer, expert, cache.layer_slot_base_[layer],
                                                               cache.layer_slot_base_[layer + 1]);
                EC_CHECK(slot >= 0);
                cache.load_expert_sync(slot, layer, expert);
                cache.release(slot);
            }
        }
        size_t actual = 0;
        for (const auto & pool : cache.odirect_->bufs) {
            actual += pool.capacity() * sizeof(ec_odirect::buf);
            for (const auto & buffer : pool) {
                actual += buffer.n;
            }
        }
        EC_CHECK(actual > 0 && actual <= cache.direct_io_memory_);
        printf("direct accounting: allocated=%zu reserved=%zu\n", actual, cache.direct_io_memory_);
    }
    cache.prefetch_init();
    std::vector<int32_t> predicted_ids;
    std::vector<float>   hidden(model->hparams.n_embd, 0.1f);
    size_t               predictor_actual = 0;
    size_t               workspace        = 0;
    for (uint32_t layer = 0; layer < cache.n_layer_; ++layer) {
        cache.worker_predict(layer, hidden.data(), hidden.size(), predicted_ids);
        const auto & graph = cache.worker_pred_graphs_[layer];
        predictor_actual += ggml_get_mem_size(graph.ctx) + ggml_backend_buffer_get_size(graph.buf);
        workspace = std::max(workspace, ggml_graph_plan(graph.gf, 1, nullptr).work_size);
    }
    EC_CHECK(predictor_actual + workspace <= cache.predictor_memory_);
    printf("predictor accounting: allocated=%zu shared_workspace=%zu reserved=%zu\n", predictor_actual, workspace,
           cache.predictor_memory_);
    llama_model_free(model);
    llama_backend_free();
    if (path != fixture.path) {
        std::filesystem::remove(path);
    }
}

static void test_cache_loading_peak() {
    const test_env_override lookahead("LLAMA_EXPERT_CACHE_DECODE_PREFETCH_LOOKAHEAD");
    expert_cache_fixture    fixture(false, 32768);
    std::string             path = fixture.path;
    if (const char * directory = getenv("LLAMA_TEST_ODIRECT_DIR")) {
        path = (std::filesystem::path(directory) / (fixture.directory.filename().string() + ".gguf")).string();
        std::filesystem::copy_file(fixture.path, path);
    }
    llama_backend_init();
    for (bool direct : { false, true }) {
        for (bool predictor : { false, true }) {
            test_set_env("LLAMA_EXPERT_CACHE_DECODE_PREFETCH_LOOKAHEAD", predictor ? "1" : "0");
            auto mp                 = llama_model_default_params();
            mp.n_gpu_layers         = 0;
            mp.use_extra_bufts      = false;
            mp.expert_cache_enabled = true;
            mp.no_alloc             = true;
            mp.load_mode            = direct ? LLAMA_LOAD_MODE_DIRECT_IO : LLAMA_LOAD_MODE_NONE;
            auto * model            = llama_model_load_from_file(path.c_str(), mp);
            EC_CHECK(model && model->expert_cache);
            auto cp    = llama_context_default_params();
            cp.n_ctx   = 64;
            cp.n_batch = cp.n_ubatch = 8;
            cp.flash_attn_type       = LLAMA_FLASH_ATTN_TYPE_DISABLED;
            auto * ctx               = llama_init_from_model(model, cp);
            EC_CHECK(ctx);
            size_t model_bytes = 0;
            for (const auto & entry : model->memory_breakdown()) {
                model_bytes += entry.second;
            }
            for (const auto & entry : model->expert_cache->memory_breakdown()) {
                model_bytes -= entry.second;
            }
            size_t required = 0;
            for (const auto & entry : llama_get_memory_breakdown(ctx)) {
                required += entry.second.total();
            }
            for (const auto & entry : llama_get_memory_breakdown(ctx)) {
                printf("measured component: %s model=%zu context=%zu compute=%zu\n",
                       ggml_backend_buft_name(entry.first), entry.second.model, entry.second.context,
                       entry.second.compute);
            }
            const size_t staging = model->expert_cache->loading_memory_;
            if (model->expert_cache->odirect_active()) {
                EC_CHECK(staging >= ggml_nbytes(model->tok_embd));
                EC_CHECK(required >= model_bytes + staging);
            } else {
                EC_CHECK(staging == 0);
            }
            printf("loading peak: direct=%d predictor=%d model=%zu staging=%zu required=%zu cache=%zu\n", direct,
                   predictor, model_bytes, staging, required, model->expert_cache->budget_bytes_);
            llama_free(ctx);
            llama_model_free(model);
            // Loading, first decode and repeated cleanup must work after measurement.
            mp.no_alloc          = false;
            mp.progress_callback = [](float progress, void *) {
                return progress < 0.5f;
            };
            EC_CHECK(llama_model_load_from_file(path.c_str(), mp) == nullptr);
            mp.progress_callback = nullptr;
            for (int cycle = 0; cycle < 2; ++cycle) {
                model = llama_model_load_from_file(path.c_str(), mp);
                EC_CHECK(model && model->expert_cache);
                ctx = llama_init_from_model(model, cp);
                EC_CHECK(ctx);
                auto * batch = llama_batch_ext_init(ctx);
                EC_CHECK(batch);
                EC_CHECK(llama_batch_ext_add_token(batch, 0, 1) >= 0);
                EC_CHECK(llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) == 0);
                size_t runtime = 0;
                for (const auto & entry : llama_get_memory_breakdown(ctx)) {
                    runtime += entry.second.total();
                }
                printf("loading runtime: direct=%d predictor=%d runtime=%zu measured=%zu\n", direct, predictor, runtime,
                       required);
                for (const auto & entry : llama_get_memory_breakdown(ctx)) {
                    printf("runtime component: %s model=%zu context=%zu compute=%zu\n",
                           ggml_backend_buft_name(entry.first), entry.second.model, entry.second.context,
                           entry.second.compute);
                }
                EC_CHECK(runtime <= required);
                llama_batch_ext_free(batch);
                llama_free(ctx);
                llama_model_free(model);
            }
        }
    }
    llama_backend_free();
    if (path != fixture.path) {
        std::filesystem::remove(path);
    }
}

static void test_cache_fitting() {
    const char * path = getenv("LLAMA_TEST_MODEL");
    if (!path) {
        printf("test_cache_fitting: SKIP (set LLAMA_TEST_MODEL to enable)\n");
        return;
    }
    llama_model_params mp   = llama_model_default_params();
    mp.n_gpu_layers         = 0;
    mp.expert_cache_enabled = true;
    mp.use_extra_bufts      = false;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx                = 64;
    cp.n_batch = cp.n_ubatch = 8;
    cp.flash_attn_type       = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    size_t     margin        = 0;
    const auto status =
        common_fit_params({ path }, &mp, &cp, nullptr, nullptr, &margin, 64, nullptr, GGML_LOG_LEVEL_ERROR);
    EC_CHECK(status == COMMON_PARAMS_FIT_STATUS_SUCCESS);
    EC_CHECK(mp.expert_cache_size_bytes > 0);
    EC_CHECK(cp.n_ctx == 64 && cp.n_ubatch == 8);
    const uint64_t selected = mp.expert_cache_size_bytes;
    EC_CHECK(common_fit_params({ path }, &mp, &cp, nullptr, nullptr, &margin, 64, nullptr, GGML_LOG_LEVEL_ERROR) ==
             COMMON_PARAMS_FIT_STATUS_SUCCESS);
    EC_CHECK(mp.expert_cache_size_bytes == selected);
    margin = INT64_MAX / 4;
    EC_CHECK(common_fit_params({ path }, &mp, &cp, nullptr, nullptr, &margin, 64, nullptr, GGML_LOG_LEVEL_ERROR) ==
             COMMON_PARAMS_FIT_STATUS_FAILURE);
    EC_CHECK(mp.expert_cache_size_bytes == selected && cp.n_ubatch == 8);
    common_params params;
    params.model.path = path;
    params.n_ctx      = 64;
    params.n_batch = params.n_ubatch = 8;
    params.n_gpu_layers              = 0;
    params.expert_cache_enabled      = true;
    params.fit_params_target.assign(llama_max_devices(), margin);
    const auto result = common_init_from_params(params);
    EC_CHECK(!result->model() && !result->context());
    printf("test_cache_fitting: PASS\n");
}

static void test_cache_batch_limit() {
    llama_expert_cache cache;
    cache.n_expert_      = 8;
    cache.n_expert_used_ = 2;
    llama_expert_cache_context state(&cache);
    llama_cparams              params{};
    params.causal_attn = true;
    for (uint32_t slots : { 2, 3, 4, 7, 8, 16 }) {
        cache.n_slots_  = slots;
        params.n_ubatch = 64;
        state.limit_batch(params);
        EC_CHECK(params.n_ubatch == (slots < 8 ? slots / 2 : 64));
    }
    cache.group_n_slots_ = { 8, 6 };
    params.n_ubatch      = 64;
    state.limit_batch(params);
    EC_CHECK(params.n_ubatch == 3);
    params.n_ubatch = 2;
    state.limit_batch(params);
    EC_CHECK(params.n_ubatch == 2);
    params.causal_attn = false;
    params.n_ubatch    = 4;
    bool rejected      = false;
    try {
        state.limit_batch(params);
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    EC_CHECK(rejected);
    cache.group_n_slots_ = { 8, 8 };
    state.limit_batch(params);
    EC_CHECK(params.n_ubatch == 4);
    printf("test_cache_batch_limit: PASS\n");
}

static void test_cache_empty_output() {
    llama_expert_cache cache;
    cache.init_kind_first(2, { 64, 0, 0, 0 });
    cache.build_offset_map(1, 4, 2, {});
    llama_expert_cache_context state(&cache);
    ggml_context *             ctx   = ggml_init({ 1024 * 1024, nullptr, true });
    ggml_cgraph *              graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_tensor *              out   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 2, 0);
    llama_expert_cache_finish_layer(&state, ctx, graph, out, 0);
    EC_CHECK(state.ops[0].pinned.empty());
    ggml_free(ctx);
    printf("test_cache_empty_output: PASS\n");
}

static void test_concurrent_cache_graphs() {
    llama_expert_cache cache;
    cache.decode_prefetch_force_cache_hit_ = false;
    cache.decode_prefetch_lookahead_       = 0;
    cache.init_kind_first(4, { 64, 0, 0, 0 });
    cache.build_offset_map(1, 4, 2, {});
    std::array<std::pair<int64_t, int64_t>, llama_expert_cache::N_KINDS> shape = {};
    std::array<ggml_type, llama_expert_cache::N_KINDS>                   type  = {};
    shape[0]                                                                   = { 4, 4 };
    type[0]                                                                    = GGML_TYPE_F32;
    cache.create_pool_tensors(shape, type);
    for (int e = 0; e < 4; ++e) {
        const int32_t s       = cache.claim_slot(0, e);
        float *       weights = (float *) cache.kind_slot_ptr(llama_expert_cache::GATE_UP, s);
        std::fill(weights, weights + 16, (float) (e + 1));
        cache.mark_occupied(s);
    }

    struct rendezvous {
        std::mutex              mutex;
        std::condition_variable cv;
        int                     arrived = 0;
    } barrier;

    auto run = [&](int first_expert) {
        llama_expert_cache_context state(&cache);
        ggml_context *             ctx     = ggml_init({ 1024 * 1024, nullptr, true });
        ggml_backend_t             backend = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(backend, 1);
        ggml_backend_sched_t sched = ggml_backend_sched_new(&backend, nullptr, 1, 64, false, true);
        ggml_cgraph *        graph = ggml_new_graph_custom(ctx, 64, false);
        ggml_tensor *        ids   = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 1);
        ggml_tensor *        probs = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 1);
        ggml_tensor *        input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 1, 1);
        ggml_set_input(ids);
        ggml_set_input(probs);
        ggml_set_input(input);
        ggml_tensor * routed = llama_expert_cache_reroute_selected(&state, ctx, ids, probs, 0);
        ggml_tensor * ready  = ggml_map_custom1_inplace(
            ctx, routed,
            [](ggml_tensor *, const ggml_tensor *, int, int, void * data) {
                auto &                       gate = *static_cast<rendezvous *>(data);
                std::unique_lock<std::mutex> lock(gate.mutex);
                ++gate.arrived;
                gate.cv.notify_all();
                EC_CHECK(gate.cv.wait_for(lock, std::chrono::seconds(10), [&] { return gate.arrived == 2; }));
            },
            1, &barrier);
        ggml_tensor * slots = llama_expert_cache_slot_ids(&state, ctx, ready, 0);
        ggml_tensor * out   = ggml_mul_mat_id(ctx, cache.pool_tensor(llama_expert_cache::GATE_UP), input, slots);
        llama_expert_cache_finish_layer(&state, ctx, graph, out, 0);
        ggml_set_output(out);
        EC_CHECK(ggml_backend_sched_alloc_graph(sched, graph));
        const int32_t selected[] = { first_expert, first_expert + 1 };
        const float   ones[]     = { 1, 1, 1, 1 };
        auto          set_inputs = [&] {
            ggml_backend_tensor_set(ids, selected, 0, sizeof(selected));
            ggml_backend_tensor_set(probs, ones, 0, sizeof(ones));
            ggml_backend_tensor_set(input, ones, 0, sizeof(ones));
        };
        set_inputs();
        EC_CHECK(llama_expert_cache_graph_compute(&state, sched, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_sched_synchronize(sched);
        float values[8];
        ggml_backend_tensor_get(out, values, 0, sizeof(values));
        for (int i = 0; i < 8; ++i) {
            EC_CHECK(values[i] == 4.0f * (first_expert + i / 4 + 1));
        }
        EC_CHECK(state.ops[0].pinned.empty());

        // Stop after reservation, then verify that error cleanup releases every reader.
        set_inputs();
        ggml_backend_cpu_set_abort_callback(
            backend,
            [](void * data) { return !static_cast<llama_expert_cache_context *>(data)->ops[0].pinned.empty(); },
            &state);
        EC_CHECK(llama_expert_cache_graph_compute(&state, sched, graph) == GGML_STATUS_ABORTED);
        EC_CHECK(state.ops[0].pinned.empty());
        ggml_backend_cpu_set_abort_callback(backend, nullptr, nullptr);
        set_inputs();
        ggml_backend_sched_set_eval_callback(
            sched, [](ggml_tensor * t, bool ask, void * data) { return ask && t == data; }, routed);
        EC_CHECK(llama_expert_cache_graph_compute(&state, sched, graph) == GGML_STATUS_SUCCESS);
        EC_CHECK(state.ops[0].pinned.empty());
        ggml_backend_sched_free(sched);
        ggml_backend_free(backend);
        ggml_free(ctx);
    };
    std::thread first(run, 0);
    std::thread second(run, 2);
    first.join();
    second.join();
    EC_CHECK(barrier.arrived == 2);
    for (uint32_t s = 0; s < cache.n_slots_; ++s) {
        EC_CHECK(cache.slots[s].readers == 0);
    }
    printf("test_concurrent_cache_graphs: PASS\n");
}

static void test_cache_capacity_wait() {
    std::string path = (std::filesystem::temp_directory_path() / "test-cache-wait-XXXXXX").string();
    const int   fd   = mkstemp(path.data());
    EC_CHECK(fd >= 0);
    uint8_t data[4 * 64];
    for (size_t i = 0; i < sizeof(data); ++i) {
        data[i] = (uint8_t) (i / 64);
    }
    EC_CHECK(write(fd, data, sizeof(data)) == sizeof(data));
    llama_expert_cache cache;
    cache.decode_prefetch_force_cache_hit_ = false;
    cache.init_kind_first(2, { 64, 0, 0, 0 });
    EC_CHECK(cache.add_file(fd));
    std::vector<llama_expert_cache::expert_slice_loc> locs(4 * llama_expert_cache::N_KINDS, { 0, 0 });
    for (size_t e = 0; e < 4; ++e) {
        locs[e] = { e * 64, 64 };
    }
    cache.build_offset_map(1, 4, 2, locs);
    for (int round = 0; round < 16; ++round) {
        int32_t     ids[]                     = { 0, 1 };
        ggml_tensor selected                  = {};
        selected.ne[0]                        = 2;
        selected.ne[1]                        = 1;
        selected.data                         = ids;
        llama_expert_cache::op_params_t first = { &cache, 0 };
        llama_expert_cache::reroute_fn(&selected, nullptr, nullptr, 0, 1, &first);
        std::promise<void> started;
        auto               pending = std::async(std::launch::async, [&] {
            int32_t     next_ids[]                 = { 1, 2 };
            ggml_tensor next                       = selected;
            next.data                              = next_ids;
            llama_expert_cache::op_params_t second = { &cache, 0 };
            started.set_value();
            llama_expert_cache::reroute_fn(&next, nullptr, nullptr, 0, 1, &second);
            for (size_t i = 0; i < 2; ++i) {
                const int32_t s = second.slot_ids[i];
                EC_CHECK(cache.kind_slot_ptr(llama_expert_cache::GATE_UP, s)[0] == next_ids[i]);
            }
            cache.release_slots(second);
        });
        EC_CHECK(started.get_future().wait_for(std::chrono::seconds(10)) == std::future_status::ready);
        for (size_t i = 0; i < 2; ++i) {
            EC_CHECK(cache.kind_slot_ptr(llama_expert_cache::GATE_UP, first.slot_ids[i])[0] == ids[i]);
        }
        cache.release_slots(first);
        EC_CHECK(pending.wait_for(std::chrono::seconds(10)) == std::future_status::ready);
        pending.get();
    }
    close(fd);
    unlink(path.c_str());
    printf("test_cache_capacity_wait: PASS\n");
}

static void test_cache_live_cross_range() {
    std::string path = (std::filesystem::temp_directory_path() / "test-cache-ranges-XXXXXX").string();
    const int   fd   = mkstemp(path.data());
    EC_CHECK(fd >= 0);
    uint8_t data[64];
    memset(data, 37, sizeof(data));
    EC_CHECK(write(fd, data, sizeof(data)) == sizeof(data));
    for (bool copy : { false, true }) {
        llama_expert_cache cache;
        cache.stats_                           = true;
        cache.ddr_copy_                        = copy;
        cache.decode_prefetch_force_cache_hit_ = false;
        cache.init_layered(2, { 2, 2 }, { 64, 0, 0, 0, 64, 0, 0, 0 });
        EC_CHECK(cache.add_file(fd));
        std::vector<llama_expert_cache::expert_slice_loc> locs(2 * 4 * llama_expert_cache::N_KINDS, { 0, 0 });
        locs[0] = { 0, 64 };
        cache.build_offset_map(2, 4, 1, locs);

        int32_t source = cache.claim_slot_in_range(0, 0, 2, 4);
        int32_t dest   = cache.claim_slot_in_range(0, 0, 0, 2);
        EC_CHECK(source >= 2 && dest >= 0);
        EC_CHECK(cache.slots[source].state.load() == (uint32_t) llama_expert_slot_state::LOADING);
        cache.release(source);
        EC_CHECK(cache.index_.at(cache.make_key(0, 0)) == (uint32_t) dest);
        cache.release(dest);

        source = cache.claim_slot_in_range(0, 0, 2, 4);
        cache.load_expert_sync(source, 0, 0, false);
        int32_t     ids[]    = { 0, 0 };
        ggml_tensor selected = {};
        selected.ne[0]       = 1;
        selected.ne[1]       = 2;
        selected.ne[2] = selected.ne[3]         = 1;
        selected.data                           = ids;
        llama_expert_cache::op_params_t prefill = { &cache, 0 };
        llama_expert_cache::op_params_t decode  = { &cache, 0 };
        llama_expert_cache::reroute_fn(&selected, nullptr, nullptr, 0, 1, &prefill);
        selected.ne[1] = 1;
        llama_expert_cache::reroute_fn(&selected, nullptr, nullptr, 0, 1, &decode);
        EC_CHECK(cache.stat_ddr_copies.load() == (copy ? 1u : 0u));
        EC_CHECK(cache.stat_hits.load() == (copy ? 1u : 0u));
        EC_CHECK(cache.stat_misses.load() == (copy ? 0u : 1u));
        dest = decode.slot_ids[0];
        EC_CHECK(dest < 2 && prefill.slot_ids[0] == source);
        for (int32_t s : { source, dest }) {
            EC_CHECK(cache.slots[s].readers == 1);
            EC_CHECK(memcmp(cache.kind_slot_ptr(llama_expert_cache::GATE_UP, s), data, sizeof(data)) == 0);
        }
        int32_t slots[2] = { -1, -1 };
        selected.data    = slots;
        selected.ne[1]   = 2;
        llama_expert_cache::id_to_slot_fn(&selected, nullptr, 0, 1, &prefill);
        EC_CHECK(slots[0] == source && slots[1] == source);
        cache.release_slots(prefill);
        cache.release(source);
        EC_CHECK(cache.index_.at(cache.make_key(0, 0)) == (uint32_t) dest);
        cache.release_slots(decode);
    }
    close(fd);
    unlink(path.c_str());
    printf("test_cache_live_cross_range: PASS\n");
}

static void test_reroute_large_microbatch() {
    llama_expert_cache cache;
    cache.decode_prefetch_force_cache_hit_ = false;
    cache.init_kind_first(2, { 64, 0, 0, 0 });
    cache.build_offset_map(1, 8, 1, {});
    cache.mark_occupied(cache.claim_slot(0, 3));
    const int32_t slot = cache.claim_slot(0, 7);
    EC_CHECK(slot >= 0);
    cache.mark_occupied(slot);

    std::vector<int32_t> ids(4097, 7);
    ggml_tensor          dst               = {};
    dst.ne[0]                              = 1;
    dst.ne[1]                              = ids.size();
    dst.data                               = ids.data();
    llama_expert_cache::op_params_t params = { &cache, 0 };
    llama_expert_cache::reroute_fn(&dst, nullptr, nullptr, 0, 1, &params);
    EC_CHECK(params.slot_ids.size() >= ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        EC_CHECK(ids[i] == 7 && params.slot_ids[i] == slot);
    }
    printf("test_reroute_large_microbatch: PASS\n");
}

static void test_expert_cache_stats_helpers() {
    llama_expert_cache cache;
    cache.stats_ = true;

    cache.stats_record_hit(false);
    assert(cache.stat_hits.load() == 1);
    assert(cache.stat_misses.load() == 0);

    cache.stats_record_miss(false);
    assert(cache.stat_hits.load() == 1);
    assert(cache.stat_misses.load() == 1);

    cache.stats_record_ddr_copy(false);
    assert(cache.stat_hits.load() == 2);
    assert(cache.stat_ddr_copies.load() == 1);
    assert(cache.stat_misses.load() == 1);

    cache.stats_record_substitution(false);
    cache.stats_record_substitution(false);
    assert(cache.stat_subs.load() == 2);

    cache.stats_record_flash_read(1024, 2000, false);
    cache.stats_record_flash_read(2048, 3000, false);
    assert(cache.stat_flash_reads.load() == 2);
    assert(cache.stat_flash_bytes.load() == 3072);
    assert(cache.stat_flash_ns.load() == 5000);

    cache.stats_set_eval_time_us(0, 7000);
    assert(cache.stat_eval_us.load() == 7000);

    cache.stats_record_prefetch_adopt(false);
    EC_CHECK(cache.stat_hits.load() == 3 && cache.stat_prefetch_adopts.load() == 1);
    cache.stats_record_hit(true);
    cache.stats_record_miss(true);
    cache.stats_record_ddr_copy(true);
    cache.stats_record_prefetch_adopt(true);
    cache.stats_record_substitution(true);
    cache.stats_record_flash_read(4096, 6000, true);
    cache.stats_set_eval_time_us(8000, 7000);
    EC_CHECK(cache.stat_p_hits.load() == 3 && cache.stat_p_misses.load() == 1);
    EC_CHECK(cache.stat_p_ddr_copies.load() == 1 && cache.stat_p_prefetch_adopts.load() == 1);
    EC_CHECK(cache.stat_p_subs.load() == 1 && cache.stat_p_flash_bytes.load() == 4096);
    EC_CHECK(cache.stat_p_flash_ns.load() == 6000 && cache.stat_p_eval_us.load() == 8000);
    const auto output = cache.stats_to_str();
    EC_CHECK(output.find("prefill: lookups=4  hits=3 (75.0%)  misses=1 (25.0%)  subs=1") != std::string::npos);
    EC_CHECK(output.find("total:   lookups=8  hits=6 (75.0%)  misses=2 (25.0%)  subs=3") != std::string::npos);
    EC_CHECK(output.find("logical_expert_cache_read_bandwidth=") != std::string::npos);
    EC_CHECK(output.find(" GiB/s") != std::string::npos);

    cache.stat_htp_main_read_ns.store(1000);
    cache.stat_htp_main_copy_ns.store(2000);
    cache.stat_htp_main_repack_ns.store(3000);
    cache.stat_htp_worker_read_ns.store(4000);
    cache.stat_htp_worker_copy_ns.store(5000);
    cache.stat_htp_worker_repack_ns.store(6000);
    cache.use_htp_pool_ = true;
    assert(cache.stats_to_str().find("[ECSTATS] htp: main_read=") != std::string::npos);

    cache.stats_reset();
    assert(cache.stat_hits.load() == 0);
    assert(cache.stat_misses.load() == 0);
    assert(cache.stat_subs.load() == 0);
    assert(cache.stat_ddr_copies.load() == 0);
    assert(cache.stat_flash_reads.load() == 0);
    assert(cache.stat_flash_bytes.load() == 0);
    assert(cache.stat_flash_ns.load() == 0);
    assert(cache.stat_eval_us.load() == 0);
    assert(cache.stat_htp_main_read_ns.load() == 0);
    assert(cache.stat_htp_main_copy_ns.load() == 0);
    assert(cache.stat_htp_main_repack_ns.load() == 0);
    assert(cache.stat_htp_worker_read_ns.load() == 0);
    assert(cache.stat_htp_worker_copy_ns.load() == 0);
    assert(cache.stat_htp_worker_repack_ns.load() == 0);
    EC_CHECK(cache.stat_prefetch_adopts.load() == 0 && cache.stat_p_prefetch_adopts.load() == 0);
    EC_CHECK(cache.stat_p_hits.load() == 0 && cache.stat_p_misses.load() == 0 && cache.stat_p_subs.load() == 0);
    EC_CHECK(cache.stat_p_flash_bytes.load() == 0 && cache.stat_p_flash_ns.load() == 0);
    EC_CHECK(cache.stat_p_eval_us.load() == 0 && cache.stat_p_ddr_copies.load() == 0);

    printf("test_expert_cache_stats_helpers: PASS\n");
}

static void test_prefetch_statistics() {
    const test_env_override lookahead("LLAMA_EXPERT_CACHE_DECODE_PREFETCH_LOOKAHEAD");
    const test_env_override stats("LLAMA_EXPERT_CACHE_STATS");
    test_set_env(stats.name, "1");
    expert_cache_fixture fixture;
    ggml_backend_dev_t   devices[] = { nullptr };
    llama_backend_init();
    for (bool enabled : { false, true }) {
        test_set_env(lookahead.name, enabled ? "1" : "0");
        auto mp                 = llama_model_default_params();
        mp.devices              = devices;
        mp.n_gpu_layers         = 0;
        mp.use_extra_bufts      = false;
        mp.expert_cache_enabled = true;
        mp.load_mode            = LLAMA_LOAD_MODE_NONE;
        auto * model            = llama_model_load_from_file(fixture.path.c_str(), mp);
        EC_CHECK(model && model->expert_cache);
        auto & cache = *model->expert_cache;
        cache.stop_worker();
        cache.stats_reset();
        EC_CHECK(cache.stat_predict_calls.load() == 0 && cache.stat_predict_ns.load() == 0);
        EC_CHECK(cache.stats_to_str().find("predict_avg_us=0.0") != std::string::npos);
        if (enabled) {
            cache.prefetch_init();
            std::vector<float>   hidden(32, 0.1f);
            std::vector<int32_t> prediction;
            for (uint64_t call = 1; call <= 2; ++call) {
                cache.worker_predict(0, hidden.data(), hidden.size(), prediction);
                EC_CHECK(prediction.size() == 2);
                EC_CHECK(cache.stat_predict_calls.load() == call);
                EC_CHECK(cache.stat_predict_ns.load() > 0);
            }
            cache.warm_cache(0, { 0, 1 });
            EC_CHECK(cache.stat_prefetch_warmed.load() == 2);
            EC_CHECK(cache.flash_bytes_worker_.load() == 49152 && cache.stat_flash_bytes.load() == 49152);
            cache.warm_cache(0, { 0, 1 });
            EC_CHECK(cache.stat_warm_skip_cached.load() == 2);
            EC_CHECK(cache.stat_prefetch_wasted.load() == 0);
            int32_t     ids[] = { 0, 1 };
            ggml_tensor dst{};
            dst.type  = GGML_TYPE_I32;
            dst.ne[0] = 2;
            dst.ne[1] = dst.ne[2] = dst.ne[3] = 1;
            dst.data                          = ids;
            llama_expert_cache::op_params_t op;
            op.cache = &cache;
            for (int repeat = 0; repeat < 2; ++repeat) {
                llama_expert_cache::reroute_fn(&dst, nullptr, nullptr, 0, 1, &op);
                if (repeat == 0) {
                    cache.warm_cache(0, { 2, 3 });
                    EC_CHECK(cache.stat_warm_skip_noslot.load() == 2);
                }
                cache.release_slots(op);
            }
            EC_CHECK(cache.stat_prefetch_useful.load() == 2);
            cache.warm_cache(0, { 2, 3 });
            EC_CHECK(cache.stat_prefetch_wasted.load() == 0);
            cache.warm_cache(0, { 4, 5 });
            EC_CHECK(cache.stat_prefetch_warmed.load() == 6);
            EC_CHECK(cache.stat_prefetch_wasted.load() == 2);
            EC_CHECK(cache.stat_worker_evictions.load() == 4);
            for (int32_t e : { 4, 5 }) {
                cache.release(cache.index_.at(cache.make_key(0, e)));
            }
            EC_CHECK(cache.stat_prefetch_wasted.load() == 4);
            cache.warm_cache(0, { 6, 7 });
            cache.stats_reset();
            for (uint32_t s = 0; s < cache.n_slots_; ++s) {
                cache.release(s);
            }
            EC_CHECK(cache.stat_prefetch_wasted.load() == 0);
            EC_CHECK(cache.stat_prefetch_warmed.load() == 0);
            EC_CHECK(cache.stat_predict_calls.load() == 0 && cache.stat_predict_ns.load() == 0);
            for (int k = 0; k < llama_expert_cache::N_KINDS; ++k) {
                cache.offsets_[(size_t) k * cache.n_expert_ + 7].size = 0;
            }
            cache.warm_cache(0, { 6, 7 });
            EC_CHECK(cache.stat_prefetch_warmed.load() == 1);
            cache.release(cache.index_.at(cache.make_key(0, 7)));
            EC_CHECK(cache.stat_prefetch_wasted.load() == 0);
            cache.release(cache.index_.at(cache.make_key(0, 6)));
            EC_CHECK(cache.stat_prefetch_wasted.load() == 1);
            cache.stats_ = false;
            cache.start_worker();
            {
                std::unique_lock<std::mutex> lock(cache.worker_mu_);
                const auto                   sequence = cache.worker_wake_seq_;
                auto                         reset    = std::async(std::launch::async, [&] { cache.stats_reset(); });
                EC_CHECK(reset.wait_for(std::chrono::seconds(10)) == std::future_status::ready);
                reset.get();
                EC_CHECK(cache.worker_thread_.joinable() && !cache.worker_shutdown_.load());
                EC_CHECK(cache.worker_wake_seq_ == sequence);
            }
        }
        printf("lookahead statistics: enabled=%d PASS\n", enabled);
        llama_model_free(model);
    }
    llama_backend_free();
}

static void test_htp_statistics() {
    const test_env_override env[] = { { "LLAMA_EXPERT_CACHE_STATS" },
                                      { "LLAMA_EXPERT_CACHE_DECODE_PREFETCH_LOOKAHEAD" },
                                      { "LLAMA_EXPERT_CACHE_BATCH_DEPTH" } };
    test_set_env("LLAMA_EXPERT_CACHE_STATS", "1");
    test_set_env("LLAMA_EXPERT_CACHE_DECODE_PREFETCH_LOOKAHEAD", "1");
    ggml_backend_load_all();
    auto dev = ggml_backend_dev_by_name("HTP0");
    if (!dev) {
        puts("test_htp_statistics: unavailable without HTP backend");
        return;
    }
    expert_cache_fixture fixture(false, 32, GGML_TYPE_Q4_0);
    llama_backend_init();
    for (int depth : { 0, 1, 4 }) {
        test_set_env("LLAMA_EXPERT_CACHE_BATCH_DEPTH", std::to_string(std::max(1, depth)).c_str());
        for (bool direct : { false, true }) {
            ggml_backend_dev_t devices[] = { dev, nullptr };
            auto               mp        = llama_model_default_params();
            mp.devices                   = devices;
            mp.n_gpu_layers              = 99;
            mp.expert_cache_enabled      = true;
            mp.load_mode                 = direct ? LLAMA_LOAD_MODE_DIRECT_IO : LLAMA_LOAD_MODE_NONE;
            auto * model                 = llama_model_load_from_file(fixture.path.c_str(), mp);
            EC_CHECK(model && model->expert_cache && model->expert_cache->use_htp_pool_);
            auto & cache = *model->expert_cache;
            cache.stop_worker();
            cache.stats_reset();
            cache.thread_pool_disabled_        = depth == 0;
            cache.worker_thread_pool_disabled_ = depth == 0;
            int32_t     ids[]                  = { 0, 1 };
            ggml_tensor dst{};
            dst.ne[0] = 2;
            dst.ne[1] = dst.ne[2] = dst.ne[3] = 1;
            dst.data                          = ids;
            llama_expert_cache::op_params_t op;
            op.cache = &cache;
            llama_expert_cache::reroute_fn(&dst, nullptr, nullptr, 0, 1, &op);
            cache.release_slots(op);
            const size_t bytes = 2 * 3 * 64 * ggml_row_size(GGML_TYPE_Q4_0, 32);
            EC_CHECK(cache.stat_hits.load() == 0 && cache.stat_misses.load() == 2);
            EC_CHECK(cache.stat_flash_bytes.load() == bytes && cache.flash_bytes_worker_.load() == 0);
            EC_CHECK(cache.stat_htp_main_read_ns.load() > 0 && cache.stat_htp_main_repack_ns.load() > 0);
            if (direct && cache.odirect_active()) {
                EC_CHECK(cache.stat_htp_main_copy_ns.load() > 0);
            }
            cache.warm_cache(0, { 2, 3 });
            EC_CHECK(cache.stat_prefetch_warmed.load() == 2 && cache.stat_prefetch_wasted.load() == 0);
            EC_CHECK(cache.stat_flash_bytes.load() == 2 * bytes && cache.flash_bytes_worker_.load() == bytes);
            EC_CHECK(cache.stat_htp_worker_read_ns.load() > 0 && cache.stat_htp_worker_repack_ns.load() > 0);
            if (!direct) {
                EC_CHECK(cache.stat_htp_main_copy_ns.load() == 0 && cache.stat_htp_worker_copy_ns.load() == 0);
            } else if (cache.odirect_active()) {
                EC_CHECK(cache.stat_htp_main_copy_ns.load() > 0 && cache.stat_htp_worker_copy_ns.load() > 0);
            }
            ids[0] = 2;
            ids[1] = 3;
            llama_expert_cache::reroute_fn(&dst, nullptr, nullptr, 0, 1, &op);
            cache.release_slots(op);
            EC_CHECK(cache.stat_hits.load() == 2 && cache.stat_misses.load() == 2);
            EC_CHECK(cache.stat_prefetch_useful.load() == 2 && cache.stat_prefetch_wasted.load() == 0);
            printf("HTP statistics: depth=%d fallback=%d direct=%d logical_bytes=%zu used=2 wasted=0 PASS\n",
                   cache.batch_depth_, depth == 0, direct, 2 * bytes);
            llama_model_free(model);
        }
    }
    llama_backend_free();
}

static void test_flash_busy_union() {
    llama_expert_cache c;
    c.stats_ = true;

    // Overlapping reads share a single busy interval.
    auto           t0  = c.flash_begin();
    auto           t1  = c.flash_begin();  // second reader enters while first is open
    const uint64_t end = t1 + 1000;
    c.flash_end(t1, 1000, false, end);
    assert(c.decode_flash_busy_ns() == 0);
    c.flash_end(t0, 1000, false, end + 1000);

    assert(c.decode_flash_bytes() == 2000);
    const uint64_t union_ns = c.decode_flash_busy_ns();
    assert(union_ns == end + 1000 - t0);
    (void) union_ns;

    // A subsequent disjoint interval adds its full span to the union.
    const uint64_t before = c.decode_flash_busy_ns();
    auto           t2     = c.flash_begin();
    c.flash_end(t2, 500, false, t2 + 2000);
    assert(c.decode_flash_busy_ns() == before + 2000);
    (void) before;
    assert(c.decode_flash_bytes() == 2500);

    printf("test_flash_busy_union: PASS\n");
}

// Mark a layer covered only when every predicted expert is loaded.
static void test_warm_cache_marker_full_vs_partial() {
    const test_env_override forced_hit("LLAMA_EXPERT_CACHE_DECODE_PREFETCH_FORCE_CACHE_HIT");
    test_set_env(forced_hit.name, "1");
    std::string tmpfile = (std::filesystem::temp_directory_path() / "test-warm-cache-marker-XXXXXX").string();
    int         fd      = mkstemp(tmpfile.data());
    assert(fd >= 0);

    const size_t expert_sz = 64;
    uint8_t      pat[expert_sz];
    for (size_t i = 0; i < expert_sz; ++i) {
        pat[i] = (uint8_t) i;
    }
    for (int e = 0; e < 4; ++e) {
        int64_t w = write(fd, pat, expert_sz);
        (void) w;
        assert(w == (int64_t) expert_sz);
    }

    llama_expert_cache cache;
    EC_CHECK(cache.decode_prefetch_force_cache_hit_);
    EC_CHECK(cache.add_file(fd));

    std::vector<uint32_t> slots_per_layer = { 2, 2 };
    std::vector<size_t>   kind_sizes_flat(2 * llama_expert_cache::N_KINDS, 0);
    kind_sizes_flat[0 * llama_expert_cache::N_KINDS + llama_expert_cache::GATE_UP] = expert_sz;
    kind_sizes_flat[1 * llama_expert_cache::N_KINDS + llama_expert_cache::GATE_UP] = expert_sz;
    cache.init_layered(2, slots_per_layer, kind_sizes_flat);

    std::vector<llama_expert_cache::expert_slice_loc> locs((size_t) 2 * llama_expert_cache::N_KINDS * 4, { 0, 0 });
    for (int l = 0; l < 2; ++l) {
        for (int e = 0; e < 4; ++e) {
            locs[(size_t) l * llama_expert_cache::N_KINDS * 4 + (size_t) llama_expert_cache::GATE_UP * 4 + e] = {
                e * expert_sz, expert_sz
            };
        }
    }
    cache.build_offset_map(2, 4, 2, locs);

    // A delayed worker must stamp its own execution, even if another context has advanced.
    cache.decode_step_counter_.store(8, std::memory_order_relaxed);
    cache.warm_cache(0, { 0, 1 }, 7);
    assert(cache.layer_prefetch_done_token_[0].load(std::memory_order_relaxed) == 7);

    // Three predictions cannot all become resident in two slots.
    int32_t s2 = cache.claim_slot(1, 2);
    assert(s2 >= 0);
    cache.mark_occupied(s2);
    int32_t s3 = cache.claim_slot(1, 3);
    assert(s3 >= 0);
    cache.mark_occupied(s3);
    cache.warm_cache(1, { 0, 1, 2 }, 8);
    assert(cache.layer_prefetch_done_token_[1].load(std::memory_order_relaxed) == UINT64_MAX);

    // A delayed worker must preserve weights between routing and the expert matmuls.
    int32_t     ids[] = { 0, 1, 0, 1 };
    ggml_tensor selected{};
    selected.ne[0]                         = 2;
    selected.ne[1]                         = 1;
    selected.data                          = ids;
    cache.decode_prefetch_force_cache_hit_ = false;
    auto params                            = llama_expert_cache::op_params_t{ &cache, 0 };
    llama_expert_cache::reroute_fn(&selected, nullptr, nullptr, 0, 1, &params);
    const int32_t slot0 = params.slot_ids[0];
    const int32_t slot1 = params.slot_ids[1];
    cache.warm_cache(0, { 2, 3 });
    EC_CHECK(cache.slots[slot0].key.expert_idx == 0);
    EC_CHECK(cache.slots[slot1].key.expert_idx == 1);

    // Decode reservations allow prefetch in another layer's slots.
    cache.warm_cache(1, { 2, 3 });
    EC_CHECK(cache.slots[2].key.expert_idx == 2);
    EC_CHECK(cache.slots[3].key.expert_idx == 3);
    cache.release_slots(params);
    selected.ne[1] = 2;
    llama_expert_cache::reroute_fn(&selected, nullptr, nullptr, 0, 1, &params);
    cache.warm_cache(1, { 0, 1 });
    EC_CHECK(cache.slots[slot0].key.expert_idx == 0);
    EC_CHECK(cache.slots[slot1].key.expert_idx == 1);
    cache.release_slots(params);

    close(fd);
    unlink(tmpfile.c_str());
    printf("test_warm_cache_marker_full_vs_partial: PASS\n");
}

// A covered layer can substitute cached experts even with a zero threshold.
static void test_reroute_fn_forced_cache_hit() {
    const test_env_override forced_hit("LLAMA_EXPERT_CACHE_DECODE_PREFETCH_FORCE_CACHE_HIT");
    test_set_env(forced_hit.name, "1");
    llama_expert_cache cache;
    cache.stats_ = true;
    EC_CHECK(cache.decode_prefetch_force_cache_hit_);

    std::vector<uint32_t> slots_per_layer = { 3 };
    std::vector<size_t>   kind_sizes_flat(llama_expert_cache::N_KINDS, 0);
    kind_sizes_flat[llama_expert_cache::GATE_UP] = 64;
    cache.init_layered(1, slots_per_layer, kind_sizes_flat);

    std::vector<llama_expert_cache::expert_slice_loc> locs((size_t) 1 * llama_expert_cache::N_KINDS * 5, { 0, 0 });
    cache.build_offset_map(1, 5, 2, locs);  // n_expert=5, n_expert_used=2

    // Leave experts 0 and 4 uncached.
    for (int32_t e : { 1, 2, 3 }) {
        int32_t s = cache.claim_slot(0, (uint32_t) e);
        assert(s >= 0);
        cache.mark_occupied(s);
    }

    cache.layer_prefetch_done_token_[0].store(42, std::memory_order_relaxed);

    // Expert 1 is already selected, so only experts 2 and 3 can replace expert 0.
    int32_t     ids_data[2] = { 1, 0 };
    ggml_tensor dst         = {};
    dst.ne[0]               = 2;  // n_expert_used
    dst.ne[1]               = 1;  // n_tokens (decode)
    dst.data                = ids_data;

    // Expert 3 has the highest eligible probability despite a zero substitution threshold.
    float       probs_data[5] = { /*0*/ 0.50f, /*1*/ 0.90f, /*2*/ 0.10f, /*3*/ 0.80f, /*4*/ 0.05f };
    ggml_tensor probs_tensor  = {};
    probs_tensor.ne[0]        = 5;  // n_expert_total
    probs_tensor.ne[1]        = 1;
    probs_tensor.data         = probs_data;

    llama_expert_cache::op_params_t params = { &cache, 0, 42 };
    llama_expert_cache::reroute_fn(&dst, nullptr, &probs_tensor, /*ith=*/0, /*nth=*/1, &params);

    assert(ids_data[0] == 1);
    assert(ids_data[1] == 3);
    EC_CHECK(cache.stat_forced_subs.load() == 1 && cache.stat_prefetch_hits.load() == 1);

    printf("test_reroute_fn_forced_cache_hit: PASS\n");
}

// A covered layer still reads a miss when no unused cached expert can replace it.
static void test_reroute_fn_forced_cache_hit_no_candidate() {
    const test_env_override forced_hit("LLAMA_EXPERT_CACHE_DECODE_PREFETCH_FORCE_CACHE_HIT");
    test_set_env(forced_hit.name, "1");
    // The fallback needs real expert data to read.
    std::string tmpfile = (std::filesystem::temp_directory_path() / "test-forced-nocand-XXXXXX").string();
    int         fd      = mkstemp(tmpfile.data());
    assert(fd >= 0);
    const size_t expert_sz = 64;
    uint8_t      pat[expert_sz];
    for (size_t i = 0; i < expert_sz; ++i) {
        pat[i] = (uint8_t) i;
    }
    for (int e = 0; e < 5; ++e) {
        int64_t w = write(fd, pat, expert_sz);
        (void) w;
        assert(w == (int64_t) expert_sz);
    }

    llama_expert_cache cache;
    EC_CHECK(cache.decode_prefetch_force_cache_hit_);
    EC_CHECK(cache.add_file(fd));

    // Leave one empty slot for the miss that cannot be substituted.
    std::vector<uint32_t> slots_per_layer = { 2 };
    std::vector<size_t>   kind_sizes_flat(llama_expert_cache::N_KINDS, 0);
    kind_sizes_flat[llama_expert_cache::GATE_UP] = expert_sz;
    cache.init_layered(1, slots_per_layer, kind_sizes_flat);

    std::vector<llama_expert_cache::expert_slice_loc> locs((size_t) 1 * llama_expert_cache::N_KINDS * 5, { 0, 0 });
    for (int e = 0; e < 5; ++e) {
        locs[(size_t) llama_expert_cache::GATE_UP * 5 + e] = { e * expert_sz, expert_sz };
    }
    cache.build_offset_map(1, 5, 2, locs);  // n_expert=5, n_expert_used=2

    {
        int32_t s = cache.claim_slot(0, 1);
        assert(s >= 0);
        cache.mark_occupied(s);
    }

    cache.layer_prefetch_done_token_[0].store(99, std::memory_order_relaxed);

    int32_t     ids_data[2] = { 1, 0 };  // expert 1 cached+selected, expert 0 uncached
    ggml_tensor dst         = {};
    dst.ne[0]               = 2;
    dst.ne[1]               = 1;
    dst.data                = ids_data;

    float       probs_data[5] = { 0.5f, 0.9f, 0.1f, 0.05f, 0.02f };
    ggml_tensor probs_tensor  = {};
    probs_tensor.ne[0]        = 5;
    probs_tensor.ne[1]        = 1;
    probs_tensor.data         = probs_data;

    llama_expert_cache::op_params_t params = { &cache, 0, 99 };
    llama_expert_cache::reroute_fn(&dst, nullptr, &probs_tensor, /*ith=*/0, /*nth=*/1, &params);

    assert(ids_data[0] == 1);
    assert(ids_data[1] == 0);
    assert(cache.stat_forced_subs.load() == 0);

    close(fd);
    unlink(tmpfile.c_str());
    printf("test_reroute_fn_forced_cache_hit_no_candidate: PASS\n");
}

static void test_read_percentage_disabled_reads_all_misses() {
    std::string tmpfile = (std::filesystem::temp_directory_path() / "test-read-percentage-disabled-XXXXXX").string();
    int         fd      = mkstemp(tmpfile.data());
    assert(fd >= 0);
    const size_t expert_sz = 64;
    uint8_t      pat[expert_sz];
    for (size_t i = 0; i < expert_sz; ++i) {
        pat[i] = (uint8_t) i;
    }
    for (int e = 0; e < 5; ++e) {
        int64_t w = write(fd, pat, expert_sz);
        (void) w;
        assert(w == (int64_t) expert_sz);
    }

    llama_expert_cache cache;
    cache.read_percentage_ = 0;  // ignore environment settings
    EC_CHECK(cache.add_file(fd));

    // An empty cache makes every router pick a miss.
    std::vector<uint32_t> slots_per_layer = { 5 };
    std::vector<size_t>   kind_sizes_flat(llama_expert_cache::N_KINDS, 0);
    kind_sizes_flat[llama_expert_cache::GATE_UP] = expert_sz;
    cache.init_layered(1, slots_per_layer, kind_sizes_flat);

    std::vector<llama_expert_cache::expert_slice_loc> locs((size_t) 1 * llama_expert_cache::N_KINDS * 5, { 0, 0 });
    for (int e = 0; e < 5; ++e) {
        locs[(size_t) llama_expert_cache::GATE_UP * 5 + e] = { e * expert_sz, expert_sz };
    }
    cache.build_offset_map(1, 5, 2, locs);  // n_expert=5, n_expert_used=2

    int32_t     ids_data[2] = { 0, 1 };     // both experts uncached
    ggml_tensor dst         = {};
    dst.ne[0]               = 2;            // n_expert_used
    dst.ne[1]               = 1;            // n_tokens (decode)
    dst.data                = ids_data;

    llama_expert_cache::op_params_t params = { &cache, 0 };
    llama_expert_cache::reroute_fn(&dst, nullptr, nullptr, /*ith=*/0, /*nth=*/1, &params);

    assert(ids_data[0] == 0);
    assert(ids_data[1] == 1);

    close(fd);
    unlink(tmpfile.c_str());
    printf("test_read_percentage_disabled_reads_all_misses: PASS\n");
}

// With a 50% read limit, four eligible misses need two reads and two substitutions.
static void test_read_percentage_ranks_and_substitutes() {
    std::string tmpfile = (std::filesystem::temp_directory_path() / "test-read-percentage-enabled-XXXXXX").string();
    int         fd      = mkstemp(tmpfile.data());
    assert(fd >= 0);
    const size_t expert_sz = 64;
    uint8_t      pat[expert_sz];
    for (size_t i = 0; i < expert_sz; ++i) {
        pat[i] = (uint8_t) i;
    }
    for (int e = 0; e < 8; ++e) {
        int64_t w = write(fd, pat, expert_sz);
        (void) w;
        assert(w == (int64_t) expert_sz);
    }

    llama_expert_cache cache;
    cache.read_percentage_ = 50;  // ceil(4 * 50 / 100) = 2 reads
    EC_CHECK(cache.add_file(fd));

    // Leave space for three cached experts and four possible misses.
    std::vector<uint32_t> slots_per_layer = { 7 };
    std::vector<size_t>   kind_sizes_flat(llama_expert_cache::N_KINDS, 0);
    kind_sizes_flat[llama_expert_cache::GATE_UP] = expert_sz;
    cache.init_layered(1, slots_per_layer, kind_sizes_flat);

    std::vector<llama_expert_cache::expert_slice_loc> locs((size_t) 1 * llama_expert_cache::N_KINDS * 8, { 0, 0 });
    for (int e = 0; e < 8; ++e) {
        locs[(size_t) llama_expert_cache::GATE_UP * 8 + e] = { e * expert_sz, expert_sz };
    }
    cache.build_offset_map(1, 8, 4, locs);  // n_expert=8, n_expert_used=4

    for (int32_t e : { 5, 6, 7 }) {
        int32_t s = cache.claim_slot(0, (uint32_t) e);
        assert(s >= 0);
        cache.mark_occupied(s);
    }

    // All four router picks are uncached.
    int32_t     ids_data[4] = { 0, 1, 2, 3 };
    ggml_tensor dst         = {};
    dst.ne[0]               = 4;  // n_expert_used
    dst.ne[1]               = 1;  // n_tokens (decode)
    dst.data                = ids_data;

    // Experts 0 and 1 have the two closest cached matches.
    float       probs_data[8] = { /*0*/ 0.40f, /*1*/ 0.10f, /*2*/ 0.60f, /*3*/ 0.15f,
                            /*4*/ 0.0f,  /*5*/ 0.90f, /*6*/ 0.42f, /*7*/ 0.05f };
    ggml_tensor probs_tensor  = {};
    probs_tensor.ne[0]        = 8;  // n_expert_total
    probs_tensor.ne[1]        = 1;
    probs_tensor.data         = probs_data;

    llama_expert_cache::op_params_t params = { &cache, 0 };
    llama_expert_cache::reroute_fn(&dst, nullptr, &probs_tensor, /*ith=*/0, /*nth=*/1, &params);

    // The two larger gaps remain real reads.
    assert(ids_data[2] == 2);
    assert(ids_data[3] == 3);

    assert(ids_data[0] == 6);  // miss 0 (p=0.40) substituted by expert 6 (p=0.42)
    assert(ids_data[1] == 7);  // miss 1 (p=0.10) substituted by expert 7 (p=0.05)

    close(fd);
    unlink(tmpfile.c_str());
    printf("test_read_percentage_ranks_and_substitutes: PASS\n");
}

// An empty cache must read every miss regardless of the read limit.
static void test_read_percentage_empty_cache_reads_all() {
    std::string tmpfile = (std::filesystem::temp_directory_path() / "test-read-percentage-empty-XXXXXX").string();
    int         fd      = mkstemp(tmpfile.data());
    assert(fd >= 0);
    const size_t expert_sz = 64;
    uint8_t      pat[expert_sz];
    for (size_t i = 0; i < expert_sz; ++i) {
        pat[i] = (uint8_t) i;
    }
    for (int e = 0; e < 4; ++e) {
        int64_t w = write(fd, pat, expert_sz);
        (void) w;
        assert(w == (int64_t) expert_sz);
    }

    llama_expert_cache cache;
    cache.read_percentage_ = 50;
    EC_CHECK(cache.add_file(fd));

    std::vector<uint32_t> slots_per_layer = { 4 };
    std::vector<size_t>   kind_sizes_flat(llama_expert_cache::N_KINDS, 0);
    kind_sizes_flat[llama_expert_cache::GATE_UP] = expert_sz;
    cache.init_layered(1, slots_per_layer, kind_sizes_flat);

    std::vector<llama_expert_cache::expert_slice_loc> locs((size_t) 1 * llama_expert_cache::N_KINDS * 4, { 0, 0 });
    for (int e = 0; e < 4; ++e) {
        locs[(size_t) llama_expert_cache::GATE_UP * 4 + e] = { e * expert_sz, expert_sz };
    }
    cache.build_offset_map(1, 4, 2, locs);  // n_expert=4, n_expert_used=2, cache is empty

    int32_t     ids_data[2] = { 0, 1 };
    ggml_tensor dst         = {};
    dst.ne[0]               = 2;
    dst.ne[1]               = 1;
    dst.data                = ids_data;

    float       probs_data[4] = { 0.5f, 0.3f, 0.1f, 0.1f };
    ggml_tensor probs_tensor  = {};
    probs_tensor.ne[0]        = 4;
    probs_tensor.ne[1]        = 1;
    probs_tensor.data         = probs_data;

    llama_expert_cache::op_params_t params = { &cache, 0 };
    llama_expert_cache::reroute_fn(&dst, nullptr, &probs_tensor, /*ith=*/0, /*nth=*/1, &params);

    assert(ids_data[0] == 0);
    assert(ids_data[1] == 1);

    close(fd);
    unlink(tmpfile.c_str());
    printf("test_read_percentage_empty_cache_reads_all: PASS\n");
}

// A substituted expert must stay pinned while another miss evicts from a full cache.
static void test_read_percentage_substitute_survives_eviction() {
    std::string tmpfile = (std::filesystem::temp_directory_path() / "test-read-percentage-pin-XXXXXX").string();
    int         fd      = mkstemp(tmpfile.data());
    assert(fd >= 0);
    const size_t expert_sz = 64;
    uint8_t      pat[expert_sz];
    for (size_t i = 0; i < expert_sz; ++i) {
        pat[i] = (uint8_t) i;
    }
    for (int e = 0; e < 6; ++e) {
        int64_t w = write(fd, pat, expert_sz);
        (void) w;
        assert(w == (int64_t) expert_sz);
    }

    llama_expert_cache cache;
    cache.read_percentage_ = 50;  // exactly 1 of the 2 misses below gets read
    EC_CHECK(cache.add_file(fd));
    cache.policy_ = LLAMA_EXPERT_CACHE_POLICY_LRU;

    // Fill every slot so the remaining miss must evict an expert.
    std::vector<uint32_t> slots_per_layer = { 4 };
    std::vector<size_t>   kind_sizes_flat(llama_expert_cache::N_KINDS, 0);
    kind_sizes_flat[llama_expert_cache::GATE_UP] = expert_sz;
    cache.init_layered(1, slots_per_layer, kind_sizes_flat);

    std::vector<llama_expert_cache::expert_slice_loc> locs((size_t) 1 * llama_expert_cache::N_KINDS * 6, { 0, 0 });
    for (int e = 0; e < 6; ++e) {
        locs[(size_t) llama_expert_cache::GATE_UP * 6 + e] = { e * expert_sz, expert_sz };
    }
    cache.build_offset_map(1, 6, 2, locs);  // n_expert=6, n_expert_used=2

    // Make expert 4 the oldest slot so pinning is the only thing protecting it.
    int32_t s4 = cache.claim_slot(0, 4);
    assert(s4 >= 0);
    cache.mark_occupied(s4);
    cache.record_use(s4);

    for (int32_t e : { 2, 3, 5 }) {
        int32_t s = cache.claim_slot(0, (uint32_t) e);
        assert(s >= 0);
        cache.mark_occupied(s);
        cache.record_use(s);
    }

    // Substitute expert 4 for one miss and load the other into an evicted slot.
    int32_t     ids_data[2] = { 0, 1 };
    ggml_tensor dst         = {};
    dst.ne[0]               = 2;
    dst.ne[1]               = 1;
    dst.data                = ids_data;

    float       probs_data[6] = { 0.85f, 0.20f, 0.0f, 0.05f, 0.90f, 0.10f };
    ggml_tensor probs_tensor  = {};
    probs_tensor.ne[0]        = 6;
    probs_tensor.ne[1]        = 1;
    probs_tensor.data         = probs_data;

    llama_expert_cache::op_params_t params = { &cache, 0 };
    llama_expert_cache::reroute_fn(&dst, nullptr, &probs_tensor, /*ith=*/0, /*nth=*/1, &params);

    // Expert 4 is closer to miss 0 than expert 5 is to miss 1.
    assert(ids_data[0] == 4);
    assert(ids_data[1] == 1);

    // Pinning must keep the oldest slot from being evicted.
    assert((llama_expert_slot_state) cache.slots[s4].state.load() == llama_expert_slot_state::OCCUPIED);
    assert(cache.slots[s4].key.expert_idx == 4);

    close(fd);
    unlink(tmpfile.c_str());
    printf("test_read_percentage_substitute_survives_eviction: PASS\n");
}

// Aligned reads must return exactly the requested expert bytes.
static void test_odirect_matches_buffered_unaligned() {
    // Select a filesystem that supports O_DIRECT.
    const char * dir = getenv("LLAMA_TEST_ODIRECT_DIR");
    char         tmpfile[4096];
    snprintf(tmpfile, sizeof(tmpfile), "%s/test-odirect-unaligned-XXXXXX",
             dir ? dir : std::filesystem::temp_directory_path().string().c_str());
    int fd = mkstemp(tmpfile);
    assert(fd >= 0);

    const size_t         file_sz = 512 * 1024;
    std::vector<uint8_t> src(file_sz);
    for (size_t i = 0; i < file_sz; ++i) {
        src[i] = (uint8_t) ((i * 31 + (i >> 8) * 7) & 0xff);
    }
    EC_CHECK(write(fd, src.data(), file_sz) == (int64_t) file_sz);

    llama_expert_cache cache;
    EC_CHECK(cache.add_file(fd, true));

    if (!cache.odirect_active()) {
        printf("test_odirect_matches_buffered_unaligned: SKIP (O_DIRECT unavailable)\n");
        close(fd);
        unlink(tmpfile);
        return;
    }

    // Include 32-byte-aligned reads that cross filesystem block boundaries.
    const struct {
        int64_t offset;
        size_t  size;
    } cases[] = {
        { 0,      32     },
        { 32,     32     },
        { 4064,   64     },
        { 4096,   4096   },
        { 8160,   8256   },
        { 130976, 128    },
        { 262144, 131072 },
    };

    for (const auto & c : cases) {
        EC_CHECK((size_t) c.offset + c.size <= file_sz);

        std::vector<uint8_t> got(c.size, 0);
        EC_CHECK(cache.odirect_pread_for_test(/*pool_slot=*/0, /*buf_idx=*/0, got.data(), c.size, c.offset) ==
                 (int64_t) c.size);
        EC_CHECK(memcmp(got.data(), src.data() + c.offset, c.size) == 0);
    }

    // the two pools must not share a bounce buffer
    std::vector<uint8_t> a(4096, 0), b(4096, 0);
    EC_CHECK(cache.odirect_pread_for_test(0, 0, a.data(), 4096, 1024) == 4096);
    EC_CHECK(cache.odirect_pread_for_test(1, 0, b.data(), 4096, 200704) == 4096);
    EC_CHECK(memcmp(a.data(), src.data() + 1024, 4096) == 0);
    EC_CHECK(memcmp(b.data(), src.data() + 200704, 4096) == 0);

    close(fd);
    unlink(tmpfile);
    printf("test_odirect_matches_buffered_unaligned: PASS\n");
}

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc == 2 && strcmp(argv[1], "--htp-stats-only") == 0) {
        test_htp_statistics();
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--prefetch-stats-only") == 0) {
        test_prefetch_statistics();
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--router-policy-only") == 0) {
        test_router_precision_policy();
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--htp-callback-only") == 0) {
        test_htp_callback_graph();
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--accounting-only") == 0) {
        test_cache_runtime_accounting();
        test_cache_loading_peak();
        return 0;
    }
    expert_cache_fixture    fixture;
    const test_env_override model_env("LLAMA_TEST_MODEL");
    test_set_env("LLAMA_TEST_MODEL", model_env.was_set ? model_env.saved.c_str() : fixture.path.c_str());
    test_htp_callback_worker();
    test_cache_config();
    if (argc == 2 && strcmp(argv[1], "--fit-only") == 0) {
        test_cache_no_alloc();
        test_cache_fitting();
        test_cache_runtime_accounting();
        test_cache_loading_peak();
        return 0;
    }
    test_empty_cache();
    test_cache_no_alloc();
    test_cache_fitting();
    test_cache_runtime_accounting();
    test_cache_loading_peak();
    test_model_minimum_cache();
    test_cache_empty_output();
    test_cache_batch_limit();
    test_reroute_preserves_in_flight_slots();
    test_concurrent_cache_graphs();
    test_cache_capacity_wait();
    test_cache_live_cross_range();
    test_slot_state_machine();
    test_lfu_eviction();
    test_release_and_reuse();
    test_claim_full_cache();
    test_load_expert_sync();
    test_split_file_reads();
    test_kind_first_layout();
    test_pool_tensors_shape();
    test_id_to_slot_fn();
    test_expert_cache_model_load_and_save();
    test_htp_group_plan_qwen_2gb_balanced();
    test_htp_group_plan_trims_above_floor();
    test_htp_group_plan_rejects_impossible_floor();
    test_htp_minimum_groups();
    test_sparse_mul_mat_id();
    test_sparse_mul_mat_id_with_scale();
    test_router_precision_policy();
    test_capacity_floor_check();
    test_reroute_large_microbatch();
    test_read_percentage_disabled_reads_all_misses();
    test_read_percentage_ranks_and_substitutes();
    test_read_percentage_empty_cache_reads_all();
    test_read_percentage_substitute_survives_eviction();
    test_expert_cache_stats_helpers();
    test_prefetch_statistics();
    test_htp_statistics();
    test_flash_busy_union();
    test_warm_cache_marker_full_vs_partial();
    test_reroute_fn_forced_cache_hit();
    test_reroute_fn_forced_cache_hit_no_candidate();
    test_odirect_matches_buffered_unaligned();
    printf("All tests PASSED\n");
    return 0;
}

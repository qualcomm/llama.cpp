// Define _GNU_SOURCE before headers to expose O_DIRECT.
#if defined(__linux__) && !defined(_GNU_SOURCE)
#    define _GNU_SOURCE
#endif

#include "llama-expert-cache.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "llama-context.h"
#include "llama-expert-cache-io.h"
#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-model.h"

// Declared here to avoid including ggml backend internals.
extern "C" ggml_backend_buffer_t ggml_backend_multi_buffer_alloc_buffer(ggml_backend_buffer_t * buffers,
                                                                        size_t                  n_buffers);
#include <fcntl.h>
#include <sys/stat.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <exception>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <thread>

#if defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <io.h>
#    include <windows.h>
#else
#    include <unistd.h>
#endif

static ec_read_batch_result read_batch_stream(ec_thread_pool *                      pool,
                                              size_t                                pool_slot,
                                              const ec_read_op *                    ops,
                                              uint32_t                              n,
                                              ec_odirect *                          od,
                                              bool                                  collect_timing,
                                              const std::function<void(uint32_t)> & on_complete,
                                              const std::function<void(uint64_t)> & on_reads_done) {
    if (pool) {
        return pool->read_batch_stream(ops, n, od, collect_timing, on_complete, on_reads_done);
    }

    const uint64_t        start_ns = collect_timing ? stats_now_ns() : 0;
    bool                  all_ok   = true;
    uint64_t              copy_ns  = 0;
    std::vector<uint32_t> completed;
    completed.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        uint64_t op_copy_ns = 0;
        if (!ops[i].file->read(od, pool_slot, i, ops[i].dst, ops[i].size, ops[i].offset,
                               collect_timing ? &op_copy_ns : nullptr)) {
            all_ok = false;
            continue;
        }
        copy_ns += op_copy_ns;
        completed.push_back(i);
    }
    const uint64_t end_ns = collect_timing ? stats_now_ns() : 0;
    on_reads_done(end_ns);
    for (uint32_t i : completed) {
        on_complete(i);
    }
    return { all_ok, collect_timing ? end_ns - start_ns : 0, copy_ns };
}

#include <iomanip>
#include <sstream>

static size_t htp_chunk_rows_for(const ggml_tensor * tensor, int chunk_kb) {
    const size_t row_size     = ggml_row_size(tensor->type, tensor->ne[0]);
    const size_t target_bytes = (size_t) chunk_kb * 1024;
    const size_t rows         = target_bytes / row_size;
    return std::max((size_t) 32, (rows / 32) * 32);
}

std::string llama_expert_cache::stats_to_str() const {
    const uint64_t p_hits   = stat_p_hits.load(std::memory_order_relaxed);
    const uint64_t p_misses = stat_p_misses.load(std::memory_order_relaxed);
    const uint64_t d_hits   = stat_hits.load(std::memory_order_relaxed);
    const uint64_t d_misses = stat_misses.load(std::memory_order_relaxed);

    const uint64_t p_total = p_hits + p_misses;
    const uint64_t d_total = d_hits + d_misses;
    const uint64_t t_total = p_total + d_total;

    const uint64_t p_subs  = stat_p_subs.load(std::memory_order_relaxed);
    const uint64_t d_subs  = stat_subs.load(std::memory_order_relaxed);
    const uint64_t p_ddr   = stat_p_ddr_copies.load(std::memory_order_relaxed);
    const uint64_t d_ddr   = stat_ddr_copies.load(std::memory_order_relaxed);
    const uint64_t p_adopt = stat_p_prefetch_adopts.load(std::memory_order_relaxed);
    const uint64_t d_adopt = stat_prefetch_adopts.load(std::memory_order_relaxed);

    const double p_flash_gb = stat_p_flash_bytes.load(std::memory_order_relaxed) / 1073741824.0;
    const double d_flash_gb = stat_flash_bytes.load(std::memory_order_relaxed) / 1073741824.0;
    const double p_flash_ms = stat_p_flash_ns.load(std::memory_order_relaxed) / 1e6;
    const double d_flash_ms = flash_busy_ns_.load(std::memory_order_relaxed) / 1e6;
    const double p_eval_ms  = stat_p_eval_us.load(std::memory_order_relaxed) / 1000.0;
    const double d_eval_ms  = stat_eval_us.load(std::memory_order_relaxed) / 1000.0;

    auto flash_speed = [](double gb, double ms) -> double {
        return (ms > 0) ? (gb / (ms / 1000.0)) : 0.0;
    };
    auto flash_pct_fn = [](double fms, double ems) -> double {
        return (ems > 0) ? (fms / ems * 100.0) : 0.0;
    };
    auto pct = [](uint64_t n, uint64_t total) -> double {
        return total ? (100.0 * n / (double) total) : 0.0;
    };

    std::stringstream ss;
    ss << std::fixed;

    ss << "[ECSTATS] mode=" << (layered_mode_ ? "layered" : (shared_mode_ ? "global-nonuniform" : "global"))
       << "  batch_depth=" << batch_depth_ << "  htp_read_chunk_kb=" << htp_read_chunk_kb_
       << "  reads=" << (odirect_ ? "O_DIRECT" : "buffered") << "\n";

    ss << std::setprecision(1);
    ss << "[ECSTATS] prefill: lookups=" << p_total << "  hits=" << p_hits << " (" << pct(p_hits, p_total) << "%)"
       << "  misses=" << p_misses << " (" << pct(p_misses, p_total) << "%)" << "  subs=" << p_subs
       << "  ddr_copies=" << p_ddr << "  prefetch_adopts=" << p_adopt << "\n";
    ss << "[ECSTATS] prefill: " << std::setprecision(3) << "flash_read=" << p_flash_gb << " GiB  "
       << std::setprecision(0) << "flash_time=" << p_flash_ms << " ms  " << std::setprecision(2)
       << "logical_expert_cache_read_bandwidth=" << flash_speed(p_flash_gb, p_flash_ms) << " GiB/s  "
       << std::setprecision(1) << "flash_pct=" << flash_pct_fn(p_flash_ms, p_eval_ms) << "%\n";

    ss << std::setprecision(1);
    ss << "[ECSTATS] decode:  lookups=" << d_total << "  hits=" << d_hits << " (" << pct(d_hits, d_total) << "%)"
       << "  misses=" << d_misses << " (" << pct(d_misses, d_total) << "%)" << "  subs=" << d_subs
       << "  ddr_copies=" << d_ddr << "  prefetch_adopts=" << d_adopt << "\n";
    ss << "[ECSTATS] decode:  " << std::setprecision(3) << "flash_read=" << d_flash_gb << " GiB  "
       << std::setprecision(0) << "flash_time=" << d_flash_ms << " ms  " << std::setprecision(2)
       << "logical_expert_cache_read_bandwidth=" << flash_speed(d_flash_gb, d_flash_ms) << " GiB/s  "
       << std::setprecision(1) << "flash_pct=" << flash_pct_fn(d_flash_ms, d_eval_ms) << "%\n";
    ss << "[ECSTATS] decode:  " << std::setprecision(3)
       << "flash_worker=" << (flash_bytes_worker_.load(std::memory_order_relaxed) / 1073741824.0) << " GiB  "
       << std::setprecision(1) << "flash_worker_pct="
       << (d_flash_gb > 0 ? 100.0 * (flash_bytes_worker_.load(std::memory_order_relaxed) / 1073741824.0) / d_flash_gb :
                            0.0)
       << "%\n";

    const double t_flash_gb = p_flash_gb + d_flash_gb;
    const double t_flash_ms = p_flash_ms + d_flash_ms;
    const double t_eval_ms  = p_eval_ms + d_eval_ms;
    ss << std::setprecision(1);
    ss << "[ECSTATS] total:   lookups=" << t_total << "  hits=" << (p_hits + d_hits) << " ("
       << pct(p_hits + d_hits, t_total) << "%)" << "  misses=" << (p_misses + d_misses) << " ("
       << pct(p_misses + d_misses, t_total) << "%)" << "  subs=" << (p_subs + d_subs)
       << "  ddr_copies=" << (p_ddr + d_ddr) << "  prefetch_adopts=" << (p_adopt + d_adopt) << "\n";
    ss << "[ECSTATS] total:   " << std::setprecision(3) << "flash_read=" << t_flash_gb << " GiB  "
       << std::setprecision(0) << "flash_time=" << t_flash_ms << " ms  " << std::setprecision(2)
       << "logical_expert_cache_read_bandwidth=" << flash_speed(t_flash_gb, t_flash_ms) << " GiB/s  "
       << std::setprecision(1) << "flash_pct=" << flash_pct_fn(t_flash_ms, t_eval_ms) << "%\n";
    const uint64_t pf_warmed  = stat_prefetch_warmed.load(std::memory_order_relaxed);
    const uint64_t pf_useful  = stat_prefetch_useful.load(std::memory_order_relaxed);
    const uint64_t pf_wasted  = stat_prefetch_wasted.load(std::memory_order_relaxed);
    const uint64_t pred_calls = stat_predict_calls.load(std::memory_order_relaxed);
    ss << std::setprecision(1);
    ss << "[ECSTATS] prefetch: warmed=" << pf_warmed << "  useful=" << pf_useful << " (" << pct(pf_useful, pf_warmed)
       << "%)" << "  wasted=" << pf_wasted << "  skip_cached=" << stat_warm_skip_cached.load(std::memory_order_relaxed)
       << "  skip_noslot=" << stat_warm_skip_noslot.load(std::memory_order_relaxed)
       << "  evictions=" << stat_worker_evictions.load(std::memory_order_relaxed) << "\n";
    ss << "[ECSTATS] prefetch: predict_calls=" << pred_calls << "  predict_avg_us="
       << (pred_calls ? (stat_predict_ns.load(std::memory_order_relaxed) / 1000.0 / pred_calls) : 0.0) << "\n";

    const uint64_t f_subs  = stat_forced_subs.load(std::memory_order_relaxed);
    const uint64_t f_hits  = stat_prefetch_hits.load(std::memory_order_relaxed);
    const uint64_t f_total = f_subs + f_hits;
    ss << std::setprecision(1);
    ss << "[ECSTATS] forced_prefetch: lookups=" << f_total << "  forced_subs=" << f_subs << " (" << pct(f_subs, f_total)
       << "%)" << "  prefetch_hits=" << f_hits << " (" << pct(f_hits, f_total) << "%)\n";

    ss << "[ECSTATS] read_percentage: subs=" << stat_read_pct_subs.load(std::memory_order_relaxed) << "\n";

    if (use_htp_pool_) {
        ss << std::setprecision(3);
        ss << "[ECSTATS] htp: main_read=" << (stat_htp_main_read_ns.load(std::memory_order_relaxed) / 1e6) << " ms"
           << "  main_copy=" << (stat_htp_main_copy_ns.load(std::memory_order_relaxed) / 1e6) << " ms"
           << "  main_repack=" << (stat_htp_main_repack_ns.load(std::memory_order_relaxed) / 1e6) << " ms\n";
        ss << "[ECSTATS] htp: worker_read=" << (stat_htp_worker_read_ns.load(std::memory_order_relaxed) / 1e6) << " ms"
           << "  worker_copy=" << (stat_htp_worker_copy_ns.load(std::memory_order_relaxed) / 1e6) << " ms"
           << "  worker_repack=" << (stat_htp_worker_repack_ns.load(std::memory_order_relaxed) / 1e6) << " ms\n";
    }

    ss << std::setprecision(0);
    ss << "[ECSTATS] total:   total_inference=" << t_eval_ms << " ms\n";

    return ss.str();
}

void llama_expert_cache::stats_reset() {
    if (!stats_) {
        return;
    }
    const bool restart_worker = worker_thread_.joinable();
    stop_worker();
    std::unique_lock<std::mutex> reservation_lock(reserve_mu_);
    for (uint32_t s = 0; slots && s < n_slots_; ++s) {
        slots[s].warmed_by_worker.store(0, std::memory_order_relaxed);
    }
    stat_hits.store(0, std::memory_order_relaxed);
    stat_misses.store(0, std::memory_order_relaxed);
    stat_subs.store(0, std::memory_order_relaxed);
    stat_ddr_copies.store(0, std::memory_order_relaxed);
    stat_prefetch_adopts.store(0, std::memory_order_relaxed);
    stat_forced_subs.store(0, std::memory_order_relaxed);
    stat_prefetch_hits.store(0, std::memory_order_relaxed);
    stat_read_pct_subs.store(0, std::memory_order_relaxed);
    stat_flash_reads.store(0, std::memory_order_relaxed);
    stat_flash_bytes.store(0, std::memory_order_relaxed);
    stat_flash_ns.store(0, std::memory_order_relaxed);
    stat_eval_us.store(0, std::memory_order_relaxed);
    stat_p_hits.store(0, std::memory_order_relaxed);
    stat_p_misses.store(0, std::memory_order_relaxed);
    stat_p_subs.store(0, std::memory_order_relaxed);
    stat_p_ddr_copies.store(0, std::memory_order_relaxed);
    stat_p_prefetch_adopts.store(0, std::memory_order_relaxed);
    stat_p_flash_reads.store(0, std::memory_order_relaxed);
    stat_p_flash_bytes.store(0, std::memory_order_relaxed);
    stat_p_flash_ns.store(0, std::memory_order_relaxed);
    stat_p_eval_us.store(0, std::memory_order_relaxed);
    stat_prefetch_warmed.store(0, std::memory_order_relaxed);
    stat_prefetch_useful.store(0, std::memory_order_relaxed);
    stat_prefetch_wasted.store(0, std::memory_order_relaxed);
    stat_warm_skip_cached.store(0, std::memory_order_relaxed);
    stat_warm_skip_noslot.store(0, std::memory_order_relaxed);
    stat_worker_evictions.store(0, std::memory_order_relaxed);
    stat_predict_calls.store(0, std::memory_order_relaxed);
    stat_predict_ns.store(0, std::memory_order_relaxed);
    stat_htp_main_read_ns.store(0, std::memory_order_relaxed);
    stat_htp_main_copy_ns.store(0, std::memory_order_relaxed);
    stat_htp_main_repack_ns.store(0, std::memory_order_relaxed);
    stat_htp_worker_read_ns.store(0, std::memory_order_relaxed);
    stat_htp_worker_copy_ns.store(0, std::memory_order_relaxed);
    stat_htp_worker_repack_ns.store(0, std::memory_order_relaxed);
    flash_busy_ns_.store(0, std::memory_order_relaxed);
    flash_bytes_worker_.store(0, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lk(flash_busy_mu_);
        flash_in_flight_  = 0;
        flash_busy_start_ = 0;
    }
    reservation_lock.unlock();
    if (restart_worker) {
        start_worker();
    }
}

void llama_expert_cache::stats_record_hit(bool is_prefill) {
    if (!stats_) {
        return;
    }
    (is_prefill ? stat_p_hits : stat_hits).fetch_add(1, std::memory_order_relaxed);
}

void llama_expert_cache::stats_record_miss(bool is_prefill) {
    if (!stats_) {
        return;
    }
    (is_prefill ? stat_p_misses : stat_misses).fetch_add(1, std::memory_order_relaxed);
}

void llama_expert_cache::stats_record_ddr_copy(bool is_prefill) {
    if (!stats_) {
        return;
    }
    (is_prefill ? stat_p_hits : stat_hits).fetch_add(1, std::memory_order_relaxed);
    (is_prefill ? stat_p_ddr_copies : stat_ddr_copies).fetch_add(1, std::memory_order_relaxed);
}

void llama_expert_cache::stats_record_prefetch_adopt(bool is_prefill) {
    if (!stats_) {
        return;
    }
    (is_prefill ? stat_p_hits : stat_hits).fetch_add(1, std::memory_order_relaxed);
    (is_prefill ? stat_p_prefetch_adopts : stat_prefetch_adopts).fetch_add(1, std::memory_order_relaxed);
}

void llama_expert_cache::stats_record_prefetch_use(int32_t slot) {
    if (stats_ && slots[slot].warmed_by_worker.exchange(0, std::memory_order_relaxed)) {
        stat_prefetch_useful.fetch_add(1, std::memory_order_relaxed);
    }
}

void llama_expert_cache::stats_record_prefetch_load(int32_t slot, uint32_t layer, uint32_t expert) {
    if (!stats_) {
        return;
    }
    for (int k = 0; k < N_KINDS; ++k) {
        if (get_loc(layer, (tensor_kind) k, expert).size != 0) {
            slots[slot].warmed_by_worker.store(1, std::memory_order_relaxed);
            stat_prefetch_warmed.fetch_add(1, std::memory_order_relaxed);
            break;
        }
    }
}

void llama_expert_cache::stats_record_substitution(bool is_prefill) {
    if (!stats_) {
        return;
    }
    (is_prefill ? stat_p_subs : stat_subs).fetch_add(1, std::memory_order_relaxed);
}

void llama_expert_cache::stats_record_flash_read(uint64_t bytes, uint64_t ns, bool is_prefill) {
    if (!stats_) {
        return;
    }
    (is_prefill ? stat_p_flash_reads : stat_flash_reads).fetch_add(1, std::memory_order_relaxed);
    (is_prefill ? stat_p_flash_bytes : stat_flash_bytes).fetch_add(bytes, std::memory_order_relaxed);
    (is_prefill ? stat_p_flash_ns : stat_flash_ns).fetch_add(ns, std::memory_order_relaxed);
}

uint64_t llama_expert_cache::flash_begin() {
    if (!stats_) {
        return 0;
    }
    std::lock_guard<std::mutex> lk(flash_busy_mu_);
    const uint64_t              now = stats_now_ns();
    if (flash_in_flight_ == 0) {
        flash_busy_start_ = now;
    }
    ++flash_in_flight_;
    return now;
}

void llama_expert_cache::flash_end(uint64_t token, uint64_t bytes, bool from_worker, uint64_t end_ns) {
    (void) token;
    if (!stats_) {
        return;
    }
    {
        std::lock_guard<std::mutex> lk(flash_busy_mu_);
        const uint64_t              now = end_ns != 0 ? end_ns : stats_now_ns();
        if (flash_in_flight_ > 0 && --flash_in_flight_ == 0) {
            flash_busy_ns_.fetch_add(now - flash_busy_start_, std::memory_order_relaxed);
        }
    }
    stat_flash_bytes.fetch_add(bytes, std::memory_order_relaxed);
    stat_flash_reads.fetch_add(1, std::memory_order_relaxed);
    if (from_worker) {
        flash_bytes_worker_.fetch_add(bytes, std::memory_order_relaxed);
    }
}

void llama_expert_cache::stats_set_eval_time_us(uint64_t p_eval_us, uint64_t eval_us) {
    if (!stats_) {
        return;
    }
    stat_p_eval_us.store(p_eval_us, std::memory_order_relaxed);
    stat_eval_us.store(eval_us, std::memory_order_relaxed);
}

llama_expert_cache::llama_expert_cache() {
    auto get_bool = [](const char * name, bool def) {
        const char * v = getenv(name);
        return v ? (atoi(v) != 0) : def;
    };
    auto get_int = [](const char * name, int def) {
        const char * v = getenv(name);
        return v ? atoi(v) : def;
    };

    stats_     = get_bool("LLAMA_EXPERT_CACHE_STATS", stats_);
    mlock_     = get_bool("LLAMA_EXPERT_CACHE_MLOCK", mlock_);
    ddr_copy_  = get_bool("LLAMA_EXPERT_CACHE_DDR_COPY", ddr_copy_);
    htp_async_ = get_bool("LLAMA_EXPERT_CACHE_HTP_ASYNC", htp_async_);
    decode_prefetch_force_cache_hit_ =
        get_bool("LLAMA_EXPERT_CACHE_DECODE_PREFETCH_FORCE_CACHE_HIT", decode_prefetch_force_cache_hit_);
    const char * read_percentage = getenv("LLAMA_EXPERT_CACHE_READ_PERCENTAGE");
    if (read_percentage) {
        read_percentage_ = std::clamp(atoi(read_percentage), 0, 100);
    }
    decode_prefetch_lookahead_ = get_int("LLAMA_EXPERT_CACHE_DECODE_PREFETCH_LOOKAHEAD", decode_prefetch_lookahead_);
    batch_depth_               = std::max(1, get_int("LLAMA_EXPERT_CACHE_BATCH_DEPTH", batch_depth_));
    htp_read_chunk_kb_         = get_int("LLAMA_EXPERT_CACHE_HTP_READ_CHUNK_KB", htp_read_chunk_kb_);
}

// Group slots by tensor kind in both CPU and HTP pools.
void llama_expert_cache::init_kind_first_layout(const size_t * kind_sizes) {
    size_t total = 0;
    expert_sz    = 0;
    for (int k = 0; k < N_KINDS; ++k) {
        kind_size_[k]          = kind_sizes[k];
        kind_region_offset_[k] = total;
        total += kind_sizes[k] * n_slots_;
        expert_sz += kind_sizes[k];
    }

    if (no_alloc_) {
        pool_buf_ =
            ggml_backend_buft_alloc_buffer(use_htp_pool_ ? htp_repack_buft_ : ggml_backend_cpu_buffer_type(), 0);
        ggml_backend_buffer_set_usage(pool_buf_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        return;
    }
    slots = std::make_unique<llama_expert_slot[]>(n_slots_);
    if (use_htp_pool_) {
        if (!layered_mode_) {
            pool_buf_ = ggml_backend_buft_alloc_buffer(htp_repack_buft_, total);
            GGML_ASSERT(pool_buf_ != nullptr && "expert cache: failed to allocate HTP REPACK pool");
        }

        // Pre-allocate staging for one full batch.
        const int EC_MAX_DEPTH = batch_depth_;
        size_t    max_op_bytes = 0;
        for (int k = 0; k < N_KINDS; ++k) {
            max_op_bytes += (kind_sizes[k] + 31) & ~(size_t) 31;
        }
        htp_read_buf_.resize((size_t) EC_MAX_DEPTH * max_op_bytes);

        if (decode_prefetch_lookahead_ != 0) {
            // The worker reads concurrently, so it needs separate staging.
            htp_read_buf_prefetch_.resize((size_t) EC_MAX_DEPTH * max_op_bytes);
        }
    } else {
        ddr_pool.resize(total);
        if (mlock_ && total > 0) {
            pool_mlock_ = std::make_unique<llama_mlock>();
            pool_mlock_->init(ddr_pool.data());
            pool_mlock_->grow_to(ddr_pool.size());
        }
    }
}

void llama_expert_cache::init_kind_first(uint32_t n_slots, const std::vector<size_t> & kind_sizes) {
    GGML_ASSERT(kind_sizes.size() == (size_t) N_KINDS);
    n_slots_ = n_slots;
    init_kind_first_layout(kind_sizes.data());
}

void llama_expert_cache::init_pool_tensor(ggml_tensor * tensor) const {
    // Name shared weights before graph construction can rename them concurrently.
    ggml_set_name(tensor, "expert_cache");
    if (use_htp_pool_ && htp_mark_tensor_expert_cache_) {
        htp_mark_tensor_expert_cache_(tensor);
    }
}

namespace {

static uint64_t sum_slots(const std::vector<uint32_t> & slots) {
    uint64_t total = 0;
    for (uint32_t s : slots) {
        total += s;
    }
    return total;
}

static bool try_partition_groups(const std::vector<uint32_t> &                                slots,
                                 size_t                                                       slot_bytes,
                                 size_t                                                       mbuf,
                                 uint32_t                                                     prefill_floor,
                                 std::vector<llama_expert_cache::htp_group_plan::group_def> & groups) {
    groups.clear();

    std::vector<uint32_t> active;
    active.reserve(slots.size());
    for (uint32_t l = 0; l < (uint32_t) slots.size(); ++l) {
        if (slots[l] == 0) {
            continue;
        }
        if ((uint64_t) slots[l] * slot_bytes > mbuf) {
            return false;
        }
        active.push_back(l);
    }
    if (active.empty()) {
        return true;
    }

    const uint32_t        n = (uint32_t) active.size();
    std::vector<uint64_t> prefix(n + 1, 0);
    for (uint32_t i = 0; i < n; ++i) {
        prefix[i + 1] = prefix[i] + slots[active[i]];
    }

    auto range_slots = [&](uint32_t i, uint32_t j) -> uint32_t {
        return (uint32_t) (prefix[j] - prefix[i]);
    };
    auto range_ok = [&](uint32_t i, uint32_t j) -> bool {
        const uint32_t n_sl = range_slots(i, j);
        return n_sl >= prefill_floor && (uint64_t) n_sl * slot_bytes <= mbuf;
    };

    const uint64_t total_slots         = sum_slots(slots);
    const uint32_t max_slots_per_group = slot_bytes ? (uint32_t) (mbuf / slot_bytes) : 0;
    if (max_slots_per_group == 0) {
        return false;
    }
    const uint32_t min_groups = (uint32_t) ((total_slots + max_slots_per_group - 1) / max_slots_per_group);

    for (uint32_t target_groups = min_groups; target_groups <= n; ++target_groups) {
        std::vector<std::vector<int32_t>> prev(target_groups + 1, std::vector<int32_t>(n + 1, -1));
        prev[0][0] = 0;

        for (uint32_t g = 1; g <= target_groups; ++g) {
            for (uint32_t end = 1; end <= n; ++end) {
                for (uint32_t begin = 0; begin < end; ++begin) {
                    if (prev[g - 1][begin] < 0) {
                        continue;
                    }
                    if (!range_ok(begin, end)) {
                        continue;
                    }
                    prev[g][end] = (int32_t) begin;
                    break;
                }
            }
        }

        if (prev[target_groups][n] < 0) {
            continue;
        }

        std::vector<std::pair<uint32_t, uint32_t>> ranges;
        uint32_t                                   end = n;
        for (uint32_t g = target_groups; g > 0; --g) {
            const uint32_t begin = (uint32_t) prev[g][end];
            ranges.push_back({ begin, end });
            end = begin;
        }
        std::reverse(ranges.begin(), ranges.end());

        uint32_t slot_base = 0;
        for (auto r : ranges) {
            const uint32_t begin    = r.first;
            const uint32_t end_excl = r.second;
            const uint32_t n_sl     = range_slots(begin, end_excl);
            groups.push_back({ active[begin], active[end_excl - 1], slot_base, n_sl });
            slot_base += n_sl;
        }
        return true;
    }

    return false;
}

}  // namespace

llama_expert_cache::htp_group_plan llama_expert_cache::minimum_htp_layered_groups(
    const std::vector<uint32_t> & slots_per_layer,
    size_t                        slot_bytes,
    size_t                        mbuf,
    uint32_t                      prefill_floor) {
    htp_group_plan plan;
    plan.slots_per_layer            = slots_per_layer;
    prefill_floor                   = std::max(1u, prefill_floor);
    const uint64_t        max_slots = slot_bytes ? mbuf / slot_bytes : 0;
    std::vector<uint32_t> active;
    std::vector<uint64_t> prefix{ 0 };
    for (uint32_t l = 0; l < slots_per_layer.size(); ++l) {
        if (slots_per_layer[l] == 0) {
            continue;
        }
        active.push_back(l);
        prefix.push_back(prefix.back() + slots_per_layer[l]);
    }

    const uint32_t        n = active.size();
    std::vector<uint64_t> cost(n + 1, UINT64_MAX);
    std::vector<uint32_t> prev(n + 1, 0);
    cost[0] = 0;
    for (uint32_t end = 1; end <= n; ++end) {
        for (uint32_t begin = 0; begin < end; ++begin) {
            const uint64_t required = std::max<uint64_t>(prefix[end] - prefix[begin], prefill_floor);
            if (required <= max_slots && cost[begin] != UINT64_MAX && cost[begin] + required < cost[end]) {
                cost[end] = cost[begin] + required;
                prev[end] = begin;
            }
        }
    }
    if (cost[n] > UINT32_MAX) {
        plan.error = "expert cache: cannot form prefill-safe HTP groups within the buffer limit";
        return plan;
    }

    for (uint32_t end = n; end > 0;) {
        const uint32_t begin   = prev[end];
        const uint32_t n_slots = std::max<uint64_t>(prefix[end] - prefix[begin], prefill_floor);
        const uint32_t extra   = n_slots - (prefix[end] - prefix[begin]);
        for (uint32_t i = begin; i < end; ++i) {
            plan.slots_per_layer[active[i]] += extra / (end - begin) + (i - begin < extra % (end - begin));
        }
        plan.groups.push_back({ active[begin], active[end - 1], 0, n_slots });
        end = begin;
    }
    std::reverse(plan.groups.begin(), plan.groups.end());
    uint32_t slot_base = 0;
    for (auto & group : plan.groups) {
        group.slot_base = slot_base;
        slot_base += group.n_slots;
    }
    return plan;
}

llama_expert_cache::htp_group_plan llama_expert_cache::plan_htp_layered_groups(
    const std::vector<uint32_t> & slots_per_layer,
    const std::vector<uint32_t> & floor_per_layer,
    uint32_t                      leftover,
    size_t                        slot_bytes,
    size_t                        mbuf,
    uint32_t                      prefill_floor) {
    htp_group_plan plan;
    GGML_ASSERT(slots_per_layer.size() == floor_per_layer.size());
    plan.slots_per_layer = slots_per_layer;
    if (prefill_floor == 0) {
        prefill_floor = 1;
    }

    auto try_current = [&]() -> bool {
        return try_partition_groups(plan.slots_per_layer, slot_bytes, mbuf, prefill_floor, plan.groups);
    };

    for (uint32_t r = 0; r < leftover; ++r) {
        int                                    best_l = -1;
        std::vector<htp_group_plan::group_def> best_groups;
        size_t                                 best_group_count = std::numeric_limits<size_t>::max();

        for (uint32_t l = 0; l < (uint32_t) plan.slots_per_layer.size(); ++l) {
            if (plan.slots_per_layer[l] == 0) {
                continue;
            }
            plan.slots_per_layer[l]++;
            std::vector<htp_group_plan::group_def> candidate;
            if (try_partition_groups(plan.slots_per_layer, slot_bytes, mbuf, prefill_floor, candidate) &&
                candidate.size() < best_group_count) {
                best_l           = (int) l;
                best_groups      = candidate;
                best_group_count = candidate.size();
            }
            plan.slots_per_layer[l]--;
        }

        if (best_l < 0) {
            break;
        }
        plan.slots_per_layer[(uint32_t) best_l]++;
        plan.groups = best_groups;
    }

    if (try_current()) {
        return plan;
    }

    while (!try_current()) {
        int      trim_l     = -1;
        uint32_t best_slots = 0;
        for (uint32_t l = 0; l < (uint32_t) plan.slots_per_layer.size(); ++l) {
            const uint32_t n_sl = plan.slots_per_layer[l];
            if (n_sl > floor_per_layer[l] && n_sl > best_slots) {
                trim_l     = (int) l;
                best_slots = n_sl;
            }
        }
        if (trim_l < 0) {
            plan.error = "expert cache: cannot form prefill-safe HTP groups without violating decode floor";
            plan.groups.clear();
            return plan;
        }
        plan.slots_per_layer[(uint32_t) trim_l]--;
    }

    return plan;
}

void llama_expert_cache::init_layered(uint32_t                      n_total_layers,
                                      const std::vector<uint32_t> & slots_per_layer,
                                      const std::vector<size_t> &   kind_sizes_flat) {
    GGML_ASSERT(slots_per_layer.size() == n_total_layers);
    GGML_ASSERT(kind_sizes_flat.size() == (size_t) n_total_layers * N_KINDS);

    layered_mode_   = true;
    n_total_layers_ = n_total_layers;

    // Build layer ranges in the shared slot array.
    layer_slot_base_.resize(n_total_layers + 1);
    layer_slot_base_[0] = 0;
    for (uint32_t l = 0; l < n_total_layers; ++l) {
        layer_slot_base_[l + 1] = layer_slot_base_[l] + slots_per_layer[l];
    }
    n_slots_ = layer_slot_base_[n_total_layers];

    layered_kind_size_ = kind_sizes_flat;
    layered_pool_tensors_.assign((size_t) n_total_layers * N_KINDS, nullptr);
    layered_prefill_tensors_.assign((size_t) n_total_layers * N_KINDS, nullptr);

    // Pad each kind to its largest expert slice so any layer can use the pool.
    std::vector<size_t> max_kind_sizes(N_KINDS, 0);
    for (uint32_t l = 0; l < n_total_layers; ++l) {
        if (slots_per_layer[l] == 0) {
            continue;
        }
        for (int k = 0; k < N_KINDS; ++k) {
            max_kind_sizes[k] = std::max(max_kind_sizes[k], kind_sizes_flat[l * N_KINDS + k]);
        }
    }
    init_kind_first_layout(max_kind_sizes.data());
}

void llama_expert_cache::init_shared(uint32_t                            n_total_layers,
                                     uint32_t                            n_slots,
                                     const std::array<size_t, N_KINDS> & max_kind_sizes) {
    shared_mode_    = true;
    n_slots_        = n_slots;
    n_total_layers_ = n_total_layers;

    // Pad slots to the largest slice of each kind.
    init_kind_first_layout(max_kind_sizes.data());
    max_expert_sz_ = expert_sz;

    layered_pool_tensors_.assign((size_t) n_total_layers * N_KINDS, nullptr);
}

int32_t llama_expert_cache::claim_slot(uint32_t layer, uint32_t expert) {
    std::lock_guard<std::mutex> lk(index_mu_);

    if (index_.count(make_key(layer, expert))) {
        return -1;  // already loading/occupied
    }

    const uint32_t scan_begin = layered_mode_ ? layer_slot_base_[layer] : 0;
    const uint32_t scan_end   = layered_mode_ ? layer_slot_base_[layer + 1] : n_slots_;

    for (uint32_t i = scan_begin; i < scan_end; ++i) {
        if (slots[i].state.load(std::memory_order_acquire) == (uint32_t) llama_expert_slot_state::EMPTY) {
            index_[make_key(layer, expert)] = i;
            slots[i].key = { layer, expert };
            slots[i].usage.store(0, std::memory_order_relaxed);
            slots[i].state.store((uint32_t) llama_expert_slot_state::LOADING, std::memory_order_release);
            return (int32_t) i;
        }
    }
    return -1;
}

int32_t llama_expert_cache::claim_slot_in_range(uint32_t layer, uint32_t expert, uint32_t begin, uint32_t end) {
    std::lock_guard<std::mutex> lk(index_mu_);
    for (uint32_t i = begin; i < end; ++i) {
        if (slots[i].state.load(std::memory_order_acquire) != (uint32_t) llama_expert_slot_state::EMPTY &&
            slots[i].key.layer_idx == layer && slots[i].key.expert_idx == expert) {
            return -1;
        }
    }
    for (uint32_t i = begin; i < end; ++i) {
        if (slots[i].state.load(std::memory_order_acquire) == (uint32_t) llama_expert_slot_state::EMPTY) {
            index_[make_key(layer, expert)] = i;
            slots[i].key = { layer, expert };
            slots[i].usage.store(0, std::memory_order_relaxed);
            slots[i].state.store((uint32_t) llama_expert_slot_state::LOADING, std::memory_order_release);
            return (int32_t) i;
        }
    }
    return -1;
}

int32_t llama_expert_cache::claim_or_copy_slot_in_range(uint32_t layer, uint32_t expert, uint32_t begin, uint32_t end) {
    int32_t source = -1;
    if (layered_mode_ && ddr_copy_) {
        std::lock_guard<std::mutex> lock(index_mu_);
        const auto                  it = index_.find(make_key(layer, expert));
        if (it != index_.end() && (it->second < begin || it->second >= end) &&
            slots[it->second].state.load(std::memory_order_acquire) == (uint32_t) llama_expert_slot_state::OCCUPIED) {
            source = (int32_t) it->second;
        }
    }

    const int32_t dest = claim_slot_in_range(layer, expert, begin, end);
    if (dest < 0 || source < 0) {
        return dest;
    }

    for (int k = 0; k < N_KINDS; ++k) {
        if (kind_size_[k] != 0) {
            memcpy(kind_slot_ptr((tensor_kind) k, dest), kind_slot_ptr((tensor_kind) k, source), kind_size_[k]);
        }
    }
    stats_record_prefetch_use(source);
    // Prefill may still be reading the copy outside the decode range.
    if (slots[source].readers == 0) {
        release(source);
    }
    mark_occupied(dest);
    return dest;
}

void llama_expert_cache::mark_occupied(int32_t slot) {
    {
        std::lock_guard<std::mutex> lock(index_mu_);
        slots[slot].state.store((uint32_t) llama_expert_slot_state::OCCUPIED, std::memory_order_release);
    }
    slots_cv_.notify_all();
}

void llama_expert_cache::record_use(int32_t slot) {
    stats_record_prefetch_use(slot);
    slots[slot].usage.fetch_add(1, std::memory_order_relaxed);
    slots[slot].last_use.store(next_tick(), std::memory_order_relaxed);
}

void llama_expert_cache::release(int32_t slot) {
    {
        std::lock_guard<std::mutex> lk(index_mu_);
        GGML_ASSERT(slots[slot].readers == 0);
        const auto it = index_.find(make_key(slots[slot].key.layer_idx, slots[slot].key.expert_idx));
        if (it != index_.end() && it->second == (uint32_t) slot) {
            index_.erase(it);
        }
        slots[slot].key = { 0, 0 };
        if (slots[slot].warmed_by_worker.exchange(0, std::memory_order_relaxed) && stats_) {
            stat_prefetch_wasted.fetch_add(1, std::memory_order_relaxed);
        }
        slots[slot].state.store((uint32_t) llama_expert_slot_state::EMPTY, std::memory_order_release);
    }
    slots_cv_.notify_all();
}

int32_t llama_expert_cache::evict_lfu_excluding(const int32_t * pinned,
                                                int             n_pinned,
                                                uint32_t        layer,
                                                uint32_t        begin,
                                                uint32_t        end) {
    while (true) {
        int32_t  victim   = -1;
        uint64_t min_use  = std::numeric_limits<uint64_t>::max();
        uint64_t min_tick = std::numeric_limits<uint64_t>::max();

        uint32_t scan_begin, scan_end;
        if (layered_mode_ && layer < n_total_layers_) {
            scan_begin = layer_slot_base_[layer];
            scan_end   = layer_slot_base_[layer + 1];
        } else {
            scan_begin = 0;
            scan_end   = n_slots_;
        }

        scan_begin = std::max(scan_begin, begin);
        scan_end   = std::min(scan_end, end);

        for (uint32_t i = scan_begin; i < scan_end; ++i) {
            if ((llama_expert_slot_state) slots[i].state.load(std::memory_order_acquire) !=
                llama_expert_slot_state::OCCUPIED) {
                continue;
            }
            if (slots[i].readers != 0) {
                continue;
            }

            bool is_pinned = false;
            for (int p = 0; p < n_pinned; ++p) {
                if (pinned[p] == (int32_t) i) {
                    is_pinned = true;
                    break;
                }
            }
            if (is_pinned) {
                continue;
            }

            uint64_t u = slots[i].usage.load(std::memory_order_relaxed);
            uint64_t t = slots[i].last_use.load(std::memory_order_relaxed);
            if (u < min_use || (u == min_use && t < min_tick)) {
                min_use  = u;
                min_tick = t;
                victim   = (int32_t) i;
            }
        }

        if (victim < 0) {
            return -1;
        }

        uint32_t expected = (uint32_t) llama_expert_slot_state::OCCUPIED;
        if (slots[victim].state.compare_exchange_strong(expected, (uint32_t) llama_expert_slot_state::EVICTING,
                                                        std::memory_order_acq_rel)) {
            return victim;
        }
        // Retry if another eviction claimed the victim.
    }
}

int32_t llama_expert_cache::evict_lru_excluding(const int32_t * pinned,
                                                int             n_pinned,
                                                uint32_t        layer,
                                                uint32_t        begin,
                                                uint32_t        end) {
    while (true) {
        int32_t  victim   = -1;
        uint64_t min_tick = std::numeric_limits<uint64_t>::max();

        uint32_t scan_begin, scan_end;
        if (layered_mode_ && layer < n_total_layers_) {
            scan_begin = layer_slot_base_[layer];
            scan_end   = layer_slot_base_[layer + 1];
        } else {
            scan_begin = 0;
            scan_end   = n_slots_;
        }

        scan_begin = std::max(scan_begin, begin);
        scan_end   = std::min(scan_end, end);

        for (uint32_t i = scan_begin; i < scan_end; ++i) {
            if ((llama_expert_slot_state) slots[i].state.load(std::memory_order_acquire) !=
                llama_expert_slot_state::OCCUPIED) {
                continue;
            }
            if (slots[i].readers != 0) {
                continue;
            }

            bool is_pinned = false;
            for (int p = 0; p < n_pinned; ++p) {
                if (pinned[p] == (int32_t) i) {
                    is_pinned = true;
                    break;
                }
            }
            if (is_pinned) {
                continue;
            }

            uint64_t t = slots[i].last_use.load(std::memory_order_relaxed);
            if (t < min_tick) {
                min_tick = t;
                victim   = (int32_t) i;
            }
        }

        if (victim < 0) {
            return -1;
        }

        uint32_t expected = (uint32_t) llama_expert_slot_state::OCCUPIED;
        if (slots[victim].state.compare_exchange_strong(expected, (uint32_t) llama_expert_slot_state::EVICTING,
                                                        std::memory_order_acq_rel)) {
            return victim;
        }
    }
}

uint8_t * llama_expert_cache::kind_slot_ptr(tensor_kind k, uint32_t slot) {
    GGML_ASSERT((int) k < N_KINDS && slot < n_slots_);
    GGML_ASSERT(kind_size_[k] > 0 && "kind_slot_ptr called for absent kind");
    if (layered_mode_ && use_htp_pool_) {
        uint32_t il           = slots[slot].key.layer_idx;
        uint32_t g            = layer_group_base_[il];
        // Kind-first offset within the group: sum of kind_size_[j]*group_n_slots_ for j<k.
        size_t   grp_kind_off = 0;
        for (int j = 0; j < k; ++j) {
            grp_kind_off += (size_t) group_n_slots_[g] * kind_size_[j];
        }
        uint32_t slot_in_group = slot - group_slot_base_[g];
        return group_base_ptr_[g] + grp_kind_off + (size_t) slot_in_group * kind_size_[k];
    }
    auto pool_base = [&]() -> uint8_t * {
        return use_htp_pool_ ? static_cast<uint8_t *>(ggml_backend_buffer_get_base(pool_buf_)) : ddr_pool.data();
    };
    return pool_base() + kind_region_offset_[k] + (size_t) slot * kind_size_[k];
}

const uint8_t * llama_expert_cache::kind_slot_ptr(tensor_kind k, uint32_t slot) const {
    GGML_ASSERT((int) k < N_KINDS && slot < n_slots_);
    GGML_ASSERT(kind_size_[k] > 0 && "kind_slot_ptr called for absent kind");
    if (layered_mode_ && use_htp_pool_) {
        uint32_t il           = slots[slot].key.layer_idx;
        uint32_t g            = layer_group_base_[il];
        size_t   grp_kind_off = 0;
        for (int j = 0; j < k; ++j) {
            grp_kind_off += (size_t) group_n_slots_[g] * kind_size_[j];
        }
        uint32_t slot_in_group = slot - group_slot_base_[g];
        return group_base_ptr_[g] + grp_kind_off + (size_t) slot_in_group * kind_size_[k];
    }
    auto pool_base = [&]() -> const uint8_t * {
        return use_htp_pool_ ? static_cast<const uint8_t *>(ggml_backend_buffer_get_base(pool_buf_)) : ddr_pool.data();
    };
    return pool_base() + kind_region_offset_[k] + (size_t) slot * kind_size_[k];
}

uint64_t llama_expert_cache::next_tick() {
    return tick.fetch_add(1, std::memory_order_relaxed);
}

void llama_expert_cache::build_offset_map(uint32_t                              n_layer,
                                          uint32_t                              n_expert,
                                          uint32_t                              n_expert_used,
                                          const std::vector<expert_slice_loc> & locs) {
    n_layer_       = n_layer;
    n_expert_      = n_expert;
    n_expert_used_ = n_expert_used;
    offsets_       = locs;

    layer_prefetch_done_token_ = std::make_unique<std::atomic<uint64_t>[]>(n_layer);
    for (uint32_t l = 0; l < n_layer; ++l) {
        layer_prefetch_done_token_[l].store(UINT64_MAX, std::memory_order_relaxed);
    }
}

void llama_expert_cache::release_slots(op_params_t & op) {
    {
        std::lock_guard<std::mutex> lock(reserve_mu_);
        std::lock_guard<std::mutex> index_lock(index_mu_);
        for (int32_t s : op.pinned) {
            GGML_ASSERT(slots[s].readers > 0);
            --slots[s].readers;
        }
        op.pinned.clear();
    }
    slots_cv_.notify_all();
}

void llama_expert_cache::release_slots_fn(ggml_tensor *, const ggml_tensor *, int ith, int, void * userdata) {
    if (ith == 0) {
        auto * op = static_cast<op_params_t *>(userdata);
        op->cache->release_slots(*op);
    }
}

void llama_expert_cache::id_to_slot_fn(struct ggml_tensor * dst,
                                       const struct ggml_tensor * /*src*/,
                                       int ith,
                                       int /*nth*/,
                                       void * userdata) {
    if (ith != 0) {
        return;  // single-threaded (n_tasks=1)
    }

    auto *       p = static_cast<op_params_t *>(userdata);
    const size_t n = ggml_nelements(dst);
    GGML_ASSERT(n <= p->slot_ids.size());
    memcpy(dst->data, p->slot_ids.data(), n * sizeof(int32_t));
}

void llama_expert_cache::reroute_fn(struct ggml_tensor * dst,
                                    const struct ggml_tensor * /*a*/,
                                    const struct ggml_tensor * b,
                                    int                        ith,
                                    int /*nth*/,
                                    void * userdata) {
    if (ith != 0) {
        return;  // single-threaded (n_tasks=1)
    }

    auto *                       p     = static_cast<op_params_t *>(userdata);
    auto *                       cache = p->cache;
    std::unique_lock<std::mutex> reservation_lock(cache->reserve_mu_);
    GGML_ASSERT(p->pinned.empty());
    const uint32_t il            = p->layer_idx;
    const int64_t  n_expert_used = dst->ne[0];
    const int64_t  n_tokens      = dst->ne[1];
    int32_t *      ids           = static_cast<int32_t *>(dst->data);

    // b holds selection scores [n_expert, n_tokens] for cache substitution.
    const float * probs_data     = (b && cache->threshold_pct_ > 0.0f) ? static_cast<const float *>(b->data) : nullptr;
    const int64_t n_expert_total = b ? b->ne[0] : 0;
    const float   threshold_frac = cache->threshold_pct_ / 100.0f;

    const size_t needed_sz = (size_t) (n_expert_used * n_tokens);
    p->slot_ids.resize(needed_sz);
    int32_t * slot_out = p->slot_ids.data();

    // Layered prefill spans a CPU pool or HTP group, while decode uses the layer's slots.
    const bool prefill = (n_tokens > 1);
    uint32_t   scan_s, scan_e, slot_base_u;
    if (prefill && cache->use_htp_pool_ && cache->layered_mode_) {
        uint32_t g  = cache->layer_group_base_[il];
        scan_s      = cache->group_slot_base_[g];
        scan_e      = scan_s + cache->group_n_slots_[g];
        slot_base_u = scan_s;
    } else if (prefill) {
        // CPU or non-layered HTP prefill uses the full pool.
        scan_s      = 0;
        scan_e      = cache->n_slots_;
        slot_base_u = 0;
    } else {
        scan_s      = cache->layered_mode_ ? cache->layer_slot_base_[il] : 0;
        scan_e      = cache->layered_mode_ ? cache->layer_slot_base_[il + 1] : cache->n_slots_;
        slot_base_u = scan_s;
    }
    auto to_local = [slot_base_u](int32_t s) -> int32_t {
        return s - (int32_t) slot_base_u;
    };

    // Build cache lookup: scan the relevant slot range, filter by layer.
    p->cached_slots.resize(cache->n_expert_);
    int32_t * cached_slots = p->cached_slots.data();
    memset(cached_slots, 0xFF, (size_t) cache->n_expert_ * sizeof(int32_t));  // fill -1
    for (uint32_t s = scan_s; s < scan_e; ++s) {
        if ((llama_expert_slot_state) cache->slots[s].state.load(std::memory_order_acquire) !=
            llama_expert_slot_state::OCCUPIED) {
            continue;
        }
        if (cache->slots[s].key.layer_idx != il) {
            continue;
        }
        cached_slots[cache->slots[s].key.expert_idx] = (int32_t) s;
    }

    // Use stack storage for small expert sets.
    constexpr int64_t STACK_MAX = 64;
    int32_t           pinned_stack[STACK_MAX];
    std::unique_ptr<int32_t[]> pinned_heap(((size_t) cache->n_expert_ <= STACK_MAX) ? nullptr :
                                                                                      new int32_t[cache->n_expert_]);
    int32_t *                  pinned   = pinned_heap ? pinned_heap.get() : pinned_stack;
    int               pinned_n = 0;

    // Pin each selected expert once, even when several tokens use it.
    p->pinned_mark.resize(cache->n_expert_);
    uint8_t * pin_mark = p->pinned_mark.data();
    memset(pin_mark, 0, (size_t) cache->n_expert_);

    // Reuse decode scratch across tokens.
    constexpr int64_t MISS_STACK_MAX   = 64;
    constexpr int64_t EXPERT_STACK_MAX = 256;
    int64_t           miss_r_stack[MISS_STACK_MAX];
    std::unique_ptr<int64_t[]> miss_r_heap((n_expert_used <= MISS_STACK_MAX) ? nullptr : new int64_t[n_expert_used]);
    int64_t *                  miss_r = miss_r_heap ? miss_r_heap.get() : miss_r_stack;
    uint8_t           claimed_e_stack[EXPERT_STACK_MAX];
    std::unique_ptr<uint8_t[]> claimed_e_heap((n_expert_total <= EXPERT_STACK_MAX) ? nullptr :
                                                                                     new uint8_t[n_expert_total]);
    uint8_t *                  claimed_e = claimed_e_heap ? claimed_e_heap.get() : claimed_e_stack;

    miss_candidate   cands_stack[MISS_STACK_MAX];
    std::unique_ptr<miss_candidate[]> cands_heap((n_expert_used <= MISS_STACK_MAX) ? nullptr :
                                                                                     new miss_candidate[n_expert_used]);
    miss_candidate *                  cands = cands_heap ? cands_heap.get() : cands_stack;
    int64_t          ranked_stack[MISS_STACK_MAX];
    std::unique_ptr<int64_t[]> ranked_heap((n_expert_used <= MISS_STACK_MAX) ? nullptr : new int64_t[n_expert_used]);
    int64_t *                  ranked = ranked_heap ? ranked_heap.get() : ranked_stack;

    // Treat a layer as covered when prefetch warmed all predicted experts.
    bool          forced_covered    = false;
    const float * forced_probs_data = b ? static_cast<const float *>(b->data) : nullptr;
    if (cache->decode_prefetch_force_cache_hit_ && !prefill) {
        forced_covered = (cache->layer_prefetch_done_token_[il].load(std::memory_order_relaxed) == p->token);
    }

    for (int64_t t = 0; t < n_tokens; ++t) {
        int32_t * row = ids + t * n_expert_used;

        // Resolve substitutions before pinning the selected slots.
        for (int64_t r = 0; r < n_expert_used; ++r) {
            int32_t eid  = row[r];
            int32_t slot = cached_slots[eid];

            if (forced_covered) {
                if (slot < 0) {
                    // Use the cached expert with the highest probability that is not already selected.
                    const float * token_probs = forced_probs_data ? (forced_probs_data + t * n_expert_total) : nullptr;
                    float         best_prob   = -1.0f;
                    int32_t       best_eid    = -1;
                    int32_t       best_slot   = -1;

                    for (int64_t e = 0; e < n_expert_total; ++e) {
                        if ((int32_t) e == eid) {
                            continue;
                        }
                        if (cached_slots[e] < 0) {
                            continue;
                        }

                        bool already_selected = false;
                        for (int64_t rr = 0; rr < n_expert_used; ++rr) {
                            if (row[rr] == (int32_t) e) {
                                already_selected = true;
                                break;
                            }
                        }
                        if (already_selected) {
                            continue;
                        }

                        float p = token_probs ? token_probs[e] : 0.0f;
                        if (p <= best_prob && best_eid >= 0) {
                            continue;
                        }

                        best_prob = p;
                        best_eid  = (int32_t) e;
                        best_slot = cached_slots[e];
                    }

                    if (best_eid >= 0) {
                        row[r] = best_eid;
                        eid    = best_eid;
                        slot   = best_slot;
                        if (cache->stats_) {
                            cache->stat_forced_subs.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                    // Leave stale markers as normal misses.
                } else if (cache->stats_) {
                    cache->stat_prefetch_hits.fetch_add(1, std::memory_order_relaxed);
                }
            } else
                // Replace an uncached pick with a unique cached alternative within the threshold.
                if (slot < 0 && probs_data && threshold_frac > 0.0f) {
                    const float * token_probs = probs_data + t * n_expert_total;
                    const float   min_prob    = token_probs[eid] * (1.0f - threshold_frac);
                    float         best_prob   = -1.0f;
                    int32_t       best_eid    = -1;
                    int32_t       best_slot   = -1;

                    for (int64_t e = 0; e < n_expert_total; ++e) {
                        if ((int32_t) e == eid) {
                            continue;
                        }
                        float p = token_probs[e];
                        if (p < min_prob || p <= best_prob) {
                            continue;
                        }
                        if (cached_slots[e] < 0) {
                            continue;
                        }

                        bool already_selected = false;
                        for (int64_t rr = 0; rr < n_expert_used; ++rr) {
                            if (row[rr] == (int32_t) e) {
                                already_selected = true;
                                break;
                            }
                        }
                        if (already_selected) {
                            continue;
                        }

                        best_prob = p;
                        best_eid  = (int32_t) e;
                        best_slot = cached_slots[e];
                    }

                    if (best_eid >= 0) {
                        row[r] = best_eid;
                        eid    = best_eid;
                        slot   = best_slot;
                        cache->stats_record_substitution(prefill);
                    }
                }
        }

        // Cap decode misses after substitutions when read percentage is enabled.
        const bool read_percentage_enabled = cache->read_percentage_ > 0 && cache->read_percentage_ < 100;
        if (read_percentage_enabled && !prefill) {
            // Gather the row's remaining misses (slot < 0 after the first pass).
            int64_t n_miss = 0;
            for (int64_t r = 0; r < n_expert_used; ++r) {
                if (cached_slots[row[r]] < 0) {
                    miss_r[n_miss++] = r;
                }
            }

            if (n_miss > 0) {
                // Read every miss when the layer has no cached substitutes.
                bool any_cached = false;
                for (int64_t e = 0; e < n_expert_total; ++e) {
                    if (cached_slots[e] >= 0) {
                        any_cached = true;
                        break;
                    }
                }

                if (any_cached) {
                    // Read probabilities even when threshold substitution is disabled.
                    const float * token_probs =
                        b ? (static_cast<const float *>(b->data) + t * n_expert_total) : nullptr;

                    // Do not reuse a substitute within the same token.
                    memset(claimed_e, 0, (size_t) n_expert_total);

                    // Find the closest available cached expert.
                    auto find_best = [&](int32_t eid, float eid_prob) -> std::pair<int32_t, int32_t> {
                        float   best_gap  = std::numeric_limits<float>::max();
                        int32_t best_eid  = -1;
                        int32_t best_slot = -1;

                        for (int64_t e = 0; e < n_expert_total; ++e) {
                            if ((int32_t) e == eid) {
                                continue;
                            }
                            if (cached_slots[e] < 0) {
                                continue;
                            }
                            if (claimed_e[e]) {
                                continue;
                            }
                            bool already_selected = false;
                            for (int64_t rr = 0; rr < n_expert_used; ++rr) {
                                if (row[rr] == (int32_t) e) {
                                    already_selected = true;
                                    break;
                                }
                            }
                            if (already_selected) {
                                continue;
                            }

                            float e_prob = token_probs ? token_probs[e] : 0.0f;
                            float gap    = std::fabs(eid_prob - e_prob);
                            if (gap > best_gap) {
                                continue;
                            }
                            if (gap == best_gap && best_eid >= 0 && (int32_t) e >= best_eid) {
                                continue;  // tie-break: lower expert id wins
                            }

                            best_gap  = gap;
                            best_eid  = (int32_t) e;
                            best_slot = cached_slots[e];
                        }

                        return { best_eid, best_slot };
                    };

                    for (int64_t mi = 0; mi < n_miss; ++mi) {
                        int64_t r                  = miss_r[mi];
                        int32_t eid                = row[r];
                        float   eid_prob           = token_probs ? token_probs[eid] : 0.0f;
                        auto [best_eid, best_slot] = find_best(eid, eid_prob);
                        float gap = (best_eid >= 0 && token_probs) ? std::fabs(eid_prob - token_probs[best_eid]) :
                                                                     std::numeric_limits<float>::max();
                        cands[mi] = { r, best_eid, best_slot, gap, best_eid >= 0 };
                    }

                    // Always read misses with no substitute and rank the rest by probability gap.
                    int64_t n_ranked = 0;
                    for (int64_t mi = 0; mi < n_miss; ++mi) {
                        if (cands[mi].has_candidate) {
                            ranked[n_ranked++] = mi;
                        }
                    }
                    std::sort(ranked, ranked + n_ranked, [&](int64_t a, int64_t b2) {
                        if (cands[a].gap != cands[b2].gap) {
                            return cands[a].gap > cands[b2].gap;    // worst gap first
                        }
                        return row[cands[a].r] < row[cands[b2].r];  // deterministic tie-break
                    });

                    const int64_t n_rankable = n_ranked;
                    const int64_t n_read     = n_rankable > 0 ? (n_rankable * cache->read_percentage_ + 99) / 100 : 0;

                    // Read the worst matches and substitute the rest in rank order.
                    for (int64_t k = n_read; k < n_rankable; ++k) {
                        int64_t mi                 = ranked[k];
                        int64_t r                  = cands[mi].r;
                        int32_t eid                = row[r];
                        float   eid_prob           = token_probs ? token_probs[eid] : 0.0f;
                        auto [best_eid, best_slot] = find_best(eid, eid_prob);
                        if (best_eid < 0) {
                            continue;  // no substitute remains
                        }
                        claimed_e[best_eid] = 1;
                        row[r]              = best_eid;
                        if (cache->stats_) {
                            cache->stat_read_pct_subs.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                }
            }
        }

        // Pin every resolved slot before eviction.
        for (int64_t r = 0; r < n_expert_used; ++r) {
            int32_t eid  = row[r];
            int32_t slot = cached_slots[eid];

            if (slot >= 0 && !pin_mark[eid]) {
                pin_mark[eid]      = 1;
                pinned[pinned_n++] = slot;
                cache->record_use(slot);
            }
        }
    }

    // Wait for a complete working set without retaining a partial reservation.
    memset(pin_mark, 0, (size_t) cache->n_expert_);
    uint32_t n_required = 0;
    for (size_t i = 0; i < needed_sz; ++i) {
        if (!pin_mark[ids[i]]) {
            pin_mark[ids[i]] = 1;
            ++n_required;
        }
    }
    GGML_ASSERT(n_required <= scan_e - scan_s && "expert cache: increase the cache size for this microbatch");
    for (;;) {
        std::unique_lock<std::mutex> lock(cache->index_mu_);
        uint32_t                     available = 0;
        memset(cached_slots, 0xFF, (size_t) cache->n_expert_ * sizeof(int32_t));
        for (uint32_t s = scan_s; s < scan_e; ++s) {
            const auto & slot  = cache->slots[s];
            const auto   state = (llama_expert_slot_state) slot.state.load(std::memory_order_acquire);
            if (state == llama_expert_slot_state::EMPTY ||
                (state == llama_expert_slot_state::OCCUPIED && slot.readers == 0)) {
                ++available;
            }
            if (state == llama_expert_slot_state::OCCUPIED && slot.key.layer_idx == il) {
                int32_t & found = cached_slots[slot.key.expert_idx];
                if (found < 0 || slot.readers > 0) {
                    found = (int32_t) s;
                }
            }
        }
        uint32_t missing = n_required;
        pinned_n         = 0;
        for (uint32_t e = 0; e < cache->n_expert_; ++e) {
            if (pin_mark[e] && cached_slots[e] >= 0) {
                const int32_t s    = cached_slots[e];
                pinned[pinned_n++] = s;
                --missing;
                if (cache->slots[s].readers == 0) {
                    --available;
                }
            }
        }
        if (available >= missing) {
            break;
        }
        reservation_lock.unlock();
        cache->slots_cv_.wait(lock);
        lock.unlock();
        reservation_lock.lock();
    }

    // Load remaining misses and write indices relative to the selected slot range.
    const bool     use_lru     = (cache->policy_ == LLAMA_EXPERT_CACHE_POLICY_LRU);
    // Prefill may evict other layers' experts within the scan range.
    const uint32_t evict_layer = prefill ? UINT32_MAX : il;

    auto do_evict = [&]() -> int32_t {
        return use_lru ? cache->evict_lru_excluding(pinned, pinned_n, evict_layer, scan_s, scan_e) :
                         cache->evict_lfu_excluding(pinned, pinned_n, evict_layer, scan_s, scan_e);
    };

    std::vector<llama_expert_cache::load_req> to_load;
    to_load.reserve((size_t) n_tokens * n_expert_used);

    struct pending_load_cleanup {
        llama_expert_cache *          cache;
        const std::vector<load_req> & requests;
        int                           exceptions = std::uncaught_exceptions();

        ~pending_load_cleanup() {
            if (std::uncaught_exceptions() <= exceptions) {
                return;
            }
            for (const auto & req : requests) {
                if (cache->slots[req.slot].state.load(std::memory_order_acquire) ==
                    (uint32_t) llama_expert_slot_state::LOADING) {
                    cache->release(req.slot);
                }
            }
        }
    } cleanup{ cache, to_load };

    for (int64_t t = 0; t < n_tokens; ++t) {
        int32_t * row = ids + t * n_expert_used;

        for (int64_t r = 0; r < n_expert_used; ++r) {
            int32_t eid  = row[r];
            int32_t slot = cached_slots[eid];

            if (slot < 0) {
                // Copy prefill weights into the decode range when enabled.
                slot                    = cache->claim_or_copy_slot_in_range(il, (uint32_t) eid, scan_s, scan_e);
                // Adopt worker-filled slots separately from DDR copies.
                bool via_prefetch_adopt = false;
                {
                    // Wait for pending reads before retrying the claim.
                    while (slot < 0) {
                        // Reuse this expert if the worker is already loading it.
                        int32_t existing_slot = -1;
                        {
                            std::lock_guard<std::mutex> lk(cache->index_mu_);
                            for (uint32_t s = scan_s; s < scan_e; ++s) {
                                if (cache->slots[s].state.load(std::memory_order_acquire) !=
                                        (uint32_t) llama_expert_slot_state::EMPTY &&
                                    cache->slots[s].key.layer_idx == il &&
                                    cache->slots[s].key.expert_idx == (uint32_t) eid) {
                                    existing_slot = (int32_t) s;
                                    break;
                                }
                            }
                        }
                        if (existing_slot >= 0) {
                            auto existing_state = (llama_expert_slot_state) cache->slots[existing_slot].state.load(
                                std::memory_order_acquire);
                            if (existing_state == llama_expert_slot_state::OCCUPIED) {
                                slot               = existing_slot;
                                via_prefetch_adopt = true;
                                break;
                            }
                            if (existing_state == llama_expert_slot_state::LOADING ||
                                existing_state == llama_expert_slot_state::EVICTING) {
                                std::unique_lock<std::mutex> lock(cache->index_mu_);
                                cache->slots_cv_.wait(lock, [&] {
                                    const auto state = (llama_expert_slot_state) cache->slots[existing_slot].state.load(
                                        std::memory_order_acquire);
                                    return state != llama_expert_slot_state::LOADING &&
                                           state != llama_expert_slot_state::EVICTING;
                                });
                                continue;
                            }
                        }

                        int32_t victim = do_evict();
                        if (victim >= 0) {
                            cache->release(victim);
                            slot = cache->claim_slot_in_range(il, (uint32_t) eid, scan_s, scan_e);
                            continue;
                        }
                        // Wait for a worker load if no slot is evictable.
                        std::unique_lock<std::mutex> lock(cache->index_mu_);
                        int32_t                      loading_slot = -1;
                        for (uint32_t s = scan_s; s < scan_e; ++s) {
                            if (std::any_of(to_load.begin(), to_load.end(),
                                            [s](const load_req & req) { return req.slot == (int32_t) s; })) {
                                continue;
                            }
                            if ((llama_expert_slot_state) cache->slots[s].state.load(std::memory_order_acquire) ==
                                llama_expert_slot_state::LOADING) {
                                loading_slot = (int32_t) s;
                                break;
                            }
                        }
                        if (loading_slot < 0) {
                            break;
                        }
                        cache->slots_cv_.wait(lock, [&] {
                            return (llama_expert_slot_state) cache->slots[loading_slot].state.load(
                                       std::memory_order_acquire) != llama_expert_slot_state::LOADING;
                        });
                        lock.unlock();
                        slot = cache->claim_slot_in_range(il, (uint32_t) eid, scan_s, scan_e);
                    }
                }
                if (slot < 0) {
                    GGML_ABORT(
                        "expert cache: no slot for layer=%u expert=%d "
                        "(n_tokens=%ld n_expert_used=%ld pinned_n=%d n_slots=%u, "
                        "prefill uses full pool/group; increase LLAMA_EXPERT_CACHE_SIZE)",
                        il, eid, (long) n_tokens, (long) n_expert_used, pinned_n, cache->n_slots_);
                }
                // Read flash only for slots still loading.
                auto slot_state = (llama_expert_slot_state) cache->slots[slot].state.load(std::memory_order_acquire);
                if (slot_state == llama_expert_slot_state::LOADING) {
                    cache->stats_record_miss(prefill);
                    to_load.push_back({ slot, il, (uint32_t) eid });
                } else if (via_prefetch_adopt) {
                    cache->stats_record_prefetch_adopt(prefill);
                } else {
                    // A DDR copy has already filled this slot.
                    cache->stats_record_ddr_copy(prefill);
                }
                cached_slots[eid]  = slot;
                pinned[pinned_n++] = slot;
                cache->record_use(slot);
            } else {
                cache->stats_record_hit(prefill);
            }

            slot_out[t * n_expert_used + r] = to_local(slot);
        }
    }

    // Load reserved slots in batches.
    if (!to_load.empty()) {
        const uint32_t D = (uint32_t) cache->batch_depth_;
        for (size_t i = 0; i < to_load.size(); i += D) {
            const uint32_t w = (uint32_t) std::min((size_t) D, to_load.size() - i);
            if (cache->use_htp_pool_) {
                cache->load_experts_batch_htp(&to_load[i], w, prefill);
            } else {
                cache->load_experts_batch_cpu(&to_load[i], w, prefill, cpu_load_origin::main);
            }
        }
    }

    p->pinned.assign(pinned, pinned + pinned_n);
    for (int32_t s : p->pinned) {
        ++cache->slots[s].readers;
    }
}

void llama_expert_cache::snapshot_fn(ggml_tensor * dst, const ggml_tensor *, int ith, int, void * userdata) {
    if (ith != 0) {
        return;
    }
    auto * p     = static_cast<op_params_t *>(userdata);
    auto * cache = p->cache;
    {
        std::lock_guard<std::mutex> lock(cache->worker_mu_);
        auto &                      snap = cache->prefetch_snapshot_;
        const size_t                n    = std::min(snap.hidden.size(), (size_t) ggml_nelements(dst));
        memcpy(snap.hidden.data(), dst->data, n * sizeof(float));
        snap.layer    = p->layer_idx;
        snap.token_id = p->token;
        ++cache->worker_wake_seq_;
    }
    cache->worker_cv_.notify_all();
}

void llama_expert_cache::start_worker() {
    if (decode_prefetch_lookahead_ == 0 || worker_thread_.joinable()) {
        return;  // disabled, or already started
    }
    worker_shutdown_.store(false, std::memory_order_relaxed);
    worker_thread_ = std::thread(&llama_expert_cache::worker_loop, this);
}

void llama_expert_cache::stop_worker() {
    if (!worker_thread_.joinable()) {
        return;  // never started (or already stopped)
    }
    worker_shutdown_.store(true, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lk(worker_mu_);
        worker_wake_seq_++;
    }
    worker_cv_.notify_all();
    worker_thread_.join();
}

bool llama_expert_cache::read_snapshot(uint32_t & out_layer, uint64_t & out_token, std::vector<float> & out_hidden) {
    std::lock_guard<std::mutex> lock(worker_mu_);
    if (worker_shutdown_.load(std::memory_order_relaxed)) {
        return false;
    }
    out_layer  = prefetch_snapshot_.layer;
    out_token  = prefetch_snapshot_.token_id;
    out_hidden = prefetch_snapshot_.hidden;
    return true;
}

// Warm one layered-cache range without using main-thread scratch buffers.
void llama_expert_cache::warm_cache(uint32_t layer, const std::vector<int32_t> & predicted_expert_ids, uint64_t token) {
    if (!layered_mode_) {
        return;  // prefetch requires per-layer slot ranges
    }
    if (layer + 1 >= layer_slot_base_.size()) {
        return;
    }

    const uint32_t begin = layer_slot_base_[layer];
    const uint32_t end   = layer_slot_base_[layer + 1];
    if (begin >= end) {
        return;  // dense layer or no slots
    }

    std::unique_lock<std::mutex> lock(reserve_mu_);

    std::vector<load_req> to_load;
    to_load.reserve(predicted_expert_ids.size());

    bool full_warm = true;

    for (int32_t eid : predicted_expert_ids) {
        if (eid < 0 || (uint32_t) eid >= n_expert_) {
            full_warm = false;
            continue;
        }

        bool cached = false;
        for (uint32_t s = begin; s < end; ++s) {
            if (slots[s].state.load(std::memory_order_acquire) != (uint32_t) llama_expert_slot_state::EMPTY &&
                slots[s].key.layer_idx == layer && slots[s].key.expert_idx == (uint32_t) eid) {
                cached = true;
                break;
            }
        }
        if (cached) {
            if (stats_) {
                stat_warm_skip_cached.fetch_add(1, std::memory_order_relaxed);
            }
            continue;
        }

        int32_t slot = claim_slot_in_range(layer, (uint32_t) eid, begin, end);
        if (slot < 0) {
            // Eviction checks reader counts even though the worker has no pins.
            const bool use_lru = (policy_ == LLAMA_EXPERT_CACHE_POLICY_LRU);
            int32_t victim = use_lru ? evict_lru_excluding(nullptr, 0, layer) : evict_lfu_excluding(nullptr, 0, layer);
            if (victim >= 0) {
                if (stats_) {
                    stat_worker_evictions.fetch_add(1, std::memory_order_relaxed);
                }
                release(victim);
                slot = claim_slot_in_range(layer, (uint32_t) eid, begin, end);
            }
        }
        if (slot < 0) {
            if (stats_) {
                stat_warm_skip_noslot.fetch_add(1, std::memory_order_relaxed);
            }
            full_warm = false;
            continue;  // prefetch is optional
        }

        to_load.push_back({ slot, layer, (uint32_t) eid });
    }
    lock.unlock();

    if (!to_load.empty()) {
        // Match the batch size used to allocate worker staging.
        const uint32_t D = (uint32_t) batch_depth_;
        for (size_t idx = 0; idx < to_load.size(); idx += D) {
            const uint32_t w = (uint32_t) std::min((size_t) D, to_load.size() - idx);
            if (use_htp_pool_) {
                load_experts_batch_htp_prefetch(&to_load[idx], w);
            } else {
                load_experts_batch_cpu(&to_load[idx], w, /*is_prefill=*/false, cpu_load_origin::worker);
            }
        }
    }

    // Stamp the marker only after every predicted expert is resident.
    if (decode_prefetch_force_cache_hit_ && full_warm) {
        layer_prefetch_done_token_[layer].store(token, std::memory_order_relaxed);
    }
}

// Run prefetch on the worker without changing decode state.
void llama_expert_cache::worker_loop() {
    prefetch_init();

    const uint32_t i = (uint32_t) decode_prefetch_lookahead_;

    uint32_t f               = i;
    uint64_t last_token      = UINT64_MAX;
    bool     have_last_token = false;

    std::vector<float>   hidden;
    std::vector<int32_t> predicted;
    uint64_t             last_seen_wake_seq = 0;

    while (!worker_shutdown_.load(std::memory_order_relaxed)) {
        // Wait for a new snapshot publish (or shutdown).
        {
            std::unique_lock<std::mutex> lk(worker_mu_);
            worker_cv_.wait(lk, [&] {
                return worker_shutdown_.load(std::memory_order_relaxed) || worker_wake_seq_ != last_seen_wake_seq;
            });
            last_seen_wake_seq = worker_wake_seq_;
        }
        if (worker_shutdown_.load(std::memory_order_relaxed)) {
            break;
        }

        uint32_t cur_layer;
        uint64_t token;
        if (!read_snapshot(cur_layer, token, hidden)) {
            break;  // shutdown mid-read
        }

        if (!have_last_token || token != last_token) {
            f               = cur_layer + i;
            last_token      = token;
            have_last_token = true;
        }

        while (f < n_layer_) {
            if (worker_shutdown_.load(std::memory_order_relaxed)) {
                break;
            }

            if (f >= router_descs_.size() || router_descs_[f].gate_inp == nullptr) {
                f++;
                continue;  // dense layer
            }
            if (!router_descs_[f].prefetch_capable) {
                // Skip routers the worker cannot reproduce.
                f++;
                continue;
            }
            if (f <= cur_layer) {
                // Skip the active layer and stop when no lookahead layer exists.
                if (cur_layer + i < n_layer_) {
                    f = cur_layer + i;
                } else if (i / 2 > 0 && cur_layer + i / 2 < n_layer_) {
                    f = cur_layer + i / 2;
                } else {
                    break;
                }
                continue;
            }

            worker_predict(f, hidden.data(), (int64_t) hidden.size(), predicted);
            warm_cache(f, predicted, token);
            f++;

            // Adopt a newer snapshot without resetting the frontier.
            uint32_t peek_layer;
            uint64_t peek_token;
            if (!read_snapshot(peek_layer, peek_token, hidden)) {
                f = n_layer_;
                break;
            }  // shutdown
            if (peek_token == token) {
                cur_layer = peek_layer;
            } else {
                // Stop this frontier when a new token appears.
                break;
            }
        }
    }

    prefetch_teardown();
}

// Load an HTP batch on the worker path with private staging.
uint64_t llama_expert_cache::load_experts_batch_htp_prefetch(const load_req * reqs, uint32_t n) {
    if (n == 0) {
        return 0;
    }

    std::vector<ec_read_op> ops;
    ops.reserve((size_t) n * N_KINDS);

    std::vector<uint32_t> op_req_idx;
    std::vector<int>      op_kind;
    std::vector<size_t>   op_expected;
    std::vector<size_t>   op_row_start;
    std::vector<size_t>   op_row_count;
    op_req_idx.reserve((size_t) n * N_KINDS);
    op_kind.reserve((size_t) n * N_KINDS);
    op_expected.reserve((size_t) n * N_KINDS);
    op_row_start.reserve((size_t) n * N_KINDS);
    op_row_count.reserve((size_t) n * N_KINDS);

    const uint64_t flash_tok   = flash_begin();
    uint64_t       total_bytes = 0;
    const bool     chunked     = htp_read_chunk_kb_ > 0 && htp_repack_range_ != nullptr && n < (uint32_t) batch_depth_;

    for (uint32_t i = 0; i < n; ++i) {
        for (int k = 0; k < N_KINDS; ++k) {
            auto loc = get_loc(reqs[i].layer, (tensor_kind) k, reqs[i].expert);
            if (loc.size == 0) {
                continue;
            }
            size_t expected_ksz =
                layered_mode_ ? layered_kind_size_[(size_t) reqs[i].layer * N_KINDS + k] : kind_size_[k];
            if (shared_mode_) {
                GGML_ASSERT(loc.size <= expected_ksz &&
                            "expert cache: kind size exceeds max slot budget (shared mode)");
            } else {
                GGML_ASSERT(loc.size == expected_ksz && "expert cache: kind size mismatch between offset map and pool");
            }
            ggml_tensor * pt = pool_tensor(reqs[i].layer, (tensor_kind) k);
            GGML_ASSERT(pt != nullptr && "expert cache: worker HTP pool tensor missing for kind");
            const size_t row_size   = ggml_row_size(pt->type, pt->ne[0]);
            const size_t n_rows     = (size_t) pt->ne[1];
            const size_t chunk_rows = chunked ? htp_chunk_rows_for(pt, htp_read_chunk_kb_) : n_rows;
            for (size_t row_start = 0; row_start < n_rows;) {
                const size_t row_count  = chunked ? std::min(chunk_rows, n_rows - row_start) : n_rows;
                const size_t chunk_size = row_count * row_size;
                ops.push_back({ kind_slot_ptr((tensor_kind) k, (uint32_t) reqs[i].slot), chunk_size,
                                loc.offset + row_start * row_size, files_.at(loc.file_idx).get() });
                op_req_idx.push_back(i);
                op_kind.push_back(k);
                op_expected.push_back(expected_ksz);
                op_row_start.push_back(row_start);
                op_row_count.push_back(row_count);
                total_bytes += chunk_size;
                if (!chunked) {
                    break;
                }
                row_start += row_count;
            }
        }
    }

    if (!ops.empty()) {
        // Align each read so REPACK can consume it directly.
        size_t aligned_total = 0;
        for (auto & op : ops) {
            aligned_total += (op.size + 31) & ~(size_t) 31;
        }
        GGML_ASSERT(aligned_total <= htp_read_buf_prefetch_.size() &&
                    "expert cache: htp_read_buf_prefetch_ too small for batch");

        std::vector<void *> original_dsts;
        original_dsts.reserve(ops.size());

        size_t offset_in_temp = 0;
        for (auto & op : ops) {
            original_dsts.push_back(op.dst);
            op.dst = htp_read_buf_prefetch_.data() + offset_in_temp;
            offset_in_temp += (op.size + 31) & ~(size_t) 31;
        }

        uint64_t             repack_ns = 0;
        ec_read_batch_result result    = read_batch_stream(
            ensure_worker_thread_pool(), 1, ops.data(), (uint32_t) ops.size(), odirect_, stats_,
            [&](uint32_t i) {
                const uint64_t repack_start = stats_ ? stats_now_ns() : 0;
                uint32_t       ri           = op_req_idx[i];
                int            k            = op_kind[i];
                ggml_tensor *  pt           = pool_tensor(reqs[ri].layer, (tensor_kind) k);
                GGML_ASSERT(pt != nullptr && "expert cache: worker HTP pool tensor missing for kind");
                ggml_tensor slot_view = *pt;
                slot_view.data        = original_dsts[i];
                slot_view.ne[2]       = 1;
                slot_view.nb[2]       = op_expected[i];
                if (chunked) {
                    htp_repack_range_(&slot_view, ops[i].dst, op_row_start[i], op_row_count[i]);
                } else {
                    std::lock_guard<std::mutex> lk(htp_repack_mu_);
                    ggml_backend_tensor_set(&slot_view, ops[i].dst, 0, ops[i].size);
                }
                if (stats_) {
                    repack_ns += stats_now_ns() - repack_start;
                }
            },
            [&](uint64_t end_ns) { flash_end(flash_tok, total_bytes, /*from_worker=*/true, end_ns); });
        GGML_ASSERT(result.ok && "expert cache: worker HTP batch pread failed");
        if (stats_) {
            stat_htp_worker_read_ns.fetch_add(result.read_ns, std::memory_order_relaxed);
            stat_htp_worker_copy_ns.fetch_add(result.copy_ns, std::memory_order_relaxed);
            stat_htp_worker_repack_ns.fetch_add(repack_ns, std::memory_order_relaxed);
        }
    }

    for (uint32_t i = 0; i < n; ++i) {
        stats_record_prefetch_load(reqs[i].slot, reqs[i].layer, reqs[i].expert);
        mark_occupied(reqs[i].slot);
    }

    return total_bytes;
}

llama_expert_cache::expert_slice_loc llama_expert_cache::get_loc(uint32_t    layer,
                                                                 tensor_kind kind,
                                                                 uint32_t    expert) const {
    size_t idx = (size_t) layer * N_KINDS * n_expert_ + (size_t) kind * n_expert_ + expert;
    GGML_ASSERT(idx < offsets_.size() && "expert cache: get_loc index out of bounds");
    return offsets_[idx];
}

void llama_expert_cache::load_expert_sync(int32_t slot, uint32_t layer, uint32_t expert, bool is_prefill) {
    GGML_ASSERT(slot >= 0);
    GGML_ASSERT((llama_expert_slot_state) slots[slot].state.load(std::memory_order_acquire) ==
                llama_expert_slot_state::LOADING);
    GGML_ASSERT(!files_.empty() && "expert cache: file descriptors not set");

    // Batch all present tensor-kind reads for the expert.
    ec_read_op ops[N_KINDS];
    void *     htp_slot_dsts[N_KINDS];  // FASTRPC destinations, saved for REPACK
    int        kind_idx[N_KINDS];
    size_t     expected[N_KINDS];
    uint32_t   n_ops = 0;

    // Give each tensor kind an aligned region in staging.
    size_t htp_buf_offset = 0;

    for (int k = 0; k < N_KINDS; ++k) {
        auto loc = get_loc(layer, (tensor_kind) k, expert);
        if (loc.size == 0) {
            continue;
        }
        size_t expected_ksz = layered_mode_ ? layered_kind_size_[(size_t) layer * N_KINDS + k] : kind_size_[k];
        if (shared_mode_) {
            GGML_ASSERT(loc.size <= expected_ksz && "expert cache: kind size exceeds max slot budget (shared mode)");
        } else {
            GGML_ASSERT(loc.size == expected_ksz && "expert cache: kind size mismatch between offset map and pool");
        }

        void * slot_ptr = kind_slot_ptr((tensor_kind) k, (uint32_t) slot);
        void * read_dst;
        if (use_htp_pool_) {
            GGML_ASSERT(htp_buf_offset + loc.size <= htp_read_buf_.size() &&
                        "expert cache: htp_read_buf_ too small for sync load");
            read_dst = htp_read_buf_.data() + htp_buf_offset;
            htp_buf_offset += (loc.size + 31) & ~(size_t) 31;
            htp_slot_dsts[n_ops] = slot_ptr;
        } else {
            read_dst = slot_ptr;
        }

        ops[n_ops]      = { read_dst, loc.size, loc.offset, files_.at(loc.file_idx).get() };
        kind_idx[n_ops] = k;
        expected[n_ops] = expected_ksz;
        n_ops++;
    }

    if (n_ops == 0) {
        mark_occupied(slot);
        return;
    }

    const uint64_t t0_read   = is_prefill ? stats_now_ns() : 0;
    const uint64_t flash_tok = is_prefill ? 0 : flash_begin();

    const bool read_ok = read_batch_main(ops, n_ops);
    GGML_ASSERT(read_ok && "expert cache: pread failed");

    uint64_t total_bytes = 0;
    for (uint32_t i = 0; i < n_ops; ++i) {
        total_bytes += (uint64_t) ops[i].size;
    }

    // Repack staged weights into the HTP slot.
    if (use_htp_pool_) {
        for (uint32_t i = 0; i < n_ops; ++i) {
            ggml_tensor * pt = pool_tensor(layer, (tensor_kind) kind_idx[i]);
            GGML_ASSERT(pt != nullptr && "expert cache: HTP pool tensor missing for kind");
            ggml_tensor slot_view = *pt;
            slot_view.data        = htp_slot_dsts[i];
            slot_view.ne[2]       = 1;
            slot_view.nb[2]       = expected[i];
            ggml_backend_tensor_set(&slot_view, ops[i].dst, 0, ops[i].size);
        }
    }

    if (is_prefill) {
        const uint64_t read_ns = stats_now_ns() - t0_read;
        stats_record_flash_read(total_bytes, read_ns, /*is_prefill=*/true);
    } else {
        flash_end(flash_tok, total_bytes, /*from_worker=*/false);
    }
    mark_occupied(slot);
}

uint64_t llama_expert_cache::load_experts_batch_cpu(const load_req * reqs,
                                                    uint32_t         n,
                                                    bool             is_prefill,
                                                    cpu_load_origin  origin) {
    if (n == 0) {
        return 0;
    }
    GGML_ASSERT(!is_prefill || origin == cpu_load_origin::main);

    std::vector<ec_read_op> ops;
    ops.clear();
    ops.reserve((size_t) n * N_KINDS);

    const uint64_t t0          = is_prefill ? stats_now_ns() : 0;
    const uint64_t flash_tok   = is_prefill ? 0 : flash_begin();
    uint64_t       total_bytes = 0;

    for (uint32_t i = 0; i < n; ++i) {
        for (int k = 0; k < N_KINDS; ++k) {
            auto loc = get_loc(reqs[i].layer, (tensor_kind) k, reqs[i].expert);
            if (loc.size == 0) {
                continue;
            }
            ops.push_back({ kind_slot_ptr((tensor_kind) k, (uint32_t) reqs[i].slot), loc.size, loc.offset,
                            files_.at(loc.file_idx).get() });
            total_bytes += loc.size;
        }
    }

    if (!ops.empty()) {
        bool read_ok = origin == cpu_load_origin::main ? read_batch_main(ops.data(), (uint32_t) ops.size()) :
                                                         read_batch_worker(ops.data(), (uint32_t) ops.size());
        GGML_ASSERT(read_ok && "expert cache: batch pread failed");
    }

    if (is_prefill) {
        const uint64_t read_ns = stats_now_ns() - t0;
        stats_record_flash_read(total_bytes, read_ns, /*is_prefill=*/true);
    } else {
        flash_end(flash_tok, total_bytes, origin == cpu_load_origin::worker);
    }

    for (uint32_t i = 0; i < n; ++i) {
        if (origin == cpu_load_origin::worker) {
            stats_record_prefetch_load(reqs[i].slot, reqs[i].layer, reqs[i].expert);
        }
        mark_occupied(reqs[i].slot);
    }

    return total_bytes;
}

uint64_t llama_expert_cache::load_experts_batch_htp(const load_req * reqs, uint32_t n, bool is_prefill) {
    if (n == 0) {
        return 0;
    }
    // Read into staging before repacking into FASTRPC slots.

    thread_local std::vector<ec_read_op> ops;
    ops.clear();
    ops.reserve((size_t) n * N_KINDS);

    std::vector<uint32_t> op_req_idx;
    std::vector<int>      op_kind;
    std::vector<size_t>   op_expected;
    std::vector<size_t>   op_row_start;
    std::vector<size_t>   op_row_count;
    op_req_idx.reserve((size_t) n * N_KINDS);
    op_kind.reserve((size_t) n * N_KINDS);
    op_expected.reserve((size_t) n * N_KINDS);
    op_row_start.reserve((size_t) n * N_KINDS);
    op_row_count.reserve((size_t) n * N_KINDS);

    const uint64_t t0            = is_prefill ? stats_now_ns() : 0;
    const uint64_t flash_tok     = is_prefill ? 0 : flash_begin();
    uint64_t       total_bytes   = 0;
    uint64_t       flash_read_ns = 0;
    const bool     chunked =
        htp_read_chunk_kb_ > 0 && htp_repack_range_ != nullptr && !is_prefill && n < (uint32_t) batch_depth_;

    for (uint32_t i = 0; i < n; ++i) {
        for (int k = 0; k < N_KINDS; ++k) {
            auto loc = get_loc(reqs[i].layer, (tensor_kind) k, reqs[i].expert);
            if (loc.size == 0) {
                continue;
            }
            size_t expected_ksz =
                layered_mode_ ? layered_kind_size_[(size_t) reqs[i].layer * N_KINDS + k] : kind_size_[k];
            if (shared_mode_) {
                GGML_ASSERT(loc.size <= expected_ksz &&
                            "expert cache: kind size exceeds max slot budget (shared mode)");
            } else {
                GGML_ASSERT(loc.size == expected_ksz && "expert cache: kind size mismatch between offset map and pool");
            }
            ggml_tensor * pt = pool_tensor(reqs[i].layer, (tensor_kind) k);
            GGML_ASSERT(pt != nullptr && "expert cache: HTP pool tensor missing for kind");
            const size_t row_size   = ggml_row_size(pt->type, pt->ne[0]);
            const size_t n_rows     = (size_t) pt->ne[1];
            const size_t chunk_rows = chunked ? htp_chunk_rows_for(pt, htp_read_chunk_kb_) : n_rows;
            for (size_t row_start = 0; row_start < n_rows;) {
                const size_t row_count  = chunked ? std::min(chunk_rows, n_rows - row_start) : n_rows;
                const size_t chunk_size = row_count * row_size;
                ops.push_back({ kind_slot_ptr((tensor_kind) k, (uint32_t) reqs[i].slot), chunk_size,
                                loc.offset + row_start * row_size, files_.at(loc.file_idx).get() });
                op_req_idx.push_back(i);
                op_kind.push_back(k);
                op_expected.push_back(expected_ksz);
                op_row_start.push_back(row_start);
                op_row_count.push_back(row_count);
                total_bytes += chunk_size;
                if (!chunked) {
                    break;
                }
                row_start += row_count;
            }
        }
    }

    if (!ops.empty()) {
        // Align each read so REPACK can consume it directly.
        size_t aligned_total = 0;
        for (auto & op : ops) {
            aligned_total += (op.size + 31) & ~(size_t) 31;
        }
        GGML_ASSERT(aligned_total <= htp_read_buf_.size() && "expert cache: htp_read_buf_ too small for batch");

        std::vector<void *> original_dsts;
        original_dsts.reserve(ops.size());

        size_t offset_in_temp = 0;
        for (auto & op : ops) {
            original_dsts.push_back(op.dst);
            op.dst = htp_read_buf_.data() + offset_in_temp;
            offset_in_temp += (op.size + 31) & ~(size_t) 31;
        }

        uint64_t             repack_ns = 0;
        ec_read_batch_result result    = read_batch_stream(
            ensure_thread_pool(), 0, ops.data(), (uint32_t) ops.size(), odirect_, stats_,
            [&](uint32_t i) {
                const uint64_t repack_start = stats_ ? stats_now_ns() : 0;
                uint32_t       ri           = op_req_idx[i];
                int            k            = op_kind[i];
                ggml_tensor *  pt           = pool_tensor(reqs[ri].layer, (tensor_kind) k);
                GGML_ASSERT(pt != nullptr && "HTP pool tensor missing for kind");
                ggml_tensor slot_view = *pt;
                slot_view.data        = original_dsts[i];
                slot_view.ne[2]       = 1;
                slot_view.nb[2]       = op_expected[i];
                if (chunked) {
                    htp_repack_range_(&slot_view, ops[i].dst, op_row_start[i], op_row_count[i]);
                } else {
                    std::lock_guard<std::mutex> lk(htp_repack_mu_);
                    ggml_backend_tensor_set(&slot_view, ops[i].dst, 0, ops[i].size);
                }
                if (stats_) {
                    repack_ns += stats_now_ns() - repack_start;
                }
            },
            [&](uint64_t end_ns) {
                if (!is_prefill) {
                    flash_end(flash_tok, total_bytes, /*from_worker=*/false, end_ns);
                }
            });
        GGML_ASSERT(result.ok && "expert cache: HTP batch pread failed");
        flash_read_ns = result.read_ns;
        if (stats_) {
            stat_htp_main_read_ns.fetch_add(result.read_ns, std::memory_order_relaxed);
            stat_htp_main_copy_ns.fetch_add(result.copy_ns, std::memory_order_relaxed);
            stat_htp_main_repack_ns.fetch_add(repack_ns, std::memory_order_relaxed);
        }
    }

    if (is_prefill) {
        stats_record_flash_read(total_bytes, flash_read_ns != 0 ? flash_read_ns : stats_now_ns() - t0,
                                /*is_prefill=*/true);
    }

    for (uint32_t i = 0; i < n; ++i) {
        mark_occupied(reqs[i].slot);
    }

    return total_bytes;
}

// Size thread pools from present kinds so fused GATE_UP does not waste threads.
int llama_expert_cache::n_kinds_present() const {
    int n = 0;
    for (int k = 0; k < N_KINDS; ++k) {
        if (kind_size_[k] != 0) {
            ++n;
        }
    }
    return n > 0 ? n : N_KINDS;  // before layout init, use the upper bound
}

ec_thread_pool * llama_expert_cache::ensure_thread_pool() {
    if (thread_pool_disabled_) {
        return nullptr;
    }
    if (!thread_pool_) {
        const int n_threads = batch_depth_ * n_kinds_present();
        try {
            thread_pool_ = new ec_thread_pool((size_t) n_threads);
        } catch (const std::exception &) {
            thread_pool_disabled_ = true;
        }
        if (stats_ && !thread_pool_logged_) {
            thread_pool_logged_ = true;
            if (thread_pool_) {
                LLAMA_LOG_WARN("%s: expert cache: read thread pool active (%d threads)\n", __func__, n_threads);
            } else {
                LLAMA_LOG_WARN("%s: expert cache: read thread pool unavailable, using pread\n", __func__);
            }
        }
    }
    return thread_pool_;
}

// Use a separate pool to avoid queue contention with inference reads.
ec_thread_pool * llama_expert_cache::ensure_worker_thread_pool() {
    if (worker_thread_pool_disabled_) {
        return nullptr;
    }
    if (!worker_thread_pool_) {
        const int n_threads = batch_depth_ * n_kinds_present();
        try {
            worker_thread_pool_ = new ec_thread_pool((size_t) n_threads, /*pool_slot=*/1);
        } catch (const std::exception &) {
            worker_thread_pool_disabled_ = true;
        }
    }
    return worker_thread_pool_;
}

static bool read_batch_sync(const ec_read_op * ops, uint32_t n, ec_odirect * od, size_t pool_slot) {
    for (uint32_t i = 0; i < n; ++i) {
        if (!ops[i].file->read(od, pool_slot, i, ops[i].dst, ops[i].size, ops[i].offset)) {
            return false;
        }
    }
    return true;
}

bool llama_expert_cache::read_batch_main(const ec_read_op * ops, uint32_t n) {
    if (ec_thread_pool * pool = ensure_thread_pool()) {
        return pool->read_batch(ops, n, odirect_);
    }
    return read_batch_sync(ops, n, odirect_, 0);
}

bool llama_expert_cache::read_batch_worker(const ec_read_op * ops, uint32_t n) {
    if (ec_thread_pool * pool = ensure_worker_thread_pool()) {
        return pool->read_batch(ops, n, odirect_);
    }
    return read_batch_sync(ops, n, odirect_, 1);
}

// Opens buffered and direct-I/O descriptors when requested.
bool llama_expert_cache::add_file(int fd, bool use_direct_io) {
    auto file = std::make_unique<ec_file>();
#if defined(_WIN32)
    struct _stat64 st {};

    if (_fstat64(fd, &st) != 0 || st.st_size < 0 || (uint64_t) st.st_size > SIZE_MAX) {
        return false;
    }
    file->fd = _dup(fd);
#else
    struct stat st {};

    if (fstat(fd, &st) != 0 || st.st_size < 0 || (uint64_t) st.st_size > SIZE_MAX) {
        return false;
    }

#    if defined(__linux__)
    const int flags = fcntl(fd, F_GETFL);
    if (flags < 0) {
        return false;
    }
    char path[64];
    snprintf(path, sizeof(path), "/proc/self/fd/%d", fd);
    file->fd = (flags & O_DIRECT) ? open(path, O_RDONLY) : dup(fd);
#    else
    file->fd = dup(fd);
#    endif
#endif
    file->size = (size_t) st.st_size;
    if (file->fd < 0) {
        LLAMA_LOG_ERROR("%s: could not open buffered GGUF source: %s\n", __func__, strerror(errno));
        return false;
    }

    if (use_direct_io) {
#if defined(__linux__)
        const size_t blk = st.st_blksize > 0 ? (size_t) st.st_blksize : 4096;
        if ((blk & (blk - 1)) == 0) {
            file->direct_fd  = (flags & O_DIRECT) ? dup(fd) : open(path, O_RDONLY | O_DIRECT);
            file->block_size = blk;
            if (file->direct_fd >= 0) {
                if (!odirect_) {
                    odirect_ = new ec_odirect();
                }
                LLAMA_LOG_INFO("%s: expert cache: O_DIRECT reads enabled (block size %zu)\n", __func__, blk);
            }
        }
#endif
        if (file->direct_fd < 0) {
            LLAMA_LOG_WARN("%s: expert cache: O_DIRECT unavailable, using buffered reads\n", __func__);
        }
    }
    files_.push_back(std::move(file));
    return true;
}

int64_t llama_expert_cache::odirect_pread_for_test(size_t  pool_slot,
                                                   size_t  buf_idx,
                                                   void *  dst,
                                                   size_t  size,
                                                   int64_t offset) {
    if (odirect_ == nullptr || files_.empty() || files_[0]->direct_fd < 0 || offset < 0) {
        return -1;
    }
    return odirect_->pread_aligned(files_[0]->direct_fd, files_[0]->block_size, pool_slot, buf_idx, dst, size,
                                   (size_t) offset);
}

llama_expert_cache::~llama_expert_cache() {
    // Join the worker before freeing its state.
    stop_worker();
    delete thread_pool_;
    thread_pool_ = nullptr;
    delete worker_thread_pool_;
    worker_thread_pool_ = nullptr;
    // Read jobs must finish before their bounce buffers are freed.
    delete odirect_;
    odirect_ = nullptr;
    files_.clear();
    if (pool_buf_) {
        ggml_backend_buffer_free(pool_buf_);
    }
    if (pool_ggml_ctx_) {
        ggml_free(pool_ggml_ctx_);
    }
    if (layered_pool_ctx_) {
        ggml_free(layered_pool_ctx_);
    }
    prefetch_teardown();
}

void llama_expert_cache::prefetch_init() {
    if (worker_backend_) {
        return;  // already initialized
    }
    worker_backend_ = ggml_backend_cpu_init();
    GGML_ASSERT(worker_backend_ != nullptr);
    // Use one router thread to avoid competing with decode for CPU cores.
    ggml_backend_cpu_set_n_threads(worker_backend_, 1);
}

void llama_expert_cache::prefetch_teardown() {
    for (auto & g : worker_pred_graphs_) {
        if (g.buf) {
            ggml_backend_buffer_free(g.buf);
        }
        if (g.ctx) {
            ggml_free(g.ctx);
        }
    }
    worker_pred_graphs_.clear();

    if (worker_ctx_) {
        ggml_free(worker_ctx_);
        worker_ctx_ = nullptr;
    }
    if (worker_backend_) {
        ggml_backend_free(worker_backend_);
        worker_backend_ = nullptr;
    }
}

llama_expert_cache::worker_pred_graph llama_expert_cache::build_worker_pred_graph(uint32_t layer,
                                                                                  int64_t  n_embd) const {
    worker_pred_graph       g;
    const moe_router_desc & rd = router_descs_[layer];

    const size_t            n_nodes_est = 24;  // router nodes + up to 3 pre-gating nodes (gemma4) + margin
    struct ggml_init_params params      = {
        /* .mem_size   = */ ggml_tensor_overhead() * n_nodes_est + ggml_graph_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    GGML_ASSERT(ctx != nullptr);
    g.ctx = ctx.get();

    g.inp = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, n_embd, 1);
    ggml_set_input(g.inp);

    ggml_tensor * selection_probs = nullptr;
    ggml_tensor * probs           = nullptr;
    g.out = llama_expert_cache_build_moe_router_standalone(g.ctx, g.inp, rd, (int) layer, &selection_probs, &probs);
    ggml_set_output(g.out);

    g.gf = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(g.gf, g.out);

    ctx.release();
    return g;
}

llama_expert_cache::worker_pred_graph & llama_expert_cache::worker_pred_graph_for(uint32_t layer, int64_t n_embd) {
    if (worker_pred_graphs_.empty()) {
        worker_pred_graphs_.resize(n_layer_);
    }
    worker_pred_graph & g = worker_pred_graphs_[layer];
    if (g.gf != nullptr) {
        GGML_ASSERT(g.inp->ne[0] == n_embd && "worker_predict: n_embd changed for a cached layer");
        return g;
    }
    g     = build_worker_pred_graph(layer, n_embd);
    g.buf = ggml_backend_alloc_ctx_tensors(g.ctx, worker_backend_);
    GGML_ASSERT(g.buf != nullptr);
    return g;
}

void llama_expert_cache::prepare_runtime_memory(int64_t n_embd) {
    auto add = [](size_t & total, size_t count, size_t size) {
        if (count != 0 && size > (SIZE_MAX - total) / count) {
            throw std::overflow_error("expert cache runtime memory exceeds size_t");
        }
        total += count * size;
    };
    direct_io_memory_ = 0;
    predictor_memory_ = 0;
    {
        std::vector<size_t> capacities;
        auto                measure_batch = [&](uint32_t layer, uint32_t count, bool chunked) {
            size_t index = 0;
            for (uint32_t i = 0; i < count; ++i) {
                for (int k = 0; k < N_KINDS; ++k) {
                    const auto first = get_loc(layer, (tensor_kind) k, 0);
                    if (first.size == 0) {
                        continue;
                    }
                    const auto * tensor = pool_tensor(layer, (tensor_kind) k);
                    const size_t row_size = use_htp_pool_ ? ggml_row_size(tensor->type, tensor->ne[0]) : first.size;
                    const size_t rows = use_htp_pool_ ? (size_t) tensor->ne[1] : 1;
                    const size_t chunk_rows = chunked ? htp_chunk_rows_for(tensor, htp_read_chunk_kb_) : rows;
                    for (size_t row = 0; row < rows;) {
                        const size_t length  = std::min(chunk_rows, rows - row) * row_size;
                        size_t       maximum = 0;
                        for (uint32_t e = 0; e < n_expert_; ++e) {
                            const auto   loc  = get_loc(layer, (tensor_kind) k, e);
                            const auto & file = *files_.at(loc.file_idx);
                            if (file.direct_fd < 0) {
                                continue;
                            }
                            const size_t block = file.block_size;
                            const size_t lead  = (loc.offset + row * row_size) & (block - 1);
                            if (length > SIZE_MAX - lead - (block - 1)) {
                                throw std::overflow_error("expert cache direct I/O size exceeds size_t");
                            }
                            maximum = std::max(maximum, (lead + length + block - 1) & ~(block - 1));
                        }
                        if (index >= capacities.size()) {
                            capacities.resize(index + 1);
                        }
                        capacities[index] = std::max(capacities[index], maximum);
                        ++index;
                        row += std::min(chunk_rows, rows - row);
                    }
                }
            }
        };
        const uint32_t depth = (uint32_t) batch_depth_;
        for (uint32_t l = 0; l < n_layer_; ++l) {
            measure_batch(l, std::min(depth, n_expert_), false);
            if (use_htp_pool_ && htp_read_chunk_kb_ > 0 && htp_repack_range_ && depth > 1) {
                measure_batch(l, std::min(depth - 1, n_expert_), true);
            }
        }
        const size_t pools = layered_mode_ && decode_prefetch_lookahead_ != 0 ? 2 : 1;
        read_batch_memory_ = 0;
        add(read_batch_memory_, pools, ec_thread_pool::batch_memory(capacities.size(), stats_));
        // HTP tracks repacking destinations and row ranges in addition to read descriptors.
        if (use_htp_pool_) {
            add(read_batch_memory_, pools * capacities.size(),
                sizeof(ec_read_op) + sizeof(uint32_t) + sizeof(int) + 3 * sizeof(size_t) + sizeof(void *));
        }
        if (odirect_) {
            for (size_t size : capacities) {
                add(direct_io_memory_, pools, size);
            }
            add(direct_io_memory_, pools * capacities.size(), sizeof(ec_odirect::buf));
            // Reserve descriptors now so later vector growth does not exceed the estimate.
            for (size_t s = 0; s < pools; ++s) {
                odirect_->bufs[s].reserve(capacities.size());
            }
        }
    }
    if (decode_prefetch_lookahead_ != 0) {
        size_t workspace = 0;
        for (uint32_t l = 0; l < router_descs_.size(); ++l) {
            if (!router_descs_[l].gate_inp || !router_descs_[l].prefetch_capable) {
                continue;
            }
            auto             g = build_worker_pred_graph(l, n_embd);
            ggml_context_ptr ctx(g.ctx);
            const size_t data = ggml_backend_alloc_ctx_tensors_from_buft_size(g.ctx, ggml_backend_cpu_buffer_type());
            if (data == 0) {
                throw std::runtime_error("expert cache could not measure predictor tensors");
            }
            add(predictor_memory_, 1, ggml_get_mem_size(g.ctx));
            add(predictor_memory_, 1, data);
            workspace = std::max(workspace, ggml_graph_plan(g.gf, 1, nullptr).work_size);
        }
        add(predictor_memory_, 1, workspace);
        add(predictor_memory_, n_layer_, sizeof(worker_pred_graph));
        add(predictor_memory_, (size_t) n_embd, sizeof(float));
        add(predictor_memory_, n_expert_used_, sizeof(int32_t) + sizeof(load_req));
    }
}

void llama_expert_cache::worker_predict(uint32_t               layer,
                                        const float *          hidden,
                                        int64_t                n_embd,
                                        std::vector<int32_t> & result) {
    GGML_ASSERT(worker_backend_ != nullptr && "prefetch_init() must be called before worker_predict()");
    GGML_ASSERT(layer < router_descs_.size());

    const moe_router_desc & rd = router_descs_[layer];
    GGML_ASSERT(rd.gate_inp != nullptr && "worker_predict called on a dense (non-MoE) layer");
    GGML_ASSERT(rd.prefetch_capable &&
                "worker_predict called on a MoE layer not marked "
                "prefetch_capable (no arch sets this today; guards a future unrepresented "
                "custom-gating arch)");

    worker_pred_graph & g = worker_pred_graph_for(layer, n_embd);

    const int64_t n_expert_used = g.out->ne[0];
    result.resize(n_expert_used);
    const uint64_t start = stats_ ? stats_now_ns() : 0;
    ggml_backend_tensor_set(g.inp, hidden, 0, n_embd * sizeof(float));
    ggml_backend_graph_compute(worker_backend_, g.gf);
    ggml_backend_tensor_get(g.out, result.data(), 0, n_expert_used * sizeof(int32_t));
    if (stats_) {
        stat_predict_ns.fetch_add(stats_now_ns() - start, std::memory_order_relaxed);
        stat_predict_calls.fetch_add(1, std::memory_order_relaxed);
    }
}

uint8_t * llama_expert_cache::ensure_pool_buf() {
    if (!pool_buf_) {
        // HTP allocates pool_buf_ during initialization.
        pool_buf_ = ggml_backend_cpu_buffer_from_ptr(ddr_pool.data(), ddr_pool.size());
        GGML_ASSERT(pool_buf_ != nullptr);
    }
    ggml_backend_buffer_set_usage(pool_buf_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    return use_htp_pool_ ? static_cast<uint8_t *>(ggml_backend_buffer_get_base(pool_buf_)) : ddr_pool.data();
}

void llama_expert_cache::alloc_pool_tensor(ggml_tensor * tensor, size_t offset) {
    if (no_alloc_) {
        tensor->buffer = pool_buf_;
        tensor->data   = nullptr;
    } else {
        uint8_t * base = ensure_pool_buf();
        ggml_backend_tensor_alloc(pool_buf_, tensor, base + offset);
    }
    init_pool_tensor(tensor);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_expert_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> result;
    const auto                                   cpu_buft  = ggml_backend_cpu_buffer_type();
    const auto                                   pool_buft = use_htp_pool_ ? htp_repack_buft_ : cpu_buft;
    result[pool_buft] = no_alloc_ ? (size_t) n_slots_ * expert_sz : ggml_backend_buffer_get_size(pool_buf_);
    size_t staging    = 0;
    for (size_t size : kind_size_) {
        staging += (size + 31) & ~(size_t) 31;
    }
    if (use_htp_pool_) {
        result[cpu_buft] += staging * batch_depth_ * (decode_prefetch_lookahead_ != 0 ? 2 : 1);
    }
    result[cpu_buft] += (size_t) n_slots_ * sizeof(llama_expert_slot) + offsets_.capacity() * sizeof(expert_slice_loc);
    result[cpu_buft] += direct_io_memory_ + predictor_memory_ + read_batch_memory_;
    result[cpu_buft] +=
        router_descs_.capacity() * sizeof(moe_router_desc) + prefetch_snapshot_.hidden.capacity() * sizeof(float);
    result[cpu_buft] += (size_t) n_layer_ * sizeof(std::atomic<uint64_t>);
    for (const auto * ctx : { pool_ggml_ctx_, layered_pool_ctx_ }) {
        if (ctx) {
            result[cpu_buft] += ggml_get_mem_size(ctx);
        }
    }
    result[cpu_buft] += layered_kind_size_.capacity() * sizeof(size_t) + layered_kind_base_.capacity() * sizeof(size_t);
    result[cpu_buft] += layered_pool_tensors_.capacity() * sizeof(ggml_tensor *) +
                        layered_prefill_tensors_.capacity() * sizeof(ggml_tensor *);
    result[cpu_buft] += layer_slot_base_.capacity() * sizeof(uint32_t);
    return result;
}

void llama_expert_cache::create_pool_tensors(const std::array<std::pair<int64_t, int64_t>, N_KINDS> & kind_ne,
                                             const std::array<ggml_type, N_KINDS> &                   kind_dtype) {
    // Single-group path (CPU or single-group HTP).
    struct ggml_init_params ctx_params = {
        /* .mem_size   = */ (size_t) N_KINDS * ggml_tensor_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    pool_ggml_ctx_ = ggml_init(ctx_params);
    GGML_ASSERT(pool_ggml_ctx_ != nullptr);

    for (int k = 0; k < N_KINDS; ++k) {
        if (kind_size_[k] == 0) {
            continue;  // kind absent
        }

        auto [ne0, ne1]  = kind_ne[k];
        pool_tensors_[k] = ggml_new_tensor_3d(pool_ggml_ctx_, kind_dtype[k], ne0, ne1, (int64_t) n_slots_);
        GGML_ASSERT(pool_tensors_[k] != nullptr);

        alloc_pool_tensor(pool_tensors_[k], kind_region_offset_[k]);

        GGML_ASSERT((size_t) pool_tensors_[k]->nb[2] == kind_size_[k] &&
                    "pool tensor nb[2] must equal kind_size for correct ggml_mul_mat_id indexing");
    }
}

void llama_expert_cache::create_pool_tensors_layered(const std::vector<std::pair<int64_t, int64_t>> & kind_ne_flat,
                                                     const std::vector<ggml_type> &                   kind_dtype_flat,
                                                     uint32_t                                         prefill_floor) {
    GGML_ASSERT(kind_ne_flat.size() == (size_t) n_total_layers_ * N_KINDS);
    GGML_ASSERT(kind_dtype_flat.size() == (size_t) n_total_layers_ * N_KINDS);
    if (prefill_floor == 0) {
        prefill_floor = 1;
    }

    if (use_htp_pool_) {
        // Each HTP group must fit within GGML_HEXAGON_MBUF.
        const size_t mbuf = htp_max_pool_size_;

        // Count active (layer, kind) pairs for tensor metadata.
        size_t n_meta_layers = 0;
        for (uint32_t l = 0; l < n_total_layers_; ++l) {
            if (layer_slot_base_[l + 1] == layer_slot_base_[l]) {
                continue;
            }
            for (int k = 0; k < N_KINDS; ++k) {
                if (kind_size_[k] > 0 && kind_ne_flat[l * N_KINDS + k].first > 0) {
                    ++n_meta_layers;
                }
            }
        }

        const size_t slot_bytes = [&] {
            size_t s = 0;
            for (int k = 0; k < N_KINDS; ++k) {
                s += kind_size_[k];
            }
            return s;
        }();

        // Keep the group boundaries chosen during model setup.
        using group_def                       = htp_group_plan::group_def;
        const std::vector<group_def> & groups = planned_htp_groups_;
        GGML_ASSERT(!groups.empty() && "expert cache: missing planned HTP groups");
        for (uint32_t l = 0; l < n_total_layers_; ++l) {
            const uint32_t n_sl = layer_slot_base_[l + 1] - layer_slot_base_[l];
            if (n_sl == 0) {
                continue;
            }
            GGML_ASSERT((size_t) n_sl * slot_bytes <= mbuf && "single layer exceeds opt_mbuf");
        }
        for (const group_def & gd : groups) {
            GGML_ASSERT(gd.first_l <= gd.last_l);
            GGML_ASSERT(gd.last_l < n_total_layers_);
            GGML_ASSERT(gd.slot_base == layer_slot_base_[gd.first_l]);
            GGML_ASSERT(gd.n_slots == layer_slot_base_[gd.last_l + 1] - layer_slot_base_[gd.first_l]);
            GGML_ASSERT((size_t) gd.n_slots * slot_bytes <= mbuf);
            GGML_ASSERT(gd.n_slots >= prefill_floor);
        }

        // Each (layer, kind) pair needs separate decode and prefill tensors.
        size_t                  n_meta = 2 * n_meta_layers;
        struct ggml_init_params mp     = {
            /* .mem_size   = */ n_meta * ggml_tensor_overhead(),
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ true,
        };
        layered_pool_ctx_ = ggml_init(mp);
        GGML_ASSERT(layered_pool_ctx_ != nullptr);

        group_buffers_.clear();
        group_base_ptr_.assign(groups.size(), nullptr);
        group_slot_base_.assign(groups.size(), 0);
        group_n_slots_.assign(groups.size(), 0);
        layer_group_base_.assign(n_total_layers_, 0);
        layer_group_count_.assign(n_total_layers_, 1);

        for (uint32_t g = 0; g < (uint32_t) groups.size(); ++g) {
            const group_def &     gd     = groups[g];
            size_t                gbytes = (size_t) gd.n_slots * slot_bytes;
            ggml_backend_buffer_t gbuf   = ggml_backend_buft_alloc_buffer(htp_repack_buft_, no_alloc_ ? 0 : gbytes);
            GGML_ASSERT(gbuf != nullptr && "expert cache: HTP group buffer alloc failed");
            ggml_backend_buffer_set_usage(gbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            group_buffers_.push_back(gbuf);

            uint8_t * gbase     = static_cast<uint8_t *>(ggml_backend_buffer_get_base(gbuf));
            group_base_ptr_[g]  = gbase;
            group_slot_base_[g] = gd.slot_base;
            group_n_slots_[g]   = gd.n_slots;

            // Group offsets must match kind_slot_ptr.
            for (uint32_t l = gd.first_l; l <= gd.last_l; ++l) {
                uint32_t n_sl = layer_slot_base_[l + 1] - layer_slot_base_[l];
                if (n_sl == 0) {
                    continue;
                }
                layer_group_base_[l] = g;
                uint32_t local_slot0 = layer_slot_base_[l] - gd.slot_base;
                for (int k = 0; k < N_KINDS; ++k) {
                    layered_pool_tensors_[l * N_KINDS + k] = nullptr;
                    if (kind_size_[k] == 0) {
                        continue;
                    }
                    auto [ne0, ne1] = kind_ne_flat[l * N_KINDS + k];
                    if (ne0 == 0) {
                        continue;
                    }
                    size_t grp_kind_off = 0;
                    for (int j = 0; j < k; ++j) {
                        grp_kind_off += (size_t) gd.n_slots * kind_size_[j];
                    }
                    ggml_tensor * pt = ggml_new_tensor_3d(layered_pool_ctx_, kind_dtype_flat[l * N_KINDS + k], ne0, ne1,
                                                          (int64_t) n_sl);
                    pt->nb[2]        = kind_size_[k];
                    pt->nb[3]        = (int64_t) n_sl * kind_size_[k];
                    pt->data   = no_alloc_ ? nullptr : gbase + grp_kind_off + (size_t) local_slot0 * kind_size_[k];
                    pt->buffer = gbuf;
                    GGML_ASSERT(ggml_backend_buffer_init_tensor(gbuf, pt) == GGML_STATUS_SUCCESS);
                    init_pool_tensor(pt);
                    layered_pool_tensors_[l * N_KINDS + k] = pt;
                }
            }

            // Prefill tensors span the group but keep each layer's dtype and shape.
            for (uint32_t l = gd.first_l; l <= gd.last_l; ++l) {
                uint32_t n_sl = layer_slot_base_[l + 1] - layer_slot_base_[l];
                if (n_sl == 0) {
                    continue;
                }
                for (int k = 0; k < N_KINDS; ++k) {
                    layered_prefill_tensors_[l * N_KINDS + k] = nullptr;
                    if (kind_size_[k] == 0) {
                        continue;
                    }
                    auto [ne0, ne1] = kind_ne_flat[l * N_KINDS + k];
                    if (ne0 == 0) {
                        continue;
                    }
                    size_t grp_kind_off = 0;
                    for (int j = 0; j < k; ++j) {
                        grp_kind_off += (size_t) gd.n_slots * kind_size_[j];
                    }
                    ggml_tensor * gt = ggml_new_tensor_3d(layered_pool_ctx_, kind_dtype_flat[l * N_KINDS + k], ne0, ne1,
                                                          (int64_t) gd.n_slots);
                    gt->nb[2]        = kind_size_[k];
                    gt->nb[3]        = (int64_t) gd.n_slots * kind_size_[k];
                    gt->data         = no_alloc_ ? nullptr : gbase + grp_kind_off;
                    gt->buffer       = gbuf;
                    GGML_ASSERT(ggml_backend_buffer_init_tensor(gbuf, gt) == GGML_STATUS_SUCCESS);
                    init_pool_tensor(gt);
                    layered_prefill_tensors_[l * N_KINDS + k] = gt;
                }
            }
        }

        for (uint32_t l = 0; l < n_total_layers_; ++l) {
            const uint32_t n_sl = layer_slot_base_[l + 1] - layer_slot_base_[l];
            if (n_sl == 0) {
                continue;
            }
            const uint32_t g = layer_group_base_[l];
            GGML_ASSERT(layer_group_count_[l] == 1);
            GGML_ASSERT(g < group_slot_base_.size());
            GGML_ASSERT(layer_slot_base_[l] >= group_slot_base_[g]);
            GGML_ASSERT(layer_slot_base_[l + 1] <= group_slot_base_[g] + group_n_slots_[g]);
        }

        if (pool_buf_) {
            ggml_backend_buffer_free(pool_buf_);
            pool_buf_ = nullptr;
        }
        if (group_buffers_.size() == 1) {
            pool_buf_ = group_buffers_[0];
        } else {
            pool_buf_ = ggml_backend_multi_buffer_alloc_buffer(group_buffers_.data(), group_buffers_.size());
            GGML_ASSERT(pool_buf_ != nullptr);
            ggml_backend_buffer_set_usage(pool_buf_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }
        GGML_ASSERT(pool_buf_ != nullptr);

        return;
    }

    // CPU prefill spans the full pool, while decode uses each layer's slot range.
    size_t n_tensors = 0;
    for (uint32_t l = 0; l < n_total_layers_; ++l) {
        uint32_t n_sl = layer_slot_base_[l + 1] - layer_slot_base_[l];
        if (n_sl == 0) {
            continue;
        }
        for (int k = 0; k < N_KINDS; ++k) {
            if (layered_kind_size_[l * N_KINDS + k] > 0) {
                n_tensors += 2;  // decode + prefill
            }
        }
    }

    struct ggml_init_params ctx_params = {
        /* .mem_size   = */ n_tensors * ggml_tensor_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    layered_pool_ctx_ = ggml_init(ctx_params);
    GGML_ASSERT(layered_pool_ctx_ != nullptr);

    for (uint32_t l = 0; l < n_total_layers_; ++l) {
        uint32_t n_sl = layer_slot_base_[l + 1] - layer_slot_base_[l];
        if (n_sl == 0) {
            continue;
        }

        for (int k = 0; k < N_KINDS; ++k) {
            size_t ksz = layered_kind_size_[l * N_KINDS + k];
            if (ksz == 0) {
                continue;
            }

            auto [ne0, ne1] = kind_ne_flat[l * N_KINDS + k];
            if (ne0 == 0) {
                continue;
            }

            // Decode tensor: layer-local slot indices, ne[2]=n_sl
            ggml_tensor * t =
                ggml_new_tensor_3d(layered_pool_ctx_, kind_dtype_flat[l * N_KINDS + k], ne0, ne1, (int64_t) n_sl);
            GGML_ASSERT(t != nullptr);
            // Use the padded slot stride even when this layer's experts are smaller.
            t->nb[2] = kind_size_[k];
            t->nb[3] = (size_t) n_sl * kind_size_[k];
            alloc_pool_tensor(t, kind_region_offset_[k] + (size_t) layer_slot_base_[l] * kind_size_[k]);
            layered_pool_tensors_[l * N_KINDS + k] = t;

            // Prefill uses global slots with this layer's dtype and row stride.
            ggml_tensor * pt =
                ggml_new_tensor_3d(layered_pool_ctx_, kind_dtype_flat[l * N_KINDS + k], ne0, ne1, (int64_t) n_slots_);
            GGML_ASSERT(pt != nullptr);
            pt->nb[2] = kind_size_[k];
            pt->nb[3] = (size_t) n_slots_ * kind_size_[k];
            alloc_pool_tensor(pt, kind_region_offset_[k]);
            layered_prefill_tensors_[l * N_KINDS + k] = pt;
        }
    }
}

void llama_expert_cache::create_pool_tensors_shared(const std::vector<std::pair<int64_t, int64_t>> & kind_ne_flat,
                                                    const std::vector<ggml_type> &                   kind_dtype_flat) {
    GGML_ASSERT(kind_ne_flat.size() == (size_t) n_total_layers_ * N_KINDS);
    GGML_ASSERT(kind_dtype_flat.size() == (size_t) n_total_layers_ * N_KINDS);

    size_t n_tensors = 0;
    for (uint32_t l = 0; l < n_total_layers_; ++l) {
        for (int k = 0; k < N_KINDS; ++k) {
            if (kind_size_[k] == 0) {
                continue;
            }
            if (kind_ne_flat[l * N_KINDS + k].first == 0) {
                continue;
            }
            ++n_tensors;
        }
    }

    struct ggml_init_params ctx_params = {
        /* .mem_size   = */ n_tensors * ggml_tensor_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    layered_pool_ctx_ = ggml_init(ctx_params);
    GGML_ASSERT(layered_pool_ctx_ != nullptr);

    for (uint32_t l = 0; l < n_total_layers_; ++l) {
        for (int k = 0; k < N_KINDS; ++k) {
            if (kind_size_[k] == 0) {
                continue;
            }
            auto [ne0, ne1] = kind_ne_flat[l * N_KINDS + k];
            if (ne0 == 0) {
                continue;
            }

            ggml_tensor * t =
                ggml_new_tensor_3d(layered_pool_ctx_, kind_dtype_flat[l * N_KINDS + k], ne0, ne1, (int64_t) n_slots_);
            GGML_ASSERT(t != nullptr);

            // Set the padded slot stride before allocating the tensor.
            t->nb[2] = kind_size_[k];
            t->nb[3] = (size_t) n_slots_ * kind_size_[k];

            alloc_pool_tensor(t, kind_region_offset_[k]);

            GGML_ASSERT((size_t) t->nb[2] == kind_size_[k] && "shared pool tensor nb[2] must equal max_kind_sz");

            layered_pool_tensors_[(size_t) l * N_KINDS + k] = t;
        }
    }
}

ggml_tensor * llama_expert_cache::pool_tensor(uint32_t layer, tensor_kind k) const {
    if (!layered_mode_ && !shared_mode_) {
        return pool_tensors_[k];
    }
    if (layer >= n_total_layers_) {
        return nullptr;
    }
    return layered_pool_tensors_[(size_t) layer * N_KINDS + k];
}

ggml_tensor * llama_expert_cache::pool_tensor_for_layer(uint32_t il, tensor_kind k) const {
    if (il >= n_total_layers_) {
        return nullptr;
    }
    return layered_pool_tensors_[(size_t) il * N_KINDS + k];
}

const char * llama_expert_cache_stats_str(const llama_context * ctx) {
    static thread_local std::string result;
    if (!ctx) {
        return "ERROR: ctx is null\n";
    }
    const auto & expert_cache = ctx->get_model().expert_cache;
    if (!expert_cache || !expert_cache->stats_) {
        return "";
    }
    result = expert_cache->stats_to_str();
    return result.c_str();
}

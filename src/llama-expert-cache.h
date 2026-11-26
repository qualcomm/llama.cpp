#pragma once
#include "ggml-backend.h"
#include "ggml.h"
#include "llama-arch.h"
#include "llama-hparams.h"
#include "llama-mmap.h"
#include "llama.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

class ec_thread_pool;

struct ec_odirect;
struct ec_file;

struct ec_read_op;

// ggml_backend_cpu_buffer_from_ptr requires 32-byte alignment.
template <typename T, std::size_t Align = 32> struct aligned_allocator {
    using value_type      = T;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer         = T *;
    using const_pointer   = const T *;
    using reference       = T &;
    using const_reference = const T &;

    template <typename U> struct rebind {
        using other = aligned_allocator<U, Align>;
    };

    aligned_allocator() noexcept = default;

    template <class U> aligned_allocator(const aligned_allocator<U, Align> &) noexcept {}

    T * allocate(std::size_t n) {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_array_new_length();
        }
        return static_cast<T *>(::operator new(n * sizeof(T), std::align_val_t(Align)));
    }

    void deallocate(T * p, std::size_t) noexcept { ::operator delete(p, std::align_val_t(Align)); }

    template <typename U, typename... Args> void construct(U * p, Args &&... args) {
        ::new ((void *) p) U(std::forward<Args>(args)...);
    }

    template <typename U> void destroy(U * p) noexcept { p->~U(); }

    bool operator==(const aligned_allocator &) const noexcept { return true; }

    bool operator!=(const aligned_allocator &) const noexcept { return false; }
};

using aligned_uint8_vec = std::vector<uint8_t, aligned_allocator<uint8_t>>;

// Match expert weights without including their scale tensors.
inline bool is_expert_weight_tensor(const char * name) {
    const size_t len     = strlen(name);
    const char * suffix  = ".weight";
    const size_t suf_len = 7;
    if (len < suf_len || strcmp(name + len - suf_len, suffix) != 0) {
        return false;
    }
    auto check = [&](const char * pattern) -> bool {
        const char * p = strstr(name, pattern);
        if (!p) {
            return false;
        }
        return strcmp(p + strlen(pattern), ".weight") == 0;
    };
    return check("ffn_gate_up_exps") || check("ffn_gate_exps") || check("ffn_up_exps") || check("ffn_down_exps");
}

// Slot states: EMPTY -> LOADING -> OCCUPIED -> EVICTING -> EMPTY
enum class llama_expert_slot_state : uint32_t {
    EMPTY    = 0,
    LOADING  = 1,
    OCCUPIED = 2,
    EVICTING = 3,
};

struct llama_expert_key {
    uint32_t layer_idx;
    uint32_t expert_idx;
};

struct llama_expert_slot {
    std::atomic<uint32_t> state{ (uint32_t) llama_expert_slot_state::EMPTY };
    std::atomic<uint64_t> usage{ 0 };
    std::atomic<uint64_t> last_use{ 0 };          // LRU tie-breaker
    llama_expert_key      key{ 0, 0 };
    std::atomic<uint8_t>  warmed_by_worker{ 0 };  // set by warm_cache, cleared on use/evict
    uint32_t              readers = 0;            // guarded by reserve_mu_
};

// Router input transforms applied before the gate_inp matmul.
enum moe_pre_gating_kind {
    MOE_PRE_GATING_NONE   = 0,
    MOE_PRE_GATING_GEMMA4 = 1,
};

// Shared router settings, with gate_inp == nullptr for dense layers.
struct moe_router_desc {
    ggml_tensor * gate_inp    = nullptr;
    ggml_tensor * gate_inp_b  = nullptr;
    ggml_tensor * exp_probs_b = nullptr;

    llama_expert_gating_func_type gating_op         = LLAMA_EXPERT_GATING_FUNC_TYPE_NONE;
    bool                          expert_router_f32 = false;

    int64_t n_expert        = 0;
    int64_t n_expert_used   = 0;
    int64_t n_expert_groups = 0;
    int64_t n_group_used    = 0;

    llm_arch arch = LLM_ARCH_UNKNOWN;

    int64_t n_group_experts = 0;  // GROVEMOE

    bool prefetch_capable = false;

    moe_pre_gating_kind pre_gating_kind = MOE_PRE_GATING_NONE;
    ggml_tensor *       gate_inp_s      = nullptr;  // scale tensor (gemma4 pre-gating)
    float               f_norm_rms_eps  = 0.0f;     // rms_norm epsilon (gemma4 pre-gating)
};

struct llama_expert_cache {
    struct expert_slice_loc {
        size_t   offset;
        size_t   size;
        uint16_t file_idx = 0;
    };

    enum tensor_kind { GATE_UP = 0, GATE = 1, UP = 2, DOWN = 3, N_KINDS = 4 };

    // Read runtime settings from the environment.
    llama_expert_cache();
    ~llama_expert_cache();

    // kind_sizes[k] = bytes per expert for tensor kind k (0 = absent)
    void init_kind_first(uint32_t n_slots, const std::vector<size_t> & kind_sizes);

    // Build the shared kind-first layout.
    void      init_kind_first_layout(const size_t * kind_sizes);
    // Return the CPU pool base when needed.
    uint8_t * ensure_pool_buf();

    // Claim an EMPTY slot as LOADING, returning its index or -1 on failure.
    int32_t  claim_slot(uint32_t layer, uint32_t expert);
    // Restrict the claim to [begin, end) for prefill groups.
    int32_t  claim_slot_in_range(uint32_t layer, uint32_t expert, uint32_t begin, uint32_t end);
    // Copy an out-of-range prefetched expert into the requested range when enabled.
    int32_t  claim_or_copy_slot_in_range(uint32_t layer, uint32_t expert, uint32_t begin, uint32_t end);
    void     mark_occupied(int32_t slot);
    void     record_use(int32_t slot);
    void     release(int32_t slot);
    uint64_t next_tick();

    uint8_t *       kind_slot_ptr(tensor_kind k, uint32_t slot);
    const uint8_t * kind_slot_ptr(tensor_kind k, uint32_t slot) const;

    std::unique_ptr<llama_expert_slot[]> slots;
    uint32_t                             n_slots_       = 0;
    uint64_t                             budget_bytes_  = 0;
    uint64_t                             minimum_bytes_ = 0;
    uint64_t                             maximum_bytes_ = 0;
    bool                                 no_alloc_      = false;
    bool                                 automatic_     = false;
    aligned_uint8_vec                    ddr_pool;  // CPU pool
    size_t                               expert_sz = 0;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;
    void                                         alloc_pool_tensor(ggml_tensor * tensor, size_t offset);

    // Configured once from LLAMA_EXPERT_CACHE_* environment variables.
    bool stats_                           = false;  // collect and print [ECSTATS] lines
    bool mlock_                           = false;  // mlock() the DDR pool after allocation
    bool ddr_copy_                        = false;  // DDR-to-DDR copy instead of re-reading flash
    bool htp_async_                       = true;
    bool decode_prefetch_force_cache_hit_ = false;  // force cache hit on prefetch-covered decode layers
    int  read_percentage_                 = 0;      // 0/100=read all, else cap decode misses to ceil(P%)
    int  decode_prefetch_lookahead_       = 0;      // initial lookahead in layers, 0 disables
    int  batch_depth_                     = 4;      // experts per read batch
    int  htp_read_chunk_kb_               = 512;    // HTP raw-read target for undersized decode batches, 0=disabled

    // Keep the CPU pool resident when mlock_ is enabled.
    std::unique_ptr<llama_mlock> pool_mlock_;

    // Set before init_* to allocate HTP buffers and repack weights on each miss.
    ggml_backend_buffer_type_t htp_repack_buft_   = nullptr;
    bool                       use_htp_pool_      = false;
    size_t                     htp_max_pool_size_ = 1ull << 30;
    using htp_repack_range_fn                     = void (*)(ggml_tensor *, const void *, size_t, size_t);
    htp_repack_range_fn htp_repack_range_         = nullptr;
    using htp_mark_tensor_expert_cache_fn         = void (*)(ggml_tensor *);
    htp_mark_tensor_expert_cache_fn htp_mark_callback_            = nullptr;

    void init_pool_tensor(ggml_tensor * tensor) const;

    // Bytes per expert for each kind, with zero for absent kinds.
    size_t kind_size_[N_KINDS]          = {};
    // kind_region_offset_[k] = sum of kind_size_[j] * n_slots_ for j < k
    size_t kind_region_offset_[N_KINDS] = {};

    ggml_tensor *         pool_tensors_[N_KINDS] = {};
    ggml_backend_buffer_t pool_buf_              = nullptr;
    ggml_context *        pool_ggml_ctx_         = nullptr;
    std::mutex            htp_repack_mu_;

    std::vector<ggml_backend_buffer_t> group_buffers_;
    // Group layout used by kind_slot_ptr in HTP mode.
    std::vector<uint8_t *>             group_base_ptr_;   // FASTRPC buffer base per group
    std::vector<uint32_t>              group_slot_base_;  // first global slot index for group g
    std::vector<uint32_t>              group_n_slots_;    // number of slots in group g

    // Create pool tensors after init_kind_first().
    void create_pool_tensors(const std::array<std::pair<int64_t, int64_t>, N_KINDS> & kind_ne,
                             const std::array<ggml_type, N_KINDS> &                   kind_dtype);

    ggml_tensor * pool_tensor(tensor_kind k) const { return pool_tensors_[k]; }

    // Layered mode: per-layer pool partitioning for non-uniform expert sizes.
    bool     layered_mode_   = false;
    uint32_t n_total_layers_ = 0;  // = n_layer_ (indexed by actual layer index)

    // Shared mode uses max-kind strides and per-layer pool tensors.
    bool   shared_mode_   = false;
    size_t max_expert_sz_ = 0;  // sum of all max_kind_sizes (total slot footprint)

    // Layer slot offsets, ending with the total slot count.
    std::vector<uint32_t> layer_slot_base_;
    std::vector<uint32_t> layer_group_base_;   // first group index for layer il
    std::vector<uint32_t> layer_group_count_;  // groups layer il spans (1 = normal)

    struct htp_group_plan {
        struct group_def {
            uint32_t first_l;
            uint32_t last_l;
            uint32_t slot_base;
            uint32_t n_slots;
        };

        std::vector<uint32_t>  slots_per_layer;
        std::vector<group_def> groups;
        std::string            error;
    };

    std::vector<htp_group_plan::group_def> planned_htp_groups_;

    static htp_group_plan minimum_htp_layered_groups(const std::vector<uint32_t> & slots_per_layer,
                                                     size_t                        slot_bytes,
                                                     size_t                        mbuf,
                                                     uint32_t                      prefill_floor);

    static htp_group_plan plan_htp_layered_groups(const std::vector<uint32_t> & slots_per_layer,
                                                  const std::vector<uint32_t> & floor_per_layer,
                                                  uint32_t                      leftover,
                                                  size_t                        slot_bytes,
                                                  size_t                        mbuf,
                                                  uint32_t                      prefill_floor);

    // Per-layer-kind data: indexed as [layer * N_KINDS + kind]
    std::vector<size_t>        layered_kind_size_;        // bytes per expert for this (layer, kind)
    std::vector<size_t>        layered_kind_base_;        // byte offset in ddr_pool for this (layer, kind)'s region
    std::vector<ggml_tensor *> layered_pool_tensors_;     // per-layer decode tensors (layer-local slot indices)
    std::vector<ggml_tensor *> layered_prefill_tensors_;  // prefill tensors spanning the CPU pool or HTP group

    ggml_context * layered_pool_ctx_ = nullptr;

    // Configure per-layer slots and kind sizes for non-uniform models.
    void init_layered(uint32_t                      n_total_layers,
                      const std::vector<uint32_t> & slots_per_layer,
                      const std::vector<size_t> &   kind_sizes_flat);

    // Share padded slots across layers with different expert layouts.
    void init_shared(uint32_t n_total_layers, uint32_t n_slots, const std::array<size_t, N_KINDS> & max_kind_sizes);

    void create_pool_tensors_shared(const std::vector<std::pair<int64_t, int64_t>> & kind_ne_flat,
                                    const std::vector<ggml_type> &                   kind_dtype_flat);

    void create_pool_tensors_layered(const std::vector<std::pair<int64_t, int64_t>> & kind_ne_flat,
                                     const std::vector<ggml_type> &                   kind_dtype_flat,
                                     uint32_t                                         prefill_floor);

    // Return the layer-specific or shared pool tensor.
    ggml_tensor * pool_tensor(uint32_t layer, tensor_kind k) const;
    ggml_tensor * pool_tensor_for_layer(uint32_t il, tensor_kind k) const;

    // Return the layer tensor used by prefill or the shared pool tensor.
    ggml_tensor * group_global_tensor(uint32_t il, tensor_kind k) const {
        if (layered_mode_ && !layered_prefill_tensors_.empty() && il < n_total_layers_) {
            return layered_prefill_tensors_[(size_t) il * N_KINDS + k];
        }
        return pool_tensors_[k];
    }

    std::atomic<uint64_t> tick{ 0 };

    // Collect stats only when enabled.
    std::atomic<uint64_t> stat_hits{ 0 };    // (token,expert) requests satisfied without flash
    std::atomic<uint64_t> stat_misses{ 0 };  // (token,expert) requests that require flash
    std::atomic<uint64_t> stat_subs{ 0 };    // threshold substitutions used
    // Keep forced substitutions separate from threshold substitutions.
    std::atomic<uint64_t> stat_forced_subs{ 0 };      // covered-layer picks that needed substitution
    std::atomic<uint64_t> stat_read_pct_subs{ 0 };    // substitutions made to limit flash reads
    std::atomic<uint64_t> stat_prefetch_hits{ 0 };    // covered-layer picks already cached
    std::atomic<uint64_t> stat_ddr_copies{ 0 };       // requests served by copying between slots
    std::atomic<uint64_t> stat_prefetch_adopts{ 0 };  // requests served by waiting for a worker load
    std::atomic<uint64_t> stat_flash_reads{ 0 };      // read batches during decode
    std::atomic<uint64_t> stat_flash_bytes{ 0 };      // bytes read from GGUF during decode
    std::atomic<uint64_t> stat_flash_ns{ 0 };         // accumulated read time in ns
    std::atomic<uint64_t> stat_eval_us{ 0 };          // decode eval time in microseconds
    // prefill phase counters (mirrored from above)
    std::atomic<uint64_t> stat_p_hits{ 0 };
    std::atomic<uint64_t> stat_p_misses{ 0 };
    std::atomic<uint64_t> stat_p_subs{ 0 };
    std::atomic<uint64_t> stat_p_ddr_copies{ 0 };
    std::atomic<uint64_t> stat_p_prefetch_adopts{ 0 };
    std::atomic<uint64_t> stat_p_flash_reads{ 0 };
    std::atomic<uint64_t> stat_p_flash_bytes{ 0 };
    std::atomic<uint64_t> stat_p_flash_ns{ 0 };
    std::atomic<uint64_t> stat_p_eval_us{ 0 };

    // Prefetch counters stay zero when the worker is disabled.
    std::atomic<uint64_t> stat_prefetch_warmed{ 0 };   // experts the worker loaded from flash
    std::atomic<uint64_t> stat_prefetch_useful{ 0 };   // warmed slots later consumed by main thread
    std::atomic<uint64_t> stat_prefetch_wasted{ 0 };   // warmed then evicted before any use
    std::atomic<uint64_t> stat_warm_skip_cached{ 0 };  // warm_cache skip: already cached
    std::atomic<uint64_t> stat_warm_skip_noslot{ 0 };  // warm_cache skip: no slot even after eviction attempt
    std::atomic<uint64_t> stat_worker_evictions{ 0 };  // evictions performed by warm_cache when a layer is full
    std::atomic<uint64_t> stat_predict_calls{ 0 };     // worker_predict invocations
    std::atomic<uint64_t> stat_predict_ns{ 0 };        // total worker_predict wall time
    std::atomic<uint64_t> stat_htp_main_read_ns{ 0 };
    std::atomic<uint64_t> stat_htp_main_copy_ns{ 0 };
    std::atomic<uint64_t> stat_htp_main_repack_ns{ 0 };
    std::atomic<uint64_t> stat_htp_worker_read_ns{ 0 };
    std::atomic<uint64_t> stat_htp_worker_copy_ns{ 0 };
    std::atomic<uint64_t> stat_htp_worker_repack_ns{ 0 };

    // Track union busy time so overlapping reads count once.
    std::mutex            flash_busy_mu_;
    uint32_t              flash_in_flight_  = 0;
    uint64_t              flash_busy_start_ = 0;
    std::atomic<uint64_t> flash_busy_ns_{ 0 };
    // Worker reads count toward decode statistics.
    std::atomic<uint64_t> flash_bytes_worker_{ 0 };

    // Start a read interval and return its timestamp.
    uint64_t flash_begin();
    // Close a read interval and add its bytes to the decode statistics.
    void     flash_end(uint64_t token, uint64_t bytes, bool from_worker = false, uint64_t end_ns = 0);

    uint64_t decode_flash_bytes() const { return stat_flash_bytes.load(); }

    uint64_t decode_flash_busy_ns() const { return flash_busy_ns_.load(); }

    std::string stats_to_str() const;
    void        stats_reset();
    void        stats_record_hit(bool is_prefill);
    void        stats_record_miss(bool is_prefill);
    void        stats_record_ddr_copy(bool is_prefill);
    void        stats_record_prefetch_adopt(bool is_prefill);
    void        stats_record_substitution(bool is_prefill);
    void        stats_record_flash_read(uint64_t bytes, uint64_t ns, bool is_prefill);
    void        stats_set_eval_time_us(uint64_t p_eval_us, uint64_t eval_us);

    std::unordered_map<uint64_t, uint32_t> index_;  // guarded by index_mu_
    mutable std::mutex                     index_mu_;
    std::condition_variable                slots_cv_;
    std::mutex                             reserve_mu_;
    std::mutex                             htp_compute_mu_;

    static uint64_t make_key(uint32_t layer, uint32_t expert) { return ((uint64_t) layer << 32) | expert; }

    // offsets_[layer * N_KINDS * n_expert_ + kind * n_expert_ + expert]
    std::vector<expert_slice_loc> offsets_;

    // Router settings indexed by layer and fixed at model load.
    std::vector<moe_router_desc> router_descs_;

    // Evaluate worker routing with one cached graph per layer.
    ggml_context * worker_ctx_     = nullptr;
    ggml_backend_t worker_backend_ = nullptr;

    // Reuse one router graph per layer to avoid rebuilding it for each prediction.
    struct worker_pred_graph {
        ggml_context *        ctx = nullptr;
        ggml_cgraph *         gf  = nullptr;
        ggml_backend_buffer_t buf = nullptr;
        ggml_tensor *         inp = nullptr;  // [n_embd, 1] input tensor
        ggml_tensor *         out = nullptr;  // selected_experts output tensor
    };

    std::vector<worker_pred_graph> worker_pred_graphs_;
    worker_pred_graph &            worker_pred_graph_for(uint32_t layer, int64_t n_embd);

    void prefetch_init();
    void prefetch_teardown();

    // Predict the layer's top-k experts from hidden[n_embd] after prefetch_init().
    void worker_predict(uint32_t layer, const float * hidden, int64_t n_embd, std::vector<int32_t> & result);

    // Copy snapshots under worker_mu_.
    struct prefetch_snapshot {
        uint32_t           layer    = 0;
        uint64_t           token_id = 0;
        std::vector<float> hidden;
    };

    prefetch_snapshot prefetch_snapshot_;

    // Unique across contexts sharing this model.
    std::atomic<uint64_t> decode_step_counter_{ 0 };

    // UINT64_MAX means that no execution has fully prefetched this layer.
    std::unique_ptr<std::atomic<uint64_t>[]> layer_prefetch_done_token_;

    // Publish the router input without changing it.
    static void snapshot_fn(struct ggml_tensor * dst, const struct ggml_tensor * a, int ith, int nth, void * userdata);

    // Best-effort prefetch follows the most recently published execution.
    std::thread             worker_thread_;
    std::mutex              worker_mu_;
    std::condition_variable worker_cv_;
    std::atomic<bool>       worker_shutdown_{ false };
    // Increment under worker_mu_ so the wait predicate cannot miss a snapshot.
    uint64_t                worker_wake_seq_ = 0;

    // Start the worker if prefetch is enabled.
    void start_worker();
    // Stop and join the worker before freeing its graph and backend.
    void stop_worker();
    // Predict and load experts ahead of the latest snapshot's layer.
    void worker_loop();
    bool read_snapshot(uint32_t & out_layer, uint64_t & out_token, std::vector<float> & out_hidden);
    // Prefetch skips slots that still have readers.
    void warm_cache(uint32_t layer, const std::vector<int32_t> & predicted_expert_ids, uint64_t token = 0);

    uint32_t                  n_layer_       = 0;
    uint32_t                  n_expert_      = 0;
    uint32_t                  n_expert_used_ = 0;
    float                     threshold_pct_ = 0.0f;  // 0 = disabled
    llama_expert_cache_policy policy_        = LLAMA_EXPERT_CACHE_POLICY_LFU;

    bool init_check_capacity_floor() const {
        // At least 2 * n_expert_used slots: one full layer's working set plus headroom.
        return (uint64_t) n_slots_ >= (uint64_t) 2 * (uint64_t) n_expert_used_;
    }

    void build_offset_map(uint32_t                              n_layer,
                          uint32_t                              n_expert,
                          uint32_t                              n_expert_used,
                          const std::vector<expert_slice_loc> & locs);

    expert_slice_loc get_loc(uint32_t layer, tensor_kind kind, uint32_t expert) const;

    struct op_params_t {
        llama_expert_cache * cache        = nullptr;
        uint32_t             layer_idx    = 0;
        uint64_t             token        = 0;
        std::vector<int32_t> slot_ids     = {};
        std::vector<int32_t> pinned       = {};
        std::vector<int32_t> cached_slots = {};
        std::vector<uint8_t> pinned_mark  = {};
    };

    void        release_slots(op_params_t & op);
    static void release_slots_fn(ggml_tensor * dst, const ggml_tensor * src, int ith, int nth, void * userdata);

    // Load selected experts, using selection_probs to choose cached substitutes.
    static void reroute_fn(struct ggml_tensor *       dst,
                           const struct ggml_tensor * a,
                           const struct ggml_tensor * b,
                           int                        ith,
                           int                        nth,
                           void *                     userdata);

    // Raw HTP weights for one read batch, repacked into FASTRPC slots after reading.
    aligned_uint8_vec htp_read_buf_;

    // Separate staging lets the worker read while main-thread reads hold reserve_mu_.
    aligned_uint8_vec htp_read_buf_prefetch_;

    // Write slot indices into the matmul ID tensor.
    static void id_to_slot_fn(struct ggml_tensor *       dst,
                              const struct ggml_tensor * src,
                              int                        ith,
                              int                        nth,
                              void *                     userdata);

    // Return an LFU victim without pins or readers, or -1 if none is evictable.
    int32_t evict_lfu_excluding(const int32_t * pinned,
                                int             n_pinned,
                                uint32_t        layer = UINT32_MAX,
                                uint32_t        begin = 0,
                                uint32_t        end   = UINT32_MAX);
    int32_t evict_lru_excluding(const int32_t * pinned,
                                int             n_pinned,
                                uint32_t        layer = UINT32_MAX,
                                uint32_t        begin = 0,
                                uint32_t        end   = UINT32_MAX);

    std::vector<std::unique_ptr<ec_file>> files_;
    bool                                  add_file(int fd, bool use_direct_io = false);

    // Bounce buffers shared by all shards, or null when direct I/O is disabled.
    ec_odirect * odirect_ = nullptr;

    bool odirect_active() const { return odirect_ != nullptr; }

    // Expose reads to tests while keeping ec_odirect private.
    int64_t odirect_pread_for_test(size_t pool_slot, size_t buf_idx, void * dst, size_t size, int64_t offset);

    // Size read pools by the tensor kinds present in the model.
    int n_kinds_present() const;

    ec_thread_pool * thread_pool_          = nullptr;
    bool             thread_pool_disabled_ = false;
    bool             thread_pool_logged_   = false;
    ec_thread_pool * ensure_thread_pool();

    ec_thread_pool * worker_thread_pool_          = nullptr;
    bool             worker_thread_pool_disabled_ = false;
    ec_thread_pool * ensure_worker_thread_pool();

    bool read_batch_main(const ec_read_op * ops, uint32_t n);

    bool read_batch_worker(const ec_read_op * ops, uint32_t n);

    void load_expert_sync(int32_t slot, uint32_t layer, uint32_t expert, bool is_prefill = false);

    struct load_req {
        int32_t  slot;
        uint32_t layer;
        uint32_t expert;
    };

    // Fill CPU slots in batches and return the bytes read.
    enum class cpu_load_origin { main, worker };
    uint64_t load_experts_batch_cpu(const load_req * reqs, uint32_t n, bool is_prefill, cpu_load_origin origin);

    // Read and repack HTP slots in batches, returning the bytes read.
    uint64_t load_experts_batch_htp(const load_req * reqs, uint32_t n, bool is_prefill);

    // Load HTP slots with worker staging and count the reads as decode.
    uint64_t load_experts_batch_htp_prefetch(const load_req * reqs, uint32_t n);
};

// Expert tensor layouts are checked at model load rather than by architecture.
inline bool arch_supports_expert_cache(llm_arch arch) {
    (void) arch;
    return true;
}

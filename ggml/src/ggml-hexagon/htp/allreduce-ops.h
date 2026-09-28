#ifndef ALLREDUCE_OPS_H
#define ALLREDUCE_OPS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#define HTP_ALLREDUCE_MAX_RANKS 4

#ifdef __cplusplus
extern "C" {
#endif

enum htp_allreduce_kernel_type {
    HTP_ALLREDUCE_KERNEL_UNSUPPORTED = 0,
    HTP_ALLREDUCE_KERNEL_DMA_1D,
    HTP_ALLREDUCE_KERNEL_DMA_2D,
};

// Collective mode for the fused ALLREDUCE(+ADD). Chosen once by the host planner
// (ggml_hexagon_precompute_allreduce_params) and consumed by every later stage --
// fusion emit, the HTP kernel, and per-phase instrumentation -- instead of being
// re-derived from n_dsts at each site.
//
//   FULL           : every rank reduces the whole tensor into its own buffer.
//                    Redundant compute (N x reduction), but no cross-rank writes.
//   SHARDED_FANOUT : each rank reduces only its own disjoint 1/N shard, then writes
//                    that shard into all N per-rank buffers (all-gather by fan-out).
//                    Cuts reduction compute N-fold; fan-out DMA is N x B traffic.
enum htp_allreduce_mode {
    HTP_ALLREDUCE_FULL           = 0,
    HTP_ALLREDUCE_SHARDED_FANOUT = 1,
};

static inline size_t htp_allreduce_vtcm_buffer_count(
    uint32_t n_ranks,
    uint32_t n_threads,
    bool has_add,
    bool is_row_bcast
) {
    return (size_t) (n_ranks + 1) * n_threads + (has_add ? (is_row_bcast ? 1 : n_threads) : 0);
}

struct htp_allreduce_kernel_params {
    int32_t rank;
    int32_t n_ranks;
    int32_t n_threads;
    int32_t block_elems;          // 1D: block_elems, 2D: block_rows
    int32_t elems_per_thread;     // 1D: nelem_per_thread, 2D: nrows_per_thread
    int32_t vtcm_size_per_thread;
    int32_t vtcm_size;
    int32_t kernel_type;
    int32_t ne0;
    int32_t ne1;
    int32_t row_size_aligned;
    int32_t rank_elem_start;
    int32_t rank_nelem;
    int32_t n_dsts;
    int32_t is_row_bcast;
    int32_t mode;                 // enum htp_allreduce_mode: FULL or SHARDED_FANOUT
};

#ifdef __cplusplus
}
#endif

#endif /* ALLREDUCE_OPS_H */

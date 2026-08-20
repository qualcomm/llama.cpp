#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>
#include <stdatomic.h>
#include <math.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "hvx-utils.h"
#include "htp-tensor.h"
#include "hex-profile.h"
#include "allreduce-ops.h"

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

struct htp_allreduce_context {
    struct htp_ops_context * octx;
    uint32_t n_ranks;
    uint32_t nelem;
    uint32_t elem_size;
    uint32_t elems_per_thread;
};

static void allreduce_thread_f16_aaa(unsigned int nth, unsigned int ith, void * data) {
    struct htp_allreduce_context * actx = (struct htp_allreduce_context *) data;
    struct htp_ops_context * octx = actx->octx;

    const uint32_t dr  = actx->elems_per_thread;
    const uint32_t ir0 = dr * ith;
    const uint32_t ir1 = MIN(ir0 + dr, actx->nelem);
    if (ir0 >= ir1) return;

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ith);

    const uint32_t n_elems  = ir1 - ir0;
    const uint32_t elem_off = ir0;

    const struct htp_tensor * dst = octx->dst;
    uint8_t * dst_ptr = (uint8_t *) dst->data + elem_off * sizeof(__fp16);
    const uint8_t * src0_ptr = (const uint8_t *) octx->src[0]->data + elem_off * sizeof(__fp16);
    const uint8_t * src1_ptr = (const uint8_t *) octx->src[1]->data + elem_off * sizeof(__fp16);

    hvx_add_f16_aaa(dst_ptr, src0_ptr, src1_ptr, n_elems);

    for (uint32_t s = 2; s < actx->n_ranks; s++) {
        const uint8_t * srcs_ptr = (const uint8_t *) octx->src[s]->data + elem_off * sizeof(__fp16);
        hvx_add_f16_aaa(dst_ptr, dst_ptr, srcs_ptr, n_elems);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ith);
}

static void allreduce_thread_f16_uuu(unsigned int nth, unsigned int ith, void * data) {
    struct htp_allreduce_context * actx = (struct htp_allreduce_context *) data;
    struct htp_ops_context * octx = actx->octx;

    const uint32_t dr  = actx->elems_per_thread;
    const uint32_t ir0 = dr * ith;
    const uint32_t ir1 = MIN(ir0 + dr, actx->nelem);
    if (ir0 >= ir1) return;

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ith);

    const uint32_t n_elems  = ir1 - ir0;
    const uint32_t elem_off = ir0;

    const struct htp_tensor * dst = octx->dst;
    uint8_t * dst_ptr = (uint8_t *) dst->data + elem_off * sizeof(__fp16);
    const uint8_t * src0_ptr = (const uint8_t *) octx->src[0]->data + elem_off * sizeof(__fp16);
    const uint8_t * src1_ptr = (const uint8_t *) octx->src[1]->data + elem_off * sizeof(__fp16);

    hvx_add_f16_uuu(dst_ptr, src0_ptr, src1_ptr, n_elems);

    for (uint32_t s = 2; s < actx->n_ranks; s++) {
        const uint8_t * srcs_ptr = (const uint8_t *) octx->src[s]->data + elem_off * sizeof(__fp16);
        hvx_add_f16_uuu(dst_ptr, dst_ptr, srcs_ptr, n_elems);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ith);
}

static void allreduce_thread_f32_aaa(unsigned int nth, unsigned int ith, void * data) {
    struct htp_allreduce_context * actx = (struct htp_allreduce_context *) data;
    struct htp_ops_context * octx = actx->octx;

    const uint32_t dr  = actx->elems_per_thread;
    const uint32_t ir0 = dr * ith;
    const uint32_t ir1 = MIN(ir0 + dr, actx->nelem);
    if (ir0 >= ir1) return;

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ith);

    const uint32_t n_elems  = ir1 - ir0;
    const uint32_t elem_off = ir0;

    const struct htp_tensor * dst = octx->dst;
    uint8_t * dst_ptr = (uint8_t *) dst->data + elem_off * sizeof(float);
    const uint8_t * src0_ptr = (const uint8_t *) octx->src[0]->data + elem_off * sizeof(float);
    const uint8_t * src1_ptr = (const uint8_t *) octx->src[1]->data + elem_off * sizeof(float);

    hvx_add_f32_aaa(dst_ptr, src0_ptr, src1_ptr, n_elems);

    for (uint32_t s = 2; s < actx->n_ranks; s++) {
        const uint8_t * srcs_ptr = (const uint8_t *) octx->src[s]->data + elem_off * sizeof(float);
        hvx_add_f32_aaa(dst_ptr, dst_ptr, srcs_ptr, n_elems);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ith);
}

static void allreduce_thread_f32_uuu(unsigned int nth, unsigned int ith, void * data) {
    struct htp_allreduce_context * actx = (struct htp_allreduce_context *) data;
    struct htp_ops_context * octx = actx->octx;

    const uint32_t dr  = actx->elems_per_thread;
    const uint32_t ir0 = dr * ith;
    const uint32_t ir1 = MIN(ir0 + dr, actx->nelem);
    if (ir0 >= ir1) return;

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ith);

    const uint32_t n_elems  = ir1 - ir0;
    const uint32_t elem_off = ir0;

    const struct htp_tensor * dst = octx->dst;
    uint8_t * dst_ptr = (uint8_t *) dst->data + elem_off * sizeof(float);
    const uint8_t * src0_ptr = (const uint8_t *) octx->src[0]->data + elem_off * sizeof(float);
    const uint8_t * src1_ptr = (const uint8_t *) octx->src[1]->data + elem_off * sizeof(float);

    hvx_add_f32_uuu(dst_ptr, src0_ptr, src1_ptr, n_elems);

    for (uint32_t s = 2; s < actx->n_ranks; s++) {
        const uint8_t * srcs_ptr = (const uint8_t *) octx->src[s]->data + elem_off * sizeof(float);
        hvx_add_f32_uuu(dst_ptr, dst_ptr, srcs_ptr, n_elems);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ith);
}

int op_allreduce(struct htp_ops_context * octx) {
    const struct htp_tensor * dst = octx->dst;

    const uint32_t rank    = (uint32_t) octx->kernel_params[0];
    const uint32_t n_ranks = (uint32_t) octx->kernel_params[1];
    const uint32_t seq     = (uint32_t) octx->kernel_params[2];

    if (n_ranks < 2 || n_ranks > 4 || rank >= n_ranks) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    if (dst->type != HTP_TYPE_F16 && dst->type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t elem_size = (dst->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);
    const uint32_t nelem     = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];

    // 1. Flush local output to DDR so peers can read it
    htp_tensor_flush_all(octx->ctx, (const struct htp_tensor * const *) &dst, 1);

    // 2. Single Barrier: Synchronize all ranks
    struct htp_thread_trace * tr0 = &octx->ctx->trace[0];
    htp_trace_event_start(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) rank);

    const struct htp_tensor * my_sync = octx->src[n_ranks + rank];
    atomic_uint * my_fence = (atomic_uint *) my_sync->data;
    Q6_dccleaninva_A((void *) my_fence);
    atomic_store(my_fence, seq);
    asm volatile ("syncht" : : : "memory");
    Q6_dccleaninva_A((void *) my_fence);

    for (uint32_t j = 0; j < n_ranks; j++) {
        if (j == rank) continue;
        const struct htp_tensor * peer_sync = octx->src[n_ranks + j];
        atomic_uint * peer_fence = (atomic_uint *) peer_sync->data;
        uint64_t spins = 0;
        while (1) {
            Q6_dccleaninva_A((void *) peer_fence);
            asm volatile ("syncht" : : : "memory");
            if (atomic_load(peer_fence) == seq) {
                break;
            }
            if (++spins > HTP_FENCE_TIMEOUT) {
                FARF(ERROR, "ggml-hex: allreduce sync-wait TIMEOUT: rank %u waiting on %u (fence %p seq %u)\n",
                     rank, j, peer_fence, seq);
                return HTP_STATUS_INTERNAL_ERR;
            }
            hex_pause();
        }
    }
    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) rank);

    // 3. Multi-threaded HVX Vector Reduction across all ranks
    if (nelem > 0) {
        bool is_aligned = hex_is_aligned((const void *)(uintptr_t) dst->data, 128);
        for (uint32_t s = 0; s < n_ranks; s++) {
            is_aligned &= hex_is_aligned((const void *)(uintptr_t) octx->src[s]->data, 128);
        }

        uint32_t n_threads = MIN(nelem, octx->n_threads);
        struct htp_allreduce_context actx;
        actx.octx             = octx;
        actx.n_ranks          = n_ranks;
        actx.nelem            = nelem;
        actx.elem_size        = elem_size;
        actx.elems_per_thread = is_aligned ? hex_round_up((nelem + n_threads - 1) / n_threads, 64)
                                           : (nelem + n_threads - 1) / n_threads;

        worker_callback_t reduce_fun;
        if (dst->type == HTP_TYPE_F16) {
            reduce_fun = is_aligned ? allreduce_thread_f16_aaa : allreduce_thread_f16_uuu;
        } else {
            reduce_fun = is_aligned ? allreduce_thread_f32_aaa : allreduce_thread_f32_uuu;
        }
        worker_pool_run_func(octx->ctx->worker_pool, reduce_fun, &actx, n_threads);
    }

    return HTP_STATUS_OK;
}

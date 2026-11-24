#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>

#include <string.h>

#include "hvx-utils.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "hex-common.h"
#include "htp-trace.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "dma-queue.h"
#include "repeat-ops.h"

struct htp_repeat_context {
    struct htp_ops_context *               octx;
    const struct htp_repeat_kernel_params * kparams;

    uint32_t nrows_per_hvx;
    uint32_t total_dst_rows;  // ne1 * ne2 * ne3
    uint32_t row_start;

    size_t   type_size;
    size_t   scratch_stride; // VTCM bytes per thread when bouncing an extended pointer
    int      bounce;
};

static void repeat_job_per_hvx(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    const struct htp_repeat_context * rctx = (const struct htp_repeat_context *) data;
    struct htp_ops_context * octx = rctx->octx;
    const struct htp_repeat_kernel_params * kparams = rctx->kparams;
    const struct htp_tensor * src = octx->src[0];
    const struct htp_tensor * dst = octx->dst;

    const uint32_t ne00 = src->ne[0];
    const uint32_t ne01 = src->ne[1];
    const uint32_t ne02 = src->ne[2];
    const uint32_t ne03 = src->ne[3];

    const uint32_t nb00 = src->nb[0];
    const uint32_t nb01 = src->nb[1];
    const uint32_t nb02 = src->nb[2];
    const uint32_t nb03 = src->nb[3];

    const uint32_t ne0 = dst->ne[0];
    const uint32_t ne1 = dst->ne[1];
    const uint32_t ne2 = dst->ne[2];
    const uint32_t ne3 = dst->ne[3];

    const uint32_t nb0 = dst->nb[0];
    const uint32_t nb1 = dst->nb[1];
    const uint32_t nb2 = dst->nb[2];
    const uint32_t nb3 = dst->nb[3];

    const uint32_t nr0 = kparams->nr0;

    const size_t row_bytes = ne00 * rctx->type_size;

    const uint32_t row_start = rctx->row_start + rctx->nrows_per_hvx * ith;
    const uint32_t row_end   = MIN(row_start + rctx->nrows_per_hvx, rctx->row_start + rctx->total_dst_rows);
    if (row_start >= row_end) {
        return;
    }

    struct htp_thread_trace * tr = th->trace;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    for (uint32_t dst_row = row_start; dst_row < row_end; dst_row++) {
        // Decompose flat dst row index into (i1, i2, i3)
        const uint32_t dst_row_div_ne1 = fastdiv(dst_row, &kparams->div_ne1);
        const uint32_t i1              = dst_row - dst_row_div_ne1 * ne1;
        const uint32_t i3              = fastdiv(dst_row, &kparams->div_ne12);
        const uint32_t i2              = dst_row_div_ne1 - i3 * ne2;

        // Map to source indices (tiling)
        const uint32_t k1 = fastmodulo(i1, ne01, &kparams->div_ne01);
        const uint32_t k2 = fastmodulo(i2, ne02, &kparams->div_ne02);
        const uint32_t k3 = fastmodulo(i3, ne03, &kparams->div_ne03);

        const uint64_t src_addr = src->data + (uint64_t) k1 * nb01 + (uint64_t) k2 * nb02 + (uint64_t) k3 * nb03;
        const uint64_t dst_base = dst->data + (uint64_t) i1 * nb1 + (uint64_t) i2 * nb2 + (uint64_t) i3 * nb3;

        // An extended pointer is a DMA64 DDR address. Casting it to a 32-bit pointer drops the high half,
        // so one source row is staged through this thread's VTCM slice and written back out.
        if (rctx->bounce) {
            uint8_t * scratch = octx->ctx->vtcm_base + (size_t) ith * rctx->scratch_stride;
            if (htp_tensor_is_extended(src)) {
                dma_sync_read(th->dma_q, scratch, src_addr, row_bytes);
            } else {
                memcpy(scratch, (const uint8_t *) (uintptr_t) src_addr, row_bytes);
            }
            for (uint32_t i0 = 0; i0 < nr0; i0++) {
                const uint64_t dst_addr = dst_base + (uint64_t) i0 * ne00 * nb0;
                if (htp_tensor_is_extended(dst)) {
                    dma_sync_write(th->dma_q, dst_addr, scratch, row_bytes);
                } else {
                    memcpy((uint8_t *) (uintptr_t) dst_addr, scratch, row_bytes);
                }
            }
            continue;
        }

        const uint8_t * src_row = (const uint8_t *) (uintptr_t) src_addr;
        uint8_t * dst_row_base  = (uint8_t *) (uintptr_t) dst_base;

        // Tile along dimension 0
        for (uint32_t i0 = 0; i0 < nr0; i0++) {
            uint8_t * dst_ptr = dst_row_base + i0 * ne00 * nb0;
            memcpy(dst_ptr, src_row, row_bytes);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    FARF(HIGH, "repeat %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth, src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         row_start, row_end);
}

int op_repeat(struct htp_ops_context * octx) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;

    size_t type_size;
    switch (src0->type) {
        case HTP_TYPE_F32: type_size = 4; break;
        case HTP_TYPE_F16: type_size = 2; break;
        default:
            FARF(ERROR, "repeat: unsupported type %u\n", src0->type);
            return HTP_STATUS_NO_SUPPORT;
    }

    const int bounce = htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst);

    const struct htp_repeat_kernel_params * kparams =
        (const struct htp_repeat_kernel_params *) octx->kernel_params;

    if (!htp_ops_context_set_n_hvx(octx, kparams->n_hvx)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    const uint32_t total_dst_rows  = dst->ne[1] * dst->ne[2] * dst->ne[3];
    const size_t dst_row_size = dst->ne[0] * type_size;

    uint32_t row_start = 0;
    uint32_t nrows     = total_dst_rows;

    if (octx->ctx->mdev.count > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, type_size, (uint32_t) dst_row_size, &rows_per_chunk);
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_dst_rows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        row_start = range.start;
        nrows     = range.count;
    }

    if (nrows == 0) {
        return HTP_STATUS_OK;
    }

    const uint32_t n_hvx = octx->n_hvx;

    size_t scratch_stride = 0;
    if (bounce) {
        scratch_stride = hex_round_up(src0->ne[0] * type_size, 128);
        if (!octx->ctx->vtcm_base || scratch_stride == 0 || scratch_stride * n_hvx > octx->ctx->vtcm_size) {
            FARF(ERROR, "repeat: extended row %zu does not fit VTCM\n", scratch_stride);
            return HTP_STATUS_NO_SUPPORT;
        }
    }

    struct htp_repeat_context rctx = {
        .octx             = octx,
        .kparams          = kparams,
        .nrows_per_hvx    = fastdiv(nrows + n_hvx - 1, &octx->n_hvx_div),
        .total_dst_rows   = nrows,
        .row_start        = row_start,
        .type_size        = type_size,
        .scratch_stride   = scratch_stride,
        .bounce           = bounce,
    };

    FARF(HIGH, "repeat: (%ux%ux%ux%u) -> (%ux%ux%ux%u) nr=(%u,%u,%u,%u)\n",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         kparams->nr0, kparams->nr1, kparams->nr2, kparams->nr3);

    work_queue_run(octx->ctx->hvx_queue, repeat_job_per_hvx, &rctx, n_hvx);

    return HTP_STATUS_OK;
}

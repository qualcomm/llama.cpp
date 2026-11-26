#include "host-callback.h"

#include <HAP_perf.h>
#include <qurt_memory.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-fence.h"

int htp_host_callback(struct htp_ops_context * octx) {
    struct htp_context * ctx     = octx->ctx;
    const uint32_t       seq     = (uint32_t) octx->op_params[0];
    uint8_t *            control = (uint8_t *) (uintptr_t) octx->src[0]->data;

    // Clean DSP data before the CPU can replace cache slots.
    dma_proxy_flush(ctx->dma_proxy);
    qurt_mem_cache_clean((qurt_addr_t) 0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
    hex_l2fetch_block(ctx, ctx->footprint);
    htp_fence_write(control, seq, octx->status);

    const uint64_t started = HAP_perf_get_qtimer_count();
    uint32_t       ready_seq, status;
    do {
        htp_fence_read(control + HTP_FENCE_SLOT_SIZE, &ready_seq, &status);
        if (ready_seq == seq) {
            break;
        }
        if (HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - started) > 120000000ULL) {
            return HTP_STATUS_INTERNAL_ERR;
        }
        hex_pause();
    } while (1);

    // Drop stale DSP lines before using CPU-written weights and IDs.
    qurt_mem_cache_clean((qurt_addr_t) 0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
    return status;
}

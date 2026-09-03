#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>
#include <qurt_memory.h>

#include <math.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-ops.h"
#include "hvx-utils.h"
#include "htp-tensor.h"

struct htp_copy_context {
    struct htp_ops_context * octx;

    uint32_t          src0_type_size;
    uint32_t          src0_block_size;

    uint32_t          dst_type_size;
    uint32_t          dst_block_size;

    uint32_t          src0_blocks_per_row;
    uint32_t          dst_blocks_per_row;

    uint32_t          mdev_elem_start;
    uint32_t          mdev_nelem;
    uint32_t          elem_per_thread;

    uint32_t          src0_nrows_per_thread;
    uint32_t          mdev_row_start;
    uint32_t          mdev_nrows;

    struct fastdiv_values div_ne01;
    struct fastdiv_values div_ne02_ne01;

    struct fastdiv_values div_ne0;
    struct fastdiv_values div_ne1_ne0;
    struct fastdiv_values div_ne2_ne1_ne0;
    struct fastdiv_values div_ne00;
    struct fastdiv_values div_ne01_ne00;
    struct fastdiv_values div_ne02_ne01_ne00;
};

#define cpy_preamble                              \
    const struct htp_tensor *src0 = octx->src[0]; \
    const struct htp_tensor *dst  = octx->dst;    \
                                                  \
    const uint32_t ne00 = src0->ne[0];            \
    const uint32_t ne01 = src0->ne[1];            \
    const uint32_t ne02 = src0->ne[2];            \
    const uint32_t ne03 = src0->ne[3];            \
                                                  \
    const uint32_t nb00 = src0->nb[0];            \
    const uint32_t nb01 = src0->nb[1];            \
    const uint32_t nb02 = src0->nb[2];            \
    const uint32_t nb03 = src0->nb[3];            \
                                                  \
    const uint32_t  ne0 = dst->ne[0];             \
    const uint32_t  ne1 = dst->ne[1];             \
    const uint32_t  ne2 = dst->ne[2];             \
    const uint32_t  ne3 = dst->ne[3];             \
                                                  \
    const uint32_t  nb0 = dst->nb[0];             \
    const uint32_t  nb1 = dst->nb[1];             \
    const uint32_t  nb2 = dst->nb[2];             \
    const uint32_t  nb3 = dst->nb[3];

#define DEFINE_CPY_SAMESHAPE(NAME, ELEM_TYPE, ELEM_SIZE)                                       \
static void cpy_thread_##NAME##_sameshape(unsigned int nth, unsigned int ith, void * data) {   \
    struct htp_copy_context * ct = (struct htp_copy_context *) data;                           \
    struct htp_ops_context * octx = ct->octx;                                                  \
    cpy_preamble;                                                                              \
    const uint32_t dr  = ct->src0_nrows_per_thread;                                            \
    const uint32_t ir0 = ct->mdev_row_start + dr * ith;                                        \
    const uint32_t ir1 = MIN(ir0 + dr, ct->mdev_row_start + ct->mdev_nrows);                   \
    if (ir0 >= ir1) return;                                                                    \
    const bool contiguous = (nb01 == ne00 * ELEM_SIZE) && (nb1 == nb01) &&                     \
                            (nb02 == ne01 * nb01)      && (nb2 == nb02) &&                     \
                            (nb03 == ne02 * nb02)      && (nb3 == nb03);                       \
    const uint32_t ne02_ne01 = ne02 * ne01;                                                    \
    uint32_t i03 = fastdiv(ir0, &ct->div_ne02_ne01);                                           \
    uint32_t rem = ir0 - i03 * ne02_ne01;                                                      \
    uint32_t i02 = fastdiv(rem, &ct->div_ne01);                                                \
    uint32_t i01 = rem - i02 * ne01;                                                           \
    uint8_t * dst_ptr  = (uint8_t *) dst->data  + i01*nb1  + i02*nb2  + i03*nb3;               \
    uint8_t * src0_ptr = (uint8_t *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;              \
    if (contiguous) {                                                                          \
        hvx_copy_uu(dst_ptr, src0_ptr, (ir1 - ir0) * ne00, ELEM_SIZE);                         \
        return;                                                                                \
    }                                                                                          \
    for (uint32_t r = ir0; r < ir1; r++) {                                                     \
        hex_l2fetch(src0_ptr, ne00 * ELEM_SIZE, nb01, 2);                                      \
        hvx_copy_uu(dst_ptr, src0_ptr, ne00, ELEM_SIZE);                                       \
        dst_ptr  += nb1;                                                                       \
        src0_ptr += nb01;                                                                      \
        if (++i01 == ne01) {                                                                   \
            i01 = 0;                                                                           \
            if (++i02 == ne02) {                                                               \
                i02 = 0;                                                                       \
                i03++;                                                                         \
            }                                                                                  \
            dst_ptr  = (uint8_t *) dst->data  + i02*nb2  + i03*nb3;                            \
            src0_ptr = (uint8_t *) src0->data + i02*nb02 + i03*nb03;                           \
        }                                                                                      \
    }                                                                                          \
}

DEFINE_CPY_SAMESHAPE(f32,  float, 4)
DEFINE_CPY_SAMESHAPE(f16, __fp16, 2)

#define DEFINE_CPY_RESHAPE(NAME, ELEM_TYPE, ELEM_SIZE)                                               \
static void cpy_thread_##NAME##_reshape(unsigned int nth, unsigned int ith, void * data) {           \
    struct htp_copy_context * ct = (struct htp_copy_context *) data;                                 \
    struct htp_ops_context * octx = ct->octx;                                                        \
    cpy_preamble;                                                                                    \
    const uint32_t th_nelem = ct->elem_per_thread;                                                   \
    const uint32_t th_start = ct->mdev_elem_start + ith * th_nelem;                                  \
    const uint32_t th_end   = MIN(th_start + th_nelem, ct->mdev_elem_start + ct->mdev_nelem);        \
    if (th_start >= th_end) return;                                                                  \
                                                                                                     \
    const uint32_t ne01_ne00      = ne01 * ne00;                                                     \
    const uint32_t ne02_ne01_ne00 = ne02 * ne01_ne00;                                                \
    const uint32_t ne1_ne0        = ne1 * ne0;                                                       \
    const uint32_t ne2_ne1_ne0    = ne2 * ne1_ne0;                                                   \
                                                                                                     \
    uint32_t e = th_start;                                                                           \
    uint32_t i13 = fastdiv(e, &ct->div_ne2_ne1_ne0);                                                 \
    uint32_t rem = e - i13 * ne2_ne1_ne0;                                                            \
    uint32_t i12 = fastdiv(rem, &ct->div_ne1_ne0);                                                   \
    uint32_t rem2 = rem - i12 * ne1_ne0;                                                             \
    uint32_t i11 = fastdiv(rem2, &ct->div_ne0);                                                      \
    uint32_t i10 = rem2 - i11 * ne0;                                                                 \
                                                                                                     \
    uint32_t i03 = fastdiv(e, &ct->div_ne02_ne01_ne00);                                              \
    uint32_t rem_s = e - i03 * ne02_ne01_ne00;                                                       \
    uint32_t i02 = fastdiv(rem_s, &ct->div_ne01_ne00);                                               \
    uint32_t rem2_s = rem_s - i02 * ne01_ne00;                                                       \
    uint32_t i01 = fastdiv(rem2_s, &ct->div_ne00);                                                   \
    uint32_t i00 = rem2_s - i01 * ne00;                                                              \
                                                                                                     \
    char * dst_ptr        = (char *)       dst->data  + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3;    \
    const char * src0_ptr = (const char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03;   \
                                                                                                     \
    for (; e < th_end; e++) {                                                                        \
        *((ELEM_TYPE *) dst_ptr) = *((const ELEM_TYPE *) src0_ptr);                                  \
                                                                                                     \
        dst_ptr += nb0;                                                                              \
        if (++i10 == ne0) {                                                                          \
            i10 = 0;                                                                                 \
            if (++i11 == ne1) {                                                                      \
                i11 = 0;                                                                             \
                if (++i12 == ne2) {                                                                  \
                    i12 = 0;                                                                         \
                    i13++;                                                                           \
                }                                                                                    \
            }                                                                                        \
            dst_ptr = (char *) dst->data + i11*nb1 + i12*nb2 + i13*nb3;                              \
        }                                                                                            \
                                                                                                     \
        src0_ptr += nb00;                                                                            \
        if (++i00 == ne00) {                                                                         \
            i00 = 0;                                                                                 \
            if (++i01 == ne01) {                                                                     \
                i01 = 0;                                                                             \
                if (++i02 == ne02) {                                                                 \
                    i02 = 0;                                                                         \
                    i03++;                                                                           \
                }                                                                                    \
            }                                                                                        \
            src0_ptr = (const char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;                   \
        }                                                                                            \
    }                                                                                                \
}

DEFINE_CPY_RESHAPE(f32,  float, 4)
DEFINE_CPY_RESHAPE(f16, __fp16, 2)

static void cpy_thread_f16_f32_sameshape(unsigned int nth, unsigned int ith, void * data) {
    struct htp_copy_context * ct = (struct htp_copy_context *) data;
    struct htp_ops_context * octx = ct->octx;
    cpy_preamble;

    const uint32_t dr  = ct->src0_nrows_per_thread;
    const uint32_t ir0 = ct->mdev_row_start + dr * ith;
    const uint32_t ir1 = MIN(ir0 + dr, ct->mdev_row_start + ct->mdev_nrows);
    if (ir0 >= ir1) return;

    const uint32_t ne02_ne01 = ne02 * ne01;
    uint32_t i03 = fastdiv(ir0, &ct->div_ne02_ne01);
    uint32_t rem = ir0 - i03 * ne02_ne01;
    uint32_t i02 = fastdiv(rem, &ct->div_ne01);
    uint32_t i01 = rem - i02 * ne01;

    uint8_t* dst_ptr  = (uint8_t*) dst->data  + i01*nb1  + i02*nb2  + i03*nb3;
    uint8_t* src0_ptr = (uint8_t*) src0->data + i01*nb01 + i02*nb02 + i03*nb03;

    for (uint32_t r = ir0; r < ir1; r++) {
        hex_l2fetch(src0_ptr, ne00 * sizeof(float), nb01, 2);
        hvx_copy_f16_f32_uu(dst_ptr, src0_ptr, ne00);
        dst_ptr  += nb1;
        src0_ptr += nb01;
        if (++i01 == ne01) {
            i01 = 0;
            if (++i02 == ne02) {
                i02 = 0;
                i03++;
            }
            dst_ptr  = (uint8_t*) dst->data  + i02*nb2  + i03*nb3;
            src0_ptr = (uint8_t*) src0->data + i02*nb02 + i03*nb03;
        }
    }
}

static void cpy_thread_f32_f16_sameshape(unsigned int nth, unsigned int ith, void * data) {
    struct htp_copy_context * ct = (struct htp_copy_context *) data;
    struct htp_ops_context * octx = ct->octx;
    cpy_preamble;

    const uint32_t dr  = ct->src0_nrows_per_thread;
    const uint32_t ir0 = ct->mdev_row_start + dr * ith;
    const uint32_t ir1 = MIN(ir0 + dr, ct->mdev_row_start + ct->mdev_nrows);
    if (ir0 >= ir1) return;

    const uint32_t ne02_ne01 = ne02 * ne01;
    uint32_t i03 = fastdiv(ir0, &ct->div_ne02_ne01);
    uint32_t rem = ir0 - i03 * ne02_ne01;
    uint32_t i02 = fastdiv(rem, &ct->div_ne01);
    uint32_t i01 = rem - i02 * ne01;

    uint8_t* dst_ptr  = (uint8_t*) dst->data  + i01*nb1  + i02*nb2  + i03*nb3;
    uint8_t* src0_ptr = (uint8_t*) src0->data + i01*nb01 + i02*nb02 + i03*nb03;

    for (uint32_t r = ir0; r < ir1; r++) {
        hex_l2fetch(src0_ptr, ne00 * sizeof(__fp16), nb01, 2);
        hvx_copy_f32_f16_uu(dst_ptr, src0_ptr, ne00);
        dst_ptr  += nb1;
        src0_ptr += nb01;
        if (++i01 == ne01) {
            i01 = 0;
            if (++i02 == ne02) {
                i02 = 0;
                i03++;
            }
            dst_ptr  = (uint8_t*) dst->data  + i02*nb2  + i03*nb3;
            src0_ptr = (uint8_t*) src0->data + i02*nb02 + i03*nb03;
        }
    }
}

static inline void cpy_dma_sametype_sameshape(
    struct htp_ops_context * octx,
    const struct htp_tensor * dst,
    const struct htp_tensor * src0,
    uint32_t elem_size,
    uint32_t ne00, uint32_t ne01, uint32_t ne02, uint32_t ne03,
    uint32_t nb01, uint32_t nb02, uint32_t nb03,
    uint32_t  nb1, uint32_t  nb2, uint32_t nb3
) {
    const bool contiguous_outer =
        (ne02 == 1 || (nb02 == ne01 * nb01 && nb2 == ne01 * nb1)) &&
        (ne03 == 1 || (nb03 == ne02 * nb02 && nb3 == ne02 * nb2));

    dma_queue * q = octx->ctx->dma[0];

    if (contiguous_outer) {
        if (!dma_queue_push(q, dma_make_ptr((void *) dst->data, (const void *) src0->data), nb1, nb01, ne00 * elem_size, ne01 * ne02 * ne03)) {
            dma_queue_flush(q);
            dma_queue_push(q, dma_make_ptr((void *) dst->data, (const void *) src0->data), nb1, nb01, ne00 * elem_size, ne01 * ne02 * ne03);
        }
        dma_queue_flush(q);
        return;
    }

    for (uint32_t i03 = 0; i03 < ne03; i03++) {
        for (uint32_t i02 = 0; i02 < ne02; i02++) {
            uint8_t * dst_ptr  = (uint8_t *) dst->data  + i02 * nb2  + i03 * nb3;
            uint8_t * src0_ptr = (uint8_t *) src0->data + i02 * nb02 + i03 * nb03;

            if (!dma_queue_push(q, dma_make_ptr(dst_ptr, src0_ptr), nb1, nb01, ne00 * elem_size, ne01)) {
                dma_queue_flush(q);
                dma_queue_push(q, dma_make_ptr(dst_ptr, src0_ptr), nb1, nb01, ne00 * elem_size, ne01);
            }
        }
    }

    dma_queue_flush(q);
}

int op_cpy(struct htp_ops_context * octx) {
    cpy_preamble;

    struct htp_copy_context ct;
    ct.octx = octx;

    switch (src0->type) {
    case HTP_TYPE_F32: ct.src0_type_size = 4; ct.src0_block_size = 1; ct.src0_blocks_per_row = ne00 / 1; break;
    case HTP_TYPE_F16: ct.src0_type_size = 2; ct.src0_block_size = 1; ct.src0_blocks_per_row = ne00 / 1; break;
    default:
        return HTP_STATUS_NO_SUPPORT;
    }

    switch (dst->type) {
    case HTP_TYPE_F32: ct.dst_type_size = 4; ct.dst_block_size = 1; ct.dst_blocks_per_row = ne0 / 1; break;
    case HTP_TYPE_F16: ct.dst_type_size = 2; ct.dst_block_size = 1; ct.dst_blocks_per_row = ne0 / 1; break;
    default:
        return HTP_STATUS_NO_SUPPORT;
    }

    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) {
        return HTP_STATUS_OK;
    }

    const bool sametype   = (src0->type == dst->type);
    const bool transposed = (nb00 > nb01) || (nb0 > nb1) ||
                            (nb00 != ct.src0_type_size) || (nb0 != ct.dst_type_size) ||
                            (nb01 < ne00 * ct.src0_type_size) || (nb1 < ne0 * ct.dst_type_size);
    const bool sameshape  = !transposed && (ne00 == ne0 && ne01 == ne1 && ne02 == ne2 && ne03 == ne3);

    const uint32_t n_threads = octx->n_threads;
    bool use_dma = false;

    const bool dst_is_contiguous = htp_tensor_is_contiguous(dst, ct.dst_type_size);

    if (sameshape) {
        const uint32_t total_rows = ne01 * ne02 * ne03;
        const uint32_t row_size   = ne00 * ct.dst_type_size;

        ct.div_ne01      = init_fastdiv_values(ne01);
        ct.div_ne02_ne01 = init_fastdiv_values(ne02 * ne01);

        uint32_t mdev_row_start, mdev_nrows;
        if (octx->mdev_count > 1) {
            const uint32_t rows_per_chunk = (row_size > 0) ? (HEX_L2_LINE_SIZE / hex_gcd_u32(row_size, HEX_L2_LINE_SIZE)) : 1;
            const uint32_t total_chunks   = dst_is_contiguous ? (total_rows / rows_per_chunk) : 0;
            if (total_chunks < octx->mdev_count) {
                mdev_row_start = (octx->mdev_idx == 0) ? 0 : total_rows;
                mdev_nrows     = (octx->mdev_idx == 0) ? total_rows : 0;
            } else {
                uint32_t chunks_per_mdev = fastdiv(total_chunks + octx->mdev_count - 1, &octx->mdev_count_div);
                mdev_row_start = MIN(octx->mdev_idx * chunks_per_mdev * rows_per_chunk, total_rows);
                if (octx->mdev_idx == octx->mdev_count - 1) {
                    mdev_nrows = total_rows - mdev_row_start;
                } else {
                    mdev_nrows = MIN(chunks_per_mdev * rows_per_chunk, total_rows - mdev_row_start);
                }
            }
        } else {
            mdev_row_start = 0;
            mdev_nrows     = total_rows;
        }

        if (mdev_nrows == 0) {
            return HTP_STATUS_OK;
        }

        ct.mdev_row_start = mdev_row_start;
        ct.mdev_nrows     = mdev_nrows;
        ct.src0_nrows_per_thread = fastdiv(mdev_nrows + n_threads - 1, &octx->n_threads_div);

        if (sametype && octx->mdev_count <= 1) {
            use_dma = true;
            cpy_dma_sametype_sameshape(octx, dst, src0, ct.src0_type_size, ne00, ne01, ne02, ne03, nb01, nb02, nb03, nb1, nb2, nb3);
        } else {
            work_queue_func_t copy_fun = NULL;
            if (sametype) {
                copy_fun = (src0->type == HTP_TYPE_F32) ? cpy_thread_f32_sameshape : cpy_thread_f16_sameshape;
            } else if (dst->type == HTP_TYPE_F16 && src0->type == HTP_TYPE_F32) {
                copy_fun = cpy_thread_f16_f32_sameshape;
            } else if (dst->type == HTP_TYPE_F32 && src0->type == HTP_TYPE_F16) {
                copy_fun = cpy_thread_f32_f16_sameshape;
            } else {
                return HTP_STATUS_NO_SUPPORT;
            }
            work_queue_run(octx->ctx->work_queue, copy_fun, &ct, n_threads);
        }
    } else if (sametype) {
        const uint32_t total_elems = ne0 * ne1 * ne2 * ne3;
        const uint32_t total_bytes = total_elems * ct.dst_type_size;
        const uint32_t n_lines     = total_bytes >> 7;
        const uint32_t elems_per_line = (ct.dst_type_size == 4) ? 32 : 64;

        ct.div_ne0            = init_fastdiv_values(ne0);
        ct.div_ne1_ne0        = init_fastdiv_values(ne1 * ne0);
        ct.div_ne2_ne1_ne0    = init_fastdiv_values(ne2 * ne1 * ne0);
        ct.div_ne00           = init_fastdiv_values(ne00);
        ct.div_ne01_ne00      = init_fastdiv_values(ne01 * ne00);
        ct.div_ne02_ne01_ne00 = init_fastdiv_values(ne02 * ne01 * ne00);

        uint32_t mdev_elem_start, mdev_nelem;
        if (octx->mdev_count > 1) {
            const uint32_t aligned_lines = dst_is_contiguous ? n_lines : 0;
            if (aligned_lines < octx->mdev_count) {
                mdev_elem_start = (octx->mdev_idx == 0) ? 0 : total_elems;
                mdev_nelem      = (octx->mdev_idx == 0) ? total_elems : 0;
            } else {
                uint32_t lines_per_mdev = fastdiv(aligned_lines + octx->mdev_count - 1, &octx->mdev_count_div);
                mdev_elem_start = MIN(octx->mdev_idx * lines_per_mdev * elems_per_line, total_elems);
                if (octx->mdev_idx == octx->mdev_count - 1) {
                    mdev_nelem = total_elems - mdev_elem_start;
                } else {
                    mdev_nelem = MIN(lines_per_mdev * elems_per_line, total_elems - mdev_elem_start);
                }
            }
        } else {
            mdev_elem_start = 0;
            mdev_nelem      = total_elems;
        }

        if (mdev_nelem == 0) {
            return HTP_STATUS_OK;
        }

        ct.mdev_elem_start = mdev_elem_start;
        ct.mdev_nelem      = mdev_nelem;
        ct.elem_per_thread = fastdiv(mdev_nelem + n_threads - 1, &octx->n_threads_div);

        work_queue_func_t copy_fun = (src0->type == HTP_TYPE_F32) ? cpy_thread_f32_reshape : cpy_thread_f16_reshape;
        work_queue_run(octx->ctx->work_queue, copy_fun, &ct, n_threads);
    } else {
        return HTP_STATUS_NO_SUPPORT;
    }

    const struct htp_tensor *sync = octx->src[1];
    if (sync && (sync->flags & HTP_TENSOR_FENCE)) {
        if (!use_dma) {
            // htp_tensor_flush_all(octx->ctx, octx->dsts, 1);
            qurt_mem_cache_clean((qurt_addr_t) 0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
        }

        atomic_uint * sync_fence = (atomic_uint *) sync->data;
        const uint32_t seq = (uint32_t) octx->op_params[0];

        atomic_store(&sync_fence[0], seq);
        asm volatile ("syncht" : : : "memory");
        Q6_dccleaninva_A((void *) sync_fence);

        FARF(HIGH, "ggml-hex: sync-release : fence %p seq %u\n", sync_fence, seq);
    }

    return HTP_STATUS_OK;
}

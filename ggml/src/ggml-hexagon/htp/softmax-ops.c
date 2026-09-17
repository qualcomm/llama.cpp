#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>

#include <math.h>
#include <string.h>

#include "dma-queue.h"
#include "work-queue.h"
#include "hvx-utils.h"
#include "hex-fastdiv.h"
#include "hex-common.h"
#include "hex-profile.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "htp/softmax-ops.h"

struct htp_softmax_context {
    struct htp_ops_context * octx;
    const struct htp_softmax_kernel_params * kparams;

    void *   compute;

    dma_addr_t data_src0;
    dma_addr_t data_src1;
    dma_addr_t data_dst;

    uint8_t * vtcm_src0;
    uint8_t * vtcm_src1;
    uint8_t * vtcm_dst;

    uint32_t  vtcm_src0_size_per_thread;
    uint32_t  vtcm_src1_size_per_thread;
    uint32_t  vtcm_dst_size_per_thread;

    uint32_t  src0_spad_half_size;
    uint32_t  src1_spad_half_size;
    uint32_t  dst_spad_half_size;

    uint32_t  src0_row_size_aligned;
    uint32_t  src1_row_size_aligned;
    uint32_t  dst_row_size_aligned;

    bool     use_f16;
    bool     use_src1;
    bool     opt_path;

    uint32_t n_head;
    uint32_t n_head_log2;

    float    scale;
    float    max_bias;
    float    m0;
    float    m1;

    struct fastdiv_values div_ne01;
    struct fastdiv_values div_ne02;
    struct fastdiv_values div_ne12;
    struct fastdiv_values div_ne13;

    uint32_t src0_nrows_per_thread;
    uint32_t row_start;
    uint32_t nrows;
};

typedef void (*softmax_compute_fn_t)(
    void * restrict dst,
    const void * restrict src0,
    const void * restrict mask,
    uint32_t ne00,
    float scale,
    float slope
);

static void apply_mask(float * restrict wp0,
                       const float * restrict mp_f32,
                       const __fp16 * restrict mp_f16,
                       uint32_t ne00,
                       float slope,
                       bool use_f16) {
    if (use_f16) {
        if (!mp_f16) return;
        for (uint32_t i = 0; i < ne00; ++i) {
            wp0[i] += slope * (float) mp_f16[i];
        }
    } else {
        if (!mp_f32) return;
        for (uint32_t i = 0; i < ne00; ++i) {
            wp0[i] += slope * mp_f32[i];
        }
    }
}

static void hvx_fast_softmax_prep_f32(const uint8_t * restrict src,
                                      uint8_t * restrict dst,
                                      const int num_elems,
                                      float     scale,
                                      const uint8_t * restrict mask,
                                      float slope) {
    const uint8_t * restrict src_curr  = src;
    uint8_t * restrict dst_curr        = dst;
    const uint8_t * restrict mask_curr = mask;

    HVX_Vector scale_vec = hvx_vec_splat_f32(scale);
    HVX_Vector slope_vec = hvx_vec_splat_f32(slope);

    int step_of_1 = num_elems >> 5;

    #pragma unroll(4)
    for (int i = 0; i < step_of_1; i++) {
        HVX_Vector v1 = *(const HVX_Vector *) src_curr;
        HVX_Vector v3 = *(const HVX_Vector *) mask_curr;

        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);
        HVX_Vector v4 = Q6_Vqf32_vmpy_VsfVsf(v3, slope_vec);
        HVX_Vector v5 = Q6_Vqf32_vadd_Vqf32Vqf32(v2, v4);

        *(HVX_Vector *) dst_curr = Q6_Vsf_equals_Vqf32(v5);

        src_curr  += VLEN;
        dst_curr  += VLEN;
        mask_curr += VLEN;
    }
}

static void hvx_fast_softmax_f32(const uint8_t * restrict src, uint8_t * restrict dst, const int num_elems) {
    const HVX_Vector * restrict v_src = (const HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;

    HVX_Vector sum_vec = Q6_V_vsplat_R(0x00000000);
    HVX_Vector max_vec = hvx_vec_splat_f32(((const float *) src)[0]);
    HVX_Vector zero_v  = Q6_V_vzero();
    HVX_Vector one_v   = hvx_vec_splat_f32(1.0f);

    int step_of_1 = num_elems >> 5;

    #pragma unroll(4)
    for (int i = 0; i < step_of_1; i++) {
        HVX_Vector v1 = v_src[i];
        max_vec       = Q6_Vsf_vmax_VsfVsf(max_vec, v1);
    }

    max_vec = hvx_vec_reduce_max_f32(max_vec);

    #pragma unroll(2)
    for (int i = 0; i < step_of_1; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector v2 = Q6_Vqf32_vsub_VsfVsf(v1, max_vec);

        HVX_Vector v3 = hvx_vec_exp_f32(Q6_Vsf_equals_Vqf32(v2));

        sum_vec = Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(sum_vec), v3);

        v_dst[i] = v3;
    }

    sum_vec = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(sum_vec));

    HVX_VectorPred pos_sum   = Q6_Q_vcmp_gt_VwVw(sum_vec, zero_v);
    HVX_Vector     v4        = hvx_vec_inverse_f32(sum_vec);
    HVX_Vector     scale_vec = Q6_V_vmux_QVV(pos_sum, v4, one_v);

    #pragma unroll(4)
    for (int i = 0; i < step_of_1; i++) {
        HVX_Vector v1 = v_dst[i];
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);
        v_dst[i]      = Q6_Vsf_equals_Vqf32(v2);
    }
}

static float hvx_softmax_f32(const uint8_t * restrict src, uint8_t * restrict dst, uint8_t * restrict spad, const int num_elems, const float max) {
    hvx_sub_scalar_f32(spad, src, max, num_elems);
    hvx_exp_f32(dst, spad, num_elems, false);
    return hvx_reduce_sum_f32(dst, num_elems);
}

static void compute_fast_softmax_f32_nomask(
    void * restrict dst,
    const void * restrict src0,
    const void * restrict mask,
    uint32_t ne00,
    float scale,
    float slope
) {
    (void) mask;
    (void) slope;
    hvx_scale_f32((uint8_t *) dst, (const uint8_t *) src0, ne00, scale);
    hvx_fast_softmax_f32((const uint8_t *) dst, (uint8_t *) dst, ne00);
}

static void compute_fast_softmax_f32_mask_f32(
    void * restrict dst,
    const void * restrict src0,
    const void * restrict mask,
    uint32_t ne00,
    float scale,
    float slope
) {
    hvx_fast_softmax_prep_f32((const uint8_t *) src0, (uint8_t *) dst, ne00, scale, (const uint8_t *) mask, slope);
    hvx_fast_softmax_f32((const uint8_t *) dst, (uint8_t *) dst, ne00);
}

static void compute_fast_softmax_f32_mask_f16(
    void * restrict dst,
    const void * restrict src0,
    const void * restrict mask,
    uint32_t ne00,
    float scale,
    float slope
) {
    hvx_scale_f32((uint8_t *) dst, (const uint8_t *) src0, ne00, scale);
    apply_mask((float *) dst, NULL, (const __fp16 *) mask, ne00, slope, true);
    hvx_fast_softmax_f32((const uint8_t *) dst, (uint8_t *) dst, ne00);
}

static void compute_softmax_f32_fallback(
    void * restrict dst,
    const void * restrict src0,
    const void * restrict mask,
    uint32_t ne00,
    float scale,
    float slope
) {
    hvx_scale_f32((uint8_t *) dst, (const uint8_t *) src0, ne00, scale);
    if (mask) {
        apply_mask((float *) dst, (const float *) mask, NULL, ne00, slope, false);
    }
    float max = hvx_reduce_max_f32((const uint8_t *) dst, ne00);
    float sum = hvx_softmax_f32((const uint8_t *) dst, (uint8_t *) dst, (uint8_t *) dst, ne00, max);
    sum = sum > 0.0f ? (1.0f / sum) : 1.0f;
    hvx_scale_f32((uint8_t *) dst, (const uint8_t *) dst, ne00, sum);
}

static void compute_softmax_f32_fallback_f16(
    void * restrict dst,
    const void * restrict src0,
    const void * restrict mask,
    uint32_t ne00,
    float scale,
    float slope
) {
    hvx_scale_f32((uint8_t *) dst, (const uint8_t *) src0, ne00, scale);
    if (mask) {
        apply_mask((float *) dst, NULL, (const __fp16 *) mask, ne00, slope, true);
    }
    float max = hvx_reduce_max_f32((const uint8_t *) dst, ne00);
    float sum = hvx_softmax_f32((const uint8_t *) dst, (uint8_t *) dst, (uint8_t *) dst, ne00, max);
    sum = sum > 0.0f ? (1.0f / sum) : 1.0f;
    hvx_scale_f32((uint8_t *) dst, (const uint8_t *) dst, ne00, sum);
}

static void softmax_thread_dma(unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    const struct htp_softmax_context * smctx = (const struct htp_softmax_context *) data;
    struct htp_ops_context * octx = smctx->octx;
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    const uint32_t src0_nrows            = smctx->nrows;
    const uint32_t src0_nrows_per_thread = smctx->src0_nrows_per_thread;

    const uint32_t src0_start_row = smctx->row_start + src0_nrows_per_thread * ith;
    const uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, smctx->row_start + src0_nrows);

    if (src0_start_row >= src0_end_row) {
        return;
    }

    const dma_addr_t data_src0 = smctx->data_src0;
    const dma_addr_t data_dst  = smctx->data_dst;

    const size_t src0_row_size = src0->ne[0] * sizeof(float);
    const size_t dst_row_size  = src0->ne[0] * sizeof(float);

    uint8_t * src0_vtcm_base = smctx->vtcm_src0 + (ith * smctx->vtcm_src0_size_per_thread);
    uint8_t * dst_vtcm_base  = smctx->vtcm_dst  + (ith * smctx->vtcm_dst_size_per_thread);

    const size_t src0_vtcm_half = smctx->src0_spad_half_size;
    const size_t dst_vtcm_half  = smctx->dst_spad_half_size;

    dma_queue * dma_q = octx->ctx->dma[ith];

    for (uint32_t r = src0_start_row, idx = 0; r < src0_end_row && idx < 2; r++, idx++) {
        dma_addr_t cur_dst  = data_dst  + (dma_addr_t) r * dst_row_size;
        dma_addr_t cur_src0 = data_src0 + (dma_addr_t) r * src0_row_size;
        void * d_spad = dst_vtcm_base  + idx * dst_vtcm_half;
        void * s_spad = src0_vtcm_base + idx * src0_vtcm_half;

        dma_queue_push(dma_q, dma_make_data(cur_dst, d_spad),
                       dst_row_size, smctx->dst_row_size_aligned, dst_row_size, 0);
        dma_queue_push(dma_q, dma_make_data(s_spad, cur_src0),
                       smctx->src0_row_size_aligned, src0_row_size, src0_row_size, 1);
    }

    softmax_compute_fn_t compute = (softmax_compute_fn_t) smctx->compute;
    const uint32_t ne00 = src0->ne[0];
    const uint32_t ne01 = src0->ne[1];
    const uint32_t ne02 = src0->ne[2];
    const struct fastdiv_values * div_ne01 = &smctx->div_ne01;
    const struct fastdiv_values * div_ne02 = &smctx->div_ne02;

    uint32_t prev_i2 = (uint32_t)-1;
    float slope = 1.0f;

    for (uint32_t r = src0_start_row; r < src0_end_row; ++r) {
        void * d_spad = (void *) (uintptr_t) dma_queue_pop(dma_q).src;
        void * s_spad = (void *) (uintptr_t) dma_queue_pop(dma_q).dst;

        if (smctx->max_bias > 0.0f) {
            uint32_t r_div_ne01 = fastdiv(r, div_ne01);
            uint32_t i2 = fastmodulo(r_div_ne01, ne02, div_ne02);
            if (i2 != prev_i2) {
                slope = (i2 < smctx->n_head_log2) ?
                    powf(smctx->m0, (float)(i2 + 1)) :
                    powf(smctx->m1, (float)(2 * (i2 - smctx->n_head_log2) + 1));
                prev_i2 = i2;
            }
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, r);
        compute(d_spad, s_spad, NULL, ne00, smctx->scale, slope);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, r);

        dma_addr_t cur_dst = data_dst + (dma_addr_t) r * dst_row_size;
        dma_queue_push(dma_q, dma_make_data(cur_dst, d_spad),
                       dst_row_size, smctx->dst_row_size_aligned, dst_row_size, 1);

        const uint32_t next_r = r + 2;
        if (next_r < src0_end_row) {
            dma_addr_t next_src0 = data_src0 + (dma_addr_t) next_r * src0_row_size;
            dma_queue_push(dma_q, dma_make_data(s_spad, next_src0),
                           smctx->src0_row_size_aligned, src0_row_size, src0_row_size, 1);
        }
    }

    dma_queue_flush(dma_q);
}

static void softmax_thread_mask_dma(unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    const struct htp_softmax_context * smctx = (const struct htp_softmax_context *) data;
    struct htp_ops_context * octx = smctx->octx;
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    const uint32_t src0_nrows            = smctx->nrows;
    const uint32_t src0_nrows_per_thread = smctx->src0_nrows_per_thread;

    const uint32_t src0_start_row = smctx->row_start + src0_nrows_per_thread * ith;
    const uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, smctx->row_start + src0_nrows);

    if (src0_start_row >= src0_end_row) {
        return;
    }

    const dma_addr_t data_src0 = smctx->data_src0;
    const dma_addr_t data_src1 = smctx->data_src1;
    const dma_addr_t data_dst  = smctx->data_dst;

    const size_t src0_row_size = src0->ne[0] * sizeof(float);
    const size_t dst_row_size  = src0->ne[0] * sizeof(float);
    const size_t mask_row_size = smctx->use_f16 ? (src1->ne[0] * sizeof(__fp16)) : (src1->ne[0] * sizeof(float));

    uint8_t * src0_vtcm_base = smctx->vtcm_src0 + (ith * smctx->vtcm_src0_size_per_thread);
    uint8_t * src1_vtcm_base = smctx->vtcm_src1 + (ith * smctx->vtcm_src1_size_per_thread);
    uint8_t * dst_vtcm_base  = smctx->vtcm_dst  + (ith * smctx->vtcm_dst_size_per_thread);

    const size_t src0_vtcm_half = smctx->src0_spad_half_size;
    const size_t src1_vtcm_half = smctx->src1_spad_half_size;
    const size_t dst_vtcm_half  = smctx->dst_spad_half_size;

    const uint32_t nb11 = src1->nb[1];
    const uint32_t nb12 = src1->nb[2];
    const uint32_t nb13 = src1->nb[3];

    const uint32_t ne00 = src0->ne[0];
    const uint32_t ne01 = src0->ne[1];
    const uint32_t ne02 = src0->ne[2];
    const uint32_t ne03 = src0->ne[3];
    const uint32_t ne12 = src1->ne[2];
    const uint32_t ne13 = src1->ne[3];

    const struct fastdiv_values * div_ne01 = &smctx->div_ne01;
    const struct fastdiv_values * div_ne02 = &smctx->div_ne02;
    const struct fastdiv_values * div_ne12 = &smctx->div_ne12;
    const struct fastdiv_values * div_ne13 = &smctx->div_ne13;

    dma_queue * dma_q = octx->ctx->dma[ith];

    for (uint32_t r = src0_start_row, idx = 0; r < src0_end_row && idx < 2; r++, idx++) {
        dma_addr_t cur_dst  = data_dst  + (dma_addr_t) r * dst_row_size;
        dma_addr_t cur_src0 = data_src0 + (dma_addr_t) r * src0_row_size;

        uint32_t i1 = fastmodulo(r, ne01, div_ne01);
        uint32_t r_div_ne01 = fastdiv(r, div_ne01);
        uint32_t i2 = fastmodulo(r_div_ne01, ne02, div_ne02);
        uint32_t i3 = fastdiv(r_div_ne01, div_ne02);
        uint32_t i12 = (ne12 == ne02) ? i2 : fastmodulo(i2, ne12, div_ne12);
        uint32_t i13 = (ne13 == ne03) ? i3 : fastmodulo(i3, ne13, div_ne13);
        dma_addr_t cur_src1 = data_src1 + (dma_addr_t) i1 * nb11 + (dma_addr_t) i12 * nb12 + (dma_addr_t) i13 * nb13;

        void * d_spad = dst_vtcm_base  + idx * dst_vtcm_half;
        void * s_spad = src0_vtcm_base + idx * src0_vtcm_half;
        void * m_spad = src1_vtcm_base + idx * src1_vtcm_half;

        dma_queue_push(dma_q, dma_make_data(cur_dst, d_spad),
                       dst_row_size, smctx->dst_row_size_aligned, dst_row_size, 0);
        dma_queue_push(dma_q, dma_make_data(s_spad, cur_src0),
                       smctx->src0_row_size_aligned, src0_row_size, src0_row_size, 1);
        dma_queue_push(dma_q, dma_make_data(m_spad, cur_src1),
                       smctx->src1_row_size_aligned, mask_row_size, mask_row_size, 1);
    }

    softmax_compute_fn_t compute = (softmax_compute_fn_t) smctx->compute;
    uint32_t prev_i2 = (uint32_t)-1;
    float slope = 1.0f;

    for (uint32_t r = src0_start_row; r < src0_end_row; ++r) {
        void * d_spad = (void *) (uintptr_t) dma_queue_pop(dma_q).src;
        void * s_spad = (void *) (uintptr_t) dma_queue_pop(dma_q).dst;
        void * m_spad = (void *) (uintptr_t) dma_queue_pop(dma_q).dst;

        uint32_t r_div_ne01 = fastdiv(r, div_ne01);
        uint32_t i2 = fastmodulo(r_div_ne01, ne02, div_ne02);
        if (i2 != prev_i2) {
            slope = (smctx->max_bias > 0.0f) ?
                ((i2 < smctx->n_head_log2) ? powf(smctx->m0, (float)(i2 + 1)) : powf(smctx->m1, (float)(2 * (i2 - smctx->n_head_log2) + 1))) :
                1.0f;
            prev_i2 = i2;
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, r);
        compute(d_spad, s_spad, m_spad, ne00, smctx->scale, slope);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, r);

        dma_addr_t cur_dst = data_dst + (dma_addr_t) r * dst_row_size;
        dma_queue_push(dma_q, dma_make_data(cur_dst, d_spad),
                       dst_row_size, smctx->dst_row_size_aligned, dst_row_size, 1);

        const uint32_t next_r = r + 2;
        if (next_r < src0_end_row) {
            dma_addr_t next_src0 = data_src0 + (dma_addr_t) next_r * src0_row_size;

            uint32_t ni1 = fastmodulo(next_r, ne01, div_ne01);
            uint32_t nr_div_ne01 = fastdiv(next_r, div_ne01);
            uint32_t ni2 = fastmodulo(nr_div_ne01, ne02, div_ne02);
            uint32_t ni3 = fastdiv(nr_div_ne01, div_ne02);
            uint32_t ni12 = (ne12 == ne02) ? ni2 : fastmodulo(ni2, ne12, div_ne12);
            uint32_t ni13 = (ne13 == ne03) ? ni3 : fastmodulo(ni3, ne13, div_ne13);
            dma_addr_t next_src1 = data_src1 + (dma_addr_t) ni1 * nb11 + (dma_addr_t) ni12 * nb12 + (dma_addr_t) ni13 * nb13;

            dma_queue_push(dma_q, dma_make_data(s_spad, next_src0),
                           smctx->src0_row_size_aligned, src0_row_size, src0_row_size, 1);
            dma_queue_push(dma_q, dma_make_data(m_spad, next_src1),
                           smctx->src1_row_size_aligned, mask_row_size, mask_row_size, 1);
        }
    }

    dma_queue_flush(dma_q);
}

static void init_softmax_ctx(struct htp_softmax_context * smctx, struct htp_ops_context * octx) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];

    memset(smctx, 0, sizeof(struct htp_softmax_context));

    const struct htp_softmax_kernel_params * kparams =
        (const struct htp_softmax_kernel_params *) octx->kernel_params;

    if (kparams && kparams->n_threads > 0) {
        smctx->kparams     = kparams;
        smctx->scale       = kparams->scale;
        smctx->max_bias    = kparams->max_bias;
        smctx->m0          = kparams->m0;
        smctx->m1          = kparams->m1;
        smctx->n_head      = kparams->n_head;
        smctx->n_head_log2 = kparams->n_head_log2;
        smctx->use_src1    = kparams->use_src1 != 0;
        smctx->use_f16     = kparams->use_f16 != 0;
        smctx->opt_path    = kparams->opt_path != 0;
        smctx->div_ne01    = kparams->div_ne01;
        smctx->div_ne02    = kparams->div_ne02;
        smctx->div_ne12    = kparams->div_ne12;
        smctx->div_ne13    = kparams->div_ne13;
    } else {
        memcpy(&smctx->scale,    (const float *) octx->op_params,     sizeof(float));
        memcpy(&smctx->max_bias, (const float *) octx->op_params + 1, sizeof(float));

        smctx->n_head      = src0->ne[2];
        smctx->n_head_log2 = 1u << (uint32_t) floor(log2(smctx->n_head));

        if (smctx->max_bias > 0.0f && smctx->n_head_log2 > 0) {
            smctx->m0 = powf(2.0f, -(smctx->max_bias) / smctx->n_head_log2);
            smctx->m1 = powf(2.0f, -(smctx->max_bias / 2.0f) / smctx->n_head_log2);
        } else {
            smctx->m0 = 1.0f;
            smctx->m1 = 1.0f;
        }

        smctx->use_src1 = (src1 != NULL);
        smctx->use_f16  = (src1 != NULL) && (src1->type == HTP_TYPE_F16);
        smctx->opt_path = ((src0->ne[0] % 32) == 0);

        if (src0->ne[1] > 0) smctx->div_ne01 = init_fastdiv_values(src0->ne[1]);
        if (src0->ne[2] > 0) smctx->div_ne02 = init_fastdiv_values(src0->ne[2]);

        const uint32_t ne12 = src1 ? src1->ne[2] : 1;
        const uint32_t ne13 = src1 ? src1->ne[3] : 1;

        if (ne12 > 0) smctx->div_ne12 = init_fastdiv_values(ne12);
        if (ne13 > 0) smctx->div_ne13 = init_fastdiv_values(ne13);
    }

    smctx->octx = octx;
}

static int execute_op_softmax_f32(struct htp_ops_context * octx) {
    int err = HTP_STATUS_OK;

    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;

    struct htp_softmax_context smctx;
    const char * op_type = "softmax-f32";

    init_softmax_ctx(&smctx, octx);

    const uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    const size_t elem_size    = sizeof(float);
    const size_t dst_row_size = dst->nb[1];

    uint32_t row_start = 0;
    uint32_t nrows     = src0_nrows;

    if (octx->ctx->mdev.count > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, (uint32_t) elem_size, (uint32_t) dst_row_size, &rows_per_chunk);
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            src0_nrows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        row_start = range.start;
        nrows     = range.count;
    }

    if (nrows == 0) {
        return HTP_STATUS_OK;
    }

    const uint32_t n_threads = MIN(octx->n_threads, nrows);
    smctx.src0_nrows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div);
    smctx.row_start             = row_start;
    smctx.nrows                 = nrows;

    struct htp_softmax_vtcm_layout vtcm_layout;
    htp_softmax_vtcm_layout_build(&vtcm_layout, src0->ne[0], src1 ? src1->ne[0] : 1,
                                 smctx.use_src1, smctx.use_f16, n_threads);

    if (octx->ctx->vtcm_size < vtcm_layout.total_bytes) {
        FARF(ERROR, "%s : current VTCM reservation %zu is too small, needed %zu\n",
             op_type, octx->ctx->vtcm_size, vtcm_layout.total_bytes);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * vtcm_base = octx->ctx->vtcm_base;
    smctx.vtcm_src0 = vtcm_base + vtcm_layout.off_src0;
    smctx.vtcm_dst  = vtcm_base + vtcm_layout.off_dst;
    smctx.vtcm_src1 = smctx.use_src1 ? (vtcm_base + vtcm_layout.off_src1) : NULL;

    smctx.vtcm_src0_size_per_thread = vtcm_layout.src0_bytes_per_thread;
    smctx.vtcm_dst_size_per_thread  = vtcm_layout.dst_bytes_per_thread;
    smctx.vtcm_src1_size_per_thread = vtcm_layout.src1_bytes_per_thread;

    smctx.src0_spad_half_size = vtcm_layout.src0_spad_half_size;
    smctx.dst_spad_half_size  = vtcm_layout.dst_spad_half_size;
    smctx.src1_spad_half_size = vtcm_layout.src1_spad_half_size;

    smctx.src0_row_size_aligned = vtcm_layout.src0_spad_half_size;
    smctx.dst_row_size_aligned  = vtcm_layout.dst_spad_half_size;
    smctx.src1_row_size_aligned = vtcm_layout.src1_spad_half_size;

    smctx.data_src0 = src0->data;
    smctx.data_dst  = dst->data;
    smctx.data_src1 = src1 ? src1->data : 0;

    softmax_compute_fn_t compute_func = NULL;
    if (smctx.opt_path) {
        if (!smctx.use_src1) {
            compute_func = compute_fast_softmax_f32_nomask;
        } else if (smctx.use_f16) {
            compute_func = compute_fast_softmax_f32_mask_f16;
        } else {
            compute_func = compute_fast_softmax_f32_mask_f32;
        }
    } else {
        if (smctx.use_f16) {
            compute_func = compute_softmax_f32_fallback_f16;
        } else {
            compute_func = compute_softmax_f32_fallback;
        }
    }

    smctx.compute = (void *) compute_func;
    work_queue_func_t task_func = smctx.use_src1 ? softmax_thread_mask_dma : softmax_thread_dma;

    work_queue_run(octx->ctx->work_queue, task_func, &smctx, n_threads);

    return err;
}

int op_softmax(struct htp_ops_context * octx) {
    int err = HTP_STATUS_OK;

    switch (octx->src[0]->type) {
        case HTP_TYPE_F32:
            err = execute_op_softmax_f32(octx);
            break;

        default:
            err = HTP_STATUS_NO_SUPPORT;
            break;
    }

    return err;
}

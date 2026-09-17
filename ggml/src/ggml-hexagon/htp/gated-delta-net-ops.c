#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#include "hvx-base.h"
#include "hvx-copy.h"
#include "hvx-reduce.h"
#include "hvx-exp.h"
#include "dma-queue.h"
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-tensor.h"
#include "htp-vtcm.h"
#include "hmx-utils.h"
#include "hmx-fa-kernels.h"
#include "hmx-queue.h"
#include "gated-delta-net-ops.h"

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

struct htp_gdn_context {
    struct htp_ops_context * octx;
    const struct htp_gdn_kernel_params * kparams;
    struct htp_gdn_vtcm_layout layout;
    uint8_t * vtcm_base;
    uint32_t row_start;
    uint32_t nrows;
};

static inline HVX_Vector gdn_mul_dot_f32(float * restrict dst, const HVX_Vector * restrict mul, const HVX_Vector * restrict dot, uint32_t n) {
    HVX_Vector acc = Q6_V_vzero();
    const uint32_t epv = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    const uint32_t nloe = n % epv;
    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vd   = hvx_vmemu(dst + i * epv);
        HVX_Vector vm   = mul[i];
        HVX_Vector vdot = dot[i];
        HVX_Vector out  = hvx_vec_mul_f32_f32(vd, vm);
        hvx_vmemu(dst + i * epv) = out;
        acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(out, vdot));
    }

    if (nloe) {
        const uint32_t off = nvec * epv;
        HVX_Vector vm = mul[nvec];
        HVX_Vector vdot = dot[nvec];
        HVX_VectorPred mask = Q6_Q_vsetq2_R(nloe * sizeof(float));
        HVX_Vector zero = Q6_V_vzero();

        HVX_Vector out = hvx_vec_mul_f32_f32(hvx_vmemu(dst + off), vm);
        hvx_vec_store_u(dst + off, nloe * sizeof(float), out);
        acc = hvx_vec_add_f32_f32(acc, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out, vdot), zero));
    }

    return hvx_vec_reduce_sum_f32(acc);
}

static inline HVX_Vector gdn_mul_scalar_dot_f32(float * restrict dst, float mul, const HVX_Vector * restrict dot, uint32_t n) {
    HVX_Vector acc = Q6_V_vzero();
    const HVX_Vector vmul = hvx_vec_splat_f32(mul);
    const uint32_t epv = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    const uint32_t nloe = n % epv;
    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vd   = hvx_vmemu(dst + i * epv);
        HVX_Vector vdot = dot[i];
        HVX_Vector out  = hvx_vec_mul_f32_f32(vd, vmul);
        hvx_vmemu(dst + i * epv) = out;
        acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(out, vdot));
    }

    if (nloe) {
        const uint32_t off = nvec * epv;
        HVX_Vector vdot = dot[nvec];
        HVX_VectorPred mask = Q6_Q_vsetq2_R(nloe * sizeof(float));
        HVX_Vector zero = Q6_V_vzero();

        HVX_Vector out = hvx_vec_mul_f32_f32(hvx_vmemu(dst + off), vmul);
        hvx_vec_store_u(dst + off, nloe * sizeof(float), out);
        acc = hvx_vec_add_f32_f32(acc, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out, vdot), zero));
    }

    return hvx_vec_reduce_sum_f32(acc);
}

static inline HVX_Vector gdn_add_scaled_dot_f32(float * restrict dst, const HVX_Vector * restrict src,
        HVX_Vector vscale, const HVX_Vector * restrict dot, uint32_t n) {
    HVX_Vector acc = Q6_V_vzero();
    const uint32_t epv = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    const uint32_t nloe = n % epv;
    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vd   = hvx_vmemu(dst + i * epv);
        HVX_Vector vs   = src[i];
        HVX_Vector vdot = dot[i];
        HVX_Vector out  = hvx_vec_add_f32_f32(vd, hvx_vec_mul_f32_f32(vs, vscale));
        hvx_vmemu(dst + i * epv) = out;
        acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(out, vdot));
    }

    if (nloe) {
        const uint32_t off = nvec * epv;
        HVX_Vector vs   = src[nvec];
        HVX_Vector vdot = dot[nvec];
        HVX_VectorPred mask = Q6_Q_vsetq2_R(nloe * sizeof(float));
        HVX_Vector zero = Q6_V_vzero();

        HVX_Vector out = hvx_vec_add_f32_f32(hvx_vmemu(dst + off), hvx_vec_mul_f32_f32(vs, vscale));
        hvx_vec_store_u(dst + off, nloe * sizeof(float), out);
        acc = hvx_vec_add_f32_f32(acc, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out, vdot), zero));
    }

    return hvx_vec_reduce_sum_f32(acc);
}

static inline HVX_Vector gdn_mul_dot4_f32(float * restrict dst0, float * restrict dst1,
        float * restrict dst2, float * restrict dst3,
        const HVX_Vector * restrict mul, const HVX_Vector * restrict dot, uint32_t n) {
    HVX_Vector acc0 = Q6_V_vzero();
    HVX_Vector acc1 = Q6_V_vzero();
    HVX_Vector acc2 = Q6_V_vzero();
    HVX_Vector acc3 = Q6_V_vzero();

    const uint32_t epv = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    const uint32_t nloe = n % epv;
    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vm   = mul[i];
        HVX_Vector vdot = dot[i];

        HVX_Vector out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + i * epv), vm);
        HVX_Vector out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + i * epv), vm);
        HVX_Vector out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + i * epv), vm);
        HVX_Vector out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + i * epv), vm);

        hvx_vmemu(dst0 + i * epv) = out0;
        hvx_vmemu(dst1 + i * epv) = out1;
        hvx_vmemu(dst2 + i * epv) = out2;
        hvx_vmemu(dst3 + i * epv) = out3;

        acc0 = hvx_vec_add_f32_f32(acc0, hvx_vec_mul_f32_f32(out0, vdot));
        acc1 = hvx_vec_add_f32_f32(acc1, hvx_vec_mul_f32_f32(out1, vdot));
        acc2 = hvx_vec_add_f32_f32(acc2, hvx_vec_mul_f32_f32(out2, vdot));
        acc3 = hvx_vec_add_f32_f32(acc3, hvx_vec_mul_f32_f32(out3, vdot));
    }

    if (nloe) {
        const uint32_t off = nvec * epv;
        HVX_Vector vm = mul[nvec];
        HVX_Vector vdot = dot[nvec];
        HVX_VectorPred mask = Q6_Q_vsetq2_R(nloe * sizeof(float));
        HVX_Vector zero = Q6_V_vzero();

        HVX_Vector out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + off), vm);
        HVX_Vector out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + off), vm);
        HVX_Vector out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + off), vm);
        HVX_Vector out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + off), vm);

        hvx_vec_store_u(dst0 + off, nloe * sizeof(float), out0);
        hvx_vec_store_u(dst1 + off, nloe * sizeof(float), out1);
        hvx_vec_store_u(dst2 + off, nloe * sizeof(float), out2);
        hvx_vec_store_u(dst3 + off, nloe * sizeof(float), out3);

        acc0 = hvx_vec_add_f32_f32(acc0, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out0, vdot), zero));
        acc1 = hvx_vec_add_f32_f32(acc1, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out1, vdot), zero));
        acc2 = hvx_vec_add_f32_f32(acc2, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out2, vdot), zero));
        acc3 = hvx_vec_add_f32_f32(acc3, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out3, vdot), zero));
    }

    HVX_Vector_x4 acc = { .v = { acc0, acc1, acc2, acc3 } };
    return hvx_vec_reduce_sum_f32x4(acc);
}

static inline HVX_Vector gdn_mul_scalar_dot4_f32(float * restrict dst0, float * restrict dst1,
        float * restrict dst2, float * restrict dst3,
        HVX_Vector vmul, const HVX_Vector * restrict dot, uint32_t n) {
    HVX_Vector acc0 = Q6_V_vzero();
    HVX_Vector acc1 = Q6_V_vzero();
    HVX_Vector acc2 = Q6_V_vzero();
    HVX_Vector acc3 = Q6_V_vzero();

    const uint32_t epv = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    const uint32_t nloe = n % epv;
    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vdot = dot[i];

        HVX_Vector out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + i * epv), vmul);
        HVX_Vector out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + i * epv), vmul);
        HVX_Vector out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + i * epv), vmul);
        HVX_Vector out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + i * epv), vmul);

        hvx_vmemu(dst0 + i * epv) = out0;
        hvx_vmemu(dst1 + i * epv) = out1;
        hvx_vmemu(dst2 + i * epv) = out2;
        hvx_vmemu(dst3 + i * epv) = out3;

        acc0 = hvx_vec_add_f32_f32(acc0, hvx_vec_mul_f32_f32(out0, vdot));
        acc1 = hvx_vec_add_f32_f32(acc1, hvx_vec_mul_f32_f32(out1, vdot));
        acc2 = hvx_vec_add_f32_f32(acc2, hvx_vec_mul_f32_f32(out2, vdot));
        acc3 = hvx_vec_add_f32_f32(acc3, hvx_vec_mul_f32_f32(out3, vdot));
    }

    if (nloe) {
        const uint32_t off = nvec * epv;
        HVX_Vector vdot = dot[nvec];
        HVX_VectorPred mask = Q6_Q_vsetq2_R(nloe * sizeof(float));
        HVX_Vector zero = Q6_V_vzero();

        HVX_Vector out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + off), vmul);
        HVX_Vector out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + off), vmul);
        HVX_Vector out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + off), vmul);
        HVX_Vector out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + off), vmul);

        hvx_vec_store_u(dst0 + off, nloe * sizeof(float), out0);
        hvx_vec_store_u(dst1 + off, nloe * sizeof(float), out1);
        hvx_vec_store_u(dst2 + off, nloe * sizeof(float), out2);
        hvx_vec_store_u(dst3 + off, nloe * sizeof(float), out3);

        acc0 = hvx_vec_add_f32_f32(acc0, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out0, vdot), zero));
        acc1 = hvx_vec_add_f32_f32(acc1, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out1, vdot), zero));
        acc2 = hvx_vec_add_f32_f32(acc2, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out2, vdot), zero));
        acc3 = hvx_vec_add_f32_f32(acc3, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out3, vdot), zero));
    }

    HVX_Vector_x4 acc = { .v = { acc0, acc1, acc2, acc3 } };
    return hvx_vec_reduce_sum_f32x4(acc);
}

static inline HVX_Vector gdn_add_scaled_dot4_f32(float * restrict dst0, float * restrict dst1,
        float * restrict dst2, float * restrict dst3,
        const HVX_Vector * restrict src, const float * restrict scale,
        const HVX_Vector * restrict dot, uint32_t n) {
    HVX_Vector acc0 = Q6_V_vzero();
    HVX_Vector acc1 = Q6_V_vzero();
    HVX_Vector acc2 = Q6_V_vzero();
    HVX_Vector acc3 = Q6_V_vzero();
    const HVX_Vector scale0 = hvx_vec_splat_f32(scale[0]);
    const HVX_Vector scale1 = hvx_vec_splat_f32(scale[1]);
    const HVX_Vector scale2 = hvx_vec_splat_f32(scale[2]);
    const HVX_Vector scale3 = hvx_vec_splat_f32(scale[3]);

    const uint32_t epv = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    const uint32_t nloe = n % epv;
    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vs   = src[i];
        HVX_Vector vdot = dot[i];

        HVX_Vector out0 = hvx_vec_add_f32_f32(hvx_vmemu(dst0 + i * epv), hvx_vec_mul_f32_f32(vs, scale0));
        HVX_Vector out1 = hvx_vec_add_f32_f32(hvx_vmemu(dst1 + i * epv), hvx_vec_mul_f32_f32(vs, scale1));
        HVX_Vector out2 = hvx_vec_add_f32_f32(hvx_vmemu(dst2 + i * epv), hvx_vec_mul_f32_f32(vs, scale2));
        HVX_Vector out3 = hvx_vec_add_f32_f32(hvx_vmemu(dst3 + i * epv), hvx_vec_mul_f32_f32(vs, scale3));

        hvx_vmemu(dst0 + i * epv) = out0;
        hvx_vmemu(dst1 + i * epv) = out1;
        hvx_vmemu(dst2 + i * epv) = out2;
        hvx_vmemu(dst3 + i * epv) = out3;

        acc0 = hvx_vec_add_f32_f32(acc0, hvx_vec_mul_f32_f32(out0, vdot));
        acc1 = hvx_vec_add_f32_f32(acc1, hvx_vec_mul_f32_f32(out1, vdot));
        acc2 = hvx_vec_add_f32_f32(acc2, hvx_vec_mul_f32_f32(out2, vdot));
        acc3 = hvx_vec_add_f32_f32(acc3, hvx_vec_mul_f32_f32(out3, vdot));
    }

    if (nloe) {
        const uint32_t off = nvec * epv;
        HVX_Vector vs   = src[nvec];
        HVX_Vector vdot = dot[nvec];
        HVX_VectorPred mask = Q6_Q_vsetq2_R(nloe * sizeof(float));
        HVX_Vector zero = Q6_V_vzero();

        HVX_Vector out0 = hvx_vec_add_f32_f32(hvx_vmemu(dst0 + off), hvx_vec_mul_f32_f32(vs, scale0));
        HVX_Vector out1 = hvx_vec_add_f32_f32(hvx_vmemu(dst1 + off), hvx_vec_mul_f32_f32(vs, scale1));
        HVX_Vector out2 = hvx_vec_add_f32_f32(hvx_vmemu(dst2 + off), hvx_vec_mul_f32_f32(vs, scale2));
        HVX_Vector out3 = hvx_vec_add_f32_f32(hvx_vmemu(dst3 + off), hvx_vec_mul_f32_f32(vs, scale3));

        hvx_vec_store_u(dst0 + off, nloe * sizeof(float), out0);
        hvx_vec_store_u(dst1 + off, nloe * sizeof(float), out1);
        hvx_vec_store_u(dst2 + off, nloe * sizeof(float), out2);
        hvx_vec_store_u(dst3 + off, nloe * sizeof(float), out3);

        acc0 = hvx_vec_add_f32_f32(acc0, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out0, vdot), zero));
        acc1 = hvx_vec_add_f32_f32(acc1, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out1, vdot), zero));
        acc2 = hvx_vec_add_f32_f32(acc2, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out2, vdot), zero));
        acc3 = hvx_vec_add_f32_f32(acc3, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out3, vdot), zero));
    }

    HVX_Vector_x4 acc = { .v = { acc0, acc1, acc2, acc3 } };
    return hvx_vec_reduce_sum_f32x4(acc);
}

static inline HVX_Vector gdn_mul_dot8_f32(float * restrict dst0, float * restrict dst1,
        float * restrict dst2, float * restrict dst3, float * restrict dst4,
        float * restrict dst5, float * restrict dst6, float * restrict dst7,
        const HVX_Vector * restrict mul, const HVX_Vector * restrict dot, uint32_t n) {
    HVX_Vector acc0 = Q6_V_vzero();
    HVX_Vector acc1 = Q6_V_vzero();
    HVX_Vector acc2 = Q6_V_vzero();
    HVX_Vector acc3 = Q6_V_vzero();
    HVX_Vector acc4 = Q6_V_vzero();
    HVX_Vector acc5 = Q6_V_vzero();
    HVX_Vector acc6 = Q6_V_vzero();
    HVX_Vector acc7 = Q6_V_vzero();

    const uint32_t epv = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    const uint32_t nloe = n % epv;
    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vm   = mul[i];
        HVX_Vector vdot = dot[i];

        HVX_Vector out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + i * epv), vm);
        HVX_Vector out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + i * epv), vm);
        HVX_Vector out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + i * epv), vm);
        HVX_Vector out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + i * epv), vm);
        HVX_Vector out4 = hvx_vec_mul_f32_f32(hvx_vmemu(dst4 + i * epv), vm);
        HVX_Vector out5 = hvx_vec_mul_f32_f32(hvx_vmemu(dst5 + i * epv), vm);
        HVX_Vector out6 = hvx_vec_mul_f32_f32(hvx_vmemu(dst6 + i * epv), vm);
        HVX_Vector out7 = hvx_vec_mul_f32_f32(hvx_vmemu(dst7 + i * epv), vm);

        hvx_vmemu(dst0 + i * epv) = out0;
        hvx_vmemu(dst1 + i * epv) = out1;
        hvx_vmemu(dst2 + i * epv) = out2;
        hvx_vmemu(dst3 + i * epv) = out3;
        hvx_vmemu(dst4 + i * epv) = out4;
        hvx_vmemu(dst5 + i * epv) = out5;
        hvx_vmemu(dst6 + i * epv) = out6;
        hvx_vmemu(dst7 + i * epv) = out7;

        acc0 = hvx_vec_add_f32_f32(acc0, hvx_vec_mul_f32_f32(out0, vdot));
        acc1 = hvx_vec_add_f32_f32(acc1, hvx_vec_mul_f32_f32(out1, vdot));
        acc2 = hvx_vec_add_f32_f32(acc2, hvx_vec_mul_f32_f32(out2, vdot));
        acc3 = hvx_vec_add_f32_f32(acc3, hvx_vec_mul_f32_f32(out3, vdot));
        acc4 = hvx_vec_add_f32_f32(acc4, hvx_vec_mul_f32_f32(out4, vdot));
        acc5 = hvx_vec_add_f32_f32(acc5, hvx_vec_mul_f32_f32(out5, vdot));
        acc6 = hvx_vec_add_f32_f32(acc6, hvx_vec_mul_f32_f32(out6, vdot));
        acc7 = hvx_vec_add_f32_f32(acc7, hvx_vec_mul_f32_f32(out7, vdot));
    }

    if (nloe) {
        const uint32_t off = nvec * epv;
        HVX_Vector vm = mul[nvec];
        HVX_Vector vdot = dot[nvec];
        HVX_VectorPred mask = Q6_Q_vsetq2_R(nloe * sizeof(float));
        HVX_Vector zero = Q6_V_vzero();

        HVX_Vector out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + off), vm);
        HVX_Vector out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + off), vm);
        HVX_Vector out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + off), vm);
        HVX_Vector out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + off), vm);
        HVX_Vector out4 = hvx_vec_mul_f32_f32(hvx_vmemu(dst4 + off), vm);
        HVX_Vector out5 = hvx_vec_mul_f32_f32(hvx_vmemu(dst5 + off), vm);
        HVX_Vector out6 = hvx_vec_mul_f32_f32(hvx_vmemu(dst6 + off), vm);
        HVX_Vector out7 = hvx_vec_mul_f32_f32(hvx_vmemu(dst7 + off), vm);

        hvx_vec_store_u(dst0 + off, nloe * sizeof(float), out0);
        hvx_vec_store_u(dst1 + off, nloe * sizeof(float), out1);
        hvx_vec_store_u(dst2 + off, nloe * sizeof(float), out2);
        hvx_vec_store_u(dst3 + off, nloe * sizeof(float), out3);
        hvx_vec_store_u(dst4 + off, nloe * sizeof(float), out4);
        hvx_vec_store_u(dst5 + off, nloe * sizeof(float), out5);
        hvx_vec_store_u(dst6 + off, nloe * sizeof(float), out6);
        hvx_vec_store_u(dst7 + off, nloe * sizeof(float), out7);

        acc0 = hvx_vec_add_f32_f32(acc0, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out0, vdot), zero));
        acc1 = hvx_vec_add_f32_f32(acc1, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out1, vdot), zero));
        acc2 = hvx_vec_add_f32_f32(acc2, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out2, vdot), zero));
        acc3 = hvx_vec_add_f32_f32(acc3, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out3, vdot), zero));
        acc4 = hvx_vec_add_f32_f32(acc4, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out4, vdot), zero));
        acc5 = hvx_vec_add_f32_f32(acc5, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out5, vdot), zero));
        acc6 = hvx_vec_add_f32_f32(acc6, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out6, vdot), zero));
        acc7 = hvx_vec_add_f32_f32(acc7, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out7, vdot), zero));
    }

    HVX_Vector_x4 accA = { .v = { acc0, acc1, acc2, acc3 } };
    HVX_Vector_x4 accB = { .v = { acc4, acc5, acc6, acc7 } };
    HVX_Vector rA = hvx_vec_reduce_sum_f32x4(accA);
    HVX_Vector rB = hvx_vec_reduce_sum_f32x4(accB);
    HVX_VectorPred q16 = Q6_Q_vsetq2_R(16);
    return Q6_V_vmux_QVV(q16, rA, Q6_V_vror_VR(rB, 128 - 16));
}

static inline HVX_Vector gdn_mul_scalar_dot8_f32(float * restrict dst0, float * restrict dst1,
        float * restrict dst2, float * restrict dst3, float * restrict dst4,
        float * restrict dst5, float * restrict dst6, float * restrict dst7,
        HVX_Vector vmul, const HVX_Vector * restrict dot, uint32_t n) {
    HVX_Vector acc0 = Q6_V_vzero();
    HVX_Vector acc1 = Q6_V_vzero();
    HVX_Vector acc2 = Q6_V_vzero();
    HVX_Vector acc3 = Q6_V_vzero();
    HVX_Vector acc4 = Q6_V_vzero();
    HVX_Vector acc5 = Q6_V_vzero();
    HVX_Vector acc6 = Q6_V_vzero();
    HVX_Vector acc7 = Q6_V_vzero();

    const uint32_t epv = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    const uint32_t nloe = n % epv;
    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vdot = dot[i];

        HVX_Vector out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + i * epv), vmul);
        HVX_Vector out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + i * epv), vmul);
        HVX_Vector out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + i * epv), vmul);
        HVX_Vector out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + i * epv), vmul);
        HVX_Vector out4 = hvx_vec_mul_f32_f32(hvx_vmemu(dst4 + i * epv), vmul);
        HVX_Vector out5 = hvx_vec_mul_f32_f32(hvx_vmemu(dst5 + i * epv), vmul);
        HVX_Vector out6 = hvx_vec_mul_f32_f32(hvx_vmemu(dst6 + i * epv), vmul);
        HVX_Vector out7 = hvx_vec_mul_f32_f32(hvx_vmemu(dst7 + i * epv), vmul);

        hvx_vmemu(dst0 + i * epv) = out0;
        hvx_vmemu(dst1 + i * epv) = out1;
        hvx_vmemu(dst2 + i * epv) = out2;
        hvx_vmemu(dst3 + i * epv) = out3;
        hvx_vmemu(dst4 + i * epv) = out4;
        hvx_vmemu(dst5 + i * epv) = out5;
        hvx_vmemu(dst6 + i * epv) = out6;
        hvx_vmemu(dst7 + i * epv) = out7;

        acc0 = hvx_vec_add_f32_f32(acc0, hvx_vec_mul_f32_f32(out0, vdot));
        acc1 = hvx_vec_add_f32_f32(acc1, hvx_vec_mul_f32_f32(out1, vdot));
        acc2 = hvx_vec_add_f32_f32(acc2, hvx_vec_mul_f32_f32(out2, vdot));
        acc3 = hvx_vec_add_f32_f32(acc3, hvx_vec_mul_f32_f32(out3, vdot));
        acc4 = hvx_vec_add_f32_f32(acc4, hvx_vec_mul_f32_f32(out4, vdot));
        acc5 = hvx_vec_add_f32_f32(acc5, hvx_vec_mul_f32_f32(out5, vdot));
        acc6 = hvx_vec_add_f32_f32(acc6, hvx_vec_mul_f32_f32(out6, vdot));
        acc7 = hvx_vec_add_f32_f32(acc7, hvx_vec_mul_f32_f32(out7, vdot));
    }

    if (nloe) {
        const uint32_t off = nvec * epv;
        HVX_Vector vdot = dot[nvec];
        HVX_VectorPred mask = Q6_Q_vsetq2_R(nloe * sizeof(float));
        HVX_Vector zero = Q6_V_vzero();

        HVX_Vector out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + off), vmul);
        HVX_Vector out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + off), vmul);
        HVX_Vector out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + off), vmul);
        HVX_Vector out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + off), vmul);
        HVX_Vector out4 = hvx_vec_mul_f32_f32(hvx_vmemu(dst4 + off), vmul);
        HVX_Vector out5 = hvx_vec_mul_f32_f32(hvx_vmemu(dst5 + off), vmul);
        HVX_Vector out6 = hvx_vec_mul_f32_f32(hvx_vmemu(dst6 + off), vmul);
        HVX_Vector out7 = hvx_vec_mul_f32_f32(hvx_vmemu(dst7 + off), vmul);

        hvx_vec_store_u(dst0 + off, nloe * sizeof(float), out0);
        hvx_vec_store_u(dst1 + off, nloe * sizeof(float), out1);
        hvx_vec_store_u(dst2 + off, nloe * sizeof(float), out2);
        hvx_vec_store_u(dst3 + off, nloe * sizeof(float), out3);
        hvx_vec_store_u(dst4 + off, nloe * sizeof(float), out4);
        hvx_vec_store_u(dst5 + off, nloe * sizeof(float), out5);
        hvx_vec_store_u(dst6 + off, nloe * sizeof(float), out6);
        hvx_vec_store_u(dst7 + off, nloe * sizeof(float), out7);

        acc0 = hvx_vec_add_f32_f32(acc0, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out0, vdot), zero));
        acc1 = hvx_vec_add_f32_f32(acc1, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out1, vdot), zero));
        acc2 = hvx_vec_add_f32_f32(acc2, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out2, vdot), zero));
        acc3 = hvx_vec_add_f32_f32(acc3, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out3, vdot), zero));
        acc4 = hvx_vec_add_f32_f32(acc4, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out4, vdot), zero));
        acc5 = hvx_vec_add_f32_f32(acc5, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out5, vdot), zero));
        acc6 = hvx_vec_add_f32_f32(acc6, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out6, vdot), zero));
        acc7 = hvx_vec_add_f32_f32(acc7, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out7, vdot), zero));
    }

    HVX_Vector_x4 accA = { .v = { acc0, acc1, acc2, acc3 } };
    HVX_Vector_x4 accB = { .v = { acc4, acc5, acc6, acc7 } };
    HVX_Vector rA = hvx_vec_reduce_sum_f32x4(accA);
    HVX_Vector rB = hvx_vec_reduce_sum_f32x4(accB);
    HVX_VectorPred q16 = Q6_Q_vsetq2_R(16);
    return Q6_V_vmux_QVV(q16, rA, Q6_V_vror_VR(rB, 128 - 16));
}

static inline HVX_Vector gdn_add_scaled_dot8_f32(float * restrict dst0, float * restrict dst1,
        float * restrict dst2, float * restrict dst3, float * restrict dst4,
        float * restrict dst5, float * restrict dst6, float * restrict dst7,
        const HVX_Vector * restrict src, const float * restrict scale,
        const HVX_Vector * restrict dot, uint32_t n) {
    HVX_Vector acc0 = Q6_V_vzero();
    HVX_Vector acc1 = Q6_V_vzero();
    HVX_Vector acc2 = Q6_V_vzero();
    HVX_Vector acc3 = Q6_V_vzero();
    HVX_Vector acc4 = Q6_V_vzero();
    HVX_Vector acc5 = Q6_V_vzero();
    HVX_Vector acc6 = Q6_V_vzero();
    HVX_Vector acc7 = Q6_V_vzero();
    const HVX_Vector scale0 = hvx_vec_splat_f32(scale[0]);
    const HVX_Vector scale1 = hvx_vec_splat_f32(scale[1]);
    const HVX_Vector scale2 = hvx_vec_splat_f32(scale[2]);
    const HVX_Vector scale3 = hvx_vec_splat_f32(scale[3]);
    const HVX_Vector scale4 = hvx_vec_splat_f32(scale[4]);
    const HVX_Vector scale5 = hvx_vec_splat_f32(scale[5]);
    const HVX_Vector scale6 = hvx_vec_splat_f32(scale[6]);
    const HVX_Vector scale7 = hvx_vec_splat_f32(scale[7]);

    const uint32_t epv = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    const uint32_t nloe = n % epv;
    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vs   = src[i];
        HVX_Vector vdot = dot[i];

        HVX_Vector out0 = hvx_vec_add_f32_f32(hvx_vmemu(dst0 + i * epv), hvx_vec_mul_f32_f32(vs, scale0));
        HVX_Vector out1 = hvx_vec_add_f32_f32(hvx_vmemu(dst1 + i * epv), hvx_vec_mul_f32_f32(vs, scale1));
        HVX_Vector out2 = hvx_vec_add_f32_f32(hvx_vmemu(dst2 + i * epv), hvx_vec_mul_f32_f32(vs, scale2));
        HVX_Vector out3 = hvx_vec_add_f32_f32(hvx_vmemu(dst3 + i * epv), hvx_vec_mul_f32_f32(vs, scale3));
        HVX_Vector out4 = hvx_vec_add_f32_f32(hvx_vmemu(dst4 + i * epv), hvx_vec_mul_f32_f32(vs, scale4));
        HVX_Vector out5 = hvx_vec_add_f32_f32(hvx_vmemu(dst5 + i * epv), hvx_vec_mul_f32_f32(vs, scale5));
        HVX_Vector out6 = hvx_vec_add_f32_f32(hvx_vmemu(dst6 + i * epv), hvx_vec_mul_f32_f32(vs, scale6));
        HVX_Vector out7 = hvx_vec_add_f32_f32(hvx_vmemu(dst7 + i * epv), hvx_vec_mul_f32_f32(vs, scale7));

        hvx_vmemu(dst0 + i * epv) = out0;
        hvx_vmemu(dst1 + i * epv) = out1;
        hvx_vmemu(dst2 + i * epv) = out2;
        hvx_vmemu(dst3 + i * epv) = out3;
        hvx_vmemu(dst4 + i * epv) = out4;
        hvx_vmemu(dst5 + i * epv) = out5;
        hvx_vmemu(dst6 + i * epv) = out6;
        hvx_vmemu(dst7 + i * epv) = out7;

        acc0 = hvx_vec_add_f32_f32(acc0, hvx_vec_mul_f32_f32(out0, vdot));
        acc1 = hvx_vec_add_f32_f32(acc1, hvx_vec_mul_f32_f32(out1, vdot));
        acc2 = hvx_vec_add_f32_f32(acc2, hvx_vec_mul_f32_f32(out2, vdot));
        acc3 = hvx_vec_add_f32_f32(acc3, hvx_vec_mul_f32_f32(out3, vdot));
        acc4 = hvx_vec_add_f32_f32(acc4, hvx_vec_mul_f32_f32(out4, vdot));
        acc5 = hvx_vec_add_f32_f32(acc5, hvx_vec_mul_f32_f32(out5, vdot));
        acc6 = hvx_vec_add_f32_f32(acc6, hvx_vec_mul_f32_f32(out6, vdot));
        acc7 = hvx_vec_add_f32_f32(acc7, hvx_vec_mul_f32_f32(out7, vdot));
    }

    if (nloe) {
        const uint32_t off = nvec * epv;
        HVX_Vector vs   = src[nvec];
        HVX_Vector vdot = dot[nvec];
        HVX_VectorPred mask = Q6_Q_vsetq2_R(nloe * sizeof(float));
        HVX_Vector zero = Q6_V_vzero();

        HVX_Vector out0 = hvx_vec_add_f32_f32(hvx_vmemu(dst0 + off), hvx_vec_mul_f32_f32(vs, scale0));
        HVX_Vector out1 = hvx_vec_add_f32_f32(hvx_vmemu(dst1 + off), hvx_vec_mul_f32_f32(vs, scale1));
        HVX_Vector out2 = hvx_vec_add_f32_f32(hvx_vmemu(dst2 + off), hvx_vec_mul_f32_f32(vs, scale2));
        HVX_Vector out3 = hvx_vec_add_f32_f32(hvx_vmemu(dst3 + off), hvx_vec_mul_f32_f32(vs, scale3));
        HVX_Vector out4 = hvx_vec_add_f32_f32(hvx_vmemu(dst4 + off), hvx_vec_mul_f32_f32(vs, scale4));
        HVX_Vector out5 = hvx_vec_add_f32_f32(hvx_vmemu(dst5 + off), hvx_vec_mul_f32_f32(vs, scale5));
        HVX_Vector out6 = hvx_vec_add_f32_f32(hvx_vmemu(dst6 + off), hvx_vec_mul_f32_f32(vs, scale6));
        HVX_Vector out7 = hvx_vec_add_f32_f32(hvx_vmemu(dst7 + off), hvx_vec_mul_f32_f32(vs, scale7));

        hvx_vec_store_u(dst0 + off, nloe * sizeof(float), out0);
        hvx_vec_store_u(dst1 + off, nloe * sizeof(float), out1);
        hvx_vec_store_u(dst2 + off, nloe * sizeof(float), out2);
        hvx_vec_store_u(dst3 + off, nloe * sizeof(float), out3);
        hvx_vec_store_u(dst4 + off, nloe * sizeof(float), out4);
        hvx_vec_store_u(dst5 + off, nloe * sizeof(float), out5);
        hvx_vec_store_u(dst6 + off, nloe * sizeof(float), out6);
        hvx_vec_store_u(dst7 + off, nloe * sizeof(float), out7);

        acc0 = hvx_vec_add_f32_f32(acc0, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out0, vdot), zero));
        acc1 = hvx_vec_add_f32_f32(acc1, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out1, vdot), zero));
        acc2 = hvx_vec_add_f32_f32(acc2, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out2, vdot), zero));
        acc3 = hvx_vec_add_f32_f32(acc3, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out3, vdot), zero));
        acc4 = hvx_vec_add_f32_f32(acc4, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out4, vdot), zero));
        acc5 = hvx_vec_add_f32_f32(acc5, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out5, vdot), zero));
        acc6 = hvx_vec_add_f32_f32(acc6, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out6, vdot), zero));
        acc7 = hvx_vec_add_f32_f32(acc7, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out7, vdot), zero));
    }

    HVX_Vector_x4 accA = { .v = { acc0, acc1, acc2, acc3 } };
    HVX_Vector_x4 accB = { .v = { acc4, acc5, acc6, acc7 } };
    HVX_Vector rA = hvx_vec_reduce_sum_f32x4(accA);
    HVX_Vector rB = hvx_vec_reduce_sum_f32x4(accB);
    HVX_VectorPred q16 = Q6_Q_vsetq2_R(16);
    return Q6_V_vmux_QVV(q16, rA, Q6_V_vror_VR(rB, 128 - 16));
}

static inline void gdn_step_kda_f32(
    float * restrict s_work,
    float * restrict attn_out,
    const float * restrict q_t,
    const float * restrict k_t,
    const float * restrict v_t,
    const float * restrict g_t,
    float beta_val,
    float scale,
    uint32_t S_v
) {
    const uint32_t epv  = 128 / sizeof(float);
    const uint32_t nvec = S_v / epv;
    const uint32_t nloe = S_v % epv;

    HVX_Vector vq[4];
    HVX_Vector vk[4];
    HVX_Vector vg[4];

    static const float kInf    = INFINITY;
    static const float kMaxExp = 88.7228f;
    const HVX_Vector max_exp = hvx_vec_splat_f32(kMaxExp);
    const HVX_Vector inf     = hvx_vec_splat_f32(kInf);

    for (uint32_t i = 0; i < nvec; ++i) {
        vq[i] = hvx_vmemu(q_t + i * epv);
        vk[i] = hvx_vmemu(k_t + i * epv);
        vg[i] = hvx_vec_exp_f32_guard(hvx_vmemu(g_t + i * epv), max_exp, inf);
    }
    if (nloe) {
        vq[nvec] = hvx_vmemu(q_t + nvec * epv);
        vk[nvec] = hvx_vmemu(k_t + nvec * epv);
        vg[nvec] = hvx_vec_exp_f32_guard(hvx_vmemu(g_t + nvec * epv), max_exp, inf);
    }

    const HVX_Vector vbeta  = hvx_vec_splat_f32(beta_val);
    const HVX_Vector vscale = hvx_vec_splat_f32(scale);

    float delta[8] __attribute__((aligned(128)));

    uint32_t j = 0;
    for (; j + 8 <= S_v; j += 8) {
        float * row0 = s_work + (uint64_t) (j + 0) * S_v;
        float * row1 = s_work + (uint64_t) (j + 1) * S_v;
        float * row2 = s_work + (uint64_t) (j + 2) * S_v;
        float * row3 = s_work + (uint64_t) (j + 3) * S_v;
        float * row4 = s_work + (uint64_t) (j + 4) * S_v;
        float * row5 = s_work + (uint64_t) (j + 5) * S_v;
        float * row6 = s_work + (uint64_t) (j + 6) * S_v;
        float * row7 = s_work + (uint64_t) (j + 7) * S_v;

        HVX_Vector vsums = gdn_mul_dot8_f32(row0, row1, row2, row3, row4, row5, row6, row7,
                                            vg, vk, S_v);

        HVX_Vector vv_t   = hvx_vmemu(v_t + j);
        HVX_Vector diff   = hvx_vec_sub_f32_f32(vv_t, vsums);
        HVX_Vector vdelta = hvx_vec_mul_f32_f32(diff, vbeta);
        hvx_vec_store_u(delta, 8 * sizeof(float), vdelta);

        HVX_Vector vattn = gdn_add_scaled_dot8_f32(row0, row1, row2, row3, row4, row5, row6, row7,
                                                   vk, delta, vq, S_v);

        HVX_Vector res_attn = hvx_vec_mul_f32_f32(vattn, vscale);
        hvx_vec_store_u(attn_out + j, 8 * sizeof(float), res_attn);
    }
    for (; j + 4 <= S_v; j += 4) {
        float * row0 = s_work + (uint64_t) (j + 0) * S_v;
        float * row1 = s_work + (uint64_t) (j + 1) * S_v;
        float * row2 = s_work + (uint64_t) (j + 2) * S_v;
        float * row3 = s_work + (uint64_t) (j + 3) * S_v;

        HVX_Vector vsums = gdn_mul_dot4_f32(row0, row1, row2, row3, vg, vk, S_v);

        HVX_Vector vv_t   = hvx_vmemu(v_t + j);
        HVX_Vector diff   = hvx_vec_sub_f32_f32(vv_t, vsums);
        HVX_Vector vdelta = hvx_vec_mul_f32_f32(diff, vbeta);
        hvx_vec_store_u(delta, 4 * sizeof(float), vdelta);

        HVX_Vector vattn = gdn_add_scaled_dot4_f32(row0, row1, row2, row3, vk, delta, vq, S_v);

        HVX_Vector res_attn = hvx_vec_mul_f32_f32(vattn, vscale);
        hvx_vec_store_u(attn_out + j, 4 * sizeof(float), res_attn);
    }
    for (; j < S_v; ++j) {
        float * row = s_work + (uint64_t) j * S_v;
        HVX_Vector vsum = gdn_mul_dot_f32(row, vg, vk, S_v);
        HVX_Vector vv_t = hvx_vec_splat_f32(v_t[j]);
        HVX_Vector vdj  = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(vv_t, vsum), vbeta);
        HVX_Vector vres = gdn_add_scaled_dot_f32(row, vk, vdj, vq, S_v);
        attn_out[j] = hvx_vec_get_f32(hvx_vec_mul_f32_f32(vres, vscale));
    }
}

static inline void gdn_step_scalar_f32(
    float * restrict s_work,
    float * restrict attn_out,
    const float * restrict q_t,
    const float * restrict k_t,
    const float * restrict v_t,
    const float * restrict g_t,
    float beta_val,
    float scale,
    uint32_t S_v
) {
    const uint32_t epv  = 128 / sizeof(float);
    const uint32_t nvec = S_v / epv;
    const uint32_t nloe = S_v % epv;

    HVX_Vector vq[4];
    HVX_Vector vk[4];

    for (uint32_t i = 0; i < nvec; ++i) {
        vq[i] = hvx_vmemu(q_t + i * epv);
        vk[i] = hvx_vmemu(k_t + i * epv);
    }
    if (nloe) {
        vq[nvec] = hvx_vmemu(q_t + nvec * epv);
        vk[nvec] = hvx_vmemu(k_t + nvec * epv);
    }

    const float gate       = expf(g_t[0]);
    const HVX_Vector vgate = hvx_vec_splat_f32(gate);
    const HVX_Vector vbeta = hvx_vec_splat_f32(beta_val);
    const HVX_Vector vscale = hvx_vec_splat_f32(scale);

    float delta[8] __attribute__((aligned(128)));

    uint32_t j = 0;
    for (; j + 8 <= S_v; j += 8) {
        float * row0 = s_work + (uint64_t) (j + 0) * S_v;
        float * row1 = s_work + (uint64_t) (j + 1) * S_v;
        float * row2 = s_work + (uint64_t) (j + 2) * S_v;
        float * row3 = s_work + (uint64_t) (j + 3) * S_v;
        float * row4 = s_work + (uint64_t) (j + 4) * S_v;
        float * row5 = s_work + (uint64_t) (j + 5) * S_v;
        float * row6 = s_work + (uint64_t) (j + 6) * S_v;
        float * row7 = s_work + (uint64_t) (j + 7) * S_v;

        HVX_Vector vsums = gdn_mul_scalar_dot8_f32(row0, row1, row2, row3, row4, row5, row6, row7,
                                                   vgate, vk, S_v);

        HVX_Vector vv_t   = hvx_vmemu(v_t + j);
        HVX_Vector diff   = hvx_vec_sub_f32_f32(vv_t, vsums);
        HVX_Vector vdelta = hvx_vec_mul_f32_f32(diff, vbeta);
        hvx_vec_store_u(delta, 8 * sizeof(float), vdelta);

        HVX_Vector vattn = gdn_add_scaled_dot8_f32(row0, row1, row2, row3, row4, row5, row6, row7,
                                                   vk, delta, vq, S_v);

        HVX_Vector res_attn = hvx_vec_mul_f32_f32(vattn, vscale);
        hvx_vec_store_u(attn_out + j, 8 * sizeof(float), res_attn);
    }
    for (; j + 4 <= S_v; j += 4) {
        float * row0 = s_work + (uint64_t) (j + 0) * S_v;
        float * row1 = s_work + (uint64_t) (j + 1) * S_v;
        float * row2 = s_work + (uint64_t) (j + 2) * S_v;
        float * row3 = s_work + (uint64_t) (j + 3) * S_v;

        HVX_Vector vsums = gdn_mul_scalar_dot4_f32(row0, row1, row2, row3, vgate, vk, S_v);

        HVX_Vector vv_t   = hvx_vmemu(v_t + j);
        HVX_Vector diff   = hvx_vec_sub_f32_f32(vv_t, vsums);
        HVX_Vector vdelta = hvx_vec_mul_f32_f32(diff, vbeta);
        hvx_vec_store_u(delta, 4 * sizeof(float), vdelta);

        HVX_Vector vattn = gdn_add_scaled_dot4_f32(row0, row1, row2, row3, vk, delta, vq, S_v);

        HVX_Vector res_attn = hvx_vec_mul_f32_f32(vattn, vscale);
        hvx_vec_store_u(attn_out + j, 4 * sizeof(float), res_attn);
    }
    for (; j < S_v; ++j) {
        float * row = s_work + (uint64_t) j * S_v;
        HVX_Vector vsum = gdn_mul_scalar_dot_f32(row, gate, vk, S_v);
        HVX_Vector vv_t = hvx_vec_splat_f32(v_t[j]);
        HVX_Vector vdj  = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(vv_t, vsum), vbeta);
        HVX_Vector vres = gdn_add_scaled_dot_f32(row, vk, vdj, vq, S_v);
        attn_out[j] = hvx_vec_get_f32(hvx_vec_mul_f32_f32(vres, vscale));
    }
}

static void gated_delta_net_f32_pp_thread(unsigned int nth, unsigned int ith, void * data) {
    struct htp_gdn_context * gctx = (struct htp_gdn_context *) data;
    struct htp_ops_context * octx = gctx->octx;
    const struct htp_gdn_kernel_params * kparams = gctx->kparams;

    const struct htp_tensor * q     = octx->src[0];
    const struct htp_tensor * k     = octx->src[1];
    const struct htp_tensor * v     = octx->src[2];
    const struct htp_tensor * g     = octx->src[3];
    const struct htp_tensor * beta  = octx->src[4];
    const struct htp_tensor * state = octx->src[5];
    const struct htp_tensor * dst   = octx->dst;

    const uint32_t S_v      = kparams->S_v;
    const uint32_t H        = kparams->H;
    const uint32_t n_tokens = kparams->n_tokens;
    const uint32_t n_seqs   = kparams->n_seqs;
    const uint32_t K        = kparams->K;
    const uint32_t row_end  = gctx->row_start + gctx->nrows;

    if (ith >= gctx->nrows) {
        return;
    }

    const struct htp_tensor * dst_cache = octx->dsts[1];
    const float scale = kparams->scale;
    float * dst_base       = (float *) (uintptr_t) dst->data;
    float * state_out_base = dst_cache ? (float *) (uintptr_t) dst_cache->data : (dst_base + S_v * H * n_tokens * n_seqs);

    dma_queue * dma_q = octx->ctx->dma[ith];
    const struct htp_gdn_vtcm_layout * layout = &gctx->layout;
    float * s_work[2];
    s_work[0] = (float *) (gctx->vtcm_base + layout->bytes_per_thread * ith);
    s_work[1] = s_work[0] + layout->state_aligned / sizeof(float);

    const struct fastdiv_values * fd_H   = &kparams->div_H;
    const struct fastdiv_values * fd_q1  = &kparams->div_q1;
    const struct fastdiv_values * fd_k1  = &kparams->div_k1;
    const struct fastdiv_values * fd_rq3 = &kparams->div_rq3;
    const struct fastdiv_values * fd_rk3 = &kparams->div_rk3;

    const uint32_t state_seq_stride = kparams->state_seq_stride;
    const uint64_t state_size_per_snap = (uint64_t) kparams->state_size_per_snap;
    const dma_addr_t state_out_dma_base = dst_cache ? dst_cache->data : (dst->data + S_v * H * n_tokens * n_seqs * sizeof(float));

    uint32_t ir_prefetch = gctx->row_start + ith;
    int spad_idx = 0;

    // Prefetch preamble (up to 2 steps)
    for (int step = 0; step < 2 && ir_prefetch < row_end; step++) {
        const uint32_t piv1 = fastmodulo(ir_prefetch, H, fd_H);
        const uint32_t piv3 = fastdiv(ir_prefetch, fd_H);
        dma_addr_t ps_in  = state->data + ((uint64_t) piv3 * state_seq_stride + (uint64_t) piv1 * S_v * S_v) * sizeof(float);
        dma_addr_t ps_out = state_out_dma_base + ((uint64_t) piv3 * H + piv1) * S_v * S_v * sizeof(float);

        // Push dummy write-back
        dma_queue_push(dma_q, dma_make_data(ps_out, s_work[spad_idx]),
                       S_v * sizeof(float), S_v * sizeof(float),
                       S_v * sizeof(float), 0);

        // Push fetch
        dma_queue_push(dma_q, dma_make_data(s_work[spad_idx], ps_in),
                       S_v * sizeof(float), S_v * sizeof(float),
                       S_v * sizeof(float), S_v);

        ir_prefetch += nth;
        spad_idx ^= 1;
    }

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    int curr_spad_idx = 0;
    for (uint32_t ir = gctx->row_start + ith; ir < row_end; ir += nth) {
        dma_queue_pop(dma_q);
        dma_queue_pop(dma_q);

        float * s_work_curr = s_work[curr_spad_idx];

        const uint32_t iv1 = fastmodulo(ir, H, fd_H);
        const uint32_t iv3 = fastdiv(ir, fd_H);

        const uint32_t iq1 = fastmodulo(iv1, q->ne[1], fd_q1);
        const uint32_t ik1 = fastmodulo(iv1, k->ne[1], fd_k1);
        const uint32_t iq3 = fastdiv(iv3, fd_rq3);
        const uint32_t ik3 = fastdiv(iv3, fd_rk3);

        dma_addr_t s_out  = state_out_dma_base + ((uint64_t) iv3 * H + iv1) * S_v * S_v * sizeof(float);
        float * attn_data = dst_base + ((uint64_t) iv3 * n_tokens * H + iv1) * S_v;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        for (uint32_t t = 0; t < n_tokens; ++t) {
            const float * q_t = (const float *) ((const uint8_t *) (uintptr_t) q->data +
                    (uint64_t) iq3 * q->nb[3] + (uint64_t) t * q->nb[2] + (uint64_t) iq1 * q->nb[1]);
            const float * k_t = (const float *) ((const uint8_t *) (uintptr_t) k->data +
                    (uint64_t) ik3 * k->nb[3] + (uint64_t) t * k->nb[2] + (uint64_t) ik1 * k->nb[1]);
            const float * v_t = (const float *) ((const uint8_t *) (uintptr_t) v->data +
                    (uint64_t) iv3 * v->nb[3] + (uint64_t) t * v->nb[2] + (uint64_t) iv1 * v->nb[1]);
            const float * g_t = (const float *) ((const uint8_t *) (uintptr_t) g->data +
                    (uint64_t) iv3 * g->nb[3] + (uint64_t) t * g->nb[2] + (uint64_t) iv1 * g->nb[1]);
            const float beta_val = *(const float *) ((const uint8_t *) (uintptr_t) beta->data +
                    (uint64_t) iv3 * beta->nb[3] + (uint64_t) t * beta->nb[2] + (uint64_t) iv1 * beta->nb[1]);

            if (kparams->kda) {
                gdn_step_kda_f32(s_work_curr, attn_data, q_t, k_t, v_t, g_t, beta_val, scale, S_v);
            } else {
                gdn_step_scalar_f32(s_work_curr, attn_data, q_t, k_t, v_t, g_t, beta_val, scale, S_v);
            }

            if (K > 1) {
                const int64_t target_slot = (int64_t) n_tokens - 1 - (int64_t) t;
                if (target_slot > 0 && target_slot < (int64_t) K) {
                    float * curr_state_o = state_out_base + (uint64_t) target_slot * state_size_per_snap + ((uint64_t) iv3 * H + iv1) * S_v * S_v;
                    hvx_copy_f32_uu((uint8_t *) curr_state_o, (const uint8_t *) s_work_curr, S_v * S_v);
                }
            }

            attn_data += (uint64_t) S_v * H;
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        // Push real write-back
        dma_queue_push(dma_q, dma_make_data(s_out, s_work_curr),
                       S_v * sizeof(float), S_v * sizeof(float),
                       S_v * sizeof(float), S_v);

        // Prefetch next block (if any)
        if (ir_prefetch < row_end) {
            const uint32_t piv1 = fastmodulo(ir_prefetch, H, fd_H);
            const uint32_t piv3 = fastdiv(ir_prefetch, fd_H);
            dma_addr_t ps_in = state->data + ((uint64_t) piv3 * state_seq_stride + (uint64_t) piv1 * S_v * S_v) * sizeof(float);

            dma_queue_push(dma_q, dma_make_data(s_work[spad_idx], ps_in),
                           S_v * sizeof(float), S_v * sizeof(float),
                           S_v * sizeof(float), S_v);

            ir_prefetch += nth;
            spad_idx ^= 1;
        }

        curr_spad_idx ^= 1;
    }
    dma_queue_flush(dma_q);
}

static void gated_delta_net_f32_tg_thread(unsigned int nth, unsigned int ith, void * data) {
    struct htp_gdn_context * gctx = (struct htp_gdn_context *) data;
    struct htp_ops_context * octx = gctx->octx;
    const struct htp_gdn_kernel_params * kparams = gctx->kparams;

    const struct htp_tensor * q     = octx->src[0];
    const struct htp_tensor * k     = octx->src[1];
    const struct htp_tensor * v     = octx->src[2];
    const struct htp_tensor * g     = octx->src[3];
    const struct htp_tensor * beta  = octx->src[4];
    const struct htp_tensor * state = octx->src[5];
    const struct htp_tensor * dst   = octx->dst;

    const uint32_t S_v      = kparams->S_v;
    const uint32_t H        = kparams->H;
    const uint32_t n_seqs   = kparams->n_seqs;
    const uint32_t row_end  = gctx->row_start + gctx->nrows;

    if (ith >= gctx->nrows) {
        return;
    }

    const struct htp_tensor * dst_cache = octx->dsts[1];
    const float scale = kparams->scale;
    float * dst_base  = (float *) (uintptr_t) dst->data;

    dma_queue * dma_q = octx->ctx->dma[ith];
    const struct htp_gdn_vtcm_layout * layout = &gctx->layout;
    float * s_work[2];
    s_work[0] = (float *) (gctx->vtcm_base + layout->bytes_per_thread * ith);
    s_work[1] = s_work[0] + layout->state_aligned / sizeof(float);

    const struct fastdiv_values * fd_H   = &kparams->div_H;
    const struct fastdiv_values * fd_q1  = &kparams->div_q1;
    const struct fastdiv_values * fd_k1  = &kparams->div_k1;
    const struct fastdiv_values * fd_rq3 = &kparams->div_rq3;
    const struct fastdiv_values * fd_rk3 = &kparams->div_rk3;

    const uint32_t state_seq_stride = kparams->state_seq_stride;
    const dma_addr_t state_out_dma_base = dst_cache ? dst_cache->data : (dst->data + S_v * H * n_seqs * sizeof(float));

    uint32_t ir_prefetch = gctx->row_start + ith;
    int spad_idx = 0;

    // Prefetch preamble (up to 2 steps)
    for (int step = 0; step < 2 && ir_prefetch < row_end; step++) {
        const uint32_t piv1 = fastmodulo(ir_prefetch, H, fd_H);
        const uint32_t piv3 = fastdiv(ir_prefetch, fd_H);
        dma_addr_t ps_in  = state->data + ((uint64_t) piv3 * state_seq_stride + (uint64_t) piv1 * S_v * S_v) * sizeof(float);
        dma_addr_t ps_out = state_out_dma_base + ((uint64_t) piv3 * H + piv1) * S_v * S_v * sizeof(float);

        // Push dummy write-back
        dma_queue_push(dma_q, dma_make_data(ps_out, s_work[spad_idx]),
                       S_v * sizeof(float), S_v * sizeof(float),
                       S_v * sizeof(float), 0);

        // Push fetch
        dma_queue_push(dma_q, dma_make_data(s_work[spad_idx], ps_in),
                       S_v * sizeof(float), S_v * sizeof(float),
                       S_v * sizeof(float), S_v);

        ir_prefetch += nth;
        spad_idx ^= 1;
    }

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    int curr_spad_idx = 0;
    for (uint32_t ir = gctx->row_start + ith; ir < row_end; ir += nth) {
        dma_queue_pop(dma_q);
        dma_queue_pop(dma_q);

        float * s_work_curr = s_work[curr_spad_idx];

        const uint32_t iv1 = fastmodulo(ir, H, fd_H);
        const uint32_t iv3 = fastdiv(ir, fd_H);

        const uint32_t iq1 = fastmodulo(iv1, q->ne[1], fd_q1);
        const uint32_t ik1 = fastmodulo(iv1, k->ne[1], fd_k1);
        const uint32_t iq3 = fastdiv(iv3, fd_rq3);
        const uint32_t ik3 = fastdiv(iv3, fd_rk3);

        dma_addr_t s_out  = state_out_dma_base + ((uint64_t) iv3 * H + iv1) * S_v * S_v * sizeof(float);
        float * attn_data = dst_base + ((uint64_t) iv3 * H + iv1) * S_v;

        const float * q_t = (const float *) ((const uint8_t *) (uintptr_t) q->data +
                (uint64_t) iq3 * q->nb[3] + (uint64_t) iq1 * q->nb[1]);
        const float * k_t = (const float *) ((const uint8_t *) (uintptr_t) k->data +
                (uint64_t) ik3 * k->nb[3] + (uint64_t) ik1 * k->nb[1]);
        const float * v_t = (const float *) ((const uint8_t *) (uintptr_t) v->data +
                (uint64_t) iv3 * v->nb[3] + (uint64_t) iv1 * v->nb[1]);
        const float * g_t = (const float *) ((const uint8_t *) (uintptr_t) g->data +
                (uint64_t) iv3 * g->nb[3] + (uint64_t) iv1 * g->nb[1]);
        const float beta_val = *(const float *) ((const uint8_t *) (uintptr_t) beta->data +
                (uint64_t) iv3 * beta->nb[3] + (uint64_t) iv1 * beta->nb[1]);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        if (kparams->kda) {
            gdn_step_kda_f32(s_work_curr, attn_data, q_t, k_t, v_t, g_t, beta_val, scale, S_v);
        } else {
            gdn_step_scalar_f32(s_work_curr, attn_data, q_t, k_t, v_t, g_t, beta_val, scale, S_v);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        // Push real write-back
        dma_queue_push(dma_q, dma_make_data(s_out, s_work_curr),
                       S_v * sizeof(float), S_v * sizeof(float),
                       S_v * sizeof(float), S_v);

        // Prefetch next block (if any)
        if (ir_prefetch < row_end) {
            const uint32_t piv1 = fastmodulo(ir_prefetch, H, fd_H);
            const uint32_t piv3 = fastdiv(ir_prefetch, fd_H);
            dma_addr_t ps_in = state->data + ((uint64_t) piv3 * state_seq_stride + (uint64_t) piv1 * S_v * S_v) * sizeof(float);

            dma_queue_push(dma_q, dma_make_data(s_work[spad_idx], ps_in),
                           S_v * sizeof(float), S_v * sizeof(float),
                           S_v * sizeof(float), S_v);

            ir_prefetch += nth;
            spad_idx ^= 1;
        }

        curr_spad_idx ^= 1;
    }
    dma_queue_flush(dma_q);
}

struct htp_gdn_hmx_gemm_job {
    const __fp16 * row_tiles;
    const __fp16 * col_tiles;
    __fp16 *       out_tiles;
    uint32_t       n_row_tiles;
    uint32_t       n_col_tiles;
    uint32_t       n_dot_tiles;
    uint32_t       dot_stride;
    uint8_t *      hmx_scales;
};

static void htp_gdn_hmx_gemm_worker(void * data) {
    struct htp_gdn_hmx_gemm_job * job = (struct htp_gdn_hmx_gemm_job *) data;
    asm volatile(HMX_SET_BIAS("%0") :: "r"((unsigned int)job->hmx_scales));

    const size_t dot_stride = job->dot_stride;
    for (uint32_t r = 0; r < job->n_row_tiles; ++r) {
        const __fp16 * r_tiles = job->row_tiles + r * dot_stride;
        const __fp16 * c_tiles = job->col_tiles;
        __fp16 *       o_tile  = job->out_tiles + r * job->n_col_tiles * HMX_FP16_TILE_N_ELMS;

        for (uint32_t c = 0; c < job->n_col_tiles; ++c) {
            hmx_fa_qk_dot_tile(r_tiles, c_tiles, o_tile, job->n_dot_tiles);
            c_tiles += dot_stride;
            o_tile  += HMX_FP16_TILE_N_ELMS;
        }
    }
}

static inline void htp_gdn_run_hmx_gemm(
    hmx_queue_t q,
    struct htp_gdn_hmx_gemm_job * job,
    const __fp16 * row_tiles,
    const __fp16 * col_tiles,
    __fp16 * out_tiles,
    uint32_t n_row_tiles,
    uint32_t n_col_tiles,
    uint32_t n_dot_tiles,
    uint8_t * scales
) {
    job->row_tiles   = row_tiles;
    job->col_tiles   = col_tiles;
    job->out_tiles   = out_tiles;
    job->n_row_tiles = n_row_tiles;
    job->n_col_tiles = n_col_tiles;
    job->n_dot_tiles = n_dot_tiles;
    job->dot_stride  = n_dot_tiles * HMX_FP16_TILE_N_ELMS;
    job->hmx_scales  = scales;

    hmx_queue_push(q, hmx_queue_make_desc(htp_gdn_hmx_gemm_worker, job));
    hmx_queue_pop(q);
}

static inline void gdn_unpack_64x64_tiles_to_vectors(
    HVX_Vector * restrict rows,
    const __fp16 * restrict tiles
) {
    const HVX_Vector * t00 = (const HVX_Vector *) (tiles + 0 * HMX_FP16_TILE_N_ELMS);
    const HVX_Vector * t01 = (const HVX_Vector *) (tiles + 1 * HMX_FP16_TILE_N_ELMS);
    const HVX_Vector * t10 = (const HVX_Vector *) (tiles + 2 * HMX_FP16_TILE_N_ELMS);
    const HVX_Vector * t11 = (const HVX_Vector *) (tiles + 3 * HMX_FP16_TILE_N_ELMS);

    for (uint32_t r = 0; r < 16; ++r) {
        HVX_VectorPair vp0 = Q6_W_vdeal_VVR(t01[r], t00[r], -2);
        rows[2 * r + 0] = Q6_V_lo_W(vp0);
        rows[2 * r + 1] = Q6_V_hi_W(vp0);

        HVX_VectorPair vp1 = Q6_W_vdeal_VVR(t11[r], t10[r], -2);
        rows[32 + 2 * r + 0] = Q6_V_lo_W(vp1);
        rows[32 + 2 * r + 1] = Q6_V_hi_W(vp1);
    }
}

static inline void gdn_pack_64x64_vectors_to_tiles(
    __fp16 * restrict tiles,
    const HVX_Vector * restrict rows
) {
    HVX_Vector * t00 = (HVX_Vector *) (tiles + 0 * HMX_FP16_TILE_N_ELMS);
    HVX_Vector * t01 = (HVX_Vector *) (tiles + 1 * HMX_FP16_TILE_N_ELMS);
    HVX_Vector * t10 = (HVX_Vector *) (tiles + 2 * HMX_FP16_TILE_N_ELMS);
    HVX_Vector * t11 = (HVX_Vector *) (tiles + 3 * HMX_FP16_TILE_N_ELMS);

    for (uint32_t r = 0; r < 16; ++r) {
        HVX_VectorPair vp0 = Q6_W_vshuff_VVR(rows[2 * r + 1], rows[2 * r + 0], -2);
        t00[r] = Q6_V_lo_W(vp0);
        t01[r] = Q6_V_hi_W(vp0);

        HVX_VectorPair vp1 = Q6_W_vshuff_VVR(rows[32 + 2 * r + 1], rows[32 + 2 * r + 0], -2);
        t10[r] = Q6_V_lo_W(vp1);
        t11[r] = Q6_V_hi_W(vp1);
    }
}

static inline void gdn_unpack_64xS_tiles_to_f32(
    float * restrict dst_f32,
    const __fp16 * restrict tiles,
    uint32_t S_v
) {
    const uint32_t n_col_tiles = S_v / 32;
    for (uint32_t r0 = 0; r0 < 2; ++r0) {
        for (uint32_t d = 0; d < S_v / 64; ++d) {
            const HVX_Vector * t0 = (const HVX_Vector *) (tiles + (r0 * n_col_tiles + 2 * d + 0) * HMX_FP16_TILE_N_ELMS);
            const HVX_Vector * t1 = (const HVX_Vector *) (tiles + (r0 * n_col_tiles + 2 * d + 1) * HMX_FP16_TILE_N_ELMS);

            for (uint32_t r = 0; r < 16; ++r) {
                HVX_VectorPair vp01 = Q6_W_vdeal_VVR(t1[r], t0[r], -2);
                HVX_VectorPair p0 = hvx_vec_f16_to_f32(Q6_V_lo_W(vp01));
                HVX_VectorPair p1 = hvx_vec_f16_to_f32(Q6_V_hi_W(vp01));

                float * out0 = dst_f32 + (r0 * 32 + 2 * r + 0) * S_v + d * 64;
                float * out1 = dst_f32 + (r0 * 32 + 2 * r + 1) * S_v + d * 64;

                hvx_vmemu(out0 + 0)  = Q6_V_lo_W(p0);
                hvx_vmemu(out0 + 32) = Q6_V_hi_W(p0);
                hvx_vmemu(out1 + 0)  = Q6_V_lo_W(p1);
                hvx_vmemu(out1 + 32) = Q6_V_hi_W(p1);
            }
        }
    }
}

static inline void gdn_unpack_64xS_tiles_to_f16(
    __fp16 * restrict dst_f16,
    const __fp16 * restrict tiles,
    uint32_t S_v
) {
    const uint32_t n_col_tiles = S_v / 32;
    for (uint32_t r0 = 0; r0 < 2; ++r0) {
        for (uint32_t d = 0; d < S_v / 64; ++d) {
            const HVX_Vector * t0 = (const HVX_Vector *) (tiles + (r0 * n_col_tiles + 2 * d + 0) * HMX_FP16_TILE_N_ELMS);
            const HVX_Vector * t1 = (const HVX_Vector *) (tiles + (r0 * n_col_tiles + 2 * d + 1) * HMX_FP16_TILE_N_ELMS);

            for (uint32_t r = 0; r < 16; ++r) {
                HVX_VectorPair vp01 = Q6_W_vdeal_VVR(t1[r], t0[r], -2);
                __fp16 * out0 = dst_f16 + (r0 * 32 + 2 * r + 0) * S_v + d * 64;
                __fp16 * out1 = dst_f16 + (r0 * 32 + 2 * r + 1) * S_v + d * 64;

                hvx_vmemu(out0) = Q6_V_lo_W(vp01);
                hvx_vmemu(out1) = Q6_V_hi_W(vp01);
            }
        }
    }
}

static inline void gdn_unpack_SxS_tiles_to_f32(
    float * restrict dst_f32,
    const __fp16 * restrict tiles,
    uint32_t S_v
) {
    const uint32_t n_tiles = S_v / 32;
    for (uint32_t r0 = 0; r0 < n_tiles; ++r0) {
        for (uint32_t d = 0; d < S_v / 64; ++d) {
            const HVX_Vector * t0 = (const HVX_Vector *) (tiles + (r0 * n_tiles + 2 * d + 0) * HMX_FP16_TILE_N_ELMS);
            const HVX_Vector * t1 = (const HVX_Vector *) (tiles + (r0 * n_tiles + 2 * d + 1) * HMX_FP16_TILE_N_ELMS);

            for (uint32_t r = 0; r < 16; ++r) {
                HVX_VectorPair vp01 = Q6_W_vdeal_VVR(t1[r], t0[r], -2);
                HVX_VectorPair p0 = hvx_vec_f16_to_f32(Q6_V_lo_W(vp01));
                HVX_VectorPair p1 = hvx_vec_f16_to_f32(Q6_V_hi_W(vp01));

                float * out0 = dst_f32 + (r0 * 32 + 2 * r + 0) * S_v + d * 64;
                float * out1 = dst_f32 + (r0 * 32 + 2 * r + 1) * S_v + d * 64;

                hvx_vmemu(out0 + 0)  = Q6_V_lo_W(p0);
                hvx_vmemu(out0 + 32) = Q6_V_hi_W(p0);
                hvx_vmemu(out1 + 0)  = Q6_V_lo_W(p1);
                hvx_vmemu(out1 + 32) = Q6_V_hi_W(p1);
            }
        }
    }
}

static inline void gdn_f32_to_hmx_row_tiles(
    __fp16 * restrict dst_tiles,
    const float * restrict src,
    const float * restrict scale_per_row,
    uint32_t n_rows,
    uint32_t n_cols
) {
    const uint32_t n_col_tiles = n_cols / 32;
    for (uint32_t r = 0; r < n_rows; r += 2) {
        uint32_t r0 = r / 32;
        uint32_t r1 = (r % 32) / 2;
        const float * p0 = src + (r + 0) * n_cols;
        const float * p1 = src + (r + 1) * n_cols;
        HVX_Vector s0 = scale_per_row ? hvx_vec_splat_f32(scale_per_row[r + 0]) : hvx_vec_splat_f32(1.0f);
        HVX_Vector s1 = scale_per_row ? hvx_vec_splat_f32(scale_per_row[r + 1]) : hvx_vec_splat_f32(1.0f);

        for (uint32_t c = 0; c < n_col_tiles; ++c) {
            HVX_Vector v0 = hvx_vec_mul_f32_f32(hvx_vmemu(p0 + c * 32), s0);
            HVX_Vector v1 = hvx_vec_mul_f32_f32(hvx_vmemu(p1 + c * 32), s1);
            HVX_Vector vh = hvx_vec_f32_to_f16_shuff(v0, v1);
            __fp16 * tile = dst_tiles + (r0 * n_col_tiles + c) * HMX_FP16_TILE_N_ELMS;
            ((HVX_Vector *) tile)[r1] = vh;
        }
    }
}

static inline void hvx_transpose_32x32_words(HVX_Vector * restrict m, HVX_Vector * restrict tmp) {
    for (int i = 0; i < 16; ++i) {
        HVX_VectorPair p = Q6_W_vshuff_VVR(m[2*i + 1], m[2*i], -4);
        tmp[2*i + 0] = Q6_V_lo_W(p);
        tmp[2*i + 1] = Q6_V_hi_W(p);
    }

    for (int b = 0; b < 32; b += 4) {
        HVX_VectorPair p0 = Q6_W_vshuff_VVR(tmp[b + 2], tmp[b + 0], -8);
        HVX_VectorPair p1 = Q6_W_vshuff_VVR(tmp[b + 3], tmp[b + 1], -8);
        m[b + 0] = Q6_V_lo_W(p0); m[b + 1] = Q6_V_hi_W(p0);
        m[b + 2] = Q6_V_lo_W(p1); m[b + 3] = Q6_V_hi_W(p1);
    }

    for (int b = 0; b < 32; b += 8) {
        for (int i = 0; i < 4; ++i) {
            HVX_VectorPair p = Q6_W_vshuff_VVR(m[b + i + 4], m[b + i], -16);
            tmp[b + 2*i + 0] = Q6_V_lo_W(p);
            tmp[b + 2*i + 1] = Q6_V_hi_W(p);
        }
    }

    for (int b = 0; b < 32; b += 16) {
        for (int i = 0; i < 8; ++i) {
            HVX_VectorPair p = Q6_W_vshuff_VVR(tmp[b + i + 8], tmp[b + i], -32);
            m[b + 2*i + 0] = Q6_V_lo_W(p);
            m[b + 2*i + 1] = Q6_V_hi_W(p);
        }
    }

    for (int i = 0; i < 16; ++i) {
        HVX_VectorPair p = Q6_W_vshuff_VVR(m[i + 16], m[i], -64);
        tmp[2 * i + 0]   = Q6_V_lo_W(p);
        tmp[2 * i + 1]   = Q6_V_hi_W(p);
    }

    for (int i = 0; i < 32; ++i) {
        m[i] = tmp[i];
    }
}

static inline void gdn_pack_d_t_row_tiles(
    __fp16 * restrict dst_tiles,
    const __fp16 * restrict src_d,
    uint32_t S_v,
    HVX_Vector * restrict m,
    HVX_Vector * restrict tmp
) {
    for (uint32_t col_half = 0; col_half < S_v / 64; ++col_half) {
        uint32_t r0_base = col_half * 2;
        for (uint32_t c0 = 0; c0 < 2; ++c0) {
            for (uint32_t s_local = 0; s_local < 32; ++s_local) {
                uint32_t s = c0 * 32 + s_local;
                m[s_local] = hvx_vmemu(src_d + s * S_v + col_half * 64);
            }

            hvx_transpose_32x32_words(m, tmp);

            uint32_t tile0_idx = (r0_base + 0) * 2 + c0;
            uint32_t tile1_idx = (r0_base + 1) * 2 + c0;
            HVX_Vector * t0 = (HVX_Vector *)(dst_tiles + tile0_idx * HMX_FP16_TILE_N_ELMS);
            HVX_Vector * t1 = (HVX_Vector *)(dst_tiles + tile1_idx * HMX_FP16_TILE_N_ELMS);

            for (uint32_t r = 0; r < 16; ++r) {
                t0[r] = m[r];
                t1[r] = m[16 + r];
            }
        }
    }
}

static __attribute__((noinline)) void gdn_build_inv_l_and_a(
    HVX_Vector * restrict rows_inv,
    HVX_Vector * restrict rows_a,
    const HVX_Vector * restrict rows_kk,
    const HVX_Vector * restrict rows_qk,
    const __fp16 * restrict decay_m,
    const __fp16 * restrict decay_a,
    const float * restrict beta
) {
    const HVX_Vector v_one_f16 = hvx_vec_splat_f16(1.0f);
    HVX_VectorAlias local_row_m;

    for (uint32_t t = 0; t < 64; ++t) {
        HVX_Vector v_decay_m = hvx_vmemu(decay_m + t * 64);
        HVX_Vector v_scale_t = hvx_vec_splat_f16(beta[t]);
        HVX_Vector row_m     = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(rows_kk[t], v_decay_m), v_scale_t);

        HVX_Vector v_decay_a = hvx_vmemu(decay_a + t * 64);
        rows_a[t]            = hvx_vec_mul_f16_f16(rows_qk[t], v_decay_a);

        HVX_Vector v_inv_t = Q6_V_vzero();
        local_row_m.v = row_m;

        for (uint32_t k_idx = 0; k_idx < t; ++k_idx) {
            if (local_row_m.fp16[k_idx] != 0.0f) {
                HVX_Vector v_lk = hvx_vec_splat_f16(local_row_m.fp16[k_idx]);
                v_inv_t = hvx_vec_sub_f16_f16(v_inv_t, hvx_vec_mul_f16_f16(v_lk, rows_inv[k_idx]));
            }
        }
        HVX_VectorPred q_diag = (t == 0) ? Q6_Q_vsetq2_R(2) : Q6_Q_and_QQn(Q6_Q_vsetq2_R(2 * (t + 1)), Q6_Q_vsetq2_R(2 * t));
        rows_inv[t] = Q6_V_vmux_QVV(q_diag, v_one_f16, v_inv_t);
    }
}

static int gated_delta_net_f32_hmx_chunked(
    struct htp_ops_context * octx,
    const struct htp_gdn_kernel_params * kparams,
    uint32_t row_start,
    uint32_t nrows
) {
    const struct htp_tensor * q     = octx->src[0];
    const struct htp_tensor * k     = octx->src[1];
    const struct htp_tensor * v     = octx->src[2];
    const struct htp_tensor * g     = octx->src[3];
    const struct htp_tensor * beta  = octx->src[4];
    const struct htp_tensor * state = octx->src[5];
    const struct htp_tensor * dst   = octx->dst;
    const struct htp_tensor * dst_cache = octx->dsts[1];

    const uint32_t S_v        = kparams->S_v;
    const uint32_t H          = kparams->H;
    const uint32_t n_tokens   = kparams->n_tokens;
    const float    scale      = kparams->scale;
    const uint32_t chunk_size = kparams->chunk_size;
    const uint32_t n_chunks   = kparams->n_chunks;
    const uint32_t n_sv_tiles = S_v / 32;

    uint8_t * vtcm_cur = (uint8_t *) octx->ctx->vtcm_base;

    float *  vtcm_s_state     = (float *)  vtcm_seq_alloc(&vtcm_cur, hex_round_up(S_v * S_v * sizeof(float), 2048));
    __fp16 * vtcm_s_f16       = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, hex_round_up(S_v * S_v * sizeof(__fp16), 2048));
    __fp16 * vtcm_s_col_tiles = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, 16 * HMX_FP16_TILE_SIZE);

    float * vtcm_q_f32 = (float *) vtcm_seq_alloc(&vtcm_cur, hex_round_up(chunk_size * S_v * sizeof(float), 2048));
    float * vtcm_k_f32 = (float *) vtcm_seq_alloc(&vtcm_cur, hex_round_up(chunk_size * S_v * sizeof(float), 2048));
    float * vtcm_v_f32 = (float *) vtcm_seq_alloc(&vtcm_cur, hex_round_up(chunk_size * S_v * sizeof(float), 2048));
    float * vtcm_g_f32 = (float *) vtcm_seq_alloc(&vtcm_cur, hex_round_up(chunk_size * sizeof(float), 2048));
    float * vtcm_b_f32 = (float *) vtcm_seq_alloc(&vtcm_cur, hex_round_up(chunk_size * sizeof(float), 2048));
    float * vtcm_o_f32 = (float *) vtcm_seq_alloc(&vtcm_cur, hex_round_up(chunk_size * S_v * sizeof(float), 2048));

    float * vtcm_v_inter_f32 = (float *) vtcm_seq_alloc(&vtcm_cur, hex_round_up(chunk_size * S_v * sizeof(float), 2048));
    float * vtcm_o_inter_f32 = (float *) vtcm_seq_alloc(&vtcm_cur, hex_round_up(chunk_size * S_v * sizeof(float), 2048));
    float * vtcm_o_intra_f32 = (float *) vtcm_seq_alloc(&vtcm_cur, hex_round_up(chunk_size * S_v * sizeof(float), 2048));
    float * vtcm_s_update_f32 = (float *) vtcm_seq_alloc(&vtcm_cur, hex_round_up(S_v * S_v * sizeof(float), 2048));

    __fp16 * vtcm_k_f16       = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, hex_round_up(chunk_size * S_v * sizeof(__fp16), 2048));
    __fp16 * vtcm_v_prime_f16 = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, hex_round_up(chunk_size * S_v * sizeof(__fp16), 2048));
    __fp16 * vtcm_delta_f16   = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, hex_round_up(chunk_size * S_v * sizeof(__fp16), 2048));
    __fp16 * vtcm_d_f16       = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, hex_round_up(chunk_size * S_v * sizeof(__fp16), 2048));

    __fp16 * vtcm_q_row_tiles        = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, 8 * HMX_FP16_TILE_SIZE);
    __fp16 * vtcm_k_row_tiles        = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, 8 * HMX_FP16_TILE_SIZE);
    __fp16 * vtcm_k_col_tiles        = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, 8 * HMX_FP16_TILE_SIZE);
    __fp16 * vtcm_k_prime_row_tiles  = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, 8 * HMX_FP16_TILE_SIZE);
    __fp16 * vtcm_k_col_tiles_64x128 = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, 8 * HMX_FP16_TILE_SIZE);
    __fp16 * vtcm_kk_tiles           = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, 4 * HMX_FP16_TILE_SIZE);
    __fp16 * vtcm_qk_tiles           = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, 4 * HMX_FP16_TILE_SIZE);
    __fp16 * vtcm_v_inter_tiles      = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, 8 * HMX_FP16_TILE_SIZE);
    __fp16 * vtcm_o_inter_tiles      = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, 8 * HMX_FP16_TILE_SIZE);
    __fp16 * vtcm_inv_row_tiles      = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, 4 * HMX_FP16_TILE_SIZE);
    __fp16 * vtcm_a_row_tiles        = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, 4 * HMX_FP16_TILE_SIZE);
    __fp16 * vtcm_v_prime_col_tiles  = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, 8 * HMX_FP16_TILE_SIZE);
    __fp16 * vtcm_delta_tiles        = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, 8 * HMX_FP16_TILE_SIZE);
    __fp16 * vtcm_delta_col_tiles    = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, 8 * HMX_FP16_TILE_SIZE);
    __fp16 * vtcm_o_intra_tiles      = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, 8 * HMX_FP16_TILE_SIZE);
    __fp16 * vtcm_d_row_tiles        = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, 8 * HMX_FP16_TILE_SIZE);
    __fp16 * vtcm_s_update_tiles     = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, 16 * HMX_FP16_TILE_SIZE);

    uint8_t * vtcm_scales_1 = vtcm_seq_alloc(&vtcm_cur, 256);

    float * gamma         = (float *) vtcm_seq_alloc(&vtcm_cur, hex_round_up(chunk_size * sizeof(float), 128));
    float * lambda_init   = (float *) vtcm_seq_alloc(&vtcm_cur, hex_round_up(chunk_size * sizeof(float), 128));
    __fp16 * decay_m      = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, 64 * 64 * sizeof(__fp16));
    __fp16 * decay_a      = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, 64 * 64 * sizeof(__fp16));

    HVX_Vector * rows_kk  = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, 64 * sizeof(HVX_Vector));
    HVX_Vector * rows_qk  = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, 64 * sizeof(HVX_Vector));
    HVX_Vector * rows_inv = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, 64 * sizeof(HVX_Vector));
    HVX_Vector * rows_a   = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, 64 * sizeof(HVX_Vector));
    HVX_Vector * vtcm_m   = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, 32 * sizeof(HVX_Vector));
    HVX_Vector * vtcm_tmp = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, 32 * sizeof(HVX_Vector));

    float  * vtcm_attn_rem = (float *)  vtcm_seq_alloc(&vtcm_cur, 128 * sizeof(float));

    if ((size_t) (vtcm_cur - octx->ctx->vtcm_base) > octx->ctx->vtcm_size) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    hmx_init_column_scales(vtcm_scales_1, Q6_V_vsplat_R(0x3c00));

    hmx_queue_t hmx_q = octx->ctx->hmx_queue;
    dma_queue * dma_q = octx->ctx->dma[0];
    struct htp_thread_trace * tr = &octx->ctx->trace[0];
    struct htp_gdn_hmx_gemm_job gemm_job;

    for (uint32_t r = row_start; r < row_start + nrows; ++r) {
        const uint32_t iv1 = fastmodulo(r, H, &kparams->div_H);
        const uint32_t iv3 = fastdiv(r, &kparams->div_H);

        const uint32_t iq1 = fastmodulo(iv1, q->ne[1], &kparams->div_q1);
        const uint32_t ik1 = fastmodulo(iv1, k->ne[1], &kparams->div_k1);
        const uint32_t iq3 = fastdiv(iv3, &kparams->div_rq3);
        const uint32_t ik3 = fastdiv(iv3, &kparams->div_rk3);

        const dma_addr_t state_in_dma = state->data +
            ((uint64_t) iv3 * kparams->state_seq_stride + (uint64_t) iv1 * S_v * S_v) * sizeof(float);

        const dma_addr_t state_out_dma = dst_cache ?
            (dst_cache->data + ((uint64_t) iv3 * H + iv1) * S_v * S_v * sizeof(float)) :
            (dst->data + ((uint64_t) S_v * H * n_tokens * kparams->n_seqs + (uint64_t) (iv3 * H + iv1) * S_v * S_v) * sizeof(float));

        dma_queue_push(dma_q, dma_make_data(vtcm_s_state, state_in_dma),
                       S_v * sizeof(float), S_v * sizeof(float), S_v * sizeof(float), S_v);
        dma_queue_pop(dma_q);

        for (uint32_t c = 0; c < n_chunks; ++c) {
            const uint32_t t_chunk = c * chunk_size;

            const dma_addr_t q_dma = q->data + (uint64_t) iq3 * q->nb[3] + (uint64_t) t_chunk * q->nb[2] + (uint64_t) iq1 * q->nb[1];
            const dma_addr_t k_dma = k->data + (uint64_t) ik3 * k->nb[3] + (uint64_t) t_chunk * k->nb[2] + (uint64_t) ik1 * k->nb[1];
            const dma_addr_t v_dma = v->data + (uint64_t) iv3 * v->nb[3] + (uint64_t) t_chunk * v->nb[2] + (uint64_t) iv1 * v->nb[1];

            dma_queue_push(dma_q, dma_make_data(vtcm_q_f32, q_dma), S_v * sizeof(float), q->nb[2], S_v * sizeof(float), chunk_size);
            dma_queue_push(dma_q, dma_make_data(vtcm_k_f32, k_dma), S_v * sizeof(float), k->nb[2], S_v * sizeof(float), chunk_size);
            dma_queue_push(dma_q, dma_make_data(vtcm_v_f32, v_dma), S_v * sizeof(float), v->nb[2], S_v * sizeof(float), chunk_size);
            dma_queue_pop(dma_q);
            dma_queue_pop(dma_q);
            dma_queue_pop(dma_q);

            for (uint32_t t = 0; t < chunk_size; ++t) {
                vtcm_g_f32[t] = *(const float *) ((const uint8_t *) (uintptr_t) g->data +
                    (uint64_t) iv3 * g->nb[3] + (uint64_t) (t_chunk + t) * g->nb[2] + (uint64_t) iv1 * g->nb[1]);
                vtcm_b_f32[t] = *(const float *) ((const uint8_t *) (uintptr_t) beta->data +
                    (uint64_t) iv3 * beta->nb[3] + (uint64_t) (t_chunk + t) * beta->nb[2] + (uint64_t) iv1 * beta->nb[1]);
            }

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) c);

            gamma[0] = vtcm_g_f32[0];
            for (uint32_t t = 1; t < 64; ++t) {
                gamma[t] = gamma[t - 1] + vtcm_g_f32[t];
            }
            for (uint32_t t = 0; t < 64; ++t) {
                float val = gamma[t];
                if (val < -20.0f) val = -20.0f;
                if (val > 0.0f) val = 0.0f;
                lambda_init[t] = expf(val);
            }

            for (uint32_t t = 0; t < 64; ++t) {
                const float gamma_t = gamma[t];
                for (uint32_t s = 0; s < 64; ++s) {
                    if (s < t) {
                        float diff = gamma_t - gamma[s];
                        if (diff < -20.0f) diff = -20.0f;
                        if (diff > 0.0f) diff = 0.0f;
                        decay_m[t * 64 + s] = (__fp16) expf(diff);
                        decay_a[t * 64 + s] = (__fp16) expf(diff);
                    } else if (s == t) {
                        decay_m[t * 64 + s] = 0.0f;
                        decay_a[t * 64 + s] = 1.0f;
                    } else {
                        decay_m[t * 64 + s] = 0.0f;
                        decay_a[t * 64 + s] = 0.0f;
                    }
                }
            }

            gdn_f32_to_hmx_row_tiles(vtcm_k_row_tiles, vtcm_k_f32, NULL, 64, S_v);

            for (uint32_t t = 0; t < 64; ++t) {
                for (uint32_t i = 0; i < S_v; i += 64) {
                    HVX_Vector v0 = hvx_vmemu(vtcm_k_f32 + t * S_v + i + 0);
                    HVX_Vector v1 = (i + 32 < S_v) ? hvx_vmemu(vtcm_k_f32 + t * S_v + i + 32) : Q6_V_vzero();
                    hvx_vmemu(vtcm_k_f16 + t * S_v + i) = hvx_vec_f32_to_f16(v0, v1);
                }
            }
            hmx_interleave_rows_to_tiles(vtcm_k_col_tiles, vtcm_k_f16, 64, S_v, S_v, 0, 64);

            gdn_f32_to_hmx_row_tiles(vtcm_q_row_tiles, vtcm_q_f32, NULL, 64, S_v);

            for (uint32_t j = 0; j < S_v; ++j) {
                for (uint32_t i = 0; i < S_v; i += 64) {
                    HVX_Vector v0 = hvx_vmemu(vtcm_s_state + j * S_v + i + 0);
                    HVX_Vector v1 = (i + 32 < S_v) ? hvx_vmemu(vtcm_s_state + j * S_v + i + 32) : Q6_V_vzero();
                    hvx_vmemu(vtcm_s_f16 + j * S_v + i) = hvx_vec_f32_to_f16(v0, v1);
                }
            }
            hmx_interleave_rows_to_tiles(vtcm_s_col_tiles, vtcm_s_f16, S_v, S_v, S_v, 0, S_v);

            gdn_f32_to_hmx_row_tiles(vtcm_k_prime_row_tiles, vtcm_k_f32, lambda_init, 64, S_v);

            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) c);

            htp_gdn_run_hmx_gemm(hmx_q, &gemm_job, vtcm_k_row_tiles, vtcm_k_col_tiles, vtcm_kk_tiles, 2, 2, n_sv_tiles, vtcm_scales_1);
            htp_gdn_run_hmx_gemm(hmx_q, &gemm_job, vtcm_q_row_tiles, vtcm_k_col_tiles, vtcm_qk_tiles, 2, 2, n_sv_tiles, vtcm_scales_1);
            htp_gdn_run_hmx_gemm(hmx_q, &gemm_job, vtcm_k_prime_row_tiles, vtcm_s_col_tiles, vtcm_v_inter_tiles, 2, n_sv_tiles, n_sv_tiles, vtcm_scales_1);

            gdn_f32_to_hmx_row_tiles(vtcm_q_row_tiles, vtcm_q_f32, lambda_init, 64, S_v);
            htp_gdn_run_hmx_gemm(hmx_q, &gemm_job, vtcm_q_row_tiles, vtcm_s_col_tiles, vtcm_o_inter_tiles, 2, n_sv_tiles, n_sv_tiles, vtcm_scales_1);

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) c);

            gdn_unpack_64x64_tiles_to_vectors(rows_kk, vtcm_kk_tiles);
            gdn_unpack_64x64_tiles_to_vectors(rows_qk, vtcm_qk_tiles);

            gdn_build_inv_l_and_a(rows_inv, rows_a, rows_kk, rows_qk, decay_m, decay_a, vtcm_b_f32);

            gdn_pack_64x64_vectors_to_tiles(vtcm_inv_row_tiles, rows_inv);
            gdn_pack_64x64_vectors_to_tiles(vtcm_a_row_tiles, rows_a);

            gdn_unpack_64xS_tiles_to_f32(vtcm_v_inter_f32, vtcm_v_inter_tiles, S_v);

            for (uint32_t t = 0; t < 64; ++t) {
                HVX_Vector vb = hvx_vec_splat_f32(vtcm_b_f32[t]);
                for (uint32_t i = 0; i < S_v; i += 64) {
                    HVX_Vector vv0 = hvx_vmemu(vtcm_v_f32 + t * S_v + i + 0);
                    HVX_Vector vv1 = (i + 32 < S_v) ? hvx_vmemu(vtcm_v_f32 + t * S_v + i + 32) : Q6_V_vzero();
                    HVX_Vector vi0 = hvx_vmemu(vtcm_v_inter_f32 + t * S_v + i + 0);
                    HVX_Vector vi1 = (i + 32 < S_v) ? hvx_vmemu(vtcm_v_inter_f32 + t * S_v + i + 32) : Q6_V_vzero();

                    HVX_Vector vp0 = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(vv0, vi0), vb);
                    HVX_Vector vp1 = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(vv1, vi1), vb);

                    hvx_vmemu(vtcm_v_prime_f16 + t * S_v + i) = hvx_vec_f32_to_f16(vp0, vp1);
                }
            }

            hmx_interleave_cols_to_tiles(vtcm_v_prime_col_tiles, vtcm_v_prime_f16, 64, S_v, S_v, 2, 0, 64);

            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) c);

            htp_gdn_run_hmx_gemm(hmx_q, &gemm_job, vtcm_inv_row_tiles, vtcm_v_prime_col_tiles, vtcm_delta_tiles, 2, n_sv_tiles, 2, vtcm_scales_1);

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) c);

            gdn_unpack_64xS_tiles_to_f16(vtcm_delta_f16, vtcm_delta_tiles, S_v);

            hmx_interleave_cols_to_tiles(vtcm_delta_col_tiles, vtcm_delta_f16, 64, S_v, S_v, 2, 0, 64);

            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) c);

            htp_gdn_run_hmx_gemm(hmx_q, &gemm_job, vtcm_a_row_tiles, vtcm_delta_col_tiles, vtcm_o_intra_tiles, 2, n_sv_tiles, 2, vtcm_scales_1);

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) c);

            gdn_unpack_64xS_tiles_to_f32(vtcm_o_inter_f32, vtcm_o_inter_tiles, S_v);
            gdn_unpack_64xS_tiles_to_f32(vtcm_o_intra_f32, vtcm_o_intra_tiles, S_v);

            HVX_Vector vscale = hvx_vec_splat_f32(scale);
            for (uint32_t i = 0; i < 64 * S_v / 32; ++i) {
                HVX_Vector vi = hvx_vmemu(vtcm_o_inter_f32 + i * 32);
                HVX_Vector va = hvx_vmemu(vtcm_o_intra_f32 + i * 32);
                hvx_vmemu(vtcm_o_f32 + i * 32) = hvx_vec_mul_f32_f32(hvx_vec_add_f32_f32(vi, va), vscale);
            }

            for (uint32_t s = 0; s < 64; ++s) {
                float decay_s = (float) decay_a[63 * 64 + s];
                HVX_Vector vs = hvx_vec_splat_f16(decay_s);
                for (uint32_t i = 0; i < S_v; i += 64) {
                    HVX_Vector vd = hvx_vmemu(vtcm_delta_f16 + s * S_v + i);
                    hvx_vmemu(vtcm_d_f16 + s * S_v + i) = hvx_vec_mul_f16_f16(vd, vs);
                }
            }

            gdn_pack_d_t_row_tiles(vtcm_d_row_tiles, vtcm_d_f16, S_v, vtcm_m, vtcm_tmp);
            hmx_interleave_cols_to_tiles(vtcm_k_col_tiles_64x128, vtcm_k_f16, 64, S_v, S_v, 2, 0, 64);

            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) c);

            htp_gdn_run_hmx_gemm(hmx_q, &gemm_job, vtcm_d_row_tiles, vtcm_k_col_tiles_64x128, vtcm_s_update_tiles, n_sv_tiles, n_sv_tiles, 2, vtcm_scales_1);

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) c);

            gdn_unpack_SxS_tiles_to_f32(vtcm_s_update_f32, vtcm_s_update_tiles, S_v);

            HVX_Vector v_l_final = hvx_vec_splat_f32(lambda_init[63]);
            for (uint32_t i = 0; i < S_v * S_v / 32; ++i) {
                HVX_Vector vs_old = hvx_vmemu(vtcm_s_state + i * 32);
                HVX_Vector vsu    = hvx_vmemu(vtcm_s_update_f32 + i * 32);
                hvx_vmemu(vtcm_s_state + i * 32) = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(vs_old, v_l_final), vsu);
            }

            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) c);

            const dma_addr_t attn_chunk_dma = dst->data +
                ((uint64_t) iv3 * n_tokens * H + (uint64_t) t_chunk * H + iv1) * S_v * sizeof(float);

            dma_queue_push(dma_q, dma_make_data(attn_chunk_dma, vtcm_o_f32),
                           dst->nb[1], S_v * sizeof(float), S_v * sizeof(float), chunk_size);
            dma_queue_pop(dma_q);
        }

        const uint32_t t_rem_start = n_chunks * chunk_size;
        if (t_rem_start < n_tokens) {
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) n_chunks);
            for (uint32_t t = t_rem_start; t < n_tokens; ++t) {
                const float * q_t = (const float *) ((const uint8_t *) (uintptr_t) q->data +
                    (uint64_t) iq3 * q->nb[3] + (uint64_t) t * q->nb[2] + (uint64_t) iq1 * q->nb[1]);
                const float * k_t = (const float *) ((const uint8_t *) (uintptr_t) k->data +
                    (uint64_t) ik3 * k->nb[3] + (uint64_t) t * k->nb[2] + (uint64_t) ik1 * k->nb[1]);
                const float * v_t = (const float *) ((const uint8_t *) (uintptr_t) v->data +
                    (uint64_t) iv3 * v->nb[3] + (uint64_t) t * v->nb[2] + (uint64_t) iv1 * v->nb[1]);
                const float * g_t = (const float *) ((const uint8_t *) (uintptr_t) g->data +
                    (uint64_t) iv3 * g->nb[3] + (uint64_t) t * g->nb[2] + (uint64_t) iv1 * g->nb[1]);
                const float   b_t = *(const float *) ((const uint8_t *) (uintptr_t) beta->data +
                    (uint64_t) iv3 * beta->nb[3] + (uint64_t) t * beta->nb[2] + (uint64_t) iv1 * beta->nb[1]);

                gdn_step_scalar_f32(vtcm_s_state, vtcm_attn_rem, q_t, k_t, v_t, g_t, b_t, scale, S_v);

                float * dst_rem = (float *) (uintptr_t) dst->data +
                    ((uint64_t) iv3 * n_tokens * H + (uint64_t) t * H + iv1) * S_v;
                hvx_copy_f32_uu((uint8_t *) dst_rem, (const uint8_t *) vtcm_attn_rem, S_v);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) n_chunks);
        }

        dma_queue_push(dma_q, dma_make_data(state_out_dma, vtcm_s_state),
                       S_v * sizeof(float), S_v * sizeof(float), S_v * sizeof(float), S_v);
        dma_queue_pop(dma_q);
    }

    return HTP_STATUS_OK;
}

int op_gated_delta_net(struct htp_ops_context * octx) {
    const struct htp_tensor * q     = octx->src[0];
    const struct htp_tensor * k     = octx->src[1];
    const struct htp_tensor * v     = octx->src[2];
    const struct htp_tensor * g     = octx->src[3];
    const struct htp_tensor * beta  = octx->src[4];
    const struct htp_tensor * state = octx->src[5];
    const struct htp_tensor * dst   = octx->dst;

    if (q->type != HTP_TYPE_F32 || k->type != HTP_TYPE_F32 || v->type != HTP_TYPE_F32 ||
        g->type != HTP_TYPE_F32 || beta->type != HTP_TYPE_F32 || state->type != HTP_TYPE_F32 ||
        dst->type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t S_v      = v->ne[0];
    const uint32_t H        = v->ne[1];
    const uint32_t n_tokens = v->ne[2];
    const uint32_t n_seqs   = v->ne[3];
    const uint32_t K        = octx->op_params[0];

    if (S_v == 0 || S_v > HTP_GDN_MAX_SV || H == 0 || n_tokens == 0 || n_seqs == 0) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if ((g->ne[0] != 1 && g->ne[0] != S_v) || beta->ne[0] != 1) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (q->ne[0] != S_v || k->ne[0] != S_v || q->ne[1] == 0 || k->ne[1] == 0 ||
        q->ne[2] != n_tokens || k->ne[2] != n_tokens || q->ne[3] == 0 || k->ne[3] == 0 ||
        (n_seqs % q->ne[3]) != 0 || (n_seqs % k->ne[3]) != 0) {
        return HTP_STATUS_NO_SUPPORT;
    }
    // state holds s0 only: [S_v, S_v, H, n_seqs]
    if (state->ne[0] != S_v || state->ne[1] != S_v || state->ne[2] != H || state->ne[3] != n_seqs) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (dst->ne[0] != S_v * H || dst->ne[1] != n_tokens * n_seqs + S_v * n_seqs * K) {
        return HTP_STATUS_NO_SUPPORT;
    }

    for (int i = 0; i < 5; i++) {
        if (htp_tensor_is_extended(octx->src[i])) {
            return HTP_STATUS_NO_SUPPORT;
        }
    }
    if (htp_tensor_is_extended(octx->dst)) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (octx->dsts[1]) {
        const struct htp_tensor * dst_cache = octx->dsts[1];
        if (dst_cache->type != HTP_TYPE_F32 || htp_tensor_is_extended(dst_cache)) {
            return HTP_STATUS_NO_SUPPORT;
        }
    }

    const struct htp_gdn_kernel_params * kparams = (const struct htp_gdn_kernel_params *) octx->kernel_params;
    struct htp_gdn_kernel_params kparams_local;
    if (!kparams || kparams->S_v == 0) {
        const uint32_t rq3 = n_seqs / q->ne[3];
        const uint32_t rk3 = n_seqs / k->ne[3];
        const uint32_t total_rows = H * n_seqs;
        uint32_t n_threads = (total_rows < octx->n_threads) ? total_rows : octx->n_threads;
        if (n_threads == 0) {
            n_threads = 1;
        }

        memset(&kparams_local, 0, sizeof(kparams_local));
        kparams_local.n_threads           = n_threads;
        kparams_local.S_v                 = S_v;
        kparams_local.H                   = H;
        kparams_local.n_tokens            = n_tokens;
        kparams_local.n_seqs              = n_seqs;
        kparams_local.K                   = K;
        kparams_local.total_rows          = total_rows;
        kparams_local.rows_per_thread     = (total_rows + n_threads - 1) / n_threads;
        const bool can_use_hmx = (octx->ctx->hmx_enabled) &&
                                 (S_v % 64 == 0) &&
                                 (n_tokens >= HTP_GDN_CHUNK_SIZE) &&
                                 (g->ne[0] == 1) &&
                                 (K == 1);

        struct htp_gdn_vtcm_layout layout_local;
        if (can_use_hmx) {
            htp_gdn_hmx_vtcm_layout_build(&layout_local, S_v, HTP_GDN_CHUNK_SIZE, 1);
            if (layout_local.total_bytes <= octx->ctx->vtcm_size) {
                kparams_local.kernel_type = HTP_GDN_KERNEL_HMX_CHUNKED;
                kparams_local.chunk_size  = HTP_GDN_CHUNK_SIZE;
                kparams_local.n_chunks    = n_tokens / HTP_GDN_CHUNK_SIZE;
            } else {
                htp_gdn_vtcm_layout_build(&layout_local, S_v, n_threads);
            }
        } else {
            htp_gdn_vtcm_layout_build(&layout_local, S_v, n_threads);
        }
        kparams_local.state_aligned       = (uint32_t) layout_local.state_aligned;
        kparams_local.vtcm_per_thread     = (uint32_t) layout_local.bytes_per_thread;
        kparams_local.vtcm_size           = (uint32_t) layout_local.total_bytes;
        kparams_local.kda                 = (g->ne[0] == S_v) ? 1 : 0;
        kparams_local.scale               = 1.0f / sqrtf((float) S_v);
        kparams_local.state_seq_stride    = (uint32_t) (state->nb[3] / sizeof(float));
        kparams_local.state_size_per_snap = S_v * S_v * H * n_seqs;

        kparams_local.div_H         = init_fastdiv_values(H);
        kparams_local.div_q1        = init_fastdiv_values(q->ne[1]);
        kparams_local.div_k1        = init_fastdiv_values(k->ne[1]);
        kparams_local.div_rq3       = init_fastdiv_values(rq3);
        kparams_local.div_rk3       = init_fastdiv_values(rk3);
        kparams_local.div_n_threads = init_fastdiv_values(n_threads);

        kparams = &kparams_local;
    }

    const uint32_t total_rows = kparams->total_rows;
    uint32_t row_start = 0;
    uint32_t nrows     = total_rows;

    if (octx->op_params[1] != 0) {
        row_start = octx->op_params[1];
        nrows     = octx->op_params[2];
    }

    if (nrows == 0) {
        return HTP_STATUS_OK;
    }

    if (kparams->kernel_type == HTP_GDN_KERNEL_HMX_CHUNKED) {
        return gated_delta_net_f32_hmx_chunked(octx, kparams, row_start, nrows);
    }

    const uint32_t n_threads = (nrows < kparams->n_threads) ? nrows : kparams->n_threads;

    struct htp_gdn_context gctx;
    gctx.octx      = octx;
    gctx.kparams   = kparams;
    gctx.row_start = row_start;
    gctx.nrows     = nrows;
    gctx.vtcm_base = octx->ctx->vtcm_base;

    htp_gdn_vtcm_layout_build(&gctx.layout, S_v, n_threads);

    if (gctx.layout.total_bytes > octx->ctx->vtcm_size) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    FARF(HIGH, "gated-delta-net-f32: q(%ux%ux%ux%u) k(%ux%ux%ux%u) v(%ux%ux%ux%u) state(%ux%ux%ux%u) -> (%ux%ux%ux%u) : "
         "vtcm-size %zu n_threads %u\n",
         q->ne[0], q->ne[1], q->ne[2], q->ne[3],
         k->ne[0], k->ne[1], k->ne[2], k->ne[3],
         v->ne[0], v->ne[1], v->ne[2], v->ne[3],
         state->ne[0], state->ne[1], state->ne[2], state->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         gctx.layout.total_bytes, n_threads);

    if (n_tokens == 1) {
        work_queue_run(octx->ctx->work_queue, gated_delta_net_f32_tg_thread, &gctx, n_threads);
    } else {
        work_queue_run(octx->ctx->work_queue, gated_delta_net_f32_pp_thread, &gctx, n_threads);
    }

    return HTP_STATUS_OK;
}

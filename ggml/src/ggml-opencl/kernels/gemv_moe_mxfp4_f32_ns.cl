#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

#define QK_MXFP4 32
#define N_SIMDGROUP 4
#define SIMDGROUP_WIDTH 64

static inline half8 mxfp4_to_fp16_packed8(ushort2 fp4x8) {
    ushort2 fp16_packed_a_0, fp16_packed_b_0, bias_a, bias_b, sign_a, sign_b;
    fp16_packed_a_0.lo = (fp4x8.s0 << 9) & 0x0E00;
    fp16_packed_a_0.hi = (fp4x8.s0 << 5) & 0x0E00;
    fp16_packed_b_0.lo = (fp4x8.s0 << 1) & 0x0E00;
    fp16_packed_b_0.hi = (fp4x8.s0 >> 3) & 0x0E00;

    bias_a.lo = (fp16_packed_a_0.lo != 0) ? 0x3800 : 0x0;
    bias_a.hi = (fp16_packed_a_0.hi != 0) ? 0x3800 : 0x0;
    bias_b.lo = (fp16_packed_b_0.lo != 0) ? 0x3800 : 0x0;
    bias_b.hi = (fp16_packed_b_0.hi != 0) ? 0x3800 : 0x0;

    fp16_packed_a_0.lo = (fp16_packed_a_0.lo != 0x0200) ? fp16_packed_a_0.lo : 0x0;
    fp16_packed_a_0.hi = (fp16_packed_a_0.hi != 0x0200) ? fp16_packed_a_0.hi : 0x0;
    fp16_packed_b_0.lo = (fp16_packed_b_0.lo != 0x0200) ? fp16_packed_b_0.lo : 0x0;
    fp16_packed_b_0.hi = (fp16_packed_b_0.hi != 0x0200) ? fp16_packed_b_0.hi : 0x0;

    sign_a.lo = (fp4x8.s0 << 12) & 0x8000;
    sign_a.hi = (fp4x8.s0 << 8) & 0x8000;
    sign_b.lo = (fp4x8.s0 << 4) & 0x8000;
    sign_b.hi = fp4x8.s0 & 0x8000;

    fp16_packed_a_0 = sign_a + bias_a + fp16_packed_a_0;
    fp16_packed_b_0 = sign_b + bias_b + fp16_packed_b_0;

    ushort2 fp16_packed_a_1, fp16_packed_b_1;
    fp16_packed_a_1.lo = (fp4x8.s1 << 9) & 0x0E00;
    fp16_packed_a_1.hi = (fp4x8.s1 << 5) & 0x0E00;
    fp16_packed_b_1.lo = (fp4x8.s1 << 1) & 0x0E00;
    fp16_packed_b_1.hi = (fp4x8.s1 >> 3) & 0x0E00;

    bias_a.lo = (fp16_packed_a_1.lo != 0) ? 0x3800 : 0x0;
    bias_a.hi = (fp16_packed_a_1.hi != 0) ? 0x3800 : 0x0;
    bias_b.lo = (fp16_packed_b_1.lo != 0) ? 0x3800 : 0x0;
    bias_b.hi = (fp16_packed_b_1.hi != 0) ? 0x3800 : 0x0;

    fp16_packed_a_1.lo = (fp16_packed_a_1.lo != 0x0200) ? fp16_packed_a_1.lo : 0x0;
    fp16_packed_a_1.hi = (fp16_packed_a_1.hi != 0x0200) ? fp16_packed_a_1.hi : 0x0;
    fp16_packed_b_1.lo = (fp16_packed_b_1.lo != 0x0200) ? fp16_packed_b_1.lo : 0x0;
    fp16_packed_b_1.hi = (fp16_packed_b_1.hi != 0x0200) ? fp16_packed_b_1.hi : 0x0;

    sign_a.lo = (fp4x8.s1 << 12) & 0x8000;
    sign_a.hi = (fp4x8.s1 << 8) & 0x8000;
    sign_b.lo = (fp4x8.s1 << 4) & 0x8000;
    sign_b.hi = fp4x8.s1 & 0x8000;

    fp16_packed_a_1 = sign_a + bias_a + fp16_packed_a_1;
    fp16_packed_b_1 = sign_b + bias_b + fp16_packed_b_1;

    return as_half8((ushort8)(fp16_packed_a_0, fp16_packed_b_0, fp16_packed_a_1, fp16_packed_b_1));
}

static inline float e8m0_to_fp32(uchar x) {
    int bits;
    bits = (x == 0) ? 0x00400000 : ((uint) x << 23);
    return as_float(bits);
}


__attribute__((qcom_reqd_sub_group_size("half")))
__kernel void kernel_gemv_moe_mxfp4_f32_ns(
    __global uint * src0_q,
    __global uchar * src0_e,
    __read_only image1d_buffer_t src1,
    __global uint * src2,
    __global float * dst,
    ulong         offsetd,
    int           ne00,
    int           ne01,
    int           ne11
) {
    uint i01  = get_global_id(0);
    uint i20  = get_global_id(2);
    uint sgid = get_local_id(1);
    uint slid = get_sub_group_local_id();

    if (i01 >= ne01) {
        return;
    }

    uint i11 = i20 % ne11;

    uint expert_id = src2[i20];
    uint expert_offset = expert_id * ne00 * ne01 / 32;

    __private float sum = 0.0f; // each thread calculate partial sum of one output

    // loop along ne00 in block granularity, skip 4 blocks every iter
    for (uint ib00 = sgid; ib00 < (ne00 / QK_MXFP4); ib00 += N_SIMDGROUP) {

        // load one block of q
        uint4 regQ;
        uint block_offset = expert_offset * 4 + ib00 * ne01 * 4 + i01;

        regQ.s0 = src0_q[block_offset];
        regQ.s1 = src0_q[block_offset + ne01];
        regQ.s2 = src0_q[block_offset + ne01 * 2];
        regQ.s3 = src0_q[block_offset + ne01 * 3];

        uint offset = i11 * ne00 / 4 + ib00 * 8;

        half8 fp16x8 = mxfp4_to_fp16_packed8(as_ushort2(regQ.s0));

        float4 shared_y4;
        shared_y4 = read_imagef(src1, (offset + 0));
        float4 acc = shared_y4 * convert_float4(fp16x8.lo);

        shared_y4 = read_imagef(src1, (offset + 1));
        acc += shared_y4 * convert_float4(fp16x8.hi);

        fp16x8 = mxfp4_to_fp16_packed8(as_ushort2(regQ.s1));

        shared_y4 = read_imagef(src1, (offset + 2));
        acc += shared_y4 * convert_float4(fp16x8.lo);

        shared_y4 = read_imagef(src1, (offset + 3));
        acc += shared_y4 * convert_float4(fp16x8.hi);


        fp16x8 = mxfp4_to_fp16_packed8(as_ushort2(regQ.s2));

        shared_y4 = read_imagef(src1, (offset + 4));
        acc += shared_y4 * convert_float4(fp16x8.lo);

        shared_y4 = read_imagef(src1, (offset + 5));
        acc += shared_y4 * convert_float4(fp16x8.hi);


        fp16x8 = mxfp4_to_fp16_packed8(as_ushort2(regQ.s3));

        shared_y4 = read_imagef(src1, (offset + 6));
        acc += shared_y4 * convert_float4(fp16x8.lo);

        shared_y4 = read_imagef(src1, (offset + 7));
        acc += shared_y4 * convert_float4(fp16x8.hi);

        uchar regE = src0_e[ib00 * ne01 + i01 + expert_offset];
        sum += e8m0_to_fp32(regE) * ((acc.s0 + acc.s1) + (acc.s2 + acc.s3));
    }

    // reduction in local memory, assumes #subgroups=4
    __local float reduceLM[SIMDGROUP_WIDTH * (N_SIMDGROUP - 1)];
    if (sgid == 1) reduceLM[SIMDGROUP_WIDTH * 0 + slid] = sum;
    if (sgid == 2) reduceLM[SIMDGROUP_WIDTH * 1 + slid] = sum;
    if (sgid == 3) reduceLM[SIMDGROUP_WIDTH * 2 + slid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sgid == 0) sum += reduceLM[SIMDGROUP_WIDTH * 0 + slid];
    if (sgid == 0) sum += reduceLM[SIMDGROUP_WIDTH * 1 + slid];
    if (sgid == 0) sum += reduceLM[SIMDGROUP_WIDTH * 2 + slid];

    // 1 outputs per thread in subgroup 0
    if (sgid == 0) {
        dst = dst + (offsetd >> 2);
        dst[i01 + i20 * ne01] = sum;
    }

}

// --- Weight-as-texture variant of kernel_gemv_moe_mxfp4_f32_ns -----------------
// Byte-identical; the mxfp4 plane is read via image1d_buffer (texture cache)
// instead of a __global uint buffer. Mirrors the q4_K _wimg MoE decode GEMV.
// Opt path: GGML_OPENCL_MOE_DECODE_WIMG (default on X2E).
__attribute__((qcom_reqd_sub_group_size("half")))
__kernel void kernel_gemv_moe_mxfp4_f32_ns_wimg(
    __read_only image1d_buffer_t src0_q,
    __global uchar * src0_e,
    __read_only image1d_buffer_t src1,
    __global uint * src2,
    __global float * dst,
    ulong         offsetd,
    int           ne00,
    int           ne01,
    int           ne11
) {
    uint i01  = get_global_id(0);
    uint i20  = get_global_id(2);
    uint sgid = get_local_id(1);
    uint slid = get_sub_group_local_id();

    if (i01 >= ne01) {
        return;
    }

    uint i11 = i20 % ne11;

    uint expert_id = src2[i20];
    uint expert_offset = expert_id * ne00 * ne01 / 32;

    __private float sum = 0.0f;

    for (uint ib00 = sgid; ib00 < (ne00 / QK_MXFP4); ib00 += N_SIMDGROUP) {

        uint4 regQ;
        uint block_offset = expert_offset * 4 + ib00 * ne01 * 4 + i01;

        regQ.s0 = read_imageui(src0_q, (int)(block_offset)).x;
        regQ.s1 = read_imageui(src0_q, (int)(block_offset + ne01)).x;
        regQ.s2 = read_imageui(src0_q, (int)(block_offset + ne01 * 2)).x;
        regQ.s3 = read_imageui(src0_q, (int)(block_offset + ne01 * 3)).x;

        uint offset = i11 * ne00 / 4 + ib00 * 8;

        half8 fp16x8 = mxfp4_to_fp16_packed8(as_ushort2(regQ.s0));

        float4 shared_y4;
        shared_y4 = read_imagef(src1, (offset + 0));
        float4 acc = shared_y4 * convert_float4(fp16x8.lo);

        shared_y4 = read_imagef(src1, (offset + 1));
        acc += shared_y4 * convert_float4(fp16x8.hi);

        fp16x8 = mxfp4_to_fp16_packed8(as_ushort2(regQ.s1));

        shared_y4 = read_imagef(src1, (offset + 2));
        acc += shared_y4 * convert_float4(fp16x8.lo);

        shared_y4 = read_imagef(src1, (offset + 3));
        acc += shared_y4 * convert_float4(fp16x8.hi);

        fp16x8 = mxfp4_to_fp16_packed8(as_ushort2(regQ.s2));

        shared_y4 = read_imagef(src1, (offset + 4));
        acc += shared_y4 * convert_float4(fp16x8.lo);

        shared_y4 = read_imagef(src1, (offset + 5));
        acc += shared_y4 * convert_float4(fp16x8.hi);

        fp16x8 = mxfp4_to_fp16_packed8(as_ushort2(regQ.s3));

        shared_y4 = read_imagef(src1, (offset + 6));
        acc += shared_y4 * convert_float4(fp16x8.lo);

        shared_y4 = read_imagef(src1, (offset + 7));
        acc += shared_y4 * convert_float4(fp16x8.hi);

        uchar regE = src0_e[ib00 * ne01 + i01 + expert_offset];
        sum += e8m0_to_fp32(regE) * ((acc.s0 + acc.s1) + (acc.s2 + acc.s3));
    }

    __local float reduceLM[SIMDGROUP_WIDTH * (N_SIMDGROUP - 1)];
    if (sgid == 1) reduceLM[SIMDGROUP_WIDTH * 0 + slid] = sum;
    if (sgid == 2) reduceLM[SIMDGROUP_WIDTH * 1 + slid] = sum;
    if (sgid == 3) reduceLM[SIMDGROUP_WIDTH * 2 + slid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sgid == 0) sum += reduceLM[SIMDGROUP_WIDTH * 0 + slid];
    if (sgid == 0) sum += reduceLM[SIMDGROUP_WIDTH * 1 + slid];
    if (sgid == 0) sum += reduceLM[SIMDGROUP_WIDTH * 2 + slid];

    if (sgid == 0) {
        dst = dst + (offsetd >> 2);
        dst[i01 + i20 * ne01] = sum;
    }
}

// --- Fused separate gate/up MoE GEMV + per-expert bias + SWIGLU_OAI epilogue ----
// gpt-oss MoE FFN: two independent mul_mat_id over ffn_gate_exps / ffn_up_exps
// (each [K, n_ff, n_expert]), each followed by an add_id of a per-expert bias
// vector, then a single GLU(SWIGLU_OAI) over the two results. The graph is the 5
// consecutive nodes {MUL_MAT_ID(gate), ADD_ID(gate_b), MUL_MAT_ID(up), ADD_ID(up_b),
// GLU}. The per-op path materializes two n_ff-wide intermediates to global, re-reads
// them for the two bias adds, and re-reads again for the GLU. This kernel folds all
// of it into one GEMV epilogue: each work-item computes BOTH dot products for output
// row i01 (gate row i01 of ffn_gate_exps and up row i01 of ffn_up_exps, same expert),
// adds the per-expert bias to each, applies SWIGLU_OAI, and writes the n_ff-wide
// result -> one dispatch, no intermediates.
// Each dot product accumulates in the SAME per-block / cross-subgroup order as the
// standalone kernel_gemv_moe_mxfp4_f32_ns above, the bias add matches add_id, and the
// epilogue is the exact scalar expression from kernels/glu.cl kernel_swiglu_oai, so
// the output is BYTE-IDENTICAL to the per-op mul_mat_id + add_id + glu path.

// One row's mxfp4 dot product, body identical to the standalone kernel above.
// Q/E are the quant/scale planes of one weight; SUM is the subgroup accumulator.
#define MXFP4_MOE_ROW_DOT(SUM, Q, E)                                                      \
    for (uint ib00 = sgid; ib00 < (ne00 / QK_MXFP4); ib00 += N_SIMDGROUP) {              \
        uint4 regQ;                                                                       \
        uint block_offset = expert_offset * 4 + ib00 * ne01 * 4 + i01;                    \
        regQ.s0 = (Q)[block_offset];                                                       \
        regQ.s1 = (Q)[block_offset + ne01];                                               \
        regQ.s2 = (Q)[block_offset + ne01 * 2];                                           \
        regQ.s3 = (Q)[block_offset + ne01 * 3];                                           \
        uint offset = i11 * ne00 / 4 + ib00 * 8;                                          \
        half8 fp16x8 = mxfp4_to_fp16_packed8(as_ushort2(regQ.s0));                         \
        float4 shared_y4 = read_imagef(src1, (offset + 0));                               \
        float4 acc = shared_y4 * convert_float4(fp16x8.lo);                               \
        shared_y4 = read_imagef(src1, (offset + 1)); acc += shared_y4 * convert_float4(fp16x8.hi); \
        fp16x8 = mxfp4_to_fp16_packed8(as_ushort2(regQ.s1));                               \
        shared_y4 = read_imagef(src1, (offset + 2)); acc += shared_y4 * convert_float4(fp16x8.lo); \
        shared_y4 = read_imagef(src1, (offset + 3)); acc += shared_y4 * convert_float4(fp16x8.hi); \
        fp16x8 = mxfp4_to_fp16_packed8(as_ushort2(regQ.s2));                               \
        shared_y4 = read_imagef(src1, (offset + 4)); acc += shared_y4 * convert_float4(fp16x8.lo); \
        shared_y4 = read_imagef(src1, (offset + 5)); acc += shared_y4 * convert_float4(fp16x8.hi); \
        fp16x8 = mxfp4_to_fp16_packed8(as_ushort2(regQ.s3));                               \
        shared_y4 = read_imagef(src1, (offset + 6)); acc += shared_y4 * convert_float4(fp16x8.lo); \
        shared_y4 = read_imagef(src1, (offset + 7)); acc += shared_y4 * convert_float4(fp16x8.hi); \
        uchar regE = (E)[ib00 * ne01 + i01 + expert_offset];                              \
        SUM += e8m0_to_fp32(regE) * ((acc.s0 + acc.s1) + (acc.s2 + acc.s3));              \
    }

__attribute__((qcom_reqd_sub_group_size("half")))
__kernel void kernel_gemv_moe_mxfp4_f32_ns_glu(
    __global uint *              gate_q,
    __global uchar *             gate_e,
    __global uint *              up_q,
    __global uchar *             up_e,
    __global float *             gate_bias,
    ulong                        gate_bias_off,
    __global float *             up_bias,
    ulong                        up_bias_off,
    __read_only image1d_buffer_t src1,
    __global uint *              src2,
    __global float *             dst,
    ulong                        offsetd,
    int                          ne00,    // K
    int                          ne01,    // n_ff (output rows of each weight)
    int                          ne11,
    float                        alpha,
    float                        limit
) {
    uint i01  = get_global_id(0);    // output row in [0, ne01)
    uint i20  = get_global_id(2);
    uint sgid = get_local_id(1);
    uint slid = get_sub_group_local_id();

    if (i01 >= (uint)ne01) {
        return;
    }

    uint i11 = i20 % ne11;

    uint expert_id = src2[i20];
    uint expert_offset = expert_id * ne00 * ne01 / 32;

    __private float gate_sum = 0.0f;
    __private float up_sum   = 0.0f;

    MXFP4_MOE_ROW_DOT(gate_sum, gate_q, gate_e)
    MXFP4_MOE_ROW_DOT(up_sum,   up_q,   up_e)

    // Cross-subgroup reduction (assumes #subgroups=4), gate (x) + up (y) packed.
    __local float2 reduceLM[SIMDGROUP_WIDTH * (N_SIMDGROUP - 1)];
    if (sgid == 1) reduceLM[SIMDGROUP_WIDTH * 0 + slid] = (float2)(gate_sum, up_sum);
    if (sgid == 2) reduceLM[SIMDGROUP_WIDTH * 1 + slid] = (float2)(gate_sum, up_sum);
    if (sgid == 3) reduceLM[SIMDGROUP_WIDTH * 2 + slid] = (float2)(gate_sum, up_sum);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sgid == 0) {
        gate_sum += reduceLM[SIMDGROUP_WIDTH * 0 + slid].s0;
        up_sum   += reduceLM[SIMDGROUP_WIDTH * 0 + slid].s1;
        gate_sum += reduceLM[SIMDGROUP_WIDTH * 1 + slid].s0;
        up_sum   += reduceLM[SIMDGROUP_WIDTH * 1 + slid].s1;
        gate_sum += reduceLM[SIMDGROUP_WIDTH * 2 + slid].s0;
        up_sum   += reduceLM[SIMDGROUP_WIDTH * 2 + slid].s1;

        // per-expert bias (add_id): bias laid out [n_ff, n_expert], natural order
        gate_sum += gate_bias[(gate_bias_off >> 2) + (uint)expert_id * ne01 + i01];
        up_sum   += up_bias[(up_bias_off >> 2) + (uint)expert_id * ne01 + i01];

        // SWIGLU_OAI: x0=gate (clamped above), x1=up (clamped both sides)
        float x0 = min(gate_sum, limit);
        float x1 = max(min(up_sum, limit), -limit);
        float out_glu = x0 / (1.0f + exp(-x0 * alpha));
        out_glu = out_glu * (1.0f + x1);

        dst = dst + (offsetd >> 2);
        dst[i01 + i20 * ne01] = out_glu;
    }
}

// Weight-as-texture variant of MXFP4_MOE_ROW_DOT: Q is an image1d_buffer_t.
#define MXFP4_MOE_ROW_DOT_IMG(SUM, Q, E)                                                  \
    for (uint ib00 = sgid; ib00 < (ne00 / QK_MXFP4); ib00 += N_SIMDGROUP) {              \
        uint4 regQ;                                                                       \
        uint block_offset = expert_offset * 4 + ib00 * ne01 * 4 + i01;                    \
        regQ.s0 = read_imageui((Q), (int)(block_offset)).x;                               \
        regQ.s1 = read_imageui((Q), (int)(block_offset + ne01)).x;                        \
        regQ.s2 = read_imageui((Q), (int)(block_offset + ne01 * 2)).x;                    \
        regQ.s3 = read_imageui((Q), (int)(block_offset + ne01 * 3)).x;                    \
        uint offset = i11 * ne00 / 4 + ib00 * 8;                                          \
        half8 fp16x8 = mxfp4_to_fp16_packed8(as_ushort2(regQ.s0));                         \
        float4 shared_y4 = read_imagef(src1, (offset + 0));                               \
        float4 acc = shared_y4 * convert_float4(fp16x8.lo);                               \
        shared_y4 = read_imagef(src1, (offset + 1)); acc += shared_y4 * convert_float4(fp16x8.hi); \
        fp16x8 = mxfp4_to_fp16_packed8(as_ushort2(regQ.s1));                               \
        shared_y4 = read_imagef(src1, (offset + 2)); acc += shared_y4 * convert_float4(fp16x8.lo); \
        shared_y4 = read_imagef(src1, (offset + 3)); acc += shared_y4 * convert_float4(fp16x8.hi); \
        fp16x8 = mxfp4_to_fp16_packed8(as_ushort2(regQ.s2));                               \
        shared_y4 = read_imagef(src1, (offset + 4)); acc += shared_y4 * convert_float4(fp16x8.lo); \
        shared_y4 = read_imagef(src1, (offset + 5)); acc += shared_y4 * convert_float4(fp16x8.hi); \
        fp16x8 = mxfp4_to_fp16_packed8(as_ushort2(regQ.s3));                               \
        shared_y4 = read_imagef(src1, (offset + 6)); acc += shared_y4 * convert_float4(fp16x8.lo); \
        shared_y4 = read_imagef(src1, (offset + 7)); acc += shared_y4 * convert_float4(fp16x8.hi); \
        uchar regE = (E)[ib00 * ne01 + i01 + expert_offset];                              \
        SUM += e8m0_to_fp32(regE) * ((acc.s0 + acc.s1) + (acc.s2 + acc.s3));              \
    }

// Weight-as-texture variant of kernel_gemv_moe_mxfp4_f32_ns_glu (gate_q/up_q as
// images). Byte-identical; gpt-oss MoE decode. gate_e/up_e/bias stay buffers.
__attribute__((qcom_reqd_sub_group_size("half")))
__kernel void kernel_gemv_moe_mxfp4_f32_ns_glu_wimg(
    __read_only image1d_buffer_t gate_q,
    __global uchar *             gate_e,
    __read_only image1d_buffer_t up_q,
    __global uchar *             up_e,
    __global float *             gate_bias,
    ulong                        gate_bias_off,
    __global float *             up_bias,
    ulong                        up_bias_off,
    __read_only image1d_buffer_t src1,
    __global uint *              src2,
    __global float *             dst,
    ulong                        offsetd,
    int                          ne00,
    int                          ne01,
    int                          ne11,
    float                        alpha,
    float                        limit
) {
    uint i01  = get_global_id(0);
    uint i20  = get_global_id(2);
    uint sgid = get_local_id(1);
    uint slid = get_sub_group_local_id();

    if (i01 >= (uint)ne01) {
        return;
    }

    uint i11 = i20 % ne11;

    uint expert_id = src2[i20];
    uint expert_offset = expert_id * ne00 * ne01 / 32;

    __private float gate_sum = 0.0f;
    __private float up_sum   = 0.0f;

    MXFP4_MOE_ROW_DOT_IMG(gate_sum, gate_q, gate_e)
    MXFP4_MOE_ROW_DOT_IMG(up_sum,   up_q,   up_e)

    __local float2 reduceLM[SIMDGROUP_WIDTH * (N_SIMDGROUP - 1)];
    if (sgid == 1) reduceLM[SIMDGROUP_WIDTH * 0 + slid] = (float2)(gate_sum, up_sum);
    if (sgid == 2) reduceLM[SIMDGROUP_WIDTH * 1 + slid] = (float2)(gate_sum, up_sum);
    if (sgid == 3) reduceLM[SIMDGROUP_WIDTH * 2 + slid] = (float2)(gate_sum, up_sum);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sgid == 0) {
        gate_sum += reduceLM[SIMDGROUP_WIDTH * 0 + slid].s0;
        up_sum   += reduceLM[SIMDGROUP_WIDTH * 0 + slid].s1;
        gate_sum += reduceLM[SIMDGROUP_WIDTH * 1 + slid].s0;
        up_sum   += reduceLM[SIMDGROUP_WIDTH * 1 + slid].s1;
        gate_sum += reduceLM[SIMDGROUP_WIDTH * 2 + slid].s0;
        up_sum   += reduceLM[SIMDGROUP_WIDTH * 2 + slid].s1;

        gate_sum += gate_bias[(gate_bias_off >> 2) + (uint)expert_id * ne01 + i01];
        up_sum   += up_bias[(up_bias_off >> 2) + (uint)expert_id * ne01 + i01];

        float x0 = min(gate_sum, limit);
        float x1 = max(min(up_sum, limit), -limit);
        float out_glu = x0 / (1.0f + exp(-x0 * alpha));
        out_glu = out_glu * (1.0f + x1);

        dst = dst + (offsetd >> 2);
        dst[i01 + i20 * ne01] = out_glu;
    }
}

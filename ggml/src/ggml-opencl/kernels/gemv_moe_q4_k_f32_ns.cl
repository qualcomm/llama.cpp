#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

#define QK_K 256
#define K_SCALE_SIZE 12
#define N_SIMDGROUP 4
#define SIMDGROUP_WIDTH 64

inline void get_scale_min_k4(
    int j,
    global const uchar * q,
    uchar * d,
    uchar * m
) {
    if (j < 4) {
        *d = q[j]   & 63;
        *m = q[j+4] & 63;
    } else {
        *d = (q[j+4] & 0x0F) | ((q[j-4] & 0xC0) >> 2);
        *m = ((q[j+4] >> 4) & 0x0F) | ((q[j]   & 0xC0) >> 2);
    }
}

static inline float8 q4_k_to_fp32_packed8(ushort2 q4x8, float scale, float minv) {
    float8 fp32x8;
    fp32x8.s0 = (q4x8.s0 & 0x000F) * scale - minv;
    fp32x8.s1 = ((q4x8.s0 & 0x00F0) >> 4) * scale - minv;
    fp32x8.s2 = ((q4x8.s0 & 0x0F00) >> 8) * scale - minv;
    fp32x8.s3 = ((q4x8.s0 & 0xF000) >> 12) * scale - minv;
    fp32x8.s4 = (q4x8.s1 & 0x000F) * scale - minv;
    fp32x8.s5 = ((q4x8.s1 & 0x00F0) >> 4) * scale - minv;
    fp32x8.s6 = ((q4x8.s1 & 0x0F00) >> 8) * scale - minv;
    fp32x8.s7 = ((q4x8.s1 & 0xF000) >> 12) * scale - minv;
    return fp32x8;
}

__attribute__((qcom_reqd_sub_group_size("half")))
__kernel void kernel_gemv_moe_q4_k_f32_ns(
    __global uint *         src0_q,
    __global half *         src0_d,
    __global half *         src0_dm,
    __global uchar *        src0_s,
    __read_only image1d_buffer_t src1,
    __global uint *         src2,
    __global float *        dst,
    ulong                   offsetd,
    int                     ne00,
    int                     ne01,
    int                     ne11
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

    int num_superblocks = ne00 / QK_K;
    int num_subblocks = ne00 / 32;
    int scales_per_row = num_superblocks * K_SCALE_SIZE;

    // Expert offsets in the transposed noshuffle layout
    uint expert_q_offset = expert_id * (ne00 / 8) * ne01;
    uint expert_d_offset = expert_id * num_superblocks * ne01;

    __private float sum = 0.0f;

    // Loop over sub-blocks of 32 elements, N_SIMDGROUP sub-blocks per iter
    for (uint ib = sgid; ib < num_subblocks; ib += N_SIMDGROUP) {
        uint sb = ib / 8;
        uint j  = ib % 8;

        // Load d and dmin for this super-block
        half d_val   = src0_d[expert_d_offset + sb * ne01 + i01];
        half dm_val  = src0_dm[expert_d_offset + sb * ne01 + i01];

        // Load sub-block scale and min
        global const uchar * sc = src0_s + (expert_id * ne01 + i01) * scales_per_row + sb * K_SCALE_SIZE;
        uchar sv, mn;
        get_scale_min_k4(j, sc, &sv, &mn);

        float scale = (float)d_val * (float)sv;
        float minv  = (float)dm_val * (float)mn;

        // Load 4 uints of quants (32 nibbles = 32 elements)
        uint q_base = expert_q_offset + ib * ne01 * 4 + i01;

        uint4 regQ;
        regQ.s0 = src0_q[q_base];
        regQ.s1 = src0_q[q_base + ne01];
        regQ.s2 = src0_q[q_base + ne01 * 2];
        regQ.s3 = src0_q[q_base + ne01 * 3];

        // Load activations: 32 floats = 8 float4s
        uint y_offset = i11 * ne00 / 4 + ib * 8;

        float8 fp32x8 = q4_k_to_fp32_packed8(as_ushort2(regQ.s0), scale, minv);

        float4 shared_y4;
        shared_y4 = read_imagef(src1, (y_offset + 0));
        float4 acc = shared_y4 * fp32x8.lo;

        shared_y4 = read_imagef(src1, (y_offset + 1));
        acc += shared_y4 * fp32x8.hi;

        fp32x8 = q4_k_to_fp32_packed8(as_ushort2(regQ.s1), scale, minv);

        shared_y4 = read_imagef(src1, (y_offset + 2));
        acc += shared_y4 * fp32x8.lo;

        shared_y4 = read_imagef(src1, (y_offset + 3));
        acc += shared_y4 * fp32x8.hi;

        fp32x8 = q4_k_to_fp32_packed8(as_ushort2(regQ.s2), scale, minv);

        shared_y4 = read_imagef(src1, (y_offset + 4));
        acc += shared_y4 * fp32x8.lo;

        shared_y4 = read_imagef(src1, (y_offset + 5));
        acc += shared_y4 * fp32x8.hi;

        fp32x8 = q4_k_to_fp32_packed8(as_ushort2(regQ.s3), scale, minv);

        shared_y4 = read_imagef(src1, (y_offset + 6));
        acc += shared_y4 * fp32x8.lo;

        shared_y4 = read_imagef(src1, (y_offset + 7));
        acc += shared_y4 * fp32x8.hi;

        sum += ((acc.s0 + acc.s1) + (acc.s2 + acc.s3));
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

    // 1 output per thread in subgroup 0
    if (sgid == 0) {
        dst = dst + (offsetd >> 2);
        dst[i01 + i20 * ne01] = sum;
    }
}

// --- Weight-as-texture variant of kernel_gemv_moe_q4_k_f32_ns ------------------
// Byte-identical math; the only change is that the q4_K nibble plane is read
// through an image1d_buffer (read_imageui -> texture/L1 cache) instead of a plain
// __global uint buffer. Dense decode GEMVs (gemv_noshuffle_*) and the MoE prefill
// dp4a GEMMs already read weights this way; the MoE decode GEMV was the one path
// still on a host buffer. src0_q is the same CL_R/UINT32 q_img used by prefill
// (1 uint/texel, index == the buffer uint index). Opt-in: GGML_OPENCL_MOE_DECODE_WIMG.
__attribute__((qcom_reqd_sub_group_size("half")))
__kernel void kernel_gemv_moe_q4_k_f32_ns_wimg(
    __read_only image1d_buffer_t src0_q,
    __global half *         src0_d,
    __global half *         src0_dm,
    __global uchar *        src0_s,
    __read_only image1d_buffer_t src1,
    __global uint *         src2,
    __global float *        dst,
    ulong                   offsetd,
    int                     ne00,
    int                     ne01,
    int                     ne11
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

    int num_superblocks = ne00 / QK_K;
    int num_subblocks = ne00 / 32;
    int scales_per_row = num_superblocks * K_SCALE_SIZE;

    uint expert_q_offset = expert_id * (ne00 / 8) * ne01;
    uint expert_d_offset = expert_id * num_superblocks * ne01;

    __private float sum = 0.0f;

    for (uint ib = sgid; ib < num_subblocks; ib += N_SIMDGROUP) {
        uint sb = ib / 8;
        uint j  = ib % 8;

        half d_val   = src0_d[expert_d_offset + sb * ne01 + i01];
        half dm_val  = src0_dm[expert_d_offset + sb * ne01 + i01];

        global const uchar * sc = src0_s + (expert_id * ne01 + i01) * scales_per_row + sb * K_SCALE_SIZE;
        uchar sv, mn;
        get_scale_min_k4(j, sc, &sv, &mn);

        float scale = (float)d_val * (float)sv;
        float minv  = (float)dm_val * (float)mn;

        uint q_base = expert_q_offset + ib * ne01 * 4 + i01;

        uint4 regQ;
        regQ.s0 = read_imageui(src0_q, (int)(q_base)).x;
        regQ.s1 = read_imageui(src0_q, (int)(q_base + ne01)).x;
        regQ.s2 = read_imageui(src0_q, (int)(q_base + ne01 * 2)).x;
        regQ.s3 = read_imageui(src0_q, (int)(q_base + ne01 * 3)).x;

        uint y_offset = i11 * ne00 / 4 + ib * 8;

        float8 fp32x8 = q4_k_to_fp32_packed8(as_ushort2(regQ.s0), scale, minv);

        float4 shared_y4;
        shared_y4 = read_imagef(src1, (y_offset + 0));
        float4 acc = shared_y4 * fp32x8.lo;

        shared_y4 = read_imagef(src1, (y_offset + 1));
        acc += shared_y4 * fp32x8.hi;

        fp32x8 = q4_k_to_fp32_packed8(as_ushort2(regQ.s1), scale, minv);

        shared_y4 = read_imagef(src1, (y_offset + 2));
        acc += shared_y4 * fp32x8.lo;

        shared_y4 = read_imagef(src1, (y_offset + 3));
        acc += shared_y4 * fp32x8.hi;

        fp32x8 = q4_k_to_fp32_packed8(as_ushort2(regQ.s2), scale, minv);

        shared_y4 = read_imagef(src1, (y_offset + 4));
        acc += shared_y4 * fp32x8.lo;

        shared_y4 = read_imagef(src1, (y_offset + 5));
        acc += shared_y4 * fp32x8.hi;

        fp32x8 = q4_k_to_fp32_packed8(as_ushort2(regQ.s3), scale, minv);

        shared_y4 = read_imagef(src1, (y_offset + 6));
        acc += shared_y4 * fp32x8.lo;

        shared_y4 = read_imagef(src1, (y_offset + 7));
        acc += shared_y4 * fp32x8.hi;

        sum += ((acc.s0 + acc.s1) + (acc.s2 + acc.s3));
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

// --- Fused combined-gate_up MoE GEMV + GLU epilogue ----------------------------
// Gemma-4 (and similar) MoE FFNs fuse the gate and up projections into a SINGLE
// expert weight `ffn_gate_up_exps` of width 2*n_ff (rows [0,n_ff) = gate,
// [n_ff,2*n_ff) = up). The graph is {MUL_MAT_ID, VIEW(gate), VIEW(up), GLU}: one
// mul_mat_id over the combined weight, then a GLU on the two row-halves. The
// per-op path materializes the full 2*n_ff-wide intermediate to global and the
// GLU re-reads it. This kernel folds the GLU into the GEMV epilogue: each WI
// computes BOTH dot products (output row i01 = gate row i01 and up row i01+n_ff
// of the same expert weight), applies the GLU activation, and writes only the
// n_ff-wide result -> one dispatch, no 2*n_ff intermediate, half the dst writes.
// The two dot products each accumulate in the SAME per-subblock and cross-
// subgroup order as the standalone kernel_gemv_moe_q4_k_f32_ns, and the GLU
// formula is the exact scalar expression from kernels/glu.cl, so the output is
// BYTE-IDENTICAL to the per-op mul_mat_id + glu path.
//   glu_op: REGLU=0, GEGLU=1, SWIGLU=2, GEGLU_ERF=4, GEGLU_QUICK=5 (ggml_glu_op).
#define GLU_GEGLU_COEF_A      0.044715f
#define GLU_SQRT_2_OVER_PI    0.79788456080286535587989211986876f
#define GLU_SQRT_2_INV        0.70710678118654752440084436210484f
#define GLU_QUICK_COEF       -1.702f

static inline float moe_glu_apply(int glu_op, float g, float u) {
    float act;
    if (glu_op == 1) {        // GEGLU (tanh-approx gelu)
        act = 0.5f*g*(1.0f + tanh(GLU_SQRT_2_OVER_PI*g*(1.0f + GLU_GEGLU_COEF_A*g*g)));
    } else if (glu_op == 2) { // SWIGLU (silu)
        act = g / (1.0f + exp(-g));
    } else if (glu_op == 0) { // REGLU
        return g*u*(g > 0.0f);
    } else if (glu_op == 4) { // GEGLU_ERF
        act = 0.5f*g*(1.0f + erf(g*GLU_SQRT_2_INV));
    } else {                  // GEGLU_QUICK (glu_op == 5)
        act = g*(1.0f/(1.0f + exp(GLU_QUICK_COEF*g)));
    }
    return act*u;
}

// One row's dot product, identical body to the standalone kernel above. ROW is the
// physical weight row (0..ne01-1); SUM is the float accumulator for this subgroup.
#define MOE_Q4K_ROW_DOT(SUM, ROW)                                                            \
    for (uint ib = sgid; ib < num_subblocks; ib += N_SIMDGROUP) {                            \
        uint sb = ib / 8;                                                                    \
        uint j  = ib % 8;                                                                    \
        half d_val  = src0_d[expert_d_offset + sb * ne01 + (ROW)];                           \
        half dm_val = src0_dm[expert_d_offset + sb * ne01 + (ROW)];                          \
        global const uchar * sc = src0_s + ((uint)expert_id * ne01 + (ROW)) * scales_per_row + sb * K_SCALE_SIZE; \
        uchar sv, mn;                                                                        \
        get_scale_min_k4(j, sc, &sv, &mn);                                                   \
        float scale = (float)d_val * (float)sv;                                              \
        float minv  = (float)dm_val * (float)mn;                                             \
        uint q_base = expert_q_offset + ib * ne01 * 4 + (ROW);                               \
        uint4 regQ;                                                                          \
        regQ.s0 = src0_q[q_base];                                                            \
        regQ.s1 = src0_q[q_base + ne01];                                                     \
        regQ.s2 = src0_q[q_base + ne01 * 2];                                                 \
        regQ.s3 = src0_q[q_base + ne01 * 3];                                                 \
        uint y_offset = i11 * ne00 / 4 + ib * 8;                                             \
        float8 fp32x8 = q4_k_to_fp32_packed8(as_ushort2(regQ.s0), scale, minv);             \
        float4 shared_y4;                                                                    \
        shared_y4 = read_imagef(src1, (y_offset + 0)); float4 acc = shared_y4 * fp32x8.lo;   \
        shared_y4 = read_imagef(src1, (y_offset + 1)); acc += shared_y4 * fp32x8.hi;         \
        fp32x8 = q4_k_to_fp32_packed8(as_ushort2(regQ.s1), scale, minv);                     \
        shared_y4 = read_imagef(src1, (y_offset + 2)); acc += shared_y4 * fp32x8.lo;         \
        shared_y4 = read_imagef(src1, (y_offset + 3)); acc += shared_y4 * fp32x8.hi;         \
        fp32x8 = q4_k_to_fp32_packed8(as_ushort2(regQ.s2), scale, minv);                     \
        shared_y4 = read_imagef(src1, (y_offset + 4)); acc += shared_y4 * fp32x8.lo;         \
        shared_y4 = read_imagef(src1, (y_offset + 5)); acc += shared_y4 * fp32x8.hi;         \
        fp32x8 = q4_k_to_fp32_packed8(as_ushort2(regQ.s3), scale, minv);                     \
        shared_y4 = read_imagef(src1, (y_offset + 6)); acc += shared_y4 * fp32x8.lo;         \
        shared_y4 = read_imagef(src1, (y_offset + 7)); acc += shared_y4 * fp32x8.hi;         \
        SUM += ((acc.s0 + acc.s1) + (acc.s2 + acc.s3));                                      \
    }

__attribute__((qcom_reqd_sub_group_size("half")))
__kernel void kernel_gemv_moe_q4_k_f32_ns_glu(
    __global uint *         src0_q,
    __global half *         src0_d,
    __global half *         src0_dm,
    __global uchar *        src0_s,
    __read_only image1d_buffer_t src1,
    __global uint *         src2,
    __global float *        dst,
    ulong                   offsetd,
    int                     ne00,    // K
    int                     ne01,    // combined output rows = 2*n_ff
    int                     ne11,
    int                     n_ff,    // GLU output rows (= ne01/2)
    int                     glu_op
) {
    uint i01  = get_global_id(0);    // GLU output row in [0, n_ff)
    uint i20  = get_global_id(2);
    uint sgid = get_local_id(1);
    uint slid = get_sub_group_local_id();

    if (i01 >= (uint)n_ff) {
        return;
    }

    uint i11 = i20 % ne11;

    uint expert_id = src2[i20];

    int num_superblocks = ne00 / QK_K;
    int num_subblocks = ne00 / 32;
    int scales_per_row = num_superblocks * K_SCALE_SIZE;

    // Expert offsets in the transposed noshuffle layout (combined-width rows).
    uint expert_q_offset = expert_id * (ne00 / 8) * ne01;
    uint expert_d_offset = expert_id * num_superblocks * ne01;

    __private float gate_sum = 0.0f;
    __private float up_sum   = 0.0f;

    MOE_Q4K_ROW_DOT(gate_sum, i01)
    MOE_Q4K_ROW_DOT(up_sum,   i01 + (uint)n_ff)

    // Cross-subgroup reduction (assumes #subgroups=4), gate (xy) + up (zw) packed.
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

        dst = dst + (offsetd >> 2);
        dst[i01 + i20 * n_ff] = moe_glu_apply(glu_op, gate_sum, up_sum);
    }
}

// Weight-as-texture variant of MOE_Q4K_ROW_DOT: src0_q is an image1d_buffer_t.
#define MOE_Q4K_ROW_DOT_IMG(SUM, ROW)                                                        \
    for (uint ib = sgid; ib < num_subblocks; ib += N_SIMDGROUP) {                            \
        uint sb = ib / 8;                                                                    \
        uint j  = ib % 8;                                                                    \
        half d_val  = src0_d[expert_d_offset + sb * ne01 + (ROW)];                           \
        half dm_val = src0_dm[expert_d_offset + sb * ne01 + (ROW)];                          \
        global const uchar * sc = src0_s + ((uint)expert_id * ne01 + (ROW)) * scales_per_row + sb * K_SCALE_SIZE; \
        uchar sv, mn;                                                                        \
        get_scale_min_k4(j, sc, &sv, &mn);                                                   \
        float scale = (float)d_val * (float)sv;                                              \
        float minv  = (float)dm_val * (float)mn;                                             \
        uint q_base = expert_q_offset + ib * ne01 * 4 + (ROW);                               \
        uint4 regQ;                                                                          \
        regQ.s0 = read_imageui(src0_q, (int)(q_base)).x;                                     \
        regQ.s1 = read_imageui(src0_q, (int)(q_base + ne01)).x;                              \
        regQ.s2 = read_imageui(src0_q, (int)(q_base + ne01 * 2)).x;                          \
        regQ.s3 = read_imageui(src0_q, (int)(q_base + ne01 * 3)).x;                          \
        uint y_offset = i11 * ne00 / 4 + ib * 8;                                             \
        float8 fp32x8 = q4_k_to_fp32_packed8(as_ushort2(regQ.s0), scale, minv);             \
        float4 shared_y4;                                                                    \
        shared_y4 = read_imagef(src1, (y_offset + 0)); float4 acc = shared_y4 * fp32x8.lo;   \
        shared_y4 = read_imagef(src1, (y_offset + 1)); acc += shared_y4 * fp32x8.hi;         \
        fp32x8 = q4_k_to_fp32_packed8(as_ushort2(regQ.s1), scale, minv);                     \
        shared_y4 = read_imagef(src1, (y_offset + 2)); acc += shared_y4 * fp32x8.lo;         \
        shared_y4 = read_imagef(src1, (y_offset + 3)); acc += shared_y4 * fp32x8.hi;         \
        fp32x8 = q4_k_to_fp32_packed8(as_ushort2(regQ.s2), scale, minv);                     \
        shared_y4 = read_imagef(src1, (y_offset + 4)); acc += shared_y4 * fp32x8.lo;         \
        shared_y4 = read_imagef(src1, (y_offset + 5)); acc += shared_y4 * fp32x8.hi;         \
        fp32x8 = q4_k_to_fp32_packed8(as_ushort2(regQ.s3), scale, minv);                     \
        shared_y4 = read_imagef(src1, (y_offset + 6)); acc += shared_y4 * fp32x8.lo;         \
        shared_y4 = read_imagef(src1, (y_offset + 7)); acc += shared_y4 * fp32x8.hi;         \
        SUM += ((acc.s0 + acc.s1) + (acc.s2 + acc.s3));                                      \
    }

// Weight-as-texture variant of kernel_gemv_moe_q4_k_f32_ns_glu (src0_q as image).
// Byte-identical; Gemma-4 combined gate_up MoE decode.
__attribute__((qcom_reqd_sub_group_size("half")))
__kernel void kernel_gemv_moe_q4_k_f32_ns_glu_wimg(
    __read_only image1d_buffer_t src0_q,
    __global half *         src0_d,
    __global half *         src0_dm,
    __global uchar *        src0_s,
    __read_only image1d_buffer_t src1,
    __global uint *         src2,
    __global float *        dst,
    ulong                   offsetd,
    int                     ne00,
    int                     ne01,
    int                     ne11,
    int                     n_ff,
    int                     glu_op
) {
    uint i01  = get_global_id(0);
    uint i20  = get_global_id(2);
    uint sgid = get_local_id(1);
    uint slid = get_sub_group_local_id();

    if (i01 >= (uint)n_ff) {
        return;
    }

    uint i11 = i20 % ne11;

    uint expert_id = src2[i20];

    int num_superblocks = ne00 / QK_K;
    int num_subblocks = ne00 / 32;
    int scales_per_row = num_superblocks * K_SCALE_SIZE;

    uint expert_q_offset = expert_id * (ne00 / 8) * ne01;
    uint expert_d_offset = expert_id * num_superblocks * ne01;

    __private float gate_sum = 0.0f;
    __private float up_sum   = 0.0f;

    MOE_Q4K_ROW_DOT_IMG(gate_sum, i01)
    MOE_Q4K_ROW_DOT_IMG(up_sum,   i01 + (uint)n_ff)

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

        dst = dst + (offsetd >> 2);
        dst[i01 + i20 * n_ff] = moe_glu_apply(glu_op, gate_sum, up_sum);
    }
}

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#else
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

#ifdef cl_intel_required_subgroup_size
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#define INTEL_GPU 1
#define REQD_SUBGROUP_SIZE_16 __attribute__((intel_reqd_sub_group_size(16)))
#define REQD_SUBGROUP_SIZE_32 __attribute__((intel_reqd_sub_group_size(32)))
#elif defined(cl_qcom_reqd_sub_group_size)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#endif

// Assumes row size (ne00) is a multiple of 4
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_f16_f32_l4(
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src0 = (global char*)((global char*)src0 + offset0);
    src1 = (global char*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    int nrows = ne11;
    int r0 = get_group_id(0);
    int im = get_group_id(2);

    int i12 = im%ne12;
    int i13 = im/ne12;

    ulong offset_src0 = r0*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;

    global half4 * x4 = (global half4 *) (src0 + offset_src0);

    for (int r1 = 0; r1 < nrows; ++r1) {
        ulong offset_src1 = r1*nb11 + (i12   )*nb12 + (i13   )*nb13;

        global float4 * y4 = (global float4 *) (src1 + offset_src1);

        float sumf = 0;
        for (int i = get_sub_group_local_id(); i < ne00/4; i += get_max_sub_group_size()) {
            sumf += convert_float(x4[i].s0) * y4[i].s0;
            sumf += convert_float(x4[i].s1) * y4[i].s1;
            sumf += convert_float(x4[i].s2) * y4[i].s2;
            sumf += convert_float(x4[i].s3) * y4[i].s3;
        }

        float all_sum = sub_group_reduce_add(sumf);
        if (get_sub_group_local_id() == 0) {
            dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
        }
    }
}

// Each subgroup produces DR_NDST outputs, assumes ne11 == 1
#define MUL_MAT_F16_F32_L4_DR_NDST 4

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_f16_f32_l4_dr(
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src0 = (global char*)((global char*)src0 + offset0);
    src1 = (global char*)((global char*)src1 + offset1);
    dst  = (global float*)((global char*)dst  + offsetd);

    const int r0_base = get_group_id(0) * MUL_MAT_F16_F32_L4_DR_NDST;
    const int im      = get_group_id(2);

    const int i12 = im % ne12;
    const int i13 = im / ne12;

    // assume ne11 == 1
    const ulong offset_src1 = i12*nb12 + i13*nb13;
    global float4 * y4 = (global float4 *)(src1 + offset_src1);

    global half4 * x4[MUL_MAT_F16_F32_L4_DR_NDST];
    float          sumf[MUL_MAT_F16_F32_L4_DR_NDST];

    const ulong   k_head_off = (i12/r2)*nb02 + (i13/r3)*nb03;

    #pragma unroll
    for (int n = 0; n < MUL_MAT_F16_F32_L4_DR_NDST; ++n) {
        int       r0   = r0_base + n;
        int       r0c  = r0 < ne01 ? r0 : 0;
        ulong     off  = (ulong)r0c*nb01 + k_head_off;
        x4[n]   = (global half4 *)(src0 + off);
        sumf[n] = 0.0f;
    }

    const int n_chunks = ne00 / 4;
    const int sg_size  = get_max_sub_group_size();
    const int lid      = get_sub_group_local_id();

    for (int i = lid; i < n_chunks; i += sg_size) {
        float4 q = y4[i];
        #pragma unroll
        for (int n = 0; n < MUL_MAT_F16_F32_L4_DR_NDST; ++n) {
            float4 k = convert_float4(x4[n][i]);
            sumf[n] = mad(k.s0, q.s0, sumf[n]);
            sumf[n] = mad(k.s1, q.s1, sumf[n]);
            sumf[n] = mad(k.s2, q.s2, sumf[n]);
            sumf[n] = mad(k.s3, q.s3, sumf[n]);
        }
    }

    #pragma unroll
    for (int n = 0; n < MUL_MAT_F16_F32_L4_DR_NDST; ++n) {
        float reduced = sub_group_reduce_add(sumf[n]);
        int   r0      = r0_base + n;
        if (lid == 0 && r0 < ne01) {
            dst[im*ne1*ne0 + r0] = reduced;
        }
    }
}

// Kernels for decoding, Adreno only for now
#define MUL_MAT_F16_F32_L4_DR_LS_R2_MAX 8

#ifdef ADRENO_GPU
#pragma OPENCL EXTENSION cl_qcom_subgroup_shuffle : enable
#define sub_group_shuffle_xor(val, mask) qcom_sub_group_shuffle_xor((val), (mask), CLK_SUB_GROUP_SHUFFLE_WIDTH_WAVE_SIZE_QCOM, 0.0f)

REQD_SUBGROUP_SIZE_64
kernel void kernel_mul_mat_f16_f32_l4_dr_ls(
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src0 = (global char*)((global char*)src0 + offset0);
    src1 = (global char*)((global char*)src1 + offset1);
    dst  = (global float*)((global char*)dst  + offsetd);

    const int r0_base = get_group_id(0) * 2;
    const int kv_grp  = get_group_id(2);   // KV head group; im = kv_grp*r2 + q

    const int i12_kv = kv_grp % ne02;
    const int i13_kv = kv_grp / ne02;

    const int lid     = get_sub_group_local_id();
    const int subhalf = lid >> 5;          // 0 or 1 (which K row in the WG)
    const int intra   = lid & 31;          // 0..31 (lane within the half)

    const int r0  = r0_base + subhalf;
    const int r0c = r0 < ne01 ? r0 : 0;    // clamp OOB to row 0; skip write below

    // K row pointer for this lane (one K row per half-wave).
    const ulong k_off = (ulong)r0c*nb01 + (ulong)i12_kv*nb02 + (ulong)i13_kv*nb03;
    global half4 * x4 = (global half4 *)(src0 + k_off);

    global float4 * y4[MUL_MAT_F16_F32_L4_DR_LS_R2_MAX];
    #pragma unroll
    for (int q = 0; q < MUL_MAT_F16_F32_L4_DR_LS_R2_MAX; ++q) {
        const int i12_q = i12_kv*r2 + q;
        const ulong q_off = (ulong)i12_q*nb12 + (ulong)i13_kv*nb13;
        y4[q] = (global float4 *)(src1 + q_off);
    }

    float partial[MUL_MAT_F16_F32_L4_DR_LS_R2_MAX];
    #pragma unroll
    for (int q = 0; q < MUL_MAT_F16_F32_L4_DR_LS_R2_MAX; ++q) {
        partial[q] = 0.0f;
    }

    const int n_chunks = ne00 / 4;

    for (int i = intra; i < n_chunks; i += 32) {
        float4 k = convert_float4(x4[i]);

        #pragma unroll
        for (int q = 0; q < MUL_MAT_F16_F32_L4_DR_LS_R2_MAX; ++q) {
            if (q < r2) {
                float4 v = y4[q][i];
                partial[q] = mad(k.s0, v.s0, partial[q]);
                partial[q] = mad(k.s1, v.s1, partial[q]);
                partial[q] = mad(k.s2, v.s2, partial[q]);
                partial[q] = mad(k.s3, v.s3, partial[q]);
            }
        }
    }

    // half-wave reduction
    #pragma unroll
    for (int q = 0; q < MUL_MAT_F16_F32_L4_DR_LS_R2_MAX; ++q) {
        if (q < r2) {
            partial[q] += sub_group_shuffle_xor(partial[q],  1u);
            partial[q] += sub_group_shuffle_xor(partial[q],  2u);
            partial[q] += sub_group_shuffle_xor(partial[q],  4u);
            partial[q] += sub_group_shuffle_xor(partial[q],  8u);
            partial[q] += sub_group_shuffle_xor(partial[q], 16u);
        }
    }

    if (intra == 0 && r0 < ne01) {
        #pragma unroll
        for (int q = 0; q < MUL_MAT_F16_F32_L4_DR_LS_R2_MAX; ++q) {
            if (q < r2) {
                const int im = i12_kv*r2 + q + i13_kv*ne12;
                dst[im*ne1*ne0 + r0] = partial[q];
            }
        }
    }
}

REQD_SUBGROUP_SIZE_64
kernel void kernel_mul_mat_f16_f32_l4_dr_lq(
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src0 = (global char*)((global char*)src0 + offset0);
    src1 = (global char*)((global char*)src1 + offset1);
    dst  = (global float*)((global char*)dst  + offsetd);

    const int r0_base = get_group_id(0) * 4;
    const int kv_grp  = get_group_id(2);

    const int i12_kv = kv_grp % ne02;
    const int i13_kv = kv_grp / ne02;

    const int lid   = get_sub_group_local_id();
    const int subq  = lid >> 4;            // 0..3 (which K row)
    const int intra = lid & 15;            // 0..15 (lane within quarter)

    const int r0  = r0_base + subq;
    const int r0c = r0 < ne01 ? r0 : 0;

    const ulong k_off = (ulong)r0c*nb01 + (ulong)i12_kv*nb02 + (ulong)i13_kv*nb03;
    global half4 * x4 = (global half4 *)(src0 + k_off);

    global float4 * y4[MUL_MAT_F16_F32_L4_DR_LS_R2_MAX];
    #pragma unroll
    for (int q = 0; q < MUL_MAT_F16_F32_L4_DR_LS_R2_MAX; ++q) {
        const int i12_q = i12_kv*r2 + q;
        const ulong q_off = (ulong)i12_q*nb12 + (ulong)i13_kv*nb13;
        y4[q] = (global float4 *)(src1 + q_off);
    }

    float partial[MUL_MAT_F16_F32_L4_DR_LS_R2_MAX];
    #pragma unroll
    for (int q = 0; q < MUL_MAT_F16_F32_L4_DR_LS_R2_MAX; ++q) {
        partial[q] = 0.0f;
    }

    const int n_chunks = ne00 / 4;

    for (int i = intra; i < n_chunks; i += 16) {
        float4 k = convert_float4(x4[i]);

        #pragma unroll
        for (int q = 0; q < MUL_MAT_F16_F32_L4_DR_LS_R2_MAX; ++q) {
            if (q < r2) {
                float4 v = y4[q][i];
                partial[q] = mad(k.s0, v.s0, partial[q]);
                partial[q] = mad(k.s1, v.s1, partial[q]);
                partial[q] = mad(k.s2, v.s2, partial[q]);
                partial[q] = mad(k.s3, v.s3, partial[q]);
            }
        }
    }

    // quarter-wave reduction
    #pragma unroll
    for (int q = 0; q < MUL_MAT_F16_F32_L4_DR_LS_R2_MAX; ++q) {
        if (q < r2) {
            partial[q] += sub_group_shuffle_xor(partial[q], 1u);
            partial[q] += sub_group_shuffle_xor(partial[q], 2u);
            partial[q] += sub_group_shuffle_xor(partial[q], 4u);
            partial[q] += sub_group_shuffle_xor(partial[q], 8u);
        }
    }

    if (intra == 0 && r0 < ne01) {
        #pragma unroll
        for (int q = 0; q < MUL_MAT_F16_F32_L4_DR_LS_R2_MAX; ++q) {
            if (q < r2) {
                const int im = i12_kv*r2 + q + i13_kv*ne12;
                dst[im*ne1*ne0 + r0] = partial[q];
            }
        }
    }
}
#endif // ADRENO_GPU

// Multi-row variant: each workgroup processes N_ROWS_PER_WG K rows instead of
// 1, amortizing dispatch overhead. The default kernel above launches one WG
// per (r0, im) which means ~262K workgroups for Qwen3.6 attn KQ at d=16k —
// 64 threads each doing one mad and a sub_group_reduce. Per-call wall time is
// ~8× over the K-bandwidth ideal because of wave-dispatch + memory-latency
// overhead. This variant collapses 8 of those WGs into one and caches Q once
// per Q-row in __local across the 8 K-row computations.
//
// Dispatched when ne11 == 1 (decode: single Q row) and ne01 % N_ROWS == 0,
// with global x = ne01/N_ROWS * subgroup_size.
#define N_ROWS_PER_WG 8
#define N_OUTS_PER_WG 8

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_f16_f32_l4_x8(
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src0 = (global char *)((global char *)src0 + offset0);
    src1 = (global char *)((global char *)src1 + offset1);
    dst  = (global float*)((global char *)dst  + offsetd);

    const int sgs_lid = get_sub_group_local_id();
    const int sgs_sz  = get_max_sub_group_size();

    const int r0_base = get_group_id(0) * N_ROWS_PER_WG;
    const int im      = get_group_id(2);

    const int i12 = im % ne12;
    const int i13 = im / ne12;

    // Single Q row only (decode). Cache Q once in __local for reuse across
    // the N_ROWS K-row computations.
    const ulong offset_src1 = (i12) * nb12 + (i13) * nb13;
    global float4 * y4 = (global float4 *)(src1 + offset_src1);

    __local float4 q_loc[64];   // ne00/4 max for sub_group_size 64
    if (sgs_lid < ne00 / 4) {
        q_loc[sgs_lid] = y4[sgs_lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    #pragma unroll
    for (int dr = 0; dr < N_ROWS_PER_WG; ++dr) {
        const int r0 = r0_base + dr;
        if (r0 >= ne01) return;

        const ulong offset_src0 = r0 * nb01 + (i12 / r2) * nb02 + (i13 / r3) * nb03;
        global half4 * x4 = (global half4 *)(src0 + offset_src0);

        float sumf = 0.0f;
        for (int i = sgs_lid; i < ne00 / 4; i += sgs_sz) {
            const half4   k4 = x4[i];
            const float4  q  = q_loc[i];
            sumf += convert_float(k4.s0) * q.s0
                  + convert_float(k4.s1) * q.s1
                  + convert_float(k4.s2) * q.s2
                  + convert_float(k4.s3) * q.s3;
        }

        const float all_sum = sub_group_reduce_add(sumf);
        if (sgs_lid == 0) {
            dst[im * ne1 * ne0 + r0] = all_sum;  // ne11 == 1, so r1==0
        }
    }
}

// Streaming-Q multi-output variant for the KQV-shaped matmul: src0 has small
// ne01 (e.g. DV=256) but large ne00 (n_kv, up to 16384 at d=16k). The x8
// kernel can't handle this because its per-WG __local Q cache is sized for
// ne00 <= 256. This variant streams Q from global (no cache) but still packs
// N_OUTS_PER_WG = 8 outputs per workgroup. Q is re-read once per output
// inside the inner loop; Adreno L1 absorbs the 8× redundancy since adjacent
// outputs in one WG hit the same Q cache lines per iter.
//
// Dispatched for the same shape pattern as x8 (ne11 == 1, ne01 divisible by 8)
// when ne00 > 256, i.e. when the x8 path can't be used.
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_f16_f32_l4_y8(
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src0 = (global char *)((global char *)src0 + offset0);
    src1 = (global char *)((global char *)src1 + offset1);
    dst  = (global float*)((global char *)dst  + offsetd);

    const int sgs_lid = get_sub_group_local_id();
    const int sgs_sz  = get_max_sub_group_size();

    const int r0_base = get_group_id(0) * N_OUTS_PER_WG;
    const int im      = get_group_id(2);

    const int i12 = im % ne12;
    const int i13 = im / ne12;

    // Q (= src1) base pointer; r1 == 0 since ne11 == 1.
    const ulong offset_src1 = (i12) * nb12 + (i13) * nb13;
    global float4 * y4 = (global float4 *)(src1 + offset_src1);

    // Per-output base pointers (per row of src0). Computed once; inner loop
    // strides float4 indices across them.
    global half4 * x4_o[N_OUTS_PER_WG];
    #pragma unroll
    for (int o = 0; o < N_OUTS_PER_WG; ++o) {
        const int r0 = r0_base + o;
        // Pre-cap: if r0 OOB, point to the first row (harmless reads, output
        // suppressed at write-time). Keeps the inner loop unconditional.
        const int r0c = (r0 < ne01) ? r0 : 0;
        const ulong off = r0c * nb01 + (i12 / r2) * nb02 + (i13 / r3) * nb03;
        x4_o[o] = (global half4 *)(src0 + off);
    }

    float sum[N_OUTS_PER_WG] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    for (int i = sgs_lid; i < ne00 / 4; i += sgs_sz) {
        const float4 q4 = y4[i];
        #pragma unroll
        for (int o = 0; o < N_OUTS_PER_WG; ++o) {
            const half4 v4 = x4_o[o][i];
            sum[o] += convert_float(v4.s0) * q4.s0
                    + convert_float(v4.s1) * q4.s1
                    + convert_float(v4.s2) * q4.s2
                    + convert_float(v4.s3) * q4.s3;
        }
    }

    #pragma unroll
    for (int o = 0; o < N_OUTS_PER_WG; ++o) {
        const int r0 = r0_base + o;
        const float s = sub_group_reduce_add(sum[o]);
        if (sgs_lid == 0 && r0 < ne01) {
            dst[im * ne1 * ne0 + r0] = s;
        }
    }
}

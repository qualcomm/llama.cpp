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

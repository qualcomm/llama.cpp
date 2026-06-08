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

// Multi-row f16xf32 GEMV for the DECODE path (single token, ne11*ne12 small).
// The legacy kernel_mul_mat_f16_f32_1row runs ONE 64-lane subgroup per workgroup =
// one output row per WG, which caps memory-level parallelism (~115 GB/s on X2,
// ~half of LPDDR5x peak). This variant packs MROW subgroups per workgroup, each
// computing a distinct output row, so a WG keeps 64*MROW loads in flight. The
// activation column y (shared by every output row) is staged into __local ONCE per
// WG and reused across the MROW rows, cutting redundant activation reads. Used for
// the f16 attention projections (Q/K/V/O) and lm_head, which dominate decode.
// Numerically equivalent to _1row (same f16->f32 widening, same float4 partial sums,
// same subgroup-reduce order), so byte-identical to the per-op path.

#define MROW 16

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_f16_f32_mrow(
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
        int r3,
        __local float * ysh
) {
    src0 = (global char*)((global char*)src0 + offset0);
    src1 = (global char*)((global char*)src1 + offset1);
    dst  = (global float*)((global char*)dst + offsetd);

    int r0   = get_group_id(0) * MROW + get_local_id(1);  // output row
    int r1   = get_group_id(1);                            // token (ne11)
    int im   = get_group_id(2);
    int lid  = get_sub_group_local_id();                   // 0..63
    int nsg  = get_local_size(1);                          // == MROW

    int i12 = im % ne12;
    int i13 = im / ne12;

    ulong offset_src1 = r1*nb11 + (i12)*nb12 + (i13)*nb13;
    global float * y = (global float *) (src1 + offset_src1);

    // Cooperatively stage the activation column (ne00 floats) into __local once per
    // WG and reuse across the MROW rows. Staging is the actual win here: dropping it
    // (each subgroup re-reading y from global) regresses below the 1-row kernel.
    for (int i = get_local_id(1)*get_sub_group_size() + lid; i < ne00; i += nsg*get_sub_group_size()) {
        ysh[i] = y[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (r0 >= ne01) {
        return;
    }

    ulong offset_src0 = r0*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;
    global half * x = (global half *) (src0 + offset_src0);

    float sumf = 0.0f;
    if (ne00 < 128) {
        for (int i = lid; i < ne00; i += get_sub_group_size()) {
            sumf += (float) x[i] * ysh[i];
        }
        float all_sum = sub_group_reduce_add(sumf);
        if (lid == 0) {
            dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
        }
    } else {
        global half4 * x4 = (global half4 *) x;
        __local float4 * ysh4 = (__local float4 *) ysh;
        for (int i = lid; i < ne00/4; i += get_sub_group_size()) {
            float4 yv = ysh4[i];
            sumf += (float) x4[i].s0 * yv.s0;
            sumf += (float) x4[i].s1 * yv.s1;
            sumf += (float) x4[i].s2 * yv.s2;
            sumf += (float) x4[i].s3 * yv.s3;
        }
        float all_sum = sub_group_reduce_add(sumf);
        if (lid == 0) {
            for (int i = 4*(ne00/4); i < ne00; ++i) {
                all_sum += (float) x[i] * ysh[i];
            }
            dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
        }
    }
}

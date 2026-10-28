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

// q2_0 split into SOA arrays by kernel_convert_block_q1_0 (the block is the same 2 + 16 bytes):
// src0_q holds 16 quant bytes per 64-weight block, src0_d one half per block. Weight 4*i + j is
// bits 2j..2j+1 of byte i and decodes to q - 1.
#define QK2_0           64
#define QK2_0_BYTES     (QK2_0/4)          // 16 quant bytes per block
#define QK2_0_BLK_BYTES (QK2_0_BYTES + 2)  // d + qs in the original tensor = 18

#define N_R0_Q2_0 4 // rows per subgroup
#define N_SG_Q2_0 2 // subgroups per workgroup

#ifdef INTEL_GPU
#define N_SIMDWIDTH 16
#elif defined (ADRENO_GPU)
#define N_SIMDWIDTH 64
#endif

// dot of 16 weights (one uint of quants) with y, where yt[k] holds y[4*i + k] for i = 0..3
inline float dot_q2_16(uint w, float4 yt0, float4 yt1, float4 yt2, float4 yt3) {
    return dot(convert_float4(as_uchar4((w     ) & 0x03030303u)), yt0)
         + dot(convert_float4(as_uchar4((w >> 2) & 0x03030303u)), yt1)
         + dot(convert_float4(as_uchar4((w >> 4) & 0x03030303u)), yt2)
         + dot(convert_float4(as_uchar4((w >> 6) & 0x03030303u)), yt3);
}

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_16
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mv_q2_0_f32_flat(
    global char * src0_q,
    global half * src0_d,
    global char * src1,
    ulong         offset1,
    global char * dst,
    ulong         offsetd,
    int           ne00,
    int           ne01,
    ulong         nb01,
    ulong         nb02,
    ulong         nb03,
    int           ne12,
    ulong         nb11,
    ulong         nb12,
    ulong         nb13,
    int           ne0,
    int           ne1,
    int           r2,
    int           r3,
    int           trans   // 1: q as [K/16][ne01] uint32 and d as [K/64][ne01] (ggml_cl_q2_0_is_trans)
) {
    src1 = (global char*)((global char*)src1 + offset1);
    dst  = (global char*)((global char*)dst  + offsetd);

    const int r0 = get_group_id(0);
    const int r1 = get_group_id(1);
    const int im = get_group_id(2);

    const int first_row = (r0*N_SG_Q2_0 + get_sub_group_id()) * N_R0_Q2_0;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    global float * y = (global float *) (src1 + r1*nb11 + i12*nb12 + i13*nb13);

    // Rows past ne01 (the padded x-grid tail) are clamped onto the last row, so no fetch
    // leaves src0; their results are dropped at the store.
    // quant word j of chunk c is ax[r][(2c + j)*stride], its scale ad[r][(c/2)*stride]
    global uint * ax[N_R0_Q2_0];
    global half * ad[N_R0_Q2_0];
    const ulong stride = trans ? (ulong)ne01 : 1;
    #pragma unroll
    for (int r = 0; r < N_R0_Q2_0; ++r) {
        const ulong row = min(first_row + r, ne01 - 1);
        const ulong blk = (row*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03) / QK2_0_BLK_BYTES;
        ax[r] = (global uint *) src0_q + (trans ? row : blk*(QK2_0_BYTES/4));
        ad[r] = src0_d + (trans ? row : blk);
    }

    // each lane takes 32 weights (8 quant bytes, half a block) per step
    const int nchunk = ne00/32;
    float sumf[N_R0_Q2_0] = { 0.f };

    for (int c = get_sub_group_local_id(); c < nchunk; c += N_SIMDWIDTH) {
        const float8 ya = vload8(0, y + c*32);
        const float8 yb = vload8(0, y + c*32 + 8);
        const float8 yc = vload8(0, y + c*32 + 16);
        const float8 yd = vload8(0, y + c*32 + 24);

        // y regrouped by bit position within each quant byte
        const float4 ta0 = (float4)(ya.s0, ya.s4, yb.s0, yb.s4);
        const float4 ta1 = (float4)(ya.s1, ya.s5, yb.s1, yb.s5);
        const float4 ta2 = (float4)(ya.s2, ya.s6, yb.s2, yb.s6);
        const float4 ta3 = (float4)(ya.s3, ya.s7, yb.s3, yb.s7);
        const float4 tb0 = (float4)(yc.s0, yc.s4, yd.s0, yd.s4);
        const float4 tb1 = (float4)(yc.s1, yc.s5, yd.s1, yd.s5);
        const float4 tb2 = (float4)(yc.s2, yc.s6, yd.s2, yd.s6);
        const float4 tb3 = (float4)(yc.s3, yc.s7, yd.s3, yd.s7);

        const float sumy = dot(ya.lo + ya.hi + yb.lo + yb.hi + yc.lo + yc.hi + yd.lo + yd.hi,
                               (float4)(1.f));

        #pragma unroll
        for (int r = 0; r < N_R0_Q2_0; ++r) {
            const uint w0 = ax[r][(2*c + 0)*stride];
            const uint w1 = ax[r][(2*c + 1)*stride];
            const float acc = dot_q2_16(w0, ta0, ta1, ta2, ta3)
                            + dot_q2_16(w1, tb0, tb1, tb2, tb3);
            sumf[r] += (float) ad[r][(c/2)*stride] * (acc - sumy);
        }
    }

    global float * dst_f32 = (global float *) dst + (ulong)im*ne0*ne1 + (ulong)r1*ne0;

    #pragma unroll
    for (int r = 0; r < N_R0_Q2_0; ++r) {
        const float tot = sub_group_reduce_add(sumf[r]);
        if (get_sub_group_local_id() == 0 && first_row + r < ne01) {
            dst_f32[first_row + r] = tot;
        }
    }
}

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define LOAD_VEC_A 8
#define LOAD_VEC_B 4

#define BM 64
#define BN 64
// NOTE: BK must stay 32 here. This loader splits each 32-element block into low
// nibbles at k+0..3 and high nibbles at k+16..19, so a 16-wide K tile would
// write past the end of buf_a. The block-reading kernels use BK=16.
#ifndef BK
#define BK 32
#endif
#ifndef TM
#define TM 4
#endif
#ifndef TN
#define TN 8
#endif

kernel void kernel_mul_mm_q4_0_f32_l4_lm(
    global uchar4 * src0_q,
    global half   * src0_d,
    global float4 * src1,
    ulong offset1,
    global float  * dst,
    ulong offsetd,

    int ne00,
    int ne01,
    int ne02,
    int ne11,
    int ne12,

    int stride_a,
    int stride_b,
    int stride_d,

    int batch_stride_a,
    int batch_stride_b,
    int batch_stride_d,

    int r2,
    int r3
) {
    src1 = (global float4*)((global char*)src1 + offset1);
    dst  = (global float *)((global char*)dst  + offsetd);

    local float buf_a[BM * BK];
    local float buf_b[BN * BK];

    const int batch_idx = get_global_id(2);

    const int i13 = batch_idx / ne12;
    const int i12 = batch_idx % ne12;

    const int i03 = i13 / r3;
    const int i02 = i12 / r2;

    const int batch_idx_a = i03 * ne02 + i02;

    const int ir = get_group_id(0);
    const int ic = get_group_id(1);

    const int tid = get_local_id(0);
    const int th_r  = tid % (BM / TM);
    const int th_c  = tid / (BM / TM);

    const int loadr_a = get_local_id(0) % (BK / LOAD_VEC_A);
    const int loadc_a = get_local_id(0) / (BK / LOAD_VEC_A);
    const int loadr_b = get_local_id(0) % (BK / LOAD_VEC_B);
    const int loadc_b = get_local_id(0) / (BK / LOAD_VEC_B);

    const int loadstride_a = get_local_size(0) * LOAD_VEC_A / BK;
    const int loadstride_b = get_local_size(0) * LOAD_VEC_B / BK;

    int pos_a = (batch_idx_a * batch_stride_a + ir * BM * stride_a) / LOAD_VEC_A;
    int pos_b = (batch_idx   * batch_stride_b + ic * BN * stride_b) / LOAD_VEC_B;

    // Accumulate four rows at a time. buf_a is contiguous in the row index, so a
    // whole TM slice arrives as float4 loads instead of TM scalar ones, and each
    // vector mad replaces four scalar ones. Same operands in the same order, so
    // the result is unchanged.
    float4 sums4[(TM/4) * TN];
    float4 cache_a4[TM/4];

    for (int i = 0; i < (TM/4) * TN; i++) {
        sums4[i] = (float4)(0.0f);
    }

    for (int block = 0; block < ne00; block += BK) {
        for (int l = 0; l < BM; l += loadstride_a) {
            if (ir*BM + loadc_a + l < ne01) {
                int idx = pos_a + (loadc_a + l) * stride_a / LOAD_VEC_A + loadr_a;
                int ib  = idx / 4;
                int iqs = idx % 4;

                float d = (float)src0_d[ib];
                global uchar4 * qs = src0_q + ib*4 + iqs;
                uchar4 q = *qs;
                float4 v1 = (convert_float4((uchar4)((q.s0   )&0x0F, (q.s1   )&0x0F, (q.s2   )&0x0F, (q.s3   )&0x0F)) - 8.0f)*d;
                float4 v2 = (convert_float4((uchar4)((q.s0>>4)&0x0F, (q.s1>>4)&0x0F, (q.s2>>4)&0x0F, (q.s3>>4)&0x0F)) - 8.0f)*d;

                buf_a[(loadr_a * 4 +  0) * BM + loadc_a + l] = v1.s0;
                buf_a[(loadr_a * 4 +  1) * BM + loadc_a + l] = v1.s1;
                buf_a[(loadr_a * 4 +  2) * BM + loadc_a + l] = v1.s2;
                buf_a[(loadr_a * 4 +  3) * BM + loadc_a + l] = v1.s3;
                buf_a[(loadr_a * 4 + 16) * BM + loadc_a + l] = v2.s0;
                buf_a[(loadr_a * 4 + 17) * BM + loadc_a + l] = v2.s1;
                buf_a[(loadr_a * 4 + 18) * BM + loadc_a + l] = v2.s2;
                buf_a[(loadr_a * 4 + 19) * BM + loadc_a + l] = v2.s3;
            } else {
                buf_a[(loadr_a * 4 +  0) * BM + loadc_a + l] = 0.0f;
                buf_a[(loadr_a * 4 +  1) * BM + loadc_a + l] = 0.0f;
                buf_a[(loadr_a * 4 +  2) * BM + loadc_a + l] = 0.0f;
                buf_a[(loadr_a * 4 +  3) * BM + loadc_a + l] = 0.0f;
                buf_a[(loadr_a * 4 + 16) * BM + loadc_a + l] = 0.0f;
                buf_a[(loadr_a * 4 + 17) * BM + loadc_a + l] = 0.0f;
                buf_a[(loadr_a * 4 + 18) * BM + loadc_a + l] = 0.0f;
                buf_a[(loadr_a * 4 + 19) * BM + loadc_a + l] = 0.0f;
            }
        }

        for (int l = 0; l < BN; l += loadstride_b) {
            if (ic*BN + loadc_b + l < ne11) {
                int idx = pos_b + (loadc_b + l) * stride_b / LOAD_VEC_B + loadr_b;
                buf_b[(loadr_b * LOAD_VEC_B + 0) * BN + loadc_b + l] = src1[idx].s0;
                buf_b[(loadr_b * LOAD_VEC_B + 1) * BN + loadc_b + l] = src1[idx].s1;
                buf_b[(loadr_b * LOAD_VEC_B + 2) * BN + loadc_b + l] = src1[idx].s2;
                buf_b[(loadr_b * LOAD_VEC_B + 3) * BN + loadc_b + l] = src1[idx].s3;
            } else {
                buf_b[(loadr_b * LOAD_VEC_B + 0) * BN + loadc_b + l] = 0.0f;
                buf_b[(loadr_b * LOAD_VEC_B + 1) * BN + loadc_b + l] = 0.0f;
                buf_b[(loadr_b * LOAD_VEC_B + 2) * BN + loadc_b + l] = 0.0f;
                buf_b[(loadr_b * LOAD_VEC_B + 3) * BN + loadc_b + l] = 0.0f;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        pos_a += BK / LOAD_VEC_A;
        pos_b += BK / LOAD_VEC_B;

        for (int i = 0; i < BK; i++) {
            for (int a = 0; a < TM/4; a++) {
                cache_a4[a] = vload4(a, buf_a + (i) * BM + th_r * TM);
            }

            for (int cc = 0; cc < TN; cc++) {
                const float cache_b = buf_b[(i) * BN + th_c * TN + cc];
                for (int a = 0; a < TM/4; a++) {
                    const int sums_idx = cc*(TM/4) + a;
                    sums4[sums_idx] = mad(cache_a4[a], (float4)cache_b, sums4[sums_idx]);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const int dr = ir * BM + th_r * TM;
    const int dc = ic * BN + th_c * TN;

    const int offsets = batch_idx * batch_stride_d;

    for (int cc = 0; cc < TN; cc++) {
        for (int a = 0; a < TM/4; a++) {
            const float4 v = sums4[cc * (TM/4) + a];
            const float  vs[4] = { v.s0, v.s1, v.s2, v.s3 };
            for (int k = 0; k < 4; k++) {
                if (dr + 4*a + k < ne01 && dc + cc < ne11) {
                    dst[offsets + (dc + cc) * stride_d + dr + 4*a + k] = vs[k];
                }
            }
        }
    }
}

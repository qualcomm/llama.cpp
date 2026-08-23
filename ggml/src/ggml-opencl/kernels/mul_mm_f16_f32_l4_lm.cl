#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define LOAD_VEC_A 4
#define LOAD_VEC_B 4

// Tile shape is overridable at build time so the same source can be compiled
// as a second, NARROW instance for skinny-N matmuls (the KQ/KQV of a
// speculative/MTP verify batch, where ne11 is the verify width 2..8).
//
// With the default BN=64/TN=8 the N dimension is tiled 64 wide: at ne11=4 the
// grid is CEIL_DIV(4,64)=1 column tile, so 7 of the 8 column groups
// (th_c = tid/(BM/TM)) hold no valid column at all, and the one that does uses
// 4 of its TN=8 slots -> 4/64 of the tile does useful work. The masked lanes
// still execute every mad; the bounds test only guards the final store.
//
// The narrow instance uses BN=8/TN=1 with the same 128 threads, which covers a
// verify width of up to 8 in a single column tile with no waste at w=8 and
// half-occupancy at w=4, instead of 6.25%.
#ifndef BM
#define BM 64
#endif
#ifndef BN
#define BN 64
#endif
#ifndef BK
#define BK 16
#endif
#ifndef TM
#define TM 4
#endif
#ifndef TN
#define TN 8
#endif

// The narrow instance is compiled from this same source, so it must export a
// DIFFERENT symbol -- otherwise the two are indistinguishable in a profile and
// an env-gated A/B cannot be checked for which variant actually ran.
#ifndef KERNEL_NAME_LM
#define KERNEL_NAME_LM kernel_mul_mm_f16_f32_l4_lm
#endif

kernel void KERNEL_NAME_LM(
    global half4 * src0,
    ulong offset0,
    global float4 * src1,
    ulong offset1,
    global float * dst,
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
    src0 = (global half4*)((global char*)src0 + offset0);
    src1 = (global float4*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    local half  buf_a[BM * BK];
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
    half4  cache_a4[TM/4];

    for (int i = 0; i < (TM/4) * TN; i++) {
        sums4[i] = (float4)(0.0f);
    }

    for (int block = 0; block < ne00; block += BK) {
        for (int l = 0; l < BM; l += loadstride_a) {
            if (ir*BM + loadc_a + l < ne01) {
                const int idx = pos_a + (loadc_a + l) * stride_a / LOAD_VEC_A + loadr_a;
                buf_a[(loadr_a * LOAD_VEC_A + 0) * BM + loadc_a + l] = src0[idx].s0;
                buf_a[(loadr_a * LOAD_VEC_A + 1) * BM + loadc_a + l] = src0[idx].s1;
                buf_a[(loadr_a * LOAD_VEC_A + 2) * BM + loadc_a + l] = src0[idx].s2;
                buf_a[(loadr_a * LOAD_VEC_A + 3) * BM + loadc_a + l] = src0[idx].s3;
            } else {
                buf_a[(loadr_a * LOAD_VEC_A + 0) * BM + loadc_a + l] = 0.0h;
                buf_a[(loadr_a * LOAD_VEC_A + 1) * BM + loadc_a + l] = 0.0h;
                buf_a[(loadr_a * LOAD_VEC_A + 2) * BM + loadc_a + l] = 0.0h;
                buf_a[(loadr_a * LOAD_VEC_A + 3) * BM + loadc_a + l] = 0.0h;
            }
        }

        // loadc_b is derived from the thread id and spans get_local_size(0)/(BK/LOAD_VEC_B)
        // regardless of BN, so once BN is narrowed below that span the extra threads must
        // sit the load out -- otherwise they write past buf_b[BN*BK]. With the default
        // BN=64 this is true for every thread and the guard is free.
        for (int l = 0; l < BN; l += loadstride_b) {
            if (loadc_b + l >= BN) {
                // nothing to load for this thread at this tile width
            } else if (ic*BN + loadc_b + l < ne11) {
                const int idx = pos_b + (loadc_b + l) * stride_b / LOAD_VEC_B + loadr_b;
                buf_b[(loadr_b * LOAD_VEC_B + 0) * BN + loadc_b + l] = src1[idx].s0;
                buf_b[(loadr_b * LOAD_VEC_B + 1) * BN + loadc_b + l] = src1[idx].s1;
                buf_b[(loadr_b * LOAD_VEC_B + 2) * BN + loadc_b + l] = src1[idx].s2;
                buf_b[(loadr_b * LOAD_VEC_B + 3) * BN + loadc_b + l] = src1[idx].s3;
            } else {
                buf_b[(loadr_b * LOAD_VEC_B + 0) * BN + loadc_b + l] = 0.0h;
                buf_b[(loadr_b * LOAD_VEC_B + 1) * BN + loadc_b + l] = 0.0h;
                buf_b[(loadr_b * LOAD_VEC_B + 2) * BN + loadc_b + l] = 0.0h;
                buf_b[(loadr_b * LOAD_VEC_B + 3) * BN + loadc_b + l] = 0.0h;
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
                    sums4[sums_idx] = mad(convert_float4(cache_a4[a]), (float4)cache_b, sums4[sums_idx]);
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

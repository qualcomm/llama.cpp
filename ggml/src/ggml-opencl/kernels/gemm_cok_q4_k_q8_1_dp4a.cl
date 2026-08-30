// cok-shaped q4_K GEMM with a dp4a inner product, for the narrow band ne1 = 2..8.
//
// WHY THIS EXISTS. The 2..8 band currently runs cok (f16 FMA) because the *prefill*
// dp4a GEMM loses there by ~14% -- but that kernel is built around a wide activation
// tile and re-unpacks the weight row every 32-K step, amortising it over TILESIZE_N
// columns. At 8 columns that amortisation is 4x worse. Narrowing its tile constant is
// not an optimisation, so it never tested whether int8 arithmetic helps at narrow batch.
//
// This keeps everything that makes cok good at narrow width -- the 4-row fold, so one
// weight read and one scale/min unpack serve four output rows, and the K-split across
// subgroups -- and changes only the inner product.
//
// Measured premise (microbench/f16_vs_dp4a, X2-90): dp4a retires 1.67x the MACs of the
// half8 FMA cok uses (5214 vs 3113 GMAC/s at matched unroll). The open question this
// kernel answers is whether that beats the q8_1 activation pre-pass it forces, which cok
// avoids entirely -- a fixed per-dispatch cost, and fixed costs hurt most at narrow batch.
//
// REGISTER BUDGET IS THE TOP RISK. The same microbench measured half8 collapsing 43x
// (3113 -> 72.6 GMAC/s) purely from crossing the 512 B/WI spill cliff. This kernel holds
// 32 int dot accumulators (128 B) + 4 float8 (128 B) + operands ~= 320 B/lane. That is
// under the cliff but only by ~1.6x, so do not widen it without re-measuring.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable

#define QK_K          256
#define K_SCALE_SIZE   12

#ifndef COK_NSG
#define COK_NSG 8
#endif
#define COK_SG  64

// One packed q4_K ushort holds 4 consecutive-K nibbles for one row; spread them into the
// 4 bytes of a uint so dp4a can take it directly. Same expansion the prefill GEMM uses.
#define EXP4(u)  ( ((uint)((u) & 0x000Fu))        | \
                  (((uint)((u) & 0x00F0u)) << 4)  | \
                  (((uint)((u) & 0x0F00u)) << 8)  | \
                  (((uint)((u) & 0xF000u)) << 12) )

inline void get_scale_min_k4_c(int j, global const uchar * q, int stride,
                               uchar * d, uchar * m,
                               uchar mask_d6, uchar mask_d4, uchar mask_hi2) {
    if (j < 4) {
        *d = q[j*stride] & mask_d6;
        *m = q[(j + 4)*stride] & mask_d6;
    } else {
        *d = (q[(j + 4)*stride] & mask_d4) | ((q[(j - 4)*stride] >> 6) << 4);
        *m = (q[(j + 4)*stride] >>   4)    | ((q[(j    )*stride] >> 6) << 4);
    }
}

__attribute__((qcom_wave_pair_mode(1)))
kernel void kernel_gemm_cok_q4_k_q8_1_dp4a(
    global const ushort * src0_q,     // q4_K nibble plane   [row + (K/4)*m]
    global const uchar  * src0_s,     // packed scales/mins
    global const half   * src0_d,     // super-block scale
    global const half   * src0_dm,    // super-block min
    global const uint   * src1_qa,    // q8_1 activations    [col*k_u + K/4]
    global const half   * src1_da,    // activation scale    [col*k_b + blk]
    global const half   * src1_sa,    // activation sum      [col*k_b + blk]
    global float * dst,
    ulong offsetd,
    int m,
    int n,
    int k,
    int n_no_padding,
    uchar mask_d6,
    uchar mask_d4,
    uchar mask_hi2
) {
    dst = (global float *)((global char *)dst + offsetd);

    const int gx   = get_global_id(0);   // 4-row group
    const int sg   = get_local_id(1);    // K-split subgroup
    const int lane = get_local_id(0);

    const int row0 = gx << 2;
    const int num_32blk = k / 32;
    const int k_u = k >> 2;              // K in uint (int8x4) units
    const int k_b = k >> 5;              // 32-blocks along K

    // Scaled results, one float8 per folded row: 8 columns in the vector lanes, as cok.
    float8 acc0 = (float8)(0.0f), acc1 = (float8)(0.0f);
    float8 acc2 = (float8)(0.0f), acc3 = (float8)(0.0f);

    for (int blk = sg; blk < num_32blk; blk += COK_NSG) {
        const int i       = blk << 5;
        const int sb_idx  = blk >> 3;
        const int sub_idx = blk & 7;

        // one vector load each: the four rows are adjacent
        half4 dd  = vload4(0, src0_d  + row0 + sb_idx * m);
        half4 dmm = vload4(0, src0_dm + row0 + sb_idx * m);

        global const uchar * sc = src0_s + sb_idx * K_SCALE_SIZE * m + row0;
        uchar sv0, mn0, sv1, mn1, sv2, mn2, sv3, mn3;
        get_scale_min_k4_c(sub_idx, sc + 0, m, &sv0, &mn0, mask_d6, mask_d4, mask_hi2);
        get_scale_min_k4_c(sub_idx, sc + 1, m, &sv1, &mn1, mask_d6, mask_d4, mask_hi2);
        get_scale_min_k4_c(sub_idx, sc + 2, m, &sv2, &mn2, mask_d6, mask_d4, mask_hi2);
        get_scale_min_k4_c(sub_idx, sc + 3, m, &sv3, &mn3, mask_d6, mask_d4, mask_hi2);

        const float sc0 = (float)dd.s0 * (float)sv0, mv0 = (float)dmm.s0 * (float)mn0;
        const float sc1 = (float)dd.s1 * (float)sv1, mv1 = (float)dmm.s1 * (float)mn1;
        const float sc2 = (float)dd.s2 * (float)sv2, mv2 = (float)dmm.s2 * (float)mn2;
        const float sc3 = (float)dd.s3 * (float)sv3, mv3 = (float)dmm.s3 * (float)mn3;

        // raw int dot per (folded row, column), reset each 32-block because the weight
        // scale/min below are per 32-block
        int s0[8], s1[8], s2[8], s3[8];
        #pragma unroll
        for (int c = 0; c < 8; ++c) { s0[c] = 0; s1[c] = 0; s2[c] = 0; s3[c] = 0; }

        // 8 sub-steps of 4 K values each.
        //
        // Deliberately NOT unrolled, here or over the columns below. Measured on X2-90:
        // unrolling both takes private memory from 496 to 928 B/lane, past the 512 B/WI
        // spill cliff, and drops the kernel's launchable workgroup from 512 to 256 --
        // which halves COK_NSG, so the K-split loses half its parallelism as well. The
        // 1.67x arithmetic advantage this kernel exists to test is far smaller than what
        // that cliff costs.
        for (int u = 0; u < 8; ++u) {
            const int ku = (i >> 2) + u;                       // K/4 index
            ushort4 bits = vload4(0, src0_q + row0 + ku * m);  // 4 rows x 4 K nibbles
            const uint w0 = EXP4(bits.s0);
            const uint w1 = EXP4(bits.s1);
            const uint w2 = EXP4(bits.s2);
            const uint w3 = EXP4(bits.s3);

            for (int c = 0; c < 8; ++c) {
                const uint a = src1_qa[(uint)c * k_u + ku];
                s0[c] = dot_acc_sat_4x8packed_ss_int(w0, a, s0[c]);
                s1[c] = dot_acc_sat_4x8packed_ss_int(w1, a, s1[c]);
                s2[c] = dot_acc_sat_4x8packed_ss_int(w2, a, s2[c]);
                s3[c] = dot_acc_sat_4x8packed_ss_int(w3, a, s3[c]);
            }
        }

        // q4_K value is (q*scale - min), so per 32-block:
        //   out += scale * d_act * dot(q, a)  -  min * sum_act
        // where sum_act is q8_1's block sum (already carrying d_act).
        // Component assignment, NOT a pointer cast into the vector: taking the address
        // of a private vector forces it to memory and would spill the register budget
        // this kernel deliberately keeps under the 512 B/WI cliff.
        float8 da, sa;
        da.s0 = (float)src1_da[0*k_b + blk];  sa.s0 = (float)src1_sa[0*k_b + blk];
        da.s1 = (float)src1_da[1*k_b + blk];  sa.s1 = (float)src1_sa[1*k_b + blk];
        da.s2 = (float)src1_da[2*k_b + blk];  sa.s2 = (float)src1_sa[2*k_b + blk];
        da.s3 = (float)src1_da[3*k_b + blk];  sa.s3 = (float)src1_sa[3*k_b + blk];
        da.s4 = (float)src1_da[4*k_b + blk];  sa.s4 = (float)src1_sa[4*k_b + blk];
        da.s5 = (float)src1_da[5*k_b + blk];  sa.s5 = (float)src1_sa[5*k_b + blk];
        da.s6 = (float)src1_da[6*k_b + blk];  sa.s6 = (float)src1_sa[6*k_b + blk];
        da.s7 = (float)src1_da[7*k_b + blk];  sa.s7 = (float)src1_sa[7*k_b + blk];

        float8 d0, d1, d2, d3;
        d0.s0 = (float)s0[0]; d0.s1 = (float)s0[1]; d0.s2 = (float)s0[2]; d0.s3 = (float)s0[3];
        d0.s4 = (float)s0[4]; d0.s5 = (float)s0[5]; d0.s6 = (float)s0[6]; d0.s7 = (float)s0[7];
        d1.s0 = (float)s1[0]; d1.s1 = (float)s1[1]; d1.s2 = (float)s1[2]; d1.s3 = (float)s1[3];
        d1.s4 = (float)s1[4]; d1.s5 = (float)s1[5]; d1.s6 = (float)s1[6]; d1.s7 = (float)s1[7];
        d2.s0 = (float)s2[0]; d2.s1 = (float)s2[1]; d2.s2 = (float)s2[2]; d2.s3 = (float)s2[3];
        d2.s4 = (float)s2[4]; d2.s5 = (float)s2[5]; d2.s6 = (float)s2[6]; d2.s7 = (float)s2[7];
        d3.s0 = (float)s3[0]; d3.s1 = (float)s3[1]; d3.s2 = (float)s3[2]; d3.s3 = (float)s3[3];
        d3.s4 = (float)s3[4]; d3.s5 = (float)s3[5]; d3.s6 = (float)s3[6]; d3.s7 = (float)s3[7];

        acc0 += sc0 * da * d0 - mv0 * sa;
        acc1 += sc1 * da * d1 - mv1 * sa;
        acc2 += sc2 * da * d2 - mv2 * sa;
        acc3 += sc3 * da * d3 - mv3 * sa;
    }

    // Cross-subgroup reduction over the K-split, one row at a time so the __local buffer
    // stays the size of the 1-row kernel's -- same shape as cok_r4.
    local float8 reduceLM[COK_SG * (COK_NSG - 1)];
    float8 out[4];
    for (int r = 0; r < 4; r++) {
        float8 acc = (r == 0) ? acc0 : (r == 1) ? acc1 : (r == 2) ? acc2 : acc3;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (sg > 0) {
            reduceLM[(sg - 1) * COK_SG + lane] = acc;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (sg == 0) {
            float8 sum = acc;
            for (int s = 0; s < COK_NSG - 1; s++) {
                sum += reduceLM[s * COK_SG + lane];
            }
            out[r] = sum;
        }
    }

    if (sg == 0) {
        // dst is [token, feature]: four adjacent rows are contiguous, one vstore4.
        int idx = row0;
        if (idx < m*n_no_padding) { vstore4((float4)(out[0].s0, out[1].s0, out[2].s0, out[3].s0), 0, dst + idx); idx += m; }
        if (idx < m*n_no_padding) { vstore4((float4)(out[0].s1, out[1].s1, out[2].s1, out[3].s1), 0, dst + idx); idx += m; }
        if (idx < m*n_no_padding) { vstore4((float4)(out[0].s2, out[1].s2, out[2].s2, out[3].s2), 0, dst + idx); idx += m; }
        if (idx < m*n_no_padding) { vstore4((float4)(out[0].s3, out[1].s3, out[2].s3, out[3].s3), 0, dst + idx); idx += m; }
        if (idx < m*n_no_padding) { vstore4((float4)(out[0].s4, out[1].s4, out[2].s4, out[3].s4), 0, dst + idx); idx += m; }
        if (idx < m*n_no_padding) { vstore4((float4)(out[0].s5, out[1].s5, out[2].s5, out[3].s5), 0, dst + idx); idx += m; }
        if (idx < m*n_no_padding) { vstore4((float4)(out[0].s6, out[1].s6, out[2].s6, out[3].s6), 0, dst + idx); idx += m; }
        if (idx < m*n_no_padding) { vstore4((float4)(out[0].s7, out[1].s7, out[2].s7, out[3].s7), 0, dst + idx); }
    }
}

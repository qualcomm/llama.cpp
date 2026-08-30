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
// NO DYNAMICALLY INDEXED PRIVATE ARRAYS, ANYWHERE. This is not a style preference. The
// accumulators are indexed by column; written as `int s0[8]` they are dynamically indexed,
// which puts them in private memory and costs a scratch round-trip on every single dp4a.
// That build never finished one pp2 pass on muse-glimmer-30B -- the process sat on the GPU
// with its CPU time frozen for minutes. A `#pragma unroll` on the column loop is only a
// hint and did NOT rescue it. Vector components are addressed by name and cannot be
// spilled that way, so the column dimension is written out explicitly, and the reduction
// uses named registers rather than a `float8 out[4]` indexed by the row loop variable.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#ifdef cl_khr_integer_dot_product
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#endif

#define QK_K          256
#define K_SCALE_SIZE   12

#ifndef COK_NSG
#define COK_NSG 8
#endif

// Debug bisect, selected at build time from GGML_OPENCL_Q4K_COK_DP4A_STAGE:
//   1 = launch geometry only (store zeros, no K loop)
//   2 = K loop, no cross-subgroup reduction
//   3 = full kernel (default)
// One build serves all three, so localising a stall costs three runs and no rebuild.
#ifndef COK_STAGE
#define COK_STAGE 3
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

    // Columns past n_no_padding are computed and discarded at the store. Clamp them to a
    // real column so every read stays in bounds and initialised -- the host then needs no
    // zero-padded activation buffer, which it could not fill with clEnqueueFillBuffer
    // anyway while a recordable queue is capturing. Hoisted out of the K loop: these
    // depend only on the dispatch.
    const int nl = n_no_padding - 1;
    const int c0 = 0;
    const int c1 = (1 < n_no_padding) ? 1 : nl;
    const int c2 = (2 < n_no_padding) ? 2 : nl;
    const int c3 = (3 < n_no_padding) ? 3 : nl;
    const int c4 = (4 < n_no_padding) ? 4 : nl;
    const int c5 = (5 < n_no_padding) ? 5 : nl;
    const int c6 = (6 < n_no_padding) ? 6 : nl;
    const int c7 = (7 < n_no_padding) ? 7 : nl;

    // Scaled results, one float8 per folded row: 8 columns in the vector lanes, as cok.
    float8 acc0 = (float8)(0.0f), acc1 = (float8)(0.0f);
    float8 acc2 = (float8)(0.0f), acc3 = (float8)(0.0f);

#if COK_STAGE == 1
    // Every work-item returns, so this is uniform and the barriers below are not reached.
    if (sg == 0 && row0 < m) {
        vstore4((float4)(0.0f, 0.0f, 0.0f, 0.0f), 0, dst + row0);
    }
    return;
#endif

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

        // Raw int dot per (folded row, column), reset each 32-block because the weight
        // scale and min above are per 32-block. int8 VECTORS, not int[8]: the column index
        // has to be a name, never a variable (see the header).
        int8 s0 = (int8)(0), s1 = (int8)(0), s2 = (int8)(0), s3 = (int8)(0);

        for (int u = 0; u < 8; ++u) {
            const int ku = (i >> 2) + u;                       // K/4 index
            ushort4 bits = vload4(0, src0_q + row0 + ku * m);  // 4 rows x 4 K nibbles
            const uint w0 = EXP4(bits.s0);
            const uint w1 = EXP4(bits.s1);
            const uint w2 = EXP4(bits.s2);
            const uint w3 = EXP4(bits.s3);

            const uint a0 = src1_qa[(uint)c0 * k_u + ku];
            const uint a1 = src1_qa[(uint)c1 * k_u + ku];
            const uint a2 = src1_qa[(uint)c2 * k_u + ku];
            const uint a3 = src1_qa[(uint)c3 * k_u + ku];
            const uint a4 = src1_qa[(uint)c4 * k_u + ku];
            const uint a5 = src1_qa[(uint)c5 * k_u + ku];
            const uint a6 = src1_qa[(uint)c6 * k_u + ku];
            const uint a7 = src1_qa[(uint)c7 * k_u + ku];

            s0.s0 = dot_acc_sat_4x8packed_ss_int(w0, a0, s0.s0);
            s0.s1 = dot_acc_sat_4x8packed_ss_int(w0, a1, s0.s1);
            s0.s2 = dot_acc_sat_4x8packed_ss_int(w0, a2, s0.s2);
            s0.s3 = dot_acc_sat_4x8packed_ss_int(w0, a3, s0.s3);
            s0.s4 = dot_acc_sat_4x8packed_ss_int(w0, a4, s0.s4);
            s0.s5 = dot_acc_sat_4x8packed_ss_int(w0, a5, s0.s5);
            s0.s6 = dot_acc_sat_4x8packed_ss_int(w0, a6, s0.s6);
            s0.s7 = dot_acc_sat_4x8packed_ss_int(w0, a7, s0.s7);

            s1.s0 = dot_acc_sat_4x8packed_ss_int(w1, a0, s1.s0);
            s1.s1 = dot_acc_sat_4x8packed_ss_int(w1, a1, s1.s1);
            s1.s2 = dot_acc_sat_4x8packed_ss_int(w1, a2, s1.s2);
            s1.s3 = dot_acc_sat_4x8packed_ss_int(w1, a3, s1.s3);
            s1.s4 = dot_acc_sat_4x8packed_ss_int(w1, a4, s1.s4);
            s1.s5 = dot_acc_sat_4x8packed_ss_int(w1, a5, s1.s5);
            s1.s6 = dot_acc_sat_4x8packed_ss_int(w1, a6, s1.s6);
            s1.s7 = dot_acc_sat_4x8packed_ss_int(w1, a7, s1.s7);

            s2.s0 = dot_acc_sat_4x8packed_ss_int(w2, a0, s2.s0);
            s2.s1 = dot_acc_sat_4x8packed_ss_int(w2, a1, s2.s1);
            s2.s2 = dot_acc_sat_4x8packed_ss_int(w2, a2, s2.s2);
            s2.s3 = dot_acc_sat_4x8packed_ss_int(w2, a3, s2.s3);
            s2.s4 = dot_acc_sat_4x8packed_ss_int(w2, a4, s2.s4);
            s2.s5 = dot_acc_sat_4x8packed_ss_int(w2, a5, s2.s5);
            s2.s6 = dot_acc_sat_4x8packed_ss_int(w2, a6, s2.s6);
            s2.s7 = dot_acc_sat_4x8packed_ss_int(w2, a7, s2.s7);

            s3.s0 = dot_acc_sat_4x8packed_ss_int(w3, a0, s3.s0);
            s3.s1 = dot_acc_sat_4x8packed_ss_int(w3, a1, s3.s1);
            s3.s2 = dot_acc_sat_4x8packed_ss_int(w3, a2, s3.s2);
            s3.s3 = dot_acc_sat_4x8packed_ss_int(w3, a3, s3.s3);
            s3.s4 = dot_acc_sat_4x8packed_ss_int(w3, a4, s3.s4);
            s3.s5 = dot_acc_sat_4x8packed_ss_int(w3, a5, s3.s5);
            s3.s6 = dot_acc_sat_4x8packed_ss_int(w3, a6, s3.s6);
            s3.s7 = dot_acc_sat_4x8packed_ss_int(w3, a7, s3.s7);
        }

        // q4_K value is (q*scale - min), so per 32-block:
        //   out += scale * d_act * dot(q, a)  -  min * sum_act
        // where sum_act is q8_1's block sum (already carrying d_act).
        float8 da, sa;
        da.s0 = (float)src1_da[c0*k_b + blk];  sa.s0 = (float)src1_sa[c0*k_b + blk];
        da.s1 = (float)src1_da[c1*k_b + blk];  sa.s1 = (float)src1_sa[c1*k_b + blk];
        da.s2 = (float)src1_da[c2*k_b + blk];  sa.s2 = (float)src1_sa[c2*k_b + blk];
        da.s3 = (float)src1_da[c3*k_b + blk];  sa.s3 = (float)src1_sa[c3*k_b + blk];
        da.s4 = (float)src1_da[c4*k_b + blk];  sa.s4 = (float)src1_sa[c4*k_b + blk];
        da.s5 = (float)src1_da[c5*k_b + blk];  sa.s5 = (float)src1_sa[c5*k_b + blk];
        da.s6 = (float)src1_da[c6*k_b + blk];  sa.s6 = (float)src1_sa[c6*k_b + blk];
        da.s7 = (float)src1_da[c7*k_b + blk];  sa.s7 = (float)src1_sa[c7*k_b + blk];

        acc0 += sc0 * da * convert_float8(s0) - mv0 * sa;
        acc1 += sc1 * da * convert_float8(s1) - mv1 * sa;
        acc2 += sc2 * da * convert_float8(s2) - mv2 * sa;
        acc3 += sc3 * da * convert_float8(s3) - mv3 * sa;
    }

#if COK_STAGE == 2
    if (sg == 0 && row0 < m) {
        vstore4((float4)(acc0.s0, acc1.s0, acc2.s0, acc3.s0), 0, dst + row0);
    }
    return;
#endif

    // Cross-subgroup reduction over the K-split, one row at a time so the __local buffer
    // stays the size of the 1-row kernel's -- same shape as cok_r4. Written out per row
    // rather than looping over a `float8 out[4]`: that array was indexed by the loop
    // variable, which is the private-memory trap this kernel exists to avoid.
    local float8 reduceLM[COK_SG * (COK_NSG - 1)];
    float8 out0 = (float8)(0.0f), out1 = (float8)(0.0f);
    float8 out2 = (float8)(0.0f), out3 = (float8)(0.0f);

#define COK_REDUCE(accv, outv)                                       \
    barrier(CLK_LOCAL_MEM_FENCE);                                    \
    if (sg > 0) { reduceLM[(sg - 1) * COK_SG + lane] = (accv); }     \
    barrier(CLK_LOCAL_MEM_FENCE);                                    \
    if (sg == 0) {                                                   \
        float8 sum = (accv);                                         \
        for (int s = 0; s < COK_NSG - 1; s++) {                      \
            sum += reduceLM[s * COK_SG + lane];                      \
        }                                                            \
        (outv) = sum;                                                \
    }

    COK_REDUCE(acc0, out0)
    COK_REDUCE(acc1, out1)
    COK_REDUCE(acc2, out2)
    COK_REDUCE(acc3, out3)

#undef COK_REDUCE

    if (sg == 0) {
        // dst is [token, feature]: four adjacent rows are contiguous, one vstore4.
        int idx = row0;
        if (idx < m*n_no_padding) { vstore4((float4)(out0.s0, out1.s0, out2.s0, out3.s0), 0, dst + idx); idx += m; }
        if (idx < m*n_no_padding) { vstore4((float4)(out0.s1, out1.s1, out2.s1, out3.s1), 0, dst + idx); idx += m; }
        if (idx < m*n_no_padding) { vstore4((float4)(out0.s2, out1.s2, out2.s2, out3.s2), 0, dst + idx); idx += m; }
        if (idx < m*n_no_padding) { vstore4((float4)(out0.s3, out1.s3, out2.s3, out3.s3), 0, dst + idx); idx += m; }
        if (idx < m*n_no_padding) { vstore4((float4)(out0.s4, out1.s4, out2.s4, out3.s4), 0, dst + idx); idx += m; }
        if (idx < m*n_no_padding) { vstore4((float4)(out0.s5, out1.s5, out2.s5, out3.s5), 0, dst + idx); idx += m; }
        if (idx < m*n_no_padding) { vstore4((float4)(out0.s6, out1.s6, out2.s6, out3.s6), 0, dst + idx); idx += m; }
        if (idx < m*n_no_padding) { vstore4((float4)(out0.s7, out1.s7, out2.s7, out3.s7), 0, dst + idx); }
    }
}

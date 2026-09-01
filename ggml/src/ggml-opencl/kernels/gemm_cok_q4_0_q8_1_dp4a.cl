// cok-shaped q4_0 GEMM with a dp4a inner product, for the narrow band ne1 = 2..4.
//
// Third of the family, after q4_K and q6_K. q4_K + q6_K covered muse-glimmer-30B; q4_0 is
// what Qwen3.8-27B-Q4_0 and Nemotron-30B-A3B-Q4_0 are made of, neither of which the other
// two touch.
//
// SIZED BY skills/opencl-adreno SKILL.md §11 before writing, not by analogy:
//
//  §11.1 packing axis. Per row per 32-K block a half8 FMA issues 32 ops (eight columns wide
//        whatever ne1 is) and dp4a issues 8 K-groups x C columns = 8C. q4_0 packs 4 weights
//        per ushort exactly as q4_K does, so the arithmetic is identical: C=2 -> 16 ops
//        (win), C=4 -> 32 (parity, cheaper op), C=8 -> 64 (loss). Only 2- and 4-column
//        builds exist; an 8-column build is known to lose before it is written.
//  §11.2 registers. This is the LIGHTEST of the three: 4 float4 acc + 4 int4 dots + 4 uint4
//        activation + 4 weights. q6_K measured 400 B at the same shape and carries two dot
//        flushes per block; this should come in under it. Cliff is 512 B/WI.
//  §11.3 loads are 16 B: uint4, never a wider vload.
//  §11.6 no dynamically indexed private arrays; column indices are compile-time constants so
//        the address arithmetic stays affine.
//
// SIMPLEST OF THE FAMILY. A q4_0 value is (q - 8) * d with ONE scale per 32-K block and no
// min, so there is no `- min * sum_act` correction (q4_K), no packed scale pair to select
// (q6_K), and the dot flushes once per block rather than twice. The -8 bias folds into the
// byte packing, which keeps the dot exact.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#ifdef cl_khr_integer_dot_product
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#endif

#ifndef COK_NSG
#define COK_NSG 4
#endif
#define COK_SG  64

#ifndef COK_ROWS
#define COK_ROWS 4
#endif
#ifndef COK_COLS
#define COK_COLS 4
#endif

#if COK_COLS == 4
typedef float4 cok_accv;
typedef int4   cok_dotv;
#define COK_CONVF convert_float4
#else
typedef float2 cok_accv;
typedef int2   cok_dotv;
#define COK_CONVF convert_float2
#endif

// Four consecutive-K q4_0 weights for one row as four signed bytes in a uint, with the -8
// bias folded in so the dot is exact and needs no correction term. (q - 8) is in [-8, 7].
#define EXP40(l)                                                                      \
    ( (((uint)((int)(( (l)         & 0x000Fu)       - 8) & 0xFF)))                    \
    | (((uint)((int)((((l) & 0x00F0u) >>  4)        - 8) & 0xFF)) <<  8)              \
    | (((uint)((int)((((l) & 0x0F00u) >>  8)        - 8) & 0xFF)) << 16)              \
    | (((uint)((int)((((l) & 0xF000u) >> 12)        - 8) & 0xFF)) << 24) )

// 8 rows per lane. The weight load reaches 16 B per lane, matching the q4_0 GEMV that
// sustains ~117 GB/s where this kernel manages ~88 on identical traffic; 4 rows loads 8 B
// and 1 row loads 2. Costs 8 accumulators and 8 dot registers, which spills -- acceptable
// only if the kernel really is load-bound. Needs ne01 % (64*8) == 0; the host gates it.
#if COK_ROWS == 8
#define COK_DOT(ci, t)                                                \
    d0.s##ci = dot_acc_sat_4x8packed_ss_int(w0, A##ci.s##t, d0.s##ci);\
    d1.s##ci = dot_acc_sat_4x8packed_ss_int(w1, A##ci.s##t, d1.s##ci);\
    d2.s##ci = dot_acc_sat_4x8packed_ss_int(w2, A##ci.s##t, d2.s##ci);\
    d3.s##ci = dot_acc_sat_4x8packed_ss_int(w3, A##ci.s##t, d3.s##ci);\
    d4.s##ci = dot_acc_sat_4x8packed_ss_int(w4, A##ci.s##t, d4.s##ci);\
    d5.s##ci = dot_acc_sat_4x8packed_ss_int(w5, A##ci.s##t, d5.s##ci);\
    d6.s##ci = dot_acc_sat_4x8packed_ss_int(w6, A##ci.s##t, d6.s##ci);\
    d7.s##ci = dot_acc_sat_4x8packed_ss_int(w7, A##ci.s##t, d7.s##ci);
#elif COK_ROWS == 4
#define COK_DOT(ci, t)                                                \
    d0.s##ci = dot_acc_sat_4x8packed_ss_int(w0, A##ci.s##t, d0.s##ci); \
    d1.s##ci = dot_acc_sat_4x8packed_ss_int(w1, A##ci.s##t, d1.s##ci); \
    d2.s##ci = dot_acc_sat_4x8packed_ss_int(w2, A##ci.s##t, d2.s##ci); \
    d3.s##ci = dot_acc_sat_4x8packed_ss_int(w3, A##ci.s##t, d3.s##ci);
#else
#define COK_DOT(ci, t)                                                 \
    d0.s##ci = dot_acc_sat_4x8packed_ss_int(w0, A##ci.s##t, d0.s##ci); \
    d1.s##ci = dot_acc_sat_4x8packed_ss_int(w1, A##ci.s##t, d1.s##ci);
#endif

#if COK_COLS == 4
#define COK_DOTS_AT(t)  COK_DOT(0,t) COK_DOT(1,t) COK_DOT(2,t) COK_DOT(3,t)
#define COK_FOR_COLS(F) F(0) F(1) F(2) F(3)
#else
#define COK_DOTS_AT(t)  COK_DOT(0,t) COK_DOT(1,t)
#define COK_FOR_COLS(F) F(0) F(1)
#endif

// One K-group (4 K values): unpack the folded rows' weights, then dot every column.
#if COK_ROWS == 8
#define COK_KSTEP(t)                                                  \
    {                                                                 \
    ushort8 bl = vload8(0, src0_q + row0 + (ku0 + t) * m);            \
    const uint w0 = EXP40(bl.s0);                                     \
    const uint w1 = EXP40(bl.s1);                                     \
    const uint w2 = EXP40(bl.s2);                                     \
    const uint w3 = EXP40(bl.s3);                                     \
    const uint w4 = EXP40(bl.s4);                                     \
    const uint w5 = EXP40(bl.s5);                                     \
    const uint w6 = EXP40(bl.s6);                                     \
    const uint w7 = EXP40(bl.s7);                                     \
    COK_DOTS_AT(t)                                                    \
    }
#elif COK_ROWS == 4
#define COK_KSTEP(t)                                                   \
    {                                                                  \
    ushort4 bl = vload4(0, src0_q + row0 + (ku0 + t) * m);             \
    const uint w0 = EXP40(bl.s0);                                      \
    const uint w1 = EXP40(bl.s1);                                      \
    const uint w2 = EXP40(bl.s2);                                      \
    const uint w3 = EXP40(bl.s3);                                      \
    COK_DOTS_AT(t)                                                     \
    }
#else
#define COK_KSTEP(t)                                                   \
    {                                                                  \
    ushort2 bl = vload2(0, src0_q + row0 + (ku0 + t) * m);             \
    const uint w0 = EXP40(bl.s0);                                      \
    const uint w1 = EXP40(bl.s1);                                      \
    COK_DOTS_AT(t)                                                     \
    }
#endif

kernel void kernel_gemm_cok_q4_0_q8_1_dp4a(
    global const ushort * src0_q,     // q4_0 nibble plane [row + (K/4)*m]
    global const half   * src0_d,     // one scale per 32-K block [row + blk*m]
    global const uint   * src1_qa,    // q8_1 activations  [col*k_u + K/4]
    global const half   * src1_da,    // activation scale  [col*k_b + blk]
    global float * dst,
    ulong offsetd,
    int m,
    int n,
    int k,
    int n_no_padding
) {
    dst = (global float *)((global char *)dst + offsetd);

    const int gx   = get_global_id(0);
    const int sg   = get_local_id(1);
    const int lane = get_local_id(0);

    const int row0 = gx * COK_ROWS;
    const int num_32blk = k / 32;
    const int k_u = k >> 2;
    const int k_b = k >> 5;

    // Compile-time column indices: the activation is allocated at the kernel's width, so
    // nothing is clamped and the address arithmetic stays affine (SKILL.md 11.6).
    const int c0 = 0, c1 = 1;
#if COK_COLS == 4
    const int c2 = 2, c3 = 3;
#endif

    cok_accv acc0 = (cok_accv)(0.0f), acc1 = (cok_accv)(0.0f);
#if COK_ROWS == 8
    cok_accv acc2 = (cok_accv)(0.0f), acc3 = (cok_accv)(0.0f);
    cok_accv acc4 = (cok_accv)(0.0f), acc5 = (cok_accv)(0.0f);
    cok_accv acc6 = (cok_accv)(0.0f), acc7 = (cok_accv)(0.0f);
#elif COK_ROWS == 4
    cok_accv acc2 = (cok_accv)(0.0f), acc3 = (cok_accv)(0.0f);
#endif

    for (int blk = sg; blk < num_32blk; blk += COK_NSG) {
#if COK_ROWS == 8
        half8 scale = vload8(0, src0_d + row0 + blk * m);
#elif COK_ROWS == 4
        half4 scale = vload4(0, src0_d + row0 + blk * m);
#else
        half2 scale = vload2(0, src0_d + row0 + blk * m);
#endif

        cok_accv da;
        da.s0 = (float)src1_da[c0*k_b + blk];
        da.s1 = (float)src1_da[c1*k_b + blk];
#if COK_COLS == 4
        da.s2 = (float)src1_da[c2*k_b + blk];
        da.s3 = (float)src1_da[c3*k_b + blk];
#endif

        // One flush per 32-K block: q4_0 has a single scale over the whole block, so the
        // dot accumulates across both halves and is scaled once. (q6_K needs two flushes
        // because its scale changes every 16 K; q4_K needs a min term as well.)
        cok_dotv d0 = (cok_dotv)(0), d1 = (cok_dotv)(0);
#if COK_ROWS == 8
        cok_dotv d2 = (cok_dotv)(0), d3 = (cok_dotv)(0);
        cok_dotv d4 = (cok_dotv)(0), d5 = (cok_dotv)(0);
        cok_dotv d6 = (cok_dotv)(0), d7 = (cok_dotv)(0);
#elif COK_ROWS == 4
        cok_dotv d2 = (cok_dotv)(0), d3 = (cok_dotv)(0);
#endif
        for (int half_idx = 0; half_idx < 2; ++half_idx) {
            const int ku0 = (blk << 3) + half_idx * 4;

            uint4 A0 = vload4(0, src1_qa + (uint)c0 * k_u + ku0);
            uint4 A1 = vload4(0, src1_qa + (uint)c1 * k_u + ku0);
#if COK_COLS == 4
            uint4 A2 = vload4(0, src1_qa + (uint)c2 * k_u + ku0);
            uint4 A3 = vload4(0, src1_qa + (uint)c3 * k_u + ku0);
#endif
            COK_KSTEP(0)
            COK_KSTEP(1)
            COK_KSTEP(2)
            COK_KSTEP(3)
        }

        acc0 += (float)scale.s0 * da * COK_CONVF(d0);
        acc1 += (float)scale.s1 * da * COK_CONVF(d1);
#if COK_ROWS == 8
        acc2 += (float)scale.s2 * da * COK_CONVF(d2);
        acc3 += (float)scale.s3 * da * COK_CONVF(d3);
        acc4 += (float)scale.s4 * da * COK_CONVF(d4);
        acc5 += (float)scale.s5 * da * COK_CONVF(d5);
        acc6 += (float)scale.s6 * da * COK_CONVF(d6);
        acc7 += (float)scale.s7 * da * COK_CONVF(d7);
#elif COK_ROWS == 4
        acc2 += (float)scale.s2 * da * COK_CONVF(d2);
        acc3 += (float)scale.s3 * da * COK_CONVF(d3);
#endif
    }

    // Cross-subgroup reduction over the K-split, one row at a time. Named registers, never an
    // out[] array indexed by the loop variable.
    local cok_accv reduceLM[COK_SG * (COK_NSG - 1)];
    cok_accv out0 = (cok_accv)(0.0f), out1 = (cok_accv)(0.0f);
#if COK_ROWS == 8
    cok_accv out2 = (cok_accv)(0.0f), out3 = (cok_accv)(0.0f);
    cok_accv out4 = (cok_accv)(0.0f), out5 = (cok_accv)(0.0f);
    cok_accv out6 = (cok_accv)(0.0f), out7 = (cok_accv)(0.0f);
#elif COK_ROWS == 4
    cok_accv out2 = (cok_accv)(0.0f), out3 = (cok_accv)(0.0f);
#endif

#define COK_REDUCE(accv, outv)                                       \
    barrier(CLK_LOCAL_MEM_FENCE);                                    \
    if (sg > 0) { reduceLM[(sg - 1) * COK_SG + lane] = (accv); }     \
    barrier(CLK_LOCAL_MEM_FENCE);                                    \
    if (sg == 0) {                                                   \
        cok_accv sum = (accv);                                       \
        for (int s = 0; s < COK_NSG - 1; s++) {                      \
            sum += reduceLM[s * COK_SG + lane];                      \
        }                                                            \
        (outv) = sum;                                                \
    }

    COK_REDUCE(acc0, out0)
    COK_REDUCE(acc1, out1)
#if COK_ROWS == 8
    COK_REDUCE(acc2, out2)
    COK_REDUCE(acc3, out3)
    COK_REDUCE(acc4, out4)
    COK_REDUCE(acc5, out5)
    COK_REDUCE(acc6, out6)
    COK_REDUCE(acc7, out7)
#elif COK_ROWS == 4
    COK_REDUCE(acc2, out2)
    COK_REDUCE(acc3, out3)
#endif

#undef COK_REDUCE

#if COK_ROWS == 8
#define COK_STORE_COL(ci)                                             \
    if (idx < m*n_no_padding) {                                       \
        vstore8((float8)(out0.s##ci, out1.s##ci, out2.s##ci, out3.s##ci,\
                         out4.s##ci, out5.s##ci, out6.s##ci, out7.s##ci),\
                0, dst + idx);                                        \
        idx += m;                                                     \
    }
#elif COK_ROWS == 4
#define COK_STORE_COL(ci)                                                                 \
    if (idx < m*n_no_padding) {                                                           \
        vstore4((float4)(out0.s##ci, out1.s##ci, out2.s##ci, out3.s##ci), 0, dst + idx);  \
        idx += m;                                                                         \
    }
#else
#define COK_STORE_COL(ci)                                                                 \
    if (idx < m*n_no_padding) {                                                           \
        vstore2((float2)(out0.s##ci, out1.s##ci), 0, dst + idx);                          \
        idx += m;                                                                         \
    }
#endif

    if (sg == 0) {
        int idx = row0;
        COK_FOR_COLS(COK_STORE_COL)
    }

#undef COK_STORE_COL
}

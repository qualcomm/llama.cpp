// Cooperative-K q4_K GEMM with a dp4a inner product, for narrow batches (ne1 = 2..4).
//
// The dense dp4a GEMM amortises its weight read and scale unpack over a wide activation
// tile, so narrowing it does not help at these widths. This keeps the cooperative-K shape
// instead -- COK_ROWS adjacent output rows folded into a lane, so one weight read and one
// scale/min unpack serve all of them, and K split across COK_NSG subgroups reduced through
// local memory -- and changes only the inner product. The activation is quantized to q8_1
// by kernel_quant_a_q8_1 beforehand.
//
// COK_COLS (4 or 2) is the number of columns a lane computes; the host builds both and
// picks by ne1, since columns past ne1 are computed and discarded at the store.
//
// Accumulators are indexed by column, so they are vector components addressed by name
// rather than an array: a dynamically indexed private array is placed in private memory
// and spills on every dp4a. The reduction uses named registers for the same reason.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#ifdef cl_khr_integer_dot_product
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#endif

#define QK_K          256
#define K_SCALE_SIZE   12

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

// Loads are 16 bytes (vload4 of a 32-bit type): a wider vload splits into two
// transactions and only adds live registers.

#if COK_COLS == 4
typedef float4 cok_accv;
typedef int4   cok_dotv;
#define COK_CONVF convert_float4
#else
typedef float2 cok_accv;
typedef int2   cok_dotv;
#define COK_CONVF convert_float2
#endif

// One packed q4_K ushort holds 4 consecutive-K nibbles for one row; spread them into the
// 4 bytes of a uint so dp4a can take it directly. Same expansion the prefill GEMM uses.
#define EXP4(u)  ( ((uint)((u) & 0x000Fu))        | \
                  (((uint)((u) & 0x00F0u)) << 4)  | \
                  (((uint)((u) & 0x0F00u)) << 8)  | \
                  (((uint)((u) & 0xF000u)) << 12) )

// Per-column work, expanded by name so no index is ever a variable. The activation comes
// from a uint4 component: four consecutive K-groups for one column are contiguous, so one
// vload4 replaces four scalar loads.
#if COK_ROWS == 4
#define COK_DOT(ci, t)                                                 \
    s0.s##ci = dot_acc_sat_4x8packed_ss_int(w0, A##ci.s##t, s0.s##ci);    \
    s1.s##ci = dot_acc_sat_4x8packed_ss_int(w1, A##ci.s##t, s1.s##ci);    \
    s2.s##ci = dot_acc_sat_4x8packed_ss_int(w2, A##ci.s##t, s2.s##ci);    \
    s3.s##ci = dot_acc_sat_4x8packed_ss_int(w3, A##ci.s##t, s3.s##ci);
#else
#define COK_DOT(ci, t)                                                 \
    s0.s##ci = dot_acc_sat_4x8packed_ss_int(w0, A##ci.s##t, s0.s##ci);    \
    s1.s##ci = dot_acc_sat_4x8packed_ss_int(w1, A##ci.s##t, s1.s##ci);
#endif

#if COK_COLS == 4
#define COK_DOTS_AT(t)  COK_DOT(0,t) COK_DOT(1,t) COK_DOT(2,t) COK_DOT(3,t)
#define COK_FOR_COLS(F) F(0) F(1) F(2) F(3)
#else
#define COK_DOTS_AT(t)  COK_DOT(0,t) COK_DOT(1,t)
#define COK_FOR_COLS(F) F(0) F(1)
#endif

// One K-group: unpack the weight nibbles for the folded rows, then dot every column.
#if COK_ROWS == 4
#define COK_KSTEP(t)                                                   \
    {                                                                     \
    ushort4 bits = vload4(0, src0_q + row0 + (ku0 + t) * m);              \
    const uint w0 = EXP4(bits.s0);                                        \
    const uint w1 = EXP4(bits.s1);                                        \
    const uint w2 = EXP4(bits.s2);                                        \
    const uint w3 = EXP4(bits.s3);                                        \
    COK_DOTS_AT(t)                                                        \
    }
#else
#define COK_KSTEP(t)                                                   \
    {                                                                     \
    ushort2 bits = vload2(0, src0_q + row0 + (ku0 + t) * m);              \
    const uint w0 = EXP4(bits.s0);                                        \
    const uint w1 = EXP4(bits.s1);                                        \
    COK_DOTS_AT(t)                                                        \
    }
#endif

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

    const int gx   = get_global_id(0);   // row group
    const int sg   = get_local_id(1);    // K-split subgroup
    const int lane = get_local_id(0);

    const int row0 = gx * COK_ROWS;
    const int num_32blk = k / 32;
    const int k_u = k >> 2;              // K in uint (int8x4) units
    const int k_b = k >> 5;              // 32-blocks along K

    // Columns are read up to `n` (the readable activation width) and stored up to
    // n_no_padding; columns past n are clamped to the last readable one. The host
    // allocates the activation at the kernel's width and passes it as n, so nothing is
    // clamped and every column reads a distinct address.
    const int nl = n - 1;
    const int c0 = 0;
    const int c1 = (1 < n) ? 1 : nl;
#if COK_COLS >= 4
    const int c2 = (2 < n) ? 2 : nl;
    const int c3 = (3 < n) ? 3 : nl;
#endif

    cok_accv acc0 = (cok_accv)(0.0f), acc1 = (cok_accv)(0.0f);
#if COK_ROWS == 4
    cok_accv acc2 = (cok_accv)(0.0f), acc3 = (cok_accv)(0.0f);
#endif

    for (int blk = sg; blk < num_32blk; blk += COK_NSG) {
        const int i       = blk << 5;
        const int sb_idx  = blk >> 3;
        const int sub_idx = blk & 7;

        global const uchar * sc = src0_s + sb_idx * K_SCALE_SIZE * m + row0;
        uchar sv0, mn0, sv1, mn1;
        get_scale_min_k4_c(sub_idx, sc + 0, m, &sv0, &mn0, mask_d6, mask_d4, mask_hi2);
        get_scale_min_k4_c(sub_idx, sc + 1, m, &sv1, &mn1, mask_d6, mask_d4, mask_hi2);
#if COK_ROWS == 4
        half4 dd  = vload4(0, src0_d  + row0 + sb_idx * m);
        half4 dmm = vload4(0, src0_dm + row0 + sb_idx * m);
        uchar sv2, mn2, sv3, mn3;
        get_scale_min_k4_c(sub_idx, sc + 2, m, &sv2, &mn2, mask_d6, mask_d4, mask_hi2);
        get_scale_min_k4_c(sub_idx, sc + 3, m, &sv3, &mn3, mask_d6, mask_d4, mask_hi2);
        const float sc2 = (float)dd.s2 * (float)sv2, mv2 = (float)dmm.s2 * (float)mn2;
        const float sc3 = (float)dd.s3 * (float)sv3, mv3 = (float)dmm.s3 * (float)mn3;
#else
        half2 dd  = vload2(0, src0_d  + row0 + sb_idx * m);
        half2 dmm = vload2(0, src0_dm + row0 + sb_idx * m);
#endif
        const float sc0 = (float)dd.s0 * (float)sv0, mv0 = (float)dmm.s0 * (float)mn0;
        const float sc1 = (float)dd.s1 * (float)sv1, mv1 = (float)dmm.s1 * (float)mn1;

        // Raw int dot per (folded row, column). Reset each 32-block because the weight
        // scale and min above are per 32-block. Vectors, never arrays.
        cok_dotv s0 = (cok_dotv)(0), s1 = (cok_dotv)(0);
#if COK_ROWS == 4
        cok_dotv s2 = (cok_dotv)(0), s3 = (cok_dotv)(0);
#endif

        // Two groups of four K-groups. The activation for one column across four
        // consecutive K-groups is contiguous, so it is one vload4 rather than four
        // scalar loads.
        for (int uq = 0; uq < 2; ++uq) {
            const int ku0 = (i >> 2) + uq * 4;

            uint4 A0 = vload4(0, src1_qa + (uint)c0 * k_u + ku0);
            uint4 A1 = vload4(0, src1_qa + (uint)c1 * k_u + ku0);
#if COK_COLS >= 4
            uint4 A2 = vload4(0, src1_qa + (uint)c2 * k_u + ku0);
            uint4 A3 = vload4(0, src1_qa + (uint)c3 * k_u + ku0);
#endif
            COK_KSTEP(0)
            COK_KSTEP(1)
            COK_KSTEP(2)
            COK_KSTEP(3)
        }

        // q4_K value is (q*scale - min), so per 32-block:
        //   out += scale * d_act * dot(q, a)  -  min * sum_act
        // where sum_act is q8_1's block sum (already carrying d_act).
        cok_accv da, sa;
        da.s0 = (float)src1_da[c0*k_b + blk];  sa.s0 = (float)src1_sa[c0*k_b + blk];
        da.s1 = (float)src1_da[c1*k_b + blk];  sa.s1 = (float)src1_sa[c1*k_b + blk];
#if COK_COLS >= 4
        da.s2 = (float)src1_da[c2*k_b + blk];  sa.s2 = (float)src1_sa[c2*k_b + blk];
        da.s3 = (float)src1_da[c3*k_b + blk];  sa.s3 = (float)src1_sa[c3*k_b + blk];
#endif

        acc0 += sc0 * da * COK_CONVF(s0) - mv0 * sa;
        acc1 += sc1 * da * COK_CONVF(s1) - mv1 * sa;
#if COK_ROWS == 4
        acc2 += sc2 * da * COK_CONVF(s2) - mv2 * sa;
        acc3 += sc3 * da * COK_CONVF(s3) - mv3 * sa;
#endif
    }

    // Cross-subgroup reduction over the K-split, one row at a time so the local buffer
    // holds one row's accumulators. Written out per row rather than looping over an out[]
    // array indexed by the loop variable.
    local cok_accv reduceLM[COK_SG * (COK_NSG - 1)];
    cok_accv out0 = (cok_accv)(0.0f), out1 = (cok_accv)(0.0f);
#if COK_ROWS == 4
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
#if COK_ROWS == 4
    COK_REDUCE(acc2, out2)
    COK_REDUCE(acc3, out3)
#endif

#undef COK_REDUCE

#if COK_ROWS == 4
#define COK_STORE_COL(ci)                                                                   \
    if (idx < m*n_no_padding) {                                                             \
        vstore4((float4)(out0.s##ci, out1.s##ci, out2.s##ci, out3.s##ci), 0, dst + idx);    \
        idx += m;                                                                           \
    }
#else
#define COK_STORE_COL(ci)                                                                   \
    if (idx < m*n_no_padding) {                                                             \
        vstore2((float2)(out0.s##ci, out1.s##ci), 0, dst + idx);                            \
        idx += m;                                                                           \
    }
#endif

    if (sg == 0) {
        // dst is [token, feature]: the folded rows are adjacent, so one vector store.
        int idx = row0;
        COK_FOR_COLS(COK_STORE_COL)
    }

#undef COK_STORE_COL
}

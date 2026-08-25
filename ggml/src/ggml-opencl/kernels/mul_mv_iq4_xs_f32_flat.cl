#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// IQ4_XS decode GEMV over the feature-major plane split.
//
// kernel_convert_block_iq4_xs_ns splits the tensor in place into four planes
// (the split is size preserving, 128 + 2 + 2 + 4 == 136 == sizeof(block_iq4_xs))
// and transposes each one, so every plane is indexed [k-group][row]:
//
//   src0_q [row + (k/4)*m]    ushort = 4 codebook indices, K = 4*grp .. +3
//   src0_d [row + (k/256)*m]  half   = super-block scale
//   src0_sh[row + (k/256)*m]  ushort = scales_h
//   src0_sl[row + (k/256)*m]  uint   = the four scales_l bytes
//
// One work item owns one output row, so lane i and lane i+1 read adjacent
// ushorts and a wave's weight read is one contiguous run. That is the whole
// reason this kernel exists: the AoS kernel_mul_mv_iq4_xs_f32 gives every lane
// a 16 byte slice at a 136 byte stride instead.
//
// 🔴 One row per lane on its own is 2.5x SLOWER than the AoS kernel, because it
// launches only M work items where the AoS one launches 16*M and the GPU is
// left with ~48 waves. K is therefore split across IQ4XS_MV_NSG subgroups whose
// partials are reduced through local memory -- the same trick, and the same
// reason, as kernel_gemv_noshuffle_q4_k_f32's lws=64xNSG.
//
// The sub-scale is rebuilt exactly as the dp4a GEMM does:
//   ls  = scales_l nibble | ((scales_h >> 2*sb) & 3) << 4
//   d_w = d * (ls - 32)

#define QK_K 256

// Subgroups per workgroup; each takes every NSG'th super-block along K.
#ifndef IQ4XS_MV_NSG
#define IQ4XS_MV_NSG 8
#endif

// IQ4XS_MV_R2=1: one lane owns TWO adjacent rows and reads their two ushorts as
// a single uint, so a 64-lane wave moves 256 bytes per weight load instead of
// 128. Needs m even, which the %64 rule on the Adreno path already guarantees.
#ifndef IQ4XS_MV_R2
#define IQ4XS_MV_R2 1
#endif

// IQ4XS_MV_R4=1: four rows per lane instead of two, so a lane's weight load is a
// uint2 (8 bytes) and a 64-lane wave moves 512 bytes instead of 256.
//
// The reason to expect anything: the same move on the q4_K/q6_K cok kernels was
// +43.8%, because those GEMVs were ISSUE-bound rather than bandwidth-bound at 2
// bytes per lane. It was also tried on the IQ3_S GEMV and LOST -- but that was
// register pressure from four live grid indices, and IQ4_XS no longer has a grid
// to be live: under IQ4XS_MV_CB=3 the codebook is immediates.
//
// MEASURED, AND IT DOES NOT SETTLE: X2-90 tg, Llama-3.2-3B IQ4_XS 40.09 -> 40.69
// (+1.5%) but Qwen3.8-27B UD-IQ4_XS 5.279 -> 5.118 (-3.0%). Same device, same
// kernel, opposite sign by model shape -- 256 rows per workgroup halves the
// workgroup count, and the float4 partials double this kernel's local memory to
// 8 KB. Left off; a shape-dependent default is not worth 1.5%.
//
// If it is ever turned on: the split gate only requires ne01 % 2 == 0, so a
// tensor with ne01 % 4 == 2 would drop its tail rows silently. That needs a
// per-dispatch choice between an R2 and an R4 kernel, not just this flag.
#ifndef IQ4XS_MV_R4
#define IQ4XS_MV_R4 0
#endif

// IQ4XS_MV_ABL=1: COST PROBE, WRONG MATH. Replaces the per-operand activation
// load with a constant, keeping every weight load and all the ALU. The question
// it answers: this kernel reads 128 bytes of f32 activations per 32-K step per
// LANE against ~34 bytes of weights for its row pair, and all 64 lanes of a wave
// read the SAME addresses. The tuned gemv_noshuffle_q4_k_f32 loads them on four
// lanes and sub_group_broadcasts instead. If that redundancy is already free on
// X2 the probe changes nothing and the broadcast rewrite is not worth doing.
#ifndef IQ4XS_MV_ABL
#define IQ4XS_MV_ABL 0
#endif

// 1 = drop the activation load. 2 = drop the CODEBOOK lookup (use the raw nibble),
// keeping every load. Probe 1 measured only +1.9% (3B) / +4.7% (27B), i.e. the
// wave-uniform activation read is almost free -- so probe 2 asks whether the 16
// divergent kvalues_iq4nl[] lookups per operand pair are what is left.
#if IQ4XS_MV_ABL == 1
#define IQ4XS_YV(g, y) ((float4)(1.0f))
#else
#define IQ4XS_YV(g, y) vload4((g), (y))
#endif

// IQ4XS_MV_CB: how a nibble becomes its codebook value. This, and not the loads
// or the activations, is what the kernel spends its time on.
//
// The ABL=2 probe settles it: dropping the lookup entirely (wrong math) is +51%
// on a 3B and +21% on the 27B. It is also why the tuned gemv_noshuffle_q4_k_f32
// is 1.72x faster at matched shapes -- q4_K is a LINEAR quant with no table at
// all, so this is a cost the IQ types carry, not a trick copied wrong.
//
// The index is per weight and therefore DIVERGENT, which is the part that hurts.
// Adreno only takes __constant through the uniform path when the index is
// wave-uniform, so each of these is a real cache load, and at eight nibbles per
// uint of weights they outnumber the weight loads 8:1.
//
//   0  __constant float[16] indexed by the nibble (the metal-derived original)
//   1  __constant uint[4], byte-extracted by shift -- shipped 2026-08-24 for
//      +8.1% on the 27B, i.e. only ~40% of the ABL=2 ceiling, so 1 is not the end
//   2  the same float[16] staged in LOCAL memory. Sixteen consecutive floats sit
//      in sixteen different LDS banks, so a fully divergent read is conflict
//      free; precedent is the IQ2_S grid, where __constant -> LDS was +55%
//   3  the four uints as IMMEDIATES, picked by a select chain, so the table never
//      leaves the register file. Three selects per nibble, zero memory traffic
//   4  OpenCL shuffle() over a uchar16 register table -- one builtin per FOUR
//      nibbles, the closest thing available to CUDA's byte-permute
//   5  control for 4: the same float4-at-a-time restructuring over the mode-1
//      table. 4 vs 5 isolates shuffle(), 5 vs 1 isolates the restructuring
//   6  one ulong immediate pair, picked by a single select and byte-extracted by
//      a 64-bit variable shift
//
// 4 and 5 sum with dot() rather than four sequential adds, so they are not
// bit-identical to the others; anything that ships has to clear the decode-path
// perplexity oracle, not just llama-bench.
//
// Measured X2-90, Llama-3.2-3B IQ4_XS tg64 / Qwen3.8-27B UD-IQ4_XS tg32, one
// binary, arms bracketed:
//
//   mode      0      1      2      3      4      5      6   | ABL=2 ceiling
//   3B    29.61  33.28  20.84  40.15  27.09  33.76  27.27   |   44.86
//   27B      --   4.81     --   5.28     --     --     --   |    5.40
//
// 3 is the default: +20.6% on the 3B and +9.7% on the 27B over the shipped
// mode 1, which is 59% and 79% of what dropping the lookup entirely would buy.
// Three results worth not re-deriving:
//   - 5 vs 1 is +1.4%, so the float4 restructuring is not the lever
//   - 4 vs 5 is -20%: there is no byte-permute here and shuffle() scalarizes
//   - 2 is -37%. LOCAL memory is the wrong tool at 64 bytes; the +55% the IQ2_S
//     GEMV took from the same move was a 8 KB table, a different regime
#ifndef IQ4XS_MV_CB
#define IQ4XS_MV_CB 3
#endif

constant float kvalues_iq4nl[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f,
      1.f,   13.f,  25.f,  38.f,  53.f,  69.f,  89.f, 113.f
};

__constant uint kvalues_iq4nl_i8x4[4] = {
    0xBFAD9881u, 0xF6EADDCFu, 0x26190D01u, 0x71594535u
};

inline float iq4nl_cbf(uint n) {
    return (float)(char)((kvalues_iq4nl_i8x4[n >> 2] >> ((n & 3u) * 8u)) & 0xFFu);
}

// n >> 2 picks the uint; its two bits are (n & 4) and (n & 8).
inline float iq4nl_cbsel(uint n) {
    const uint t01 = (n & 4u) ? 0xF6EADDCFu : 0xBFAD9881u;
    const uint t23 = (n & 4u) ? 0x71594535u : 0x26190D01u;
    const uint t   = (n & 8u) ? t23 : t01;
    return (float)(char)((t >> ((n & 3u) * 8u)) & 0xFFu);
}

inline float iq4nl_cbsel64(uint n) {
    const ulong t = (n & 8u) ? 0x7159453526190D01ul : 0xF6EADDCFBFAD9881ul;
    return (float)(char)((uint)((t >> ((n & 7u) * 8u)) & 0xFFul));
}

inline float4 iq4nl_cb4_pack(uint w) {
    return (float4)(iq4nl_cbf( w        & 0xFu), iq4nl_cbf((w >>  4) & 0xFu),
                    iq4nl_cbf((w >>  8) & 0xFu), iq4nl_cbf((w >> 12) & 0xFu));
}

// shuffle() indexes by the four low bits of each mask element, which is exactly a
// 16-entry byte table held in registers.
inline float4 iq4nl_cb4_shuf(uint w) {
    const uchar16 tbl = (uchar16)(0x81, 0x98, 0xAD, 0xBF, 0xCF, 0xDD, 0xEA, 0xF6,
                                  0x01, 0x0D, 0x19, 0x26, 0x35, 0x45, 0x59, 0x71);
    const uchar4  idx = (uchar4)((uchar)( w        & 0xFu), (uchar)((w >>  4) & 0xFu),
                                 (uchar)((w >>  8) & 0xFu), (uchar)((w >> 12) & 0xFu));
    return convert_float4(as_char4(shuffle(tbl, idx)));
}

#if IQ4XS_MV_ABL == 2
#define IQ4XS_CB_DECL
#define IQ4XS_CB(n) ((float)(n))
#elif IQ4XS_MV_CB == 0
#define IQ4XS_CB_DECL
#define IQ4XS_CB(n) kvalues_iq4nl[(n)]
#elif IQ4XS_MV_CB == 2
#define IQ4XS_CB_DECL                                                          \
    __local float iq4nl_lds[16];                                               \
    if (get_local_id(1) == 0 && get_local_id(0) < 16) {                        \
        iq4nl_lds[get_local_id(0)] = kvalues_iq4nl[get_local_id(0)];           \
    }                                                                          \
    barrier(CLK_LOCAL_MEM_FENCE);
#define IQ4XS_CB(n) iq4nl_lds[(n)]
#elif IQ4XS_MV_CB == 3
#define IQ4XS_CB_DECL
#define IQ4XS_CB(n) iq4nl_cbsel((n))
#elif IQ4XS_MV_CB == 6
#define IQ4XS_CB_DECL
#define IQ4XS_CB(n) iq4nl_cbsel64((n))
#else
#define IQ4XS_CB_DECL
#define IQ4XS_CB(n) iq4nl_cbf((n))
#endif

// One super-block quarter: four nibbles of `w` against the float4 `yv`.
#if IQ4XS_MV_ABL != 2 && IQ4XS_MV_CB == 4
#define IQ4XS_ACC(a, yv, w) a += dot((yv), iq4nl_cb4_shuf((uint)(w)))
#elif IQ4XS_MV_ABL != 2 && IQ4XS_MV_CB == 5
#define IQ4XS_ACC(a, yv, w) a += dot((yv), iq4nl_cb4_pack((uint)(w)))
#else
#define IQ4XS_ACC(a, yv, w)                                    \
    do {                                                       \
        a += (yv).s0 * IQ4XS_CB(((uint)(w)      ) & 0xFu);     \
        a += (yv).s1 * IQ4XS_CB(((uint)(w) >>  4) & 0xFu);     \
        a += (yv).s2 * IQ4XS_CB(((uint)(w) >>  8) & 0xFu);     \
        a += (yv).s3 * IQ4XS_CB(((uint)(w) >> 12) & 0xFu);     \
    } while (0)
#endif

kernel void kernel_mul_mv_iq4_xs_f32_flat(
        global const ushort * src0_q,
        global const half   * src0_d,
        global const ushort * src0_sh,
        global const uint   * src0_sl,
        global const float  * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,      // K
        int ne01,      // M, the number of output rows
        int ne10,      // activation row stride, == K
        int ne0        // dst row stride
) {
    src1 = (global const float *)((global const char *)src1 + offset1);
    dst  = (global float       *)((global char       *)dst  + offsetd);

    const uint m   = (uint)ne01;
    const uint K   = (uint)ne00;
    const uint nsb = K / QK_K;                  // super blocks along K

    const uint lid = get_local_id(0);           // lane
    const uint sg  = get_local_id(1);           // K-split slice
    const uint col = get_group_id(1);           // token

    global const float * y = src1 + (ulong)col * (uint)ne10;

    IQ4XS_CB_DECL

#if IQ4XS_MV_R4
    const uint mq  = m >> 2;                    // rows per plane row, as uint2s
    const uint j   = get_group_id(0) * 64u + lid;   // row quad index
    const uint row = j << 2;

    float4 sum4 = (float4)(0.f);

    if (j < mq) {
        global const uint * qu = (global const uint *)src0_q;
        global const uint * su = (global const uint *)src0_sh;

        for (uint ib = sg; ib < nsb; ib += IQ4XS_MV_NSG) {
            const uint  sbase = j + ib * mq;
            const uint4 slv   = vload4(sbase, src0_sl);
            const uint2 shp   = vload2(sbase, su);
            const half4 dh    = vload4(sbase, src0_d);

            float4 acc = (float4)(0.f);
            for (uint sb = 0; sb < 8u; ++sb) {
                const uint lo = (8u * (sb >> 1)) + (4u * (sb & 1u));
                const float4 ls = (float4)(
                    (float)((int)((slv.s0 >> lo) & 0xFu) | (int)((((shp.s0      ) >> (2u * sb)) & 3u) << 4)) - 32.f,
                    (float)((int)((slv.s1 >> lo) & 0xFu) | (int)((((shp.s0 >> 16) >> (2u * sb)) & 3u) << 4)) - 32.f,
                    (float)((int)((slv.s2 >> lo) & 0xFu) | (int)((((shp.s1      ) >> (2u * sb)) & 3u) << 4)) - 32.f,
                    (float)((int)((slv.s3 >> lo) & 0xFu) | (int)((((shp.s1 >> 16) >> (2u * sb)) & 3u) << 4)) - 32.f);

                const uint grp = ib * 64u + sb * 8u;

                float4 a = (float4)(0.f);
                for (uint u = 0; u < 8u; ++u) {
                    const uint2  w  = vload2(j + (grp + u) * mq, qu);   // four rows, one load
                    const float4 yv = IQ4XS_YV(grp + u, y);
                    IQ4XS_ACC(a.s0, yv, (w.s0      ) & 0xFFFFu);
                    IQ4XS_ACC(a.s1, yv, (w.s0 >> 16)         );
                    IQ4XS_ACC(a.s2, yv, (w.s1      ) & 0xFFFFu);
                    IQ4XS_ACC(a.s3, yv, (w.s1 >> 16)         );
                }
                acc += ls * a;
            }
            sum4 += convert_float4(dh) * acc;
        }
    }
#elif IQ4XS_MV_R2
    const uint mh  = m >> 1;                    // rows per plane row, as uints
    const uint j   = get_group_id(0) * 64u + lid;   // row pair index
    const uint row = j << 1;

    float sumf  = 0.f;
    float sumf1 = 0.f;

    if (j < mh) {
        global const uint * qu = (global const uint *)src0_q;
        global const uint * su = (global const uint *)src0_sh;

        for (uint ib = sg; ib < nsb; ib += IQ4XS_MV_NSG) {
            const uint  sbase = j + ib * mh;
            const uint2 slv   = vload2(sbase, src0_sl);
            const uint  shp   = su[sbase];
            const half2 dh    = vload2(sbase, src0_d);
            const float d0    = (float)dh.s0;
            const float d1    = (float)dh.s1;

            float acc0 = 0.f, acc1 = 0.f;
            for (uint sb = 0; sb < 8u; ++sb) {
                const int ls0 = (int)(((slv.s0 >> (8u * (sb >> 1))) >> (4u * (sb & 1u))) & 0xFu)
                              | (int)((((shp      ) >> (2u * sb)) & 3u) << 4);
                const int ls1 = (int)(((slv.s1 >> (8u * (sb >> 1))) >> (4u * (sb & 1u))) & 0xFu)
                              | (int)((((shp >> 16) >> (2u * sb)) & 3u) << 4);

                const uint grp = ib * 64u + sb * 8u;
                const uint qb  = j + grp * mh;

                float a0 = 0.f, a1 = 0.f;
                for (uint u = 0; u < 8u; ++u) {
                    const uint   w  = qu[qb + u * mh];   // row pair, one load
                    const ushort w0 = (ushort)(w & 0xFFFFu);
                    const ushort w1 = (ushort)(w >> 16);
                    const float4 yv = IQ4XS_YV(grp + u, y);
                    IQ4XS_ACC(a0, yv, w0);
                    IQ4XS_ACC(a1, yv, w1);
                }
                acc0 += (float)(ls0 - 32) * a0;
                acc1 += (float)(ls1 - 32) * a1;
            }
            sumf  += d0 * acc0;
            sumf1 += d1 * acc1;
        }
    }
#else
    const uint row = get_group_id(0) * 64u + lid;

    float sumf = 0.f;

    if (row < m) {
        for (uint ib = sg; ib < nsb; ib += IQ4XS_MV_NSG) {
            const uint  base = row + ib * m;
            const uint  slv  = src0_sl[base];
            const uint  shv  = (uint)src0_sh[base];
            const float d    = (float)src0_d[base];

            float acc = 0.f;
            for (uint sb = 0; sb < 8u; ++sb) {
                const int ls = (int)(((slv >> (8u * (sb >> 1))) >> (4u * (sb & 1u))) & 0xFu)
                             | (int)(((shv >> (2u * sb)) & 3u) << 4);

                const uint grp = ib * 64u + sb * 8u;
                const uint qb  = row + grp * m;

                float a = 0.f;
                for (uint u = 0; u < 8u; ++u) {
                    const ushort w  = src0_q[qb + u * m];
                    const float4 yv = IQ4XS_YV(grp + u, y);
                    IQ4XS_ACC(a, yv, w);
                }
                acc += (float)(ls - 32) * a;
            }
            sumf += d * acc;
        }
    }
#endif

#if IQ4XS_MV_NSG > 1
#if IQ4XS_MV_R4
    __local float4 part4[IQ4XS_MV_NSG][64];
    part4[sg][lid] = sum4;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sg != 0) {
        return;
    }
    for (uint s = 1; s < IQ4XS_MV_NSG; ++s) {
        sum4 += part4[s][lid];
    }
#elif IQ4XS_MV_R2
    __local float2 part[IQ4XS_MV_NSG][64];
    part[sg][lid] = (float2)(sumf, sumf1);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sg != 0) {
        return;
    }
    for (uint s = 1; s < IQ4XS_MV_NSG; ++s) {
        const float2 p = part[s][lid];
        sumf  += p.s0;
        sumf1 += p.s1;
    }
#else
    __local float part[IQ4XS_MV_NSG][64];
    part[sg][lid] = sumf;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sg != 0) {
        return;
    }
    for (uint s = 1; s < IQ4XS_MV_NSG; ++s) {
        sumf += part[s][lid];
    }
#endif
#endif

#if IQ4XS_MV_R4
    if (j < mq) {
        vstore4(sum4, 0, dst + (ulong)col * (uint)ne0 + row);
    }
#elif IQ4XS_MV_R2
    if (j < mh) {
        vstore2((float2)(sumf, sumf1), 0, dst + (ulong)col * (uint)ne0 + row);
    }
#else
    if (row < m) {
        dst[(ulong)col * (uint)ne0 + row] = sumf;
    }
#endif
}


// Weights-as-texture twin of the kernel above (`_wimg`). Identical arithmetic; the
// quant plane is read through an image1d_buffer instead of a global pointer.
//
// Why this one is a good candidate: shape-matched IN THE SAME 27B decode run, this
// kernel costs 0.7279 ms/call at out=17408 against 0.4226 for the tuned
// gemv_noshuffle_q4_k_f32 -- 1.72x, and q4_K moves MORE bytes per weight. That
// tuned kernel differs in exactly two ways, and this is the first: its weights come
// from an image1d_buffer. (The second is loading activations on four lanes and
// sub_group_broadcast-ing them.)
//
// The standing rule is that a texture pays only when a lane takes a WHOLE texel.
// Under IQ4XS_MV_R2 that is exactly what happens -- a lane owns a row pair and
// reads their two ushorts as one uint -- so this is the favourable case, more so
// than the prefill GEMM where a lane takes half a texel and it still won 8.6%.
//
// Opt-in with GGML_OPENCL_IQ4XS_MV_WIMG until measured.
kernel void kernel_mul_mv_iq4_xs_f32_flat_wimg(
        __read_only image1d_buffer_t src0_q_img,
        global const half   * src0_d,
        global const ushort * src0_sh,
        global const uint   * src0_sl,
        global const float  * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,      // K
        int ne01,      // M, the number of output rows
        int ne10,      // activation row stride, == K
        int ne0        // dst row stride
) {
    src1 = (global const float *)((global const char *)src1 + offset1);
    dst  = (global float       *)((global char       *)dst  + offsetd);

    const uint m   = (uint)ne01;
    const uint K   = (uint)ne00;
    const uint nsb = K / QK_K;                  // super blocks along K

    const uint lid = get_local_id(0);           // lane
    const uint sg  = get_local_id(1);           // K-split slice
    const uint col = get_group_id(1);           // token

    global const float * y = src1 + (ulong)col * (uint)ne10;

    IQ4XS_CB_DECL

#if IQ4XS_MV_R4
    const uint mq  = m >> 2;                    // rows per plane row, as uint2s
    const uint j   = get_group_id(0) * 64u + lid;   // row quad index
    const uint row = j << 2;

    float4 sum4 = (float4)(0.f);

    if (j < mq) {
        global const uint * su = (global const uint *)src0_sh;

        for (uint ib = sg; ib < nsb; ib += IQ4XS_MV_NSG) {
            const uint  sbase = j + ib * mq;
            const uint4 slv   = vload4(sbase, src0_sl);
            const uint2 shp   = vload2(sbase, su);
            const half4 dh    = vload4(sbase, src0_d);

            float4 acc = (float4)(0.f);
            for (uint sb = 0; sb < 8u; ++sb) {
                const uint lo = (8u * (sb >> 1)) + (4u * (sb & 1u));
                const float4 ls = (float4)(
                    (float)((int)((slv.s0 >> lo) & 0xFu) | (int)((((shp.s0      ) >> (2u * sb)) & 3u) << 4)) - 32.f,
                    (float)((int)((slv.s1 >> lo) & 0xFu) | (int)((((shp.s0 >> 16) >> (2u * sb)) & 3u) << 4)) - 32.f,
                    (float)((int)((slv.s2 >> lo) & 0xFu) | (int)((((shp.s1      ) >> (2u * sb)) & 3u) << 4)) - 32.f,
                    (float)((int)((slv.s3 >> lo) & 0xFu) | (int)((((shp.s1 >> 16) >> (2u * sb)) & 3u) << 4)) - 32.f);

                const uint grp = ib * 64u + sb * 8u;

                float4 a = (float4)(0.f);
                for (uint u = 0; u < 8u; ++u) {
                    // four rows are two whole texels
                    const uint   wi = 2u * (j + (grp + u) * mq);
                    const uint2  w  = (uint2)(read_imageui(src0_q_img, (int)(wi     )).x,
                                              read_imageui(src0_q_img, (int)(wi + 1)).x);
                    const float4 yv = IQ4XS_YV(grp + u, y);
                    IQ4XS_ACC(a.s0, yv, (w.s0      ) & 0xFFFFu);
                    IQ4XS_ACC(a.s1, yv, (w.s0 >> 16)         );
                    IQ4XS_ACC(a.s2, yv, (w.s1      ) & 0xFFFFu);
                    IQ4XS_ACC(a.s3, yv, (w.s1 >> 16)         );
                }
                acc += ls * a;
            }
            sum4 += convert_float4(dh) * acc;
        }
    }
#elif IQ4XS_MV_R2
    const uint mh  = m >> 1;                    // rows per plane row, as uints
    const uint j   = get_group_id(0) * 64u + lid;   // row pair index
    const uint row = j << 1;

    float sumf  = 0.f;
    float sumf1 = 0.f;

    if (j < mh) {
        global const uint * su = (global const uint *)src0_sh;

        for (uint ib = sg; ib < nsb; ib += IQ4XS_MV_NSG) {
            const uint  sbase = j + ib * mh;
            const uint2 slv   = vload2(sbase, src0_sl);
            const uint  shp   = su[sbase];
            const half2 dh    = vload2(sbase, src0_d);
            const float d0    = (float)dh.s0;
            const float d1    = (float)dh.s1;

            float acc0 = 0.f, acc1 = 0.f;
            for (uint sb = 0; sb < 8u; ++sb) {
                const int ls0 = (int)(((slv.s0 >> (8u * (sb >> 1))) >> (4u * (sb & 1u))) & 0xFu)
                              | (int)((((shp      ) >> (2u * sb)) & 3u) << 4);
                const int ls1 = (int)(((slv.s1 >> (8u * (sb >> 1))) >> (4u * (sb & 1u))) & 0xFu)
                              | (int)((((shp >> 16) >> (2u * sb)) & 3u) << 4);

                const uint grp = ib * 64u + sb * 8u;
                const uint qb  = j + grp * mh;

                float a0 = 0.f, a1 = 0.f;
                for (uint u = 0; u < 8u; ++u) {
                    // a row PAIR is exactly one texel here, which is the case the
                    // weights-as-texture rule says should pay
                    const uint   w  = read_imageui(src0_q_img, (int)(qb + u * mh)).x;
                    const ushort w0 = (ushort)(w & 0xFFFFu);
                    const ushort w1 = (ushort)(w >> 16);
                    const float4 yv = IQ4XS_YV(grp + u, y);
                    IQ4XS_ACC(a0, yv, w0);
                    IQ4XS_ACC(a1, yv, w1);
                }
                acc0 += (float)(ls0 - 32) * a0;
                acc1 += (float)(ls1 - 32) * a1;
            }
            sumf  += d0 * acc0;
            sumf1 += d1 * acc1;
        }
    }
#else
    const uint row = get_group_id(0) * 64u + lid;

    float sumf = 0.f;

    if (row < m) {
        for (uint ib = sg; ib < nsb; ib += IQ4XS_MV_NSG) {
            const uint  base = row + ib * m;
            const uint  slv  = src0_sl[base];
            const uint  shv  = (uint)src0_sh[base];
            const float d    = (float)src0_d[base];

            float acc = 0.f;
            for (uint sb = 0; sb < 8u; ++sb) {
                const int ls = (int)(((slv >> (8u * (sb >> 1))) >> (4u * (sb & 1u))) & 0xFu)
                             | (int)(((shv >> (2u * sb)) & 3u) << 4);

                const uint grp = ib * 64u + sb * 8u;
                const uint qb  = row + grp * m;

                float a = 0.f;
                for (uint u = 0; u < 8u; ++u) {
                    const uint   we = qb + u * m;
                    const ushort w  = (ushort)((read_imageui(src0_q_img, (int)(we >> 1)).x
                                                >> ((we & 1u) * 16u)) & 0xFFFFu);
                    const float4 yv = IQ4XS_YV(grp + u, y);
                    IQ4XS_ACC(a, yv, w);
                }
                acc += (float)(ls - 32) * a;
            }
            sumf += d * acc;
        }
    }
#endif

#if IQ4XS_MV_NSG > 1
#if IQ4XS_MV_R4
    __local float4 part4[IQ4XS_MV_NSG][64];
    part4[sg][lid] = sum4;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sg != 0) {
        return;
    }
    for (uint s = 1; s < IQ4XS_MV_NSG; ++s) {
        sum4 += part4[s][lid];
    }
#elif IQ4XS_MV_R2
    __local float2 part[IQ4XS_MV_NSG][64];
    part[sg][lid] = (float2)(sumf, sumf1);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sg != 0) {
        return;
    }
    for (uint s = 1; s < IQ4XS_MV_NSG; ++s) {
        const float2 p = part[s][lid];
        sumf  += p.s0;
        sumf1 += p.s1;
    }
#else
    __local float part[IQ4XS_MV_NSG][64];
    part[sg][lid] = sumf;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sg != 0) {
        return;
    }
    for (uint s = 1; s < IQ4XS_MV_NSG; ++s) {
        sumf += part[s][lid];
    }
#endif
#endif

#if IQ4XS_MV_R4
    if (j < mq) {
        vstore4(sum4, 0, dst + (ulong)col * (uint)ne0 + row);
    }
#elif IQ4XS_MV_R2
    if (j < mh) {
        vstore2((float2)(sumf, sumf1), 0, dst + (ulong)col * (uint)ne0 + row);
    }
#else
    if (row < m) {
        dst[(ulong)col * (uint)ne0 + row] = sumf;
    }
#endif
}


// ---------------------------------------------------------------------------
// Fused ffn_gate + ffn_up + GLU, and a workgroup-level K split, for IQ4_XS.
//
// IQ4_XS had NEITHER, which is why it is worth doing: on a Llama-3.2-3B roster it
// is the fastest decode after Q4_0 (tg64 40.0 against 47.7) and Q4_0's advantage
// is largely these two things. Unlike the IQ1/IQ2/IQ3 family this is a LINEAR
// quant, not a codebook one, so the q4_0 precedent (+10.2% from the fusion) is
// the right prior rather than the weaker codebook-type results.
//
// Both kernels are R2 only -- one lane owns a row pair -- which is the default
// (IQ4XS_MV_R4 measured +1.5% on a 3B and -3.0% on a 27B and stays off). The host
// declines both when R4 is on.
// ---------------------------------------------------------------------------

// Fifth copy of the shared GLU epilogue: each .cl is its own program and cannot
// include the others. Op numbering and expressions are identical to
// q40_glu_apply, iq2s_glu_apply, iq1s_glu_apply and iq1m_glu_apply on purpose --
// if one of these is ever changed, change all of them.
#define IQ4XS_GLU_GEGLU_COEF_A   0.044715f
#define IQ4XS_GLU_SQRT_2_OVER_PI 0.79788456080286535587989211986876f
#define IQ4XS_GLU_SQRT_2_INV     0.70710678118654752440084436210484f
#define IQ4XS_GLU_QUICK_COEF    -1.702f
inline float iq4xs_glu_apply(int glu_op, float g, float u) {
    float act;
    if (glu_op == 1) {        // GEGLU (tanh-approx gelu)
        act = 0.5f*g*(1.0f + tanh(IQ4XS_GLU_SQRT_2_OVER_PI*g*(1.0f + IQ4XS_GLU_GEGLU_COEF_A*g*g)));
    } else if (glu_op == 2) { // SWIGLU (silu)
        act = g / (1.0f + exp(-g));
    } else if (glu_op == 0) { // REGLU
        return g*u*(g > 0.0f);
    } else if (glu_op == 4) { // GEGLU_ERF
        act = 0.5f*g*(1.0f + erf(g*IQ4XS_GLU_SQRT_2_INV));
    } else {                  // GEGLU_QUICK
        act = g*(1.0f/(1.0f + exp(IQ4XS_GLU_QUICK_COEF*g)));
    }
    return act*u;
}

// The two weight streams are interleaved so each activation float4 is loaded ONCE
// and feeds gate and up in the same iteration -- that is the point, not just the
// saved dispatch. Costs four accumulators instead of two.
kernel void kernel_mul_mv_iq4_xs_f32_flat_glu(
        global const ushort * g_q,
        global const half   * g_d,
        global const ushort * g_sh,
        global const uint   * g_sl,
        global const ushort * u_q,
        global const half   * u_d,
        global const ushort * u_sh,
        global const uint   * u_sl,
        global const float  * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne10,
        int ne0,
        int glu_op
) {
    src1 = (global const float *)((global const char *)src1 + offset1);
    dst  = (global float       *)((global char       *)dst  + offsetd);

    const uint m   = (uint)ne01;
    const uint K   = (uint)ne00;
    const uint nsb = K / QK_K;

    const uint lid = get_local_id(0);
    const uint sg  = get_local_id(1);
    const uint col = get_group_id(1);

    global const float * y = src1 + (ulong)col * (uint)ne10;

    IQ4XS_CB_DECL

    const uint mh  = m >> 1;
    const uint j   = get_group_id(0) * 64u + lid;
    const uint row = j << 1;

    float gs0 = 0.f, gs1 = 0.f, us0 = 0.f, us1 = 0.f;

    if (j < mh) {
        global const uint * gqu = (global const uint *)g_q;
        global const uint * gsu = (global const uint *)g_sh;
        global const uint * uqu = (global const uint *)u_q;
        global const uint * usu = (global const uint *)u_sh;

        for (uint ib = sg; ib < nsb; ib += IQ4XS_MV_NSG) {
            const uint  sbase = j + ib * mh;

            const uint2 gslv = vload2(sbase, g_sl);
            const uint  gshp = gsu[sbase];
            const half2 gdh  = vload2(sbase, g_d);
            const float gd0  = (float)gdh.s0;
            const float gd1  = (float)gdh.s1;

            const uint2 uslv = vload2(sbase, u_sl);
            const uint  ushp = usu[sbase];
            const half2 udh  = vload2(sbase, u_d);
            const float ud0  = (float)udh.s0;
            const float ud1  = (float)udh.s1;

            float gacc0 = 0.f, gacc1 = 0.f, uacc0 = 0.f, uacc1 = 0.f;
            for (uint sb = 0; sb < 8u; ++sb) {
                const int gls0 = (int)(((gslv.s0 >> (8u * (sb >> 1))) >> (4u * (sb & 1u))) & 0xFu)
                               | (int)((((gshp      ) >> (2u * sb)) & 3u) << 4);
                const int gls1 = (int)(((gslv.s1 >> (8u * (sb >> 1))) >> (4u * (sb & 1u))) & 0xFu)
                               | (int)((((gshp >> 16) >> (2u * sb)) & 3u) << 4);
                const int uls0 = (int)(((uslv.s0 >> (8u * (sb >> 1))) >> (4u * (sb & 1u))) & 0xFu)
                               | (int)((((ushp      ) >> (2u * sb)) & 3u) << 4);
                const int uls1 = (int)(((uslv.s1 >> (8u * (sb >> 1))) >> (4u * (sb & 1u))) & 0xFu)
                               | (int)((((ushp >> 16) >> (2u * sb)) & 3u) << 4);

                const uint grp = ib * 64u + sb * 8u;
                const uint qb  = j + grp * mh;

                float ga0 = 0.f, ga1 = 0.f, ua0 = 0.f, ua1 = 0.f;
                for (uint u = 0; u < 8u; ++u) {
                    // activation float4 read ONCE for both streams and both rows
                    const float4 yv = IQ4XS_YV(grp + u, y);

                    const uint   gw  = gqu[qb + u * mh];
                    const ushort gw0 = (ushort)(gw & 0xFFFFu);
                    const ushort gw1 = (ushort)(gw >> 16);
                    IQ4XS_ACC(ga0, yv, gw0);
                    IQ4XS_ACC(ga1, yv, gw1);

                    const uint   uw  = uqu[qb + u * mh];
                    const ushort uw0 = (ushort)(uw & 0xFFFFu);
                    const ushort uw1 = (ushort)(uw >> 16);
                    IQ4XS_ACC(ua0, yv, uw0);
                    IQ4XS_ACC(ua1, yv, uw1);
                }
                gacc0 += (float)(gls0 - 32) * ga0;
                gacc1 += (float)(gls1 - 32) * ga1;
                uacc0 += (float)(uls0 - 32) * ua0;
                uacc1 += (float)(uls1 - 32) * ua1;
            }
            gs0 += gd0 * gacc0;
            gs1 += gd1 * gacc1;
            us0 += ud0 * uacc0;
            us1 += ud1 * uacc1;
        }
    }

#if IQ4XS_MV_NSG > 1
    __local float4 gpart[IQ4XS_MV_NSG][64];
    gpart[sg][lid] = (float4)(gs0, gs1, us0, us1);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sg != 0) {
        return;
    }
    for (uint s = 1; s < IQ4XS_MV_NSG; ++s) {
        const float4 p = gpart[s][lid];
        gs0 += p.s0; gs1 += p.s1; us0 += p.s2; us1 += p.s3;
    }
#endif

    if (j < mh) {
        global float * o = dst + (ulong)col * (uint)ne0 + row;
        o[0] = iq4xs_glu_apply(glu_op, gs0, us0);
        o[1] = iq4xs_glu_apply(glu_op, gs1, us1);
    }
}

// Slice ks accumulates its own range of super-blocks and writes
// partial[ks*M + row]; the existing generic kernel_gemv_splitk_reduce_f32 sums
// them. ksplit comes from the same occupancy heuristic the IQ1/IQ2 splits use, so
// a shape that already fills the device stays at 1 and never reaches this kernel.
kernel void kernel_mul_mv_iq4_xs_f32_flat_splitk(
        global const ushort * src0_q,
        global const half   * src0_d,
        global const ushort * src0_sh,
        global const uint   * src0_sl,
        global const float  * src1,
        ulong offset1,
        global float * partial,
        int ne00,
        int ne01,
        int ne10
) {
    src1 = (global const float *)((global const char *)src1 + offset1);

    const uint m   = (uint)ne01;
    const uint K   = (uint)ne00;
    const uint nsb = K / QK_K;

    const uint lid = get_local_id(0);
    const uint sg  = get_local_id(1);
    const uint col = get_group_id(1);
    const uint ks  = get_group_id(2);
    const uint nks = get_num_groups(2);

    // this slice's super-block range; the subgroups stride within it
    const uint ib0 = (nsb * ks)        / nks;
    const uint ib1 = (nsb * (ks + 1u)) / nks;

    global const float * y = src1 + (ulong)col * (uint)ne10;

    IQ4XS_CB_DECL

    const uint mh  = m >> 1;
    const uint j   = get_group_id(0) * 64u + lid;
    const uint row = j << 1;

    float sumf = 0.f, sumf1 = 0.f;

    if (j < mh) {
        global const uint * qu = (global const uint *)src0_q;
        global const uint * su = (global const uint *)src0_sh;

        for (uint ib = ib0 + sg; ib < ib1; ib += IQ4XS_MV_NSG) {
            const uint  sbase = j + ib * mh;
            const uint2 slv   = vload2(sbase, src0_sl);
            const uint  shp   = su[sbase];
            const half2 dh    = vload2(sbase, src0_d);
            const float d0    = (float)dh.s0;
            const float d1    = (float)dh.s1;

            float acc0 = 0.f, acc1 = 0.f;
            for (uint sb = 0; sb < 8u; ++sb) {
                const int ls0 = (int)(((slv.s0 >> (8u * (sb >> 1))) >> (4u * (sb & 1u))) & 0xFu)
                              | (int)((((shp      ) >> (2u * sb)) & 3u) << 4);
                const int ls1 = (int)(((slv.s1 >> (8u * (sb >> 1))) >> (4u * (sb & 1u))) & 0xFu)
                              | (int)((((shp >> 16) >> (2u * sb)) & 3u) << 4);

                const uint grp = ib * 64u + sb * 8u;
                const uint qb  = j + grp * mh;

                float a0 = 0.f, a1 = 0.f;
                for (uint u = 0; u < 8u; ++u) {
                    const uint   w  = qu[qb + u * mh];
                    const ushort w0 = (ushort)(w & 0xFFFFu);
                    const ushort w1 = (ushort)(w >> 16);
                    const float4 yv = IQ4XS_YV(grp + u, y);
                    IQ4XS_ACC(a0, yv, w0);
                    IQ4XS_ACC(a1, yv, w1);
                }
                acc0 += (float)(ls0 - 32) * a0;
                acc1 += (float)(ls1 - 32) * a1;
            }
            sumf  += d0 * acc0;
            sumf1 += d1 * acc1;
        }
    }

#if IQ4XS_MV_NSG > 1
    __local float2 skpart[IQ4XS_MV_NSG][64];
    skpart[sg][lid] = (float2)(sumf, sumf1);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sg != 0) {
        return;
    }
    for (uint s = 1; s < IQ4XS_MV_NSG; ++s) {
        const float2 p = skpart[s][lid];
        sumf  += p.s0;
        sumf1 += p.s1;
    }
#endif

    if (j < mh) {
        // [ksplit][M], the layout kernel_gemv_splitk_reduce_f32 expects
        global float * o = partial + (ulong)ks * m + row;
        o[0] = sumf;
        o[1] = sumf1;
    }
}

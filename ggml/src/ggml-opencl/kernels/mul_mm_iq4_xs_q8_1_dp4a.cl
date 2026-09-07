#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#ifdef cl_khr_integer_dot_product
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#endif

// Dense IQ4_XS prefill GEMM, dp4a (int8) inner loop.
//
// IQ4_XS uses the same non-linear 4-bit codebook as IQ4_NL; it only adds a
// 256-element super-block with eight 6-bit sub-scales. So the arithmetic is the
// IQ4_NL dp4a case with the scale rebuilt per 32-block:
//   ls  = scales_l[sb/2] nibble | ((scales_h >> 2*sb) & 3) << 4
//   d_w = d * (ls - 32)
//   sum w*a = d_w * a_d * dp4a(kvalues[nibble], a_i8)
//
// Unlike kernel_gemm_noshuffle_iq4_nl_q8_1_dp4a this reads the weights in their
// original AoS block layout, so it needs no SoA upload path. The IQ4_XS SoA form
// would also be LARGER than the source (144 vs 136 bytes per 256 weights, since
// flat per-32 scales cost more than the packed 6-bit ones), so it could not
// reuse the in-place subbuffer trick the other noshuffle kernels rely on.
// Reading AoS directly is what the CUDA MMQ loader does.
//
// The point: the weights never expand to f32. mul_mm_iq4_xs_f32_l4_lm stages
// dequantized f32 tiles, 16 KB of local memory per workgroup, and is occupancy
// bound; this stages only the int8 activations, about 1.1 KB.
//
// Activations are pre-quantized to q8_1 by kernel_quant_a_q8_1, as in the other
// dense dp4a GEMMs. Large-batch (prefill) only; K must be a multiple of 256.

#define QK_K 256

typedef struct {
    half   d;
    ushort scales_h;
    uchar  scales_l[QK_K/64];
    uchar  qs[QK_K/2];
} block_iq4_xs;

// TILESIZE_N is the token tile: it fixes the accumulator count and the LDS
// staging width, so it is compile-time, and the right value is PER DEVICE -- the
// X2-tuned 32 over-occupies LDS on an X1-85. Safe to vary here: the activation
// tile is staged with a strided `for (idx = lid; idx < TILESIZE_N*8; idx += 64)`
// loop, correct at any tile. The dispatch must pass the SAME value; see
// ggml_cl_lowbit_dp4a_ts and the note at its d_global.
#ifndef TILESIZE_N
#define TILESIZE_N 32
#endif

// IQ4_NL codebook as signed int8, packed 4 codes per uint. A divergent nibble
// lookup must read a __constant *uint* array and shift; a __constant byte array
// serializes on Adreno (about 4.6x) -- see gemm_noshuffle_iq4_nl_q8_1_dp4a.cl.
__constant uint kvalues_iq4nl_i8x4[4] = {
    0xBFAD9881u, 0xF6EADDCFu, 0x26190D01u, 0x71594535u
};

inline uint iq4nl_code(uint n) {
    return (kvalues_iq4nl_i8x4[n >> 2] >> ((n & 3u) * 8u)) & 0xFFu;
}

// A 32-block stores elements 0..15 as the LOW nibbles of qs[0..15] and elements
// 16..31 as the HIGH nibbles of those same bytes.
inline uint iq4_pack_lo(uchar4 b) {
    return  iq4nl_code((uint)(b.s0 & 0xF))
         | (iq4nl_code((uint)(b.s1 & 0xF)) <<  8)
         | (iq4nl_code((uint)(b.s2 & 0xF)) << 16)
         | (iq4nl_code((uint)(b.s3 & 0xF)) << 24);
}
inline uint iq4_pack_hi(uchar4 b) {
    return  iq4nl_code((uint)(b.s0 >> 4))
         | (iq4nl_code((uint)(b.s1 >> 4)) <<  8)
         | (iq4nl_code((uint)(b.s2 >> 4)) << 16)
         | (iq4nl_code((uint)(b.s3 >> 4)) << 24);
}

inline int dot8_q8a(uint8 qw, __local const uint * a) {
    int r = 0;
    r = dot_acc_sat_4x8packed_ss_int(qw.s0, a[0], r);
    r = dot_acc_sat_4x8packed_ss_int(qw.s1, a[1], r);
    r = dot_acc_sat_4x8packed_ss_int(qw.s2, a[2], r);
    r = dot_acc_sat_4x8packed_ss_int(qw.s3, a[3], r);
    r = dot_acc_sat_4x8packed_ss_int(qw.s4, a[4], r);
    r = dot_acc_sat_4x8packed_ss_int(qw.s5, a[5], r);
    r = dot_acc_sat_4x8packed_ss_int(qw.s6, a[6], r);
    r = dot_acc_sat_4x8packed_ss_int(qw.s7, a[7], r);
    return r;
}

__attribute__((qcom_wave_pair_mode(1)))
kernel void kernel_mul_mm_iq4_xs_q8_1_dp4a(
        __global const char  * src0,
        ulong  offset0,
        __global const uint  * src1_qa,
        __global const half  * src1_da,
        __global       float * dst,
        ulong  offsetd,
        int    m,
        int    n_no_padding,
        int    k,
        ulong  nb01
) {
    src0 = src0 + offset0;
    dst  = (global float *)((global char *)dst + offsetd);

    const uint lid = get_local_id(0);
    const uint block_id_m = get_global_id(1);
    const uint block_id_n = get_global_id(2);

    const uint row      = block_id_m * 64 + lid;
    const uint col_base = block_id_n * TILESIZE_N;
    const bool row_valid = row < (uint)m;
    const uint rrow     = row_valid ? row : 0;

    const uint k_u = (uint)k >> 2;
    const uint k_b = (uint)k >> 5;

    __local uint sh_qa[TILESIZE_N][8];
    __local half sh_d[TILESIZE_N];

    __global const block_iq4_xs * xrow =
        (__global const block_iq4_xs *)(src0 + (ulong)rrow * nb01);

#define NGROUPS (TILESIZE_N / 4)
    float4 acc[NGROUPS];
    #pragma unroll
    for (int g = 0; g < NGROUPS; ++g) acc[g] = (float4)(0.0f);

    for (uint step = 0; step < (uint)k; step += 32) {
        const uint sub = step >> 5;
        const uint ib  = step >> 8;
        const uint sb  = sub & 7u;

        __global const block_iq4_xs * xb = xrow + ib;

        const int ls = (int)((((uint)xb->scales_l[sb >> 1]) >> (4u * (sb & 1u))) & 0xFu)
                     | (int)(((((uint)xb->scales_h) >> (2u * sb)) & 3u) << 4);
        const float d_w = (float)xb->d * (float)(ls - 32);

        __global const uchar * qs = xb->qs + 16u * sb;
        uint8 qw;
        qw.s0 = iq4_pack_lo(vload4(0, qs));
        qw.s1 = iq4_pack_lo(vload4(1, qs));
        qw.s2 = iq4_pack_lo(vload4(2, qs));
        qw.s3 = iq4_pack_lo(vload4(3, qs));
        qw.s4 = iq4_pack_hi(vload4(0, qs));
        qw.s5 = iq4_pack_hi(vload4(1, qs));
        qw.s6 = iq4_pack_hi(vload4(2, qs));
        qw.s7 = iq4_pack_hi(vload4(3, qs));

        for (uint idx = lid; idx < TILESIZE_N * 8; idx += 64) {
            const uint t = idx >> 3;
            const uint u = idx & 7;
            const uint c = col_base + t;
            sh_qa[t][u] = (c < (uint)n_no_padding) ? src1_qa[c * k_u + (step >> 2) + u] : 0u;
        }
        if (lid < TILESIZE_N) {
            const uint c = col_base + lid;
            sh_d[lid] = (c < (uint)n_no_padding) ? src1_da[c * k_b + sub] : (half)0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

#define LD4(arr, b) ((float4)((float)arr[(b)+0], (float)arr[(b)+1], (float)arr[(b)+2], (float)arr[(b)+3]))
        #pragma unroll
        for (int g = 0; g < NGROUPS; ++g) {
            const int b = g * 4;
            float4 rf;
            rf.s0 = (float)dot8_q8a(qw, sh_qa[b+0]);  rf.s1 = (float)dot8_q8a(qw, sh_qa[b+1]);
            rf.s2 = (float)dot8_q8a(qw, sh_qa[b+2]);  rf.s3 = (float)dot8_q8a(qw, sh_qa[b+3]);
            acc[g] += d_w * LD4(sh_d, b) * rf;
        }
#undef LD4
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (!row_valid) {
        return;
    }

    #pragma unroll
    for (int g = 0; g < NGROUPS; ++g) {
        const uint b = (uint)(g * 4);
        const float4 a = acc[g];
        const uint c0 = col_base + b;
        if (c0 + 0 < (uint)n_no_padding) dst[(c0 + 0) * (uint)m + row] = a.s0;
        if (c0 + 1 < (uint)n_no_padding) dst[(c0 + 1) * (uint)m + row] = a.s1;
        if (c0 + 2 < (uint)n_no_padding) dst[(c0 + 2) * (uint)m + row] = a.s2;
        if (c0 + 3 < (uint)n_no_padding) dst[(c0 + 3) * (uint)m + row] = a.s3;
    }
#undef NGROUPS
}

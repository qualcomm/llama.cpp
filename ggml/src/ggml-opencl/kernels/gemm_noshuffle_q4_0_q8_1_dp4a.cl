#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#ifdef cl_khr_integer_dot_product
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#endif

// Dense q4_0 prefill GEMM, dp4a (int8) inner loop.
//
// dp4a alternative to kernel_gemm_noshuffle_q4_0_f32 (the f16 half-dot GEMM for
// dense q4_0 matmuls -- attention / ffn projections in a full-Q4_0 model).
// Activations pre-quantized to q8_1 (kernel_quant_a_q8_1) from the original
// [N, K] token-major buffer; the per-32-block dot uses the qcom int8 dp4a.
//
// q4_0 weight = d * (q - 8), q in [0,15], one fp16 scale per 32-block per row.
// Mirrors kernel_gemm_noshuffle_iq4_nl_q8_1_dp4a (same feature-major nibble
// layout: src0_q[row + (k/4)*m], ushort = 4 nibbles, low nibble = lowest K) but
// uses the plain nibble (EXP4) instead of the IQ4_NL codebook, and adds the
// constant -8 zero-point via the q8_1 sum term:
//   Sum w*a = d_w * (a_d * dp4a(q, qa) - 8 * a_s),  a_s = a_d * Sum(qa)
// Mirrors vec_dot_q4_0_q8_1. Large-batch (prefill) only; ne1<=8 keeps the f16 path.

// Guarded so the host's -DTILESIZE_N=16 for the narrow variant actually takes effect.
// Without the guard the -D is silently overridden by this definition and the "narrow"
// program compiles at 32, i.e. it is not narrow at all. q4_K has always had the guard.
#ifndef TILESIZE_N
#define TILESIZE_N 32
#endif

// Expand the 4 nibbles in the low 16 bits of u into 4 bytes (value 0..15),
// packed for the int8 dp4a. The -8 zero-point is applied via the sum term.
#define EXP4(u)  ( ((uint)((u) & 0x000Fu))        | \
                  (((uint)((u) & 0x00F0u)) << 4)  | \
                  (((uint)((u) & 0x0F00u)) << 8)  | \
                  (((uint)((u) & 0xF000u)) << 12) )

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
kernel void kernel_gemm_noshuffle_q4_0_q8_1_dp4a(
        __global const ushort * src0_q,    // q4_0 nibbles (4/ushort, feature-major)
        __global const half   * src0_d,    // per-32-block scale, feature-major
        __global const uint   * src1_qa,   // q8_1 activations int8 (as uint, 4/elem) [N, K]
        __global const half   * src1_da,   // q8_1 per-block scale [N, K/32]
        __global const half   * src1_sa,   // q8_1 per-block sum*d  [N, K/32]
        __global       float  * dst,
        ulong  offsetd,
        int    m,                          // output features (rows)
        int    n_no_padding,               // tokens (cols)
        int    k                           // K (== ne00)
) {
    dst = (global float *)((global char *)dst + offsetd);

    const uint lid = get_local_id(0);          // 0..63 -> row within the M-tile
    const uint block_id_m = get_global_id(1);
    const uint block_id_n = get_global_id(2);

    const uint row      = block_id_m * 64 + lid;
    const uint col_base = block_id_n * TILESIZE_N;
    const bool row_valid = row < (uint)m;
    const uint rrow     = row_valid ? row : 0;  // clamp OOB rows; their writes are masked

    const uint k_u = (uint)k >> 2;   // K in uint (int8x4) units
    const uint k_b = (uint)k >> 5;   // blocks-of-32 along K

    __local uint sh_qa[TILESIZE_N][8];
    __local half sh_d[TILESIZE_N];
    __local half sh_s[TILESIZE_N];

#define NGROUPS (TILESIZE_N / 4)
    float4 acc[NGROUPS];
    #pragma unroll
    for (int g = 0; g < NGROUPS; ++g) acc[g] = (float4)(0.0f);

    for (uint step = 0; step < (uint)k; step += 32) {
        const uint sub = step >> 5;

        const float d_w = (float)src0_d[rrow + sub * (uint)m];

        // 8 weight uints (32 nibbles) for this row, this 32-block. Feature-major:
        // src0_q[row + (k/4 + u)*m], k/4 = step/4 (= step>>2). EXP4 -> dp4a int8.
        const uint qsbase = rrow + (step >> 2) * (uint)m;
        uint8 qw;
        qw.s0 = EXP4(src0_q[qsbase + 0 * m]);
        qw.s1 = EXP4(src0_q[qsbase + 1 * m]);
        qw.s2 = EXP4(src0_q[qsbase + 2 * m]);
        qw.s3 = EXP4(src0_q[qsbase + 3 * m]);
        qw.s4 = EXP4(src0_q[qsbase + 4 * m]);
        qw.s5 = EXP4(src0_q[qsbase + 5 * m]);
        qw.s6 = EXP4(src0_q[qsbase + 6 * m]);
        qw.s7 = EXP4(src0_q[qsbase + 7 * m]);

        // cooperatively stage the 32-token x 32-K int8 activations to LDS
        for (uint idx = lid; idx < TILESIZE_N * 8; idx += 64) {
            const uint t = idx >> 3;
            const uint u = idx & 7;
            const uint c = col_base + t;
            sh_qa[t][u] = (c < (uint)n_no_padding) ? src1_qa[c * k_u + (step >> 2) + u] : 0u;
        }
        if (lid < TILESIZE_N) {
            const uint c = col_base + lid;
            sh_d[lid] = (c < (uint)n_no_padding) ? src1_da[c * k_b + sub] : (half)0;
            sh_s[lid] = (c < (uint)n_no_padding) ? src1_sa[c * k_b + sub] : (half)0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

#define LD4(arr, b) ((float4)((float)arr[(b)+0], (float)arr[(b)+1], (float)arr[(b)+2], (float)arr[(b)+3]))
        #pragma unroll
        for (int g = 0; g < NGROUPS; ++g) {
            const int b = g * 4;
            float4 rf;
            rf.s0 = (float)dot8_q8a(qw, sh_qa[b+0]);  rf.s1 = (float)dot8_q8a(qw, sh_qa[b+1]);
            rf.s2 = (float)dot8_q8a(qw, sh_qa[b+2]);  rf.s3 = (float)dot8_q8a(qw, sh_qa[b+3]);
            // q4_0: w = d*(q-8) -> d_w * (a_d * dp4a(q,qa) - 8 * a_s)
            acc[g] += d_w * (LD4(sh_d, b) * rf - 8.0f * LD4(sh_s, b));
        }
#undef LD4
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (!row_valid) {
        return;
    }

    // dst is [token, feature] row-major (stride m): dst[col*m + row].
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

// Weights-as-texture variant of kernel_gemm_noshuffle_q4_0_q8_1_dp4a.
//
// Byte-identical math; the only change is that the q4_0 weight plane is read through
// an image1d_buffer instead of a plain global buffer. q4_K has had this since
// c06213299 and it is worth +5.8% in the narrow band there; q4_0 never had one, which
// is why a SMALLER q4_0 drafter measured SLOWER than the q4_K_M one.
//
// Measured on X2-90, muse-glimmer DFlash drafter (ne1=16, the block width DFlash
// decodes in one pass): q4_K_M moves its weights at 25.9 GB/s, q4_0 with dp4a forced
// on at 19.9, and q8_0 at 35.3 - so q4_0 was the slowest per byte despite the simplest
// dequant of the three. Acceptance is 71.4% against q4_K_M's 71.7%, i.e. the format
// costs nothing in draft quality and the whole gap is the read path.
//
// Bound as CL_R/CL_UNSIGNED_INT32 (one texel = 2 packed ushorts), the format the q4_K
// _wimg and _kimg kernels use. The host only selects this when m is even, so
// (rrow + (step>>2)*m) has constant ushort parity per row and the wanted half is
// picked with one hoisted shift; adjacent lanes share each uint32 texel.
__attribute__((qcom_wave_pair_mode(1)))
kernel void kernel_gemm_noshuffle_q4_0_q8_1_dp4a_wimg(
        __read_only image1d_buffer_t src0_q_img, // q4_0 nibbles as uint32 texels
        __global const half   * src0_d,
        __global const uint   * src1_qa,
        __global const half   * src1_da,
        __global const half   * src1_sa,
        __global       float  * dst,
        ulong  offsetd,
        int    m,
        int    n_no_padding,
        int    k
) {
    dst = (global float *)((global char *)dst + offsetd);

    const uint lid = get_local_id(0);
    const uint block_id_m = get_global_id(1);
    const uint block_id_n = get_global_id(2);

    const uint row      = block_id_m * 64 + lid;
    const uint col_base = block_id_n * TILESIZE_N;
    const bool row_valid = row < (uint)m;
    const uint rrow     = row_valid ? row : 0;

    const uint k_u = (uint)k >> 2;
    const uint k_b = (uint)k >> 5;

    // m is even (host-gated), so the half-select is loop-invariant.
    const uint sel = (rrow & 1u) * 16u;

    __local uint sh_qa[TILESIZE_N][8];
    __local half sh_d[TILESIZE_N];
    __local half sh_s[TILESIZE_N];

#define NGROUPS (TILESIZE_N / 4)
    float4 acc[NGROUPS];
    #pragma unroll
    for (int g = 0; g < NGROUPS; ++g) acc[g] = (float4)(0.0f);

    for (uint step = 0; step < (uint)k; step += 32) {
        const uint sub = step >> 5;

        const float d_w = (float)src0_d[rrow + sub * (uint)m];

        const uint qsbase = rrow + (step >> 2) * (uint)m;
        uint8 qw;
        qw.s0 = EXP4(read_imageui(src0_q_img, (int)((qsbase + 0 * m) >> 1)).x >> sel);
        qw.s1 = EXP4(read_imageui(src0_q_img, (int)((qsbase + 1 * m) >> 1)).x >> sel);
        qw.s2 = EXP4(read_imageui(src0_q_img, (int)((qsbase + 2 * m) >> 1)).x >> sel);
        qw.s3 = EXP4(read_imageui(src0_q_img, (int)((qsbase + 3 * m) >> 1)).x >> sel);
        qw.s4 = EXP4(read_imageui(src0_q_img, (int)((qsbase + 4 * m) >> 1)).x >> sel);
        qw.s5 = EXP4(read_imageui(src0_q_img, (int)((qsbase + 5 * m) >> 1)).x >> sel);
        qw.s6 = EXP4(read_imageui(src0_q_img, (int)((qsbase + 6 * m) >> 1)).x >> sel);
        qw.s7 = EXP4(read_imageui(src0_q_img, (int)((qsbase + 7 * m) >> 1)).x >> sel);

        for (uint idx = lid; idx < TILESIZE_N * 8; idx += 64) {
            const uint t = idx >> 3;
            const uint u = idx & 7;
            const uint c = col_base + t;
            sh_qa[t][u] = (c < (uint)n_no_padding) ? src1_qa[c * k_u + (step >> 2) + u] : 0u;
        }
        if (lid < TILESIZE_N) {
            const uint c = col_base + lid;
            sh_d[lid] = (c < (uint)n_no_padding) ? src1_da[c * k_b + sub] : (half)0;
            sh_s[lid] = (c < (uint)n_no_padding) ? src1_sa[c * k_b + sub] : (half)0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

#define LD4(arr, b) ((float4)((float)arr[(b)+0], (float)arr[(b)+1], (float)arr[(b)+2], (float)arr[(b)+3]))
        #pragma unroll
        for (int g = 0; g < NGROUPS; ++g) {
            const int b = g * 4;
            float4 rf;
            rf.s0 = (float)dot8_q8a(qw, sh_qa[b+0]);  rf.s1 = (float)dot8_q8a(qw, sh_qa[b+1]);
            rf.s2 = (float)dot8_q8a(qw, sh_qa[b+2]);  rf.s3 = (float)dot8_q8a(qw, sh_qa[b+3]);
            acc[g] += d_w * (LD4(sh_d, b) * rf - 8.0f * LD4(sh_s, b));
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


// ---------------------------------------------------------------------------
// uint4 staging-tile variants (`_alds4`).
//
// The kernels above stage a TILESIZE_N x 32-K tile of q8_1 activations into
// __local and then read it back ONE UINT AT A TIME, so the inner loop issues
// TILESIZE_N*8 scalar __local loads per 32-K step against the same number of
// dot instructions. The eight uints a token needs for one step are contiguous,
// so they are two uint4s: declaring the tile uint4 cuts the inner-loop __local
// load count 4x and widens the cooperative staging load from 4 to 16 bytes per
// lane. K%32==0 and the K slices are superblock aligned, so (c*k_u + step/4) is
// always a multiple of 8 and every vload4 is aligned.
//
// Measured first on q4_K (Adreno X2-90, muse-glimmer-30B): pp9 +7.1%,
// pp16 +7.4%, pp512 +15.2%, byte-identical output, and private/local/workgroup
// footprint unchanged. Same defect, same fix, here.
// ---------------------------------------------------------------------------

// 32-K dp4a dot with the token's 8 packed activation uints arriving as two
// uint4s in PRIVATE memory (also the documented workaround for dp4a
// miscompiling with a __local operand inside an unrolled loop).
inline int dot8_q8a_v(uint8 qw, uint4 a0, uint4 a1) {
    int r = 0;
    r = dot_acc_sat_4x8packed_ss_int(qw.s0, a0.x, r);
    r = dot_acc_sat_4x8packed_ss_int(qw.s1, a0.y, r);
    r = dot_acc_sat_4x8packed_ss_int(qw.s2, a0.z, r);
    r = dot_acc_sat_4x8packed_ss_int(qw.s3, a0.w, r);
    r = dot_acc_sat_4x8packed_ss_int(qw.s4, a1.x, r);
    r = dot_acc_sat_4x8packed_ss_int(qw.s5, a1.y, r);
    r = dot_acc_sat_4x8packed_ss_int(qw.s6, a1.z, r);
    r = dot_acc_sat_4x8packed_ss_int(qw.s7, a1.w, r);
    return r;
}

__attribute__((qcom_wave_pair_mode(1)))
kernel void kernel_gemm_noshuffle_q4_0_q8_1_dp4a_alds4(
        __global const ushort * src0_q,    // q4_0 nibbles (4/ushort, feature-major)
        __global const half   * src0_d,    // per-32-block scale, feature-major
        __global const uint   * src1_qa,   // q8_1 activations int8 (as uint, 4/elem) [N, K]
        __global const half   * src1_da,   // q8_1 per-block scale [N, K/32]
        __global const half   * src1_sa,   // q8_1 per-block sum*d  [N, K/32]
        __global       float  * dst,
        ulong  offsetd,
        int    m,                          // output features (rows)
        int    n_no_padding,               // tokens (cols)
        int    k                           // K (== ne00)
) {
    dst = (global float *)((global char *)dst + offsetd);

    const uint lid = get_local_id(0);          // 0..63 -> row within the M-tile
    const uint block_id_m = get_global_id(1);
    const uint block_id_n = get_global_id(2);

    const uint row      = block_id_m * 64 + lid;
    const uint col_base = block_id_n * TILESIZE_N;
    const bool row_valid = row < (uint)m;
    const uint rrow     = row_valid ? row : 0;  // clamp OOB rows; their writes are masked

    const uint k_u = (uint)k >> 2;   // K in uint (int8x4) units
    const uint k_b = (uint)k >> 5;   // blocks-of-32 along K

    __local uint4 sh_qa4[TILESIZE_N][2];
    __local half sh_d[TILESIZE_N];
    __local half sh_s[TILESIZE_N];

#define NGROUPS (TILESIZE_N / 4)
    float4 acc[NGROUPS];
    #pragma unroll
    for (int g = 0; g < NGROUPS; ++g) acc[g] = (float4)(0.0f);

    for (uint step = 0; step < (uint)k; step += 32) {
        const uint sub = step >> 5;

        const float d_w = (float)src0_d[rrow + sub * (uint)m];

        // 8 weight uints (32 nibbles) for this row, this 32-block. Feature-major:
        // src0_q[row + (k/4 + u)*m], k/4 = step/4 (= step>>2). EXP4 -> dp4a int8.
        const uint qsbase = rrow + (step >> 2) * (uint)m;
        uint8 qw;
        qw.s0 = EXP4(src0_q[qsbase + 0 * m]);
        qw.s1 = EXP4(src0_q[qsbase + 1 * m]);
        qw.s2 = EXP4(src0_q[qsbase + 2 * m]);
        qw.s3 = EXP4(src0_q[qsbase + 3 * m]);
        qw.s4 = EXP4(src0_q[qsbase + 4 * m]);
        qw.s5 = EXP4(src0_q[qsbase + 5 * m]);
        qw.s6 = EXP4(src0_q[qsbase + 6 * m]);
        qw.s7 = EXP4(src0_q[qsbase + 7 * m]);

        // cooperatively stage the 32-token x 32-K int8 activations to LDS
        for (uint idx = lid; idx < TILESIZE_N * 2; idx += 64) {
            const uint t = idx >> 1;
            const uint v = idx & 1;
            const uint c = col_base + t;
            sh_qa4[t][v] = (c < (uint)n_no_padding)
                         ? vload4(0, src1_qa + c * k_u + (step >> 2) + (v << 2))
                         : (uint4)(0u);
        }
        if (lid < TILESIZE_N) {
            const uint c = col_base + lid;
            sh_d[lid] = (c < (uint)n_no_padding) ? src1_da[c * k_b + sub] : (half)0;
            sh_s[lid] = (c < (uint)n_no_padding) ? src1_sa[c * k_b + sub] : (half)0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

#define LD4(arr, b) ((float4)((float)arr[(b)+0], (float)arr[(b)+1], (float)arr[(b)+2], (float)arr[(b)+3]))
        #pragma unroll
        for (int g = 0; g < NGROUPS; ++g) {
            const int b = g * 4;
            float4 rf;
#define DOTV(T) dot8_q8a_v(qw, (uint4)(sh_qa4[T][0]), (uint4)(sh_qa4[T][1]))
            rf.s0 = (float)DOTV(b+0);  rf.s1 = (float)DOTV(b+1);
            rf.s2 = (float)DOTV(b+2);  rf.s3 = (float)DOTV(b+3);
#undef DOTV
            // q4_0: w = d*(q-8) -> d_w * (a_d * dp4a(q,qa) - 8 * a_s)
            acc[g] += d_w * (LD4(sh_d, b) * rf - 8.0f * LD4(sh_s, b));
        }
#undef LD4
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (!row_valid) {
        return;
    }

    // dst is [token, feature] row-major (stride m): dst[col*m + row].
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

__attribute__((qcom_wave_pair_mode(1)))
kernel void kernel_gemm_noshuffle_q4_0_q8_1_dp4a_wimg_alds4(
        __read_only image1d_buffer_t src0_q_img, // q4_0 nibbles as uint32 texels
        __global const half   * src0_d,
        __global const uint   * src1_qa,
        __global const half   * src1_da,
        __global const half   * src1_sa,
        __global       float  * dst,
        ulong  offsetd,
        int    m,
        int    n_no_padding,
        int    k
) {
    dst = (global float *)((global char *)dst + offsetd);

    const uint lid = get_local_id(0);
    const uint block_id_m = get_global_id(1);
    const uint block_id_n = get_global_id(2);

    const uint row      = block_id_m * 64 + lid;
    const uint col_base = block_id_n * TILESIZE_N;
    const bool row_valid = row < (uint)m;
    const uint rrow     = row_valid ? row : 0;

    const uint k_u = (uint)k >> 2;
    const uint k_b = (uint)k >> 5;

    // m is even (host-gated), so the half-select is loop-invariant.
    const uint sel = (rrow & 1u) * 16u;

    __local uint4 sh_qa4[TILESIZE_N][2];
    __local half sh_d[TILESIZE_N];
    __local half sh_s[TILESIZE_N];

#define NGROUPS (TILESIZE_N / 4)
    float4 acc[NGROUPS];
    #pragma unroll
    for (int g = 0; g < NGROUPS; ++g) acc[g] = (float4)(0.0f);

    for (uint step = 0; step < (uint)k; step += 32) {
        const uint sub = step >> 5;

        const float d_w = (float)src0_d[rrow + sub * (uint)m];

        const uint qsbase = rrow + (step >> 2) * (uint)m;
        uint8 qw;
        qw.s0 = EXP4(read_imageui(src0_q_img, (int)((qsbase + 0 * m) >> 1)).x >> sel);
        qw.s1 = EXP4(read_imageui(src0_q_img, (int)((qsbase + 1 * m) >> 1)).x >> sel);
        qw.s2 = EXP4(read_imageui(src0_q_img, (int)((qsbase + 2 * m) >> 1)).x >> sel);
        qw.s3 = EXP4(read_imageui(src0_q_img, (int)((qsbase + 3 * m) >> 1)).x >> sel);
        qw.s4 = EXP4(read_imageui(src0_q_img, (int)((qsbase + 4 * m) >> 1)).x >> sel);
        qw.s5 = EXP4(read_imageui(src0_q_img, (int)((qsbase + 5 * m) >> 1)).x >> sel);
        qw.s6 = EXP4(read_imageui(src0_q_img, (int)((qsbase + 6 * m) >> 1)).x >> sel);
        qw.s7 = EXP4(read_imageui(src0_q_img, (int)((qsbase + 7 * m) >> 1)).x >> sel);

        for (uint idx = lid; idx < TILESIZE_N * 2; idx += 64) {
            const uint t = idx >> 1;
            const uint v = idx & 1;
            const uint c = col_base + t;
            sh_qa4[t][v] = (c < (uint)n_no_padding)
                         ? vload4(0, src1_qa + c * k_u + (step >> 2) + (v << 2))
                         : (uint4)(0u);
        }
        if (lid < TILESIZE_N) {
            const uint c = col_base + lid;
            sh_d[lid] = (c < (uint)n_no_padding) ? src1_da[c * k_b + sub] : (half)0;
            sh_s[lid] = (c < (uint)n_no_padding) ? src1_sa[c * k_b + sub] : (half)0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

#define LD4(arr, b) ((float4)((float)arr[(b)+0], (float)arr[(b)+1], (float)arr[(b)+2], (float)arr[(b)+3]))
        #pragma unroll
        for (int g = 0; g < NGROUPS; ++g) {
            const int b = g * 4;
            float4 rf;
#define DOTV(T) dot8_q8a_v(qw, (uint4)(sh_qa4[T][0]), (uint4)(sh_qa4[T][1]))
            rf.s0 = (float)DOTV(b+0);  rf.s1 = (float)DOTV(b+1);
            rf.s2 = (float)DOTV(b+2);  rf.s3 = (float)DOTV(b+3);
#undef DOTV
            acc[g] += d_w * (LD4(sh_d, b) * rf - 8.0f * LD4(sh_s, b));
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


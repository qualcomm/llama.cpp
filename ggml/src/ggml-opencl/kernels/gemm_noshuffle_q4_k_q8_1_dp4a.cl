#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#ifdef cl_khr_integer_dot_product
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#endif

// Dense q4_K prefill GEMM, dp4a (int8) inner loop.
//
// dp4a alternative to kernel_gemm_noshuffle_q4_k_f32 (the f16 half-dot GEMM used
// for the dense q4_K matmuls — attention Q/K/V/O projections). The activations
// are pre-quantized to q8_1 (kernel_quant_a_q8_1) straight from the original
// [N, K] token-major buffer (no transpose), and the per-subblock dot uses the
// qcom int8 dp4a, ~4x faster than half4 MAD on the X2.
//
// Each WI owns one output row (feature) and a TILESIZE_N-token tile, doing a
// dot over K via dp4a. Mirrors the MoE dp4a kernel without routing/scatter.
// q4_K reassociation per 32-subblock: Sum w*a = scale*a_d*dp4a(q,a) - minv*a_s.

#ifndef TILESIZE_N
#define TILESIZE_N 32
#endif
#define QK_K 256
#define K_SCALE_SIZE 12

// scales are transposed: consecutive codes of a row are `stride` apart
inline void get_scale_min_k4(
    int j,
    global const uchar * q,
    uint stride,
    uchar * d,
    uchar * m,
    uchar mask_d6,
    uchar mask_d4,
    uchar mask_hi2
) {
    if (j < 4) {
        *d = q[j*stride]     & mask_d6;
        *m = q[(j+4)*stride] & mask_d6;
    } else {
        *d = (q[(j+4)*stride] & mask_d4) | ((q[(j-4)*stride] & mask_hi2) >> 2);
        *m = ((q[(j+4)*stride] >> 4) & mask_d4) | ((q[j*stride] & mask_hi2) >> 2);
    }
}

// Expand the 4 nibbles in the low 16 bits of `u` into 4 bytes (one nibble per
// byte, value 0..15), packed for the int8 dp4a.
#define EXP4(u)  ( ((uint)((u) & 0x000Fu))        | \
                  (((uint)((u) & 0x00F0u)) << 4)  | \
                  (((uint)((u) & 0x0F00u)) << 8)  | \
                  (((uint)((u) & 0xF000u)) << 12) )

// 32-K dp4a dot of one token's int8 activations (8 packed uints in LDS) against the
// row's 8 packed weight uints. qw passed by value as a uint8 (register), not an array.
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
kernel void kernel_gemm_noshuffle_q4_k_q8_1_dp4a(
        __global const ushort * src0_q,    // q4_K weights (noshuffle, packed nibbles)
        __global const uchar  * src0_s,    // 6-bit scale/min codes
        __global const half   * src0_d,    // per-superblock scale
        __global const half   * src0_dm,   // per-superblock min
        __global const uint   * src1_qa,   // q8_1 activations int8 (as uint, 4/elem) [N, K]
        __global const half   * src1_da,   // q8_1 per-block scale [N, K/32]
        __global const half   * src1_sa,   // q8_1 per-block sum*d [N, K/32]
        __global       float  * dst,
        ulong  offsetd,
        int    m,                          // output features (rows)
        int    n_no_padding,               // tokens (cols)
        int    k,                          // K (== ne00)
        uchar  mask_d6,
        uchar  mask_d4,
        uchar  mask_hi2,
        int    ksplit                      // K-slices spread across workgroups; 1 = off
) {
    dst = (global float *)((global char *)dst + offsetd);

    const uint lid = get_local_id(0);          // 0..63 -> row within the M-tile
    const uint block_id_m = get_global_id(1);
    // dim2 packs (column tile, K-slice). At the verify widths there is exactly one
    // column tile, so the grid is one workgroup per 64 rows and the SP is starved;
    // splitting K is the only axis that adds workgroups when M and N are both fixed.
    const uint n_tiles    = ((uint)n_no_padding + TILESIZE_N - 1) / TILESIZE_N;
    const uint gid2       = get_global_id(2);
    const uint block_id_n = (n_tiles > 0) ? (gid2 % n_tiles) : gid2;
    const uint ks         = (n_tiles > 0) ? (gid2 / n_tiles) : 0u;

    const uint row      = block_id_m * 64 + lid;
    const uint col_base = block_id_n * TILESIZE_N;
    const bool row_valid = row < (uint)m;
    const uint rrow     = row_valid ? row : 0;  // clamp OOB rows; their writes are masked

    // Slices are superblock-aligned so the scale/min lookups stay valid inside one.
    // Spread the remainder so EVERY slice owns at least one superblock. A ceil()
    // split leaves trailing slices empty, and an empty slice returns without writing
    // its partial - which the reduce then sums as uninitialised memory. That is
    // invisible on a freshly allocated (zeroed) buffer and corrupts results once the
    // pre-allocated buffer is reused. The host clamps ksplit <= nsb, so sb_n >= 1.
    const uint nsb   = ((uint)k + QK_K - 1) / QK_K;
    const uint base  = nsb / (uint)ksplit;
    const uint rem   = nsb - base * (uint)ksplit;
    const uint sb_lo = ks * base + (ks < rem ? ks : rem);
    const uint sb_n  = base + (ks < rem ? 1u : 0u);
    const uint k_lo  = sb_lo * QK_K;
    uint       k_hi  = (sb_lo + sb_n) * QK_K;
    if (k_hi > (uint)k) { k_hi = (uint)k; }
    if (k_lo >= k_hi) {
        return;   // unreachable while the host clamps ksplit <= nsb; kept as a guard
    }
    dst += (size_t)ks * (size_t)n_no_padding * (size_t)m;

    const uint k_u = (uint)k >> 2;   // K in uint (int8x4) units
    const uint k_b = (uint)k >> 5;   // blocks-of-32 along K

    __local uint sh_qa[TILESIZE_N][8];
    __local half sh_d[TILESIZE_N];
    __local half sh_s[TILESIZE_N];

    // One float4 vector-register accumulator per group of 4 tokens (NGROUPS =
    // TILESIZE_N/4). NO per-WI private acc[] array: on Adreno X1 a private array
    // spills to private memory whose loads are issued per-wave with no cross-WI
    // coalescing (each WI pulls its own 512-bit line, no reuse) — that spill, not
    // the dp4a or LDS path, is the main cost. float4 (not float8) is Adreno's
    // native 128-bit register/transaction width: it packs into the register file
    // without 256-bit alignment padding. Byte-identical to the scalar acc[].
#define NGROUPS (TILESIZE_N / 4)
    float4 acc[NGROUPS];
    #pragma unroll
    for (int g = 0; g < NGROUPS; ++g) acc[g] = (float4)(0.0f);

    for (uint step = k_lo; step < k_hi; step += 32) {
        const uint sub     = step >> 5;
        const uint sb_idx  = step / QK_K;
        const uint sub_idx = sub & 7;

        // weight scale/min for this WI's row, this subblock
        const float dd  = (float)src0_d [rrow + sb_idx * m];
        const float dmm = (float)src0_dm[rrow + sb_idx * m];
        global const uchar * sc = src0_s + sb_idx * K_SCALE_SIZE * (uint)m + rrow;
        uchar sv, mn;
        get_scale_min_k4(sub_idx, sc, (uint)m, &sv, &mn, mask_d6, mask_d4, mask_hi2);
        const float scale = dd  * (float)sv;
        const float minv  = dmm * (float)mn;

        // repack this row's 32 weight nibbles into 8 dp4a uints. The packed q4_K
        // layout stores one ushort = 4 consecutive-K nibbles for a row at
        // src0_q[row + (K_group)*m], K_group = step/4 + u.
        const uint wbase = rrow + (step >> 2) * (uint)m;
        uint8 qw;
        qw.s0 = EXP4(src0_q[wbase + 0 * m]);
        qw.s1 = EXP4(src0_q[wbase + 1 * m]);
        qw.s2 = EXP4(src0_q[wbase + 2 * m]);
        qw.s3 = EXP4(src0_q[wbase + 3 * m]);
        qw.s4 = EXP4(src0_q[wbase + 4 * m]);
        qw.s5 = EXP4(src0_q[wbase + 5 * m]);
        qw.s6 = EXP4(src0_q[wbase + 6 * m]);
        qw.s7 = EXP4(src0_q[wbase + 7 * m]);

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
            acc[g] += scale * LD4(sh_d, b) * rf - minv * LD4(sh_s, b);
        }
#undef LD4
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (!row_valid) {
        return;
    }

    // dst is [token, feature] row-major (stride m): dst[col*m + row]. Scatter each
    // lane with a per-token padding guard (dst is non-contiguous in token).
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

// Weight-as-texture variant of kernel_gemm_noshuffle_q4_k_q8_1_dp4a.
//
// Byte-identical math; the ONLY change is that the q4_K weight plane is read
// through an image1d_buffer (read_imageui -> texture/L1 cache) instead of a
// plain __global buffer. Motivation (Adreno X1): the f16 half-dot GEMM beats
// the dp4a buffer GEMM on X1 not because the int8 dot is slow (it is ~2x the
// f16-mad rate) but because the f16 path streams weights through the texture
// cache and is near-BW-optimal, while the dp4a buffer path gives that up. The
// cross-N-tile weight reuse (same weight texels re-read for every 32-token
// tile) is exactly what the texture cache captures. This variant keeps the
// int8 ALU win AND the texture path; gated to X1, opt-in via
// GGML_OPENCL_Q4K_DENSE_DP4A_WIMG.
//
// The weight buffer is bound as CL_R/CL_UNSIGNED_INT32 (one texel = 2 packed
// ushorts), the same proven format the f16 _kimg / MoE _ns kernels use. The
// dispatch guarantees M%64==0 so m is even, hence every weight read for a
// given output row has constant ushort parity (= rrow&1): the wanted ushort is
// selected with a single hoisted shift, and adjacent lanes (rows) share each
// uint32 texel -> good cache-line / coalescing behaviour.
__attribute__((qcom_wave_pair_mode(1)))
kernel void kernel_gemm_noshuffle_q4_k_q8_1_dp4a_wimg(
        __read_only image1d_buffer_t src0_q_img, // q4_K weights as uint32 texels (2 ushorts/texel)
        __global const uchar  * src0_s,    // 6-bit scale/min codes
        __global const half   * src0_d,    // per-superblock scale
        __global const half   * src0_dm,   // per-superblock min
        __global const uint   * src1_qa,   // q8_1 activations int8 (as uint, 4/elem) [N, K]
        __global const half   * src1_da,   // q8_1 per-block scale [N, K/32]
        __global const half   * src1_sa,   // q8_1 per-block sum*d [N, K/32]
        __global       float  * dst,
        ulong  offsetd,
        int    m,                          // output features (rows)
        int    n_no_padding,               // tokens (cols)
        int    k,                          // K (== ne00)
        uchar  mask_d6,
        uchar  mask_d4,
        uchar  mask_hi2,
        int    ksplit                      // K-slices spread across workgroups; 1 = off
) {
    dst = (global float *)((global char *)dst + offsetd);

    const uint lid = get_local_id(0);          // 0..63 -> row within the M-tile
    const uint block_id_m = get_global_id(1);
    // dim2 packs (column tile, K-slice). At the verify widths there is exactly one
    // column tile, so the grid is one workgroup per 64 rows and the SP is starved;
    // splitting K is the only axis that adds workgroups when M and N are both fixed.
    const uint n_tiles    = ((uint)n_no_padding + TILESIZE_N - 1) / TILESIZE_N;
    const uint gid2       = get_global_id(2);
    const uint block_id_n = (n_tiles > 0) ? (gid2 % n_tiles) : gid2;
    const uint ks         = (n_tiles > 0) ? (gid2 / n_tiles) : 0u;

    const uint row      = block_id_m * 64 + lid;
    const uint col_base = block_id_n * TILESIZE_N;
    const bool row_valid = row < (uint)m;
    const uint rrow     = row_valid ? row : 0;  // clamp OOB rows; their writes are masked

    // Slices are superblock-aligned so the scale/min lookups stay valid inside one.
    // Spread the remainder so EVERY slice owns at least one superblock. A ceil()
    // split leaves trailing slices empty, and an empty slice returns without writing
    // its partial - which the reduce then sums as uninitialised memory. That is
    // invisible on a freshly allocated (zeroed) buffer and corrupts results once the
    // pre-allocated buffer is reused. The host clamps ksplit <= nsb, so sb_n >= 1.
    const uint nsb   = ((uint)k + QK_K - 1) / QK_K;
    const uint base  = nsb / (uint)ksplit;
    const uint rem   = nsb - base * (uint)ksplit;
    const uint sb_lo = ks * base + (ks < rem ? ks : rem);
    const uint sb_n  = base + (ks < rem ? 1u : 0u);
    const uint k_lo  = sb_lo * QK_K;
    uint       k_hi  = (sb_lo + sb_n) * QK_K;
    if (k_hi > (uint)k) { k_hi = (uint)k; }
    if (k_lo >= k_hi) {
        return;   // unreachable while the host clamps ksplit <= nsb; kept as a guard
    }
    dst += (size_t)ks * (size_t)n_no_padding * (size_t)m;

    // Constant per WI: the ushort the row needs always sits in the same half of
    // its uint32 texel (m even => index parity == rrow parity). Hoist the shift.
    const uint sel = (rrow & 1u) * 16u;

    const uint k_u = (uint)k >> 2;   // K in uint (int8x4) units
    const uint k_b = (uint)k >> 5;   // blocks-of-32 along K

    __local uint sh_qa[TILESIZE_N][8];
    __local half sh_d[TILESIZE_N];
    __local half sh_s[TILESIZE_N];

#define NGROUPS (TILESIZE_N / 4)
    float4 acc[NGROUPS];
    #pragma unroll
    for (int g = 0; g < NGROUPS; ++g) acc[g] = (float4)(0.0f);

    for (uint step = k_lo; step < k_hi; step += 32) {
        const uint sub     = step >> 5;
        const uint sb_idx  = step / QK_K;
        const uint sub_idx = sub & 7;

        const float dd  = (float)src0_d [rrow + sb_idx * m];
        const float dmm = (float)src0_dm[rrow + sb_idx * m];
        global const uchar * sc = src0_s + sb_idx * K_SCALE_SIZE * (uint)m + rrow;
        uchar sv, mn;
        get_scale_min_k4(sub_idx, sc, (uint)m, &sv, &mn, mask_d6, mask_d4, mask_hi2);
        const float scale = dd  * (float)sv;
        const float minv  = dmm * (float)mn;

        // Same logical ushort index (wbase + j*m) as the buffer kernel, read
        // through the texture: uint32 texel = ushort_index>>1, half = sel.
        const uint wbase = rrow + (step >> 2) * (uint)m;
        uint8 qw;
        qw.s0 = EXP4(read_imageui(src0_q_img, (int)((wbase + 0 * m) >> 1)).x >> sel);
        qw.s1 = EXP4(read_imageui(src0_q_img, (int)((wbase + 1 * m) >> 1)).x >> sel);
        qw.s2 = EXP4(read_imageui(src0_q_img, (int)((wbase + 2 * m) >> 1)).x >> sel);
        qw.s3 = EXP4(read_imageui(src0_q_img, (int)((wbase + 3 * m) >> 1)).x >> sel);
        qw.s4 = EXP4(read_imageui(src0_q_img, (int)((wbase + 4 * m) >> 1)).x >> sel);
        qw.s5 = EXP4(read_imageui(src0_q_img, (int)((wbase + 5 * m) >> 1)).x >> sel);
        qw.s6 = EXP4(read_imageui(src0_q_img, (int)((wbase + 6 * m) >> 1)).x >> sel);
        qw.s7 = EXP4(read_imageui(src0_q_img, (int)((wbase + 7 * m) >> 1)).x >> sel);

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
            acc[g] += scale * LD4(sh_d, b) * rf - minv * LD4(sh_s, b);
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

// Activation-as-texture variant of kernel_gemm_noshuffle_q4_k_q8_1_dp4a_wimg.
//
// Same math, same weight path (texture). The ONE change is where the q8_1
// ACTIVATIONS come from: the buffer kernels stage a TILESIZE_N x 32-K tile into
// __local and then read it back inside the dot, so the inner loop issues
// TILESIZE_N*8 = 128 (at TS=16) __local loads per 32-K step against 128 dp4a
// instructions -- LDS is ~35% of the inner-loop issue slots.
//
// Every one of those __local reads is WAVE-UNIFORM: sh_qa[t][u] depends only on
// loop constants, so all 64 lanes read the same address (only `qw`, the weight,
// is per-lane). That is exactly the pattern measured on X2-90 in the cok
// experiment (kernel_gemm_noshuffle_q4_k_f32_cok_lds): staging a wave-uniform
// read into __local cost -50%, reproduced with a byte-identical LDS footprint,
// i.e. it was not occupancy. The conclusion there was that a wave-uniform
// read_image is serviced as a broadcast by the texture pipe + L1 CONCURRENTLY
// with the ALU, while LDS competes with the ALU issue port. cok therefore reads
// its activations from an image; every dp4a kernel in the tree still uses LDS.
//
// Two wins, not one:
//   1. the 128 LDS loads/step become texture reads on the free path, and the
//      staging loop plus both barriers per step disappear;
//   2. the image is bound CL_RGBA/CL_UNSIGNED_INT32, so ONE read returns a uint4
//      = the 4 packed int8x4 words a dp4a quartet needs. 8 uints per token per
//      32-K step = 2 texel reads instead of 8 scalar loads, so the activation
//      read COUNT also drops 4x (128 -> 32 per step).
// Alignment for (2) is guaranteed: the host gates K%32==0, so k/4 (uints per
// token row) is a multiple of 8, and step is a multiple of 32, hence every
// (c*k/4 + step/4) index is a multiple of 4 and lands on a uint4 texel boundary.
//
// Padded token slots are clamped to the last real column rather than branched
// or zero-filled: their accumulators are computed with the wrong activations and
// then DISCARDED by the same n_no_padding guard that already masks the stores,
// so a clamp is cheaper than a select and cannot read out of bounds.
//
// It also sidesteps a documented hazard: dp4a with a __local operand inside an
// unrolled hot loop miscompiles on some X2 drivers (the workaround on record is
// "keep dp4a operands in private memory"). Here they arrive in registers.
//
// Index arithmetic uses mad24; the host gates n*k so the products stay < 2^24.
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
kernel void kernel_gemm_noshuffle_q4_k_q8_1_dp4a_wimg_aimg(
        __read_only image1d_buffer_t src0_q_img,  // q4_K weights, CL_R/UINT32 (2 ushorts/texel)
        __global const uchar  * src0_s,    // 6-bit scale/min codes
        __global const half   * src0_d,    // per-superblock scale
        __global const half   * src0_dm,   // per-superblock min
        __read_only image1d_buffer_t src1_qa_img, // q8_1 activations, CL_RGBA/UINT32 [N, K/16]
        __global const half   * src1_da,   // q8_1 per-block scale [N, K/32]
        __global const half   * src1_sa,   // q8_1 per-block sum*d [N, K/32]
        __global       float  * dst,
        ulong  offsetd,
        int    m,                          // output features (rows)
        int    n_no_padding,               // tokens (cols)
        int    k,                          // K (== ne00)
        uchar  mask_d6,
        uchar  mask_d4,
        uchar  mask_hi2,
        int    ksplit                      // K-slices spread across workgroups; 1 = off
) {
    dst = (global float *)((global char *)dst + offsetd);

    const uint lid = get_local_id(0);          // 0..63 -> row within the M-tile
    const uint block_id_m = get_global_id(1);
    const uint n_tiles    = ((uint)n_no_padding + TILESIZE_N - 1) / TILESIZE_N;
    const uint gid2       = get_global_id(2);
    const uint block_id_n = (n_tiles > 0) ? (gid2 % n_tiles) : gid2;
    const uint ks         = (n_tiles > 0) ? (gid2 / n_tiles) : 0u;

    const uint row      = block_id_m * 64 + lid;
    const uint col_base = block_id_n * TILESIZE_N;
    const bool row_valid = row < (uint)m;
    const uint rrow     = row_valid ? row : 0;  // clamp OOB rows; their writes are masked

    // Superblock-aligned K slices; every slice owns at least one superblock (see
    // the buffer kernel above for why an empty slice corrupts the reduce).
    const uint nsb   = ((uint)k + QK_K - 1) / QK_K;
    const uint base  = nsb / (uint)ksplit;
    const uint rem   = nsb - base * (uint)ksplit;
    const uint sb_lo = ks * base + (ks < rem ? ks : rem);
    const uint sb_n  = base + (ks < rem ? 1u : 0u);
    const uint k_lo  = sb_lo * QK_K;
    uint       k_hi  = (sb_lo + sb_n) * QK_K;
    if (k_hi > (uint)k) { k_hi = (uint)k; }
    if (k_lo >= k_hi) {
        return;
    }
    dst += (size_t)ks * (size_t)n_no_padding * (size_t)m;

    // Constant per WI: m even => the row's ushort always sits in the same half of
    // its uint32 weight texel. Hoist the shift.
    const uint sel = (rrow & 1u) * 16u;

    const uint k_b  = (uint)k >> 5;   // blocks-of-32 along K (da/sa stride)
    const uint ku4  = (uint)k >> 4;   // uint4 texels per token row = (k/4)/4
    const uint nm1  = (uint)n_no_padding - 1u;

    // Clamped token index for tile slot T. Loop-invariant in `step`, so the
    // compiler hoists the min and both multiplies out of the K sweep.
#define TCOL(T)  min((uint)(col_base + (uint)(T)), nm1)
#define TDOT(T)  dot8_q8a_v(qw, \
                     read_imageui(src1_qa_img, (int)(mad24(TCOL(T), ku4, st4))), \
                     read_imageui(src1_qa_img, (int)(mad24(TCOL(T), ku4, st4) + 1u)))

    // The per-block q8_1 scales stay staged in __local exactly as the buffer
    // kernel has them: 2 loads per step instead of 2*TILESIZE_N, and moving them
    // is a separate question from moving the activations.
    __local half sh_d[TILESIZE_N];
    __local half sh_s[TILESIZE_N];

#define NGROUPS (TILESIZE_N / 4)
    float4 acc[NGROUPS];
    #pragma unroll
    for (int g = 0; g < NGROUPS; ++g) acc[g] = (float4)(0.0f);

    for (uint step = k_lo; step < k_hi; step += 32) {
        const uint sub     = step >> 5;
        const uint sb_idx  = step / QK_K;
        const uint sub_idx = sub & 7;
        const uint st4     = step >> 4;   // uint4-texel offset within a token row

        const float dd  = (float)src0_d [rrow + sb_idx * m];
        const float dmm = (float)src0_dm[rrow + sb_idx * m];
        global const uchar * sc = src0_s + sb_idx * K_SCALE_SIZE * (uint)m + rrow;
        uchar sv, mn;
        get_scale_min_k4(sub_idx, sc, (uint)m, &sv, &mn, mask_d6, mask_d4, mask_hi2);
        const float scale = dd  * (float)sv;
        const float minv  = dmm * (float)mn;

        const uint wbase = rrow + (step >> 2) * (uint)m;
        uint8 qw;
        qw.s0 = EXP4(read_imageui(src0_q_img, (int)((wbase + 0 * m) >> 1)).x >> sel);
        qw.s1 = EXP4(read_imageui(src0_q_img, (int)((wbase + 1 * m) >> 1)).x >> sel);
        qw.s2 = EXP4(read_imageui(src0_q_img, (int)((wbase + 2 * m) >> 1)).x >> sel);
        qw.s3 = EXP4(read_imageui(src0_q_img, (int)((wbase + 3 * m) >> 1)).x >> sel);
        qw.s4 = EXP4(read_imageui(src0_q_img, (int)((wbase + 4 * m) >> 1)).x >> sel);
        qw.s5 = EXP4(read_imageui(src0_q_img, (int)((wbase + 5 * m) >> 1)).x >> sel);
        qw.s6 = EXP4(read_imageui(src0_q_img, (int)((wbase + 6 * m) >> 1)).x >> sel);
        qw.s7 = EXP4(read_imageui(src0_q_img, (int)((wbase + 7 * m) >> 1)).x >> sel);

        if (lid < TILESIZE_N) {
            const uint c = col_base + lid;
            sh_d[lid] = (c < (uint)n_no_padding) ? src1_da[c * k_b + sub] : (half)0;
            sh_s[lid] = (c < (uint)n_no_padding) ? src1_sa[c * k_b + sub] : (half)0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // The activations - and only the activations - come from the texture.
#define LD4(arr, b) ((float4)((float)arr[(b)+0], (float)arr[(b)+1], (float)arr[(b)+2], (float)arr[(b)+3]))
        #pragma unroll
        for (int g = 0; g < NGROUPS; ++g) {
            const int b = g * 4;
            const float4 rf = (float4)((float)TDOT(b+0), (float)TDOT(b+1),
                                       (float)TDOT(b+2), (float)TDOT(b+3));
            acc[g] += scale * LD4(sh_d, b) * rf - minv * LD4(sh_s, b);
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
#undef TDOT
#undef TCOL
}


// Same question, cheaper answer: keep the activations in __local, but read them
// FOUR AT A TIME.
//
// The buffer kernels declare the staging tile as `__local uint sh_qa[TS][8]` and
// the dot reads it one uint at a time, so the inner loop issues TS*8 = 128 scalar
// __local loads per 32-K step (TS=16). Nothing about the data requires that: the
// eight uints a token needs for one 32-K step are contiguous, so they are two
// uint4s. Declaring the tile as uint4 cuts the inner-loop __local load count 4x,
// and the cooperative staging load from global widens from 4 to 16 bytes per lane
// at the same time.
//
// This isolates the LOAD COUNT from the ADDRESS SPACE: the _aimg variant above
// changes both (fewer reads AND a texture); this one changes only the count.
// The uint4 is copied into a private temp before the dp4a, because dp4a with a
// __local operand inside an unrolled loop is a documented miscompile on X2.
__attribute__((qcom_wave_pair_mode(1)))
kernel void kernel_gemm_noshuffle_q4_k_q8_1_dp4a_wimg_alds4(
        __read_only image1d_buffer_t src0_q_img,  // q4_K weights, CL_R/UINT32
        __global const uchar  * src0_s,
        __global const half   * src0_d,
        __global const half   * src0_dm,
        __global const uint   * src1_qa,   // q8_1 activations int8 (as uint, 4/elem) [N, K]
        __global const half   * src1_da,
        __global const half   * src1_sa,
        __global       float  * dst,
        ulong  offsetd,
        int    m,
        int    n_no_padding,
        int    k,
        uchar  mask_d6,
        uchar  mask_d4,
        uchar  mask_hi2,
        int    ksplit
) {
    dst = (global float *)((global char *)dst + offsetd);

    const uint lid = get_local_id(0);
    const uint block_id_m = get_global_id(1);
    const uint n_tiles    = ((uint)n_no_padding + TILESIZE_N - 1) / TILESIZE_N;
    const uint gid2       = get_global_id(2);
    const uint block_id_n = (n_tiles > 0) ? (gid2 % n_tiles) : gid2;
    const uint ks         = (n_tiles > 0) ? (gid2 / n_tiles) : 0u;

    const uint row      = block_id_m * 64 + lid;
    const uint col_base = block_id_n * TILESIZE_N;
    const bool row_valid = row < (uint)m;
    const uint rrow     = row_valid ? row : 0;

    const uint nsb   = ((uint)k + QK_K - 1) / QK_K;
    const uint base  = nsb / (uint)ksplit;
    const uint rem   = nsb - base * (uint)ksplit;
    const uint sb_lo = ks * base + (ks < rem ? ks : rem);
    const uint sb_n  = base + (ks < rem ? 1u : 0u);
    const uint k_lo  = sb_lo * QK_K;
    uint       k_hi  = (sb_lo + sb_n) * QK_K;
    if (k_hi > (uint)k) { k_hi = (uint)k; }
    if (k_lo >= k_hi) {
        return;
    }
    dst += (size_t)ks * (size_t)n_no_padding * (size_t)m;

    const uint sel = (rrow & 1u) * 16u;
    const uint k_u = (uint)k >> 2;
    const uint k_b = (uint)k >> 5;

    __local uint4 sh_qa4[TILESIZE_N][2];
    __local half  sh_d[TILESIZE_N];
    __local half  sh_s[TILESIZE_N];

#define NGROUPS (TILESIZE_N / 4)
    float4 acc[NGROUPS];
    #pragma unroll
    for (int g = 0; g < NGROUPS; ++g) acc[g] = (float4)(0.0f);

    for (uint step = k_lo; step < k_hi; step += 32) {
        const uint sub     = step >> 5;
        const uint sb_idx  = step / QK_K;
        const uint sub_idx = sub & 7;

        const float dd  = (float)src0_d [rrow + sb_idx * m];
        const float dmm = (float)src0_dm[rrow + sb_idx * m];
        global const uchar * sc = src0_s + sb_idx * K_SCALE_SIZE * (uint)m + rrow;
        uchar sv, mn;
        get_scale_min_k4(sub_idx, sc, (uint)m, &sv, &mn, mask_d6, mask_d4, mask_hi2);
        const float scale = dd  * (float)sv;
        const float minv  = dmm * (float)mn;

        const uint wbase = rrow + (step >> 2) * (uint)m;
        uint8 qw;
        qw.s0 = EXP4(read_imageui(src0_q_img, (int)((wbase + 0 * m) >> 1)).x >> sel);
        qw.s1 = EXP4(read_imageui(src0_q_img, (int)((wbase + 1 * m) >> 1)).x >> sel);
        qw.s2 = EXP4(read_imageui(src0_q_img, (int)((wbase + 2 * m) >> 1)).x >> sel);
        qw.s3 = EXP4(read_imageui(src0_q_img, (int)((wbase + 3 * m) >> 1)).x >> sel);
        qw.s4 = EXP4(read_imageui(src0_q_img, (int)((wbase + 4 * m) >> 1)).x >> sel);
        qw.s5 = EXP4(read_imageui(src0_q_img, (int)((wbase + 5 * m) >> 1)).x >> sel);
        qw.s6 = EXP4(read_imageui(src0_q_img, (int)((wbase + 6 * m) >> 1)).x >> sel);
        qw.s7 = EXP4(read_imageui(src0_q_img, (int)((wbase + 7 * m) >> 1)).x >> sel);

        // 16-byte cooperative staging: TILESIZE_N*2 uint4s instead of TILESIZE_N*8
        // uints. (c*k_u + step/4) is a multiple of 8, so vload4 is aligned.
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
#define TDOT4(T) dot8_q8a_v(qw, (uint4)(sh_qa4[T][0]), (uint4)(sh_qa4[T][1]))
        #pragma unroll
        for (int g = 0; g < NGROUPS; ++g) {
            const int b = g * 4;
            const float4 rf = (float4)((float)TDOT4(b+0), (float)TDOT4(b+1),
                                       (float)TDOT4(b+2), (float)TDOT4(b+3));
            acc[g] += scale * LD4(sh_d, b) * rf - minv * LD4(sh_s, b);
        }
#undef TDOT4
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

// Activation-from-global twin of _wimg_alds4 (`_agm`): same weights-as-texture path, but
// the q8_1 activation tile is read straight from the global buffer instead of being staged
// into __local, which also removes both barriers per 32-K step.
//
// Rationale and the risk, both on record. The tile is wave-uniform, and on the
// cooperative-K kernel ADDING an LDS stage to exactly this pattern measured -50%, i.e. a
// wave-uniform broadcast read beats LDS, which competes with the ALU issue port. But the
// texture version of this idea (`_aimg`) measured -46% here, and the cause was register
// pressure, not the address space: a dp4a needs 8 words live before it can consume them,
// so the compiler hoists the loads and private memory went 336 -> 528 B with the workgroup
// cap halved.
//
// ANSWERED, AND IT IS WORSE THAN THE TEXTURE: pp9 40.5 -> 17.6, pp16 69.2 -> 21.8
// (-68%), bookended to 0.2% on an Adreno X2-90. DEFAULT OFF, kept as the recorded
// negative result alongside _aimg.
//
// Resources explain only half of it: private 336 -> 464 B (the compiler hoists the
// loads, same direction as _aimg's 528 but milder), local 576 -> 0, and the workgroup
// cap goes 384 -> 1024 because a barrier-free kernel is not register capped on this
// part. The rest is the redundancy itself: staging loads the tile ONCE cooperatively
// and then lets 64 lanes read it from LDS, whereas reading from global makes all 64
// lanes issue their own load for the same words -- 64x the load instructions and L1
// traffic, which no broadcast recovers.
//
// So the cok finding ("a wave-uniform read is free, LDS competes with the ALU issue
// port") does NOT generalise to dp4a, and the reason is not the address space: cok
// feeds its activation into a half8 MAC immediately, while a dp4a needs 8 words live
// per token and 4 tokens per group. Any high-latency per-lane path loses here,
// texture or buffer alike. The __local tile is earning its keep.

__attribute__((qcom_wave_pair_mode(1)))
kernel void kernel_gemm_noshuffle_q4_k_q8_1_dp4a_wimg_agm(
        __read_only image1d_buffer_t src0_q_img,  // q4_K weights, CL_R/UINT32
        __global const uchar  * src0_s,
        __global const half   * src0_d,
        __global const half   * src0_dm,
        __global const uint   * src1_qa,   // q8_1 activations int8 (as uint, 4/elem) [N, K]
        __global const half   * src1_da,
        __global const half   * src1_sa,
        __global       float  * dst,
        ulong  offsetd,
        int    m,
        int    n_no_padding,
        int    k,
        uchar  mask_d6,
        uchar  mask_d4,
        uchar  mask_hi2,
        int    ksplit
) {
    dst = (global float *)((global char *)dst + offsetd);

    const uint lid = get_local_id(0);
    const uint block_id_m = get_global_id(1);
    const uint n_tiles    = ((uint)n_no_padding + TILESIZE_N - 1) / TILESIZE_N;
    const uint gid2       = get_global_id(2);
    const uint block_id_n = (n_tiles > 0) ? (gid2 % n_tiles) : gid2;
    const uint ks         = (n_tiles > 0) ? (gid2 / n_tiles) : 0u;

    const uint row      = block_id_m * 64 + lid;
    const uint col_base = block_id_n * TILESIZE_N;
    const bool row_valid = row < (uint)m;
    const uint rrow     = row_valid ? row : 0;

    const uint nsb   = ((uint)k + QK_K - 1) / QK_K;
    const uint base  = nsb / (uint)ksplit;
    const uint rem   = nsb - base * (uint)ksplit;
    const uint sb_lo = ks * base + (ks < rem ? ks : rem);
    const uint sb_n  = base + (ks < rem ? 1u : 0u);
    const uint k_lo  = sb_lo * QK_K;
    uint       k_hi  = (sb_lo + sb_n) * QK_K;
    if (k_hi > (uint)k) { k_hi = (uint)k; }
    if (k_lo >= k_hi) {
        return;
    }
    dst += (size_t)ks * (size_t)n_no_padding * (size_t)m;

    const uint sel = (rrow & 1u) * 16u;
    const uint k_u = (uint)k >> 2;
    const uint k_b = (uint)k >> 5;

#define NGROUPS (TILESIZE_N / 4)
    float4 acc[NGROUPS];
    #pragma unroll
    for (int g = 0; g < NGROUPS; ++g) acc[g] = (float4)(0.0f);

    for (uint step = k_lo; step < k_hi; step += 32) {
        const uint sub     = step >> 5;
        const uint sb_idx  = step / QK_K;
        const uint sub_idx = sub & 7;

        const float dd  = (float)src0_d [rrow + sb_idx * m];
        const float dmm = (float)src0_dm[rrow + sb_idx * m];
        global const uchar * sc = src0_s + sb_idx * K_SCALE_SIZE * (uint)m + rrow;
        uchar sv, mn;
        get_scale_min_k4(sub_idx, sc, (uint)m, &sv, &mn, mask_d6, mask_d4, mask_hi2);
        const float scale = dd  * (float)sv;
        const float minv  = dmm * (float)mn;

        const uint wbase = rrow + (step >> 2) * (uint)m;
        uint8 qw;
        qw.s0 = EXP4(read_imageui(src0_q_img, (int)((wbase + 0 * m) >> 1)).x >> sel);
        qw.s1 = EXP4(read_imageui(src0_q_img, (int)((wbase + 1 * m) >> 1)).x >> sel);
        qw.s2 = EXP4(read_imageui(src0_q_img, (int)((wbase + 2 * m) >> 1)).x >> sel);
        qw.s3 = EXP4(read_imageui(src0_q_img, (int)((wbase + 3 * m) >> 1)).x >> sel);
        qw.s4 = EXP4(read_imageui(src0_q_img, (int)((wbase + 4 * m) >> 1)).x >> sel);
        qw.s5 = EXP4(read_imageui(src0_q_img, (int)((wbase + 5 * m) >> 1)).x >> sel);
        qw.s6 = EXP4(read_imageui(src0_q_img, (int)((wbase + 6 * m) >> 1)).x >> sel);
        qw.s7 = EXP4(read_imageui(src0_q_img, (int)((wbase + 7 * m) >> 1)).x >> sel);

        // Activations straight from global, no __local tile and no barriers. The tile is
        // WAVE-UNIFORM -- it is indexed by token and K-offset, never by lid, so all 64
        // lanes of the workgroup want the same words and the load is an L1 broadcast.
        // Columns past n_no_padding must contribute zero; the guard is wave-uniform too.
#define AQ(T)    (src1_qa + (size_t)(col_base + (uint)(T)) * k_u + (step >> 2))
#define AOK(T)   ((col_base + (uint)(T)) < (uint)n_no_padding)
#define TDOT4(T) (AOK(T) ? dot8_q8a_v(qw, vload4(0, AQ(T)), vload4(0, AQ(T) + 4)) : 0)
#define AD(T)    (AOK(T) ? (float)src1_da[(size_t)(col_base + (uint)(T)) * k_b + sub] : 0.0f)
#define AS(T)    (AOK(T) ? (float)src1_sa[(size_t)(col_base + (uint)(T)) * k_b + sub] : 0.0f)
#define LD4(arr, b) ((float4)(arr(b+0), arr(b+1), arr(b+2), arr(b+3)))
        #pragma unroll
        for (int g = 0; g < NGROUPS; ++g) {
            const int b = g * 4;
            const float4 rf = (float4)((float)TDOT4(b+0), (float)TDOT4(b+1),
                                       (float)TDOT4(b+2), (float)TDOT4(b+3));
            acc[g] += scale * LD4(AD, b) * rf - minv * LD4(AS, b);
        }
#undef TDOT4
#undef AQ
#undef AOK
#undef AD
#undef AS
#undef LD4
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

// Buffer-weight twin of _wimg_alds4: the uint4 staging change is independent of
// how the weights are read, and the weight-texture gate is a narrow-band default,
// so the plain-buffer path needs the same kernel to get the same win.
__attribute__((qcom_wave_pair_mode(1)))
kernel void kernel_gemm_noshuffle_q4_k_q8_1_dp4a_alds4(
        __global const ushort * src0_q,    // q4_K weights (noshuffle, packed nibbles)
        __global const uchar  * src0_s,
        __global const half   * src0_d,
        __global const half   * src0_dm,
        __global const uint   * src1_qa,   // q8_1 activations int8 (as uint, 4/elem) [N, K]
        __global const half   * src1_da,
        __global const half   * src1_sa,
        __global       float  * dst,
        ulong  offsetd,
        int    m,
        int    n_no_padding,
        int    k,
        uchar  mask_d6,
        uchar  mask_d4,
        uchar  mask_hi2,
        int    ksplit
) {
    dst = (global float *)((global char *)dst + offsetd);

    const uint lid = get_local_id(0);
    const uint block_id_m = get_global_id(1);
    const uint n_tiles    = ((uint)n_no_padding + TILESIZE_N - 1) / TILESIZE_N;
    const uint gid2       = get_global_id(2);
    const uint block_id_n = (n_tiles > 0) ? (gid2 % n_tiles) : gid2;
    const uint ks         = (n_tiles > 0) ? (gid2 / n_tiles) : 0u;

    const uint row      = block_id_m * 64 + lid;
    const uint col_base = block_id_n * TILESIZE_N;
    const bool row_valid = row < (uint)m;
    const uint rrow     = row_valid ? row : 0;

    const uint nsb   = ((uint)k + QK_K - 1) / QK_K;
    const uint base  = nsb / (uint)ksplit;
    const uint rem   = nsb - base * (uint)ksplit;
    const uint sb_lo = ks * base + (ks < rem ? ks : rem);
    const uint sb_n  = base + (ks < rem ? 1u : 0u);
    const uint k_lo  = sb_lo * QK_K;
    uint       k_hi  = (sb_lo + sb_n) * QK_K;
    if (k_hi > (uint)k) { k_hi = (uint)k; }
    if (k_lo >= k_hi) {
        return;
    }
    dst += (size_t)ks * (size_t)n_no_padding * (size_t)m;

    const uint k_u = (uint)k >> 2;
    const uint k_b = (uint)k >> 5;

    __local uint4 sh_qa4[TILESIZE_N][2];
    __local half  sh_d[TILESIZE_N];
    __local half  sh_s[TILESIZE_N];

#define NGROUPS (TILESIZE_N / 4)
    float4 acc[NGROUPS];
    #pragma unroll
    for (int g = 0; g < NGROUPS; ++g) acc[g] = (float4)(0.0f);

    for (uint step = k_lo; step < k_hi; step += 32) {
        const uint sub     = step >> 5;
        const uint sb_idx  = step / QK_K;
        const uint sub_idx = sub & 7;

        const float dd  = (float)src0_d [rrow + sb_idx * m];
        const float dmm = (float)src0_dm[rrow + sb_idx * m];
        global const uchar * sc = src0_s + sb_idx * K_SCALE_SIZE * (uint)m + rrow;
        uchar sv, mn;
        get_scale_min_k4(sub_idx, sc, (uint)m, &sv, &mn, mask_d6, mask_d4, mask_hi2);
        const float scale = dd  * (float)sv;
        const float minv  = dmm * (float)mn;

        const uint wbase = rrow + (step >> 2) * (uint)m;
        uint8 qw;
        qw.s0 = EXP4(src0_q[wbase + 0 * m]);
        qw.s1 = EXP4(src0_q[wbase + 1 * m]);
        qw.s2 = EXP4(src0_q[wbase + 2 * m]);
        qw.s3 = EXP4(src0_q[wbase + 3 * m]);
        qw.s4 = EXP4(src0_q[wbase + 4 * m]);
        qw.s5 = EXP4(src0_q[wbase + 5 * m]);
        qw.s6 = EXP4(src0_q[wbase + 6 * m]);
        qw.s7 = EXP4(src0_q[wbase + 7 * m]);

        // 16-byte cooperative staging: TILESIZE_N*2 uint4s instead of TILESIZE_N*8
        // uints. (c*k_u + step/4) is a multiple of 8, so vload4 is aligned.
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
#define TDOT4(T) dot8_q8a_v(qw, (uint4)(sh_qa4[T][0]), (uint4)(sh_qa4[T][1]))
        #pragma unroll
        for (int g = 0; g < NGROUPS; ++g) {
            const int b = g * 4;
            const float4 rf = (float4)((float)TDOT4(b+0), (float)TDOT4(b+1),
                                       (float)TDOT4(b+2), (float)TDOT4(b+3));
            acc[g] += scale * LD4(sh_d, b) * rf - minv * LD4(sh_s, b);
        }
#undef TDOT4
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

// Sums the per-K-slice partials written by the split-K dp4a GEMM above.
// partial is [ksplit][n][m] contiguous; dst is the usual [n][m].
kernel void kernel_gemm_q4_k_splitk_reduce_f32(
        __global const float * partial,
        __global       float * dst,
        ulong  offsetd,
        int    m,
        int    n,
        int    ksplit
) {
    dst = (global float *)((global char *)dst + offsetd);
    const uint idx = get_global_id(0);
    const uint tot = (uint)m * (uint)n;
    if (idx >= tot) {
        return;
    }
    float sum = 0.0f;
    for (int t = 0; t < ksplit; ++t) {
        sum += partial[(size_t)t * (size_t)tot + idx];
    }
    dst[idx] = sum;
}

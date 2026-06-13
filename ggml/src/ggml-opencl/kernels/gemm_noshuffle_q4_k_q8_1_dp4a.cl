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

#define TILESIZE_N 32
#define QK_K 256
#define K_SCALE_SIZE 12

inline void get_scale_min_k4(
    int j,
    global const uchar * q,
    uchar * d,
    uchar * m,
    uchar mask_d6,
    uchar mask_d4,
    uchar mask_hi2
) {
    if (j < 4) {
        *d = q[j]   & mask_d6;
        *m = q[j+4] & mask_d6;
    } else {
        *d = (q[j+4] & mask_d4) | ((q[j-4] & mask_hi2) >> 2);
        *m = ((q[j+4] >> 4) & mask_d4) | ((q[j]   & mask_hi2) >> 2);
    }
}

// Expand the 4 nibbles in the low 16 bits of `u` into 4 bytes (one nibble per
// byte, value 0..15), packed for the int8 dp4a.
#define EXP4(u)  ( ((uint)((u) & 0x000Fu))        | \
                  (((uint)((u) & 0x00F0u)) << 4)  | \
                  (((uint)((u) & 0x0F00u)) << 8)  | \
                  (((uint)((u) & 0xF000u)) << 12) )

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
        uchar  mask_hi2
) {
    dst = (global float *)((global char *)dst + offsetd);

    const uint lid = get_local_id(0);          // 0..63 -> row within the M-tile
    const uint block_id_m = get_global_id(1);
    const uint block_id_n = get_global_id(2);

    const uint row      = block_id_m * 64 + lid;
    const uint col_base = block_id_n * TILESIZE_N;
    const bool row_valid = row < (uint)m;
    const uint rrow     = row_valid ? row : 0;  // clamp OOB rows; their writes are masked

    const uint num_superblocks = (uint)k / QK_K;
    const uint k_u = (uint)k >> 2;   // K in uint (int8x4) units
    const uint k_b = (uint)k >> 5;   // blocks-of-32 along K

    __local uint sh_qa[TILESIZE_N][8];
    __local half sh_d[TILESIZE_N];
    __local half sh_s[TILESIZE_N];

    float acc[TILESIZE_N];
    #pragma unroll
    for (int t = 0; t < TILESIZE_N; ++t) acc[t] = 0.0f;

    for (uint step = 0; step < (uint)k; step += 32) {
        const uint sub     = step >> 5;
        const uint sb_idx  = step / QK_K;
        const uint sub_idx = sub & 7;

        // weight scale/min for this WI's row, this subblock
        const float dd  = (float)src0_d [rrow + sb_idx * m];
        const float dmm = (float)src0_dm[rrow + sb_idx * m];
        global const uchar * sc = src0_s + rrow * num_superblocks * K_SCALE_SIZE + sb_idx * K_SCALE_SIZE;
        uchar sv, mn;
        get_scale_min_k4(sub_idx, sc, &sv, &mn, mask_d6, mask_d4, mask_hi2);
        const float scale = dd  * (float)sv;
        const float minv  = dmm * (float)mn;

        // repack this row's 32 weight nibbles into 8 dp4a uints. The packed q4_K
        // layout stores one ushort = 4 consecutive-K nibbles for a row at
        // src0_q[row + (K_group)*m], K_group = step/4 + u.
        const uint wbase = rrow + (step >> 2) * (uint)m;
        uint qw[8];
        qw[0] = EXP4(src0_q[wbase + 0 * m]);
        qw[1] = EXP4(src0_q[wbase + 1 * m]);
        qw[2] = EXP4(src0_q[wbase + 2 * m]);
        qw[3] = EXP4(src0_q[wbase + 3 * m]);
        qw[4] = EXP4(src0_q[wbase + 4 * m]);
        qw[5] = EXP4(src0_q[wbase + 5 * m]);
        qw[6] = EXP4(src0_q[wbase + 6 * m]);
        qw[7] = EXP4(src0_q[wbase + 7 * m]);

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

        #pragma unroll
        for (int t = 0; t < TILESIZE_N; ++t) {
            int raw = 0;
            raw = dot_acc_sat_4x8packed_ss_int(qw[0], sh_qa[t][0], raw);
            raw = dot_acc_sat_4x8packed_ss_int(qw[1], sh_qa[t][1], raw);
            raw = dot_acc_sat_4x8packed_ss_int(qw[2], sh_qa[t][2], raw);
            raw = dot_acc_sat_4x8packed_ss_int(qw[3], sh_qa[t][3], raw);
            raw = dot_acc_sat_4x8packed_ss_int(qw[4], sh_qa[t][4], raw);
            raw = dot_acc_sat_4x8packed_ss_int(qw[5], sh_qa[t][5], raw);
            raw = dot_acc_sat_4x8packed_ss_int(qw[6], sh_qa[t][6], raw);
            raw = dot_acc_sat_4x8packed_ss_int(qw[7], sh_qa[t][7], raw);
            acc[t] += scale * (float)sh_d[t] * (float)raw - minv * (float)sh_s[t];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (!row_valid) {
        return;
    }

    // dst is [token, feature] row-major (stride m): dst[col*m + row]
    #pragma unroll
    for (int t = 0; t < TILESIZE_N; ++t) {
        const uint c = col_base + t;
        if (c < (uint)n_no_padding) {
            dst[c * (uint)m + row] = acc[t];
        }
    }
}

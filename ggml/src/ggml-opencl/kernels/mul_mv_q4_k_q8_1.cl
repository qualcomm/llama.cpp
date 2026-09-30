// Q4_K decode GEMV with q8_1 (int8) activation — OpenCL analogue of SYCL's MMVQ
// (mul_mat_vec_q_reorder). Ports what SYCL does for the decode Q4_K matmul:
//   * activation pre-quantized to q8_1 (kernel_quant_a_q8_1: qa int8 element-order,
//     da = block scale, sa = da*Sum(qa));
//   * ONE output row per subgroup (N_DST=1) => nrows subgroups = MAX occupancy —
//     the point vs the f32 flat kernel (N_DST=16 => nrows/16 subgroups, which
//     starves small-m GEMVs like K/V-cur, m=1024 = 64 subgroups = ~10% of 672
//     Xe-LP HW threads);
//   * int8 dot with the Q4_K reassociation  Sum(w*a) = scale*a_d*dp4a - min*a_s,
//     scale = d_sb*sc6, min = dm_sb*mn6.
//
// WEIGHT LAYOUT = kernel_convert_block_q4_K_noshuffle output (cvt.cl:2048),
// ROW-MAJOR per superblock, un-shuffled so byte b of a superblock holds the two
// nibbles for ELEMENTS (2b, 2b+1) [low, high]. Hence a ushort at
//   q_ns[(row*nb + sb)*64 + sub*8 + u]
// EXP4's to the 4 CONTIGUOUS element nibbles {4t .. 4t+3}, t = sub*8+u, pairing
// 1:1 with the q8_1 activation uint qa[sb*64 + sub*8 + u]. d/dm/s are per
// superblock, row-major: d_ns[row*nb+sb], s_ns[(row*nb+sb)*12].

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#else
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

#ifdef cl_intel_required_subgroup_size
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#define REQD_SG __attribute__((intel_reqd_sub_group_size(Q8K_SG)))
#else
#define REQD_SG
#endif

#ifndef Q8K_SG
#define Q8K_SG 16
#endif

#define QK_K          256
#define K_SCALE_SIZE  12

inline void get_scale_min_k4(int j, __global const uchar * q, uchar * d, uchar * m) {
    if (j < 4) {
        *d = q[j]   & 0x3f;
        *m = q[j+4] & 0x3f;
    } else {
        *d = (q[j+4] & 0x0f) | ((q[j-4] & 0xc0) >> 2);
        *m = ((q[j+4] >> 4) & 0x0f) | ((q[j] & 0xc0) >> 2);
    }
}

// Expand 4 nibbles (low 16 bits of a ushort) to 4 bytes, one nibble (0..15) per byte.
#define EXP4(u)  ( ((uint)((u) & 0x000Fu))        | \
                  (((uint)((u) & 0x00F0u)) << 4)  | \
                  (((uint)((u) & 0x0F00u)) << 8)  | \
                  (((uint)((u) & 0xF000u)) << 12) )

// 4x int8 * 4x int8 -> int accumulate. Portable (Xe-LP maps int8 dot to the float
// ALU rate; the win here is occupancy + activation BW, not the op).
inline int dp4a4(uint w, uint a, int acc) {
    const char4 wv = as_char4(w);   // nibbles 0..15 (non-negative)
    const char4 av = as_char4(a);   // q8_1 int8 activation (signed)
    acc += (int)wv.s0 * (int)av.s0;
    acc += (int)wv.s1 * (int)av.s1;
    acc += (int)wv.s2 * (int)av.s2;
    acc += (int)wv.s3 * (int)av.s3;
    return acc;
}

// One subgroup (Q8K_SG lanes) per output row; lanes split the K sub-blocks and
// reduce at the end. n_q == 1 (decode).
REQD_SG
__kernel void kernel_mul_mv_q4_k_q8_1(
        __global const ushort * src0_q,   // q4_K nibbles (noshuffle row-major)
        __global const uchar  * src0_s,   // 6-bit scale/min codes (row-major, 12/superblock)
        __global const half   * src0_d,   // per-superblock scale  (row-major)
        __global const half   * src0_dm,  // per-superblock min    (row-major)
        __global const uint   * src1_qa,  // q8_1 int8 activation, 4/uint  [K/4]
        __global const half   * src1_da,  // q8_1 per-32-block scale       [K/32]
        __global const half   * src1_sa,  // q8_1 per-32-block sum*d        [K/32]
        __global       float  * dst,
        int    offsetd,
        int    m,                         // rows (ne01)
        int    k                          // ne00
) {
    dst = (__global float *)((__global char *)dst + offsetd);

    const int row = get_group_id(0);      // one subgroup per row
    if (row >= m) return;
    const int lane = get_sub_group_local_id();

    const int nb   = k / QK_K;            // superblocks per row
    const int k_b  = k >> 5;              // 32-blocks along K

    float partial = 0.0f;

    for (int blk = lane; blk < k_b; blk += Q8K_SG) {
        const int sb  = blk >> 3;         // superblock
        const int sub = blk & 7;          // subblock within it

        const int   sbrow = row * nb + sb;
        const float dd  = (float) src0_d [sbrow];
        const float dmm = (float) src0_dm[sbrow];
        __global const uchar * sc = src0_s + sbrow * K_SCALE_SIZE;
        uchar sv, mn;
        get_scale_min_k4(sub, sc, &sv, &mn);
        const float scale = dd  * (float) sv;
        const float minv  = dmm * (float) mn;

        const int wbase = sbrow * 64 + (sub << 3);   // ushort base for this subblock
        const int abase = sb * 64 + (sub << 3);      // activation uint base
        int idot = 0;
        idot = dp4a4(EXP4(src0_q[wbase + 0]), src1_qa[abase + 0], idot);
        idot = dp4a4(EXP4(src0_q[wbase + 1]), src1_qa[abase + 1], idot);
        idot = dp4a4(EXP4(src0_q[wbase + 2]), src1_qa[abase + 2], idot);
        idot = dp4a4(EXP4(src0_q[wbase + 3]), src1_qa[abase + 3], idot);
        idot = dp4a4(EXP4(src0_q[wbase + 4]), src1_qa[abase + 4], idot);
        idot = dp4a4(EXP4(src0_q[wbase + 5]), src1_qa[abase + 5], idot);
        idot = dp4a4(EXP4(src0_q[wbase + 6]), src1_qa[abase + 6], idot);
        idot = dp4a4(EXP4(src0_q[wbase + 7]), src1_qa[abase + 7], idot);

        const float a_d = (float) src1_da[blk];
        const float a_s = (float) src1_sa[blk];
        partial += scale * a_d * (float) idot - minv * a_s;
    }

    const float sum = sub_group_reduce_add(partial);
    if (lane == 0) {
        dst[row] = sum;
    }
}

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_qcom_kernel_cooperative_matrix : enable

// MoE mxfp4 x f32 GEMM with cl_qcom_kernel_cooperative_matrix, register-A / packed-B variant of
// kernel_gemm_moe_mxfp4_f32_ns_coopmat: m64n32k16, f16 in, f32 accumulate, one wave per 64 weight
// rows x one tile of 32 token slots.
// A (the transposed mxfp4 weight of kernel_convert_block_mxfp4_trans4_ns) is dequantized straight
// into the A operand, one row per lane, as kernel_gemm_noshuffle_q4_0_f32_coopmat does: no local
// memory and no barrier in the K loop. B is the f16 copy of kernel_moe_reorder_b_f16_packed, laid
// out as contiguous 16x32 column-major blocks, [tile][K/16][32][16], so each B load reads 1 KB.
// Keep this kernel alone in its program, as for the original.

typedef half  mat_a_t __attribute__((qcom_coop_mat(CLK_COOPERATIVE_MATRIX_SCOPE_SUBGROUP_QCOM, 64, 16, CLK_COOPERATIVE_MATRIX_A_QCOM)));
typedef half  mat_b_t __attribute__((qcom_coop_mat(CLK_COOPERATIVE_MATRIX_SCOPE_SUBGROUP_QCOM, 16, 32, CLK_COOPERATIVE_MATRIX_B_QCOM)));
typedef float mat_c_t __attribute__((qcom_coop_mat(CLK_COOPERATIVE_MATRIX_SCOPE_SUBGROUP_QCOM, 64, 32, CLK_COOPERATIVE_MATRIX_ACCUMULATOR_QCOM)));

typedef half half1024 __attribute__((ext_vector_type(1024)));

typedef union {
    half1024 v;
    half16   h;
} a_row_t;

// 8 E2M1 nibbles -> f16 in k order 0 4 1 5 2 6 3 7 (the packed B is written in the same order).
// The E2M1 value times 2^-14 for all codes; d holds the 2^14.
inline half8 mxfp4_to_f16x8(uint v, half d) {
    uint4 t;
    t.s0 = ((v << 9) & 0x0E000E00) | ((v << 12) & 0x80008000);
    t.s1 = ((v << 5) & 0x0E000E00) | ((v <<  8) & 0x80008000);
    t.s2 = ((v << 1) & 0x0E000E00) | ((v <<  4) & 0x80008000);
    t.s3 = ((v >> 3) & 0x0E000E00) | ( v        & 0x80008000);
    return as_half8(t) * d;
}

#define STORE_C(t, v) { \
        const uint idx = src2[col + (t)]; \
        if (idx != 0xFFFFFFFFu && wr) { \
            write_imagef(dst, (int)(idx * ne01 + row + lane), (v)); \
        } }

// One K step of 32: dequantize 64 rows x 32 k into two A operands and multiply by two B blocks.
#define STEP32(bq) { \
        uint4 q; \
        q.s0 = read_imageui(src0_q, (int)(qoff           )).x; \
        q.s1 = read_imageui(src0_q, (int)(qoff +     ne01)).x; \
        q.s2 = read_imageui(src0_q, (int)(qoff + 2 * ne01)).x; \
        q.s3 = read_imageui(src0_q, (int)(qoff + 3 * ne01)).x; \
        const uint e = src0_d[eoff]; \
        qoff += 4 * ne01; \
        eoff += ne01; \
        const half d = (half)(as_float(min(e, 128u) << 23) * 16384.0f); \
        a_row_t u0; \
        a_row_t u1; \
        u0.h = (half16)(mxfp4_to_f16x8(q.s0, d), mxfp4_to_f16x8(q.s1, d)); \
        u1.h = (half16)(mxfp4_to_f16x8(q.s2, d), mxfp4_to_f16x8(q.s3, d)); \
        if (e > 128) { \
            const half d2 = (half)as_float((e - 1) << 23); \
            u0.h *= d2; \
            u1.h *= d2; \
        } \
        mat_a_t a0 = __builtin_astype(u0.v, mat_a_t); \
        mat_a_t a1 = __builtin_astype(u1.v, mat_a_t); \
        mat_b_t b0 = qcom_coop_mat_load((bq),       CLK_COOPERATIVE_MATRIX_LAYOUT_COLUMN_MAJOR_QCOM, 16); \
        mat_b_t b1 = qcom_coop_mat_load((bq) + 512, CLK_COOPERATIVE_MATRIX_LAYOUT_COLUMN_MAJOR_QCOM, 16); \
        acc = qcom_coop_mat_mulAdd(a0, b0, acc, CLK_COOPERATIVE_MATRIX_OPERAND_NONE_QCOM); \
        acc = qcom_coop_mat_mulAdd(a1, b1, acc, CLK_COOPERATIVE_MATRIX_OPERAND_NONE_QCOM); }

// Same arguments as kernel_gemm_moe_mxfp4_f32_ns_coopmat, except src1 is the packed f16 buffer of
// kernel_moe_reorder_b_f16_packed.
// global size (64, ne01/64 rounded up, max tiles), local size (64, 1, 1)
__attribute__((reqd_work_group_size(64, 1, 1)))
kernel void kernel_gemm_moe_mxfp4_f32_ns_coopmat_rp(
        __read_only  image1d_buffer_t src0_q,    // A: [E][K/8][M] u32, 8 E2M1 in natural k order
        __global     uchar *          src0_d,    // A: [E][K/32][M] e8m0
        __global     const half *     src1,      // B: [tile][K/16][32][16] f16
        __global     uint *           src2,      // token slot -> dst row, 0xFFFFFFFF for padding
        __global     ushort *         src2_emap, // tile -> expert
        __write_only image1d_buffer_t dst,
        __global     int *            total_tiles,
        uint ne00,
        uint ne01,
        uint is_ragged,
        uint skip_gran
) {
    const uint lane = get_local_id(0);
    const uint row  = get_global_id(1) * 64;
    const uint tile = get_global_id(2);

    if (tile >= (uint)total_tiles[0]) {
        return;
    }

    const uint expert = src2_emap[tile];
    const uint col    = tile * 32;
    // rows past ne01 (ne01 % 64 == 32) load the last row and are not stored
    const uint rl     = min(row + lane, ne01 - 1);

    uint qoff = expert * (ne00 / 8)  * ne01 + rl;
    uint eoff = expert * (ne00 / 32) * ne01 + rl;
    global const half * bp = src1 + (ulong)tile * ne00 * 32;

    local float lc[64 * 32];     // C tile, row-major, ld 32

    mat_c_t acc = qcom_coop_mat_init(0.0f);

    uint k0 = 0;
    if ((ne00 / 32) & 1) {
        STEP32(bp)
        bp += 1024;
        k0 = 32;
    }
#pragma unroll 1
    for (; k0 < ne00; k0 += 64) {
        STEP32(bp)
        STEP32(bp + 1024)
        bp += 2048;
    }

    qcom_coop_mat_store(lc, acc, CLK_COOPERATIVE_MATRIX_LAYOUT_ROW_MAJOR_QCOM, 32);
    barrier(CLK_LOCAL_MEM_FENCE);

    const bool wr = row + lane < ne01;
    const local float * cr = lc + lane * 32;
#pragma unroll
    for (uint t = 0; t < 32; t++) {
        STORE_C(t, cr[t])
    }
}

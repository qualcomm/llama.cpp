// Eight-column q6_K GEMM with a dp4a inner product on the FLAT weight layout, ne1 = 5..8.
//
// Vocab-scale q6_K weights (the 151936 x 5120 lm_head of Qwen3.8-27B) are kept in the
// flat layout on Adreno because the batch-1 flat GEMV is the fastest head kernel there.
// The flat layout had no eight-column kernel: the DFlash2 verify width ran the
// four-column GEMV twice (18.6 ms, ~35 GB/s), slower than the CPU, which is what kept
// the head off the GPU. This kernel does that pass in one read of the weight: 6.6 ms
// at 151936 x 8 x 5120 on X2-90 (96 GB/s of the 152 the bus gives), level with the
// K-split noshuffle cok8 kernel, without moving the weight out of the flat layout.
//
// Flat layout, per row and 256-K superblock: 128 bytes of low nibbles, 64 bytes of high
// bit pairs, 16 int8 scales, one half. A row is 210 bytes per superblock, and rows are
// contiguous, so the only way to read it at bus speed is to spread the LANES OVER K:
//  - a wave covers four rows; its 16 K lanes per row cover COK_SBL consecutive
//    superblocks, 16/COK_SBL lanes each. A lane walks the 16-K scale groups of its
//    superblock; a group is 16 consecutive bytes of the nibble plane (one nibble half)
//    and 16 consecutive bytes of the high-bit plane (one bit pair), two uint4 loads.
//    Covering two superblocks per row at once makes each row's run 256 bytes and
//    keeps the number of rows a wave streams from at four, which is what the memory
//    system wants from this layout (one row per lane streams at 7 GB/s, one
//    superblock per 16 lanes at 82 GB/s, two at 96 GB/s; four is no better, and the
//    group walk must not be fully unrolled or the activation words spill).
//  - each lane keeps four rows, so the activation words of a group (eight columns,
//    32 uints, plain loads: they differ per lane so the texture broadcast of the
//    noshuffle kernels does not apply) are reused four times.
//  - the 6-bit code is used unsigned (0..63) in the dot; the -32 offset enters through
//    the accumulator start (-32 * sum of the group's activations, computed once per
//    group and reused for the four rows), so the words need no sign fix-up.
// Partial sums of the 16 K lanes meet through local memory at the end.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#ifdef cl_khr_integer_dot_product
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#endif

#ifndef COK_NSG
#define COK_NSG 2
#endif
#ifndef COK_SBL
#define COK_SBL 2           // superblocks per row covered by one wave at a time
#endif
#define COK_SG   64
#define COK_ROWS 4                      // rows per lane
#define COK_RL   4                      // row lanes per wave
#define COK_KL   16                     // K lanes per row
#define COK_KG   (COK_KL / COK_SBL)     // lanes per superblock; each walks 16/COK_KG groups
#define SG_ROWS  (COK_ROWS * COK_RL)    // rows per subgroup

#define COK_PACK4(a, b, c, e)                                         \
    ( ((uint)(a) & 0xFFu) | (((uint)(b) & 0xFFu) << 8)               \
    | (((uint)(c) & 0xFFu) << 16) | (((uint)(e) & 0xFFu) << 24) )

// q8_1 activation quantization, natural K order. One work-item per 32-K block; column c,
// block b is global block c*k_b + b. ds holds the eight column scales of each block as one
// half8, [b]; the host allocates at width 8.
kernel void kernel_quant_a_q8_1_k8n(
    global const float * src,     // [N * K] f32, K contiguous per column
    global uint        * qa,      // [N * K / 4] int8 x4, natural order
    global half        * ds,      // [K/32 * 8] block scale, width 8
    int total_blocks,             // N * (K/32)
    int k_b                       // K/32
) {
    const int blk = get_global_id(0);
    if (blk >= total_blocks) {
        return;
    }

    const int base = blk * 32;

    float v[32];
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        v[i] = src[base + i];
        amax = fmax(amax, fabs(v[i]));
    }

    const float d  = amax / 127.0f;
    const float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;

    int q[32];
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        q[i] = (int)rint(v[i] * id);
    }

    uint8 out;
    out.s0 = COK_PACK4(q[ 0], q[ 1], q[ 2], q[ 3]);
    out.s1 = COK_PACK4(q[ 4], q[ 5], q[ 6], q[ 7]);
    out.s2 = COK_PACK4(q[ 8], q[ 9], q[10], q[11]);
    out.s3 = COK_PACK4(q[12], q[13], q[14], q[15]);
    out.s4 = COK_PACK4(q[16], q[17], q[18], q[19]);
    out.s5 = COK_PACK4(q[20], q[21], q[22], q[23]);
    out.s6 = COK_PACK4(q[24], q[25], q[26], q[27]);
    out.s7 = COK_PACK4(q[28], q[29], q[30], q[31]);
    vstore8(out, 0, qa + blk * 8);

    const int col = blk / k_b;
    const int b   = blk - col * k_b;
    ds[b * 8 + col] = (half)d;
}

// Word i of a 16-K group: nibble half nsh of the low plane, bit pair qsh of the high
// plane, as four unsigned 6-bit codes.
#define COK_WORD(ql_, qh_) ((((ql_) >> nsh) & 0x0F0F0F0Fu) | ((((qh_) >> qsh) & 0x03030303u) << 4))

// Dot of the four words with column c, starting from -32 * (sum of the column's 16 activations).
#define COK_DOT_COL(c)                                                             \
    dots.s##c = dot_acc_sat_4x8packed_ss_int(w0, a[c].s0, off.s##c);               \
    dots.s##c = dot_acc_sat_4x8packed_ss_int(w1, a[c].s1, dots.s##c);              \
    dots.s##c = dot_acc_sat_4x8packed_ss_int(w2, a[c].s2, dots.s##c);              \
    dots.s##c = dot_acc_sat_4x8packed_ss_int(w3, a[c].s3, dots.s##c);

#define COK_SUM_COL(c)                                                             \
    off.s##c = dot_acc_sat_4x8packed_ss_int(0x01010101u, a[c].s0, 0);              \
    off.s##c = dot_acc_sat_4x8packed_ss_int(0x01010101u, a[c].s1, off.s##c);       \
    off.s##c = dot_acc_sat_4x8packed_ss_int(0x01010101u, a[c].s2, off.s##c);       \
    off.s##c = dot_acc_sat_4x8packed_ss_int(0x01010101u, a[c].s3, off.s##c);       \
    off.s##c = -32 * off.s##c;

kernel void kernel_gemm_cok8_q6_k_flat_q8_1_dp4a(
    global const uchar * src0_ql,     // [row][sb][128] low nibbles
    global const uchar * src0_qh,     // [row][sb][64] high bit pairs
    global const char  * src0_s,      // [row][sb][16] int8 scales, one per 16 K
    global const half  * src0_d,      // [row][sb] superblock scale
    global const uint4 * src1_qa,     // q8_1 activations, natural order, [col][K/16]
    global const half  * src1_ds,     // [K/32][8] the eight column scales of the block
    global float * dst,               // [n][m]
    ulong offsetd,
    int m,
    int k,
    int n_no_padding
) {
    dst = (global float *)((global char *)dst + offsetd);

    const int lane = get_local_id(0);
    const int sg   = get_local_id(1);
    const int rl   = lane >> 4;                 // row lane 0..3
    const int kl16 = lane & 15;                 // K lane of the row
    const int sbl  = kl16 / COK_KG;             // superblock lane
    const int kl   = kl16 - sbl * COK_KG;       // lane within the superblock

    const int nb  = k >> 8;             // superblocks per row
    const int k16 = k >> 4;             // uint4 per activation column

    // m % 16 == 0; a subgroup past the last row group recomputes it and skips the store.
    const int  sgi  = get_group_id(0) * COK_NSG + sg;
    const bool live = sgi * SG_ROWS < m;
    const int  row0 = (live ? sgi * SG_ROWS : m - SG_ROWS) + rl * COK_ROWS;

    // Superblock index of (row0, 0); row r, superblock ib is sb0 + r*nb + ib. Kept as one
    // index rather than four pointers per plane to spare registers.
    const long sb0 = (long)row0 * nb;

    float8 acc[COK_ROWS];
    #pragma unroll
    for (int r = 0; r < COK_ROWS; ++r) {
        acc[r] = (float8)(0.0f);
    }

    for (int ibq = 0; ibq < nb; ibq += COK_SBL) {
        // This lane's superblock; past the row's end it recomputes the last one for nothing
        // (the scale is multiplied by 0, not selected: a select turns the scale loads into
        // a branch and the kernel runs 55% slower).
        const int   sbi = ibq + sbl;
        const bool  ok  = sbi < nb;
        const int   sbc = ok ? sbi : nb - 1;
        const float okf = ok ? 1.0f : 0.0f;

        #pragma unroll 2
        for (int t = 0; t < 16 / COK_KG; ++t) {
            // Group g of the superblock: half ip (128 K), 32-K segment j of the half,
            // 16-K offset l0 in the segment. Low plane byte 64*ip + 32*(j&1) + l0 + i
            // holds K i of segment j (nibble j>>1); high plane byte 32*ip + l0 + i holds
            // bit pair j of the same K. Consecutive lanes take consecutive 16-byte chunks.
            const int  g   = kl + COK_KG * t;
            const int  ip  = g >> 3;
            const int  j   = (g & 7) >> 1;
            const int  l0  = (g & 1) << 4;
            const uint nsh = (uint)(j >> 1) << 2;
            const uint qsh = (uint)j << 1;
            const int  off_ql = (ip << 6) + ((j & 1) << 5) + l0;
            const int  off_qh = (ip << 5) + l0;

            // The group's 16 K of every column, the block scales of those K, and the
            // -32 offsets.
            uint4 a[8];
            #pragma unroll
            for (int c = 0; c < 8; ++c) {
                a[c] = src1_qa[c * k16 + (sbc << 4) + g];
            }
            const float8 da = convert_float8(vload8(0, src1_ds + (((sbc << 3) + (g >> 1)) << 3)));

            int8 off;
            COK_SUM_COL(0) COK_SUM_COL(1) COK_SUM_COL(2) COK_SUM_COL(3)
            COK_SUM_COL(4) COK_SUM_COL(5) COK_SUM_COL(6) COK_SUM_COL(7)

            #pragma unroll
            for (int r = 0; r < COK_ROWS; ++r) {
                const long  sb = sb0 + (long)r * nb + sbc;
                const uint4 ql = vload4(0, (global const uint *)(src0_ql + sb * 128 + off_ql));
                const uint4 qh = vload4(0, (global const uint *)(src0_qh + sb * 64 + off_qh));
                const uint w0 = COK_WORD(ql.s0, qh.s0);
                const uint w1 = COK_WORD(ql.s1, qh.s1);
                const uint w2 = COK_WORD(ql.s2, qh.s2);
                const uint w3 = COK_WORD(ql.s3, qh.s3);

                int8 dots;
                COK_DOT_COL(0) COK_DOT_COL(1) COK_DOT_COL(2) COK_DOT_COL(3)
                COK_DOT_COL(4) COK_DOT_COL(5) COK_DOT_COL(6) COK_DOT_COL(7)

                const float sw = (float)src0_s[sb * 16 + g] * (float)src0_d[sb] * okf;
                acc[r] = mad(convert_float8(dots), da * sw, acc[r]);
            }
        }
    }

    // Sum the 16 K lanes of each row lane, one row per round through local memory
    // ([lane][8 columns]); lanes 0..31 then each own one (row lane, column).
    local float8 red[COK_NSG][COK_SG];
    const int o_rl = (lane >> 3) & 3;   // row lane
    const int o_c  = lane & 7;          // column

    #pragma unroll
    for (int r = 0; r < COK_ROWS; ++r) {
        barrier(CLK_LOCAL_MEM_FENCE);
        red[sg][lane] = acc[r];
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lane < COK_RL * 8) {
            local const float * src = (local const float *)&red[sg][o_rl * COK_KL] + o_c;
            float sum = 0.0f;
            #pragma unroll
            for (int q = 0; q < COK_KL; ++q) {
                sum += src[q * 8];
            }
            if (live && o_c < n_no_padding) {
                const int row = (row0 - rl * COK_ROWS) + o_rl * COK_ROWS + r;
                dst[o_c * m + row] = sum;
            }
        }
    }
}

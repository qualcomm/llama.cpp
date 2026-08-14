#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#else
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

#ifdef cl_intel_required_subgroup_size
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#define INTEL_GPU 1
#define REQD_SUBGROUP_SIZE_16 __attribute__((intel_reqd_sub_group_size(16)))
#define REQD_SUBGROUP_SIZE_32 __attribute__((intel_reqd_sub_group_size(32)))
#elif defined(cl_qcom_reqd_sub_group_size)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#endif

//------------------------------------------------------------------------------
// kernel_mul_mv_q6_K_f32_flat_mc
//
// Multi-column twin of kernel_mul_mv_q6_K_f32_flat.
//
// The single-column kernel takes the output column from get_group_id(1), so a
// matmul with ne1 columns launches ne1 column-tiles and EVERY tile re-streams
// the whole weight matrix. That is invisible for a per-layer weight (it stays in
// cache) but it is the dominant cost for a vocab-scale lm_head: gemma-4-26B's
// head is 0.606 GB, so a 4-wide speculative-decode verify batch reads 2.4 GB
// where the CPU GEMM reads 0.606 GB once -- which is why the GPU head measured
// neutral at ne1 == 1 and cost ~22% of spec decode at ne1 == 4.
//
// This variant loads each weight block once and dots it against N_COLS
// activation columns, so weight traffic is 1x for any ne1 <= N_COLS. Registers
// grow as N_DST*N_COLS accumulators + 4*N_COLS cached activation vectors, and
// that block -- not the weight stream -- is what limits this kernel: N_DST must
// come down as N_COLS goes up or the tile spills and loses to the single-column
// kernel outright. The host builds it twice (2 columns x 8 rows, 4 columns x 4
// rows) and picks by ne1; see the measured table at the build site. Both are
// -D overridable (Q6K_MC_N_DST / Q6K_MC_N_COLS).
//
// Arithmetic per (row, column) is identical to the single-column kernel --
// same block unpack, same scale fold, same dot()/reduce order -- so results are
// bit-identical to the flat GEMV, not an approximation.
//------------------------------------------------------------------------------
#define Q6_K_MASK1 0x03
#define Q6_K_MASK2 0x0C
#define Q6_K_MASK3 0x30
#define Q6_K_MASK4 0xC0

#define QK_K       256

#undef N_DST
#undef N_SIMDGROUP
#undef N_SIMDWIDTH

#ifndef Q6K_MC_N_COLS
#define Q6K_MC_N_COLS 4
#endif

// Where the column tile's activations live across the row loop.
//   1: cached in registers, loaded once per superblock and reused by all N_DST
//      rows. Fewest load instructions, but costs 16*N_COLS registers -- at
//      N_COLS=4 that is 64, MORE than the N_DST*N_COLS accumulator block, and it
//      is what forces N_DST down to 4.
//   0: re-read from global inside the row loop. Frees those 64 registers so
//      N_DST can rise; the working set is one superblock x N_COLS = 4 KB, so the
//      re-reads are L1 hits, but they are still N_DST times the load issue.
// Discriminates occupancy-bound from issue-bound: if the kernel is register
// starved, 0 with a larger N_DST wins.
#ifndef Q6K_MC_Y_CACHE
#define Q6K_MC_Y_CACHE 1
#endif

#ifdef INTEL_GPU
#ifndef Q6K_MC_N_DST
#define Q6K_MC_N_DST 4
#endif
#define N_SIMDGROUP 2
#define N_SIMDWIDTH 16
#elif defined (ADRENO_GPU)
#ifndef Q6K_MC_N_DST
#define Q6K_MC_N_DST 8
#endif
#define N_SIMDGROUP 2
#define N_SIMDWIDTH 64
#else
#ifndef Q6K_MC_N_DST
#define Q6K_MC_N_DST 8
#endif
#define N_SIMDGROUP 2
#define N_SIMDWIDTH 32
#endif

#define N_DST  Q6K_MC_N_DST
#define N_COLS Q6K_MC_N_COLS

#define BLOCK_STRIDE (N_SIMDWIDTH/16) // number of blocks each subgroup processes

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_16
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mv_q6_K_f32_flat_mc(
        global uchar * src0_ql,
        global uchar * src0_qh,
        global char  * src0_s,
        global half  * src0_d,
        global float * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne10,
        int ne12,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src1 = (global float*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    int nb = ne00/QK_K;

    int r0 = get_group_id(0);
    int r1 = get_group_id(1);
    int im = get_group_id(2);

    int i12 = im%ne12;
    int i13 = im/ne12;

    int first_row = (N_SIMDGROUP * r0 + get_sub_group_id()) * N_DST;
    int first_col = r1 * N_COLS;

    ulong offset_src0    = first_row*nb + (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);
    ulong offset_src0_ql = offset_src0 * 128;
    ulong offset_src0_qh = offset_src0 * 64;
    ulong offset_src0_s  = offset_src0 * 16;
    ulong offset_src0_d  = offset_src0;

    global uchar * blk_ql     = (global uchar *) src0_ql + offset_src0_ql;
    global uchar * blk_qh     = (global uchar *) src0_qh + offset_src0_qh;
    global char  * blk_scales = (global char  *) src0_s  + offset_src0_s;
    global half  * blk_d      = (global half  *) src0_d  + offset_src0_d;
    global float * yy         = (global float *) src1    + im*ne00*ne1;

    int tid = get_sub_group_local_id()%(N_SIMDWIDTH/BLOCK_STRIDE); // within-super-block part, 0..15
    int ix  = get_sub_group_local_id()/(N_SIMDWIDTH/BLOCK_STRIDE); // super-block selector, 0..BLOCK_STRIDE-1
    int ip  = tid/8;   // first or second half of (super) block (0 or 1)
    int il  = tid%8;   // each half has 8 parts, one per scale
    int n   = 4;       // 4 scales at a time (and 4 sums)
    int l0  = n*il;    // offset into half-block, 0..28

    // A tail tile (ne1 not a multiple of N_COLS) reads a clamped, in-range column
    // for its dead lanes -- the work is wasted but never out of bounds -- and the
    // store below drops it. Keeping the column index in a compile-time-sized array
    // is what lets the accumulator block stay in registers.
    int ycol[N_COLS];
    for (int c = 0; c < N_COLS; c++) {
        ycol[c] = min(first_col + c, ne1 - 1);
    }

    int q_offset_l = 64*ip + l0;
    int q_offset_h = 32*ip + l0;
    int sj         = l0/16;   // 0 or 1; the scale pair selector inside the half-block

    float sumf[N_DST][N_COLS];
    for (int row = 0; row < N_DST; row++) {
        for (int c = 0; c < N_COLS; c++) {
            sumf[row][c] = 0.f;
        }
    }

    for (int ib = ix; ib < nb; ib += BLOCK_STRIDE) {
#if Q6K_MC_Y_CACHE
        // Activations for all N_COLS columns of this super-block, loaded once and
        // reused across the N_DST rows below. This is the reuse the single-column
        // kernel cannot express.
        float4 y0[N_COLS], y1[N_COLS], y2[N_COLS], y3[N_COLS];
        for (int c = 0; c < N_COLS; c++) {
            global float * y = yy + (ulong)ycol[c]*ne10 + ib*QK_K + 128*ip + l0;
            y0[c] = vload4(0, y +  0);
            y1[c] = vload4(0, y + 32);
            y2[c] = vload4(0, y + 64);
            y3[c] = vload4(0, y + 96);
        }
#endif

        for (int row = 0; row < N_DST; row++) {
            if (first_row + row < ne01) {
                global uchar * q1 = blk_ql + row*nb*128 + ib*128 + q_offset_l;
                global uchar * q2 = q1 + QK_K/8;
                global uchar * qh = blk_qh + row*nb*64  + ib*64  + q_offset_h;

                // One aligned 8-byte load for the four scales this lane needs
                // (is, is+2, is+4, is+6 all live in the half-block at 8*ip).
                char8 scv = vload8(0, blk_scales + row*nb*16 + ib*16 + 8*ip);

                float dall = blk_d[row*nb + ib];

                uchar4 q1v = vload4(0, q1);
                uchar4 q2v = vload4(0, q2);
                uchar4 qhv = vload4(0, qh);

                int4 q1i = convert_int4(q1v);
                int4 q2i = convert_int4(q2v);
                int4 qhi = convert_int4(qhv);

                float4 w0 = convert_float4((q1i & 0xF) | ((qhi & Q6_K_MASK1) << 4)) - 32.f;
                float4 w1 = convert_float4((q2i & 0xF) | ((qhi & Q6_K_MASK2) << 2)) - 32.f;
                float4 w2 = convert_float4((q1i >> 4)  | ((qhi & Q6_K_MASK3)     )) - 32.f;
                float4 w3 = convert_float4((q2i >> 4)  | ((qhi & Q6_K_MASK4) >> 2)) - 32.f;

                char4 s4 = (sj == 0) ? (char4)(scv.s0, scv.s2, scv.s4, scv.s6)
                                     : (char4)(scv.s1, scv.s3, scv.s5, scv.s7);

                for (int c = 0; c < N_COLS; c++) {
#if Q6K_MC_Y_CACHE
                    const float4 yc0 = y0[c], yc1 = y1[c], yc2 = y2[c], yc3 = y3[c];
#else
                    global float * y = yy + (ulong)ycol[c]*ne10 + ib*QK_K + 128*ip + l0;
                    const float4 yc0 = vload4(0, y +  0);
                    const float4 yc1 = vload4(0, y + 32);
                    const float4 yc2 = vload4(0, y + 64);
                    const float4 yc3 = vload4(0, y + 96);
#endif
                    sumf[row][c] += dall * (dot(yc0, w0) * s4.x + dot(yc1, w1) * s4.y +
                                            dot(yc2, w2) * s4.z + dot(yc3, w3) * s4.w);
                }
            }
        }
    }

    for (int row = 0; row < N_DST; row++) {
        for (int c = 0; c < N_COLS; c++) {
            float tot = sub_group_reduce_add(sumf[row][c]);
            if (get_sub_group_local_id() == 0 &&
                first_row + row < ne01 && first_col + c < ne1) {
                dst[(first_col + c)*ne0 + im*ne0*ne1 + first_row + row] = tot;
            }
        }
    }
}

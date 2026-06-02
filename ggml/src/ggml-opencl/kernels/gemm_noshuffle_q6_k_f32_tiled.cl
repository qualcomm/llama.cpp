// Batched (N>1) q6_K GEMM over the 64-row-TILED canonical layout produced by
// kernel_convert_block_q6_k_tiled_ns (cvt.cl). Companion to the decode kernel
// kernel_gemv_noshuffle_q6_K_f32_tiled: SAME pack, SAME canonical e-order
// dequant (correct by construction vs reference ggml q6_K), just extended to N
// output columns. This makes the batched lm_head/embed (perplexity, spec-decode
// verify, batched serving) correct on GPU instead of falling back to CPU, while
// keeping the tiled convert that the fast decode GEMV depends on.
//
// One work-item owns one output ROW for a block of BN columns. A work-group is
// {64 lanes, 4 subgroups}: the 64 lanes cover the 64 rows of one tile (coalesced
// weight loads), the 4 subgroups split the K-superblocks and reduce through
// __local at the end. The global z dimension tiles the N columns by BN.
//
// Weights are read from __global (coalesced), matching the decode kernel: the
// lm_head weight is streamed with little reuse, where coalesced global beats the
// Adreno texture cache.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
#endif

#define NSUBGROUPS 4
#define TILE_ROWS  64
#define BN         8        // output columns handled per work-group (global z step)

#if defined(ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_gemm_noshuffle_q6_K_f32_tiled(
    __global uint4 * src0_ql,   // tiled: 8 uint4 granules / superblock
    __global uint4 * src0_qh,   // tiled: 4 uint4 granules / superblock
    __global char  * src0_s,    // tiled: 16 chars / superblock
    __global half  * src0_d,    // tiled: 1 half  / superblock
    read_only image1d_buffer_t src1,   // activation [ne00, ne11] f32 (RGBA), column-major
    global float * dst,
    ulong offsetd,
    int ne00,
    int ne01,
    int ne11
) {
    int grp = get_local_id(1);          // subgroup index 0..3 (splits K)
    int row = get_global_id(0);         // output row along ne01
    int rt  = row / TILE_ROWS;
    int rit = row % TILE_ROWS;
    int col0 = get_global_id(2) * BN;   // first output column of this block

    int nb = ne00 / 256;                // superblocks per row
    int act_col_stride = ne00 / 4;      // activation float4 pixels per column

    float acc[BN];
    #pragma unroll
    for (int j = 0; j < BN; ++j) acc[j] = 0.0f;

    for (int sb = grp; sb < nb; sb += NSUBGROUPS) {
        int tile_blk = rt * nb + sb;    // ne02 == 1 for lm_head/embed

        float dval = (float)src0_d[tile_blk * TILE_ROWS + rit];
        __global char * sc = src0_s + (tile_blk * TILE_ROWS + rit) * 16;

        uint ql[32];
        uint qh[16];
        #pragma unroll
        for (int g = 0; g < 8; ++g) {
            uint4 v = src0_ql[(tile_blk * 8 + g) * TILE_ROWS + rit];
            ql[g*4+0] = v.x; ql[g*4+1] = v.y; ql[g*4+2] = v.z; ql[g*4+3] = v.w;
        }
        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            uint4 v = src0_qh[(tile_blk * 4 + g) * TILE_ROWS + rit];
            qh[g*4+0] = v.x; qh[g*4+1] = v.y; qh[g*4+2] = v.z; qh[g*4+3] = v.w;
        }

        int act_e_base = sb * 64;       // activation float4 pixel base within a column
        // NOTE: the e4 loop is deliberately NOT unrolled. Fully unrolling 64 * 4 * BN
        // MACs plus 64 * BN image reads overflows the in-process Adreno compiler
        // (host stack overflow at clBuildProgram), the same failure class as the
        // FA DK=512 OOM. The inner t/j loops (small) stay unrolled.
        for (int e4 = 0; e4 < 64; ++e4) {
            // preload BN columns' activation float4 for this e4 (shared by the 4 codes)
            float4 av[BN];
            #pragma unroll
            for (int j = 0; j < BN; ++j) {
                int c = col0 + j;
                av[j] = (c < ne11)
                    ? read_imagef(src1, c * act_col_stride + act_e_base + e4)
                    : (float4)(0.0f);
            }
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                int  e    = e4 * 4 + t;
                uint low4 = (ql[e >> 3] >> ((e & 7) * 4)) & 0xF;
                uint hi2  = (qh[e >> 4] >> ((e & 15) * 2)) & 0x3;
                int  code = (int)(low4 | (hi2 << 4)) - 32;
                int  sidx = ((e >> 7) << 3) + (((e >> 5) & 3) << 1) + ((e >> 4) & 1);
                float cs  = (float)code * (float)sc[sidx] * dval;
                #pragma unroll
                for (int j = 0; j < BN; ++j) {
                    float avt = (t == 0) ? av[j].x : (t == 1) ? av[j].y : (t == 2) ? av[j].z : av[j].w;
                    acc[j] += cs * avt;
                }
            }
        }
    }

    // reduce across the NSUBGROUPS subgroups (same rit, different K-subset), per column
    local float reduce_lm[NSUBGROUPS * TILE_ROWS * BN];
    #pragma unroll
    for (int j = 0; j < BN; ++j) {
        reduce_lm[(grp * TILE_ROWS + rit) * BN + j] = acc[j];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (grp == 0) {
        dst = (global float*)((global char*)dst + offsetd);
        #pragma unroll
        for (int j = 0; j < BN; ++j) {
            int c = col0 + j;
            if (c < ne11) {
                float total = reduce_lm[(0 * TILE_ROWS + rit) * BN + j]
                            + reduce_lm[(1 * TILE_ROWS + rit) * BN + j]
                            + reduce_lm[(2 * TILE_ROWS + rit) * BN + j]
                            + reduce_lm[(3 * TILE_ROWS + rit) * BN + j];
                dst[(ulong)c * ne01 + row] = total;
            }
        }
    }
}

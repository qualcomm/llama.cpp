// Tiled-wide q6_K GEMV for the long-vocab lm_head/embed (decode path).
//
// Pairs with kernel_convert_block_q6_k_tiled_ns (cvt.cl): the weights are laid
// out CANONICALLY (6-bit code in element order e in [0,256)) and TILED by 64
// output rows so the 64-thread lane group coalesces every weight load. Both the
// pack (convert) and the unpack (here) are owned by us — correct by construction
// against the reference ggml q6_K dequant, no bit-interleave reverse-engineering.
//
// One work-item produces one output row. A work-group is {64 lanes, 4 subgroups}:
// the 64 lanes cover the 64 rows of one tile (coalesced reads), the 4 subgroups
// split the K-blocks and reduce through __local at the end.
//
// Weights are read from __global (coalesced) rather than image1d_buffer: the
// lm_head is read once per token with no reuse, and the Adreno texture cache
// caps such a streaming read well below the coalesced-global rate
// (see opencl_q6k_gemv_o4_shipped / x2-90 roofline notes).

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
#endif

#define NSUBGROUPS 4
#define TILE_ROWS  64

#if defined(ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_gemv_noshuffle_q6_K_f32_tiled(
    __global uint4 * src0_ql,   // tiled: 8 uint4 granules / superblock
    __global uint4 * src0_qh,   // tiled: 4 uint4 granules / superblock
    __global char  * src0_s,    // tiled: 16 chars / superblock
    __global half  * src0_d,    // tiled: 1 half  / superblock
    read_only image1d_buffer_t src1,   // activation (RGBA f32)
    global float * dst,
    ulong offsetd,
    int ne00,
    int ne01
) {
    int grp = get_local_id(1);          // subgroup index 0..3 (splits K)
    int row = get_global_id(0);         // output row along ne01
    int rt  = row / TILE_ROWS;
    int rit = row % TILE_ROWS;

    int nb = ne00 / 256;                // superblocks per row

    float acc = 0.0f;

    for (int sb = grp; sb < nb; sb += NSUBGROUPS) {
        int tile_blk = rt * nb + sb;    // ne02 == 1 for lm_head/embed

        // d + 16 scales for this (row, superblock)
        float dval = (float)src0_d[tile_blk * TILE_ROWS + rit];
        __global char * sc = src0_s + (tile_blk * TILE_ROWS + rit) * 16;

        // 32 ql-uints (8 codes/uint) + 16 qh-uints (16 codes/uint)
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

        // dequant 256 codes in canonical e-order, MAC with activation.
        int act_base = sb * 64;         // activation float4 pixel base (256/4)
        #pragma unroll
        for (int e4 = 0; e4 < 64; ++e4) {
            float4 a = read_imagef(src1, act_base + e4);
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                int  e    = e4 * 4 + t;
                uint low4 = (ql[e >> 3] >> ((e & 7) * 4)) & 0xF;
                uint hi2  = (qh[e >> 4] >> ((e & 15) * 2)) & 0x3;
                int  code = (int)(low4 | (hi2 << 4)) - 32;
                int  sidx = ((e >> 7) << 3) + (((e >> 5) & 3) << 1) + ((e >> 4) & 1);
                float scale = (float)sc[sidx] * dval;
                float av = (t == 0) ? a.x : (t == 1) ? a.y : (t == 2) ? a.z : a.w;
                acc += (float)code * scale * av;
            }
        }
    }

    // reduce across the NSUBGROUPS subgroups (same rit, different K-subset)
    local float reduce_lm[NSUBGROUPS * TILE_ROWS];
    reduce_lm[grp * TILE_ROWS + rit] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (grp == 0) {
        float total = reduce_lm[0 * TILE_ROWS + rit]
                    + reduce_lm[1 * TILE_ROWS + rit]
                    + reduce_lm[2 * TILE_ROWS + rit]
                    + reduce_lm[3 * TILE_ROWS + rit];
        dst = (global float*)((global char*)dst + offsetd);
        dst[row] = total;
    }
}

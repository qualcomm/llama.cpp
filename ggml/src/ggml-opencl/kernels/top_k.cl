// GGML_OP_TOP_K -- the k largest elements of each row, as column indices.
//
// The op contract does NOT specify output order (the CPU reference deliberately
// swaps dst[0] and dst[1] "to emphasize that the order is not important"), so we
// only have to produce the correct SET. That rules out needing to sort the row.
//
// Rows can be vocabulary-wide (202048 on muse-glimmer), far past what the
// single-workgroup bitonic argsort can take, so wide rows use a two-stage
// tournament:
//
//   stage 1 (here)  one workgroup per (row, tile); emit the tile's own top k
//   stage 2         the existing argsort over the ntiles*k candidates
//   stage 3 (here)  map the ranked candidate positions back to columns
//
// Correctness: a global top-k element has at most k-1 elements larger than it,
// hence at most k-1 inside its own tile, so keeping k per tile cannot drop a
// winner.
//
// No subgroup builtins are used anywhere in this file. Some Adreno compilers
// fail (-6) or crash when a program defines any kernel AFTER one that calls a
// subgroup builtin, and this program defines two.

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#endif

// Items each lane scans. TILE = BLOCK * ITEMS_PER_LANE. Set by the host so that
// ntiles*k fits the workgroup the argsort in stage 2 can actually launch.
#ifndef ITEMS_PER_LANE
#define ITEMS_PER_LANE 16
#endif

// Order-preserving map from a float's bits to an unsigned key: flipping the sign
// bit for positives and inverting everything for negatives makes unsigned
// compare agree with float compare. Packing (ncols-1-col) underneath makes a
// plain max() break ties toward the LOWEST column, deterministically.
inline ulong topk_pack(float v, int col, int ncols) {
    uint b = as_uint(v);
    b = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
    return ((ulong)b << 32) | (ulong)(uint)(ncols - 1 - col);
}

kernel void kernel_top_k_tile(
        global const float * src0,
        ulong                offset0,
        global float       * cand_val,
        ulong                offset_val,
        global int         * cand_idx,
        ulong                offset_idx,
        int                  ncols,
        int                  ntiles,
        int                  k,
        local  ulong       * smem) {
    src0     = (global const float *)((global char *)src0     + offset0);
    cand_val = (global float       *)((global char *)cand_val + offset_val);
    cand_idx = (global int         *)((global char *)cand_idx + offset_idx);

    const int lid   = get_local_id(0);
    const int bsz   = get_local_size(0);
    const int row   = get_group_id(0) / ntiles;
    const int tile  = get_group_id(0) % ntiles;

    global const float * row_ptr = src0 + (size_t)row * ncols;

    const int tile_w = bsz * ITEMS_PER_LANE;
    const int base   = tile * tile_w;

    // One pass over the tile, held in registers; the k rounds below re-reduce
    // these instead of re-reading global memory.
    ulong keys[ITEMS_PER_LANE];
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_LANE; ++i) {
        const int col = base + lid + i * bsz;
        keys[i] = (col < ncols) ? topk_pack(row_ptr[col], col, ncols) : 0UL;
    }

    const size_t out = ((size_t)row * ntiles + tile) * k;

    for (int j = 0; j < k; ++j) {
        ulong local_best = 0UL;
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_LANE; ++i) {
            local_best = max(local_best, keys[i]);
        }

        smem[lid] = local_best;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int s = bsz / 2; s > 0; s >>= 1) {
            if (lid < s) {
                smem[lid] = max(smem[lid], smem[lid + s]);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        const ulong best = smem[0];
        if (lid == 0) {
            const int col = ncols - 1 - (int)(uint)(best & 0xFFFFFFFFUL);
            // A tile can be short of k live entries only when the row itself is
            // short; -INFINITY keeps those out of the stage-2 ranking.
            cand_val[out + j] = (best != 0UL) ? row_ptr[col] : -INFINITY;
            cand_idx[out + j] = (best != 0UL) ? col : 0;
        }

        // Retire the winner so the next round finds the next one.
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_LANE; ++i) {
            if (keys[i] == best) {
                keys[i] = 0UL;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

// stage 3: argsort ranked the candidate VALUES and gave back positions within
// the candidate array; turn the first k of those back into source columns.
kernel void kernel_top_k_unmap(
        global const int * cand_idx,
        ulong              offset_idx,
        global const int * order,
        ulong              offset_order,
        global int       * dst,
        ulong              offsetd,
        int                ncand,
        int                k) {
    cand_idx = (global const int *)((global char *)cand_idx + offset_idx);
    order    = (global const int *)((global char *)order    + offset_order);
    dst      = (global int       *)((global char *)dst      + offsetd);

    const int row = get_group_id(0);
    for (int i = get_local_id(0); i < k; i += get_local_size(0)) {
        dst[(size_t)row * k + i] = cand_idx[(size_t)row * ncand + order[(size_t)row * ncand + i]];
    }
}

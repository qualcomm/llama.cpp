#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#ifdef cl_khr_integer_dot_product
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#endif

// int8 KQV for the decomposed prefill attention path.
//
//   out[d, q, h] = sum_kv  V^T[d, kv, h] * P[kv, q, h]
//
// Both operands are ACTIVATIONS, not weights, so both are quantised at runtime by the passes that
// already write them: P by kernel_soft_max_4_f16_q8 (u8, P = exp(x-max) is non-negative so it gets
// 255 levels), V^T by flash_attn_v_transpose_q8 (s8). Scales are per 32 elements along kv, which
// is the contiguous axis of both, so a 32-block is 8 uints and feeds dp4a directly.
//
// Both are symmetric (no zero point), so the epilogue is just d_v * d_p * raw - there is no q8_1
// sum term to subtract, unlike the MoE GEMM this borrows its inner loop from.
//
// Motivation, measured on Qwen3.5-35B at d16384: this kernel's f16 predecessor runs at
// 1.13 TFLOP/s while dp4a GEMMs on the same device reach 5.1 TOP/s at comparable shapes, and KQV
// is 29.7% of device time there. The normalisation stays deferred - kernel_fa_scale_rows_f32
// still applies 1/sum to the (far smaller) output.

#define KQV_TM 32   // d rows per workgroup
#define KQV_TN 16   // queries per workgroup
#define KQV_WG 64
#define KQV_NB 8   // P blocks staged per barrier

__attribute__((reqd_work_group_size(KQV_WG, 1, 1)))
kernel void kernel_mul_mm_q8_kqv(
        global const uint  * vq,        // V^T int8, [head_kv][dv][n_kv]        as packed uints
        ulong                off_vq,
        global const half  * vd,        // V^T scales, [head_kv][dv][n_kv/32]
        ulong                off_vd,
        global const uint  * pq,        // P u8,     [head][n_q][n_kv]          as packed uints
        ulong                off_pq,
        global const half  * pd,        // P scales, [head][n_q][n_kv/32]
        ulong                off_pd,
        global       float * dst,       // [head][n_q][dv]
        ulong                off_dst,
        int                  dv,
        int                  n_kv,
        int                  n_q,
        int                  n_head,
        int                  n_head_kv
) {
    vq  = (global const uint  *)((global const char *)vq  + off_vq);
    vd  = (global const half  *)((global const char *)vd  + off_vd);
    pq  = (global const uint  *)((global const char *)pq  + off_pq);
    pd  = (global const half  *)((global const char *)pd  + off_pd);
    dst = (global       float *)((global const char *)dst + off_dst);

    const int tid  = get_local_id(0);
    const int dm0  = get_group_id(0) * KQV_TM;      // first d row of this tile
    const int qn0  = get_group_id(1) * KQV_TN;      // first query of this tile
    const int head = get_group_id(2);

    const int head_kv = n_head_kv > 0 ? head % n_head_kv : 0;
    const int nblk    = n_kv / 32;                  // dispatch guarantees n_kv % 32 == 0
    const int nu      = n_kv / 4;                   // uints per row

    // lane layout: 32 consecutive d rows x 2 query groups of 8
    const int dl  = tid & (KQV_TM - 1);             // 0..31 -> d row
    const int qg  = tid / KQV_TM;                   // 0 or 1 -> which half of the queries
    const int d   = dm0 + dl;

    // P is staged KQV_NB blocks at a time. One block per barrier put two barriers around only
    // 64 dp4a instructions, and at n_kv=16384 that is 1024 barriers per workgroup - the first
    // cut measured exactly at parity with the f16 kernel because it was synchronisation bound,
    // not math bound. Eight blocks per barrier raises the work between them to 512 instructions.
    __local uint sh_pq[KQV_TN][KQV_NB][8];
    __local half sh_pd[KQV_TN][KQV_NB];

    float acc[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        acc[j] = 0.0f;
    }

    const size_t vbase = ((size_t)head_kv*dv + d)*nu;
    const size_t vdbas = ((size_t)head_kv*dv + d)*nblk;

    for (int bg = 0; bg < nblk; bg += KQV_NB) {
        const int nb_here = min(KQV_NB, nblk - bg);

        for (int i = tid; i < KQV_TN*KQV_NB*8; i += KQV_WG) {
            const int qq = i >> 6;            // KQV_NB*8 == 64 per query
            const int bb = (i >> 3) & (KQV_NB - 1);
            const int uu = i & 7;
            const int qi = qn0 + qq;
            sh_pq[qq][bb][uu] = (qi < n_q && bb < nb_here)
                ? pq[((size_t)head*n_q + qi)*nu + (size_t)(bg + bb)*8 + uu] : 0u;
        }
        for (int i = tid; i < KQV_TN*KQV_NB; i += KQV_WG) {
            const int qq = i / KQV_NB;
            const int bb = i % KQV_NB;
            const int qi = qn0 + qq;
            sh_pd[qq][bb] = (qi < n_q && bb < nb_here)
                ? pd[((size_t)head*n_q + qi)*nblk + bg + bb] : (half)0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (d < dv) {
            for (int bb = 0; bb < nb_here; ++bb) {
                uint w[8];
                #pragma unroll
                for (int u = 0; u < 8; ++u) {
                    w[u] = vq[vbase + (size_t)(bg + bb)*8 + u];
                }
                const float dvs = (float)vd[vdbas + bg + bb];

                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int qq = qg*8 + j;

                    int raw = 0;
                    #pragma unroll
                    for (int u = 0; u < 8; ++u) {
                        raw = dot_acc_sat_4x8packed_us_int(sh_pq[qq][bb][u], w[u], raw);
                    }

                    acc[j] += dvs * (float)sh_pd[qq][bb] * (float)raw;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (d >= dv) {
        return;
    }

    // dst is [head][n_q][dv] with dv contiguous, so the 32 lanes of a d-run write contiguously
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int qi = qn0 + qg*8 + j;
        if (qi < n_q) {
            dst[((size_t)head*n_q + qi)*dv + d] = acc[j];
        }
    }
}

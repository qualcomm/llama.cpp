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
// sum term to subtract, unlike the MoE GEMM this borrows its tile from.
//
// Tile: the shipped kernel_gemm_moe_q8_1_dp4a shape. One lane owns one d row and all KQV_TN
// queries of the tile, so every V load from global is reused KQV_TN times from registers and P
// is read from local memory. The first cut (32 x 16, two lanes per d row) streamed V^T once per
// 16 queries: at n_kv 32768 that is 4 GB per 512-query chunk, the same time at 150 GB/s as the
// dp4a math itself, and the kernel captured about a fifth of dp4a's measured multiplier.
//
// Dispatch order (host and kernel are a matched pair, see the host): d-block fastest, then the
// query heads that share one KV head, then the query tile, then the KV head. The heads of one
// GQA group read the same V^T slice, so they hit it in L2 while it is hot.

#define KQV_TM 64   // d rows per workgroup, one per lane
#ifndef KQV_TN
#define KQV_TN 32   // queries per workgroup, all owned by every lane
#endif
#ifndef KQV_NB
#define KQV_NB 4    // 32-blocks of P staged per barrier
#endif
#define KQV_WG 64
#ifndef KQV_WAVE_PAIR
#define KQV_WAVE_PAIR 1
#endif
#ifndef KQV_LDS_VEC
#define KQV_LDS_VEC 1   // 1: 16-byte loads from local memory, 0: scalar
#endif
#ifndef KQV_GLB_VEC
#define KQV_GLB_VEC 1   // 1: 16-byte loads of V from global, 0: scalar
#endif

#if KQV_WAVE_PAIR
__attribute__((qcom_wave_pair_mode(1)))
#else
__attribute__((reqd_work_group_size(KQV_WG, 1, 1)))
#endif
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

    const int lid  = get_local_id(0);
    const int gsz  = n_head / n_head_kv;            // query heads per KV head
    const int g1   = get_group_id(1);
    const int head_kv = get_group_id(2);
    const int head = head_kv*gsz + (g1 % gsz);
    const int qn0  = (g1 / gsz) * KQV_TN;           // first query of this tile
    const int d    = get_group_id(0)*KQV_TM + lid;  // this lane's d row

    const int nblk = n_kv / 32;                     // dispatch guarantees n_kv % 32 == 0
    const int nu   = n_kv / 4;                      // uints per row

    __local uint sh_pq[KQV_TN][KQV_NB][8];
    __local half sh_pd[KQV_TN][KQV_NB];

    float acc[KQV_TN];
    #pragma unroll
    for (int t = 0; t < KQV_TN; ++t) {
        acc[t] = 0.0f;
    }

    const int    dl    = min(d, dv - 1);            // clamp so a tail lane loads in bounds
    const size_t vbase = ((size_t)head_kv*dv + dl)*nu;
    const size_t vdbas = ((size_t)head_kv*dv + dl)*nblk;
    const size_t pbase = (size_t)head*n_q;

    for (int bg = 0; bg < nblk; bg += KQV_NB) {
        const int nb_here = min(KQV_NB, nblk - bg);

        // stage P: each (query, block) is 8 uints = two 16-byte loads
        for (int i = lid; i < KQV_TN*KQV_NB*2; i += KQV_WG) {
            const int t  = i / (KQV_NB*2);
            const int bb = (i / 2) % KQV_NB;
            const int h  = (i & 1) * 4;
            const int qi = qn0 + t;
            uint4 v = (uint4)(0u);
            if (qi < n_q && bb < nb_here) {
                v = vload4(0, &pq[(pbase + qi)*nu + (size_t)(bg + bb)*8 + h]);
            }
            vstore4(v, 0, &sh_pq[t][bb][h]);
        }
        for (int i = lid; i < KQV_TN*KQV_NB; i += KQV_WG) {
            const int t  = i / KQV_NB;
            const int bb = i % KQV_NB;
            const int qi = qn0 + t;
            sh_pd[t][bb] = (qi < n_q && bb < nb_here)
                ? pd[(pbase + qi)*nblk + bg + bb] : (half)0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int bb = 0; bb < nb_here; ++bb) {
#if KQV_GLB_VEC
            const uint4 w0 = vload4(0, &vq[vbase + (size_t)(bg + bb)*8]);
            const uint4 w1 = vload4(0, &vq[vbase + (size_t)(bg + bb)*8 + 4]);
#else
            const size_t vo = vbase + (size_t)(bg + bb)*8;
            const uint4 w0 = (uint4)(vq[vo+0], vq[vo+1], vq[vo+2], vq[vo+3]);
            const uint4 w1 = (uint4)(vq[vo+4], vq[vo+5], vq[vo+6], vq[vo+7]);
#endif
            const float dvs = (float)vd[vdbas + bg + bb];

            #pragma unroll
            for (int t = 0; t < KQV_TN; ++t) {
#if KQV_LDS_VEC
                const uint4 a0 = vload4(0, &sh_pq[t][bb][0]);
                const uint4 a1 = vload4(0, &sh_pq[t][bb][4]);
#else
                const uint4 a0 = (uint4)(sh_pq[t][bb][0], sh_pq[t][bb][1], sh_pq[t][bb][2], sh_pq[t][bb][3]);
                const uint4 a1 = (uint4)(sh_pq[t][bb][4], sh_pq[t][bb][5], sh_pq[t][bb][6], sh_pq[t][bb][7]);
#endif
                int raw = 0;
                raw = dot_acc_sat_4x8packed_us_int(a0.s0, w0.s0, raw);
                raw = dot_acc_sat_4x8packed_us_int(a0.s1, w0.s1, raw);
                raw = dot_acc_sat_4x8packed_us_int(a0.s2, w0.s2, raw);
                raw = dot_acc_sat_4x8packed_us_int(a0.s3, w0.s3, raw);
                raw = dot_acc_sat_4x8packed_us_int(a1.s0, w1.s0, raw);
                raw = dot_acc_sat_4x8packed_us_int(a1.s1, w1.s1, raw);
                raw = dot_acc_sat_4x8packed_us_int(a1.s2, w1.s2, raw);
                raw = dot_acc_sat_4x8packed_us_int(a1.s3, w1.s3, raw);
                acc[t] += dvs * (float)sh_pd[t][bb] * (float)raw;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (d >= dv) {
        return;
    }

    // dst is [head][n_q][dv] with dv contiguous, so the 64 lanes of a d-block write contiguously
    #pragma unroll
    for (int t = 0; t < KQV_TN; ++t) {
        const int qi = qn0 + t;
        if (qi < n_q) {
            dst[(pbase + qi)*dv + d] = acc[t];
        }
    }
}

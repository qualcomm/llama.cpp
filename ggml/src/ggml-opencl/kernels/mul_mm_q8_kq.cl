#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#ifdef cl_khr_integer_dot_product
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#endif

// int8 KQ for the decomposed prefill attention path.
//
//   out[kv, q, h] = sum_d  K[d, kv, h_kv] * Q[d, q, h]
//
// K stays f16 in the cache and Q f32 in the graph; both are quantised here per prefill call to
// int8 with one half scale per 32 elements along d. An 8-bit K is indistinguishable from f16 on
// long-context retrieval (measured on the quantised KV cache), and Q gets the same treatment the
// q8_1 activations of every dp4a weight GEMM get. Both are symmetric, so the epilogue is
// d_k * d_q * raw per block pair.
//
// The contraction is short (dk = 256 is 8 blocks) and the output is the largest tensor in the
// attention, so the tile is sized for the output: one lane owns one kv row and KQ_TN queries,
// the whole Q tile is staged in local memory once, and there are no per-block barriers. A
// workgroup then walks KQ_MB consecutive 64-row kv blocks with that one staged tile: with a
// single block the staging and the launch cost more than the 8-block contraction they serve.
//
// Dispatch order (host and kernel are a matched pair): query tile fastest across the heads of a
// GQA group, then the kv block, then the KV head. A 64-row K block is 16 KB and is reused from
// L2 by every query tile of its group before the next block is touched.
//
// The packed dot takes the K word (from global, in registers) FIRST and the Q word (from local
// memory) second; see mul_mm_q8_kqv.cl for why the order matters on the Adreno compiler.

#ifndef KQ_TN
#define KQ_TN 32        // queries per workgroup
#endif
#ifndef KQ_MB
#define KQ_MB 8         // 64-row kv blocks per workgroup, sharing one staged Q tile
#endif
#define KQ_TM 64        // kv rows per block, one per lane
#define KQ_WG 64
#define KQ_DK_MAX 256   // local memory is sized for this; the host declines larger heads

// ---- row quantisation: [nhead][nrow][ne0] -> int8 + per-32 half scale --------------------
// One lane per 32-block. src rows are strided (nb1 bytes per row, nb2 per head) so a KV-cache
// view is read in place; the outputs are dense.

kernel void kernel_fa_q8_rows_f16(
        global const char * src, ulong off_src, ulong nb1, ulong nb2,
        global char * dq, global half * dd,
        int ne0, int nrow, int nhead
) {
    const int nblk = ne0 / 32;
    const int gid  = get_global_id(0);
    if (gid >= nblk*nrow*nhead) {
        return;
    }
    const int b = gid % nblk;
    const int r = (gid / nblk) % nrow;
    const int h = gid / (nblk*nrow);

    global const half * p = (global const half *)(src + off_src + (ulong)h*nb2 + (ulong)r*nb1) + b*32;
    float v[32];
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float4 x = convert_float4(vload4(i, p));
        v[i*4+0] = x.s0; v[i*4+1] = x.s1; v[i*4+2] = x.s2; v[i*4+3] = x.s3;
        amax = fmax(amax, fmax(fmax(fabs(x.s0), fabs(x.s1)), fmax(fabs(x.s2), fabs(x.s3))));
    }
    const float d  = amax / 127.0f;
    const float id = amax > 0.0f ? 127.0f / amax : 0.0f;
    const size_t row = (size_t)h*nrow + r;
    dd[row*nblk + b] = (half)d;
    global char * q = dq + row*ne0 + b*32;
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        q[i] = (char)clamp((int)rint(v[i]*id), -127, 127);
    }
}

kernel void kernel_fa_q8_rows_f32(
        global const char * src, ulong off_src, ulong nb1, ulong nb2,
        global char * dq, global half * dd,
        int ne0, int nrow, int nhead
) {
    const int nblk = ne0 / 32;
    const int gid  = get_global_id(0);
    if (gid >= nblk*nrow*nhead) {
        return;
    }
    const int b = gid % nblk;
    const int r = (gid / nblk) % nrow;
    const int h = gid / (nblk*nrow);

    global const float * p = (global const float *)(src + off_src + (ulong)h*nb2 + (ulong)r*nb1) + b*32;
    float v[32];
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float4 x = vload4(i, p);
        v[i*4+0] = x.s0; v[i*4+1] = x.s1; v[i*4+2] = x.s2; v[i*4+3] = x.s3;
        amax = fmax(amax, fmax(fmax(fabs(x.s0), fabs(x.s1)), fmax(fabs(x.s2), fabs(x.s3))));
    }
    const float d  = amax / 127.0f;
    const float id = amax > 0.0f ? 127.0f / amax : 0.0f;
    const size_t row = (size_t)h*nrow + r;
    dd[row*nblk + b] = (half)d;
    global char * q = dq + row*ne0 + b*32;
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        q[i] = (char)clamp((int)rint(v[i]*id), -127, 127);
    }
}

// ---- the GEMM -----------------------------------------------------------------------------

__attribute__((qcom_wave_pair_mode(1)))
kernel void kernel_mul_mm_q8_kq(
        global const uint  * kq,        // K int8, [head_kv][n_kv][dk]   as packed uints
        global const half  * kd,        // K scales, [head_kv][n_kv][dk/32]
        global const uint  * qq,        // Q int8, [head][n_q][dk]       as packed uints
        global const half  * qd,        // Q scales, [head][n_q][dk/32]
        global       float * dst,       // [head][n_q][n_kv]
        ulong                off_dst,
        int                  dk,
        int                  n_kv,
        int                  n_q,
        int                  n_head,
        int                  n_head_kv
) {
    dst = (global float *)((global char *)dst + off_dst);

    const int lid  = get_local_id(0);
    const int gsz  = n_head / n_head_kv;
    const int g0   = get_group_id(0);
    const int head_kv = get_group_id(2);
    const int head = head_kv*gsz + (g0 % gsz);
    const int qn0  = (g0 / gsz) * KQ_TN;
    const int kv0  = get_group_id(1)*KQ_TM*KQ_MB + lid;

    const int nblk = dk / 32;
    const int nu   = dk / 4;

    __local uint sh_q [KQ_TN][KQ_DK_MAX/4];
    __local half sh_qd[KQ_TN][KQ_DK_MAX/32];

    const size_t qbase = (size_t)head*n_q;
    for (int i = lid; i < KQ_TN*(KQ_DK_MAX/16); i += KQ_WG) {
        const int t = i / (KQ_DK_MAX/16);
        const int u = (i % (KQ_DK_MAX/16)) * 4;
        const int qi = qn0 + t;
        uint4 v = (uint4)(0u);
        if (qi < n_q && u < nu) {
            v = vload4(0, &qq[(qbase + qi)*nu + u]);
        }
        vstore4(v, 0, &sh_q[t][u]);
    }
    for (int i = lid; i < KQ_TN*(KQ_DK_MAX/32); i += KQ_WG) {
        const int t = i / (KQ_DK_MAX/32);
        const int b = i % (KQ_DK_MAX/32);
        const int qi = qn0 + t;
        sh_qd[t][b] = (qi < n_q && b < nblk) ? qd[(qbase + qi)*nblk + b] : (half)0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int m = 0; m < KQ_MB; ++m) {
        const int kv = kv0 + m*KQ_TM;
        if (kv - lid >= n_kv) {
            break;                                  // whole block past the end, uniform per workgroup
        }

        float acc[KQ_TN];
        #pragma unroll
        for (int t = 0; t < KQ_TN; ++t) {
            acc[t] = 0.0f;
        }

        const int    kl    = min(kv, n_kv - 1);     // clamp so a tail lane loads in bounds
        const size_t kbase = ((size_t)head_kv*n_kv + kl)*nu;
        const size_t kdbas = ((size_t)head_kv*n_kv + kl)*nblk;

        for (int b = 0; b < nblk; ++b) {
            const uint4 w0 = vload4(0, &kq[kbase + b*8]);
            const uint4 w1 = vload4(0, &kq[kbase + b*8 + 4]);
            const float dks = (float)kd[kdbas + b];

            #pragma unroll
            for (int t = 0; t < KQ_TN; ++t) {
                const uint4 a0 = vload4(0, &sh_q[t][b*8]);
                const uint4 a1 = vload4(0, &sh_q[t][b*8 + 4]);
                int raw = 0;
                raw = dot_acc_sat_4x8packed_ss_int(w0.s0, a0.s0, raw);
                raw = dot_acc_sat_4x8packed_ss_int(w0.s1, a0.s1, raw);
                raw = dot_acc_sat_4x8packed_ss_int(w0.s2, a0.s2, raw);
                raw = dot_acc_sat_4x8packed_ss_int(w0.s3, a0.s3, raw);
                raw = dot_acc_sat_4x8packed_ss_int(w1.s0, a1.s0, raw);
                raw = dot_acc_sat_4x8packed_ss_int(w1.s1, a1.s1, raw);
                raw = dot_acc_sat_4x8packed_ss_int(w1.s2, a1.s2, raw);
                raw = dot_acc_sat_4x8packed_ss_int(w1.s3, a1.s3, raw);
                acc[t] += dks * (float)sh_qd[t][b] * (float)raw;
            }
        }

        if (kv < n_kv) {
            // dst is [head][n_q][n_kv] with n_kv contiguous, so the 64 lanes of a kv block write contiguously
            #pragma unroll
            for (int t = 0; t < KQ_TN; ++t) {
                const int qi = qn0 + t;
                if (qi < n_q) {
                    dst[(qbase + qi)*n_kv + kv] = acc[t];
                }
            }
        }
    }
}

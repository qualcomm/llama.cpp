#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Lightning indexer (DeepSeek-V3.2 DSA, glm5-next, qwen4exp QSA).
// dst[ik, t, 0, s] = sum_h relu(q[:, h, t, s] . k[:, 0, ik, s]) * w[h, t, s] + mask[ik, t, 0, s % nem3]
// Reference: ggml-cpu/ops.cpp ggml_compute_forward_lightning_indexer.
//
// A workgroup of LI_WG lanes scores LI_TK = LI_WG*LI_RK K rows against LI_TQ tokens, four heads at a time.
// K and q go through local memory in slices of LI_DC elements, so neighbouring lanes load neighbouring
// bytes of a K row; each lane then keeps LI_RK rows x LI_TQ tokens x 4 heads of dot products in registers,
// so one q load feeds LI_RK dot products and one K load feeds 4*LI_TQ.
// Built without fast math: the mask carries -inf.

#ifndef LI_TQ
#define LI_TQ 8
#endif

#define LI_WG 64
#define LI_RK 1
#define LI_TK (LI_WG*LI_RK)
#define LI_DC 32            // K elements per slice; n_embd must be a multiple
#define LI_DC4 (LI_DC/4)
#define LI_NH 4             // heads per pass
#define LI_PAD 1            // float4 of padding per K row against bank conflicts

#ifdef LI_K_F16
#define LOAD_K4(p, i) vload_half4((i), (global const half *)(p))
#else
#define LOAD_K4(p, i) vload4((i), (global const float *)(p))
#endif

kernel __attribute__((reqd_work_group_size(LI_WG, 1, 1)))
void kernel_lightning_indexer(
        global const char * q,   ulong offq,
        global const char * k,   ulong offk,
        global const char * w,   ulong offw,
        global const char * m,   ulong offm,
        global       char * dst, ulong offd,
        int   n_embd,
        int   n_head,
        int   n_tokens,
        int   n_kv,
        int   nem3,
        ulong nbq1, ulong nbq2, ulong nbq3,
        ulong nbk2, ulong nbk3,
        ulong nbw1, ulong nbw3,
        ulong nbm1, ulong nbm3,
        ulong nb1,  ulong nb3
) {
    local float4 lk[LI_TK][LI_DC4 + LI_PAD];
    local float4 lq[LI_TQ][LI_NH][LI_DC4];

    const int lid = get_local_id(0);
    const int ik0 = get_group_id(0)*LI_TK;
    const int t0  = get_group_id(1)*LI_TQ;
    const int s   = get_group_id(2);

    k += offk + s*nbk3;
    q += offq + s*nbq3;
    w += offw + s*nbw3;

    float score[LI_RK][LI_TQ];
    for (int r = 0; r < LI_RK; ++r) {
        for (int j = 0; j < LI_TQ; ++j) {
            score[r][j] = 0.0f;
        }
    }

    for (int h0 = 0; h0 < n_head; h0 += LI_NH) {
        float acc[LI_RK][LI_NH][LI_TQ];
        for (int r = 0; r < LI_RK; ++r) {
            for (int h = 0; h < LI_NH; ++h) {
                for (int j = 0; j < LI_TQ; ++j) {
                    acc[r][h][j] = 0.0f;
                }
            }
        }

        for (int d0 = 0; d0 < n_embd; d0 += LI_DC) {
            barrier(CLK_LOCAL_MEM_FENCE);

            // K slice: LI_DC4 consecutive lanes cover one row, rows past the end re-read the last row
            for (int e = lid; e < LI_TK*LI_DC4; e += LI_WG) {
                const int row = e / LI_DC4;
                const int c   = e % LI_DC4;
                const int ik  = min(ik0 + row, n_kv - 1);
                lk[row][c] = convert_float4(LOAD_K4(k + ik*nbk2, d0/4 + c));
            }
            // q slice; heads past n_head are zero, tokens past the end re-read the last token
            for (int e = lid; e < LI_TQ*LI_NH*LI_DC4; e += LI_WG) {
                const int c = e % LI_DC4;
                const int h = (e / LI_DC4) % LI_NH;
                const int j = e / (LI_DC4*LI_NH);
                const int t = min(t0 + j, n_tokens - 1);
                lq[j][h][c] = h0 + h < n_head
                    ? vload4(d0/4 + c, (global const float *)(q + (h0 + h)*nbq1 + t*nbq2))
                    : (float4)(0.0f);
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            for (int c = 0; c < LI_DC4; ++c) {
                float4 kv[LI_RK];
                for (int r = 0; r < LI_RK; ++r) {
                    kv[r] = lk[lid + r*LI_WG][c];
                }
                for (int j = 0; j < LI_TQ; ++j) {
                    for (int h = 0; h < LI_NH; ++h) {
                        const float4 qv = lq[j][h][c];
                        for (int r = 0; r < LI_RK; ++r) {
                            float a = acc[r][h][j];
                            a = mad(qv.s0, kv[r].s0, a);
                            a = mad(qv.s1, kv[r].s1, a);
                            a = mad(qv.s2, kv[r].s2, a);
                            a = mad(qv.s3, kv[r].s3, a);
                            acc[r][h][j] = a;
                        }
                    }
                }
            }
        }

        for (int j = 0; j < LI_TQ; ++j) {
            const int t = min(t0 + j, n_tokens - 1);
            for (int h = 0; h < LI_NH; ++h) {
                if (h0 + h < n_head) {
                    const float wv = *(global const float *)(w + (h0 + h)*sizeof(float) + t*nbw1);
                    for (int r = 0; r < LI_RK; ++r) {
                        score[r][j] = mad(fmax(acc[r][h][j], 0.0f), wv, score[r][j]);
                    }
                }
            }
        }
    }

    m   += offm + (s % nem3)*nbm3;
    dst += offd + s*nb3;
    for (int j = 0; j < LI_TQ; ++j) {
        const int t = t0 + j;
        if (t >= n_tokens) {
            break;
        }
        for (int r = 0; r < LI_RK; ++r) {
            const int ik = ik0 + lid + r*LI_WG;
            if (ik < n_kv) {
                const float mv = vload_half(ik, (global const half *)(m + t*nbm1));
                ((global float *)(dst + t*nb1))[ik] = score[r][j] + mv;
            }
        }
    }
}

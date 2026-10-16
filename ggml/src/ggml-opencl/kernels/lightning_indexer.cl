#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Lightning indexer (DeepSeek-V3.2 DSA, glm5-next, qwen4exp QSA).
// dst[ik, t, 0, s] = sum_h relu(q[:, h, t, s] . k[:, 0, ik, s]) * w[h, t, s] + mask[ik, t, 0, s % nem3]
// Reference: ggml-cpu/ops.cpp ggml_compute_forward_lightning_indexer.
//
// One work item per K row and LI_TQ tokens. Every lane of a workgroup reads the same q and w
// elements, so those loads are broadcasts; each lane streams its own K row once per head.
// Built without fast math: the mask carries -inf.

#ifndef LI_TQ
#define LI_TQ 4
#endif

#ifdef LI_K_F16
#define K_TYPE  half
#define LOAD_K4(p, i) vload_half4((i), (p))
#else
#define K_TYPE  float
#define LOAD_K4(p, i) vload4((i), (p))
#endif

kernel void kernel_lightning_indexer(
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
    const int ik = get_global_id(0);
    const int t0 = get_global_id(1)*LI_TQ;
    const int s  = get_global_id(2);
    if (ik >= n_kv) {
        return;
    }

    global const K_TYPE * k_row = (global const K_TYPE *)(k + offk + ik*nbk2 + s*nbk3);
    q += offq + s*nbq3;
    w += offw + s*nbw3;

    // tokens past the end re-read the last token; their results are not stored
    int tt[LI_TQ];
    for (int j = 0; j < LI_TQ; ++j) {
        tt[j] = min(t0 + j, n_tokens - 1);
    }

    float score[LI_TQ];
    for (int j = 0; j < LI_TQ; ++j) {
        score[j] = 0.0f;
    }

    const int n4 = n_embd/4;

    for (int h = 0; h < n_head; ++h) {
        float acc[LI_TQ];
        for (int j = 0; j < LI_TQ; ++j) {
            acc[j] = 0.0f;
        }

        for (int i = 0; i < n4; ++i) {
            const float4 kv = convert_float4(LOAD_K4(k_row, i));
            for (int j = 0; j < LI_TQ; ++j) {
                global const float * q_row = (global const float *)(q + h*nbq1 + tt[j]*nbq2);
                acc[j] += dot(vload4(i, q_row), kv);
            }
        }

        for (int j = 0; j < LI_TQ; ++j) {
            const float wv = *(global const float *)(w + h*sizeof(float) + tt[j]*nbw1);
            score[j] += fmax(acc[j], 0.0f) * wv;
        }
    }

    m   += offm + (s % nem3)*nbm3;
    dst += offd + s*nb3;
    for (int j = 0; j < LI_TQ; ++j) {
        const int t = t0 + j;
        if (t < n_tokens) {
            const float mv = vload_half(ik, (global const half *)(m + t*nbm1));
            ((global float *)(dst + t*nb1))[ik] = score[j] + mv;
        }
    }
}

// Hyper-connection ops (DeepSeek-V4, qwen4exp, glm5-next).
// Reference: ggml-cpu/ops.cpp ggml_compute_forward_dsv4_hc_{comb,pre,post}_f32.

#define HC_COMB 4

static inline void hc_comb_norm_cols(float * comb, float eps) {
    for (int idst = 0; idst < HC_COMB; ++idst) {
        float sum = eps;
        for (int isrc = 0; isrc < HC_COMB; ++isrc) {
            sum += comb[idst + HC_COMB*isrc];
        }
        const float inv_sum = 1.0f / sum;
        for (int isrc = 0; isrc < HC_COMB; ++isrc) {
            comb[idst + HC_COMB*isrc] *= inv_sum;
        }
    }
}

static inline void hc_comb_norm_rows(float * comb, float eps) {
    for (int isrc = 0; isrc < HC_COMB; ++isrc) {
        float sum = eps;
        for (int idst = 0; idst < HC_COMB; ++idst) {
            sum += comb[idst + HC_COMB*isrc];
        }
        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < HC_COMB; ++idst) {
            comb[idst + HC_COMB*isrc] *= inv_sum;
        }
    }
}

// one work item per token: softmax over each source row, then Sinkhorn normalisation
kernel void kernel_dsv4_hc_comb(
        global const char * mixes, ulong offm,
        global const char * scale, ulong offs,
        global const char * base,  ulong offb,
        global       char * dst,   ulong offd,
        int   n_tokens,
        ulong nbm0, ulong nbm1,
        ulong nbs0,
        ulong nbb0,
        ulong nbd0, ulong nbd1, ulong nbd2,
        float eps,
        int   n_iter
) {
    const int it = get_global_id(0);
    if (it >= n_tokens) {
        return;
    }

    mixes += offm;
    scale += offs;
    base  += offb;
    dst   += offd;

    const int comb_offset = 2*HC_COMB;
    const float scale_comb = *(global const float *)(scale + 2*nbs0);

    float comb[HC_COMB*HC_COMB];

    for (int isrc = 0; isrc < HC_COMB; ++isrc) {
        float m = -INFINITY;
        for (int idst = 0; idst < HC_COMB; ++idst) {
            const int idx = idst + HC_COMB*isrc;
            const float xv = *(global const float *)(mixes + (comb_offset + idx)*nbm0 + it*nbm1);
            const float bv = *(global const float *)(base  + (comb_offset + idx)*nbb0);
            const float v  = xv*scale_comb + bv;
            comb[idx] = v;
            m = fmax(m, v);
        }

        float sum = 0.0f;
        for (int idst = 0; idst < HC_COMB; ++idst) {
            const int idx = idst + HC_COMB*isrc;
            const float v = exp(comb[idx] - m);
            comb[idx] = v;
            sum += v;
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < HC_COMB; ++idst) {
            const int idx = idst + HC_COMB*isrc;
            comb[idx] = comb[idx]*inv_sum + eps;
        }
    }

    hc_comb_norm_cols(comb, eps);
    for (int i = 1; i < n_iter; ++i) {
        hc_comb_norm_rows(comb, eps);
        hc_comb_norm_cols(comb, eps);
    }

    for (int isrc = 0; isrc < HC_COMB; ++isrc) {
        for (int idst = 0; idst < HC_COMB; ++idst) {
            *(global float *)(dst + idst*nbd0 + isrc*nbd1 + it*nbd2) = comb[idst + HC_COMB*isrc];
        }
    }
}

// dst[i0, t] = scale * sum_h x[i0, h, t] * w, w = sigmoid(gate[i0, h, t]) if gated else weights[h, t]
kernel void kernel_dsv4_hc_pre(
        global const char * x,   ulong offx,
        global const char * w,   ulong offw,
        global       char * dst, ulong offd,
        int   n_embd,
        int   hc,
        ulong nbx0, ulong nbx1, ulong nbx2,
        ulong nbw0, ulong nbw1, ulong nbw2,
        ulong nbd0, ulong nbd1,
        float scale,
        int   gated
) {
    const int i0 = get_global_id(0);
    const int it = get_global_id(1);
    if (i0 >= n_embd) {
        return;
    }

    x   += offx + i0*nbx0 + it*nbx2;
    w   += offw;
    dst += offd;

    float sum = 0.0f;
    if (gated) {
        w += i0*nbw0 + it*nbw2;
        for (int ih = 0; ih < hc; ++ih) {
            const float xv = *(global const float *)(x + ih*nbx1);
            const float gv = *(global const float *)(w + ih*nbw1);
            sum += xv * (1.0f / (1.0f + exp(-gv)));
        }
    } else {
        w += it*nbw1;
        for (int ih = 0; ih < hc; ++ih) {
            const float xv = *(global const float *)(x + ih*nbx1);
            const float wv = *(global const float *)(w + ih*nbw0);
            sum += xv * wv;
        }
    }

    *(global float *)(dst + i0*nbd0 + it*nbd1) = scale * sum;
}

// dst[i0, d, t] = x[i0, t] * post[d, t] + (comb ? sum_s residual[i0, s, t] * comb[d, s, t] : residual[i0, d, t])
kernel void kernel_dsv4_hc_post(
        global const char * x,    ulong offx,
        global const char * res,  ulong offr,
        global const char * post, ulong offp,
        global const char * comb, ulong offc,
        global       char * dst,  ulong offd,
        int   n_embd,
        int   hc,
        int   has_comb,
        ulong nbx0, ulong nbx1,
        ulong nbr0, ulong nbr1, ulong nbr2,
        ulong nbp0, ulong nbp1,
        ulong nbc0, ulong nbc1, ulong nbc2,
        ulong nbd0, ulong nbd1, ulong nbd2
) {
    const int i0 = get_global_id(0);
    const int it = get_global_id(1);
    if (i0 >= n_embd) {
        return;
    }

    const float xv = *(global const float *)(x + offx + i0*nbx0 + it*nbx1);

    res  += offr + i0*nbr0 + it*nbr2;
    post += offp + it*nbp1;
    comb += offc + it*nbc2;
    dst  += offd + i0*nbd0 + it*nbd2;

    for (int idst = 0; idst < hc; ++idst) {
        float sum = xv * *(global const float *)(post + idst*nbp0);
        if (has_comb) {
            for (int isrc = 0; isrc < hc; ++isrc) {
                const float rv = *(global const float *)(res  + isrc*nbr1);
                const float cv = *(global const float *)(comb + idst*nbc0 + isrc*nbc1);
                sum += rv * cv;
            }
        } else {
            sum += *(global const float *)(res + idst*nbr1);
        }
        *(global float *)(dst + idst*nbd1) = sum;
    }
}

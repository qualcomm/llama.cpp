// Gated DeltaNet (Qwen3-Next / KDA linear attention) fused op — autoregressive
// (n_tokens == 1) case only. Reference: ggml/src/ggml-cpu/ops.cpp
// ggml_compute_forward_gated_delta_net_f32, ggml/src/ggml-cuda/gated_delta_net.cu.
//
// One thread per (column j, head h, sequence s). Thread owns column j of the
// per-head state matrix S, stored transposed in the output buffer's state
// region as state_out[(h_seq)*S_v*S_v + j*S_v + i] = S[i][j] — i.e. the
// contiguous run state_out[j*S_v .. j*S_v+S_v-1]. The state is read/written
// directly in global memory (this op is memory-bound; no benefit from caching
// the full column in private, which overflows the Adreno register file).
//
// Single step (n_tokens == 1):
//   copy:    S_out[i][j] = S_in[i][j]
//   decay:   S_out[i][j] *= exp(g[i])  (kda)  or  S_out *= exp(g[0])  (scalar)
//   kv[j]  = sum_i S_out[i][j] * k[i]
//   delta[j] = (v[j] - kv[j]) * beta
//   S_out[i][j] += k[i] * delta[j]
//   out[j] = (sum_i S_out[i][j] * q[i]) * scale

kernel void kernel_gated_delta_net_f32(
    global char * q_base,    ulong q_off,
    global char * k_base,    ulong k_off,
    global char * v_base,    ulong v_off,
    global char * g_base,    ulong g_off,
    global char * b_base,    ulong b_off,
    global char * s_base,    ulong s_off,
    global char * dst_base,  ulong dst_off,
    // q/k/v strides in bytes ("contiguous rows": nb?0 == sizeof(float)).
    // nb?1 = head stride, nb?3 = seq stride (nb?2 = token stride, unused: n_tokens == 1)
    ulong nbq1, ulong nbq3,
    ulong nbk1, ulong nbk3,
    ulong nbv1, ulong nbv3,
    int s_v,                   // S_v = state dim
    int neq1, int nek1,        // q/k head counts (<= H)
    int neq3, int nek3,        // q/k seq counts  (<= n_seqs)
    int H,                     // = src_v->ne[1]   (== n_heads_v)
    int n_seqs,
    int kda,                   // 1 if g per-element ([S_v,...]), 0 if scalar ([1,...])
    int neg0                   // g->ne[0]  (== S_v if kda else 1)
) {
    const int gid = get_global_id(0);       // flattened (column j, head, seq)
    if (gid >= s_v * H * n_seqs) return;
    const int j   = gid % s_v;              // column owned by this thread
    const int hs  = gid / s_v;              // flattened (head, seq)
    const int iv1 = hs % H;                 // head index   (0..H-1)
    const int iv3 = hs / H;                 // sequence     (0..n_seqs-1)

    const int rq3 = n_seqs / neq3;
    const int rk3 = n_seqs / nek3;
    const int iq1 = iv1 % neq1;
    const int ik1 = iv1 % nek1;
    const int iq3 = iv3 / rq3;
    const int ik3 = iv3 / rk3;

    const float scale = 1.0f / sqrt((float) s_v);

    q_base   += q_off;
    k_base   += k_off;
    v_base   += v_off;
    g_base   += g_off;
    b_base   += b_off;
    s_base   += s_off;
    dst_base += dst_off;

    // output: [ attn (S_v*H*1*n_seqs) | new_states (S_v*S_v*H*n_seqs) ]
    const ulong attn_elems = (ulong)s_v * H * n_seqs;   // n_tokens == 1
    global float * attn_out  = (global float *)dst_base;
    global float * state_out = (global float *)dst_base + attn_elems;

    // input/output state column j (contiguous run [j*s_v ..]) for this (head,seq)
    global const float * s_in  = (global const float *)s_base + ((ulong)iv3 * H + iv1) * s_v * s_v + (ulong)j * s_v;
    global       float * s_out = state_out                    + ((ulong)iv3 * H + iv1) * s_v * s_v + (ulong)j * s_v;

    global const float * q_d = (global const float *)(q_base + (ulong)iq3*nbq3 + (ulong)iq1*nbq1);  // t == 0
    global const float * k_d = (global const float *)(k_base + (ulong)ik3*nbk3 + (ulong)ik1*nbk1);
    global const float * v_d = (global const float *)(v_base + (ulong)iv3*nbv3 + (ulong)iv1*nbv1);
    const ulong hb = ((ulong)iv3*H + iv1);                              // t == 0
    const float beta = ((global const float *)b_base)[hb];
    global const float * g_d = (global const float *)g_base + hb * (ulong)neg0;

    // copy + decay
    if (kda) {
        for (int i = 0; i < s_v; ++i) s_out[i] = s_in[i] * exp(g_d[i]);
    } else {
        const float gd = exp(g_d[0]);
        for (int i = 0; i < s_v; ++i) s_out[i] = s_in[i] * gd;
    }

    // kv[j] = sum_i S[i][j] * k[i]
    float kv = 0.0f;
    for (int i = 0; i < s_v; ++i) kv = mad(s_out[i], k_d[i], kv);

    const float delta = (v_d[j] - kv) * beta;

    // outer product + output: S[i][j] += k[i]*delta ; out[j] = sum_i S[i][j]*q[i]
    float o = 0.0f;
    for (int i = 0; i < s_v; ++i) {
        const float sij = mad(k_d[i], delta, s_out[i]);
        s_out[i] = sij;
        o = mad(sij, q_d[i], o);
    }

    // attn layout: [S_v, H, 1, n_seqs]
    attn_out[((ulong)iv3*H + iv1) * s_v + j] = o * scale;
}

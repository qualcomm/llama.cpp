#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Gated Delta Net — Qwen3-Next style attention.
// One work-group per (B, H, j) where j = state row; local size = S_v threads.
// Each thread tx owns column tx of state row j (one float in private memory).
// Sequential token loop. KDA mode = g per-state, otherwise g per-head scalar.
//
// State stored TRANSPOSED: s_in[j*S_v + i] = S[i, j], so row j is contiguous.

kernel void kernel_gated_delta_net_f32(
    global const float * q,        ulong off_q,
    global const float * k,        ulong off_k,
    global const float * v,        ulong off_v,
    global const float * g,        ulong off_g,
    global const float * b,        ulong off_b,
    global const float * s_in,     ulong off_s,
    global       float * dst,      ulong off_d,
    int                  S_v,
    int                  H,
    int                  T,
    int                  B,
    int                  G,        // 1 or S_v (kda)
    local        float * shared
) {
    q     = (global const float *)((global const char *) q     + off_q);
    k     = (global const float *)((global const char *) k     + off_k);
    v     = (global const float *)((global const char *) v     + off_v);
    g     = (global const float *)((global const char *) g     + off_g);
    b     = (global const float *)((global const char *) b     + off_b);
    s_in  = (global const float *)((global const char *) s_in  + off_s);
    dst   = (global       float *)((global       char *) dst   + off_d);

    const int j   = get_group_id(2);   // row of state
    const int h   = get_group_id(1);   // head
    const int bi  = get_group_id(0);   // batch
    const int i   = get_local_id(0);   // column of state row (S_v lanes)
    const int lsz = get_local_size(0);

    if (i >= S_v) return;

    const float scale = 1.0f / sqrt((float)S_v);

    // Load initial state: ls = s_in[bi, h, j, i]
    const ulong s_off = (ulong)((bi * H + h) * S_v * S_v) + (ulong)(j * S_v) + (ulong)i;
    float ls = s_in[s_off];

    // attn output: dst[bi, t, h, j]; layout = [S_v, H, T, B] so flat
    // attn[bi*T*H*S_v + t*H*S_v + h*S_v + j]
    // state output: after attn block of size S_v*H*T*B
    const ulong attn_base   = (ulong)(bi * T * H * S_v) + (ulong)(h * S_v) + (ulong)j;
    const ulong state_off   = (ulong)(B * T * H * S_v) +
                              (ulong)((bi * H + h) * S_v * S_v) + (ulong)(j * S_v) + (ulong)i;

    // q/k/v base for (bi, h): contiguous layout [S_v, H, T, B]
    const ulong qkv_t_stride = (ulong)(H * S_v);
    // beta layout: [1, H, T, B] -> elem index = ((bi*T + t)*H + h)
    // gate layout: [G, H, T, B] -> elem index = ((bi*T + t)*H + h)*G + i_or_0

    for (int t = 0; t < T; t++) {
        const ulong tok_off = (ulong)(bi * T + t) * qkv_t_stride + (ulong)(h * S_v);
        const float v_j  = v[tok_off + j];
        const float k_i  = k[tok_off + i];
        const float q_i  = q[tok_off + i];

        const ulong b_off = (ulong)(bi * T + t) * H + h;
        const float beta_v = b[b_off];

        const ulong g_off = (ulong)((bi * T + t) * H + h) * G;
        const float g_v = (G == 1) ? g[g_off] : g[g_off + i];

        ls *= exp(g_v);

        // s_k = sum_i ls[i] * k[i]
        shared[i] = ls * k_i;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int s = lsz / 2; s > 0; s >>= 1) {
            if (i < s) shared[i] += shared[i + s];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        const float s_k = shared[0];

        const float d = (v_j - s_k) * beta_v;

        ls += k_i * d;

        // y = sum_i ls[i] * q[i]
        barrier(CLK_LOCAL_MEM_FENCE);
        shared[i] = ls * q_i;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int s = lsz / 2; s > 0; s >>= 1) {
            if (i < s) shared[i] += shared[i + s];
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (i == 0) {
            const ulong attn_off = attn_base + (ulong)(t * H * S_v);
            dst[attn_off] = shared[0] * scale;
        }
    }

    dst[state_off] = ls;
}

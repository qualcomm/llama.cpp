#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Gated Delta Net — Qwen3-Next style attention.
// One work-group per (iv3, iv1, j) where j = state row; local size = S_v threads.
// Each thread tx owns column tx of state row j (one float in private memory).
// Sequential token loop. KDA mode = g per-state, otherwise g per-head scalar.
//
// State stored TRANSPOSED: s_in[j*S_v + i] = S[i, j], so row j is contiguous.
//
// Supports v_repeat>1 (q/k head broadcast) and permuted q/k/v via nb strides.

kernel void kernel_gated_delta_net_f32(
    global const float * q,        ulong off_q,
    global const float * k,        ulong off_k,
    global const float * v,        ulong off_v,
    global const float * g,        ulong off_g,
    global const float * b,        ulong off_b,
    global const float * s_in,     ulong off_s,
    global       float * dst,      ulong off_d,
    int                  S_v,
    int                  H,        // n_head in v (= q-heads * v_repeat)
    int                  T,
    int                  B,
    int                  G,        // 1 or S_v (kda)
    int                  neq1,     // q n_head
    int                  nek1,     // k n_head
    int                  rq3,
    int                  rk3,
    ulong                nbq1, ulong nbq2, ulong nbq3,
    ulong                nbk1, ulong nbk2, ulong nbk3,
    ulong                nbv1, ulong nbv2, ulong nbv3,
    ulong                nbg1, ulong nbg2, ulong nbg3,
    ulong                nbb1, ulong nbb2, ulong nbb3,
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
    const int iv1 = get_group_id(1);   // head in v
    const int iv3 = get_group_id(0);   // seq
    const int i   = get_local_id(0);   // column of state row (S_v lanes)
    const int lsz = get_local_size(0);

    if (i >= S_v) return;

    const int iq1 = iv1 % neq1;
    const int ik1 = iv1 % nek1;
    const int iq3 = iv3 / rq3;
    const int ik3 = iv3 / rk3;

    const float scale = 1.0f / sqrt((float)S_v);

    // Initial state load (state layout in src5 is contiguous [S_v*S_v, H, B])
    const ulong s_off_in = (ulong)((iv3 * H + iv1) * S_v * S_v) + (ulong)(j * S_v) + (ulong)i;
    float ls = s_in[s_off_in];

    // attn output: dst layout = [S_v, H, T, B] flat
    const ulong attn_base   = (ulong)(iv3 * T * H * S_v) + (ulong)(iv1 * S_v) + (ulong)j;
    const ulong state_off   = (ulong)(B * T * H * S_v) +
                              (ulong)((iv3 * H + iv1) * S_v * S_v) + (ulong)(j * S_v) + (ulong)i;

    for (int t = 0; t < T; t++) {
        global const float * q_row = (global const float *)((global const char *) q + (ulong)iq3*nbq3 + (ulong)t*nbq2 + (ulong)iq1*nbq1);
        global const float * k_row = (global const float *)((global const char *) k + (ulong)ik3*nbk3 + (ulong)t*nbk2 + (ulong)ik1*nbk1);
        global const float * v_row = (global const float *)((global const char *) v + (ulong)iv3*nbv3 + (ulong)t*nbv2 + (ulong)iv1*nbv1);

        const float beta_v = *(global const float *)((global const char *) b + (ulong)iv3*nbb3 + (ulong)t*nbb2 + (ulong)iv1*nbb1);
        global const float * g_row = (global const float *)((global const char *) g + (ulong)iv3*nbg3 + (ulong)t*nbg2 + (ulong)iv1*nbg1);

        const float v_j = v_row[j];
        const float k_i = k_row[i];
        const float q_i = q_row[i];
        const float g_v = (G == 1) ? g_row[0] : g_row[i];

        ls *= exp(g_v);

        shared[i] = ls * k_i;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int s = lsz / 2; s > 0; s >>= 1) {
            if (i < s) shared[i] += shared[i + s];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        const float s_k = shared[0];

        const float d = (v_j - s_k) * beta_v;

        ls += k_i * d;

        barrier(CLK_LOCAL_MEM_FENCE);
        shared[i] = ls * q_i;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int s = lsz / 2; s > 0; s >>= 1) {
            if (i < s) shared[i] += shared[i + s];
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (i == 0) {
            dst[attn_base + (ulong)(t * H * S_v)] = shared[0] * scale;
        }
    }

    dst[state_off] = ls;
}

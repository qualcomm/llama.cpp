// Gated DeltaNet (Qwen3-Next / KDA linear attention) fused op — autoregressive
// (n_tokens == 1) case only. Reference: ggml/src/ggml-cpu/ops.cpp
// ggml_compute_forward_gated_delta_net_f32, ggml/src/ggml-cuda/gated_delta_net.cu,
// ggml/src/ggml-vulkan/vulkan-shaders/gated_delta_net.comp.
//
// State layout (matches Vulkan / CPU): state[(h_seq)*S_v*S_v + j*S_v + i] = S[i][j]
// i.e. each column j is contiguous along i.
//
// Single step (n_tokens == 1):
//   copy:    S_out[i][j] = S_in[i][j]
//   decay:   S_out[i][j] *= exp(g[i])  (kda)  or  S_out *= exp(g[0])  (scalar)
//   kv[j]  = sum_i S_out[i][j] * k[i]
//   delta[j] = (v[j] - kv[j]) * beta
//   S_out[i][j] += k[i] * delta[j]
//   out[j] = (sum_i S_out[i][j] * q[i]) * scale

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_khr_subgroup_shuffle
#pragma OPENCL EXTENSION cl_khr_subgroup_shuffle : enable
#define HAS_SUBGROUP_SHUFFLE 1
#elif defined(cl_qcom_subgroup_shuffle)
#pragma OPENCL EXTENSION cl_qcom_subgroup_shuffle : enable
#define HAS_SUBGROUP_SHUFFLE 1
#endif

#if defined(cl_qcom_reqd_sub_group_size)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#else
#define REQD_SUBGROUP_SIZE_128
#endif

// ============================================================================
// Generic fallback: one thread per (column j, head h, sequence s). Used when
// the S_v=128 specialization is not applicable.
// ============================================================================
kernel void kernel_gated_delta_net_f32(
    global char * q_base,    ulong q_off,
    global char * k_base,    ulong k_off,
    global char * v_base,    ulong v_off,
    global char * g_base,    ulong g_off,
    global char * b_base,    ulong b_off,
    global char * s_base,    ulong s_off,
    global char * dst_base,  ulong dst_off,
    ulong nbq1, ulong nbq3,
    ulong nbk1, ulong nbk3,
    ulong nbv1, ulong nbv3,
    int s_v,
    int neq1, int nek1,
    int neq3, int nek3,
    int H,
    int n_seqs,
    int kda,
    int neg0
) {
    const int gid = get_global_id(0);
    if (gid >= s_v * H * n_seqs) return;
    const int j   = gid % s_v;
    const int hs  = gid / s_v;
    const int iv1 = hs % H;
    const int iv3 = hs / H;

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

    const ulong attn_elems = (ulong)s_v * H * n_seqs;
    global float * attn_out  = (global float *)dst_base;
    global float * state_out = (global float *)dst_base + attn_elems;

    global const float * s_in  = (global const float *)s_base + ((ulong)iv3 * H + iv1) * s_v * s_v + (ulong)j * s_v;
    global       float * s_out = state_out                    + ((ulong)iv3 * H + iv1) * s_v * s_v + (ulong)j * s_v;

    global const float * q_d = (global const float *)(q_base + (ulong)iq3*nbq3 + (ulong)iq1*nbq1);
    global const float * k_d = (global const float *)(k_base + (ulong)ik3*nbk3 + (ulong)ik1*nbk1);
    global const float * v_d = (global const float *)(v_base + (ulong)iv3*nbv3 + (ulong)iv1*nbv1);
    const ulong hb = ((ulong)iv3*H + iv1);
    const float beta = ((global const float *)b_base)[hb];
    global const float * g_d = (global const float *)g_base + hb * (ulong)neg0;

    if (kda) {
        for (int i = 0; i < s_v; ++i) s_out[i] = s_in[i] * exp(g_d[i]);
    } else {
        const float gd = exp(g_d[0]);
        for (int i = 0; i < s_v; ++i) s_out[i] = s_in[i] * gd;
    }

    float kv = 0.0f;
    for (int i = 0; i < s_v; ++i) kv = mad(s_out[i], k_d[i], kv);

    const float delta = (v_d[j] - kv) * beta;

    float o = 0.0f;
    for (int i = 0; i < s_v; ++i) {
        const float sij = mad(k_d[i], delta, s_out[i]);
        s_out[i] = sij;
        o = mad(sij, q_d[i], o);
    }

    attn_out[((ulong)iv3*H + iv1) * s_v + j] = o * scale;
}

// ============================================================================
// S_v=128 specialization (Qwen3-Next / Qwen3.6-A3B).
//
// Layout per workgroup (1 full Adreno subgroup of 128 lanes):
//   lane           = lid % 32       — row-lane within column (0..31)
//   col_in_wg      = lid / 32       — column within workgroup (0..3)
//   COLS_PER_WG    = 4              — 4 columns processed per workgroup
//   LANES_PER_COL  = 32             — 32 lanes cooperate per column
//   ROWS_PER_LANE  = 4              — each lane owns 4 rows of state in private
//
// Grid: (head_id, seq_id, col_block) with col_block in [0 .. 128/4 = 32).
//   col = col_block * COLS_PER_WG + col_in_wg
//
// kv/attn reductions are cluster-of-32 sums via sub_group_shuffle_xor — each
// 32-lane cluster within the 128-wide subgroup reduces independently because
// XOR with mask < 32 never crosses cluster boundaries.
// ============================================================================
#if defined(HAS_SUBGROUP_SHUFFLE)

#define GDN_SV    128
#define GDN_LPC   32
#define GDN_CPWG  4
#define GDN_RPL   4

inline float gdn_cluster32_sum(float v) {
    v += sub_group_shuffle_xor(v,  1);
    v += sub_group_shuffle_xor(v,  2);
    v += sub_group_shuffle_xor(v,  4);
    v += sub_group_shuffle_xor(v,  8);
    v += sub_group_shuffle_xor(v, 16);
    return v;
}

REQD_SUBGROUP_SIZE_128
kernel void kernel_gated_delta_net_f32_sv128(
    global char * q_base,    ulong q_off,
    global char * k_base,    ulong k_off,
    global char * v_base,    ulong v_off,
    global char * g_base,    ulong g_off,
    global char * b_base,    ulong b_off,
    global char * s_base,    ulong s_off,
    global char * dst_base,  ulong dst_off,
    ulong nbq1, ulong nbq3,
    ulong nbk1, ulong nbk3,
    ulong nbv1, ulong nbv3,
    int neq1, int nek1,
    int neq3, int nek3,
    int H,
    int n_seqs,
    int kda,
    int neg0
) {
    const int lid       = get_local_id(0);
    const int lane      = lid & (GDN_LPC - 1);
    const int col_in_wg = lid >> 5;

    const int head_id   = get_group_id(0);
    const int seq_id    = get_group_id(1);
    const int col_block = get_group_id(2);
    const int col       = col_block * GDN_CPWG + col_in_wg;

    const int iv1 = head_id;
    const int iv3 = seq_id;
    const int rq3 = n_seqs / neq3;
    const int rk3 = n_seqs / nek3;
    const int iq1 = iv1 % neq1;
    const int ik1 = iv1 % nek1;
    const int iq3 = iv3 / rq3;
    const int ik3 = iv3 / rk3;

    q_base   += q_off;
    k_base   += k_off;
    v_base   += v_off;
    g_base   += g_off;
    b_base   += b_off;
    s_base   += s_off;
    dst_base += dst_off;

    const ulong attn_elems = (ulong)GDN_SV * H * n_seqs;
    global float * attn_out  = (global float *)dst_base;
    global float * state_out = (global float *)dst_base + attn_elems;

    global const float * s_in  = (global const float *)s_base + ((ulong)iv3 * H + iv1) * GDN_SV * GDN_SV + (ulong)col * GDN_SV;
    global       float * s_out = state_out                    + ((ulong)iv3 * H + iv1) * GDN_SV * GDN_SV + (ulong)col * GDN_SV;

    global const float * q_d = (global const float *)(q_base + (ulong)iq3*nbq3 + (ulong)iq1*nbq1);
    global const float * k_d = (global const float *)(k_base + (ulong)ik3*nbk3 + (ulong)ik1*nbk1);
    global const float * v_d = (global const float *)(v_base + (ulong)iv3*nbv3 + (ulong)iv1*nbv1);
    const ulong hb = (ulong)iv3 * H + iv1;
    const float beta_val = ((global const float *)b_base)[hb];
    global const float * g_d = (global const float *)g_base + hb * (ulong)neg0;

    float s_shard[GDN_RPL];
    float k_reg  [GDN_RPL];
    float q_reg  [GDN_RPL];
    float g_exp  [GDN_RPL];

    #pragma unroll
    for (int r = 0; r < GDN_RPL; r++) {
        const int i = r * GDN_LPC + lane;
        s_shard[r] = s_in[i];
        k_reg[r]   = k_d[i];
        q_reg[r]   = q_d[i];
    }

    if (kda) {
        #pragma unroll
        for (int r = 0; r < GDN_RPL; r++) {
            g_exp[r] = exp(g_d[r * GDN_LPC + lane]);
        }
    } else {
        const float gv = exp(g_d[0]);
        #pragma unroll
        for (int r = 0; r < GDN_RPL; r++) g_exp[r] = gv;
    }

    const float v_val = v_d[col];

    float kv_shard = 0.0f;
    #pragma unroll
    for (int r = 0; r < GDN_RPL; r++) {
        kv_shard = mad(g_exp[r] * s_shard[r], k_reg[r], kv_shard);
    }
    const float kv_col = gdn_cluster32_sum(kv_shard);

    const float delta = (v_val - kv_col) * beta_val;

    float attn_partial = 0.0f;
    #pragma unroll
    for (int r = 0; r < GDN_RPL; r++) {
        const float sij = mad(k_reg[r], delta, g_exp[r] * s_shard[r]);
        s_shard[r] = sij;
        attn_partial = mad(sij, q_reg[r], attn_partial);
    }
    const float attn_col = gdn_cluster32_sum(attn_partial);

    if (lane == 0) {
        attn_out[((ulong)iv3 * H + iv1) * GDN_SV + col] = attn_col * (1.0f / sqrt((float) GDN_SV));
    }

    #pragma unroll
    for (int r = 0; r < GDN_RPL; r++) {
        s_out[r * GDN_LPC + lane] = s_shard[r];
    }
}

#endif // HAS_SUBGROUP_SHUFFLE

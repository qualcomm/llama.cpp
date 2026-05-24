// Gated DeltaNet (Qwen3-Next / Qwen3.5 MTP / KDA linear attention) fused op.
// Reference: ggml/src/ggml-cpu/ops.cpp ggml_compute_forward_gated_delta_net_f32,
// ggml/src/ggml-cuda/gated_delta_net.cu (the K>1 / keep_rs_t version).
//
// K>1 snapshot slots for MTP speculative-decoding rollback (upstream PR #22673):
//   - Input state shape (S_v*S_v*H, K, n_seqs). Only slot 0 holds the seed; the
//     rest of K is caller-owned and untouched by us (used to roll back to an
//     earlier draft position).
//   - Output state layout: K slots stacked as the outermost dim of dst, each
//     slot of size S_v*S_v*H*n_seqs. Slot k holds the state AFTER processing the
//     (shift+k)-th token, where shift = n_tokens - K (negative when n_tokens<K,
//     so the last n_tokens slots get written and earlier ones are left alone).
//   - K==1: backwards-compatible — only slot 0 gets the final state.
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
// Max s_v supported by the private state buffer in the generic kernel.
// All known GDN-bearing models (Qwen3-Next, Qwen3.5/3.6 MoE) use s_v <= 128.
#ifndef GDN_GENERIC_MAX_SV
#define GDN_GENERIC_MAX_SV 128
#endif

kernel void kernel_gated_delta_net_f32(
    global char * q_base,    ulong q_off,
    global char * k_base,    ulong k_off,
    global char * v_base,    ulong v_off,
    global char * g_base,    ulong g_off,
    global char * b_base,    ulong b_off,
    global char * s_base,    ulong s_off,
    global char * dst_base,  ulong dst_off,
    ulong nbq1, ulong nbq2, ulong nbq3,
    ulong nbk1, ulong nbk2, ulong nbk3,
    ulong nbv1, ulong nbv2, ulong nbv3,
    ulong nbb1, ulong nbb2, ulong nbb3,
    ulong nbg1, ulong nbg2, ulong nbg3,
    int s_v,
    int neq1, int nek1,
    int neq3, int nek3,
    int H,
    int n_tokens,
    int n_seqs,
    int kda,
    int neg0,
    int K
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

    const ulong attn_elems = (ulong)s_v * H * (ulong)n_tokens * n_seqs;
    global float * attn_out_base  = (global float *)dst_base;
    global float * state_out_base = (global float *)dst_base + attn_elems;

    // Input state: always slot 0 of the K-snapshot input (layout (D, K, n_seqs)).
    //   For K == 1: per_seq_stride = 1 * H * s_v * s_v (matches the legacy offset).
    //   For K  > 1: per_seq_stride = K * H * s_v * s_v.
    global const float * s_in =
        (global const float *)s_base
        + ((ulong)iv3 * K * H + iv1) * s_v * s_v
        + (ulong)j * s_v;

    // Output state: K slots stacked, each S_v*S_v*H*n_seqs floats.
    const ulong state_size_per_slot = (ulong)s_v * s_v * H * n_seqs;
    const ulong state_out_seq_head  = ((ulong)iv3 * H + iv1) * s_v * s_v + (ulong)j * s_v;

    // Working state column in private memory. Capped at GDN_GENERIC_MAX_SV.
    float s_col[GDN_GENERIC_MAX_SV];
    for (int i = 0; i < s_v; ++i) s_col[i] = s_in[i];

    global char * q_hd = q_base + (ulong)iq3*nbq3 + (ulong)iq1*nbq1;
    global char * k_hd = k_base + (ulong)ik3*nbk3 + (ulong)ik1*nbk1;
    global char * v_hd = v_base + (ulong)iv3*nbv3 + (ulong)iv1*nbv1;
    global char * b_hd = b_base + (ulong)iv3 * nbb3 + (ulong)iv1 * nbb1;
    global char * g_hd = g_base + (ulong)iv3 * nbg3 + (ulong)iv1 * nbg1;

    global float * attn_data = attn_out_base + ((ulong)iv3 * (ulong)n_tokens * H + iv1) * s_v;

    // Slot mapping per CUDA / SYCL: target_slot = t - (n_tokens - K).
    //   K == 1, t == n_tokens-1: target_slot = 0     -> final state -> slot 0.
    //   K  > 1, n_tokens >= K:   last K iters fill slots 0..K-1.
    //   K  > 1, n_tokens <  K:   last n_tokens iters fill slots K-n_tokens..K-1.
    const int shift = n_tokens - K;

    for (int t = 0; t < n_tokens; t++) {
        global const float * q_d = (global const float *)(q_hd + (ulong)t * nbq2);
        global const float * k_d = (global const float *)(k_hd + (ulong)t * nbk2);
        global const float * v_d = (global const float *)(v_hd + (ulong)t * nbv2);
        const float beta         = *(global const float *)(b_hd + (ulong)t * nbb2);
        global const float * g_d = (global const float *)(g_hd + (ulong)t * nbg2);

        if (kda) {
            for (int i = 0; i < s_v; ++i) s_col[i] *= exp(g_d[i]);
        } else {
            const float gd = exp(g_d[0]);
            for (int i = 0; i < s_v; ++i) s_col[i] *= gd;
        }

        float kv = 0.0f;
        for (int i = 0; i < s_v; ++i) kv = mad(s_col[i], k_d[i], kv);

        const float delta = (v_d[j] - kv) * beta;

        float o = 0.0f;
        for (int i = 0; i < s_v; ++i) {
            const float sij = mad(k_d[i], delta, s_col[i]);
            s_col[i] = sij;
            o = mad(sij, q_d[i], o);
        }

        attn_data[j] = o * scale;
        attn_data += (ulong)s_v * H;

        const int target_slot = t - shift;
        if (target_slot >= 0 && target_slot < K) {
            global float * slot_ptr =
                state_out_base + (ulong)target_slot * state_size_per_slot + state_out_seq_head;
            for (int i = 0; i < s_v; ++i) slot_ptr[i] = s_col[i];
        }
    }
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
    ulong nbq1, ulong nbq2, ulong nbq3,
    ulong nbk1, ulong nbk2, ulong nbk3,
    ulong nbv1, ulong nbv2, ulong nbv3,
    ulong nbb1, ulong nbb2, ulong nbb3,
    ulong nbg1, ulong nbg2, ulong nbg3,
    int neq1, int nek1,
    int neq3, int nek3,
    int H,
    int n_tokens,
    int n_seqs,
    int kda,
    int neg0,
    int K
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

    // Output layout: [ attn (S_v * H * n_tokens * n_seqs) | new_state (S_v * S_v * H * n_seqs) ]
    const ulong attn_elems = (ulong)GDN_SV * H * (ulong)n_tokens * n_seqs;
    global float * attn_out_base  = (global float *)dst_base;
    global float * state_out_base = (global float *)dst_base + attn_elems;

    // Input state: slot 0 only, layout (D, K, n_seqs) — seq stride is K * D.
    global const float * s_in  = (global const float *)s_base
        + ((ulong)iv3 * K * H + iv1) * GDN_SV * GDN_SV + (ulong)col * GDN_SV;

    // Output state: K slots stacked, each S_v*S_v*H*n_seqs floats.
    const ulong gdn_slot_size      = (ulong)GDN_SV * GDN_SV * H * n_seqs;
    const ulong gdn_state_seq_head = ((ulong)iv3 * H + iv1) * GDN_SV * GDN_SV + (ulong)col * GDN_SV;

    // Per-head per-seq base pointers; per-token offsets applied inside the t-loop.
    global char * q_hd = q_base + (ulong)iq3*nbq3 + (ulong)iq1*nbq1;
    global char * k_hd = k_base + (ulong)ik3*nbk3 + (ulong)ik1*nbk1;
    global char * v_hd = v_base + (ulong)iv3*nbv3 + (ulong)iv1*nbv1;
    global char * b_hd = b_base + (ulong)iv3*nbb3 + (ulong)iv1*nbb1;
    global char * g_hd = g_base + (ulong)iv3*nbg3 + (ulong)iv1*nbg1;

    // Load state column 'col' into private once for the whole t-loop.
    float s_shard[GDN_RPL];
    #pragma unroll
    for (int r = 0; r < GDN_RPL; r++) {
        s_shard[r] = s_in[r * GDN_LPC + lane];
    }

    const float scale = 1.0f / sqrt((float) GDN_SV);

    // attn output advances by GDN_SV * H per token, starting at first token of
    // this (seq, head): attn_data[t][col] = base + (iv3*n_tokens + t)*H*S_v + iv1*S_v + col.
    global float * attn_data = attn_out_base + ((ulong)iv3 * (ulong)n_tokens * H + iv1) * GDN_SV;

    // Slot mapping: target_slot = t - (n_tokens - K). See generic kernel comment.
    const int sv128_shift = n_tokens - K;

    // For decode (n_tokens==1) the __local-cache variant was a slight win but
    // barriers would dominate for the prefill t-loop. We read k/q/g directly
    // from global on every iter — the 4 cols sharing a head only need ~4 cache
    // lines per (r,token) read, which the Adreno L1 absorbs across the 4
    // cluster-of-32 reads in the same workgroup. No barriers in the hot loop.
    for (int t = 0; t < n_tokens; t++) {
        global const float * q_t = (global const float *)(q_hd + (ulong)t * nbq2);
        global const float * k_t = (global const float *)(k_hd + (ulong)t * nbk2);
        global const float * v_t = (global const float *)(v_hd + (ulong)t * nbv2);
        const float beta_val     = *(global const float *)(b_hd + (ulong)t * nbb2);
        global const float * g_t = (global const float *)(g_hd + (ulong)t * nbg2);

        float k_reg[GDN_RPL];
        float q_reg[GDN_RPL];
        float g_exp[GDN_RPL];

        #pragma unroll
        for (int r = 0; r < GDN_RPL; r++) {
            const int i = r * GDN_LPC + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if (kda) {
            #pragma unroll
            for (int r = 0; r < GDN_RPL; r++) {
                g_exp[r] = exp(g_t[r * GDN_LPC + lane]);
            }
        } else {
            const float gv = exp(g_t[0]);
            #pragma unroll
            for (int r = 0; r < GDN_RPL; r++) g_exp[r] = gv;
        }

        const float v_val = v_t[col];

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
            attn_data[col] = attn_col * scale;
        }
        attn_data += (ulong)GDN_SV * H;

        // Write this t's state to slot target_slot if it falls in [0, K).
        // For K==1 only the last iteration writes (target_slot=0). For K>1
        // the last K iterations fill slots 0..K-1 in order.
        const int target_slot = t - sv128_shift;
        if (target_slot >= 0 && target_slot < K) {
            global float * slot_ptr =
                state_out_base + (ulong)target_slot * gdn_slot_size + gdn_state_seq_head;
            #pragma unroll
            for (int r = 0; r < GDN_RPL; r++) {
                slot_ptr[r * GDN_LPC + lane] = s_shard[r];
            }
        }
    }
}

#endif // HAS_SUBGROUP_SHUFFLE

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#ifdef cl_khr_integer_dot_product
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#define FA_HAVE_INT_DOT 1
#endif

#ifdef cl_khr_subgroup_shuffle
#pragma OPENCL EXTENSION cl_khr_subgroup_shuffle : enable
#define HAS_SUBGROUP_SHUFFLE 1
#elif defined(cl_qcom_subgroup_shuffle)
#pragma OPENCL EXTENSION cl_qcom_subgroup_shuffle : enable
#define HAS_SUBGROUP_SHUFFLE 1
// Adreno compilers that expose only cl_qcom_subgroup_shuffle do not declare the KHR
// name, so calling it is an implicit declaration and the program fails to build.
// Route it to the qcom builtin.
#define sub_group_shuffle_xor(val, mask) qcom_sub_group_shuffle_xor((val), (mask), CLK_SUB_GROUP_SHUFFLE_WIDTH_WAVE_SIZE_QCOM, 0.0f)
#endif

// Flash attention: Q=f32, K=q8_0, V=q8_0.

#define ACC_TYPE float
#define ACC_TYPE4 float4
#define Q_DATA_TYPE4 float4
#define O_DATA_TYPE4 float4
#define MASK_DATA_TYPE half
#define CONVERT_Q_ACC4(x) (x)
#define CONVERT_O_DATA4(x) (x)

#define DK_VEC (DK/4)
#define DV_VEC (DV/4)
// q1 reduces over a Q1_WG_SIZE-wide WG via work-group barriers; the launch WG
// must match. Defaults to the Adreno sg (64); host passes -D FA_SG=32 on Intel.
#ifndef FA_SG
#define FA_SG 64
#endif
#define Q1_WG_SIZE FA_SG

// The kernels are built with -cl-finite-math-only. On some older Adreno GPUs,
// infinite operand can cause undefined behavior and miscompilation for exp.
// Therefore, a large negative value is used instead.
#define FA_M_INIT (-3.0e38f)

// q8_0 block: 2B scale (half) + 32B int8 quants.
#define QK8_0 32
#define Q8_0_BLOCK_SIZE 34

#define DK_Q8_BLOCKS (DK / QK8_0)
#define DV_Q8_BLOCKS (DV / QK8_0)

// ---------------------------------------------------------------------------
// Asymmetric native KV: K = q8_0, V = q4_0.
//
// This file is flash_attn_f32_q8_0.cl with the V side swapped to q4_0. The K
// path -- including the dp4a/int-dot QK variants -- is byte-for-byte the q8_0
// one, so every K symbol above stays as-is; only V loads change.
//
// Motivation (gpt-oss): the K cache carries attention-sink outliers that need
// q8 precision, while V is lossless at q4. Dequantising both sides to f16 and
// running the f16 kernel pays full f16 attention bandwidth plus a dequant pass.
//
// q4_0 block: 2B scale (half) + 16B nibbles = 32 elements in 18 bytes.
// Nibble mapping is ggml's: qs[j] low nibble -> element j, high nibble ->
// element j+16, value = d * (nibble - 8).
// QK4_0 == QK8_0 == 32, so blocks-per-row, DV_VEC and every local-array
// dimension are unchanged from the q8_0 path -- only the byte stride and the
// unpack differ.
#define QK4_0 32
#define Q4_0_BLOCK_SIZE 18
#define DV_Q4_BLOCKS (DV / QK4_0)

inline float dot_q8_0_f32(const global char * block_ptr, ACC_TYPE4 * q_slice) {
    float d = vload_half(0, (const global half *)block_ptr);
    const global char * qs = block_ptr + 2;

    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        float4 qv = (float4)((float)qs[i*4], (float)qs[i*4+1], (float)qs[i*4+2], (float)qs[i*4+3]);
        sum += dot(q_slice[i], qv);
    }
    return sum * d;
}

#ifdef FA_HAVE_INT_DOT
inline uint pack_i8x4(char a, char b, char c, char d) {
    return ((uint)(uchar)a)       |
           ((uint)(uchar)b) <<  8  |
           ((uint)(uchar)c) << 16  |
           ((uint)(uchar)d) << 24;
}

inline float quant_q_block_int8_packed(const ACC_TYPE4 * q_block,
                                       uint *            out_packed) {
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float4 av = fabs(q_block[i]);
        amax = fmax(amax, fmax(fmax(av.s0, av.s1), fmax(av.s2, av.s3)));
    }
    float qd  = amax / 127.0f;
    float qid = (amax > 0.0f) ? 127.0f / amax : 0.0f;

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float4 v = q_block[i] * qid;
        char a = (char)((int)round(v.s0));
        char b = (char)((int)round(v.s1));
        char c = (char)((int)round(v.s2));
        char d = (char)((int)round(v.s3));
        out_packed[i] = pack_i8x4(a, b, c, d);
    }
    return qd;
}

inline float dot_q8_0_int(const global char * k_block_ptr,
                          const uint *        q_packed,
                          float               q_d) {
    float kd = vload_half(0, (const global half *)k_block_ptr);
    const global uchar * k_qs = (const global uchar *)(k_block_ptr + 2);

    // k_qs is 2-byte aligned; pack chars per iteration rather than cast to uint*.
    int sum = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint k_packed =
              (uint)k_qs[i*4 + 0]        |
             ((uint)k_qs[i*4 + 1]) <<  8 |
             ((uint)k_qs[i*4 + 2]) << 16 |
             ((uint)k_qs[i*4 + 3]) << 24;
        sum = dot_acc_sat_4x8packed_ss_int(q_packed[i], k_packed, sum);
    }
    return (float)sum * q_d * kd;
}
#endif // FA_HAVE_INT_DOT

inline void dequant_q8_0_f32(const global char * block_ptr, ACC_TYPE4 * out) {
    float d = vload_half(0, (const global half *)block_ptr);
    const global char * qs = block_ptr + 2;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        out[i] = d * (float4)((float)qs[i*4], (float)qs[i*4+1], (float)qs[i*4+2], (float)qs[i*4+3]);
    }
}

// V dequant (q4_0), whole block. out[m] holds block elements 4m..4m+3, exactly
// the ordering dequant_q8_0_f32 produces, so every consumer of this array -- the
// o_acc accumulate and the l_v staging indices -- is unchanged.
//
// Elements 0..15 are the low nibbles of qs[0..15] and 16..31 the high nibbles of
// the same bytes, so out[0..3] read low nibbles and out[4..7] read high nibbles
// of the same 16 bytes.
inline void dequant_q4_0_f32(const global char * block_ptr, ACC_TYPE4 * out) {
    float d = vload_half(0, (const global half *)block_ptr);
    const global uchar * qs = (const global uchar *)(block_ptr + 2);

    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        out[g] = d * (float4)((float)(int)(qs[g*4 + 0] & 0x0F) - 8.0f,
                              (float)(int)(qs[g*4 + 1] & 0x0F) - 8.0f,
                              (float)(int)(qs[g*4 + 2] & 0x0F) - 8.0f,
                              (float)(int)(qs[g*4 + 3] & 0x0F) - 8.0f);
    }
    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        out[4 + g] = d * (float4)((float)(int)(qs[g*4 + 0] >> 4) - 8.0f,
                                  (float)(int)(qs[g*4 + 1] >> 4) - 8.0f,
                                  (float)(int)(qs[g*4 + 2] >> 4) - 8.0f,
                                  (float)(int)(qs[g*4 + 3] >> 4) - 8.0f);
    }
}

// max_bias<=0 returns 1.0 so score += 1.0 * mask[k] stays a no-op multiplier.
inline float get_alibi_slope(float max_bias, int head_idx, int n_head_log2, float m0, float m1) {
    if (max_bias <= 0.0f) return 1.0f;
    float base = (head_idx < n_head_log2) ? m0 : m1;
    int   exph = (head_idx < n_head_log2) ? (head_idx + 1) : (2*(head_idx - n_head_log2) + 1);
    return pow(base, (float)exph);
}

// q1 decode: one query row per WG, threads sweep KV positions.
__kernel void flash_attn_f32_q8_0_q4_0_q1(
    const global void * q_void, ulong q_offset,
    const global void * k_void, ulong k_offset,
    const global void * v_void, ulong v_offset,
    global void * o_void, ulong o_offset,
    const float scale,
    const int n_q,
    const int n_kv,
    const int is_causal,
    const int n_head,
    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
    const ulong k_nb1, const ulong k_nb2, const ulong k_nb3,
    const ulong v_nb1, const ulong v_nb2, const ulong v_nb3,
    const ulong o_nb1, const ulong o_nb2, const ulong o_nb3,
    const float max_bias,
    const float m0,
    const float m1,
    const int n_head_log2,
    const float logit_softcap,
    const int n_head_kv,
    const global void* mask_void,
    const ulong mask_offset,
    const ulong mask_nb1,
    const ulong mask_nb2,
    const ulong mask_nb3,
    const int mask_ne2,
    const int mask_ne3,
    const global void* sinks_void,
    const ulong sinks_offset
) {
    const int tid = get_local_id(0);
    const int head_batch_idx = get_global_id(1);

    const int batch_idx = head_batch_idx / n_head;
    const int head_idx = head_batch_idx % n_head;

    const int gqa_ratio = n_head / n_head_kv;
    const int head_kv_idx = head_idx / gqa_ratio;

    const global char* q_base = (const global char*)q_void + q_offset;
    const global char* k_base = (const global char*)k_void + k_offset;
    const global char* v_base = (const global char*)v_void + v_offset;
    global char* o_base = (global char*)o_void + o_offset;

    const global char* mask_base = NULL;
    if (mask_void != NULL) {
        const int mask_head_idx = head_idx % mask_ne2;
        const int mask_batch_idx = batch_idx % mask_ne3;
        mask_base = (const global char*)mask_void + mask_offset + mask_batch_idx * mask_nb3 + mask_head_idx * mask_nb2;
    }

    ACC_TYPE4 q_priv[DK_VEC];
    const ulong q_row_offset = batch_idx * q_nb3 + head_idx * q_nb2;
    const global Q_DATA_TYPE4* q_ptr = (const global Q_DATA_TYPE4*)(q_base + q_row_offset);
    #pragma unroll
    for (int i = 0; i < DK_VEC; ++i) {
        q_priv[i] = CONVERT_Q_ACC4(q_ptr[i]);
    }

#ifdef FA_HAVE_INT_DOT
    // Quantise Q once per thread; q_priv stays as fp for the V accumulate.
    uint  q_packed[DK_Q8_BLOCKS * 8];
    float q_d_scale[DK_Q8_BLOCKS];
    #pragma unroll
    for (int b = 0; b < DK_Q8_BLOCKS; ++b) {
        q_d_scale[b] = quant_q_block_int8_packed(&q_priv[b * 8], &q_packed[b * 8]);
    }
#endif

    float slope = get_alibi_slope(max_bias, head_idx, n_head_log2, m0, m1);

    const global ACC_TYPE* sinks_ptr = NULL;
    if (sinks_void != NULL) {
        sinks_ptr = (const global ACC_TYPE*)((const global char*)sinks_void + sinks_offset);
    }

    // One-pass online softmax: per-thread maintains running (m_i, l_i, o_acc),
    // updating each as new K positions are processed. Eliminates the second
    // K read of the original two-pass implementation. After the loop, threads
    // are merged via the standard FA-2 cross-thread reduction (rescale each
    // thread's l_i and o_acc by alpha=exp(m_i_thread - m_final), then sum).
    ACC_TYPE m_i = (sinks_ptr != NULL) ? sinks_ptr[head_idx] : FA_M_INIT;
    ACC_TYPE l_i = 0.0f;
    ACC_TYPE4 o_acc[DV_VEC];
    #pragma unroll
    for (int i = 0; i < DV_VEC; ++i) o_acc[i] = (ACC_TYPE4)(0.0f);

    for (int k_idx = tid; k_idx < n_kv; k_idx += Q1_WG_SIZE) {
        const global char* k_row = k_base + batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_idx * k_nb1;
        const global char* v_row = v_base + batch_idx * v_nb3 + head_kv_idx * v_nb2 + k_idx * v_nb1;

        ACC_TYPE score = 0.0f;
        #pragma unroll
        for (int b = 0; b < DK_Q8_BLOCKS; b++) {
#ifdef FA_HAVE_INT_DOT
            score += dot_q8_0_int(k_row + b * Q8_0_BLOCK_SIZE,
                                   &q_packed[b * 8], q_d_scale[b]);
#else
            score += dot_q8_0_f32(k_row + b * Q8_0_BLOCK_SIZE, &q_priv[b * 8]);
#endif
        }
        score *= scale;

        if (mask_base != NULL) {
            const global MASK_DATA_TYPE* mask_ptr = (const global MASK_DATA_TYPE*)(mask_base);
            score += slope * (ACC_TYPE)mask_ptr[k_idx];
        }
        if (logit_softcap > 0.0f) {
            score = logit_softcap * tanh(score / logit_softcap);
        }

        // Online softmax step.
        const ACC_TYPE m_new = max(m_i, score);
        const ACC_TYPE alpha = exp(m_i  - m_new);
        const ACC_TYPE p     = exp(score - m_new);

        l_i = alpha * l_i + p;
        #pragma unroll
        for (int i = 0; i < DV_VEC; ++i) o_acc[i] *= alpha;

        #pragma unroll
        for (int b = 0; b < DV_Q4_BLOCKS; b++) {
            ACC_TYPE4 v_dequant[8];
            dequant_q4_0_f32(v_row + b * Q4_0_BLOCK_SIZE, v_dequant);
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                o_acc[b * 8 + i] = mad(p, v_dequant[i], o_acc[b * 8 + i]);
            }
        }

        m_i = m_new;
    }

    // Cross-thread reduce: max(m_i) -> m_final, then rescale per-thread l_i
    // and o_acc by alpha = exp(m_i_thread - m_final) before sum-reduce.
    __local ACC_TYPE local_m[Q1_WG_SIZE];
    local_m[tid] = m_i;
    barrier(CLK_LOCAL_MEM_FENCE);
    #pragma unroll
    for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) local_m[tid] = max(local_m[tid], local_m[tid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const ACC_TYPE m_final = local_m[0];

    const ACC_TYPE alpha_final = exp(m_i - m_final);
    l_i *= alpha_final;
    #pragma unroll
    for (int i = 0; i < DV_VEC; ++i) o_acc[i] *= alpha_final;

    __local ACC_TYPE local_l[Q1_WG_SIZE];
    __local ACC_TYPE4 local_o_comp[Q1_WG_SIZE];
    local_l[tid] = l_i;
    barrier(CLK_LOCAL_MEM_FENCE);
    #pragma unroll
    for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) local_l[tid] += local_l[tid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const ulong o_row_offset = batch_idx * o_nb3 + head_idx * o_nb1;
    global O_DATA_TYPE4 *o_row = (global O_DATA_TYPE4 *)(o_base + o_row_offset);
    ACC_TYPE l_final = local_l[0];

    if (sinks_ptr != NULL) {
        l_final += exp(sinks_ptr[head_idx] - m_final);
    }

    if (l_final > 0.0f) {
        const ACC_TYPE l_inv = 1.0f / l_final;
        for (int i = 0; i < DV_VEC; i++) {
            local_o_comp[tid] = o_acc[i];
            barrier(CLK_LOCAL_MEM_FENCE);
            #pragma unroll
            for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
                if (tid < s) local_o_comp[tid] += local_o_comp[tid + s];
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (tid == 0) {
                o_row[i] = CONVERT_O_DATA4(local_o_comp[0] * l_inv);
            }
        }
    } else if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < DV_VEC; ++i) o_row[i] = (O_DATA_TYPE4)(0.0f);
    }
}

#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#else
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define REQD_SUBGROUP_SIZE_64 __attribute__((qcom_reqd_sub_group_size("half")))
#else
#define REQD_SUBGROUP_SIZE_64
#endif

#define VEC_NSG          4
#define VEC_WG_SIZE      (Q1_WG_SIZE * VEC_NSG)
#define Q1V_DV_PER_THREAD ((DV_VEC + Q1_WG_SIZE - 1) / Q1_WG_SIZE)

inline float4 dequant_q8_0_lane(const global char * block_ptr, int lane) {
    const float d = vload_half(0, (const global half *)block_ptr);
    const global char * qs = block_ptr + 2 + lane * 4;
    // Keep the scalar char loads: a vload4/convert_float4 form measured
    // -1.6..-2.8% tg on the X2-90 (DX.50.39) - the compiler does better here
    // than the explicit vector load.
    return d * (float4)((float)qs[0], (float)qs[1], (float)qs[2], (float)qs[3]);
}

// V lane dequant (q4_0). Returns block elements lane*4 .. lane*4+3, matching
// dequant_q8_0_lane's contract, so the (block_idx = i/8, lane = i%8) addressing
// at the call sites is unchanged.
//
// Elements lane*4.. for lane 0..3 are all < 16 -> low nibbles of qs[lane*4..].
// For lane 4..7 they are 16.. -> high nibbles of qs[(lane-4)*4..]. Both cases
// read the same byte offset (lane & 3) * 4 and differ only in the nibble, so
// this is a shift select rather than a branch.
inline float4 dequant_q4_0_lane(const global char * block_ptr, int lane) {
    const float d = vload_half(0, (const global half *)block_ptr);
    const global uchar * qs = (const global uchar *)(block_ptr + 2) + (lane & 3) * 4;
    const int sh = (lane >> 2) * 4;   // 0 for low nibble, 4 for high

    return d * (float4)((float)(int)((qs[0] >> sh) & 0x0F) - 8.0f,
                        (float)(int)((qs[1] >> sh) & 0x0F) - 8.0f,
                        (float)(int)((qs[2] >> sh) & 0x0F) - 8.0f,
                        (float)(int)((qs[3] >> sh) & 0x0F) - 8.0f);
}

// Timing probes for the q8_0 decode gap. q8_0 KV moves HALF the bytes of f16
// and is ~20% slower at depth, and the cluster kernels are structurally
// identical apart from these two loads -- so the cost is in here, but the
// block layout hides two candidates behind each other:
//
//   the 2-byte scale is loaded per LANE while only 2 distinct scales exist per
//   cluster row (dk=64, FA_CL_C=16 => 8 lanes share a block), and
//
//   the payload sits at block+2, so every 4-byte quartet read is 2-byte
//   misaligned, where the f16 twin reads aligned float4s.
//
// Both probes are numerically WRONG on purpose. They keep the control flow,
// the shuffle chain and the mad count identical, so the delta is the load
// being removed and nothing else. Same method as GGML_OPENCL_FA16_PROBE_NO_LDS.
//
//   FA_Q8_PROBE_NOSCALE ? drop the scale load only (payload still read).
//   FA_Q8_PROBE_NODEQ   ? drop the scale AND the payload (ceiling: what the
//                         kernel costs with the KV read removed entirely).
inline float4 dequant_q8_0_lane_probe(const global char * block_ptr, int lane) {
#if defined(FA_Q8_PROBE_NODEQ)
    return (float4)(1.0f, 1.0f, 1.0f, 1.0f);
#elif defined(FA_Q8_PROBE_NOSCALE)
    const global char * qs = block_ptr + 2 + lane * 4;
    return (float4)((float)qs[0], (float)qs[1], (float)qs[2], (float)qs[3]);
#else
    return dequant_q8_0_lane(block_ptr, lane);
#endif
}

// V-side twin of the probe above. The probe variants are deliberately wrong
// numerically; they exist to price the load, so the q4_0 payload probe reads the
// low nibble without the scale or the -8 bias for the same reason.
inline float4 dequant_q4_0_lane_probe(const global char * block_ptr, int lane) {
#if defined(FA_Q8_PROBE_NODEQ)
    return (float4)(1.0f, 1.0f, 1.0f, 1.0f);
#elif defined(FA_Q8_PROBE_NOSCALE)
    const global uchar * qs = (const global uchar *)(block_ptr + 2) + (lane & 3) * 4;
    const int sh = (lane >> 2) * 4;
    return (float4)((float)(int)((qs[0] >> sh) & 0x0F),
                    (float)(int)((qs[1] >> sh) & 0x0F),
                    (float)(int)((qs[2] >> sh) & 0x0F),
                    (float)(int)((qs[3] >> sh) & 0x0F));
#else
    return dequant_q4_0_lane(block_ptr, lane);
#endif
}

// dp4a QK dot for the decode (q1) kernels: staged Q rows are requantized to
// packed int8 once per WG (same requantization as the q1_split int path), so
// the KV sweep replaces dequant+float-mad with one integer dot per quartet.
// Requires each lane's quartet index to fit one sweep (DK_VEC <= subgroup).
#if defined(FA_HAVE_INT_DOT) && (DK_VEC <= FA_SG) && !defined(FA_Q8_INT_QK_OFF)
#define FA_Q8_INT_QK 1
#endif

REQD_SUBGROUP_SIZE_64
__kernel void flash_attn_f32_q8_0_q4_0_q1_vec(
    const global void * q_void, ulong q_offset,
    const global void * k_void, ulong k_offset,
    const global void * v_void, ulong v_offset,
    global void * o_void, ulong o_offset,
    const float scale,
    const int n_q,
    const int n_kv,
    const int is_causal,
    const int n_head,
    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
    const ulong k_nb1, const ulong k_nb2, const ulong k_nb3,
    const ulong v_nb1, const ulong v_nb2, const ulong v_nb3,
    const ulong o_nb1, const ulong o_nb2, const ulong o_nb3,
    const float max_bias,
    const float m0,
    const float m1,
    const int n_head_log2,
    const float logit_softcap,
    const int n_head_kv,
    const global void* mask_void,
    const ulong mask_offset,
    const ulong mask_nb1,
    const ulong mask_nb2,
    const ulong mask_nb3,
    const int mask_ne2,
    const int mask_ne3,
    const global void* sinks_void,
    const ulong sinks_offset
) {
    const int tid             = get_local_id(0);
    const int sgid            = tid / Q1_WG_SIZE;
    const int tid_sg          = tid % Q1_WG_SIZE;
    const int head_batch_idx  = get_global_id(1);

    const int batch_idx = head_batch_idx / n_head;
    const int head_idx  = head_batch_idx % n_head;

    const int gqa_ratio   = n_head / n_head_kv;
    const int head_kv_idx = head_idx / gqa_ratio;

    const global char * q_base = (const global char *) q_void + q_offset;
    const global char * k_base = (const global char *) k_void + k_offset;
    const global char * v_base = (const global char *) v_void + v_offset;
    global       char * o_base = (global       char *) o_void + o_offset;

    const global char * mask_base = NULL;
    if (mask_void != NULL) {
        const int mask_head_idx  = head_idx  % mask_ne2;
        const int mask_batch_idx = batch_idx % mask_ne3;
        mask_base = (const global char *) mask_void + mask_offset +
                    mask_batch_idx * mask_nb3 + mask_head_idx * mask_nb2;
    }

    __local ACC_TYPE4 q_shared[DK_VEC];
    {
        const ulong q_row_offset = batch_idx * q_nb3 + head_idx * q_nb2;
        const global Q_DATA_TYPE4 * q_ptr = (const global Q_DATA_TYPE4 *) (q_base + q_row_offset);
        for (int i = tid; i < DK_VEC; i += VEC_WG_SIZE) {
            q_shared[i] = CONVERT_Q_ACC4(q_ptr[i]);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const float slope = get_alibi_slope(max_bias, head_idx, n_head_log2, m0, m1);

    const global ACC_TYPE * sinks_ptr = NULL;
    if (sinks_void != NULL) {
        sinks_ptr = (const global ACC_TYPE *) ((const global char *) sinks_void + sinks_offset);
    }

    ACC_TYPE4 o_acc[Q1V_DV_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < Q1V_DV_PER_THREAD; ++i) o_acc[i] = (ACC_TYPE4)(0.0f);

    ACC_TYPE m_i = FA_M_INIT;
    ACC_TYPE l_i = 0.0f;

    const int kv_per_sg = (n_kv + VEC_NSG - 1) / VEC_NSG;
    const int kv_start  = sgid * kv_per_sg;
    const int kv_end    = min(n_kv, kv_start + kv_per_sg);

    for (int k_idx = kv_start; k_idx < kv_end; ++k_idx) {
        const global char * k_row = k_base + batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_idx * k_nb1;
        const global char * v_row = v_base + batch_idx * v_nb3 + head_kv_idx * v_nb2 + k_idx * v_nb1;

        ACC_TYPE4 dot4 = (ACC_TYPE4)(0.0f);
        for (int qk = tid_sg; qk < DK_VEC; qk += Q1_WG_SIZE) {
            const int block_idx = qk / 8;
            const int lane      = qk % 8;
            const float4 k_v = dequant_q8_0_lane(k_row + block_idx * Q8_0_BLOCK_SIZE, lane);
            dot4 = mad(q_shared[qk], k_v, dot4);
        }
        ACC_TYPE dot_partial = dot4.s0 + dot4.s1 + dot4.s2 + dot4.s3;
        ACC_TYPE score = sub_group_reduce_add(dot_partial) * scale;

        if (mask_base != NULL) {
            const global MASK_DATA_TYPE * mask_ptr = (const global MASK_DATA_TYPE *) mask_base;
            score += slope * (ACC_TYPE) mask_ptr[k_idx];
        }
        if (logit_softcap > 0.0f) {
            score = logit_softcap * tanh(score / logit_softcap);
        }

        const ACC_TYPE m_new      = max(m_i, score);
        const ACC_TYPE scale_prev = native_exp(m_i - m_new);
        const ACC_TYPE p          = native_exp(score - m_new);

        int idx = 0;
        for (int dv = tid_sg; dv < DV_VEC; dv += Q1_WG_SIZE, ++idx) {
            const int block_idx = dv / 8;
            const int lane      = dv % 8;
            const float4 v_v = dequant_q4_0_lane(v_row + block_idx * Q4_0_BLOCK_SIZE, lane);
            o_acc[idx] = mad(p, v_v, o_acc[idx] * scale_prev);
        }
        l_i = l_i * scale_prev + p;
        m_i = m_new;
    }

    __local ACC_TYPE  sg_m[VEC_NSG];
    __local ACC_TYPE  sg_l[VEC_NSG];
    __local ACC_TYPE4 sg_o[VEC_NSG][DV_VEC];

    if (tid_sg == 0) {
        sg_m[sgid] = m_i;
        sg_l[sgid] = l_i;
    }
    {
        int idx = 0;
        for (int dv = tid_sg; dv < DV_VEC; dv += Q1_WG_SIZE, ++idx) {
            sg_o[sgid][dv] = o_acc[idx];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (sgid == 0) {
        ACC_TYPE m_final = sg_m[0];
        #pragma unroll
        for (int s = 1; s < VEC_NSG; ++s) {
            m_final = max(m_final, sg_m[s]);
        }
        if (sinks_ptr != NULL) {
            m_final = max(m_final, sinks_ptr[head_idx]);
        }

        ACC_TYPE l_final = 0.0f;
        #pragma unroll
        for (int s = 0; s < VEC_NSG; ++s) {
            l_final += sg_l[s] * native_exp(sg_m[s] - m_final);
        }
        if (sinks_ptr != NULL) {
            l_final += native_exp(sinks_ptr[head_idx] - m_final);
        }
        const ACC_TYPE l_inv = (l_final > 0.0f) ? (1.0f / l_final) : 0.0f;

        const ulong o_row_offset = batch_idx * o_nb3 + head_idx * o_nb1;
        global O_DATA_TYPE4 * o_row = (global O_DATA_TYPE4 *) (o_base + o_row_offset);

        int idx = 0;
        for (int dv = tid_sg; dv < DV_VEC; dv += Q1_WG_SIZE, ++idx) {
            ACC_TYPE4 o_merged = (ACC_TYPE4)(0.0f);
            #pragma unroll
            for (int s = 0; s < VEC_NSG; ++s) {
                const ACC_TYPE alpha = native_exp(sg_m[s] - m_final);
                o_merged = mad((ACC_TYPE4)(alpha), sg_o[s][dv], o_merged);
            }
            o_row[dv] = CONVERT_O_DATA4(o_merged * l_inv);
        }
    }
}

// Flash-decoding split pass for q8_0 KV. Partial record: [m, l, O[DV]].
// Merge kernel from flash_attn_f32_f16.cl is type-agnostic and reused.
#define FA_PARTIAL_FLOATS (2 + DV)

__kernel void flash_attn_f32_q8_0_q4_0_q1_split(
    const global void * q_void, ulong q_offset,
    const global void * k_void, ulong k_offset,
    const global void * v_void, ulong v_offset,
    const float scale,
    const int n_q,
    const int n_kv,
    const int n_head,
    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
    const ulong k_nb1, const ulong k_nb2, const ulong k_nb3,
    const ulong v_nb1, const ulong v_nb2, const ulong v_nb3,
    const float max_bias,
    const float m0,
    const float m1,
    const int n_head_log2,
    const float logit_softcap,
    const int n_head_kv,
    const global void * mask_void,
    const ulong mask_offset,
    const ulong mask_nb1,
    const ulong mask_nb2,
    const ulong mask_nb3,
    const int mask_ne2,
    const int mask_ne3,
    global float * partial_void,
    const int n_splits,
    const int kv_per_split
) {
    const int tid            = get_local_id(0);
    const int head_batch_idx = get_global_id(1);
    const int split_q_idx    = get_global_id(2);
    const int split_idx      = split_q_idx % n_splits;
    const int q_idx          = split_q_idx / n_splits;
    const int batch_idx      = head_batch_idx / n_head;
    const int head_idx       = head_batch_idx % n_head;
    const int gqa_ratio      = n_head / n_head_kv;
    const int head_kv_idx    = head_idx / gqa_ratio;

    const int kv_start = split_idx * kv_per_split;
    const int kv_end   = min(kv_start + kv_per_split, n_kv);

    const ulong record_stride = (ulong) FA_PARTIAL_FLOATS;
    const ulong record_idx    = ((((ulong) batch_idx * n_head + head_idx) * n_q + q_idx)
                                 * n_splits + split_idx);
    global float  * rec       = partial_void + record_idx * record_stride;
    global float4 * rec_o     = (global float4 *) (rec + 2);

    if (kv_start >= kv_end) {
        // Empty split: leave sentinel partial for merge.
        if (tid == 0) {
            rec[0] = FA_M_INIT;
            rec[1] = 0.0f;
        }
        return;
    }

    const global char * q_base = (const global char *) q_void + q_offset;
    const global char * k_base = (const global char *) k_void + k_offset;
    const global char * v_base = (const global char *) v_void + v_offset;

    const global char * mask_base = NULL;
    if (mask_void != NULL) {
        const int mask_head_idx  = head_idx  % mask_ne2;
        const int mask_batch_idx = batch_idx % mask_ne3;
        mask_base = (const global char *) mask_void + mask_offset +
                    mask_batch_idx * mask_nb3 + mask_head_idx * mask_nb2 +
                    (ulong) q_idx * mask_nb1;
    }

    ACC_TYPE4 q_priv[DK_VEC];
    const ulong q_row_offset = batch_idx * q_nb3 + head_idx * q_nb2 + (ulong) q_idx * q_nb1;
    const global Q_DATA_TYPE4 * q_ptr = (const global Q_DATA_TYPE4 *) (q_base + q_row_offset);
    #pragma unroll
    for (int i = 0; i < DK_VEC; ++i) {
        q_priv[i] = CONVERT_Q_ACC4(q_ptr[i]);
    }

#ifdef FA_HAVE_INT_DOT
    uint  q_packed[DK_Q8_BLOCKS * 8];
    float q_d_scale[DK_Q8_BLOCKS];
    #pragma unroll
    for (int b = 0; b < DK_Q8_BLOCKS; ++b) {
        q_d_scale[b] = quant_q_block_int8_packed(&q_priv[b * 8], &q_packed[b * 8]);
    }
#endif

    const float slope = get_alibi_slope(max_bias, head_idx, n_head_log2, m0, m1);

    // One-pass online softmax (FA-2): single sweep over the split's K range,
    // updating per-thread (m_i, l_i, o_acc) per position. Eliminates the
    // second K read of the original two-pass implementation.
    ACC_TYPE m_i = FA_M_INIT;
    ACC_TYPE l_i = 0.0f;
    ACC_TYPE4 o_acc[DV_VEC];
    #pragma unroll
    for (int i = 0; i < DV_VEC; ++i) o_acc[i] = (ACC_TYPE4)(0.0f);

    for (int k_idx = kv_start + tid; k_idx < kv_end; k_idx += Q1_WG_SIZE) {
        const global char * k_row = k_base + batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_idx * k_nb1;
        const global char * v_row = v_base + batch_idx * v_nb3 + head_kv_idx * v_nb2 + k_idx * v_nb1;
        ACC_TYPE score = 0.0f;
        #pragma unroll
        for (int b = 0; b < DK_Q8_BLOCKS; ++b) {
#ifdef FA_HAVE_INT_DOT
            score += dot_q8_0_int(k_row + b * Q8_0_BLOCK_SIZE, &q_packed[b * 8], q_d_scale[b]);
#else
            score += dot_q8_0_f32(k_row + b * Q8_0_BLOCK_SIZE, &q_priv[b * 8]);
#endif
        }
        score *= scale;
        if (mask_base != NULL) {
            const global MASK_DATA_TYPE * mask_ptr = (const global MASK_DATA_TYPE *) (mask_base);
            score += slope * (ACC_TYPE) mask_ptr[k_idx];
        }
        if (logit_softcap > 0.0f) {
            score = logit_softcap * tanh(score / logit_softcap);
        }

        // Online softmax step.
        const ACC_TYPE m_new = max(m_i, score);
        const ACC_TYPE alpha = exp(m_i  - m_new);
        const ACC_TYPE p     = exp(score - m_new);

        l_i = alpha * l_i + p;
        #pragma unroll
        for (int i = 0; i < DV_VEC; ++i) o_acc[i] *= alpha;

        #pragma unroll
        for (int b = 0; b < DV_Q4_BLOCKS; ++b) {
            ACC_TYPE4 v_dequant[8];
            dequant_q4_0_f32(v_row + b * Q4_0_BLOCK_SIZE, v_dequant);
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                o_acc[b * 8 + i] = mad(p, v_dequant[i], o_acc[b * 8 + i]);
            }
        }

        m_i = m_new;
    }

    // Cross-thread reduce: max(m_i) -> m_c, then rescale per-thread l_i and
    // o_acc by alpha = exp(m_i_thread - m_c) before sum-reduce.
    __local ACC_TYPE local_m[Q1_WG_SIZE];
    local_m[tid] = m_i;
    barrier(CLK_LOCAL_MEM_FENCE);
    #pragma unroll
    for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) local_m[tid] = max(local_m[tid], local_m[tid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const ACC_TYPE m_c = local_m[0];

    const ACC_TYPE alpha_final = exp(m_i - m_c);
    l_i *= alpha_final;
    #pragma unroll
    for (int i = 0; i < DV_VEC; ++i) o_acc[i] *= alpha_final;

    __local ACC_TYPE  local_l[Q1_WG_SIZE];
    __local ACC_TYPE4 local_o[Q1_WG_SIZE];
    local_l[tid] = l_i;
    barrier(CLK_LOCAL_MEM_FENCE);
    #pragma unroll
    for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) local_l[tid] += local_l[tid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const ACC_TYPE l_c = local_l[0];

    if (tid == 0) {
        rec[0] = (float) m_c;
        rec[1] = (float) l_c;
    }
    for (int i = 0; i < DV_VEC; ++i) {
        local_o[tid] = o_acc[i];
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
            if (tid < s) local_o[tid] += local_o[tid + s];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (tid == 0) {
            rec_o[i] = local_o[0];
        }
    }
}

// Prefill: q8_0 K/V, n_q > 1. BLOCK_M ? BLOCK_N tiling.
// K path keeps packed int8 in local for dp4a QK dot; V path dequant -> half in local.
// Requires DK % QK8_0 == 0 and DV % QK8_0 == 0 (gated in supports_op).
#define KV_DATA_TYPE4 half4
#define CONVERT_KV_ACC4(x) convert_float4(x)

#define DK_Q8_BLOCKS_PREFILL (DK / QK8_0)
#define DV_Q8_BLOCKS_PREFILL (DV / QK8_0)

// N_SPLIT>1 splits DK/DV across N_SPLIT threads per query row; needs
// sub_group_shuffle_xor and DK_Q8_BLOCKS_PREFILL % N_SPLIT == 0.
#ifndef N_SPLIT
#define N_SPLIT 1
#endif

#if N_SPLIT > 1
#define SPLIT_DK_VEC        (DK_VEC / N_SPLIT)
#define SPLIT_DV_VEC        (DV_VEC / N_SPLIT)
#define SPLIT_DK_Q8_BLOCKS  (DK_Q8_BLOCKS_PREFILL / N_SPLIT)
#define WG_SIZE             (BLOCK_M * N_SPLIT)
#else
#define SPLIT_DK_VEC        DK_VEC
#define SPLIT_DV_VEC        DV_VEC
#define SPLIT_DK_Q8_BLOCKS  DK_Q8_BLOCKS_PREFILL
#define WG_SIZE             BLOCK_M
#endif

// FA_V_STRATEGY: 0 = dequant V to half in local (default); 2 = keep packed
// int8 in local, dequant in the accumulate loop (smaller local, slightly slower).
#ifndef FA_V_STRATEGY
#define FA_V_STRATEGY 0
#endif
// Strategy 2 stages V as packed int8 quartets in local memory and unpacks them
// inline. That packing is q8_0-specific -- q4_0 carries two elements per byte,
// so the same uint would hold eight values, not four, and the unpack below would
// silently read the wrong V. Rather than let the knob produce a wrong answer,
// refuse it: this kernel supports the dequant-to-half staging only.
#if FA_V_STRATEGY == 2
#error "FA_V_STRATEGY=2 is q8_0-only; V is q4_0 in this kernel. Use FA_V_STRATEGY=0."
#endif

// ---------------------------------------------------------------------------
// flash_attn_f32_q8_0_q1_vec_mq_split ? Multi-Query KV-head-coalesced FA decode
// with flash-decoding split for q8_0 KV. Structural port of
// flash_attn_f32_f16_q1_vec_mq_split (in flash_attn_f32_f16.cl); each WG handles
// MQ_GQA Q-heads sharing one KV-head, so K and V are read once and reused
// MQ_GQA times. Per-h state (m_i, l_i, o_acc, slope, mask_base) tracked
// separately.
//
// Compile-time params:
//   MQ_GQA       ? Q-heads coalesced per WG (defaults 4; second compile with =8
//                  for Qwen3-30B-A3B class).
//   MQ_NSG_SPLIT ? subgroups per WG (defaults 4; second compile with =3 paired
//                  with MQ_GQA=8 on Adreno X2 where the per-kernel WG cap drops
//                  to 192 due to register pressure).
//
// Writes one partial record per (batch, head_idx, q_idx, split_idx) into
// partial_void; flash_attn_f32_merge (below) normalizes across splits.
// ---------------------------------------------------------------------------

#ifndef MQ_GQA
#define MQ_GQA 4
#endif
#ifndef MQ_NSG_SPLIT
#define MQ_NSG_SPLIT 4
#endif
#define MQ_SPLIT_WG_SIZE_Q8 (Q1_WG_SIZE * MQ_NSG_SPLIT)

REQD_SUBGROUP_SIZE_64
__kernel void flash_attn_f32_q8_0_q4_0_q1_vec_mq_split(
    const global void * q_void, ulong q_offset,
    const global void * k_void, ulong k_offset,
    const global void * v_void, ulong v_offset,
    const float scale,
    const int n_q,
    const int n_kv,
    const int n_head,
    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
    const ulong k_nb1, const ulong k_nb2, const ulong k_nb3,
    const ulong v_nb1, const ulong v_nb2, const ulong v_nb3,
    const float max_bias,
    const float m0,
    const float m1,
    const int n_head_log2,
    const float logit_softcap,
    const int n_head_kv,
    const global void * mask_void,
    const ulong mask_offset,
    const ulong mask_nb1,
    const ulong mask_nb2,
    const ulong mask_nb3,
    const int mask_ne2,
    const int mask_ne3,
    global float * partial_void,
    const int n_splits,
    const int kv_per_split
) {
    const int tid              = get_local_id(0);
    const int sgid             = tid / Q1_WG_SIZE;
    const int tid_sg           = tid % Q1_WG_SIZE;
    const int kvhead_batch_idx = get_global_id(1);
    const int split_q_idx      = get_global_id(2);
    const int split_idx        = split_q_idx % n_splits;
    const int q_idx            = split_q_idx / n_splits;

    const int batch_idx   = kvhead_batch_idx / n_head_kv;
    const int head_kv_idx = kvhead_batch_idx % n_head_kv;

    const int kv_start = split_idx * kv_per_split;
    const int kv_end   = min(kv_start + kv_per_split, n_kv);

    const ulong record_stride = (ulong) FA_PARTIAL_FLOATS;

    if (kv_start >= kv_end) {
        // Empty split ? write sentinel for each of the MQ_GQA Q-heads.
        if (tid == 0) {
            #pragma unroll
            for (int h = 0; h < MQ_GQA; ++h) {
                const int head_idx = head_kv_idx * MQ_GQA + h;
                const ulong rec_idx = ((((ulong) batch_idx * n_head + head_idx) * n_q + q_idx)
                                       * n_splits + split_idx);
                global float * rec = partial_void + rec_idx * record_stride;
                rec[0] = FA_M_INIT;
                rec[1] = 0.0f;
            }
        }
        return;
    }

    const global char * q_base = (const global char *) q_void + q_offset;
    const global char * k_base = (const global char *) k_void + k_offset;
    const global char * v_base = (const global char *) v_void + v_offset;

    // Stage MQ_GQA Q rows in __local as float4 once (uniform across WG).
    __local ACC_TYPE4 q_shared[MQ_GQA * DK_VEC];
    for (int i = tid; i < MQ_GQA * DK_VEC; i += MQ_SPLIT_WG_SIZE_Q8) {
        const int h        = i / DK_VEC;
        const int k        = i % DK_VEC;
        const int head_idx = head_kv_idx * MQ_GQA + h;
        const ulong q_row_offset = batch_idx * q_nb3 + head_idx * q_nb2 + (ulong) q_idx * q_nb1;
        const global Q_DATA_TYPE4 * q_ptr = (const global Q_DATA_TYPE4 *) (q_base + q_row_offset);
        q_shared[h * DK_VEC + k] = CONVERT_Q_ACC4(q_ptr[k]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

#ifdef FA_Q8_INT_QK
    // Requantize the staged Q rows to packed int8 once per WG: one thread per
    // (head, block). The KV sweep then runs dp4a against raw q8_0 K bytes.
    __local uint  q_packed_l[MQ_GQA * DK_Q8_BLOCKS * 8];
    __local float q_d_l[MQ_GQA * DK_Q8_BLOCKS];
    for (int i = tid; i < MQ_GQA * DK_Q8_BLOCKS; i += MQ_SPLIT_WG_SIZE_Q8) {
        const int h = i / DK_Q8_BLOCKS;
        const int b = i % DK_Q8_BLOCKS;
        ACC_TYPE4 qb[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            qb[j] = q_shared[h * DK_VEC + b * 8 + j];
        }
        uint packed_tmp[8];
        q_d_l[i] = quant_q_block_int8_packed(qb, packed_tmp);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            q_packed_l[i * 8 + j] = packed_tmp[j];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

    float slope[MQ_GQA];
    #pragma unroll
    for (int h = 0; h < MQ_GQA; ++h) {
        slope[h] = get_alibi_slope(max_bias, head_kv_idx * MQ_GQA + h, n_head_log2, m0, m1);
    }

    const global char * mask_base[MQ_GQA];
    if (mask_void != NULL) {
        const int mask_batch_idx = batch_idx % mask_ne3;
        const global char * mask_base_b = (const global char *) mask_void + mask_offset +
                                          mask_batch_idx * mask_nb3 +
                                          (ulong) q_idx * mask_nb1;
        #pragma unroll
        for (int h = 0; h < MQ_GQA; ++h) {
            const int head_idx      = head_kv_idx * MQ_GQA + h;
            const int mask_head_idx = head_idx % mask_ne2;
            mask_base[h] = mask_base_b + mask_head_idx * mask_nb2;
        }
    } else {
        #pragma unroll
        for (int h = 0; h < MQ_GQA; ++h) mask_base[h] = NULL;
    }

    ACC_TYPE4 o_acc[MQ_GQA][Q1V_DV_PER_THREAD];
    ACC_TYPE  m_i[MQ_GQA];
    ACC_TYPE  l_i[MQ_GQA];
    #pragma unroll
    for (int h = 0; h < MQ_GQA; ++h) {
        m_i[h] = FA_M_INIT;
        l_i[h] = 0.0f;
        #pragma unroll
        for (int i = 0; i < Q1V_DV_PER_THREAD; ++i) o_acc[h][i] = (ACC_TYPE4)(0.0f);
    }

    // Each subgroup sweeps its slice of this split's kv range.
    const int kv_len    = kv_end - kv_start;
    const int kv_per_sg = (kv_len + MQ_NSG_SPLIT - 1) / MQ_NSG_SPLIT;
    const int kv_lo     = kv_start + sgid * kv_per_sg;
    const int kv_hi     = min(kv_end, kv_lo + kv_per_sg);

#ifdef FA_Q8_INT_QK
    // DK_VEC <= subgroup size: this lane's quartet index is fixed for the
    // whole sweep, so its packed-Q words and block scales hoist to registers.
    uint  qp_h[MQ_GQA];
    float qd_h[MQ_GQA];
    {
        const int b = tid_sg / 8;
        const int j = tid_sg % 8;
        #pragma unroll
        for (int h = 0; h < MQ_GQA; ++h) {
            if (tid_sg < DK_VEC) {
                qp_h[h] = q_packed_l[(h * DK_Q8_BLOCKS + b) * 8 + j];
                qd_h[h] = q_d_l[h * DK_Q8_BLOCKS + b];
            } else {
                qp_h[h] = 0;
                qd_h[h] = 0.0f;
            }
        }
    }
#endif

    for (int k_idx = kv_lo; k_idx < kv_hi; ++k_idx) {
        const global char * k_row = k_base + batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_idx * k_nb1;
        const global char * v_row = v_base + batch_idx * v_nb3 + head_kv_idx * v_nb2 + k_idx * v_nb1;

#ifdef FA_Q8_INT_QK
        ACC_TYPE dot_s[MQ_GQA];
        #pragma unroll
        for (int h = 0; h < MQ_GQA; ++h) dot_s[h] = 0.0f;
        if (tid_sg < DK_VEC) {
            const global char * kb = k_row + (tid_sg / 8) * Q8_0_BLOCK_SIZE;
            const float kd       = vload_half(0, (const global half *) kb);
            const uint  k_packed = as_uint(vload4(tid_sg % 8, (const global uchar *)(kb + 2)));
            #pragma unroll
            for (int h = 0; h < MQ_GQA; ++h) {
                const int idot = dot_acc_sat_4x8packed_ss_int(qp_h[h], k_packed, 0);
                dot_s[h] = mad((ACC_TYPE) idot, qd_h[h] * kd, dot_s[h]);
            }
        }
#else
        ACC_TYPE4 dot4[MQ_GQA];
        #pragma unroll
        for (int h = 0; h < MQ_GQA; ++h) dot4[h] = (ACC_TYPE4)(0.0f);

        for (int qk = tid_sg; qk < DK_VEC; qk += Q1_WG_SIZE) {
            const int block_idx = qk / 8;
            const int lane      = qk % 8;
            const float4 k_v = dequant_q8_0_lane(k_row + block_idx * Q8_0_BLOCK_SIZE, lane);
            #pragma unroll
            for (int h = 0; h < MQ_GQA; ++h) {
                dot4[h] = mad(q_shared[h * DK_VEC + qk], k_v, dot4[h]);
            }
        }
#endif

        ACC_TYPE score[MQ_GQA];
        #pragma unroll
        for (int h = 0; h < MQ_GQA; ++h) {
#ifdef FA_Q8_INT_QK
            ACC_TYPE s = sub_group_reduce_add(dot_s[h]) * scale;
#else
            const ACC_TYPE dot_partial = dot4[h].s0 + dot4[h].s1 + dot4[h].s2 + dot4[h].s3;
            ACC_TYPE s = sub_group_reduce_add(dot_partial) * scale;
#endif
            if (mask_base[h] != NULL) {
                const global MASK_DATA_TYPE * mask_ptr = (const global MASK_DATA_TYPE *) mask_base[h];
                s += slope[h] * (ACC_TYPE) mask_ptr[k_idx];
            }
            if (logit_softcap > 0.0f) {
                s = logit_softcap * tanh(s / logit_softcap);
            }
            score[h] = s;
        }

        ACC_TYPE p_h[MQ_GQA];
        ACC_TYPE sp_h[MQ_GQA];
        #pragma unroll
        for (int h = 0; h < MQ_GQA; ++h) {
            const ACC_TYPE m_new = max(m_i[h], score[h]);
            sp_h[h] = native_exp(m_i[h] - m_new);
            p_h[h]  = native_exp(score[h] - m_new);
            l_i[h]  = l_i[h] * sp_h[h] + p_h[h];
            m_i[h]  = m_new;
        }

        int idx = 0;
        for (int dv = tid_sg; dv < DV_VEC; dv += Q1_WG_SIZE, ++idx) {
            const int block_idx = dv / 8;
            const int lane      = dv % 8;
            const float4 v_v = dequant_q4_0_lane(v_row + block_idx * Q4_0_BLOCK_SIZE, lane);
            #pragma unroll
            for (int h = 0; h < MQ_GQA; ++h) {
                o_acc[h][idx] = mad(p_h[h], v_v, o_acc[h][idx] * sp_h[h]);
            }
        }
    }

    // Per-h cross-subgroup merge: subgroup 0 folds NSG partials into a single
    // (m, l, O) record per Q-head, written to the partial buffer.
    __local ACC_TYPE  sg_m[MQ_GQA][MQ_NSG_SPLIT];
    __local ACC_TYPE  sg_l[MQ_GQA][MQ_NSG_SPLIT];
    __local ACC_TYPE4 sg_o[MQ_NSG_SPLIT][DV_VEC];

    if (tid_sg == 0) {
        #pragma unroll
        for (int h = 0; h < MQ_GQA; ++h) {
            sg_m[h][sgid] = m_i[h];
            sg_l[h][sgid] = l_i[h];
        }
    }

    #pragma unroll
    for (int h = 0; h < MQ_GQA; ++h) {
        {
            int idx = 0;
            for (int dv_idx = tid_sg; dv_idx < DV_VEC; dv_idx += Q1_WG_SIZE, ++idx) {
                sg_o[sgid][dv_idx] = o_acc[h][idx];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (sgid == 0) {
            const int head_idx = head_kv_idx * MQ_GQA + h;

            ACC_TYPE m_c = sg_m[h][0];
            #pragma unroll
            for (int s = 1; s < MQ_NSG_SPLIT; ++s) {
                m_c = max(m_c, sg_m[h][s]);
            }
            ACC_TYPE l_c = 0.0f;
            #pragma unroll
            for (int s = 0; s < MQ_NSG_SPLIT; ++s) {
                l_c += sg_l[h][s] * native_exp(sg_m[h][s] - m_c);
            }

            const ulong rec_idx = ((((ulong) batch_idx * n_head + head_idx) * n_q + q_idx)
                                   * n_splits + split_idx);
            global float  * rec   = partial_void + rec_idx * record_stride;
            global float4 * rec_o = (global float4 *) (rec + 2);

            if (tid_sg == 0) {
                rec[0] = (float) m_c;
                rec[1] = (float) l_c;
            }
            for (int dv_idx = tid_sg; dv_idx < DV_VEC; dv_idx += Q1_WG_SIZE) {
                ACC_TYPE4 o_merged = (ACC_TYPE4)(0.0f);
                #pragma unroll
                for (int s = 0; s < MQ_NSG_SPLIT; ++s) {
                    const ACC_TYPE alpha = native_exp(sg_m[h][s] - m_c);
                    o_merged = mad((ACC_TYPE4)(alpha), sg_o[s][dv_idx], o_merged);
                }
                rec_o[dv_idx] = o_merged;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

// ---------------------------------------------------------------------------
// flash_attn_f32_q8_0_q1_vec_mq_split_c8 ? cluster-parallel variant of the MQ
// split above (structural port of the f16/q4_0 c8 kernels; see
// notes/asus-x2-fa-decode-roofline-20260704.md). FA_CL_NCL clusters of
// FA_CL_C lanes per subgroup, each cluster owning its own KV position stream
// with private online-softmax state. K/V dequant stays the float
// dequant_q8_0_lane path of the baseline (no dp4a staging needed for q8_0
// decode). Uniform trip count + clamped row + FA_M_INIT tail keeps shuffles
// convergent. Same partial format -> unchanged FD merge (sinks included).
// Target: GQA=4 DK=DV=128 quantized-KV long-context decode (o_acc =
// DV_VEC/8 x 4 heads = 256B/lane ? inside the register budget; the GQA=8
// variant of this kernel would be 512B/lane = the measured -71% spill
// pathology, so no g8 program is compiled for it).
// ---------------------------------------------------------------------------

#ifdef HAS_SUBGROUP_SHUFFLE

#ifndef FA_CL_C
#define FA_CL_C 8
#endif

// Workgroups per gqa group (1 = one WG owns all MQ_GQA heads of a KV head).
#ifndef FA_HEAD_SUB
#define FA_HEAD_SUB 1
#endif

// Lane striping requires DK/DV to divide across the cluster (see f16 c8).
#if (DK_VEC % FA_CL_C) == 0 && (DV_VEC % FA_CL_C) == 0
#define FA_CL_NCL  (Q1_WG_SIZE / FA_CL_C)   // clusters (position streams) per subgroup
#define FA_CL_DKQ  (DK_VEC / FA_CL_C)       // K quartets per lane per row
#define FA_CL_DVQ  (DV_VEC / FA_CL_C)       // V quartets (o_acc float4s) per lane per head

// FA_C8_NO_SG_PIN: X1E drops the explicit sub-group pin (slow-codegen
// landmine on the X1 compiler); X2 keeps it. See the f16 c8 note.
#ifdef FA_C8_NO_SG_PIN
#define FA_C8_SG_ATTR_Q8
#else
#define FA_C8_SG_ATTR_Q8 REQD_SUBGROUP_SIZE_64
#endif

FA_C8_SG_ATTR_Q8
__kernel void flash_attn_f32_q8_0_q4_0_q1_vec_mq_split_c8(
    const global void * q_void, ulong q_offset,
    const global void * k_void, ulong k_offset,
    const global void * v_void, ulong v_offset,
    const float scale,
    const int n_q,
    const int n_kv,
    const int n_head,
    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
    const ulong k_nb1, const ulong k_nb2, const ulong k_nb3,
    const ulong v_nb1, const ulong v_nb2, const ulong v_nb3,
    const float max_bias,
    const float m0,
    const float m1,
    const int n_head_log2,
    const float logit_softcap,
    const int n_head_kv,
    const global void * mask_void,
    const ulong mask_offset,
    const ulong mask_nb1,
    const ulong mask_nb2,
    const ulong mask_nb3,
    const int mask_ne2,
    const int mask_ne3,
    global float * partial_void,
    const int n_splits,
    const int kv_per_split
) {
    const int tid              = get_local_id(0);
    const int sgid             = tid / Q1_WG_SIZE;
    const int tid_sg           = tid % Q1_WG_SIZE;
    const int cl               = tid_sg / FA_CL_C;   // cluster id
    const int lic              = tid_sg % FA_CL_C;   // lane in cluster
    const int kvhead_batch_idx = get_global_id(1);
    const int split_q_idx      = get_global_id(2);
    const int split_idx        = split_q_idx % n_splits;
    const int q_idx            = split_q_idx / n_splits;

    // FA_HEAD_SUB > 1 splits the gqa group across that many workgroups, so a
    // gqa=8 model can run an MQ_GQA=4 kernel: half the per-head state per lane
    // (o_acc, m_i, l_i, slope) and twice the grid, at the cost of reading each
    // KV row FA_HEAD_SUB times. At FA_HEAD_SUB == 1 this is the original
    // indexing exactly. Same scheme as the f16 kernel, where it took the
    // dk=64 gqa8 shape from 576 to 368 B/WI -- this fork still sits at 592 for
    // the same shape, which is the measured cause of its decode deficit.
    // head_kv_idx still selects the KV head, so every sub-group reads the same
    // K/V rows; only the Q heads it owns differ.
    const int hgroups     = n_head_kv * FA_HEAD_SUB;
    const int batch_idx   = kvhead_batch_idx / hgroups;
    const int hg          = kvhead_batch_idx % hgroups;
    const int head_kv_idx = hg / FA_HEAD_SUB;
    const int head_sub    = hg % FA_HEAD_SUB;
#define FA_Q8CL_HEAD_IDX(h) (head_kv_idx * (MQ_GQA * FA_HEAD_SUB) + head_sub * MQ_GQA + (h))

    const int kv_start = split_idx * kv_per_split;
    const int kv_end   = min(kv_start + kv_per_split, n_kv);

    const ulong record_stride = (ulong) FA_PARTIAL_FLOATS;

    if (kv_start >= kv_end) {
        if (tid == 0) {
            #pragma unroll
            for (int h = 0; h < MQ_GQA; ++h) {
                const int head_idx = FA_Q8CL_HEAD_IDX(h);
                const ulong rec_idx = ((((ulong) batch_idx * n_head + head_idx) * n_q + q_idx)
                                       * n_splits + split_idx);
                global float * rec = partial_void + rec_idx * record_stride;
                rec[0] = FA_M_INIT;
                rec[1] = 0.0f;
            }
        }
        return;
    }

    const global char * q_base = (const global char *) q_void + q_offset;
    const global char * k_base = (const global char *) k_void + k_offset;
    const global char * v_base = (const global char *) v_void + v_offset;

    // Stage MQ_GQA Q rows in __local once (uniform across WG).
    __local ACC_TYPE4 q_shared[MQ_GQA * DK_VEC];
    for (int i = tid; i < MQ_GQA * DK_VEC; i += MQ_SPLIT_WG_SIZE_Q8) {
        const int h        = i / DK_VEC;
        const int k        = i % DK_VEC;
        const int head_idx = FA_Q8CL_HEAD_IDX(h);
        const ulong q_row_offset = batch_idx * q_nb3 + head_idx * q_nb2 + (ulong) q_idx * q_nb1;
        const global Q_DATA_TYPE4 * q_ptr = (const global Q_DATA_TYPE4 *) (q_base + q_row_offset);
        q_shared[h * DK_VEC + k] = CONVERT_Q_ACC4(q_ptr[k]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

#ifdef FA_Q8_INT_QK
    // Same once-per-WG Q requantization as the mq_split kernel above.
    __local uint  q_packed_l[MQ_GQA * DK_Q8_BLOCKS * 8];
    __local float q_d_l[MQ_GQA * DK_Q8_BLOCKS];
    for (int i = tid; i < MQ_GQA * DK_Q8_BLOCKS; i += MQ_SPLIT_WG_SIZE_Q8) {
        const int h = i / DK_Q8_BLOCKS;
        const int b = i % DK_Q8_BLOCKS;
        ACC_TYPE4 qb[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            qb[j] = q_shared[h * DK_VEC + b * 8 + j];
        }
        uint packed_tmp[8];
        q_d_l[i] = quant_q_block_int8_packed(qb, packed_tmp);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            q_packed_l[i * 8 + j] = packed_tmp[j];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

    float slope[MQ_GQA];
    #pragma unroll
    for (int h = 0; h < MQ_GQA; ++h) {
        slope[h] = get_alibi_slope(max_bias, FA_Q8CL_HEAD_IDX(h), n_head_log2, m0, m1);
    }

#ifdef FA_CL_MASK_BCAST
    // At mask_ne2 == 1 every clustered head reads the same mask row, so keep
    // one base and load the element ONCE PER POSITION rather than once per
    // head -- and, more to the point, issue that load at the top of the KV
    // iteration alongside the K row instead of inside the per-head score loop.
    //
    // In this kernel the mask was a late, scattered, dependent load with
    // nothing to hide it: a probe that deletes it outright
    // (GGML_OPENCL_FA_Q8_PROBE=nomask) measured +8.8% @d4096 / +14.0% @d8192,
    // which is the entire remaining deficit against the f16 kernel.
    const global char * mask_base_b = NULL;
    if (mask_void != NULL) {
        mask_base_b = (const global char *) mask_void + mask_offset +
                      (batch_idx % mask_ne3) * mask_nb3 +
                      (ulong) q_idx * mask_nb1;
    }
    const int mask_bcast = mask_base_b != NULL && mask_ne2 == 1;
    // Heads still need their own row whenever the mask is per-head.
    const global char * mask_base[MQ_GQA];
    #pragma unroll
    for (int h = 0; h < MQ_GQA; ++h) {
        mask_base[h] = (mask_base_b != NULL && !mask_bcast)
            ? mask_base_b + (FA_Q8CL_HEAD_IDX(h) % mask_ne2) * mask_nb2
            : NULL;
    }
#else
    const global char * mask_base[MQ_GQA];
    if (mask_void != NULL) {
        const int mask_batch_idx = batch_idx % mask_ne3;
        const global char * mask_base_b = (const global char *) mask_void + mask_offset +
                                          mask_batch_idx * mask_nb3 +
                                          (ulong) q_idx * mask_nb1;
        #pragma unroll
        for (int h = 0; h < MQ_GQA; ++h) {
            const int head_idx      = FA_Q8CL_HEAD_IDX(h);
            const int mask_head_idx = head_idx % mask_ne2;
            mask_base[h] = mask_base_b + mask_head_idx * mask_nb2;
        }
    } else {
        #pragma unroll
        for (int h = 0; h < MQ_GQA; ++h) mask_base[h] = NULL;
    }
#endif

    // Per-CLUSTER online state; o_acc holds this lane's V quartets {lic + FA_CL_C*i}.
    ACC_TYPE4 o_acc[MQ_GQA][FA_CL_DVQ];
    ACC_TYPE  m_i[MQ_GQA];
    ACC_TYPE  l_i[MQ_GQA];
    #pragma unroll
    for (int h = 0; h < MQ_GQA; ++h) {
        m_i[h] = FA_M_INIT;
        l_i[h] = 0.0f;
        #pragma unroll
        for (int i = 0; i < FA_CL_DVQ; ++i) o_acc[h][i] = (ACC_TYPE4)(0.0f);
    }

    const int kv_len    = kv_end - kv_start;
    const int kv_per_sg = (kv_len + MQ_NSG_SPLIT - 1) / MQ_NSG_SPLIT;
    const int kv_lo     = kv_start + sgid * kv_per_sg;
    const int kv_hi     = min(kv_end, kv_lo + kv_per_sg);

    // Uniform trip count; tail clamps the row address and drops the score to
    // FA_M_INIT (p underflows to 0) so shuffles stay convergent.
    const int n_iter = (kv_hi - kv_lo + FA_CL_NCL - 1) / FA_CL_NCL;
    const ulong k_row_base = batch_idx * k_nb3 + head_kv_idx * k_nb2;
    const ulong v_row_base = batch_idx * v_nb3 + head_kv_idx * v_nb2;

#if defined(FA_CL_MASK_BCAST) && defined(FA_CL_MASK_SG)
    // Subgroup-staged mask.
    //
    // What is left of the mask cost is neither redundancy (the broadcast took
    // that) nor issue distance (pipelining it measured 0, at identical
    // private_mem), so it is the transaction itself: one extra scattered
    // stream touched every iteration.
    //
    // The access pattern removes that for free. Cluster cl at iteration it
    // reads kv_lo + cl + it*FA_CL_NCL, so across FA_CL_C iterations the
    // subgroup's clusters touch exactly Q1_WG_SIZE CONSECUTIVE positions --
    // one per lane. Load the block once, fully coalesced, and redistribute it
    // by shuffle: FA_CL_C scattered loads become 1 coalesced load plus
    // FA_CL_C shuffles.
    //
    // Deliberately NOT staged in LDS: this kernel is occupancy-bound (the
    // head-split's 592 -> 352 B/WI bought +4.6/+7.2%) and that same change
    // doubled the workgroups per KV head, so LDS would be spending exactly
    // the resource the last win came from. Shuffles cost none.
    ACC_TYPE mask_sg = (ACC_TYPE) 0.0f;
#endif

    for (int it = 0; it < n_iter; ++it) {
        const int k_idx  = kv_lo + cl + it * FA_CL_NCL;
        const int valid  = k_idx < kv_hi;
        const int k_safe = valid ? k_idx : (kv_hi - 1);

        const global char * k_row = k_base + k_row_base + (ulong) k_safe * k_nb1;
        const global char * v_row = v_base + v_row_base + (ulong) k_safe * v_nb1;

#if defined(FA_CL_MASK_BCAST) && defined(FA_CL_MASK_SG)
        // Restage every FA_CL_C iterations, then shuffle out this cluster's
        // element. mblk and mask_bcast are subgroup-uniform, so the reload is
        // convergent and the shuffle index (cl + FA_CL_NCL*mblk) covers
        // 0..Q1_WG_SIZE-1 exactly once per block. Tail lanes clamp like
        // k_safe; their iterations are already dropped to FA_M_INIT.
        const int mblk = it % FA_CL_C;
        if (mask_bcast && mblk == 0) {
            const int k_stg = kv_lo + it * FA_CL_NCL + tid_sg;
            mask_sg = (ACC_TYPE) ((const global MASK_DATA_TYPE *) mask_base_b)
                          [k_stg < kv_hi ? k_stg : (kv_hi - 1)];
        }
        const ACC_TYPE mask_val = mask_bcast
            ? sub_group_shuffle(mask_sg, cl + FA_CL_NCL * mblk)
            : (ACC_TYPE) 0.0f;
#elif defined(FA_CL_MASK_BCAST)
        // Issue the broadcast mask load with the K row so both are in flight.
        // This is the point of the change: previously the mask was read inside
        // the per-head score loop, i.e. AFTER the cluster reduce, where its
        // latency is fully exposed.
        ACC_TYPE mask_val = (ACC_TYPE) 0.0f;
        if (mask_bcast) {
            mask_val = (ACC_TYPE) ((const global MASK_DATA_TYPE *) mask_base_b)[k_safe];
        }
#endif

#ifdef FA_Q8_INT_QK
        // dp4a K dot over this lane's quartets of the cluster's row.
        ACC_TYPE dot_s[MQ_GQA];
        #pragma unroll
        for (int h = 0; h < MQ_GQA; ++h) dot_s[h] = 0.0f;
        #pragma unroll
        for (int i = 0; i < FA_CL_DKQ; ++i) {
            const int qk = lic + FA_CL_C * i;
            const int b  = qk / 8;
            const int j  = qk % 8;
            const global char * kb = k_row + b * Q8_0_BLOCK_SIZE;
            const float kd       = vload_half(0, (const global half *) kb);
            const uint  k_packed = as_uint(vload4(j, (const global uchar *)(kb + 2)));
            #pragma unroll
            for (int h = 0; h < MQ_GQA; ++h) {
                const int idot = dot_acc_sat_4x8packed_ss_int(
                    q_packed_l[(h * DK_Q8_BLOCKS + b) * 8 + j], k_packed, 0);
                dot_s[h] = mad((ACC_TYPE) idot, q_d_l[h * DK_Q8_BLOCKS + b] * kd, dot_s[h]);
            }
        }
#else
        // Float-dequant K dot over this lane's quartets of the cluster's row.
        ACC_TYPE4 dot4[MQ_GQA];
        #pragma unroll
        for (int h = 0; h < MQ_GQA; ++h) dot4[h] = (ACC_TYPE4)(0.0f);
        #pragma unroll
        for (int i = 0; i < FA_CL_DKQ; ++i) {
            const int qk = lic + FA_CL_C * i;
            const float4 k_v = dequant_q8_0_lane_probe(k_row + (qk / 8) * Q8_0_BLOCK_SIZE, qk % 8);
            #pragma unroll
            for (int h = 0; h < MQ_GQA; ++h) {
                dot4[h] = mad(q_shared[h * DK_VEC + qk], k_v, dot4[h]);
            }
        }
#endif

        // This lane's partial score per head, before the cluster all-reduce.
        ACC_TYPE mh_p[MQ_GQA];
        #pragma unroll
        for (int h = 0; h < MQ_GQA; ++h) {
#ifdef FA_Q8_INT_QK
            mh_p[h] = dot_s[h];
#else
            mh_p[h] = dot4[h].s0 + dot4[h].s1 + dot4[h].s2 + dot4[h].s3;
#endif
        }

#if defined(FA_CL_MHRED) && MQ_GQA == 4 && FA_CL_C == 16 && FA_CL_DKQ == 1
        // MQ_GQA=4 form, for the head-split program (FA_HEAD_SUB=2 turns a
        // gqa=8 model into two MQ_GQA=4 workgroups). 4 values per lane fold to
        // 1 over 4 lanes (2+1 shuffles), two plain steps finish the 16-lane
        // sum, 3 shuffles expand -- 8 instead of MQ_GQA*log2(C)=16. Increasing
        // xor distance, so scores stay bit-identical.
        const int mh_b0 = lic & 1;
        const int mh_b1 = lic & 2;

        ACC_TYPE mh_r2[2];
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const ACC_TYPE keep = mh_b0 ? mh_p[j + 2] : mh_p[j];
            const ACC_TYPE send = mh_b0 ? mh_p[j]     : mh_p[j + 2];
            mh_r2[j] = keep + sub_group_shuffle_xor(send, 1);
        }
        ACC_TYPE mh_r1 = (mh_b1 ? mh_r2[1] : mh_r2[0]) +
                         sub_group_shuffle_xor(mh_b1 ? mh_r2[0] : mh_r2[1], 2);
        mh_r1 += sub_group_shuffle_xor(mh_r1, 4);
        mh_r1 += sub_group_shuffle_xor(mh_r1, 8);

        ACC_TYPE mh_e2[2];
        {
            const ACC_TYPE other = sub_group_shuffle_xor(mh_r1, 2);
            mh_e2[0] = mh_b1 ? other : mh_r1;
            mh_e2[1] = mh_b1 ? mh_r1 : other;
        }
        ACC_TYPE mh_s[MQ_GQA];
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const ACC_TYPE other = sub_group_shuffle_xor(mh_e2[j], 1);
            mh_s[j]     = mh_b0 ? other    : mh_e2[j];
            mh_s[j + 2] = mh_b0 ? mh_e2[j] : other;
        }
#endif

#if defined(FA_CL_MHRED) && MQ_GQA == 8 && FA_CL_C == 16 && FA_CL_DKQ == 1
        // Multi-head fused cluster reduce, ported from the f16 kernel, which
        // got it in the gpt-oss FD round while this fork did not.
        //
        // The per-head loop below runs MQ_GQA independent all-reduces over
        // FA_CL_C lanes = MQ_GQA * log2(FA_CL_C) = 32 shuffles per KV row --
        // and a probe showed this kernel's cost is NOT its KV read (deleting
        // the read entirely still left it behind f16), so the shuffle chain is
        // what there is to cut. One halving butterfly costs 4+2+1+1 = 8
        // shuffles to reach one head per lane and 1+2+4 = 7 to expand every
        // head back: 15 instead of 32.
        //
        // Rounds run in INCREASING xor distance so each head's summation tree
        // is pairwise identical to the per-head butterfly's => bit-identical
        // scores, not merely mathematically equivalent ones.
        const int mh_b0 = lic & 1;
        const int mh_b1 = lic & 2;
        const int mh_b2 = lic & 4;

        ACC_TYPE mh_r4[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const ACC_TYPE keep = mh_b0 ? mh_p[j + 4] : mh_p[j];
            const ACC_TYPE send = mh_b0 ? mh_p[j]     : mh_p[j + 4];
            mh_r4[j] = keep + sub_group_shuffle_xor(send, 1);
        }
        ACC_TYPE mh_r2[2];
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const ACC_TYPE keep = mh_b1 ? mh_r4[j + 2] : mh_r4[j];
            const ACC_TYPE send = mh_b1 ? mh_r4[j]     : mh_r4[j + 2];
            mh_r2[j] = keep + sub_group_shuffle_xor(send, 2);
        }
        ACC_TYPE mh_r1 = (mh_b2 ? mh_r2[1] : mh_r2[0]) +
                         sub_group_shuffle_xor(mh_b2 ? mh_r2[0] : mh_r2[1], 4);
        mh_r1 += sub_group_shuffle_xor(mh_r1, 8);

        ACC_TYPE mh_e2[2];
        {
            const ACC_TYPE other = sub_group_shuffle_xor(mh_r1, 4);
            mh_e2[0] = mh_b2 ? other : mh_r1;
            mh_e2[1] = mh_b2 ? mh_r1 : other;
        }
        ACC_TYPE mh_e4[4];
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const ACC_TYPE other = sub_group_shuffle_xor(mh_e2[j], 2);
            mh_e4[j]     = mh_b1 ? other    : mh_e2[j];
            mh_e4[j + 2] = mh_b1 ? mh_e2[j] : other;
        }
        ACC_TYPE mh_s[MQ_GQA];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const ACC_TYPE other = sub_group_shuffle_xor(mh_e4[j], 1);
            mh_s[j]     = mh_b0 ? other    : mh_e4[j];
            mh_s[j + 4] = mh_b0 ? mh_e4[j] : other;
        }
#endif

        // Cluster-reduce (xor steps < FA_CL_C stay inside the cluster) + score.
        ACC_TYPE score[MQ_GQA];
        #pragma unroll
        for (int h = 0; h < MQ_GQA; ++h) {
#if defined(FA_CL_MHRED) && (MQ_GQA == 8 || MQ_GQA == 4) && FA_CL_C == 16 && FA_CL_DKQ == 1
            ACC_TYPE s = mh_s[h];
#else
            ACC_TYPE s = mh_p[h];
            #pragma unroll
            for (int step = 1; step < FA_CL_C; step <<= 1) {
                s += sub_group_shuffle_xor(s, step);
            }
#endif
            s *= scale;
            // FA_Q8_PROBE_NOMASK: timing probe for whether FA_CL_MASK_BCAST is
            // worth porting. Every lane of the cluster loads mask_ptr[k_safe]
            // for the same k_safe, so per cluster row this is MQ_GQA*FA_CL_C =
            // 128 loads of 8 distinct values -- the same redundancy shape as
            // the block scale, which measured EXACTLY 0 because the lanes hit
            // one cache line. Numerically wrong; delete the load, keep the
            // control flow, and see whether there is anything there to win.
#ifndef FA_Q8_PROBE_NOMASK
#ifdef FA_CL_MASK_BCAST
            if (mask_bcast) {
                s += slope[h] * mask_val;
            } else
#endif
            if (mask_base[h] != NULL) {
                const global MASK_DATA_TYPE * mask_ptr = (const global MASK_DATA_TYPE *) mask_base[h];
                s += slope[h] * (ACC_TYPE) mask_ptr[k_safe];
            }
#endif
            if (logit_softcap > 0.0f) {
                s = logit_softcap * tanh(s / logit_softcap);
            }
            score[h] = valid ? s : FA_M_INIT;
        }

        // Per-cluster online update (serial chain depth n_iter, not kv_per_sg).
        ACC_TYPE p_h[MQ_GQA];
        ACC_TYPE sp_h[MQ_GQA];
        #pragma unroll
        for (int h = 0; h < MQ_GQA; ++h) {
            const ACC_TYPE m_new = max(m_i[h], score[h]);
            sp_h[h] = native_exp(m_i[h] - m_new);
            p_h[h]  = native_exp(score[h] - m_new);
            l_i[h]  = l_i[h] * sp_h[h] + p_h[h];
            m_i[h]  = m_new;
        }

        // V accumulate on this lane's quartets (p = 0 on tail -> inert).
        #pragma unroll
        for (int i = 0; i < FA_CL_DVQ; ++i) {
            const int dv = lic + FA_CL_C * i;
            const float4 v_v = dequant_q4_0_lane_probe(v_row + (dv / 8) * Q4_0_BLOCK_SIZE, dv % 8);
            #pragma unroll
            for (int h = 0; h < MQ_GQA; ++h) {
                o_acc[h][i] = mad(p_h[h], v_v, o_acc[h][i] * sp_h[h]);
            }
        }
    }

    // Merge stage 1: fold cluster partials inside the subgroup via shuffles.
    #pragma unroll
    for (int h = 0; h < MQ_GQA; ++h) {
        ACC_TYPE m_c = m_i[h];
        #pragma unroll
        for (int step = FA_CL_C; step < Q1_WG_SIZE; step <<= 1) {
            m_c = max(m_c, sub_group_shuffle_xor(m_c, step));
        }
        const ACC_TYPE alpha = native_exp(m_i[h] - m_c);
        ACC_TYPE l_c = l_i[h] * alpha;
        #pragma unroll
        for (int step = FA_CL_C; step < Q1_WG_SIZE; step <<= 1) {
            l_c += sub_group_shuffle_xor(l_c, step);
        }
        #pragma unroll
        for (int i = 0; i < FA_CL_DVQ; ++i) {
            ACC_TYPE4 o = o_acc[h][i] * alpha;
            #pragma unroll
            for (int step = FA_CL_C; step < Q1_WG_SIZE; step <<= 1) {
                o.s0 += sub_group_shuffle_xor(o.s0, step);
                o.s1 += sub_group_shuffle_xor(o.s1, step);
                o.s2 += sub_group_shuffle_xor(o.s2, step);
                o.s3 += sub_group_shuffle_xor(o.s3, step);
            }
            o_acc[h][i] = o;
        }
        m_i[h] = m_c;
        l_i[h] = l_c;
    }

    // Merge stage 2: baseline cross-subgroup LDS merge (o published by
    // cluster 0's lanes; layout identical to the baseline sg_o).
    __local ACC_TYPE  sg_m[MQ_GQA][MQ_NSG_SPLIT];
    __local ACC_TYPE  sg_l[MQ_GQA][MQ_NSG_SPLIT];
    __local ACC_TYPE4 sg_o[MQ_NSG_SPLIT][DV_VEC];

    if (tid_sg == 0) {
        #pragma unroll
        for (int h = 0; h < MQ_GQA; ++h) {
            sg_m[h][sgid] = m_i[h];
            sg_l[h][sgid] = l_i[h];
        }
    }

    #pragma unroll
    for (int h = 0; h < MQ_GQA; ++h) {
        if (cl == 0) {
            #pragma unroll
            for (int i = 0; i < FA_CL_DVQ; ++i) {
                sg_o[sgid][lic + FA_CL_C * i] = o_acc[h][i];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (sgid == 0) {
            const int head_idx = FA_Q8CL_HEAD_IDX(h);

            ACC_TYPE m_c = sg_m[h][0];
            #pragma unroll
            for (int s = 1; s < MQ_NSG_SPLIT; ++s) {
                m_c = max(m_c, sg_m[h][s]);
            }
            ACC_TYPE l_c = 0.0f;
            #pragma unroll
            for (int s = 0; s < MQ_NSG_SPLIT; ++s) {
                l_c += sg_l[h][s] * native_exp(sg_m[h][s] - m_c);
            }

            const ulong rec_idx = ((((ulong) batch_idx * n_head + head_idx) * n_q + q_idx)
                                   * n_splits + split_idx);
            global float  * rec   = partial_void + rec_idx * record_stride;
            global float4 * rec_o = (global float4 *) (rec + 2);

            if (tid_sg == 0) {
                rec[0] = (float) m_c;
                rec[1] = (float) l_c;
            }
            for (int dv_idx = tid_sg; dv_idx < DV_VEC; dv_idx += Q1_WG_SIZE) {
                ACC_TYPE4 o_merged = (ACC_TYPE4)(0.0f);
                #pragma unroll
                for (int s = 0; s < MQ_NSG_SPLIT; ++s) {
                    const ACC_TYPE alpha = native_exp(sg_m[h][s] - m_c);
                    o_merged = mad((ACC_TYPE4)(alpha), sg_o[s][dv_idx], o_merged);
                }
                rec_o[dv_idx] = o_merged;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

#endif  // DK_VEC/DV_VEC divisible by FA_CL_C
#endif  // HAS_SUBGROUP_SHUFFLE (q1_vec_mq_split_c8)

__kernel void flash_attn_f32_q8_0_q4_0(
    const global void * q_void, ulong q_offset,
    const global void * k_void, ulong k_offset,
    const global void * v_void, ulong v_offset,
    global void * o_void, ulong o_offset,
    const float scale,
    const int n_q,
    const int n_kv,
    const int is_causal,
    const int n_head,
    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
    const ulong k_nb1, const ulong k_nb2, const ulong k_nb3,
    const ulong v_nb1, const ulong v_nb2, const ulong v_nb3,
    const ulong o_nb1, const ulong o_nb2, const ulong o_nb3,
    const float max_bias,
    const float m0,
    const float m1,
    const int n_head_log2,
    const float logit_softcap,
    const int n_head_kv,
    const global void* mask_void,
    const ulong mask_offset,
    const ulong mask_nb1,
    const ulong mask_nb2,
    const ulong mask_nb3,
    const int mask_ne2,
    const int mask_ne3,
    const global void* sinks_void,
    const ulong sinks_offset,
    // blk: per-(qblock,kvblock) class from flash_attn_blk_f16
    // (0=masked, 1=mixed, 2=unmasked). NULL disables the prepass opt.
    const global void * blk_void
) {
    const int tid = get_local_id(0);
    const int block_q_idx = get_group_id(0);
    const int head_batch_idx = get_global_id(1);

#if N_SPLIT > 1
    const int q_lane    = tid / N_SPLIT;
    const int split_idx = tid % N_SPLIT;
#else
    const int q_lane    = tid;
    const int split_idx = 0;
#endif
    const int my_query_row = block_q_idx * BLOCK_M + q_lane;
    const int query_valid = my_query_row < n_q;

    const int batch_idx = head_batch_idx / n_head;
    const int head_idx  = head_batch_idx % n_head;

    const int gqa_ratio   = n_head / n_head_kv;
    const int head_kv_idx = head_idx / gqa_ratio;
    const int mask_head_idx  = mask_void != NULL ? head_idx  % mask_ne2 : 0;
    const int mask_batch_idx = mask_void != NULL ? batch_idx % mask_ne3 : 0;

    const global char * q_base = (const global char *) q_void + q_offset;
    const global char * k_base = (const global char *) k_void + k_offset;
    const global char * v_base = (const global char *) v_void + v_offset;
    global       char * o_base = (global       char *) o_void + o_offset;

    const global char * mask_base = NULL;
    if (mask_void != NULL) {
        mask_base = (const global char *) mask_void + mask_offset +
                    mask_batch_idx * mask_nb3 + mask_head_idx * mask_nb2;
    }

    // BLK_PREPASS_BM may differ from this kernel's BLOCK_M; scale q-block idx.
    #ifndef BLK_PREPASS_BM
    #define BLK_PREPASS_BM BLOCK_M
    #endif
    const global char * blk_base = NULL;
    int n_kv_blocks = 0;
    if (blk_void != NULL) {
        n_kv_blocks = (n_kv + BLOCK_N - 1) / BLOCK_N;
        const int n_q_blocks_prepass = (n_q + BLK_PREPASS_BM - 1) / BLK_PREPASS_BM;
        const int prepass_q_block    = (block_q_idx * BLOCK_M) / BLK_PREPASS_BM;
        blk_base = (const global char *) blk_void +
                   (((mask_batch_idx * mask_ne2) + mask_head_idx) * n_q_blocks_prepass + prepass_q_block) * n_kv_blocks;
    }

    const int dk_off_vec = split_idx * SPLIT_DK_VEC;
    ACC_TYPE4 q_priv[SPLIT_DK_VEC];
    if (query_valid) {
        const ulong q_row_offset = batch_idx * q_nb3 + head_idx * q_nb2 + my_query_row * q_nb1;
        const global float4 * q_ptr = (const global float4 *) (q_base + q_row_offset);
        #pragma unroll
        for (int i = 0; i < SPLIT_DK_VEC; ++i) {
            q_priv[i] = q_ptr[dk_off_vec + i];
        }
    } else {
        #pragma unroll
        for (int i = 0; i < SPLIT_DK_VEC; ++i) q_priv[i] = (ACC_TYPE4)(0.0f);
    }

#ifdef FA_HAVE_INT_DOT
    uint  q_packed_pf[SPLIT_DK_Q8_BLOCKS * 8];
    float q_d_pf[SPLIT_DK_Q8_BLOCKS];
    #pragma unroll
    for (int b = 0; b < SPLIT_DK_Q8_BLOCKS; ++b) {
        q_d_pf[b] = quant_q_block_int8_packed(&q_priv[b * 8], &q_packed_pf[b * 8]);
    }
#endif

    const int dv_off_vec = split_idx * SPLIT_DV_VEC;
    ACC_TYPE4 o_acc[SPLIT_DV_VEC];
    #pragma unroll
    for (int i = 0; i < SPLIT_DV_VEC; ++i) o_acc[i] = (ACC_TYPE4)(0.0f);

    ACC_TYPE m_i = FA_M_INIT;
    ACC_TYPE l_i = 0.0f;

    float slope = get_alibi_slope(max_bias, head_idx, n_head_log2, m0, m1);

#ifdef FA_HAVE_INT_DOT
// Accessors so the staging code is layout-agnostic.
#ifdef FA_K_LDS_T
#define FA_K_PACKED(ROW, IDX) l_k_packed[IDX][ROW]
#define FA_K_SCALE(ROW, BLK)  l_k_scale[BLK][ROW]
#else
#define FA_K_PACKED(ROW, IDX) l_k_packed[ROW][IDX]
#define FA_K_SCALE(ROW, BLK)  l_k_scale[ROW][BLK]
#endif

#ifdef FA_K_LDS_T
    // K tile transposed: [block*8 + g][kv row] instead of [kv row][block*8 + g].
    //
    // The QK loop walks 4 KV rows at a time against the same (b, g), so in the original
    // layout those 4 values are BLOCK_N*8 uints apart and cost 4 separate 32-bit local
    // reads. Transposed they are adjacent, so they are one 128-bit read -- 4x fewer LDS
    // issues for the same bytes and no extra registers. That matters because the QK loop
    // is LDS-read-issue-bound: a wrong-math probe that kept every dp4a but cut the LDS
    // reads ran the whole kernel 41% faster (18.51 -> 10.91 ms/op), and deleting QK
    // outright only reached 10.88 -- i.e. essentially ALL of QK's cost is these reads.
    __local uint  l_k_packed[DK_Q8_BLOCKS_PREFILL * 8][BLOCK_N];
    __local float l_k_scale [DK_Q8_BLOCKS_PREFILL][BLOCK_N];
#else
    __local uint  l_k_packed[BLOCK_N][DK_Q8_BLOCKS_PREFILL * 8];
    __local float l_k_scale [BLOCK_N][DK_Q8_BLOCKS_PREFILL];
#endif
#else
    __local half4 l_k[BLOCK_N][DK_VEC];
#endif

#if FA_V_STRATEGY == 2
    __local uint  l_v_packed[BLOCK_N][DV_Q8_BLOCKS_PREFILL * 8];
    __local float l_v_scale [BLOCK_N][DV_Q8_BLOCKS_PREFILL];
#else
    __local half4 l_v[BLOCK_N][DV_VEC];
#endif

    for (int k_start = 0; k_start < n_kv; k_start += BLOCK_N) {
        // Skip fully-masked KV tiles (uniform branch across WG).
        char blk_cur = 1;
        if (blk_base != NULL) {
            blk_cur = blk_base[k_start / BLOCK_N];
            if (blk_cur == 0) continue;
        }

        {
#ifdef FA_HAVE_INT_DOT
            const int k_blocks_per_row = DK_Q8_BLOCKS_PREFILL;
            const int n_blocks_total = BLOCK_N * k_blocks_per_row;
            for (int i = tid; i < n_blocks_total; i += WG_SIZE) {
                const int row = i / k_blocks_per_row;
                const int blk = i % k_blocks_per_row;
                const int k_row_idx = k_start + row;
                if (k_row_idx < n_kv) {
                    const ulong k_row_off = batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_row_idx * k_nb1;
                    const global char * blk_ptr = k_base + k_row_off + blk * Q8_0_BLOCK_SIZE;
                    const float df = (float) vload_half(0, (const global half *) blk_ptr);
                    const global uchar * qs = (const global uchar *)(blk_ptr + 2);
                    FA_K_SCALE(row, blk) = df;
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        uint k_packed =
                              (uint) qs[j*4 + 0]        |
                             ((uint) qs[j*4 + 1]) <<  8 |
                             ((uint) qs[j*4 + 2]) << 16 |
                             ((uint) qs[j*4 + 3]) << 24;
                        FA_K_PACKED(row, blk * 8 + j) = k_packed;
                    }
                } else {
                    FA_K_SCALE(row, blk) = 0.0f;
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) FA_K_PACKED(row, blk * 8 + j) = 0u;
                }
            }
#else
            // Fallback: dequant q8_0 -> half in local memory.
            const int k_blocks_per_row = DK / QK8_0;
            const int n_blocks_total = BLOCK_N * k_blocks_per_row;
            for (int i = tid; i < n_blocks_total; i += WG_SIZE) {
                const int row = i / k_blocks_per_row;
                const int blk = i % k_blocks_per_row;
                const int k_row_idx = k_start + row;
                if (k_row_idx < n_kv) {
                    const ulong k_row_off = batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_row_idx * k_nb1;
                    const global char * blk_ptr = k_base + k_row_off + blk * Q8_0_BLOCK_SIZE;
                    const float df = (float) vload_half(0, (const global half *) blk_ptr);
                    const global char * qs = blk_ptr + 2;
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const float4 v = df * (float4)((float) qs[j*4 + 0],
                                                       (float) qs[j*4 + 1],
                                                       (float) qs[j*4 + 2],
                                                       (float) qs[j*4 + 3]);
                        l_k[row][blk * 8 + j] = (half4)((half) v.s0, (half) v.s1, (half) v.s2, (half) v.s3);
                    }
                } else {
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) l_k[row][blk * 8 + j] = (half4)(0.0h);
                }
            }
#endif
        }
        // V tile load ? strategy-dependent.
#if FA_V_STRATEGY == 2
        {
            // Int8 packed V in local memory + per-block scale. Accumulate
            // step unpacks inline.
            const int v_blocks_per_row = DV_Q8_BLOCKS_PREFILL;
            const int n_blocks_total = BLOCK_N * v_blocks_per_row;
            for (int i = tid; i < n_blocks_total; i += WG_SIZE) {
                const int row = i / v_blocks_per_row;
                const int blk = i % v_blocks_per_row;
                const int v_row_idx = k_start + row;
                if (v_row_idx < n_kv) {
                    const ulong v_row_off = batch_idx * v_nb3 + head_kv_idx * v_nb2 + v_row_idx * v_nb1;
                    const global char * blk_ptr = v_base + v_row_off + blk * Q8_0_BLOCK_SIZE;
                    const float df = (float) vload_half(0, (const global half *) blk_ptr);
                    const global uchar * qs = (const global uchar *)(blk_ptr + 2);
                    l_v_scale[row][blk] = df;
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        uint v_packed =
                              (uint) qs[j*4 + 0]        |
                             ((uint) qs[j*4 + 1]) <<  8 |
                             ((uint) qs[j*4 + 2]) << 16 |
                             ((uint) qs[j*4 + 3]) << 24;
                        l_v_packed[row][blk * 8 + j] = v_packed;
                    }
                } else {
                    l_v_scale[row][blk] = 0.0f;
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) l_v_packed[row][blk * 8 + j] = 0u;
                }
            }
        }
#else
        {
            // Default: dequant q4_0 V -> half in local memory. l_v[row][blk*8 + m]
            // holds block elements 4m..4m+3, so m 0..3 are the low nibbles of
            // qs[0..15] (elements 0..15) and m 4..7 the high nibbles of the same
            // bytes (elements 16..31) -- the same slot meaning as the q8_0 path,
            // which is what lets the accumulate below stay untouched.
            const int v_blocks_per_row = DV / QK4_0;
            const int n_blocks_total = BLOCK_N * v_blocks_per_row;
            for (int i = tid; i < n_blocks_total; i += WG_SIZE) {
                const int row = i / v_blocks_per_row;
                const int blk = i % v_blocks_per_row;
                const int v_row_idx = k_start + row;
                if (v_row_idx < n_kv) {
                    const ulong v_row_off = batch_idx * v_nb3 + head_kv_idx * v_nb2 + v_row_idx * v_nb1;
                    const global char * blk_ptr = v_base + v_row_off + blk * Q4_0_BLOCK_SIZE;
                    const float df = (float) vload_half(0, (const global half *) blk_ptr);
                    const global uchar * qs = (const global uchar *)(blk_ptr + 2);
                    #pragma unroll
                    for (int g = 0; g < 4; ++g) {
                        const float4 lo = df * (float4)((float)(int)(qs[g*4 + 0] & 0x0F) - 8.0f,
                                                        (float)(int)(qs[g*4 + 1] & 0x0F) - 8.0f,
                                                        (float)(int)(qs[g*4 + 2] & 0x0F) - 8.0f,
                                                        (float)(int)(qs[g*4 + 3] & 0x0F) - 8.0f);
                        l_v[row][blk * 8 + g] = (half4)((half) lo.s0, (half) lo.s1, (half) lo.s2, (half) lo.s3);

                        const float4 hi = df * (float4)((float)(int)(qs[g*4 + 0] >> 4) - 8.0f,
                                                        (float)(int)(qs[g*4 + 1] >> 4) - 8.0f,
                                                        (float)(int)(qs[g*4 + 2] >> 4) - 8.0f,
                                                        (float)(int)(qs[g*4 + 3] >> 4) - 8.0f);
                        l_v[row][blk * 8 + 4 + g] = (half4)((half) hi.s0, (half) hi.s1, (half) hi.s2, (half) hi.s3);
                    }
                } else {
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) l_v[row][blk * 8 + j] = (half4)(0.0h);
                }
            }
        }
#endif
        barrier(CLK_LOCAL_MEM_FENCE);

        // QK dot + online softmax. N_SPLIT>1 reduces per-thread partials via shuffle_xor.
#if N_SPLIT > 1
        {
#else
        if (query_valid) {
#endif
            const int k_blk_base = split_idx * SPLIT_DK_Q8_BLOCKS;
            for (int j = 0; j < BLOCK_N; j += 4) {
                const int k_row0 = k_start + j;
                const int k_row1 = k_start + j + 1;
                const int k_row2 = k_start + j + 2;
                const int k_row3 = k_start + j + 3;

                ACC_TYPE s0, s1, s2, s3;
#ifdef FA_HAVE_INT_DOT
                // dp4a-accelerated QK dot over owned blocks.
                s0 = 0.0f; s1 = 0.0f; s2 = 0.0f; s3 = 0.0f;
                #pragma unroll
                for (int b_local = 0; b_local < SPLIT_DK_Q8_BLOCKS; ++b_local) {
                    const int b = k_blk_base + b_local;
                    int sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
#if defined(FA_PROBE_NO_QK)
                    // WRONG MATH, diagnostic. Delete the QK dot entirely (no LDS reads, no
                    // dp4a) to size what share of this kernel the QK loop even is -- the
                    // kernel also does the online softmax and the PV accumulate.
                    sum0 = (int) l_k_packed[j][b * 8];
                    sum1 = sum0; sum2 = sum0; sum3 = sum0;
#elif defined(FA_PROBE_NO_LDS)
                    // WRONG MATH, diagnostic. The 4-row unroll amortizes the PRIVATE Q read,
                    // but each dp4a still takes its K operand from a distinct __local address,
                    // so the K side is 1 LDS read per dp4a. Hoist one K value into a register
                    // and reuse it: same dp4a count, 32x fewer LDS reads. If that is much
                    // faster the QK loop is LDS-issue-bound.
                    const uint kprobe = l_k_packed[j][b * 8];
                    #pragma unroll
                    for (int g = 0; g < 8; ++g) {
                        const uint qp = q_packed_pf[b_local * 8 + g];
                        sum0 = dot_acc_sat_4x8packed_ss_int(qp, kprobe, sum0);
                        sum1 = dot_acc_sat_4x8packed_ss_int(qp, kprobe, sum1);
                        sum2 = dot_acc_sat_4x8packed_ss_int(qp, kprobe, sum2);
                        sum3 = dot_acc_sat_4x8packed_ss_int(qp, kprobe, sum3);
                    }
#elif defined(FA_K_LDS_T)
                    // The 4 KV rows are adjacent in the transposed tile, so each (b, g)
                    // step is ONE 128-bit local read instead of four 32-bit ones.
                    #pragma unroll
                    for (int g = 0; g < 8; ++g) {
                        const uint qp = q_packed_pf[b_local * 8 + g];
                        const uint4 kq4 = vload4(0, &l_k_packed[b * 8 + g][j]);
                        sum0 = dot_acc_sat_4x8packed_ss_int(qp, kq4.s0, sum0);
                        sum1 = dot_acc_sat_4x8packed_ss_int(qp, kq4.s1, sum1);
                        sum2 = dot_acc_sat_4x8packed_ss_int(qp, kq4.s2, sum2);
                        sum3 = dot_acc_sat_4x8packed_ss_int(qp, kq4.s3, sum3);
                    }
#else
                    #pragma unroll
                    for (int g = 0; g < 8; ++g) {
                        const uint qp = q_packed_pf[b_local * 8 + g];
                        sum0 = dot_acc_sat_4x8packed_ss_int(qp, l_k_packed[j  ][b * 8 + g], sum0);
                        sum1 = dot_acc_sat_4x8packed_ss_int(qp, l_k_packed[j+1][b * 8 + g], sum1);
                        sum2 = dot_acc_sat_4x8packed_ss_int(qp, l_k_packed[j+2][b * 8 + g], sum2);
                        sum3 = dot_acc_sat_4x8packed_ss_int(qp, l_k_packed[j+3][b * 8 + g], sum3);
                    }
#endif
                    const float qd = q_d_pf[b_local];
#ifdef FA_K_LDS_T
                    const float4 ks4 = vload4(0, &l_k_scale[b][j]);
                    s0 += (float)sum0 * qd * ks4.s0;
                    s1 += (float)sum1 * qd * ks4.s1;
                    s2 += (float)sum2 * qd * ks4.s2;
                    s3 += (float)sum3 * qd * ks4.s3;
#else
                    s0 += (float)sum0 * qd * l_k_scale[j  ][b];
                    s1 += (float)sum1 * qd * l_k_scale[j+1][b];
                    s2 += (float)sum2 * qd * l_k_scale[j+2][b];
                    s3 += (float)sum3 * qd * l_k_scale[j+3][b];
#endif
                }
#else
                ACC_TYPE4 dot_acc0 = (ACC_TYPE4)(0.0f);
                ACC_TYPE4 dot_acc1 = (ACC_TYPE4)(0.0f);
                ACC_TYPE4 dot_acc2 = (ACC_TYPE4)(0.0f);
                ACC_TYPE4 dot_acc3 = (ACC_TYPE4)(0.0f);
                #pragma unroll
                for (int k = 0; k < SPLIT_DK_VEC; ++k) {
                    const ACC_TYPE4 qk = q_priv[k];
                    const int k_abs = dk_off_vec + k;
                    dot_acc0 = mad(qk, CONVERT_KV_ACC4(l_k[j  ][k_abs]), dot_acc0);
                    dot_acc1 = mad(qk, CONVERT_KV_ACC4(l_k[j+1][k_abs]), dot_acc1);
                    dot_acc2 = mad(qk, CONVERT_KV_ACC4(l_k[j+2][k_abs]), dot_acc2);
                    dot_acc3 = mad(qk, CONVERT_KV_ACC4(l_k[j+3][k_abs]), dot_acc3);
                }
                s0 = dot_acc0.s0 + dot_acc0.s1 + dot_acc0.s2 + dot_acc0.s3;
                s1 = dot_acc1.s0 + dot_acc1.s1 + dot_acc1.s2 + dot_acc1.s3;
                s2 = dot_acc2.s0 + dot_acc2.s1 + dot_acc2.s2 + dot_acc2.s3;
                s3 = dot_acc3.s0 + dot_acc3.s1 + dot_acc3.s2 + dot_acc3.s3;
#endif

#if N_SPLIT > 1
                // Power-of-2 N_SPLIT: shuffle_xor butterfly. N_SPLIT=3 (DK=96): 3-way shuffle.
                #if (N_SPLIT & (N_SPLIT - 1)) == 0
                    #pragma unroll
                    for (int step = 1; step < N_SPLIT; step <<= 1) {
                        s0 += sub_group_shuffle_xor(s0, step);
                        s1 += sub_group_shuffle_xor(s1, step);
                        s2 += sub_group_shuffle_xor(s2, step);
                        s3 += sub_group_shuffle_xor(s3, step);
                    }
                #else
                    const uint tri_base = (get_sub_group_local_id() / N_SPLIT) * N_SPLIT;
                    s0 = sub_group_shuffle(s0, tri_base + 0) + sub_group_shuffle(s0, tri_base + 1) + sub_group_shuffle(s0, tri_base + 2);
                    s1 = sub_group_shuffle(s1, tri_base + 0) + sub_group_shuffle(s1, tri_base + 1) + sub_group_shuffle(s1, tri_base + 2);
                    s2 = sub_group_shuffle(s2, tri_base + 0) + sub_group_shuffle(s2, tri_base + 1) + sub_group_shuffle(s2, tri_base + 2);
                    s3 = sub_group_shuffle(s3, tri_base + 0) + sub_group_shuffle(s3, tri_base + 1) + sub_group_shuffle(s3, tri_base + 2);
                #endif
                if (!query_valid) { s0 = FA_M_INIT; s1 = FA_M_INIT; s2 = FA_M_INIT; s3 = FA_M_INIT; }
#endif
                s0 *= scale; s1 *= scale; s2 *= scale; s3 *= scale;

                if (is_causal) {
                    const int causal_limit = n_kv - n_q + my_query_row;
                    if (k_row0 > causal_limit) s0 = FA_M_INIT;
                    if (k_row1 > causal_limit) s1 = FA_M_INIT;
                    if (k_row2 > causal_limit) s2 = FA_M_INIT;
                    if (k_row3 > causal_limit) s3 = FA_M_INIT;
                }
                if (k_row0 >= n_kv) s0 = FA_M_INIT;
                if (k_row1 >= n_kv) s1 = FA_M_INIT;
                if (k_row2 >= n_kv) s2 = FA_M_INIT;
                if (k_row3 >= n_kv) s3 = FA_M_INIT;

                if (query_valid && mask_base != NULL && blk_cur != 2) {
                    const global MASK_DATA_TYPE * mask_ptr =
                        (const global MASK_DATA_TYPE *) (mask_base + my_query_row * mask_nb1);
                    if (k_row0 < n_kv) s0 += slope * (ACC_TYPE) mask_ptr[k_row0];
                    if (k_row1 < n_kv) s1 += slope * (ACC_TYPE) mask_ptr[k_row1];
                    if (k_row2 < n_kv) s2 += slope * (ACC_TYPE) mask_ptr[k_row2];
                    if (k_row3 < n_kv) s3 += slope * (ACC_TYPE) mask_ptr[k_row3];
                }
                if (logit_softcap > 0.0f) {
                    s0 = logit_softcap * tanh(s0 / logit_softcap);
                    s1 = logit_softcap * tanh(s1 / logit_softcap);
                    s2 = logit_softcap * tanh(s2 / logit_softcap);
                    s3 = logit_softcap * tanh(s3 / logit_softcap);
                }

                const ACC_TYPE m_new      = max(m_i, max(max(s0, s1), max(s2, s3)));
                // Whole tile masked (m_new == FA_M_INIT): force the exp() args
                // far negative so the tile contributes 0, not exp(0)=1.
                const ACC_TYPE m_exp      = (m_new == FA_M_INIT) ? 0.0f : m_new;
                const ACC_TYPE scale_prev = native_exp(m_i - m_exp);
                const ACC_TYPE p0         = native_exp(s0 - m_exp);
                const ACC_TYPE p1         = native_exp(s1 - m_exp);
                const ACC_TYPE p2         = native_exp(s2 - m_exp);
                const ACC_TYPE p3         = native_exp(s3 - m_exp);

#if FA_V_STRATEGY == 2
                #pragma unroll
                for (int b_local = 0; b_local < DV_Q8_BLOCKS_PREFILL / N_SPLIT; ++b_local) {
                    const int b_abs = split_idx * (DV_Q8_BLOCKS_PREFILL / N_SPLIT) + b_local;
                    const float d0 = l_v_scale[j  ][b_abs];
                    const float d1 = l_v_scale[j+1][b_abs];
                    const float d2 = l_v_scale[j+2][b_abs];
                    const float d3 = l_v_scale[j+3][b_abs];
                    #pragma unroll
                    for (int g = 0; g < 8; ++g) {
                        const int lane_abs   = b_abs   * 8 + g;
                        const int lane_local = b_local * 8 + g;
                        uint pk0 = l_v_packed[j  ][lane_abs];
                        uint pk1 = l_v_packed[j+1][lane_abs];
                        uint pk2 = l_v_packed[j+2][lane_abs];
                        uint pk3 = l_v_packed[j+3][lane_abs];
                        float4 v0 = d0 * (float4)((float)(char)(pk0 & 0xff), (float)(char)((pk0>>8)&0xff), (float)(char)((pk0>>16)&0xff), (float)(char)((pk0>>24)&0xff));
                        float4 v1 = d1 * (float4)((float)(char)(pk1 & 0xff), (float)(char)((pk1>>8)&0xff), (float)(char)((pk1>>16)&0xff), (float)(char)((pk1>>24)&0xff));
                        float4 v2 = d2 * (float4)((float)(char)(pk2 & 0xff), (float)(char)((pk2>>8)&0xff), (float)(char)((pk2>>16)&0xff), (float)(char)((pk2>>24)&0xff));
                        float4 v3 = d3 * (float4)((float)(char)(pk3 & 0xff), (float)(char)((pk3>>8)&0xff), (float)(char)((pk3>>16)&0xff), (float)(char)((pk3>>24)&0xff));
                        o_acc[lane_local] = mad(p3, v3,
                                           mad(p2, v2,
                                           mad(p1, v1,
                                           mad(p0, v0,
                                           o_acc[lane_local] * scale_prev))));
                    }
                }
#else  // FA_V_STRATEGY == 0
                #pragma unroll
                for (int i = 0; i < SPLIT_DV_VEC; ++i) {
                    const int i_abs = dv_off_vec + i;
                    o_acc[i] = mad(p3, CONVERT_KV_ACC4(l_v[j+3][i_abs]),
                               mad(p2, CONVERT_KV_ACC4(l_v[j+2][i_abs]),
                               mad(p1, CONVERT_KV_ACC4(l_v[j+1][i_abs]),
                               mad(p0, CONVERT_KV_ACC4(l_v[j  ][i_abs]),
                               o_acc[i] * scale_prev))));
                }
#endif
                l_i = l_i * scale_prev + p0 + p1 + p2 + p3;
                m_i = m_new;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write output. With N_SPLIT>1 each thread writes its SPLIT_DV_VEC slice.
    if (query_valid) {
        if (sinks_void != NULL) {
            const global ACC_TYPE * sinks_ptr =
                (const global ACC_TYPE *) ((const global char *) sinks_void + sinks_offset);
            const ACC_TYPE m_sink  = sinks_ptr[head_idx];
            const ACC_TYPE m_final = max(m_i, m_sink);
            const ACC_TYPE scale_o = exp(m_i - m_final);
            #pragma unroll
            for (int i = 0; i < SPLIT_DV_VEC; ++i) o_acc[i] *= scale_o;
            l_i = l_i * scale_o + exp(m_sink - m_final);
            m_i = m_final;
        }
        const ACC_TYPE l_inv = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
        const ulong o_row_offset = batch_idx * o_nb3 + my_query_row * o_nb2 + head_idx * o_nb1;
        global float4 * o_row = (global float4 *) (o_base + o_row_offset);
        if (l_inv > 0.0f) {
            #pragma unroll
            for (int i = 0; i < SPLIT_DV_VEC; ++i) o_row[dv_off_vec + i] = o_acc[i] * l_inv;
        } else {
            #pragma unroll
            for (int i = 0; i < SPLIT_DV_VEC; ++i) o_row[dv_off_vec + i] = (float4)(0.0f);
        }
    }
}

// FD Pass 2: merge split partials. Identical across q4_0/q8_0/f16; each FA
// source owns a copy since kernels compile per-source-program.
__kernel void flash_attn_f32_merge(
    const global float * partial_void,
    global void * o_void,
    const ulong o_offset,
    const int n_head,
    const int n_splits,
    const ulong o_nb1, const ulong o_nb2, const ulong o_nb3,
    const global void * sinks_void,
    const ulong sinks_offset,
    const int n_q
) {
    const int lane           = get_local_id(0);
    const int head_batch_idx = get_global_id(1);
    const int q_idx          = get_global_id(2);
    const int batch_idx      = head_batch_idx / n_head;
    const int head_idx       = head_batch_idx % n_head;

    const ulong record_stride = (ulong) FA_PARTIAL_FLOATS;
    const ulong record_idx_0  = (((ulong) batch_idx * n_head + head_idx) * n_q + q_idx) * n_splits;
    const global float * rec0 = partial_void + record_idx_0 * record_stride;

    __local ACC_TYPE m_final_shared;
    __local ACC_TYPE l_final_shared;
    if (lane == 0) {
        ACC_TYPE m = FA_M_INIT;
        for (int c = 0; c < n_splits; ++c) {
            const ACC_TYPE m_c = rec0[c * record_stride + 0];
            m = max(m, m_c);
        }
        ACC_TYPE m_sink = 0.0f;
        bool has_sink = false;
        if (sinks_void != NULL) {
            const global ACC_TYPE * sinks_ptr =
                (const global ACC_TYPE *) ((const global char *) sinks_void + sinks_offset);
            m_sink = sinks_ptr[head_idx];
            has_sink = true;
            m = max(m, m_sink);
        }
        ACC_TYPE l = 0.0f;
        for (int c = 0; c < n_splits; ++c) {
            const ACC_TYPE m_c = rec0[c * record_stride + 0];
            const ACC_TYPE l_c = rec0[c * record_stride + 1];
            if (m_c > FA_M_INIT) {
                l += l_c * exp(m_c - m);
            }
        }
        if (has_sink) {
            l += exp(m_sink - m);
        }
        m_final_shared = m;
        l_final_shared = l;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const ACC_TYPE m_final = m_final_shared;
    const ACC_TYPE l_final = l_final_shared;
    const ACC_TYPE l_inv   = (l_final > 0.0f) ? (1.0f / l_final) : 0.0f;

    ACC_TYPE4 o = (ACC_TYPE4)(0.0f);
    for (int c = 0; c < n_splits; ++c) {
        const global float * rec_c   = rec0 + c * record_stride;
        const ACC_TYPE       m_c     = rec_c[0];
        if (m_c <= FA_M_INIT) continue;
        const global float4 * rec_oc = (const global float4 *) (rec_c + 2);
        const ACC_TYPE scale_c = exp(m_c - m_final);
        o = mad((ACC_TYPE4)(scale_c), rec_oc[lane], o);
    }
    o = o * l_inv;

    const ulong o_row_offset = (ulong) batch_idx * o_nb3 + (ulong) q_idx * o_nb2 + (ulong) head_idx * o_nb1;
    global O_DATA_TYPE4 * o_row = (global O_DATA_TYPE4 *) ((global char *) o_void + o_offset + o_row_offset);
    o_row[lane] = CONVERT_O_DATA4(o);
}

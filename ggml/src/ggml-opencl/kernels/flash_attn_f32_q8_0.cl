// Enable extensions at the very top of the file. OpenCL pragmas must appear
// before any non-directive tokens; some drivers don't register overloads if
// the enable pragma is interleaved with other directives or comments.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#ifdef cl_khr_integer_dot_product
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#define FA_HAVE_INT_DOT 1
#endif

// sub_group_shuffle_xor: needed by the N_SPLIT>1 path to reduce per-thread
// QK partial dots across the N_SPLIT threads that share a query row.
#ifdef cl_khr_subgroup_shuffle
#pragma OPENCL EXTENSION cl_khr_subgroup_shuffle : enable
#define HAS_SUBGROUP_SHUFFLE 1
#elif defined(cl_qcom_subgroup_shuffle)
#pragma OPENCL EXTENSION cl_qcom_subgroup_shuffle : enable
#define HAS_SUBGROUP_SHUFFLE 1
#endif

// 4-way 8-bit integer dot product (CUDA dp4a analogue). On Adreno X1-85 the
// khr_integer_dot_product extension maps to the hardware dp4a unit: microbench
// confirms ~2× speedup vs float-dot (the CL_DEVICE_INTEGER_DOT_PRODUCT_
// ACCELERATION_PROPERTIES_*_KHR fields return 0 but that's an advisory flag,
// not a reflection of the real backend — the HW unit is used).
// The packed uint form (`dot_acc_sat_4x8packed_ss_int`) is preferred over
// char4 for compatibility: some drivers register one overload but not the other.

// Flash attention kernel for Q=f32, K=q8_0, V=q8_0.
// q8_0 block: half d (scale) + char qs[32] (signed 8-bit quants).
// Dequantize: val[i] = d * qs[i]
//
// This kernel reads KV in q8_0 format directly, avoiding the f16 dequant
// path. For decode (n_q=1), this reduces KV bandwidth by ~47% vs f16.
//
// When cl_khr_integer_dot_product is available we additionally quantize the
// private Q row to int8 per 32-element block and replace the per-block QK
// dot with 8 packed 4-way int8 dp4a ops — each collapses 4 int8 multiplies
// and adds into a single instruction on Adreno. The output scales up by
// Qd * Kd after summation.

#define ACC_TYPE float
#define ACC_TYPE4 float4
#define Q_DATA_TYPE4 float4
#define O_DATA_TYPE4 float4
#define MASK_DATA_TYPE half
#define CONVERT_Q_ACC4(x) (x)
#define CONVERT_O_DATA4(x) (x)

#define DK_VEC (DK/4)
#define DV_VEC (DV/4)
#define Q1_WG_SIZE 64

// q8_0 block layout: 2 bytes scale (half) + 32 bytes quants = 34 bytes per block
#define QK8_0 32
#define Q8_0_BLOCK_SIZE 34

// Number of q8_0 blocks per row of K or V
#define DK_Q8_BLOCKS (DK / QK8_0)
#define DV_Q8_BLOCKS (DV / QK8_0)

// Inline dequantize: load a q8_0 block and compute dot product with a slice of q_priv.
// q_slice points to 8 float4 elements (32 floats) from the query.
// Returns the dot product of the dequantized block with the query slice.
inline float dot_q8_0_f32(const global char * block_ptr, ACC_TYPE4 * q_slice) {
    // Read scale
    float d = vload_half(0, (const global half *)block_ptr);
    // Read 32 int8 quants starting at offset 2
    const global char * qs = block_ptr + 2;

    float sum = 0.0f;
    // Process 32 elements as 8 groups of 4
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        float4 qv = (float4)((float)qs[i*4], (float)qs[i*4+1], (float)qs[i*4+2], (float)qs[i*4+3]);
        sum += dot(q_slice[i], qv);
    }
    return sum * d;
}

#ifdef FA_HAVE_INT_DOT
// Pack four signed 8-bit chars into a uint32 (little-endian byte order —
// matches the layout `dot_4x8packed_ss_int` expects).
inline uint pack_i8x4(char a, char b, char c, char d) {
    return ((uint)(uchar)a)       |
           ((uint)(uchar)b) <<  8  |
           ((uint)(uchar)c) << 16  |
           ((uint)(uchar)d) << 24;
}

// Quantize a single q8_0-sized block (32 floats, 8 float4s) of Q to int8 and
// store it as 8 packed uints. Returns the per-block quant scale Qd.
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

// QK dot product using the packed 4x8-bit signed×signed integer dot product.
// Each call computes dot(char4, char4) via `dot_4x8packed_ss_int` on a
// uint-packed representation; the OpenCL compiler lowers this to IMADs on
// Adreno X1-85 (accelerated=0) but still avoids the int->float conversion
// that the scalar float path has to do per element.
inline float dot_q8_0_int(const global char * k_block_ptr,
                          const uint *        q_packed,
                          float               q_d) {
    float kd = vload_half(0, (const global half *)k_block_ptr);
    const global uchar * k_qs = (const global uchar *)(k_block_ptr + 2);

    // Can't cast k_qs to uint* directly: it sits at (block+2), only 2-byte
    // aligned. Pack four chars → uint per call instead.
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

// Dequantize a q8_0 block into 8 float4 values (32 floats).
inline void dequant_q8_0_f32(const global char * block_ptr, ACC_TYPE4 * out) {
    float d = vload_half(0, (const global half *)block_ptr);
    const global char * qs = block_ptr + 2;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        out[i] = d * (float4)((float)qs[i*4], (float)qs[i*4+1], (float)qs[i*4+2], (float)qs[i*4+3]);
    }
}

// ALiBi slope computation. When max_bias <= 0 (no ALiBi) this returns 1.0f so
// the mask term is applied directly (score += 1.0 * mask[k_idx]); the baseline
// f16 FA kernel does the same. Returning 0 here would silently drop the mask.
inline float get_alibi_slope(float max_bias, int head_idx, int n_head_log2, float m0, float m1) {
    if (max_bias <= 0.0f) return 1.0f;
    float base = (head_idx < n_head_log2) ? m0 : m1;
    int   exph = (head_idx < n_head_log2) ? (head_idx + 1) : (2*(head_idx - n_head_log2) + 1);
    return pow(base, (float)exph);
}

// ============================================================================
// q1 decode kernel: Q=f32, K=q8_0, V=q8_0
// One query row, each thread processes a different KV position.
// ============================================================================
__kernel void flash_attn_f32_q8_0_q1(
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

    // Load query row into private registers (f32)
    ACC_TYPE4 q_priv[DK_VEC];
    const ulong q_row_offset = batch_idx * q_nb3 + head_idx * q_nb2;
    const global Q_DATA_TYPE4* q_ptr = (const global Q_DATA_TYPE4*)(q_base + q_row_offset);
    #pragma unroll
    for (int i = 0; i < DK_VEC; ++i) {
        q_priv[i] = CONVERT_Q_ACC4(q_ptr[i]);
    }

#ifdef FA_HAVE_INT_DOT
    // Quantize Q once to int8 per q8_0-sized block, packed as 8 uints per
    // block for dot_4x8packed_ss_int. Each thread owns its own copy since
    // every thread walks a different slice of K rows. Q_priv stays intact
    // for the V-accumulation (V path multiplies by fp32 attention weights p).
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

    // === Pass 1: find max score ===
    ACC_TYPE m_i = (sinks_ptr != NULL) ? sinks_ptr[head_idx] : -INFINITY;
    for (int k_idx = tid; k_idx < n_kv; k_idx += Q1_WG_SIZE) {
        const global char* k_row = k_base + batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_idx * k_nb1;

        // Compute QK dot product over DK_Q8_BLOCKS q8_0 blocks
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
        m_i = max(m_i, score);
    }

    // Reduce max across workgroup
    __local ACC_TYPE local_m[Q1_WG_SIZE];
    local_m[tid] = m_i;
    barrier(CLK_LOCAL_MEM_FENCE);
    #pragma unroll
    for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) local_m[tid] = max(local_m[tid], local_m[tid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const ACC_TYPE m_final = local_m[0];

    // === Pass 2: compute softmax-weighted V accumulation ===
    ACC_TYPE4 o_acc[DV_VEC];
    #pragma unroll
    for (int i = 0; i < DV_VEC; ++i) o_acc[i] = (ACC_TYPE4)(0.0f);
    ACC_TYPE l_i = 0.0f;

    for (int k_idx = tid; k_idx < n_kv; k_idx += Q1_WG_SIZE) {
        const global char* k_row = k_base + batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_idx * k_nb1;
        const global char* v_row = v_base + batch_idx * v_nb3 + head_kv_idx * v_nb2 + k_idx * v_nb1;

        // Recompute QK score
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

        const ACC_TYPE p = exp(score - m_final);
        l_i += p;

        // Accumulate p * V (dequantize V inline)
        #pragma unroll
        for (int b = 0; b < DV_Q8_BLOCKS; b++) {
            ACC_TYPE4 v_dequant[8];
            dequant_q8_0_f32(v_row + b * Q8_0_BLOCK_SIZE, v_dequant);
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                o_acc[b * 8 + i] = mad(p, v_dequant[i], o_acc[b * 8 + i]);
            }
        }
    }

    // === Reduce and write output ===
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

// ============================================================================
// Flash-Decoding (K-split) — Pass 1 for q8_0 KV.
// Mirrors flash_attn_f32_f16_q1_split from flash_attn_f32_f16.cl but reads
// q8_0-packed K/V and optionally uses dp4a for the QK dot.
// Partial layout matches the f16 version: [batch][head][query][split][m, l, O[DV]]
// floats, stride (2 + DV) per split. gid(2) = q_idx*n_splits + split_idx so
// the same kernel serves n_q=1 (single-token FD) and n_q>1 (speculative /
// parallel decode). The merge kernel (defined in flash_attn_f32_f16.cl) is
// type-agnostic and reused as-is.
// No sinks / final normalisation here — merge handles those.
// ============================================================================
#define FA_PARTIAL_FLOATS (2 + DV)

__kernel void flash_attn_f32_q8_0_q1_split(
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

    // Empty trailing split → marker.
    if (kv_start >= kv_end) {
        if (tid == 0) {
            rec[0] = -INFINITY;
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

    // Load query row.
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

    // Pass 1a — split-local max (no sink here).
    ACC_TYPE m_i = -INFINITY;
    for (int k_idx = kv_start + tid; k_idx < kv_end; k_idx += Q1_WG_SIZE) {
        const global char * k_row = k_base + batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_idx * k_nb1;
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
        m_i = max(m_i, score);
    }

    __local ACC_TYPE local_m[Q1_WG_SIZE];
    local_m[tid] = m_i;
    barrier(CLK_LOCAL_MEM_FENCE);
    #pragma unroll
    for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) local_m[tid] = max(local_m[tid], local_m[tid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const ACC_TYPE m_c = local_m[0];

    // Pass 1b — softmax-weighted V accumulate (dequant V inline).
    ACC_TYPE4 o_acc[DV_VEC];
    #pragma unroll
    for (int i = 0; i < DV_VEC; ++i) o_acc[i] = (ACC_TYPE4)(0.0f);
    ACC_TYPE l_i = 0.0f;

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
        const ACC_TYPE p = exp(score - m_c);
        l_i += p;
        #pragma unroll
        for (int b = 0; b < DV_Q8_BLOCKS; ++b) {
            ACC_TYPE4 v_dequant[8];
            dequant_q8_0_f32(v_row + b * Q8_0_BLOCK_SIZE, v_dequant);
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                o_acc[b * 8 + i] = mad(p, v_dequant[i], o_acc[b * 8 + i]);
            }
        }
    }

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

// ============================================================================
// Prefill kernel: Q=f32, K=q8_0, V=q8_0, n_q > 1.
// BLOCK_M × BLOCK_N tiling.
//
// K path: stored in local memory as packed int8 (8 uints per QK8_0-sized block,
// 1 float scale per block). QK dot uses `dot_acc_sat_4x8packed_ss_int` (the
// KHR cl_khr_integer_dot_product builtin that maps to the Adreno hardware
// dp4a unit — confirmed 2× faster than the float-dot path in microbench).
// Q is quantised once per block into int8+uint-packed form before the KV
// iteration loop.
//
// V path: dequantised to half in local memory since p×V needs fp weights.
//
// Assumes DK % QK8_0 == 0 and DV % QK8_0 == 0 (supports_op already gates on
// this). No N_SPLIT, no kv_pad/mask_pad/blk prepass — boundary handling is
// done inline so the kernel is self-contained.
// ============================================================================
#define KV_DATA_TYPE4 half4
#define CONVERT_KV_ACC4(x) ((float4)((float)(x).s0, (float)(x).s1, (float)(x).s2, (float)(x).s3))

#define DK_Q8_BLOCKS_PREFILL (DK / QK8_0)
#define DV_Q8_BLOCKS_PREFILL (DV / QK8_0)

// N_SPLIT: number of threads that collaborate on each query row's QK dot.
// Mirrors the f32_f16 kernel's N_SPLIT mechanism — when set >1, each thread
// owns 1/N_SPLIT of the DK and DV dimensions. This reduces register pressure
// for large DK and usually increases WG occupancy on Adreno (WG_SIZE grows
// from BLOCK_M to BLOCK_M*N_SPLIT, filling the wavefront more fully).
// N_SPLIT>1 requires sub_group_shuffle_xor (Adreno has it).
// Requires DK_Q8_BLOCKS_PREFILL % N_SPLIT == 0 (each thread gets whole blocks).
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

// V-path strategy.
//   FA_V_STRATEGY==0 (default): dequant V → half into local memory up-front,
//                     then p*V from local. Best on Adreno X1-85: BLOCK_M
//                     threads share a single local V tile, and half4 → float4
//                     convert is a fast hardware path.
//   FA_V_STRATEGY==2 (opt-in): V stays as packed int8 + scale in local
//                     memory; unpack and dequant inside the accumulate loop.
//                     Halves local V footprint (34 vs 64 bytes/row). Measured
//                     ~3% slower on X1-85 because the shift+mask+cvt cost
//                     exceeds the local-memory savings. Keep as a knob in case
//                     local-memory pressure matters for larger BLOCK_N.
//
// FA_V_STRATEGY==1 (on-the-fly from global) was tested and removed: each of
// BLOCK_M threads re-reads the same V rows from global memory, which defeats
// local-memory reuse and produced -36% at pp2048 on X1-85.
#ifndef FA_V_STRATEGY
#define FA_V_STRATEGY 0
#endif

__kernel void flash_attn_f32_q8_0(
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
    // blk (optional): per-(qblock, kvblock) classification from flash_attn_blk_f16.
    //   0 = fully masked → skip tile, 1 = mixed → apply per-row mask,
    //   2 = fully unmasked → skip mask application.
    // Pass NULL to disable the prepass optimisation.
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

    // blk_base: pointer into the blk classification buffer for this
    // (batch, head, q_block) slice. n_kv_blocks entries per slice.
    // BLK_PREPASS_BM is the BLOCK_M the prepass kernel was compiled with; it
    // may differ from this kernel's BLOCK_M (e.g., DK=96 quant uses BM=32 here
    // but the prepass used f16's BM=64). Derive our prepass q-block index from
    // my_query_row / BLK_PREPASS_BM so multiple smaller quant q-blocks map into
    // the same prepass entry when needed.
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

    // --- Load Q row slice into private registers -----------------------------
    // Each thread owns SPLIT_DK_VEC float4 lanes of the query row (only this
    // thread's slice — dk_off offsets which slice).
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
    // Quantise the owned Q slice into packed int8 (8 uints per QK8_0-sized
    // block) + per-block scale qd. Each thread only processes SPLIT_DK_Q8_BLOCKS
    // blocks (its share of the DK dimension).
    uint  q_packed_pf[SPLIT_DK_Q8_BLOCKS * 8];
    float q_d_pf[SPLIT_DK_Q8_BLOCKS];
    #pragma unroll
    for (int b = 0; b < SPLIT_DK_Q8_BLOCKS; ++b) {
        q_d_pf[b] = quant_q_block_int8_packed(&q_priv[b * 8], &q_packed_pf[b * 8]);
    }
#endif

    // --- Output accumulator, softmax state -----------------------------------
    // Each thread owns SPLIT_DV_VEC float4 lanes of o_acc (for N_SPLIT==1,
    // that's the full output row; for N_SPLIT>1, only this thread's slice).
    const int dv_off_vec = split_idx * SPLIT_DV_VEC;
    ACC_TYPE4 o_acc[SPLIT_DV_VEC];
    #pragma unroll
    for (int i = 0; i < SPLIT_DV_VEC; ++i) o_acc[i] = (ACC_TYPE4)(0.0f);

    ACC_TYPE m_i = -INFINITY;
    ACC_TYPE l_i = 0.0f;

    float slope = get_alibi_slope(max_bias, head_idx, n_head_log2, m0, m1);

#ifdef FA_HAVE_INT_DOT
    // K tile: packed int8 (8 uints per block) + per-block scale.
    __local uint  l_k_packed[BLOCK_N][DK_Q8_BLOCKS_PREFILL * 8];
    __local float l_k_scale [BLOCK_N][DK_Q8_BLOCKS_PREFILL];
#else
    __local half4 l_k[BLOCK_N][DK_VEC];
#endif

#if FA_V_STRATEGY == 2
    // Packed int8 V: 8 uints per QK8_0-sized block + 1 fp32 scale per block.
    __local uint  l_v_packed[BLOCK_N][DV_Q8_BLOCKS_PREFILL * 8];
    __local float l_v_scale [BLOCK_N][DV_Q8_BLOCKS_PREFILL];
#else
    __local half4 l_v[BLOCK_N][DV_VEC];
#endif

    // --- KV iteration --------------------------------------------------------
    for (int k_start = 0; k_start < n_kv; k_start += BLOCK_N) {
        // Skip fully-masked KV blocks before loading K/V tiles. blk_base[k] is
        // uniform across all threads in the WG (same pointer, same k_start), so
        // the continue is a uniform branch — safe with local barriers below.
        // For causal PP this cuts ~50% of KV global memory reads.
        char blk_cur = 1;
        if (blk_base != NULL) {
            blk_cur = blk_base[k_start / BLOCK_N];
            if (blk_cur == 0) continue;
        }

        // K tile load.
        {
#ifdef FA_HAVE_INT_DOT
            // Pack q8_0 quants into 8 uints; store scale once per block.
            // One thread per q8_0 block amortises the scale load.
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
                    l_k_scale[row][blk] = df;
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        uint k_packed =
                              (uint) qs[j*4 + 0]        |
                             ((uint) qs[j*4 + 1]) <<  8 |
                             ((uint) qs[j*4 + 2]) << 16 |
                             ((uint) qs[j*4 + 3]) << 24;
                        l_k_packed[row][blk * 8 + j] = k_packed;
                    }
                } else {
                    l_k_scale[row][blk] = 0.0f;
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) l_k_packed[row][blk * 8 + j] = 0u;
                }
            }
#else
            // Fallback: dequant q8_0 → half in local memory.
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
        // V tile load — strategy-dependent.
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
            // Default: dequant V → half in local memory.
            const int v_blocks_per_row = DV / QK8_0;
            const int n_blocks_total = BLOCK_N * v_blocks_per_row;
            for (int i = tid; i < n_blocks_total; i += WG_SIZE) {
                const int row = i / v_blocks_per_row;
                const int blk = i % v_blocks_per_row;
                const int v_row_idx = k_start + row;
                if (v_row_idx < n_kv) {
                    const ulong v_row_off = batch_idx * v_nb3 + head_kv_idx * v_nb2 + v_row_idx * v_nb1;
                    const global char * blk_ptr = v_base + v_row_off + blk * Q8_0_BLOCK_SIZE;
                    const float df = (float) vload_half(0, (const global half *) blk_ptr);
                    const global char * qs = blk_ptr + 2;
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const float4 v = df * (float4)((float) qs[j*4 + 0],
                                                       (float) qs[j*4 + 1],
                                                       (float) qs[j*4 + 2],
                                                       (float) qs[j*4 + 3]);
                        l_v[row][blk * 8 + j] = (half4)((half) v.s0, (half) v.s1, (half) v.s2, (half) v.s3);
                    }
                } else {
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) l_v[row][blk * 8 + j] = (half4)(0.0h);
                }
            }
        }
#endif
        barrier(CLK_LOCAL_MEM_FENCE);

        // --- QK dot + online softmax (j += 4 unroll, mirrors f32_f16) ---
        // Each thread computes partial s0..s3 over its owned SPLIT_DK_Q8_BLOCKS
        // slice of DK (N_SPLIT==1: whole DK). For N_SPLIT>1 the partials are
        // summed across split_idx threads via sub_group_shuffle_xor.
        // Mask/causal/softmax are computed identically on every split_idx, so
        // (m_i, l_i) evolve the same way on each — no extra barriers needed.
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
                    #pragma unroll
                    for (int g = 0; g < 8; ++g) {
                        const uint qp = q_packed_pf[b_local * 8 + g];
                        sum0 = dot_acc_sat_4x8packed_ss_int(qp, l_k_packed[j  ][b * 8 + g], sum0);
                        sum1 = dot_acc_sat_4x8packed_ss_int(qp, l_k_packed[j+1][b * 8 + g], sum1);
                        sum2 = dot_acc_sat_4x8packed_ss_int(qp, l_k_packed[j+2][b * 8 + g], sum2);
                        sum3 = dot_acc_sat_4x8packed_ss_int(qp, l_k_packed[j+3][b * 8 + g], sum3);
                    }
                    const float qd = q_d_pf[b_local];
                    s0 += (float)sum0 * qd * l_k_scale[j  ][b];
                    s1 += (float)sum1 * qd * l_k_scale[j+1][b];
                    s2 += (float)sum2 * qd * l_k_scale[j+2][b];
                    s3 += (float)sum3 * qd * l_k_scale[j+3][b];
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
                // Reduce partials across the N_SPLIT threads that share this
                // query row. Power-of-2 N_SPLIT uses shuffle_xor butterfly;
                // N_SPLIT=3 (DK=96 case, where DK_QK_BLOCKS=3) uses explicit
                // 3-way shuffle since butterfly doesn't cover a 3-lane group.
                #if (N_SPLIT & (N_SPLIT - 1)) == 0
                    #pragma unroll
                    for (int step = 1; step < N_SPLIT; step <<= 1) {
                        s0 += sub_group_shuffle_xor(s0, step);
                        s1 += sub_group_shuffle_xor(s1, step);
                        s2 += sub_group_shuffle_xor(s2, step);
                        s3 += sub_group_shuffle_xor(s3, step);
                    }
                #else
                    // 3-way reduction: each triplet of adjacent lanes (base+0,1,2)
                    // shares a query row. Each thread reads all three lanes'
                    // partials and sums. Requires all 3 lanes to be in the same
                    // subgroup → WG_SIZE must be a multiple of 3 and ≤ subgroup size.
                    const uint tri_base = (get_sub_group_local_id() / N_SPLIT) * N_SPLIT;
                    s0 = sub_group_shuffle(s0, tri_base + 0) + sub_group_shuffle(s0, tri_base + 1) + sub_group_shuffle(s0, tri_base + 2);
                    s1 = sub_group_shuffle(s1, tri_base + 0) + sub_group_shuffle(s1, tri_base + 1) + sub_group_shuffle(s1, tri_base + 2);
                    s2 = sub_group_shuffle(s2, tri_base + 0) + sub_group_shuffle(s2, tri_base + 1) + sub_group_shuffle(s2, tri_base + 2);
                    s3 = sub_group_shuffle(s3, tri_base + 0) + sub_group_shuffle(s3, tri_base + 1) + sub_group_shuffle(s3, tri_base + 2);
                #endif
                if (!query_valid) { s0 = -INFINITY; s1 = -INFINITY; s2 = -INFINITY; s3 = -INFINITY; }
#endif
                s0 *= scale; s1 *= scale; s2 *= scale; s3 *= scale;

                if (is_causal) {
                    const int causal_limit = n_kv - n_q + my_query_row;
                    if (k_row0 > causal_limit) s0 = -INFINITY;
                    if (k_row1 > causal_limit) s1 = -INFINITY;
                    if (k_row2 > causal_limit) s2 = -INFINITY;
                    if (k_row3 > causal_limit) s3 = -INFINITY;
                }
                if (k_row0 >= n_kv) s0 = -INFINITY;
                if (k_row1 >= n_kv) s1 = -INFINITY;
                if (k_row2 >= n_kv) s2 = -INFINITY;
                if (k_row3 >= n_kv) s3 = -INFINITY;

                // Skip per-row mask when blk_cur==2 (fully unmasked tile —
                // all mask entries are 0.0h with no -inf). Saves BLOCK_M ×
                // BLOCK_N mask-lookup half reads per fully-unmasked tile.
                if (mask_base != NULL && blk_cur != 2) {
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
                const ACC_TYPE scale_prev = native_exp(m_i - m_new);
                const ACC_TYPE p0         = native_exp(s0 - m_new);
                const ACC_TYPE p1         = native_exp(s1 - m_new);
                const ACC_TYPE p2         = native_exp(s2 - m_new);
                const ACC_TYPE p3         = native_exp(s3 - m_new);

#if FA_V_STRATEGY == 2
                // V in int8 packed in local; unpack per block inline.
                // Each thread owns SPLIT_DV_VEC float4 output lanes = its
                // share of the DV dimension (DV_Q8_BLOCKS_PREFILL/N_SPLIT
                // blocks per thread).
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
#else  // FA_V_STRATEGY == 0 (default: half-local)
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

    // --- Write output --------------------------------------------------------
    // With N_SPLIT>1 all N_SPLIT threads for a query row hold identical
    // (m_i, l_i) state (shuffle-reduced QK made them evolve the same way).
    // Each thread writes its SPLIT_DV_VEC slice at offset dv_off_vec.
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

// Flash-Decoding Pass 2: merge partials from all splits into the final output.
// Type-agnostic — operates only on float partials. Identical to the merge kernel
// in flash_attn_f32_f16.cl; duplicated here so q8_0 FD is self-contained.
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
        ACC_TYPE m = -INFINITY;
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
            if (m_c > -INFINITY) {
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
        if (m_c <= -INFINITY) continue;
        const global float4 * rec_oc = (const global float4 *) (rec_c + 2);
        const ACC_TYPE scale_c = exp(m_c - m_final);
        o = mad((ACC_TYPE4)(scale_c), rec_oc[lane], o);
    }
    o = o * l_inv;

    const ulong o_row_offset = (ulong) batch_idx * o_nb3 + (ulong) q_idx * o_nb2 + (ulong) head_idx * o_nb1;
    global O_DATA_TYPE4 * o_row = (global O_DATA_TYPE4 *) ((global char *) o_void + o_offset + o_row_offset);
    o_row[lane] = CONVERT_O_DATA4(o);
}

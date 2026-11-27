#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#else
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
#else
#define REQD_SUBGROUP_SIZE_64
#endif

#ifdef cl_khr_subgroup_shuffle
#pragma OPENCL EXTENSION cl_khr_subgroup_shuffle : enable
#elif defined(cl_qcom_subgroup_shuffle)
#pragma OPENCL EXTENSION cl_qcom_subgroup_shuffle : enable
#define sub_group_shuffle_xor(val, mask) qcom_sub_group_shuffle_xor((val), (mask), CLK_SUB_GROUP_SHUFFLE_WIDTH_WAVE_SIZE_QCOM, 0.0f)
#endif

#define GELU_COEF_A     0.044715f
#define SQRT_2_OVER_PI  0.79788456080286535587989211986876f

// Single-token f16 x f32 GEMV for small matrices: dst[r] = W[r,:] . x.
//
// Built for the gemma-4 per-layer-embedding matmuls (256 x 2560 and 2560 x 256), where
// the general decode GEMVs leave bandwidth on the table: a 512-byte row (K = 256) fills
// only half of a 64-lane subgroup's half8 loads, and 256 rows of 2560 give few rows to
// spread over the CUs. Here a row is split over LPR lanes, so one pass of a subgroup
// covers 64/LPR rows with every lane loading 16 bytes, and each subgroup keeps RB such
// passes in flight. The partials are summed with a shuffle-xor butterfly across the LPR
// lanes of a row.
//
// mode 1 applies the GELU(tanh) * gate[r] epilogue (the per-layer inp_gate -> gelu ->
// mul chain), so the two element-wise ops that follow need no dispatch of their own.
//
// Requires K % 512 == 0 (LPR 64) or K == 256 (LPR 32); W rows contiguous (row stride K).
// LPR is a compile-time constant with an unrolled butterfly: a shuffle-xor loop with a
// runtime trip count crashes clBuildProgram on the Adreno 850 compiler (E17).
#define PL_GEMV(NAME, LPR, RB)                                                              \
REQD_SUBGROUP_SIZE_64                                                                       \
kernel void NAME(                                                                           \
        global const half  * W,                                                             \
        ulong                offW,                                                          \
        global const float * x,                                                             \
        ulong                offx,                                                          \
        global const float * gate,                                                          \
        ulong                offg,                                                          \
        global float       * dst,                                                           \
        ulong                offd,                                                          \
        int                  K,                                                             \
        int                  M,                                                             \
        int                  mode) {                                                        \
    W    = (global const half  *)((global const char *)W    + offW);                        \
    x    = (global const float *)((global const char *)x    + offx);                        \
    gate = (global const float *)((global const char *)gate + offg);                        \
    dst  = (global float       *)((global char       *)dst  + offd);                        \
                                                                                            \
    const uint lane = get_sub_group_local_id();                                             \
    const uint rin  = lane / LPR;                                                           \
    const uint col  = lane % LPR;                                                           \
    const uint sg   = get_group_id(0) * get_num_sub_groups() + get_sub_group_id();          \
    const uint row0 = sg * (RB * (64 / LPR));                                               \
                                                                                            \
    float acc[RB];                                                                          \
    for (int b = 0; b < RB; ++b) { acc[b] = 0.0f; }                                         \
                                                                                            \
    for (uint k = col * 8u; k < (uint)K; k += LPR * 8u) {                                   \
        const float8 xv = vload8(0, x + k);                                                 \
        for (int b = 0; b < RB; ++b) {                                                      \
            const uint row = min(row0 + b * (64 / LPR) + rin, (uint)M - 1u);                \
            const float8 wv = convert_float8(vload8(0, W + (ulong)row * K + k));            \
            acc[b] += dot(wv.lo, xv.lo) + dot(wv.hi, xv.hi);                                \
        }                                                                                   \
    }                                                                                       \
                                                                                            \
    for (int b = 0; b < RB; ++b) {                                                          \
        float v = acc[b];                                                                   \
        if (LPR > 32) { v += sub_group_shuffle_xor(v, 32u); }                               \
        if (LPR > 16) { v += sub_group_shuffle_xor(v, 16u); }                               \
        if (LPR >  8) { v += sub_group_shuffle_xor(v,  8u); }                               \
        if (LPR >  4) { v += sub_group_shuffle_xor(v,  4u); }                               \
        if (LPR >  2) { v += sub_group_shuffle_xor(v,  2u); }                               \
        if (LPR >  1) { v += sub_group_shuffle_xor(v,  1u); }                               \
        const uint row = row0 + b * (64 / LPR) + rin;                                       \
        if (col == 0u && row < (uint)M) {                                                   \
            if (mode == 1) {                                                                \
                v = 0.5f*v*(1.0f + tanh(SQRT_2_OVER_PI*v*(1.0f + GELU_COEF_A*v*v))) * gate[row]; \
            }                                                                               \
            dst[row] = v;                                                                   \
        }                                                                                   \
    }                                                                                       \
}

// K % 512 == 0: a row over all 64 lanes
PL_GEMV(kernel_mul_mv_f16_f32_pl_l64_rb1, 64, 1)
PL_GEMV(kernel_mul_mv_f16_f32_pl_l64_rb2, 64, 2)
PL_GEMV(kernel_mul_mv_f16_f32_pl_l64_rb4, 64, 4)
// K == 256: two rows per subgroup pass
PL_GEMV(kernel_mul_mv_f16_f32_pl_l32_rb2, 32, 2)
PL_GEMV(kernel_mul_mv_f16_f32_pl_l32_rb4, 32, 4)
PL_GEMV(kernel_mul_mv_f16_f32_pl_l32_rb8, 32, 8)

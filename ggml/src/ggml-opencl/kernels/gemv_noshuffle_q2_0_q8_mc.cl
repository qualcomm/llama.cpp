#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64 __attribute__((qcom_reqd_sub_group_size("half")))
#endif

// q2_0 GEMV for 2 or 4 columns on the transposed SOA layout of kernel_gemv_noshuffle_q2_0_f32
// (quants [K/16][M] uint32, weight m of a word at bits 2m..2m+1, value q - 1; scales [K/64][M]
// half), reading each weight word once for all N_COLS columns.
//
// The activations are int8 (kernel_quant_q2a_f32, one scale per 64), stored so that word 4j+s
// of a 64-block holds activations 16j+s, +4, +8, +12: then (w >> 2s) & 0x03030303 is the
// matching 4 weights of word j, and the -1 offset is exact, sum (q-1)*a = dot(q, a) - sum(a).
#define QK2_0 64

#ifndef N_COLS
#define N_COLS 4
#endif

#ifndef N_SIMDGROUP
#define N_SIMDGROUP 4
#endif

// each WI quantizes one 64-block of one column: 16 packed words + (da, da * sum q)
kernel void kernel_quant_q2a_f32(
        global const float * src1,
        ulong offset1,
        global uint * qa,
        global float2 * qd,
        int K)
{
    const int blk = get_global_id(0);
    const int col = get_global_id(1);
    if (blk >= K / QK2_0) {
        return;
    }

    global const float4 * a = (global const float4 *)((global const char *)src1 + offset1) + ((ulong)col * K + blk * QK2_0) / 4;

    float4 v[16];
    float amax = 0.0f;
    for (int i = 0; i < 16; ++i) {
        v[i] = a[i];
        float4 t = fabs(v[i]);
        amax = fmax(amax, fmax(fmax(t.x, t.y), fmax(t.z, t.w)));
    }
    const float da  = amax / 127.0f;
    const float ida = amax > 0.0f ? 127.0f / amax : 0.0f;

    global uint * q = qa + ((ulong)col * K + blk * QK2_0) / 4;
    int sumq = 0;
    for (int g = 0; g < 4; ++g) {
        // texels 4g..4g+3 hold activations 16g..16g+15; word s takes component s of each
        int4 c0 = convert_int4_sat_rte(v[4*g + 0] * ida);
        int4 c1 = convert_int4_sat_rte(v[4*g + 1] * ida);
        int4 c2 = convert_int4_sat_rte(v[4*g + 2] * ida);
        int4 c3 = convert_int4_sat_rte(v[4*g + 3] * ida);
        int4 s4 = c0 + c1 + c2 + c3;
        sumq += s4.x + s4.y + s4.z + s4.w;
        q[4*g + 0] = (uint)(c0.x & 0xff) | ((uint)(c1.x & 0xff) << 8) | ((uint)(c2.x & 0xff) << 16) | ((uint)(c3.x & 0xff) << 24);
        q[4*g + 1] = (uint)(c0.y & 0xff) | ((uint)(c1.y & 0xff) << 8) | ((uint)(c2.y & 0xff) << 16) | ((uint)(c3.y & 0xff) << 24);
        q[4*g + 2] = (uint)(c0.z & 0xff) | ((uint)(c1.z & 0xff) << 8) | ((uint)(c2.z & 0xff) << 16) | ((uint)(c3.z & 0xff) << 24);
        q[4*g + 3] = (uint)(c0.w & 0xff) | ((uint)(c1.w & 0xff) << 8) | ((uint)(c2.w & 0xff) << 16) | ((uint)(c3.w & 0xff) << 24);
    }
    qd[(ulong)col * (K / QK2_0) + blk] = (float2)(da, da * (float)sumq);
}

// 2 rows per fiber (one uint2 texel = the same quant word of rows 2g, 2g+1)
#define DOT2(acc, w, a) \
    acc.x = dot_acc_sat_4x8packed_us_int((w).x, (a), acc.x); \
    acc.y = dot_acc_sat_4x8packed_us_int((w).y, (a), acc.y);

// word j (weights 16j..16j+15) of the block against activation words 4j..4j+3 of column c,
// held by fiber 16c + 4j + s
#define WORD(j, w)                                                                  \
    {                                                                               \
        const uint2 m0 = (w) & 0x03030303u;                                         \
        const uint2 m1 = ((w) >> 2) & 0x03030303u;                                  \
        const uint2 m2 = ((w) >> 4) & 0x03030303u;                                  \
        const uint2 m3 = ((w) >> 6) & 0x03030303u;                                  \
        for (int c = 0; c < N_COLS; ++c) {                                          \
            DOT2(acc[c], m0, sub_group_broadcast(aw, 16*c + 4*(j) + 0));            \
            DOT2(acc[c], m1, sub_group_broadcast(aw, 16*c + 4*(j) + 1));            \
            DOT2(acc[c], m2, sub_group_broadcast(aw, 16*c + 4*(j) + 2));            \
            DOT2(acc[c], m3, sub_group_broadcast(aw, 16*c + 4*(j) + 3));            \
        }                                                                           \
    }

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_gemv_noshuffle_q2_0_q8_mc(
        read_only image1d_buffer_t src0_q,
        global half  * src0_d,
        global uint  * qa,
        global float2 * qd,
        global float * dst,
        ulong offsetd,
        int K,
        int M,
        int col0,
        int ncols,
        int ne0)
{
    const uint groupId = get_local_id(1);
    const uint gid     = get_global_id(0);   // row pair: rows 2*gid, 2*gid+1
    const ushort slid  = get_sub_group_local_id();

    const uint G     = M / 2;
    const uint gid_s = min(gid, G - 1);
    const uint nb    = K / QK2_0;

    // fiber 16c + i holds activation word i of column col0 + c (fibers past N_COLS*16 idle)
    const uint my_c = slid / 16;
    const uint my_w = slid % 16;
    // columns past ncols (a ragged last group) re-read the last one and are not stored
    global const uint   * qa_c = qa + (ulong)(col0 + min(my_c, (uint)(ncols - 1))) * (K / 4) + my_w;

    float2 total[N_COLS];
    for (int c = 0; c < N_COLS; ++c) {
        total[c] = 0.0f;
    }

    #pragma unroll 1
    for (uint kb = groupId; kb < nb; kb += N_SIMDGROUP) {
        const float2 d = convert_float2(vload2(0, src0_d + kb * M + gid_s * 2));
        const uint aw = qa_c[kb * 16];

        const uint2 w0 = read_imageui(src0_q, (int)(gid_s + kb * 4 * G + G * 0)).xy;
        const uint2 w1 = read_imageui(src0_q, (int)(gid_s + kb * 4 * G + G * 1)).xy;
        const uint2 w2 = read_imageui(src0_q, (int)(gid_s + kb * 4 * G + G * 2)).xy;
        const uint2 w3 = read_imageui(src0_q, (int)(gid_s + kb * 4 * G + G * 3)).xy;

        int2 acc[N_COLS];
        for (int c = 0; c < N_COLS; ++c) {
            acc[c] = 0;
        }
        WORD(0, w0)
        WORD(1, w1)
        WORD(2, w2)
        WORD(3, w3)

        for (int c = 0; c < N_COLS; ++c) {
            const float2 s = qd[(ulong)(col0 + min(c, ncols - 1)) * nb + kb];
            total[c] += d * (s.x * convert_float2(acc[c]) - s.y);
        }
    }

#if N_SIMDGROUP > 1
    local float2 reduceLM[N_COLS][SIMDGROUP_WIDTH * (N_SIMDGROUP - 1)];
    if (groupId > 0) {
        for (int c = 0; c < N_COLS; ++c) {
            reduceLM[c][SIMDGROUP_WIDTH * (groupId - 1) + slid] = total[c];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (groupId == 0) {
        for (uint w = 0; w < N_SIMDGROUP - 1; ++w) {
            for (int c = 0; c < N_COLS; ++c) {
                total[c] += reduceLM[c][SIMDGROUP_WIDTH * w + slid];
            }
        }
    }
#endif

    if (groupId == 0 && gid < G) {
        global float * d0 = (global float *)((global char *)dst + offsetd);
        for (int c = 0; c < N_COLS && c < ncols; ++c) {
            vstore2(total[c], gid, d0 + (ulong)(col0 + c) * ne0);
        }
    }
}

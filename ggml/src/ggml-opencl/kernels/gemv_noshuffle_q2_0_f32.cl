#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64 __attribute__((qcom_reqd_sub_group_size("half")))
#endif

// q2_0 GEMV on the transposed SOA layout (the q1_0 noshuffle scheme): quants as [K/16][M]
// uint32 (16 weights each, weight m at bits 2m..2m+1, value q - 1), scales as [K/64][M] half,
// activations broadcast across the wave. Each fiber owns Q2_ROWS adjacent rows: one texel of
// the Q2_ROWS-channel uint image is the same quant word of those rows, so every broadcast
// activation is used Q2_ROWS times (q2_0 decodes 4 weights per byte, twice q4_0's ALU per byte).
#define QK2_0 64

#ifndef Q2_ROWS
#define Q2_ROWS 2
#endif

#if Q2_ROWS == 4
#define uintR       uint4
#define floatR      float4
#define convert_floatR convert_float4
#define READ_Q(img, i) read_imageui(img, (int)(i))
#define LOAD_D(p)   convert_float4(vload4(0, p))
#elif Q2_ROWS == 2
#define uintR       uint2
#define floatR      float2
#define convert_floatR convert_float2
#define READ_Q(img, i) read_imageui(img, (int)(i)).xy
#define LOAD_D(p)   convert_float2(vload2(0, p))
#else
#define uintR       uint
#define floatR      float
#define convert_floatR convert_float
#define READ_Q(img, i) read_imageui(img, (int)(i)).x
#define LOAD_D(p)   ((float)*(p))
#endif

// Waves that split K; tall M has enough rows on its own and is built with -DN_SIMDGROUP=1.
#ifndef N_SIMDGROUP
#define N_SIMDGROUP 4
#endif

// 16 weights of one quant word (per row) against activations regB.s0..s7 of fibers lb, lb+1
#define dequantizeAccum_q2(acc, w, regB, lb)                                                  \
    acc += convert_floatR(((w) >>  0) & 3u) * sub_group_broadcast(regB.s0, lb+0);              \
    acc += convert_floatR(((w) >>  2) & 3u) * sub_group_broadcast(regB.s1, lb+0);              \
    acc += convert_floatR(((w) >>  4) & 3u) * sub_group_broadcast(regB.s2, lb+0);              \
    acc += convert_floatR(((w) >>  6) & 3u) * sub_group_broadcast(regB.s3, lb+0);              \
    acc += convert_floatR(((w) >>  8) & 3u) * sub_group_broadcast(regB.s4, lb+0);              \
    acc += convert_floatR(((w) >> 10) & 3u) * sub_group_broadcast(regB.s5, lb+0);              \
    acc += convert_floatR(((w) >> 12) & 3u) * sub_group_broadcast(regB.s6, lb+0);              \
    acc += convert_floatR(((w) >> 14) & 3u) * sub_group_broadcast(regB.s7, lb+0);              \
    acc += convert_floatR(((w) >> 16) & 3u) * sub_group_broadcast(regB.s0, lb+1);              \
    acc += convert_floatR(((w) >> 18) & 3u) * sub_group_broadcast(regB.s1, lb+1);              \
    acc += convert_floatR(((w) >> 20) & 3u) * sub_group_broadcast(regB.s2, lb+1);              \
    acc += convert_floatR(((w) >> 22) & 3u) * sub_group_broadcast(regB.s3, lb+1);              \
    acc += convert_floatR(((w) >> 24) & 3u) * sub_group_broadcast(regB.s4, lb+1);              \
    acc += convert_floatR(((w) >> 26) & 3u) * sub_group_broadcast(regB.s5, lb+1);              \
    acc += convert_floatR(((w) >> 28) & 3u) * sub_group_broadcast(regB.s6, lb+1);              \
    acc += convert_floatR(((w) >> 30) & 3u) * sub_group_broadcast(regB.s7, lb+1);

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
__kernel void kernel_gemv_noshuffle_q2_0_f32(
        read_only  image1d_buffer_t src0_q,
        global half  * src0_d,
        read_only  image1d_buffer_t src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne10,
        int ne12,
        int ne0,
        int ne1,
        int r2,
        int r3)
{
    uint groupId = get_local_id(1);
    uint gid     = get_global_id(0);   // row group: rows Q2_ROWS*gid ..
    uint col     = get_global_id(2);   // src1 column; ne1 > 1 is run one column per z slice
    ushort slid  = get_sub_group_local_id();

    uint K = ne00;
    uint M = ne01;
    uint G = M / Q2_ROWS;              // row groups (texels per quant line)

    uint LINE_STRIDE_A  = G;
    uint BLOCK_STRIDE_A = 4 * G;       // 4 quant words per 64-weight block
    uint B_OFF          = col * K/4;   // src1 texels (float4) of this column

    // The x-grid is padded to whole waves; clamp the group for every fetch so the tail fibers
    // stay in range (and active for the broadcasts). The store below is guarded.
    uint gid_s = min(gid, G - 1);

    uintR  w0, w1, w2, w3;
    float8 regB;

    floatR totalSum = 0.0f;

    #pragma unroll 1
    for (uint kb = groupId; kb < (K / QK2_0); kb += N_SIMDGROUP) {
        floatR scale = LOAD_D(src0_d + kb * M + gid_s * Q2_ROWS);

        // first 8 fibers load 8 activations each -> the 64 of this block
        if (slid < 8) {
            regB.s0123 = read_imagef(src1, (int)(B_OFF + slid * 2 + kb * 16));
            regB.s4567 = read_imagef(src1, (int)(B_OFF + slid * 2 + kb * 16 + 1));
        }

        w0 = READ_Q(src0_q, gid_s + kb * BLOCK_STRIDE_A + LINE_STRIDE_A * 0);
        w1 = READ_Q(src0_q, gid_s + kb * BLOCK_STRIDE_A + LINE_STRIDE_A * 1);
        w2 = READ_Q(src0_q, gid_s + kb * BLOCK_STRIDE_A + LINE_STRIDE_A * 2);
        w3 = READ_Q(src0_q, gid_s + kb * BLOCK_STRIDE_A + LINE_STRIDE_A * 3);

        // sum of q*y per row, and of y (shared) for the -1 offset
        floatR acc = 0.0f;
        dequantizeAccum_q2(acc, w0, regB, 0);
        dequantizeAccum_q2(acc, w1, regB, 2);
        dequantizeAccum_q2(acc, w2, regB, 4);
        dequantizeAccum_q2(acc, w3, regB, 6);

        float part = (slid < 8) ? (regB.s0 + regB.s1 + regB.s2 + regB.s3 +
                                   regB.s4 + regB.s5 + regB.s6 + regB.s7) : 0.0f;
        float sumy = sub_group_reduce_add(part);

        totalSum += scale * (acc - sumy);
    }

#if N_SIMDGROUP > 1
    // reduction in local memory across the K-split waves
    local floatR reduceLM[SIMDGROUP_WIDTH * (N_SIMDGROUP - 1)];
    if (groupId > 0) {
        reduceLM[SIMDGROUP_WIDTH * (groupId - 1) + slid] = totalSum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (groupId == 0) {
        for (uint w = 0; w < N_SIMDGROUP - 1; ++w) {
            totalSum += reduceLM[SIMDGROUP_WIDTH * w + slid];
        }
    }
#endif // N_SIMDGROUP > 1

    if (groupId == 0 && gid < G) {
        dst = (global float*)((global char*)dst + offsetd) + (ulong)col * ne0;
#if Q2_ROWS == 4
        vstore4(totalSum, gid, dst);
#elif Q2_ROWS == 2
        vstore2(totalSum, gid, dst);
#else
        dst[gid] = totalSum;
#endif
    }
}

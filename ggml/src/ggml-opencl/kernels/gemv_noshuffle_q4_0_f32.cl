#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64 __attribute__((qcom_reqd_sub_group_size("half")))
#endif

// assume
#define QK4_0 32
#define N_SIMDGROUP 4

#define dequantizeBlockAccum_ns_sgbroadcast_1_hi(total_sums, bits4, scale, y) \
    float shared_y; \
    shared_y = sub_group_broadcast(y.s0, 0); \
    total_sums.s0 += ((bits4.s0 & 0x000F) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += ((bits4.s1 & 0x000F) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s1, 0); \
    total_sums.s0 += (((bits4.s0 & 0x00F0) >> 4) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s1 & 0x00F0) >> 4) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s2, 0); \
    total_sums.s0 += (((bits4.s0 & 0x0F00) >> 8) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s1 & 0x0F00) >> 8) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s3, 0); \
    total_sums.s0 += (((bits4.s0 & 0xF000) >> 12) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s1 & 0xF000) >> 12) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s4, 0); \
    total_sums.s0 += ((bits4.s2 & 0x000F) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += ((bits4.s3 & 0x000F) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s5, 0); \
    total_sums.s0 += (((bits4.s2 & 0x00F0) >> 4) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s3 & 0x00F0) >> 4) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s6, 0); \
    total_sums.s0 += (((bits4.s2 & 0x0F00) >> 8) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s3 & 0x0F00) >> 8) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s7, 0); \
    total_sums.s0 += (((bits4.s2 & 0xF000) >> 12) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s3 & 0xF000) >> 12) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s0, 1); \
    total_sums.s0 += ((bits4.s4 & 0x000F) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += ((bits4.s5 & 0x000F) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s1, 1); \
    total_sums.s0 += (((bits4.s4 & 0x00F0) >> 4) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s5 & 0x00F0) >> 4) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s2, 1); \
    total_sums.s0 += (((bits4.s4 & 0x0F00) >> 8) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s5 & 0x0F00) >> 8) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s3, 1); \
    total_sums.s0 += (((bits4.s4 & 0xF000) >> 12) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s5 & 0xF000) >> 12) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s4, 1); \
    total_sums.s0 += ((bits4.s6 & 0x000F) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += ((bits4.s7 & 0x000F) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s5, 1); \
    total_sums.s0 += (((bits4.s6 & 0x00F0) >> 4) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s7 & 0x00F0) >> 4) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s6, 1); \
    total_sums.s0 += (((bits4.s6 & 0x0F00) >> 8) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s7 & 0x0F00) >> 8) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s7, 1); \
    total_sums.s0 += (((bits4.s6 & 0xF000) >> 12) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s7 & 0xF000) >> 12) - 8) * scale.s1 * shared_y; \


#define dequantizeBlockAccum_ns_sgbroadcast_1_lo(total_sums, bits4, scale, y) \
    shared_y = sub_group_broadcast(y.s0, 2); \
    total_sums.s0 += ((bits4.s0 & 0x000F) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += ((bits4.s1 & 0x000F) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s1, 2); \
    total_sums.s0 += (((bits4.s0 & 0x00F0) >> 4) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s1 & 0x00F0) >> 4) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s2, 2); \
    total_sums.s0 += (((bits4.s0 & 0x0F00) >> 8) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s1 & 0x0F00) >> 8) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s3, 2); \
    total_sums.s0 += (((bits4.s0 & 0xF000) >> 12) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s1 & 0xF000) >> 12) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s4, 2); \
    total_sums.s0 += ((bits4.s2 & 0x000F) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += ((bits4.s3 & 0x000F) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s5, 2); \
    total_sums.s0 += (((bits4.s2 & 0x00F0) >> 4) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s3 & 0x00F0) >> 4) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s6, 2); \
    total_sums.s0 += (((bits4.s2 & 0x0F00) >> 8) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s3 & 0x0F00) >> 8) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s7, 2); \
    total_sums.s0 += (((bits4.s2 & 0xF000) >> 12) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s3 & 0xF000) >> 12) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s0, 3); \
    total_sums.s0 += ((bits4.s4 & 0x000F) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += ((bits4.s5 & 0x000F) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s1, 3); \
    total_sums.s0 += (((bits4.s4 & 0x00F0) >> 4) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s5 & 0x00F0) >> 4) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s2, 3); \
    total_sums.s0 += (((bits4.s4 & 0x0F00) >> 8) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s5 & 0x0F00) >> 8) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s3, 3); \
    total_sums.s0 += (((bits4.s4 & 0xF000) >> 12) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s5 & 0xF000) >> 12) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s4, 3); \
    total_sums.s0 += ((bits4.s6 & 0x000F) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += ((bits4.s7 & 0x000F) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s5, 3); \
    total_sums.s0 += (((bits4.s6 & 0x00F0) >> 4) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s7 & 0x00F0) >> 4) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s6, 3); \
    total_sums.s0 += (((bits4.s6 & 0x0F00) >> 8) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s7 & 0x0F00) >> 8) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s7, 3); \
    total_sums.s0 += (((bits4.s6 & 0xF000) >> 12) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s7 & 0xF000) >> 12) - 8) * scale.s1 * shared_y; \


#define dequantizeBlockAccum_ns_sgbroadcast_8_hi(total_sums, bits4, scale, y) \
    float8 shared_y; \
    shared_y = sub_group_broadcast(y, 0); \
    total_sums.s0 += ((bits4.s0 & 0x000F) - 8) * scale.s0 * shared_y.s0; \
    total_sums.s0 += (((bits4.s0 & 0x00F0) >> 4) - 8) * scale.s0 * shared_y.s1; \
    total_sums.s0 += (((bits4.s0 & 0x0F00) >> 8) - 8) * scale.s0 * shared_y.s2; \
    total_sums.s0 += (((bits4.s0 & 0xF000) >> 12) - 8) * scale.s0 * shared_y.s3; \
    total_sums.s0 += ((bits4.s2 & 0x000F) - 8) * scale.s0 * shared_y.s4; \
    total_sums.s0 += (((bits4.s2 & 0x00F0) >> 4) - 8) * scale.s0 * shared_y.s5; \
    total_sums.s0 += (((bits4.s2 & 0x0F00) >> 8) - 8) * scale.s0 * shared_y.s6; \
    total_sums.s0 += (((bits4.s2 & 0xF000) >> 12) - 8) * scale.s0 * shared_y.s7; \
    total_sums.s1 += ((bits4.s1 & 0x000F) - 8) * scale.s1 * shared_y.s0; \
    total_sums.s1 += (((bits4.s1 & 0x00F0) >> 4) - 8) * scale.s1 * shared_y.s1; \
    total_sums.s1 += (((bits4.s1 & 0x0F00) >> 8) - 8) * scale.s1 * shared_y.s2; \
    total_sums.s1 += (((bits4.s1 & 0xF000) >> 12) - 8) * scale.s1 * shared_y.s3; \
    total_sums.s1 += ((bits4.s3 & 0x000F) - 8) * scale.s1 * shared_y.s4; \
    total_sums.s1 += (((bits4.s3 & 0x00F0) >> 4) - 8) * scale.s1 * shared_y.s5; \
    total_sums.s1 += (((bits4.s3 & 0x0F00) >> 8) - 8) * scale.s1 * shared_y.s6; \
    total_sums.s1 += (((bits4.s3 & 0xF000) >> 12) - 8) * scale.s1 * shared_y.s7; \
    shared_y = sub_group_broadcast(y, 1); \
    total_sums.s0 += ((bits4.s4 & 0x000F) - 8) * scale.s0 * shared_y.s0; \
    total_sums.s0 += (((bits4.s4 & 0x00F0) >> 4) - 8) * scale.s0 * shared_y.s1; \
    total_sums.s0 += (((bits4.s4 & 0x0F00) >> 8) - 8) * scale.s0 * shared_y.s2; \
    total_sums.s0 += (((bits4.s4 & 0xF000) >> 12) - 8) * scale.s0 * shared_y.s3; \
    total_sums.s0 += ((bits4.s6 & 0x000F) - 8) * scale.s0 * shared_y.s4; \
    total_sums.s0 += (((bits4.s6 & 0x00F0) >> 4) - 8) * scale.s0 * shared_y.s5; \
    total_sums.s0 += (((bits4.s6 & 0x0F00) >> 8) - 8) * scale.s0 * shared_y.s6; \
    total_sums.s0 += (((bits4.s6 & 0xF000) >> 12) - 8) * scale.s0 * shared_y.s7; \
    total_sums.s1 += ((bits4.s5 & 0x000F) - 8) * scale.s1 * shared_y.s0; \
    total_sums.s1 += (((bits4.s5 & 0x00F0) >> 4) - 8) * scale.s1 * shared_y.s1; \
    total_sums.s1 += (((bits4.s5 & 0x0F00) >> 8) - 8) * scale.s1 * shared_y.s2; \
    total_sums.s1 += (((bits4.s5 & 0xF000) >> 12) - 8) * scale.s1 * shared_y.s3; \
    total_sums.s1 += ((bits4.s7 & 0x000F) - 8) * scale.s1 * shared_y.s4; \
    total_sums.s1 += (((bits4.s7 & 0x00F0) >> 4) - 8) * scale.s1 * shared_y.s5; \
    total_sums.s1 += (((bits4.s7 & 0x0F00) >> 8) - 8) * scale.s1 * shared_y.s6; \
    total_sums.s1 += (((bits4.s7 & 0xF000) >> 12) - 8) * scale.s1 * shared_y.s7; \


#define dequantizeBlockAccum_ns_sgbroadcast_8_lo(total_sums, bits4, scale, y) \
    shared_y = sub_group_broadcast(y, 2); \
    total_sums.s0 += ((bits4.s0 & 0x000F) - 8) * scale.s0 * shared_y.s0; \
    total_sums.s0 += (((bits4.s0 & 0x00F0) >> 4) - 8) * scale.s0 * shared_y.s1; \
    total_sums.s0 += (((bits4.s0 & 0x0F00) >> 8) - 8) * scale.s0 * shared_y.s2; \
    total_sums.s0 += (((bits4.s0 & 0xF000) >> 12) - 8) * scale.s0 * shared_y.s3; \
    total_sums.s0 += ((bits4.s2 & 0x000F) - 8) * scale.s0 * shared_y.s4; \
    total_sums.s0 += (((bits4.s2 & 0x00F0) >> 4) - 8) * scale.s0 * shared_y.s5; \
    total_sums.s0 += (((bits4.s2 & 0x0F00) >> 8) - 8) * scale.s0 * shared_y.s6; \
    total_sums.s0 += (((bits4.s2 & 0xF000) >> 12) - 8) * scale.s0 * shared_y.s7; \
    total_sums.s1 += ((bits4.s1 & 0x000F) - 8) * scale.s1 * shared_y.s0; \
    total_sums.s1 += (((bits4.s1 & 0x00F0) >> 4) - 8) * scale.s1 * shared_y.s1; \
    total_sums.s1 += (((bits4.s1 & 0x0F00) >> 8) - 8) * scale.s1 * shared_y.s2; \
    total_sums.s1 += (((bits4.s1 & 0xF000) >> 12) - 8) * scale.s1 * shared_y.s3; \
    total_sums.s1 += ((bits4.s3 & 0x000F) - 8) * scale.s1 * shared_y.s4; \
    total_sums.s1 += (((bits4.s3 & 0x00F0) >> 4) - 8) * scale.s1 * shared_y.s5; \
    total_sums.s1 += (((bits4.s3 & 0x0F00) >> 8) - 8) * scale.s1 * shared_y.s6; \
    total_sums.s1 += (((bits4.s3 & 0xF000) >> 12) - 8) * scale.s1 * shared_y.s7; \
    shared_y = sub_group_broadcast(y, 3); \
    total_sums.s0 += ((bits4.s4 & 0x000F) - 8) * scale.s0 * shared_y.s0; \
    total_sums.s0 += (((bits4.s4 & 0x00F0) >> 4) - 8) * scale.s0 * shared_y.s1; \
    total_sums.s0 += (((bits4.s4 & 0x0F00) >> 8) - 8) * scale.s0 * shared_y.s2; \
    total_sums.s0 += (((bits4.s4 & 0xF000) >> 12) - 8) * scale.s0 * shared_y.s3; \
    total_sums.s0 += ((bits4.s6 & 0x000F) - 8) * scale.s0 * shared_y.s4; \
    total_sums.s0 += (((bits4.s6 & 0x00F0) >> 4) - 8) * scale.s0 * shared_y.s5; \
    total_sums.s0 += (((bits4.s6 & 0x0F00) >> 8) - 8) * scale.s0 * shared_y.s6; \
    total_sums.s0 += (((bits4.s6 & 0xF000) >> 12) - 8) * scale.s0 * shared_y.s7; \
    total_sums.s1 += ((bits4.s5 & 0x000F) - 8) * scale.s1 * shared_y.s0; \
    total_sums.s1 += (((bits4.s5 & 0x00F0) >> 4) - 8) * scale.s1 * shared_y.s1; \
    total_sums.s1 += (((bits4.s5 & 0x0F00) >> 8) - 8) * scale.s1 * shared_y.s2; \
    total_sums.s1 += (((bits4.s5 & 0xF000) >> 12) - 8) * scale.s1 * shared_y.s3; \
    total_sums.s1 += ((bits4.s7 & 0x000F) - 8) * scale.s1 * shared_y.s4; \
    total_sums.s1 += (((bits4.s7 & 0x00F0) >> 4) - 8) * scale.s1 * shared_y.s5; \
    total_sums.s1 += (((bits4.s7 & 0x0F00) >> 8) - 8) * scale.s1 * shared_y.s6; \
    total_sums.s1 += (((bits4.s7 & 0xF000) >> 12) - 8) * scale.s1 * shared_y.s7; \

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
__kernel void kernel_gemv_noshuffle_q4_0_f32(
        __read_only  image1d_buffer_t src0_q,  // quantized A
        global half2  * src0_d,  // A scales
        __read_only  image1d_buffer_t src1,    // B
        ulong offset1,            // offset to B (0)
        global float * dst,     // C
        ulong offsetd,            // offset to C (0)
        int ne00,               // K
        int ne01,               // M
        int ne02,               // 1
        int ne10,               // K
        int ne12,               // 1
        int ne0,                // M
        int ne1,                // N
        int r2,                 // 1
        int r3)
{
    uint groupId = get_local_id(1);
    uint gid     = get_global_id(0);
    ushort slid    = get_sub_group_local_id();

    uint K = ne00;
    uint M = ne01;

    uint LINE_STRIDE_A = M / 2;
    uint BLOCK_STRIDE_A = N_SIMDGROUP * M;

    __private uint4     regA;
    __private half2     regS;
    __private float8    regB;

    __private float2 totalSum = (float2)(0.0f);

    // loop along K in block granularity, skip 4 blocks every iter
    for (uint k = groupId; k < (K / QK4_0); k += N_SIMDGROUP) {
        regS = src0_d[gid + k * LINE_STRIDE_A]; // each fiber loads scale of two rows
        // first 4 fibers in each wave load 8 B values to its private scope
        if (slid < 4) {
            regB.s0123 = read_imagef(src1, (slid * 2 + k * 8));
            regB.s4567 = read_imagef(src1, (1 + slid * 2 + k * 8));
        }

        // load half weights for two blocks in consecutive rows
        regA.s0 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 0)).x;
        regA.s1 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 1)).x;
        regA.s2 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 2)).x;
        regA.s3 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 3)).x;
#ifdef VECTOR_SUB_GROUP_BROADCAST
        dequantizeBlockAccum_ns_sgbroadcast_8_hi(totalSum, as_ushort8(regA), regS, regB);
#else
        dequantizeBlockAccum_ns_sgbroadcast_1_hi(totalSum, as_ushort8(regA), regS, regB);
#endif // VECTOR_SUB_GROUP_BROADCAST

        regA.s0 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 4)).x;
        regA.s1 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 5)).x;
        regA.s2 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 6)).x;
        regA.s3 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 7)).x;
#ifdef VECTOR_SUB_GROUP_BROADCAST
        dequantizeBlockAccum_ns_sgbroadcast_8_lo(totalSum, as_ushort8(regA), regS, regB);
#else
        dequantizeBlockAccum_ns_sgbroadcast_1_lo(totalSum, as_ushort8(regA), regS, regB);
#endif // VECTOR_SUB_GROUP_BROADCAST
    }

    // reduction in local memory, assumes #wave=4
    __local float2 reduceLM[SIMDGROUP_WIDTH * 3];
    if (groupId == 1) reduceLM[SIMDGROUP_WIDTH * 0 + slid] = totalSum;
    if (groupId == 2) reduceLM[SIMDGROUP_WIDTH * 1 + slid] = totalSum;
    if (groupId == 3) reduceLM[SIMDGROUP_WIDTH * 2 + slid] = totalSum;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (groupId == 0) totalSum += reduceLM[SIMDGROUP_WIDTH * 0 + slid];
    if (groupId == 0) totalSum += reduceLM[SIMDGROUP_WIDTH * 1 + slid];
    if (groupId == 0) totalSum += reduceLM[SIMDGROUP_WIDTH * 2 + slid];

    // 2 outputs per fiber in wave 0
    if (groupId == 0) {
        dst = (global float*)((global char*)dst + offsetd);
        // Guard the two output rows. The x-grid is padded to CEIL_DIV(ne01/2,64)*64,
        // so when ne01 is not a multiple of 128 the tail row-pairs run past row ne01
        // and would overrun dst into the adjacent tensor. No-op / byte-identical when
        // ne01 % 128 == 0 (M/2 already a multiple of 64 -> no padding).
        if (gid * 2 + 0 < M) dst[gid * 2 + 0] = totalSum.s0;
        if (gid * 2 + 1 < M) dst[gid * 2 + 1] = totalSum.s1;
    }

}

// --- Fused gate+up decode GEMV + GLU (q4_0 FFN) ---------------------------------
// Folds the FFN's two q4_0 decode GEMVs (ffn_gate, ffn_up) and the GLU into one
// dispatch: two sequential K-loops accumulate gate and up over the SHARED activation
// (read once), then a float4-packed cross-subgroup reduce and the GLU epilogue emit
// silu/gelu(gate)*up directly. Removes the separate geglu kernel's global round-trip
// (gate/up intermediate writes + re-reads) and two dispatches. Coherent (the K-sum
// order matches the base GEMV; nsg<16 makes it not byte-identical). Mirrors the q4_K
// GLU kernel, minus the min/6-bit-scale machinery. Gated host-side (q4_0 gate/up).
#define GLU_GEGLU_COEF_A      0.044715f
#define GLU_SQRT_2_OVER_PI    0.79788456080286535587989211986876f
#define GLU_SQRT_2_INV        0.70710678118654752440084436210484f
#define GLU_QUICK_COEF       -1.702f
inline float q40_glu_apply(int glu_op, float g, float u) {
    float act;
    if (glu_op == 1) {        // GEGLU (tanh-approx gelu)
        act = 0.5f*g*(1.0f + tanh(GLU_SQRT_2_OVER_PI*g*(1.0f + GLU_GEGLU_COEF_A*g*g)));
    } else if (glu_op == 2) { // SWIGLU (silu)
        act = g / (1.0f + exp(-g));
    } else if (glu_op == 0) { // REGLU
        return g*u*(g > 0.0f);
    } else if (glu_op == 4) { // GEGLU_ERF
        act = 0.5f*g*(1.0f + erf(g*GLU_SQRT_2_INV));
    } else {                  // GEGLU_QUICK
        act = g*(1.0f/(1.0f + exp(GLU_QUICK_COEF*g)));
    }
    return act*u;
}
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
__kernel void kernel_gemv_noshuffle_q4_0_f32_glu(
        __read_only  image1d_buffer_t src0g_q,   // gate weights
        global half2  * src0g_d,                  // gate scales
        __read_only  image1d_buffer_t src0u_q,    // up weights
        global half2  * src0u_d,                  // up scales
        __read_only  image1d_buffer_t src1,       // shared activation
        global float * dst,
        ulong offsetd,
        int ne00,                                 // K
        int ne01,                                 // M
        int glu_op)
{
    uint groupId = get_local_id(1);
    uint gid     = get_global_id(0);
    ushort slid  = get_sub_group_local_id();
    uint nsg     = get_local_size(1);

    uint K = ne00;
    uint M = ne01;
    uint LINE_STRIDE_A  = M / 2;
    uint BLOCK_STRIDE_A = N_SIMDGROUP * M;

    __private uint4  regA;
    __private half2  regS;
    __private float8 regB;
    __private float2 gateSum = (float2)(0.0f);
    __private float2 upSum   = (float2)(0.0f);

#ifdef VECTOR_SUB_GROUP_BROADCAST
#define Q40_DEQ_HI dequantizeBlockAccum_ns_sgbroadcast_8_hi
#define Q40_DEQ_LO dequantizeBlockAccum_ns_sgbroadcast_8_lo
#else
#define Q40_DEQ_HI dequantizeBlockAccum_ns_sgbroadcast_1_hi
#define Q40_DEQ_LO dequantizeBlockAccum_ns_sgbroadcast_1_lo
#endif
    // The x-grid is padded to CEIL_DIV(M/2,64)*64, so when M % 128 != 0 the tail
    // lanes hold gid >= M/2. Their outputs are discarded below, but the SCALE fetch
    // is a raw global pointer (the weight fetches are images, which clamp), so those
    // lanes read past the end of the scales allocation -- up to 31 half2 beyond it on
    // the last K-block. Clamp the row used for the scale so every lane stays in
    // bounds; the lanes remain active, which subgroup broadcast requires, and their
    // results are still thrown away by the output guard.
    // gemma-4-26B-A4B is the shape that exposes this: ffn_gate/ffn_up are
    // [2816, 2112], and 2112 % 128 == 64. Models whose ffn ne01 is a multiple of 128
    // (gemma-4-12B 15360, E4B 10240) never form a tail and were unaffected.
    uint gid_s = min(gid, LINE_STRIDE_A - 1);
#define Q40_GLU_LOOP(SUM, Q, DD)                                                    \
    for (uint k = groupId; k < (K / QK4_0); k += nsg) {                            \
        regS = DD[gid_s + k * LINE_STRIDE_A];                                      \
        if (slid < 4) {                                                            \
            regB.s0123 = read_imagef(src1, (slid * 2 + k * 8));                    \
            regB.s4567 = read_imagef(src1, (1 + slid * 2 + k * 8));                \
        }                                                                          \
        regA.s0 = read_imageui(Q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 0)).x;\
        regA.s1 = read_imageui(Q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 1)).x;\
        regA.s2 = read_imageui(Q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 2)).x;\
        regA.s3 = read_imageui(Q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 3)).x;\
        Q40_DEQ_HI(SUM, as_ushort8(regA), regS, regB);                             \
        regA.s0 = read_imageui(Q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 4)).x;\
        regA.s1 = read_imageui(Q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 5)).x;\
        regA.s2 = read_imageui(Q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 6)).x;\
        regA.s3 = read_imageui(Q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 7)).x;\
        Q40_DEQ_LO(SUM, as_ushort8(regA), regS, regB);                             \
    }
    Q40_GLU_LOOP(gateSum, src0g_q, src0g_d)
    Q40_GLU_LOOP(upSum,   src0u_q, src0u_d)
#undef Q40_DEQ_HI
#undef Q40_DEQ_LO
#undef Q40_GLU_LOOP

    __local float4 reduceLM[SIMDGROUP_WIDTH * 7];
    if (groupId > 0) {
        reduceLM[SIMDGROUP_WIDTH * (groupId - 1) + slid] = (float4)(gateSum, upSum);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (groupId == 0) {
        for (uint i = 0; i < nsg - 1; ++i) {
            float4 p = reduceLM[SIMDGROUP_WIDTH * i + slid];
            gateSum += p.xy;
            upSum   += p.zw;
        }
        dst = (global float*)((global char*)dst + offsetd);
        if (gid * 2 + 0 < M) dst[gid * 2 + 0] = q40_glu_apply(glu_op, gateSum.s0, upSum.s0);
        if (gid * 2 + 1 < M) dst[gid * 2 + 1] = q40_glu_apply(glu_op, gateSum.s1, upSum.s1);
    }
}

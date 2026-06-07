#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64 __attribute__((qcom_reqd_sub_group_size("half")))
#endif

#define QK_K  256
#define NSUBGROUPS 4
#define SUBGROUP_SIZE 64

// scales are transposed: consecutive codes of a row are `stride` apart
inline void get_scale_min_k4(
    int j,
    global const uchar * q,
    uint stride,
    uchar * d,
    uchar * m,
    uchar mask_d6,
    uchar mask_d4,
    uchar mask_hi2
) {
    if (j < 4) {
        *d = q[j*stride]     & mask_d6;
        *m = q[(j+4)*stride] & mask_d6;
    } else {
        *d = (q[(j+4)*stride] & mask_d4) | ((q[(j-4)*stride] & mask_hi2) >> 2);
        *m = ((q[(j+4)*stride] >> 4) & mask_d4) | ((q[j*stride] & mask_hi2) >> 2);
    }
}

#define dequantizeBlockAccum_ns_sgbroadcast_1_hi(total_sums, bits4, scale, minv, y) \
    float shared_y; \
    shared_y = sub_group_broadcast(y.s0, 0); \
    total_sums.s0 += ((bits4.s0 & 0x000F) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((bits4.s1 & 0x000F) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s1, 0); \
    total_sums.s0 += (((bits4.s0 & 0x00F0) >> 4) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s1 & 0x00F0) >> 4) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s2, 0); \
    total_sums.s0 += (((bits4.s0 & 0x0F00) >> 8) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s1 & 0x0F00) >> 8) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s3, 0); \
    total_sums.s0 += (((bits4.s0 & 0xF000) >> 12) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s1 & 0xF000) >> 12) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s4, 0); \
    total_sums.s0 += ((bits4.s2 & 0x000F) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((bits4.s3 & 0x000F) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s5, 0); \
    total_sums.s0 += (((bits4.s2 & 0x00F0) >> 4) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s3 & 0x00F0) >> 4) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s6, 0); \
    total_sums.s0 += (((bits4.s2 & 0x0F00) >> 8) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s3 & 0x0F00) >> 8) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s7, 0); \
    total_sums.s0 += (((bits4.s2 & 0xF000) >> 12) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s3 & 0xF000) >> 12) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s0, 1); \
    total_sums.s0 += ((bits4.s4 & 0x000F) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((bits4.s5 & 0x000F) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s1, 1); \
    total_sums.s0 += (((bits4.s4 & 0x00F0) >> 4) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s5 & 0x00F0) >> 4) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s2, 1); \
    total_sums.s0 += (((bits4.s4 & 0x0F00) >> 8) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s5 & 0x0F00) >> 8) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s3, 1); \
    total_sums.s0 += (((bits4.s4 & 0xF000) >> 12) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s5 & 0xF000) >> 12) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s4, 1); \
    total_sums.s0 += ((bits4.s6 & 0x000F) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((bits4.s7 & 0x000F) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s5, 1); \
    total_sums.s0 += (((bits4.s6 & 0x00F0) >> 4) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s7 & 0x00F0) >> 4) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s6, 1); \
    total_sums.s0 += (((bits4.s6 & 0x0F00) >> 8) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s7 & 0x0F00) >> 8) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s7, 1); \
    total_sums.s0 += (((bits4.s6 & 0xF000) >> 12) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s7 & 0xF000) >> 12) * scale.s1 - minv.s1) * shared_y; \


#define dequantizeBlockAccum_ns_sgbroadcast_1_lo(total_sums, bits4, scale, minv, y) \
    shared_y = sub_group_broadcast(y.s0, 2); \
    total_sums.s0 += ((bits4.s0 & 0x000F) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((bits4.s1 & 0x000F) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s1, 2); \
    total_sums.s0 += (((bits4.s0 & 0x00F0) >> 4) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s1 & 0x00F0) >> 4) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s2, 2); \
    total_sums.s0 += (((bits4.s0 & 0x0F00) >> 8) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s1 & 0x0F00) >> 8) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s3, 2); \
    total_sums.s0 += (((bits4.s0 & 0xF000) >> 12) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s1 & 0xF000) >> 12) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s4, 2); \
    total_sums.s0 += ((bits4.s2 & 0x000F) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((bits4.s3 & 0x000F) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s5, 2); \
    total_sums.s0 += (((bits4.s2 & 0x00F0) >> 4) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s3 & 0x00F0) >> 4) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s6, 2); \
    total_sums.s0 += (((bits4.s2 & 0x0F00) >> 8) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s3 & 0x0F00) >> 8) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s7, 2); \
    total_sums.s0 += (((bits4.s2 & 0xF000) >> 12) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s3 & 0xF000) >> 12) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s0, 3); \
    total_sums.s0 += ((bits4.s4 & 0x000F) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((bits4.s5 & 0x000F) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s1, 3); \
    total_sums.s0 += (((bits4.s4 & 0x00F0) >> 4) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s5 & 0x00F0) >> 4) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s2, 3); \
    total_sums.s0 += (((bits4.s4 & 0x0F00) >> 8) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s5 & 0x0F00) >> 8) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s3, 3); \
    total_sums.s0 += (((bits4.s4 & 0xF000) >> 12) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s5 & 0xF000) >> 12) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s4, 3); \
    total_sums.s0 += ((bits4.s6 & 0x000F) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((bits4.s7 & 0x000F) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s5, 3); \
    total_sums.s0 += (((bits4.s6 & 0x00F0) >> 4) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s7 & 0x00F0) >> 4) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s6, 3); \
    total_sums.s0 += (((bits4.s6 & 0x0F00) >> 8) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s7 & 0x0F00) >> 8) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s7, 3); \
    total_sums.s0 += (((bits4.s6 & 0xF000) >> 12) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s7 & 0xF000) >> 12) * scale.s1 - minv.s1) * shared_y; \


#define dequantizeBlockAccum_ns_sgbroadcast_8_hi(total_sums, bits4, scale, minv, y) \
    float8 shared_y; \
    shared_y = sub_group_broadcast(y, 0); \
    total_sums.s0 += ((bits4.s0 & 0x000F)         * scale.s0 - minv.s0) * shared_y.s0; \
    total_sums.s0 += (((bits4.s0 & 0x00F0) >> 4)  * scale.s0 - minv.s0) * shared_y.s1; \
    total_sums.s0 += (((bits4.s0 & 0x0F00) >> 8)  * scale.s0 - minv.s0) * shared_y.s2; \
    total_sums.s0 += (((bits4.s0 & 0xF000) >> 12) * scale.s0 - minv.s0) * shared_y.s3; \
    total_sums.s0 += ((bits4.s2 & 0x000F)         * scale.s0 - minv.s0) * shared_y.s4; \
    total_sums.s0 += (((bits4.s2 & 0x00F0) >> 4)  * scale.s0 - minv.s0) * shared_y.s5; \
    total_sums.s0 += (((bits4.s2 & 0x0F00) >> 8)  * scale.s0 - minv.s0) * shared_y.s6; \
    total_sums.s0 += (((bits4.s2 & 0xF000) >> 12) * scale.s0 - minv.s0) * shared_y.s7; \
    total_sums.s1 += ((bits4.s1 & 0x000F)         * scale.s1 - minv.s1) * shared_y.s0; \
    total_sums.s1 += (((bits4.s1 & 0x00F0) >> 4)  * scale.s1 - minv.s1) * shared_y.s1; \
    total_sums.s1 += (((bits4.s1 & 0x0F00) >> 8)  * scale.s1 - minv.s1) * shared_y.s2; \
    total_sums.s1 += (((bits4.s1 & 0xF000) >> 12) * scale.s1 - minv.s1) * shared_y.s3; \
    total_sums.s1 += ((bits4.s3 & 0x000F)         * scale.s1 - minv.s1) * shared_y.s4; \
    total_sums.s1 += (((bits4.s3 & 0x00F0) >> 4)  * scale.s1 - minv.s1) * shared_y.s5; \
    total_sums.s1 += (((bits4.s3 & 0x0F00) >> 8)  * scale.s1 - minv.s1) * shared_y.s6; \
    total_sums.s1 += (((bits4.s3 & 0xF000) >> 12) * scale.s1 - minv.s1) * shared_y.s7; \
    shared_y = sub_group_broadcast(y, 1); \
    total_sums.s0 += ((bits4.s4 & 0x000F)         * scale.s0 - minv.s0) * shared_y.s0; \
    total_sums.s0 += (((bits4.s4 & 0x00F0) >> 4)  * scale.s0 - minv.s0) * shared_y.s1; \
    total_sums.s0 += (((bits4.s4 & 0x0F00) >> 8)  * scale.s0 - minv.s0) * shared_y.s2; \
    total_sums.s0 += (((bits4.s4 & 0xF000) >> 12) * scale.s0 - minv.s0) * shared_y.s3; \
    total_sums.s0 += ((bits4.s6 & 0x000F)         * scale.s0 - minv.s0) * shared_y.s4; \
    total_sums.s0 += (((bits4.s6 & 0x00F0) >> 4)  * scale.s0 - minv.s0) * shared_y.s5; \
    total_sums.s0 += (((bits4.s6 & 0x0F00) >> 8)  * scale.s0 - minv.s0) * shared_y.s6; \
    total_sums.s0 += (((bits4.s6 & 0xF000) >> 12) * scale.s0 - minv.s0) * shared_y.s7; \
    total_sums.s1 += ((bits4.s5 & 0x000F)         * scale.s1 - minv.s1) * shared_y.s0; \
    total_sums.s1 += (((bits4.s5 & 0x00F0) >> 4)  * scale.s1 - minv.s1) * shared_y.s1; \
    total_sums.s1 += (((bits4.s5 & 0x0F00) >> 8)  * scale.s1 - minv.s1) * shared_y.s2; \
    total_sums.s1 += (((bits4.s5 & 0xF000) >> 12) * scale.s1 - minv.s1) * shared_y.s3; \
    total_sums.s1 += ((bits4.s7 & 0x000F)         * scale.s1 - minv.s1) * shared_y.s4; \
    total_sums.s1 += (((bits4.s7 & 0x00F0) >> 4)  * scale.s1 - minv.s1) * shared_y.s5; \
    total_sums.s1 += (((bits4.s7 & 0x0F00) >> 8)  * scale.s1 - minv.s1) * shared_y.s6; \
    total_sums.s1 += (((bits4.s7 & 0xF000) >> 12) * scale.s1 - minv.s1) * shared_y.s7; \


#define dequantizeBlockAccum_ns_sgbroadcast_8_lo(total_sums, bits4, scale, minv, y) \
    shared_y = sub_group_broadcast(y, 2); \
    total_sums.s0 += ((bits4.s0 & 0x000F)         * scale.s0 - minv.s0) * shared_y.s0; \
    total_sums.s0 += (((bits4.s0 & 0x00F0) >> 4)  * scale.s0 - minv.s0) * shared_y.s1; \
    total_sums.s0 += (((bits4.s0 & 0x0F00) >> 8)  * scale.s0 - minv.s0) * shared_y.s2; \
    total_sums.s0 += (((bits4.s0 & 0xF000) >> 12) * scale.s0 - minv.s0) * shared_y.s3; \
    total_sums.s0 += ((bits4.s2 & 0x000F)         * scale.s0 - minv.s0) * shared_y.s4; \
    total_sums.s0 += (((bits4.s2 & 0x00F0) >> 4)  * scale.s0 - minv.s0) * shared_y.s5; \
    total_sums.s0 += (((bits4.s2 & 0x0F00) >> 8)  * scale.s0 - minv.s0) * shared_y.s6; \
    total_sums.s0 += (((bits4.s2 & 0xF000) >> 12) * scale.s0 - minv.s0) * shared_y.s7; \
    total_sums.s1 += ((bits4.s1 & 0x000F)         * scale.s1 - minv.s1) * shared_y.s0; \
    total_sums.s1 += (((bits4.s1 & 0x00F0) >> 4)  * scale.s1 - minv.s1) * shared_y.s1; \
    total_sums.s1 += (((bits4.s1 & 0x0F00) >> 8)  * scale.s1 - minv.s1) * shared_y.s2; \
    total_sums.s1 += (((bits4.s1 & 0xF000) >> 12) * scale.s1 - minv.s1) * shared_y.s3; \
    total_sums.s1 += ((bits4.s3 & 0x000F)         * scale.s1 - minv.s1) * shared_y.s4; \
    total_sums.s1 += (((bits4.s3 & 0x00F0) >> 4)  * scale.s1 - minv.s1) * shared_y.s5; \
    total_sums.s1 += (((bits4.s3 & 0x0F00) >> 8)  * scale.s1 - minv.s1) * shared_y.s6; \
    total_sums.s1 += (((bits4.s3 & 0xF000) >> 12) * scale.s1 - minv.s1) * shared_y.s7; \
    shared_y = sub_group_broadcast(y, 3); \
    total_sums.s0 += ((bits4.s4 & 0x000F)         * scale.s0 - minv.s0) * shared_y.s0; \
    total_sums.s0 += (((bits4.s4 & 0x00F0) >> 4)  * scale.s0 - minv.s0) * shared_y.s1; \
    total_sums.s0 += (((bits4.s4 & 0x0F00) >> 8)  * scale.s0 - minv.s0) * shared_y.s2; \
    total_sums.s0 += (((bits4.s4 & 0xF000) >> 12) * scale.s0 - minv.s0) * shared_y.s3; \
    total_sums.s0 += ((bits4.s6 & 0x000F)         * scale.s0 - minv.s0) * shared_y.s4; \
    total_sums.s0 += (((bits4.s6 & 0x00F0) >> 4)  * scale.s0 - minv.s0) * shared_y.s5; \
    total_sums.s0 += (((bits4.s6 & 0x0F00) >> 8)  * scale.s0 - minv.s0) * shared_y.s6; \
    total_sums.s0 += (((bits4.s6 & 0xF000) >> 12) * scale.s0 - minv.s0) * shared_y.s7; \
    total_sums.s1 += ((bits4.s5 & 0x000F)         * scale.s1 - minv.s1) * shared_y.s0; \
    total_sums.s1 += (((bits4.s5 & 0x00F0) >> 4)  * scale.s1 - minv.s1) * shared_y.s1; \
    total_sums.s1 += (((bits4.s5 & 0x0F00) >> 8)  * scale.s1 - minv.s1) * shared_y.s2; \
    total_sums.s1 += (((bits4.s5 & 0xF000) >> 12) * scale.s1 - minv.s1) * shared_y.s3; \
    total_sums.s1 += ((bits4.s7 & 0x000F)         * scale.s1 - minv.s1) * shared_y.s4; \
    total_sums.s1 += (((bits4.s7 & 0x00F0) >> 4)  * scale.s1 - minv.s1) * shared_y.s5; \
    total_sums.s1 += (((bits4.s7 & 0x0F00) >> 8)  * scale.s1 - minv.s1) * shared_y.s6; \
    total_sums.s1 += (((bits4.s7 & 0xF000) >> 12) * scale.s1 - minv.s1) * shared_y.s7; \

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_gemv_noshuffle_q4_k_f32(
        read_only  image1d_buffer_t src0_q,
        global half2  * src0_d,
        global half2  * src0_m,
        global uchar  * src0_s,
        read_only  image1d_buffer_t src1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        uchar mask_d6,
        uchar mask_d4,
        uchar mask_hi2)
{
    uint groupId = get_local_id(1);
    uint gid     = get_global_id(0);
    ushort slid  = get_sub_group_local_id();

    uint K = ne00;
    uint M = ne01;

    uint LINE_STRIDE_A  = M / 2;
    uint BLOCK_STRIDE_A = NSUBGROUPS * M;

    private uint4     regA;
    private half2     regS;
    private half2     regM;
    private float8    regB;

    private float2 totalSum = (float2)(0.0f);

    for (uint k = groupId; k < (K / 32); k += NSUBGROUPS) {
        uint sb = k / 8;
        uint j  = k % 8;

        half2 d   = src0_d[gid + sb * LINE_STRIDE_A];
        half2 dm  = src0_m[gid + sb * LINE_STRIDE_A];

        global const uchar * sc0 = src0_s + sb * 12 * M + 2 * gid;
        global const uchar * sc1 = sc0 + 1;

        uchar sv0, mn0, sv1, mn1;
        get_scale_min_k4(j, sc0, M, &sv0, &mn0, mask_d6, mask_d4, mask_hi2);
        get_scale_min_k4(j, sc1, M, &sv1, &mn1, mask_d6, mask_d4, mask_hi2);

        regS = convert_half2(convert_float2(d)  * convert_float2((uchar2)(sv0, sv1)));
        regM = convert_half2(convert_float2(dm) * convert_float2((uchar2)(mn0, mn1)));

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
        dequantizeBlockAccum_ns_sgbroadcast_8_hi(totalSum, as_ushort8(regA), regS, regM, regB);
#else
        dequantizeBlockAccum_ns_sgbroadcast_1_hi(totalSum, as_ushort8(regA), regS, regM, regB);
#endif // VECTOR_SUB_GROUP_BROADCAST

        regA.s0 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 4)).x;
        regA.s1 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 5)).x;
        regA.s2 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 6)).x;
        regA.s3 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 7)).x;
#ifdef VECTOR_SUB_GROUP_BROADCAST
        dequantizeBlockAccum_ns_sgbroadcast_8_lo(totalSum, as_ushort8(regA), regS, regM, regB);
#else
        dequantizeBlockAccum_ns_sgbroadcast_1_lo(totalSum, as_ushort8(regA), regS, regM, regB);
#endif // VECTOR_SUB_GROUP_BROADCAST
    }

    // reduction in local memory, assumes #wave=4
    local float2 reduceLM[SUBGROUP_SIZE * 3];
    if (groupId == 1) {
        reduceLM[SUBGROUP_SIZE * 0 + slid] = totalSum;
    }
    if (groupId == 2) {
        reduceLM[SUBGROUP_SIZE * 1 + slid] = totalSum;
    }
    if (groupId == 3) {
        reduceLM[SUBGROUP_SIZE * 2 + slid] = totalSum;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (groupId == 0) {
        totalSum += reduceLM[SUBGROUP_SIZE * 0 + slid];
    }
    if (groupId == 0) {
        totalSum += reduceLM[SUBGROUP_SIZE * 1 + slid];
    }
    if (groupId == 0) {
        totalSum += reduceLM[SUBGROUP_SIZE * 2 + slid];
    }

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

// --- Fused gate+up GEMV + GLU epilogue (FFN) ------------------------------------
// Folds the FFN's two decode GEMVs (ffn_gate, ffn_up) and the following GLU into a
// SINGLE dispatch: {MUL_MAT(Wg,x), MUL_MAT(Wu,x), GLU}. Both matmuls share the same
// activation x (ffn_norm), so the activation image read is issued ONCE per K-block
// and reused for the gate and up dot products (the per-op path re-reads it twice and
// also materializes the two full ffn-wide intermediates to global, which the GLU
// then re-reads). The gate/up partial sums are accumulated in the SAME per-fiber
// order and reduced in the SAME cross-subgroup order as the standalone GEMV, and the
// GLU formula is the exact scalar expression from kernels/glu.cl, so the output is
// BYTE-IDENTICAL to the per-op matmul+matmul+glu path -> safe to default on.
//   glu_op: REGLU=0, GEGLU=1, SWIGLU=2, GEGLU_ERF=4, GEGLU_QUICK=5 (ggml_glu_op).
// Weights: src0g_* = gate (= GLU src[0]); src0u_* = up (= GLU src[1]).
#define GLU_GEGLU_COEF_A      0.044715f
#define GLU_SQRT_2_OVER_PI    0.79788456080286535587989211986876f
#define GLU_SQRT_2_INV        0.70710678118654752440084436210484f
#define GLU_QUICK_COEF       -1.702f

inline float glu_apply(int glu_op, float g, float u) {
    float act;
    if (glu_op == 1) {        // GEGLU (tanh-approx gelu)
        act = 0.5f*g*(1.0f + tanh(GLU_SQRT_2_OVER_PI*g*(1.0f + GLU_GEGLU_COEF_A*g*g)));
    } else if (glu_op == 2) { // SWIGLU (silu)
        act = g / (1.0f + exp(-g));
    } else if (glu_op == 0) { // REGLU
        return g*u*(g > 0.0f);
    } else if (glu_op == 4) { // GEGLU_ERF
        act = 0.5f*g*(1.0f + erf(g*GLU_SQRT_2_INV));
    } else {                  // GEGLU_QUICK (glu_op == 5)
        act = g*(1.0f/(1.0f + exp(GLU_QUICK_COEF*g)));
    }
    return act*u;
}

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_gemv_noshuffle_q4_k_f32_glu(
        read_only  image1d_buffer_t src0g_q,
        global half2  * src0g_d,
        global half2  * src0g_m,
        global uchar  * src0g_s,
        read_only  image1d_buffer_t src0u_q,
        global half2  * src0u_d,
        global half2  * src0u_m,
        global uchar  * src0u_s,
        read_only  image1d_buffer_t src1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int glu_op,
        uchar mask_d6,
        uchar mask_d4,
        uchar mask_hi2)
{
    uint groupId = get_local_id(1);
    uint gid     = get_global_id(0);
    ushort slid  = get_sub_group_local_id();
    uint nsg     = get_local_size(1);

    uint K = ne00;
    uint M = ne01;

    uint LINE_STRIDE_A  = M / 2;
    uint BLOCK_STRIDE_A = 4 * M;
    uint scales_per_row = (K / QK_K) * 12;

    private uint4  regA;
    private half2  regS, regM;
    private float8 regB;

    private float2 gateSum = (float2)(0.0f);
    private float2 upSum   = (float2)(0.0f);

    // Two SEQUENTIAL K-loops (gate fully, then up). Keeping only one weight's
    // working set live at a time holds the kernel's register footprint at ~the
    // base single-weight GEMV's, so its max WG stays 1024 (16 subgroups) and the
    // per-subgroup K-split matches the standalone wide GEMV exactly -> the gate
    // and up partial sums are BYTE-IDENTICAL to the per-op path. The macro body
    // is the base kernel's inner loop verbatim, parameterized by weight source.
#define Q4K_GLU_LOOP(SUM, Q, DD, MM, SS)                                                       \
    for (uint k = groupId; k < (K / 32); k += nsg) {                                           \
        uint sb = k / 8;                                                                       \
        uint j  = k % 8;                                                                       \
        half2 d   = DD[gid + sb * LINE_STRIDE_A];                                              \
        half2 dm  = MM[gid + sb * LINE_STRIDE_A];                                              \
        global const uchar * sc0 = SS + 2 * gid * scales_per_row + sb * 12;                    \
        global const uchar * sc1 = SS + (2 * gid + 1) * scales_per_row + sb * 12;              \
        uchar sv0, mn0, sv1, mn1;                                                              \
        get_scale_min_k4(j, sc0, &sv0, &mn0, mask_d6, mask_d4, mask_hi2);                      \
        get_scale_min_k4(j, sc1, &sv1, &mn1, mask_d6, mask_d4, mask_hi2);                      \
        regS = convert_half2(convert_float2(d)  * convert_float2((uchar2)(sv0, sv1)));         \
        regM = convert_half2(convert_float2(dm) * convert_float2((uchar2)(mn0, mn1)));         \
        if (slid < 4) {                                                                        \
            regB.s0123 = read_imagef(src1, (slid * 2 + k * 8));                                \
            regB.s4567 = read_imagef(src1, (1 + slid * 2 + k * 8));                            \
        }                                                                                      \
        regA.s0 = read_imageui(Q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 0)).x;           \
        regA.s1 = read_imageui(Q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 1)).x;           \
        regA.s2 = read_imageui(Q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 2)).x;           \
        regA.s3 = read_imageui(Q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 3)).x;           \
        DEQ_HI(SUM, as_ushort8(regA), regS, regM, regB);                                       \
        regA.s0 = read_imageui(Q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 4)).x;           \
        regA.s1 = read_imageui(Q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 5)).x;           \
        regA.s2 = read_imageui(Q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 6)).x;           \
        regA.s3 = read_imageui(Q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 7)).x;           \
        DEQ_LO(SUM, as_ushort8(regA), regS, regM, regB);                                       \
    }

#ifdef VECTOR_SUB_GROUP_BROADCAST
#define DEQ_HI dequantizeBlockAccum_ns_sgbroadcast_8_hi
#define DEQ_LO dequantizeBlockAccum_ns_sgbroadcast_8_lo
#else
#define DEQ_HI dequantizeBlockAccum_ns_sgbroadcast_1_hi
#define DEQ_LO dequantizeBlockAccum_ns_sgbroadcast_1_lo
#endif

    Q4K_GLU_LOOP(gateSum, src0g_q, src0g_d, src0g_m, src0g_s)
    Q4K_GLU_LOOP(upSum,   src0u_q, src0u_d, src0u_m, src0u_s)

#undef DEQ_HI
#undef DEQ_LO
#undef Q4K_GLU_LOOP

    // Cross-subgroup reduction in local memory. Packs gate (xy) + up (zw) into a
    // float4 so both reduce in one pass; summation order matches the base GEMV's
    // per-channel loop -> byte-identical partial sums.
    local float4 reduceLM[SUBGROUP_SIZE * 15];
    if (groupId > 0) {
        reduceLM[SUBGROUP_SIZE * (groupId - 1) + slid] = (float4)(gateSum, upSum);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (groupId == 0) {
        for (uint i = 0; i < nsg - 1; ++i) {
            float4 p = reduceLM[SUBGROUP_SIZE * i + slid];
            gateSum += p.xy;
            upSum   += p.zw;
        }
        dst = (global float*)((global char*)dst + offsetd);
        dst[gid * 2 + 0] = glu_apply(glu_op, gateSum.s0, upSum.s0);
        dst[gid * 2 + 1] = glu_apply(glu_op, gateSum.s1, upSum.s1);
    }
}

// --- Dequant-once macros for the mc3 verify GEMV (Q4K_MC3_DEQUANT_ONCE) ---
// The inline dequantizeBlockAccum_* macros recompute the dequantized weight
// ((code & mask)>>shift)*scale - minv ONCE PER COLUMN (3x), and the flat
// 32-FMA unroll spills ~430 B of temporaries. These macros split the work:
// DEQUANT_Q4K_BLOCK computes the 16 weights/row of one 32-block ONCE into a
// half2[] (row0 in .s0, row1 in .s1) — stored as half, the exact type the
// inline expression yields (int*half-half), so no extra rounding. MAC_Q4K_BLOCK
// then accumulates them against a column's broadcast activation in the SAME
// per-accumulator order as the inline macro. Each weight value and each
// accumulator's add-chain is bit-for-bit identical => byte-identical output,
// while the dequant ALU drops 3x->1x and the live set shrinks. Requires the
// Qualcomm vector sub_group_broadcast (float8); enabled opt-in on Adreno.
#define DEQ_Q4K_HALF2(b0, b1, msk, sh, scale, minv) \
    (half2)( ((b0 & msk) >> sh) * scale.s0 - minv.s0, \
             ((b1 & msk) >> sh) * scale.s1 - minv.s1 )

#define DEQUANT_Q4K_BLOCK(wq, bits, scale, minv) \
    wq[0]  = DEQ_Q4K_HALF2(bits.s0, bits.s1, 0x000F, 0,  scale, minv); \
    wq[1]  = DEQ_Q4K_HALF2(bits.s0, bits.s1, 0x00F0, 4,  scale, minv); \
    wq[2]  = DEQ_Q4K_HALF2(bits.s0, bits.s1, 0x0F00, 8,  scale, minv); \
    wq[3]  = DEQ_Q4K_HALF2(bits.s0, bits.s1, 0xF000, 12, scale, minv); \
    wq[4]  = DEQ_Q4K_HALF2(bits.s2, bits.s3, 0x000F, 0,  scale, minv); \
    wq[5]  = DEQ_Q4K_HALF2(bits.s2, bits.s3, 0x00F0, 4,  scale, minv); \
    wq[6]  = DEQ_Q4K_HALF2(bits.s2, bits.s3, 0x0F00, 8,  scale, minv); \
    wq[7]  = DEQ_Q4K_HALF2(bits.s2, bits.s3, 0xF000, 12, scale, minv); \
    wq[8]  = DEQ_Q4K_HALF2(bits.s4, bits.s5, 0x000F, 0,  scale, minv); \
    wq[9]  = DEQ_Q4K_HALF2(bits.s4, bits.s5, 0x00F0, 4,  scale, minv); \
    wq[10] = DEQ_Q4K_HALF2(bits.s4, bits.s5, 0x0F00, 8,  scale, minv); \
    wq[11] = DEQ_Q4K_HALF2(bits.s4, bits.s5, 0xF000, 12, scale, minv); \
    wq[12] = DEQ_Q4K_HALF2(bits.s6, bits.s7, 0x000F, 0,  scale, minv); \
    wq[13] = DEQ_Q4K_HALF2(bits.s6, bits.s7, 0x00F0, 4,  scale, minv); \
    wq[14] = DEQ_Q4K_HALF2(bits.s6, bits.s7, 0x0F00, 8,  scale, minv); \
    wq[15] = DEQ_Q4K_HALF2(bits.s6, bits.s7, 0xF000, 12, scale, minv);

// ln0/ln1 = the two source lanes whose activation float8 this block consumes
// (0,1 for the hi block, 2,3 for the lo block — matching the inline _hi/_lo).
#define MAC_Q4K_BLOCK(ts, wq, y, ln0, ln1) { \
    float8 sy = sub_group_broadcast(y, ln0); \
    ts.s0 += wq[0].s0*sy.s0; ts.s0 += wq[1].s0*sy.s1; ts.s0 += wq[2].s0*sy.s2; ts.s0 += wq[3].s0*sy.s3; \
    ts.s0 += wq[4].s0*sy.s4; ts.s0 += wq[5].s0*sy.s5; ts.s0 += wq[6].s0*sy.s6; ts.s0 += wq[7].s0*sy.s7; \
    ts.s1 += wq[0].s1*sy.s0; ts.s1 += wq[1].s1*sy.s1; ts.s1 += wq[2].s1*sy.s2; ts.s1 += wq[3].s1*sy.s3; \
    ts.s1 += wq[4].s1*sy.s4; ts.s1 += wq[5].s1*sy.s5; ts.s1 += wq[6].s1*sy.s6; ts.s1 += wq[7].s1*sy.s7; \
    sy = sub_group_broadcast(y, ln1); \
    ts.s0 += wq[8].s0*sy.s0;  ts.s0 += wq[9].s0*sy.s1;  ts.s0 += wq[10].s0*sy.s2; ts.s0 += wq[11].s0*sy.s3; \
    ts.s0 += wq[12].s0*sy.s4; ts.s0 += wq[13].s0*sy.s5; ts.s0 += wq[14].s0*sy.s6; ts.s0 += wq[15].s0*sy.s7; \
    ts.s1 += wq[8].s1*sy.s0;  ts.s1 += wq[9].s1*sy.s1;  ts.s1 += wq[10].s1*sy.s2; ts.s1 += wq[11].s1*sy.s3; \
    ts.s1 += wq[12].s1*sy.s4; ts.s1 += wq[13].s1*sy.s5; ts.s1 += wq[14].s1*sy.s6; ts.s1 += wq[15].s1*sy.s7; \
}

// Multi-column (N=3) variant of the q4_K decode GEMV, for the speculative /
// MTP verify batch (ne1=3 = 2 drafts + 1 bonus). Stays on the efficient GEMV
// path (subgroup-broadcast activation, NSUBGROUPS K-split) instead of the
// transposed-GEMM dead-zone path. Each K-block's weights (regA_hi/regA_lo) are
// loaded ONCE and reused across all 3 activation columns — same weight traffic
// as one decode, ~3x the (cheap) dequant ALU. Per-column accumulation is
// independent and identical to 3 standalone GEMVs => byte-identical, so it does
// NOT perturb the lm_head logits / spec accept rate.
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_gemv_noshuffle_q4_k_f32_mc3(
        read_only  image1d_buffer_t src0_q,
        global half2  * src0_d,
        global half2  * src0_m,
        global uchar  * src0_s,
        read_only  image1d_buffer_t src1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        uchar mask_d6,
        uchar mask_d4,
        uchar mask_hi2)
{
    uint groupId = get_local_id(1);
    uint gid     = get_global_id(0);
    ushort slid  = get_sub_group_local_id();

    uint K = ne00;
    uint M = ne01;

    uint LINE_STRIDE_A  = M / 2;
    uint BLOCK_STRIDE_A = NSUBGROUPS * M;
    uint scales_per_row = (K / QK_K) * 12;
    uint COL_STRIDE     = K / 4;   // float4 pixels per activation column

    private uint4  regA_hi, regA_lo;
    private half2  regS, regM;
    private float8 regB;

    private float2 ts0 = (float2)(0.0f);
    private float2 ts1 = (float2)(0.0f);
    private float2 ts2 = (float2)(0.0f);

#ifdef Q4K_MC3_DEQUANT_LDS
    // One 16-half2 block buffer per WI (reused hi->lo): forces the dequantized
    // weights into LDS instead of private arrays (which spill to slow global on
    // Adreno). 64*NSUBGROUPS WIs * 16 half2 = 16 KB; each WI owns its own slot
    // range (flat*16) -> no cross-lane sharing, no barrier needed.
    local half2 wstage[SUBGROUP_SIZE * NSUBGROUPS * 16];
    local half2 * ws = wstage + (groupId * SUBGROUP_SIZE + slid) * 16;
#endif

    for (uint k = groupId; k < (K / 32); k += NSUBGROUPS) {
        uint sb = k / 8;
        uint j  = k % 8;

        half2 d   = src0_d[gid + sb * LINE_STRIDE_A];
        half2 dm  = src0_m[gid + sb * LINE_STRIDE_A];

        global const uchar * sc0 = src0_s + 2 * gid * scales_per_row + sb * 12;
        global const uchar * sc1 = src0_s + (2 * gid + 1) * scales_per_row + sb * 12;

        uchar sv0, mn0, sv1, mn1;
        get_scale_min_k4(j, sc0, &sv0, &mn0, mask_d6, mask_d4, mask_hi2);
        get_scale_min_k4(j, sc1, &sv1, &mn1, mask_d6, mask_d4, mask_hi2);

        regS = convert_half2(convert_float2(d)  * convert_float2((uchar2)(sv0, sv1)));
        regM = convert_half2(convert_float2(dm) * convert_float2((uchar2)(mn0, mn1)));

        // weights loaded ONCE, reused across the 3 columns
        regA_hi.s0 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 0)).x;
        regA_hi.s1 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 1)).x;
        regA_hi.s2 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 2)).x;
        regA_hi.s3 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 3)).x;
        regA_lo.s0 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 4)).x;
        regA_lo.s1 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 5)).x;
        regA_lo.s2 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 6)).x;
        regA_lo.s3 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 7)).x;

#ifdef Q4K_MC3_DEQUANT_ONCE
        // Dequant the 32 weights/row (16 hi + 16 lo) ONCE into half2[] (byte-
        // identical to the inline intermediate), then MAC against each column's
        // activation. Drops the dequant ALU 3x->1x and the macro-temp spill.
        half2 wq_hi[16], wq_lo[16];
        DEQUANT_Q4K_BLOCK(wq_hi, as_ushort8(regA_hi), regS, regM);
        DEQUANT_Q4K_BLOCK(wq_lo, as_ushort8(regA_lo), regS, regM);
        { if (slid < 4) { regB.s0123 = read_imagef(src1, 0*COL_STRIDE     + slid*2 + k*8);
                          regB.s4567 = read_imagef(src1, 0*COL_STRIDE + 1 + slid*2 + k*8); }
          MAC_Q4K_BLOCK(ts0, wq_hi, regB, 0, 1); MAC_Q4K_BLOCK(ts0, wq_lo, regB, 2, 3); }
        { if (slid < 4) { regB.s0123 = read_imagef(src1, 1*COL_STRIDE     + slid*2 + k*8);
                          regB.s4567 = read_imagef(src1, 1*COL_STRIDE + 1 + slid*2 + k*8); }
          MAC_Q4K_BLOCK(ts1, wq_hi, regB, 0, 1); MAC_Q4K_BLOCK(ts1, wq_lo, regB, 2, 3); }
        { if (slid < 4) { regB.s0123 = read_imagef(src1, 2*COL_STRIDE     + slid*2 + k*8);
                          regB.s4567 = read_imagef(src1, 2*COL_STRIDE + 1 + slid*2 + k*8); }
          MAC_Q4K_BLOCK(ts2, wq_hi, regB, 0, 1); MAC_Q4K_BLOCK(ts2, wq_lo, regB, 2, 3); }
#elif defined(Q4K_MC3_DEQUANT_LDS)
        // LDS-staged dequant: dequant a 32-block ONCE into the per-WI LDS slot
        // (hi pass then lo pass, overwriting), MAC each column from LDS. ts*
        // receive hi-then-lo in the same order as DEQUANT_ONCE -> byte-identical.
        // Activations reloaded per pass (cheap, imaged); only one regB + 0 weight
        // regs live -> the weight working set lives in LDS, not spilled private.
        DEQUANT_Q4K_BLOCK(ws, as_ushort8(regA_hi), regS, regM);
        { if (slid < 4) { regB.s0123 = read_imagef(src1, 0*COL_STRIDE     + slid*2 + k*8);
                          regB.s4567 = read_imagef(src1, 0*COL_STRIDE + 1 + slid*2 + k*8); }
          MAC_Q4K_BLOCK(ts0, ws, regB, 0, 1); }
        { if (slid < 4) { regB.s0123 = read_imagef(src1, 1*COL_STRIDE     + slid*2 + k*8);
                          regB.s4567 = read_imagef(src1, 1*COL_STRIDE + 1 + slid*2 + k*8); }
          MAC_Q4K_BLOCK(ts1, ws, regB, 0, 1); }
        { if (slid < 4) { regB.s0123 = read_imagef(src1, 2*COL_STRIDE     + slid*2 + k*8);
                          regB.s4567 = read_imagef(src1, 2*COL_STRIDE + 1 + slid*2 + k*8); }
          MAC_Q4K_BLOCK(ts2, ws, regB, 0, 1); }
        DEQUANT_Q4K_BLOCK(ws, as_ushort8(regA_lo), regS, regM);
        { if (slid < 4) { regB.s0123 = read_imagef(src1, 0*COL_STRIDE     + slid*2 + k*8);
                          regB.s4567 = read_imagef(src1, 0*COL_STRIDE + 1 + slid*2 + k*8); }
          MAC_Q4K_BLOCK(ts0, ws, regB, 2, 3); }
        { if (slid < 4) { regB.s0123 = read_imagef(src1, 1*COL_STRIDE     + slid*2 + k*8);
                          regB.s4567 = read_imagef(src1, 1*COL_STRIDE + 1 + slid*2 + k*8); }
          MAC_Q4K_BLOCK(ts1, ws, regB, 2, 3); }
        { if (slid < 4) { regB.s0123 = read_imagef(src1, 2*COL_STRIDE     + slid*2 + k*8);
                          regB.s4567 = read_imagef(src1, 2*COL_STRIDE + 1 + slid*2 + k*8); }
          MAC_Q4K_BLOCK(ts2, ws, regB, 2, 3); }
#else
        // Per-column: load only this column's activation (single regB live at a
        // time -> 1/3 the activation register pressure vs holding all 3) then
        // dequant against the shared weights. Cuts the private-mem spill.
#ifdef VECTOR_SUB_GROUP_BROADCAST
        { if (slid < 4) { regB.s0123 = read_imagef(src1, 0*COL_STRIDE     + slid*2 + k*8);
                          regB.s4567 = read_imagef(src1, 0*COL_STRIDE + 1 + slid*2 + k*8); }
          dequantizeBlockAccum_ns_sgbroadcast_8_hi(ts0, as_ushort8(regA_hi), regS, regM, regB);
          dequantizeBlockAccum_ns_sgbroadcast_8_lo(ts0, as_ushort8(regA_lo), regS, regM, regB); }
        { if (slid < 4) { regB.s0123 = read_imagef(src1, 1*COL_STRIDE     + slid*2 + k*8);
                          regB.s4567 = read_imagef(src1, 1*COL_STRIDE + 1 + slid*2 + k*8); }
          dequantizeBlockAccum_ns_sgbroadcast_8_hi(ts1, as_ushort8(regA_hi), regS, regM, regB);
          dequantizeBlockAccum_ns_sgbroadcast_8_lo(ts1, as_ushort8(regA_lo), regS, regM, regB); }
        { if (slid < 4) { regB.s0123 = read_imagef(src1, 2*COL_STRIDE     + slid*2 + k*8);
                          regB.s4567 = read_imagef(src1, 2*COL_STRIDE + 1 + slid*2 + k*8); }
          dequantizeBlockAccum_ns_sgbroadcast_8_hi(ts2, as_ushort8(regA_hi), regS, regM, regB);
          dequantizeBlockAccum_ns_sgbroadcast_8_lo(ts2, as_ushort8(regA_lo), regS, regM, regB); }
#else
        { if (slid < 4) { regB.s0123 = read_imagef(src1, 0*COL_STRIDE     + slid*2 + k*8);
                          regB.s4567 = read_imagef(src1, 0*COL_STRIDE + 1 + slid*2 + k*8); }
          dequantizeBlockAccum_ns_sgbroadcast_1_hi(ts0, as_ushort8(regA_hi), regS, regM, regB);
          dequantizeBlockAccum_ns_sgbroadcast_1_lo(ts0, as_ushort8(regA_lo), regS, regM, regB); }
        { if (slid < 4) { regB.s0123 = read_imagef(src1, 1*COL_STRIDE     + slid*2 + k*8);
                          regB.s4567 = read_imagef(src1, 1*COL_STRIDE + 1 + slid*2 + k*8); }
          dequantizeBlockAccum_ns_sgbroadcast_1_hi(ts1, as_ushort8(regA_hi), regS, regM, regB);
          dequantizeBlockAccum_ns_sgbroadcast_1_lo(ts1, as_ushort8(regA_lo), regS, regM, regB); }
        { if (slid < 4) { regB.s0123 = read_imagef(src1, 2*COL_STRIDE     + slid*2 + k*8);
                          regB.s4567 = read_imagef(src1, 2*COL_STRIDE + 1 + slid*2 + k*8); }
          dequantizeBlockAccum_ns_sgbroadcast_1_hi(ts2, as_ushort8(regA_hi), regS, regM, regB);
          dequantizeBlockAccum_ns_sgbroadcast_1_lo(ts2, as_ushort8(regA_lo), regS, regM, regB); }
#endif
#endif // Q4K_MC3_DEQUANT_ONCE
    }

    // cross-subgroup reduce: pack the 3 columns' float2 into a float8 (6 used).
    local float8 reduceLM[SUBGROUP_SIZE * 3];
    float8 acc = (float8)(ts0.s0, ts0.s1, ts1.s0, ts1.s1, ts2.s0, ts2.s1, 0.0f, 0.0f);
    if (groupId == 1) { reduceLM[SUBGROUP_SIZE * 0 + slid] = acc; }
    if (groupId == 2) { reduceLM[SUBGROUP_SIZE * 1 + slid] = acc; }
    if (groupId == 3) { reduceLM[SUBGROUP_SIZE * 2 + slid] = acc; }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (groupId == 0) {
        acc += reduceLM[SUBGROUP_SIZE * 0 + slid];
        acc += reduceLM[SUBGROUP_SIZE * 1 + slid];
        acc += reduceLM[SUBGROUP_SIZE * 2 + slid];
        dst = (global float*)((global char*)dst + offsetd);
        // dst is column-major [M rows x 3 cols]: (row, col) at col*M + row
        vstore2((float2)(acc.s0, acc.s1), 0, &(dst[0 * M + gid * 2]));
        vstore2((float2)(acc.s2, acc.s3), 0, &(dst[1 * M + gid * 2]));
        vstore2((float2)(acc.s4, acc.s5), 0, &(dst[2 * M + gid * 2]));
    }
}

// --- Split-K-across-workgroups decode GEMV (small-M projections) ----------------
// A single-token GEMV makes only ceil(M/2/64) workgroups; a WG runs on one Adreno
// compute unit, so for small M (Kcur/Vcur, M=512 -> 4 WGs) most of the 16 CUs sit
// idle and the matmul caps at ~30 GB/s even with a wide intra-WG K-split. This
// variant adds a SECOND grid dimension of `ksplit` workgroups that each reduce a
// disjoint slice of K and write a per-slice partial; kernel_gemv_splitk_reduce_f32
// then sums the partials into dst. Microbench (X2): M=512 31->47 GB/s (+52%),
// M=1024 60->72 (+20%). Identical math/layout to the base kernel (physical block
// stride 4*M, get_scale_min_k4) -> coherent. Gated host-side to M<=1024 (M>=2048
// already fills the CUs and the extra reduce dispatch only hurts).
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_gemv_noshuffle_q4_k_f32_splitk(
        read_only  image1d_buffer_t src0_q,
        global half2  * src0_d,
        global half2  * src0_m,
        global uchar  * src0_s,
        read_only  image1d_buffer_t src1,
        global float * partial,          // [ksplit * M], slice-major
        int ne00,
        int ne01,
        uchar mask_d6,
        uchar mask_d4,
        uchar mask_hi2)
{
    uint groupId = get_local_id(1);
    uint gid     = get_global_id(0);
    ushort slid  = get_sub_group_local_id();
    uint nsg     = get_local_size(1);
    uint ksplit  = get_num_groups(1);
    uint kslice  = get_group_id(1);

    uint K = ne00;
    uint M = ne01;
    uint LINE_STRIDE_A  = M / 2;
    uint BLOCK_STRIDE_A = 4 * M;      // physical, independent of the K-split
    uint scales_per_row = (K / QK_K) * 12;

    private uint4  regA;
    private half2  regS, regM;
    private float8 regB;
    private float2 totalSum = (float2)(0.0f);

    // each (kslice, subgroup) pair owns a disjoint set of K-blocks
    for (uint k = kslice * nsg + groupId; k < (K / 32); k += ksplit * nsg) {
        uint sb = k / 8;
        uint j  = k % 8;
        half2 d   = src0_d[gid + sb * LINE_STRIDE_A];
        half2 dm  = src0_m[gid + sb * LINE_STRIDE_A];
        global const uchar * sc0 = src0_s + 2 * gid * scales_per_row + sb * 12;
        global const uchar * sc1 = src0_s + (2 * gid + 1) * scales_per_row + sb * 12;
        uchar sv0, mn0, sv1, mn1;
        get_scale_min_k4(j, sc0, &sv0, &mn0, mask_d6, mask_d4, mask_hi2);
        get_scale_min_k4(j, sc1, &sv1, &mn1, mask_d6, mask_d4, mask_hi2);
        regS = convert_half2(convert_float2(d)  * convert_float2((uchar2)(sv0, sv1)));
        regM = convert_half2(convert_float2(dm) * convert_float2((uchar2)(mn0, mn1)));
        if (slid < 4) {
            regB.s0123 = read_imagef(src1, (slid * 2 + k * 8));
            regB.s4567 = read_imagef(src1, (1 + slid * 2 + k * 8));
        }
        regA.s0 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 0)).x;
        regA.s1 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 1)).x;
        regA.s2 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 2)).x;
        regA.s3 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 3)).x;
#ifdef VECTOR_SUB_GROUP_BROADCAST
        dequantizeBlockAccum_ns_sgbroadcast_8_hi(totalSum, as_ushort8(regA), regS, regM, regB);
#else
        dequantizeBlockAccum_ns_sgbroadcast_1_hi(totalSum, as_ushort8(regA), regS, regM, regB);
#endif
        regA.s0 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 4)).x;
        regA.s1 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 5)).x;
        regA.s2 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 6)).x;
        regA.s3 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 7)).x;
#ifdef VECTOR_SUB_GROUP_BROADCAST
        dequantizeBlockAccum_ns_sgbroadcast_8_lo(totalSum, as_ushort8(regA), regS, regM, regB);
#else
        dequantizeBlockAccum_ns_sgbroadcast_1_lo(totalSum, as_ushort8(regA), regS, regM, regB);
#endif
    }

    local float2 reduceLM[SUBGROUP_SIZE * 15];
    if (groupId > 0) {
        reduceLM[SUBGROUP_SIZE * (groupId - 1) + slid] = totalSum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (groupId == 0) {
        for (uint i = 0; i < nsg - 1; ++i) {
            totalSum += reduceLM[SUBGROUP_SIZE * i + slid];
        }
        vstore2(totalSum, 0, &(partial[kslice * M + gid * 2]));
    }
}

// Sum the per-slice partials [ksplit * M] into dst[M]; applies the dst byte offset.
kernel void kernel_gemv_splitk_reduce_f32(
        global float * partial,
        global float * dst,
        ulong offsetd,
        int   ne01,         // M
        int   ksplit)
{
    uint r = get_global_id(0);
    if (r >= (uint)ne01) return;
    float acc = 0.0f;
    for (uint s = 0; s < (uint)ksplit; ++s) {
        acc += partial[s * (uint)ne01 + r];
    }
    dst = (global float*)((global char*)dst + offsetd);
    dst[r] = acc;
}

// ============================================================================
// kernel_gemma4_perlayer_block  --  software-grid-barrier MEGAKERNEL
// ----------------------------------------------------------------------------
// Fuses the entire Gemma-4 per-layer-embedding block (graph nodes DG48..65,
// one per layer x 42) into a SINGLE dispatch, removing 7 of 8 GPU launch
// boundaries. The block is dispatch-bound at decode (both matmuls are small --
// one dim is n_pl=256 -- so ~8 x ~5us launch overhead dwarfs the few-us of
// q4_K BW work). Per-layer-block + q4_K-cap microbenches predicted ~+5% tg E4B
// and settled the resident-cap / capped-R-GEMV risks (see memory notes).
//
// Op chain (decode, n_tokens=1), both MUL_MATs are q4_K:
//   1 MUL_MAT inp_gate : y1[n_pl] = Wg[n_pl x D] . x[D]            (all-WG GEMV)
//   2 UNARY   gelu      : y1 = gelu(y1)                            (single-WG)
//   3 MUL     gate      : y2 = y1 * gate[n_pl]                     (single-WG)
//   4 MUL_MAT proj      : y3[D]   = Wp[D x n_pl] . y2[n_pl]        (all-WG GEMV)
//   5 RMS_NORM          : r = rms_norm(y3)                         (single-WG)
//   6 MUL     post_norm : r = r * wn[D]                            (single-WG)
//   7 ADD     residual  : r = r + x[D]   (x == pe_in)              (single-WG)
//   8 MUL     out_scale : out = r * oscale[D]                      (single-WG)
//
// Persistent-threads grid: launch R workgroups (R=get_num_groups(0), host picks
// R=128, well under the X2 resident cap of ~256 measured for this occupancy);
// each WG is 64 x nsg lanes (REQD_SUBGROUP_SIZE_64, nsg = K-split subgroups).
// The GEMV stages grid-stride over output row-pairs; the small stages run on
// WG 0. A Xiao&Feng atomic-counter grid_barrier (plain global atomics, L2-
// coherent on Adreno -- validated in microbench/global_barrier_microbench.cpp)
// separates the 4 stages. counters[3] must be pre-zeroed by the host each call.
//
// Activations are read from PLAIN global buffers (not the image1d the standalone
// GEMV uses): the proj activation y2 is computed mid-kernel and can't be bound
// as an image. Activations are tiny (<=2560 floats) so the texture-cache loss is
// negligible; the WEIGHTS (the BW item) stay as image1d_buffer reads. The dequant
// math reuses the file's get_scale_min_k4 + dequantizeBlockAccum_* macros
// verbatim, so per-block weight handling is identical to the standalone GEMV.
// nsg differs from the standalone wide-split (16), so output is numerically
// faithful / coherent, NOT byte-identical to the per-op path.
// ============================================================================

#define GELU_COEF_A    0.044715f
#define SQRT_2_OVER_PI 0.79788456080286535587989211986876f

// Xiao&Feng arrival-counter grid barrier across all R = get_num_groups(0) WGs.
// A plain barrier(CLK_GLOBAL_MEM_FENCE) only orders accesses WITHIN a workgroup;
// it does NOT publish one WG's regular global writes to another WG. So we use
// C11 atomics with DEVICE-SCOPE release/acquire: the WG barrier orders all lanes'
// data writes before lane-0's release atomic_fetch_add; readers acquire-load the
// counter, which makes those data writes visible cross-WG. counter pre-zeroed.
inline void mega_grid_barrier(volatile global atomic_int * counter) {
    int R = get_num_groups(0);
    work_group_barrier(CLK_GLOBAL_MEM_FENCE);
    if (get_local_id(0) == 0 && get_local_id(1) == 0) {
        atomic_fetch_add_explicit(counter, 1, memory_order_release, memory_scope_device);
        while (atomic_load_explicit(counter, memory_order_acquire, memory_scope_device) < R) {}
    }
    work_group_barrier(CLK_GLOBAL_MEM_FENCE);
}

// Cross-WG-coherent scratch access. On Adreno X2 regular global loads/stores are
// NOT visible across workgroups even after a grid barrier (per-CU L1 is not
// snooped), but ATOMICS are coherent (they go to L2). So route the shared scratch
// (y1/y2/y3) through traditional int atomics on the float bits: atomic_xchg to
// store (write-through to L2), atomic_add(.,0) to load (read-from-L2, bypass L1).
inline void  mega_st(volatile global float * p, uint i, float v) {
#ifdef MEGA_REGULAR
    p[i] = v;   // regular store: rely on the grid barrier's release for cross-WG publish
#else
    atomic_xchg((volatile global int *)p + i, as_int(v));
#endif
}
inline float mega_ld(volatile global float * p, uint i) {
#ifdef MEGA_REGULAR
    return p[i];   // regular load: barrier's acquire makes other WGs' writes visible
#else
    return as_float(atomic_add((volatile global int *)p + i, 0));
#endif
}

// One q4_K GEMV output fiber (2 rows): totalSum valid on groupId==0 only.
// act is the activation vector as a plain global float buffer (K elems).
// reduceLM is a WG-shared scratch (>= SUBGROUP_SIZE * (nsg-1) float2).
inline float2 mega_q4k_gemv_fiber(
        read_only image1d_buffer_t src0_q,
        global const half2 * src0_d,
        global const half2 * src0_m,
        global const uchar * src0_s,
        global const float * act,
        uint gid, uint groupId, uint nsg, ushort slid,
        uint K, uint M,
        uchar mask_d6, uchar mask_d4, uchar mask_hi2,
        local float2 * reduceLM) {
    uint LINE_STRIDE_A  = M / 2;
    uint BLOCK_STRIDE_A = 4 * M;
    uint scales_per_row = (K / QK_K) * 12;

    private uint4  regA;
    private half2  regS, regM;
    private float8 regB;
    private float2 totalSum = (float2)(0.0f);

    for (uint k = groupId; k < (K / 32); k += nsg) {
        uint sb = k / 8;
        uint j  = k % 8;

        half2 d   = src0_d[gid + sb * LINE_STRIDE_A];
        half2 dm  = src0_m[gid + sb * LINE_STRIDE_A];

        global const uchar * sc0 = src0_s + 2 * gid * scales_per_row + sb * 12;
        global const uchar * sc1 = src0_s + (2 * gid + 1) * scales_per_row + sb * 12;

        uchar sv0, mn0, sv1, mn1;
        get_scale_min_k4(j, sc0, &sv0, &mn0, mask_d6, mask_d4, mask_hi2);
        get_scale_min_k4(j, sc1, &sv1, &mn1, mask_d6, mask_d4, mask_hi2);

        regS = convert_half2(convert_float2(d)  * convert_float2((uchar2)(sv0, sv1)));
        regM = convert_half2(convert_float2(dm) * convert_float2((uchar2)(mn0, mn1)));

        if (slid < 4) {
            regB.s0123 = vload4(slid * 2 + k * 8, act);
            regB.s4567 = vload4(1 + slid * 2 + k * 8, act);
        }

        regA.s0 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 0)).x;
        regA.s1 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 1)).x;
        regA.s2 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 2)).x;
        regA.s3 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 3)).x;
#ifdef VECTOR_SUB_GROUP_BROADCAST
        dequantizeBlockAccum_ns_sgbroadcast_8_hi(totalSum, as_ushort8(regA), regS, regM, regB);
#else
        dequantizeBlockAccum_ns_sgbroadcast_1_hi(totalSum, as_ushort8(regA), regS, regM, regB);
#endif

        regA.s0 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 4)).x;
        regA.s1 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 5)).x;
        regA.s2 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 6)).x;
        regA.s3 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 7)).x;
#ifdef VECTOR_SUB_GROUP_BROADCAST
        dequantizeBlockAccum_ns_sgbroadcast_8_lo(totalSum, as_ushort8(regA), regS, regM, regB);
#else
        dequantizeBlockAccum_ns_sgbroadcast_1_lo(totalSum, as_ushort8(regA), regS, regM, regB);
#endif
    }

    if (groupId > 0) {
        reduceLM[SUBGROUP_SIZE * (groupId - 1) + slid] = totalSum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (groupId == 0) {
        for (uint i = 0; i < nsg - 1; ++i) {
            totalSum += reduceLM[SUBGROUP_SIZE * i + slid];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);   // protect reduceLM reuse on the next fiber
    return totalSum;
}

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_gemma4_perlayer_block(
        read_only image1d_buffer_t g_q,        // inp_gate q4_K weights (image)
        global const half2  * g_d,
        global const half2  * g_m,
        global const uchar  * g_s,
        read_only image1d_buffer_t p_q,        // proj q4_K weights (image)
        global const half2  * p_d,
        global const half2  * p_m,
        global const uchar  * p_s,
        global const float  * x,               // pe_in [D] (inp_gate act + residual)
        ulong                 offset_x,
        global const float  * gate,            // inp_this_layer [n_pl]
        ulong                 offset_gate,
        global const float  * wn,              // per_layer_post_norm weight [D]
        ulong                 offset_wn,
        global const float  * oscale,          // layer_output_scale weight [D]
        ulong                 offset_oscale,
        global float        * out,
        ulong                 offset_out,
        global float        * scratch,         // y1[n_pl] | y2[n_pl] | y3[D]
        volatile global atomic_int * counters,  // [3], pre-zeroed
        int   D,
        int   n_pl,
        float eps,
        uchar mask_d6,
        uchar mask_d4,
        uchar mask_hi2) {
    local float2 reduceLM[SUBGROUP_SIZE * 15];   // up to 16 subgroups

    x      = (global const float *)((global const char *)x      + offset_x);
    gate   = (global const float *)((global const char *)gate   + offset_gate);
    wn     = (global const float *)((global const char *)wn     + offset_wn);
    oscale = (global const float *)((global const char *)oscale + offset_oscale);

    global float * y1 = scratch;
    global float * y2 = scratch + n_pl;
    global float * y3 = scratch + 2 * n_pl;

    uint lid0    = get_local_id(0);
    uint groupId = get_local_id(1);
    uint nsg     = get_local_size(1);
    ushort slid  = get_sub_group_local_id();
    uint flid    = groupId * SUBGROUP_SIZE + lid0;   // flat lane id
    uint nflat   = nsg * SUBGROUP_SIZE;
    uint gstride = get_global_size(0);

    // ---- stage 1: y1 = Wg . x  (q4_K GEMV, K=D, M=n_pl) ----
    for (uint fiber = get_global_id(0); fiber < (uint)(n_pl / 2); fiber += gstride) {
        float2 r = mega_q4k_gemv_fiber(g_q, g_d, g_m, g_s, x, fiber, groupId, nsg, slid,
                                       (uint)D, (uint)n_pl, mask_d6, mask_d4, mask_hi2, reduceLM);
        if (groupId == 0) { y1[fiber * 2] = r.s0; y1[fiber * 2 + 1] = r.s1; }
    }
    mega_grid_barrier(counters + 0);

    // ---- stage 2/3: y2 = gelu(y1) * gate  (single WG) ----
    if (get_group_id(0) == 0) {
        for (uint i = flid; i < (uint)n_pl; i += nflat) {
            float v  = y1[i];
            float ge = 0.5f * v * (1.0f + tanh(SQRT_2_OVER_PI * v * (1.0f + GELU_COEF_A * v * v)));
            y2[i] = ge * gate[i];
        }
    }
    mega_grid_barrier(counters + 1);

    // ---- stage 4: y3 = Wp . y2  (q4_K GEMV, K=n_pl, M=D) ----
    for (uint fiber = get_global_id(0); fiber < (uint)(D / 2); fiber += gstride) {
        float2 r = mega_q4k_gemv_fiber(p_q, p_d, p_m, p_s, y2, fiber, groupId, nsg, slid,
                                       (uint)n_pl, (uint)D, mask_d6, mask_d4, mask_hi2, reduceLM);
        if (groupId == 0) { y3[fiber * 2] = r.s0; y3[fiber * 2 + 1] = r.s1; }
    }
    mega_grid_barrier(counters + 2);

    // ---- stage 5-8: out = (rms_norm(y3)*wn + x) * oscale  (single WG) ----
    if (get_group_id(0) == 0) {
        local float * fred = (local float *) reduceLM;
        float p = 0.0f;
        for (uint i = flid; i < (uint)D; i += nflat) p += y3[i] * y3[i];
        fred[flid] = p;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint s = nflat / 2; s > 0; s >>= 1) {
            if (flid < s) fred[flid] += fred[flid + s];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        float rms = rsqrt(fred[0] / (float)D + eps);
        float os  = oscale[0];   // layer_output_scale is a SCALAR [1], broadcast over D
        global float * o = (global float *)((global char *)out + offset_out);
        for (uint i = flid; i < (uint)D; i += nflat) {
            o[i] = (y3[i] * rms * wn[i] + x[i]) * os;
        }
    }
}

// ============================================================================
// kernel_gemma4_perlayer_block_f32  --  F32-weight variant of the megakernel.
// In Gemma-4-E4B-Q4_K_M the per-layer inp_gate/proj weights are stored F32 (not
// q4_K), so the q4_K variant above never fires. This variant embeds a plain f32
// GEMV: fiber = output row (grid-strided), K split over the nsg subgroups, cross-
// subgroup reduce in __local (mirrors the q4_K kernel's WG-uniform barrier
// structure -> avoids the Adreno full-subgroup-reduce miscompile). Same grid
// barrier + gelu_mul + rms_mul_add_scale stages.
// ============================================================================

// One f32 GEMV output row, activation from GLOBAL (coherent input). groupId==0.
// K is split across the nsg subgroups; when K is a multiple of nsg*4 each
// subgroup streams a CONTIGUOUS chunk with float4 loads (4x fewer load insns +
// better BW than the scalar interleaved path), else falls back to scalar stride.
inline float mega_f32_gemv_fiber(global const float * W, global const float * act,
                                 uint row, uint lid0, uint groupId, uint nsg, uint K,
                                 local float * red) {
    float p = 0.0f;
    global const float * Wr = W + (ulong)row * K;
    if ((K % (nsg * 4u)) == 0u) {
        uint kc = K / nsg;            // contiguous chunk per subgroup (mult of 4)
        uint k0 = groupId * kc;       // chunk base (mult of 4)
        float4 acc = (float4)(0.0f);
        for (uint k = 0; k < kc; k += 4) {
            acc += vload4((k0 + k) >> 2, Wr) * vload4((k0 + k) >> 2, act);
        }
        p = acc.x + acc.y + acc.z + acc.w;
    } else {
        for (uint k = groupId; k < K; k += nsg) p += Wr[k] * act[k];
    }
    if (groupId > 0) red[SUBGROUP_SIZE * (groupId - 1) + lid0] = p;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (groupId == 0) { for (uint i = 0; i < nsg - 1; ++i) p += red[SUBGROUP_SIZE * i + lid0]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    return p;
}

// Same, activation from LOCAL (used after atomically staging cross-WG y2).
inline float mega_f32_gemv_fiber_l(global const float * W, local const float * act,
                                   uint row, uint lid0, uint groupId, uint nsg, uint K,
                                   local float * red) {
    float p = 0.0f;
    global const float * Wr = W + (ulong)row * K;
    if ((K % (nsg * 4u)) == 0u) {
        uint kc = K / nsg;
        uint k0 = groupId * kc;
        float4 acc = (float4)(0.0f);
        for (uint k = 0; k < kc; k += 4) {
            acc += vload4((k0 + k) >> 2, Wr) * vload4((k0 + k) >> 2, act);
        }
        p = acc.x + acc.y + acc.z + acc.w;
    } else {
        for (uint k = groupId; k < K; k += nsg) p += Wr[k] * act[k];
    }
    if (groupId > 0) red[SUBGROUP_SIZE * (groupId - 1) + lid0] = p;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (groupId == 0) { for (uint i = 0; i < nsg - 1; ++i) p += red[SUBGROUP_SIZE * i + lid0]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    return p;
}

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_gemma4_perlayer_block_f32(
        global const float  * g_w,             // inp_gate f32 weights [n_pl x D]
        ulong                 offset_gw,
        global const float  * p_w,             // proj f32 weights [D x n_pl]
        ulong                 offset_pw,
        global const float  * x,               // pe_in [D]
        ulong                 offset_x,
        global const float  * gate,            // inp_this_layer [n_pl]
        ulong                 offset_gate,
        global const float  * wn,              // per_layer_post_norm weight [D]
        ulong                 offset_wn,
        global const float  * oscale,          // layer_output_scale weight [D]
        ulong                 offset_oscale,
        global float        * out,
        ulong                 offset_out,
        volatile global float * scratch,       // y1[n_pl] | y2[n_pl] | y3[D]
        volatile global atomic_int * counters,  // [3], pre-zeroed
        int   D,
        int   n_pl,
        float eps) {
    local float2 reduceLM[SUBGROUP_SIZE * 15];
    local float * red = (local float *) reduceLM;

    g_w    = (global const float *)((global const char *)g_w    + offset_gw);
    p_w    = (global const float *)((global const char *)p_w    + offset_pw);
    x      = (global const float *)((global const char *)x      + offset_x);
    gate   = (global const float *)((global const char *)gate   + offset_gate);
    wn     = (global const float *)((global const char *)wn     + offset_wn);
    oscale = (global const float *)((global const char *)oscale + offset_oscale);

    uint lid0    = get_local_id(0);
    uint groupId = get_local_id(1);
    uint nsg     = get_local_size(1);
    uint flid    = groupId * SUBGROUP_SIZE + lid0;
    uint nflat   = nsg * SUBGROUP_SIZE;
    uint gstride = get_global_size(0);

    volatile global float * y1 = scratch;
    volatile global float * y2 = scratch + n_pl;
    volatile global float * y3 = scratch + 2 * n_pl;
    local float ly2[256];   // staged y2 (n_pl<=256 for gemma4)

    // stage 1: y1 = Wg . x  (M=n_pl, K=D). x is a coherent prior-dispatch input
    // (regular read OK); y1 is cross-WG scratch -> atomic store (mega_st).
    for (uint row = get_global_id(0); row < (uint)n_pl; row += gstride) {
        float r = mega_f32_gemv_fiber(g_w, x, row, lid0, groupId, nsg, (uint)D, red);
        if (groupId == 0) mega_st(y1, row, r);
    }
    mega_grid_barrier(counters + 0);

    // stage 2/3: y2 = gelu(y1) * gate  (single WG). y1 written by other WGs ->
    // atomic load (mega_ld); y2 read by all WGs next -> atomic store.
    if (get_group_id(0) == 0) {
        for (uint i = flid; i < (uint)n_pl; i += nflat) {
            float v  = mega_ld(y1, i);
            float ge = 0.5f * v * (1.0f + tanh(SQRT_2_OVER_PI * v * (1.0f + GELU_COEF_A * v * v)));
            mega_st(y2, i, ge * gate[i]);
        }
    }
    mega_grid_barrier(counters + 1);

    // stage 4: y3 = Wp . y2  (M=D, K=n_pl). Stage cross-WG y2 into __local once
    // via atomic loads, then GEMV reads it from local (avoids per-MAC atomics).
    for (uint i = flid; i < (uint)n_pl; i += nflat) ly2[i] = mega_ld(y2, i);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint row = get_global_id(0); row < (uint)D; row += gstride) {
        float r = mega_f32_gemv_fiber_l(p_w, ly2, row, lid0, groupId, nsg, (uint)n_pl, red);
        if (groupId == 0) mega_st(y3, row, r);
    }
    mega_grid_barrier(counters + 2);

    // stage 5-8: out = (rms_norm(y3)*wn + x) * oscale  (single WG). y3 written by
    // other WGs -> atomic load.
    if (get_group_id(0) == 0) {
        float p = 0.0f;
        for (uint i = flid; i < (uint)D; i += nflat) { float v = mega_ld(y3, i); p += v * v; }
        red[flid] = p;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint s = nflat / 2; s > 0; s >>= 1) {
            if (flid < s) red[flid] += red[flid + s];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        float rms = rsqrt(red[0] / (float)D + eps);
        float os  = oscale[0];   // layer_output_scale is a SCALAR [1], broadcast over D
        global float * o = (global float *)((global char *)out + offset_out);
        for (uint i = flid; i < (uint)D; i += nflat) {
            float v = mega_ld(y3, i);
            o[i] = (v * rms * wn[i] + x[i]) * os;
        }
    }
}

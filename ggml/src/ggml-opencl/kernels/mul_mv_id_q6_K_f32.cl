#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#else
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

#ifdef cl_intel_required_subgroup_size
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#define INTEL_GPU 1
#define REQD_SUBGROUP_SIZE_16 __attribute__((intel_reqd_sub_group_size(16)))
#define REQD_SUBGROUP_SIZE_32 __attribute__((intel_reqd_sub_group_size(32)))
#elif defined(cl_qcom_reqd_sub_group_size)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#endif

#define Q6_K_MASK1 0x03
#define Q6_K_MASK2 0x0C
#define Q6_K_MASK3 0x30
#define Q6_K_MASK4 0xC0

#define QK_K       256

inline float block_q_6_K_dot_y_flat_id(
    global uchar * blk_ql,
    global uchar * blk_qh,
    global char  * blk_scales,
    global half  * blk_d,
    global float * yy,
    int ib,
    int ip,
    int is,
    int l0
) {
    int y_offset   = 128*ip + l0;
    int q_offset_l =  64*ip + l0;
    int q_offset_h =  32*ip + l0;

    global uchar * q1 = blk_ql     + ib*128 + q_offset_l;
    global uchar * q2 = q1         + QK_K/8;
    global uchar * qh = blk_qh     + ib*64 + q_offset_h;
    global char  * sc = blk_scales + ib*16 + is;

    global float * y = yy + ib * QK_K + y_offset;

    float dall = blk_d[ib];

    float  sumf = 0;
    float4 sums = {0.f, 0.f, 0.f, 0.f};

    sums.s0 += y[0+ 0] * ((float)((q1[0] & 0xF) | ((qh[0] & Q6_K_MASK1) << 4)) - 32.f);
    sums.s1 += y[0+32] * ((float)((q2[0] & 0xF) | ((qh[0] & Q6_K_MASK2) << 2)) - 32.f);
    sums.s2 += y[0+64] * ((float)((q1[0]  >> 4) | ((qh[0] & Q6_K_MASK3) << 0)) - 32.f);
    sums.s3 += y[0+96] * ((float)((q2[0]  >> 4) | ((qh[0] & Q6_K_MASK4) >> 2)) - 32.f);

    sums.s0 += y[1+ 0] * ((float)((q1[1] & 0xF) | ((qh[1] & Q6_K_MASK1) << 4)) - 32.f);
    sums.s1 += y[1+32] * ((float)((q2[1] & 0xF) | ((qh[1] & Q6_K_MASK2) << 2)) - 32.f);
    sums.s2 += y[1+64] * ((float)((q1[1]  >> 4) | ((qh[1] & Q6_K_MASK3) << 0)) - 32.f);
    sums.s3 += y[1+96] * ((float)((q2[1]  >> 4) | ((qh[1] & Q6_K_MASK4) >> 2)) - 32.f);

    sums.s0 += y[2+ 0] * ((float)((q1[2] & 0xF) | ((qh[2] & Q6_K_MASK1) << 4)) - 32.f);
    sums.s1 += y[2+32] * ((float)((q2[2] & 0xF) | ((qh[2] & Q6_K_MASK2) << 2)) - 32.f);
    sums.s2 += y[2+64] * ((float)((q1[2]  >> 4) | ((qh[2] & Q6_K_MASK3) << 0)) - 32.f);
    sums.s3 += y[2+96] * ((float)((q2[2]  >> 4) | ((qh[2] & Q6_K_MASK4) >> 2)) - 32.f);

    sums.s0 += y[3+ 0] * ((float)((q1[3] & 0xF) | ((qh[3] & Q6_K_MASK1) << 4)) - 32.f);
    sums.s1 += y[3+32] * ((float)((q2[3] & 0xF) | ((qh[3] & Q6_K_MASK2) << 2)) - 32.f);
    sums.s2 += y[3+64] * ((float)((q1[3]  >> 4) | ((qh[3] & Q6_K_MASK3) << 0)) - 32.f);
    sums.s3 += y[3+96] * ((float)((q2[3]  >> 4) | ((qh[3] & Q6_K_MASK4) >> 2)) - 32.f);

    sumf += dall * (sums.s0 * sc[0] + sums.s1 * sc[2] + sums.s2 * sc[4] + sums.s3 * sc[6]);

    return sumf;
}

#ifdef INTEL_GPU
#define N_DST 4
#define N_SIMDGROUP 2
#define N_SIMDWIDTH 16
#elif defined (ADRENO_GPU)
#define N_DST 4
#define N_SIMDGROUP 2
#define N_SIMDWIDTH 64
#endif

#define BLOCK_STRIDE (N_SIMDWIDTH/16)

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_16
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mv_id_q6_K_f32(
        global uchar * src0_ql,
        global uchar * src0_qh,
        global char  * src0_s,
        global half  * src0_d,
        global float * src1,
        ulong          offset1,
        global char  * src2,
        ulong          offset2,
        global float * dst,
        ulong          offsetd,
        int            ne00,
        int            ne01,
        int            ne11,
        int            ne12,
        ulong          nb11,
        ulong          nb12,
        int            ne20,
        int            ne21,
        ulong          nb21,
        int            ne0,
        int            ne1
) {
    src1 = (global float *) ((global char *) src1 + offset1);
    src2 = (global char  *) src2 + offset2;
    dst  = (global float *) ((global char *) dst  + offsetd);

    int iid1 = get_group_id(2)/ne20;
    int idx  = get_group_id(2)%ne20;

    int i02 = ((global int *) (src2 + iid1*nb21))[idx];

    int i11_ = idx % ne11;
    int i12_ = iid1;

    int i1 = idx;
    int i2 = i12_;

    int nb = ne00/QK_K;

    int r0 = get_group_id(0);
    int first_row = (N_SIMDGROUP * r0 + get_sub_group_id()) * N_DST;

    // src0 layout: [ne00, ne01, ne02=n_experts] in flat SoA buffers.
    // offset_src0 (in super-block units) = first_row's blocks within expert i02.
    ulong offset_src0    = (ulong)first_row*nb + (ulong)i02*nb*ne01;
    ulong offset_src0_ql = offset_src0 * 128;
    ulong offset_src0_qh = offset_src0 * 64;
    ulong offset_src0_s  = offset_src0 * 16;
    ulong offset_src0_d  = offset_src0;

    global uchar * blk_ql     = src0_ql + offset_src0_ql;
    global uchar * blk_qh     = src0_qh + offset_src0_qh;
    global char  * blk_scales = src0_s  + offset_src0_s;
    global half  * blk_d      = src0_d  + offset_src0_d;

    global float * yy = (global float *)
        ((global char *) src1 + i11_*nb11 + i12_*nb12);

    int tid = get_sub_group_local_id()/BLOCK_STRIDE;
    int ix  = get_sub_group_local_id()%BLOCK_STRIDE;
    int ip  = tid/8;
    int il  = tid%8;
    int n   = 4;
    int l0  = n*il;
    int is  = 8*ip + l0/16;

    float4 sumf = 0;

    for (int ib = ix; ib < nb; ib += BLOCK_STRIDE) {
        if (first_row + 0 < ne01) {
            sumf.s0 += block_q_6_K_dot_y_flat_id(blk_ql + 0*nb*128, blk_qh + 0*nb*64, blk_scales + 0*nb*16, blk_d + 0*nb, yy, ib, ip, is, l0);
        }
        if (first_row + 1 < ne01) {
            sumf.s1 += block_q_6_K_dot_y_flat_id(blk_ql + 1*nb*128, blk_qh + 1*nb*64, blk_scales + 1*nb*16, blk_d + 1*nb, yy, ib, ip, is, l0);
        }
        if (first_row + 2 < ne01) {
            sumf.s2 += block_q_6_K_dot_y_flat_id(blk_ql + 2*nb*128, blk_qh + 2*nb*64, blk_scales + 2*nb*16, blk_d + 2*nb, yy, ib, ip, is, l0);
        }
        if (first_row + 3 < ne01) {
            sumf.s3 += block_q_6_K_dot_y_flat_id(blk_ql + 3*nb*128, blk_qh + 3*nb*64, blk_scales + 3*nb*16, blk_d + 3*nb, yy, ib, ip, is, l0);
        }
    }

    float4 tot = (float4)(
        sub_group_reduce_add(sumf.s0),
        sub_group_reduce_add(sumf.s1),
        sub_group_reduce_add(sumf.s2),
        sub_group_reduce_add(sumf.s3)
    );

    global float * dst_row = dst + (ulong)i1*ne0 + (ulong)i2*ne1*ne0;

    if (get_sub_group_local_id() == 0) {
        if (first_row + 0 < ne01) dst_row[first_row + 0] = tot.s0;
        if (first_row + 1 < ne01) dst_row[first_row + 1] = tot.s1;
        if (first_row + 2 < ne01) dst_row[first_row + 2] = tot.s2;
        if (first_row + 3 < ne01) dst_row[first_row + 3] = tot.s3;
    }
}

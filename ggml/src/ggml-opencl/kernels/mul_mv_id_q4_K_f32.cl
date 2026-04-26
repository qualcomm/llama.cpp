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

#define QK_K            256
#define BLOCK_Q4K_SIZE  144
#define K_SCALE_SIZE    12

#ifdef INTEL_GPU
#define N_DST 4
#define N_SIMDGROUP 1
#define N_SIMDWIDTH 16
#elif defined (ADRENO_GPU)
#define N_DST 16
#define N_SIMDGROUP 2
#define N_SIMDWIDTH 64
#endif

#define BLOCK_STRIDE (N_SIMDWIDTH/8)

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_16
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mv_id_q4_K_f32(
        global uchar * src0_q,
        global uchar * src0_s,
        global half  * src0_d,
        global half  * src0_dm,
        global char  * src1,
        ulong          offset1,
        global char  * src2,
        ulong          offset2,
        global char  * dst,
        ulong          offsetd,
        int            ne00,
        int            ne01,
        ulong          nb01,
        ulong          nb02,
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
    src1 = src1 + offset1;
    src2 = src2 + offset2;
    dst  = dst  + offsetd;

    int iid1 = get_group_id(2)/ne20;
    int idx  = get_group_id(2)%ne20;

    int i02 = ((global int *) (src2 + iid1*nb21))[idx];

    int i11_ = idx % ne11;
    int i12_ = iid1;

    int i1 = idx;
    int i2 = i12_;

    global char * src1_cur = src1 + i11_*nb11 + i12_*nb12;
    global char * dst_cur  = dst  + (i1*ne0 + i2*ne1*ne0)*sizeof(float);

    ushort kmask1 = 0x3f3f;
    ushort kmask2 = 0x0f0f;
    ushort kmask3 = 0xc0c0;

    int ix = get_sub_group_local_id()/8;
    int it = get_sub_group_local_id()%8;
    int iq = it/4;
    int ir = it%4;

    int nb = ne00/QK_K;

    int r0 = get_group_id(0);
    int first_row = (r0 * N_SIMDGROUP + get_sub_group_id()) * N_DST;

    // Per-expert offset into the SoA-flat buffers, in super-blocks.
    ulong offset_src0_blk = (i02*nb02 + first_row*nb01) / BLOCK_Q4K_SIZE;
    uint  blk             = (uint)(nb01 / BLOCK_Q4K_SIZE);

    global uchar * blk_q  = src0_q  + offset_src0_blk*(QK_K/2);
    global uchar * blk_s  = src0_s  + offset_src0_blk*K_SCALE_SIZE;
    global half  * blk_d  = src0_d  + offset_src0_blk;
    global half  * blk_dm = src0_dm + offset_src0_blk;

    global float * y = (global float *) src1_cur;

    float yl[16];
    float yh[16];
    float sumf[N_DST] = {0.f};
    float all_sum;

    global float * y4 = y + ix * QK_K + 64 * iq + 8 * ir;

    ushort  sc16[4];
    uchar * sc8 = (uchar *) sc16;

    for (int ib = ix; ib < nb; ib += BLOCK_STRIDE) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (int i = 0; i < 8; ++i) {
            yl[i+0] = y4[i+0];   sumy.s0 += yl[i+0];
            yl[i+8] = y4[i+32];  sumy.s1 += yl[i+8];
            yh[i+0] = y4[i+128]; sumy.s2 += yh[i+0];
            yh[i+8] = y4[i+160]; sumy.s3 += yh[i+8];
        }

        global ushort * q1 = (global ushort *)(blk_q + ib * (QK_K/2)) + (16 * iq + 4 * ir);
        global ushort * sc = (global ushort *)(blk_s + ib * K_SCALE_SIZE) + iq;
        global half   * d  = blk_d  + ib;
        global half   * dm = blk_dm + ib;

        for (int row = 0; row < N_DST; row++) {
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            global ushort * q2 = q1 + 32;

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
            for (int i = 0; i < 8; i += 2) {
                acc1.s0 += yl[i+0] * (q1[i/2] & 0x000F);
                acc1.s1 += yl[i+1] * (q1[i/2] & 0x0F00);
                acc1.s2 += yl[i+8] * (q1[i/2] & 0x00F0);
                acc1.s3 += yl[i+9] * (q1[i/2] & 0xF000);
                acc2.s0 += yh[i+0] * (q2[i/2] & 0x000F);
                acc2.s1 += yh[i+1] * (q2[i/2] & 0x0F00);
                acc2.s2 += yh[i+8] * (q2[i/2] & 0x00F0);
                acc2.s3 += yh[i+9] * (q2[i/2] & 0xF000);
            }

            float dall = *d;
            float dmin = *dm;
            sumf[row] += dall * ((acc1.s0 + 1.f/256.f * acc1.s1) * sc8[0] +
                                 (acc1.s2 + 1.f/256.f * acc1.s3) * sc8[1] * 1.f/16.f +
                                 (acc2.s0 + 1.f/256.f * acc2.s1) * sc8[4] +
                                 (acc2.s2 + 1.f/256.f * acc2.s3) * sc8[5] * 1.f/16.f) -
                         dmin * (sumy.s0 * sc8[2] + sumy.s1 * sc8[3] + sumy.s2 * sc8[6] + sumy.s3 * sc8[7]);

            q1 += blk*64;
            sc += blk*6;
            d  += blk;
            dm += blk;
        }

        y4 += BLOCK_STRIDE * QK_K;
    }

    global float * dst_f32 = (global float *) dst_cur;

    for (int row = 0; row < N_DST; ++row) {
        all_sum = sub_group_reduce_add(sumf[row]);
        if (first_row + row < ne01) {
            if (get_sub_group_local_id() == 0) {
                dst_f32[first_row + row] = all_sum;
            }
        }
    }
}

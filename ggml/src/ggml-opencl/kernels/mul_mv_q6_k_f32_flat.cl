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

//------------------------------------------------------------------------------
// kernel_mul_mv_q6_K_f32_flat
//------------------------------------------------------------------------------
#define Q6_K_MASK1 0x03
#define Q6_K_MASK2 0x0C
#define Q6_K_MASK3 0x30
#define Q6_K_MASK4 0xC0

#define QK_K       256

inline float block_q_6_K_dot_y_flat(
    global uchar * blk_ql,
    global uchar * blk_qh,
    global char  * blk_scales,
    global half  * blk_d,
    int ib,
    int ip,
    int is,
    int l0,
    float4 y0,
    float4 y1,
    float4 y2,
    float4 y3
) {
    int q_offset_l =  64*ip + l0;
    int q_offset_h =  32*ip + l0;

    global uchar * q1 = blk_ql     + ib*128 + q_offset_l;
    global uchar * q2 = q1         + QK_K/8;
    global uchar * qh = blk_qh     + ib*64 + q_offset_h;

    // Scales: the four this lane needs are is, is+2, is+4, is+6 with
    // is = 8*ip + l0/16, i.e. all inside the 8-scale half-block at 8*ip. Fetch
    // that half in ONE aligned 8-byte load instead of four scalar byte reads --
    // the inner loop was issuing 8 memory instructions per (row, superblock) to
    // move 18 bytes, five of them for six bytes of scale/d.
    // NB: vload8 at `blk_scales + ib*16 + is` would instead run one byte past the
    // block at is=9, overrunning the last superblock's scales.
    char8 scv = vload8(0, blk_scales + ib*16 + 8*ip);
    int   sj  = l0/16;   // 0 or 1; is == 8*ip + sj

    float dall = blk_d[ib];

    // Vectorized loads: 3 uchar4 weight loads instead of 12 scalar byte reads.
    // q_offset_l/h are 4-aligned, so these are aligned vector loads.
    uchar4 q1v = vload4(0, q1);
    uchar4 q2v = vload4(0, q2);
    uchar4 qhv = vload4(0, qh);

    int4 q1i = convert_int4(q1v);
    int4 q2i = convert_int4(q2v);
    int4 qhi = convert_int4(qhv);

    // Reconstruct the four 6-bit weight groups (low/high nibble of ql OR'd with the
    // matching 2-bit plane of qh), same arithmetic as the scalar version, then dot()
    // against the cached activation lanes.
    float4 w0 = convert_float4((q1i & 0xF) | ((qhi & Q6_K_MASK1) << 4)) - 32.f;
    float4 w1 = convert_float4((q2i & 0xF) | ((qhi & Q6_K_MASK2) << 2)) - 32.f;
    float4 w2 = convert_float4((q1i >> 4)  | ((qhi & Q6_K_MASK3)     )) - 32.f;
    float4 w3 = convert_float4((q2i >> 4)  | ((qhi & Q6_K_MASK4) >> 2)) - 32.f;

    // sj + {0,2,4,6} is the old sc[0]/sc[2]/sc[4]/sc[6] relative to is.
    char4 s4 = (sj == 0) ? (char4)(scv.s0, scv.s2, scv.s4, scv.s6)
                         : (char4)(scv.s1, scv.s3, scv.s5, scv.s7);

    return dall * (dot(y0, w0) * s4.x + dot(y1, w1) * s4.y +
                   dot(y2, w2) * s4.z + dot(y3, w3) * s4.w);
}

#undef N_DST
#undef N_SIMDGROUP
#undef N_SIMDWIDTH

// Rows per subgroup and subgroups per workgroup. Overridable at build time
// (-DQ6K_FLAT_N_DST / -DQ6K_FLAT_NSG, host: GGML_OPENCL_Q6K_FLAT_N_DST/_NSG) so
// the ne1==1 tile can be swept on device -- this kernel carries the whole
// vocab-scale lm_head at decode, where it runs at ~122 of a measured 130.6 GB/s.
#ifdef INTEL_GPU
#ifndef Q6K_FLAT_N_DST
#define Q6K_FLAT_N_DST 4
#endif
#ifndef Q6K_FLAT_NSG
#define Q6K_FLAT_NSG 2
#endif
#define N_DST Q6K_FLAT_N_DST
#define N_SIMDGROUP Q6K_FLAT_NSG
#define N_SIMDWIDTH 16
#elif defined (ADRENO_GPU)
#ifndef Q6K_FLAT_N_DST
#define Q6K_FLAT_N_DST 16
#endif
#ifndef Q6K_FLAT_NSG
#define Q6K_FLAT_NSG 2
#endif
#define N_DST Q6K_FLAT_N_DST
#define N_SIMDGROUP Q6K_FLAT_NSG
#define N_SIMDWIDTH 64
#endif

#define BLOCK_STRIDE (N_SIMDWIDTH/16) // number of blocks each subgroup processes

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_16
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mv_q6_K_f32_flat(
        global uchar * src0_ql,
        global uchar * src0_qh,
        global char  * src0_s,
        global half  * src0_d,
        global float * src1,
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
        int r3
) {
    src1 = (global float*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    int nb = ne00/QK_K;

    int r0 = get_group_id(0);
    int r1 = get_group_id(1);
    int im = get_group_id(2);

    int i12 = im%ne12;
    int i13 = im/ne12;

    int first_row = (N_SIMDGROUP * r0 + get_sub_group_id()) * N_DST;

    ulong offset_src0    = first_row*nb + (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);
    ulong offset_src0_ql = offset_src0 * 128;
    ulong offset_src0_qh = offset_src0 * 64;
    ulong offset_src0_s  = offset_src0 * 16;
    ulong offset_src0_d  = offset_src0;

    global uchar * blk_ql     = (global uchar *) src0_ql + offset_src0_ql;
    global uchar * blk_qh     = (global uchar *) src0_qh + offset_src0_qh;
    global char  * blk_scales = (global char  *) src0_s  + offset_src0_s;
    global half  * blk_d      = (global half  *) src0_d  + offset_src0_d;
    global float * yy         = (global float *) src1    + r1*ne10 + im*ne00*ne1;

    int tid = get_sub_group_local_id()%(N_SIMDWIDTH/BLOCK_STRIDE); // within-super-block part, 0..15
    int ix  = get_sub_group_local_id()/(N_SIMDWIDTH/BLOCK_STRIDE); // super-block selector, 0..BLOCK_STRIDE-1
    int ip  = tid/8;   // first or second half of (super) block (0 or 1)
    int il  = tid%8;   // each half has 8 parts, one per scale
    int n   = 4;       // 4 scales at a time (and 4 sums)
    int l0  = n*il;    // offset into half-block, 0..28
    int is  = 8*ip + l0/16; // 0, 1, 8, 9

    float sumf[N_DST];
    for (int row = 0; row < N_DST; row++) {
        sumf[row] = 0.f;
    }

    for (int ib = ix; ib < nb; ib += BLOCK_STRIDE) {
        global float * y = yy + ib * QK_K + 128*ip + l0;
        float4 y0 = vload4(0, y +  0);
        float4 y1 = vload4(0, y + 32);
        float4 y2 = vload4(0, y + 64);
        float4 y3 = vload4(0, y + 96);

        for (int row = 0; row < N_DST; row++) {
            if (first_row + row < ne01) {
                sumf[row] += block_q_6_K_dot_y_flat(
                    blk_ql + row*nb*128, blk_qh + row*nb*64, blk_scales + row*nb*16, blk_d + row*nb,
                    ib, ip, is, l0, y0, y1, y2, y3);
            }
        }
    }

    for (int row = 0; row < N_DST; row++) {
        float tot = sub_group_reduce_add(sumf[row]);
        if (get_sub_group_local_id() == 0 && first_row + row < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = tot;
        }
    }
}

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

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

#define QK_K 256

// The quant bytes are read through vloadn rather than one uchar at a time.
// GGML_OPENCL_IQ_MV_VEC=0 puts the scalar loads back for an A/B.
#ifndef IQ_MV_VEC
#define IQ_MV_VEC 1
#endif

constant float kvalues_iq4nl[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f,
      1.f,   13.f,  25.f,  38.f,  53.f,  69.f,  89.f, 113.f
};

//------------------------------------------------------------------------------
// block_iq4_xs
//------------------------------------------------------------------------------
// 8 sub blocks of 32 elements, same 16 entry codebook as iq4_nl
typedef struct {
    half   d;
    ushort scales_h;
    uchar  scales_l[QK_K/64];
    uchar  qs[QK_K/2];
} block_iq4_xs;

#undef N_DST
#undef N_SIMDGROUP
#undef N_SIMDWIDTH

#ifdef INTEL_GPU
#define N_DST 4 // number of rows each SIMD group works on
#define N_SIMDGROUP 1 // number of SIMD groups in a thread group
#define N_SIMDWIDTH 16 // SIMD group size
#elif defined (ADRENO_GPU)
#define N_DST 4
#define N_SIMDGROUP 1
#define N_SIMDWIDTH 64
#endif

#undef  BLOCK_STRIDE
// 8 threads cover one super block, one sub block each
#define BLOCK_STRIDE (N_SIMDWIDTH/8)

// One packed uint holds 4 quant bytes = 8 weights. yl points at the activation
// for the first low nibble; the matching high nibbles are 16 slots further on.
inline float iq4xs_word(uint w, private float * yl) {
    float acc = 0.f;
    acc += yl[ 0] * kvalues_iq4nl[(w      ) & 0xf];
    acc += yl[16] * kvalues_iq4nl[(w >>  4) & 0xf];
    acc += yl[ 1] * kvalues_iq4nl[(w >>  8) & 0xf];
    acc += yl[17] * kvalues_iq4nl[(w >> 12) & 0xf];
    acc += yl[ 2] * kvalues_iq4nl[(w >> 16) & 0xf];
    acc += yl[18] * kvalues_iq4nl[(w >> 20) & 0xf];
    acc += yl[ 3] * kvalues_iq4nl[(w >> 24) & 0xf];
    acc += yl[19] * kvalues_iq4nl[(w >> 28) & 0xf];
    return acc;
}

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_16
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mv_iq4_xs_f32(
        global char * src0,
        int offset0,
        global char * src1,
        int offset1,
        global char * dst,
        int offsetd,
        int ne00,
        int ne01,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne12,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    int ix = get_sub_group_local_id()/8;  // super block index
    int it = get_sub_group_local_id()%8;  // sub block inside the super block

    int nb = ne00/QK_K;

    int r0 = get_group_id(0);
    int r1 = get_group_id(1);
    int im = get_group_id(2);
    int first_row = (r0 * N_SIMDGROUP + get_sub_group_id()) * N_DST;

    int i12 = im%ne12;
    int i13 = im/ne12;

    int offset_src0 = first_row*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;
    int offset_src1 =        r1*nb11 + (i12   )*nb12 + (i13   )*nb13;

    global block_iq4_xs * x = (global block_iq4_xs *) (src0 + offset_src0);
    global float        * y = (global float        *) (src1 + offset_src1);

    float yl[32];
    float sumf[N_DST] = {0.f};
    float all_sum;

    global float * y4 = y + ix * QK_K + 32 * it;

    for (int ib = ix; ib < nb; ib += BLOCK_STRIDE) {
#if IQ_MV_VEC
        // y4 starts at a multiple of 32 floats, so float4 alignment holds
        for (int i = 0; i < 8; ++i) {
            float4 yv = ((global float4 *)y4)[i];
            yl[4*i+0] = yv.s0;
            yl[4*i+1] = yv.s1;
            yl[4*i+2] = yv.s2;
            yl[4*i+3] = yv.s3;
        }
#else
        for (int i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }
#endif

        global char * xrow = (global char *)(x + ib);

        // only ne01 output rows exist; reading past them can pull in an inf scale
        for (int row = 0; row < N_DST && first_row + row < ne01; row++) {
            global block_iq4_xs * xb = (global block_iq4_xs *)(xrow + row*nb01);

            int ls = ((xb->scales_l[it/2] >> (4*(it%2))) & 0xf) | (((xb->scales_h >> (2*it)) & 3) << 4);
            float dl = (float)xb->d * (float)(ls - 32);

            float acc = 0.f;
#if IQ_MV_VEC
            // qs sits 8 bytes into a 136 byte block, so uint2 alignment holds
            // but uint4 does not.
            global uint2 * q2 = (global uint2 *)(xb->qs + 16*it);
            uint2 wa = q2[0];
            uint2 wb = q2[1];
            acc += iq4xs_word(wa.s0, yl +  0);
            acc += iq4xs_word(wa.s1, yl +  4);
            acc += iq4xs_word(wb.s0, yl +  8);
            acc += iq4xs_word(wb.s1, yl + 12);
#else
            global uchar * q = xb->qs + 16*it;
            for (int j = 0; j < 16; ++j) {
                acc += yl[j]    * kvalues_iq4nl[q[j] & 0xf];
                acc += yl[j+16] * kvalues_iq4nl[q[j] >>  4];
            }
#endif

            sumf[row] += dl * acc;
        }

        y4 += BLOCK_STRIDE * QK_K;
    }

    global float * dst_f32 = (global float *) dst + im*ne0*ne1 + r1*ne0;

    for (int row = 0; row < N_DST; ++row) {
        all_sum = sub_group_reduce_add(sumf[row]);
        if (first_row + row < ne01) {
            if (get_sub_group_local_id() == 0) {
                dst_f32[first_row + row] = all_sum;
            }
        }
    }
}

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

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_soft_max_4_f16(
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global char * src2,
        ulong offset2,
        global char * dst,
        ulong offsetd,
        int ne00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne12,
        int ne13,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        ulong nb1,
        ulong nb2,
        ulong nb3,
        float scale,
        float max_bias,
        float m0,
        float m1,
        int n_head_log2
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    src2 = src2 + offset2;
    dst  = dst  + offsetd;

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    int i13 = i03%ne13;
    int i12 = i02%ne12;
    int i11 = i01;

    global float4 * psrc4 = (global float4 *)(src0 + i01*nb01 + i02*nb02 + i03*nb03);
    global half4  * pmask = src1 != src0 ? (global half4 *)(src1 + i11*nb11 + i12*nb12 + i13*nb13) : 0;
    global float  * psrc2 = src2 != src0 ? (global float *)(src2) : 0;
    global float4 * pdst4 = (global float4 *)(dst  + i01*nb1 + i02*nb2 + i03*nb3);

    float slope = 1.0f;

    // ALiBi
    if (max_bias > 0.0f) {
        int h = i02;

        float base = h < n_head_log2 ? m0 : m1;
        int   exp  = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

        slope = pow(base, exp);
    }

    // parallel max
    float4 lmax4 = psrc2 ? psrc2[i02] : -INFINITY;
    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
        lmax4 = fmax(lmax4, psrc4[i00]*scale + slope*(pmask ? convert_float4(pmask[i00]) : 0.0f));
    }
    float lmax = fmax(fmax(lmax4.s0, lmax4.s1), fmax(lmax4.s2, lmax4.s3));

    const float max = sub_group_reduce_max(lmax);

    // parallel sum
    float4 lsum4 = 0.0f;
    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
        const float4 exp_psrc4 = exp((psrc4[i00]*scale + slope*(pmask ? convert_float4(pmask[i00]) : 0.0f)) - max);
        lsum4 += exp_psrc4;
        pdst4[i00] = exp_psrc4;
    }
    float lsum = lsum4.s0 + lsum4.s1 + lsum4.s2 + lsum4.s3;

    float sum = sub_group_reduce_add(lsum);

    if (psrc2) {
        sum += exp(psrc2[i02] - max);
    }

    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
        pdst4[i00] /= sum;
    }
}

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_soft_max_4_f16_nonorm(
        global float * psums,
        ulong offset_sums,
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global char * src2,
        ulong offset2,
        global char * dst,
        ulong offsetd,
        int ne00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne12,
        int ne13,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        ulong nb1,
        ulong nb2,
        ulong nb3,
        float scale,
        float max_bias,
        float m0,
        float m1,
        int n_head_log2
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    src2 = src2 + offset2;
    dst  = dst  + offsetd;

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    int i13 = i03%ne13;
    int i12 = i02%ne12;
    int i11 = i01;

    global float4 * psrc4 = (global float4 *)(src0 + i01*nb01 + i02*nb02 + i03*nb03);
    global half4  * pmask = src1 != src0 ? (global half4 *)(src1 + i11*nb11 + i12*nb12 + i13*nb13) : 0;
    global float  * psrc2 = src2 != src0 ? (global float *)(src2) : 0;
    global float4 * pdst4 = (global float4 *)(dst  + i01*nb1 + i02*nb2 + i03*nb3);

    float slope = 1.0f;

    // ALiBi
    if (max_bias > 0.0f) {
        int h = i02;

        float base = h < n_head_log2 ? m0 : m1;
        int   exp  = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

        slope = pow(base, exp);
    }

    // parallel max
    float4 lmax4 = psrc2 ? psrc2[i02] : -INFINITY;
    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
        lmax4 = fmax(lmax4, psrc4[i00]*scale + slope*(pmask ? convert_float4(pmask[i00]) : 0.0f));
    }
    float lmax = fmax(fmax(lmax4.s0, lmax4.s1), fmax(lmax4.s2, lmax4.s3));

    const float max = sub_group_reduce_max(lmax);

    // parallel sum
    float4 lsum4 = 0.0f;
    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
        const float4 exp_psrc4 = exp((psrc4[i00]*scale + slope*(pmask ? convert_float4(pmask[i00]) : 0.0f)) - max);
        lsum4 += exp_psrc4;
        pdst4[i00] = exp_psrc4;
    }
    float lsum = lsum4.s0 + lsum4.s1 + lsum4.s2 + lsum4.s3;

    float sum = sub_group_reduce_add(lsum);

    if (psrc2) {
        sum += exp(psrc2[i02] - max);
    }

    // The normalize pass is deferred: this kernel leaves exp(x - max) in dst and publishes the
    // row sum, and the consumer folds 1/sum into its own (far smaller) output. That removes a
    // whole read+write of the score matrix, which is the dominant traffic here.
    if (get_local_id(0) == 0) {
        psums = (global float *)((global char *)psums + offset_sums);
        psums[i03*get_num_groups(1)*get_num_groups(0) + i02*get_num_groups(0) + i01] = sum;
    }
}


// Applies the deferred softmax normalisation to the KQV result: out[:, n, h] /= sums[h*ne1 + n].
// The KQV output is dv x n_q x n_head, which is orders of magnitude smaller than the score
// matrix, so doing the division here instead of over the scores is nearly free.
kernel void kernel_fa_scale_rows_f32(
        global float * dst,
        ulong offsetd,
        global float * sums,
        ulong offset_sums,
        int dv,
        int ne1
) {
    dst  = (global float *)((global char *)dst  + offsetd);
    sums = (global float *)((global char *)sums + offset_sums);

    const int n = get_group_id(0);
    const int h = get_group_id(1);

    const float s = sums[h*ne1 + n];
    const float r = s > 0.0f ? 1.0f/s : 0.0f;

    global float * row = dst + (size_t)h*ne1*dv + (size_t)n*dv;
    for (int i = get_local_id(0); i < dv; i += get_local_size(0)) {
        row[i] *= r;
    }
}

// Deferred-normalisation softmax that emits P as u8 + per-32-block scale instead of f32.
//
// P = exp(x - max) is in (0,1] with the row maximum exactly 1.0 by construction, so a single
// row-wide scale looks free - but it is not good enough: measured against real score
// distributions a global 1/255 costs up to 3.79% on the KQV result, where a per-32-block scale
// stays under 0.5%. The block layout matches kernel_moe_reorder_quant_a_q8_1, which is what the
// dp4a GEMM already consumes.
//
// This also cuts the score matrix from 4 bytes per element to 1 + a half per 32, so it is a
// traffic win as well as the operand an int8 KQV needs.
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_soft_max_4_f16_q8(
        global uchar * qp,
        ulong offset_qp,
        global half * dp,
        ulong offset_dp,
        global float * psums,
        ulong offset_sums,
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global char * src2,
        ulong offset2,
        int ne00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne12,
        int ne13,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        float scale,
        float max_bias,
        float m0,
        float m1,
        int n_head_log2
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    src2 = src2 + offset2;

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    int i13 = i03%ne13;
    int i12 = i02%ne12;
    int i11 = i01;

    global float4 * psrc4 = (global float4 *)(src0 + i01*nb01 + i02*nb02 + i03*nb03);
    global half4  * pmask = src1 != src0 ? (global half4 *)(src1 + i11*nb11 + i12*nb12 + i13*nb13) : 0;
    global float  * psrc2 = src2 != src0 ? (global float *)(src2) : 0;

    float slope = 1.0f;

    if (max_bias > 0.0f) {
        int h = i02;

        float base = h < n_head_log2 ? m0 : m1;
        int   exp  = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

        slope = pow(base, exp);
    }

    // pass 1: row max, strided so the reads coalesce across lanes
    float4 lmax4 = psrc2 ? psrc2[i02] : -INFINITY;
    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
        lmax4 = fmax(lmax4, psrc4[i00]*scale + slope*(pmask ? convert_float4(pmask[i00]) : 0.0f));
    }
    float lmax = fmax(fmax(lmax4.s0, lmax4.s1), fmax(lmax4.s2, lmax4.s3));

    const float max = sub_group_reduce_max(lmax);

    // pass 2: one 32-block per lane, so a lane owns the whole block it has to scale.
    // A lane reads 32 contiguous floats rather than a strided float4 - same bytes, and the
    // block reduction stays in registers.
    const int nblk = ne00 / 32;
    const uint row  = (uint)(i03*get_num_groups(1)*get_num_groups(0) + i02*get_num_groups(0) + i01);

    global uchar * qrow = (global uchar *)((global char *)qp + offset_qp) + (size_t)row*ne00;
    global half  * drow = (global half  *)((global char *)dp + offset_dp) + (size_t)row*nblk;

    float lsum = 0.0f;
    for (int b = get_local_id(0); b < nblk; b += get_local_size(0)) {
        float v[32];
        float amax = 0.0f;

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float4 e = exp((psrc4[b*8 + i]*scale + slope*(pmask ? convert_float4(pmask[b*8 + i]) : 0.0f)) - max);
            v[i*4 + 0] = e.s0; v[i*4 + 1] = e.s1; v[i*4 + 2] = e.s2; v[i*4 + 3] = e.s3;
            amax = fmax(amax, fmax(fmax(e.s0, e.s1), fmax(e.s2, e.s3)));
        }

        // P >= 0, so quantise unsigned: 255 levels instead of 127
        const float d  = amax / 255.0f;
        const float id = amax > 0.0f ? 255.0f / amax : 0.0f;

        drow[b] = (half)d;

        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            const int q = (int)rint(v[i] * id);
            qrow[b*32 + i] = (uchar)clamp(q, 0, 255);
            lsum += v[i];
        }
    }

    float sum = sub_group_reduce_add(lsum);

    if (psrc2) {
        sum += exp(psrc2[i02] - max);
    }

    if (get_local_id(0) == 0) {
        psums = (global float *)((global char *)psums + offset_sums);
        psums[row] = sum;
    }
}

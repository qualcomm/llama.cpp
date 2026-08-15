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

#if defined(SOFTMAX_ONLINE)
    // Online (running-max) statistics: one read pass over src+mask produces both
    // the max and the sum, then a single pass writes the normalised result. The
    // three-pass form below costs an extra read AND write of dst -- 8 of 24
    // bytes per element, i.e. a third of this kernel's traffic, which matters
    // because it is pure bandwidth: on Nemotron-3.5 at d16384 the decomposed
    // attention hands it a KQ matrix of n_kv x n_q x n_head and the kernel was
    // 9.6% of prefill GPU time.
    //
    // Per lane keep (m, s) with s = sum(exp(v - m)); merging a new value or
    // another lane's pair rescales the smaller sum by exp(dm). Same recurrence
    // flash attention uses, so the numerics are the established ones rather
    // than a new approximation.
    float lmax = psrc2 ? psrc2[i02] : -INFINITY;
    float lsum = 0.0f;
    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
        const float4 v = psrc4[i00]*scale + slope*(pmask ? convert_float4(pmask[i00]) : 0.0f);
        const float vmax = fmax(fmax(v.s0, v.s1), fmax(v.s2, v.s3));
        const float m_new = fmax(lmax, vmax);
        // exp(-INFINITY - -INFINITY) is NaN; the first iteration always takes
        // the m_new == vmax branch with lsum == 0, so guard the rescale.
        const float rescale = (lsum == 0.0f) ? 0.0f : exp(lmax - m_new);
        const float4 e = exp(v - m_new);
        lsum = lsum*rescale + ((e.s0 + e.s1) + (e.s2 + e.s3));
        lmax = m_new;
    }

    // Merge the per-lane (m, s) pairs: reduce the max, then rescale each lane's
    // sum into that common max before summing.
    const float max = sub_group_reduce_max(lmax);
    float sum = sub_group_reduce_add(lsum * exp(lmax - max));

    if (psrc2) {
        sum += exp(psrc2[i02] - max);
    }

    const float inv_sum = 1.0f / sum;
    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
        pdst4[i00] = exp((psrc4[i00]*scale + slope*(pmask ? convert_float4(pmask[i00]) : 0.0f)) - max) * inv_sum;
    }
#else
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

#if defined(SOFTMAX_RECIP)
    // Attribution arm: keep the 3-pass walk but replace the per-element
    // division with one reciprocal, to separate "one pass fewer" from
    // "no per-element divide" in the online kernel's win.
    const float inv_sum_r = 1.0f / sum;
    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
        pdst4[i00] *= inv_sum_r;
    }
#else
    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
        pdst4[i00] /= sum;
    }
#endif
#endif
}

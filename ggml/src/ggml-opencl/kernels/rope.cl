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
// kernel_rope
//------------------------------------------------------------------------------
float rope_yarn_ramp(float low, float high, int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
float2 rope_yarn(
    float theta_extrap, float freq_scale, float2 corr_dims, int i0, float ext_factor, float mscale
) {
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims.s0, corr_dims.s1, i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * log(1.0f / freq_scale);
    }
    return (float2)(cos(theta) * mscale, sin(theta) * mscale);
}

// Apparently solving `n_rot = 2pi * x * base^((2 * max_pos_emb) / n_dims)` for x, we get
// `corr_fac(n_rot) = n_dims * log(max_pos_emb / (n_rot * 2pi)) / (2 * log(base))`
float rope_yarn_corr_factor(int n_dims, int n_ctx_orig, float n_rot, float base) {
    return n_dims * log(n_ctx_orig / (n_rot * 2 * M_PI_F)) / (2 * log(base));
}

float2 rope_yarn_corr_dims(
    int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow
) {
    // start and end correction dims
    return (float2)(
        max(0.0f,         floor(rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_fast, freq_base))),
        min(n_dims - 1.0f, ceil(rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_slow, freq_base)))
    );
}

kernel void kernel_rope_norm_f32(
        global void * src0,
        ulong offset0,
        global int * src1,
        ulong offset1,
        global float * src2,
        ulong offset2,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3,
        int n_past,
        int n_dims,
        int n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow
) {
    src0 = (global void*)((global char*)src0 + offset0);
    src1 = (global int*)((global char*)src1 + offset1);
    src2 = (global float*)((global char*)src2 + offset2);
    dst = (global float*)((global char*)dst + offsetd);

    int i3 = get_group_id(2);
    int i2 = get_group_id(1);
    int i1 = get_group_id(0);

    float2 corr_dims = rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow);

    global int * pos = src1;

    float theta_base = (float) pos[i2];
    float inv_ndims = -1.f/n_dims;

    for (int i0 = 2*get_local_id(0); i0 < ne0; i0 += 2*get_local_size(0)) {
        if (i0 < n_dims) {
            int ic = i0/2;

            float theta = theta_base * pow(freq_base, inv_ndims*i0);

            float freq_factor = src2 != src0 ? src2[ic] : 1.0f;

            float2 cos_sin_theta = rope_yarn(theta/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor);

            global float * src       = (global float *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
            global float * dst_data  = (global float *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

            float x0 = src[0];
            float x1 = src[1];

            dst_data[0] = x0*cos_sin_theta.s0 - x1*cos_sin_theta.s1;
            dst_data[1] = x0*cos_sin_theta.s1 + x1*cos_sin_theta.s0;
        } else {
            global float * src      = (global float *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
            global float * dst_data = (global float *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

            dst_data[0] = src[0];
            dst_data[1] = src[1];
        }
    }
}

kernel void kernel_rope_norm_f16(
        global void * src0,
        ulong offset0,
        global int * src1,
        ulong offset1,
        global float * src2,
        ulong offset2,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3,
        int n_past,
        int n_dims,
        int n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow
) {
    src0 = (global void*)((global char*)src0 + offset0);
    src1 = (global int*)((global char*)src1 + offset1);
    src2 = (global float*)((global char*)src2 + offset2);
    dst = (global float*)((global char*)dst + offsetd);

    int i3 = get_group_id(2);
    int i2 = get_group_id(1);
    int i1 = get_group_id(0);

    float2 corr_dims = rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow);

    global int * pos = src1;

    float theta_base = (float) pos[i2];
    float inv_ndims = -1.f/n_dims;

    for (int i0 = 2*get_local_id(0); i0 < ne0; i0 += 2*get_local_size(0)) {
        if (i0 < n_dims) {
            int ic = i0/2;

            float theta = theta_base * pow(freq_base, inv_ndims*i0);

            float freq_factor = src2 != src0 ? src2[ic] : 1.0f;

            float2 cos_sin_theta = rope_yarn(theta/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor);

            global half * src       = (global half *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
            global half * dst_data  = (global half *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

            float x0 = src[0];
            float x1 = src[1];

            dst_data[0] = x0*cos_sin_theta.s0 - x1*cos_sin_theta.s1;
            dst_data[1] = x0*cos_sin_theta.s1 + x1*cos_sin_theta.s0;
        } else {
            global half * src      = (global half *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
            global half * dst_data = (global half *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

            dst_data[0] = src[0];
            dst_data[1] = src[1];
        }
    }
}

kernel void kernel_rope_neox_f32(
        global void * src0,
        ulong offset0,
        global int * src1,
        ulong offset1,
        global float * src2,
        ulong offset2,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3,
        int n_past,
        int n_dims,
        int n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow
) {
    src0 = (global void*)((global char*)src0 + offset0);
    src1 = (global int*)((global char*)src1 + offset1);
    src2 = (global float*)((global char*)src2 + offset2);
    dst = (global float*)((global char*)dst + offsetd);

    int i3 = get_group_id(2);
    int i2 = get_group_id(1);
    int i1 = get_group_id(0);

    float2 corr_dims = rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow);

    global int * pos = src1;

    float theta_base = (float) pos[i2];
    float inv_ndims = -1.f/n_dims;

    for (int i0 = 2*get_local_id(0); i0 < ne0; i0 += 2*get_local_size(0)) {
        if (i0 < n_dims) {
            int ic = i0/2;

            const float theta = theta_base * pow(freq_base, inv_ndims*i0);

            const float freq_factor = src2 != src0 ? src2[ic] : 1.0f;

            float2 cos_sin_theta = rope_yarn(theta/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor);

            global float * src      = (global float *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);
            global float * dst_data = (global float *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + ic*nb0);

            const float x0 = src[0];
            const float x1 = src[n_dims/2];

            dst_data[0]        = x0*cos_sin_theta.s0 - x1*cos_sin_theta.s1;
            dst_data[n_dims/2] = x0*cos_sin_theta.s1 + x1*cos_sin_theta.s0;
        } else {
            global float * const src = (global float *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
            global float * dst_data  = (global float *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

            dst_data[0] = src[0];
            dst_data[1] = src[1];
        }
    }
}

kernel void kernel_rope_neox_f16(
        global void * src0,
        ulong offset0,
        global int * src1,
        ulong offset1,
        global float * src2,
        ulong offset2,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3,
        int n_past,
        int n_dims,
        int n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow
) {
    src0 = (global void*)((global char*)src0 + offset0);
    src1 = (global int*)((global char*)src1 + offset1);
    src2 = (global float*)((global char*)src2 + offset2);
    dst = (global float*)((global char*)dst + offsetd);

    int i3 = get_group_id(2);
    int i2 = get_group_id(1);
    int i1 = get_group_id(0);

    float2 corr_dims = rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow);

    global int * pos = src1;

    float theta_base = (float) pos[i2];
    float inv_ndims = -1.f/n_dims;

    for (int i0 = 2*get_local_id(0); i0 < ne0; i0 += 2*get_local_size(0)) {
        if (i0 < n_dims) {
            int ic = i0/2;

            const float theta = theta_base * pow(freq_base, inv_ndims*i0);

            const float freq_factor = src2 != src0 ? src2[ic] : 1.0f;

            float2 cos_sin_theta = rope_yarn(theta/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor);

            global half * src       = (global half *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);
            global half * dst_data  = (global half *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + ic*nb0);

            const float x0 = src[0];
            const float x1 = src[n_dims/2];

            dst_data[0]        = x0*cos_sin_theta.s0 - x1*cos_sin_theta.s1;
            dst_data[n_dims/2] = x0*cos_sin_theta.s1 + x1*cos_sin_theta.s0;
        } else {
            global half * const src = (global half *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
            global half * dst_data  = (global half *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

            dst_data[0] = src[0];
            dst_data[1] = src[1];
        }
    }
}

kernel void kernel_rope_multi_f32(
        global void * src0,
        ulong offset0,
        global int * src1,
        ulong offset1,
        global float * src2,
        ulong offset2,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3,
        int n_past,
        int n_dims,
        int n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        int4 sections,
        int  is_imrope
) {
    src0 = (global void*)((global char*)src0 + offset0);
    src1 = (global int*)((global char*)src1 + offset1);
    src2 = (global float*)((global char*)src2 + offset2);
    dst = (global float*)((global char*)dst + offsetd);

    int i3 = get_group_id(2);
    int i2 = get_group_id(1);
    int i1 = get_group_id(0);

    float2 corr_dims = rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow);

    global int * pos = src1;

    const int sect_dims = sections.s0 + sections.s1 + sections.s2 + sections.s3;
    const int sec_w = sections.s1 + sections.s0;

    float inv_ndims = -1.f/n_dims;

    for (int i0 = 2*get_local_id(0); i0 < ne0; i0 += 2*get_local_size(0)) {
        if (i0 < n_dims) {
            int ic = i0/2;

            const int sector = (i0 / 2) % sect_dims;
            float theta_base = 0.0f;

            if (is_imrope) {
                if (sector % 3 == 1 && sector < 3 * sections.s1) { // h
                    theta_base = (float) pos[i2 + ne02 * 1];
                } else if (sector % 3 == 2 && sector < 3 * sections.s2) { // w
                    theta_base = (float) pos[i2 + ne02 * 2];
                } else if (sector % 3 == 0 && sector < 3 * sections.s0) { // t
                    theta_base = (float) pos[i2 + ne02 * 0];
                } else { // e
                    theta_base = (float) pos[i2 + ne02 * 3];
                }
            } else {
                if (sector < sections.s0) {
                    theta_base = pos[i2];
                }
                else if (sector >= sections.s0 && sector < sec_w) {
                    theta_base = pos[i2 + ne2 * 1];
                }
                else if (sector >= sec_w && sector < sec_w + sections.s2) {
                    theta_base = pos[i2 + ne2 * 2];
                }
                else if (sector >= sec_w + sections.s2) {
                    theta_base = pos[i2 + ne2 * 3];
                }
            }

            const float theta = theta_base * pow(freq_base, inv_ndims*i0);

            const float freq_factor = src2 != src0 ? src2[ic] : 1.0f;

            float2 cos_sin_theta = rope_yarn(theta/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor);

            global float * src      = (global float *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);
            global float * dst_data = (global float *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + ic*nb0);

            const float x0 = src[0];
            const float x1 = src[n_dims/2];

            dst_data[0]        = x0*cos_sin_theta.s0 - x1*cos_sin_theta.s1;
            dst_data[n_dims/2] = x0*cos_sin_theta.s1 + x1*cos_sin_theta.s0;
        } else {
            global float * const src = (global float *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
            global float * dst_data  = (global float *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

            dst_data[0] = src[0];
            dst_data[1] = src[1];
        }
    }
}

kernel void kernel_rope_multi_f16(
        global void * src0,
        ulong offset0,
        global int * src1,
        ulong offset1,
        global float * src2,
        ulong offset2,
        global half * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3,
        int n_past,
        int n_dims,
        int n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        int4 sections,
        int  is_imrope
) {
    src0 = (global void*)((global char*)src0 + offset0);
    src1 = (global int*)((global char*)src1 + offset1);
    src2 = (global float*)((global char*)src2 + offset2);
    dst = (global float*)((global char*)dst + offsetd);

    int i3 = get_group_id(2);
    int i2 = get_group_id(1);
    int i1 = get_group_id(0);

    float2 corr_dims = rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow);

    global int * pos = src1;

    const int sect_dims = sections.s0 + sections.s1 + sections.s2 + sections.s3;
    const int sec_w = sections.s1 + sections.s0;

    float inv_ndims = -1.f/n_dims;

    for (int i0 = 2*get_local_id(0); i0 < ne0; i0 += 2*get_local_size(0)) {
        if (i0 < n_dims) {
            int ic = i0/2;

            const int sector = (i0 / 2) % sect_dims;
            float theta_base = 0.0f;

            if (is_imrope) {
                if (sector % 3 == 1 && sector < 3 * sections.s1) { // h
                    theta_base = (float) pos[i2 + ne02 * 1];
                } else if (sector % 3 == 2 && sector < 3 * sections.s2) { // w
                    theta_base = (float) pos[i2 + ne02 * 2];
                } else if (sector % 3 == 0 && sector < 3 * sections.s0) { // t
                    theta_base = (float) pos[i2 + ne02 * 0];
                } else { // e
                    theta_base = (float) pos[i2 + ne02 * 3];
                }
            } else {
                if (sector < sections.s0) {
                    theta_base = pos[i2];
                }
                else if (sector >= sections.s0 && sector < sec_w) {
                    theta_base = pos[i2 + ne2 * 1];
                }
                else if (sector >= sec_w && sector < sec_w + sections.s2) {
                    theta_base = pos[i2 + ne2 * 2];
                }
                else if (sector >= sec_w + sections.s2) {
                    theta_base = pos[i2 + ne2 * 3];
                }
            }

            const float theta = theta_base * pow(freq_base, inv_ndims*i0);

            const float freq_factor = src2 != src0 ? src2[ic] : 1.0f;

            float2 cos_sin_theta = rope_yarn(theta/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor);

            global half * src      = (global half *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);
            global half * dst_data = (global half *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + ic*nb0);

            const float x0 = src[0];
            const float x1 = src[n_dims/2];

            dst_data[0]        = x0*cos_sin_theta.s0 - x1*cos_sin_theta.s1;
            dst_data[n_dims/2] = x0*cos_sin_theta.s1 + x1*cos_sin_theta.s0;
        } else {
            global half * const src = (global half *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
            global half * dst_data  = (global half *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

            dst_data[0] = src[0];
            dst_data[1] = src[1];
        }
    }
}

kernel void kernel_rope_vision_f32(
        global void * src0,
        ulong offset0,
        global int * src1,
        ulong offset1,
        global float * src2,
        ulong offset2,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3,
        int n_past,
        int n_dims,
        int n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        int4 sections
) {
    src0 = (global void*)((global char*)src0 + offset0);
    src1 = (global int*)((global char*)src1 + offset1);
    src2 = (global float*)((global char*)src2 + offset2);
    dst = (global float*)((global char*)dst + offsetd);

    int i3 = get_group_id(2);
    int i2 = get_group_id(1);
    int i1 = get_group_id(0);

    float2 corr_dims = rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow);

    global int * pos = src1;

    const int sect_dims = sections.s0 + sections.s1;
    const int sec_w = sections.s1 + sections.s0;

    float inv_ndims = -1.f/n_dims;

    for (int i0 = 2*get_local_id(0); i0 < ne0; i0 += 2*get_local_size(0)) {
        int ic = i0/2;

        const int sector = (i0/2) % sect_dims;
        float theta_base = 0.0f;

        if (sector < sections.s0) {
            const int p = sector;
            theta_base = pos[i2] * pow(freq_base, inv_ndims*2.0f*p);
        } else if (sector >= sections.s0 && sector < sec_w) {
            const int p = sector - sections.s0;
            theta_base = pos[i2 + ne2] * pow(freq_base, inv_ndims*2.0f*p);
        }

        const float freq_factor = src2 != src0 ? src2[ic] : 1.0f;

        float2 cos_sin_theta = rope_yarn(theta_base/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor);

        global float * src      = (global float *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);
        global float * dst_data = (global float *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + ic*nb0);

        const float x0 = src[0];
        const float x1 = src[n_dims];

        dst_data[0]      = x0*cos_sin_theta.s0 - x1*cos_sin_theta.s1;
        dst_data[n_dims] = x0*cos_sin_theta.s1 + x1*cos_sin_theta.s0;
    }
}

kernel void kernel_rope_vision_f16(
        global void * src0,
        ulong offset0,
        global int * src1,
        ulong offset1,
        global float * src2,
        ulong offset2,
        global half * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3,
        int n_past,
        int n_dims,
        int n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        int4 sections
) {
    src0 = (global void*)((global char*)src0 + offset0);
    src1 = (global int*)((global char*)src1 + offset1);
    src2 = (global float*)((global char*)src2 + offset2);
    dst = (global float*)((global char*)dst + offsetd);

    int i3 = get_group_id(2);
    int i2 = get_group_id(1);
    int i1 = get_group_id(0);

    float2 corr_dims = rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow);

    global int * pos = src1;

    const int sect_dims = sections.s0 + sections.s1;
    const int sec_w = sections.s1 + sections.s0;

    float inv_ndims = -1.f/n_dims;

    for (int i0 = 2*get_local_id(0); i0 < ne0; i0 += 2*get_local_size(0)) {
        int ic = i0/2;

        const int sector = (i0/2) % sect_dims;
        float theta_base = 0.0f;

        if (sector < sections.s0) {
            const int p = sector;
            theta_base = pos[i2] * pow(freq_base, inv_ndims*2.0f*p);
        } else if (sector >= sections.s0 && sector < sec_w) {
            const int p = sector - sections.s0;
            theta_base = pos[i2 + ne2] * pow(freq_base, inv_ndims*2.0f*p);
        }

        const float freq_factor = src2 != src0 ? src2[ic] : 1.0f;

        float2 cos_sin_theta = rope_yarn(theta_base/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor);

        global half * src      = (global half *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);
        global half * dst_data = (global half *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + ic*nb0);

        const float x0 = src[0];
        const float x1 = src[n_dims];

        dst_data[0]      = x0*cos_sin_theta.s0 - x1*cos_sin_theta.s1;
        dst_data[n_dims] = x0*cos_sin_theta.s1 + x1*cos_sin_theta.s0;
    }
}

// Fused ROPE_NORM + VIEW + SET_ROWS for f32 source / f16 destination.
// NORM-style rope pairs are adjacent (dst[0], dst[1]) vs NEOX split halves.
// Mirrors the CUDA `rope_norm_cuda<float, half>(..., row_indices, set_rows_stride)` path.
kernel void kernel_rope_norm_f32_set_rows_f16(
        global void * src0,
        ulong offset0,
        global int * src1,
        ulong offset1,
        global float * src2,
        ulong offset2,
        global void * dst,
        ulong offsetd,
        global long * row_indices,
        ulong offset_row_indices,
        int set_rows_stride,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3,
        int n_past,
        int n_dims,
        int n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow
) {
    src0        = (global void  *)((global char *)src0        + offset0);
    src1        = (global int   *)((global char *)src1        + offset1);
    src2        = (global float *)((global char *)src2        + offset2);
    dst         = (global void  *)((global char *)dst         + offsetd);
    row_indices = (global long  *)((global char *)row_indices + offset_row_indices);

    int i3 = get_group_id(2);
    int i2 = get_group_id(1);
    int i1 = get_group_id(0);

    float2 corr_dims = rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow);

    global int * pos = src1;

    float theta_base = (float) pos[i2];
    float inv_ndims = -1.f/n_dims;

    long row_idx = row_indices[i2];
    global half * dst_row = (global half *)dst + row_idx * (long)set_rows_stride + (long)i1 * (long)ne00;

    for (int i0 = 2*get_local_id(0); i0 < ne0; i0 += 2*get_local_size(0)) {
        if (i0 < n_dims) {
            int ic = i0/2;

            float theta = theta_base * pow(freq_base, inv_ndims*i0);
            float freq_factor = src2 != src0 ? src2[ic] : 1.0f;
            float2 cos_sin_theta = rope_yarn(theta/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor);

            global float * src = (global float *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);

            float x0 = src[0];
            float x1 = src[1];

            dst_row[i0]     = (half)(x0*cos_sin_theta.s0 - x1*cos_sin_theta.s1);
            dst_row[i0 + 1] = (half)(x0*cos_sin_theta.s1 + x1*cos_sin_theta.s0);
        } else {
            global float * src = (global float *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
            dst_row[i0]     = (half)src[0];
            dst_row[i0 + 1] = (half)src[1];
        }
    }
}

// Fused ROPE_NEOX + VIEW + SET_ROWS for f32 source / f16 destination.
// Mirrors the CUDA `rope_neox_cuda<float, half>(..., row_indices, set_rows_stride)` path.
// The rope output is redirected into the K-cache at the row indicated by row_indices[i2].
kernel void kernel_rope_neox_f32_set_rows_f16(
        global void * src0,
        ulong offset0,
        global int * src1,
        ulong offset1,
        global float * src2,
        ulong offset2,
        global void * dst,
        ulong offsetd,
        global long * row_indices,
        ulong offset_row_indices,
        int set_rows_stride,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3,
        int n_past,
        int n_dims,
        int n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow
) {
    src0        = (global void  *)((global char *)src0        + offset0);
    src1        = (global int   *)((global char *)src1        + offset1);
    src2        = (global float *)((global char *)src2        + offset2);
    dst         = (global void  *)((global char *)dst         + offsetd);
    row_indices = (global long  *)((global char *)row_indices + offset_row_indices);

    int i3 = get_group_id(2);
    int i2 = get_group_id(1);
    int i1 = get_group_id(0);

    float2 corr_dims = rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow);

    global int * pos = src1;

    float theta_base = (float) pos[i2];
    float inv_ndims = -1.f/n_dims;

    // Token → K-cache row redirection. Per-head stride inside the cache row is ne00 (head_dim).
    long row_idx = row_indices[i2];
    global half * dst_row = (global half *)dst + row_idx * (long)set_rows_stride + (long)i1 * (long)ne00;

    for (int i0 = 2*get_local_id(0); i0 < ne0; i0 += 2*get_local_size(0)) {
        if (i0 < n_dims) {
            int ic = i0/2;

            const float theta = theta_base * pow(freq_base, inv_ndims*i0);
            const float freq_factor = src2 != src0 ? src2[ic] : 1.0f;
            float2 cos_sin_theta = rope_yarn(theta/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor);

            global float * src = (global float *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);

            const float x0 = src[0];
            const float x1 = src[n_dims/2];

            dst_row[ic]          = (half)(x0*cos_sin_theta.s0 - x1*cos_sin_theta.s1);
            dst_row[ic + n_dims/2] = (half)(x0*cos_sin_theta.s1 + x1*cos_sin_theta.s0);
        } else {
            global float * src = (global float *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
            dst_row[i0]     = (half)src[0];
            dst_row[i0 + 1] = (half)src[1];
        }
    }
}

//------------------------------------------------------------------------------
// rms_norm + mul(weight) + rope, fused. Folds the per-head Q/K RMS-norm and its
// learned-weight multiply (Gemma/Qwen q_norm/k_norm) into the rope kernel, so a
// chain that was {rms_norm+mul} + {rope} = 2 dispatches becomes 1 (and drops the
// normed intermediate's global round-trip). Input src0 is the un-normed Q/K
// (contiguous head rows of ne00), src3 is the [ne00] norm weight (broadcast over
// heads/tokens). One WG per (head=i1, token=i2) row reduces sum(x^2) across the
// row, then applies (rmsnorm(x)*w) followed by rope. f32 source only.
//------------------------------------------------------------------------------
#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_32
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_rope_neox_f32_rms(
        global void * src0,
        ulong offset0,
        global int * src1,
        ulong offset1,
        global float * src2,
        ulong offset2,
        global float * src3,
        ulong offset3,
        int ne_w0,
        float eps,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3,
        int n_past,
        int n_dims,
        int n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        local float * sum
) {
    src0 = (global void*)((global char*)src0 + offset0);
    src1 = (global int*)((global char*)src1 + offset1);
    src2 = (global float*)((global char*)src2 + offset2);
    src3 = (global float*)((global char*)src3 + offset3);
    dst = (global float*)((global char*)dst + offsetd);

    int i3 = get_group_id(2);
    int i2 = get_group_id(1);
    int i1 = get_group_id(0);

    // rms reduction over the contiguous row (ne00 elements)
    if (get_sub_group_id() == 0) {
        sum[get_sub_group_local_id()] = 0.0f;
    }
    global float4 * x4 = (global float4 *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01);
    float sumf = 0.0f;
    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
        sumf += dot(x4[i00], x4[i00]);
    }
    sumf = sub_group_reduce_add(sumf);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_sub_group_local_id() == 0) {
        sum[get_sub_group_id()] = sumf;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    sumf = sum[get_sub_group_local_id()];
    sumf = sub_group_reduce_add(sumf);

    const float mean  = sumf / ne00;
    const float scale = 1.0f/sqrt(mean + eps);

    float2 corr_dims = rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow);

    global int * pos = src1;
    float theta_base = (float) pos[i2];
    float inv_ndims = -1.f/n_dims;

    for (int i0 = 2*get_local_id(0); i0 < ne0; i0 += 2*get_local_size(0)) {
        if (i0 < n_dims) {
            int ic = i0/2;

            const float theta = theta_base * pow(freq_base, inv_ndims*i0);

            const float freq_factor = src2 != src0 ? src2[ic] : 1.0f;

            float2 cos_sin_theta = rope_yarn(theta/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor);

            global float * src      = (global float *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);
            global float * dst_data = (global float *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + ic*nb0);

            const float x0 = src[0]        * scale * src3[ic % ne_w0];
            const float x1 = src[n_dims/2] * scale * src3[(ic + n_dims/2) % ne_w0];

            dst_data[0]        = x0*cos_sin_theta.s0 - x1*cos_sin_theta.s1;
            dst_data[n_dims/2] = x0*cos_sin_theta.s1 + x1*cos_sin_theta.s0;
        } else {
            global float * const src = (global float *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
            global float * dst_data  = (global float *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

            dst_data[0] = src[0] * scale * src3[i0 % ne_w0];
            dst_data[1] = src[1] * scale * src3[(i0 + 1) % ne_w0];
        }
    }
}

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_32
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_rope_norm_f32_rms(
        global void * src0,
        ulong offset0,
        global int * src1,
        ulong offset1,
        global float * src2,
        ulong offset2,
        global float * src3,
        ulong offset3,
        int ne_w0,
        float eps,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3,
        int n_past,
        int n_dims,
        int n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        local float * sum
) {
    src0 = (global void*)((global char*)src0 + offset0);
    src1 = (global int*)((global char*)src1 + offset1);
    src2 = (global float*)((global char*)src2 + offset2);
    src3 = (global float*)((global char*)src3 + offset3);
    dst = (global float*)((global char*)dst + offsetd);

    int i3 = get_group_id(2);
    int i2 = get_group_id(1);
    int i1 = get_group_id(0);

    if (get_sub_group_id() == 0) {
        sum[get_sub_group_local_id()] = 0.0f;
    }
    global float4 * x4 = (global float4 *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01);
    float sumf = 0.0f;
    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
        sumf += dot(x4[i00], x4[i00]);
    }
    sumf = sub_group_reduce_add(sumf);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_sub_group_local_id() == 0) {
        sum[get_sub_group_id()] = sumf;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    sumf = sum[get_sub_group_local_id()];
    sumf = sub_group_reduce_add(sumf);

    const float mean  = sumf / ne00;
    const float scale = 1.0f/sqrt(mean + eps);

    float2 corr_dims = rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow);

    global int * pos = src1;
    float theta_base = (float) pos[i2];
    float inv_ndims = -1.f/n_dims;

    for (int i0 = 2*get_local_id(0); i0 < ne0; i0 += 2*get_local_size(0)) {
        if (i0 < n_dims) {
            int ic = i0/2;

            float theta = theta_base * pow(freq_base, inv_ndims*i0);

            float freq_factor = src2 != src0 ? src2[ic] : 1.0f;

            float2 cos_sin_theta = rope_yarn(theta/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor);

            global float * src       = (global float *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
            global float * dst_data  = (global float *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

            float x0 = src[0] * scale * src3[i0 % ne_w0];
            float x1 = src[1] * scale * src3[(i0 + 1) % ne_w0];

            dst_data[0] = x0*cos_sin_theta.s0 - x1*cos_sin_theta.s1;
            dst_data[1] = x0*cos_sin_theta.s1 + x1*cos_sin_theta.s0;
        } else {
            global float * src      = (global float *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
            global float * dst_data = (global float *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

            dst_data[0] = src[0] * scale * src3[i0 % ne_w0];
            dst_data[1] = src[1] * scale * src3[(i0 + 1) % ne_w0];
        }
    }
}


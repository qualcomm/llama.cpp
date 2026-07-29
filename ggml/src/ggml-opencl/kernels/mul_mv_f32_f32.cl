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

#define N_F32_F32 4

#if !defined(GGML_CL_ONLY) || GGML_CL_ONLY == 1
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_f32_f32(
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src0 = (global char*)((global char*)src0 + offset0);
    src1 = (global char*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    int r0 = get_group_id(0);
    int rb = get_group_id(1)*N_F32_F32;
    int im = get_group_id(2);

    int i12 = im%ne12;
    int i13 = im/ne12;

    ulong offset_src0 = r0*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;

    global float * x = (global float *) (src0 + offset_src0);

    if (ne00 < 128) {
        for (int row = 0; row < N_F32_F32; ++row) {
            int r1 = rb + row;
            if (r1 >= ne11) {
                break;
            }

            ulong offset_src1 = r1*nb11 + (i12   )*nb12 + (i13   )*nb13;

            global float * y = (global float *) (src1 + offset_src1);

            float sumf = 0;
            for (int i = get_sub_group_local_id(); i < ne00; i += get_max_sub_group_size()) {
                sumf += (float) x[i] * (float) y[i];
            }

            float all_sum = sub_group_reduce_add(sumf);
            if (get_sub_group_local_id() == 0) {
                dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
            }
        }
    } else {
        global float4 * x4 = (global float4 *)x;
        for (int row = 0; row < N_F32_F32; ++row) {
            int r1 = rb + row;
            if (r1 >= ne11) {
                break;
            }

            ulong offset_src1 = r1*nb11 + (i12   )*nb12 + (i13   )*nb13;

            global float  * y  = (global float  *) (src1 + offset_src1);
            global float4 * y4 = (global float4 *) y;

            float sumf = 0;
            for (int i = get_sub_group_local_id(); i < ne00/4; i += get_max_sub_group_size()) {
                sumf += (float) x4[i].s0 * y4[i].s0;
                sumf += (float) x4[i].s1 * y4[i].s1;
                sumf += (float) x4[i].s2 * y4[i].s2;
                sumf += (float) x4[i].s3 * y4[i].s3;
            }

            float all_sum = sub_group_reduce_add(sumf);
            if (get_sub_group_local_id() == 0) {
                for (int i = 4*(ne00/4); i < ne00; ++i) {
                    all_sum += (float) x[i] * y[i];
                }
                dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
            }
        }
    }
}
#endif

// kernel_mul_mat_f32_f32 with a fused GELU (tanh-approx) epilogue on the output.
// Identical matmul math to the kernel above; only the two output writes apply
// gelu(all_sum). Folds a MUL_MAT(f32) + UNARY(GELU) adjacency (Gemma-4 inp_gate)
// into one dispatch, byte-identical to the separate matmul + gelu path.
#define GELU_COEF_A    0.044715f
#define SQRT_2_OVER_PI 0.79788456080286535587989211986876f

#if !defined(GGML_CL_ONLY) || GGML_CL_ONLY == 2
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_f32_f32_gelu(
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src0 = (global char*)((global char*)src0 + offset0);
    src1 = (global char*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    int r0 = get_group_id(0);
    int rb = get_group_id(1)*N_F32_F32;
    int im = get_group_id(2);

    int i12 = im%ne12;
    int i13 = im/ne12;

    ulong offset_src0 = r0*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;

    global float * x = (global float *) (src0 + offset_src0);

    if (ne00 < 128) {
        for (int row = 0; row < N_F32_F32; ++row) {
            int r1 = rb + row;
            if (r1 >= ne11) {
                break;
            }

            ulong offset_src1 = r1*nb11 + (i12   )*nb12 + (i13   )*nb13;

            global float * y = (global float *) (src1 + offset_src1);

            float sumf = 0;
            for (int i = get_sub_group_local_id(); i < ne00; i += get_max_sub_group_size()) {
                sumf += (float) x[i] * (float) y[i];
            }

            float all_sum = sub_group_reduce_add(sumf);
            if (get_sub_group_local_id() == 0) {
                all_sum = 0.5f*all_sum*(1.0f + tanh(SQRT_2_OVER_PI*all_sum*(1.0f + GELU_COEF_A*all_sum*all_sum)));
                dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
            }
        }
    } else {
        global float4 * x4 = (global float4 *)x;
        for (int row = 0; row < N_F32_F32; ++row) {
            int r1 = rb + row;
            if (r1 >= ne11) {
                break;
            }

            ulong offset_src1 = r1*nb11 + (i12   )*nb12 + (i13   )*nb13;

            global float  * y  = (global float  *) (src1 + offset_src1);
            global float4 * y4 = (global float4 *) y;

            float sumf = 0;
            for (int i = get_sub_group_local_id(); i < ne00/4; i += get_max_sub_group_size()) {
                sumf += (float) x4[i].s0 * y4[i].s0;
                sumf += (float) x4[i].s1 * y4[i].s1;
                sumf += (float) x4[i].s2 * y4[i].s2;
                sumf += (float) x4[i].s3 * y4[i].s3;
            }

            float all_sum = sub_group_reduce_add(sumf);
            if (get_sub_group_local_id() == 0) {
                for (int i = 4*(ne00/4); i < ne00; ++i) {
                    all_sum += (float) x[i] * y[i];
                }
                all_sum = 0.5f*all_sum*(1.0f + tanh(SQRT_2_OVER_PI*all_sum*(1.0f + GELU_COEF_A*all_sum*all_sum)));
                dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
            }
        }
    }
}
#endif

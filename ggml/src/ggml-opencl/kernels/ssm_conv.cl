kernel void kernel_ssm_conv_f32_f32(
    global char * src0,
    ulong         offset0,
    global char * src1,
    ulong         offset1,
    global char * dst,
    ulong         offsetd,
    ulong         nb00,
    ulong         nb01,
    ulong         nb02,
    int           ne10,
    ulong         nb11,
    ulong         nb0,
    ulong         nb1,
    ulong         nb2
){
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    int ir = get_global_id(0);
    int i2 = get_global_id(1);
    int i3 = get_global_id(2);

    int nc  = ne10;

    global float * s = (global float *) (src0 + ir*nb01 + i2*nb00 + i3*nb02);
    global float * c = (global float *) (src1 + ir*nb11);
    global float * d = (global float *) (dst  + ir*nb0  + i2*nb1  + i3*nb2);

    float sumf = 0.0f;

    for (int i0 = 0; i0 < nc; ++i0) {
        sumf += s[i0] * c[i0];
    }

    d[0] = sumf;
}

kernel void kernel_ssm_conv_f32_f32_4(
    global char * src0,
    ulong         offset0,
    global char * src1,
    ulong         offset1,
    global char * dst,
    ulong         offsetd,
    ulong         nb00,
    ulong         nb01,
    ulong         nb02,
    int           ne10,
    ulong         nb11,
    ulong         nb0,
    ulong         nb1,
    ulong         nb2
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    int ir = get_global_id(0);
    int i2 = get_global_id(1);
    int i3 = get_global_id(2);

    int nc = ne10;

    global float4 * s = (global float4 *) (src0 + ir*nb01 + i2*nb00 + i3*nb02);
    global float4 * c = (global float4 *) (src1 + ir*nb11);
    global float  * d = (global float  *) (dst  + ir*nb0  + i2*nb1  + i3*nb2);

    float sumf = 0.0f;

    for (int i0 = 0; i0 < nc/4; ++i0) {
        sumf += dot(s[i0], c[i0]);
    }

    d[0] = sumf;
}

// One work item makes TPI consecutive tokens of one channel. The window slides by one element
// per token, so keeping it in registers costs one input load per output, against nc input loads
// plus nc weight loads when a work item makes a single output.
kernel void kernel_ssm_conv_f32_f32_tpi(
    global char * src0,
    ulong         offset0,
    global char * src1,
    ulong         offset1,
    global char * dst,
    ulong         offsetd,
    ulong         nb00,
    ulong         nb01,
    ulong         nb02,
    int           ne10,
    ulong         nb11,
    ulong         nb0,
    ulong         nb1,
    ulong         nb2,
    int           ne1,
    int           tpi
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    int ir = get_global_id(0);
    int t0 = get_global_id(1)*tpi;
    int i3 = get_global_id(2);

    if (t0 >= ne1) {
        return;
    }

    int nc = ne10;
    int nt = min(tpi, ne1 - t0);

    global float * s = (global float *) (src0 + ir*nb01 + t0*nb00 + i3*nb02);
    global float * c = (global float *) (src1 + ir*nb11);
    global char  * d = dst + ir*nb0 + t0*nb1 + i3*nb2;

    if (nc == 4) {
        const float w0 = c[0];
        const float w1 = c[1];
        const float w2 = c[2];
        const float w3 = c[3];

        float x0 = s[0];
        float x1 = s[1];
        float x2 = s[2];

        for (int t = 0; t < nt; ++t) {
            const float x3 = s[t + 3];

            *(global float *)(d + t*nb1) = x0*w0 + x1*w1 + x2*w2 + x3*w3;

            x0 = x1;
            x1 = x2;
            x2 = x3;
        }

        return;
    }

    for (int t = 0; t < nt; ++t) {
        float sumf = 0.0f;

        for (int i0 = 0; i0 < nc; ++i0) {
            sumf += s[t + i0] * c[i0];
        }

        *(global float *)(d + t*nb1) = sumf;
    }
}

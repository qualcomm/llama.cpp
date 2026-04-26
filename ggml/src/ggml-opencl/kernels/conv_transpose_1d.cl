#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// 1D transpose convolution.
// src0 (kernel): [K, Cout, Cin]      => ne00=K, ne01=Cout, ne02=Cin
// src1 (input):  [Lin, Cin]           => ne10=Lin, ne11=Cin
// dst (output):  [Lout, Cout]         => ne0=Lout, ne1=Cout
// One work-item per output element (i1=Cout, t=Lout position).

kernel void kernel_conv_transpose_1d_f32(
    global const float * src0,
    ulong                offset0,
    global const float * src1,
    ulong                offset1,
    global       float * dst,
    ulong                offsetd,
    int                  K,    // ne00
    int                  Cout, // ne01
    int                  Cin,  // ne02
    int                  Lin,  // ne10
    int                  Lout, // ne0
    ulong                nb01, ulong nb02,
    ulong                nb11,
    ulong                nb1,
    int                  s0
) {
    src0 = (global const float *)((global const char *) src0 + offset0);
    src1 = (global const float *)((global const char *) src1 + offset1);
    dst  = (global       float *)((global       char *) dst  + offsetd);

    const int gid = get_global_id(0);
    const int total = Lout * Cout;
    if (gid >= total) return;

    const int t  = gid % Lout;
    const int i1 = gid / Lout;

    float out = 0.0f;

    for (int i00 = 0; i00 < K; i00++) {
        const int delta = t - i00;
        if (delta < 0) continue;
        if (delta % s0 != 0) continue;
        const int i10 = delta / s0;
        if (i10 >= Lin) continue;

        for (int c = 0; c < Cin; c++) {
            const float k_v = *(global const float *)((global const char *) src0 + (ulong)c*nb02 + (ulong)i1*nb01 + (ulong)i00*sizeof(float));
            const float i_v = *(global const float *)((global const char *) src1 + (ulong)c*nb11 + (ulong)i10*sizeof(float));
            out += k_v * i_v;
        }
    }

    global float * dst_row = (global float *)((global char *) dst + (ulong)i1*nb1);
    dst_row[t] = out;
}

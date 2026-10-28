#pragma OPENCL EXTENSION cl_khr_fp16 : enable

kernel void kernel_gemv_noshuffle_q2_0_f32(
        global const uchar * src0_qs,
        global const half * src0_d,
        global const float * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne10,
        int ne0
) {
    src1 = (global const float *)((global const char *)src1 + offset1);
    dst = (global float *)((global char *)dst + offsetd);

    const uint m = (uint)ne01;
    const uint row = get_global_id(0);
    const uint col = get_group_id(1);
    if (row >= m) {
        return;
    }
    global const float * y = src1 + (ulong)col * (uint)ne10;
    float sum = 0.0f;
    for (uint ib = 0; ib < (uint)ne00/64; ++ib) {
        float acc = 0.0f;
        for (uint u = 0; u < 16; ++u) {
            const uint group = ib*16 + u;
            const uint q = src0_qs[row + group*m];
            const int4 v = convert_int4(((uint4)(q) >> (uint4)(0, 2, 4, 6)) & (uint4)(3)) - (int4)(1);
            acc += dot(vload4(group, y), convert_float4(v));
        }
        sum += (float)src0_d[row + ib*m]*acc;
    }
    dst[(ulong)col*(uint)ne0 + row] = sum;
}

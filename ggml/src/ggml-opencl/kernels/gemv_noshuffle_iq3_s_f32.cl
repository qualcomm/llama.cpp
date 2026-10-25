#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define QK_K 256

inline float4 iq3_s_vals(uint gv, uint sg, uint base) {

    const uint  s   = sg >> base;
    const uint4 sgn = (uint4)(s << 31, s << 30, s << 29, s << 28) & 0x80000000u;
    return as_float4(as_uint4(convert_float4(as_uchar4(gv))) ^ sgn);
}

kernel void kernel_gemv_noshuffle_iq3_s_f32(
        global const uchar * src0_qs,
        global const uchar * src0_sg,
        global const uchar * src0_qh,
        global const uchar * src0_sc,
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
    for (uint ib = 0; ib < (uint)ne00/QK_K; ++ib) {
        float acc = 0.0f;
        for (uint sb = 0; sb < 8; ++sb) {
            const uint sub = ib*8 + sb;
            const uint qh = src0_qh[row + sub*m];
            const uint sc = src0_sc[row + (sub/2)*m];
            const float scale = (float)(1 + 2*((sc >> (4*(sb & 1))) & 15u));
            for (uint l = 0; l < 4; ++l) {
                const uint group = sub*4 + l;
                const uint gi0 = src0_qs[row + (2*group)*m] | (((qh >> (2*l)) & 1u) << 8);
                const uint gi1 = src0_qs[row + (2*group + 1)*m] | (((qh >> (2*l + 1)) & 1u) << 8);
                const uint sg = src0_sg[row + group*m];
                const float a = dot(vload4(2*group, y), iq3_s_vals(iq3_s_grid[gi0], sg, 0))
                              + dot(vload4(2*group + 1, y), iq3_s_vals(iq3_s_grid[gi1], sg, 4));
                acc += scale*a;
            }
        }
        sum += (float)src0_d[row + ib*m]*acc;
    }
    dst[(ulong)col*(uint)ne0 + row] = sum;
}

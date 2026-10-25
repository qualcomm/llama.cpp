#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define QK_K 256

inline float4 iq1_s_vals(uint g) {
    return convert_float4(as_uchar4(g & 0x0F0F0F0Fu));
}

kernel void kernel_gemv_noshuffle_iq1_s_f32(
        global const uchar * src0_qs,
        global const ushort * src0_qh,
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
            for (uint l = 0; l < 4; ++l) {
                const uint group = sub*4 + l;
                const uint gi = (uint)src0_qs[row + group*m] | (((qh >> (3*l)) & 7u) << 8);
                const uint g = iq1_s_grid[gi];
                const float scale = (float)(2*((qh >> 12) & 7u) + 1);
                const float delta = (qh & 0x8000u) ? -1.125f : -0.875f;
                const float a = dot(vload4(2*group, y), iq1_s_vals(g) + delta)
                              + dot(vload4(2*group + 1, y), iq1_s_vals(g >> 4) + delta);
                acc += scale*a;
            }
        }
        sum += (float)src0_d[row + ib*m]*acc;
    }
    dst[(ulong)col*(uint)ne0 + row] = sum;
}

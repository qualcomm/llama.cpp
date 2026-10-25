#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define QK_K 256

inline float4 iq1_m_vals(uint g) {
    return convert_float4(as_uchar4(g & 0x0F0F0F0Fu));
}

kernel void kernel_gemv_noshuffle_iq1_m_f32(
        global const uchar * src0_qs,
        global const uchar * src0_qh,
        global const ushort * src0_sc,
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
        const uint w0 = src0_sc[row + (4*ib + 0)*m];
        const uint w1 = src0_sc[row + (4*ib + 1)*m];
        const uint w2 = src0_sc[row + (4*ib + 2)*m];
        const uint w3 = src0_sc[row + (4*ib + 3)*m];
        const ushort u16 = (ushort)((w0 >> 12) | ((w1 >> 8) & 0x00f0u)
                                  | ((w2 >> 4) & 0x0f00u) | (w3 & 0xf000u));
        const float d = (float)as_half(u16);
        float acc = 0.0f;
        for (uint sb = 0; sb < 8; ++sb) {
            const uint sub = ib*8 + sb;
            const uint sc = src0_sc[row + (4*ib + sb/2)*m];
            for (uint l = 0; l < 4; ++l) {
                const uint group = sub*4 + l;
                const uint qh = src0_qh[row + (2*sub + l/2)*m];
                const uint gi = (uint)src0_qs[row + group*m] | (((qh >> (4*(l%2))) & 7u) << 8);
                const uint g = iq1_s_grid[gi];
                const float scale = (float)(2*((sc >> (6*(sb%2) + 3*(l/2))) & 7u) + 1);
                const float delta = (qh & (0x08u << (4*(l%2)))) ? -1.125f : -0.875f;
                const float a = dot(vload4(2*group, y), iq1_m_vals(g) + delta)
                              + dot(vload4(2*group + 1, y), iq1_m_vals(g >> 4) + delta);
                acc += scale*a;
            }
        }
        sum += d*acc;
    }
    dst[(ulong)col*(uint)ne0 + row] = sum;
}

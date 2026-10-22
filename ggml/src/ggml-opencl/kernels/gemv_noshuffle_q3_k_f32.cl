#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define QK_K 256

// Decode one of the sixteen packed 6-bit scales.
inline int q3_k_scale(uint sc0, uint sc1, uint sc2, uint is) {
    const uint w  = is >> 2;
    const uint sh = 8u * (is & 3u);
    const uint tb = (sc2 >> sh) & 0xFFu;
    uint v;
    if      (w == 0u) { v = ( (sc0 >> sh)       & 0xFu) | (((tb >> 0) & 3u) << 4); }
    else if (w == 1u) { v = ( (sc1 >> sh)       & 0xFu) | (((tb >> 2) & 3u) << 4); }
    else if (w == 2u) { v = (((sc0 >> sh) >> 4) & 0xFu) | (((tb >> 4) & 3u) << 4); }
    else              { v = (((sc1 >> sh) >> 4) & 0xFu) | (((tb >> 6) & 3u) << 4); }
    return (int)v - 32;
}

inline float4 q3_k_vals(uint pk, uint hb) {
    float4 v;
    v.s0 = (float)((int)((pk      ) & 3u) - 4 + 4*(int)((hb     ) & 1u));
    v.s1 = (float)((int)((pk >> 2) & 3u) - 4 + 4*(int)((hb >> 1) & 1u));
    v.s2 = (float)((int)((pk >> 4) & 3u) - 4 + 4*(int)((hb >> 2) & 1u));
    v.s3 = (float)((int)((pk >> 6) & 3u) - 4 + 4*(int)((hb >> 3) & 1u));
    return v;
}

kernel void kernel_gemv_noshuffle_q3_k_f32(
        global const uchar * src0_qs,
        global const uchar * src0_hm,
        global const uint  * src0_sc,
        global const half  * src0_d,
        global const float * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,      // K
        int ne01,      // M
        int ne10,      // activation row stride
        int ne0        // dst row stride
) {
    src1 = (global const float *)((global const char *)src1 + offset1);
    dst = (global float *)((global char *)dst + offsetd);

    const uint m = (uint)ne01;
    const uint nsb = (uint)ne00 / QK_K;
    const uint lid = get_local_id(0);
    const uint col = get_group_id(1);
    global const float * y = src1 + (ulong)col * (uint)ne10;

    const uint row = get_group_id(0) * 64u + lid;

    float sumf = 0.f;

    if (row < m) {
        for (uint ib = 0; ib < nsb; ++ib) {
            const float d  = (float)src0_d[row + ib * m];
            const uint  s0 = src0_sc[row + (3u * ib + 0u) * m];
            const uint  s1 = src0_sc[row + (3u * ib + 1u) * m];
            const uint  s2 = src0_sc[row + (3u * ib + 2u) * m];

            float acc = 0.f;
            for (uint sb = 0; sb < 8u; ++sb) {
                const uint grp = ib * 64u + sb * 8u;
                const uint qsb = row + grp * m;
                const uint hmb = row + (grp >> 1) * m;

                for (uint h = 0; h < 2u; ++h) {
                    const int l = q3_k_scale(s0, s1, s2, 2u*sb + h);
                    float a = 0.f;
                    for (uint u = 0; u < 4u; ++u) {
                        const uint gg  = 4u*h + u;
                        const uint qsv = (uint)src0_qs[qsb + gg * m];
                        const uint hmv = (uint)src0_hm[hmb + (gg >> 1) * m];
                        const float4 yv = vload4(grp + gg, y);
                        a += dot(yv, q3_k_vals(qsv, (hmv >> (4u * (gg & 1u))) & 0xFu));
                    }
                    acc += (float)l * a;
                }
            }
            sumf += d * acc;
        }
    }
    if (row < m) {
        dst[(ulong)col * (uint)ne0 + row] = sumf;
    }
}

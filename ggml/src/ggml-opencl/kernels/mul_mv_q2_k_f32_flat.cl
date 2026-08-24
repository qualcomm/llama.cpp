#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Q2_K decode GEMV over the feature-major plane split. Same structure as
// mul_mv_q3_k_f32_flat: K split across Q2K_MV_NSG subgroups, and Q2K_MV_R2 gives
// a lane two adjacent rows so the uchar planes are read a ushort at a time.
//
// The min term needs a per-16 activation sum, which here is shared between the
// two rows of the pair and computed alongside the dot products.

#define QK_K 256

#ifndef Q2K_MV_NSG
#define Q2K_MV_NSG 8
#endif

#ifndef Q2K_MV_R2
#define Q2K_MV_R2 1
#endif

// Four weights of one group as floats (0..3).
inline float4 q2k_vals(uint pk) {
    return (float4)((float)( pk       & 3u), (float)((pk >> 2) & 3u),
                    (float)((pk >> 4) & 3u), (float)((pk >> 6) & 3u));
}

kernel void kernel_mul_mv_q2_k_f32_flat(
        global const uchar * src0_qs,
        global const uchar * src0_sc,
        global const half  * src0_dm,
        global const float * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,      // K
        int ne01,      // M
        int ne10,      // activation row stride, == K
        int ne0        // dst row stride
) {
    src1 = (global const float *)((global const char *)src1 + offset1);
    dst  = (global float       *)((global char       *)dst  + offsetd);

    const uint m   = (uint)ne01;
    const uint K   = (uint)ne00;
    const uint nsb = K / QK_K;

    const uint lid = get_local_id(0);
    const uint sgi = get_local_id(1);
    const uint col = get_group_id(1);

    global const float * y = src1 + (ulong)col * (uint)ne10;

#if Q2K_MV_R2
    const uint mh  = m >> 1;
    const uint j   = get_group_id(0) * 64u + lid;
    const uint row = j << 1;

    float sumf  = 0.f;
    float sumf1 = 0.f;

    if (j < mh) {
        global const ushort * qsu = (global const ushort *)src0_qs;
        global const ushort * scu = (global const ushort *)src0_sc;

        for (uint ib = sgi; ib < nsb; ib += Q2K_MV_NSG) {
            // d0, dmin0, d1, dmin1
            const half4 dm = vload4(j + ib * mh, src0_dm);

            float ad0 = 0.f, am0 = 0.f, ad1 = 0.f, am1 = 0.f;
            for (uint sb = 0; sb < 8u; ++sb) {
                const uint grp = ib * 64u + sb * 8u;
                const uint qsb = j + grp * mh;

                for (uint h = 0; h < 2u; ++h) {
                    const uint scv = (uint)scu[j + (2u * (ib * 8u + sb) + h) * mh];
                    const uint s0  = scv & 0xFFu;
                    const uint s1  = scv >> 8;

                    float a0 = 0.f, a1 = 0.f, asum = 0.f;
                    for (uint u = 0; u < 4u; ++u) {
                        const uint gg  = 4u*h + u;
                        const uint qsv = (uint)qsu[qsb + gg * mh];
                        const float4 yv = vload4(grp + gg, y);
                        asum += yv.s0 + yv.s1 + yv.s2 + yv.s3;
                        a0 += dot(yv, q2k_vals( qsv       & 0xFFu));
                        a1 += dot(yv, q2k_vals((qsv >> 8) & 0xFFu));
                    }
                    ad0 += (float)(s0 & 0xFu) * a0;   am0 += (float)(s0 >> 4) * asum;
                    ad1 += (float)(s1 & 0xFu) * a1;   am1 += (float)(s1 >> 4) * asum;
                }
            }
            sumf  += (float)dm.s0 * ad0 - (float)dm.s1 * am0;
            sumf1 += (float)dm.s2 * ad1 - (float)dm.s3 * am1;
        }
    }
#else
    const uint row = get_group_id(0) * 64u + lid;

    float sumf = 0.f;

    if (row < m) {
        for (uint ib = sgi; ib < nsb; ib += Q2K_MV_NSG) {
            const half2 dm = vload2(row + ib * m, src0_dm);

            float ad = 0.f, am = 0.f;
            for (uint sb = 0; sb < 8u; ++sb) {
                const uint grp = ib * 64u + sb * 8u;
                const uint qsb = row + grp * m;

                for (uint h = 0; h < 2u; ++h) {
                    const uint sc = (uint)src0_sc[row + (2u * (ib * 8u + sb) + h) * m];

                    float a = 0.f, asum = 0.f;
                    for (uint u = 0; u < 4u; ++u) {
                        const uint gg = 4u*h + u;
                        const float4 yv = vload4(grp + gg, y);
                        asum += yv.s0 + yv.s1 + yv.s2 + yv.s3;
                        a += dot(yv, q2k_vals((uint)src0_qs[qsb + gg * m]));
                    }
                    ad += (float)(sc & 0xFu) * a;   am += (float)(sc >> 4) * asum;
                }
            }
            sumf += (float)dm.s0 * ad - (float)dm.s1 * am;
        }
    }
#endif

#if Q2K_MV_NSG > 1
#if Q2K_MV_R2
    __local float2 part[Q2K_MV_NSG][64];
    part[sgi][lid] = (float2)(sumf, sumf1);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sgi != 0) {
        return;
    }
    for (uint s = 1; s < Q2K_MV_NSG; ++s) {
        const float2 p = part[s][lid];
        sumf  += p.s0;
        sumf1 += p.s1;
    }
#else
    __local float part[Q2K_MV_NSG][64];
    part[sgi][lid] = sumf;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sgi != 0) {
        return;
    }
    for (uint s = 1; s < Q2K_MV_NSG; ++s) {
        sumf += part[s][lid];
    }
#endif
#endif

#if Q2K_MV_R2
    if (j < mh) {
        vstore2((float2)(sumf, sumf1), 0, dst + (ulong)col * (uint)ne0 + row);
    }
#else
    if (row < m) {
        dst[(ulong)col * (uint)ne0 + row] = sumf;
    }
#endif
}

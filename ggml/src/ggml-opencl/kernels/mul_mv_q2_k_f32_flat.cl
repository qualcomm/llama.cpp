#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Q2_K decode GEMV over the feature-major plane split. Same structure as
// mul_mv_q3_k_f32_flat: K split across Q2K_MV_NSG subgroups, and Q2K_MV_R
// adjacent rows per lane so the uchar planes are read a word at a time.
//
// 🔑 Q2_K wants MORE rows per lane than the other split types, for a reason that
// has nothing to do with load width. It carries a min, so every 16-weight run
// also needs the sum of that run's activations -- and that sum does not depend
// on the row. The AoS kernel this replaces computes it once per lane and reuses
// it across its N_DST = 4 rows; at 2 rows per lane the flat kernel paid it twice
// as often per row and LOST 12% of decode despite doubling prefill. So the
// default here is 4, where for IQ3_S (no min term, nothing to amortise) 4 rows
// was a regression because it only shrinks the grid.

#define QK_K 256

#ifndef Q2K_MV_NSG
#define Q2K_MV_NSG 8
#endif

// Rows per work item: 1, 2 or 4. Each is hand-written -- an unrolled generic
// r-loop was measured on IQ3_S and cost 8% by itself.
#ifndef Q2K_MV_R
#define Q2K_MV_R 4
#endif

// Four weights of one group as floats (0..3).
// MV_WORK2=1: COST PROBE, WRONG MATH. Repeat this kernel's per-operand ARITHMETIC
// on data that is already in registers -- no extra loads at all. Doubling the work
// while holding the loads fixed is the only way to tell a compute-bound kernel from
// a bandwidth-bound one; every ablation probe removes a computation AND its load
// together and therefore cannot. A bandwidth-bound kernel is flat under this.
// Operands are perturbed so the duplicate cannot be common-subexpression eliminated.
#ifndef MV_WORK2
#define MV_WORK2 0
#endif

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

    const uint mr  = m / Q2K_MV_R;                  // row groups
    const uint j   = get_group_id(0) * 64u + lid;   // row group this lane owns
    const uint row = j * Q2K_MV_R;

#if Q2K_MV_R == 4
    float sumf0 = 0.f, sumf1 = 0.f, sumf2 = 0.f, sumf3 = 0.f;

    if (j < mr) {
        global const uint * qsu = (global const uint *)src0_qs;
        global const uint * scu = (global const uint *)src0_sc;

        for (uint ib = sgi; ib < nsb; ib += Q2K_MV_NSG) {
            // the d/dmin plane element is a half PAIR, so four rows of one
            // super-block are eight halves starting at 2*(j + ib*mr) half4s
            const half4 dma = vload4(2u*(j + ib*mr) + 0u, src0_dm);  // d0, dmin0, d1, dmin1
            const half4 dmb = vload4(2u*(j + ib*mr) + 1u, src0_dm);  // d2, dmin2, d3, dmin3

            float ad0 = 0.f, am0 = 0.f, ad1 = 0.f, am1 = 0.f;
            float ad2 = 0.f, am2 = 0.f, ad3 = 0.f, am3 = 0.f;

            for (uint sb = 0; sb < 8u; ++sb) {
                const uint grp = ib * 64u + sb * 8u;
                const uint qsb = j + grp * mr;

                for (uint h = 0; h < 2u; ++h) {
                    const uint scv = scu[j + (2u * (ib * 8u + sb) + h) * mr];

                    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
                    float4 as = (float4)(0.f);
                    for (uint u = 0; u < 4u; ++u) {
                        const uint gg  = 4u*h + u;
                        const uint qsv = qsu[qsb + gg * mr];   // four rows, one load
                        const float4 yv = vload4(grp + gg, y);
                        as += yv;
                        a0 += dot(yv, q2k_vals( qsv        & 0xFFu));
#if MV_WORK2
                        a0 += dot(yv, q2k_vals((qsv + 1u)  & 0xFFu));
#endif
                        a1 += dot(yv, q2k_vals((qsv >>  8) & 0xFFu));
                        a2 += dot(yv, q2k_vals((qsv >> 16) & 0xFFu));
                        a3 += dot(yv, q2k_vals((qsv >> 24) & 0xFFu));
                    }
                    // one activation sum, four rows -- this is the whole point
                    const float asum = as.s0 + as.s1 + as.s2 + as.s3;

                    const uint s0 =  scv        & 0xFFu;
                    const uint s1 = (scv >>  8) & 0xFFu;
                    const uint s2 = (scv >> 16) & 0xFFu;
                    const uint s3 = (scv >> 24) & 0xFFu;
                    ad0 += (float)(s0 & 0xFu) * a0;   am0 += (float)(s0 >> 4) * asum;
                    ad1 += (float)(s1 & 0xFu) * a1;   am1 += (float)(s1 >> 4) * asum;
                    ad2 += (float)(s2 & 0xFu) * a2;   am2 += (float)(s2 >> 4) * asum;
                    ad3 += (float)(s3 & 0xFu) * a3;   am3 += (float)(s3 >> 4) * asum;
                }
            }
            sumf0 += (float)dma.s0 * ad0 - (float)dma.s1 * am0;
            sumf1 += (float)dma.s2 * ad1 - (float)dma.s3 * am1;
            sumf2 += (float)dmb.s0 * ad2 - (float)dmb.s1 * am2;
            sumf3 += (float)dmb.s2 * ad3 - (float)dmb.s3 * am3;
        }
    }

#if Q2K_MV_NSG > 1
    __local float4 part[Q2K_MV_NSG][64];
    part[sgi][lid] = (float4)(sumf0, sumf1, sumf2, sumf3);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sgi != 0) {
        return;
    }
    for (uint s = 1; s < Q2K_MV_NSG; ++s) {
        const float4 p = part[s][lid];
        sumf0 += p.s0; sumf1 += p.s1; sumf2 += p.s2; sumf3 += p.s3;
    }
#endif

    if (j < mr) {
        global float * o = dst + (ulong)col * (uint)ne0 + row;
        o[0] = sumf0; o[1] = sumf1; o[2] = sumf2; o[3] = sumf3;
    }

#elif Q2K_MV_R == 2
    float sumf0 = 0.f, sumf1 = 0.f;

    if (j < mr) {
        global const ushort * qsu = (global const ushort *)src0_qs;
        global const ushort * scu = (global const ushort *)src0_sc;

        for (uint ib = sgi; ib < nsb; ib += Q2K_MV_NSG) {
            const half4 dm = vload4(j + ib * mr, src0_dm);  // d0, dmin0, d1, dmin1

            float ad0 = 0.f, am0 = 0.f, ad1 = 0.f, am1 = 0.f;
            for (uint sb = 0; sb < 8u; ++sb) {
                const uint grp = ib * 64u + sb * 8u;
                const uint qsb = j + grp * mr;

                for (uint h = 0; h < 2u; ++h) {
                    const uint scv = (uint)scu[j + (2u * (ib * 8u + sb) + h) * mr];
                    const uint s0  = scv & 0xFFu;
                    const uint s1  = scv >> 8;

                    float a0 = 0.f, a1 = 0.f;
                    float4 as = (float4)(0.f);
                    for (uint u = 0; u < 4u; ++u) {
                        const uint gg  = 4u*h + u;
                        const uint qsv = (uint)qsu[qsb + gg * mr];
                        const float4 yv = vload4(grp + gg, y);
                        as += yv;
                        a0 += dot(yv, q2k_vals( qsv       & 0xFFu));
#if MV_WORK2
                        a0 += dot(yv, q2k_vals((qsv + 1u) & 0xFFu));
#endif
                        a1 += dot(yv, q2k_vals((qsv >> 8) & 0xFFu));
                    }
                    const float asum = as.s0 + as.s1 + as.s2 + as.s3;
                    ad0 += (float)(s0 & 0xFu) * a0;   am0 += (float)(s0 >> 4) * asum;
                    ad1 += (float)(s1 & 0xFu) * a1;   am1 += (float)(s1 >> 4) * asum;
                }
            }
            sumf0 += (float)dm.s0 * ad0 - (float)dm.s1 * am0;
            sumf1 += (float)dm.s2 * ad1 - (float)dm.s3 * am1;
        }
    }

#if Q2K_MV_NSG > 1
    __local float2 part[Q2K_MV_NSG][64];
    part[sgi][lid] = (float2)(sumf0, sumf1);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sgi != 0) {
        return;
    }
    for (uint s = 1; s < Q2K_MV_NSG; ++s) {
        const float2 p = part[s][lid];
        sumf0 += p.s0; sumf1 += p.s1;
    }
#endif

    if (j < mr) {
        vstore2((float2)(sumf0, sumf1), 0, dst + (ulong)col * (uint)ne0 + row);
    }

#else
    float sumf0 = 0.f;

    if (j < mr) {
        for (uint ib = sgi; ib < nsb; ib += Q2K_MV_NSG) {
            const half2 dm = vload2(row + ib * m, src0_dm);

            float ad = 0.f, am = 0.f;
            for (uint sb = 0; sb < 8u; ++sb) {
                const uint grp = ib * 64u + sb * 8u;
                const uint qsb = row + grp * m;

                for (uint h = 0; h < 2u; ++h) {
                    const uint sc = (uint)src0_sc[row + (2u * (ib * 8u + sb) + h) * m];

                    float a = 0.f;
                    float4 as = (float4)(0.f);
                    for (uint u = 0; u < 4u; ++u) {
                        const uint gg = 4u*h + u;
                        const float4 yv = vload4(grp + gg, y);
                        as += yv;
                        const uint pkv = (uint)src0_qs[qsb + gg * m];
                        a += dot(yv, q2k_vals(pkv));
#if MV_WORK2
                        a += dot(yv, q2k_vals(pkv + 1u));
#endif
                    }
                    const float asum = as.s0 + as.s1 + as.s2 + as.s3;
                    ad += (float)(sc & 0xFu) * a;   am += (float)(sc >> 4) * asum;
                }
            }
            sumf0 += (float)dm.s0 * ad - (float)dm.s1 * am;
        }
    }

#if Q2K_MV_NSG > 1
    __local float part[Q2K_MV_NSG][64];
    part[sgi][lid] = sumf0;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sgi != 0) {
        return;
    }
    for (uint s = 1; s < Q2K_MV_NSG; ++s) {
        sumf0 += part[s][lid];
    }
#endif

    if (j < mr) {
        dst[(ulong)col * (uint)ne0 + row] = sumf0;
    }
#endif
}


// ---------------------------------------------------------------------------
// Fused ffn_gate + ffn_up + GLU for Q2_K.
//
// Q2_K is the type this fusion should suit best, and the reason is written at the
// top of this file already: it carries a MIN, so every 16-weight run needs that
// run's activation SUM, and that sum does not depend on the row. The whole reason
// Q2K_MV_R defaults to 4 is to amortise it -- at 2 rows per lane the plain kernel
// paid it twice as often per row and lost 12% of decode.
//
// The 2026-08-25 round found that the GLU fusion pays exactly where it shares
// COMPUTATION rather than merely an activation load: IQ1_S, IQ1_M and IQ2_S each
// have a delta or min term multiplied by such a sum and gained 10-13%, while
// IQ4_XS, which has no such term, was a measured wash. Q2_K has the term.
//
// 🔑 WHY THIS KERNEL IS R2 WHERE THE PLAIN ONE IS R4. Fusing shares `asum` between
// the gate and up streams, so at 2 rows per lane a fused lane amortises it over
// FOUR row-streams -- exactly the amortisation R4 was introduced to get, and at
// half the accumulator count (4 running pairs instead of 8) and half the LDS for
// the cross-subgroup reduce. R4 here would need eight accumulator pairs and a
// float8 reduce, 16 KB of local memory, on a kernel already at 304 B/WI.
//
// So the host dispatches this one at 128 rows per workgroup, not 256.
// ---------------------------------------------------------------------------

// Fifth copy of the shared GLU epilogue -- each .cl is its own program and cannot
// include the others. Op numbering and expressions are identical to
// q40_glu_apply, iq2s_glu_apply, iq1s_glu_apply, iq1m_glu_apply and
// iq4xs_glu_apply on purpose; if one is ever changed, change all of them.
#define Q2K_GLU_GEGLU_COEF_A   0.044715f
#define Q2K_GLU_SQRT_2_OVER_PI 0.79788456080286535587989211986876f
#define Q2K_GLU_SQRT_2_INV     0.70710678118654752440084436210484f
#define Q2K_GLU_QUICK_COEF    -1.702f
inline float q2k_glu_apply(int glu_op, float g, float u) {
    float act;
    if (glu_op == 1) {        // GEGLU (tanh-approx gelu)
        act = 0.5f*g*(1.0f + tanh(Q2K_GLU_SQRT_2_OVER_PI*g*(1.0f + Q2K_GLU_GEGLU_COEF_A*g*g)));
    } else if (glu_op == 2) { // SWIGLU (silu)
        act = g / (1.0f + exp(-g));
    } else if (glu_op == 0) { // REGLU
        return g*u*(g > 0.0f);
    } else if (glu_op == 4) { // GEGLU_ERF
        act = 0.5f*g*(1.0f + erf(g*Q2K_GLU_SQRT_2_INV));
    } else {                  // GEGLU_QUICK
        act = g*(1.0f/(1.0f + exp(Q2K_GLU_QUICK_COEF*g)));
    }
    return act*u;
}

kernel void kernel_mul_mv_q2_k_f32_flat_glu(
        global const uchar * g_qs,
        global const uchar * g_sc,
        global const half  * g_dm,
        global const uchar * u_qs,
        global const uchar * u_sc,
        global const half  * u_dm,
        global const float * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne10,
        int ne0,
        int glu_op
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

    const uint mr  = m >> 1;                        // row pairs
    const uint j   = get_group_id(0) * 64u + lid;
    const uint row = j << 1;

    float gs0 = 0.f, gs1 = 0.f, us0 = 0.f, us1 = 0.f;

    if (j < mr) {
        global const ushort * gqsu = (global const ushort *)g_qs;
        global const ushort * gscu = (global const ushort *)g_sc;
        global const ushort * uqsu = (global const ushort *)u_qs;
        global const ushort * uscu = (global const ushort *)u_sc;

        for (uint ib = sgi; ib < nsb; ib += Q2K_MV_NSG) {
            const half4 gdm = vload4(j + ib * mr, g_dm);   // d0, dmin0, d1, dmin1
            const half4 udm = vload4(j + ib * mr, u_dm);

            float gad0 = 0.f, gam0 = 0.f, gad1 = 0.f, gam1 = 0.f;
            float uad0 = 0.f, uam0 = 0.f, uad1 = 0.f, uam1 = 0.f;

            for (uint sb = 0; sb < 8u; ++sb) {
                const uint grp = ib * 64u + sb * 8u;
                const uint qsb = j + grp * mr;

                for (uint h = 0; h < 2u; ++h) {
                    const uint gscv = (uint)gscu[j + (2u * (ib * 8u + sb) + h) * mr];
                    const uint uscv = (uint)uscu[j + (2u * (ib * 8u + sb) + h) * mr];

                    float ga0 = 0.f, ga1 = 0.f, ua0 = 0.f, ua1 = 0.f;
                    float4 as = (float4)(0.f);
                    for (uint u = 0; u < 4u; ++u) {
                        const uint gg = 4u*h + u;
                        // activation float4 read ONCE, and its running sum
                        // accumulated ONCE, for both streams and both rows
                        const float4 yv = vload4(grp + gg, y);
                        as += yv;

                        const uint gqsv = (uint)gqsu[qsb + gg * mr];
                        ga0 += dot(yv, q2k_vals( gqsv       & 0xFFu));
                        ga1 += dot(yv, q2k_vals((gqsv >> 8) & 0xFFu));

                        const uint uqsv = (uint)uqsu[qsb + gg * mr];
                        ua0 += dot(yv, q2k_vals( uqsv       & 0xFFu));
                        ua1 += dot(yv, q2k_vals((uqsv >> 8) & 0xFFu));
                    }
                    // ONE activation sum, four row-streams -- the point of the fusion
                    const float asum = as.s0 + as.s1 + as.s2 + as.s3;

                    const uint gs_0 = gscv & 0xFFu, gs_1 = gscv >> 8;
                    const uint us_0 = uscv & 0xFFu, us_1 = uscv >> 8;
                    gad0 += (float)(gs_0 & 0xFu) * ga0;  gam0 += (float)(gs_0 >> 4) * asum;
                    gad1 += (float)(gs_1 & 0xFu) * ga1;  gam1 += (float)(gs_1 >> 4) * asum;
                    uad0 += (float)(us_0 & 0xFu) * ua0;  uam0 += (float)(us_0 >> 4) * asum;
                    uad1 += (float)(us_1 & 0xFu) * ua1;  uam1 += (float)(us_1 >> 4) * asum;
                }
            }
            gs0 += (float)gdm.s0 * gad0 - (float)gdm.s1 * gam0;
            gs1 += (float)gdm.s2 * gad1 - (float)gdm.s3 * gam1;
            us0 += (float)udm.s0 * uad0 - (float)udm.s1 * uam0;
            us1 += (float)udm.s2 * uad1 - (float)udm.s3 * uam1;
        }
    }

#if Q2K_MV_NSG > 1
    __local float4 gpart[Q2K_MV_NSG][64];
    gpart[sgi][lid] = (float4)(gs0, gs1, us0, us1);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sgi != 0) {
        return;
    }
    for (uint s = 1; s < Q2K_MV_NSG; ++s) {
        const float4 p = gpart[s][lid];
        gs0 += p.s0; gs1 += p.s1; us0 += p.s2; us1 += p.s3;
    }
#endif

    if (j < mr) {
        global float * o = dst + (ulong)col * (uint)ne0 + row;
        o[0] = q2k_glu_apply(glu_op, gs0, us0);
        o[1] = q2k_glu_apply(glu_op, gs1, us1);
    }
}

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

// Q2K_MV_ABL: COST PROBE, WRONG MATH, default off. Prices the ACTIVATION LOAD by
// removing it while leaving the loop and every arithmetic operation in place.
//
// Why this term and why a probe first. This kernel is not compute bound
// (MV_WORK2 doubles the arithmetic for 1.3%), not bandwidth bound (tg64 runs at
// 34% of this part's roofline) and not starved (adding a workgroup K split was
// worth 1.0%). What is left is a dependent-load latency, and the activation read
// is the candidate: `grp` carries no row index, so every lane of a subgroup
// reads the SAME address, 64x over.
//
// Routing that read through a texture was +20.0% on IQ2_S and +21.6% on IQ1_S,
// but -3.7% on IQ4_XS -- the boundary is how heavy the kernel is per weight, and
// q2_K is light like IQ4_XS: a 2-bit unpack with no codebook gather. So it could
// go either way, and this probe decides it for one run instead of a build.
//
// The replacement value depends on the group index so the loop cannot be folded
// away. Never ship non-zero; read the tg delta as the ceiling on what optimising
// the activation read can buy.
//
// MEASURED, Llama-3.2-3B-Q2_K tg64: 39.08 -> 40.81, so the activation read is
// **4.4%** of decode. For comparison IQ3_S prices its own at 13.2% and the
// texture there delivered +10.5%. A 4.4% ceiling makes the texture worth at most
// two or three percent here, against a precedent (IQ4_XS, equally light) where
// the same change measured -3.7%. Not built on that basis.
//
// 🔑 And the arithmetic does not close: ALU 1.3% + activation 4.4% + the
// workgroup K split 1.0% is about 7%, against a kernel running at 34% of
// roofline. Roughly two thirds of the time is in neither the arithmetic nor the
// activation nor occupancy. The next instrument is a frame profile -- dispatch
// count, GPU busy against idle -- not another kernel micro-optimisation.
// Q2K_MV_PF=1: issue a half sub-block's four weight words before consuming any
// of them, rather than walking load -> unpack -> dot four times in series.
//
// plane_bw replays this kernel's exact plane addressing with no arithmetic and
// reaches 141 GB/s at the ffn shape -- 93% of this part's 152.4 GB/s roofline --
// while the kernel itself runs at 44%. So the layout, the multi-plane split and
// the hardware are all cleared; what is left is that each load is followed by
// dependent work before the next one issues, and the latency is exposed.
//
// MEASURED, and DEFAULT OFF. 3B Q2_K tg64, with kinfo alongside:
//
//   variant                                   private/WI  wg_cap    tg64
//   baseline, interleaved                            304     768   44.79
//   this, weights + 4 activations, balanced tree     560     384   45.50  +1.7%
//   weights only, activations one at a time,
//     serial accumulate                              448     512   41.75  -6.8%
//
// +1.7% is not worth crossing the 512-byte spill cliff: wg_cap falls to 384, so
// the kernel only still launches because NSG=4 asks for 256 work items, and any
// future widening would be refused outright.
//
// 🔑 THE ISOLATING EXPERIMENT WAS RUN, AND IT REFUTED THE OBVIOUS READING.
//
// From the first two variants it looked as though the heavy one won by breaking
// the accumulator dependency chain rather than by hoisting loads. PF=2 tests
// exactly that -- balanced summation, loads left where the baseline has them:
//
//   variant                                    priv/WI  wg_cap    tg64
//   PF=0  baseline, serial accumulate              304     768   44.68
//   PF=1  hoist + balanced tree                    560     384   45.41  +1.6%
//   PF=2  balanced tree, NO hoist                  448     512   41.23  -7.7%
//   (earlier) hoist, serial accumulate             448     512   41.75  -6.8%
//
// PF=2 and the earlier light variant changed OPPOSITE things -- one the
// summation, one the load order -- and landed on the SAME footprint (448) and
// the SAME loss (~-7%). So neither the summation shape nor the load order is
// the variable. **The variable is the register footprint.** 304 -> 448 crosses a
// cliff and costs more occupancy than either restructuring returns; PF=1 goes
// further to 560 and only nets +1.6% because its extra parallelism just about
// pays back the spill.
//
// ⇒ this kernel is REGISTER-CLIFF BOUND at the shape it runs. Adding live state
// loses regardless of what the state is for. The route to the plane_bw ceiling
// (93% of roofline for this access pattern) is to REDUCE the footprint below 304
// so more waves are resident, or to change the decomposition -- not to add ILP.
// Note R=2 halves the rows per lane and also loses badly (-18%), so simply
// shrinking the fold is not the answer either.
#ifndef Q2K_MV_PF
#define Q2K_MV_PF 0
#endif

#ifndef Q2K_MV_ABL
#define Q2K_MV_ABL 0
#endif

#if Q2K_MV_ABL == 1
#define Q2K_YV(g) ((float4)((float)((g) & 3u)))
#else
#define Q2K_YV(g) vload4((g), y)
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
#if Q2K_MV_PF == 2
                    // Break the ACCUMULATOR chain, hold nothing extra from memory.
                    //
                    // The baseline runs a0 += dot(...) four times in series, so each
                    // accumulator is a four-deep chain of dependent float adds, and
                    // the same for a1..a3. Two partials per accumulator halve that
                    // to two, and the loads stay exactly where the baseline has
                    // them -- one activation live at a time, no hoist.
                    //
                    // This is the variant the other two could not isolate: the
                    // heavy hoist changed BOTH the load order and the summation
                    // shape and gained 1.7%; the light one changed only the load
                    // order and lost 6.8%, which pointed at the summation. Costs
                    // eight floats, against sixteen for holding four activations.
                    float a0a = 0.f, a0b = 0.f, a1a = 0.f, a1b = 0.f;
                    float a2a = 0.f, a2b = 0.f, a3a = 0.f, a3b = 0.f;
                    #define Q2K_ACC(U, P)                                                                        {                                                                                                const uint   w  = qsu[qsb + (4u*h + (U)) * mr];                                              const float4 yv = Q2K_YV(grp + 4u*h + (U));                                                  as += yv;                                                                                    a0##P += dot(yv, q2k_vals( w        & 0xFFu));                                               a1##P += dot(yv, q2k_vals((w >>  8) & 0xFFu));                                               a2##P += dot(yv, q2k_vals((w >> 16) & 0xFFu));                                               a3##P += dot(yv, q2k_vals((w >> 24) & 0xFFu));                                           }
                    Q2K_ACC(0u, a)
                    Q2K_ACC(1u, b)
                    Q2K_ACC(2u, a)
                    Q2K_ACC(3u, b)
                    #undef Q2K_ACC
                    a0 = a0a + a0b;
                    a1 = a1a + a1b;
                    a2 = a2a + a2b;
                    a3 = a3a + a3b;
#elif Q2K_MV_PF == 1
                    // Issue this half's four weight words BEFORE consuming any of
                    // them, instead of load -> unpack -> dot four times in series.
                    // Hand unrolled so a compiler that declines to unroll cannot
                    // spill an indexed array to private memory.
                    //
                    // Why: plane_bw replays this exact addressing with no
                    // arithmetic and reaches 141 GB/s, 93% of this part's
                    // roofline, while the kernel itself runs at 44%. The layout is
                    // not the limit; the dependent chain load -> unpack -> dot is.
                    const uint w0 = qsu[qsb + (4u*h + 0u) * mr];
                    const uint w1 = qsu[qsb + (4u*h + 1u) * mr];
                    const uint w2 = qsu[qsb + (4u*h + 2u) * mr];
                    const uint w3 = qsu[qsb + (4u*h + 3u) * mr];
                    {
                        const float4 y0 = Q2K_YV(grp + 4u*h + 0u);
                        const float4 y1 = Q2K_YV(grp + 4u*h + 1u);
                        const float4 y2 = Q2K_YV(grp + 4u*h + 2u);
                        const float4 y3 = Q2K_YV(grp + 4u*h + 3u);
                        as = ((y0 + y1) + (y2 + y3));
                        a0 = ((dot(y0, q2k_vals( w0        & 0xFFu))  + dot(y1, q2k_vals( w1        & 0xFFu)))
                            + (dot(y2, q2k_vals( w2        & 0xFFu))  + dot(y3, q2k_vals( w3        & 0xFFu))));
                        a1 = ((dot(y0, q2k_vals((w0 >>  8) & 0xFFu))  + dot(y1, q2k_vals((w1 >>  8) & 0xFFu)))
                            + (dot(y2, q2k_vals((w2 >>  8) & 0xFFu))  + dot(y3, q2k_vals((w3 >>  8) & 0xFFu))));
                        a2 = ((dot(y0, q2k_vals((w0 >> 16) & 0xFFu))  + dot(y1, q2k_vals((w1 >> 16) & 0xFFu)))
                            + (dot(y2, q2k_vals((w2 >> 16) & 0xFFu))  + dot(y3, q2k_vals((w3 >> 16) & 0xFFu))));
                        a3 = ((dot(y0, q2k_vals((w0 >> 24) & 0xFFu))  + dot(y1, q2k_vals((w1 >> 24) & 0xFFu)))
                            + (dot(y2, q2k_vals((w2 >> 24) & 0xFFu))  + dot(y3, q2k_vals((w3 >> 24) & 0xFFu))));
                    }
#else
                    for (uint u = 0; u < 4u; ++u) {
                        const uint gg  = 4u*h + u;
                        const uint qsv = qsu[qsb + gg * mr];   // four rows, one load
                        const float4 yv = Q2K_YV(grp + gg);
                        as += yv;
                        a0 += dot(yv, q2k_vals( qsv        & 0xFFu));
#if MV_WORK2
                        a0 += dot(yv, q2k_vals((qsv + 1u)  & 0xFFu));
#endif
                        a1 += dot(yv, q2k_vals((qsv >>  8) & 0xFFu));
                        a2 += dot(yv, q2k_vals((qsv >> 16) & 0xFFu));
                        a3 += dot(yv, q2k_vals((qsv >> 24) & 0xFFu));
                    }
#endif
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
                        const float4 yv = Q2K_YV(grp + gg);
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
                        const float4 yv = Q2K_YV(grp + gg);
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

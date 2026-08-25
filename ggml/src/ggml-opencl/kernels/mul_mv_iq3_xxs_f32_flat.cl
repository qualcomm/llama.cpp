#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// IQ3_XXS decode GEMV over the feature-major plane split.
//
//   src0_qs [row + (k/4)*m]    uchar  grid index, K = 4*grp .. +3
//   src0_sas[row + (k/32)*m]   uint   scale in bits 28..31, four 7-bit sign codes
//   src0_d  [row + (k/256)*m]  half   super-block scale
//
// Structure copied from mul_mv_iq3_s_f32_flat, which is where both lessons were
// measured: K is split across IQ3XXS_MV_NSG subgroups because one row per lane
// leaves the GPU short of work items, and IQ3XXS_MV_R2 gives a lane two adjacent
// rows so the uchar quant plane is read a ushort at a time. Do NOT widen that to
// four rows -- it was tried on IQ3_S and LOSES, because it quarters the grid.

#define QK_K 256

#ifndef IQ3XXS_MV_NSG
#define IQ3XXS_MV_NSG 8
#endif

#ifndef IQ3XXS_MV_R2
#define IQ3XXS_MV_R2 1
#endif

// IQ3XXS_MV_LDSGRID=1: stage iq3xxs_grid into local memory once per workgroup and
// read it from there.
//
// iq3xxs_grid is 256 uints = 1 KB, the smallest of the split types' tables.
//
// MEASURED AND REFUTED: X2-90, q4b-IQ3_XXS tg64 15.57 -> 14.73, -5.3%. Stays
// off; see the IQ3_S GEMV header for why the table size decides this.
//
#ifndef IQ3XXS_MV_LDSGRID
#define IQ3XXS_MV_LDSGRID 0
#endif
constant uint iq3xxs_grid[256] = {
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04
};

// One iq3xxs_grid entry is a uint holding four values, i.e. exactly the four
// weights of K = 4u..4u+3, so a grid lookup IS one dp4a operand. Values top out
// at 62, so signed they are well inside int8.
//
// The sign byte comes from a 7-bit code, and ggml expands it through the
// ksigns_iq2xs[128] BYTE table. That table is exactly
//     ksigns_iq2xs[v] == v | ((popcount(v) & 1) << 7)
// (verified for all 128 entries), so this kernel computes it instead of reading
// it. That matters on Adreno beyond saving a KB: byte-indexed __constant loads
// serialize, which is why the codebook in the IQ4_XS kernels is packed into a
// uint array rather than indexed as bytes.
inline uint iq3xxs_signs(uint code7) {
    return code7 | ((uint)(popcount(code7) & 1u) << 7);
}

// Four grid values with their signs applied, packed for dp4a. base picks which
// nibble of the sign byte this operand uses.
inline uint iq3xxs_pack(uint gv, uint sg, uint base) {
    int v0 = (int)((gv >>  0) & 0xFF); if (sg & (1u << (base + 0))) { v0 = -v0; }
    int v1 = (int)((gv >>  8) & 0xFF); if (sg & (1u << (base + 1))) { v1 = -v1; }
    int v2 = (int)((gv >> 16) & 0xFF); if (sg & (1u << (base + 2))) { v2 = -v2; }
    int v3 = (int)((gv >> 24) & 0xFF); if (sg & (1u << (base + 3))) { v3 = -v3; }
    return ((uint)v0 & 0xFFu) | (((uint)v1 & 0xFFu) <<  8)
         | (((uint)v2 & 0xFFu) << 16) | (((uint)v3 & 0xFFu) << 24);
}

// Four grid values with their signs applied, as floats.
// IQ_MV_SIGNXOR=1: apply the per-weight signs by XOR-ing the float sign bit
// rather than by four conditional negations.
//
// Why: the IQ3_S cost probe (GGML_OPENCL_IQ3S_MV_ABL=3) says dropping the sign
// application entirely is worth 13.6 percent -- the second-largest cost in these
// kernels after the grid lookup, and ahead of the activation load at 5.5. The
// three grid types share this helper verbatim and are together about 30 percent
// of Qwen3.8-27B decode.
//
// MEASURED: q4b-IQ3_XXS tg64 15.46 -> 16.16 (+4.6%), and it holds at scale --
// Qwen3.8-27B UD-IQ3_XXS tg32 3.410 -> 3.468 (+1.7%, smaller because that file
// is a hybrid). On. The IDENTICAL helper is -4.7% in the IQ3_S GEMV, so this
// knob is per-kernel on purpose -- do not re-merge them.
#ifndef IQ3XXS_MV_SIGNXOR
#define IQ3XXS_MV_SIGNXOR 1
#endif

inline float4 iq3xxs_vals(uint gv, uint sgv, uint base) {
#if IQ3XXS_MV_SIGNXOR
    // A sign flip is bit 31, so the four conditional negations collapse to one
    // XOR once the four sign bits are spread into place. Exact, not approximate.
    const uint  s   = sgv >> base;
    const uint4 sgn = (uint4)(s << 31, s << 30, s << 29, s << 28) & 0x80000000u;
    return as_float4(as_uint4(convert_float4(as_uchar4(gv))) ^ sgn);
#else
    const uint s = sgv >> base;
    float4 v;
    v.s0 = (float)((gv      ) & 0xFFu); if (s & 1u) { v.s0 = -v.s0; }
    v.s1 = (float)((gv >>  8) & 0xFFu); if (s & 2u) { v.s1 = -v.s1; }
    v.s2 = (float)((gv >> 16) & 0xFFu); if (s & 4u) { v.s2 = -v.s2; }
    v.s3 = (float)((gv >> 24) & 0xFFu); if (s & 8u) { v.s3 = -v.s3; }
    return v;
#endif
}

// IQ3XXS_MV_GRIDIMG=1: read the grid through an image1d_buffer.
//
// This grid is still on __constant, where local memory was measured NEGATIVE
// (-16.5% on IQ3_S, -5.3% on IQ3_XXS -- the table is too small to earn the LDS
// traffic). The image is the remaining tier, and on the two kernels big enough to
// want LDS it is better than LDS: IQ1_S +2.5%, IQ2_S +4.4%. So it is worth asking
// here even though the LDS answer was no.
#ifndef IQ3XXS_MV_GRIDIMG
#define IQ3XXS_MV_GRIDIMG 0
#endif

// IQ3XXS_MV_AIMG=1: read the ACTIVATION through an image1d_buffer.
//
// Measured on the siblings: IQ2_S +20.0%, IQ3_S +10.5%, and IQ4_XS -3.7%. The
// boundary is how heavy the kernel is per weight, not how redundant the read is:
// the redundancy is identical everywhere (`grp` carries no row index, so every
// lane of a subgroup reads the SAME address, 64x over), but it only pays where
// that load is a large share of a large total. This type has a codebook gather
// and a delta term per 8 weights, which puts it on the winning side -- measured,
// not assumed.
//
// One CL_RGBA/CL_FLOAT texel IS the float4 the scalar path loads, over src1's own
// buffer, so there is no copy and no pre-pass. Applied to every kernel in the
// file, because whichever ones this type's fusion and split-K defaults route
// through are the ones carrying the frame.
#ifndef IQ3XXS_MV_AIMG
#define IQ3XXS_MV_AIMG 0
#endif

#if IQ3XXS_MV_AIMG
#define IQ3XXS_YV(g) read_imagef(y_img, (int)(y_tex + (g)))
#else
#define IQ3XXS_YV(g) vload4((g), y)
#endif

kernel void kernel_iq3xxs_grid_export(global uint * out) {
    const uint i = get_global_id(0);
    if (i < 256u) {
        out[i] = iq3xxs_grid[i];
    }
}

kernel void kernel_mul_mv_iq3_xxs_f32_flat(
        __read_only image1d_buffer_t grid_img,
        __read_only image1d_buffer_t y_img,   // see IQ3XXS_MV_AIMG
        uint y_off,                           // offset1/16, in float4 texels
        global const uchar * src0_qs,
        global const uint  * src0_sas,
        global const half  * src0_d,
        global const float * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,      // K
        int ne01,      // M, the number of output rows
        int ne10,      // activation row stride, == K
        int ne0        // dst row stride
) {
    src1 = (global const float *)((global const char *)src1 + offset1);
    dst  = (global float       *)((global char       *)dst  + offsetd);

    const uint m   = (uint)ne01;
    const uint K   = (uint)ne00;
    const uint nsb = K / QK_K;                  // super blocks along K

    const uint lid = get_local_id(0);           // lane
    const uint sgi = get_local_id(1);           // K-split slice
    const uint col = get_group_id(1);           // token

    global const float * y = src1 + (ulong)col * (uint)ne10;
#if IQ3XXS_MV_AIMG
    const uint y_tex = y_off + col * ((uint)ne10 >> 2);
#endif

#if IQ3XXS_MV_GRIDIMG
#define IQ3XXS_GRID(i) (read_imageui(grid_img, (int)(i)).x)
#elif IQ3XXS_MV_LDSGRID
    __local uint sh_grid[256];
    {
        const uint tid  = sgi * 64u + lid;
        const uint nthr = (uint)(get_local_size(0) * get_local_size(1));
        for (uint i = tid; i < 256u; i += nthr) {
            sh_grid[i] = iq3xxs_grid[i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#define IQ3XXS_GRID(i) sh_grid[(i)]
#else
#define IQ3XXS_GRID(i) iq3xxs_grid[(i)]
#endif

#if IQ3XXS_MV_R2
    const uint mh  = m >> 1;                        // rows per plane row, as pairs
    const uint j   = get_group_id(0) * 64u + lid;   // row pair index
    const uint row = j << 1;

    float sumf  = 0.f;
    float sumf1 = 0.f;

    if (j < mh) {
        global const ushort * qsu = (global const ushort *)src0_qs;

        for (uint ib = sgi; ib < nsb; ib += IQ3XXS_MV_NSG) {
            const half2 dh = vload2(j + ib * mh, src0_d);
            const float d0 = (float)dh.s0;
            const float d1 = (float)dh.s1;

            float acc0 = 0.f, acc1 = 0.f;
            for (uint sb = 0; sb < 8u; ++sb) {
                const uint  sub = ib * 8u + sb;
                const uint2 aux = vload2(j + sub * mh, src0_sas);   // row pair, one load

                const uint grp = sub * 8u;
                const uint qsb = j + grp * mh;

                float a0 = 0.f, a1 = 0.f;
                for (uint u = 0; u < 8u; ++u) {
                    const uint qsv  = (uint)qsu[qsb + u * mh];      // row pair, one load
                    const uint sh   = 7u * (u >> 1);
                    const uint base = (u & 1u) * 4u;
                    const float4 yv = IQ3XXS_YV(grp + u);
                    a0 += dot(yv, iq3xxs_vals(IQ3XXS_GRID( qsv        & 0xFFu),
                                              iq3xxs_signs((aux.s0 >> sh) & 127u), base));
                    a1 += dot(yv, iq3xxs_vals(IQ3XXS_GRID((qsv >> 8)  & 0xFFu),
                                              iq3xxs_signs((aux.s1 >> sh) & 127u), base));
                }
                acc0 += (0.5f + (float)(aux.s0 >> 28)) * a0;
                acc1 += (0.5f + (float)(aux.s1 >> 28)) * a1;
            }
            sumf  += d0 * 0.5f * acc0;
            sumf1 += d1 * 0.5f * acc1;
        }
    }
#else
    const uint row = get_group_id(0) * 64u + lid;

    float sumf = 0.f;

    if (row < m) {
        for (uint ib = sgi; ib < nsb; ib += IQ3XXS_MV_NSG) {
            const float d = (float)src0_d[row + ib * m];

            float acc = 0.f;
            for (uint sb = 0; sb < 8u; ++sb) {
                const uint sub = ib * 8u + sb;
                const uint aux = src0_sas[row + sub * m];

                const uint grp = sub * 8u;
                const uint qsb = row + grp * m;

                float a = 0.f;
                for (uint u = 0; u < 8u; ++u) {
                    const uint g    = (uint)src0_qs[qsb + u * m];
                    const uint sh   = 7u * (u >> 1);
                    const uint base = (u & 1u) * 4u;
                    const float4 yv = IQ3XXS_YV(grp + u);
                    a += dot(yv, iq3xxs_vals(IQ3XXS_GRID(g),
                                             iq3xxs_signs((aux >> sh) & 127u), base));
                }
                acc += (0.5f + (float)(aux >> 28)) * a;
            }
            sumf += d * 0.5f * acc;
        }
    }
#endif

#if IQ3XXS_MV_NSG > 1
#if IQ3XXS_MV_R2
    __local float2 part[IQ3XXS_MV_NSG][64];
    part[sgi][lid] = (float2)(sumf, sumf1);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sgi != 0) {
        return;
    }
    for (uint s = 1; s < IQ3XXS_MV_NSG; ++s) {
        const float2 p = part[s][lid];
        sumf  += p.s0;
        sumf1 += p.s1;
    }
#else
    __local float part[IQ3XXS_MV_NSG][64];
    part[sgi][lid] = sumf;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sgi != 0) {
        return;
    }
    for (uint s = 1; s < IQ3XXS_MV_NSG; ++s) {
        sumf += part[s][lid];
    }
#endif
#endif

#if IQ3XXS_MV_R2
    if (j < mh) {
        vstore2((float2)(sumf, sumf1), 0, dst + (ulong)col * (uint)ne0 + row);
    }
#else
    if (row < m) {
        dst[(ulong)col * (uint)ne0 + row] = sumf;
    }
#endif
}

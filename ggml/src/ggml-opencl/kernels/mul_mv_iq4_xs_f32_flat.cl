#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// IQ4_XS decode GEMV over the feature-major plane split.
//
// kernel_convert_block_iq4_xs_ns splits the tensor in place into four planes
// (the split is size preserving, 128 + 2 + 2 + 4 == 136 == sizeof(block_iq4_xs))
// and transposes each one, so every plane is indexed [k-group][row]:
//
//   src0_q [row + (k/4)*m]    ushort = 4 codebook indices, K = 4*grp .. +3
//   src0_d [row + (k/256)*m]  half   = super-block scale
//   src0_sh[row + (k/256)*m]  ushort = scales_h
//   src0_sl[row + (k/256)*m]  uint   = the four scales_l bytes
//
// One work item owns one output row, so lane i and lane i+1 read adjacent
// ushorts and a wave's weight read is one contiguous run. That is the whole
// reason this kernel exists: the AoS kernel_mul_mv_iq4_xs_f32 gives every lane
// a 16 byte slice at a 136 byte stride instead.
//
// 🔴 One row per lane on its own is 2.5x SLOWER than the AoS kernel, because it
// launches only M work items where the AoS one launches 16*M and the GPU is
// left with ~48 waves. K is therefore split across IQ4XS_MV_NSG subgroups whose
// partials are reduced through local memory -- the same trick, and the same
// reason, as kernel_gemv_noshuffle_q4_k_f32's lws=64xNSG.
//
// The sub-scale is rebuilt exactly as the dp4a GEMM does:
//   ls  = scales_l nibble | ((scales_h >> 2*sb) & 3) << 4
//   d_w = d * (ls - 32)

#define QK_K 256

// Subgroups per workgroup; each takes every NSG'th super-block along K.
#ifndef IQ4XS_MV_NSG
#define IQ4XS_MV_NSG 8
#endif

constant float kvalues_iq4nl[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f,
      1.f,   13.f,  25.f,  38.f,  53.f,  69.f,  89.f, 113.f
};

kernel void kernel_mul_mv_iq4_xs_f32_flat(
        global const ushort * src0_q,
        global const half   * src0_d,
        global const ushort * src0_sh,
        global const uint   * src0_sl,
        global const float  * src1,
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

    const uint lid = get_local_id(0);           // lane == row within the block
    const uint sg  = get_local_id(1);           // K-split slice
    const uint row = get_group_id(0) * 64u + lid;
    const uint col = get_group_id(1);           // token

    global const float * y = src1 + (ulong)col * (uint)ne10;

    float sumf = 0.f;

    if (row < m) {
        for (uint ib = sg; ib < nsb; ib += IQ4XS_MV_NSG) {
            const uint  base = row + ib * m;
            const uint  slv  = src0_sl[base];
            const uint  shv  = (uint)src0_sh[base];
            const float d    = (float)src0_d[base];

            float acc = 0.f;
            for (uint sb = 0; sb < 8u; ++sb) {
                const int ls = (int)(((slv >> (8u * (sb >> 1))) >> (4u * (sb & 1u))) & 0xFu)
                             | (int)(((shv >> (2u * sb)) & 3u) << 4);

                // k/4 index of this 32-element sub block
                const uint grp = ib * 64u + sb * 8u;
                const uint qb  = row + grp * m;

                float a = 0.f;
                for (uint u = 0; u < 8u; ++u) {
                    const ushort w  = src0_q[qb + u * m];
                    const float4 yv = vload4(grp + u, y);
                    a += yv.s0 * kvalues_iq4nl[(w      ) & 0xF];
                    a += yv.s1 * kvalues_iq4nl[(w >>  4) & 0xF];
                    a += yv.s2 * kvalues_iq4nl[(w >>  8) & 0xF];
                    a += yv.s3 * kvalues_iq4nl[(w >> 12) & 0xF];
                }
                acc += (float)(ls - 32) * a;
            }
            sumf += d * acc;
        }
    }

#if IQ4XS_MV_NSG > 1
    __local float part[IQ4XS_MV_NSG][64];
    part[sg][lid] = sumf;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sg != 0) {
        return;
    }
    for (uint s = 1; s < IQ4XS_MV_NSG; ++s) {
        sumf += part[s][lid];
    }
#endif

    if (row < m) {
        dst[(ulong)col * (uint)ne0 + row] = sumf;
    }
}

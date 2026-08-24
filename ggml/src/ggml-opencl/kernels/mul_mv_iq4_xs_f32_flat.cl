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
// The sub-scale is rebuilt exactly as the dp4a GEMM does:
//   ls  = scales_l nibble | ((scales_h >> 2*sb) & 3) << 4
//   d_w = d * (ls - 32)

#define QK_K 256

// Rows each work item accumulates, strided by 64 so every row still lands in a
// coalesced wave read. More rows means more loads in flight per lane.
#ifndef IQ4XS_MV_ROWS
#define IQ4XS_MV_ROWS 1
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

    // Each workgroup owns 64*IQ4XS_MV_ROWS consecutive rows; derive the base from
    // the group id, NOT from get_global_id(0), or the row blocks overlap when
    // IQ4XS_MV_ROWS > 1.
    const uint row0 = get_group_id(0) * (64u * IQ4XS_MV_ROWS) + get_local_id(0);
    const uint col  = get_group_id(1);          // token

    global const float * y = src1 + (ulong)col * (uint)ne10;

    float sumf[IQ4XS_MV_ROWS];
    for (int j = 0; j < IQ4XS_MV_ROWS; ++j) {
        sumf[j] = 0.f;
    }

    for (uint ib = 0; ib < nsb; ++ib) {
        for (int j = 0; j < IQ4XS_MV_ROWS; ++j) {
            const uint row = row0 + (uint)j * 64u;
            if (row >= m) {
                continue;
            }

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
            sumf[j] += d * acc;
        }
    }

    for (int j = 0; j < IQ4XS_MV_ROWS; ++j) {
        const uint row = row0 + (uint)j * 64u;
        if (row < m) {
            dst[(ulong)col * (uint)ne0 + row] = sumf[j];
        }
    }
}

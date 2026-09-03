// cok-shaped q5_K GEMM with a dp4a inner product at EIGHT columns, for ne1 = 5..8.
//
// The q4_0 twin (gemm_cok8_q4_0_q8_1_dp4a.cl) replaced the f16 r4 cok on the DFlash2
// verify width and this is the same trade for q5_K, which Qwen3.8-27B-Q4_0 carries on
// its ssm_out projection (5120 x 6144, 48 launches per verify round). The f16 r4_splitk
// kernel runs that shape at 62 GB/s of a 152 GB/s bus: per row per 4 K it unpacks two
// planes (nibble and high bit) with ~9 ops per element in front of 16 half2 FMA slots.
// Here the same 4 K cost 8 dp4a plus ~7 unpack ops.
//
// What is specific to q5_K:
//  - the weight word for four K is built from the ushort nibble group and the qh nibble
//    without a per-element loop: (u | (u >> 4) << 16) & 0x0F0F0F0F spreads the nibbles to bytes
//    (n0, n2, n1, n3), and (h * 0x02080410) & 0x10101010 lands the four high bits on
//    bit 4 of the same bytes. Codes stay 0..31, unsigned, so the min term is applied at
//    the flush like the dense q5_K dp4a kernel: acc += sc*da*dot - mn*sa.
//  - the activation quant (kernel_quant_a_q8_1_k4) stores each 4-K group in that
//    (K0, K2, K1, K3) byte order, and the per-block scale d and sum d*Sum(q) as two
//    half8 texels per block, column-interleaved at width 8.
//  - scales/mins are 12 packed bytes per (row, superblock), ROW-major, so a lane walks
//    its K slice a superblock at a time: three uint loads per row per superblock, and
//    the (scale, min) of each 32-K sub-block is a wave-uniform byte pick. Slices are
//    contiguous runs of 32-K blocks (not the interleave of the q4_0 twin) so a run
//    usually covers whole superblocks; the sub-block range per superblock is uniform.
//  - the K-split geometry (COK_NSG subgroups in the workgroup, ksplit slices across
//    workgroups writing partials for kernel_gemv_splitk_reduce_f32) is the one the f16
//    r4_splitk kernel launches, so the host dispatch mirrors that one.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#ifdef cl_khr_integer_dot_product
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#endif

#ifndef COK_NSG
#define COK_NSG 6
#endif
#define COK_SG   64
#define COK_ROWS 4

#define COK_PACK4(a, b, c, e)                                         \
    ( ((uint)(a) & 0xFFu) | (((uint)(b) & 0xFFu) << 8)               \
    | (((uint)(c) & 0xFFu) << 16) | (((uint)(e) & 0xFFu) << 24) )

// q8_1 activation quantization in the byte order this file's GEMM reads. One work-item
// per 32-K block; column c, block b is global block c*k_b + b. Every 4-K group is stored
// as the bytes (K0, K2, K1, K3). ds holds two half8 texels per block: [2b] the eight
// column scales d, [2b+1] the eight column sums d*Sum(q); the host allocates at width 8.
kernel void kernel_quant_a_q8_1_k4(
    global const float * src,     // [N * K] f32, K contiguous per column
    global uint        * qa,      // [N * K / 4] int8 x4, (0 2 1 3) order per 4 K
    global half        * ds,      // [K/32 * 16] block scale and sum, width 8
    int total_blocks,             // N * (K/32)
    int k_b                       // K/32
) {
    const int blk = get_global_id(0);
    if (blk >= total_blocks) {
        return;
    }

    const int base = blk * 32;

    float v[32];
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        v[i] = src[base + i];
        amax = fmax(amax, fabs(v[i]));
    }

    const float d  = amax / 127.0f;
    const float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;

    int q[32];
    int sum = 0;
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        q[i] = (int)rint(v[i] * id);
        sum += q[i];
    }

    uint8 out;
    out.s0 = COK_PACK4(q[ 0], q[ 2], q[ 1], q[ 3]);
    out.s1 = COK_PACK4(q[ 4], q[ 6], q[ 5], q[ 7]);
    out.s2 = COK_PACK4(q[ 8], q[10], q[ 9], q[11]);
    out.s3 = COK_PACK4(q[12], q[14], q[13], q[15]);
    out.s4 = COK_PACK4(q[16], q[18], q[17], q[19]);
    out.s5 = COK_PACK4(q[20], q[22], q[21], q[23]);
    out.s6 = COK_PACK4(q[24], q[26], q[25], q[27]);
    out.s7 = COK_PACK4(q[28], q[30], q[29], q[31]);
    vstore8(out, 0, qa + blk * 8);

    const int col = blk / k_b;
    const int b   = blk - col * k_b;
    ds[(2 * b)     * 8 + col] = (half)d;
    ds[(2 * b + 1) * 8 + col] = (half)(d * (float)sum);
}

// Four K of one row as one dp4a word. u: the ushort nibble group, nibble i = K i.
// h: the four qh bits of the same K, bit i = K i. Bytes come out as (K0, K2, K1, K3).
// The multiply places h<<4, h<<10, h<<19, h<<25, which do not overlap, so the four
// masked bits are exactly h.s0..3 with no carries between them.
// The nibble spread is the 16-bit pack of u and u >> 4, not u | u << 12: the Adreno 840
// compiler evaluates that shift of a ushort at 16 bits and zeroes the two odd K.
#define COK_LO(u) (((uint)(u) | ((uint)(ushort)((u) >> 4) << 16)) & 0x0F0F0F0Fu)
#define COK_HI(h) ((((uint)(h)) * 0x02080410u) & 0x10101010u)

// Row r: K-groups 0..3 of the half block (bl0..bl3), qh bytes hq0 (groups 0, 1) and
// hq1 (groups 2, 3), low nibble for the even group.
#define COK_UNPACK(r)                                                              \
    const uint w0_##r = COK_LO(bl0.s##r) | COK_HI(hq0.s##r & 0xFu);                \
    const uint w1_##r = COK_LO(bl1.s##r) | COK_HI(hq0.s##r >> 4);                  \
    const uint w2_##r = COK_LO(bl2.s##r) | COK_HI(hq1.s##r & 0xFu);                \
    const uint w3_##r = COK_LO(bl3.s##r) | COK_HI(hq1.s##r >> 4);

// Row r against column c over the same sixteen K, the column's texel in A.
#define COK_DOT_RC(r, c)                                                           \
    d##r.s##c = dot_acc_sat_4x8packed_ss_int(w0_##r, A.s0, d##r.s##c);             \
    d##r.s##c = dot_acc_sat_4x8packed_ss_int(w1_##r, A.s1, d##r.s##c);             \
    d##r.s##c = dot_acc_sat_4x8packed_ss_int(w2_##r, A.s2, d##r.s##c);             \
    d##r.s##c = dot_acc_sat_4x8packed_ss_int(w3_##r, A.s3, d##r.s##c);

// Column-outer: one activation texel at a time against the unpacked rows. The wave-
// uniform operands (activation texel, block scale and sum) go through the texture path;
// the same bytes as a buffer load cost one L1 transaction per lane (-40% on the q4_0 twin).
#define COK_DOT_COL(c, t)                                                          \
    { const uint4 A = read_imageui(src1_qa, (c) * k_t + (t));                      \
      COK_DOT_RC(0, c) COK_DOT_RC(1, c) COK_DOT_RC(2, c) COK_DOT_RC(3, c) }

// (scale, min) codes of sub-block j from the 12 packed bytes held as the uint3 s, the
// get_scale_min_k4 rule with sh = 8*(j%4) and j wave-uniform: j < 4 reads bytes j and
// j+4 (6 bits each); j >= 4 reads the low/high nibble of byte j+4 and the top two bits
// of bytes j-4 / j. j is a loop counter, not a template index: unrolling the eight
// sub-blocks to make the byte picks compile-time constants replicated the 256-dp4a
// body sixteen times and ran 4x slower than the f16 kernel it replaces.
#define COK_SCM(s, sc, mn)                                                         \
    if (j < 4) {                                                                   \
        sc = ((s).s0 >> sh) & md6;                                                 \
        mn = ((s).s1 >> sh) & md6;                                                 \
    } else {                                                                       \
        const uint b2 = ((s).s2 >> sh) & 0xFFu;                                    \
        sc = (b2 & md4) | ((((s).s0 >> sh) & mh2) >> 2);                           \
        mn = (b2 >> 4)  | ((((s).s1 >> sh) & mh2) >> 2);                           \
    }

kernel void kernel_gemm_cok8_q5_k_q8_1_dp4a(
    global const ushort * src0_q,     // q5_K nibble plane [row + (K/4)*m]
    global const uchar  * src0_qh,    // q5_K high-bit plane [row + (K/8)*m]
    global const uchar  * src0_s,     // packed scales/mins [row][superblock][12]
    global const half   * src0_d,     // superblock scale [row + sb*m]
    global const half   * src0_dm,    // superblock min   [row + sb*m]
    read_only image1d_buffer_t src1_qa,  // q8_1 activations, (0 2 1 3) order, RGBA32UI texels [col*K/16 + K/16]
    read_only image1d_buffer_t src1_ds,  // activation scale / sum, half8 texels [2*blk], [2*blk+1]
    global float * dst,               // ksplit == 1: [n][m] output; else [ksplit][n][m] partials
    ulong offsetd,
    int m,
    int k,
    int n_no_padding,
    uchar mask_d6,
    uchar mask_d4,
    uchar mask_hi2,
    int ksplit
) {
    dst = (global float *)((global char *)dst + offsetd);

    const int gx   = get_global_id(0);
    const int sg   = get_local_id(1);
    const int lane = get_local_id(0);
    const int ks   = get_group_id(1);

    const int row0 = gx * COK_ROWS;
    const int num_32blk = k / 32;
    const int nsb = k / 256;                      // superblocks per row
    const int k_t = k >> 4;                       // activation texels (16 K) per column
    const uint md6 = mask_d6;
    const uint md4 = mask_d4;
    const uint mh2 = mask_hi2;
    const int  fence = (int)(md6 >> 6);            // 0 at run time, unknown to the compiler

    // This subgroup's K slice: a contiguous run of 32-K blocks, ksplit*COK_NSG slices
    // over the whole K, walked a superblock at a time.
    const int nslice = ksplit * COK_NSG;
    const int chunk  = (num_32blk + nslice - 1) / nslice;
    const int b_beg  = (ks * COK_NSG + sg) * chunk;
    const int b_end  = min(b_beg + chunk, num_32blk);

    global const uint * sw = (global const uint *)src0_s;   // 12-byte groups, 4-byte aligned

    float8 acc0 = (float8)(0.0f), acc1 = (float8)(0.0f);
    float8 acc2 = (float8)(0.0f), acc3 = (float8)(0.0f);

    for (int sb = b_beg >> 3; sb * 8 < b_end; ++sb) {
        const uint3 s0 = vload3(0, sw + ((row0 + 0) * nsb + sb) * 3);
        const uint3 s1 = vload3(0, sw + ((row0 + 1) * nsb + sb) * 3);
        const uint3 s2 = vload3(0, sw + ((row0 + 2) * nsb + sb) * 3);
        const uint3 s3 = vload3(0, sw + ((row0 + 3) * nsb + sb) * 3);
        const half4 dd  = vload4(0, src0_d  + row0 + sb * m);
        const half4 dmm = vload4(0, src0_dm + row0 + sb * m);

        // The slice's sub-blocks of this superblock; both bounds are wave-uniform.
        const int j_beg = max(b_beg - sb * 8, 0);
        const int j_end = min(b_end - sb * 8, 8);

        for (int j = j_beg; j < j_end; ++j) {
            const int blk = sb * 8 + j;
            int8 d0 = (int8)(0), d1 = (int8)(0), d2 = (int8)(0), d3 = (int8)(0);

            for (int h = 0; h < 2; ++h) {
                const int g0 = (blk << 3) + (h << 2);     // first K-group of this half block
                const int at = (blk << 1) + h;            // its texel in the activation
                const ushort4 bl0 = vload4(0, src0_q  + row0 + (g0 + 0) * m);
                const ushort4 bl1 = vload4(0, src0_q  + row0 + (g0 + 1) * m);
                const ushort4 bl2 = vload4(0, src0_q  + row0 + (g0 + 2) * m);
                const ushort4 bl3 = vload4(0, src0_q  + row0 + (g0 + 3) * m);
                const uchar4  hq0 = vload4(0, src0_qh + row0 + ((g0 >> 1) + 0) * m);
                const uchar4  hq1 = vload4(0, src0_qh + row0 + ((g0 >> 1) + 1) * m);
                COK_UNPACK(0) COK_UNPACK(1) COK_UNPACK(2) COK_UNPACK(3)
                // Two column groups of four. The second group's texel index carries a
                // dependency on the first group's dot (fence is 0, so the value is at),
                // which keeps the compiler from issuing all eight texel reads up front:
                // with the reads hoisted this shape ran 307 us, gated 211 us, at the same
                // register count. One gate per column (four texels in flight at most)
                // fits the 384-work-item cap but serialises the reads: 286 us.
                COK_DOT_COL(0, at) COK_DOT_COL(1, at) COK_DOT_COL(2, at) COK_DOT_COL(3, at)
                const int at4 = at + (d0.s0 & fence);
                COK_DOT_COL(4, at4) COK_DOT_COL(5, at4) COK_DOT_COL(6, at4) COK_DOT_COL(7, at4)
            }

            // Flush: acc += (d*sc) * da * dot - (dm*mn) * sa, the scale term for all
            // four rows before the min term so one activation texel is live at a time.
            const uint sh = (uint)(j & 3) * 8u;
            uint sc0, mn0, sc1, mn1, sc2, mn2, sc3, mn3;
            COK_SCM(s0, sc0, mn0) COK_SCM(s1, sc1, mn1)
            COK_SCM(s2, sc2, mn2) COK_SCM(s3, sc3, mn3)
            {
                const float8 da = convert_float8(as_half8(read_imageui(src1_ds, 2 * blk)));
                acc0 = mad(convert_float8(d0), da * ((float)dd.s0 * (float)sc0), acc0);
                acc1 = mad(convert_float8(d1), da * ((float)dd.s1 * (float)sc1), acc1);
                acc2 = mad(convert_float8(d2), da * ((float)dd.s2 * (float)sc2), acc2);
                acc3 = mad(convert_float8(d3), da * ((float)dd.s3 * (float)sc3), acc3);
            }
            {
                const float8 sa = convert_float8(as_half8(read_imageui(src1_ds, 2 * blk + 1)));
                acc0 = mad(sa, -((float)dmm.s0 * (float)mn0), acc0);
                acc1 = mad(sa, -((float)dmm.s1 * (float)mn1), acc1);
                acc2 = mad(sa, -((float)dmm.s2 * (float)mn2), acc2);
                acc3 = mad(sa, -((float)dmm.s3 * (float)mn3), acc3);
            }
        }
    }

    // Cross-subgroup reduction over the in-workgroup K-split, one row at a time.
    local float8 reduceLM[COK_SG * (COK_NSG - 1)];
    float8 out0 = (float8)(0.0f), out1 = (float8)(0.0f);
    float8 out2 = (float8)(0.0f), out3 = (float8)(0.0f);

#define COK_REDUCE(accv, outv)                                       \
    barrier(CLK_LOCAL_MEM_FENCE);                                    \
    if (sg > 0) { reduceLM[(sg - 1) * COK_SG + lane] = (accv); }     \
    barrier(CLK_LOCAL_MEM_FENCE);                                    \
    if (sg == 0) {                                                   \
        float8 sum = (accv);                                         \
        for (int s = 0; s < COK_NSG - 1; s++) {                      \
            sum += reduceLM[s * COK_SG + lane];                      \
        }                                                            \
        (outv) = sum;                                                \
    }

    COK_REDUCE(acc0, out0)
    COK_REDUCE(acc1, out1)
    COK_REDUCE(acc2, out2)
    COK_REDUCE(acc3, out3)
#undef COK_REDUCE

    // Columns beyond n_no_padding were computed on whatever the activation buffer held
    // past the quantized region and are dropped here. Slice ks writes its own [n][m] plane.
#define COK_STORE_COL(c)                                                                  \
    if ((c) < n_no_padding) {                                                             \
        vstore4((float4)(out0.s##c, out1.s##c, out2.s##c, out3.s##c), 0,                  \
                dst + (c) * m + row0);                                                    \
    }

    if (sg == 0) {
        dst += (size_t)ks * (size_t)m * (size_t)n_no_padding;
        COK_STORE_COL(0) COK_STORE_COL(1) COK_STORE_COL(2) COK_STORE_COL(3)
        COK_STORE_COL(4) COK_STORE_COL(5) COK_STORE_COL(6) COK_STORE_COL(7)
    }
#undef COK_STORE_COL
}

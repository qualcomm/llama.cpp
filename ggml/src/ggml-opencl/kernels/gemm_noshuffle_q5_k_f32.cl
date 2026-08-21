#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
#endif
#define QK_K         256
#define K_SCALE_SIZE 12

inline void get_scale_min_k4(
    int j,
    global const uchar * q,
    uchar * d,
    uchar * m,
    uchar mask_d6,
    uchar mask_d4,
    uchar mask_hi2
) {
    if (j < 4) {
        *d = q[j]   & mask_d6;
        *m = q[j+4] & mask_d6;
    } else {
        *d = (q[j+4] & mask_d4) | ((q[j-4] & mask_hi2) >> 2);
        *m = ((q[j+4] >> 4) & mask_d4) | ((q[j]   & mask_hi2) >> 2);
    }
}

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_128
#endif
kernel void kernel_gemm_noshuffle_q5_k_f32(
    global const ushort * src0_q,
    global const uchar  * src0_qh,
    global const uchar  * src0_s,
    global const half   * src0_d,
    global const half   * src0_dm,
    read_only image1d_buffer_t src1,
    global float * dst,
    ulong offsetd,
    int m,
    int n,
    int k,
    int n_no_padding,
    uchar mask_d6,
    uchar mask_d4,
    uchar mask_hi2
) {
    dst = (global float *)((global char *)dst + offsetd);
    int n_4 = n >> 2;
    int gy = get_global_id(0);
    int gx = get_global_id(1);
    int gx_2 = gx << 2;

    half8 c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    half8 B;
    half4 dequantized_weights;

    int num_blocks_K = k / QK_K;

    global const ushort * weight_ptr = src0_q  + gx_2;
    global const uchar  * qh_ptr     = src0_qh + gx_2;
    global const half   * d_ptr      = src0_d  + gx_2;
    global const half   * dm_ptr     = src0_dm + gx_2;

    for (int i = 0; i < k; i += 32) {
        int sb_idx  = i / QK_K;
        int sub_idx = (i / 32) % 8;

        half4 d  = vload4(0, d_ptr  + sb_idx * m);
        half4 dm = vload4(0, dm_ptr + sb_idx * m);

        global const uchar * sc0 = src0_s + (gx_2+0) * num_blocks_K * K_SCALE_SIZE + sb_idx * K_SCALE_SIZE;
        global const uchar * sc1 = src0_s + (gx_2+1) * num_blocks_K * K_SCALE_SIZE + sb_idx * K_SCALE_SIZE;
        global const uchar * sc2 = src0_s + (gx_2+2) * num_blocks_K * K_SCALE_SIZE + sb_idx * K_SCALE_SIZE;
        global const uchar * sc3 = src0_s + (gx_2+3) * num_blocks_K * K_SCALE_SIZE + sb_idx * K_SCALE_SIZE;

        uchar sv0, mn0, sv1, mn1, sv2, mn2, sv3, mn3;
        get_scale_min_k4(sub_idx, sc0, &sv0, &mn0, mask_d6, mask_d4, mask_hi2);
        get_scale_min_k4(sub_idx, sc1, &sv1, &mn1, mask_d6, mask_d4, mask_hi2);
        get_scale_min_k4(sub_idx, sc2, &sv2, &mn2, mask_d6, mask_d4, mask_hi2);
        get_scale_min_k4(sub_idx, sc3, &sv3, &mn3, mask_d6, mask_d4, mask_hi2);

        half4 scale = convert_half4(convert_float4(d)  * convert_float4((uchar4)(sv0, sv1, sv2, sv3)));
        half4 mval  = convert_half4(convert_float4(dm) * convert_float4((uchar4)(mn0, mn1, mn2, mn3)));

        for (int l = 0; l < 32; l += 4) {
            int ki = i + l;
            ushort4 bits4   = vload4(0, weight_ptr + (ki/4) * m);
            uchar4  qh_bits = vload4(0, qh_ptr     + (ki/8) * m);
            int     qh_shift = ki % 8;

            // j=0
            B.s0123 = read_imageh(src1, gy*2   + (ki+0) * n_4);
            B.s4567 = read_imageh(src1, gy*2+1 + (ki+0) * n_4);
            dequantized_weights.s0 = ((bits4.s0 & 0x000F) | (((qh_bits.s0 >> (qh_shift+0)) & 1) << 4)) * scale.s0 - mval.s0;
            dequantized_weights.s1 = ((bits4.s1 & 0x000F) | (((qh_bits.s1 >> (qh_shift+0)) & 1) << 4)) * scale.s1 - mval.s1;
            dequantized_weights.s2 = ((bits4.s2 & 0x000F) | (((qh_bits.s2 >> (qh_shift+0)) & 1) << 4)) * scale.s2 - mval.s2;
            dequantized_weights.s3 = ((bits4.s3 & 0x000F) | (((qh_bits.s3 >> (qh_shift+0)) & 1) << 4)) * scale.s3 - mval.s3;
            c0 += B * dequantized_weights.s0;
            c1 += B * dequantized_weights.s1;
            c2 += B * dequantized_weights.s2;
            c3 += B * dequantized_weights.s3;

            // j=1
            B.s0123 = read_imageh(src1, gy*2   + (ki+1) * n_4);
            B.s4567 = read_imageh(src1, gy*2+1 + (ki+1) * n_4);
            dequantized_weights.s0 = (((bits4.s0 & 0x00F0) >> 4) | (((qh_bits.s0 >> (qh_shift+1)) & 1) << 4)) * scale.s0 - mval.s0;
            dequantized_weights.s1 = (((bits4.s1 & 0x00F0) >> 4) | (((qh_bits.s1 >> (qh_shift+1)) & 1) << 4)) * scale.s1 - mval.s1;
            dequantized_weights.s2 = (((bits4.s2 & 0x00F0) >> 4) | (((qh_bits.s2 >> (qh_shift+1)) & 1) << 4)) * scale.s2 - mval.s2;
            dequantized_weights.s3 = (((bits4.s3 & 0x00F0) >> 4) | (((qh_bits.s3 >> (qh_shift+1)) & 1) << 4)) * scale.s3 - mval.s3;
            c0 += B * dequantized_weights.s0;
            c1 += B * dequantized_weights.s1;
            c2 += B * dequantized_weights.s2;
            c3 += B * dequantized_weights.s3;

            // j=2
            B.s0123 = read_imageh(src1, gy*2   + (ki+2) * n_4);
            B.s4567 = read_imageh(src1, gy*2+1 + (ki+2) * n_4);
            dequantized_weights.s0 = (((bits4.s0 & 0x0F00) >> 8) | (((qh_bits.s0 >> (qh_shift+2)) & 1) << 4)) * scale.s0 - mval.s0;
            dequantized_weights.s1 = (((bits4.s1 & 0x0F00) >> 8) | (((qh_bits.s1 >> (qh_shift+2)) & 1) << 4)) * scale.s1 - mval.s1;
            dequantized_weights.s2 = (((bits4.s2 & 0x0F00) >> 8) | (((qh_bits.s2 >> (qh_shift+2)) & 1) << 4)) * scale.s2 - mval.s2;
            dequantized_weights.s3 = (((bits4.s3 & 0x0F00) >> 8) | (((qh_bits.s3 >> (qh_shift+2)) & 1) << 4)) * scale.s3 - mval.s3;
            c0 += B * dequantized_weights.s0;
            c1 += B * dequantized_weights.s1;
            c2 += B * dequantized_weights.s2;
            c3 += B * dequantized_weights.s3;

            // j=3
            B.s0123 = read_imageh(src1, gy*2   + (ki+3) * n_4);
            B.s4567 = read_imageh(src1, gy*2+1 + (ki+3) * n_4);
            dequantized_weights.s0 = (((bits4.s0 & 0xF000) >> 12) | (((qh_bits.s0 >> (qh_shift+3)) & 1) << 4)) * scale.s0 - mval.s0;
            dequantized_weights.s1 = (((bits4.s1 & 0xF000) >> 12) | (((qh_bits.s1 >> (qh_shift+3)) & 1) << 4)) * scale.s1 - mval.s1;
            dequantized_weights.s2 = (((bits4.s2 & 0xF000) >> 12) | (((qh_bits.s2 >> (qh_shift+3)) & 1) << 4)) * scale.s2 - mval.s2;
            dequantized_weights.s3 = (((bits4.s3 & 0xF000) >> 12) | (((qh_bits.s3 >> (qh_shift+3)) & 1) << 4)) * scale.s3 - mval.s3;
            c0 += B * dequantized_weights.s0;
            c1 += B * dequantized_weights.s1;
            c2 += B * dequantized_weights.s2;
            c3 += B * dequantized_weights.s3;
        }
    }

    int idx = (gy<<3)*m + (gx<<2);

    if (idx+3 < m*n_no_padding) {
        vstore4((float4)(c0.s0, c1.s0, c2.s0, c3.s0), 0, dst + idx);
        idx += m;
    }
    if (idx+3 < m*n_no_padding) {
        vstore4((float4)(c0.s1, c1.s1, c2.s1, c3.s1), 0, dst + idx);
        idx += m;
    }
    if (idx+3 < m*n_no_padding) {
        vstore4((float4)(c0.s2, c1.s2, c2.s2, c3.s2), 0, dst + idx);
        idx += m;
    }
    if (idx+3 < m*n_no_padding) {
        vstore4((float4)(c0.s3, c1.s3, c2.s3, c3.s3), 0, dst + idx);
        idx += m;
    }
    if (idx+3 < m*n_no_padding) {
        vstore4((float4)(c0.s4, c1.s4, c2.s4, c3.s4), 0, dst + idx);
        idx += m;
    }
    if (idx+3 < m*n_no_padding) {
        vstore4((float4)(c0.s5, c1.s5, c2.s5, c3.s5), 0, dst + idx);
        idx += m;
    }
    if (idx+3 < m*n_no_padding) {
        vstore4((float4)(c0.s6, c1.s6, c2.s6, c3.s6), 0, dst + idx);
        idx += m;
    }
    if (idx+3 < m*n_no_padding) {
        vstore4((float4)(c0.s7, c1.s7, c2.s7, c3.s7), 0, dst + idx);
    }
}

// Cooperative-K GEMM for the small-batch (n_q in [2..8]) path. Identical
// structure to kernel_gemm_noshuffle_q4_k_f32_cok (WG = COK_SG lanes x
// COK_NSG subgroups; each lane owns ONE output row and its 8 padded columns,
// the COK_NSG subgroups split the K reduction round-robin and combine via a
// __local float8 reduction) plus the q5_K high-bit (qh) plane folded into each
// dequant. Closes the q5_K medium-batch dead-zone the same way q4_K/q6_K cok
// did; q5_K is the #2 verify chunk on Qwen3.5 spec/MTP. REQD_SUBGROUP_SIZE_64
// + barrier (never full-width sub_group_reduce on X2).
#define COK_NSG 8
#define COK_SG  64
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_gemm_noshuffle_q5_k_f32_cok(
    global const ushort * src0_q,
    global const uchar  * src0_qh,
    global const uchar  * src0_s,
    global const half   * src0_d,
    global const half   * src0_dm,
    read_only image1d_buffer_t src1,
    global float * dst,
    ulong offsetd,
    int m,
    int n,
    int k,
    int n_no_padding,
    uchar mask_d6,
    uchar mask_d4,
    uchar mask_hi2
) {
    dst = (global float *)((global char *)dst + offsetd);
    int n_4  = n >> 2;
    int gx   = get_global_id(0);     // output row
    int sg   = get_local_id(1);      // subgroup index (K-split lane)
    int lane = get_local_id(0);      // lane within subgroup (0..COK_SG-1)

    int num_blocks_K = k / QK_K;
    int num_32blk    = k / 32;

    global const ushort * weight_ptr = src0_q  + gx;
    global const uchar  * qh_ptr     = src0_qh + gx;
    global const half   * d_ptr      = src0_d  + gx;
    global const half   * dm_ptr     = src0_dm + gx;

    half8 acc = 0;
    half8 B;
    half  dq;

    for (int blk = sg; blk < num_32blk; blk += COK_NSG) {
        int i       = blk << 5;       // blk * 32
        int sb_idx  = blk >> 3;       // (blk*32) / QK_K  (QK_K = 256 = 32*8)
        int sub_idx = blk & 7;        // (i/32) % 8

        half dd  = d_ptr [sb_idx * m];
        half dmm = dm_ptr[sb_idx * m];

        global const uchar * sc0 = src0_s + gx * num_blocks_K * K_SCALE_SIZE + sb_idx * K_SCALE_SIZE;
        uchar sv0, mn0;
        get_scale_min_k4(sub_idx, sc0, &sv0, &mn0, mask_d6, mask_d4, mask_hi2);
        half scale = convert_half(convert_float(dd)  * (float)sv0);
        half mval  = convert_half(convert_float(dmm) * (float)mn0);

        for (int l = 0; l < 32; l += 4) {
            int ki = i + l;
            ushort bits = weight_ptr[(ki>>2) * m];
            uchar  qh   = qh_ptr[(ki>>3) * m];
            int    qh_shift = ki & 7;

            B.s0123 = read_imageh(src1,     (ki+0) * n_4);
            B.s4567 = read_imageh(src1, 1 + (ki+0) * n_4);
            dq = ((half)((bits & 0x000F)        | (((qh >> (qh_shift+0)) & 1) << 4))) * scale - mval;
            acc += B * dq;

            B.s0123 = read_imageh(src1,     (ki+1) * n_4);
            B.s4567 = read_imageh(src1, 1 + (ki+1) * n_4);
            dq = ((half)(((bits & 0x00F0) >> 4) | (((qh >> (qh_shift+1)) & 1) << 4))) * scale - mval;
            acc += B * dq;

            B.s0123 = read_imageh(src1,     (ki+2) * n_4);
            B.s4567 = read_imageh(src1, 1 + (ki+2) * n_4);
            dq = ((half)(((bits & 0x0F00) >> 8) | (((qh >> (qh_shift+2)) & 1) << 4))) * scale - mval;
            acc += B * dq;

            B.s0123 = read_imageh(src1,     (ki+3) * n_4);
            B.s4567 = read_imageh(src1, 1 + (ki+3) * n_4);
            dq = ((half)(((bits & 0xF000) >> 12) | (((qh >> (qh_shift+3)) & 1) << 4))) * scale - mval;
            acc += B * dq;
        }
    }

    // cross-subgroup reduction over the K-split (float for accuracy)
    local float8 reduceLM[COK_SG * (COK_NSG - 1)];
    if (sg > 0) {
        reduceLM[(sg - 1) * COK_SG + lane] = convert_float8(acc);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (sg == 0) {
        float8 sum = convert_float8(acc);
        for (int s = 0; s < COK_NSG - 1; s++) {
            sum += reduceLM[s * COK_SG + lane];
        }
        int idx = gx;
        if (idx < m*n_no_padding) { dst[idx] = sum.s0; idx += m; }
        if (idx < m*n_no_padding) { dst[idx] = sum.s1; idx += m; }
        if (idx < m*n_no_padding) { dst[idx] = sum.s2; idx += m; }
        if (idx < m*n_no_padding) { dst[idx] = sum.s3; idx += m; }
        if (idx < m*n_no_padding) { dst[idx] = sum.s4; idx += m; }
        if (idx < m*n_no_padding) { dst[idx] = sum.s5; idx += m; }
        if (idx < m*n_no_padding) { dst[idx] = sum.s6; idx += m; }
        if (idx < m*n_no_padding) { dst[idx] = sum.s7; }
    }
}

// 4 output rows per lane, the change that took the q4_K, q6_K and q4_0 cooperative-K kernels
// +32.9%, +43.8% and +66%.
//
// The 1-row kernel above loads ONE ushort (2 bytes) of q and ONE uchar of qh per lane per 4
// K-values, where a GEMV loads 16, so it is ISSUE bound rather than bandwidth bound: on the
// X2-90 it runs Qwen3.8-27B's ssm_out (21.6 MB) in 429 us, i.e. 50 GB/s against a 152 GB/s
// bus. Four adjacent rows are contiguous in the packed layout, so q vector-loads as one 8-byte
// read, qh as one 4-byte read, d/dm as one half4 each, AND all four share the activation
// vector B, which quarters the image reads too.
//
// 🔴 The SCALE bytes are the exception. src0_s is laid out per ROW
// (gx * num_blocks_K * K_SCALE_SIZE), not column-major with stride m like every other array
// here, so four adjacent rows' scales are num_blocks_K * K_SCALE_SIZE apart and cannot be
// vector-loaded. get_scale_min_k4 therefore runs four times per 32-block -- amortised over 32
// elements and 8 output columns, which is why this is still worth doing.
//
// Needs m % 4 == 0; the host's m % 64 == 0 gate already implies it.
//
// Every dequant is cast to half BEFORE the multiply: E17.51 (Adreno 850) rejects
// `half8 * float` and kills the whole program, a failure an env-gated A/B cannot see because
// the kernel simply never builds.
REQD_SUBGROUP_SIZE_64
kernel void kernel_gemm_noshuffle_q5_k_f32_cok_r4(
    global const ushort * src0_q,
    global const uchar  * src0_qh,
    global const uchar  * src0_s,
    global const half   * src0_d,
    global const half   * src0_dm,
    read_only image1d_buffer_t src1,
    global float * dst,
    ulong offsetd,
    int m,
    int n,
    int k,
    int n_no_padding,
    uchar mask_d6,
    uchar mask_d4,
    uchar mask_hi2
) {
    dst = (global float *)((global char *)dst + offsetd);
    int n_4  = n >> 2;
    int gx   = get_global_id(0);     // 4-row group
    int sg   = get_local_id(1);
    int lane = get_local_id(0);

    int row0 = gx << 2;              // first of this lane's 4 rows
    int num_blocks_K = k / QK_K;
    int num_32blk    = k / 32;

    half8 acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
    half8 B;

    for (int blk = sg; blk < num_32blk; blk += COK_NSG) {
        int i       = blk << 5;
        int sb_idx  = blk >> 3;
        int sub_idx = blk & 7;

        half4 dd  = vload4(0, src0_d  + row0 + sb_idx * m);
        half4 dmm = vload4(0, src0_dm + row0 + sb_idx * m);

        // per-row scale bytes: row-major, so four separate unpacks
        half4 scale, mval;
        {
            uchar sv, mn;
            global const uchar * scp;
            scp = src0_s + (row0 + 0) * num_blocks_K * K_SCALE_SIZE + sb_idx * K_SCALE_SIZE;
            get_scale_min_k4(sub_idx, scp, &sv, &mn, mask_d6, mask_d4, mask_hi2);
            scale.s0 = convert_half(convert_float(dd.s0)  * (float)sv);
            mval .s0 = convert_half(convert_float(dmm.s0) * (float)mn);

            scp = src0_s + (row0 + 1) * num_blocks_K * K_SCALE_SIZE + sb_idx * K_SCALE_SIZE;
            get_scale_min_k4(sub_idx, scp, &sv, &mn, mask_d6, mask_d4, mask_hi2);
            scale.s1 = convert_half(convert_float(dd.s1)  * (float)sv);
            mval .s1 = convert_half(convert_float(dmm.s1) * (float)mn);

            scp = src0_s + (row0 + 2) * num_blocks_K * K_SCALE_SIZE + sb_idx * K_SCALE_SIZE;
            get_scale_min_k4(sub_idx, scp, &sv, &mn, mask_d6, mask_d4, mask_hi2);
            scale.s2 = convert_half(convert_float(dd.s2)  * (float)sv);
            mval .s2 = convert_half(convert_float(dmm.s2) * (float)mn);

            scp = src0_s + (row0 + 3) * num_blocks_K * K_SCALE_SIZE + sb_idx * K_SCALE_SIZE;
            get_scale_min_k4(sub_idx, scp, &sv, &mn, mask_d6, mask_d4, mask_hi2);
            scale.s3 = convert_half(convert_float(dd.s3)  * (float)sv);
            mval .s3 = convert_half(convert_float(dmm.s3) * (float)mn);
        }

        for (int l = 0; l < 32; l += 4) {
            int ki = i + l;
            ushort4 bits = vload4(0, src0_q  + row0 + (ki>>2) * m);
            uchar4  qh   = vload4(0, src0_qh + row0 + (ki>>3) * m);
            int     qs   = ki & 7;

            B.s0123 = read_imageh(src1,     (ki+0) * n_4);
            B.s4567 = read_imageh(src1, 1 + (ki+0) * n_4);
            acc0 += B * (half)(((half)(( bits.s0        & 0x000F) | (((qh.s0 >> (qs+0)) & 1) << 4))) * scale.s0 - mval.s0);
            acc1 += B * (half)(((half)(( bits.s1        & 0x000F) | (((qh.s1 >> (qs+0)) & 1) << 4))) * scale.s1 - mval.s1);
            acc2 += B * (half)(((half)(( bits.s2        & 0x000F) | (((qh.s2 >> (qs+0)) & 1) << 4))) * scale.s2 - mval.s2);
            acc3 += B * (half)(((half)(( bits.s3        & 0x000F) | (((qh.s3 >> (qs+0)) & 1) << 4))) * scale.s3 - mval.s3);

            B.s0123 = read_imageh(src1,     (ki+1) * n_4);
            B.s4567 = read_imageh(src1, 1 + (ki+1) * n_4);
            acc0 += B * (half)(((half)(((bits.s0 & 0x00F0) >> 4) | (((qh.s0 >> (qs+1)) & 1) << 4))) * scale.s0 - mval.s0);
            acc1 += B * (half)(((half)(((bits.s1 & 0x00F0) >> 4) | (((qh.s1 >> (qs+1)) & 1) << 4))) * scale.s1 - mval.s1);
            acc2 += B * (half)(((half)(((bits.s2 & 0x00F0) >> 4) | (((qh.s2 >> (qs+1)) & 1) << 4))) * scale.s2 - mval.s2);
            acc3 += B * (half)(((half)(((bits.s3 & 0x00F0) >> 4) | (((qh.s3 >> (qs+1)) & 1) << 4))) * scale.s3 - mval.s3);

            B.s0123 = read_imageh(src1,     (ki+2) * n_4);
            B.s4567 = read_imageh(src1, 1 + (ki+2) * n_4);
            acc0 += B * (half)(((half)(((bits.s0 & 0x0F00) >> 8) | (((qh.s0 >> (qs+2)) & 1) << 4))) * scale.s0 - mval.s0);
            acc1 += B * (half)(((half)(((bits.s1 & 0x0F00) >> 8) | (((qh.s1 >> (qs+2)) & 1) << 4))) * scale.s1 - mval.s1);
            acc2 += B * (half)(((half)(((bits.s2 & 0x0F00) >> 8) | (((qh.s2 >> (qs+2)) & 1) << 4))) * scale.s2 - mval.s2);
            acc3 += B * (half)(((half)(((bits.s3 & 0x0F00) >> 8) | (((qh.s3 >> (qs+2)) & 1) << 4))) * scale.s3 - mval.s3);

            B.s0123 = read_imageh(src1,     (ki+3) * n_4);
            B.s4567 = read_imageh(src1, 1 + (ki+3) * n_4);
            acc0 += B * (half)(((half)(((bits.s0 & 0xF000) >> 12) | (((qh.s0 >> (qs+3)) & 1) << 4))) * scale.s0 - mval.s0);
            acc1 += B * (half)(((half)(((bits.s1 & 0xF000) >> 12) | (((qh.s1 >> (qs+3)) & 1) << 4))) * scale.s1 - mval.s1);
            acc2 += B * (half)(((half)(((bits.s2 & 0xF000) >> 12) | (((qh.s2 >> (qs+3)) & 1) << 4))) * scale.s2 - mval.s2);
            acc3 += B * (half)(((half)(((bits.s3 & 0xF000) >> 12) | (((qh.s3 >> (qs+3)) & 1) << 4))) * scale.s3 - mval.s3);
        }
    }

    // Cross-subgroup reduction over the K-split, one row at a time so the __local buffer stays
    // the size the 1-row kernel uses. Four barriers instead of one.
    local float8 reduceLM[COK_SG * (COK_NSG - 1)];
    float8 out[4];
    for (int r = 0; r < 4; r++) {
        half8 acc = (r == 0) ? acc0 : (r == 1) ? acc1 : (r == 2) ? acc2 : acc3;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (sg > 0) {
            reduceLM[(sg - 1) * COK_SG + lane] = convert_float8(acc);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (sg == 0) {
            float8 sum = convert_float8(acc);
            for (int s = 0; s < COK_NSG - 1; s++) {
                sum += reduceLM[s * COK_SG + lane];
            }
            out[r] = sum;
        }
    }

    if (sg == 0) {
        // dst is [token, feature]: four adjacent rows are contiguous, one vstore4.
        int idx = row0;
        if (idx < m*n_no_padding) { vstore4((float4)(out[0].s0, out[1].s0, out[2].s0, out[3].s0), 0, dst + idx); idx += m; }
        if (idx < m*n_no_padding) { vstore4((float4)(out[0].s1, out[1].s1, out[2].s1, out[3].s1), 0, dst + idx); idx += m; }
        if (idx < m*n_no_padding) { vstore4((float4)(out[0].s2, out[1].s2, out[2].s2, out[3].s2), 0, dst + idx); idx += m; }
        if (idx < m*n_no_padding) { vstore4((float4)(out[0].s3, out[1].s3, out[2].s3, out[3].s3), 0, dst + idx); idx += m; }
        if (idx < m*n_no_padding) { vstore4((float4)(out[0].s4, out[1].s4, out[2].s4, out[3].s4), 0, dst + idx); idx += m; }
        if (idx < m*n_no_padding) { vstore4((float4)(out[0].s5, out[1].s5, out[2].s5, out[3].s5), 0, dst + idx); idx += m; }
        if (idx < m*n_no_padding) { vstore4((float4)(out[0].s6, out[1].s6, out[2].s6, out[3].s6), 0, dst + idx); idx += m; }
        if (idx < m*n_no_padding) { vstore4((float4)(out[0].s7, out[1].s7, out[2].s7, out[3].s7), 0, dst + idx); }
    }
}

// q5_K r4 with the K split ACROSS workgroups as well as inside one.
//
// r4 quarters the workgroup count, which is free when the row axis is generous and RUINOUS
// when it is not. Measured on Adreno X2-90 (16 CUs), q5_K, us/run at n=4:
//
//     shape          1-row wg   r4 wg    1-row      r4
//     4096 x 14336      64        16     803.3   630.8   -21.5%
//     5120 x  6144      80        20     432.8   508.3   +17.5%   <- ssm_out, the real one
//
// 5120 rows is 80 workgroups at 1 row per lane -- exactly 5 full waves -- and 20 at 4 rows per
// lane, which is 1.25 waves. The issue-rate win is real but the occupancy loss is bigger, so r4
// ALONE is a regression on the shape this kernel actually runs. Splitting K back across
// workgroups restores the wave count and keeps the wider loads. ggml_opencl_cok_ksplit() picks
// ksplit from the r4 workgroup count, so 4096 rows stay on the plain r4 kernel (16 wg is
// already a full wave) and 5120 rows come here with ksplit 4.
REQD_SUBGROUP_SIZE_64
kernel void kernel_gemm_noshuffle_q5_k_f32_cok_r4_splitk(
    global const ushort * src0_q,
    global const uchar  * src0_qh,
    global const uchar  * src0_s,
    global const half   * src0_d,
    global const half   * src0_dm,
    read_only image1d_buffer_t src1,
    global float * partial,          // [ksplit][n_no_padding][m]
    ulong offsetd,                   // always 0 here; kept so the host swaps one buffer
    int m,
    int n,
    int k,
    int n_no_padding,
    uchar mask_d6,
    uchar mask_d4,
    uchar mask_hi2,
    int ksplit
) {
    partial = (global float *)((global char *)partial + offsetd);
    int n_4  = n >> 2;
    int gx   = get_global_id(0);     // 4-row group
    int sg   = get_local_id(1);
    int lane = get_local_id(0);

    int ks   = get_group_id(1);      // which K slice

    int row0 = gx << 2;              // first of this lane's 4 rows
    int num_blocks_K = k / QK_K;
    int num_32blk    = k / 32;

    int chunk   = (num_32blk + ksplit - 1) / ksplit;
    int blk_beg = ks * chunk;
    int blk_end = min(blk_beg + chunk, num_32blk);

    half8 acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
    half8 B;

    for (int blk = blk_beg + sg; blk < blk_end; blk += COK_NSG) {
        int i       = blk << 5;
        int sb_idx  = blk >> 3;
        int sub_idx = blk & 7;

        half4 dd  = vload4(0, src0_d  + row0 + sb_idx * m);
        half4 dmm = vload4(0, src0_dm + row0 + sb_idx * m);

        // per-row scale bytes: row-major, so four separate unpacks
        half4 scale, mval;
        {
            uchar sv, mn;
            global const uchar * scp;
            scp = src0_s + (row0 + 0) * num_blocks_K * K_SCALE_SIZE + sb_idx * K_SCALE_SIZE;
            get_scale_min_k4(sub_idx, scp, &sv, &mn, mask_d6, mask_d4, mask_hi2);
            scale.s0 = convert_half(convert_float(dd.s0)  * (float)sv);
            mval .s0 = convert_half(convert_float(dmm.s0) * (float)mn);

            scp = src0_s + (row0 + 1) * num_blocks_K * K_SCALE_SIZE + sb_idx * K_SCALE_SIZE;
            get_scale_min_k4(sub_idx, scp, &sv, &mn, mask_d6, mask_d4, mask_hi2);
            scale.s1 = convert_half(convert_float(dd.s1)  * (float)sv);
            mval .s1 = convert_half(convert_float(dmm.s1) * (float)mn);

            scp = src0_s + (row0 + 2) * num_blocks_K * K_SCALE_SIZE + sb_idx * K_SCALE_SIZE;
            get_scale_min_k4(sub_idx, scp, &sv, &mn, mask_d6, mask_d4, mask_hi2);
            scale.s2 = convert_half(convert_float(dd.s2)  * (float)sv);
            mval .s2 = convert_half(convert_float(dmm.s2) * (float)mn);

            scp = src0_s + (row0 + 3) * num_blocks_K * K_SCALE_SIZE + sb_idx * K_SCALE_SIZE;
            get_scale_min_k4(sub_idx, scp, &sv, &mn, mask_d6, mask_d4, mask_hi2);
            scale.s3 = convert_half(convert_float(dd.s3)  * (float)sv);
            mval .s3 = convert_half(convert_float(dmm.s3) * (float)mn);
        }

        for (int l = 0; l < 32; l += 4) {
            int ki = i + l;
            ushort4 bits = vload4(0, src0_q  + row0 + (ki>>2) * m);
            uchar4  qh   = vload4(0, src0_qh + row0 + (ki>>3) * m);
            int     qs   = ki & 7;

            B.s0123 = read_imageh(src1,     (ki+0) * n_4);
            B.s4567 = read_imageh(src1, 1 + (ki+0) * n_4);
            acc0 += B * (half)(((half)(( bits.s0        & 0x000F) | (((qh.s0 >> (qs+0)) & 1) << 4))) * scale.s0 - mval.s0);
            acc1 += B * (half)(((half)(( bits.s1        & 0x000F) | (((qh.s1 >> (qs+0)) & 1) << 4))) * scale.s1 - mval.s1);
            acc2 += B * (half)(((half)(( bits.s2        & 0x000F) | (((qh.s2 >> (qs+0)) & 1) << 4))) * scale.s2 - mval.s2);
            acc3 += B * (half)(((half)(( bits.s3        & 0x000F) | (((qh.s3 >> (qs+0)) & 1) << 4))) * scale.s3 - mval.s3);

            B.s0123 = read_imageh(src1,     (ki+1) * n_4);
            B.s4567 = read_imageh(src1, 1 + (ki+1) * n_4);
            acc0 += B * (half)(((half)(((bits.s0 & 0x00F0) >> 4) | (((qh.s0 >> (qs+1)) & 1) << 4))) * scale.s0 - mval.s0);
            acc1 += B * (half)(((half)(((bits.s1 & 0x00F0) >> 4) | (((qh.s1 >> (qs+1)) & 1) << 4))) * scale.s1 - mval.s1);
            acc2 += B * (half)(((half)(((bits.s2 & 0x00F0) >> 4) | (((qh.s2 >> (qs+1)) & 1) << 4))) * scale.s2 - mval.s2);
            acc3 += B * (half)(((half)(((bits.s3 & 0x00F0) >> 4) | (((qh.s3 >> (qs+1)) & 1) << 4))) * scale.s3 - mval.s3);

            B.s0123 = read_imageh(src1,     (ki+2) * n_4);
            B.s4567 = read_imageh(src1, 1 + (ki+2) * n_4);
            acc0 += B * (half)(((half)(((bits.s0 & 0x0F00) >> 8) | (((qh.s0 >> (qs+2)) & 1) << 4))) * scale.s0 - mval.s0);
            acc1 += B * (half)(((half)(((bits.s1 & 0x0F00) >> 8) | (((qh.s1 >> (qs+2)) & 1) << 4))) * scale.s1 - mval.s1);
            acc2 += B * (half)(((half)(((bits.s2 & 0x0F00) >> 8) | (((qh.s2 >> (qs+2)) & 1) << 4))) * scale.s2 - mval.s2);
            acc3 += B * (half)(((half)(((bits.s3 & 0x0F00) >> 8) | (((qh.s3 >> (qs+2)) & 1) << 4))) * scale.s3 - mval.s3);

            B.s0123 = read_imageh(src1,     (ki+3) * n_4);
            B.s4567 = read_imageh(src1, 1 + (ki+3) * n_4);
            acc0 += B * (half)(((half)(((bits.s0 & 0xF000) >> 12) | (((qh.s0 >> (qs+3)) & 1) << 4))) * scale.s0 - mval.s0);
            acc1 += B * (half)(((half)(((bits.s1 & 0xF000) >> 12) | (((qh.s1 >> (qs+3)) & 1) << 4))) * scale.s1 - mval.s1);
            acc2 += B * (half)(((half)(((bits.s2 & 0xF000) >> 12) | (((qh.s2 >> (qs+3)) & 1) << 4))) * scale.s2 - mval.s2);
            acc3 += B * (half)(((half)(((bits.s3 & 0xF000) >> 12) | (((qh.s3 >> (qs+3)) & 1) << 4))) * scale.s3 - mval.s3);
        }
    }

    // Cross-subgroup reduction over the K-split, one row at a time so the __local buffer stays
    // the size the 1-row kernel uses. Four barriers instead of one.
    local float8 reduceLM[COK_SG * (COK_NSG - 1)];
    float8 out[4];
    for (int r = 0; r < 4; r++) {
        half8 acc = (r == 0) ? acc0 : (r == 1) ? acc1 : (r == 2) ? acc2 : acc3;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (sg > 0) {
            reduceLM[(sg - 1) * COK_SG + lane] = convert_float8(acc);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (sg == 0) {
            float8 sum = convert_float8(acc);
            for (int s = 0; s < COK_NSG - 1; s++) {
                sum += reduceLM[s * COK_SG + lane];
            }
            out[r] = sum;
        }
    }

    if (sg == 0) {
        // dst is [token, feature]: four adjacent rows are contiguous, one vstore4.
        global float * dst = partial + (size_t)ks * (size_t)m * (size_t)n_no_padding;
        int idx = row0;
        if (idx < m*n_no_padding) { vstore4((float4)(out[0].s0, out[1].s0, out[2].s0, out[3].s0), 0, dst + idx); idx += m; }
        if (idx < m*n_no_padding) { vstore4((float4)(out[0].s1, out[1].s1, out[2].s1, out[3].s1), 0, dst + idx); idx += m; }
        if (idx < m*n_no_padding) { vstore4((float4)(out[0].s2, out[1].s2, out[2].s2, out[3].s2), 0, dst + idx); idx += m; }
        if (idx < m*n_no_padding) { vstore4((float4)(out[0].s3, out[1].s3, out[2].s3, out[3].s3), 0, dst + idx); idx += m; }
        if (idx < m*n_no_padding) { vstore4((float4)(out[0].s4, out[1].s4, out[2].s4, out[3].s4), 0, dst + idx); idx += m; }
        if (idx < m*n_no_padding) { vstore4((float4)(out[0].s5, out[1].s5, out[2].s5, out[3].s5), 0, dst + idx); idx += m; }
        if (idx < m*n_no_padding) { vstore4((float4)(out[0].s6, out[1].s6, out[2].s6, out[3].s6), 0, dst + idx); idx += m; }
        if (idx < m*n_no_padding) { vstore4((float4)(out[0].s7, out[1].s7, out[2].s7, out[3].s7), 0, dst + idx); }
    }
}

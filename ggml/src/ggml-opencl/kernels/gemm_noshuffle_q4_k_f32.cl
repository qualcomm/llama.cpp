#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
#endif
#define QK_K         256
#define K_SCALE_SIZE 12

// scales are transposed: consecutive codes of a row are `stride` apart
inline void get_scale_min_k4(
    int j,
    global const uchar * q,
    int stride,
    uchar * d,
    uchar * m,
    uchar mask_d6,
    uchar mask_d4,
    uchar mask_hi2
) {
    if (j < 4) {
        *d = q[j*stride]     & mask_d6;
        *m = q[(j+4)*stride] & mask_d6;
    } else {
        *d = (q[(j+4)*stride] & mask_d4) | ((q[(j-4)*stride] & mask_hi2) >> 2);
        *m = ((q[(j+4)*stride] >> 4) & mask_d4) | ((q[j*stride] & mask_hi2) >> 2);
    }
}

// Four-row form of the above, for the kernels that give each lane 4 adjacent rows.
// Codes of ONE row are `stride` apart, but the four rows' copies of a given code are
// four ADJACENT bytes, so what the scalar version issues as 8-12 single-byte loads is
// 2-3 uchar4 loads here. Same arithmetic, same result; only the issue count changes.
inline void get_scale_min_k4_v4(
    int j,
    global const uchar * q,
    int stride,
    uchar4 * d,
    uchar4 * m,
    uchar mask_d6,
    uchar mask_d4,
    uchar mask_hi2
) {
    if (j < 4) {
        *d = vload4(0, q + j*stride)     & (uchar4)mask_d6;
        *m = vload4(0, q + (j+4)*stride) & (uchar4)mask_d6;
    } else {
        uchar4 hi = vload4(0, q + (j+4)*stride);
        uchar4 lo = vload4(0, q + (j-4)*stride);
        uchar4 cu = vload4(0, q + j*stride);
        *d = (hi & (uchar4)mask_d4) | ((lo & (uchar4)mask_hi2) >> (uchar4)2);
        *m = ((hi >> (uchar4)4) & (uchar4)mask_d4) | ((cu & (uchar4)mask_hi2) >> (uchar4)2);
    }
}

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_128
#endif
kernel void kernel_gemm_noshuffle_q4_k_f32(
    global const ushort * src0_q,
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


    global const ushort * weight_ptr = src0_q + gx_2;
    global const half   * d_ptr      = src0_d  + gx_2;
    global const half   * dm_ptr     = src0_dm + gx_2;

    for (int i = 0; i < k; i += 32) {
        int sb_idx  = i / QK_K;
        int sub_idx = (i / 32) % 8;

        half4 d  = vload4(0, d_ptr  + sb_idx * m);
        half4 dm = vload4(0, dm_ptr + sb_idx * m);

        global const uchar * sc0 = src0_s + sb_idx * K_SCALE_SIZE * m + (gx_2+0);
        global const uchar * sc1 = sc0 + 1;
        global const uchar * sc2 = sc0 + 2;
        global const uchar * sc3 = sc0 + 3;

        uchar sv0, mn0, sv1, mn1, sv2, mn2, sv3, mn3;
        get_scale_min_k4(sub_idx, sc0, m, &sv0, &mn0, mask_d6, mask_d4, mask_hi2);
        get_scale_min_k4(sub_idx, sc1, m, &sv1, &mn1, mask_d6, mask_d4, mask_hi2);
        get_scale_min_k4(sub_idx, sc2, m, &sv2, &mn2, mask_d6, mask_d4, mask_hi2);
        get_scale_min_k4(sub_idx, sc3, m, &sv3, &mn3, mask_d6, mask_d4, mask_hi2);

        half4 scale = convert_half4(convert_float4(d)  * convert_float4((uchar4)(sv0, sv1, sv2, sv3)));
        half4 mval  = convert_half4(convert_float4(dm) * convert_float4((uchar4)(mn0, mn1, mn2, mn3)));

        for (int l = 0; l < 32; l += 4) {
            int ki = i + l;
            ushort4 bits4 = vload4(0, weight_ptr + (ki/4) * m);

            // j=0
            B.s0123 = read_imageh(src1, gy*2   + (ki+0) * n_4);
            B.s4567 = read_imageh(src1, gy*2+1 + (ki+0) * n_4);
            dequantized_weights.s0 = (bits4.s0 & 0x000F) * scale.s0 - mval.s0;
            dequantized_weights.s1 = (bits4.s1 & 0x000F) * scale.s1 - mval.s1;
            dequantized_weights.s2 = (bits4.s2 & 0x000F) * scale.s2 - mval.s2;
            dequantized_weights.s3 = (bits4.s3 & 0x000F) * scale.s3 - mval.s3;
            c0 += B * dequantized_weights.s0;
            c1 += B * dequantized_weights.s1;
            c2 += B * dequantized_weights.s2;
            c3 += B * dequantized_weights.s3;

            // j=1
            B.s0123 = read_imageh(src1, gy*2   + (ki+1) * n_4);
            B.s4567 = read_imageh(src1, gy*2+1 + (ki+1) * n_4);
            dequantized_weights.s0 = ((bits4.s0 & 0x00F0) >> 4) * scale.s0 - mval.s0;
            dequantized_weights.s1 = ((bits4.s1 & 0x00F0) >> 4) * scale.s1 - mval.s1;
            dequantized_weights.s2 = ((bits4.s2 & 0x00F0) >> 4) * scale.s2 - mval.s2;
            dequantized_weights.s3 = ((bits4.s3 & 0x00F0) >> 4) * scale.s3 - mval.s3;
            c0 += B * dequantized_weights.s0;
            c1 += B * dequantized_weights.s1;
            c2 += B * dequantized_weights.s2;
            c3 += B * dequantized_weights.s3;

            // j=2
            B.s0123 = read_imageh(src1, gy*2   + (ki+2) * n_4);
            B.s4567 = read_imageh(src1, gy*2+1 + (ki+2) * n_4);
            dequantized_weights.s0 = ((bits4.s0 & 0x0F00) >> 8) * scale.s0 - mval.s0;
            dequantized_weights.s1 = ((bits4.s1 & 0x0F00) >> 8) * scale.s1 - mval.s1;
            dequantized_weights.s2 = ((bits4.s2 & 0x0F00) >> 8) * scale.s2 - mval.s2;
            dequantized_weights.s3 = ((bits4.s3 & 0x0F00) >> 8) * scale.s3 - mval.s3;
            c0 += B * dequantized_weights.s0;
            c1 += B * dequantized_weights.s1;
            c2 += B * dequantized_weights.s2;
            c3 += B * dequantized_weights.s3;

            // j=3
            B.s0123 = read_imageh(src1, gy*2   + (ki+3) * n_4);
            B.s4567 = read_imageh(src1, gy*2+1 + (ki+3) * n_4);
            dequantized_weights.s0 = ((bits4.s0 & 0xF000) >> 12) * scale.s0 - mval.s0;
            dequantized_weights.s1 = ((bits4.s1 & 0xF000) >> 12) * scale.s1 - mval.s1;
            dequantized_weights.s2 = ((bits4.s2 & 0xF000) >> 12) * scale.s2 - mval.s2;
            dequantized_weights.s3 = ((bits4.s3 & 0xF000) >> 12) * scale.s3 - mval.s3;
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

// 1x8 per-WI tile (1 output row x 8 output cols). For the small-batch
// (medium n_q, e.g. MTP/spec verify) path where the 2x8 kernel is starved:
// at ne1<=8 the grid is (1, ceil(M/2)) -> only ~M/256 workgroups, leaving
// the SP under-occupied. 1 row per WI doubles the M-axis workgroup count
// (ceil(M/1)/128 vs ceil(M/2)/128) AND collapses the accumulators to a
// single half8 (16 regs, no spill), so more waves co-reside. Same weight
// traffic as 2x8 (rows never share weights); the win is pure occupancy.
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_128
#endif
kernel void kernel_gemm_noshuffle_q4_k_f32_r1(
    global const ushort * src0_q,
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
    int gx = get_global_id(1);       // 1 row per WI

    half8 c0 = 0;
    half8 B;
    half dq;


    global const ushort * weight_ptr = src0_q + gx;
    global const half   * d_ptr      = src0_d  + gx;
    global const half   * dm_ptr     = src0_dm + gx;

    for (int i = 0; i < k; i += 32) {
        int sb_idx  = i / QK_K;
        int sub_idx = (i / 32) % 8;

        half dd  = d_ptr [sb_idx * m];
        half dmm = dm_ptr[sb_idx * m];

        global const uchar * sc0 = src0_s + sb_idx * K_SCALE_SIZE * m + gx;

        uchar sv0, mn0;
        get_scale_min_k4(sub_idx, sc0, m, &sv0, &mn0, mask_d6, mask_d4, mask_hi2);

        half scale = convert_half(convert_float(dd)  * (float)sv0);
        half mval  = convert_half(convert_float(dmm) * (float)mn0);

        for (int l = 0; l < 32; l += 4) {
            int ki = i + l;
            ushort bits = weight_ptr[(ki/4) * m];

            B.s0123 = read_imageh(src1, gy*2   + (ki+0) * n_4);
            B.s4567 = read_imageh(src1, gy*2+1 + (ki+0) * n_4);
            dq = (bits & 0x000F) * scale - mval;
            c0 += B * dq;

            B.s0123 = read_imageh(src1, gy*2   + (ki+1) * n_4);
            B.s4567 = read_imageh(src1, gy*2+1 + (ki+1) * n_4);
            dq = ((bits & 0x00F0) >> 4) * scale - mval;
            c0 += B * dq;

            B.s0123 = read_imageh(src1, gy*2   + (ki+2) * n_4);
            B.s4567 = read_imageh(src1, gy*2+1 + (ki+2) * n_4);
            dq = ((bits & 0x0F00) >> 8) * scale - mval;
            c0 += B * dq;

            B.s0123 = read_imageh(src1, gy*2   + (ki+3) * n_4);
            B.s4567 = read_imageh(src1, gy*2+1 + (ki+3) * n_4);
            dq = ((bits & 0xF000) >> 12) * scale - mval;
            c0 += B * dq;
        }
    }

    // Output: 8 cols, 1 row per col-step. Scalar store, coalesced across
    // neighbouring WIs (consecutive gx -> consecutive dst addresses).
    int idx = (gy<<3)*m + gx;
    if (idx < m*n_no_padding) { dst[idx] = c0.s0; idx += m; }
    if (idx < m*n_no_padding) { dst[idx] = c0.s1; idx += m; }
    if (idx < m*n_no_padding) { dst[idx] = c0.s2; idx += m; }
    if (idx < m*n_no_padding) { dst[idx] = c0.s3; idx += m; }
    if (idx < m*n_no_padding) { dst[idx] = c0.s4; idx += m; }
    if (idx < m*n_no_padding) { dst[idx] = c0.s5; idx += m; }
    if (idx < m*n_no_padding) { dst[idx] = c0.s6; idx += m; }
    if (idx < m*n_no_padding) { dst[idx] = c0.s7; }
}

// 2x8 tile, but weights read through an image1d_buffer (CL_R/UINT32 over the
// same packed-q buffer) instead of a plain global buffer. The ne1==1 GEMV
// already does this and is ~2.75x faster per weight byte than this GEMM at
// small n_q; the structural difference is the image path hits the dedicated
// TPL1 weight cache (L1) while the global path only reaches L2. At small n_q
// the forward is weight-read-bound, so L1-cached weights is the lever.
// The 2 adjacent rows the 2x8 tile reads as a ushort2 are exactly one uint32,
// so the vload2 becomes a single read_imageui at index gx + (ki/4)*(m/2).
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_128
#endif
kernel void kernel_gemm_noshuffle_q4_k_f32_kimg(
    read_only image1d_buffer_t src0_q_img,
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
    int m_2 = m >> 1;
    int gy = get_global_id(0);
    int gx = get_global_id(1);
    int gx_2 = gx << 1;

    half8 c0 = 0, c1 = 0;
    half8 B;
    half2 dequantized_weights;


    global const half * d_ptr  = src0_d  + gx_2;
    global const half * dm_ptr = src0_dm + gx_2;

    for (int i = 0; i < k; i += 32) {
        int sb_idx  = i / QK_K;
        int sub_idx = (i / 32) % 8;

        half2 d  = vload2(0, d_ptr  + sb_idx * m);
        half2 dm = vload2(0, dm_ptr + sb_idx * m);

        global const uchar * sc0 = src0_s + sb_idx * K_SCALE_SIZE * m + (gx_2+0);
        global const uchar * sc1 = sc0 + 1;

        uchar sv0, mn0, sv1, mn1;
        get_scale_min_k4(sub_idx, sc0, m, &sv0, &mn0, mask_d6, mask_d4, mask_hi2);
        get_scale_min_k4(sub_idx, sc1, m, &sv1, &mn1, mask_d6, mask_d4, mask_hi2);

        half2 scale = convert_half2(convert_float2(d)  * convert_float2((uchar2)(sv0, sv1)));
        half2 mval  = convert_half2(convert_float2(dm) * convert_float2((uchar2)(mn0, mn1)));

        for (int l = 0; l < 32; l += 4) {
            int ki = i + l;
            uint wpacked = read_imageui(src0_q_img, gx + (ki/4) * m_2).x;
            ushort2 bits2 = (ushort2)((ushort)(wpacked & 0xFFFFu), (ushort)(wpacked >> 16));

            // j=0
            B.s0123 = read_imageh(src1, gy*2   + (ki+0) * n_4);
            B.s4567 = read_imageh(src1, gy*2+1 + (ki+0) * n_4);
            dequantized_weights.s0 = (bits2.s0 & 0x000F) * scale.s0 - mval.s0;
            dequantized_weights.s1 = (bits2.s1 & 0x000F) * scale.s1 - mval.s1;
            c0 += B * dequantized_weights.s0;
            c1 += B * dequantized_weights.s1;

            // j=1
            B.s0123 = read_imageh(src1, gy*2   + (ki+1) * n_4);
            B.s4567 = read_imageh(src1, gy*2+1 + (ki+1) * n_4);
            dequantized_weights.s0 = ((bits2.s0 & 0x00F0) >> 4) * scale.s0 - mval.s0;
            dequantized_weights.s1 = ((bits2.s1 & 0x00F0) >> 4) * scale.s1 - mval.s1;
            c0 += B * dequantized_weights.s0;
            c1 += B * dequantized_weights.s1;

            // j=2
            B.s0123 = read_imageh(src1, gy*2   + (ki+2) * n_4);
            B.s4567 = read_imageh(src1, gy*2+1 + (ki+2) * n_4);
            dequantized_weights.s0 = ((bits2.s0 & 0x0F00) >> 8) * scale.s0 - mval.s0;
            dequantized_weights.s1 = ((bits2.s1 & 0x0F00) >> 8) * scale.s1 - mval.s1;
            c0 += B * dequantized_weights.s0;
            c1 += B * dequantized_weights.s1;

            // j=3
            B.s0123 = read_imageh(src1, gy*2   + (ki+3) * n_4);
            B.s4567 = read_imageh(src1, gy*2+1 + (ki+3) * n_4);
            dequantized_weights.s0 = ((bits2.s0 & 0xF000) >> 12) * scale.s0 - mval.s0;
            dequantized_weights.s1 = ((bits2.s1 & 0xF000) >> 12) * scale.s1 - mval.s1;
            c0 += B * dequantized_weights.s0;
            c1 += B * dequantized_weights.s1;
        }
    }

    int idx = (gy<<3)*m + (gx<<1);
    if (idx+1 < m*n_no_padding) { vstore2((float2)(c0.s0, c1.s0), 0, dst + idx); idx += m; }
    if (idx+1 < m*n_no_padding) { vstore2((float2)(c0.s1, c1.s1), 0, dst + idx); idx += m; }
    if (idx+1 < m*n_no_padding) { vstore2((float2)(c0.s2, c1.s2), 0, dst + idx); idx += m; }
    if (idx+1 < m*n_no_padding) { vstore2((float2)(c0.s3, c1.s3), 0, dst + idx); idx += m; }
    if (idx+1 < m*n_no_padding) { vstore2((float2)(c0.s4, c1.s4), 0, dst + idx); idx += m; }
    if (idx+1 < m*n_no_padding) { vstore2((float2)(c0.s5, c1.s5), 0, dst + idx); idx += m; }
    if (idx+1 < m*n_no_padding) { vstore2((float2)(c0.s6, c1.s6), 0, dst + idx); idx += m; }
    if (idx+1 < m*n_no_padding) { vstore2((float2)(c0.s7, c1.s7), 0, dst + idx); }
}

// Cooperative-K GEMM for the small-batch (n_q in [2..8]) path. Mirrors the
// ne1==1 GEMV's structure: a WG is (COK_SG lanes x COK_NSG subgroups); each
// lane owns ONE output row and computes its 8 (padded) columns, and the
// COK_NSG subgroups SPLIT the K reduction round-robin, combining via a
// __local reduction. This is the thing the per-WI GEMM lacked — at small n_q
// the old kernel had ~M/256 workgroups each walking all of K serially; this
// has M/64 workgroups AND COK_NSG-way K parallelism. Uses REQD_SUBGROUP_SIZE_64
// + barrier (same safe reduction pattern as the GEMV; never sub_group_reduce
// at full width on X2 per the GDN miscompile note).
#define COK_NSG 8
#define COK_SG  64
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_gemm_noshuffle_q4_k_f32_cok(
    global const ushort * src0_q,
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

    int num_32blk    = k / 32;

    global const ushort * weight_ptr = src0_q + gx;
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

        global const uchar * sc0 = src0_s + sb_idx * K_SCALE_SIZE * m + gx;
        uchar sv0, mn0;
        get_scale_min_k4(sub_idx, sc0, m, &sv0, &mn0, mask_d6, mask_d4, mask_hi2);
        half scale = convert_half(convert_float(dd)  * (float)sv0);
        half mval  = convert_half(convert_float(dmm) * (float)mn0);

        for (int l = 0; l < 32; l += 4) {
            int ki = i + l;
            ushort bits = weight_ptr[(ki>>2) * m];

            B.s0123 = read_imageh(src1,     (ki+0) * n_4);
            B.s4567 = read_imageh(src1, 1 + (ki+0) * n_4);
            dq = (bits & 0x000F) * scale - mval;
            acc += B * dq;

            B.s0123 = read_imageh(src1,     (ki+1) * n_4);
            B.s4567 = read_imageh(src1, 1 + (ki+1) * n_4);
            dq = ((bits & 0x00F0) >> 4) * scale - mval;
            acc += B * dq;

            B.s0123 = read_imageh(src1,     (ki+2) * n_4);
            B.s4567 = read_imageh(src1, 1 + (ki+2) * n_4);
            dq = ((bits & 0x0F00) >> 8) * scale - mval;
            acc += B * dq;

            B.s0123 = read_imageh(src1,     (ki+3) * n_4);
            B.s4567 = read_imageh(src1, 1 + (ki+3) * n_4);
            dq = ((bits & 0xF000) >> 12) * scale - mval;
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

// Weights-as-texture variant of kernel_gemm_noshuffle_q4_k_f32_cok.
//
// Same math; the only change is that the q4_K weight plane is read through an
// image1d_buffer instead of a plain global buffer. The shipped cok has the two
// operands the wrong way round: it reads the ACTIVATIONS (53 KB at n_q=8) through
// a texture and the WEIGHTS (74.8 MB for one 19968x6656 projection) as a plain
// buffer, and the weights are the whole traffic.
//
// Profiled on Adreno X2-90, muse-glimmer-30B Q4_K_M at ne1=8: cok owns 73.1% of
// the pass and moves its weight bytes at 50.5 GB/s, while the q4_K GEMV reads the
// same bytes through read_imageui at 117.7. 99.0% of the pass is GPU-busy, so the
// gap is in the read path, not in host time.
//
// Bound as CL_R/CL_UNSIGNED_INT32 (one texel = 2 packed ushorts), the same format
// the _kimg and dp4a _wimg kernels use. The host only selects this variant when m
// is even, so (gx + (ki>>2)*m) has constant ushort parity per row: the wanted half
// is picked with one hoisted shift, and adjacent lanes share each uint32 texel.
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_gemm_noshuffle_q4_k_f32_cok_wimg(
    read_only image1d_buffer_t src0_q_img,
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

    int num_32blk = k / 32;

    global const half * d_ptr  = src0_d  + gx;
    global const half * dm_ptr = src0_dm + gx;

    // m is even (host-gated), so every weight index for this row has the same
    // parity and the half-select is loop-invariant.
    const uint sel = ((uint)gx & 1u) * 16u;

    half8 acc = 0;
    half8 B;
    half  dq;

    for (int blk = sg; blk < num_32blk; blk += COK_NSG) {
        int i       = blk << 5;       // blk * 32
        int sb_idx  = blk >> 3;       // (blk*32) / QK_K  (QK_K = 256 = 32*8)
        int sub_idx = blk & 7;        // (i/32) % 8

        half dd  = d_ptr [sb_idx * m];
        half dmm = dm_ptr[sb_idx * m];

        global const uchar * sc0 = src0_s + sb_idx * K_SCALE_SIZE * m + gx;
        uchar sv0, mn0;
        get_scale_min_k4(sub_idx, sc0, m, &sv0, &mn0, mask_d6, mask_d4, mask_hi2);
        half scale = convert_half(convert_float(dd)  * (float)sv0);
        half mval  = convert_half(convert_float(dmm) * (float)mn0);

        for (int l = 0; l < 32; l += 4) {
            int ki = i + l;
            ushort bits = (ushort)(read_imageui(src0_q_img, (int)(((uint)gx + (uint)(ki>>2) * (uint)m) >> 1)).x >> sel);

            B.s0123 = read_imageh(src1,     (ki+0) * n_4);
            B.s4567 = read_imageh(src1, 1 + (ki+0) * n_4);
            dq = (bits & 0x000F) * scale - mval;
            acc += B * dq;

            B.s0123 = read_imageh(src1,     (ki+1) * n_4);
            B.s4567 = read_imageh(src1, 1 + (ki+1) * n_4);
            dq = ((bits & 0x00F0) >> 4) * scale - mval;
            acc += B * dq;

            B.s0123 = read_imageh(src1,     (ki+2) * n_4);
            B.s4567 = read_imageh(src1, 1 + (ki+2) * n_4);
            dq = ((bits & 0x0F00) >> 8) * scale - mval;
            acc += B * dq;

            B.s0123 = read_imageh(src1,     (ki+3) * n_4);
            B.s4567 = read_imageh(src1, 1 + (ki+3) * n_4);
            dq = ((bits & 0xF000) >> 12) * scale - mval;
            acc += B * dq;
        }
    }

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

// 4-rows-per-lane cok. Same cooperative-K structure; the only change is that each
// lane owns FOUR adjacent output rows instead of one.
//
// Why: profiled on X2-90 (muse-glimmer-30B Q4_K_M, ne1=8) the 1-row cok owns 73.1%
// of the pass and moves weights at 50.5 GB/s against 117.7 for the q4_K GEMV. Its
// read is already perfectly coalesced - 64 lanes cover 128 contiguous bytes - but it
// is only TWO BYTES per lane per load, while the GEMV loads a uint4. Same bytes,
// about 8x the load instructions, which is an issue-rate problem and not a bandwidth
// one. Swapping the read to a texture was tried and lost 39%, which is the other way
// of confirming the read path is not where the time goes.
//
// Four adjacent rows are contiguous in the packed layout, so their four ushorts come
// in as one 8-byte vector load, and their scales/mins as one half4 each. The
// activation vector B is shared by all four rows, so read_imageh is issued a quarter
// as often for the same work.
//
// Register cost is four half8 accumulators instead of one. That is the risk here -
// register pressure, not ALU, has governed every previous attempt on this kernel.
//
// Needs m % 4 == 0 for the vector loads and the row-group split; the host checks it.
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_gemm_noshuffle_q4_k_f32_cok_r4(
    global const ushort * src0_q,
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
    int sg   = get_local_id(1);      // subgroup index (K-split lane)
    int lane = get_local_id(0);      // lane within subgroup (0..COK_SG-1)

    int row0 = gx << 2;              // first of this lane's 4 rows
    int num_32blk = k / 32;

    half8 acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
    half8 B;

    for (int blk = sg; blk < num_32blk; blk += COK_NSG) {
        int i       = blk << 5;       // blk * 32
        int sb_idx  = blk >> 3;       // (blk*32) / QK_K
        int sub_idx = blk & 7;        // (i/32) % 8

        // the four rows are adjacent, so one vector load each
        half4 dd  = vload4(0, src0_d  + row0 + sb_idx * m);
        half4 dmm = vload4(0, src0_dm + row0 + sb_idx * m);

        global const uchar * sc = src0_s + sb_idx * K_SCALE_SIZE * m + row0;
        uchar sv0, mn0, sv1, mn1, sv2, mn2, sv3, mn3;
        get_scale_min_k4(sub_idx, sc + 0, m, &sv0, &mn0, mask_d6, mask_d4, mask_hi2);
        get_scale_min_k4(sub_idx, sc + 1, m, &sv1, &mn1, mask_d6, mask_d4, mask_hi2);
        get_scale_min_k4(sub_idx, sc + 2, m, &sv2, &mn2, mask_d6, mask_d4, mask_hi2);
        get_scale_min_k4(sub_idx, sc + 3, m, &sv3, &mn3, mask_d6, mask_d4, mask_hi2);

        half4 scale, mval;
        scale.s0 = convert_half(convert_float(dd.s0)  * (float)sv0);
        scale.s1 = convert_half(convert_float(dd.s1)  * (float)sv1);
        scale.s2 = convert_half(convert_float(dd.s2)  * (float)sv2);
        scale.s3 = convert_half(convert_float(dd.s3)  * (float)sv3);
        mval.s0  = convert_half(convert_float(dmm.s0) * (float)mn0);
        mval.s1  = convert_half(convert_float(dmm.s1) * (float)mn1);
        mval.s2  = convert_half(convert_float(dmm.s2) * (float)mn2);
        mval.s3  = convert_half(convert_float(dmm.s3) * (float)mn3);

        for (int l = 0; l < 32; l += 4) {
            int ki = i + l;
            // 8 bytes per lane: this lane's four rows at this K-group
            ushort4 bits = vload4(0, src0_q + row0 + (ki >> 2) * m);

            B.s0123 = read_imageh(src1,     (ki+0) * n_4);
            B.s4567 = read_imageh(src1, 1 + (ki+0) * n_4);
            acc0 += B * ((half)( bits.s0        & 0x000F) * scale.s0 - mval.s0);
            acc1 += B * ((half)( bits.s1        & 0x000F) * scale.s1 - mval.s1);
            acc2 += B * ((half)( bits.s2        & 0x000F) * scale.s2 - mval.s2);
            acc3 += B * ((half)( bits.s3        & 0x000F) * scale.s3 - mval.s3);

            B.s0123 = read_imageh(src1,     (ki+1) * n_4);
            B.s4567 = read_imageh(src1, 1 + (ki+1) * n_4);
            acc0 += B * ((half)((bits.s0 >> 4)  & 0x000F) * scale.s0 - mval.s0);
            acc1 += B * ((half)((bits.s1 >> 4)  & 0x000F) * scale.s1 - mval.s1);
            acc2 += B * ((half)((bits.s2 >> 4)  & 0x000F) * scale.s2 - mval.s2);
            acc3 += B * ((half)((bits.s3 >> 4)  & 0x000F) * scale.s3 - mval.s3);

            B.s0123 = read_imageh(src1,     (ki+2) * n_4);
            B.s4567 = read_imageh(src1, 1 + (ki+2) * n_4);
            acc0 += B * ((half)((bits.s0 >> 8)  & 0x000F) * scale.s0 - mval.s0);
            acc1 += B * ((half)((bits.s1 >> 8)  & 0x000F) * scale.s1 - mval.s1);
            acc2 += B * ((half)((bits.s2 >> 8)  & 0x000F) * scale.s2 - mval.s2);
            acc3 += B * ((half)((bits.s3 >> 8)  & 0x000F) * scale.s3 - mval.s3);

            B.s0123 = read_imageh(src1,     (ki+3) * n_4);
            B.s4567 = read_imageh(src1, 1 + (ki+3) * n_4);
            acc0 += B * ((half)((bits.s0 >> 12) & 0x000F) * scale.s0 - mval.s0);
            acc1 += B * ((half)((bits.s1 >> 12) & 0x000F) * scale.s1 - mval.s1);
            acc2 += B * ((half)((bits.s2 >> 12) & 0x000F) * scale.s2 - mval.s2);
            acc3 += B * ((half)((bits.s3 >> 12) & 0x000F) * scale.s3 - mval.s3);
        }
    }

    // Cross-subgroup reduction over the K-split, one row at a time so the __local
    // buffer stays the same size as the 1-row kernel's. Four barriers instead of one;
    // a WG barrier is 0.154 ns/op on X2 at WG=64 (measured), so that is not a cost.
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

// _cok_r4 with the SCALE reads vectorised. Identical arithmetic to _cok_r4; the only
// change is that the four rows' scale codes are read as uchar4 instead of one byte at
// a time, cutting 8-12 narrow loads per 32-K step to 2-3. The weights, d and dm were
// already vector loads after r4, so the scale codes were the last narrow read left in
// the inner loop, and the r4 win showed this kernel family is issue bound.
REQD_SUBGROUP_SIZE_64
kernel void kernel_gemm_noshuffle_q4_k_f32_cok_r4_sv(
    global const ushort * src0_q,
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
    int sg   = get_local_id(1);      // subgroup index (K-split lane)
    int lane = get_local_id(0);      // lane within subgroup (0..COK_SG-1)

    int row0 = gx << 2;              // first of this lane's 4 rows
    int num_32blk = k / 32;

    half8 acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
    half8 B;

    for (int blk = sg; blk < num_32blk; blk += COK_NSG) {
        int i       = blk << 5;       // blk * 32
        int sb_idx  = blk >> 3;       // (blk*32) / QK_K
        int sub_idx = blk & 7;        // (i/32) % 8

        // the four rows are adjacent, so one vector load each
        half4 dd  = vload4(0, src0_d  + row0 + sb_idx * m);
        half4 dmm = vload4(0, src0_dm + row0 + sb_idx * m);

        global const uchar * sc = src0_s + sb_idx * K_SCALE_SIZE * m + row0;
        uchar4 sv, mn;
        get_scale_min_k4_v4(sub_idx, sc, m, &sv, &mn, mask_d6, mask_d4, mask_hi2);

        half4 scale = convert_half4(convert_float4(dd)  * convert_float4(sv));
        half4 mval  = convert_half4(convert_float4(dmm) * convert_float4(mn));

        for (int l = 0; l < 32; l += 4) {
            int ki = i + l;
            // 8 bytes per lane: this lane's four rows at this K-group
            ushort4 bits = vload4(0, src0_q + row0 + (ki >> 2) * m);

            B.s0123 = read_imageh(src1,     (ki+0) * n_4);
            B.s4567 = read_imageh(src1, 1 + (ki+0) * n_4);
            acc0 += B * ((half)( bits.s0        & 0x000F) * scale.s0 - mval.s0);
            acc1 += B * ((half)( bits.s1        & 0x000F) * scale.s1 - mval.s1);
            acc2 += B * ((half)( bits.s2        & 0x000F) * scale.s2 - mval.s2);
            acc3 += B * ((half)( bits.s3        & 0x000F) * scale.s3 - mval.s3);

            B.s0123 = read_imageh(src1,     (ki+1) * n_4);
            B.s4567 = read_imageh(src1, 1 + (ki+1) * n_4);
            acc0 += B * ((half)((bits.s0 >> 4)  & 0x000F) * scale.s0 - mval.s0);
            acc1 += B * ((half)((bits.s1 >> 4)  & 0x000F) * scale.s1 - mval.s1);
            acc2 += B * ((half)((bits.s2 >> 4)  & 0x000F) * scale.s2 - mval.s2);
            acc3 += B * ((half)((bits.s3 >> 4)  & 0x000F) * scale.s3 - mval.s3);

            B.s0123 = read_imageh(src1,     (ki+2) * n_4);
            B.s4567 = read_imageh(src1, 1 + (ki+2) * n_4);
            acc0 += B * ((half)((bits.s0 >> 8)  & 0x000F) * scale.s0 - mval.s0);
            acc1 += B * ((half)((bits.s1 >> 8)  & 0x000F) * scale.s1 - mval.s1);
            acc2 += B * ((half)((bits.s2 >> 8)  & 0x000F) * scale.s2 - mval.s2);
            acc3 += B * ((half)((bits.s3 >> 8)  & 0x000F) * scale.s3 - mval.s3);

            B.s0123 = read_imageh(src1,     (ki+3) * n_4);
            B.s4567 = read_imageh(src1, 1 + (ki+3) * n_4);
            acc0 += B * ((half)((bits.s0 >> 12) & 0x000F) * scale.s0 - mval.s0);
            acc1 += B * ((half)((bits.s1 >> 12) & 0x000F) * scale.s1 - mval.s1);
            acc2 += B * ((half)((bits.s2 >> 12) & 0x000F) * scale.s2 - mval.s2);
            acc3 += B * ((half)((bits.s3 >> 12) & 0x000F) * scale.s3 - mval.s3);
        }
    }

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

// _cok_r4 with the per-weight scale and min folded OUT of the inner loop.
//
// Two independent load-side experiments came back null on this kernel -- 8 rows per
// lane (16 B, double r4's) and reading the four rows' scale codes as uchar4 -- so what
// binds it is not load issue rate any more. This kernel is the ALU-side test: it cuts
// arithmetic without moving a single byte.
//
// Per weight the inner loop computed  acc += B * (w*scale - mval), i.e. two scalar ops
// per (k, row) before the half8 FMA. Over one 32-K sub-block scale and mval are
// constant, so
//
//     sum_k B_k*(w_k*scale - mval) = scale * sum_k(B_k*w_k) - mval * sum_k(B_k)
//
// and sum_k(B_k) does not depend on the row, so the four rows share it. The inner loop
// then costs one convert and one FMA per (k, row) plus one shared add per k, and the
// scale/min are applied once per 32-K block instead of once per weight.
//
// NOT bit-identical to _cok_r4: the products are summed before scaling rather than
// after, so half rounding falls differently. The raw accumulator is reset every 32-K
// block, which bounds it at 32*15*|B| and keeps it in half's range.
REQD_SUBGROUP_SIZE_64
kernel void kernel_gemm_noshuffle_q4_k_f32_cok_r4_ma(
    global const ushort * src0_q,
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
    int gx   = get_global_id(0);
    int sg   = get_local_id(1);
    int lane = get_local_id(0);

    int row0 = gx << 2;
    int num_32blk = k / 32;

    half8 acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
    half8 B;

    for (int blk = sg; blk < num_32blk; blk += COK_NSG) {
        int i       = blk << 5;
        int sb_idx  = blk >> 3;
        int sub_idx = blk & 7;

        half4 dd  = vload4(0, src0_d  + row0 + sb_idx * m);
        half4 dmm = vload4(0, src0_dm + row0 + sb_idx * m);

        global const uchar * sc = src0_s + sb_idx * K_SCALE_SIZE * m + row0;
        uchar sv0, mn0, sv1, mn1, sv2, mn2, sv3, mn3;
        get_scale_min_k4(sub_idx, sc + 0, m, &sv0, &mn0, mask_d6, mask_d4, mask_hi2);
        get_scale_min_k4(sub_idx, sc + 1, m, &sv1, &mn1, mask_d6, mask_d4, mask_hi2);
        get_scale_min_k4(sub_idx, sc + 2, m, &sv2, &mn2, mask_d6, mask_d4, mask_hi2);
        get_scale_min_k4(sub_idx, sc + 3, m, &sv3, &mn3, mask_d6, mask_d4, mask_hi2);

        half4 scale, mval;
        scale.s0 = convert_half(convert_float(dd.s0)  * (float)sv0);
        scale.s1 = convert_half(convert_float(dd.s1)  * (float)sv1);
        scale.s2 = convert_half(convert_float(dd.s2)  * (float)sv2);
        scale.s3 = convert_half(convert_float(dd.s3)  * (float)sv3);
        mval.s0  = convert_half(convert_float(dmm.s0) * (float)mn0);
        mval.s1  = convert_half(convert_float(dmm.s1) * (float)mn1);
        mval.s2  = convert_half(convert_float(dmm.s2) * (float)mn2);
        mval.s3  = convert_half(convert_float(dmm.s3) * (float)mn3);

        // block-local sums: raw_r = sum_k B_k*w_kr, bsum = sum_k B_k (shared by the rows)
        half8 raw0 = 0, raw1 = 0, raw2 = 0, raw3 = 0, bsum = 0;

        for (int l = 0; l < 32; l += 4) {
            int ki = i + l;
            ushort4 bits = vload4(0, src0_q + row0 + (ki >> 2) * m);

            B.s0123 = read_imageh(src1,     (ki+0) * n_4);
            B.s4567 = read_imageh(src1, 1 + (ki+0) * n_4);
            bsum += B;
            raw0 += B * (half)( bits.s0        & 0x000F);
            raw1 += B * (half)( bits.s1        & 0x000F);
            raw2 += B * (half)( bits.s2        & 0x000F);
            raw3 += B * (half)( bits.s3        & 0x000F);

            B.s0123 = read_imageh(src1,     (ki+1) * n_4);
            B.s4567 = read_imageh(src1, 1 + (ki+1) * n_4);
            bsum += B;
            raw0 += B * (half)((bits.s0 >> 4)  & 0x000F);
            raw1 += B * (half)((bits.s1 >> 4)  & 0x000F);
            raw2 += B * (half)((bits.s2 >> 4)  & 0x000F);
            raw3 += B * (half)((bits.s3 >> 4)  & 0x000F);

            B.s0123 = read_imageh(src1,     (ki+2) * n_4);
            B.s4567 = read_imageh(src1, 1 + (ki+2) * n_4);
            bsum += B;
            raw0 += B * (half)((bits.s0 >> 8)  & 0x000F);
            raw1 += B * (half)((bits.s1 >> 8)  & 0x000F);
            raw2 += B * (half)((bits.s2 >> 8)  & 0x000F);
            raw3 += B * (half)((bits.s3 >> 8)  & 0x000F);

            B.s0123 = read_imageh(src1,     (ki+3) * n_4);
            B.s4567 = read_imageh(src1, 1 + (ki+3) * n_4);
            bsum += B;
            raw0 += B * (half)((bits.s0 >> 12) & 0x000F);
            raw1 += B * (half)((bits.s1 >> 12) & 0x000F);
            raw2 += B * (half)((bits.s2 >> 12) & 0x000F);
            raw3 += B * (half)((bits.s3 >> 12) & 0x000F);
        }

        acc0 += raw0 * scale.s0 - bsum * mval.s0;
        acc1 += raw1 * scale.s1 - bsum * mval.s1;
        acc2 += raw2 * scale.s2 - bsum * mval.s2;
        acc3 += raw3 * scale.s3 - bsum * mval.s3;
    }

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

// 8-rows-per-lane cok. Same idea as _cok_r4, taken one step further: eight adjacent
// rows load as a single ushort8, which is 16 bytes per lane and finally matches the
// uint4 the q4_K GEMV issues. r4 (8 bytes) was worth +32.0% on muse pp8; the question
// this kernel asks is whether the last doubling still pays or whether the accumulators
// spill first.
//
// Register pressure is the whole risk. Eight half8 accumulators is 64 halves before B,
// the weight vector and the scales, and register pressure - not ALU, not bandwidth -
// has governed every previous attempt on this kernel family. Accumulators are kept as
// NAMED registers, never a private array: on Adreno a per-WI array spills to private
// memory whose loads are issued per-wave with no cross-WI coalescing, and that spill
// is the dominant cost when it happens.
//
// Needs m % 8 == 0 so the vector loads stay 16-byte aligned; the host checks it.
#define COK_R8_ACC(P)                                                             \
    B.s0123 = read_imageh(src1,     (ki+(P)) * n_4);                              \
    B.s4567 = read_imageh(src1, 1 + (ki+(P)) * n_4);                              \
    acc0 += B * ((half)((bits.s0 >> (4*(P))) & 0x000F) * scale.s0 - mval.s0);     \
    acc1 += B * ((half)((bits.s1 >> (4*(P))) & 0x000F) * scale.s1 - mval.s1);     \
    acc2 += B * ((half)((bits.s2 >> (4*(P))) & 0x000F) * scale.s2 - mval.s2);     \
    acc3 += B * ((half)((bits.s3 >> (4*(P))) & 0x000F) * scale.s3 - mval.s3);     \
    acc4 += B * ((half)((bits.s4 >> (4*(P))) & 0x000F) * scale.s4 - mval.s4);     \
    acc5 += B * ((half)((bits.s5 >> (4*(P))) & 0x000F) * scale.s5 - mval.s5);     \
    acc6 += B * ((half)((bits.s6 >> (4*(P))) & 0x000F) * scale.s6 - mval.s6);     \
    acc7 += B * ((half)((bits.s7 >> (4*(P))) & 0x000F) * scale.s7 - mval.s7);

#define COK_R8_RED(ACC, OUT)                                                      \
    barrier(CLK_LOCAL_MEM_FENCE);                                                 \
    if (sg > 0) { reduceLM[(sg - 1) * COK_SG + lane] = convert_float8(ACC); }     \
    barrier(CLK_LOCAL_MEM_FENCE);                                                 \
    if (sg == 0) {                                                                \
        float8 sum = convert_float8(ACC);                                         \
        for (int s = 0; s < COK_NSG - 1; s++) { sum += reduceLM[s * COK_SG + lane]; } \
        OUT = sum;                                                                \
    }

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_gemm_noshuffle_q4_k_f32_cok_r8(
    global const ushort * src0_q,
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
    int gx   = get_global_id(0);     // 8-row group
    int sg   = get_local_id(1);
    int lane = get_local_id(0);

    int row0 = gx << 3;
    int num_32blk = k / 32;

    half8 acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
    half8 acc4 = 0, acc5 = 0, acc6 = 0, acc7 = 0;
    half8 B;

    for (int blk = sg; blk < num_32blk; blk += COK_NSG) {
        int i       = blk << 5;
        int sb_idx  = blk >> 3;
        int sub_idx = blk & 7;

        half8 dd  = vload8(0, src0_d  + row0 + sb_idx * m);
        half8 dmm = vload8(0, src0_dm + row0 + sb_idx * m);

        global const uchar * sc = src0_s + sb_idx * K_SCALE_SIZE * m + row0;
        uchar sv[8], mn[8];
        get_scale_min_k4(sub_idx, sc + 0, m, &sv[0], &mn[0], mask_d6, mask_d4, mask_hi2);
        get_scale_min_k4(sub_idx, sc + 1, m, &sv[1], &mn[1], mask_d6, mask_d4, mask_hi2);
        get_scale_min_k4(sub_idx, sc + 2, m, &sv[2], &mn[2], mask_d6, mask_d4, mask_hi2);
        get_scale_min_k4(sub_idx, sc + 3, m, &sv[3], &mn[3], mask_d6, mask_d4, mask_hi2);
        get_scale_min_k4(sub_idx, sc + 4, m, &sv[4], &mn[4], mask_d6, mask_d4, mask_hi2);
        get_scale_min_k4(sub_idx, sc + 5, m, &sv[5], &mn[5], mask_d6, mask_d4, mask_hi2);
        get_scale_min_k4(sub_idx, sc + 6, m, &sv[6], &mn[6], mask_d6, mask_d4, mask_hi2);
        get_scale_min_k4(sub_idx, sc + 7, m, &sv[7], &mn[7], mask_d6, mask_d4, mask_hi2);

        half8 scale, mval;
        scale.s0 = convert_half(convert_float(dd.s0) * (float)sv[0]);
        scale.s1 = convert_half(convert_float(dd.s1) * (float)sv[1]);
        scale.s2 = convert_half(convert_float(dd.s2) * (float)sv[2]);
        scale.s3 = convert_half(convert_float(dd.s3) * (float)sv[3]);
        scale.s4 = convert_half(convert_float(dd.s4) * (float)sv[4]);
        scale.s5 = convert_half(convert_float(dd.s5) * (float)sv[5]);
        scale.s6 = convert_half(convert_float(dd.s6) * (float)sv[6]);
        scale.s7 = convert_half(convert_float(dd.s7) * (float)sv[7]);
        mval.s0  = convert_half(convert_float(dmm.s0) * (float)mn[0]);
        mval.s1  = convert_half(convert_float(dmm.s1) * (float)mn[1]);
        mval.s2  = convert_half(convert_float(dmm.s2) * (float)mn[2]);
        mval.s3  = convert_half(convert_float(dmm.s3) * (float)mn[3]);
        mval.s4  = convert_half(convert_float(dmm.s4) * (float)mn[4]);
        mval.s5  = convert_half(convert_float(dmm.s5) * (float)mn[5]);
        mval.s6  = convert_half(convert_float(dmm.s6) * (float)mn[6]);
        mval.s7  = convert_half(convert_float(dmm.s7) * (float)mn[7]);

        for (int l = 0; l < 32; l += 4) {
            int ki = i + l;
            // 16 bytes per lane: eight rows at this K-group, matching the GEMV's uint4
            ushort8 bits = vload8(0, src0_q + row0 + (ki >> 2) * m);
            COK_R8_ACC(0)
            COK_R8_ACC(1)
            COK_R8_ACC(2)
            COK_R8_ACC(3)
        }
    }

    local float8 reduceLM[COK_SG * (COK_NSG - 1)];
    float8 o0, o1, o2, o3, o4, o5, o6, o7;
    COK_R8_RED(acc0, o0)
    COK_R8_RED(acc1, o1)
    COK_R8_RED(acc2, o2)
    COK_R8_RED(acc3, o3)
    COK_R8_RED(acc4, o4)
    COK_R8_RED(acc5, o5)
    COK_R8_RED(acc6, o6)
    COK_R8_RED(acc7, o7)

    if (sg == 0) {
        int idx = row0;
        if (idx < m*n_no_padding) { vstore8((float8)(o0.s0,o1.s0,o2.s0,o3.s0,o4.s0,o5.s0,o6.s0,o7.s0), 0, dst + idx); idx += m; }
        if (idx < m*n_no_padding) { vstore8((float8)(o0.s1,o1.s1,o2.s1,o3.s1,o4.s1,o5.s1,o6.s1,o7.s1), 0, dst + idx); idx += m; }
        if (idx < m*n_no_padding) { vstore8((float8)(o0.s2,o1.s2,o2.s2,o3.s2,o4.s2,o5.s2,o6.s2,o7.s2), 0, dst + idx); idx += m; }
        if (idx < m*n_no_padding) { vstore8((float8)(o0.s3,o1.s3,o2.s3,o3.s3,o4.s3,o5.s3,o6.s3,o7.s3), 0, dst + idx); idx += m; }
        if (idx < m*n_no_padding) { vstore8((float8)(o0.s4,o1.s4,o2.s4,o3.s4,o4.s4,o5.s4,o6.s4,o7.s4), 0, dst + idx); idx += m; }
        if (idx < m*n_no_padding) { vstore8((float8)(o0.s5,o1.s5,o2.s5,o3.s5,o4.s5,o5.s5,o6.s5,o7.s5), 0, dst + idx); idx += m; }
        if (idx < m*n_no_padding) { vstore8((float8)(o0.s6,o1.s6,o2.s6,o3.s6,o4.s6,o5.s6,o6.s6,o7.s6), 0, dst + idx); idx += m; }
        if (idx < m*n_no_padding) { vstore8((float8)(o0.s7,o1.s7,o2.s7,o3.s7,o4.s7,o5.s7,o6.s7,o7.s7), 0, dst + idx); }
    }
}

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#endif

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_128
#endif
kernel void kernel_gemm_noshuffle_q6_K_f32(
        global const ushort * src0_ql,
        global const uchar  * src0_qh,
        global const ushort * src0_s,
        global const half   * src0_d,
        read_only image1d_buffer_t src1,
        global float * dst,
        ulong offsetd,
        int m,
        int n,
        int k,
        int n_no_padding,
        ushort mask_f000,
        uchar  mask_c0,
        int m_dst
) {
    dst = (global float *)( (global char *)dst + offsetd );

    int m_4 = m >> 2;
    int n_4 = n >> 2;

    int gy = get_global_id(0); // n
    int gx = get_global_id(1); // m
    int gx_2 = gx << 2;

    half8 c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    half8 B;
    half4 dequantized_weights;

    global const ushort * ptr_ql = src0_ql + gx_2;
    global const uchar  * ptr_qh = src0_qh + gx_2;
    global const ushort * ptr_s  = src0_s  + gx_2;
    global const half   * ptr_d  = src0_d  + gx_2;

    for (int i = 0; i < k; i += 4) {
        // load 4x elements (ushort) of ql on M, each ushort contains 4 weights
        // 4x ushort correspons to 4 rows on M
        ushort4 bits4 = vload4(0, ptr_ql + (i/4)*m); // ql packed in 4s in ushort
        uchar4  bits2 = vload4(0, ptr_qh + (i/4)*m); // qh packed in 4s in uchar

        // load 4 consecutive scales
        char8 scale_s_8 = as_char8(vload4(0, ptr_s + (i/16/2)*m)); // 1 char scale every 16 elements, packed in 2s
        char4   scale_s = ((i/16) % 2) == 0 ? scale_s_8.s0246 : scale_s_8.s1357; // transposed as ushort, 2 blocks
        half4   scale_d = vload4(0, ptr_d + (i/256)*m);  // 1 half scale every 256 elements

        // j=0
        // load 2x 4 elements of activations on N, corresponding to 8 rows on N
        B.s0123 = read_imageh(src1, gy*2 + (i + 0)*n_4 + 0);
        B.s4567 = read_imageh(src1, gy*2 + (i + 0)*n_4 + 1);
        dequantized_weights.s0 = (convert_half((bits4.s0 & 0x000F) | ((bits2.s0 & 0x03) << 4)) - 32.f) * scale_s.s0 * scale_d.s0;
        dequantized_weights.s1 = (convert_half((bits4.s1 & 0x000F) | ((bits2.s1 & 0x03) << 4)) - 32.f) * scale_s.s1 * scale_d.s1;
        dequantized_weights.s2 = (convert_half((bits4.s2 & 0x000F) | ((bits2.s2 & 0x03) << 4)) - 32.f) * scale_s.s2 * scale_d.s2;
        dequantized_weights.s3 = (convert_half((bits4.s3 & 0x000F) | ((bits2.s3 & 0x03) << 4)) - 32.f) * scale_s.s3 * scale_d.s3;
        c0 += B * dequantized_weights.s0;
        c1 += B * dequantized_weights.s1;
        c2 += B * dequantized_weights.s2;
        c3 += B * dequantized_weights.s3;

        // j=1
        B.s0123 = read_imageh(src1, gy*2 + (i + 1)*n_4 + 0);
        B.s4567 = read_imageh(src1, gy*2 + (i + 1)*n_4 + 1);
        dequantized_weights.s0 = (convert_half((((bits4.s0 & 0x00F0) >> 4) | ((bits2.s0 & 0x0C) << 2))) - 32.f) * scale_s.s0 * scale_d.s0;
        dequantized_weights.s1 = (convert_half((((bits4.s1 & 0x00F0) >> 4) | ((bits2.s1 & 0x0C) << 2))) - 32.f) * scale_s.s1 * scale_d.s1;
        dequantized_weights.s2 = (convert_half((((bits4.s2 & 0x00F0) >> 4) | ((bits2.s2 & 0x0C) << 2))) - 32.f) * scale_s.s2 * scale_d.s2;
        dequantized_weights.s3 = (convert_half((((bits4.s3 & 0x00F0) >> 4) | ((bits2.s3 & 0x0C) << 2))) - 32.f) * scale_s.s3 * scale_d.s3;
        c0 += B * dequantized_weights.s0;
        c1 += B * dequantized_weights.s1;
        c2 += B * dequantized_weights.s2;
        c3 += B * dequantized_weights.s3;

        // j=2
        B.s0123 = read_imageh(src1, gy*2 + (i + 2)*n_4 + 0);
        B.s4567 = read_imageh(src1, gy*2 + (i + 2)*n_4 + 1);
        dequantized_weights.s0 = (convert_half((((bits4.s0 & 0x0F00) >> 8) | (bits2.s0 & 0x30))) - 32.f) * scale_s.s0 * scale_d.s0;
        dequantized_weights.s1 = (convert_half((((bits4.s1 & 0x0F00) >> 8) | (bits2.s1 & 0x30))) - 32.f) * scale_s.s1 * scale_d.s1;
        dequantized_weights.s2 = (convert_half((((bits4.s2 & 0x0F00) >> 8) | (bits2.s2 & 0x30))) - 32.f) * scale_s.s2 * scale_d.s2;
        dequantized_weights.s3 = (convert_half((((bits4.s3 & 0x0F00) >> 8) | (bits2.s3 & 0x30))) - 32.f) * scale_s.s3 * scale_d.s3;
        c0 += B * dequantized_weights.s0;
        c1 += B * dequantized_weights.s1;
        c2 += B * dequantized_weights.s2;
        c3 += B * dequantized_weights.s3;

        // j=3
        B.s0123 = read_imageh(src1, gy*2 + (i + 3)*n_4 + 0);
        B.s4567 = read_imageh(src1, gy*2 + (i + 3)*n_4 + 1);
        dequantized_weights.s0 = (convert_half((((bits4.s0 & mask_f000) >> 12) | ((bits2.s0 & mask_c0) >> 2))) - 32.f) * scale_s.s0 * scale_d.s0;
        dequantized_weights.s1 = (convert_half((((bits4.s1 & mask_f000) >> 12) | ((bits2.s1 & mask_c0) >> 2))) - 32.f) * scale_s.s1 * scale_d.s1;
        dequantized_weights.s2 = (convert_half((((bits4.s2 & mask_f000) >> 12) | ((bits2.s2 & mask_c0) >> 2))) - 32.f) * scale_s.s2 * scale_d.s2;
        dequantized_weights.s3 = (convert_half((((bits4.s3 & mask_f000) >> 12) | ((bits2.s3 & mask_c0) >> 2))) - 32.f) * scale_s.s3 * scale_d.s3;
        c0 += B * dequantized_weights.s0;
        c1 += B * dequantized_weights.s1;
        c2 += B * dequantized_weights.s2;
        c3 += B * dequantized_weights.s3;
    }

    // Store stride is m_dst = the REAL dst row count. m (above) is the weight's
    // (possibly padded-to-128) row stride used for the src0 layout. They differ only
    // for a non-128-multiple ne01 (e.g. an odd-vocab lm_head); then the last row-tile
    // straddles the real/padded boundary and its padded rows must not be written
    // (a vstore4 there would spill into the next column at the real stride). For the
    // common m_dst == m case rv == 4 and the vstore4 fast path is taken unchanged.
    int row = gx << 2;
    int col0 = gy << 3;
    int rv = m_dst - row;              // real rows in this 4-row tile
    if (rv > 4) rv = 4;
    if (rv > 0) {
        #define ST_COL(I, sel) \
            if (col0 + (I) < n_no_padding) { \
                int b = (col0 + (I)) * m_dst + row; \
                if (rv == 4) { \
                    vstore4((float4)(c0.sel, c1.sel, c2.sel, c3.sel), 0, dst + b); \
                } else { \
                    dst[b] = c0.sel; \
                    if (rv > 1) dst[b + 1] = c1.sel; \
                    if (rv > 2) dst[b + 2] = c2.sel; \
                } \
            }
        ST_COL(0, s0) ST_COL(1, s1) ST_COL(2, s2) ST_COL(3, s3)
        ST_COL(4, s4) ST_COL(5, s5) ST_COL(6, s6) ST_COL(7, s7)
        #undef ST_COL
    }
}

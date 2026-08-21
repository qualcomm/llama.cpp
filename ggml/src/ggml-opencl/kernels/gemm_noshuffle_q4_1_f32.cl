#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
#endif

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_128
#endif

kernel void kernel_gemm_noshuffle_q4_1_f32(
    global const ushort * src0_q,
    global const half  * src0_d,
    global const half  * src0_m,
    read_only image1d_buffer_t src1,
    global float * dst,
    ulong offsetd,
    int m,
    int n,
    int k,
    int n_no_padding
) {
    dst = (global float *)((global char *)dst + offsetd);

    int m_4 = m >> 2;
    int n_4 = n >> 2;

    int gy = get_global_id(0);
    int gx = get_global_id(1);
    int gx_2 = gx << 2;

    half8 c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    half8 B;
    half4 dequantized_weights;

    global const ushort* weight_ptr = src0_q + gx_2;
    global const half*   scale_ptr  = src0_d + gx_2;
    global const half*   min_ptr    = src0_m + gx_2;

    for(int i = 0; i < k; i += 4) {
        B.s0123 = read_imageh(src1, gy*2 + (i)*(n_4));
        B.s4567 = read_imageh(src1, gy*2 + (i)*(n_4)+1);

        ushort4 bits4 = vload4(0, weight_ptr + (i/4)*(m));

        half4 scale = vload4(0, scale_ptr + (i/32)*(m));
        half4 minv  = vload4(0,   min_ptr + (i/32)*(m));

        // j=0
        dequantized_weights.s0 = (bits4.s0 & (0x000F)) * scale.s0 + minv.s0;
        dequantized_weights.s1 = (bits4.s1 & (0x000F)) * scale.s1 + minv.s1;
        dequantized_weights.s2 = (bits4.s2 & (0x000F)) * scale.s2 + minv.s2;
        dequantized_weights.s3 = (bits4.s3 & (0x000F)) * scale.s3 + minv.s3;
        c0 += B * dequantized_weights.s0;
        c1 += B * dequantized_weights.s1;
        c2 += B * dequantized_weights.s2;
        c3 += B * dequantized_weights.s3;

        // j=1
        B.s0123 = read_imageh(src1, gy*2 + (i+1)*(n_4));
        B.s4567 = read_imageh(src1, gy*2 + (i+1)*(n_4)+1);
        dequantized_weights.s0 = ((bits4.s0 & (0x00F0)) >> 4) * scale.s0 + minv.s0;
        dequantized_weights.s1 = ((bits4.s1 & (0x00F0)) >> 4) * scale.s1 + minv.s1;
        dequantized_weights.s2 = ((bits4.s2 & (0x00F0)) >> 4) * scale.s2 + minv.s2;
        dequantized_weights.s3 = ((bits4.s3 & (0x00F0)) >> 4) * scale.s3 + minv.s3;
        c0 += B * dequantized_weights.s0;
        c1 += B * dequantized_weights.s1;
        c2 += B * dequantized_weights.s2;
        c3 += B * dequantized_weights.s3;

        // j=2
        B.s0123 = read_imageh(src1, gy*2 + (i+2)*(n_4));
        B.s4567 = read_imageh(src1, gy*2 + (i+2)*(n_4)+1);
        dequantized_weights.s0 = ((bits4.s0 & (0x0F00)) >> 8) * scale.s0 + minv.s0;
        dequantized_weights.s1 = ((bits4.s1 & (0x0F00)) >> 8) * scale.s1 + minv.s1;
        dequantized_weights.s2 = ((bits4.s2 & (0x0F00)) >> 8) * scale.s2 + minv.s2;
        dequantized_weights.s3 = ((bits4.s3 & (0x0F00)) >> 8) * scale.s3 + minv.s3;
        c0 += B * dequantized_weights.s0;
        c1 += B * dequantized_weights.s1;
        c2 += B * dequantized_weights.s2;
        c3 += B * dequantized_weights.s3;

        // j=3
        B.s0123 = read_imageh(src1, gy*2 + (i+3)*(n_4));
        B.s4567 = read_imageh(src1, gy*2 + (i+3)*(n_4)+1);
        dequantized_weights.s0 = ((bits4.s0 & (0xF000)) >> 12) * scale.s0 + minv.s0;
        dequantized_weights.s1 = ((bits4.s1 & (0xF000)) >> 12) * scale.s1 + minv.s1;
        dequantized_weights.s2 = ((bits4.s2 & (0xF000)) >> 12) * scale.s2 + minv.s2;
        dequantized_weights.s3 = ((bits4.s3 & (0xF000)) >> 12) * scale.s3 + minv.s3;
        c0 += B * dequantized_weights.s0;
        c1 += B * dequantized_weights.s1;
        c2 += B * dequantized_weights.s2;
        c3 += B * dequantized_weights.s3;
    }

    int idx = (gy<<3)*m + (gx<<2);

    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s0, c1.s0, c2.s0, c3.s0), 0, dst + idx);
        idx += m;
    }
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s1, c1.s1, c2.s1, c3.s1), 0, dst + idx);
        idx += m;
    }
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s2, c1.s2, c2.s2, c3.s2), 0, dst + idx);
        idx += m;
    }
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s3, c1.s3, c2.s3, c3.s3), 0, dst + idx);
        idx += m;
    }
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s4, c1.s4, c2.s4, c3.s4), 0, dst + idx);
        idx += m;
    }
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s5, c1.s5, c2.s5, c3.s5), 0, dst + idx);
        idx += m;
    }
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s6, c1.s6, c2.s6, c3.s6), 0, dst + idx);
        idx += m;
    }
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s7, c1.s7, c2.s7, c3.s7), 0, dst + idx);
    }
}

// Cooperative-K GEMM for the small-batch (n_q in [2..8]) path -- the q4_1 twin of
// kernel_gemm_noshuffle_q4_0_f32_cok. A WG is (COK_SG lanes x COK_NSG subgroups); each lane
// owns ONE output row and computes its 8 (padded) columns, and the COK_NSG subgroups SPLIT the
// K reduction round-robin, combining through __local.
//
// Why this exists: q4_1 had no small-batch kernel at all. mc3 covers n_q 2..4 but is OPT-IN and
// measured as a regression when defaulted on (it cannot read its K-split from
// get_local_size(1), so it trades nothing for its low workgroup count), and the dense dp4a path
// needs N > 8. So every n_q in 2..8 fell through to the tiled GEMM above, which walks all of K
// serially in each work-item and launches a degenerate workgroup count for a narrow output.
//
// Measured on Adreno X2-90, Qwen3.8-27B-Q4_0 -- a file that is 83% q4_0 but whose eight
// ffn_down tensors are q4_1 -- profiling a speculative-decode round: the SAME ffn_out op cost
// 832 us/call on the q4_0 cooperative-K kernel and 2880 us/call here, so 2.8% of the weights
// were taking 8.1% of decode GPU time.
//
// q4_1 vs q4_0: the block carries a min alongside the scale and the nibble has no -8 bias, so
// the dequant is (nibble * scale + min) and there is one extra half4 load per 32-element block.
// The layout of src0_m matches src0_d exactly.
#define COK_NSG 8
#define COK_SG  64
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_gemm_noshuffle_q4_1_f32_cok(
        global const ushort * src0_q,       // quantized A (4 nibbles per ushort, row-strided)
        global const half   * src0_d,       // A scales (one per 32 elements per row)
        global const half   * src0_m,       // A mins   (same layout as src0_d)
        read_only image1d_buffer_t src1,    // B (1d image)
        global float * dst,                 // C
        ulong offsetd,
        int m,                              // M
        int n,                              // N with padding (8 on this path)
        int k,                              // K
        int n_no_padding                    // N without padding
) {
    dst = (global float *)((global char *)dst + offsetd);

    int n_4  = n >> 2;
    int gx   = get_global_id(0);     // output row
    int sg   = get_local_id(1);      // subgroup index (K-split lane)
    int lane = get_local_id(0);      // lane within subgroup (0..COK_SG-1)

    int num_32blk = k / 32;

    global const ushort * weight_ptr = src0_q + gx;
    global const half   * scale_ptr  = src0_d + gx;
    global const half   * min_ptr    = src0_m + gx;

    half8 acc = 0;
    half8 B;
    half  dq;

    for (int blk = sg; blk < num_32blk; blk += COK_NSG) {
        int i = blk << 5;            // blk * 32

        half scale = scale_ptr[blk * m];
        half minv  = min_ptr  [blk * m];

        for (int l = 0; l < 32; l += 4) {
            int ki = i + l;
            ushort bits = weight_ptr[(ki >> 2) * m];

            B.s0123 = read_imageh(src1,     (ki+0) * n_4);
            B.s4567 = read_imageh(src1, 1 + (ki+0) * n_4);
            dq = (half)((bits & 0x000F) * scale + minv);
            acc += B * dq;

            B.s0123 = read_imageh(src1,     (ki+1) * n_4);
            B.s4567 = read_imageh(src1, 1 + (ki+1) * n_4);
            dq = (half)(((bits & 0x00F0) >> 4) * scale + minv);
            acc += B * dq;

            B.s0123 = read_imageh(src1,     (ki+2) * n_4);
            B.s4567 = read_imageh(src1, 1 + (ki+2) * n_4);
            dq = (half)(((bits & 0x0F00) >> 8) * scale + minv);
            acc += B * dq;

            B.s0123 = read_imageh(src1,     (ki+3) * n_4);
            B.s4567 = read_imageh(src1, 1 + (ki+3) * n_4);
            dq = (half)(((bits & 0xF000) >> 12) * scale + minv);
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

// 4 output rows per lane. The 1-row kernel above loads ONE ushort (2 bytes) per lane per 4
// K-values where a GEMV loads 16, so it is ISSUE bound rather than bandwidth bound: the read is
// already perfectly coalesced, it is just narrow. Four adjacent rows are contiguous in the
// packed layout, so they vector-load as one 8-byte read AND share the activation vector B,
// which quarters the image reads too. Same change that took the q4_K and q4_0 cooperative-K
// kernels +32.9% and +66%.
//
// The four rows' scales and mins are one half4 load each.
//
// Needs m % 4 == 0; the host's m % 64 == 0 gate already implies it.
//
// Every dequant is cast to half BEFORE the multiply: E17.51 (Adreno 850) rejects
// `half8 * float` and kills the whole program, and that is a failure an env-gated A/B cannot
// see, because the kernel simply never builds.
REQD_SUBGROUP_SIZE_64
kernel void kernel_gemm_noshuffle_q4_1_f32_cok_r4(
        global const ushort * src0_q,
        global const half   * src0_d,
        global const half   * src0_m,
        read_only image1d_buffer_t src1,
        global float * dst,
        ulong offsetd,
        int m,
        int n,
        int k,
        int n_no_padding
) {
    dst = (global float *)((global char *)dst + offsetd);

    int n_4  = n >> 2;
    int gx   = get_global_id(0);     // 4-row group
    int sg   = get_local_id(1);
    int lane = get_local_id(0);

    int row0 = gx << 2;              // first of this lane's 4 rows
    int num_32blk = k / 32;

    half8 acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
    half8 B;

    for (int blk = sg; blk < num_32blk; blk += COK_NSG) {
        int i = blk << 5;

        half4 scale = vload4(0, src0_d + row0 + blk * m);
        half4 minv  = vload4(0, src0_m + row0 + blk * m);

        for (int l = 0; l < 32; l += 4) {
            int ki = i + l;
            ushort4 bits = vload4(0, src0_q + row0 + (ki >> 2) * m);

            B.s0123 = read_imageh(src1,     (ki+0) * n_4);
            B.s4567 = read_imageh(src1, 1 + (ki+0) * n_4);
            acc0 += B * (half)(( bits.s0        & 0x000F) * scale.s0 + minv.s0);
            acc1 += B * (half)(( bits.s1        & 0x000F) * scale.s1 + minv.s1);
            acc2 += B * (half)(( bits.s2        & 0x000F) * scale.s2 + minv.s2);
            acc3 += B * (half)(( bits.s3        & 0x000F) * scale.s3 + minv.s3);

            B.s0123 = read_imageh(src1,     (ki+1) * n_4);
            B.s4567 = read_imageh(src1, 1 + (ki+1) * n_4);
            acc0 += B * (half)((((bits.s0 & 0x00F0) >> 4)) * scale.s0 + minv.s0);
            acc1 += B * (half)((((bits.s1 & 0x00F0) >> 4)) * scale.s1 + minv.s1);
            acc2 += B * (half)((((bits.s2 & 0x00F0) >> 4)) * scale.s2 + minv.s2);
            acc3 += B * (half)((((bits.s3 & 0x00F0) >> 4)) * scale.s3 + minv.s3);

            B.s0123 = read_imageh(src1,     (ki+2) * n_4);
            B.s4567 = read_imageh(src1, 1 + (ki+2) * n_4);
            acc0 += B * (half)((((bits.s0 & 0x0F00) >> 8)) * scale.s0 + minv.s0);
            acc1 += B * (half)((((bits.s1 & 0x0F00) >> 8)) * scale.s1 + minv.s1);
            acc2 += B * (half)((((bits.s2 & 0x0F00) >> 8)) * scale.s2 + minv.s2);
            acc3 += B * (half)((((bits.s3 & 0x0F00) >> 8)) * scale.s3 + minv.s3);

            B.s0123 = read_imageh(src1,     (ki+3) * n_4);
            B.s4567 = read_imageh(src1, 1 + (ki+3) * n_4);
            acc0 += B * (half)((((bits.s0 & 0xF000) >> 12)) * scale.s0 + minv.s0);
            acc1 += B * (half)((((bits.s1 & 0xF000) >> 12)) * scale.s1 + minv.s1);
            acc2 += B * (half)((((bits.s2 & 0xF000) >> 12)) * scale.s2 + minv.s2);
            acc3 += B * (half)((((bits.s3 & 0xF000) >> 12)) * scale.s3 + minv.s3);
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

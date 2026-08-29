#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
// _64 is what the cok kernel below needs; this file only ever defined _128, so using it
// was an undefined identifier and broke the WHOLE program build for q8_0.
#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
#endif

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_128
#endif

kernel void kernel_gemm_noshuffle_q8_0_f32(
        global const uint * src0_q,
        global const half  * src0_d,
        __read_only image1d_buffer_t src1,
        global float * dst,
        int k,
        int m,
        int n,
        int n_no_padding,
        ulong offsetd
) {

    int m_4 = m >> 2;
    int n_4 = n >> 2;

    int gy   = get_global_id(0);
    int gx   = get_global_id(1);
    int gx_2 = gx << 2;
    dst  = (global float *)((global char*)dst  + offsetd);


    half8 c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    half8 B;
    half4 deq;

    __global const uint* wptr = src0_q + gx_2;
    __global const half* sptr = src0_d + gx_2;

      for (int i = 0; i < k; i += 4) {
        uint4 pack4 = vload4(0, wptr + (i / 4) * m);
        half4 scale = vload4(0, sptr + (i / 32) * m);

        char4 p0 = as_char4(pack4.s0);
        char4 p1 = as_char4(pack4.s1);
        char4 p2 = as_char4(pack4.s2);
        char4 p3 = as_char4(pack4.s3);

        // ------------------- j = 0 (k = i+0) -------------------
        B.s0123 = read_imageh(src1, gy * 2 + (i + 0) * n_4);
        B.s4567 = read_imageh(src1, gy * 2 + (i + 0) * n_4 + 1);

        half4 wj0 = convert_half4((char4)(p0.s0, p1.s0, p2.s0, p3.s0)) * scale;

        c0 += B * wj0.s0;
        c1 += B * wj0.s1;
        c2 += B * wj0.s2;
        c3 += B * wj0.s3;

        // ------------------- j = 1 (k = i+1) -------------------
        B.s0123 = read_imageh(src1, gy * 2 + (i + 1) * n_4);
        B.s4567 = read_imageh(src1, gy * 2 + (i + 1) * n_4 + 1);

        half4 wj1 = convert_half4((char4)(p0.s1, p1.s1, p2.s1, p3.s1)) * scale;

        c0 += B * wj1.s0;
        c1 += B * wj1.s1;
        c2 += B * wj1.s2;
        c3 += B * wj1.s3;

        // ------------------- j = 2 (k = i+2) -------------------
        B.s0123 = read_imageh(src1, gy * 2 + (i + 2) * n_4);
        B.s4567 = read_imageh(src1, gy * 2 + (i + 2) * n_4 + 1);

        half4 wj2 = convert_half4((char4)(p0.s2, p1.s2, p2.s2, p3.s2)) * scale;

        c0 += B * wj2.s0;
        c1 += B * wj2.s1;
        c2 += B * wj2.s2;
        c3 += B * wj2.s3;

        // ------------------- j = 3 (k = i+3) -------------------
        B.s0123 = read_imageh(src1, gy * 2 + (i + 3) * n_4);
        B.s4567 = read_imageh(src1, gy * 2 + (i + 3) * n_4 + 1);

        half4 wj3 = convert_half4((char4)(p0.s3, p1.s3, p2.s3, p3.s3)) * scale;

        c0 += B * wj3.s0;
        c1 += B * wj3.s1;
        c2 += B * wj3.s2;
        c3 += B * wj3.s3;
    }

    int idx = (gy << 3) * m + (gx << 2);

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

// Cooperative-K q8_0 GEMM for the small-batch (n_q in [2..8]) path -- the q8_0 twin of the
// q4_K/q4_0/q5_K/q6_K _cok kernels, which q8_0 never had. Until now ne1 in 2..8 fell to the
// PREFILL GEMM above: no mc3, no cok, and the dense dp4a is gated N > 8.
//
// The row tiling is already right and is kept verbatim: 4 output rows per lane, weights read
// as one uint4 (16 B per lane), half8 accumulators over the 8 padded columns. That is exactly
// the shape the q4_K cok had to be rewritten into for +32.9%, so there is nothing to win there.
//
// What is missing is the K parallelism. The prefill launch is
// global{CEIL_DIV(N,8), CEIL_DIV(M,4)} local{2,128}, which at N=8 M=4096 is EIGHT workgroups
// for the whole GEMM -- the SP sits idle. This splits K round-robin across COK_NSG subgroups
// and combines through __local, the same structure as the q4_K cok: M/256 workgroups each with
// 64 lanes x COK_NSG subgroups.
//
// Measured headroom: q8_0 runs 81-86 GB/s at n=2..8 against 123.4 at n=1 through its own GEMV,
// so about 1.45x is on the table and none of it is row tiling.
// Q80_COK_NSG is overridable so the host can narrow it when a device refuses the
// 64 x Q80_COK_NSG workgroup; see ggml_cl_build_cok_program.
#ifndef Q80_COK_NSG
#define Q80_COK_NSG 8
#endif
#define Q80_COK_SG  64
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_gemm_noshuffle_q8_0_f32_cok(
        global const uint  * src0_q,
        global const half  * src0_d,
        __read_only image1d_buffer_t src1,
        global float * dst,
        int k,
        int m,
        int n,
        int n_no_padding,
        ulong offsetd
) {
    int n_4 = n >> 2;

    int gx   = get_global_id(0);   // 4-row group
    int sg   = get_local_id(1);    // K-split index
    int lane = get_local_id(0);
    int gy   = 0;                  // one column tile: this path is ne1 <= 8

    int gx_2 = gx << 2;
    dst = (global float *)((global char*)dst + offsetd);

    half8 c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    half8 B;

    __global const uint * wptr = src0_q + gx_2;
    __global const half * sptr = src0_d + gx_2;

    // each subgroup walks its own slice of K, stepping COK_NSG * 4 elements
    for (int i = sg << 2; i < k; i += Q80_COK_NSG << 2) {
        uint4 pack4 = vload4(0, wptr + (i / 4) * m);
        half4 scale = vload4(0, sptr + (i / 32) * m);

        char4 p0 = as_char4(pack4.s0);
        char4 p1 = as_char4(pack4.s1);
        char4 p2 = as_char4(pack4.s2);
        char4 p3 = as_char4(pack4.s3);

        B.s0123 = read_imageh(src1, gy * 2 + (i + 0) * n_4);
        B.s4567 = read_imageh(src1, gy * 2 + (i + 0) * n_4 + 1);
        half4 wj0 = convert_half4((char4)(p0.s0, p1.s0, p2.s0, p3.s0)) * scale;
        c0 += B * wj0.s0;  c1 += B * wj0.s1;  c2 += B * wj0.s2;  c3 += B * wj0.s3;

        B.s0123 = read_imageh(src1, gy * 2 + (i + 1) * n_4);
        B.s4567 = read_imageh(src1, gy * 2 + (i + 1) * n_4 + 1);
        half4 wj1 = convert_half4((char4)(p0.s1, p1.s1, p2.s1, p3.s1)) * scale;
        c0 += B * wj1.s0;  c1 += B * wj1.s1;  c2 += B * wj1.s2;  c3 += B * wj1.s3;

        B.s0123 = read_imageh(src1, gy * 2 + (i + 2) * n_4);
        B.s4567 = read_imageh(src1, gy * 2 + (i + 2) * n_4 + 1);
        half4 wj2 = convert_half4((char4)(p0.s2, p1.s2, p2.s2, p3.s2)) * scale;
        c0 += B * wj2.s0;  c1 += B * wj2.s1;  c2 += B * wj2.s2;  c3 += B * wj2.s3;

        B.s0123 = read_imageh(src1, gy * 2 + (i + 3) * n_4);
        B.s4567 = read_imageh(src1, gy * 2 + (i + 3) * n_4 + 1);
        half4 wj3 = convert_half4((char4)(p0.s3, p1.s3, p2.s3, p3.s3)) * scale;
        c0 += B * wj3.s0;  c1 += B * wj3.s1;  c2 += B * wj3.s2;  c3 += B * wj3.s3;
    }

    // Reduce across the K-split, ONE ROW AT A TIME so __local stays the size the 1-row q4_K
    // cok uses. Holding all four rows at once would need ~57 KB. Four extra barriers cost
    // nothing here: a WG barrier is 0.154 ns/op on X2 at WG=64 (measured).
    local float8 reduceLM[Q80_COK_SG * (Q80_COK_NSG - 1)];
    float8 o0, o1, o2, o3;

#define Q80_COK_RED(ACC, OUT)                                                        \
    barrier(CLK_LOCAL_MEM_FENCE);                                                    \
    if (sg > 0) { reduceLM[(sg - 1) * Q80_COK_SG + lane] = convert_float8(ACC); }     \
    barrier(CLK_LOCAL_MEM_FENCE);                                                    \
    if (sg == 0) {                                                                   \
        float8 sum = convert_float8(ACC);                                            \
        for (int s = 0; s < Q80_COK_NSG - 1; s++) { sum += reduceLM[s * Q80_COK_SG + lane]; } \
        OUT = sum;                                                                   \
    }

    Q80_COK_RED(c0, o0)
    Q80_COK_RED(c1, o1)
    Q80_COK_RED(c2, o2)
    Q80_COK_RED(c3, o3)
#undef Q80_COK_RED

    if (sg != 0) {
        return;
    }

    // same store as the prefill kernel: dst[col*m + row], four adjacent rows per vstore4
    int idx = gx_2;
    if (idx+3 < m*n_no_padding) { vstore4((float4)(o0.s0, o1.s0, o2.s0, o3.s0), 0, dst + idx); idx += m; }
    if (idx+3 < m*n_no_padding) { vstore4((float4)(o0.s1, o1.s1, o2.s1, o3.s1), 0, dst + idx); idx += m; }
    if (idx+3 < m*n_no_padding) { vstore4((float4)(o0.s2, o1.s2, o2.s2, o3.s2), 0, dst + idx); idx += m; }
    if (idx+3 < m*n_no_padding) { vstore4((float4)(o0.s3, o1.s3, o2.s3, o3.s3), 0, dst + idx); idx += m; }
    if (idx+3 < m*n_no_padding) { vstore4((float4)(o0.s4, o1.s4, o2.s4, o3.s4), 0, dst + idx); idx += m; }
    if (idx+3 < m*n_no_padding) { vstore4((float4)(o0.s5, o1.s5, o2.s5, o3.s5), 0, dst + idx); idx += m; }
    if (idx+3 < m*n_no_padding) { vstore4((float4)(o0.s6, o1.s6, o2.s6, o3.s6), 0, dst + idx); idx += m; }
    if (idx+3 < m*n_no_padding) { vstore4((float4)(o0.s7, o1.s7, o2.s7, o3.s7), 0, dst + idx); }
}

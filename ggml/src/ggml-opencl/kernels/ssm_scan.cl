#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Mamba-1 / Mamba-2 selective state-space scan.
// One work-group per (d_inner, n_head, n_seqs); one work-item per d_state.
// Tree-reduction within WG to compute y = dot(state, C) per token.

kernel void kernel_ssm_scan_f32(
    global const float * src0,   // s     [d_state, d_inner, n_head, n_seqs_total]
    ulong                offset0,
    global const float * src1,   // x     [d_inner, n_head, n_seq_tokens, n_seqs]
    ulong                offset1,
    global const float * src2,   // dt    [n_head, n_seq_tokens, n_seqs]
    ulong                offset2,
    global const float * src3,   // A     [(d_state | 1), n_head]
    ulong                offset3,
    global const float * src4,   // B     [d_state, n_group, n_seq_tokens, n_seqs]
    ulong                offset4,
    global const float * src5,   // C     [d_state, n_group, n_seq_tokens, n_seqs]
    ulong                offset5,
    global const int   * src6,   // ids   [n_seqs]
    ulong                offset6,
    global       float * dst,
    ulong                offsetd,
    int                  d_state,
    int                  d_inner,
    int                  n_head,
    int                  n_group,
    int                  n_seq_tokens,
    int                  n_seqs,
    ulong                s_off,
    int                  A_d_state,
    ulong                nb02,
    ulong                nb03,
    ulong                nb11,
    ulong                nb12,
    ulong                nb13,
    ulong                nb21,
    ulong                nb22,
    ulong                nb31,
    ulong                nb41,
    ulong                nb42,
    ulong                nb43,
    ulong                nb51,
    ulong                nb52,
    ulong                nb53,
    local        float * shared
) {
    src0 = (global const float *)((global const char *) src0 + offset0);
    src1 = (global const float *)((global const char *) src1 + offset1);
    src2 = (global const float *)((global const char *) src2 + offset2);
    src3 = (global const float *)((global const char *) src3 + offset3);
    src4 = (global const float *)((global const char *) src4 + offset4);
    src5 = (global const float *)((global const char *) src5 + offset5);
    src6 = (global const int   *)((global const char *) src6 + offset6);
    dst  = (global       float *)((global       char *) dst  + offsetd);

    const int i0 = get_local_id(0);   // d_state index
    const int i1 = get_group_id(0);   // d_inner index
    const int ir = get_group_id(1);   // head index
    const int i3 = get_group_id(2);   // seq index

    const int g  = ir / (n_head / n_group);

    const int seq_id = src6[i3];

    global const float * s0_buf = (global const float *)((global const char *) src0 + ir*nb02 + seq_id*nb03);
    global       float * s_buf  = (global       float *)((global       char *) dst  + ir*nb02 + i3    *nb03 + s_off);

    const int i = i0 + i1 * d_state;

    float s0 = s0_buf[i];
    float s  = 0.0f;

    global const float * A = (global const float *)((global const char *) src3 + ir * nb31);
    const float A_val = A[i0 % A_d_state];

    const ulong elem = sizeof(float);

    global const float * x  = (global const float *)((global const char *) src1 + i1*elem + ir*nb11 + i3*nb13);
    global const float * dt = (global const float *)((global const char *) src2 + ir*elem + i3*nb22);
    global const float * B  = (global const float *)((global const char *) src4 + g *nb41 + i3*nb43);
    global const float * C  = (global const float *)((global const char *) src5 + g *nb51 + i3*nb53);

    global float * y = dst + (i1 + ir * d_inner + i3 * n_seq_tokens * n_head * d_inner);

    const int ns12 = (int)(nb12 / elem);
    const int ns21 = (int)(nb21 / elem);
    const int ns42 = (int)(nb42 / elem);
    const int ns52 = (int)(nb52 / elem);

    for (int i2 = 0; i2 < n_seq_tokens; ++i2) {
        const float dt_v   = dt[i2 * ns21];
        const float dt_sp  = dt_v <= 20.0f ? log(1.0f + exp(dt_v)) : dt_v;
        const float x_dt   = x[i2 * ns12] * dt_sp;
        const float dA     = exp(dt_sp * A_val);
        const float B_v    = B[i0 + i2 * ns42];
        const float C_v    = C[i0 + i2 * ns52];

        s = s0 * dA + B_v * x_dt;

        shared[i0] = s * C_v;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int stride = d_state / 2; stride > 0; stride >>= 1) {
            if (i0 < stride) {
                shared[i0] += shared[i0 + stride];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (i0 == 0) {
            y[i2 * n_head * d_inner] = shared[0];
        }

        s0 = s;
    }

    s_buf[i] = s;
}

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_qcom_kernel_cooperative_matrix : enable

// Fused flash attention (Adreno 850) with cl_qcom_kernel_cooperative_matrix, DK = DV = 128.
// One wave = 64 rows; lane r owns one (query token, head) row, so the online softmax is per lane.
//
//   F == 1 (prefill): dim 1 = head h, rows = 64 consecutive query tokens of that head.
//   F  > 1 (decode / small batch, GQA fold): dim 1 = kv head hk, row r = token r / F, head hk*F + r % F,
//                     so the F query heads that share a kv head read K/V once. 64/F tokens per wave.
//   dim 2 = KV split. With n_split > 1 each split writes an unnormalized partial O and its (m, l) to
//   part_o / part_ml, and kernel_fa_coopmat_reduce combines them.
//
// q    f32 [DK, n_q, n_head]          element (k, i, h) at q + i*nbq1 + h*nbq2         (bytes)
// k,v  f16 [DK, n_kv, n_head_kv]      element (k, j, h) at k + j*nbk1 + h*nbk2
// mask f16 [n_kv, n_q]                element (j, i)    at mask + i*nbm1
// dst  f32 [DV, n_head, n_q]          element (d, h, i) at dst + ((i*n_head + h)*DV + d)*4
// n_kv must be a multiple of 64 (llama pads the KV to 256).

#define DK 128
#define DV 128

typedef half mat_a_t __attribute__((qcom_coop_mat(CLK_COOPERATIVE_MATRIX_SCOPE_SUBGROUP_QCOM, 64, 16, CLK_COOPERATIVE_MATRIX_A_QCOM)));
typedef half mat_b_t __attribute__((qcom_coop_mat(CLK_COOPERATIVE_MATRIX_SCOPE_SUBGROUP_QCOM, 16, 64, CLK_COOPERATIVE_MATRIX_B_QCOM)));
typedef half mat_c_t __attribute__((qcom_coop_mat(CLK_COOPERATIVE_MATRIX_SCOPE_SUBGROUP_QCOM, 64, 64, CLK_COOPERATIVE_MATRIX_ACCUMULATOR_QCOM)));

typedef half half1024 __attribute__((ext_vector_type(1024)));
typedef half half4096 __attribute__((ext_vector_type(4096)));

typedef union { half1024 v; half16 h; } a_row_t;
typedef union { half4096 v; struct { half16 h0, h1, h2, h3; } s; } c_row_t;

// Built with -cl-fast-relaxed-math (the softmax is exp-bound), which lets the compiler assume no
// infinities. The mask's -inf is therefore rewritten to the most negative finite half with integer
// ops, and the running max starts at a finite value: a masked score then contributes exp(-65504 - m) == 0
// as soon as the row (or another KV split) has one unmasked score.
#define NEG_BIG (-1e30f)
#define Q_LD    (DK + 8)   // local Q row pitch in halves

inline float16 load_mask16(global const half * p) {
    ushort16 b = vload16(0, (global const ushort *) p);
    b = select(b, (ushort16)(0xFBFF), b == (ushort16)(0xFC00));
    return convert_float16(as_half16(b));
}

kernel void kernel_fa_coopmat_dk128(
        global const char * q,    ulong offq, ulong nbq1, ulong nbq2,
        global const char * k,    ulong offk, ulong nbk1, ulong nbk2,
        global const char * v,    ulong offv, ulong nbv1, ulong nbv2,
        global const char * mask, ulong offm, ulong nbm1,
        global       char * dst_base, ulong offd,
        global       float * part_o,
        global       float * part_ml,
        int n_q, int n_kv, int n_head, int n_head_kv, float scale,
        int F_arg, int kv_per_split
) {
    const int F = F_arg;
    q    += offq;
    k    += offk;
    v    += offv;
    mask += offm;
    global float * dst = (global float *)(dst_base + offd);

    const int lane   = get_local_id(0);
    const int split  = get_group_id(2);
    const int n_split = get_num_groups(2);
    const int T      = 64 / F;                       // query tokens per wave

    int h, hk;
    if (F == 1) {
        h  = get_group_id(1);
        hk = h / (n_head / n_head_kv);
    } else {
        hk = get_group_id(1);
        h  = hk * F + lane % F;
    }
    const int  i     = get_group_id(0) * T + lane / F;
    const bool valid = lane < T * F && i < n_q;
    const int  ic    = min(i, n_q - 1);

    // Q (pre-scaled, f16) in local memory, row = lane: keeping it as 8 A operands in registers spills
    local half q_lds[64 * Q_LD];
    {
        global const float * qr = (global const float *)(q + (ulong)ic * nbq1 + (ulong)h * nbq2);
        for (int kk = 0; kk < DK / 16; kk++) {
            vstore16(convert_half16(vload16(0, qr + kk * 16) * scale), 0, q_lds + lane * Q_LD + kk * 16);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    global const half * kb = (global const half *)(k + (ulong)hk * nbk2);
    global const half * vb = (global const half *)(v + (ulong)hk * nbv2);
    const int ldk = (int)(nbk1 / 2);
    const int ldv = (int)(nbv1 / 2);
    global const half * mr = (global const half *)(mask + (ulong)ic * nbm1);

    mat_c_t o0 = qcom_coop_mat_init(0.0);
    mat_c_t o1 = qcom_coop_mat_init(0.0);
    float m = NEG_BIG;
    float l = 0.0f;

    const int j_beg = split * kv_per_split;
    const int j_end = min(n_kv, j_beg + kv_per_split);

    for (int j0 = j_beg; j0 < j_end; j0 += 64) {
        mat_c_t s = qcom_coop_mat_init(0.0);
        for (int kk = 0; kk < DK / 16; kk++) {
            a_row_t qk;
            qk.h = vload16(0, q_lds + lane * Q_LD + kk * 16);
            mat_a_t a = __builtin_astype(qk.v, mat_a_t);
            mat_b_t b = qcom_coop_mat_load(kb + (ulong)j0 * ldk + kk * 16, CLK_COOPERATIVE_MATRIX_LAYOUT_COLUMN_MAJOR_QCOM, ldk);
            s = qcom_coop_mat_mulAdd(a, b, s, CLK_COOPERATIVE_MATRIX_OPERAND_NONE_QCOM);
        }

        c_row_t us;
        us.v = __builtin_astype(s, half4096);

#ifndef FA_STREAM_P
        // prefill build: all 64 probabilities at once, then the four P.V updates
        float16 s0 = convert_float16(us.s.h0) + load_mask16(mr + j0 +  0);
        float16 s1 = convert_float16(us.s.h1) + load_mask16(mr + j0 + 16);
        float16 s2 = convert_float16(us.s.h2) + load_mask16(mr + j0 + 32);
        float16 s3 = convert_float16(us.s.h3) + load_mask16(mr + j0 + 48);

        float16 mx = fmax(fmax(s0, s1), fmax(s2, s3));
        float8  m8 = fmax(mx.lo, mx.hi);
        float4  m4 = fmax(m8.lo, m8.hi);
        float2  m2 = fmax(m4.lo, m4.hi);
        const float mnew  = fmax(m, fmax(m2.x, m2.y));
        const float alpha = exp(m - mnew);
        m = mnew;

        const float16 p0 = exp(s0 - mnew);
        const float16 p1 = exp(s1 - mnew);
        const float16 p2 = exp(s2 - mnew);
        const float16 p3 = exp(s3 - mnew);
        float16 ps = p0 + p1 + p2 + p3;
        float8  p8 = ps.lo + ps.hi;
        float4  p4 = p8.lo + p8.hi;
        float2  pp = p4.lo + p4.hi;
        l = l * alpha + pp.x + pp.y;

        {
            const half ha = (half)alpha;
            c_row_t u;
            u.v = __builtin_astype(o0, half4096);
            u.s.h0 *= ha; u.s.h1 *= ha; u.s.h2 *= ha; u.s.h3 *= ha;
            o0 = __builtin_astype(u.v, mat_c_t);
            u.v = __builtin_astype(o1, half4096);
            u.s.h0 *= ha; u.s.h1 *= ha; u.s.h2 *= ha; u.s.h3 *= ha;
            o1 = __builtin_astype(u.v, mat_c_t);
        }

        // convert all four P chunks before the P.V updates (converting inside each update is 1.4x slower here)
        a_row_t pa0, pa1, pa2, pa3;
        pa0.h = convert_half16(p0);
        pa1.h = convert_half16(p1);
        pa2.h = convert_half16(p2);
        pa3.h = convert_half16(p3);
        mat_a_t a0 = __builtin_astype(pa0.v, mat_a_t);
        mat_a_t a1 = __builtin_astype(pa1.v, mat_a_t);
        mat_a_t a2 = __builtin_astype(pa2.v, mat_a_t);
        mat_a_t a3 = __builtin_astype(pa3.v, mat_a_t);

        global const half * vr = vb + (ulong)j0 * ldv;
#define PV(a, c) { \
            mat_b_t b0 = qcom_coop_mat_load(vr + (ulong)(c) * 16 * ldv,      CLK_COOPERATIVE_MATRIX_LAYOUT_ROW_MAJOR_QCOM, ldv); \
            mat_b_t b1 = qcom_coop_mat_load(vr + (ulong)(c) * 16 * ldv + 64, CLK_COOPERATIVE_MATRIX_LAYOUT_ROW_MAJOR_QCOM, ldv); \
            o0 = qcom_coop_mat_mulAdd(a, b0, o0, CLK_COOPERATIVE_MATRIX_OPERAND_NONE_QCOM); \
            o1 = qcom_coop_mat_mulAdd(a, b1, o1, CLK_COOPERATIVE_MATRIX_OPERAND_NONE_QCOM); }
        PV(a0, 0)
        PV(a1, 1)
        PV(a2, 2)
        PV(a3, 3)
#undef PV
#else
        // small-batch build: row max first, then one 16-column chunk of P at a time straight into
        // its P.V update, which keeps fewer f32 temporaries live
        float16 mx = fmax(fmax(convert_float16(us.s.h0) + load_mask16(mr + j0 +  0),
                               convert_float16(us.s.h1) + load_mask16(mr + j0 + 16)),
                          fmax(convert_float16(us.s.h2) + load_mask16(mr + j0 + 32),
                               convert_float16(us.s.h3) + load_mask16(mr + j0 + 48)));
        float8  m8 = fmax(mx.lo, mx.hi);
        float4  m4 = fmax(m8.lo, m8.hi);
        float2  m2 = fmax(m4.lo, m4.hi);
        const float mnew  = fmax(m, fmax(m2.x, m2.y));
        const float alpha = exp(m - mnew);
        m = mnew;

        {
            const half ha = (half)alpha;
            c_row_t u;
            u.v = __builtin_astype(o0, half4096);
            u.s.h0 *= ha; u.s.h1 *= ha; u.s.h2 *= ha; u.s.h3 *= ha;
            o0 = __builtin_astype(u.v, mat_c_t);
            u.v = __builtin_astype(o1, half4096);
            u.s.h0 *= ha; u.s.h1 *= ha; u.s.h2 *= ha; u.s.h3 *= ha;
            o1 = __builtin_astype(u.v, mat_c_t);
        }

        // P = exp(S - m) one chunk at a time, straight into its P.V update
        global const half * vr = vb + (ulong)j0 * ldv;
        float lsum = 0.0f;
#define PV_CHUNK(hc, c) { \
            const float16 p = exp(convert_float16(hc) + load_mask16(mr + j0 + (c) * 16) - mnew); \
            const float8 p8 = p.lo + p.hi; const float4 p4 = p8.lo + p8.hi; const float2 p2 = p4.lo + p4.hi; \
            lsum += p2.x + p2.y; \
            a_row_t pa; \
            pa.h = convert_half16(p); \
            mat_a_t a = __builtin_astype(pa.v, mat_a_t); \
            mat_b_t b0 = qcom_coop_mat_load(vr + (ulong)(c) * 16 * ldv,      CLK_COOPERATIVE_MATRIX_LAYOUT_ROW_MAJOR_QCOM, ldv); \
            mat_b_t b1 = qcom_coop_mat_load(vr + (ulong)(c) * 16 * ldv + 64, CLK_COOPERATIVE_MATRIX_LAYOUT_ROW_MAJOR_QCOM, ldv); \
            o0 = qcom_coop_mat_mulAdd(a, b0, o0, CLK_COOPERATIVE_MATRIX_OPERAND_NONE_QCOM); \
            o1 = qcom_coop_mat_mulAdd(a, b1, o1, CLK_COOPERATIVE_MATRIX_OPERAND_NONE_QCOM); }
        PV_CHUNK(us.s.h0, 0)
        PV_CHUNK(us.s.h1, 1)
        PV_CHUNK(us.s.h2, 2)
        PV_CHUNK(us.s.h3, 3)
#undef PV_CHUNK
        l = l * alpha + lsum;
#endif
    }

    if (valid) {
        float il = 1.0f;
        global float * out;
        if (n_split == 1) {
            il  = l > 0.0f ? 1.0f / l : 0.0f;
            out = dst + ((ulong)i * n_head + h) * DV;
        } else {
            const ulong pr = ((ulong)split * n_head + h) * n_q + i;
            out = part_o + pr * DV;
            part_ml[pr * 2 + 0] = m;
            part_ml[pr * 2 + 1] = l;
        }
        c_row_t u;
        u.v = __builtin_astype(o0, half4096);
        vstore16(convert_float16(u.s.h0) * il, 0, out +   0);
        vstore16(convert_float16(u.s.h1) * il, 0, out +  16);
        vstore16(convert_float16(u.s.h2) * il, 0, out +  32);
        vstore16(convert_float16(u.s.h3) * il, 0, out +  48);
        u.v = __builtin_astype(o1, half4096);
        vstore16(convert_float16(u.s.h0) * il, 0, out +  64);
        vstore16(convert_float16(u.s.h1) * il, 0, out +  80);
        vstore16(convert_float16(u.s.h2) * il, 0, out +  96);
        vstore16(convert_float16(u.s.h3) * il, 0, out + 112);
    }
}

// combine the KV splits: global (DV, n_head, n_q)
kernel void kernel_fa_coopmat_reduce(
        global const float * part_o,
        global const float * part_ml,
        global       char  * dst_base,
        ulong                offd,
        int n_q, int n_head, int n_split
) {
    global float * dst = (global float *)(dst_base + offd);
    const int d = get_global_id(0);
    const int h = get_global_id(1);
    const int i = get_global_id(2);

    float M = NEG_BIG;
    for (int s = 0; s < n_split; s++) {
        M = fmax(M, part_ml[(((ulong)s * n_head + h) * n_q + i) * 2]);
    }
    float L = 0.0f, O = 0.0f;
    for (int s = 0; s < n_split; s++) {
        const ulong pr = ((ulong)s * n_head + h) * n_q + i;
        const float w  = exp(part_ml[pr * 2] - M);
        L += w * part_ml[pr * 2 + 1];
        O += w * part_o[pr * DV + d];
    }
    dst[((ulong)i * n_head + h) * DV + d] = L > 0.0f ? O / L : 0.0f;
}

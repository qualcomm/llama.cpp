#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_qcom_kernel_cooperative_matrix : enable

// Batched F16 x F32 GEMM for the attention matmuls (KQ and KQV) with cl_qcom_kernel_cooperative_matrix.
//
//   dst[h][n][m] = sum_k A[h / r2][m][k] * B[h][n][k]
//
// A is f16 with k contiguous (the K cache for KQ, the transposed V cache for KQV), row stride nb01 and
// head stride nb02 in bytes. B is f32 with k contiguous, packed per head into f16 64x16 tiles by
// kernel_pack_b_f16_attn_coopmat. dst is contiguous f32 [ne12][N][M]. One 64x64 C tile per wave.

typedef half mat_a_t __attribute__((qcom_coop_mat(CLK_COOPERATIVE_MATRIX_SCOPE_SUBGROUP_QCOM, 64, 16, CLK_COOPERATIVE_MATRIX_A_QCOM)));
typedef half mat_b_t __attribute__((qcom_coop_mat(CLK_COOPERATIVE_MATRIX_SCOPE_SUBGROUP_QCOM, 16, 64, CLK_COOPERATIVE_MATRIX_B_QCOM)));
typedef half mat_c_t __attribute__((qcom_coop_mat(CLK_COOPERATIVE_MATRIX_SCOPE_SUBGROUP_QCOM, 64, 64, CLK_COOPERATIVE_MATRIX_ACCUMULATOR_QCOM)));

typedef half half1024 __attribute__((ext_vector_type(1024)));
typedef half half4096 __attribute__((ext_vector_type(4096)));

typedef union {
    half1024 v;
    half16   h;
} a_row_t;

typedef union {
    half4096 v;
    struct {
        half16 h0;
        half16 h1;
        half16 h2;
        half16 h3;
    } s;
} c_row_t;

#define PACK_LDS (64 + 8)

// Pack B f32 [ne12][N][K] (row stride nb11, head stride nb12, in bytes) into f16 64x16 tiles
// (column-major, ld 16) at ((h * n_tiles + tn) * (K/16) + ks) * 1024. Columns >= N are zero.
// A workgroup (64, 4) packs 64 columns x 64 k of one head.
kernel void kernel_pack_b_f16_attn_coopmat(
        global const char * src1,
        ulong               offset1,
        ulong               nb11,
        ulong               nb12,
        global half       * dst,
        int                 K,
        int                 N
) {
    local half tile[64 * PACK_LDS];

    const int l       = get_local_id(1) * 64 + get_local_id(0);
    const int tn      = get_group_id(0);
    const int kt      = get_group_id(1);
    const int h       = get_group_id(2);
    const int k0      = kt * 64;
    const int n_tiles = (N + 63) / 64;

    global const char * src = src1 + offset1 + (ulong)h * nb12;

    for (int j = 0; j < 4; j++) {
        const int f = j * 256 + l;
        const int r = f >> 4;
        const int c = (f & 15) * 4;
        const int n = tn * 64 + r;
        float4 v = (float4)(0.0f);
        if (n < N && k0 + c < K) {
            v = vload4(0, (global const float *)(src + (ulong)n * nb11) + k0 + c);
        }
        vstore_half4(v, 0, tile + r * PACK_LDS + c);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    global half * out = dst + (((ulong)h * n_tiles + tn) * (K / 16) + kt * 4) * 1024;
    for (int j = 0; j < 2; j++) {
        const int q  = j * 256 + l;
        const int ks = q >> 7;
        const int n  = (q >> 1) & 63;
        const int hh = q & 1;
        if (kt * 4 + ks < K / 16) {
            half8 v = vload8(0, tile + n * PACK_LDS + ks * 16 + hh * 8);
            vstore8(v, 0, out + q * 8);
        }
    }
}

inline void store_c16(global float * p, half16 h, int j0, int nv, int M) {
    float16 f = convert_float16(h);
    if (j0 +  0 < nv) p[(ulong)(j0 +  0) * M] = f.s0;
    if (j0 +  1 < nv) p[(ulong)(j0 +  1) * M] = f.s1;
    if (j0 +  2 < nv) p[(ulong)(j0 +  2) * M] = f.s2;
    if (j0 +  3 < nv) p[(ulong)(j0 +  3) * M] = f.s3;
    if (j0 +  4 < nv) p[(ulong)(j0 +  4) * M] = f.s4;
    if (j0 +  5 < nv) p[(ulong)(j0 +  5) * M] = f.s5;
    if (j0 +  6 < nv) p[(ulong)(j0 +  6) * M] = f.s6;
    if (j0 +  7 < nv) p[(ulong)(j0 +  7) * M] = f.s7;
    if (j0 +  8 < nv) p[(ulong)(j0 +  8) * M] = f.s8;
    if (j0 +  9 < nv) p[(ulong)(j0 +  9) * M] = f.s9;
    if (j0 + 10 < nv) p[(ulong)(j0 + 10) * M] = f.sa;
    if (j0 + 11 < nv) p[(ulong)(j0 + 11) * M] = f.sb;
    if (j0 + 12 < nv) p[(ulong)(j0 + 12) * M] = f.sc;
    if (j0 + 13 < nv) p[(ulong)(j0 + 13) * M] = f.sd;
    if (j0 + 14 < nv) p[(ulong)(j0 + 14) * M] = f.se;
    if (j0 + 15 < nv) p[(ulong)(j0 + 15) * M] = f.sf;
}

kernel void kernel_mul_mm_f16_f32_attn_coopmat(
        global const char * src0,
        ulong               offset0,
        ulong               nb01,
        ulong               nb02,
        global const half * src1,     // B: packed f16 tiles
        global       char * dst,
        ulong               offsetd,
        int                 K,
        int                 M,
        int                 N,
        int                 ne12,
        int                 r2
) {
    const int lane    = get_local_id(0);
    const int row     = get_global_id(1) * 64;
    const int z       = get_global_id(2);
    const int n_tiles = (N + 63) / 64;

    if (z >= n_tiles * ne12) {
        return;
    }

    const int h  = z / n_tiles;
    const int tn = z - h * n_tiles;

    global const half * ap = (global const half *)(src0 + offset0 + (ulong)(h / r2) * nb02 + (ulong)(row + lane) * nb01);
    global const half * bp = src1 + ((ulong)h * n_tiles + tn) * (K / 16) * 1024;

    mat_c_t acc = qcom_coop_mat_init(0.0);

#pragma unroll 2
    for (int k0 = 0; k0 < K; k0 += 16) {
        a_row_t ua;
        ua.h = vload16(0, ap + k0);
        mat_a_t a = __builtin_astype(ua.v, mat_a_t);
        mat_b_t b = qcom_coop_mat_load(bp, CLK_COOPERATIVE_MATRIX_LAYOUT_COLUMN_MAJOR_QCOM, 16);
        acc = qcom_coop_mat_mulAdd(a, b, acc, CLK_COOPERATIVE_MATRIX_OPERAND_NONE_QCOM);
        bp += 1024;
    }

    c_row_t uc;
    uc.v = __builtin_astype(acc, half4096);

    global float * p = (global float *)(dst + offsetd) + ((ulong)h * N + (ulong)tn * 64) * M + row + lane;
    const int nv = N - tn * 64;
    store_c16(p, uc.s.h0,  0, nv, M);
    store_c16(p, uc.s.h1, 16, nv, M);
    store_c16(p, uc.s.h2, 32, nv, M);
    store_c16(p, uc.s.h3, 48, nv, M);
}

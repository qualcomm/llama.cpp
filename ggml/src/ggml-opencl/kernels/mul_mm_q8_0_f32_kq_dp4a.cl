#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#ifdef cl_khr_integer_dot_product
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#endif

// fa=0 prefill KQ (scores = K . Q^T) straight off a q8_0 K cache, int8 dp4a inner
// product. The stock path for a quantized K at prefill dequantizes the whole K view to
// a scratch f16 buffer and then runs a f32 GEMM on it -- and because that scratch is
// contiguous it does not even qualify for the tuned KQ kernel, so it lands on the
// generic mul_mm. That is why a q8_0 K cache currently COSTS ~11% of pp4096 instead of
// saving anything. Here K is consumed in place as q8_0 and Q is quantized to int8 once
// per op, so the dequant pass disappears and the inner product runs on dp4a.
//
// Both operands are symmetric (q8_0 carries a scale and no min, and the Q quantizer
// below is symmetric too), so a block product needs no sum-correction term: it is just
// dot(k_i8, q_i8) * d_k * d_q.

// Rows of K each lane owns. dp4a folds 4 MACs into one instruction, which moves the
// bottleneck off the ALU and onto local-memory read issue: at one row per lane the
// inner loop issues one LDS uint read per dp4a (1:1), and a wrong-math probe that cut
// LDS reads 32x while keeping every dp4a ran 4.7x faster (0.66 vs 3.07 ms/op) -- the
// int8 math was only ~21% of the time. Holding KQ_ROWS rows per lane feeds KQ_ROWS
// dp4a ops from each Q fetch and cuts LDS traffic by that factor. (Register blocking
// did nothing for the f16 KQ kernel because that one is FMA-bound; dp4a changes which
// lever works.)
// What matters is the ratio (dp4a per LDS fetch) = KQ_ROWS, at a fixed accumulator budget
// of KQ_ROWS * KQ_COLS floats. 2x32 and 4x16 both cost 64 accumulators, but 4x16 issues
// half the LDS reads for the same dp4a volume. Above ~64 accumulators the compiler spills
// and the gain reverses (4x32 = 128 accumulators measured slower than 2x32).
#ifndef KQ_ROWS
#define KQ_ROWS 2
#endif
#ifndef KQ_COLS
#define KQ_COLS 32
#endif

// K-cache type this program is compiled for: 80 = q8_0, 40 = q4_0.
#ifndef KQ_KTYPE
#define KQ_KTYPE 80
#endif

#define TILESIZE_M (64 * KQ_ROWS)
#define TILESIZE_N KQ_COLS
#define Q8_0_BLK   32
#define Q8_0_SZ    34   // f16 scale + 32 int8
#define Q4_0_SZ    18   // f16 scale + 32 nibbles

#if KQ_KTYPE == 40
#define KQ_BLK_SZ Q4_0_SZ
#else
#define KQ_BLK_SZ Q8_0_SZ
#endif

// ---------------------------------------------------------------------------
// Pre-pass: quantize the KQ op's permuted f32 Q to int8 + one scale per 32-block.
//
// Q element (k, h, n) sits at float index n*(K*D_B) + h*K + k -- the same permuted
// B addressing mul_mm_f16_f32_kq uses. The output is tight and 4-byte aligned so the
// GEMM can read it as uints:
//   qq[(h*N + n)*K + k]            int8
//   qd[(h*N + n)*(K/32) + b]       half
//
// Quantizing once per op (rather than inside the GEMM) also shrinks every re-read of Q
// by 4x: the GEMM streams the Q tile once per m-block. One work-item per (block, n, h).
// qs is the plain sum of the block's int8 codes. A q4_0 K stores its nibbles biased by
// +8, and the cheapest way to undo that is not to subtract 8 from every unpacked byte
// (which would need a borrow-safe SWAR subtract) but to fold it into the block result:
//   sum_i (n_i - 8) * q_i  ==  dot(n, q) - 8 * sum_i q_i
// so the GEMM needs sum(q) per (column, block). q8_0 K is symmetric and ignores it.
__kernel void kernel_kq_quant_q_i8(
        __global const float * q,
        int              offset1_words,
        __global char  * qq,
        __global half  * qd,
        __global float * qs,
        int              K,
        int              N,
        int              D_B
) {
    const int b = get_global_id(0);   // 32-block along K
    const int n = get_global_id(1);   // query token
    const int h = get_global_id(2);   // head

    const int n_blk = K / Q8_0_BLK;
    if (b >= n_blk || n >= N || h >= D_B) {
        return;
    }

    __global const float * src =
        q + offset1_words + (ulong)n * (K * D_B) + (ulong)h * K + b * Q8_0_BLK;

    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < Q8_0_BLK; ++i) {
        amax = fmax(amax, fabs(src[i]));
    }

    const float d  = amax / 127.0f;
    const float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;

    __global char * dq = qq + ((ulong)(h * N + n) * K + b * Q8_0_BLK);
    int sum = 0;
    #pragma unroll
    for (int i = 0; i < Q8_0_BLK; ++i) {
        const int c = (int)rint(src[i] * id);
        dq[i] = (char)c;
        sum  += c;
    }

    qd[(ulong)(h * N + n) * n_blk + b] = (half)d;
    qs[(ulong)(h * N + n) * n_blk + b] = (float)sum;
}

// ---------------------------------------------------------------------------
// 32 int8 x 32 int8 -> int32, as 8 packed 4x8 dot-accumulates. Both operands are
// private: staging Q through registers before the dp4a chain is what makes each LDS
// fetch feed KQ_ROWS of them, and it also keeps the builtin away from a __local
// operand (which miscompiles on X2 in an unrolled hot loop).
inline int kq_dot8(uint8 kv, uint8 qv) {
    int r = 0;
    r = dot_acc_sat_4x8packed_ss_int(kv.s0, qv.s0, r);
    r = dot_acc_sat_4x8packed_ss_int(kv.s1, qv.s1, r);
    r = dot_acc_sat_4x8packed_ss_int(kv.s2, qv.s2, r);
    r = dot_acc_sat_4x8packed_ss_int(kv.s3, qv.s3, r);
    r = dot_acc_sat_4x8packed_ss_int(kv.s4, qv.s4, r);
    r = dot_acc_sat_4x8packed_ss_int(kv.s5, qv.s5, r);
    r = dot_acc_sat_4x8packed_ss_int(kv.s6, qv.s6, r);
    r = dot_acc_sat_4x8packed_ss_int(kv.s7, qv.s7, r);
    return r;
}

// Load one K block's 32 codes, packed 4-per-uint in natural element order so that the
// dp4a chain lines up with the Q the pre-pass wrote. Both q8_0 (34 B) and q4_0 (18 B)
// blocks put their quants 2 bytes into an even-length block, so the quants are
// uint-aligned for odd b and 2-byte-shifted for even b; the shifted case stitches each
// uint from the top half of one word and the bottom of the next. b is workgroup-uniform,
// so this branch never diverges within a wave.
inline uint8 kq_load_k(__global const uint * k_row_u, int b) {
    const int byte_qs = KQ_BLK_SZ * b + 2;
    const int i0      = byte_qs >> 2;
    const bool shift  = (byte_qs & 3) != 0;

#if KQ_KTYPE == 40
    // q4_0: 16 bytes of nibbles. Element i<16 is the low nibble of byte i, element i+16
    // the high nibble -- so the 4 source uints expand to uints 0..3 (low) and 4..7 (high),
    // which is exactly Q's 4-per-uint order. Bytes come out 0..15, which fits signed int8,
    // so the plain signed dot works; the +8 bias is removed by the block-sum term.
    uint4 u;
    if (shift) {
        u.s0 = (k_row_u[i0 + 0] >> 16) | (k_row_u[i0 + 1] << 16);
        u.s1 = (k_row_u[i0 + 1] >> 16) | (k_row_u[i0 + 2] << 16);
        u.s2 = (k_row_u[i0 + 2] >> 16) | (k_row_u[i0 + 3] << 16);
        u.s3 = (k_row_u[i0 + 3] >> 16) | (k_row_u[i0 + 4] << 16);
    } else {
        u.s0 = k_row_u[i0 + 0];
        u.s1 = k_row_u[i0 + 1];
        u.s2 = k_row_u[i0 + 2];
        u.s3 = k_row_u[i0 + 3];
    }

    uint8 kv;
    kv.s0 =  u.s0       & 0x0F0F0F0Fu;
    kv.s1 =  u.s1       & 0x0F0F0F0Fu;
    kv.s2 =  u.s2       & 0x0F0F0F0Fu;
    kv.s3 =  u.s3       & 0x0F0F0F0Fu;
    kv.s4 = (u.s0 >> 4) & 0x0F0F0F0Fu;
    kv.s5 = (u.s1 >> 4) & 0x0F0F0F0Fu;
    kv.s6 = (u.s2 >> 4) & 0x0F0F0F0Fu;
    kv.s7 = (u.s3 >> 4) & 0x0F0F0F0Fu;
    return kv;
#else
    uint8 kv;
    if (shift) {
        kv.s0 = (k_row_u[i0 + 0] >> 16) | (k_row_u[i0 + 1] << 16);
        kv.s1 = (k_row_u[i0 + 1] >> 16) | (k_row_u[i0 + 2] << 16);
        kv.s2 = (k_row_u[i0 + 2] >> 16) | (k_row_u[i0 + 3] << 16);
        kv.s3 = (k_row_u[i0 + 3] >> 16) | (k_row_u[i0 + 4] << 16);
        kv.s4 = (k_row_u[i0 + 4] >> 16) | (k_row_u[i0 + 5] << 16);
        kv.s5 = (k_row_u[i0 + 5] >> 16) | (k_row_u[i0 + 6] << 16);
        kv.s6 = (k_row_u[i0 + 6] >> 16) | (k_row_u[i0 + 7] << 16);
        kv.s7 = (k_row_u[i0 + 7] >> 16) | (k_row_u[i0 + 8] << 16);
    } else {
        kv.s0 = k_row_u[i0 + 0];
        kv.s1 = k_row_u[i0 + 1];
        kv.s2 = k_row_u[i0 + 2];
        kv.s3 = k_row_u[i0 + 3];
        kv.s4 = k_row_u[i0 + 4];
        kv.s5 = k_row_u[i0 + 5];
        kv.s6 = k_row_u[i0 + 6];
        kv.s7 = k_row_u[i0 + 7];
    }
    return kv;
#endif
}

// One query column against all KQ_ROWS of this lane's K rows: 8 LDS reads feed
// KQ_ROWS dp4a chains. ACC/COMP name the accumulator vector and the component that
// this column lands in.
//
// q4_0 unpacks to 0..15, i.e. the true code biased by +8, so the block result is
// (dot - 8*sum(q)) * d_k * d_q. sb is the column's sum(q); q8_0 is symmetric and skips it.
#if KQ_KTYPE == 40
#define KQ_BIAS(CIDX) (- 8.0f * sb[CIDX])
#else
#define KQ_BIAS(CIDX) 0.0f
#endif

#define KQ_COL(CIDX, ACC, COMP)                                                       \
    {                                                                                 \
        uint8 qv;                                                                     \
        qv.s0 = qb[(CIDX) * 8 + 0];                                                   \
        qv.s1 = qb[(CIDX) * 8 + 1];                                                   \
        qv.s2 = qb[(CIDX) * 8 + 2];                                                   \
        qv.s3 = qb[(CIDX) * 8 + 3];                                                   \
        qv.s4 = qb[(CIDX) * 8 + 4];                                                   \
        qv.s5 = qb[(CIDX) * 8 + 5];                                                   \
        qv.s6 = qb[(CIDX) * 8 + 6];                                                   \
        qv.s7 = qb[(CIDX) * 8 + 7];                                                   \
        const float dq   = db[CIDX];                                                  \
        const float bias = KQ_BIAS(CIDX);                                             \
        _Pragma("unroll")                                                             \
        for (int r = 0; r < KQ_ROWS; ++r) {                                           \
            ACC[r].COMP += ((float)kq_dot8(kv[r], qv) + bias) * dk[r] * dq;           \
        }                                                                             \
    }

// The n-tiles run on the fast-varying axis and the m-blocks on the slow one, matching
// mul_mm_f16_f32_kq: with the axes the other way round a whole plane of A (the K cache)
// is re-streamed for every n-tile and the op cost jumps ~3.5x once A stops fitting.
__kernel void mul_mm_q8_0_f32_kq_dp4a(
        __global const uchar * matrix_A,   // q8_0 K cache, AoS, in place
        int              offset0,          // bytes
        __global const uint  * qq,         // int8 Q from the pre-pass
        __global const half  * qd,         // per-32-block Q scales
        __global const float * qsum,       // per-32-block sum(q) -- q4_0 zero-point only
        __global float * dst,
        int              offsetd,          // bytes
        int M, int K, int N,
        int D_A, int D_B,
        int nb01, int nb02                 // q8_0 K strides (bytes): row, head
) {
    dst = (__global float *)((__global char *)dst + offsetd);

    const uint m_blocks = (M + TILESIZE_M - 1) / TILESIZE_M;

    const uint block_id_n = get_global_id(1);
    const uint block_id_m = get_global_id(2) % m_blocks;
    const uint block_id_d = get_global_id(2) / m_blocks;

    const uint lane = get_local_id(0);   // 0..63
    const uint sg   = get_local_id(1);   // n-tile within the workgroup

    const uint col = block_id_m * TILESIZE_M;   // first K row of this m-block
    const uint row = block_id_n * TILESIZE_N;   // first query column of this n-tile

    const uint depth_A = block_id_d / (D_B / D_A);   // KV head
    const uint depth_B = block_id_d;                 // query head

    const int n_blk = K / Q8_0_BLK;

    // This lane's KQ_ROWS K rows, strided by 64 so that neighbouring lanes stay on
    // neighbouring rows. nb01/nb02 are multiples of 4, so each row base is uint-aligned
    // even though the 34-byte blocks inside it are not.
    uint m_row[KQ_ROWS];
    __global const uint * k_row_u[KQ_ROWS];
    __global const half * k_row_h[KQ_ROWS];

    #pragma unroll
    for (int r = 0; r < KQ_ROWS; ++r) {
        m_row[r]   = col + r * 64 + lane;
        k_row_u[r] = (__global const uint *)(
            matrix_A + offset0 +
            (ulong)min(m_row[r], (uint)M - 1) * nb01 + (ulong)depth_A * nb02);
        k_row_h[r] = (__global const half *)k_row_u[r];
    }

    __local uint  q_lds[2 * TILESIZE_N * 8];   // 2 n-tiles x KQ_COLS columns x 32 int8
    __local float q_dl [2 * TILESIZE_N];
#if KQ_KTYPE == 40
    __local float q_sl [2 * TILESIZE_N];
#endif

    float16 regC0[KQ_ROWS];
#if KQ_COLS == 32
    float16 regC1[KQ_ROWS];
#endif

    #pragma unroll
    for (int r = 0; r < KQ_ROWS; ++r) {
        regC0[r] = (float16)(0.0f);
#if KQ_COLS == 32
        regC1[r] = (float16)(0.0f);
#endif
    }

    for (int b = 0; b < n_blk; ++b) {
        // Stage this k-block's Q columns (int8 + scale) into local memory. Each lane owns
        // one fixed uint slot j within a column and walks the columns 8 at a time, so the
        // 64 lanes write contiguous runs.
        barrier(CLK_LOCAL_MEM_FENCE);
        {
            const uint j     = lane & 7;    // which of the 8 uints of a column
            const uint c0    = lane >> 3;   // 0..7: first column this lane serves

            #pragma unroll
            for (int i = 0; i < KQ_COLS / 8; ++i) {
                const uint n_local = c0 + 8 * i;
                const uint gcol    = row + n_local;

                __local uint * dl = q_lds + sg * (TILESIZE_N * 8) + n_local * 8 + j;

                *dl = (gcol < (uint)N)
                    ? qq[(((ulong)(depth_B * N + gcol) * K + b * Q8_0_BLK) >> 2) + j]
                    : 0u;
            }

            if (lane < TILESIZE_N) {
                const uint gc  = row + lane;
                const bool ok  = gc < (uint)N;
                const ulong bi = (ulong)(depth_B * N + gc) * n_blk + b;
                q_dl[sg * TILESIZE_N + lane] = ok ? convert_float(qd[bi]) : 0.0f;
#if KQ_KTYPE == 40
                q_sl[sg * TILESIZE_N + lane] = ok ? qsum[bi] : 0.0f;
#endif
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        uint8 kv[KQ_ROWS];
        float dk[KQ_ROWS];

        #pragma unroll
        for (int r = 0; r < KQ_ROWS; ++r) {
            kv[r] = kq_load_k(k_row_u[r], b);
            // The scale sits at byte KQ_BLK_SZ*b, always even, so index it as a half.
            dk[r] = convert_float(k_row_h[r][(KQ_BLK_SZ / 2) * b]);
        }

        __local const uint  * qb = q_lds + sg * (TILESIZE_N * 8);
        __local const float * db = q_dl  + sg * TILESIZE_N;
#if KQ_KTYPE == 40
        __local const float * sb = q_sl  + sg * TILESIZE_N;
#endif

        KQ_COL( 0, regC0, s0) KQ_COL( 1, regC0, s1) KQ_COL( 2, regC0, s2) KQ_COL( 3, regC0, s3)
        KQ_COL( 4, regC0, s4) KQ_COL( 5, regC0, s5) KQ_COL( 6, regC0, s6) KQ_COL( 7, regC0, s7)
        KQ_COL( 8, regC0, s8) KQ_COL( 9, regC0, s9) KQ_COL(10, regC0, sa) KQ_COL(11, regC0, sb)
        KQ_COL(12, regC0, sc) KQ_COL(13, regC0, sd) KQ_COL(14, regC0, se) KQ_COL(15, regC0, sf)
#if KQ_COLS == 32
        KQ_COL(16, regC1, s0) KQ_COL(17, regC1, s1) KQ_COL(18, regC1, s2) KQ_COL(19, regC1, s3)
        KQ_COL(20, regC1, s4) KQ_COL(21, regC1, s5) KQ_COL(22, regC1, s6) KQ_COL(23, regC1, s7)
        KQ_COL(24, regC1, s8) KQ_COL(25, regC1, s9) KQ_COL(26, regC1, sa) KQ_COL(27, regC1, sb)
        KQ_COL(28, regC1, sc) KQ_COL(29, regC1, sd) KQ_COL(30, regC1, se) KQ_COL(31, regC1, sf)
#endif
    }

    // dst(m, n, head) = depth_B*N*M + n*M + m, matching mul_mm_f16_f32_kq's C layout.
    const int tail = N - (int)row;   // query columns left in this tile

    #pragma unroll
    for (int r = 0; r < KQ_ROWS; ++r) {
        if (m_row[r] >= (uint)M) {
            continue;
        }
        __global float * c = dst + (ulong)depth_B * N * M + (ulong)row * M + m_row[r];

        #define KQ_ST(i, v) if ((i) < tail) { c[(ulong)(i) * M] = (v); }
        KQ_ST( 0, regC0[r].s0) KQ_ST( 1, regC0[r].s1) KQ_ST( 2, regC0[r].s2) KQ_ST( 3, regC0[r].s3)
        KQ_ST( 4, regC0[r].s4) KQ_ST( 5, regC0[r].s5) KQ_ST( 6, regC0[r].s6) KQ_ST( 7, regC0[r].s7)
        KQ_ST( 8, regC0[r].s8) KQ_ST( 9, regC0[r].s9) KQ_ST(10, regC0[r].sa) KQ_ST(11, regC0[r].sb)
        KQ_ST(12, regC0[r].sc) KQ_ST(13, regC0[r].sd) KQ_ST(14, regC0[r].se) KQ_ST(15, regC0[r].sf)
#if KQ_COLS == 32
        KQ_ST(16, regC1[r].s0) KQ_ST(17, regC1[r].s1) KQ_ST(18, regC1[r].s2) KQ_ST(19, regC1[r].s3)
        KQ_ST(20, regC1[r].s4) KQ_ST(21, regC1[r].s5) KQ_ST(22, regC1[r].s6) KQ_ST(23, regC1[r].s7)
        KQ_ST(24, regC1[r].s8) KQ_ST(25, regC1[r].s9) KQ_ST(26, regC1[r].sa) KQ_ST(27, regC1[r].sb)
        KQ_ST(28, regC1[r].sc) KQ_ST(29, regC1[r].sd) KQ_ST(30, regC1[r].se) KQ_ST(31, regC1[r].sf)
#endif
        #undef KQ_ST
    }
}

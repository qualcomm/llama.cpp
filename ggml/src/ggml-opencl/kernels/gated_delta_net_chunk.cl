// Chunkwise gated delta net for prefill (WY representation).
//
// The recurrent kernel in gated_delta_net.cl walks the tokens one at a time and
// carries the [S_V, S_V] state through a chain of two cross-lane reductions per
// token, so a 1024-token prefill is 1024 dependent steps on ~512 workgroups. The
// chunk form processes the tokens 64 at a time with dense block products, in the
// formulation of the delta-net literature (Yang et al. 2024) that llama.cpp's
// graph-level build_delta_net_chunking also uses:
//
//   per chunk of C = 64 tokens, gate g_t (log space, <= 0), G_i = sum_{j<=i} g_j:
//     D[i][j] = exp(G_i - G_j)                        j <= i
//     A       = strict_lower((K*beta) K^T . D)        T = (I + A)^-1
//     W       = T (K * beta * exp(G))                 U = T (V * beta)
//     Qg      = scale * Q * exp(G)                    Kg = K * exp(G_C - G)
//   then, carrying the state S across chunks:
//     V'      = W S                V_new = U - V'
//     O       = Qg S + lower(scale * Q K^T . D) V_new
//     S       = exp(G_C) S + Kg^T V_new
//
// kernel_gdn_chunk_prep builds W, U, Qg, Kg and P = lower(scale * Q K^T . D) for
// every (chunk, head, seq), one 64-lane workgroup each. Everything stage two
// streams per r-step is written r-blocked ([r/4][token][4]) so a wave reads it as
// one contiguous float4 per lane.
//
// kernel_gdn_chunk_scan owns NCOL columns of the state for one (head, seq) and
// walks the chunks in order with the state block resident in local memory, laid
// out [col][r] as ggml stores it.
//
// Measured on Adreno 840 while shaping these kernels: a wave-uniform read of local
// memory (every lane the same address) is affordable when one float4 feeds ~8 FMAs,
// a uniform read of global memory is not, lane groups reading distinct addresses
// are worse than a pure broadcast, and register footprint decides occupancy, so
// small tiles beat wide ones (scan: 8 columns per lane, one token row per lane).
//
// S_V = 128 only (Qwen3.5/3.6 geometry); the host keeps other sizes, the KDA
// vector gate and the token tail on the recurrent kernel.

#pragma OPENCL EXTENSION cl_khr_subgroups : enable

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define REQD_SUBGROUP_SIZE_64 __attribute__((qcom_reqd_sub_group_size("half")))
#else
#define REQD_SUBGROUP_SIZE_64
#endif

#ifndef USE_QCOM_SUBGROUP_SHUFFLE
#define USE_QCOM_SUBGROUP_SHUFFLE 0
#endif
#if USE_QCOM_SUBGROUP_SHUFFLE
#pragma OPENCL EXTENSION cl_qcom_subgroup_shuffle : enable
#endif

#ifndef S_V
#define S_V 128
#endif
#ifndef NCOL
#define NCOL 8
#endif
#ifndef MSPLIT
#define MSPLIT 1            // scan: lane groups the reduction axis is split over == rows per lane
#endif

#define CH      64          // tokens per chunk == lanes per workgroup
#define RB      16          // r-block of the A/P phase
#define JB      16          // token block of the A/P phase
#define TSTRIDE 68          // row stride of the A/T triangle in local memory: keeps the T rows float4-aligned

#if S_V != 128
#error "gated_delta_net_chunk.cl is written for S_V == 128"
#endif

// Scratch layout per (chunk, head, seq) = one "chunk-head" index ch, in floats:
//   W  : [S_V/4][CH][4]   r-blocked
//   Qg : [S_V/4][CH][4]   r-blocked
//   Kg : [CH/4][S_V][4]   t-blocked (lane = r in the scan)
//   U  : [CH][S_V]        row-major (lane = token reads its own row segment)
//   P  : [CH/4][CH][4]    t-blocked, zero above the diagonal
//   gamma : exp(G_C)      one float
//   Kb, Qb : [S_V/4][CH][4]  r-blocked copies of k and q, read only by the prep kernel
#define SCR_W_SIZE   (S_V * CH)
#define SCR_QG_SIZE  (S_V * CH)
#define SCR_KG_SIZE  (S_V * CH)
#define SCR_U_SIZE   (S_V * CH)
#define SCR_P_SIZE   (CH * CH)

// Sum v over the lanes that differ in the bits >= log2(CH/m) (the reduction-axis split).
static inline float gdn_split_sum(float v, uint m, __local float * xchg, uint lane) {
#if USE_QCOM_SUBGROUP_SHUFFLE
    for (uint s = CH / m; s < CH; s <<= 1) {
        v += qcom_sub_group_shuffle_xor(v, s, CLK_SUB_GROUP_SHUFFLE_WIDTH_WAVE_SIZE_QCOM, v);
    }
    return v;
#else
    for (uint s = CH / m; s < CH; s <<= 1) {
        xchg[lane] = v;
        sub_group_barrier(CLK_LOCAL_MEM_FENCE);
        v += xchg[lane ^ s];
        sub_group_barrier(CLK_LOCAL_MEM_FENCE);
    }
    return v;
#endif
}

__attribute__((reqd_work_group_size(CH, 1, 1)))
REQD_SUBGROUP_SIZE_64
kernel void kernel_gdn_chunk_prep(
        global const char * q_buf,    ulong off_q,
        global const char * k_buf,    ulong off_k,
        global const char * v_buf,    ulong off_v,
        global const char * g_buf,    ulong off_g,
        global const char * beta_buf, ulong off_beta,
        global       char * scr_buf,  ulong off_w, ulong off_qg, ulong off_kg, ulong off_u, ulong off_p, ulong off_gamma,
        ulong off_kb, ulong off_qb,
        uint  H_v,
        uint  H_k,
        uint  n_chunks,
        uint  rq3,
        uint  sq1, uint sq2, uint sq3,
        uint  sk1, uint sk2, uint sk3,
        uint  sv1, uint sv2, uint sv3,
        uint  sb1, uint sb2, uint sb3,
        float scale) {

    global const float * data_q    = (global const float *)(q_buf    + off_q);
    global const float * data_k    = (global const float *)(k_buf    + off_k);
    global const float * data_v    = (global const float *)(v_buf    + off_v);
    global const float * data_g    = (global const float *)(g_buf    + off_g);
    global const float * data_beta = (global const float *)(beta_buf + off_beta);

    const uint chunk = get_group_id(0);
    const uint head  = get_group_id(1);
    const uint seq   = get_group_id(2);
    const uint lane  = get_local_id(0);
    const uint tok   = chunk * CH + lane;

    const uint ch = (seq * H_v + head) * n_chunks + chunk;

    global float * scr_w     = (global float *)(scr_buf + off_w)     + (ulong) ch * SCR_W_SIZE;
    global float * scr_qg    = (global float *)(scr_buf + off_qg)    + (ulong) ch * SCR_QG_SIZE;
    global float * scr_kg    = (global float *)(scr_buf + off_kg)    + (ulong) ch * SCR_KG_SIZE;
    global float * scr_u     = (global float *)(scr_buf + off_u)     + (ulong) ch * SCR_U_SIZE;
    global float * scr_p     = (global float *)(scr_buf + off_p)     + (ulong) ch * SCR_P_SIZE;
    global float * scr_gamma = (global float *)(scr_buf + off_gamma) + ch;
    global float * scr_kb    = (global float *)(scr_buf + off_kb)    + (ulong) ch * SCR_W_SIZE;
    global float * scr_qb    = (global float *)(scr_buf + off_qb)    + (ulong) ch * SCR_W_SIZE;

    const uint iq1 = head % H_k;
    const uint iq3 = seq / rq3;

    // first row of the chunk in q/k/v; row j is + j * stride
    global const float * q_chunk = data_q + iq3 * sq3 + iq1 * sq1 + (ulong) (chunk * CH) * sq2;
    global const float * k_chunk = data_k + iq3 * sk3 + iq1 * sk1 + (ulong) (chunk * CH) * sk2;
    global const float * v_chunk = data_v + seq * sv3 + head * sv1 + (ulong) (chunk * CH) * sv2;

    const uint gb_off = seq * sb3 + head * sb1 + tok * sb2;

    __local float AT[CH * TSTRIDE];     // A row-major strictly below the diagonal; T^T on and above it
    __local float tile[JB * RB];        // k tile of the A/P product
    __local float Gs[CH];               // cumulative gate G_j
    __local float eG[CH];               // exp(G_j)
    __local float eGd[CH];              // exp(G_C - G_j)
    __local float Bs[CH];               // beta_j

    // --- gate prefix, per-token scalars ----------------------------------------------------
    const float g_i    = data_g[gb_off];
    const float beta_i = data_beta[gb_off];
    const float G_i    = sub_group_scan_inclusive_add(g_i);
    const float G_C    = sub_group_broadcast(G_i, CH - 1);

    Gs[lane]  = G_i;
    eG[lane]  = exp(G_i);
    eGd[lane] = exp(G_C - G_i);
    Bs[lane]  = beta_i;
    if (lane == 0) {
        *scr_gamma = exp(G_C);
    }
    sub_group_barrier(CLK_LOCAL_MEM_FENCE);

    // --- r-blocked copies Kb, Qb (own-row operands of the A/P product), Qg, and Kg -----------
    // lane = token: one strided row read, coalesced blocked writes
    {
        global const float * q_row = q_chunk + (ulong) lane * sq2;
        global const float * k_row = k_chunk + (ulong) lane * sk2;
        const float sg = scale * eG[lane];
        #pragma unroll 8
        for (uint r4 = 0; r4 < S_V / 4; r4++) {
            const float4 kv = vload4(0, k_row + r4 * 4);
            const float4 qv = vload4(0, q_row + r4 * 4);
            vstore4(kv,      0, scr_kb + (r4 * CH + lane) * 4);
            vstore4(qv,      0, scr_qb + (r4 * CH + lane) * 4);
            vstore4(sg * qv, 0, scr_qg + (r4 * CH + lane) * 4);
        }
    }
    // Kg = e^{G_C - G_t} k_t, t-blocked with lane = r (two rows per lane)
    #pragma unroll 2
    for (uint rh = 0; rh < S_V / CH; rh++) {
        const uint r = rh * CH + lane;
        #pragma unroll 4
        for (uint t4 = 0; t4 < CH / 4; t4++) {
            float kg[4];
            #pragma unroll
            for (uint e = 0; e < 4; e++) {
                const uint t = t4 * 4 + e;
                kg[e] = eGd[t] * k_chunk[(ulong) t * sk2 + r];
            }
            vstore4((float4)(kg[0], kg[1], kg[2], kg[3]), 0, scr_kg + (t4 * S_V + r) * 4);
        }
    }
    sub_group_barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

    // --- A = strict_lower(beta_i D (K K^T)), P = lower(scale D (Q K^T)), lane = row i --------
    // JB tokens j at a time over RB-wide r blocks: the lane's own k and q blocks sit in
    // registers, the [JB][RB] k tile is read wave-uniformly and each float4 feeds both dots.
#ifndef GDN_SKIP_AP
    for (uint jb = 0; jb < CH / JB; jb++) {
        float acc_a[JB], acc_p[JB];
        #pragma unroll
        for (uint j = 0; j < JB; j++) {
            acc_a[j] = 0.0f;
            acc_p[j] = 0.0f;
        }

        for (uint rb = 0; rb < S_V / RB; rb++) {
            float4 ok[RB / 4], oq[RB / 4];
            #pragma unroll
            for (uint r4 = 0; r4 < RB / 4; r4++) {
                ok[r4] = vload4(0, scr_kb + ((rb * (RB / 4) + r4) * CH + lane) * 4);
                oq[r4] = vload4(0, scr_qb + ((rb * (RB / 4) + r4) * CH + lane) * 4);
            }

            sub_group_barrier(CLK_LOCAL_MEM_FENCE);
            {
                // JB tokens x RB floats = 64 lanes x one float4: lane -> (token lane/4, r4 lane%4)
                const uint j  = lane >> 2;
                const uint r4 = lane & 3;
                vstore4(vload4(0, scr_kb + ((rb * (RB / 4) + r4) * CH + jb * JB + j) * 4), 0, tile + j * RB + r4 * 4);
            }
            sub_group_barrier(CLK_LOCAL_MEM_FENCE);

            #pragma unroll
            for (uint j = 0; j < JB; j++) {
                #pragma unroll
                for (uint r4 = 0; r4 < RB / 4; r4++) {
                    const float4 kj = vload4(0, tile + j * RB + r4 * 4);
                    acc_a[j] += dot(ok[r4], kj);
                    acc_p[j] += dot(oq[r4], kj);
                }
            }
        }

        // A[i][j] = beta_i exp(G_i - G_j) k_i.k_j for j < i
        #pragma unroll
        for (uint j = 0; j < JB; j++) {
            const uint jj = jb * JB + j;
            if (jj < lane) {
                AT[lane * TSTRIDE + jj] = beta_i * exp(G_i - Gs[jj]) * acc_a[j];
            }
        }
        // P[i][j] = scale exp(G_i - G_j) q_i.k_j for j <= i, else 0; t-blocked [j/4][i][4]
        #pragma unroll
        for (uint j4 = 0; j4 < JB / 4; j4++) {
            float pe[4];
            #pragma unroll
            for (uint e = 0; e < 4; e++) {
                const uint jj = jb * JB + j4 * 4 + e;
                pe[e] = (jj <= lane) ? scale * exp(G_i - Gs[jj]) * acc_p[j4 * 4 + e] : 0.0f;
            }
            vstore4((float4)(pe[0], pe[1], pe[2], pe[3]), 0, scr_p + ((jb * (JB / 4) + j4) * CH + lane) * 4);
        }
    }
#endif
    sub_group_barrier(CLK_LOCAL_MEM_FENCE);

    // --- T = (I + A)^-1 by forward substitution, lane = column l --------------------------------
    // T[i][l] = d_il - sum_{j<i} A[i][j] T[j][l]: A row i is read four at a time wave-uniformly,
    // T[j][l] is this lane's own earlier write, kept transposed at AT[l][j] (on and above the
    // diagonal, disjoint from A; the entries below it belong to A and are masked). No barrier.
#ifndef GDN_SKIP_SOLVE
    for (uint i = 1; i < CH; i++) {
        float t = (i == lane) ? 1.0f : 0.0f;
        uint j = 0;
        for (; j + 4 <= i; j += 4) {
            const float4 a4 = vload4(0, AT + i * TSTRIDE + j);
            float4 t4 = vload4(0, AT + lane * TSTRIDE + j);
            t4.s0 = (lane <= j)     ? t4.s0 : 0.0f;
            t4.s1 = (lane <= j + 1) ? t4.s1 : 0.0f;
            t4.s2 = (lane <= j + 2) ? t4.s2 : 0.0f;
            t4.s3 = (lane <= j + 3) ? t4.s3 : 0.0f;
            t -= dot(a4, t4);
        }
        for (; j < i; j++) {
            const float tjl = (lane <= j) ? AT[lane * TSTRIDE + j] : 0.0f;
            t -= AT[i * TSTRIDE + j] * tjl;
        }
        if (lane <= i) {
            AT[lane * TSTRIDE + i] = t;
        }
    }
    if (lane == 0) {
        AT[0] = 1.0f;
    }
#endif
    sub_group_barrier(CLK_LOCAL_MEM_FENCE);

    // --- W = T (K beta e^G), U = T (V beta), lane = column pair (r, r + 64) --------------------
    // 16-row blocks of i; T[i][j] for the block is four uniform float4 (AT[j][i0..i0+15]), each
    // feeding eight FMAs, and j stops at the block's last row (T is lower triangular).
#ifndef GDN_SKIP_WU
    for (uint mtx = 0; mtx < 2; mtx++) {
        global const float * src = (mtx == 0) ? k_chunk : v_chunk;
        const ulong sstride = (mtx == 0) ? sk2 : sv2;
        for (uint ib = 0; ib < CH / 16; ib++) {
            float acc0[16], acc1[16];
            #pragma unroll
            for (uint e = 0; e < 16; e++) {
                acc0[e] = 0.0f;
                acc1[e] = 0.0f;
            }
            for (uint j = 0; j < ib * 16 + 16; j++) {
                const float g  = (mtx == 0) ? Bs[j] * eG[j] : Bs[j];
                const float x0 = src[(ulong) j * sstride + lane]      * g;
                const float x1 = src[(ulong) j * sstride + CH + lane] * g;
                #pragma unroll
                for (uint e4 = 0; e4 < 4; e4++) {
                    const float4 t4 = vload4(0, AT + j * TSTRIDE + ib * 16 + e4 * 4);
                    acc0[e4 * 4 + 0] += t4.s0 * x0; acc1[e4 * 4 + 0] += t4.s0 * x1;
                    acc0[e4 * 4 + 1] += t4.s1 * x0; acc1[e4 * 4 + 1] += t4.s1 * x1;
                    acc0[e4 * 4 + 2] += t4.s2 * x0; acc1[e4 * 4 + 2] += t4.s2 * x1;
                    acc0[e4 * 4 + 3] += t4.s3 * x0; acc1[e4 * 4 + 3] += t4.s3 * x1;
                }
            }
            #pragma unroll
            for (uint e = 0; e < 16; e++) {
                const uint i = ib * 16 + e;
                if (mtx == 0) {
                    scr_w[((lane >> 2) * CH + i) * 4 + (lane & 3)]        = acc0[e];
                    scr_w[(((lane + CH) >> 2) * CH + i) * 4 + (lane & 3)] = acc1[e];
                } else {
                    scr_u[i * S_V + lane]      = acc0[e];
                    scr_u[i * S_V + CH + lane] = acc1[e];
                }
            }
        }
    }
#endif
}

// One workgroup per (column block, head, seq): NCOL state columns resident in local
// memory across the chunk loop. Lane = (token group lg = lane % (CH/MSPLIT)) x (split
// ls = lane / (CH/MSPLIT)); a lane owns tokens lg + (CH/MSPLIT) e and state rows
// lg + (CH/MSPLIT) f, and the r (t) axis of each product is split across ls, so one
// uniform S (V_new) float4 feeds MSPLIT rows of W and Qg (P, and 2*MSPLIT rows of Kg).
// Writes the attention rows of the chunked tokens and the state after the last chunk
// (to the snapshot slot, or the tail scratch when the recurrent kernel finishes the
// remaining tokens).
#define TG      (CH / MSPLIT)       // token groups == lanes per split
#define R4S     (S_V / 4 / MSPLIT)  // r4 steps per lane in GEMM1
#define T4S     (CH / 4 / MSPLIT)   // t4 steps per lane in GEMM2
#define NRF     (S_V / TG)          // state rows per lane == 2*MSPLIT

__attribute__((reqd_work_group_size(CH, 1, 1)))
REQD_SUBGROUP_SIZE_64
kernel void kernel_gdn_chunk_scan(
        global const char * scr_buf,  ulong off_w, ulong off_qg, ulong off_kg, ulong off_u, ulong off_p, ulong off_gamma,
        global const char * state_buf, ulong off_state,
        global const char * rows_buf,  ulong off_rows,
        ulong state_row_stride,
        global       char * dst_buf,   ulong off_dst,
        global       char * sout_buf,  ulong off_sout,
        uint  H_v,
        uint  n_tokens,
        uint  n_chunks) {

    const uint cb   = get_group_id(0);
    const uint head = get_group_id(1);
    const uint seq  = get_group_id(2);
    const uint lane = get_local_id(0);
    const uint lg   = lane % TG;
    const uint ls   = lane / TG;

    global const float * data_state = (global const float *)(state_buf + off_state);
    global       float * data_dst   = (global       float *)(dst_buf   + off_dst);
    global       float * data_sout  = (global       float *)(sout_buf  + off_sout);

    const uint state_size = S_V * S_V;
    const uint state_out_base = (seq * H_v + head) * state_size;
    uint state_in_base = state_out_base;
    if (state_row_stride != 0) {
        global const int * state_rows = (global const int *)(rows_buf + off_rows);
        data_state   += (ulong) state_rows[seq] * state_row_stride;
        state_in_base = head * state_size;
    }

    __local float S_l [NCOL * S_V];     // [col][r]
    __local float VN_l[NCOL * CH];      // [col][t]
    __local float xchg[CH];

    const uint col0 = cb * NCOL;

    #pragma unroll
    for (uint c = 0; c < NCOL; c++) {
        S_l[c * S_V + lane]      = data_state[state_in_base + (col0 + c) * S_V + lane];
        S_l[c * S_V + CH + lane] = data_state[state_in_base + (col0 + c) * S_V + CH + lane];
    }
    sub_group_barrier(CLK_LOCAL_MEM_FENCE);

    const uint ch0 = (seq * H_v + head) * n_chunks;
    global float * attn = data_dst + ((ulong) seq * n_tokens * H_v + head) * S_V + col0;

    for (uint chunk = 0; chunk < n_chunks; chunk++) {
        const uint ch = ch0 + chunk;
        global const float * scr_w     = (global const float *)(scr_buf + off_w)     + (ulong) ch * SCR_W_SIZE;
        global const float * scr_qg    = (global const float *)(scr_buf + off_qg)    + (ulong) ch * SCR_QG_SIZE;
        global const float * scr_kg    = (global const float *)(scr_buf + off_kg)    + (ulong) ch * SCR_KG_SIZE;
        global const float * scr_u     = (global const float *)(scr_buf + off_u)     + (ulong) ch * SCR_U_SIZE;
        global const float * scr_p     = (global const float *)(scr_buf + off_p)     + (ulong) ch * SCR_P_SIZE;
        global const float * scr_gamma = (global const float *)(scr_buf + off_gamma) + ch;

        // (V'; Oi) = (W; Qg) S over this lane's r4 range, MSPLIT token rows
        float accv[MSPLIT][NCOL];
        float acco[MSPLIT][NCOL];
        #pragma unroll
        for (uint e = 0; e < MSPLIT; e++) {
            #pragma unroll
            for (uint c = 0; c < NCOL; c++) {
                accv[e][c] = 0.0f;
                acco[e][c] = 0.0f;
            }
        }
        for (uint r4 = ls * R4S; r4 < (ls + 1) * R4S; r4++) {
            float4 w[MSPLIT], qg[MSPLIT];
            #pragma unroll
            for (uint e = 0; e < MSPLIT; e++) {
                w[e]  = vload4(0, scr_w  + (r4 * CH + lg + e * TG) * 4);
                qg[e] = vload4(0, scr_qg + (r4 * CH + lg + e * TG) * 4);
            }
            #pragma unroll
            for (uint c = 0; c < NCOL; c++) {
                const float4 s = vload4(0, S_l + c * S_V + r4 * 4);
                #pragma unroll
                for (uint e = 0; e < MSPLIT; e++) {
                    accv[e][c] += dot(w[e],  s);
                    acco[e][c] += dot(qg[e], s);
                }
            }
        }
        #pragma unroll
        for (uint e = 0; e < MSPLIT; e++) {
            #pragma unroll
            for (uint c = 0; c < NCOL; c++) {
                accv[e][c] = gdn_split_sum(accv[e][c], MSPLIT, xchg, lane);
                acco[e][c] = gdn_split_sum(acco[e][c], MSPLIT, xchg, lane);
            }
        }

        // V_new = U - V' into local memory, [col][t]; split 0 writes
        if (ls == 0) {
            #pragma unroll
            for (uint e = 0; e < MSPLIT; e++) {
                const uint i = lg + e * TG;
                #pragma unroll
                for (uint c4 = 0; c4 < NCOL / 4; c4++) {
                    const float4 u = vload4(0, scr_u + i * S_V + col0 + c4 * 4);
                    VN_l[(c4 * 4 + 0) * CH + i] = u.s0 - accv[e][c4 * 4 + 0];
                    VN_l[(c4 * 4 + 1) * CH + i] = u.s1 - accv[e][c4 * 4 + 1];
                    VN_l[(c4 * 4 + 2) * CH + i] = u.s2 - accv[e][c4 * 4 + 2];
                    VN_l[(c4 * 4 + 3) * CH + i] = u.s3 - accv[e][c4 * 4 + 3];
                }
            }
        }
        sub_group_barrier(CLK_LOCAL_MEM_FENCE);

        // O += P V_new (MSPLIT token rows); dS = Kg^T V_new (NRF state rows), t4 range split
        float accs[NRF][NCOL];
        #pragma unroll
        for (uint f = 0; f < NRF; f++) {
            #pragma unroll
            for (uint c = 0; c < NCOL; c++) {
                accs[f][c] = 0.0f;
            }
        }
        for (uint t4 = ls * T4S; t4 < (ls + 1) * T4S; t4++) {
            float4 p[MSPLIT], kg[NRF];
            #pragma unroll
            for (uint e = 0; e < MSPLIT; e++) {
                p[e] = vload4(0, scr_p + (t4 * CH + lg + e * TG) * 4);
            }
            #pragma unroll
            for (uint f = 0; f < NRF; f++) {
                kg[f] = vload4(0, scr_kg + (t4 * S_V + lg + f * TG) * 4);
            }
            #pragma unroll
            for (uint c = 0; c < NCOL; c++) {
                const float4 vn = vload4(0, VN_l + c * CH + t4 * 4);
                #pragma unroll
                for (uint e = 0; e < MSPLIT; e++) {
                    acco[e][c] += dot(p[e], vn);
                }
                #pragma unroll
                for (uint f = 0; f < NRF; f++) {
                    accs[f][c] += dot(kg[f], vn);
                }
            }
        }
        #pragma unroll
        for (uint e = 0; e < MSPLIT; e++) {
            #pragma unroll
            for (uint c = 0; c < NCOL; c++) {
                acco[e][c] = gdn_split_sum(acco[e][c], MSPLIT, xchg, lane);
            }
        }
        #pragma unroll
        for (uint f = 0; f < NRF; f++) {
            #pragma unroll
            for (uint c = 0; c < NCOL; c++) {
                accs[f][c] = gdn_split_sum(accs[f][c], MSPLIT, xchg, lane);
            }
        }

        if (ls == 0) {
            #pragma unroll
            for (uint e = 0; e < MSPLIT; e++) {
                global float * orow = attn + (ulong) (chunk * CH + lg + e * TG) * (S_V * H_v);
                #pragma unroll
                for (uint c4 = 0; c4 < NCOL / 4; c4++) {
                    vstore4((float4)(acco[e][c4 * 4 + 0], acco[e][c4 * 4 + 1], acco[e][c4 * 4 + 2], acco[e][c4 * 4 + 3]), 0, orow + c4 * 4);
                }
            }
        }

        // S = gamma S + dS; the barrier also orders this chunk's VN reads before the next writes
        const float gamma = *scr_gamma;
        if (ls == 0) {
            #pragma unroll
            for (uint f = 0; f < NRF; f++) {
                const uint r = lg + f * TG;
                #pragma unroll
                for (uint c = 0; c < NCOL; c++) {
                    S_l[c * S_V + r] = gamma * S_l[c * S_V + r] + accs[f][c];
                }
            }
        }
        sub_group_barrier(CLK_LOCAL_MEM_FENCE);
    }

    #pragma unroll
    for (uint c = 0; c < NCOL; c++) {
        data_sout[state_out_base + (col0 + c) * S_V + lane]      = S_l[c * S_V + lane];
        data_sout[state_out_base + (col0 + c) * S_V + CH + lane] = S_l[c * S_V + CH + lane];
    }
}

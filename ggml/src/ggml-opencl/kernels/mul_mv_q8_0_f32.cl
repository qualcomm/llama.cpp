#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#else
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

#ifdef cl_intel_required_subgroup_size
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#define INTEL_GPU 1
#define REQD_SUBGROUP_SIZE_16 __attribute__((intel_reqd_sub_group_size(16)))
#define REQD_SUBGROUP_SIZE_32 __attribute__((intel_reqd_sub_group_size(32)))
#elif defined(cl_qcom_reqd_sub_group_size)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#endif

#ifdef cl_khr_subgroup_shuffle
#pragma OPENCL EXTENSION cl_khr_subgroup_shuffle : enable
#define HAS_SUBGROUP_SHUFFLE 1
#elif defined(cl_qcom_subgroup_shuffle)
#pragma OPENCL EXTENSION cl_qcom_subgroup_shuffle : enable
#define HAS_SUBGROUP_SHUFFLE 1
// Adreno compilers that expose only cl_qcom_subgroup_shuffle do not declare the KHR
// name, so calling it is an implicit declaration and the program fails to build.
// Route it to the qcom builtin.
#define sub_group_shuffle_xor(val, mask) qcom_sub_group_shuffle_xor((val), (mask), CLK_SUB_GROUP_SHUFFLE_WIDTH_WAVE_SIZE_QCOM, 0.0f)
#endif

#define QK8_0 32
typedef struct {
    half d;       // delta
    char qs[QK8_0]; // quants
} block_q8_0;

#define NB_Q8_0 8

#ifdef INTEL_GPU
#define N_R0_Q8_0 4 // number of rows each subgroup works on
#define N_SG_Q8_0 2 // number of subgroups in a work group
#define N_SIMDWIDTH 16 // subgroup size
#elif defined (ADRENO_GPU)
#define N_R0_Q8_0 4
#define N_SG_Q8_0 2
#define N_SIMDWIDTH 64
#endif

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_16
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mv_q8_0_f32(
    global char * src0,
    ulong         offset0,
    global char * src1,
    ulong         offset1,
    global char * dst,
    ulong         offsetd,
    int           ne00,
    int           ne01,
    ulong         nb01,
    ulong         nb02,
    ulong         nb03,
    int           ne12,
    ulong         nb11,
    ulong         nb12,
    ulong         nb13,
    int           ne0,
    int           ne1,
    int           r2,
    int           r3
) {
    src0 = (global char*)((global char*)src0 + offset0);
    src1 = (global char*)((global char*)src1 + offset1);
    dst  = (global char*)((global char*)dst  + offsetd);

    int nb = ne00/QK8_0;

    int r0 = get_group_id(0);
    int r1 = get_group_id(1);
    int im = get_group_id(2);

    int first_row = (r0*N_SG_Q8_0 + get_sub_group_id()) * N_R0_Q8_0;

    uint i12 = im%ne12;
    uint i13 = im/ne12;

    ulong offset_src1 = r1*nb11 + i12*nb12 + i13*nb13;
    global float * y  = (global float *) (src1 + offset_src1);

    // pointers to src0 rows
    global block_q8_0 * ax[N_R0_Q8_0];
    for (int row = 0; row < N_R0_Q8_0; ++row) {
        ulong offset_src0 = (first_row + row)*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;
        ax[row] = (global block_q8_0 *) ((global char *) src0 + offset_src0);
    }

    float yl[NB_Q8_0];
    float sumf[N_R0_Q8_0] = { 0.f };

    const short ix = get_sub_group_local_id()/4;
    const short il = get_sub_group_local_id()%4;

    global float * yb = y + ix*QK8_0 + il*NB_Q8_0;

    // each thread handles NB_Q8_0 quants at a time
    for (int ib = ix; ib < nb; ib += N_SIMDWIDTH/4) {
        for (short i = 0; i < NB_Q8_0; ++i) {
            yl[i] = yb[i];
        }

        for (short row = 0; row < N_R0_Q8_0; row++) {
            global char * qs = ax[row][ib].qs + il*NB_Q8_0;
            float sumq = 0.f;
            for (short iq = 0; iq < NB_Q8_0; ++iq) {
                sumq += qs[iq] * yl[iq];
            }
            sumf[row] += sumq*ax[row][ib].d;
        }

        yb += N_SIMDWIDTH*NB_Q8_0;
    }

    global float * dst_f32 = (global float *) dst + (ulong)im*ne0*ne1 + (ulong)r1*ne0;

    for (int row = 0; row < N_R0_Q8_0; ++row) {
        float tot = sub_group_reduce_add(sumf[row]);

        if (get_sub_group_local_id() == 0 && first_row + row < ne01) {
            dst_f32[first_row + row] = tot;
        }
    }
}

// GQA-coalesced decode KQ for a q8_0 K-cache (DK=128, r2=8, r3=1, ne11==1).
// The fa=0 quant-K analog of the f16 _x8_gqa4 coalesce: read each K-row once per
// K-head and fan it to GQA_RATIO=8 Q-heads, instead of the plain q8_0 GEMV which
// re-reads K once per Q-head. K stays q8_0 (-23% KV DDR vs f16; reads ~half the
// f16-coalesce K bytes). 64-lane subgroup = 8 Q-heads x 8 lanes; each lane takes
// a contiguous 16-element chunk of the 128-wide row = 4 float4 of Q against half
// of one q8_0 block (block = lane_q>>1, byte-half = (lane_q&1)*16). Dequant is
// d * qs (f32 mad). Q (f32) for the 8 Q-heads pre-staged in __local.
// Opt-in via GGML_OPENCL_MM_KQ_GQA_Q8_0 on the host.
#define N_K_ROWS_Q8GQA   16
#define GQA_RATIO_Q8GQA  8
#define DK_VEC_Q8GQA     32   // DK/4 for DK=128

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_q8_0_f32_gqa8_dk128(
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src0 = (global char *)((global char *)src0 + offset0);
    src1 = (global char *)((global char *)src1 + offset1);
    dst  = (global float*)((global char *)dst  + offsetd);

    const int sgs_lid = get_sub_group_local_id();
    const int q_id    = sgs_lid >> 3;   // 0..7: Q-head
    const int lane_q  = sgs_lid & 7;    // 0..7: lane within Q-head partition

    const int r0_base = get_group_id(0) * N_K_ROWS_Q8GQA;
    const int im_kv   = get_group_id(2);

    const int i02 = im_kv % ne02;       // K-head index
    const int i03 = im_kv / ne02;       // n13 batch index

    const int q_head_lo = i02 * GQA_RATIO_Q8GQA;

    // Stage 8 Q-heads x 32 float4 (DK=128) into __local (f32 Q, 4 KB). Only the
    // first DK_VEC_Q8GQA=32 lanes load per Q-head.
    __local float4 q_loc[GQA_RATIO_Q8GQA * DK_VEC_Q8GQA];
    #pragma unroll
    for (int qh = 0; qh < GQA_RATIO_Q8GQA; ++qh) {
        const int qh_idx = q_head_lo + qh;
        global float4 * y4 = (global float4 *)(src1 + qh_idx * nb12 + i03 * nb13);
        if (sgs_lid < DK_VEC_Q8GQA) {
            q_loc[qh * DK_VEC_Q8GQA + sgs_lid] = y4[sgs_lid];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // This lane's q8_0 block within the row, and its byte-half offset.
    const int blk  = lane_q >> 1;           // 0..3 (which 32-elem q8_0 block)
    const int hoff = (lane_q & 1) * 16;     // 0 or 16 (byte offset in qs[32])
    const int qf4  = lane_q * 4;            // first of this lane's 4 float4

    const ulong head_off = (i02) * nb02 + (i03 / r3) * nb03;

    #pragma unroll
    for (int dr = 0; dr < N_K_ROWS_Q8GQA; ++dr) {
        const int r0 = r0_base + dr;
        // Dispatch guarantees ne01 % N_K_ROWS_Q8GQA == 0, so r0 < ne01.
        global block_q8_0 * kb = (global block_q8_0 *)(src0 + r0 * nb01 + head_off) + blk;
        const float d = convert_float(kb->d);
        global char * qs = kb->qs + hoff;   // 16 int8 for this lane

        float sumf = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float4 qv = q_loc[q_id * DK_VEC_Q8GQA + qf4 + j];
            const int b = j * 4;
            sumf += (float)qs[b + 0] * qv.s0
                  + (float)qs[b + 1] * qv.s1
                  + (float)qs[b + 2] * qv.s2
                  + (float)qs[b + 3] * qv.s3;
        }
        sumf *= d;

        // Reduce within 8-lane Q-head partition.
        sumf += sub_group_shuffle_xor(sumf, 4);
        sumf += sub_group_shuffle_xor(sumf, 2);
        sumf += sub_group_shuffle_xor(sumf, 1);

        if (lane_q == 0) {
            const int im_out = i03 * ne12 + (q_head_lo + q_id);
            dst[im_out * ne1 * ne0 + r0] = sumf;
        }
    }
}

// image1d_buffer_t (texture-cache) variant of kernel_mul_mat_q8_0_f32_gqa8_dk128.
//
// Same GQA-coalesced decode KQ, but the q8_0 K-cache is read through a
// CL_R/CL_UNSIGNED_INT32 image1d_buffer (4-byte pixels) so K reads use the
// texture-cache BW lane -- the lever behind the f16 _x8_gqa4_img win that a plain
// __global buffer kernel can't reach. The image is bound over a sub-buffer at
// offset0 host-side, so pixel indices here are relative to offset0 (no offset0
// arg, matching the f16 img kernel).
//
// q8_0 layout per row (DK=128): 4 blocks x 34 B = 136 B = exactly 34 uint32
// pixels. Block b: d (2 B) at byte 34b, qs[32] at byte 34b+2. Lane lane_q owns
// half a block (16 int8) = K/Q elements [lane_q*16, lane_q*16+16); blk=lane_q>>1,
// hoff=(lane_q&1)*16. d sits on a 2-byte boundary inside a pixel; the 16 quant
// bytes start 2-byte-shifted for even blk (qsh=16) and pixel-aligned for odd blk
// (qsh=0), so even-blk lanes read 5 pixels and shift-combine.
// Opt-in via GGML_OPENCL_MM_KQ_GQA_Q8_0_IMG=1 on the host.
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_q8_0_f32_gqa8_dk128_img(
        __read_only image1d_buffer_t src0_img,
        global char * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src1 = (global char *)((global char *)src1 + offset1);
    dst  = (global float*)((global char *)dst  + offsetd);

    const int sgs_lid = get_sub_group_local_id();
    const int q_id    = sgs_lid >> 3;   // 0..7: Q-head
    const int lane_q  = sgs_lid & 7;    // 0..7: lane within Q-head partition

    const int r0_base = get_group_id(0) * N_K_ROWS_Q8GQA;
    const int im_kv   = get_group_id(2);

    const int i02 = im_kv % ne02;       // K-head index
    const int i03 = im_kv / ne02;       // n13 batch index

    const int q_head_lo = i02 * GQA_RATIO_Q8GQA;

    __local float4 q_loc[GQA_RATIO_Q8GQA * DK_VEC_Q8GQA];
    #pragma unroll
    for (int qh = 0; qh < GQA_RATIO_Q8GQA; ++qh) {
        const int qh_idx = q_head_lo + qh;
        global float4 * y4 = (global float4 *)(src1 + qh_idx * nb12 + i03 * nb13);
        if (sgs_lid < DK_VEC_Q8GQA) {
            q_loc[qh * DK_VEC_Q8GQA + sgs_lid] = y4[sgs_lid];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int blk  = lane_q >> 1;           // 0..3
    const int hoff = (lane_q & 1) * 16;     // 0 or 16
    const int qf4  = lane_q * 4;            // first of this lane's 4 float4

    // Pixel pitches (4-byte pixels). nb01=136 -> 34 px/row; nb02/nb03 divisible by 4.
    const int pitch_px_row  = (int)(nb01 >> 2);
    const int pitch_px_head = (int)(nb02 >> 2);
    const int pitch_px_n13  = (int)(nb03 >> 2);
    const int head_px_base  = i02 * pitch_px_head + (i03 / r3) * pitch_px_n13;

    // d byte = 34*blk: pixel offset + bit shift inside that pixel.
    const int d_byte   = 34 * blk;
    const int d_pxoff  = d_byte >> 2;
    const int d_bit    = (d_byte & 3) * 8;
    // quant bytes start at 34*blk + 2 + hoff.
    const int q_byte   = 34 * blk + 2 + hoff;
    const int q_pxoff  = q_byte >> 2;
    const uint q_sh    = (uint)((q_byte & 3) * 8);   // 0 or 16

    #pragma unroll
    for (int dr = 0; dr < N_K_ROWS_Q8GQA; ++dr) {
        const int r0 = r0_base + dr;
        const int row_px = r0 * pitch_px_row + head_px_base;

        const half  d  = as_half((ushort)((read_imageui(src0_img, row_px + d_pxoff).x >> d_bit) & 0xFFFFu));
        const float df = convert_float(d);

        const int qpx = row_px + q_pxoff;
        uint w0, w1, w2, w3;
        if (q_sh == 0u) {
            w0 = read_imageui(src0_img, qpx + 0).x;
            w1 = read_imageui(src0_img, qpx + 1).x;
            w2 = read_imageui(src0_img, qpx + 2).x;
            w3 = read_imageui(src0_img, qpx + 3).x;
        } else {
            const uint p0 = read_imageui(src0_img, qpx + 0).x;
            const uint p1 = read_imageui(src0_img, qpx + 1).x;
            const uint p2 = read_imageui(src0_img, qpx + 2).x;
            const uint p3 = read_imageui(src0_img, qpx + 3).x;
            const uint p4 = read_imageui(src0_img, qpx + 4).x;
            w0 = (p0 >> q_sh) | (p1 << (32u - q_sh));
            w1 = (p1 >> q_sh) | (p2 << (32u - q_sh));
            w2 = (p2 >> q_sh) | (p3 << (32u - q_sh));
            w3 = (p3 >> q_sh) | (p4 << (32u - q_sh));
        }

        const char4 c0 = as_char4(w0);
        const char4 c1 = as_char4(w1);
        const char4 c2 = as_char4(w2);
        const char4 c3 = as_char4(w3);
        const float4 qa = q_loc[q_id * DK_VEC_Q8GQA + qf4 + 0];
        const float4 qb = q_loc[q_id * DK_VEC_Q8GQA + qf4 + 1];
        const float4 qc = q_loc[q_id * DK_VEC_Q8GQA + qf4 + 2];
        const float4 qd = q_loc[q_id * DK_VEC_Q8GQA + qf4 + 3];

        float sumf =
              (float)c0.s0*qa.s0 + (float)c0.s1*qa.s1 + (float)c0.s2*qa.s2 + (float)c0.s3*qa.s3
            + (float)c1.s0*qb.s0 + (float)c1.s1*qb.s1 + (float)c1.s2*qb.s2 + (float)c1.s3*qb.s3
            + (float)c2.s0*qc.s0 + (float)c2.s1*qc.s1 + (float)c2.s2*qc.s2 + (float)c2.s3*qc.s3
            + (float)c3.s0*qd.s0 + (float)c3.s1*qd.s1 + (float)c3.s2*qd.s2 + (float)c3.s3*qd.s3;
        sumf *= df;

        sumf += sub_group_shuffle_xor(sumf, 4);
        sumf += sub_group_shuffle_xor(sumf, 2);
        sumf += sub_group_shuffle_xor(sumf, 1);

        if (lane_q == 0) {
            const int im_out = i03 * ne12 + (q_head_lo + q_id);
            dst[im_out * ne1 * ne0 + r0] = sumf;
        }
    }
}

// ===========================================================================
// DK=256, r2=8 variants for Qwen3.6-35B-A3B (n_head=16, n_head_kv=2 => r2=8,
// head_dim=256). Same GQA-coalesced decode KQ as gqa8_dk128, but DK=256 means a
// row is 8 q8_0 blocks; the 64-lane subgroup is still 8 Q-heads x 8 lanes, so
// each lane owns a WHOLE q8_0 block (32 elems, blk=lane_q, hoff=0) instead of a
// half block. Q (f32) for the 8 Q-heads pre-staged in __local (DK_VEC=64 float4).
// ===========================================================================
#define N_K_ROWS_Q8GQA256   16
#define GQA_RATIO_Q8GQA256  8
#define DK_VEC_Q8GQA256     64   // DK/4 for DK=256

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_q8_0_f32_gqa8_dk256(
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src0 = (global char *)((global char *)src0 + offset0);
    src1 = (global char *)((global char *)src1 + offset1);
    dst  = (global float*)((global char *)dst  + offsetd);

    const int sgs_lid = get_sub_group_local_id();
    const int q_id    = sgs_lid >> 3;   // 0..7: Q-head
    const int lane_q  = sgs_lid & 7;    // 0..7: lane within Q-head -> owns block lane_q

    const int r0_base = get_group_id(0) * N_K_ROWS_Q8GQA256;
    const int im_kv   = get_group_id(2);

    const int i02 = im_kv % ne02;       // K-head index
    const int i03 = im_kv / ne02;       // n13 batch index

    const int q_head_lo = i02 * GQA_RATIO_Q8GQA256;

    // Stage 8 Q-heads x 64 float4 (DK=256) into __local (f32 Q, 8 KB). DK_VEC=64
    // == subgroup size, so each lane loads exactly one float4 per Q-head.
    __local float4 q_loc[GQA_RATIO_Q8GQA256 * DK_VEC_Q8GQA256];
    #pragma unroll
    for (int qh = 0; qh < GQA_RATIO_Q8GQA256; ++qh) {
        const int qh_idx = q_head_lo + qh;
        global float4 * y4 = (global float4 *)(src1 + qh_idx * nb12 + i03 * nb13);
        q_loc[qh * DK_VEC_Q8GQA256 + sgs_lid] = y4[sgs_lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int blk = lane_q;          // 0..7: this lane's whole q8_0 block
    const int qf4 = lane_q * 8;      // first of this lane's 8 float4 (32 elems)

    const ulong head_off = (i02) * nb02 + (i03 / r3) * nb03;

    #pragma unroll
    for (int dr = 0; dr < N_K_ROWS_Q8GQA256; ++dr) {
        const int r0 = r0_base + dr;
        global block_q8_0 * kb = (global block_q8_0 *)(src0 + r0 * nb01 + head_off) + blk;
        const float d = convert_float(kb->d);
        global char * qs = kb->qs;   // 32 int8 for this lane (full block)

        float sumf = 0.0f;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float4 qv = q_loc[q_id * DK_VEC_Q8GQA256 + qf4 + j];
            const int b = j * 4;
            sumf += (float)qs[b + 0] * qv.s0
                  + (float)qs[b + 1] * qv.s1
                  + (float)qs[b + 2] * qv.s2
                  + (float)qs[b + 3] * qv.s3;
        }
        sumf *= d;

        // Reduce within 8-lane Q-head partition.
        sumf += sub_group_shuffle_xor(sumf, 4);
        sumf += sub_group_shuffle_xor(sumf, 2);
        sumf += sub_group_shuffle_xor(sumf, 1);

        if (lane_q == 0) {
            const int im_out = i03 * ne12 + (q_head_lo + q_id);
            dst[im_out * ne1 * ne0 + r0] = sumf;
        }
    }
}

// image1d_buffer_t (texture-cache) variant of kernel_mul_mat_q8_0_f32_gqa8_dk256.
// q8_0 row (DK=256): 8 blocks x 34 B = 272 B = 68 uint32 pixels. Lane lane_q owns
// whole block lane_q: d at byte 34*lane_q, qs[32] at byte 34*lane_q+2. d_byte&3 is
// 0 for even lane_q, 2 for odd; q_byte=34*lane_q+2 is 2-byte-shifted (q_sh=16) for
// even lane_q and pixel-aligned (q_sh=0) for odd -- so even lanes read 9 pixels and
// shift-combine 8 quant words.
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_q8_0_f32_gqa8_dk256_img(
        __read_only image1d_buffer_t src0_img,
        global char * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src1 = (global char *)((global char *)src1 + offset1);
    dst  = (global float*)((global char *)dst  + offsetd);

    const int sgs_lid = get_sub_group_local_id();
    const int q_id    = sgs_lid >> 3;   // 0..7: Q-head
    const int lane_q  = sgs_lid & 7;    // 0..7: lane -> owns block lane_q

    const int r0_base = get_group_id(0) * N_K_ROWS_Q8GQA256;
    const int im_kv   = get_group_id(2);

    const int i02 = im_kv % ne02;
    const int i03 = im_kv / ne02;

    const int q_head_lo = i02 * GQA_RATIO_Q8GQA256;

    __local float4 q_loc[GQA_RATIO_Q8GQA256 * DK_VEC_Q8GQA256];
    #pragma unroll
    for (int qh = 0; qh < GQA_RATIO_Q8GQA256; ++qh) {
        const int qh_idx = q_head_lo + qh;
        global float4 * y4 = (global float4 *)(src1 + qh_idx * nb12 + i03 * nb13);
        q_loc[qh * DK_VEC_Q8GQA256 + sgs_lid] = y4[sgs_lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int qf4 = lane_q * 8;       // first of this lane's 8 float4 (32 elems)

    // Pixel pitches (4-byte pixels). nb01=272 -> 68 px/row; nb02/nb03 divisible by 4.
    const int pitch_px_row  = (int)(nb01 >> 2);
    const int pitch_px_head = (int)(nb02 >> 2);
    const int pitch_px_n13  = (int)(nb03 >> 2);
    const int head_px_base  = i02 * pitch_px_head + (i03 / r3) * pitch_px_n13;

    const int d_byte   = 34 * lane_q;
    const int d_pxoff  = d_byte >> 2;
    const int d_bit    = (d_byte & 3) * 8;
    const int q_byte   = 34 * lane_q + 2;
    const int q_pxoff  = q_byte >> 2;
    const uint q_sh    = (uint)((q_byte & 3) * 8);   // 0 (odd lane) or 16 (even lane)

    #pragma unroll
    for (int dr = 0; dr < N_K_ROWS_Q8GQA256; ++dr) {
        const int r0 = r0_base + dr;
        const int row_px = r0 * pitch_px_row + head_px_base;

        const half  d  = as_half((ushort)((read_imageui(src0_img, row_px + d_pxoff).x >> d_bit) & 0xFFFFu));
        const float df = convert_float(d);

        const int qpx = row_px + q_pxoff;
        uint w[8];
        if (q_sh == 0u) {
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                w[k] = read_imageui(src0_img, qpx + k).x;
            }
        } else {
            uint p[9];
            #pragma unroll
            for (int k = 0; k < 9; ++k) {
                p[k] = read_imageui(src0_img, qpx + k).x;
            }
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                w[k] = (p[k] >> q_sh) | (p[k + 1] << (32u - q_sh));
            }
        }

        float sumf = 0.0f;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            const char4  c  = as_char4(w[k]);
            const float4 qv = q_loc[q_id * DK_VEC_Q8GQA256 + qf4 + k];
            sumf += (float)c.s0 * qv.s0 + (float)c.s1 * qv.s1
                  + (float)c.s2 * qv.s2 + (float)c.s3 * qv.s3;
        }
        sumf *= df;

        sumf += sub_group_shuffle_xor(sumf, 4);
        sumf += sub_group_shuffle_xor(sumf, 2);
        sumf += sub_group_shuffle_xor(sumf, 1);

        if (lane_q == 0) {
            const int im_out = i03 * ne12 + (q_head_lo + q_id);
            dst[im_out * ne1 * ne0 + r0] = sumf;
        }
    }
}

// ===========================================================================
// DK=256, r2=4 variants for Qwen3.5-9B (n_head_kv=4 => GQA r=4, head_dim=256).
// Memory-footprint play: q8/q4 K cache halves/quarters K DDR at depth. 64-lane
// subgroup = 4 Q-heads x 16 lanes; DK=256 = 8 q8_0 blocks, so each lane owns a
// HALF block (16 elems, blk=lane_q>>1, hoff=(lane_q&1)*16) -- same per-lane body
// as gqa8_dk128, but 16 lanes/Q-head (reduce masks {8,4,2,1}) and DK_VEC=64.
// ===========================================================================
#define N_K_ROWS_Q8GQA_R4_256   16
#define GQA_RATIO_Q8GQA_R4_256  4
#define DK_VEC_Q8GQA_R4_256     64   // DK/4 for DK=256

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_q8_0_f32_gqa_r4_dk256(
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src0 = (global char *)((global char *)src0 + offset0);
    src1 = (global char *)((global char *)src1 + offset1);
    dst  = (global float*)((global char *)dst  + offsetd);

    const int sgs_lid = get_sub_group_local_id();
    const int q_id    = sgs_lid >> 4;   // 0..3: Q-head
    const int lane_q  = sgs_lid & 15;   // 0..15: lane within Q-head partition

    const int r0_base = get_group_id(0) * N_K_ROWS_Q8GQA_R4_256;
    const int im_kv   = get_group_id(2);

    const int i02 = im_kv % ne02;
    const int i03 = im_kv / ne02;

    const int q_head_lo = i02 * GQA_RATIO_Q8GQA_R4_256;

    __local float4 q_loc[GQA_RATIO_Q8GQA_R4_256 * DK_VEC_Q8GQA_R4_256];
    #pragma unroll
    for (int qh = 0; qh < GQA_RATIO_Q8GQA_R4_256; ++qh) {
        const int qh_idx = q_head_lo + qh;
        global float4 * y4 = (global float4 *)(src1 + qh_idx * nb12 + i03 * nb13);
        q_loc[qh * DK_VEC_Q8GQA_R4_256 + sgs_lid] = y4[sgs_lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int blk  = lane_q >> 1;           // 0..7 (q8_0 block within the DK=256 row)
    const int hoff = (lane_q & 1) * 16;     // 0 or 16 (byte offset in qs[32])
    const int qf4  = lane_q * 4;            // first of this lane's 4 float4

    const ulong head_off = (i02) * nb02 + (i03 / r3) * nb03;

    #pragma unroll
    for (int dr = 0; dr < N_K_ROWS_Q8GQA_R4_256; ++dr) {
        const int r0 = r0_base + dr;
        global block_q8_0 * kb = (global block_q8_0 *)(src0 + r0 * nb01 + head_off) + blk;
        const float d = convert_float(kb->d);
        global char * qs = kb->qs + hoff;   // 16 int8 for this lane

        float sumf = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float4 qv = q_loc[q_id * DK_VEC_Q8GQA_R4_256 + qf4 + j];
            const int b = j * 4;
            sumf += (float)qs[b + 0] * qv.s0
                  + (float)qs[b + 1] * qv.s1
                  + (float)qs[b + 2] * qv.s2
                  + (float)qs[b + 3] * qv.s3;
        }
        sumf *= d;

        // Reduce within 16-lane Q-head partition.
        sumf += sub_group_shuffle_xor(sumf, 8);
        sumf += sub_group_shuffle_xor(sumf, 4);
        sumf += sub_group_shuffle_xor(sumf, 2);
        sumf += sub_group_shuffle_xor(sumf, 1);

        if (lane_q == 0) {
            const int im_out = i03 * ne12 + (q_head_lo + q_id);
            dst[im_out * ne1 * ne0 + r0] = sumf;
        }
    }
}

// image1d_buffer_t (texture-cache) variant of kernel_mul_mat_q8_0_f32_gqa_r4_dk256.
// Row = 8 q8_0 blocks x 34 B = 272 B = 68 px (same as the gqa8_dk256 row). Lane owns
// a half block: blk=lane_q>>1, hoff=(lane_q&1)*16; d at byte 34*blk, qs+hoff at
// 34*blk+2+hoff -- generic q_sh = (q_byte&3)*8 (read 5 px + shift-combine when shifted).
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_q8_0_f32_gqa_r4_dk256_img(
        __read_only image1d_buffer_t src0_img,
        global char * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src1 = (global char *)((global char *)src1 + offset1);
    dst  = (global float*)((global char *)dst  + offsetd);

    const int sgs_lid = get_sub_group_local_id();
    const int q_id    = sgs_lid >> 4;
    const int lane_q  = sgs_lid & 15;

    const int r0_base = get_group_id(0) * N_K_ROWS_Q8GQA_R4_256;
    const int im_kv   = get_group_id(2);

    const int i02 = im_kv % ne02;
    const int i03 = im_kv / ne02;

    const int q_head_lo = i02 * GQA_RATIO_Q8GQA_R4_256;

    __local float4 q_loc[GQA_RATIO_Q8GQA_R4_256 * DK_VEC_Q8GQA_R4_256];
    #pragma unroll
    for (int qh = 0; qh < GQA_RATIO_Q8GQA_R4_256; ++qh) {
        const int qh_idx = q_head_lo + qh;
        global float4 * y4 = (global float4 *)(src1 + qh_idx * nb12 + i03 * nb13);
        q_loc[qh * DK_VEC_Q8GQA_R4_256 + sgs_lid] = y4[sgs_lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int blk  = lane_q >> 1;
    const int hoff = (lane_q & 1) * 16;
    const int qf4  = lane_q * 4;

    const int pitch_px_row  = (int)(nb01 >> 2);
    const int pitch_px_head = (int)(nb02 >> 2);
    const int pitch_px_n13  = (int)(nb03 >> 2);
    const int head_px_base  = i02 * pitch_px_head + (i03 / r3) * pitch_px_n13;

    const int d_byte   = 34 * blk;
    const int d_pxoff  = d_byte >> 2;
    const int d_bit    = (d_byte & 3) * 8;
    const int q_byte   = 34 * blk + 2 + hoff;
    const int q_pxoff  = q_byte >> 2;
    const uint q_sh    = (uint)((q_byte & 3) * 8);

    #pragma unroll
    for (int dr = 0; dr < N_K_ROWS_Q8GQA_R4_256; ++dr) {
        const int r0 = r0_base + dr;
        const int row_px = r0 * pitch_px_row + head_px_base;

        const half  d  = as_half((ushort)((read_imageui(src0_img, row_px + d_pxoff).x >> d_bit) & 0xFFFFu));
        const float df = convert_float(d);

        const int qpx = row_px + q_pxoff;
        uint w0, w1, w2, w3;
        if (q_sh == 0u) {
            w0 = read_imageui(src0_img, qpx + 0).x;
            w1 = read_imageui(src0_img, qpx + 1).x;
            w2 = read_imageui(src0_img, qpx + 2).x;
            w3 = read_imageui(src0_img, qpx + 3).x;
        } else {
            const uint p0 = read_imageui(src0_img, qpx + 0).x;
            const uint p1 = read_imageui(src0_img, qpx + 1).x;
            const uint p2 = read_imageui(src0_img, qpx + 2).x;
            const uint p3 = read_imageui(src0_img, qpx + 3).x;
            const uint p4 = read_imageui(src0_img, qpx + 4).x;
            w0 = (p0 >> q_sh) | (p1 << (32u - q_sh));
            w1 = (p1 >> q_sh) | (p2 << (32u - q_sh));
            w2 = (p2 >> q_sh) | (p3 << (32u - q_sh));
            w3 = (p3 >> q_sh) | (p4 << (32u - q_sh));
        }

        const char4 c0 = as_char4(w0);
        const char4 c1 = as_char4(w1);
        const char4 c2 = as_char4(w2);
        const char4 c3 = as_char4(w3);
        const float4 qa = q_loc[q_id * DK_VEC_Q8GQA_R4_256 + qf4 + 0];
        const float4 qb = q_loc[q_id * DK_VEC_Q8GQA_R4_256 + qf4 + 1];
        const float4 qc = q_loc[q_id * DK_VEC_Q8GQA_R4_256 + qf4 + 2];
        const float4 qd = q_loc[q_id * DK_VEC_Q8GQA_R4_256 + qf4 + 3];

        float sumf =
              (float)c0.s0*qa.s0 + (float)c0.s1*qa.s1 + (float)c0.s2*qa.s2 + (float)c0.s3*qa.s3
            + (float)c1.s0*qb.s0 + (float)c1.s1*qb.s1 + (float)c1.s2*qb.s2 + (float)c1.s3*qb.s3
            + (float)c2.s0*qc.s0 + (float)c2.s1*qc.s1 + (float)c2.s2*qc.s2 + (float)c2.s3*qc.s3
            + (float)c3.s0*qd.s0 + (float)c3.s1*qd.s1 + (float)c3.s2*qd.s2 + (float)c3.s3*qd.s3;
        sumf *= df;

        sumf += sub_group_shuffle_xor(sumf, 8);
        sumf += sub_group_shuffle_xor(sumf, 4);
        sumf += sub_group_shuffle_xor(sumf, 2);
        sumf += sub_group_shuffle_xor(sumf, 1);

        if (lane_q == 0) {
            const int im_out = i03 * ne12 + (q_head_lo + q_id);
            dst[im_out * ne1 * ne0 + r0] = sumf;
        }
    }
}

// ===========================================================================
// r2=4 variants (DK=128) for Llama-3-8B (n_head=32, n_head_kv=8 => r2=4, all-
// global => K grows every layer; fa=0 tg64 collapses 13.0->5.5 @4k->16k, i.e.
// severely KV-BW-bound at depth). 64-lane subgroup = 4 Q-heads x 16 lanes; each
// lane owns 8 elements = a QUARTER q8_0 block (blk=lane_q>>2, eoff=(lane_q&3)*8),
// pairs with 2 float4 of staged Q (qf2=lane_q*2). K elem = lane_q*8+m. Reduce
// over 16 lanes (masks {8,4,2,1}, all <32, safe on X2).
// ===========================================================================
#define GQA_RATIO_Q8GQA_R4  4
#define DK_VEC_Q8GQA_R4     32   // DK/4 for DK=128

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_q8_0_f32_gqa_r4_dk128(
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src0 = (global char *)((global char *)src0 + offset0);
    src1 = (global char *)((global char *)src1 + offset1);
    dst  = (global float*)((global char *)dst  + offsetd);

    const int sgs_lid = get_sub_group_local_id();
    const int q_id    = sgs_lid >> 4;   // 0..3: Q-head
    const int lane_q  = sgs_lid & 15;   // 0..15: lane within Q-head partition

    const int r0_base = get_group_id(0) * N_K_ROWS_Q8GQA;
    const int im_kv   = get_group_id(2);

    const int i02 = im_kv % ne02;
    const int i03 = im_kv / ne02;

    const int q_head_lo = i02 * GQA_RATIO_Q8GQA_R4;

    __local float4 q_loc[GQA_RATIO_Q8GQA_R4 * DK_VEC_Q8GQA_R4];
    #pragma unroll
    for (int qh = 0; qh < GQA_RATIO_Q8GQA_R4; ++qh) {
        const int qh_idx = q_head_lo + qh;
        global float4 * y4 = (global float4 *)(src1 + qh_idx * nb12 + i03 * nb13);
        if (sgs_lid < DK_VEC_Q8GQA_R4) {
            q_loc[qh * DK_VEC_Q8GQA_R4 + sgs_lid] = y4[sgs_lid];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int blk  = lane_q >> 2;           // 0..3 (which q8_0 block)
    const int eoff = (lane_q & 3) * 8;      // 0,8,16,24 (elem offset in block)
    const int qf2  = lane_q * 2;            // first of this lane's 2 float4

    const ulong head_off = (ulong)i02 * nb02 + (ulong)(i03 / r3) * nb03;

    #pragma unroll
    for (int dr = 0; dr < N_K_ROWS_Q8GQA; ++dr) {
        const int r0 = r0_base + dr;
        global block_q8_0 * kb = (global block_q8_0 *)(src0 + r0 * nb01 + head_off) + blk;
        const float d = convert_float(kb->d);
        global char * qs = kb->qs + eoff;   // 8 int8 for this lane

        const float4 qa = q_loc[q_id * DK_VEC_Q8GQA_R4 + qf2 + 0];
        const float4 qb = q_loc[q_id * DK_VEC_Q8GQA_R4 + qf2 + 1];
        float sumf =
              (float)qs[0]*qa.s0 + (float)qs[1]*qa.s1 + (float)qs[2]*qa.s2 + (float)qs[3]*qa.s3
            + (float)qs[4]*qb.s0 + (float)qs[5]*qb.s1 + (float)qs[6]*qb.s2 + (float)qs[7]*qb.s3;
        sumf *= d;

        sumf += sub_group_shuffle_xor(sumf, 8);
        sumf += sub_group_shuffle_xor(sumf, 4);
        sumf += sub_group_shuffle_xor(sumf, 2);
        sumf += sub_group_shuffle_xor(sumf, 1);

        if (lane_q == 0) {
            const int im_out = i03 * ne12 + (q_head_lo + q_id);
            dst[im_out * ne1 * ne0 + r0] = sumf;
        }
    }
}

// image1d_buffer_t (texture-cache) variant of the r2=4 kernel above. Same q8_0
// row->pixel mapping as the r2=8 image kernel (136-B row = 34 uint32 px; even-blk
// half shifted 2 B). Each lane reads 8 qs bytes = 2 words (w0,w1) + its block d.
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_q8_0_f32_gqa_r4_dk128_img(
        __read_only image1d_buffer_t src0_img,
        global char * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src1 = (global char *)((global char *)src1 + offset1);
    dst  = (global float*)((global char *)dst  + offsetd);

    const int sgs_lid = get_sub_group_local_id();
    const int q_id    = sgs_lid >> 4;
    const int lane_q  = sgs_lid & 15;

    const int r0_base = get_group_id(0) * N_K_ROWS_Q8GQA;
    const int im_kv   = get_group_id(2);

    const int i02 = im_kv % ne02;
    const int i03 = im_kv / ne02;

    const int q_head_lo = i02 * GQA_RATIO_Q8GQA_R4;

    __local float4 q_loc[GQA_RATIO_Q8GQA_R4 * DK_VEC_Q8GQA_R4];
    #pragma unroll
    for (int qh = 0; qh < GQA_RATIO_Q8GQA_R4; ++qh) {
        const int qh_idx = q_head_lo + qh;
        global float4 * y4 = (global float4 *)(src1 + qh_idx * nb12 + i03 * nb13);
        if (sgs_lid < DK_VEC_Q8GQA_R4) {
            q_loc[qh * DK_VEC_Q8GQA_R4 + sgs_lid] = y4[sgs_lid];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int blk  = lane_q >> 2;
    const int eoff = (lane_q & 3) * 8;
    const int qf2  = lane_q * 2;

    const int pitch_px_row  = (int)(nb01 >> 2);
    const int pitch_px_head = (int)(nb02 >> 2);
    const int pitch_px_n13  = (int)(nb03 >> 2);
    const int head_px_base  = i02 * pitch_px_head + (i03 / r3) * pitch_px_n13;

    const int d_byte   = 34 * blk;
    const int d_pxoff  = d_byte >> 2;
    const int d_bit    = (d_byte & 3) * 8;
    const int q_byte   = 34 * blk + 2 + eoff;
    const int q_pxoff  = q_byte >> 2;
    const uint q_sh    = (uint)((q_byte & 3) * 8);   // 0 or 16

    #pragma unroll
    for (int dr = 0; dr < N_K_ROWS_Q8GQA; ++dr) {
        const int r0 = r0_base + dr;
        const int row_px = r0 * pitch_px_row + head_px_base;

        const half  d  = as_half((ushort)((read_imageui(src0_img, row_px + d_pxoff).x >> d_bit) & 0xFFFFu));
        const float df = convert_float(d);

        const int qpx = row_px + q_pxoff;
        uint w0, w1;
        if (q_sh == 0u) {
            w0 = read_imageui(src0_img, qpx + 0).x;
            w1 = read_imageui(src0_img, qpx + 1).x;
        } else {
            const uint p0 = read_imageui(src0_img, qpx + 0).x;
            const uint p1 = read_imageui(src0_img, qpx + 1).x;
            const uint p2 = read_imageui(src0_img, qpx + 2).x;
            w0 = (p0 >> q_sh) | (p1 << (32u - q_sh));
            w1 = (p1 >> q_sh) | (p2 << (32u - q_sh));
        }

        const char4 c0 = as_char4(w0);
        const char4 c1 = as_char4(w1);
        const float4 qa = q_loc[q_id * DK_VEC_Q8GQA_R4 + qf2 + 0];
        const float4 qb = q_loc[q_id * DK_VEC_Q8GQA_R4 + qf2 + 1];
        float sumf =
              (float)c0.s0*qa.s0 + (float)c0.s1*qa.s1 + (float)c0.s2*qa.s2 + (float)c0.s3*qa.s3
            + (float)c1.s0*qb.s0 + (float)c1.s1*qb.s1 + (float)c1.s2*qb.s2 + (float)c1.s3*qb.s3;
        sumf *= df;

        sumf += sub_group_shuffle_xor(sumf, 8);
        sumf += sub_group_shuffle_xor(sumf, 4);
        sumf += sub_group_shuffle_xor(sumf, 2);
        sumf += sub_group_shuffle_xor(sumf, 1);

        if (lane_q == 0) {
            const int im_out = i03 * ne12 + (q_head_lo + q_id);
            dst[im_out * ne1 * ne0 + r0] = sumf;
        }
    }
}

// Generic decode KQ for a q8_0 K-cache: any GQA ratio, any DK in [32, 512] that
// is a multiple of 32. The GQA-coalesced kernels above bake the GQA ratio and DK
// into their lane geometry (8 Q-heads x 8 lanes at DK=128), so a model whose
// attention shape falls outside that table -- e.g. Gemma-4, whose SWA layers are
// DK=256/r2=2 and whose global layers are DK=512/r2=8 -- misses them entirely and
// falls back to dequantizing the whole K view to f16 on every attention op. That
// fallback costs ~2.5x end-to-end even though the GPU does the same work, because
// the dequant pass runs off the critical path of the GPU kernels.
//
// This kernel takes the other axis: instead of splitting lanes across Q-heads, it
// splits them across DK. One subgroup owns one (row-block, Q-head, batch) and each
// lane owns 8 contiguous elements of the K row -- lane l covers [8l, 8l+8), i.e.
// q8_0 block l/4, byte offset (l%4)*8. 64 lanes x 8 = 512 elements, so DK<=512
// needs at most one pass and lanes beyond DK/8 sit out. K stays q8_0: no dequant
// pass, no per-op scratch buffer.
//
// Slower per row than the coalesced kernels (it re-reads K once per Q-head), so
// the host only selects it when the coalesced table misses.
#define N_ROWS_Q8GEN  4      // K rows per subgroup, to amortize the Q load
#define DK_MAX_Q8GEN  512

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_q8_0_f32_kq_gen(
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src0 = (global char *)((global char *)src0 + offset0);
    src1 = (global char *)((global char *)src1 + offset1);
    dst  = (global float*)((global char *)dst  + offsetd);

    const int lid = get_sub_group_local_id();   // 0..63

    const int r0_base = get_group_id(0) * N_ROWS_Q8GEN;
    const int im      = get_group_id(2);        // Q-head across batch

    const int i12 = im % ne12;                  // Q-head
    const int i13 = im / ne12;                  // batch
    const int i02 = i12 / r2;                   // K-head this Q-head reads
    const int i03 = i13 / r3;

    // This lane's slice of the row: 8 contiguous elements.
    const int e0   = lid * 8;                   // first element
    const bool act = (e0 < ne00);               // lanes past DK sit out
    const int blk  = e0 >> 5;                   // q8_0 block index (32 elems each)
    const int boff = e0 & 31;                   // byte offset inside the block

    // Q for this head: the 8 f32 this lane needs (2 float4, contiguous).
    float4 qa = (float4)(0.0f);
    float4 qb = (float4)(0.0f);
    if (act) {
        global float4 * q4 = (global float4 *)(src1 + i12 * nb12 + i13 * nb13);
        qa = q4[(e0 >> 2) + 0];
        qb = q4[(e0 >> 2) + 1];
    }

    const ulong head_off = i02 * nb02 + i03 * nb03;

    for (int dr = 0; dr < N_ROWS_Q8GEN; ++dr) {
        const int r0 = r0_base + dr;
        if (r0 >= ne01) {
            break;
        }

        float sumf = 0.0f;
        if (act) {
            global block_q8_0 * kb = (global block_q8_0 *)(src0 + r0 * nb01 + head_off) + blk;
            const float d = convert_float(kb->d);
            global char * qs = kb->qs + boff;

            sumf = (float)qs[0] * qa.s0
                 + (float)qs[1] * qa.s1
                 + (float)qs[2] * qa.s2
                 + (float)qs[3] * qa.s3
                 + (float)qs[4] * qb.s0
                 + (float)qs[5] * qb.s1
                 + (float)qs[6] * qb.s2
                 + (float)qs[7] * qb.s3;
            sumf *= d;
        }

        const float tot = sub_group_reduce_add(sumf);
        if (lid == 0) {
            dst[im * ne1 * ne0 + r0] = tot;
        }
    }
}

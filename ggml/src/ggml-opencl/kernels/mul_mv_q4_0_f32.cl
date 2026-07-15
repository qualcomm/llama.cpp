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

#define QK4_0                   32
#define QR4_0                   2
#define QK4_1                   32
#define QR4_1                   2
#define QK5_0                   32
#define QR5_0                   2
#define QK5_1                   32
#define QR5_1                   2
#define QK8_0                   32
#define QR8_0                   1
#define QK_K                    256
#define K_QUANTS_PER_ITERATION  2

typedef char int8_t;
typedef uchar uint8_t;
typedef short int16_t;
typedef ushort uint16_t;
typedef int int32_t;
typedef uint uint32_t;

//------------------------------------------------------------------------------
// block_q4_0
//------------------------------------------------------------------------------
struct block_q4_0
{
    half d;
    uint8_t qs[QK4_0 / 2];
};

//------------------------------------------------------------------------------
// mul_vec_q_n_f32
//------------------------------------------------------------------------------
// function for calculate inner product between half a q4_0 block and 16 floats (yl), sumy is SUM(yl[i])
// il indicates where the q4 quants begin (0 or QK4_0/4)
// we assume that the yl's have been multiplied with the appropriate scale factor
// that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)
inline float block_q_4_0_dot_y(
        global struct block_q4_0 * qb_curr,
        float sumy,
        private float * yl,
        int il
) {
    float d = qb_curr->d;
    float2 acc = 0.f;
    global ushort * qs = ((global ushort *)qb_curr + 1 + il/2);
    for (int i = 0; i < 8; i+=2) {
        acc.s0 += yl[i + 0] * (qs[i / 2] & 0x000F)
                + yl[i + 1] * (qs[i / 2] & 0x0F00);
        acc.s1 += yl[i + 8] * (qs[i / 2] & 0x00F0)
                + yl[i + 9] * (qs[i / 2] & 0xF000);
    }
    return d * (sumy * -8.f + acc.s0 + acc.s1);
}

#ifdef INTEL_GPU
#define N_DST 4 // each SIMD group works on 4 rows
#define N_SIMDGROUP 1 // number of SIMD groups in a thread group
#define N_SIMDWIDTH 16 // assuming SIMD group size is 16
#elif defined (ADRENO_GPU)
#define N_DST 4
#define N_SIMDGROUP 1
#define N_SIMDWIDTH 64
#endif

inline void mul_vec_q_n_f32(
        global void * src0,
        global float * src1,
        global float * dst,
        int ne00,
        int ne01,
        int ne02,
        int ne10,
        int ne12,
        int ne0,
        int ne1,
        int r2,
        int r3
) {

    const ulong nb = ne00/QK4_0;

    int r0 = get_group_id(0);
    int r1 = get_group_id(1);
    int im = get_group_id(2);

    // (r0 * N_SIMDGROUP + get_sub_group_id()) is essenatially the linear global
    // id of a SIMD group in the grid.
    int first_row = (r0 * N_SIMDGROUP + get_sub_group_id()) * N_DST;

    int i12 = im%ne12;
    int i13 = im/ne12;

    ulong offset0 = first_row * nb + (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);

    global struct block_q4_0 * x = (global struct block_q4_0 *) src0 + offset0;
    global float             * y = (global float             *) src1 + r1*ne10 + im*ne00*ne1;

    float yl[16];       // src1 vector cache
    float sumf[N_DST]={0.f};

    int ix = get_sub_group_local_id()/2;
    int il = 8*(get_sub_group_local_id()%2);

    global float * yb = y + ix * QK4_0 + il;

    // each thread in a SIMD group deals with half a block.
    for (int ib = ix; ib < nb; ib += N_SIMDWIDTH/2) {
        float sumy = 0;
        for (int i = 0; i < 8; i += 2) {
            sumy += yb[i] + yb[i+1];
            yl[i+0] = yb[i+ 0];
            yl[i+1] = yb[i+ 1]/256.f;
            sumy += yb[i+16] + yb[i+17];
            yl[i+8] = yb[i+16]/16.f;
            yl[i+9] = yb[i+17]/4096.f;
        }

        for (int row = 0; row < N_DST; row++) {
            sumf[row] += block_q_4_0_dot_y(x+ib+row*nb, sumy, yl, il);
        }

        // One thread in a SIMD group (i.e., subgroup) handles a half block,
        // hence then entire SIMD group handles SIMDWIDTH/2 blocks.
        // y points to the activation matrix (of type float). Therefore for
        // one thread, the # of blocks y should advance is SIMDWIDTH/2 (because
        // SIMDWIDTH/2 blocks are processed by a SIMD group) - in terms of
        // floats, it is QK4_0 * (SIMDWIDTH/2), where QK4_0 is the block size.
        yb += QK4_0 * (N_SIMDWIDTH/2);
    }

    // The above does not work for Adreno - it produces incorrect results for
    // row = 1, 2, 3 and only row = 0 gives the correct result.
    // If N_DST is changed, the below array must be initialized accordingly.
    // This also seems to perform better on Intel.
    float tot[N_DST] = {
        sub_group_reduce_add(sumf[0]), sub_group_reduce_add(sumf[1]),
        sub_group_reduce_add(sumf[2]), sub_group_reduce_add(sumf[3])};
    for (int row = 0; row < N_DST; ++row) {
        if (get_sub_group_local_id() == 0 && first_row + row < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = tot[row];
        }
    }
}

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_16
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_q4_0_f32(
        global void * src0,
        ulong offset0,
        global float * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne10,
        int ne12,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src0 = (global void*)((global char*)src0 + offset0);
    src1 = (global float*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    mul_vec_q_n_f32(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3);
}

// GQA-coalesced decode KQ for a q4_0 K-cache (DK=128, r2=8, r3=1, ne11==1) --
// the -36% KV-DDR analog of the q8_0 _gqa8_dk128 kernels. K stays q4_0 (AoS
// block_q4_0, runtime tensor, never SoA). 64-lane subgroup = 8 Q-heads x 8 lanes;
// lane lane_q owns K/Q elements [lane_q*16, +16) (blk=lane_q>>1, half=lane_q&1).
// q4_0 packs 32 elems per 18-B block into qs[16]: elem i<16 = low nibble of qs[i],
// i>=16 = high nibble of qs[i-16] -> the two half-lanes of a block read the SAME
// 16 qs bytes, one taking low nibbles, the other high. Dequant = (nibble-8)*d.
#define N_K_ROWS_Q4GQA   16
#define GQA_RATIO_Q4GQA  8
#define DK_VEC_Q4GQA     32   // DK/4 for DK=128

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_q4_0_f32_gqa8_dk128(
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
    const int q_id    = sgs_lid >> 3;
    const int lane_q  = sgs_lid & 7;

    const int r0_base = get_group_id(0) * N_K_ROWS_Q4GQA;
    const int im_kv   = get_group_id(2);

    const int i02 = im_kv % ne02;
    const int i03 = im_kv / ne02;

    const int q_head_lo = i02 * GQA_RATIO_Q4GQA;

    __local float4 q_loc[GQA_RATIO_Q4GQA * DK_VEC_Q4GQA];
    #pragma unroll
    for (int qh = 0; qh < GQA_RATIO_Q4GQA; ++qh) {
        const int qh_idx = q_head_lo + qh;
        global float4 * y4 = (global float4 *)(src1 + qh_idx * nb12 + i03 * nb13);
        if (sgs_lid < DK_VEC_Q4GQA) {
            q_loc[qh * DK_VEC_Q4GQA + sgs_lid] = y4[sgs_lid];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int blk = lane_q >> 1;            // 0..3 (q4_0 block)
    const int nsh = (lane_q & 1) * 4;       // 0 (low nibble) or 4 (high nibble)
    const int qf4 = lane_q * 4;

    const ulong head_off = (ulong)i02 * nb02 + (ulong)(i03 / r3) * nb03;

    #pragma unroll
    for (int dr = 0; dr < N_K_ROWS_Q4GQA; ++dr) {
        const int r0 = r0_base + dr;
        global struct block_q4_0 * kb =
            (global struct block_q4_0 *)(src0 + r0 * nb01 + head_off) + blk;
        const float d = convert_float(kb->d);
        global uchar * qs = kb->qs;         // 16 bytes shared by both half-lanes

        float sumf = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float4 qv = q_loc[q_id * DK_VEC_Q4GQA + qf4 + j];
            const int b = j * 4;
            sumf += (float)(((qs[b + 0] >> nsh) & 0xF) - 8) * qv.s0
                  + (float)(((qs[b + 1] >> nsh) & 0xF) - 8) * qv.s1
                  + (float)(((qs[b + 2] >> nsh) & 0xF) - 8) * qv.s2
                  + (float)(((qs[b + 3] >> nsh) & 0xF) - 8) * qv.s3;
        }
        sumf *= d;

        sumf += sub_group_shuffle_xor(sumf, 4);
        sumf += sub_group_shuffle_xor(sumf, 2);
        sumf += sub_group_shuffle_xor(sumf, 1);

        if (lane_q == 0) {
            const int im_out = i03 * ne12 + (q_head_lo + q_id);
            dst[im_out * ne1 * ne0 + r0] = sumf;
        }
    }
}

// image1d_buffer_t (texture-cache) variant of kernel_mul_mat_q4_0_f32_gqa8_dk128.
// q4_0 row (DK=128) = 4 blocks x 18 B = 72 B = exactly 18 uint32 pixels. d (2 B)
// at byte 18*blk, qs[16] at 18*blk+2 -> same even/odd 2-byte-shift handling as the
// q8_0 image kernel (read 5 px + shift-combine for shifted blocks), plus 4-bit
// nibble unpack. CL_R/CL_UNSIGNED_INT32 image, opt via the host img dispatch.
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_q4_0_f32_gqa8_dk128_img(
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
    const int q_id    = sgs_lid >> 3;
    const int lane_q  = sgs_lid & 7;

    const int r0_base = get_group_id(0) * N_K_ROWS_Q4GQA;
    const int im_kv   = get_group_id(2);

    const int i02 = im_kv % ne02;
    const int i03 = im_kv / ne02;

    const int q_head_lo = i02 * GQA_RATIO_Q4GQA;

    __local float4 q_loc[GQA_RATIO_Q4GQA * DK_VEC_Q4GQA];
    #pragma unroll
    for (int qh = 0; qh < GQA_RATIO_Q4GQA; ++qh) {
        const int qh_idx = q_head_lo + qh;
        global float4 * y4 = (global float4 *)(src1 + qh_idx * nb12 + i03 * nb13);
        if (sgs_lid < DK_VEC_Q4GQA) {
            q_loc[qh * DK_VEC_Q4GQA + sgs_lid] = y4[sgs_lid];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int blk = lane_q >> 1;
    const int nsh = (lane_q & 1) * 4;
    const int qf4 = lane_q * 4;

    const int pitch_px_row  = (int)(nb01 >> 2);   // 72 B -> 18 px
    const int pitch_px_head = (int)(nb02 >> 2);
    const int pitch_px_n13  = (int)(nb03 >> 2);
    const int head_px_base  = i02 * pitch_px_head + (i03 / r3) * pitch_px_n13;

    const int d_byte  = 18 * blk;
    const int d_pxoff = d_byte >> 2;
    const int d_bit   = (d_byte & 3) * 8;
    const int q_byte  = 18 * blk + 2;
    const int q_pxoff = q_byte >> 2;
    const uint q_sh   = (uint)((q_byte & 3) * 8);   // 0 or 16

    #pragma unroll
    for (int dr = 0; dr < N_K_ROWS_Q4GQA; ++dr) {
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

        float sumf = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint w = (j == 0) ? w0 : (j == 1) ? w1 : (j == 2) ? w2 : w3;
            const float4 nv = (float4)(
                (float)((w >> (0u  + nsh)) & 0xFu),
                (float)((w >> (8u  + nsh)) & 0xFu),
                (float)((w >> (16u + nsh)) & 0xFu),
                (float)((w >> (24u + nsh)) & 0xFu)) - 8.0f;
            const float4 qv = q_loc[q_id * DK_VEC_Q4GQA + qf4 + j];
            sumf += nv.s0*qv.s0 + nv.s1*qv.s1 + nv.s2*qv.s2 + nv.s3*qv.s3;
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
// DK=256, r2=8 variants for Qwen3.6-35B-A3B (n_head_kv=2 => GQA r=8, head_dim=256).
// 64-lane subgroup = 8 Q-heads x 8 lanes; each lane owns a WHOLE q4_0 block (32
// elems = both nibble halves of all 16 qs bytes). Low nibbles -> K elems [0,16) ->
// Q float4 [qf4,qf4+4); high nibbles -> K elems [16,32) -> Q float4 [qf4+4,qf4+8).
// ===========================================================================
#define N_K_ROWS_Q4GQA256   16
#define GQA_RATIO_Q4GQA256  8
#define DK_VEC_Q4GQA256     64   // DK/4 for DK=256

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_q4_0_f32_gqa8_dk256(
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
    const int q_id    = sgs_lid >> 3;
    const int lane_q  = sgs_lid & 7;

    const int r0_base = get_group_id(0) * N_K_ROWS_Q4GQA256;
    const int im_kv   = get_group_id(2);

    const int i02 = im_kv % ne02;
    const int i03 = im_kv / ne02;

    const int q_head_lo = i02 * GQA_RATIO_Q4GQA256;

    __local float4 q_loc[GQA_RATIO_Q4GQA256 * DK_VEC_Q4GQA256];
    #pragma unroll
    for (int qh = 0; qh < GQA_RATIO_Q4GQA256; ++qh) {
        const int qh_idx = q_head_lo + qh;
        global float4 * y4 = (global float4 *)(src1 + qh_idx * nb12 + i03 * nb13);
        q_loc[qh * DK_VEC_Q4GQA256 + sgs_lid] = y4[sgs_lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int blk = lane_q;          // this lane's whole q4_0 block
    const int qf4 = lane_q * 8;      // 8 float4 (32 elems)

    const ulong head_off = (ulong)i02 * nb02 + (ulong)(i03 / r3) * nb03;

    #pragma unroll
    for (int dr = 0; dr < N_K_ROWS_Q4GQA256; ++dr) {
        const int r0 = r0_base + dr;
        global struct block_q4_0 * kb =
            (global struct block_q4_0 *)(src0 + r0 * nb01 + head_off) + blk;
        const float d = convert_float(kb->d);
        global uchar * qs = kb->qs;   // 16 bytes (full block)

        float sumf = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {       // low nibbles -> K elems [0,16)
            const float4 qv = q_loc[q_id * DK_VEC_Q4GQA256 + qf4 + j];
            const int b = j * 4;
            sumf += (float)(((int)(qs[b + 0] & 0xF)) - 8) * qv.s0
                  + (float)(((int)(qs[b + 1] & 0xF)) - 8) * qv.s1
                  + (float)(((int)(qs[b + 2] & 0xF)) - 8) * qv.s2
                  + (float)(((int)(qs[b + 3] & 0xF)) - 8) * qv.s3;
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {       // high nibbles -> K elems [16,32)
            const float4 qv = q_loc[q_id * DK_VEC_Q4GQA256 + qf4 + 4 + j];
            const int b = j * 4;
            sumf += (float)(((int)(qs[b + 0] >> 4)) - 8) * qv.s0
                  + (float)(((int)(qs[b + 1] >> 4)) - 8) * qv.s1
                  + (float)(((int)(qs[b + 2] >> 4)) - 8) * qv.s2
                  + (float)(((int)(qs[b + 3] >> 4)) - 8) * qv.s3;
        }
        sumf *= d;

        sumf += sub_group_shuffle_xor(sumf, 4);
        sumf += sub_group_shuffle_xor(sumf, 2);
        sumf += sub_group_shuffle_xor(sumf, 1);

        if (lane_q == 0) {
            const int im_out = i03 * ne12 + (q_head_lo + q_id);
            dst[im_out * ne1 * ne0 + r0] = sumf;
        }
    }
}

// image1d_buffer_t variant of kernel_mul_mat_q4_0_f32_gqa8_dk256.
// q4_0 row (DK=256) = 8 blocks x 18 B = 144 B = 36 uint32 pixels. Lane owns whole
// block lane_q: d at byte 18*lane_q, qs[16] at 18*lane_q+2 (even/odd 2-byte-shift
// as the DK=128 kernel). Unpack 16 qs bytes (4 words): low nibbles -> Q[qf4,qf4+4),
// high nibbles -> Q[qf4+4,qf4+8).
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_q4_0_f32_gqa8_dk256_img(
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
    const int q_id    = sgs_lid >> 3;
    const int lane_q  = sgs_lid & 7;

    const int r0_base = get_group_id(0) * N_K_ROWS_Q4GQA256;
    const int im_kv   = get_group_id(2);

    const int i02 = im_kv % ne02;
    const int i03 = im_kv / ne02;

    const int q_head_lo = i02 * GQA_RATIO_Q4GQA256;

    __local float4 q_loc[GQA_RATIO_Q4GQA256 * DK_VEC_Q4GQA256];
    #pragma unroll
    for (int qh = 0; qh < GQA_RATIO_Q4GQA256; ++qh) {
        const int qh_idx = q_head_lo + qh;
        global float4 * y4 = (global float4 *)(src1 + qh_idx * nb12 + i03 * nb13);
        q_loc[qh * DK_VEC_Q4GQA256 + sgs_lid] = y4[sgs_lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int qf4 = lane_q * 8;

    const int pitch_px_row  = (int)(nb01 >> 2);   // 144 B -> 36 px
    const int pitch_px_head = (int)(nb02 >> 2);
    const int pitch_px_n13  = (int)(nb03 >> 2);
    const int head_px_base  = i02 * pitch_px_head + (i03 / r3) * pitch_px_n13;

    const int d_byte  = 18 * lane_q;
    const int d_pxoff = d_byte >> 2;
    const int d_bit   = (d_byte & 3) * 8;
    const int q_byte  = 18 * lane_q + 2;
    const int q_pxoff = q_byte >> 2;
    const uint q_sh   = (uint)((q_byte & 3) * 8);   // 0 or 16

    #pragma unroll
    for (int dr = 0; dr < N_K_ROWS_Q4GQA256; ++dr) {
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

        float sumf = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {       // low nibbles -> Q[qf4,qf4+4)
            const uint w = (j == 0) ? w0 : (j == 1) ? w1 : (j == 2) ? w2 : w3;
            const float4 nv = (float4)(
                (float)((w >> 0u)  & 0xFu),
                (float)((w >> 8u)  & 0xFu),
                (float)((w >> 16u) & 0xFu),
                (float)((w >> 24u) & 0xFu)) - 8.0f;
            const float4 qv = q_loc[q_id * DK_VEC_Q4GQA256 + qf4 + j];
            sumf += nv.s0*qv.s0 + nv.s1*qv.s1 + nv.s2*qv.s2 + nv.s3*qv.s3;
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {       // high nibbles -> Q[qf4+4,qf4+8)
            const uint w = (j == 0) ? w0 : (j == 1) ? w1 : (j == 2) ? w2 : w3;
            const float4 nv = (float4)(
                (float)((w >> 4u)  & 0xFu),
                (float)((w >> 12u) & 0xFu),
                (float)((w >> 20u) & 0xFu),
                (float)((w >> 28u) & 0xFu)) - 8.0f;
            const float4 qv = q_loc[q_id * DK_VEC_Q4GQA256 + qf4 + 4 + j];
            sumf += nv.s0*qv.s0 + nv.s1*qv.s1 + nv.s2*qv.s2 + nv.s3*qv.s3;
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
// Memory-footprint play (q4 K cache = quarter K DDR at depth). 64-lane subgroup =
// 4 Q-heads x 16 lanes; DK=256 = 8 q4_0 blocks, each lane owns a HALF block (16
// elems = one nibble-half of a block: blk=lane_q>>1, nsh=(lane_q&1)*4). Same body
// as gqa8_dk128, 16 lanes/Q-head (reduce {8,4,2,1}), DK_VEC=64.
// ===========================================================================
#define N_K_ROWS_Q4GQA_R4_256   16
#define GQA_RATIO_Q4GQA_R4_256  4
#define DK_VEC_Q4GQA_R4_256     64   // DK/4 for DK=256

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_q4_0_f32_gqa_r4_dk256(
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
    const int q_id    = sgs_lid >> 4;
    const int lane_q  = sgs_lid & 15;

    const int r0_base = get_group_id(0) * N_K_ROWS_Q4GQA_R4_256;
    const int im_kv   = get_group_id(2);

    const int i02 = im_kv % ne02;
    const int i03 = im_kv / ne02;

    const int q_head_lo = i02 * GQA_RATIO_Q4GQA_R4_256;

    __local float4 q_loc[GQA_RATIO_Q4GQA_R4_256 * DK_VEC_Q4GQA_R4_256];
    #pragma unroll
    for (int qh = 0; qh < GQA_RATIO_Q4GQA_R4_256; ++qh) {
        const int qh_idx = q_head_lo + qh;
        global float4 * y4 = (global float4 *)(src1 + qh_idx * nb12 + i03 * nb13);
        q_loc[qh * DK_VEC_Q4GQA_R4_256 + sgs_lid] = y4[sgs_lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int blk = lane_q >> 1;            // 0..7 (q4_0 block)
    const int nsh = (lane_q & 1) * 4;       // 0 (low nibble) or 4 (high nibble)
    const int qf4 = lane_q * 4;

    const ulong head_off = (ulong)i02 * nb02 + (ulong)(i03 / r3) * nb03;

    #pragma unroll
    for (int dr = 0; dr < N_K_ROWS_Q4GQA_R4_256; ++dr) {
        const int r0 = r0_base + dr;
        global struct block_q4_0 * kb =
            (global struct block_q4_0 *)(src0 + r0 * nb01 + head_off) + blk;
        const float d = convert_float(kb->d);
        global uchar * qs = kb->qs;         // 16 bytes shared by both half-lanes

        float sumf = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float4 qv = q_loc[q_id * DK_VEC_Q4GQA_R4_256 + qf4 + j];
            const int b = j * 4;
            sumf += (float)(((qs[b + 0] >> nsh) & 0xF) - 8) * qv.s0
                  + (float)(((qs[b + 1] >> nsh) & 0xF) - 8) * qv.s1
                  + (float)(((qs[b + 2] >> nsh) & 0xF) - 8) * qv.s2
                  + (float)(((qs[b + 3] >> nsh) & 0xF) - 8) * qv.s3;
        }
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

// image1d_buffer_t variant of kernel_mul_mat_q4_0_f32_gqa_r4_dk256.
// Row = 8 q4_0 blocks x 18 B = 144 B = 36 px. Lane owns a half block (one nibble
// half): blk=lane_q>>1, nsh=(lane_q&1)*4; d at 18*blk, qs[16] at 18*blk+2.
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_q4_0_f32_gqa_r4_dk256_img(
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

    const int r0_base = get_group_id(0) * N_K_ROWS_Q4GQA_R4_256;
    const int im_kv   = get_group_id(2);

    const int i02 = im_kv % ne02;
    const int i03 = im_kv / ne02;

    const int q_head_lo = i02 * GQA_RATIO_Q4GQA_R4_256;

    __local float4 q_loc[GQA_RATIO_Q4GQA_R4_256 * DK_VEC_Q4GQA_R4_256];
    #pragma unroll
    for (int qh = 0; qh < GQA_RATIO_Q4GQA_R4_256; ++qh) {
        const int qh_idx = q_head_lo + qh;
        global float4 * y4 = (global float4 *)(src1 + qh_idx * nb12 + i03 * nb13);
        q_loc[qh * DK_VEC_Q4GQA_R4_256 + sgs_lid] = y4[sgs_lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int blk = lane_q >> 1;
    const int nsh = (lane_q & 1) * 4;
    const int qf4 = lane_q * 4;

    const int pitch_px_row  = (int)(nb01 >> 2);   // 144 B -> 36 px
    const int pitch_px_head = (int)(nb02 >> 2);
    const int pitch_px_n13  = (int)(nb03 >> 2);
    const int head_px_base  = i02 * pitch_px_head + (i03 / r3) * pitch_px_n13;

    const int d_byte  = 18 * blk;
    const int d_pxoff = d_byte >> 2;
    const int d_bit   = (d_byte & 3) * 8;
    const int q_byte  = 18 * blk + 2;
    const int q_pxoff = q_byte >> 2;
    const uint q_sh   = (uint)((q_byte & 3) * 8);

    #pragma unroll
    for (int dr = 0; dr < N_K_ROWS_Q4GQA_R4_256; ++dr) {
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

        float sumf = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint w = (j == 0) ? w0 : (j == 1) ? w1 : (j == 2) ? w2 : w3;
            const float4 nv = (float4)(
                (float)((w >> (0u  + nsh)) & 0xFu),
                (float)((w >> (8u  + nsh)) & 0xFu),
                (float)((w >> (16u + nsh)) & 0xFu),
                (float)((w >> (24u + nsh)) & 0xFu)) - 8.0f;
            const float4 qv = q_loc[q_id * DK_VEC_Q4GQA_R4_256 + qf4 + j];
            sumf += nv.s0*qv.s0 + nv.s1*qv.s1 + nv.s2*qv.s2 + nv.s3*qv.s3;
        }
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
// r2=4 variants (DK=128) for Llama-3-8B. 4 Q-heads x 16 lanes; each lane owns 8
// head_dim elements (a QUARTER block). For q4_0 a quarter = 8 nibbles from 8 qs
// bytes: blk=lane_q>>2, nibble half nsh=((lane_q>>1)&1)*4, byte offset
// qoff=(lane_q&1)*8. K elem = lane_q*8+m, pairs 2 float4 of Q (qf2=lane_q*2).
// ===========================================================================
#define GQA_RATIO_Q4GQA_R4  4
#define DK_VEC_Q4GQA_R4     32

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_q4_0_f32_gqa_r4_dk128(
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
    const int q_id    = sgs_lid >> 4;
    const int lane_q  = sgs_lid & 15;

    const int r0_base = get_group_id(0) * N_K_ROWS_Q4GQA;
    const int im_kv   = get_group_id(2);

    const int i02 = im_kv % ne02;
    const int i03 = im_kv / ne02;

    const int q_head_lo = i02 * GQA_RATIO_Q4GQA_R4;

    __local float4 q_loc[GQA_RATIO_Q4GQA_R4 * DK_VEC_Q4GQA_R4];
    #pragma unroll
    for (int qh = 0; qh < GQA_RATIO_Q4GQA_R4; ++qh) {
        const int qh_idx = q_head_lo + qh;
        global float4 * y4 = (global float4 *)(src1 + qh_idx * nb12 + i03 * nb13);
        if (sgs_lid < DK_VEC_Q4GQA_R4) {
            q_loc[qh * DK_VEC_Q4GQA_R4 + sgs_lid] = y4[sgs_lid];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int blk  = lane_q >> 2;
    const int nsh  = ((lane_q >> 1) & 1) * 4;   // low/high nibble
    const int qoff = (lane_q & 1) * 8;          // qs byte offset
    const int qf2  = lane_q * 2;

    const ulong head_off = (ulong)i02 * nb02 + (ulong)(i03 / r3) * nb03;

    #pragma unroll
    for (int dr = 0; dr < N_K_ROWS_Q4GQA; ++dr) {
        const int r0 = r0_base + dr;
        global struct block_q4_0 * kb =
            (global struct block_q4_0 *)(src0 + r0 * nb01 + head_off) + blk;
        const float d = convert_float(kb->d);
        global uchar * qs = kb->qs + qoff;      // 8 bytes

        const float4 qa = q_loc[q_id * DK_VEC_Q4GQA_R4 + qf2 + 0];
        const float4 qb = q_loc[q_id * DK_VEC_Q4GQA_R4 + qf2 + 1];
        float sumf =
              (float)(((qs[0] >> nsh) & 0xF) - 8) * qa.s0
            + (float)(((qs[1] >> nsh) & 0xF) - 8) * qa.s1
            + (float)(((qs[2] >> nsh) & 0xF) - 8) * qa.s2
            + (float)(((qs[3] >> nsh) & 0xF) - 8) * qa.s3
            + (float)(((qs[4] >> nsh) & 0xF) - 8) * qb.s0
            + (float)(((qs[5] >> nsh) & 0xF) - 8) * qb.s1
            + (float)(((qs[6] >> nsh) & 0xF) - 8) * qb.s2
            + (float)(((qs[7] >> nsh) & 0xF) - 8) * qb.s3;
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

// image1d_buffer_t variant of the q4_0 r2=4 kernel. q4_0 row = 72 B = 18 px;
// lane reads 8 qs bytes = 2 words (+ block d) and unpacks one nibble each.
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_q4_0_f32_gqa_r4_dk128_img(
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

    const int r0_base = get_group_id(0) * N_K_ROWS_Q4GQA;
    const int im_kv   = get_group_id(2);

    const int i02 = im_kv % ne02;
    const int i03 = im_kv / ne02;

    const int q_head_lo = i02 * GQA_RATIO_Q4GQA_R4;

    __local float4 q_loc[GQA_RATIO_Q4GQA_R4 * DK_VEC_Q4GQA_R4];
    #pragma unroll
    for (int qh = 0; qh < GQA_RATIO_Q4GQA_R4; ++qh) {
        const int qh_idx = q_head_lo + qh;
        global float4 * y4 = (global float4 *)(src1 + qh_idx * nb12 + i03 * nb13);
        if (sgs_lid < DK_VEC_Q4GQA_R4) {
            q_loc[qh * DK_VEC_Q4GQA_R4 + sgs_lid] = y4[sgs_lid];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int blk  = lane_q >> 2;
    const int nsh  = ((lane_q >> 1) & 1) * 4;
    const int qoff = (lane_q & 1) * 8;
    const int qf2  = lane_q * 2;

    const int pitch_px_row  = (int)(nb01 >> 2);
    const int pitch_px_head = (int)(nb02 >> 2);
    const int pitch_px_n13  = (int)(nb03 >> 2);
    const int head_px_base  = i02 * pitch_px_head + (i03 / r3) * pitch_px_n13;

    const int d_byte  = 18 * blk;
    const int d_pxoff = d_byte >> 2;
    const int d_bit   = (d_byte & 3) * 8;
    const int q_byte  = 18 * blk + 2 + qoff;
    const int q_pxoff = q_byte >> 2;
    const uint q_sh   = (uint)((q_byte & 3) * 8);

    #pragma unroll
    for (int dr = 0; dr < N_K_ROWS_Q4GQA; ++dr) {
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

        const float4 qa = q_loc[q_id * DK_VEC_Q4GQA_R4 + qf2 + 0];
        const float4 qb = q_loc[q_id * DK_VEC_Q4GQA_R4 + qf2 + 1];
        const float4 nv0 = (float4)(
            (float)((w0 >> (0u  + nsh)) & 0xFu),
            (float)((w0 >> (8u  + nsh)) & 0xFu),
            (float)((w0 >> (16u + nsh)) & 0xFu),
            (float)((w0 >> (24u + nsh)) & 0xFu)) - 8.0f;
        const float4 nv1 = (float4)(
            (float)((w1 >> (0u  + nsh)) & 0xFu),
            (float)((w1 >> (8u  + nsh)) & 0xFu),
            (float)((w1 >> (16u + nsh)) & 0xFu),
            (float)((w1 >> (24u + nsh)) & 0xFu)) - 8.0f;
        float sumf = nv0.s0*qa.s0 + nv0.s1*qa.s1 + nv0.s2*qa.s2 + nv0.s3*qa.s3
                   + nv1.s0*qb.s0 + nv1.s1*qb.s1 + nv1.s2*qb.s2 + nv1.s3*qb.s3;
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

// Generic decode KQ for a q4_0 K-cache: any GQA ratio, any DK in [32, 512] that
// is a multiple of 32. The q4_0 analog of kernel_mul_mat_q8_0_f32_kq_gen -- see
// that kernel for why this exists (the GQA-coalesced table only covers
// DK in {128,256} x r2 in {4,8}, and a model outside it, e.g. Gemma-4, falls back
// to dequantizing the whole K view to f16 on every attention op).
//
// Lanes split across DK: lane l owns the 8 contiguous elements [8l, 8l+8) of the
// K row, i.e. q4_0 block l/4 at nibble offset (l%4)*8. In a q4_0 block the low
// nibbles of qs[0..15] hold elements 0..15 and the high nibbles hold 16..31, so a
// lane's 8 elements are always 8 consecutive bytes read at a single nibble
// position. 64 lanes x 8 = 512, so DK <= 512 needs one pass; lanes past DK/8 idle.
#define N_ROWS_Q4GEN  4

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_q4_0_f32_kq_gen(
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

    const int r0_base = get_group_id(0) * N_ROWS_Q4GEN;
    const int im      = get_group_id(2);        // Q-head across batch

    const int i12 = im % ne12;
    const int i13 = im / ne12;
    const int i02 = i12 / r2;                   // K-head this Q-head reads
    const int i03 = i13 / r3;

    const int e0   = lid * 8;                   // this lane's first element
    const bool act = (e0 < ne00);
    const int blk  = e0 >> 5;                   // q4_0 block index
    const int eoff = e0 & 31;                   // element offset inside the block
    const int boff = eoff & 15;                 // byte offset  (0 or 8)
    const int nsh  = (eoff >> 4) * 4;           // nibble shift (0 = low, 4 = high)

    float4 qa = (float4)(0.0f);
    float4 qb = (float4)(0.0f);
    if (act) {
        global float4 * q4 = (global float4 *)(src1 + i12 * nb12 + i13 * nb13);
        qa = q4[(e0 >> 2) + 0];
        qb = q4[(e0 >> 2) + 1];
    }

    const ulong head_off = i02 * nb02 + i03 * nb03;

    for (int dr = 0; dr < N_ROWS_Q4GEN; ++dr) {
        const int r0 = r0_base + dr;
        if (r0 >= ne01) {
            break;
        }

        float sumf = 0.0f;
        if (act) {
            global struct block_q4_0 * kb =
                (global struct block_q4_0 *)(src0 + r0 * nb01 + head_off) + blk;
            const float d = convert_float(kb->d);
            global uchar * qs = kb->qs + boff;

            sumf = (float)(((qs[0] >> nsh) & 0xF) - 8) * qa.s0
                 + (float)(((qs[1] >> nsh) & 0xF) - 8) * qa.s1
                 + (float)(((qs[2] >> nsh) & 0xF) - 8) * qa.s2
                 + (float)(((qs[3] >> nsh) & 0xF) - 8) * qa.s3
                 + (float)(((qs[4] >> nsh) & 0xF) - 8) * qb.s0
                 + (float)(((qs[5] >> nsh) & 0xF) - 8) * qb.s1
                 + (float)(((qs[6] >> nsh) & 0xF) - 8) * qb.s2
                 + (float)(((qs[7] >> nsh) & 0xF) - 8) * qb.s3;
            sumf *= d;
        }

        const float tot = sub_group_reduce_add(sumf);
        if (lid == 0) {
            dst[im * ne1 * ne0 + r0] = tot;
        }
    }
}

//------------------------------------------------------------------------------
// This file is contains kernels for data conversion.
// These kernels are used when loading the model, so its performance is less
// important.
//------------------------------------------------------------------------------
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

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
// kernel_convert_block_q4_0
// Convert the block_q4_0 format to 2 separate arrays (AOS -> SOA).
// This kernel does not deshuffle the bits.
//------------------------------------------------------------------------------
kernel void kernel_convert_block_q4_0(
    global struct block_q4_0 * src0,
    global uchar * dst_q,
    global half  * dst_d
) {
    global struct block_q4_0 * b = (global struct block_q4_0 *) src0 + get_global_id(0);
    global uchar * q = (global uchar *) dst_q + QK4_0/2*get_global_id(0);
    global half  * d = (global half *) dst_d + get_global_id(0);

    *d = b->d;

    for (int i = 0; i < QK4_0/2; ++i) {
        q[i] = b->qs[i];
    }
}

kernel void kernel_restore_block_q4_0(
    global uchar * src_q,
    global half  * src_d,
    global struct block_q4_0 * dst
) {
    global struct block_q4_0 * b = (global struct block_q4_0 *) dst + get_global_id(0);
    global uchar * q = (global uchar *) src_q + QK4_0/2*get_global_id(0);
    global half  * d = (global half *) src_d + get_global_id(0);

    b->d = *d;
    for (int i = 0; i < QK4_0/2; ++i) {
        b->qs[i] = q[i];
    }
}

//------------------------------------------------------------------------------
// kernel_convert_block_q4_0_noshuffle
// Flatten q4_0 weights and unshuffle the bits
//------------------------------------------------------------------------------

kernel void kernel_convert_block_q4_0_noshuffle(
    global struct block_q4_0 * src0,
    global uchar * dst_q,
    global half  * dst_d
) {
    global struct block_q4_0 * b = (global struct block_q4_0 *) src0 + get_global_id(0);
    global uchar * q = (global uchar *) dst_q + QK4_0/2*get_global_id(0);
    global half  * d = (global half *) dst_d + get_global_id(0);

    *d = b->d;
    for (int i = 0; i < QK4_0/4; ++i) {
        uchar x0 = b->qs[2*i + 0];
        uchar x1 = b->qs[2*i + 1];

        q[i + 0      ] = convert_uchar(x0 & 0x0F) | convert_uchar((x1 & 0x0F) << 4);
        q[i + QK4_0/4] = convert_uchar((x0 & 0xF0) >> 4) | convert_uchar(x1 & 0xF0);

#ifdef ADRENO_GPU
        // Workaround for adreno - must have the following printf statement for
        // the kernel to work properly. Otherwise it produces incorrect result.
        // convert_uchar above also seems necessary.
        // Compare against a large number so that it does not print anything.
        // get_sub_group_local_id() also works.
        if (get_global_id(0) == 65536*4096) {
            printf("%04x - %02x\n", *(global ushort*)d, ((x0 & 0xF0) >> 4) | (x1 & 0xF0));
        }
#endif
    }
}

kernel void kernel_restore_block_q4_0_noshuffle(
    global uchar * src_q,
    global half  * src_d,
    global struct block_q4_0 * dst,
    uchar mask_0F,
    uchar mask_F0
) {
    global struct block_q4_0 * b = (global struct block_q4_0 *) dst + get_global_id(0);
    global uchar * q = (global uchar *) src_q + QK4_0/2*get_global_id(0);
    global half  * d = (global half *) src_d + get_global_id(0);

    b->d = *d;
    for (int i = 0; i < QK4_0/4; ++i) {
        uchar x0 = q[i + 0      ] ;
        uchar x1 = q[i + QK4_0/4];

        b->qs[2*i + 0] = convert_uchar((x0 & mask_0F) | ((x1 & mask_0F) << 4));
        b->qs[2*i + 1] = convert_uchar(((x0 & mask_F0) >> 4) | (x1 & mask_F0));
    }
}

//------------------------------------------------------------------------------
// kernel_convert_block_q4_0_trans4_ns
// Convert + unshuffle + transpose q4_0 expert weights into the noshuffle MoE
// (_ns) layout consumed by kernel_gemm/gemv_moe_q4_0_f32_ns.
//------------------------------------------------------------------------------
kernel void kernel_convert_block_q4_0_trans4_ns(
    global struct block_q4_0 * src0,
    __global uint * dst_q,
    __global half * dst_d,
    uint ne00,
    uint ne01
) {
    uint i00 = get_global_id(1);
    uint i01 = get_global_id(0);
    uint i02 = get_global_id(2);

    if (i01 >= ne01) {
        return;
    }

    uint ne00_blk = ne00 / QK4_0;
    uint src_blk_offset = i00 + i01 * ne00_blk + i02 * ne00_blk * ne01;
    uint dst_blk_offset = i01 + i00 * ne01 + i02 * ne00_blk * ne01;

    global struct block_q4_0 * b = src0 + src_blk_offset;
    dst_d[dst_blk_offset] = b->d;

    // extract quantization and unshuffle
    ushort8 pre_block = ((global ushort8 *)(&(b->qs[0])))[0];

    ushort8 post_block = (ushort8)(0);

    uchar * pre_block_ptr = (uchar *)(&pre_block);
    uchar * post_block_ptr = (uchar *)(&post_block);

    for (int i = 0; i < QK4_0 / 4; ++i) {
        uchar x0 = pre_block_ptr[2*i + 0];
        uchar x1 = pre_block_ptr[2*i + 1];

        post_block_ptr[i + 0        ] = convert_uchar(x0 & 0x0F) | convert_uchar((x1 & 0x0F) << 4);
        post_block_ptr[i + QK4_0 / 4] = convert_uchar((x0 & 0xF0) >> 4) | convert_uchar(x1 & 0xF0);
    }

    uint4 q_block = as_uint4(post_block);

    uint offset = i02 * ne00_blk * ne01 * 4 + i00 * ne01 * 4 + i01;
    dst_q[offset] = q_block.x;
    dst_q[offset + ne01] = q_block.y;
    dst_q[offset + ne01 * 2] = q_block.z;
    dst_q[offset + ne01 * 3] = q_block.w;
}

kernel void kernel_restore_block_q4_0_trans4_ns(
    __global uint * src_q,
    __global half * src_d,
    __global struct block_q4_0 * dst0,
    uint ne00,
    uint ne01
) {
    uint i00 = get_global_id(1);
    uint i01 = get_global_id(0);
    uint i02 = get_global_id(2);

    if (i01 >= ne01) {
        return;
    }

    uint ne00_blk = ne00 / QK4_0;
    uint dst_blk_offset = i00 + i01 * ne00_blk + i02 * ne00_blk * ne01;
    uint src_d_offset = i01 + i00 * ne01 + i02 * ne00_blk * ne01;

    __global struct block_q4_0 * b = dst0 + dst_blk_offset;
    b->d = src_d[src_d_offset];

    // collect transposed quantization parts for a block
    uint src_q_offset = i02 * ne00_blk * ne01 * 4 + i00 * ne01 * 4 + i01;
    uint4 q_block;
    q_block.x = src_q[src_q_offset];
    q_block.y = src_q[src_q_offset + ne01];
    q_block.z = src_q[src_q_offset + ne01 * 2];
    q_block.w = src_q[src_q_offset + ne01 * 3];

    ushort8 post_block = as_ushort8(q_block);
    ushort8 pre_block = (ushort8)(0);

    uchar * pre_block_ptr = (uchar *)(&pre_block);
    uchar * post_block_ptr = (uchar *)(&post_block);

    for (int i = 0; i < QK4_0 / 4; ++i) {
        uchar x0 = post_block_ptr[i + 0];
        uchar x1 = post_block_ptr[i + QK4_0 / 4];

        pre_block_ptr[2 * i + 0] = convert_uchar(x0 & 0x0F) | convert_uchar((x1 & 0x0F) << 4);
        pre_block_ptr[2 * i + 1] = convert_uchar((x0 & 0xF0) >> 4) | convert_uchar(x1 & 0xF0);
    }

    ((__global ushort8 *)(&(b->qs[0])))[0] = pre_block;
}

//------------------------------------------------------------------------------
// block_mxfp4
//------------------------------------------------------------------------------
#define QK_MXFP4 32
struct block_mxfp4 {
    uchar e; // E8M0
    uchar qs[QK_MXFP4 / 2];
};

//------------------------------------------------------------------------------
// kernel_convert_block_mxfp4
// Convert the block_mxfp4 format to 2 separate arrays (AOS -> SOA).
// This kernel does not deshuffle the bits.
//------------------------------------------------------------------------------
kernel void kernel_convert_block_mxfp4(
    global struct block_mxfp4 * src0,
    global uchar * dst_q,
    global uchar * dst_e
) {
    global struct block_mxfp4 * b = (global struct block_mxfp4 *) src0 + get_global_id(0);
    global uchar * q = (global uchar *) dst_q + QK_MXFP4 / 2 * get_global_id(0);
    global uchar * e = (global uchar *) dst_e + get_global_id(0);

    *e = b->e;

    for (int i = 0; i < QK_MXFP4 / 2; ++i) {
        q[i] = b->qs[i];
    }
}

kernel void kernel_convert_block_mxfp4_trans(
    global struct block_mxfp4 * src0,
    __global uint4 * dst_q,
    __global uchar * dst_e,
    uint ne00,
    uint ne01
) {
    int i00 = get_global_id(1);
    uint i01 = get_global_id(0);
    uint i02 = get_global_id(2);

    uint ne00_blk = ne00 / QK_MXFP4;
    uint src_blk_offset = i00 + i01 * ne00_blk + i02 * ne00_blk * ne01;
    uint dst_blk_offset = i01 + i00 * ne01 + i02 * ne00_blk * ne01;

    global struct block_mxfp4 * b = src0 + src_blk_offset;

    dst_q[dst_blk_offset] = ((global uint4 *)(&(b->qs[0])))[0];
    dst_e[dst_blk_offset] = b->e;
}

kernel void kernel_restore_block_mxfp4(
    global uchar * src_q,
    global half  * src_e,
    global struct block_mxfp4 * dst
) {
    global struct block_mxfp4 * b = (global struct block_mxfp4 *) dst + get_global_id(0);
    global uchar * q = (global uchar *) src_q + QK_MXFP4 / 2 * get_global_id(0);
    global uchar * e = (global uchar *) src_e + get_global_id(0);

    b->e = *e;
    for (int i = 0; i < QK_MXFP4 / 2; ++i) {
        b->qs[i] = q[i];
    }
}

kernel void kernel_restore_block_mxfp4_trans(
    __global uint4 * src_q,
    __global uchar * src_e,
    global struct block_mxfp4 * dst,
    uint ne00,
    uint ne01
) {
    int i00 = get_global_id(1);
    uint i01 = get_global_id(0);
    uint i02 = get_global_id(2);

    uint ne00_blk = ne00 / QK_MXFP4;
    uint src_blk_offset = i01 + i00 * ne01 + i02 * ne00_blk * ne01;
    uint dst_blk_offset = i00 + i01 * ne00_blk + i02 * ne00_blk * ne01;

    global struct block_mxfp4 * b = dst + dst_blk_offset;

    ((global uint4 *)(&(b->qs[0])))[0] = src_q[src_blk_offset];
    b->e = src_e[src_blk_offset];
}

kernel void kernel_convert_block_mxfp4_trans4_ns(
    global struct block_mxfp4 * src0,
    __global uint * dst_q,
    __global uchar * dst_e,
    uint ne00,
    uint ne01
) {
    uint i00 = get_global_id(1);
    uint i01 = get_global_id(0);
    uint i02 = get_global_id(2);

    if (i01 >= ne01) {
        return;
    }

    uint ne00_blk = ne00 / QK_MXFP4;
    uint src_blk_offset = i00 + i01 * ne00_blk + i02 * ne00_blk * ne01;
    uint dst_blk_offset = i01 + i00 * ne01 + i02 * ne00_blk * ne01;

    global struct block_mxfp4 * b = src0 + src_blk_offset;
    dst_e[dst_blk_offset] = b->e;

    // extract quantization and unshuffle
    ushort8 pre_block = ((global ushort8 *)(&(b->qs[0])))[0];

    ushort8 post_block = (ushort8)(0);

    uchar * pre_block_ptr = (uchar *)(&pre_block);
    uchar * post_block_ptr = (uchar *)(&post_block);

    for (int i = 0; i < QK_MXFP4 / 4; ++i) {
        uchar x0 = pre_block_ptr[2*i + 0];
        uchar x1 = pre_block_ptr[2*i + 1];

        post_block_ptr[i + 0        ] = convert_uchar(x0 & 0x0F) | convert_uchar((x1 & 0x0F) << 4);
        post_block_ptr[i + QK_MXFP4 / 4] = convert_uchar((x0 & 0xF0) >> 4) | convert_uchar(x1 & 0xF0);
    }

    uint4 q_block = as_uint4(post_block);

    uint offset = i02 * ne00_blk * ne01 * 4 + i00 * ne01 * 4 + i01;
    dst_q[offset] = q_block.x;
    dst_q[offset + ne01] = q_block.y;
    dst_q[offset + ne01 * 2] = q_block.z;
    dst_q[offset + ne01 * 3] = q_block.w;
}

kernel void kernel_restore_block_mxfp4_trans4_ns(
    __global uint * src_q,
    __global uchar * src_e,
    __global struct block_mxfp4 * dst0,
    uint ne00,
    uint ne01
) {
    uint i00 = get_global_id(1);
    uint i01 = get_global_id(0);
    uint i02 = get_global_id(2);

    if (i01 >= ne01) {
        return;
    }

    uint ne00_blk = ne00 / QK_MXFP4;
    uint dst_blk_offset = i00 + i01 * ne00_blk + i02 * ne00_blk * ne01;
    uint src_d_offset = i01 + i00 * ne01 + i02 * ne00_blk * ne01;

    __global struct block_mxfp4 * b = dst0 + dst_blk_offset;
    b->e = src_e[src_d_offset];

    // collect transposed quantization parts for a block
    uint src_q_offset = i02 * ne00_blk * ne01 * 4 + i00 * ne01 * 4 + i01;
    uint4 q_block;
    q_block.x = src_q[src_q_offset];
    q_block.y = src_q[src_q_offset + ne01];
    q_block.z = src_q[src_q_offset + ne01 * 2];
    q_block.w = src_q[src_q_offset + ne01 * 3];

    ushort8 post_block = as_ushort8(q_block);
    ushort8 pre_block = (ushort8)(0);

    uchar * pre_block_ptr = (uchar *)(&pre_block);
    uchar * post_block_ptr = (uchar *)(&post_block);

    for (int i = 0; i < QK_MXFP4 / 4; ++i) {
        uchar x0 = post_block_ptr[i + 0];
        uchar x1 = post_block_ptr[i + QK_MXFP4 / 4];

        pre_block_ptr[2 * i + 0] = convert_uchar(x0 & 0x0F) | convert_uchar((x1 & 0x0F) << 4);
        pre_block_ptr[2 * i + 1] = convert_uchar((x0 & 0xF0) >> 4) | convert_uchar(x1 & 0xF0);
    }

    ((__global ushort8 *)(&(b->qs[0])))[0] = pre_block;
}

//------------------------------------------------------------------------------
// block_q8_0
//------------------------------------------------------------------------------
typedef struct {
    half d;       // delta
    char qs[QK8_0]; // quants
} block_q8_0;

kernel void kernel_convert_block_q8_0(
    global block_q8_0 * src0,
    global uchar * dst_q,
    global half  * dst_d
) {
    global block_q8_0 * b = (global block_q8_0 *) src0 + get_global_id(0);
    global uchar      * q = (global uchar *) dst_q + QK8_0*get_global_id(0);
    global half       * d = (global half *) dst_d + get_global_id(0);

    *d = b->d;

    for (int i = 0; i < QK8_0; ++i) {
        q[i] = b->qs[i];
    }
}

kernel void kernel_restore_block_q8_0(
    global uchar * src_q,
    global half  * src_d,
    global block_q8_0 * dst
) {
    global block_q8_0 * b = (global block_q8_0 *) dst + get_global_id(0);
    global uchar      * q = (global uchar *) src_q + QK8_0*get_global_id(0);
    global half       * d = (global half *) src_d + get_global_id(0);

    b->d = *d;
    for (int i = 0; i < QK8_0; ++i) {
        b->qs[i] = q[i];
    }
}

//------------------------------------------------------------------------------
// View-aware AoS quant -> f16 dequant kernels for the asymmetric KV-cache flash
// attention fallback. They dequantise a (possibly strided-view) q8_0 / q4_0
// tensor into a tight-packed f16 buffer, honouring the source view offset and
// row/slice strides, so the f32_f16 (mixed) FA kernel can read a quant KV side.
//------------------------------------------------------------------------------
kernel void kernel_dequant_q8_0_f16_view_aos(
    global char * src,
    ulong         src_offset,
    ulong         src_nb1,
    ulong         src_nb2,
    ulong         src_nb3,
    int           nblk0,
    int           ne1,
    int           ne2,
    int           ne3,
    global half * dst
) {
    int blk_i0 = get_global_id(0);
    int i1     = get_global_id(1);
    int batch  = get_global_id(2);

    if (blk_i0 >= nblk0) return;
    if (i1     >= ne1)   return;

    int i2 = batch % ne2;
    int i3 = batch / ne2;
    if (i3 >= ne3) return;

    global char * block = src + src_offset + (ulong)i3*src_nb3 + (ulong)i2*src_nb2 + (ulong)i1*src_nb1 + (ulong)blk_i0 * (2 + QK8_0);
    float d = vload_half(0, (global half *)block);
    global char * qs = block + 2;

    ulong dst_row_base = ((ulong)i3 * ne2 * ne1 + (ulong)i2 * ne1 + (ulong)i1) * nblk0;
    global half * out = dst + (dst_row_base + blk_i0) * QK8_0;

    for (int i = 0; i < QK8_0; ++i) {
        out[i] = (half)(d * (float)qs[i]);
    }
}

kernel void kernel_dequant_q4_0_f16_view_aos(
    global char * src,
    ulong         src_offset,
    ulong         src_nb1,
    ulong         src_nb2,
    ulong         src_nb3,
    int           nblk0,
    int           ne1,
    int           ne2,
    int           ne3,
    global half * dst
) {
    int blk_i0 = get_global_id(0);
    int i1     = get_global_id(1);
    int batch  = get_global_id(2);

    if (blk_i0 >= nblk0) return;
    if (i1     >= ne1)   return;

    int i2 = batch % ne2;
    int i3 = batch / ne2;
    if (i3 >= ne3) return;

    global char * block = src + src_offset + (ulong)i3*src_nb3 + (ulong)i2*src_nb2 + (ulong)i1*src_nb1 + (ulong)blk_i0 * (2 + QK4_0/2);
    float d = vload_half(0, (global half *)block);
    global uchar * qs = (global uchar *)(block + 2);

    ulong dst_row_base = ((ulong)i3 * ne2 * ne1 + (ulong)i2 * ne1 + (ulong)i1) * nblk0;
    global half * out = dst + (dst_row_base + blk_i0) * QK4_0;

    for (int i = 0; i < QK4_0/2; ++i) {
        uchar byte = qs[i];
        int q0 = (int)(byte & 0x0F) - 8;
        int q1 = (int)(byte >> 4)   - 8;
        out[i]           = (half)(d * (float)q0);
        out[i + QK4_0/2] = (half)(d * (float)q1);
    }
}

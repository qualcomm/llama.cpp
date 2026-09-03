#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#else
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

#ifdef cl_khr_subgroup_shuffle
#pragma OPENCL EXTENSION cl_khr_subgroup_shuffle : enable
#define HAS_SUBGROUP_SHUFFLE 1
#elif defined(cl_qcom_subgroup_shuffle)
#pragma OPENCL EXTENSION cl_qcom_subgroup_shuffle : enable
#define HAS_SUBGROUP_SHUFFLE 1
// Adreno compilers that expose only cl_qcom_subgroup_shuffle do not declare
// the KHR name; route it to the qcom builtin.
#define sub_group_shuffle_xor(val, mask) qcom_sub_group_shuffle_xor((val), (mask), CLK_SUB_GROUP_SHUFFLE_WIDTH_WAVE_SIZE_QCOM, 0.0f)
#endif

// Skinny-N f16 x f32 batched matmul for the KQ / KQV of a speculative verify
// batch after the GQA fold: N = verify width * gqa (12..48 on Qwen3.8), M and
// K the context or the head dim. The 64x64 LDS tile of mul_mm_f16_f32_l4_lm
// stages both operands through local memory with scalar loads and runs at a
// fraction of the bus on that shape.
//
// Here a cluster of KL lanes shares one src0 row and splits its K range: lane
// l of the cluster reads halfs [8l, 8l+8) of every KC = 8*KL deep K step, so
// one cluster load instruction covers 16*KL contiguous bytes of the row and
// each lane keeps TR*TC partial sums that are folded across the cluster at
// the end (subgroup shuffles, or local memory where they are missing). Only
// src1, reused by every row, is staged in local memory, as half: the kernel
// is bound by local memory reads, staging as float costs the whole gain, and
// the CPU reference rounds src1 to f16 for an f16 matmul too.
//
// Workgroup: ceil(ne11 / TC) column groups of CR clusters x KL lanes; each
// cluster owns TR consecutive rows, so a workgroup covers CR*TR rows. The
// host sizes the workgroup from ne11 and passes the tile as -D constants.
// Split-K as in mul_mm_f16_f32_l4_lm: the grid's third dimension is nsplit
// copies of the batch range, and the partial sums are folded by
// kernel_mul_mm_f16_f32_l4_lm_splitk_reduce.
//
// Host contract: ne00 % 8 == 0, kslice % 8 == 0, offset1 16-byte aligned,
// stride_b % 4 == 0 (src1 is read as float4), ne11 <= BN_MAX, KL a power of
// two no wider than the subgroup.

#ifndef TR
#define TR 2
#endif
#ifndef TC
#define TC 16
#endif
#ifndef KL
#define KL 4
#endif
#ifndef CR
#define CR 16
#endif
#ifndef BN_MAX
#define BN_MAX 64
#endif
#define KC (8 * KL)
// The last column group may run past ne11 (its columns are masked at the
// store); the staging buffer covers the whole group so those reads stay in
// bounds.
#define BN_PAD (((BN_MAX + TC - 1) / TC) * TC)
#define NTH_MAX ((BN_PAD / TC) * CR * KL)

kernel void kernel_mul_mm_f16_f32_skinny(
    global const half  * src0,
    ulong offset0,
    global const float * src1,
    ulong offset1,
    global float * dst,
    ulong offsetd,

    int ne00,
    int ne01,
    int ne02,
    int ne11,
    int ne12,

    int stride_a,
    int stride_b,
    int stride_d,

    int batch_stride_a,
    int batch_stride_b,
    int batch_stride_d,

    int r2,
    int r3,

    int nsplit,
    int kslice
) {
    src0 = (global const half  *)((global const char *)src0 + offset0);
    src1 = (global const float *)((global const char *)src1 + offset1);
    dst  = (global       float *)((global       char *)dst  + offsetd);

    local half buf_b[BN_PAD * KC];
#ifndef HAS_SUBGROUP_SHUFFLE
    local float red[NTH_MAX * TC];
#endif

    const int nbatch    = get_global_size(2) / nsplit;
    const int ks        = get_global_id(2) / nbatch;
    const int batch_idx = get_global_id(2) - ks * nbatch;

    const int k_begin = ks * kslice;
    const int k_end   = min(k_begin + kslice, ne00);

    const int i13 = batch_idx / ne12;
    const int i12 = batch_idx % ne12;

    const int i03 = i13 / r3;
    const int i02 = i12 / r2;

    const int batch_idx_a = i03 * ne02 + i02;

    const int tid  = get_local_id(0);
    const int nth  = get_local_size(0);
    const int lane = tid % KL;          // K phase inside the cluster
    const int cl   = (tid / KL) % CR;   // cluster inside the column group
    const int cg   = tid / (KL * CR);   // column group

    const int row0 = get_group_id(0) * (CR * TR) + cl * TR;
    const int col0 = cg * TC;

    // Rows past ne01 are clamped onto a valid row and masked at the store, so
    // the load path stays branch-free.
    global const half * a_row[TR];
    for (int j = 0; j < TR; j++) {
        const int row = min(row0 + j, ne01 - 1);
        a_row[j] = src0 + batch_idx_a * batch_stride_a + row * stride_a + lane * 8;
    }

    global const float * b_base = src1 + batch_idx * batch_stride_b;

    float acc[TR][TC];
    for (int j = 0; j < TR; j++) {
        for (int c = 0; c < TC; c++) {
            acc[j][c] = 0.0f;
        }
    }

    for (int k0 = k_begin; k0 < k_end; k0 += KC) {
        // Stage src1[n][k0 .. k0+KC) as half. The k tail past k_end is never
        // read because the compute lanes for it are masked.
        for (int i = tid; i < ne11 * (KC / 4); i += nth) {
            const int n  = i / (KC / 4);
            const int k4 = i - n * (KC / 4);
            const int k  = k0 + k4 * 4;
            float4 v = (float4)(0.0f);
            if (k < k_end) {
                v = vload4(0, b_base + n * stride_b + k);
            }
            vstore_half4_rte(v, 0, buf_b + n * KC + k4 * 4);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (k0 + lane * 8 < k_end) {
            float8 a[TR];
            for (int j = 0; j < TR; j++) {
                a[j] = convert_float8(vload8(0, a_row[j] + k0));
            }
            const local half * bp = buf_b + col0 * KC + lane * 8;
            for (int c = 0; c < TC; c++) {
                const float8 b = vload_half8(0, bp + c * KC);
                for (int j = 0; j < TR; j++) {
                    float s = acc[j][c];
                    s = mad(a[j].s0, b.s0, s);
                    s = mad(a[j].s1, b.s1, s);
                    s = mad(a[j].s2, b.s2, s);
                    s = mad(a[j].s3, b.s3, s);
                    s = mad(a[j].s4, b.s4, s);
                    s = mad(a[j].s5, b.s5, s);
                    s = mad(a[j].s6, b.s6, s);
                    s = mad(a[j].s7, b.s7, s);
                    acc[j][c] = s;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const int offsets = (ks * nbatch + batch_idx) * batch_stride_d;

#ifdef HAS_SUBGROUP_SHUFFLE
    // Butterfly over the cluster: every lane ends with the full sums, and
    // lane l stores the columns c = l, l + KL, ... so the store is spread
    // across the cluster.
    for (int j = 0; j < TR; j++) {
        for (int c = 0; c < TC; c++) {
            float s = acc[j][c];
            for (int m = 1; m < KL; m <<= 1) {
                s += sub_group_shuffle_xor(s, m);
            }
            acc[j][c] = s;
        }
    }
    for (int j = 0; j < TR; j++) {
        const int row = row0 + j;
        if (row >= ne01) {
            break;
        }
        for (int c = lane; c < TC; c += KL) {
            const int col = col0 + c;
            if (col < ne11) {
                dst[offsets + col * stride_d + row] = acc[j][c];
            }
        }
    }
#else
    for (int j = 0; j < TR; j++) {
        for (int c = 0; c < TC; c++) {
            red[tid * TC + c] = acc[j][c];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        const int row = row0 + j;
        if (row < ne01) {
            const int base = (tid - lane) * TC;
            for (int c = lane; c < TC; c += KL) {
                float s = 0.0f;
                for (int l = 0; l < KL; l++) {
                    s += red[base + l * TC + c];
                }
                const int col = col0 + c;
                if (col < ne11) {
                    dst[offsets + col * stride_d + row] = s;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
#endif
}

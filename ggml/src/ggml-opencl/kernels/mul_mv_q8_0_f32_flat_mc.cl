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

//------------------------------------------------------------------------------
// kernel_mul_mv_q8_0_f32_flat_mc
//
// Multi-column twin of kernel_mul_mv_q8_0_f32_flat (same arguments, same grid
// along x). The single-column kernel takes the output column from
// get_group_id(1), so a matmul with ne1 columns re-streams the whole weight once
// per column. The flat q8_0 layout is what a vocab-scale q8_0 lm_head stays in
// (it is past the transposed layout's element limit), so a 4-token speculative
// verify reads gpt-oss-20b's 0.615 GB head four times.
//
// This variant reads each weight block once and dots it against Q8_MC_N_COLS
// activation columns, so a group of get_group_id(1) now covers N_COLS columns.
// The per-(row, column) arithmetic is the single-column kernel's: same 8-quant
// lane split, same sequential product sum, same scale multiply and subgroup
// reduce. Rows, columns and their accumulators are named scalars / vectors
// rather than arrays on purpose: indexed arrays here put the accumulator block
// in local memory (32 KB of LDS, 296-440 B private per work-item) and ran 2-3x
// slower than the single-column kernel.
//------------------------------------------------------------------------------

#define QK8_0 32
#define NB_Q8_0 8

#ifndef Q8_MC_N_COLS
#define Q8_MC_N_COLS 4
#endif
#define N_COLS Q8_MC_N_COLS
#if N_COLS != 2 && N_COLS != 4
#error "Q8_MC_N_COLS must be 2 or 4"
#endif

#ifdef INTEL_GPU
#define N_R0_Q8_0 4 // number of rows each subgroup works on
#define N_SG_Q8_0 2 // number of subgroups in a work group
#define N_SIMDWIDTH 16 // subgroup size
#elif defined (ADRENO_GPU)
#define N_R0_Q8_0 4
#define N_SG_Q8_0 2
#define N_SIMDWIDTH 64
#endif

inline float q8_0_dot8(float8 qv, float8 yl) {
    float sumq = 0;
    sumq += qv.s0*yl.s0;
    sumq += qv.s1*yl.s1;
    sumq += qv.s2*yl.s2;
    sumq += qv.s3*yl.s3;
    sumq += qv.s4*yl.s4;
    sumq += qv.s5*yl.s5;
    sumq += qv.s6*yl.s6;
    sumq += qv.s7*yl.s7;
    return sumq;
}

// One row against the column tile: s.s<c> += dot(row quants, column c) * d.
#if N_COLS == 4
#define Q8_MC_ROW(s, ax, ad)                                                  \
    {                                                                         \
        const float8 qv = convert_float8(vload8(0, ax + ib*QK8_0 + il*NB_Q8_0)); \
        const float  d  = ad[ib];                                             \
        s.s0 += q8_0_dot8(qv, y0)*d;                                          \
        s.s1 += q8_0_dot8(qv, y1)*d;                                          \
        s.s2 += q8_0_dot8(qv, y2)*d;                                          \
        s.s3 += q8_0_dot8(qv, y3)*d;                                          \
    }
#else
#define Q8_MC_ROW(s, ax, ad)                                                  \
    {                                                                         \
        const float8 qv = convert_float8(vload8(0, ax + ib*QK8_0 + il*NB_Q8_0)); \
        const float  d  = ad[ib];                                             \
        s.s0 += q8_0_dot8(qv, y0)*d;                                          \
        s.s1 += q8_0_dot8(qv, y1)*d;                                          \
    }
#endif

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_16
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mv_q8_0_f32_flat_mc(
    global char * src0_q,
    global half * src0_d,
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
    src1 = (global char*)((global char*)src1 + offset1);
    dst  = (global char*)((global char*)dst  + offsetd);

    int nb = ne00/QK8_0;

    int r0 = get_group_id(0);
    int r1 = get_group_id(1);
    int im = get_group_id(2);

    int first_row = (r0*N_SG_Q8_0 + get_sub_group_id()) * N_R0_Q8_0;
    // Same tail handling as the single-column kernel: slide the last window back
    // inside src0; the rows it repeats are stored identically by their owner.
    first_row = min(first_row, max(ne01 - N_R0_Q8_0, 0));

    const int first_col = r1*N_COLS;

    uint i12 = im%ne12;
    uint i13 = im/ne12;

    const short ix = get_sub_group_local_id()/4;
    const short il = get_sub_group_local_id()%4;

    // A tail tile (ne1 not a multiple of N_COLS) reads a clamped, in-range column
    // for its dead lanes and the store drops it.
    global char * y_base = src1 + i12*nb12 + i13*nb13 + (ulong)(ix*QK8_0 + il*NB_Q8_0)*sizeof(float);
    global float * yb0 = (global float *) (y_base + (ulong)min(first_col + 0, ne1 - 1)*nb11);
    global float * yb1 = (global float *) (y_base + (ulong)min(first_col + 1, ne1 - 1)*nb11);
#if N_COLS == 4
    global float * yb2 = (global float *) (y_base + (ulong)min(first_col + 2, ne1 - 1)*nb11);
    global float * yb3 = (global float *) (y_base + (ulong)min(first_col + 3, ne1 - 1)*nb11);
#endif

    // ulong: a vocab-scale head is past 4 GB of row offsets on the widest models.
    const ulong blk0 = ((ulong)first_row*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03)/34;
    const ulong nbr  = nb01/34;   // blocks per row

    global char * ax0 = src0_q + (blk0 + 0*nbr)*QK8_0;
    global char * ax1 = src0_q + (blk0 + 1*nbr)*QK8_0;
    global char * ax2 = src0_q + (blk0 + 2*nbr)*QK8_0;
    global char * ax3 = src0_q + (blk0 + 3*nbr)*QK8_0;
    global half * ad0 = src0_d + blk0 + 0*nbr;
    global half * ad1 = src0_d + blk0 + 1*nbr;
    global half * ad2 = src0_d + blk0 + 2*nbr;
    global half * ad3 = src0_d + blk0 + 3*nbr;

    // s<r>.s<c>: row r, column c
#if N_COLS == 4
    float4 s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
#else
    float2 s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
#endif

    // each thread handles NB_Q8_0 quants of one block at a time
    for (int ib = ix; ib < nb; ib += N_SIMDWIDTH/4) {
        const float8 y0 = vload8(0, yb0);
        const float8 y1 = vload8(0, yb1);
#if N_COLS == 4
        const float8 y2 = vload8(0, yb2);
        const float8 y3 = vload8(0, yb3);
#endif
        Q8_MC_ROW(s0, ax0, ad0)
        Q8_MC_ROW(s1, ax1, ad1)
        Q8_MC_ROW(s2, ax2, ad2)
        Q8_MC_ROW(s3, ax3, ad3)

        yb0 += N_SIMDWIDTH*NB_Q8_0;
        yb1 += N_SIMDWIDTH*NB_Q8_0;
#if N_COLS == 4
        yb2 += N_SIMDWIDTH*NB_Q8_0;
        yb3 += N_SIMDWIDTH*NB_Q8_0;
#endif
    }

    global float * dst_f32 = (global float *) dst + (ulong)im*ne0*ne1 + (ulong)first_col*ne0 + first_row;

#define Q8_MC_STORE(c, cc)                                                    \
    {                                                                         \
        const float4 tot = (float4)(sub_group_reduce_add(s0.cc), sub_group_reduce_add(s1.cc), \
                                    sub_group_reduce_add(s2.cc), sub_group_reduce_add(s3.cc)); \
        if (get_sub_group_local_id() == 0 && first_col + c < ne1) {           \
            global float * o = dst_f32 + (ulong)c*ne0;                        \
            if (first_row + 0 < ne01) { o[0] = tot.s0; }                      \
            if (first_row + 1 < ne01) { o[1] = tot.s1; }                      \
            if (first_row + 2 < ne01) { o[2] = tot.s2; }                      \
            if (first_row + 3 < ne01) { o[3] = tot.s3; }                      \
        }                                                                     \
    }

    Q8_MC_STORE(0, s0)
    Q8_MC_STORE(1, s1)
#if N_COLS == 4
    Q8_MC_STORE(2, s2)
    Q8_MC_STORE(3, s3)
#endif
}

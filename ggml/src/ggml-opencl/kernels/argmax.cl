#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// One work-group per row of input (i1 = group_id(0)).
// Each thread reduces a chunk of ne00 into a (max, idx); tree reduction
// across the WG yields the row's argmax. Output is i32 per row.

kernel void kernel_argmax_f32_i32(
    global const float * src0,
    ulong                offset0,
    global       int   * dst,
    ulong                offsetd,
    int                  ne00,
    ulong                nb01,
    local        float * sval,
    local        int   * sidx
) {
    src0 = (global const float *)((global const char *) src0 + offset0);
    dst  = (global       int   *)((global       char *) dst  + offsetd);

    const int row  = get_group_id(0);
    const int lid  = get_local_id(0);
    const int lsz  = get_local_size(0);

    global const float * row_ptr = (global const float *)((global const char *) src0 + (ulong)row * nb01);

    float local_max = -INFINITY;
    int   local_idx = 0;
    for (int i = lid; i < ne00; i += lsz) {
        const float v = row_ptr[i];
        if (v > local_max) {
            local_max = v;
            local_idx = i;
        }
    }

    sval[lid] = local_max;
    sidx[lid] = local_idx;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = lsz / 2; s > 0; s >>= 1) {
        if (lid < s) {
            const float a = sval[lid];
            const float b = sval[lid + s];
            if (b > a) {
                sval[lid] = b;
                sidx[lid] = sidx[lid + s];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        dst[row] = sidx[0];
    }
}

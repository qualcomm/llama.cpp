#pragma OPENCL EXTENSION cl_khr_fp16 : enable

inline int wrap_index(int i, int ne) {
    if (i < 0)        return i + ne;
    else if (i >= ne) return i - ne;
    return i;
}

// One WG per (i1, i2, i3) of output; threads stride over i0.
// Tensors assumed contiguous (matches CPU roll which uses sizeof(float) strides).
kernel void kernel_roll_f32(
    global const float * src0,
    ulong                offset0,
    global       float * dst,
    ulong                offsetd,
    int                  ne00, int ne01, int ne02, int ne03,
    int                  s0, int s1, int s2, int s3
) {
    src0 = (global const float *)((global const char *) src0 + offset0);
    dst  = (global       float *)((global       char *) dst  + offsetd);

    const int i1 = get_group_id(0);
    const int i2 = get_group_id(1);
    const int i3 = get_group_id(2);

    const int j1 = wrap_index(i1 - s1, ne01);
    const int j2 = wrap_index(i2 - s2, ne02);
    const int j3 = wrap_index(i3 - s3, ne03);

    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);

    for (int i0 = lid; i0 < ne00; i0 += lsz) {
        const int j0 = wrap_index(i0 - s0, ne00);

        const ulong src_idx = (ulong)j3*ne02*ne01*ne00 + (ulong)j2*ne01*ne00 + (ulong)j1*ne00 + (ulong)j0;
        const ulong dst_idx = (ulong)i3*ne02*ne01*ne00 + (ulong)i2*ne01*ne00 + (ulong)i1*ne00 + (ulong)i0;

        dst[dst_idx] = src0[src_idx];
    }
}

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

kernel void kernel_arange_f32(
    global float * dst,
    ulong          offsetd,
    int            n,
    float          start,
    float          step
) {
    dst = (global float *)((global char *) dst + offsetd);
    const int i = get_global_id(0);
    if (i >= n) return;
    dst[i] = start + step * (float)i;
}

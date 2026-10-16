#pragma OPENCL EXTENSION cl_khr_fp16 : enable

//------------------------------------------------------------------------------
// fill
//------------------------------------------------------------------------------
__kernel void kernel_fill_f32(
        __global float *dst,
        ulong offsetd,
        float v,
        int n

) {
    dst = (global float*)((global char*)dst + offsetd);
    if(get_global_id(0) < n){
        dst[get_global_id(0)] = v;
    }
}

__kernel void kernel_fill_f16(
        __global half *dst,
        ulong offsetd,
        float v,
        int n
) {
    dst = (global half*)((global char*)dst + offsetd);
    if (get_global_id(0) < n) {
        vstore_half(v, get_global_id(0), dst);
    }
}

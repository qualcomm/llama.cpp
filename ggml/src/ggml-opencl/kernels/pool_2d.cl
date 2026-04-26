#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Match ggml_op_pool enum order in ggml.h
#define POOL_MAX 0
#define POOL_AVG 1

// One work-item per output element of [W_out, H_out, C, N].
kernel void kernel_pool_2d_f32(
    global const float * src,
    ulong                offset_s,
    global       float * dst,
    ulong                offset_d,
    int                  ne00, int ne01, int ne02, int ne03,
    int                  ne0,  int ne1,
    ulong                nb01, ulong nb02, ulong nb03,
    int                  op_type,
    int                  k0, int k1,
    int                  s0, int s1,
    int                  p0, int p1
) {
    src = (global const float *)((global const char *) src + offset_s);
    dst = (global       float *)((global       char *) dst + offset_d);

    const int total = ne0 * ne1 * ne02 * ne03;
    const int gid = get_global_id(0);
    if (gid >= total) return;

    const int ox = gid % ne0;
    const int oy = (gid / ne0) % ne1;
    const int ic = (gid / (ne0 * ne1)) % ne02;
    const int in_ = gid / (ne0 * ne1 * ne02);

    const int ka = k0 * k1;
    const int ix0 = ox * s0 - p0;
    const int iy0 = oy * s1 - p1;

    float res = (op_type == POOL_MAX) ? -INFINITY : 0.0f;

    global const char * src_plane = (global const char *) src + (ulong)in_*nb03 + (ulong)ic*nb02;

    for (int ky = 0; ky < k1; ky++) {
        const int iy = iy0 + ky;
        if (iy < 0 || iy >= ne01) continue;
        global const float * src_row = (global const float *)(src_plane + (ulong)iy*nb01);
        for (int kx = 0; kx < k0; kx++) {
            const int ix = ix0 + kx;
            if (ix < 0 || ix >= ne00) continue;
            const float v = src_row[ix];
            if (op_type == POOL_MAX) {
                res = fmax(res, v);
            } else {
                res += v;
            }
        }
    }

    if (op_type == POOL_AVG) {
        res /= (float)ka;
    }

    dst[gid] = res;
}

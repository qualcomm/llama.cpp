// Fast Walsh-Hadamard transform for MUL_MAT nodes tagged GGML_HINT_SRC0_IS_HADAMARD.
//
// llama.cpp rotates Q, K and V (and un-rotates the attention output) by an orthonormal
// Walsh-Hadamard matrix when the KV cache is quantized. As a MUL_MAT that is a dense
// n x n f32 product per row; the transform itself is n*log2(n) adds. src0 (the matrix) is
// not read: each row of src1 becomes dst = H_n * row / sqrt(n), H_n in Sylvester (natural)
// order, the same result the CPU, CUDA, Metal, SYCL and Vulkan backends produce.
//
// One 64-lane workgroup per row; the row is staged in local memory and transformed in place
// in log2(n) butterfly stages. n is 64, 128, 256 or 512 (the host declines anything else).

#define FWHT_WG  64
#define FWHT_MAX 512

kernel void kernel_fwht_f32(
        global const char * src, ulong off_src,
        global       char * dst, ulong off_dst,
        int n, float scale
) {
    local float buf[FWHT_MAX];

    const int   lid = get_local_id(0);
    const ulong row = get_group_id(0);

    global const float * s = (global const float *)(src + off_src) + row*n;
    global       float * d = (global       float *)(dst + off_dst) + row*n;

    for (int i = lid; i < n; i += FWHT_WG) {
        buf[i] = s[i]*scale;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int h = 1; h < n; h <<= 1) {
        for (int t = lid; t < n/2; t += FWHT_WG) {
            const int i = (t/h)*2*h + (t%h);   // lower element of the pair (i, i + h)
            const float a = buf[i];
            const float b = buf[i + h];
            buf[i]     = a + b;
            buf[i + h] = a - b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int i = lid; i < n; i += FWHT_WG) {
        d[i] = buf[i];
    }
}

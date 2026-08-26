#pragma OPENCL EXTENSION cl_khr_fp16 : enable

//------------------------------------------------------------------------------
// Fused sigmoid + mul: dst = a * sigmoid(b)
//
// Gated-attention blocks (muse-glimmer, afmoe, qwen3.5) emit a SIGMOID over the
// gate projection immediately followed by a MUL against the attention output.
// Both operands are tiny (n_embd_head_k * n_head), so the pair costs two full
// dispatches to move almost no data -- pure launch overhead on Adreno.
//
// Kept scalar and in the exact expression form of kernel_sigmoid_f32 followed by
// a plain float multiply, so the result is bit-identical to the per-op path.
// (kernel_sigmoid_f32 has no float4 variant, so there is no scalar-vs-vector
// transcendental mismatch to worry about here.)
//------------------------------------------------------------------------------
kernel void kernel_sigmoid_mul_f32(
        global float * a,
        ulong offseta,
        global float * b,
        ulong offsetb,
        global float * dst,
        ulong offsetd
) {
    a   = (global float*)((global char*)a   + offseta);
    b   = (global float*)((global char*)b   + offsetb);
    dst = (global float*)((global char*)dst + offsetd);

    const int i = get_global_id(0);

    dst[i] = a[i] * (1.0f / (1.0f + exp(-b[i])));
}

//------------------------------------------------------------------------------
// Fused silu + mul: dst = a * silu(b), silu(b) = b / (1 + exp(-b))
//
// Gated delta net blocks (qwen3.5's build_norm_gated: RMSNorm(x) then
// silu(gate) then mul) emit the same launch-bound tiny-tensor SILU+MUL pair
// as the sigmoid+mul case above, just with the other activation. Kept in the
// exact expression form of kernel_silu followed by a plain float multiply,
// so the result is bit-identical to the per-op path.
//------------------------------------------------------------------------------
kernel void kernel_silu_mul_f32(
        global float * a,
        ulong offseta,
        global float * b,
        ulong offsetb,
        global float * dst,
        ulong offsetd
) {
    a   = (global float*)((global char*)a   + offseta);
    b   = (global float*)((global char*)b   + offsetb);
    dst = (global float*)((global char*)dst + offsetd);

    const int i = get_global_id(0);

    const float x = b[i];
    dst[i] = a[i] * (x / (1.0f + exp(-x)));
}

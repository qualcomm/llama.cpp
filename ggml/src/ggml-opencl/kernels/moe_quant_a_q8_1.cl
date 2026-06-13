#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Quantize the reordered MoE activation tiles (row-major [token_slot, ne00]
// f32) into q8_1 blocks of 32: int8 quants + per-block scale d + per-block
// sum s (= d * Sum(qs)). Consumed by kernel_gemm_moe_q4_k_q8_1_dp4a so the
// GEMM inner loop can use the qcom int8 dp4a (dot_acc_sat_4x8packed_ss_int).
//
// One work-item per 32-element block. Padded/zero token slots (produced by the
// router reorder for unfilled tile lanes) quantize to d=0,s=0,qs=0 and so
// contribute nothing to the GEMM, matching the f32 path.
__kernel void kernel_moe_quant_a_q8_1(
        __global const float * src,   // [n_tok_slots * ne00]
        __global       char  * qa,    // [n_tok_slots * ne00]
        __global       half  * da,    // [n_tok_slots * (ne00/32)]
        __global       half  * sa,    // [n_tok_slots * (ne00/32)]
        int total_blocks              // n_tok_slots * (ne00/32)
) {
    const int blk = get_global_id(0);
    if (blk >= total_blocks) {
        return;
    }

    const int base = blk * 32;

    float v[32];
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        v[i] = src[base + i];
        amax = fmax(amax, fabs(v[i]));
    }

    const float d  = amax / 127.0f;
    const float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;

    int sum = 0;
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        const int q = (int)rint(v[i] * id);
        qa[base + i] = (char)q;
        sum += q;
    }

    da[blk] = (half)d;
    sa[blk] = (half)(d * (float)sum);
}

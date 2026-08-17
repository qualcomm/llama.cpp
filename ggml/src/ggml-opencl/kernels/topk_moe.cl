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

// Bounds are enforced host-side; they only size the local scratch.
#define TOPK_MOE_MAX_EXPERT 512
#define TOPK_MOE_MAX_K       64

// Fused MoE router top-k + late softmax.
//
// Replaces {ARGSORT, VIEW, GET_ROWS, RESHAPE, SOFT_MAX, RESHAPE} — the
// SOFTMAX_WEIGHT gating tail in build_moe_ffn — with a single dispatch.
// One workgroup per token, one wave wide: the workgroup selects the k largest
// router logits, then softmaxes exactly those k values.
//
// The softmax stage is a transcription of kernel_soft_max_4 (softmax_4_f32.cl)
// with psrc4 replaced by the k selected values: same lane->element mapping,
// same float4 accumulation order, same sub_group reductions, and lanes past the
// data contribute the same identities they do there (-INFINITY to the max, 0 to
// the sum). The result is therefore bit-identical to the unfused path. That
// holds only for k % 4 == 0 at the wave width the standalone kernel runs at;
// the host declines the fusion otherwise.
//
// Selection differs from the unfused path in one respect: ggml_argsort_top_k
// takes the prefix of a full bitonic descending sort, which has no defined
// tie order, whereas this kernel resolves ties to the lowest expert index.
// Exactly equal router logits therefore may select a different expert here.
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_topk_moe_late_softmax(
        global char * logits,
        ulong         offset_l,
        global char * ids,
        ulong         offset_i,
        global char * weights,
        ulong         offset_w,
        int           n_expert,
        int           k,
        uint          l1,        // logits  row stride, floats
        uint          i1,        // ids     row stride, ints
        uint          w1         // weights row stride, floats
) {
    local float  lv[TOPK_MOE_MAX_EXPERT];   // working copy of the row
    local int    sel[TOPK_MOE_MAX_K];       // selected expert indices
    local float4 sv4[TOPK_MOE_MAX_K / 4];   // the selected logits
    local float * sv = (local float *)sv4;
    local int    lidx;                      // arg-max index for the current round

    const int t   = get_group_id(0);
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);

    global const float * lg  = (global const float *)(logits + offset_l) + (size_t)t * l1;
    global int         * pid = (global int   *)(ids     + offset_i) + (size_t)t * i1;
    global float       * pw  = (global float *)(weights + offset_w) + (size_t)t * w1;

    for (int e = lid; e < n_expert; e += lsz) {
        lv[e] = lg[e];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // k rounds of arg-max. Ties resolve to the lowest expert index: the
    // per-lane scan keeps the first of equal values (strict >), and the
    // cross-lane pass takes the smallest index among the lanes holding the max.
    // The workgroup is exactly one wave, so sub_group_reduce_max is workgroup-wide.
    for (int r = 0; r < k; ++r) {
        if (lid == 0) {
            lidx = INT_MAX;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        float best = -INFINITY;
        int   bidx = INT_MAX;

        for (int e = lid; e < n_expert; e += lsz) {
            const float v = lv[e];
            if (v > best) {
                best = v;
                bidx = e;
            }
        }

        const float rmax = sub_group_reduce_max(best);
        if (bidx != INT_MAX && best == rmax) {
            atomic_min(&lidx, bidx);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lid == 0) {
            sel[r]   = lidx;
            sv[r]    = rmax;
            lv[lidx] = -INFINITY;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // softmax over the k selected values — see the note above on bit-exactness
    const int k4 = k / 4;

    float4 lmax4 = -INFINITY;
    for (int j = lid; j < k4; j += lsz) {
        lmax4 = fmax(lmax4, sv4[j]);
    }
    const float lmax = fmax(fmax(lmax4.s0, lmax4.s1), fmax(lmax4.s2, lmax4.s3));
    const float mx   = sub_group_reduce_max(lmax);

    float4 lsum4 = 0.0f;
    for (int j = lid; j < k4; j += lsz) {
        lsum4 += exp(sv4[j] - mx);
    }
    const float lsum = lsum4.s0 + lsum4.s1 + lsum4.s2 + lsum4.s3;
    const float sum  = sub_group_reduce_add(lsum);

    for (int j = lid; j < k4; j += lsz) {
        const float4 e = exp(sv4[j] - mx) / sum;

        pw[j*4 + 0] = e.s0;
        pw[j*4 + 1] = e.s1;
        pw[j*4 + 2] = e.s2;
        pw[j*4 + 3] = e.s3;

        pid[j*4 + 0] = sel[j*4 + 0];
        pid[j*4 + 1] = sel[j*4 + 1];
        pid[j*4 + 2] = sel[j*4 + 2];
        pid[j*4 + 3] = sel[j*4 + 3];
    }
}

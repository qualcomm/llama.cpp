#pragma once

// Flash-attention per-(dk,dv) tile tuning for the Adreno OpenCL backend.
// Isolated from ggml-opencl.cpp so the tuning numbers are easy to find and
// edit; the FA dispatch and kernel-compile logic stay in the main file.
// This header is a file section — it is #included exactly once, at the point
// in ggml-opencl.cpp where the ggml logging macros are already in scope.

// Per-(dk, dv) FA config; shared by dispatch and supports_op.
struct ggml_opencl_fa_dim {
    int dk; int dv; int bm; int bn; int n_split; int nkv_split_threshold;
};

// Split variant fires when n_kv >= threshold (threshold=0 -> always split).
// Default tuning covers Adreno 7xx/8xx mobile and X1-series laptop GPUs.
static const ggml_opencl_fa_dim g_fa_dims_adreno_default[] = {
    { 40,  40, 64, 32, 1, 0}, { 64,  64, 64, 32, 2, 64},
    { 80,  80, 64, 32, 2, 64}, { 96,  96, 64, 32, 2, 64},
    {112, 112, 64, 32, 2, 64},
    // DK=DV=128 (Qwen3-30B-A3B / Qwen3-4B class). BM=64 is a prefill tuning that
    // outlived prefill: the decompose path took the large-n_q shapes, leaving the
    // tile serving batches of a handful of query rows, where a 64-row tile leaves
    // 56+ of its 64 query lanes idle. Narrowing to BM=16 keeps the work-group at
    // BM*N_SPLIT=512 by raising N_SPLIT to 32, so the lanes move from padding onto
    // the DK reduction. X2-90, Qwen3-30B-A3B-Q4_0, tile route, pp<n_q> t/s:
    //   n_q=5  d2048 20.29 -> 23.58 (+16%)   d4096 13.25 -> 16.43 (+24%)
    //   n_q=8  d2048 30.99 -> 37.59 (+21%)   d4096 20.57 -> 26.40 (+28%)
    // Free for prefill, which does not use the tile: pp512@d2048 348.7 -> 349.3
    // and pp2048@d2048 297.2 -> 297.4, both inside run-to-run spread. Below
    // n_q=5 the flash-decoding route serves this shape and is faster than either
    // tuning. GGML_OPENCL_FA_TUNE=128:128:64:32:2:64 restores the old entry.
    //
    // Portability note: raising N_SPLIT also raises WG_SIZE (= BLOCK_M*N_SPLIT)
    // from 128 to 512, and on a device WITHOUT subgroup shuffle the tile falls
    // back to reducing through __local ACC_TYPE local_partial[BLOCK_N][WG_SIZE].
    // That array grows 16 KB -> 64 KB, taking the kernel's local total from
    // 32 KB (l_k 8 + l_v 8 + partial 16, exactly the Adreno budget) to 80 KB, so
    // this entry is only viable where cl_{khr,qcom}_subgroup_shuffle is present
    // and the shuffle reduce is compiled in. That holds on every device the tile
    // reaches today (X2-90 and 840 verified: test-backend-ops FLASH_ATTN_EXT
    // 2742/2742 and 2743/2743 with this entry, both matching the BM=64 arm), and
    // an over-budget build would fail clBuildProgram, leave the variant
    // unregistered and fall back rather than corrupt -- the FA programs are built
    // with fatal=false. Re-check this arithmetic before widening N_SPLIT further
    // or enabling the tile on a no-shuffle generation.
    {128, 128, 16, 32, 32, 64},
    {192, 128, 16, 16, 1, 0},
    {192, 192, 16, 16, 1, 0},
    {256, 256, 16, 16, 16, 0},
    // DK=DV=512 covers Gemma-4's global-attention layers (SWA layers run on
    // the 256 path). BLOCK_N=16 fixes l_k+l_v at 32 KB local (2×16×128×8 B), so
    // only one WG is resident per CU — N_SPLIT=64 (WG = BM×N_SPLIT = 512) gives
    // that single WG enough threads to hide the K/V load latency: measured
    // +25% on Gemma-4-26B pp2048@d16384 (52.97 → 66) vs N_SPLIT=32/WG=256.
    // N_SPLIT=128/WG=1024 overshoots (58.8); BM<8 starves on query rows.
    // SPLIT_DK_VEC/SPLIT_DV_VEC = 2 float4 each → tiny per-thread footprint.
    {512, 512,  8, 16, 64, 0},
};

struct ggml_opencl_fa_dim_table {
    const ggml_opencl_fa_dim * data;
    size_t                     count;

    const ggml_opencl_fa_dim * begin() const { return data; }
    const ggml_opencl_fa_dim * end()   const { return data + count; }
};

// Mutable copy of the active table; GGML_OPENCL_FA_TUNE patches entries here
// at backend init without touching the const source table.
static ggml_opencl_fa_dim g_fa_dims_runtime[
    sizeof(g_fa_dims_adreno_default) / sizeof(g_fa_dims_adreno_default[0])];

static ggml_opencl_fa_dim_table g_opencl_fa_dims = {
    g_fa_dims_adreno_default,
    sizeof(g_fa_dims_adreno_default) / sizeof(g_fa_dims_adreno_default[0]),
};

// GGML_OPENCL_FA_TUNE=dk:dv:bm:bn:nsplit:thr[,…] — patches matching entries
// in the active table at backend init, before the first FA kernel compiles.
// Unmatched (dk,dv) pairs are warned and ignored.
static void ggml_opencl_fa_apply_env_overrides() {
    const char * e = std::getenv("GGML_OPENCL_FA_TUNE");
    if (!e || !e[0]) {
        return;
    }

    std::string s = e;
    size_t pos = 0;
    while (pos < s.size()) {
        size_t comma = s.find(',', pos);
        std::string entry = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        int dk, dv, bm, bn, nsplit, thr;
        if (std::sscanf(entry.c_str(), "%d:%d:%d:%d:%d:%d", &dk, &dv, &bm, &bn, &nsplit, &thr) == 6) {
            bool patched = false;
            for (size_t i = 0; i < g_opencl_fa_dims.count; ++i) {
                ggml_opencl_fa_dim & d = g_fa_dims_runtime[i];
                if (d.dk == dk && d.dv == dv) {
                    d.bm = bm; d.bn = bn; d.n_split = nsplit; d.nkv_split_threshold = thr;
                    GGML_LOG_INFO("ggml_opencl: FA tune override DK=%d DV=%d -> bm=%d bn=%d n_split=%d thr=%d\n",
                                  dk, dv, bm, bn, nsplit, thr);
                    patched = true;
                    break;
                }
            }
            if (!patched) {
                GGML_LOG_WARN("ggml_opencl: FA tune override DK=%d DV=%d ignored (no matching dim)\n", dk, dv);
            }
        } else {
            GGML_LOG_WARN("ggml_opencl: FA tune override entry malformed: '%s'\n", entry.c_str());
        }
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
}

// Copy the default table into the mutable runtime buffer and apply any
// GGML_OPENCL_FA_TUNE overrides. A per-generation table can be added here
// once it has been tuned on hardware.
static void ggml_cl_init_fa_dims_table() {
    const size_t count = sizeof(g_fa_dims_adreno_default) / sizeof(g_fa_dims_adreno_default[0]);
    for (size_t i = 0; i < count; ++i) {
        g_fa_dims_runtime[i] = g_fa_dims_adreno_default[i];
    }
    g_opencl_fa_dims = { g_fa_dims_runtime, count };
    ggml_opencl_fa_apply_env_overrides();
}

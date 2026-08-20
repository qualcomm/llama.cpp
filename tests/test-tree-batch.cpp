// Does llama.cpp give TREE attention when one batch carries a shared node alongside its own
// divergent children?
//
// This is the load-bearing assumption under tree speculative decoding: a draft tree is flattened
// into one verify batch where an interior node belongs to every path through it, so it carries
// several seq_ids, while its children carry disjoint subsets AT THE SAME POSITIONS.
//
//   batch:  r  at pos n   seq {0,1}      <- shared root, in the SAME batch as its children
//           a0 at pos n+1 seq {0}
//           b0 at pos n+1 seq {1}
//           a1 at pos n+2 seq {0}
//           b1 at pos n+2 seq {1}
//
// Two things have to hold. (1) llama_batch_allocr must accept the shape -- it rejects
// "partial sequence sub-sets", and whether a tree trips that rule is not obvious from the
// code. (2) The mask must actually isolate the branches: a1 must see r and a0 but NOT b0.
//
// The test decodes the tree in one batch and compares its logits against the SAME tokens
// decoded as two ordinary linear sequences. Equal logits => tree attention works.
//
// Usage: test-tree-batch <model.gguf> [n_gpu_layers]

#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static std::vector<float> logits_copy(llama_context * ctx, int i, int n_vocab) {
    const float * p = llama_get_logits_ith(ctx, i);
    return p ? std::vector<float>(p, p + n_vocab) : std::vector<float>();
}

// max |a-b| over the two logit rows, and the argmax agreement
static void compare(const char * what, const std::vector<float> & a, const std::vector<float> & b,
                    double & max_abs, bool & argmax_same) {
    max_abs = 0.0;
    size_t ia = 0, ib = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        max_abs = std::max(max_abs, (double) std::fabs(a[i] - b[i]));
        if (a[i] > a[ia]) { ia = i; }
        if (b[i] > b[ib]) { ib = i; }
    }
    argmax_same = (ia == ib);
    printf("  %-28s max|dlogit| = %.6f   argmax %s (%zu vs %zu)\n",
           what, max_abs, argmax_same ? "MATCH" : "DIFFER", ia, ib);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [n_gpu_layers]\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const int    ngl        = argc > 2 ? atoi(argv[2]) : 99;

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = ngl;

    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = 512;
    cparams.n_batch   = 512;
    cparams.n_ubatch  = 512;
    cparams.n_seq_max = 4;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }

    llama_memory_t mem = llama_get_memory(ctx);

    // A short real prompt so the branch tokens have some context to attend to.
    std::vector<llama_token> prompt(64);
    {
        const char * text = "The quick brown fox jumps over the lazy dog. The capital of France is";
        int n = llama_tokenize(vocab, text, (int) strlen(text), prompt.data(), (int) prompt.size(),
                               /*add_special=*/ true, /*parse_special=*/ false);
        if (n < 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
        prompt.resize(n);
    }
    const int n_prompt = (int) prompt.size();

    // Five distinct, arbitrary tokens: the shared root r and two 2-token continuations.
    const llama_token r  = 262;
    const llama_token a0 = 526, a1 = 691;
    const llama_token b0 = 1031, b1 = 44;

    // Reusable batch big enough for the prompt.
    llama_batch batch = llama_batch_init(n_prompt + 8, 0, 4);

    auto reset_batch = [&]() { batch.n_tokens = 0; };
    auto add = [&](llama_token tok, llama_pos pos, const std::vector<llama_seq_id> & seqs, bool logits) {
        const int i = batch.n_tokens;
        batch.token[i]    = tok;
        batch.pos[i]      = pos;
        batch.n_seq_id[i] = (int32_t) seqs.size();
        for (size_t s = 0; s < seqs.size(); ++s) { batch.seq_id[i][s] = seqs[s]; }
        batch.logits[i]   = logits;
        batch.n_tokens++;
    };

    // ---- prompt into seq 0, then share it with seq 1 -------------------------------------
    // The scratch sequence MUST be seq_cp'd from the canonical one before the tree batch:
    // llama_batch_allocr rejects a batch that couples two sequences whose cached positions
    // have diverged, and an untouched seq 1 has pos_max = -1.
    reset_batch();
    for (int i = 0; i < n_prompt; ++i) { add(prompt[i], i, {0}, false); }
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "prompt decode failed\n"); return 1; }
    llama_memory_seq_cp(mem, 0, 1, -1, -1);

    // ---- reference: two ORDINARY linear sequences ----------------------------------------
    // seq 0 gets [r, a0, a1], seq 1 gets [r, b0, b1]. Each token carries exactly one seq_id,
    // so this is the plain path llama.cpp has always supported.
    reset_batch();
    add(r,  n_prompt,     {0}, false);
    add(a0, n_prompt + 1, {0}, true);
    add(a1, n_prompt + 2, {0}, true);
    add(r,  n_prompt,     {1}, false);
    add(b0, n_prompt + 1, {1}, true);
    add(b1, n_prompt + 2, {1}, true);
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "reference decode failed\n"); return 1; }

    const std::vector<float> ref_a0 = logits_copy(ctx, 1, n_vocab);
    const std::vector<float> ref_a1 = logits_copy(ctx, 2, n_vocab);
    const std::vector<float> ref_b0 = logits_copy(ctx, 4, n_vocab);
    const std::vector<float> ref_b1 = logits_copy(ctx, 5, n_vocab);

    // ---- tree: ONE shared root carrying both seq_ids, children disjoint ------------------
    // Rewind both sequences to the end of the prompt and replay the same tokens as a tree.
    llama_memory_seq_rm(mem, 0, n_prompt, -1);
    llama_memory_seq_rm(mem, 1, n_prompt, -1);

    reset_batch();
    add(r,  n_prompt,     {0, 1}, false);   // <- the node under test
    add(a0, n_prompt + 1, {0},    true);
    add(b0, n_prompt + 1, {1},    true);
    add(a1, n_prompt + 2, {0},    true);
    add(b1, n_prompt + 2, {1},    true);

    const int rc = llama_decode(ctx, batch);
    if (rc != 0) {
        fprintf(stderr, "\nTREE BATCH REJECTED: llama_decode returned %d\n", rc);
        fprintf(stderr, "=> tree speculative decoding cannot use the multi-seq batch shape\n");
        return 2;
    }

    const std::vector<float> tre_a0 = logits_copy(ctx, 1, n_vocab);
    const std::vector<float> tre_b0 = logits_copy(ctx, 2, n_vocab);
    const std::vector<float> tre_a1 = logits_copy(ctx, 3, n_vocab);
    const std::vector<float> tre_b1 = logits_copy(ctx, 4, n_vocab);

    printf("\ntree vs linear (same tokens, same positions):\n");
    double d;  bool same;
    bool ok = true;
    compare("branch A depth 1 (a0)", ref_a0, tre_a0, d, same); ok = ok && same && d < 0.5;
    compare("branch B depth 1 (b0)", ref_b0, tre_b0, d, same); ok = ok && same && d < 0.5;
    compare("branch A depth 2 (a1)", ref_a1, tre_a1, d, same); ok = ok && same && d < 0.5;
    compare("branch B depth 2 (b1)", ref_b1, tre_b1, d, same); ok = ok && same && d < 0.5;

    // Negative control: A and B must NOT agree with each other, otherwise "matching" above
    // would prove nothing (e.g. if the mask leaked and both branches saw the same context).
    compare("control: a1 vs b1 (differ)", tre_a1, tre_b1, d, same);
    const bool control_ok = !same || d > 0.5;
    printf("  control %s\n", control_ok ? "OK (branches are distinguishable)"
                                        : "USELESS (branches identical - test proves nothing)");

    printf("\n%s\n", (ok && control_ok) ? "TREE BATCH OK" : "TREE BATCH MISMATCH");

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return (ok && control_ok) ? 0 : 3;
}

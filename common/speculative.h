#pragma once

#include "llama.h"
#include "common.h"

struct common_speculative;

// comma separated list the provided types
std::string common_speculative_type_name_str(const std::vector<enum common_speculative_type> & types);

// comma separated list of all types
const char * common_speculative_all_types_str();

// parse user provided types
std::vector<enum common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names);

// infer the spec types from the GGUF metadata of a draft model; empty if unknown
std::vector<enum common_speculative_type> common_speculative_types_from_gguf(const std::string & path);

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

// return the max number of draft tokens based on the speculative parameters
int32_t common_speculative_n_max(const common_params_speculative * spec);

common_params common_base_params_to_speculative(const common_params & params);

struct common_speculative_output_limits {
    int32_t total;
    int32_t per_seq;
};

// return the output limits needed for speculative decoding
common_speculative_output_limits common_speculative_get_output_limits(
        int32_t n_batch, int32_t n_parallel, int32_t n_draft);

common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq);

void common_speculative_free(common_speculative * spec);

// A draft TREE, for tree speculative decoding.
//
// A linear draft only ever offers the drafter's rank-1 token at each position, and measurement
// on muse-glimmer-30B says the target's own token is the drafter's RANK 2 in 44.7% of
// rejections -- the drafter already knows the answer the linear draft throws away.
//
// node i carries token[i]; parent[i] is an index into these arrays, or -1 for a child of the
// root (the root is the target's last sampled token, which is not itself a node). depth[i] is
// the draft depth, so the batch position is pos_next + depth[i].
//
// 🔴 Nodes MUST be stored in non-decreasing DEPTH order. The flattened verify batch inherits
// that order, and llama_batch_allocr rejects a batch whose positions decrease within a
// sequence. Every path is a root-to-leaf chain, so the seq_id sets nest and the "partial
// sequence sub-sets" rule is satisfied automatically.
struct common_speculative_draft_tree {
    std::vector<llama_token> token;
    std::vector<int32_t>     parent;
    std::vector<int32_t>     depth;

    void clear() {
        token .clear();
        parent.clear();
        depth .clear();
    }

    size_t size()  const { return token.empty() ? 0 : token.size(); }
    bool   empty() const { return token.empty(); }

    int32_t add(llama_token tok, int32_t par, int32_t dep) {
        token .push_back(tok);
        parent.push_back(par);
        depth .push_back(dep);
        return (int32_t) token.size() - 1;
    }

    // the leaves, i.e. one per root-to-leaf path -- each needs its own seq_id in the verify batch
    std::vector<int32_t> leaves() const {
        std::vector<bool> has_child(token.size(), false);
        for (size_t i = 0; i < parent.size(); ++i) {
            if (parent[i] >= 0) {
                has_child[parent[i]] = true;
            }
        }
        std::vector<int32_t> out;
        for (size_t i = 0; i < token.size(); ++i) {
            if (!has_child[i]) {
                out.push_back((int32_t) i);
            }
        }
        return out;
    }
};

struct common_speculative_draft_params {
    // this flag is used to chain the drafts through all the available implementations
    // after the first successful draft from an implementation, we set it
    //   to false to prevent further drafts for that sequence
    // at the end of the draft() call, all drafting flags will be reset to false
    bool drafting = false;

    // overrides individual configurations (-1 disabled)
    // can be used to constraint the max draft based on the remaining context size
    int32_t n_max = -1;

    llama_pos   n_past;
    llama_token id_last;

    // TODO: remove in the future by keeping track of the prompt from the _begin() call and the consecutive accept calls
    const llama_tokens * prompt;

    // the generated draft from the last _draft() call
    llama_tokens * result;

    // Optional TREE form of the same draft. When the caller supplies this and the
    // implementation supports branching, `result` still holds the SPINE (the rank-1 chain,
    // exactly what a linear caller would get) and `tree` additionally holds the branches.
    // A caller that ignores `tree` therefore behaves exactly as before.
    // Left empty by implementations that do not branch.
    common_speculative_draft_tree * tree = nullptr;
};

common_speculative_draft_params & common_speculative_get_draft_params(common_speculative * spec, llama_seq_id seq_id);

// optionally call once at the beginning of a new generation
void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt);

// process the batch and update the internal state of the speculative context
bool common_speculative_process(common_speculative * spec, const llama_batch & batch);

// generate drafts for the sequences specified with `common_speculative_get_draft_params`
void common_speculative_draft(common_speculative * spec);

// informs the speculative context that n_accepted tokens were accepted by the target model
void common_speculative_accept(common_speculative * spec, llama_seq_id, uint16_t n_accepted);

// diagnostics only: hand the drafter the target's own tokens so it can report where the
// target's choice sat in its candidate list at the position the draft was rejected. That
// bounds what a TREE draft (which would offer the top-b, not just rank 1) could win, without
// building the tree. `accepted` is n accepted draft tokens followed by the target's token.
void common_speculative_rank_probe(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & accepted);

// (optional) get/set internal state
bool common_speculative_get_state(common_speculative * spec, llama_seq_id seq_id, std::vector<uint8_t> & data);
void common_speculative_set_state(common_speculative * spec, llama_seq_id seq_id, const std::vector<uint8_t> & data);

// print statistics about the speculative decoding
void common_speculative_print_stats(const common_speculative * spec);

struct common_speculative_deleter {
    void operator()(common_speculative * s) { common_speculative_free(s); }
};

typedef std::unique_ptr<common_speculative, common_speculative_deleter> common_speculative_ptr;

struct common_speculative_init_result {
    common_speculative_init_result(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt);
    ~common_speculative_init_result();

    llama_model   * model();
    llama_context * context();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using common_speculative_init_result_ptr = std::unique_ptr<common_speculative_init_result>;

common_speculative_init_result_ptr common_speculative_init_from_params(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt);

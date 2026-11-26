#include "llama-expert-cache.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <new>

#define CHECK(c)                                                            \
    do {                                                                    \
        if (!(c)) {                                                         \
            fprintf(stderr, "check failed: %s at line %d\n", #c, __LINE__); \
            abort();                                                        \
        }                                                                   \
    } while (0)

static thread_local bool                     tracking    = false;
static thread_local size_t                   allocations = 0;
static thread_local size_t                   failure     = 0;
static thread_local std::array<void *, 4096> live{};

void operator delete(void * p, size_t) noexcept;
void operator delete[](void * p, size_t) noexcept;

void * operator new(size_t size) {
    if (tracking && ++allocations == failure) {
        throw std::bad_alloc();
    }
    void * p = std::malloc(size ? size : 1);
    if (!p) {
        throw std::bad_alloc();
    }
    if (tracking) {
        CHECK(allocations < live.size());
        live[allocations] = p;
    }
    return p;
}

void operator delete(void * p) noexcept {
    for (auto & entry : live) {
        if (entry == p) {
            entry = nullptr;
        }
    }
    std::free(p);
}

void operator delete(void * p, size_t) noexcept {
    ::operator delete(p);
}

void * operator new[](size_t size) {
    return ::operator new(size);
}

void operator delete[](void * p) noexcept {
    ::operator delete(p);
}

void operator delete[](void * p, size_t) noexcept {
    ::operator delete(p);
}

static size_t route(uint32_t experts, uint32_t used, uint32_t resident, size_t fail_at) {
    size_t count = 0;
    live.fill(nullptr);
    bool caught = false;
    {
        llama_expert_cache cache;
        cache.init_kind_first(used, { 32, 0, 0, 0 });
        cache.n_expert_      = experts;
        cache.n_expert_used_ = used;
        cache.offsets_.resize((size_t) experts * llama_expert_cache::N_KINDS);
        for (uint32_t e = 0; e < resident; ++e) {
            cache.mark_occupied(cache.claim_slot(0, e));
        }
        llama_expert_cache::op_params_t op;
        op.cache = &cache;
        std::vector<int32_t> ids(used);
        std::vector<float>   scores(experts, 1.0f);
        for (uint32_t e = 0; e < used; ++e) {
            ids[e] = e;
        }
        ggml_tensor dst{}, probs{};
        dst.type  = GGML_TYPE_I32;
        dst.ne[0] = used;
        dst.ne[1] = dst.ne[2] = dst.ne[3] = 1;
        dst.data                          = ids.data();
        probs.type                        = GGML_TYPE_F32;
        probs.ne[0]                       = experts;
        probs.ne[1] = probs.ne[2] = probs.ne[3] = 1;
        probs.data                              = scores.data();
        allocations                             = 0;
        failure                                 = fail_at;
        tracking                                = true;
        try {
            llama_expert_cache::reroute_fn(&dst, nullptr, &probs, 0, 1, &op);
        } catch (const std::bad_alloc &) {
            caught = true;
        }
        tracking = false;
        count    = allocations;
        CHECK(caught == (fail_at != 0));
        cache.release_slots(op);
        for (uint32_t s = 0; s < used; ++s) {
            CHECK(cache.slots[s].state.load() != (uint32_t) llama_expert_slot_state::LOADING);
        }
        if (caught) {
            for (int repeat = 0; repeat < 2; ++repeat) {
                llama_expert_cache::reroute_fn(&dst, nullptr, &probs, 0, 1, &op);
                CHECK(op.pinned.size() == used);
                cache.release_slots(op);
            }
        }
    }
    for (auto p : live) {
        CHECK(p == nullptr);
    }
    return count;
}

int main(int argc, char **) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    for (uint32_t experts : { 63, 64, 65, 255, 256, 257 }) {
        for (uint32_t used : { 4, 63, 64, 65 }) {
            if (used > experts) {
                continue;
            }
            for (uint32_t resident : { 0u, used / 2, used }) {
                if (argc > 1 && resident != used) {
                    continue;
                }
                const size_t count = route(experts, used, resident, 0);
                for (size_t i = 1; i <= count; ++i) {
                    route(experts, used, resident, i);
                    route(experts, used, resident, i);
                }
                printf("routing allocation sweep: experts=%u used=%u resident=%u boundaries=%zu PASS\n", experts, used,
                       resident, count);
            }
        }
    }
    puts("test-expert-cache-alloc: PASS");
}

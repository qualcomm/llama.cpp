#include "expert-cache-fixture.h"
#include "llama.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>

#define CHECK(c)                                                            \
    do {                                                                    \
        if (!(c)) {                                                         \
            fprintf(stderr, "check failed: %s at line %d\n", #c, __LINE__); \
            abort();                                                        \
        }                                                                   \
    } while (0)

static std::array<std::string, 2> split_model(const expert_cache_fixture & fixture) {
    ggml_context *   ctx = nullptr;
    gguf_context_ptr source(gguf_init_from_file(fixture.path.c_str(), { false, &ctx }));
    ggml_context_ptr tensors(ctx);
    CHECK(source && tensors);
    std::array<std::string, 2> paths;
    for (int split = 0; split < 2; ++split) {
        gguf_context_ptr file(gguf_init_empty());
        gguf_set_kv(file.get(), source.get());
        gguf_set_val_u16(file.get(), "split.no", split);
        gguf_set_val_u16(file.get(), "split.count", 2);
        gguf_set_val_i32(file.get(), "split.tensors.count", gguf_get_n_tensors(source.get()));
        for (int64_t i = 0; i < gguf_get_n_tensors(source.get()); ++i) {
            auto * t = ggml_get_tensor(ctx, gguf_get_tensor_name(source.get(), i));
            CHECK(t);
            if ((std::strncmp(t->name, "blk.1.", 6) == 0) == (split == 1)) {
                gguf_add_tensor(file.get(), t);
            }
        }
        paths[split] =
            (fixture.directory / (split ? "model-00002-of-00002.gguf" : "model-00001-of-00002.gguf")).string();
        CHECK(gguf_write_to_file(file.get(), paths[split].c_str(), false));
    }
    return paths;
}

static void check_load(const std::string & path, bool cached, bool validate, llama_load_mode mode, bool expected) {
    ggml_backend_dev_t devices[] = { nullptr };
    auto               params    = llama_model_default_params();
    params.devices               = devices;
    params.n_gpu_layers          = 0;
    params.use_extra_bufts       = false;
    params.expert_cache_enabled  = cached;
    params.check_tensors         = validate;
    params.load_mode             = mode;
    auto * model                 = llama_model_load_from_file(path.c_str(), params);
    printf("tensor validation: mode=%d cache=%d check=%d loaded=%d expected=%d\n", mode, cached, validate,
           model != nullptr, expected);
    CHECK((model != nullptr) == expected);
    llama_model_free(model);
}

static void test_validation(ggml_type type, bool split) {
    expert_cache_fixture fixture(false, 32, type, 2048);
    const auto       paths = split ? split_model(fixture) : std::array<std::string, 2>{ fixture.path, fixture.path };
    gguf_context_ptr meta(gguf_init_from_file(paths[1].c_str(), { true, nullptr }));
    CHECK(meta);
    const auto index = gguf_find_tensor(meta.get(), "blk.1.ffn_gate_exps.weight");
    CHECK(index >= 0);
    const size_t start     = gguf_get_data_offset(meta.get()) + gguf_get_tensor_offset(meta.get(), index);
    const size_t bytes     = gguf_get_tensor_size(meta.get(), index);
    const size_t block     = ggml_type_size(type);
    const size_t offsets[] = { 0, (bytes / block / 2) * block, bytes - block };
    for (auto mode : { LLAMA_LOAD_MODE_MMAP, LLAMA_LOAD_MODE_NONE, LLAMA_LOAD_MODE_DIRECT_IO }) {
        for (bool cached : { false, true }) {
            check_load(paths[0], cached, true, mode, true);
            for (size_t offset : offsets) {
                for (bool infinity : { false, true }) {
                    std::fstream file(paths[1], std::ios::in | std::ios::out | std::ios::binary);
                    CHECK(file);
                    std::array<char, 4> saved{}, invalid{};
                    const size_t        size = type == GGML_TYPE_F32 ? 4 : 2;
                    const uint32_t      bits = type == GGML_TYPE_F32  ? (infinity ? 0x7f800000 : 0x7fc00000) :
                                               type == GGML_TYPE_BF16 ? (infinity ? 0x7f80 : 0x7fc0) :
                                                                        (infinity ? 0x7c00 : 0x7e00);
                    std::memcpy(invalid.data(), &bits, size);
                    file.seekg(start + offset);
                    file.read(saved.data(), size);
                    CHECK(file);
                    file.seekp(start + offset);
                    file.write(invalid.data(), size);
                    file.flush();
                    CHECK(file);
                    printf("corruption: type=%s split=%d byte=%zu inf=%d\n", ggml_type_name(type), split, offset,
                           infinity);
                    check_load(paths[0], cached, true, mode, false);
                    check_load(paths[0], cached, false, mode, true);
                    file.seekp(start + offset);
                    file.write(saved.data(), size);
                    file.close();
                    CHECK(file);
                }
            }
        }
    }
}

static void test_read_padding() {
    expert_cache_fixture fixture;
    std::ofstream        file(fixture.path, std::ios::binary | std::ios::app);
    CHECK(file);
    const uint32_t nan = 0x7fc00000;
    for (int i = 0; i < 1024; ++i) {
        file.write(reinterpret_cast<const char *>(&nan), sizeof(nan));
    }
    file.close();
    CHECK(file);
    for (auto mode : { LLAMA_LOAD_MODE_MMAP, LLAMA_LOAD_MODE_NONE, LLAMA_LOAD_MODE_DIRECT_IO }) {
        for (bool cached : { false, true }) {
            check_load(fixture.path, cached, true, mode, true);
        }
    }
    puts("NaN outside tensor data is excluded from validation: PASS");
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    llama_backend_init();
    for (auto type : { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q4_0 }) {
        for (bool split : { false, true }) {
            test_validation(type, split);
        }
    }
    test_read_padding();
    llama_backend_free();
    puts("test-expert-cache-validation: PASS");
}

#if defined(__linux__) && !defined(_GNU_SOURCE)
#    define _GNU_SOURCE
#endif

#if !defined(_WIN32)
#    include <unistd.h>
static ssize_t test_pread(int fd, void * dst, size_t size, off_t offset);
#    define pread test_pread
#endif
#include <cstddef>
static void test_point(const char * name, size_t index);
#define EC_IO_TEST_POINT(name, index) test_point(name, index)
#include "llama-expert-cache-io.h"
#undef EC_IO_TEST_POINT
#include "llama.h"
#if !defined(_WIN32)
#    undef pread
#endif

#include <array>
#include <cstdlib>
#include <string>

#define IO_CHECK(cond)                                                                 \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            fprintf(stderr, "check failed: %s at %s:%d\n", #cond, __FILE__, __LINE__); \
            abort();                                                                   \
        }                                                                              \
    } while (0)

// Failure injection is local to the submitting thread.
static thread_local bool   inject_allocations = false;
static thread_local size_t allocation_count   = 0;
static thread_local size_t fail_allocation    = 0;
static thread_local const char * named_failure      = nullptr;
static thread_local size_t       named_index        = 0;

static void test_point(const char * name, size_t index) {
    if (named_failure && std::strcmp(named_failure, name) == 0 && named_index == index) {
        named_failure = nullptr;
        throw std::bad_alloc();
    }
}

#if !defined(_WIN32)
static std::mutex              read_mu;
static std::condition_variable read_cv;
static bool                    hold_read        = false;
static bool                    read_entered     = false;
static bool                    release_read     = false;
static bool                    batch_returned   = false;
static bool                    callback_entered = false;

template <typename Predicate>
static void wait_read(std::unique_lock<std::mutex> & lock, Predicate ready, const char * event) {
    if (!read_cv.wait_for(lock, std::chrono::seconds(10), ready)) {
        fprintf(stderr, "timeout waiting for %s\n", event);
        abort();
    }
}

// Delay one real file read so callback failure must drain an outstanding job.
static ssize_t test_pread(int fd, void * dst, size_t size, off_t offset) {
    {
        std::unique_lock<std::mutex> lock(read_mu);
        if (hold_read && offset == (off_t) sizeof(uint64_t)) {
            read_entered = true;
            read_cv.notify_all();
            wait_read(lock, [] { return release_read; }, "delayed read release");
        }
    }
    return ::pread(fd, dst, size, offset);
}
#endif

void * operator new(std::size_t n) {
    if (inject_allocations && ++allocation_count == fail_allocation) {
        inject_allocations = false;
        throw std::bad_alloc();
    }
    if (void * p = std::malloc(n ? n : 1)) {
        return p;
    }
    throw std::bad_alloc();
}

void operator delete(void * p, std::size_t) noexcept;
void operator delete[](void * p, std::size_t) noexcept;

void operator delete(void * p) noexcept {
    std::free(p);
}

void operator delete(void * p, std::size_t) noexcept {
    std::free(p);
}

void * operator new[](std::size_t n) {
    return ::operator new(n);
}

void operator delete[](void * p) noexcept {
    ::operator delete(p);
}

void operator delete[](void * p, std::size_t) noexcept {
    ::operator delete(p);
}

struct read_fixture {
    static constexpr uint32_t N = 40;
    std::array<uint64_t, N>   source{};
    std::array<uint64_t, N>   destination{};
    std::array<ec_read_op, N> ops{};
    ec_file                   file;

    read_fixture() {
        for (uint32_t i = 0; i < N; ++i) {
            source[i] = 11 * (i + 1);
        }
        FILE * tmp = std::tmpfile();
        IO_CHECK(tmp != nullptr);
        IO_CHECK(std::fwrite(source.data(), sizeof(uint64_t), N, tmp) == N);
        IO_CHECK(std::fflush(tmp) == 0);
#if defined(_WIN32)
        file.fd = _dup(_fileno(tmp));
#else
        file.fd = dup(fileno(tmp));
#endif
        IO_CHECK(file.fd >= 0);
        file.size = sizeof(source);
        IO_CHECK(std::fclose(tmp) == 0);
        for (uint32_t i = 0; i < N; ++i) {
            ops[i] = { &destination[i], sizeof(uint64_t), i * sizeof(uint64_t), &file };
        }
    }

    uint32_t completed() const {
        uint32_t n = 0;
        for (uint32_t i = 0; i < N; ++i) {
            IO_CHECK(destination[i] == 0 || destination[i] == source[i]);
            if (destination[i] != 0) {
                ++n;
            }
        }
        return n;
    }
};

static size_t test_allocation_failure(size_t     fail_at,
                                      uint32_t   n,
                                      size_t     depth,
                                      size_t     pool_slot,
                                      bool       streamed,
                                      bool       reuse,
                                      uint32_t * completed_out = nullptr) {
    read_fixture fixture;
    size_t       allocations = 0;
    bool         failed      = false;
    {
        ec_thread_pool pool(depth, pool_slot);
        allocation_count   = 0;
        fail_allocation    = fail_at;
        inject_allocations = true;
        try {
            const bool ok = streamed ? pool.read_batch_stream(
                                               fixture.ops.data(), n, nullptr, false, [](uint32_t) {}, [](uint64_t) {})
                                           .ok :
                                       pool.read_batch(fixture.ops.data(), n, nullptr);
            IO_CHECK(ok);
        } catch (const std::bad_alloc &) {
            failed = true;
        }
        inject_allocations = false;
        allocations        = allocation_count;
        IO_CHECK(failed == (fail_at != 0));
        // Reads must be finished before the caller handles an exception.
        const uint32_t completed = fixture.completed();
        if (completed_out) {
            *completed_out = completed;
        }
        for (uint32_t i = 0; i < n; ++i) {
            IO_CHECK(fixture.destination[i] == (i < completed ? fixture.source[i] : 0));
        }
        if (reuse) {
            IO_CHECK(pool.read_batch(fixture.ops.data(), n, nullptr));
            IO_CHECK(fixture.completed() == n);
        }
    }
    return allocations;
}

static void test_allocation_sweep() {
    for (bool streamed : { false, true }) {
        for (size_t depth : { 1, 2, 4 }) {
            for (size_t pool_slot : { 0, 1 }) {
                const size_t total = test_allocation_failure(0, read_fixture::N, depth, pool_slot, streamed, false);
                IO_CHECK(total > read_fixture::N);
                std::array<bool, read_fixture::N> boundaries{};
                for (size_t fail_at = 1; fail_at <= total; ++fail_at) {
                    uint32_t completed = 0;
                    test_allocation_failure(fail_at, read_fixture::N, depth, pool_slot, streamed, false, &completed);
                    IO_CHECK(completed < read_fixture::N);
                    boundaries[completed] = true;
                    test_allocation_failure(fail_at, read_fixture::N, depth, pool_slot, streamed, true);
                }
                for (bool covered : boundaries) {
                    IO_CHECK(covered);
                }
                printf("allocation sweep: streamed=%d depth=%zu pool=%zu sites=%zu PASS\n", streamed, depth, pool_slot,
                       total);
            }
        }
    }
}

static void test_completion_failure(bool at_end, bool allocation) {
    read_fixture                          fixture;
    ec_thread_pool                        pool(2);
    std::array<unsigned, read_fixture::N> completions{};
    bool                                  failed = false;
#if !defined(_WIN32)
    {
        std::lock_guard<std::mutex> lock(read_mu);
        hold_read    = !at_end;
        read_entered = release_read = batch_returned = callback_entered = false;
    }
    bool        returned_early = false;
    std::thread release;
    if (!at_end) {
        release = std::thread([&] {
            std::unique_lock<std::mutex> lock(read_mu);
            wait_read(lock, [] { return read_entered && callback_entered; }, "read and failure callback");
            returned_early = read_cv.wait_for(lock, std::chrono::milliseconds(200), [] { return batch_returned; });
            release_read   = true;
            read_cv.notify_all();
        });
    }
#endif
    auto fail = [&] {
        if (allocation) {
            named_failure = at_end ? "batch-completion" : "read-completion";
            named_index   = 0;
            test_point(named_failure, 0);
            IO_CHECK(false);
        }
        throw std::runtime_error("completion failure");
    };
    try {
        pool.read_batch_stream(
            fixture.ops.data(), read_fixture::N, nullptr, true,
            [&](uint32_t i) {
                IO_CHECK(++completions[i] == 1);
                if (!at_end) {
#if !defined(_WIN32)
                    std::unique_lock<std::mutex> lock(read_mu);
                    wait_read(lock, [] { return read_entered; }, "delayed read start");
                    callback_entered = true;
                    read_cv.notify_all();
#endif
                    fail();
                }
            },
            [&](uint64_t) {
                if (at_end) {
                    fail();
                }
            });
    } catch (const std::bad_alloc &) {
        IO_CHECK(allocation);
        failed = true;
    } catch (const std::runtime_error & e) {
        IO_CHECK(!allocation);
        IO_CHECK(std::string(e.what()) == "completion failure");
        failed = true;
    }
    inject_allocations = false;
#if !defined(_WIN32)
    if (release.joinable()) {
        {
            std::lock_guard<std::mutex> lock(read_mu);
            batch_returned = true;
            read_cv.notify_all();
        }
        release.join();
        IO_CHECK(!returned_early);
    }
    {
        std::lock_guard<std::mutex> lock(read_mu);
        hold_read = false;
    }
#endif
    IO_CHECK(failed);
    IO_CHECK(fixture.completed() == read_fixture::N);
    IO_CHECK(pool.read_batch(fixture.ops.data(), read_fixture::N, nullptr));
    printf("completion failure: at_end=%d allocation=%d PASS\n", at_end, allocation);
}

static void test_read_failure() {
    read_fixture   fixture;
    ec_thread_pool pool(2);
    fixture.ops[1].offset = fixture.file.size;
    IO_CHECK(!pool.read_batch(fixture.ops.data(), read_fixture::N, nullptr));
    IO_CHECK(fixture.completed() == read_fixture::N - 1);
    fixture.ops[1].offset = sizeof(uint64_t);
    IO_CHECK(pool.read_batch(fixture.ops.data(), read_fixture::N, nullptr));
    IO_CHECK(fixture.completed() == read_fixture::N);
    IO_CHECK(pool.read_batch(nullptr, 0, nullptr));
    printf("read failure and empty batch: PASS\n");
}

#if !defined(_WIN32)
[[noreturn]] static void test_log_failure(ggml_log_level, const char *, void *) {
    {
        std::lock_guard<std::mutex> lock(read_mu);
        callback_entered = true;
        read_cv.notify_all();
    }
    throw std::runtime_error("log failure");
}

static void test_submission_log_failure() {
    read_fixture      fixture;
    ec_thread_pool    pool(2);
    ggml_log_callback saved_log;
    void *            saved_data;
    llama_log_get(&saved_log, &saved_data);
    {
        std::lock_guard<std::mutex> lock(read_mu);
        hold_read    = true;
        read_entered = release_read = batch_returned = callback_entered = false;
    }
    bool        returned_early = false;
    std::thread release([&] {
        std::unique_lock<std::mutex> lock(read_mu);
        wait_read(lock, [] { return read_entered && callback_entered; }, "read and failure callback");
        returned_early = read_cv.wait_for(lock, std::chrono::milliseconds(200), [] { return batch_returned; });
        release_read   = true;
        read_cv.notify_all();
    });
    llama_log_set(test_log_failure, nullptr);
    named_failure       = "read-callback";
    named_index         = 2;
    bool original_error = false;
    try {
        pool.read_batch(fixture.ops.data(), read_fixture::N, nullptr);
    } catch (const std::bad_alloc &) {
        original_error = true;
    } catch (const std::runtime_error &) {
    }
    inject_allocations = false;
    {
        std::lock_guard<std::mutex> lock(read_mu);
        batch_returned = true;
        read_cv.notify_all();
    }
    release.join();
    llama_log_set(saved_log, saved_data);
    {
        std::lock_guard<std::mutex> lock(read_mu);
        hold_read = false;
    }
    IO_CHECK(!returned_early);
    IO_CHECK(original_error);
    IO_CHECK(fixture.completed() == 2);
    IO_CHECK(pool.read_batch(fixture.ops.data(), read_fixture::N, nullptr));
    puts("submission diagnostic failure drains reads and preserves original error: PASS");
}
#endif

static void test_repeated_batches() {
    read_fixture   fixture;
    ec_thread_pool pool(4, 1);
    for (unsigned iteration = 0; iteration < 32; ++iteration) {
        fixture.destination.fill(0);
        named_failure      = "read-callback";
        named_index        = iteration % read_fixture::N;
        bool failed        = false;
        try {
            pool.read_batch(fixture.ops.data(), read_fixture::N, nullptr);
        } catch (const std::bad_alloc &) {
            failed = true;
        }
        inject_allocations = false;
        IO_CHECK(failed);
        IO_CHECK(fixture.completed() < read_fixture::N);
        IO_CHECK(pool.read_batch(fixture.ops.data(), read_fixture::N, nullptr));
        IO_CHECK(fixture.completed() == read_fixture::N);
    }
    printf("repeated failed and recovered batches: PASS\n");
}

static void test_unwind_cleanup() {
    for (const char * point : { "batch-state", "read-callback", "submitted" }) {
        bool failed = false;
        try {
            read_fixture   fixture;
            ec_thread_pool pool(2);
            named_failure = point;
            named_index   = std::strcmp(point, "submitted") == 0 ? read_fixture::N : 1;
            if (std::strcmp(point, "batch-state") == 0) {
                named_index = 0;
            }
            pool.read_batch(fixture.ops.data(), read_fixture::N, nullptr);
            IO_CHECK(false);
        } catch (const std::bad_alloc &) {
            failed = true;
        }
        inject_allocations = false;
        IO_CHECK(failed);
    }
    bool failed = false;
    try {
        ec_thread_pool pool(0);
    } catch (const std::invalid_argument &) {
        failed = true;
    }
    IO_CHECK(failed);
    printf("pool and buffer cleanup during unwinding: PASS\n");
}

static void test_normal_completion() {
    read_fixture                          fixture;
    ec_thread_pool                        pool(3);
    std::array<unsigned, read_fixture::N> completed{};
    unsigned                              batches = 0;
    IO_CHECK(pool.read_batch_stream(
                     fixture.ops.data(), read_fixture::N, nullptr, true,
                     [&](uint32_t i) { IO_CHECK(++completed[i] == 1); },
                     [&](uint64_t end_ns) { IO_CHECK(end_ns != 0 && ++batches == 1); })
                 .ok);
    IO_CHECK(fixture.completed() == read_fixture::N);
    for (unsigned count : completed) {
        IO_CHECK(count == 1);
    }
    printf("exactly-once read and batch completion: PASS\n");
}

static void test_original_submission_error() {
    read_fixture   fixture;
    ec_thread_pool pool(2);
    named_failure       = "read-callback";
    named_index         = 1;
    bool original_error = false;
    try {
        pool.read_batch_stream(
            fixture.ops.data(), read_fixture::N, nullptr, false,
            [](uint32_t) { throw std::runtime_error("later completion failure"); },
            [](uint64_t) { throw std::runtime_error("later batch failure"); });
    } catch (const std::bad_alloc &) {
        original_error = true;
    }
    inject_allocations = false;
    IO_CHECK(original_error);
    IO_CHECK(fixture.completed() == 1);
    IO_CHECK(pool.read_batch(fixture.ops.data(), read_fixture::N, nullptr));
    printf("original submission error survives callback failure: PASS\n");
}

static void test_named_boundaries() {
    for (const char * point : { "batch-state", "ops", "completed", "timing-start", "timing-finished", "ready",
                                "read-callback", "queue", "submitted", "before-drain" }) {
        for (uint32_t accepted : { 0u, 1u, read_fixture::N / 2, read_fixture::N }) {
            const bool submitting = std::strcmp(point, "read-callback") == 0 || std::strcmp(point, "queue") == 0;
            if ((!submitting && accepted != 0) || (submitting && accepted == read_fixture::N)) {
                continue;
            }
            read_fixture   fixture;
            ec_thread_pool pool(2);
            named_failure = point;
            named_index   = std::strcmp(point, "submitted") == 0 ? read_fixture::N : accepted;
            bool failed   = false;
            try {
                pool.read_batch_stream(
                    fixture.ops.data(), read_fixture::N, nullptr, true, [](uint32_t) {}, [](uint64_t) {});
            } catch (const std::bad_alloc &) {
                failed = true;
            }
            IO_CHECK(failed && named_failure == nullptr);
            const uint32_t expected =
                submitting ?
                    accepted :
                    (std::strcmp(point, "submitted") == 0 || std::strcmp(point, "before-drain") == 0 ? read_fixture::N :
                                                                                                       0);
            IO_CHECK(fixture.completed() == expected);
            IO_CHECK(pool.read_batch(fixture.ops.data(), read_fixture::N, nullptr));
            IO_CHECK(fixture.completed() == read_fixture::N);
            printf("named failure: %s accepted=%u drained=%u PASS\n", point, accepted, expected);
        }
    }
}

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const std::string mode = argc > 1 ? argv[1] : "all";
    if (mode == "partial") {
        test_named_boundaries();
        return 0;
    }
    if (mode == "all" || mode == "allocations") {
        test_named_boundaries();
        test_allocation_sweep();
    }
    if (mode == "all" || mode == "callbacks") {
        for (bool allocation : { false, true }) {
            test_completion_failure(false, allocation);
            test_completion_failure(true, allocation);
        }
        test_original_submission_error();
#if !defined(_WIN32)
        test_submission_log_failure();
#endif
    }
    if (mode == "all" || mode == "normal") {
        test_read_failure();
        test_normal_completion();
    }
    if (mode == "all" || mode == "cleanup") {
        test_repeated_batches();
        test_unwind_cleanup();
    }
    return 0;
}

#pragma once

#if defined(__linux__) && !defined(_GNU_SOURCE)
#    define _GNU_SOURCE
#endif

#include "llama-impl.h"

#include <fcntl.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <thread>
#include <vector>

#if defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <io.h>
#    include <windows.h>
#else
#    include <unistd.h>
#endif

struct ec_file;

#ifndef EC_IO_TEST_POINT
#    define EC_IO_TEST_POINT(name, index) ((void) 0)
#    define EC_IO_TEST_POINT_DEFAULT
#endif

static int64_t ec_pread(int fd, void * dst, size_t size, int64_t offset) {
#if defined(_WIN32)
    OVERLAPPED ov{};
    ov.Offset     = (DWORD) offset;
    ov.OffsetHigh = (DWORD) ((uint64_t) offset >> 32);
    DWORD got     = 0;
    if (!ReadFile((HANDLE) _get_osfhandle(fd), dst, (DWORD) std::min(size, (size_t) MAXDWORD), &got, &ov)) {
        errno = EIO;
        return -1;
    }
    return got;
#else
    if (offset < 0 || (uint64_t) offset > (uint64_t) std::numeric_limits<off_t>::max()) {
        errno = EINVAL;
        return -1;
    }
    return pread(fd, dst, std::min(size, (size_t) std::numeric_limits<ssize_t>::max()), (off_t) offset);
#endif
}

struct ec_read_op {
    void *    dst;
    size_t    size;
    size_t    offset;
    ec_file * file;
};

static inline uint64_t stats_now_ns() {
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// Use aligned bounce buffers to read 32-byte-aligned expert slices with O_DIRECT.
struct ec_odirect {
    struct buf {
        void * p         = nullptr;
        size_t n         = 0;
        size_t alignment = 0;
    };

    static constexpr size_t N_POOLS = 2;  // 0 = main pool, 1 = worker pool

    // Each (pool, op) needs a private buffer because reads can overlap.
    std::vector<buf> bufs[N_POOLS];
    std::mutex       bufs_mu;

    ~ec_odirect() {
        for (size_t s = 0; s < N_POOLS; ++s) {
            for (auto & b : bufs[s]) {
                if (b.p) {
                    ::operator delete(b.p, std::align_val_t(b.alignment));
                }
            }
        }
    }

    void * buf_for(size_t s, size_t i, size_t n, size_t blk) {
        std::lock_guard<std::mutex> lk(bufs_mu);
        if (s >= N_POOLS) {
            return nullptr;
        }
        auto & v = bufs[s];
        if (i >= v.size()) {
            v.resize(i + 1);
        }
        buf & b = v[i];
        if (b.n < n || b.alignment < blk) {
            if (b.p) {
                ::operator delete(b.p, std::align_val_t(b.alignment));
            }
            b.p = ::operator new(n, std::align_val_t(blk), std::nothrow);
            if (b.p == nullptr) {
                b.n = 0;
                return nullptr;
            }
            b.n         = n;
            b.alignment = blk;
        }
        return b.p;
    }

    // Return the byte count, or -1 to retry with buffered I/O.
    int64_t pread_aligned(int        fd,
                          size_t     blk,
                          size_t     pool_slot,
                          size_t     buf_idx,
                          void *     dst,
                          size_t     size,
                          size_t     offset,
                          uint64_t * copy_ns = nullptr) {
        const uint64_t max_offset = INT64_MAX;
        if (offset > max_offset || size > max_offset - offset) {
            return -1;
        }
        const int64_t a_off = (int64_t) (offset & ~(blk - 1));
        const size_t  lead  = offset - (size_t) a_off;
        if (size > SIZE_MAX - lead - (blk - 1)) {
            return -1;
        }
        const size_t a_len = (lead + size + blk - 1) & ~(size_t) (blk - 1);
        if (a_len > max_offset - (uint64_t) a_off) {
            return -1;
        }

        void * bounce = buf_for(pool_slot, buf_idx, a_len, blk);
        if (bounce == nullptr) {
            return -1;
        }

        size_t done = 0;
        while (done < a_len) {
            int64_t got = ec_pread(fd, (uint8_t *) bounce + done, a_len - done, a_off + (int64_t) done);
            if (got < 0) {
                if (errno == EINTR) {
                    continue;
                }
                return -1;
            }
            if (got == 0) {
                break;  // the last block of the file can be short
            }
            done += (size_t) got;
        }
        if (done < lead + size) {
            return -1;
        }
        const uint64_t copy_start = copy_ns ? stats_now_ns() : 0;
        memcpy(dst, (uint8_t *) bounce + lead, size);
        if (copy_ns) {
            *copy_ns = stats_now_ns() - copy_start;
        }
        return (int64_t) size;
    }
};

struct ec_file {
    int    fd         = -1;
    int    direct_fd  = -1;
    size_t size       = 0;
    size_t block_size = 0;

    ~ec_file() {
#if defined(_WIN32)
        if (fd >= 0) {
            _close(fd);
        }
        if (direct_fd >= 0) {
            _close(direct_fd);
        }
#else
        if (fd >= 0) {
            close(fd);
        }
        if (direct_fd >= 0) {
            close(direct_fd);
        }
#endif
    }

    bool read(ec_odirect * direct,
              size_t       pool_slot,
              size_t       buf_idx,
              void *       dst,
              size_t       len,
              size_t       offset,
              uint64_t *   copy_ns = nullptr) const {
        if (offset > size || len > size - offset) {
            return false;
        }
        if (len == 0) {
            return true;
        }
        if (direct_fd >= 0 && direct && len <= (size_t) INT64_MAX) {
            try {
                if (direct->pread_aligned(direct_fd, block_size, pool_slot, buf_idx, dst, len, offset, copy_ns) ==
                    (int64_t) len) {
                    return true;
                }
            } catch (const std::bad_alloc &) {
            }
        }
        const uint64_t max_offset = INT64_MAX;
        if (offset > max_offset || len > max_offset - offset) {
            return false;
        }
        size_t done = 0;
        while (done < len) {
            const int64_t got = ec_pread(fd, (char *) dst + done, len - done, (int64_t) (offset + done));
            if (got < 0 && errno == EINTR) {
                continue;
            }
            if (got <= 0) {
                return false;
            }
            done += (size_t) got;
        }
        return true;
    }
};

struct ec_read_batch_result {
    bool     ok;
    uint64_t read_ns;
    uint64_t copy_ns;
};

// Reuse read threads across batches.
class ec_thread_pool {
    struct completed_read {
        uint32_t index;
        bool     ok;
        uint64_t copy_ns;
    };

    struct batch_state {
        std::vector<ec_read_op>     ops;
        std::vector<completed_read> completed;
        std::vector<uint64_t>       started_ns;
        std::vector<uint8_t>        finished;
        std::mutex                  mu;
        std::condition_variable     cv;
        uint32_t                    n_completed = 0;
        uint32_t                    n_consumed  = 0;
        uint32_t                    remaining   = 0;
        uint64_t                    done_ns     = 0;
    };

    struct read_work {
        std::shared_ptr<batch_state> state;
        uint32_t                     i;
        ec_odirect *                 od;
        bool                         collect_timing;
        size_t                       pool_slot;

        void operator()() const {
            if (collect_timing) {
                std::lock_guard<std::mutex> lk(state->mu);
                state->started_ns[i] = stats_now_ns();
            }

            uint64_t op_copy_ns = 0;
            bool     ok         = false;
            try {
                const auto & op = state->ops[i];
                ok =
                    op.file->read(od, pool_slot, i, op.dst, op.size, op.offset, collect_timing ? &op_copy_ns : nullptr);
            } catch (...) {
                ok = false;
            }

            std::lock_guard<std::mutex> lk(state->mu);
            state->completed[state->n_completed++] = { i, ok, op_copy_ns };
            if (collect_timing) {
                state->finished[i] = 1;
            }
            if (--state->remaining == 0) {
                state->done_ns = collect_timing ? stats_now_ns() : 0;
            }
            state->cv.notify_one();
        }
    };

  public:
    static size_t batch_memory(size_t n, bool timing) {
        const size_t item = sizeof(ec_read_op) + 2 * sizeof(completed_read) + sizeof(std::function<void()>) +
                            sizeof(read_work) + (timing ? sizeof(uint64_t) + sizeof(uint8_t) : 0);
        if (n > (SIZE_MAX - sizeof(batch_state)) / item) {
            throw std::overflow_error("expert cache read batch memory exceeds size_t");
        }
        return sizeof(batch_state) + n * item;
    }

    // Main and prefetch pools use separate O_DIRECT bounce buffers.
    explicit ec_thread_pool(size_t n_threads, size_t pool_slot = 0) : pool_slot_(pool_slot) {
        if (n_threads == 0) {
            throw std::invalid_argument("expert cache: read pool needs at least one worker");
        }
        threads_.reserve(n_threads);
        try {
            for (size_t i = 0; i < n_threads; ++i) {
                threads_.emplace_back(&ec_thread_pool::worker_loop, this);
            }
        } catch (...) {
            stop();
            throw;
        }
    }

    ~ec_thread_pool() { stop(); }

    // Submit the batch and return true only when every read succeeds.
    bool read_batch(const ec_read_op * ops, uint32_t n, ec_odirect * od) {
        return read_batch_stream(ops, n, od, false, [](uint32_t) {}, [](uint64_t) {}).ok;
    }

    ec_read_batch_result read_batch_stream(const ec_read_op *                    ops,
                                           uint32_t                              n,
                                           ec_odirect *                          od,
                                           bool                                  collect_timing,
                                           const std::function<void(uint32_t)> & on_complete,
                                           const std::function<void(uint64_t)> & on_reads_done) {
        if (n == 0) {
            return { true, 0, 0 };
        }

        const uint64_t start_ns = collect_timing ? stats_now_ns() : 0;
        EC_IO_TEST_POINT("batch-state", 0);
        auto           state    = std::make_shared<batch_state>();
        EC_IO_TEST_POINT("ops", 0);
        state->ops.assign(ops, ops + n);
        EC_IO_TEST_POINT("completed", 0);
        state->completed.resize(n);
        if (collect_timing) {
            EC_IO_TEST_POINT("timing-start", 0);
            state->started_ns.resize(n);
            EC_IO_TEST_POINT("timing-finished", 0);
            state->finished.resize(n);
        }

        std::vector<completed_read> ready;
        EC_IO_TEST_POINT("ready", 0);
        ready.reserve(n);

        uint64_t           copy_ns      = 0;
        bool               all_ok       = true;
        bool               callbacks_ok = true;
        std::exception_ptr error;
        const size_t       pool_slot = pool_slot_;

        {
            std::lock_guard<std::mutex> lk(mu_);
            if (shutdown_) {
                throw std::runtime_error("expert cache: read pool is shutting down");
            }
            for (uint32_t i = 0; i < n; ++i) {
                try {
                    EC_IO_TEST_POINT("read-callback", i);
                    std::function<void()> job = read_work{ state, i, od, collect_timing, pool_slot };
                    EC_IO_TEST_POINT("queue", i);
                    queue_.push_back(std::move(job));
                    ++state->remaining;
                } catch (...) {
                    error = std::current_exception();
                    break;
                }
            }
            if (!error) {
                try {
                    EC_IO_TEST_POINT("submitted", n);
                } catch (...) {
                    error = std::current_exception();
                }
            }
        }
        cv_.notify_all();

        try {
            EC_IO_TEST_POINT("before-drain", 0);
            if (error) {
                LLAMA_LOG_WARN("%s: expert cache: failed to queue a streamed read batch\n", __func__);
            }

            for (;;) {
                uint32_t oldest      = 0;
                uint64_t elapsed_ns  = 0;
                uint32_t outstanding = 0;
                bool     done        = false;
                {
                    std::unique_lock<std::mutex> lk(state->mu);
                    if (collect_timing) {
                        while (state->n_consumed == state->n_completed && state->remaining != 0) {
                            if (state->cv.wait_for(lk, std::chrono::seconds(30), [&] {
                                    return state->n_consumed != state->n_completed || state->remaining == 0;
                                })) {
                                continue;
                            }

                            outstanding = state->remaining;
                            for (uint32_t i = 0; i < n; ++i) {
                                if (state->finished[i]) {
                                    continue;
                                }
                                oldest     = i;
                                elapsed_ns = state->started_ns[i] == 0 ? 0 : stats_now_ns() - state->started_ns[i];
                                break;
                            }
                            break;
                        }
                    } else {
                        state->cv.wait(
                            lk, [&] { return state->n_consumed != state->n_completed || state->remaining == 0; });
                    }

                    ready.clear();
                    while (state->n_consumed < state->n_completed) {
                        ready.push_back(state->completed[state->n_consumed++]);
                    }
                    done = state->remaining == 0;
                }

                if (outstanding != 0) {
                    const ec_read_op & op = state->ops[oldest];
                    LLAMA_LOG_WARN(
                        "%s: expert cache: waiting for %u read(s), op=%u offset=%lld size=%zu elapsed=%.3f s\n",
                        __func__, outstanding, oldest, (long long) op.offset, op.size, (double) elapsed_ns / 1e9);
                }

                if (done && callbacks_ok) {
                    try {
                        on_reads_done(state->done_ns);
                    } catch (...) {
                        if (!error) {
                            error = std::current_exception();
                        }
                        callbacks_ok = false;
                    }
                }
                for (const completed_read & result : ready) {
                    all_ok = all_ok && result.ok;
                    copy_ns += result.copy_ns;
                    if (result.ok && callbacks_ok) {
                        try {
                            on_complete(result.index);
                        } catch (...) {
                            if (!error) {
                                error = std::current_exception();
                            }
                            callbacks_ok = false;
                        }
                    }
                }
                if (done) {
                    break;
                }
            }
        } catch (...) {
            if (!error) {
                error = std::current_exception();
            }
            std::unique_lock<std::mutex> lk(state->mu);
            state->cv.wait(lk, [&] { return state->remaining == 0; });
        }

        // Borrowed files and destinations must survive every submitted read.
        if (error) {
            std::rethrow_exception(error);
        }

        const uint64_t read_ns =
            collect_timing ? (state->done_ns != 0 ? state->done_ns - start_ns : stats_now_ns() - start_ns) : 0;
        return { all_ok, read_ns, copy_ns };
    }

  private:
    void stop() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            shutdown_ = true;
        }
        cv_.notify_all();
        for (auto & t : threads_) {
            t.join();
        }
    }

    void worker_loop() {
        for (;;) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return shutdown_ || !queue_.empty(); });
                if (queue_.empty()) {
                    if (shutdown_) {
                        return;
                    }
                    continue;
                }
                job = std::move(queue_.front());
                queue_.pop_front();
            }
            job();
        }
    }

    std::vector<std::thread>          threads_;
    std::mutex                        mu_;
    std::condition_variable           cv_;
    std::deque<std::function<void()>> queue_;
    bool                              shutdown_  = false;
    size_t                            pool_slot_ = 0;
};

#ifdef EC_IO_TEST_POINT_DEFAULT
#    undef EC_IO_TEST_POINT
#    undef EC_IO_TEST_POINT_DEFAULT
#endif

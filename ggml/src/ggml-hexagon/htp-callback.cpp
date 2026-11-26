#include "htp-callback.h"

#include "ggml-impl.h"
#include "htp-common.h"
#include "htp/htp-ops.h"

#include <chrono>
#include <cstring>
#include <exception>
#include <new>

static constexpr int     marker_index = GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1;
static constexpr int32_t marker       = 0x48544342;
static_assert(sizeof(ggml_map_custom1_op_params) <= marker_index * sizeof(int32_t), "callback marker overlaps params");
static_assert(sizeof(ggml_map_custom2_op_params) <= marker_index * sizeof(int32_t), "callback marker overlaps params");
static_assert(sizeof(std::atomic<uint32_t>) == sizeof(uint32_t), "fence atomic size mismatch");
static_assert(std::atomic<uint32_t>::is_always_lock_free, "fences need lock-free atomics");

bool htp_callback_is_marked(const ggml_tensor * tensor) {
    if (!tensor || tensor->op_params[marker_index] != marker) {
        return false;
    }
    if (tensor->op == GGML_OP_MAP_CUSTOM1) {
        ggml_map_custom1_op_params params;
        memcpy(&params, tensor->op_params, sizeof(params));
        return params.fun && params.n_tasks == 1;
    }
    if (tensor->op == GGML_OP_MAP_CUSTOM2) {
        ggml_map_custom2_op_params params;
        memcpy(&params, tensor->op_params, sizeof(params));
        return params.fun && params.n_tasks == 1;
    }
    return false;
}

void htp_callback_mark(ggml_tensor * tensor) {
    tensor->op_params[marker_index] = marker;
    GGML_ASSERT(htp_callback_is_marked(tensor));
}

bool htp_callback_supports(const ggml_tensor * tensor) {
    if (!htp_callback_is_marked(tensor) || htp_tensor_is_stored_f16(tensor)) {
        return false;
    }
    for (int i = 0; i < 2; ++i) {
        if (htp_tensor_is_stored_f16(tensor->src[i])) {
            return false;
        }
    }
    return true;
}

static std::atomic<uint32_t> * fence_words(const uint8_t * fence) {
    return reinterpret_cast<std::atomic<uint32_t> *>(const_cast<uint8_t *>(fence));
}

uint32_t htp_callback_fence_seq(const uint8_t * fence) {
    return fence_words(fence)[0].load(std::memory_order_acquire);
}

uint32_t htp_callback_fence_status(const uint8_t * fence) {
    return fence_words(fence)[1].load(std::memory_order_relaxed);
}

void htp_callback_fence_write(uint8_t * fence, uint32_t seq, uint32_t status) {
    fence_words(fence)[1].store(status, std::memory_order_relaxed);
    fence_words(fence)[0].store(seq, std::memory_order_release);
}

htp_callback_worker::htp_callback_worker(uint8_t * control) : control(control), error(HTP_STATUS_OK) {
    for (size_t offset : { size_t(0), size_t(HTP_FENCE_SLOT_SIZE) }) {
        auto * words = fence_words(control + offset);
        new (&words[0]) std::atomic<uint32_t>(0);
        new (&words[1]) std::atomic<uint32_t>(HTP_STATUS_OK);
    }
    thread = std::thread(&htp_callback_worker::run, this);
}

htp_callback_worker::~htp_callback_worker() {
    {
        std::lock_guard<std::mutex> lock(mutex);
        stopping.store(true, std::memory_order_relaxed);
    }
    cv.notify_all();
    thread.join();
}

uint32_t htp_callback_worker::enqueue(const ggml_tensor * tensor) {
    GGML_ASSERT(htp_callback_is_marked(tensor));
    std::lock_guard<std::mutex> lock(mutex);
    if (++next_seq == 0) {
        ++next_seq;
    }
    request req{};
    req.seq = next_seq;
    req.dst = *tensor;
    for (int i = 0; i < 2; ++i) {
        if (tensor->src[i]) {
            req.src[i] = *tensor->src[i];
        }
    }
    requests.push_back(req);
    cv.notify_one();
    return req.seq;
}

void htp_callback_worker::run() {
    while (true) {
        request req;
        {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock, [&] { return stopping.load(std::memory_order_relaxed) || !requests.empty(); });
            if (stopping.load(std::memory_order_relaxed)) {
                return;
            }
            req = requests.front();
            requests.pop_front();
        }
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
        unsigned   polls    = 0;
        while (htp_callback_fence_seq(control) != req.seq) {
            if (stopping.load(std::memory_order_relaxed)) {
                return;
            }
            if (std::chrono::steady_clock::now() >= deadline) {
                error = HTP_STATUS_INTERNAL_ERR;
                break;
            }
            if (++polls < 256) {
                std::this_thread::yield();
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }
        }
        const uint32_t status = htp_callback_fence_status(control);
        if (status > HTP_STATUS_OK) {
            error = status;
        }
        if (error == HTP_STATUS_OK) {
            try {
                if (req.dst.op == GGML_OP_MAP_CUSTOM1) {
                    ggml_map_custom1_op_params params;
                    memcpy(&params, req.dst.op_params, sizeof(params));
                    params.fun(&req.dst, &req.src[0], 0, 1, params.userdata);
                } else {
                    ggml_map_custom2_op_params params;
                    memcpy(&params, req.dst.op_params, sizeof(params));
                    params.fun(&req.dst, &req.src[0], &req.src[1], 0, 1, params.userdata);
                }
            } catch (const std::exception & exc) {
                GGML_LOG_ERROR("ggml-hex: cache callback failed: %s\n", exc.what());
                error = HTP_STATUS_INTERNAL_ERR;
            } catch (...) {
                error = HTP_STATUS_INTERNAL_ERR;
            }
        }
        htp_callback_fence_write(control + HTP_FENCE_SLOT_SIZE, req.seq, error);
    }
}

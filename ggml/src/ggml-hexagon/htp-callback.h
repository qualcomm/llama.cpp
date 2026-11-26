#ifndef HTP_CALLBACK_H
#define HTP_CALLBACK_H

#include "ggml.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <thread>

void     htp_callback_mark(ggml_tensor * tensor);
bool     htp_callback_is_marked(const ggml_tensor * tensor);
bool     htp_callback_supports(const ggml_tensor * tensor);
uint32_t htp_callback_fence_seq(const uint8_t * fence);
uint32_t htp_callback_fence_status(const uint8_t * fence);
void     htp_callback_fence_write(uint8_t * fence, uint32_t seq, uint32_t status);

class htp_callback_worker {
  public:
    explicit htp_callback_worker(uint8_t * control);
    ~htp_callback_worker();
    uint32_t enqueue(const ggml_tensor * tensor);

  private:
    struct request {
        uint32_t    seq;
        ggml_tensor dst;
        ggml_tensor src[2];
    };

    void                    run();
    uint8_t *               control;
    uint32_t                next_seq = 0;
    uint32_t                error;
    std::atomic<bool>       stopping{ false };
    std::mutex              mutex;
    std::condition_variable cv;
    std::deque<request>     requests;
    std::thread             thread;
};

#endif

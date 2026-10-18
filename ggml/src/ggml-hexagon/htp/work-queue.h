#ifndef HTP_WORK_QUEUE_H
#define HTP_WORK_QUEUE_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#include "htp-thread.h"

typedef void (*work_queue_func_t)(const htp_thread_t * th, unsigned int n, unsigned int i, void *);

struct work_queue_s;
typedef struct work_queue_s * work_queue_t;

size_t       work_queue_sizeof(uint32_t n_threads, uint32_t capacity, uint32_t stack_size);
size_t       work_queue_alignof(void);
work_queue_t work_queue_init(void * ptr, uint32_t n_threads, uint32_t capacity, uint32_t stack_size, dma_queue_t * dma, struct htp_thread_trace * trace);
void         work_queue_free(work_queue_t q);

void work_queue_wakeup(work_queue_t q);
void work_queue_suspend(work_queue_t q);

bool work_queue_push(work_queue_t q, work_queue_func_t func, void * data, unsigned int n);
bool work_queue_pop(work_queue_t q);
void work_queue_flush(work_queue_t q);

static inline bool work_queue_run(work_queue_t q, work_queue_func_t func, void * data, unsigned int n) {
    if (!work_queue_push(q, func, data, n)) {
        return false;
    }
    return work_queue_pop(q);
}

#endif  // #ifndef HTP_WORK_QUEUE_H

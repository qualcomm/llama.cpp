#ifndef HTP_FENCE_H
#define HTP_FENCE_H

#include <stdatomic.h>
#include <stdint.h>

#include "hex-utils.h"
#include "htp-ops.h"

static inline atomic_uint * htp_mdev_fence(const void * fence_base, uint32_t idx) {
    return (atomic_uint *) ((const uint8_t *) fence_base + (size_t) idx * HTP_FENCE_SLOT_SIZE);
}

static inline void htp_fence_write(void * fence_ptr, uint32_t seq, uint32_t status) {
    atomic_uint * fence = (atomic_uint *) fence_ptr;
    atomic_store(&fence[0], seq);
    atomic_store(&fence[1], status);
    asm volatile ("syncht" : : : "memory");
    Q6_dccleaninva_A((void *) fence);
}

static inline void htp_fence_read(const void * fence_ptr, uint32_t * seq, uint32_t * status) {
    const atomic_uint * fence = (const atomic_uint *) fence_ptr;
    Q6_dccleaninva_A((void *) fence);
    asm volatile ("syncht" : : : "memory");
    *seq = atomic_load(&fence[0]);
    *status = atomic_load(&fence[1]);
}

#endif // HTP_FENCE_H

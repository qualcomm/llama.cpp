#ifndef HTP_TENSOR_H
#define HTP_TENSOR_H

#include <stdint.h>
#include <stdbool.h>
#include "htp-ops.h"
#include "hex-bitmap.h"

static inline void * htp_tensor_data(const struct htp_tensor * t) {
    return (void *) (uintptr_t) t->data;
}

static inline uint32_t * htp_tensor_flags(const struct htp_tensor * t) {
    return (uint32_t *) &t->flags;
}

static inline bool htp_tensor_is_contiguous(const struct htp_tensor * t, uint32_t type_size) {
    uint32_t next_nb = type_size;
    if (t->ne[0] != 1 && t->nb[0] != next_nb) {
        return false;
    }
    next_nb *= t->ne[0];
    for (int i = 1; i < HTP_OP_MAX_DIMS; i++) {
        if (t->ne[i] != 1 && t->nb[i] != next_nb) {
            return false;
        }
        next_nb *= t->ne[i];
    }
    return true;
}

static inline uint32_t htp_tensor_get_row_size(int type, uint32_t ne00) {
    switch (type) {
        case HTP_TYPE_F32:  return ne00 * 4;
        case HTP_TYPE_F16:  return ne00 * 2;
        case HTP_TYPE_Q8_0: return (ne00 / 32) * 34;
        default:            return 0;
    }
}

struct htp_context;
void htp_tensor_flush_all(struct htp_context * ctx, const struct htp_tensor * const * tensors, uint32_t n);
void htp_tensor_dirty_all(struct htp_context * ctx, const struct htp_tensor * const * tensors, uint32_t n);

#endif // HTP_TENSOR_H

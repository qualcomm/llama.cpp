#ifndef HTP_POOL_OPS_H
#define HTP_POOL_OPS_H

#include <stdint.h>
#include <stdbool.h>

struct htp_pool_2d_kernel_params {
    uint32_t src_x;
    uint32_t src_y;
    uint32_t dst_x;
    uint32_t dst_y;
    uint32_t kernel_x;
    uint32_t kernel_y;
    uint32_t stride_x;
    uint32_t stride_y;
    int32_t  pad_x;
    int32_t  pad_y;
    uint32_t src_plane_bytes;
    uint32_t dst_plane_bytes;
    uint32_t src_plane_bytes_aligned;
    uint32_t dst_plane_bytes_aligned;
    uint32_t planes_per_thread;
    uint32_t use_dma;
    uint32_t pool_op;
    uint32_t fast_path;
    uint32_t narrow_path;
    uint32_t global_path;
    uint32_t block_path;
    uint32_t exact_path;
    uint32_t narrow_general_path;
    float    inv_kernel_area;
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_pool_2d_kernel_params) <= 128, "htp_pool_2d_kernel_params is too large");
#else
_Static_assert(sizeof(struct htp_pool_2d_kernel_params) <= 128, "htp_pool_2d_kernel_params is too large");
#endif

// Interior column range [ox_lo, ox_hi) where every kernel column kx in [0, kernel_x) is
// guaranteed in-bounds for every ox. The general path can use either contiguous loads,
// deinterleaving, or HVX gather depending on stride_x.
static inline bool htp_pool2d_vec_interior_range(
    uint32_t src_x, uint32_t dst_x, uint32_t kernel_x, uint32_t stride_x, int32_t pad_x,
    uint32_t * ox_lo, uint32_t * ox_hi) {
    if (stride_x == 0) {
        *ox_lo = 0;
        *ox_hi = 0;
        return false;
    }

    const uint32_t lo_raw = ((uint32_t) pad_x + stride_x - 1) / stride_x;
    *ox_lo = (lo_raw < dst_x) ? lo_raw : dst_x;

    const int32_t numer_hi = (int32_t) src_x - (int32_t) kernel_x + pad_x;
    if (numer_hi < 0) {
        *ox_hi = 0;
    } else {
        const uint32_t hi_raw = (uint32_t) numer_hi / stride_x + 1;
        *ox_hi = (hi_raw < dst_x) ? hi_raw : dst_x;
    }
    if (*ox_hi < *ox_lo) {
        *ox_hi = *ox_lo;
    }
    return true;
}

#endif

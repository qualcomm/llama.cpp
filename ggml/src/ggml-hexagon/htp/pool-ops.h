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
    uint32_t n_threads;
    uint32_t planes;
    uint32_t use_dma;
    uint32_t pool_op;
    uint32_t fast_path;
    uint32_t narrow_path;
    uint32_t global_path;
    uint32_t block_path;
    uint32_t exact_path;
    uint32_t avg_divide_count;
    uint32_t ox_lo;
    uint32_t ox_hi;
    float    inv_kernel_area;
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_pool_2d_kernel_params) <= 128, "htp_pool_2d_kernel_params is too large");
#else
_Static_assert(sizeof(struct htp_pool_2d_kernel_params) <= 128, "htp_pool_2d_kernel_params is too large");
#endif

#endif // HTP_POOL_OPS_H

#pragma clang diagnostic ignored "-Wunused-variable"

#include <float.h>
#include <HAP_farf.h>

#include "hex-common.h"
#include "dma-queue.h"
#include "hex-profile.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "hvx-inverse.h"
#include "hvx-types.h"
#include "hvx-utils.h"
#include "pool-ops.h"

#define HTP_POOL_MAX 0
#define HTP_POOL_AVG 1

static inline float pool_inv_count(uint32_t count) {
    HVX_VectorAlias inv;
    inv.v = hvx_vec_inverse_f32(hvx_vec_splat_f32((float) count));
    return inv.fp32[0];
}

#define POOL_INV_CACHE_SIZE 16

struct pool_inv_cache {
    uint32_t count[POOL_INV_CACHE_SIZE];
    float reciprocal[POOL_INV_CACHE_SIZE];
    uint32_t size;
};

static inline float pool_inv_count_cached(struct pool_inv_cache * cache, uint32_t count) {
    for (uint32_t i = 0; i < cache->size; ++i) {
        if (cache->count[i] == count) {
            return cache->reciprocal[i];
        }
    }

    const float reciprocal = pool_inv_count(count);
    if (cache->size < POOL_INV_CACHE_SIZE) {
        cache->count[cache->size] = count;
        cache->reciprocal[cache->size] = reciprocal;
        ++cache->size;
    }
    return reciprocal;
}

// Fast path: exact non-overlapping tiling (stride == kernel, no padding), kernel_x in {1,2}.
// Every window is guaranteed fully in-bounds, so this never needs boundary clamping.
static inline void pool_plane_hvx(
    const float * src, float * dst, const struct htp_pool_2d_kernel_params * p) {
    const bool is_max = (p->pool_op == HTP_POOL_MAX);
    const HVX_Vector scale = hvx_vec_splat_f32(p->inv_kernel_area);
    const HVX_Vector seed  = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    const uint32_t nvec = p->dst_x / VLEN_FP32;
    for (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        const uint32_t sy = oy * p->kernel_y;
        for (uint32_t vx = 0; vx < nvec; ++vx) {
            const uint32_t ox = vx * VLEN_FP32;
            HVX_Vector acc = seed;
            for (uint32_t ky = 0; ky < p->kernel_y; ++ky) {
                const float * row = src + (sy + ky) * p->src_x;
                if (p->kernel_x == 2) {
                    const HVX_Vector v0 = *(const HVX_UVector *) (row + ox * 2);
                    const HVX_Vector v1 = *(const HVX_UVector *) (row + ox * 2 + VLEN_FP32);
                    const HVX_VectorPair deinterleaved = Q6_W_vdeal_VVR(v1, v0, -4);
                    const HVX_Vector lo = Q6_V_lo_W(deinterleaved);
                    const HVX_Vector hi = Q6_V_hi_W(deinterleaved);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, Q6_Vsf_vmax_VsfVsf(lo, hi))
                                 : hvx_vec_add_f32_f32(acc, hvx_vec_add_f32_f32(lo, hi));
                } else if (p->kernel_x == 1) {
                    const HVX_Vector v = *(const HVX_UVector *) (row + ox);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v) : hvx_vec_add_f32_f32(acc, v);
                }
            }
            hvx_vec_store_u(dst + oy * p->dst_x + ox, VLEN, is_max ? acc : hvx_vec_mul_f32_f32(acc, scale));
        }

        for (uint32_t ox = nvec * VLEN_FP32; ox < p->dst_x; ++ox) {
            const uint32_t sx = ox * p->kernel_x;
            float acc = is_max ? -FLT_MAX : 0.0f;
            for (uint32_t ky = 0; ky < p->kernel_y; ++ky) {
                const float * row = src + (sy + ky) * p->src_x + sx;
                for (uint32_t kx = 0; kx < p->kernel_x; ++kx) {
                    acc = is_max ? MAX(acc, row[kx]) : acc + row[kx];
                }
            }
            dst[oy * p->dst_x + ox] = is_max ? acc : acc * p->inv_kernel_area;
        }
    }
}

// Narrow exact-tiling path. The input is staged in VTCM with one vector of
// guard space, so full-width loads are safe even when src_x is below 32.
static inline void pool_plane_hvx_narrow(
    const float * src, float * dst, const struct htp_pool_2d_kernel_params * p) {
    const bool is_max = (p->pool_op == HTP_POOL_MAX);
    const HVX_Vector scale = hvx_vec_splat_f32(p->inv_kernel_area);
    const HVX_Vector seed  = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    for (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        const uint32_t sy = oy * p->kernel_y;
        HVX_Vector acc = seed;
        for (uint32_t ky = 0; ky < p->kernel_y; ++ky) {
            const float * row = src + (sy + ky) * p->src_x;
            const HVX_Vector v0 = *(const HVX_UVector *) row;
            HVX_Vector v = v0;
            if (p->kernel_x == 2) {
                // The second vector is zero because a narrow row has fewer
                // than 32 input elements. The low lanes still contain the
                // complete even/odd pairs needed by the output.
                const HVX_Vector zero = Q6_V_vsplat_R(0);
                const HVX_VectorPair deinterleaved = Q6_W_vdeal_VVR(zero, v0, -4);
                const HVX_Vector even = Q6_V_lo_W(deinterleaved);
                const HVX_Vector odd  = Q6_V_hi_W(deinterleaved);
                v = is_max ? Q6_Vsf_vmax_VsfVsf(even, odd)
                           : hvx_vec_add_f32_f32(even, odd);
            }
            acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v)
                         : hvx_vec_add_f32_f32(acc, v);
        }
        hvx_vec_store_u(dst + oy * p->dst_x, p->dst_x * sizeof(float),
                        is_max ? acc : hvx_vec_mul_f32_f32(acc, scale));
    }
}

static inline void pool_plane_global(
    const float * src, float * dst, const struct htp_pool_2d_kernel_params * p) {
    const uint32_t n = p->src_x * p->src_y;
    if (p->pool_op == HTP_POOL_MAX) {
        dst[0] = hvx_reduce_max_f32((const uint8_t *) src, n);
    } else {
        dst[0] = hvx_reduce_sum_f32((const uint8_t *) src, n) * p->inv_kernel_area;
    }
}

static inline void pool_plane_block(
    const float * src, float * dst, const struct htp_pool_2d_kernel_params * p) {
    for (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        const float * row = src + oy * p->src_x;
        float * dst_row = dst + oy * p->dst_x;
        for (uint32_t ox = 0; ox < p->dst_x; ++ox) {
            const uint8_t * block = (const uint8_t *) (row + ox * p->kernel_x);
            if (p->pool_op == HTP_POOL_MAX) {
                dst_row[ox] = hvx_reduce_max_f32(block, p->kernel_x);
            } else {
                dst_row[ox] = hvx_reduce_sum_f32(block, p->kernel_x) * p->inv_kernel_area;
            }
        }
    }
}

// Generic exact-tiling path for kernels that do not have a lane-shuffle fast path.
// Gather one source element per output lane, including narrow output rows. The
// gather window is rounded up to one HVX vector because the instruction requires
// a full-width address span even when the valid source row is shorter.
static inline void pool_plane_exact(
    const float * src, float * dst, const struct htp_pool_2d_kernel_params * p) {
    const bool is_max = (p->pool_op == HTP_POOL_MAX);
    const HVX_Vector scale = hvx_vec_splat_f32(p->inv_kernel_area);
    const HVX_Vector seed  = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);
    const uint32_t row_bytes = p->src_x * sizeof(float);
    // ARMVw gather is not reliable on V73.  The scalar packing fallback below
    // still feeds the reduction through HVX and is used there instead.
    const bool use_gather = p->src_x >= VLEN_FP32 && __HVX_ARCH__ >= 75;
    const uint32_t gather_span = row_bytes < VLEN ? VLEN : row_bytes;
    int32_t offsets[VLEN_FP32] __attribute__((aligned(VLEN)));
    for (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        const uint32_t sy = oy * p->kernel_y;
        for (uint32_t ox = 0; ox < p->dst_x; ox += VLEN_FP32) {
            const uint32_t lanes = MIN(VLEN_FP32, p->dst_x - ox);
            for (uint32_t lane = 0; lane < VLEN_FP32; ++lane) {
                offsets[lane] = lane < lanes ?
                    (int32_t) (lane * p->kernel_x * sizeof(float)) : 0;
            }
            const HVX_Vector gather_offsets = *(const HVX_UVector *) offsets;
            HVX_Vector acc = seed;

            for (uint32_t ky = 0; ky < p->kernel_y; ++ky) {
                const float * row = src + (sy + ky) * p->src_x + ox * p->kernel_x;
                for (uint32_t kx = 0; kx < p->kernel_x; ++kx) {
                    HVX_Vector gathered;
                    if (use_gather) {
                        Q6_vgather_ARMVw(&gathered, (size_t) (row + kx), gather_span, gather_offsets);
                    } else {
                        HVX_VectorAlias packed;
                        for (uint32_t lane = 0; lane < lanes; ++lane) {
                            packed.fp32[lane] = row[lane * p->kernel_x + kx];
                        }
                        for (uint32_t lane = lanes; lane < VLEN_FP32; ++lane) {
                            packed.fp32[lane] = is_max ? -FLT_MAX : 0.0f;
                        }
                        gathered = packed.v;
                    }

                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, gathered)
                                 : hvx_vec_add_f32_f32(acc, gathered);
                }
            }

            hvx_vec_store_u(dst + oy * p->dst_x + ox, lanes * sizeof(float),
                            is_max ? acc : hvx_vec_mul_f32_f32(acc, scale));
        }
    }
}

// Narrow general path. Address calculation and bounds checks are scalar, but all
// valid values for a batch of output columns are accumulated in HVX lanes. This
// keeps small padded/overlapping cases on HTP without unsafe narrow-row gathers.
static inline void pool_plane_general_narrow(
    const float * src, float * dst, const struct htp_pool_2d_kernel_params * p) {
    const bool is_max = (p->pool_op == HTP_POOL_MAX);
    const HVX_Vector scale = hvx_vec_splat_f32(p->inv_kernel_area);
    const HVX_Vector seed  = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    for (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        const int32_t iy0 = (int32_t) (oy * p->stride_y) - p->pad_y;
        for (uint32_t ox = 0; ox < p->dst_x; ox += VLEN_FP32) {
            const uint32_t lanes = MIN(VLEN_FP32, p->dst_x - ox);
            HVX_Vector acc = seed;

            for (uint32_t ky = 0; ky < p->kernel_y; ++ky) {
                const int32_t iy = iy0 + (int32_t) ky;
                for (uint32_t kx = 0; kx < p->kernel_x; ++kx) {
                    HVX_VectorAlias packed;
                    for (uint32_t lane = 0; lane < lanes; ++lane) {
                        const int32_t ix = (int32_t) ((ox + lane) * p->stride_x) - p->pad_x + (int32_t) kx;
                        packed.fp32[lane] = (iy >= 0 && iy < (int32_t) p->src_y &&
                                             ix >= 0 && ix < (int32_t) p->src_x) ?
                            src[iy * p->src_x + ix] : (is_max ? -FLT_MAX : 0.0f);
                    }
                    for (uint32_t lane = lanes; lane < VLEN_FP32; ++lane) {
                        packed.fp32[lane] = is_max ? -FLT_MAX : 0.0f;
                    }

                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, packed.v)
                                 : hvx_vec_add_f32_f32(acc, packed.v);
                }
            }

            hvx_vec_store_u(dst + oy * p->dst_x + ox, lanes * sizeof(float),
                            is_max ? acc : hvx_vec_mul_f32_f32(acc, scale));
        }
    }
}

// General path: arbitrary kernel/stride/padding

// Vertical padding is uniform across a whole output row (iy0 depends only on oy, not ox),
// so it collapses to one valid-ky range per row instead of a per-element check.
static inline void pool_row_bounds_y(
    const struct htp_pool_2d_kernel_params * p, uint32_t oy, int32_t * iy0, uint32_t * ky_lo, uint32_t * ky_hi) {
    *iy0 = (int32_t) (oy * p->stride_y) - p->pad_y;
    const int32_t lo = -(*iy0);
    const int32_t hi = (int32_t) p->src_y - *iy0;
    *ky_lo = (uint32_t) MAX(0, lo);
    *ky_hi = (uint32_t) MAX(0, MIN((int32_t) p->kernel_y, hi));
}

static inline void pool_row_general_scalar(
    const float * src, float * dst_row, const struct htp_pool_2d_kernel_params * p,
    uint32_t ox_start, uint32_t ox_end, int32_t iy0, uint32_t ky_lo, uint32_t ky_hi, bool is_max,
    struct pool_inv_cache * inv_cache) {
    for (uint32_t ox = ox_start; ox < ox_end; ++ox) {
        const int32_t ix0 = (int32_t) (ox * p->stride_x) - p->pad_x;
        float acc = is_max ? -FLT_MAX : 0.0f;
        for (uint32_t ky = ky_lo; ky < ky_hi; ++ky) {
            const float * row = src + (uint32_t) (iy0 + (int32_t) ky) * p->src_x;
            for (uint32_t kx = 0; kx < p->kernel_x; ++kx) {
                const int32_t ix = ix0 + (int32_t) kx;
                if (ix < 0 || ix >= (int32_t) p->src_x) {
                    continue;
                }
                acc = is_max ? MAX(acc, row[ix]) : acc + row[ix];
            }
        }
        if (is_max) {
            dst_row[ox] = acc;
        } else if (p->avg_divide_count) {
            const int32_t ix0 = (int32_t) (ox * p->stride_x) - p->pad_x;
            int count = 0;
            for (uint32_t kx = 0; kx < p->kernel_x; ++kx) {
                const int32_t ix = ix0 + (int32_t) kx;
                if (ix >= 0 && ix < (int32_t) p->src_x) {
                    ++count;
                }
            }
            dst_row[ox] = count > 0 ? acc * pool_inv_count_cached(inv_cache, (uint32_t) count) : 0.0f;
        } else {
            dst_row[ox] = acc * p->inv_kernel_area;
        }
    }
}

// Vectorized interior loop. Callers guarantee every kx in [0, kernel_x) is in-bounds
// for every ox in [ox_start, ox_end), so no per-lane masking is needed.
static inline void pool_row_general_vec(
    const float * src, float * dst_row, const struct htp_pool_2d_kernel_params * p,
    uint32_t ox_start, uint32_t ox_end, int32_t iy0, uint32_t ky_lo, uint32_t ky_hi, bool is_max,
    struct pool_inv_cache * inv_cache) {
    const HVX_Vector scale = hvx_vec_splat_f32(p->inv_kernel_area);
    const HVX_Vector seed  = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    uint32_t ox = ox_start;
    for (; ox + VLEN_FP32 <= ox_end; ox += VLEN_FP32) {
        const int32_t ix0 = (int32_t) (ox * p->stride_x) - p->pad_x;
        HVX_Vector acc = seed;
        for (uint32_t ky = ky_lo; ky < ky_hi; ++ky) {
            const float * row = src + (uint32_t) (iy0 + (int32_t) ky) * p->src_x;
            if (p->stride_x == 1) {
                // Consecutive ox map to consecutive columns: one unaligned load per kx.
                for (uint32_t kx = 0; kx < p->kernel_x; ++kx) {
                    const HVX_Vector v = *(const HVX_UVector *) (row + ix0 + (int32_t) kx);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v) : hvx_vec_add_f32_f32(acc, v);
                }
            } else if (p->stride_x == 2) {
                // Consecutive ox are 2 columns apart: deinterleave gives both kx and kx+1
                // in one shot, same trick as the fast path's kernel_x==2 case, generalized.
                for (uint32_t kx = 0; kx < p->kernel_x; kx += 2) {
                    const HVX_Vector v0 = *(const HVX_UVector *) (row + ix0 + (int32_t) kx);
                    const HVX_Vector v1 = *(const HVX_UVector *) (row + ix0 + (int32_t) kx + VLEN_FP32);
                    const HVX_VectorPair deinterleaved = Q6_W_vdeal_VVR(v1, v0, -4);
                    const HVX_Vector lo = Q6_V_lo_W(deinterleaved);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, lo) : hvx_vec_add_f32_f32(acc, lo);
                    if (kx + 1 < p->kernel_x) {
                        const HVX_Vector hi = Q6_V_hi_W(deinterleaved);
                        acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, hi) : hvx_vec_add_f32_f32(acc, hi);
                    }
                }
            } else {
                for (uint32_t kx = 0; kx < p->kernel_x; ++kx) {
                    HVX_VectorAlias packed;
                    for (uint32_t lane = 0; lane < VLEN_FP32; ++lane) {
                        packed.fp32[lane] = row[ix0 + (int32_t) (lane * p->stride_x) + (int32_t) kx];
                    }
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, packed.v)
                                 : hvx_vec_add_f32_f32(acc, packed.v);
                }
            }
        }
        hvx_vec_store_u(dst_row + ox, VLEN, is_max ? acc : hvx_vec_mul_f32_f32(acc, scale));
    }
    if (ox < ox_end) {
        pool_row_general_scalar(src, dst_row, p, ox, ox_end, iy0, ky_lo, ky_hi, is_max, inv_cache);
    }
}

static inline void pool_plane_general(
    const float * src, float * dst, const struct htp_pool_2d_kernel_params * p) {
    const bool is_max = (p->pool_op == HTP_POOL_MAX);

    // Compute the in-bounds x-interior for vectorized pooling.
    // x-strides other than 1 and 2 use the scalar path.
    uint32_t ox_lo, ox_hi;
    htp_pool2d_interior_range(p->src_x, p->dst_x, p->kernel_x, p->stride_x, p->pad_x, &ox_lo, &ox_hi);

    struct pool_inv_cache inv_cache = { 0 };

    for (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        int32_t iy0;
        uint32_t ky_lo, ky_hi;
        pool_row_bounds_y(p, oy, &iy0, &ky_lo, &ky_hi);
        float * dst_row = dst + oy * p->dst_x;

        inv_cache.size = 0;
        pool_row_general_scalar(src, dst_row, p, 0, ox_lo, iy0, ky_lo, ky_hi, is_max, &inv_cache);
        pool_row_general_vec(src, dst_row, p, ox_lo, ox_hi, iy0, ky_lo, ky_hi, is_max, &inv_cache);
        pool_row_general_scalar(src, dst_row, p, ox_hi, p->dst_x, iy0, ky_lo, ky_hi, is_max, &inv_cache);
    }
}

static inline void pool_plane(const float * src, float * dst, const struct htp_pool_2d_kernel_params * p) {
    if (p->global_path) {
        pool_plane_global(src, dst, p);
    } else if (p->block_path) {
        pool_plane_block(src, dst, p);
    } else if (p->narrow_path) {
        pool_plane_hvx_narrow(src, dst, p);
    } else if (p->fast_path) {
        pool_plane_hvx(src, dst, p);
    } else if (p->narrow_general_path && !p->avg_divide_count) {
        pool_plane_general_narrow(src, dst, p);
    } else if (p->exact_path) {
        pool_plane_exact(src, dst, p);
    } else {
        pool_plane_general(src, dst, p);
    }
}

struct pool_2d_context {
    struct htp_ops_context * octx;
    const struct htp_pool_2d_kernel_params * kparams;
    uint32_t plane_start;
    uint32_t plane_count;
};

static void pool_2d_thread(unsigned int nth, unsigned int ith, void * data) {
    struct pool_2d_context * ctx = (struct pool_2d_context *) data;
    const struct htp_pool_2d_kernel_params * p = ctx->kparams;
    const struct htp_tensor * src0 = ctx->octx->src[0];
    const struct htp_tensor * dst = ctx->octx->dst;
    const uint32_t first = ctx->plane_start + ith * p->planes_per_thread;
    const uint32_t last = MIN(first + p->planes_per_thread, ctx->plane_start + ctx->plane_count);

    if (first >= last) {
        return;
    }

    const uint8_t * src_data = (const uint8_t *) src0->data;
    uint8_t * dst_data = (uint8_t *) dst->data;
    struct htp_thread_trace * tr = &ctx->octx->ctx->trace[ith];

    if (!p->use_dma) {
        for (uint32_t plane = first; plane < last; ++plane) {
            const float * src_plane = (const float *) (src_data + plane * p->src_plane_bytes);
            float * dst_plane = (float *) (dst_data + plane * p->dst_plane_bytes);

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) plane);
            pool_plane(src_plane, dst_plane, p);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) plane);
        }

        FARF(HIGH, "pool2d-f32 %d/%d: %ux%ux%ux%u -> %ux%ux%ux%u (%u:%u)\n",
             ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
             dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], first, last);
        (void) nth;
        return;
    }

    dma_queue * dma_queue = ctx->octx->ctx->dma[ith];
    const size_t spad_per_thread = 2 * (p->src_plane_bytes_aligned + p->dst_plane_bytes_aligned);
    uint8_t * src_spad = ctx->octx->ctx->vtcm_base + ith * spad_per_thread;
    uint8_t * dst_spad = src_spad + 2 * p->src_plane_bytes_aligned;
    float * srcb2[2] = { (float *) src_spad, (float *) (src_spad + p->src_plane_bytes_aligned) };
    float * dstb2[2] = { (float *) dst_spad, (float *) (dst_spad + p->dst_plane_bytes_aligned) };

    const uint32_t total = last - first;

    if (total == 0) {
        return;
    }

    // Stage the first input before entering the overlapped pipeline.
    {
        const float * src_plane = (const float *) (src_data + (uint64_t) first * p->src_plane_bytes);
        dma_queue_push(dma_queue, dma_make_data(srcb2[0], src_plane),
                       p->src_plane_bytes_aligned, p->src_plane_bytes,
                       p->src_plane_bytes, 1);
        dma_queue_pop(dma_queue);
    }

    for (uint32_t i = 0; i < total; ++i) {
        const uint32_t plane = first + i;
        const uint32_t buf   = i & 1u;
        float * srcb = srcb2[buf];
        float * dstb = dstb2[buf];

        // Transfers are queued in this order for each plane:
        // next input, current output
        // Before reusing a slot, wait for the older output and current input
        // in FIFO order. This keeps DMA and compute overlapped.
        if (i > 1) {
            dma_queue_pop(dma_queue); // output from plane i - 2
        }

        if (i > 0) {
            dma_queue_pop(dma_queue); // input for plane i
        }

        if (i + 1 < total) {
            const uint32_t nbuf = 1u - buf;
            const float * next_src_plane = (const float *) (src_data + (uint64_t) (plane + 1) * p->src_plane_bytes);
            dma_queue_push(dma_queue, dma_make_data(srcb2[nbuf], next_src_plane),
                           p->src_plane_bytes_aligned, p->src_plane_bytes,
                           p->src_plane_bytes, 1);
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) plane);
        pool_plane(srcb, dstb, p);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) plane);

        float * dst_plane = (float *) (dst_data + plane * p->dst_plane_bytes);
        dma_queue_push(dma_queue, dma_make_data(dst_plane, dstb),
                       p->dst_plane_bytes, p->dst_plane_bytes_aligned,
                       p->dst_plane_bytes, 1);
    }

    // The last one or two output transfers are still outstanding.
    dma_queue_pop(dma_queue);
    if (total > 1) {
        dma_queue_pop(dma_queue);
    }

    FARF(HIGH, "pool2d-f32-dma %d/%d: %ux%ux%ux%u -> %ux%ux%ux%u (%u:%u)\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], first, last);
    (void) nth;
}

int op_pool_2d(struct htp_ops_context * octx) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst = octx->dst;
    const int32_t * params = octx->op_params;
    const struct htp_pool_2d_kernel_params * p =
        (const struct htp_pool_2d_kernel_params *) octx->kernel_params;

    if (src0->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32 ||
        (params[0] != HTP_POOL_AVG && params[0] != HTP_POOL_MAX)) {
        return HTP_STATUS_NO_SUPPORT;
    }
    uint32_t plane_start = 0;
    uint32_t plane_count = (octx->op == HTP_OP_POOL_1D)
        ? src0->ne[1] * src0->ne[2] * src0->ne[3]
        : src0->ne[2] * src0->ne[3];
    if (octx->ctx->mdev.count > 1) {
        const uint32_t planes_per_chunk = (p->dst_plane_bytes > 0) ?
            (HEX_L2_LINE_SIZE / hex_gcd_u32(p->dst_plane_bytes, HEX_L2_LINE_SIZE)) : 1;
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            plane_count, htp_tensor_mdev_data_aligned(dst) ? planes_per_chunk : 0,
            octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        plane_start = range.start;
        plane_count = range.count;
    }
    if (plane_count == 0) {
        return HTP_STATUS_OK;
    }

    struct pool_2d_context ctx = {
        .octx = octx,
        .kparams = p,
        .plane_start = plane_start,
        .plane_count = plane_count,
    };
    work_queue_run(octx->ctx->work_queue, pool_2d_thread, &ctx, octx->n_threads);
    return HTP_STATUS_OK;
}

int op_pool_1d(struct htp_ops_context * octx) {
    return op_pool_2d(octx);
}

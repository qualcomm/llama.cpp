#pragma clang diagnostic ignored "-Wunused-variable"

#include <float.h>
#include <math.h>
#include <HAP_farf.h>

#include "hex-common.h"
#include "dma-queue.h"
#include "hex-profile.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "hvx-types.h"
#include "hvx-base.h"
#include "hvx-utils.h"
#include "upscale-ops.h"

static inline int32_t clamp_i32(int32_t v, int32_t lo, int32_t hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static inline float clamp_f32(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static inline float bicubic_weight1(float x) {
    const float a = -0.75f;
    return ((a + 2.0f) * x - (a + 3.0f)) * x * x + 1.0f;
}

static inline float bicubic_weight2(float x) {
    const float a = -0.75f;
    return ((a * x - 5.0f * a) * x + 8.0f * a) * x - 4.0f * a;
}

static inline void bicubic_weights(float x, float w[4]) {
    w[0] = bicubic_weight2(x + 1.0f);
    w[1] = bicubic_weight1(x);
    w[2] = bicubic_weight1(1.0f - x);
    w[3] = bicubic_weight2(2.0f - x);
}

static void upscale_init_luts_nearest(
    uint32_t * lut_x,
    uint32_t * lut_y,
    const struct htp_tensor * src0,
    const struct htp_tensor * dst,
    const struct htp_upscale_kernel_params * p
) {
    const uint32_t src_x = src0->ne[0];
    const uint32_t src_y = src0->ne[1];
    const uint32_t dst_x = dst->ne[0];
    const uint32_t dst_y = dst->ne[1];

    for (uint32_t ox = 0; ox < dst_x; ++ox) {
        uint32_t x0 = (uint32_t) ((float) ox * p->inv_sf0);
        lut_x[ox] = MIN(x0, src_x - 1);
    }
    for (uint32_t oy = 0; oy < dst_y; ++oy) {
        uint32_t y0 = (uint32_t) ((float) oy * p->inv_sf1);
        lut_y[oy] = MIN(y0, src_y - 1);
    }
}

static void upscale_init_luts_bilinear(
    struct upscale_bilinear_x * lut_x,
    struct upscale_bilinear_y * lut_y,
    const struct htp_tensor * src0,
    const struct htp_tensor * dst,
    const struct htp_upscale_kernel_params * p
) {
    const uint32_t src_x = src0->ne[0];
    const uint32_t src_y = src0->ne[1];
    const uint32_t dst_x = dst->ne[0];
    const uint32_t dst_y = dst->ne[1];

    for (uint32_t ox = 0; ox < dst_x; ++ox) {
        float xf = ((float) ox + p->pixel_offset) * p->inv_sf0 - p->pixel_offset;
        int32_t x0 = (int32_t) floorf(xf);
        int32_t x1 = x0 + 1;
        x0 = clamp_i32(x0, 0, (int32_t) src_x - 1);
        x1 = clamp_i32(x1, 0, (int32_t) src_x - 1);
        float dx = clamp_f32(xf - (float) x0, 0.0f, 1.0f);

        lut_x[ox].x0 = (uint32_t) x0;
        lut_x[ox].x1 = (uint32_t) x1;
        lut_x[ox].dx = dx;
        lut_x[ox].w0 = 1.0f - dx;
    }
    for (uint32_t oy = 0; oy < dst_y; ++oy) {
        float yf = ((float) oy + p->pixel_offset) * p->inv_sf1 - p->pixel_offset;
        int32_t y0 = (int32_t) floorf(yf);
        int32_t y1 = y0 + 1;
        y0 = clamp_i32(y0, 0, (int32_t) src_y - 1);
        y1 = clamp_i32(y1, 0, (int32_t) src_y - 1);
        float dy = clamp_f32(yf - (float) y0, 0.0f, 1.0f);

        lut_y[oy].y0  = (uint32_t) y0;
        lut_y[oy].y1  = (uint32_t) y1;
        lut_y[oy].wy1 = dy;
        lut_y[oy].wy0 = 1.0f - dy;
    }
}

static void upscale_init_luts_bicubic(
    struct upscale_bicubic_x * lut_x,
    struct upscale_bicubic_y * lut_y,
    const struct htp_tensor * src0,
    const struct htp_tensor * dst,
    const struct htp_upscale_kernel_params * p
) {
    const uint32_t src_x = src0->ne[0];
    const uint32_t src_y = src0->ne[1];
    const uint32_t dst_x = dst->ne[0];
    const uint32_t dst_y = dst->ne[1];

    for (uint32_t ox = 0; ox < dst_x; ++ox) {
        float xf = ((float) ox + p->pixel_offset) * p->inv_sf0 - p->pixel_offset;
        int32_t x0 = (int32_t) floorf(xf);
        float dx = xf - (float) x0;
        bicubic_weights(dx, lut_x[ox].w);

        lut_x[ox].x[0] = clamp_i32(x0 - 1, 0, (int32_t) src_x - 1);
        lut_x[ox].x[1] = clamp_i32(x0 + 0, 0, (int32_t) src_x - 1);
        lut_x[ox].x[2] = clamp_i32(x0 + 1, 0, (int32_t) src_x - 1);
        lut_x[ox].x[3] = clamp_i32(x0 + 2, 0, (int32_t) src_x - 1);
    }
    for (uint32_t oy = 0; oy < dst_y; ++oy) {
        float yf = ((float) oy + p->pixel_offset) * p->inv_sf1 - p->pixel_offset;
        int32_t y0 = (int32_t) floorf(yf);
        float dy = yf - (float) y0;
        bicubic_weights(dy, lut_y[oy].w);

        lut_y[oy].y[0] = clamp_i32(y0 - 1, 0, (int32_t) src_y - 1);
        lut_y[oy].y[1] = clamp_i32(y0 + 0, 0, (int32_t) src_y - 1);
        lut_y[oy].y[2] = clamp_i32(y0 + 1, 0, (int32_t) src_y - 1);
        lut_y[oy].y[3] = clamp_i32(y0 + 2, 0, (int32_t) src_y - 1);
    }
}

static void upscale_init_luts_antialias(
    struct upscale_antialias_x * lut_x,
    struct upscale_antialias_y * lut_y,
    const struct htp_tensor * src0,
    const struct htp_tensor * dst,
    const struct htp_upscale_kernel_params * p
) {
    const uint32_t src_x = src0->ne[0];
    const uint32_t src_y = src0->ne[1];
    const uint32_t dst_x = dst->ne[0];
    const uint32_t dst_y = dst->ne[1];

    const float support0 = p->support0;
    const float support1 = p->support1;

    for (uint32_t ox = 0; ox < dst_x; ++ox) {
        const float xf = ((float) ox + p->pixel_offset) * p->inv_sf0;
        int32_t x_min = (int32_t) MAX(xf - support0 + p->pixel_offset, 0.0f);
        int32_t x_max = (int32_t) MIN(xf + support0 + p->pixel_offset, (float) src_x);
        lut_x[ox].x_min = clamp_i32(x_min, 0, (int32_t) src_x);
        lut_x[ox].x_max = clamp_i32(x_max, 0, (int32_t) src_x);
        lut_x[ox].xf    = xf;
    }
    for (uint32_t oy = 0; oy < dst_y; ++oy) {
        const float yf = ((float) oy + p->pixel_offset) * p->inv_sf1;
        int32_t y_min = (int32_t) MAX(yf - support1 + p->pixel_offset, 0.0f);
        int32_t y_max = (int32_t) MIN(yf + support1 + p->pixel_offset, (float) src_y);
        lut_y[oy].y_min = clamp_i32(y_min, 0, (int32_t) src_y);
        lut_y[oy].y_max = clamp_i32(y_max, 0, (int32_t) src_y);
        lut_y[oy].yf    = yf;
    }
}

static inline void get_src_y_window(
    const struct htp_tensor * src0,
    const struct htp_tensor * dst,
    const struct htp_upscale_kernel_params * p,
    const void * lut_y,
    uint32_t dst_y_start,
    uint32_t dst_y_count,
    uint32_t * out_src_y_start,
    uint32_t * out_src_y_count
) {
    const uint32_t src_y = src0->ne[1];
    const uint32_t dst_y = dst->ne[1];

    if (dst_y_count == dst_y) {
        *out_src_y_start = 0;
        *out_src_y_count = src_y;
        return;
    }

    const uint32_t last_oy = dst_y_start + dst_y_count - 1;

    if (p->mode == HTP_UPSCALE_MODE_NEAREST) {
        const uint32_t * ly = (const uint32_t *) lut_y;
        const uint32_t y0 = ly[dst_y_start];
        const uint32_t y1 = ly[last_oy];
        *out_src_y_start = y0;
        *out_src_y_count = y1 - y0 + 1;
    } else if (p->mode == HTP_UPSCALE_MODE_BICUBIC) {
        const struct upscale_bicubic_y * ly = (const struct upscale_bicubic_y *) lut_y;
        const int32_t y0 = ly[dst_y_start].y[0];
        const int32_t y1 = ly[last_oy].y[3];
        *out_src_y_start = (uint32_t) y0;
        *out_src_y_count = (uint32_t) (y1 - y0 + 1);
    } else if (p->antialias) {
        const struct upscale_antialias_y * ly = (const struct upscale_antialias_y *) lut_y;
        const int32_t y0 = ly[dst_y_start].y_min;
        const int32_t y1 = ly[last_oy].y_max;
        *out_src_y_start = (uint32_t) y0;
        *out_src_y_count = (uint32_t) (y1 - y0);
    } else {
        const struct upscale_bilinear_y * ly = (const struct upscale_bilinear_y *) lut_y;
        const uint32_t y0 = ly[dst_y_start].y0;
        const uint32_t y1 = ly[last_oy].y1;
        *out_src_y_start = y0;
        *out_src_y_count = y1 - y0 + 1;
    }
}

static inline void resolve_dst_slice(
    const struct htp_tensor * dst,
    const struct htp_upscale_kernel_params * p,
    uint32_t chunk_idx,
    dma_addr_t * out_slice_dst_off,
    uint32_t * out_dst_ys,
    uint32_t * out_dst_yc
) {
    const uint32_t s_idx = fastdiv(chunk_idx, &p->div_chunks_per_slice);
    const uint32_t c_idx = chunk_idx - s_idx * p->chunks_per_slice;
    const uint32_t d_ys  = c_idx * p->dst_nrows_per_chunk;
    const uint32_t d_yc  = MIN(p->dst_nrows_per_chunk, dst->ne[1] - d_ys);
    const uint32_t i3    = fastdiv(s_idx, &p->div_dst_z);
    const uint32_t i2    = s_idx - i3 * dst->ne[2];

    *out_slice_dst_off = (dma_addr_t) i2 * dst->nb[2] + (dma_addr_t) i3 * dst->nb[3];
    *out_dst_ys        = d_ys;
    *out_dst_yc        = d_yc;
}

static inline void resolve_src_slice(
    const struct htp_tensor * src0,
    const struct htp_tensor * dst,
    const struct htp_upscale_kernel_params * p,
    const void * lut_y,
    uint32_t chunk_idx,
    dma_addr_t * out_slice_src_off,
    uint32_t * out_src_ys,
    uint32_t * out_src_yc
) {
    const uint32_t s_idx = fastdiv(chunk_idx, &p->div_chunks_per_slice);
    const uint32_t c_idx = chunk_idx - s_idx * p->chunks_per_slice;
    const uint32_t d_ys  = c_idx * p->dst_nrows_per_chunk;
    const uint32_t d_yc  = MIN(p->dst_nrows_per_chunk, dst->ne[1] - d_ys);
    uint32_t s_ys, s_yc;
    get_src_y_window(src0, dst, p, lut_y, d_ys, d_yc, &s_ys, &s_yc);

    const uint32_t i3  = fastdiv(s_idx, &p->div_dst_z);
    const uint32_t i2  = s_idx - i3 * dst->ne[2];
    const uint32_t i03 = (uint32_t) ((float) i3 * p->inv_sf3);
    const uint32_t i02 = (uint32_t) ((float) i2 * p->inv_sf2);

    *out_slice_src_off = (dma_addr_t) i02 * src0->nb[2] + (dma_addr_t) i03 * src0->nb[3];
    *out_src_ys        = s_ys;
    *out_src_yc        = s_yc;
}

static void compute_nearest_chunk(
    const float * srcb,
    float * dstb,
    const struct htp_tensor * src0,
    const struct htp_tensor * dst,
    const uint32_t * lut_x,
    const uint32_t * lut_y,
    uint32_t dst_y_start,
    uint32_t dst_y_count,
    uint32_t src_y_start
) {
    const uint32_t src_x = src0->ne[0];
    const uint32_t dst_x = dst->ne[0];

    for (uint32_t oy = dst_y_start; oy < dst_y_start + dst_y_count; ++oy) {
        const uint32_t y0 = lut_y[oy];
        const float * r0  = srcb + (y0 - src_y_start) * src_x;
        float * dst_row   = dstb + (oy - dst_y_start) * dst_x;

        for (uint32_t ox = 0; ox < dst_x; ox += VLEN_FP32) {
            const uint32_t rem = dst_x - ox;
            const uint32_t nbytes = (rem < VLEN_FP32) ? (rem * sizeof(float)) : VLEN;
            float tmp[VLEN_FP32] __attribute__((aligned(128)));
            const uint32_t cnt = rem < VLEN_FP32 ? rem : VLEN_FP32;
            for (uint32_t l = 0; l < cnt; ++l) {
                tmp[l] = r0[lut_x[ox + l]];
            }
            hvx_vec_store_u(dst_row + ox, nbytes, *(const HVX_Vector *) tmp);
        }
    }
}

static void compute_nearest_2x_chunk(
    const float * srcb,
    float * dstb,
    const struct htp_tensor * src0,
    const struct htp_tensor * dst,
    uint32_t dst_y_start,
    uint32_t dst_y_count,
    uint32_t src_y_start
) {
    const uint32_t src_x = src0->ne[0];
    const uint32_t dst_x = dst->ne[0];

    for (uint32_t oy = dst_y_start; oy < dst_y_start + dst_y_count; ++oy) {
        const uint32_t sy = oy >> 1;
        const float * r0  = srcb + (sy - src_y_start) * src_x;
        float * dst_row   = dstb + (oy - dst_y_start) * dst_x;

        for (uint32_t ix = 0; ix < src_x; ix += VLEN_FP32) {
            const HVX_Vector vin = hvx_vmemu(r0 + ix);

            const HVX_VectorPair p = Q6_W_vshuff_VVR(vin, vin, -4);
            const HVX_Vector v_lo = Q6_V_lo_W(p);
            const HVX_Vector v_hi = Q6_V_hi_W(p);

            const uint32_t ox0 = 2 * ix;
            const uint32_t rem_out0 = (dst_x > ox0) ? (dst_x - ox0) : 0;
            const uint32_t nbytes_out0 = (rem_out0 < VLEN_FP32) ? (rem_out0 * sizeof(float)) : VLEN;
            hvx_vec_store_u(dst_row + ox0, nbytes_out0, v_lo);

            const uint32_t ox1 = ox0 + VLEN_FP32;
            if (dst_x > ox1) {
                const uint32_t rem_out1 = dst_x - ox1;
                const uint32_t nbytes_out1 = (rem_out1 < VLEN_FP32) ? (rem_out1 * sizeof(float)) : VLEN;
                hvx_vec_store_u(dst_row + ox1, nbytes_out1, v_hi);
            }
        }
    }
}

static void compute_nearest_4x_chunk(
    const float * srcb,
    float * dstb,
    const struct htp_tensor * src0,
    const struct htp_tensor * dst,
    uint32_t dst_y_start,
    uint32_t dst_y_count,
    uint32_t src_y_start
) {
    const uint32_t src_x = src0->ne[0];
    const uint32_t dst_x = dst->ne[0];

    for (uint32_t oy = dst_y_start; oy < dst_y_start + dst_y_count; ++oy) {
        const uint32_t sy = oy >> 2;
        const float * r0  = srcb + (sy - src_y_start) * src_x;
        float * dst_row   = dstb + (oy - dst_y_start) * dst_x;

        for (uint32_t ix = 0; ix < src_x; ix += VLEN_FP32) {
            const HVX_Vector vin = hvx_vmemu(r0 + ix);

            const HVX_VectorPair p = Q6_W_vshuff_VVR(vin, vin, -4);
            const HVX_Vector v_lo = Q6_V_lo_W(p);
            const HVX_Vector v_hi = Q6_V_hi_W(p);

            const HVX_VectorPair p0 = Q6_W_vshuff_VVR(v_lo, v_lo, -4);
            const HVX_VectorPair p1 = Q6_W_vshuff_VVR(v_hi, v_hi, -4);

            const HVX_Vector out0 = Q6_V_lo_W(p0);
            const HVX_Vector out1 = Q6_V_hi_W(p0);
            const HVX_Vector out2 = Q6_V_lo_W(p1);
            const HVX_Vector out3 = Q6_V_hi_W(p1);

            const uint32_t ox = 4 * ix;
            if (dst_x > ox) {
                const uint32_t rem = dst_x - ox;
                hvx_vec_store_u(dst_row + ox, (rem < VLEN_FP32) ? (rem * sizeof(float)) : VLEN, out0);
            }
            if (dst_x > ox + VLEN_FP32) {
                const uint32_t rem = dst_x - (ox + VLEN_FP32);
                hvx_vec_store_u(dst_row + ox + VLEN_FP32, (rem < VLEN_FP32) ? (rem * sizeof(float)) : VLEN, out1);
            }
            if (dst_x > ox + 2 * VLEN_FP32) {
                const uint32_t rem = dst_x - (ox + 2 * VLEN_FP32);
                hvx_vec_store_u(dst_row + ox + 2 * VLEN_FP32, (rem < VLEN_FP32) ? (rem * sizeof(float)) : VLEN, out2);
            }
            if (dst_x > ox + 3 * VLEN_FP32) {
                const uint32_t rem = dst_x - (ox + 3 * VLEN_FP32);
                hvx_vec_store_u(dst_row + ox + 3 * VLEN_FP32, (rem < VLEN_FP32) ? (rem * sizeof(float)) : VLEN, out3);
            }
        }
    }
}

static void compute_bilinear_chunk(
    const float * srcb,
    float * dstb,
    const struct htp_tensor * src0,
    const struct htp_tensor * dst,
    const struct upscale_bilinear_x * lut_x,
    const struct upscale_bilinear_y * lut_y,
    uint32_t dst_y_start,
    uint32_t dst_y_count,
    uint32_t src_y_start
) {
    const uint32_t src_x = src0->ne[0];
    const uint32_t dst_x = dst->ne[0];

    for (uint32_t oy = dst_y_start; oy < dst_y_start + dst_y_count; ++oy) {
        const struct upscale_bilinear_y * ey = &lut_y[oy];
        const float * r0 = srcb + (ey->y0 - src_y_start) * src_x;
        const float * r1 = srcb + (ey->y1 - src_y_start) * src_x;
        float * dst_row  = dstb + (oy - dst_y_start) * dst_x;

        const HVX_Vector vwy0 = hvx_vec_splat_f32(ey->wy0);
        const HVX_Vector vwy1 = hvx_vec_splat_f32(ey->wy1);

        for (uint32_t ox = 0; ox < dst_x; ox += VLEN_FP32) {
            const uint32_t rem = dst_x - ox;
            const uint32_t nbytes = (rem < VLEN_FP32) ? (rem * sizeof(float)) : VLEN;

            float a[VLEN_FP32]  __attribute__((aligned(128)));
            float b[VLEN_FP32]  __attribute__((aligned(128)));
            float c[VLEN_FP32]  __attribute__((aligned(128)));
            float d[VLEN_FP32]  __attribute__((aligned(128)));
            float w0[VLEN_FP32] __attribute__((aligned(128)));
            float dx[VLEN_FP32] __attribute__((aligned(128)));

            const uint32_t cnt = rem < VLEN_FP32 ? rem : VLEN_FP32;
            for (uint32_t l = 0; l < cnt; ++l) {
                const struct upscale_bilinear_x * e = &lut_x[ox + l];
                a[l]  = r0[e->x0];
                b[l]  = r0[e->x1];
                c[l]  = r1[e->x0];
                d[l]  = r1[e->x1];
                w0[l] = e->w0;
                dx[l] = e->dx;
            }
            for (uint32_t l = cnt; l < VLEN_FP32; ++l) {
                a[l] = b[l] = c[l] = d[l] = w0[l] = dx[l] = 0.0f;
            }

            const HVX_Vector va  = *(const HVX_Vector *) a;
            const HVX_Vector vb  = *(const HVX_Vector *) b;
            const HVX_Vector vc  = *(const HVX_Vector *) c;
            const HVX_Vector vd  = *(const HVX_Vector *) d;
            const HVX_Vector vw0 = *(const HVX_Vector *) w0;
            const HVX_Vector vdx = *(const HVX_Vector *) dx;

            const HVX_Vector val0 = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(vw0, va), hvx_vec_mul_f32_f32(vdx, vb));
            const HVX_Vector val1 = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(vw0, vc), hvx_vec_mul_f32_f32(vdx, vd));
            const HVX_Vector val  = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(vwy0, val0), hvx_vec_mul_f32_f32(vwy1, val1));

            hvx_vec_store_u(dst_row + ox, nbytes, val);
        }
    }
}

static void compute_bilinear_2x_chunk(
    const float * srcb,
    float * dstb,
    const struct htp_tensor * src0,
    const struct htp_tensor * dst,
    const struct upscale_bilinear_y * lut_y,
    uint32_t dst_y_start,
    uint32_t dst_y_count,
    uint32_t src_y_start
) {
    const uint32_t src_x = src0->ne[0];
    const uint32_t dst_x = dst->ne[0];
    const HVX_Vector c_075 = hvx_vec_splat_f32(0.75f);
    const HVX_Vector c_025 = hvx_vec_splat_f32(0.25f);

    for (uint32_t oy = dst_y_start; oy < dst_y_start + dst_y_count; ++oy) {
        const struct upscale_bilinear_y * ey = &lut_y[oy];
        const float * r0 = srcb + (ey->y0 - src_y_start) * src_x;
        const float * r1 = srcb + (ey->y1 - src_y_start) * src_x;
        float * dst_row  = dstb + (oy - dst_y_start) * dst_x;

        const bool same_row = (ey->y0 == ey->y1);
        const float wy0 = ey->wy0;
        const float wy1 = ey->wy1;
        const HVX_Vector vwy0 = hvx_vec_splat_f32(wy0);
        const HVX_Vector vwy1 = hvx_vec_splat_f32(wy1);

        const float first_val = same_row ? r0[0] : (wy0 * r0[0] + wy1 * r1[0]);
        HVX_Vector v_prev = hvx_vec_splat_f32(first_val);

        HVX_Vector v_curr = hvx_vmemu(r0);
        if (!same_row) {
            const HVX_Vector v1 = hvx_vmemu(r1);
            v_curr = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(vwy0, v_curr), hvx_vec_mul_f32_f32(vwy1, v1));
        }

        for (uint32_t ix = 0; ix < src_x; ix += VLEN_FP32) {
            HVX_Vector v_next_chunk;
            const uint32_t next_ix = ix + VLEN_FP32;
            if (next_ix < src_x) {
                v_next_chunk = hvx_vmemu(r0 + next_ix);
                if (!same_row) {
                    const HVX_Vector v1 = hvx_vmemu(r1 + next_ix);
                    v_next_chunk = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(vwy0, v_next_chunk), hvx_vec_mul_f32_f32(vwy1, v1));
                }
            } else {
                const float last_val = same_row ? r0[src_x - 1] : (wy0 * r0[src_x - 1] + wy1 * r1[src_x - 1]);
                v_next_chunk = hvx_vec_splat_f32(last_val);
            }

            const HVX_Vector v_shl = Q6_V_valign_VVR(v_curr, v_prev, 124);
            const HVX_Vector v_shr = Q6_V_valign_VVR(v_next_chunk, v_curr, 4);

            const HVX_Vector ve = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(c_075, v_curr), hvx_vec_mul_f32_f32(c_025, v_shl));
            const HVX_Vector vo = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(c_075, v_curr), hvx_vec_mul_f32_f32(c_025, v_shr));

            const HVX_VectorPair p = Q6_W_vshuff_VVR(vo, ve, -4);
            const HVX_Vector out0 = Q6_V_lo_W(p);
            const HVX_Vector out1 = Q6_V_hi_W(p);

            const uint32_t ox0 = 2 * ix;
            hvx_vec_store_u(dst_row + ox0, VLEN, out0);
            hvx_vec_store_u(dst_row + ox0 + VLEN_FP32, VLEN, out1);

            v_prev = v_curr;
            v_curr = v_next_chunk;
        }
    }
}

static void compute_bilinear_antialias_chunk(
    const float * srcb,
    float * dstb,
    const struct htp_tensor * src0,
    const struct htp_tensor * dst,
    const struct htp_upscale_kernel_params * p,
    const struct upscale_antialias_x * lut_x,
    const struct upscale_antialias_y * lut_y,
    uint32_t dst_y_start,
    uint32_t dst_y_count,
    uint32_t src_y_start
) {
    const uint32_t src_x = src0->ne[0];
    const uint32_t dst_x = dst->ne[0];

    const float invscale1 = p->invscale1;
    const float invscale0 = p->invscale0;

    for (uint32_t oy = dst_y_start; oy < dst_y_start + dst_y_count; ++oy) {
        const struct upscale_antialias_y * ey = &lut_y[oy];
        const float yf = ey->yf;
        const int32_t y_min = ey->y_min;
        const int32_t y_max = ey->y_max;

        float * dst_row = dstb + (oy - dst_y_start) * dst_x;

        for (uint32_t ox = 0; ox < dst_x; ox += VLEN_FP32) {
            const uint32_t rem = dst_x - ox;
            const uint32_t nbytes = (rem < VLEN_FP32) ? (rem * sizeof(float)) : VLEN;
            float tmp[VLEN_FP32] __attribute__((aligned(128)));
            const uint32_t cnt = rem < VLEN_FP32 ? rem : VLEN_FP32;

            for (uint32_t l = 0; l < cnt; ++l) {
                const uint32_t cur_ox = ox + l;
                const struct upscale_antialias_x * ex = &lut_x[cur_ox];
                const float xf = ex->xf;
                const int32_t x_min = ex->x_min;
                const int32_t x_max = ex->x_max;

                float val = 0.0f;
                float total_weight = 0.0f;

                for (int32_t sy = y_min; sy < y_max; ++sy) {
                    const float diff_y = fabsf((float) sy - yf + p->pixel_offset) * invscale1;
                    const float weight_y = MAX(1.0f - diff_y, 0.0f);
                    if (weight_y <= 0.0f) {
                        continue;
                    }
                    const float * r = srcb + (uint32_t) (sy - (int32_t) src_y_start) * src_x;

                    for (int32_t sx = x_min; sx < x_max; ++sx) {
                        const float diff_x = fabsf((float) sx - xf + p->pixel_offset) * invscale0;
                        const float weight_x = MAX(1.0f - diff_x, 0.0f);
                        const float weight = weight_x * weight_y;
                        if (weight <= 0.0f) {
                            continue;
                        }
                        val += r[sx] * weight;
                        total_weight += weight;
                    }
                }

                tmp[l] = (total_weight > 0.0f) ? (val / total_weight) : 0.0f;
            }

            hvx_vec_store_u(dst_row + ox, nbytes, *(const HVX_Vector *) tmp);
        }
    }
}

static void compute_bicubic_chunk(
    const float * srcb,
    float * dstb,
    const struct htp_tensor * src0,
    const struct htp_tensor * dst,
    const struct upscale_bicubic_x * lut_x,
    const struct upscale_bicubic_y * lut_y,
    uint32_t dst_y_start,
    uint32_t dst_y_count,
    uint32_t src_y_start
) {
    const uint32_t src_x = src0->ne[0];
    const uint32_t dst_x = dst->ne[0];

    for (uint32_t oy = dst_y_start; oy < dst_y_start + dst_y_count; ++oy) {
        const struct upscale_bicubic_y * ey = &lut_y[oy];
        const float * r0 = srcb + (uint32_t) (ey->y[0] - (int32_t) src_y_start) * src_x;
        const float * r1 = srcb + (uint32_t) (ey->y[1] - (int32_t) src_y_start) * src_x;
        const float * r2 = srcb + (uint32_t) (ey->y[2] - (int32_t) src_y_start) * src_x;
        const float * r3 = srcb + (uint32_t) (ey->y[3] - (int32_t) src_y_start) * src_x;

        float * dst_row = dstb + (oy - dst_y_start) * dst_x;

        const HVX_Vector vwy0 = hvx_vec_splat_f32(ey->w[0]);
        const HVX_Vector vwy1 = hvx_vec_splat_f32(ey->w[1]);
        const HVX_Vector vwy2 = hvx_vec_splat_f32(ey->w[2]);
        const HVX_Vector vwy3 = hvx_vec_splat_f32(ey->w[3]);

        for (uint32_t ox = 0; ox < dst_x; ox += VLEN_FP32) {
            const uint32_t rem = dst_x - ox;
            const uint32_t nbytes = (rem < VLEN_FP32) ? (rem * sizeof(float)) : VLEN;

            float v0[VLEN_FP32] __attribute__((aligned(128)));
            float v1[VLEN_FP32] __attribute__((aligned(128)));
            float v2[VLEN_FP32] __attribute__((aligned(128)));
            float v3[VLEN_FP32] __attribute__((aligned(128)));

            const uint32_t cnt = rem < VLEN_FP32 ? rem : VLEN_FP32;
            for (uint32_t l = 0; l < cnt; ++l) {
                const struct upscale_bicubic_x * e = &lut_x[ox + l];
                v0[l] = r0[e->x[0]] * e->w[0] + r0[e->x[1]] * e->w[1] + r0[e->x[2]] * e->w[2] + r0[e->x[3]] * e->w[3];
                v1[l] = r1[e->x[0]] * e->w[0] + r1[e->x[1]] * e->w[1] + r1[e->x[2]] * e->w[2] + r1[e->x[3]] * e->w[3];
                v2[l] = r2[e->x[0]] * e->w[0] + r2[e->x[1]] * e->w[1] + r2[e->x[2]] * e->w[2] + r2[e->x[3]] * e->w[3];
                v3[l] = r3[e->x[0]] * e->w[0] + r3[e->x[1]] * e->w[1] + r3[e->x[2]] * e->w[2] + r3[e->x[3]] * e->w[3];
            }
            for (uint32_t l = cnt; l < VLEN_FP32; ++l) {
                v0[l] = v1[l] = v2[l] = v3[l] = 0.0f;
            }

            const HVX_Vector vv0 = *(const HVX_Vector *) v0;
            const HVX_Vector vv1 = *(const HVX_Vector *) v1;
            const HVX_Vector vv2 = *(const HVX_Vector *) v2;
            const HVX_Vector vv3 = *(const HVX_Vector *) v3;

            const HVX_Vector acc0 = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(vwy0, vv0), hvx_vec_mul_f32_f32(vwy1, vv1));
            const HVX_Vector acc1 = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(vwy2, vv2), hvx_vec_mul_f32_f32(vwy3, vv3));
            const HVX_Vector val  = hvx_vec_add_f32_f32(acc0, acc1);

            hvx_vec_store_u(dst_row + ox, nbytes, val);
        }
    }
}

struct upscale_context {
    struct htp_ops_context * octx;
    const struct htp_upscale_kernel_params * kparams;
    struct htp_upscale_vtcm_layout layout;
    const void * lut_x;
    const void * lut_y;
    uint32_t chunk_start;
    uint32_t chunk_count;
    uint32_t chunks_per_thread;
};

static void upscale_worker(unsigned int nth, unsigned int ith, void * data) {
    struct upscale_context * ctx = (struct upscale_context *) data;
    const struct htp_upscale_kernel_params * p = ctx->kparams;
    const struct htp_upscale_vtcm_layout * L = &ctx->layout;
    const struct htp_tensor * src0 = ctx->octx->src[0];
    const struct htp_tensor * dst  = ctx->octx->dst;

    const uint32_t first = ctx->chunk_start + ith * ctx->chunks_per_thread;
    const uint32_t last  = MIN(first + ctx->chunks_per_thread, ctx->chunk_start + ctx->chunk_count);

    if (first >= last) {
        return;
    }

    struct htp_thread_trace * tr = &ctx->octx->ctx->trace[ith];
    dma_queue * dma_q            = ctx->octx->ctx->dma[ith];

    const uint32_t src_spad_half = (uint32_t) L->src_spad_half_size;
    const uint32_t dst_spad_half = (uint32_t) L->dst_spad_half_size;

    uint8_t * vtcm_base = (uint8_t *) ctx->octx->ctx->vtcm_base;
    uint8_t * src_spad  = VTCM_LAYOUT_PTR(uint8_t, vtcm_base, L->off_src) + ith * L->src_bytes_per_thread;
    uint8_t * dst_spad  = VTCM_LAYOUT_PTR(uint8_t, vtcm_base, L->off_dst) + ith * L->dst_bytes_per_thread;
    const void * lut_x  = ctx->lut_x;
    const void * lut_y  = ctx->lut_y;

    float * srcb2[2] = { (float *) src_spad, (float *) (src_spad + src_spad_half) };
    float * dstb2[2] = { (float *) dst_spad, (float *) (dst_spad + dst_spad_half) };

    const uint32_t src_row_size = src0->ne[0] * sizeof(float);
    const uint32_t dst_row_size = dst->ne[0] * sizeof(float);
    const uint32_t src_stride   = src0->nb[1];
    const uint32_t dst_stride   = dst->nb[1];

    const uint32_t total = last - first;

    // Warmup: prime pipeline with initial transfers
    for (uint32_t i = 0; i < total && i < 2; ++i) {
        dma_queue_push(dma_q, dma_make_data(dst->data, dstb2[i]), dst_stride, dst_row_size, dst_row_size, 0);

        dma_addr_t slice_src_off;
        uint32_t src_ys, src_yc;
        resolve_src_slice(src0, dst, p, lut_y, first + i, &slice_src_off, &src_ys, &src_yc);

        const dma_addr_t src_addr = src0->data + slice_src_off + (dma_addr_t) src_ys * src_stride;
        dma_queue_push(dma_q, dma_make_data(srcb2[i], src_addr), src_row_size, src_stride, src_row_size, src_yc);
    }

    // Steady state loop
    for (uint32_t i = 0; i < total; ++i) {
        const uint32_t buf = i & 1u;
        float * srcb = srcb2[buf];
        float * dstb = dstb2[buf];

        dma_queue_pop(dma_q);
        dma_queue_pop(dma_q);

        dma_addr_t slice_dst_off;
        uint32_t dst_ys, dst_yc;
        resolve_dst_slice(dst, p, first + i, &slice_dst_off, &dst_ys, &dst_yc);

        uint32_t src_ys, src_yc;
        get_src_y_window(src0, dst, p, lut_y, dst_ys, dst_yc, &src_ys, &src_yc);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) (first + i));
        if (p->mode == HTP_UPSCALE_MODE_NEAREST) {
            if (dst->ne[0] == 2 * src0->ne[0] && dst->ne[1] == 2 * src0->ne[1]) {
                compute_nearest_2x_chunk(srcb, dstb, src0, dst, dst_ys, dst_yc, src_ys);
            } else if (dst->ne[0] == 4 * src0->ne[0] && dst->ne[1] == 4 * src0->ne[1]) {
                compute_nearest_4x_chunk(srcb, dstb, src0, dst, dst_ys, dst_yc, src_ys);
            } else {
                compute_nearest_chunk(srcb, dstb, src0, dst, (const uint32_t *) lut_x, (const uint32_t *) lut_y, dst_ys, dst_yc, src_ys);
            }
        } else if (p->mode == HTP_UPSCALE_MODE_BICUBIC) {
            compute_bicubic_chunk(srcb, dstb, src0, dst, (const struct upscale_bicubic_x *) lut_x, (const struct upscale_bicubic_y *) lut_y, dst_ys, dst_yc, src_ys);
        } else if (p->antialias) {
            compute_bilinear_antialias_chunk(srcb, dstb, src0, dst, p, (const struct upscale_antialias_x *) lut_x, (const struct upscale_antialias_y *) lut_y, dst_ys, dst_yc, src_ys);
        } else {
            if (dst->ne[0] == 2 * src0->ne[0] && dst->ne[1] == 2 * src0->ne[1] &&
                !p->align_corners && (src0->ne[0] % VLEN_FP32 == 0)) {
                compute_bilinear_2x_chunk(srcb, dstb, src0, dst, (const struct upscale_bilinear_y *) lut_y, dst_ys, dst_yc, src_ys);
            } else {
                compute_bilinear_chunk(srcb, dstb, src0, dst, (const struct upscale_bilinear_x *) lut_x, (const struct upscale_bilinear_y *) lut_y, dst_ys, dst_yc, src_ys);
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) (first + i));

        const dma_addr_t dst_addr = dst->data + slice_dst_off + (dma_addr_t) dst_ys * dst_stride;
        dma_queue_push(dma_q, dma_make_data(dst_addr, dstb), dst_stride, dst_row_size, dst_row_size, dst_yc);

        const uint32_t next = i + 2;
        if (next < total) {
            dma_addr_t next_src_off;
            uint32_t next_src_ys, next_src_yc;
            resolve_src_slice(src0, dst, p, lut_y, first + next, &next_src_off, &next_src_ys, &next_src_yc);

            const dma_addr_t next_src_addr = src0->data + next_src_off + (dma_addr_t) next_src_ys * src_stride;
            dma_queue_push(dma_q, dma_make_data(srcb, next_src_addr), src_row_size, src_stride, src_row_size, next_src_yc);
        }
    }

    dma_queue_flush(dma_q);

    (void) nth;
}

int op_upscale(struct htp_ops_context * octx) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;
    const struct htp_upscale_kernel_params * p =
        (const struct htp_upscale_kernel_params *) octx->kernel_params;

    const uint32_t src_x = src0->ne[0];
    const uint32_t src_y = src0->ne[1];
    const uint32_t dst_x = dst->ne[0];
    const uint32_t dst_y = dst->ne[1];

    uint32_t chunk_start = 0;
    uint32_t chunk_count = p->total_chunks;

    if (octx->ctx->mdev.count > 1) {
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            chunk_count, htp_tensor_mdev_data_aligned(dst) ? p->slices_per_chunk : 0,
            octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        chunk_start = range.start;
        chunk_count = range.count;
    }

    if (chunk_count == 0) {
        return HTP_STATUS_OK;
    }

    const uint32_t n_threads = MIN(p->n_threads, chunk_count);
    if (!htp_ops_context_set_n_threads(octx, n_threads)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    const uint32_t chunks_per_thread = fastdiv(chunk_count + n_threads - 1, &octx->n_threads_div);

    size_t elem_x_sz = sizeof(uint32_t);
    size_t elem_y_sz = sizeof(uint32_t);

    if (p->mode == HTP_UPSCALE_MODE_BICUBIC) {
        elem_x_sz = sizeof(struct upscale_bicubic_x);
        elem_y_sz = sizeof(struct upscale_bicubic_y);
    } else if (p->antialias) {
        elem_x_sz = sizeof(struct upscale_antialias_x);
        elem_y_sz = sizeof(struct upscale_antialias_y);
    } else if (p->mode == HTP_UPSCALE_MODE_BILINEAR) {
        elem_x_sz = sizeof(struct upscale_bilinear_x);
        elem_y_sz = sizeof(struct upscale_bilinear_y);
    }

    const size_t lut_x_size = hex_round_up(dst_x * elem_x_sz, 128);
    const size_t lut_y_size = hex_round_up(dst_y * elem_y_sz, 128);
    const size_t total_lut_size = lut_x_size + lut_y_size;

    void * base_lut  = NULL;
    void * alloc_lut = NULL;

    if (octx->ctx->ddr_spad_base && total_lut_size <= octx->ctx->ddr_spad_size) {
        base_lut = octx->ctx->ddr_spad_base;
    } else {
        alloc_lut = memalign(128, total_lut_size);
        base_lut  = alloc_lut;
    }

    void * lut_x = base_lut;
    void * lut_y = (uint8_t *) base_lut + lut_x_size;

    if (p->mode == HTP_UPSCALE_MODE_NEAREST) {
        upscale_init_luts_nearest((uint32_t *) lut_x, (uint32_t *) lut_y, src0, dst, p);
    } else if (p->mode == HTP_UPSCALE_MODE_BICUBIC) {
        upscale_init_luts_bicubic((struct upscale_bicubic_x *) lut_x, (struct upscale_bicubic_y *) lut_y, src0, dst, p);
    } else if (p->antialias) {
        upscale_init_luts_antialias((struct upscale_antialias_x *) lut_x, (struct upscale_antialias_y *) lut_y, src0, dst, p);
    } else {
        upscale_init_luts_bilinear((struct upscale_bilinear_x *) lut_x, (struct upscale_bilinear_y *) lut_y, src0, dst, p);
    }

    struct upscale_context ctx = {
        .octx              = octx,
        .kparams           = p,
        .lut_x             = lut_x,
        .lut_y             = lut_y,
        .chunk_start       = chunk_start,
        .chunk_count       = chunk_count,
        .chunks_per_thread = chunks_per_thread,
    };

    htp_upscale_vtcm_layout_build(&ctx.layout, src_x, src_y, dst_x, dst_y,
                                  p->dst_nrows_per_chunk, p->inv_sf1, n_threads);

    work_queue_run(octx->ctx->work_queue, upscale_worker, &ctx, n_threads);

    if (alloc_lut) {
        free(alloc_lut);
    }

    return HTP_STATUS_OK;
}

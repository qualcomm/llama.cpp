kernel void kernel_upscale(
    global const void * p_src0,
    ulong off_src0,
    global void * p_dst,
    ulong off_dst,
    ulong nb00,
    ulong nb01,
    ulong nb02,
    ulong nb03,
    int ne10,
    int ne11,
    int ne12,
    int ne13,
    float sf0,
    float sf1,
    float sf2,
    float sf3
) {
    global const char * src_base = (global const char *)p_src0 + off_src0;
    global float * dst_base = (global float *)((global char *)p_dst + off_dst);

    int index = get_global_id(0);
    int dst_total_elements = ne10 * ne11 * ne12 * ne13;

    if (index >= dst_total_elements) {
        return;
    }

    int i10 = index % ne10;
    int i11 = (index / ne10) % ne11;
    int i12 = (index / (ne10 * ne11)) % ne12;
    int i13 = index / (ne10 * ne11 * ne12);

    int i00 = (int)(i10 / sf0);
    int i01 = (int)(i11 / sf1);
    int i02 = (int)(i12 / sf2);
    int i03 = (int)(i13 / sf3);

    ulong offset_src_element = (ulong)i03 * nb03 + (ulong)i02 * nb02 + (ulong)i01 * nb01 + (ulong)i00 * nb00;
    global const float * src_element_ptr = (global const float *)(src_base + offset_src_element);

    dst_base[index] = *src_element_ptr;
}

// Bilinear with antialiasing (triangle filter), matching the CPU reference
// (PyTorch F.interpolate(..., antialias=True)): note the AA coordinate
// transform is (i + pixel_offset)/sf with NO trailing -pixel_offset, the
// window bounds truncate like the C++ float->int conversion, and the result
// normalizes by the accumulated weight.
kernel void kernel_upscale_bilinear_aa(
    global const void * p_src0,
    ulong off_src0,
    global void * p_dst,
    ulong off_dst,
    ulong nb00,
    ulong nb01,
    ulong nb02,
    ulong nb03,
    int ne00_src,
    int ne01_src,
    int ne10_dst,
    int ne11_dst,
    int ne12_dst,
    int ne13_dst,
    float sf0,
    float sf1,
    float sf2,
    float sf3,
    float pixel_offset
) {
    global const char * src_base = (global const char *)p_src0 + off_src0;
    global float * dst_base = (global float *)((global char *)p_dst + off_dst);

    int index = get_global_id(0);
    int dst_total_elements = ne10_dst * ne11_dst * ne12_dst * ne13_dst;

    if (index >= dst_total_elements) {
        return;
    }

    int i10_dst = index % ne10_dst;
    int i11_dst = (index / ne10_dst) % ne11_dst;
    int i12_dst = (index / (ne10_dst * ne11_dst)) % ne12_dst;
    int i13_dst = index / (ne10_dst * ne11_dst * ne12_dst);

    int i02_src = (int)(i12_dst / sf2);
    int i03_src = (int)(i13_dst / sf3);

    const float support0  = fmax(1.0f, 1.0f / sf0);
    const float invscale0 = 1.0f / support0;
    const float support1  = fmax(1.0f, 1.0f / sf1);
    const float invscale1 = 1.0f / support1;

    const float y = ((float)i11_dst + pixel_offset) / sf1;
    const float x = ((float)i10_dst + pixel_offset) / sf0;

    const int x_min = max((int)(x - support0 + pixel_offset), 0);
    const int x_max = min((int)(x + support0 + pixel_offset), ne00_src);
    const int y_min = max((int)(y - support1 + pixel_offset), 0);
    const int y_max = min((int)(y + support1 + pixel_offset), ne01_src);

    const ulong plane = (ulong)i02_src * nb02 + (ulong)i03_src * nb03;

    float val = 0.0f;
    float total_weight = 0.0f;

    for (int sy = y_min; sy < y_max; sy++) {
        const float weight_y = fmax(1.0f - fabs(((float)sy - y + pixel_offset) * invscale1), 0.0f);
        for (int sx = x_min; sx < x_max; sx++) {
            const float weight_x = fmax(1.0f - fabs(((float)sx - x + pixel_offset) * invscale0), 0.0f);
            const float weight = weight_x * weight_y;
            if (weight <= 0.0f) {
                continue;
            }
            const float pixel = *(global const float *)(src_base + (ulong)sx * nb00 + (ulong)sy * nb01 + plane);
            val += pixel * weight;
            total_weight += weight;
        }
    }

    if (total_weight > 0.0f) {
        val /= total_weight;
    }

    dst_base[index] = val;
}

// Bicubic convolution, alpha = -0.75, matching the CPU reference (PyTorch
// semantics) bit-for-bit in structure: unclamped dx/dy, per-tap edge clamp.
// NOT the cl_qcom_filter_bicubic HW sampler: that filter hardwires Catmull-Rom
// alpha = -0.5, which diverges from the reference the model was trained with.
inline float bicubic_w1(float x) {
    // ((a + 2)*x - (a + 3))*x*x + 1, a = -0.75
    return ((-0.75f + 2.0f) * x - (-0.75f + 3.0f)) * x * x + 1.0f;
}
inline float bicubic_w2(float x) {
    // ((a*x - 5a)*x + 8a)*x - 4a, a = -0.75
    return ((-0.75f * x - 5.0f * -0.75f) * x + 8.0f * -0.75f) * x - 4.0f * -0.75f;
}
inline float bicubic_row(float p0, float p1, float p2, float p3, float x) {
    return p0 * bicubic_w2(x + 1.0f)
         + p1 * bicubic_w1(x)
         + p2 * bicubic_w1(1.0f - x)
         + p3 * bicubic_w2(2.0f - x);
}

kernel void kernel_upscale_bicubic(
    global const void * p_src0,
    ulong off_src0,
    global void * p_dst,
    ulong off_dst,
    ulong nb00,
    ulong nb01,
    ulong nb02,
    ulong nb03,
    int ne00_src,
    int ne01_src,
    int ne10_dst,
    int ne11_dst,
    int ne12_dst,
    int ne13_dst,
    float sf0,
    float sf1,
    float sf2,
    float sf3,
    float pixel_offset
) {
    global const char * src_base = (global const char *)p_src0 + off_src0;
    global float * dst_base = (global float *)((global char *)p_dst + off_dst);

    int index = get_global_id(0);
    int dst_total_elements = ne10_dst * ne11_dst * ne12_dst * ne13_dst;

    if (index >= dst_total_elements) {
        return;
    }

    int i10_dst = index % ne10_dst;
    int i11_dst = (index / ne10_dst) % ne11_dst;
    int i12_dst = (index / (ne10_dst * ne11_dst)) % ne12_dst;
    int i13_dst = index / (ne10_dst * ne11_dst * ne12_dst);

    int i02_src = (int)(i12_dst / sf2);
    int i03_src = (int)(i13_dst / sf3);

    float y_src_f = ((float)i11_dst + pixel_offset) / sf1 - pixel_offset;
    int y0 = (int)floor(y_src_f);
    float dy = y_src_f - (float)y0;

    float x_src_f = ((float)i10_dst + pixel_offset) / sf0 - pixel_offset;
    int x0 = (int)floor(x_src_f);
    float dx = x_src_f - (float)x0;

    const ulong plane = (ulong)i02_src * nb02 + (ulong)i03_src * nb03;

    float rows[4];
    for (int r = 0; r < 4; ++r) {
        const int i01 = max(0, min(y0 + r - 1, ne01_src - 1));
        float taps[4];
        for (int c = 0; c < 4; ++c) {
            const int i00 = max(0, min(x0 + c - 1, ne00_src - 1));
            taps[c] = *(global const float *)(src_base + (ulong)i00 * nb00 + (ulong)i01 * nb01 + plane);
        }
        rows[r] = bicubic_row(taps[0], taps[1], taps[2], taps[3], dx);
    }

    dst_base[index] = bicubic_row(rows[0], rows[1], rows[2], rows[3], dy);
}

kernel void kernel_upscale_bilinear(
    global const void * p_src0,
    ulong off_src0,
    global void * p_dst,
    ulong off_dst,
    ulong nb00,
    ulong nb01,
    ulong nb02,
    ulong nb03,
    int ne00_src,
    int ne01_src,
    int ne10_dst,
    int ne11_dst,
    int ne12_dst,
    int ne13_dst,
    float sf0,
    float sf1,
    float sf2,
    float sf3,
    float pixel_offset
) {
    global const char * src_base = (global const char *)p_src0 + off_src0;
    global float * dst_base = (global float *)((global char *)p_dst + off_dst);

    int index = get_global_id(0);
    int dst_total_elements = ne10_dst * ne11_dst * ne12_dst * ne13_dst;

    if (index >= dst_total_elements) {
        return;
    }

    int i10_dst = index % ne10_dst;
    int i11_dst = (index / ne10_dst) % ne11_dst;
    int i12_dst = (index / (ne10_dst * ne11_dst)) % ne12_dst;
    int i13_dst = index / (ne10_dst * ne11_dst * ne12_dst);

    int i02_src = (int)(i12_dst / sf2);
    int i03_src = (int)(i13_dst / sf3);

    float y_src_f = ((float)i11_dst + pixel_offset) / sf1 - pixel_offset;
    long y0_src = (long)floor(y_src_f);
    long y1_src = y0_src + 1;

    y0_src = max(0L, min(y0_src, (long)ne01_src - 1));
    y1_src = max(0L, min(y1_src, (long)ne01_src - 1));

    float dy = y_src_f - (float)y0_src;
    dy = max(0.0f, min(dy, 1.0f));

    float x_src_f = ((float)i10_dst + pixel_offset) / sf0 - pixel_offset;
    long x0_src = (long)floor(x_src_f);
    long x1_src = x0_src + 1;

    x0_src = max(0L, min(x0_src, (long)ne00_src - 1));
    x1_src = max(0L, min(x1_src, (long)ne00_src - 1));

    float dx = x_src_f - (float)x0_src;
    dx = max(0.0f, min(dx, 1.0f));

    global const float * p_a = (global const float *)(src_base + (ulong)x0_src * nb00 + (ulong)y0_src * nb01 + (ulong)i02_src * nb02 + (ulong)i03_src * nb03);
    global const float * p_b = (global const float *)(src_base + (ulong)x1_src * nb00 + (ulong)y0_src * nb01 + (ulong)i02_src * nb02 + (ulong)i03_src * nb03);
    global const float * p_c = (global const float *)(src_base + (ulong)x0_src * nb00 + (ulong)y1_src * nb01 + (ulong)i02_src * nb02 + (ulong)i03_src * nb03);
    global const float * p_d = (global const float *)(src_base + (ulong)x1_src * nb00 + (ulong)y1_src * nb01 + (ulong)i02_src * nb02 + (ulong)i03_src * nb03);

    const float val_a = *p_a;
    const float val_b = *p_b;
    const float val_c = *p_c;
    const float val_d = *p_d;

    float result = val_a * (1.0f - dx) * (1.0f - dy) +
                   val_b * dx * (1.0f - dy) +
                   val_c * (1.0f - dx) * dy +
                   val_d * dx * dy;

    dst_base[index] = result;
}

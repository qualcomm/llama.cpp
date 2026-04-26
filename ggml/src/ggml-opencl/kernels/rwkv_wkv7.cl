#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// RWKV-7 wkv kernel. Mirrors Metal kernel_rwkv_wkv7_f32.
// Hardcoded head_size = 64.
// One work-group per (batch_id, head_id); local size = head_size; thread tid
// owns ROW `tid` of the head_size x head_size state matrix (different layout
// from WKV-6 — state[i] holds state[tid, i]).

#define HEAD_SIZE 64

kernel void kernel_rwkv_wkv7_f32(
    global const float * r,
    ulong                offset_r,
    global const float * w,
    ulong                offset_w,
    global const float * k,
    ulong                offset_k,
    global const float * v,
    ulong                offset_v,
    global const float * a,
    ulong                offset_a,
    global const float * b,
    ulong                offset_b,
    global const float * state_in,
    ulong                offset_si,
    global       float * dst,
    ulong                offset_d,
    uint                 B,
    uint                 T,
    uint                 C,
    uint                 H
) {
    r        = (global const float *)((global const char *) r        + offset_r);
    w        = (global const float *)((global const char *) w        + offset_w);
    k        = (global const float *)((global const char *) k        + offset_k);
    v        = (global const float *)((global const char *) v        + offset_v);
    a        = (global const float *)((global const char *) a        + offset_a);
    b        = (global const float *)((global const char *) b        + offset_b);
    state_in = (global const float *)((global const char *) state_in + offset_si);
    dst      = (global       float *)((global       char *) dst      + offset_d);

    const uint batch_id = get_group_id(0) / H;
    const uint head_id  = get_group_id(0) % H;
    const uint tid      = get_local_id(0);

    if (batch_id >= B || head_id >= H) return;

    const uint state_size   = C * HEAD_SIZE;
    const uint n_seq_tokens = T / B;

    local float _r[HEAD_SIZE];
    local float _w[HEAD_SIZE];
    local float _k[HEAD_SIZE];
    local float _a[HEAD_SIZE];
    local float _b[HEAD_SIZE];

    float state[HEAD_SIZE];

    for (uint i = 0; i < HEAD_SIZE; i++) {
        state[i] = state_in[batch_id*state_size + head_id*HEAD_SIZE*HEAD_SIZE + tid*HEAD_SIZE + i];
    }

    const uint start_t = batch_id*n_seq_tokens*C + head_id*HEAD_SIZE + tid;
    const uint end_t   = (batch_id + 1)*n_seq_tokens*C + head_id*HEAD_SIZE + tid;

    for (uint t = start_t; t < end_t; t += C) {
        barrier(CLK_LOCAL_MEM_FENCE);
        _r[tid] = r[t];
        _w[tid] = w[t];
        _k[tid] = k[t];
        _a[tid] = a[t];
        _b[tid] = b[t];
        barrier(CLK_LOCAL_MEM_FENCE);

        const float v_val = v[t];
        float sa = 0.0f;
        float4 sa_vec = (float4)(0.0f);

        for (uint j = 0; j < HEAD_SIZE; j += 4) {
            float4 a_vec = (float4)(_a[j], _a[j+1], _a[j+2], _a[j+3]);
            float4 s_vec = (float4)(state[j], state[j+1], state[j+2], state[j+3]);
            sa_vec += a_vec * s_vec;
        }
        sa = sa_vec.x + sa_vec.y + sa_vec.z + sa_vec.w;

        float y = 0.0f;
        for (uint j = 0; j < HEAD_SIZE; j += 4) {
            float4 r_vec = (float4)(_r[j], _r[j+1], _r[j+2], _r[j+3]);
            float4 w_vec = (float4)(_w[j], _w[j+1], _w[j+2], _w[j+3]);
            float4 k_vec = (float4)(_k[j], _k[j+1], _k[j+2], _k[j+3]);
            float4 b_vec = (float4)(_b[j], _b[j+1], _b[j+2], _b[j+3]);
            float4 s_vec = (float4)(state[j], state[j+1], state[j+2], state[j+3]);

            float4 kv = k_vec * v_val;

            s_vec = s_vec * w_vec + kv + sa * b_vec;
            y += dot(s_vec, r_vec);

            state[j]   = s_vec.x;
            state[j+1] = s_vec.y;
            state[j+2] = s_vec.z;
            state[j+3] = s_vec.w;
        }

        dst[t] = y;
    }

    for (uint i = 0; i < HEAD_SIZE; i++) {
        dst[T*C + batch_id*state_size + head_id*HEAD_SIZE*HEAD_SIZE + tid*HEAD_SIZE + i] = state[i];
    }
}

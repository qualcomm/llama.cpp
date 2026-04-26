#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// RWKV-6 wkv kernel. Mirrors Metal kernel_rwkv_wkv6_f32.
// Hardcoded head_size = 64 (matches all current model deployments and TBO tests).
// One work-group per (batch_id, head_id); local size = head_size; thread tid
// owns column `tid` of the head_size x head_size state matrix.

#define HEAD_SIZE 64

kernel void kernel_rwkv_wkv6_f32(
    global const float * k,
    ulong                offset_k,
    global const float * v,
    ulong                offset_v,
    global const float * r,
    ulong                offset_r,
    global const float * tf,
    ulong                offset_tf,
    global const float * td,
    ulong                offset_td,
    global const float * state_in,
    ulong                offset_si,
    global       float * dst,
    ulong                offset_d,
    uint                 B,
    uint                 T,
    uint                 C,
    uint                 H
) {
    k        = (global const float *)((global const char *) k        + offset_k);
    v        = (global const float *)((global const char *) v        + offset_v);
    r        = (global const float *)((global const char *) r        + offset_r);
    tf       = (global const float *)((global const char *) tf       + offset_tf);
    td       = (global const float *)((global const char *) td       + offset_td);
    state_in = (global const float *)((global const char *) state_in + offset_si);
    dst      = (global       float *)((global       char *) dst      + offset_d);

    const uint batch_id = get_group_id(0) / H;
    const uint head_id  = get_group_id(0) % H;
    const uint tid      = get_local_id(0);

    if (batch_id >= B || head_id >= H) return;

    const uint state_size   = C * HEAD_SIZE;
    const uint n_seq_tokens = T / B;

    local float _k[HEAD_SIZE];
    local float _r[HEAD_SIZE];
    local float _tf[HEAD_SIZE];
    local float _td[HEAD_SIZE];

    float state[HEAD_SIZE];

    for (uint i = 0; i < HEAD_SIZE; i++) {
        state[i] = state_in[batch_id*state_size + head_id*HEAD_SIZE*HEAD_SIZE + i*HEAD_SIZE + tid];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    _tf[tid] = tf[head_id*HEAD_SIZE + tid];
    barrier(CLK_LOCAL_MEM_FENCE);

    const uint start_t = batch_id*n_seq_tokens*C + head_id*HEAD_SIZE + tid;
    const uint end_t   = (batch_id + 1)*n_seq_tokens*C + head_id*HEAD_SIZE + tid;

    for (uint t = start_t; t < end_t; t += C) {
        barrier(CLK_LOCAL_MEM_FENCE);
        _k[tid]  = k[t];
        _r[tid]  = r[t];
        _td[tid] = td[t];
        barrier(CLK_LOCAL_MEM_FENCE);

        const float v_val = v[t];
        float y = 0.0f;

        for (uint j = 0; j < HEAD_SIZE; j += 4) {
            float4 k_vec  = (float4)(_k[j],  _k[j+1],  _k[j+2],  _k[j+3]);
            float4 r_vec  = (float4)(_r[j],  _r[j+1],  _r[j+2],  _r[j+3]);
            float4 tf_vec = (float4)(_tf[j], _tf[j+1], _tf[j+2], _tf[j+3]);
            float4 td_vec = (float4)(_td[j], _td[j+1], _td[j+2], _td[j+3]);
            float4 s_vec  = (float4)(state[j], state[j+1], state[j+2], state[j+3]);

            float4 kv   = k_vec * v_val;
            float4 temp = tf_vec * kv + s_vec;
            y += dot(r_vec, temp);

            s_vec = s_vec * td_vec + kv;
            state[j]   = s_vec.x;
            state[j+1] = s_vec.y;
            state[j+2] = s_vec.z;
            state[j+3] = s_vec.w;
        }

        dst[t] = y;
    }

    for (uint i = 0; i < HEAD_SIZE; i++) {
        dst[T*C + batch_id*state_size + head_id*HEAD_SIZE*HEAD_SIZE + i*HEAD_SIZE + tid] = state[i];
    }
}

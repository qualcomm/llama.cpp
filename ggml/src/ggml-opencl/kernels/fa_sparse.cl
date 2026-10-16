#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Sparse flash attention support: a FLASH_ATTN_EXT whose op carries n_kv_max (each mask row has at
// most n_kv_max finite entries, e.g. the qwen4exp QSA selection) attends over the union of the rows'
// visible cells only. These kernels compact that union and gather it into dense K/V/mask buffers,
// which the ordinary flash attention kernels then consume.

// a mask entry hides its cell when it is -inf
static inline bool fa_sparse_visible(half v) {
    return !(isinf(v) && v < 0.0h);
}

// one work item per KV cell: append the cell when any of the n_q rows sees it.
// count must be zero on entry; cells past cap are dropped (n_kv_max bounds the union by cap).
kernel void kernel_fa_sparse_compact(
        global const char * mask, ulong offm,
        global       int  * idx,
        global       int  * count,
        int   n_kv,
        int   n_q,
        ulong nbm1,
        int   cap
) {
    const int c = get_global_id(0);
    if (c >= n_kv) {
        return;
    }
    mask += offm;

    bool vis = false;
    for (int iq = 0; iq < n_q && !vis; ++iq) {
        vis = fa_sparse_visible(((global const half *)(mask + iq*nbm1))[c]);
    }
    if (vis) {
        const int i = atomic_inc(count);
        if (i < cap) {
            idx[i] = c;
        }
    }
}

// grid (row chunks of 8 halves, cap slots, n_head_kv): copy the selected cells' K and V rows into
// dense [d, cap, n_head_kv] buffers; padding slots get zero rows. Work items of chunk 0 and head 0
// also write the compact mask, -inf for the padding slots.
kernel void kernel_fa_sparse_gather(
        global const char * k,    ulong offk,
        global const char * v,    ulong offv,
        global const char * mask, ulong offm,
        global const int  * idx,
        global const int  * count,
        global       char * kc,
        global       char * vc,
        global       char * mc,
        int   dk8,
        int   dv8,
        int   cap,
        int   n_mrows,
        ulong nbk1, ulong nbk2,
        ulong nbv1, ulong nbv2,
        ulong nbm1
) {
    const int ch = get_global_id(0);
    const int i  = get_global_id(1);
    const int h  = get_global_id(2);

    const int  n     = min(count[0], cap);
    const bool valid = i < n;
    const int  c     = valid ? idx[i] : 0;

    if (ch < dk8) {
        ushort8 val = (ushort8)(0);
        if (valid) {
            val = vload8(ch, (global const ushort *)(k + offk + c*nbk1 + h*nbk2));
        }
        vstore8(val, ch, (global ushort *)(kc + ((size_t) h*cap + i)*dk8*16));
    }
    if (ch < dv8) {
        ushort8 val = (ushort8)(0);
        if (valid) {
            val = vload8(ch, (global const ushort *)(v + offv + c*nbv1 + h*nbv2));
        }
        vstore8(val, ch, (global ushort *)(vc + ((size_t) h*cap + i)*dv8*16));
    }
    if (ch == 0 && h == 0) {
        for (int iq = 0; iq < n_mrows; ++iq) {
            half mv = -INFINITY;
            if (valid) {
                mv = ((global const half *)(mask + offm + iq*nbm1))[c];
            }
            ((global half *)(mc + (size_t) iq*cap*sizeof(half)))[i] = mv;
        }
    }
}

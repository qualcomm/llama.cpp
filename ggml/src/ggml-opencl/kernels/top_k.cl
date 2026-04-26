#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define SWAP(x, y, T) { T tmp = (x); (x) = (y); (y) = tmp; }

kernel void kernel_top_k_f32_i32(
    global float * src0,
    ulong          offset0,
    global int   * dst,
    ulong          offsetd,
    const int      ne00,
    const int      ne00_pad,
    const int      k,
    local int    * dst_row
) {
    // bitonic sort indices by descending value, write first k indices to dst
    int col = get_local_id(0);
    int row = get_group_id(1);

    if (col >= ne00_pad) {
        return;
    }

    src0 = (global float *)((global char *)src0 + offset0);
    dst  = (global int   *)((global char *)dst  + offsetd);

    global float * x_row = src0 + row * ne00;

    dst_row[col] = col;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int kk = 2; kk <= ne00_pad; kk *= 2) {
        for (int j = kk / 2; j > 0; j /= 2) {
            int ixj = col ^ j;
            if (ixj > col) {
                if ((col & kk) == 0) {
                    if (dst_row[col] >= ne00 ||
                        (dst_row[ixj] < ne00 && x_row[dst_row[col]] < x_row[dst_row[ixj]])
                    ) {
                        SWAP(dst_row[col], dst_row[ixj], int);
                    }
                } else {
                    if (dst_row[ixj] >= ne00 ||
                        (dst_row[col] < ne00 && x_row[dst_row[col]] > x_row[dst_row[ixj]])
                    ) {
                        SWAP(dst_row[col], dst_row[ixj], int);
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    if (col < k) {
        dst[row * k + col] = dst_row[col];
    }
}

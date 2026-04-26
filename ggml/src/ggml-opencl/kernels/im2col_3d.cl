#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// im2col_3d (F32 input, F32 dst variant).
// src1 (image): [N*IC, ID, IH, IW]; ne13=N*IC, ne12=ID, ne11=IH, ne10=IW
// dst:          [N*OD, OH, OW, IC*KD*KH*KW] flattened
//
// One work-item per (in, iod, ioh, iow, iic) — but iic ∈ [0, IC), so flatten.

kernel void kernel_im2col_3d_f32(
    global const float * src1,
    ulong                offset1,
    global       float * dst,
    ulong                offsetd,
    int                  N,
    int                  IC,
    int                  ID, int IH, int IW,
    int                  KD, int KH, int KW,
    int                  OD, int OH, int OW,
    int                  s0, int s1, int s2,
    int                  p0, int p1, int p2,
    int                  d0, int d1, int d2,
    ulong                nb11, ulong nb12, ulong nb13
) {
    src1 = (global const float *)((global const char *) src1 + offset1);
    dst  = (global       float *)((global       char *) dst  + offsetd);

    const int OH_OW = OH * OW;
    const int KH_KW = KH * KW;
    const int KD_KH_KW = KD * KH_KW;
    const int IC_KD_KH_KW = IC * KD_KH_KW;

    // gid encodes (in, iod, ioh, iow, iic)
    const int gid = get_global_id(0);
    const int total = N * OD * OH * OW * IC;
    if (gid >= total) return;

    const int iic = gid % IC;
    const int iow = (gid / IC) % OW;
    const int ioh = (gid / (IC * OW)) % OH;
    const int iod = (gid / (IC * OW * OH)) % OD;
    const int in  = gid / (IC * OW * OH * OD);

    global float * dst_data = dst + (ulong)(in*OD*OH_OW + iod*OH_OW + ioh*OW + iow) * IC_KD_KH_KW;
    global const char * src_data_base = (global const char *) src1 + (ulong)(in*IC + iic) * nb13;

    for (int ikd = 0; ikd < KD; ikd++) {
        for (int ikh = 0; ikh < KH; ikh++) {
            for (int ikw = 0; ikw < KW; ikw++) {
                const int iiw = iow*s0 + ikw*d0 - p0;
                const int iih = ioh*s1 + ikh*d1 - p1;
                const int iid = iod*s2 + ikd*d2 - p2;

                const int dst_idx = iic*KD_KH_KW + ikd*KH_KW + ikh*KW + ikw;
                if (iid < 0 || iid >= ID || iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                    dst_data[dst_idx] = 0.0f;
                } else {
                    global const float * s = (global const float *)(src_data_base + (ulong)iid*nb12 + (ulong)iih*nb11 + (ulong)iiw*sizeof(float));
                    dst_data[dst_idx] = *s;
                }
            }
        }
    }
}

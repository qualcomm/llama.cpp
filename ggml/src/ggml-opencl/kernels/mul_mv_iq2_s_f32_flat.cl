#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// IQ2_S decode GEMV over the feature-major plane split. Same structure as
// mul_mv_iq3_s_f32_flat: K split across IQ2S_MV_NSG subgroups, and IQ2S_MV_R2
// gives a lane two adjacent rows so the uchar planes are read a ushort at a time.
// IQ2_S has no min term, so there is nothing row-independent to amortise and two
// rows is the right count here (see the Q2_K kernel for the case where it is not).

#define QK_K 256

#ifndef IQ2S_MV_NSG
#define IQ2S_MV_NSG 8
#endif

#ifndef IQ2S_MV_R2
#define IQ2S_MV_R2 1
#endif

// IQ2S_MV_LDSGRID=1: stage iq2s_grid into local memory once per workgroup and
// read it from there.
//
// iq2s_grid is 2048 uints = 8 KB, four times IQ3_S's table, and every lane
// indexes it divergently with a 10-bit index. Read from __constant the kernel
// showed the signature of a serialized resource rather than a latency-bound one:
// tg32 was FLAT across IQ2S_MV_NSG 2/4/8/16 (12.17-12.41) where every other split
// type moves several percent, and IQ2S_MV_R2=0 BEAT R2=1 (13.58 vs 12.28) because
// pairing rows doubles the distinct grid indices a lane holds live.
#ifndef IQ2S_MV_LDSGRID
#define IQ2S_MV_LDSGRID 1
#endif
constant uint iq2s_grid[2048] = {
    0x08080808, 0x08080808, 0x0808082b, 0x08080808, 0x08081919, 0x08080808, 0x08082b08, 0x08080808,
    0x08082b2b, 0x08080808, 0x08190819, 0x08080808, 0x08191908, 0x08080808, 0x0819192b, 0x08080808,
    0x08192b19, 0x08080808, 0x082b0808, 0x08080808, 0x082b082b, 0x08080808, 0x082b1919, 0x08080808,
    0x082b2b08, 0x08080808, 0x19080819, 0x08080808, 0x19081908, 0x08080808, 0x1908192b, 0x08080808,
    0x19082b19, 0x08080808, 0x19190808, 0x08080808, 0x1919082b, 0x08080808, 0x19191919, 0x08080808,
    0x19192b08, 0x08080808, 0x192b0819, 0x08080808, 0x192b1908, 0x08080808, 0x192b192b, 0x08080808,
    0x192b2b19, 0x08080808, 0x2b080808, 0x08080808, 0x2b08082b, 0x08080808, 0x2b081919, 0x08080808,
    0x2b082b08, 0x08080808, 0x2b190819, 0x08080808, 0x2b191908, 0x08080808, 0x2b2b0808, 0x08080808,
    0x2b2b1919, 0x08080808, 0x2b2b2b2b, 0x08080808, 0x08080819, 0x08080819, 0x08081908, 0x08080819,
    0x0808192b, 0x08080819, 0x08082b19, 0x08080819, 0x08190808, 0x08080819, 0x0819082b, 0x08080819,
    0x08191919, 0x08080819, 0x08192b08, 0x08080819, 0x082b0819, 0x08080819, 0x082b1908, 0x08080819,
    0x19080808, 0x08080819, 0x1908082b, 0x08080819, 0x19081919, 0x08080819, 0x19082b08, 0x08080819,
    0x19190819, 0x08080819, 0x19191908, 0x08080819, 0x1919192b, 0x08080819, 0x19192b19, 0x08080819,
    0x192b0808, 0x08080819, 0x192b1919, 0x08080819, 0x192b2b08, 0x08080819, 0x2b080819, 0x08080819,
    0x2b081908, 0x08080819, 0x2b190808, 0x08080819, 0x2b19082b, 0x08080819, 0x2b191919, 0x08080819,
    0x2b2b0819, 0x08080819, 0x2b2b1908, 0x08080819, 0x08080808, 0x0808082b, 0x0808082b, 0x0808082b,
    0x08081919, 0x0808082b, 0x08082b08, 0x0808082b, 0x08190819, 0x0808082b, 0x08191908, 0x0808082b,
    0x082b0808, 0x0808082b, 0x082b2b2b, 0x0808082b, 0x19080819, 0x0808082b, 0x19081908, 0x0808082b,
    0x1908192b, 0x0808082b, 0x19082b19, 0x0808082b, 0x19190808, 0x0808082b, 0x19191919, 0x0808082b,
    0x2b080808, 0x0808082b, 0x2b081919, 0x0808082b, 0x2b082b2b, 0x0808082b, 0x2b191908, 0x0808082b,
    0x2b2b082b, 0x0808082b, 0x08080819, 0x08081908, 0x08081908, 0x08081908, 0x0808192b, 0x08081908,
    0x08082b19, 0x08081908, 0x08190808, 0x08081908, 0x0819082b, 0x08081908, 0x08191919, 0x08081908,
    0x08192b08, 0x08081908, 0x082b0819, 0x08081908, 0x082b1908, 0x08081908, 0x082b192b, 0x08081908,
    0x082b2b19, 0x08081908, 0x19080808, 0x08081908, 0x1908082b, 0x08081908, 0x19081919, 0x08081908,
    0x19082b08, 0x08081908, 0x19082b2b, 0x08081908, 0x19190819, 0x08081908, 0x19191908, 0x08081908,
    0x1919192b, 0x08081908, 0x19192b19, 0x08081908, 0x192b0808, 0x08081908, 0x192b082b, 0x08081908,
    0x192b1919, 0x08081908, 0x2b080819, 0x08081908, 0x2b081908, 0x08081908, 0x2b08192b, 0x08081908,
    0x2b082b19, 0x08081908, 0x2b190808, 0x08081908, 0x2b191919, 0x08081908, 0x2b192b08, 0x08081908,
    0x2b2b0819, 0x08081908, 0x2b2b1908, 0x08081908, 0x08080808, 0x08081919, 0x0808082b, 0x08081919,
    0x08081919, 0x08081919, 0x08082b08, 0x08081919, 0x08082b2b, 0x08081919, 0x08190819, 0x08081919,
    0x08191908, 0x08081919, 0x0819192b, 0x08081919, 0x08192b19, 0x08081919, 0x082b0808, 0x08081919,
    0x082b1919, 0x08081919, 0x082b2b08, 0x08081919, 0x19080819, 0x08081919, 0x19081908, 0x08081919,
    0x1908192b, 0x08081919, 0x19082b19, 0x08081919, 0x19190808, 0x08081919, 0x1919082b, 0x08081919,
    0x19191919, 0x08081919, 0x19192b08, 0x08081919, 0x192b0819, 0x08081919, 0x192b1908, 0x08081919,
    0x2b080808, 0x08081919, 0x2b08082b, 0x08081919, 0x2b081919, 0x08081919, 0x2b082b08, 0x08081919,
    0x2b190819, 0x08081919, 0x2b191908, 0x08081919, 0x2b2b0808, 0x08081919, 0x08080819, 0x0808192b,
    0x08081908, 0x0808192b, 0x0808192b, 0x0808192b, 0x08082b19, 0x0808192b, 0x08190808, 0x0808192b,
    0x08191919, 0x0808192b, 0x19080808, 0x0808192b, 0x19081919, 0x0808192b, 0x19082b08, 0x0808192b,
    0x19190819, 0x0808192b, 0x19191908, 0x0808192b, 0x192b0808, 0x0808192b, 0x2b080819, 0x0808192b,
    0x2b081908, 0x0808192b, 0x2b190808, 0x0808192b, 0x08080808, 0x08082b08, 0x0808082b, 0x08082b08,
    0x08081919, 0x08082b08, 0x08082b08, 0x08082b08, 0x08190819, 0x08082b08, 0x08191908, 0x08082b08,
    0x0819192b, 0x08082b08, 0x08192b19, 0x08082b08, 0x082b0808, 0x08082b08, 0x082b1919, 0x08082b08,
    0x082b2b2b, 0x08082b08, 0x19080819, 0x08082b08, 0x19081908, 0x08082b08, 0x1908192b, 0x08082b08,
    0x19082b19, 0x08082b08, 0x19190808, 0x08082b08, 0x1919082b, 0x08082b08, 0x19191919, 0x08082b08,
    0x19192b08, 0x08082b08, 0x192b0819, 0x08082b08, 0x192b1908, 0x08082b08, 0x2b080808, 0x08082b08,
    0x2b081919, 0x08082b08, 0x2b191908, 0x08082b08, 0x2b2b2b2b, 0x08082b08, 0x08080819, 0x08082b19,
    0x08081908, 0x08082b19, 0x08190808, 0x08082b19, 0x0819082b, 0x08082b19, 0x08191919, 0x08082b19,
    0x08192b08, 0x08082b19, 0x082b0819, 0x08082b19, 0x19080808, 0x08082b19, 0x19081919, 0x08082b19,
    0x19082b08, 0x08082b19, 0x19190819, 0x08082b19, 0x19191908, 0x08082b19, 0x192b0808, 0x08082b19,
    0x2b080819, 0x08082b19, 0x2b190808, 0x08082b19, 0x08080808, 0x08082b2b, 0x08190819, 0x08082b2b,
    0x08191908, 0x08082b2b, 0x082b082b, 0x08082b2b, 0x082b2b08, 0x08082b2b, 0x082b2b2b, 0x08082b2b,
    0x19190808, 0x08082b2b, 0x2b192b19, 0x08082b2b, 0x08080819, 0x08190808, 0x08081908, 0x08190808,
    0x0808192b, 0x08190808, 0x08082b19, 0x08190808, 0x08190808, 0x08190808, 0x0819082b, 0x08190808,
    0x08191919, 0x08190808, 0x08192b08, 0x08190808, 0x082b0819, 0x08190808, 0x082b1908, 0x08190808,
    0x082b192b, 0x08190808, 0x19080808, 0x08190808, 0x1908082b, 0x08190808, 0x19081919, 0x08190808,
    0x19082b08, 0x08190808, 0x19190819, 0x08190808, 0x19191908, 0x08190808, 0x1919192b, 0x08190808,
    0x19192b19, 0x08190808, 0x192b0808, 0x08190808, 0x192b082b, 0x08190808, 0x192b1919, 0x08190808,
    0x192b2b08, 0x08190808, 0x2b080819, 0x08190808, 0x2b081908, 0x08190808, 0x2b08192b, 0x08190808,
    0x2b190808, 0x08190808, 0x2b191919, 0x08190808, 0x2b192b08, 0x08190808, 0x2b2b0819, 0x08190808,
    0x2b2b1908, 0x08190808, 0x08080808, 0x08190819, 0x0808082b, 0x08190819, 0x08081919, 0x08190819,
    0x08082b08, 0x08190819, 0x08082b2b, 0x08190819, 0x08190819, 0x08190819, 0x08191908, 0x08190819,
    0x0819192b, 0x08190819, 0x08192b19, 0x08190819, 0x082b0808, 0x08190819, 0x082b082b, 0x08190819,
    0x082b1919, 0x08190819, 0x082b2b08, 0x08190819, 0x19080819, 0x08190819, 0x19081908, 0x08190819,
    0x1908192b, 0x08190819, 0x19082b19, 0x08190819, 0x19190808, 0x08190819, 0x1919082b, 0x08190819,
    0x19191919, 0x08190819, 0x19192b08, 0x08190819, 0x192b0819, 0x08190819, 0x192b1908, 0x08190819,
    0x2b080808, 0x08190819, 0x2b08082b, 0x08190819, 0x2b081919, 0x08190819, 0x2b082b08, 0x08190819,
    0x2b190819, 0x08190819, 0x2b191908, 0x08190819, 0x08080819, 0x0819082b, 0x08081908, 0x0819082b,
    0x08082b19, 0x0819082b, 0x08190808, 0x0819082b, 0x08191919, 0x0819082b, 0x082b0819, 0x0819082b,
    0x082b1908, 0x0819082b, 0x19080808, 0x0819082b, 0x19081919, 0x0819082b, 0x19190819, 0x0819082b,
    0x19191908, 0x0819082b, 0x2b080819, 0x0819082b, 0x2b081908, 0x0819082b, 0x2b190808, 0x0819082b,
    0x08080808, 0x08191908, 0x0808082b, 0x08191908, 0x08081919, 0x08191908, 0x08082b08, 0x08191908,
    0x08190819, 0x08191908, 0x08191908, 0x08191908, 0x0819192b, 0x08191908, 0x08192b19, 0x08191908,
    0x082b0808, 0x08191908, 0x082b1919, 0x08191908, 0x082b2b08, 0x08191908, 0x19080819, 0x08191908,
    0x19081908, 0x08191908, 0x1908192b, 0x08191908, 0x19082b19, 0x08191908, 0x19190808, 0x08191908,
    0x1919082b, 0x08191908, 0x19191919, 0x08191908, 0x19192b08, 0x08191908, 0x192b0819, 0x08191908,
    0x192b1908, 0x08191908, 0x2b080808, 0x08191908, 0x2b08082b, 0x08191908, 0x2b081919, 0x08191908,
    0x2b082b08, 0x08191908, 0x2b190819, 0x08191908, 0x2b191908, 0x08191908, 0x2b2b0808, 0x08191908,
    0x08080819, 0x08191919, 0x08081908, 0x08191919, 0x0808192b, 0x08191919, 0x08082b19, 0x08191919,
    0x08190808, 0x08191919, 0x0819082b, 0x08191919, 0x08191919, 0x08191919, 0x08192b08, 0x08191919,
    0x082b0819, 0x08191919, 0x082b1908, 0x08191919, 0x19080808, 0x08191919, 0x1908082b, 0x08191919,
    0x19081919, 0x08191919, 0x19082b08, 0x08191919, 0x19190819, 0x08191919, 0x19191908, 0x08191919,
    0x192b0808, 0x08191919, 0x2b080819, 0x08191919, 0x2b081908, 0x08191919, 0x2b190808, 0x08191919,
    0x08080808, 0x0819192b, 0x08081919, 0x0819192b, 0x08082b08, 0x0819192b, 0x08190819, 0x0819192b,
    0x08191908, 0x0819192b, 0x082b0808, 0x0819192b, 0x19080819, 0x0819192b, 0x19081908, 0x0819192b,
    0x19190808, 0x0819192b, 0x2b080808, 0x0819192b, 0x2b2b2b2b, 0x0819192b, 0x08080819, 0x08192b08,
    0x08081908, 0x08192b08, 0x0808192b, 0x08192b08, 0x08082b19, 0x08192b08, 0x08190808, 0x08192b08,
    0x08191919, 0x08192b08, 0x08192b08, 0x08192b08, 0x082b0819, 0x08192b08, 0x19080808, 0x08192b08,
    0x1908082b, 0x08192b08, 0x19081919, 0x08192b08, 0x19082b08, 0x08192b08, 0x19190819, 0x08192b08,
    0x19191908, 0x08192b08, 0x192b0808, 0x08192b08, 0x2b080819, 0x08192b08, 0x2b081908, 0x08192b08,
    0x08080808, 0x08192b19, 0x0808082b, 0x08192b19, 0x08081919, 0x08192b19, 0x08082b08, 0x08192b19,
    0x08190819, 0x08192b19, 0x08191908, 0x08192b19, 0x082b0808, 0x08192b19, 0x19080819, 0x08192b19,
    0x19081908, 0x08192b19, 0x19190808, 0x08192b19, 0x192b2b19, 0x08192b19, 0x2b2b082b, 0x08192b19,
    0x08081908, 0x08192b2b, 0x08190808, 0x08192b2b, 0x19080808, 0x08192b2b, 0x1919192b, 0x08192b2b,
    0x08080808, 0x082b0808, 0x0808082b, 0x082b0808, 0x08081919, 0x082b0808, 0x08082b08, 0x082b0808,
    0x08190819, 0x082b0808, 0x08191908, 0x082b0808, 0x0819192b, 0x082b0808, 0x08192b19, 0x082b0808,
    0x082b0808, 0x082b0808, 0x082b1919, 0x082b0808, 0x082b2b2b, 0x082b0808, 0x19080819, 0x082b0808,
    0x19081908, 0x082b0808, 0x19190808, 0x082b0808, 0x1919082b, 0x082b0808, 0x19191919, 0x082b0808,
    0x192b1908, 0x082b0808, 0x2b080808, 0x082b0808, 0x2b082b2b, 0x082b0808, 0x2b191908, 0x082b0808,
    0x2b2b2b2b, 0x082b0808, 0x08080819, 0x082b0819, 0x08081908, 0x082b0819, 0x08190808, 0x082b0819,
    0x0819082b, 0x082b0819, 0x08191919, 0x082b0819, 0x082b0819, 0x082b0819, 0x19080808, 0x082b0819,
    0x1908082b, 0x082b0819, 0x19081919, 0x082b0819, 0x19190819, 0x082b0819, 0x19191908, 0x082b0819,
    0x192b0808, 0x082b0819, 0x2b080819, 0x082b0819, 0x2b081908, 0x082b0819, 0x2b190808, 0x082b0819,
    0x08080808, 0x082b082b, 0x08082b2b, 0x082b082b, 0x082b082b, 0x082b082b, 0x082b2b08, 0x082b082b,
    0x082b2b2b, 0x082b082b, 0x19081908, 0x082b082b, 0x19190808, 0x082b082b, 0x2b082b08, 0x082b082b,
    0x2b082b2b, 0x082b082b, 0x2b2b2b08, 0x082b082b, 0x08080819, 0x082b1908, 0x08081908, 0x082b1908,
    0x0808192b, 0x082b1908, 0x08082b19, 0x082b1908, 0x08190808, 0x082b1908, 0x08191919, 0x082b1908,
    0x08192b08, 0x082b1908, 0x082b0819, 0x082b1908, 0x082b1908, 0x082b1908, 0x19080808, 0x082b1908,
    0x1908082b, 0x082b1908, 0x19081919, 0x082b1908, 0x19082b08, 0x082b1908, 0x19190819, 0x082b1908,
    0x19191908, 0x082b1908, 0x192b0808, 0x082b1908, 0x2b080819, 0x082b1908, 0x2b081908, 0x082b1908,
    0x2b190808, 0x082b1908, 0x08080808, 0x082b1919, 0x08081919, 0x082b1919, 0x08082b08, 0x082b1919,
    0x08190819, 0x082b1919, 0x08191908, 0x082b1919, 0x082b0808, 0x082b1919, 0x19080819, 0x082b1919,
    0x19081908, 0x082b1919, 0x19190808, 0x082b1919, 0x192b192b, 0x082b1919, 0x2b080808, 0x082b1919,
    0x08080819, 0x082b192b, 0x08081908, 0x082b192b, 0x08190808, 0x082b192b, 0x19080808, 0x082b192b,
    0x19192b19, 0x082b192b, 0x08080808, 0x082b2b08, 0x08081919, 0x082b2b08, 0x08190819, 0x082b2b08,
    0x08191908, 0x082b2b08, 0x19080819, 0x082b2b08, 0x19081908, 0x082b2b08, 0x19190808, 0x082b2b08,
    0x2b082b2b, 0x082b2b08, 0x2b2b2b2b, 0x082b2b08, 0x08080819, 0x082b2b19, 0x08081908, 0x082b2b19,
    0x08190808, 0x082b2b19, 0x2b191919, 0x082b2b19, 0x08082b2b, 0x082b2b2b, 0x082b082b, 0x082b2b2b,
    0x192b1908, 0x082b2b2b, 0x2b082b08, 0x082b2b2b, 0x2b082b2b, 0x082b2b2b, 0x08080819, 0x19080808,
    0x08081908, 0x19080808, 0x0808192b, 0x19080808, 0x08082b19, 0x19080808, 0x08190808, 0x19080808,
    0x0819082b, 0x19080808, 0x08191919, 0x19080808, 0x08192b08, 0x19080808, 0x08192b2b, 0x19080808,
    0x082b0819, 0x19080808, 0x082b1908, 0x19080808, 0x082b192b, 0x19080808, 0x19080808, 0x19080808,
    0x1908082b, 0x19080808, 0x19081919, 0x19080808, 0x19082b08, 0x19080808, 0x19082b2b, 0x19080808,
    0x19190819, 0x19080808, 0x19191908, 0x19080808, 0x1919192b, 0x19080808, 0x19192b19, 0x19080808,
    0x192b0808, 0x19080808, 0x192b082b, 0x19080808, 0x192b1919, 0x19080808, 0x2b080819, 0x19080808,
    0x2b081908, 0x19080808, 0x2b190808, 0x19080808, 0x2b191919, 0x19080808, 0x2b192b08, 0x19080808,
    0x2b2b0819, 0x19080808, 0x2b2b1908, 0x19080808, 0x08080808, 0x19080819, 0x0808082b, 0x19080819,
    0x08081919, 0x19080819, 0x08082b08, 0x19080819, 0x08190819, 0x19080819, 0x08191908, 0x19080819,
    0x0819192b, 0x19080819, 0x08192b19, 0x19080819, 0x082b0808, 0x19080819, 0x082b082b, 0x19080819,
    0x082b1919, 0x19080819, 0x19080819, 0x19080819, 0x19081908, 0x19080819, 0x1908192b, 0x19080819,
    0x19082b19, 0x19080819, 0x19190808, 0x19080819, 0x1919082b, 0x19080819, 0x19191919, 0x19080819,
    0x19192b08, 0x19080819, 0x192b0819, 0x19080819, 0x192b1908, 0x19080819, 0x2b080808, 0x19080819,
    0x2b08082b, 0x19080819, 0x2b081919, 0x19080819, 0x2b082b08, 0x19080819, 0x2b190819, 0x19080819,
    0x2b191908, 0x19080819, 0x2b2b0808, 0x19080819, 0x08080819, 0x1908082b, 0x08081908, 0x1908082b,
    0x08190808, 0x1908082b, 0x0819082b, 0x1908082b, 0x08191919, 0x1908082b, 0x08192b08, 0x1908082b,
    0x082b1908, 0x1908082b, 0x19080808, 0x1908082b, 0x19081919, 0x1908082b, 0x19082b08, 0x1908082b,
    0x19190819, 0x1908082b, 0x19191908, 0x1908082b, 0x192b0808, 0x1908082b, 0x2b080819, 0x1908082b,
    0x2b081908, 0x1908082b, 0x08080808, 0x19081908, 0x0808082b, 0x19081908, 0x08081919, 0x19081908,
    0x08082b08, 0x19081908, 0x08082b2b, 0x19081908, 0x08190819, 0x19081908, 0x08191908, 0x19081908,
    0x0819192b, 0x19081908, 0x08192b19, 0x19081908, 0x082b0808, 0x19081908, 0x082b082b, 0x19081908,
    0x082b1919, 0x19081908, 0x082b2b08, 0x19081908, 0x19080819, 0x19081908, 0x19081908, 0x19081908,
    0x1908192b, 0x19081908, 0x19082b19, 0x19081908, 0x19190808, 0x19081908, 0x1919082b, 0x19081908,
    0x19191919, 0x19081908, 0x19192b08, 0x19081908, 0x192b0819, 0x19081908, 0x192b1908, 0x19081908,
    0x2b080808, 0x19081908, 0x2b08082b, 0x19081908, 0x2b081919, 0x19081908, 0x2b082b08, 0x19081908,
    0x2b190819, 0x19081908, 0x2b191908, 0x19081908, 0x2b2b0808, 0x19081908, 0x08080819, 0x19081919,
    0x08081908, 0x19081919, 0x0808192b, 0x19081919, 0x08082b19, 0x19081919, 0x08190808, 0x19081919,
    0x0819082b, 0x19081919, 0x08191919, 0x19081919, 0x08192b08, 0x19081919, 0x082b0819, 0x19081919,
    0x082b1908, 0x19081919, 0x19080808, 0x19081919, 0x1908082b, 0x19081919, 0x19081919, 0x19081919,
    0x19082b08, 0x19081919, 0x19190819, 0x19081919, 0x19191908, 0x19081919, 0x192b0808, 0x19081919,
    0x192b2b2b, 0x19081919, 0x2b080819, 0x19081919, 0x2b081908, 0x19081919, 0x2b190808, 0x19081919,
    0x08080808, 0x1908192b, 0x0808082b, 0x1908192b, 0x08081919, 0x1908192b, 0x08082b08, 0x1908192b,
    0x08190819, 0x1908192b, 0x08191908, 0x1908192b, 0x082b0808, 0x1908192b, 0x19080819, 0x1908192b,
    0x19081908, 0x1908192b, 0x19190808, 0x1908192b, 0x2b080808, 0x1908192b, 0x2b2b1919, 0x1908192b,
    0x08080819, 0x19082b08, 0x08081908, 0x19082b08, 0x08082b19, 0x19082b08, 0x08190808, 0x19082b08,
    0x0819082b, 0x19082b08, 0x08191919, 0x19082b08, 0x08192b08, 0x19082b08, 0x082b0819, 0x19082b08,
    0x082b1908, 0x19082b08, 0x19080808, 0x19082b08, 0x1908082b, 0x19082b08, 0x19081919, 0x19082b08,
    0x19082b08, 0x19082b08, 0x19190819, 0x19082b08, 0x19191908, 0x19082b08, 0x192b0808, 0x19082b08,
    0x2b081908, 0x19082b08, 0x2b190808, 0x19082b08, 0x08080808, 0x19082b19, 0x0808082b, 0x19082b19,
    0x08081919, 0x19082b19, 0x08082b08, 0x19082b19, 0x08190819, 0x19082b19, 0x08191908, 0x19082b19,
    0x082b0808, 0x19082b19, 0x19080819, 0x19082b19, 0x19081908, 0x19082b19, 0x19190808, 0x19082b19,
    0x2b080808, 0x19082b19, 0x2b19192b, 0x19082b19, 0x08080819, 0x19082b2b, 0x08081908, 0x19082b2b,
    0x08190808, 0x19082b2b, 0x19080808, 0x19082b2b, 0x08080808, 0x19190808, 0x0808082b, 0x19190808,
    0x08081919, 0x19190808, 0x08082b08, 0x19190808, 0x08190819, 0x19190808, 0x08191908, 0x19190808,
    0x0819192b, 0x19190808, 0x08192b19, 0x19190808, 0x082b0808, 0x19190808, 0x082b082b, 0x19190808,
    0x082b1919, 0x19190808, 0x082b2b08, 0x19190808, 0x19080819, 0x19190808, 0x19081908, 0x19190808,
    0x1908192b, 0x19190808, 0x19082b19, 0x19190808, 0x19190808, 0x19190808, 0x1919082b, 0x19190808,
    0x19191919, 0x19190808, 0x19192b08, 0x19190808, 0x192b0819, 0x19190808, 0x192b1908, 0x19190808,
    0x2b080808, 0x19190808, 0x2b08082b, 0x19190808, 0x2b081919, 0x19190808, 0x2b082b08, 0x19190808,
    0x2b190819, 0x19190808, 0x2b191908, 0x19190808, 0x08080819, 0x19190819, 0x08081908, 0x19190819,
    0x0808192b, 0x19190819, 0x08082b19, 0x19190819, 0x08190808, 0x19190819, 0x0819082b, 0x19190819,
    0x08191919, 0x19190819, 0x08192b08, 0x19190819, 0x082b0819, 0x19190819, 0x082b1908, 0x19190819,
    0x19080808, 0x19190819, 0x1908082b, 0x19190819, 0x19081919, 0x19190819, 0x19082b08, 0x19190819,
    0x19190819, 0x19190819, 0x19191908, 0x19190819, 0x192b0808, 0x19190819, 0x2b080819, 0x19190819,
    0x2b081908, 0x19190819, 0x2b190808, 0x19190819, 0x08080808, 0x1919082b, 0x08081919, 0x1919082b,
    0x08082b08, 0x1919082b, 0x08190819, 0x1919082b, 0x08191908, 0x1919082b, 0x082b0808, 0x1919082b,
    0x19080819, 0x1919082b, 0x19081908, 0x1919082b, 0x19190808, 0x1919082b, 0x192b2b19, 0x1919082b,
    0x2b080808, 0x1919082b, 0x08080819, 0x19191908, 0x08081908, 0x19191908, 0x0808192b, 0x19191908,
    0x08082b19, 0x19191908, 0x08190808, 0x19191908, 0x0819082b, 0x19191908, 0x08191919, 0x19191908,
    0x08192b08, 0x19191908, 0x082b0819, 0x19191908, 0x082b1908, 0x19191908, 0x19080808, 0x19191908,
    0x1908082b, 0x19191908, 0x19081919, 0x19191908, 0x19082b08, 0x19191908, 0x19190819, 0x19191908,
    0x19191908, 0x19191908, 0x192b0808, 0x19191908, 0x2b080819, 0x19191908, 0x2b081908, 0x19191908,
    0x2b190808, 0x19191908, 0x08080808, 0x19191919, 0x0808082b, 0x19191919, 0x08081919, 0x19191919,
    0x08082b08, 0x19191919, 0x08190819, 0x19191919, 0x08191908, 0x19191919, 0x082b0808, 0x19191919,
    0x19080819, 0x19191919, 0x19081908, 0x19191919, 0x19190808, 0x19191919, 0x2b080808, 0x19191919,
    0x08080819, 0x1919192b, 0x08081908, 0x1919192b, 0x08190808, 0x1919192b, 0x082b192b, 0x1919192b,
    0x19080808, 0x1919192b, 0x08080808, 0x19192b08, 0x0808082b, 0x19192b08, 0x08081919, 0x19192b08,
    0x08082b08, 0x19192b08, 0x08190819, 0x19192b08, 0x08191908, 0x19192b08, 0x082b0808, 0x19192b08,
    0x19080819, 0x19192b08, 0x19081908, 0x19192b08, 0x19190808, 0x19192b08, 0x19192b2b, 0x19192b08,
    0x2b080808, 0x19192b08, 0x08080819, 0x19192b19, 0x08081908, 0x19192b19, 0x08190808, 0x19192b19,
    0x19080808, 0x19192b19, 0x08080808, 0x19192b2b, 0x08192b19, 0x19192b2b, 0x2b081919, 0x19192b2b,
    0x2b2b2b08, 0x19192b2b, 0x08080819, 0x192b0808, 0x08081908, 0x192b0808, 0x0808192b, 0x192b0808,
    0x08190808, 0x192b0808, 0x0819082b, 0x192b0808, 0x08191919, 0x192b0808, 0x08192b08, 0x192b0808,
    0x082b0819, 0x192b0808, 0x082b1908, 0x192b0808, 0x19080808, 0x192b0808, 0x19081919, 0x192b0808,
    0x19082b08, 0x192b0808, 0x19190819, 0x192b0808, 0x19191908, 0x192b0808, 0x192b0808, 0x192b0808,
    0x2b081908, 0x192b0808, 0x2b190808, 0x192b0808, 0x08080808, 0x192b0819, 0x0808082b, 0x192b0819,
    0x08081919, 0x192b0819, 0x08082b08, 0x192b0819, 0x08190819, 0x192b0819, 0x08191908, 0x192b0819,
    0x082b0808, 0x192b0819, 0x19080819, 0x192b0819, 0x19081908, 0x192b0819, 0x19190808, 0x192b0819,
    0x2b080808, 0x192b0819, 0x2b192b19, 0x192b0819, 0x08081908, 0x192b082b, 0x08190808, 0x192b082b,
    0x19080808, 0x192b082b, 0x1919192b, 0x192b082b, 0x2b2b0819, 0x192b082b, 0x08080808, 0x192b1908,
    0x08081919, 0x192b1908, 0x08082b08, 0x192b1908, 0x08190819, 0x192b1908, 0x08191908, 0x192b1908,
    0x082b0808, 0x192b1908, 0x19080819, 0x192b1908, 0x19081908, 0x192b1908, 0x19190808, 0x192b1908,
    0x2b080808, 0x192b1908, 0x08080819, 0x192b1919, 0x08081908, 0x192b1919, 0x08190808, 0x192b1919,
    0x19080808, 0x192b1919, 0x19082b2b, 0x192b1919, 0x192b2b08, 0x192b1919, 0x2b19082b, 0x192b1919,
    0x08080808, 0x192b192b, 0x2b191908, 0x192b192b, 0x08080819, 0x192b2b08, 0x08081908, 0x192b2b08,
    0x08190808, 0x192b2b08, 0x192b1919, 0x192b2b08, 0x2b192b08, 0x192b2b08, 0x08080808, 0x192b2b19,
    0x082b2b2b, 0x192b2b19, 0x1908082b, 0x192b2b2b, 0x2b2b0819, 0x192b2b2b, 0x08080808, 0x2b080808,
    0x0808082b, 0x2b080808, 0x08081919, 0x2b080808, 0x08082b08, 0x2b080808, 0x08190819, 0x2b080808,
    0x08191908, 0x2b080808, 0x08192b19, 0x2b080808, 0x082b0808, 0x2b080808, 0x082b1919, 0x2b080808,
    0x19080819, 0x2b080808, 0x19081908, 0x2b080808, 0x19190808, 0x2b080808, 0x1919082b, 0x2b080808,
    0x19191919, 0x2b080808, 0x19192b08, 0x2b080808, 0x192b0819, 0x2b080808, 0x2b080808, 0x2b080808,
    0x2b081919, 0x2b080808, 0x2b190819, 0x2b080808, 0x2b191908, 0x2b080808, 0x08080819, 0x2b080819,
    0x08081908, 0x2b080819, 0x08082b19, 0x2b080819, 0x08190808, 0x2b080819, 0x0819082b, 0x2b080819,
    0x08191919, 0x2b080819, 0x08192b08, 0x2b080819, 0x082b0819, 0x2b080819, 0x082b1908, 0x2b080819,
    0x19080808, 0x2b080819, 0x1908082b, 0x2b080819, 0x19081919, 0x2b080819, 0x19082b08, 0x2b080819,
    0x19190819, 0x2b080819, 0x19191908, 0x2b080819, 0x2b080819, 0x2b080819, 0x2b081908, 0x2b080819,
    0x2b190808, 0x2b080819, 0x2b2b2b19, 0x2b080819, 0x08080808, 0x2b08082b, 0x08081919, 0x2b08082b,
    0x08082b2b, 0x2b08082b, 0x08190819, 0x2b08082b, 0x08191908, 0x2b08082b, 0x19080819, 0x2b08082b,
    0x19081908, 0x2b08082b, 0x19190808, 0x2b08082b, 0x08080819, 0x2b081908, 0x08081908, 0x2b081908,
    0x0808192b, 0x2b081908, 0x08082b19, 0x2b081908, 0x08190808, 0x2b081908, 0x0819082b, 0x2b081908,
    0x08191919, 0x2b081908, 0x08192b08, 0x2b081908, 0x082b0819, 0x2b081908, 0x19080808, 0x2b081908,
    0x1908082b, 0x2b081908, 0x19081919, 0x2b081908, 0x19082b08, 0x2b081908, 0x19190819, 0x2b081908,
    0x19191908, 0x2b081908, 0x192b0808, 0x2b081908, 0x2b080819, 0x2b081908, 0x2b081908, 0x2b081908,
    0x2b190808, 0x2b081908, 0x08080808, 0x2b081919, 0x0808082b, 0x2b081919, 0x08081919, 0x2b081919,
    0x08082b08, 0x2b081919, 0x08190819, 0x2b081919, 0x08191908, 0x2b081919, 0x082b0808, 0x2b081919,
    0x19080819, 0x2b081919, 0x19081908, 0x2b081919, 0x19190808, 0x2b081919, 0x2b080808, 0x2b081919,
    0x2b082b2b, 0x2b081919, 0x08080819, 0x2b08192b, 0x08081908, 0x2b08192b, 0x08190808, 0x2b08192b,
    0x082b2b19, 0x2b08192b, 0x19080808, 0x2b08192b, 0x08080808, 0x2b082b08, 0x08081919, 0x2b082b08,
    0x08190819, 0x2b082b08, 0x08191908, 0x2b082b08, 0x19080819, 0x2b082b08, 0x19081908, 0x2b082b08,
    0x19190808, 0x2b082b08, 0x2b2b082b, 0x2b082b08, 0x08080819, 0x2b082b19, 0x08081908, 0x2b082b19,
    0x19080808, 0x2b082b19, 0x192b1919, 0x2b082b19, 0x082b082b, 0x2b082b2b, 0x19192b08, 0x2b082b2b,
    0x19192b2b, 0x2b082b2b, 0x2b08082b, 0x2b082b2b, 0x2b2b082b, 0x2b082b2b, 0x08080819, 0x2b190808,
    0x08081908, 0x2b190808, 0x08082b19, 0x2b190808, 0x08190808, 0x2b190808, 0x0819082b, 0x2b190808,
    0x08191919, 0x2b190808, 0x08192b08, 0x2b190808, 0x082b1908, 0x2b190808, 0x19080808, 0x2b190808,
    0x1908082b, 0x2b190808, 0x19081919, 0x2b190808, 0x19082b08, 0x2b190808, 0x19190819, 0x2b190808,
    0x19191908, 0x2b190808, 0x192b0808, 0x2b190808, 0x2b080819, 0x2b190808, 0x2b081908, 0x2b190808,
    0x2b190808, 0x2b190808, 0x08080808, 0x2b190819, 0x08081919, 0x2b190819, 0x08190819, 0x2b190819,
    0x08191908, 0x2b190819, 0x19080819, 0x2b190819, 0x19081908, 0x2b190819, 0x19190808, 0x2b190819,
    0x19192b2b, 0x2b190819, 0x08080819, 0x2b19082b, 0x08081908, 0x2b19082b, 0x08190808, 0x2b19082b,
    0x19080808, 0x2b19082b, 0x2b2b192b, 0x2b19082b, 0x08080808, 0x2b191908, 0x0808082b, 0x2b191908,
    0x08081919, 0x2b191908, 0x08082b08, 0x2b191908, 0x08190819, 0x2b191908, 0x08191908, 0x2b191908,
    0x082b0808, 0x2b191908, 0x19080819, 0x2b191908, 0x19081908, 0x2b191908, 0x19190808, 0x2b191908,
    0x2b080808, 0x2b191908, 0x2b19192b, 0x2b191908, 0x08080819, 0x2b191919, 0x08081908, 0x2b191919,
    0x08190808, 0x2b191919, 0x19080808, 0x2b191919, 0x2b192b08, 0x2b191919, 0x2b2b0819, 0x2b191919,
    0x08080808, 0x2b19192b, 0x1908192b, 0x2b19192b, 0x192b1908, 0x2b19192b, 0x08080819, 0x2b192b08,
    0x08081908, 0x2b192b08, 0x08190808, 0x2b192b08, 0x082b192b, 0x2b192b08, 0x19080808, 0x2b192b08,
    0x2b2b2b19, 0x2b192b08, 0x08080808, 0x2b192b19, 0x19082b19, 0x2b192b19, 0x1919082b, 0x2b192b19,
    0x2b190808, 0x2b192b2b, 0x08080808, 0x2b2b0808, 0x08081919, 0x2b2b0808, 0x08082b2b, 0x2b2b0808,
    0x08191908, 0x2b2b0808, 0x082b082b, 0x2b2b0808, 0x082b2b2b, 0x2b2b0808, 0x19080819, 0x2b2b0808,
    0x19081908, 0x2b2b0808, 0x19190808, 0x2b2b0808, 0x2b2b082b, 0x2b2b0808, 0x2b2b2b2b, 0x2b2b0808,
    0x19080808, 0x2b2b0819, 0x192b1919, 0x2b2b0819, 0x0808082b, 0x2b2b082b, 0x08082b2b, 0x2b2b082b,
    0x082b082b, 0x2b2b082b, 0x082b2b08, 0x2b2b082b, 0x082b2b2b, 0x2b2b082b, 0x2b08082b, 0x2b2b082b,
    0x2b082b08, 0x2b2b082b, 0x2b082b2b, 0x2b2b082b, 0x2b2b2b08, 0x2b2b082b, 0x08080819, 0x2b2b1908,
    0x08081908, 0x2b2b1908, 0x08190808, 0x2b2b1908, 0x19080808, 0x2b2b1908, 0x2b082b19, 0x2b2b1908,
    0x2b2b1908, 0x2b2b1908, 0x08080808, 0x2b2b1919, 0x08192b19, 0x2b2b1919, 0x19190819, 0x2b2b192b,
    0x08082b2b, 0x2b2b2b08, 0x082b2b08, 0x2b2b2b08, 0x2b2b082b, 0x2b2b2b08, 0x19191908, 0x2b2b2b19,
    0x2b08192b, 0x2b2b2b19, 0x08082b08, 0x2b2b2b2b, 0x08082b2b, 0x2b2b2b2b, 0x082b0808, 0x2b2b2b2b,
    0x082b082b, 0x2b2b2b2b, 0x082b2b08, 0x2b2b2b2b, 0x2b082b08, 0x2b2b2b2b, 0x2b2b2b2b, 0x2b2b2b2b
};

// IQ_MV_SIGNXOR=1: apply the per-weight signs by XOR-ing the float sign bit
// rather than by four conditional negations.
//
// Why: the IQ3_S cost probe (GGML_OPENCL_IQ3S_MV_ABL=3) says dropping the sign
// application entirely is worth 13.6 percent -- the second-largest cost in these
// kernels after the grid lookup, and ahead of the activation load at 5.5. The
// three grid types share this helper verbatim and are together about 30 percent
// of Qwen3.8-27B decode.
//
// MEASURED: Qwen3.8-27B UD-IQ2_S tg32 2.389 -> 2.428, +1.6%. On.
// The identical helper is +4.6% on IQ3_XXS and -4.7% on IQ3_S, which is why each
// of the three carries its own knob. Do not re-merge them.
#ifndef IQ2S_MV_SIGNXOR
#define IQ2S_MV_SIGNXOR 1
#endif

// IQ2S_MV_ABL: COST PROBE, WRONG MATH. Prices the three per-operand costs of this
// kernel's inner loop so a lever is chosen from a measurement instead of a guess.
// The IQ3_S twin of this probe overturned that kernel's own header -- it assumed
// the codebook gather dominated, and the gather turned out to be 0.2% while the
// sign application was 14.2%. IQ2_S has never been priced.
//
//   1  drop the ACTIVATION load  (keeps the grid gather and the signs)
//   2  drop the GRID lookup      (keeps every weight/sign load and the index math)
//   3  drop the SIGN application (keeps the grid gather)
//   4  drop ALL THREE at once    (only the weight-plane loads and the loop remain)
//
// 🔑 4 IS THE ONE THAT MATTERS, and it is why the arm exists. Ablating one term at
// a time measures its MARGINAL cost, which understates a chain whose terms overlap:
// here the grid gather DEPENDS on the qs load and the dot DEPENDS on the gather, so
// removing any single link lets the others keep their latency hidden. 1+2+3 sum to
// ~17% while the kernel runs at 38% of roofline and a pure replay of this same plane
// addressing reaches 93%. If arm 4 is worth far more than the sum, the loop is
// LATENCY-bound on that dependent chain, not bound by any one term.
//
// Never enable in a real run.
#ifndef IQ2S_MV_ABL
#define IQ2S_MV_ABL 0
#endif

// IQ2S_MV_WORK=1: do the sign+dot TWICE on the ALREADY-FETCHED grid word, so the
// ARITHMETIC doubles while every load is held fixed. Flat means not compute-bound.
// IQ1_S and Q2_K measure 0.6-1.1% here; IQ3_S measures 33%.
// Never enable in a real run.
#ifndef IQ2S_MV_WORK
#define IQ2S_MV_WORK 0
#endif

// IQ2S_MV_WIMG=1: read the qs and signs PLANES through image1d_buffers instead of
// as global ushort pointers. Split-K kernel only, which carries most of this
// type's decode frame.
//
// Why the weight side and not another arithmetic lever: this kernel's own cost
// probe says the activation load is 3.7%, the grid lookup 6.7% and the sign
// application 6.4% -- about 17% in total -- while doubling the arithmetic costs
// only 2.6%. It is not compute-bound and no inner-loop lever can be worth more
// than that 17%, yet it runs at 38% of the part's bandwidth roofline. What is
// left is how the weights are fetched: the tuned q4_K GEMV reads its quant plane
// as whole texels through an image1d_buffer at 106-118 GB/s where the same bytes
// read as a buffer get 50.
//
// 🔑 THE PRECONDITION IS A WHOLE TEXEL PER LANE, and that is why the format is
// 16-bit. The cok weight texture was refuted at -39% because a lane took only
// part of a texel; here a lane owns a ROW PAIR and reads exactly one ushort, so
// the image is CL_R/CL_UNSIGNED_INT16 -- one texel IS the load. Probed on the
// X2-90: it is one of 87 supported image1d_buffer formats, and the 134.2 M pixel
// limit is far above any plane here. Consecutive lanes read consecutive ushorts,
// which is the coalesced pattern the q4_K path relies on.
#ifndef IQ2S_MV_WIMG
#define IQ2S_MV_WIMG 0
#endif

// Four grid values with their signs applied, as floats.
inline float4 iq2s_vals(uint gv, uint sg, uint base) {
#if IQ2S_MV_ABL == 3 || IQ2S_MV_ABL == 4
    return (float4)((float)((gv      ) & 0xFFu), (float)((gv >>  8) & 0xFFu),
                    (float)((gv >> 16) & 0xFFu), (float)((gv >> 24) & 0xFFu));
#elif IQ2S_MV_SIGNXOR
    // A sign flip is bit 31, so the four conditional negations collapse to one
    // XOR once the four sign bits are spread into place. Exact, not approximate.
    const uint  s   = sg >> base;
    const uint4 sgn = (uint4)(s << 31, s << 30, s << 29, s << 28) & 0x80000000u;
    return as_float4(as_uint4(convert_float4(as_uchar4(gv))) ^ sgn);
#else
    const uint s = sg >> base;
    float4 v;
    v.s0 = (float)((gv      ) & 0xFFu); if (s & 1u) { v.s0 = -v.s0; }
    v.s1 = (float)((gv >>  8) & 0xFFu); if (s & 2u) { v.s1 = -v.s1; }
    v.s2 = (float)((gv >> 16) & 0xFFu); if (s & 4u) { v.s2 = -v.s2; }
    v.s3 = (float)((gv >> 24) & 0xFFu); if (s & 8u) { v.s3 = -v.s3; }
    return v;
#endif
}

// IQ2S_MV_GRIDIMG=1: read the grid through an image1d_buffer instead of staging it
// in LOCAL memory.
//
// Why: the staged grid is worth +20-25% over __constant here, so the divergent grid
// read is this kernel's cost -- but LDS is then the tier serving it, and a divergent
// LDS read is bank-conflict serialised. The texture path has its own cache and is
// built for gather access, so it is the remaining tier to try. The Adreno guide's
// tier note rules __constant out for a data-dependent index and does not cover
// images, so this has to be measured rather than reasoned.
//
// The image is filled by kernel_iq2s_grid_export below, so the table never has to be
// duplicated on the host and cannot drift from the one this kernel compiles in.
#ifndef IQ2S_MV_GRIDIMG
#define IQ2S_MV_GRIDIMG 0
#endif

// Copies the compiled-in grid into a plain buffer once at init; the backend wraps
// that buffer in the image1d_buffer the GEMV reads.
// IQ2S_MV_AIMG=1: read the ACTIVATION through an image1d_buffer.
//
// The IQ3_S twin of this is +10.5% on a 3B and +2.8% on a 27B. The mechanism is
// structural and identical here: `grp` carries no row index, so every lane of a
// subgroup reads the SAME activation address -- a 64x redundant, wave-uniform
// load, and a wave-uniform image read is established as free on this part while
// LDS staging of the same redundancy measured -50%.
//
// One CL_RGBA/CL_FLOAT texel IS the float4 the scalar path loads, over src1's own
// buffer, so there is no copy and no pre-pass. y_tex is this token's row:
// (offset1/16) + col*ne10/4, and the host declines unless offset1 lands on a
// texel and ne10 is a whole number of them.
//
// Applied to ALL THREE kernels in this file. Unlike IQ3_S, whose fused GLU is
// default off, the IQ2_S fusion is default ON and serves ffn_gate+ffn_up -- so
// texturing only the plain GEMV would leave most of the frame untouched.
#ifndef IQ2S_MV_AIMG
#define IQ2S_MV_AIMG 0
#endif

#if IQ2S_MV_ABL == 1 || IQ2S_MV_ABL == 4
#define IQ2S_YV(g) ((float4)(1.0f))
#elif IQ2S_MV_AIMG
#define IQ2S_YV(g) read_imagef(y_img, (int)(y_tex + (g)))
#else
#define IQ2S_YV(g) vload4((g), y)
#endif

kernel void kernel_iq2s_grid_export(global uint * out) {
    const uint i = get_global_id(0);
    if (i < 2048u) {
        out[i] = iq2s_grid[i];
    }
}

kernel void kernel_mul_mv_iq2_s_f32_flat(
        __read_only image1d_buffer_t grid_img,
        __read_only image1d_buffer_t y_img,   // see IQ2S_MV_AIMG
        global const uchar * src0_qs,
        global const uchar * src0_sg,
        global const uchar * src0_qh,
        global const uchar * src0_sc,
        global const half  * src0_d,
        global const float * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne10,
        int ne0,
        uint y_off   // offset1/16, in float4 texels (IQ2S_MV_AIMG only)
) {
    src1 = (global const float *)((global const char *)src1 + offset1);
    dst  = (global float       *)((global char       *)dst  + offsetd);

    const uint m   = (uint)ne01;
    const uint K   = (uint)ne00;
    const uint nsb = K / QK_K;

    const uint lid = get_local_id(0);
    const uint sgi = get_local_id(1);
    const uint col = get_group_id(1);

    global const float * y = src1 + (ulong)col * (uint)ne10;
#if IQ2S_MV_AIMG
    const uint y_tex = y_off + col * ((uint)ne10 >> 2);
#endif

#if IQ2S_MV_ABL == 2 || IQ2S_MV_ABL == 4
// keeps the index math, drops the gather
#define IQ2S_GRID(i) (((i) * 0x01010101u) | 0x01010101u)
#elif IQ2S_MV_GRIDIMG
#define IQ2S_GRID(i) (read_imageui(grid_img, (int)(i)).x)
#elif IQ2S_MV_LDSGRID
    __local uint sh_grid[2048];
    {
        const uint tid  = sgi * 64u + lid;
        const uint nthr = (uint)(get_local_size(0) * get_local_size(1));
        for (uint i = tid; i < 2048u; i += nthr) {
            sh_grid[i] = iq2s_grid[i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#define IQ2S_GRID(i) sh_grid[(i)]
#else
#define IQ2S_GRID(i) iq2s_grid[(i)]
#endif

#if IQ2S_MV_R2
    const uint mh  = m >> 1;
    const uint j   = get_group_id(0) * 64u + lid;
    const uint row = j << 1;

    float sumf = 0.f, sumf1 = 0.f;

    if (j < mh) {
        global const ushort * qsu = (global const ushort *)src0_qs;
        global const ushort * sgu = (global const ushort *)src0_sg;
        global const ushort * qhu = (global const ushort *)src0_qh;
        global const ushort * scu = (global const ushort *)src0_sc;

        for (uint ib = sgi; ib < nsb; ib += IQ2S_MV_NSG) {
            const half2 dh = vload2(j + ib * mh, src0_d);

            float acc0 = 0.f, acc1 = 0.f;
            for (uint sb = 0; sb < 8u; ++sb) {
                const uint sub = ib * 8u + sb;
                const uint qhv = (uint)qhu[j + sub * mh];
                const uint scv = (uint)scu[j + sub * mh];
                const uint gb  = j + (sub * 4u) * mh;

                for (uint h = 0; h < 2u; ++h) {
                    const uint n0 = (h == 0u) ? ( scv        & 0xFu) : (( scv        >> 4) & 0xFu);
                    const uint n1 = (h == 0u) ? ((scv >> 8)  & 0xFu) : (((scv >> 8)  >> 4) & 0xFu);

                    float a0 = 0.f, a1 = 0.f;
                    for (uint t = 0; t < 2u; ++t) {
                        const uint l   = 2u*h + t;
                        const uint qsv = (uint)qsu[gb + l * mh];   // row pair, one load
                        const uint sgv = (uint)sgu[gb + l * mh];
                        const uint gi0 = ( qsv       & 0xFFu) | ((((qhv      ) >> (2u*l)) & 3u) << 8);
                        const uint gi1 = ((qsv >> 8) & 0xFFu) | ((((qhv >> 8) >> (2u*l)) & 3u) << 8);
                        const uint s0  =  sgv       & 0xFFu;
                        const uint s1  = (sgv >> 8) & 0xFFu;

                        const uint grp = (ib * 64u + sb * 8u) + l * 2u;   // K/4 group
                        const float4 y0 = IQ2S_YV(grp + 0u);
                        const float4 y1 = IQ2S_YV(grp + 1u);
                        a0 += dot(y0, iq2s_vals(IQ2S_GRID(2u*gi0 + 0u), s0, 0u));
                        a0 += dot(y1, iq2s_vals(IQ2S_GRID(2u*gi0 + 1u), s0, 4u));
                        a1 += dot(y0, iq2s_vals(IQ2S_GRID(2u*gi1 + 0u), s1, 0u));
                        a1 += dot(y1, iq2s_vals(IQ2S_GRID(2u*gi1 + 1u), s1, 4u));
                    }
                    acc0 += (0.5f + (float)n0) * a0;
                    acc1 += (0.5f + (float)n1) * a1;
                }
            }
            sumf  += (float)dh.s0 * 0.25f * acc0;
            sumf1 += (float)dh.s1 * 0.25f * acc1;
        }
    }
#else
    const uint row = get_group_id(0) * 64u + lid;

    float sumf = 0.f;

    if (row < m) {
        for (uint ib = sgi; ib < nsb; ib += IQ2S_MV_NSG) {
            const float d = (float)src0_d[row + ib * m];

            float acc = 0.f;
            for (uint sb = 0; sb < 8u; ++sb) {
                const uint sub = ib * 8u + sb;
                const uint qhv = (uint)src0_qh[row + sub * m];
                const uint scv = (uint)src0_sc[row + sub * m];
                const uint gb  = row + (sub * 4u) * m;

                for (uint h = 0; h < 2u; ++h) {
                    const uint nib = (h == 0u) ? (scv & 0xFu) : ((scv >> 4) & 0xFu);

                    float a = 0.f;
                    for (uint t = 0; t < 2u; ++t) {
                        const uint l   = 2u*h + t;
                        const uint gi  = (uint)src0_qs[gb + l * m] | (((qhv >> (2u*l)) & 3u) << 8);
                        const uint sgv = (uint)src0_sg[gb + l * m];

                        const uint grp = (ib * 64u + sb * 8u) + l * 2u;
                        a += dot(IQ2S_YV(grp + 0u), iq2s_vals(IQ2S_GRID(2u*gi + 0u), sgv, 0u));
                        a += dot(IQ2S_YV(grp + 1u), iq2s_vals(IQ2S_GRID(2u*gi + 1u), sgv, 4u));
                    }
                    acc += (0.5f + (float)nib) * a;
                }
            }
            sumf += d * 0.25f * acc;
        }
    }
#endif

#if IQ2S_MV_NSG > 1
#if IQ2S_MV_R2
    __local float2 part[IQ2S_MV_NSG][64];
    part[sgi][lid] = (float2)(sumf, sumf1);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sgi != 0) {
        return;
    }
    for (uint s = 1; s < IQ2S_MV_NSG; ++s) {
        const float2 p = part[s][lid];
        sumf  += p.s0;
        sumf1 += p.s1;
    }
#else
    __local float part[IQ2S_MV_NSG][64];
    part[sgi][lid] = sumf;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sgi != 0) {
        return;
    }
    for (uint s = 1; s < IQ2S_MV_NSG; ++s) {
        sumf += part[s][lid];
    }
#endif
#endif

#if IQ2S_MV_R2
    if (j < mh) {
        vstore2((float2)(sumf, sumf1), 0, dst + (ulong)col * (uint)ne0 + row);
    }
#else
    if (row < m) {
        dst[(ulong)col * (uint)ne0 + row] = sumf;
    }
#endif
#undef IQ2S_GRID
}


// ---------------------------------------------------------------------------
// Two COLUMNS per workgroup, for the ne11 = 2..31 band. The IQ2_S twin of
// kernel_mul_mv_iq1_s_f32_flat_mc; see that kernel for the full rationale.
//
// The plain GEMV takes its column from get_group_id(1), so at ne11 = N it reads
// the whole weight matrix N times and gathers the grid N times. Nothing else
// covers that band: split-K is gated ne11 == 1 and the prefill GEMM starts at
// ne11 >= 32.
//
// MEASURED on IQ1_S, which shares this shape: +8.5% at ne11 2 rising to +17.3%
// at ne11 8, and it gains MORE on a 27B than on a 3B rather than inverting.
// Unlike the row fold and the codebook prefetch, this removes redundant weight
// READS instead of spending registers, so a device that is already full has
// nothing to give back.
//
// The four grid entries of a step are gathered once and held; their sign-applied
// float4s are rebuilt per column rather than kept, because iq2s_vals is a shift
// and an XOR and these kernels are flat under doubled arithmetic, while holding
// eight float4s would spend exactly the register budget that is the constraint.
//
// An odd trailing column is computed against a duplicate of its partner and
// dropped at the store, so the inner loop has no divergent branch. That wastes
// one column of 2*ceil(n/2); the host declines ne11 == 3, the only width where
// that outweighs the saving.
// ---------------------------------------------------------------------------

// Columns per workgroup: 2 or 4. The file is compiled TWICE so both exist as
// separate kernels and the host dispatches whichever width divides ne11 -- see
// the IQ1_M twin, where guarding the trailing columns inside one kernel instead
// cost 7-9% at the widths that had no tail to guard.
#ifndef IQ2S_MV_NC
#define IQ2S_MV_NC 2
#endif

#if IQ2S_MV_AIMG
#define IQ2S_MCYA(g) read_imagef(y_img, (int)(y_tex_a + (g)))
#define IQ2S_MCYB(g) read_imagef(y_img, (int)(y_tex_b + (g)))
#define IQ2S_MCYC(g) read_imagef(y_img, (int)(y_tex_c + (g)))
#define IQ2S_MCYD(g) read_imagef(y_img, (int)(y_tex_d + (g)))
#else
#define IQ2S_MCYA(g) vload4((g), ya)
#define IQ2S_MCYB(g) vload4((g), yb)
#define IQ2S_MCYC(g) vload4((g), yc)
#define IQ2S_MCYD(g) vload4((g), yd)
#endif

kernel void kernel_mul_mv_iq2_s_f32_flat_mc(
        __read_only image1d_buffer_t grid_img,
        __read_only image1d_buffer_t y_img,   // see IQ2S_MV_AIMG
        global const uchar * src0_qs,
        global const uchar * src0_sg,
        global const uchar * src0_qh,
        global const uchar * src0_sc,
        global const half  * src0_d,
        global const float * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne10,
        int ne0,
        uint y_off,   // offset1/16, in float4 texels (IQ2S_MV_AIMG only)
        int ne11
) {
    src1 = (global const float *)((global const char *)src1 + offset1);
    dst  = (global float       *)((global char       *)dst  + offsetd);

    const uint m   = (uint)ne01;
    const uint K   = (uint)ne00;
    const uint nsb = K / QK_K;

    const uint lid = get_local_id(0);
    const uint sgi = get_local_id(1);

    const uint nc    = (uint)ne11;
    const uint ca    = get_group_id(1) * IQ2S_MV_NC;
    const uint has_b = (ca + 1u < nc) ? 1u : 0u;
    const uint cb    = ca + has_b;        // duplicate of ca when the pair is odd

    global const float * ya = src1 + (ulong)ca * (uint)ne10;
    global const float * yb = src1 + (ulong)cb * (uint)ne10;
#if IQ2S_MV_AIMG
    const uint y_tex_a = y_off + ca * ((uint)ne10 >> 2);
    const uint y_tex_b = y_off + cb * ((uint)ne10 >> 2);
#endif
#if IQ2S_MV_NC == 4
    const uint has_c = (ca + 2u < nc) ? 1u : 0u;
    const uint has_d = (ca + 3u < nc) ? 1u : 0u;
    const uint cc    = ca + (has_c ? 2u : 0u);
    const uint cd    = ca + (has_d ? 3u : 0u);
    global const float * yc = src1 + (ulong)cc * (uint)ne10;
    global const float * yd = src1 + (ulong)cd * (uint)ne10;
#if IQ2S_MV_AIMG
    const uint y_tex_c = y_off + cc * ((uint)ne10 >> 2);
    const uint y_tex_d = y_off + cd * ((uint)ne10 >> 2);
#endif
#endif

#if IQ2S_MV_ABL == 2 || IQ2S_MV_ABL == 4
// keeps the index math, drops the gather
#define IQ2S_MCGRID(i) (((i) * 0x01010101u) | 0x01010101u)
#elif IQ2S_MV_GRIDIMG
#define IQ2S_MCGRID(i) (read_imageui(grid_img, (int)(i)).x)
#else
#define IQ2S_MCGRID(i) iq2s_grid[(i)]
#endif

    const uint mh  = m >> 1;
    const uint j   = get_group_id(0) * 64u + lid;
    const uint row = j << 1;

    float sa0 = 0.f, sa1 = 0.f, sb0 = 0.f, sb1 = 0.f;   // [column][row]
#if IQ2S_MV_NC == 4
    float sc0 = 0.f, sc1 = 0.f, sd0 = 0.f, sd1 = 0.f;
#endif

    if (j < mh) {
        global const ushort * qsu = (global const ushort *)src0_qs;
        global const ushort * sgu = (global const ushort *)src0_sg;
        global const ushort * qhu = (global const ushort *)src0_qh;
        global const ushort * scu = (global const ushort *)src0_sc;

        for (uint ib = sgi; ib < nsb; ib += IQ2S_MV_NSG) {
            const half2 dh = vload2(j + ib * mh, src0_d);

            float aa0 = 0.f, aa1 = 0.f, ab0 = 0.f, ab1 = 0.f;
#if IQ2S_MV_NC == 4
            float ac0 = 0.f, ac1 = 0.f, ad0 = 0.f, ad1 = 0.f;
#endif
            for (uint sb = 0; sb < 8u; ++sb) {
                const uint sub = ib * 8u + sb;
                const uint qhv = (uint)qhu[j + sub * mh];
                const uint scv = (uint)scu[j + sub * mh];
                const uint gb  = j + (sub * 4u) * mh;

                for (uint h = 0; h < 2u; ++h) {
                    const uint n0 = (h == 0u) ? ( scv        & 0xFu) : (( scv        >> 4) & 0xFu);
                    const uint n1 = (h == 0u) ? ((scv >> 8)  & 0xFu) : (((scv >> 8)  >> 4) & 0xFu);

                    float pa0 = 0.f, pa1 = 0.f, pb0 = 0.f, pb1 = 0.f;
#if IQ2S_MV_NC == 4
                    float pc0 = 0.f, pc1 = 0.f, pd0 = 0.f, pd1 = 0.f;
#endif
                    for (uint t = 0; t < 2u; ++t) {
                        const uint l   = 2u*h + t;
                        const uint qsv = (uint)qsu[gb + l * mh];   // row pair, one load
                        const uint sgv = (uint)sgu[gb + l * mh];
                        const uint gi0 = ( qsv       & 0xFFu) | ((((qhv      ) >> (2u*l)) & 3u) << 8);
                        const uint gi1 = ((qsv >> 8) & 0xFFu) | ((((qhv >> 8) >> (2u*l)) & 3u) << 8);
                        const uint s0  =  sgv       & 0xFFu;
                        const uint s1  = (sgv >> 8) & 0xFFu;

                        // FOUR gathers, serving BOTH columns -- the whole point
                        const uint g00 = IQ2S_MCGRID(2u*gi0 + 0u);
                        const uint g01 = IQ2S_MCGRID(2u*gi0 + 1u);
                        const uint g10 = IQ2S_MCGRID(2u*gi1 + 0u);
                        const uint g11 = IQ2S_MCGRID(2u*gi1 + 1u);

                        const uint grp = (ib * 64u + sb * 8u) + l * 2u;   // K/4 group
                        const float4 a0v = IQ2S_MCYA(grp + 0u);
                        const float4 a1v = IQ2S_MCYA(grp + 1u);
                        pa0 += dot(a0v, iq2s_vals(g00, s0, 0u)) + dot(a1v, iq2s_vals(g01, s0, 4u));
                        pa1 += dot(a0v, iq2s_vals(g10, s1, 0u)) + dot(a1v, iq2s_vals(g11, s1, 4u));

                        const float4 b0v = IQ2S_MCYB(grp + 0u);
                        const float4 b1v = IQ2S_MCYB(grp + 1u);
                        pb0 += dot(b0v, iq2s_vals(g00, s0, 0u)) + dot(b1v, iq2s_vals(g01, s0, 4u));
                        pb1 += dot(b0v, iq2s_vals(g10, s1, 0u)) + dot(b1v, iq2s_vals(g11, s1, 4u));
#if IQ2S_MV_NC == 4
                        const float4 c0v = IQ2S_MCYC(grp + 0u);
                        const float4 c1v = IQ2S_MCYC(grp + 1u);
                        pc0 += dot(c0v, iq2s_vals(g00, s0, 0u)) + dot(c1v, iq2s_vals(g01, s0, 4u));
                        pc1 += dot(c0v, iq2s_vals(g10, s1, 0u)) + dot(c1v, iq2s_vals(g11, s1, 4u));

                        const float4 d0v = IQ2S_MCYD(grp + 0u);
                        const float4 d1v = IQ2S_MCYD(grp + 1u);
                        pd0 += dot(d0v, iq2s_vals(g00, s0, 0u)) + dot(d1v, iq2s_vals(g01, s0, 4u));
                        pd1 += dot(d0v, iq2s_vals(g10, s1, 0u)) + dot(d1v, iq2s_vals(g11, s1, 4u));
#endif
                    }
                    aa0 += (0.5f + (float)n0) * pa0;
                    aa1 += (0.5f + (float)n1) * pa1;
                    ab0 += (0.5f + (float)n0) * pb0;
                    ab1 += (0.5f + (float)n1) * pb1;
#if IQ2S_MV_NC == 4
                    ac0 += (0.5f + (float)n0) * pc0;
                    ac1 += (0.5f + (float)n1) * pc1;
                    ad0 += (0.5f + (float)n0) * pd0;
                    ad1 += (0.5f + (float)n1) * pd1;
#endif
                }
            }
            sa0 += (float)dh.s0 * 0.25f * aa0;
            sa1 += (float)dh.s1 * 0.25f * aa1;
            sb0 += (float)dh.s0 * 0.25f * ab0;
            sb1 += (float)dh.s1 * 0.25f * ab1;
#if IQ2S_MV_NC == 4
            sc0 += (float)dh.s0 * 0.25f * ac0;
            sc1 += (float)dh.s1 * 0.25f * ac1;
            sd0 += (float)dh.s0 * 0.25f * ad0;
            sd1 += (float)dh.s1 * 0.25f * ad1;
#endif
        }
    }

#if IQ2S_MV_NSG > 1
    __local float4 mcpart[IQ2S_MV_NSG][64];
#if IQ2S_MV_NC == 4
    __local float4 mcpart2[IQ2S_MV_NSG][64];
    mcpart2[sgi][lid] = (float4)(sc0, sc1, sd0, sd1);
#endif
    mcpart[sgi][lid] = (float4)(sa0, sa1, sb0, sb1);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sgi != 0) {
        return;
    }
    for (uint s = 1; s < IQ2S_MV_NSG; ++s) {
        const float4 p = mcpart[s][lid];
        sa0 += p.s0; sa1 += p.s1; sb0 += p.s2; sb1 += p.s3;
#if IQ2S_MV_NC == 4
        const float4 q = mcpart2[s][lid];
        sc0 += q.s0; sc1 += q.s1; sd0 += q.s2; sd1 += q.s3;
#endif
    }
#endif

    if (j < mh) {
        vstore2((float2)(sa0, sa1), 0, dst + (ulong)ca * (uint)ne0 + row);
        if (has_b) {
            vstore2((float2)(sb0, sb1), 0, dst + (ulong)cb * (uint)ne0 + row);
        }
#if IQ2S_MV_NC == 4
        if (has_c) {
            vstore2((float2)(sc0, sc1), 0, dst + (ulong)cc * (uint)ne0 + row);
        }
        if (has_d) {
            vstore2((float2)(sd0, sd1), 0, dst + (ulong)cd * (uint)ne0 + row);
        }
#endif
    }
#undef IQ2S_MCGRID
}


// ---------------------------------------------------------------------------
// Fused ffn_gate + ffn_up + GLU.
//
// Profiled on Llama-3.2-3B-UD-IQ2_M decode: ffn_gate and ffn_up are BOTH served
// by kernel_mul_mv_iq2_s_f32_flat and together cost 762 ms of a 2449 ms frame,
// 31.1%, across 2990 dispatches. q4_0 already does the same work as ONE fused
// ffn_swiglu; every IQ type still runs two separate GEMVs and a third elementwise
// pass. The q4_0 fusion is worth +10.2% of decode.
//
// Both GEMVs read the SAME activation, so the two weight streams are interleaved
// here rather than run one after the other: y0/y1 are loaded once per K group and
// feed gate and up in the same iteration. That is the point of the fusion. It also
// removes a dispatch per layer per token and the GLU's global round trip, which
// matters twice over because X2 decode is host and GPU co-bottlenecked.
//
// Cost: four accumulators instead of two under IQ2S_MV_R2 (gate/up x row pair).
// MEASURED on X2-90, fired-checked (391 fused dispatches in a 16-token profile,
// so the A/B is not vacuous), both arms repeated:
//     Llama-3.2-3B-UD-IQ2_M  tg64  25.773 -> 29.166  (+13.2%)
// wikitext PPL 14.8367 either way.
//
// On accuracy: the per-row K-sum order and the cross-subgroup reduce shape are
// unchanged from the base GEMV, and the GLU expression is the same one q40_glu_apply
// uses, so the only difference is that the activation is applied in registers
// instead of after a round trip. PPL agreeing to four decimals is consistent with
// bit-identical but does NOT prove it, and it was not checked bit-wise -- treat it
// as coherent, not as byte-identical.
//
// R2 ONLY. The host declines odd row counts before it gets here, exactly as the
// plane split itself does.
// ---------------------------------------------------------------------------

#define IQ2S_GLU_GEGLU_COEF_A   0.044715f
#define IQ2S_GLU_SQRT_2_OVER_PI 0.79788456080286535587989211986876f
#define IQ2S_GLU_SQRT_2_INV     0.70710678118654752440084436210484f
#define IQ2S_GLU_QUICK_COEF    -1.702f

// Same op numbering and the same expressions as q40_glu_apply, so the two fused
// paths cannot drift apart.
inline float iq2s_glu_apply(int glu_op, float g, float u) {
    float act;
    if (glu_op == 1) {        // GEGLU (tanh-approx gelu)
        act = 0.5f*g*(1.0f + tanh(IQ2S_GLU_SQRT_2_OVER_PI*g*(1.0f + IQ2S_GLU_GEGLU_COEF_A*g*g)));
    } else if (glu_op == 2) { // SWIGLU (silu)
        act = g / (1.0f + exp(-g));
    } else if (glu_op == 0) { // REGLU
        return g*u*(g > 0.0f);
    } else if (glu_op == 4) { // GEGLU_ERF
        act = 0.5f*g*(1.0f + erf(g*IQ2S_GLU_SQRT_2_INV));
    } else {                  // GEGLU_QUICK
        act = g*(1.0f/(1.0f + exp(IQ2S_GLU_QUICK_COEF*g)));
    }
    return act*u;
}

kernel void kernel_mul_mv_iq2_s_f32_flat_glu(
        __read_only image1d_buffer_t grid_img,
        __read_only image1d_buffer_t y_img,   // see IQ2S_MV_AIMG
        global const uchar * g_qs,
        global const uchar * g_sg,
        global const uchar * g_qh,
        global const uchar * g_sc,
        global const half  * g_d,
        global const uchar * u_qs,
        global const uchar * u_sg,
        global const uchar * u_qh,
        global const uchar * u_sc,
        global const half  * u_d,
        global const float * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne10,
        int ne0,
        int glu_op,
        uint y_off   // offset1/16, in float4 texels (IQ2S_MV_AIMG only)
) {
    src1 = (global const float *)((global const char *)src1 + offset1);
    dst  = (global float       *)((global char       *)dst  + offsetd);

    const uint m   = (uint)ne01;
    const uint K   = (uint)ne00;
    const uint nsb = K / QK_K;

    const uint lid = get_local_id(0);
    const uint sgi = get_local_id(1);
    const uint col = get_group_id(1);

    global const float * y = src1 + (ulong)col * (uint)ne10;
#if IQ2S_MV_AIMG
    const uint y_tex = y_off + col * ((uint)ne10 >> 2);
#endif

#if IQ2S_MV_ABL == 2 || IQ2S_MV_ABL == 4
// keeps the index math, drops the gather
#define IQ2S_GGRID(i) (((i) * 0x01010101u) | 0x01010101u)
#elif IQ2S_MV_GRIDIMG
#define IQ2S_GGRID(i) (read_imageui(grid_img, (int)(i)).x)
#else
#define IQ2S_GGRID(i) iq2s_grid[(i)]
#endif

    const uint mh  = m >> 1;
    const uint j   = get_group_id(0) * 64u + lid;
    const uint row = j << 1;

    float gs0 = 0.f, gs1 = 0.f, us0 = 0.f, us1 = 0.f;

    if (j < mh) {
        global const ushort * gqsu = (global const ushort *)g_qs;
        global const ushort * gsgu = (global const ushort *)g_sg;
        global const ushort * gqhu = (global const ushort *)g_qh;
        global const ushort * gscu = (global const ushort *)g_sc;
        global const ushort * uqsu = (global const ushort *)u_qs;
        global const ushort * usgu = (global const ushort *)u_sg;
        global const ushort * uqhu = (global const ushort *)u_qh;
        global const ushort * uscu = (global const ushort *)u_sc;

        for (uint ib = sgi; ib < nsb; ib += IQ2S_MV_NSG) {
            const half2 gdh = vload2(j + ib * mh, g_d);
            const half2 udh = vload2(j + ib * mh, u_d);

            float gacc0 = 0.f, gacc1 = 0.f, uacc0 = 0.f, uacc1 = 0.f;
            for (uint sb = 0; sb < 8u; ++sb) {
                const uint sub  = ib * 8u + sb;
                const uint gqhv = (uint)gqhu[j + sub * mh];
                const uint gscv = (uint)gscu[j + sub * mh];
                const uint uqhv = (uint)uqhu[j + sub * mh];
                const uint uscv = (uint)uscu[j + sub * mh];
                const uint gb   = j + (sub * 4u) * mh;

                for (uint h = 0; h < 2u; ++h) {
                    const uint gn0 = (h == 0u) ? ( gscv        & 0xFu) : (( gscv        >> 4) & 0xFu);
                    const uint gn1 = (h == 0u) ? ((gscv >> 8)  & 0xFu) : (((gscv >> 8)  >> 4) & 0xFu);
                    const uint un0 = (h == 0u) ? ( uscv        & 0xFu) : (( uscv        >> 4) & 0xFu);
                    const uint un1 = (h == 0u) ? ((uscv >> 8)  & 0xFu) : (((uscv >> 8)  >> 4) & 0xFu);

                    float ga0 = 0.f, ga1 = 0.f, ua0 = 0.f, ua1 = 0.f;
                    for (uint t = 0; t < 2u; ++t) {
                        const uint l = 2u*h + t;

                        // the activation is read ONCE and used by both streams
                        const uint grp = (ib * 64u + sb * 8u) + l * 2u;
                        const float4 y0 = IQ2S_YV(grp + 0u);
                        const float4 y1 = IQ2S_YV(grp + 1u);

                        const uint gqsv = (uint)gqsu[gb + l * mh];
                        const uint gsgv = (uint)gsgu[gb + l * mh];
                        const uint ggi0 = ( gqsv       & 0xFFu) | ((((gqhv      ) >> (2u*l)) & 3u) << 8);
                        const uint ggi1 = ((gqsv >> 8) & 0xFFu) | ((((gqhv >> 8) >> (2u*l)) & 3u) << 8);
                        const uint gsv0 =  gsgv       & 0xFFu;
                        const uint gsv1 = (gsgv >> 8) & 0xFFu;
                        ga0 += dot(y0, iq2s_vals(IQ2S_GGRID(2u*ggi0 + 0u), gsv0, 0u));
                        ga0 += dot(y1, iq2s_vals(IQ2S_GGRID(2u*ggi0 + 1u), gsv0, 4u));
                        ga1 += dot(y0, iq2s_vals(IQ2S_GGRID(2u*ggi1 + 0u), gsv1, 0u));
                        ga1 += dot(y1, iq2s_vals(IQ2S_GGRID(2u*ggi1 + 1u), gsv1, 4u));

                        const uint uqsv = (uint)uqsu[gb + l * mh];
                        const uint usgv = (uint)usgu[gb + l * mh];
                        const uint ugi0 = ( uqsv       & 0xFFu) | ((((uqhv      ) >> (2u*l)) & 3u) << 8);
                        const uint ugi1 = ((uqsv >> 8) & 0xFFu) | ((((uqhv >> 8) >> (2u*l)) & 3u) << 8);
                        const uint usv0 =  usgv       & 0xFFu;
                        const uint usv1 = (usgv >> 8) & 0xFFu;
                        ua0 += dot(y0, iq2s_vals(IQ2S_GGRID(2u*ugi0 + 0u), usv0, 0u));
                        ua0 += dot(y1, iq2s_vals(IQ2S_GGRID(2u*ugi0 + 1u), usv0, 4u));
                        ua1 += dot(y0, iq2s_vals(IQ2S_GGRID(2u*ugi1 + 0u), usv1, 0u));
                        ua1 += dot(y1, iq2s_vals(IQ2S_GGRID(2u*ugi1 + 1u), usv1, 4u));
                    }
                    gacc0 += (0.5f + (float)gn0) * ga0;
                    gacc1 += (0.5f + (float)gn1) * ga1;
                    uacc0 += (0.5f + (float)un0) * ua0;
                    uacc1 += (0.5f + (float)un1) * ua1;
                }
            }
            gs0 += (float)gdh.s0 * 0.25f * gacc0;
            gs1 += (float)gdh.s1 * 0.25f * gacc1;
            us0 += (float)udh.s0 * 0.25f * uacc0;
            us1 += (float)udh.s1 * 0.25f * uacc1;
        }
    }

#if IQ2S_MV_NSG > 1
    __local float4 gpart[IQ2S_MV_NSG][64];
    gpart[sgi][lid] = (float4)(gs0, gs1, us0, us1);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sgi != 0) {
        return;
    }
    for (uint s = 1; s < IQ2S_MV_NSG; ++s) {
        const float4 p = gpart[s][lid];
        gs0 += p.s0; gs1 += p.s1; us0 += p.s2; us1 += p.s3;
    }
#endif

    if (j < mh) {
        global float * o = dst + (ulong)col * (uint)ne0 + row;
        o[0] = iq2s_glu_apply(glu_op, gs0, us0);
        o[1] = iq2s_glu_apply(glu_op, gs1, us1);
    }
#undef IQ2S_GGRID
}


// ---------------------------------------------------------------------------
// Workgroup-level K split.
//
// This GEMV launches ceil(M/128) workgroups because its K split lives INSIDE a
// workgroup (IQ2S_MV_NSG subgroups). On a 16-CU X2-90 that leaves the low-M
// projections running a short tail on a half-idle device. Profiled on a
// Llama-3.2-3B-UD-IQ2_M decode frame, AFTER the gate/up fusion:
//
//     tensor        ms     % of frame   workgroups   occupancy
//     ffn_out     453.6      21.1%          24          75%
//     Qcur        179.8       8.4%          24          75%
//     Kcur         79.8       3.7%           8          50%
//
// Splitting K across workgroups multiplies the workgroup count by ksplit. Slice
// ks accumulates its own range of super-blocks and writes partial[ks*M + row];
// the existing kernel_gemv_splitk_reduce_f32 sums them.
//
// It is NOT free: the partial write plus the reduce pass cost ksplit*M, and they
// add a dispatch, which on this part costs host time as well as GPU time. A shape
// that already fills the device gets SLOWER when split, which is why the host
// picks ksplit from an occupancy heuristic and leaves well-filled shapes at 1.
//
// MEASURED on X2-90, fired-checked (1088 split dispatches and 1088 reduces in a
// 16-token profile), arms repeated, ON TOP of the gate/up fusion:
//     Llama-3.2-3B-UD-IQ2_M  tg64  29.147 -> 31.223  (+7.1%)
// wikitext PPL 14.8367 either way.
//
// The pinned sweep confirms the heuristic picks the optimum, and confirms the
// occupancy model rather than just agreeing with it. base_wg is 24 here:
//     auto  31.223      k=2  31.188   48 wg, 3 waves, 48/48 = 100%
//                       k=3  29.956   72 wg, 5 waves, 72/80 =  90%
//                       k=4  30.363   96 wg, 6 waves, 96/96 = 100%
// k=3 is WORSE than k=4 -- non-monotonic, and exactly where the model says the
// occupancy dips. k=4 matches k=2 on occupancy but pays more partial and reduce
// traffic, so it loses to it. Both facts fall out of "maximise wg/(waves*CU),
// prefer the smaller factor", which is why that rule is reused verbatim.
// ---------------------------------------------------------------------------

kernel void kernel_mul_mv_iq2_s_f32_flat_splitk(
        __read_only image1d_buffer_t qs_img,   // see IQ2S_MV_WIMG
        __read_only image1d_buffer_t sg_img,   // see IQ2S_MV_WIMG
        __read_only image1d_buffer_t grid_img,
        __read_only image1d_buffer_t y_img,   // see IQ2S_MV_AIMG
        global const uchar * src0_qs,
        global const uchar * src0_sg,
        global const uchar * src0_qh,
        global const uchar * src0_sc,
        global const half  * src0_d,
        global const float * src1,
        ulong offset1,
        global float * partial,
        int ne00,
        int ne01,
        int ne10,
        uint y_off   // offset1/16, in float4 texels (IQ2S_MV_AIMG only)
) {
    src1 = (global const float *)((global const char *)src1 + offset1);

    const uint m   = (uint)ne01;
    const uint K   = (uint)ne00;
    const uint nsb = K / QK_K;

    const uint lid = get_local_id(0);
    const uint sgi = get_local_id(1);
    const uint col = get_group_id(1);
    const uint ks  = get_group_id(2);
    const uint nks = get_num_groups(2);

    // this slice's super-block range; the subgroups stride within it
    const uint ib0 = (nsb * ks)        / nks;
    const uint ib1 = (nsb * (ks + 1u)) / nks;

    global const float * y = src1 + (ulong)col * (uint)ne10;
#if IQ2S_MV_AIMG
    const uint y_tex = y_off + col * ((uint)ne10 >> 2);
#endif

#if IQ2S_MV_ABL == 2 || IQ2S_MV_ABL == 4
// keeps the index math, drops the gather
#define IQ2S_SKGRID(i) (((i) * 0x01010101u) | 0x01010101u)
#elif IQ2S_MV_GRIDIMG
#define IQ2S_SKGRID(i) (read_imageui(grid_img, (int)(i)).x)
#else
#define IQ2S_SKGRID(i) iq2s_grid[(i)]
#endif

    const uint mh  = m >> 1;
    const uint j   = get_group_id(0) * 64u + lid;
    const uint row = j << 1;

    float sumf = 0.f, sumf1 = 0.f;

    if (j < mh) {
        global const ushort * qsu = (global const ushort *)src0_qs;
        global const ushort * sgu = (global const ushort *)src0_sg;
        global const ushort * qhu = (global const ushort *)src0_qh;
        global const ushort * scu = (global const ushort *)src0_sc;

        for (uint ib = ib0 + sgi; ib < ib1; ib += IQ2S_MV_NSG) {
            const half2 dh = vload2(j + ib * mh, src0_d);

            float acc0 = 0.f, acc1 = 0.f;
            for (uint sb = 0; sb < 8u; ++sb) {
                const uint sub = ib * 8u + sb;
                const uint qhv = (uint)qhu[j + sub * mh];
                const uint scv = (uint)scu[j + sub * mh];
                const uint gb  = j + (sub * 4u) * mh;

                for (uint h = 0; h < 2u; ++h) {
                    const uint n0 = (h == 0u) ? ( scv        & 0xFu) : (( scv        >> 4) & 0xFu);
                    const uint n1 = (h == 0u) ? ((scv >> 8)  & 0xFu) : (((scv >> 8)  >> 4) & 0xFu);

                    float a0 = 0.f, a1 = 0.f;
                    for (uint t = 0; t < 2u; ++t) {
                        const uint l   = 2u*h + t;
#if IQ2S_MV_WIMG
                        // one CL_R/UNSIGNED_INT16 texel IS the ushort a lane owns
                        const uint qsv = read_imageui(qs_img, (int)(gb + l * mh)).x;
                        const uint sgv = read_imageui(sg_img, (int)(gb + l * mh)).x;
#else
                        const uint qsv = (uint)qsu[gb + l * mh];
                        const uint sgv = (uint)sgu[gb + l * mh];
#endif
                        const uint gi0 = ( qsv       & 0xFFu) | ((((qhv      ) >> (2u*l)) & 3u) << 8);
                        const uint gi1 = ((qsv >> 8) & 0xFFu) | ((((qhv >> 8) >> (2u*l)) & 3u) << 8);
                        const uint s0  =  sgv       & 0xFFu;
                        const uint s1  = (sgv >> 8) & 0xFFu;

                        const uint grp = (ib * 64u + sb * 8u) + l * 2u;
                        const float4 y0 = IQ2S_YV(grp + 0u);
                        const float4 y1 = IQ2S_YV(grp + 1u);
#if IQ2S_MV_WORK
                        // same four grid words, sign+dot done twice: ARITHMETIC
                        // doubles, every load is held fixed
                        const uint gw00 = IQ2S_SKGRID(2u*gi0 + 0u);
                        const uint gw01 = IQ2S_SKGRID(2u*gi0 + 1u);
                        const uint gw10 = IQ2S_SKGRID(2u*gi1 + 0u);
                        const uint gw11 = IQ2S_SKGRID(2u*gi1 + 1u);
                        a0 += dot(y0, iq2s_vals(gw00, s0, 0u));
                        a0 += dot(y1, iq2s_vals(gw01, s0, 4u));
                        a1 += dot(y0, iq2s_vals(gw10, s1, 0u));
                        a1 += dot(y1, iq2s_vals(gw11, s1, 4u));
                        a0 += dot(y0, iq2s_vals(gw00, s0 + 1u, 0u));
                        a0 += dot(y1, iq2s_vals(gw01, s0 + 1u, 4u));
                        a1 += dot(y0, iq2s_vals(gw10, s1 + 1u, 0u));
                        a1 += dot(y1, iq2s_vals(gw11, s1 + 1u, 4u));
#else
                        a0 += dot(y0, iq2s_vals(IQ2S_SKGRID(2u*gi0 + 0u), s0, 0u));
                        a0 += dot(y1, iq2s_vals(IQ2S_SKGRID(2u*gi0 + 1u), s0, 4u));
                        a1 += dot(y0, iq2s_vals(IQ2S_SKGRID(2u*gi1 + 0u), s1, 0u));
                        a1 += dot(y1, iq2s_vals(IQ2S_SKGRID(2u*gi1 + 1u), s1, 4u));
#endif
                    }
                    acc0 += (0.5f + (float)n0) * a0;
                    acc1 += (0.5f + (float)n1) * a1;
                }
            }
            sumf  += (float)dh.s0 * 0.25f * acc0;
            sumf1 += (float)dh.s1 * 0.25f * acc1;
        }
    }

#if IQ2S_MV_NSG > 1
    __local float2 skpart[IQ2S_MV_NSG][64];
    skpart[sgi][lid] = (float2)(sumf, sumf1);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sgi != 0) {
        return;
    }
    for (uint s = 1; s < IQ2S_MV_NSG; ++s) {
        const float2 p = skpart[s][lid];
        sumf  += p.s0;
        sumf1 += p.s1;
    }
#endif

    if (j < mh) {
        // [ksplit][M], the layout kernel_gemv_splitk_reduce_f32 expects
        global float * o = partial + (ulong)ks * m + row;
        o[0] = sumf;
        o[1] = sumf1;
    }
#undef IQ2S_SKGRID
}

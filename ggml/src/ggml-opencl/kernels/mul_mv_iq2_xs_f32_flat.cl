#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// IQ2_XS decode GEMV over the feature-major plane split.
//
//   src0_qs [row + (k/8)*m]    ushort  9-bit grid index, 7-bit sign code above it
//   src0_sc [row + (k/32)*m]   uchar   two 4-bit sub-scales
//   src0_d  [row + (k/256)*m]  half    super-block scale
//
// Size preserving: 2*32 + 8 + 2 == 74 == sizeof(block_iq2_xs), so the three
// planes are subbuffers of the tensor's own allocation.
//
// Same structure as mul_mv_iq2_xxs_f32_flat, and it differs from that kernel in
// exactly two places: the grid index and its sign travel together in one ushort
// instead of being split across a byte plane and a packed word, and the scale is
// a nibble pair per 32 weights rather than the top four bits of the sign word.
// The grid is twice as long (512 entries) and is indexed the same way, as uint
// pairs, because one entry is eight bytes.

// MEASURED on X2-90: Llama-3.2-3B-UD-IQ2_M pp512 679.5 -> 685.1 (+0.8%),
// tg64 25.55 -> 25.69 (+0.6%). Small because that file holds only five IQ2_XS
// tensors -- the rest is IQ2_S, which was already split. No model on hand is
// IQ2_XS-dominant, so the large-gain case for this kernel is still unmeasured;
// its twin one step down, IQ2_XXS, is 1.92x prefill and 1.71x decode on a model
// made of it.
//
// Correctness: the plane round trip is byte exact on all five tensors, and
// wikitext PPL agrees within its error bar (17.078 -> 17.099 +/- 0.88).

#define QK_K 256

#ifndef IQ2XS_MV_NSG
#define IQ2XS_MV_NSG 8
#endif

// IQ2XS_MV_R2=1: a lane takes two adjacent rows so the ushort quant plane is
// read a uint at a time and the uchar scale plane a ushort at a time.
#ifndef IQ2XS_MV_R2
#define IQ2XS_MV_R2 1
#endif

// IQ2XS_MV_GRIDIMG=1: read the grid through an image1d_buffer rather than from
// __constant. Default on for the same reason as every other split grid type; see
// the IQ2_XXS GEMV header for the five measurements behind it. This grid is the
// largest of the set at 4 KB, which is the one size where local memory has ever
// won on this part, so LDS is worth asking about here even though it lost on the
// smaller tables -- it is not offered yet.
#ifndef IQ2XS_MV_GRIDIMG
#define IQ2XS_MV_GRIDIMG 1
#endif

constant uint iq2xs_grid[1024] = {
    0x08080808, 0x08080808, 0x0808082b, 0x08080808, 0x08081919, 0x08080808, 0x08082b08, 0x08080808,
    0x08082b2b, 0x08080808, 0x08190819, 0x08080808, 0x08191908, 0x08080808, 0x0819192b, 0x08080808,
    0x08192b19, 0x08080808, 0x082b0808, 0x08080808, 0x082b082b, 0x08080808, 0x082b1919, 0x08080808,
    0x082b2b08, 0x08080808, 0x19080819, 0x08080808, 0x19081908, 0x08080808, 0x1908192b, 0x08080808,
    0x19082b19, 0x08080808, 0x19190808, 0x08080808, 0x1919082b, 0x08080808, 0x19191919, 0x08080808,
    0x19192b08, 0x08080808, 0x192b0819, 0x08080808, 0x192b1908, 0x08080808, 0x2b080808, 0x08080808,
    0x2b08082b, 0x08080808, 0x2b081919, 0x08080808, 0x2b082b08, 0x08080808, 0x2b190819, 0x08080808,
    0x2b191908, 0x08080808, 0x2b192b19, 0x08080808, 0x2b2b0808, 0x08080808, 0x08080819, 0x08080819,
    0x08081908, 0x08080819, 0x0808192b, 0x08080819, 0x08082b19, 0x08080819, 0x08190808, 0x08080819,
    0x0819082b, 0x08080819, 0x08191919, 0x08080819, 0x08192b08, 0x08080819, 0x08192b2b, 0x08080819,
    0x082b0819, 0x08080819, 0x082b1908, 0x08080819, 0x19080808, 0x08080819, 0x1908082b, 0x08080819,
    0x19081919, 0x08080819, 0x19082b08, 0x08080819, 0x19190819, 0x08080819, 0x19191908, 0x08080819,
    0x192b0808, 0x08080819, 0x192b2b08, 0x08080819, 0x2b080819, 0x08080819, 0x2b081908, 0x08080819,
    0x2b190808, 0x08080819, 0x08080808, 0x0808082b, 0x0808082b, 0x0808082b, 0x08081919, 0x0808082b,
    0x08082b08, 0x0808082b, 0x08190819, 0x0808082b, 0x08191908, 0x0808082b, 0x082b0808, 0x0808082b,
    0x19080819, 0x0808082b, 0x19081908, 0x0808082b, 0x19190808, 0x0808082b, 0x19191919, 0x0808082b,
    0x2b080808, 0x0808082b, 0x2b082b2b, 0x0808082b, 0x08080819, 0x08081908, 0x08081908, 0x08081908,
    0x0808192b, 0x08081908, 0x08082b19, 0x08081908, 0x08190808, 0x08081908, 0x0819082b, 0x08081908,
    0x08191919, 0x08081908, 0x08192b08, 0x08081908, 0x082b0819, 0x08081908, 0x082b1908, 0x08081908,
    0x19080808, 0x08081908, 0x1908082b, 0x08081908, 0x19081919, 0x08081908, 0x19082b08, 0x08081908,
    0x19190819, 0x08081908, 0x19191908, 0x08081908, 0x1919192b, 0x08081908, 0x192b0808, 0x08081908,
    0x2b080819, 0x08081908, 0x2b081908, 0x08081908, 0x2b190808, 0x08081908, 0x08080808, 0x08081919,
    0x0808082b, 0x08081919, 0x08081919, 0x08081919, 0x08082b08, 0x08081919, 0x08190819, 0x08081919,
    0x08191908, 0x08081919, 0x082b0808, 0x08081919, 0x19080819, 0x08081919, 0x19081908, 0x08081919,
    0x19190808, 0x08081919, 0x192b0819, 0x08081919, 0x2b080808, 0x08081919, 0x08080819, 0x0808192b,
    0x08081908, 0x0808192b, 0x08190808, 0x0808192b, 0x082b192b, 0x0808192b, 0x19080808, 0x0808192b,
    0x1908082b, 0x0808192b, 0x2b081908, 0x0808192b, 0x08080808, 0x08082b08, 0x0808082b, 0x08082b08,
    0x08081919, 0x08082b08, 0x08082b08, 0x08082b08, 0x08082b2b, 0x08082b08, 0x08190819, 0x08082b08,
    0x08191908, 0x08082b08, 0x082b0808, 0x08082b08, 0x082b1919, 0x08082b08, 0x19080819, 0x08082b08,
    0x19081908, 0x08082b08, 0x19190808, 0x08082b08, 0x19192b08, 0x08082b08, 0x2b080808, 0x08082b08,
    0x2b2b0808, 0x08082b08, 0x2b2b2b2b, 0x08082b08, 0x08080819, 0x08082b19, 0x08081908, 0x08082b19,
    0x08190808, 0x08082b19, 0x19080808, 0x08082b19, 0x2b080819, 0x08082b19, 0x2b082b19, 0x08082b19,
    0x08080808, 0x08082b2b, 0x082b0808, 0x08082b2b, 0x082b2b08, 0x08082b2b, 0x2b19192b, 0x08082b2b,
    0x2b2b0808, 0x08082b2b, 0x08080819, 0x08190808, 0x08081908, 0x08190808, 0x0808192b, 0x08190808,
    0x08082b19, 0x08190808, 0x08190808, 0x08190808, 0x0819082b, 0x08190808, 0x08191919, 0x08190808,
    0x08192b08, 0x08190808, 0x082b0819, 0x08190808, 0x082b1908, 0x08190808, 0x19080808, 0x08190808,
    0x1908082b, 0x08190808, 0x19081919, 0x08190808, 0x19082b08, 0x08190808, 0x19190819, 0x08190808,
    0x19191908, 0x08190808, 0x192b0808, 0x08190808, 0x192b2b2b, 0x08190808, 0x2b080819, 0x08190808,
    0x2b081908, 0x08190808, 0x2b190808, 0x08190808, 0x08080808, 0x08190819, 0x0808082b, 0x08190819,
    0x08081919, 0x08190819, 0x08082b08, 0x08190819, 0x08190819, 0x08190819, 0x08191908, 0x08190819,
    0x082b0808, 0x08190819, 0x19080819, 0x08190819, 0x19081908, 0x08190819, 0x19190808, 0x08190819,
    0x2b080808, 0x08190819, 0x2b191908, 0x08190819, 0x2b19192b, 0x08190819, 0x08080819, 0x0819082b,
    0x08081908, 0x0819082b, 0x0808192b, 0x0819082b, 0x08190808, 0x0819082b, 0x19080808, 0x0819082b,
    0x192b0808, 0x0819082b, 0x08080808, 0x08191908, 0x0808082b, 0x08191908, 0x08081919, 0x08191908,
    0x08082b08, 0x08191908, 0x08190819, 0x08191908, 0x08191908, 0x08191908, 0x082b0808, 0x08191908,
    0x19080819, 0x08191908, 0x19081908, 0x08191908, 0x19082b19, 0x08191908, 0x19190808, 0x08191908,
    0x192b1908, 0x08191908, 0x2b080808, 0x08191908, 0x08080819, 0x08191919, 0x08081908, 0x08191919,
    0x08190808, 0x08191919, 0x19080808, 0x08191919, 0x08080808, 0x0819192b, 0x08191908, 0x0819192b,
    0x19082b19, 0x0819192b, 0x08080819, 0x08192b08, 0x08081908, 0x08192b08, 0x08190808, 0x08192b08,
    0x0819082b, 0x08192b08, 0x19080808, 0x08192b08, 0x19191908, 0x08192b08, 0x2b08192b, 0x08192b08,
    0x08080808, 0x08192b19, 0x08081919, 0x08192b19, 0x192b192b, 0x08192b19, 0x19190819, 0x08192b2b,
    0x2b2b2b19, 0x08192b2b, 0x08080808, 0x082b0808, 0x0808082b, 0x082b0808, 0x08081919, 0x082b0808,
    0x08082b08, 0x082b0808, 0x08082b2b, 0x082b0808, 0x08190819, 0x082b0808, 0x08191908, 0x082b0808,
    0x082b0808, 0x082b0808, 0x19080819, 0x082b0808, 0x19081908, 0x082b0808, 0x19190808, 0x082b0808,
    0x2b080808, 0x082b0808, 0x2b2b0808, 0x082b0808, 0x08080819, 0x082b0819, 0x08081908, 0x082b0819,
    0x08190808, 0x082b0819, 0x19080808, 0x082b0819, 0x19082b08, 0x082b0819, 0x192b1919, 0x082b0819,
    0x08080808, 0x082b082b, 0x082b082b, 0x082b082b, 0x2b080808, 0x082b082b, 0x2b2b2b08, 0x082b082b,
    0x08080819, 0x082b1908, 0x08081908, 0x082b1908, 0x08190808, 0x082b1908, 0x082b2b19, 0x082b1908,
    0x19080808, 0x082b1908, 0x08080808, 0x082b1919, 0x19080819, 0x082b1919, 0x1919082b, 0x082b1919,
    0x2b192b19, 0x082b1919, 0x08080819, 0x082b192b, 0x08192b2b, 0x082b192b, 0x2b2b192b, 0x082b192b,
    0x08080808, 0x082b2b08, 0x08082b08, 0x082b2b08, 0x08082b2b, 0x082b2b08, 0x082b0808, 0x082b2b08,
    0x19191919, 0x082b2b08, 0x2b082b08, 0x082b2b08, 0x2b2b082b, 0x082b2b08, 0x192b2b08, 0x082b2b19,
    0x2b190808, 0x082b2b19, 0x08082b08, 0x082b2b2b, 0x082b0808, 0x082b2b2b, 0x2b08082b, 0x082b2b2b,
    0x2b082b08, 0x082b2b2b, 0x2b082b2b, 0x082b2b2b, 0x08080819, 0x19080808, 0x08081908, 0x19080808,
    0x0808192b, 0x19080808, 0x08082b19, 0x19080808, 0x08190808, 0x19080808, 0x0819082b, 0x19080808,
    0x08191919, 0x19080808, 0x08192b08, 0x19080808, 0x082b0819, 0x19080808, 0x082b1908, 0x19080808,
    0x19080808, 0x19080808, 0x1908082b, 0x19080808, 0x19081919, 0x19080808, 0x19082b08, 0x19080808,
    0x19082b2b, 0x19080808, 0x19190819, 0x19080808, 0x19191908, 0x19080808, 0x192b0808, 0x19080808,
    0x192b1919, 0x19080808, 0x2b080819, 0x19080808, 0x2b081908, 0x19080808, 0x2b190808, 0x19080808,
    0x08080808, 0x19080819, 0x0808082b, 0x19080819, 0x08081919, 0x19080819, 0x08082b08, 0x19080819,
    0x08190819, 0x19080819, 0x08191908, 0x19080819, 0x082b0808, 0x19080819, 0x19080819, 0x19080819,
    0x19081908, 0x19080819, 0x19190808, 0x19080819, 0x2b080808, 0x19080819, 0x2b081919, 0x19080819,
    0x2b2b082b, 0x19080819, 0x08080819, 0x1908082b, 0x08081908, 0x1908082b, 0x08190808, 0x1908082b,
    0x0819082b, 0x1908082b, 0x082b2b19, 0x1908082b, 0x19080808, 0x1908082b, 0x08080808, 0x19081908,
    0x0808082b, 0x19081908, 0x08081919, 0x19081908, 0x08082b08, 0x19081908, 0x08190819, 0x19081908,
    0x08191908, 0x19081908, 0x08192b19, 0x19081908, 0x082b0808, 0x19081908, 0x19080819, 0x19081908,
    0x19081908, 0x19081908, 0x19190808, 0x19081908, 0x2b080808, 0x19081908, 0x2b191908, 0x19081908,
    0x08080819, 0x19081919, 0x08081908, 0x19081919, 0x08190808, 0x19081919, 0x082b1908, 0x19081919,
    0x19080808, 0x19081919, 0x2b192b2b, 0x19081919, 0x08080808, 0x1908192b, 0x08082b2b, 0x1908192b,
    0x19081908, 0x1908192b, 0x19190808, 0x1908192b, 0x08080819, 0x19082b08, 0x08081908, 0x19082b08,
    0x08190808, 0x19082b08, 0x19080808, 0x19082b08, 0x19081919, 0x19082b08, 0x19191908, 0x19082b08,
    0x192b082b, 0x19082b08, 0x08080808, 0x19082b19, 0x08190819, 0x19082b19, 0x19081908, 0x19082b19,
    0x19190808, 0x19082b19, 0x192b2b19, 0x19082b19, 0x08081908, 0x19082b2b, 0x08080808, 0x19190808,
    0x0808082b, 0x19190808, 0x08081919, 0x19190808, 0x08082b08, 0x19190808, 0x08190819, 0x19190808,
    0x08191908, 0x19190808, 0x082b0808, 0x19190808, 0x082b2b08, 0x19190808, 0x19080819, 0x19190808,
    0x19081908, 0x19190808, 0x19190808, 0x19190808, 0x2b080808, 0x19190808, 0x08080819, 0x19190819,
    0x08081908, 0x19190819, 0x08190808, 0x19190819, 0x08191919, 0x19190819, 0x19080808, 0x19190819,
    0x1908082b, 0x19190819, 0x08080808, 0x1919082b, 0x19081908, 0x1919082b, 0x2b2b2b2b, 0x1919082b,
    0x08080819, 0x19191908, 0x08081908, 0x19191908, 0x08190808, 0x19191908, 0x082b0819, 0x19191908,
    0x19080808, 0x19191908, 0x192b0808, 0x19191908, 0x2b080819, 0x19191908, 0x2b2b0819, 0x19191908,
    0x08080808, 0x19191919, 0x08082b08, 0x19191919, 0x2b080808, 0x19191919, 0x2b082b08, 0x19191919,
    0x082b0819, 0x1919192b, 0x192b2b08, 0x1919192b, 0x2b2b0819, 0x1919192b, 0x08080808, 0x19192b08,
    0x08191908, 0x19192b08, 0x19080819, 0x19192b08, 0x19190808, 0x19192b08, 0x2b192b19, 0x19192b08,
    0x08192b2b, 0x19192b19, 0x19080808, 0x19192b19, 0x1908082b, 0x19192b19, 0x2b081919, 0x19192b2b,
    0x08080819, 0x192b0808, 0x08081908, 0x192b0808, 0x08190808, 0x192b0808, 0x19080808, 0x192b0808,
    0x19191908, 0x192b0808, 0x192b082b, 0x192b0808, 0x2b08192b, 0x192b0808, 0x2b2b2b19, 0x192b0808,
    0x08080808, 0x192b0819, 0x082b1908, 0x192b082b, 0x19082b2b, 0x192b082b, 0x2b19082b, 0x192b082b,
    0x08080808, 0x192b1908, 0x0819192b, 0x192b1908, 0x08190808, 0x192b1919, 0x19080808, 0x192b1919,
    0x19081919, 0x192b1919, 0x2b2b1908, 0x192b1919, 0x08080819, 0x192b2b08, 0x192b2b2b, 0x192b2b08,
    0x082b1919, 0x192b2b19, 0x0808192b, 0x192b2b2b, 0x19191908, 0x192b2b2b, 0x192b082b, 0x192b2b2b,
    0x08080808, 0x2b080808, 0x0808082b, 0x2b080808, 0x08081919, 0x2b080808, 0x08082b08, 0x2b080808,
    0x08190819, 0x2b080808, 0x08191908, 0x2b080808, 0x082b0808, 0x2b080808, 0x082b2b2b, 0x2b080808,
    0x19080819, 0x2b080808, 0x19081908, 0x2b080808, 0x19190808, 0x2b080808, 0x2b080808, 0x2b080808,
    0x2b08082b, 0x2b080808, 0x2b2b2b08, 0x2b080808, 0x2b2b2b2b, 0x2b080808, 0x08080819, 0x2b080819,
    0x08081908, 0x2b080819, 0x0808192b, 0x2b080819, 0x08190808, 0x2b080819, 0x19080808, 0x2b080819,
    0x19190819, 0x2b080819, 0x19192b19, 0x2b080819, 0x08080808, 0x2b08082b, 0x082b0808, 0x2b08082b,
    0x2b080808, 0x2b08082b, 0x2b08082b, 0x2b08082b, 0x2b2b0808, 0x2b08082b, 0x2b2b2b08, 0x2b08082b,
    0x08080819, 0x2b081908, 0x08081908, 0x2b081908, 0x08190808, 0x2b081908, 0x0819082b, 0x2b081908,
    0x08191919, 0x2b081908, 0x19080808, 0x2b081908, 0x192b0808, 0x2b081908, 0x2b082b19, 0x2b081908,
    0x08080808, 0x2b081919, 0x19081908, 0x2b081919, 0x2b2b1919, 0x2b081919, 0x08192b08, 0x2b08192b,
    0x192b2b2b, 0x2b08192b, 0x08080808, 0x2b082b08, 0x08082b08, 0x2b082b08, 0x082b1919, 0x2b082b08,
    0x19192b2b, 0x2b082b08, 0x2b080808, 0x2b082b08, 0x2b08082b, 0x2b082b08, 0x2b2b2b08, 0x2b082b08,
    0x0808192b, 0x2b082b19, 0x082b082b, 0x2b082b2b, 0x2b080808, 0x2b082b2b, 0x2b082b08, 0x2b082b2b,
    0x2b19192b, 0x2b082b2b, 0x2b2b2b08, 0x2b082b2b, 0x08080819, 0x2b190808, 0x08081908, 0x2b190808,
    0x08190808, 0x2b190808, 0x19080808, 0x2b190808, 0x1919192b, 0x2b190808, 0x2b081908, 0x2b190808,
    0x08080808, 0x2b190819, 0x082b082b, 0x2b190819, 0x192b1908, 0x2b190819, 0x1919192b, 0x2b19082b,
    0x2b082b19, 0x2b19082b, 0x08080808, 0x2b191908, 0x08081919, 0x2b191908, 0x19081908, 0x2b191908,
    0x19190808, 0x2b191908, 0x19192b08, 0x2b191908, 0x082b2b19, 0x2b191919, 0x2b190808, 0x2b191919,
    0x2b19082b, 0x2b191919, 0x19080819, 0x2b19192b, 0x19190819, 0x2b192b08, 0x2b2b192b, 0x2b192b08,
    0x19082b19, 0x2b192b19, 0x08191919, 0x2b192b2b, 0x192b0808, 0x2b192b2b, 0x08080808, 0x2b2b0808,
    0x0808082b, 0x2b2b0808, 0x08082b08, 0x2b2b0808, 0x08082b2b, 0x2b2b0808, 0x082b0808, 0x2b2b0808,
    0x082b2b2b, 0x2b2b0808, 0x2b2b0808, 0x2b2b0808, 0x19190819, 0x2b2b0819, 0x19192b19, 0x2b2b0819,
    0x2b2b192b, 0x2b2b0819, 0x08080808, 0x2b2b082b, 0x0808082b, 0x2b2b082b, 0x08082b08, 0x2b2b082b,
    0x082b2b2b, 0x2b2b082b, 0x2b080808, 0x2b2b082b, 0x2b2b0808, 0x2b2b082b, 0x19080808, 0x2b2b1908,
    0x2b191919, 0x2b2b1908, 0x192b1919, 0x2b2b192b, 0x2b192b08, 0x2b2b192b, 0x08082b2b, 0x2b2b2b08,
    0x082b0808, 0x2b2b2b08, 0x082b082b, 0x2b2b2b08, 0x082b2b08, 0x2b2b2b08, 0x2b2b0808, 0x2b2b2b08,
    0x2b2b2b08, 0x2b2b2b08, 0x08081908, 0x2b2b2b19, 0x2b081908, 0x2b2b2b19, 0x2b08192b, 0x2b2b2b19,
    0x082b2b08, 0x2b2b2b2b, 0x082b2b2b, 0x2b2b2b2b, 0x2b190819, 0x2b2b2b2b, 0x2b2b2b2b, 0x2b2b2b2b
};

// ksigns_iq2xs[v] == v | ((popcount(v) & 1) << 7), computed rather than read;
// byte-indexed __constant loads serialize on Adreno.
inline uint iq2xs_signs(uint code7) {
    return code7 | ((uint)(popcount(code7) & 1u) << 7);
}

#ifndef IQ2XS_MV_SIGNXOR
#define IQ2XS_MV_SIGNXOR 1
#endif

// Four grid bytes with their signs applied, as floats. base picks which nibble
// of the sign byte this half of the entry uses: 0 for the low uint, 4 for the
// high one.
inline float4 iq2xs_vals(uint gv, uint sgv, uint base) {
#if IQ2XS_MV_SIGNXOR
    const uint  s   = sgv >> base;
    const uint4 sgn = (uint4)(s << 31, s << 30, s << 29, s << 28) & 0x80000000u;
    return as_float4(as_uint4(convert_float4(as_uchar4(gv))) ^ sgn);
#else
    const uint s = sgv >> base;
    float4 v;
    v.s0 = (float)((gv      ) & 0xFFu); if (s & 1u) { v.s0 = -v.s0; }
    v.s1 = (float)((gv >>  8) & 0xFFu); if (s & 2u) { v.s1 = -v.s1; }
    v.s2 = (float)((gv >> 16) & 0xFFu); if (s & 4u) { v.s2 = -v.s2; }
    v.s3 = (float)((gv >> 24) & 0xFFu); if (s & 8u) { v.s3 = -v.s3; }
    return v;
#endif
}

kernel void kernel_mul_mv_iq2_xs_f32_flat(
        __read_only image1d_buffer_t grid_img,
        global const ushort * src0_qs,
        global const uchar  * src0_sc,
        global const half   * src0_d,
        global const float  * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,      // K
        int ne01,      // M, the number of output rows
        int ne10,      // activation row stride, == K
        int ne0        // dst row stride
) {
    src1 = (global const float *)((global const char *)src1 + offset1);
    dst  = (global float       *)((global char       *)dst  + offsetd);

    const uint m   = (uint)ne01;
    const uint K   = (uint)ne00;
    const uint nsb = K / QK_K;                  // super blocks along K

    const uint lid = get_local_id(0);           // lane
    const uint sgi = get_local_id(1);           // K-split slice
    const uint col = get_group_id(1);           // token

    global const float * y = src1 + (ulong)col * (uint)ne10;

#if IQ2XS_MV_GRIDIMG
#define IQ2XS_GRID(i) (read_imageui(grid_img, (int)(i)).x)
#else
#define IQ2XS_GRID(i) iq2xs_grid[(i)]
#endif

#if IQ2XS_MV_R2
    const uint mh  = m >> 1;                        // rows per plane row, as pairs
    const uint j   = get_group_id(0) * 64u + lid;   // row pair index
    const uint row = j << 1;

    float sumf  = 0.f;
    float sumf1 = 0.f;

    if (j < mh) {
        global const uint   * qsu = (global const uint   *)src0_qs;
        global const ushort * scu = (global const ushort *)src0_sc;

        for (uint ib = sgi; ib < nsb; ib += IQ2XS_MV_NSG) {
            const half2 dh = vload2(j + ib * mh, src0_d);
            const float d0 = (float)dh.s0;
            const float d1 = (float)dh.s1;

            float acc0 = 0.f, acc1 = 0.f;
            for (uint sb = 0; sb < 8u; ++sb) {
                const uint sub = ib * 8u + sb;
                const uint scv = (uint)scu[j + sub * mh];       // row pair, one load

                const float lo0 = 0.5f + (float)( scv        & 0xFu);
                const float hi0 = 0.5f + (float)((scv >>  4) & 0xFu);
                const float lo1 = 0.5f + (float)((scv >>  8) & 0xFu);
                const float hi1 = 0.5f + (float)((scv >> 12) & 0xFu);

                const uint grp = sub * 4u;      // quant plane units of 8 weights
                const uint qsb = j + grp * mh;

                float alo0 = 0.f, ahi0 = 0.f, alo1 = 0.f, ahi1 = 0.f;
                for (uint u = 0; u < 4u; ++u) {
                    const uint qsv = qsu[qsb + u * mh];         // row pair, one load
                    const uint q0  = qsv & 0xFFFFu;
                    const uint q1  = qsv >> 16;

                    const uint g0  = (q0 & 511u) << 1;          // uint pair index
                    const uint g1  = (q1 & 511u) << 1;
                    const uint sg0 = iq2xs_signs(q0 >> 9);
                    const uint sg1 = iq2xs_signs(q1 >> 9);

                    const float4 yl = vload4(sub * 8u + u * 2u + 0u, y);
                    const float4 yh = vload4(sub * 8u + u * 2u + 1u, y);

                    const float a0 = dot(yl, iq2xs_vals(IQ2XS_GRID(g0    ), sg0, 0u))
                                   + dot(yh, iq2xs_vals(IQ2XS_GRID(g0 + 1), sg0, 4u));
                    const float a1 = dot(yl, iq2xs_vals(IQ2XS_GRID(g1    ), sg1, 0u))
                                   + dot(yh, iq2xs_vals(IQ2XS_GRID(g1 + 1), sg1, 4u));

                    if (u < 2u) { alo0 += a0; alo1 += a1; }
                    else        { ahi0 += a0; ahi1 += a1; }
                }
                acc0 += lo0 * alo0 + hi0 * ahi0;
                acc1 += lo1 * alo1 + hi1 * ahi1;
            }
            sumf  += d0 * 0.25f * acc0;
            sumf1 += d1 * 0.25f * acc1;
        }
    }
#else
    const uint row = get_group_id(0) * 64u + lid;

    float sumf = 0.f;

    if (row < m) {
        for (uint ib = sgi; ib < nsb; ib += IQ2XS_MV_NSG) {
            const float d = (float)src0_d[row + ib * m];

            float acc = 0.f;
            for (uint sb = 0; sb < 8u; ++sb) {
                const uint sub = ib * 8u + sb;
                const uint scv = (uint)src0_sc[row + sub * m];

                const float lo = 0.5f + (float)( scv       & 0xFu);
                const float hi = 0.5f + (float)((scv >> 4) & 0xFu);

                const uint grp = sub * 4u;
                const uint qsb = row + grp * m;

                float alo = 0.f, ahi = 0.f;
                for (uint u = 0; u < 4u; ++u) {
                    const uint q  = (uint)src0_qs[qsb + u * m];
                    const uint g  = (q & 511u) << 1;
                    const uint sg = iq2xs_signs(q >> 9);

                    const float4 yl = vload4(sub * 8u + u * 2u + 0u, y);
                    const float4 yh = vload4(sub * 8u + u * 2u + 1u, y);

                    const float a = dot(yl, iq2xs_vals(IQ2XS_GRID(g    ), sg, 0u))
                                  + dot(yh, iq2xs_vals(IQ2XS_GRID(g + 1), sg, 4u));

                    if (u < 2u) { alo += a; } else { ahi += a; }
                }
                acc += lo * alo + hi * ahi;
            }
            sumf += d * 0.25f * acc;
        }
    }
#endif

#if IQ2XS_MV_NSG > 1
#if IQ2XS_MV_R2
    __local float2 part[IQ2XS_MV_NSG][64];
    part[sgi][lid] = (float2)(sumf, sumf1);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sgi != 0) {
        return;
    }
    for (uint s = 1; s < IQ2XS_MV_NSG; ++s) {
        const float2 p = part[s][lid];
        sumf  += p.s0;
        sumf1 += p.s1;
    }
#else
    __local float part[IQ2XS_MV_NSG][64];
    part[sgi][lid] = sumf;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sgi != 0) {
        return;
    }
    for (uint s = 1; s < IQ2XS_MV_NSG; ++s) {
        sumf += part[s][lid];
    }
#endif
#endif

#if IQ2XS_MV_R2
    if (j < mh) {
        vstore2((float2)(sumf, sumf1), 0, dst + (ulong)col * (uint)ne0 + row);
    }
#else
    if (row < m) {
        dst[(ulong)col * (uint)ne0 + row] = sumf;
    }
#endif
}

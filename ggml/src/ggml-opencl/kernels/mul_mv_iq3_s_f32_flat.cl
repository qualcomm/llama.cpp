#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#ifdef cl_khr_integer_dot_product
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#endif

// IQ3_S decode GEMV over the feature-major plane split.
//
// kernel_convert_block_iq3_s_ns splits the tensor in place into five planes --
// size preserving at 64 + 8 + 32 + 4 + 2 == 110 == sizeof(block_iq3_s) -- and
// transposes each one, so every plane is indexed [k-group][row]:
//
//   src0_qs[row + (k/4)*m]    uchar  grid index low 8 bits, K = 4*grp .. +3
//   src0_qh[row + (k/32)*m]   uchar  one 9th index bit per operand
//   src0_sg[row + (k/8)*m]    uchar  8 sign bits, 2 operands each
//   src0_sc[row + (k/64)*m]   uchar  two 4-bit sub-scales
//   src0_d [row + (k/256)*m]  half   super-block scale
//
// This is mul_mv_iq4_xs_f32_flat with the codebook nibble replaced by the
// iq3s_grid + sign unpack:
//   grid index of operand u = qs[u] | (((qh >> u) & 1) << 8)
//   sign of value j         = bit (4*(u&1) + j) of sg[u/2]
//   sub-scale of block sb   = nibble (sb&1) of sc[sb/2], then d * (1 + 2*nib)
//
// Same two structural lessons as the IQ4_XS kernel, both measured there:
//   - one row per lane alone launches only M work items and LOSES to the AoS
//     kernel outright, so K is split across IQ3S_MV_NSG subgroups and the
//     partials reduced through local memory;
//   - a lane must move more than one plane element per load. IQ3_S is worse off
//     than IQ4_XS here -- its quant plane is one uchar per 4 weights, not a
//     ushort -- so IQ3S_MV_R2 pairs adjacent rows and reads every plane through
//     a ushort view, taking a 64-lane wave from 64 bytes to 128 per weight load.

#define QK_K 256

// Subgroups per workgroup; each takes every NSG'th super-block along K.
#ifndef IQ3S_MV_NSG
#define IQ3S_MV_NSG 8
#endif

// IQ3S_MV_R2=1: one lane owns TWO adjacent rows. Needs m even, which
// ggml_cl_iq3s_is_split() declines at conversion time rather than here.
// Qwen3.5-4B tg64: 1 row 15.14, 2 rows 19.73, against the AoS kernel's 20.55.
//
// FOUR rows per lane -- what it would take to read a whole uint, the width
// IQ4_XS already gets from two -- was built and is a REGRESSION: 17.71 against
// 18.06 for two rows in the same generic form. It quarters the grid, and this
// kernel is short of work items to begin with; that is the whole reason the
// K-split across subgroups exists. The per-lane row count and the launch shape
// are ONE decision, not two. Generalising the hand-written pair to an unrolled
// r-loop also cost 8% by itself (19.73 -> 18.06), so the pair is what ships.
#ifndef IQ3S_MV_R2
#define IQ3S_MV_R2 1
#endif

// IQ3S_MV_R4=1: the PLAIN decode GEMV folds FOUR rows into a work item instead
// of two, so every plane is read a uint at a time and one activation vload4
// serves four rows.
//
// 🔑 PER KERNEL ON PURPOSE. The fold is bounded by the ~512 B/WI spill cliff, and
// crossing it costs more occupancy than the fold returns however the extra state
// is spent: IQ1_S's fused-GLU kernel goes 408 -> 648 B/WI at four rows and took
// that whole type's fold negative, while its plain and split-K kernels went
// 272 -> 400/384 and gained. So this applies to the plain kernel only, which is
// also the one that matters: it is 100% of the iq3_s time in a 3B IQ3_M decode
// (3425 dispatches, 487.6 ms) because the fusion is default off here and the
// split-K heuristic leaves this shape at ksplit=1.
//
// 🔑 WHY THIS TYPE AND NOT ITS SIBLINGS. The fold and the fused GLU are
// SUBSTITUTES -- both exist to collect the same row-independent activation read,
// so whichever lands first takes the win and the other finds nothing left. IQ1_S
// and IQ2_S run their fusion by default and measure the fold as a wash; IQ3_S's
// fusion is default OFF (it measured -4.9%), so nothing has collected it here.
//
// ⚠ Unlike its siblings this kernel IS compute-bound -- IQ3S_MV_WORK=3 costs 33%
// -- so the fold does not reduce total work, it only widens the loads and shares
// the activation. Measured, not assumed.
#ifndef IQ3S_MV_R4
#define IQ3S_MV_R4 0
#endif

// IQ3S_MV_LDSGRID=1: stage iq3s_grid into local memory once per workgroup and
// read it from there.
//
// iq3s_grid is 512 uints = 2 KB and every lane indexes it divergently with a
// 9-bit index. The IQ2_S GEMV took +55% from this move, but its grid is 8 KB.
//
// MEASURED AND REFUTED: X2-90, q4b-IQ3_S tg64 19.68 -> 16.44, -16.5%. Stays off.
// The table size is the whole story -- the same move is -5.3% on IQ3_XXS's 1 KB
// grid and -37% on the IQ4_XS GEMV's 64-byte codebook. IQ2_S at 8 KB is the only
// table big enough to be worth the local-memory traffic and the extra barrier.
//
#ifndef IQ3S_MV_LDSGRID
#define IQ3S_MV_LDSGRID 0
#endif
// IQ3S_MV_GRIDSRC=1: put the grid in the __global address space at program scope
// instead of __constant.
//
// The question this asks is the one the IQ4_XS codebook answered: Adreno takes
// __constant through the uniform path only for a wave-uniform index, and this
// index is per weight. There the fix was to leave memory entirely, which a
// 512-entry table cannot do, and local memory is already measured at -16.5% here.
// A program-scope __global array is the remaining way to ask whether the
// __constant PATH is the cost, and it needs no host plumbing at all.
//
// Program-scope variables in __global are OpenCL C 2.0; the backend compiles with
// -cl-std=CL<device version>, so this arm simply fails to build on a device that
// does not support them. That is a legitimate answer, not a crash.
//
// MEASURED AND REFUTED on X2-90: it builds, so the device does support it, and it
// is SLOWER -- q4b-IQ3_S tg64 19.68 -> 18.20, -7.5%. Left off.
//
// 🔑 So a 2 KB table is not paying for the __constant PATH the way the 16-entry
// IQ4_XS codebook was. That one was fixable because it was small enough to become
// immediates; this one is not, and the address space is not the lever. Together
// with the local-memory arm (-16.5%) and the two knobs below, this kernel is at a
// local optimum for its current algorithm:
//
//   IQ3S_MV_NSG   2 / 4 / 8 / 16  ->  14.19 / 17.75 / 19.70 / 19.26  (8 is best)
//   IQ3S_MV_R2    0 / 1           ->  15.12 / 19.68
//
// NSG flat-to-worse above 8 says the grid's dependent load is not latency more
// occupancy can hide. What is left is algorithmic -- fewer ALU ops per weight via
// dp4a over quantised activations -- not another way to read the same table.
#ifndef IQ3S_MV_GRIDSRC
#define IQ3S_MV_GRIDSRC 0
#endif

#if IQ3S_MV_GRIDSRC == 1
#define IQ3S_GRID_AS __global
#else
#define IQ3S_GRID_AS constant
#endif

IQ3S_GRID_AS uint iq3s_grid[512] = {
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101
};

// IQ3S_MV_ABL: COST PROBE, WRONG MATH. Answers which of this kernel's three
// per-operand costs it is actually paying for, before anything gets built.
//
// Why ask: shape-matched in one 27B decode profile at out=17408, this kernel is
// 0.7925 ms/call where q3_K -- the SAME 3.4375 bits per weight, but a linear
// quant with no grid -- is 0.5186, and the tuned q4_K GEMV is 0.4217. That is
// 48 GB/s of a 152 GB/s bus. The IQ4_XS GEMV had the same signature and it was
// entirely the codebook, so the grid is the suspect here; but the IQ4_XS round
// also began with two wrong hypotheses that one probe each killed cheaply.
//
//   1  drop the activation load (expected near-null; it was +1.9/+4.7% there)
//   2  drop the GRID lookup, keeping every weight and sign load
//   3  drop the SIGN application, keeping the grid lookup
//
// Never enable in a real run.
#ifndef IQ3S_MV_ABL
#define IQ3S_MV_ABL 0
#endif

// IQ3S_MV_AIMG=1: read the ACTIVATION through an image1d_buffer instead of the
// global pointer.
//
// Why this one and not the grid: the kernel's own cost probe prices every term of
// this loop on a 3B IQ3_M, and the answer is not what the header above assumed.
//     ABL=2, drop the GRID lookup      32.480 vs 32.402   0.2%
//     ABL=1, drop the ACTIVATION load  36.666 vs 32.402  13.2%
//     ABL=3, drop the SIGN application 37.005 vs 32.402  14.2%
// The codebook gather is FREE here. The activation load is not, and it is
// 64x REDUNDANT: `grp` carries no row index, so every lane of a subgroup reads
// the identical address. A wave-uniform image read is established as free on this
// part, where LDS staging of the same redundancy measured -50%
// (see the cok weight-texture note), so the texture path is the one to try.
//
// The image is CL_RGBA/CL_FLOAT over src1's own buffer, so one texel is exactly
// the float4 the scalar path loads and no copy is involved. y_tex is the texel
// index of this token's row: (offset1/16) + col*ne10/4. The host declines unless
// offset1 is 16-byte aligned and ne10 is a multiple of 4.
#ifndef IQ3S_MV_AIMG
#define IQ3S_MV_AIMG 0
#endif

#if IQ3S_MV_ABL == 1
#define IQ3S_YV(g, y) ((float4)(1.0f))
#elif IQ3S_MV_AIMG
#define IQ3S_YV(g, y) read_imagef(y_img, (int)(y_tex + (g)))
#else
#define IQ3S_YV(g, y) vload4((g), (y))
#endif

// IQ_MV_SIGNXOR=1: apply the per-weight signs by XOR-ing the float sign bit
// rather than by four conditional negations.
//
// Why: the IQ3_S cost probe (GGML_OPENCL_IQ3S_MV_ABL=3) says dropping the sign
// application entirely is worth 13.6 percent -- the second-largest cost in these
// kernels after the grid lookup, and ahead of the activation load at 5.5. The
// three grid types share this helper verbatim and are together about 30 percent
// of Qwen3.8-27B decode.
//
// MEASURED, AND IT DOES NOT TRANSFER: q4b-IQ3_S tg64 19.67 -> 18.75, -4.7% here,
// against +4.6% on IQ3_XXS running the identical helper. Off for this kernel.
#ifndef IQ3S_MV_SIGNXOR
#define IQ3S_MV_SIGNXOR 0
#endif

// IQ3S_MV_SGNMUL=1: apply the four signs as ONE vector multiply by a +-1 float4
// taken from a 16-entry table, instead of four conditional negations.
//
// This is the third form of the same idea, and it exists because the other two
// bracket it. IQ3S_MV_SIGNXOR spreads the sign bits by hand (four shifts, a uint4
// build and an as_float4 round trip) and measures -28.5% here. The pre-signed
// table (IQ3S_MV_SGRID) removes the arithmetic outright and is a WASH, because at
// 32 KB it stops being cache-resident and the gather costs back what it saved.
//
// So the target is the arithmetic, and the constraint is the HOT TABLE SIZE.
// iq3s_sgn4 is 16 float4 = 256 bytes, which is an order of magnitude inside the
// 1-2 KB tier that the fleet's __constant probe measured as fastest, and it is
// per-nibble rather than per (grid entry, nibble) -- so it does not scale with the
// grid the way the pre-signed table did.
#ifndef IQ3S_MV_SGNMUL
#define IQ3S_MV_SGNMUL 0
#endif

#if IQ3S_MV_SGNMUL
// Bit k of the index negates lane k, matching the four `if (s & 1<<k)` below.
constant float4 iq3s_sgn4[16] = {
    (float4)( 1.f, 1.f, 1.f, 1.f), (float4)(-1.f, 1.f, 1.f, 1.f),
    (float4)( 1.f,-1.f, 1.f, 1.f), (float4)(-1.f,-1.f, 1.f, 1.f),
    (float4)( 1.f, 1.f,-1.f, 1.f), (float4)(-1.f, 1.f,-1.f, 1.f),
    (float4)( 1.f,-1.f,-1.f, 1.f), (float4)(-1.f,-1.f,-1.f, 1.f),
    (float4)( 1.f, 1.f, 1.f,-1.f), (float4)(-1.f, 1.f, 1.f,-1.f),
    (float4)( 1.f,-1.f, 1.f,-1.f), (float4)(-1.f,-1.f, 1.f,-1.f),
    (float4)( 1.f, 1.f,-1.f,-1.f), (float4)(-1.f, 1.f,-1.f,-1.f),
    (float4)( 1.f,-1.f,-1.f,-1.f), (float4)(-1.f,-1.f,-1.f,-1.f),
};
#endif

// Four grid values with their signs applied. base picks the nibble of sgv.
inline float4 iq3s_vals(uint gv, uint sgv, uint base) {
#if IQ3S_MV_ABL == 3
    return (float4)((float)((gv      ) & 0xFFu), (float)((gv >>  8) & 0xFFu),
                    (float)((gv >> 16) & 0xFFu), (float)((gv >> 24) & 0xFFu));
#else
#if IQ3S_MV_SGNMUL
    // Exact: multiplying a float by +-1.0f is a sign flip, no rounding.
    return convert_float4(as_uchar4(gv)) * iq3s_sgn4[(sgv >> base) & 0xFu];
#elif IQ3S_MV_SIGNXOR
    // A sign flip is bit 31, so the four conditional negations collapse to one
    // XOR once the four sign bits are spread into place. Exact, not approximate.
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
#endif
}

// IQ3S_MV_WORK: COST PROBE, WRONG MATH. Settles "is this compute-bound or
// memory-bound" by varying the WORK while holding the LOADS fixed -- the one
// discriminator the ablation probes above cannot give, because each of those
// removes work AND its load together.
//
//   2  do the grid read + sign + dot TWICE per operand (2x lookups, 2x ALU,
//      loads of qs/qh/sg/sc unchanged)
//   3  do the sign + dot twice on the ALREADY-FETCHED grid word (1x lookup,
//      2x ALU, same loads)
//
// Reading it: 2 vs 1 is total compute sensitivity, 3 vs 1 is pure ALU
// sensitivity, and 2 vs 3 isolates the divergent grid lookup. If a kernel is
// bandwidth-bound all three are flat. Indices/signs are perturbed so the
// compiler cannot common-subexpression the duplicate away.
//
// Never enable in a real run.
#ifndef IQ3S_MV_WORK
#define IQ3S_MV_WORK 0
#endif

// IQ3S_MV_GRIDIMG=1: read the grid through an image1d_buffer.
//
// This grid is still on __constant, where local memory was measured NEGATIVE
// (-16.5% on IQ3_S, -5.3% on IQ3_XXS -- the table is too small to earn the LDS
// traffic). The image is the remaining tier, and on the two kernels big enough to
// want LDS it is better than LDS: IQ1_S +2.5%, IQ2_S +4.4%. So it is worth asking
// here even though the LDS answer was no.
#ifndef IQ3S_MV_GRIDIMG
#define IQ3S_MV_GRIDIMG 0
#endif

kernel void kernel_iq3s_grid_export(global uint * out) {
    const uint i = get_global_id(0);
    if (i < 512u) {
        out[i] = iq3s_grid[i];
    }
}

// IQ3S_MV_SGRID=1: read the four weights with their SIGNS ALREADY APPLIED, out of
// a pre-signed table, instead of applying the signs in the inner loop.
//
// Why here and not on the types this was rejected for. The cost probe above says
// the sign application is 14.2% of this kernel while the grid gather is 0.2%
// (ABL=3 vs ABL=2 on a 3B IQ3_M), and IQ3S_MV_WORK=3 costs 33%, so IQ3_S is
// COMPUTE-bound -- the opposite of IQ1_S/Q2_K, which are flat under doubled
// arithmetic. A pre-signed table trades table size, which is nearly free here,
// for the one term that is not.
//
// 🔑 THE SIZE ARGUMENT THAT KILLED THIS BEFORE WAS ARITHMETIC, AND IT WAS WRONG.
// It was recorded as "the operand depends on (grid entry, sign byte), which is
// 512 x 256 combinations" = 128 K entries, obviously too big. But a grid entry is
// FOUR weights, so only FOUR sign bits ever apply to it -- `iq3s_vals` takes a
// nibble, not a byte. The real product is 512 x 16 = 8192 entries = 32 KB, the
// same size as the IQ1_M biased operand table that shipped.
//
// The table is int8, which the grid's value range allows outright: every byte of
// iq3s_grid is in {1,3,5,7,9,11,13,15}, so negating it stays inside int8. One
// convert_float4(as_char4(v)) then replaces four extracts plus four conditional
// negations.
//
// Entry (g, s) is iq3s_grid[g] with byte k negated when bit k of s is set, laid
// out at index g*16 + s so that the 16 sign variants of one grid entry are
// contiguous -- consecutive indices for a fixed g, which is the access pattern
// when a row's sign nibble changes and its grid index does not.
//
// Needs the image tier: 32 KB is past the 2-4 KB __constant cache cliff measured
// on this fleet, so the host binds it as an image1d_buffer and declines the mode
// otherwise.
#ifndef IQ3S_MV_SGRID
#define IQ3S_MV_SGRID 0
#endif

kernel void kernel_iq3s_signed_grid_export(global uint * out) {
    const uint i = get_global_id(0);
    if (i < 8192u) {
        const uint g = i >> 4;          // grid entry
        const uint s = i & 0xFu;        // sign nibble
        const uint gv = iq3s_grid[g];
        char4 v;
        v.s0 = (char)((gv      ) & 0xFFu); if (s & 1u) { v.s0 = -v.s0; }
        v.s1 = (char)((gv >>  8) & 0xFFu); if (s & 2u) { v.s1 = -v.s1; }
        v.s2 = (char)((gv >> 16) & 0xFFu); if (s & 4u) { v.s2 = -v.s2; }
        v.s3 = (char)((gv >> 24) & 0xFFu); if (s & 8u) { v.s3 = -v.s3; }
        out[i] = as_uint(v);
    }
}

// One signed-table fetch: index by grid entry and sign nibble, unpack four int8.
#define IQ3S_SVALS(g, sgv, base) \
    convert_float4(as_char4(read_imageui(sgrid_img, \
        (int)(((g) << 4) | (((sgv) >> (base)) & 0xFu))).x))

// One weight quad, from whichever tier is enabled. `g` is the grid index and
// `gw` the already-fetched grid word; the signed table needs the former, the
// sign-applying path the latter, so both are passed and one is discarded.
#if IQ3S_MV_SGRID
#define IQ3S_W(g, gw, sgv, base) IQ3S_SVALS(g, sgv, base)
#define IQ3S_GW(g, expr)         (0u)
#else
#define IQ3S_W(g, gw, sgv, base) iq3s_vals(gw, sgv, base)
#define IQ3S_GW(g, expr)         (expr)
#endif

kernel void kernel_mul_mv_iq3_s_f32_flat(
        __read_only image1d_buffer_t grid_img,
        __read_only image1d_buffer_t y_img,      // see IQ3S_MV_AIMG
        global const uchar * src0_qs,
        global const uchar * src0_qh,
        global const uchar * src0_sg,
        global const uchar * src0_sc,
        global const half  * src0_d,
        global const float * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,      // K
        int ne01,      // M, the number of output rows
        int ne10,      // activation row stride, == K
        int ne0,       // dst row stride
        uint y_off,     // offset1/16, in float4 texels (IQ3S_MV_AIMG only)
        __read_only image1d_buffer_t sgrid_img   // see IQ3S_MV_SGRID
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
#if IQ3S_MV_AIMG
    // texel index of this token's activation row; y_off is offset1/16 from the host
    const uint y_tex = y_off + col * ((uint)ne10 >> 2);
#endif

#if IQ3S_MV_GRIDIMG
#define IQ3S_GRID(i) (read_imageui(grid_img, (int)(i)).x)
#elif IQ3S_MV_LDSGRID
    __local uint sh_grid[512];
    {
        const uint tid  = sgi * 64u + lid;
        const uint nthr = (uint)(get_local_size(0) * get_local_size(1));
        for (uint i = tid; i < 512u; i += nthr) {
            sh_grid[i] = iq3s_grid[i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#define IQ3S_GRID(i) sh_grid[(i)]
#elif IQ3S_MV_ABL == 2
// must still CONSUME the index, or the qs/qh loads that compute it are dead
// code and the probe silently measures "no grid AND no weights".
#define IQ3S_GRID(i) (((i) * 0x01010101u) | 0x01010101u)
#else
#define IQ3S_GRID(i) iq3s_grid[(i)]
#endif

#if IQ3S_MV_R4
    const uint mq  = m >> 2;                        // rows per plane row, as uints
    const uint j   = get_group_id(0) * 64u + lid;   // row quad index
    const uint row = j << 2;

    float sumf = 0.f, sumf1 = 0.f, sumf2 = 0.f, sumf3 = 0.f;

    if (j < mq) {
        global const uint * qsw = (global const uint *)src0_qs;
        global const uint * qhw = (global const uint *)src0_qh;
        global const uint * sgw = (global const uint *)src0_sg;
        global const uint * scw = (global const uint *)src0_sc;

        for (uint ib = sgi; ib < nsb; ib += IQ3S_MV_NSG) {
            const half4 dh = vload4(j + ib * mq, src0_d);

            const uint scbase = j + ib * 4u * mq;
            uint sc4[4];
            sc4[0] = scw[scbase + 0u * mq];
            sc4[1] = scw[scbase + 1u * mq];
            sc4[2] = scw[scbase + 2u * mq];
            sc4[3] = scw[scbase + 3u * mq];

            float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
            for (uint sb = 0; sb < 8u; ++sb) {
                const uint scv = sc4[sb >> 1];
                const uint sh  = (sb & 1u) ? 4u : 0u;   // which nibble of each row's byte
                const uint nib0 = ((scv      ) >> sh) & 0xFu;
                const uint nib1 = ((scv >>  8) >> sh) & 0xFu;
                const uint nib2 = ((scv >> 16) >> sh) & 0xFu;
                const uint nib3 = ((scv >> 24) >> sh) & 0xFu;

                const uint qhv = qhw[j + (ib * 8u + sb) * mq];
                const uint qh0 = (qhv      ) & 0xFFu;
                const uint qh1 = (qhv >>  8) & 0xFFu;
                const uint qh2 = (qhv >> 16) & 0xFFu;
                const uint qh3 = (qhv >> 24) & 0xFFu;

                const uint grp = ib * 64u + sb * 8u;
                const uint qsb = j + grp * mq;
                const uint sgb = j + (ib * 32u + sb * 4u) * mq;

                float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
                for (uint u = 0; u < 8u; ++u) {
                    const uint qsv = qsw[qsb + u * mq];             // four rows, one load
                    const uint sgv = sgw[sgb + (u >> 1) * mq];
                    const uint g0  = ((qsv      ) & 0xFFu) | (((qh0 >> u) & 1u) << 8);
                    const uint g1  = ((qsv >>  8) & 0xFFu) | (((qh1 >> u) & 1u) << 8);
                    const uint g2  = ((qsv >> 16) & 0xFFu) | (((qh2 >> u) & 1u) << 8);
                    const uint g3  = ((qsv >> 24) & 0xFFu) | (((qh3 >> u) & 1u) << 8);
                    const uint base = (u & 1u) * 4u;
                    const float4 yv = IQ3S_YV(grp + u, y);          // shared by all four
                    a0 += dot(yv, IQ3S_W(g0, IQ3S_GW(g0, IQ3S_GRID(g0)), (sgv      ) & 0xFFu, base));
                    a1 += dot(yv, IQ3S_W(g1, IQ3S_GW(g1, IQ3S_GRID(g1)), (sgv >>  8) & 0xFFu, base));
                    a2 += dot(yv, IQ3S_W(g2, IQ3S_GW(g2, IQ3S_GRID(g2)), (sgv >> 16) & 0xFFu, base));
                    a3 += dot(yv, IQ3S_W(g3, IQ3S_GW(g3, IQ3S_GRID(g3)), (sgv >> 24) & 0xFFu, base));
                }
                acc0 += (float)(1u + 2u * nib0) * a0;
                acc1 += (float)(1u + 2u * nib1) * a1;
                acc2 += (float)(1u + 2u * nib2) * a2;
                acc3 += (float)(1u + 2u * nib3) * a3;
            }
            sumf  += (float)dh.s0 * acc0;
            sumf1 += (float)dh.s1 * acc1;
            sumf2 += (float)dh.s2 * acc2;
            sumf3 += (float)dh.s3 * acc3;
        }
    }
#elif IQ3S_MV_R2
    const uint mh  = m >> 1;                        // rows per plane row, as ushorts
    const uint j   = get_group_id(0) * 64u + lid;   // row pair index
    const uint row = j << 1;

    float sumf  = 0.f;
    float sumf1 = 0.f;

    if (j < mh) {
        global const ushort * qsu = (global const ushort *)src0_qs;
        global const ushort * qhu = (global const ushort *)src0_qh;
        global const ushort * sgu = (global const ushort *)src0_sg;
        global const ushort * scu = (global const ushort *)src0_sc;

        for (uint ib = sgi; ib < nsb; ib += IQ3S_MV_NSG) {
            const half2 dh = vload2(j + ib * mh, src0_d);
            const float d0 = (float)dh.s0;
            const float d1 = (float)dh.s1;

            const uint scbase = j + ib * 4u * mh;
            ushort sc4[4];
            sc4[0] = scu[scbase + 0u * mh];
            sc4[1] = scu[scbase + 1u * mh];
            sc4[2] = scu[scbase + 2u * mh];
            sc4[3] = scu[scbase + 3u * mh];

            float acc0 = 0.f, acc1 = 0.f;
            for (uint sb = 0; sb < 8u; ++sb) {
                const uint scv  = (uint)sc4[sb >> 1];
                const uint nib0 = (sb & 1u) ? ((scv >>  4) & 0xFu) : ( scv        & 0xFu);
                const uint nib1 = (sb & 1u) ? ((scv >> 12) & 0xFu) : ((scv >>  8) & 0xFu);

                const uint qhv = (uint)qhu[j + (ib * 8u + sb) * mh];
                const uint qh0 = qhv & 0xFFu;
                const uint qh1 = qhv >> 8;

                const uint grp = ib * 64u + sb * 8u;
                const uint qsb = j + grp * mh;
                const uint sgb = j + (ib * 32u + sb * 4u) * mh;

                float a0 = 0.f, a1 = 0.f;
                for (uint u = 0; u < 8u; ++u) {
                    const uint qsv = (uint)qsu[qsb + u * mh];   // row pair, one load
                    const uint sgv = (uint)sgu[sgb + (u >> 1) * mh];
                    const uint g0  = ( qsv       & 0xFFu) | (((qh0 >> u) & 1u) << 8);
                    const uint g1  = ((qsv >> 8) & 0xFFu) | (((qh1 >> u) & 1u) << 8);
                    const uint base = (u & 1u) * 4u;
                    const float4 yv = IQ3S_YV(grp + u, y);
                    const uint gw0 = IQ3S_GW(g0, IQ3S_GRID(g0));
                    const uint gw1 = IQ3S_GW(g1, IQ3S_GRID(g1));
                    a0 += dot(yv, IQ3S_W(g0, gw0,  sgv       & 0xFFu, base));
                    a1 += dot(yv, IQ3S_W(g1, gw1, (sgv >> 8) & 0xFFu, base));
#if IQ3S_MV_WORK == 2
                    a0 += dot(yv, iq3s_vals(IQ3S_GRID(g0 ^ 1u),  (sgv + 1u) & 0xFFu, base));
                    a1 += dot(yv, iq3s_vals(IQ3S_GRID(g1 ^ 1u), ((sgv >> 8) + 1u) & 0xFFu, base));
#elif IQ3S_MV_WORK == 3
                    a0 += dot(yv, iq3s_vals(gw0,  (sgv + 1u) & 0xFFu, base));
                    a1 += dot(yv, iq3s_vals(gw1, ((sgv >> 8) + 1u) & 0xFFu, base));
#endif
                }
                acc0 += (float)(1u + 2u * nib0) * a0;
                acc1 += (float)(1u + 2u * nib1) * a1;
            }
            sumf  += d0 * acc0;
            sumf1 += d1 * acc1;
        }
    }
#else
    const uint row = get_group_id(0) * 64u + lid;

    float sumf = 0.f;

    if (row < m) {
        for (uint ib = sgi; ib < nsb; ib += IQ3S_MV_NSG) {
            const float d = (float)src0_d[row + ib * m];

            float acc = 0.f;
            for (uint sb = 0; sb < 8u; ++sb) {
                const uint scv = (uint)src0_sc[row + (ib * 4u + (sb >> 1)) * m];
                const uint nib = (sb & 1u) ? (scv >> 4) : (scv & 0xFu);
                const uint qhv = (uint)src0_qh[row + (ib * 8u + sb) * m];

                const uint grp = ib * 64u + sb * 8u;
                const uint qsb = row + grp * m;
                const uint sgb = row + (ib * 32u + sb * 4u) * m;

                float a = 0.f;
                for (uint u = 0; u < 8u; ++u) {
                    const uint g   = (uint)src0_qs[qsb + u * m] | (((qhv >> u) & 1u) << 8);
                    const uint sgv = (uint)src0_sg[sgb + (u >> 1) * m];
                    const float4 yv = IQ3S_YV(grp + u, y);
                    const uint gw = IQ3S_GW(g, IQ3S_GRID(g));
                    a += dot(yv, IQ3S_W(g, gw, sgv, (u & 1u) * 4u));
#if IQ3S_MV_WORK == 2
                    a += dot(yv, iq3s_vals(IQ3S_GRID(g ^ 1u), sgv + 1u, (u & 1u) * 4u));
#elif IQ3S_MV_WORK == 3
                    a += dot(yv, iq3s_vals(gw, sgv + 1u, (u & 1u) * 4u));
#endif
                }
                acc += (float)(1u + 2u * nib) * a;
            }
            sumf += d * acc;
        }
    }
#endif

#if IQ3S_MV_NSG > 1
#if IQ3S_MV_R4
    __local float4 part[IQ3S_MV_NSG][64];
    part[sgi][lid] = (float4)(sumf, sumf1, sumf2, sumf3);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sgi != 0) {
        return;
    }
    for (uint s = 1; s < IQ3S_MV_NSG; ++s) {
        const float4 p = part[s][lid];
        sumf  += p.s0;
        sumf1 += p.s1;
        sumf2 += p.s2;
        sumf3 += p.s3;
    }
#elif IQ3S_MV_R2
    __local float2 part[IQ3S_MV_NSG][64];
    part[sgi][lid] = (float2)(sumf, sumf1);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sgi != 0) {
        return;
    }
    for (uint s = 1; s < IQ3S_MV_NSG; ++s) {
        const float2 p = part[s][lid];
        sumf  += p.s0;
        sumf1 += p.s1;
    }
#else
    __local float part[IQ3S_MV_NSG][64];
    part[sgi][lid] = sumf;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sgi != 0) {
        return;
    }
    for (uint s = 1; s < IQ3S_MV_NSG; ++s) {
        sumf += part[s][lid];
    }
#endif
#endif

#if IQ3S_MV_R4
    if (j < mq) {
        vstore4((float4)(sumf, sumf1, sumf2, sumf3), 0, dst + (ulong)col * (uint)ne0 + row);
    }
#elif IQ3S_MV_R2
    if (j < mh) {
        vstore2((float2)(sumf, sumf1), 0, dst + (ulong)col * (uint)ne0 + row);
    }
#else
    if (row < m) {
        dst[(ulong)col * (uint)ne0 + row] = sumf;
    }
#endif
}


// ---------------------------------------------------------------------------
// Two COLUMNS per workgroup, for the ne11 = 2..31 band. The IQ3_S twin of
// kernel_mul_mv_iq1_s_f32_flat_mc; see that kernel for the full rationale.
//
// The plain GEMV takes its column from get_group_id(1), so at ne11 = N it reads
// the whole weight matrix N times and gathers the grid N times. Nothing else
// covers that band: split-K is gated ne11 == 1 and the prefill GEMM starts at
// ne11 >= 32.
//
// This type gathers twice per step, one entry per row of the pair, and applies a
// sign table on top. IQ1_S, which gathers twice with no signs, measured +8.5% at
// ne11 2 rising to +17.3% at 8; IQ2_S, which gathers four times, +26.7% to
// +50.2%. More per-step weight work to amortise means a larger win.
//
// The two grid entries are gathered once and held; their sign-applied float4s
// are rebuilt per column rather than kept, because iq3s_vals is a shift and an
// XOR and holding four float4s would spend the register budget that the row fold
// and the codebook prefetch both proved is the binding constraint on this family.
//
// An odd trailing column is computed against a duplicate of its partner and
// dropped at the store, so the inner loop has no divergent branch. That wastes
// one column of 2*ceil(n/2); the host declines ne11 == 3, the only width where
// that outweighs the saving.
// ---------------------------------------------------------------------------

#if IQ3S_MV_AIMG
#define IQ3S_MCYA(g) read_imagef(y_img, (int)(y_tex_a + (g)))
#define IQ3S_MCYB(g) read_imagef(y_img, (int)(y_tex_b + (g)))
#else
#define IQ3S_MCYA(g) vload4((g), ya)
#define IQ3S_MCYB(g) vload4((g), yb)
#endif

kernel void kernel_mul_mv_iq3_s_f32_flat_mc(
        __read_only image1d_buffer_t grid_img,
        __read_only image1d_buffer_t y_img,      // see IQ3S_MV_AIMG
        global const uchar * src0_qs,
        global const uchar * src0_qh,
        global const uchar * src0_sg,
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
        uint y_off,    // offset1/16, in float4 texels (IQ3S_MV_AIMG only)
        int ne11,
        __read_only image1d_buffer_t sgrid_img   // see IQ3S_MV_SGRID
) {
    src1 = (global const float *)((global const char *)src1 + offset1);
    dst  = (global float       *)((global char       *)dst  + offsetd);

    const uint m   = (uint)ne01;
    const uint K   = (uint)ne00;
    const uint nsb = K / QK_K;

    const uint lid = get_local_id(0);
    const uint sgi = get_local_id(1);

    const uint nc    = (uint)ne11;
    const uint ca    = get_group_id(1) * 2u;
    const uint has_b = (ca + 1u < nc) ? 1u : 0u;
    const uint cb    = ca + has_b;        // duplicate of ca when the pair is odd

    global const float * ya = src1 + (ulong)ca * (uint)ne10;
    global const float * yb = src1 + (ulong)cb * (uint)ne10;
#if IQ3S_MV_AIMG
    const uint y_tex_a = y_off + ca * ((uint)ne10 >> 2);
    const uint y_tex_b = y_off + cb * ((uint)ne10 >> 2);
#endif

#if IQ3S_MV_GRIDIMG
#define IQ3S_MCGRID(i) (read_imageui(grid_img, (int)(i)).x)
#else
#define IQ3S_MCGRID(i) iq3s_grid[(i)]
#endif

    const uint mh  = m >> 1;
    const uint j   = get_group_id(0) * 64u + lid;
    const uint row = j << 1;

    float sa0 = 0.f, sa1 = 0.f, sb0 = 0.f, sb1 = 0.f;   // [column][row]

    if (j < mh) {
        global const ushort * qsu = (global const ushort *)src0_qs;
        global const ushort * qhu = (global const ushort *)src0_qh;
        global const ushort * sgu = (global const ushort *)src0_sg;
        global const ushort * scu = (global const ushort *)src0_sc;

        for (uint ib = sgi; ib < nsb; ib += IQ3S_MV_NSG) {
            const half2 dh = vload2(j + ib * mh, src0_d);
            const float d0 = (float)dh.s0;
            const float d1 = (float)dh.s1;

            const uint scbase = j + ib * 4u * mh;
            ushort sc4[4];
            sc4[0] = scu[scbase + 0u * mh];
            sc4[1] = scu[scbase + 1u * mh];
            sc4[2] = scu[scbase + 2u * mh];
            sc4[3] = scu[scbase + 3u * mh];

            float aa0 = 0.f, aa1 = 0.f, ab0 = 0.f, ab1 = 0.f;
            for (uint sb = 0; sb < 8u; ++sb) {
                const uint scv  = (uint)sc4[sb >> 1];
                const uint nib0 = (sb & 1u) ? ((scv >>  4) & 0xFu) : ( scv        & 0xFu);
                const uint nib1 = (sb & 1u) ? ((scv >> 12) & 0xFu) : ((scv >>  8) & 0xFu);

                const uint qhv = (uint)qhu[j + (ib * 8u + sb) * mh];
                const uint qh0 = qhv & 0xFFu;
                const uint qh1 = qhv >> 8;

                const uint grp = ib * 64u + sb * 8u;
                const uint qsb = j + grp * mh;
                const uint sgb = j + (ib * 32u + sb * 4u) * mh;

                float pa0 = 0.f, pa1 = 0.f, pb0 = 0.f, pb1 = 0.f;
                for (uint u = 0; u < 8u; ++u) {
                    const uint qsv = (uint)qsu[qsb + u * mh];   // row pair, one load
                    const uint sgv = (uint)sgu[sgb + (u >> 1) * mh];
                    const uint g0  = ( qsv       & 0xFFu) | (((qh0 >> u) & 1u) << 8);
                    const uint g1  = ((qsv >> 8) & 0xFFu) | (((qh1 >> u) & 1u) << 8);
                    const uint base = (u & 1u) * 4u;
                    const uint s0   =  sgv       & 0xFFu;
                    const uint s1   = (sgv >> 8) & 0xFFu;

                    // Two gathers AND their sign application, shared by both
                    // columns. The siblings rebuild the float4 per column to save
                    // registers, because for them the gather is the cost. IQ3_S is
                    // the opposite: its own ablation prices the grid lookup at
                    // 0.2% and the SIGN APPLICATION at 14.2%, and it is the one
                    // type in the family that loses under doubled arithmetic at
                    // fixed loads. So here the value is what must be shared;
                    // rebuilding it per column measured -4 to -7%.
                    const float4 w0 = IQ3S_W(g0, IQ3S_MCGRID(g0), s0, base);
                    const float4 w1 = IQ3S_W(g1, IQ3S_MCGRID(g1), s1, base);

                    const float4 av = IQ3S_MCYA(grp + u);
                    pa0 += dot(av, w0);
                    pa1 += dot(av, w1);

                    const float4 bv = IQ3S_MCYB(grp + u);
                    pb0 += dot(bv, w0);
                    pb1 += dot(bv, w1);
                }
                aa0 += (float)(1u + 2u * nib0) * pa0;
                aa1 += (float)(1u + 2u * nib1) * pa1;
                ab0 += (float)(1u + 2u * nib0) * pb0;
                ab1 += (float)(1u + 2u * nib1) * pb1;
            }
            sa0 += d0 * aa0;
            sa1 += d1 * aa1;
            sb0 += d0 * ab0;
            sb1 += d1 * ab1;
        }
    }

#if IQ3S_MV_NSG > 1
    __local float4 mcpart[IQ3S_MV_NSG][64];
    mcpart[sgi][lid] = (float4)(sa0, sa1, sb0, sb1);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sgi != 0) {
        return;
    }
    for (uint s = 1; s < IQ3S_MV_NSG; ++s) {
        const float4 p = mcpart[s][lid];
        sa0 += p.s0; sa1 += p.s1; sb0 += p.s2; sb1 += p.s3;
    }
#endif

    if (j < mh) {
        vstore2((float2)(sa0, sa1), 0, dst + (ulong)ca * (uint)ne0 + row);
        if (has_b) {
            vstore2((float2)(sb0, sb1), 0, dst + (ulong)cb * (uint)ne0 + row);
        }
    }
#undef IQ3S_MCGRID
}


// ---------------------------------------------------------------------------
// dp4a (int8) twin of the GEMV above.
//
// Why this and not the other levers: an additive work probe
// (GGML_OPENCL_IQ3S_MV_WORK=3, which DOUBLES the arithmetic while holding every
// load fixed) costs this kernel 33% of its throughput, and it still costs 33%
// after the grid moved to an image -- so the arithmetic is a large, INDEPENDENT
// term here, not something hiding in the grid read's shadow. IQ1_S by contrast
// is +-0.0% under the same probe, which is why this port is IQ3_S-only.
// The ceiling is visible: q3_K, linear and at the IDENTICAL 3.4375 bits per
// weight, is 1.53x faster at matched shapes in the same run.
//
// What changes: four float converts + four conditional negates + a float4 dot
// become one packed int8x4 and one dot_acc_sat. The activation arrives already
// quantised to q8_1 by kernel_quant_a_q8_1, and the block geometry lines up
// exactly -- one sub-block of this kernel is 32 K-values, which is one q8_1
// block, so the per-block activation scale needs no interpolation.
//
// IQ3S_MV_DP4A_FASTPACK=1 replaces iq3s_pack's four branches with a packed
// two's-complement negate. Safe ONLY because iq3s_grid values are odd 1..15 and
// never 0: (b ^ 0xFF) + 1 would carry into the next byte iff b == 0.
// MEASURED RESULT: this loses, and it is the kernel that loses, not the
// pre-pass. At matched dispatch counts (5544 each) the float GEMV takes
// 1086 ms of GPU time and this one 1183 ms (+8.9%); kernel_quant_a_q8_1 adds
// only 15 ms on top, so amortising the activation quantisation across GEMVs
// that share an activation could not close the gap. End to end it is -6.6%
// on a 3B and -2.6% on a 27B, the difference being the fixed pre-pass cost
// spread over more rows.
//
// The reason is the operand, not the dot. dp4a pays when the int8 weight
// falls out of the stored bits with a shift and a mask, as it does for the
// linear quants. A codebook quant has to SYNTHESISE the operand -- table
// lookup, then a packed sign flip -- and that synthesis costs more than the
// four multiply-accumulates the dot removes. FASTPACK being worth +13.6%
// over the branchy pack is the same fact seen from the other side.
//
// Left in, default off, so the next person can re-measure rather than
// re-derive it.
#ifndef IQ3S_MV_DP4A_FASTPACK
#define IQ3S_MV_DP4A_FASTPACK 1
#endif

inline uint iq3s_pack_ref(uint gv, uint sg, uint base) {
    int v0 = (int)((gv >>  0) & 0xFF); if (sg & (1u << (base + 0))) { v0 = -v0; }
    int v1 = (int)((gv >>  8) & 0xFF); if (sg & (1u << (base + 1))) { v1 = -v1; }
    int v2 = (int)((gv >> 16) & 0xFF); if (sg & (1u << (base + 2))) { v2 = -v2; }
    int v3 = (int)((gv >> 24) & 0xFF); if (sg & (1u << (base + 3))) { v3 = -v3; }
    return ((uint)v0 & 0xFFu) | (((uint)v1 & 0xFFu) <<  8)
         | (((uint)v2 & 0xFFu) << 16) | (((uint)v3 & 0xFFu) << 24);
}

// Spread the four sign bits into four 0x00/0xFF bytes with two multiplies, then
// negate all four lanes at once. The multiply-spread cannot carry between bytes
// because each byte holds only 0 or 1 before the expand.
inline uint iq3s_pack_packed(uint gv, uint sg, uint base) {
    const uint s = (sg >> base) & 0xFu;
    const uint m = (((s * 0x00204081u) & 0x01010101u) * 0xFFu);
    return (gv ^ m) + (m & 0x01010101u);
}

#if IQ3S_MV_DP4A_FASTPACK
#define IQ3S_PACK(g, s, b) iq3s_pack_packed((g), (s), (b))
#else
#define IQ3S_PACK(g, s, b) iq3s_pack_ref((g), (s), (b))
#endif

// 🔴 GUARDED. This kernel calls dot_acc_sat_4x8packed_ss_int, and it lives in the
// SAME program as the plain GEMV, the fused GLU and the split-K kernel. On a device
// without cl_khr_integer_dot_product the call is an implicit declaration, which is
// an ERROR under -Werror, so the WHOLE PROGRAM fails to build and every kernel in
// this file is lost -- not just this one.
//
// Caught by the fleet gate on the Adreno 642L (dp4a false, E031.38):
//   kernel compile error (err=-11): implicit declaration of function
//   'dot_acc_sat_4x8packed_ss_int' is invalid in C99
// It was latent for the whole round because every other device in the fleet, and
// the development part, report the extension.
//
// The host already tolerates a null handle for this kernel, so guarding the SOURCE
// is the whole fix: the program builds everywhere and only the dp4a variant is
// absent where the extension is.
#ifdef cl_khr_integer_dot_product
kernel void kernel_mul_mv_iq3_s_f32_flat_dp4a(
        __read_only image1d_buffer_t grid_img,
        global const uchar * src0_qs,
        global const uchar * src0_qh,
        global const uchar * src0_sg,
        global const uchar * src0_sc,
        global const half  * src0_d,
        global const uint  * qa,      // q8_1 activation quants, four int8 per uint
        global const half  * da,      // per-32 activation scale
        global float * dst,
        ulong offsetd,
        int ne00,      // K
        int ne01,      // M
        int ne0        // dst row stride
) {
    dst = (global float *)((global char *)dst + offsetd);

    const uint m   = (uint)ne01;
    const uint K   = (uint)ne00;
    const uint nsb = K / QK_K;

    const uint lid = get_local_id(0);
    const uint sgi = get_local_id(1);
    const uint col = get_group_id(1);

    const uint qbase = col * (K >> 2);     // uints of qa per token
    const uint dbase = col * (K >> 5);     // q8_1 blocks per token

#if IQ3S_MV_GRIDIMG
#define IQ3S_GRID_D(i) (read_imageui(grid_img, (int)(i)).x)
#else
#define IQ3S_GRID_D(i) iq3s_grid[(i)]
#endif

    const uint mh  = m >> 1;
    const uint j   = get_group_id(0) * 64u + lid;
    const uint row = j << 1;

    float sumf  = 0.f;
    float sumf1 = 0.f;

    if (j < mh) {
        global const ushort * qsu = (global const ushort *)src0_qs;
        global const ushort * qhu = (global const ushort *)src0_qh;
        global const ushort * sgu = (global const ushort *)src0_sg;
        global const ushort * scu = (global const ushort *)src0_sc;

        for (uint ib = sgi; ib < nsb; ib += IQ3S_MV_NSG) {
            const half2 dh = vload2(j + ib * mh, src0_d);
            const float d0 = (float)dh.s0;
            const float d1 = (float)dh.s1;

            const uint scbase = j + ib * 4u * mh;
            ushort sc4[4];
            sc4[0] = scu[scbase + 0u * mh];
            sc4[1] = scu[scbase + 1u * mh];
            sc4[2] = scu[scbase + 2u * mh];
            sc4[3] = scu[scbase + 3u * mh];

            float acc0 = 0.f, acc1 = 0.f;
            for (uint sb = 0; sb < 8u; ++sb) {
                const uint scv  = (uint)sc4[sb >> 1];
                const uint nib0 = (sb & 1u) ? ((scv >>  4) & 0xFu) : ( scv        & 0xFu);
                const uint nib1 = (sb & 1u) ? ((scv >> 12) & 0xFu) : ((scv >>  8) & 0xFu);

                const uint qhv = (uint)qhu[j + (ib * 8u + sb) * mh];
                const uint qh0 = qhv & 0xFFu;
                const uint qh1 = qhv >> 8;

                const uint grp = ib * 64u + sb * 8u;
                const uint qsb = j + grp * mh;
                const uint sgb = j + (ib * 32u + sb * 4u) * mh;

                // one sub-block == 32 K-values == exactly one q8_1 block
                const float dy = (float)da[dbase + ib * 8u + sb];

                int ia0 = 0, ia1 = 0;
                for (uint u = 0; u < 8u; ++u) {
                    const uint qsv  = (uint)qsu[qsb + u * mh];
                    const uint sgv  = (uint)sgu[sgb + (u >> 1) * mh];
                    const uint g0   = ( qsv       & 0xFFu) | (((qh0 >> u) & 1u) << 8);
                    const uint g1   = ((qsv >> 8) & 0xFFu) | (((qh1 >> u) & 1u) << 8);
                    const uint base = (u & 1u) * 4u;
                    const uint ay   = qa[qbase + grp + u];
                    ia0 = dot_acc_sat_4x8packed_ss_int(IQ3S_PACK(IQ3S_GRID_D(g0),  sgv       & 0xFFu, base), ay, ia0);
                    ia1 = dot_acc_sat_4x8packed_ss_int(IQ3S_PACK(IQ3S_GRID_D(g1), (sgv >> 8) & 0xFFu, base), ay, ia1);
                }
                acc0 += (float)(1u + 2u * nib0) * dy * (float)ia0;
                acc1 += (float)(1u + 2u * nib1) * dy * (float)ia1;
            }
            sumf  += d0 * acc0;
            sumf1 += d1 * acc1;
        }
    }

#if IQ3S_MV_NSG > 1
    __local float2 partd[IQ3S_MV_NSG][64];
    partd[sgi][lid] = (float2)(sumf, sumf1);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sgi != 0) {
        return;
    }
    for (uint t = 1; t < IQ3S_MV_NSG; ++t) {
        const float2 p = partd[t][lid];
        sumf  += p.s0;
        sumf1 += p.s1;
    }
#endif

    if (j < mh) {
        vstore2((float2)(sumf, sumf1), 0, dst + (ulong)col * (uint)ne0 + row);
    }
}
#endif  // cl_khr_integer_dot_product

// ---------------------------------------------------------------------------
// Fused ffn_gate + ffn_up + GLU for IQ3_S, the fourth of this family.
//
// Same shape as the IQ2_S, IQ1_S and IQ2_XXS twins: the two FFN projections
// share an activation, so the streams are interleaved and each activation vector
// is read once. R2 only.
//
// MEASURED, and it LOSES. Qwen3.8-27B-UD-IQ3_S tg32 5.076 -> 4.829, -4.9%,
// fired-checked at 135 dispatches so the arms really differ, PPL 6.0921 either
// way. DEFAULT OFF. The kernel is kept so the result is re-measurable.
//
// The prediction written here beforehand was that IQ3_S would gain LESS than the
// lower-bit twins, because its cost probe says the arithmetic is a large
// independent term (doubling it costs 33% of throughput). That was the right
// direction and the wrong magnitude: it does not gain less, it goes negative.
//
// Where the family lands, all on the same construction:
//     IQ2_S    +13.2%   4 codebook lookups per 32-weight sub-block
//     IQ1_S    +12.4% / +10.3%
//     IQ2_XXS   +4.5%   4 lookups, but measured on a 27B
//     IQ3_S     -4.9%   EIGHT lookups per sub-block
//
// REGISTER PRESSURE WAS THE WRONG ANSWER. That hypothesis was written here, then
// tested with kinfo against the X2-90 compiler, and it is backwards:
//
//     kernel                base -> fused   private B/WI    wg cap    fusion
//     IQ2_S                   320 -> 472              472   768->512   +13.2%
//     IQ1_S                   288 -> 424              424   896->512   +12.4%
//     IQ2_XXS                 288 -> 456              456   896->512    +4.5%
//     IQ3_S                   256 -> 328              328  1024->768    -4.9%
//
// The kernel that REGRESSED has the LOWEST footprint of the four fused kernels and
// the HIGHEST workgroup cap, so the best occupancy. Nothing spills, and all four
// clear the 512 work-item dispatch.
//
// WHAT FITS INSTEAD: the codebook gather is this kernel's dominant term, and
// fusing DOUBLES it. Per 32 weights per row every one of these does 8 dots and 8
// activation loads, so the fusion saves the same activation traffic in all four --
// but IQ3_S does EIGHT codebook lookups where the others do four, and fusing takes
// it from 16 to 32 gathers per sub-block per row pair against 8 to 16 elsewhere.
//
// The cross-check is the grid-image measurement, which is a ready-made proxy for
// how gather-dominated each kernel is, and it anti-correlates almost perfectly:
//     GRIDIMG was worth   IQ1_S +2.5%   IQ2_S +4.4%   IQ3_S +10.4%   IQ3_XXS +18.8%
//     fusion is worth     IQ1_S +12.4%  IQ2_S +13.2%  IQ3_S  -4.9%   IQ3_XXS   ?
//
// PREDICTION, UNTESTED: IQ3_XXS should regress HARDER than IQ3_S, since the image
// was worth the most there. That is why the fused IQ3_XXS kernel was never built.
// ---------------------------------------------------------------------------

// Fifthth copy of the shared GLU epilogue: each .cl is its own program and cannot
// include the others. Op numbering and expressions match q40_glu_apply exactly --
// if one copy is ever changed, change them all.
#define IQ3S_GLU_GEGLU_COEF_A   0.044715f
#define IQ3S_GLU_SQRT_2_OVER_PI 0.79788456080286535587989211986876f
#define IQ3S_GLU_SQRT_2_INV     0.70710678118654752440084436210484f
#define IQ3S_GLU_QUICK_COEF    -1.702f
inline float iq3s_glu_apply(int glu_op, float g, float u) {
    float act;
    if (glu_op == 1) {        // GEGLU (tanh-approx gelu)
        act = 0.5f*g*(1.0f + tanh(IQ3S_GLU_SQRT_2_OVER_PI*g*(1.0f + IQ3S_GLU_GEGLU_COEF_A*g*g)));
    } else if (glu_op == 2) { // SWIGLU (silu)
        act = g / (1.0f + exp(-g));
    } else if (glu_op == 0) { // REGLU
        return g*u*(g > 0.0f);
    } else if (glu_op == 4) { // GEGLU_ERF
        act = 0.5f*g*(1.0f + erf(g*IQ3S_GLU_SQRT_2_INV));
    } else {                  // GEGLU_QUICK
        act = g*(1.0f/(1.0f + exp(IQ3S_GLU_QUICK_COEF*g)));
    }
    return act*u;
}

kernel void kernel_mul_mv_iq3_s_f32_flat_glu(
        __read_only image1d_buffer_t grid_img,
        global const uchar * g_qs,
        global const uchar * g_qh,
        global const uchar * g_sg,
        global const uchar * g_sc,
        global const half  * g_d,
        global const uchar * u_qs,
        global const uchar * u_qh,
        global const uchar * u_sg,
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
        __read_only image1d_buffer_t sgrid_img   // see IQ3S_MV_SGRID
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

#if IQ3S_MV_GRIDIMG
#define IQ3S_GGRID(i) (read_imageui(grid_img, (int)(i)).x)
#else
#define IQ3S_GGRID(i) iq3s_grid[(i)]
#endif

    const uint mh  = m >> 1;
    const uint j   = get_group_id(0) * 64u + lid;
    const uint row = j << 1;

    float gs0 = 0.f, gs1 = 0.f, us0 = 0.f, us1 = 0.f;

    if (j < mh) {
        global const ushort * gqsu = (global const ushort *)g_qs;
        global const ushort * gqhu = (global const ushort *)g_qh;
        global const ushort * gsgu = (global const ushort *)g_sg;
        global const ushort * gscu = (global const ushort *)g_sc;
        global const ushort * uqsu = (global const ushort *)u_qs;
        global const ushort * uqhu = (global const ushort *)u_qh;
        global const ushort * usgu = (global const ushort *)u_sg;
        global const ushort * uscu = (global const ushort *)u_sc;

        for (uint ib = sgi; ib < nsb; ib += IQ3S_MV_NSG) {
            const half2 gdh = vload2(j + ib * mh, g_d);
            const half2 udh = vload2(j + ib * mh, u_d);

            const uint scbase = j + ib * 4u * mh;
            ushort gsc4[4], usc4[4];
            for (uint t = 0; t < 4u; ++t) {
                gsc4[t] = gscu[scbase + t * mh];
                usc4[t] = uscu[scbase + t * mh];
            }

            float gacc0 = 0.f, gacc1 = 0.f, uacc0 = 0.f, uacc1 = 0.f;
            for (uint sb = 0; sb < 8u; ++sb) {
                const uint gscv  = (uint)gsc4[sb >> 1];
                const uint uscv  = (uint)usc4[sb >> 1];
                const uint gnib0 = (sb & 1u) ? ((gscv >>  4) & 0xFu) : ( gscv        & 0xFu);
                const uint gnib1 = (sb & 1u) ? ((gscv >> 12) & 0xFu) : ((gscv >>  8) & 0xFu);
                const uint unib0 = (sb & 1u) ? ((uscv >>  4) & 0xFu) : ( uscv        & 0xFu);
                const uint unib1 = (sb & 1u) ? ((uscv >> 12) & 0xFu) : ((uscv >>  8) & 0xFu);

                const uint gqhv = (uint)gqhu[j + (ib * 8u + sb) * mh];
                const uint uqhv = (uint)uqhu[j + (ib * 8u + sb) * mh];

                const uint grp = ib * 64u + sb * 8u;
                const uint qsb = j + grp * mh;
                const uint sgb = j + (ib * 32u + sb * 4u) * mh;

                float ga0 = 0.f, ga1 = 0.f, ua0 = 0.f, ua1 = 0.f;
                for (uint u = 0; u < 8u; ++u) {
                    const float4 yv   = vload4(grp + u, y);   // once, both streams
                    const uint   base = (u & 1u) * 4u;

                    const uint gqsv = (uint)gqsu[qsb + u * mh];
                    const uint gsgv = (uint)gsgu[sgb + (u >> 1) * mh];
                    const uint gg0  = ( gqsv       & 0xFFu) | ((((gqhv & 0xFFu) >> u) & 1u) << 8);
                    const uint gg1  = ((gqsv >> 8) & 0xFFu) | ((((gqhv >> 8)    >> u) & 1u) << 8);
                    ga0 += dot(yv, IQ3S_W(gg0, IQ3S_GGRID(gg0),  gsgv       & 0xFFu, base));
                    ga1 += dot(yv, IQ3S_W(gg1, IQ3S_GGRID(gg1), (gsgv >> 8) & 0xFFu, base));

                    const uint uqsv = (uint)uqsu[qsb + u * mh];
                    const uint usgv = (uint)usgu[sgb + (u >> 1) * mh];
                    const uint ug0  = ( uqsv       & 0xFFu) | ((((uqhv & 0xFFu) >> u) & 1u) << 8);
                    const uint ug1  = ((uqsv >> 8) & 0xFFu) | ((((uqhv >> 8)    >> u) & 1u) << 8);
                    ua0 += dot(yv, IQ3S_W(ug0, IQ3S_GGRID(ug0),  usgv       & 0xFFu, base));
                    ua1 += dot(yv, IQ3S_W(ug1, IQ3S_GGRID(ug1), (usgv >> 8) & 0xFFu, base));
                }
                gacc0 += (float)(1u + 2u * gnib0) * ga0;
                gacc1 += (float)(1u + 2u * gnib1) * ga1;
                uacc0 += (float)(1u + 2u * unib0) * ua0;
                uacc1 += (float)(1u + 2u * unib1) * ua1;
            }
            gs0 += (float)gdh.s0 * gacc0;
            gs1 += (float)gdh.s1 * gacc1;
            us0 += (float)udh.s0 * uacc0;
            us1 += (float)udh.s1 * uacc1;
        }
    }

#if IQ3S_MV_NSG > 1
    __local float4 gpart[IQ3S_MV_NSG][64];
    gpart[sgi][lid] = (float4)(gs0, gs1, us0, us1);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sgi != 0) {
        return;
    }
    for (uint s = 1; s < IQ3S_MV_NSG; ++s) {
        const float4 p = gpart[s][lid];
        gs0 += p.s0; gs1 += p.s1; us0 += p.s2; us1 += p.s3;
    }
#endif

    if (j < mh) {
        global float * o = dst + (ulong)col * (uint)ne0 + row;
        o[0] = iq3s_glu_apply(glu_op, gs0, us0);
        o[1] = iq3s_glu_apply(glu_op, gs1, us1);
    }
#undef IQ3S_GGRID
}


// ---------------------------------------------------------------------------
// Workgroup-level K split, the IQ3_S twin of the IQ2_S kernel.
//
// This is an OCCUPANCY change, not an arithmetic one, so the reason the GLU
// fusion regressed on this kernel does not apply: the fusion doubled the codebook
// gathers per workgroup, which is this kernel's dominant term, whereas splitting K
// leaves the gather count per unit of work unchanged and only spreads it over more
// workgroups. Measured anyway.
//
// On the profiled UD-IQ2_M frame IQ3_S serves attn_out (8.9%, 24 wg, 75%),
// Vcur (7.4%, 8 wg, 50%) and part of ffn_out (2.8%).
//
// MEASURED and it does NOT pay. DEFAULT OFF.
//     Llama-3.2-3B-UD-IQ2_M  tg64  31.172 -> 31.359  (+0.6%, noise)
//     Qwen3.8-27B-UD-IQ3_S   tg32   5.092 ->  5.015  (-1.5%)
// Fired-checked at 1020 dispatches, PPL 14.8367 either way.
//
// The prediction above -- that the mechanism which sank the GLU fusion on this
// kernel would not apply, because splitting K leaves gathers per unit of work
// alone -- was WRONG. Two predictions about this kernel, both wrong.
//
// AND THE MECHANISM IS STILL OPEN. The same split on the IQ2_S GEMV is +7.1% on
// the 3B and +1.1% / +0.9% on two 27Bs, so it is not simply "split-K is bad" nor
// "large models lose". The obvious stories do not survive: the partial and reduce
// traffic is ksplit*M for both types, and IQ3_S serves the WORST-filled dispatch
// in the frame (Vcur at 8 workgroups, 50%), which should favour it most. Do not
// invent a reason -- measure per type. The kernel stays so this is re-measurable.
// ---------------------------------------------------------------------------

kernel void kernel_mul_mv_iq3_s_f32_flat_splitk(
        __read_only image1d_buffer_t grid_img,
        global const uchar * src0_qs,
        global const uchar * src0_qh,
        global const uchar * src0_sg,
        global const uchar * src0_sc,
        global const half  * src0_d,
        global const float * src1,
        ulong offset1,
        global float * partial,
        int ne00,
        int ne01,
        int ne10,
        __read_only image1d_buffer_t sgrid_img   // see IQ3S_MV_SGRID
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

    const uint ib0 = (nsb * ks)        / nks;
    const uint ib1 = (nsb * (ks + 1u)) / nks;

    global const float * y = src1 + (ulong)col * (uint)ne10;

#if IQ3S_MV_GRIDIMG
#define IQ3S_SKGRID(i) (read_imageui(grid_img, (int)(i)).x)
#else
#define IQ3S_SKGRID(i) iq3s_grid[(i)]
#endif

    const uint mh  = m >> 1;
    const uint j   = get_group_id(0) * 64u + lid;
    const uint row = j << 1;

    float sumf = 0.f, sumf1 = 0.f;

    if (j < mh) {
        global const ushort * qsu = (global const ushort *)src0_qs;
        global const ushort * qhu = (global const ushort *)src0_qh;
        global const ushort * sgu = (global const ushort *)src0_sg;
        global const ushort * scu = (global const ushort *)src0_sc;

        for (uint ib = ib0 + sgi; ib < ib1; ib += IQ3S_MV_NSG) {
            const half2 dh = vload2(j + ib * mh, src0_d);
            const float d0 = (float)dh.s0;
            const float d1 = (float)dh.s1;

            const uint scbase = j + ib * 4u * mh;
            ushort sc4[4];
            sc4[0] = scu[scbase + 0u * mh];
            sc4[1] = scu[scbase + 1u * mh];
            sc4[2] = scu[scbase + 2u * mh];
            sc4[3] = scu[scbase + 3u * mh];

            float acc0 = 0.f, acc1 = 0.f;
            for (uint sb = 0; sb < 8u; ++sb) {
                const uint scv  = (uint)sc4[sb >> 1];
                const uint nib0 = (sb & 1u) ? ((scv >>  4) & 0xFu) : ( scv        & 0xFu);
                const uint nib1 = (sb & 1u) ? ((scv >> 12) & 0xFu) : ((scv >>  8) & 0xFu);

                const uint qhv = (uint)qhu[j + (ib * 8u + sb) * mh];
                const uint qh0 = qhv & 0xFFu;
                const uint qh1 = qhv >> 8;

                const uint grp = ib * 64u + sb * 8u;
                const uint qsb = j + grp * mh;
                const uint sgb = j + (ib * 32u + sb * 4u) * mh;

                float a0 = 0.f, a1 = 0.f;
                for (uint u = 0; u < 8u; ++u) {
                    const uint qsv = (uint)qsu[qsb + u * mh];
                    const uint sgv = (uint)sgu[sgb + (u >> 1) * mh];
                    const uint g0  = ( qsv       & 0xFFu) | (((qh0 >> u) & 1u) << 8);
                    const uint g1  = ((qsv >> 8) & 0xFFu) | (((qh1 >> u) & 1u) << 8);
                    const uint base = (u & 1u) * 4u;
                    const float4 yv = vload4(grp + u, y);
                    a0 += dot(yv, IQ3S_W(g0, IQ3S_SKGRID(g0),  sgv       & 0xFFu, base));
                    a1 += dot(yv, IQ3S_W(g1, IQ3S_SKGRID(g1), (sgv >> 8) & 0xFFu, base));
                }
                acc0 += (float)(1u + 2u * nib0) * a0;
                acc1 += (float)(1u + 2u * nib1) * a1;
            }
            sumf  += d0 * acc0;
            sumf1 += d1 * acc1;
        }
    }

#if IQ3S_MV_NSG > 1
    __local float2 skpart[IQ3S_MV_NSG][64];
    skpart[sgi][lid] = (float2)(sumf, sumf1);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sgi != 0) {
        return;
    }
    for (uint s = 1; s < IQ3S_MV_NSG; ++s) {
        const float2 p = skpart[s][lid];
        sumf  += p.s0;
        sumf1 += p.s1;
    }
#endif

    if (j < mh) {
        global float * o = partial + (ulong)ks * m + row;
        o[0] = sumf;
        o[1] = sumf1;
    }
#undef IQ3S_SKGRID
}

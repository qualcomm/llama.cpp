#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

#define LM_FIRST_256B   0
#define LM_SECOND_256B  64
#define LM_THIRD_256B   128
#define LM_FOURTH_256B  192


inline float16 mm_load_a(image1d_buffer_t matrix_A, uint subMatrixAStartInElements, int nb01, int line_stride_matrix_A_in_bytes)
{
    __private float8 regA;
    size_t sub_block_id_m = get_local_id(0);

#ifdef KQV
   uint a_texCoord = subMatrixAStartInElements/2 + (sub_block_id_m * nb01/4);
#else // KQ
   uint a_texCoord = subMatrixAStartInElements/2 + (sub_block_id_m * line_stride_matrix_A_in_bytes/4);
#endif

   regA.s0123  = read_imagef(matrix_A, a_texCoord/4);
   regA.s4567  = read_imagef(matrix_A, (a_texCoord+4)/4);

    return convert_float16(as_half16(regA));
}

inline void alu_32(float* regC, float16 regA, __local float* matrix_B_local, int wave_offset) {

   __private float4 rC = 0;
   int i = wave_offset;

   rC += regA.s0  * ((__local float4*)matrix_B_local)[i];
   rC += regA.s1  * ((__local float4*)matrix_B_local)[i + 16];
   rC += regA.s4  * ((__local float4*)matrix_B_local)[i + 1];
   rC += regA.s5  * ((__local float4*)matrix_B_local)[i + 17];
   rC += regA.s8  * ((__local float4*)matrix_B_local)[i + 2];
   rC += regA.s9  * ((__local float4*)matrix_B_local)[i + 18];
   rC += regA.sc  * ((__local float4*)matrix_B_local)[i + 3];
   rC += regA.sd  * ((__local float4*)matrix_B_local)[i + 19];

   i += 32;

   rC += regA.s2  * ((__local float4*)matrix_B_local)[i];
   rC += regA.s3  * ((__local float4*)matrix_B_local)[i + 16];
   rC += regA.s6  * ((__local float4*)matrix_B_local)[i + 1];
   rC += regA.s7  * ((__local float4*)matrix_B_local)[i + 17];
   rC += regA.sa  * ((__local float4*)matrix_B_local)[i + 2];
   rC += regA.sb  * ((__local float4*)matrix_B_local)[i + 18];
   rC += regA.se  * ((__local float4*)matrix_B_local)[i + 3];
   rC += regA.sf  * ((__local float4*)matrix_B_local)[i + 19];
   
   float4* regC_vec = (float4*)regC;
   *regC_vec += rC;
}

inline void mm_mad(__local float* matrix_B_local, float* regC1_ptr, float* regC2_ptr, float* regC3_ptr, float* regC4_ptr, float* regC5_ptr, float* regC6_ptr, float* regC7_ptr, float* regC8_ptr, float16 regA, float8 regB, uint b_localOffsetInWords)
{

    short  linearIndex = get_local_id(0);

    int wave_offset = get_sub_group_id() * 64;
    int offset = b_localOffsetInWords + get_sub_group_id() * 256;
    
    matrix_B_local[offset + LM_FIRST_256B] = regB.s0;
    matrix_B_local[offset + LM_SECOND_256B] = regB.s1;

    matrix_B_local[offset + LM_THIRD_256B] = regB.s2;
    matrix_B_local[offset + LM_FOURTH_256B] = regB.s3;

    alu_32(regC1_ptr, regA, matrix_B_local + 0, wave_offset);

    alu_32(regC2_ptr, regA, matrix_B_local + 16, wave_offset);
   
    alu_32(regC3_ptr, regA, matrix_B_local + 32, wave_offset);
  
    alu_32(regC4_ptr, regA, matrix_B_local + 48, wave_offset);

       
    matrix_B_local[offset + LM_FIRST_256B] = regB.s4;
    matrix_B_local[offset + LM_SECOND_256B] = regB.s5;

    matrix_B_local[offset + LM_THIRD_256B] = regB.s6;
    matrix_B_local[offset + LM_FOURTH_256B] = regB.s7;

    alu_32(regC5_ptr, regA, matrix_B_local + 0, wave_offset);

    alu_32(regC6_ptr, regA, matrix_B_local + 16, wave_offset);
   
    alu_32(regC7_ptr, regA, matrix_B_local + 32, wave_offset);
  
    alu_32(regC8_ptr, regA, matrix_B_local + 48, wave_offset);

}

inline void mm_store_c_N(__write_only image1d_buffer_t matrix_C, float4 regC_1, float4 regC_2, float4 regC_3, float4 regC_4,
float4  regC_5, float4 regC_6, float4 regC_7, float4 regC_8, uint subMatrixCStartInElements, int line_stride_matrix_C_in_bytes, int mask)
{
    size_t sub_block_id_m = get_local_id(0);
    short  linearIndex = get_local_id(0);

    uint strideInWords     = line_stride_matrix_C_in_bytes/4;
    uint c_coordInWords_0  = (subMatrixCStartInElements + sub_block_id_m);

    uint c_coordInWords_1  = c_coordInWords_0 + 1  * strideInWords;
    uint c_coordInWords_2  = c_coordInWords_0 + 2  * strideInWords;
    uint c_coordInWords_3  = c_coordInWords_0 + 3  * strideInWords;
    uint c_coordInWords_4  = c_coordInWords_0 + 4  * strideInWords;
    uint c_coordInWords_5  = c_coordInWords_0 + 5  * strideInWords;
    uint c_coordInWords_6  = c_coordInWords_0 + 6  * strideInWords;
    uint c_coordInWords_7  = c_coordInWords_0 + 7  * strideInWords;
    uint c_coordInWords_8  = c_coordInWords_0 + 8  * strideInWords;
    uint c_coordInWords_9  = c_coordInWords_0 + 9  * strideInWords;
    uint c_coordInWords_10 = c_coordInWords_0 + 10 * strideInWords;
    uint c_coordInWords_11 = c_coordInWords_0 + 11 * strideInWords;
    uint c_coordInWords_12 = c_coordInWords_0 + 12 * strideInWords;
    uint c_coordInWords_13 = c_coordInWords_0 + 13 * strideInWords;
    uint c_coordInWords_14 = c_coordInWords_0 + 14 * strideInWords;
    uint c_coordInWords_15 = c_coordInWords_0 + 15 * strideInWords;
    uint c_coordInWords_16 = c_coordInWords_0 + 16 * strideInWords;
    uint c_coordInWords_17 = c_coordInWords_0 + 17 * strideInWords;
    uint c_coordInWords_18 = c_coordInWords_0 + 18 * strideInWords;
    uint c_coordInWords_19 = c_coordInWords_0 + 19 * strideInWords;
    uint c_coordInWords_20 = c_coordInWords_0 + 20 * strideInWords;
    uint c_coordInWords_21 = c_coordInWords_0 + 21 * strideInWords;
    uint c_coordInWords_22 = c_coordInWords_0 + 22 * strideInWords;
    uint c_coordInWords_23 = c_coordInWords_0 + 23 * strideInWords;
    uint c_coordInWords_24 = c_coordInWords_0 + 24 * strideInWords;
    uint c_coordInWords_25 = c_coordInWords_0 + 25 * strideInWords;
    uint c_coordInWords_26 = c_coordInWords_0 + 26 * strideInWords;
    uint c_coordInWords_27 = c_coordInWords_0 + 27 * strideInWords;
    uint c_coordInWords_28 = c_coordInWords_0 + 28 * strideInWords;
    uint c_coordInWords_29 = c_coordInWords_0 + 29 * strideInWords;
    uint c_coordInWords_30 = c_coordInWords_0 + 30 * strideInWords;
    uint c_coordInWords_31 = c_coordInWords_0 + 31 * strideInWords;
    
    if (mask > 0)  { write_imagef(matrix_C, c_coordInWords_0, regC_1.s0);  }
    if (mask > 1)  { write_imagef(matrix_C, c_coordInWords_1, regC_1.s1);  }
    if (mask > 2)  { write_imagef(matrix_C, c_coordInWords_2, regC_1.s2);  }
    if (mask > 3)  { write_imagef(matrix_C, c_coordInWords_3, regC_1.s3);  }
    if (mask > 4)  { write_imagef(matrix_C, c_coordInWords_4, regC_2.s0);  }
    if (mask > 5)  { write_imagef(matrix_C, c_coordInWords_5, regC_2.s1);  }
    if (mask > 6)  { write_imagef(matrix_C, c_coordInWords_6, regC_2.s2);  }
    if (mask > 7)  { write_imagef(matrix_C, c_coordInWords_7, regC_2.s3);  }
    if (mask > 8)  { write_imagef(matrix_C, c_coordInWords_8, regC_3.s0);  }
    if (mask > 9)  { write_imagef(matrix_C, c_coordInWords_9, regC_3.s1);  }
    if (mask > 10) { write_imagef(matrix_C, c_coordInWords_10, regC_3.s2); }
    if (mask > 11) { write_imagef(matrix_C, c_coordInWords_11, regC_3.s3); }
    if (mask > 12) { write_imagef(matrix_C, c_coordInWords_12, regC_4.s0); }
    if (mask > 13) { write_imagef(matrix_C, c_coordInWords_13, regC_4.s1); }
    if (mask > 14) { write_imagef(matrix_C, c_coordInWords_14, regC_4.s2); }
    if (mask > 15) { write_imagef(matrix_C, c_coordInWords_15, regC_4.s3); }
    if (mask > 16) { write_imagef(matrix_C, c_coordInWords_16, regC_5.s0); }
    if (mask > 17) { write_imagef(matrix_C, c_coordInWords_17, regC_5.s1); }
    if (mask > 18) { write_imagef(matrix_C, c_coordInWords_18, regC_5.s2); }
    if (mask > 19) { write_imagef(matrix_C, c_coordInWords_19, regC_5.s3); }
    if (mask > 20) { write_imagef(matrix_C, c_coordInWords_20, regC_6.s0); }
    if (mask > 21) { write_imagef(matrix_C, c_coordInWords_21, regC_6.s1); }
    if (mask > 22) { write_imagef(matrix_C, c_coordInWords_22, regC_6.s2); }
    if (mask > 23) { write_imagef(matrix_C, c_coordInWords_23, regC_6.s3); }
    if (mask > 24) { write_imagef(matrix_C, c_coordInWords_24, regC_7.s0); }
    if (mask > 25) { write_imagef(matrix_C, c_coordInWords_25, regC_7.s1); }
    if (mask > 26) { write_imagef(matrix_C, c_coordInWords_26, regC_7.s2); }
    if (mask > 27) { write_imagef(matrix_C, c_coordInWords_27, regC_7.s3); }
    if (mask > 28) { write_imagef(matrix_C, c_coordInWords_28, regC_8.s0); }
    if (mask > 29) { write_imagef(matrix_C, c_coordInWords_29, regC_8.s1); }
    if (mask > 30) { write_imagef(matrix_C, c_coordInWords_30, regC_8.s2); }
    if (mask > 31) { write_imagef(matrix_C, c_coordInWords_31, regC_8.s3); }

}

#define TILESIZE_K 16
#define TILESIZE_M 64
#define TILESIZE_N 32
#ifdef KQV
__kernel void mul_mm_f16_f32_kqv(
#else
__kernel void mul_mm_f16_f32_kq(
#endif
        __read_only  image1d_buffer_t matrix_A,
        int offset0,
        __global float* matrix_B,
        int offset1,
        __write_only image1d_buffer_t matrix_C,
        int offsetd,
        int M, int K, int N,
        int D_A,
        int D_B,
        int nb01
)
{

   uint   block_id_m   = get_global_id(1);                              
   uint   block_id_n   = get_global_id(2) % ((N+TILESIZE_N-1)/TILESIZE_N);              
   uint   block_id_d   = get_global_id(2) / ((N+TILESIZE_N-1)/TILESIZE_N);
   
   __private float16  regA;
   __private float8   regB;
   __private float4 regC_1;
   __private float4 regC_2;
   __private float4 regC_3;
   __private float4 regC_4;
   __private float4 regC_5;
   __private float4 regC_6;
   __private float4 regC_7;
   __private float4 regC_8;

   const uint col   = block_id_m * TILESIZE_M;
   const uint row   = block_id_n * TILESIZE_N;
   const uint depth_A = block_id_d / (D_B/D_A);
   const uint depth_B = block_id_d;

#ifdef KQV
   int line_stride_matrix_A_in_bytes = nb01 * M;
   int line_stride_matrix_B_in_bytes = K * N * 4;
#else
   int line_stride_matrix_A_in_bytes = K * D_A * 2;
   int line_stride_matrix_B_in_bytes = K * D_B * 4;
#endif

   int line_stride_matrix_C_in_bytes = M * 4;

   const uint strideAinElements = line_stride_matrix_A_in_bytes / 2;
   const uint strideBinElements = line_stride_matrix_B_in_bytes / 4;

   size_t sub_block_id_m = get_local_id(0); 

   uint b_localOffsetInWords = (sub_block_id_m/16)*16
                           + ((((sub_block_id_m)>>0)&1)<<2)
                           + ((((sub_block_id_m)>>1)&1)<<3)
                           + ((((sub_block_id_m)>>2)&1)<<0)
                           + ((((sub_block_id_m)>>3)&1)<<1);

   uint2 b_globalOffsetInWords_xy = {((sub_block_id_m%4)*4), (sub_block_id_m>>2)}; 
   uint b_globalOffsetInWords00, b_globalOffsetInWords16;
   #ifdef KQV    
      b_globalOffsetInWords00 = b_globalOffsetInWords_xy.x + b_globalOffsetInWords_xy.y*K;
      b_globalOffsetInWords16 = b_globalOffsetInWords00 + (16 * K); 
      uint subMatrixAStartInElements = depth_A * strideAinElements + col * nb01 / 2;
      uint subMatrixBStartInElements = depth_B * strideBinElements + row * K;
   #else
      b_globalOffsetInWords00 = b_globalOffsetInWords_xy.x + b_globalOffsetInWords_xy.y*line_stride_matrix_B_in_bytes/4;
      b_globalOffsetInWords16 = b_globalOffsetInWords00 + (16 * line_stride_matrix_B_in_bytes/4); 
      uint subMatrixAStartInElements = col * strideAinElements + depth_A * K;
      uint subMatrixBStartInElements = row * strideBinElements + depth_B * K;
   #endif

   __local float matrix_B_local[1024];

    for (uint step=0; step < K; step+=TILESIZE_K)
    {
      size_t sub_block_id_m = get_local_id(0);
      regA = mm_load_a(matrix_A, subMatrixAStartInElements, nb01, line_stride_matrix_A_in_bytes);
      

      uint b_coordInWords00 = subMatrixBStartInElements + b_globalOffsetInWords00;
      uint b_coordInWords16 = subMatrixBStartInElements + b_globalOffsetInWords16;

      regB.s0123 = vload4(b_coordInWords00/4, matrix_B);
      regB.s4567 = vload4(b_coordInWords16/4, matrix_B);

      mm_mad(matrix_B_local, (float *)&regC_1, (float *)&regC_2, (float *)&regC_3, (float *)&regC_4, (float *)&regC_5, (float *)&regC_6, (float *)&regC_7, (float *)&regC_8, regA, regB, b_localOffsetInWords);

      subMatrixAStartInElements += TILESIZE_K;
      subMatrixBStartInElements += TILESIZE_K;

    }

   uint subMatrixCStartInElements = depth_B * N * M + row * M + col;
   mm_store_c_N(matrix_C, regC_1, regC_2, regC_3, regC_4,regC_5, regC_6, regC_7,regC_8, subMatrixCStartInElements, line_stride_matrix_C_in_bytes, (N-block_id_n*32));
   
}

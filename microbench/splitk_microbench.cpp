// Prototype + microbench for split-K-ACROSS-WORKGROUPS on the q4_K decode GEMV.
// The main microbench showed M=512 matmuls make only ~4 workgroups (4/16 CUs
// used) and cap at ~29 GB/s. This splits the K reduction across n_ksplit
// workgroups (grid-y), each writing a partial, then a reduce kernel sums them.
// It appends two kernels to the REAL q4_k .cl (reusing its dequant macros) so
// the prototype matches production exactly, then times partial+reduce combined.
//
// Build: see build_splitk.bat.  Usage: splitk_microbench.exe [iters]

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>

#define CL_CHECK(expr) do { cl_int _e=(expr); if(_e!=CL_SUCCESS){fprintf(stderr,"CL err %d at %d (%s)\n",_e,__LINE__,#expr);exit(1);} } while(0)

static std::string read_file(const char* p){ std::ifstream f(p,std::ios::binary); if(!f){fprintf(stderr,"open %s\n",p);exit(1);} std::stringstream s; s<<f.rdbuf(); return s.str(); }

static cl_program build(cl_context c, cl_device_id d, const std::string& src, const std::string& o){
    const char* s=src.c_str(); cl_int e; cl_program p=clCreateProgramWithSource(c,1,&s,0,&e); CL_CHECK(e);
    e=clBuildProgram(p,1,&d,o.c_str(),0,0);
    if(e!=CL_SUCCESS){ size_t n=0; clGetProgramBuildInfo(p,d,CL_PROGRAM_BUILD_LOG,0,0,&n); std::vector<char> l(n+1,0);
        clGetProgramBuildInfo(p,d,CL_PROGRAM_BUILD_LOG,n,l.data(),0); fprintf(stderr,"BUILD FAIL:\n%s\n",l.data()); exit(1);}
    return p;
}
static double median(std::vector<double> v){ std::sort(v.begin(),v.end()); return v[v.size()/2]; }
static cl_mem mkbuf(cl_context c,size_t b){ cl_int e; cl_mem m=clCreateBuffer(c,CL_MEM_READ_WRITE,b,0,&e); CL_CHECK(e); return m; }
static cl_mem mkimg(cl_context c,cl_mem bk,cl_channel_order o,cl_channel_type t,size_t w){
    cl_image_format f{o,t}; cl_image_desc d; memset(&d,0,sizeof(d)); d.image_type=CL_MEM_OBJECT_IMAGE1D_BUFFER; d.image_width=w; d.buffer=bk;
    cl_int e; cl_mem m=clCreateImage(c,CL_MEM_READ_ONLY,&f,&d,0,&e); CL_CHECK(e); return m; }

// split-K q4_K GEMV + reduce, appended to the real q4_k source (reuses macros).
static const char* SPLITK_SRC = R"CLC(
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_q4k_splitk(
        read_only image1d_buffer_t src0_q,
        global half2 * src0_d, global half2 * src0_m, global uchar * src0_s,
        read_only image1d_buffer_t src1,
        global float * partial,        // [n_ksplit * M]
        int ne00, int ne01,
        uchar mask_d6, uchar mask_d4, uchar mask_hi2)
{
    uint groupId = get_local_id(1);
    uint gid     = get_global_id(0);
    ushort slid  = get_sub_group_local_id();
    uint nsg     = get_local_size(1);
    uint ksplit  = get_num_groups(1);
    uint kslice  = get_group_id(1);

    uint K = ne00, M = ne01;
    uint LINE_STRIDE_A  = M / 2;
    uint BLOCK_STRIDE_A = 4 * M;
    uint scales_per_row = (K / QK_K) * 12;

    private uint4 regA; private half2 regS, regM; private float8 regB;
    private float2 totalSum = (float2)(0.0f);

    for (uint k = kslice*nsg + groupId; k < (K/32); k += ksplit*nsg) {
        uint sb = k / 8, j = k % 8;
        half2 d  = src0_d[gid + sb * LINE_STRIDE_A];
        half2 dm = src0_m[gid + sb * LINE_STRIDE_A];
        global const uchar * sc0 = src0_s + 2*gid*scales_per_row + sb*12;
        global const uchar * sc1 = src0_s + (2*gid+1)*scales_per_row + sb*12;
        uchar sv0,mn0,sv1,mn1;
        get_scale_min_k4(j, sc0, &sv0,&mn0, mask_d6,mask_d4,mask_hi2);
        get_scale_min_k4(j, sc1, &sv1,&mn1, mask_d6,mask_d4,mask_hi2);
        regS = convert_half2(convert_float2(d)  * convert_float2((uchar2)(sv0,sv1)));
        regM = convert_half2(convert_float2(dm) * convert_float2((uchar2)(mn0,mn1)));
        if (slid < 4) {
            regB.s0123 = read_imagef(src1, (slid*2 + k*8));
            regB.s4567 = read_imagef(src1, (1 + slid*2 + k*8));
        }
        regA.s0 = read_imageui(src0_q, (gid + k*BLOCK_STRIDE_A + LINE_STRIDE_A*0)).x;
        regA.s1 = read_imageui(src0_q, (gid + k*BLOCK_STRIDE_A + LINE_STRIDE_A*1)).x;
        regA.s2 = read_imageui(src0_q, (gid + k*BLOCK_STRIDE_A + LINE_STRIDE_A*2)).x;
        regA.s3 = read_imageui(src0_q, (gid + k*BLOCK_STRIDE_A + LINE_STRIDE_A*3)).x;
#ifdef VECTOR_SUB_GROUP_BROADCAST
        dequantizeBlockAccum_ns_sgbroadcast_8_hi(totalSum, as_ushort8(regA), regS, regM, regB);
#else
        dequantizeBlockAccum_ns_sgbroadcast_1_hi(totalSum, as_ushort8(regA), regS, regM, regB);
#endif
        regA.s0 = read_imageui(src0_q, (gid + k*BLOCK_STRIDE_A + LINE_STRIDE_A*4)).x;
        regA.s1 = read_imageui(src0_q, (gid + k*BLOCK_STRIDE_A + LINE_STRIDE_A*5)).x;
        regA.s2 = read_imageui(src0_q, (gid + k*BLOCK_STRIDE_A + LINE_STRIDE_A*6)).x;
        regA.s3 = read_imageui(src0_q, (gid + k*BLOCK_STRIDE_A + LINE_STRIDE_A*7)).x;
#ifdef VECTOR_SUB_GROUP_BROADCAST
        dequantizeBlockAccum_ns_sgbroadcast_8_lo(totalSum, as_ushort8(regA), regS, regM, regB);
#else
        dequantizeBlockAccum_ns_sgbroadcast_1_lo(totalSum, as_ushort8(regA), regS, regM, regB);
#endif
    }

    local float2 reduceLM[SUBGROUP_SIZE * 15];
    if (groupId > 0) reduceLM[SUBGROUP_SIZE*(groupId-1) + slid] = totalSum;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (groupId == 0) {
        for (uint i = 0; i < nsg-1; ++i) totalSum += reduceLM[SUBGROUP_SIZE*i + slid];
        partial[kslice*M + gid*2]     = totalSum.s0;
        partial[kslice*M + gid*2 + 1] = totalSum.s1;
    }
}

kernel void kernel_splitk_reduce(global float* partial, global float* dst, int M, int ksplit) {
    uint r = get_global_id(0);
    if (r >= (uint)M) return;
    float acc = 0.0f;
    for (uint s = 0; s < (uint)ksplit; ++s) acc += partial[s*M + r];
    dst[r] = acc;
}
)CLC";

int main(int argc, char** argv) {
    const char* kdir = "d:/work/llm/llama.cpp/ggml/src/ggml-opencl/kernels/";
    int iters = (argc>1)?atoi(argv[1]):200;

    cl_uint np=0; clGetPlatformIDs(0,0,&np); std::vector<cl_platform_id> pl(np); clGetPlatformIDs(np,pl.data(),0);
    cl_device_id dev=0;
    for(auto p:pl){ cl_uint n=0; if(clGetDeviceIDs(p,CL_DEVICE_TYPE_GPU,0,0,&n)!=CL_SUCCESS||!n)continue; std::vector<cl_device_id> ds(n); clGetDeviceIDs(p,CL_DEVICE_TYPE_GPU,n,ds.data(),0); dev=ds[0]; break; }
    if(!dev){fprintf(stderr,"no gpu\n");return 1;}
    char nm[256]={0}; clGetDeviceInfo(dev,CL_DEVICE_NAME,sizeof(nm),nm,0);
    cl_uint cu=0; clGetDeviceInfo(dev,CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(cu),&cu,0);
    printf("device: %s  (max_compute_units=%u)\n", nm, cu);

    cl_int e; cl_context ctx=clCreateContext(0,1,&dev,0,0,&e); CL_CHECK(e);
    cl_command_queue q=clCreateCommandQueue(ctx,dev,CL_QUEUE_PROFILING_ENABLE,&e); CL_CHECK(e);

    std::string src = read_file((std::string(kdir)+"gemv_noshuffle_q4_k_f32.cl").c_str());
    src += SPLITK_SRC;
    cl_program prog = build(ctx, dev, src, "-cl-std=CL3.0 -cl-mad-enable -DVECTOR_SUB_GROUP_BROADCAST ");
    cl_kernel ksk, krd;
    CL_CHECK((ksk = clCreateKernel(prog,"kernel_q4k_splitk",&e),e));
    CL_CHECK((krd = clCreateKernel(prog,"kernel_splitk_reduce",&e),e));

    int shapes[][2] = { {2560,512}, {2560,1024}, {2560,2048} };
    cl_uchar md6=0x3F, md4=0x0F, mhi2=0xC0;

    for (auto& sh : shapes) {
        int K=sh[0], M=sh[1];
        double wbytes = 0.5*(double)K*M;
        cl_mem qb=mkbuf(ctx,(size_t)M*K/2+4096), db=mkbuf(ctx,(size_t)(K/256+1)*(M/2)*4+65536),
               dmb=mkbuf(ctx,(size_t)(K/256+1)*(M/2)*4+65536), sbf=mkbuf(ctx,(size_t)M*(K/256+1)*12+65536),
               bb=mkbuf(ctx,(size_t)K*4+4096), dst=mkbuf(ctx,(size_t)M*4+4096),
               part=mkbuf(ctx,(size_t)16*M*4+4096);
        cl_mem qi=mkimg(ctx,qb,CL_R,CL_UNSIGNED_INT32,(size_t)M*K/2/4), bbi=mkimg(ctx,bb,CL_RGBA,CL_FLOAT,(size_t)K/4);

        printf("\n--- q4_K K=%d M=%d  (%.2f MB/call)  [single-WG nsg=16 baseline + split-K] ---\n", K,M, wbytes/1e6);
        printf("  config                | total ms  GB/s\n");

        // configs: (nsg, ksplit). ksplit=1 nsg=16 is the current shipped wide-split.
        int cfgs[][2] = { {16,1}, {4,2}, {4,4}, {8,4}, {4,8}, {8,8}, {4,16} };
        for (auto& cf : cfgs) {
            int nsg=cf[0], ksplit=cf[1];
            CL_CHECK(clSetKernelArg(ksk,0,sizeof(cl_mem),&qi));
            CL_CHECK(clSetKernelArg(ksk,1,sizeof(cl_mem),&db));
            CL_CHECK(clSetKernelArg(ksk,2,sizeof(cl_mem),&dmb));
            CL_CHECK(clSetKernelArg(ksk,3,sizeof(cl_mem),&sbf));
            CL_CHECK(clSetKernelArg(ksk,4,sizeof(cl_mem),&bbi));
            CL_CHECK(clSetKernelArg(ksk,5,sizeof(cl_mem),&part));
            CL_CHECK(clSetKernelArg(ksk,6,sizeof(cl_int),&K));
            CL_CHECK(clSetKernelArg(ksk,7,sizeof(cl_int),&M));
            CL_CHECK(clSetKernelArg(ksk,8,sizeof(cl_uchar),&md6));
            CL_CHECK(clSetKernelArg(ksk,9,sizeof(cl_uchar),&md4));
            CL_CHECK(clSetKernelArg(ksk,10,sizeof(cl_uchar),&mhi2));
            CL_CHECK(clSetKernelArg(krd,0,sizeof(cl_mem),&part));
            CL_CHECK(clSetKernelArg(krd,1,sizeof(cl_mem),&dst));
            CL_CHECK(clSetKernelArg(krd,2,sizeof(cl_int),&M));
            CL_CHECK(clSetKernelArg(krd,3,sizeof(cl_int),&ksplit));
            size_t gx=((size_t)((M/2+63)/64))*64;
            size_t lsk[3]={64,(size_t)nsg,1}, gsk[3]={gx,(size_t)(nsg*ksplit),1};
            size_t lrd[3]={64,1,1}, grd[3]={((size_t)((M+63)/64))*64,1,1};
            bool need_reduce = (ksplit>1);
            for(int i=0;i<40;i++){ CL_CHECK(clEnqueueNDRangeKernel(q,ksk,3,0,gsk,lsk,0,0,0));
                                   if(need_reduce) CL_CHECK(clEnqueueNDRangeKernel(q,krd,3,0,grd,lrd,0,0,0)); }
            CL_CHECK(clFinish(q));
            std::vector<double> ts;
            for(int i=0;i<iters;i++){
                cl_event e1,e2; double t=0;
                CL_CHECK(clEnqueueNDRangeKernel(q,ksk,3,0,gsk,lsk,0,0,&e1));
                if(need_reduce) CL_CHECK(clEnqueueNDRangeKernel(q,krd,3,0,grd,lrd,0,0,&e2));
                CL_CHECK(clFinish(q));
                cl_ulong s0,e0; clGetEventProfilingInfo(e1,CL_PROFILING_COMMAND_START,8,&s0,0); clGetEventProfilingInfo(e1,CL_PROFILING_COMMAND_END,8,&e0,0);
                t += (e0-s0)*1e-6;
                if(need_reduce){ clGetEventProfilingInfo(e2,CL_PROFILING_COMMAND_START,8,&s0,0); clGetEventProfilingInfo(e2,CL_PROFILING_COMMAND_END,8,&e0,0); t += (e0-s0)*1e-6; clReleaseEvent(e2);}
                ts.push_back(t); clReleaseEvent(e1);
            }
            double tm=median(ts);
            char lbl[64]; snprintf(lbl,sizeof(lbl),"nsg=%-2d ksplit=%-2d (%d WGs)", nsg, ksplit, (int)((M/2+63)/64)*ksplit);
            printf("  %-21s | %7.4f %5.1f\n", lbl, tm, wbytes/tm/1e6);
        }
        clReleaseMemObject(qi);clReleaseMemObject(bbi);clReleaseMemObject(qb);clReleaseMemObject(db);
        clReleaseMemObject(dmb);clReleaseMemObject(sbf);clReleaseMemObject(bb);clReleaseMemObject(dst);clReleaseMemObject(part);
    }
    return 0;
}

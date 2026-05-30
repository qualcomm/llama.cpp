// Adreno peak FP32/FP16 throughput microbench (FMA-bound) + device specs.
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <cstdio>
#include <cstdlib>

#define CK(x) do{ cl_int cke_=(x); if(cke_!=CL_SUCCESS){ printf("CL err %d @ %d\n",cke_,__LINE__); exit(1);} }while(0)

// 16 independent FMA accumulators per thread to saturate the ALU pipeline,
// ITERS loop iterations, each doing 16 mads = 32 flops. Final reduce -> out.
static const char* SRC = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define NACC 16
__kernel void f32(__global float* out, const uint iters, const float seed){
    float a[NACC];
    #pragma unroll
    for(int i=0;i<NACC;i++) a[i]=seed+i;
    float b=seed*1.1f, c=seed*0.9f;
    for(uint k=0;k<iters;k++){
        #pragma unroll
        for(int i=0;i<NACC;i++) a[i]=fma(a[i],b,c);
    }
    float s=0; for(int i=0;i<NACC;i++) s+=a[i];
    out[get_global_id(0)]=s;
}
__kernel void f16(__global float* out, const uint iters, const float seedf){
    half a[NACC]; half seed=(half)seedf;
    #pragma unroll
    for(int i=0;i<NACC;i++) a[i]=seed+(half)i;
    half b=seed*(half)1.1f, c=seed*(half)0.9f;
    for(uint k=0;k<iters;k++){
        #pragma unroll
        for(int i=0;i<NACC;i++) a[i]=fma(a[i],b,c);
    }
    float s=0; for(int i=0;i<NACC;i++) s+=(float)a[i];
    out[get_global_id(0)]=s;
}
// half8 vectorized FMA (Adreno packs fp16x2; vec may expose more)
__kernel void f16v(__global float* out, const uint iters, const float seedf){
    half8 a[NACC]; half seed=(half)seedf;
    #pragma unroll
    for(int i=0;i<NACC;i++) a[i]=(half8)(seed+(half)i);
    half8 b=(half8)(seed*(half)1.1f), c=(half8)(seed*(half)0.9f);
    for(uint k=0;k<iters;k++){
        #pragma unroll
        for(int i=0;i<NACC;i++) a[i]=fma(a[i],b,c);
    }
    float s=0; for(int i=0;i<NACC;i++){ half8 v=a[i]; s+=v.s0+v.s1+v.s2+v.s3+v.s4+v.s5+v.s6+v.s7; }
    out[get_global_id(0)]=s;
}
)CLC";

int main(){
    cl_platform_id plat; CK(clGetPlatformIDs(1,&plat,0));
    cl_device_id dev; CK(clGetDeviceIDs(plat,CL_DEVICE_TYPE_GPU,1,&dev,0));
    char nm[256]; clGetDeviceInfo(dev,CL_DEVICE_NAME,sizeof(nm),nm,0);
    cl_uint cu=0,freq=0; clGetDeviceInfo(dev,CL_DEVICE_MAX_COMPUTE_UNITS,4,&cu,0);
    clGetDeviceInfo(dev,CL_DEVICE_MAX_CLOCK_FREQUENCY,4,&freq,0);
    printf("device: %s\n  CL_DEVICE_MAX_COMPUTE_UNITS=%u  CL_DEVICE_MAX_CLOCK_FREQUENCY=%u MHz\n",nm,cu,freq);
    cl_int e; cl_context ctx=clCreateContext(0,1,&dev,0,0,&e); CK(e);
    cl_command_queue q=clCreateCommandQueue(ctx,dev,CL_QUEUE_PROFILING_ENABLE,&e); CK(e);
    cl_program prog=clCreateProgramWithSource(ctx,1,&SRC,0,&e); CK(e);
    e=clBuildProgram(prog,1,&dev,"",0,0);
    if(e!=CL_SUCCESS){char log[8192];clGetProgramBuildInfo(prog,dev,CL_PROGRAM_BUILD_LOG,sizeof(log),log,0);printf("%s\n",log);exit(1);}
    const size_t NT=1u<<20; const cl_uint ITERS=3000;
    cl_mem out=clCreateBuffer(ctx,CL_MEM_WRITE_ONLY,NT*4,0,&e); CK(e);
    struct K{const char*n; double fpw;}; // flops per work-item per iter = NACC*2*width
    K ks[]={{"f32",16*2*1},{"f16",16*2*1},{"f16v",16*2*8}};
    size_t lws=256,gws=NT; float seed=1.0001f;
    for(auto&k:ks){
        cl_kernel ker=clCreateKernel(prog,k.n,&e); CK(e);
        CK(clSetKernelArg(ker,0,sizeof(cl_mem),&out));
        CK(clSetKernelArg(ker,1,sizeof(cl_uint),&ITERS));
        CK(clSetKernelArg(ker,2,sizeof(float),&seed));
        for(int w=0;w<3;w++) CK(clEnqueueNDRangeKernel(q,ker,1,0,&gws,&lws,0,0,0));
        CK(clFinish(q));
        double best=1e30;
        for(int it=0;it<10;it++){cl_event ev;CK(clEnqueueNDRangeKernel(q,ker,1,0,&gws,&lws,0,0,&ev));CK(clWaitForEvents(1,&ev));
            cl_ulong s,en;clGetEventProfilingInfo(ev,CL_PROFILING_COMMAND_START,8,&s,0);clGetEventProfilingInfo(ev,CL_PROFILING_COMMAND_END,8,&en,0);clReleaseEvent(ev);
            double ms=(en-s)/1e6; if(ms<best)best=ms;}
        double flops=(double)NT*ITERS*k.fpw;
        printf("  %-5s best %.2f ms -> %6.0f GFLOP/s (%.2f TFLOP/s)\n",k.n,best,flops/(best/1e3)/1e9,flops/(best/1e3)/1e12);
        clReleaseKernel(ker);
    }
    return 0;
}

// De-risk for the GEMV weight-layout redesign: does a tiled-WIDE (uint4,
// cross-thread-coalesced) weight read beat the current strided-NARROW (uint,
// cross-thread-coalesced) read at matched work? Same total bytes + same MACs;
// only the per-thread load width/pattern differs.
//   strided : w[k*M + row]      -> 1 uint/load, K loads/row, threads coalesced (narrow)
//   tiled   : w[(rt*K4+kb)*64+rit] -> uint4/load, K/4 loads/row, threads coalesced (wide)
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <cstdio>
#include <cstdlib>
#define CK(x) do{ cl_int e_=(x); if(e_!=CL_SUCCESS){printf("CL err %d @ %d\n",e_,__LINE__);exit(1);} }while(0)

static const char* SRC = R"CLC(
// REGP dummy local accumulators emulate GEMV register pressure (o_acc etc.)
__kernel void strided(__global const uint* w, __global const float* act, __global float* out, uint M, uint K){
    uint row=get_global_id(0); if(row>=M) return;
    float acc=0.0f;
    for(uint k=0;k<K;k++){ uint q=w[k*M+row]; acc=fma((float)(q&0xFFu),act[k],acc); }
    out[row]=acc;
}
__kernel void tiled(__global const uint4* w, __global const float* act, __global float* out, uint M, uint K4){
    uint row=get_global_id(0); if(row>=M) return;
    uint rt=row>>6, rit=row&63u;            // 64-row tiles
    float acc=0.0f;
    for(uint kb=0;kb<K4;kb++){
        uint4 q=w[(rt*K4+kb)*64u+rit];      // adjacent rit -> adjacent uint4 (coalesced); contiguous 4 K per thread
        acc=fma((float)(q.x&0xFFu),act[kb*4+0],acc);
        acc=fma((float)(q.y&0xFFu),act[kb*4+1],acc);
        acc=fma((float)(q.z&0xFFu),act[kb*4+2],acc);
        acc=fma((float)(q.w&0xFFu),act[kb*4+3],acc);
    }
    out[row]=acc;
}
)CLC";

int main(){
    cl_platform_id p; CK(clGetPlatformIDs(1,&p,0)); cl_device_id d; CK(clGetDeviceIDs(p,CL_DEVICE_TYPE_GPU,1,&d,0));
    char nm[256]; clGetDeviceInfo(d,CL_DEVICE_NAME,sizeof(nm),nm,0); printf("device: %s\n",nm);
    cl_int e; cl_context c=clCreateContext(0,1,&d,0,0,&e); CK(e);
    cl_command_queue q=clCreateCommandQueue(c,d,CL_QUEUE_PROFILING_ENABLE,&e); CK(e);
    cl_program pr=clCreateProgramWithSource(c,1,&SRC,0,&e); CK(e);
    e=clBuildProgram(pr,1,&d,"",0,0); if(e){char l[8192];clGetProgramBuildInfo(pr,d,CL_PROGRAM_BUILD_LOG,sizeof(l),l,0);printf("%s\n",l);exit(1);}
    const cl_uint M=65536, K=2816, K4=K/4;      // q6_K-embed-like row count; K=2816 (Gemma embd)
    const size_t WBYTES=(size_t)M*K*4;          // 738 MB, cache-busting
    cl_mem w=clCreateBuffer(c,CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR,WBYTES,0,&e); CK(e);
    cl_mem act=clCreateBuffer(c,CL_MEM_READ_ONLY,K*4,0,&e); CK(e);
    cl_mem out=clCreateBuffer(c,CL_MEM_WRITE_ONLY,(size_t)M*4,0,&e); CK(e);
    { unsigned* wp=(unsigned*)clEnqueueMapBuffer(q,w,CL_TRUE,CL_MAP_WRITE,0,WBYTES,0,0,0,&e);CK(e); for(size_t i=0;i<(size_t)M*K;i++) wp[i]=(unsigned)i; CK(clEnqueueUnmapMemObject(q,w,wp,0,0,0)); }
    { float* ap=(float*)clEnqueueMapBuffer(q,act,CL_TRUE,CL_MAP_WRITE,0,K*4,0,0,0,&e);CK(e); for(unsigned i=0;i<K;i++) ap[i]=1.0f/(i+1); CK(clEnqueueUnmapMemObject(q,act,ap,0,0,0)); }
    CK(clFinish(q));
    size_t lws=64, gws=M;
    const char* names[2]={"strided","tiled"}; cl_uint ksz[2]={K,K4};
    for(int t=0;t<2;t++){
        cl_kernel k=clCreateKernel(pr,names[t],&e); CK(e);
        CK(clSetKernelArg(k,0,sizeof(cl_mem),&w)); CK(clSetKernelArg(k,1,sizeof(cl_mem),&act));
        CK(clSetKernelArg(k,2,sizeof(cl_mem),&out)); CK(clSetKernelArg(k,3,sizeof(cl_uint),&M));
        CK(clSetKernelArg(k,4,sizeof(cl_uint),&ksz[t]));
        for(int w_=0;w_<3;w_++) CK(clEnqueueNDRangeKernel(q,k,1,0,&gws,&lws,0,0,0)); CK(clFinish(q));
        double best=1e30;
        for(int it=0;it<20;it++){ cl_event ev; CK(clEnqueueNDRangeKernel(q,k,1,0,&gws,&lws,0,0,&ev)); CK(clWaitForEvents(1,&ev));
            cl_ulong s,en; clGetEventProfilingInfo(ev,CL_PROFILING_COMMAND_START,8,&s,0); clGetEventProfilingInfo(ev,CL_PROFILING_COMMAND_END,8,&en,0); clReleaseEvent(ev);
            double ms=(en-s)/1e6; if(ms<best)best=ms; }
        printf("  %-8s best %.2f ms -> %6.1f GB/s\n",names[t],best,WBYTES/(best/1e3)/1e9);
        clReleaseKernel(k);
    }
    return 0;
}

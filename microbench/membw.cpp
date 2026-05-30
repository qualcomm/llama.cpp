// Adreno read-bandwidth microbench: scalar vs 128-bit (float4) vs wide-unrolled
// vectorized loads over a large (cache-busting) device buffer.
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>

#define CK(x) do{ cl_int e=(x); if(e!=CL_SUCCESS){ printf("CL err %d at %s:%d\n",e,__FILE__,__LINE__); exit(1);} }while(0)

static const char* SRC = R"CLC(
// Each kernel reads the whole buffer once (grid-stride) and accumulates, then
// writes one value per thread so the compiler can't DCE the loads.
__kernel void read_scalar(__global const float* a, __global float* out, const uint n){
    uint gid=get_global_id(0), stride=get_global_size(0);
    float acc=0.0f;
    for(uint i=gid;i<n;i+=stride) acc+=a[i];
    out[gid]=acc;
}
__kernel void read_f4(__global const float4* a, __global float* out, const uint n4){
    uint gid=get_global_id(0), stride=get_global_size(0);
    float4 acc=(float4)(0.0f);
    for(uint i=gid;i<n4;i+=stride) acc+=a[i];
    out[gid]=acc.x+acc.y+acc.z+acc.w;
}
// 4x unrolled float4 (process 4 consecutive float4 per thread per step)
__kernel void read_f4x4(__global const float4* a, __global float* out, const uint n4){
    uint gid=get_global_id(0), stride=get_global_size(0)*4;
    float4 acc=(float4)(0.0f);
    for(uint i=gid*4;i+3<n4;i+=stride){
        acc+=a[i]+a[i+1]+a[i+2]+a[i+3];
    }
    out[gid]=acc.x+acc.y+acc.z+acc.w;
}
// 8x unrolled float4
__kernel void read_f4x8(__global const float4* a, __global float* out, const uint n4){
    uint gid=get_global_id(0), stride=get_global_size(0)*8;
    float4 acc=(float4)(0.0f);
    for(uint i=gid*8;i+7<n4;i+=stride){
        acc+=a[i]+a[i+1]+a[i+2]+a[i+3]+a[i+4]+a[i+5]+a[i+6]+a[i+7];
    }
    out[gid]=acc.x+acc.y+acc.z+acc.w;
}
// image1d_buffer (texture cache) float4 read — the path the q6_K embed used
__kernel void read_img(__read_only image1d_buffer_t a, __global float* out, const uint n4){
    uint gid=get_global_id(0), stride=get_global_size(0);
    float4 acc=(float4)(0.0f);
    for(uint i=gid;i<n4;i+=stride) acc+=read_imagef(a,(int)i);
    out[gid]=acc.x+acc.y+acc.z+acc.w;
}
)CLC";

int main(){
    cl_platform_id plat; CK(clGetPlatformIDs(1,&plat,0));
    cl_device_id dev; CK(clGetDeviceIDs(plat,CL_DEVICE_TYPE_GPU,1,&dev,0));
    char nm[256]; clGetDeviceInfo(dev,CL_DEVICE_NAME,sizeof(nm),nm,0);
    printf("device: %s\n",nm);
    cl_int e;
    cl_context ctx=clCreateContext(0,1,&dev,0,0,&e); CK(e);
    cl_command_queue q=clCreateCommandQueue(ctx,dev,CL_QUEUE_PROFILING_ENABLE,&e); CK(e);
    cl_program prog=clCreateProgramWithSource(ctx,1,&SRC,0,&e); CK(e);
    e=clBuildProgram(prog,1,&dev,"-cl-fast-relaxed-math",0,0);
    if(e!=CL_SUCCESS){ char log[8192]; clGetProgramBuildInfo(prog,dev,CL_PROGRAM_BUILD_LOG,sizeof(log),log,0); printf("build:\n%s\n",log); exit(1);}

    const size_t BYTES = 512ull*1024*1024;     // 512 MB, well past any cache
    const cl_uint N = (cl_uint)(BYTES/4);
    const cl_uint N4 = N/4;
    // ALLOC_HOST_PTR per the Adreno zero-copy convention
    cl_mem a=clCreateBuffer(ctx,CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR,BYTES,0,&e); CK(e);
    // init via mapped pointer (no clEnqueueCopyBuffer)
    float* ap=(float*)clEnqueueMapBuffer(q,a,CL_TRUE,CL_MAP_WRITE,0,BYTES,0,0,0,&e); CK(e);
    for(cl_uint i=0;i<N;i++) ap[i]=1.0f;
    CK(clEnqueueUnmapMemObject(q,a,ap,0,0,0));
    CK(clFinish(q));

    const size_t NTHREADS = 1u<<20;            // 1M threads
    cl_mem out=clCreateBuffer(ctx,CL_MEM_WRITE_ONLY,NTHREADS*4,0,&e); CK(e);

    // image1d_buffer view over the same buffer (CL_RGBA/CL_FLOAT = 16 B/pixel = 1 float4)
    cl_mem a_img=0;
    {
        size_t maxw=0; clGetDeviceInfo(dev,CL_DEVICE_IMAGE_MAX_BUFFER_SIZE,sizeof(maxw),&maxw,0);
        cl_image_format f={CL_RGBA,CL_FLOAT};
        cl_image_desc d; memset(&d,0,sizeof(d)); d.image_type=CL_MEM_OBJECT_IMAGE1D_BUFFER;
        d.image_width=N4; d.buffer=a;
        if(N4<=maxw){ a_img=clCreateImage(ctx,CL_MEM_READ_ONLY,&f,&d,0,&e); if(e!=CL_SUCCESS){printf("img create err %d (maxw=%zu)\n",e,maxw); a_img=0;} }
        else printf("(image skip: N4=%u > max %zu)\n",N4,maxw);
    }

    struct K{const char*name; cl_uint n; bool img;};
    K ks[]={{"read_scalar",N,false},{"read_f4",N4,false},{"read_f4x4",N4,false},{"read_f4x8",N4,false},{"read_img",N4,true}};
    size_t lws=256, gws=NTHREADS;
    for(auto&k:ks){
        if(k.img && !a_img) continue;
        cl_kernel ker=clCreateKernel(prog,k.name,&e); CK(e);
        CK(clSetKernelArg(ker,0,sizeof(cl_mem),k.img?&a_img:&a));
        CK(clSetKernelArg(ker,1,sizeof(cl_mem),&out));
        CK(clSetKernelArg(ker,2,sizeof(cl_uint),&k.n));
        // warmup
        for(int w=0;w<3;w++) CK(clEnqueueNDRangeKernel(q,ker,1,0,&gws,&lws,0,0,0));
        CK(clFinish(q));
        double best=1e30;
        for(int it=0;it<30;it++){
            cl_event ev; CK(clEnqueueNDRangeKernel(q,ker,1,0,&gws,&lws,0,0,&ev));
            CK(clWaitForEvents(1,&ev));
            cl_ulong s,en; clGetEventProfilingInfo(ev,CL_PROFILING_COMMAND_START,8,&s,0);
            clGetEventProfilingInfo(ev,CL_PROFILING_COMMAND_END,8,&en,0); clReleaseEvent(ev);
            double ms=(en-s)/1.e6; if(ms<best) best=ms;
        }
        double gbps=BYTES/(best/1e3)/1e9;
        printf("  %-12s best %.3f ms  -> %6.1f GB/s\n",k.name,best,gbps);
        clReleaseKernel(ker);
    }
    return 0;
}

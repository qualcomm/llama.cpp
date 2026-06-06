// Megakernel prototype: a decode-shaped heterogeneous stage chain run in ONE
// launch with software grid barriers between stages, vs the equivalent sequence
// of separate dispatches. Builds on the validated barrier from
// global_barrier_microbench.cpp.
//
// Chain (S iterations of two stages, mimicking a transformer block's backbone):
//   stage A  rms_norm(x) -> xn       — SINGLE workgroup (decode norm is 1 token
//                                       = 1 row; the other WGs idle at the barrier)
//   stage B  y = W . xn              — ALL workgroups, persistent grid-stride over
//                                       the D output rows (D > resident WG count)
//   then x <- y for the next iteration.
//
// This exercises the two things a real block megakernel needs that the trivial
// microbench didn't: (1) heterogeneous stages (single-WG and all-WG) under one
// barrier, (2) realistic per-stage work, so we can find the CROSSOVER — the stage
// size below which replacing a dispatch boundary (~3.5-5 us) with a software
// barrier (~2 us) actually wins. Above it, the matmul dwarfs the overhead.
//
// Grid is sized to R WGs (<= resident capacity, so the barrier never deadlocks by
// construction); each WG grid-strides over its rows.
//
// Build: microbench/build_megablock.bat   Usage: mb_megablock.exe [D] [S] [R] [iters]

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>

#define CL_CHECK(expr) do { cl_int _e=(expr); if(_e!=CL_SUCCESS){fprintf(stderr,"CL err %d at %d (%s)\n",_e,__LINE__,#expr);exit(1);} } while(0)
static cl_program build(cl_context c, cl_device_id d, const char* src, const char* opts){
    cl_int e; cl_program p=clCreateProgramWithSource(c,1,&src,0,&e); CL_CHECK(e);
    e=clBuildProgram(p,1,&d,opts,0,0);
    if(e!=CL_SUCCESS){ size_t n=0; clGetProgramBuildInfo(p,d,CL_PROGRAM_BUILD_LOG,0,0,&n); std::vector<char> l(n+1,0);
        clGetProgramBuildInfo(p,d,CL_PROGRAM_BUILD_LOG,n,l.data(),0); fprintf(stderr,"BUILD FAIL:\n%s\n",l.data()); exit(1);}
    return p;
}
static double now_ms(){ using namespace std::chrono; return duration<double,std::milli>(steady_clock::now().time_since_epoch()).count(); }
static double median(std::vector<double> v){ std::sort(v.begin(),v.end()); return v.empty()?0:v[v.size()/2]; }

static const char* SRC = R"CLC(
#define LWS 128

inline void grid_barrier(volatile global int* counter, int R, int lid) {
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (lid == 0) { atomic_inc(counter); while (atomic_add(counter,0) < R) {} }
    barrier(CLK_GLOBAL_MEM_FENCE);
}

// rms_norm of x[D] -> xn[D], done cooperatively by ONE workgroup (lid 0..LWS-1).
inline void rms_norm_1wg(global const float* x, global float* xn, int D, int lid, local float* red) {
    float p = 0.0f;
    for (int i = lid; i < D; i += LWS) p += x[i]*x[i];
    red[lid] = p; barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = LWS/2; s > 0; s >>= 1) { if (lid < s) red[lid]+=red[lid+s]; barrier(CLK_LOCAL_MEM_FENCE); }
    float scale = rsqrt(red[0]/D + 1e-6f);
    for (int i = lid; i < D; i += LWS) xn[i] = x[i]*scale;
}

// y[M] = W[M*K] . xn[K], all WGs, persistent grid-stride over rows m.
inline void gemv(global const float* W, global const float* xn, global float* y,
                 int K, int M, int R, int wg, int lid, local float* red) {
    for (int m = wg; m < M; m += R) {
        float p = 0.0f;
        global const float* Wr = W + (long)m*K;
        for (int k = lid; k < K; k += LWS) p += Wr[k]*xn[k];
        red[lid] = p; barrier(CLK_LOCAL_MEM_FENCE);
        for (int s = LWS/2; s > 0; s >>= 1) { if (lid < s) red[lid]+=red[lid+s]; barrier(CLK_LOCAL_MEM_FENCE); }
        if (lid == 0) y[m] = red[0];
        barrier(CLK_LOCAL_MEM_FENCE); // reuse red next row
    }
}

// Megakernel: S iterations of [rms (WG0) | barrier | gemv (all) | barrier].
// bufs: a[D], b[D] ping-pong; W[D*D]; counters[2*S] pre-zeroed.
kernel void mega_block(global float* a, global float* b, global const float* W,
                       volatile global int* counters, int D, int S, int R) {
    local float red[LWS];
    int wg = get_group_id(0), lid = get_local_id(0);
    global float* x = a; global float* xn = b;
    for (int s = 0; s < S; ++s) {
        if (wg == 0) rms_norm_1wg(x, xn, D, lid, red);   // single-WG stage
        grid_barrier(counters + 2*s, R, lid);
        gemv(W, xn, x, D, D, R, wg, lid, red);           // all-WG stage, writes back into x
        if (s + 1 < S) grid_barrier(counters + 2*s + 1, R, lid);
    }
}

// Dispatch-per-stage baseline kernels.
kernel void k_rms(global const float* x, global float* xn, int D) {
    local float red[LWS];
    int lid = get_local_id(0);
    if (get_group_id(0) == 0) rms_norm_1wg(x, xn, D, lid, red);
}
kernel void k_gemv(global const float* W, global const float* xn, global float* y, int K, int M, int R) {
    local float red[LWS];
    gemv(W, xn, y, K, M, R, get_group_id(0), get_local_id(0), red);
}
)CLC";

int main(int argc, char** argv){
    int D     = argc>1 ? atoi(argv[1]) : 2560;
    int S     = argc>2 ? atoi(argv[2]) : 16;
    int R     = argc>3 ? atoi(argv[3]) : 128;
    int iters = argc>4 ? atoi(argv[4]) : 100;
    const int LWS = 128;

    cl_platform_id plat; CL_CHECK(clGetPlatformIDs(1,&plat,0));
    cl_device_id dev; CL_CHECK(clGetDeviceIDs(plat,CL_DEVICE_TYPE_GPU,1,&dev,0));
    char name[256]={0}; clGetDeviceInfo(dev,CL_DEVICE_NAME,sizeof(name),name,0);
    printf("device: %s | D=%d S=%d R=%d LWS=%d iters=%d\n", name, D, S, R, LWS, iters);

    cl_int e;
    cl_context ctx=clCreateContext(0,1,&dev,0,0,&e); CL_CHECK(e);
    cl_command_queue q=clCreateCommandQueue(ctx,dev,0,&e); CL_CHECK(e);
    cl_program prog=build(ctx,dev,SRC,"");
    cl_kernel k_mega=clCreateKernel(prog,"mega_block",&e); CL_CHECK(e);
    cl_kernel k_rms =clCreateKernel(prog,"k_rms",&e); CL_CHECK(e);
    cl_kernel k_gemv=clCreateKernel(prog,"k_gemv",&e); CL_CHECK(e);

    std::vector<float> hW((size_t)D*D), hx(D);
    for (size_t i=0;i<hW.size();++i) hW[i] = ((i*1103515245u+12345u)>>16 & 0xff)/2550.0f - 0.05f;
    for (int i=0;i<D;++i) hx[i] = ((i*7u+3u)%101)/101.0f;

    cl_mem a=clCreateBuffer(ctx,CL_MEM_READ_WRITE,(size_t)D*sizeof(float),0,&e); CL_CHECK(e);
    cl_mem b=clCreateBuffer(ctx,CL_MEM_READ_WRITE,(size_t)D*sizeof(float),0,&e); CL_CHECK(e);
    cl_mem W=clCreateBuffer(ctx,CL_MEM_READ_ONLY,(size_t)D*D*sizeof(float),0,&e); CL_CHECK(e);
    cl_mem counters=clCreateBuffer(ctx,CL_MEM_READ_WRITE,2*S*sizeof(int),0,&e); CL_CHECK(e);
    CL_CHECK(clEnqueueWriteBuffer(q,W,CL_TRUE,0,(size_t)D*D*sizeof(float),hW.data(),0,0,0));
    int zero=0; size_t gws=(size_t)R*LWS, lws=LWS;

    // ---- correctness: megakernel vs sequential CPU reference ----
    CL_CHECK(clEnqueueWriteBuffer(q,a,CL_TRUE,0,(size_t)D*sizeof(float),hx.data(),0,0,0));
    CL_CHECK(clEnqueueFillBuffer(q,counters,&zero,sizeof(int),0,2*S*sizeof(int),0,0,0));
    CL_CHECK(clFinish(q));
    CL_CHECK(clSetKernelArg(k_mega,0,sizeof(cl_mem),&a));
    CL_CHECK(clSetKernelArg(k_mega,1,sizeof(cl_mem),&b));
    CL_CHECK(clSetKernelArg(k_mega,2,sizeof(cl_mem),&W));
    CL_CHECK(clSetKernelArg(k_mega,3,sizeof(cl_mem),&counters));
    CL_CHECK(clSetKernelArg(k_mega,4,sizeof(int),&D));
    CL_CHECK(clSetKernelArg(k_mega,5,sizeof(int),&S));
    CL_CHECK(clSetKernelArg(k_mega,6,sizeof(int),&R));
    CL_CHECK(clEnqueueNDRangeKernel(q,k_mega,1,0,&gws,&lws,0,0,0));
    std::vector<float> gpu(D); CL_CHECK(clEnqueueReadBuffer(q,a,CL_TRUE,0,(size_t)D*sizeof(float),gpu.data(),0,0,0));
    // CPU ref
    std::vector<float> cx=hx, cxn(D);
    for (int s=0;s<S;++s){
        double ss=0; for(int i=0;i<D;++i) ss+=(double)cx[i]*cx[i];
        float sc=1.0f/std::sqrt((float)(ss/D)+1e-6f);
        for(int i=0;i<D;++i) cxn[i]=cx[i]*sc;
        for(int m=0;m<D;++m){ double p=0; const float*Wr=&hW[(size_t)m*D]; for(int k=0;k<D;++k) p+=(double)Wr[k]*cxn[k]; cx[m]=(float)p; }
    }
    double num=0, den=0; for(int i=0;i<D;++i){ double d=(double)gpu[i]-cx[i]; num+=d*d; den+=(double)cx[i]*cx[i]; }
    double l2rel = std::sqrt(num/(den+1e-12));
    printf("correctness: L2 rel err vs CPU ref = %.3e  (%s)\n", l2rel, l2rel<1e-3?"OK":"WRONG");

    // ---- perf: megakernel (1 launch + 2S-1 barriers) vs 2S dispatches ----
    std::vector<double> tm, td;
    for(int it=0; it<iters; ++it){
        CL_CHECK(clEnqueueWriteBuffer(q,a,CL_FALSE,0,(size_t)D*sizeof(float),hx.data(),0,0,0));
        CL_CHECK(clEnqueueFillBuffer(q,counters,&zero,sizeof(int),0,2*S*sizeof(int),0,0,0));
        CL_CHECK(clFinish(q));
        double t0=now_ms();
        CL_CHECK(clEnqueueNDRangeKernel(q,k_mega,1,0,&gws,&lws,0,0,0));
        CL_CHECK(clFinish(q));
        tm.push_back(now_ms()-t0);
    }
    CL_CHECK(clSetKernelArg(k_rms,2,sizeof(int),&D));
    CL_CHECK(clSetKernelArg(k_gemv,3,sizeof(int),&D));
    CL_CHECK(clSetKernelArg(k_gemv,4,sizeof(int),&D));
    CL_CHECK(clSetKernelArg(k_gemv,5,sizeof(int),&R));
    for(int it=0; it<iters; ++it){
        CL_CHECK(clEnqueueWriteBuffer(q,a,CL_FALSE,0,(size_t)D*sizeof(float),hx.data(),0,0,0));
        CL_CHECK(clFinish(q));
        double t0=now_ms();
        for(int s=0;s<S;++s){
            CL_CHECK(clSetKernelArg(k_rms,0,sizeof(cl_mem),&a));
            CL_CHECK(clSetKernelArg(k_rms,1,sizeof(cl_mem),&b));
            CL_CHECK(clEnqueueNDRangeKernel(q,k_rms,1,0,&gws,&lws,0,0,0));
            CL_CHECK(clSetKernelArg(k_gemv,0,sizeof(cl_mem),&W));
            CL_CHECK(clSetKernelArg(k_gemv,1,sizeof(cl_mem),&b));
            CL_CHECK(clSetKernelArg(k_gemv,2,sizeof(cl_mem),&a));
            CL_CHECK(clEnqueueNDRangeKernel(q,k_gemv,1,0,&gws,&lws,0,0,0));
        }
        CL_CHECK(clFinish(q));
        td.push_back(now_ms()-t0);
    }
    double mm=median(tm), md=median(td);
    int nstage=2*S;
    printf("megakernel (1 launch, %d sw-barriers): %.4f ms  (%.3f us/stage)\n", nstage-1, mm, mm*1000.0/nstage);
    printf("%d dispatches                        : %.4f ms  (%.3f us/stage)\n", nstage, md, md*1000.0/nstage);
    printf("=> megakernel %s by %.1f%%  (saved %.3f us/stage-boundary)\n",
           mm<md?"FASTER":"slower", 100.0*(md-mm)/md, (md-mm)*1000.0/(nstage-1));
    return 0;
}

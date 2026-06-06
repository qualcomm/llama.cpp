// Microbench: software global (grid-wide) barrier on Adreno X2.
//
// Goal: de-risk the persistent-kernel / megakernel idea (run many graph stages
// in ONE kernel launch, replacing per-dispatch launch latency + L2 flush with an
// in-kernel cross-workgroup barrier). Two experiments:
//
//   1. CORRECTNESS + DEADLOCK THRESHOLD. A 2-stage kernel: stage-1 every WG
//      writes its id+1 into data[wg]; a global barrier; stage-2 every WG reads a
//      NEIGHBOR's stage-1 value. If the barrier truly synchronizes all WGs, the
//      neighbor read is correct. Sweep numWG upward; the barrier deadlocks once
//      numWG exceeds the number of WGs that can be co-resident on the GPU
//      (non-resident WGs never arrive -> resident WGs spin forever). A host
//      watchdog (poll the event, no clFinish) finds that threshold without
//      hanging the harness.
//
//   2. PERF. An N-barrier megakernel (1 launch, N internal global barriers, tiny
//      work per stage) vs the current model of N separate dispatches (N launches,
//      N-1 implicit driver barriers). If the megakernel is faster, the software
//      barrier is cheaper than a dispatch boundary -> the megakernel premise holds.
//
// The grid barrier is the classic Xiao&Feng arrival-counter design: one
// representative lane per WG atomically increments a per-phase counter, then all
// WGs spin (uncached atomic read) until the counter == numWG, with global mem
// fences around it for cross-WG visibility. Each phase uses a fresh pre-zeroed
// counter slot, so no sense-reversal/reset is needed mid-kernel.
//
// Build: microbench/build_barrier.bat   Usage: mb_barrier.exe [maxWG] [N_stages] [iters]

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// ---------------------------------------------------------------------------
static const char* SRC = R"CLC(
// One representative lane (lid 0) per workgroup arrives at counter[phase], then
// all lanes of the WG spin until every WG has arrived. Global mem fences make
// the prior stage's global writes visible across WGs before/after the barrier.
inline void grid_barrier(volatile global int* counter, int numWG, int lid) {
    barrier(CLK_GLOBAL_MEM_FENCE);                 // finish this WG's stage writes
    if (lid == 0) {
        atomic_inc(counter);                       // arrival (release via fence above)
        while (atomic_add(counter, 0) < numWG) { } // spin on uncached atomic read
    }
    barrier(CLK_GLOBAL_MEM_FENCE);                 // other lanes wait for lid 0; acquire
}

// Experiment 1: 2-stage cross-WG correctness / deadlock probe.
kernel void two_stage(global int* data, global int* out,
                      volatile global int* counter, int numWG) {
    int wg = get_group_id(0);
    int lid = get_local_id(0);
    if (lid == 0) data[wg] = wg + 1;               // stage 1
    grid_barrier(counter, numWG, lid);
    if (lid == 0) out[wg] = data[(wg + 1) % numWG]; // stage 2: read a neighbor
}

// Experiment 2a: N-barrier megakernel. counters[] is pre-zeroed [N]. Each stage
// does a tiny dependent op (read neighbor's accumulator, add) so the barrier is
// load-bearing. One launch, N-1 software barriers.
kernel void megakernel(global int* acc, volatile global int* counters,
                       int numWG, int nstages) {
    int wg = get_group_id(0);
    int lid = get_local_id(0);
    for (int s = 0; s < nstages; ++s) {
        if (lid == 0) acc[wg] += acc[(wg + 1) % numWG] + 1; // tiny stage work
        if (s + 1 < nstages) grid_barrier(counters + s, numWG, lid);
    }
}

// Experiment 2b: single stage of the dispatch-per-stage baseline (N launches).
kernel void one_stage(global int* acc, int numWG) {
    int wg = get_group_id(0);
    int lid = get_local_id(0);
    if (lid == 0) acc[wg] += acc[(wg + 1) % numWG] + 1;
}
)CLC";

int main(int argc, char** argv){
    int maxWG    = argc>1 ? atoi(argv[1]) : 256;
    int nstages  = argc>2 ? atoi(argv[2]) : 64;
    int iters    = argc>3 ? atoi(argv[3]) : 200;
    const int LWS = 64;

    cl_platform_id plat; CL_CHECK(clGetPlatformIDs(1,&plat,0));
    cl_device_id dev; CL_CHECK(clGetDeviceIDs(plat,CL_DEVICE_TYPE_GPU,1,&dev,0));
    char name[256]={0}; clGetDeviceInfo(dev,CL_DEVICE_NAME,sizeof(name),name,0);
    cl_uint cus=0; clGetDeviceInfo(dev,CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(cus),&cus,0);
    size_t maxwgs=0; clGetDeviceInfo(dev,CL_DEVICE_MAX_WORK_GROUP_SIZE,sizeof(maxwgs),&maxwgs,0);
    printf("device: %s | CL_DEVICE_MAX_COMPUTE_UNITS=%u | max WG size=%zu | LWS=%d\n", name, cus, maxwgs, LWS);

    cl_int e;
    cl_context ctx=clCreateContext(0,1,&dev,0,0,&e); CL_CHECK(e);
    cl_command_queue q=clCreateCommandQueue(ctx,dev,0,&e); CL_CHECK(e);
    cl_program prog=build(ctx,dev,SRC,"");
    cl_kernel k_two=clCreateKernel(prog,"two_stage",&e); CL_CHECK(e);
    cl_kernel k_mega=clCreateKernel(prog,"megakernel",&e); CL_CHECK(e);
    cl_kernel k_one=clCreateKernel(prog,"one_stage",&e); CL_CHECK(e);

    // ---- Experiment 1: correctness + deadlock threshold sweep ----
    printf("\n[1] cross-WG correctness + deadlock sweep (watchdog 2s, no clFinish on hang)\n");
    int last_ok = 0;
    for (int numWG = 4; numWG <= maxWG; numWG *= 2) {
        cl_mem data=clCreateBuffer(ctx,CL_MEM_READ_WRITE,numWG*sizeof(int),0,&e); CL_CHECK(e);
        cl_mem out =clCreateBuffer(ctx,CL_MEM_READ_WRITE,numWG*sizeof(int),0,&e); CL_CHECK(e);
        cl_mem ctr =clCreateBuffer(ctx,CL_MEM_READ_WRITE,sizeof(int),0,&e); CL_CHECK(e);
        int zero=0; CL_CHECK(clEnqueueFillBuffer(q,ctr,&zero,sizeof(int),0,sizeof(int),0,0,0));
        CL_CHECK(clEnqueueFillBuffer(q,out,&zero,sizeof(int),0,numWG*sizeof(int),0,0,0));
        CL_CHECK(clFinish(q));
        CL_CHECK(clSetKernelArg(k_two,0,sizeof(cl_mem),&data));
        CL_CHECK(clSetKernelArg(k_two,1,sizeof(cl_mem),&out));
        CL_CHECK(clSetKernelArg(k_two,2,sizeof(cl_mem),&ctr));
        CL_CHECK(clSetKernelArg(k_two,3,sizeof(int),&numWG));
        size_t gws=(size_t)numWG*LWS, lws=LWS;
        cl_event ev;
        CL_CHECK(clEnqueueNDRangeKernel(q,k_two,1,0,&gws,&lws,0,0,&ev));
        clFlush(q);
        // watchdog: poll event status, do NOT clFinish (would hang on deadlock)
        double t0=now_ms(); bool done=false;
        while (now_ms()-t0 < 2000.0) {
            cl_int st; clGetEventInfo(ev,CL_EVENT_COMMAND_EXECUTION_STATUS,sizeof(st),&st,0);
            if (st==CL_COMPLETE){ done=true; break; }
            if (st<0){ printf("  numWG=%4d: error status %d\n",numWG,st); break; }
        }
        if (!done) {
            printf("  numWG=%4d: DEADLOCK (no completion in 2s) -> resident WG capacity exceeded\n", numWG);
            printf("  => max co-resident workgroups is between %d and %d (LWS=%d).\n", last_ok, numWG, LWS);
            clReleaseEvent(ev);
            // abandon the wedged queue; exit without clFinish.
            printf("  (skipping perf test — would reuse the wedged context; rerun with maxWG <= %d)\n", last_ok);
            return 0;
        }
        // verify
        std::vector<int> h(numWG); CL_CHECK(clEnqueueReadBuffer(q,out,CL_TRUE,0,numWG*sizeof(int),h.data(),0,0,0));
        int bad=0; for(int i=0;i<numWG;i++){ int exp=((i+1)%numWG)+1; if(h[i]!=exp) bad++; }
        printf("  numWG=%4d: %s (%d/%d correct)\n", numWG, bad?"WRONG":"OK", numWG-bad, numWG);
        clReleaseEvent(ev); clReleaseMemObject(data); clReleaseMemObject(out); clReleaseMemObject(ctr);
        if(!bad) last_ok=numWG; else break;
    }

    // ---- Experiment 2: N-barrier megakernel vs N dispatches ----
    int numWG = last_ok>0 ? std::min(last_ok, maxWG) : 32;
    printf("\n[2] perf: %d-stage megakernel (1 launch, %d barriers) vs %d dispatches  @ numWG=%d, %d iters\n",
           nstages, nstages-1, nstages, numWG, iters);
    cl_mem acc=clCreateBuffer(ctx,CL_MEM_READ_WRITE,numWG*sizeof(int),0,&e); CL_CHECK(e);
    cl_mem counters=clCreateBuffer(ctx,CL_MEM_READ_WRITE,nstages*sizeof(int),0,&e); CL_CHECK(e);
    int zero=0; size_t gws=(size_t)numWG*LWS, lws=LWS;

    // megakernel
    std::vector<double> tm;
    for(int it=0; it<iters; ++it){
        CL_CHECK(clEnqueueFillBuffer(q,acc,&zero,sizeof(int),0,numWG*sizeof(int),0,0,0));
        CL_CHECK(clEnqueueFillBuffer(q,counters,&zero,sizeof(int),0,nstages*sizeof(int),0,0,0));
        CL_CHECK(clFinish(q));
        CL_CHECK(clSetKernelArg(k_mega,0,sizeof(cl_mem),&acc));
        CL_CHECK(clSetKernelArg(k_mega,1,sizeof(cl_mem),&counters));
        CL_CHECK(clSetKernelArg(k_mega,2,sizeof(int),&numWG));
        CL_CHECK(clSetKernelArg(k_mega,3,sizeof(int),&nstages));
        double t0=now_ms();
        CL_CHECK(clEnqueueNDRangeKernel(q,k_mega,1,0,&gws,&lws,0,0,0));
        CL_CHECK(clFinish(q));
        tm.push_back(now_ms()-t0);
    }
    // N dispatches
    std::vector<double> td;
    CL_CHECK(clSetKernelArg(k_one,0,sizeof(cl_mem),&acc));
    CL_CHECK(clSetKernelArg(k_one,1,sizeof(int),&numWG));
    for(int it=0; it<iters; ++it){
        CL_CHECK(clEnqueueFillBuffer(q,acc,&zero,sizeof(int),0,numWG*sizeof(int),0,0,0));
        CL_CHECK(clFinish(q));
        double t0=now_ms();
        for(int s=0;s<nstages;++s) CL_CHECK(clEnqueueNDRangeKernel(q,k_one,1,0,&gws,&lws,0,0,0));
        CL_CHECK(clFinish(q));
        td.push_back(now_ms()-t0);
    }
    double mm=median(tm), md=median(td);
    printf("  megakernel (1 launch + %d sw-barriers): %.4f ms/run  (%.3f us/stage)\n", nstages-1, mm, mm*1000.0/nstages);
    printf("  %d dispatches                          : %.4f ms/run  (%.3f us/dispatch)\n", nstages, md, md*1000.0/nstages);
    printf("  => software barrier vs dispatch boundary: %.3f us vs %.3f us per stage; megakernel %s (%.1f%%)\n",
           mm*1000.0/nstages, md*1000.0/nstages, mm<md?"FASTER":"slower", 100.0*(md-mm)/md);
    return 0;
}

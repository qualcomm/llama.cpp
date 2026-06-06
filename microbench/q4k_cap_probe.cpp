// q4_K GEMV cap + parallelism probe — settles the one risk left by the per-layer
// megakernel microbench: the real megakernel must embed a q4_K GEMV and run it at
// a persistent-grid R (<= resident cap), NOT the M WGs the standalone kernel uses.
// Two questions:
//   (A) DEADLOCK / RESIDENT CAP for a register-heavy q4_K-dequant kernel that holds
//       a grid barrier (lower than the light-probe's 256-511 → sets the max R).
//   (B) Does capping the q4_K GEMV to R grid-stride WGs lose much vs full-width M?
//       If capped-R ~= full-width, the embedded matmul won't eat the ~40us/block
//       dispatch saving. If capped-R is much slower, the megakernel is a wash.
//
// q4_K block bytes are synthesized (valid d/dmin/scales/qs) and dequantized by the
// SAME formula on CPU and GPU, so we validate the in-kernel dequant+GEMV without a
// real quantizer. Weight VALUES are arbitrary; BW + dequant ALU + registers match
// real q4_K (144 B / 256 weights = 4.5 bit). Build: build_q4k_probe.bat
//   Usage: mb_q4kprobe.exe [K] [M] [maxR] [iters]

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
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

// minimal f32 -> f16 (round-to-nearest-even, normal range only; values stay small)
static uint16_t f2h(float f){
    uint32_t x; memcpy(&x,&f,4);
    uint32_t sign=(x>>16)&0x8000; int32_t exp=((x>>23)&0xff)-127+15; uint32_t man=x&0x7fffff;
    if(exp<=0) return (uint16_t)sign;            // flush tiny to 0 (fine for probe)
    if(exp>=31) return (uint16_t)(sign|0x7c00);  // inf
    uint16_t h=(uint16_t)(sign | (exp<<10) | (man>>13));
    if(man&0x1000) h++;                           // round
    return h;
}
static float h2f(uint16_t h){
    uint32_t sign=(h&0x8000)<<16; int32_t exp=(h>>10)&0x1f; uint32_t man=h&0x3ff; uint32_t x;
    if(exp==0){ if(man==0){ x=sign; } else { exp=1; while(!(man&0x400)){man<<=1;exp--;} man&=0x3ff; x=sign|((exp+112)<<23)|(man<<13); } }
    else if(exp==31){ x=sign|0x7f800000|(man<<13); }
    else { x=sign|((exp+112)<<23)|(man<<13); }
    float f; memcpy(&f,&x,4); return f;
}

static const int QK_K=256, BLK=144; // bytes per q4_K block

static const char* SRC = R"CLC(
#define LWS 128

inline void grid_barrier(volatile global int* counter, int R, int lid) {
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (lid == 0) { atomic_inc(counter); while (atomic_add(counter,0) < R) {} }
    barrier(CLK_GLOBAL_MEM_FENCE);
}

// ggml get_scale_min_k4: 6-bit scale/min for sub-block j (0..7) from scales[12].
inline void get_scale_min_k4(int j, global const uchar* q, uchar* d, uchar* m){
    if (j < 4){ *d = q[j] & 63; *m = q[j+4] & 63; }
    else { *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
           *m = (q[j+4] >> 4)  | ((q[j-0] >> 6) << 4); }
}

// dot of q4_K row m with v[K], cooperatively by one WG (LWS lanes), grid-stride
// over rows. Each lane dequantizes its own k indices (register/ALU realistic).
inline void qk_gemv(global const uchar* W, global const float* v, global float* y,
                    int K, int M, int R, int wg, int lid, local float* red){
    int bpr = K / 256; // blocks per row
    for (int m = wg; m < M; m += R){
        global const uchar* row = W + (long)m * bpr * 144;
        float acc = 0.0f;
        for (int k = lid; k < K; k += LWS){
            int b = k >> 8;            // block
            int r = k & 255;           // within block 0..255
            global const uchar* blk = row + (long)b*144;
            float d    = vload_half(0,(global const half*)(blk));
            float dmin = vload_half(0,(global const half*)(blk+2));
            global const uchar* scales = blk + 4;
            global const uchar* qs     = blk + 16;
            int p  = r >> 6;           // 0..3 (64-group)
            int w64 = r & 63;
            int lo = w64 < 32;
            int qi = p*32 + (lo ? w64 : (w64-32));
            int q4 = lo ? (qs[qi] & 0xF) : (qs[qi] >> 4);
            int is = lo ? 2*p : 2*p+1;
            uchar sc, mn; get_scale_min_k4(is, scales, &sc, &mn);
            float wv = d * (float)sc * (float)q4 - dmin * (float)mn;
            acc += wv * v[k];
        }
        red[lid] = acc; barrier(CLK_LOCAL_MEM_FENCE);
        for (int s = LWS/2; s > 0; s >>= 1){ if (lid < s) red[lid]+=red[lid+s]; barrier(CLK_LOCAL_MEM_FENCE); }
        if (lid == 0) y[m] = red[0];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

// (B) pure GEMV — launched both full-width (R=M) and capped (R<M grid-stride).
kernel void k_qk_gemv(global const uchar* W, global const float* v, global float* y,
                      int K, int M, int R){
    local float red[LWS];
    qk_gemv(W, v, y, K, M, R, get_group_id(0), get_local_id(0), red);
}

// (A) deadlock probe: q4_K gemv (register pressure) then grid barrier then a
// cross-WG neighbor read. Verifies sync correctness AND finds the resident cap
// UNDER the gemv's occupancy. data/out integer check is independent of y.
kernel void k_qk_barrier(global const uchar* W, global const float* v, global float* y,
                         int K, int M, int R, global int* data, global int* out,
                         volatile global int* counter){
    local float red[LWS];
    int wg = get_group_id(0), lid = get_local_id(0);
    qk_gemv(W, v, y, K, M, R, wg, lid, red);     // stage 1 (work + register pressure)
    if (lid == 0) data[wg] = wg + 1;
    grid_barrier(counter, R, lid);
    if (lid == 0) out[wg] = data[(wg + 1) % R];  // stage 2: neighbor
}
)CLC";

int main(int argc, char** argv){
    int K     = argc>1 ? atoi(argv[1]) : 256;   // proj K=256 (worst case: tiny K, low work/WG)
    int M     = argc>2 ? atoi(argv[2]) : 2560;
    int maxR  = argc>3 ? atoi(argv[3]) : 512;
    int iters = argc>4 ? atoi(argv[4]) : 200;
    const int LWS = 128;
    int bpr = K/256;

    cl_platform_id plat; CL_CHECK(clGetPlatformIDs(1,&plat,0));
    cl_device_id dev; CL_CHECK(clGetDeviceIDs(plat,CL_DEVICE_TYPE_GPU,1,&dev,0));
    char name[256]={0}; clGetDeviceInfo(dev,CL_DEVICE_NAME,sizeof(name),name,0);
    cl_uint cus=0; clGetDeviceInfo(dev,CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(cus),&cus,0);
    printf("device: %s | CUs=%u | K=%d M=%d (q4_K, %d blocks/row) maxR=%d\n", name, cus, K, M, bpr, maxR);

    cl_int e;
    cl_context ctx=clCreateContext(0,1,&dev,0,0,&e); CL_CHECK(e);
    cl_command_queue q=clCreateCommandQueue(ctx,dev,0,&e); CL_CHECK(e);
    cl_program prog=build(ctx,dev,SRC,"");
    cl_kernel k_gemv=clCreateKernel(prog,"k_qk_gemv",&e); CL_CHECK(e);
    cl_kernel k_bar =clCreateKernel(prog,"k_qk_barrier",&e); CL_CHECK(e);

    // synthesize q4_K weight bytes for M rows
    size_t wbytes=(size_t)M*bpr*BLK;
    std::vector<uint8_t> hW(wbytes);
    auto rb=[](size_t i){ return (uint8_t)(((i*1103515245u+12345u)>>16)&0xff); };
    for(size_t b=0;b<(size_t)M*bpr;++b){
        uint8_t* blk=&hW[b*BLK];
        uint16_t d=f2h(0.02f+0.001f*(b%7)), dmin=f2h(0.01f+0.0005f*(b%5));
        memcpy(blk,&d,2); memcpy(blk+2,&dmin,2);
        for(int i=0;i<12;++i) blk[4+i]=rb(b*13+i)&0x3f;   // 6-bit-ish scale bytes
        for(int i=0;i<128;++i) blk[16+i]=rb(b*131+i);     // qs nibbles
    }
    std::vector<float> hv(K); for(int i=0;i<K;++i) hv[i]=(((i*7u+3u)%101)/101.0f)-0.5f;

    // CPU reference dequant + gemv (same formula as kernel)
    auto gsm=[&](int j,const uint8_t*qd,uint8_t&d,uint8_t&m){
        if(j<4){ d=qd[j]&63; m=qd[j+4]&63; }
        else { d=(qd[j+4]&0xF)|((qd[j-4]>>6)<<4); m=(qd[j+4]>>4)|((qd[j-0]>>6)<<4); } };
    std::vector<float> cref(M);
    for(int m=0;m<M;++m){
        const uint8_t* row=&hW[(size_t)m*bpr*BLK];
        double acc=0;
        for(int k=0;k<K;++k){
            int b=k>>8, r=k&255; const uint8_t* blk=row+(size_t)b*BLK;
            uint16_t dh,dmh; memcpy(&dh,blk,2); memcpy(&dmh,blk+2,2);
            float d=h2f(dh), dmin=h2f(dmh);
            const uint8_t* scales=blk+4; const uint8_t* qs=blk+16;
            int p=r>>6, w64=r&63, lo=w64<32; int qi=p*32+(lo?w64:(w64-32));
            int q4=lo?(qs[qi]&0xF):(qs[qi]>>4); int is=lo?2*p:2*p+1;
            uint8_t sc,mn; gsm(is,scales,sc,mn);
            float wv=d*(float)sc*(float)q4 - dmin*(float)mn;
            acc += (double)wv*hv[k];
        }
        cref[m]=(float)acc;
    }

    cl_mem W=clCreateBuffer(ctx,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,wbytes,hW.data(),&e); CL_CHECK(e);
    cl_mem v=clCreateBuffer(ctx,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,K*4,hv.data(),&e); CL_CHECK(e);
    cl_mem y=clCreateBuffer(ctx,CL_MEM_READ_WRITE,M*4,0,&e); CL_CHECK(e);

    // correctness at full width (R=M)
    { int R=M; size_t gws=(size_t)R*LWS, lws=LWS;
      clSetKernelArg(k_gemv,0,sizeof(cl_mem),&W); clSetKernelArg(k_gemv,1,sizeof(cl_mem),&v); clSetKernelArg(k_gemv,2,sizeof(cl_mem),&y);
      clSetKernelArg(k_gemv,3,sizeof(int),&K); clSetKernelArg(k_gemv,4,sizeof(int),&M); clSetKernelArg(k_gemv,5,sizeof(int),&R);
      CL_CHECK(clEnqueueNDRangeKernel(q,k_gemv,1,0,&gws,&lws,0,0,0));
      std::vector<float> hy(M); CL_CHECK(clEnqueueReadBuffer(q,y,CL_TRUE,0,M*4,hy.data(),0,0,0));
      double num=0,den=0; for(int i=0;i<M;++i){ double dd=(double)hy[i]-cref[i]; num+=dd*dd; den+=(double)cref[i]*cref[i]; }
      double l2=std::sqrt(num/(den+1e-12));
      printf("correctness (q4_K dequant+gemv vs CPU): L2 rel err = %.3e (%s)\n", l2, l2<1e-4?"OK":"WRONG");
    }

    // ---- (A) deadlock / resident-cap sweep under gemv occupancy ----
    printf("\n[A] deadlock sweep (q4_K gemv + grid barrier, watchdog 2s):\n");
    int last_ok=0;
    cl_mem data=clCreateBuffer(ctx,CL_MEM_READ_WRITE,maxR*sizeof(int),0,&e); CL_CHECK(e);
    cl_mem out =clCreateBuffer(ctx,CL_MEM_READ_WRITE,maxR*sizeof(int),0,&e); CL_CHECK(e);
    cl_mem ctr =clCreateBuffer(ctx,CL_MEM_READ_WRITE,sizeof(int),0,&e); CL_CHECK(e);
    for(int R=32; R<=maxR; R*=2){
        if(R>M) break;
        int zero=0; CL_CHECK(clEnqueueFillBuffer(q,ctr,&zero,sizeof(int),0,sizeof(int),0,0,0));
        CL_CHECK(clEnqueueFillBuffer(q,out,&zero,sizeof(int),0,R*sizeof(int),0,0,0));
        CL_CHECK(clFinish(q));
        clSetKernelArg(k_bar,0,sizeof(cl_mem),&W); clSetKernelArg(k_bar,1,sizeof(cl_mem),&v); clSetKernelArg(k_bar,2,sizeof(cl_mem),&y);
        clSetKernelArg(k_bar,3,sizeof(int),&K); clSetKernelArg(k_bar,4,sizeof(int),&M); clSetKernelArg(k_bar,5,sizeof(int),&R);
        clSetKernelArg(k_bar,6,sizeof(cl_mem),&data); clSetKernelArg(k_bar,7,sizeof(cl_mem),&out); clSetKernelArg(k_bar,8,sizeof(cl_mem),&ctr);
        size_t gws=(size_t)R*LWS, lws=LWS; cl_event ev;
        CL_CHECK(clEnqueueNDRangeKernel(q,k_bar,1,0,&gws,&lws,0,0,&ev)); clFlush(q);
        double t0=now_ms(); bool done=false;
        while(now_ms()-t0<2000.0){ cl_int st; clGetEventInfo(ev,CL_EVENT_COMMAND_EXECUTION_STATUS,sizeof(st),&st,0);
            if(st==CL_COMPLETE){done=true;break;} if(st<0){printf("  R=%4d: err %d\n",R,st);break;} }
        if(!done){ printf("  R=%4d: DEADLOCK (>2s) -> resident cap exceeded; max co-resident in (%d, %d]\n",R,last_ok,R);
            clReleaseEvent(ev); printf("  => SAFE MEGAKERNEL R <= %d for this q4_K occupancy\n", last_ok); goto perf; }
        std::vector<int> h(R); CL_CHECK(clEnqueueReadBuffer(q,out,CL_TRUE,0,R*sizeof(int),h.data(),0,0,0));
        int bad=0; for(int i=0;i<R;++i){ int exp=((i+1)%R)+1; if(h[i]!=exp) bad++; }
        printf("  R=%4d: %s (%d/%d cross-WG correct)\n", R, bad?"WRONG":"OK", R-bad, R);
        clReleaseEvent(ev); if(!bad) last_ok=R; else break;
    }
perf:
    // ---- (B) capped-R grid-stride vs full-width M speed ----
    printf("\n[B] q4_K GEMV speed: full-width (R=M=%d) vs capped grid-stride (no barrier):\n", M);
    auto bench=[&](int R){
        size_t gws=(size_t)R*LWS, lws=LWS;
        clSetKernelArg(k_gemv,0,sizeof(cl_mem),&W); clSetKernelArg(k_gemv,1,sizeof(cl_mem),&v); clSetKernelArg(k_gemv,2,sizeof(cl_mem),&y);
        clSetKernelArg(k_gemv,3,sizeof(int),&K); clSetKernelArg(k_gemv,4,sizeof(int),&M); clSetKernelArg(k_gemv,5,sizeof(int),&R);
        std::vector<double> t;
        for(int it=0;it<iters;++it){ CL_CHECK(clFinish(q)); double t0=now_ms();
            CL_CHECK(clEnqueueNDRangeKernel(q,k_gemv,1,0,&gws,&lws,0,0,0)); CL_CHECK(clFinish(q)); t.push_back(now_ms()-t0); }
        return median(t)*1000.0; // us
    };
    double full=bench(M);
    printf("  full-width R=M=%d : %.2f us\n", M, full);
    for(int R : {32,64,96,128,192,256}){ if(R>=M||R>last_ok&&last_ok>0) continue;
        double c=bench(R); printf("  capped   R=%4d  : %.2f us  (%.0f%% of full-width%s)\n", R, c, 100.0*full/c, c<=full*1.15?" -- OK":""); }
    return 0;
}

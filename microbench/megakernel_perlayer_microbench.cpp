// Megakernel prototype for Gemma-4 per-layer-embedding block (decode, n_tokens=1).
//
// Real op chain (from GGML_OPENCL_DUMP_GRAPH, layer L, nodes DG112..120):
//   1  MUL_MAT  inp_gate : y1[n_pl]  = Wg[n_pl x D]  . x[D]        (all-WG, M=n_pl,  K=D)
//   2  UNARY    gelu      : y2 = gelu(y1)                          (single-WG, n_pl)
//   3  MUL      gate      : y2 = y2 * g[n_pl]                      (single-WG, n_pl)
//   4  MUL_MAT  proj      : y3[D]    = Wp[D x n_pl] . y2[n_pl]     (all-WG, M=D, K=n_pl)
//   5  RMS_NORM           : r = rms_norm(y3)                       (single-WG, D)
//   6  MUL      post_norm : r = r * wn[D]                          (single-WG, D)
//   7  ADD      residual  : r = r + res[D]   (res == pe_in)        (single-WG, D)
//   8  MUL      out_scale : out = r * scale                        (single-WG, D)
//
// The two matmuls are SMALL (one dim is n_pl=256), so the block is dispatch-bound
// at decode (~8 ops x ~5 us launch >> the few-us of matmul BW work). This is the
// megakernel's target regime -- unlike the D x D crossover bench whose matmuls
// dwarfed the barrier. f32 weights here => CONSERVATIVE: real weights are q4_K
// (8x less BW), so the matmul shrinks further and the megakernel wins by more.
//
// Build: microbench/build_megablock_pl.bat   Usage: mb_megapl.exe [D] [n_pl] [R] [iters]

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

static const char* SRC = R"CLC(
#define LWS 128

inline void grid_barrier(volatile global int* counter, int R, int lid) {
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (lid == 0) { atomic_inc(counter); while (atomic_add(counter,0) < R) {} }
    barrier(CLK_GLOBAL_MEM_FENCE);
}

// tanh-approx gelu (matches ggml GGML_UNARY_OP_GELU default).
inline float gelu(float x) {
    const float c = 0.7978845608028654f; // sqrt(2/pi)
    return 0.5f*x*(1.0f + tanh(c*(x + 0.044715f*x*x*x)));
}

// y[M] = W[M*K] . v[K], all WGs grid-stride over rows m.
inline void gemv(global const float* W, global const float* v, global float* y,
                 int K, int M, int R, int wg, int lid, local float* red) {
    for (int m = wg; m < M; m += R) {
        float p = 0.0f;
        global const float* Wr = W + (long)m*K;
        for (int k = lid; k < K; k += LWS) p += Wr[k]*v[k];
        red[lid] = p; barrier(CLK_LOCAL_MEM_FENCE);
        for (int s = LWS/2; s > 0; s >>= 1) { if (lid < s) red[lid]+=red[lid+s]; barrier(CLK_LOCAL_MEM_FENCE); }
        if (lid == 0) y[m] = red[0];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

// single-WG: y2[i] = gelu(y1[i]) * g[i]   over n_pl elems
inline void gelu_mul(global const float* y1, global const float* g, global float* y2, int n, int lid) {
    for (int i = lid; i < n; i += LWS) y2[i] = gelu(y1[i]) * g[i];
}

// single-WG: out[i] = (rms_norm(y3)[i] * wn[i] + res[i]) * scale   over D elems
inline void rms_mul_add_scale(global const float* y3, global const float* wn, global const float* res,
                              global float* out, int D, float eps, float scale, int lid, local float* red) {
    float p = 0.0f;
    for (int i = lid; i < D; i += LWS) p += y3[i]*y3[i];
    red[lid] = p; barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = LWS/2; s > 0; s >>= 1) { if (lid < s) red[lid]+=red[lid+s]; barrier(CLK_LOCAL_MEM_FENCE); }
    float rms = rsqrt(red[0]/D + eps);
    for (int i = lid; i < D; i += LWS) out[i] = (y3[i]*rms*wn[i] + res[i]) * scale;
}

// Megakernel: whole block in 1 launch, 3 software grid-barriers.
// scratch: y1[n_pl], y2[n_pl], y3[D]; counters[3] pre-zeroed.
kernel void mega_pl(global const float* Wg, global const float* x, global const float* g,
                    global const float* Wp, global const float* wn, global const float* res,
                    global float* out, global float* y1, global float* y2, global float* y3,
                    volatile global int* counters, int D, int n_pl, int R, float eps, float scale) {
    local float red[LWS];
    int wg = get_group_id(0), lid = get_local_id(0);
    gemv(Wg, x, y1, D, n_pl, R, wg, lid, red);            // stage 1: all-WG
    grid_barrier(counters + 0, R, lid);
    if (wg == 0) gelu_mul(y1, g, y2, n_pl, lid);          // stage 2/3: single-WG
    grid_barrier(counters + 1, R, lid);
    gemv(Wp, y2, y3, n_pl, D, R, wg, lid, red);           // stage 4: all-WG
    grid_barrier(counters + 2, R, lid);
    if (wg == 0) rms_mul_add_scale(y3, wn, res, out, D, eps, scale, lid, red); // stage 5-8: single-WG
}

// ---- dispatch-per-op baseline kernels ----
kernel void k_gemv(global const float* W, global const float* v, global float* y, int K, int M, int R) {
    local float red[LWS];
    gemv(W, v, y, K, M, R, get_group_id(0), get_local_id(0), red);
}
kernel void k_gelu(global const float* y1, global float* o, int n) {
    int i = get_global_id(0); if (i < n) o[i] = gelu(y1[i]);
}
kernel void k_mul(global const float* a, global const float* b, global float* o, int n) {
    int i = get_global_id(0); if (i < n) o[i] = a[i]*b[i];
}
kernel void k_add(global const float* a, global const float* b, global float* o, int n) {
    int i = get_global_id(0); if (i < n) o[i] = a[i]+b[i];
}
kernel void k_scale(global const float* a, global float* o, int n, float s) {
    int i = get_global_id(0); if (i < n) o[i] = a[i]*s;
}
kernel void k_rms(global const float* x, global float* o, int D, float eps) {
    local float red[LWS];
    int lid = get_local_id(0);
    if (get_group_id(0) != 0) return;
    float p = 0.0f;
    for (int i = lid; i < D; i += LWS) p += x[i]*x[i];
    red[lid] = p; barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = LWS/2; s > 0; s >>= 1) { if (lid < s) red[lid]+=red[lid+s]; barrier(CLK_LOCAL_MEM_FENCE); }
    float rms = rsqrt(red[0]/D + eps);
    for (int i = lid; i < D; i += LWS) o[i] = x[i]*rms;
}
)CLC";

int main(int argc, char** argv){
    int D     = argc>1 ? atoi(argv[1]) : 2560;
    int n_pl  = argc>2 ? atoi(argv[2]) : 256;
    int R     = argc>3 ? atoi(argv[3]) : 64;
    int iters = argc>4 ? atoi(argv[4]) : 200;
    const int LWS = 128;
    const float eps = 1e-6f, scale = 1.0625f;

    cl_platform_id plat; CL_CHECK(clGetPlatformIDs(1,&plat,0));
    cl_device_id dev; CL_CHECK(clGetDeviceIDs(plat,CL_DEVICE_TYPE_GPU,1,&dev,0));
    char name[256]={0}; clGetDeviceInfo(dev,CL_DEVICE_NAME,sizeof(name),name,0);
    printf("device: %s | D=%d n_pl=%d R=%d LWS=%d iters=%d\n", name, D, n_pl, R, LWS, iters);

    cl_int e;
    cl_context ctx=clCreateContext(0,1,&dev,0,0,&e); CL_CHECK(e);
    cl_command_queue q=clCreateCommandQueue(ctx,dev,0,&e); CL_CHECK(e);
    cl_program prog=build(ctx,dev,SRC,"");
    cl_kernel k_mega =clCreateKernel(prog,"mega_pl",&e); CL_CHECK(e);
    cl_kernel k_gemv =clCreateKernel(prog,"k_gemv",&e); CL_CHECK(e);
    cl_kernel k_gelu =clCreateKernel(prog,"k_gelu",&e); CL_CHECK(e);
    cl_kernel k_mul  =clCreateKernel(prog,"k_mul", &e); CL_CHECK(e);
    cl_kernel k_add  =clCreateKernel(prog,"k_add", &e); CL_CHECK(e);
    cl_kernel k_scale=clCreateKernel(prog,"k_scale",&e); CL_CHECK(e);
    cl_kernel k_rms  =clCreateKernel(prog,"k_rms", &e); CL_CHECK(e);

    auto rnd = [](size_t i){ return (((i*1103515245u+12345u)>>16) & 0xff)/255.0f - 0.5f; };
    std::vector<float> hWg((size_t)n_pl*D), hWp((size_t)D*n_pl), hx(D), hg(n_pl), hwn(D), hres(D);
    for(size_t i=0;i<hWg.size();++i) hWg[i]=rnd(i)*0.04f;
    for(size_t i=0;i<hWp.size();++i) hWp[i]=rnd(i+7)*0.04f;
    for(int i=0;i<D;++i){ hx[i]=rnd(i+1)*0.5f; hwn[i]=1.0f+rnd(i+3)*0.1f; hres[i]=rnd(i+5)*0.5f; }
    for(int i=0;i<n_pl;++i) hg[i]=rnd(i+9);

    auto mk=[&](size_t bytes, const void* host, cl_mem_flags fl){
        cl_mem m=clCreateBuffer(ctx, fl|(host?CL_MEM_COPY_HOST_PTR:0), bytes, (void*)host, &e); CL_CHECK(e); return m; };
    cl_mem Wg=mk(hWg.size()*4,hWg.data(),CL_MEM_READ_ONLY), Wp=mk(hWp.size()*4,hWp.data(),CL_MEM_READ_ONLY);
    cl_mem x=mk(D*4,hx.data(),CL_MEM_READ_ONLY), g=mk(n_pl*4,hg.data(),CL_MEM_READ_ONLY);
    cl_mem wn=mk(D*4,hwn.data(),CL_MEM_READ_ONLY), res=mk(D*4,hres.data(),CL_MEM_READ_ONLY);
    cl_mem out=mk(D*4,0,CL_MEM_WRITE_ONLY);
    cl_mem y1=mk(n_pl*4,0,CL_MEM_READ_WRITE), y2=mk(n_pl*4,0,CL_MEM_READ_WRITE), y3=mk(D*4,0,CL_MEM_READ_WRITE);
    cl_mem counters=mk(3*4,0,CL_MEM_READ_WRITE);
    int zero=0; size_t gws=(size_t)R*LWS, lws=LWS;

    auto set_mega=[&](){
        int a=0; CL_CHECK(clSetKernelArg(k_mega,a++,sizeof(cl_mem),&Wg)); CL_CHECK(clSetKernelArg(k_mega,a++,sizeof(cl_mem),&x));
        CL_CHECK(clSetKernelArg(k_mega,a++,sizeof(cl_mem),&g)); CL_CHECK(clSetKernelArg(k_mega,a++,sizeof(cl_mem),&Wp));
        CL_CHECK(clSetKernelArg(k_mega,a++,sizeof(cl_mem),&wn)); CL_CHECK(clSetKernelArg(k_mega,a++,sizeof(cl_mem),&res));
        CL_CHECK(clSetKernelArg(k_mega,a++,sizeof(cl_mem),&out)); CL_CHECK(clSetKernelArg(k_mega,a++,sizeof(cl_mem),&y1));
        CL_CHECK(clSetKernelArg(k_mega,a++,sizeof(cl_mem),&y2)); CL_CHECK(clSetKernelArg(k_mega,a++,sizeof(cl_mem),&y3));
        CL_CHECK(clSetKernelArg(k_mega,a++,sizeof(cl_mem),&counters));
        CL_CHECK(clSetKernelArg(k_mega,a++,sizeof(int),&D)); CL_CHECK(clSetKernelArg(k_mega,a++,sizeof(int),&n_pl));
        CL_CHECK(clSetKernelArg(k_mega,a++,sizeof(int),&R)); CL_CHECK(clSetKernelArg(k_mega,a++,sizeof(float),&eps));
        CL_CHECK(clSetKernelArg(k_mega,a++,sizeof(float),&scale));
    };
    auto run_baseline=[&](){
        // 1 gemv inp_gate
        clSetKernelArg(k_gemv,0,sizeof(cl_mem),&Wg); clSetKernelArg(k_gemv,1,sizeof(cl_mem),&x); clSetKernelArg(k_gemv,2,sizeof(cl_mem),&y1);
        clSetKernelArg(k_gemv,3,sizeof(int),&D); clSetKernelArg(k_gemv,4,sizeof(int),&n_pl); clSetKernelArg(k_gemv,5,sizeof(int),&R);
        CL_CHECK(clEnqueueNDRangeKernel(q,k_gemv,1,0,&gws,&lws,0,0,0));
        // 2 gelu y1->y3(tmp), 3 mul y3*g->y2
        size_t gpl=((n_pl+LWS-1)/LWS)*LWS;
        clSetKernelArg(k_gelu,0,sizeof(cl_mem),&y1); clSetKernelArg(k_gelu,1,sizeof(cl_mem),&y3); clSetKernelArg(k_gelu,2,sizeof(int),&n_pl);
        CL_CHECK(clEnqueueNDRangeKernel(q,k_gelu,1,0,&gpl,&lws,0,0,0));
        clSetKernelArg(k_mul,0,sizeof(cl_mem),&y3); clSetKernelArg(k_mul,1,sizeof(cl_mem),&g); clSetKernelArg(k_mul,2,sizeof(cl_mem),&y2); clSetKernelArg(k_mul,3,sizeof(int),&n_pl);
        CL_CHECK(clEnqueueNDRangeKernel(q,k_mul,1,0,&gpl,&lws,0,0,0));
        // 4 gemv proj
        clSetKernelArg(k_gemv,0,sizeof(cl_mem),&Wp); clSetKernelArg(k_gemv,1,sizeof(cl_mem),&y2); clSetKernelArg(k_gemv,2,sizeof(cl_mem),&y3);
        clSetKernelArg(k_gemv,3,sizeof(int),&n_pl); clSetKernelArg(k_gemv,4,sizeof(int),&D); clSetKernelArg(k_gemv,5,sizeof(int),&R);
        CL_CHECK(clEnqueueNDRangeKernel(q,k_gemv,1,0,&gws,&lws,0,0,0));
        // 5 rms y3->out, 6 mul out*wn->out, 7 add out+res->out, 8 scale
        size_t gD=((D+LWS-1)/LWS)*LWS;
        clSetKernelArg(k_rms,0,sizeof(cl_mem),&y3); clSetKernelArg(k_rms,1,sizeof(cl_mem),&out); clSetKernelArg(k_rms,2,sizeof(int),&D); clSetKernelArg(k_rms,3,sizeof(float),&eps);
        CL_CHECK(clEnqueueNDRangeKernel(q,k_rms,1,0,&gws,&lws,0,0,0));
        clSetKernelArg(k_mul,0,sizeof(cl_mem),&out); clSetKernelArg(k_mul,1,sizeof(cl_mem),&wn); clSetKernelArg(k_mul,2,sizeof(cl_mem),&out); clSetKernelArg(k_mul,3,sizeof(int),&D);
        CL_CHECK(clEnqueueNDRangeKernel(q,k_mul,1,0,&gD,&lws,0,0,0));
        clSetKernelArg(k_add,0,sizeof(cl_mem),&out); clSetKernelArg(k_add,1,sizeof(cl_mem),&res); clSetKernelArg(k_add,2,sizeof(cl_mem),&out); clSetKernelArg(k_add,3,sizeof(int),&D);
        CL_CHECK(clEnqueueNDRangeKernel(q,k_add,1,0,&gD,&lws,0,0,0));
        clSetKernelArg(k_scale,0,sizeof(cl_mem),&out); clSetKernelArg(k_scale,1,sizeof(cl_mem),&out); clSetKernelArg(k_scale,2,sizeof(int),&D); clSetKernelArg(k_scale,3,sizeof(float),&scale);
        CL_CHECK(clEnqueueNDRangeKernel(q,k_scale,1,0,&gD,&lws,0,0,0));
    };

    // ---- correctness: megakernel vs CPU ref ----
    set_mega();
    CL_CHECK(clEnqueueFillBuffer(q,counters,&zero,sizeof(int),0,3*sizeof(int),0,0,0));
    CL_CHECK(clFinish(q));
    CL_CHECK(clEnqueueNDRangeKernel(q,k_mega,1,0,&gws,&lws,0,0,0));
    std::vector<float> gpu(D); CL_CHECK(clEnqueueReadBuffer(q,out,CL_TRUE,0,D*4,gpu.data(),0,0,0));
    std::vector<float> cy1(n_pl), cy2(n_pl), cy3(D), cout(D);
    auto cgelu=[&](float v){ float c=0.7978845608028654f; return 0.5f*v*(1.0f+std::tanh(c*(v+0.044715f*v*v*v))); };
    for(int m=0;m<n_pl;++m){ double p=0; for(int k=0;k<D;++k) p+=(double)hWg[(size_t)m*D+k]*hx[k]; cy1[m]=(float)p; }
    for(int i=0;i<n_pl;++i) cy2[i]=cgelu(cy1[i])*hg[i];
    for(int m=0;m<D;++m){ double p=0; for(int k=0;k<n_pl;++k) p+=(double)hWp[(size_t)m*n_pl+k]*cy2[k]; cy3[m]=(float)p; }
    double ss=0; for(int i=0;i<D;++i) ss+=(double)cy3[i]*cy3[i];
    float rms=1.0f/std::sqrt((float)(ss/D)+eps);
    for(int i=0;i<D;++i) cout[i]=(cy3[i]*rms*hwn[i]+hres[i])*scale;
    double num=0,den=0; for(int i=0;i<D;++i){ double d=(double)gpu[i]-cout[i]; num+=d*d; den+=(double)cout[i]*cout[i]; }
    double l2rel=std::sqrt(num/(den+1e-12));
    printf("correctness: L2 rel err vs CPU ref = %.3e  (%s)\n", l2rel, l2rel<1e-3?"OK":"WRONG");

    // ---- perf ----
    std::vector<double> tm, td;
    for(int it=0; it<iters; ++it){
        CL_CHECK(clEnqueueFillBuffer(q,counters,&zero,sizeof(int),0,3*sizeof(int),0,0,0));
        CL_CHECK(clFinish(q));
        double t0=now_ms();
        CL_CHECK(clEnqueueNDRangeKernel(q,k_mega,1,0,&gws,&lws,0,0,0));
        CL_CHECK(clFinish(q));
        tm.push_back(now_ms()-t0);
    }
    for(int it=0; it<iters; ++it){
        CL_CHECK(clFinish(q));
        double t0=now_ms();
        run_baseline();
        CL_CHECK(clFinish(q));
        td.push_back(now_ms()-t0);
    }
    double mm=median(tm), md=median(td);
    printf("megakernel (1 launch, 3 sw-barriers): %.4f ms\n", mm);
    printf("8 separate dispatches               : %.4f ms\n", md);
    printf("=> megakernel %s by %.1f%%  (saved %.3f us/block)\n",
           mm<md?"FASTER":"slower", 100.0*(md-mm)/md, (md-mm)*1000.0);
    printf("   per-block dispatch overhead implied: ~%.2f us/dispatch (baseline/8)\n", md*1000.0/8.0);
    return 0;
}

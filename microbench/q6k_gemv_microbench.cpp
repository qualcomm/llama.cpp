// Standalone microbenchmark to settle whether the q6_K decode GEMV is ALU-bound
// or memory-bound on Adreno X2, vs the q4_K GEMV (which the wide-K-split win
// showed is latency/memory-bound).
//
// For each kernel it builds two variants from the REAL .cl source:
//   FULL    - the production kernel.
//   NO_ALU  - the dequant macro invocations replaced by a trivial accumulate
//             that still CONSUMES every weight register (so all image loads
//             stay live -> identical memory traffic) but drops ~30x of the
//             per-weight dequant ALU.
// If FULL time >> NO_ALU time -> ALU-bound. If FULL ~= NO_ALU -> memory-bound.
//
// It also queries CL_KERNEL_PRIVATE_MEM_SIZE (register spill) and sweeps the
// K-split (subgroups/WG) so we can see the occupancy/latency response directly.
//
// Build (arm64 VS dev shell):
//   clang-cl /O2 /EHsc q6k_gemv_microbench.cpp ^
//     /I C:/llama-shared/skills/opencl-adreno/raw/opencl-sdk/inc ^
//     /link C:/llama-shared/skills/opencl-adreno/raw/opencl-sdk/libs/Windows/OpenCL.lib
//
// Usage: q6k_gemv_microbench.exe [K M iters]
//   defaults run the E4B q6_K ffn_down and q4_K ffn_gate shapes.

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

#define CL_CHECK(expr) do { cl_int _e = (expr); if (_e != CL_SUCCESS) { \
    fprintf(stderr, "CL error %d at %s:%d (%s)\n", _e, __FILE__, __LINE__, #expr); exit(1); } } while(0)

static std::string read_file(const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    std::stringstream ss; ss << f.rdbuf(); return ss.str();
}

static std::string replace_one(std::string s, const std::string& from, const std::string& to) {
    size_t p = s.find(from);
    if (p == std::string::npos) { fprintf(stderr, "PATTERN NOT FOUND: %.60s...\n", from.c_str()); exit(1); }
    return s.replace(p, from.size(), to);
}

static cl_program build(cl_context ctx, cl_device_id dev, const std::string& src, const std::string& opts) {
    const char* s = src.c_str();
    cl_int err;
    cl_program p = clCreateProgramWithSource(ctx, 1, &s, nullptr, &err); CL_CHECK(err);
    err = clBuildProgram(p, 1, &dev, opts.c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logsz = 0; clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logsz);
        std::vector<char> log(logsz + 1, 0);
        clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, logsz, log.data(), nullptr);
        fprintf(stderr, "BUILD FAILED:\n%s\n", log.data()); exit(1);
    }
    return p;
}

struct Buf { cl_mem mem; };

static cl_mem mkbuf(cl_context ctx, size_t bytes) {
    cl_int err; cl_mem m = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, nullptr, &err); CL_CHECK(err); return m;
}
static cl_mem mkimg(cl_context ctx, cl_mem backing, cl_channel_order order, cl_channel_type type, size_t width) {
    cl_image_format fmt; fmt.image_channel_order = order; fmt.image_channel_data_type = type;
    cl_image_desc d; memset(&d, 0, sizeof(d));
    d.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER; d.image_width = width; d.buffer = backing;
    cl_int err; cl_mem img = clCreateImage(ctx, CL_MEM_READ_ONLY, &fmt, &d, nullptr, &err); CL_CHECK(err); return img;
}

static void print_kinfo(const char* tag, cl_kernel k, cl_device_id dev) {
    cl_ulong priv = 0, local = 0; size_t wg = 0, mult = 0;
    clGetKernelWorkGroupInfo(k, dev, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(priv), &priv, nullptr);
    clGetKernelWorkGroupInfo(k, dev, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(local), &local, nullptr);
    clGetKernelWorkGroupInfo(k, dev, CL_KERNEL_WORK_GROUP_SIZE, sizeof(wg), &wg, nullptr);
    clGetKernelWorkGroupInfo(k, dev, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(mult), &mult, nullptr);
    printf("  %-18s private=%llu B  local=%llu B  wg_cap=%zu  pref_mult=%zu\n",
           tag, (unsigned long long)priv, (unsigned long long)local, wg, mult);
}

static double median(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    return v[v.size()/2];
}

// time a kernel over `iters` launches at a given K-split (subgroups/WG)
static double time_kernel(cl_command_queue q, cl_kernel k, int M, int nsg, int iters) {
    size_t gx = ((size_t)((M/2 + 63)/64))*64;
    size_t local[3]  = {64, (size_t)nsg, 1};
    size_t global[3] = {gx, (size_t)nsg, 1};
    // warmup (X2 cold-clock ramp)
    for (int i = 0; i < 40; i++) CL_CHECK(clEnqueueNDRangeKernel(q, k, 3, nullptr, global, local, 0, nullptr, nullptr));
    CL_CHECK(clFinish(q));
    std::vector<double> ts;
    for (int i = 0; i < iters; i++) {
        cl_event e;
        CL_CHECK(clEnqueueNDRangeKernel(q, k, 3, nullptr, global, local, 0, nullptr, &e));
        CL_CHECK(clWaitForEvents(1, &e));
        cl_ulong s0 = 0, e0 = 0;
        clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_START, sizeof(s0), &s0, nullptr);
        clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_END,   sizeof(e0), &e0, nullptr);
        ts.push_back((e0 - s0) * 1e-6); // ms
        clReleaseEvent(e);
    }
    return median(ts);
}

int main(int argc, char** argv) {
    const char* kdir = "d:/work/llm/llama.cpp/ggml/src/ggml-opencl/kernels/";
    int iters = 200;
    // shape overrides
    int argK = (argc > 1) ? atoi(argv[1]) : 0;
    int argM = (argc > 2) ? atoi(argv[2]) : 0;
    if (argc > 3) iters = atoi(argv[3]);

    // --- platform / device (first Adreno/QUALCOMM GPU) ---
    cl_uint nplat = 0; clGetPlatformIDs(0, nullptr, &nplat);
    std::vector<cl_platform_id> plats(nplat); clGetPlatformIDs(nplat, plats.data(), nullptr);
    cl_device_id dev = nullptr;
    for (auto p : plats) {
        cl_uint nd = 0; if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &nd) != CL_SUCCESS || nd == 0) continue;
        std::vector<cl_device_id> ds(nd); clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, nd, ds.data(), nullptr);
        dev = ds[0]; break;
    }
    if (!dev) { fprintf(stderr, "no GPU device\n"); return 1; }
    char dname[256] = {0}; clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(dname), dname, nullptr);
    cl_uint cu = 0; clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, nullptr);
    size_t maxwg = 0; clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxwg), &maxwg, nullptr);
    printf("device: %s  (max_compute_units=%u  max_wg=%zu)\n", dname, cu, maxwg);

    cl_int err;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err); CL_CHECK(err);
    cl_command_queue q = clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &err); CL_CHECK(err);

    std::string opts = "-cl-std=CL3.0 -cl-mad-enable -DVECTOR_SUB_GROUP_BROADCAT ";

    // ============================ q6_K ============================
    {
        std::string src = read_file((std::string(kdir) + "gemv_noshuffle_q6_k_f32.cl").c_str());
        // NO_ALU variant: replace the two active dequant macro invocations.
        std::string hi = "        dequantize_block_acc_bcast_8_hi(total_sum, as_ushort8(reg_a_l), as_uchar8(reg_a_h), reg_d, reg_s, reg_b);";
        std::string lo = "        dequantize_block_acc_bcast_8_lo(total_sum, as_ushort8(reg_a_l), as_uchar8(reg_a_h), reg_d, reg_s, reg_b);";
        std::string cheap =
          "        total_sum.s0 += (float)(reg_a_l.s0+reg_a_l.s1+reg_a_l.s2+reg_a_l.s3"
          "+reg_a_h.s0+reg_a_h.s1+reg_a_h.s2+reg_a_h.s3)*reg_b.s0*(float)reg_d.s0*(float)reg_s.s0;";
        std::string noalu = replace_one(replace_one(src, hi, cheap), lo, cheap);

        cl_program pf = build(ctx, dev, src,   opts);
        cl_program pn = build(ctx, dev, noalu, opts);
        cl_kernel kf, kn;
        CL_CHECK((kf = clCreateKernel(pf, "kernel_gemv_noshuffle_q6_K_f32", &err), err));
        CL_CHECK((kn = clCreateKernel(pn, "kernel_gemv_noshuffle_q6_K_f32", &err), err));

        printf("\n=== q6_K GEMV kernel info ===\n");
        print_kinfo("FULL",   kf, dev);
        print_kinfo("NO_ALU", kn, dev);

        struct Shape { int K, M; const char* name; };
        std::vector<Shape> shapes = { {10240, 2560, "ffn_down"}, {2560, 512, "attn_v"} };
        if (argK && argM) shapes = { {argK, argM, "custom"} };

        for (auto sh : shapes) {
            int K = sh.K, M = sh.M;
            double wbytes = 0.75 * (double)K * (double)M; // ql (K*M/2) + qh (K*M/4)
            // buffers (generous slack)
            cl_mem ql = mkbuf(ctx, (size_t)K*M/2 + 4096);
            cl_mem qh = mkbuf(ctx, (size_t)K*M/4 + 4096);
            cl_mem sb = mkbuf(ctx, (size_t)(K/32)*(M/2)*4 + 65536);
            cl_mem db = mkbuf(ctx, (size_t)(K/256+1)*(M/2)*4 + 65536);
            cl_mem bb = mkbuf(ctx, (size_t)K*4 + 4096);
            cl_mem dst= mkbuf(ctx, (size_t)M*4 + 4096);
            cl_mem qli = mkimg(ctx, ql, CL_R, CL_FLOAT,      (size_t)M*K/8);
            cl_mem qhi = mkimg(ctx, qh, CL_R, CL_HALF_FLOAT, (size_t)M*K/8);
            cl_mem bbi = mkimg(ctx, bb, CL_RGBA, CL_FLOAT,   (size_t)K/4);
            cl_ulong offsetd = 0;
            for (cl_kernel k : {kf, kn}) {
                CL_CHECK(clSetKernelArg(k, 0, sizeof(cl_mem), &qli));
                CL_CHECK(clSetKernelArg(k, 1, sizeof(cl_mem), &qhi));
                CL_CHECK(clSetKernelArg(k, 2, sizeof(cl_mem), &sb));
                CL_CHECK(clSetKernelArg(k, 3, sizeof(cl_mem), &db));
                CL_CHECK(clSetKernelArg(k, 4, sizeof(cl_mem), &bbi));
                CL_CHECK(clSetKernelArg(k, 5, sizeof(cl_mem), &dst));
                CL_CHECK(clSetKernelArg(k, 6, sizeof(cl_ulong), &offsetd));
                CL_CHECK(clSetKernelArg(k, 7, sizeof(cl_int), &K));
                CL_CHECK(clSetKernelArg(k, 8, sizeof(cl_int), &M));
            }
            printf("\n--- q6_K %s  K=%d M=%d  (weight read %.2f MB/call) ---\n", sh.name, K, M, wbytes/1e6);
            printf("  nsg |   FULL ms  GB/s |  NO_ALU ms  GB/s | ALU share\n");
            for (int nsg : {4, 8, 16}) {
                double tf = time_kernel(q, kf, M, nsg, iters);
                double tn = time_kernel(q, kn, M, nsg, iters);
                printf("  %3d | %8.4f %5.1f | %9.4f %5.1f | %5.1f%%\n",
                       nsg, tf, wbytes/tf/1e6, tn, wbytes/tn/1e6, 100.0*(tf-tn)/tf);
            }
            clReleaseMemObject(qli); clReleaseMemObject(qhi); clReleaseMemObject(bbi);
            clReleaseMemObject(ql); clReleaseMemObject(qh); clReleaseMemObject(sb);
            clReleaseMemObject(db); clReleaseMemObject(bb); clReleaseMemObject(dst);
        }
        clReleaseKernel(kf); clReleaseKernel(kn); clReleaseProgram(pf); clReleaseProgram(pn);
    }

    // ============================ q4_K ============================
    {
        std::string src = read_file((std::string(kdir) + "gemv_noshuffle_q4_k_f32.cl").c_str());
        std::string hi = "        dequantizeBlockAccum_ns_sgbroadcast_8_hi(totalSum, as_ushort8(regA), regS, regM, regB);";
        std::string lo = "        dequantizeBlockAccum_ns_sgbroadcast_8_lo(totalSum, as_ushort8(regA), regS, regM, regB);";
        std::string cheap =
          "        totalSum.s0 += (float)(regA.s0+regA.s1+regA.s2+regA.s3)*regB.s0*(float)regS.s0*(float)regM.s0;";
        std::string noalu = replace_one(replace_one(src, hi, cheap), lo, cheap);

        cl_program pf = build(ctx, dev, src,   opts);
        cl_program pn = build(ctx, dev, noalu, opts);
        cl_kernel kf, kn;
        CL_CHECK((kf = clCreateKernel(pf, "kernel_gemv_noshuffle_q4_k_f32", &err), err));
        CL_CHECK((kn = clCreateKernel(pn, "kernel_gemv_noshuffle_q4_k_f32", &err), err));

        printf("\n=== q4_K GEMV kernel info ===\n");
        print_kinfo("FULL",   kf, dev);
        print_kinfo("NO_ALU", kn, dev);

        struct Shape { int K, M; const char* name; };
        std::vector<Shape> shapes = { {2560, 10240, "ffn_gate"}, {2560, 512, "Kcur"} };

        cl_uchar mask_d6 = 0x3F, mask_d4 = 0x0F, mask_hi2 = 0xC0;
        for (auto sh : shapes) {
            int K = sh.K, M = sh.M;
            double wbytes = 0.5 * (double)K * (double)M; // q only (K*M/2)
            cl_mem qb = mkbuf(ctx, (size_t)M*K/2 + 4096);
            cl_mem db = mkbuf(ctx, (size_t)(K/256+1)*(M/2)*4 + 65536);
            cl_mem dmb= mkbuf(ctx, (size_t)(K/256+1)*(M/2)*4 + 65536);
            cl_mem sbf= mkbuf(ctx, (size_t)M*(K/256+1)*12 + 65536);
            cl_mem bb = mkbuf(ctx, (size_t)K*4 + 4096);
            cl_mem dst= mkbuf(ctx, (size_t)M*4 + 4096);
            cl_mem qi = mkimg(ctx, qb, CL_R, CL_UNSIGNED_INT32, (size_t)M*K/2/4);
            cl_mem bbi= mkimg(ctx, bb, CL_RGBA, CL_FLOAT,       (size_t)K/4);
            cl_ulong offsetd = 0;
            for (cl_kernel k : {kf, kn}) {
                CL_CHECK(clSetKernelArg(k, 0, sizeof(cl_mem), &qi));
                CL_CHECK(clSetKernelArg(k, 1, sizeof(cl_mem), &db));
                CL_CHECK(clSetKernelArg(k, 2, sizeof(cl_mem), &dmb));
                CL_CHECK(clSetKernelArg(k, 3, sizeof(cl_mem), &sbf));
                CL_CHECK(clSetKernelArg(k, 4, sizeof(cl_mem), &bbi));
                CL_CHECK(clSetKernelArg(k, 5, sizeof(cl_mem), &dst));
                CL_CHECK(clSetKernelArg(k, 6, sizeof(cl_ulong), &offsetd));
                CL_CHECK(clSetKernelArg(k, 7, sizeof(cl_int), &K));
                CL_CHECK(clSetKernelArg(k, 8, sizeof(cl_int), &M));
                CL_CHECK(clSetKernelArg(k, 9, sizeof(cl_uchar), &mask_d6));
                CL_CHECK(clSetKernelArg(k, 10, sizeof(cl_uchar), &mask_d4));
                CL_CHECK(clSetKernelArg(k, 11, sizeof(cl_uchar), &mask_hi2));
            }
            printf("\n--- q4_K %s  K=%d M=%d  (weight read %.2f MB/call) ---\n", sh.name, K, M, wbytes/1e6);
            printf("  nsg |   FULL ms  GB/s |  NO_ALU ms  GB/s | ALU share\n");
            for (int nsg : {4, 8, 16}) {
                double tf = time_kernel(q, kf, M, nsg, iters);
                double tn = time_kernel(q, kn, M, nsg, iters);
                printf("  %3d | %8.4f %5.1f | %9.4f %5.1f | %5.1f%%\n",
                       nsg, tf, wbytes/tf/1e6, tn, wbytes/tn/1e6, 100.0*(tf-tn)/tf);
            }
            clReleaseMemObject(qi); clReleaseMemObject(bbi);
            clReleaseMemObject(qb); clReleaseMemObject(db); clReleaseMemObject(dmb);
            clReleaseMemObject(sbf); clReleaseMemObject(bb); clReleaseMemObject(dst);
        }
        clReleaseKernel(kf); clReleaseKernel(kn); clReleaseProgram(pf); clReleaseProgram(pn);
    }

    clReleaseCommandQueue(q); clReleaseContext(ctx);
    return 0;
}

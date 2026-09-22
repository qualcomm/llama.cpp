# Fused Allreduce Reduce-Scatter for Tensor-Split Inference

Branch: `dev/ebateni/scatter-on-master`  
Commit: `b279c9a35`  
Device: SA8797 PVM `1b531bdd`, 4-core v81 HTP, 16 MB VTCM/core

---

## Background

In multi-core tensor-split inference, every transformer layer contains a
fused `ALLREDUCE+ADD` collective: each core holds a partial matmul result
and the N cores must sum them together so every core ends up with the full
(replicated) activation.

The naive implementation has each core reduce the **full** tensor:

```
core 0 reads all N partial buffers → reduces → writes full result
core 1 reads all N partial buffers → reduces → writes full result
    ... (N-fold redundant compute)
```

This means N cores do N times the reduction work. The "reduce-scatter"
optimization assigns each core a disjoint 1/N shard of the output:

```
core 0 reduces only elements [0 .. nelem/N)        → fans shard out to all N dst buffers
core 1 reduces only elements [nelem/N .. 2*nelem/N) → fans shard out to all N dst buffers
    ... (1/N compute each, same total reduction, N-fold HVX saving)
```

Each core still ends up with the full replicated result (fan-out via DMA),
but the reduction compute is cut N-fold.

---

## Implementation

### Files changed

| File | Change |
|------|--------|
| `ggml/src/ggml-hexagon/ggml-hexagon.cpp` | Fusion planner: `is_shard_ok` guard, `htp_allreduce_mode` selection |
| `ggml/src/ggml-hexagon/htp/allreduce-ops.c` | HTP kernel: conditional `syncht`, per-phase trace events |
| `ggml/src/ggml-hexagon/htp/allreduce-ops.h` | `htp_allreduce_mode` enum, `mode` field in `htp_allreduce_kernel_params` |
| `ggml/src/ggml-hexagon/htp/htp-ops.h` | `AR_ENTRY_BARRIER`, `AR_REDUCE`, `AR_EXIT_BARRIER` trace event IDs |
| `scripts/snapdragon/run.py` | `--hex-ar-scatter` flag → `GGML_HEXAGON_AR_SCATTER` |

### `htp_allreduce_mode` enum

```c
enum htp_allreduce_mode {
    HTP_ALLREDUCE_FULL           = 0,  // every core reduces the full tensor
    HTP_ALLREDUCE_SHARDED_FANOUT = 1,  // each core reduces its 1/N shard, fans out via DMA
};
```

Carried in `htp_allreduce_kernel_params.mode`. Decided once at fusion time
by the planner (`ggml-hexagon.cpp`), consumed by the HTP kernel and
instrumentation.

### `is_shard_ok` — correctness guard

The shard eligibility check happens **before** `precompute_allreduce_params`
so the shard range (`rank_elem_start` / `rank_nelem`) is correct on both
paths:

```cpp
const bool is_shard_ok = n_ranks <= HTP_OP_MAX_OUTPUTS &&
                         add_dst->data == ar_local->data;   // in-place add only
```

**Why this matters (Qwen corruption fix):** Qwen3.5's hybrid graph contains
non-in-place residual adds (`linear_attn_out` layers, `same=0`). Without this
guard, FULL-fallback ops inherited the 1/N shard range, leaving the rest of
each buffer stale — causing silent corruption on 2c/4c Qwen with scatter
enabled. The `is_shard_ok` check auto-disables scatter for those ops;
standard tensor matmuls remain all-scatter. Qwen 2c/4c passes all quality
checks with scatter=1 after this fix.

### Conditional `syncht`

The pre-exit-fence DMA drain only runs for `SHARDED_FANOUT`; the FULL path,
single-core, and decode paths skip it:

```c
if (mode == HTP_ALLREDUCE_SHARDED_FANOUT) {
    asm volatile ("syncht" : : : "memory");
}
```

### Per-phase trace events

Three new trace event IDs (`AR_ENTRY_BARRIER`, `AR_REDUCE`, `AR_EXIT_BARRIER`)
bracket each phase for profiling with `GGML_HEXAGON_PROFILE=3`.

Profile (llama-3.2-3B, 4c tensor-split, scatter=1, `pp1024 ub1024`):

| Phase | % of collective | avg cyc/op |
|-------|-----------------|------------|
| `AR_ENTRY_BARRIER` (cross-core arrival skew) | **71.2%** | 32,477 |
| `AR_REDUCE` (sharded compute) | 20.8% | 9,490 |
| `AR_EXIT_BARRIER` (fan-out DMA drain) | 8.0% | 3,669 |

The dominant cost is cross-core arrival skew at the entry barrier, not
compute or DMA. This sets the ceiling for further scatter-related
optimizations.

### Env var

```
GGML_HEXAGON_AR_SCATTER=0|1   (default: 1 — enabled)
```

---

## Test Commands

All tests run on device `1b531bdd` via the harness at
`llama.cpp_multinsp/llama.cpp-build-and-test.sh`.  
Package deployed to `/data/local/tmp/pkg-bat`.  
Models at `/data/local/tmp/models/`.  
Context: 4096. Reps: 3. PP depth: 0. TG depth: 1024.

### PP benchmark (llama-bench)

```bash
# PP — prompt processing (pp1024, ubatch=1024)
export GGML_HEXAGON_AR_SCATTER=1
export GGML_HEXAGON_ARCH=81
export GGML_HEXAGON_USE_HMX=1
export GGML_HEXAGON_DEVICES=HTP0,HTP1,HTP2,HTP3
/data/local/tmp/pkg-bat/bin/llama-bench \
  -m /data/local/tmp/models/<model>.gguf \
  --device HTP0/HTP1/HTP2/HTP3 \
  -sm tensor \
  -ngl 99 -p 1024 -n 0 -d 0 -ub 1024 -r 3

# TG — token generation (tg256, KV pre-filled to depth 1024)
/data/local/tmp/pkg-bat/bin/llama-bench \
  -m /data/local/tmp/models/<model>.gguf \
  --device HTP0/HTP1/HTP2/HTP3 \
  -sm tensor \
  -ngl 99 -p 0 -n 256 -d 1024 -r 3
```

Row-split uses `HTP0[0-3]` as the device string and `-sm row`.
Layer-split uses `HTP0,HTP1,...` with `-sm layer`.
Single-core omits `-sm`.

### IQ test (llama-completion, 10 questions, greedy)

```bash
export GGML_HEXAGON_AR_SCATTER=1
export GGML_HEXAGON_ARCH=81
export GGML_HEXAGON_USE_HMX=1
export GGML_HEXAGON_DEVICES=HTP0,HTP1,HTP2,HTP3
/data/local/tmp/pkg-bat/bin/llama-completion \
  -m /data/local/tmp/models/<model>.gguf \
  -dev HTP0,HTP1,HTP2,HTP3 \
  -sm tensor \
  -c 4096 -ngl 99 -no-cnv --temp 0 \
  -sys "You are a helpful assistant. Answer concisely." \
  -p "<question>" -n 8
# Scored: case-insensitive substring match against expected answer
```

### Cookie test (coherence check, greedy, 128 tokens)

```bash
/data/local/tmp/pkg-bat/bin/llama-completion \
  -m /data/local/tmp/models/<model>.gguf \
  -dev HTP0,HTP1,HTP2,HTP3 \
  -sm tensor \
  -c 4096 -ngl 99 -no-cnv --temp 0 \
  -f /data/local/tmp/cookie.txt -n 128
# prompt: "What is the best cookie in the world?"
# scored: GARBAGE if output contains repetitive phrase loops
```

### Needle test (long-context recall, 128 tokens)

```bash
/data/local/tmp/pkg-bat/bin/llama-completion \
  -m /data/local/tmp/models/<model>.gguf \
  -dev HTP0,HTP1,HTP2,HTP3 \
  -sm tensor \
  -c 4096 -ngl 99 -no-cnv \
  -f /data/local/tmp/prompt_1k.txt -n 128
# PASS if output contains the hidden needle string from the prompt
```

---

## Results

Baseline run: `20260921-191207` (`dev/ebateni/scatter-on-master`, `cf5b841d6`).  
Split modes: `single`, `tensor`, `tensor-scatter` (= tensor + `AR_SCATTER=1`), `row`.  
Quality columns: `IQ score · needle · cookie`.  
`*` = ubatch reduced 1024→512 due to upstream VTCM regression (#28589).  
**Bold** = best PP or TG in that row across split methods.

### llama-3.2-3b-instruct-q4_0

| Cores | PP tensor | PP tensor-scatter | PP row | TG tensor | TG tensor-scatter | TG row | Qual tensor | Qual tensor-scatter | Qual row |
|-------|-----------|-------------------|--------|-----------|-------------------|--------|-------------|---------------------|----------|
| 1 (baseline) | 1793.59 (1.00x) | 1793.59 (1.00x) | 1793.59 (1.00x) | 23.09 (1.00x) | 23.09 (1.00x) | 23.09 (1.00x) | 9/10 · PASS · OK | 9/10 · PASS · OK | 9/10 · PASS · OK |
| 2 | 2739.17 (1.53x) | 2922.22 (1.63x) | **3187.84 (1.78x)** | 34.64 (1.50x) | 35.08 (1.52x) | **35.48 (1.54x)** | 9/10 · PASS · OK | 9/10 · PASS · OK | 9/10 · PASS · OK |
| 4 | 3220.73 (1.80x) | **3809.71 (2.12x)** | 3729.32 (2.08x) | 37.71 (1.63x) | 39.24 (1.70x) | **50.09 (2.17x)** | 8/10 · PASS · OK | 8/10 · PASS · OK | 9/10 · PASS · OK |

Scatter gain (tensor → tensor-scatter, 4c): **+18.3% PP**.

### Qwen3.5-4B-Q4

| Cores | PP tensor | PP tensor-scatter | PP row | TG tensor | TG tensor-scatter | TG row | Qual tensor | Qual tensor-scatter | Qual row |
|-------|-----------|-------------------|--------|-----------|-------------------|--------|-------------|---------------------|----------|
| 1 (baseline) | 535.43 (1.00x) | 535.43 (1.00x) | 535.43 (1.00x) | 17.29 (1.00x) | 17.29 (1.00x) | 17.29 (1.00x) | 9/10 · PASS · OK | 9/10 · PASS · OK | 9/10 · PASS · OK |
| 2 | 971.36 (1.81x) | **987.30 (1.84x)** | 681.67 (1.27x) | 25.53 (1.48x) | **25.68 (1.48x)** | 27.01 (1.56x) | 9/10 · PASS · OK | 9/10 · PASS · OK | 9/10 · PASS · OK |
| 4 | 1331.69* (2.49x) | **1382.44* (2.58x)** | 749.72 (1.40x) | 25.99 (1.50x) | **26.36 (1.52x)** | 32.43 (1.88x) | 9/10 · PASS · OK | 9/10 · PASS · OK | 9/10 · PASS · OK |

`*` ubatch=512 (upstream VTCM regression at ub=1024 for Qwen 4c).  
Scatter gain (tensor → tensor-scatter, 4c): **+3.8% PP**.  
Row-split regression at 4c is an upstream issue (#28589), not this patch.

### gemma-4-E4B_q4_0-it

| Cores | PP tensor | PP tensor-scatter | PP row | TG tensor | TG tensor-scatter | TG row | Qual tensor | Qual tensor-scatter | Qual row |
|-------|-----------|-------------------|--------|-----------|-------------------|--------|-------------|---------------------|----------|
| 1 (baseline) | 1201.45 (1.00x) | 1201.45 (1.00x) | 1201.45 (1.00x) | 17.33 (1.00x) | 17.33 (1.00x) | 17.33 (1.00x) | 6/10 · PASS · OK | 6/10 · PASS · OK | 6/10 · PASS · OK |
| 2 | 1678.96 (1.40x) | 1673.09 (1.39x) | **2123.88 (1.77x)** | 23.30 (1.34x) | 23.33 (1.35x) | **26.59 (1.53x)** | 6/10 · PASS · OK | 6/10 · PASS · OK | 6/10 · PASS · OK |
| 4 | not runnable | not runnable | **2861.56 (2.38x)** | not runnable | not runnable | **34.24 (1.98x)** | not runnable | not runnable | 6/10 · PASS · OK |

Gemma 4c tensor/tensor-scatter: not runnable (model too large for 4-core VTCM at ub≥512).  
Scatter gain for gemma: ~0% (GQA-capped tensor-split, few eligible allreduce ops at 4c anyway).

---

## Analysis

- **llama (dense, small)**: scatter gives the full theoretical benefit. 4c PP
  +18.3%, matching the expected N-fold reduction work saving for a pure
  contraction-split graph.
- **Qwen (hybrid SSM+attention)**: mixed-mode graph — 144 FULL-mode ops (non-in-place
  residual adds in `linear_attn_out` layers) + 240 scatter-eligible ops. The FULL
  ops are the bottleneck; scatter helps only the eligible fraction. Result: +3.8%
  at 4c, correct output.
- **gemma (GQA, large)**: GQA caps tensor-split to 2c. At 2c the allreduce cost
  is lower relative to compute; scatter gain is within noise.
- **Profile (llama 4c)**: 71% of collective time is cross-core arrival skew at the
  entry barrier, 21% is reduction compute, 8% is fan-out DMA drain. The scatter
  optimization addresses the 21%. The 71% barrier skew is load-imbalance and is
  not addressable by reduce-scatter tuning — it is the architectural ceiling for
  this parallelization strategy.

---

## Known Limitations / Future Work

- **Qwen row-split regression** (upstream #28589): Qwen 4c row drops from 3.31x
  (`hexagon-mdev`) to 1.40x (upstream). Unrelated to this patch — the upstream
  reimplementation of row-split lost Qwen's scaling. Tracked separately.
- **Upstream VTCM abort** at large ubatch (upstream #28589): `enqueue_allreduce`
  ignores `precompute_allreduce_params`'s false return, causing a silent abort.
  Tracked separately.
- **Megatron-style column/row weight partitioning**: the 71% barrier-wait cost
  can only be meaningfully reduced by cutting the number of collectives per layer
  (Megatron pairing collapses 2 allreduces per MLP/attention block into 1). This
  is a meta-backend / weight-layout change, not a scatter-reduce tweak.

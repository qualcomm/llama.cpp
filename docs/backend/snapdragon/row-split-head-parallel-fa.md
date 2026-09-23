# Head-Parallel Flash Attention for Row-Split Multicore

## Overview

In row-split mode (`-sm row`), all physical CDSP cores are grouped into a single logical device.
Each op runs across all cores in parallel, with each core computing its assigned shard of the
output, separated by a cheap MCW barrier between ops (no allreduce).

Previously, `flash_attn_ext` partitioned work by **Q tokens**: core `i` computed attention for
token rows `[i*total/N, (i+1)*total/N)`, but read the **full KV cache** (all `n_kv_heads` heads)
on every core.

This optimization changes the partitioning to **KV heads**: core `i` computes attention only for
heads `[i*n_kv_heads/N, (i+1)*n_kv_heads/N)`, reading only its 1/N head shard of the KV cache.

## How It Works

- **HMX kernel**: the outer `kv_head` loop is restricted to `[kv_head_min, kv_head_max)` per core.
  DMA pipeline prefetch is also updated to use the per-core head range.
- **HVX kernel**: `qrow_start` and `qrows` are aligned to head boundaries:
  `qrow_start = head_start * neq1`, `qrows = heads_per_core * neq1`.
- **Fallback**: when `n_kv_heads % n_cores != 0` (e.g. Gemma-4 with 2 KV heads on 4 cores),
  the original token-based partitioning is used automatically.
- **Toggle**: `GGML_HEXAGON_FA_HEAD_SPLIT=0` disables the optimization (default: 1).

## Why PP Improves More Than TG

PP (prompt processing) runs large batches — more Q tokens means more work per flash_attn call,
so reducing KV bandwidth per core has a larger relative impact. Models with fewer KV heads per
Q head (lower GQA ratio, e.g. llama-3.2-3B with 8 KV heads / 24 Q heads) benefit most because
each core's KV shard is smaller relative to its Q workload.

TG (token generation) is already compute-light (1 token per step); gains are modest.

## Results

Device: Snapdragon 8 Gen 3 (Nord PVM), 4 cores row-split, ubatch=1024, commit `8b9d8a136`.
All results from device `1b531bdd`. 1c numbers are the average of ab=0 and ab=1 (no effect
expected or observed at 1 core). vs-1c ratios are relative to the baseline 1c value.

### LLaMA-3.2-3B-Instruct Q4_0 · n_kv_heads=8
| config | PP baseline (vs 1c) | PP opt (vs 1c) | TG baseline (vs 1c) | TG opt (vs 1c) |
|--------|---------------------|----------------|---------------------|----------------|
| 1c     | 1784 t/s            | 1784 t/s       | 25.3 t/s            | 25.3 t/s       |
| 2c row | 3263 t/s (1.83×)    | 3346 t/s (1.88×) | 39.6 t/s (1.57×)  | 38.2 t/s (1.51×) |
| 4c row | 3813 t/s (2.14×)    | **5775 t/s (3.24×)** | 55.0 t/s (2.17×) | 55.0 t/s (2.17×) |

### Qwen3-0.6B Q4_0 · n_kv_heads=8
| config | PP baseline (vs 1c) | PP opt (vs 1c) | TG baseline (vs 1c) | TG opt (vs 1c) |
|--------|---------------------|----------------|---------------------|----------------|
| 1c     | 4406 t/s            | 4406 t/s       | 65.9 t/s            | 65.9 t/s       |
| 2c row | 5990 t/s (1.36×)    | 7608 t/s (1.73×) | 79.2 t/s (1.20×)  | 79.3 t/s (1.20×) |
| 4c row | 7269 t/s (1.65×)    | **11706 t/s (2.66×)** | 87.7 t/s (1.33×) | 87.1 t/s (1.32×) |

### Qwen3.5-0.8B Q4_0 · n_kv_heads=4
| config | PP baseline (vs 1c) | PP opt (vs 1c) | TG baseline (vs 1c) | TG opt (vs 1c) |
|--------|---------------------|----------------|---------------------|----------------|
| 1c     | 2923 t/s            | 2923 t/s       | 60.4 t/s            | 60.4 t/s       |
| 2c row | 5056 t/s (1.73×)    | 4963 t/s (1.70×) | 83.2 t/s (1.38×)  | 82.5 t/s (1.37×) |
| 4c row | 7320 t/s (2.50×)    | 7469 t/s (2.55×) | 94.0 t/s (1.56×)  | 98.1 t/s (1.62×) |

### Qwen3.5-4B Q4 · n_kv_heads=8
| config | PP baseline (vs 1c) | PP opt (vs 1c) | TG baseline (vs 1c) | TG opt (vs 1c) |
|--------|---------------------|----------------|---------------------|----------------|
| 1c     | 1020 t/s            | 1020 t/s       | 17.9 t/s            | 17.9 t/s       |
| 2c row | 1870 t/s (1.83×)    | 1853 t/s (1.82×) | 27.7 t/s (1.55×)  | 28.1 t/s (1.57×) |
| 4c row | 2975 t/s (2.92×)    | 3128 t/s (3.07×) | 36.6 t/s (2.04×)  | 38.1 t/s (2.13×) |

### Qwen3-4B-Instruct Q4 · n_kv_heads=8
| config | PP baseline (vs 1c) | PP opt (vs 1c) | TG baseline (vs 1c) | TG opt (vs 1c) |
|--------|---------------------|----------------|---------------------|----------------|
| 1c     | 1305 t/s            | 1305 t/s       | 19.2 t/s            | 19.2 t/s       |
| 2c row | 2044 t/s (1.57×)    | 2455 t/s (1.88×) | 31.2 t/s (1.63×)  | 31.2 t/s (1.63×) |
| 4c row | 2773 t/s (2.12×)    | **4075 t/s (3.12×)** | 39.7 t/s (2.07×) | 40.1 t/s (2.09×) |

### Qwen3-VL-4B-Instruct Q4_0 · n_kv_heads=8
| config | PP baseline (vs 1c) | PP opt (vs 1c) | TG baseline (vs 1c) | TG opt (vs 1c) |
|--------|---------------------|----------------|---------------------|----------------|
| 1c     | 1262 t/s            | 1262 t/s       | 19.1 t/s            | 19.1 t/s       |
| 2c row | 1966 t/s (1.56×)    | 2389 t/s (1.89×) | 29.9 t/s (1.57×)  | 30.0 t/s (1.57×) |
| 4c row | 2733 t/s (2.17×)    | **4062 t/s (3.22×)** | 40.7 t/s (2.13×) | 39.1 t/s (2.05×) |

### Gemma-4-E4B Q4_0-it · n_kv_heads=2
*Note: 4-core head-split falls back to token-based (2 heads not divisible by 4 cores).*
| config | PP baseline (vs 1c) | PP opt (vs 1c) | TG baseline (vs 1c) | TG opt (vs 1c) |
|--------|---------------------|----------------|---------------------|----------------|
| 1c     | 1189 t/s            | 1189 t/s       | 18.0 t/s            | 18.0 t/s       |
| 2c row | 1835 t/s (1.54×)    | 1930 t/s (1.62×) | 28.5 t/s (1.58×)  | 29.5 t/s (1.64×) |
| 4c row | 2554 t/s (2.15×)    | 2554 t/s (2.15×) | 36.2 t/s (2.01×)  | 36.2 t/s (2.01×) |

### Gemma-2-2B Q4_0 · n_kv_heads=8
| config | PP baseline (vs 1c) | PP opt (vs 1c) | TG baseline (vs 1c) | TG opt (vs 1c) |
|--------|---------------------|----------------|---------------------|----------------|
| 1c     | 2329 t/s            | 2329 t/s       | 26.4 t/s            | 26.4 t/s       |
| 2c row | 3639 t/s (1.56×)    | 4054 t/s (1.74×) | 36.8 t/s (1.39×)  | 39.6 t/s (1.50×) |
| 4c row | 4846 t/s (2.08×)    | **6199 t/s (2.66×)** | 43.4 t/s (1.64×) | 49.6 t/s (1.88×) |

### Phi-3.5-mini-Instruct Q4_0 · n_kv_heads=8
| config | PP baseline (vs 1c) | PP opt (vs 1c) | TG baseline (vs 1c) | TG opt (vs 1c) |
|--------|---------------------|----------------|---------------------|----------------|
| 1c     | 1322 t/s            | 1322 t/s       | 19.0 t/s            | 19.0 t/s       |
| 2c row | 2004 t/s (1.52×)    | 2485 t/s (1.88×) | 28.8 t/s (1.52×)  | 32.1 t/s (1.69×) |
| 4c row | 2664 t/s (2.01×)    | **4032 t/s (3.05×)** | 37.3 t/s (1.96×) | 43.5 t/s (2.29×) |

## Summary

| model | n_kv_heads | 4c PP baseline (vs 1c) | 4c PP opt (vs 1c) | 4c TG baseline (vs 1c) | 4c TG opt (vs 1c) |
|-------|-----------|------------------------|-------------------|------------------------|-------------------|
| Qwen3-0.6B | 8 | 1.65× | **2.66×** | 1.33× | 1.32× |
| LLaMA-3.2-3B | 8 | 2.14× | **3.24×** | 2.17× | 2.17× |
| Phi-3.5-mini | 8 | 2.01× | **3.05×** | 1.96× | **2.29×** |
| Qwen3-4B-Instruct | 8 | 2.12× | **3.12×** | 2.07× | 2.09× |
| Qwen3-VL-4B | 8 | 2.17× | **3.22×** | 2.13× | 2.05× |
| Gemma-2-2B | 8 | 2.08× | **2.66×** | 1.64× | **1.88×** |
| Qwen3.5-4B | 8 | 2.92× | 3.07× | 2.04× | 2.13× |
| Qwen3.5-0.8B | 4 | 2.50× | 2.55× | 1.56× | 1.62× |
| Gemma-4-E4B | 2 | 2.15× | 2.15× (fallback) | 2.01× | 2.01× |

Models with 8 KV heads on 4 cores (2 heads/core) benefit most from head-parallel partitioning.
Models with fewer KV heads see smaller gains as GQA limits head-level parallelism.

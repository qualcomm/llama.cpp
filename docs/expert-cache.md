# MoE expert cache

Expert caching keeps a bounded subset of MoE expert weights in a RAM/DDR cache for CPU or HTP inference. Missing weights are loaded from the model's backing files when needed. This can make models practical when keeping all expert weights resident would exceed available system memory.

This is primarily a memory capacity feature. It trades lower resident expert memory requirements for storage reads and cache management overhead. If the model already fits comfortably in memory, normal loading may be faster.

## Usage

Caching is disabled by default. Enable it with `--expert-cache`:

```sh
llama-cli --model model.gguf --expert-cache --fit on -p "Hello"
```

Set environment variables before loading. Boolean options use `0` (off) or `1` (on).

| Variable | Default | Description |
|---|---|---|
| `LLAMA_EXPERT_CACHE_SIZE` | `0` | Expert pool budget in bytes. `0` uses automatic sizing with fitting. |
| `LLAMA_EXPERT_CACHE_THRESHOLD` | `0` | Allowed relative router score drop (%) for cached substitutions. `0` disables threshold substitution. |
| `LLAMA_EXPERT_CACHE_MODE` | `layered` | Per-layer pools (`layered`) or shared pools. `auto` chooses by layout, `global` requires matching layouts, and `global-nonuniform` supports differing layouts. |
| `LLAMA_EXPERT_CACHE_POLICY` | `lru` | Choose least recently used (`lru`) or least frequently used (`lfu`) replacement. |
| `LLAMA_EXPERT_CACHE_MLOCK` | `0` | Request page locking for the CPU expert pool. |
| `LLAMA_EXPERT_CACHE_DDR_COPY` | `0` | In layered mode, copy resident experts between cache ranges instead of rereading storage. |
| `LLAMA_EXPERT_CACHE_READ_PERCENTAGE` | `0` | Read a percentage of eligible decode misses, rounded up. `0` and `100` disable the limit. |
| `LLAMA_EXPERT_CACHE_DECODE_PREFETCH_LOOKAHEAD` | `0` | Initial lookahead distance in decode layers. A positive value enables background prefetch in layered mode. |
| `LLAMA_EXPERT_CACHE_DECODE_PREFETCH_FORCE_CACHE_HIT` | `0` | With lookahead, allow cached substitutes after complete matching prefetch coverage, ignoring the threshold. |
| `LLAMA_EXPERT_CACHE_BATCH_DEPTH` | `4` | Maximum experts per read batch. Minimum `1`. |
| `LLAMA_EXPERT_CACHE_HTP_READ_CHUNK_KB` | `512` | Target HTP read chunk size in KiB for small decode batches. `0` disables chunking. |
| `LLAMA_EXPERT_CACHE_HTP_ASYNC` | `1` | Enable asynchronous HTP cache callbacks where supported. `0` uses synchronous callbacks. |
| `LLAMA_EXPERT_CACHE_STATS` | `0` | Collect cache statistics. |

## Automatic fitting

With `--fit on`, a size of `0` selects the largest valid pool estimated to fit alongside model, execution, loading, and related memory, with configured headroom. Without fitting, zero uses the model minimum. The pool budget does not cap total RAM use, and fitting does not guarantee against OOM.

## Accuracy

Defaults preserve router-selected experts. A positive threshold, read percentages from `1` through `99`, and forced cache hits can substitute experts and change model output. Lookahead alone does not change expert selection. Check output quality before using approximation.

## Statistics

Set `LLAMA_EXPERT_CACHE_STATS=1` to report hits, misses, substitutions, prefetch activity, and expert-read volume and timing. Reported bandwidth is logical expert-cache read bandwidth, not physical storage or flash bandwidth.

## Limitations and tradeoffs

- Smaller caches can increase misses and limit prompt microbatch size.
- Model and backend support varies.
- `--moe-cache-mib` caches copies of host expert weights in device memory and cannot be combined with `--expert-cache`.
- Saving a model with active expert caching is rejected. Reload without caching before saving.

# Performance Tuning

This guide covers runtime knobs, environment variables, and benchmark helpers.

## Runtime Configuration (Config)

Key knobs in `include/granite/config.h`:

- `attention_backend`: `Auto`, `MetalFlash`, `MetalLegacy`, `CoreML`, `MPS`, `CPU`.
- `prefill_chunk_size`: enable chunked prefill to reduce peak memory.
- `kv_cache_max_seq`: maximum context length for KV cache.
- `kv_cache_gpu_max_seq`: GPU KV cap (0 = same as `kv_cache_max_seq`).
- `kv_cache_offload`: allow CPU KV fallback if GPU cache is smaller.
- `max_memory_mb`: global budget hint (0 = auto).
- `enable_profiling` and `log_kernel_selection` for debug visibility.

### Example

```cpp
granite::Config cfg = granite::Config::Balanced();
cfg.prefill_chunk_size = 128;
cfg.kv_cache_max_seq = 4096;
```

## Model-Specific Controls

- `TransformerModel::set_prefill_last_token_only(true)` reduces prefill output
  bandwidth when you only need the last logits.
- `TransformerModel::allocate_paged_kv_cache(max_seq, block_size)` enables
  paged attention for long contexts (Metal).

## Environment Variables

- `GRANITE_LOG_LEVEL=trace|debug|info|warn|error|critical|off`
- `GRANITE_METAL_DEBUG_MARKERS=1` (labels Metal command buffers)
- `GRANITE_VULKAN_PRECOMPILED_DIR=/path/to/spv`
- `GRANITE_VULKAN_SPIRV_CACHE_DIR=/path/to/cache`

Benchmark-only overrides:

- `GRANITE_BACKEND=cpu|metal|vulkan`
- `GRANITE_PREFILL_CHUNK_SIZE=128`
- `GRANITE_KV_CACHE_MAX_SEQ=4096`

## Benchmark Tool

`granite_benchmark` supports extra profiling flags:

```bash
./build/granite_benchmark model.gguf --kernel-timing
./build/granite_benchmark model.gguf --device-profile ios-a16
```

Use `--device-profile` for mobile presets (`ios-a14`, `ios-a16`, `ios-a17`,
`android-mali-g78`, `android-adreno-740`).

## Practical Tips

- Start with `Config::Balanced()` and tune `prefill_chunk_size` to fit memory.
- Keep `kv_cache_max_seq` as low as the app requires.
- Use `GRANITE_METAL_DEBUG_MARKERS=1` when profiling in Xcode.

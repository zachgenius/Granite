# Memory Budgets

Recommended defaults for mobile-class devices. These are conservative targets
meant to keep UI responsive and leave headroom for the OS.

| Device Class | Total RAM | Inference Budget | KV Cache Budget |
|--------------|-----------|------------------|-----------------|
| Low-end (2-3GB) | 2-3 GB | 256 MB | 64 MB |
| Mid-range (4-6GB) | 4-6 GB | 512 MB | 128 MB |
| High-end (8GB+) | 8+ GB | 1 GB | 256 MB |

## How to Apply

- `Config::max_memory_mb` to cap overall inference memory.
- `Config::kv_cache_max_seq` to limit KV cache growth.
- `Config::kv_cache_gpu_max_seq` if GPU memory is tighter than CPU memory.
- `Config::prefill_chunk_size` to reduce prefill intermediates.

Example:

```cpp
auto cfg = granite::Config::Balanced();
cfg.max_memory_mb = 512;
cfg.kv_cache_max_seq = 2048;
cfg.prefill_chunk_size = 128;
```

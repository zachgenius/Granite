# Granite API Reference (High Level)

This is a compact guide to the most commonly used public APIs. For details,
see headers under `include/granite/`.

## Include + Namespace

```cpp
#include <granite/granite.h>
```
All APIs live in the `granite` namespace.

## Core Types

- `Result<T>` / `Error` for error handling without exceptions.
- `DataType`, `BackendType`, `BufferHandle` in `include/granite/types.h`.

## Backends

- `IComputeBackend` is the abstract interface.
- `create_backend(BackendType type)` and `create_default_backend()` create a backend.
- `is_backend_available(BackendType type)` checks availability.

Typical flow:
```cpp
auto backend = granite::create_default_backend();
backend->initialize();
```

## Tensor

- `Tensor::allocate(shape, dtype, backend)` allocates a tensor.
- `Tensor::from_buffer(handle, shape, dtype, backend)` wraps an existing buffer.
- `tensor.data<T>()` / `tensor.unmap()` map/unmap CPU-visible memory.
- View utilities: `view`, `slice`, `transpose`, `squeeze`, `unsqueeze`.
- Copy utilities: `clone`, `copy_from`, `to(backend)`.

## LLM Inference

### TransformerModel

- `TransformerModel::load(path, backend[, config])` loads a GGUF model.
- `forward(...)` runs prefill on a token batch.
- `forward_single(token_id, kv_cache)` runs decode on one token.
- `allocate_gpu_kv_cache(max_seq_len)` (Metal) allocates GPU KV cache.
- `allocate_paged_kv_cache(max_seq_len, block_size)` enables paged attention.
- `memory_stats()` reports model pool usage (Metal builds).

### LLMRunner

- `LLMRunner::load(path[, config])` creates a ready-to-use runner.
- `generate(prompt, config)` blocking generation.
- `generate_streaming(prompt, config, token_callback[, progress_callback])` streams tokens.
- `cancel()` interrupts generation.
- `reset()` clears KV cache for a new conversation.

### Tokenization + Config

- `Tokenizer::from_gguf(gguf)`
- `GenerationConfig` (temperature, top-k, top-p, repetition penalty)
- `Config` (runtime performance, memory, backend selection)

### Speculative Decoding

- `SpeculativeRunner::load(target, draft[, config])`
- `SpeculativeConfig` controls speculation depth and tree mode.

## Graph Execution (Optional)

- `Graph`, `Scheduler`, and `Optimization` provide a compute graph pipeline.
- See `include/granite/graph.h`, `include/granite/scheduler.h`, and
  `include/granite/optimization.h` for details.

## File Formats

- `GGUFFile::load(path)` in `include/granite/gguf.h` loads GGUF models.
- `load_onnx_model(path)` in `include/granite/onnx.h` loads ONNX (optional build).

## Logging

- `init_logging()` or rely on auto-init via `get_logger()`.
- `GRANITE_LOG_LEVEL` environment variable sets the default log level.

## Related Design Docs

- `document/llm-inference-design.md`
- `document/ATTENTION_BACKEND_DESIGN.md`
- `document/SIMDGROUP_FLASH_ATTENTION_DESIGN.md`

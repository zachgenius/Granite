# Granite

Granite is a C++20 on-device inference framework targeting Apple (Metal) and
Android (Vulkan), with a CPU fallback. It focuses on efficient LLM inference
with GGUF model support.

## Features

- Metal and Vulkan backends with CPU fallback
- GGUF model loader with quantization support
- LLM inference, speculative decoding, and paged attention
- Benchmark tooling and profiling hooks

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Optional backend toggles:

```bash
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGRANITE_BUILD_METAL=ON \
  -DGRANITE_BUILD_VULKAN=ON \
  -DGRANITE_BUILD_CPU=ON
```

## Run

```bash
./build/granite_main /path/to/model.gguf "Hello"
```

## Tests

```bash
./build/granite_tests
# or
ctest --test-dir build --output-on-failure
```

## Documentation

- API reference: `document/API_REFERENCE.md`
- iOS integration: `document/INTEGRATION_IOS.md`
- Android integration: `document/INTEGRATION_ANDROID.md`
- Model conversion: `document/MODEL_CONVERSION.md`
- Performance tuning: `document/PERFORMANCE_TUNING.md`
- Memory budgets: `document/MEMORY_BUDGETS.md`

## Benchmarks

```bash
./build/granite_benchmark /path/to/model.gguf --full-logits --seq-lens 128,256,512
```

Common env vars:

- `GRANITE_LOG_LEVEL=trace|debug|info|warn|error|critical|off`
- `GRANITE_METAL_DEBUG_MARKERS=1`
- `GRANITE_VULKAN_PRECOMPILED_DIR=/path/to/spv`
- `GRANITE_VULKAN_SPIRV_CACHE_DIR=/path/to/cache`

## Notes

- Third-party dependencies live in `third_party/` (git submodules).
- For Metal headers: `./scripts/download_metal_cpp.sh`.
- Android build helper: `./scripts/build_android.sh`.

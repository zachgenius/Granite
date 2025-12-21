# iOS Integration

This guide is intentionally lightweight and focuses on the minimal steps to
embed Granite in an iOS app.

## Build Notes

- Granite uses Metal via metal-cpp. Fetch headers once:

```bash
./scripts/download_metal_cpp.sh
```

- Build Granite as a static library with Metal enabled. Example CMake flags:

```bash
cmake -B build-ios \
  -DCMAKE_BUILD_TYPE=Release \
  -DGRANITE_BUILD_METAL=ON \
  -DGRANITE_BUILD_VULKAN=OFF \
  -DGRANITE_BUILD_CPU=ON \
  -DGRANITE_BUILD_TESTS=OFF \
  -DGRANITE_BUILD_EXAMPLES=OFF
cmake --build build-ios --parallel
```

- Link `libgranite.a` into your Xcode target and add `include/` to header search
  paths. Ensure Metal, Foundation, and Accelerate frameworks are linked.

## Minimal Usage (C++)

```cpp
#include <granite/granite.h>

int main() {
    auto config = granite::Config::Balanced();
    auto runner = granite::LLMRunner::load("/path/to/model.gguf", config);
    if (!runner.ok()) {
        return 1;
    }

    granite::GenerationConfig gen;
    gen.max_tokens = 128;

    auto result = (*runner)->generate("Hello from iOS", gen);
    if (!result.ok()) {
        return 1;
    }
    return 0;
}
```

## Recommended Runtime Settings

- Use `Config::Balanced()` for general usage and `Config::Battery()` for UI-first
  apps.
- Adjust `kv_cache_max_seq` to cap memory usage for long contexts.
- Use `prefill_chunk_size` to lower peak memory during prompt processing.

## Streaming Tokens

```cpp
(*runner)->generate_streaming(
    prompt,
    gen,
    [](const std::string& token) {
        // Update UI
        return true;  // return false to stop
    },
    [](const granite::GenerationProgress& progress) {
        // progress.prompt_tokens / progress.generated_tokens
    });
```

## Debugging

- Set `GRANITE_LOG_LEVEL=debug` for verbose logging.
- Set `GRANITE_METAL_DEBUG_MARKERS=1` to label Metal command buffers.

# Android Integration

Granite ships with a minimal Android demo and helper scripts.

## Quick Start

- Use the template app in `android/GraniteDemo`.
- The JNI shim is `src/android/granite_jni.cpp`.

```bash
./scripts/build_android.sh --precompile-shaders
```

That script builds a Vulkan-enabled static library and optionally precompiles
SPIR-V shaders. See `scripts/build_android.sh` for all flags.

## Android Studio Template

Open `android/GraniteDemo` in Android Studio and build/run. The template
extracts SPIR-V assets at runtime and sets:

- `GRANITE_VULKAN_PRECOMPILED_DIR` (from assets)

See `android/README.md` for the template layout.

## Runtime Tips

- Prefer Vulkan on device; keep CPU backend enabled as a fallback.
- If you skip precompiled SPIR-V, set `GRANITE_VULKAN_SPIRV_CACHE_DIR` to a
  writable location so Granite can cache compiled shaders.

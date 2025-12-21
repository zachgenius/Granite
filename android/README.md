# Android Integration (Template)

This folder contains a minimal Gradle app that builds Granite with the NDK and
loads a JNI shim.

Quick start:

1) Precompile SPIR-V shaders (optional but recommended):
```
./scripts/precompile_vulkan_shaders.sh --out-dir=android/GraniteDemo/app/src/main/assets/granite_spv
```

2) Open `android/GraniteDemo` in Android Studio and build/run.

Notes:
- JNI shim: `src/android/granite_jni.cpp`
- Assets loader sets `GRANITE_VULKAN_PRECOMPILED_DIR` at runtime.

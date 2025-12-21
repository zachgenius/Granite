cmake_minimum_required(VERSION 3.20)
project(Granite VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Enable Objective-C++ only on Apple platforms (required for Metal backend)
if(APPLE)
    enable_language(OBJCXX)
endif()

# Options
option(GRANITE_BUILD_METAL "Build Metal backend (Apple only)" ON)
option(GRANITE_BUILD_VULKAN "Build Vulkan backend (Linux/Android/Windows)" OFF)
option(GRANITE_BUILD_CPU "Build CPU backend" ON)
option(GRANITE_BUILD_TESTS "Build tests" ON)
option(GRANITE_BUILD_EXAMPLES "Build examples" ON)
option(GRANITE_ENABLE_ONNX "Enable ONNX model loader (requires Protobuf)" OFF)

# =============================================================================
# Dependencies
# =============================================================================

# GLM (header-only math library)
add_subdirectory(third_party/glm)

# fmt (formatting library - used by spdlog)
add_subdirectory(third_party/fmt)

# spdlog (logging library)
set(SPDLOG_FMT_EXTERNAL ON CACHE BOOL "Use external fmt" FORCE)
add_subdirectory(third_party/spdlog)

# Catch2 (testing framework)
if(GRANITE_BUILD_TESTS)
    add_subdirectory(third_party/catch2)
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third_party/catch2/extras)
endif()

# metal-cpp (header-only, manually downloaded)
add_library(metal-cpp INTERFACE)
target_include_directories(metal-cpp INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/metal-cpp
)

# ONNX (optional, requires protoc + protobuf)
if(GRANITE_ENABLE_ONNX)
    find_package(Protobuf REQUIRED)
    protobuf_generate_cpp(ONNX_PROTO_SRCS ONNX_PROTO_HDRS
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnx/onnx.proto
    )
endif()

# =============================================================================
# Granite Core Library
# =============================================================================

set(GRANITE_CORE_SOURCES
    src/core/tensor.cpp
    src/core/memory_manager.cpp
    src/core/operator_registry.cpp
    src/core/error.cpp
    src/core/backend_factory.cpp
    src/core/config.cpp
)

set(GRANITE_GRAPH_SOURCES
    src/graph/graph.cpp
    src/graph/scheduler.cpp
    src/graph/optimization.cpp
)

set(GRANITE_MODEL_SOURCES
    src/model/gguf.cpp
)

if(GRANITE_ENABLE_ONNX)
    list(APPEND GRANITE_MODEL_SOURCES src/model/onnx.cpp)
endif()

set(GRANITE_LLM_SOURCES
    src/llm/math_ops.cpp
    src/llm/kv_cache.cpp
    src/llm/kv_cache_pool.cpp
    src/llm/paged_kv_cache.cpp
    src/llm/rope.cpp
    src/llm/tokenizer.cpp
    src/llm/model_config.cpp
    src/llm/runner.cpp
    src/llm/speculative_runner.cpp
    src/llm/speculation_tree.cpp
    src/llm/batch_scheduler.cpp
)

# TransformerModel uses Metal - compile as .mm on Apple
if(APPLE AND GRANITE_BUILD_METAL)
    list(APPEND GRANITE_LLM_SOURCES src/llm/transformer.mm)
    set_source_files_properties(src/llm/transformer.mm PROPERTIES
        COMPILE_FLAGS "-fobjc-arc"
    )
else()
    # For non-Metal builds, compile as .cpp (rename or use conditional)
    list(APPEND GRANITE_LLM_SOURCES src/llm/transformer.mm)
endif()

# Placeholder sources (will be implemented)
# For now, create stub files to allow compilation

add_library(granite STATIC
    ${GRANITE_CORE_SOURCES}
    ${GRANITE_GRAPH_SOURCES}
    ${GRANITE_MODEL_SOURCES}
    ${GRANITE_LLM_SOURCES}
    ${ONNX_PROTO_SRCS}
    ${ONNX_PROTO_HDRS}
)

target_include_directories(granite
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(granite
    PUBLIC
        glm::glm
        spdlog::spdlog
    PRIVATE
        fmt::fmt
)

if(GRANITE_ENABLE_ONNX)
    target_include_directories(granite PRIVATE ${Protobuf_INCLUDE_DIRS})
    target_link_libraries(granite PRIVATE ${Protobuf_LIBRARIES})
endif()

target_compile_definitions(granite PUBLIC
    GRANITE_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
    GRANITE_VERSION_MINOR=${PROJECT_VERSION_MINOR}
    GRANITE_VERSION_PATCH=${PROJECT_VERSION_PATCH}
)

if(GRANITE_ENABLE_ONNX)
    target_compile_definitions(granite PUBLIC GRANITE_HAS_ONNX=1)
endif()

# =============================================================================
# Metal Backend (Apple only)
# =============================================================================

if(APPLE AND GRANITE_BUILD_METAL)
    set(GRANITE_METAL_SOURCES
        src/backend/metal/metal_backend.mm
        src/backend/metal/metal_compute.mm
        src/operators/metal/metal_operators.mm
    )

    target_sources(granite PRIVATE ${GRANITE_METAL_SOURCES})

    target_link_libraries(granite
        PRIVATE
            metal-cpp
            "-framework Metal"
            "-framework MetalKit"
            "-framework Foundation"
            "-framework QuartzCore"
    )

    target_compile_definitions(granite PUBLIC GRANITE_HAS_METAL=1)

    # Enable ARC for Objective-C++ files
    set_source_files_properties(${GRANITE_METAL_SOURCES} PROPERTIES
        COMPILE_FLAGS "-fobjc-arc"
    )
endif()

# =============================================================================
# CPU Backend
# =============================================================================

if(GRANITE_BUILD_CPU)
    set(GRANITE_CPU_SOURCES
        src/backend/cpu/cpu_backend.cpp
        src/operators/cpu/cpu_operators.cpp
    )

    target_sources(granite PRIVATE ${GRANITE_CPU_SOURCES})
    target_compile_definitions(granite PUBLIC GRANITE_HAS_CPU=1)

    # SIMD flags
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
        # ARM NEON is always available on arm64
        target_compile_definitions(granite PRIVATE GRANITE_HAS_NEON=1)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        # Check for AVX2 support
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
        if(COMPILER_SUPPORTS_AVX2)
            target_compile_options(granite PRIVATE -mavx2)
            target_compile_definitions(granite PRIVATE GRANITE_HAS_AVX2=1)
        endif()
    endif()
endif()

# =============================================================================
# Vulkan Backend (Linux/Android/Windows)
# =============================================================================

option(GRANITE_WITH_SHADERC "Enable runtime shader compilation with shaderc" OFF)
set(GRANITE_VULKAN_PRECOMPILED_DIR "" CACHE PATH "Directory for precompiled Vulkan SPIR-V shaders")
option(GRANITE_BUILD_ANDROID_JNI "Build Android JNI shared library" OFF)

if(GRANITE_BUILD_VULKAN)
    find_package(Vulkan REQUIRED)

    set(GRANITE_VULKAN_SOURCES
        src/backend/vulkan/vulkan_backend.cpp
        src/backend/vulkan/vulkan_compute.cpp
        src/operators/vulkan/vulkan_operators.cpp
    )

    target_sources(granite PRIVATE ${GRANITE_VULKAN_SOURCES})

    target_link_libraries(granite
        PRIVATE
            Vulkan::Vulkan
    )

    target_compile_definitions(granite PUBLIC GRANITE_HAS_VULKAN=1)
    target_compile_definitions(granite PRIVATE GRANITE_VULKAN_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders/vulkan")
    if(GRANITE_VULKAN_PRECOMPILED_DIR)
        target_compile_definitions(granite PRIVATE GRANITE_VULKAN_PRECOMPILED_DIR="${GRANITE_VULKAN_PRECOMPILED_DIR}")
    endif()

    # Optional: Link shaderc for runtime shader compilation
    if(GRANITE_WITH_SHADERC)
        # Try vcpkg-style first
        find_package(unofficial-shaderc CONFIG QUIET)
        if(unofficial-shaderc_FOUND)
            target_link_libraries(granite PRIVATE unofficial::shaderc::shaderc)
            target_compile_definitions(granite PUBLIC GRANITE_HAS_SHADERC=1)
            message(STATUS "Found shaderc via vcpkg")
        else()
            # Try system shaderc
            find_library(SHADERC_LIB shaderc_combined HINTS /usr/lib /usr/local/lib)
            find_path(SHADERC_INCLUDE shaderc/shaderc.hpp HINTS /usr/include /usr/local/include)
            if(SHADERC_LIB AND SHADERC_INCLUDE)
                target_link_libraries(granite PRIVATE ${SHADERC_LIB})
                target_include_directories(granite PRIVATE ${SHADERC_INCLUDE})
                target_compile_definitions(granite PUBLIC GRANITE_HAS_SHADERC=1)
                message(STATUS "Found system shaderc: ${SHADERC_LIB}")
            else()
                message(WARNING "GRANITE_WITH_SHADERC enabled but shaderc not found")
            endif()
        endif()
    endif()
endif()

# =============================================================================
# Android JNI Library
# =============================================================================

if(ANDROID AND GRANITE_BUILD_ANDROID_JNI)
    add_library(granite_jni SHARED src/android/granite_jni.cpp)
    target_link_libraries(granite_jni PRIVATE granite android log)
    target_include_directories(granite_jni PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
endif()

# =============================================================================
# Accelerate Framework (Apple BLAS/LAPACK)
# =============================================================================

if(APPLE)
    target_link_libraries(granite PRIVATE "-framework Accelerate")
    target_compile_definitions(granite PRIVATE GRANITE_HAS_ACCELERATE=1)
endif()

# =============================================================================
# OpenMP (optional, for CPU parallelization)
# =============================================================================

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(granite PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(granite PRIVATE GRANITE_HAS_OPENMP=1)
endif()

# =============================================================================
# Main executable (for development/testing)
# =============================================================================

add_executable(granite_main main.cpp)
target_link_libraries(granite_main PRIVATE granite)

# =============================================================================
# Tests
# =============================================================================

if(GRANITE_BUILD_TESTS)
    enable_testing()
    # Note: We use Catch2's test discovery instead of full CTest/CDash integration
    # to avoid cluttering the target list with Continuous*/Experimental*/Nightly* targets
    set(CTEST_USE_LAUNCHERS OFF)
    include(Catch)

    add_executable(granite_tests
        tests/test_main.cpp
        tests/test_types.cpp
        tests/test_tensor.cpp
        tests/test_operators.cpp
        tests/test_graph.cpp
        tests/test_memory.cpp
        tests/test_scheduler.cpp
        tests/test_optimization.cpp
        tests/test_gguf.cpp
        tests/test_llm.cpp
        tests/test_config.cpp
        tests/test_onnx.cpp
        tests/test_vulkan.cpp
    )

    target_link_libraries(granite_tests
        PRIVATE
            granite
            Catch2::Catch2WithMain
    )

    if(GRANITE_ENABLE_ONNX)
        target_include_directories(granite_tests PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
        target_include_directories(granite_tests PRIVATE ${Protobuf_INCLUDE_DIRS})
    endif()

    catch_discover_tests(granite_tests)
endif()

# =============================================================================
# Examples
# =============================================================================

if(GRANITE_BUILD_EXAMPLES)
    add_executable(example_basic examples/basic.cpp)
    target_link_libraries(example_basic PRIVATE granite)
endif()

# =============================================================================
# Tools
# =============================================================================

add_executable(granite_benchmark tools/benchmark.cpp)
target_link_libraries(granite_benchmark PRIVATE granite)

add_executable(granite_detailed_bench tools/detailed_benchmark.mm)
set_source_files_properties(tools/detailed_benchmark.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
target_link_libraries(granite_detailed_bench PRIVATE granite metal-cpp "-framework Metal" "-framework Foundation")

add_executable(granite_tree_bench tools/benchmark_tree_batching.cpp)
target_link_libraries(granite_tree_bench PRIVATE granite)

add_executable(granite_bench_vs_llamacpp tools/benchmark_vs_llamacpp.cpp)
add_executable(granite_stress_inference tools/stress_inference.cpp)
target_link_libraries(granite_stress_inference PRIVATE granite)
target_link_libraries(granite_bench_vs_llamacpp PRIVATE granite)

if(GRANITE_ENABLE_ONNX)
    add_executable(granite_onnx_smoke tools/onnx_smoke.cpp)
    target_link_libraries(granite_onnx_smoke PRIVATE granite)
endif()

add_executable(profile_q4k tools/profile_q4k.mm)
set_source_files_properties(tools/profile_q4k.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
target_link_libraries(profile_q4k PRIVATE granite metal-cpp "-framework Metal" "-framework Foundation")

add_executable(measure_alloc tools/measure_alloc.mm)
set_source_files_properties(tools/measure_alloc.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
target_link_libraries(measure_alloc PRIVATE granite metal-cpp "-framework Metal" "-framework Foundation")

add_executable(measure_overhead tools/measure_overhead.mm)
set_source_files_properties(tools/measure_overhead.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
target_link_libraries(measure_overhead PRIVATE granite metal-cpp "-framework Metal" "-framework Foundation")

add_executable(pipeline_test tools/pipeline_test.mm)
set_source_files_properties(tools/pipeline_test.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
target_link_libraries(pipeline_test PRIVATE granite metal-cpp "-framework Metal" "-framework Foundation")

add_executable(granite_matvec_bench tools/benchmark_matvec.mm)
set_source_files_properties(tools/benchmark_matvec.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
target_link_libraries(granite_matvec_bench PRIVATE granite metal-cpp "-framework Metal" "-framework Foundation")

add_executable(granite_attention_bench tools/benchmark_attention.mm)
set_source_files_properties(tools/benchmark_attention.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
target_link_libraries(granite_attention_bench PRIVATE granite metal-cpp "-framework Metal" "-framework Foundation")

add_executable(test_simdgroup_matmul tools/test_simdgroup_matmul.mm)
set_source_files_properties(tools/test_simdgroup_matmul.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
target_link_libraries(test_simdgroup_matmul PRIVATE granite metal-cpp "-framework Metal" "-framework Foundation")

add_executable(test_paged_flash_attention tools/test_paged_flash_attention.mm)
set_source_files_properties(tools/test_paged_flash_attention.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
target_link_libraries(test_paged_flash_attention PRIVATE granite metal-cpp "-framework Metal" "-framework Foundation")

add_executable(benchmark_paged_attention tools/benchmark_paged_attention.mm)
set_source_files_properties(tools/benchmark_paged_attention.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
target_link_libraries(benchmark_paged_attention PRIVATE granite metal-cpp "-framework Metal" "-framework Foundation")

# =============================================================================
# Installation
# =============================================================================

install(TARGETS granite
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/granite
    DESTINATION include
)

add_executable(benchmark_matmul_gpu tools/benchmark_matmul_gpu.mm)
target_link_libraries(benchmark_matmul_gpu granite)
target_include_directories(benchmark_matmul_gpu PRIVATE ${CMAKE_SOURCE_DIR}/src)

add_executable(profile_prefill_kernels tools/profile_prefill_kernels.mm)
set_source_files_properties(tools/profile_prefill_kernels.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
target_link_libraries(profile_prefill_kernels PRIVATE granite metal-cpp "-framework Metal" "-framework Foundation")
target_include_directories(profile_prefill_kernels PRIVATE ${CMAKE_SOURCE_DIR}/src)

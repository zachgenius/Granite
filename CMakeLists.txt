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
option(GRANITE_BUILD_CPU "Build CPU backend" ON)
option(GRANITE_BUILD_TESTS "Build tests" ON)
option(GRANITE_BUILD_EXAMPLES "Build examples" ON)

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

# =============================================================================
# Granite Core Library
# =============================================================================

set(GRANITE_CORE_SOURCES
    src/core/tensor.cpp
    src/core/memory_manager.cpp
    src/core/operator_registry.cpp
    src/core/error.cpp
    src/core/backend_factory.cpp
)

set(GRANITE_GRAPH_SOURCES
    src/graph/graph.cpp
    src/graph/scheduler.cpp
    src/graph/optimization.cpp
)

set(GRANITE_MODEL_SOURCES
    src/model/gguf.cpp
)

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
)

target_include_directories(granite
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(granite
    PUBLIC
        glm::glm
        spdlog::spdlog
    PRIVATE
        fmt::fmt
)

target_compile_definitions(granite PUBLIC
    GRANITE_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
    GRANITE_VERSION_MINOR=${PROJECT_VERSION_MINOR}
    GRANITE_VERSION_PATCH=${PROJECT_VERSION_PATCH}
)

# =============================================================================
# Metal Backend (Apple only)
# =============================================================================

if(APPLE AND GRANITE_BUILD_METAL)
    set(GRANITE_METAL_SOURCES
        src/backend/metal/metal_backend.mm
        src/backend/metal/metal_buffer.mm
        src/backend/metal/metal_pipeline.mm
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
    )

    target_link_libraries(granite_tests
        PRIVATE
            granite
            Catch2::Catch2WithMain
    )

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

add_executable(granite_tree_bench tools/benchmark_tree_batching.cpp)
target_link_libraries(granite_tree_bench PRIVATE granite)

add_executable(granite_bench_vs_llamacpp tools/benchmark_vs_llamacpp.cpp)
target_link_libraries(granite_bench_vs_llamacpp PRIVATE granite)

add_executable(granite_matvec_bench tools/benchmark_matvec.mm)
set_source_files_properties(tools/benchmark_matvec.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
target_link_libraries(granite_matvec_bench PRIVATE granite metal-cpp "-framework Metal" "-framework Foundation")

add_executable(granite_attention_bench tools/benchmark_attention.mm)
set_source_files_properties(tools/benchmark_attention.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
target_link_libraries(granite_attention_bench PRIVATE granite metal-cpp "-framework Metal" "-framework Foundation")

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

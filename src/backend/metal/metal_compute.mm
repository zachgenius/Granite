// MetalCompute - High-level GPU compute interface for LLM inference
// Manages shader compilation, pipeline states, and command buffer batching
//
// This file is organized into the following sections:
// 1. Implementation class (Impl) - manages Metal resources and shader compilation
// 2. Dispatch helpers - reduce boilerplate for common dispatch patterns
// 3. Core operations - matvec/matmul for all quantization types
// 4. Normalization operations - RMS norm variants
// 5. Element-wise operations - SiLU, multiply, add, RoPE
// 6. Fused kernels - combined operations for memory bandwidth reduction
// 7. Attention operations - single/multi-head, tree, paged attention
// 8. Buffer management - KV cache allocation and management

#include "granite/metal_compute.h"
#include "granite/log.h"

#ifdef GRANITE_HAS_METAL

// Note: NS_PRIVATE_IMPLEMENTATION and MTL_PRIVATE_IMPLEMENTATION are defined
// in metal_backend.mm to avoid duplicate symbols
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <unordered_map>
#include <string>
#include <vector>
#include <mutex>
#include <chrono>
#include <ctime>

namespace granite {

// Include embedded shader source
#include "kernels/metal_shaders.h"


// =============================================================================
// SECTION 1: Implementation Class
// =============================================================================

class MetalCompute::Impl {
public:
    // Profiling stats
    struct ProfilingStats {
        uint64_t dispatch_count = 0;
        uint64_t sync_count = 0;
        double sync_time_ms = 0.0;
        uint64_t command_buffer_count = 0;
    };
    ProfilingStats stats_;
    bool profiling_enabled_ = false;

    Impl() = default;
    ~Impl() { shutdown(); }

    Result<void> initialize(MTL::Device* device) {
        if (initialized_) return {};

        device_ = device;
        if (!device_) {
            return Error(ErrorCode::BackendNotInitialized, "No Metal device");
        }

        command_queue_ = device_->newCommandQueue();
        if (!command_queue_) {
            return Error(ErrorCode::InternalError, "Failed to create command queue");
        }

        auto compile_result = compile_shaders();
        if (!compile_result.ok()) {
            return compile_result.error();
        }

        initialized_ = true;
        GRANITE_LOG_INFO("MetalCompute initialized");
        return {};
    }

    void shutdown() {
        if (!initialized_) return;

        sync();

        for (auto& [name, pipeline] : pipelines_) {
            pipeline->release();
        }
        pipelines_.clear();

        if (command_queue_) {
            command_queue_->release();
            command_queue_ = nullptr;
        }

        initialized_ = false;
    }

    void sync() {
        if (current_command_buffer_) {
            if (current_encoder_) {
                current_encoder_->endEncoding();
                current_encoder_ = nullptr;
            }

            auto start = std::chrono::high_resolution_clock::now();
            current_command_buffer_->commit();
            current_command_buffer_->waitUntilCompleted();
            auto end = std::chrono::high_resolution_clock::now();

            if (profiling_enabled_) {
                stats_.sync_count++;
                stats_.sync_time_ms += std::chrono::duration<double, std::milli>(end - start).count();
            }

            current_command_buffer_->release();
            current_command_buffer_ = nullptr;
        }
    }

    void commit() {
        if (current_command_buffer_) {
            if (current_encoder_) {
                current_encoder_->endEncoding();
                current_encoder_ = nullptr;
            }
            current_command_buffer_->commit();
            current_command_buffer_ = nullptr;
        }
    }

    bool is_initialized() const { return initialized_; }
    MTL::Device* device() const { return device_; }

    MTL::ComputeCommandEncoder* get_encoder() {
        if (!current_encoder_) {
            current_command_buffer_ = command_queue_->commandBuffer();
            current_encoder_ = current_command_buffer_->computeCommandEncoder();
            if (profiling_enabled_) {
                stats_.command_buffer_count++;
            }
        }
        if (profiling_enabled_) {
            stats_.dispatch_count++;
        }
        return current_encoder_;
    }

    void enable_profiling(bool enable) {
        profiling_enabled_ = enable;
        if (enable) {
            stats_ = ProfilingStats{};  // Reset stats
        }
    }

    ProfilingStats get_stats() const { return stats_; }

    void reset_stats() { stats_ = ProfilingStats{}; }

    MTL::ComputePipelineState* get_pipeline(const std::string& name) {
        auto it = pipelines_.find(name);
        return (it != pipelines_.end()) ? it->second : nullptr;
    }

    MTL::Buffer* create_buffer(size_t size, bool shared) {
        MTL::ResourceOptions options = shared ?
            MTL::ResourceStorageModeShared :
            MTL::ResourceStorageModePrivate;
        return device_->newBuffer(size, options);
    }

    // -------------------------------------------------------------------------
    // Dispatch Helpers (reduce boilerplate for common patterns)
    // -------------------------------------------------------------------------

    // Standard matvec dispatch: y = x @ W^T
    // Uses 2 SIMD groups (64 threads) matching llama.cpp for better occupancy
    Result<void> dispatch_matvec(
        const char* kernel_name,
        MTL::Buffer* x, MTL::Buffer* W, MTL::Buffer* y,
        uint32_t K, uint32_t N,
        uint32_t rows_per_simd = 2)
    {
        auto* encoder = get_encoder();
        auto* pipeline = get_pipeline(kernel_name);
        if (!pipeline) {
            return Error(ErrorCode::InternalError, std::string(kernel_name) + " pipeline not found");
        }

        encoder->setComputePipelineState(pipeline);
        encoder->setBuffer(x, 0, 0);
        encoder->setBuffer(W, 0, 1);
        encoder->setBuffer(y, 0, 2);
        encoder->setBytes(&K, sizeof(K), 3);
        encoder->setBytes(&N, sizeof(N), 4);

        // 2 SIMD groups per threadgroup (64 threads) - matches llama.cpp
        // Smaller threadgroups = better occupancy and more parallelism
        constexpr uint32_t simd_groups = 2;
        uint32_t rows_per_tg = simd_groups * rows_per_simd;
        uint32_t num_threadgroups = (N + rows_per_tg - 1) / rows_per_tg;
        MTL::Size grid_size = MTL::Size::Make(num_threadgroups, 1, 1);
        MTL::Size threadgroup_size = MTL::Size::Make(32 * simd_groups, 1, 1);
        encoder->dispatchThreadgroups(grid_size, threadgroup_size);

        return {};
    }

    // Standard matmul dispatch: Y = X @ W^T
    // Uses dispatchThreads with 16x16 threadgroups
    Result<void> dispatch_matmul(
        const char* kernel_name,
        MTL::Buffer* X, MTL::Buffer* W, MTL::Buffer* Y,
        uint32_t M, uint32_t K, uint32_t N)
    {
        auto* encoder = get_encoder();
        auto* pipeline = get_pipeline(kernel_name);
        if (!pipeline) {
            return Error(ErrorCode::InternalError, std::string(kernel_name) + " pipeline not found");
        }

        encoder->setComputePipelineState(pipeline);
        encoder->setBuffer(X, 0, 0);
        encoder->setBuffer(W, 0, 1);
        encoder->setBuffer(Y, 0, 2);
        encoder->setBytes(&M, sizeof(M), 3);
        encoder->setBytes(&K, sizeof(K), 4);
        encoder->setBytes(&N, sizeof(N), 5);

        MTL::Size grid_size = MTL::Size::Make(N, M, 1);
        MTL::Size threadgroup_size = MTL::Size::Make(16, 16, 1);
        encoder->dispatchThreads(grid_size, threadgroup_size);

        return {};
    }

    // Register-tiled matmul dispatch for prefill
    // Each thread handles 2 rows, so grid Y dimension is halved
    Result<void> dispatch_matmul_tiled(
        const char* kernel_name,
        MTL::Buffer* X, MTL::Buffer* W, MTL::Buffer* Y,
        uint32_t M, uint32_t K, uint32_t N)
    {
        auto* encoder = get_encoder();
        auto* pipeline = get_pipeline(kernel_name);
        if (!pipeline) {
            return Error(ErrorCode::InternalError, std::string(kernel_name) + " pipeline not found");
        }

        encoder->setComputePipelineState(pipeline);
        encoder->setBuffer(X, 0, 0);
        encoder->setBuffer(W, 0, 1);
        encoder->setBuffer(Y, 0, 2);
        encoder->setBytes(&M, sizeof(M), 3);
        encoder->setBytes(&K, sizeof(K), 4);
        encoder->setBytes(&N, sizeof(N), 5);

        // Grid Y is halved since each thread handles 2 rows
        uint32_t grid_M = (M + 1) / 2;
        MTL::Size grid_size = MTL::Size::Make(N, grid_M, 1);
        MTL::Size threadgroup_size = MTL::Size::Make(16, 16, 1);
        encoder->dispatchThreads(grid_size, threadgroup_size);

        return {};
    }

    // SIMD K-parallel matmul dispatch for prefill
    // Each simdgroup (32 threads) handles 4 rows x 1 col with 8-way K parallelism
    Result<void> dispatch_matmul_simd(
        const char* kernel_name,
        MTL::Buffer* X, MTL::Buffer* W, MTL::Buffer* Y,
        uint32_t M, uint32_t K, uint32_t N)
    {
        auto* encoder = get_encoder();
        auto* pipeline = get_pipeline(kernel_name);
        if (!pipeline) {
            return Error(ErrorCode::InternalError, std::string(kernel_name) + " pipeline not found");
        }

        encoder->setComputePipelineState(pipeline);
        encoder->setBuffer(X, 0, 0);
        encoder->setBuffer(W, 0, 1);
        encoder->setBuffer(Y, 0, 2);
        encoder->setBytes(&M, sizeof(M), 3);
        encoder->setBytes(&K, sizeof(K), 4);
        encoder->setBytes(&N, sizeof(N), 5);

        // Grid of threadgroups: (N cols, M/4 row groups)
        // Each threadgroup (1 simdgroup of 32 threads) handles 4 rows x 1 col
        // IMPORTANT: Use dispatchThreadgroups so that all 32 threads in a simdgroup
        // share the same threadgroup_position_in_grid, enabling K-parallel reduction.
        uint32_t grid_M = (M + 3) / 4;
        MTL::Size grid_size = MTL::Size::Make(N, grid_M, 1);
        MTL::Size threadgroup_size = MTL::Size::Make(32, 1, 1);  // 1 simdgroup
        encoder->dispatchThreadgroups(grid_size, threadgroup_size);

        return {};
    }

    // Simdgroup matrix matmul dispatch for prefill
    // Uses simdgroup_half8x8 matrices and simdgroup_multiply_accumulate
    // Tile sizes: NR0=64 for N (output cols), NR1=32 for M (batch rows)
    // Requires 8KB threadgroup memory (4KB for weights + 2KB for activations)
    Result<void> dispatch_matmul_simdgroup(
        const char* kernel_name,
        MTL::Buffer* X, MTL::Buffer* W, MTL::Buffer* Y,
        uint32_t M, uint32_t K, uint32_t N)
    {
        auto* encoder = get_encoder();
        auto* pipeline = get_pipeline(kernel_name);
        if (!pipeline) {
            return Error(ErrorCode::InternalError, std::string(kernel_name) + " pipeline not found");
        }

        encoder->setComputePipelineState(pipeline);
        encoder->setBuffer(X, 0, 0);
        encoder->setBuffer(W, 0, 1);
        encoder->setBuffer(Y, 0, 2);
        encoder->setBytes(&M, sizeof(M), 3);
        encoder->setBytes(&K, sizeof(K), 4);
        encoder->setBytes(&N, sizeof(N), 5);

        // Set threadgroup memory for simdgroup operations
        constexpr size_t shmem_size = 8192;  // 8KB: 4KB for weights + 2KB for activations
        encoder->setThreadgroupMemoryLength(shmem_size, 0);

        // Kernel uses:
        //   r0 = tgpig.y * NR0 for N dimension (64 output columns per TG)
        //   r1 = tgpig.x * NR1 for M dimension (32 batch rows per TG)
        constexpr uint32_t NR0 = 64;  // N (output cols) per threadgroup
        constexpr uint32_t NR1 = 32;  // M (batch rows) per threadgroup

        // Grid: x = M tiles, y = N tiles (matches kernel's tgpig usage)
        uint32_t num_m_tiles = (M + NR1 - 1) / NR1;  // tgpig.x range
        uint32_t num_n_tiles = (N + NR0 - 1) / NR0;  // tgpig.y range
        MTL::Size grid_size = MTL::Size::Make(num_m_tiles, num_n_tiles, 1);

        // 128 threads per threadgroup (4 simdgroups of 32 threads)
        MTL::Size threadgroup_size = MTL::Size::Make(128, 1, 1);
        encoder->dispatchThreadgroups(grid_size, threadgroup_size);

        return {};
    }


    // Fused gate+up simdgroup matmul dispatch
    // Computes both gate and up projections in a single kernel launch
    // Grid z=0 for gate, z=1 for up - both share the same input X loading
    Result<void> dispatch_fused_gate_up(
        const char* kernel_name,
        MTL::Buffer* X, MTL::Buffer* W_gate, MTL::Buffer* W_up,
        MTL::Buffer* Y_gate, MTL::Buffer* Y_up,
        uint32_t M, uint32_t K, uint32_t N)
    {
        auto* encoder = get_encoder();
        auto* pipeline = get_pipeline(kernel_name);
        if (!pipeline) {
            return Error(ErrorCode::InternalError, std::string(kernel_name) + " pipeline not found");
        }

        encoder->setComputePipelineState(pipeline);
        encoder->setBuffer(X, 0, 0);
        encoder->setBuffer(W_gate, 0, 1);
        encoder->setBuffer(W_up, 0, 2);
        encoder->setBuffer(Y_gate, 0, 3);
        encoder->setBuffer(Y_up, 0, 4);
        encoder->setBytes(&M, sizeof(M), 5);
        encoder->setBytes(&K, sizeof(K), 6);
        encoder->setBytes(&N, sizeof(N), 7);

        // Set threadgroup memory (same as simdgroup matmul)
        constexpr size_t shmem_size = 8192;  // 8KB
        encoder->setThreadgroupMemoryLength(shmem_size, 0);

        // Tiling constants (same as matmul_q4k_simdgroup)
        constexpr uint32_t NR0 = 64;  // N (output cols) per threadgroup
        constexpr uint32_t NR1 = 32;  // M (batch rows) per threadgroup

        // Grid: x = M tiles, y = N tiles, z = 2 (gate and up)
        uint32_t num_m_tiles = (M + NR1 - 1) / NR1;
        uint32_t num_n_tiles = (N + NR0 - 1) / NR0;
        MTL::Size grid_size = MTL::Size::Make(num_m_tiles, num_n_tiles, 2);

        MTL::Size threadgroup_size = MTL::Size::Make(128, 1, 1);
        encoder->dispatchThreadgroups(grid_size, threadgroup_size);

        return {};
    }

    // Fused RMSNorm + matvec dispatch
    // Uses 8 rows per threadgroup with 256 threads
    Result<void> dispatch_rms_norm_matvec(
        const char* kernel_name,
        MTL::Buffer* x, MTL::Buffer* norm_weight, MTL::Buffer* W, MTL::Buffer* y,
        uint32_t K, uint32_t N, float eps)
    {
        auto* encoder = get_encoder();
        auto* pipeline = get_pipeline(kernel_name);
        if (!pipeline) {
            return Error(ErrorCode::InternalError, std::string(kernel_name) + " pipeline not found");
        }

        encoder->setComputePipelineState(pipeline);
        encoder->setBuffer(x, 0, 0);
        encoder->setBuffer(norm_weight, 0, 1);
        encoder->setBuffer(W, 0, 2);
        encoder->setBuffer(y, 0, 3);
        encoder->setBytes(&K, sizeof(K), 4);
        encoder->setBytes(&N, sizeof(N), 5);
        encoder->setBytes(&eps, sizeof(eps), 6);

        // FUSED_ROWS_PER_TG = 8
        constexpr uint32_t rows_per_tg = 8;
        uint32_t num_threadgroups = (N + rows_per_tg - 1) / rows_per_tg;
        MTL::Size grid_size = MTL::Size::Make(num_threadgroups, 1, 1);
        MTL::Size threadgroup_size = MTL::Size::Make(256, 1, 1);
        encoder->dispatchThreadgroups(grid_size, threadgroup_size);

        return {};
    }

    // Matvec with residual add dispatch
    Result<void> dispatch_matvec_residual(
        const char* kernel_name,
        MTL::Buffer* x, MTL::Buffer* W, MTL::Buffer* residual, MTL::Buffer* y,
        uint32_t K, uint32_t N,
        uint32_t rows_per_tg = 8)
    {
        auto* encoder = get_encoder();
        auto* pipeline = get_pipeline(kernel_name);
        if (!pipeline) {
            return Error(ErrorCode::InternalError, std::string(kernel_name) + " pipeline not found");
        }

        encoder->setComputePipelineState(pipeline);
        encoder->setBuffer(x, 0, 0);
        encoder->setBuffer(W, 0, 1);
        encoder->setBuffer(residual, 0, 2);
        encoder->setBuffer(y, 0, 3);
        encoder->setBytes(&K, sizeof(K), 4);
        encoder->setBytes(&N, sizeof(N), 5);

        uint32_t num_threadgroups = (N + rows_per_tg - 1) / rows_per_tg;
        MTL::Size grid_size = MTL::Size::Make(num_threadgroups, 1, 1);
        MTL::Size threadgroup_size = MTL::Size::Make(256, 1, 1);
        encoder->dispatchThreadgroups(grid_size, threadgroup_size);

        return {};
    }

private:
    Result<void> compile_shaders() {
        NS::Error* error = nullptr;
        MTL::CompileOptions* options = MTL::CompileOptions::alloc()->init();

        NS::String* source = NS::String::string(METAL_SHADER_SOURCE.c_str(), NS::UTF8StringEncoding);
        MTL::Library* library = device_->newLibrary(source, options, &error);
        options->release();

        if (!library) {
            std::string msg = "Shader compilation failed";
            if (error) {
                msg = error->localizedDescription()->utf8String();
            }
            return Error(ErrorCode::ShaderCompilationFailed, msg);
        }

        // All kernel names to compile
        std::vector<std::string> kernels = {
            // Core quantized kernels
            "matvec_q4k", "matmul_q4k", "matmul_q4k_vec", "matmul_q4k_tiled", "matmul_q4k_simd",
            // NOTE: matmul_q4k_simdgroup uses function constants and is compiled separately below
            "matvec_f16", "matmul_f16", "matmul_f16_tiled",
            "matvec_q8_0", "matmul_q8_0",
            "matvec_q4_0", "matmul_q4_0",
            "matvec_iq4_nl", "matmul_iq4_nl",
            "matvec_iq4_xs", "matmul_iq4_xs",
            "matvec_iq3_s", "matmul_iq3_s",
            "matvec_q6_k", "matmul_q6_k",
            "matvec_q5_k", "matmul_q5_k",
            "matvec_q3_k", "matmul_q3_k",
            "matvec_q2_k", "matmul_q2_k",
            // Normalization and element-wise
            "rms_norm", "rms_norm_f16", "rms_norm_batch", "rms_norm_batch_f16",
            "rms_norm_batch_half", "rms_norm_batch_half_f32w",  // Half I/O for prefill
            "rms_norm_batch_f32_to_f16", "rms_norm_batch_f16w_to_f16",  // Float input, half output
            "silu", "elementwise_mul", "rope",
            "elementwise_add", "elementwise_add_half",
            "convert_f32_to_f16", "convert_f16_to_f32",  // Format conversion
            "rope_multihead", "softmax_row",
            // Basic attention
            "attention_decode", "kv_cache_append", "kv_cache_append_f16",
            "multihead_attention_decode", "multihead_attention_decode_f16kv",
            "embedding_lookup",
            // Fused kernels
            "silu_mul", "silu_mul_half",
            "rms_norm_matvec_q4k", "rms_norm_matvec_f16",
            "rms_norm_matvec_q8_0", "rms_norm_matvec_q4_0", "rms_norm_matvec_iq4_nl",
            "rms_norm_matvec_iq4_xs", "rms_norm_matvec_iq3_s", "rms_norm_matvec_q6_k",
            "rms_norm_matvec_q5_k", "rms_norm_matvec_q3_k", "rms_norm_matvec_q2_k",
            // Phase 2 fused kernels
            "rms_norm_dual_matvec_q4k", "rms_norm_dual_matvec_q3k", "rms_norm_dual_matvec_q2k",
            "matvec_residual_q4k", "matvec_residual_q3k", "matvec_residual_q2k",
            // Fused QKV
            "fused_qkv_matvec_q4k",
            // Fused gate+up (non-Q4_K)
            "fused_gate_up_q8_0", "fused_gate_up_q4_0",
            // Simdgroup Flash Attention (high performance decode)
            "simdgroup_flash_attention_decode_f16kv_d64", "simdgroup_flash_attention_decode_f16kv_d128",
            // llama.cpp-style Flash Attention (highest performance)
            "flash_attention_decode_d64", "flash_attention_decode_d128",
            // Prefill attention (direct global V loads, half scores)
            "flash_attention_prefill",
            // Tree attention (speculative decoding)
            "attention_tree_f16kv", "attention_tree_nocontext_f16kv",
            // Paged attention (continuous batching)
            "paged_attention_decode", "paged_kv_cache_append",
            "batched_paged_attention_decode",
            // Paged flash attention (long context support)
            "paged_flash_attention_decode_d64", "paged_flash_attention_decode_d64_v2",
            "paged_flash_attention_decode_d128"
        };

        for (const auto& name : kernels) {
            NS::String* func_name = NS::String::string(name.c_str(), NS::UTF8StringEncoding);
            MTL::Function* func = library->newFunction(func_name);

            if (!func) {
                GRANITE_LOG_WARN("Function '{}' not found in shader library", name);
                continue;
            }

            MTL::ComputePipelineState* pipeline = device_->newComputePipelineState(func, &error);
            func->release();

            if (!pipeline) {
                GRANITE_LOG_WARN("Failed to create pipeline for '{}'", name);
                continue;
            }

            pipelines_[name] = pipeline;
        }

        // Compile matmul_q4k_simdgroup with function constants (llama.cpp style)
        // FC_MUL_MM + 0 = FC_mm_bc_inp (bounds check input)
        // FC_MUL_MM + 1 = FC_mm_bc_out (bounds check output)
        constexpr uint32_t FC_MUL_MM = 100;
        struct SgmmVariant {
            const char* name;
            bool bc_inp;
            bool bc_out;
        };
        SgmmVariant sgmm_variants[] = {
            {"matmul_q4k_simdgroup_00", false, false},  // Inner tiles - no bounds check
            {"matmul_q4k_simdgroup_01", false, true},   // Edge N only
            {"matmul_q4k_simdgroup_10", true, false},   // Edge M only
            {"matmul_q4k_simdgroup_11", true, true},    // Edge tile - full bounds check
        };

        NS::String* sgmm_func_name = NS::String::string("matmul_q4k_simdgroup", NS::UTF8StringEncoding);

        for (const auto& v : sgmm_variants) {
            MTL::FunctionConstantValues* fc_values = MTL::FunctionConstantValues::alloc()->init();
            fc_values->setConstantValue(&v.bc_inp, MTL::DataTypeBool, FC_MUL_MM + 0);
            fc_values->setConstantValue(&v.bc_out, MTL::DataTypeBool, FC_MUL_MM + 1);

            MTL::Function* func = library->newFunction(sgmm_func_name, fc_values, &error);
            fc_values->release();

            if (!func) {
                GRANITE_LOG_WARN("Function '{}' not found with constants", v.name);
                continue;
            }

            MTL::ComputePipelineState* pipeline = device_->newComputePipelineState(func, &error);
            func->release();

            if (!pipeline) {
                GRANITE_LOG_WARN("Failed to create pipeline for '{}'", v.name);
                continue;
            }

            pipelines_[v.name] = pipeline;
        }

        // Compile FP16 input variants of simdgroup matmul
        SgmmVariant sgmm_f16_variants[] = {
            {"matmul_q4k_simdgroup_f16_00", false, false},
            {"matmul_q4k_simdgroup_f16_01", false, true},
            {"matmul_q4k_simdgroup_f16_10", true, false},
            {"matmul_q4k_simdgroup_f16_11", true, true},
        };

        NS::String* sgmm_f16_func_name = NS::String::string("matmul_q4k_simdgroup_f16", NS::UTF8StringEncoding);

        for (const auto& v : sgmm_f16_variants) {
            MTL::FunctionConstantValues* fc_values = MTL::FunctionConstantValues::alloc()->init();
            fc_values->setConstantValue(&v.bc_inp, MTL::DataTypeBool, FC_MUL_MM + 0);
            fc_values->setConstantValue(&v.bc_out, MTL::DataTypeBool, FC_MUL_MM + 1);

            MTL::Function* func = library->newFunction(sgmm_f16_func_name, fc_values, &error);
            fc_values->release();

            if (!func) {
                GRANITE_LOG_WARN("Function '{}' not found with constants", v.name);
                continue;
            }

            MTL::ComputePipelineState* pipeline = device_->newComputePipelineState(func, &error);
            func->release();

            if (!pipeline) {
                GRANITE_LOG_WARN("Failed to create pipeline for '{}'", v.name);
                continue;
            }

            pipelines_[v.name] = pipeline;
        }

        // Compile FP16 weight simdgroup matmul (for full logits output)
        SgmmVariant sgmm_f16w_variants[] = {
            {"matmul_f16_simdgroup_00", false, false},
            {"matmul_f16_simdgroup_01", false, true},
            {"matmul_f16_simdgroup_10", true, false},
            {"matmul_f16_simdgroup_11", true, true},
        };

        NS::String* sgmm_f16w_func_name = NS::String::string("matmul_f16_simdgroup", NS::UTF8StringEncoding);

        for (const auto& v : sgmm_f16w_variants) {
            MTL::FunctionConstantValues* fc_values = MTL::FunctionConstantValues::alloc()->init();
            fc_values->setConstantValue(&v.bc_inp, MTL::DataTypeBool, FC_MUL_MM + 0);
            fc_values->setConstantValue(&v.bc_out, MTL::DataTypeBool, FC_MUL_MM + 1);

            MTL::Function* func = library->newFunction(sgmm_f16w_func_name, fc_values, &error);
            fc_values->release();

            if (!func) {
                GRANITE_LOG_WARN("Function '{}' not found with constants", v.name);
                continue;
            }

            MTL::ComputePipelineState* pipeline = device_->newComputePipelineState(func, &error);
            func->release();

            if (!pipeline) {
                GRANITE_LOG_WARN("Failed to create pipeline for '{}'", v.name);
                continue;
            }

            pipelines_[v.name] = pipeline;
        }

        // Compile FP16 weight simdgroup matmul (NK=64 variant)
        SgmmVariant sgmm_f16w_nk64_variants[] = {
            {"matmul_f16_simdgroup_nk64_00", false, false},
            {"matmul_f16_simdgroup_nk64_01", false, true},
            {"matmul_f16_simdgroup_nk64_10", true, false},
            {"matmul_f16_simdgroup_nk64_11", true, true},
        };

        NS::String* sgmm_f16w_nk64_func_name = NS::String::string("matmul_f16_simdgroup_nk64", NS::UTF8StringEncoding);

        for (const auto& v : sgmm_f16w_nk64_variants) {
            MTL::FunctionConstantValues* fc_values = MTL::FunctionConstantValues::alloc()->init();
            fc_values->setConstantValue(&v.bc_inp, MTL::DataTypeBool, FC_MUL_MM + 0);
            fc_values->setConstantValue(&v.bc_out, MTL::DataTypeBool, FC_MUL_MM + 1);

            MTL::Function* func = library->newFunction(sgmm_f16w_nk64_func_name, fc_values, &error);
            fc_values->release();

            if (!func) {
                GRANITE_LOG_WARN("Function '{}' not found with constants", v.name);
                continue;
            }

            MTL::ComputePipelineState* pipeline = device_->newComputePipelineState(func, &error);
            func->release();

            if (!pipeline) {
                GRANITE_LOG_WARN("Failed to create pipeline for '{}'", v.name);
                continue;
            }

            pipelines_[v.name] = pipeline;
        }

        // Compile fused_gate_up_q4k_simdgroup with function constants
        // This kernel fuses gate and up projections into a single dispatch
        SgmmVariant fused_gate_up_variants[] = {
            {"fused_gate_up_q4k_simdgroup_00", false, false},
            {"fused_gate_up_q4k_simdgroup_01", false, true},
            {"fused_gate_up_q4k_simdgroup_10", true, false},
            {"fused_gate_up_q4k_simdgroup_11", true, true},
        };

        NS::String* fused_gate_up_func_name = NS::String::string("fused_gate_up_q4k_simdgroup", NS::UTF8StringEncoding);

        for (const auto& v : fused_gate_up_variants) {
            MTL::FunctionConstantValues* fc_values = MTL::FunctionConstantValues::alloc()->init();
            fc_values->setConstantValue(&v.bc_inp, MTL::DataTypeBool, FC_MUL_MM + 0);
            fc_values->setConstantValue(&v.bc_out, MTL::DataTypeBool, FC_MUL_MM + 1);

            MTL::Function* func = library->newFunction(fused_gate_up_func_name, fc_values, &error);
            fc_values->release();

            if (!func) {
                GRANITE_LOG_WARN("Function '{}' not found with constants", v.name);
                continue;
            }

            MTL::ComputePipelineState* pipeline = device_->newComputePipelineState(func, &error);
            func->release();

            if (!pipeline) {
                GRANITE_LOG_WARN("Failed to create pipeline for '{}'", v.name);
                continue;
            }

            pipelines_[v.name] = pipeline;
        }

        // Compile fused_gate_up_q4k_simdgroup_f32 with function constants
        // FP32 input variant for non-FP16 RMSNorm path.
        SgmmVariant fused_gate_up_f32_variants[] = {
            {"fused_gate_up_q4k_simdgroup_f32_00", false, false},
            {"fused_gate_up_q4k_simdgroup_f32_01", false, true},
            {"fused_gate_up_q4k_simdgroup_f32_10", true, false},
            {"fused_gate_up_q4k_simdgroup_f32_11", true, true},
        };

        NS::String* fused_gate_up_f32_func_name =
            NS::String::string("fused_gate_up_q4k_simdgroup_f32", NS::UTF8StringEncoding);

        for (const auto& v : fused_gate_up_f32_variants) {
            MTL::FunctionConstantValues* fc_values = MTL::FunctionConstantValues::alloc()->init();
            fc_values->setConstantValue(&v.bc_inp, MTL::DataTypeBool, FC_MUL_MM + 0);
            fc_values->setConstantValue(&v.bc_out, MTL::DataTypeBool, FC_MUL_MM + 1);

            MTL::Function* func = library->newFunction(fused_gate_up_f32_func_name, fc_values, &error);
            fc_values->release();

            if (!func) {
                GRANITE_LOG_WARN("Function '{}' not found with constants", v.name);
                continue;
            }

            MTL::ComputePipelineState* pipeline = device_->newComputePipelineState(func, &error);
            func->release();

            if (!pipeline) {
                GRANITE_LOG_WARN("Failed to create pipeline for '{}'", v.name);
                continue;
            }

            pipelines_[v.name] = pipeline;
        }

        library->release();
        return {};
    }

    bool initialized_ = false;
    MTL::Device* device_ = nullptr;
    MTL::CommandQueue* command_queue_ = nullptr;
    MTL::CommandBuffer* current_command_buffer_ = nullptr;
    MTL::ComputeCommandEncoder* current_encoder_ = nullptr;
    std::unordered_map<std::string, MTL::ComputePipelineState*> pipelines_;
};


// =============================================================================
// SECTION 2: MetalCompute Public Interface and Core Operations
// =============================================================================

MetalCompute::MetalCompute() : impl_(new Impl()) {}
MetalCompute::~MetalCompute() { delete impl_; }

Result<void> MetalCompute::initialize(MTL::Device* device) {
    return impl_->initialize(device);
}

void MetalCompute::shutdown() { impl_->shutdown(); }
void MetalCompute::sync() { impl_->sync(); }
void MetalCompute::commit() { impl_->commit(); }
bool MetalCompute::is_initialized() const { return impl_->is_initialized(); }
MTL::Device* MetalCompute::device() const { return impl_->device(); }

// Profiling API
void MetalCompute::enable_profiling(bool enable) { impl_->enable_profiling(enable); }
void MetalCompute::reset_profiling_stats() { impl_->reset_stats(); }
void MetalCompute::get_profiling_stats(uint64_t& dispatches, uint64_t& syncs, double& sync_time_ms, uint64_t& cmd_buffers) const {
    auto stats = impl_->get_stats();
    dispatches = stats.dispatch_count;
    syncs = stats.sync_count;
    sync_time_ms = stats.sync_time_ms;
    cmd_buffers = stats.command_buffer_count;
}

// GPU Capture API for Xcode profiler
bool MetalCompute::begin_capture(const char* capture_path) {
    auto* capture_manager = MTL::CaptureManager::sharedCaptureManager();
    if (!capture_manager) {
        GRANITE_LOG_ERROR("Failed to get MTLCaptureManager");
        return false;
    }

    auto* descriptor = MTL::CaptureDescriptor::alloc()->init();
    descriptor->setCaptureObject((__bridge id)impl_->device());

    if (capture_path) {
        // Save to file for later analysis
        auto* url = NS::URL::fileURLWithPath(NS::String::string(capture_path, NS::UTF8StringEncoding));
        descriptor->setOutputURL(url);
        descriptor->setDestination(MTL::CaptureDestinationGPUTraceDocument);
    } else {
        // Default: generate timestamped filename
        time_t now = time(nullptr);
        char path[256];
        strftime(path, sizeof(path), "/tmp/granite_gpu_%Y%m%d_%H%M%S.gputrace", localtime(&now));
        auto* url = NS::URL::fileURLWithPath(NS::String::string(path, NS::UTF8StringEncoding));
        descriptor->setOutputURL(url);
        descriptor->setDestination(MTL::CaptureDestinationGPUTraceDocument);
        GRANITE_LOG_INFO("GPU capture will be saved to: {}", path);
    }

    NS::Error* error = nullptr;
    if (!capture_manager->startCapture(descriptor, &error)) {
        if (error) {
            GRANITE_LOG_ERROR("Failed to start GPU capture: {}",
                error->localizedDescription()->utf8String());
        }
        descriptor->release();
        return false;
    }

    descriptor->release();
    GRANITE_LOG_INFO("GPU capture started");
    return true;
}

void MetalCompute::end_capture() {
    auto* capture_manager = MTL::CaptureManager::sharedCaptureManager();
    if (capture_manager && capture_manager->isCapturing()) {
        capture_manager->stopCapture();
        GRANITE_LOG_INFO("GPU capture stopped - open .gputrace file in Xcode to analyze");
    }
}

MTL::Buffer* MetalCompute::create_buffer(size_t size, bool shared) {
    return impl_->create_buffer(size, shared);
}

// -----------------------------------------------------------------------------
// Q4_K Quantized Operations
// -----------------------------------------------------------------------------

Result<void> MetalCompute::matvec_q4k(
    MTL::Buffer* x, MTL::Buffer* W, MTL::Buffer* y,
    uint32_t K, uint32_t N)
{
    return impl_->dispatch_matvec("matvec_q4k", x, W, y, K, N);
}

Result<void> MetalCompute::matmul_q4k(
    MTL::Buffer* X, MTL::Buffer* W, MTL::Buffer* Y,
    uint32_t M, uint32_t K, uint32_t N)
{
    // Simdgroup matrix kernel for prefill batches (M >= 32)
    // Uses simdgroup_half8x8 matrices and simdgroup_multiply_accumulate.
    // Compiled with function constants for compile-time branch elimination:
    //   - kernel_00: No bounds checking (aligned dimensions)
    //   - kernel_11: Full bounds checking (edge tiles)
    if (M >= 32) {
        // Use fast kernel when dimensions are perfectly aligned (no bounds checking needed)
        // NR1=32 (M rows), NR0=64 (N cols)
        if (M % 32 == 0 && N % 64 == 0) {
            return impl_->dispatch_matmul_simdgroup("matmul_q4k_simdgroup_00", X, W, Y, M, K, N);
        }
        return impl_->dispatch_matmul_simdgroup("matmul_q4k_simdgroup_11", X, W, Y, M, K, N);
    }
    // Use tiled kernel for medium batches
    if (M > 2) {
        return impl_->dispatch_matmul_tiled("matmul_q4k_tiled", X, W, Y, M, K, N);
    }
    // Use vectorized kernel for M=2
    if (M > 1) {
        return impl_->dispatch_matmul("matmul_q4k_vec", X, W, Y, M, K, N);
    }
    // Scalar kernel for M=1
    return impl_->dispatch_matmul("matmul_q4k", X, W, Y, M, K, N);
}

Result<void> MetalCompute::matmul_q4k_f16(
    MTL::Buffer* X, MTL::Buffer* W, MTL::Buffer* Y,
    uint32_t M, uint32_t K, uint32_t N)
{
    // FP16 input simdgroup matmul kernel for prefill batches (M >= 32)
    // Uses simdgroup_half8x8 matrices with FP16 activation input.
    // This reduces memory bandwidth by 2x for activations.
    if (M >= 32) {
        // Use fast kernel when dimensions are perfectly aligned (no bounds checking needed)
        if (M % 32 == 0 && N % 64 == 0) {
            return impl_->dispatch_matmul_simdgroup("matmul_q4k_simdgroup_f16_00", X, W, Y, M, K, N);
        }
        return impl_->dispatch_matmul_simdgroup("matmul_q4k_simdgroup_f16_11", X, W, Y, M, K, N);
    }
    // For small batches, convert to FP32 on-the-fly is not worth it - fall back to FP32 kernel
    // The caller should use convert_f32_to_f16 only when M >= 32
    return matmul_q4k(X, W, Y, M, K, N);
}

Result<void> MetalCompute::fused_gate_up_q4k(
    MTL::Buffer* X, MTL::Buffer* W_gate, MTL::Buffer* W_up,
    MTL::Buffer* Y_gate, MTL::Buffer* Y_up,
    uint32_t M, uint32_t K, uint32_t N)
{
    // Fused gate+up kernel for FFN in transformer prefill
    // Computes both gate and up projections in a single dispatch:
    //   Y_gate = X @ W_gate^T  (for SiLU gate)
    //   Y_up   = X @ W_up^T    (for up projection)
    // Both share the same input X loading, reducing memory bandwidth.
    // Uses FP16 input X, Q4_K quantized weights, FP32 output.
    if (M >= 32) {
        if (M % 32 == 0 && N % 64 == 0) {
            return impl_->dispatch_fused_gate_up(
                "fused_gate_up_q4k_simdgroup_00",
                X, W_gate, W_up, Y_gate, Y_up, M, K, N);
        }
        return impl_->dispatch_fused_gate_up(
            "fused_gate_up_q4k_simdgroup_11",
            X, W_gate, W_up, Y_gate, Y_up, M, K, N);
    }
    // For small batches, fall back to separate matmul calls
    // (fused kernel not worth the overhead for decode)
    auto res = matmul_q4k_f16(X, W_gate, Y_gate, M, K, N);
    if (!res.ok()) return res;
    return matmul_q4k_f16(X, W_up, Y_up, M, K, N);
}

Result<void> MetalCompute::fused_gate_up_q4k_f32(
    MTL::Buffer* X, MTL::Buffer* W_gate, MTL::Buffer* W_up,
    MTL::Buffer* Y_gate, MTL::Buffer* Y_up,
    uint32_t M, uint32_t K, uint32_t N)
{
    // FP32 input variant for the fused gate+up kernel.
    if (M >= 32) {
        if (M % 32 == 0 && N % 64 == 0) {
            return impl_->dispatch_fused_gate_up(
                "fused_gate_up_q4k_simdgroup_f32_00",
                X, W_gate, W_up, Y_gate, Y_up, M, K, N);
        }
        return impl_->dispatch_fused_gate_up(
            "fused_gate_up_q4k_simdgroup_f32_11",
            X, W_gate, W_up, Y_gate, Y_up, M, K, N);
    }
    // For small batches, fall back to separate matmul calls.
    auto res = matmul_q4k(X, W_gate, Y_gate, M, K, N);
    if (!res.ok()) return res;
    return matmul_q4k(X, W_up, Y_up, M, K, N);
}

// -----------------------------------------------------------------------------
// Q8_0 Quantized Operations
// -----------------------------------------------------------------------------

Result<void> MetalCompute::matvec_q8_0(
    MTL::Buffer* x, MTL::Buffer* W, MTL::Buffer* y,
    uint32_t K, uint32_t N)
{
    return impl_->dispatch_matvec("matvec_q8_0", x, W, y, K, N);
}

Result<void> MetalCompute::matmul_q8_0(
    MTL::Buffer* X, MTL::Buffer* W, MTL::Buffer* Y,
    uint32_t M, uint32_t K, uint32_t N)
{
    return impl_->dispatch_matmul("matmul_q8_0", X, W, Y, M, K, N);
}

Result<void> MetalCompute::fused_gate_up_q8_0(
    MTL::Buffer* X, MTL::Buffer* W_gate, MTL::Buffer* W_up,
    MTL::Buffer* Y_gate, MTL::Buffer* Y_up,
    uint32_t M, uint32_t K, uint32_t N)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("fused_gate_up_q8_0");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "fused_gate_up_q8_0 pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(X, 0, 0);
    encoder->setBuffer(W_gate, 0, 1);
    encoder->setBuffer(W_up, 0, 2);
    encoder->setBuffer(Y_gate, 0, 3);
    encoder->setBuffer(Y_up, 0, 4);
    encoder->setBytes(&M, sizeof(M), 5);
    encoder->setBytes(&K, sizeof(K), 6);
    encoder->setBytes(&N, sizeof(N), 7);

    MTL::Size grid_size = MTL::Size::Make(N, M, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(256, 1, 1);
    encoder->dispatchThreads(grid_size, threadgroup_size);

    return {};
}

// -----------------------------------------------------------------------------
// Q4_0 Quantized Operations
// -----------------------------------------------------------------------------

Result<void> MetalCompute::matvec_q4_0(
    MTL::Buffer* x, MTL::Buffer* W, MTL::Buffer* y,
    uint32_t K, uint32_t N)
{
    return impl_->dispatch_matvec("matvec_q4_0", x, W, y, K, N);
}

Result<void> MetalCompute::matmul_q4_0(
    MTL::Buffer* X, MTL::Buffer* W, MTL::Buffer* Y,
    uint32_t M, uint32_t K, uint32_t N)
{
    return impl_->dispatch_matmul("matmul_q4_0", X, W, Y, M, K, N);
}

Result<void> MetalCompute::fused_gate_up_q4_0(
    MTL::Buffer* X, MTL::Buffer* W_gate, MTL::Buffer* W_up,
    MTL::Buffer* Y_gate, MTL::Buffer* Y_up,
    uint32_t M, uint32_t K, uint32_t N)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("fused_gate_up_q4_0");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "fused_gate_up_q4_0 pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(X, 0, 0);
    encoder->setBuffer(W_gate, 0, 1);
    encoder->setBuffer(W_up, 0, 2);
    encoder->setBuffer(Y_gate, 0, 3);
    encoder->setBuffer(Y_up, 0, 4);
    encoder->setBytes(&M, sizeof(M), 5);
    encoder->setBytes(&K, sizeof(K), 6);
    encoder->setBytes(&N, sizeof(N), 7);

    MTL::Size grid_size = MTL::Size::Make(N, M, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(256, 1, 1);
    encoder->dispatchThreads(grid_size, threadgroup_size);

    return {};
}

// -----------------------------------------------------------------------------
// IQ4_NL Quantized Operations (Non-linear 4-bit I-quant)
// -----------------------------------------------------------------------------

Result<void> MetalCompute::matvec_iq4_nl(
    MTL::Buffer* x, MTL::Buffer* W, MTL::Buffer* y,
    uint32_t K, uint32_t N)
{
    return impl_->dispatch_matvec("matvec_iq4_nl", x, W, y, K, N);
}

Result<void> MetalCompute::matmul_iq4_nl(
    MTL::Buffer* X, MTL::Buffer* W, MTL::Buffer* Y,
    uint32_t M, uint32_t K, uint32_t N)
{
    return impl_->dispatch_matmul("matmul_iq4_nl", X, W, Y, M, K, N);
}

// -----------------------------------------------------------------------------
// IQ4_XS Quantized Operations (4-bit with super-block scales)
// -----------------------------------------------------------------------------

Result<void> MetalCompute::matvec_iq4_xs(
    MTL::Buffer* x, MTL::Buffer* W, MTL::Buffer* y,
    uint32_t K, uint32_t N)
{
    return impl_->dispatch_matvec("matvec_iq4_xs", x, W, y, K, N);
}

Result<void> MetalCompute::matmul_iq4_xs(
    MTL::Buffer* X, MTL::Buffer* W, MTL::Buffer* Y,
    uint32_t M, uint32_t K, uint32_t N)
{
    return impl_->dispatch_matmul("matmul_iq4_xs", X, W, Y, M, K, N);
}

// -----------------------------------------------------------------------------
// IQ3_S Quantized Operations (3-bit I-quant with grid)
// -----------------------------------------------------------------------------

Result<void> MetalCompute::matvec_iq3_s(
    MTL::Buffer* x, MTL::Buffer* W, MTL::Buffer* y,
    uint32_t K, uint32_t N)
{
    return impl_->dispatch_matvec("matvec_iq3_s", x, W, y, K, N);
}

Result<void> MetalCompute::matmul_iq3_s(
    MTL::Buffer* X, MTL::Buffer* W, MTL::Buffer* Y,
    uint32_t M, uint32_t K, uint32_t N)
{
    return impl_->dispatch_matmul("matmul_iq3_s", X, W, Y, M, K, N);
}

// -----------------------------------------------------------------------------
// Q6_K Quantized Operations
// -----------------------------------------------------------------------------

Result<void> MetalCompute::matvec_q6_k(
    MTL::Buffer* x, MTL::Buffer* W, MTL::Buffer* y,
    uint32_t K, uint32_t N)
{
    return impl_->dispatch_matvec("matvec_q6_k", x, W, y, K, N);
}

Result<void> MetalCompute::matmul_q6_k(
    MTL::Buffer* X, MTL::Buffer* W, MTL::Buffer* Y,
    uint32_t M, uint32_t K, uint32_t N)
{
    return impl_->dispatch_matmul("matmul_q6_k", X, W, Y, M, K, N);
}

// -----------------------------------------------------------------------------
// Q5_K Quantized Operations
// -----------------------------------------------------------------------------

Result<void> MetalCompute::matvec_q5_k(
    MTL::Buffer* x, MTL::Buffer* W, MTL::Buffer* y,
    uint32_t K, uint32_t N)
{
    return impl_->dispatch_matvec("matvec_q5_k", x, W, y, K, N);
}

Result<void> MetalCompute::matmul_q5_k(
    MTL::Buffer* X, MTL::Buffer* W, MTL::Buffer* Y,
    uint32_t M, uint32_t K, uint32_t N)
{
    return impl_->dispatch_matmul("matmul_q5_k", X, W, Y, M, K, N);
}

// -----------------------------------------------------------------------------
// Q3_K Quantized Operations
// -----------------------------------------------------------------------------

Result<void> MetalCompute::matvec_q3_k(
    MTL::Buffer* x, MTL::Buffer* W, MTL::Buffer* y,
    uint32_t K, uint32_t N)
{
    return impl_->dispatch_matvec("matvec_q3_k", x, W, y, K, N);
}

Result<void> MetalCompute::matmul_q3_k(
    MTL::Buffer* X, MTL::Buffer* W, MTL::Buffer* Y,
    uint32_t M, uint32_t K, uint32_t N)
{
    return impl_->dispatch_matmul("matmul_q3_k", X, W, Y, M, K, N);
}

// -----------------------------------------------------------------------------
// Q2_K Quantized Operations
// -----------------------------------------------------------------------------

Result<void> MetalCompute::matvec_q2_k(
    MTL::Buffer* x, MTL::Buffer* W, MTL::Buffer* y,
    uint32_t K, uint32_t N)
{
    return impl_->dispatch_matvec("matvec_q2_k", x, W, y, K, N);
}

Result<void> MetalCompute::matmul_q2_k(
    MTL::Buffer* X, MTL::Buffer* W, MTL::Buffer* Y,
    uint32_t M, uint32_t K, uint32_t N)
{
    return impl_->dispatch_matmul("matmul_q2_k", X, W, Y, M, K, N);
}

// -----------------------------------------------------------------------------
// FP16 Operations
// -----------------------------------------------------------------------------

Result<void> MetalCompute::matvec_f16(
    MTL::Buffer* x, MTL::Buffer* W, MTL::Buffer* y,
    uint32_t K, uint32_t N)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("matvec_f16");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "matvec_f16 pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(x, 0, 0);
    encoder->setBuffer(W, 0, 1);
    encoder->setBuffer(y, 0, 2);
    encoder->setBytes(&K, sizeof(K), 3);
    encoder->setBytes(&N, sizeof(N), 4);

    // One SIMD group per output row.
    MTL::Size grid_size = MTL::Size::Make(N, 1, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(32, 1, 1);
    encoder->dispatchThreadgroups(grid_size, threadgroup_size);

    return {};
}

Result<void> MetalCompute::matmul_f16(
    MTL::Buffer* X, MTL::Buffer* W, MTL::Buffer* Y,
    uint32_t M, uint32_t K, uint32_t N)
{
    // Simdgroup matrix kernel for prefill batches (M >= 32)
    // Uses simdgroup_half8x8 matrices and simdgroup_multiply_accumulate.
    if (M >= 32 && (K % 32 == 0)) {
        if (K % 64 == 0) {
            auto* encoder = impl_->get_encoder();
            const bool aligned = (M % 32 == 0) && (N % 64 == 0);
            const char* kernel_name = aligned ? "matmul_f16_simdgroup_nk64_00"
                                              : "matmul_f16_simdgroup_nk64_11";
            auto* pipeline = impl_->get_pipeline(kernel_name);
            if (!pipeline) {
                return Error(ErrorCode::InternalError, "matmul_f16_simdgroup_nk64 pipeline not found");
            }

            encoder->setComputePipelineState(pipeline);
            encoder->setBuffer(X, 0, 0);
            encoder->setBuffer(W, 0, 1);
            encoder->setBuffer(Y, 0, 2);
            encoder->setBytes(&M, sizeof(M), 3);
            encoder->setBytes(&K, sizeof(K), 4);
            encoder->setBytes(&N, sizeof(N), 5);

            constexpr uint32_t NR0 = 64;
            constexpr uint32_t NR1 = 32;
            constexpr size_t shmem_size = 16384;
            encoder->setThreadgroupMemoryLength(shmem_size, 0);

            uint32_t num_m_tiles = (M + NR1 - 1) / NR1;
            uint32_t num_n_tiles = (N + NR0 - 1) / NR0;
            MTL::Size grid_size = MTL::Size::Make(num_m_tiles, num_n_tiles, 1);
            MTL::Size threadgroup_size = MTL::Size::Make(128, 1, 1);
            encoder->dispatchThreadgroups(grid_size, threadgroup_size);

            return {};
        }

        if (M % 32 == 0 && N % 64 == 0) {
            return impl_->dispatch_matmul_simdgroup("matmul_f16_simdgroup_00", X, W, Y, M, K, N);
        }
        return impl_->dispatch_matmul_simdgroup("matmul_f16_simdgroup_11", X, W, Y, M, K, N);
    }

    auto* encoder = impl_->get_encoder();

    if (M >= 16 && N >= 16) {
        auto* pipeline = impl_->get_pipeline("matmul_f16_tiled");
        if (!pipeline) {
            return Error(ErrorCode::InternalError, "matmul_f16_tiled pipeline not found");
        }

        encoder->setComputePipelineState(pipeline);
        encoder->setBuffer(X, 0, 0);
        encoder->setBuffer(W, 0, 1);
        encoder->setBuffer(Y, 0, 2);
        encoder->setBytes(&M, sizeof(M), 3);
        encoder->setBytes(&K, sizeof(K), 4);
        encoder->setBytes(&N, sizeof(N), 5);

        constexpr uint32_t tile = 16;
        uint32_t grid_x = (N + tile - 1) / tile;
        uint32_t grid_y = (M + tile - 1) / tile;
        MTL::Size grid_size = MTL::Size::Make(grid_x, grid_y, 1);
        MTL::Size threadgroup_size = MTL::Size::Make(tile, tile, 1);
        encoder->dispatchThreadgroups(grid_size, threadgroup_size);

        return {};
    }

    // Fallback for small batches
    return impl_->dispatch_matmul("matmul_f16", X, W, Y, M, K, N);
}


// =============================================================================
// SECTION 4: Normalization Operations
// =============================================================================

Result<void> MetalCompute::rms_norm(
    MTL::Buffer* x, MTL::Buffer* weight, MTL::Buffer* out,
    uint32_t size, float eps)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("rms_norm");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "rms_norm pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(x, 0, 0);
    encoder->setBuffer(weight, 0, 1);
    encoder->setBuffer(out, 0, 2);
    encoder->setBytes(&size, sizeof(size), 3);
    encoder->setBytes(&eps, sizeof(eps), 4);

    MTL::Size grid_size = MTL::Size::Make(size, 1, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(256, 1, 1);
    encoder->dispatchThreads(grid_size, threadgroup_size);

    return {};
}

Result<void> MetalCompute::rms_norm_f16(
    MTL::Buffer* x, MTL::Buffer* weight, MTL::Buffer* out,
    uint32_t size, float eps)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("rms_norm_f16");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "rms_norm_f16 pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(x, 0, 0);
    encoder->setBuffer(weight, 0, 1);
    encoder->setBuffer(out, 0, 2);
    encoder->setBytes(&size, sizeof(size), 3);
    encoder->setBytes(&eps, sizeof(eps), 4);

    MTL::Size grid_size = MTL::Size::Make(size, 1, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(256, 1, 1);
    encoder->dispatchThreads(grid_size, threadgroup_size);

    return {};
}

Result<void> MetalCompute::rms_norm_batch(
    MTL::Buffer* x, MTL::Buffer* weight, MTL::Buffer* out,
    uint32_t batch_size, uint32_t hidden_dim, float eps)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("rms_norm_batch");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "rms_norm_batch pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(x, 0, 0);
    encoder->setBuffer(weight, 0, 1);
    encoder->setBuffer(out, 0, 2);
    encoder->setBytes(&batch_size, sizeof(batch_size), 3);
    encoder->setBytes(&hidden_dim, sizeof(hidden_dim), 4);
    encoder->setBytes(&eps, sizeof(eps), 5);

    // One threadgroup per token
    MTL::Size grid_size = MTL::Size::Make(batch_size, 1, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(256, 1, 1);
    encoder->dispatchThreadgroups(grid_size, threadgroup_size);

    return {};
}

Result<void> MetalCompute::rms_norm_batch_f16(
    MTL::Buffer* x, MTL::Buffer* weight, MTL::Buffer* out,
    uint32_t batch_size, uint32_t hidden_dim, float eps)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("rms_norm_batch_f16");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "rms_norm_batch_f16 pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(x, 0, 0);
    encoder->setBuffer(weight, 0, 1);
    encoder->setBuffer(out, 0, 2);
    encoder->setBytes(&batch_size, sizeof(batch_size), 3);
    encoder->setBytes(&hidden_dim, sizeof(hidden_dim), 4);
    encoder->setBytes(&eps, sizeof(eps), 5);

    // One threadgroup per token
    MTL::Size grid_size = MTL::Size::Make(batch_size, 1, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(256, 1, 1);
    encoder->dispatchThreadgroups(grid_size, threadgroup_size);

    return {};
}

Result<void> MetalCompute::rms_norm_batch_f32_to_f16(
    MTL::Buffer* x, MTL::Buffer* weight, MTL::Buffer* out,
    uint32_t batch_size, uint32_t hidden_dim, float eps)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("rms_norm_batch_f32_to_f16");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "rms_norm_batch_f32_to_f16 pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(x, 0, 0);
    encoder->setBuffer(weight, 0, 1);
    encoder->setBuffer(out, 0, 2);
    encoder->setBytes(&batch_size, sizeof(batch_size), 3);
    encoder->setBytes(&hidden_dim, sizeof(hidden_dim), 4);
    encoder->setBytes(&eps, sizeof(eps), 5);

    MTL::Size grid_size = MTL::Size::Make(batch_size, 1, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(256, 1, 1);
    encoder->dispatchThreadgroups(grid_size, threadgroup_size);

    return {};
}

Result<void> MetalCompute::rms_norm_batch_f16w_to_f16(
    MTL::Buffer* x, MTL::Buffer* weight, MTL::Buffer* out,
    uint32_t batch_size, uint32_t hidden_dim, float eps)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("rms_norm_batch_f16w_to_f16");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "rms_norm_batch_f16w_to_f16 pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(x, 0, 0);
    encoder->setBuffer(weight, 0, 1);
    encoder->setBuffer(out, 0, 2);
    encoder->setBytes(&batch_size, sizeof(batch_size), 3);
    encoder->setBytes(&hidden_dim, sizeof(hidden_dim), 4);
    encoder->setBytes(&eps, sizeof(eps), 5);

    MTL::Size grid_size = MTL::Size::Make(batch_size, 1, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(256, 1, 1);
    encoder->dispatchThreadgroups(grid_size, threadgroup_size);

    return {};
}


// =============================================================================
// SECTION 5: Element-wise Operations
// =============================================================================

Result<void> MetalCompute::silu(MTL::Buffer* x, uint32_t size) {
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("silu");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "silu pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(x, 0, 0);
    encoder->setBytes(&size, sizeof(size), 1);

    MTL::Size grid_size = MTL::Size::Make(size, 1, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(256, 1, 1);
    encoder->dispatchThreads(grid_size, threadgroup_size);

    return {};
}

Result<void> MetalCompute::elementwise_mul(
    MTL::Buffer* a, MTL::Buffer* b, MTL::Buffer* c, uint32_t size)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("elementwise_mul");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "elementwise_mul pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(a, 0, 0);
    encoder->setBuffer(b, 0, 1);
    encoder->setBuffer(c, 0, 2);
    encoder->setBytes(&size, sizeof(size), 3);

    MTL::Size grid_size = MTL::Size::Make(size, 1, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(256, 1, 1);
    encoder->dispatchThreads(grid_size, threadgroup_size);

    return {};
}

Result<void> MetalCompute::rope(
    MTL::Buffer* x, uint32_t seq_len, uint32_t head_dim,
    uint32_t start_pos, float freq_base)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("rope");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "rope pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(x, 0, 0);
    encoder->setBytes(&seq_len, sizeof(seq_len), 1);
    encoder->setBytes(&head_dim, sizeof(head_dim), 2);
    encoder->setBytes(&start_pos, sizeof(start_pos), 3);
    encoder->setBytes(&freq_base, sizeof(freq_base), 4);

    MTL::Size grid_size = MTL::Size::Make(head_dim / 2, seq_len, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(32, 1, 1);
    encoder->dispatchThreads(grid_size, threadgroup_size);

    return {};
}

Result<void> MetalCompute::elementwise_add(
    MTL::Buffer* a, MTL::Buffer* b, MTL::Buffer* c, uint32_t size)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("elementwise_add");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "elementwise_add pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(a, 0, 0);
    encoder->setBuffer(b, 0, 1);
    encoder->setBuffer(c, 0, 2);
    encoder->setBytes(&size, sizeof(size), 3);

    MTL::Size grid_size = MTL::Size::Make(size, 1, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(256, 1, 1);
    encoder->dispatchThreads(grid_size, threadgroup_size);

    return {};
}

Result<void> MetalCompute::convert_f32_to_f16(
    MTL::Buffer* src, MTL::Buffer* dst, uint32_t size)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("convert_f32_to_f16");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "convert_f32_to_f16 pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(src, 0, 0);
    encoder->setBuffer(dst, 0, 1);
    encoder->setBytes(&size, sizeof(size), 2);

    MTL::Size grid_size = MTL::Size::Make(size, 1, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(256, 1, 1);
    encoder->dispatchThreads(grid_size, threadgroup_size);

    return {};
}

Result<void> MetalCompute::rope_multihead(
    MTL::Buffer* q, MTL::Buffer* k,
    uint32_t num_heads_q, uint32_t num_heads_k,
    uint32_t seq_len, uint32_t head_dim,
    uint32_t start_pos, float freq_base)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("rope_multihead");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "rope_multihead pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(q, 0, 0);
    encoder->setBuffer(k, 0, 1);
    encoder->setBytes(&num_heads_q, sizeof(num_heads_q), 2);
    encoder->setBytes(&num_heads_k, sizeof(num_heads_k), 3);
    encoder->setBytes(&seq_len, sizeof(seq_len), 4);
    encoder->setBytes(&head_dim, sizeof(head_dim), 5);
    encoder->setBytes(&start_pos, sizeof(start_pos), 6);
    encoder->setBytes(&freq_base, sizeof(freq_base), 7);

    // Grid: [head_dim/2, seq_len, num_heads_q + num_heads_k]
    MTL::Size grid_size = MTL::Size::Make(head_dim / 2, seq_len, num_heads_q + num_heads_k);
    MTL::Size threadgroup_size = MTL::Size::Make(32, 1, 1);
    encoder->dispatchThreads(grid_size, threadgroup_size);

    return {};
}

Result<void> MetalCompute::softmax(MTL::Buffer* x, uint32_t M, uint32_t N)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("softmax_row");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "softmax_row pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(x, 0, 0);
    encoder->setBytes(&M, sizeof(M), 1);
    encoder->setBytes(&N, sizeof(N), 2);

    MTL::Size grid_size = MTL::Size::Make(N, M, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(256, 1, 1);
    encoder->dispatchThreads(grid_size, threadgroup_size);

    return {};
}


// =============================================================================
// SECTION 6: Fused Kernels (Memory Bandwidth Optimization)
// =============================================================================

// Fused SiLU + Multiply
Result<void> MetalCompute::silu_mul(
    MTL::Buffer* a, MTL::Buffer* b, MTL::Buffer* c, uint32_t size)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("silu_mul");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "silu_mul pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(a, 0, 0);
    encoder->setBuffer(b, 0, 1);
    encoder->setBuffer(c, 0, 2);
    encoder->setBytes(&size, sizeof(size), 3);

    MTL::Size grid_size = MTL::Size::Make(size, 1, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(256, 1, 1);
    encoder->dispatchThreads(grid_size, threadgroup_size);

    return {};
}

// -----------------------------------------------------------------------------
// Fused RMSNorm + MatVec (all quantization types)
// -----------------------------------------------------------------------------

Result<void> MetalCompute::rms_norm_matvec_q4k(
    MTL::Buffer* x, MTL::Buffer* norm_weight, MTL::Buffer* W, MTL::Buffer* y,
    uint32_t K, uint32_t N, float eps)
{
    return impl_->dispatch_rms_norm_matvec("rms_norm_matvec_q4k", x, norm_weight, W, y, K, N, eps);
}

Result<void> MetalCompute::rms_norm_matvec_f16(
    MTL::Buffer* x, MTL::Buffer* norm_weight, MTL::Buffer* W, MTL::Buffer* y,
    uint32_t K, uint32_t N, float eps)
{
    return impl_->dispatch_rms_norm_matvec("rms_norm_matvec_f16", x, norm_weight, W, y, K, N, eps);
}

Result<void> MetalCompute::rms_norm_matvec_q8_0(
    MTL::Buffer* x, MTL::Buffer* norm_weight, MTL::Buffer* W, MTL::Buffer* y,
    uint32_t K, uint32_t N, float eps)
{
    return impl_->dispatch_rms_norm_matvec("rms_norm_matvec_q8_0", x, norm_weight, W, y, K, N, eps);
}

Result<void> MetalCompute::rms_norm_matvec_q4_0(
    MTL::Buffer* x, MTL::Buffer* norm_weight, MTL::Buffer* W, MTL::Buffer* y,
    uint32_t K, uint32_t N, float eps)
{
    return impl_->dispatch_rms_norm_matvec("rms_norm_matvec_q4_0", x, norm_weight, W, y, K, N, eps);
}

Result<void> MetalCompute::rms_norm_matvec_iq4_nl(
    MTL::Buffer* x, MTL::Buffer* norm_weight, MTL::Buffer* W, MTL::Buffer* y,
    uint32_t K, uint32_t N, float eps)
{
    return impl_->dispatch_rms_norm_matvec("rms_norm_matvec_iq4_nl", x, norm_weight, W, y, K, N, eps);
}

Result<void> MetalCompute::rms_norm_matvec_iq4_xs(
    MTL::Buffer* x, MTL::Buffer* norm_weight, MTL::Buffer* W, MTL::Buffer* y,
    uint32_t K, uint32_t N, float eps)
{
    return impl_->dispatch_rms_norm_matvec("rms_norm_matvec_iq4_xs", x, norm_weight, W, y, K, N, eps);
}

Result<void> MetalCompute::rms_norm_matvec_iq3_s(
    MTL::Buffer* x, MTL::Buffer* norm_weight, MTL::Buffer* W, MTL::Buffer* y,
    uint32_t K, uint32_t N, float eps)
{
    return impl_->dispatch_rms_norm_matvec("rms_norm_matvec_iq3_s", x, norm_weight, W, y, K, N, eps);
}

Result<void> MetalCompute::rms_norm_matvec_q6_k(
    MTL::Buffer* x, MTL::Buffer* norm_weight, MTL::Buffer* W, MTL::Buffer* y,
    uint32_t K, uint32_t N, float eps)
{
    return impl_->dispatch_rms_norm_matvec("rms_norm_matvec_q6_k", x, norm_weight, W, y, K, N, eps);
}

Result<void> MetalCompute::rms_norm_matvec_q5_k(
    MTL::Buffer* x, MTL::Buffer* norm_weight, MTL::Buffer* W, MTL::Buffer* y,
    uint32_t K, uint32_t N, float eps)
{
    return impl_->dispatch_rms_norm_matvec("rms_norm_matvec_q5_k", x, norm_weight, W, y, K, N, eps);
}

Result<void> MetalCompute::rms_norm_matvec_q3_k(
    MTL::Buffer* x, MTL::Buffer* norm_weight, MTL::Buffer* W, MTL::Buffer* y,
    uint32_t K, uint32_t N, float eps)
{
    return impl_->dispatch_rms_norm_matvec("rms_norm_matvec_q3_k", x, norm_weight, W, y, K, N, eps);
}

Result<void> MetalCompute::rms_norm_matvec_q2_k(
    MTL::Buffer* x, MTL::Buffer* norm_weight, MTL::Buffer* W, MTL::Buffer* y,
    uint32_t K, uint32_t N, float eps)
{
    return impl_->dispatch_rms_norm_matvec("rms_norm_matvec_q2_k", x, norm_weight, W, y, K, N, eps);
}

// -----------------------------------------------------------------------------
// Dual MatVec (for FFN gate + up projections)
// -----------------------------------------------------------------------------

Result<void> MetalCompute::rms_norm_dual_matvec_q4k(
    MTL::Buffer* x, MTL::Buffer* norm_weight,
    MTL::Buffer* W_gate, MTL::Buffer* W_up,
    MTL::Buffer* y_gate, MTL::Buffer* y_up,
    uint32_t K, uint32_t N, float eps)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("rms_norm_dual_matvec_q4k");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "rms_norm_dual_matvec_q4k pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(x, 0, 0);
    encoder->setBuffer(norm_weight, 0, 1);
    encoder->setBuffer(W_gate, 0, 2);
    encoder->setBuffer(W_up, 0, 3);
    encoder->setBuffer(y_gate, 0, 4);
    encoder->setBuffer(y_up, 0, 5);
    encoder->setBytes(&K, sizeof(K), 6);
    encoder->setBytes(&N, sizeof(N), 7);
    encoder->setBytes(&eps, sizeof(eps), 8);

    // DUAL_ROWS_PER_TG = 8
    constexpr uint32_t rows_per_tg = 8;
    uint32_t num_threadgroups = (N + rows_per_tg - 1) / rows_per_tg;
    MTL::Size grid_size = MTL::Size::Make(num_threadgroups, 1, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(256, 1, 1);
    encoder->dispatchThreadgroups(grid_size, threadgroup_size);

    return {};
}

Result<void> MetalCompute::rms_norm_dual_matvec_q3k(
    MTL::Buffer* x, MTL::Buffer* norm_weight,
    MTL::Buffer* W_gate, MTL::Buffer* W_up,
    MTL::Buffer* y_gate, MTL::Buffer* y_up,
    uint32_t K, uint32_t N, float eps)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("rms_norm_dual_matvec_q3k");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "rms_norm_dual_matvec_q3k pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(x, 0, 0);
    encoder->setBuffer(norm_weight, 0, 1);
    encoder->setBuffer(W_gate, 0, 2);
    encoder->setBuffer(W_up, 0, 3);
    encoder->setBuffer(y_gate, 0, 4);
    encoder->setBuffer(y_up, 0, 5);
    encoder->setBytes(&K, sizeof(K), 6);
    encoder->setBytes(&N, sizeof(N), 7);
    encoder->setBytes(&eps, sizeof(eps), 8);

    constexpr uint32_t rows_per_tg = 8;
    uint32_t num_threadgroups = (N + rows_per_tg - 1) / rows_per_tg;
    MTL::Size grid_size = MTL::Size::Make(num_threadgroups, 1, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(256, 1, 1);
    encoder->dispatchThreadgroups(grid_size, threadgroup_size);

    return {};
}

Result<void> MetalCompute::rms_norm_dual_matvec_q2k(
    MTL::Buffer* x, MTL::Buffer* norm_weight,
    MTL::Buffer* W_gate, MTL::Buffer* W_up,
    MTL::Buffer* y_gate, MTL::Buffer* y_up,
    uint32_t K, uint32_t N, float eps)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("rms_norm_dual_matvec_q2k");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "rms_norm_dual_matvec_q2k pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(x, 0, 0);
    encoder->setBuffer(norm_weight, 0, 1);
    encoder->setBuffer(W_gate, 0, 2);
    encoder->setBuffer(W_up, 0, 3);
    encoder->setBuffer(y_gate, 0, 4);
    encoder->setBuffer(y_up, 0, 5);
    encoder->setBytes(&K, sizeof(K), 6);
    encoder->setBytes(&N, sizeof(N), 7);
    encoder->setBytes(&eps, sizeof(eps), 8);

    constexpr uint32_t rows_per_tg = 8;
    uint32_t num_threadgroups = (N + rows_per_tg - 1) / rows_per_tg;
    MTL::Size grid_size = MTL::Size::Make(num_threadgroups, 1, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(256, 1, 1);
    encoder->dispatchThreadgroups(grid_size, threadgroup_size);

    return {};
}

// -----------------------------------------------------------------------------
// MatVec + Residual (for down projection + residual add)
// -----------------------------------------------------------------------------

Result<void> MetalCompute::matvec_residual_q4k(
    MTL::Buffer* x, MTL::Buffer* W, MTL::Buffer* residual, MTL::Buffer* y,
    uint32_t K, uint32_t N)
{
    return impl_->dispatch_matvec_residual("matvec_residual_q4k", x, W, residual, y, K, N, 8);
}

Result<void> MetalCompute::matvec_residual_q3k(
    MTL::Buffer* x, MTL::Buffer* W, MTL::Buffer* residual, MTL::Buffer* y,
    uint32_t K, uint32_t N)
{
    return impl_->dispatch_matvec_residual("matvec_residual_q3k", x, W, residual, y, K, N, 16);
}

Result<void> MetalCompute::matvec_residual_q2k(
    MTL::Buffer* x, MTL::Buffer* W, MTL::Buffer* residual, MTL::Buffer* y,
    uint32_t K, uint32_t N)
{
    return impl_->dispatch_matvec_residual("matvec_residual_q2k", x, W, residual, y, K, N, 16);
}


// =============================================================================
// SECTION 7: Attention Operations
// =============================================================================

// -----------------------------------------------------------------------------
// Single-Head Attention
// -----------------------------------------------------------------------------

Result<void> MetalCompute::attention_single_head(
    MTL::Buffer* Q, MTL::Buffer* K, MTL::Buffer* V, MTL::Buffer* output,
    uint32_t seq_q, uint32_t seq_kv, uint32_t head_dim,
    uint32_t start_pos, float scale)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("attention_decode");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "attention_decode pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(Q, 0, 0);
    encoder->setBuffer(K, 0, 1);
    encoder->setBuffer(V, 0, 2);
    encoder->setBuffer(output, 0, 3);
    encoder->setBytes(&seq_q, sizeof(seq_q), 4);
    encoder->setBytes(&seq_kv, sizeof(seq_kv), 5);
    encoder->setBytes(&head_dim, sizeof(head_dim), 6);
    encoder->setBytes(&start_pos, sizeof(start_pos), 7);
    encoder->setBytes(&scale, sizeof(scale), 8);

    if (seq_q == 1) {
        // Decode mode: single query token
        MTL::Size grid_size = MTL::Size::Make(head_dim, 1, 1);
        MTL::Size threadgroup_size = MTL::Size::Make(128, 1, 1);
        encoder->dispatchThreads(grid_size, threadgroup_size);
    } else {
        // Prefill mode: multiple query tokens
        MTL::Size grid_size = MTL::Size::Make(head_dim, seq_q, 1);
        MTL::Size threadgroup_size = MTL::Size::Make(32, 4, 1);
        encoder->dispatchThreads(grid_size, threadgroup_size);
    }

    return {};
}

// -----------------------------------------------------------------------------
// KV Cache Operations
// -----------------------------------------------------------------------------

std::pair<MTL::Buffer*, MTL::Buffer*> MetalCompute::create_kv_cache(
    uint32_t num_kv_heads, uint32_t max_seq_len, uint32_t head_dim)
{
    // FP16 storage for KV cache
    // Use shared memory so CPU can sync data during prefill->decode transition
    size_t size = num_kv_heads * max_seq_len * head_dim * sizeof(uint16_t);
    return {
        impl_->create_buffer(size, true),  // K cache (shared memory for CPU sync)
        impl_->create_buffer(size, true)   // V cache (shared memory for CPU sync)
    };
}

std::pair<MTL::Buffer*, MTL::Buffer*> MetalCompute::create_kv_cache_f32(
    uint32_t num_kv_heads, uint32_t max_seq_len, uint32_t head_dim)
{
    // Use shared memory so CPU can sync data
    size_t size = num_kv_heads * max_seq_len * head_dim * sizeof(float);
    return {
        impl_->create_buffer(size, true),  // K cache (shared memory)
        impl_->create_buffer(size, true)   // V cache (shared memory)
    };
}

Result<void> MetalCompute::kv_cache_append(
    MTL::Buffer* cache_k, MTL::Buffer* cache_v,
    MTL::Buffer* new_k, MTL::Buffer* new_v,
    uint32_t num_kv_heads, uint32_t head_dim,
    uint32_t current_len, uint32_t new_len, uint32_t max_seq_len)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("kv_cache_append_f16");
    if (!pipeline) {
        pipeline = impl_->get_pipeline("kv_cache_append");
        if (!pipeline) {
            return Error(ErrorCode::InternalError, "kv_cache_append pipeline not found");
        }
    }

    // Grid: [head_dim, new_len, num_kv_heads]
    MTL::Size grid_size = MTL::Size::Make(head_dim, new_len, num_kv_heads);
    MTL::Size threadgroup_size = MTL::Size::Make(32, 1, 1);

    // Append K to cache_k
    // Kernel signature: new_kv (FP32) at buffer(0), cache (FP16) at buffer(1)
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(new_k, 0, 0);     // Source: new K (FP32)
    encoder->setBuffer(cache_k, 0, 1);   // Dest: K cache (FP16)
    encoder->setBytes(&num_kv_heads, sizeof(num_kv_heads), 2);
    encoder->setBytes(&head_dim, sizeof(head_dim), 3);
    encoder->setBytes(&current_len, sizeof(current_len), 4);
    encoder->setBytes(&new_len, sizeof(new_len), 5);
    encoder->setBytes(&max_seq_len, sizeof(max_seq_len), 6);
    encoder->dispatchThreads(grid_size, threadgroup_size);

    // Append V to cache_v
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(new_v, 0, 0);     // Source: new V (FP32)
    encoder->setBuffer(cache_v, 0, 1);   // Dest: V cache (FP16)
    encoder->setBytes(&num_kv_heads, sizeof(num_kv_heads), 2);
    encoder->setBytes(&head_dim, sizeof(head_dim), 3);
    encoder->setBytes(&current_len, sizeof(current_len), 4);
    encoder->setBytes(&new_len, sizeof(new_len), 5);
    encoder->setBytes(&max_seq_len, sizeof(max_seq_len), 6);
    encoder->dispatchThreads(grid_size, threadgroup_size);

    return {};
}

// -----------------------------------------------------------------------------
// Multi-Head Attention
// -----------------------------------------------------------------------------

Result<void> MetalCompute::multihead_attention(
    MTL::Buffer* Q, MTL::Buffer* K, MTL::Buffer* V, MTL::Buffer* output,
    uint32_t num_heads, uint32_t num_kv_heads,
    uint32_t seq_q, uint32_t seq_kv, uint32_t head_dim, float scale,
    uint32_t max_seq)
{
    // If max_seq is 0, use seq_kv as stride (backwards compat for decode)
    if (max_seq == 0) max_seq = seq_kv;
    auto* encoder = impl_->get_encoder();

    // Select kernel based on seq_q:
    // - seq_q == 1: decode kernel (single query, FP16 KV cache)
    // - seq_q > 1: prefill kernel (batched queries)
    if (seq_q == 1) {
        // Decode path: kernel selection priority:
        // 1. llama.cpp-style flash attention (NEW - highest performance)
        // 2. Simdgroup flash attention (previous best)
        // 3. Legacy multihead_attention_decode_f16kv

        const char* kernel_name = nullptr;
        MTL::ComputePipelineState* pipeline = nullptr;

        // Try llama.cpp-style flash attention kernels first (highest performance)
        if (head_dim == 64) {
            kernel_name = "flash_attention_decode_d64";
            pipeline = impl_->get_pipeline(kernel_name);
        } else if (head_dim == 128) {
            kernel_name = "flash_attention_decode_d128";
            pipeline = impl_->get_pipeline(kernel_name);
        }

        // Fall back to older simdgroup flash attention
        if (!pipeline) {
            if (head_dim == 64) {
                kernel_name = "simdgroup_flash_attention_decode_f16kv_d64";
                pipeline = impl_->get_pipeline(kernel_name);
            } else if (head_dim == 128) {
                kernel_name = "simdgroup_flash_attention_decode_f16kv_d128";
                pipeline = impl_->get_pipeline(kernel_name);
            }
        }

        // Fall back to legacy kernel if flash attention not available
        if (!pipeline) {
            kernel_name = "multihead_attention_decode_f16kv";
            pipeline = impl_->get_pipeline(kernel_name);
        }

        // Last resort: try non-f16kv version
        if (!pipeline) {
            kernel_name = "multihead_attention_decode";
            pipeline = impl_->get_pipeline(kernel_name);
            if (!pipeline) {
                return Error(ErrorCode::InternalError, "multihead_attention_decode kernel not found");
            }
        }

        // Debug: log which kernel was selected (only log once)
        static bool logged_kernel = false;
        if (!logged_kernel) {
            GRANITE_LOG_INFO("Attention kernel selected: {} (head_dim={})", kernel_name, head_dim);
            logged_kernel = true;
        }

        encoder->setComputePipelineState(pipeline);
        encoder->setBuffer(Q, 0, 0);
        encoder->setBuffer(K, 0, 1);
        encoder->setBuffer(V, 0, 2);
        encoder->setBuffer(output, 0, 3);
        encoder->setBytes(&num_heads, sizeof(num_heads), 4);
        encoder->setBytes(&num_kv_heads, sizeof(num_kv_heads), 5);
        encoder->setBytes(&seq_kv, sizeof(seq_kv), 6);        // seq_kv at buffer 6
        encoder->setBytes(&head_dim, sizeof(head_dim), 7);    // head_dim at buffer 7
        encoder->setBytes(&scale, sizeof(scale), 8);          // scale at buffer 8

        // One threadgroup per head
        MTL::Size grid_size = MTL::Size::Make(num_heads, 1, 1);
        MTL::Size threadgroup_size = MTL::Size::Make(128, 1, 1);
        encoder->dispatchThreadgroups(grid_size, threadgroup_size);
    } else {
        // Prefill path: use optimized flash attention (direct global V loads)
        auto* pipeline = impl_->get_pipeline("flash_attention_prefill");
        if (!pipeline) {
            return Error(ErrorCode::InternalError, "flash_attention_prefill kernel not found");
        }

        encoder->setComputePipelineState(pipeline);
        encoder->setBuffer(Q, 0, 0);
        encoder->setBuffer(K, 0, 1);
        encoder->setBuffer(V, 0, 2);
        encoder->setBuffer(output, 0, 3);
        encoder->setBytes(&num_heads, sizeof(num_heads), 4);
        encoder->setBytes(&num_kv_heads, sizeof(num_kv_heads), 5);
        encoder->setBytes(&seq_q, sizeof(seq_q), 6);
        encoder->setBytes(&seq_kv, sizeof(seq_kv), 7);
        encoder->setBytes(&head_dim, sizeof(head_dim), 8);
        encoder->setBytes(&scale, sizeof(scale), 9);
        uint32_t start_pos = 0;  // For prefill, start_pos is 0
        encoder->setBytes(&start_pos, sizeof(start_pos), 10);
        encoder->setBytes(&max_seq, sizeof(max_seq), 11);  // KV cache stride

        // Threadgroup memory layout (no V buffer, half scores):
        // sq[Q_TILE * DK] = 8 * 64 = 512 halfs = 1024 bytes
        // so[Q_TILE * DK] = 8 * 64 = 512 halfs = 1024 bytes
        // ss[Q_TILE * K_TILE] = 8 * 64 = 512 halfs = 1024 bytes
        // Total: 3072 bytes
        constexpr uint32_t Q_TILE = 8;
        constexpr uint32_t K_TILE = 64;  // Matches llama.cpp's NCPSG for optimal performance
        constexpr uint32_t DK = 64;
        constexpr uint32_t threadgroup_mem_size = (Q_TILE * DK * 2 + Q_TILE * DK * 2 + Q_TILE * K_TILE * 2);
        encoder->setThreadgroupMemoryLength(threadgroup_mem_size, 0);

        // One threadgroup per (head, query_block) pair
        // Each threadgroup handles Q_TILE=8 query positions
        uint32_t num_q_blocks = (seq_q + Q_TILE - 1) / Q_TILE;
        MTL::Size grid_size = MTL::Size::Make(num_heads, num_q_blocks, 1);
        MTL::Size threadgroup_size = MTL::Size::Make(128, 1, 1);
        encoder->dispatchThreadgroups(grid_size, threadgroup_size);
    }

    return {};
}

// -----------------------------------------------------------------------------
// Embedding Lookup
// -----------------------------------------------------------------------------

Result<void> MetalCompute::embedding_lookup(
    MTL::Buffer* token_ids, MTL::Buffer* embeddings, MTL::Buffer* output,
    uint32_t num_tokens, uint32_t hidden_dim, uint32_t vocab_size)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("embedding_lookup");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "embedding_lookup pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(token_ids, 0, 0);
    encoder->setBuffer(embeddings, 0, 1);
    encoder->setBuffer(output, 0, 2);
    // Note: kernel uses buffer(3)=hidden_dim, buffer(4)=vocab_size
    // num_tokens is encoded in grid size, not as a parameter
    encoder->setBytes(&hidden_dim, sizeof(hidden_dim), 3);
    encoder->setBytes(&vocab_size, sizeof(vocab_size), 4);

    MTL::Size grid_size = MTL::Size::Make(hidden_dim, num_tokens, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(256, 1, 1);
    encoder->dispatchThreads(grid_size, threadgroup_size);

    return {};
}

// -----------------------------------------------------------------------------
// Fused QKV Projection
// -----------------------------------------------------------------------------

Result<void> MetalCompute::fused_qkv_matvec_q4k(
    MTL::Buffer* x,
    MTL::Buffer* Wq, MTL::Buffer* Wk, MTL::Buffer* Wv,
    MTL::Buffer* yq, MTL::Buffer* yk, MTL::Buffer* yv,
    uint32_t K, uint32_t Nq, uint32_t Nkv)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("fused_qkv_matvec_q4k");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "fused_qkv_matvec_q4k pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(x, 0, 0);
    encoder->setBuffer(Wq, 0, 1);
    encoder->setBuffer(Wk, 0, 2);
    encoder->setBuffer(Wv, 0, 3);
    encoder->setBuffer(yq, 0, 4);
    encoder->setBuffer(yk, 0, 5);
    encoder->setBuffer(yv, 0, 6);
    encoder->setBytes(&K, sizeof(K), 7);
    encoder->setBytes(&Nq, sizeof(Nq), 8);
    encoder->setBytes(&Nkv, sizeof(Nkv), 9);

    // Calculate threadgroups: 2 SIMD groups per TG, 2 rows per SIMD = 4 rows per TG
    constexpr uint32_t rows_per_tg = 4;
    uint32_t q_threadgroups = (Nq + rows_per_tg - 1) / rows_per_tg;
    uint32_t kv_threadgroups = (Nkv + rows_per_tg - 1) / rows_per_tg;
    uint32_t total_threadgroups = q_threadgroups + 2 * kv_threadgroups;  // Q + K + V

    encoder->setBytes(&q_threadgroups, sizeof(q_threadgroups), 10);
    encoder->setBytes(&kv_threadgroups, sizeof(kv_threadgroups), 11);

    MTL::Size grid_size = MTL::Size::Make(total_threadgroups, 1, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(64, 1, 1);  // 2 SIMD groups * 32 threads
    encoder->dispatchThreadgroups(grid_size, threadgroup_size);

    return {};
}

// -----------------------------------------------------------------------------
// Tree Attention (Speculative Decoding)
// -----------------------------------------------------------------------------

Result<void> MetalCompute::attention_tree(
    MTL::Buffer* Q,
    MTL::Buffer* K_cache, MTL::Buffer* V_cache,
    MTL::Buffer* K_tree, MTL::Buffer* V_tree,
    MTL::Buffer* parent_indices, MTL::Buffer* output,
    uint32_t num_heads, uint32_t num_kv_heads,
    uint32_t num_nodes, uint32_t cache_len,
    uint32_t head_dim, float scale)
{
    auto* encoder = impl_->get_encoder();

    if (cache_len > 0) {
        // Use full tree attention with cache
        auto* pipeline = impl_->get_pipeline("attention_tree_f16kv");
        if (!pipeline) {
            return Error(ErrorCode::InternalError, "attention_tree_f16kv pipeline not found");
        }

        encoder->setComputePipelineState(pipeline);
        encoder->setBuffer(Q, 0, 0);
        encoder->setBuffer(K_cache, 0, 1);
        encoder->setBuffer(V_cache, 0, 2);
        encoder->setBuffer(K_tree, 0, 3);
        encoder->setBuffer(V_tree, 0, 4);
        encoder->setBuffer(parent_indices, 0, 5);
        encoder->setBuffer(output, 0, 6);
        encoder->setBytes(&num_heads, sizeof(num_heads), 7);
        encoder->setBytes(&num_kv_heads, sizeof(num_kv_heads), 8);
        encoder->setBytes(&num_nodes, sizeof(num_nodes), 9);
        encoder->setBytes(&cache_len, sizeof(cache_len), 10);
        encoder->setBytes(&head_dim, sizeof(head_dim), 11);
        encoder->setBytes(&scale, sizeof(scale), 12);

        // Each threadgroup handles one (head, node) pair
        MTL::Size grid_size = MTL::Size::Make(num_heads, num_nodes, 1);
        MTL::Size threadgroup_size = MTL::Size::Make(128, 1, 1);
        encoder->dispatchThreadgroups(grid_size, threadgroup_size);
    } else {
        // Use simpler no-context version
        auto* pipeline = impl_->get_pipeline("attention_tree_nocontext_f16kv");
        if (!pipeline) {
            return Error(ErrorCode::InternalError, "attention_tree_nocontext_f16kv pipeline not found");
        }

        encoder->setComputePipelineState(pipeline);
        encoder->setBuffer(Q, 0, 0);
        encoder->setBuffer(K_tree, 0, 1);
        encoder->setBuffer(V_tree, 0, 2);
        encoder->setBuffer(parent_indices, 0, 3);
        encoder->setBuffer(output, 0, 4);
        encoder->setBytes(&num_heads, sizeof(num_heads), 5);
        encoder->setBytes(&num_kv_heads, sizeof(num_kv_heads), 6);
        encoder->setBytes(&num_nodes, sizeof(num_nodes), 7);
        encoder->setBytes(&head_dim, sizeof(head_dim), 8);
        encoder->setBytes(&scale, sizeof(scale), 9);

        MTL::Size grid_size = MTL::Size::Make(num_heads, num_nodes, 1);
        MTL::Size threadgroup_size = MTL::Size::Make(128, 1, 1);
        encoder->dispatchThreadgroups(grid_size, threadgroup_size);
    }

    return {};
}

// -----------------------------------------------------------------------------
// Paged Attention (Continuous Batching)
// -----------------------------------------------------------------------------

Result<void> MetalCompute::paged_attention_decode(
    MTL::Buffer* Q,
    MTL::Buffer* K_cache, MTL::Buffer* V_cache,
    MTL::Buffer* block_table, MTL::Buffer* output,
    uint32_t num_heads, uint32_t num_kv_heads,
    uint32_t seq_len, uint32_t head_dim,
    uint32_t block_size, float scale)
{
    auto* encoder = impl_->get_encoder();

    // Use paged flash attention for longer sequences or by default for efficiency
    // Flash attention uses online softmax - no O(seq_len) memory requirement
    // Old kernel limited to 4096 tokens due to threadgroup memory
    // v2 kernel uses per-simdgroup accumulators + NE/NL parallelism (llama.cpp style)
    const char* kernel_name = nullptr;
    if (head_dim == 64) {
        kernel_name = "paged_flash_attention_decode_d64_v2";
    } else if (head_dim == 128) {
        kernel_name = "paged_flash_attention_decode_d128";
    } else {
        // Fall back to old kernel for unsupported head dims (limited to 4096 tokens)
        if (seq_len > 4096) {
            return Error(ErrorCode::InvalidArgument,
                "paged_attention_decode: seq_len > 4096 requires head_dim 64 or 128");
        }
        kernel_name = "paged_attention_decode";
    }

    auto* pipeline = impl_->get_pipeline(kernel_name);
    if (!pipeline) {
        // Fall back to old kernel if flash kernel not available
        pipeline = impl_->get_pipeline("paged_attention_decode");
        if (!pipeline) {
            return Error(ErrorCode::InternalError, "paged_attention_decode pipeline not found");
        }
        if (seq_len > 4096) {
            return Error(ErrorCode::InvalidArgument,
                "paged_attention_decode: seq_len > 4096 requires flash attention kernels");
        }

        // Use old kernel (limited to 4096)
        encoder->setComputePipelineState(pipeline);
        encoder->setBuffer(Q, 0, 0);
        encoder->setBuffer(K_cache, 0, 1);
        encoder->setBuffer(V_cache, 0, 2);
        encoder->setBuffer(block_table, 0, 3);
        encoder->setBuffer(output, 0, 4);
        encoder->setBytes(&num_heads, sizeof(uint32_t), 5);
        encoder->setBytes(&num_kv_heads, sizeof(uint32_t), 6);
        encoder->setBytes(&seq_len, sizeof(uint32_t), 7);
        encoder->setBytes(&head_dim, sizeof(uint32_t), 8);
        encoder->setBytes(&block_size, sizeof(uint32_t), 9);
        uint32_t kv_stride = 0;
        encoder->setBytes(&kv_stride, sizeof(uint32_t), 10);
        encoder->setBytes(&scale, sizeof(float), 11);

        MTL::Size grid_size = MTL::Size::Make(num_heads, 1, 1);
        MTL::Size threadgroup_size = MTL::Size::Make(128, 1, 1);
        encoder->dispatchThreadgroups(grid_size, threadgroup_size);
        return {};
    }

    // Use flash attention kernel (supports arbitrary seq_len)
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(Q, 0, 0);
    encoder->setBuffer(K_cache, 0, 1);
    encoder->setBuffer(V_cache, 0, 2);
    encoder->setBuffer(block_table, 0, 3);
    encoder->setBuffer(output, 0, 4);
    encoder->setBytes(&num_heads, sizeof(uint32_t), 5);
    encoder->setBytes(&num_kv_heads, sizeof(uint32_t), 6);
    encoder->setBytes(&seq_len, sizeof(uint32_t), 7);
    encoder->setBytes(&head_dim, sizeof(uint32_t), 8);
    encoder->setBytes(&block_size, sizeof(uint32_t), 9);
    encoder->setBytes(&scale, sizeof(float), 10);

    // One threadgroup per head, 128 threads (4 simdgroups)
    MTL::Size grid_size = MTL::Size::Make(num_heads, 1, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(128, 1, 1);
    encoder->dispatchThreadgroups(grid_size, threadgroup_size);

    return {};
}

Result<void> MetalCompute::paged_kv_cache_append(
    MTL::Buffer* new_k, MTL::Buffer* new_v,
    MTL::Buffer* K_cache, MTL::Buffer* V_cache,
    MTL::Buffer* block_table,
    uint32_t num_kv_heads, uint32_t head_dim,
    uint32_t start_pos, uint32_t new_len, uint32_t block_size)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("paged_kv_cache_append");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "paged_kv_cache_append pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(new_k, 0, 0);
    encoder->setBuffer(new_v, 0, 1);
    encoder->setBuffer(K_cache, 0, 2);
    encoder->setBuffer(V_cache, 0, 3);
    encoder->setBuffer(block_table, 0, 4);
    encoder->setBytes(&num_kv_heads, sizeof(uint32_t), 5);
    encoder->setBytes(&head_dim, sizeof(uint32_t), 6);
    encoder->setBytes(&start_pos, sizeof(uint32_t), 7);
    encoder->setBytes(&new_len, sizeof(uint32_t), 8);
    encoder->setBytes(&block_size, sizeof(uint32_t), 9);

    // Grid: [head_dim, new_len, num_kv_heads]
    MTL::Size grid_size = MTL::Size::Make(head_dim, new_len, num_kv_heads);
    MTL::Size threadgroup_size = MTL::Size::Make(32, 1, 1);
    encoder->dispatchThreads(grid_size, threadgroup_size);

    return {};
}

Result<void> MetalCompute::batched_paged_attention_decode(
    MTL::Buffer* Q,
    MTL::Buffer* K_cache, MTL::Buffer* V_cache,
    MTL::Buffer* block_tables, MTL::Buffer* seq_lens,
    MTL::Buffer* output,
    uint32_t batch_size, uint32_t num_heads, uint32_t num_kv_heads,
    uint32_t head_dim, uint32_t block_size,
    uint32_t max_blocks_per_seq, float scale)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("batched_paged_attention_decode");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "batched_paged_attention_decode pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(Q, 0, 0);
    encoder->setBuffer(K_cache, 0, 1);
    encoder->setBuffer(V_cache, 0, 2);
    encoder->setBuffer(block_tables, 0, 3);
    encoder->setBuffer(seq_lens, 0, 4);
    encoder->setBuffer(output, 0, 5);
    encoder->setBytes(&batch_size, sizeof(uint32_t), 6);
    encoder->setBytes(&num_heads, sizeof(uint32_t), 7);
    encoder->setBytes(&num_kv_heads, sizeof(uint32_t), 8);
    encoder->setBytes(&head_dim, sizeof(uint32_t), 9);
    encoder->setBytes(&block_size, sizeof(uint32_t), 10);
    encoder->setBytes(&max_blocks_per_seq, sizeof(uint32_t), 11);
    encoder->setBytes(&scale, sizeof(float), 12);

    // Grid: [num_heads, batch_size]
    MTL::Size grid_size = MTL::Size::Make(num_heads, batch_size, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(128, 1, 1);
    encoder->dispatchThreadgroups(grid_size, threadgroup_size);

    return {};
}


// =============================================================================
// SECTION 8: Batched Encoding API
// =============================================================================

void* MetalCompute::begin_batch() {
    // Return raw encoder for tight-loop encoding
    return static_cast<void*>(impl_->get_encoder());
}

void MetalCompute::end_batch() {
    sync();
}

void* MetalCompute::get_pipeline(const char* name) {
    return static_cast<void*>(impl_->get_pipeline(name));
}


// =============================================================================
// SECTION 9: Global Singleton
// =============================================================================

static std::unique_ptr<MetalCompute> g_metal_compute;
static std::once_flag g_metal_compute_init;

MetalCompute* get_metal_compute() {
    std::call_once(g_metal_compute_init, []() {
        g_metal_compute = std::make_unique<MetalCompute>();

        MTL::Device* device = MTL::CreateSystemDefaultDevice();
        if (device) {
            auto result = g_metal_compute->initialize(device);
            if (!result.ok()) {
                GRANITE_LOG_ERROR("Failed to initialize MetalCompute: {}",
                                 result.error().message());
                g_metal_compute.reset();
            }
        }
    });

    return g_metal_compute.get();
}

}  // namespace granite

#endif  // GRANITE_HAS_METAL

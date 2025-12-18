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

    // SIMD-optimized matmul dispatch for prefill
    // Uses dispatchThreadgroups with specialized SIMD config
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

        // SIMD-optimized: 8 SIMD groups, each handles 2 output columns
        constexpr uint32_t simd_groups = 8;
        constexpr uint32_t cols_per_tg = 2 * simd_groups;  // 16 columns per TG
        uint32_t tg_x = (N + cols_per_tg - 1) / cols_per_tg;
        MTL::Size grid_size = MTL::Size::Make(tg_x, M, 1);
        MTL::Size threadgroup_size = MTL::Size::Make(32 * simd_groups, 1, 1);
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

        NS::String* source = NS::String::string(METAL_SHADER_SOURCE, NS::UTF8StringEncoding);
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
            "matvec_q4k", "matmul_q4k", "matvec_f16", "matvec_f32",
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
            "rms_norm", "rms_norm_f16", "silu", "elementwise_mul", "rope",
            "elementwise_add", "rope_multihead", "softmax_row",
            // Basic attention
            "attention_decode", "kv_cache_append", "kv_cache_append_f16",
            "multihead_attention_decode", "multihead_attention_decode_f16kv",
            "embedding_lookup",
            // Fused kernels
            "silu_mul",
            "rms_norm_matvec_q4k", "rms_norm_matvec_f16",
            "rms_norm_matvec_q8_0", "rms_norm_matvec_q4_0", "rms_norm_matvec_iq4_nl",
            "rms_norm_matvec_iq4_xs", "rms_norm_matvec_iq3_s", "rms_norm_matvec_q6_k",
            "rms_norm_matvec_q5_k", "rms_norm_matvec_q3_k", "rms_norm_matvec_q2_k",
            // Phase 2 fused kernels
            "rms_norm_dual_matvec_q4k", "rms_norm_dual_matvec_q3k", "rms_norm_dual_matvec_q2k",
            "matvec_residual_q4k", "matvec_residual_q3k", "matvec_residual_q2k",
            // Fused QKV
            "fused_qkv_matvec_q4k",
            // Flash attention
            "flash_attention_decode",
            // Prefill attention
            "attention_prefill", "attention_prefill_f16kv",
            // Tree attention (speculative decoding)
            "attention_tree_f16kv", "attention_tree_nocontext_f16kv",
            // Paged attention (continuous batching)
            "paged_attention_decode", "paged_kv_cache_append",
            "batched_paged_attention_decode"
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
            GRANITE_LOG_DEBUG("Created Metal pipeline: {}", name);
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
    return impl_->dispatch_matmul_simd("matmul_q4k", X, W, Y, M, K, N);
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
    return impl_->dispatch_matmul_simd("matmul_iq4_xs", X, W, Y, M, K, N);
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
    return impl_->dispatch_matmul_simd("matmul_iq3_s", X, W, Y, M, K, N);
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
    return impl_->dispatch_matvec("matvec_f16", x, W, y, K, N);
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
        // Try flash attention as fallback
        pipeline = impl_->get_pipeline("flash_attention_decode");
        if (!pipeline) {
            return Error(ErrorCode::InternalError, "attention_decode pipeline not found");
        }
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
    size_t size = num_kv_heads * max_seq_len * head_dim * sizeof(uint16_t);
    return {
        impl_->create_buffer(size, false),  // K cache (private memory)
        impl_->create_buffer(size, false)   // V cache (private memory)
    };
}

std::pair<MTL::Buffer*, MTL::Buffer*> MetalCompute::create_kv_cache_f32(
    uint32_t num_kv_heads, uint32_t max_seq_len, uint32_t head_dim)
{
    size_t size = num_kv_heads * max_seq_len * head_dim * sizeof(float);
    return {
        impl_->create_buffer(size, false),
        impl_->create_buffer(size, false)
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

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(cache_k, 0, 0);
    encoder->setBuffer(cache_v, 0, 1);
    encoder->setBuffer(new_k, 0, 2);
    encoder->setBuffer(new_v, 0, 3);
    encoder->setBytes(&num_kv_heads, sizeof(num_kv_heads), 4);
    encoder->setBytes(&head_dim, sizeof(head_dim), 5);
    encoder->setBytes(&current_len, sizeof(current_len), 6);
    encoder->setBytes(&new_len, sizeof(new_len), 7);
    encoder->setBytes(&max_seq_len, sizeof(max_seq_len), 8);

    // Grid: [head_dim, new_len, num_kv_heads]
    MTL::Size grid_size = MTL::Size::Make(head_dim, new_len, num_kv_heads);
    MTL::Size threadgroup_size = MTL::Size::Make(32, 1, 1);
    encoder->dispatchThreads(grid_size, threadgroup_size);

    // Also sync to ensure writes complete
    MTL::Size grid_size2 = MTL::Size::Make(head_dim, new_len, num_kv_heads);
    encoder->dispatchThreads(grid_size2, threadgroup_size);

    return {};
}

// -----------------------------------------------------------------------------
// Multi-Head Attention
// -----------------------------------------------------------------------------

Result<void> MetalCompute::multihead_attention(
    MTL::Buffer* Q, MTL::Buffer* K, MTL::Buffer* V, MTL::Buffer* output,
    uint32_t num_heads, uint32_t num_kv_heads,
    uint32_t seq_q, uint32_t seq_kv, uint32_t head_dim, float scale)
{
    auto* encoder = impl_->get_encoder();

    // Select appropriate kernel based on configuration
    const char* kernel_name = "multihead_attention_decode_f16kv";
    auto* pipeline = impl_->get_pipeline(kernel_name);
    if (!pipeline) {
        kernel_name = "multihead_attention_decode";
        pipeline = impl_->get_pipeline(kernel_name);
        if (!pipeline) {
            return Error(ErrorCode::InternalError, "multihead_attention kernel not found");
        }
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

    // One threadgroup per head
    MTL::Size grid_size = MTL::Size::Make(num_heads, seq_q, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(128, 1, 1);
    encoder->dispatchThreadgroups(grid_size, threadgroup_size);

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
    encoder->setBytes(&num_tokens, sizeof(num_tokens), 3);
    encoder->setBytes(&hidden_dim, sizeof(hidden_dim), 4);
    encoder->setBytes(&vocab_size, sizeof(vocab_size), 5);

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

    // Calculate threadgroups: 8 SIMD groups per TG, 2 rows per SIMD = 16 rows per TG
    constexpr uint32_t rows_per_tg = 16;
    uint32_t q_threadgroups = (Nq + rows_per_tg - 1) / rows_per_tg;
    uint32_t kv_threadgroups = (Nkv + rows_per_tg - 1) / rows_per_tg;
    uint32_t total_threadgroups = q_threadgroups + 2 * kv_threadgroups;  // Q + K + V

    encoder->setBytes(&q_threadgroups, sizeof(q_threadgroups), 10);
    encoder->setBytes(&kv_threadgroups, sizeof(kv_threadgroups), 11);

    MTL::Size grid_size = MTL::Size::Make(total_threadgroups, 1, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(256, 1, 1);  // 8 SIMD groups * 32 threads
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
    auto* pipeline = impl_->get_pipeline("paged_attention_decode");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "paged_attention_decode pipeline not found");
    }

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
    uint32_t kv_stride = 0;  // Not used directly, computed in kernel
    encoder->setBytes(&kv_stride, sizeof(uint32_t), 10);
    encoder->setBytes(&scale, sizeof(float), 11);

    // One threadgroup per head
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
// SECTION 8: Global Singleton
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

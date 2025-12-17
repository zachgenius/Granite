// MetalCompute - High-level GPU compute interface for LLM inference
// Manages shader compilation, pipeline states, and command buffer batching

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

namespace granite {

// Include embedded shader source
#include "kernels/metal_shaders.h"


// Implementation class
class MetalCompute::Impl {
public:
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
            current_command_buffer_->commit();
            current_command_buffer_->waitUntilCompleted();
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
        }
        return current_encoder_;
    }

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

        std::vector<std::string> kernels = {
            "matvec_q4k", "matmul_q4k", "matvec_f16", "matvec_f32",
            // Q8_0 kernels
            "matvec_q8_0", "matmul_q8_0",
            // Q4_0 kernels
            "matvec_q4_0", "matmul_q4_0",
            // IQ4_NL kernels
            "matvec_iq4_nl", "matmul_iq4_nl",
            // IQ4_XS kernels
            "matvec_iq4_xs", "matmul_iq4_xs",
            // IQ3_S kernels
            "matvec_iq3_s", "matmul_iq3_s",
            // Q6_K kernels
            "matvec_q6_k", "matmul_q6_k",
            // Q5_K kernels
            "matvec_q5_k", "matmul_q5_k",
            // Q3_K kernels
            "matvec_q3_k", "matmul_q3_k",
            // Q2_K kernels
            "matvec_q2_k", "matmul_q2_k",
            "rms_norm", "rms_norm_f16", "silu", "elementwise_mul", "rope",
            "elementwise_add", "rope_multihead", "softmax_row", "attention_decode",
            "kv_cache_append", "kv_cache_append_f16", "multihead_attention_decode",
            "multihead_attention_decode_f16kv", "embedding_lookup",
            // Fused kernels
            "silu_mul", "rms_norm_matvec_q4k", "rms_norm_matvec_f16",
            "rms_norm_matvec_q8_0", "rms_norm_matvec_q4_0", "rms_norm_matvec_iq4_nl",
            "rms_norm_matvec_iq4_xs", "rms_norm_matvec_iq3_s", "rms_norm_matvec_q6_k", "rms_norm_matvec_q5_k",
            "rms_norm_matvec_q3_k", "rms_norm_matvec_q2_k",
            // Phase 2 fused kernels (eliminates redundant computation)
            "rms_norm_dual_matvec_q4k", "rms_norm_dual_matvec_q3k", "rms_norm_dual_matvec_q2k",
            "matvec_residual_q4k", "matvec_residual_q3k", "matvec_residual_q2k",
            // Fused QKV attention projection
            "fused_qkv_matvec_q4k",
            // Flash attention
            "flash_attention_decode",
            // Prefill attention
            "attention_prefill", "attention_prefill_f16kv"
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

// MetalCompute public interface implementation
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

MTL::Buffer* MetalCompute::create_buffer(size_t size, bool shared) {
    return impl_->create_buffer(size, shared);
}

Result<void> MetalCompute::matvec_q4k(
    MTL::Buffer* x, MTL::Buffer* W, MTL::Buffer* y,
    uint32_t K, uint32_t N)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("matvec_q4k");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "matvec_q4k pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(x, 0, 0);
    encoder->setBuffer(W, 0, 1);
    encoder->setBuffer(y, 0, 2);
    encoder->setBytes(&K, sizeof(K), 3);
    encoder->setBytes(&N, sizeof(N), 4);

    // 8 SIMD groups per threadgroup (256 threads), each SIMD group handles 2 rows
    uint32_t simd_groups = 8;
    uint32_t rows_per_simd = 2;  // NR0_Q4K
    uint32_t rows_per_tg = simd_groups * rows_per_simd;  // 16 rows per threadgroup
    uint32_t num_threadgroups = (N + rows_per_tg - 1) / rows_per_tg;
    MTL::Size grid_size = MTL::Size::Make(num_threadgroups, 1, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(32 * simd_groups, 1, 1);  // 8 SIMD groups = 256 threads
    encoder->dispatchThreadgroups(grid_size, threadgroup_size);

    return {};
}

Result<void> MetalCompute::matmul_q4k(
    MTL::Buffer* X, MTL::Buffer* W, MTL::Buffer* Y,
    uint32_t M, uint32_t K, uint32_t N)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("matmul_q4k");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "matmul_q4k pipeline not found");
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

// =============================================================================
// Q8_0 Quantized Operations
// =============================================================================

Result<void> MetalCompute::matvec_q8_0(
    MTL::Buffer* x, MTL::Buffer* W, MTL::Buffer* y,
    uint32_t K, uint32_t N)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("matvec_q8_0");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "matvec_q8_0 pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(x, 0, 0);
    encoder->setBuffer(W, 0, 1);
    encoder->setBuffer(y, 0, 2);
    encoder->setBytes(&K, sizeof(K), 3);
    encoder->setBytes(&N, sizeof(N), 4);

    // 8 SIMD groups per threadgroup (256 threads), each SIMD group handles 2 rows
    uint32_t simd_groups = 8;
    uint32_t rows_per_simd = 2;  // NR0_Q8_0
    uint32_t rows_per_tg = simd_groups * rows_per_simd;  // 16 rows per threadgroup
    uint32_t num_threadgroups = (N + rows_per_tg - 1) / rows_per_tg;
    MTL::Size grid_size = MTL::Size::Make(num_threadgroups, 1, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(32 * simd_groups, 1, 1);  // 8 SIMD groups = 256 threads
    encoder->dispatchThreadgroups(grid_size, threadgroup_size);

    return {};
}

Result<void> MetalCompute::matmul_q8_0(
    MTL::Buffer* X, MTL::Buffer* W, MTL::Buffer* Y,
    uint32_t M, uint32_t K, uint32_t N)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("matmul_q8_0");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "matmul_q8_0 pipeline not found");
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

// =============================================================================
// Q4_0 Quantized Operations
// =============================================================================

Result<void> MetalCompute::matvec_q4_0(
    MTL::Buffer* x, MTL::Buffer* W, MTL::Buffer* y,
    uint32_t K, uint32_t N)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("matvec_q4_0");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "matvec_q4_0 pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(x, 0, 0);
    encoder->setBuffer(W, 0, 1);
    encoder->setBuffer(y, 0, 2);
    encoder->setBytes(&K, sizeof(K), 3);
    encoder->setBytes(&N, sizeof(N), 4);

    // 8 SIMD groups per threadgroup (256 threads), each SIMD group handles 2 rows
    uint32_t simd_groups = 8;
    uint32_t rows_per_simd = 2;  // NR0_Q4_0
    uint32_t rows_per_tg = simd_groups * rows_per_simd;  // 16 rows per threadgroup
    uint32_t num_threadgroups = (N + rows_per_tg - 1) / rows_per_tg;
    MTL::Size grid_size = MTL::Size::Make(num_threadgroups, 1, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(32 * simd_groups, 1, 1);  // 8 SIMD groups = 256 threads
    encoder->dispatchThreadgroups(grid_size, threadgroup_size);

    return {};
}

Result<void> MetalCompute::matmul_q4_0(
    MTL::Buffer* X, MTL::Buffer* W, MTL::Buffer* Y,
    uint32_t M, uint32_t K, uint32_t N)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("matmul_q4_0");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "matmul_q4_0 pipeline not found");
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

// =============================================================================
// IQ4_NL Quantized Operations (Non-linear 4-bit I-quant)
// =============================================================================

Result<void> MetalCompute::matvec_iq4_nl(
    MTL::Buffer* x, MTL::Buffer* W, MTL::Buffer* y,
    uint32_t K, uint32_t N)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("matvec_iq4_nl");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "matvec_iq4_nl pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(x, 0, 0);
    encoder->setBuffer(W, 0, 1);
    encoder->setBuffer(y, 0, 2);
    encoder->setBytes(&K, sizeof(K), 3);
    encoder->setBytes(&N, sizeof(N), 4);

    // 8 SIMD groups per threadgroup (256 threads), each SIMD group handles 2 rows
    uint32_t simd_groups = 8;
    uint32_t rows_per_simd = 2;  // NR0_IQ4_NL
    uint32_t rows_per_tg = simd_groups * rows_per_simd;  // 16 rows per threadgroup
    uint32_t num_threadgroups = (N + rows_per_tg - 1) / rows_per_tg;
    MTL::Size grid_size = MTL::Size::Make(num_threadgroups, 1, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(32 * simd_groups, 1, 1);  // 8 SIMD groups = 256 threads
    encoder->dispatchThreadgroups(grid_size, threadgroup_size);

    return {};
}

Result<void> MetalCompute::matmul_iq4_nl(
    MTL::Buffer* X, MTL::Buffer* W, MTL::Buffer* Y,
    uint32_t M, uint32_t K, uint32_t N)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("matmul_iq4_nl");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "matmul_iq4_nl pipeline not found");
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

// =============================================================================
// IQ4_XS Quantized Operations (Non-linear 4-bit I-quant with 256-element super-blocks)
// =============================================================================

Result<void> MetalCompute::matvec_iq4_xs(
    MTL::Buffer* x, MTL::Buffer* W, MTL::Buffer* y,
    uint32_t K, uint32_t N)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("matvec_iq4_xs");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "matvec_iq4_xs pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(x, 0, 0);
    encoder->setBuffer(W, 0, 1);
    encoder->setBuffer(y, 0, 2);
    encoder->setBytes(&K, sizeof(K), 3);
    encoder->setBytes(&N, sizeof(N), 4);

    // 8 SIMD groups per threadgroup (256 threads), each SIMD group handles 2 rows
    uint32_t simd_groups = 8;
    uint32_t rows_per_simd = 2;  // NR0_IQ4_XS
    uint32_t rows_per_tg = simd_groups * rows_per_simd;  // 16 rows per threadgroup
    uint32_t num_threadgroups = (N + rows_per_tg - 1) / rows_per_tg;
    MTL::Size grid_size = MTL::Size::Make(num_threadgroups, 1, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(32 * simd_groups, 1, 1);  // 8 SIMD groups = 256 threads
    encoder->dispatchThreadgroups(grid_size, threadgroup_size);

    return {};
}

Result<void> MetalCompute::matmul_iq4_xs(
    MTL::Buffer* X, MTL::Buffer* W, MTL::Buffer* Y,
    uint32_t M, uint32_t K, uint32_t N)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("matmul_iq4_xs");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "matmul_iq4_xs pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(X, 0, 0);
    encoder->setBuffer(W, 0, 1);
    encoder->setBuffer(Y, 0, 2);
    encoder->setBytes(&M, sizeof(M), 3);
    encoder->setBytes(&K, sizeof(K), 4);
    encoder->setBytes(&N, sizeof(N), 5);

    // Optimized dispatch: 8 SIMD groups × 32 threads = 256 threads per threadgroup
    // Each threadgroup handles 1 row × 16 columns (8 SIMD groups × 2 cols each)
    const uint32_t cols_per_threadgroup = 16;
    MTL::Size grid_size = MTL::Size::Make((N + cols_per_threadgroup - 1) / cols_per_threadgroup, M, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(256, 1, 1);
    encoder->dispatchThreadgroups(grid_size, threadgroup_size);

    return {};
}

// =============================================================================
// IQ3_S Quantized Operations (3-bit I-quant with 256-element super-blocks)
// =============================================================================

Result<void> MetalCompute::matvec_iq3_s(
    MTL::Buffer* x, MTL::Buffer* W, MTL::Buffer* y,
    uint32_t K, uint32_t N)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("matvec_iq3_s");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "matvec_iq3_s pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(x, 0, 0);
    encoder->setBuffer(W, 0, 1);
    encoder->setBuffer(y, 0, 2);
    encoder->setBytes(&K, sizeof(K), 3);
    encoder->setBytes(&N, sizeof(N), 4);

    // 8 SIMD groups per threadgroup (256 threads), each SIMD group handles 2 rows
    uint32_t simd_groups = 8;
    uint32_t rows_per_simd = 2;  // NR0_IQ3_S
    uint32_t rows_per_tg = simd_groups * rows_per_simd;  // 16 rows per threadgroup
    uint32_t num_threadgroups = (N + rows_per_tg - 1) / rows_per_tg;
    MTL::Size grid_size = MTL::Size::Make(num_threadgroups, 1, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(32 * simd_groups, 1, 1);  // 8 SIMD groups = 256 threads
    encoder->dispatchThreadgroups(grid_size, threadgroup_size);

    return {};
}

Result<void> MetalCompute::matmul_iq3_s(
    MTL::Buffer* X, MTL::Buffer* W, MTL::Buffer* Y,
    uint32_t M, uint32_t K, uint32_t N)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("matmul_iq3_s");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "matmul_iq3_s pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(X, 0, 0);
    encoder->setBuffer(W, 0, 1);
    encoder->setBuffer(Y, 0, 2);
    encoder->setBytes(&M, sizeof(M), 3);
    encoder->setBytes(&K, sizeof(K), 4);
    encoder->setBytes(&N, sizeof(N), 5);

    // Optimized dispatch: 8 SIMD groups × 32 threads = 256 threads per threadgroup
    // Each threadgroup handles 1 row × 16 columns (8 SIMD groups × 2 cols each)
    const uint32_t cols_per_threadgroup = 16;
    MTL::Size grid_size = MTL::Size::Make((N + cols_per_threadgroup - 1) / cols_per_threadgroup, M, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(256, 1, 1);
    encoder->dispatchThreadgroups(grid_size, threadgroup_size);

    return {};
}

// =============================================================================
// Q6_K Quantized Operations
// =============================================================================

Result<void> MetalCompute::matvec_q6_k(
    MTL::Buffer* x, MTL::Buffer* W, MTL::Buffer* y,
    uint32_t K, uint32_t N)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("matvec_q6_k");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "matvec_q6_k pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(x, 0, 0);
    encoder->setBuffer(W, 0, 1);
    encoder->setBuffer(y, 0, 2);
    encoder->setBytes(&K, sizeof(K), 3);
    encoder->setBytes(&N, sizeof(N), 4);

    // 8 SIMD groups per threadgroup (256 threads), each SIMD group handles 2 rows
    uint32_t simd_groups = 8;
    uint32_t rows_per_simd = 2;  // NR0_Q6K
    uint32_t rows_per_tg = simd_groups * rows_per_simd;  // 16 rows per threadgroup
    uint32_t num_threadgroups = (N + rows_per_tg - 1) / rows_per_tg;
    MTL::Size grid_size = MTL::Size::Make(num_threadgroups, 1, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(32 * simd_groups, 1, 1);  // 8 SIMD groups = 256 threads
    encoder->dispatchThreadgroups(grid_size, threadgroup_size);

    return {};
}

Result<void> MetalCompute::matmul_q6_k(
    MTL::Buffer* X, MTL::Buffer* W, MTL::Buffer* Y,
    uint32_t M, uint32_t K, uint32_t N)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("matmul_q6_k");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "matmul_q6_k pipeline not found");
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

// =============================================================================
// Q5_K Quantized Operations
// =============================================================================

Result<void> MetalCompute::matvec_q5_k(
    MTL::Buffer* x, MTL::Buffer* W, MTL::Buffer* y,
    uint32_t K, uint32_t N)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("matvec_q5_k");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "matvec_q5_k pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(x, 0, 0);
    encoder->setBuffer(W, 0, 1);
    encoder->setBuffer(y, 0, 2);
    encoder->setBytes(&K, sizeof(K), 3);
    encoder->setBytes(&N, sizeof(N), 4);

    // 8 SIMD groups per threadgroup (256 threads), each SIMD group handles 2 rows
    uint32_t simd_groups = 8;
    uint32_t rows_per_simd = 2;  // NR0_Q5K
    uint32_t rows_per_tg = simd_groups * rows_per_simd;  // 16 rows per threadgroup
    uint32_t num_threadgroups = (N + rows_per_tg - 1) / rows_per_tg;
    MTL::Size grid_size = MTL::Size::Make(num_threadgroups, 1, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(32 * simd_groups, 1, 1);  // 8 SIMD groups = 256 threads
    encoder->dispatchThreadgroups(grid_size, threadgroup_size);

    return {};
}

Result<void> MetalCompute::matmul_q5_k(
    MTL::Buffer* X, MTL::Buffer* W, MTL::Buffer* Y,
    uint32_t M, uint32_t K, uint32_t N)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("matmul_q5_k");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "matmul_q5_k pipeline not found");
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

// =============================================================================
// Q3_K Quantized Operations
// =============================================================================

Result<void> MetalCompute::matvec_q3_k(
    MTL::Buffer* x, MTL::Buffer* W, MTL::Buffer* y,
    uint32_t K, uint32_t N)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("matvec_q3_k");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "matvec_q3_k pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(x, 0, 0);
    encoder->setBuffer(W, 0, 1);
    encoder->setBuffer(y, 0, 2);
    encoder->setBytes(&K, sizeof(K), 3);
    encoder->setBytes(&N, sizeof(N), 4);

    // 8 SIMD groups per threadgroup (256 threads), each SIMD group handles 2 rows
    uint32_t simd_groups = 8;
    uint32_t rows_per_simd = 2;  // NR0_Q3K
    uint32_t rows_per_tg = simd_groups * rows_per_simd;  // 16 rows per threadgroup
    uint32_t num_threadgroups = (N + rows_per_tg - 1) / rows_per_tg;
    MTL::Size grid_size = MTL::Size::Make(num_threadgroups, 1, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(32 * simd_groups, 1, 1);  // 8 SIMD groups = 256 threads
    encoder->dispatchThreadgroups(grid_size, threadgroup_size);

    return {};
}

Result<void> MetalCompute::matmul_q3_k(
    MTL::Buffer* X, MTL::Buffer* W, MTL::Buffer* Y,
    uint32_t M, uint32_t K, uint32_t N)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("matmul_q3_k");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "matmul_q3_k pipeline not found");
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

// =============================================================================
// Q2_K Quantized Operations
// =============================================================================

Result<void> MetalCompute::matvec_q2_k(
    MTL::Buffer* x, MTL::Buffer* W, MTL::Buffer* y,
    uint32_t K, uint32_t N)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("matvec_q2_k");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "matvec_q2_k pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(x, 0, 0);
    encoder->setBuffer(W, 0, 1);
    encoder->setBuffer(y, 0, 2);
    encoder->setBytes(&K, sizeof(K), 3);
    encoder->setBytes(&N, sizeof(N), 4);

    // 8 SIMD groups per threadgroup (256 threads), each SIMD group handles 2 rows
    uint32_t simd_groups = 8;
    uint32_t rows_per_simd = 2;  // NR0_Q2K
    uint32_t rows_per_tg = simd_groups * rows_per_simd;  // 16 rows per threadgroup
    uint32_t num_threadgroups = (N + rows_per_tg - 1) / rows_per_tg;
    MTL::Size grid_size = MTL::Size::Make(num_threadgroups, 1, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(32 * simd_groups, 1, 1);  // 8 SIMD groups = 256 threads
    encoder->dispatchThreadgroups(grid_size, threadgroup_size);

    return {};
}

Result<void> MetalCompute::matmul_q2_k(
    MTL::Buffer* X, MTL::Buffer* W, MTL::Buffer* Y,
    uint32_t M, uint32_t K, uint32_t N)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("matmul_q2_k");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "matmul_q2_k pipeline not found");
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

    // Each threadgroup (32 threads = 1 SIMD group) handles one output row
    MTL::Size grid_size = MTL::Size::Make(N, 1, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(32, 1, 1);  // One SIMD group
    encoder->dispatchThreadgroups(grid_size, threadgroup_size);

    return {};
}

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

    // Use single threadgroup for reduction
    uint32_t tg_size = std::min((uint32_t)256, size);
    MTL::Size grid_size = MTL::Size::Make(tg_size, 1, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(tg_size, 1, 1);
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

    // Use single threadgroup for reduction
    uint32_t tg_size = std::min((uint32_t)256, size);
    MTL::Size grid_size = MTL::Size::Make(tg_size, 1, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(tg_size, 1, 1);
    encoder->dispatchThreads(grid_size, threadgroup_size);

    return {};
}

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
    MTL::Size threadgroup_size = MTL::Size::Make(std::min((uint32_t)256, size), 1, 1);
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
    MTL::Size threadgroup_size = MTL::Size::Make(std::min((uint32_t)256, size), 1, 1);
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
    MTL::Size threadgroup_size = MTL::Size::Make(16, 16, 1);
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
    MTL::Size threadgroup_size = MTL::Size::Make(std::min((uint32_t)256, size), 1, 1);
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

    uint32_t max_heads = std::max(num_heads_q, num_heads_k);
    MTL::Size grid_size = MTL::Size::Make(head_dim / 2, max_heads, seq_len);
    MTL::Size threadgroup_size = MTL::Size::Make(
        std::min((uint32_t)32, head_dim / 2),
        std::min((uint32_t)4, max_heads),
        std::min((uint32_t)2, seq_len)
    );
    encoder->dispatchThreads(grid_size, threadgroup_size);

    return {};
}

Result<void> MetalCompute::softmax(
    MTL::Buffer* x, uint32_t M, uint32_t N)
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

    MTL::Size grid_size = MTL::Size::Make(M, 1, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(std::min((uint32_t)256, M), 1, 1);
    encoder->dispatchThreads(grid_size, threadgroup_size);

    return {};
}

Result<void> MetalCompute::attention_single_head(
    MTL::Buffer* Q, MTL::Buffer* K, MTL::Buffer* V, MTL::Buffer* output,
    uint32_t seq_q, uint32_t seq_kv, uint32_t head_dim,
    uint32_t start_pos, float scale)
{
    // For decode (seq_q == 1), use optimized attention_decode kernel
    if (seq_q == 1) {
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
        encoder->setBytes(&seq_kv, sizeof(seq_kv), 4);
        encoder->setBytes(&head_dim, sizeof(head_dim), 5);
        encoder->setBytes(&scale, sizeof(scale), 6);

        // Use one threadgroup for the whole head
        MTL::Size grid_size = MTL::Size::Make(head_dim, 1, 1);
        MTL::Size threadgroup_size = MTL::Size::Make(head_dim, 1, 1);
        encoder->dispatchThreads(grid_size, threadgroup_size);

        return {};
    }

    // For prefill (seq_q > 1), we would need a different kernel
    // For now, return error (caller should fall back to CPU)
    return Error(ErrorCode::NotImplemented, "GPU prefill attention not implemented");
}

std::pair<MTL::Buffer*, MTL::Buffer*> MetalCompute::create_kv_cache(
    uint32_t num_kv_heads,
    uint32_t max_seq_len,
    uint32_t head_dim)
{
    // Use FP16 KV cache by default for better memory bandwidth
    size_t size = num_kv_heads * max_seq_len * head_dim * sizeof(uint16_t);  // FP16
    MTL::Buffer* k_cache = impl_->create_buffer(size, true);
    MTL::Buffer* v_cache = impl_->create_buffer(size, true);
    return {k_cache, v_cache};
}

std::pair<MTL::Buffer*, MTL::Buffer*> MetalCompute::create_kv_cache_f32(
    uint32_t num_kv_heads,
    uint32_t max_seq_len,
    uint32_t head_dim)
{
    size_t size = num_kv_heads * max_seq_len * head_dim * sizeof(float);
    MTL::Buffer* k_cache = impl_->create_buffer(size, true);
    MTL::Buffer* v_cache = impl_->create_buffer(size, true);
    return {k_cache, v_cache};
}

Result<void> MetalCompute::kv_cache_append(
    MTL::Buffer* cache_k,
    MTL::Buffer* cache_v,
    MTL::Buffer* new_k,
    MTL::Buffer* new_v,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t current_len,
    uint32_t new_len,
    uint32_t max_seq_len)
{
    auto* encoder = impl_->get_encoder();

    // Use FP16 version by default (float -> half conversion on append)
    auto* pipeline = impl_->get_pipeline("kv_cache_append_f16");
    if (!pipeline) {
        // Fallback to FP32 version
        pipeline = impl_->get_pipeline("kv_cache_append");
        if (!pipeline) {
            return Error(ErrorCode::InternalError, "kv_cache_append pipeline not found");
        }
    }

    // Append K
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(new_k, 0, 0);
    encoder->setBuffer(cache_k, 0, 1);
    encoder->setBytes(&num_kv_heads, sizeof(num_kv_heads), 2);
    encoder->setBytes(&head_dim, sizeof(head_dim), 3);
    encoder->setBytes(&current_len, sizeof(current_len), 4);
    encoder->setBytes(&new_len, sizeof(new_len), 5);
    encoder->setBytes(&max_seq_len, sizeof(max_seq_len), 6);

    MTL::Size grid_size = MTL::Size::Make(head_dim, new_len, num_kv_heads);
    MTL::Size threadgroup_size = MTL::Size::Make(
        std::min((uint32_t)64, head_dim),
        1,
        1
    );
    encoder->dispatchThreads(grid_size, threadgroup_size);

    // Append V
    encoder->setBuffer(new_v, 0, 0);
    encoder->setBuffer(cache_v, 0, 1);
    encoder->dispatchThreads(grid_size, threadgroup_size);

    return {};
}

Result<void> MetalCompute::multihead_attention(
    MTL::Buffer* Q,
    MTL::Buffer* K,
    MTL::Buffer* V,
    MTL::Buffer* output,
    uint32_t num_heads,
    uint32_t num_kv_heads,
    uint32_t seq_q,
    uint32_t seq_kv,
    uint32_t head_dim,
    float scale)
{
    auto* encoder = impl_->get_encoder();

    // For decode (seq_q == 1), use multihead_attention_decode kernel with FP16 KV
    if (seq_q == 1) {
        // Try FP16 KV version first (matches our FP16 KV cache)
        auto* pipeline = impl_->get_pipeline("multihead_attention_decode_f16kv");
        if (!pipeline) {
            // Fallback to FP32 version
            pipeline = impl_->get_pipeline("multihead_attention_decode");
            if (!pipeline) {
                return Error(ErrorCode::InternalError, "multihead_attention_decode pipeline not found");
            }
        }

        encoder->setComputePipelineState(pipeline);
        encoder->setBuffer(Q, 0, 0);
        encoder->setBuffer(K, 0, 1);
        encoder->setBuffer(V, 0, 2);
        encoder->setBuffer(output, 0, 3);
        encoder->setBytes(&num_heads, sizeof(num_heads), 4);
        encoder->setBytes(&num_kv_heads, sizeof(num_kv_heads), 5);
        encoder->setBytes(&seq_kv, sizeof(seq_kv), 6);
        encoder->setBytes(&head_dim, sizeof(head_dim), 7);
        encoder->setBytes(&scale, sizeof(scale), 8);

        uint32_t threads_per_group = std::min((uint32_t)128, std::max((uint32_t)32, head_dim));
        MTL::Size grid_size = MTL::Size::Make(num_heads, 1, 1);
        MTL::Size threadgroup_size = MTL::Size::Make(threads_per_group, 1, 1);
        encoder->dispatchThreadgroups(grid_size, threadgroup_size);

        return {};
    }

    // For prefill (seq_q > 1), use attention_prefill kernel
    // Try FP16 KV version first
    auto* pipeline = impl_->get_pipeline("attention_prefill_f16kv");
    if (!pipeline) {
        // Fallback to FP32 version
        pipeline = impl_->get_pipeline("attention_prefill");
        if (!pipeline) {
            return Error(ErrorCode::InternalError, "attention_prefill pipeline not found");
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
    uint32_t start_pos = 0;  // For prefill, start_pos is 0
    encoder->setBytes(&start_pos, sizeof(start_pos), 10);

    // Each threadgroup handles one (head, query_position) pair
    // Grid: [num_heads, seq_q]
    MTL::Size grid_size = MTL::Size::Make(num_heads, seq_q, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(128, 1, 1);  // 128 threads per threadgroup
    encoder->dispatchThreadgroups(grid_size, threadgroup_size);

    return {};
}

Result<void> MetalCompute::embedding_lookup(
    MTL::Buffer* token_ids,
    MTL::Buffer* embeddings,
    MTL::Buffer* output,
    uint32_t num_tokens,
    uint32_t hidden_dim,
    uint32_t vocab_size)
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
    encoder->setBytes(&hidden_dim, sizeof(hidden_dim), 3);
    encoder->setBytes(&vocab_size, sizeof(vocab_size), 4);

    // 2D grid: [hidden_dim, num_tokens]
    MTL::Size grid_size = MTL::Size::Make(hidden_dim, num_tokens, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(std::min((uint32_t)256, hidden_dim), 1, 1);
    encoder->dispatchThreads(grid_size, threadgroup_size);

    return {};
}

// =============================================================================
// Fused Kernel Dispatch Functions
// =============================================================================

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
    MTL::Size threadgroup_size = MTL::Size::Make(std::min((uint32_t)256, size), 1, 1);
    encoder->dispatchThreads(grid_size, threadgroup_size);

    return {};
}

Result<void> MetalCompute::rms_norm_matvec_q4k(
    MTL::Buffer* x, MTL::Buffer* norm_weight, MTL::Buffer* W, MTL::Buffer* y,
    uint32_t K, uint32_t N, float eps)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("rms_norm_matvec_q4k");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "rms_norm_matvec_q4k pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(x, 0, 0);
    encoder->setBuffer(norm_weight, 0, 1);
    encoder->setBuffer(W, 0, 2);
    encoder->setBuffer(y, 0, 3);
    encoder->setBytes(&K, sizeof(K), 4);
    encoder->setBytes(&N, sizeof(N), 5);
    encoder->setBytes(&eps, sizeof(eps), 6);

    // 8 SIMD groups per threadgroup, each handles one output row
    uint32_t rows_per_tg = 8;
    uint32_t num_threadgroups = (N + rows_per_tg - 1) / rows_per_tg;
    MTL::Size grid_size = MTL::Size::Make(num_threadgroups, 1, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(256, 1, 1);  // 8 SIMD groups = 256 threads
    encoder->dispatchThreadgroups(grid_size, threadgroup_size);

    return {};
}

Result<void> MetalCompute::rms_norm_matvec_f16(
    MTL::Buffer* x, MTL::Buffer* norm_weight, MTL::Buffer* W, MTL::Buffer* y,
    uint32_t K, uint32_t N, float eps)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("rms_norm_matvec_f16");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "rms_norm_matvec_f16 pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(x, 0, 0);
    encoder->setBuffer(norm_weight, 0, 1);
    encoder->setBuffer(W, 0, 2);
    encoder->setBuffer(y, 0, 3);
    encoder->setBytes(&K, sizeof(K), 4);
    encoder->setBytes(&N, sizeof(N), 5);
    encoder->setBytes(&eps, sizeof(eps), 6);

    uint32_t rows_per_tg = 8;
    uint32_t num_threadgroups = (N + rows_per_tg - 1) / rows_per_tg;
    MTL::Size grid_size = MTL::Size::Make(num_threadgroups, 1, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(256, 1, 1);
    encoder->dispatchThreadgroups(grid_size, threadgroup_size);

    return {};
}

Result<void> MetalCompute::rms_norm_matvec_q8_0(
    MTL::Buffer* x, MTL::Buffer* norm_weight, MTL::Buffer* W, MTL::Buffer* y,
    uint32_t K, uint32_t N, float eps)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("rms_norm_matvec_q8_0");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "rms_norm_matvec_q8_0 pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(x, 0, 0);
    encoder->setBuffer(norm_weight, 0, 1);
    encoder->setBuffer(W, 0, 2);
    encoder->setBuffer(y, 0, 3);
    encoder->setBytes(&K, sizeof(K), 4);
    encoder->setBytes(&N, sizeof(N), 5);
    encoder->setBytes(&eps, sizeof(eps), 6);

    uint32_t rows_per_tg = 8;
    uint32_t num_threadgroups = (N + rows_per_tg - 1) / rows_per_tg;
    MTL::Size grid_size = MTL::Size::Make(num_threadgroups, 1, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(256, 1, 1);
    encoder->dispatchThreadgroups(grid_size, threadgroup_size);

    return {};
}

Result<void> MetalCompute::rms_norm_matvec_q4_0(
    MTL::Buffer* x, MTL::Buffer* norm_weight, MTL::Buffer* W, MTL::Buffer* y,
    uint32_t K, uint32_t N, float eps)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("rms_norm_matvec_q4_0");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "rms_norm_matvec_q4_0 pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(x, 0, 0);
    encoder->setBuffer(norm_weight, 0, 1);
    encoder->setBuffer(W, 0, 2);
    encoder->setBuffer(y, 0, 3);
    encoder->setBytes(&K, sizeof(K), 4);
    encoder->setBytes(&N, sizeof(N), 5);
    encoder->setBytes(&eps, sizeof(eps), 6);

    uint32_t rows_per_tg = 8;
    uint32_t num_threadgroups = (N + rows_per_tg - 1) / rows_per_tg;
    MTL::Size grid_size = MTL::Size::Make(num_threadgroups, 1, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(256, 1, 1);
    encoder->dispatchThreadgroups(grid_size, threadgroup_size);

    return {};
}

Result<void> MetalCompute::rms_norm_matvec_iq4_nl(
    MTL::Buffer* x, MTL::Buffer* norm_weight, MTL::Buffer* W, MTL::Buffer* y,
    uint32_t K, uint32_t N, float eps)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("rms_norm_matvec_iq4_nl");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "rms_norm_matvec_iq4_nl pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(x, 0, 0);
    encoder->setBuffer(norm_weight, 0, 1);
    encoder->setBuffer(W, 0, 2);
    encoder->setBuffer(y, 0, 3);
    encoder->setBytes(&K, sizeof(K), 4);
    encoder->setBytes(&N, sizeof(N), 5);
    encoder->setBytes(&eps, sizeof(eps), 6);

    uint32_t rows_per_tg = 8;
    uint32_t num_threadgroups = (N + rows_per_tg - 1) / rows_per_tg;
    MTL::Size grid_size = MTL::Size::Make(num_threadgroups, 1, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(256, 1, 1);
    encoder->dispatchThreadgroups(grid_size, threadgroup_size);

    return {};
}

Result<void> MetalCompute::rms_norm_matvec_iq4_xs(
    MTL::Buffer* x, MTL::Buffer* norm_weight, MTL::Buffer* W, MTL::Buffer* y,
    uint32_t K, uint32_t N, float eps)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("rms_norm_matvec_iq4_xs");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "rms_norm_matvec_iq4_xs pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(x, 0, 0);
    encoder->setBuffer(norm_weight, 0, 1);
    encoder->setBuffer(W, 0, 2);
    encoder->setBuffer(y, 0, 3);
    encoder->setBytes(&K, sizeof(K), 4);
    encoder->setBytes(&N, sizeof(N), 5);
    encoder->setBytes(&eps, sizeof(eps), 6);

    uint32_t rows_per_tg = 8;
    uint32_t num_threadgroups = (N + rows_per_tg - 1) / rows_per_tg;
    MTL::Size grid_size = MTL::Size::Make(num_threadgroups, 1, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(256, 1, 1);
    encoder->dispatchThreadgroups(grid_size, threadgroup_size);

    return {};
}

Result<void> MetalCompute::rms_norm_matvec_iq3_s(
    MTL::Buffer* x, MTL::Buffer* norm_weight, MTL::Buffer* W, MTL::Buffer* y,
    uint32_t K, uint32_t N, float eps)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("rms_norm_matvec_iq3_s");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "rms_norm_matvec_iq3_s pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(x, 0, 0);
    encoder->setBuffer(norm_weight, 0, 1);
    encoder->setBuffer(W, 0, 2);
    encoder->setBuffer(y, 0, 3);
    encoder->setBytes(&K, sizeof(K), 4);
    encoder->setBytes(&N, sizeof(N), 5);
    encoder->setBytes(&eps, sizeof(eps), 6);

    uint32_t rows_per_tg = 8;
    uint32_t num_threadgroups = (N + rows_per_tg - 1) / rows_per_tg;
    MTL::Size grid_size = MTL::Size::Make(num_threadgroups, 1, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(256, 1, 1);
    encoder->dispatchThreadgroups(grid_size, threadgroup_size);

    return {};
}

Result<void> MetalCompute::rms_norm_matvec_q6_k(
    MTL::Buffer* x, MTL::Buffer* norm_weight, MTL::Buffer* W, MTL::Buffer* y,
    uint32_t K, uint32_t N, float eps)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("rms_norm_matvec_q6_k");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "rms_norm_matvec_q6_k pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(x, 0, 0);
    encoder->setBuffer(norm_weight, 0, 1);
    encoder->setBuffer(W, 0, 2);
    encoder->setBuffer(y, 0, 3);
    encoder->setBytes(&K, sizeof(K), 4);
    encoder->setBytes(&N, sizeof(N), 5);
    encoder->setBytes(&eps, sizeof(eps), 6);

    uint32_t rows_per_tg = 8;
    uint32_t num_threadgroups = (N + rows_per_tg - 1) / rows_per_tg;
    MTL::Size grid_size = MTL::Size::Make(num_threadgroups, 1, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(256, 1, 1);
    encoder->dispatchThreadgroups(grid_size, threadgroup_size);

    return {};
}

Result<void> MetalCompute::rms_norm_matvec_q5_k(
    MTL::Buffer* x, MTL::Buffer* norm_weight, MTL::Buffer* W, MTL::Buffer* y,
    uint32_t K, uint32_t N, float eps)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("rms_norm_matvec_q5_k");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "rms_norm_matvec_q5_k pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(x, 0, 0);
    encoder->setBuffer(norm_weight, 0, 1);
    encoder->setBuffer(W, 0, 2);
    encoder->setBuffer(y, 0, 3);
    encoder->setBytes(&K, sizeof(K), 4);
    encoder->setBytes(&N, sizeof(N), 5);
    encoder->setBytes(&eps, sizeof(eps), 6);

    uint32_t rows_per_tg = 8;
    uint32_t num_threadgroups = (N + rows_per_tg - 1) / rows_per_tg;
    MTL::Size grid_size = MTL::Size::Make(num_threadgroups, 1, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(256, 1, 1);
    encoder->dispatchThreadgroups(grid_size, threadgroup_size);

    return {};
}

Result<void> MetalCompute::rms_norm_matvec_q3_k(
    MTL::Buffer* x, MTL::Buffer* norm_weight, MTL::Buffer* W, MTL::Buffer* y,
    uint32_t K, uint32_t N, float eps)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("rms_norm_matvec_q3_k");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "rms_norm_matvec_q3_k pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(x, 0, 0);
    encoder->setBuffer(norm_weight, 0, 1);
    encoder->setBuffer(W, 0, 2);
    encoder->setBuffer(y, 0, 3);
    encoder->setBytes(&K, sizeof(K), 4);
    encoder->setBytes(&N, sizeof(N), 5);
    encoder->setBytes(&eps, sizeof(eps), 6);

    uint32_t rows_per_tg = 8;
    uint32_t num_threadgroups = (N + rows_per_tg - 1) / rows_per_tg;
    MTL::Size grid_size = MTL::Size::Make(num_threadgroups, 1, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(256, 1, 1);
    encoder->dispatchThreadgroups(grid_size, threadgroup_size);

    return {};
}

Result<void> MetalCompute::rms_norm_matvec_q2_k(
    MTL::Buffer* x, MTL::Buffer* norm_weight, MTL::Buffer* W, MTL::Buffer* y,
    uint32_t K, uint32_t N, float eps)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("rms_norm_matvec_q2_k");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "rms_norm_matvec_q2_k pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(x, 0, 0);
    encoder->setBuffer(norm_weight, 0, 1);
    encoder->setBuffer(W, 0, 2);
    encoder->setBuffer(y, 0, 3);
    encoder->setBytes(&K, sizeof(K), 4);
    encoder->setBytes(&N, sizeof(N), 5);
    encoder->setBytes(&eps, sizeof(eps), 6);

    uint32_t rows_per_tg = 8;
    uint32_t num_threadgroups = (N + rows_per_tg - 1) / rows_per_tg;
    MTL::Size grid_size = MTL::Size::Make(num_threadgroups, 1, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(256, 1, 1);
    encoder->dispatchThreadgroups(grid_size, threadgroup_size);

    return {};
}

// =============================================================================
// Phase 2 Fused Kernels - Eliminates redundant computation
// =============================================================================

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

    // DUAL_ROWS_PER_TG = 4 (4 SIMD groups, each handles 1 row for gate+up)
    uint32_t rows_per_tg = 4;
    uint32_t num_threadgroups = (N + rows_per_tg - 1) / rows_per_tg;
    MTL::Size grid_size = MTL::Size::Make(num_threadgroups, 1, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(256, 1, 1);  // 8 SIMD groups × 32 threads
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

    // 8 SIMD groups, each handles 1 row
    uint32_t rows_per_tg = 8;
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

    // 8 SIMD groups, each handles 1 row
    uint32_t rows_per_tg = 8;
    uint32_t num_threadgroups = (N + rows_per_tg - 1) / rows_per_tg;
    MTL::Size grid_size = MTL::Size::Make(num_threadgroups, 1, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(256, 1, 1);
    encoder->dispatchThreadgroups(grid_size, threadgroup_size);

    return {};
}

Result<void> MetalCompute::matvec_residual_q4k(
    MTL::Buffer* x, MTL::Buffer* W, MTL::Buffer* residual, MTL::Buffer* y,
    uint32_t K, uint32_t N)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("matvec_residual_q4k");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "matvec_residual_q4k pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(x, 0, 0);
    encoder->setBuffer(W, 0, 1);
    encoder->setBuffer(residual, 0, 2);
    encoder->setBuffer(y, 0, 3);
    encoder->setBytes(&K, sizeof(K), 4);
    encoder->setBytes(&N, sizeof(N), 5);

    // FUSED_ROWS_PER_TG = 8
    uint32_t rows_per_tg = 8;
    uint32_t num_threadgroups = (N + rows_per_tg - 1) / rows_per_tg;
    MTL::Size grid_size = MTL::Size::Make(num_threadgroups, 1, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(256, 1, 1);
    encoder->dispatchThreadgroups(grid_size, threadgroup_size);

    return {};
}

Result<void> MetalCompute::matvec_residual_q3k(
    MTL::Buffer* x, MTL::Buffer* W, MTL::Buffer* residual, MTL::Buffer* y,
    uint32_t K, uint32_t N)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("matvec_residual_q3k");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "matvec_residual_q3k pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(x, 0, 0);
    encoder->setBuffer(W, 0, 1);
    encoder->setBuffer(residual, 0, 2);
    encoder->setBuffer(y, 0, 3);
    encoder->setBytes(&K, sizeof(K), 4);
    encoder->setBytes(&N, sizeof(N), 5);

    // Q3K_ROWS_PER_TG = 8, NR0_Q3K = 2 -> 16 rows per threadgroup
    uint32_t rows_per_tg = 16;  // 8 SIMD groups * 2 rows each
    uint32_t num_threadgroups = (N + rows_per_tg - 1) / rows_per_tg;
    MTL::Size grid_size = MTL::Size::Make(num_threadgroups, 1, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(256, 1, 1);
    encoder->dispatchThreadgroups(grid_size, threadgroup_size);

    return {};
}

Result<void> MetalCompute::matvec_residual_q2k(
    MTL::Buffer* x, MTL::Buffer* W, MTL::Buffer* residual, MTL::Buffer* y,
    uint32_t K, uint32_t N)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("matvec_residual_q2k");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "matvec_residual_q2k pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(x, 0, 0);
    encoder->setBuffer(W, 0, 1);
    encoder->setBuffer(residual, 0, 2);
    encoder->setBuffer(y, 0, 3);
    encoder->setBytes(&K, sizeof(K), 4);
    encoder->setBytes(&N, sizeof(N), 5);

    // Q2K_ROWS_PER_TG = 8, NR0_Q2K = 2 -> 16 rows per threadgroup
    uint32_t rows_per_tg = 16;  // 8 SIMD groups * 2 rows each
    uint32_t num_threadgroups = (N + rows_per_tg - 1) / rows_per_tg;
    MTL::Size grid_size = MTL::Size::Make(num_threadgroups, 1, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(256, 1, 1);
    encoder->dispatchThreadgroups(grid_size, threadgroup_size);

    return {};
}

// =============================================================================
// Fused QKV Attention Projection
// =============================================================================

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
    uint32_t rows_per_tg = 16;
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

// Global singleton
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

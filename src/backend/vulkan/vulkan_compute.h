// vulkan_compute.h - High-level compute interface for Vulkan backend
//
// This file provides the VulkanCompute class that mirrors MetalCompute,
// offering high-level dispatch functions for quantized operations.

#pragma once

#include "granite/error.h"

#ifdef GRANITE_HAS_VULKAN

#include <vulkan/vulkan.h>
#include <string>
#include <vector>
#include <cstdint>
#include <memory>

namespace granite {

// Forward declarations
class VulkanComputeImpl;
class VulkanCompute;
class IComputeBackend;

class VulkanCompute {
public:
    struct ProfilingStats {
        uint64_t dispatch_count = 0;
        uint64_t sync_count = 0;
        double sync_time_ms = 0;
        uint64_t command_buffer_count = 0;
    };

    VulkanCompute();
    ~VulkanCompute();

    // Non-copyable
    VulkanCompute(const VulkanCompute&) = delete;
    VulkanCompute& operator=(const VulkanCompute&) = delete;

    // Movable
    VulkanCompute(VulkanCompute&&) noexcept;
    VulkanCompute& operator=(VulkanCompute&&) noexcept;

    // Initialization
    bool initialize(VkDevice device, VkPhysicalDevice physical_device,
                   VkQueue compute_queue, uint32_t queue_family,
                   const std::string& shader_dir = "");
    void shutdown();

    // Synchronization
    void sync();

    // =========================================================================
    // Activation Operations
    // =========================================================================

    Result<void> silu(VkBuffer x, VkBuffer out, uint32_t size);
    Result<void> gelu(VkBuffer x, VkBuffer out, uint32_t size);

    // =========================================================================
    // Elementwise Operations
    // =========================================================================

    Result<void> add(VkBuffer a, VkBuffer b, VkBuffer out, uint32_t size);
    Result<void> mul(VkBuffer a, VkBuffer b, VkBuffer out, uint32_t size);

    // =========================================================================
    // Normalization Operations
    // =========================================================================

    Result<void> rms_norm(VkBuffer x, VkBuffer weight, VkBuffer out,
                         uint32_t size, float eps);

    // =========================================================================
    // Quantized Matrix-Vector Operations (Decode - Single Token)
    // =========================================================================

    Result<void> matvec_q4k(VkBuffer x, VkBuffer W, VkBuffer y, uint32_t K, uint32_t N);
    Result<void> matvec_q8_0(VkBuffer x, VkBuffer W, VkBuffer y, uint32_t K, uint32_t N);

    // Basic FP32 matmul (A[M,K] @ B[K,N] = C[M,N])
    Result<void> matmul_f32(VkBuffer a, VkBuffer b, VkBuffer out,
                            uint32_t M, uint32_t K, uint32_t N);

    // =========================================================================
    // RoPE (Rotary Position Embedding)
    // =========================================================================

    Result<void> rope(VkBuffer x, VkBuffer freq_cos, VkBuffer freq_sin,
                     uint32_t head_dim, uint32_t num_heads, float position);

    // =========================================================================
    // Fused Operations
    // =========================================================================

    // SiLU activation fused with elementwise multiply (for FFN gate * up)
    Result<void> silu_mul(VkBuffer gate, VkBuffer up, uint32_t size);

    // =========================================================================
    // Softmax Operations
    // =========================================================================

    Result<void> softmax(VkBuffer x, VkBuffer out, uint32_t size, float scale = 1.0f);
    Result<void> softmax_rows(VkBuffer x, VkBuffer out, uint32_t rows, uint32_t cols,
                              float scale = 1.0f);

    // =========================================================================
    // Attention Operations
    // =========================================================================

    // TODO: Implement flash attention decode
    // Result<void> flash_attention_decode(...);

    // =========================================================================
    // Profiling
    // =========================================================================

    void enable_profiling(bool enable);
    ProfilingStats get_stats() const;
    void reset_stats();

    // =========================================================================
    // Shader Management
    // =========================================================================

    // Compile GLSL shader at runtime (requires shaderc)
    Result<void> compile_shader(const std::string& name, const std::string& glsl_code,
                               uint32_t num_buffers = 4);

    // Load pre-compiled SPIR-V shader
    Result<void> load_spirv(const std::string& name, const std::vector<uint32_t>& spirv,
                           uint32_t num_buffers = 4);

    // Check if compute interface is initialized
    bool is_initialized() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// Global accessor for VulkanCompute singleton
// Returns nullptr if Vulkan is not available or initialization failed
VulkanCompute* get_vulkan_compute();

// Accessor tied to a specific backend instance
// Returns nullptr if Vulkan is not available or initialization failed
VulkanCompute* get_vulkan_compute(IComputeBackend* backend);

// Release compute instance tied to backend (called before backend shutdown)
void release_vulkan_compute(IComputeBackend* backend);

} // namespace granite

#endif // GRANITE_HAS_VULKAN

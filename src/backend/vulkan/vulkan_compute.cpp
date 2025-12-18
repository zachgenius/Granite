// vulkan_compute.cpp - High-level compute interface for Vulkan backend
//
// This file provides the VulkanCompute class that mirrors MetalCompute,
// offering high-level dispatch functions for quantized operations.
//
// Shaders are adapted from llama.cpp's MIT-licensed Vulkan backend.

#include "granite/log.h"

#ifdef GRANITE_HAS_VULKAN

#include <vulkan/vulkan.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace granite {

// Forward declaration
class VulkanBackend;

class VulkanCompute {
public:
    struct ProfilingStats {
        uint64_t dispatch_count = 0;
        uint64_t sync_count = 0;
        double sync_time_ms = 0;
        uint64_t command_buffer_count = 0;
    };

    VulkanCompute() = default;
    ~VulkanCompute() { shutdown(); }

    bool initialize(VkDevice device, VkPhysicalDevice physical_device,
                   VkQueue compute_queue, uint32_t queue_family) {
        device_ = device;
        physical_device_ = physical_device;
        compute_queue_ = compute_queue;
        queue_family_ = queue_family;

        // Create command pool
        VkCommandPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.queueFamilyIndex = queue_family_;
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

        if (vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_) != VK_SUCCESS) {
            GRANITE_LOG_ERROR("Failed to create Vulkan command pool");
            return false;
        }

        // TODO: Load and compile shaders
        // TODO: Create pipelines for all kernel types

        initialized_ = true;
        GRANITE_LOG_INFO("VulkanCompute initialized");
        return true;
    }

    void shutdown() {
        if (!initialized_) return;

        if (device_ != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(device_);

            // Destroy pipelines
            for (auto& [name, pipeline] : pipelines_) {
                vkDestroyPipeline(device_, pipeline, nullptr);
            }
            pipelines_.clear();

            // Destroy pipeline layouts
            for (auto& [name, layout] : pipeline_layouts_) {
                vkDestroyPipelineLayout(device_, layout, nullptr);
            }
            pipeline_layouts_.clear();

            // Destroy command pool
            if (command_pool_ != VK_NULL_HANDLE) {
                vkDestroyCommandPool(device_, command_pool_, nullptr);
                command_pool_ = VK_NULL_HANDLE;
            }
        }

        initialized_ = false;
    }

    void sync() {
        if (device_ != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(device_);
            stats_.sync_count++;
        }
    }

    // =========================================================================
    // Quantized Matrix-Vector Operations (Decode - Single Token)
    // =========================================================================
    // TODO: Implement these using llama.cpp shaders

    // Result<void> matvec_q4k(VkBuffer x, VkBuffer W, VkBuffer y, uint32_t K, uint32_t N);
    // Result<void> matvec_q8_0(VkBuffer x, VkBuffer W, VkBuffer y, uint32_t K, uint32_t N);
    // Result<void> matvec_q4_0(VkBuffer x, VkBuffer W, VkBuffer y, uint32_t K, uint32_t N);
    // ... etc

    // =========================================================================
    // Quantized Matrix-Matrix Operations (Prefill - Batched)
    // =========================================================================
    // TODO: Implement these using llama.cpp shaders

    // Result<void> matmul_q4k(VkBuffer X, VkBuffer W, VkBuffer Y, uint32_t M, uint32_t K, uint32_t N);
    // ... etc

    // =========================================================================
    // Normalization Operations
    // =========================================================================
    // TODO: Implement these

    // Result<void> rms_norm(VkBuffer x, VkBuffer weight, VkBuffer out, uint32_t size, float eps);
    // Result<void> layer_norm(VkBuffer x, VkBuffer weight, VkBuffer bias, VkBuffer out, uint32_t size, float eps);

    // =========================================================================
    // Attention Operations
    // =========================================================================
    // TODO: Implement these

    // Result<void> flash_attention_decode(...);
    // Result<void> multihead_attention_decode(...);

    // =========================================================================
    // Utility Operations
    // =========================================================================
    // TODO: Implement these

    // Result<void> rope_multihead(...);
    // Result<void> kv_cache_append(...);
    // Result<void> silu(VkBuffer x, uint32_t size);
    // Result<void> gelu(VkBuffer x, uint32_t size);
    // Result<void> elementwise_add(VkBuffer a, VkBuffer b, VkBuffer c, uint32_t size);
    // Result<void> elementwise_mul(VkBuffer a, VkBuffer b, VkBuffer c, uint32_t size);

    // Profiling
    void enable_profiling(bool enable) { profiling_enabled_ = enable; }
    ProfilingStats get_stats() const { return stats_; }
    void reset_stats() { stats_ = ProfilingStats{}; }

private:
    bool initialized_ = false;
    bool profiling_enabled_ = false;
    ProfilingStats stats_;

    VkDevice device_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkQueue compute_queue_ = VK_NULL_HANDLE;
    uint32_t queue_family_ = 0;

    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer current_command_buffer_ = VK_NULL_HANDLE;

    // Pipeline cache
    std::unordered_map<std::string, VkPipeline> pipelines_;
    std::unordered_map<std::string, VkPipelineLayout> pipeline_layouts_;
};

} // namespace granite

#endif // GRANITE_HAS_VULKAN

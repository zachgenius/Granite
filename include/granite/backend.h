#pragma once

#include "granite/types.h"
#include "granite/error.h"

#include <memory>
#include <string>
#include <span>

namespace granite {

// =============================================================================
// Pipeline Description
// =============================================================================

struct PipelineDesc {
    std::string shader_source;      // MSL source or SPIR-V path
    std::string entry_point = "main";

    // Threadgroup size hints (0 = let backend decide)
    uint32_t threadgroup_size_x = 0;
    uint32_t threadgroup_size_y = 0;
    uint32_t threadgroup_size_z = 0;
};

// =============================================================================
// Compute Backend Interface
// =============================================================================

class IComputeBackend {
public:
    virtual ~IComputeBackend() = default;

    // -------------------------------------------------------------------------
    // Lifecycle
    // -------------------------------------------------------------------------

    /// Initialize the backend (create device, queues, etc.)
    virtual Result<void> initialize() = 0;

    /// Shutdown and release all resources
    virtual void shutdown() = 0;

    /// Check if the backend is initialized
    [[nodiscard]] virtual bool is_initialized() const = 0;

    // -------------------------------------------------------------------------
    // Device Info
    // -------------------------------------------------------------------------

    /// Get device capabilities
    [[nodiscard]] virtual DeviceCapabilities get_capabilities() const = 0;

    /// Get backend type
    [[nodiscard]] virtual BackendType get_type() const = 0;

    // -------------------------------------------------------------------------
    // Buffer Management
    // -------------------------------------------------------------------------

    /// Create a buffer with the given description
    virtual Result<BufferHandle> create_buffer(const BufferDesc& desc) = 0;

    /// Create a buffer from existing host memory (best-effort zero-copy)
    virtual Result<BufferHandle> create_buffer_from_host(const void* data,
                                                         const BufferDesc& desc) {
        if (!data) {
            return Error(ErrorCode::InvalidArgument, "Null host buffer");
        }

        auto buffer_result = create_buffer(desc);
        if (!buffer_result.ok()) {
            return buffer_result.error();
        }

        auto write_result = write_buffer(buffer_result.value(), data, desc.size);
        if (!write_result.ok()) {
            destroy_buffer(buffer_result.value());
            return write_result.error();
        }

        return buffer_result;
    }

    /// Destroy a buffer
    virtual void destroy_buffer(BufferHandle handle) = 0;

    /// Map buffer memory for CPU access (only valid for Shared/Managed buffers)
    virtual Result<void*> map_buffer(BufferHandle handle) = 0;

    /// Unmap buffer memory
    virtual void unmap_buffer(BufferHandle handle) = 0;

    /// Copy data from CPU to buffer
    virtual Result<void> write_buffer(BufferHandle handle,
                                       const void* data,
                                       size_t size,
                                       size_t offset = 0) = 0;

    /// Copy data from buffer to CPU
    virtual Result<void> read_buffer(BufferHandle handle,
                                      void* data,
                                      size_t size,
                                      size_t offset = 0) = 0;

    /// Copy between buffers
    virtual Result<void> copy_buffer(BufferHandle src,
                                      BufferHandle dst,
                                      size_t size,
                                      size_t src_offset = 0,
                                      size_t dst_offset = 0) = 0;

    /// Get native buffer pointer (e.g., MTL::Buffer* for Metal)
    /// Returns nullptr if not supported or handle is invalid
    virtual void* get_native_buffer(BufferHandle handle) { (void)handle; return nullptr; }

    /// Get native device handle (e.g., VkDevice for Vulkan)
    virtual void* get_native_device() { return nullptr; }

    /// Get native physical device handle (e.g., VkPhysicalDevice for Vulkan)
    virtual void* get_native_physical_device() { return nullptr; }

    /// Get native queue handle (e.g., VkQueue for Vulkan)
    virtual void* get_native_queue() { return nullptr; }

    /// Get native queue family index (e.g., Vulkan compute queue family)
    virtual uint32_t get_native_queue_family() { return 0; }

    // -------------------------------------------------------------------------
    // Compute Pipeline
    // -------------------------------------------------------------------------

    /// Create a compute pipeline from shader source
    virtual Result<PipelineHandle> create_pipeline(const PipelineDesc& desc) = 0;

    /// Destroy a pipeline
    virtual void destroy_pipeline(PipelineHandle handle) = 0;

    // -------------------------------------------------------------------------
    // Command Recording & Execution
    // -------------------------------------------------------------------------

    /// Begin recording commands
    virtual Result<void> begin_commands() = 0;

    /// Bind a compute pipeline
    virtual Result<void> bind_pipeline(PipelineHandle pipeline) = 0;

    /// Bind a buffer to a binding index
    virtual Result<void> bind_buffer(uint32_t index,
                                      BufferHandle buffer,
                                      size_t offset = 0) = 0;

    /// Set push constants (small uniform data)
    virtual Result<void> set_push_constants(const void* data, size_t size) = 0;

    /// Dispatch compute work
    virtual Result<void> dispatch(uint32_t groups_x,
                                   uint32_t groups_y = 1,
                                   uint32_t groups_z = 1) = 0;

    /// End recording commands
    virtual Result<void> end_commands() = 0;

    /// Submit recorded commands for execution
    virtual Result<void> submit() = 0;

    /// Wait for all submitted commands to complete
    virtual Result<void> wait_for_completion() = 0;

    // -------------------------------------------------------------------------
    // Synchronization
    // -------------------------------------------------------------------------

    /// Create a fence for synchronization
    virtual Result<FenceHandle> create_fence() = 0;

    /// Wait for a fence to be signaled
    virtual Result<void> wait_fence(FenceHandle fence, uint64_t timeout_ns = UINT64_MAX) = 0;

    /// Destroy a fence
    virtual void destroy_fence(FenceHandle fence) = 0;
};

// =============================================================================
// Backend Factory
// =============================================================================

/// Create a compute backend of the specified type
/// Returns nullptr if the backend type is not available on this platform
std::unique_ptr<IComputeBackend> create_backend(BackendType type);

/// Get the default backend for this platform
/// (Metal on Apple, Vulkan on Android/Linux, CPU fallback otherwise)
std::unique_ptr<IComputeBackend> create_default_backend();

/// Check if a backend type is available on this platform
bool is_backend_available(BackendType type);

}  // namespace granite

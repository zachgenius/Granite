#include "granite/backend.h"
#include "granite/log.h"

#ifdef GRANITE_HAS_METAL

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <unordered_map>
#include <atomic>
#include <string>
#include <cstring>

namespace granite {

class MetalBackend : public IComputeBackend {
public:
    MetalBackend() = default;

    ~MetalBackend() override {
        shutdown();
    }

    Result<void> initialize() override {
        if (initialized_) {
            return {};
        }

        GRANITE_LOG_INFO("Initializing Metal backend");

        // Get default device
        device_ = MTL::CreateSystemDefaultDevice();
        if (!device_) {
            return Error(ErrorCode::BackendNotSupported, "No Metal device available");
        }

        GRANITE_LOG_INFO("Metal device: {}", device_->name()->utf8String());

        // Create command queue
        command_queue_ = device_->newCommandQueue();
        if (!command_queue_) {
            device_->release();
            device_ = nullptr;
            return Error(ErrorCode::InternalError, "Failed to create command queue");
        }

        initialized_ = true;
        return {};
    }

    void shutdown() override {
        if (!initialized_) {
            return;
        }

        GRANITE_LOG_INFO("Shutting down Metal backend");

        // Wait for any pending work
        if (current_command_buffer_) {
            current_command_buffer_->waitUntilCompleted();
            current_command_buffer_->release();
            current_command_buffer_ = nullptr;
        }

        // Release all buffers
        for (auto& [handle, buffer] : buffers_) {
            buffer->release();
        }
        buffers_.clear();

        // Release all pipelines
        for (auto& [handle, pipeline] : pipelines_) {
            pipeline->release();
        }
        pipelines_.clear();

        // Release command queue and device
        if (command_queue_) {
            command_queue_->release();
            command_queue_ = nullptr;
        }

        if (device_) {
            device_->release();
            device_ = nullptr;
        }

        initialized_ = false;
    }

    [[nodiscard]] bool is_initialized() const override {
        return initialized_;
    }

    [[nodiscard]] DeviceCapabilities get_capabilities() const override {
        DeviceCapabilities caps;

        if (device_) {
            caps.name = device_->name()->utf8String();
            caps.max_buffer_size = device_->maxBufferLength();
            caps.max_threadgroup_size = device_->maxThreadsPerThreadgroup().width;
            caps.shared_memory_size = device_->maxThreadgroupMemoryLength();
            caps.simd_width = 32;  // Typical for Apple GPUs
            caps.supports_fp16 = true;
            caps.supports_bf16 = false;  // Not widely supported yet
            caps.supports_int8 = true;
            caps.supports_int4 = true;   // Via custom shaders
            caps.supports_simd_groups = true;
        }

        return caps;
    }

    [[nodiscard]] BackendType get_type() const override {
        return BackendType::Metal;
    }

    Result<BufferHandle> create_buffer_from_host(const void* data,
                                                 const BufferDesc& desc) override {
        if (!initialized_) {
            return Error(ErrorCode::BackendNotInitialized);
        }
        if (!data) {
            return Error(ErrorCode::InvalidArgument, "Null host buffer");
        }

        MTL::ResourceOptions options;
        bool can_no_copy = false;
        switch (desc.memory_type) {
            case MemoryType::Device:
                options = MTL::ResourceStorageModePrivate;
                break;
            case MemoryType::Shared:
                options = MTL::ResourceStorageModeShared;
                can_no_copy = true;
                break;
            case MemoryType::Managed:
#if TARGET_OS_OSX
                if (device_->hasUnifiedMemory()) {
                    options = MTL::ResourceStorageModeShared;
                    can_no_copy = true;
                } else {
                    options = MTL::ResourceStorageModeManaged;
                }
#else
                options = MTL::ResourceStorageModeShared;
                can_no_copy = true;
#endif
                break;
        }

        if (!can_no_copy) {
            auto buffer_result = create_buffer(desc);
            if (!buffer_result.ok()) {
                return buffer_result.error();
            }
            auto write_result = write_buffer(buffer_result.value(), data, desc.size, 0);
            if (!write_result.ok()) {
                destroy_buffer(buffer_result.value());
                return write_result.error();
            }
            return buffer_result;
        }

        MTL::Buffer* buffer = device_->newBuffer(
            data,
            desc.size,
            options,
            nullptr);
        if (!buffer) {
            return Error(ErrorCode::AllocationFailed,
                         fmt::format("Failed to allocate Metal buffer of size {}", desc.size));
        }

        BufferHandle handle{next_handle_++};
        buffers_[handle] = buffer;

        return handle;
    }

    Result<BufferHandle> create_buffer(const BufferDesc& desc) override {
        if (!initialized_) {
            return Error(ErrorCode::BackendNotInitialized);
        }

        MTL::ResourceOptions options;
        switch (desc.memory_type) {
            case MemoryType::Device:
                options = MTL::ResourceStorageModePrivate;
                break;
            case MemoryType::Shared:
                options = MTL::ResourceStorageModeShared;
                break;
            case MemoryType::Managed:
#if TARGET_OS_OSX
                if (device_->hasUnifiedMemory()) {
                    options = MTL::ResourceStorageModeShared;
                } else {
                    options = MTL::ResourceStorageModeManaged;
                }
#else
                // iOS doesn't have managed mode, use shared
                options = MTL::ResourceStorageModeShared;
#endif
                break;
        }

        MTL::Buffer* buffer = device_->newBuffer(desc.size, options);
        if (!buffer) {
            return Error(ErrorCode::AllocationFailed,
                         fmt::format("Failed to allocate Metal buffer of size {}", desc.size));
        }

        BufferHandle handle{next_handle_++};
        buffers_[handle] = buffer;

        return handle;
    }

    void destroy_buffer(BufferHandle handle) override {
        auto it = buffers_.find(handle);
        if (it != buffers_.end()) {
            it->second->release();
            buffers_.erase(it);
        }
    }

    Result<void*> map_buffer(BufferHandle handle) override {
        auto it = buffers_.find(handle);
        if (it == buffers_.end()) {
            return Error(ErrorCode::InvalidArgument, "Invalid buffer handle");
        }

        void* ptr = it->second->contents();
        if (!ptr) {
            return Error(ErrorCode::InvalidArgument, "Buffer is not mappable (private storage)");
        }

        return ptr;
    }

    void unmap_buffer(BufferHandle handle) override {
#if TARGET_OS_OSX
        auto it = buffers_.find(handle);
        if (it != buffers_.end()) {
            // Notify Metal that the buffer was modified
            MTL::Buffer* buffer = it->second;
            if (buffer->storageMode() == MTL::StorageModeManaged) {
                buffer->didModifyRange(NS::Range::Make(0, buffer->length()));
            }
        }
#else
        (void)handle;
#endif
    }

    Result<void> write_buffer(BufferHandle handle,
                               const void* data,
                               size_t size,
                               size_t offset) override {
        auto it = buffers_.find(handle);
        if (it == buffers_.end()) {
            return Error(ErrorCode::InvalidArgument, "Invalid buffer handle");
        }

        MTL::Buffer* buffer = it->second;
        if (buffer->storageMode() == MTL::StorageModePrivate) {
            MTL::Buffer* staging = device_->newBuffer(size, MTL::ResourceStorageModeShared);
            if (!staging) {
                return Error(ErrorCode::AllocationFailed,
                             fmt::format("Failed to allocate staging buffer of size {}", size));
            }
            std::memcpy(staging->contents(), data, size);

            MTL::CommandBuffer* cmd = command_queue_->commandBuffer();
            MTL::BlitCommandEncoder* blit = cmd->blitCommandEncoder();
            blit->copyFromBuffer(staging, 0, buffer, offset, size);
            blit->endEncoding();
            cmd->commit();
            cmd->waitUntilCompleted();

            staging->release();
            return {};
        }

        auto map_result = map_buffer(handle);
        if (!map_result.ok()) {
            return map_result.error();
        }

        std::memcpy(static_cast<uint8_t*>(map_result.value()) + offset, data, size);
        unmap_buffer(handle);
        return {};
    }

    Result<void> read_buffer(BufferHandle handle,
                              void* data,
                              size_t size,
                              size_t offset) override {
        auto map_result = map_buffer(handle);
        if (!map_result.ok()) {
            return map_result.error();
        }

        std::memcpy(data, static_cast<uint8_t*>(map_result.value()) + offset, size);
        return {};
    }

    Result<void> copy_buffer(BufferHandle src,
                              BufferHandle dst,
                              size_t size,
                              size_t src_offset,
                              size_t dst_offset) override {
        auto src_it = buffers_.find(src);
        auto dst_it = buffers_.find(dst);

        if (src_it == buffers_.end() || dst_it == buffers_.end()) {
            return Error(ErrorCode::InvalidArgument, "Invalid buffer handle");
        }

        // Use blit encoder for buffer copy
        MTL::CommandBuffer* cmd = command_queue_->commandBuffer();
        MTL::BlitCommandEncoder* blit = cmd->blitCommandEncoder();

        blit->copyFromBuffer(src_it->second, src_offset,
                             dst_it->second, dst_offset,
                             size);

        blit->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();

        return {};
    }

    void* get_native_buffer(BufferHandle handle) override {
        auto it = buffers_.find(handle);
        if (it == buffers_.end()) {
            return nullptr;
        }
        return static_cast<void*>(it->second);
    }

    Result<PipelineHandle> create_pipeline(const PipelineDesc& desc) override {
        if (!initialized_) {
            return Error(ErrorCode::BackendNotInitialized);
        }

        // Compile shader source
        NS::Error* error = nullptr;
        MTL::CompileOptions* options = MTL::CompileOptions::alloc()->init();

        NS::String* source = NS::String::string(desc.shader_source.c_str(),
                                                 NS::UTF8StringEncoding);

        MTL::Library* library = device_->newLibrary(source, options, &error);
        options->release();

        if (!library) {
            std::string error_msg = "Shader compilation failed";
            if (error) {
                error_msg = error->localizedDescription()->utf8String();
            }
            return Error(ErrorCode::ShaderCompilationFailed, error_msg);
        }

        // Get function
        NS::String* func_name = NS::String::string(desc.entry_point.c_str(),
                                                    NS::UTF8StringEncoding);
        MTL::Function* function = library->newFunction(func_name);
        library->release();

        if (!function) {
            return Error(ErrorCode::ShaderCompilationFailed,
                         fmt::format("Function '{}' not found", desc.entry_point));
        }

        // Create pipeline state
        MTL::ComputePipelineState* pipeline = device_->newComputePipelineState(function, &error);
        function->release();

        if (!pipeline) {
            std::string error_msg = "Pipeline creation failed";
            if (error) {
                error_msg = error->localizedDescription()->utf8String();
            }
            return Error(ErrorCode::ShaderCompilationFailed, error_msg);
        }

        PipelineHandle handle{next_handle_++};
        pipelines_[handle] = pipeline;

        return handle;
    }

    void destroy_pipeline(PipelineHandle handle) override {
        auto it = pipelines_.find(handle);
        if (it != pipelines_.end()) {
            it->second->release();
            pipelines_.erase(it);
        }
    }

    Result<void> begin_commands() override {
        if (!initialized_) {
            return Error(ErrorCode::BackendNotInitialized);
        }

        if (current_command_buffer_) {
            return Error(ErrorCode::InvalidArgument, "Command buffer already in progress");
        }

        current_command_buffer_ = command_queue_->commandBuffer();
        current_encoder_ = current_command_buffer_->computeCommandEncoder();

        return {};
    }

    Result<void> bind_pipeline(PipelineHandle pipeline) override {
        if (!current_encoder_) {
            return Error(ErrorCode::InvalidArgument, "No command buffer in progress");
        }

        auto it = pipelines_.find(pipeline);
        if (it == pipelines_.end()) {
            return Error(ErrorCode::InvalidArgument, "Invalid pipeline handle");
        }

        current_encoder_->setComputePipelineState(it->second);
        current_pipeline_ = it->second;

        return {};
    }

    Result<void> bind_buffer(uint32_t index, BufferHandle buffer, size_t offset) override {
        if (!current_encoder_) {
            return Error(ErrorCode::InvalidArgument, "No command buffer in progress");
        }

        auto it = buffers_.find(buffer);
        if (it == buffers_.end()) {
            return Error(ErrorCode::InvalidArgument, "Invalid buffer handle");
        }

        current_encoder_->setBuffer(it->second, offset, index);
        return {};
    }

    Result<void> set_push_constants(const void* data, size_t size) override {
        if (!current_encoder_) {
            return Error(ErrorCode::InvalidArgument, "No command buffer in progress");
        }

        // Use buffer index 30 for push constants (common convention)
        current_encoder_->setBytes(data, size, 30);
        return {};
    }

    Result<void> dispatch(uint32_t groups_x, uint32_t groups_y, uint32_t groups_z) override {
        if (!current_encoder_ || !current_pipeline_) {
            return Error(ErrorCode::InvalidArgument, "No pipeline bound");
        }

        MTL::Size grid_size = MTL::Size::Make(groups_x, groups_y, groups_z);

        // Get threadgroup size from pipeline
        NS::UInteger w = current_pipeline_->threadExecutionWidth();
        NS::UInteger h = current_pipeline_->maxTotalThreadsPerThreadgroup() / w;
        MTL::Size threadgroup_size = MTL::Size::Make(w, h, 1);

        current_encoder_->dispatchThreadgroups(grid_size, threadgroup_size);
        return {};
    }

    Result<void> end_commands() override {
        if (!current_encoder_) {
            return Error(ErrorCode::InvalidArgument, "No command buffer in progress");
        }

        current_encoder_->endEncoding();
        current_encoder_ = nullptr;
        current_pipeline_ = nullptr;

        return {};
    }

    Result<void> submit() override {
        if (!current_command_buffer_) {
            return Error(ErrorCode::InvalidArgument, "No command buffer to submit");
        }

        current_command_buffer_->commit();
        return {};
    }

    Result<void> wait_for_completion() override {
        if (!current_command_buffer_) {
            return Error(ErrorCode::InvalidArgument, "No command buffer to wait on");
        }

        current_command_buffer_->waitUntilCompleted();

        // Check for errors
        if (current_command_buffer_->status() == MTL::CommandBufferStatusError) {
            std::string error_msg = "Command buffer execution failed";
            if (auto* error = current_command_buffer_->error()) {
                error_msg = error->localizedDescription()->utf8String();
            }
            current_command_buffer_->release();
            current_command_buffer_ = nullptr;
            return Error(ErrorCode::ExecutionFailed, error_msg);
        }

        current_command_buffer_->release();
        current_command_buffer_ = nullptr;

        return {};
    }

    Result<FenceHandle> create_fence() override {
        // Metal uses events for synchronization
        MTL::Event* event = device_->newEvent();
        if (!event) {
            return Error(ErrorCode::AllocationFailed, "Failed to create Metal event");
        }

        FenceHandle handle{next_handle_++};
        events_[handle] = event;

        return handle;
    }

    Result<void> wait_fence(FenceHandle fence, uint64_t /*timeout_ns*/) override {
        auto it = events_.find(fence);
        if (it == events_.end()) {
            return Error(ErrorCode::InvalidArgument, "Invalid fence handle");
        }

        // Create a shared event listener and wait
        // For simplicity, we use command buffer completion
        // A full implementation would use MTLSharedEventListener

        return {};
    }

    void destroy_fence(FenceHandle fence) override {
        auto it = events_.find(fence);
        if (it != events_.end()) {
            it->second->release();
            events_.erase(it);
        }
    }

private:
    bool initialized_ = false;
    std::atomic<uint64_t> next_handle_{1};

    MTL::Device* device_ = nullptr;
    MTL::CommandQueue* command_queue_ = nullptr;
    MTL::CommandBuffer* current_command_buffer_ = nullptr;
    MTL::ComputeCommandEncoder* current_encoder_ = nullptr;
    MTL::ComputePipelineState* current_pipeline_ = nullptr;

    std::unordered_map<BufferHandle, MTL::Buffer*> buffers_;
    std::unordered_map<PipelineHandle, MTL::ComputePipelineState*> pipelines_;
    std::unordered_map<FenceHandle, MTL::Event*> events_;
};

// Factory function for Metal backend
std::unique_ptr<IComputeBackend> create_metal_backend() {
    return std::make_unique<MetalBackend>();
}

}  // namespace granite

#endif  // GRANITE_HAS_METAL

#include "granite/backend.h"
#include "granite/log.h"

#include <unordered_map>
#include <vector>
#include <cstring>
#include <atomic>

namespace granite {

class CPUBackend : public IComputeBackend {
public:
    CPUBackend() = default;
    ~CPUBackend() override { shutdown(); }

    Result<void> initialize() override {
        if (initialized_) {
            return {};
        }

        GRANITE_LOG_INFO("Initializing CPU backend");

        // Detect SIMD capabilities
#if defined(GRANITE_HAS_NEON)
        GRANITE_LOG_DEBUG("NEON SIMD available");
#elif defined(GRANITE_HAS_AVX2)
        GRANITE_LOG_DEBUG("AVX2 SIMD available");
#endif

        initialized_ = true;
        return {};
    }

    void shutdown() override {
        if (!initialized_) {
            return;
        }

        GRANITE_LOG_INFO("Shutting down CPU backend");

        // Free all buffers
        buffers_.clear();
        initialized_ = false;
    }

    [[nodiscard]] bool is_initialized() const override {
        return initialized_;
    }

    [[nodiscard]] DeviceCapabilities get_capabilities() const override {
        DeviceCapabilities caps;
        caps.name = "CPU";
        caps.max_buffer_size = SIZE_MAX;
        caps.max_threadgroup_size = 1;
        caps.shared_memory_size = 0;
        caps.simd_width = 1;
        caps.supports_fp16 = false;  // CPU uses FP32
        caps.supports_bf16 = false;
        caps.supports_int8 = true;
        caps.supports_int4 = true;
        caps.supports_simd_groups = false;
        return caps;
    }

    [[nodiscard]] BackendType get_type() const override {
        return BackendType::CPU;
    }

    Result<BufferHandle> create_buffer(const BufferDesc& desc) override {
        if (!initialized_) {
            return Error(ErrorCode::BackendNotInitialized);
        }

        BufferHandle handle{next_handle_++};
        buffers_[handle] = std::vector<uint8_t>(desc.size);

        GRANITE_LOG_DEBUG("Created CPU buffer: id={}, size={}", handle.id, desc.size);
        return handle;
    }

    void destroy_buffer(BufferHandle handle) override {
        auto it = buffers_.find(handle);
        if (it != buffers_.end()) {
            GRANITE_LOG_DEBUG("Destroyed CPU buffer: id={}", handle.id);
            buffers_.erase(it);
        }
    }

    Result<void*> map_buffer(BufferHandle handle) override {
        auto it = buffers_.find(handle);
        if (it == buffers_.end()) {
            return Error(ErrorCode::InvalidArgument, "Invalid buffer handle");
        }
        return it->second.data();
    }

    void unmap_buffer(BufferHandle /*handle*/) override {
        // No-op for CPU backend
    }

    Result<void> write_buffer(BufferHandle handle,
                               const void* data,
                               size_t size,
                               size_t offset) override {
        auto it = buffers_.find(handle);
        if (it == buffers_.end()) {
            return Error(ErrorCode::InvalidArgument, "Invalid buffer handle");
        }

        if (offset + size > it->second.size()) {
            return Error(ErrorCode::BufferTooSmall);
        }

        std::memcpy(it->second.data() + offset, data, size);
        return {};
    }

    Result<void> read_buffer(BufferHandle handle,
                              void* data,
                              size_t size,
                              size_t offset) override {
        auto it = buffers_.find(handle);
        if (it == buffers_.end()) {
            return Error(ErrorCode::InvalidArgument, "Invalid buffer handle");
        }

        if (offset + size > it->second.size()) {
            return Error(ErrorCode::BufferTooSmall);
        }

        std::memcpy(data, it->second.data() + offset, size);
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

        if (src_offset + size > src_it->second.size() ||
            dst_offset + size > dst_it->second.size()) {
            return Error(ErrorCode::BufferTooSmall);
        }

        std::memcpy(dst_it->second.data() + dst_offset,
                    src_it->second.data() + src_offset,
                    size);
        return {};
    }

    Result<PipelineHandle> create_pipeline(const PipelineDesc& /*desc*/) override {
        // CPU backend doesn't use pipelines
        return Error(ErrorCode::NotImplemented, "CPU backend does not support pipelines");
    }

    void destroy_pipeline(PipelineHandle /*handle*/) override {
        // No-op
    }

    Result<void> begin_commands() override { return {}; }
    Result<void> bind_pipeline(PipelineHandle /*pipeline*/) override { return {}; }
    Result<void> bind_buffer(uint32_t /*index*/, BufferHandle /*buffer*/, size_t /*offset*/) override { return {}; }
    Result<void> set_push_constants(const void* /*data*/, size_t /*size*/) override { return {}; }
    Result<void> dispatch(uint32_t /*groups_x*/, uint32_t /*groups_y*/, uint32_t /*groups_z*/) override { return {}; }
    Result<void> end_commands() override { return {}; }
    Result<void> submit() override { return {}; }
    Result<void> wait_for_completion() override { return {}; }

    Result<FenceHandle> create_fence() override {
        return FenceHandle{next_handle_++};
    }

    Result<void> wait_fence(FenceHandle /*fence*/, uint64_t /*timeout_ns*/) override {
        return {};
    }

    void destroy_fence(FenceHandle /*fence*/) override {
        // No-op
    }

private:
    bool initialized_ = false;
    std::atomic<uint64_t> next_handle_{1};
    std::unordered_map<BufferHandle, std::vector<uint8_t>> buffers_;
};

// Factory function for CPU backend
std::unique_ptr<IComputeBackend> create_cpu_backend() {
    return std::make_unique<CPUBackend>();
}

}  // namespace granite

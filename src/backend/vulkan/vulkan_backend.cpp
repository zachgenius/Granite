// vulkan_backend.cpp - Vulkan compute backend for Granite
//
// This backend provides GPU acceleration via Vulkan for Linux/Android/Windows.
// Shaders are adapted from llama.cpp's MIT-licensed Vulkan backend.

#include "granite/backend.h"
#include "granite/log.h"

#ifdef GRANITE_HAS_VULKAN

#include <vulkan/vulkan.h>

#include <unordered_map>
#include <atomic>
#include <string>
#include <vector>
#include <cstring>

namespace granite {

class VulkanBackend : public IComputeBackend {
public:
    VulkanBackend() = default;

    ~VulkanBackend() override {
        shutdown();
    }

    Result<void> initialize() override {
        if (initialized_) {
            return {};
        }

        GRANITE_LOG_INFO("Initializing Vulkan backend");

        // Create Vulkan instance
        VkApplicationInfo app_info{};
        app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.pApplicationName = "Granite";
        app_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
        app_info.pEngineName = "Granite";
        app_info.engineVersion = VK_MAKE_VERSION(0, 1, 0);
        app_info.apiVersion = VK_API_VERSION_1_0;

        VkInstanceCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        create_info.pApplicationInfo = &app_info;

        // Enable validation layers in debug builds
#ifndef NDEBUG
        const char* validation_layers[] = {"VK_LAYER_KHRONOS_validation"};
        create_info.enabledLayerCount = 1;
        create_info.ppEnabledLayerNames = validation_layers;
#endif

        VkResult result = vkCreateInstance(&create_info, nullptr, &instance_);
        if (result != VK_SUCCESS) {
            return Error(ErrorCode::BackendNotSupported,
                        fmt::format("Failed to create Vulkan instance: {}", static_cast<int>(result)));
        }

        // Select physical device
        auto device_result = select_physical_device();
        if (!device_result.ok()) {
            vkDestroyInstance(instance_, nullptr);
            instance_ = VK_NULL_HANDLE;
            return device_result.error();
        }

        // Create logical device with compute queue
        auto logical_result = create_logical_device();
        if (!logical_result.ok()) {
            vkDestroyInstance(instance_, nullptr);
            instance_ = VK_NULL_HANDLE;
            return logical_result.error();
        }

        // Create command pool
        VkCommandPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.queueFamilyIndex = compute_queue_family_;
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

        result = vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_);
        if (result != VK_SUCCESS) {
            shutdown();
            return Error(ErrorCode::InternalError, "Failed to create command pool");
        }

        // Create descriptor pool
        auto desc_result = create_descriptor_pool();
        if (!desc_result.ok()) {
            shutdown();
            return desc_result.error();
        }

        initialized_ = true;
        GRANITE_LOG_INFO("Vulkan backend initialized successfully");
        return {};
    }

    void shutdown() override {
        if (!initialized_ && instance_ == VK_NULL_HANDLE) {
            return;
        }

        GRANITE_LOG_INFO("Shutting down Vulkan backend");

        if (device_ != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(device_);

            // Release all buffers
            for (auto& [handle, buffer] : buffers_) {
                vkDestroyBuffer(device_, buffer, nullptr);
            }
            buffers_.clear();

            for (auto& [handle, memory] : buffer_memory_) {
                vkFreeMemory(device_, memory, nullptr);
            }
            buffer_memory_.clear();

            // Release all pipelines
            for (auto& [handle, pipeline] : pipelines_) {
                vkDestroyPipeline(device_, pipeline, nullptr);
            }
            pipelines_.clear();

            for (auto& [handle, layout] : pipeline_layouts_) {
                vkDestroyPipelineLayout(device_, layout, nullptr);
            }
            pipeline_layouts_.clear();

            // Release descriptor resources
            if (descriptor_pool_ != VK_NULL_HANDLE) {
                vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
                descriptor_pool_ = VK_NULL_HANDLE;
            }

            if (descriptor_set_layout_ != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(device_, descriptor_set_layout_, nullptr);
                descriptor_set_layout_ = VK_NULL_HANDLE;
            }

            // Release command pool
            if (command_pool_ != VK_NULL_HANDLE) {
                vkDestroyCommandPool(device_, command_pool_, nullptr);
                command_pool_ = VK_NULL_HANDLE;
            }

            vkDestroyDevice(device_, nullptr);
            device_ = VK_NULL_HANDLE;
        }

        if (instance_ != VK_NULL_HANDLE) {
            vkDestroyInstance(instance_, nullptr);
            instance_ = VK_NULL_HANDLE;
        }

        physical_device_ = VK_NULL_HANDLE;
        initialized_ = false;
    }

    [[nodiscard]] bool is_initialized() const override {
        return initialized_;
    }

    [[nodiscard]] DeviceCapabilities get_capabilities() const override {
        DeviceCapabilities caps;

        if (physical_device_ != VK_NULL_HANDLE) {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(physical_device_, &props);

            caps.name = props.deviceName;
            caps.max_buffer_size = props.limits.maxStorageBufferRange;
            caps.max_threadgroup_size = props.limits.maxComputeWorkGroupSize[0];
            caps.shared_memory_size = props.limits.maxComputeSharedMemorySize;
            caps.simd_width = 32;  // Typical, varies by vendor
            caps.supports_fp16 = supports_fp16_;
            caps.supports_bf16 = false;
            caps.supports_int8 = true;
            caps.supports_int4 = true;
            caps.supports_simd_groups = supports_subgroups_;
        }

        return caps;
    }

    [[nodiscard]] BackendType get_type() const override {
        return BackendType::Vulkan;
    }

    Result<BufferHandle> create_buffer(const BufferDesc& desc) override {
        if (!initialized_) {
            return Error(ErrorCode::BackendNotInitialized);
        }

        VkBufferCreateInfo buffer_info{};
        buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer_info.size = desc.size;
        buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                           VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                           VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VkBuffer buffer;
        VkResult result = vkCreateBuffer(device_, &buffer_info, nullptr, &buffer);
        if (result != VK_SUCCESS) {
            return Error(ErrorCode::AllocationFailed,
                        fmt::format("Failed to create Vulkan buffer of size {}", desc.size));
        }

        // Allocate memory
        VkMemoryRequirements mem_requirements;
        vkGetBufferMemoryRequirements(device_, buffer, &mem_requirements);

        VkMemoryPropertyFlags memory_flags;
        switch (desc.memory_type) {
            case MemoryType::Device:
                memory_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
                break;
            case MemoryType::Shared:
            case MemoryType::Managed:
            default:
                memory_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
                break;
        }

        uint32_t memory_type_index = find_memory_type(mem_requirements.memoryTypeBits, memory_flags);
        if (memory_type_index == UINT32_MAX) {
            vkDestroyBuffer(device_, buffer, nullptr);
            return Error(ErrorCode::AllocationFailed, "No suitable memory type found");
        }

        VkMemoryAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc_info.allocationSize = mem_requirements.size;
        alloc_info.memoryTypeIndex = memory_type_index;

        VkDeviceMemory memory;
        result = vkAllocateMemory(device_, &alloc_info, nullptr, &memory);
        if (result != VK_SUCCESS) {
            vkDestroyBuffer(device_, buffer, nullptr);
            return Error(ErrorCode::AllocationFailed, "Failed to allocate buffer memory");
        }

        vkBindBufferMemory(device_, buffer, memory, 0);

        BufferHandle handle = next_handle_++;
        buffers_[handle] = buffer;
        buffer_memory_[handle] = memory;
        buffer_sizes_[handle] = desc.size;

        return handle;
    }

    void destroy_buffer(BufferHandle handle) override {
        auto buffer_it = buffers_.find(handle);
        auto memory_it = buffer_memory_.find(handle);

        if (buffer_it != buffers_.end()) {
            vkDestroyBuffer(device_, buffer_it->second, nullptr);
            buffers_.erase(buffer_it);
        }

        if (memory_it != buffer_memory_.end()) {
            vkFreeMemory(device_, memory_it->second, nullptr);
            buffer_memory_.erase(memory_it);
        }

        buffer_sizes_.erase(handle);
    }

    Result<void*> map_buffer(BufferHandle handle) override {
        auto memory_it = buffer_memory_.find(handle);
        if (memory_it == buffer_memory_.end()) {
            return Error(ErrorCode::InvalidHandle, "Buffer not found");
        }

        auto size_it = buffer_sizes_.find(handle);
        if (size_it == buffer_sizes_.end()) {
            return Error(ErrorCode::InvalidHandle, "Buffer size not found");
        }

        void* data;
        VkResult result = vkMapMemory(device_, memory_it->second, 0, size_it->second, 0, &data);
        if (result != VK_SUCCESS) {
            return Error(ErrorCode::MapFailed, "Failed to map buffer memory");
        }

        return data;
    }

    void unmap_buffer(BufferHandle handle) override {
        auto memory_it = buffer_memory_.find(handle);
        if (memory_it != buffer_memory_.end()) {
            vkUnmapMemory(device_, memory_it->second);
        }
    }

    Result<void> write_buffer(BufferHandle handle, const void* data, size_t size, size_t offset) override {
        auto map_result = map_buffer(handle);
        if (!map_result.ok()) {
            return map_result.error();
        }

        std::memcpy(static_cast<char*>(map_result.value()) + offset, data, size);
        unmap_buffer(handle);
        return {};
    }

    Result<void> read_buffer(BufferHandle handle, void* data, size_t size, size_t offset) override {
        auto map_result = map_buffer(handle);
        if (!map_result.ok()) {
            return map_result.error();
        }

        std::memcpy(data, static_cast<const char*>(map_result.value()) + offset, size);
        unmap_buffer(handle);
        return {};
    }

    Result<void> copy_buffer(BufferHandle src, BufferHandle dst, size_t size, size_t src_offset, size_t dst_offset) override {
        auto src_it = buffers_.find(src);
        auto dst_it = buffers_.find(dst);

        if (src_it == buffers_.end() || dst_it == buffers_.end()) {
            return Error(ErrorCode::InvalidHandle, "Buffer not found");
        }

        // Create one-time command buffer
        VkCommandBufferAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandPool = command_pool_;
        alloc_info.commandBufferCount = 1;

        VkCommandBuffer cmd_buffer;
        vkAllocateCommandBuffers(device_, &alloc_info, &cmd_buffer);

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(cmd_buffer, &begin_info);

        VkBufferCopy copy_region{};
        copy_region.srcOffset = src_offset;
        copy_region.dstOffset = dst_offset;
        copy_region.size = size;
        vkCmdCopyBuffer(cmd_buffer, src_it->second, dst_it->second, 1, &copy_region);

        vkEndCommandBuffer(cmd_buffer);

        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &cmd_buffer;

        vkQueueSubmit(compute_queue_, 1, &submit_info, VK_NULL_HANDLE);
        vkQueueWaitIdle(compute_queue_);

        vkFreeCommandBuffers(device_, command_pool_, 1, &cmd_buffer);
        return {};
    }

    Result<PipelineHandle> create_pipeline(const PipelineDesc& desc) override {
        if (!initialized_) {
            return Error(ErrorCode::BackendNotInitialized);
        }

        // TODO: Implement shader compilation from SPIR-V
        // For now, return a placeholder
        return Error(ErrorCode::NotImplemented, "Pipeline creation not yet implemented");
    }

    void destroy_pipeline(PipelineHandle handle) override {
        auto pipeline_it = pipelines_.find(handle);
        auto layout_it = pipeline_layouts_.find(handle);

        if (pipeline_it != pipelines_.end()) {
            vkDestroyPipeline(device_, pipeline_it->second, nullptr);
            pipelines_.erase(pipeline_it);
        }

        if (layout_it != pipeline_layouts_.end()) {
            vkDestroyPipelineLayout(device_, layout_it->second, nullptr);
            pipeline_layouts_.erase(layout_it);
        }
    }

    Result<void> begin_commands() override {
        if (current_command_buffer_ != VK_NULL_HANDLE) {
            return Error(ErrorCode::InvalidState, "Command buffer already active");
        }

        VkCommandBufferAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandPool = command_pool_;
        alloc_info.commandBufferCount = 1;

        VkResult result = vkAllocateCommandBuffers(device_, &alloc_info, &current_command_buffer_);
        if (result != VK_SUCCESS) {
            return Error(ErrorCode::InternalError, "Failed to allocate command buffer");
        }

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(current_command_buffer_, &begin_info);
        return {};
    }

    Result<void> bind_pipeline(PipelineHandle handle) override {
        auto it = pipelines_.find(handle);
        if (it == pipelines_.end()) {
            return Error(ErrorCode::InvalidHandle, "Pipeline not found");
        }

        current_pipeline_ = it->second;
        vkCmdBindPipeline(current_command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, current_pipeline_);
        return {};
    }

    Result<void> bind_buffer(uint32_t index, BufferHandle handle, size_t offset) override {
        // TODO: Implement descriptor set binding
        return Error(ErrorCode::NotImplemented, "Buffer binding not yet implemented");
    }

    Result<void> set_push_constants(const void* data, size_t size) override {
        // TODO: Implement push constants
        return Error(ErrorCode::NotImplemented, "Push constants not yet implemented");
    }

    Result<void> dispatch(uint32_t groups_x, uint32_t groups_y, uint32_t groups_z) override {
        if (current_command_buffer_ == VK_NULL_HANDLE) {
            return Error(ErrorCode::InvalidState, "No active command buffer");
        }

        vkCmdDispatch(current_command_buffer_, groups_x, groups_y, groups_z);
        return {};
    }

    Result<void> end_commands() override {
        if (current_command_buffer_ == VK_NULL_HANDLE) {
            return Error(ErrorCode::InvalidState, "No active command buffer");
        }

        vkEndCommandBuffer(current_command_buffer_);
        return {};
    }

    Result<void> submit() override {
        if (current_command_buffer_ == VK_NULL_HANDLE) {
            return Error(ErrorCode::InvalidState, "No command buffer to submit");
        }

        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &current_command_buffer_;

        VkResult result = vkQueueSubmit(compute_queue_, 1, &submit_info, VK_NULL_HANDLE);
        if (result != VK_SUCCESS) {
            return Error(ErrorCode::InternalError, "Failed to submit command buffer");
        }

        return {};
    }

    Result<void> wait_for_completion() override {
        vkQueueWaitIdle(compute_queue_);

        if (current_command_buffer_ != VK_NULL_HANDLE) {
            vkFreeCommandBuffers(device_, command_pool_, 1, &current_command_buffer_);
            current_command_buffer_ = VK_NULL_HANDLE;
        }

        return {};
    }

    // Fence operations
    Result<FenceHandle> create_fence() override {
        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

        VkFence fence;
        VkResult result = vkCreateFence(device_, &fence_info, nullptr, &fence);
        if (result != VK_SUCCESS) {
            return Error(ErrorCode::InternalError, "Failed to create fence");
        }

        FenceHandle handle = next_handle_++;
        fences_[handle] = fence;
        return handle;
    }

    Result<void> wait_fence(FenceHandle handle, uint64_t timeout_ns) override {
        auto it = fences_.find(handle);
        if (it == fences_.end()) {
            return Error(ErrorCode::InvalidHandle, "Fence not found");
        }

        VkResult result = vkWaitForFences(device_, 1, &it->second, VK_TRUE, timeout_ns);
        if (result == VK_TIMEOUT) {
            return Error(ErrorCode::Timeout, "Fence wait timed out");
        }
        if (result != VK_SUCCESS) {
            return Error(ErrorCode::InternalError, "Fence wait failed");
        }

        return {};
    }

    void destroy_fence(FenceHandle handle) override {
        auto it = fences_.find(handle);
        if (it != fences_.end()) {
            vkDestroyFence(device_, it->second, nullptr);
            fences_.erase(it);
        }
    }

    // Native handle access
    void* get_native_device() override {
        return device_;
    }

    void* get_native_buffer(BufferHandle handle) override {
        auto it = buffers_.find(handle);
        return it != buffers_.end() ? it->second : nullptr;
    }

private:
    Result<void> select_physical_device() {
        uint32_t device_count = 0;
        vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);

        if (device_count == 0) {
            return Error(ErrorCode::BackendNotSupported, "No Vulkan-capable GPU found");
        }

        std::vector<VkPhysicalDevice> devices(device_count);
        vkEnumeratePhysicalDevices(instance_, &device_count, devices.data());

        // Select first device with compute queue (prefer discrete GPU)
        VkPhysicalDevice selected = VK_NULL_HANDLE;
        for (const auto& device : devices) {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(device, &props);

            // Check for compute queue
            uint32_t queue_family_count = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, nullptr);

            std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
            vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_families.data());

            for (uint32_t i = 0; i < queue_family_count; i++) {
                if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                    compute_queue_family_ = i;
                    selected = device;

                    // Prefer discrete GPU
                    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                        GRANITE_LOG_INFO("Selected discrete GPU: {}", props.deviceName);
                        physical_device_ = device;
                        return {};
                    }
                    break;
                }
            }
        }

        if (selected != VK_NULL_HANDLE) {
            physical_device_ = selected;
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(physical_device_, &props);
            GRANITE_LOG_INFO("Selected GPU: {}", props.deviceName);
            return {};
        }

        return Error(ErrorCode::BackendNotSupported, "No suitable GPU found");
    }

    Result<void> create_logical_device() {
        float queue_priority = 1.0f;
        VkDeviceQueueCreateInfo queue_create_info{};
        queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_create_info.queueFamilyIndex = compute_queue_family_;
        queue_create_info.queueCount = 1;
        queue_create_info.pQueuePriorities = &queue_priority;

        VkPhysicalDeviceFeatures device_features{};

        VkDeviceCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        create_info.queueCreateInfoCount = 1;
        create_info.pQueueCreateInfos = &queue_create_info;
        create_info.pEnabledFeatures = &device_features;

        VkResult result = vkCreateDevice(physical_device_, &create_info, nullptr, &device_);
        if (result != VK_SUCCESS) {
            return Error(ErrorCode::InternalError, "Failed to create logical device");
        }

        vkGetDeviceQueue(device_, compute_queue_family_, 0, &compute_queue_);

        // Check for subgroup and FP16 support
        VkPhysicalDeviceSubgroupProperties subgroup_props{};
        subgroup_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;

        VkPhysicalDeviceProperties2 props2{};
        props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props2.pNext = &subgroup_props;
        vkGetPhysicalDeviceProperties2(physical_device_, &props2);

        supports_subgroups_ = (subgroup_props.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0;
        subgroup_size_ = subgroup_props.subgroupSize;

        GRANITE_LOG_INFO("Subgroup size: {}, arithmetic support: {}",
                        subgroup_size_, supports_subgroups_ ? "yes" : "no");

        return {};
    }

    Result<void> create_descriptor_pool() {
        // Create a large descriptor pool for storage buffers
        VkDescriptorPoolSize pool_size{};
        pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pool_size.descriptorCount = 1000;  // Generous limit

        VkDescriptorPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = &pool_size;
        pool_info.maxSets = 100;
        pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

        VkResult result = vkCreateDescriptorPool(device_, &pool_info, nullptr, &descriptor_pool_);
        if (result != VK_SUCCESS) {
            return Error(ErrorCode::InternalError, "Failed to create descriptor pool");
        }

        // Create default descriptor set layout for compute shaders
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        for (uint32_t i = 0; i < 8; i++) {  // 8 buffer bindings
            VkDescriptorSetLayoutBinding binding{};
            binding.binding = i;
            binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            binding.descriptorCount = 1;
            binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            bindings.push_back(binding);
        }

        VkDescriptorSetLayoutCreateInfo layout_info{};
        layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
        layout_info.pBindings = bindings.data();

        result = vkCreateDescriptorSetLayout(device_, &layout_info, nullptr, &descriptor_set_layout_);
        if (result != VK_SUCCESS) {
            return Error(ErrorCode::InternalError, "Failed to create descriptor set layout");
        }

        return {};
    }

    uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties mem_properties;
        vkGetPhysicalDeviceMemoryProperties(physical_device_, &mem_properties);

        for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
            if ((type_filter & (1 << i)) &&
                (mem_properties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        return UINT32_MAX;
    }

    // State
    bool initialized_ = false;
    std::atomic<uint64_t> next_handle_{1};

    // Vulkan objects
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue compute_queue_ = VK_NULL_HANDLE;
    uint32_t compute_queue_family_ = 0;

    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer current_command_buffer_ = VK_NULL_HANDLE;

    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layout_ = VK_NULL_HANDLE;

    VkPipeline current_pipeline_ = VK_NULL_HANDLE;

    // Device capabilities
    bool supports_fp16_ = false;
    bool supports_subgroups_ = false;
    uint32_t subgroup_size_ = 32;

    // Resource maps
    std::unordered_map<BufferHandle, VkBuffer> buffers_;
    std::unordered_map<BufferHandle, VkDeviceMemory> buffer_memory_;
    std::unordered_map<BufferHandle, size_t> buffer_sizes_;
    std::unordered_map<PipelineHandle, VkPipeline> pipelines_;
    std::unordered_map<PipelineHandle, VkPipelineLayout> pipeline_layouts_;
    std::unordered_map<FenceHandle, VkFence> fences_;
};

// Factory function
std::unique_ptr<IComputeBackend> create_vulkan_backend() {
    return std::make_unique<VulkanBackend>();
}

} // namespace granite

#endif // GRANITE_HAS_VULKAN

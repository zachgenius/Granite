// vulkan_backend.cpp - Vulkan compute backend for Granite
//
// This backend provides GPU acceleration via Vulkan for Linux/Android/Windows.
// Shaders are adapted from llama.cpp's MIT-licensed Vulkan backend.

#include "granite/backend.h"
#include "granite/log.h"
#include "backend/vulkan/vulkan_compute.h"

#ifdef GRANITE_HAS_VULKAN

#include <vulkan/vulkan.h>

#include <unordered_map>
#include <atomic>
#include <string>
#include <vector>
#include <optional>
#include <cstring>
#include <fstream>
#include <filesystem>

#ifdef GRANITE_HAS_SHADERC
#include <shaderc/shaderc.hpp>
#endif

namespace granite {

#ifdef GRANITE_HAS_SHADERC
namespace {
class ShadercIncluder : public shaderc::CompileOptions::IncluderInterface {
public:
    explicit ShadercIncluder(std::filesystem::path base_dir)
        : base_dir_(std::move(base_dir)) {}

    shaderc_include_result* GetInclude(const char* requested_source,
                                       shaderc_include_type type,
                                       const char* requesting_source,
                                       size_t /*include_depth*/) override {
        std::filesystem::path include_path;

        if (type == shaderc_include_type_relative && requesting_source) {
            std::filesystem::path requester(requesting_source);
            include_path = requester.parent_path() / requested_source;
        } else {
            include_path = base_dir_ / requested_source;
        }

        std::string content;
        std::ifstream file(include_path);
        if (file.is_open()) {
            content.assign((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
        }

        auto* result = new shaderc_include_result();
        auto* data = new IncludeData();

        if (content.empty()) {
            data->content = "";
            data->source_name = include_path.string();
            result->source_name = data->source_name.c_str();
            result->source_name_length = data->source_name.size();
            result->content = "";
            result->content_length = 0;
            result->user_data = data;
            return result;
        }

        data->content = std::move(content);
        data->source_name = include_path.string();

        result->source_name = data->source_name.c_str();
        result->source_name_length = data->source_name.size();
        result->content = data->content.c_str();
        result->content_length = data->content.size();
        result->user_data = data;

        return result;
    }

    void ReleaseInclude(shaderc_include_result* include_result) override {
        if (!include_result) {
            return;
        }
        delete static_cast<IncludeData*>(include_result->user_data);
        delete include_result;
    }

private:
    struct IncludeData {
        std::string content;
        std::string source_name;
    };

    std::filesystem::path base_dir_;
};
}  // namespace
#endif

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
        app_info.apiVersion = VK_API_VERSION_1_1;

        VkInstanceCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        create_info.pApplicationInfo = &app_info;

        // Enable portability enumeration when available (required for MoltenVK)
        std::vector<const char*> instance_extensions;
        uint32_t extension_count = 0;
        if (vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, nullptr) == VK_SUCCESS &&
            extension_count > 0) {
            std::vector<VkExtensionProperties> extensions(extension_count);
            if (vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, extensions.data()) == VK_SUCCESS) {
                for (const auto& ext : extensions) {
                    if (std::strcmp(ext.extensionName, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) == 0) {
                        instance_extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
                        create_info.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
                    }
                }
            }
        }

        if (!instance_extensions.empty()) {
            create_info.enabledExtensionCount = static_cast<uint32_t>(instance_extensions.size());
            create_info.ppEnabledExtensionNames = instance_extensions.data();
        }

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

        detect_memory_properties();

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

        release_vulkan_compute(this);

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

        VkMemoryPropertyFlags required_flags;
        VkMemoryPropertyFlags preferred_flags = 0;
        switch (desc.memory_type) {
            case MemoryType::Device:
                required_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
                if (supports_unified_memory_) {
                    preferred_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
                }
                break;
            case MemoryType::Shared:
            case MemoryType::Managed:
            default:
                required_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
                if (supports_unified_memory_) {
                    preferred_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
                }
                break;
        }

        uint32_t memory_type_index = find_memory_type(mem_requirements.memoryTypeBits,
                                                      required_flags,
                                                      preferred_flags);
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

        BufferHandle handle{next_handle_++};
        buffers_[handle] = buffer;
        buffer_memory_[handle] = memory;
        buffer_sizes_[handle] = desc.size;
        buffer_memory_props_[handle] = get_memory_properties(memory_type_index);

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
        buffer_memory_props_.erase(handle);
    }

    Result<void*> map_buffer(BufferHandle handle) override {
        auto memory_it = buffer_memory_.find(handle);
        if (memory_it == buffer_memory_.end()) {
            return Error(ErrorCode::InvalidArgument, "Buffer not found");
        }

        auto size_it = buffer_sizes_.find(handle);
        if (size_it == buffer_sizes_.end()) {
            return Error(ErrorCode::InvalidArgument, "Buffer size not found");
        }

        auto props_it = buffer_memory_props_.find(handle);
        if (props_it == buffer_memory_props_.end() ||
            (props_it->second & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0) {
            return Error(ErrorCode::InvalidArgument, "Buffer is not host-visible");
        }

        void* data;
        VkResult result = vkMapMemory(device_, memory_it->second, 0, size_it->second, 0, &data);
        if (result != VK_SUCCESS) {
            return Error(ErrorCode::InternalError, "Failed to map buffer memory");
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
        if (!is_host_visible(handle)) {
            BufferDesc staging_desc{};
            staging_desc.size = size;
            staging_desc.memory_type = MemoryType::Shared;

            auto staging_result = create_buffer(staging_desc);
            if (!staging_result.ok()) {
                return staging_result.error();
            }
            BufferHandle staging = staging_result.value();

            auto map_result = map_buffer(staging);
            if (!map_result.ok()) {
                destroy_buffer(staging);
                return map_result.error();
            }

            std::memcpy(static_cast<char*>(map_result.value()), data, size);
            unmap_buffer(staging);

            auto copy_result = copy_buffer(staging, handle, size, 0, offset);
            destroy_buffer(staging);
            return copy_result;
        }

        auto map_result = map_buffer(handle);
        if (!map_result.ok()) {
            return map_result.error();
        }

        std::memcpy(static_cast<char*>(map_result.value()) + offset, data, size);
        unmap_buffer(handle);
        return {};
    }

    Result<void> read_buffer(BufferHandle handle, void* data, size_t size, size_t offset) override {
        if (!is_host_visible(handle)) {
            BufferDesc staging_desc{};
            staging_desc.size = size;
            staging_desc.memory_type = MemoryType::Shared;

            auto staging_result = create_buffer(staging_desc);
            if (!staging_result.ok()) {
                return staging_result.error();
            }
            BufferHandle staging = staging_result.value();

            auto copy_result = copy_buffer(handle, staging, size, offset, 0);
            if (!copy_result.ok()) {
                destroy_buffer(staging);
                return copy_result;
            }

            auto map_result = map_buffer(staging);
            if (!map_result.ok()) {
                destroy_buffer(staging);
                return map_result.error();
            }

            std::memcpy(data, static_cast<const char*>(map_result.value()), size);
            unmap_buffer(staging);
            destroy_buffer(staging);
            return {};
        }

        auto map_result = map_buffer(handle);
        if (!map_result.ok()) {
            return map_result.error();
        }

        std::memcpy(data, static_cast<const char*>(map_result.value()) + offset, size);
        unmap_buffer(handle);
        return {};
    }

    Result<BufferHandle> create_buffer_from_host(const void* data,
                                                 const BufferDesc& desc) override {
        if (!data) {
            return Error(ErrorCode::InvalidArgument, "Null host buffer");
        }

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

    Result<void> copy_buffer(BufferHandle src, BufferHandle dst, size_t size, size_t src_offset, size_t dst_offset) override {
        auto src_it = buffers_.find(src);
        auto dst_it = buffers_.find(dst);

        if (src_it == buffers_.end() || dst_it == buffers_.end()) {
            return Error(ErrorCode::InvalidArgument, "Buffer not found");
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

        std::vector<uint32_t> spirv;
        std::filesystem::path shader_path(desc.shader_source);
        std::string extension = shader_path.extension().string();

        if (extension == ".comp" || extension == ".glsl") {
            if (auto precompiled = find_precompiled_spirv(shader_path)) {
                auto spirv_result = load_spirv_file(*precompiled);
                if (!spirv_result.ok()) {
                    return spirv_result.error();
                }
                spirv = std::move(spirv_result).take();
            } else {
#ifdef GRANITE_HAS_SHADERC
                std::ifstream file(desc.shader_source);
                if (!file.is_open()) {
                    return Error(ErrorCode::FileNotFound,
                                 fmt::format("GLSL shader not found: {}", desc.shader_source));
                }
                std::string source((std::istreambuf_iterator<char>(file)),
                                   std::istreambuf_iterator<char>());

                shaderc::Compiler compiler;
                shaderc::CompileOptions options;
                options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_1);
                options.SetIncluder(std::make_unique<ShadercIncluder>(shader_path.parent_path()));

                auto result = compiler.CompileGlslToSpv(
                    source,
                    shaderc_compute_shader,
                    desc.shader_source.c_str(),
                    desc.entry_point.c_str(),
                    options);

                if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
                    return Error(ErrorCode::ShaderCompilationFailed,
                                 fmt::format("GLSL compile failed: {}", result.GetErrorMessage()));
                }

                spirv.assign(result.cbegin(), result.cend());
                write_spirv_cache(shader_path, spirv);
#else
                return Error(ErrorCode::NotImplemented,
                             "GLSL compilation requires GRANITE_WITH_SHADERC");
#endif
            }
        } else {
            // shader_source is expected to be a SPIR-V path for Vulkan
            auto spirv_result = load_spirv_file(desc.shader_source);
            if (!spirv_result.ok()) {
                return spirv_result.error();
            }
            spirv = std::move(spirv_result).take();
        }

        VkShaderModuleCreateInfo module_info{};
        module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        module_info.codeSize = spirv.size() * sizeof(uint32_t);
        module_info.pCode = spirv.data();

        VkShaderModule shader_module = VK_NULL_HANDLE;
        VkResult result = vkCreateShaderModule(device_, &module_info, nullptr, &shader_module);
        if (result != VK_SUCCESS) {
            return Error(ErrorCode::ShaderCompilationFailed, "Failed to create shader module");
        }

        VkPushConstantRange push_constant_range{};
        push_constant_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_constant_range.offset = 0;
        push_constant_range.size = max_push_constant_size_;

        VkPipelineLayoutCreateInfo layout_info{};
        layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layout_info.setLayoutCount = 1;
        layout_info.pSetLayouts = &descriptor_set_layout_;
        layout_info.pushConstantRangeCount = 1;
        layout_info.pPushConstantRanges = &push_constant_range;

        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        result = vkCreatePipelineLayout(device_, &layout_info, nullptr, &pipeline_layout);
        if (result != VK_SUCCESS) {
            vkDestroyShaderModule(device_, shader_module, nullptr);
            return Error(ErrorCode::InternalError, "Failed to create pipeline layout");
        }

        VkComputePipelineCreateInfo pipeline_info{};
        pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeline_info.stage.module = shader_module;
        pipeline_info.stage.pName = desc.entry_point.c_str();
        pipeline_info.layout = pipeline_layout;

        VkPipeline pipeline = VK_NULL_HANDLE;
        result = vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1,
                                          &pipeline_info, nullptr, &pipeline);

        vkDestroyShaderModule(device_, shader_module, nullptr);

        if (result != VK_SUCCESS) {
            vkDestroyPipelineLayout(device_, pipeline_layout, nullptr);
            return Error(ErrorCode::InternalError, "Failed to create compute pipeline");
        }

        PipelineHandle handle{next_handle_++};
        pipelines_[handle] = pipeline;
        pipeline_layouts_[handle] = pipeline_layout;

        return handle;
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
            return Error(ErrorCode::InvalidArgument, "Pipeline not found");
        }

        current_pipeline_ = it->second;
        vkCmdBindPipeline(current_command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, current_pipeline_);

        auto layout_it = pipeline_layouts_.find(handle);
        if (layout_it == pipeline_layouts_.end()) {
            return Error(ErrorCode::InvalidArgument, "Pipeline layout not found");
        }
        current_pipeline_layout_ = layout_it->second;

        VkDescriptorSetAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc_info.descriptorPool = descriptor_pool_;
        alloc_info.descriptorSetCount = 1;
        alloc_info.pSetLayouts = &descriptor_set_layout_;

        VkResult result = vkAllocateDescriptorSets(device_, &alloc_info, &current_descriptor_set_);
        if (result != VK_SUCCESS) {
            return Error(ErrorCode::InternalError, "Failed to allocate descriptor set");
        }

        bound_buffers_.assign(kMaxBufferBindings, {});
        bound_buffer_sizes_.assign(kMaxBufferBindings, 0);
        bound_buffer_valid_.assign(kMaxBufferBindings, false);
        return {};
    }

    Result<void> bind_buffer(uint32_t index, BufferHandle handle, size_t offset) override {
        if (current_command_buffer_ == VK_NULL_HANDLE || current_descriptor_set_ == VK_NULL_HANDLE) {
            return Error(ErrorCode::InvalidState, "No active command buffer or descriptor set");
        }
        if (index >= kMaxBufferBindings) {
            return Error(ErrorCode::InvalidArgument, "Binding index out of range");
        }

        auto buffer_it = buffers_.find(handle);
        if (buffer_it == buffers_.end()) {
            return Error(ErrorCode::InvalidArgument, "Buffer not found");
        }

        auto size_it = buffer_sizes_.find(handle);
        if (size_it == buffer_sizes_.end()) {
            return Error(ErrorCode::InvalidArgument, "Buffer size not found");
        }

        VkDescriptorBufferInfo buffer_info{};
        buffer_info.buffer = buffer_it->second;
        buffer_info.offset = offset;
        buffer_info.range = size_it->second > offset ? size_it->second - offset : 0;

        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = current_descriptor_set_;
        write.dstBinding = index;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &buffer_info;

        vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);

        bound_buffers_[index] = buffer_info;
        bound_buffer_sizes_[index] = buffer_info.range;
        bound_buffer_valid_[index] = true;
        return {};
    }

    Result<void> set_push_constants(const void* data, size_t size) override {
        if (current_command_buffer_ == VK_NULL_HANDLE || current_pipeline_layout_ == VK_NULL_HANDLE) {
            return Error(ErrorCode::InvalidState, "No active pipeline for push constants");
        }
        if (size > max_push_constant_size_) {
            return Error(ErrorCode::InvalidArgument, "Push constants exceed max size");
        }

        vkCmdPushConstants(current_command_buffer_, current_pipeline_layout_,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           static_cast<uint32_t>(size), data);
        return {};
    }

    Result<void> dispatch(uint32_t groups_x, uint32_t groups_y, uint32_t groups_z) override {
        if (current_command_buffer_ == VK_NULL_HANDLE) {
            return Error(ErrorCode::InvalidState, "No active command buffer");
        }

        if (current_descriptor_set_ != VK_NULL_HANDLE) {
            vkCmdBindDescriptorSets(current_command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    current_pipeline_layout_, 0, 1,
                                    &current_descriptor_set_, 0, nullptr);
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

        if (current_descriptor_set_ != VK_NULL_HANDLE) {
            vkFreeDescriptorSets(device_, descriptor_pool_, 1, &current_descriptor_set_);
            current_descriptor_set_ = VK_NULL_HANDLE;
        }

        current_pipeline_layout_ = VK_NULL_HANDLE;
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

        FenceHandle handle{next_handle_++};
        fences_[handle] = fence;
        return handle;
    }

    Result<void> wait_fence(FenceHandle handle, uint64_t timeout_ns) override {
        auto it = fences_.find(handle);
        if (it == fences_.end()) {
            return Error(ErrorCode::InvalidArgument, "Fence not found");
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

    void* get_native_physical_device() override {
        return physical_device_;
    }

    void* get_native_queue() override {
        return compute_queue_;
    }

    uint32_t get_native_queue_family() override {
        return compute_queue_family_;
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

    uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags required,
                              VkMemoryPropertyFlags preferred) {
        VkPhysicalDeviceMemoryProperties mem_properties;
        vkGetPhysicalDeviceMemoryProperties(physical_device_, &mem_properties);

        uint32_t fallback = UINT32_MAX;
        for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
            if ((type_filter & (1 << i)) &&
                (mem_properties.memoryTypes[i].propertyFlags & required) == required) {
                if ((mem_properties.memoryTypes[i].propertyFlags & preferred) == preferred) {
                    return i;
                }
                if (fallback == UINT32_MAX) {
                    fallback = i;
                }
            }
        }

        return fallback;
    }

    VkMemoryPropertyFlags get_memory_properties(uint32_t memory_type_index) const {
        VkPhysicalDeviceMemoryProperties mem_properties;
        vkGetPhysicalDeviceMemoryProperties(physical_device_, &mem_properties);
        if (memory_type_index >= mem_properties.memoryTypeCount) {
            return 0;
        }
        return mem_properties.memoryTypes[memory_type_index].propertyFlags;
    }

    void detect_memory_properties() {
        VkPhysicalDeviceMemoryProperties mem_properties;
        vkGetPhysicalDeviceMemoryProperties(physical_device_, &mem_properties);

        supports_unified_memory_ = false;
        for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
            auto flags = mem_properties.memoryTypes[i].propertyFlags;
            if ((flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
                (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
                supports_unified_memory_ = true;
                break;
            }
        }

        GRANITE_LOG_INFO("Vulkan unified memory: {}", supports_unified_memory_ ? "yes" : "no");
    }

    bool is_host_visible(BufferHandle handle) const {
        auto it = buffer_memory_props_.find(handle);
        if (it == buffer_memory_props_.end()) {
            return false;
        }
        return (it->second & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
    }

    static std::optional<std::filesystem::path> get_precompiled_dir() {
        if (const char* dir = std::getenv("GRANITE_VULKAN_PRECOMPILED_DIR")) {
            if (dir[0] != '\0') {
                return std::filesystem::path(dir);
            }
        }
#ifdef GRANITE_VULKAN_PRECOMPILED_DIR
        return std::filesystem::path(GRANITE_VULKAN_PRECOMPILED_DIR);
#endif
        return std::nullopt;
    }

    static std::optional<std::filesystem::path> get_spirv_cache_dir() {
        if (const char* dir = std::getenv("GRANITE_VULKAN_SPIRV_CACHE_DIR")) {
            if (dir[0] != '\0') {
                return std::filesystem::path(dir);
            }
        }
        return std::nullopt;
    }

    static std::optional<std::filesystem::path> find_precompiled_spirv(
        const std::filesystem::path& shader_path) {
        std::vector<std::filesystem::path> search_dirs;
        if (auto pre_dir = get_precompiled_dir()) {
            search_dirs.push_back(*pre_dir);
        }
        search_dirs.push_back(shader_path.parent_path());

        const std::string filename = shader_path.filename().string();
        const std::string stem = shader_path.stem().string();

        for (const auto& dir : search_dirs) {
            std::filesystem::path candidate1 = dir / (filename + ".spv");
            if (std::filesystem::exists(candidate1)) {
                return candidate1;
            }
            std::filesystem::path candidate2 = dir / (stem + ".spv");
            if (std::filesystem::exists(candidate2)) {
                return candidate2;
            }
        }

        return std::nullopt;
    }

    static Result<std::vector<uint32_t>> load_spirv_file(const std::filesystem::path& path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            return Error(ErrorCode::FileNotFound,
                         fmt::format("SPIR-V not found: {}", path.string()));
        }
        std::streamsize size = file.tellg();
        if (size <= 0) {
            return Error(ErrorCode::InvalidArgument,
                         fmt::format("Empty SPIR-V file: {}", path.string()));
        }
        file.seekg(0, std::ios::beg);
        if (size % 4 != 0) {
            return Error(ErrorCode::InvalidArgument,
                         fmt::format("Invalid SPIR-V size: {}", path.string()));
        }
        std::vector<uint32_t> spirv(static_cast<size_t>(size / 4));
        if (!file.read(reinterpret_cast<char*>(spirv.data()), size)) {
            return Error(ErrorCode::IOError,
                         fmt::format("Failed to read SPIR-V: {}", path.string()));
        }
        return spirv;
    }

    static void write_spirv_cache(const std::filesystem::path& shader_path,
                                  const std::vector<uint32_t>& spirv) {
        auto cache_dir = get_spirv_cache_dir();
        if (!cache_dir) {
            return;
        }

        std::error_code ec;
        std::filesystem::create_directories(*cache_dir, ec);

        std::filesystem::path out_path = *cache_dir /
            (shader_path.filename().string() + ".spv");

        std::ofstream out(out_path, std::ios::binary);
        if (!out.is_open()) {
            return;
        }
        out.write(reinterpret_cast<const char*>(spirv.data()),
                  static_cast<std::streamsize>(spirv.size() * sizeof(uint32_t)));
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
    VkDescriptorSet current_descriptor_set_ = VK_NULL_HANDLE;

    VkPipeline current_pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout current_pipeline_layout_ = VK_NULL_HANDLE;

    // Device capabilities
    bool supports_fp16_ = false;
    bool supports_subgroups_ = false;
    uint32_t subgroup_size_ = 32;
    bool supports_unified_memory_ = false;

    // Resource maps
    std::unordered_map<BufferHandle, VkBuffer> buffers_;
    std::unordered_map<BufferHandle, VkDeviceMemory> buffer_memory_;
    std::unordered_map<BufferHandle, size_t> buffer_sizes_;
    std::unordered_map<BufferHandle, VkMemoryPropertyFlags> buffer_memory_props_;
    std::unordered_map<PipelineHandle, VkPipeline> pipelines_;
    std::unordered_map<PipelineHandle, VkPipelineLayout> pipeline_layouts_;
    std::unordered_map<FenceHandle, VkFence> fences_;

    static constexpr uint32_t kMaxBufferBindings = 8;
    static constexpr uint32_t max_push_constant_size_ = 128;
    std::vector<VkDescriptorBufferInfo> bound_buffers_;
    std::vector<size_t> bound_buffer_sizes_;
    std::vector<bool> bound_buffer_valid_;
};

// Factory function
std::unique_ptr<IComputeBackend> create_vulkan_backend() {
    return std::make_unique<VulkanBackend>();
}

} // namespace granite

#endif // GRANITE_HAS_VULKAN

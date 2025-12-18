// vulkan_compute.cpp - High-level compute interface for Vulkan backend
//
// This file provides the VulkanCompute class that mirrors MetalCompute,
// offering high-level dispatch functions for quantized operations.
//
// Shaders are adapted from llama.cpp's MIT-licensed Vulkan backend.

#include "granite/error.h"
#include "granite/log.h"

#ifdef GRANITE_HAS_VULKAN

#include <vulkan/vulkan.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <cstring>
#include <regex>
#include <chrono>

#ifdef GRANITE_HAS_SHADERC
#include <shaderc/shaderc.hpp>
#endif

namespace granite {

// =============================================================================
// Shader Preprocessor
// =============================================================================

class ShaderPreprocessor {
public:
    explicit ShaderPreprocessor(const std::string& shader_dir)
        : shader_dir_(shader_dir) {}

    // Process a shader file, resolving #include directives
    Result<std::string> process(const std::string& filename,
                                const std::vector<std::pair<std::string, std::string>>& defines = {}) {
        included_files_.clear();
        auto result = process_file(filename, 0);
        if (!result.ok()) {
            return result;
        }

        // Prepend defines
        std::string header = "#version 450\n";
        for (const auto& [name, value] : defines) {
            header += "#define " + name + " " + value + "\n";
        }

        return header + result.value();
    }

private:
    Result<std::string> process_file(const std::string& filename, int depth) {
        if (depth > 32) {
            return Error(ErrorCode::InvalidArgument, "Include depth exceeded");
        }

        // Prevent circular includes
        if (included_files_.count(filename)) {
            return std::string{};
        }
        included_files_.insert(filename);

        std::filesystem::path filepath = shader_dir_ / filename;
        std::ifstream file(filepath);
        if (!file.is_open()) {
            return Error(ErrorCode::FileNotFound,
                        fmt::format("Shader file not found: {}", filepath.string()));
        }

        std::stringstream result;
        std::string line;
        std::regex include_regex(R"(^\s*#include\s*[<"]([^>"]+)[>"])");

        while (std::getline(file, line)) {
            std::smatch match;
            if (std::regex_match(line, match, include_regex)) {
                std::string include_file = match[1].str();
                auto include_result = process_file(include_file, depth + 1);
                if (!include_result.ok()) {
                    return include_result;
                }
                result << include_result.value() << "\n";
            } else if (line.find("#version") == std::string::npos) {
                // Skip #version directives (we add our own)
                result << line << "\n";
            }
        }

        return result.str();
    }

    std::filesystem::path shader_dir_;
    std::unordered_set<std::string> included_files_;
};

// =============================================================================
// VulkanCompute Implementation
// =============================================================================

class VulkanCompute {
public:
    struct ProfilingStats {
        uint64_t dispatch_count = 0;
        uint64_t sync_count = 0;
        double sync_time_ms = 0;
        uint64_t command_buffer_count = 0;
    };

    // Push constant structure for simple operations
    struct PushConstantsSimple {
        uint32_t KX;        // Input size
        uint32_t KY;        // Output size (or second dimension)
        float param1;       // eps for norm, etc.
        float param2;       // Additional parameter
    };

    VulkanCompute() = default;
    ~VulkanCompute() { shutdown(); }

    bool initialize(VkDevice device, VkPhysicalDevice physical_device,
                   VkQueue compute_queue, uint32_t queue_family,
                   const std::string& shader_dir = "") {
        device_ = device;
        physical_device_ = physical_device;
        compute_queue_ = compute_queue;
        queue_family_ = queue_family;
        shader_dir_ = shader_dir;

        // Get device properties
        vkGetPhysicalDeviceProperties(physical_device_, &device_props_);
        subgroup_size_ = 32;  // Default, will be updated

        // Query subgroup size
        VkPhysicalDeviceSubgroupProperties subgroup_props{};
        subgroup_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
        VkPhysicalDeviceProperties2 props2{};
        props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props2.pNext = &subgroup_props;
        vkGetPhysicalDeviceProperties2(physical_device_, &props2);
        subgroup_size_ = subgroup_props.subgroupSize;

        GRANITE_LOG_INFO("VulkanCompute: subgroup size = {}", subgroup_size_);

        // Create command pool
        VkCommandPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.queueFamilyIndex = queue_family_;
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

        if (vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_) != VK_SUCCESS) {
            GRANITE_LOG_ERROR("Failed to create Vulkan command pool");
            return false;
        }

        // Create descriptor pool
        if (!create_descriptor_pool()) {
            GRANITE_LOG_ERROR("Failed to create descriptor pool");
            return false;
        }

        // Create default descriptor set layout (for 4 storage buffers)
        if (!create_descriptor_set_layout(4)) {
            GRANITE_LOG_ERROR("Failed to create descriptor set layout");
            return false;
        }

        // Load and compile shaders
        if (!shader_dir_.empty()) {
            compile_builtin_shaders();
        }

        initialized_ = true;
        GRANITE_LOG_INFO("VulkanCompute initialized");
        return true;
    }

    void shutdown() {
        if (!initialized_) return;

        if (device_ != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(device_);

            // Destroy shader modules
            for (auto& [name, module] : shader_modules_) {
                vkDestroyShaderModule(device_, module, nullptr);
            }
            shader_modules_.clear();

            // Destroy pipelines
            for (auto& [name, pipeline] : pipelines_) {
                vkDestroyPipeline(device_, pipeline.pipeline, nullptr);
                vkDestroyPipelineLayout(device_, pipeline.layout, nullptr);
            }
            pipelines_.clear();

            // Destroy descriptor resources
            if (descriptor_pool_ != VK_NULL_HANDLE) {
                vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
                descriptor_pool_ = VK_NULL_HANDLE;
            }

            for (auto& [count, layout] : descriptor_set_layouts_) {
                vkDestroyDescriptorSetLayout(device_, layout, nullptr);
            }
            descriptor_set_layouts_.clear();

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
            auto start = std::chrono::high_resolution_clock::now();
            vkDeviceWaitIdle(device_);
            auto end = std::chrono::high_resolution_clock::now();
            stats_.sync_count++;
            stats_.sync_time_ms += std::chrono::duration<double, std::milli>(end - start).count();
        }
    }

    // =========================================================================
    // Activation Operations
    // =========================================================================

    Result<void> silu(VkBuffer x, VkBuffer out, uint32_t size) {
        auto* pipeline = get_pipeline("silu");
        if (!pipeline) {
            return Error(ErrorCode::NotImplemented, "SiLU pipeline not available");
        }

        PushConstantsSimple pc{};
        pc.KX = size;

        return dispatch_simple(*pipeline, {x, out}, pc, (size + 511) / 512);
    }

    Result<void> gelu(VkBuffer x, VkBuffer out, uint32_t size) {
        auto* pipeline = get_pipeline("gelu");
        if (!pipeline) {
            return Error(ErrorCode::NotImplemented, "GELU pipeline not available");
        }

        PushConstantsSimple pc{};
        pc.KX = size;

        return dispatch_simple(*pipeline, {x, out}, pc, (size + 511) / 512);
    }

    // =========================================================================
    // Elementwise Operations
    // =========================================================================

    Result<void> add(VkBuffer a, VkBuffer b, VkBuffer out, uint32_t size) {
        auto* pipeline = get_pipeline("add");
        if (!pipeline) {
            return Error(ErrorCode::NotImplemented, "Add pipeline not available");
        }

        PushConstantsSimple pc{};
        pc.KX = size;

        return dispatch_simple(*pipeline, {a, b, out}, pc, (size + 511) / 512);
    }

    Result<void> mul(VkBuffer a, VkBuffer b, VkBuffer out, uint32_t size) {
        auto* pipeline = get_pipeline("mul");
        if (!pipeline) {
            return Error(ErrorCode::NotImplemented, "Mul pipeline not available");
        }

        PushConstantsSimple pc{};
        pc.KX = size;

        return dispatch_simple(*pipeline, {a, b, out}, pc, (size + 511) / 512);
    }

    // =========================================================================
    // Normalization Operations
    // =========================================================================

    Result<void> rms_norm(VkBuffer x, VkBuffer weight, VkBuffer out,
                         uint32_t size, float eps) {
        auto* pipeline = get_pipeline("rms_norm_simple");
        if (!pipeline) {
            return Error(ErrorCode::NotImplemented, "RMSNorm pipeline not available");
        }

        PushConstantsSimple pc{};
        pc.KX = size;
        pc.param1 = eps;

        // One workgroup per row
        return dispatch_simple(*pipeline, {x, weight, out}, pc, 1);
    }

    // =========================================================================
    // Quantized Matrix-Vector Operations
    // =========================================================================

    // Push constants for matvec operations
    struct PushConstantsMatvec {
        uint32_t K;         // Input dimension (hidden size)
        uint32_t N;         // Output dimension (number of rows)
        uint32_t num_blocks_per_row;  // K / 256 for Q4_K
        uint32_t padding;
    };

    Result<void> matvec_q4k(VkBuffer x, VkBuffer W, VkBuffer y,
                           uint32_t K, uint32_t N) {
        auto* pipeline = get_pipeline("matvec_q4k_simple");
        if (!pipeline) {
            return Error(ErrorCode::NotImplemented, "Q4_K matvec pipeline not available");
        }

        // Q4_K has 256 elements per super-block
        constexpr uint32_t SUPER_BLOCK_SIZE = 256;
        uint32_t num_blocks = K / SUPER_BLOCK_SIZE;

        PushConstantsSimple pc{};
        pc.KX = K;
        pc.KY = N;
        pc.param1 = static_cast<float>(num_blocks);  // num_blocks_per_row

        // One workgroup per output row
        return dispatch_simple(*pipeline, {x, W, y}, pc, N);
    }

    // =========================================================================
    // Softmax (for attention)
    // =========================================================================

    Result<void> softmax(VkBuffer x, VkBuffer out, uint32_t size, float scale = 1.0f) {
        auto* pipeline = get_pipeline("softmax");
        if (!pipeline) {
            return Error(ErrorCode::NotImplemented, "Softmax pipeline not available");
        }

        PushConstantsSimple pc{};
        pc.KX = size;
        pc.param1 = scale;

        // One workgroup for the softmax
        return dispatch_simple(*pipeline, {x, out}, pc, 1);
    }

    Result<void> softmax_rows(VkBuffer x, VkBuffer out, uint32_t rows, uint32_t cols,
                              float scale = 1.0f) {
        auto* pipeline = get_pipeline("softmax");
        if (!pipeline) {
            return Error(ErrorCode::NotImplemented, "Softmax pipeline not available");
        }

        PushConstantsSimple pc{};
        pc.KX = cols;  // Each row has 'cols' elements
        pc.KY = rows;
        pc.param1 = scale;

        // One workgroup per row
        return dispatch_simple(*pipeline, {x, out}, pc, rows);
    }

    // =========================================================================
    // Profiling
    // =========================================================================

    void enable_profiling(bool enable) { profiling_enabled_ = enable; }
    ProfilingStats get_stats() const { return stats_; }
    void reset_stats() { stats_ = ProfilingStats{}; }

    // =========================================================================
    // Shader Compilation
    // =========================================================================

    Result<void> compile_shader(const std::string& name, const std::string& glsl_code,
                               uint32_t num_buffers = 4) {
#ifdef GRANITE_HAS_SHADERC
        shaderc::Compiler compiler;
        shaderc::CompileOptions options;
        options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_0);
        options.SetOptimizationLevel(shaderc_optimization_level_performance);

        auto result = compiler.CompileGlslToSpv(
            glsl_code, shaderc_glsl_compute_shader, name.c_str(), options);

        if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
            return Error(ErrorCode::InvalidArgument,
                        fmt::format("Shader compilation failed: {}", result.GetErrorMessage()));
        }

        std::vector<uint32_t> spirv(result.cbegin(), result.cend());
        return create_pipeline_from_spirv(name, spirv, num_buffers);
#else
        return Error(ErrorCode::NotImplemented,
                    "Runtime shader compilation requires shaderc");
#endif
    }

    Result<void> load_spirv(const std::string& name, const std::vector<uint32_t>& spirv,
                           uint32_t num_buffers = 4) {
        return create_pipeline_from_spirv(name, spirv, num_buffers);
    }

private:
    struct Pipeline {
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkDescriptorSetLayout desc_layout = VK_NULL_HANDLE;
        uint32_t num_buffers = 0;
    };

    bool create_descriptor_pool() {
        VkDescriptorPoolSize pool_size{};
        pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pool_size.descriptorCount = 1000;

        VkDescriptorPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = &pool_size;
        pool_info.maxSets = 100;
        pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

        return vkCreateDescriptorPool(device_, &pool_info, nullptr, &descriptor_pool_) == VK_SUCCESS;
    }

    bool create_descriptor_set_layout(uint32_t num_buffers) {
        if (descriptor_set_layouts_.count(num_buffers)) {
            return true;
        }

        std::vector<VkDescriptorSetLayoutBinding> bindings(num_buffers);
        for (uint32_t i = 0; i < num_buffers; i++) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }

        VkDescriptorSetLayoutCreateInfo layout_info{};
        layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout_info.bindingCount = num_buffers;
        layout_info.pBindings = bindings.data();

        VkDescriptorSetLayout layout;
        if (vkCreateDescriptorSetLayout(device_, &layout_info, nullptr, &layout) != VK_SUCCESS) {
            return false;
        }

        descriptor_set_layouts_[num_buffers] = layout;
        return true;
    }

    Result<void> create_pipeline_from_spirv(const std::string& name,
                                            const std::vector<uint32_t>& spirv,
                                            uint32_t num_buffers) {
        // Create shader module
        VkShaderModuleCreateInfo module_info{};
        module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        module_info.codeSize = spirv.size() * sizeof(uint32_t);
        module_info.pCode = spirv.data();

        VkShaderModule shader_module;
        if (vkCreateShaderModule(device_, &module_info, nullptr, &shader_module) != VK_SUCCESS) {
            return Error(ErrorCode::InternalError, "Failed to create shader module");
        }

        // Ensure descriptor set layout exists
        if (!create_descriptor_set_layout(num_buffers)) {
            vkDestroyShaderModule(device_, shader_module, nullptr);
            return Error(ErrorCode::InternalError, "Failed to create descriptor set layout");
        }

        VkDescriptorSetLayout desc_layout = descriptor_set_layouts_[num_buffers];

        // Create pipeline layout with push constants
        VkPushConstantRange push_constant_range{};
        push_constant_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_constant_range.offset = 0;
        push_constant_range.size = sizeof(PushConstantsSimple);

        VkPipelineLayoutCreateInfo layout_info{};
        layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layout_info.setLayoutCount = 1;
        layout_info.pSetLayouts = &desc_layout;
        layout_info.pushConstantRangeCount = 1;
        layout_info.pPushConstantRanges = &push_constant_range;

        VkPipelineLayout pipeline_layout;
        if (vkCreatePipelineLayout(device_, &layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
            vkDestroyShaderModule(device_, shader_module, nullptr);
            return Error(ErrorCode::InternalError, "Failed to create pipeline layout");
        }

        // Create compute pipeline
        VkComputePipelineCreateInfo pipeline_info{};
        pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeline_info.stage.module = shader_module;
        pipeline_info.stage.pName = "main";
        pipeline_info.layout = pipeline_layout;

        VkPipeline pipeline;
        VkResult result = vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1,
                                                   &pipeline_info, nullptr, &pipeline);

        // Shader module can be destroyed after pipeline creation
        vkDestroyShaderModule(device_, shader_module, nullptr);

        if (result != VK_SUCCESS) {
            vkDestroyPipelineLayout(device_, pipeline_layout, nullptr);
            return Error(ErrorCode::InternalError, "Failed to create compute pipeline");
        }

        Pipeline p{};
        p.pipeline = pipeline;
        p.layout = pipeline_layout;
        p.desc_layout = desc_layout;
        p.num_buffers = num_buffers;
        pipelines_[name] = p;

        GRANITE_LOG_DEBUG("Created pipeline: {}", name);
        return {};
    }

    Pipeline* get_pipeline(const std::string& name) {
        auto it = pipelines_.find(name);
        return it != pipelines_.end() ? &it->second : nullptr;
    }

    Result<void> dispatch_simple(const Pipeline& pipeline,
                                const std::vector<VkBuffer>& buffers,
                                const PushConstantsSimple& push_constants,
                                uint32_t groups_x,
                                uint32_t groups_y = 1,
                                uint32_t groups_z = 1) {
        // Allocate descriptor set
        VkDescriptorSetAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc_info.descriptorPool = descriptor_pool_;
        alloc_info.descriptorSetCount = 1;
        alloc_info.pSetLayouts = &pipeline.desc_layout;

        VkDescriptorSet descriptor_set;
        if (vkAllocateDescriptorSets(device_, &alloc_info, &descriptor_set) != VK_SUCCESS) {
            return Error(ErrorCode::InternalError, "Failed to allocate descriptor set");
        }

        // Update descriptor set with buffers
        std::vector<VkDescriptorBufferInfo> buffer_infos(buffers.size());
        std::vector<VkWriteDescriptorSet> writes(buffers.size());

        for (size_t i = 0; i < buffers.size(); i++) {
            buffer_infos[i].buffer = buffers[i];
            buffer_infos[i].offset = 0;
            buffer_infos[i].range = VK_WHOLE_SIZE;

            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = descriptor_set;
            writes[i].dstBinding = static_cast<uint32_t>(i);
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &buffer_infos[i];
        }

        vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()),
                              writes.data(), 0, nullptr);

        // Create command buffer
        VkCommandBufferAllocateInfo cmd_alloc_info{};
        cmd_alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmd_alloc_info.commandPool = command_pool_;
        cmd_alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmd_alloc_info.commandBufferCount = 1;

        VkCommandBuffer cmd_buffer;
        if (vkAllocateCommandBuffers(device_, &cmd_alloc_info, &cmd_buffer) != VK_SUCCESS) {
            vkFreeDescriptorSets(device_, descriptor_pool_, 1, &descriptor_set);
            return Error(ErrorCode::InternalError, "Failed to allocate command buffer");
        }

        // Record commands
        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(cmd_buffer, &begin_info);

        vkCmdBindPipeline(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
        vkCmdBindDescriptorSets(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline.layout, 0, 1, &descriptor_set, 0, nullptr);
        vkCmdPushConstants(cmd_buffer, pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstantsSimple), &push_constants);
        vkCmdDispatch(cmd_buffer, groups_x, groups_y, groups_z);

        vkEndCommandBuffer(cmd_buffer);

        // Submit
        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &cmd_buffer;

        vkQueueSubmit(compute_queue_, 1, &submit_info, VK_NULL_HANDLE);
        vkQueueWaitIdle(compute_queue_);

        // Cleanup
        vkFreeCommandBuffers(device_, command_pool_, 1, &cmd_buffer);
        vkFreeDescriptorSets(device_, descriptor_pool_, 1, &descriptor_set);

        stats_.dispatch_count++;
        return {};
    }

    void compile_builtin_shaders() {
#ifdef GRANITE_HAS_SHADERC
        // Compile simplified standalone shaders
        const char* silu_shader = R"(
#version 450
#extension GL_EXT_shader_16bit_storage : require

layout(local_size_x = 512, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint KX;
    uint KY;
    float param1;
    float param2;
} p;

layout(binding = 0) readonly buffer X { float data_x[]; };
layout(binding = 1) writeonly buffer D { float data_d[]; };

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= p.KX) return;

    float x = data_x[i];
    data_d[i] = x / (1.0 + exp(-x));
}
)";

        const char* gelu_shader = R"(
#version 450
#extension GL_EXT_shader_16bit_storage : require

layout(local_size_x = 512, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint KX;
    uint KY;
    float param1;
    float param2;
} p;

layout(binding = 0) readonly buffer X { float data_x[]; };
layout(binding = 1) writeonly buffer D { float data_d[]; };

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= p.KX) return;

    float x = data_x[i];
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float SQRT_2_OVER_PI = 0.7978845608;
    float x3 = x * x * x;
    data_d[i] = 0.5 * x * (1.0 + tanh(SQRT_2_OVER_PI * (x + 0.044715 * x3)));
}
)";

        const char* add_shader = R"(
#version 450

layout(local_size_x = 512, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint KX;
    uint KY;
    float param1;
    float param2;
} p;

layout(binding = 0) readonly buffer A { float data_a[]; };
layout(binding = 1) readonly buffer B { float data_b[]; };
layout(binding = 2) writeonly buffer D { float data_d[]; };

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= p.KX) return;

    data_d[i] = data_a[i] + data_b[i];
}
)";

        const char* mul_shader = R"(
#version 450

layout(local_size_x = 512, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint KX;
    uint KY;
    float param1;
    float param2;
} p;

layout(binding = 0) readonly buffer A { float data_a[]; };
layout(binding = 1) readonly buffer B { float data_b[]; };
layout(binding = 2) writeonly buffer D { float data_d[]; };

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= p.KX) return;

    data_d[i] = data_a[i] * data_b[i];
}
)";

        const char* rms_norm_shader = R"(
#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint KX;     // Hidden dimension
    uint KY;
    float param1; // eps
    float param2;
} p;

layout(binding = 0) readonly buffer X { float data_x[]; };
layout(binding = 1) readonly buffer W { float data_w[]; };
layout(binding = 2) writeonly buffer D { float data_d[]; };

shared float shared_sum[256];

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint gid = gl_WorkGroupID.x;
    uint n = p.KX;

    // Compute sum of squares for this row
    float sum = 0.0;
    for (uint i = tid; i < n; i += 256) {
        float val = data_x[gid * n + i];
        sum += val * val;
    }

    shared_sum[tid] = sum;
    barrier();

    // Parallel reduction
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        barrier();
    }

    float rms = sqrt(shared_sum[0] / float(n) + p.param1);
    float scale = 1.0 / rms;

    // Apply normalization and weight
    for (uint i = tid; i < n; i += 256) {
        data_d[gid * n + i] = data_x[gid * n + i] * scale * data_w[i];
    }
}
)";

        // Compile shaders
        auto silu_result = compile_shader("silu", silu_shader, 2);
        if (!silu_result.ok()) {
            GRANITE_LOG_WARN("Failed to compile silu shader: {}", silu_result.error().message());
        }

        auto gelu_result = compile_shader("gelu", gelu_shader, 2);
        if (!gelu_result.ok()) {
            GRANITE_LOG_WARN("Failed to compile gelu shader: {}", gelu_result.error().message());
        }

        auto add_result = compile_shader("add", add_shader, 3);
        if (!add_result.ok()) {
            GRANITE_LOG_WARN("Failed to compile add shader: {}", add_result.error().message());
        }

        auto mul_result = compile_shader("mul", mul_shader, 3);
        if (!mul_result.ok()) {
            GRANITE_LOG_WARN("Failed to compile mul shader: {}", mul_result.error().message());
        }

        auto rms_result = compile_shader("rms_norm_simple", rms_norm_shader, 3);
        if (!rms_result.ok()) {
            GRANITE_LOG_WARN("Failed to compile rms_norm shader: {}", rms_result.error().message());
        }

        // Simplified Q4_K matvec shader
        // Q4_K format: 256 elements per super-block, 4-bit quantization with K-means clustering
        // Block structure: d (fp16), dmin (fp16), scales[12] (6-bit packed), qs[128] (4-bit weights)
        const char* matvec_q4k_shader = R"(
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage : require

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint K;              // Input dimension
    uint N;              // Output rows
    float num_blocks;    // Number of blocks per row
    float padding;
} p;

// Q4_K block structure (144 bytes per 256 elements)
struct block_q4_k {
    float16_t d;         // Global scale
    float16_t dmin;      // Global minimum
    uint8_t scales[12];  // Sub-block scales (6-bit packed)
    uint8_t qs[128];     // Quantized weights (4-bit, 2 per byte)
};

layout(binding = 0) readonly buffer X { float data_x[]; };
layout(binding = 1) readonly buffer W { block_q4_k data_w[]; };
layout(binding = 2) writeonly buffer Y { float data_y[]; };

shared float shared_sum[256];

// Unpack 6-bit scales from 12 bytes into 8 scale values
void unpack_scales(uint8_t packed[12], out float scales[8], out float mins[8]) {
    // Simplified scale unpacking for Q4_K
    for (int i = 0; i < 4; i++) {
        scales[i] = float(packed[i] & 0x3F);
        scales[i + 4] = float(packed[i + 4] & 0x3F);
        mins[i] = float((packed[i] >> 6) | ((packed[i + 8] & 0x03) << 2));
        mins[i + 4] = float((packed[i + 4] >> 6) | ((packed[i + 8] >> 2) & 0x0F));
    }
}

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint row = gl_WorkGroupID.x;

    if (row >= p.N) return;

    uint num_blocks_per_row = uint(p.num_blocks);
    float sum = 0.0;

    // Each thread processes part of the input
    for (uint blk = tid / 32; blk < num_blocks_per_row; blk += 8) {
        uint block_idx = row * num_blocks_per_row + blk;
        block_q4_k block = data_w[block_idx];

        float d = float(block.d);
        float dmin = float(block.dmin);

        // Process elements within this block
        uint lane = tid % 32;
        uint elem_start = lane * 8;

        if (elem_start < 256) {
            // Determine which sub-block we're in (8 sub-blocks of 32 elements each)
            uint sub_block = elem_start / 32;

            // Get scale for this sub-block (simplified)
            float scale = d * float(block.scales[sub_block % 12] & 0x3F);
            float min_val = dmin * float((block.scales[sub_block % 12] >> 6) + 1);

            for (uint i = 0; i < 8 && (elem_start + i) < 256; i++) {
                uint elem = elem_start + i;
                uint k_idx = blk * 256 + elem;

                if (k_idx < p.K) {
                    // Get quantized value (4-bit)
                    uint byte_idx = elem / 2;
                    uint nibble = (elem % 2 == 0) ?
                        (block.qs[byte_idx] & 0x0F) :
                        (block.qs[byte_idx] >> 4);

                    // Dequantize: w = scale * q - min
                    float w = scale * float(nibble) - min_val;

                    sum += data_x[k_idx] * w;
                }
            }
        }
    }

    // Parallel reduction in shared memory
    shared_sum[tid] = sum;
    barrier();

    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        barrier();
    }

    if (tid == 0) {
        data_y[row] = shared_sum[0];
    }
}
)";

        auto matvec_result = compile_shader("matvec_q4k_simple", matvec_q4k_shader, 3);
        if (!matvec_result.ok()) {
            GRANITE_LOG_WARN("Failed to compile matvec_q4k shader: {}", matvec_result.error().message());
        }

        // Softmax shader with numerically stable implementation
        const char* softmax_shader = R"(
#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint KX;        // Row size (number of elements per row)
    uint KY;        // Number of rows (for batched softmax)
    float scale;    // Pre-softmax scale (e.g., 1/sqrt(d_k) for attention)
    float padding;
} p;

layout(binding = 0) readonly buffer X { float data_x[]; };
layout(binding = 1) writeonly buffer D { float data_d[]; };

shared float shared_max;
shared float shared_sum;
shared float partial_max[256];
shared float partial_sum[256];

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint row = gl_WorkGroupID.x;
    uint n = p.KX;
    uint offset = row * n;

    // Step 1: Find max for numerical stability (parallel reduction)
    float local_max = -1e38;
    for (uint i = tid; i < n; i += 256) {
        float val = data_x[offset + i] * p.scale;
        local_max = max(local_max, val);
    }

    partial_max[tid] = local_max;
    barrier();

    // Reduce to find global max
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            partial_max[tid] = max(partial_max[tid], partial_max[tid + s]);
        }
        barrier();
    }

    if (tid == 0) {
        shared_max = partial_max[0];
    }
    barrier();

    float max_val = shared_max;

    // Step 2: Compute exp(x - max) and sum
    float local_sum = 0.0;
    for (uint i = tid; i < n; i += 256) {
        float val = data_x[offset + i] * p.scale;
        local_sum += exp(val - max_val);
    }

    partial_sum[tid] = local_sum;
    barrier();

    // Reduce to find global sum
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            partial_sum[tid] += partial_sum[tid + s];
        }
        barrier();
    }

    if (tid == 0) {
        shared_sum = partial_sum[0];
    }
    barrier();

    float sum_exp = shared_sum;
    float inv_sum = 1.0 / sum_exp;

    // Step 3: Write normalized values
    for (uint i = tid; i < n; i += 256) {
        float val = data_x[offset + i] * p.scale;
        data_d[offset + i] = exp(val - max_val) * inv_sum;
    }
}
)";

        auto softmax_result = compile_shader("softmax", softmax_shader, 2);
        if (!softmax_result.ok()) {
            GRANITE_LOG_WARN("Failed to compile softmax shader: {}", softmax_result.error().message());
        }

        GRANITE_LOG_INFO("Compiled {} builtin shaders", pipelines_.size());
#else
        GRANITE_LOG_INFO("Shaderc not available, using pre-compiled SPIR-V only");
#endif
    }

    // State
    bool initialized_ = false;
    bool profiling_enabled_ = false;
    ProfilingStats stats_;

    // Vulkan handles
    VkDevice device_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkQueue compute_queue_ = VK_NULL_HANDLE;
    uint32_t queue_family_ = 0;

    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;

    // Device properties
    VkPhysicalDeviceProperties device_props_{};
    uint32_t subgroup_size_ = 32;

    // Shader directory
    std::string shader_dir_;

    // Caches
    std::unordered_map<std::string, VkShaderModule> shader_modules_;
    std::unordered_map<std::string, Pipeline> pipelines_;
    std::unordered_map<uint32_t, VkDescriptorSetLayout> descriptor_set_layouts_;
};

} // namespace granite

#endif // GRANITE_HAS_VULKAN

// vulkan_operators.cpp - Vulkan operator implementations
//
// This file provides Vulkan compute implementations of operators.
// Shaders are adapted from llama.cpp's MIT-licensed Vulkan backend.

#include "granite/operators.h"
#include "granite/log.h"

#ifdef GRANITE_HAS_VULKAN

#include <vulkan/vulkan.h>

namespace granite {

// =============================================================================
// Vulkan Operator Base Class
// =============================================================================

class VulkanOperator : public IOperator {
protected:
    // Vulkan pipeline handle (set by derived classes)
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
};

// =============================================================================
// TODO: Vulkan Binary Operators (Add, Sub, Mul, Div)
// =============================================================================

// These will use shaders adapted from llama.cpp

// =============================================================================
// TODO: Vulkan Unary Operators (ReLU, GELU, SiLU)
// =============================================================================

// Shaders: silu.comp, gelu.comp from llama.cpp

// =============================================================================
// TODO: Vulkan MatMul Operators
// =============================================================================

// Shaders: mul_mat_vec_*.comp from llama.cpp for decode (matvec)
// Shaders: mul_mat_*.comp from llama.cpp for prefill (matmul)

// =============================================================================
// TODO: Vulkan Normalization Operators
// =============================================================================

// Shaders: rms_norm.comp, layer_norm.comp from llama.cpp

// =============================================================================
// TODO: Vulkan Attention Operators
// =============================================================================

// Shaders: flash_attn.comp, soft_max.comp from llama.cpp

// =============================================================================
// Register Vulkan Operators
// =============================================================================

void register_vulkan_operators() {
    // auto& registry = OperatorRegistry::instance();

    // TODO: Register operators as they are implemented
    // Example:
    // registry.register_op(OpType::Add, BackendType::Vulkan,
    //                     []() { return std::make_unique<VulkanAddOp>(); });

}

}  // namespace granite

#endif  // GRANITE_HAS_VULKAN

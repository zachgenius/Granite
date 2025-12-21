// vulkan_operators.cpp - Vulkan operator implementations
//
// This file provides Vulkan compute implementations of operators.
// Shaders are adapted from llama.cpp's MIT-licensed Vulkan backend.

#include "granite/operators.h"
#include "granite/log.h"
#include "vulkan_compute.h"

#ifdef GRANITE_HAS_VULKAN

#include <vulkan/vulkan.h>
#include <limits>

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
// Vulkan Binary Operators (Add, Mul)
// =============================================================================

template<OpType Op>
class VulkanBinaryOp : public VulkanOperator {
public:
    OpType type() const override { return Op; }

    Result<void> validate(const OpContext& ctx) const override {
        if (ctx.num_inputs() != 2) {
            return Error(ErrorCode::InvalidArgument, "Binary op requires 2 inputs");
        }
        if (ctx.inputs[0].dtype() != DataType::FP32 ||
            ctx.inputs[1].dtype() != DataType::FP32) {
            return Error(ErrorCode::NotImplemented, "Vulkan ops only support FP32");
        }
        if (ctx.inputs[0].shape() != ctx.inputs[1].shape()) {
            return Error(ErrorCode::InvalidShape, "Vulkan ops require matching shapes");
        }
        return {};
    }

    Result<std::vector<std::vector<int64_t>>> infer_shapes(const OpContext& ctx) const override {
        return std::vector<std::vector<int64_t>>{
            std::vector<int64_t>(ctx.inputs[0].shape().begin(), ctx.inputs[0].shape().end())
        };
    }

    Result<void> execute(OpContext& ctx) override {
        const auto& a = ctx.inputs[0];
        const auto& b = ctx.inputs[1];
        auto& out = ctx.outputs[0];

        size_t n = out.numel();
        if (n > std::numeric_limits<uint32_t>::max()) {
            return Error(ErrorCode::InvalidArgument, "Tensor too large for Vulkan op");
        }

        auto* compute = get_vulkan_compute();
        if (!compute || !compute->is_initialized()) {
            return Error(ErrorCode::BackendNotInitialized, "VulkanCompute not initialized");
        }

        auto* buf_a = static_cast<VkBuffer>(ctx.backend->get_native_buffer(a.buffer()));
        auto* buf_b = static_cast<VkBuffer>(ctx.backend->get_native_buffer(b.buffer()));
        auto* buf_out = static_cast<VkBuffer>(ctx.backend->get_native_buffer(out.buffer()));
        if (!buf_a || !buf_b || !buf_out) {
            return Error(ErrorCode::InvalidHandle, "Invalid Vulkan buffer handle");
        }

        if constexpr (Op == OpType::Add) {
            return compute->add(buf_a, buf_b, buf_out, static_cast<uint32_t>(n));
        }
        if constexpr (Op == OpType::Mul) {
            return compute->mul(buf_a, buf_b, buf_out, static_cast<uint32_t>(n));
        }

        return Error(ErrorCode::NotImplemented, "Unsupported Vulkan binary op");
    }
};

// =============================================================================
// Vulkan Unary Operators (SiLU, GELU)
// =============================================================================

template<OpType Op>
class VulkanUnaryOp : public VulkanOperator {
public:
    OpType type() const override { return Op; }

    Result<void> validate(const OpContext& ctx) const override {
        if (ctx.num_inputs() != 1) {
            return Error(ErrorCode::InvalidArgument, "Unary op requires 1 input");
        }
        if (ctx.inputs[0].dtype() != DataType::FP32) {
            return Error(ErrorCode::NotImplemented, "Vulkan ops only support FP32");
        }
        return {};
    }

    Result<std::vector<std::vector<int64_t>>> infer_shapes(const OpContext& ctx) const override {
        return std::vector<std::vector<int64_t>>{
            std::vector<int64_t>(ctx.inputs[0].shape().begin(), ctx.inputs[0].shape().end())
        };
    }

    Result<void> execute(OpContext& ctx) override {
        const auto& x = ctx.inputs[0];
        auto& out = ctx.outputs[0];

        size_t n = out.numel();
        if (n > std::numeric_limits<uint32_t>::max()) {
            return Error(ErrorCode::InvalidArgument, "Tensor too large for Vulkan op");
        }

        auto* compute = get_vulkan_compute();
        if (!compute || !compute->is_initialized()) {
            return Error(ErrorCode::BackendNotInitialized, "VulkanCompute not initialized");
        }

        auto* buf_x = static_cast<VkBuffer>(ctx.backend->get_native_buffer(x.buffer()));
        auto* buf_out = static_cast<VkBuffer>(ctx.backend->get_native_buffer(out.buffer()));
        if (!buf_x || !buf_out) {
            return Error(ErrorCode::InvalidHandle, "Invalid Vulkan buffer handle");
        }

        if constexpr (Op == OpType::SiLU) {
            return compute->silu(buf_x, buf_out, static_cast<uint32_t>(n));
        }
        if constexpr (Op == OpType::GELU) {
            return compute->gelu(buf_x, buf_out, static_cast<uint32_t>(n));
        }

        return Error(ErrorCode::NotImplemented, "Unsupported Vulkan unary op");
    }
};

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
    auto& registry = OperatorRegistry::instance();

    registry.register_op(OpType::Add, BackendType::Vulkan,
                        []() { return std::make_unique<VulkanBinaryOp<OpType::Add>>(); });
    registry.register_op(OpType::Mul, BackendType::Vulkan,
                        []() { return std::make_unique<VulkanBinaryOp<OpType::Mul>>(); });
    registry.register_op(OpType::SiLU, BackendType::Vulkan,
                        []() { return std::make_unique<VulkanUnaryOp<OpType::SiLU>>(); });
    registry.register_op(OpType::GELU, BackendType::Vulkan,
                        []() { return std::make_unique<VulkanUnaryOp<OpType::GELU>>(); });

}

}  // namespace granite

#endif  // GRANITE_HAS_VULKAN

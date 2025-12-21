// vulkan_operators.cpp - Vulkan operator implementations
//
// This file provides Vulkan compute implementations of operators.
// Shaders are adapted from llama.cpp's MIT-licensed Vulkan backend.

#include "granite/operators.h"
#include "granite/log.h"
#include "backend/vulkan/vulkan_compute.h"

#ifdef GRANITE_HAS_VULKAN

#include <vulkan/vulkan.h>
#include <limits>
#include <algorithm>

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
        auto shape_a = ctx.inputs[0].shape();
        auto shape_b = ctx.inputs[1].shape();
        if (shape_a.size() != shape_b.size() ||
            !std::equal(shape_a.begin(), shape_a.end(), shape_b.begin())) {
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

        auto* compute = get_vulkan_compute(ctx.backend);
        if (!compute || !compute->is_initialized()) {
            return Error(ErrorCode::BackendNotInitialized, "VulkanCompute not initialized");
        }

        auto* buf_a = static_cast<VkBuffer>(ctx.backend->get_native_buffer(a.buffer()));
        auto* buf_b = static_cast<VkBuffer>(ctx.backend->get_native_buffer(b.buffer()));
        auto* buf_out = static_cast<VkBuffer>(ctx.backend->get_native_buffer(out.buffer()));
        if (!buf_a || !buf_b || !buf_out) {
            return Error(ErrorCode::InvalidArgument, "Invalid Vulkan buffer handle");
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

        auto* compute = get_vulkan_compute(ctx.backend);
        if (!compute || !compute->is_initialized()) {
            return Error(ErrorCode::BackendNotInitialized, "VulkanCompute not initialized");
        }

        auto* buf_x = static_cast<VkBuffer>(ctx.backend->get_native_buffer(x.buffer()));
        auto* buf_out = static_cast<VkBuffer>(ctx.backend->get_native_buffer(out.buffer()));
        if (!buf_x || !buf_out) {
            return Error(ErrorCode::InvalidArgument, "Invalid Vulkan buffer handle");
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

class VulkanMatMulOp : public VulkanOperator {
public:
    OpType type() const override { return OpType::MatMul; }

    Result<void> validate(const OpContext& ctx) const override {
        if (ctx.num_inputs() != 2) {
            return Error(ErrorCode::InvalidArgument, "MatMul requires 2 inputs");
        }

        const auto& a = ctx.inputs[0];
        const auto& b = ctx.inputs[1];

        if (a.ndim() != 2 || b.ndim() != 2) {
            return Error(ErrorCode::NotImplemented, "Vulkan MatMul supports 2D tensors only");
        }
        if (a.dtype() != DataType::FP32 || b.dtype() != DataType::FP32) {
            return Error(ErrorCode::NotImplemented, "Vulkan MatMul only supports FP32");
        }

        int64_t k_a = a.size(1);
        int64_t k_b = b.size(0);
        if (k_a != k_b) {
            return Error(ErrorCode::ShapeMismatch,
                         fmt::format("MatMul inner dimensions mismatch: {} vs {}", k_a, k_b));
        }

        return {};
    }

    Result<std::vector<std::vector<int64_t>>> infer_shapes(const OpContext& ctx) const override {
        const auto& a = ctx.inputs[0];
        const auto& b = ctx.inputs[1];
        return std::vector<std::vector<int64_t>>{{a.size(0), b.size(1)}};
    }

    Result<void> execute(OpContext& ctx) override {
        const auto& a = ctx.inputs[0];
        const auto& b = ctx.inputs[1];
        auto& out = ctx.outputs[0];

        auto* compute = get_vulkan_compute(ctx.backend);
        if (!compute || !compute->is_initialized()) {
            return Error(ErrorCode::BackendNotInitialized, "VulkanCompute not initialized");
        }

        auto* buf_a = static_cast<VkBuffer>(ctx.backend->get_native_buffer(a.buffer()));
        auto* buf_b = static_cast<VkBuffer>(ctx.backend->get_native_buffer(b.buffer()));
        auto* buf_out = static_cast<VkBuffer>(ctx.backend->get_native_buffer(out.buffer()));
        if (!buf_a || !buf_b || !buf_out) {
            return Error(ErrorCode::InvalidArgument, "Invalid Vulkan buffer handle");
        }

        uint32_t M = static_cast<uint32_t>(a.size(0));
        uint32_t K = static_cast<uint32_t>(a.size(1));
        uint32_t N = static_cast<uint32_t>(b.size(1));

        return compute->matmul_f32(buf_a, buf_b, buf_out, M, K, N);
    }
};

// =============================================================================
// TODO: Vulkan Normalization Operators
// =============================================================================

// Shaders: rms_norm.comp, layer_norm.comp from llama.cpp

class VulkanRMSNormOp : public VulkanOperator {
public:
    OpType type() const override { return OpType::RMSNorm; }

    Result<void> validate(const OpContext& ctx) const override {
        if (ctx.num_inputs() != 2) {
            return Error(ErrorCode::InvalidArgument, "RMSNorm requires 2 inputs");
        }
        if (ctx.inputs[0].dtype() != DataType::FP32 ||
            ctx.inputs[1].dtype() != DataType::FP32) {
            return Error(ErrorCode::NotImplemented, "Vulkan RMSNorm only supports FP32");
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
        const auto& w = ctx.inputs[1];
        auto& out = ctx.outputs[0];

        if (x.ndim() != 1) {
            return Error(ErrorCode::NotImplemented, "Vulkan RMSNorm supports 1D tensors only");
        }
        if (w.ndim() != 1 || w.numel() != x.numel()) {
            return Error(ErrorCode::InvalidShape, "RMSNorm weight shape mismatch");
        }

        auto* compute = get_vulkan_compute(ctx.backend);
        if (!compute || !compute->is_initialized()) {
            return Error(ErrorCode::BackendNotInitialized, "VulkanCompute not initialized");
        }

        auto* buf_x = static_cast<VkBuffer>(ctx.backend->get_native_buffer(x.buffer()));
        auto* buf_w = static_cast<VkBuffer>(ctx.backend->get_native_buffer(w.buffer()));
        auto* buf_out = static_cast<VkBuffer>(ctx.backend->get_native_buffer(out.buffer()));
        if (!buf_x || !buf_w || !buf_out) {
            return Error(ErrorCode::InvalidArgument, "Invalid Vulkan buffer handle");
        }

        double eps = ctx.attrs.get<double>("eps", 1e-5);
        return compute->rms_norm(buf_x, buf_w, buf_out,
                                 static_cast<uint32_t>(x.numel()),
                                 static_cast<float>(eps));
    }
};

// =============================================================================
// TODO: Vulkan Attention Operators
// =============================================================================

// Shaders: flash_attn.comp, soft_max.comp from llama.cpp

class VulkanSoftmaxOp : public VulkanOperator {
public:
    OpType type() const override { return OpType::Softmax; }

    Result<void> validate(const OpContext& ctx) const override {
        if (ctx.num_inputs() != 1) {
            return Error(ErrorCode::InvalidArgument, "Softmax requires 1 input");
        }
        if (ctx.inputs[0].dtype() != DataType::FP32) {
            return Error(ErrorCode::NotImplemented, "Vulkan Softmax only supports FP32");
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

        int axis = static_cast<int>(ctx.attrs.get<int64_t>("axis", -1));
        if (axis < 0) {
            axis += static_cast<int>(x.ndim());
        }
        if (axis != static_cast<int>(x.ndim()) - 1) {
            return Error(ErrorCode::NotImplemented, "Vulkan Softmax supports last-axis only");
        }

        size_t rows = 1;
        for (int i = 0; i < axis; i++) {
            rows *= static_cast<size_t>(x.size(i));
        }
        size_t cols = static_cast<size_t>(x.size(axis));

        if (rows > std::numeric_limits<uint32_t>::max() ||
            cols > std::numeric_limits<uint32_t>::max()) {
            return Error(ErrorCode::InvalidArgument, "Softmax size exceeds Vulkan limits");
        }

        auto* compute = get_vulkan_compute(ctx.backend);
        if (!compute || !compute->is_initialized()) {
            return Error(ErrorCode::BackendNotInitialized, "VulkanCompute not initialized");
        }

        auto* buf_x = static_cast<VkBuffer>(ctx.backend->get_native_buffer(x.buffer()));
        auto* buf_out = static_cast<VkBuffer>(ctx.backend->get_native_buffer(out.buffer()));
        if (!buf_x || !buf_out) {
            return Error(ErrorCode::InvalidArgument, "Invalid Vulkan buffer handle");
        }

        return compute->softmax_rows(
            buf_x,
            buf_out,
            static_cast<uint32_t>(rows),
            static_cast<uint32_t>(cols),
            1.0f);
    }
};

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
    registry.register_op(OpType::RMSNorm, BackendType::Vulkan,
                        []() { return std::make_unique<VulkanRMSNormOp>(); });
    registry.register_op(OpType::Softmax, BackendType::Vulkan,
                        []() { return std::make_unique<VulkanSoftmaxOp>(); });
    registry.register_op(OpType::MatMul, BackendType::Vulkan,
                        []() { return std::make_unique<VulkanMatMulOp>(); });

}

}  // namespace granite

#endif  // GRANITE_HAS_VULKAN

#include "granite/operators.h"
#include "granite/log.h"

#include <cmath>
#include <algorithm>

#ifdef GRANITE_HAS_CPU

namespace granite {

// =============================================================================
// CPU Binary Operators (Add, Sub, Mul, Div)
// =============================================================================

template<OpType Op>
class CPUBinaryOp : public IOperator {
public:
    OpType type() const override { return Op; }

    Result<void> validate(const OpContext& ctx) const override {
        if (ctx.num_inputs() != 2) {
            return Error(ErrorCode::InvalidArgument, "Binary op requires 2 inputs");
        }
        if (ctx.inputs[0].dtype() != ctx.inputs[1].dtype()) {
            return Error(ErrorCode::DTypeMismatch, "Input dtypes must match");
        }
        return {};
    }

    Result<std::vector<std::vector<int64_t>>> infer_shapes(const OpContext& ctx) const override {
        auto shape = broadcast_shapes(ctx.inputs[0].shape(), ctx.inputs[1].shape());
        return std::vector<std::vector<int64_t>>{shape};
    }

    Result<void> execute(OpContext& ctx) override {
        const auto& a = ctx.inputs[0];
        const auto& b = ctx.inputs[1];
        auto& out = ctx.outputs[0];

        if (a.dtype() != DataType::FP32) {
            return Error(ErrorCode::NotImplemented, "CPU ops only support FP32");
        }

        auto map_a = ctx.backend->map_buffer(a.buffer());
        auto map_b = ctx.backend->map_buffer(b.buffer());
        auto map_out = ctx.backend->map_buffer(out.buffer());

        if (!map_a.ok() || !map_b.ok() || !map_out.ok()) {
            return Error(ErrorCode::InternalError, "Failed to map buffers");
        }

        const float* pa = static_cast<const float*>(map_a.value());
        const float* pb = static_cast<const float*>(map_b.value());
        float* pout = static_cast<float*>(map_out.value());

        size_t n = out.numel();

        // Simple case: same shape, no broadcasting
        if (a.numel() == b.numel() && a.numel() == n) {
            for (size_t i = 0; i < n; i++) {
                if constexpr (Op == OpType::Add) {
                    pout[i] = pa[i] + pb[i];
                } else if constexpr (Op == OpType::Sub) {
                    pout[i] = pa[i] - pb[i];
                } else if constexpr (Op == OpType::Mul) {
                    pout[i] = pa[i] * pb[i];
                } else if constexpr (Op == OpType::Div) {
                    pout[i] = pa[i] / pb[i];
                }
            }
        } else {
            return Error(ErrorCode::NotImplemented, "Broadcasting not yet implemented for CPU");
        }

        ctx.backend->unmap_buffer(a.buffer());
        ctx.backend->unmap_buffer(b.buffer());
        ctx.backend->unmap_buffer(out.buffer());

        return {};
    }
};

// =============================================================================
// CPU Unary Operators (ReLU, GELU, SiLU)
// =============================================================================

template<OpType Op>
class CPUUnaryOp : public IOperator {
public:
    OpType type() const override { return Op; }

    Result<void> validate(const OpContext& ctx) const override {
        if (ctx.num_inputs() != 1) {
            return Error(ErrorCode::InvalidArgument, "Unary op requires 1 input");
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

        if (x.dtype() != DataType::FP32) {
            return Error(ErrorCode::NotImplemented, "CPU ops only support FP32");
        }

        auto map_x = ctx.backend->map_buffer(x.buffer());
        auto map_out = ctx.backend->map_buffer(out.buffer());

        if (!map_x.ok() || !map_out.ok()) {
            return Error(ErrorCode::InternalError, "Failed to map buffers");
        }

        const float* px = static_cast<const float*>(map_x.value());
        float* pout = static_cast<float*>(map_out.value());

        size_t n = x.numel();

        for (size_t i = 0; i < n; i++) {
            float val = px[i];
            if constexpr (Op == OpType::ReLU) {
                pout[i] = std::max(val, 0.0f);
            } else if constexpr (Op == OpType::GELU) {
                // GELU approximation
                float x3 = val * val * val;
                float inner = 0.7978845608f * (val + 0.044715f * x3);
                pout[i] = 0.5f * val * (1.0f + std::tanh(inner));
            } else if constexpr (Op == OpType::SiLU) {
                pout[i] = val / (1.0f + std::exp(-val));
            } else if constexpr (Op == OpType::Sigmoid) {
                pout[i] = 1.0f / (1.0f + std::exp(-val));
            } else if constexpr (Op == OpType::Tanh) {
                pout[i] = std::tanh(val);
            }
        }

        ctx.backend->unmap_buffer(x.buffer());
        ctx.backend->unmap_buffer(out.buffer());

        return {};
    }
};

// =============================================================================
// CPU Softmax
// =============================================================================

class CPUSoftmaxOp : public IOperator {
public:
    OpType type() const override { return OpType::Softmax; }

    Result<void> validate(const OpContext& ctx) const override {
        if (ctx.num_inputs() != 1) {
            return Error(ErrorCode::InvalidArgument, "Softmax requires 1 input");
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
        if (axis < 0) axis += static_cast<int>(x.ndim());

        auto map_x = ctx.backend->map_buffer(x.buffer());
        auto map_out = ctx.backend->map_buffer(out.buffer());

        if (!map_x.ok() || !map_out.ok()) {
            return Error(ErrorCode::InternalError, "Failed to map buffers");
        }

        const float* px = static_cast<const float*>(map_x.value());
        float* pout = static_cast<float*>(map_out.value());

        // For simplicity, assume softmax over last dimension
        size_t outer = 1;
        for (int i = 0; i < axis; i++) {
            outer *= x.size(i);
        }
        size_t inner = x.size(axis);

        for (size_t row = 0; row < outer; row++) {
            const float* row_in = px + row * inner;
            float* row_out = pout + row * inner;

            // Find max for numerical stability
            float max_val = row_in[0];
            for (size_t i = 1; i < inner; i++) {
                max_val = std::max(max_val, row_in[i]);
            }

            // Compute exp and sum
            float sum = 0.0f;
            for (size_t i = 0; i < inner; i++) {
                row_out[i] = std::exp(row_in[i] - max_val);
                sum += row_out[i];
            }

            // Normalize
            float inv_sum = 1.0f / sum;
            for (size_t i = 0; i < inner; i++) {
                row_out[i] *= inv_sum;
            }
        }

        ctx.backend->unmap_buffer(x.buffer());
        ctx.backend->unmap_buffer(out.buffer());

        return {};
    }
};

// =============================================================================
// CPU MatMul
// =============================================================================

class CPUMatMulOp : public IOperator {
public:
    OpType type() const override { return OpType::MatMul; }

    Result<void> validate(const OpContext& ctx) const override {
        if (ctx.num_inputs() != 2) {
            return Error(ErrorCode::InvalidArgument, "MatMul requires 2 inputs");
        }
        const auto& a = ctx.inputs[0];
        const auto& b = ctx.inputs[1];

        if (a.ndim() < 2 || b.ndim() < 2) {
            return Error(ErrorCode::InvalidShape, "MatMul requires at least 2D tensors");
        }

        int64_t k_a = a.size(a.ndim() - 1);
        int64_t k_b = b.size(b.ndim() - 2);
        if (k_a != k_b) {
            return Error(ErrorCode::ShapeMismatch,
                         fmt::format("MatMul inner dimensions mismatch: {} vs {}", k_a, k_b));
        }

        return {};
    }

    Result<std::vector<std::vector<int64_t>>> infer_shapes(const OpContext& ctx) const override {
        const auto& a = ctx.inputs[0];
        const auto& b = ctx.inputs[1];

        int64_t m = a.size(a.ndim() - 2);
        int64_t n = b.size(b.ndim() - 1);

        return std::vector<std::vector<int64_t>>{{m, n}};
    }

    Result<void> execute(OpContext& ctx) override {
        const auto& a = ctx.inputs[0];
        const auto& b = ctx.inputs[1];
        auto& out = ctx.outputs[0];

        auto map_a = ctx.backend->map_buffer(a.buffer());
        auto map_b = ctx.backend->map_buffer(b.buffer());
        auto map_out = ctx.backend->map_buffer(out.buffer());

        if (!map_a.ok() || !map_b.ok() || !map_out.ok()) {
            return Error(ErrorCode::InternalError, "Failed to map buffers");
        }

        const float* pa = static_cast<const float*>(map_a.value());
        const float* pb = static_cast<const float*>(map_b.value());
        float* pout = static_cast<float*>(map_out.value());

        size_t M = a.size(0);
        size_t K = a.size(1);
        size_t N = b.size(1);

        // Naive matrix multiplication
        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                float sum = 0.0f;
                for (size_t k = 0; k < K; k++) {
                    sum += pa[i * K + k] * pb[k * N + j];
                }
                pout[i * N + j] = sum;
            }
        }

        ctx.backend->unmap_buffer(a.buffer());
        ctx.backend->unmap_buffer(b.buffer());
        ctx.backend->unmap_buffer(out.buffer());

        return {};
    }
};

// =============================================================================
// CPU LayerNorm
// =============================================================================

class CPULayerNormOp : public IOperator {
public:
    OpType type() const override { return OpType::LayerNorm; }

    Result<void> validate(const OpContext& ctx) const override {
        if (ctx.num_inputs() != 3) {
            return Error(ErrorCode::InvalidArgument, "LayerNorm requires 3 inputs (x, weight, bias)");
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
        const auto& weight = ctx.inputs[1];
        const auto& bias = ctx.inputs[2];
        auto& out = ctx.outputs[0];

        float eps = static_cast<float>(ctx.attrs.get<double>("eps", 1e-5));

        auto map_x = ctx.backend->map_buffer(x.buffer());
        auto map_w = ctx.backend->map_buffer(weight.buffer());
        auto map_b = ctx.backend->map_buffer(bias.buffer());
        auto map_out = ctx.backend->map_buffer(out.buffer());

        if (!map_x.ok() || !map_w.ok() || !map_b.ok() || !map_out.ok()) {
            return Error(ErrorCode::InternalError, "Failed to map buffers");
        }

        const float* px = static_cast<const float*>(map_x.value());
        const float* pw = static_cast<const float*>(map_w.value());
        const float* pb = static_cast<const float*>(map_b.value());
        float* pout = static_cast<float*>(map_out.value());

        size_t norm_size = x.size(x.ndim() - 1);
        size_t num_rows = x.numel() / norm_size;

        for (size_t row = 0; row < num_rows; row++) {
            const float* row_in = px + row * norm_size;
            float* row_out = pout + row * norm_size;

            // Compute mean
            float sum = 0.0f;
            for (size_t i = 0; i < norm_size; i++) {
                sum += row_in[i];
            }
            float mean = sum / static_cast<float>(norm_size);

            // Compute variance
            float var_sum = 0.0f;
            for (size_t i = 0; i < norm_size; i++) {
                float diff = row_in[i] - mean;
                var_sum += diff * diff;
            }
            float var = var_sum / static_cast<float>(norm_size);
            float inv_std = 1.0f / std::sqrt(var + eps);

            // Normalize and apply affine
            for (size_t i = 0; i < norm_size; i++) {
                float normalized = (row_in[i] - mean) * inv_std;
                row_out[i] = normalized * pw[i] + pb[i];
            }
        }

        ctx.backend->unmap_buffer(x.buffer());
        ctx.backend->unmap_buffer(weight.buffer());
        ctx.backend->unmap_buffer(bias.buffer());
        ctx.backend->unmap_buffer(out.buffer());

        return {};
    }
};

// =============================================================================
// CPU RMSNorm
// =============================================================================

class CPURMSNormOp : public IOperator {
public:
    OpType type() const override { return OpType::RMSNorm; }

    Result<void> validate(const OpContext& ctx) const override {
        if (ctx.num_inputs() != 2) {
            return Error(ErrorCode::InvalidArgument, "RMSNorm requires 2 inputs (x, weight)");
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
        const auto& weight = ctx.inputs[1];
        auto& out = ctx.outputs[0];

        float eps = static_cast<float>(ctx.attrs.get<double>("eps", 1e-5));

        auto map_x = ctx.backend->map_buffer(x.buffer());
        auto map_w = ctx.backend->map_buffer(weight.buffer());
        auto map_out = ctx.backend->map_buffer(out.buffer());

        if (!map_x.ok() || !map_w.ok() || !map_out.ok()) {
            return Error(ErrorCode::InternalError, "Failed to map buffers");
        }

        const float* px = static_cast<const float*>(map_x.value());
        const float* pw = static_cast<const float*>(map_w.value());
        float* pout = static_cast<float*>(map_out.value());

        size_t norm_size = x.size(x.ndim() - 1);
        size_t num_rows = x.numel() / norm_size;

        for (size_t row = 0; row < num_rows; row++) {
            const float* row_in = px + row * norm_size;
            float* row_out = pout + row * norm_size;

            // Compute sum of squares
            float sum_sq = 0.0f;
            for (size_t i = 0; i < norm_size; i++) {
                sum_sq += row_in[i] * row_in[i];
            }

            float rms = std::sqrt(sum_sq / static_cast<float>(norm_size) + eps);
            float inv_rms = 1.0f / rms;

            // Normalize and apply weight
            for (size_t i = 0; i < norm_size; i++) {
                row_out[i] = row_in[i] * inv_rms * pw[i];
            }
        }

        ctx.backend->unmap_buffer(x.buffer());
        ctx.backend->unmap_buffer(weight.buffer());
        ctx.backend->unmap_buffer(out.buffer());

        return {};
    }
};

// =============================================================================
// Register CPU Operators
// =============================================================================

void register_cpu_operators() {
    auto& registry = OperatorRegistry::instance();

    registry.register_op(OpType::Add, BackendType::CPU,
                        []() { return std::make_unique<CPUBinaryOp<OpType::Add>>(); });
    registry.register_op(OpType::Sub, BackendType::CPU,
                        []() { return std::make_unique<CPUBinaryOp<OpType::Sub>>(); });
    registry.register_op(OpType::Mul, BackendType::CPU,
                        []() { return std::make_unique<CPUBinaryOp<OpType::Mul>>(); });
    registry.register_op(OpType::Div, BackendType::CPU,
                        []() { return std::make_unique<CPUBinaryOp<OpType::Div>>(); });
    registry.register_op(OpType::ReLU, BackendType::CPU,
                        []() { return std::make_unique<CPUUnaryOp<OpType::ReLU>>(); });
    registry.register_op(OpType::GELU, BackendType::CPU,
                        []() { return std::make_unique<CPUUnaryOp<OpType::GELU>>(); });
    registry.register_op(OpType::SiLU, BackendType::CPU,
                        []() { return std::make_unique<CPUUnaryOp<OpType::SiLU>>(); });
    registry.register_op(OpType::Sigmoid, BackendType::CPU,
                        []() { return std::make_unique<CPUUnaryOp<OpType::Sigmoid>>(); });
    registry.register_op(OpType::Tanh, BackendType::CPU,
                        []() { return std::make_unique<CPUUnaryOp<OpType::Tanh>>(); });
    registry.register_op(OpType::Softmax, BackendType::CPU,
                        []() { return std::make_unique<CPUSoftmaxOp>(); });
    registry.register_op(OpType::MatMul, BackendType::CPU,
                        []() { return std::make_unique<CPUMatMulOp>(); });
    registry.register_op(OpType::LayerNorm, BackendType::CPU,
                        []() { return std::make_unique<CPULayerNormOp>(); });
    registry.register_op(OpType::RMSNorm, BackendType::CPU,
                        []() { return std::make_unique<CPURMSNormOp>(); });

    GRANITE_LOG_DEBUG("Registered CPU operators");
}

}  // namespace granite

#endif  // GRANITE_HAS_CPU

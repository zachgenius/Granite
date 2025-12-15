#include "granite/operators.h"
#include "granite/gguf.h"
#include "granite/log.h"

#include <cmath>
#include <algorithm>
#include <cstring>

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
// FP16 Conversion Helpers
// =============================================================================

namespace {

inline float fp16_to_fp32_cpu(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    if (exp == 0) {
        if (mant == 0) {
            uint32_t bits = sign << 31;
            float result;
            std::memcpy(&result, &bits, 4);
            return result;
        }
        while ((mant & 0x400) == 0) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= 0x3FF;
    } else if (exp == 31) {
        uint32_t bits = (sign << 31) | 0x7F800000 | (mant << 13);
        float result;
        std::memcpy(&result, &bits, 4);
        return result;
    }

    uint32_t bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    float result;
    std::memcpy(&result, &bits, 4);
    return result;
}

inline uint16_t fp32_to_fp16_cpu(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, 4);

    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3FF;

    if (exp <= 0) {
        return static_cast<uint16_t>(sign);
    } else if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00);
    }
    return static_cast<uint16_t>(sign | (exp << 10) | mant);
}

}  // anonymous namespace

// =============================================================================
// CPU Quantized MatMul
// =============================================================================

class CPUQuantizedMatMulOp : public IOperator {
public:
    OpType type() const override { return OpType::QuantizedMatMul; }

    Result<void> validate(const OpContext& ctx) const override {
        // Inputs: [0] activation (FP16/FP32), [1] quantized weights (raw bytes)
        // Attributes: quant_type, weight_shape
        if (ctx.num_inputs() < 2) {
            return Error(ErrorCode::InvalidArgument, "QuantizedMatMul requires at least 2 inputs");
        }
        return {};
    }

    Result<std::vector<std::vector<int64_t>>> infer_shapes(const OpContext& ctx) const override {
        const auto& a = ctx.inputs[0];

        // Get weight dimensions from attributes
        auto weight_shape = ctx.attrs.get<std::vector<int64_t>>("weight_shape");
        if (weight_shape.size() < 2) {
            return Error(ErrorCode::InvalidArgument, "weight_shape attribute required");
        }

        int64_t m = a.size(a.ndim() - 2);
        int64_t n = weight_shape[weight_shape.size() - 1];

        return std::vector<std::vector<int64_t>>{{m, n}};
    }

    Result<void> execute(OpContext& ctx) override {
        const auto& activation = ctx.inputs[0];  // FP16 or FP32
        const auto& weights = ctx.inputs[1];      // Quantized weights
        auto& out = ctx.outputs[0];

        // Get quantization type
        auto quant_type_val = ctx.attrs.get<int64_t>("quant_type", static_cast<int64_t>(GGMLType::Q8_0));
        auto quant_type = static_cast<GGMLType>(quant_type_val);

        // Get weight shape
        auto weight_shape = ctx.attrs.get<std::vector<int64_t>>("weight_shape");
        if (weight_shape.size() < 2) {
            return Error(ErrorCode::InvalidArgument, "weight_shape required for QuantizedMatMul");
        }

        // Map buffers
        auto map_act = ctx.backend->map_buffer(activation.buffer());
        auto map_w = ctx.backend->map_buffer(weights.buffer());
        auto map_out = ctx.backend->map_buffer(out.buffer());

        if (!map_act.ok() || !map_w.ok() || !map_out.ok()) {
            return Error(ErrorCode::InternalError, "Failed to map buffers");
        }

        // Dimensions: A[M, K] @ W[K, N] = Out[M, N]
        size_t M = activation.size(activation.ndim() - 2);
        size_t K = activation.size(activation.ndim() - 1);
        size_t N = static_cast<size_t>(weight_shape[weight_shape.size() - 1]);

        const auto* w_data = static_cast<const uint8_t*>(map_w.value());

        // Dispatch based on activation type
        if (activation.dtype() == DataType::FP32) {
            const float* a_ptr = static_cast<const float*>(map_act.value());
            float* out_ptr = static_cast<float*>(map_out.value());

            auto result = execute_fp32(a_ptr, w_data, out_ptr, M, K, N, quant_type);
            if (!result.ok()) return result;
        } else if (activation.dtype() == DataType::FP16) {
            const uint16_t* a_ptr = static_cast<const uint16_t*>(map_act.value());
            uint16_t* out_ptr = static_cast<uint16_t*>(map_out.value());

            auto result = execute_fp16(a_ptr, w_data, out_ptr, M, K, N, quant_type);
            if (!result.ok()) return result;
        } else {
            return Error(ErrorCode::NotImplemented, "QuantizedMatMul only supports FP16/FP32 activations");
        }

        ctx.backend->unmap_buffer(activation.buffer());
        ctx.backend->unmap_buffer(weights.buffer());
        ctx.backend->unmap_buffer(out.buffer());

        return {};
    }

private:
    Result<void> execute_fp32(const float* A, const uint8_t* W,
                              float* out, size_t M, size_t K, size_t N,
                              GGMLType quant_type) {
        switch (quant_type) {
            case GGMLType::Q8_0:
                matmul_q8_0_fp32(A, W, out, M, K, N);
                break;
            case GGMLType::Q4_0:
                matmul_q4_0_fp32(A, W, out, M, K, N);
                break;
            case GGMLType::Q4_K:
                matmul_q4_k_fp32(A, W, out, M, K, N);
                break;
            default:
                return Error(ErrorCode::NotImplemented,
                            fmt::format("QuantizedMatMul not implemented for {}",
                                        ggml_type_name(quant_type)));
        }
        return {};
    }

    Result<void> execute_fp16(const uint16_t* A, const uint8_t* W,
                              uint16_t* out, size_t M, size_t K, size_t N,
                              GGMLType quant_type) {
        // Convert FP16 to FP32, compute, convert back
        std::vector<float> a_fp32(M * K);
        std::vector<float> out_fp32(M * N, 0.0f);

        for (size_t i = 0; i < M * K; i++) {
            a_fp32[i] = fp16_to_fp32_cpu(A[i]);
        }

        auto result = execute_fp32(a_fp32.data(), W, out_fp32.data(), M, K, N, quant_type);
        if (!result.ok()) return result;

        for (size_t i = 0; i < M * N; i++) {
            out[i] = fp32_to_fp16_cpu(out_fp32[i]);
        }

        return {};
    }

    // Q8_0 MatMul: weights are 32-element blocks with FP16 scale + 32 int8 values
    void matmul_q8_0_fp32(const float* A, const uint8_t* W,
                         float* out, size_t M, size_t K, size_t N) {
        constexpr size_t BLOCK_SIZE = 32;
        constexpr size_t BYTES_PER_BLOCK = 34;  // 2 (scale) + 32 (data)
        size_t num_blocks_per_row = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // For each output element
        for (size_t m = 0; m < M; m++) {
            for (size_t n = 0; n < N; n++) {
                float sum = 0.0f;

                // Process blocks
                const uint8_t* w_row = W + n * num_blocks_per_row * BYTES_PER_BLOCK;

                for (size_t blk = 0; blk < num_blocks_per_row; blk++) {
                    const uint8_t* block = w_row + blk * BYTES_PER_BLOCK;

                    // Read scale
                    uint16_t scale_bits;
                    std::memcpy(&scale_bits, block, 2);
                    float scale = fp16_to_fp32_cpu(scale_bits);

                    // Dot product with dequantized values
                    const int8_t* qs = reinterpret_cast<const int8_t*>(block + 2);
                    size_t k_start = blk * BLOCK_SIZE;

                    for (size_t i = 0; i < BLOCK_SIZE && (k_start + i) < K; i++) {
                        float w_val = static_cast<float>(qs[i]) * scale;
                        sum += A[m * K + k_start + i] * w_val;
                    }
                }

                out[m * N + n] = sum;
            }
        }
    }

    // Q4_0 MatMul: weights are 32-element blocks with FP16 scale + 16 bytes (4-bit values)
    void matmul_q4_0_fp32(const float* A, const uint8_t* W,
                         float* out, size_t M, size_t K, size_t N) {
        constexpr size_t BLOCK_SIZE = 32;
        constexpr size_t BYTES_PER_BLOCK = 18;  // 2 (scale) + 16 (data)
        size_t num_blocks_per_row = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

        for (size_t m = 0; m < M; m++) {
            for (size_t n = 0; n < N; n++) {
                float sum = 0.0f;

                const uint8_t* w_row = W + n * num_blocks_per_row * BYTES_PER_BLOCK;

                for (size_t blk = 0; blk < num_blocks_per_row; blk++) {
                    const uint8_t* block = w_row + blk * BYTES_PER_BLOCK;

                    uint16_t scale_bits;
                    std::memcpy(&scale_bits, block, 2);
                    float scale = fp16_to_fp32_cpu(scale_bits);

                    const uint8_t* qs = block + 2;
                    size_t k_start = blk * BLOCK_SIZE;

                    for (size_t i = 0; i < 16 && (k_start + i * 2) < K; i++) {
                        uint8_t byte = qs[i];
                        int8_t q0 = (byte & 0xF) - 8;
                        int8_t q1 = (byte >> 4) - 8;

                        float w0 = static_cast<float>(q0) * scale;
                        float w1 = static_cast<float>(q1) * scale;

                        sum += A[m * K + k_start + i * 2] * w0;
                        if (k_start + i * 2 + 1 < K) {
                            sum += A[m * K + k_start + i * 2 + 1] * w1;
                        }
                    }
                }

                out[m * N + n] = sum;
            }
        }
    }

    // Q4_K MatMul: 256-element super-blocks
    void matmul_q4_k_fp32(const float* A, const uint8_t* W,
                         float* out, size_t M, size_t K, size_t N) {
        constexpr size_t SUPER_BLOCK_SIZE = 256;
        constexpr size_t BYTES_PER_BLOCK = 144;
        size_t num_blocks_per_row = (K + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;

        for (size_t m = 0; m < M; m++) {
            for (size_t n = 0; n < N; n++) {
                float sum = 0.0f;

                const uint8_t* w_row = W + n * num_blocks_per_row * BYTES_PER_BLOCK;

                for (size_t blk = 0; blk < num_blocks_per_row; blk++) {
                    const uint8_t* block = w_row + blk * BYTES_PER_BLOCK;

                    // Read global scales
                    uint16_t d_bits, dmin_bits;
                    std::memcpy(&d_bits, block, 2);
                    std::memcpy(&dmin_bits, block + 2, 2);
                    float d = fp16_to_fp32_cpu(d_bits);
                    float dmin = fp16_to_fp32_cpu(dmin_bits);

                    // Read sub-block scales (simplified)
                    const uint8_t* scales_ptr = block + 4;
                    uint8_t scales[8], mins[8];

                    // Simplified scale extraction
                    for (int j = 0; j < 4; j++) {
                        scales[j] = scales_ptr[j] & 63;
                        scales[j + 4] = (scales_ptr[j + 4] & 0xF) | ((scales_ptr[j] >> 6) << 4);
                        mins[j] = scales_ptr[j + 4] >> 4;
                        mins[j + 4] = scales_ptr[j + 8] >> 4;
                    }

                    const uint8_t* qs = block + 16;
                    size_t k_start = blk * SUPER_BLOCK_SIZE;

                    for (int sub = 0; sub < 8; sub++) {
                        float sub_scale = d * static_cast<float>(scales[sub]);
                        float sub_min = dmin * static_cast<float>(mins[sub]);

                        for (int i = 0; i < 16; i++) {
                            size_t k = k_start + sub * 32 + i * 2;
                            if (k >= K) break;

                            uint8_t byte = qs[sub * 16 + i];
                            int8_t q0 = byte & 0xF;
                            int8_t q1 = byte >> 4;

                            float w0 = sub_scale * static_cast<float>(q0) - sub_min;
                            float w1 = sub_scale * static_cast<float>(q1) - sub_min;

                            sum += A[m * K + k] * w0;
                            if (k + 1 < K) {
                                sum += A[m * K + k + 1] * w1;
                            }
                        }
                    }
                }

                out[m * N + n] = sum;
            }
        }
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
    registry.register_op(OpType::QuantizedMatMul, BackendType::CPU,
                        []() { return std::make_unique<CPUQuantizedMatMulOp>(); });
    registry.register_op(OpType::LayerNorm, BackendType::CPU,
                        []() { return std::make_unique<CPULayerNormOp>(); });
    registry.register_op(OpType::RMSNorm, BackendType::CPU,
                        []() { return std::make_unique<CPURMSNormOp>(); });

    GRANITE_LOG_DEBUG("Registered CPU operators");
}

}  // namespace granite

#endif  // GRANITE_HAS_CPU

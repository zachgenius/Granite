#include "granite/operators.h"
#include "granite/gguf.h"
#include "granite/log.h"

#include <cmath>
#include <algorithm>
#include <cstring>

#ifdef GRANITE_HAS_CPU

extern "C" {
#include "grml/grml.h"
}

namespace granite {

// =============================================================================
// CPU Binary Operators (Add, Sub, Mul, Div) — backed by GrML SIMD kernels
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

        int64_t n = static_cast<int64_t>(out.numel());

        // Simple case: same shape, no broadcasting
        if (a.numel() == b.numel() && a.numel() == static_cast<size_t>(n)) {
            if constexpr (Op == OpType::Add) {
                grml_kernel_add_f32(n, pout, pa, pb);
            } else if constexpr (Op == OpType::Sub) {
                grml_kernel_sub_f32(n, pout, pa, pb);
            } else if constexpr (Op == OpType::Mul) {
                grml_kernel_mul_f32(n, pout, pa, pb);
            } else if constexpr (Op == OpType::Div) {
                grml_kernel_div_f32(n, pout, pa, pb);
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
// CPU Unary Operators (ReLU, GELU, SiLU, Sigmoid, Tanh) — backed by GrML SIMD
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

        int64_t n = static_cast<int64_t>(x.numel());

        if constexpr (Op == OpType::ReLU) {
            grml_kernel_relu_f32(n, pout, px);
        } else if constexpr (Op == OpType::GELU) {
            grml_kernel_gelu_f32(n, pout, px);
        } else if constexpr (Op == OpType::SiLU) {
            grml_kernel_silu_f32(n, pout, px);
        } else if constexpr (Op == OpType::Sigmoid) {
            grml_kernel_sigmoid_f32(n, pout, px);
        } else if constexpr (Op == OpType::Tanh) {
            grml_kernel_tanh_f32(n, pout, px);
        }

        ctx.backend->unmap_buffer(x.buffer());
        ctx.backend->unmap_buffer(out.buffer());

        return {};
    }
};

// =============================================================================
// CPU Softmax — backed by GrML SIMD kernel
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

        int64_t outer = 1;
        for (int i = 0; i < axis; i++) {
            outer *= static_cast<int64_t>(x.size(i));
        }
        int64_t inner = static_cast<int64_t>(x.size(axis));

        grml_kernel_softmax_f32(outer, inner, pout, px);

        ctx.backend->unmap_buffer(x.buffer());
        ctx.backend->unmap_buffer(out.buffer());

        return {};
    }
};

// =============================================================================
// CPU MatMul — backed by GrML SIMD kernel
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

        int64_t M = static_cast<int64_t>(a.size(0));
        int64_t K = static_cast<int64_t>(a.size(1));
        int64_t N = static_cast<int64_t>(b.size(1));

        grml_kernel_mul_mat_f32(M, N, K, pout, pa, pb, NULL);

        ctx.backend->unmap_buffer(a.buffer());
        ctx.backend->unmap_buffer(b.buffer());
        ctx.backend->unmap_buffer(out.buffer());

        return {};
    }
};

// =============================================================================
// CPU LayerNorm — backed by GrML SIMD kernel
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

        int64_t dim = static_cast<int64_t>(x.size(x.ndim() - 1));
        int64_t nrows = static_cast<int64_t>(x.numel()) / dim;

        grml_kernel_layer_norm_f32(nrows, dim, eps, pout, px, pw, pb);

        ctx.backend->unmap_buffer(x.buffer());
        ctx.backend->unmap_buffer(weight.buffer());
        ctx.backend->unmap_buffer(bias.buffer());
        ctx.backend->unmap_buffer(out.buffer());

        return {};
    }
};

// =============================================================================
// CPU RMSNorm — backed by GrML SIMD kernel
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

        int64_t dim = static_cast<int64_t>(x.size(x.ndim() - 1));
        int64_t nrows = static_cast<int64_t>(x.numel()) / dim;

        grml_kernel_rms_norm_f32(nrows, dim, eps, pout, px, pw);

        ctx.backend->unmap_buffer(x.buffer());
        ctx.backend->unmap_buffer(weight.buffer());
        ctx.backend->unmap_buffer(out.buffer());

        return {};
    }
};

// =============================================================================
// CPU Quantized MatMul — Q8_0/Q4_0 backed by GrML SIMD, Q4_K kept as-is
// =============================================================================

namespace {

// FP16 helpers retained for Q4_K path (GrML has no Q4_K support yet)
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

class CPUQuantizedMatMulOp : public IOperator {
public:
    OpType type() const override { return OpType::QuantizedMatMul; }

    Result<void> validate(const OpContext& ctx) const override {
        if (ctx.num_inputs() < 2) {
            return Error(ErrorCode::InvalidArgument, "QuantizedMatMul requires at least 2 inputs");
        }
        return {};
    }

    Result<std::vector<std::vector<int64_t>>> infer_shapes(const OpContext& ctx) const override {
        const auto& a = ctx.inputs[0];

        auto weight_shape = ctx.attrs.get<std::vector<int64_t>>("weight_shape");
        if (weight_shape.size() < 2) {
            return Error(ErrorCode::InvalidArgument, "weight_shape attribute required");
        }

        int64_t m = a.size(a.ndim() - 2);
        int64_t n = weight_shape[weight_shape.size() - 1];

        return std::vector<std::vector<int64_t>>{{m, n}};
    }

    Result<void> execute(OpContext& ctx) override {
        const auto& activation = ctx.inputs[0];
        const auto& weights = ctx.inputs[1];
        auto& out = ctx.outputs[0];

        auto quant_type_val = ctx.attrs.get<int64_t>("quant_type", static_cast<int64_t>(GGMLType::Q8_0));
        auto quant_type = static_cast<GGMLType>(quant_type_val);

        auto weight_shape = ctx.attrs.get<std::vector<int64_t>>("weight_shape");
        if (weight_shape.size() < 2) {
            return Error(ErrorCode::InvalidArgument, "weight_shape required for QuantizedMatMul");
        }

        auto map_act = ctx.backend->map_buffer(activation.buffer());
        auto map_w = ctx.backend->map_buffer(weights.buffer());
        auto map_out = ctx.backend->map_buffer(out.buffer());

        if (!map_act.ok() || !map_w.ok() || !map_out.ok()) {
            return Error(ErrorCode::InternalError, "Failed to map buffers");
        }

        int64_t M = static_cast<int64_t>(activation.size(activation.ndim() - 2));
        int64_t K = static_cast<int64_t>(activation.size(activation.ndim() - 1));
        int64_t N = weight_shape[weight_shape.size() - 1];

        const auto* w_data = static_cast<const uint8_t*>(map_w.value());

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
                              float* out, int64_t M, int64_t K, int64_t N,
                              GGMLType quant_type) {
        switch (quant_type) {
            case GGMLType::Q8_0:
                grml_kernel_mul_mat_q8_0_f32(M, N, K, out, A, W, NULL);
                break;
            case GGMLType::Q4_0:
                grml_kernel_mul_mat_q4_0_f32(M, N, K, out, A, W, NULL);
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
                              uint16_t* out, int64_t M, int64_t K, int64_t N,
                              GGMLType quant_type) {
        // Convert FP16 to FP32 using GrML, compute, convert back
        std::vector<float> a_fp32(static_cast<size_t>(M * K));
        std::vector<float> out_fp32(static_cast<size_t>(M * N), 0.0f);

        grml_f16_to_f32_row(A, a_fp32.data(), M * K);

        auto result = execute_fp32(a_fp32.data(), W, out_fp32.data(), M, K, N, quant_type);
        if (!result.ok()) return result;

        grml_f32_to_f16_row(out_fp32.data(), out, M * N);

        return {};
    }

    // Q4_K MatMul: kept as-is (GrML has no Q4_K support yet)
    void matmul_q4_k_fp32(const float* A, const uint8_t* W,
                         float* out, int64_t M, int64_t K, int64_t N) {
        constexpr int64_t SUPER_BLOCK_SIZE = 256;
        constexpr int64_t BYTES_PER_BLOCK = 144;
        int64_t num_blocks_per_row = (K + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;

        for (int64_t m = 0; m < M; m++) {
            for (int64_t n = 0; n < N; n++) {
                float sum = 0.0f;

                const uint8_t* w_row = W + n * num_blocks_per_row * BYTES_PER_BLOCK;

                for (int64_t blk = 0; blk < num_blocks_per_row; blk++) {
                    const uint8_t* block = w_row + blk * BYTES_PER_BLOCK;

                    uint16_t d_bits, dmin_bits;
                    std::memcpy(&d_bits, block, 2);
                    std::memcpy(&dmin_bits, block + 2, 2);
                    float d = fp16_to_fp32_cpu(d_bits);
                    float dmin = fp16_to_fp32_cpu(dmin_bits);

                    const uint8_t* scales_ptr = block + 4;
                    uint8_t scales[8], mins[8];

                    for (int j = 0; j < 4; j++) {
                        scales[j] = scales_ptr[j] & 63;
                        scales[j + 4] = (scales_ptr[j + 4] & 0xF) | ((scales_ptr[j] >> 6) << 4);
                        mins[j] = scales_ptr[j + 4] >> 4;
                        mins[j + 4] = scales_ptr[j + 8] >> 4;
                    }

                    const uint8_t* qs = block + 16;
                    int64_t k_start = blk * SUPER_BLOCK_SIZE;

                    for (int sub = 0; sub < 8; sub++) {
                        float sub_scale = d * static_cast<float>(scales[sub]);
                        float sub_min = dmin * static_cast<float>(mins[sub]);

                        for (int i = 0; i < 16; i++) {
                            int64_t k = k_start + sub * 32 + i * 2;
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

}

}  // namespace granite

#endif  // GRANITE_HAS_CPU

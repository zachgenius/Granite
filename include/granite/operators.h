#pragma once

#include "granite/types.h"
#include "granite/error.h"
#include "granite/tensor.h"

#include <memory>
#include <functional>
#include <unordered_map>
#include <vector>
#include <variant>
#include <string>

namespace granite {

// =============================================================================
// Operator Types
// =============================================================================

enum class OpType : uint32_t {
    // Tensor operations
    Reshape,
    Transpose,
    Concat,
    Split,
    Slice,
    Gather,

    // Element-wise math
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Sqrt,
    Exp,
    Log,
    Pow,

    // Activations
    ReLU,
    GELU,
    SiLU,
    Sigmoid,
    Tanh,
    Softmax,

    // Matrix operations
    MatMul,
    BatchMatMul,
    QuantizedMatMul,  // MatMul with quantized weights (dequantize on-the-fly)

    // Convolution
    Conv2D,
    DepthwiseConv2D,
    ConvTranspose2D,

    // Normalization
    LayerNorm,
    RMSNorm,
    BatchNorm,
    GroupNorm,

    // Pooling
    MaxPool2D,
    AvgPool2D,
    GlobalAvgPool,
    AdaptiveAvgPool,

    // Attention
    ScaledDotProductAttention,
    MultiHeadAttention,

    // Embedding & Position
    Embedding,
    RoPE,           // Rotary Position Embedding
    CausalMask,     // Generate causal attention mask

    // Quantization
    Quantize,
    Dequantize,

    // Custom
    Custom,
};

constexpr const char* op_type_name(OpType op) {
    switch (op) {
        case OpType::Reshape: return "Reshape";
        case OpType::Transpose: return "Transpose";
        case OpType::Concat: return "Concat";
        case OpType::Split: return "Split";
        case OpType::Slice: return "Slice";
        case OpType::Gather: return "Gather";
        case OpType::Add: return "Add";
        case OpType::Sub: return "Sub";
        case OpType::Mul: return "Mul";
        case OpType::Div: return "Div";
        case OpType::Neg: return "Neg";
        case OpType::Sqrt: return "Sqrt";
        case OpType::Exp: return "Exp";
        case OpType::Log: return "Log";
        case OpType::Pow: return "Pow";
        case OpType::ReLU: return "ReLU";
        case OpType::GELU: return "GELU";
        case OpType::SiLU: return "SiLU";
        case OpType::Sigmoid: return "Sigmoid";
        case OpType::Tanh: return "Tanh";
        case OpType::Softmax: return "Softmax";
        case OpType::MatMul: return "MatMul";
        case OpType::BatchMatMul: return "BatchMatMul";
        case OpType::QuantizedMatMul: return "QuantizedMatMul";
        case OpType::Conv2D: return "Conv2D";
        case OpType::DepthwiseConv2D: return "DepthwiseConv2D";
        case OpType::ConvTranspose2D: return "ConvTranspose2D";
        case OpType::LayerNorm: return "LayerNorm";
        case OpType::RMSNorm: return "RMSNorm";
        case OpType::BatchNorm: return "BatchNorm";
        case OpType::GroupNorm: return "GroupNorm";
        case OpType::MaxPool2D: return "MaxPool2D";
        case OpType::AvgPool2D: return "AvgPool2D";
        case OpType::GlobalAvgPool: return "GlobalAvgPool";
        case OpType::AdaptiveAvgPool: return "AdaptiveAvgPool";
        case OpType::ScaledDotProductAttention: return "ScaledDotProductAttention";
        case OpType::MultiHeadAttention: return "MultiHeadAttention";
        case OpType::Embedding: return "Embedding";
        case OpType::RoPE: return "RoPE";
        case OpType::CausalMask: return "CausalMask";
        case OpType::Quantize: return "Quantize";
        case OpType::Dequantize: return "Dequantize";
        case OpType::Custom: return "Custom";
    }
    return "Unknown";
}

// =============================================================================
// Operator Attributes
// =============================================================================

using AttrValue = std::variant<
    int64_t,
    double,
    bool,
    std::string,
    std::vector<int64_t>,
    std::vector<double>
>;

class Attributes {
public:
    void set(const std::string& key, AttrValue value) {
        attrs_[key] = std::move(value);
    }

    template<typename T>
    T get(const std::string& key, T default_value = T{}) const {
        auto it = attrs_.find(key);
        if (it == attrs_.end()) {
            return default_value;
        }
        if (auto* val = std::get_if<T>(&it->second)) {
            return *val;
        }
        return default_value;
    }

    bool has(const std::string& key) const {
        return attrs_.find(key) != attrs_.end();
    }

private:
    std::unordered_map<std::string, AttrValue> attrs_;
};

// =============================================================================
// Operator Context
// =============================================================================

struct OpContext {
    IComputeBackend* backend = nullptr;
    std::vector<Tensor> inputs;
    std::vector<Tensor> outputs;
    Attributes attrs;

    // Helper accessors
    const Tensor& input(size_t i) const { return inputs.at(i); }
    Tensor& output(size_t i) { return outputs.at(i); }
    size_t num_inputs() const { return inputs.size(); }
    size_t num_outputs() const { return outputs.size(); }
};

// =============================================================================
// Operator Interface
// =============================================================================

class IOperator {
public:
    virtual ~IOperator() = default;

    /// Get the operator type
    [[nodiscard]] virtual OpType type() const = 0;

    /// Validate inputs and attributes
    [[nodiscard]] virtual Result<void> validate(const OpContext& ctx) const = 0;

    /// Infer output shapes from inputs
    [[nodiscard]] virtual Result<std::vector<std::vector<int64_t>>>
        infer_shapes(const OpContext& ctx) const = 0;

    /// Execute the operator
    [[nodiscard]] virtual Result<void> execute(OpContext& ctx) = 0;

    /// Check if this operator can be fused with another
    [[nodiscard]] virtual bool can_fuse_with(OpType next_op) const {
        (void)next_op;
        return false;
    }
};

// =============================================================================
// Operator Registry
// =============================================================================

using OperatorFactory = std::function<std::unique_ptr<IOperator>()>;

class OperatorRegistry {
public:
    static OperatorRegistry& instance();

    /// Register an operator implementation
    void register_op(OpType op, BackendType backend, OperatorFactory factory);

    /// Create an operator instance
    [[nodiscard]] std::unique_ptr<IOperator> create(OpType op, BackendType backend) const;

    /// Check if an implementation exists
    [[nodiscard]] bool has_implementation(OpType op, BackendType backend) const;

    /// Get all registered operators for a backend
    [[nodiscard]] std::vector<OpType> get_ops_for_backend(BackendType backend) const;

private:
    OperatorRegistry() = default;

    struct Key {
        OpType op;
        BackendType backend;

        bool operator==(const Key& other) const {
            return op == other.op && backend == other.backend;
        }
    };

    struct KeyHash {
        size_t operator()(const Key& k) const {
            return std::hash<uint32_t>{}(static_cast<uint32_t>(k.op)) ^
                   (std::hash<uint8_t>{}(static_cast<uint8_t>(k.backend)) << 16);
        }
    };

    std::unordered_map<Key, OperatorFactory, KeyHash> factories_;
};

// Registration macro
#define GRANITE_REGISTER_OP(op_type, backend_type, cls) \
    static bool _granite_reg_##op_type##_##backend_type = [] { \
        ::granite::OperatorRegistry::instance().register_op( \
            ::granite::OpType::op_type, \
            ::granite::BackendType::backend_type, \
            []() { return std::make_unique<cls>(); }); \
        return true; \
    }()

// =============================================================================
// Backend-specific registration functions
// =============================================================================

/// Register CPU operators (called automatically during initialization)
void register_cpu_operators();

/// Register Metal operators (called automatically during initialization)
void register_metal_operators();

/// Initialize all available operators
void initialize_operators();

// =============================================================================
// Functional API (convenience functions)
// =============================================================================

namespace ops {

/// Element-wise addition
Result<Tensor> add(const Tensor& a, const Tensor& b);

/// Element-wise subtraction
Result<Tensor> sub(const Tensor& a, const Tensor& b);

/// Element-wise multiplication
Result<Tensor> mul(const Tensor& a, const Tensor& b);

/// Element-wise division
Result<Tensor> div(const Tensor& a, const Tensor& b);

/// Matrix multiplication
Result<Tensor> matmul(const Tensor& a, const Tensor& b);

/// ReLU activation
Result<Tensor> relu(const Tensor& x);

/// GELU activation
Result<Tensor> gelu(const Tensor& x);

/// SiLU (Swish) activation
Result<Tensor> silu(const Tensor& x);

/// Softmax along axis
Result<Tensor> softmax(const Tensor& x, int axis = -1);

/// Layer normalization
Result<Tensor> layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias,
                          float eps = 1e-5f);

/// RMS normalization
Result<Tensor> rms_norm(const Tensor& x, const Tensor& weight, float eps = 1e-5f);

}  // namespace ops

}  // namespace granite

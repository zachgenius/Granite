#include "granite/operators.h"
#include "granite/log.h"

namespace granite {

// =============================================================================
// Operator Registry Implementation
// =============================================================================

OperatorRegistry& OperatorRegistry::instance() {
    static OperatorRegistry registry;
    return registry;
}

void OperatorRegistry::register_op(OpType op, BackendType backend, OperatorFactory factory) {
    Key key{op, backend};
    factories_[key] = std::move(factory);
}

std::unique_ptr<IOperator> OperatorRegistry::create(OpType op, BackendType backend) const {
    Key key{op, backend};
    auto it = factories_.find(key);
    if (it == factories_.end()) {
        GRANITE_LOG_WARN("No implementation for {} on {}",
                         op_type_name(op), backend_name(backend));
        return nullptr;
    }
    return it->second();
}

bool OperatorRegistry::has_implementation(OpType op, BackendType backend) const {
    Key key{op, backend};
    return factories_.find(key) != factories_.end();
}

std::vector<OpType> OperatorRegistry::get_ops_for_backend(BackendType backend) const {
    std::vector<OpType> ops;
    for (const auto& [key, factory] : factories_) {
        if (key.backend == backend) {
            ops.push_back(key.op);
        }
    }
    return ops;
}

// =============================================================================
// Functional API Implementation
// =============================================================================

namespace ops {

namespace {

// Helper to run a unary operator
Result<Tensor> run_unary_op(OpType op, const Tensor& x) {
    if (!x.backend()) {
        return Error(ErrorCode::NullPointer, "Tensor has no backend");
    }

    auto& registry = OperatorRegistry::instance();
    auto impl = registry.create(op, x.backend()->get_type());
    if (!impl) {
        return Error(ErrorCode::OperatorNotImplemented,
                     fmt::format("{} not implemented for {}", op_type_name(op),
                                 backend_name(x.backend()->get_type())));
    }

    // Infer output shape
    OpContext ctx;
    ctx.backend = x.backend();
    ctx.inputs = {x};

    auto shapes_result = impl->infer_shapes(ctx);
    if (!shapes_result.ok()) {
        return shapes_result.error();
    }

    // Allocate output
    auto& out_shape = shapes_result.value()[0];
    auto out_result = Tensor::allocate(out_shape, x.dtype(), x.backend());
    if (!out_result.ok()) {
        return out_result.error();
    }

    ctx.outputs = {std::move(out_result).take()};

    // Execute
    auto exec_result = impl->execute(ctx);
    if (!exec_result.ok()) {
        return exec_result.error();
    }

    return std::move(ctx.outputs[0]);
}

// Helper to run a binary operator
Result<Tensor> run_binary_op(OpType op, const Tensor& a, const Tensor& b) {
    if (!a.backend() || !b.backend()) {
        return Error(ErrorCode::NullPointer, "Tensor has no backend");
    }

    if (a.backend() != b.backend()) {
        return Error(ErrorCode::InvalidArgument, "Tensors must be on the same backend");
    }

    auto& registry = OperatorRegistry::instance();
    auto impl = registry.create(op, a.backend()->get_type());
    if (!impl) {
        return Error(ErrorCode::OperatorNotImplemented,
                     fmt::format("{} not implemented for {}", op_type_name(op),
                                 backend_name(a.backend()->get_type())));
    }

    // Infer output shape
    OpContext ctx;
    ctx.backend = a.backend();
    ctx.inputs = {a, b};

    auto shapes_result = impl->infer_shapes(ctx);
    if (!shapes_result.ok()) {
        return shapes_result.error();
    }

    // Allocate output (use dtype of first input)
    auto& out_shape = shapes_result.value()[0];
    auto out_result = Tensor::allocate(out_shape, a.dtype(), a.backend());
    if (!out_result.ok()) {
        return out_result.error();
    }

    ctx.outputs = {std::move(out_result).take()};

    // Execute
    auto exec_result = impl->execute(ctx);
    if (!exec_result.ok()) {
        return exec_result.error();
    }

    return std::move(ctx.outputs[0]);
}

}  // anonymous namespace

Result<Tensor> add(const Tensor& a, const Tensor& b) {
    return run_binary_op(OpType::Add, a, b);
}

Result<Tensor> sub(const Tensor& a, const Tensor& b) {
    return run_binary_op(OpType::Sub, a, b);
}

Result<Tensor> mul(const Tensor& a, const Tensor& b) {
    return run_binary_op(OpType::Mul, a, b);
}

Result<Tensor> div(const Tensor& a, const Tensor& b) {
    return run_binary_op(OpType::Div, a, b);
}

Result<Tensor> matmul(const Tensor& a, const Tensor& b) {
    return run_binary_op(OpType::MatMul, a, b);
}

Result<Tensor> relu(const Tensor& x) {
    return run_unary_op(OpType::ReLU, x);
}

Result<Tensor> gelu(const Tensor& x) {
    return run_unary_op(OpType::GELU, x);
}

Result<Tensor> silu(const Tensor& x) {
    return run_unary_op(OpType::SiLU, x);
}

Result<Tensor> softmax(const Tensor& x, int axis) {
    if (!x.backend()) {
        return Error(ErrorCode::NullPointer, "Tensor has no backend");
    }

    auto& registry = OperatorRegistry::instance();
    auto impl = registry.create(OpType::Softmax, x.backend()->get_type());
    if (!impl) {
        return Error(ErrorCode::OperatorNotImplemented,
                     fmt::format("Softmax not implemented for {}",
                                 backend_name(x.backend()->get_type())));
    }

    OpContext ctx;
    ctx.backend = x.backend();
    ctx.inputs = {x};
    ctx.attrs.set("axis", static_cast<int64_t>(axis));

    auto shapes_result = impl->infer_shapes(ctx);
    if (!shapes_result.ok()) {
        return shapes_result.error();
    }

    auto out_result = Tensor::allocate(shapes_result.value()[0], x.dtype(), x.backend());
    if (!out_result.ok()) {
        return out_result.error();
    }

    ctx.outputs = {std::move(out_result).take()};

    auto exec_result = impl->execute(ctx);
    if (!exec_result.ok()) {
        return exec_result.error();
    }

    return std::move(ctx.outputs[0]);
}

Result<Tensor> layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias, float eps) {
    if (!x.backend()) {
        return Error(ErrorCode::NullPointer, "Tensor has no backend");
    }

    auto& registry = OperatorRegistry::instance();
    auto impl = registry.create(OpType::LayerNorm, x.backend()->get_type());
    if (!impl) {
        return Error(ErrorCode::OperatorNotImplemented,
                     fmt::format("LayerNorm not implemented for {}",
                                 backend_name(x.backend()->get_type())));
    }

    OpContext ctx;
    ctx.backend = x.backend();
    ctx.inputs = {x, weight, bias};
    ctx.attrs.set("eps", static_cast<double>(eps));

    auto shapes_result = impl->infer_shapes(ctx);
    if (!shapes_result.ok()) {
        return shapes_result.error();
    }

    auto out_result = Tensor::allocate(shapes_result.value()[0], x.dtype(), x.backend());
    if (!out_result.ok()) {
        return out_result.error();
    }

    ctx.outputs = {std::move(out_result).take()};

    auto exec_result = impl->execute(ctx);
    if (!exec_result.ok()) {
        return exec_result.error();
    }

    return std::move(ctx.outputs[0]);
}

Result<Tensor> rms_norm(const Tensor& x, const Tensor& weight, float eps) {
    if (!x.backend()) {
        return Error(ErrorCode::NullPointer, "Tensor has no backend");
    }

    auto& registry = OperatorRegistry::instance();
    auto impl = registry.create(OpType::RMSNorm, x.backend()->get_type());
    if (!impl) {
        return Error(ErrorCode::OperatorNotImplemented,
                     fmt::format("RMSNorm not implemented for {}",
                                 backend_name(x.backend()->get_type())));
    }

    OpContext ctx;
    ctx.backend = x.backend();
    ctx.inputs = {x, weight};
    ctx.attrs.set("eps", static_cast<double>(eps));

    auto shapes_result = impl->infer_shapes(ctx);
    if (!shapes_result.ok()) {
        return shapes_result.error();
    }

    auto out_result = Tensor::allocate(shapes_result.value()[0], x.dtype(), x.backend());
    if (!out_result.ok()) {
        return out_result.error();
    }

    ctx.outputs = {std::move(out_result).take()};

    auto exec_result = impl->execute(ctx);
    if (!exec_result.ok()) {
        return exec_result.error();
    }

    return std::move(ctx.outputs[0]);
}

}  // namespace ops

// =============================================================================
// Operator Initialization
// =============================================================================

// Weak symbols - defined in backend-specific files if available
#ifdef GRANITE_HAS_CPU
// Defined in src/operators/cpu/cpu_operators.cpp
#else
void register_cpu_operators() {}
#endif

#ifdef GRANITE_HAS_METAL
// Defined in src/operators/metal/metal_operators.mm
#else
void register_metal_operators() {}
#endif

#ifdef GRANITE_HAS_VULKAN
// Defined in src/operators/vulkan/vulkan_operators.cpp
#else
void register_vulkan_operators() {}
#endif

namespace {
    bool operators_initialized = false;
}

void initialize_operators() {
    if (operators_initialized) {
        return;
    }

#ifdef GRANITE_HAS_CPU
    register_cpu_operators();
#endif

#ifdef GRANITE_HAS_METAL
    register_metal_operators();
#endif

#ifdef GRANITE_HAS_VULKAN
    register_vulkan_operators();
#endif

    operators_initialized = true;
}

}  // namespace granite

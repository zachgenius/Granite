#include "granite/operators.h"
#include "granite/log.h"

#ifdef GRANITE_HAS_METAL

#include <Metal/Metal.hpp>
#include <unordered_map>
#include <string>

namespace granite {

// =============================================================================
// Metal Shader Source (embedded)
// =============================================================================

static const char* ELEMENTWISE_SHADER = R"(
#include <metal_stdlib>
using namespace metal;

kernel void add_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id < n) {
        out[id] = a[id] + b[id];
    }
}

kernel void sub_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id < n) {
        out[id] = a[id] - b[id];
    }
}

kernel void mul_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id < n) {
        out[id] = a[id] * b[id];
    }
}

kernel void div_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id < n) {
        out[id] = a[id] / b[id];
    }
}

kernel void relu_f32(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    if (id < n) {
        out[id] = max(x[id], 0.0f);
    }
}

kernel void gelu_f32(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    if (id < n) {
        float val = x[id];
        float x3 = val * val * val;
        float inner = 0.7978845608f * (val + 0.044715f * x3);
        out[id] = 0.5f * val * (1.0f + tanh(inner));
    }
}

kernel void silu_f32(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    if (id < n) {
        float val = x[id];
        out[id] = val / (1.0f + exp(-val));
    }
}

kernel void softmax_fused_f32(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint row [[thread_position_in_grid]])
{
    if (row < rows) {
        float max_val = -INFINITY;
        for (uint i = 0; i < cols; i++) {
            max_val = max(max_val, x[row * cols + i]);
        }
        float sum = 0.0f;
        for (uint i = 0; i < cols; i++) {
            float e = exp(x[row * cols + i] - max_val);
            out[row * cols + i] = e;
            sum += e;
        }
        float inv_sum = 1.0f / sum;
        for (uint i = 0; i < cols; i++) {
            out[row * cols + i] *= inv_sum;
        }
    }
}
)";

static const char* MATMUL_SHADER = R"(
#include <metal_stdlib>
using namespace metal;

#define TILE_SIZE 16

kernel void matmul_tiled_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]])
{
    threadgroup float As[TILE_SIZE][TILE_SIZE];
    threadgroup float Bs[TILE_SIZE][TILE_SIZE];

    uint row = gid.y;
    uint col = gid.x;
    uint local_row = tid.y;
    uint local_col = tid.x;

    float sum = 0.0f;
    uint num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < num_tiles; t++) {
        uint a_col = t * TILE_SIZE + local_col;
        if (row < M && a_col < K) {
            As[local_row][local_col] = A[row * K + a_col];
        } else {
            As[local_row][local_col] = 0.0f;
        }

        uint b_row = t * TILE_SIZE + local_row;
        if (b_row < K && col < N) {
            Bs[local_row][local_col] = B[b_row * N + col];
        } else {
            Bs[local_row][local_col] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += As[local_row][k] * Bs[k][local_col];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
)";

static const char* NORM_SHADER = R"(
#include <metal_stdlib>
using namespace metal;

kernel void layer_norm_f32(
    device const float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& num_rows [[buffer(4)]],
    constant uint& norm_size [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= num_rows) return;

    uint offset = row * norm_size;

    float sum = 0.0f;
    for (uint i = 0; i < norm_size; i++) {
        sum += x[offset + i];
    }
    float mean = sum / float(norm_size);

    float var_sum = 0.0f;
    for (uint i = 0; i < norm_size; i++) {
        float diff = x[offset + i] - mean;
        var_sum += diff * diff;
    }
    float var = var_sum / float(norm_size);
    float inv_std = rsqrt(var + eps);

    for (uint i = 0; i < norm_size; i++) {
        float normalized = (x[offset + i] - mean) * inv_std;
        out[offset + i] = normalized * weight[i] + bias[i];
    }
}

kernel void rms_norm_f32(
    device const float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& num_rows [[buffer(3)]],
    constant uint& norm_size [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= num_rows) return;

    uint offset = row * norm_size;

    float sum_sq = 0.0f;
    for (uint i = 0; i < norm_size; i++) {
        float val = x[offset + i];
        sum_sq += val * val;
    }

    float rms = sqrt(sum_sq / float(norm_size) + eps);
    float inv_rms = 1.0f / rms;

    for (uint i = 0; i < norm_size; i++) {
        out[offset + i] = x[offset + i] * inv_rms * weight[i];
    }
}
)";

// =============================================================================
// Metal Pipeline Cache
// =============================================================================

class MetalPipelineCache {
public:
    static MetalPipelineCache& instance() {
        static MetalPipelineCache cache;
        return cache;
    }

    MTL::ComputePipelineState* get_pipeline(MTL::Device* device, const std::string& name) {
        auto key = std::make_pair(device, name);
        auto it = pipelines_.find(name);
        if (it != pipelines_.end()) {
            return it->second;
        }

        // Compile the appropriate shader
        const char* source = nullptr;
        if (name.find("add_") == 0 || name.find("sub_") == 0 ||
            name.find("mul_") == 0 || name.find("div_") == 0 ||
            name.find("relu_") == 0 || name.find("gelu_") == 0 ||
            name.find("silu_") == 0 || name.find("softmax_") == 0) {
            source = ELEMENTWISE_SHADER;
        } else if (name.find("matmul_") == 0) {
            source = MATMUL_SHADER;
        } else if (name.find("layer_norm_") == 0 || name.find("rms_norm_") == 0) {
            source = NORM_SHADER;
        }

        if (!source) {
            GRANITE_LOG_ERROR("Unknown shader: {}", name);
            return nullptr;
        }

        NS::Error* error = nullptr;
        MTL::CompileOptions* options = MTL::CompileOptions::alloc()->init();
        NS::String* src = NS::String::string(source, NS::UTF8StringEncoding);

        MTL::Library* library = device->newLibrary(src, options, &error);
        options->release();

        if (!library) {
            if (error) {
                GRANITE_LOG_ERROR("Shader compilation failed: {}",
                                  error->localizedDescription()->utf8String());
            }
            return nullptr;
        }

        NS::String* func_name = NS::String::string(name.c_str(), NS::UTF8StringEncoding);
        MTL::Function* function = library->newFunction(func_name);
        library->release();

        if (!function) {
            GRANITE_LOG_ERROR("Function not found: {}", name);
            return nullptr;
        }

        MTL::ComputePipelineState* pipeline = device->newComputePipelineState(function, &error);
        function->release();

        if (!pipeline) {
            if (error) {
                GRANITE_LOG_ERROR("Pipeline creation failed: {}",
                                  error->localizedDescription()->utf8String());
            }
            return nullptr;
        }

        pipelines_[name] = pipeline;
        return pipeline;
    }

private:
    MetalPipelineCache() = default;
    std::unordered_map<std::string, MTL::ComputePipelineState*> pipelines_;
};

// =============================================================================
// Metal Operator Base
// =============================================================================

class MetalOperatorBase : public IOperator {
protected:
    MTL::ComputePipelineState* get_pipeline(IComputeBackend* backend, const std::string& name) {
        // Get the device from the backend
        // This is a simplified approach - in production we'd have a proper interface
        auto* device = MTL::CreateSystemDefaultDevice();
        return MetalPipelineCache::instance().get_pipeline(device, name);
    }

    Result<void> dispatch_1d(IComputeBackend* backend, MTL::ComputePipelineState* pipeline,
                              const std::vector<BufferHandle>& buffers,
                              const std::vector<uint32_t>& constants,
                              uint32_t num_threads) {
        GRANITE_TRY(backend->begin_commands());

        // Bind the pipeline
        PipelineHandle ph{reinterpret_cast<uint64_t>(pipeline)};
        // Note: This is a simplified approach. In production, we'd properly integrate
        // with the backend's pipeline management.

        // For now, we'll use the backend's command recording directly
        // This requires the backend to expose the Metal objects
        GRANITE_TRY(backend->end_commands());
        GRANITE_TRY(backend->submit());
        GRANITE_TRY(backend->wait_for_completion());

        return {};
    }
};

// =============================================================================
// Binary Operators (Add, Sub, Mul, Div)
// =============================================================================

template<OpType Op>
class MetalBinaryOp : public IOperator {
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
        // Broadcasting: return the broadcast shape
        auto shape = broadcast_shapes(ctx.inputs[0].shape(), ctx.inputs[1].shape());
        return std::vector<std::vector<int64_t>>{shape};
    }

    Result<void> execute(OpContext& ctx) override {
        const auto& a = ctx.inputs[0];
        const auto& b = ctx.inputs[1];
        auto& out = ctx.outputs[0];

        // For now, require same shape (no broadcasting)
        if (a.numel() != b.numel() || a.numel() != out.numel()) {
            return Error(ErrorCode::ShapeMismatch, "Broadcasting not yet implemented");
        }

        std::string kernel_name;
        switch (Op) {
            case OpType::Add: kernel_name = "add_f32"; break;
            case OpType::Sub: kernel_name = "sub_f32"; break;
            case OpType::Mul: kernel_name = "mul_f32"; break;
            case OpType::Div: kernel_name = "div_f32"; break;
            default: return Error(ErrorCode::InternalError, "Unknown binary op");
        }

        auto* device = MTL::CreateSystemDefaultDevice();
        auto* pipeline = MetalPipelineCache::instance().get_pipeline(device, kernel_name);
        if (!pipeline) {
            return Error(ErrorCode::ShaderCompilationFailed, "Failed to get pipeline");
        }

        auto* queue = device->newCommandQueue();
        auto* cmd = queue->commandBuffer();
        auto* encoder = cmd->computeCommandEncoder();

        encoder->setComputePipelineState(pipeline);

        // Get Metal buffers from handles
        // Note: This is simplified - we'd need proper access to the backend's buffer map
        auto map_a = ctx.backend->map_buffer(a.buffer());
        auto map_b = ctx.backend->map_buffer(b.buffer());
        auto map_out = ctx.backend->map_buffer(out.buffer());

        if (!map_a.ok() || !map_b.ok() || !map_out.ok()) {
            encoder->endEncoding();
            cmd->commit();
            queue->release();
            return Error(ErrorCode::InternalError, "Failed to map buffers");
        }

        // Create temporary Metal buffers
        auto* buf_a = device->newBuffer(map_a.value(), a.size_bytes(), MTL::ResourceStorageModeShared);
        auto* buf_b = device->newBuffer(map_b.value(), b.size_bytes(), MTL::ResourceStorageModeShared);
        auto* buf_out = device->newBuffer(out.size_bytes(), MTL::ResourceStorageModeShared);

        encoder->setBuffer(buf_a, 0, 0);
        encoder->setBuffer(buf_b, 0, 1);
        encoder->setBuffer(buf_out, 0, 2);

        uint32_t n = static_cast<uint32_t>(a.numel());
        encoder->setBytes(&n, sizeof(n), 3);

        MTL::Size grid_size = MTL::Size::Make(n, 1, 1);
        MTL::Size tg_size = MTL::Size::Make(std::min(n, 256u), 1, 1);
        encoder->dispatchThreads(grid_size, tg_size);

        encoder->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();

        // Copy result back
        std::memcpy(map_out.value(), buf_out->contents(), out.size_bytes());

        buf_a->release();
        buf_b->release();
        buf_out->release();
        queue->release();

        ctx.backend->unmap_buffer(a.buffer());
        ctx.backend->unmap_buffer(b.buffer());
        ctx.backend->unmap_buffer(out.buffer());

        return {};
    }
};

// =============================================================================
// Unary Operators (ReLU, GELU, SiLU)
// =============================================================================

template<OpType Op>
class MetalUnaryOp : public IOperator {
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

        std::string kernel_name;
        switch (Op) {
            case OpType::ReLU: kernel_name = "relu_f32"; break;
            case OpType::GELU: kernel_name = "gelu_f32"; break;
            case OpType::SiLU: kernel_name = "silu_f32"; break;
            default: return Error(ErrorCode::InternalError, "Unknown unary op");
        }

        auto* device = MTL::CreateSystemDefaultDevice();
        auto* pipeline = MetalPipelineCache::instance().get_pipeline(device, kernel_name);
        if (!pipeline) {
            return Error(ErrorCode::ShaderCompilationFailed, "Failed to get pipeline");
        }

        auto* queue = device->newCommandQueue();
        auto* cmd = queue->commandBuffer();
        auto* encoder = cmd->computeCommandEncoder();

        encoder->setComputePipelineState(pipeline);

        auto map_x = ctx.backend->map_buffer(x.buffer());
        auto map_out = ctx.backend->map_buffer(out.buffer());

        if (!map_x.ok() || !map_out.ok()) {
            encoder->endEncoding();
            cmd->commit();
            queue->release();
            return Error(ErrorCode::InternalError, "Failed to map buffers");
        }

        auto* buf_x = device->newBuffer(map_x.value(), x.size_bytes(), MTL::ResourceStorageModeShared);
        auto* buf_out = device->newBuffer(out.size_bytes(), MTL::ResourceStorageModeShared);

        encoder->setBuffer(buf_x, 0, 0);
        encoder->setBuffer(buf_out, 0, 1);

        uint32_t n = static_cast<uint32_t>(x.numel());
        encoder->setBytes(&n, sizeof(n), 2);

        MTL::Size grid_size = MTL::Size::Make(n, 1, 1);
        MTL::Size tg_size = MTL::Size::Make(std::min(n, 256u), 1, 1);
        encoder->dispatchThreads(grid_size, tg_size);

        encoder->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();

        std::memcpy(map_out.value(), buf_out->contents(), out.size_bytes());

        buf_x->release();
        buf_out->release();
        queue->release();

        ctx.backend->unmap_buffer(x.buffer());
        ctx.backend->unmap_buffer(out.buffer());

        return {};
    }
};

// =============================================================================
// MatMul Operator
// =============================================================================

class MetalMatMulOp : public IOperator {
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

        // Check inner dimensions match
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

        std::vector<int64_t> out_shape;

        // Handle batched case
        if (a.ndim() > 2 || b.ndim() > 2) {
            // Broadcast batch dimensions
            // For simplicity, just handle 2D case for now
        }

        int64_t m = a.size(a.ndim() - 2);
        int64_t n = b.size(b.ndim() - 1);

        out_shape = {m, n};
        return std::vector<std::vector<int64_t>>{out_shape};
    }

    Result<void> execute(OpContext& ctx) override {
        const auto& a = ctx.inputs[0];
        const auto& b = ctx.inputs[1];
        auto& out = ctx.outputs[0];

        auto* device = MTL::CreateSystemDefaultDevice();
        auto* pipeline = MetalPipelineCache::instance().get_pipeline(device, "matmul_tiled_f32");
        if (!pipeline) {
            return Error(ErrorCode::ShaderCompilationFailed, "Failed to get matmul pipeline");
        }

        uint32_t M = static_cast<uint32_t>(a.size(0));
        uint32_t K = static_cast<uint32_t>(a.size(1));
        uint32_t N = static_cast<uint32_t>(b.size(1));

        auto* queue = device->newCommandQueue();
        auto* cmd = queue->commandBuffer();
        auto* encoder = cmd->computeCommandEncoder();

        encoder->setComputePipelineState(pipeline);

        auto map_a = ctx.backend->map_buffer(a.buffer());
        auto map_b = ctx.backend->map_buffer(b.buffer());
        auto map_out = ctx.backend->map_buffer(out.buffer());

        if (!map_a.ok() || !map_b.ok() || !map_out.ok()) {
            encoder->endEncoding();
            cmd->commit();
            queue->release();
            return Error(ErrorCode::InternalError, "Failed to map buffers");
        }

        auto* buf_a = device->newBuffer(map_a.value(), a.size_bytes(), MTL::ResourceStorageModeShared);
        auto* buf_b = device->newBuffer(map_b.value(), b.size_bytes(), MTL::ResourceStorageModeShared);
        auto* buf_out = device->newBuffer(out.size_bytes(), MTL::ResourceStorageModeShared);

        encoder->setBuffer(buf_a, 0, 0);
        encoder->setBuffer(buf_b, 0, 1);
        encoder->setBuffer(buf_out, 0, 2);
        encoder->setBytes(&M, sizeof(M), 3);
        encoder->setBytes(&N, sizeof(N), 4);
        encoder->setBytes(&K, sizeof(K), 5);

        // Use tiled dispatch
        constexpr uint32_t TILE_SIZE = 16;
        MTL::Size grid_size = MTL::Size::Make(
            (N + TILE_SIZE - 1) / TILE_SIZE * TILE_SIZE,
            (M + TILE_SIZE - 1) / TILE_SIZE * TILE_SIZE,
            1);
        MTL::Size tg_size = MTL::Size::Make(TILE_SIZE, TILE_SIZE, 1);
        encoder->dispatchThreads(grid_size, tg_size);

        encoder->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();

        std::memcpy(map_out.value(), buf_out->contents(), out.size_bytes());

        buf_a->release();
        buf_b->release();
        buf_out->release();
        queue->release();

        ctx.backend->unmap_buffer(a.buffer());
        ctx.backend->unmap_buffer(b.buffer());
        ctx.backend->unmap_buffer(out.buffer());

        return {};
    }
};

// =============================================================================
// Register Operators
// =============================================================================

void register_metal_operators() {
    auto& registry = OperatorRegistry::instance();

    registry.register_op(OpType::Add, BackendType::Metal,
                        []() { return std::make_unique<MetalBinaryOp<OpType::Add>>(); });
    registry.register_op(OpType::Sub, BackendType::Metal,
                        []() { return std::make_unique<MetalBinaryOp<OpType::Sub>>(); });
    registry.register_op(OpType::Mul, BackendType::Metal,
                        []() { return std::make_unique<MetalBinaryOp<OpType::Mul>>(); });
    registry.register_op(OpType::Div, BackendType::Metal,
                        []() { return std::make_unique<MetalBinaryOp<OpType::Div>>(); });
    registry.register_op(OpType::ReLU, BackendType::Metal,
                        []() { return std::make_unique<MetalUnaryOp<OpType::ReLU>>(); });
    registry.register_op(OpType::GELU, BackendType::Metal,
                        []() { return std::make_unique<MetalUnaryOp<OpType::GELU>>(); });
    registry.register_op(OpType::SiLU, BackendType::Metal,
                        []() { return std::make_unique<MetalUnaryOp<OpType::SiLU>>(); });
    registry.register_op(OpType::MatMul, BackendType::Metal,
                        []() { return std::make_unique<MetalMatMulOp>(); });

    GRANITE_LOG_DEBUG("Registered Metal operators");
}

}  // namespace granite

#endif  // GRANITE_HAS_METAL

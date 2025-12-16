#pragma once

#include "granite/error.h"

#ifdef GRANITE_HAS_METAL

// Forward declarations to avoid exposing Metal headers
namespace MTL {
    class Device;
    class Buffer;
}

namespace granite {

class MetalCompute;

// Get global MetalCompute instance (lazy initialized)
MetalCompute* get_metal_compute();

// MetalCompute provides high-level GPU operations for LLM inference
class MetalCompute {
public:
    MetalCompute();
    ~MetalCompute();

    Result<void> initialize(MTL::Device* device);
    void shutdown();

    // Synchronize all pending GPU operations
    void sync();

    // Commit current command buffer (async)
    void commit();

    // Check if initialized
    bool is_initialized() const;

    // Get Metal device
    MTL::Device* device() const;

    // =============================================================================
    // LLM Operations - Quantized Matrix Multiply
    // =============================================================================

    // y = x @ W^T where W is Q4_K quantized
    Result<void> matvec_q4k(
        MTL::Buffer* x,      // Input [K] float
        MTL::Buffer* W,      // Weights [N, K/256] Q4_K blocks (144 bytes each)
        MTL::Buffer* y,      // Output [N] float
        uint32_t K,          // Input dimension
        uint32_t N           // Output dimension
    );

    // Y = X @ W^T where W is Q4_K quantized (batched)
    Result<void> matmul_q4k(
        MTL::Buffer* X,      // Input [M, K] float
        MTL::Buffer* W,      // Weights [N, K/256] Q4_K blocks
        MTL::Buffer* Y,      // Output [M, N] float
        uint32_t M,          // Batch size (number of tokens)
        uint32_t K,          // Input dimension
        uint32_t N           // Output dimension
    );

    // =============================================================================
    // LLM Operations - FP16/FP32 Matrix Multiply
    // =============================================================================

    // y = x @ W^T where W is FP16
    Result<void> matvec_f16(
        MTL::Buffer* x,
        MTL::Buffer* W,
        MTL::Buffer* y,
        uint32_t K,
        uint32_t N
    );

    // =============================================================================
    // LLM Operations - Element-wise
    // =============================================================================

    // RMS Normalization
    Result<void> rms_norm(
        MTL::Buffer* x,
        MTL::Buffer* weight,
        MTL::Buffer* out,
        uint32_t size,
        float eps
    );

    // SiLU activation (in-place)
    Result<void> silu(MTL::Buffer* x, uint32_t size);

    // Element-wise multiply: c = a * b
    Result<void> elementwise_mul(
        MTL::Buffer* a,
        MTL::Buffer* b,
        MTL::Buffer* c,
        uint32_t size
    );

    // RoPE (Rotary Position Embedding)
    Result<void> rope(
        MTL::Buffer* x,
        uint32_t seq_len,
        uint32_t head_dim,
        uint32_t start_pos,
        float freq_base = 10000.0f
    );

    // =============================================================================
    // Buffer Management
    // =============================================================================

    // Create a Metal buffer
    MTL::Buffer* create_buffer(size_t size, bool shared = true);

private:
    class Impl;
    Impl* impl_ = nullptr;
};

}  // namespace granite

#endif  // GRANITE_HAS_METAL

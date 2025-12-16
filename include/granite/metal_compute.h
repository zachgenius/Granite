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

    // RMS Normalization (FP32 weights)
    Result<void> rms_norm(
        MTL::Buffer* x,
        MTL::Buffer* weight,
        MTL::Buffer* out,
        uint32_t size,
        float eps
    );

    // RMS Normalization with FP16 weights
    Result<void> rms_norm_f16(
        MTL::Buffer* x,
        MTL::Buffer* weight,  // FP16 weights
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

    // Element-wise add: c = a + b
    Result<void> elementwise_add(
        MTL::Buffer* a,
        MTL::Buffer* b,
        MTL::Buffer* c,
        uint32_t size
    );

    // =============================================================================
    // Attention Operations
    // =============================================================================

    // RoPE applied to Q and K tensors for multi-head attention
    // Q/K shape: [num_heads, seq_len, head_dim]
    Result<void> rope_multihead(
        MTL::Buffer* q,           // Q tensor
        MTL::Buffer* k,           // K tensor
        uint32_t num_heads_q,     // Number of Q heads
        uint32_t num_heads_k,     // Number of K heads (for GQA)
        uint32_t seq_len,         // Sequence length
        uint32_t head_dim,        // Head dimension
        uint32_t start_pos,       // Starting position for RoPE
        float freq_base = 10000.0f
    );

    // Softmax over last dimension (in-place)
    // x shape: [M, N], softmax over N dimension
    Result<void> softmax(
        MTL::Buffer* x,
        uint32_t M,               // Number of rows
        uint32_t N                // Number of columns (softmax dimension)
    );

    // Single-head attention: output = softmax(Q @ K^T / sqrt(d)) @ V
    // Q: [seq_q, head_dim], K: [seq_kv, head_dim], V: [seq_kv, head_dim]
    // output: [seq_q, head_dim]
    Result<void> attention_single_head(
        MTL::Buffer* Q,           // Query [seq_q, head_dim]
        MTL::Buffer* K,           // Key [seq_kv, head_dim]
        MTL::Buffer* V,           // Value [seq_kv, head_dim]
        MTL::Buffer* output,      // Output [seq_q, head_dim]
        uint32_t seq_q,           // Query sequence length (usually 1 for decode)
        uint32_t seq_kv,          // KV sequence length (includes cache)
        uint32_t head_dim,        // Head dimension
        uint32_t start_pos,       // Position for causal mask
        float scale               // 1/sqrt(head_dim)
    );

    // =============================================================================
    // GPU KV Cache
    // =============================================================================

    // Allocate GPU KV cache for a layer
    // Returns pair of (K buffer, V buffer)
    // Shape: [num_kv_heads, max_seq_len, head_dim]
    std::pair<MTL::Buffer*, MTL::Buffer*> create_kv_cache(
        uint32_t num_kv_heads,
        uint32_t max_seq_len,
        uint32_t head_dim
    );

    // Append new K/V to GPU cache
    // new_k, new_v: [num_kv_heads, seq_len, head_dim]
    // cache_k, cache_v: [num_kv_heads, max_seq_len, head_dim]
    Result<void> kv_cache_append(
        MTL::Buffer* cache_k,
        MTL::Buffer* cache_v,
        MTL::Buffer* new_k,
        MTL::Buffer* new_v,
        uint32_t num_kv_heads,
        uint32_t head_dim,
        uint32_t current_len,      // Current cache length
        uint32_t new_len,          // Length of new K/V (usually 1 for decode)
        uint32_t max_seq_len       // Max cache length
    );

    // Multi-head attention on GPU
    // Q: [num_heads, seq_q, head_dim]
    // K: [num_kv_heads, seq_kv, head_dim]
    // V: [num_kv_heads, seq_kv, head_dim]
    // output: [num_heads, seq_q, head_dim]
    Result<void> multihead_attention(
        MTL::Buffer* Q,
        MTL::Buffer* K,
        MTL::Buffer* V,
        MTL::Buffer* output,
        uint32_t num_heads,
        uint32_t num_kv_heads,
        uint32_t seq_q,
        uint32_t seq_kv,
        uint32_t head_dim,
        float scale
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

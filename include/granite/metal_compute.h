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
    // LLM Operations - Q8_0 Quantized Matrix Multiply
    // =============================================================================

    // y = x @ W^T where W is Q8_0 quantized
    Result<void> matvec_q8_0(
        MTL::Buffer* x,      // Input [K] float
        MTL::Buffer* W,      // Weights [N, K/32] Q8_0 blocks (34 bytes each)
        MTL::Buffer* y,      // Output [N] float
        uint32_t K,          // Input dimension
        uint32_t N           // Output dimension
    );

    // Y = X @ W^T where W is Q8_0 quantized (batched)
    Result<void> matmul_q8_0(
        MTL::Buffer* X,      // Input [M, K] float
        MTL::Buffer* W,      // Weights [N, K/32] Q8_0 blocks
        MTL::Buffer* Y,      // Output [M, N] float
        uint32_t M,          // Batch size (number of tokens)
        uint32_t K,          // Input dimension
        uint32_t N           // Output dimension
    );

    // =============================================================================
    // LLM Operations - Q4_0 Quantized Matrix Multiply
    // =============================================================================

    // y = x @ W^T where W is Q4_0 quantized
    Result<void> matvec_q4_0(
        MTL::Buffer* x,      // Input [K] float
        MTL::Buffer* W,      // Weights [N, K/32] Q4_0 blocks (18 bytes each)
        MTL::Buffer* y,      // Output [N] float
        uint32_t K,          // Input dimension
        uint32_t N           // Output dimension
    );

    // Y = X @ W^T where W is Q4_0 quantized (batched)
    Result<void> matmul_q4_0(
        MTL::Buffer* X,      // Input [M, K] float
        MTL::Buffer* W,      // Weights [N, K/32] Q4_0 blocks
        MTL::Buffer* Y,      // Output [M, N] float
        uint32_t M,          // Batch size (number of tokens)
        uint32_t K,          // Input dimension
        uint32_t N           // Output dimension
    );

    // =============================================================================
    // LLM Operations - IQ4_NL Quantized Matrix Multiply (Non-linear 4-bit I-quant)
    // =============================================================================

    // y = x @ W^T where W is IQ4_NL quantized
    // IQ4_NL uses a non-linear lookup table for better quality at 4-bit
    Result<void> matvec_iq4_nl(
        MTL::Buffer* x,      // Input [K] float
        MTL::Buffer* W,      // Weights [N, K/32] IQ4_NL blocks (18 bytes each)
        MTL::Buffer* y,      // Output [N] float
        uint32_t K,          // Input dimension
        uint32_t N           // Output dimension
    );

    // Y = X @ W^T where W is IQ4_NL quantized (batched)
    Result<void> matmul_iq4_nl(
        MTL::Buffer* X,      // Input [M, K] float
        MTL::Buffer* W,      // Weights [N, K/32] IQ4_NL blocks
        MTL::Buffer* Y,      // Output [M, N] float
        uint32_t M,          // Batch size (number of tokens)
        uint32_t K,          // Input dimension
        uint32_t N           // Output dimension
    );

    // =============================================================================
    // LLM Operations - IQ4_XS Quantized Matrix Multiply (Non-linear 4-bit with scales)
    // =============================================================================

    // y = x @ W^T where W is IQ4_XS quantized
    // IQ4_XS uses 256-element super-blocks with per-sub-block 6-bit scales
    Result<void> matvec_iq4_xs(
        MTL::Buffer* x,      // Input [K] float
        MTL::Buffer* W,      // Weights [N, K/256] IQ4_XS blocks (136 bytes each)
        MTL::Buffer* y,      // Output [N] float
        uint32_t K,          // Input dimension
        uint32_t N           // Output dimension
    );

    // Y = X @ W^T where W is IQ4_XS quantized (batched)
    Result<void> matmul_iq4_xs(
        MTL::Buffer* X,      // Input [M, K] float
        MTL::Buffer* W,      // Weights [N, K/256] IQ4_XS blocks
        MTL::Buffer* Y,      // Output [M, N] float
        uint32_t M,          // Batch size (number of tokens)
        uint32_t K,          // Input dimension
        uint32_t N           // Output dimension
    );

    // =============================================================================
    // LLM Operations - Q6_K Quantized Matrix Multiply
    // =============================================================================

    // y = x @ W^T where W is Q6_K quantized
    Result<void> matvec_q6_k(
        MTL::Buffer* x,      // Input [K] float
        MTL::Buffer* W,      // Weights [N, K/256] Q6_K blocks (210 bytes each)
        MTL::Buffer* y,      // Output [N] float
        uint32_t K,          // Input dimension
        uint32_t N           // Output dimension
    );

    // Y = X @ W^T where W is Q6_K quantized (batched)
    Result<void> matmul_q6_k(
        MTL::Buffer* X,      // Input [M, K] float
        MTL::Buffer* W,      // Weights [N, K/256] Q6_K blocks
        MTL::Buffer* Y,      // Output [M, N] float
        uint32_t M,          // Batch size (number of tokens)
        uint32_t K,          // Input dimension
        uint32_t N           // Output dimension
    );

    // =============================================================================
    // LLM Operations - Q5_K Quantized Matrix Multiply
    // =============================================================================

    // y = x @ W^T where W is Q5_K quantized
    Result<void> matvec_q5_k(
        MTL::Buffer* x,      // Input [K] float
        MTL::Buffer* W,      // Weights [N, K/256] Q5_K blocks (176 bytes each)
        MTL::Buffer* y,      // Output [N] float
        uint32_t K,          // Input dimension
        uint32_t N           // Output dimension
    );

    // Y = X @ W^T where W is Q5_K quantized (batched)
    Result<void> matmul_q5_k(
        MTL::Buffer* X,      // Input [M, K] float
        MTL::Buffer* W,      // Weights [N, K/256] Q5_K blocks
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

    // =============================================================================
    // Fused Kernels (for reduced memory bandwidth)
    // =============================================================================

    // Fused SiLU + Multiply: c = silu(a) * b
    // Used in SwiGLU FFN to eliminate intermediate buffer
    Result<void> silu_mul(
        MTL::Buffer* a,       // gate input
        MTL::Buffer* b,       // up input
        MTL::Buffer* c,       // output
        uint32_t size
    );

    // Fused RMSNorm + Q4_K MatVec: y = RMSNorm(x, weight) @ W^T
    // Eliminates intermediate normalized buffer write/read
    Result<void> rms_norm_matvec_q4k(
        MTL::Buffer* x,           // Input [K] float
        MTL::Buffer* norm_weight, // RMSNorm weight [K] half
        MTL::Buffer* W,           // Q4_K weights [N, K/256] blocks
        MTL::Buffer* y,           // Output [N] float
        uint32_t K,               // Input dimension
        uint32_t N,               // Output dimension
        float eps                 // RMSNorm epsilon
    );

    // Fused RMSNorm + FP16 MatVec: y = RMSNorm(x, weight) @ W^T
    Result<void> rms_norm_matvec_f16(
        MTL::Buffer* x,           // Input [K] float
        MTL::Buffer* norm_weight, // RMSNorm weight [K] half
        MTL::Buffer* W,           // FP16 weights [N, K]
        MTL::Buffer* y,           // Output [N] float
        uint32_t K,               // Input dimension
        uint32_t N,               // Output dimension
        float eps                 // RMSNorm epsilon
    );

    // Fused RMSNorm + Q8_0 MatVec: y = RMSNorm(x, weight) @ W^T
    Result<void> rms_norm_matvec_q8_0(
        MTL::Buffer* x,           // Input [K] float
        MTL::Buffer* norm_weight, // RMSNorm weight [K] half
        MTL::Buffer* W,           // Q8_0 weights [N, K/32] blocks
        MTL::Buffer* y,           // Output [N] float
        uint32_t K,               // Input dimension
        uint32_t N,               // Output dimension
        float eps                 // RMSNorm epsilon
    );

    // Fused RMSNorm + Q4_0 MatVec: y = RMSNorm(x, weight) @ W^T
    Result<void> rms_norm_matvec_q4_0(
        MTL::Buffer* x,           // Input [K] float
        MTL::Buffer* norm_weight, // RMSNorm weight [K] half
        MTL::Buffer* W,           // Q4_0 weights [N, K/32] blocks
        MTL::Buffer* y,           // Output [N] float
        uint32_t K,               // Input dimension
        uint32_t N,               // Output dimension
        float eps                 // RMSNorm epsilon
    );

    // Fused RMSNorm + IQ4_NL MatVec: y = RMSNorm(x, weight) @ W^T
    Result<void> rms_norm_matvec_iq4_nl(
        MTL::Buffer* x,           // Input [K] float
        MTL::Buffer* norm_weight, // RMSNorm weight [K] half
        MTL::Buffer* W,           // IQ4_NL weights [N, K/32] blocks
        MTL::Buffer* y,           // Output [N] float
        uint32_t K,               // Input dimension
        uint32_t N,               // Output dimension
        float eps                 // RMSNorm epsilon
    );

    // Fused RMSNorm + IQ4_XS MatVec: y = RMSNorm(x, weight) @ W^T
    Result<void> rms_norm_matvec_iq4_xs(
        MTL::Buffer* x,           // Input [K] float
        MTL::Buffer* norm_weight, // RMSNorm weight [K] half
        MTL::Buffer* W,           // IQ4_XS weights [N, K/256] blocks
        MTL::Buffer* y,           // Output [N] float
        uint32_t K,               // Input dimension
        uint32_t N,               // Output dimension
        float eps                 // RMSNorm epsilon
    );

    // Fused RMSNorm + Q6_K MatVec: y = RMSNorm(x, weight) @ W^T
    Result<void> rms_norm_matvec_q6_k(
        MTL::Buffer* x,           // Input [K] float
        MTL::Buffer* norm_weight, // RMSNorm weight [K] half
        MTL::Buffer* W,           // Q6_K weights [N, K/256] blocks
        MTL::Buffer* y,           // Output [N] float
        uint32_t K,               // Input dimension
        uint32_t N,               // Output dimension
        float eps                 // RMSNorm epsilon
    );

    // Fused RMSNorm + Q5_K MatVec: y = RMSNorm(x, weight) @ W^T
    Result<void> rms_norm_matvec_q5_k(
        MTL::Buffer* x,           // Input [K] float
        MTL::Buffer* norm_weight, // RMSNorm weight [K] half
        MTL::Buffer* W,           // Q5_K weights [N, K/256] blocks
        MTL::Buffer* y,           // Output [N] float
        uint32_t K,               // Input dimension
        uint32_t N,               // Output dimension
        float eps                 // RMSNorm epsilon
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

    // Allocate GPU KV cache for a layer (FP16 by default for better bandwidth)
    // Returns pair of (K buffer, V buffer)
    // Shape: [num_kv_heads, max_seq_len, head_dim]
    std::pair<MTL::Buffer*, MTL::Buffer*> create_kv_cache(
        uint32_t num_kv_heads,
        uint32_t max_seq_len,
        uint32_t head_dim
    );

    // Allocate FP32 KV cache (for compatibility)
    std::pair<MTL::Buffer*, MTL::Buffer*> create_kv_cache_f32(
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
    // Embedding Operations
    // =============================================================================

    // Embedding lookup: gather rows from FP16 embedding table
    // token_ids: [num_tokens] int32
    // embeddings: [vocab_size, hidden_dim] half
    // output: [num_tokens, hidden_dim] float
    Result<void> embedding_lookup(
        MTL::Buffer* token_ids,
        MTL::Buffer* embeddings,
        MTL::Buffer* output,
        uint32_t num_tokens,
        uint32_t hidden_dim,
        uint32_t vocab_size
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

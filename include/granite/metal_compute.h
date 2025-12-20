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

    // Profiling API
    void enable_profiling(bool enable);
    void reset_profiling_stats();
    void get_profiling_stats(uint64_t& dispatches, uint64_t& syncs, double& sync_time_ms, uint64_t& cmd_buffers) const;

    // GPU Capture API (for Xcode GPU profiler)
    // Captures a GPU trace that can be opened in Xcode for detailed analysis
    bool begin_capture(const char* capture_path = nullptr);  // nullptr = default path
    void end_capture();

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
    // LLM Operations - IQ3_S Quantized Matrix Multiply (3-bit I-quant with grid)
    // =============================================================================

    // y = x @ W^T where W is IQ3_S quantized
    // IQ3_S uses 256-element super-blocks with 3-bit grid-based non-linear quantization
    Result<void> matvec_iq3_s(
        MTL::Buffer* x,      // Input [K] float
        MTL::Buffer* W,      // Weights [N, K/256] IQ3_S blocks (110 bytes each)
        MTL::Buffer* y,      // Output [N] float
        uint32_t K,          // Input dimension
        uint32_t N           // Output dimension
    );

    // Y = X @ W^T where W is IQ3_S quantized (batched)
    Result<void> matmul_iq3_s(
        MTL::Buffer* X,      // Input [M, K] float
        MTL::Buffer* W,      // Weights [N, K/256] IQ3_S blocks
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
    // LLM Operations - Q3_K Quantized Matrix Multiply
    // =============================================================================

    // y = x @ W^T where W is Q3_K quantized
    Result<void> matvec_q3_k(
        MTL::Buffer* x,      // Input [K] float
        MTL::Buffer* W,      // Weights [N, K/256] Q3_K blocks (110 bytes each)
        MTL::Buffer* y,      // Output [N] float
        uint32_t K,          // Input dimension
        uint32_t N           // Output dimension
    );

    // Y = X @ W^T where W is Q3_K quantized (batched)
    Result<void> matmul_q3_k(
        MTL::Buffer* X,      // Input [M, K] float
        MTL::Buffer* W,      // Weights [N, K/256] Q3_K blocks
        MTL::Buffer* Y,      // Output [M, N] float
        uint32_t M,          // Batch size (number of tokens)
        uint32_t K,          // Input dimension
        uint32_t N           // Output dimension
    );

    // =============================================================================
    // LLM Operations - Q2_K Quantized Matrix Multiply
    // =============================================================================

    // y = x @ W^T where W is Q2_K quantized
    Result<void> matvec_q2_k(
        MTL::Buffer* x,      // Input [K] float
        MTL::Buffer* W,      // Weights [N, K/256] Q2_K blocks (84 bytes each)
        MTL::Buffer* y,      // Output [N] float
        uint32_t K,          // Input dimension
        uint32_t N           // Output dimension
    );

    // Y = X @ W^T where W is Q2_K quantized (batched)
    Result<void> matmul_q2_k(
        MTL::Buffer* X,      // Input [M, K] float
        MTL::Buffer* W,      // Weights [N, K/256] Q2_K blocks
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

    // Y = X @ W^T where W is FP16 (batched)
    Result<void> matmul_f16(
        MTL::Buffer* X,      // Input [M, K] float
        MTL::Buffer* W,      // Weights [N, K] half (row-major, transposed)
        MTL::Buffer* Y,      // Output [M, N] float
        uint32_t M,          // Batch size (number of tokens)
        uint32_t K,          // Input dimension
        uint32_t N           // Output dimension
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

    // Batched RMS Normalization with FP16 weights
    // x: [batch_size, hidden_dim] input
    // out: [batch_size, hidden_dim] output
    // weight: [hidden_dim] broadcast across batch
    Result<void> rms_norm_batch_f16(
        MTL::Buffer* x,
        MTL::Buffer* weight,  // FP16 weights [hidden_dim]
        MTL::Buffer* out,
        uint32_t batch_size,
        uint32_t hidden_dim,
        float eps
    );

    // Batched RMS Normalization with FP32 weights
    // x: [batch_size, hidden_dim] input
    // out: [batch_size, hidden_dim] output
    // weight: [hidden_dim] broadcast across batch
    Result<void> rms_norm_batch(
        MTL::Buffer* x,
        MTL::Buffer* weight,  // FP32 weights [hidden_dim]
        MTL::Buffer* out,
        uint32_t batch_size,
        uint32_t hidden_dim,
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

    // Fused RMSNorm + IQ3_S MatVec: y = RMSNorm(x, weight) @ W^T
    Result<void> rms_norm_matvec_iq3_s(
        MTL::Buffer* x,           // Input [K] float
        MTL::Buffer* norm_weight, // RMSNorm weight [K] half
        MTL::Buffer* W,           // IQ3_S weights [N, K/256] blocks
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

    // Fused RMSNorm + Q3_K MatVec: y = RMSNorm(x, weight) @ W^T
    Result<void> rms_norm_matvec_q3_k(
        MTL::Buffer* x,           // Input [K] float
        MTL::Buffer* norm_weight, // RMSNorm weight [K] half
        MTL::Buffer* W,           // Q3_K weights [N, K/256] blocks
        MTL::Buffer* y,           // Output [N] float
        uint32_t K,               // Input dimension
        uint32_t N,               // Output dimension
        float eps                 // RMSNorm epsilon
    );

    // Fused RMSNorm + Q2_K MatVec: y = RMSNorm(x, weight) @ W^T
    Result<void> rms_norm_matvec_q2_k(
        MTL::Buffer* x,           // Input [K] float
        MTL::Buffer* norm_weight, // RMSNorm weight [K] half
        MTL::Buffer* W,           // Q2_K weights [N, K/256] blocks
        MTL::Buffer* y,           // Output [N] float
        uint32_t K,               // Input dimension
        uint32_t N,               // Output dimension
        float eps                 // RMSNorm epsilon
    );

    // ==========================================================================
    // Phase 2 Fused Kernels - Eliminates redundant computation
    // ==========================================================================

    // Fused RMSNorm + Dual Q4_K MatVec for gate and up projections
    // Computes RMSNorm once and outputs both gate and up projections
    // Eliminates redundant RMSNorm computation in FFN path
    Result<void> rms_norm_dual_matvec_q4k(
        MTL::Buffer* x,           // Input [K] float
        MTL::Buffer* norm_weight, // RMSNorm weight [K] half
        MTL::Buffer* W_gate,      // Gate Q4_K weights [N, K/256] blocks
        MTL::Buffer* W_up,        // Up Q4_K weights [N, K/256] blocks
        MTL::Buffer* y_gate,      // Gate output [N] float
        MTL::Buffer* y_up,        // Up output [N] float
        uint32_t K,               // Input dimension (hidden_dim)
        uint32_t N,               // Output dimension (intermediate_dim)
        float eps                 // RMSNorm epsilon
    );

    // Fused RMSNorm + Dual Q3_K MatVec for gate and up projections
    Result<void> rms_norm_dual_matvec_q3k(
        MTL::Buffer* x,           // Input [K] float
        MTL::Buffer* norm_weight, // RMSNorm weight [K] half
        MTL::Buffer* W_gate,      // Gate Q3_K weights [N, K/256] blocks
        MTL::Buffer* W_up,        // Up Q3_K weights [N, K/256] blocks
        MTL::Buffer* y_gate,      // Gate output [N] float
        MTL::Buffer* y_up,        // Up output [N] float
        uint32_t K,               // Input dimension (hidden_dim)
        uint32_t N,               // Output dimension (intermediate_dim)
        float eps                 // RMSNorm epsilon
    );

    // Fused RMSNorm + Dual Q2_K MatVec for gate and up projections
    Result<void> rms_norm_dual_matvec_q2k(
        MTL::Buffer* x,           // Input [K] float
        MTL::Buffer* norm_weight, // RMSNorm weight [K] half
        MTL::Buffer* W_gate,      // Gate Q2_K weights [N, K/256] blocks
        MTL::Buffer* W_up,        // Up Q2_K weights [N, K/256] blocks
        MTL::Buffer* y_gate,      // Gate output [N] float
        MTL::Buffer* y_up,        // Up output [N] float
        uint32_t K,               // Input dimension (hidden_dim)
        uint32_t N,               // Output dimension (intermediate_dim)
        float eps                 // RMSNorm epsilon
    );

    // Fused MatVec + Residual Add for down projection
    // Combines down projection with residual connection in one kernel
    Result<void> matvec_residual_q4k(
        MTL::Buffer* x,           // Input [K] float (intermediate activations)
        MTL::Buffer* W,           // Q4_K weights [N, K/256] blocks
        MTL::Buffer* residual,    // Residual input [N] float
        MTL::Buffer* y,           // Output [N] float (= residual + x @ W)
        uint32_t K,               // Input dimension (intermediate_dim)
        uint32_t N                // Output dimension (hidden_dim)
    );

    // Fused MatVec + Residual Add for Q3_K
    Result<void> matvec_residual_q3k(
        MTL::Buffer* x,           // Input [K] float (intermediate activations)
        MTL::Buffer* W,           // Q3_K weights [N, K/256] blocks
        MTL::Buffer* residual,    // Residual input [N] float
        MTL::Buffer* y,           // Output [N] float (= residual + x @ W)
        uint32_t K,               // Input dimension (intermediate_dim)
        uint32_t N                // Output dimension (hidden_dim)
    );

    // Fused MatVec + Residual Add for Q2_K
    Result<void> matvec_residual_q2k(
        MTL::Buffer* x,           // Input [K] float (intermediate activations)
        MTL::Buffer* W,           // Q2_K weights [N, K/256] blocks
        MTL::Buffer* residual,    // Residual input [N] float
        MTL::Buffer* y,           // Output [N] float (= residual + x @ W)
        uint32_t K,               // Input dimension (intermediate_dim)
        uint32_t N                // Output dimension (hidden_dim)
    );

    // =============================================================================
    // Fused QKV Attention Projection
    // =============================================================================

    // Fused Q/K/V projection - computes all 3 attention projections in single dispatch
    // Reduces kernel launch overhead from 3 dispatches to 1
    Result<void> fused_qkv_matvec_q4k(
        MTL::Buffer* x,           // Input [K] float (hidden state)
        MTL::Buffer* Wq,          // Q weight [Nq, K/256] Q4_K blocks
        MTL::Buffer* Wk,          // K weight [Nkv, K/256] Q4_K blocks
        MTL::Buffer* Wv,          // V weight [Nkv, K/256] Q4_K blocks
        MTL::Buffer* yq,          // Q output [Nq] float
        MTL::Buffer* yk,          // K output [Nkv] float
        MTL::Buffer* yv,          // V output [Nkv] float
        uint32_t K,               // Input dimension (hidden_dim)
        uint32_t Nq,              // Q output dimension (num_heads * head_dim)
        uint32_t Nkv              // KV output dimension (num_kv_heads * head_dim)
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
    // K: [num_kv_heads, max_seq, head_dim] (max_seq is cache allocation size, seq_kv is actual length)
    // V: [num_kv_heads, max_seq, head_dim]
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
        float scale,
        uint32_t max_seq = 0  // KV cache stride (0 = use seq_kv for backwards compat/decode)
    );

    // =============================================================================
    // Tree Attention (for Speculative Decoding)
    // =============================================================================

    // Tree attention with ancestor-based masking for speculative decoding
    // Each tree node attends to its ancestors (determined by parent_indices) + KV cache
    //
    // Q: [num_heads, num_nodes, head_dim] - Query for each tree node
    // K_cache/V_cache: [num_kv_heads, cache_len, head_dim] - Past context (FP16)
    // K_tree/V_tree: [num_kv_heads, num_nodes, head_dim] - Tree node K/V (FP16)
    // parent_indices: [num_nodes] - Parent index for each node (-1 for root)
    // output: [num_heads, num_nodes, head_dim] - Output for each tree node
    Result<void> attention_tree(
        MTL::Buffer* Q,
        MTL::Buffer* K_cache,
        MTL::Buffer* V_cache,
        MTL::Buffer* K_tree,
        MTL::Buffer* V_tree,
        MTL::Buffer* parent_indices,
        MTL::Buffer* output,
        uint32_t num_heads,
        uint32_t num_kv_heads,
        uint32_t num_nodes,       // Number of tree nodes
        uint32_t cache_len,       // Length of KV cache (0 if no context)
        uint32_t head_dim,
        float scale
    );

    // =============================================================================
    // Paged Attention Operations
    // =============================================================================
    // PagedAttention enables memory-efficient KV cache through block-based paging.
    // K/V are stored in scattered physical blocks, and a block table maps logical
    // sequence positions to physical block locations.
    //
    // Memory layout:
    //   K_cache/V_cache: [num_blocks * block_size, num_kv_heads, head_dim] (half)
    //   block_table: [num_logical_blocks] int32 - maps logical block -> physical block
    //
    // For a token at position `pos`:
    //   logical_block = pos / block_size
    //   block_offset = pos % block_size
    //   physical_block = block_table[logical_block]
    //   physical_pos = physical_block * block_size + block_offset

    // Single-sequence paged attention decode (seq_q = 1)
    // Q: [num_heads, head_dim] float
    // K_cache: [num_blocks * block_size, num_kv_heads, head_dim] half
    // V_cache: [num_blocks * block_size, num_kv_heads, head_dim] half
    // block_table: [num_logical_blocks] int32
    // output: [num_heads, head_dim] float
    Result<void> paged_attention_decode(
        MTL::Buffer* Q,
        MTL::Buffer* K_cache,
        MTL::Buffer* V_cache,
        MTL::Buffer* block_table,
        MTL::Buffer* output,
        uint32_t num_heads,
        uint32_t num_kv_heads,
        uint32_t seq_len,
        uint32_t head_dim,
        uint32_t block_size,
        float scale
    );

    // Append new K/V to paged cache
    // new_k/new_v: [num_kv_heads, new_len, head_dim] float
    // K_cache/V_cache: [num_blocks * block_size, num_kv_heads, head_dim] half
    // block_table: [num_logical_blocks] int32
    Result<void> paged_kv_cache_append(
        MTL::Buffer* new_k,
        MTL::Buffer* new_v,
        MTL::Buffer* K_cache,
        MTL::Buffer* V_cache,
        MTL::Buffer* block_table,
        uint32_t num_kv_heads,
        uint32_t head_dim,
        uint32_t start_pos,
        uint32_t new_len,
        uint32_t block_size
    );

    // Batched paged attention decode - multiple sequences in one kernel launch
    // Q: [batch_size, num_heads, head_dim] float
    // K_cache/V_cache: [num_blocks * block_size, num_kv_heads, head_dim] half - shared pool
    // block_tables: [batch_size, max_blocks_per_seq] int32
    // seq_lens: [batch_size] int32
    // output: [batch_size, num_heads, head_dim] float
    Result<void> batched_paged_attention_decode(
        MTL::Buffer* Q,
        MTL::Buffer* K_cache,
        MTL::Buffer* V_cache,
        MTL::Buffer* block_tables,
        MTL::Buffer* seq_lens,
        MTL::Buffer* output,
        uint32_t batch_size,
        uint32_t num_heads,
        uint32_t num_kv_heads,
        uint32_t head_dim,
        uint32_t block_size,
        uint32_t max_blocks_per_seq,
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

    // =============================================================================
    // Batched Encoding API (for prefill optimization)
    // =============================================================================
    // These methods expose raw Metal encoder access for tight-loop encoding
    // that minimizes CPU overhead. Use begin_batch/end_batch to wrap encoding.

    // Begin batched encoding - returns raw encoder for direct use
    // The encoder is valid until end_batch() is called
    void* begin_batch();

    // End batched encoding and sync
    void end_batch();

    // Get pre-cached pipeline state by name (for tight-loop encoding)
    // Returns MTL::ComputePipelineState* or nullptr if not found
    void* get_pipeline(const char* name);

private:
    class Impl;
    Impl* impl_ = nullptr;
};

}  // namespace granite

#endif  // GRANITE_HAS_METAL

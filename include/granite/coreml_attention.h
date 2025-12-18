#pragma once

#include "granite/error.h"
#include "granite/config.h"

#if defined(__APPLE__)

// Forward declarations for Metal types (must be outside granite namespace)
namespace MTL {
    class Device;
    class Buffer;
}

namespace granite {

/// CoreML/MPSGraph-based attention for Apple Neural Engine acceleration
///
/// This backend uses MPSGraph which can automatically route operations to ANE
/// for improved power efficiency on iOS devices.
///
/// Best for:
/// - iOS devices with ANE (A11+)
/// - Battery-conscious inference
/// - Prefill phase with static shapes
///
/// Limitations:
/// - ANE prefers static shapes (use Metal for variable-length decode)
/// - Higher latency than Metal for small batches
/// - Requires FP16 for best ANE utilization
class CoreMLAttention {
public:
    CoreMLAttention();
    ~CoreMLAttention();

    /// Initialize with Metal device for buffer sharing
    Result<void> initialize(MTL::Device* device);

    /// Shutdown and release resources
    void shutdown();

    /// Check if initialized
    bool is_initialized() const;

    /// Check if ANE is available on this device
    static bool is_ane_available();

    /// Multi-head attention using MPSGraph
    /// Q: [num_heads, seq_q, head_dim] float
    /// K: [num_kv_heads, seq_kv, head_dim] half (FP16)
    /// V: [num_kv_heads, seq_kv, head_dim] half (FP16)
    /// output: [num_heads, seq_q, head_dim] float
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

    /// Single-head attention (more efficient for decode with seq_q=1)
    Result<void> attention_single_head(
        MTL::Buffer* Q,           // [seq_q, head_dim] float
        MTL::Buffer* K,           // [seq_kv, head_dim] half
        MTL::Buffer* V,           // [seq_kv, head_dim] half
        MTL::Buffer* output,      // [seq_q, head_dim] float
        uint32_t seq_q,
        uint32_t seq_kv,
        uint32_t head_dim,
        float scale
    );

    /// Get compute unit preference
    enum class ComputeUnit {
        Auto,           // Let system decide (default)
        ANE,            // Force Neural Engine
        GPU,            // Force GPU
        CPU             // Force CPU
    };

    void set_compute_unit(ComputeUnit unit);
    ComputeUnit get_compute_unit() const;

private:
    class Impl;
    Impl* impl_ = nullptr;
};

/// Get global CoreMLAttention instance (lazy initialized)
CoreMLAttention* get_coreml_attention();

}  // namespace granite

#endif  // __APPLE__

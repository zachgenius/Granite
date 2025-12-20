#pragma once

#include "granite/error.h"

#if defined(__APPLE__)

namespace MTL {
    class Device;
    class Buffer;
}

namespace granite {

/// CoreML/MPSGraph-based FFN for Apple Neural Engine acceleration
///
/// FFN is well-suited for ANE because:
/// 1. Fixed shapes (hidden_dim × intermediate_dim don't change)
/// 2. Large matrix multiplies benefit from ANE's compute
/// 3. Pre-compilation amortizes MPSGraph overhead
///
/// NOTE: Requires FP16 weights. Quantized models (Q4_K, etc.) must
/// dequantize weights at load time to use ANE FFN offload.
///
/// Performance characteristics:
/// - ~10ms fixed overhead per MPSGraph execution
/// - Best for power-efficient inference on iOS (10x better power/FLOP than GPU)
/// - Not faster than Metal for latency-critical decode
/// - Most beneficial when batching multiple tokens (prefill)
class CoreMLFFN {
public:
    CoreMLFFN();
    ~CoreMLFFN();

    /// Initialize with Metal device for buffer sharing
    Result<void> initialize(MTL::Device* device);

    /// Shutdown and release resources
    void shutdown();

    /// Check if initialized
    bool is_initialized() const;

    /// Check if ANE is available on this device
    static bool is_ane_available();

    /// Pre-compile FFN graph for specific dimensions
    /// Call this at model load time for each layer's dimensions
    Result<void> compile_ffn(
        uint32_t hidden_dim,
        uint32_t intermediate_dim
    );

    /// Execute SwiGLU FFN: output = down(silu(gate(x)) * up(x))
    ///
    /// All weights must be FP16, input/output are FP32
    ///
    /// @param input       [batch, hidden_dim] FP32
    /// @param w_gate      [hidden_dim, intermediate_dim] FP16
    /// @param w_up        [hidden_dim, intermediate_dim] FP16
    /// @param w_down      [intermediate_dim, hidden_dim] FP16
    /// @param output      [batch, hidden_dim] FP32
    /// @param batch_size  Number of tokens (typically 1 for decode)
    /// @param hidden_dim  Model hidden dimension
    /// @param intermediate_dim  FFN intermediate dimension
    Result<void> forward(
        MTL::Buffer* input,
        MTL::Buffer* w_gate,
        MTL::Buffer* w_up,
        MTL::Buffer* w_down,
        MTL::Buffer* output,
        uint32_t batch_size,
        uint32_t hidden_dim,
        uint32_t intermediate_dim
    );

    /// Power mode selection
    enum class PowerMode {
        Performance,    // Use Metal GPU (faster, more power)
        Balanced,       // Auto-select based on batch size
        LowPower        // Use ANE when possible (slower, less power)
    };

    void set_power_mode(PowerMode mode);
    PowerMode get_power_mode() const;

private:
    class Impl;
    Impl* impl_ = nullptr;
};

/// Get global CoreMLFFN instance (lazy initialized)
CoreMLFFN* get_coreml_ffn();

}  // namespace granite

#endif  // __APPLE__

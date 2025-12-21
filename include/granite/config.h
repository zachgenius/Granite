#pragma once

#include <cstdint>
#include <string>

namespace granite {

// =============================================================================
// Configuration Enums
// =============================================================================

/// Attention backend selection
enum class AttentionBackend {
    Auto,           ///< Platform-optimal selection (default)
    MetalFlash,     ///< Custom Metal flash attention kernels
    MetalLegacy,    ///< Original Metal attention (compatibility)
    CoreML,         ///< CoreML/ANE (compiles attention subgraph)
    MPS,            ///< Metal Performance Shaders
    CPU,            ///< CPU reference implementation
};

/// Thermal management mode
enum class ThermalMode {
    Unrestricted,   ///< Ignore thermal state (desktop/plugged in)
    Adaptive,       ///< Adjust based on thermal pressure (default)
    Conservative,   ///< Prioritize device temperature
};

/// Power management mode
enum class PowerMode {
    Performance,    ///< Max speed, ignore battery
    Balanced,       ///< Balance speed and power (default)
    LowPower,       ///< Minimize power consumption
};

/// Weight cache policy
enum class CachePolicy {
    Eager,          ///< Load all weights upfront
    Lazy,           ///< Load weights on first use (default)
    Streaming,      ///< Stream from disk (memory constrained)
};

// =============================================================================
// Platform Detection
// =============================================================================

namespace platform {

enum class Platform {
    macOS,
    iOS,
    tvOS,
    Simulator,
    Unknown
};

enum class ChipFamily {
    Unknown,
    // A-series (iPhone/iPad)
    A14, A15, A16, A17, A18,
    // M-series (Mac)
    M1, M2, M3, M4
};

/// Device information for runtime decisions
struct DeviceInfo {
    Platform platform = Platform::Unknown;
    ChipFamily chip_family = ChipFamily::Unknown;
    uint64_t total_memory_bytes = 0;
    uint64_t available_memory_bytes = 0;
    bool has_neural_engine = false;
    bool has_metal3 = false;
    float thermal_state = 0.0f;   ///< 0.0 = cool, 1.0 = critical
    bool is_plugged_in = true;
    bool is_low_power_mode = false;
};

/// Get current device information
DeviceInfo get_device_info();

} // namespace platform

// ==============================================================================
// Configuration Structure
// =============================================================================

/// Main configuration for Granite inference
struct Config {
    // === Attention Configuration ===
    AttentionBackend attention_backend = AttentionBackend::Auto;
    bool allow_attention_fallback = true;  ///< Fall back if preferred unavailable
    uint32_t prefill_chunk_size = 0;       ///< 0 = no chunking (memory-efficient prefill)

    // === Memory Configuration ===
    size_t max_memory_mb = 0;              ///< 0 = auto (based on device)
    size_t kv_cache_max_seq = 2048;        ///< Maximum sequence length for KV cache
    size_t kv_cache_gpu_max_seq = 0;       ///< 0 = use kv_cache_max_seq
    bool kv_cache_offload = false;         ///< Allow CPU KV fallback when GPU cache is smaller
    CachePolicy weight_cache_policy = CachePolicy::Lazy;
    bool use_memory_mapping = true;        ///< mmap for weights

    // === Thermal Configuration ===
    ThermalMode thermal_mode = ThermalMode::Adaptive;
    float thermal_throttle_threshold = 0.8f;  ///< Throttle at 80% thermal headroom
    uint32_t thermal_check_interval_ms = 1000;

    // === Power Configuration ===
    PowerMode power_mode = PowerMode::Balanced;
    bool use_neural_engine = true;         ///< Use ANE when available
    bool prefer_efficiency_cores = false;  ///< iOS: use E-cores

    // === Threading Configuration ===
    uint32_t compute_threads = 0;          ///< 0 = auto
    uint32_t yield_interval_ms = 0;        ///< 0 = no yielding

    // === Generation Configuration ===
    uint32_t max_batch_size = 1;           ///< For continuous batching
    bool enable_speculative_decoding = false;

    // === Debug Configuration ===
    bool enable_profiling = false;
    bool log_kernel_selection = false;

    // =========================================================================
    // Preset Factories
    // =========================================================================

    /// Maximum performance, for desktop/plugged-in use
    static Config Performance() {
        Config c;
        c.attention_backend = AttentionBackend::MetalFlash;
        c.power_mode = PowerMode::Performance;
        c.thermal_mode = ThermalMode::Unrestricted;
        c.max_memory_mb = 0;  // Unlimited
        c.kv_cache_max_seq = 4096;
        c.use_neural_engine = false;  // Prefer GPU
        return c;
    }

    /// Balanced performance and power (default)
    static Config Balanced() {
        Config c;
        c.attention_backend = AttentionBackend::Auto;
        c.power_mode = PowerMode::Balanced;
        c.thermal_mode = ThermalMode::Adaptive;
        c.max_memory_mb = 1024;  // 1GB limit
        c.kv_cache_max_seq = 2048;
        return c;
    }

    /// Battery-conscious for mobile devices
    static Config Battery() {
        Config c;
        c.attention_backend = AttentionBackend::CoreML;  // Use ANE
        c.power_mode = PowerMode::LowPower;
        c.thermal_mode = ThermalMode::Conservative;
        c.max_memory_mb = 512;
        c.kv_cache_max_seq = 1024;
        c.use_neural_engine = true;
        c.prefer_efficiency_cores = true;
        c.yield_interval_ms = 10;  // Yield for UI responsiveness
        return c;
    }

    /// Memory-constrained devices
    static Config LowMemory() {
        Config c;
        c.attention_backend = AttentionBackend::CoreML;
        c.power_mode = PowerMode::LowPower;
        c.thermal_mode = ThermalMode::Conservative;
        c.max_memory_mb = 256;
        c.kv_cache_max_seq = 512;
        c.weight_cache_policy = CachePolicy::Streaming;
        c.use_neural_engine = true;
        return c;
    }

    /// Background processing (iOS)
    static Config Background() {
        Config c;
        c.attention_backend = AttentionBackend::CoreML;
        c.power_mode = PowerMode::LowPower;
        c.thermal_mode = ThermalMode::Conservative;
        c.max_memory_mb = 256;
        c.kv_cache_max_seq = 512;
        c.use_neural_engine = true;
        c.prefer_efficiency_cores = true;
        c.yield_interval_ms = 50;  // Very responsive yielding
        return c;
    }
};

// =============================================================================
// Config Builder (Fluent API)
// =============================================================================

/// Builder for creating custom configurations
class ConfigBuilder {
public:
    ConfigBuilder() = default;
    explicit ConfigBuilder(const Config& base) : config_(base) {}

    // Attention
    ConfigBuilder& attention_backend(AttentionBackend backend) {
        config_.attention_backend = backend;
        return *this;
    }

    ConfigBuilder& allow_attention_fallback(bool allow) {
        config_.allow_attention_fallback = allow;
        return *this;
    }

    // Memory
    ConfigBuilder& max_memory_mb(size_t mb) {
        config_.max_memory_mb = mb;
        return *this;
    }

    ConfigBuilder& kv_cache_max_seq(size_t seq) {
        config_.kv_cache_max_seq = seq;
        return *this;
    }

    ConfigBuilder& kv_cache_gpu_max_seq(size_t seq) {
        config_.kv_cache_gpu_max_seq = seq;
        return *this;
    }

    ConfigBuilder& kv_cache_offload(bool enable) {
        config_.kv_cache_offload = enable;
        return *this;
    }

    ConfigBuilder& weight_cache_policy(CachePolicy policy) {
        config_.weight_cache_policy = policy;
        return *this;
    }

    ConfigBuilder& use_memory_mapping(bool use) {
        config_.use_memory_mapping = use;
        return *this;
    }

    // Thermal
    ConfigBuilder& thermal_mode(ThermalMode mode) {
        config_.thermal_mode = mode;
        return *this;
    }

    ConfigBuilder& thermal_throttle_threshold(float threshold) {
        config_.thermal_throttle_threshold = threshold;
        return *this;
    }

    // Power
    ConfigBuilder& power_mode(PowerMode mode) {
        config_.power_mode = mode;
        return *this;
    }

    ConfigBuilder& use_neural_engine(bool use) {
        config_.use_neural_engine = use;
        return *this;
    }

    ConfigBuilder& prefer_efficiency_cores(bool prefer) {
        config_.prefer_efficiency_cores = prefer;
        return *this;
    }

    // Threading
    ConfigBuilder& compute_threads(uint32_t threads) {
        config_.compute_threads = threads;
        return *this;
    }

    ConfigBuilder& yield_interval_ms(uint32_t ms) {
        config_.yield_interval_ms = ms;
        return *this;
    }

    // Generation
    ConfigBuilder& max_batch_size(uint32_t size) {
        config_.max_batch_size = size;
        return *this;
    }

    ConfigBuilder& enable_speculative_decoding(bool enable) {
        config_.enable_speculative_decoding = enable;
        return *this;
    }

    // Debug
    ConfigBuilder& enable_profiling(bool enable) {
        config_.enable_profiling = enable;
        return *this;
    }

    ConfigBuilder& log_kernel_selection(bool log) {
        config_.log_kernel_selection = log;
        return *this;
    }

    /// Build the final configuration
    Config build() const {
        return config_;
    }

private:
    Config config_;
};

// =============================================================================
// String Conversion Utilities
// =============================================================================

inline const char* to_string(AttentionBackend backend) {
    switch (backend) {
        case AttentionBackend::Auto: return "Auto";
        case AttentionBackend::MetalFlash: return "MetalFlash";
        case AttentionBackend::MetalLegacy: return "MetalLegacy";
        case AttentionBackend::CoreML: return "CoreML";
        case AttentionBackend::MPS: return "MPS";
        case AttentionBackend::CPU: return "CPU";
    }
    return "Unknown";
}

inline const char* to_string(ThermalMode mode) {
    switch (mode) {
        case ThermalMode::Unrestricted: return "Unrestricted";
        case ThermalMode::Adaptive: return "Adaptive";
        case ThermalMode::Conservative: return "Conservative";
    }
    return "Unknown";
}

inline const char* to_string(PowerMode mode) {
    switch (mode) {
        case PowerMode::Performance: return "Performance";
        case PowerMode::Balanced: return "Balanced";
        case PowerMode::LowPower: return "LowPower";
    }
    return "Unknown";
}

inline const char* to_string(CachePolicy policy) {
    switch (policy) {
        case CachePolicy::Eager: return "Eager";
        case CachePolicy::Lazy: return "Lazy";
        case CachePolicy::Streaming: return "Streaming";
    }
    return "Unknown";
}

// =============================================================================
// Backend Selection (now that Config is fully defined)
// =============================================================================

namespace platform {

/// Auto-select best attention backend for current device + config
AttentionBackend select_attention_backend(const Config& config, const DeviceInfo& device);

} // namespace platform

} // namespace granite

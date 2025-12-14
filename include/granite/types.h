#pragma once

#include <cstdint>
#include <cstddef>
#include <functional>

namespace granite {

// =============================================================================
// Data Types
// =============================================================================

enum class DataType : uint8_t {
    FP32,
    FP16,
    BF16,
    INT32,
    INT16,
    INT8,
    UINT8,
    INT4,    // Packed 4-bit integers (2 per byte)
    UINT4,
};

/// Returns the size in bytes of a single element of the given type
constexpr size_t dtype_size(DataType dtype) {
    switch (dtype) {
        case DataType::FP32:  return 4;
        case DataType::FP16:  return 2;
        case DataType::BF16:  return 2;
        case DataType::INT32: return 4;
        case DataType::INT16: return 2;
        case DataType::INT8:  return 1;
        case DataType::UINT8: return 1;
        case DataType::INT4:  return 1;  // 2 elements per byte
        case DataType::UINT4: return 1;  // 2 elements per byte
    }
    return 0;
}

/// Returns true if the type is a floating point type
constexpr bool dtype_is_float(DataType dtype) {
    return dtype == DataType::FP32 ||
           dtype == DataType::FP16 ||
           dtype == DataType::BF16;
}

/// Returns true if the type is a quantized type
constexpr bool dtype_is_quantized(DataType dtype) {
    return dtype == DataType::INT4 ||
           dtype == DataType::UINT4 ||
           dtype == DataType::INT8;
}

/// Returns a string representation of the data type
constexpr const char* dtype_name(DataType dtype) {
    switch (dtype) {
        case DataType::FP32:  return "fp32";
        case DataType::FP16:  return "fp16";
        case DataType::BF16:  return "bf16";
        case DataType::INT32: return "int32";
        case DataType::INT16: return "int16";
        case DataType::INT8:  return "int8";
        case DataType::UINT8: return "uint8";
        case DataType::INT4:  return "int4";
        case DataType::UINT4: return "uint4";
    }
    return "unknown";
}

// =============================================================================
// Memory Layout
// =============================================================================

enum class MemoryLayout : uint8_t {
    RowMajor,   // C-style: last dimension contiguous
    ColMajor,   // Fortran-style: first dimension contiguous
    NCHW,       // Batch, Channel, Height, Width
    NHWC,       // Batch, Height, Width, Channel (often faster on GPU)
    Custom,     // Arbitrary strides
};

// =============================================================================
// Backend Types
// =============================================================================

enum class BackendType : uint8_t {
    CPU,
    Metal,
    Vulkan,
};

constexpr const char* backend_name(BackendType type) {
    switch (type) {
        case BackendType::CPU:    return "CPU";
        case BackendType::Metal:  return "Metal";
        case BackendType::Vulkan: return "Vulkan";
    }
    return "unknown";
}

// =============================================================================
// Handle Types
// =============================================================================

// Opaque handles for backend resources
struct BufferHandle {
    uint64_t id = 0;

    bool valid() const { return id != 0; }
    bool operator==(const BufferHandle& other) const { return id == other.id; }
    bool operator!=(const BufferHandle& other) const { return id != other.id; }
};

struct PipelineHandle {
    uint64_t id = 0;

    bool valid() const { return id != 0; }
    bool operator==(const PipelineHandle& other) const { return id == other.id; }
    bool operator!=(const PipelineHandle& other) const { return id != other.id; }
};

struct FenceHandle {
    uint64_t id = 0;

    bool valid() const { return id != 0; }
    bool operator==(const FenceHandle& other) const { return id == other.id; }
    bool operator!=(const FenceHandle& other) const { return id != other.id; }
};

// =============================================================================
// Memory Types
// =============================================================================

enum class MemoryType : uint8_t {
    Device,      // GPU-only memory (fastest for compute)
    Shared,      // CPU/GPU shared (for uploads/downloads)
    Managed,     // Automatic migration (Metal unified memory)
};

// =============================================================================
// Buffer Description
// =============================================================================

struct BufferDesc {
    size_t size = 0;
    MemoryType memory_type = MemoryType::Device;
    bool allow_aliasing = false;  // Can reuse memory from other buffers

    static BufferDesc device(size_t size) {
        return {size, MemoryType::Device, false};
    }

    static BufferDesc shared(size_t size) {
        return {size, MemoryType::Shared, false};
    }

    static BufferDesc managed(size_t size) {
        return {size, MemoryType::Managed, false};
    }
};

// =============================================================================
// Device Capabilities
// =============================================================================

struct DeviceCapabilities {
    const char* name = nullptr;
    size_t max_buffer_size = 0;
    size_t max_threadgroup_size = 0;
    size_t shared_memory_size = 0;
    uint32_t simd_width = 0;

    bool supports_fp16 = false;
    bool supports_bf16 = false;
    bool supports_int8 = false;
    bool supports_int4 = false;
    bool supports_simd_groups = false;  // Metal: simdgroups, Vulkan: subgroups
};

}  // namespace granite

// Hash functions for handle types
namespace std {

template<>
struct hash<granite::BufferHandle> {
    size_t operator()(const granite::BufferHandle& h) const {
        return std::hash<uint64_t>{}(h.id);
    }
};

template<>
struct hash<granite::PipelineHandle> {
    size_t operator()(const granite::PipelineHandle& h) const {
        return std::hash<uint64_t>{}(h.id);
    }
};

template<>
struct hash<granite::FenceHandle> {
    size_t operator()(const granite::FenceHandle& h) const {
        return std::hash<uint64_t>{}(h.id);
    }
};

}  // namespace std

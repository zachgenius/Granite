#pragma once

#include "granite/types.h"
#include "granite/error.h"
#include "granite/graph.h"

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <variant>
#include <optional>

namespace granite {

// =============================================================================
// GGUF Constants
// =============================================================================

constexpr uint32_t GGUF_MAGIC = 0x46554747;  // "GGUF" in little-endian
constexpr uint32_t GGUF_VERSION_MIN = 2;
constexpr uint32_t GGUF_VERSION_MAX = 3;

// =============================================================================
// GGUF Data Types
// =============================================================================

enum class GGUFType : uint32_t {
    UINT8   = 0,
    INT8    = 1,
    UINT16  = 2,
    INT16   = 3,
    UINT32  = 4,
    INT32   = 5,
    FLOAT32 = 6,
    BOOL    = 7,
    STRING  = 8,
    ARRAY   = 9,
    UINT64  = 10,
    INT64   = 11,
    FLOAT64 = 12,
};

// GGML tensor types (quantization formats)
enum class GGMLType : uint32_t {
    F32     = 0,
    F16     = 1,
    Q4_0    = 2,
    Q4_1    = 3,
    Q5_0    = 6,
    Q5_1    = 7,
    Q8_0    = 8,
    Q8_1    = 9,
    Q2_K    = 10,
    Q3_K    = 11,
    Q4_K    = 12,
    Q5_K    = 13,
    Q6_K    = 14,
    Q8_K    = 15,
    IQ2_XXS = 16,
    IQ2_XS  = 17,
    IQ3_XXS = 18,
    IQ1_S   = 19,
    IQ4_NL  = 20,
    IQ3_S   = 21,
    IQ2_S   = 22,
    IQ4_XS  = 23,
    I8      = 24,
    I16     = 25,
    I32     = 26,
    I64     = 27,
    F64     = 28,
    BF16    = 30,
    COUNT,
};

/// Get the name of a GGML type
const char* ggml_type_name(GGMLType type);

/// Get bytes per element for a GGML type (for non-quantized types)
/// For quantized types, use ggml_type_block_size and ggml_type_bytes_per_block
size_t ggml_type_size(GGMLType type);

/// Get block size for quantized types (number of elements per block)
size_t ggml_type_block_size(GGMLType type);

/// Get bytes per block for quantized types
size_t ggml_type_bytes_per_block(GGMLType type);

/// Check if type is quantized
bool ggml_type_is_quantized(GGMLType type);

/// Convert GGML type to Granite DataType (for supported types)
std::optional<DataType> ggml_type_to_dtype(GGMLType type);

// =============================================================================
// GGUF Metadata Value
// =============================================================================

using GGUFValue = std::variant<
    uint8_t,
    int8_t,
    uint16_t,
    int16_t,
    uint32_t,
    int32_t,
    float,
    bool,
    std::string,
    uint64_t,
    int64_t,
    double,
    std::vector<uint8_t>,    // Array of uint8
    std::vector<int32_t>,    // Array of int32
    std::vector<float>,      // Array of float
    std::vector<std::string> // Array of string
>;

// =============================================================================
// GGUF Tensor Info
// =============================================================================

struct GGUFTensorInfo {
    std::string name;
    uint32_t n_dims = 0;
    std::vector<uint64_t> dimensions;
    GGMLType type = GGMLType::F32;
    uint64_t offset = 0;  // Offset into data section

    // Computed properties
    [[nodiscard]] size_t numel() const;
    [[nodiscard]] size_t size_bytes() const;
};

// =============================================================================
// GGUF File Header
// =============================================================================

struct GGUFHeader {
    uint32_t magic = 0;
    uint32_t version = 0;
    uint64_t tensor_count = 0;
    uint64_t metadata_kv_count = 0;
};

// =============================================================================
// GGUF File
// =============================================================================

class GGUFFile {
public:
    GGUFFile() = default;
    ~GGUFFile();

    // Disable copy (owns file handle)
    GGUFFile(const GGUFFile&) = delete;
    GGUFFile& operator=(const GGUFFile&) = delete;

    // Allow move
    GGUFFile(GGUFFile&& other) noexcept;
    GGUFFile& operator=(GGUFFile&& other) noexcept;

    /// Open and parse a GGUF file
    [[nodiscard]] static Result<GGUFFile> open(const std::string& path);

    /// Check if file is open
    [[nodiscard]] bool is_open() const { return data_ != nullptr; }

    /// Get file path
    [[nodiscard]] const std::string& path() const { return path_; }

    // -------------------------------------------------------------------------
    // Header & Metadata
    // -------------------------------------------------------------------------

    [[nodiscard]] const GGUFHeader& header() const { return header_; }
    [[nodiscard]] uint32_t version() const { return header_.version; }
    [[nodiscard]] uint64_t tensor_count() const { return header_.tensor_count; }

    /// Get all metadata keys
    [[nodiscard]] std::vector<std::string> metadata_keys() const;

    /// Check if metadata key exists
    [[nodiscard]] bool has_metadata(const std::string& key) const;

    /// Get metadata value (returns nullopt if not found)
    [[nodiscard]] std::optional<GGUFValue> get_metadata(const std::string& key) const;

    /// Get metadata as specific type
    template<typename T>
    [[nodiscard]] std::optional<T> get_metadata_as(const std::string& key) const {
        auto val = get_metadata(key);
        if (!val) return std::nullopt;
        if (auto* ptr = std::get_if<T>(&*val)) {
            return *ptr;
        }
        return std::nullopt;
    }

    // Convenience accessors for common metadata
    [[nodiscard]] std::optional<std::string> get_architecture() const;
    [[nodiscard]] std::optional<uint32_t> get_context_length() const;
    [[nodiscard]] std::optional<uint32_t> get_embedding_length() const;
    [[nodiscard]] std::optional<uint32_t> get_block_count() const;
    [[nodiscard]] std::optional<uint32_t> get_head_count() const;
    [[nodiscard]] std::optional<uint32_t> get_head_count_kv() const;

    // -------------------------------------------------------------------------
    // Tensors
    // -------------------------------------------------------------------------

    /// Get all tensor infos
    [[nodiscard]] const std::vector<GGUFTensorInfo>& tensors() const { return tensors_; }

    /// Find tensor by name
    [[nodiscard]] const GGUFTensorInfo* find_tensor(const std::string& name) const;

    /// Get raw pointer to tensor data (memory-mapped)
    [[nodiscard]] const void* tensor_data(const GGUFTensorInfo& info) const;

    /// Get tensor data with type checking
    template<typename T>
    [[nodiscard]] const T* tensor_data_as(const GGUFTensorInfo& info) const {
        return static_cast<const T*>(tensor_data(info));
    }

    // -------------------------------------------------------------------------
    // Utilities
    // -------------------------------------------------------------------------

    /// Get total file size
    [[nodiscard]] size_t file_size() const { return file_size_; }

    /// Get data section offset
    [[nodiscard]] size_t data_offset() const { return data_offset_; }

    /// Get summary string for debugging
    [[nodiscard]] std::string summary() const;

private:
    std::string path_;
    GGUFHeader header_;
    std::unordered_map<std::string, GGUFValue> metadata_;
    std::vector<GGUFTensorInfo> tensors_;

    // Memory-mapped file data
    void* data_ = nullptr;
    size_t file_size_ = 0;
    size_t data_offset_ = 0;  // Where tensor data starts
    int fd_ = -1;

    // Parsing helpers
    Result<void> parse_header();
    Result<void> parse_metadata();
    Result<void> parse_tensors();
    Result<GGUFValue> read_value(GGUFType type, size_t& offset);
    std::string read_string(size_t& offset);
};

// =============================================================================
// Model Loader
// =============================================================================

struct ModelInfo {
    std::string architecture;
    std::string name;

    uint32_t vocab_size = 0;
    uint32_t context_length = 0;
    uint32_t embedding_length = 0;
    uint32_t block_count = 0;
    uint32_t head_count = 0;
    uint32_t head_count_kv = 0;
    uint32_t feed_forward_length = 0;

    float rope_freq_base = 10000.0f;
    float rope_freq_scale = 1.0f;

    GGMLType weight_type = GGMLType::F16;
};

class ModelLoader {
public:
    explicit ModelLoader(IComputeBackend* backend, bool use_memory_mapping = true);

    /// Load model info from GGUF file
    [[nodiscard]] Result<ModelInfo> load_info(const std::string& path);

    /// Load model weights into tensors
    /// Returns a map of tensor name -> Tensor
    [[nodiscard]] Result<std::unordered_map<std::string, Tensor>> load_weights(
        const GGUFFile& file);

    /// Load specific tensor by name
    [[nodiscard]] Result<Tensor> load_tensor(
        const GGUFFile& file,
        const std::string& name);

private:
    IComputeBackend* backend_;
    bool use_memory_mapping_ = true;

    // Dequantize weights to FP16/FP32 if needed
    Result<Tensor> dequantize_tensor(
        const GGUFTensorInfo& info,
        const void* data);
};

}  // namespace granite
